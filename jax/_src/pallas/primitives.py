# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas-specific JAX primitives."""

from __future__ import annotations

import enum
import functools
import string
from typing import Any

import jax
from jax import lax
from jax import tree_util
from jax._src import ad_util
from jax._src import core as jax_core
from jax._src import effects
from jax._src import pretty_printer as pp
from jax._src import state
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.pallas import core as pallas_core
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as sp
from jax.interpreters import mlir
import jax.numpy as jnp

partial = functools.partial
Slice = indexing.Slice
NDIndexer = indexing.NDIndexer

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

program_id_p = jax_core.Primitive("program_id")

def program_id(axis: int) -> jax.Array:
  """Returns the kernel execution position along the given axis of the grid."""
  return program_id_p.bind(axis=axis)

def program_id_bind(*, axis: int):
  grid_env = pallas_core.current_grid_env()
  if grid_env:
    return grid_env[axis].index
  frame = pallas_core.axis_frame()
  # Query the size of the axis to make sure its a valid axis (and error
  # otherwise).
  _ = frame.size(axis)
  return jax_core.Primitive.bind(program_id_p, axis=axis)
program_id_p.def_custom_bind(program_id_bind)

def _program_id_abstract_eval(**_):
  return jax_core.ShapedArray((), jnp.int32)
program_id_p.def_abstract_eval(_program_id_abstract_eval)

num_programs_p = jax_core.Primitive("num_programs")

def num_programs(axis: int) -> int | jax.Array:
  """Returns the size of the grid along the given axis."""
  return num_programs_p.bind(axis=axis)

@num_programs_p.def_custom_bind
def _num_programs_bind(*, axis: int):
  # We might be using a local grid env
  grid_env = pallas_core.current_grid_env()
  if grid_env:
    return grid_env[axis].size
  # Otherwise, we look up the size of the grid in the axis env
  frame = pallas_core.axis_frame()
  size = frame.size(axis)
  if size is pallas_core.dynamic_grid_dim:
    return jax_core.Primitive.bind(num_programs_p, axis=axis)
  return size

@num_programs_p.def_abstract_eval
def _num_programs_abstract_eval(**_):
  return jax_core.ShapedArray((), jnp.int32)

class AtomicOpType(enum.Enum):
  XCHG = "xchg"
  ADD = "add"
  MAX = "max"
  MIN = "min"
  AND = "and"
  OR = "or"
  XOR = "xor"

atomic_rmw_p = jax_core.Primitive("atomic_rmw")


def _atomic_rmw_discharge_rule(
    in_avals, out_avals, *args_flat, args_tree, atomic_type: AtomicOpType
):
  del out_avals  # Unused.
  ref, indexers, val, mask = args_tree.unflatten(args_flat)
  if len(indexers) > 1:
    raise NotImplementedError("Only one indexer is supported.")
  idx = indexers[0]

  if mask is not None:
    raise NotImplementedError

  if atomic_type == AtomicOpType.ADD:
    monoid = lambda x, y: x + y
  elif atomic_type == AtomicOpType.MAX:
    monoid = jnp.maximum
  elif atomic_type == AtomicOpType.MIN:
    monoid = jnp.minimum
  else:
    raise NotImplementedError(atomic_type)

  if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
    indices = idx.indices
    scalar_dims = [not isinstance(s, Slice) and s.shape == () for s in indices]
    slice_starts = [s.start if isinstance(s, Slice) else s for s in indices]
    slice_sizes = tuple(s.size if isinstance(s, Slice) else 1 for s in indices)
    out_ones = lax.dynamic_slice(ref, slice_starts, slice_sizes=slice_sizes)
    val_indexer = tuple(None if scalar else slice(None) for scalar in scalar_dims)
    val = val[val_indexer]
    val = monoid(val, out_ones)
    x_new = lax.dynamic_update_slice(ref, val, start_indices=slice_starts)
    out_indexer = tuple(0 if scalar else slice(None) for scalar in scalar_dims)
    out = out_ones[out_indexer]
  elif all(not isinstance(s, Slice) for s in idx.indices):
    out = ref[idx.indices]
    x_new = ref.at[idx.indices].set(monoid(out, val))
  else:
    raise NotImplementedError
  return (x_new,) + (None,) * (len(in_avals) - 1), out


state_discharge.register_discharge_rule(atomic_rmw_p)(_atomic_rmw_discharge_rule)


def _atomic_abstract_eval(*avals_flat, args_tree, atomic_type: AtomicOpType):
  ref, _, _, _ = args_tree.unflatten(avals_flat)
  if ref.dtype == jnp.dtype("float16") and atomic_type != AtomicOpType.ADD:
    raise ValueError(f"`atomic_{atomic_type.value}` does not support f16.")
  if ref.dtype in {
      jnp.dtype("bool"),
      jnp.dtype("int8"),
      jnp.dtype("int16"),
      jnp.bfloat16,
  }:
    raise ValueError(
        f"`atomic_{atomic_type.value}` does not support {ref.dtype}."
    )
  return _swap_abstract_eval(*avals_flat, args_tree=args_tree)


atomic_rmw_p.def_effectful_abstract_eval(_atomic_abstract_eval)

def atomic_rmw(x_ref_or_view, idx, val, *, mask: Any | None = None,
               atomic_type: AtomicOpType):
  x_ref, indexers = sp.get_ref_and_indexers(x_ref_or_view, idx, "atomic_rmw")
  args_flat, args_tree = tree_util.tree_flatten((x_ref, indexers, val, mask))
  return atomic_rmw_p.bind(
      *args_flat, args_tree=args_tree, atomic_type=atomic_type
  )

atomic_xchg = functools.partial(atomic_rmw, atomic_type=AtomicOpType.XCHG)
atomic_add = functools.partial(atomic_rmw, atomic_type=AtomicOpType.ADD)
atomic_max = functools.partial(atomic_rmw, atomic_type=AtomicOpType.MAX)
atomic_min = functools.partial(atomic_rmw, atomic_type=AtomicOpType.MIN)
atomic_and = functools.partial(atomic_rmw, atomic_type=AtomicOpType.AND)
atomic_or = functools.partial(atomic_rmw, atomic_type=AtomicOpType.OR)
atomic_xor = functools.partial(atomic_rmw, atomic_type=AtomicOpType.XOR)

atomic_cas_p = jax_core.Primitive("atomic_cas")

def _atomic_cas_abstract_eval(ref_aval, cmp_aval, val_aval):
  if cmp_aval.dtype != val_aval.dtype:
    raise ValueError("Dtypes in cmp/val need to match")
  if ref_aval.shape:
    raise ValueError("Ref must be scalar.")
  if cmp_aval.shape:
    raise ValueError("Cmp must be scalar.")
  if val_aval.shape:
    raise ValueError("Val must be scalar.")
  if cmp_aval.shape != val_aval.shape:
    raise ValueError("Dtypes in cmp/val need to match")
  return jax_core.ShapedArray(val_aval.shape, val_aval.dtype), {state.WriteEffect(0)}
atomic_cas_p.def_effectful_abstract_eval(_atomic_cas_abstract_eval)

def atomic_cas(ref, cmp, val):
  return atomic_cas_p.bind(ref, cmp, val)

@state_discharge.register_discharge_rule(atomic_cas_p)
def _atomic_cas_discharge_rule(in_avals, out_avals, ref, cmp, val):
  del in_avals, out_avals
  new_val = jnp.where(ref == cmp, val, ref)
  return (new_val, None, None), ref

max_contiguous_p = jax_core.Primitive("max_contiguous")

max_contiguous_p.def_impl(lambda x, **_: x)
mlir.register_lowering(max_contiguous_p, lambda _, x, **__: [x])

def max_contiguous(x, values):
  if not isinstance(values, list):
    values = [values]
  return max_contiguous_p.bind(x, values=values)

def _max_contiguous_abstract_eval(aval, **_):
  return aval
max_contiguous_p.def_abstract_eval(_max_contiguous_abstract_eval)

multiple_of_p = jax_core.Primitive("multiple_of")

multiple_of_p.def_impl(lambda x, **_: x)
mlir.register_lowering(multiple_of_p, lambda _, x, **__: [x])

def multiple_of(x: jax.Array, values: list[int] | int) -> jax.Array:
  if not isinstance(values, list):
    values = [values]
  return multiple_of_p.bind(x, values=values)

def _multiple_of_abstract_eval(aval, **_):
  return aval
multiple_of_p.def_abstract_eval(_multiple_of_abstract_eval)

load_p = jax_core.Primitive('masked_load')


def _load_abstract_eval(*avals_flat, args_tree, **_):
  ref, indexers, _, _ = args_tree.unflatten(avals_flat)
  return (
      jax_core.ShapedArray(indexers[-1].get_indexer_shape(), ref.dtype),
      {state.ReadEffect(0)},
  )


load_p.def_effectful_abstract_eval(_load_abstract_eval)

def _load_pp_rule(eqn, context, settings):
  # Pretty prints `a = load x i` as `x[i] <- a`
  y, = eqn.outvars
  x, indexers, mask, other  = tree_util.tree_unflatten(eqn.params["args_tree"],
                                                       eqn.invars)
  # TODO(sharadmv): pretty print mask and other
  lhs = jax_core.pp_vars([y], context, print_shapes=settings.print_shapes)
  result = [
      lhs,
      pp.text(' <- '),
      sp.pp_ref_indexers(context, x, indexers)
  ]
  if mask is not None:
    result += [
        pp.text(" "),
        pp.text("mask="),
        pp.text(jax_core.pp_var(mask, context)),
    ]
  if other is not None:
    result += [
        pp.text(" "),
        pp.text("other="),
        pp.text(jax_core.pp_var(other, context)),
    ]
  return pp.concat(result)
jax_core.pp_eqn_rules[load_p] = _load_pp_rule


def _load_jvp(primals, tangents, args_tree, **params):
  ref_primal, indexers, mask, other_primal = args_tree.unflatten(primals)
  ref_tangent, _, _, other_tangent = args_tree.unflatten(tangents)
  if other_tangent is not None:
    other_tangent = ad_util.instantiate(other_tangent)
  return (
      load_p.bind(
          *tree_util.tree_leaves((ref_primal, indexers, mask, other_primal)),
          args_tree=args_tree,
          **params,
      ),
      load_p.bind(
          *tree_util.tree_leaves((ref_tangent, indexers, mask, other_tangent)),
          args_tree=args_tree,
          **params,
      ),
  )


ad.primitive_jvps[load_p] = _load_jvp

def uninitialized_value(shape, dtype):
  if jnp.issubdtype(dtype, jnp.floating):
    return jnp.full(shape, jnp.nan, dtype)
  elif jnp.issubdtype(dtype, jnp.integer):
    return jnp.full(shape, jnp.iinfo(dtype).min, dtype)
  elif jnp.issubdtype(dtype, jnp.bool):
    return jnp.full(shape, False, dtype)
  raise NotImplementedError(dtype)

def _pad_values_to_avoid_dynamic_slice_oob_shift(value,
                                   slice_sizes, unpad=False):
  """
  DynamicSlice and DynamicUpdateSlice adjust the start index in cases where the
  requested slice overruns the bounds of the array. This pads the array with
  uninitialised values such that the requested slice will never overrun.

  For example, if arr is [1.,2.,3.,4.] and a slice of size 4, start index 2 is
  requested then the result will be [3.,4.,NaN,NaN] after padding, rather than
  [1.,2.,3.,4.] from the unpadded array

  unpad=True performs the inverse operation
  """

  padding_config = tuple((0, slice_size, 0) for slice_size in slice_sizes)
  if unpad:
    padding_config = tuple((-low, -high, -interior)
                           for (low, high, interior) in padding_config)
  padding_value = uninitialized_value(shape=(), dtype=value.dtype)
  value = lax.pad(value,
                  padding_config=padding_config,
                  padding_value=padding_value)
  return value

_unpad_values_to_avoid_dynamic_slice_oob_shift = partial(
  _pad_values_to_avoid_dynamic_slice_oob_shift, unpad=True)

def _load_discharge_rule(in_avals, out_avals, *args_flat, args_tree, **_):
  del out_avals  # Unused.
  ref, indexers, mask, other = args_tree.unflatten(args_flat)
  # TODO(sharadmv): add support for multiple indexers
  if len(indexers) > 1:
    raise NotImplementedError("Only one indexer supported in discharge rule.")
  idx = indexers[0]
  if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
    # TODO(b/329733289): support strided load/store in interpret mode.
    for s in idx.indices:
      if isinstance(s, Slice) and s.stride > 1:
        raise NotImplementedError("Unimplemented stride support.")
    indices = idx.indices
    scalar_dims = [not isinstance(s, Slice) and not s.shape for s in indices]
    slice_starts = [s.start if isinstance(s, Slice) else s for s in indices]
    slice_sizes = tuple(s.size if isinstance(s, Slice) else 1 for s in indices)
    # fixes an inconstency with lax.dynamic_slice where if the slice goes out
    # of bounds, it will instead move the start_index backwards so the slice
    # will fit in memory.
    ref = _pad_values_to_avoid_dynamic_slice_oob_shift(ref, slice_sizes)
    out_ones = lax.dynamic_slice(ref, slice_starts, slice_sizes=slice_sizes)
    out_indexer = tuple(0 if scalar else slice(None) for scalar in scalar_dims)
    out = out_ones[out_indexer]
  elif all(not isinstance(s, Slice) for s in idx.indices):
    out = ref[idx.indices]
  else:
    raise NotImplementedError
  if mask is not None and other is not None:
    out = jnp.where(mask, out, other)
  return (None,) * len(in_avals), out


state_discharge.register_discharge_rule(load_p)(_load_discharge_rule)

swap_p = jax_core.Primitive('masked_swap')


def _swap_abstract_eval(*avals_flat, args_tree, **_):
  ref, indexers, val, _ = args_tree.unflatten(avals_flat)
  expected_output_shape = indexers[-1].get_indexer_shape()
  if expected_output_shape != val.shape:
    raise ValueError(
        f"Invalid shape for `swap`. Ref shape: {ref.shape}. "
        f"Value shape: {val.shape}. Indices: {indexers}. "
    )
  if ref.dtype != val.dtype:
    raise ValueError(
        f"Invalid dtype for `swap`. Ref dtype: {ref.dtype}. "
        f"Value dtype: {val.dtype}. "
    )
  return (
      jax_core.ShapedArray(expected_output_shape, ref.dtype),
      {state.WriteEffect(0)},
  )


swap_p.def_effectful_abstract_eval(_swap_abstract_eval)

def _swap_pp_rule(eqn, context, settings):
  # Pretty prints `a = swap x v i` as `a, x[i] <- x[i], v`
  # or:
  # Pretty prints `_ = swap x v i` as `x[i] <- v`
  y, = eqn.outvars
  x, indexers, val, mask = eqn.params["args_tree"].unflatten(eqn.invars)
  x_i = sp.pp_ref_indexers(context, x, indexers)
  if isinstance(y, jax_core.DropVar):
    return pp.concat([
        x_i,
        pp.text(" <- "), pp.text(jax_core.pp_var(val, context))])
  y = jax_core.pp_vars([y], context, print_shapes=settings.print_shapes)
  result = [
      y,
      pp.text(", "),
      x_i,
      pp.text(" <- "),
      x_i,
      pp.text(", "),
      pp.text(jax_core.pp_var(val, context)),
  ]
  if mask is not None:
    result += [
        pp.text(" "),
        pp.text("mask="),
        pp.text(jax_core.pp_var(mask, context)),
    ]
  return pp.concat(result)
jax_core.pp_eqn_rules[swap_p] = _swap_pp_rule


def _swap_jvp(primals, tangents, *, args_tree, **params):
  ref_primal, indexers, val_primal, mask = args_tree.unflatten(primals)
  ref_tangent, _, val_tangent, _ = args_tree.unflatten(tangents)
  val_tangent = ad_util.instantiate(val_tangent)
  return (
      swap_p.bind(
          *tree_util.tree_leaves((ref_primal, indexers, val_primal, mask)),
          args_tree=args_tree,
          **params,
      ),
      swap_p.bind(
          *tree_util.tree_leaves((ref_tangent, indexers, val_tangent, mask)),
          args_tree=args_tree,
          **params,
      ),
  )


ad.primitive_jvps[swap_p] = _swap_jvp


def _swap_discharge_rule(in_avals, out_avals, *args_flat, args_tree, **_):
  del out_avals  # Unused.
  ref, indexers, val, mask = args_tree.unflatten(args_flat)
  if len(indexers) > 1:
    raise NotImplementedError("Only one indexer supported in discharge rule.")
  idx = indexers[0]
  if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
    # TODO(b/329733289): support strided load/store in interpret mode.
    for s in idx.indices:
      if isinstance(s, Slice) and s.stride > 1:
        raise NotImplementedError("Unimplemented stride support.")
    indices = idx.indices
    scalar_dims = [
        i
        for i, s in enumerate(indices)
        if not isinstance(s, Slice) and not s.shape
    ]
    slice_starts = [s.start if isinstance(s, Slice) else s for s in indices]
    slice_sizes = tuple(s.size if isinstance(s, Slice) else 1 for s in indices)
    # Fixes an inconsistency with lax.dynamic_update_slice where if the slice
    # goes out of bounds, it will instead move the start_index backwards so the
    # slice will fit in memory.
    ref = _pad_values_to_avoid_dynamic_slice_oob_shift(ref, slice_sizes)
    out = lax.dynamic_slice(ref, slice_starts, slice_sizes=slice_sizes)
    out = jnp.squeeze(out, scalar_dims)
    if mask is not None:
      out_ = out
      out = jnp.where(mask, out, val)
      val = jnp.where(mask, val, out_)
    val = jnp.expand_dims(val, scalar_dims)
    x_new = lax.dynamic_update_slice(ref, val, start_indices=slice_starts)
    x_new = _unpad_values_to_avoid_dynamic_slice_oob_shift(x_new, slice_sizes)
  elif all(not isinstance(s, Slice) for s in idx.indices):
    out = ref[idx.indices]
    if mask is not None:
      out_ = out
      out = jnp.where(mask, out, val)
      val = jnp.where(mask, val, out_)
    x_new = ref.at[idx.indices].set(val)
  else:
    raise NotImplementedError
  return (x_new,) + (None,) * (len(in_avals) - 1), out


state_discharge.register_discharge_rule(swap_p)(_swap_discharge_rule)


def load(x_ref_or_view, idx, *, mask=None, other=None, cache_modifier=None,
         eviction_policy=None, volatile=False) -> jax.Array:
  x_ref, indexers = sp.get_ref_and_indexers(x_ref_or_view, idx, "load")
  args_flat, args_tree = tree_util.tree_flatten((x_ref, indexers, mask, other))
  return load_p.bind(
      *args_flat,
      args_tree=args_tree,
      cache_modifier=cache_modifier,
      eviction_policy=eviction_policy,
      is_volatile=volatile,
  )

def swap(x_ref_or_view, idx, val, *, mask=None, eviction_policy=None,
         _function_name="swap") -> Any:
  x_ref, indexers = sp.get_ref_and_indexers(x_ref_or_view, idx, _function_name)
  args_flat, args_tree = tree_util.tree_flatten((x_ref, indexers, val, mask))
  return swap_p.bind(
      *args_flat, args_tree=args_tree, eviction_policy=eviction_policy
  )

def store(x_ref_or_view, idx, val, *, mask=None, eviction_policy=None) -> None:
  _ = swap(x_ref_or_view, idx, val, mask=mask, eviction_policy=eviction_policy,
           _function_name="store")

def dot(a, b, trans_a: bool = False, trans_b: bool = False,
        allow_tf32: bool | None = None, precision=None):
  if (a.ndim != 2) or (b.ndim != 2):
    raise ValueError("`a` and `b` must be 2D arrays.")
  lhs_contract_dim = 0 if trans_a else 1
  rhs_contract_dim = 0 if not trans_b else 1
  if allow_tf32 is not None:
    if precision is not None:
      raise ValueError("Only one of allow_tf32 and precision can be specified")
    precision = lax.Precision.HIGH if allow_tf32 else lax.Precision.HIGHEST
  return jax.lax.dot_general(
      a,
      b,
      dimension_numbers=(((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())),
      precision=precision,
      preferred_element_type=jnp.float32,
  )


class PrintEffect(effects.Effect):
  __str__ = lambda self: "Print"


debug_print_effect = PrintEffect()

# TODO(slebedev): Consider making the effect ordered.
effects.lowerable_effects.add_type(PrintEffect)
effects.control_flow_allowed_effects.add_type(PrintEffect)
effects.remat_allowed_effects.add_type(PrintEffect)
effects.custom_derivatives_allowed_effects.add_type(PrintEffect)


debug_print_p = jax_core.Primitive("debug_print")
debug_print_p.multiple_results = True


def debug_print(fmt: str, *args: jax.ArrayLike):
  """Prints scalar values from inside a Pallas kernel.

  Args:
    fmt: A format string to be included in the output. The restrictions on the
      format string depend on the backend:
        * On GPU, when using Triton, ``fmt`` must not contain any placeholders
          (``{...}``), since it is always printed before any of the values.
        * On GPU, when using the experimental Mosaic GPU backend, ``fmt`` must
          contain a placeholder for each value to be printed. Format specs and
          conversions are not supported.
        * In TPU, if ``fmt`` contains placeholders, all values must be 32-bit
          integers. If there are no placeholders, the values are printed after
          the format string.
    *args: The scalar values to print.
  """  # fmt: skip
  has_placeholders = False
  if fmt:
    _, field_name, *_ = next(iter(string.Formatter().parse(fmt)))
    has_placeholders = field_name is not None
  return debug_print_p.bind(*args, fmt=fmt, has_placeholders=has_placeholders)


def check_debug_print_format(
    fmt: str, *args: jax.ArrayLike
):
  n_placeholders = 0
  for _, field, spec, conversion in string.Formatter().parse(fmt):
    if field is not None:
      n_placeholders += 1
    if spec or conversion:
      raise ValueError(
          "The format string should not contain any format specs or conversions"
      )
    if field:
      raise ValueError(
          "The format string should not reference arguments by position or name"
      )

  if len(args) != n_placeholders:
    raise TypeError(
        f"The format string expects {n_placeholders} "
        f"argument{'' if n_placeholders == 1 else 's'}, but got {len(args)}"
    )


@debug_print_p.def_impl
def debug_print_impl(*args: Any, fmt: str, has_placeholders: bool):
  if has_placeholders:
    print(fmt.format(*args))
  else:
    print(fmt, *args)
  return ()


@debug_print_p.def_effectful_abstract_eval
def debug_print_abstract_eval(*avals: Any, fmt: str, has_placeholders: bool):
  del fmt, has_placeholders
  if any(aval.shape for aval in avals):
    raise ValueError("Only scalar values are supported")
  return [], {debug_print_effect}


def debug_print_batching_rule(args, dims, **params):
  """Unrolls the print primitive across the mapped axis."""
  axis_size = next(x.shape[i] for x, i in zip(args, dims) if i is not None)

  # TODO(sharadmv): implement in terms of rolled loop unstead of unrolled.
  def get_arg_at_dim(i, dim, arg):
    if dim is batching.not_mapped:
      # Broadcast unmapped argument
      return arg
    return lax.index_in_dim(arg, i, axis=dim, keepdims=False)

  outs = []
  for i in range(axis_size):
    args_idx = map(functools.partial(get_arg_at_dim, i), dims, args)
    outs.append(debug_print_p.bind(*args_idx, **params))
  outs = [jnp.stack(xs) for xs in zip(*outs)]
  return outs, (0,) * len(outs)


batching.primitive_batchers[debug_print_p] = functools.partial(
    debug_print_batching_rule, debug_print_p
)


@functools.partial(mlir.register_lowering, debug_print_p)
def debug_print_lowering_rule(ctx, *args, **params):
  result, _, _ = mlir.emit_python_callback(
      ctx,
      functools.partial(debug_print_p.impl, **params),
      None,
      list(args),
      ctx.avals_in,
      ctx.avals_out,
      has_side_effect=True,
  )
  return result
