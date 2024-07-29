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

"""Module for calling pallas functions from JAX."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial, reduce
import itertools
from typing import Any

import jax
from jax import api_util
from jax import lax
from jax._src import ad_util
from jax._src import checkify
from jax._src import config
from jax._src import core as jax_core
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.primitives import uninitialized_value
from jax._src.state import discharge as state_discharge
from jax._src.state import utils as state_utils
from jax._src.util import (
    safe_map,
    safe_zip,
    split_list,
    tuple_insert,
    unzip2,
    weakref_lru_cache,
)
import jax.numpy as jnp
import numpy as np

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Grid = pallas_core.Grid
TupleGrid = pallas_core.TupleGrid
GridSpec = pallas_core.GridSpec
BlockMapping = pallas_core.BlockMapping
GridMapping = pallas_core.GridMapping
BlockSpec = pallas_core.BlockSpec
BlockSpecTree = pallas_core.BlockSpecTree
NoBlockSpec = pallas_core.NoBlockSpec
no_block_spec = pallas_core.no_block_spec

# See the docstring for GridMapping for the calling convention
pallas_call_p = jax_core.Primitive('pallas_call')
pallas_call_p.multiple_results = True

def _maybe_dynamic_slice(start_idx, block_shape, value, is_indexing):
  if start_idx is None:
    assert is_indexing is None
    return value
  assert is_indexing is not None
  start_idx = tuple(jnp.asarray(s, dtype=jnp.int32) for s in start_idx)
  output = lax.dynamic_slice(value, start_idx, slice_sizes=block_shape)
  squeeze_dims = tuple(np.arange(len(is_indexing))[np.array(is_indexing,
                                                            dtype=np.bool_)])
  return lax.squeeze(output, squeeze_dims)

def _maybe_dynamic_update_slice(start_idx, block_shape, value, update,
                                is_indexing):
  if start_idx is None:
    assert is_indexing is None
    return update
  assert is_indexing is not None
  start_idx = tuple(jnp.asarray(s, dtype=jnp.int32) for s in start_idx)
  broadcast_dims = tuple(i for i, b in enumerate(is_indexing)
                         if not b)
  update = lax.broadcast_in_dim(update, block_shape, broadcast_dims)
  assert update.shape == block_shape
  return lax.dynamic_update_slice(value, update, start_idx)

def _pad_values_to_block_dimension(value,
                                   block_shape):
  """Pads values so the shape evenly divides into block dimensions.

  For example, if values has a shape of (33, 2, 5) with a block_shape of
  (32, 2, 4), this function will pad the value of shape to (64, 2, 8).

  Args:
    value: Array to be padded.
    block_shape: Block shapes to use for padding. If None, no padding will
      be performed.

  Returns:
    A padded array.
  """
  padded_shape = tuple(
      ((v - 1) // b + 1) * b for v, b in zip(value.shape, block_shape)
  )
  if padded_shape != value.shape:
    pad_width = tuple((0, a-b) for a, b in zip(padded_shape, value.shape))
    pad_value = uninitialized_value(shape=(), dtype=value.dtype)
    value = jnp.pad(value, pad_width, constant_values=pad_value)
  return value

def _initialize_scratch_vals(scratch_avals) -> tuple[jax.Array, ...]:
  scratch_avals = (jax_core.raise_to_shaped(x) for x in scratch_avals)
  return tuple(uninitialized_value(a.shape, a.dtype) for a in scratch_avals)

def _initialize_output_vals(
    block_mappings_output: Iterable[BlockMapping],
    input_args, input_output_aliases) -> Sequence[jax.Array]:
  oi_map = {v: k for k, v in input_output_aliases}
  output_vals = []
  for i, bm in enumerate(block_mappings_output):
    if i in oi_map:
      output_vals.append(input_args[oi_map[i]])
    else:
      output_vals.append(uninitialized_value(bm.array_shape_dtype.shape,
                                             bm.array_shape_dtype.dtype))
  return output_vals

def _logical_to_interpret_mode_dtype(dtype):
  """Converts logical dtypes into JAX dtypes for interpret mode.

  This function is used to convert device-specific dtypes that have no
  corresponding equivalent in JAX/XLA into a type that can be executed
  by the XLA interpreter (e.g. TPU semaphores -> int32).
  """
  if (hasattr(dtype, "_rules") and
      hasattr(dtype._rules, "pallas_interpret_element_aval")):
    return dtype._rules.pallas_interpret_element_aval(dtype).dtype
  return dtype

def _logical_aval_to_interpret_mode_aval(aval):
  """Logical to interpret mode aval conversion."""
  if isinstance(aval, pallas_core.AbstractMemoryRef):
    inner_aval = _logical_aval_to_interpret_mode_aval(aval.inner_aval)
    return aval.update(inner_aval=inner_aval)
  if isinstance(aval, jax_core.ShapedArray):
    inner_dtype = _logical_to_interpret_mode_dtype(aval.dtype)
    return jax_core.ShapedArray(aval.shape, inner_dtype, weak_type=aval.weak_type)
  return aval

def _get_next_indices(grid, indices):
  next_indices = []
  carry = True
  for dim_size, index in reversed(list(zip(grid, indices))):
    i = jnp.where(carry, index + 1, index)
    carry = dim_size == i
    next_indices.append(jnp.where(carry, 0, i))
  return tuple(reversed(next_indices))

def _pallas_call_impl(*args, **kwargs):
  assert False  # We always jit a pallas call, we only need the lowering rule

def _pallas_call_impl_interpret(
    *args,
    jaxpr: jax_core.Jaxpr,
    name: str,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    compiler_params: Any):
  del compiler_params, name
  # If we're in interpreter mode, we *scan* over the grid and eval the
  # discharged jaxpr.
  dynamic_grid_args, args = split_list(  # type: ignore
      args, [grid_mapping.num_dynamic_grid_bounds]
  )
  dynamic_grid_args_iter = iter(dynamic_grid_args)
  grid = tuple(
      a if a is not pallas_core.dynamic_grid_dim
      else next(dynamic_grid_args_iter)
      for a in grid_mapping.grid
  )
  assert next(dynamic_grid_args_iter, None) is None
  with grid_mapping.trace_env():
    discharged_jaxpr, discharged_consts = state_discharge.discharge_state(jaxpr, ())
  if debug:
    print(discharged_jaxpr)
  out = _initialize_output_vals(grid_mapping.block_mappings_output,
                                args, input_output_aliases)
  scalars = args[grid_mapping.slice_index_ops]
  block_args = args[len(scalars):]
  # invars: [*scalar_prefetch, *consts, *inputs, *outputs, *scratch]
  # block_args now contains: *consts, *inputs, *outputs
  scratch_invars = jaxpr.invars[grid_mapping.slice_scratch_ops]
  scratch_avals = [v.aval for v in scratch_invars]
  scratch_values = _initialize_scratch_vals(scratch_avals)

  carry = []
  for x, bm in zip(itertools.chain(block_args, out), grid_mapping.block_mappings):
    if isinstance(bm.indexing_mode, pallas_core.Unblocked):
      padding = bm.indexing_mode.padding
      if padding is not None and any(p != (0, 0) for p in padding):
        if input_output_aliases:
          raise NotImplementedError("Padding with aliasing not supported.")
        pad_value = uninitialized_value(shape=(), dtype=x.dtype)
        x = lax.pad(x, pad_value, [(*p, 0) for p in padding])
    carry.append(x)

  is_indexing_dim = [
      tuple(b is pallas_core.mapped for b in bm.block_shape)
      for bm in grid_mapping.block_mappings
  ]
  block_shapes = [
      None if iid is None
      else tuple(1 if i else b for i, b in zip(iid, bm.block_shape))
      for iid, bm in zip(is_indexing_dim, grid_mapping.block_mappings)
  ]

  # Pad values to evenly divide into block dimensions. This matches the
  # behavior of the non-interpret mode. We pad with NaN, to make it easier
  # to catch OOB accesses.
  carry = map(_pad_values_to_block_dimension, carry, block_shapes)
  carry.extend(scratch_values)

  num_inout_blocks = len(block_args) + len(out)
  grid_start_indices = (jnp.int32(0),) * len(grid)
  if grid:
    num_iterations = reduce(jnp.multiply, grid)
  else:
    # Base case is always one iteration when grid is ()
    num_iterations = 1

  # The scan carry: (i, loop_idx, *consts, *ins, *outs, *scratch)
  # i:int32 is the interation index
  # loop_idx: tuple[int32] are the program ids for each grid axis
  def cond(carry):
    i, *_ = carry
    return i < num_iterations
  def body(carry):
    i, loop_idx, *carry_blocks = carry
    local_grid_env = tuple(
        pallas_core.GridAxis(idx, b)
        for dim, (idx, b) in enumerate(zip(loop_idx, grid))
        if dim not in grid_mapping.vmapped_dims
    )
    carry_consts_ins, scratch = split_list(carry_blocks, [num_inout_blocks])
    with pallas_core.grid_env(local_grid_env):
      start_indices = [
          None if bm is None else bm.compute_start_indices_interpret(loop_idx, *scalars)
          for bm in grid_mapping.block_mappings]
    blocks = map(_maybe_dynamic_slice, start_indices, block_shapes,
                 carry_consts_ins, is_indexing_dim)
    with pallas_core.grid_env(local_grid_env):
      assert len(discharged_jaxpr.invars) == len(scalars) + len(blocks) + len(
          scratch_values
      ), (
          len(discharged_jaxpr.invars),
          len(scalars),
          len(blocks),
          len(scratch_values),
      )
      blocks = jax_core.eval_jaxpr(discharged_jaxpr, discharged_consts, *scalars,
                                   *blocks, *scratch)

    _, out_inout, out_scratch = split_list(
        blocks, [grid_mapping.num_index_operands, num_inout_blocks])
    out_carry = map(_maybe_dynamic_update_slice, start_indices, block_shapes,
                    carry_consts_ins, out_inout, is_indexing_dim)
    return (i + 1, _get_next_indices(grid, loop_idx),
            *out_carry, *out_scratch)

  (_, _, *carry) = lax.while_loop(
      cond, body, (jnp.int32(0), grid_start_indices, *carry)
  )

  out_out = carry[len(block_args):len(block_args) + len(out)]
  out_nopad = []
  for o, bm in zip(out_out, grid_mapping.block_mappings_output):
    if isinstance(bm.indexing_mode, pallas_core.Unblocked):
      padding = bm.indexing_mode.padding
      if padding is not None and any(p != (0, 0) for p in padding):
        if input_output_aliases:
          raise NotImplementedError("Padding with aliasing not supported.")
        pad_low, pad_high = zip(*padding)
        limit_indices = [s - p for s, p in zip(o.shape, pad_high)]
        o = lax.slice(o, pad_low, limit_indices)
    if o.shape != bm.array_shape_dtype.shape:
      o = lax.slice(o, (0,) * o.ndim, bm.array_shape_dtype.shape)
    out_nopad.append(o)
  return out_nopad

pallas_call_p.def_impl(_pallas_call_impl)

def _pallas_call_abstract_eval(*avals, grid_mapping: GridMapping, **_):
  return tuple(jax_core.ShapedArray(bm.array_shape_dtype.shape,
                                    bm.array_shape_dtype.dtype)
               for bm in grid_mapping.block_mappings_output)
pallas_call_p.def_abstract_eval(_pallas_call_abstract_eval)

def _pallas_call_jvp_rule(primals, tangents, *, jaxpr, name,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping, debug, interpret, compiler_params: Any):
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError("interpret with dynamic grid bounds unsupported")
  if grid_mapping.num_index_operands:
    raise NotImplementedError
  if input_output_aliases:
    raise NotImplementedError("JVP with aliasing not supported.")
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  nonzero_tangents_with_outputs = nonzero_tangents + [True] * grid_mapping.num_outputs
  closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, nonzero_tangents_with_outputs, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  # `pallas_call` takes in inputs and returns outputs but its jaxpr *does not*.
  # `pallas_call` takes in a stateful jaxpr, meaning the jaxpr accepts input
  # `Ref`s that are read from followed by output `Ref`s that are written to.
  # This means that when we do `jvp_jaxpr` on the `jaxpr`, we get out a new
  # jaxpr that has tangents following primals. In order for this jaxpr to be
  # compatible w/ `pallas_call` (inputs then outputs), we need to shuffle around
  # the jaxpr's invars.
  primal_refs, primal_out_refs, tangent_refs, tangent_out_refs = split_list(
      jvp_jaxpr.invars, [len(primals), grid_mapping.num_outputs, len(tangents)]
  )
  invars = (*primal_refs, *tangent_refs, *primal_out_refs, *tangent_out_refs)
  effs = []
  for eff in jvp_jaxpr.effects:
    if isinstance(eff, effects.JaxprInputEffect):
      eff = eff.replace(
          input_index=invars.index(jvp_jaxpr.invars[eff.input_index])
      )
    effs.append(eff)
  jvp_jaxpr = jvp_jaxpr.replace(invars=invars, effects=effs)
  if debug:
    print(jvp_jaxpr)
  # TODO(necula): does this work with consts?
  in_bms, out_bms = split_list(grid_mapping.block_mappings, [len(primals)])
  jvp_bms = (*in_bms, *in_bms, *out_bms, *out_bms)
  jvp_grid_mapping = grid_mapping.replace(
      block_mappings=jvp_bms,
      num_inputs=grid_mapping.num_inputs * 2,
      num_outputs=grid_mapping.num_outputs * 2,
  )
  out_flat = pallas_call_p.bind(
      *primals,
      *tangents,
      jaxpr=jvp_jaxpr,
      name=f"{name}_jvp",
      grid_mapping=jvp_grid_mapping,
      interpret=interpret,
      debug=debug,
      input_output_aliases=(),
      compiler_params=compiler_params,
  )
  out_primals, out_tangents = split_list(out_flat, [len(out_flat) // 2])
  return out_primals, out_tangents
ad.primitive_jvps[pallas_call_p] = _pallas_call_jvp_rule

def _batch_block_mapping(grid_mapping: GridMapping,
                         axis_size: int,
                         aval: jax_core.ShapedArray,
                         dim: int | batching.NotMapped,
                         block_mapping: BlockMapping) -> BlockMapping:
  def _block_map_function(new_idx, *args):
    indices = jax_core.eval_jaxpr(block_mapping.index_map_jaxpr.jaxpr,
                                  block_mapping.index_map_jaxpr.consts,
                                  *args)
    if dim is not batching.not_mapped:
      indices.insert(dim, new_idx)
    return tuple(indices)
  idx_avals = [pallas_core.index_map_grid_aval, *block_mapping.index_map_jaxpr.in_avals]
  with grid_mapping.trace_env():
    block_mapping_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(_block_map_function), idx_avals)
  shape = block_mapping.block_shape
  if dim is batching.not_mapped:
    new_block_shape = shape
    new_array_shape_dtype = block_mapping.array_shape_dtype
  else:
    new_block_shape = tuple_insert(shape, dim, pallas_core.mapped)
    new_array_shape_dtype = jax.ShapeDtypeStruct(
        tuple_insert(block_mapping.array_shape_dtype.shape,
                     dim,
                     axis_size),
        block_mapping.array_shape_dtype.dtype)

  jaxpr = jax_core.ClosedJaxpr(block_mapping_jaxpr, consts)
  return block_mapping.replace(block_shape=new_block_shape,
                               array_shape_dtype=new_array_shape_dtype,
                               index_map_jaxpr=jaxpr)


def _broadcast_input_output_aliases(
    args: Sequence[jax.Array],
    dims: Sequence[int | batching.NotMapped],
    *,
    input_output_aliases: tuple[tuple[int, int], ...],
    axis_size: int,
) -> tuple[tuple[jax.Array, ...], tuple[int | batching.NotMapped, ...]]:
  """Broadcast input/output operands.

  When we have input/output aliasing, since the output will be mapped, we need
  to make sure to broadcast the input across that dimension if it is not
  mapped. If the input is mapped, but on a different axis, we tranpose the input
  to match the output.
  """

  args_ = list(args)
  dims_ = list(dims)
  for input_index, _ in input_output_aliases:
    dim = dims_[input_index]
    dims_[input_index] = 0
    if dim is batching.not_mapped:
      args_[input_index] = batching.broadcast(args_[input_index], axis_size, 0)
    elif dim != 0:
      # TODO(cjfj): Change output batching axis instead?
      args_[input_index] = jnp.moveaxis(args[input_index], dim, 0)

  return tuple(args_), tuple(dims_)


def _batch_with_explicit_loop(
    args: Sequence[jax.Array],
    dims: Sequence[int | batching.NotMapped],
    *,
    jaxpr: jax_core.Jaxpr,
    name: str,
    grid_mapping: GridMapping,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug: bool,
    interpret: bool,
    compiler_params: Any,
):
  """Batch the pallas_call by calling it in loop over the batch size.

  This function provides a fallback implementation of batching a pallas_call
  for the cases in which adding a batch dimension to the pallas grid is not
  supported. This is currently the case when the batched dimension corresponds
  to a dynamic axis or a scalar prefetch argument.

  This implementation builds a HLO loop that dynamic_slices the inputs according
  to the current iteration index and dynamic_updates an (initially empty) output
  allocation.
  """
  if not dims:
    raise NotImplementedError("vmapping pallas_call with no arguments.")

  (axis_size,) = {
      arg.shape[dim]
      for arg, dim in zip(args, dims)
      if dim is not batching.not_mapped
  }

  args, dims = _broadcast_input_output_aliases(
      args,
      dims,
      input_output_aliases=input_output_aliases,
      axis_size=axis_size,
  )

  # The output arrays are completelly overwritten, so we can just initialize
  # empty arrays.
  initial_state = [
      jnp.empty(tuple_insert(bm.array_shape_dtype.shape, 0, axis_size),
                dtype=bm.array_shape_dtype.dtype)
      for bm in grid_mapping.block_mappings_output
  ]

  def body(batch_index: jax.Array, state: list[jax.Array]) -> list[jax.Array]:
    batch_args = []

    for arg, dim in zip(args, dims):
      # If the argument is mapped, extract a slice of size 1 in the mapped
      # dimension at the current index.
      if dim is batching.not_mapped:
        batch_args.append(arg)
      else:
        batch_args.append(
            jnp.squeeze(
                jax.lax.dynamic_slice_in_dim(
                    operand=arg,
                    start_index=batch_index,
                    slice_size=1,
                    axis=dim,
                ),
                axis=dim,
            )
        )
    batch_out = pallas_call_p.bind(
        *batch_args,
        jaxpr=jaxpr,
        name=name,
        grid_mapping=grid_mapping,
        input_output_aliases=input_output_aliases,
        debug=debug,
        interpret=interpret,
        compiler_params=compiler_params,
    )
    for i, batch_out_array in enumerate(batch_out):
      state[i] = jax.lax.dynamic_update_index_in_dim(
          state[i],
          batch_out_array,
          batch_index,
          axis=0,
      )

    return state

  result = jax.lax.fori_loop(0, axis_size, body, initial_state, unroll=False)

  return result, (0,) * len(result)


def _pallas_call_batching_rule(
    args,
    dims,
    *,
    jaxpr: jax_core.Jaxpr,
    name: str,
    grid_mapping: GridMapping,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug: bool,
    interpret: bool,
    compiler_params: Any,
):
  def _maybe_squeeze_out_bdim(
      x: jax.Array, bdim: int | batching.NotMapped
  ) -> jax.Array:
    if bdim is batching.not_mapped:
      return x
    return jnp.squeeze(x, axis=bdim)

  axis_size, = {x.shape[d] for x, d in zip(args, dims)
                if d is not batching.not_mapped}
  if axis_size == 1:
    # Why are we even vmapping?
    args = map(_maybe_squeeze_out_bdim, args, dims)
    out = pallas_call_p.bind(
        *args,
        jaxpr=jaxpr,
        name=name,
        grid_mapping=grid_mapping,
        input_output_aliases=input_output_aliases,
        debug=debug,
        interpret=interpret,
        compiler_params=compiler_params,
    )
    return [jnp.expand_dims(x, 0) for x in out], (0,) * len(out)

  # The first num_dynamic_grid_bounds arguments are size-1 arrays that store
  # the size of the dynamic bounds.
  dynamic_grid_args, args = split_list(
      args, [grid_mapping.num_dynamic_grid_bounds]
  )
  dynamic_grid_dims, dims = split_list(
      dims, [grid_mapping.num_dynamic_grid_bounds]
  )
  if all(
      bdim is batching.not_mapped or arg.shape[bdim] == 1
      for arg, bdim in zip(dynamic_grid_args, dynamic_grid_dims)
  ):
    dynamic_grid_args = safe_map(
        _maybe_squeeze_out_bdim, dynamic_grid_args, dynamic_grid_dims
    )
  elif any(bdim is not batching.not_mapped for bdim in dynamic_grid_dims):
    # TODO(amagni, sharadmv): Explore possibility of batching dynamic grid
    # bounds.
    return _batch_with_explicit_loop(
        args=dynamic_grid_args + args,
        dims=dynamic_grid_dims + dims,
        jaxpr=jaxpr,
        name=name,
        grid_mapping=grid_mapping,
        input_output_aliases=input_output_aliases,
        debug=debug,
        interpret=interpret,
        compiler_params=compiler_params,
    )
  else:
    pass  # No dynamic grid dimensions
  del dynamic_grid_dims
  if grid_mapping.num_index_operands:
    scalar_args, args = split_list(args, [grid_mapping.num_index_operands])
    scalar_bdims, bdims = split_list(dims, [grid_mapping.num_index_operands])
    # Ordinarily, adding support for scalar prefetch in vmap would involve
    # modifying the block specs in a nontrivial way. However, if we are only
    # vmapping over 1-sized dimensions, we can just get rid of the dimensions
    # and pretend we were never vmapped over them at all.
    if all(
        bdim is batching.not_mapped or arg.shape[bdim] == 1
        for arg, bdim in zip(scalar_args, scalar_bdims)
    ):
      scalar_args = safe_map(_maybe_squeeze_out_bdim, scalar_args, scalar_bdims)
      scalar_bdims = [batching.not_mapped] * len(scalar_args)
      args = (*scalar_args, *args)
      dims = (*scalar_bdims, *bdims)
    else:
      # TODO(amagni,sharadmv,apaszke): enable efficient batching over
      # prefetched scalar args.
      return _batch_with_explicit_loop(
          args=scalar_args + args,
          dims=scalar_bdims + bdims,
          jaxpr=jaxpr,
          name=name,
          grid_mapping=grid_mapping,
          input_output_aliases=input_output_aliases,
          debug=debug,
          interpret=interpret,
          compiler_params=compiler_params,
      )

  if not dims:
    raise NotImplementedError("vmapping pallas_call with no arguments.")
  block_mappings = grid_mapping.block_mappings
  avals = [v.aval for v in jaxpr.invars]
  # How should we pick output dimensions? This actually matters because XLA
  # can't optimize our pallas kernels, and this layout impacts performance. For
  # now, because `vmap` doesn't really offer a way of inferring good output
  # dimensions. For now, we just use 0.
  # TODO(sharadmv): explore inferring better output dimensions via a heuristic
  # TODO(sharadmv): explore a long term solution to output dim inference

  args, dims = _broadcast_input_output_aliases(
      args, dims, input_output_aliases=input_output_aliases, axis_size=axis_size
  )

  all_dims = list(dims) + [0] * grid_mapping.num_outputs

  num_index_operands = grid_mapping.num_index_operands
  num_scratch_operands = grid_mapping.num_scratch_operands

  # Only add a batch dimension for the avals that actually have a grid mapping.
  # This excludes scalar prefetch inputs (the first in the list) and scratch
  # operands (the last in the list).
  avals_to_batch = avals[num_index_operands:(len(avals) - num_scratch_operands)]
  batched_block_mappings = map(
      partial(_batch_block_mapping, grid_mapping, axis_size),
      avals_to_batch,
      all_dims[num_index_operands:],
      block_mappings,
  )

  index_map_tree_args, index_map_tree_kwargs = grid_mapping.index_map_tree.unflatten(
      grid_mapping.index_map_avals)
  assert not index_map_tree_kwargs
  batched_index_map_args = (pallas_core.index_map_grid_aval,) + index_map_tree_args
  batched_index_map_avals, batched_index_map_tree = tree_util.tree_flatten(
      (batched_index_map_args, {}))
  batched_grid_mapping = grid_mapping.replace(
      grid=(axis_size, *grid_mapping.grid),
      block_mappings=tuple(batched_block_mappings),
      index_map_avals=batched_index_map_avals,
      index_map_tree=batched_index_map_tree,
      vmapped_dims=(0,) + tuple(a + 1 for a in grid_mapping.vmapped_dims))
  out = pallas_call_p.bind(
      *dynamic_grid_args,
      *args,
      jaxpr=jaxpr,
      name=f"batched_{name}",
      grid_mapping=batched_grid_mapping,
      input_output_aliases=input_output_aliases,
      debug=debug,
      interpret=interpret,
      compiler_params=compiler_params,
  )
  return out, (0,) * len(out)


batching.primitive_batchers[pallas_call_p] = _pallas_call_batching_rule


def checkify_pallas_kernel_body_jaxpr(
    body_jaxpr: jax_core.ClosedJaxpr,
    enabled_errors,
    error: checkify.Error,
    grid_mapping: GridMapping) -> tuple[
        jax_core.ClosedJaxpr, tree_util.PyTreeDef, set[checkify.ErrorEffect]]:
  err_vals, err_tree = tree_util.tree_flatten(error)
  err_vals = map(checkify.get_shaped_aval, err_vals)
  flat_err_and_in_vals = [*err_vals, *body_jaxpr.in_avals]

  with pallas_core.tracing_grid_env(grid_mapping.grid, ()):
    checked_jaxpr, out_tree, error_effects = checkify.jaxpr_to_checkify_jaxpr(
        body_jaxpr, enabled_errors, err_tree, *flat_err_and_in_vals)
  return checked_jaxpr, out_tree, error_effects

def pallas_call_checkify_rule(error: checkify.Error,
                              enabled_errors,
                              *args: jax_core.Value,
                              jaxpr: jax_core.Jaxpr,
                              interpret: bool,
                              input_output_aliases: tuple[tuple[int, int], ...],
                              grid_mapping: GridMapping,
                              **kwargs):
  # We implement the checkify rule in 4 steps:
  # 1) First, trace the kernel body to get the expected error shapes.
  # 2) Checkify the kernel body to obtain a jaxpr with errors as inputs
  #   and outputs.
  # 3) Create a new kernel which stores the errors in output memrefs instead of
  #   returning them, since pallas kernels do not return outputs.
  # 4) Create block specs for the error state and call pallas_call with
  #   the new kernel.
  dynamic_grid_bounds, scalars, args = split_list(  # type: ignore
      args, [grid_mapping.num_dynamic_grid_bounds,
             grid_mapping.num_index_operands]
  )
  num_scalars = len(scalars)
  num_kernel_inputs = len(args)
  num_kernel_outputs = grid_mapping.num_outputs

  # Trace the jaxpr to get an initial error value so the kernel jaxpr has all of
  # the required inputs.
  closed_jaxpr = pe.close_jaxpr(jaxpr)
  _jaxpr, _, error_effects = checkify_pallas_kernel_body_jaxpr(
      closed_jaxpr, enabled_errors, error, grid_mapping)
  error = error._add_placeholder_effects(error_effects)
  err_vals, err_in_tree = jax.tree.flatten(error)
  shaped_err_avals = map(checkify.get_shaped_aval, err_vals)

  # Trace the kernel jaxpr to get a checkified jaxpr. This jaxpr will have
  # all enabled errors removed, but have the error as inputs and return values.
  input_avals = [v.aval for v in jaxpr.invars]
  num_err_vals = len(err_vals)
  shaped_input_avals = tuple(jax_core.raise_to_shaped(x) for x in input_avals)
  checkify_in_avals = [*shaped_err_avals,
                       *shaped_input_avals]
  closed_kernel_jaxpr = pe.close_jaxpr(jaxpr)
  with pallas_core.tracing_grid_env(grid_mapping.grid, ()):
    checked_jaxpr, error_out_tree, _ = checkify.jaxpr_to_checkify_jaxpr(
        closed_kernel_jaxpr, enabled_errors, err_in_tree, *checkify_in_avals)

  # Create a new kernel to remove the error as an return value and instead
  # write them to a memref. This is because pallas kernels are expected
  # to have no return values but instead write their outputs to a ref.
  def checked_kernel_fn(*args):
    (scalars, in_error_refs, inputs, out_error_refs, outputs, scratch
     ) = split_list(
        args,
        [num_scalars, num_err_vals,
         num_kernel_inputs, num_err_vals, num_kernel_outputs])
    # TODO(b/350593266): Remove zero-indexing once we support ()-shaped scalars.
    input_error_vals = [err_ref[0, 0] for err_ref in in_error_refs]
    # We need to re-order the inputs here. A checkified jaxpr always expects
    # errors before other arguments.
    jaxpr_args = [*input_error_vals, *scalars, *inputs, *outputs, *scratch]
    assert len(checked_jaxpr.jaxpr.invars) == len(jaxpr_args)
    result_flat = jax_core.eval_jaxpr(
        checked_jaxpr.jaxpr, checked_jaxpr.consts, *jaxpr_args)
    output_errors, _ = split_list(result_flat, [num_err_vals])
    # Store new errors back in the error refs.
    for in_ref, out_ref, error in zip(
        in_error_refs, out_error_refs, output_errors):
      in_ref[0, 0] = error
      out_ref[0, 0] = error
    return []

  # Trace the new checked_kernel_fn with Memref inputs so that
  # we can replace the old kernel jaxpr with the new checked jaxpr in
  # pallas_call.

  # ensure_2d_shape is only necessary because pallas does not support
  # ()-shaped Memrefs.
  # TODO(b/350593266): Remove once we support ()-shaped scalars.
  def _ensure_2d_error_shape(arg):
    if isinstance(arg, jax_core.ShapedArray):
      dtype = arg.dtype
      return jax_core.ShapedArray((1, 1) + arg.shape, dtype=dtype,
                                  weak_type=arg.weak_type)
    elif isinstance(arg, jax.Array):
      return jnp.reshape(arg, (1, 1) + arg.shape)
    else:
      return jnp.array([[arg]])
  shaped_err_avals = map(_ensure_2d_error_shape, shaped_err_avals)
  err_vals = map(_ensure_2d_error_shape, err_vals)

  error_memref_aval = [pallas_core.AbstractMemoryRef(
      err_val, pallas_core.MemorySpace.ERROR) for err_val in shaped_err_avals]
  shaped_scalar_avals, input_aval, output_aval, scratch_aval = split_list(
      shaped_input_avals, [num_scalars, num_kernel_inputs, num_kernel_outputs])
  retrace_in_avals = [*shaped_scalar_avals, *error_memref_aval, *input_aval,
                      *error_memref_aval, *output_aval, *scratch_aval]
  jaxpr_flat_avals, jaxpr_in_tree = tree_util.tree_flatten(retrace_in_avals)
  wrapped_kernel_with_err, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(checked_kernel_fn), jaxpr_in_tree)
  debug = pe.debug_info(
    checked_kernel_fn, jaxpr_in_tree, out_tree_thunk, False, "checkify_pallas")
  with pallas_core.tracing_grid_env(grid_mapping.grid, ()):
    final_jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
        wrapped_kernel_with_err, jaxpr_flat_avals, debug)

  # Prepare pallas_call inputs. We need to create new block specs
  # for the new error inputs and outputs.
  error_block_specs = [pallas_core.BlockSpec(None, None)] * len(shaped_err_avals)
  error_paths, _ = unzip2(tree_util.tree_flatten_with_path(error_block_specs)[0])
  error_block_mappings = map(
        partial(
            pallas_core._convert_block_spec_to_block_mapping,
            index_map_avals=grid_mapping.index_map_avals,
            index_map_tree=grid_mapping.index_map_tree,
            grid=grid_mapping.grid,
            mapped_dims=grid_mapping.vmapped_dims,
            what="error"),
        error_block_specs, error_paths, shaped_err_avals)
  input_block_mappings, output_block_mappings = split_list(
      grid_mapping.block_mappings, [num_kernel_inputs,])
  grid_mapping_with_error = grid_mapping.replace(
      block_mappings=(*error_block_mappings, *input_block_mappings,
                      *error_block_mappings, *output_block_mappings),
      num_inputs=grid_mapping.num_inputs + len(error_block_mappings),
      num_outputs=grid_mapping.num_outputs + len(error_block_mappings)
  )
  # Bump all input_output_aliases by num_err_vals to make room for error
  # TODO(justinfu): Don't bump scalars here.
  input_output_aliases = tuple(
      (i+num_err_vals, o+num_err_vals) for (i, o) in input_output_aliases)
  input_output_aliases_with_error = tuple(
      (i+num_scalars, i) for i in range(num_err_vals)) + input_output_aliases

  new_vals_in = [*scalars, *err_vals, *args]
  result = pallas_call_p.bind(*dynamic_grid_bounds, *new_vals_in,
    jaxpr=final_jaxpr,
    interpret=interpret,
    grid_mapping=grid_mapping_with_error,
    input_output_aliases=input_output_aliases_with_error,
    **kwargs)
  errors, results = split_list(result, [num_err_vals])
  # TODO(b/350593266): Remove line below once we support ()-shaped scalars.
  errors = [err_val[0, 0] for err_val in errors]
  new_error, _ = jax.tree.unflatten(error_out_tree, errors)
  return new_error, results
checkify.error_checks[pallas_call_p] = pallas_call_checkify_rule

@weakref_lru_cache
def _trace_kernel_to_jaxpr(fun: Callable,
                           grid_mapping: GridMapping,
                           kernel_avals: tuple[pallas_core.AbstractMemRef, ...],
                           kernel_in_tree: tree_util.PyTreeDef,
                           interpret: bool):
  if interpret:
    kernel_avals = tuple(map(_logical_aval_to_interpret_mode_aval,
                             kernel_avals))
  wrapped_kernel_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(fun), kernel_in_tree)
  debug = pe.debug_info(fun, kernel_in_tree, out_tree_thunk, False, "pallas_call")
  with grid_mapping.trace_env():
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_kernel_fun,
                                                     kernel_avals, debug)
    if consts:
      # Pad ``block_mappings`` to account for the hoisted constants.
      # The constants will be right after the index operands and just before
      # the real inputs and outputs.
      jaxpr = state_utils.hoist_consts_to_refs(
          jaxpr,
          index=grid_mapping.num_index_operands,
          make_abstract_ref=lambda aval: pallas_core.AbstractMemoryRef(aval, None))
      num_constant_operands = len(consts)
      const_block_mappings = []
      for c_idx, c in enumerate(consts):
        const_block_mapping = pallas_core._convert_block_spec_to_block_mapping(
            pallas_core.BlockSpec(None, None),
            path=(tree_util.SequenceKey(c_idx),),
            array_aval=jax_core.ShapedArray(c.shape, c.dtype),
            index_map_avals=grid_mapping.index_map_avals,
            index_map_tree=grid_mapping.index_map_tree,
            grid=grid_mapping.grid,
            mapped_dims=(),
            what="consts",
        )
        const_block_mappings.append(const_block_mapping)

      grid_mapping = grid_mapping.replace(
          block_mappings=(*const_block_mappings, *grid_mapping.block_mappings),
          num_constant_operands=num_constant_operands,
      )
  kernel_out_tree = out_tree_thunk()
  if kernel_out_tree != tree_util.tree_structure(None):
    raise ValueError(
        "The kernel function in a pallas_call should return None. "
        f"Found a PyTree: {kernel_out_tree}")
  return grid_mapping, jaxpr, consts

def _extract_function_name(f: Callable, name: str | None) -> str:
  if name is None:
    name = f.__name__ if hasattr(f, "__name__") and f.__name__ else "func"
  return name


_PALLAS_USE_MOSAIC_GPU = config.bool_flag(
    "jax_pallas_use_mosaic_gpu",
    default=config.bool_env("JAX_PALLAS_USE_MOSAIC_GPU", False),
    help=(
        "If True, lower Pallas kernels to the experimental Mosaic GPU"
        " dialect, instead of Trition IR."
    ),
)


def _unsupported_lowering_error(platform: str) -> Exception:
  return ValueError(
      f"Cannot lower pallas_call on platform: {platform}. To use Pallas on GPU,"
      " install jaxlib GPU 0.4.24 or newer. To use Pallas on TPU, install"
      " jaxlib TPU and libtpu. See"
      " https://jax.readthedocs.io/en/latest/installation.html."
  )


def _pallas_call_lowering(
    ctx: mlir.LoweringRuleContext, *in_nodes, interpret: bool, **params
):
  if interpret:
    # If we are in interpret mode, we don't care what platform we are on.
    impl = partial(_pallas_call_impl_interpret, **params)
    return mlir.lower_fun(impl, multiple_results=True)(ctx, *in_nodes)

  def cpu_lowering(ctx: mlir.LoweringRuleContext,
                   *in_nodes: mlir.ir.Value | Sequence[mlir.ir.Value],
                   **params):
    raise ValueError("Only interpret mode is supported on CPU backend.")

  def tpu_lowering(ctx: mlir.LoweringRuleContext,
                   *in_nodes: mlir.ir.Value | Sequence[mlir.ir.Value],
                   **params):
    if mosaic_tpu_backend is None:
      raise _unsupported_lowering_error("tpu")
    return mosaic_tpu_backend.pallas_call_tpu_lowering_rule(
        ctx, *in_nodes, **params
    )

  def gpu_lowering(ctx: mlir.LoweringRuleContext,
                   *in_nodes: mlir.ir.Value | Sequence[mlir.ir.Value],
                   **params):
    try:
      if _PALLAS_USE_MOSAIC_GPU.value:
        from jax._src.pallas.mosaic_gpu import pallas_call_registration
      else:
        from jax._src.pallas.triton import pallas_call_registration  # type: ignore
    except ImportError:
      raise _unsupported_lowering_error("gpu")
    return pallas_call_registration.pallas_call_lowering(
        ctx, *in_nodes, **params
    )

  return mlir.lower_per_platform(ctx, "pallas_call",
                                 dict(cpu=cpu_lowering,
                                      tpu=tpu_lowering,
                                      cuda=gpu_lowering,
                                      rocm=gpu_lowering),
                                 None,  # default_rule
                                 effects.no_effects,
                                 *in_nodes,
                                 interpret=interpret,
                                 **params)


mlir.register_lowering(pallas_call_p, _pallas_call_lowering)


def _pallas_custom_str_eqn_compact(
    prim: jax_core.Primitive, params: dict[Any, Any]
) -> str:
  del prim, params
  # Hide most info from compact str representation
  return "pallas_call"
jax_core.custom_str_eqn_compact_rules[pallas_call_p] = (
    _pallas_custom_str_eqn_compact
)

def _pallas_call_typecheck_rule(*in_avals, grid_mapping, **params):
  with grid_mapping.trace_env():
    return pallas_call_p.abstract_eval(
        *in_avals, grid_mapping=grid_mapping, **params
    )
jax_core.custom_typechecks[pallas_call_p] = _pallas_call_typecheck_rule


def pallas_call(
    f: Callable[..., None],
    out_shape: Any,
    *,
    grid_spec: GridSpec | None = None,
    grid: TupleGrid = (),
    in_specs: BlockSpecTree = no_block_spec,
    out_specs: BlockSpecTree = no_block_spec,
    input_output_aliases: dict[int, int] = {},
    debug: bool = False,
    interpret: bool = False,
    name: str | None = None,
    compiler_params: dict[str, Any] | None = None,
) -> Callable[..., Any]:
  """Invokes a Pallas kernel on some inputs.

  See `Pallas Quickstart <https://jax.readthedocs.io/en/latest/pallas/quickstart.html>`_.

  Args:
    f: the kernel function, that receives a Ref for each input and output.
      The shape of the Refs are given by the ``block_shape`` in the
      corresponding ``in_specs`` and ``out_specs``.
    out_shape: a PyTree of :class:`jax.ShapeDtypeStruct` describing the shape
      and dtypes of the outputs.
    grid_spec: An alternative way to specify ``grid``, ``in_specs``, and
      ``out_specs``. If given, those other parameters must not be also given.
    grid: the iteration space, as a tuple of integers. The kernel is executed
      as many times as ``prod(grid)``.
      See details at :ref:`pallas_grid`.
    in_specs: a PyTree of :class:`jax.experimental.pallas.BlockSpec` with
      a structure matching that of the positional arguments.
      The default value for ``in_specs`` specifies the whole array for all
      inputs, e.g., as ``pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)``.
      See details at :ref:`pallas_blockspec`.
    out_specs: a PyTree of :class:`jax.experimental.pallas.BlockSpec` with
      a structure matching that of the outputs.
      The default value for ``out_specs`` specifies the whole array,
      e.g., as ``pl.BlockSpec(x.shape, lambda *indices: (0,) * x.ndim)``.
      See details at :ref:`pallas_blockspec`.
    input_output_aliases: a dictionary mapping the index of some inputs to
      the index of the output that aliases them. These indices are in the
      flattened inputs and outputs.
    debug: if True, Pallas prints various intermediate forms of the kernel
      as it is being processed.
    interpret: runs the ``pallas_call`` as a ``jax.jit`` of a scan over the
      grid whose body is the kernel lowered as a JAX function. This does not
      require a TPU or a GPU, and is the only way to run Pallas kernels on CPU.
      This is useful for debugging.
    name: TO BE DOCUMENTED.
    compiler_params: TO BE DOCUMENTED.

  Returns:
    A function that can be called on a number of positional array arguments to
    invoke the Pallas kernel.

  """
  name = _extract_function_name(f, name)
  if compiler_params is None:
    compiler_params = {}

  if grid_spec is None:
    grid_spec = GridSpec(grid, in_specs, out_specs)
  else:
    if grid:
      raise ValueError(
          "If `grid_spec` is specified, then `grid` must "
          f"be `()`. It is {grid}")
    if in_specs is not no_block_spec:
      raise ValueError(
          "If `grid_spec` is specified, then `in_specs` must "
          f"be `no_block_spec`. It is {in_specs}")
    if out_specs is not no_block_spec:
      raise ValueError(
          "If `grid_spec` is specified, then `out_specs` must "
          f"be `no_block_spec`. It is {out_specs}")
  del grid, in_specs, out_specs
  grid_spec, dynamic_grid_bounds = pallas_core.unzip_dynamic_grid_bounds(grid_spec)
  # TODO(necula): this canonicalization may be convenient for some usage
  # but it is lossy, because it prevents expressing functions that return
  # lists.
  if isinstance(out_shape, list):
    out_shape = tuple(out_shape)
  flat_out_shapes_with_paths, out_tree = tree_util.tree_flatten_with_path(out_shape)
  out_paths, flat_out_shapes = unzip2(flat_out_shapes_with_paths)
  flat_out_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype)  # type: ignore
                     for x in flat_out_shapes]
  @jax.jit
  def wrapped(*args):
    flat_args_with_paths, in_tree = tree_util.tree_flatten_with_path(args)
    in_paths, flat_args = unzip2(flat_args_with_paths)
    flat_in_avals = tuple(jax_core.raise_to_shaped(jax_core.get_aval(a))
                          for a in flat_args)
    flat_out_avals = tuple(jax_core.ShapedArray(v.shape, v.dtype)
                           for v in flat_out_shapes)
    # TODO(necula): check that input_output_aliases is well-formed: no duplicates, etc.
    kernel_avals, grid_mapping = pallas_core.get_grid_mapping(
        grid_spec,
        flat_in_avals, in_tree, in_paths,
        flat_out_avals, out_tree, out_paths)
    flat_kernel_avals, kernel_in_tree = tree_util.tree_flatten(kernel_avals)
    grid_mapping, jaxpr, consts = _trace_kernel_to_jaxpr(
        f, grid_mapping, tuple(flat_kernel_avals), kernel_in_tree,
        interpret=interpret)
    for i_idx, o_idx in input_output_aliases.items():
      if i_idx not in range(len(flat_in_avals)):
        raise ValueError(
            f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' with "
            f"input index {i_idx} outside the range "
            f"[0, {len(flat_in_avals)})")
      if o_idx not in range(len(flat_out_avals)):
        raise ValueError(
            f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' with "
            f"output index {o_idx} outside the range "
            f"[0, {len(flat_out_avals)})")
      in_aval = flat_in_avals[i_idx]
      out_aval = flat_out_avals[o_idx]
      if in_aval.shape != out_aval.shape or in_aval.dtype != out_aval.dtype:
        raise ValueError(
            f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
            f"referring to input{tree_util.keystr(in_paths[i_idx])} with "
            f"abstract value {in_aval} "
            f"and to output{tree_util.keystr(out_paths[o_idx])} with "
            f"a different abstract value {out_aval}.")

    index_args, rest_args = split_list(flat_args, [grid_mapping.num_index_operands])
    out_flat = pallas_call_p.bind(
        *dynamic_grid_bounds, *index_args, *consts, *rest_args,
        jaxpr=jaxpr, name=name,
        debug=debug,
        interpret=interpret,
        grid_mapping=grid_mapping,
        input_output_aliases=tuple(input_output_aliases.items()),
        compiler_params=compiler_params)
    out = tree_util.tree_unflatten(out_tree, out_flat)
    return out
  return wrapped


# We import the TPU backend at the top level because it defines flags. Note that
# we can only do that at the bottom of this file, beacuse it also depends on
# this module already being initialized.

try:
  from jax._src.pallas.mosaic import pallas_call_registration as mosaic_tpu_backend
except ImportError:
  mosaic_tpu_backend = None  # type: ignore
