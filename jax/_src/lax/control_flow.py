# coding=utf-8
# Copyright 2019 Google LLC
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
"""
Control flow primitives.
"""


import collections
import functools
from functools import partial
import inspect
import itertools
import operator
import os
from typing import Any, Callable, Optional, Sequence, Tuple, List, TypeVar

import numpy as np

import jax
from jax._src import api
from jax import core
from jax._src import ad_checkpoint
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import util
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax import windowed_reductions
from jax import linear_util as lu
from jax.core import ConcreteArray, ShapedArray, raise_to_shaped
from jax._src.api_util import flatten_fun_nokwargs
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters import batching
from jax.interpreters import masking
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib import xla_client
from jax._src.traceback_util import api_boundary
from jax._src.util import (unzip2, unzip3, safe_map, safe_zip, split_list,
                           extend_name_stack, wrap_name, weakref_lru_cache)
from jax.tree_util import (tree_flatten, tree_unflatten, treedef_is_leaf,
                           treedef_children, treedef_tuple, tree_map,
                           tree_leaves, tree_structure)
from jax._src import ad_util
from jax.config import config

xops = xla_client.ops

_map = safe_map
zip = safe_zip
_reduce = functools.reduce

T = TypeVar('T')
Array = Any
BooleanNumeric = Any  # A bool, or a Boolean array.

@weakref_lru_cache
def _initial_style_open_jaxpr(fun: Callable, in_tree, in_avals,
                              primitive_name: Optional[str] = None):
  wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  debug = pe.debug_info(fun, in_tree, False, primitive_name or "<unknown>")
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals, debug)
  return jaxpr, consts, out_tree()

@weakref_lru_cache
def _initial_style_jaxpr(fun: Callable, in_tree, in_avals,
                         primitive_name: Optional[str] = None):
  jaxpr, consts, out_tree = _initial_style_open_jaxpr(
      fun, in_tree, in_avals, primitive_name)
  closed_jaxpr = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
  return closed_jaxpr, consts, out_tree

class WeakRefSequence:
  lst: List[Any]
  def __init__(self, lst):
    self.lst = list(lst)

@weakref_lru_cache
def _initial_style_jaxprs_with_common_consts(
    funs_: WeakRefSequence, in_tree, in_avals, primitive_name: str):
  # When staging the branches of a conditional into jaxprs, constants are
  # extracted from each branch and converted to jaxpr arguments. To use the
  # staged jaxprs as the branches to a conditional *primitive*, we need for
  # their (input) signatures to match. This function "joins" the staged jaxprs:
  # for each one, it makes another that accepts *all* constants, but only uses
  # those that it needs (dropping the rest).

  funs: List[Callable] = funs_.lst
  jaxprs, all_consts, all_out_trees = \
      unzip3(_initial_style_open_jaxpr(fun, in_tree, in_avals, primitive_name)
             for fun in funs)

  newvar = core.gensym(jaxprs, suffix='_')
  all_const_avals = [[raise_to_shaped(core.get_aval(c)) for c in consts]
                     for consts in all_consts]
  unused_const_vars = [[newvar(aval) for aval in const_avals]
                       for const_avals in all_const_avals]

  def pad_jaxpr_constvars(i, jaxpr):
    prefix = util.concatenate(unused_const_vars[:i])
    suffix = util.concatenate(unused_const_vars[i + 1:])
    constvars = [*prefix, *jaxpr.constvars, *suffix]
    return core.Jaxpr(constvars=constvars, invars=jaxpr.invars,
                      outvars=jaxpr.outvars, eqns=jaxpr.eqns)

  consts = util.concatenate(all_consts)
  jaxprs = [pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs)]
  closed_jaxprs = [core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
                   for jaxpr in jaxprs]
  return closed_jaxprs, consts, all_out_trees

def _abstractify(x):
  return raise_to_shaped(core.get_aval(x))

def _typecheck_param(prim, param, name, msg_required, pred):
  if not pred:
    msg = (f'invalid {prim} param {name} of type {type(param).__name__}, '
           f'{msg_required} required:')
    param_str = str(param)
    sep = os.linesep if os.linesep in param_str else ' '
    msg = sep.join([msg, param_str])
    raise core.JaxprTypeError(msg)


### fori_loop and while_loop

def _fori_cond_fun(loop_carry):
  i, upper, _ = loop_carry
  return lax.lt(i, upper)

@weakref_lru_cache
def _fori_body_fun(body_fun):
  def while_body_fun(loop_carry):
    i, upper, x = loop_carry
    return lax.add(i, lax._const(i, 1)), upper, body_fun(i, x)
  return while_body_fun

@weakref_lru_cache
def _fori_scan_body_fun(body_fun):
  def scanned_fun(loop_carry, _):
    i, x = loop_carry
    return (i + 1, body_fun(i, x)), None
  return scanned_fun

@api_boundary
def fori_loop(lower, upper, body_fun, init_val):
  """Loop from ``lower`` to ``upper`` by reduction to :func:`jax.lax.while_loop`.

  The type signature in brief is

  .. code-block:: haskell

    fori_loop :: Int -> Int -> ((Int, a) -> a) -> a -> a

  The semantics of ``fori_loop`` are given by this Python implementation::

    def fori_loop(lower, upper, body_fun, init_val):
      val = init_val
      for i in range(lower, upper):
        val = body_fun(i, val)
      return val

  Unlike that Python version, ``fori_loop`` is implemented in terms of either a
  call to :func:`jax.lax.while_loop` or a call to :func:`jax.lax.scan`. If the
  trip count is static (meaning known at tracing time, perhaps because ``lower``
  and ``upper`` are Python integer literals) then the ``fori_loop`` is
  implemented in terms of ``scan`` and reverse-mode autodiff is supported;
  otherwise, a ``while_loop`` is used and reverse-mode autodiff is not
  supported.  See those functions' docstrings for more information.

  Also unlike the Python analogue, the loop-carried value ``val`` must hold a
  fixed shape and dtype across all iterations (and not just be consistent up to
  NumPy rank/shape broadcasting and dtype promotion rules, for example). In
  other words, the type ``a`` in the type signature above represents an array
  with a fixed shape and dtype (or a nested tuple/list/dict container data
  structure with a fixed structure and arrays with fixed shape and dtype at the
  leaves).

  Args:
    lower: an integer representing the loop index lower bound (inclusive)
    upper: an integer representing the loop index upper bound (exclusive)
    body_fun: function of type ``(int, a) -> a``.
    init_val: initial loop carry value of type ``a``.

  Returns:
    Loop value from the final iteration, of type ``a``.
  """
  # TODO(phawkins): perhaps do more type checking here, better error messages.
  lower_dtype = dtypes.canonicalize_dtype(lax.dtype(lower))
  upper_dtype = dtypes.canonicalize_dtype(lax.dtype(upper))
  if lower_dtype != upper_dtype:
    msg = ("lower and upper arguments to fori_loop must have equal types, "
           "got {} and {}")
    raise TypeError(msg.format(lower_dtype.name, upper_dtype.name))

  # If we can specialize on the trip count, call scan instead of a while_loop
  # to enable efficient reverse-mode differentiation.
  if (isinstance(core.get_aval(lower), ConcreteArray) and
      isinstance(core.get_aval(upper), ConcreteArray)):
    try:
      lower_ = int(lower)
      upper_ = int(upper)
    except TypeError:
      use_scan = False
    else:
      use_scan = True
  else:
    use_scan = False

  if use_scan:
    if config.jax_disable_jit and upper_ == lower_:
      # non-jit implementation of scan does not support length=0
      return init_val

    (_, result), _ = scan(_fori_scan_body_fun(body_fun), (lower_, init_val),
                          None, length=upper_ - lower_)
  else:
    _, _, result = while_loop(_fori_cond_fun, _fori_body_fun(body_fun),
                              (lower, upper, init_val))
  return result


@api_boundary
def while_loop(cond_fun: Callable[[T], BooleanNumeric],
               body_fun: Callable[[T], T],
               init_val: T) -> T:
  """Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.

  The type signature in brief is

  .. code-block:: haskell

    while_loop :: (a -> Bool) -> (a -> a) -> a -> a

  The semantics of ``while_loop`` are given by this Python implementation::

    def while_loop(cond_fun, body_fun, init_val):
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val

  Unlike that Python version, ``while_loop`` is a JAX primitive and is lowered
  to a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Also unlike the Python analogue, the loop-carried value ``val`` must hold a
  fixed shape and dtype across all iterations (and not just be consistent up to
  NumPy rank/shape broadcasting and dtype promotion rules, for example). In
  other words, the type ``a`` in the type signature above represents an array
  with a fixed shape and dtype (or a nested tuple/list/dict container data
  structure with a fixed structure and arrays with fixed shape and dtype at the
  leaves).

  Another difference from using Python-native loop constructs is that
  ``while_loop`` is not reverse-mode differentiable because XLA computations
  require static bounds on memory requirements.

  Args:
    cond_fun: function of type ``a -> Bool``.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop carry value.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.
  """
  if config.jax_disable_jit:
    try:
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val
    except core.ConcretizationTypeError:
      # Can't run this while_loop in Python (e.g. because there's a vmap
      # transformation on it), so we fall back to the primitive version.
      pass

  def _create_jaxpr(init_val):
    init_vals, in_tree = tree_flatten((init_val,))
    init_avals = tuple(_map(_abstractify, init_vals))
    cond_jaxpr, cond_consts, cond_tree = _initial_style_jaxpr(
        cond_fun, in_tree, init_avals, "while_cond")
    body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
        body_fun, in_tree, init_avals, "while_loop")
    if not treedef_is_leaf(cond_tree) or len(cond_jaxpr.out_avals) != 1:
      msg = "cond_fun must return a boolean scalar, but got pytree {}."
      raise TypeError(msg.format(cond_tree))
    pred_aval = cond_jaxpr.out_avals[0]
    if (not isinstance(pred_aval, ShapedArray)
        or pred_aval.strip_weak_type().strip_named_shape() != ShapedArray((), np.bool_)):
      msg = "cond_fun must return a boolean scalar, but got output type(s) {}."
      raise TypeError(msg.format(cond_jaxpr.out_avals))
    return init_vals, init_avals, body_jaxpr, in_tree, cond_jaxpr, cond_consts, body_consts, body_tree

  # The body input and output avals must match exactly. However, we want to account for
  # the case when init contains weakly-typed values (e.g. Python scalars), with avals that
  # may not match the output despite being compatible by virtue of their weak type.
  # To do this, we compute the jaxpr in two passes: first with the raw inputs, and if
  # necessary, a second time with modified init values.
  init_vals, init_avals, body_jaxpr, in_tree, *rest = _create_jaxpr(init_val)
  new_init_vals, changed = _promote_weak_typed_inputs(init_vals, init_avals, body_jaxpr.out_avals)
  if changed:
    new_init_val, = tree_unflatten(in_tree, new_init_vals)
    init_vals, init_avals, body_jaxpr, in_tree, *rest = _create_jaxpr(new_init_val)
  cond_jaxpr, cond_consts, body_consts, body_tree = rest

  in_tree_children = in_tree.children()
  assert len(in_tree_children) == 1
  _check_tree_and_avals("body_fun output and input",
                        body_tree, body_jaxpr.out_avals,
                        in_tree_children[0], init_avals)
  outs = while_p.bind(*cond_consts, *body_consts, *init_vals,
                      cond_nconsts=len(cond_consts), cond_jaxpr=cond_jaxpr,
                      body_nconsts=len(body_consts), body_jaxpr=body_jaxpr)
  return tree_unflatten(body_tree, outs)

def _while_loop_abstract_eval(*args, **kwargs):
  return _map(raise_to_shaped, kwargs["body_jaxpr"].out_avals)

def _while_loop_translation_rule(ctx, avals_in, avals_out, *args, cond_jaxpr,
                                 body_jaxpr, cond_nconsts, body_nconsts):
  c = ctx.builder
  cond_consts, body_consts, init_vals = split_list(args, [cond_nconsts, body_nconsts])
  batched = bool(cond_jaxpr.out_avals[0].shape)

  # Since jaxprs don't have tuples and have multiple return values, but we need
  # the HLO While loop to take a single tuple input and output a single boolean
  # (for the cond computation) or a single tuple output (for the body
  # computation), we build XLA computations that handle the tuple munging before
  # generating a Call into the computations formed from the jaxprs.

  init_carry = xops.Tuple(c, cond_consts + body_consts + init_vals)

  cond_c = xla_client.XlaBuilder("cond_computation")
  cond_carry = xla.parameter(cond_c, 0, c.get_shape(init_carry))
  cond_carry_elts = [xops.GetTupleElement(cond_carry, i) for i in range(len(args))]
  x, _, z = split_list(cond_carry_elts, [cond_nconsts, body_nconsts])
  name_stack = extend_name_stack(ctx.name_stack, 'while')
  cond_ctx = ctx.replace(builder=cond_c,
                         name_stack=extend_name_stack(name_stack, 'cond'))
  pred, = xla.jaxpr_subcomp(
      cond_ctx, cond_jaxpr.jaxpr,
      _map(partial(xla.pyval_to_ir_constant, cond_c), cond_jaxpr.consts),
      *(x + z))
  if batched:
    scalar = ShapedArray((), np.bool_)
    or_ = xla.primitive_subcomputation(ctx.platform, ctx.axis_env, lax.or_p,
                                       scalar, scalar)
    pred = xops.Reduce(cond_c, [pred], [xops.Constant(cond_c, np.array(False))],
                       or_, list(range(cond_jaxpr.out_avals[0].ndim)))

  body_c = xla_client.XlaBuilder("body_computation")
  body_carry = xla.parameter(body_c, 0, c.get_shape(init_carry))
  body_carry_elts = [xops.GetTupleElement(body_carry, i) for i in range(len(args))]
  x, y, z = split_list(body_carry_elts, [cond_nconsts, body_nconsts])
  body_ctx = ctx.replace(builder=body_c,
                         name_stack=extend_name_stack(name_stack, 'body'))
  new_z = xla.jaxpr_subcomp(
      body_ctx, body_jaxpr.jaxpr,
      _map(partial(xla.pyval_to_ir_constant, body_c), body_jaxpr.consts),
      *(y + z))
  if batched:
    body_pred_ctx = body_ctx.replace(
        name_stack=extend_name_stack(name_stack, 'body_pred'))
    body_pred, = xla.jaxpr_subcomp(
        body_pred_ctx, cond_jaxpr.jaxpr,
        _map(partial(xla.pyval_to_ir_constant, body_c), cond_jaxpr.consts),
        *(x + z))
    new_z = _map(partial(_pred_bcast_select, body_c, body_pred), new_z, z,
                 body_jaxpr.out_avals)
    assert _map(body_c.get_shape, new_z) == _map(body_c.get_shape, z) # no broadcast
  new_carry = xops.Tuple(body_c, [*x, *y, *new_z])

  ans = xops.While(cond_c.build(pred), body_c.build(new_carry), init_carry)
  ans_elts = [xops.GetTupleElement(ans, i) for i in range(len(args))]
  _,  _, z = split_list(ans_elts, [cond_nconsts, body_nconsts])
  return z

def _pred_bcast_select(c, pred, x, y, x_y_aval: core.AbstractValue):
  pred_shape = c.get_shape(pred).dimensions()
  x_shape = c.get_shape(x).dimensions()
  y_shape = c.get_shape(y).dimensions()
  assert x_shape == y_shape
  if x_y_aval is core.abstract_unit:
    return x
  elif x_y_aval is core.abstract_token:
    return xops.AfterAll(c, [x, y])
  else:
    assert pred_shape == x_shape[:len(pred_shape)] == y_shape[:len(pred_shape)], (pred_shape, x_shape, y_shape)
    bcast_pred = xops.BroadcastInDim(pred, x_shape, list(range(len(pred_shape))))
    return xops.Select(bcast_pred, x, y)

def _while_loop_batching_rule(axis_size, axis_name, main_type, args, dims,
                              cond_nconsts, cond_jaxpr,
                              body_nconsts, body_jaxpr):
  orig_batched = [d is not batching.not_mapped for d in dims]
  cconst_bat, bconst_bat, init_bat = split_list(orig_batched, [cond_nconsts, body_nconsts])
  cconsts, bconsts, init = split_list(args, [cond_nconsts, body_nconsts])
  cconst_dims, bconst_dims, init_dims = split_list(dims, [cond_nconsts, body_nconsts])

  carry_bat = init_bat
  # Fixpoint computation of which carry are batched: either
  # batched from init, or the carry out is batched. Each iteration promotes
  # at least one carry to batched. We need at most len(carry) iterations to
  # reach a fixpoint.
  for _ in range(1 + len(carry_bat)):
    _, carry_bat_out = batching.batch_jaxpr(
        body_jaxpr, axis_size, bconst_bat + carry_bat, instantiate=carry_bat,
        axis_name=axis_name, main_type=main_type)
    if carry_bat == carry_bat_out:
      break
    carry_bat = safe_map(operator.or_, carry_bat, carry_bat_out)
  else:
    assert False, "Fixpoint not reached"

  # Knowing how the carry is batched now, we can determine if the predicate is
  # batched.
  _, (pred_bat,) = batching.batch_jaxpr(
      cond_jaxpr, axis_size, cconst_bat + carry_bat, instantiate=False,
      axis_name=axis_name, main_type=main_type)

  if pred_bat:
    # If the predicate is batched, we have to batch *all* of the carry
    # regardless of if the body needs it.
    carry_bat = [True] * len(carry_bat)
    carry_dims = [0] * len(carry_bat)
    body_jaxpr_batched, _ = batching.batch_jaxpr_axes(
        body_jaxpr, axis_size, bconst_dims + carry_dims,
        carry_dims, axis_name=axis_name, main_type=main_type)
    cond_jaxpr_batched, _ = batching.batch_jaxpr_axes(
        cond_jaxpr, axis_size, cconst_dims + carry_dims, [0],
        axis_name=axis_name, main_type=main_type)
  else:
    # If the predicate is not batched, we can look at the `cond_jaxpr`'s out
    # shape to determine the rank of the predicate. From this rank we pick the
    # dims of the carry to be batched to ensure that the predicate shape is a
    # prefix of the carry in and out shapes. We can then batch the `body_jaxpr`
    # according to these new batch dims.
    cond_rank = len(cond_jaxpr.out_avals[0].shape)
    carry_dims = [cond_rank if b else None for b in carry_bat]
    body_jaxpr_batched, _ = batching.batch_jaxpr_axes(
        body_jaxpr, axis_size, bconst_dims + carry_dims, carry_dims,
        axis_name=axis_name, main_type=main_type)
    # Now we need to rebatch the `cond_jaxpr` according to the new dims of the
    # carry.
    cond_jaxpr_batched, _ = batching.batch_jaxpr_axes(
        cond_jaxpr, axis_size, cconst_dims + carry_dims, (None,),
        axis_name=axis_name, main_type=main_type)

  # To prepare the `init` to the `while_p`, we broadcast values if they are
  # unbatched and need to have an out axis. If their current batch axis does not
  # match the one it needs to be for the translation rule to work, we move it
  # into place.
  new_init = []
  for x, old_axis, new_axis in zip(init, init_dims, carry_dims):
    if old_axis is batching.not_mapped and new_axis is not batching.not_mapped:
      new_init.append(batching.broadcast(x, axis_size, new_axis))
    elif old_axis is batching.not_mapped and new_axis is batching.not_mapped:
      new_init.append(x)
    else:
      assert new_axis is not batching.not_mapped
      new_init.append(batching.moveaxis(x, old_axis, new_axis))

  outs = while_p.bind(*(cconsts + bconsts + new_init),
                      cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr_batched,
                      body_nconsts=body_nconsts, body_jaxpr=body_jaxpr_batched)
  return outs, carry_dims

def _while_loop_jvp(primals, tangents, cond_nconsts, cond_jaxpr, body_nconsts,
                    body_jaxpr):
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  cconst_nz, bconst_nz, init_nz = split_list(nonzeros, [cond_nconsts, body_nconsts])

  carry_nz = init_nz
  for _ in range(1 + len(carry_nz)):
    body_nonzeros = bconst_nz + carry_nz
    body_jvp, nonzeros_out = ad.jvp_jaxpr(
        body_jaxpr, body_nonzeros, instantiate=carry_nz)
    if nonzeros_out == carry_nz:
      break
    carry_nz = _map(operator.or_, carry_nz, nonzeros_out)
  else:
    assert False, "Fixpoint not reached"

  nonzeros = cconst_nz + body_nonzeros
  tangents = [ad.instantiate_zeros(t) if nz else t
              for t, nz in zip(tangents, nonzeros)]

  cconst, bconst, init = split_list(primals, [cond_nconsts, body_nconsts])
  _, bconst_dot, init_dot = split_list(tangents, [cond_nconsts, body_nconsts])
  bconst_dot = _prune_zeros(bconst_dot)
  init_dot = _prune_zeros(init_dot)

  num_carry = len(primals) - cond_nconsts - body_nconsts

  body_jvp_rearranged = ad.rearrange_binders(
      body_jvp,
      [body_nconsts, num_carry], [len(bconst_dot), len(init_dot)],
      [num_carry], [len(init_dot)])

  newvar = core.gensym([cond_jaxpr.jaxpr])
  invars_aug = (
      cond_jaxpr.jaxpr.invars + [newvar(core.get_aval(x)) for x in init_dot])
  cond_jaxpr_augmented = core.Jaxpr(cond_jaxpr.jaxpr.constvars,
                                    invars_aug,
                                    cond_jaxpr.jaxpr.outvars,
                                    cond_jaxpr.jaxpr.eqns)
  cond_jaxpr_augmented = core.ClosedJaxpr(cond_jaxpr_augmented, cond_jaxpr.consts)

  out = while_p.bind(
      *(cconst + bconst + bconst_dot + init + init_dot),
      cond_nconsts=cond_nconsts,
      cond_jaxpr=cond_jaxpr_augmented,
      body_nconsts=len(bconst) + len(bconst_dot),
      body_jaxpr=body_jvp_rearranged)

  out_carry, out_carry_dot = split_list(out, [num_carry])
  out_tangents_iter = iter(out_carry_dot)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_carry, nonzeros_out)]
  return out_carry, out_tangents

def _while_partial_eval(trace: pe.JaxprTrace, *tracers: pe.Tracer, cond_nconsts: int,
                        cond_jaxpr: pe.ClosedJaxpr, body_nconsts: int,
                        body_jaxpr: pe.ClosedJaxpr) -> Sequence[pe.Tracer]:
  """An implementation of partial evaluation for while.
  As long as some carry (and hence output) are known and the output
  of `cond_jaxpr` is known, we use a portion of the loop body to compute the known
  outputs of the `while_loop`. For the unknown outputs we generate Jaxpr to run
  the whole while, including recomputing the known parts.

  This means that we don't actually save any computation by partial
  evaluation if there are unknown outputs.

  What this achieves is that we can give a proper error for reverse
  differentiation of `while`, because in that use of partial evaluation the
  primal inputs are considered "known", and only the tangent computation is
  unknown (see issue #2129).
  """
  unknowns = [not t.pval.is_known() for t in tracers]
  params = dict(cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr,
                body_nconsts=body_nconsts, body_jaxpr=body_jaxpr)

  cond_consts_uk, body_consts_uk, carry_init_uk = split_list(unknowns, [cond_nconsts, body_nconsts])
  # Fixpoint computation of unknown carry. Each iteration promotes
  # at least one carry to unknown. We need one last iteration to prepare the jaxpr.
  carry_uk = carry_init_uk
  for _ in range(1 + len(carry_uk)):
    body_jaxpr_known, _, carry_out_uk = pe.partial_eval_jaxpr(  # type: ignore
        body_jaxpr, body_consts_uk + carry_uk, instantiate=carry_uk)
    if carry_out_uk == carry_uk:
      break
    else:
      carry_uk = _map(operator.or_, carry_uk, carry_out_uk)
  else:
    assert False, "Fixpoint not reached"

  cond_jaxpr_known, _, cond_uk = pe.partial_eval_jaxpr(  # type: ignore
      cond_jaxpr, cond_consts_uk + carry_uk, instantiate=False)

  if cond_uk[0] or all([not uk for uk in unknowns]) or all(unknowns):
    # If conditional is unknown, or all inputs are known, or all are unknown,
    # just do the default processing.
    return trace.default_process_primitive(while_p, tracers, params)

  # Run the known part of the while. Prepare the inputs, as constants (if known), or
  # as core.unit.
  in_consts = [ core.unit if uk else t.pval.get_known()
                for uk, t in zip(cond_consts_uk + body_consts_uk + carry_uk,
                                 tracers)]
  # There should be no residuals for the cond_jaxpr_known
  assert 1 == len(cond_jaxpr_known.out_avals)
  # We ignore the residuals from the body_jaxpr_known, so the type of inputs matches
  # the type of outputs; residuals are at the end
  if len(body_jaxpr_known.out_avals) > len(body_jaxpr.out_avals):
    # TODO(necula): this is not quite enough; we should drop the residual computations also
    body_jaxpr_known.jaxpr.outvars = body_jaxpr_known.jaxpr.outvars[:len(body_jaxpr.out_avals)]
  out_known = while_p.bind(
    *in_consts,
    cond_nconsts=cond_nconsts,
    cond_jaxpr=cond_jaxpr_known,
    body_nconsts=body_nconsts,
    body_jaxpr=body_jaxpr_known)

  # Run the whole while_loop to get all the outputs, then merge with known ones
  out_all: Sequence[pe.Tracer] = trace.default_process_primitive(while_p, tracers, params)
  out_tracers: Sequence[pe.Tracer] = [
    out_unknown if uk
    else pe.JaxprTracer(trace, pe.PartialVal.known(known), out_unknown.recipe)
    for uk, out_unknown, known in zip(carry_uk, out_all, out_known)]

  return out_tracers

def _while_transpose_error(*_, **kwargs):
  raise ValueError("Reverse-mode differentiation does not work for "
                   "lax.while_loop or lax.fori_loop. "
                   "Try using lax.scan instead.")

while_p = core.AxisPrimitive('while')
while_p.multiple_results = True
while_p.def_impl(partial(xla.apply_primitive, while_p))
while_p.def_abstract_eval(_while_loop_abstract_eval)
ad.primitive_jvps[while_p] = _while_loop_jvp
pe.custom_partial_eval_rules[while_p] = _while_partial_eval
xla.register_translation(while_p, _while_loop_translation_rule,
                         initial_style=True)
ad.primitive_transposes[while_p] = _while_transpose_error
batching.axis_primitive_batchers[while_p] = _while_loop_batching_rule
pe.partial_eval_jaxpr_custom_rules[while_p] = \
    partial(pe.partial_eval_jaxpr_custom_rule_not_implemented, 'while_loop')


def _pred_bcast_select_mhlo(
    pred_aval: core.ShapedArray, pred: ir.Value, xs: Sequence[ir.Value],
    ys: Sequence[ir.Value], x_y_aval: core.AbstractValue) -> Sequence[ir.Value]:
  if x_y_aval is core.abstract_unit:
    return []
  elif x_y_aval is core.abstract_token:
    x, = xs
    y, = ys
    return [mhlo.AfterAllOp(mlir.aval_to_ir_type(x_y_aval), [x, y]).result]
  else:
    assert isinstance(x_y_aval, core.ShapedArray), x_y_aval
    x, = xs
    y, = ys
    assert x.type == y.type, (x.type, y.type)
    assert (pred_aval.shape == x_y_aval.shape[:len(pred_aval.shape)]), (
            pred_aval.shape, x_y_aval)
    bcast_pred = mhlo.BroadcastInDimOp(
        mlir.aval_to_ir_type(x_y_aval.update(dtype=np.dtype(np.bool_))),
        pred, mlir.dense_int_elements(list(range(len(pred_aval.shape))))).result
    return mhlo.SelectOp(bcast_pred, x, y).results


def _while_lowering(ctx, *args, cond_jaxpr, body_jaxpr, cond_nconsts,
                    body_nconsts):
  pred_aval = cond_jaxpr.out_avals[0]
  batched = bool(pred_aval.shape)

  loop_carry_types = _map(mlir.aval_to_ir_types, ctx.avals_in)
  flat_loop_carry_types = util.flatten(loop_carry_types)

  flat_args = mlir.flatten_lowering_ir_args(args)
  while_op = mhlo.WhileOp(flat_loop_carry_types, flat_args)

  # Loop condition
  cond_block = while_op.regions[0].blocks.append(*flat_loop_carry_types)
  with ir.InsertionPoint(cond_block):
    flat_cond_args = [
        cond_block.arguments[i] for i in range(len(flat_loop_carry_types))
    ]
    cond_args = util.unflatten(flat_cond_args, _map(len, loop_carry_types))
    x, _, z = util.split_list(cond_args, [cond_nconsts, body_nconsts])
    cond_ctx = ctx.module_context.replace(
        name_stack=xla.extend_name_stack(ctx.module_context.name_stack,
                                         'cond'))
    (pred,), = mlir.jaxpr_subcomp(cond_ctx, cond_jaxpr.jaxpr,
                                  _map(mlir.ir_constants, cond_jaxpr.consts),
                                  *(x + z))
    if batched:
      pred_ctx = mlir.LoweringRuleContext(
          module_context=ctx.module_context,
          primitive=None,
          avals_in=[pred_aval],
          avals_out=[pred_aval.update(shape=())])
      pred, = lax._unary_reduce_lower(
          mhlo.OrOp,
          lambda dtype: np.array(False, dtype),
          pred_ctx,
          pred,
          axes=tuple(range(len(pred_aval.shape))))
    mhlo.ReturnOp([pred])

  # Loop body
  body_block = while_op.regions[1].blocks.append(*flat_loop_carry_types)
  with ir.InsertionPoint(body_block):
    flat_body_args = [
        body_block.arguments[i] for i in range(len(flat_loop_carry_types))
    ]
    body_args = util.unflatten(flat_body_args, _map(len, loop_carry_types))
    x, y, z = util.split_list(body_args, [cond_nconsts, body_nconsts])
    body_ctx = ctx.module_context.replace(
        name_stack=xla.extend_name_stack(ctx.module_context.name_stack,
                                         'body'))
    new_z = mlir.jaxpr_subcomp(body_ctx, body_jaxpr.jaxpr,
                               _map(mlir.ir_constants, body_jaxpr.consts),
                               *(y + z))
    if batched:
      body_pred_ctx = ctx.module_context.replace(
          name_stack=xla.extend_name_stack(ctx.module_context.name_stack,
                                           'body_pred'))
      (body_pred,), = mlir.jaxpr_subcomp(
          body_pred_ctx, cond_jaxpr.jaxpr,
          _map(mlir.ir_constants, cond_jaxpr.consts), *(x + z))
      new_z = _map(
          partial(_pred_bcast_select_mhlo, pred_aval, body_pred), new_z, z,
          body_jaxpr.out_avals)

    mhlo.ReturnOp([*util.flatten(x), *util.flatten(y), *util.flatten(new_z)])

  outputs = util.unflatten(while_op.results, _map(len, loop_carry_types))
  _, _, z = util.split_list(outputs, [cond_nconsts, body_nconsts])
  return z

mlir.register_lowering(while_p, _while_lowering)


### cond and switch


# For backward compatibility with a previous switch/cond calling convention,
# we allow a single (pytree) `operand` argument to be passed by keyword. We use
# a sentinel object as its default value to indicate when it is _not_ passed.
_no_operand_sentinel = object()


@api_boundary
def switch(index, branches: Sequence[Callable], *operands,
           operand=_no_operand_sentinel):
  """Apply exactly one of ``branches`` given by ``index``.

  If ``index`` is out of bounds, it is clamped to within bounds.

  Has the semantics of the following Python::

    def switch(index, branches, operand):
      index = clamp(0, index, len(branches) - 1)
      return branches[index](operand)

  Args:
    index: Integer scalar type, indicating which branch function to apply.
    branches: Sequence of functions (A -> B) to be applied based on ``index``.
    operands: Operands (A) input to whichever branch is applied.

  Returns:
    Value (B) of ``branch(*operands)`` for the branch that was selected based
    on ``index``.
  """
  if operand is not _no_operand_sentinel:
    if operands:
      raise TypeError("if 'operand' keyword is passed then no positional "
                      f"operands can be passed, got operand={operand} "
                      f"and positional operands {operands}")
    operands = (operand,)
  del operand

  if len(np.shape(index)) != 0:
    raise TypeError(
        f"Branch index must be scalar, "
        f"got {index} of shape {np.shape(index)}.")

  try:
    index_dtype = dtypes.result_type(index)
  except TypeError as err:
    msg = f"Index type must be an integer, got {index}."
    raise TypeError(msg) from err

  if index_dtype.kind not in 'iu':
    raise TypeError(
        f"Index type must be an integer, got {index} as {index_dtype}")

  branches = tuple(branches)

  if len(branches) == 0:
    raise ValueError("Empty branch sequence")
  elif len(branches) == 1:
    return branches[0](*operands)

  index = lax.convert_element_type(index, np.int32)
  lo = np.array(0, np.int32)
  hi = np.array(len(branches) - 1, np.int32)
  index = lax.clamp(lo, index, hi)

  if (config.jax_disable_jit and
      isinstance(core.get_aval(index), ConcreteArray)):
    return branches[int(index)](*operands)

  ops, ops_tree = tree_flatten(operands)
  ops_avals = tuple(_map(_abstractify, ops))

  jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
      WeakRefSequence(branches), ops_tree, ops_avals, primitive_name='switch')

  for i, (out_tree, jaxpr) in enumerate(zip(out_trees[1:], jaxprs[1:])):
    _check_tree_and_avals(f"branch 0 and {i + 1} outputs",
                          out_trees[0], jaxprs[0].out_avals,
                          out_tree, jaxpr.out_avals)

  linear = (False,) * (len(consts) + len(ops))
  out = cond_p.bind(
      index, *consts, *ops, branches=tuple(jaxprs), linear=linear)
  return tree_unflatten(out_trees[0], out)


def _cond(pred, true_fun: Callable, false_fun: Callable, *operands,
          operand=_no_operand_sentinel, linear=None):
  """Conditionally apply ``true_fun`` or ``false_fun``.

  ``cond()`` has equivalent semantics to this Python implementation::

    def cond(pred, true_fun, false_fun, *operands):
      if pred:
        return true_fun(*operands)
      else:
        return false_fun(*operands)

  ``pred`` must be a scalar type.

  Args:
    pred: Boolean scalar type, indicating which branch function to apply.
    true_fun: Function (A -> B), to be applied if ``pred`` is True.
    false_fun: Function (A -> B), to be applied if ``pred`` is False.
    operands: Operands (A) input to either branch depending on ``pred``. The
      type can be a scalar, array, or any pytree (nested Python tuple/list/dict)
      thereof.

  Returns:
    Value (B) of either ``true_fun(*operands)`` or ``false_fun(*operands)``,
    depending on the value of ``pred``. The type can be a scalar, array, or any
    pytree (nested Python tuple/list/dict) thereof.
  """
  if operand is not _no_operand_sentinel:
    if operands:
      raise TypeError("if 'operand' keyword is passed then no positional "
                      f"operands can be passed, got operand={operand} "
                      f"and positional operands {operands}")
    operands = (operand,)
  del operand

  if isinstance(pred, Sequence) or np.ndim(pred) != 0:
    raise TypeError(
        f"Pred must be a scalar, got {pred} of " +
        (f"type {type(pred)}" if isinstance(pred, Sequence)
         else f"shape {np.shape(pred)}."))

  try:
    pred_dtype = dtypes.result_type(pred)
  except TypeError as err:
    msg = ("Pred type must be either boolean or number, got {}.")
    raise TypeError(msg.format(pred)) from err

  if pred_dtype.kind != 'b':
    if pred_dtype.kind in 'iuf':
      pred = pred != 0
    else:
      msg = ("Pred type must be either boolean or number, got {}.")
      raise TypeError(msg.format(pred_dtype))

  if config.jax_disable_jit and isinstance(core.get_aval(pred), ConcreteArray):
    if pred:
      return true_fun(*operands)
    else:
      return false_fun(*operands)

  ops, ops_tree = tree_flatten(operands)
  if linear is None:
    linear_ops = [False] * len(ops)
  else:
    linear_ops, ops_tree2 = tree_flatten(linear)
    if ops_tree != ops_tree2:
      raise TypeError('linear tree and operand tree mismatch')
  ops_avals = tuple(_map(_abstractify, ops))

  jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
      WeakRefSequence((true_fun, false_fun)), ops_tree, ops_avals, 'cond')
  true_jaxpr, false_jaxpr = jaxprs
  out_tree, false_out_tree = out_trees

  _check_tree_and_avals("true_fun and false_fun output",
                        out_tree, true_jaxpr.out_avals,
                        false_out_tree, false_jaxpr.out_avals)

  index = lax.convert_element_type(pred, np.int32)

  linear = [False] * len(consts) + linear_ops
  out = cond_p.bind(
      index, *consts, *ops,
      branches=(false_jaxpr, true_jaxpr), linear=tuple(linear))
  return tree_unflatten(out_tree, out)

@api_boundary
@functools.wraps(_cond)
def cond(*args, **kwargs):
  # detect an attempt to call the former, deprecated cond
  try:
    ba = inspect.signature(_cond_with_per_branch_args).bind(*args, **kwargs)
  except TypeError:
    pass
  else:
    assert not ba.kwargs  # no catch-all **kwargs in _cond_with_per_branch
    _, _, maybe_true_fun, _, maybe_false_fun = ba.args
    if callable(maybe_true_fun) and callable(maybe_false_fun):
      return _cond_with_per_branch_args(*ba.args)

  return _cond(*args, **kwargs)

def _cond_with_per_branch_args(pred,
                               true_operand, true_fun: Callable,
                               false_operand, false_fun: Callable):
  """Conditionally apply ``true_fun`` or ``false_fun``.

  Has equivalent semantics to this Python implementation::

    def cond(pred, true_operand, true_fun, false_operand, false_fun):
      if pred:
        return true_fun(true_operand)
      else:
        return false_fun(false_operand)

  Pred has to be a scalar type, collection types (list, tuple) are not supported
  """
  return _cond(pred,
               lambda op: true_fun(op[0]),
               lambda op: false_fun(op[1]),
               (true_operand, false_operand))

def _cond_abstract_eval(*args, **kwargs):
  return _map(raise_to_shaped, kwargs["branches"][0].out_avals)

def _cond_translation_rule(ctx, avals_in, avals_out, index, *args, branches,
                           linear):
  del linear  # Unused.

  name_stack = extend_name_stack(ctx.name_stack, "cond")
  def make_computation(name, jaxpr, op_shape):
    c = xla_client.XlaBuilder(name + '_comp')
    op = xla.parameter(c, 0, op_shape)
    ops = [xops.GetTupleElement(op, i) for i in range(len(jaxpr.in_avals))]
    subctx = ctx.replace(
        builder=c, name_stack=extend_name_stack(name_stack, name + '_fun'))
    outs = xla.jaxpr_subcomp(
        subctx, jaxpr.jaxpr,
        _map(partial(xla.pyval_to_ir_constant, c), jaxpr.consts), *ops)
    return c.build(xops.Tuple(c, outs))

  c = ctx.builder
  op = xops.Tuple(c, args)
  op_shape = c.get_shape(op)
  branch_computations = [
      make_computation(f'branch_{i}', jaxpr, op_shape)
      for i, jaxpr in enumerate(branches)]
  return xla.xla_destructure(
      c, xops.Conditional(index, branch_computations, [op] * len(branches)))


def _bcast_select(pred, on_true, on_false):
  if np.ndim(pred) != np.ndim(on_true):
    idx = list(range(np.ndim(pred)))
    pred = lax.broadcast_in_dim(pred, np.shape(on_true), idx)
  return lax.select(pred, on_true, on_false)

def _bcast_select_n(pred, *cases):
  if np.ndim(pred) != np.ndim(cases[0]):
    idx = list(range(np.ndim(pred)))
    pred = lax.broadcast_in_dim(pred, np.shape(cases[0]), idx)
  return lax.select_n(pred, *cases)

def _cond_batching_rule(axis_size, axis_name, main_type, args, dims, branches, linear):
  index, *ops = args
  index_dim, *op_dims = dims

  if index_dim is not batching.not_mapped:
    # Convert to a lax.select. While we could get away with not broadcasting
    # some operands yet, because all outputs must be broadcast together anyway
    # for the select we broadcast the input operands for simplicity and leave
    # optimizations to XLA.
    # TODO(mattjj,frostig): assumes branches are side-effect-free, revise!
    index, *ops = [
        batching.bdim_at_front(x, d, axis_size) for x, d in zip(args, dims)]

    in_batched  = [a is not core.abstract_unit for a in branches[0].in_avals]
    out_batched = [a is not core.abstract_unit for a in branches[0].out_avals]

    branches_batched = [
        batching.batch_jaxpr(
            jaxpr, axis_size, in_batched, out_batched, axis_name, main_type)[0]
        for jaxpr in branches]

    branch_outs = []
    for i, jaxpr in enumerate(branches_batched):
      # Perform a select on the inputs for safety of reverse-mode autodiff; see
      # https://github.com/google/jax/issues/1052
      predicate = lax.eq(index, lax._const(index, i))
      ops_ = [_bcast_select(predicate, x, lax.stop_gradient(x))
              if x is not core.unit else x for x in ops]
      branch_outs.append(core.jaxpr_as_fun(jaxpr)(*ops_))
    out = [_bcast_select_n(index, *outs) if outs[0] is not core.unit else outs[0]
           for outs in zip(*branch_outs)]
    return out, [0 if b else None for b in out_batched]
  else:
    ops_bat = [d is not batching.not_mapped for d in op_dims]
    ops = [batching.moveaxis(x, d, 0) if b else x
           for b, x, d in zip(ops_bat, ops, op_dims)]

    branches_out_bat = [
        batching.batch_jaxpr(jaxpr, axis_size, ops_bat, False, axis_name, main_type)[1]
        for jaxpr in branches]
    out_bat = [any(bat) for bat in zip(*branches_out_bat)]
    branches_batched = tuple(
        batching.batch_jaxpr(jaxpr, axis_size, ops_bat, out_bat, axis_name, main_type)[0]
        for jaxpr in branches)

    out_dims = [0 if b else batching.not_mapped for b in out_bat]
    out = cond_p.bind(
        index, *ops, branches=branches_batched, linear=linear)
    return out, out_dims

def _cond_jvp(primals, tangents, branches, linear):
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]

  index_nz, *ops_nz = nonzeros
  assert index_nz is False

  branches_out_nz = [ad.jvp_jaxpr(jaxpr, ops_nz, instantiate=False)[1]
                     for jaxpr in branches]
  out_nz = [any(nz) for nz in zip(*branches_out_nz)]

  branches_jvp = tuple(ad.jvp_jaxpr(jaxpr, ops_nz, instantiate=out_nz)[0]
                       for jaxpr in branches)

  index, *ops = primals
  _, *ops_dot = tangents
  ops_dot = _prune_zeros(ops_dot)

  ops_lin = tuple(linear)
  linear_jvp = ops_lin + (True,) * len(ops_dot)
  out = cond_p.bind(
      index, *ops, *ops_dot, branches=branches_jvp, linear=linear_jvp)
  out_primals, out_tangents = split_list(out, [len(out_nz)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_primals, out_nz)]
  return out_primals, out_tangents

def _cond_partial_eval(trace, *tracers, branches, linear):
  unknowns = [t.pval[0] is not None for t in tracers]
  index_uk, *ops_uk = unknowns

  if index_uk:
    # When the branch index is unknown, we stage out the whole cond.
    params = dict(branches=branches, linear=linear)
    return trace.default_process_primitive(cond_p, tracers, params)

  branches_out_uks = []
  for branch_jaxpr in branches:
    _, _, out_uks = pe.partial_eval_jaxpr(branch_jaxpr, ops_uk, instantiate=False)
    branches_out_uks.append(out_uks)
  out_uks = [any(uks) for uks in zip(*branches_out_uks)]

  branches_1, branches_2, branch_res_avals = [], [], []
  for branch_jaxpr in branches:
    branch_jaxpr_1, branch_jaxpr_2, _ = pe.partial_eval_jaxpr(
        branch_jaxpr, ops_uk, instantiate=out_uks)
    branch_num_res = len(branch_jaxpr_1.out_avals) - len(out_uks)

    # move residuals to the front
    move = [False] * len(ops_uk) + [True] * branch_num_res
    branch_jaxpr_2 = pe.move_binders_to_front(branch_jaxpr_2, move)

    # TODO(frostig,mattjj): pe.partial_eval_jaxpr should raise to shaped avals
    res_avals = _map(
        raise_to_shaped, branch_jaxpr_2.in_avals[:branch_num_res])

    branches_1.append(branch_jaxpr_1)
    branches_2.append(branch_jaxpr_2)
    branch_res_avals.append(res_avals)

  branches_1 = tuple(branches_1)
  branches_2 = tuple(branches_2)

  for jaxpr in branches_2[1:]:
    assert len(jaxpr.out_avals) == len(branches_2[0].out_avals)

  num_outs = len(branches_2[0].out_avals)

  all_res_avals, res_avals_per_branch = _merge_branch_residuals(
      branch_res_avals)

  branches_1 = _join_cond_outputs(
      branches_1, all_res_avals, res_avals_per_branch, num_outs)
  branches_2 = _join_cond_pe_staged_jaxpr_inputs(
      branches_2, all_res_avals, res_avals_per_branch)

  # TODO(frostig,mattjj): reinstate this assertion once pe.partial_eval_jaxpr
  # raises to shaped avals
  # for j in branches_1[1:]:
  #   assert j.out_avals == branches_1[0].out_avals
  num_res = len(all_res_avals)

  _, in_consts = unzip2([t.pval for t in tracers])
  out_consts_res = cond_p.bind(*in_consts, branches=branches_1, linear=linear)
  out_consts, res = split_list(out_consts_res, [len(out_consts_res) - num_res])

  # TODO(frostig,mattjj): remove raised_to_shaped of avals once
  # pe.partial_eval_jaxpr handles it
  out_avals = _map(raise_to_shaped, branches_2[0].out_avals)
  out_pvs = [aval if uk else None for aval, uk in zip(out_avals, out_uks)]

  index_tracer = trace.instantiate_const(tracers[0])

  ops_tracers = [trace.instantiate_const(t) if uk
                 else trace.new_instantiated_literal(core.unit)
                 for uk, t in zip(unknowns[1:], tracers[1:])]

  res_tracers = _map(trace.new_instantiated_const, res)

  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal((pv, const)), None)
                 for pv, const in zip(out_pvs, out_consts)]

  linear_2 = (False,) * num_res + linear
  params = dict(branches=branches_2, linear=linear_2)
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)
  eqn = pe.new_eqn_recipe(
      [index_tracer] + res_tracers + ops_tracers, out_tracers, cond_p, params,
      source)
  for t in out_tracers: t.recipe = eqn
  return out_tracers

# When partially evaluating conditionals, each branch produces residuals
# depending on the computation carried out by the branch, and a corresponding
# staged jaxpr that accepts those residuals as its first few inputs. The
# residual-producing branches are staged as jaxprs and bound right away in a
# conditional. The residual-consuming jaxprs are assembled together in a jaxpr
# conditional. The following helper functions ensure that both collections of
# jaxprs (those evaluated and those staged) are valid for joint use under their
# respective conditionals.
#
# In particular, the residuals derived from each original branch may have
# distinct types. Because the branches of conditionals must have identical type
# signatures, we join residuals together across branches into a common format.

# In order to set up a type signature that all branches can conform to, it would
# suffice to concatenate all branches' residuals. But concatenation can result
# in redundant inputs and outputs, and might lead to memory allocation that
# scales unnecessarily with the branch count. This function finds common
# residual types across branches for reuse, so as to avoid redundant
# allocation. It returns a list L of types (avals) representing the collection
# of residuals merged according to type, and, for each branch, a lookup table to
# match its residuals to their positions/types in L. Example input/output:
#
# [x], [y], [x, x]             -> [x, y, x],    [[0], [1], [0, 2]]
# [x], [x], [x, x]             -> [x, x],       [[0], [0], [0, 1]]
# [y, x, x], [x, z, y], [z, x] -> [y, x, x, z], [[0, 1, 2], [1, 3, 0], [3, 1]]
def _merge_branch_residuals(branch_res_avals):
  def enumerate_equal(xs):
    counts = {v: itertools.count() for v in set(xs)}
    return [(x, next(counts[x])) for x in xs]
  branch_res_tagged_avals = _map(enumerate_equal, branch_res_avals)
  all_tagged_avals = _ordered_unique(util.concatenate(branch_res_tagged_avals))
  indices = {v: i for i, v in enumerate(all_tagged_avals)}
  branch_indices = [
      [indices[aval] for aval in avals] for avals in branch_res_tagged_avals]
  all_avals = [x for x, _ in all_tagged_avals]
  return all_avals, branch_indices

# This function augments branch outputs to agree with the merged residual
# format: each branch is made to return zero-filled values in the places of
# residual outputs that it does not populate.
def _join_cond_outputs(jaxprs, all_res_avals, res_aval_indices_per_jaxpr,
                       num_non_res_outputs):
  def augment_jaxpr(jaxpr, res_indices):
    @lu.wrap_init
    def f_aug(*args):
      outs_and_residuals = core.jaxpr_as_fun(jaxpr)(*args)
      outs, residuals = split_list(outs_and_residuals, [num_non_res_outputs])
      aug_residuals = _map(ad_util.zeros_like_aval, all_res_avals)
      aug_residuals = util.subvals(aug_residuals, zip(res_indices, residuals))
      return outs + list(aug_residuals)

    return _make_closed_jaxpr(f_aug, jaxpr.in_avals)

  return tuple(_map(augment_jaxpr, jaxprs, res_aval_indices_per_jaxpr))

# This function augments branch inputs to agree with the merged residual format:
# each branch is made to accept all residuals, even though it will ignore those
# that it does not read.
def _join_cond_pe_staged_jaxpr_inputs(jaxprs, all_res_avals,
                                      res_aval_indices_per_jaxpr):
  newvar = core.gensym([j.jaxpr for j in jaxprs], suffix='_')
  all_res_vars = _map(newvar, all_res_avals)

  def augment_jaxpr(jaxpr, res_indices):
    num_res = len(res_indices)
    res_vars = jaxpr.jaxpr.invars[:num_res]
    non_res_vars = jaxpr.jaxpr.invars[num_res:]

    aug_res_vars = list(util.subvals(all_res_vars, zip(res_indices, res_vars)))
    aug_invars = aug_res_vars + non_res_vars
    jaxpr_aug = core.Jaxpr(jaxpr.jaxpr.constvars, aug_invars,
                           jaxpr.jaxpr.outvars, jaxpr.jaxpr.eqns)
    jaxpr_aug = core.ClosedJaxpr(jaxpr_aug, jaxpr.consts)
    return jaxpr_aug

  return tuple(_map(augment_jaxpr, jaxprs, res_aval_indices_per_jaxpr))

def _ordered_unique(xs):
  d = collections.OrderedDict((x, None) for x in xs)
  return list(d.keys())

def _transpose_cond_jaxpr(jaxpr, num_res, reduce_axes):
  res_avals, primal_avals = split_list(jaxpr.in_avals, [num_res])
  primal_avals = _map(raise_to_shaped, primal_avals)

  @lu.wrap_init
  def transposed(*args):
    res, cts_out = split_list(args, [num_res])
    primals = res + [ad.UndefinedPrimal(aval) for aval in primal_avals]
    cts_in = ad.backward_pass(
        jaxpr.jaxpr, reduce_axes, False, jaxpr.consts, primals, cts_out)
    _, cts_in = split_list(cts_in, [num_res])
    return _map(ad.instantiate_zeros_aval, primal_avals, cts_in)

  return _make_closed_jaxpr(transposed, res_avals + jaxpr.out_avals)

def _cond_transpose(reduce_axes, cts, *args, branches, linear):
  index, *ops = args
  in_avals = _map(raise_to_shaped, branches[0].in_avals)
  num_res = len(ops) - sum(linear)

  branches_trans = tuple(
      _transpose_cond_jaxpr(jaxpr, num_res, reduce_axes) for jaxpr in branches)
  lin_in_avals = [raise_to_shaped(a, weak_type=False)
                  for a, l in zip(in_avals, linear) if l]
  assert all(core.typematch(out_aval, lin_in_aval)
             for jaxpr in branches_trans
             for out_aval, lin_in_aval in zip(jaxpr.out_avals, lin_in_avals))

  res = ops[:num_res]
  cts = _map(ad.instantiate_zeros_aval, branches[0].out_avals, cts)
  linear_trans = (False,) * num_res + (True,) * len(cts)

  out = cond_p.bind(
      index, *res, *cts, branches=branches_trans, linear=linear_trans)
  assert all(_map(core.typecheck, lin_in_avals, out))

  out_iter = iter(out)
  out = [next(out_iter) if l else None for l in linear]
  assert next(out_iter, None) is None
  return [None] + out

def _avals_short(avals):
  to_str = lambda aval: getattr(aval, 'str_short', partial(str, aval))()
  return ' '.join(_map(to_str, avals))

def _cond_typecheck(*avals, branches, linear):
  tc = partial(_typecheck_param, 'cond')
  tc(branches, 'branches', 'tuple of ClosedJaxpr',
     type(branches) is tuple and
     all(type(x) is core.ClosedJaxpr for x in branches))
  tc(linear, 'linear', 'tuple of bool',
     type(linear) is tuple and all(type(x) is bool for x in linear))

  if len(branches) == 0:
    raise core.JaxprTypeError('cond requires at least one branch function')
  if len(linear) + 1 != len(avals):
    raise core.JaxprTypeError(f'cond given {len(linear)} linear flags for '
                              f'{len(avals) - 1} non-predicate operands')

  jaxpr0 = branches[0]
  jaxpr0_in_avals_str = _avals_short(jaxpr0.in_avals)
  jaxpr0_out_avals_str = _avals_short(jaxpr0.out_avals)

  for i, jaxpr in enumerate(branches[1:]):
    if len(jaxpr0.in_avals) != len(jaxpr.in_avals):
      raise core.JaxprTypeError(
        f'cond branch 0 takes {len(jaxpr0.in_avals)} inputs, '
        f'branch {i+1} takes {len(jaxpr.in_avals)}')
    if len(jaxpr0.out_avals) != len(jaxpr.out_avals):
      raise core.JaxprTypeError(
        f'cond branch 0 outputs {len(jaxpr0.out_avals)} values, '
        f'branch {i+1} outputs {len(jaxpr.out_avals)}')
    if not all(_map(core.typematch, jaxpr0.in_avals, jaxpr.in_avals)):
      raise core.JaxprTypeError(
        f'cond branches 0 and {i+1} have mismatching input types: '
        f'{jaxpr0_in_avals_str} vs {_avals_short(jaxpr.in_avals)}')
    if not all(_map(core.typematch, jaxpr0.out_avals, jaxpr.out_avals)):
      raise core.JaxprTypeError(
        f'cond branches 0 and {i+1} have mismatching output types: '
        f'{jaxpr0_out_avals_str} vs {_avals_short(jaxpr.out_avals)}')

  if len(avals) != 1 + len(jaxpr0.in_avals):
    raise core.JaxprTypeError(
      f'cond called with {len(avals) - 1} non-predicate operands, '
      f'but branches take {len(jaxpr0.in_avals)} inputs')

  index_aval, *op_avals = avals
  if index_aval.dtype != np.int32:
    raise core.JaxprTypeError(
      f'cond called with index of type {index_aval.dtype} instead of int32')
  if not all(_map(core.typecompat, jaxpr0.in_avals, op_avals)):
    raise core.JaxprTypeError(
      f'cond branches take input types {jaxpr0_in_avals_str}, '
      f'called with operands of type {_avals_short(op_avals)}')

def cond_bind(*args, branches, linear):
  if config.jax_enable_checks:
    avals = _map(core.get_aval, args)
    _cond_typecheck(*avals, branches=branches, linear=linear)
    for jaxpr in branches:
      core.check_jaxpr(jaxpr.jaxpr)
  return core.AxisPrimitive.bind(cond_p, *args, branches=branches, linear=linear)

cond_p = core.AxisPrimitive('cond')
cond_p.multiple_results = True
cond_p.def_impl(partial(xla.apply_primitive, cond_p))
cond_p.def_abstract_eval(_cond_abstract_eval)
cond_p.def_custom_bind(cond_bind)
ad.primitive_jvps[cond_p] = _cond_jvp
ad.reducing_transposes[cond_p] = _cond_transpose
pe.custom_partial_eval_rules[cond_p] = _cond_partial_eval
batching.axis_primitive_batchers[cond_p] = _cond_batching_rule
xla.register_translation(cond_p, _cond_translation_rule, initial_style=True)
core.custom_typechecks[cond_p] = _cond_typecheck
pe.partial_eval_jaxpr_custom_rules[cond_p] = \
    partial(pe.partial_eval_jaxpr_custom_rule_not_implemented, 'cond')

def _cond_lowering(ctx, index, *args, branches, linear):
  del linear  # Unused.
  output_types = _map(mlir.aval_to_ir_types, ctx.avals_out)
  flat_output_types = util.flatten(output_types)

  # mhlo.CaseOp takes a single argument 'index' and the corresponding blocks
  # have no arguments; the computation within the block uses implicit
  # captures.

  # TODO(phawkins): avoid build_generic when CaseOp is fixed.
  case_op = mhlo.CaseOp.build_generic(
      flat_output_types, [index], regions=len(branches))
  for i, jaxpr in enumerate(branches):
    branch = case_op.regions[i].blocks.append()
    with ir.InsertionPoint(branch):
      out_vals = mlir.jaxpr_subcomp(
          ctx.module_context, jaxpr.jaxpr,
          _map(mlir.ir_constants, jaxpr.consts),
          *_map(mlir.wrap_singleton_ir_values, args))
      mhlo.ReturnOp(util.flatten(out_vals))

  return util.unflatten(case_op.results, _map(len, output_types))

mlir.register_lowering(cond_p, _cond_lowering)



### scan

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

@api_boundary
def scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: Optional[int] = None,
         reverse: bool = False,
         unroll: int = 1) -> Tuple[Carry, Y]:
  """Scan a function over leading array axes while carrying along state.

  The type signature in brief is

  .. code-block:: haskell

    scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

  where we use [t] here to denote the type t with an additional leading axis.
  That is, if t is an array type then [t] represents the type with an additional
  leading axis, and if t is a pytree (container) type with array leaves then [t]
  represents the type with the same pytree structure and corresponding leaves
  each with an additional leading axis.

  When ``a`` is an array type or None, and ``b`` is an array type, the semantics
  of ``scan`` are given roughly by this Python implementation::

    def scan(f, init, xs, length=None):
      if xs is None:
        xs = [None] * length
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, np.stack(ys)

  Unlike that Python version, both ``a`` and ``b`` may be arbitrary pytree
  types, and so multiple arrays can be scanned over at once and produce multiple
  output arrays. (None is actually an empty pytree.)

  Also unlike that Python version, ``scan`` is a JAX primitive and is lowered to
  a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Finally, the loop-carried value ``carry`` must hold a fixed shape and dtype
  across all iterations (and not just be consistent up to NumPy rank/shape
  broadcasting and dtype promotion rules, for example). In other words, the type
  ``c`` in the type signature above represents an array with a fixed shape and
  dtype (or a nested tuple/list/dict container data structure with a fixed
  structure and arrays with fixed shape and dtype at the leaves).

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value. This value must have the same structure as
      the first element of the pair returned by ``f``.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    reverse: optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.
    unroll: optional positive int specifying, in the underlying operation of the
      scan primitive, how many scan iterations to unroll within a single
      iteration of a loop.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.
  """
  xs_flat, xs_tree = tree_flatten(xs)

  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    msg = "scan got value with no leading axis to scan over: {}."
    raise ValueError(
      msg.format(', '.join(str(x) for x in xs_flat
                           if not hasattr(x, 'shape')))) from err

  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      msg = ("scan got `length` argument of {} which disagrees with "
             "leading axis sizes {}.")
      raise ValueError(msg.format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      msg = "scan got no values to scan over and `length` not provided."
      raise ValueError(msg)
    else:
      length, = unique_lengths

  if config.jax_disable_jit:
    if length == 0:
      raise ValueError("zero-length scan is not supported in disable_jit() mode because the output type is unknown.")
    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(length)):
      xs_slice = [_index_array(i, core.get_aval(x), x) for x in xs_flat]
      carry, y = f(carry, tree_unflatten(xs_tree, xs_slice))
      ys.append(y)
    stack = lambda y, *ys: (y if core.get_aval(y) is core.abstract_unit
                            else jax.numpy.stack((y, *ys)))
    stacked_y = tree_map(stack, *maybe_reversed(ys))
    return carry, stacked_y

  x_shapes = [masking.padded_shape_as_value(x.shape[1:]) for x in xs_flat]
  x_dtypes = [dtypes.canonicalize_dtype(x.dtype) for x in xs_flat]
  x_avals = tuple(_map(ShapedArray, x_shapes, x_dtypes))

  def _create_jaxpr(init):
    init_flat, init_tree = tree_flatten(init)
    in_flat, in_tree = tree_flatten((init, xs))

    carry_avals = tuple(_map(_abstractify, init_flat))
    jaxpr, consts, out_tree = _initial_style_jaxpr(
        f, in_tree, carry_avals + x_avals, "scan")
    out_tree_children = out_tree.children()
    if len(out_tree_children) != 2:
      msg = "scan body output must be a pair, got {}."
      raise TypeError(msg.format(tree_unflatten(out_tree, jaxpr.out_avals)))
    carry_avals_out = jaxpr.out_avals[:out_tree_children[0].num_leaves]
    return init_flat, carry_avals, carry_avals_out, init_tree, in_flat, jaxpr, consts, out_tree, out_tree_children

  # The carry input and output avals must match exactly. However, we want to account for
  # the case when init contains weakly-typed values (e.g. Python scalars), with avals that
  # may not match the output despite being compatible by virtue of their weak type.
  # To do this, we compute the jaxpr in two passes: first with the raw inputs, and if
  # necessary, a second time with modified init values.
  init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(init)
  new_init_flat, changed = _promote_weak_typed_inputs(init_flat, carry_avals, carry_avals_out)
  if changed:
    new_init = tree_unflatten(init_tree, new_init_flat)
    init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(new_init)
  in_flat, jaxpr, consts, out_tree, out_tree_children = rest

  _check_tree_and_avals("scan carry output and input",
                        # Extract the subtree and avals for the first element of the return tuple
                        out_tree_children[0], carry_avals_out,
                        init_tree, carry_avals)

  out = scan_p.bind(*consts, *in_flat,
                    reverse=reverse, length=length, jaxpr=jaxpr,
                    num_consts=len(consts), num_carry=len(init_flat),
                    linear=(False,) * (len(consts) + len(in_flat)),
                    unroll=unroll)
  return tree_unflatten(out_tree, out)

def _scan_impl_unrolled(*args, reverse, length, num_consts, num_carry, linear,
                        f_impl, x_avals, y_avals):
  consts, init, xs = split_list(args, [num_consts, num_carry])

  carry = init
  ys = []

  for i in range(length):
    i_ = length - i - 1 if reverse else i
    x = _map(partial(_index_array, i_), x_avals, xs)
    out = f_impl(*consts, *carry, *x)
    carry, y = split_list(out, [num_carry])
    ys.append(y)

  ys = list(reversed(ys)) if reverse else ys
  ys = list(zip(*ys))
  ys = _map(_stack, y_avals, ys)
  return (*carry, *ys)

def _scan_impl_loop(*args, reverse, length, num_consts, num_carry, linear,
                    f_impl, x_avals, y_avals):
  consts, init, xs = split_list(args, [num_consts, num_carry])

  def cond_fun(vals):
    i, *_ = vals
    return i < length

  def body_fun(vals):
    [i], carry, ys = split_list(vals, [1, num_carry])
    i_ = length - i - 1 if reverse else i
    x = _map(partial(_dynamic_index_array, i_), x_avals, xs)
    out_flat = f_impl(*consts, *carry, *x)
    carry_out, y_updates = split_list(out_flat, [num_carry])
    ys_out = _map(partial(_update_array, i_), y_avals, ys, y_updates)
    return [i + 1] + carry_out + ys_out

  ys_init = _map(partial(_empty_array, length), y_avals)
  if length == 0:
    return init + ys_init
  else:
    init_val = [lax._const(length, 0)] + init + ys_init
    _, *outs = while_loop(cond_fun, body_fun, init_val)
    return outs

def _scan_impl_block_unrolled(*args, reverse, length, num_consts, num_carry,
                              linear, block_length, f_impl, x_avals, y_avals):
  consts, init, xs = split_list(args, [num_consts, num_carry])

  num_blocks, rem = divmod(length, block_length)
  assert rem == 0

  partition = partial(_partition_leading, num_blocks, block_length)
  xs_block = _map(partition, x_avals, xs)

  prepend_aval = partial(_prepend_dim_to_aval, block_length)
  x_block_avals = _map(prepend_aval, x_avals)
  y_block_avals = _map(prepend_aval, y_avals)

  f_impl_block = partial(
      _scan_impl_unrolled, reverse=reverse, length=block_length,
      num_consts=num_consts, num_carry=num_carry, linear=linear,
      f_impl=f_impl, x_avals=x_avals, y_avals=y_avals)

  outs = _scan_impl_loop(
      *consts, *init, *xs_block, reverse=reverse, length=num_blocks,
      num_consts=num_consts, num_carry=num_carry, linear=linear,
      f_impl=f_impl_block, x_avals=x_block_avals, y_avals=y_block_avals)

  carry, ys_blocks = split_list(outs, [num_carry])
  combine = partial(_combine_leading, num_blocks, block_length)
  ys = _map(combine, y_avals, ys_blocks)
  return (*carry, *ys)

def _scan_impl(*args, reverse, length, num_consts, num_carry, jaxpr, linear,
               unroll):
  _, _, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  _, y_avals = split_list(jaxpr.out_avals, [num_carry])
  f_impl = core.jaxpr_as_fun(jaxpr)

  if unroll == 1:
    return _scan_impl_loop(
        *args, reverse=reverse, length=length, num_consts=num_consts,
        num_carry=num_carry, linear=linear, f_impl=f_impl, x_avals=x_avals,
        y_avals=y_avals)

  consts, init, xs = split_list(args, [num_consts, num_carry])
  num_blocks, rem = divmod(length, unroll)
  length_div = num_blocks * unroll

  if rem > 0:
    if reverse:
      split = partial(_split_leading_dim, rem)
      xs_rem, xs = unzip2(_map(split, x_avals, xs))
    else:
      split = partial(_split_leading_dim, length_div)
      xs, xs_rem = unzip2(_map(split, x_avals, xs))

  outs = _scan_impl_block_unrolled(
      *consts, *init, *xs, reverse=reverse, length=length_div,
      num_consts=num_consts, num_carry=num_carry, linear=linear,
      block_length=unroll, f_impl=f_impl, x_avals=x_avals, y_avals=y_avals)

  carry, ys = split_list(outs, [num_carry])

  if rem > 0:
    outs = _scan_impl_unrolled(
        *consts, *carry, *xs_rem, reverse=reverse, length=rem,
        num_consts=num_consts, num_carry=num_carry, linear=linear,
        f_impl=f_impl, x_avals=x_avals, y_avals=y_avals)
    carry, ys_rem = split_list(outs, [num_carry])
    if reverse:
      ys = _map(_concatenate, y_avals, ys_rem, ys)
    else:
      ys = _map(_concatenate, y_avals, ys, ys_rem)

  return (*carry, *ys)

def _stack(aval, vals):
  if aval is core.abstract_unit:
    return core.unit
  else:
    vals = [lax.expand_dims(x, (0,)) for x in vals]
    return lax.concatenate(vals, 0)

def _concatenate(aval, x1, x2):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return lax.concatenate([x1, x2], 0)

def _split_leading_dim(i, aval, x):
  if aval is core.abstract_unit:
    return (core.unit, core.unit)
  else:
    assert x.ndim >= 1
    return (slicing.slice_in_dim(x, 0, i),
            slicing.slice_in_dim(x, i, x.shape[0]))

def _dynamic_index_array(i, aval, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return slicing.dynamic_index_in_dim(x, i, keepdims=False)

def _index_array(i, aval, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return slicing.index_in_dim(x, i, keepdims=False)

def _empty_array(sz, aval):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return lax.full((sz,) + aval.shape, 0, aval.dtype)

def _update_array(i, aval, xs, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return slicing.dynamic_update_index_in_dim(xs, x, i, 0)

def _partition_leading(sz0, sz1, aval, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    assert x.ndim >= 1
    assert x.shape[0] == sz0 * sz1
    return lax.reshape(x, (sz0, sz1, *x.shape[1:]))

def _combine_leading(sz0, sz1, aval, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    assert x.ndim >= 2
    assert x.shape[0] == sz0
    assert x.shape[1] == sz1
    return lax.collapse(x, 0, 2)

def _prepend_dim_to_aval(sz, aval):
  if aval is core.abstract_unit:
    return aval
  elif isinstance(aval, ShapedArray):
    return aval.update(shape=(sz, *aval.shape), weak_type=False)
  else:
    raise TypeError(f'Prepending dim {sz} to aval {aval}')

def _scan_abstract_eval(*args, reverse, length, num_consts, num_carry, jaxpr,
                        linear, unroll):
  carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_avals = _map(partial(_prepend_dim_to_aval, length), y_avals)
  return carry_avals + ys_avals

def _scan_jvp(primals, tangents, reverse, length, jaxpr, num_consts, num_carry,
              linear, unroll):
  num_xs = len(jaxpr.in_avals) - num_carry - num_consts
  num_ys = len(jaxpr.out_avals) - num_carry
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  const_nz, init_nz, xs_nz = split_list(nonzeros, [num_consts, num_carry])

  # Fixpoint computation of which carry are not ad.zero: either
  # non-zero from init, or the carry out is non-zero. Each iteration promotes
  # at least one carry to non-zero. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_nz.
  carry_nz = init_nz
  for _ in range(1 + len(carry_nz)):
    nonzeros = const_nz + carry_nz + xs_nz
    jaxpr_jvp, nonzeros_out = ad.jvp_jaxpr(
        jaxpr, nonzeros, instantiate=carry_nz + [False] * num_ys)
    carry_nz_out, _ = nonzeros_out[:num_carry], nonzeros_out[num_carry:]
    if carry_nz_out == carry_nz:
      break
    else:
      carry_nz = _map(operator.or_, carry_nz, carry_nz_out)
  else:
    assert False, "Fixpoint not reached"

  tangents = [ad.instantiate_zeros(t) if nz else t
              for t, nz in zip(tangents, nonzeros)]

  consts, init, xs = split_list(primals, [num_consts, num_carry])
  all_tangents = split_list(tangents, [num_consts, num_carry])
  consts_dot, init_dot, xs_dot = _map(_prune_zeros, all_tangents)

  jaxpr_jvp_rearranged = ad.rearrange_binders(
      jaxpr_jvp,
      [num_consts, num_carry, num_xs], [len(consts_dot), len(init_dot), len(xs_dot)],
      [num_carry, num_ys], [len(init_dot), sum(nonzeros_out) - len(init_dot)])

  consts_linear, init_linear, xs_linear = split_list(linear, [num_consts, num_carry])
  jaxpr_jvp_linear = tuple(consts_linear + [True] * len(consts_dot)
                           + init_linear + [True] * len(init_dot)
                           + xs_linear + [True] * len(xs_dot))

  out_flat = scan_p.bind(
      *(consts + consts_dot + init + init_dot + xs + xs_dot),
      reverse=reverse, length=length, jaxpr=jaxpr_jvp_rearranged,
      num_consts=num_consts + len(consts_dot),
      num_carry=num_carry + len(init_dot),
      linear=jaxpr_jvp_linear, unroll=unroll)

  carry, carry_dot, ys, ys_dot = split_list(out_flat, [num_carry, len(init_dot), num_ys])
  primals_out = carry + ys
  tangents_out_iter = iter(carry_dot + ys_dot)
  tangents_out = [next(tangents_out_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(primals_out, nonzeros_out)]
  return primals_out, tangents_out

def _prune_zeros(ts):
  return [t for t in ts if type(t) is not ad_util.Zero]

def _scan_partial_eval(trace, *tracers, reverse, length, num_consts, num_carry,
                       jaxpr, linear, unroll):
  num_ys = len(jaxpr.out_avals) - num_carry

  unknowns = [t.pval[0] is not None for t in tracers]
  const_uk, init_uk, xs_uk = split_list(unknowns, [num_consts, num_carry])

  # Fixpoint computation of which carry are unknown (not a constant): either
  # unknown from init, or the carry out is unknown. Each iteration promotes
  # at least one carry to unknown. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_uk.
  carry_uk = init_uk
  for _ in range(1 + len(carry_uk)):
    unknowns = const_uk + carry_uk + xs_uk
    jaxpr_1, jaxpr_2, out_uk = pe.partial_eval_jaxpr(
        jaxpr, unknowns, instantiate=carry_uk + [False] * num_ys)
    carry_uk_out = out_uk[:num_carry]
    if carry_uk_out == carry_uk:
      break
    else:
      carry_uk = _map(operator.or_, carry_uk, carry_uk_out)
  else:
    assert False, "Fixpoint not reached"
  num_res = len(jaxpr_1.out_avals) - len(jaxpr_2.out_avals)

  # The residuals are treated as extensive outputs of jaxpr_1 (and extensive
  # inputs to jaxpr_2), but residuals that are loop-invariant can be hoisted.
  # TODO(mattjj): hoist other loop-invariant values here too (instantiate=False)
  invariant_pvals = [pe.PartialVal.known(core.unit if uk else t.pval[1])
                     for uk, t in zip(unknowns[:num_consts], tracers[:num_consts])]
  other_pvals = [pe.PartialVal.unknown(a) for a in jaxpr_1.in_avals[num_consts:]]
  in_pvals_1 = invariant_pvals + other_pvals
  with source_info_util.reset_name_stack():
    jaxpr_1_opt, out_pvals_1, consts_1 = pe.trace_to_jaxpr(
        lu.wrap_init(core.jaxpr_as_fun(jaxpr_1)), in_pvals_1,
        instantiate=[True] * (num_carry + num_ys) + [False] * num_res)
  jaxpr_1_opt = pe.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr_1_opt), ())
  num_consts_1 = num_consts + len(consts_1)
  # any now-known residuals are intensive, so we want to revise jaxpr_2 to take
  # those inputs as constants rather than as extensive inputs
  _, _, res_pvals = split_list(out_pvals_1, [num_carry, num_ys])
  intensive_residuals = [const for pv, const in res_pvals if pv is None]
  move = [False] * len(jaxpr_1.in_avals) + [pv is None for pv, _ in res_pvals]
  jaxpr_2_opt = pe.move_binders_to_front(jaxpr_2, move)
  num_consts_2 = num_consts + len(intensive_residuals)

  # As another optimization, for any extensive inputs that are just forwarded to
  # extensive outputs, to avoid a copy (looping over dynamic-update-slice) we'd
  # rather just forward the input tracer. That means pruning some extensive
  # outputs from the jaxpr here, and updating out_flat below.
  extensive_invars = jaxpr_1_opt.jaxpr.invars[num_consts_1 + num_carry:]
  extensive_outvars = jaxpr_1_opt.jaxpr.outvars[num_carry:]
  extensive_avals = [core.unmapped_aval(length, core.no_axis_name, 0,
                                        core.raise_to_shaped(v.aval))
                     for v in extensive_outvars]
  fwd_extensive = [num_consts + num_carry + extensive_invars.index(v)
                   if v in extensive_invars else None for v in extensive_outvars]
  jaxpr_1_opt.jaxpr.outvars = (
      jaxpr_1_opt.jaxpr.outvars[:num_carry] +
      [v for i, v in zip(fwd_extensive, extensive_outvars) if i is None])

  in_consts = (list(consts_1) + [core.unit] * num_consts +
               [core.unit if uk else t.pval[1]
                for uk, t in zip(unknowns[num_consts:], tracers[num_consts:])])
  linear_1 = ([False] * len(consts_1) + [True] * num_consts +
              [lin or uk for uk, lin
               in zip(unknowns[num_consts:], linear[num_consts:])])
  out_flat = scan_p.bind(
      *in_consts, reverse=reverse, length=length, jaxpr=jaxpr_1_opt,
      num_consts=num_consts_1, num_carry=num_carry, linear=tuple(linear_1),
      unroll=unroll)

  # Propagate the forwarded extensive outputs using fwd_extensive. Any
  # numpy.ndarray inputs should be converted to JAX DeviceArrays.
  out_carry, out_extensive = split_list(out_flat, [num_carry])
  out_extensive_iter = iter(out_extensive)
  out_extensive = [next(out_extensive_iter) if i is None
                   else _maybe_device_put(tracers[i].pval[1]) if tracers[i].is_known()
                   else tracers[i] for i in fwd_extensive]
  assert all(core.typematch(a, core.get_aval(out))
             for a, out in zip(extensive_avals, out_extensive))
  out_flat = out_carry + out_extensive

  out_carry, ys, res_and_units = split_list(out_flat, [num_carry, num_ys])
  extensive_residuals = [r for r, (pv, _) in zip(res_and_units, res_pvals) if pv is not None]

  new_tracers = [trace.instantiate_const(t) if uk else trace.new_instantiated_literal(core.unit)
                 for uk, t in zip(unknowns, tracers)]
  carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_avals = _map(partial(_prepend_dim_to_aval, length), y_avals)
  out_avals = carry_avals + ys_avals
  out_pvs = [aval if uk else None for aval, uk in zip(out_avals, out_uk)]

  out_consts = out_carry + ys
  int_res_tracers = _map(trace.new_instantiated_const, intensive_residuals)
  ext_res_tracers = _map(trace.new_instantiated_const, extensive_residuals)
  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal((pv, const)), None)
                 for pv, const in zip(out_pvs, out_consts)]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)
  linear_2 = ([False] * len(int_res_tracers) +
              [lin or not uk for uk, lin in zip(unknowns, linear)] +
              [False] * len(ext_res_tracers))
  eqn = pe.new_eqn_recipe(int_res_tracers + new_tracers + ext_res_tracers,
                          out_tracers, scan_p,
                          dict(reverse=reverse, length=length, jaxpr=jaxpr_2_opt,
                               num_consts=num_consts_2,
                               num_carry=num_carry, linear=tuple(linear_2),
                               unroll=unroll),
                          source)
  for t in out_tracers: t.recipe = eqn
  return out_tracers

def _maybe_device_put(x):
  if isinstance(x, np.ndarray):
    return lax._device_put_raw(x)
  else:
    return x

def _scan_transpose(reduce_axes, cts, *args, reverse, length, num_consts,
                    num_carry, jaxpr, linear, unroll):
  # we've only implemented transposing scans with specific lin/nonlin patterns
  consts_lin, init_lin, xs_lin = split_list(linear, [num_consts, num_carry])
  num_ires = len(consts_lin) - sum(consts_lin)
  num_eres = len(xs_lin) - sum(xs_lin)
  if consts_lin != [False] * num_ires + [True] * (len(consts_lin) - num_ires):
    raise NotImplementedError
  if xs_lin != [True] * (len(xs_lin) - num_eres) + [False] * num_eres:
    raise NotImplementedError
  if not all(init_lin):
    pass  # TODO(mattjj): error check https://github.com/google/jax/issues/1963

  consts, _, xs = split_list(args, [num_consts, num_carry])
  ires, _ = split_list(consts, [num_ires])
  _, eres = split_list(xs, [sum(xs_lin)])
  assert not any(ad.is_undefined_primal(r) for r in ires)
  assert not any(ad.is_undefined_primal(r) for r in eres)

  carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_avals = _map(partial(_prepend_dim_to_aval, length), y_avals)
  ct_carry, ct_ys = split_list(cts, [num_carry])
  ct_carry = _map(ad.instantiate_zeros_aval, carry_avals, ct_carry)
  ct_ys = _map(ad.instantiate_zeros_aval, ys_avals, ct_ys)
  ct_consts = _map(ad_util.zeros_like_aval, jaxpr.in_avals[num_ires:num_consts])

  #       jaxpr :: [ires, T d] -> [T c] -> [T a, eres] -> ([T c], [T b])
  # jaxpr_trans :: [ires] -> [CT d, CT c] -> [CT b, eres] -> ([CT d, CT c], [CT a])
  jaxpr_trans = _transpose_scan_jaxpr(
      num_ires, num_consts - num_ires, num_eres, jaxpr, reduce_axes)
  linear_trans = ([False] * num_ires +
                  [True] * (len(ct_consts) + len(ct_carry) + len(ct_ys)) +
                  [False] * num_eres)

  outs = scan_p.bind(
      *(ires + ct_consts + ct_carry + ct_ys + eres), reverse=not reverse,
      length=length, jaxpr=jaxpr_trans, num_consts=num_ires,
      num_carry=num_consts-num_ires+num_carry, linear=tuple(linear_trans),
      unroll=unroll)
  ct_consts, ct_init, ct_xs = split_list(outs, [num_consts - num_ires, num_carry])
  return [None] * num_ires + ct_consts + ct_init + ct_xs + [None] * num_eres

# transpose_scan_jaxpr :: ([res1, c, a, res2] -> b)
#                         -> ([res1, CT c, CT b, res2] -> [CT c, CT a])
def _transpose_scan_jaxpr(num_res1, num_c, num_res2, jaxpr, reduce_axes):
  num_a = len(jaxpr.in_avals) - num_res1 - num_c - num_res2
  # TODO: allow input cotangent avals to be batched relative to jaxpr.in_avals
  # if an axis isn't reduced
  res1_avals, c_avals, a_avals, res2_avals = split_list(
      jaxpr.in_avals, [num_res1, num_c, num_a])
  num_b = len(jaxpr.out_avals)
  b_avals = list(jaxpr.out_avals)

  @lu.wrap_init
  def transposed(*res1_cbar_bbar_res2):
    res1, c_bar, b_bar, res2 = split_list(
        res1_cbar_bbar_res2, [num_res1, num_c, num_b])
    primals = (res1 + [ad.UndefinedPrimal(aval) for aval in c_avals] +
               [ad.UndefinedPrimal(aval) for aval in a_avals] + res2)
    cbar_abar = ad.backward_pass(jaxpr.jaxpr, reduce_axes, False, jaxpr.consts,
                                 primals, b_bar)
    _, new_c_bar, a_bar, _ = split_list(cbar_abar, [num_res1, num_c, num_a])
    a_bar = _map(ad.instantiate_zeros_aval, a_avals, a_bar)
    c_bar = _map(ad.instantiate_zeros_aval, c_avals,
                _map(ad.add_tangents, c_bar, new_c_bar))
    return c_bar + a_bar
  return _make_closed_jaxpr(transposed, res1_avals + c_avals + b_avals + res2_avals)

def _make_closed_jaxpr(traceable: lu.WrappedFun, in_avals: Sequence[core.AbstractValue]):
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(traceable, in_avals)
  return core.ClosedJaxpr(jaxpr, consts)


def _scan_batching_rule(axis_size, axis_name, main_type, args, dims, reverse, length,
                        jaxpr, num_consts, num_carry, linear, unroll):
  num_ys = len(jaxpr.out_avals) - num_carry
  orig_batched = [d is not batching.not_mapped for d in dims]
  const_batched, init_batched, xs_batched = split_list(orig_batched, [num_consts, num_carry])

  # Fixpoint computation of which carry are batched: either
  # batched from init, or the carry out is batched. Each iteration promotes
  # at least one carry to batched. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_batched.
  carry_batched = init_batched
  for _ in range(1 + len(carry_batched)):
    batched = const_batched + carry_batched + xs_batched
    jaxpr_batched, batched_out = batching.batch_jaxpr(
        jaxpr, axis_size, batched,
        instantiate=carry_batched + [False] * num_ys,
        axis_name=axis_name,
        main_type=main_type)
    carry_batched_out, ys_batched = batched_out[:num_carry], batched_out[num_carry:]
    if carry_batched_out == carry_batched:
      break
    else:
      carry_batched = _map(operator.or_, carry_batched, carry_batched_out)
  else:
    assert False, "Fixpoint not reached"

  consts, init, xs = split_list(args, [num_consts, num_carry])
  consts_bdims, init_bdims, xs_bdims = split_list(dims, [num_consts, num_carry])
  new_consts = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
                else x for x, d in zip(consts, consts_bdims)]
  new_init = [batching.broadcast(x, axis_size, 0) if now_batched and not was_batched
              else batching.moveaxis(x, d, 0) if now_batched else x
              for x, d, was_batched, now_batched in
              zip(init, init_bdims, init_batched, carry_batched)]
  new_xs = [batching.moveaxis(x, d, 1) if d is not batching.not_mapped and d != 1
            else x for x, d in zip(xs, xs_bdims)]
  new_args = new_consts + new_init + new_xs

  outs = scan_p.bind(
      *new_args, reverse=reverse, length=length, jaxpr=jaxpr_batched,
      num_consts=num_consts, num_carry=num_carry, linear=linear, unroll=unroll)
  carry_bdims = [0 if b else batching.not_mapped for b in carry_batched]
  ys_bdims = [1 if b else batching.not_mapped for b in ys_batched]
  return outs, carry_bdims + ys_bdims

def _scan_masking_rule(padded_vals, logical_shapes, reverse, length,
                       jaxpr, num_consts, num_carry, linear, unroll):
  dynamic_length, = masking.shape_as_value((length,))
  masked_jaxpr = _masked_scan_jaxpr(jaxpr, num_consts, num_carry)
  consts, init, xs = split_list(padded_vals, [num_consts, num_carry])
  max_length, = {x.shape[0] for x in xs}
  const_linear, init_linear, xs_linear = split_list(linear, [num_consts, num_carry])
  dynamic_length = lax.convert_element_type(dynamic_length, dtypes.int_)
  out_vals = scan_p.bind(dynamic_length, *consts, dtypes.int_(0), *init, *xs,
      reverse=reverse, length=max_length, jaxpr=masked_jaxpr,
      num_consts=1 + num_consts, num_carry=1 + num_carry,
      linear=tuple([False] + const_linear + [False] + init_linear + xs_linear),
      unroll=unroll)
  return out_vals[1:]

def _masked_scan_jaxpr(jaxpr, num_consts, num_carry):
  fun = core.jaxpr_as_fun(jaxpr)

  @lu.wrap_init
  def masked(*args):
    [dynamic_length], consts, [i], carry, xs = split_list(
        args, [1, num_consts, 1, num_carry])
    out = fun(*(consts + carry + xs))
    new_carry, ys = split_list(out, [num_carry])
    new_carry = [lax.select(i < dynamic_length, new_c, c)
                 for new_c, c in zip(new_carry, carry)]
    return [i + 1] + new_carry + ys

  aval = ShapedArray((), dtypes.canonicalize_dtype(dtypes.int_))
  const_avals, carry_avals, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  return _make_closed_jaxpr(masked, [aval] + const_avals + [aval] + carry_avals + x_avals)

def _scan_typecheck(bind_time, *avals, reverse, length, num_consts, num_carry,
                    jaxpr, linear, unroll):
  tc = partial(_typecheck_param, 'scan')
  tc(reverse, 'reverse', 'bool', type(reverse) is bool)
  tc(num_consts, 'num_consts', 'non-negative int',
     type(num_consts) is int and num_consts >= 0)
  tc(num_carry, 'num_carry', 'non-negative int',
     type(num_carry) is int and num_carry >= 0)
  tc(jaxpr, 'jaxpr', 'ClosedJaxpr', type(jaxpr) is core.ClosedJaxpr)
  tc(linear, 'linear', 'tuple of bool',
     type(linear) is tuple and all(type(x) is bool for x in linear))
  tc(unroll, 'unroll', 'positive int', type(unroll) is int and unroll > 0)

  length_types = (int, masking.Poly) if bind_time else (int,)
  tc(length, 'length', 'non-negative int',
     type(length) in length_types and length >= 0)

  if len(linear) != len(avals):
    raise core.JaxprTypeError(
      f'scan param linear has length {len(linear)} for {len(avals)} operands')

  const_avals, init_avals, x_avals = split_list(avals, [num_consts, num_carry])
  const_avals_jaxpr, init_avals_jaxpr, x_avals_jaxpr = split_list(
      jaxpr.in_avals, [num_consts, num_carry])
  carry_avals_jaxpr, _ = split_list(jaxpr.out_avals, [num_carry])
  x_avals_mapped = _map(partial(core.mapped_aval, length, 0), x_avals)

  if not all(_map(core.typematch, init_avals_jaxpr, carry_avals_jaxpr)):
    raise core.JaxprTypeError(
      f'scan input carry input and output types mismatch: '
      f'\n{_avals_short(init_avals_jaxpr)}\nvs\n{_avals_short(carry_avals_jaxpr)}')
  if not all(_map(core.typecompat, const_avals_jaxpr, const_avals)):
    raise core.JaxprTypeError(
      f'scan jaxpr takes input const types\n{_avals_short(const_avals_jaxpr)},\n'
      f'called with consts of type\n{_avals_short(const_avals)}')
  if not all(_map(core.typecompat, init_avals_jaxpr, init_avals)):
    raise core.JaxprTypeError(
      f'scan jaxpr takes input carry types\n{_avals_short(init_avals_jaxpr)},\n'
      f'called with initial carry of type\n{_avals_short(init_avals)}')
  if not all(_map(core.typecompat, x_avals_jaxpr, x_avals_mapped)):
    raise core.JaxprTypeError(
      f'scan jaxpr takes input sequence types\n{_avals_short(x_avals_jaxpr)},\n'
      f'called with sequence of type\n{_avals_short(x_avals)}')

def scan_bind(*args, **params):
  if config.jax_enable_checks:
    avals = _map(core.get_aval, args)
    _scan_typecheck(True, *avals, **params)
    core.check_jaxpr(params['jaxpr'].jaxpr)
  return core.AxisPrimitive.bind(scan_p, *args, **params)

scan_p = core.AxisPrimitive("scan")
scan_p.multiple_results = True
scan_p.def_custom_bind(scan_bind)
scan_p.def_impl(partial(xla.apply_primitive, scan_p))
scan_p.def_abstract_eval(_scan_abstract_eval)
ad.primitive_jvps[scan_p] = _scan_jvp
ad.reducing_transposes[scan_p] = _scan_transpose
pe.custom_partial_eval_rules[scan_p] = _scan_partial_eval
xla.register_translation(scan_p, xla.lower_fun(_scan_impl, new_style=True,
                                               multiple_results=True),
                         initial_style=True)
batching.axis_primitive_batchers[scan_p] = _scan_batching_rule
masking.masking_rules[scan_p] = _scan_masking_rule
core.custom_typechecks[scan_p] = partial(_scan_typecheck, False)
pe.partial_eval_jaxpr_custom_rules[scan_p] = \
    partial(pe.partial_eval_jaxpr_custom_rule_not_implemented, 'scan')

mlir.register_lowering(scan_p,
                       mlir.lower_fun(_scan_impl, multiple_results=True))


@api_boundary
def map(f, xs):
  """Map a function over leading array axes.

  Like Python's builtin map, except inputs and outputs are in the form of
  stacked arrays. Consider using the ``jax.vmap`` transform instead, unless you
  need to apply a function element by element for reduced memory usage or
  heterogeneous computation with other control flow primitives.

  When ``xs`` is an array type, the semantics of ``map`` are given by this
  Python implementation::

    def map(f, xs):
      return np.stack([f(x) for x in xs])

  Like ``scan``, ``map`` is implemented in terms of JAX primitives so many of
  the same advantages over a Python loop apply: ``xs`` may be an arbitrary
  nested pytree type, and the mapped computation is compiled only once.

  Args:
    f: a Python function to apply element-wise over the first axis or axes of
      ``xs``.
    xs: values over which to map along the leading axis.

  Returns:
    Mapped values.
  """
  g = lambda _, x: ((), f(x))
  _, ys = scan(g, (), xs)
  return ys


def _concat_masking_rule(padded_vals, logical_shapes, dimension):
  result = lax.concatenate(padded_vals, dimension)  # fragmented
  offset = 0
  for padded_val, logical_shape in zip(padded_vals, logical_shapes):
    result = _memcpy(dimension, logical_shape[dimension], padded_val,
                     result, offset)
    offset = offset + logical_shape[dimension]
  return result

def _memcpy(axis, num, src, dst, offset):
  def body(i, dst):
    update = slicing.dynamic_index_in_dim(src, i, axis)
    return slicing.dynamic_update_index_in_dim(dst, update, i + offset, axis)
  return fori_loop(0, num, body, dst)

masking.masking_rules[lax.concatenate_p] = _concat_masking_rule  # type: ignore

def _rng_bit_generator_batching_rule(batched_args, batch_dims, *, shape, dtype, algorithm):
  """Calls RBG in a loop and stacks the results."""
  key, = batched_args
  bd, = batch_dims
  if bd is batching.not_mapped:
    return lax.rng_bit_generator_p.bind(key, shape=shape, dtype=dtype,
                                        algorithm=algorithm), (None, None)
  key = batching.moveaxis(key, bd, 0)
  map_body = lambda k: lax.rng_bit_generator_p.bind(k, shape=shape, dtype=dtype, algorithm=algorithm)
  stacked_keys, stacked_bits = map(map_body, key)
  return (stacked_keys, stacked_bits), (0, 0)

batching.primitive_batchers[lax.rng_bit_generator_p] = _rng_bit_generator_batching_rule

def _show_diff(array1, array2):
  if core.typematch(array1, array2):
    return f"{array1}"
  return f"DIFFERENT {array1} vs. {array2}"

def _check_tree_and_avals(what, tree1, avals1, tree2, avals2):
  """Raises TypeError if (tree1, avals1) does not match (tree2, avals2).

  Corresponding `tree` and `avals` must match in the sense that the number of
  leaves in `tree` must be equal to the length of `avals`. `what` will be
  prepended to details of the mismatch in TypeError.
  """
  if tree1 != tree2:
    raise TypeError(
        f"{what} must have same type structure, got {tree1} and {tree2}.")
  if not all(_map(core.typematch, avals1, avals2)):
    diff = tree_map(_show_diff, tree_unflatten(tree1, avals1),
                    tree_unflatten(tree2, avals2))
    raise TypeError(f"{what} must have identical types, got\n{diff}.")


def _check_tree(func_name, expected_name, actual_tree, expected_tree, has_aux=False):
  if has_aux:
    actual_tree_children = actual_tree.children()

    if len(actual_tree_children) == 2:
      # select first child as result tree
      actual_tree = tree_structure(actual_tree_children[0])
    else:
      raise ValueError(
        f"{func_name}() produced a pytree with structure "
        f"{actual_tree}, but a pytree tuple with auxiliary "
        f"output was expected because has_aux was set to True.")

  if actual_tree != expected_tree:
    raise TypeError(
        f"{func_name}() output pytree structure must match {expected_name}, "
        f"got {actual_tree} and {expected_tree}.")


def _promote_weak_typed_inputs(in_vals, in_avals, out_avals):
  """Promote weakly-typed in_vals to be compatible with out_avals.

  Args:
    in_vals : flattened list of input values.
    in_avals : corresponding list of avals.
    out_avals : list of target output avals.
  Returns:
    in_vals_new : flattened list of modified in_vals with no weak types.
    changed : bool; true if in_vals required modification.
  """
  if len(in_vals) != len(in_avals) or len(in_avals) != len(out_avals):
    # Calling function is responsible for catching this.
    return in_vals, False
  weak_mismatches = [i for i, (a1, a2) in enumerate(zip(in_avals, out_avals))
                    if getattr(a1, 'weak_type', False) and not core.typematch(a1, a2)]
  if not weak_mismatches:
    return in_vals, False
  for i in weak_mismatches:
    new_dtype = dtypes.result_type(in_vals[i], out_avals[i])
    in_vals[i] = lax.convert_element_type(in_vals[i], new_dtype)
  return in_vals, True


_RootTuple = collections.namedtuple('_RootTuple', 'f, solve, l_and_s')


def _split_root_args(args, const_lengths):
  params_list = split_list(args, list(const_lengths))
  return _RootTuple(*params_list[:-1]), params_list[-1]


@api_boundary
def custom_root(f, initial_guess, solve, tangent_solve, has_aux=False):
  """Differentiably solve for a roots of a function.

  This is a low-level routine, mostly intended for internal use in JAX.
  Gradients of custom_root() are defined with respect to closed-over variables
  from the provided function ``f`` via the implicit function theorem:
  https://en.wikipedia.org/wiki/Implicit_function_theorem

  Args:
    f: function for which to find a root. Should accept a single argument,
      return a tree of arrays with the same structure as its input.
    initial_guess: initial guess for a zero of f.
    solve: function to solve for the roots of f. Should take two positional
      arguments, f and initial_guess, and return a solution with the same
      structure as initial_guess such that func(solution) = 0. In other words,
      the following is assumed to be true (but not checked)::

        solution = solve(f, initial_guess)
        error = f(solution)
        assert all(error == 0)

    tangent_solve: function to solve the tangent system. Should take two
      positional arguments, a linear function ``g`` (the function ``f``
      linearized at its root) and a tree of array(s) ``y`` with the same
      structure as initial_guess, and return a solution ``x`` such that
      ``g(x)=y``:

      - For scalar ``y``, use ``lambda g, y: y / g(1.0)``.
      - For vector ``y``, you could use a linear solve with the Jacobian, if
        dimensionality of ``y`` is not too large:
        ``lambda g, y: np.linalg.solve(jacobian(g)(y), y)``.
    has_aux: bool indicating whether the ``solve`` function returns
      auxiliary data like solver diagnostics as a second argument.

  Returns:
    The result of calling solve(f, initial_guess) with gradients defined via
    implicit differentiation assuming ``f(solve(f, initial_guess)) == 0``.
  """
  guess_flat, in_args_tree = tree_flatten((initial_guess,))
  guess_avals = tuple(_map(_abstractify, guess_flat))
  f_jaxpr, f_consts, out_tree = _initial_style_jaxpr(
      f, in_args_tree, guess_avals)

  in_tree, = treedef_children(in_args_tree)
  _check_tree("f", "initial_guess", out_tree, in_tree, False)

  solve_jaxpr, solve_consts, solution_tree = _initial_style_jaxpr(
      partial(solve, f), in_args_tree, guess_avals)
  _check_tree("solve", "initial_guess", solution_tree, in_tree, has_aux)

  def linearize_and_solve(x, b):
    unchecked_zeros, f_jvp = jax.linearize(f, x)
    return tangent_solve(f_jvp, b)

  l_and_s_jaxpr, l_and_s_consts, out_tree = _initial_style_jaxpr(
      linearize_and_solve, treedef_tuple((in_tree,) * 2), guess_avals * 2)
  _check_tree("tangent_solve", "x", out_tree, in_tree, False)

  all_consts = [f_consts, solve_consts, l_and_s_consts]
  const_lengths = _RootTuple(*_map(len, all_consts))
  jaxprs = _RootTuple(f_jaxpr, solve_jaxpr, l_and_s_jaxpr)

  solution_flat = _custom_root(
      const_lengths, jaxprs, *(_flatten(all_consts) + guess_flat))
  return tree_unflatten(solution_tree, solution_flat)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _custom_root(const_lengths, jaxprs, *args):
  params, initial_guess = _split_root_args(args, const_lengths)
  solution = core.jaxpr_as_fun(jaxprs.solve)(*(params.solve + initial_guess))
  return solution


@_custom_root.defjvp
def _root_jvp(const_lengths, jaxprs, primals, tangents):
  params, _ = _split_root_args(primals, const_lengths)
  sol = _custom_root(const_lengths, jaxprs, *primals)

  f_out_vals = len(jaxprs.f.out_avals)
  solution, aux = split_list(sol, [f_out_vals])

  params_dot, _ = _split_root_args(tangents, const_lengths)

  # F(m, u) = 0      # system of equations in u, parameterized by m
  #                  # solution is u*(m) defined in a neighborhood
  # F(m, u*(m)) = 0  # satisfied in a neighborhood
  #
  # ∂_0 F(m, u*(m)) + ∂_1 F(m, u*(m)) ∂ u*(m) = 0       # implied by line above
  # ∂ u*(m) = - (∂_1 F(m, u*(m)))^{-1} ∂_0 F(m, u*(m))  # rearrange
  #
  # ∂ u*(m)[v] = - (∂_1 F(m, u*(m)))^{-1} [∂_0 F(m, u*(m))[v]]  # jvp

  f = core.jaxpr_as_fun(jaxprs.f)
  linearize_and_solve = partial(
      core.jaxpr_as_fun(jaxprs.l_and_s), *params.l_and_s)
  f_at_solution = lambda *params: f(*params, *solution)
  _, rhs = ad.jvp(lu.wrap_init(f_at_solution)).call_wrapped(
      params.f, params_dot.f)
  solution_dot = _map(
      operator.neg, linearize_and_solve(*solution, *rhs))
  # append aux, create symbolic zero tangents for the aux values
  solution += aux
  solution_dot += _map(lax.zeros_like_array, aux)

  return solution, solution_dot


class _LinearSolveTuple(collections.namedtuple(
    '_LinearSolveTuple', 'matvec, vecmat, solve, transpose_solve')):

  def transpose(self):
    return type(self)(self.vecmat, self.matvec, self.transpose_solve, self.solve)


def _split_linear_solve_args(args, const_lengths):
  params_list = split_list(args, list(const_lengths))
  return _LinearSolveTuple(*params_list[:-1]), params_list[-1]


def _transpose_one_output(linear_fun, primals):
  transpose_fun = jax.linear_transpose(linear_fun, primals)
  def transposed_fun(x):
    (y,) = transpose_fun(x)
    return y
  return transposed_fun


def _flatten(args):
  return [x for arg in args for x in arg]


def _check_shapes(func_name, expected_name, actual, expected):
  actual_shapes = _map(np.shape, tree_leaves(actual))
  expected_shapes = _map(np.shape, tree_leaves(expected))
  if actual_shapes != expected_shapes:
    raise ValueError(
        f"{func_name}() output shapes must match {expected_name}, "
        f"got {actual_shapes} and {expected_shapes}")


@api_boundary
def custom_linear_solve(
    matvec, b, solve, transpose_solve=None, symmetric=False, has_aux=False):
  """Perform a matrix-free linear solve with implicitly defined gradients.

  This function allows for overriding or defining gradients for a linear
  solve directly via implicit differentiation at the solution, rather than by
  differentiating *through* the solve operation. This can sometimes be much faster
  or more numerically stable, or differentiating through the solve operation
  may not even be implemented (e.g., if ``solve`` uses ``lax.while_loop``).

  Required invariant::

      x = solve(matvec, b)  # solve the linear equation
      assert matvec(x) == b  # not checked

  Args:
    matvec: linear function to invert. Must be differentiable.
    b: constant right handle side of the equation. May be any nested structure
      of arrays.
    solve: higher level function that solves for solution to the linear
      equation, i.e., ``solve(matvec, x) == x`` for all ``x`` of the same form
      as ``b``. This function need not be differentiable.
    transpose_solve: higher level function for solving the transpose linear
      equation, i.e., ``transpose_solve(vecmat, x) == x``, where ``vecmat`` is
      the transpose of the linear map ``matvec`` (computed automatically with
      autodiff). Required for backwards mode automatic differentiation, unless
      ``symmetric=True``, in which case ``solve`` provides the default value.
    symmetric: bool indicating if it is safe to assume the linear map
      corresponds to a symmetric matrix, i.e., ``matvec == vecmat``.
    has_aux: bool indicating whether the ``solve`` and ``transpose_solve`` functions
      return auxiliary data like solver diagnostics as a second argument.

  Returns:
    Result of ``solve(matvec, b)``, with gradients defined assuming that the
      solution ``x`` satisfies the linear equation ``matvec(x) == b``.
  """
  if transpose_solve is None and symmetric:
    transpose_solve = solve

  b_flat, in_args_tree = tree_flatten((b,))
  b_avals = tuple(_map(_abstractify, b_flat))

  tree, = treedef_children(in_args_tree)

  def _shape_checked(fun, name, has_aux):
    def f(x):
      y = fun(x)
      _check_shapes(name, "b", y, b_flat)
      return y

    def f_aux(x):
      y, aux = fun(x)
      _check_shapes(name, "b", y, b_flat)
      return y, aux

    return f_aux if has_aux else f

  # no auxiliary data assumed for matvec
  matvec_jaxpr, matvec_consts, out_tree = _initial_style_jaxpr(
      _shape_checked(matvec, "matvec", False), in_args_tree, b_avals,
      'custom_linear_solve')
  _check_tree("matvec", "b", out_tree, tree, False)

  solve_jaxpr, solve_consts, out_tree = _initial_style_jaxpr(
      _shape_checked(partial(solve, matvec), "solve", has_aux), in_args_tree, b_avals,
      'custom_linear_solve')
  _check_tree("solve", "b", out_tree, tree, has_aux)

  if transpose_solve is None:
    vecmat_jaxpr = tr_solve_jaxpr = None
    vecmat_consts = tr_solve_consts = []
  else:
    if symmetric:
      vecmat = matvec
      vecmat_jaxpr = matvec_jaxpr
      vecmat_consts = matvec_consts
    else:
      vecmat = _transpose_one_output(matvec, b)
      vecmat_jaxpr, vecmat_consts, out_tree = _initial_style_jaxpr(
          vecmat, in_args_tree, b_avals, 'custom_linear_solve')
      assert out_tree == tree

    tr_solve_jaxpr, tr_solve_consts, out_tree = _initial_style_jaxpr(
        _shape_checked(partial(transpose_solve, vecmat), "transpose_solve", has_aux),
        in_args_tree, b_avals, 'custom_linear_solve')
    _check_tree("transpose_solve", "b", out_tree, tree, has_aux)

  all_consts = [matvec_consts, vecmat_consts, solve_consts, tr_solve_consts]
  const_lengths = _LinearSolveTuple(*_map(len, all_consts))
  jaxprs = _LinearSolveTuple(
      matvec_jaxpr, vecmat_jaxpr, solve_jaxpr, tr_solve_jaxpr)

  out_flat = linear_solve_p.bind(
      *(_flatten(all_consts) + b_flat),
      const_lengths=const_lengths, jaxprs=jaxprs)

  return tree_unflatten(out_tree, out_flat)


def _linear_solve_abstract_eval(*args, const_lengths, jaxprs):
  args_to_raise = args[sum(const_lengths):]

  # raise aux_args to shaped arrays as well if present
  # number of aux args is the difference in out_avals
  # of solve and matvec (since they map to the same vector space)

  num_aux = len(jaxprs.solve.out_avals) - len(jaxprs.matvec.out_avals)
  if num_aux > 0:
    args_to_raise += tuple(jaxprs.solve.out_avals[-num_aux:])
  return _map(raise_to_shaped, args_to_raise)


def _custom_linear_solve_impl(*args, const_lengths, jaxprs):
  params, b = _split_linear_solve_args(args, const_lengths)
  x = core.jaxpr_as_fun(jaxprs.solve)(*(params.solve + b))
  return x


def _tangent_linear_map(func, params, params_dot, *x):
  """Compute the tangent of a linear map.

  Assuming ``func(*params, *x)`` is linear in ``x`` and computes ``A @ x``,
  this function computes ``∂A @ x``.
  """
  assert any(type(p) is not ad_util.Zero for p in params_dot)
  zeros = _map(ad_util.Zero.from_value, x)
  _, out_tangent = ad.jvp(lu.wrap_init(func)).call_wrapped(
      params + list(x), params_dot + zeros)
  return out_tangent


def _custom_linear_solve_jvp(primals, tangents, const_lengths, jaxprs):
  # A x - b = 0
  # ∂A x + A ∂x - ∂b = 0
  # ∂x = A^{-1} (∂b - ∂A x)

  kwargs = dict(const_lengths=const_lengths, jaxprs=jaxprs)
  x = linear_solve_p.bind(*primals, **kwargs)

  params, _ = _split_linear_solve_args(primals, const_lengths)
  params_dot, b_dot = _split_linear_solve_args(tangents, const_lengths)

  num_x_leaves = len(b_dot)
  # x is a flat tree with possible aux values appended
  # since x_tree == b_tree == b_dot_tree, we can cut off
  # aux values with len info provided by b_dot tree here
  x_leaves, _ = split_list(x, [num_x_leaves])

  if all(type(p) is ad_util.Zero for p in params_dot.matvec):
    # no need to evaluate matvec_tangents
    rhs = b_dot
  else:
    matvec_tangents = _tangent_linear_map(
        core.jaxpr_as_fun(jaxprs.matvec), params.matvec, params_dot.matvec, *x_leaves)
    rhs = _map(ad.add_tangents, b_dot, _map(operator.neg, matvec_tangents))

  x_dot = linear_solve_p.bind(*(_flatten(params) + rhs), **kwargs)

  # split into x tangents and aux tangents (these become zero)
  dx_leaves, daux_leaves = split_list(x_dot, [num_x_leaves])

  daux_leaves = _map(ad_util.Zero.from_value, daux_leaves)

  x_dot = dx_leaves + daux_leaves

  return x, x_dot


def _linear_solve_transpose_rule(cotangent, *primals, const_lengths, jaxprs):
  if jaxprs.transpose_solve is None:
    raise TypeError('transpose_solve required for backwards mode automatic '
                    'differentiation of custom_linear_solve')

  params, b = _split_linear_solve_args(primals, const_lengths)
  # split off symbolic zeros in the cotangent if present
  x_cotangent, _ = split_list(cotangent, [len(b)])
  assert all(ad.is_undefined_primal(x) for x in b)
  cotangent_b_full = linear_solve_p.bind(
      *(_flatten(params.transpose()) + x_cotangent),
      const_lengths=const_lengths.transpose(), jaxprs=jaxprs.transpose())
  # drop aux values in cotangent computation
  cotangent_b, _ = split_list(cotangent_b_full, [len(b)])
  return [None] * sum(const_lengths) + cotangent_b


def _linear_solve_batching_rule(axis_size, axis_name, main_type, args, dims,
                                const_lengths, jaxprs):
  orig_bat = [d is not batching.not_mapped for d in dims]

  params, b = _split_linear_solve_args(args, const_lengths)
  params_dims, b_dims = _split_linear_solve_args(dims, const_lengths)
  params_bat, orig_b_bat = _split_linear_solve_args(orig_bat, const_lengths)

  (matvec, vecmat, solve, solve_t) = jaxprs
  (matvec_bat, vecmat_bat, solve_bat, solve_t_bat) = params_bat

  num_aux = len(solve.out_avals) - len(matvec.out_avals)
  # Fixpoint computation of which parts of x and b are batched; we need to
  # ensure this is consistent between all four jaxprs
  b_bat = orig_b_bat
  x_bat = [False] * len(solve.out_avals)
  for i in range(1 + len(orig_b_bat) + len(solve.out_avals)):
    # Apply vecmat and solve -> new batched parts of x
    solve_jaxpr_batched, solve_x_bat = batching.batch_jaxpr(
        solve, axis_size, solve_bat + b_bat, instantiate=x_bat,
        axis_name=axis_name, main_type=main_type)
    if vecmat is None:
      vecmat_jaxpr_batched = None
      x_bat_out = solve_x_bat
    else:
      vecmat_jaxpr_batched, vecmat_x_bat = batching.batch_jaxpr(
          vecmat, axis_size, vecmat_bat + b_bat, instantiate=x_bat,
          axis_name=axis_name, main_type=main_type)
      # batch all aux data by default
      x_bat_out = _map(operator.or_, vecmat_x_bat + [True] * num_aux, solve_x_bat)

    # Apply matvec and solve_t -> new batched parts of b
    matvec_jaxpr_batched, matvec_b_bat = batching.batch_jaxpr(
        matvec, axis_size, matvec_bat + x_bat_out, instantiate=b_bat,
        axis_name=axis_name, main_type=main_type)
    if solve_t is None:
      solve_t_jaxpr_batched = None
      b_bat_out = _map(operator.or_, matvec_b_bat, orig_b_bat)
    else:
      solve_t_jaxpr_batched, solve_t_b_aux_bat = batching.batch_jaxpr(
          solve_t, axis_size, solve_t_bat + x_bat_out, instantiate=b_bat,
          axis_name=axis_name, main_type=main_type)
      assert len(solve_t_b_aux_bat) == len(orig_b_bat) + num_aux
      solve_t_b_bat, _ = split_list(solve_t_b_aux_bat, [len(orig_b_bat)])
      b_bat_out = _map(lambda m, s, o: m or s or o, matvec_b_bat, solve_t_b_bat,
                      orig_b_bat)
    if x_bat_out == x_bat and b_bat_out == b_bat:
      break
    else:
      x_bat = x_bat_out
      b_bat = b_bat_out
  else:
    assert False, "Fixedpoint not reached"

  batched_jaxprs = _LinearSolveTuple(matvec_jaxpr_batched, vecmat_jaxpr_batched,
                                     solve_jaxpr_batched, solve_t_jaxpr_batched)

  # Move batched axes to the front
  new_params = [
      batching.moveaxis(x, d, 0)
      if d is not batching.not_mapped and d != 0 else x
      for x, d in zip(_flatten(params), _flatten(params_dims))
  ]
  # Broadcast out b if necessary
  new_b = [
      batching.broadcast(x, axis_size, 0) if now_bat and not was_bat else
      batching.moveaxis(x, d, 0) if now_bat and d != 0 else x
      for x, d, was_bat, now_bat in zip(b, b_dims, orig_b_bat, b_bat)
  ]

  outs = linear_solve_p.bind(
      *(new_params + new_b),
      const_lengths=const_lengths,
      jaxprs=batched_jaxprs)
  out_dims = [0 if batched else batching.not_mapped for batched in solve_x_bat]
  return outs, out_dims


linear_solve_p = core.AxisPrimitive('custom_linear_solve')
linear_solve_p.multiple_results = True
linear_solve_p.def_impl(_custom_linear_solve_impl)
linear_solve_p.def_abstract_eval(_linear_solve_abstract_eval)
ad.primitive_jvps[linear_solve_p] = _custom_linear_solve_jvp
xla.register_translation(
    linear_solve_p, xla.lower_fun(_custom_linear_solve_impl, new_style=True,
                                  multiple_results=True),
    initial_style=True)
ad.primitive_transposes[linear_solve_p] = _linear_solve_transpose_rule
batching.axis_primitive_batchers[linear_solve_p] = _linear_solve_batching_rule
pe.partial_eval_jaxpr_custom_rules[linear_solve_p] = \
    partial(pe.partial_eval_jaxpr_custom_rule_not_implemented, 'linear_solve')


def _interleave(a, b, axis):
  """Given two Tensors of static shape, interleave them along the first axis."""
  assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
  a_pad = [(0, 0, 0)] * a.ndim
  b_pad = [(0, 0, 0)] * b.ndim
  a_pad[axis] = (0, 1 if a.shape[axis] == b.shape[axis] else 0, 1)
  b_pad[axis] = (1, 0 if a.shape[axis] == b.shape[axis] else 1, 1)
  op = lax.bitwise_or if a.dtype == np.bool_ else lax.add
  return op(lax.pad(a, lax._const(a, 0), a_pad),
            lax.pad(b, lax._const(b, 0), b_pad))

@api_boundary
def associative_scan(fn: Callable, elems, reverse: bool = False, axis: int = 0):
  """Performs a scan with an associative binary operation, in parallel.

  For an introduction to associative scans, see [BLE1990]_.

  Args:
    fn: A Python callable implementing an associative binary operation with
      signature ``r = fn(a, b)``. Function `fn` must be associative, i.e., it
      must satisfy the equation
      ``fn(a, fn(b, c)) == fn(fn(a, b), c)``.

      The inputs and result are (possibly nested Python tree structures of)
      array(s) matching ``elems``. Each array has a dimension in place
      of the ``axis`` dimension. `fn` should be applied elementwise over
      the ``axis`` dimension (for example, by using :func:`jax.vmap` over the
      elementwise function.)

      The result ``r`` has the same shape (and structure) as the two inputs
      ``a`` and ``b``.
    elems: A (possibly nested Python tree structure of) array(s), each with
      an ``axis`` dimension of size ``num_elems``.
    reverse: A boolean stating if the scan should be reversed with respect to
      the ``axis`` dimension.
    axis: an integer identifying the axis over which the scan should occur.

  Returns:
    A (possibly nested Python tree structure of) array(s) of the same shape
    and structure as ``elems``, in which the ``k``'th element of ``axis`` is the
    result of recursively applying ``fn`` to combine the first ``k`` elements
    of ``elems`` along ``axis``. For example, given ``elems = [a, b, c, ...]``,
    the result would be ``[a, fn(a, b), fn(fn(a, b), c), ...]``.

  Example 1: partial sums of an array of numbers:

  >>> lax.associative_scan(jnp.add, jnp.arange(0, 4))
  DeviceArray([0, 1, 3, 6], dtype=int32)

  Example 2: partial products of an array of matrices

  >>> mats = jax.random.uniform(jax.random.PRNGKey(0), (4, 2, 2))
  >>> partial_prods = lax.associative_scan(jnp.matmul, mats)
  >>> partial_prods.shape
  (4, 2, 2)

  Example 3: reversed partial sums of an array of numbers

  >>> lax.associative_scan(jnp.add, jnp.arange(0, 4), reverse=True)
  DeviceArray([6, 6, 5, 3], dtype=int32)

  .. [BLE1990] Blelloch, Guy E. 1990. "Prefix Sums and Their Applications.",
    Technical Report CMU-CS-90-190, School of Computer Science, Carnegie Mellon
    University.
  """
  elems_flat, tree = tree_flatten(elems)

  if reverse:
    elems_flat = [lax.rev(elem, [axis]) for elem in elems_flat]

  def combine(a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(tree, a_flat)
    b = tree_unflatten(tree, b_flat)
    c = fn(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat

  # Check that all inputs have a consistent leading dimension `num_elems`.
  axis = util.canonicalize_axis(axis, elems_flat[0].ndim)
  num_elems = int(elems_flat[0].shape[axis])
  if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
    raise ValueError('Array inputs to associative_scan must have the same '
                     'first dimension. (saw: {})'
                     .format([elem.shape for elem in elems_flat]))


  # Summary of algorithm:
  #
  # Consider elements of `_scan(elems)` at odd indices. That's the same as first
  # summing successive pairs of elements of `elems` and performing a scan on
  # that half sized tensor. We perform the latter scan by recursion.
  #
  # Now consider the even elements of `_scan(elems)`. These can be computed
  # from the odd elements of `_scan(elems)` by adding each odd element of
  # `_scan(elems)` to the matching even element in the original `elems`.
  #
  # We return the odd and even elements interleaved.
  #
  # For the base case of the recursion we return the first element
  # of `elems` followed by the sum of the first two elements computed as
  # a (small two-down-to-one) reduction step.
  def _scan(elems):
    """Perform scan on `elems`."""

    num_elems = elems[0].shape[axis]

    if num_elems < 2:
      return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(
      [slicing.slice_in_dim(elem, 0, -1, stride=2, axis=axis) for elem in elems],
      [slicing.slice_in_dim(elem, 1, None, stride=2, axis=axis)
       for elem in elems])

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(reduced_elems)

    if num_elems % 2 == 0:
      even_elems = combine(
        [slicing.slice_in_dim(e, 0, -1, axis=axis) for e in odd_elems],
        [slicing.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems])
    else:
      even_elems = combine(
        odd_elems,
        [slicing.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems])

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [
      lax.concatenate([slicing.slice_in_dim(elem, 0, 1, axis=axis), result],
                      dimension=axis)
      for (elem, result) in zip(elems, even_elems)]
    return list(_map(partial(_interleave, axis=axis), even_elems, odd_elems))

  scans = _scan(elems_flat)

  if reverse:
    scans = [lax.rev(scanned, [axis]) for scanned in scans]

  return tree_unflatten(tree, scans)


# Cumulative reductions.

def cumsum(operand: Array, axis: int = 0, reverse: bool = False) -> Array:
  """Computes a cumulative sum along `axis`."""
  return cumsum_p.bind(operand, axis=int(axis), reverse=bool(reverse))

def cumprod(operand: Array, axis: int = 0, reverse: bool = False) -> Array:
  """Computes a cumulative product along `axis`."""
  return cumprod_p.bind(operand, axis=int(axis), reverse=bool(reverse))

def cummax(operand: Array, axis: int = 0, reverse: bool = False) -> Array:
  """Computes a cumulative maximum along `axis`."""
  return cummax_p.bind(operand, axis=int(axis), reverse=bool(reverse))

def cummin(operand: Array, axis: int = 0, reverse: bool = False) -> Array:
  """Computes a cumulative minimum along `axis`."""
  return cummin_p.bind(operand, axis=int(axis), reverse=bool(reverse))

def _cumred_shape_rule(x, *, axis: int, reverse: bool):
  if axis < 0 or axis >= x.ndim:
    raise ValueError(
        "axis {} is out of bounds for array of shape {}".format(axis, x.shape))
  return x.shape

def _cumsum_transpose_rule(t, operand, *, axis: int, reverse: bool):
  return [cumsum(t, axis=axis, reverse=not reverse)]



def _cumred_tpu_translation_rule(window_reduce: Callable, x, *,
                                 axis: int, reverse: bool):
  # On TPU, an implementation using reduce_window is handled specially by the
  # compiler and is efficient. On other backends, it is O(n^2).
  n = x.shape[axis]
  if n == 0:
    return x
  padding = [(0, 0)] * x.ndim
  padding[axis] = (0, n - 1) if reverse else (n - 1, 0)
  strides = [1] * x.ndim
  window_dims = [1] * x.ndim
  window_dims[axis] = n
  return window_reduce(x, window_dims, strides, padding)

def _cumred_batch_rule(prim, batched_args, batch_dims, *, axis: int,
                       reverse: bool):
  operand, = batched_args
  bdim, = batch_dims
  axis = axis if axis < bdim else axis + 1
  return prim.bind(operand, axis=axis, reverse=reverse), bdim

def _cumred_dtype_rule(name, operand, *args, **kw):
  if not dtypes.issubdtype(operand.dtype, np.number):
    raise TypeError("{} does not accept dtype {}. Accepted dtypes are subtypes "
                    "of number.".format(name, np.dtype(operand.dtype).name))
  return dtypes.canonicalize_dtype(operand.dtype)


def _cumulative_reduction_primitive(name,
                                    reduce_fn,
                                    tpu_reduce_window_fn):
  reducer_p = lax.standard_primitive(
    _cumred_shape_rule, partial(_cumred_dtype_rule, name),
    name,
    translation_rule=xla.lower_fun(
      partial(associative_scan, reduce_fn),
      multiple_results=False, new_style=True))
  xla.register_translation(reducer_p, xla.lower_fun(
      partial(_cumred_tpu_translation_rule, tpu_reduce_window_fn),
      multiple_results=False, new_style=True), platform='tpu')
  batching.primitive_batchers[reducer_p] = partial(_cumred_batch_rule,
                                                   reducer_p)
  mlir.register_lowering(
      reducer_p,
      mlir.cache_lowering(
          mlir.lower_fun(partial(associative_scan, reduce_fn),
                         multiple_results=False)))
  mlir.register_lowering(
      reducer_p,
      mlir.lower_fun(partial(_cumred_tpu_translation_rule,
                             tpu_reduce_window_fn), multiple_results=False),
      platform='tpu')
  return reducer_p

cumsum_p = _cumulative_reduction_primitive("cumsum", lax.add, windowed_reductions._reduce_window_sum)
ad.deflinear2(cumsum_p, _cumsum_transpose_rule)
cumprod_p = _cumulative_reduction_primitive("cumprod", lax.mul, windowed_reductions._reduce_window_prod)
cummax_p = _cumulative_reduction_primitive("cummax", lax.max, windowed_reductions._reduce_window_max)
cummin_p = _cumulative_reduction_primitive("cummin", lax.min, windowed_reductions._reduce_window_min)


def _cumulative_jvp_rule(primals, tangents, *, axis: int, reverse: bool,
                         combine_fn: Callable):
  # Irrespective of backend, we always use the parallel prefix scan
  # implementation when differentiating because reduce_window is not
  # arbitrarily differentiable.
  return api.jvp(partial(associative_scan, combine_fn, axis=axis,
                         reverse=reverse),
                 primals, tangents)

ad.primitive_jvps[cumprod_p] = partial(_cumulative_jvp_rule, combine_fn=lax.mul)
ad.primitive_jvps[cummin_p] = partial(_cumulative_jvp_rule, combine_fn=lax.min)
ad.primitive_jvps[cummax_p] = partial(_cumulative_jvp_rule, combine_fn=lax.max)


def _dummy_remat_result(aval: core.AbstractValue):
  """A result that will be discarded"""
  if aval is core.abstract_token:
    return lax.create_token()
  elif aval is core.abstract_unit:
    return core.unit
  else:
    return lax.broadcast(np.array(0, dtype=aval.dtype), aval.shape)  # type: ignore

def _remat_translation_using_cond(*args,
                                  jaxpr: core.Jaxpr):
  # Implements:
  #  if(rng(0, 1) < 2)
  #    return eval_jaxpr(*args)
  #  else:
  #    return 0
  avals_out = tuple(ov.aval for ov in jaxpr.outvars)

  def remat_comp(*args):
    return tuple(core.eval_jaxpr(jaxpr, (), *args))
  def dummy_comp(*args):
    return tuple(_map(_dummy_remat_result, avals_out))

  cond_pred = (lax.rng_uniform(np.float32(0), np.float32(1), shape=()) < np.float32(2))
  return cond(cond_pred, remat_comp, dummy_comp, *args)

def _remat_translation_using_while(*args,
                                   jaxpr: core.Jaxpr):
  # Implements:
  #  for(counter=0, result=0; counter < rng(1, 2); counter ++) {
  #     result = eval_jaxpr(*args)
  #  }
  # The loop carry is a tuple: (counter, result, args)
  avals_out = tuple(ov.aval for ov in jaxpr.outvars)
  dummies_like_result = tuple(_map(_dummy_remat_result, avals_out))
  carry_init = (np.int32(0), dummies_like_result, args)
  def cond(carry):
    counter, _, _ = carry
    return counter < lax.rng_uniform(np.int32(1), np.int32(2), shape=())

  def body(carry):
    counter, _, args = carry
    results = core.eval_jaxpr(jaxpr, (), *args)
    return (counter + 1, tuple(results), args)

  carry_res = while_loop(cond, body, carry_init)
  return carry_res[1]


def _remat_translation_using_opt_barrier(*args, jaxpr: core.Jaxpr):
  args = _optimization_barrier(args)
  return core.eval_jaxpr(jaxpr, (), *args)


def _remat_translation_rule(*args,
                            call_jaxpr: Optional[core.Jaxpr] = None,
                            jaxpr: Optional[core.Jaxpr] = None,
                            platform: str,
                            prevent_cse: bool, differentiated: bool,
                            policy,
                            concrete: bool = False,
                            name: str = "checkpoint"):
  # Support either "jaxpr" (for remat2) and "call_jaxpr" (for remat)
  # name is not passed for remat2, defaults to "checkpoint"
  # TODO: remove call_jaxpr once we drop the remat call primitive
  if jaxpr is None:
    jaxpr = call_jaxpr
  assert jaxpr is not None
  assert not jaxpr.constvars

  del concrete, policy  # Unused.
  if differentiated and prevent_cse:
    if platform == "gpu":
      translation_rule = _remat_translation_using_while
    elif platform == "tpu" and config.jax_remat_opt_barrier:
      translation_rule = _remat_translation_using_opt_barrier
    else:
      translation_rule = _remat_translation_using_cond
  else:
    translation_rule = lambda *args, jaxpr: core.eval_jaxpr(jaxpr, (), *args)

  return jax.named_call(translation_rule, name=wrap_name(name, "remat"))(*args, jaxpr=jaxpr)

for platform in ("cpu", "gpu", "tpu"):
  for remat_primitive in (pe.remat_call_p, ad_checkpoint.remat_p):  # type: ignore
    xla.register_translation(remat_primitive,
                             xla.lower_fun(partial(_remat_translation_rule,
                                                   platform=platform),
                                           new_style=True, multiple_results=True,
                                           backend=platform),
                            platform=platform)
    mlir.register_lowering(remat_primitive,
                           mlir.lower_fun(partial(_remat_translation_rule,
                                                   platform=platform),
                                          multiple_results=True),
                           platform=platform)


def _optimization_barrier_abstract_eval(*args):
  return args


def _optimization_barrier_translation_rule(ctx, avals_in, avals_out, *args):
  out = xops.OptimizationBarrier(xops.Tuple(ctx.builder, args))
  return [xops.GetTupleElement(out, i) for i in range(len(args))]


def _optimization_barrier_lowering_rule(ctx, *args):
  barrier_types = _map(mlir.aval_to_ir_types, ctx.avals_in)
  flat_barrier_types = util.flatten(barrier_types)

  flat_args = mlir.flatten_lowering_ir_args(args)
  barrier_op = mhlo.OptimizationBarrierOp(flat_barrier_types, flat_args)
  return util.unflatten(barrier_op.results, _map(len, barrier_types))


def _optimization_barrier(arg):
  flat_args, treedef = tree_flatten(arg)
  return tree_unflatten(treedef, optimization_barrier_p.bind(*flat_args))


optimization_barrier_p = core.Primitive('optimization_barrier')
optimization_barrier_p.multiple_results = True
optimization_barrier_p.def_impl(
    partial(xla.apply_primitive, optimization_barrier_p))
optimization_barrier_p.def_abstract_eval(_optimization_barrier_abstract_eval)
xla.register_translation(optimization_barrier_p,
                         _optimization_barrier_translation_rule)
mlir.register_lowering(optimization_barrier_p,
                       _optimization_barrier_lowering_rule)
