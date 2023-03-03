# Copyright 2018 The JAX Authors.
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
from __future__ import annotations

from collections import namedtuple
import contextlib
import functools
from functools import partial
import inspect
import itertools as it
import operator as op
from typing import (Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple,
                    List, Union, Hashable, Set, cast)
from weakref import ref

import numpy as np

from jax._src import linear_util as lu
from jax.config import config
from jax._src import api_util
from jax._src import core
from jax._src import effects
from jax._src import dtypes
from jax._src import profiler
from jax._src import source_info_util
from jax._src.api_util import flattened_fun_in_tree, flatten_fun_nokwargs
from jax._src.core import (Trace, Tracer, Jaxpr, Literal, get_aval,
                           AbstractValue, ClosedJaxpr, new_jaxpr_eqn,
                           ConcreteArray, Var, DropVar,
                           raise_to_shaped, Atom, JaxprEqn, Primitive,
                           ShapedArray, DShapedArray, mapped_aval,
                           unmapped_aval, DBIdx, InDBIdx, OutDBIdx,
                           InputType, OutputType, get_referent)
from jax._src.tree_util import (PyTreeDef, treedef_tuple, tree_unflatten,
                                tree_leaves)
from jax._src.util import (unzip2, safe_zip, safe_map, toposort, split_list,
                           merge_lists, partition_list, OrderedSet,
                           as_hashable_function, weakref_lru_cache)


map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
def identity(x): return x

TracerId = int
AvalId = int
ConstId = int

def _update_annotation_known(
    f: lu.WrappedFun,
    orig_type: Optional[InputType],
    in_knowns: List[bool]
  ) -> lu.WrappedFun:
  if orig_type is None: return f
  # orig_type might contain DBIdx, but we're tossing out some args so we have to
  # re-index. moreover some of the implicit args may not be needed anymore.
  # so we basically just re-infer the lambda input type
  if (all(e for _, e in orig_type) and
      not any(type(d) is DBIdx for a, _ in orig_type for d in a.shape
              if type(a) is DShapedArray)):
    new_type = [ty for ty, known in zip(orig_type, in_knowns) if known]
    return lu.annotate(f, tuple(new_type))

  # Replace DBIdx with names, prune down to explicit only.
  class Name:
    def __init__(self, a): self.a = a
  names = [Name(a) for a, _  in orig_type]
  avals = [a.update(shape=tuple(names[d.val] if type(d) is DBIdx else d  # type: ignore
                                for d in a.shape))
           if type(a) is DShapedArray else a for a, e in orig_type if e]
  avals = [a for a, known in zip(avals, in_knowns) if known]
  # Figure out the implicit part: names which aren't explicit and known.
  expl_names = [o for o, (_, e) in zip(names, orig_type) if e]
  expl_names = [o for o, k in zip(expl_names, in_knowns) if k]
  expl_names_ = set(expl_names)
  impl_names = {d for a in avals if type(a) is DShapedArray for d in a.shape
                if type(d) is Name and d not in expl_names_}
  impl_part = [(n.a, False) for n in impl_names]  # type: ignore
  # Figure out the explicit part: known explicit avals, replacing names w/ dbidx
  name_map = {n: DBIdx(i) for i, n in enumerate((*impl_names, *expl_names))}
  expl_part = [(a.update(shape=tuple(name_map.get(d, d) for d in a.shape))
                if type(a) is DShapedArray else a, True) for a in avals]
  return lu.annotate(f, (*impl_part, *expl_part))

class PartialVal(tuple):
  """Partial value: either a known value or an unknown (abstract) value.

  Represented as a pair `(aval_opt, const)` of one of two kinds:
  * `(None, <Constant>)` indicates a known value, where the constant is either a
    Tracer or satisfies `core.valid_jaxtype(const)`;
  * `(<AbstractValue>, None)` indicates an unknown value characterized by an
    abstract value.
  """
  def __new__(cls, xs: Tuple[Optional[AbstractValue], core.Value]):
    pv, const = xs
    if config.jax_enable_checks:
      # type checks
      assert isinstance(pv, (AbstractValue, type(None))), xs
      assert (const is None or isinstance(const, core.Tracer) or
              core.valid_jaxtype(const)), const
      # invariant checks
      assert (pv is None) ^ (const is None)
    return tuple.__new__(cls, xs)

  @classmethod
  def known(cls, const: core.Value) -> PartialVal:
    return PartialVal((None, const))

  @classmethod
  def unknown(cls, aval: AbstractValue) -> PartialVal:
    return PartialVal((aval, None))

  def is_known(self) -> bool:
    return self[0] is None

  def get_known(self) -> Optional[core.Value]:
    """Get the known value, if known, else None."""
    return self[1] if self[0] is None else None

  def get_aval(self) -> AbstractValue:
    """Get AbstractValue directly (if unknown) or from the constant (known)."""
    known = self.get_known()
    if known is not None:
      return get_aval(known)
    else:
      return self[0]


class JaxprTrace(Trace):

  def __init__(self, *args, name_stack: source_info_util.NameStack):
    super().__init__(*args)
    self.name_stack = name_stack

  def pure(self, val: Any) -> JaxprTracer:
    return self.new_const(val)

  def lift(self, val: Tracer) -> JaxprTracer:
    return self.new_const(val)

  def sublift(self, val: JaxprTracer) -> JaxprTracer:
    return JaxprTracer(self, val.pval, FreeVar(val))

  def new_const(self, val) -> JaxprTracer:
    if isinstance(val, Tracer) and val._trace.level == self.level:
      raise Exception
    return JaxprTracer(self, PartialVal.known(val), None)

  def new_instantiated_literal(self, val) -> JaxprTracer:
    aval = get_aval(val)
    return JaxprTracer(self, PartialVal.unknown(aval),
                       Literal(val, raise_to_shaped(aval)))

  def new_instantiated_const(self, val) -> JaxprTracer:
    aval = get_aval(val)
    if isinstance(aval, DShapedArray):
      shape = [self.new_instantiated_const(d)
               if isinstance(d, Tracer) and d._trace.level < self.level else d
               for d in aval.shape]
      aval = aval.update(shape=tuple(shape))
    return JaxprTracer(self, PartialVal.unknown(aval), ConstVar(val))

  def new_arg(self, pval: PartialVal) -> JaxprTracer:
    const = pval.get_known()
    # XXX: Think twice before changing this constant argument pruning!
    # This has really important consequences for partial_eval_jaxpr.
    # Most importantly, this guarantees that the unknown jaxpr never uses
    # known inputs (if it needs them, then they get passed through residuals).
    if const is None:
      aval = pval.get_aval()
      if type(aval) is DShapedArray:
        shape = [self.new_instantiated_const(d)
                 if isinstance(d, Tracer) and d._trace.level < self.level else d
                 for d in aval.shape]
        aval = aval.update(shape=tuple(shape))
      return JaxprTracer(self, PartialVal.unknown(aval), LambdaBinding())
    else:
      return self.new_const(const)

  def instantiate_const(self, tracer) -> Tracer:
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      if type(const) in core.literalable_types and np.shape(const) == ():
        return self.new_instantiated_literal(const)
      else:
        return self.new_instantiated_const(const)

  def instantiate_const_abstracted(self, tracer) -> JaxprTracer:
    const = tracer.pval.get_known()
    if const is None:
      return tracer
    else:
      aval = raise_to_shaped(get_aval(const), np.isscalar(const))
      return JaxprTracer(self, PartialVal.unknown(aval), ConstVar(const))

  def process_primitive(self, primitive, tracers, params):
    if primitive in custom_partial_eval_rules:
      return custom_partial_eval_rules[primitive](self, *tracers, **params)
    else:
      return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    # By default, if all the input tracers are known, then bind the primitive
    # and consider all outputs known. Otherwise, stage the application into the
    # jaxpr and consider all outputs unknown.
    consts = [t.pval.get_known() for t in tracers]
    if all(c is not None for c in consts):
      return primitive.bind(*consts, **params)
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval, effects = primitive.abstract_eval(*avals, **params)
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    if primitive.multiple_results:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in out_aval]
      eqn = new_eqn_recipe(tracers, out_tracers, primitive, params, effects, source)
      for t in out_tracers: t.recipe = eqn
      return out_tracers
    else:
      out_tracer = JaxprTracer(self, PartialVal.unknown(out_aval), None)
      out_tracer.recipe = new_eqn_recipe(tracers, [out_tracer], primitive,
                                         params, effects, source)
      return out_tracer

  def process_call(self, primitive, f, tracers, params):
    rule = call_partial_eval_rules.get(primitive)
    if rule:
      return rule(self, primitive, f, tracers, params)

    update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
    in_knowns, in_avals, in_consts = partition_pvals([t.pval for t in tracers])
    # TODO(mattjj): check in_avals are consistent with f.in_type

    # We want to partially evaluate this call into two calls: one evaluated now
    # taking known values (in_consts) as inputs and producing known values
    # (out_consts) as outputs, and the other staged out as an eqn into the jaxpr
    # being built. The latter takes as input residuals (res) produced as outputs
    # of the first call, shared closed-over values (env), and explicit arguments
    # which were unknown to the first call (corresponding to in_avals).

    # Wrap f to perform the partial evaluation and plumb out aux data.
    if not config.jax_dynamic_shapes:
      f_ = trace_to_subjaxpr_nounits_fwd(f, self.main, False)
      f_, aux = partial_eval_wrapper_nounits(f_, tuple(in_knowns),
                                             tuple(in_avals))
    else:
      if f.in_type is None:
        f = lu.annotate(f, tuple((a, True) for a in in_avals))
      f_, aux = trace_to_subjaxpr_nounits_dyn(f, self.main, tuple(in_knowns),
                                              f.in_type, False)
    # Adjust parameters (e.g. donated_invars) for the call to be evaluated now.
    const_params = update_params(params, in_knowns, 0)

    # Run the call, getting known out vals and aux data used for staged-out call
    out = primitive.bind(_update_annotation_known(f_, f.in_type, in_knowns),
                         *in_consts, **const_params)
    fwds, out_knowns, out_type, jaxpr, env = aux()
    # Split apart known outputs from the original call and non-fwded residuals.
    out_consts, non_fwd_res_ = split_list(out, [sum(out_knowns)])

    # Form the complete list of residuals by forwarding some inputs.
    if config.jax_dynamic_shapes:
      # With dynamic shapes, we may need to forward implicit arguments.
      in_consts_, in_knowns_ = iter(in_consts), iter(in_knowns)
      in_consts_full = [None] * len(f.in_type)
      for idx, (aval, explicit) in enumerate(f.in_type):
        if explicit and next(in_knowns_):
          c = in_consts_full[idx] = next(in_consts_)
          if aval.shape:
            for d1, d2 in zip(aval.shape, c.shape):
              if type(d1) is DBIdx:
                in_consts_full[d1.val] = d2
    else:
      in_consts_full = in_consts
    non_fwd_res = iter(non_fwd_res_)
    res = [next(non_fwd_res) if i is None else in_consts_full[i] for i in fwds]
    sentinel = object()
    assert next(non_fwd_res, sentinel) is sentinel

    # Create the input tracers for the staged-out (unknown-value) call.
    res_tracers = map(self.instantiate_const, map(self.new_const, res))
    env_tracers = map(self.full_raise, env)
    unknown_arg_tracers = [t for t in tracers if not t.is_known()]
    # Adjust parameters (e.g. donated_invars) for the staged-out call's args.
    num_new_args = len(res_tracers) + len(env_tracers)
    staged_params = dict(params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
    staged_params = update_params(staged_params, map(op.not_, in_knowns),
                                  num_new_args)
    # The outputs of the staged-out call are Tracers with the new eqn as recipe.
    if config.jax_dynamic_shapes:
      # With dynamic shapes, we may need to substitute Tracers into avals.
      out_tracers = []
      for aval, _ in out_type:
        assert not isinstance(aval, ConcreteArray)
        if type(aval) is DShapedArray:
          shape = [[*res_tracers, *env_tracers, *unknown_arg_tracers][d.val]
                  if type(d) is InDBIdx else d for d in aval.shape]
          aval = aval.update(shape=tuple(shape))
        out_tracers.append(JaxprTracer(self, PartialVal.unknown(aval), None))
    else:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(a), None)
                     for a in out_type]
    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    eqn = new_eqn_recipe((*res_tracers, *env_tracers, *unknown_arg_tracers),
                         out_tracers, primitive, staged_params, jaxpr.effects,
                         source)
    for t in out_tracers: t.recipe = eqn
    return merge_lists(out_knowns, out_tracers, out_consts)

  def process_map(self, primitive, f: lu.WrappedFun, tracers, params):
    update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
    in_knowns, in_avals, in_consts = partition_pvals([t.pval for t in tracers])

    # This method is like process_call above, except:
    #   1. we delete an axis from mapped-over input avals' shapes, and
    #      analogously add an axis to mapped-over output avals' shapes;
    #   2. we update the in_axes and out_axes/out_axes_thunk parameters to
    #      reflect the inputs and outputs pruned from the unknown/known sides.

    # Map (delete an axis from) unknown inputs' avals as dictated by in_axes.
    unk_in_axes, const_in_axes = partition_list(in_knowns, params['in_axes'])
    in_avals_mapped = [mapped_aval(params['axis_size'], ax, aval)
                       for ax, aval in zip(unk_in_axes, in_avals)]

    # Wrap f to perform partial evaluation and plumb out aux data.
    f = trace_to_subjaxpr_nounits(f, self.main, False)
    f, aux = partial_eval_wrapper_nounits(f, tuple(in_knowns),
                                          tuple(in_avals_mapped))
    # Adjust params for knowns (e.g. donated_invars, in_axes, out_axes_thunk)
    const_params = update_params(params, in_knowns, 0)  # handles donated_invars
    out_axes_thunk = params['out_axes_thunk']
    @as_hashable_function(closure=out_axes_thunk)
    def const_out_axes_thunk():
      out_knowns, _, jaxpr, _ = aux()
      _, out_axes = partition_list(out_knowns, out_axes_thunk())
      return tuple(out_axes) + (0,) * len(jaxpr.constvars)  # res mapped axis 0
    const_params = dict(const_params, in_axes=tuple(const_in_axes),
                        out_axes_thunk=const_out_axes_thunk)

    # Run the map, getting known out vals and aux data used for staged-out map.
    out = primitive.bind(f, *in_consts, **const_params)
    out_knowns, out_avals_mapped, jaxpr, env = aux()
    # Split apart known outputs from the original call and residuals.
    out_consts, res = split_list(out, [len(out) - len(jaxpr.constvars)])

    # We can only check_jaxpr with the dynamic axis environment extended:
    with core.extend_axis_env(params['axis_name'], params['axis_size'], None):
      call_jaxpr = convert_constvars_jaxpr(jaxpr)

    # Compute staged and const out_axes, taking into account residuals.
    out_axes = params['out_axes_thunk']()
    staged_out_axes, _ = partition_list(out_knowns, out_axes)
    staged_in_axes = (0,) * len(res) + (None,) * len(env) + (*unk_in_axes,)

    # Create the input tracers for the staged-out (unkonwn-value) call.
    const_tracers = map(self.new_instantiated_const, res)
    env_tracers = map(self.full_raise, env)
    unknown_arg_tracers = [t for t in tracers if not t.is_known()]
    # Adjust params for staged-out call on unknown values.
    num_new_args = len(const_tracers) + len(env_tracers)
    staged_params = update_params(params, map(op.not_, in_knowns), num_new_args)
    staged_params = dict(staged_params, in_axes=staged_in_axes,
                         out_axes=tuple(staged_out_axes), call_jaxpr=call_jaxpr)
    del staged_params['out_axes_thunk']
    # The outputs of the staged-out call are Tracers with the new eqn as recipe.
    out_avals = [unmapped_aval(params['axis_size'], params['axis_name'], ax, a)
                 for ax, a in zip(staged_out_axes, out_avals_mapped)]
    out_tracers = [JaxprTracer(self, PartialVal.unknown(a), None)
                   for a in out_avals]
    eqn = new_eqn_recipe((*const_tracers, *env_tracers, *unknown_arg_tracers),  # type: ignore[arg-type]
                         out_tracers, primitive, staged_params,
                         jaxpr.effects,
                         source_info_util.current())
    for t in out_tracers: t.recipe = eqn

    return merge_lists(out_knowns, out_tracers, out_consts)

  def post_process_call(self, primitive, out_tracers, params):
    unknown_out_tracers = [t for t in out_tracers if not t.is_known()]
    jaxpr, res, env = tracers_to_jaxpr([], unknown_out_tracers)
    out_pvals = [t.pval for t in out_tracers]
    out_knowns, out_avals, out_consts = partition_pvals(out_pvals)
    out = [*out_consts, *res]
    main = self.main

    def todo(out):
      trace = main.with_cur_sublevel()
      out_consts, res = split_list(out, [len(out) - len(jaxpr.constvars)])
      const_tracers = map(trace.new_instantiated_const, res)
      in_tracers = (*const_tracers, *map(trace.full_raise, env))
      out_tracers = [JaxprTracer(trace, PartialVal.unknown(a), None)
                     for a in out_avals]
      update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
      new_params = update_params(params, [], len(in_tracers))
      new_params = dict(new_params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
      name_stack = self._current_truncated_name_stack()
      source = source_info_util.current().replace(name_stack=name_stack)
      eqn = new_eqn_recipe(in_tracers, out_tracers, primitive, new_params,
          jaxpr.effects, source)
      for t in out_tracers: t.recipe = eqn
      return merge_lists(out_knowns, out_tracers, out_consts)

    return out, todo

  def post_process_map(self, primitive, out_tracers, params):
    unknown_out_tracers = [t for t in out_tracers if not t.is_known()]
    jaxpr, res, env = tracers_to_jaxpr([], unknown_out_tracers)
    out_pvals = [t.pval for t in out_tracers]
    out_knowns, out_avals_mapped, out_consts = partition_pvals(out_pvals)
    out = [*out_consts, *res]
    main = self.main

    with core.extend_axis_env(params['axis_name'], params['axis_size'], None):
      call_jaxpr = convert_constvars_jaxpr(jaxpr)

    def todo(out):
      trace = main.with_cur_sublevel()
      out_consts, res = split_list(out, [len(out) - len(jaxpr.constvars)])
      const_tracers = map(trace.new_instantiated_const, res)
      env_tracers = map(trace.full_raise, env)

      staged_out_axes = tuple(out_axes_unknown)  # set by out_axes_transform
      staged_in_axes = (0,) * len(res) + (None,) * len(env)

      update_params = call_param_updaters.get(primitive) or (lambda p, _, __: p)
      staged_params = update_params(params, [], len(res) + len(env))
      staged_params = dict(staged_params, in_axes=staged_in_axes,
                           out_axes=tuple(staged_out_axes),
                           call_jaxpr=call_jaxpr)

      out_avals = [unmapped_aval(params['axis_size'], params['axis_name'], d, a)
                   for d, a in zip(staged_out_axes, out_avals_mapped)]
      out_tracers = [JaxprTracer(trace, PartialVal.unknown(a), None)
                     for a in out_avals]
      name_stack = self._current_truncated_name_stack()
      source = source_info_util.current().replace(name_stack=name_stack)
      eqn = new_eqn_recipe((*const_tracers, *env_tracers), out_tracers,
                           primitive, staged_params, jaxpr.effects, source)
      for t in out_tracers: t.recipe = eqn
      return merge_lists(out_knowns, out_tracers, out_consts)

    def out_axes_transform(out_axes):
      nonlocal out_axes_unknown
      out_axes_unknown, out_axes_known = partition_list(out_knowns, out_axes)
      return tuple(out_axes_known) + (0,) * len(jaxpr.constvars)
    out_axes_unknown: Optional[list] = None

    return out, (todo, out_axes_transform)

  def _current_truncated_name_stack(self):
    return source_info_util.current_name_stack()[len(self.name_stack):]

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros):
    # We assume partial evaluation is only performed to build linear functions,
    # and hence we don't need to keep the custom JVP rule around anymore.
    del jvp, symbolic_zeros
    assert not all(t.is_known() for t in tracers)
    return fun.call_wrapped(*tracers)

  def post_process_custom_jvp_call(self, out_tracers, _):
    # This path should only be reachable if we expose a partial eval API
    # unrelated to autodiff, since we raise an error when differentiation with
    # respect to values over which a custom_jvp function closes is detected.
    raise NotImplementedError  # TODO(mattjj)

  def process_custom_transpose(self, prim, call, tracers, **params):
    res_ts, lin_ts = split_list(tracers, [params['res_tree'].num_leaves])
    assert all(t.is_known()     for t in res_ts)
    lin_all_known   = all(t.is_known()     for t in lin_ts)
    if lin_all_known:
      res_cvals = [t.pval[1] for t in res_ts]
      lin_cvals = [t.pval[1] for t in lin_ts]
      return prim.bind(call, *res_cvals, *lin_cvals, **params)
    else:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in params['out_types']]
      in_tracers = map(self.instantiate_const, tracers)
      new_params = dict(params, call=call)
      eqn = new_eqn_recipe(in_tracers, out_tracers, prim, new_params,
          core.no_effects, source_info_util.current())
      for t in out_tracers: t.recipe = eqn
      return out_tracers

  def process_custom_vjp_call(self, prim, f, fwd, bwd, tracers, out_trees):
    # TODO(mattjj): after old remat is deleted, make this method trivial.
    # Because we instantiate all tracers, in_knowns is all False.
    tracers = map(self.instantiate_const_abstracted, tracers)
    in_knowns, in_avals, () = partition_pvals([t.pval for t in tracers])
    f = trace_to_subjaxpr_nounits(f, self.main, True)
    f, aux = partial_eval_wrapper_nounits(f, tuple(in_knowns), tuple(in_avals))
    out_flat = prim.bind(f, fwd, bwd, out_trees=out_trees)
    out_knowns, out_avals, jaxpr, env = aux()
    out_consts, res = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    res_tracers = map(self.new_instantiated_const, res)
    env_tracers = map(self.full_raise, env)
    out_tracers = [JaxprTracer(self, PartialVal.unknown(a), None)
                   for a in out_avals]
    closed_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ())

    @_memoize
    def fwd_jaxpr_thunk():
      fwd_ = trace_to_subjaxpr_nounits(fwd, self.main, True)
      fwd_, aux = partial_eval_wrapper_nounits(
          fwd_, tuple(in_knowns), tuple(in_avals))
      with core.new_sublevel():
        out_flat = fwd_.call_wrapped()
      out_knowns, out_avals, jaxpr, env = aux()
      _, res = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
      converted_jaxpr = convert_envvars_to_constvars(jaxpr, len(env))
      return converted_jaxpr, (*res, *env)

    name_stack = self._current_truncated_name_stack()
    source = source_info_util.current().replace(name_stack=name_stack)
    eqn = new_eqn_recipe((*res_tracers, *env_tracers, *tracers),
                         out_tracers, prim.initial_style,
                         dict(fun_jaxpr=closed_jaxpr,
                              fwd_jaxpr_thunk=fwd_jaxpr_thunk,
                              num_consts=len(res) + len(env),
                              bwd=bwd, out_trees=out_trees),
                         jaxpr.effects, source)
    for t in out_tracers: t.recipe = eqn
    return merge_lists(out_knowns, out_tracers, out_consts)

  def post_process_custom_vjp_call(self, out_tracers, _):
    # This path should only be reachable if we expose a partial eval API
    # unrelated to autodiff, since we raise an error when differentiation with
    # respect to values over which a custom_vjp function closes is detected.
    raise NotImplementedError  # TODO(mattjj)

def partition_pvals(
    pvals: List[PartialVal]
  ) -> Tuple[List[bool], List[AbstractValue], List[Any]]:
  knowns = [pval.is_known()  for pval in pvals                       ]
  avals  = [pval.get_aval()  for pval in pvals if not pval.is_known()]
  consts = [pval.get_known() for pval in pvals if     pval.is_known()]
  return knowns, avals, consts

@lu.transformation_with_aux
def partial_eval_wrapper_nounits(
    in_knowns: Sequence[bool], in_avals: Sequence[AbstractValue],
    *in_consts: Any):
  in_avals_, in_consts_ = iter(in_avals), iter(in_consts)
  in_pvals = [PartialVal.known(next(in_consts_)) if known else
              PartialVal.unknown(next(in_avals_)) for known in in_knowns]
  sentinel = object()
  assert next(in_avals_, sentinel) is next(in_consts_, sentinel) is sentinel
  jaxpr, (*maybe_fwds, out_pvals, res, env) = yield (in_pvals,), {}
  out_knowns, out_avals, out_consts = partition_pvals(out_pvals)
  yield (*out_consts, *res), (*maybe_fwds, out_knowns, out_avals, jaxpr, env)

@lu.transformation_with_aux
def trace_to_subjaxpr_nounits_dyn(
    main: core.MainTrace, in_knowns: Sequence[bool], in_type: InputType,
    instantiate: Union[bool, Sequence[bool]],
    *in_consts: Any):
  trace = main.with_cur_sublevel()
  in_avals, which_explicit = unzip2(in_type)

  # To form input tracers from in_type, we need to first build ConstVar tracers
  # for all axis sizes, so that we can then use those tracers in the shapes of
  # avals for unknown inputs' tracers. We use ConstVar recipes for on-the-fly
  # type agreement checking via get_referent.
  in_consts_full: List[Optional[JaxprTracer]] = [None] * len(in_type)
  in_consts_iter, in_knowns_iter = iter(in_consts), iter(in_knowns)
  for idx, (aval, explicit) in enumerate(in_type):
    if explicit and next(in_knowns_iter):
      constval = next(in_consts_iter)
      if isinstance(aval, DShapedArray):
        for i, d in enumerate(aval.shape):
          if isinstance(d, DBIdx):
            if in_consts_full[d.val] is None:
              in_consts_full[d.val] = \
                  JaxprTracer(trace, PartialVal.unknown(in_avals[d.val]),
                              ConstVar(constval.shape[i]))
            assert core.same_referent(constval.shape[i], in_consts_full[d.val])
        shape = [in_consts_full[d.val] if type(d) is DBIdx else d  # type: ignore
                 for d in aval.shape]
        aval = aval.update(shape=tuple(shape))
      in_consts_full[idx] = JaxprTracer(trace, PartialVal.unknown(aval),
                                        ConstVar(constval))
  # Check that we covered all axis sizes with ConstVar tracers.
  for idx, (aval, explicit) in enumerate(in_type):
    if not explicit: assert in_consts_full[idx] is not None
    if isinstance(aval, DShapedArray):
      assert all(type(d) is not DBIdx or in_consts_full[d.val] is not None  # type: ignore
                 for d in aval.shape)

  # Next, build tracers for all unknown inputs, using the in_consts_full list
  # for axis size tracers when necessary.
  in_tracers = []
  in_knowns_iter = iter(in_knowns)
  for aval, explicit in in_type:
    if explicit and not next(in_knowns_iter):
      if isinstance(aval, DShapedArray):
        shape = [in_consts_full[d.val] if type(d) is DBIdx else d  # type: ignore
                 for d in aval.shape]
        aval = aval.update(shape=tuple(shape))
      tracer = JaxprTracer(trace, PartialVal.unknown(aval), LambdaBinding())
      in_tracers.append(tracer)

  # Merge in_consts and in_tracers and call wrapped fn with  explicit arguments.
  in_args = merge_lists(in_knowns, in_tracers, in_consts)
  ans = yield in_args, {}

  # Instantiate outputs and build jaxpr.
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(ans)
  out_tracers = map(trace.full_raise, map(core.full_lower, ans))
  out_tracers = [trace.instantiate_const(trace.full_raise(t)) if inst else t
                 for inst, t in zip(instantiate, out_tracers)]

  # Collect known outputs.
  out_knowns: List[bool] = [t.is_known() for t in out_tracers]
  out_consts: List[Any] = [t.pval.get_known() for t in out_tracers
                           if t.is_known()]

  # Build the jaxpr.
  out_tracers = [t for t in out_tracers if not t.is_known()]
  jaxpr, res, env = tracers_to_jaxpr(in_tracers, out_tracers)
  out_avals = [v.aval for v in jaxpr.outvars]
  idx_map = {v: InDBIdx(i)
             for i, v in enumerate(it.chain(jaxpr.constvars, jaxpr.invars))}
  out_type = [(a.update(shape=tuple(idx_map.get(d, d) for d in a.shape))  # type: ignore
               if type(a) is DShapedArray else a, True) for a in out_avals]

  # Which residuals are just forwarded inputs? Check obj id, then prune.
  id_map = {id(c.recipe.val): i for i, c in enumerate(in_consts_full)  # type: ignore
            if c is not None}
  fwds: List[Optional[int]] = [id_map.get(id(c)) for c in res]
  res = tuple([c for c, fwd in zip(res, fwds) if fwd is None])

  del main, in_consts, trace, in_consts_iter, in_knowns_iter, in_consts_full, \
      in_tracers, in_args, ans, out_tracers, out_avals
  yield (*out_consts, *res), (fwds, out_knowns, tuple(out_type), jaxpr, env)


custom_partial_eval_rules: Dict[Primitive, Callable] = {}
call_partial_eval_rules: Dict[Primitive, Callable] = {}
call_param_updaters: Dict[Primitive, Callable] = {}

def _closed_call_param_updater(params, _, __):
  jaxpr = params.get('call_jaxpr')
  if jaxpr is None: return params
  assert type(jaxpr) is core.Jaxpr
  return dict(params, call_jaxpr=core.ClosedJaxpr(jaxpr, ()))
call_param_updaters[core.closed_call_p] = _closed_call_param_updater

def abstract_eval_fun(fun, *avals, debug_info=None, **params):
  _, avals_out, _ = trace_to_jaxpr_dynamic(
      lu.wrap_init(fun, params), avals, debug_info)
  assert all(isinstance(aval, AbstractValue) for aval in avals_out)
  return avals_out


JaxprTracerRecipe = Union['JaxprEqnRecipe', 'LambdaBinding', 'FreeVar',
                          'ConstVar', Literal]

class JaxprTracer(Tracer):
  __slots__ = ['pval', 'recipe']

  def __init__(self, trace: JaxprTrace, pval: PartialVal,
               recipe: Optional[JaxprTracerRecipe]):
    assert isinstance(pval, PartialVal)
    pv, const = pval
    if isinstance(const, Tracer) and const._trace.level >= trace.level:
      raise core.escaped_tracer_error(
          const, f"Tracer from a higher level: {const} in trace {trace}")
    if isinstance(pv, DShapedArray):
      assert all(not isinstance(d, Tracer) or isinstance(d, JaxprTracer) and
                 d._trace.level == trace.level for d in pv.shape)
    self._trace = trace
    self.pval = pval
    self.recipe = recipe

  def __repr__(self):
    return f'Traced<{self.aval}:{self._trace}>'

  @property
  def aval(self) -> AbstractValue:
    return self.pval.get_aval()

  @property
  def parents(self) -> Sequence[JaxprTracer]:
    if isinstance(self.recipe, JaxprEqnRecipe):
      # TODO broadcast_in_dim can create a new tracer...
      return self.recipe.in_tracers
    elif isinstance(self.aval, DShapedArray):
      return [d for d in self.aval.shape if isinstance(d, JaxprTracer)]
    else:
      return []

  def full_lower(self):
    known = self.pval.get_known()
    if known is not None:
      return core.full_lower(known)
    else:
      return self

  def is_known(self):
    return self.pval.is_known()

  def get_referent(self):
    if self.pval.is_known():
      return get_referent(self.pval.get_known())
    elif isinstance(self.recipe, (FreeVar, ConstVar, Literal)):
      return get_referent(self.recipe.val)  # pytype: disable=attribute-error
    else:
      return self


@profiler.annotate_function
def trace_to_jaxpr(
    fun: lu.WrappedFun, pvals: Sequence[PartialVal],
    instantiate: Union[bool, Sequence[bool]] = False,
  ) -> Tuple[Jaxpr, List[PartialVal], List[core.Value]]:
  """
  Partially evaluate a function, building a jaxpr for un-evaluated computation.

  Args:
    fun: lu.WrappedFun representing the function to be partially evaluated. The
      function must be flattened, in the sense of accepting jaxpr type arguments
      and returning a flat list of jaxpr type outputs.
    pvals: sequence of PartialVals of length equal to the number of inputs to
      `fun` indicating which inputs are known or unknown.
    instantiate: optional bool or sequence of bools of length equal to the
      number of outputs of `fun` indicating which outputs should be forced to be
      treated as unknown and hence instantiated in the jaxpr. If a single bool,
      the value is applied to all outputs. Default False.

  Returns:
    A triple where the first element is a jaxpr representing the computation
    which depends on unknown inputs; the second element is a list of PartialVals
    of length equal to the length of the output of `fun` representing which
    outputs are known and unknown (along with their values and abstract values,
    respectively); the third element is a list of known residual values. The
    returned jaxpr takes as inputs the known residual values followed by values
    of the originally unknown inputs.
  """
  current_name_stack = source_info_util.current_name_stack()
  with core.new_main(JaxprTrace, name_stack=current_name_stack) as main:
    fun = trace_to_subjaxpr(fun, main, instantiate)
    jaxpr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
    assert not env
    del main, fun, env

  return jaxpr, out_pvals, consts

@profiler.annotate_function
def trace_to_jaxpr_nounits(
    fun: lu.WrappedFun, pvals: Sequence[PartialVal],
    instantiate: Union[bool, Sequence[bool]] = False,
  ) -> Tuple[Jaxpr, List[PartialVal], List[core.Value]]:
  current_name_stack = source_info_util.current_name_stack()
  with core.new_main(JaxprTrace, name_stack=current_name_stack) as main:
    fun = trace_to_subjaxpr_nounits(fun, main, instantiate)
    jaxpr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
    assert not env
    del main, fun, env
  return jaxpr, out_pvals, consts


@lu.transformation
def trace_to_subjaxpr_nounits(
    main: core.MainTrace,
    instantiate: Union[bool, Sequence[bool]],
    in_pvals: Sequence[PartialVal]):
  assert all([isinstance(pv, PartialVal) for pv in in_pvals]), in_pvals
  out_tracers, jaxpr, out_consts, env = yield from _trace_to_subjaxpr_nounits(
      main, instantiate, in_pvals)
  out_pvals = [t.pval for t in out_tracers]
  del out_tracers
  yield jaxpr, (out_pvals, out_consts, env)

def _trace_to_subjaxpr_nounits(main, instantiate, in_pvals):
  trace = main.with_cur_sublevel()
  in_knowns  = [pval.is_known()     for pval in in_pvals]
  in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
  in_tracers = [trace.new_arg(pval) for pval in in_pvals if not pval.is_known()]
  in_args = merge_lists(in_knowns, in_tracers, in_consts)
  ans = yield in_args, {}
  assert isinstance(ans, (list, tuple)), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  assert all(isinstance(x, core.Tracer) or core.valid_jaxtype(x) for x in ans), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(ans)
  out_tracers = map(trace.full_raise, map(core.full_lower, ans))
  out_tracers = [trace.instantiate_const(trace.full_raise(t)) if inst else t
                 for inst, t in zip(instantiate, out_tracers)]
  out_tracers_ = [t for t in out_tracers if not t.is_known()]
  jaxpr, out_consts, env = tracers_to_jaxpr(in_tracers, out_tracers_)
  return out_tracers, jaxpr, out_consts, env

# The below variant implements an optimization where residuals which are also
# inputs are indicated in auxiliary data rather than passed as outputs.
# TODO(mattjj): update all callers to use this version, delete other version.
@lu.transformation
def trace_to_subjaxpr_nounits_fwd(
    main: core.MainTrace,
    instantiate: Union[bool, Sequence[bool]],
    in_pvals: Sequence[PartialVal]):
  assert all([isinstance(pv, PartialVal) for pv in in_pvals]), in_pvals
  out_tracers, jaxpr, out_consts, env = yield from _trace_to_subjaxpr_nounits(
      main, instantiate, in_pvals)
  out_pvals = [t.pval for t in out_tracers]

  # Which out_consts (aka residuals) are just forwarded inputs? Check obj id.
  in_consts  = [pval.get_known()    for pval in in_pvals if     pval.is_known()]
  id_map = {id(c): i for i, c in enumerate(in_consts)}
  fwds: List[Optional[int]] = [id_map.get(id(c)) for c in out_consts]
  pruned_consts = [c for c, fwd in zip(out_consts, fwds) if fwd is None]

  del out_tracers
  yield jaxpr, (fwds, out_pvals, pruned_consts, env)


FreeVar = namedtuple('FreeVar', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])
LambdaBinding = namedtuple('LambdaBinding', [])
class JaxprEqnRecipe(NamedTuple):
  eqn_id: Any
  in_tracers: Sequence[JaxprTracer]
  out_tracer_refs: Sequence[ref[JaxprTracer]]
  out_avals: Sequence[core.AbstractValue]
  primitive: Primitive
  params: Dict[str, Any]
  effects: core.Effects
  source_info: source_info_util.SourceInfo

def new_eqn_recipe(in_tracers: Sequence[JaxprTracer],
                   out_tracers: Sequence[JaxprTracer],
                   primitive: Primitive,
                   params: Dict[str, Any],
                   effects: core.Effects,
                   source_info: source_info_util.SourceInfo
                  ) -> JaxprEqnRecipe:
  # TODO(necula): move these checks to core.check_jaxpr, and call in more places
  if primitive.call_primitive or primitive.map_primitive:
    assert "call_jaxpr" in params
    assert ("donated_invars" not in params or
            len(params["donated_invars"]) == len(params["call_jaxpr"].invars))
  if primitive.map_primitive:
    assert ("in_axes" in params and
            len(params["in_axes"]) == len(params["call_jaxpr"].invars))
    assert ("donated_invars" in params and
            len(params["donated_invars"]) == len(params["call_jaxpr"].invars))
  out_avals = [core.raise_to_shaped(t.aval) for t in out_tracers]
  return JaxprEqnRecipe(object(), tuple(in_tracers), map(ref, out_tracers),
                        out_avals, primitive, params, effects, source_info)


def recipe_to_eqn(getvar: Callable[[JaxprTracer], Atom],
                  recipe: JaxprEqnRecipe) -> core.JaxprEqn:
  (_, in_tracers, out_tracer_refs, out_avals, prim, params, eff, src) = recipe
  invars  = [getvar(t) for t in in_tracers]
  out_tracers = [t_ref() for t_ref in out_tracer_refs]
  outvars = [DropVar(a) if t is None else getvar(t)  # type: ignore
             for a, t in zip(out_avals, out_tracers)]
  return new_jaxpr_eqn(invars, outvars, prim, params, eff, src)

def tracers_to_jaxpr(
  in_tracers: Sequence[JaxprTracer],
  out_tracers: Sequence[JaxprTracer]
  ) -> Tuple[Jaxpr, Tuple[Any, ...], Tuple[Any, ...]]:
  """Constructs Jaxpr given tracers for inputs and outputs.

  Params:
    in_tracers: the tracers that were created for the function inputs
    out_tracers: the tracers that were output by the function.

  Returns: a triple of a `Jaxpr`, a list of constant values corresponding to
    the `constvars` in the returned Jaxps, and a list of environment values.
    The vars for the environment values have been prepended to the Jaxpr's
    `invars`.
  """
  gensym = core.gensym()

  t_to_var: Dict[TracerId, Var] = {}
  consts: Dict[Var, Any] = {}
  env: Dict[Var, JaxprTracer] = {}
  constid_to_var: Dict[ConstId, Var] = {}  # for deduplication

  def get_atom(t: JaxprTracer) -> Atom:
    return t.recipe if type(t.recipe) is Literal else t_to_var[id(t)]

  def newvar(t: Optional[JaxprTracer]) -> Var:
    assert t is not None
    var = gensym(type_substitute(t.aval))
    var_ = t_to_var.setdefault(id(t), var)
    assert var is var_
    return var

  def type_substitute(aval: AbstractValue) -> AbstractValue:
    if isinstance(aval, DShapedArray):
      # Replace any Tracers in aval.shape with Vars or Literal values
      shape = [get_atom(d) if type(d) is JaxprTracer else d for d in aval.shape]
      shape = [d.val if type(d) is Literal else d for d in shape]
      aval = aval.update(shape=tuple(shape))
    return aval

  processed_eqn_ids = set()
  eqns: List[core.JaxprEqn] = []
  for t in toposort([*in_tracers, *out_tracers]):
    r = t.recipe
    if isinstance(r, JaxprEqnRecipe):
      # TODO broadcast_in_dim can create a new tracer, not present in parents
      if r.eqn_id not in processed_eqn_ids:
        in_atoms = map(get_atom, r.in_tracers)
        outvars = [DropVar(type_substitute(a)) if rf() is None else newvar(rf())
                   for a, rf in zip(r.out_avals, r.out_tracer_refs)]
        eqns.append(new_jaxpr_eqn(in_atoms, outvars, r.primitive, r.params,
                                  r.effects, r.source_info))
        processed_eqn_ids.add(r.eqn_id)
    elif isinstance(r, LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise core.escaped_tracer_error(t, f"Tracer not in input tracers: {t}")
      newvar(t)
    elif isinstance(r, ConstVar):
      var = constid_to_var.get(id(r.val))
      if var is None:
        var = constid_to_var[id(r.val)] = newvar(t)
        consts[var] = r.val
      t_to_var[id(t)] = var
    elif isinstance(r, FreeVar):
      env[newvar(t)] = r.val  # type: ignore
    elif isinstance(r, Literal):
      pass
    elif r is None:
      assert False
    else:
      raise TypeError(r)

  env_vars, env_vals = unzip2(env.items())
  invars = [*env_vars, *map(get_atom, in_tracers)]
  const_vars, const_vals = unzip2(consts.items())
  outvars = map(get_atom, out_tracers)  # type: ignore[arg-type]
  jaxpr_effects = make_jaxpr_effects(const_vars, invars, outvars, eqns)
  jaxpr = Jaxpr(const_vars, invars,  # type: ignore[list-item,arg-type]
                outvars, eqns, jaxpr_effects)
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  # del getvar  # needed to avoid cyclic-reference closure, apparently!
  return jaxpr, const_vals, env_vals

@weakref_lru_cache
def convert_constvars_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
  """Moves the constvars to the start of invars."""
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  lifted_jaxpr = Jaxpr(constvars=(),
                       invars=jaxpr.constvars + jaxpr.invars,
                       outvars=jaxpr.outvars, eqns=jaxpr.eqns,
                       effects=jaxpr.effects)
  config.jax_enable_checks and core.check_jaxpr(lifted_jaxpr)
  return lifted_jaxpr

@weakref_lru_cache
def convert_invars_to_constvars(jaxpr: Jaxpr, n: int) -> Jaxpr:
  """Move n invars to constvars. Like an inverse of convert_constvars_Jaxpr."""
  if any(isinstance(eff, effects.JaxprInputEffect) for eff in jaxpr.effects):
    raise NotImplementedError
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  constvars, invars = split_list(jaxpr.invars, [n])
  lifted_jaxpr = Jaxpr(constvars=tuple(constvars), invars=invars,
                       outvars=jaxpr.outvars, eqns=jaxpr.eqns,
                       effects=jaxpr.effects)
  config.jax_enable_checks and core.check_jaxpr(lifted_jaxpr)
  return lifted_jaxpr

def convert_envvars_to_constvars(jaxpr: Jaxpr, num_env_vars: int) -> Jaxpr:
  if any(isinstance(eff, effects.JaxprInputEffect) for eff in jaxpr.effects):
    raise NotImplementedError
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  env_vars, invars = split_list(jaxpr.invars, [num_env_vars])
  converted_jaxpr = Jaxpr(constvars=jaxpr.constvars + env_vars,
                          invars=invars, outvars=jaxpr.outvars, eqns=jaxpr.eqns,
                          effects=jaxpr.effects)
  config.jax_enable_checks and core.check_jaxpr(converted_jaxpr)
  return converted_jaxpr


def partial_eval_jaxpr_nounits(
    jaxpr: ClosedJaxpr, unknowns: Sequence[bool],
    instantiate: Union[bool, Sequence[bool]],
  ) -> Tuple[ClosedJaxpr, ClosedJaxpr, List[bool], List[AbstractValue]]:
  """Unzip a jaxpr in two by data dependence into 'known' and 'unknown' parts.

  That is, given a jaxpr and a sequence of booleans indicating which jaxpr
  inputs (i.e. invars) are considered unknown, produce two jaxprs, a list of
  booleans representing which of the original jaxpr's outputs are unknown (i.e.
  have a data dependence on an unknown input), and a list of abstract values
  representing residuals (part of the first jaxpr's output and the second
  jaxpr's input). The two jaxprs result from partitioning the original jaxpr's
  first-order primitive applications based on whether all the inputs to the
  application are known (in which case the application is represented in the
  'known' jaxpr and its result is considered known) or whether any inputs to the
  application are unknown (in which case the application is represented in the
  'unknown' jaxpr and its result is considered unknown). Higher-order primitives
  are recursively unzipped in two.

  The `instantiate` argument can be used to ensure some outputs are lifted into
  the 'unknown' jaxpr.

  For example, give an input jaxpr:

    { lambda ; a:f32[] b:f32[]. let
        c:f32[] = cos a
        d:f32[] = sin a
        e:f32[] = neg d
        f:f32[] = mul e b
      in (c, f) }

  then applying this function with `unknowns=[False, True]` and
  `instantiate=False` produces as an output triple:

    # jaxpr_known
    { lambda ; a:f32[]. let
       b:f32[] = cos a
       c:f32[] = sin a
       d:f32[] = neg c
     in (b, d) }

    # jaxpr_unknown
    { lambda ; a:f32[] b:f32[]. let c:f32[] = mul b a in (c,) }

    # out_unknowns
    [False, True]

  Notice in particular that the first output (jaxpr_known) contains all the
  primitive applications which do not have a data dependence on an unknown
  input. Also notice the input and output types: the input type of the first
  jaxpr produced represents the type of the known inputs of the original jaxpr,
  and the output type of the second jaxpr produced represents the type of the
  unknown outputs of the original jaxpr.

  In the above example, the output of jaxpr_known named `d` is a _residual_
  output, and corresponds to the input named `a` in jaxpr_unknown. In general,
  jaxpr_known will produce extra outputs (at the end of its output list)
  corresponding to intermeidate values of the original jaxpr which must be
  passed to jaxpr_unknown (as leading inputs).
  """
  instantiate = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  return _partial_eval_jaxpr_nounits(jaxpr, tuple(unknowns), instantiate)

@weakref_lru_cache
def _partial_eval_jaxpr_nounits(jaxpr, in_unknowns, instantiate):
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))

  cell = []
  def fun(*known_vals_in):
    known_vals_in = iter(known_vals_in)
    unknown_avals = (a for a, uk in zip(jaxpr.in_avals, in_unknowns) if uk)
    in_pvals = [PartialVal.unknown(next(unknown_avals)) if uk
                else PartialVal.known(next(known_vals_in)) for uk in in_unknowns]
    assert next(known_vals_in, None) is next(unknown_avals, None) is None
    jaxpr_unknown_, out_pvals, residuals = trace_to_jaxpr_nounits(
        f, in_pvals, instantiate=instantiate)
    jaxpr_unknown = convert_constvars_jaxpr(jaxpr_unknown_)
    out_unknowns = [not pval.is_known() for pval in out_pvals]
    res_avals = [core.raise_to_shaped(core.get_aval(r)) for r in residuals]
    cell.append((out_unknowns, jaxpr_unknown, res_avals))
    known_vals_out = [pval.get_known() for pval in out_pvals if pval.is_known()]
    return [*known_vals_out, *residuals]

  known_avals = [a for a, uk in zip(jaxpr.in_avals, in_unknowns) if not uk]
  jaxpr_known, _, consts_known = trace_to_jaxpr_dynamic(lu.wrap_init(fun), known_avals)
  (out_unknowns, jaxpr_unknown, res_avals), = cell  # pytype: disable=bad-unpacking

  # check jaxpr_known and jaxpr_unknown in isolation
  # TODO(mattjj): enable weak type checking here
  if config.jax_enable_checks:
    core.check_jaxpr(jaxpr_known)
    core.check_jaxpr(jaxpr_unknown)
  # check jaxpr_known has input type corresponding to known inputs of jaxpr
  assert ([v.aval for v in jaxpr_known.invars] ==
          [a for a, uk in zip(jaxpr.in_avals, in_unknowns) if not uk])
  # check jaxpr_known has out type corresponding to known outs of jaxpr plus res
  assert ([v.aval.strip_weak_type() for v in jaxpr_known.outvars] ==
          [a.strip_weak_type() for a, uk in zip(jaxpr.out_avals, out_unknowns)
           if not uk] + [a.strip_weak_type() for a in res_avals])
  # check jaxpr_unknown has input type corresponding to res plus unknown inputs
  assert ([v.aval.strip_weak_type() for v in jaxpr_unknown.invars] ==
          [a.strip_weak_type() for a in res_avals] +
          [a.strip_weak_type() for a, uk in zip(jaxpr.in_avals, in_unknowns)
           if uk])
  # check jaxpr_unknown has output type corresponding to unknown outputs
  assert ([v.aval.strip_weak_type() for v in jaxpr_unknown.outvars] ==
          [a.strip_weak_type() for a, uk in zip(jaxpr.out_avals, out_unknowns)
           if uk])

  closed_jaxpr_known = ClosedJaxpr(jaxpr_known, consts_known)
  closed_jaxpr_unknown = ClosedJaxpr(jaxpr_unknown, ())
  return closed_jaxpr_known, closed_jaxpr_unknown, out_unknowns, res_avals


def partial_eval_jaxpr_custom(
    jaxpr: Jaxpr,
    in_unknowns: Sequence[bool],
    in_inst: Union[bool, Sequence[bool]],
    ensure_out_unknowns: Union[bool, Sequence[bool]],
    ensure_out_inst: Union[bool, Sequence[bool]],
    saveable: Callable[..., bool],
  ) -> Tuple[Jaxpr, Jaxpr, List[bool], List[bool], int]:
  if type(in_inst) is bool:
    in_inst = (in_inst,) * len(jaxpr.invars)
  if type(ensure_out_unknowns) is bool:
    ensure_out_unknowns = (ensure_out_unknowns,) * len(jaxpr.outvars)
  if type(ensure_out_inst) is bool:
    ensure_out_inst = (ensure_out_inst,) * len(jaxpr.outvars)
  return _partial_eval_jaxpr_custom_cached(
      jaxpr, tuple(in_unknowns), tuple(in_inst), tuple(ensure_out_unknowns),
      tuple(ensure_out_inst), saveable)

@weakref_lru_cache
def _partial_eval_jaxpr_custom_cached(
    jaxpr: Jaxpr,
    in_unknowns: Tuple[bool, ...],
    in_inst: Tuple[bool, ...],
    ensure_out_unknowns: Tuple[bool, ...],
    ensure_out_inst: Tuple[bool, ...],
    saveable: Callable[..., bool],
  ) -> Tuple[Jaxpr, Jaxpr, List[bool], List[bool], int]:
  env: Dict[Var, Tuple[bool, bool]] = {}
  residuals: OrderedSet[Var] = OrderedSet()

  def read(x: Atom) -> Tuple[bool, bool]:
    if type(x) is Var:
      return env[x]
    return (False, True)

  def write(unk: bool, inst: bool, v: Var) -> None:
    assert (unk, inst) != (True, False)
    env[v] = (unk, inst)

  def ensure_instantiated(inst: bool, x: Atom) -> Atom:
    if type(x) is Var and not inst:
      residuals.add(x)
    return x

  known_eqns, staged_eqns = [], []
  map(write, in_unknowns, in_inst, jaxpr.invars)
  map(partial(write, False, True), jaxpr.constvars)
  for eqn in jaxpr.eqns:
    unks_in, inst_in = unzip2(map(read, eqn.invars))
    rule = partial_eval_jaxpr_custom_rules.get(eqn.primitive)
    if rule:
      eqn1, eqn2, unks_out, inst_out, res = rule(saveable, unks_in, inst_in, eqn)
      eqn1 and known_eqns.append(eqn1); eqn2 and staged_eqns.append(eqn2)  # type: ignore
      residuals.update(res)
      map(write, unks_out, inst_out, eqn.outvars)
    elif any(unks_in):
      inputs = map(ensure_instantiated, inst_in, eqn.invars)
      staged_eqns.append(eqn.replace(invars=inputs))
      map(partial(write, True, True), eqn.outvars)
    else:
      known_eqns.append(eqn)
      # If it's an effectful primitive, we always to run and avoid staging it.
      if eqn.effects or saveable(
          eqn.primitive, *[x.aval for x in eqn.invars], **eqn.params):
        map(partial(write, False, False), eqn.outvars)
      else:
        inputs = map(ensure_instantiated, inst_in, eqn.invars)
        staged_eqns.append(eqn.replace(invars=inputs))
        map(partial(write, False, True), eqn.outvars)
  unzipped = unzip2(map(read, jaxpr.outvars))
  out_unknowns, out_inst = list(unzipped[0]), list(unzipped[1])
  assert all(type(v) is Var for v in residuals), residuals

  for x, inst, ensure_inst in zip(jaxpr.outvars, out_inst, ensure_out_inst):
    if ensure_inst: ensure_instantiated(inst, x)
  out_unknowns = map(op.or_, out_unknowns, ensure_out_unknowns)
  out_inst     = map(op.or_, out_inst,     ensure_out_inst)

  ins_known, _ = partition_list(in_unknowns, jaxpr.invars)
  outs_known, _ = partition_list(out_unknowns, jaxpr.outvars)
  known_outvars = [*outs_known, *residuals]
  known_effects = make_jaxpr_effects(jaxpr.constvars, ins_known, known_outvars,
                                      known_eqns)
  jaxpr_known = Jaxpr(jaxpr.constvars, ins_known, known_outvars,
                      known_eqns, known_effects)
  config.jax_enable_checks and core.check_jaxpr(jaxpr_known)

  _, ins_staged = partition_list(in_inst, jaxpr.invars)
  _, outs_staged = partition_list(out_inst, jaxpr.outvars)
  staged_effects = core.join_effects(*(eqn.effects for eqn in staged_eqns))
  staged_invars = [*residuals, *ins_staged]
  staged_effects = make_jaxpr_effects(jaxpr.constvars, staged_invars,
                                       outs_staged, staged_eqns)
  jaxpr_staged = Jaxpr(jaxpr.constvars, staged_invars,
                       outs_staged, staged_eqns, staged_effects)
  config.jax_enable_checks and core.check_jaxpr(jaxpr_staged)

  return jaxpr_known, jaxpr_staged, out_unknowns, out_inst, len(residuals)

# A primitive rule for policy-driven partial evaluation returns a 5-tuple
# with the components representing, respectively:
#  * the JaxprEqn for the 'known' side (or None if there is no known component),
#  * the JaxprEqn for the 'unknown' side (or None),
#  * a list of booleans indicating which of the original outputs are unknown,
#  * a list of booleans indicating which of the original outputs are
#    instantiated (i.e. available) in the 'unknown' side,
#  * a list of Var instances representing residuals to be added (i.e. to be
#    plumbed as outputs of the 'known' side jaxpr and added as input binders to
#    the 'unknown' jaxpr).
PartialEvalCustomResult = Tuple[Optional[JaxprEqn], Optional[JaxprEqn],
                                Sequence[bool], Sequence[bool], List[Var]]
PartialEvalCustomRule = Callable[
    [Callable[..., bool], Sequence[bool], Sequence[bool], JaxprEqn],
    PartialEvalCustomResult]
partial_eval_jaxpr_custom_rules: Dict[Primitive, PartialEvalCustomRule] = {}

def partial_eval_jaxpr_custom_rule_not_implemented(
    name: str, saveable: Callable[..., bool], unks_in: Sequence[bool],
    inst_in: Sequence[bool], eqn: JaxprEqn) -> PartialEvalCustomResult:
  msg = (f'custom-policy remat rule not implemented for {name}, '
         'open a feature request at https://github.com/google/jax/issues!')
  raise NotImplementedError(msg)


ParamsUpdater = Callable[[Sequence[bool], Sequence[bool], Sequence[bool],
                          Sequence[bool], int, dict, dict],
                         Tuple[dict, dict]]
ResAvalUpdater = Callable[[Dict[str, Any], AbstractValue], AbstractValue]
def _default_res_aval_updater(
    params: Dict[str, Any], aval: AbstractValue) -> AbstractValue:
  return aval

def call_partial_eval_custom_rule(
    jaxpr_param_name: str, params_updater: ParamsUpdater,
    saveable: Callable[..., bool], unks_in: List[bool], inst_in: List[bool],
    eqn: JaxprEqn, *, res_aval: ResAvalUpdater = _default_res_aval_updater,
  ) -> Tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], List[Var]]:
  jaxpr = eqn.params[jaxpr_param_name]
  jaxpr_known, jaxpr_staged, unks_out, inst_out, num_res = \
      partial_eval_jaxpr_custom(jaxpr, unks_in, inst_in, False, False, saveable)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  _, ins_staged = partition_list(inst_in, eqn.invars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  newvar = core.gensym([jaxpr_known, jaxpr_staged])
  params_known = {**eqn.params, jaxpr_param_name: jaxpr_known}
  params_staged = {**eqn.params, jaxpr_param_name: jaxpr_staged}
  params_known, params_staged = params_updater(
      unks_in, inst_in, map(op.not_, unks_out), inst_out, num_res, params_known,
      params_staged)
  residuals = [newvar(res_aval(params_known, var.aval))
               for var in jaxpr_staged.invars[:num_res]]
  eqn_known = new_jaxpr_eqn(ins_known, [*out_binders_known, *residuals],
                            eqn.primitive, params_known, jaxpr_known.effects, eqn.source_info)
  eqn_staged = new_jaxpr_eqn([*residuals, *ins_staged], out_binders_staged,
                             eqn.primitive, params_staged,
                             jaxpr_staged.effects, eqn.source_info)
  assert len(eqn_staged.invars) == len(jaxpr_staged.invars)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  return eqn_known, eqn_staged, unks_out, inst_out, new_inst + residuals

def closed_call_partial_eval_custom_rule(
    jaxpr_param_name: str, params_updater: ParamsUpdater,
    saveable: Callable[..., bool], unks_in: List[bool], inst_in: List[bool],
    eqn: JaxprEqn, *, res_aval: ResAvalUpdater = _default_res_aval_updater,
  ) -> Tuple[JaxprEqn, JaxprEqn, Sequence[bool], Sequence[bool], List[Var]]:
  # TODO(sharadmv,mattjj): dedup this rule with call_partial_eval_custom_rule.
  closed_jaxpr = eqn.params[jaxpr_param_name]
  jaxpr_known_, jaxpr_staged_, unks_out, inst_out, num_res = \
      partial_eval_jaxpr_custom(closed_jaxpr.jaxpr, unks_in, inst_in,
                                False, False, saveable)
  # Forming these fresh ClosedJaxprs defeats caching, but caller handles caching
  jaxpr_known = core.ClosedJaxpr(jaxpr_known_, closed_jaxpr.consts)
  jaxpr_staged = core.ClosedJaxpr(jaxpr_staged_, closed_jaxpr.consts)
  ins_known, _ = partition_list(unks_in, eqn.invars)
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  _, ins_staged = partition_list(inst_in, eqn.invars)
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  newvar = core.gensym([jaxpr_known.jaxpr, jaxpr_staged.jaxpr])
  params_known = {**eqn.params, jaxpr_param_name: jaxpr_known}
  params_staged = {**eqn.params, jaxpr_param_name: jaxpr_staged}
  params_known, params_staged = params_updater(
      unks_in, inst_in, map(op.not_, unks_out), inst_out, num_res, params_known,
      params_staged)
  residuals = [newvar(res_aval(params_known, a))
               for a in jaxpr_staged.in_avals[:num_res]]
  eqn_known = new_jaxpr_eqn(ins_known, [*out_binders_known, *residuals],
                            eqn.primitive, params_known, jaxpr_known.effects,
                            eqn.source_info)
  eqn_staged = new_jaxpr_eqn([*residuals, *ins_staged], out_binders_staged,
                             eqn.primitive, params_staged, jaxpr_staged.effects,
                             eqn.source_info)
  assert len(eqn_staged.invars) == len(jaxpr_staged.in_avals)
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is Var and not inst]
  return eqn_known, eqn_staged, unks_out, inst_out, new_inst + residuals

partial_eval_jaxpr_custom_rules[core.call_p] = \
    partial(call_partial_eval_custom_rule, 'call_jaxpr',
            lambda _, __, ___, ____, _____, x, y: (x, y))
partial_eval_jaxpr_custom_rules[core.closed_call_p] = \
    partial(closed_call_partial_eval_custom_rule, 'call_jaxpr',
            lambda _, __, ___, ____, _____, x, y: (x, y))


def _jaxpr_forwarding(jaxpr: Jaxpr) -> List[Optional[int]]:
  # Compute which inputs are just forwarded to outputs.
  fwds: Dict[Var, Var] = dict(zip(jaxpr.invars, jaxpr.invars))
  for eqn in jaxpr.eqns:
    if eqn.primitive in forwarding_rules:
      eqn = eqn.replace(invars=[a if type(a) is Literal else fwds.get(a, a)  # type: ignore
                                for a in eqn.invars])
      fwd_vars, _ = forwarding_rules[eqn.primitive](eqn)
      for v_orig, v_new in zip(eqn.outvars, fwd_vars):
        if v_new is not None:
          fwds[v_orig] = v_new
  idxs: Dict[Var, int] = {v: i for i, v in enumerate(jaxpr.invars)}
  return [None if type(v) is Literal else idxs.get(fwds.get(v))  # type: ignore
          for v in jaxpr.outvars]


def dce_jaxpr(jaxpr: Jaxpr, used_outputs: Sequence[bool],
              instantiate: Union[bool, Sequence[bool]] = False,
              ) -> Tuple[Jaxpr, List[bool]]:
  if type(instantiate) is bool:
    instantiate = (instantiate,) * len(jaxpr.invars)
  return _dce_jaxpr(jaxpr, tuple(used_outputs), tuple(instantiate))


def dce_jaxpr_consts(jaxpr: Jaxpr, used_outputs: Sequence[bool],
                     instantiate: Union[bool, Sequence[bool]] = False,
                     ) -> Tuple[Jaxpr, List[bool], List[bool]]:
  jaxpr_ = convert_constvars_jaxpr(jaxpr)
  new_jaxpr_, used_inputs_ = dce_jaxpr(jaxpr_, used_outputs)
  used_consts, used_inputs = split_list(used_inputs_, [len(jaxpr.constvars)])
  new_jaxpr = convert_invars_to_constvars(new_jaxpr_, sum(used_consts))
  return new_jaxpr, used_consts, used_inputs


@weakref_lru_cache
def _dce_jaxpr(jaxpr: Jaxpr, used_outputs: Tuple[bool, ...],
               instantiate: Tuple[bool, ...]
               ) -> Tuple[Jaxpr, List[bool]]:
  env: Dict[Var, bool] = {}

  def read(v: Var) -> bool:
    return env.get(v, False)

  def write(x: Atom, b: bool) -> None:
    if type(x) is Var:
      env[x] = read(x) or b

  def has_effects(e: JaxprEqn) -> bool:
    return bool(e.effects) or core.primitive_uses_outfeed(e.primitive, e.params)

  new_eqns = []
  map(write, jaxpr.outvars, used_outputs)
  for eqn in jaxpr.eqns[::-1]:
    used_outs = map(read, eqn.outvars)
    if not any(used_outs) and not has_effects(eqn):
      used_ins = [False] * len(eqn.invars)
    else:
      rule = dce_rules.get(eqn.primitive, _default_dce_rule)
      used_ins, new_eqn = rule(used_outs, eqn)
      if new_eqn is not None:
        new_eqns.append(new_eqn)
    map(write, eqn.invars, used_ins)
  used_inputs = map(read, jaxpr.invars)
  used_inputs = map(op.or_, instantiate, used_inputs)

  invars = [v for v, b in zip(jaxpr.invars, used_inputs)   if b]
  outvars = [v for v, b in zip(jaxpr.outvars, used_outputs) if b]
  eqns = new_eqns[::-1]
  jaxpr_effects = make_jaxpr_effects(jaxpr.constvars, invars, outvars, eqns)

  new_jaxpr = Jaxpr(jaxpr.constvars, invars, outvars, eqns, jaxpr_effects)
  config.jax_enable_checks and core.check_jaxpr(new_jaxpr)

  return new_jaxpr, used_inputs

DCERule = Callable[[List[bool], JaxprEqn], Tuple[List[bool], Optional[JaxprEqn]]]

def _default_dce_rule(
    used_outs: List[bool], eqn: JaxprEqn
  ) -> Tuple[List[bool], JaxprEqn]:
  return [True] * len(eqn.invars), eqn

dce_rules: Dict[Primitive, DCERule] = {}


def dce_jaxpr_call_rule(used_outputs: List[bool], eqn: JaxprEqn
                        ) -> Tuple[List[bool], Optional[JaxprEqn]]:
  new_jaxpr, used_inputs = dce_jaxpr(eqn.params['call_jaxpr'], used_outputs)
  new_params = dict(eqn.params, call_jaxpr=new_jaxpr)
  update_params = call_param_updaters.get(eqn.primitive)
  if update_params:
    new_params = update_params(new_params, used_inputs, 0)
  if not any(used_inputs) and not any(used_outputs) and not new_jaxpr.effects:
    return used_inputs, None
  else:
    new_eqn = new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, new_jaxpr.effects, eqn.source_info)
    return used_inputs, new_eqn
dce_rules[core.call_p] = dce_jaxpr_call_rule


def dce_jaxpr_closed_call_rule(used_outputs: List[bool], eqn: JaxprEqn
                               ) -> Tuple[List[bool], JaxprEqn]:
  # TODO(mattjj): de-duplicate with above rule?
  jaxpr_ = eqn.params['call_jaxpr']
  jaxpr, consts = jaxpr_.jaxpr, jaxpr_.consts
  new_jaxpr, used_inputs = dce_jaxpr(jaxpr, used_outputs)
  new_params = dict(eqn.params, call_jaxpr=core.ClosedJaxpr(new_jaxpr, consts))
  new_eqn = new_jaxpr_eqn(
      [v for v, used in zip(eqn.invars, used_inputs) if used],
      [v for v, used in zip(eqn.outvars, used_outputs) if used],
      eqn.primitive, new_params, new_jaxpr.effects, eqn.source_info)
  return used_inputs, new_eqn
dce_rules[core.closed_call_p] = dce_jaxpr_closed_call_rule

@weakref_lru_cache
def close_jaxpr(jaxpr: Jaxpr) -> ClosedJaxpr:
  return ClosedJaxpr(jaxpr, ())

def move_binders_to_front(closed_jaxpr: ClosedJaxpr, to_move: Sequence[bool]
                          ) -> ClosedJaxpr:
  """Reorder `invars` by moving those indicated in `to_move` to the front."""
  return _move_binders_to_front(closed_jaxpr, tuple(to_move))

@weakref_lru_cache
def _move_binders_to_front(closed_jaxpr: ClosedJaxpr, to_move: Tuple[bool, ...]
                           ) -> ClosedJaxpr:
  assert len(closed_jaxpr.in_avals) == len(to_move)
  new_invars = _move_to_front(closed_jaxpr.jaxpr.invars, to_move)
  new_jaxpr = Jaxpr(closed_jaxpr.jaxpr.constvars, new_invars,
                    closed_jaxpr.jaxpr.outvars, closed_jaxpr.jaxpr.eqns,
                    closed_jaxpr.jaxpr.effects)
  new_closed_jaxpr = core.ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
  return new_closed_jaxpr

def _move_to_front(lst: Sequence, to_move: Sequence[bool]) -> Sequence:
  return ([elt for elt, move in zip(lst, to_move) if move] +
          [elt for elt, move in zip(lst, to_move) if not move])

def move_binders_to_back(closed_jaxpr: ClosedJaxpr, to_move: Sequence[bool]
                         ) -> ClosedJaxpr:
  """Reorder `invars` by moving those indicated in `to_move` to the back."""
  return move_binders_to_front(closed_jaxpr, map(op.not_, to_move))

class DynamicJaxprTracer(core.Tracer):
  __slots__ = ['aval', '_debug_info']

  def __init__(self, trace, aval, line_info=None):
    self._trace = trace
    self._line_info = line_info
    # Needed for UnexpectedTracerError.
    self._debug_info = self._trace.frame.debug_info
    self.aval = aval

  def full_lower(self):
    return self

  def _contents(self):
    return ()

  def _origin_msg(self):
    if not self._trace.main.jaxpr_stack:  # type: ignore
      # If this Tracer has been leaked the jaxpr stack may no longer be
      # available. So we can't print as much origin information.
      return ("\nThis DynamicJaxprTracer was created on line "
              f"{source_info_util.summarize(self._line_info)}")
    else:
      invar_pos, progenitor_eqns = self._trace.frame.find_progenitors(self)
    dbg = self._debug_info
    if dbg is None:
      return ""

    origin = (f"The error occurred while tracing the function {dbg.func_src_info} "
              f"for {dbg.traced_for}. ")
    if invar_pos:
      origin += ("This concrete value was not available in Python because it "
                 f"depends on the value{'s' if len(invar_pos) > 1 else ''} "
                 f"of {dbg.arg_info(invar_pos)}.")
    elif progenitor_eqns:
      msts = ["  operation "
              f"{core.pp_eqn(eqn, core.JaxprPpContext(), core.JaxprPpSettings(print_shapes=True))}\n"
              f"    from line {source_info_util.summarize(eqn.source_info)}"
              for eqn in progenitor_eqns[:5]]  # show at most 5
      origin += ("This value became a tracer due to JAX operations on these lines:"
                 "\n\n" + "\n\n".join(msts))
      if len(progenitor_eqns) > 5:
        origin += "\n\n(Additional originating lines are not shown.)"
    return "\n" + origin

  def _assert_live(self) -> None:
    if not self._trace.main.jaxpr_stack:  # type: ignore
      raise core.escaped_tracer_error(self, None)

  def get_referent(self):
    frame = self._trace.frame
    val = frame.constvar_to_val.get(frame.tracer_to_var.get(id(self)))
    return self if val is None else get_referent(val)
api_util._shaped_abstractify_handlers[DynamicJaxprTracer] = op.attrgetter("aval")

def make_jaxpr_effects(constvars, invars, outvars, eqns) -> effects.Effects:
  del outvars
  jaxpr_effects = set()
  all_vars = [*constvars, *invars]
  for eqn in eqns:
    for eff in eqn.effects:
      if isinstance(eff, effects.JaxprInputEffect):
        invar = eqn.invars[eff.input_index]
        if invar not in all_vars:
          raise ValueError(
              "`JaxprInputEffect` does not have corresponding input.")
        eff = eff.replace(input_index=all_vars.index(invar))
      jaxpr_effects.add(eff)
  return jaxpr_effects


class JaxprStackFrame:
  gensym: Callable[[AbstractValue], Var]
  tracer_to_var: Dict[TracerId, Var]
  constid_to_tracer: Dict[ConstId, Tracer]
  constvar_to_val: Dict[Var, Any]
  tracers: List[DynamicJaxprTracer]  # hold onto strong refs for all tracers
  eqns: List[JaxprEqn]
  invars: List[Var]
  effects: core.Effects
  debug_info: Optional[DebugInfo]

  def __init__(self):
    self.gensym = core.gensym()
    self.tracer_to_var = {}
    self.constid_to_tracer = {}
    self.constvar_to_val = {}
    self.tracers = []   # circ refs, frame->tracer->trace->main->frame,
    self.eqns = []      # cleared when we pop frame from main
    self.invars = []
    self.effects = set()
    self.debug_info = None

  def add_eqn(self, eqn: core.JaxprEqn):
    self.eqns.append(eqn)

  def to_jaxpr(self, out_tracers):
    # It's not necessary, but we keep the tracer-to-var mapping injective:
    assert len(self.tracer_to_var) == len(set(self.tracer_to_var.values()))
    outvars = [self.tracer_to_var[id(t)] for t in out_tracers]
    constvars, constvals = unzip2(self.constvar_to_val.items())
    jaxpr_effects = make_jaxpr_effects(constvars, self.invars, outvars,
                                        self.eqns)
    jaxpr = Jaxpr(constvars, self.invars, outvars, self.eqns, jaxpr_effects)
    jaxpr, constvals = _const_folding_and_forwarding(jaxpr, constvals)
    jaxpr, constvals = _inline_literals(jaxpr, constvals)
    return jaxpr, constvals

  def to_jaxpr2(self, out_tracers):
    # It's not necessary, but we keep the tracer-to-var mapping injective:
    assert len(self.tracer_to_var) == len(set(self.tracer_to_var.values()))
    constvars, constvals = unzip2(self.constvar_to_val.items())
    expl_outvars = [self.tracer_to_var[id(t)] for t in out_tracers]
    jaxpr_effects = make_jaxpr_effects(constvars, self.invars, expl_outvars,
                                        self.eqns)
    jaxpr = Jaxpr(constvars, self.invars, expl_outvars, self.eqns,
                  jaxpr_effects)
    # We can't run check_jaxpr until after we normalize.
    jaxpr, constvals = _const_folding_and_forwarding(jaxpr, constvals)
    jaxpr, constvals = _inline_literals(jaxpr, constvals)
    jaxpr, out_type = _add_implicit_outputs(jaxpr)
    config.jax_enable_checks and core.check_jaxpr(jaxpr)
    return jaxpr, out_type, constvals

  def newvar(self, aval):
    if isinstance(aval, DShapedArray):
      # this aval may have tracers in it, so we replace those with variables
      new_shape = [self.tracer_to_var[id(d)] if isinstance(d, Tracer) else d
                   for d in aval.shape]
      aval = aval.update(shape=tuple(new_shape))
    return self.gensym(aval)

  def find_progenitors(self, tracer):
    var = self.tracer_to_var.get(id(tracer))
    if not var:
      return None, None
    active_vars = {var}
    for eqn in self.eqns[::-1]:
      produced = set(eqn.outvars) & active_vars
      if produced:
        active_vars.difference_update(produced)
        active_vars.update(eqn.invars)
    invar_positions = [i for i, v in enumerate(self.invars) if v in active_vars]
    constvars = active_vars & set(self.constvar_to_val)
    const_eqns = [eqn for eqn in self.eqns if set(eqn.invars) & constvars]
    return invar_positions, const_eqns

def _const_folding_and_forwarding(jaxpr, constvals):
  consts: Dict[Var, Any] = dict(zip(jaxpr.constvars, constvals))
  var_subs: Dict[Var, Var] = {}  # not Dict[Var, Atom] b/c literals not inlined
  new_eqns = []
  for eqn in jaxpr.eqns:
    # always apply invar substitutions
    eqn = eqn.replace(invars=[var_subs.get(v, v) for v in eqn.invars])
    # if any inputs are constants and we have a constant-folding rule, apply it
    has_input_effect = any(isinstance(eff, effects.JaxprInputEffect)
                           for eff in eqn.effects)
    if (eqn.primitive in const_fold_rules and any(v in consts for v in eqn.invars)
        and not has_input_effect):
      consts_in = [consts.get(v) for v in eqn.invars]
      consts_out, new_eqn = const_fold_rules[eqn.primitive](consts_in, eqn)
      assert (new_eqn is None) == all(c is not None for c in consts_out)
      for v, c in zip(eqn.outvars, consts_out):
        if c is not None: consts[v] = c
      if new_eqn is None: continue
      else: eqn = new_eqn
    # if the application trivially maps some inputs to outputs, simplify
    if eqn.primitive in forwarding_rules and not has_input_effect:
      fwd_vars, new_eqn = forwarding_rules[eqn.primitive](eqn)
      assert (new_eqn is None) == all(v is not None for v in fwd_vars)
      for v_orig, v_new in zip(eqn.outvars, fwd_vars):
        if v_new is not None: var_subs[v_orig] = v_new
      if new_eqn is None: continue
      else: eqn = new_eqn
    new_eqns.append(eqn)
  new_constvars, new_constvals = unzip2(consts.items())
  new_outvars = [var_subs.get(v, v) for v in jaxpr.outvars]
  jaxpr_effects = make_jaxpr_effects(new_constvars, jaxpr.invars, new_outvars,
                                      new_eqns)
  new_jaxpr = Jaxpr(new_constvars, jaxpr.invars, new_outvars, new_eqns,
                    jaxpr_effects)
  return new_jaxpr, new_constvals

ConstFoldRule = Callable[[List[Optional[Any]], JaxprEqn],
                         Tuple[List[Optional[Any]], Optional[JaxprEqn]]]
const_fold_rules: Dict[Primitive, ConstFoldRule] = {}

ForwardingRule = Callable[[JaxprEqn],
                          Tuple[List[Optional[Var]], Optional[JaxprEqn]]]
forwarding_rules: Dict[Primitive, ForwardingRule] = {}

def _inline_literals(jaxpr, constvals):
  # This function also prunes unused constants and inserts `dropvar` symbols.
  input_effects = {eff for eff in jaxpr.effects
                   if isinstance(eff, effects.JaxprInputEffect)}
  # Don't inline any literal with an input effect
  has_input_effect = [any(eff.input_index == i for eff in input_effects)
                      for i in range(len(constvals))]
  lits = {v: Literal(c, v.aval) for v, c, e in zip(jaxpr.constvars, constvals,
                                                   has_input_effect)
          if type(c) in core.literalable_types and not np.shape(c) and not e}
  lit: Callable[[Var], Optional[Literal]] = lits.get
  newname: Callable[[AbstractValue], Var] = core.gensym()
  newvars: Dict[Var, Var] = {}
  newvar = lambda aval: newname(_substitute_vars_in_type(lits, newvars, aval))
  var = lambda v: newvars.get(v) or newvars.setdefault(v, newvar(v.aval))
  dropvar = lambda aval: DropVar(_substitute_vars_in_type(lits, newvars, aval))

  def vars_in_shape(aval: AbstractValue) -> Sequence[Var]:
    if isinstance(aval, DShapedArray):
      return [d for d in aval.shape if isinstance(d, Var)]
    return []

  used = {v for eqn in jaxpr.eqns for invar in eqn.invars
          for v in it.chain([invar], vars_in_shape(invar.aval))}
  used |= {v for outvar in jaxpr.outvars
           for v in it.chain([outvar], vars_in_shape(outvar.aval))}
  new_constvars = [var(v) for v in jaxpr.constvars if v in used and not lit(v)]
  new_constvals = [c for v, c in zip(jaxpr.constvars, constvals)
                   if v in used and not lit(v)]
  new_invars = [var(v) for v in jaxpr.invars]
  new_eqns = []
  for eqn in jaxpr.eqns:
    invars = [lit(v) or var(v) for v in eqn.invars]
    outvars = [var(v) if v in used else dropvar(v.aval) for v in eqn.outvars]
    new_eqns.append(eqn.replace(invars=invars, outvars=outvars))
  new_outvars = [lit(v) or var(v) for v in jaxpr.outvars]
  jaxpr_effects = make_jaxpr_effects(new_constvars, new_invars, new_outvars,
                                      new_eqns)
  new_jaxpr = Jaxpr(new_constvars, new_invars, new_outvars, new_eqns,
                    jaxpr_effects)
  return new_jaxpr, new_constvals

class DynamicJaxprTrace(core.Trace):
  __slots__ = []  # type: ignore

  @property
  def frame(self):
    return self.main.jaxpr_stack[-1]  # pytype: disable=attribute-error

  def new_arg(self, aval):
    tracer = DynamicJaxprTracer(self, aval, source_info_util.current())
    self.frame.tracers.append(tracer)
    self.frame.tracer_to_var[id(tracer)] = var = self.frame.newvar(aval)
    self.frame.invars.append(var)
    return tracer

  def new_const(self, c):
    # TODO(mattjj): for ints, or hashable consts, don't rely on id
    tracer = self.frame.constid_to_tracer.get(id(c))
    if tracer is None:
      aval = raise_to_shaped(get_aval(c), weak_type=dtypes.is_weakly_typed(c))
      tracer = self._new_const(aval, c)
    return tracer

  pure = lift = new_const

  def _new_const(self, aval, c):
    tracer = DynamicJaxprTracer(self, aval, source_info_util.current())
    self.frame.tracers.append(tracer)
    self.frame.tracer_to_var[id(tracer)] = var = self.frame.newvar(aval)
    self.frame.constid_to_tracer[id(c)] = tracer
    self.frame.constvar_to_val[var] = c
    return tracer

  def sublift(self, t):
    # When lifting closed-over tracers corresponding to this same trace, the
    # variable to lift could have tracers (representing axis size variables) in
    # its shape. We must lift those too!
    tracer = self.frame.constid_to_tracer.get(id(t))
    if tracer is None:
      aval = raise_to_shaped(get_aval(t), weak_type=dtypes.is_weakly_typed(t))
      aval = self._lift_tracers_in_aval(aval)
      tracer = self._new_const(aval, t)
    return tracer

  def _lift_tracers_in_aval(self, aval):
    if (not isinstance(aval, DShapedArray) or
        not any(isinstance(d, Tracer) for d in aval.shape)):
      return aval
    shape = [self.full_raise(d) if isinstance(d, Tracer) else d
             for d in aval.shape]
    return aval.update(shape=tuple(shape))

  def getvar(self, tracer):
    var = self.frame.tracer_to_var.get(id(tracer))
    if var is None:
      raise core.escaped_tracer_error(tracer)
    return var

  def makevar(self, tracer):
    var = self.frame.tracer_to_var.get(id(tracer))
    assert var is None, "a jaxpr variable must be created only once per tracer"
    self.frame.tracers.append(tracer)
    var = self.frame.tracer_to_var[id(tracer)] = self.frame.newvar(tracer.aval)
    return var

  def instantiate_const(self, val):
    if (isinstance(val, Tracer) and val._trace.main is self.main
        and val._trace.sublevel == self.sublevel):
      return val
    else:
      return self.new_const(val)

  def process_primitive(self, primitive, tracers, params):
    if primitive in custom_staging_rules:
      return custom_staging_rules[primitive](self, *tracers, **params)
    return self.default_process_primitive(primitive, tracers, params)

  def default_process_primitive(self, primitive, tracers, params):
    avals = [t.aval for t in tracers]
    out_avals, effects = primitive.abstract_eval(*avals, **params)
    out_avals = [out_avals] if not primitive.multiple_results else out_avals
    source_info = source_info_util.current()
    out_tracers = [DynamicJaxprTracer(self, a, source_info) for a in out_avals]
    invars = map(self.getvar, tracers)
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn(invars, outvars, primitive, params, effects, source_info)
    self.frame.add_eqn(eqn)
    return out_tracers if primitive.multiple_results else out_tracers.pop()

  def process_call(self, call_primitive, f, explicit_tracers, params):
    if f.in_type is None:
      f = lu.annotate(f, tuple((raise_to_shaped(t.aval), True)
                               for t in explicit_tracers))
    implicit_tracers = _extract_implicit_args(self, f.in_type, explicit_tracers)
    in_tracers = [*implicit_tracers, *explicit_tracers]
    # TODO(mattjj): check in_tracers are consistent with f.in_type annotation
    with core.new_sublevel():
      # TODO(lenamartens): Make call_primitive name -> API function name mapping.
      # (currently this will display eg. 'xla_call' instead of `jit`)
      dbg = debug_info_final(f, call_primitive.name)
      jaxpr, out_type, consts = trace_to_subjaxpr_dynamic2(f, self.main, debug_info=dbg)
    if params.get('inline', False):
      return core.eval_jaxpr(jaxpr, consts, *in_tracers)
    source_info = source_info_util.current()
    out_tracers = []
    for aval, _ in out_type:
      if type(aval) is DShapedArray:
        shape = [[*consts, *in_tracers][d.val] if type(d) is InDBIdx else
                out_tracers[d.val] if type(d) is OutDBIdx else
                d for d in aval.shape]
        aval = aval.update(shape=tuple(get_referent(d) for d in shape))
      out_tracers.append(DynamicJaxprTracer(self, aval, source_info))
    invars = map(self.getvar, in_tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    new_params = dict(params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
    update_params = call_param_updaters.get(call_primitive)
    if update_params:
      new_params = update_params(new_params, [True] * len(explicit_tracers),
                                 len(consts) + len(implicit_tracers))
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, call_primitive,
                        new_params, new_params['call_jaxpr'].effects,
                        source_info)
    self.frame.add_eqn(eqn)
    return [t for t, (_, keep) in zip(out_tracers, out_type) if keep]

  def post_process_call(self, call_primitive, out_tracers, params):
    assert False  # unreachable

  def process_map(self, map_primitive, f, in_tracers, params):
    in_avals = map_primitive.map_avals([t.aval for t in in_tracers], **params)
    debug_info = debug_info_final(f, map_primitive.name)
    with map_primitive.extend_axis_env(**params):
      with core.new_sublevel():
        jaxpr, out_avals_body, consts = trace_to_subjaxpr_dynamic(
            f, self.main, in_avals, debug_info=debug_info)
      jaxpr = convert_constvars_jaxpr(jaxpr)
    ordered_effects = effects.ordered_effects.filter_in(jaxpr.effects)
    if ordered_effects:
      raise ValueError("Ordered effects not supported for "
                       f"map primitives: {ordered_effects}")
    params = map_primitive.get_staged_params(jaxpr, **params)
    out_avals = map_primitive.unmap_avals(out_avals_body, **params)
    src_info = source_info_util.current()
    out_tracers = [DynamicJaxprTracer(self, a, src_info) for a in out_avals]
    invars = map(self.getvar, in_tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, map_primitive, params,
                        jaxpr.effects, src_info)
    self.frame.add_eqn(eqn)
    return out_tracers

  def post_process_map(self, map_primitive, out_tracers, params):
    assert False  # unreachable

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros):
    in_avals = [t.aval for t in tracers]
    with core.new_sublevel():
      fun_jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, self.main, in_avals)
    closed_fun_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(fun_jaxpr), ())
    main_ = ref(self.main)
    jvp_jaxpr_thunk = _memoize(
        lambda: trace_to_subjaxpr_dynamic(jvp, main_(), 2 * in_avals)[::2])
    out_tracers = [DynamicJaxprTracer(self, a) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, prim,
                        dict(call_jaxpr=closed_fun_jaxpr,
                             jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                             num_consts=len(consts),
                             symbolic_zeros=symbolic_zeros),
                        fun_jaxpr.effects,
                        source_info_util.current())
    self.frame.add_eqn(eqn)
    return out_tracers

  def post_process_custom_jvp_call(self, out_tracers, _):
    assert False  # unreachable

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees):
    in_avals = [t.aval for t in tracers]
    with core.new_sublevel():
      fun_jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, self.main, in_avals)
    closed_fun_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(fun_jaxpr), ())
    main_ = ref(self.main)
    fwd_jaxpr_thunk = _memoize(
        lambda: trace_to_subjaxpr_dynamic(fwd, main_(), in_avals)[::2])
    out_tracers = [DynamicJaxprTracer(self, a) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, prim.initial_style,
                        dict(fun_jaxpr=closed_fun_jaxpr,
                             fwd_jaxpr_thunk=fwd_jaxpr_thunk,
                             num_consts=len(consts),
                             bwd=bwd, out_trees=out_trees),
                        fun_jaxpr.effects,
                        source_info_util.current())
    self.frame.add_eqn(eqn)
    return out_tracers

  def post_process_custom_vjp_call(self, out_tracers, _):
    assert False  # unreachable

  def process_custom_transpose(self, prim, call, tracers,
                               transpose, out_types,
                               lin_tree, res_tree, out_tree):
    tracers_res, tracers_lin = split_list(tracers, [res_tree.num_leaves])

    in_avals_p = [t.aval for t in tracers]
    in_avals_t = [*[t.aval for t in tracers_res], *out_types]

    with core.new_sublevel():
      call_jaxpr, out_avals, call_consts = trace_to_subjaxpr_dynamic(
          call, self.main, in_avals_p)
    closed_call_jaxpr = core.ClosedJaxpr(
        convert_constvars_jaxpr(call_jaxpr), ())

    transpose_flat, in_tree2 = flatten_fun_nokwargs(
        lu.wrap_init(transpose), treedef_tuple((res_tree, out_tree)))

    main_ = ref(self.main)
    # the following thunk evaluates to a pair: transpose_jaxpr, transpose_consts
    transpose_jaxpr_thunk = _memoize(
        lambda: trace_to_subjaxpr_dynamic(
            transpose_flat, main_(), in_avals_t)[::2])

    out_tracers = [DynamicJaxprTracer(self, a) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, call_consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, prim,
                        dict(call_jaxpr=closed_call_jaxpr,
                             transpose_jaxpr_thunk=transpose_jaxpr_thunk,
                             out_types=out_types, res_tree=res_tree,
                             lin_tree=lin_tree, out_tree=out_tree),
                        closed_call_jaxpr.effects,
                        source_info_util.current())
    self.frame.add_eqn(eqn)
    return out_tracers


custom_staging_rules: Dict[Primitive, Callable] = {}

def _memoize(thunk):
  cell = []
  saved_state = [core.thread_local_state.trace_state.copy()]
  def memoized():
    if not cell:
      prev_state = core.thread_local_state.trace_state
      core.thread_local_state.trace_state = saved_state.pop()
      try:
        cell.append(thunk())
      finally:
        core.thread_local_state.trace_state = prev_state
    return cell[0]
  return memoized


class DebugInfo(NamedTuple):
  func_src_info: str
  traced_for: str
  arg_info: Callable[[int], str]


def debug_info_final(fn: lu.WrappedFun, traced_for: str) -> DebugInfo:
  in_tree, has_kwargs = flattened_fun_in_tree(fn) or (None, False)
  return debug_info(fn.f, in_tree, has_kwargs, traced_for)

def debug_info(fn: Callable, in_tree: Optional[PyTreeDef], has_kwargs: bool,
               traced_for: str) -> DebugInfo:
  func_src_info = fun_sourceinfo(fn)
  if in_tree is not None:
    arg_info = partial(arg_info_pytree, fn, in_tree, has_kwargs)
  else:
    arg_info = arg_info_flattened  # type: ignore
  return DebugInfo(func_src_info, traced_for, arg_info)

def fun_sourceinfo(fun: Callable):
  while isinstance(fun, functools.partial):
    fun = fun.func
  fun = inspect.unwrap(fun)
  try:
    filename = fun.__code__.co_filename
    lineno = fun.__code__.co_firstlineno
    line_info = f"{fun.__name__} at {filename}:{lineno}"
    return line_info
  except AttributeError:
    return "<unknown>"

def arg_info_pytree(fn: Callable, in_tree: PyTreeDef, has_kwargs: bool,
                    flat_pos: List[int]) -> str:
  dummy_args = [False] * in_tree.num_leaves
  for i in flat_pos: dummy_args[i] = True
  if has_kwargs:
    args, kwargs = tree_unflatten(in_tree, dummy_args)
  else:
    args, kwargs = tree_unflatten(in_tree, dummy_args), {}
  try:
    ba = inspect.signature(fn).bind(*args, **kwargs)
  except (TypeError, ValueError):
    return arg_info_flattened(flat_pos)
  arg_names = [f"'{name}'" for name, x in ba.arguments.items()
               if any(tree_leaves(x))]
  if len(arg_names) == 1:
    return f"the argument {arg_names[0]}"
  elif len(arg_names) == 2:
    return f"the arguments {arg_names[0]} and {arg_names[1]}"
  else:
    *rest, last = arg_names
    return f"the arguments {', '.join(rest)}, and {last}"

def arg_info_flattened(flat_pos: List[int]) -> str:
  if len(flat_pos) > 1:
    return f"the argument passed at flattened positions {flat_pos}"
  else:
    return f"the argument passed at flattened position {flat_pos[0]}"


@profiler.annotate_function
def trace_to_jaxpr_dynamic(fun: lu.WrappedFun,
                           in_avals: Sequence[AbstractValue],
                           debug_info: Optional[DebugInfo] = None,
                           *,
                           keep_inputs: Optional[List[bool]] = None):
  with core.new_main(DynamicJaxprTrace, dynamic=True) as main:  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(
      fun, main, in_avals, keep_inputs=keep_inputs, debug_info=debug_info)
    del main, fun
  return jaxpr, out_avals, consts

def trace_to_subjaxpr_dynamic(fun: lu.WrappedFun, main: core.MainTrace,
                              in_avals: Sequence[AbstractValue], *,
                              keep_inputs: Optional[Sequence[bool]] = None,
                              debug_info: Optional[DebugInfo] = None):
  keep_inputs = [True] * len(in_avals) if keep_inputs is None else keep_inputs

  frame = JaxprStackFrame()
  frame.debug_info = debug_info
  with extend_jaxpr_stack(main, frame), source_info_util.reset_name_stack():
    trace = DynamicJaxprTrace(main, core.cur_sublevel())
    in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
    in_tracers_ = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
    ans = fun.call_wrapped(*in_tracers_)
    out_tracers = map(trace.full_raise, ans)
    jaxpr, consts = frame.to_jaxpr(out_tracers)
    del fun, main, trace, frame, in_tracers, out_tracers, ans
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  return jaxpr, [v.aval for v in jaxpr.outvars], consts


@profiler.annotate_function
def trace_to_jaxpr_dynamic2(
    fun: lu.WrappedFun, debug_info: Optional[DebugInfo] = None
  ) -> Tuple[Jaxpr, OutputType, List[Any]]:
  with core.new_main(DynamicJaxprTrace, dynamic=True) as main:  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    jaxpr, out_type, consts = trace_to_subjaxpr_dynamic2(fun, main, debug_info)
    del main, fun
  return jaxpr, out_type, consts

def trace_to_subjaxpr_dynamic2(
    fun: lu.WrappedFun, main: core.MainTrace,
    debug_info: Optional[DebugInfo] = None
) -> Tuple[Jaxpr, OutputType, List[Any]]:
  in_avals, keep_inputs = unzip2(fun.in_type)
  frame = JaxprStackFrame()
  frame.debug_info = debug_info
  with extend_jaxpr_stack(main, frame), source_info_util.reset_name_stack():
    trace = DynamicJaxprTrace(main, core.cur_sublevel())
    in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)
    in_tracers_ = [t for t, keep in zip(in_tracers, keep_inputs) if keep]
    ans = fun.call_wrapped(*in_tracers_)
    out_tracers = map(trace.full_raise, ans)
    jaxpr, out_type, consts = frame.to_jaxpr2(out_tracers)
    del fun, main, trace, frame, in_tracers, out_tracers, ans
  return jaxpr, out_type, consts


@contextlib.contextmanager
def extend_jaxpr_stack(main, frame):
  main.jaxpr_stack = main.jaxpr_stack + (frame,)
  try:
    yield
  finally:
    assert frame is main.jaxpr_stack[-1]
    main.jaxpr_stack = main.jaxpr_stack[:-1]

@profiler.annotate_function
def trace_to_jaxpr_final(fun: lu.WrappedFun,
                         in_avals: Sequence[AbstractValue],
                         debug_info: Optional[DebugInfo] = None,
                         keep_inputs: Optional[Sequence[bool]] = None):
  with core.new_base_main(DynamicJaxprTrace) as main:  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    with core.new_sublevel():
      jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(
        fun, main, in_avals, keep_inputs=keep_inputs, debug_info=debug_info)
    del fun, main
  return jaxpr, out_avals, consts

@profiler.annotate_function
def trace_to_jaxpr_final2(
    fun: lu.WrappedFun, debug_info: Optional[DebugInfo] = None
  ) -> Tuple[Jaxpr, OutputType, List[Any]]:
  with core.new_base_main(DynamicJaxprTrace) as main:  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    with core.new_sublevel():
      jaxpr, out_type, consts = trace_to_subjaxpr_dynamic2(fun, main, debug_info)
    del fun, main
  return jaxpr, out_type, consts


AbstractedAxisName = Hashable
AbstractedAxesSpec = Union[Dict[int, AbstractedAxisName],
                           Tuple[AbstractedAxisName, ...]]
def infer_lambda_input_type(
    axes_specs: Optional[Sequence[AbstractedAxesSpec]],
    args: Sequence[Any]
  ) -> InputType:
  ndims = [getattr(get_aval(x), 'ndim', 0) for x in args]
  partial_specs = _canonicalize_specs(ndims, axes_specs)
  specs = _complete_specs(args, partial_specs)
  idxs, implicit_types = _collect_implicit(args, specs)
  implicit_sig = [(ty, False) for ty in implicit_types]
  explicit_sig = [(_arg_type(idxs, x, s), True) for x, s in zip(args, specs)]
  input_type = (*implicit_sig, *explicit_sig)
  lu._check_input_type(input_type)
  return input_type

def _spec_to_dict(spec: AbstractedAxesSpec) -> Dict[int, AbstractedAxisName]:
  if isinstance(spec, tuple):
    return {i: d for i, d in enumerate(spec) if d is not None}
  else:
    return spec

def _canonicalize_specs(
    ndims: Sequence[int], specs: Optional[Sequence[AbstractedAxesSpec]]
  ) -> List[Dict[int, AbstractedAxisName]]:
  if specs is None:
    return [{}] * len(ndims)
  else:
    return [_spec_to_dict(s) for n, s in zip(ndims, specs)]

def _complete_specs(
    args: Sequence[Any], partial_specs: List[Dict[int, AbstractedAxisName]]
  ) -> List[Dict[int, AbstractedAxisName]]:
  # The abstracted axes specification in `partial_specs` is partial in the sense
  # that there could be additional axis abstraction represented in `args` due to
  # Tracers existing in the shapes of elements of `args`. The purpose of this
  # function is to produce a full specification, for each argument mapping any
  # abstracted axis positions to a name, introducing new names as needed for
  # Tracers in axis sizes which don't already correspond to abstracted axis
  # names (with one new name per unique Tracer object id).

  # Identify each user-supplied name in partial_specs with a size.
  sizes: Dict[AbstractedAxisName, Union[int, DynamicJaxprTracer]] = {}
  for x, spec in zip(args, partial_specs):
    for i, name in spec.items():
      d = sizes.setdefault(name, x.shape[i])
      if d is not x.shape[i] and d != x.shape[i]: raise TypeError

  # Introduce new names as needed for Tracers in shapes.
  named_tracers: Dict[TracerId, AbstractedAxisName] = {
      id(d): name for name, d in sizes.items() if isinstance(d, Tracer)}
  specs: List[Dict[int, AbstractedAxisName]] = []
  for x, spec in zip(args, partial_specs):
    if isinstance(get_aval(x), DShapedArray):
      spec = dict(spec)
      for i, d in enumerate(x.shape):
        if isinstance(d, Tracer):
          spec[i] = named_tracers.get(id(d), TracerAsName(d))
    specs.append(spec)

  # Assert that `specs` is now complete in the sense that there are no Tracers
  # which don't correspond to an AbstractedAxisName.
  assert all(not spec or not any(isinstance(d, Tracer) and i not in spec
                                 for i, d in enumerate(x.shape))
             for x, spec in zip(args, specs))
  return specs


def _collect_implicit(
    args: Sequence[Any], specs: List[Dict[int, AbstractedAxisName]]
  ) -> Tuple[Dict[AbstractedAxisName, DBIdx], List[AbstractValue]]:
  # Given an explicit argument list and a specification of abstracted axes, we
  # want to produce an InputType by identifying AbstractedAxisNames with DBIdxs
  # and figuring out which AbstractedAxisNames correspond to implicit arguments.

  idxs: Dict[AbstractedAxisName, DBIdx] = {}
  implicit_types: List[AbstractValue] = []
  explicit_tracers: Dict[TracerId, int] = {}
  counter = it.count()

  # Add implicit arguments to idxs.
  for explicit_idx, (x, spec) in enumerate(zip(args, specs)):
    for i, name in spec.items():
      if name not in idxs and id(x.shape[i]) not in explicit_tracers:
        idxs[name] = DBIdx(next(counter))
        implicit_types.append(raise_to_shaped(get_aval(x.shape[i])))
    if isinstance(x, Tracer):
      explicit_tracers.setdefault(id(x), explicit_idx)  # use the first

  # Now that we know the implicit args, add explicit args to idxs.
  offset = len(implicit_types)
  for x, spec in zip(args, specs):
    for i, name in spec.items():
      if id(x.shape[i]) in explicit_tracers:
        idxs.setdefault(name, DBIdx(offset + explicit_tracers[id(x.shape[i])]))

  return idxs, implicit_types

def _arg_type(
    idxs: Dict[AbstractedAxisName, DBIdx], x: Any,
    spec: Dict[int, AbstractedAxisName]
  ) -> AbstractValue:
  # Produce an AbstractValue by substituting DBIdxs for AbstractedAxisNames.
  aval = get_aval(x)  # aval.shape could contain Tracers
  if not spec: return core.raise_to_shaped(aval)
  shape: List[Union[int, DBIdx]] = [idxs[spec[i]] if i in spec else d
                                    for i, d in enumerate(aval.shape)]
  assert not any(isinstance(d, Tracer) for d in shape)
  return DShapedArray(tuple(shape), aval.dtype, False)

def _add_implicit_outputs(jaxpr: Jaxpr) -> Tuple[Jaxpr, OutputType]:
  invars = [*jaxpr.constvars, *jaxpr.invars]
  expl_outvars = jaxpr.outvars

  # First do a pass to collect implicit outputs, meaning variables which occurr
  # in explicit_outvars types but not in invars or to the left in outvars.
  seen: Set[Var] = set(invars)
  impl_outvars = [seen.add(d) or d for x in expl_outvars if type(x) is Var and  # type: ignore
                  (seen.add(x) or type(x.aval) is DShapedArray)  # type: ignore
                  for d in x.aval.shape if type(d) is Var and d not in seen]
  outvars = [*impl_outvars, *expl_outvars]

  # Now assemble an OutputType by mapping vars in shapes to InDBIdx/OutDBIdx.
  in_map : Dict[Var,  InDBIdx] = {v:  InDBIdx(i) for i, v in enumerate( invars)}
  out_map: Dict[Var, OutDBIdx] = {x: OutDBIdx(i) for i, x in enumerate(outvars)
                                  if type(x) is Var}
  out_avals_ = (x.aval for x in outvars)
  out_avals = [a.update(shape=tuple(in_map.get(d, out_map.get(d))
                                    if type(d) is Var else d for d in a.shape))
               if type(a) is DShapedArray else a for a in out_avals_]
  kept_outs = [False] * len(impl_outvars) + [True] * len(expl_outvars)
  out_type = tuple(zip(out_avals, kept_outs))

  new_jaxpr = Jaxpr(jaxpr.constvars, jaxpr.invars, outvars, jaxpr.eqns,
                    jaxpr.effects)
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  return new_jaxpr, out_type


class TracerAsName:
  ref: Any
  def __init__(self, tracer):
    self.ref = core.get_referent(tracer)
  def __eq__(self, other):
    return isinstance(other, TracerAsName) and self.ref is other.ref
  def __hash__(self):
    return id(self.ref)

def _extract_implicit_args(
    trace: DynamicJaxprTrace, in_type: Sequence[Tuple[AbstractValue, bool]],
    explicit_tracers: Sequence[DynamicJaxprTracer]
  ) -> Sequence[DynamicJaxprTracer]:
  # First, construct a list to represent the full argument list, leaving the
  # implicit arguments as Nones for now.
  explicit_tracers_ = iter(explicit_tracers)
  tracers = [next(explicit_tracers_) if expl else None for _, expl in in_type]
  assert next(explicit_tracers_, None) is None
  del explicit_tracers_

  # Next, populate the implicit arguments using DBIdxs in in_type.
  for i, (aval, explicit) in enumerate(in_type):
    if not explicit or not isinstance(aval, DShapedArray):
      continue  # can't populate an implicit argument
    tracer = tracers[i]
    assert tracer is not None
    for d1, d2 in zip(aval.shape, tracer.aval.shape):
      if isinstance(d1, DBIdx):
        if tracers[d1.val] is None:
          tracers[d1.val] = trace.instantiate_const(d2)
        assert tracers[d1.val] is trace.instantiate_const(d2)
  assert all(t is not None for t in tracers)
  tracers_validated = cast(List[DynamicJaxprTracer], tracers)  # remove Optional annotation.
  return [t for t, (_, e) in zip(tracers_validated, in_type) if not e]

def _input_type_to_tracers(
    new_arg: Callable[[AbstractValue], Tracer],
    in_avals: Sequence[AbstractValue]
  )  -> Sequence[Tracer]:
  # Create input Tracers given input AbstractValues, each of which can contain
  # DeBruijn indices which refer to positions in the input argument list. That
  # is, each element `a` of `in_avals` can have DBIdx instances in its shape,
  # which must refer to positions left of `a`'s.
  in_tracers: List[Tracer] = []

  def _substitute_tracers_in_aval(a: AbstractValue) -> AbstractValue:
    if isinstance(a, DShapedArray) and any(type(d) is DBIdx for d in a.shape):
      shape = [in_tracers[d.val] if type(d) is DBIdx else d for d in a.shape]  # type: ignore
      return a.update(shape=tuple(shape))
    return a

  for a in in_avals:
    in_tracers.append(new_arg(_substitute_tracers_in_aval(a)))
  return in_tracers

def _substitute_vars_in_type(
    consts: Dict[Var, Literal], env: Dict[Var, Var], a: AbstractValue
  ) -> AbstractValue:
  if isinstance(a, DShapedArray) and any(isinstance(d, Var) for d in a.shape):
    shape = [consts[d].val if d in consts else env[d]  # type: ignore
             if isinstance(d, Var) else d for d in a.shape]
    return a.update(shape=tuple(shape))
  else:
    return a


class DimensionHandlerTracer(core.DimensionHandler):
  """See core.DimensionHandler.

  Most methods are inherited.
  """
  def is_constant(self, d: core.DimSize) -> bool:
    assert isinstance(d, Tracer)
    return False

  def symbolic_equal(self, d1: core.DimSize, d2: core.DimSize) -> bool:
    return d1 is d2

  def greater_equal(self, d1: core.DimSize, d2: core.DimSize):
    raise core.InconclusiveDimensionOperation("TODO")

  def divide_shape_sizes(self, s1: core.Shape, s2: core.Shape) -> core.DimSize:
    """Computes integer "i" such that i  * size(s2) == size(s1).

    Raise InconclusiveDimensionOperation if there is no such integer for all
    contexts.
    """
    s1_size = functools.reduce(op.mul, s1, 1)
    s2_size = functools.reduce(op.mul, s2, 1)
    q, r = divmod(s1_size, s2_size)
    # TODO(necula): must check that r == 0!
    return q

  def stride(self, d: core.DimSize, window_size: core.DimSize, window_stride: core.DimSize) -> core.DimSize:
    """Implements `(d - window_size) // window_stride + 1`"""
    raise core.InconclusiveDimensionOperation("TODO")

  def as_value(self, d: core.DimSize):
    """Turns a dimension size into a Jax value that we can compute with."""
    raise core.InconclusiveDimensionOperation("TODO")

core._SPECIAL_DIMENSION_HANDLERS[DynamicJaxprTracer] = DimensionHandlerTracer()

Const = Any
Val = Any

def pad_jaxpr(jaxpr: Jaxpr, consts: Sequence[Const]
              ) -> Tuple[Jaxpr, List[Const]]:
  bounds = {v: v.aval.dtype.bound for v in jaxpr.invars
            if isinstance(v.aval, core.UnshapedArray) and
            type(v.aval.dtype) is core.bint and not v.aval.shape}
  idxs = {v: DBIdx(i) for i, v in enumerate(jaxpr.invars)}

  def substitute(aval: AbstractValue) -> AbstractValue:
    if (isinstance(aval, core.UnshapedArray) and type(aval.dtype) is core.bint
        and not aval.shape):
      return ShapedArray((), dtypes._scalar_type_to_dtype(int))
    elif isinstance(aval, DShapedArray):
      shape = [bounds.get(d, idxs.get(d, d)) for d in aval.shape]  # type: ignore
      typ = ShapedArray if all(type(d) is int for d in shape) else DShapedArray
      return typ(tuple(shape), aval.dtype, aval.weak_type)
    else:
      return aval

  in_avals = [substitute(v.aval) for v in jaxpr.invars]
  eval_padded = lu.wrap_init(partial(_eval_jaxpr_padded, jaxpr, consts))
  padded_jaxpr, _, padded_consts = trace_to_jaxpr_dynamic(eval_padded, in_avals)
  return padded_jaxpr, padded_consts

class BoundedAxisSize(NamedTuple):
  val: Union[int, DynamicJaxprTracer]
  bound: int

def _eval_jaxpr_padded(
    jaxpr: Jaxpr, consts: List[Const], *args: DynamicJaxprTracer
  ) -> List[Union[Const, DynamicJaxprTracer]]:
  env: Dict[Var, Val] = {}

  def read(x):
    return x.val if type(x) is Literal else env[x]

  def write(v, val) -> None:
    env[v] = val

  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_avals  = [_substitute_axis_sizes(env, v.aval) for v in eqn.invars]
    out_avals = [_substitute_axis_sizes(env, v.aval) for v in eqn.outvars]
    rule = padding_rules[eqn.primitive]
    outs = rule(in_avals, out_avals, *map(read, eqn.invars), **eqn.params)
    map(write, eqn.outvars, outs)
  return map(read, jaxpr.outvars)

def _substitute_axis_sizes(env: Dict, aval: AbstractValue) -> AbstractValue:
  if isinstance(aval, DShapedArray):
    shp = []
    for d in aval.shape:
      if isinstance(d, core.DArray):
        assert not d.shape and type(d.dtype) is core.bint
        shp.append(BoundedAxisSize(int(d._data), int(d.dtype.bound)))
      elif (type(d) is core.Var and isinstance(d.aval, core.DShapedArray) and
            type(d.aval.dtype) is core.bint):
        assert not d.aval.shape
        shp.append(BoundedAxisSize(env[d], d.aval.dtype.bound))
      else:
        shp.append(env.get(d, d))
    return DShapedArray(tuple(shp), aval.dtype, aval.weak_type)
  else:
    return aval

def _is_bint_axis_size(d: Union[int, core.DArray, core.Var]) -> bool:
  if isinstance(d, core.DArray):
    assert not d.shape                 # pytype: disable=attribute-error
    return type(d.dtype) is core.bint  # pytype: disable=attribute-error
  elif isinstance(d, core.Var):
    return (isinstance(d.aval, core.DShapedArray) and  # pytype: disable=attribute-error
            type(d.aval.dtype) is core.bint)           # pytype: disable=attribute-error
  return False


padding_rules: Dict[Primitive, Callable] = {}

def def_trivial_padding(prim: Primitive) -> None:
  if prim.multiple_results:
    padding_rules[prim] = partial(_trivial_padding_rule_multi, prim)
  else:
    padding_rules[prim] = partial(_trivial_padding_rule, prim)

def _trivial_padding_rule(prim, _, __, *args, **params):
  return [prim.bind(*args, **params)]

def _trivial_padding_rule_multi(prim, _, __, *args, **params):
  return prim.bind(*args, **params)

def call_padding_rule(prim, in_avals, out_avals, *args, call_jaxpr, **params):
  if call_jaxpr.constvars: raise NotImplementedError
  padded_jaxpr, padded_consts = pad_jaxpr(call_jaxpr, ())
  if padded_consts: raise NotImplementedError
  new_params = dict(params, call_jaxpr=padded_jaxpr)
  subfuns, bind_params = prim.get_bind_params(new_params)
  return prim.bind(*subfuns, *args, **bind_params)


# TODO(mattjj): the following are deprecated; update callers to _nounits version
# See https://github.com/google/jax/pull/9498
@lu.transformation
def trace_to_subjaxpr(main: core.MainTrace, instantiate: Union[bool, Sequence[bool]],
                      pvals: Sequence[PartialVal]):
  assert all([isinstance(pv, PartialVal) for pv in pvals]), pvals
  trace = main.with_cur_sublevel()
  in_tracers = map(trace.new_arg, pvals)
  ans = yield in_tracers, {}
  assert isinstance(ans, (list, tuple)), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  assert all(isinstance(x, core.Tracer) or core.valid_jaxtype(x) for x in ans), (
      f"Got unexpected return type when tracing function to jaxpr: {ans}")
  instantiate = [instantiate] * len(ans) if isinstance(instantiate, bool) else instantiate
  out_tracers = map(trace.full_raise, map(core.full_lower, ans))
  out_tracers = map(partial(instantiate_const_at, trace), instantiate, out_tracers)
  jaxpr, consts, env = tracers_to_jaxpr(in_tracers, out_tracers)
  out_pvals = [t.pval for t in out_tracers]
  del trace, in_tracers, out_tracers
  yield jaxpr, (out_pvals, consts, env)

partial_eval_jaxpr: Callable

def instantiate_const_at(trace: JaxprTrace, instantiate: bool, tracer):
  if instantiate:
    return trace.instantiate_const(trace.full_raise(tracer))
  else:
    return tracer
