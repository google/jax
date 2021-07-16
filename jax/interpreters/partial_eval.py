# Copyright 2018 Google LLC
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

import inspect
import itertools as it
from collections import namedtuple
import contextlib
import functools
from operator import attrgetter
from typing import (Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple,
                    List, Union, cast)
from weakref import ref

import numpy as np

from .. import core
from .._src import dtypes
from .. import linear_util as lu
from jax._src.ad_util import Zero
from ..api_util import flattened_fun_in_tree
from .._src.tree_util import tree_unflatten, tree_leaves
from .._src.util import (unzip2, safe_zip, safe_map, toposort, partial,
                         split_list, cache, as_hashable_function)
from ..core import (Trace, Tracer, Jaxpr, Literal, get_aval, AbstractValue,
                    unit, unitvar, abstract_unit, ClosedJaxpr, new_jaxpr_eqn,
                    dropvar, ConcreteArray, raise_to_shaped)
from jax._src import source_info_util
from ..config import config

map = safe_map
zip = safe_zip
def identity(x): return x

class PartialVal(tuple):
  """Partial value: either a known value or an unknown (abstract) value.

  Represented as a pair `(aval_opt, const)` of one of two kinds:
  * `(None, <Constant>)` indicates a known value, either a Python regular
    value, or a Tracer.
  * `(<AbstractValue>, *)` indicates an unknown value characterized by an
    abstract value.
  """
  def __new__(cls, xs: Tuple[Optional[AbstractValue], core.Value]):
    pv, const = xs
    if config.jax_enable_checks:
      # type checks
      assert isinstance(pv, (AbstractValue, type(None))), xs
      assert isinstance(const, core.Tracer) or type(const) is Zero or core.valid_jaxtype(const), xs
      # invariant checks
      if isinstance(pv, AbstractValue):
        assert get_aval(const) == core.abstract_unit, xs
    return tuple.__new__(cls, xs)

  @classmethod
  def known(cls, const: core.Value) -> 'PartialVal':
    return PartialVal((None, const))

  @classmethod
  def unknown(cls, aval: AbstractValue) -> 'PartialVal':
    return PartialVal((aval, core.unit))

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

  def merge_with_known(self, val: core.Value) -> core.Value:
    """Either the stored known value, or the given 'val'."""
    known = self.get_known()
    return known if known is not None else val


class JaxprTrace(Trace):
  def pure(self, val) -> 'JaxprTracer':
    return self.new_const(val)

  def lift(self, val) -> 'JaxprTracer':
    return self.new_const(val)

  def sublift(self, val) -> 'JaxprTracer':
    return JaxprTracer(self, val.pval, FreeVar(val))

  def new_const(self, val) -> 'JaxprTracer':
    if isinstance(val, Tracer) and val._trace.level == self.level:
      raise Exception
    return JaxprTracer(self, PartialVal.known(val), unit)

  def new_instantiated_literal(self, val) -> 'JaxprTracer':
    return JaxprTracer(self, PartialVal.unknown(get_aval(val)), Literal(val))

  def new_instantiated_const(self, val) -> 'JaxprTracer':
    return JaxprTracer(self, PartialVal.unknown(get_aval(val)), ConstVar(val))

  def new_arg(self, pval: PartialVal) -> 'JaxprTracer':
    const = pval.get_known()
    if const is None:
      return JaxprTracer(self, pval, LambdaBinding())
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

  def instantiate_const_abstracted(self, tracer) -> 'JaxprTracer':
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
    """By default, if all the input tracers are known, then execute the primitive
    and all the ouputs are known. Otherwise, all the outputs are unknown."""
    consts = [t.pval.get_known() for t in tracers]
    if all(c is not None for c in consts):
      return primitive.bind(*consts, **params)
    tracers = map(self.instantiate_const, tracers)
    avals = [t.aval for t in tracers]
    out_aval = primitive.abstract_eval(*avals, **params)
    source = source_info_util.current()
    if primitive.multiple_results:
      out_tracers = [JaxprTracer(self, PartialVal.unknown(aval), None)
                     for aval in out_aval]
      eqn = new_eqn_recipe(tracers, out_tracers, primitive, params, source)
      for t in out_tracers: t.recipe = eqn
      return out_tracers
    else:
      out_tracer = JaxprTracer(self, PartialVal.unknown(out_aval), None)
      out_tracer.recipe = new_eqn_recipe(tracers, [out_tracer], primitive,
                                         params, source)
      return out_tracer

  def process_call(self, primitive, f, tracers, params):
    in_pvs, in_consts = unzip2(t.pval for t in tracers)
    f = trace_to_subjaxpr(f, self.main, False)
    f, aux = partial_eval_wrapper(f, in_pvs)
    todo = partial(partial_eval_todo, self, primitive, tracers, params, aux)
    return f, in_consts, params, todo

  def process_map(self, primitive, f, tracers, params):
    in_pvs, in_consts = unzip2(t.pval for t in tracers)
    f = trace_to_subjaxpr(f, self.main, False)
    f, aux = partial_eval_wrapper_map(
        f, in_pvs, params['axis_name'], params['axis_size'],
        params['in_axes'], params['out_axes_thunk'])
    f, num_outputs = count_outputs(f)
    todo = partial(partial_eval_todo, self, primitive, tracers, params, aux)

    @as_hashable_function(closure=params['out_axes_thunk'])
    def new_out_axes_thunk():
      out_axes = params['out_axes_thunk']()
      return out_axes + (0,) * (num_outputs() - len(out_axes))

    return f, in_consts, dict(params, out_axes_thunk=new_out_axes_thunk), todo

  def process_custom_jvp_call(self, prim, fun, jvp, tracers):
    tracers = map(self.instantiate_const_abstracted, tracers)
    in_avals, in_consts = unzip2(t.pval for t in tracers)  # in_consts are units
    fun = trace_to_subjaxpr(fun, self.main, True)
    fun, aux = partial_eval_wrapper(fun, tuple(in_avals))
    out_flat = prim.bind(fun, jvp, *in_consts)
    out_avals, jaxpr, env = aux()
    out_consts, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    out_pvals = map(PartialVal, zip(out_avals, out_consts))  # out_consts are units
    env_tracers = map(self.full_raise, env)
    out_tracers = [JaxprTracer(self, pval, None) for pval in out_pvals]
    const_tracers = map(self.new_instantiated_const, consts)
    in_tracers = (*const_tracers, *env_tracers, *tracers)
    closed_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ())

    @_memoize
    def jvp_jaxpr_thunk():
      jvp_ = trace_to_subjaxpr(jvp, self.main, True)
      jvp_, aux = partial_eval_wrapper(jvp_, tuple(in_avals) * 2)
      with core.new_sublevel():
        out_flat = jvp_.call_wrapped(*(in_consts * 2))  # in_consts are units
      out_avals, jaxpr, env = aux()
      _, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
      converted_jaxpr = convert_envvars_to_constvars(jaxpr, len(env))
      return converted_jaxpr, (*consts, *env)

    eqn = new_eqn_recipe(in_tracers, out_tracers, prim.initial_style,
                         dict(fun_jaxpr=closed_jaxpr,
                              jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                              num_consts=len(consts) + len(env)),
                         source_info_util.current())
    for t in out_tracers: t.recipe = eqn
    return out_tracers

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees):
    tracers = map(self.instantiate_const_abstracted, tracers)
    in_avals, in_consts = unzip2(t.pval for t in tracers)  # in_consts are units
    fun = trace_to_subjaxpr(fun, self.main, True)
    fun, aux = partial_eval_wrapper(fun, tuple(in_avals))
    out_flat = prim.bind(fun, fwd, bwd, *in_consts, out_trees=out_trees)
    out_avals, jaxpr, env = aux()
    out_consts, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
    out_pvals = map(PartialVal, zip(out_avals, out_consts))  # out_consts are units
    env_tracers = map(self.full_raise, env)
    out_tracers = [JaxprTracer(self, pval, None) for pval in out_pvals]
    const_tracers = map(self.new_instantiated_const, consts)
    in_tracers = (*const_tracers, *env_tracers, *tracers)
    closed_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(jaxpr), ())

    @_memoize
    def fwd_jaxpr_thunk():
      fwd_ = trace_to_subjaxpr(fwd, self.main, True)
      fwd_, aux = partial_eval_wrapper(fwd_, tuple(in_avals))
      with core.new_sublevel():
        out_flat = fwd_.call_wrapped(*in_consts)  # in_consts are units
      out_avals, jaxpr, env = aux()
      _, consts = split_list(out_flat, [len(out_flat)-len(jaxpr.constvars)])
      converted_jaxpr = convert_envvars_to_constvars(jaxpr, len(env))
      return converted_jaxpr, (*consts, *env)

    eqn = new_eqn_recipe(in_tracers, out_tracers, prim.initial_style,
                         dict(fun_jaxpr=closed_jaxpr,
                              fwd_jaxpr_thunk=fwd_jaxpr_thunk,
                              num_consts=len(consts) + len(env),
                              bwd=bwd, out_trees=out_trees),
                         source_info_util.current())
    for t in out_tracers: t.recipe = eqn
    return out_tracers

  def process_remat_call(self, primitive, f, tracers, params):
    # Unlike JaxprTrace.process_call, we want to form a jaxpr for the entirety of
    # the function being called, not just for the unknown parts. To do that, we
    # instantiate all the input tracers as constants in the jaxpr being formed.
    # Those tracers might have concrete avals, and doing abstract interpretation
    # on concrete avals engenders a tradeoff: it allows data-dependent Python
    # control flow to work, but it can in some cases lead to redundant FLOPs (done
    # both in the `bind` call below and the `core.jaxpr_as_fun` call). We use the
    # `concrete` parameter to switch this behavior, and if `concrete` is False
    # then we raise the avals to the Shaped level.
    if params['concrete']:
      instantiated_tracers = map(self.instantiate_const, tracers)
    else:
      instantiated_tracers = map(self.instantiate_const_abstracted, tracers)

    in_pvs, in_consts = unzip2(t.pval for t in instantiated_tracers)
    f = trace_to_subjaxpr(f, self.main, False)
    f, aux = partial_eval_wrapper(f, in_pvs)
    todo = partial(remat_todo, self, instantiated_tracers, tracers, params, aux)
    return f, in_consts, params, todo

def partial_eval_todo(trace, primitive, tracers, params, aux, out_flat):
  out_pvs, jaxpr, env = aux()
  out_consts, consts = split_list(out_flat, [len(out_flat) - len(jaxpr.constvars)])
  out_pvals = map(PartialVal, zip(out_pvs, out_consts))
  env_tracers = map(trace.full_raise, env)

  # Skip known invars and outvars, and lift constants as regular invars
  in_knowns = tuple(t.pval.is_known() for t in it.chain(env_tracers, tracers))
  out_unknowns = tuple(not pval.is_known() for pval in out_pvals)
  jaxpr = _drop_invars(jaxpr, in_knowns)
  jaxpr = _dce_open_jaxpr(jaxpr, out_unknowns, drop_outputs=True)

  # Known tracers get propagated as if they were constants
  known_tracers_out = [trace.new_const(pval.get_known()) for pval in out_pvals
                        if pval.is_known()]

  # Unknown tracers need to have the jaxpr set up as their recipe
  unknown_tracers_out = [JaxprTracer(trace, pval, None) for pval in out_pvals
                          if not pval.is_known()]
  unknown_tracers_in = [t for t in tracers if not t.pval.is_known()]
  const_tracers = map(trace.new_instantiated_const, consts)
  in_tracers = (*const_tracers, *env_tracers, *unknown_tracers_in)

  # Set up new params
  new_params = dict(params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
  update_params = call_param_updaters.get(primitive)
  if update_params:
    new_params = update_params(new_params, len(const_tracers), len(env_tracers),
                                [not t.pval.is_known() for t in tracers],
                                [not pval.is_known() for pval in out_pvals])

  eqn = new_eqn_recipe(in_tracers, unknown_tracers_out, primitive,
                        new_params, source_info_util.current())
  for t in unknown_tracers_out: t.recipe = eqn
  return _zip_knowns(known_tracers_out, unknown_tracers_out, out_unknowns)

@lu.transformation_with_aux
def partial_eval_wrapper(pvs: Sequence[Optional[AbstractValue]], *consts):
  in_pvals = map(PartialVal, zip(pvs, consts))
  jaxpr, (out_pvals, consts, env) = yield (in_pvals,), {}
  out_pvs, out_consts = unzip2(out_pvals)
  out = tuple(out_consts) + tuple(consts)
  yield out, (out_pvs, jaxpr, env)

@lu.transformation_with_aux
def partial_eval_wrapper_map(
    pvs: List[Optional[AbstractValue]], axis_name: core.AxisName, axis_size: int,
    in_axes: List[Optional[int]], out_axes_thunk: Callable[[], List[Optional[int]]],
    *consts):
  in_pvals = [PartialVal((pv, const)) if pv is None or in_axis is None
              else PartialVal.unknown(core.mapped_aval(axis_size, in_axis, pv))
              for pv, in_axis, const in zip(pvs, in_axes, consts)]
  with core.extend_axis_env(axis_name, axis_size, None):
    jaxpr, (out_pvals, consts, env) = yield (in_pvals,), {}
  out_pvals = [pval if pval.is_known() else
               PartialVal.unknown(core.unmapped_aval(axis_size, out_axis, pval[0]))
               for pval, out_axis in zip(out_pvals, out_axes_thunk())]
  out_pvs, out_consts = unzip2(out_pvals)
  out = tuple(out_consts) + tuple(consts)
  yield out, (out_pvs, jaxpr, env)

@lu.transformation_with_aux
def count_outputs(*args, **kwargs):
  ans = yield args, kwargs
  yield ans, len(ans)

custom_partial_eval_rules: Dict[core.Primitive, Callable] = {}
call_param_updaters: Dict[core.Primitive, Callable] = {}


def abstract_eval_fun(fun, *avals, debug_info=None, **params):
  _, avals_out, _ = trace_to_jaxpr_dynamic(
      lu.wrap_init(fun, params), avals, debug_info)
  assert all(isinstance(aval, AbstractValue) for aval in avals_out)
  return avals_out


JaxprTracerRecipe = Union['JaxprEqnRecipe', 'LambdaBinding', 'FreeVar',
                          'ConstVar', Literal, core.Unit]

class JaxprTracer(Tracer):
  __slots__ = ['pval', 'recipe']

  def __init__(self, trace: JaxprTrace, pval: PartialVal,
               recipe: Optional[JaxprTracerRecipe]):
    assert isinstance(pval, PartialVal)
    pv, const = pval
    if isinstance(const, Tracer) and const._trace.level >= trace.level:
      raise core.escaped_tracer_error(
          const, "Tracer from a higher level: {} in trace {}".format(const, trace))
    self._trace = trace
    self.pval = pval
    self.recipe = recipe

  def __repr__(self):
    return 'Traced<{}:{}>'.format(self.aval, self._trace)

  @property
  def aval(self) -> AbstractValue:
    return self.pval.get_aval()

  @property
  def parents(self) -> Sequence['JaxprTracer']:
    if isinstance(self.recipe, JaxprEqnRecipe):
      return self.recipe.invars
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

# TODO(necula): this could return a ClosedJaxpr with out_pvals
def trace_to_jaxpr(fun: lu.WrappedFun, pvals: Sequence[PartialVal],
                   instantiate: Union[bool, Sequence[bool]] = False,
                   ) -> Tuple[Jaxpr, Tuple[PartialVal, ...], Tuple[core.Value, ...]]:
  """Traces a function into a Jaxpr, given PartialVals for inputs.

  Returns (`jaxpr`, `out_pvals`, `consts`). The `jaxpr` contains only the
  computation that depends on unknown inputs. The `out_pvals` are the PartialVal
  for the outputs. The intermediate values that depend only on known inputs and
  are needed to compute the output of `jaxpr` are in `consts` and are passed in
  as the constvars of the `jaxpr`. The handling of the known outputs depends on
  `instantiate`.

  For example, given `fun` defined as follows::

    def fun(ki, ui):  # ki will be a known input in this example
      ka = ki + 2
      kb = ka + 3
      return (kb, ui + ka)

  with `ki` the known PartialVal `1.`, and `ui` an unknown PartialVal. The only
  computation that depends on unknown inputs is `ui + ka` and will be the only
  computation in the body of the `jaxpr`. This computation depends on the known
  intermediate value `ka`, which will be computed statically. Currently, such
  constants are either embedded in the Jaxpr if they are scalars, or passed as a
  constvar to `jaxpr`, and then the value of the actual constant will be in
  `consts`:

  When `instantiate=False` we get::

    jaxpr =
      { lambda ka ; ki ui.
        let c = add ui ka
        in (*, c) }   # known outputs are `*`
    out_pvals = [PartialVal.known(6), PartialVal.unknown(ShapedArray)]
    consts = [3]  # the constant for `ka`

  When `instantiate=True` we get::

    jaxpr =
      { lambda ka kb ; ki ui.
        let c = add ui ka
        in (kb, c) }   # known output are explicit
    out_pvals = [PartialVal.unknown(ConcreteArray(6)), PartialVal.unknown(ShapedArray)]
    consts = [3, 6]  # values for `ka` and `kb` constvars
  """
  with core.new_main(JaxprTrace) as main:
    fun = trace_to_subjaxpr(fun, main, instantiate)
    jaxpr, (out_pvals, consts, env) = fun.call_wrapped(pvals)
    assert not env
    del main, fun, env

  return jaxpr, out_pvals, consts


@lu.transformation
def trace_to_subjaxpr(main: core.MainTrace, instantiate: Union[bool, Sequence[bool]],
                      pvals: Sequence[PartialVal]):
  assert all([isinstance(pv, PartialVal) for pv in pvals]), pvals
  trace = JaxprTrace(main, core.cur_sublevel())
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

def instantiate_const_at(trace: JaxprTrace, instantiate: bool, tracer):
  if instantiate:
    return trace.instantiate_const(trace.full_raise(tracer))
  else:
    return tracer


FreeVar = namedtuple('FreeVar', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])
LambdaBinding = namedtuple('LambdaBinding', [])
class JaxprEqnRecipe(NamedTuple):
  eqn_id: object
  invars: Sequence[JaxprTracer]
  outvars: 'Sequence[ref[JaxprTracer]]'
  primitive: core.Primitive
  params: Dict[str, Any]
  source_info: Optional[source_info_util.Traceback]

def new_eqn_recipe(invars: Sequence[JaxprTracer],
                   outvars: Sequence[JaxprTracer],
                   primitive: core.Primitive,
                   params: Dict[str, Any],
                   source_info: Optional[source_info_util.Traceback]
                  ) -> JaxprEqnRecipe:
  """Constructs a new JaxEqnRecipe.

  Params:
    invars: the tracers for the primitive inputs.
    outvars: the tracers for the primitive outputs.
    primitive: the primitive.
    params: the primitive params
  """
  # TODO(necula): move these checks to core.check_jaxpr, and call in more places
  if primitive.call_primitive or primitive.map_primitive:
    assert "call_jaxpr" in params
  if primitive.map_primitive:
    assert ("in_axes" in params and
            len(params["in_axes"]) == len(params["call_jaxpr"].invars))
    assert ("donated_invars" in params and
            len(params["donated_invars"]) == len(params["call_jaxpr"].invars))
  return JaxprEqnRecipe(object(), tuple(invars), map(ref, outvars), primitive,
                        params, source_info)


def recipe_to_eqn(getvar: Callable[[JaxprTracer], core.Atom],
                  recipe: JaxprEqnRecipe) -> core.JaxprEqn:
  _, in_tracers, out_tracer_refs, primitive, params, source_info = recipe
  out_tracers = [t_ref() for t_ref in out_tracer_refs]
  invars  = [getvar(t) for t in in_tracers]
  outvars = [core.dropvar if t is None else cast(core.Var, getvar(t))
             for t in out_tracers]
  return new_jaxpr_eqn(invars, outvars, primitive, params, source_info)

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
  newvar = core.gensym()
  t_to_var: Dict[int, core.Atom] = {}
  def getvar(t: JaxprTracer) -> core.Atom:
    var = t_to_var.get(id(t))
    if var is None:
      aval = t.pval.get_aval() if not t.pval.is_known() else abstract_unit
      var = t_to_var[id(t)] = newvar(aval)
    return var
  sorted_tracers = toposort(out_tracers)
  invars = map(getvar, in_tracers)
  eqns: List[core.JaxprEqn] = []
  env: Dict[core.Var, Any] = {}
  consts: Dict[core.Var, Any] = {}
  const_to_var: Dict[int, core.Var] = {}
  def getconstvar(c):
    var = const_to_var.get(id(c))
    if var is None:
      var = const_to_var[id(c)] = newvar(get_aval(c))
    return var
  processed_eqn_ids = set()
  for t in sorted_tracers:
    recipe = t.recipe
    if isinstance(recipe, JaxprEqnRecipe):
      if recipe.eqn_id not in processed_eqn_ids:
        eqns.append(recipe_to_eqn(getvar, recipe))
        processed_eqn_ids.add(recipe.eqn_id)
    elif isinstance(recipe, LambdaBinding):
      if not any(t is in_tracer for in_tracer in in_tracers):
        raise core.escaped_tracer_error(
            t, "Tracer not among input tracers {}".format(t))
      assert in_tracers, "Lambda binding with no args"
    elif isinstance(recipe, FreeVar):
      env[cast(core.Var, getvar(t))] = recipe.val
    elif isinstance(recipe, ConstVar):
      v = t_to_var[id(t)] = getconstvar(recipe.val)
      consts[v] = recipe.val
    elif isinstance(recipe, Literal):
      t_to_var[id(t)] = recipe
    elif recipe is unit:
      t_to_var[id(t)] = unitvar
    else:
      raise TypeError(recipe)

  env_vars, env_vals = unzip2(env.items())
  const_vars, const_vals = unzip2(consts.items())
  # The env_vars are pre-pended to the invars
  jaxpr = Jaxpr(const_vars, [*env_vars, *invars], map(getvar, out_tracers), eqns)
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  return jaxpr, const_vals, env_vals

@cache()
def convert_constvars_jaxpr(jaxpr: Jaxpr):
  """Moves the constvars to the start of invars."""
  return Jaxpr(constvars=(),
               invars=jaxpr.constvars + jaxpr.invars,
               outvars=jaxpr.outvars, eqns=jaxpr.eqns)

def convert_envvars_to_constvars(jaxpr: Jaxpr, num_env_vars: int):
  config.jax_enable_checks and core.check_jaxpr(jaxpr)
  env_vars, invars = split_list(jaxpr.invars, [num_env_vars])
  converted_jaxpr = Jaxpr(constvars=jaxpr.constvars + env_vars,
                          invars=invars, outvars=jaxpr.outvars, eqns=jaxpr.eqns)
  config.jax_enable_checks and core.check_jaxpr(converted_jaxpr)
  return converted_jaxpr


def _split_aval(unknown: bool, aval: AbstractValue) -> Tuple[AbstractValue, AbstractValue]:
  return (abstract_unit, aval) if unknown else (aval, abstract_unit)

def partial_eval_jaxpr(jaxpr: ClosedJaxpr, unknowns: Sequence[bool],
                       instantiate: Union[bool, Sequence[bool]],
                       ) -> Tuple[ClosedJaxpr, ClosedJaxpr, Sequence[bool]]:
  """Specializes a Jaxpr given an indication of which inputs are known.

  Returns: (jaxpr_known, jaxpr_unknown, out_unknowns).

  `out_unknowns` specifies which outputs are unknown (depend on some unknown inputs).
  `jaxpr_known` takes the same inputs as `jaxpr`, ignores the unknown inputs,
  and performs *all* the computation in `jaxpr` that depends only on the known inputs.
  Outputs correspond to those of `jaxpr`, with the unknown ones replaced with `*`,
  appended with the known residuals (the intermediate computations in `jaxpr`
  that depend only on known inputs and that are needed to compute the unknown outputs).

  `jaxpr_unknown` takes the same inputs as `jaxpr` along with the known residuals
  computed by `jaxpr_known` and returns the same outputs as `jaxpr` with the known
  outputs replaced by `*`.

  Roughly, `jaxpr(ki, ui)` is decomposed assuming `ki` and `ui` are the known and respectively
  unknown inputs into:

    jaxpr(ki, ui) = let kout, _, kresidual = jaxpr_known(kin, *)
                    let _, uout = jaxpr_unknown(ki, ui, kresidual)
                    in (kout, uout)

  For example, if `jaxpr` is lambda ki, ui: let ka = ki + 2
                                            in (ki + 3, ui + ka)"
  then
    `jaxpr_known` = lambda ki, ui: let ka = ki + 2
                                    in (ki + 3, *, ka)
    'jaxpr_unknown` = lambda ki, ui, ka: (*, ui + ka)

  Note that if instantiate is True for a given output, then jaxpr_known always returns a
  unit in its place. So when instantiate is True, the expectation is the one doesn't
  run `jaxpr_known` for any of its outputs, but only to generate residuals that will allow
  to obtain the full outputs once `jaxpr_unknown` is ran. Outputs known ahead of time will
  simply get passed as residual constants and returned immediately.
  """
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))

  cell = []
  def fun(*vals):
    pvals = [PartialVal.unknown(aval) if uk else PartialVal.known(val)
            for aval, val, uk in zip(jaxpr.in_avals, vals, unknowns)]
    jaxpr_2, out_pvals_2, consts_2 = trace_to_jaxpr(f, pvals, instantiate=instantiate)
    out_pvs_2, out_consts_2 = unzip2(out_pvals_2)
    cell.append((out_pvs_2, jaxpr_2, len(consts_2)))
    return out_consts_2 + consts_2

  # For jaxpr_known we pass core.unit for the unknown inputs, and known PartialVal for the
  # known inputs.
  in_avals = [abstract_unit if uk else a for a, uk in zip(jaxpr.in_avals, unknowns)]
  jaxpr_1, out_avals, consts_1 = trace_to_jaxpr_dynamic(lu.wrap_init(fun), in_avals)
  (out_pvs_2, jaxpr_2, num_res), = cell
  assert len(jaxpr_2.constvars) == num_res

  #   jaxpr :: a -> b
  # jaxpr_1 :: a1 -> [b1, res]
  # jaxpr_2 :: res | a2 -> b2
  # jaxpr_2 :: [a2, res] -> b2
  jaxpr_2 = convert_constvars_jaxpr(jaxpr_2)
  jaxpr_2.invars = jaxpr_2.invars[num_res:] + jaxpr_2.invars[:num_res]
  for var, unknown in zip(jaxpr_2.invars[:len(unknowns)], unknowns):
    if not unknown:
      var.aval = abstract_unit

  uk_out = [pv is not None for pv in out_pvs_2]

  in_avals_1, in_avals_2 = unzip2(map(_split_aval, unknowns, jaxpr.in_avals))
  out_avals_1, out_avals_2 = unzip2(map(_split_aval, uk_out, jaxpr.out_avals))
  # out_avals_1 and in_avals_2 need the residuals added
  res_avals = out_avals[len(jaxpr.out_avals):]
  assert len(res_avals) == num_res
  out_avals_1 = [*out_avals_1, *res_avals]
  in_avals_2 = [*in_avals_2, *res_avals]

  return ClosedJaxpr(jaxpr_1, consts_1), ClosedJaxpr(jaxpr_2, ()), uk_out


remat_call_p = core.CallPrimitive('remat_call')

@remat_call_p.def_custom_bind
def remat_call(fun, *args, **params):
    return core.final_style_bind(attrgetter('process_remat_call'),
                                 remat_call_p, fun, args, params)

def remat_todo(trace, instantiated_tracers, tracers, params, aux, out_flat):
  # This part is identical to partial_eval_todo, just unpacking aux and results.
  out_pvs, jaxpr, env = aux()
  out_consts, consts = split_list(out_flat, [len(out_flat) - len(jaxpr.constvars)])
  out_pvals = map(PartialVal, zip(out_pvs, out_consts))
  env_tracers = map(trace.full_raise, env)

  # Convert consts to inputs, since they may contain Tracer instances.
  jaxpr = convert_constvars_jaxpr(jaxpr)
  const_tracers = map(trace.new_instantiated_const, consts)

  # Since we traced with everything marked as unknown, but we need to know which
  # outputs are known/unknown, we use partial_eval_jaxpr to get out_unknowns.
  closed_jaxpr = core.ClosedJaxpr(jaxpr, ())
  in_unknowns = ([False] * len(consts) +
                 [not t.is_known() for t in it.chain(env_tracers, tracers)])
  jaxpr_known, jaxpr_unknown, out_unknowns = partial_eval_jaxpr(
      closed_jaxpr, in_unknowns, instantiate=False)  # type: ignore
  out_knowns = [not b for b in out_unknowns]
  out_known_pvals, out_unknown_pvals = _partition_knowns(out_pvals, out_unknowns)

  # Next, we need values for the outputs that should be known. Since consts
  # weren't passed through Python for evaluation, we need to evaluate
  # jaxpr_known, minus the residual outputs that we don't need. When
  # `concrete=True`, as an optimization we can avoid redoing *some* redundant
  # FLOPs, namely those that produced concrete avals at the output, simply by
  # using those as computed values.
  to_compute = [type(pval[0]) is not ConcreteArray
                for uk, pval in zip(out_unknowns, out_pvals) if not uk]
  num_outputs = len(jaxpr_unknown.out_avals)
  num_res = len(jaxpr_known.out_avals) - num_outputs
  jaxpr_known_nores = _dce_jaxpr(jaxpr_known, out_knowns + [False] * num_res, drop_outputs=True)
  jaxpr_known_comp = _dce_jaxpr(jaxpr_known_nores, to_compute)
  _, in_consts = unzip2(t.pval for t in it.chain(env_tracers, tracers))
  reconstructed_consts = core.jaxpr_as_fun(jaxpr_known_comp)(*consts, *in_consts)
  out_known_pvals = map(_reconstruct_pval, out_known_pvals, reconstructed_consts)

  # Known outputs should keep propagating as constants
  assert all(pv.is_known() for pv in out_known_pvals)
  known_output_tracers = [trace.new_const(pval.get_known())
                          for pval in out_known_pvals]
  # Unknown outputs get wrapped in tracers with the appropriate recipe
  unknown_output_tracers = [JaxprTracer(trace, out_pval, None)
                            for out_pval in out_unknown_pvals]

  # dce jaxpr outputs
  new_jaxpr = _dce_jaxpr(closed_jaxpr, out_unknowns, drop_outputs=True).jaxpr
  new_params = dict(params, call_jaxpr=new_jaxpr, differentiated=True)

  # set up eqn for unknown outputs
  in_tracers = (*const_tracers, *env_tracers, *instantiated_tracers)
  eqn = new_eqn_recipe(in_tracers, unknown_output_tracers, remat_call_p,
                       new_params, source_info_util.current())
  for t in unknown_output_tracers: t.recipe = eqn
  return _zip_knowns(known_output_tracers, unknown_output_tracers, out_unknowns)

def _partition_knowns(pvals, unknowns: Sequence[bool]):
  return ([e for e, unknown in zip(pvals, unknowns) if not unknown],
          [e for e, unknown in zip(pvals, unknowns) if unknown])

def _zip_knowns(known_list, unknown_list, which_unknown: Sequence[bool]):
  known_iter, unknown_iter = iter(known_list), iter(unknown_list)
  return [next(unknown_iter) if uk else next(known_iter) for uk in which_unknown]


def _dce_jaxpr(closed_jaxpr: ClosedJaxpr, outputs: Sequence[bool], drop_outputs=False) -> ClosedJaxpr:
  new_jaxpr = _dce_open_jaxpr(closed_jaxpr.jaxpr, tuple(outputs), drop_outputs)
  return core.ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)

@cache()
def _dce_open_jaxpr(jaxpr: Jaxpr, outputs: Tuple[bool, ...], drop_outputs=False) -> Jaxpr:
  # This dead-code elimination is pretty rudimentary, and in particular doesn't
  # nontrivially DCE through scan, call, or other higher-order primitives.
  # TODO(mattjj): better DCE
  if drop_outputs:
    new_outvars = [var for var, output in zip(jaxpr.outvars, outputs) if output]
  else:
    new_outvars = [var if output else unitvar
                   for var, output in zip(jaxpr.outvars, outputs)]

  needed_vars = {v for v in new_outvars if type(v) is not Literal}
  new_eqns = []
  for eqn in jaxpr.eqns[::-1]:
    if set(eqn.outvars) & needed_vars:
      new_eqns.append(eqn)
      needed_vars.update(v for v in eqn.invars if type(v) is not Literal)
  new_eqns = new_eqns[::-1]
  return core.Jaxpr(jaxpr.constvars, jaxpr.invars,
                    new_outvars, new_eqns)

@cache()
def _drop_invars(jaxpr: Jaxpr, drop: Tuple[bool, ...]):
  return core.Jaxpr(jaxpr.constvars, [v for v, d in zip(jaxpr.invars, drop) if not d],
                    jaxpr.outvars, jaxpr.eqns)


def _reconstruct_pval(pval1: PartialVal, const2: core.Value):
  pv1, _ = pval1
  if pval1.is_known():
    return pval1
  else:
    if type(pv1) is ConcreteArray:
      return PartialVal.known(pv1.val)  # pytype: disable=attribute-error
    else:
      return PartialVal.known(const2)


def move_binders_to_front(closed_jaxpr: ClosedJaxpr, to_move: Sequence[bool]) -> ClosedJaxpr:
  """Reorder the `invars` to move to front the ones for which `to_move` is True."""
  assert not closed_jaxpr.jaxpr.constvars
  assert len(closed_jaxpr.in_avals) == len(to_move)
  new_invars = _move_to_front(closed_jaxpr.jaxpr.invars, to_move)
  new_jaxpr = core.Jaxpr((), new_invars, closed_jaxpr.jaxpr.outvars,
                         closed_jaxpr.jaxpr.eqns)
  new_closed_jaxpr = core.ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
  return new_closed_jaxpr

def _move_to_front(lst: Sequence, to_move: Sequence[bool]) -> Sequence:
  return ([elt for elt, move in zip(lst, to_move) if move] +
          [elt for elt, move in zip(lst, to_move) if not move])


class DynamicJaxprTracer(core.Tracer):
  __slots__ = ['aval']

  def __init__(self, trace, aval, line_info=None):
    self._trace = trace
    self._line_info = line_info
    self.aval = aval

  def full_lower(self):
    return self

  def _contents(self):
    return ()

  def _origin_msg(self):
    invar_pos, progenitor_eqns = self._trace.frame.find_progenitors(self)
    dbg = self._trace.main.debug_info
    if dbg is None:
      return ""
    if invar_pos:
      origin = (f"While tracing the function {dbg.func_src_info} "
                f"for {dbg.traced_for}, "
                "this concrete value was not available in Python because it "
                f"depends on the value{'s' if len(invar_pos) > 1 else ''} "
                f"of {dbg.arg_info(invar_pos)}.")
    elif progenitor_eqns:
      msts = [f"  operation {core.pp_eqn(eqn, print_shapes=True)}\n"
              f"    from line {source_info_util.summarize(eqn.source_info)}"
              for eqn in progenitor_eqns]
      origin = (f"While tracing the function {dbg.func_src_info} "
                f"for {dbg.traced_for}, "
                "this value became a tracer due to JAX operations on these lines:"
                "\n\n" + "\n\n".join(msts))
    else:
      origin = (f"The error occured while tracing the function {dbg.func_src_info} "
                f"for {dbg.traced_for}.")
    return "\n" + origin

  def _assert_live(self) -> None:
    if not self._trace.main.jaxpr_stack:  # type: ignore
      raise core.escaped_tracer_error(self, None)

class JaxprStackFrame:
  __slots__ = ['gensym', 'tracer_to_var', 'constid_to_var', 'constvar_to_val',
               'tracers', 'eqns', 'invars']

  def __init__(self):
    self.gensym = core.gensym()
    self.tracer_to_var = {}
    self.constid_to_var = {}
    self.constvar_to_val = {}
    self.tracers = []   # circ refs, frame->tracer->trace->main->frame,
    self.eqns = []      # cleared when we pop frame from main
    self.invars = []

  def to_jaxpr(self, in_tracers, out_tracers):
    invars = [self.tracer_to_var[id(t)] for t in in_tracers]
    outvars = [self.tracer_to_var[id(t)] for t in out_tracers]
    constvars, constvals = unzip2(self.constvar_to_val.items())
    jaxpr = Jaxpr(constvars, invars, outvars, self.eqns)
    jaxpr, constvals = _inline_literals(jaxpr, constvals)
    out_avals = [t.aval for t in out_tracers]
    return jaxpr, out_avals, constvals

  def newvar(self, aval):
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

def _inline_literals(jaxpr, constvals):
  consts = dict(zip(jaxpr.constvars, constvals))
  newvar = core.gensym()
  newvars = {}
  var = lambda v: newvars.get(v) or newvars.setdefault(v, newvar(v.aval))

  def lit(var: core.Var) -> Optional[Any]:
    val = consts.get(var)
    if type(val) in core.literalable_types and not np.shape(val):
      return Literal(val)
    else:
      return None

  used = {v for eqn in jaxpr.eqns for v in eqn.invars} | set(jaxpr.outvars)
  new_constvars = [var(v) for v in jaxpr.constvars if not lit(v)]
  new_constvals = [c for v, c in zip(jaxpr.constvars, constvals) if not lit(v)]
  new_invars = [var(v) for v in jaxpr.invars]
  new_eqns = []
  for eqn in jaxpr.eqns:
    invars = [lit(v) or var(v) for v in eqn.invars]
    if (eqn.primitive is core.convert_element_type_p and type(invars[0]) is Literal):
      # constant-fold dtype conversion of literals to be inlined
      consts[eqn.outvars[0]] = np.array(invars[0].val, eqn.params['new_dtype'])
    else:
      # might do DCE here, but we won't until we're more careful about effects
      outvars = [var(v) if v in used else dropvar for v in eqn.outvars]
      new_eqns.append(new_jaxpr_eqn(invars, outvars, eqn.primitive, eqn.params,
                                    eqn.source_info))
  new_outvars = [lit(v) or var(v) for v in jaxpr.outvars]
  new_jaxpr = Jaxpr(new_constvars, new_invars, new_outvars, new_eqns)
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

  def new_const(self, val):
    aval = raise_to_shaped(get_aval(val), weak_type=dtypes.is_weakly_typed(val))
    tracer = DynamicJaxprTracer(self, aval, source_info_util.current())
    self.frame.tracers.append(tracer)
    var = self.frame.tracer_to_var[id(tracer)] = self.getconstvar(val)
    self.frame.constvar_to_val[var] = val
    return tracer

  pure = lift = sublift = new_const

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

  def getconstvar(self, c):
    var = self.frame.constid_to_var.get(id(c))
    if var is None:
      var = self.frame.constid_to_var[id(c)] = self.frame.newvar(get_aval(c))
    return var

  def instantiate_const(self, val):
    if (isinstance(val, Tracer) and val._trace.main is self.main
        and val._trace.sublevel == self.sublevel):
      return val
    else:
      return self.new_const(val)

  def process_primitive(self, primitive, tracers, params):
    avals = [t.aval for t in tracers]
    out_avals = primitive.abstract_eval(*avals, **params)
    out_avals = [out_avals] if not primitive.multiple_results else out_avals
    source_info = source_info_util.current()
    out_tracers = [DynamicJaxprTracer(self, a, source_info) for a in out_avals]
    invars = map(self.getvar, tracers)
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn(invars, outvars, primitive, params, source_info)
    self.frame.eqns.append(eqn)
    return out_tracers if primitive.multiple_results else out_tracers.pop()

  def process_call(self, call_primitive, f, tracers, params):
    in_avals = [t.aval for t in tracers]
    with core.new_sublevel():
      jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(f, self.main, in_avals)
    if params.get('inline', False):
      return core.eval_jaxpr(jaxpr, consts, *tracers)
    source_info = source_info_util.current()
    out_tracers = [DynamicJaxprTracer(self, a, source_info) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    new_params = dict(params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
    update_params = call_param_updaters.get(call_primitive)
    if update_params:
      new_params = update_params(new_params, 0, len(constvars),
                                 [True] * len(tracers), [True] * len(outvars))
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, call_primitive,
                        new_params, source_info)
    self.frame.eqns.append(eqn)
    return out_tracers

  def process_map(self, map_primitive, f, tracers, params):
    in_avals = [t.aval for t in tracers]
    axis_name, axis_size = params['axis_name'], params['axis_size']
    reduced_in_avals = [core.mapped_aval(axis_size, in_axis, a)
                        if in_axis is not None else a
                        for a, in_axis in zip(in_avals, params['in_axes'])]
    with core.extend_axis_env(axis_name, axis_size, None):  # type: ignore
      with core.new_sublevel():
        jaxpr, reduced_out_avals, consts = trace_to_subjaxpr_dynamic(
            f, self.main, reduced_in_avals)
      out_avals = [core.unmapped_aval(params['axis_size'], out_axis, a)
                  if out_axis is not None else a
                  for a, out_axis in zip(reduced_out_avals, params['out_axes_thunk']())]
      source_info = source_info_util.current()
      out_tracers = [DynamicJaxprTracer(self, a, source_info) for a in out_avals]
      invars = map(self.getvar, tracers)
      constvars = map(self.getvar, map(self.instantiate_const, consts))
      outvars = map(self.makevar, out_tracers)
      new_params = dict(params, call_jaxpr=convert_constvars_jaxpr(jaxpr))
      update_params = call_param_updaters.get(map_primitive)
      if update_params:
        new_params = update_params(new_params, 0, len(constvars),
                                   [True] * len(tracers), [True] * len(outvars))
      eqn = new_jaxpr_eqn([*constvars, *invars], outvars, map_primitive,
                          new_params, source_info)
      self.frame.eqns.append(eqn)
    return out_tracers

  def process_custom_jvp_call(self, prim, fun, jvp, tracers):
    in_avals = [t.aval for t in tracers]
    with core.new_sublevel():
      fun_jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, self.main, in_avals)
    closed_fun_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(fun_jaxpr), ())
    jvp_jaxpr_thunk = _memoize(
        lambda: trace_to_subjaxpr_dynamic(jvp, self.main, 2 * in_avals)[::2])
    out_tracers = [DynamicJaxprTracer(self, a) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, prim.initial_style,
                        dict(fun_jaxpr=closed_fun_jaxpr,
                             jvp_jaxpr_thunk=jvp_jaxpr_thunk,
                             num_consts=len(consts)),
                        source_info_util.current())
    self.frame.eqns.append(eqn)
    return out_tracers

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees):
    in_avals = [t.aval for t in tracers]
    with core.new_sublevel():
      fun_jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, self.main, in_avals)
    closed_fun_jaxpr = core.ClosedJaxpr(convert_constvars_jaxpr(fun_jaxpr), ())
    fwd_jaxpr_thunk = _memoize(
        lambda: trace_to_subjaxpr_dynamic(fwd, self.main, in_avals)[::2])
    out_tracers = [DynamicJaxprTracer(self, a) for a in out_avals]
    invars = map(self.getvar, tracers)
    constvars = map(self.getvar, map(self.instantiate_const, consts))
    outvars = map(self.makevar, out_tracers)
    eqn = new_jaxpr_eqn([*constvars, *invars], outvars, prim.initial_style,
                        dict(fun_jaxpr=closed_fun_jaxpr,
                             fwd_jaxpr_thunk=fwd_jaxpr_thunk,
                             num_consts=len(consts),
                             bwd=bwd, out_trees=out_trees),
                        source_info_util.current())
    self.frame.eqns.append(eqn)
    return out_tracers

def _memoize(thunk):
  cell = []
  saved_state = core.thread_local_state.trace_state.copy()
  def memoized():
    if not cell:
      prev_state = core.thread_local_state.trace_state
      core.thread_local_state.trace_state = saved_state
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

PyTreeDef = Any

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


def trace_to_jaxpr_dynamic(fun: lu.WrappedFun,
                           in_avals: Sequence[AbstractValue],
                           debug_info: Optional[DebugInfo] = None):
  with core.new_main(DynamicJaxprTrace, dynamic=True) as main:  # type: ignore
    main.debug_info = debug_info  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, main, in_avals)
    del main, fun
  return jaxpr, out_avals, consts

def trace_to_subjaxpr_dynamic(fun: lu.WrappedFun, main: core.MainTrace,
                              in_avals: Sequence[AbstractValue]):
  frame = JaxprStackFrame()
  with extend_jaxpr_stack(main, frame):
    trace = DynamicJaxprTrace(main, core.cur_sublevel())
    in_tracers = map(trace.new_arg, in_avals)
    ans = fun.call_wrapped(*in_tracers)
    out_tracers = map(trace.full_raise, ans)
    jaxpr, out_avals, consts = frame.to_jaxpr(in_tracers, out_tracers)
    del fun, main, trace, frame, in_tracers, out_tracers, ans
  return jaxpr, out_avals, consts

@contextlib.contextmanager
def extend_jaxpr_stack(main, frame):
  main.jaxpr_stack = main.jaxpr_stack + (frame,)
  try:
    yield
  finally:
    assert frame is main.jaxpr_stack[-1]
    main.jaxpr_stack = main.jaxpr_stack[:-1]

def trace_to_jaxpr_final(fun: lu.WrappedFun,
                         in_avals: Sequence[AbstractValue],
                         debug_info: Optional[DebugInfo] = None):
  with core.new_base_main(DynamicJaxprTrace) as main:  # type: ignore
    main.debug_info = debug_info  # type: ignore
    main.jaxpr_stack = ()  # type: ignore
    with core.new_sublevel():
      jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(fun, main, in_avals)
    del fun, main
  return jaxpr, out_avals, consts

def partial_eval_to_jaxpr_dynamic(fun: lu.WrappedFun, in_pvals: Sequence[PartialVal]):
  # This function provides a partial evaluation behavior used by Flax. We can't
  # use trace_to_jaxpr directly because of an interaction with the curent
  # custom_derivatives.py, which we work around by adding the EvalTrace.
  # TODO(mattjj): alias to trace_to_jaxpr after revising custom_derivatives.py
  with core.new_main(core.EvalTrace, dynamic=True) as _:  # type: ignore
    return trace_to_jaxpr(fun, in_pvals)
