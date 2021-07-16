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


import functools
import itertools as it
from typing import Any, Callable, Dict

import jax
from . import partial_eval as pe
from ..config import config
from .. import core
from .._src.dtypes import dtype, float0
from ..core import (Trace, Tracer, get_aval, call_p, Primitive, Literal,
                    raise_to_shaped)
from .._src.ad_util import (add_jaxvals, add_jaxvals_p, zeros_like_jaxval,
                              zeros_like_aval, zeros_like_p, Zero)
from .._src.util import (unzip2, safe_map, safe_zip, partial, split_list,
                         wrap_name, as_hashable_function)
from ..tree_util import register_pytree_node
from .. import linear_util as lu
from ..api_util import flatten_fun, flatten_fun_nokwargs
from ..tree_util import tree_flatten, tree_unflatten, Partial
from .._src import source_info_util

zip = safe_zip
map = safe_map
def identity(x): return x


def jvp(fun: lu.WrappedFun, has_aux=False, instantiate=True) -> Any:
  if not has_aux:
    return jvpfun(jvp_subtrace(fun), instantiate)
  else:
    fun, aux = jvp_subtrace_aux(fun)
    return jvpfun(fun, instantiate), aux


@lu.transformation
def jvpfun(instantiate, primals, tangents):
  tangents = [Zero.from_value(t) if not isinstance(t, Zero)
              and dtype(t) is float0 else t for t in tangents]
  with core.new_main(JVPTrace) as main:
    out_primals, out_tangents = yield (main, primals, tangents), {}
    del main
  if type(instantiate) is bool:
    instantiate = [instantiate] * len(out_tangents)
  out_tangents = [instantiate_zeros(t) if inst else t for t, inst
                  in zip(out_tangents, instantiate)]
  yield out_primals, out_tangents

@lu.transformation
def jvp_subtrace(main, primals, tangents):
  trace = JVPTrace(main, core.cur_sublevel())
  for x in list(primals) + list(tangents):
    if isinstance(x, Tracer):
      assert x._trace.level < trace.level
  in_tracers = [JVPTracer(trace, x, t) if type(t) is not Zero else x
                for x, t in zip(primals, tangents)]
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  yield unzip2([(out_tracer.primal, out_tracer.tangent)
                for out_tracer in out_tracers])

@lu.transformation_with_aux
def jvp_subtrace_aux(main, primals, tangents):
  trace = JVPTrace(main, core.cur_sublevel())
  for x in list(primals) + list(tangents):
    if isinstance(x, Tracer):
      assert x._trace.level < trace.level
  ans, aux = yield map(partial(JVPTracer, trace), primals, tangents), {}
  ans_tracers = map(trace.full_raise, ans)
  out_primals, out_tangents = unzip2((t.primal, t.tangent) for t in ans_tracers)
  aux_primals = [core.full_lower(x.primal)
                 if isinstance(x, JVPTracer) and x._trace.level == trace.level
                 else x for x in aux]
  yield (out_primals, out_tangents), aux_primals

def linearize(traceable, *primals, **kwargs):
  has_aux = kwargs.pop('has_aux', False)
  if not has_aux:
    jvpfun = jvp(traceable)
  else:
    jvpfun, aux = jvp(traceable, has_aux=True)

  in_pvals = (tuple(pe.PartialVal.known(p) for p in primals)
              + tuple(pe.PartialVal.unknown(get_aval(p).at_least_vspace())
                    for p in primals))
  _, in_tree = tree_flatten(((primals, primals), {}))
  jvpfun_flat, out_tree = flatten_fun(jvpfun, in_tree)
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(jvpfun_flat, in_pvals)
  out_primals_pvals, out_tangents_pvals = tree_unflatten(out_tree(), out_pvals)
  assert all(out_primal_pval.is_known() for out_primal_pval in out_primals_pvals)
  _, out_primals_consts = unzip2(out_primals_pvals)
  jaxpr.invars = jaxpr.invars[len(primals):]
  jaxpr.outvars = jaxpr.outvars[len(out_primals_pvals):]
  if not has_aux:
    return out_primals_consts, out_tangents_pvals, jaxpr, consts
  else:
    return out_primals_consts, out_tangents_pvals, jaxpr, consts, aux()

def vjp(traceable, primals, has_aux=False, reduce_axes=()):
  if not has_aux:
    out_primals, pvals, jaxpr, consts = linearize(traceable, *primals)
  else:
    out_primals, pvals, jaxpr, consts, aux = linearize(traceable, *primals, has_aux=True)

  def unbound_vjp(pvals, jaxpr, consts, *cts):
    cts = tuple(map(ignore_consts, cts, pvals))
    dummy_args = [UndefinedPrimal(v.aval) for v in jaxpr.invars]
    arg_cts = backward_pass(jaxpr, reduce_axes, consts, dummy_args, cts)
    return map(instantiate_zeros, arg_cts)

  # Ensure that vjp_ is a PyTree so that we can pass it from the forward to the backward
  # pass in a custom VJP.
  vjp_ =  Partial(partial(unbound_vjp, pvals, jaxpr), consts)
  if not has_aux:
    return out_primals, vjp_
  else:
    return out_primals, vjp_, aux

def ignore_consts(ct, pval):
  aval, const = pval
  if isinstance(aval, core.AbstractValue):
    return ct
  elif aval is None:
    return core.unit
  else:
    raise TypeError(aval)

def unpair_pval(pval):
  aval, const = pval
  const_1, const_2 = const
  if aval is None:
    return (None, const_1), (None, const_2)
  else:
    aval_1, aval_2 = aval
    return (aval_1, const_1), (aval_2, const_2)

def replace_float0s(primal, tangent):
  if dtype(tangent) is float0:
    return core.zeros_like_float0(tangent, dtype(primal))
  else:
    return tangent

def recast_to_float0(primal, tangent):
  if core.primal_dtype_to_tangent_dtype(dtype(primal)) == float0:
    return Zero(get_aval(primal).at_least_vspace())
  else:
    return tangent

# NOTE: The FIXMEs below are caused by primal/tangent mixups (type errors if you will)
def backward_pass(jaxpr: core.Jaxpr, reduce_axes, consts, primals_in, cotangents_in):
  if all(type(ct) is Zero for ct in cotangents_in):
    return map(lambda v: Zero(v.aval), jaxpr.invars)

  def write_cotangent(prim, v, ct):
    # assert v not in primal_env
    assert ct is not Zero, (prim, v.aval)  # check for an old harmless type error
    if ct is None or type(v) is Literal:
      return
    if type(ct) is Zero:
      # FIXME: This triggers a lot of failures!
      # assert v.aval == ct.aval, (prim, v.aval, ct.aval)
      return
    axes_to_reduce = tuple(axis_name for axis_name in reduce_axes
                           if axis_name in core.get_aval(ct).named_shape
                           and axis_name not in v.aval.named_shape)
    if axes_to_reduce:
      ct = jax.lax.psum(ct, axis_name=axes_to_reduce)
    ct_env[v] = add_tangents(ct_env[v], ct) if v in ct_env else ct
    if config.jax_enable_checks:
      ct_aval = core.get_aval(ct_env[v])
      joined_aval = core.lattice_join(v.aval, ct_aval).strip_weak_type().strip_named_shape()
      assert v.aval.strip_weak_type().strip_named_shape() == joined_aval, (prim, v.aval, ct_aval)

  def read_cotangent(v):
    return ct_env.pop(v, Zero(v.aval))

  def read_primal(v):
    if type(v) is Literal:
      return v.val
    else:
      return primal_env.get(v, UndefinedPrimal(v.aval))

  def write_primal(v, val):
    if not is_undefined_primal(val):
      primal_env[v] = val

  primal_env: Dict[Any, Any] = {}
  write_primal(core.unitvar, core.unit)
  map(write_primal, jaxpr.constvars, consts)
  # FIXME: invars can contain both primal and tangent values, and this line
  #        forces primal_in to contain UndefinedPrimals for tangent values!
  map(write_primal, jaxpr.invars, primals_in)

  ct_env: Dict[Any, Any] = {}
  map(partial(write_cotangent, 'outvars'), jaxpr.outvars, cotangents_in)
  for eqn in jaxpr.eqns[::-1]:
    # FIXME: Some invars correspond to tangents
    invals = map(read_primal, eqn.invars)
    if eqn.primitive.multiple_results:
      cts_in = map(read_cotangent, eqn.outvars)
    else:
      cts_in, = map(read_cotangent, eqn.outvars)
    with source_info_util.user_context(eqn.source_info):
      if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
        cts_in_avals = [v.aval for v in eqn.outvars]
        call_jaxpr, params = core.extract_call_jaxpr(eqn.primitive, eqn.params)
        cts_out = get_primitive_transpose(eqn.primitive)(
            params, call_jaxpr, invals, cts_in, cts_in_avals, reduce_axes)
      elif eqn.primitive in reducing_transposes:
        cts_out = reducing_transposes[eqn.primitive](
            reduce_axes, cts_in, *invals, **eqn.params)
      else:
        cts_out = get_primitive_transpose(eqn.primitive)(cts_in, *invals,
                                                         **eqn.params)
    cts_out = [Zero(v.aval) for v in eqn.invars] if cts_out is Zero else cts_out
    # FIXME: Some invars correspond to primals!
    map(partial(write_cotangent, eqn.primitive), eqn.invars, cts_out)

  cotangents_out = map(read_cotangent, jaxpr.invars)
  return cotangents_out

class UndefinedPrimal:
  __slots__ = ['aval']
  def __init__(self, aval):
    self.aval = aval
  def __repr__(self):
    return 'UndefinedPrimal({})'.format(self.aval)

def is_undefined_primal(x):
  return type(x) is UndefinedPrimal

register_pytree_node(UndefinedPrimal,
                     lambda z: ((), z.aval),
                     lambda aval, _: UndefinedPrimal(aval))

def get_primitive_transpose(p):
  try:
    return primitive_transposes[p]
  except KeyError as err:
    raise NotImplementedError(
        "Transpose rule (for reverse-mode differentiation) for '{}' "
        "not implemented".format(p)) from err

@lu.transformation_with_aux
def nonzero_tangent_outputs(*args, **kwargs):
  results = (_, tangents_out) = yield args, kwargs
  yield results, [type(r) is not Zero for r in tangents_out]


class JVPTrace(Trace):

  def pure(self, val):
    tangent_zero = Zero(get_aval(val).at_least_vspace())
    return JVPTracer(self, val, tangent_zero)

  def lift(self, val):
    tangent_zero = Zero(get_aval(val).at_least_vspace())
    return JVPTracer(self, val, tangent_zero)

  def sublift(self, val):
    return JVPTracer(self, val.primal, val.tangent)

  def process_primitive(self, primitive, tracers, params):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    jvp = primitive_jvps.get(primitive)
    if not jvp:
      msg = f"Differentiation rule for '{primitive}' not implemented"
      raise NotImplementedError(msg)
    primal_out, tangent_out = jvp(primals_in, tangents_in, **params)
    if primitive.multiple_results:
      return [JVPTracer(self, x, t) for x, t in zip(primal_out, tangent_out)]
    else:
      return JVPTracer(self, primal_out, tangent_out)

  def process_call(self, primitive, f, tracers, params):
    primals, tangents = unzip2((t.primal, t.tangent) for t in tracers)
    nonzero_tangents, tangent_tree_def = tree_flatten(tangents)
    nz_tangents = [type(t) is not Zero for t in tangents]
    if 'name' in params:
      params = dict(params, name=wrap_name(params['name'], 'jvp'))
    f_jvp = jvp_subtrace(f, self.main)
    f_jvp, nz_tangents_out = nonzero_tangent_outputs(f_jvp)
    f_jvp, out_tree_def = traceable(f_jvp, len(primals), tangent_tree_def)
    update_params = call_param_updaters.get(primitive)
    new_params = (update_params(params, nz_tangents, nz_tangents_out)
                  if update_params else params)
    def todo(result):
      primal_out, tangent_out = tree_unflatten(out_tree_def(), result)
      return [JVPTracer(self, p, t) for p, t in zip(primal_out, tangent_out)]
    return f_jvp, (*primals, *nonzero_tangents), new_params, todo

  def process_map(self, primitive, f, tracers, params):
    primals, tangents = unzip2((t.primal, t.tangent) for t in tracers)
    nonzero_tangents, tangent_tree_def = tree_flatten(tangents)
    nz_tangents = [type(t) is not Zero for t in tangents]
    params = dict(params, name=wrap_name(params['name'], 'jvp'))
    f_jvp = jvp_subtrace(f, self.main)
    f_jvp, nz_tangents_out = nonzero_tangent_outputs(f_jvp)
    in_axes = params['in_axes']
    tangent_in_axes = [ax for ax, nz in zip(in_axes, nz_tangents) if nz]
    out_axes_thunk = params['out_axes_thunk']

    @as_hashable_function(closure=(tuple(nz_tangents), out_axes_thunk))
    def new_out_axes_thunk():
      out_axes = out_axes_thunk()
      return (*out_axes, *(ax for ax, nz in zip(out_axes, nz_tangents_out()) if nz))

    params = dict(params, in_axes=(*in_axes, *tangent_in_axes),
                  out_axes_thunk=new_out_axes_thunk)
    f_jvp, out_tree_def = traceable(f_jvp, len(primals), tangent_tree_def)
    update_params = call_param_updaters.get(primitive)
    new_params = (update_params(params, nz_tangents, nz_tangents_out)
                  if update_params else params)
    def todo(result):
      primal_out, tangent_out = tree_unflatten(out_tree_def(), result)
      return [JVPTracer(self, p, t) for p, t in zip(primal_out, tangent_out)]
    return f_jvp, (*primals, *nonzero_tangents), new_params, todo

  def process_custom_jvp_call(self, _, __, f_jvp, tracers):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    primals_in = map(core.full_lower, primals_in)
    tangents_in = map(instantiate_zeros, tangents_in)
    # Cast float0 to zeros with the primal dtype because custom jvp rules don't
    # currently handle float0s
    tangents_in = map(replace_float0s, primals_in, tangents_in)
    outs = f_jvp.call_wrapped(*primals_in, *tangents_in)
    primals_out, tangents_out = split_list(outs, [len(outs) // 2])
    tangents_out = map(recast_to_float0, primals_out, tangents_out)
    return map(partial(JVPTracer, self), primals_out, tangents_out)

  def post_process_custom_jvp_call(self, out_tracers):
    raise CustomJVPException()

  def process_custom_vjp_call(self, _, __, fwd, bwd, tracers, *, out_trees):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    tangents_in = map(instantiate_zeros, tangents_in)
    res_and_primals_out = fwd.call_wrapped(*map(core.full_lower, primals_in))
    out_tree, res_tree = out_trees()
    res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
    avals_out = [raise_to_shaped(core.get_aval(x)) for x in primals_out]
    tangents_out = custom_lin_p.bind(
        *res, *tangents_in, num_res=res_tree.num_leaves, bwd=bwd,
        avals_out=avals_out)
    tangents_out = map(recast_to_float0, primals_out, tangents_out)
    return map(partial(JVPTracer, self), primals_out, tangents_out)

  def post_process_custom_vjp_call(self, out_tracers):
    raise CustomVJPException()

  def join(self, xt, yt):
    xz, yz = type(xt) is Zero, type(yt) is Zero
    if xz == yz:
      return xt, yt
    elif yz and not xz:
      return xt, zeros_like_jaxval(xt)
    elif xz and not yz:
      return zeros_like_jaxval(yt), yt
    else:
      raise TypeError((xt, yt))


class JVPTracer(Tracer):
  __slots__ = ['primal', 'tangent']

  def __init__(self, trace, primal, tangent):
    if config.jax_enable_checks:
      _primal_tangent_shapes_match(primal, tangent)
    self._trace = trace
    self.primal = primal
    self.tangent = tangent

  @property
  def aval(self):
    # TODO(dougalm): add epsilon ball
    return get_aval(self.primal)

  def full_lower(self):
    if type(self.tangent) is Zero:
      return core.full_lower(self.primal)
    else:
      return self

def _primal_tangent_shapes_match(primal, tangent):
  if type(tangent) is not Zero:
    primal_aval = raise_to_shaped(get_aval(primal), weak_type=False)
    tangent_aval = raise_to_shaped(get_aval(tangent), weak_type=False)
    assert primal_aval.shape == tangent_aval.shape, (primal_aval.shape, tangent_aval.shape)
    expected_tangent_dtype = core.primal_dtype_to_tangent_dtype(primal_aval.dtype)
    assert expected_tangent_dtype == tangent_aval.dtype, (expected_tangent_dtype, tangent_aval.dtype)

call_param_updaters: Dict[core.Primitive, Callable] = {}
call_transpose_param_updaters: Dict[core.Primitive, Callable] = {}


# -------------------- Primitives --------------------

primitive_jvps : Dict[core.Primitive, Callable] = {}

primitive_transposes: Dict[core.Primitive, Callable] = {}
# transpose rules that internally perform reductions over the given named axes
reducing_transposes: Dict[core.Primitive, Callable] = {}


def deflinear(primitive, transpose_rule):
  primitive_jvps[primitive] = partial(linear_jvp, primitive)
  primitive_transposes[primitive] = partial(linear_transpose, transpose_rule)

def linear_jvp(primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  if all(type(tangent) is Zero for tangent in tangents):
    return val_out, Zero.from_value(val_out)
  else:
    tangents = map(instantiate_zeros, tangents)
    return val_out, primitive.bind(*tangents, **params)

def linear_transpose(transpose_rule, cotangent, *args, **kwargs):
  return Zero if type(cotangent) is Zero else transpose_rule(cotangent, **kwargs)


def deflinear2(primitive, transpose_rule):
  primitive_jvps[primitive] = partial(linear_jvp, primitive)
  primitive_transposes[primitive] = partial(linear_transpose2, transpose_rule)

def linear_transpose2(transpose_rule, cotangent, *args, **kwargs):
  return Zero if type(cotangent) is Zero else transpose_rule(cotangent, *args, **kwargs)


def defjvp(primitive, *jvprules):
  assert isinstance(primitive, Primitive)
  assert not primitive.multiple_results
  primitive_jvps[primitive] = partial(standard_jvp, jvprules, primitive)


def standard_jvp(jvprules, primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  tangents_out = [rule(t, *primals, **params) for rule, t in zip(jvprules, tangents)
                  if rule is not None and type(t) is not Zero]
  return val_out, functools.reduce(add_tangents, tangents_out, Zero.from_value(val_out))

def defjvp2(primitive, *jvprules):
  assert isinstance(primitive, Primitive)
  assert not primitive.multiple_results
  primitive_jvps[primitive] = partial(standard_jvp2, jvprules, primitive)

def standard_jvp2(jvprules, primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  tangents_out = (rule(t, val_out, *primals, **params) for rule, t in zip(jvprules, tangents)
                  if rule is not None and type(t) is not Zero)
  tangents_out = list(tangents_out)
  return val_out, functools.reduce(add_tangents, tangents_out, Zero.from_value(val_out))

def add_tangents(x, y):
  if type(x) is Zero:
    return y
  elif type(y) is Zero:
    return x
  else:
    return add_jaxvals(x, y)


def defbilinear(prim, lhs_rule, rhs_rule):
  assert isinstance(prim, Primitive)
  lhs_jvp = lambda g, x, y, **kwargs: prim.bind(g, y, **kwargs)
  rhs_jvp = lambda g, x, y, **kwargs: prim.bind(x, g, **kwargs)
  defjvp(prim, lhs_jvp, rhs_jvp)
  primitive_transposes[prim] = partial(bilinear_transpose, lhs_rule, rhs_rule)

def bilinear_transpose(lhs_rule, rhs_rule, cotangent, x, y, **kwargs):
  assert is_undefined_primal(x) ^ is_undefined_primal(y)
  if type(cotangent) is Zero:
    return Zero
  if is_undefined_primal(x):
    out = lhs_rule(cotangent, y, **kwargs)
    return Zero if out is Zero else (out, None)
  else:
    out = rhs_rule(cotangent, x, **kwargs)
    return Zero if out is Zero else (None, out)


def defjvp_zero(primitive):
  assert isinstance(primitive, Primitive)
  primitive_jvps[primitive] = partial(zero_jvp, primitive)

def zero_jvp(primitive, primals, tangents, **params):
  r = primitive.bind(*primals, **params)
  return r, Zero.from_value(r)


deflinear2(zeros_like_p, lambda t, _: [Zero.from_value(t)])
deflinear2(add_jaxvals_p, lambda t, *args: (t, t))

def instantiate_zeros(tangent):
  if type(tangent) is Zero:
    if isinstance(tangent.aval, Tracer):
      return tangent.aval
    return zeros_like_aval(tangent.aval)
  else:
    return tangent

# This function seems similar to instantiate_zeros, but it is sometimes used
# to instantiate zero abstract units with a different aval
def instantiate_zeros_aval(aval, tangent):
  if type(tangent) is Zero:
    assert type(tangent.aval) is core.AbstractUnit or tangent.aval == aval
    return zeros_like_aval(aval)
  else:
    return tangent

@lu.transformation_with_aux
def traceable(num_primals, in_tree_def, *primals_and_tangents):
  new_primals  = primals_and_tangents[:num_primals]
  new_tangents = primals_and_tangents[num_primals:]
  new_tangents = tree_unflatten(in_tree_def, new_tangents)
  primal_out, tangent_out = yield (new_primals, new_tangents), {}
  out_flat, tree_def = tree_flatten((primal_out, tangent_out))
  yield out_flat, tree_def


def call_transpose(primitive, params, call_jaxpr, args, ct, _, reduce_axes):
  all_args, in_tree_def = tree_flatten(((), args, ct))  # empty consts
  fun = lu.hashable_partial(lu.wrap_init(backward_pass), call_jaxpr, reduce_axes)
  fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
  new_params = dict(params, name=wrap_name(params['name'], 'transpose'))
  update_params = call_transpose_param_updaters.get(primitive)
  if update_params:
    new_params = update_params(new_params, map(is_undefined_primal, args),
                               [type(x) is not Zero for x in ct])
  out_flat = primitive.bind(fun, *all_args, **new_params)
  return tree_unflatten(out_tree(), out_flat)
primitive_transposes[core.call_p] = partial(call_transpose, call_p)


def remat_transpose(params, call_jaxpr, primals_in, cotangents_in,
                    cotangent_in_avals, reduce_axes):
  # backward_pass can only transpose linear computations, but the call_jaxpr embedded in
  # remat contains primal (non-linear) equations too. Hence, we have to eliminate those
  # (in this case via partial_eval) before we call into backward_pass again.
  typed_call_jaxpr = core.ClosedJaxpr(call_jaxpr, [])
  unknowns = map(is_undefined_primal, primals_in)
  primal_jaxpr, tangent_jaxpr, out_unknowns = \
    pe.partial_eval_jaxpr(typed_call_jaxpr, unknowns=unknowns, instantiate=True)  # type: ignore

  def do_transpose(primals_in, cotangents_in):
    # NOTE: This is passing in undefined primals in place of tangent arguments, but it
    #       should all work out, because we're only computing the primal part here.
    residuals = core.jaxpr_as_fun(primal_jaxpr)(*primals_in)[len(cotangents_in):]
    # Now that we have a purely linear jaxpr, we can transpose it
    cotangents_out = backward_pass(
        tangent_jaxpr.jaxpr, reduce_axes, (), primals_in + residuals, cotangents_in)
    # backward_pass will return cotangents computed for all invars, but some of them
    # are residuals appended by partial eval, so we need to skip those before we return.
    return cotangents_out[:len(primals_in)]

  flat_args, in_tree_def = tree_flatten((primals_in, cotangents_in))
  flat_do_transpose, out_tree = flatten_fun_nokwargs(lu.wrap_init(do_transpose), in_tree_def)
  flat_cotangents_out = pe.remat_call_p.bind(flat_do_transpose, *flat_args, **params)
  return tree_unflatten(out_tree(), flat_cotangents_out)
primitive_transposes[pe.remat_call_p] = remat_transpose

@lu.transformation_with_aux
def nonzero_outputs(*args, **kwargs):
  results = yield args, kwargs
  yield results, [type(r) is not Zero for r in results]


def map_transpose(primitive, params, call_jaxpr, args, ct, _, reduce_axes):
  all_args, in_tree_def = tree_flatten(((), args, ct))  # empty consts
  fun = lu.hashable_partial(lu.wrap_init(backward_pass), call_jaxpr, reduce_axes)
  fun, nz_arg_cts = nonzero_outputs(fun)
  fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
  # Preserve axis for primal arguments, skip tangents (represented as undefined primals).
  in_axes, out_axes = params['in_axes'], params['out_axes']
  new_in_axes = (*[axis for axis, x in zip(in_axes, args)
                   if not is_undefined_primal(x)],
                 *[axis for axis, x in zip(out_axes, ct)
                   if type(x) is not Zero])
  # The interim strategy we use below (until avals-with-names) only works
  # when all outputs are mapped.
  assert all(out_axis is not None for out_axis in out_axes), out_axes
  # NOTE: This assumes that the output cotangents being zero is a deterministic
  #       function of which input cotangents were zero.
  @as_hashable_function(closure=(in_axes, tuple(type(c) is Zero for c in ct)))
  def out_axes_thunk():
    return tuple(axis or 0 for axis, nz in zip(in_axes, nz_arg_cts()) if nz)
  new_params = dict(params, name=wrap_name(params['name'], 'transpose'),
                    in_axes=new_in_axes, out_axes_thunk=out_axes_thunk)
  del new_params['out_axes']
  update_params = call_transpose_param_updaters.get(primitive)
  if update_params:
    new_params = update_params(new_params, map(is_undefined_primal, args),
                               [type(x) is not Zero for x in ct])
  out_flat = primitive.bind(fun, *all_args, **new_params)
  arg_cts = tree_unflatten(out_tree(), out_flat)

  # The freevars are being fanned out (not mapped). During transpose the
  # dual of fan-out is fan-in-sum. We apply it to the unmapped invars.
  assert len(in_axes) == len(arg_cts)
  def unmap_zero(zero, in_axis):
    return (zero if in_axis is None else
            Zero(core.unmapped_aval(params['axis_size'], in_axis, zero.aval)))
  arg_cts = (unmap_zero(arg_ct, in_axis) if type(arg_ct) is Zero else
             arg_ct if in_axis is not None else
             arg_ct.sum(0)
             for arg_ct, in_axis in zip(arg_cts, in_axes))
  return tuple(arg_cts)


def jvp_jaxpr(jaxpr, nonzeros, instantiate):
  assert len(jaxpr.in_avals) == len(nonzeros)
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  f_jvp, out_nonzeros = f_jvp_traceable(jvp(f, instantiate=instantiate), nonzeros)
  tangent_avals = [aval for aval, nz in zip(jaxpr.in_avals, nonzeros) if nz]
  avals_in = list(it.chain(jaxpr.in_avals, tangent_avals))
  jaxpr_out, avals_out, literals_out = pe.trace_to_jaxpr_dynamic(f_jvp, avals_in)
  return core.ClosedJaxpr(jaxpr_out, literals_out), out_nonzeros()

@lu.transformation_with_aux
def f_jvp_traceable(nonzeros, *primals_and_nztangents):
  num_primals = len(nonzeros)
  primals = list(primals_and_nztangents[:num_primals])
  nonzero_tangents = iter(primals_and_nztangents[num_primals:])
  tangents = [next(nonzero_tangents) if nz else Zero.from_value(p)
              for p, nz in zip(primals, nonzeros)]
  primals_out, tangents_out = yield (primals, tangents), {}
  out_nonzeros = [type(t) is not Zero for t in tangents_out]
  nonzero_tangents_out = [t for t in tangents_out if type(t) is not Zero]
  yield list(primals_out) + nonzero_tangents_out, out_nonzeros

def rearrange_binders(jaxpr: core.ClosedJaxpr, primals_in, tangents_in, primals_out, tangents_out):
  new_invars = _perm(primals_in, tangents_in, jaxpr.jaxpr.invars)
  new_outvars = _perm(primals_out, tangents_out, jaxpr.jaxpr.outvars)
  new_jaxpr = core.Jaxpr(jaxpr.jaxpr.constvars,
                         new_invars, new_outvars, jaxpr.jaxpr.eqns)
  return core.ClosedJaxpr(new_jaxpr, jaxpr.consts)

def _perm(primal_counts, tangent_counts, lst):
  n = sum(primal_counts)
  primals, tangents = lst[:n], lst[n:]
  primal_groups = split_list(primals, primal_counts[:-1])
  tangent_groups = split_list(tangents, tangent_counts[:-1])
  return _interleave(primal_groups, tangent_groups)

def _interleave(xs, ys):
  assert len(xs) == len(ys)
  return [e for pair in zip(xs, ys) for l in pair for e in l]


custom_lin_p = core.Primitive('custom_lin')
custom_lin_p.def_abstract_eval(lambda *_, avals_out, **__: avals_out)
custom_lin_p.multiple_results = True

def _raise_custom_vjp_error_on_jvp(*_, **__):
  raise TypeError("can't apply forward-mode autodiff (jvp) to a custom_vjp "
                  "function.")
custom_lin_p.def_impl(_raise_custom_vjp_error_on_jvp)

def _custom_lin_transpose(cts_out, *invals, num_res, bwd, avals_out):
  res, _ = split_list(invals, [num_res])
  cts_out = map(instantiate_zeros_aval, avals_out, cts_out)
  cts_in = bwd.call_wrapped(*res, *cts_out)
  return [None] * num_res + list(cts_in)
primitive_transposes[custom_lin_p] = _custom_lin_transpose


class CustomJVPException(Exception):
  def __init__(self):
    # TODO(mattjj): track source provenance on AD tracers, improve error
    msg = ("Detected differentiation of a custom_jvp function with respect to "
           "a closed-over value. That isn't supported because the custom JVP "
           "rule only specifies how to differentiate the custom_jvp function "
           "with respect to explicit input parameters. Try passing the "
           "closed-over value into the custom_jvp function as an argument, and "
           "adapting the custom_jvp rule.")
    super().__init__(msg)

class CustomVJPException(Exception):
  def __init__(self):
    # TODO(mattjj): track source provenance on AD tracers, improve error
    msg = ("Detected differentiation of a custom_vjp function with respect to "
           "a closed-over value. That isn't supported because the custom VJP "
           "rule only specifies how to differentiate the custom_vjp function "
           "with respect to explicit input parameters. Try passing the "
           "closed-over value into the custom_vjp function as an argument, and "
           "adapting the custom_vjp fwd and bwd rules.")
    super().__init__(msg)
