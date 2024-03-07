# Copyright 2024 The JAX Authors.
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

from contextlib import contextmanager
from typing import Any

from jax._src import core
from jax._src import api_util
from jax._src import linear_util as lu
from jax._src.api_util import flatten_fun_nokwargs
from jax._src.interpreters import ad
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (tree_flatten, tree_unflatten, tree_structure,
                                treedef_tuple)
from jax._src.util import unzip2, safe_map, safe_zip, split_list

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

JaxVal = Any
Pytree = Any

register = api_util.register_class_with_attrs

@contextmanager
def top_trace():
  stack = core.thread_local_state.trace_state.trace_stack.stack
  main = stack.pop()
  try:
    trace = main.with_cur_sublevel()
    yield trace
  finally:
    stack.append(main)

def jax_getattr(obj: Any, attr: str):
  with top_trace() as trace:
    return trace.process_getattr(obj, attr)

def jax_setattr(obj: Any, attr: str, val: Pytree):
  with top_trace() as trace:
    return trace.process_setattr(obj, attr, val)

def _getattr_impl(_, obj, attr):
  return getattr(obj, attr)
core.EvalTrace.process_getattr = _getattr_impl

def _setattr_impl(_, obj, attr, val):
  setattr(obj, attr, val)
core.EvalTrace.process_setattr = _setattr_impl

def _ensure_tracked(trace: pe.DynamicJaxprTrace, obj: Any, attr: str):
  frame = trace.main.jaxpr_stack[-1]  # type: ignore

  def new_tracer(x):
    aval = core.raise_to_shaped(core.get_aval(x))
    tracer = pe.DynamicJaxprTracer(trace, aval, pe.source_info_util.current())
    var = frame.tracer_to_var[id(tracer)] = frame.newvar(aval)
    frame.attrs_vars.append(var)
    frame.tracers.append(tracer)
    return tracer

  if (obj, attr) not in frame.attrs_tracked:
    init_val = getattr(obj, attr)
    frame.attrs_inits.append(init_val)
    init_vals, init_tree = tree_flatten(init_val)
    tracers = map(new_tracer, init_vals)
    setattr(obj, attr, tree_unflatten(init_tree, tracers))
    frame.attrs_tracked.append((obj, attr))
pe.DynamicJaxprTrace._ensure_tracked = _ensure_tracked

def _getattr_staging(trace, obj, attr):
  trace._ensure_tracked(obj, attr)
  return getattr(obj, attr)
pe.DynamicJaxprTrace.process_getattr = _getattr_staging

def _setattr_staging(trace, obj, attr, val):
  trace._ensure_tracked(obj, attr)
  setattr(obj, attr, val)
pe.DynamicJaxprTrace.process_setattr = _setattr_staging


def jvp(f, primals, tangents, attr_tangents):
  attrs, attr_tangents = unzip2(((o, a), t) for o, a, t in attr_tangents)
  attr_primals = tuple(jax_getattr(o, a) for o, a in attrs)
  primals_flat, in_tree = tree_flatten((attr_primals, *primals))
  tangents_flat, in_tree_ = tree_flatten((attr_tangents, *tangents))
  if in_tree != in_tree_: raise Exception
  f_, out_tree = flatten_fun_nokwargs(_set_attrs(lu.wrap_init(f), attrs), in_tree)
  out_primals_flat, out_tangents_flat, tangent_attrs_out = _jvp(f_).call_wrapped(
      primals_flat, tangents_flat)
  out_primals = tree_unflatten(out_tree(), out_primals_flat)
  out_tangents = tree_unflatten(out_tree(), out_tangents_flat)
  return out_primals, out_tangents, tangent_attrs_out

@lu.transformation
def _set_attrs(attrs, attr_vals, *args):
  for (o, a), x in zip(attrs, attr_vals):
    jax_setattr(o, a, x)
  yield (yield args, {})

def _jvp(fun: lu.WrappedFun):
  return jvpfun2(jvp_subtrace2(fun))

@lu.transformation
def jvpfun2(primals, tangents):
  with core.new_main(ad.JVPTrace) as main:
    out_primals, out_tangents, tangent_attrs_out = \
        yield (main, primals, tangents), {}
    del main
  yield out_primals, out_tangents, tangent_attrs_out

@lu.transformation
def jvp_subtrace2(main, primals, tangents):
  main.attrs_tracked = []  # attrs written to
  trace = main.with_cur_sublevel()
  in_tracers = [ad.JVPTracer(trace, x, t) if type(t) is not ad.Zero else x
                for x, t in zip(primals, tangents)]
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  out_primals, out_tangents = unzip2((t.primal, t.tangent) for t in out_tracers)
  tangent_attrs_out = []
  for (obj, name) in main.attrs_tracked:
    tracer = trace.full_raise(jax_getattr(obj, name))
    jax_setattr(obj, name, tracer.primal)
    if type(tracer.tangent) is not ad.Zero:
      tangent_attrs_out.append((obj, name, tracer.tangent))
  del main.attrs_tracked
  yield out_primals, out_tangents, tangent_attrs_out

def _setattr_jvp(trace, obj, attr, maybe_tracer):
  tracer = trace.full_raise(maybe_tracer)
  if isinstance(tracer.tangent, ad.Zero):
    return setattr(obj, attr, tracer.primal)
  if (obj, attr) not in trace.main.attrs_tracked:
    trace.main.attrs_tracked.append((obj, attr))
  return setattr(obj, attr, tracer)
ad.JVPTrace.process_setattr = _setattr_jvp

def _getattr_jvp(trace, obj, attr):
  return getattr(obj, attr)
ad.JVPTrace.process_getattr = _getattr_jvp


def linearize(f, *primals, attrs: list[tuple[Any, str]] = []):
  attr_primals = [jax_getattr(o, a) for o, a in attrs]
  attr_avals = [core.raise_to_shaped(core.get_aval(p)) for p in attr_primals]
  primals_flat, in_tree = tree_flatten(primals)
  tree = treedef_tuple((tree_structure(attr_primals), *in_tree.children()))
  f_, out_tree = flatten_fun_nokwargs(_set_attrs(lu.wrap_init(f), attrs), tree)
  primal_out, out_pvals, jaxpr, consts, attrs_out = _linearize(
      f_, *attr_primals, *primals_flat)
  f_lin = _lin_wrap(jaxpr, consts, out_pvals, attr_avals, (in_tree, out_tree()),
                    attrs, attrs_out)
  return tree_unflatten(out_tree(), primal_out), f_lin

def _linearize(traceable: lu.WrappedFun, *primals):
  jvpfun, attrs = _split_attrs(_jvp(traceable))
  in_pvals = (tuple(pe.PartialVal.known(p) for p in primals)
              + tuple(pe.PartialVal.unknown(core.get_aval(p).at_least_vspace())
                      for p in primals))
  _, in_tree = tree_flatten((primals, primals))
  jvpfun_flat, out_tree = flatten_fun_nokwargs(jvpfun, in_tree)
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr_nounits(jvpfun_flat, in_pvals)
  out_primals_pvals, out_tangents_pvals, out_tangent_attr_pvals = \
      tree_unflatten(out_tree(), out_pvals)
  out_primals_consts = [pval.get_known() for pval in out_primals_pvals]
  return (out_primals_consts, [*out_tangents_pvals, *out_tangent_attr_pvals],
          jaxpr, consts, attrs())

@lu.transformation_with_aux
def _split_attrs(*args, **kwargs):
  primals, tangents, tangent_attrs = yield args, kwargs
  attrs, tangent_attr_vals = unzip2(((o, a), t) for o, a, t in tangent_attrs)
  yield (primals, tangents, tangent_attr_vals), attrs

def _lin_wrap(jaxpr, consts, out_pvals, attr_avals, io_tree, in_attrs, out_attrs):
  in_tree, out_tree = io_tree
  def f_lin(*tangents, attr_tangents):
    if set(attr_tangents) - set(in_attrs): raise Exception
    tangents_, in_tree_ = tree_flatten(tangents)
    assert in_tree == in_tree_
    attr_tangents_ = [attr_tangents.get(a, ad.Zero(aval))
                      for a, aval in zip(in_attrs, attr_avals)]
    out = core.eval_jaxpr(jaxpr, consts, *attr_tangents_, *tangents_)
    out_ = iter(out)
    out = [p.get_known() if p.is_known() else next(out_) for p in out_pvals]
    assert next(out_, None) is None
    tangents_out, attr_tangents_out = split_list(out, [len(out)-len(out_attrs)])
    out_ct = tree_unflatten(out_tree, tangents_out)
    return out_ct, dict(zip(out_attrs, attr_tangents_out))
  return f_lin


def vjp(f, *primals, attrs: list[tuple[Any, str]] = []):
  attr_primals = [jax_getattr(o, a) for o, a in attrs]
  primals_flat, in_tree = tree_flatten(primals)
  tree = treedef_tuple((tree_structure(attr_primals), *in_tree.children()))
  f_, out_tree = flatten_fun_nokwargs(_set_attrs(lu.wrap_init(f), attrs), tree)
  primal_out, out_pvals, jaxpr, consts, attrs_out = _linearize(
      f_, *attr_primals, *primals_flat)
  attr_avals = [core.raise_to_shaped(core.get_aval(jax_getattr(o, a))).at_least_vspace()
                for o, a in attrs_out]
  f_vjp = _vjp_wrap(jaxpr, consts, out_pvals, attr_avals, (in_tree, out_tree()),
                    attrs, attrs_out)
  return tree_unflatten(out_tree(), primal_out), f_vjp

def _vjp_wrap(jaxpr, consts, out_pvals, attr_avals, io_tree, in_attrs, out_attrs):
  in_tree, out_tree = io_tree
  dummies = [ad.UndefinedPrimal(v.aval) for v in jaxpr.invars]
  def f_vjp(out_ct, *, attr_cotangents: dict[tuple[Any, str], JaxVal] = {}):
    out_cts, out_tree_ = tree_flatten(out_ct)
    assert out_tree == out_tree_
    attr_cts = [attr_cotangents.get(a, ad.Zero(aval))
                for a, aval in zip(out_attrs, attr_avals)]
    out = ad.backward_pass(jaxpr, (), consts, dummies, (*out_cts, *attr_cts))
    in_attr_bars, arg_cts = split_list(out, [len(in_attrs)])
    args_ct = tree_unflatten(in_tree, map(ad.instantiate_zeros, arg_cts))
    return args_ct, dict(zip(in_attrs, in_attr_bars))
  return f_vjp
