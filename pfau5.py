import pdb, sys, traceback
def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    pdb.pm()
sys.excepthook = info

from functools import partial
import operator as op

import jax
import jax.numpy as jnp
jax.config.update('jax_platforms', 'cpu')

from jax import core
from jax import lax
from jax._src import sharding_impls
from jax._src.interpreters import partial_eval as pe

from jax._src import pjit
from jax._src import linear_util as lu
from jax._src.api_util import flatten_fun_nokwargs, shaped_abstractify
from jax._src.util import (safe_map, safe_zip, unzip3, weakref_lru_cache,
                           partition_list, merge_lists)
from jax._src.tree_util import (tree_map, tree_flatten, tree_unflatten,
                                tree_leaves)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


class LapTracer(core.Tracer):
  __slots__ = ['primal', 'jacobian', 'lapvec']

  def __init__(self, trace, primal, jacobian, lapvec):
    self._trace = trace
    self.primal = primal
    self.jacobian = jacobian
    self.lapvec = lapvec

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def full_lower(self):
    if self.jacobian is self.lapvec is None:
      return core.full_lower(self.primal)
    else:
      return self

class LapTrace(core.Trace):
  pure = lift = lambda self, val: LapTracer(self, val, None, None)
  sublift = lambda self: LapTracer(self, val.primal, val.jacobian, val.lapvec)

  def process_primitive(self, prim, tracers, params):
    xs, jacs, laps = unzip3((t.primal, t.jacobian, t.lapvec) for t in tracers)
    rule = rules.get(prim, partial(generic_rule, prim))
    ys, out_jacs, out_laps = rule(xs, jacs, laps, **params)
    if prim.multiple_results:
      return map(partial(LapTracer, self), ys, out_jacs, out_laps)
    else:
      return LapTracer(self, ys, out_jacs, out_laps)

rules = {}

## generic rule

def generic_rule(prim, in_primals, in_jacs, in_lapvecs, **params):
  merge, primals, nz_jacs, nz_laps = partition(in_primals, in_jacs, in_lapvecs)
  def f(*primals): return prim.bind(*merge(primals), **params)
  out_primals, f_jvp = jax.linearize(f, *primals)
  out_jacs, hess_term = hqf2(f, primals, nz_jacs)
  out_lapvecs = tree_map(op.add, trace(hess_term), f_jvp(*nz_laps))
  return out_primals, out_jacs, out_lapvecs

def partition(primals, jacs, laps):
  nones, nones_ = [j is None for j in jacs], [l is None for l in laps]
  assert nones == nones_
  new_primals, const_primals = partition_list(nones, primals)
  merge = lambda new_primals: merge_lists(nones, new_primals, const_primals)
  return merge, (*new_primals,), (*tree_leaves(jacs),), (*tree_leaves(laps),)

trace = partial(tree_map, partial(jnp.trace, axis1=-1, axis2=-2))

def hbf(f, xs, vs1, vs2):
  return jax.jvp(lambda *xs: jax.jvp(f, xs, vs1)[1], xs, vs2)

def hqf2(f, xs, vs):
  return jax.vmap(jax.vmap(partial(hbf, f, xs), (-1, None), -1),
                  (None, -1), (None, -1))(vs, vs)

## internal transformation machinery

def fwdlap3(fun, primals, jacobians, lapvecs):
  primals_, in_tree = tree_flatten(primals)
  jacs_, in_tree2 = tree_flatten(jacobians)
  laps_, in_tree3 = tree_flatten(lapvecs)
  if not in_tree == in_tree2 == in_tree3: raise Exception
  fun_, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  fun_ = _fwdlap(_fwdlap_subtrace(fun_))
  outs = fun_.call_wrapped(primals_, jacs_, laps_)
  return map(partial(tree_unflatten, out_tree()), outs)

@lu.transformation
def _fwdlap(primals, jacobians, lapvecs):
  with core.new_main(LapTrace) as main:
    out_primals, out_jacs, out_laps = yield (main, primals, jacobians, lapvecs), {}
    del main
  out_jacs_ = [J if J is not None else jnp.zeros(*p.shape, *p.shape)
               for p, J in zip(out_primals, out_jacs)]
  out_laps_ = [l if l is not None else jnp.zeros(p.shape)
               for p, l in zip(out_primals, out_laps)]
  yield out_primals, out_jacs, out_laps

@lu.transformation
def _fwdlap_subtrace(main, primals, jacobians, lapvecs):
  trace = LapTrace(main, core.cur_sublevel())
  in_tracers = map(partial(LapTracer, trace), primals, jacobians, lapvecs)
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  out_primals, out_jacs, out_laps = unzip3((t.primal, t.jacobian, t.lapvec)
                                           for t in out_tracers)
  yield out_primals, out_jacs, out_laps

@weakref_lru_cache
def _fwdlap_jaxpr(jaxpr: core.ClosedJaxpr, in_avals, in_tree):
  f = lu.wrap_init(partial(core.eval_jaxpr, jaxpr.jaxpr, jaxpr.consts))
  f = _fwdlap(_fwdlap_subtrace(f))
  f, out_tree = flatten_fun_nokwargs(f, in_tree)
  jaxpr_, _, consts = pe.trace_to_jaxpr_dynamic(f, in_avals)
  return core.ClosedJaxpr(jaxpr_, consts), out_tree()

## rules for higher-order primitives

def pjit_rule(xs, jacs, laps, *, jaxpr, in_shardings, out_shardings,
              donated_invars, **params):
  args, in_tree = tree_flatten((xs, jacs, laps))
  avals = tuple(map(shaped_abstractify, args))
  jaxpr_, out_tree = _fwdlap_jaxpr(jaxpr, avals, in_tree)
  n_in = len(jaxpr_.in_avals) - len(jaxpr.in_avals)
  n_out = len(jaxpr_.out_avals) - len(jaxpr.out_avals)
  donated_invars_ = donated_invars + (False,) * n_in
  in_shardings_ = in_shardings + (sharding_impls.UNSPECIFIED,) * n_in
  out_shardings_ = out_shardings + (sharding_impls.UNSPECIFIED,) * n_out
  outs = pjit.pjit_p.bind(*args, in_shardings=in_shardings_,
                          out_shardings=out_shardings_, jaxpr=jaxpr_,
                          donated_invars=donated_invars_, **params)
  return tree_unflatten(out_tree, outs)
rules[pjit.pjit_p] = pjit_rule

# api

def fwdlap(fun):
  def lapfun(*args):
    jacs = tree_map(_eye_like, args)
    lapvecs = tree_map(jnp.zeros_like, args)
    _, _, lapvecs = fwdlap3(fun, args, jacs, lapvecs)
    return lapvecs
  return lapfun

def _eye_like(x): return jnp.eye(x.size).reshape(*x.shape, x.size)

## test

def f(x):
  n, = x.shape
  A = jax.random.normal(jax.random.key(0), (n, n))
  return jnp.tanh(A @ x)

x = jnp.arange(3.)
lap = fwdlap(f)(x)
print(lap)
print(jnp.trace(jax.hessian(f)(x), 0, -1, -2))
