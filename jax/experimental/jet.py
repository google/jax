# Copyright 2020 Google LLC
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

from typing import Callable

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import core
from jax._src.util import unzip2
from jax._src import ad_util
from jax.tree_util import (register_pytree_node, tree_structure,
                           treedef_is_leaf, tree_flatten, tree_unflatten)
import jax.linear_util as lu
from jax.interpreters import xla
from jax.custom_derivatives import custom_jvp_call_jaxpr_p
from jax._src.lax import lax
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import fft as lax_fft

def jet(fun, primals, series):
  try:
    order, = set(map(len, series))
  except ValueError:
    msg = "jet terms have inconsistent lengths for different arguments"
    raise ValueError(msg) from None

  # TODO(mattjj): consider supporting pytree inputs
  for i, (x, terms) in enumerate(zip(primals, series)):
    treedef = tree_structure(x)
    if not treedef_is_leaf(treedef):
      raise ValueError("primal value at position {} is not an array".format(i))
    for j, t in enumerate(terms):
      treedef = tree_structure(t)
      if not treedef_is_leaf(treedef):
        raise ValueError("term {} for argument {} is not an array".format(j, i))

  @lu.transformation_with_aux
  def flatten_fun_output(*args):
    ans = yield args, {}
    yield tree_flatten(ans)

  f, out_tree = flatten_fun_output(lu.wrap_init(fun))
  out_primals, out_terms = jet_fun(jet_subtrace(f), order).call_wrapped(primals, series)
  return tree_unflatten(out_tree(), out_primals), tree_unflatten(out_tree(), out_terms)

@lu.transformation
def jet_fun(order, primals, series):
  with core.new_main(JetTrace) as main:
    main.order = order
    out_primals, out_terms = yield (main, primals, series), {}
    del main
  out_terms = [[np.zeros_like(p)] * order if s is zero_series else s
               for p, s in zip(out_primals, out_terms)]
  yield out_primals, out_terms

@lu.transformation
def jet_subtrace(main, primals, series):
  trace = JetTrace(main, core.cur_sublevel())
  in_tracers = map(partial(JetTracer, trace), primals, series)
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  out_primals, out_terms = unzip2((t.primal, t.terms) for t in out_tracers)
  yield out_primals, out_terms

@lu.transformation_with_aux
def traceable(in_tree_def, *primals_and_series):
  primals_in, series_in = tree_unflatten(in_tree_def, primals_and_series)
  primals_out, series_out = yield (primals_in, series_in), {}
  out_flat, out_tree_def = tree_flatten((primals_out, series_out))
  yield out_flat, out_tree_def


class JetTracer(core.Tracer):
  __slots__ = ["primal", "terms"]

  def __init__(self, trace, primal, terms):
    assert type(terms) in (ZeroSeries, list, tuple)
    self._trace = trace
    self.primal = primal
    self.terms = terms

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def full_lower(self):
    if self.terms is zero_series or all(t is zero_term for t in self.terms):
      return core.full_lower(self.primal)
    else:
      return self

class JetTrace(core.Trace):

  def pure(self, val):
    return JetTracer(self, val, zero_series)

  def lift(self, val):
    return JetTracer(self, val, zero_series)

  def sublift(self, val):
    return JetTracer(self, val.primal, val.terms)

  def process_primitive(self, primitive, tracers, params):
    order = self.main.order              # pytype: disable=attribute-error
    primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
    series_in = [[zero_term] * order if s is zero_series else s
                 for s in series_in]
    # TODO(mattjj): avoid always instantiating zeros
    series_in = [[np.zeros(np.shape(x), dtype=np.result_type(x))
                  if t is zero_term else t for t in series]
                 for x, series in zip(primals_in, series_in)]
    rule = jet_rules[primitive]
    primal_out, terms_out = rule(primals_in, series_in, **params)
    if not primitive.multiple_results:
      return JetTracer(self, primal_out, terms_out)
    else:
      return [JetTracer(self, p, ts) for p, ts in zip(primal_out, terms_out)]

  def process_call(self, call_primitive, f, tracers, params):
    primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
    primals_and_series, in_tree_def = tree_flatten((primals_in, series_in))
    f_jet, out_tree_def = traceable(jet_subtrace(f, self.main), in_tree_def)
    update_params = call_param_updaters.get(call_primitive)
    new_params = (update_params(params, len(primals_and_series))
                  if update_params else params)
    result = call_primitive.bind(f_jet, *primals_and_series, **new_params)
    primals_out, series_out = tree_unflatten(out_tree_def(), result)
    return [JetTracer(self, p, ts) for p, ts in zip(primals_out, series_out)]

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    # TODO(mattjj): don't just ignore custom jvp rules?
    del primitive, jvp  # Unused.
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
    del primitive, fwd, bwd, out_trees  # Unused.
    return fun.call_wrapped(*tracers)


class ZeroTerm: pass
zero_term = ZeroTerm()
register_pytree_node(ZeroTerm, lambda z: ((), None), lambda _, xs: zero_term)

class ZeroSeries: pass
zero_series = ZeroSeries()
register_pytree_node(ZeroSeries, lambda z: ((), None), lambda _, xs: zero_series)


call_param_updaters = {}

def _xla_call_param_updater(params, num_inputs):
  donated_invars = params['donated_invars']
  if any(donated_invars):
    raise NotImplementedError("donated_invars not supported with jet")
  return dict(params, donated_invars=(False,) * num_inputs)
call_param_updaters[xla.xla_call_p] = _xla_call_param_updater


### rule definitions

jet_rules = {}

def defzero(prim):
  jet_rules[prim] = partial(zero_prop, prim)

def zero_prop(prim, primals_in, series_in, **params):
  primal_out = prim.bind(*primals_in, **params)
  return primal_out, zero_series

defzero(lax.le_p)
defzero(lax.lt_p)
defzero(lax.gt_p)
defzero(lax.ge_p)
defzero(lax.eq_p)
defzero(lax.ne_p)
defzero(lax.not_p)
defzero(lax.and_p)
defzero(lax.or_p)
defzero(lax.xor_p)
defzero(lax.floor_p)
defzero(lax.ceil_p)
defzero(lax.round_p)
defzero(lax.sign_p)
defzero(ad_util.stop_gradient_p)
defzero(lax.is_finite_p)
defzero(lax.shift_left_p)
defzero(lax.shift_right_arithmetic_p)
defzero(lax.shift_right_logical_p)
defzero(lax.bitcast_convert_type_p)


def deflinear(prim):
  jet_rules[prim] = partial(linear_prop, prim)

def linear_prop(prim, primals_in, series_in, **params):
  primal_out = prim.bind(*primals_in, **params)
  series_out = [prim.bind(*terms_in, **params) for terms_in in zip(*series_in)]
  return primal_out, series_out

deflinear(lax.neg_p)
deflinear(lax.real_p)
deflinear(lax.complex_p)
deflinear(lax.conj_p)
deflinear(lax.imag_p)
deflinear(lax.add_p)
deflinear(ad_util.add_jaxvals_p)
deflinear(lax.sub_p)
deflinear(lax.convert_element_type_p)
deflinear(lax.broadcast_in_dim_p)
deflinear(lax.concatenate_p)
deflinear(lax.pad_p)
deflinear(lax.reshape_p)
deflinear(lax.rev_p)
deflinear(lax.transpose_p)
deflinear(lax.slice_p)
deflinear(lax.reduce_sum_p)
deflinear(lax.reduce_window_sum_p)
deflinear(lax_fft.fft_p)
deflinear(xla.device_put_p)

def _cumulative_jet_rule(primals_in, series_in, *, axis: int, reverse: bool,
                         combine_fn: Callable):
  # Irrespective of backend, we always use the parallel prefix scan
  # implementation when differentiating because reduce_window is not
  # arbitrarily differentiable.
  return jet(partial(lax_control_flow.associative_scan, combine_fn, axis=axis,
                     reverse=reverse),
             primals_in, series_in)

deflinear(lax_control_flow.cumsum_p)
jet_rules[lax_control_flow.cumprod_p] = partial(_cumulative_jet_rule,
                                                combine_fn=lax.mul)
jet_rules[lax_control_flow.cummax_p] = partial(_cumulative_jet_rule,
                                               combine_fn=lax.max)
jet_rules[lax_control_flow.cummin_p] = partial(_cumulative_jet_rule,
                                               combine_fn=lax.min)


def def_deriv(prim, deriv):
  """
  Define the jet rule for a primitive in terms of its first derivative.
  """
  jet_rules[prim] = partial(deriv_prop, prim, deriv)


def deriv_prop(prim, deriv, primals_in, series_in):
  x, = primals_in
  series, = series_in
  primal_out = prim.bind(x)
  c0, cs = jet(deriv, primals_in, series_in)
  c = [c0] + cs
  u = [x] + series
  v = [primal_out] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))
  primal_out, *series_out = v
  return primal_out, series_out


def_deriv(lax.erf_p, lambda x: lax.mul(lax._const(x, 2. / np.sqrt(np.pi)), lax.exp(lax.neg(lax.square(x)))))


def def_comp(prim, comp):
  """
  Define the jet rule for a primitive in terms of a composition of simpler primitives.
  """
  jet_rules[prim] = partial(jet, comp)


def_comp(lax.expm1_p, lambda x: lax.exp(x) - 1)
def_comp(lax.log1p_p, lambda x: lax.log(1 + x))
def_comp(lax.sqrt_p, lambda x: x ** 0.5)
def_comp(lax.rsqrt_p, lambda x: x ** -0.5)
def_comp(lax.asinh_p, lambda x: lax.log(x + lax.sqrt(lax.square(x) + 1)))
def_comp(lax.acosh_p, lambda x: lax.log(x + lax.sqrt(lax.square(x) - 1)))
def_comp(lax.atanh_p, lambda x: 0.5 * lax.log(lax.div(1 + x, 1 - x)))
def_comp(lax.erfc_p, lambda x: 1 - lax.erf(x))
def_comp(lax.rem_p, lambda x, y: x - y * lax.floor(x / y))
def_comp(lax.clamp_p, lambda a, x, b: lax.min(lax.max(a, x), b))


def _erf_inv_rule(primals_in, series_in):
  x, = primals_in
  series, = series_in

  u = [x] + series
  primal_out = lax.erf_inv(x)
  v = [primal_out] + [None] * len(series)

  # derivative on co-domain for caching purposes
  deriv_const = np.sqrt(np.pi) / 2.
  deriv_y = lambda y: lax.mul(deriv_const, lax.exp(lax.square(y)))

  # manually propagate through deriv_y since we don't have lazy evaluation of sensitivities

  c = [deriv_y(primal_out)] + [None] * (len(series) - 1)
  tmp_sq = [lax.square(v[0])] + [None] * (len(series) - 1)
  tmp_exp = [lax.exp(tmp_sq[0])] + [None] * (len(series) - 1)
  for k in range(1, len(series)):
    # we know c[:k], we compute c[k]

    # propagate c to get v
    v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))

    # propagate v to get next c

    # square
    tmp_sq[k] = fact(k) * sum(_scale2(k, j) * v[k-j] * v[j] for j in range(k + 1))

    # exp
    tmp_exp[k] = fact(k-1) * sum(_scale(k, j) * tmp_exp[k-j] * tmp_sq[j] for j in range(1, k + 1))

    # const
    c[k] = deriv_const * tmp_exp[k]

  # we can't, and don't need, to compute c[k+1], just need to get the last v[k]
  k = len(series)
  v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))

  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.erf_inv_p] = _erf_inv_rule

### More complicated rules

def fact(n):
  return lax.exp(lax.lgamma(n+1.))

def _scale(k, j):
  return 1. / (fact(k - j) * fact(j - 1))

def _scale2(k, j):
  return 1. / (fact(k - j) * fact(j))

def _exp_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.exp(x)] + [None] * len(series)
  for k in range(1,len(v)):
    v[k] = fact(k-1) * sum([_scale(k, j)* v[k-j] * u[j] for j in range(1, k+1)])
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.exp_p] = _exp_taylor

def _pow_taylor(primals_in, series_in):
  u_, r_ = primals_in

  x, series = jet(lambda x, y: lax.mul(y, lax.log(x)), primals_in, series_in)

  u = [x] + series
  v = [u_ ** r_] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum([_scale(k, j)* v[k-j] * u[j] for j in range(1, k+1)])
  primal_out, *series_out = v

  return primal_out, series_out
jet_rules[lax.pow_p] = _pow_taylor

def _integer_pow_taylor(primals_in, series_in, *, y):
  if y == 0:
    return jet(jnp.ones_like, primals_in, series_in)
  elif y == 1:
    return jet(lambda x: x, primals_in, series_in)
  elif y == 2:
    return jet(lambda x: x * x, primals_in, series_in)
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.integer_pow(x, y)] + [None] * len(series)
  for k in range(1, len(v)):
    vu = sum(_scale(k, j) * v[k-j] * u[j] for j in range(1, k + 1))
    uv = sum(_scale(k, j) * u[k-j] * v[j] for j in range(1, k))
    v[k] = jnp.where(x == 0, 0, fact(k-1) * (y * vu - uv) / x)
  primal_out, *series_out = v

  return primal_out, series_out
jet_rules[lax.integer_pow_p] = _integer_pow_taylor


def _expit_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [jax.scipy.special.expit(x)] + [None] * len(series)
  e = [v[0] * (1 - v[0])] + [None] * len(series)  # terms for sigmoid' = sigmoid * (1 - sigmoid)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum([_scale(k, j) * e[k-j] * u[j] for j in range(1, k+1)])
    e[k] = (1 - v[0]) * v[k] - fact(k) * sum([_scale2(k, j)* v[j] * v[k-j] for j in range(1, k+1)])

  primal_out, *series_out = v
  return primal_out, series_out

def _tanh_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [2*x] + [2 * series_ for series_ in series]
  primals_in, *series_in = u
  primal_out, series_out = _expit_taylor((primals_in, ), (series_in, ))
  series_out = [2 * series_ for series_ in series_out]
  return 2 * primal_out - 1, series_out
jet_rules[lax.tanh_p] = _tanh_taylor

def _log_taylor(primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  v = [lax.log(x)] + [None] * len(series)
  for k in range(1, len(v)):
    conv = sum([_scale(k, j) * v[j] * u[k-j] for j in range(1, k)])
    v[k] = (u[k] - fact(k - 1) * conv) / u[0]
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.log_p] = _log_taylor

def _atan2_taylor(primals_in, series_in):
  x, y = primals_in
  primal_out = lax.atan2(x, y)

  x, series = jet(lax.div, primals_in, series_in)
  c0, cs = jet(lambda x: lax.div(1, 1 + lax.square(x)), (x, ), (series, ))
  c = [c0] + cs
  u = [x] + series
  v = [primal_out] + [None] * len(series)
  for k in range(1, len(v)):
    v[k] = fact(k-1) * sum(_scale(k, j) * c[k-j] * u[j] for j in range(1, k + 1))
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.atan2_p] = _atan2_taylor

def _div_taylor_rule(primals_in, series_in):
  x, y = primals_in
  x_terms, y_terms = series_in
  u = [x] + x_terms
  w = [y] + y_terms
  v = [None] * len(u)
  def scale(k, j): return 1. / (fact(k - j) * fact(j))
  for k in range(0, len(v)):
    conv = sum([scale(k, j) * v[j] * w[k-j] for j in range(0, k)])
    v[k] = (u[k] - fact(k) * conv) / w[0]
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.div_p] = _div_taylor_rule

def _sinusoidal_rule(sign, prims, primals_in, series_in):
  x, = primals_in
  series, = series_in
  u = [x] + series
  s, c = prims
  s = [s(x)] + [None] * len(series)
  c = [c(x)] + [None] * len(series)
  for k in range(1, len(s)):
    s[k] = fact(k-1) * sum(_scale(k, j) * u[j] * c[k-j] for j in range(1, k + 1))
    c[k] = fact(k-1) * sum(_scale(k, j) * u[j] * s[k-j] for j in range(1, k + 1)) * sign
  return (s[0], s[1:]), (c[0], c[1:])

def _get_ind(f, ind):
  return lambda *args: f(*args)[ind]

jet_rules[lax.sin_p] = _get_ind(partial(_sinusoidal_rule, -1, (lax.sin, lax.cos)), 0)
jet_rules[lax.cos_p] = _get_ind(partial(_sinusoidal_rule, -1, (lax.sin, lax.cos)), 1)
jet_rules[lax.sinh_p] = _get_ind(partial(_sinusoidal_rule, 1, (lax.sinh, lax.cosh)), 0)
jet_rules[lax.cosh_p] = _get_ind(partial(_sinusoidal_rule, 1, (lax.sinh, lax.cosh)), 1)

def _bilinear_taylor_rule(prim, primals_in, series_in, **params):
  x, y = primals_in
  x_terms, y_terms = series_in
  u = [x] + x_terms
  w = [y] + y_terms
  v = [None] * len(u)
  op = partial(prim.bind, **params)
  def scale(k, j): return 1. / (fact(k - j) * fact(j))
  for k in range(0, len(v)):
    v[k] = fact(k) * sum([scale(k, j) * op(u[j], w[k-j]) for j in range(0, k+1)])
  primal_out, *series_out = v
  return primal_out, series_out
jet_rules[lax.dot_general_p] = partial(_bilinear_taylor_rule, lax.dot_general_p)
jet_rules[lax.mul_p] = partial(_bilinear_taylor_rule, lax.mul_p)
jet_rules[lax.conv_general_dilated_p] = partial(_bilinear_taylor_rule, lax.conv_general_dilated_p)

def _gather_taylor_rule(primals_in, series_in, **params):
  operand, start_indices = primals_in
  gs, _ = series_in
  primal_out = lax.gather_p.bind(operand, start_indices, **params)
  series_out = [lax.gather_p.bind(g, start_indices, **params) for g in gs]
  return primal_out, series_out
jet_rules[lax.gather_p] = _gather_taylor_rule

def _gen_reduce_choose_taylor_rule(chooser_fun):
  def chooser_taylor_rule(primals_in, series_in, **params):
    operand, = primals_in
    gs, = series_in
    primal_out = chooser_fun(operand, **params)
    axes = params.pop("axes", None)
    primal_dtype = gs[0].dtype
    shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
    location_indicators = lax.convert_element_type(
          lax._eq_meet(operand, lax.reshape(primal_out, shape)), primal_dtype)
    counts = lax._reduce_sum(location_indicators, axes)
    def _reduce_chooser_taylor_rule(g):
      return lax.div(lax._reduce_sum(lax.mul(g, location_indicators), axes), counts)
    series_out = [_reduce_chooser_taylor_rule(g) for g in gs]
    return primal_out, series_out
  return chooser_taylor_rule
jet_rules[lax.reduce_max_p] = _gen_reduce_choose_taylor_rule(lax._reduce_max)
jet_rules[lax.reduce_min_p] = _gen_reduce_choose_taylor_rule(lax._reduce_min)

def _abs_taylor_rule(x, series_in, **params):
  x, = x
  zero = lax.full_like(x, 0, shape=())
  primal_out = lax.abs_p.bind(x, **params)
  negs = lax.select(lax.lt(x, zero), lax.full_like(x, -1), lax.full_like(x, 1.0))
  fix_sign = lambda y: negs * y
  series_out = [fix_sign(*terms_in, **params) for terms_in in zip(*series_in)]
  return primal_out, series_out
jet_rules[lax.abs_p] = _abs_taylor_rule

def _select_taylor_rule(primal_in, series_in, **params):
  b, x, y = primal_in
  primal_out = lax.select_p.bind(b, x, y, **params)
  sel = lambda _, x, y: lax.select(b, x, y)
  series_out = [sel(*terms_in, **params) for terms_in in zip(*series_in)]
  return primal_out, series_out
jet_rules[lax.select_p] = _select_taylor_rule

def _lax_max_taylor_rule(primal_in, series_in):
    x, y = primal_in

    xgy = x > y   # greater than mask
    xey = x == y  # equal to mask
    primal_out = lax.select(xgy, x, y)

    def select_max_and_avg_eq(x_i, y_i):
        """Select x where x>y or average when x==y"""
        max_i = lax.select(xgy, x_i, y_i)
        max_i = lax.select(xey, (x_i + y_i)/2, max_i)
        return max_i

    series_out = [select_max_and_avg_eq(*terms_in) for terms_in in zip(*series_in)]
    return primal_out, series_out
jet_rules[lax.max_p] = _lax_max_taylor_rule

def _lax_min_taylor_rule(primal_in, series_in):
    x, y = primal_in
    xgy = x < y   # less than mask
    xey = x == y  # equal to mask
    primal_out = lax.select(xgy, x, y)

    def select_min_and_avg_eq(x_i, y_i):
        """Select x where x>y or average when x==y"""
        min_i = lax.select(xgy, x_i, y_i)
        min_i = lax.select(xey, (x_i + y_i)/2, min_i)
        return min_i

    series_out = [select_min_and_avg_eq(*terms_in) for terms_in in zip(*series_in)]
    return primal_out, series_out
jet_rules[lax.min_p] = _lax_min_taylor_rule

def _custom_jvp_call_jaxpr_rule(primals_in, series_in, *, fun_jaxpr,
                                jvp_jaxpr_thunk):
  # TODO(mattjj): do something better than ignoring custom jvp rules for jet?
  del jvp_jaxpr_thunk
  return jet(core.jaxpr_as_fun(fun_jaxpr), primals_in, series_in)
jet_rules[custom_jvp_call_jaxpr_p] = _custom_jvp_call_jaxpr_rule

def _scatter_add_rule(primals_in, series_in, *, update_jaxpr, update_consts,
                      dimension_numbers, indices_are_sorted, unique_indices):
  bind = partial(lax.scatter_add_p.bind, update_jaxpr=update_jaxpr,
                 update_consts=update_consts, dimension_numbers=dimension_numbers,
                 indices_are_sorted=indices_are_sorted, unique_indices=unique_indices)
  operand, scatter_indices, updates = primals_in
  primal_out = bind(operand, scatter_indices, updates)
  series_out = [bind(d1, scatter_indices, d2) for d1, _, d2 in zip(*series_in)]
  return primal_out, series_out
jet_rules[lax.scatter_add_p] = _scatter_add_rule
