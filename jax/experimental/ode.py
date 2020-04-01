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

"""JAX-based Dormand-Prince ODE integration with adaptive stepsize.

Integrate systems of ordinary differential equations (ODEs) using the JAX
autograd/diff library and the Dormand-Prince method for adaptive integration
stepsize calculation. Provides improved integration accuracy over fixed
stepsize integration methods.

Adjoint algorithm based on Appendix C of https://arxiv.org/pdf/1806.07366.pdf
"""


from functools import partial
import operator as op
import time

import jax
import jax.numpy as np
from jax import lax
from jax import ops
from jax.util import safe_map, safe_zip
from jax.flatten_util import ravel_pytree
from jax.test_util import check_grads
from jax.tree_util import tree_map
from jax import linear_util as lu
import numpy as onp
import scipy.integrate as osp_integrate

map = safe_map
zip = safe_zip


def ravel_first_arg(f, unravel):
  return ravel_first_arg_(lu.wrap_init(f), unravel).call_wrapped

@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
  y = unravel(y_flat)
  ans = yield (y,) + args, {}
  ans_flat, _ = ravel_pytree(ans)
  yield ans_flat

def interp_fit_dopri(y0, y1, k, dt):
  # Fit a polynomial to the results of a Runge-Kutta step.
  dps_c_mid = np.array([
      6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
      -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
      -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2])
  y_mid = y0 + dt * np.dot(dps_c_mid, k)
  return np.array(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))

def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
  a = -2.*dt*dy0 + 2.*dt*dy1 -  8.*y0 -  8.*y1 + 16.*y_mid
  b =  5.*dt*dy0 - 3.*dt*dy1 + 18.*y0 + 14.*y1 - 32.*y_mid
  c = -4.*dt*dy0 +    dt*dy1 - 11.*y0 -  5.*y1 + 16.*y_mid
  d = dt * dy0
  e = y0
  return a, b, c, d, e

def initial_step_size(fun, t0, y0, order, rtol, atol, f0):
  # Algorithm from:
  # E. Hairer, S. P. Norsett G. Wanner,
  # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
  scale = atol + np.abs(y0) * rtol
  d0 = np.linalg.norm(y0 / scale)
  d1 = np.linalg.norm(f0 / scale)

  h0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

  y1 = y0 + h0 * f0
  f1 = fun(y1, t0 + h0)
  d2 = np.linalg.norm((f1 - f0) / scale) / h0

  h1 = np.where((d1 <= 1e-15) & (d2 <= 1e-15),
                np.maximum(1e-6, h0 * 1e-3),
                (0.01 / np.max(d1 + d2)) ** (1. / (order + 1.)))

  return np.minimum(100. * h0, h1)

def runge_kutta_step(func, y0, f0, t0, dt):
  # Dopri5 Butcher tableaux
  alpha = np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0])
  beta = np.array([
      [1 / 5, 0, 0, 0, 0, 0, 0],
      [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
      [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
  ])
  c_sol = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
  c_error = np.array([35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085,
                      125 / 192 - 451 / 720, -2187 / 6784 - -12231 / 42400,
                      11 / 84 - 649 / 6300, -1. / 60.])

  def body_fun(i, k):
    ti = t0 + dt * alpha[i-1]
    yi = y0 + dt * np.dot(beta[i-1, :], k)
    ft = func(yi, ti)
    return ops.index_update(k, jax.ops.index[i, :], ft)

  k = ops.index_update(np.zeros((7, f0.shape[0])), ops.index[0, :], f0)
  k = lax.fori_loop(1, 7, body_fun, k)

  y1 = dt * np.dot(c_sol, k) + y0
  y1_error = dt * np.dot(c_error, k)
  f1 = k[-1]
  return y1, f1, y1_error, k

def error_ratio(error_estimate, rtol, atol, y0, y1):
  err_tol = atol + rtol * np.maximum(np.abs(y0), np.abs(y1))
  err_ratio = error_estimate / err_tol
  return np.mean(err_ratio ** 2)

def optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0,
                      dfactor=0.2, order=5.0):
  """Compute optimal Runge-Kutta stepsize."""
  mean_error_ratio = np.max(mean_error_ratio)
  dfactor = np.where(mean_error_ratio < 1, 1.0, dfactor)

  err_ratio = np.sqrt(mean_error_ratio)
  factor = np.maximum(1.0 / ifactor,
                      np.minimum(err_ratio**(1.0 / order) / safety, 1.0 / dfactor))
  return np.where(mean_error_ratio == 0, last_step * ifactor, last_step / factor)

def odeint(func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=np.inf):
  """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

  Args:
    func: function to evaluate the time derivative of the solution `y` at time
      `t` as `func(y, t, *args)`, producing the same shape/structure as `y0`.
    y0: array or pytree of arrays representing the initial value for the state.
    t: array of float times for evaluation, like `np.linspace(0., 10., 101)`,
      in which the values must be strictly increasing.
    *args: tuple of additional arguments for `func`.
    rtol: float, relative local error tolerance for solver (optional).
    atol: float, absolute local error tolerance for solver (optional).
    mxstep: int, maximum number of steps to take for each timepoint (optional).

  Returns:
    Values of the solution `y` (i.e. integrated system values) at each time
    point in `t`, represented as an array (or pytree of arrays) with the same
    shape/structure as `y0` except with a new leading axis of length `len(t)`.
  """
  _init_nfe = 0.  # add argument to input signature so that VJP can return b-NFE
  return _odeint_wrapper(func, rtol, atol, mxstep, _init_nfe, y0, t, *args)

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _odeint_wrapper(func, rtol, atol, mxstep, _init_nfe, y0, ts, *args):
  y0, unravel = ravel_pytree(y0)
  func = ravel_first_arg(func, unravel)
  nfe, out = _odeint(func, rtol, atol, mxstep, _init_nfe, y0, ts, *args)
  flat_, unravel = ravel_pytree((nfe, jax.vmap(unravel)(out)))
  return unravel(flat_)

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _odeint(func, rtol, atol, mxstep, _init_nfe, y0, ts, *args):
  func_ = lambda y, t: func(y, t, *args)

  def scan_fun(carry, target_t):

    def cond_fun(state):
      i, _, _, t, _, _, _ = state
      return (t < target_t) & (i < mxstep)

    def body_fun(state):
      i, y, f, t, dt, last_t, interp_coeff = state
      next_y, next_f, next_y_error, k = runge_kutta_step(func_, y, f, t, dt)
      next_t = t + dt
      error_ratios = error_ratio(next_y_error, rtol, atol, y, next_y)
      new_interp_coeff = interp_fit_dopri(y, next_y, k, dt)
      dt = optimal_step_size(dt, error_ratios)

      new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
      old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
      return map(partial(np.where, np.all(error_ratios <= 1.)), new, old)

    nfe = carry[-1]
    n_steps, *carry_ = lax.while_loop(cond_fun, body_fun, [0] + carry[:-1])
    carry = carry_ + [nfe + 6 * n_steps]
    _, _, t, _, last_t, interp_coeff = carry[:-1]
    relative_output_time = (target_t - last_t) / (t - last_t)
    y_target = np.polyval(interp_coeff, relative_output_time)
    return carry, y_target

  f0 = func_(y0, ts[0])
  dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
  interp_coeff = np.array([y0] * 5)
  init_nfe = 2.  # real init NFE, since picking initial step size takes 2 NFE
  init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff, init_nfe]
  carry, ys = lax.scan(scan_fun, init_carry, ts[1:])
  nfe = carry[-1]
  return nfe, np.concatenate((y0[None], ys))

def _odeint_fwd(func, rtol, atol, mxstep, _init_nfe, y0, ts, *args):
  nfe, ys = _odeint(func, rtol, atol, mxstep, _init_nfe, y0, ts, *args)
  return (nfe, ys), (ys, ts, args)

def _odeint_rev(func, rtol, atol, mxstep, res, g):
  ys, ts, args = res

  def aug_dynamics(augmented_state, t, *args):
    """Original system augmented with vjp_y, vjp_t and vjp_args."""
    y, y_bar, *_ = augmented_state
    y_dot, vjpfun = jax.vjp(func, y, -t, *args)
    return (-y_dot, *vjpfun(y_bar))

  y_bar = g[-1]
  ts_bar = []
  t0_bar = 0.

  def scan_fun(carry, i):
    y_bar, t0_bar, args_bar, nfe = carry
    # Compute effect of moving measurement time
    t_bar = np.dot(func(ys[i], ts[i], *args), g[i])
    t0_bar = t0_bar - t_bar
    # Run augmented system backwards to previous observation
    cur_nfe, (_, y_bar, t0_bar, args_bar) = odeint(
        aug_dynamics, (ys[i], y_bar, t0_bar, args_bar), np.array([ts[i - 1], ts[i]]),
        *args, rtol=rtol, atol=atol, mxstep=mxstep)
    nfe += cur_nfe + 1  # add 1 for calculating t_bar
    y_bar, t0_bar, args_bar = tree_map(op.itemgetter(1), (y_bar, t0_bar, args_bar))
    # Add gradient from current output
    y_bar = y_bar + g[i - 1]
    return (y_bar, t0_bar, args_bar, nfe), t_bar

  # remove cotangent wrt nfe
  # this assumes the nfe output has no outgoing edges in the computation graph
  _, g = g
  init_carry = (g[-1], 0., tree_map(np.zeros_like, args), 0.)
  (y_bar, t0_bar, args_bar, nfe), rev_ts_bar = lax.scan(
      scan_fun, init_carry, np.arange(len(ts) - 1, 0, -1))
  ts_bar = np.concatenate([np.array([t0_bar]), rev_ts_bar[::-1]])
  # use spot for gradient wrt NFE argument for reporting b-NFE
  return (nfe, y_bar, ts_bar, *args_bar)

_odeint.defvjp(_odeint_fwd, _odeint_rev)


def pend(np, y, _, m, g):
  theta, omega = y
  return [omega, -m * omega - g * np.sin(theta)]

def benchmark_odeint(fun, y0, tspace, *args):
  """Time performance of JAX odeint method against scipy.integrate.odeint."""
  n_trials = 10
  n_repeat = 100
  y0, tspace = onp.array(y0), onp.array(tspace)
  onp_fun = partial(fun, onp)
  scipy_times = []
  for k in range(n_trials):
    start = time.time()
    for _ in range(n_repeat):
      scipy_result = osp_integrate.odeint(onp_fun, y0, tspace, args)
    end = time.time()
    print('scipy odeint elapsed time ({} of {}): {}'.format(k+1, n_trials, end-start))
    scipy_times.append(end - start)
  y0, tspace = np.array(y0), np.array(tspace)
  jax_fun = partial(fun, np)
  jax_times = []
  for k in range(n_trials):
    start = time.time()
    for _ in range(n_repeat):
      nfe, jax_result = odeint(jax_fun, y0, tspace, *args)
    jax_result.block_until_ready()
    end = time.time()
    print('JAX odeint elapsed time ({} of {}): {}'.format(k+1, n_trials, end-start))
    jax_times.append(end - start)
  print('(avg scipy time) / (avg jax time) = {}'.format(
      onp.mean(scipy_times[1:]) / onp.mean(jax_times[1:])))
  print('norm(scipy result-jax result): {}'.format(
      np.linalg.norm(np.asarray(scipy_result) - jax_result)))
  return scipy_result, jax_result

def pend_benchmark_odeint():
  _, _ = benchmark_odeint(pend, [np.pi - 0.1, 0.0], np.linspace(0., 10., 101),
                          0.25, 9.8)

def pend_check_grads():
  def f(y0, ts, *args):
    return odeint(partial(pend, np), y0, ts, *args)[1]

  y0 = [np.pi - 0.1, 0.0]
  ts = np.linspace(0., 1., 11)
  args = (0.25, 9.8)

  check_grads(f, (y0, ts, *args), modes=["rev"], order=2,
              atol=1e-1, rtol=1e-1)

def pend_get_nfe():
  def f(init_nfe, y0, ts, *args):
    return _odeint_wrapper(partial(pend, np),
                           1.4e-8,
                           1.4e-8,
                           np.inf,
                           init_nfe,
                           y0, ts, *args)

  y0 = [np.pi - 0.1, 0.0]
  ts = np.linspace(0., 1., 11)
  args = (0.25, 9.8)

  f_nfe, ys = f(0., y0, ts, *args)
  _, vjpfun_ = jax.vjp(f, 0., y0, ts, *args)
  b_nfe, *cotangent_out = vjpfun_((f_nfe, ys))
  print("Forward NFE:\t", int(f_nfe))
  print("Backward NFE:\t", int(b_nfe))


if __name__ == '__main__':
  pend_benchmark_odeint()
  pend_check_grads()
  pend_get_nfe()
