# Copyright 2022 The JAX Authors.
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

"""Tests for lobpcg routine.

If LOBPCG_DEBUG_PLOT_DIR is set, exports debug visuals to that directory.
Requires matplotlib.
"""

import functools
import re
import os
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps

import jax
from jax._src import test_util as jtu
from jax.experimental.sparse import linalg, bcoo
import jax.numpy as jnp


def _clean_matrix_name(name):
  return re.sub('[^0-9a-zA-Z]+', '_', name)

def _make_concrete_cases(f64):
  dtype = np.float64 if f64 else np.float32
  example_names = list(_concrete_generators(dtype))
  cases = []
  for name in example_names:
    n, k, m, tol = 100, 10, 20, None
    if name == 'ring laplacian':
      m *= 3
    if name.startswith('linear'):
      m *= 2
    if f64:
      m *= 2
    if name.startswith('cluster') and not f64:
      tol = 2e-6
    clean_matrix_name = _clean_matrix_name(name)
    case = {
        'matrix_name': name,
        'n': n,
        'k': k,
        'm': m,
        'tol': tol,
        'testcase_name': f'{clean_matrix_name}_n{n}'
    }
    cases.append(case)

  assert len({c['testcase_name'] for c in cases}) == len(cases)
  return cases

def _make_callable_cases(f64):
  dtype = np.float64 if f64 else np.float32
  example_names = list(_callable_generators(dtype))
  return [{'testcase_name': _clean_matrix_name(n), 'matrix_name': n}
          for n in example_names]

def _make_ring(n):
  # from lobpcg scipy tests
  col = np.zeros(n)
  col[1] = 1
  A = sla.toeplitz(col)
  D = np.diag(A.sum(axis=1))
  L = D - A
  # Compute the full eigendecomposition using tricks, e.g.
  # http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
  tmp = np.pi * np.arange(n) / n
  analytic_w = 2 * (1 - np.cos(tmp))
  analytic_w.sort()
  analytic_w = analytic_w[::-1]

  return L, analytic_w

def _make_diag(diag):
  diag.sort()
  diag = diag[::-1]
  return np.diag(diag), diag

def _make_cluster(to_cluster, n):
  return _make_diag(
      np.array([1000] * to_cluster + [1] * (n - to_cluster)))

def _concrete_generators(dtype):
  d = {
      'id': lambda n, _k: _make_diag(np.ones(n)),
      'linear cond=1k': lambda n, _k: _make_diag(np.linspace(1, 1000, n)),
      'linear cond=100k':
          lambda n, _k: _make_diag(np.linspace(1, 100 * 1000, n)),
      'geom cond=1k': lambda n, _k: _make_diag(np.logspace(0, 3, n)),
      'geom cond=100k': lambda n, _k: _make_diag(np.logspace(0, 5, n)),
      'ring laplacian': lambda n, _k: _make_ring(n),
      'cluster(k/2)': lambda n, k: _make_cluster(k // 2, n),
      'cluster(k-1)': lambda n, k: _make_cluster(k - 1, n),
      'cluster(k)': lambda n, k: _make_cluster(k, n)}
  def cast_fn(fn):
    def casted_fn(n, k):
      result = fn(n, k)
      cast = functools.partial(np.array, dtype=dtype)
      return tuple(map(cast, result))
    return casted_fn
  return {k: cast_fn(v) for k, v in d.items()}

def _make_id_fn(n):
  return lambda x: x, np.ones(n), 5

def _make_diag_fn(diagonal, m):
  return lambda x: diagonal.astype(x.dtype) * x, diagonal, m

def _make_ring_fn(n, m):
  _, eigs = _make_ring(n)
  def ring_action(x):
    degree = 2 * x
    lnbr = jnp.roll(x, 1)
    rnbr = jnp.roll(x, -1)
    return degree - lnbr - rnbr
  return ring_action, eigs, m

def _make_randn_fn(n, k, m):
  rng = np.random.default_rng(1234)
  tall_skinny = rng.standard_normal((n, k))
  def randn_action(x):
    ts = jnp.array(tall_skinny, dtype=x.dtype)
    p = jax.lax.Precision.HIGHEST
    return ts.dot(ts.T.dot(x, precision=p), precision=p)
  _, s, _ = np.linalg.svd(tall_skinny, full_matrices=False)
  return randn_action, s ** 2, m

def _make_sparse_fn(n, fill):
  rng = np.random.default_rng(1234)
  slots = n ** 2
  filled = max(int(slots * fill), 1)
  pos = rng.choice(slots, size=filled, replace=False)
  posx, posy = divmod(pos, n)
  data = rng.standard_normal(len(pos))
  coo = sps.coo_matrix((data, (posx, posy)), shape=(n, n))

  def sparse_action(x):
    coo_typed = coo.astype(np.dtype(x.dtype))
    sps_mat = bcoo.BCOO.from_scipy_sparse(coo_typed)
    dn = (((1,), (0,)), ((), ()))  # Good old fashioned matmul.
    x = bcoo.bcoo_dot_general(sps_mat, x, dimension_numbers=dn)
    sps_mat_T = sps_mat.transpose()
    return bcoo.bcoo_dot_general(sps_mat_T, x, dimension_numbers=dn)

  dense = coo.todense()
  _, s, _ = np.linalg.svd(dense, full_matrices=False)
  return sparse_action, s ** 2, 20


def _callable_generators(dtype):
  n = 100
  topk = 10
  d = {
      'id': _make_id_fn(n),
      'linear cond=1k': _make_diag_fn(np.linspace(1, 1000, n), 40),
      'linear cond=100k': _make_diag_fn(np.linspace(1, 100 * 1000, n), 40),
      'geom cond=1k': _make_diag_fn(np.logspace(0, 3, n), 20),
      'geom cond=100k': _make_diag_fn(np.logspace(0, 5, n), 20),
      'ring laplacian': _make_ring_fn(n, 40),
      'randn': _make_randn_fn(n, topk, 40),
      'sparse 1%': _make_sparse_fn(n, 0.01),
      'sparse 10%': _make_sparse_fn(n, 0.10),
  }

  ret = {}
  for k, (vec_mul_fn, eigs, m) in d.items():
    if jtu.num_float_bits(dtype) > 32:
      m *= 3
    eigs.sort()

    # Note we must lift the vector multiply into matmul
    fn = jax.vmap(vec_mul_fn, in_axes=1, out_axes=1)

    ret[k] = (fn, eigs[::-1][:topk].astype(dtype), n, m)
  return ret


@jtu.with_config(
    jax_enable_checks=True,
    jax_debug_nans=True,
    jax_numpy_rank_promotion='raise',
    jax_traceback_filtering='off')
class LobpcgTest(jtu.JaxTestCase):

  def checkLobpcgConsistency(self, matrix_name, n, k, m, tol, dtype):
    A, eigs = _concrete_generators(dtype)[matrix_name](n, k)
    X = self.rng().standard_normal(size=(n, k)).astype(dtype)

    A, X = (jnp.array(i, dtype=dtype) for i in (A, X))
    theta, U, i = linalg.lobpcg_standard(A, X, m, tol)

    self.assertDtypesMatch(theta, A)
    self.assertDtypesMatch(U, A)

    self.assertLess(
        i, m, msg=f'expected early convergence iters {int(i)} < max {m}')

    issorted = theta[:-1] >= theta[1:]
    all_true = np.ones_like(issorted).astype(bool)
    self.assertArraysEqual(issorted, all_true)

    k = X.shape[1]
    relerr = np.abs(theta - eigs[:k]) / eigs[:k]
    for i in range(k):
      # The self-consistency property should be ensured.
      u = np.asarray(U[:, i], dtype=A.dtype)
      t = float(theta[i])
      Au = A.dot(u)
      resid = Au - t * u
      resid_norm = np.linalg.norm(resid)
      vector_norm = np.linalg.norm(Au)
      adjusted_error = resid_norm / n / (t + vector_norm) / 10

      eps = float(jnp.finfo(dtype).eps) if tol is None else tol
      self.assertLessEqual(
          adjusted_error,
          eps,
          msg=f'convergence criterion for eigenvalue {i} not satisfied, '
          f'floating point error {adjusted_error} not <= {eps}')

      # There's no real guarantee we can be within x% of the true eigenvalue.
      # However, for these simple unit test examples this should be met.
      tol = float(np.sqrt(eps)) * 10
      self.assertLessEqual(
          relerr[i],
          tol,
          msg=f'expected relative error within {tol}, was {float(relerr[i])}'
          f' for eigenvalue {i} (actual {float(theta[i])}, '
          f'expected {float(eigs[i])})')

  def checkLobpcgMonotonicity(self, matrix_name, n, k, m, tol, dtype):
    del tol
    A, eigs = _concrete_generators(dtype)[matrix_name](n, k)
    X = self.rng().standard_normal(size=(n, k)).astype(dtype)

    _theta, _U, _i, info = linalg._lobpcg_standard_matrix(
        A, X, m, tol=0, debug=True)
    self.assertArraysEqual(info['X zeros'], jnp.zeros_like(info['X zeros']))

    # To check for any divergence, make sure that the last 20% of
    # steps have lower worst-case relerr than first 20% of steps,
    # at least up to an order of magnitude.
    #
    # This is non-trivial, as many implementations have catastrophic
    # cancellations at convergence for residual terms, and rely on
    # brittle locking tolerance to avoid divergence.
    eigs = eigs[:k]
    relerrs = np.abs(np.array(info['lambda history']) - eigs) / eigs
    few_steps = max(m // 5, 1)
    self.assertLess(
        relerrs[-few_steps:].max(axis=1).mean(),
        10 * relerrs[:few_steps].max(axis=1).mean())

    self._possibly_plot(A, eigs, X, m, matrix_name)

  def _possibly_plot(self, A, eigs, X, m, matrix_name):
    if not os.getenv('LOBPCG_EMIT_DEBUG_PLOTS'):
      return

    if isinstance(A, (np.ndarray, jax.Array)):
      lobpcg = linalg._lobpcg_standard_matrix
    else:
      lobpcg = linalg._lobpcg_standard_callable
    _theta, _U, _i, info = lobpcg(A, X, m, tol=0, debug=True)
    plot_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')
    assert plot_dir, 'expected TEST_UNDECLARED_OUTPUTS_DIR for lobpcg plots'
    self._debug_plots(X, eigs, info, matrix_name, plot_dir)

  def _debug_plots(self, X, eigs, info, matrix_name, lobpcg_debug_plot_dir):
    # We import matplotlib lazily because (a) it's faster this way, and
    # (b) concurrent imports of matplotlib appear to trigger some sort of
    # collision on the matplotlib cache lock on Windows.
    try:
      from matplotlib import pyplot as plt
    except (ModuleNotFoundError, ImportError):
      return  # If matplotlib isn't available, don't emit plots.

    os.makedirs(lobpcg_debug_plot_dir, exist_ok=True)
    clean_matrix_name = _clean_matrix_name(matrix_name)
    n, k = X.shape
    dt = 'f32' if X.dtype == np.float32 else 'f64'
    figpath = os.path.join(
        lobpcg_debug_plot_dir,
        f'{clean_matrix_name}_n{n}_k{k}_{dt}.png')

    plt.switch_backend('Agg')

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(24, 4))
    fig.suptitle(fr'{matrix_name} ${n=},{k=}$, {dt}')
    line_styles = [':', '--', '-.', '-']

    for key, ls in zip(['X orth', 'P orth', 'P.X'], line_styles):
      ax0.semilogy(info[key], ls=ls, label=key)
    ax0.set_title('basis average orthogonality')
    ax0.legend()

    relerrs = np.abs(np.array(info['lambda history']) - eigs) / eigs
    keys = ['max', 'p50', 'min']
    fns = [np.max, np.median, np.min]
    for key, fn, ls in zip(keys, fns, line_styles):
      ax1.semilogy(fn(relerrs, axis=1), ls=ls, label=key)
    ax1.set_title('eigval relerr')
    ax1.legend()

    for key, ls in zip(['basis rank', 'converged', 'P zeros'], line_styles):
      ax2.plot(info[key], ls=ls, label=key)
    ax2.set_title('basis dimension counts')
    ax2.legend()

    prefix = 'adjusted residual'
    for key, ls in zip(keys, line_styles):
      ax3.semilogy(info[prefix + ' ' + key], ls=ls, label=key)
    ax3.axhline(np.finfo(X.dtype).eps, label='eps', c='k')
    ax3.legend()
    ax3.set_title(prefix + rf' $\lambda_{{\max}}=\ ${eigs[0]:.1e}')

    fig.savefig(figpath, bbox_inches='tight')
    plt.close(fig)

  def checkApproxEigs(self, example_name, dtype):
    fn, eigs, n, m = _callable_generators(dtype)[example_name]

    k = len(eigs)
    X = self.rng().standard_normal(size=(n, k)).astype(dtype)

    theta, U, iters = linalg.lobpcg_standard(fn, X, m, tol=0.0)

    # Given tolerance is zero all iters should be used.
    self.assertEqual(iters, m)

    # Evaluate in f64.
    as64 = functools.partial(np.array, dtype=np.float64)
    theta, eigs, U = (as64(x) for x in (theta, eigs, U))

    relerr = np.abs(theta - eigs) / eigs
    UTU = U.T.dot(U)

    tol = np.sqrt(jnp.finfo(dtype).eps) * 100
    if example_name == 'ring laplacian':
      tol = 1e-2

    for i in range(k):
      self.assertLessEqual(
          relerr[i], tol,
          msg=f'eigenvalue {i} (actual {theta[i]} expected {eigs[i]})')
      self.assertAllClose(UTU[i, i], 1.0, rtol=tol)
      UTU[i, i] = 0
      self.assertArraysAllClose(UTU[i], np.zeros_like(UTU[i]), atol=tol)

    self._possibly_plot(fn, eigs, X, m, 'callable_' + example_name)


class F32LobpcgTest(LobpcgTest):

  def setUp(self):
    # TODO(phawkins): investigate this failure
    if jtu.test_device_matches(["gpu"]):
      raise unittest.SkipTest("Test is failing on CUDA gpus")
    super().setUp()

  def testLobpcgValidatesArguments(self):
    A, _ = _concrete_generators(np.float32)['id'](100, 10)
    X = self.rng().standard_normal(size=(100, 10)).astype(np.float32)

    with self.assertRaisesRegex(ValueError, 'search dim > 0'):
      linalg.lobpcg_standard(A, X[:,:0])

    with self.assertRaisesRegex(ValueError, 'A, X must have same dtypes'):
      linalg.lobpcg_standard(
          lambda x: jnp.array(A).dot(x).astype(jnp.float16), X)

    with self.assertRaisesRegex(ValueError, r'A must be \(100, 100\)'):
      linalg.lobpcg_standard(A[:60, :], X)

    with self.assertRaisesRegex(ValueError, r'search dim \* 5 < matrix dim'):
      linalg.lobpcg_standard(A[:50, :50], X[:50])

  @parameterized.named_parameters(_make_concrete_cases(f64=False))
  @jtu.skip_on_devices("gpu")
  def testLobpcgConsistencyF32(self, matrix_name, n, k, m, tol):
    self.checkLobpcgConsistency(matrix_name, n, k, m, tol, jnp.float32)

  @parameterized.named_parameters(_make_concrete_cases(f64=False))
  def testLobpcgMonotonicityF32(self, matrix_name, n, k, m, tol):
    self.checkLobpcgMonotonicity(matrix_name, n, k, m, tol, jnp.float32)

  @parameterized.named_parameters(_make_callable_cases(f64=False))
  def testCallableMatricesF32(self, matrix_name):
    self.checkApproxEigs(matrix_name, jnp.float32)


@jtu.with_config(jax_enable_x64=True)
class F64LobpcgTest(LobpcgTest):

  def setUp(self):
    # TODO(phawkins): investigate this failure
    if jtu.test_device_matches(["gpu"]):
      raise unittest.SkipTest("Test is failing on CUDA gpus")
    super().setUp()

  @parameterized.named_parameters(_make_concrete_cases(f64=True))
  @jtu.skip_on_devices("tpu", "gpu")
  def testLobpcgConsistencyF64(self, matrix_name, n, k, m, tol):
    self.checkLobpcgConsistency(matrix_name, n, k, m, tol, jnp.float64)

  @parameterized.named_parameters(_make_concrete_cases(f64=True))
  @jtu.skip_on_devices("tpu", "gpu")
  def testLobpcgMonotonicityF64(self, matrix_name, n, k, m, tol):
    self.checkLobpcgMonotonicity(matrix_name, n, k, m, tol, jnp.float64)

  @parameterized.named_parameters(_make_callable_cases(f64=True))
  @jtu.skip_on_devices("tpu", "gpu")
  def testCallableMatricesF64(self, matrix_name):
    self.checkApproxEigs(matrix_name, jnp.float64)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
