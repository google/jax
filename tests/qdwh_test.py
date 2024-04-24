# Copyright 2021 The JAX Authors.
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
# limitations under the License

"""Tests for the library of QDWH-based polar decomposition."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as osp_linalg
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import qdwh

from absl.testing import absltest


config.parse_flags_with_absl()
_JAX_ENABLE_X64_QDWH = config.enable_x64.value

# Input matrix data type for QdwhTest.
_QDWH_TEST_DTYPE = np.float64 if _JAX_ENABLE_X64_QDWH else np.float32

# Machine epsilon used by QdwhTest.
_QDWH_TEST_EPS = jnp.finfo(_QDWH_TEST_DTYPE).eps

# Largest log10 value of condition numbers used by QdwhTest.
_MAX_LOG_CONDITION_NUM = np.log10(int(1 / _QDWH_TEST_EPS))


def _check_symmetry(x: jax.Array) -> bool:
  """Check if the array is symmetric."""
  m, n = x.shape
  eps = jnp.finfo(x.dtype).eps
  tol = 50.0 * eps
  is_hermitian = False
  if m == n:
    if np.linalg.norm(x - x.T.conj()) / np.linalg.norm(x) < tol:
      is_hermitian = True

  return is_hermitian

def _compute_relative_diff(actual, expected):
  """Computes relative difference between two matrices."""
  return np.linalg.norm(actual - expected) / np.linalg.norm(expected)

_dot = functools.partial(jnp.dot, precision="highest")


class QdwhTest(jtu.JaxTestCase):

  @jtu.sample_product(
    [dict(m=m, n=n) for m, n in [(8, 6), (10, 10), (20, 18)]],
    log_cond=np.linspace(1, _MAX_LOG_CONDITION_NUM, 4),
  )
  def testQdwhUnconvergedAfterMaxNumberIterations(
      self, m, n, log_cond):
    """Tests unconvergence after maximum number of iterations."""
    a = jnp.triu(jnp.ones((m, n)))
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.expand_dims(jnp.linspace(cond, 1, min(m, n)), range(u.ndim - 1))
    with jax.numpy_dtype_promotion('standard'):
      a = (u * s) @ v
    is_hermitian = _check_symmetry(a)
    max_iterations = 2

    _, _, actual_num_iterations, is_converged = qdwh.qdwh(
        a, is_hermitian=is_hermitian, max_iterations=max_iterations)

    with self.subTest('Number of iterations.'):
      self.assertEqual(max_iterations, actual_num_iterations)

    with self.subTest('Converged.'):
      self.assertFalse(is_converged)

  @jtu.sample_product(
    [dict(m=m, n=n) for m, n in [(8, 6), (10, 10), (20, 18)]],
    log_cond=np.linspace(1, _MAX_LOG_CONDITION_NUM, 4),
  )
  def testQdwhWithUpperTriangularInputAllOnes(self, m, n, log_cond):
    """Tests qdwh with upper triangular input of all ones."""
    a = jnp.triu(jnp.ones((m, n))).astype(_QDWH_TEST_DTYPE)
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.expand_dims(jnp.linspace(cond, 1, min(m, n)), range(u.ndim - 1))
    a = (u * s) @ v
    is_hermitian = _check_symmetry(a)
    max_iterations = 10

    actual_u, actual_h, _, _ = qdwh.qdwh(a, is_hermitian=is_hermitian,
                                         max_iterations=max_iterations)
    expected_u, expected_h = osp_linalg.polar(a)

    # Sets the test tolerance.
    rtol = 1E6 * _QDWH_TEST_EPS

    with self.subTest('Test u.'):
      relative_diff_u = _compute_relative_diff(actual_u, expected_u)
      np.testing.assert_almost_equal(relative_diff_u, 1E-6, decimal=5)

    with self.subTest('Test h.'):
      relative_diff_h = _compute_relative_diff(actual_h, expected_h)
      np.testing.assert_almost_equal(relative_diff_h, 1E-6, decimal=5)

    with self.subTest('Test u.dot(h).'):
      a_round_trip = _dot(actual_u, actual_h)
      relative_diff_a = _compute_relative_diff(a_round_trip, a)
      np.testing.assert_almost_equal(relative_diff_a, 1E-6, decimal=5)

    with self.subTest('Test orthogonality.'):
      actual_results = _dot(actual_u.T, actual_u)
      expected_results = np.eye(n)
      self.assertAllClose(
          actual_results, expected_results, rtol=rtol, atol=1E-5)

  @jtu.sample_product(
    [dict(m=m, n=n) for m, n in [(6, 6), (8, 4)]],
    padding=(None, (3, 2)),
    log_cond=np.linspace(1, 4, 4),
  )
  def testQdwhWithRandomMatrix(self, m, n, log_cond, padding):
    """Tests qdwh with random input."""
    rng = jtu.rand_uniform(self.rng(), low=0.3, high=0.9)
    a = rng((m, n), _QDWH_TEST_DTYPE)
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    cond = 10**log_cond
    s = jnp.expand_dims(jnp.linspace(cond, 1, min(m, n)), range(u.ndim - 1))
    a = (u * s) @ v
    is_hermitian = _check_symmetry(a)
    max_iterations = 10

    def lsp_linalg_fn(a):
      if padding is not None:
        pm, pn = padding
        a = jnp.pad(a, [(0, pm), (0, pn)], constant_values=jnp.nan)
      u, h, _, _ = qdwh.qdwh(
          a, is_hermitian=is_hermitian, max_iterations=max_iterations,
          dynamic_shape=(m, n) if padding else None)
      if padding is not None:
        u = u[:m, :n]
        h = h[:n, :n]
      return u, h

    args_maker = lambda: [a]

    # Sets the test tolerance.
    rtol = 1E6 * _QDWH_TEST_EPS

    with self.subTest('Test JIT compatibility'):
      self._CompileAndCheck(lsp_linalg_fn, args_maker)

    with self.subTest('Test against numpy.'):
      self._CheckAgainstNumpy(osp_linalg.polar, lsp_linalg_fn, args_maker,
                              rtol=rtol, atol=1E-3)

  @jtu.sample_product(
    [dict(m=m, n=n, r=r) for m, n, r in [(10, 10, 8), (8, 8, 7), (12, 8, 5)]],
    log_cond=np.linspace(1, 4, 4),
  )
  def testQdwhOnRankDeficientInput(self, m, n, r, log_cond):
    """Tests qdwh on rank-deficient input."""
    a = np.triu(np.ones((m, n))).astype(_QDWH_TEST_DTYPE)

    # Generates a rank-deficient input.
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    s = 10**jnp.linspace(log_cond, 0, min(m, n))
    s = jnp.expand_dims(s.at[r:].set(0), range(u.ndim - 1))
    a = (u * s) @ vh

    actual_u, actual_h, _, _ = qdwh.qdwh(a, is_hermitian=_check_symmetry(a))
    _, expected_h = osp_linalg.polar(a)

    with self.subTest('Test h.'):
      relative_diff_h = _compute_relative_diff(actual_h, expected_h)
      np.testing.assert_almost_equal(relative_diff_h, 1E-6, decimal=5)

    with self.subTest('Test u.dot(h).'):
      a_round_trip = _dot(actual_u, actual_h)
      relative_diff_a = _compute_relative_diff(a_round_trip, a)
      np.testing.assert_almost_equal(relative_diff_a, 1E-6, decimal=5)

    # QDWH gives U_p = U Σₖ V* for input A with SVD A = U Σ V*. For full rank
    # input, we expect convergence Σₖ → I, giving the correct polar factor
    # U_p = U V*. Zero singular values stay at 0 in exact arithmetic, but can
    # end up anywhere in [0, 1] as a result of rounding errors---in particular,
    # we do not generally expect convergence to 1. As a result, we can only
    # expect (U_p V_r) to be orthogonal, where V_r are the columns of V
    # corresponding to nonzero singular values.
    with self.subTest('Test orthogonality.'):
      vr = vh.conj().T[:, :r]
      uvr = _dot(actual_u, vr)
      actual_results = _dot(uvr.T.conj(), uvr)
      expected_results = np.eye(r)
      self.assertAllClose(
          actual_results, expected_results, rtol=_QDWH_TEST_EPS, atol=1e-6
      )

  @jtu.sample_product(
    [dict(m=m, n=n, r=r, c=c) for m, n, r, c in [(4, 3, 1, 1), (5, 2, 0, 0)]],
    dtype=jtu.dtypes.floating,
  )
  def testQdwhWithTinyElement(self, m, n, r, c, dtype):
    """Tests qdwh on matrix with zeros and close-to-zero entries."""
    a = jnp.zeros((m, n), dtype=dtype)
    tiny_elem = jnp.finfo(a.dtype).tiny
    a = a.at[r, c].set(tiny_elem)

    is_hermitian = _check_symmetry(a)
    max_iterations = 10

    @jax.jit
    def lsp_linalg_fn(a):
      u, h, _, _ = qdwh.qdwh(
          a, is_hermitian=is_hermitian, max_iterations=max_iterations)
      return u, h

    actual_u, actual_h = lsp_linalg_fn(a)

    expected_u = jnp.zeros((m, n), dtype=dtype)
    expected_u = expected_u.at[r, c].set(1.0)
    with self.subTest('Test u.'):
      np.testing.assert_array_equal(expected_u, actual_u)

    expected_h = jnp.zeros((n, n), dtype=dtype)
    expected_h = expected_h.at[r, c].set(tiny_elem)
    with self.subTest('Test h.'):
      np.testing.assert_array_equal(expected_h, actual_h)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
