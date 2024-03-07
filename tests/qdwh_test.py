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

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import qdwh
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()

float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex


def _compute_relative_normwise_diff(actual, expected):
  """Computes relative difference between two matrices."""
  return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


_dot = functools.partial(jnp.dot, precision='highest')


class QdwhTest(jtu.JaxTestCase):

  def _testReconstruction(self, a, u, h, tol):
    """Tests that a = u*p."""
    with self.subTest('Test reconstruction'):
      diff = _compute_relative_normwise_diff(_dot(u, h), a)
      self.assertLessEqual(diff, tol)

  def _testUnitary(self, u, tol):
    """Tests that u is unitary."""
    with self.subTest('Test unitary'):
      m, n = u.shape
      self.assertAllClose(
          _dot(u.conj().T, u), np.eye(n, dtype=u.dtype), atol=tol, rtol=tol
      )

  def _testHermitian(self, h, tol):
    """Tests that h is Hermitian."""
    with self.subTest('Test hermitian'):
      self.assertAllClose(h, h.conj().T, atol=tol, rtol=tol)

  def _testPolarDecomposition(self, a, u, h, tol):
    """Tests that u*h is the polar decomposition of a"""
    self._testReconstruction(a, u, h, tol)
    self._testUnitary(u, tol)
    self._testHermitian(h, tol)

  def _testQdwh(self, a, dynamic_shape=None):
    """Computes the polar decomposition and tests its basic properties."""
    eps = jnp.finfo(a.dtype).eps
    u, h, iters, conv = qdwh.qdwh(a, dynamic_shape=dynamic_shape)
    tol = 13 * eps
    if dynamic_shape is not None:
      m, n = dynamic_shape
      a = a[:m, :n]
      u = u[:m, :n]
      h = h[:n, :n]
    self._testPolarDecomposition(a, u, h, tol=tol)

  @jtu.sample_product(
      shape=[(8, 6), (10, 10), (20, 18)],
      dtype=float_types + complex_types,
  )
  def testQdwhWithUpperTriangularInputAllOnes(self, shape, dtype):
    """Tests qdwh with upper triangular input of all ones."""
    eps = jnp.finfo(dtype).eps
    m, n = shape
    a = jnp.triu(jnp.ones((m, n))).astype(dtype)
    self._testQdwh(a)

  @jtu.sample_product(
      shape=[(2, 2), (5, 5), (8, 5), (10, 10)],
      dtype=float_types + complex_types,
  )
  def testQdwhWithDynamicShape(self, shape, dtype):
    """Tests qdwh with dynamic shapes."""
    rng = jtu.rand_uniform(self.rng())
    a = rng((10, 10), dtype)
    self._testQdwh(a, dynamic_shape=shape)

  @jtu.sample_product(
      shape=[(8, 6), (10, 10), (20, 18), (300, 300)],
      log_cond=np.linspace(0, 1, 4),
      dtype=float_types + complex_types,
  )
  def testQdwhWithRandomMatrix(self, shape, log_cond, dtype):
    """Tests qdwh with upper triangular input of all ones."""
    eps = jnp.finfo(dtype).eps
    m, n = shape
    max_cond = np.log10(1.0 / eps)
    log_cond = log_cond * max_cond
    cond = 10**log_cond

    # Generates input matrix with prescribed condition number.
    rng = jtu.rand_uniform(self.rng())
    a = rng((m, n), dtype)
    u, _, v = jnp.linalg.svd(a, full_matrices=False)
    s = jnp.expand_dims(jnp.linspace(cond, 1, min(m, n)), range(u.ndim - 1))
    a = (u * s.astype(u.dtype)) @ v
    self._testQdwh(a)

  @jtu.sample_product(
      [dict(m=m, n=n) for m, n in [(6, 6), (8, 4)]],
      padding=(None, (3, 2)),
      dtype=float_types + complex_types,
  )
  def testQdwhJitCompatibility(self, m, n, padding, dtype):
    """Tests JIT compilation of QDWH with and without dynamic shape."""
    rng = jtu.rand_uniform(self.rng())
    a = rng((m, n), dtype)
    def lsp_linalg_fn(a):
      if padding is not None:
        pm, pn = padding
        a = jnp.pad(a, [(0, pm), (0, pn)], constant_values=jnp.nan)
      u, h, _, _ = qdwh.qdwh(a, dynamic_shape=(m, n) if padding else None)
      if padding is not None:
        u = u[:m, :n]
        h = h[:n, :n]
      return u, h

    args_maker = lambda: [a]
    with self.subTest('Test JIT compatibility'):
      self._CompileAndCheck(lsp_linalg_fn, args_maker)

  @jtu.sample_product(
      [dict(m=m, n=n, r=r) for m, n, r in [(10, 10, 8), (8, 8, 7), (12, 8, 5)]],
      log_cond=np.linspace(0, 1, 4),
      dtype=float_types + complex_types,
  )
  def testQdwhOnRankDeficientInput(self, m, n, r, log_cond, dtype):
    """Tests qdwh on rank-deficient input."""
    eps = jnp.finfo(dtype).eps
    a = np.triu(np.ones((m, n))).astype(dtype)

    # Generates a rank-deficient input with prescribed condition number.
    max_cond = np.log10(1.0 / eps)
    log_cond = log_cond * max_cond
    u, _, vh = np.linalg.svd(a, full_matrices=False)
    s = 10**jnp.linspace(log_cond, 0, min(m, n))
    print(s)
    s = jnp.expand_dims(s.at[r:].set(0), range(u.ndim - 1))
    a = (u * s.astype(u.dtype)) @ vh

    actual_u, actual_h, _, _ = qdwh.qdwh(a)

    self._testHermitian(actual_h, 10 * eps)
    self._testReconstruction(a, actual_u, actual_h, 60 * eps)

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
      expected_results = np.eye(r, dtype=actual_u.dtype)
      self.assertAllClose(
          actual_results, expected_results, atol=25 * eps, rtol=25 * eps
      )

  @jtu.sample_product(
      [dict(m=m, n=n, r=r, c=c) for m, n, r, c in [(4, 3, 1, 1), (5, 2, 0, 0)]],
      dtype=float_types + complex_types,
  )
  def testQdwhWithTinyElement(self, m, n, r, c, dtype):
    """Tests qdwh on matrix with zeros and close-to-zero entries."""
    a = jnp.zeros((m, n), dtype=dtype)
    one = dtype(1.0)
    tiny_elem = dtype(jnp.finfo(a.dtype).tiny)
    a = a.at[r, c].set(tiny_elem)

    @jax.jit
    def lsp_linalg_fn(a):
      u, h, _, _ = qdwh.qdwh(a)
      return u, h

    actual_u, actual_h = lsp_linalg_fn(a)

    expected_u = jnp.zeros((m, n), dtype=dtype)
    expected_u = expected_u.at[r, c].set(one)
    with self.subTest('Test u.'):
      np.testing.assert_array_equal(expected_u, actual_u)

    expected_h = jnp.zeros((n, n), dtype=dtype)
    expected_h = expected_h.at[r, c].set(tiny_elem)
    with self.subTest('Test h.'):
      np.testing.assert_array_equal(expected_h, actual_h)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
