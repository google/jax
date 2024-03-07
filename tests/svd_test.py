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
# limitations under the License

"""Tests for the library of QDWH-based singular value decomposition."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as osp_linalg
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import svd

from absl.testing import absltest


config.parse_flags_with_absl()
_JAX_ENABLE_X64 = config.enable_x64.value

# Input matrix data type for SvdTest.
_SVD_TEST_DTYPE = np.float64 if _JAX_ENABLE_X64 else np.float32

# Machine epsilon used by SvdTest.
_SVD_TEST_EPS = jnp.finfo(_SVD_TEST_DTYPE).eps

# SvdTest relative tolerance.
_SVD_RTOL = 1E-6 if _JAX_ENABLE_X64 else 1E-2

_MAX_LOG_CONDITION_NUM = 9 if _JAX_ENABLE_X64 else 4


@jtu.with_config(jax_numpy_rank_promotion='allow')
class SvdTest(jtu.JaxTestCase):

  @jtu.sample_product(
      shape=[(4, 5), (3, 4, 5), (2, 3, 4, 5)],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision('float32')
  def testSvdvals(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_fun = jax.numpy.linalg.svdvals
    if jtu.numpy_version() < (2, 0, 0):
      np_fun = lambda x: np.linalg.svd(x, compute_uv=False)
    else:
      np_fun = np.linalg.svdvals
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=_SVD_RTOL, atol=1E-5)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=_SVD_RTOL)

  @jtu.sample_product(
    [dict(m=m, n=n) for m, n in zip([2, 8, 10, 20], [4, 6, 10, 18])],
    log_cond=np.linspace(1, _MAX_LOG_CONDITION_NUM, 4),
    full_matrices=[True, False],
  )
  def testSvdWithRectangularInput(self, m, n, log_cond, full_matrices):
    """Tests SVD with rectangular input."""
    with jax.default_matmul_precision('float32'):
      a = np.random.uniform(
          low=0.3, high=0.9, size=(m, n)).astype(_SVD_TEST_DTYPE)
      u, s, v = osp_linalg.svd(a, full_matrices=False)
      cond = 10**log_cond
      s = jnp.linspace(cond, 1, min(m, n))
      a = (u * s) @ v
      a = a.astype(complex) * (1 + 1j)

      osp_linalg_fn = functools.partial(
          osp_linalg.svd, full_matrices=full_matrices)
      actual_u, actual_s, actual_v = svd.svd(a, full_matrices=full_matrices)

      k = min(m, n)
      if m > n:
        unitary_u = jnp.real(actual_u.T.conj() @ actual_u)
        unitary_v = jnp.real(actual_v.T.conj() @ actual_v)
        unitary_u_size = m if full_matrices else k
        unitary_v_size = k
      else:
        unitary_u = jnp.real(actual_u @ actual_u.T.conj())
        unitary_v = jnp.real(actual_v @ actual_v.T.conj())
        unitary_u_size = k
        unitary_v_size = n if full_matrices else k

      _, expected_s, _ = osp_linalg_fn(a)

      svd_fn = lambda a: svd.svd(a, full_matrices=full_matrices)
      args_maker = lambda: [a]

      with self.subTest('Test JIT compatibility'):
        self._CompileAndCheck(svd_fn, args_maker)

      with self.subTest('Test unitary u.'):
        self.assertAllClose(np.eye(unitary_u_size), unitary_u, rtol=_SVD_RTOL,
                            atol=2E-3)

      with self.subTest('Test unitary v.'):
        self.assertAllClose(np.eye(unitary_v_size), unitary_v, rtol=_SVD_RTOL,
                            atol=2E-3)

      with self.subTest('Test s.'):
        self.assertAllClose(
            expected_s, jnp.real(actual_s), rtol=_SVD_RTOL, atol=1E-6)

  @jtu.sample_product(
    [dict(m=m, n=n) for m, n in zip([50, 6], [3, 60])],
  )
  def testSvdWithSkinnyTallInput(self, m, n):
    """Tests SVD with skinny and tall input."""
    # Generates a skinny and tall input
    with jax.default_matmul_precision('float32'):
      np.random.seed(1235)
      a = np.random.randn(m, n).astype(_SVD_TEST_DTYPE)
      u, s, v = svd.svd(a, full_matrices=False, hermitian=False)

      relative_diff = np.linalg.norm(a - (u * s) @ v) / np.linalg.norm(a)

      np.testing.assert_almost_equal(relative_diff, 1E-6, decimal=6)

  @jtu.sample_product(
    [dict(m=m, r=r) for m, r in zip([8, 8, 8, 10], [3, 5, 7, 9])],
    log_cond=np.linspace(1, 3, 3),
  )
  def testSvdWithOnRankDeficientInput(self, m, r, log_cond):
    """Tests SVD with rank-deficient input."""
    with jax.default_matmul_precision('float32'):
      a = jnp.triu(jnp.ones((m, m))).astype(_SVD_TEST_DTYPE)

      # Generates a rank-deficient input.
      u, s, v = jnp.linalg.svd(a, full_matrices=False)
      cond = 10**log_cond
      s = jnp.linspace(cond, 1, m)
      s = s.at[r:m].set(0)
      a = (u * s) @ v

      with jax.default_matmul_precision('float32'):
        u, s, v = svd.svd(a, full_matrices=False, hermitian=False)
      diff = np.linalg.norm(a - (u * s) @ v)

      np.testing.assert_almost_equal(diff, 1E-4, decimal=2)

  @jtu.sample_product(
      [dict(m=m, r=r) for m, r in zip([8, 8, 8, 10], [3, 5, 7, 9])],
  )
  def testSvdWithOnRankDeficientInputZeroColumns(self, m, r):
    """Tests SVD with rank-deficient input."""
    with jax.default_matmul_precision('float32'):
      np.random.seed(1235)
      a = np.random.randn(m, m).astype(_SVD_TEST_DTYPE)
      d = np.ones(m).astype(_SVD_TEST_DTYPE)
      d[r:m] = 0
      a = a @ np.diag(d)

      with jax.default_matmul_precision('float32'):
        u, s, v = svd.svd(a, full_matrices=True, hermitian=False)
      diff = np.linalg.norm(a - (u * s) @ v)
      np.testing.assert_almost_equal(diff, 1e-4, decimal=2)
      # Check that u and v are orthogonal.
      self.assertAllClose(u.T.conj() @ u, np.eye(m), atol=10 * _SVD_TEST_EPS)
      self.assertAllClose(v.T.conj() @ v, np.eye(m), atol=10 * _SVD_TEST_EPS)

  @jtu.sample_product(
    [dict(m=m, n=n) for m, n in zip([2, 8, 10, 20], [4, 6, 10, 18])],
    log_cond=np.linspace(1, _MAX_LOG_CONDITION_NUM, 4),
    full_matrices=[True, False],
  )
  def testSingularValues(self, m, n, log_cond, full_matrices):
    """Tests singular values."""
    with jax.default_matmul_precision('float32'):
      a = np.random.uniform(
          low=0.3, high=0.9, size=(m, n)).astype(_SVD_TEST_DTYPE)
      u, s, v = osp_linalg.svd(a, full_matrices=False)
      cond = 10**log_cond
      s = np.linspace(cond, 1, min(m, n))
      a = (u * s) @ v
      a = a + 1j * a

      # Only computes singular values.
      compute_uv = False

      osp_linalg_fn = functools.partial(
          osp_linalg.svd, full_matrices=full_matrices, compute_uv=compute_uv)
      actual_s = svd.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

      expected_s = osp_linalg_fn(a)

      svd_fn = lambda a: svd.svd(a, full_matrices=full_matrices)
      args_maker = lambda: [a]

      with self.subTest('Test JIT compatibility'):
        self._CompileAndCheck(svd_fn, args_maker)

      with self.subTest('Test s.'):
        self.assertAllClose(expected_s, actual_s, rtol=_SVD_RTOL, atol=1E-6)

      with self.subTest('Test non-increasing order.'):
        # Computes `actual_diff[i] = s[i+1] - s[i]`.
        actual_diff = jnp.diff(actual_s, append=0)
        np.testing.assert_array_less(actual_diff, np.zeros_like(actual_diff))

  @jtu.sample_product(
      [dict(m=m, n=n) for m, n in zip([2, 4, 8], [4, 4, 6])],
      full_matrices=[True, False],
      compute_uv=[True, False],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def testSvdAllZero(self, m, n, full_matrices, compute_uv, dtype):
    """Tests SVD on matrix of all zeros, +/-infinity or NaN."""
    osp_fun = functools.partial(
        osp_linalg.svd, full_matrices=full_matrices, compute_uv=compute_uv
    )
    lax_fun = functools.partial(
        svd.svd, full_matrices=full_matrices, compute_uv=compute_uv
    )
    args_maker_svd = lambda: [jnp.zeros((m, n), dtype=dtype)]
    self._CheckAgainstNumpy(osp_fun, lax_fun, args_maker_svd)
    self._CompileAndCheck(lax_fun, args_maker_svd)

  @jtu.sample_product(
      [dict(m=m, n=n) for m, n in zip([2, 4, 8], [4, 4, 6])],
      fill_value=[-np.inf, np.inf, np.nan],
      full_matrices=[True, False],
      compute_uv=[True, False],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def testSvdNonFiniteValues(
      self, m, n, fill_value, full_matrices, compute_uv, dtype
  ):
    """Tests SVD on matrix of all zeros, +/-infinity or NaN."""
    lax_fun = functools.partial(
        svd.svd, full_matrices=full_matrices, compute_uv=compute_uv
    )
    args_maker_svd = lambda: [
        jnp.full((m, n), fill_value=fill_value, dtype=dtype)
    ]
    result = lax_fun(args_maker_svd()[0])
    for r in result:
      self.assertTrue(jnp.all(jnp.isnan(r)))
    self._CompileAndCheck(lax_fun, args_maker_svd)

  @jtu.sample_product(
    [dict(m=m, n=n, r=r, c=c)
     for m, n, r, c in zip([2, 4, 8], [4, 4, 6], [1, 0, 1], [1, 0, 1])],
    dtype=jtu.dtypes.floating,
  )
  def testSvdOnTinyElement(self, m, n, r, c, dtype):
    """Tests SVD on matrix of zeros and close-to-zero entries."""
    a = jnp.zeros((m, n), dtype=dtype)
    tiny_element = jnp.finfo(a.dtype).tiny
    a = a.at[r, c].set(tiny_element)

    @jax.jit
    def lax_fun(a):
      return svd.svd(a, full_matrices=False, compute_uv=False, hermitian=False)

    actual_s = lax_fun(a)

    k = min(m, n)
    expected_s = np.zeros((k,), dtype=dtype)
    expected_s[0] = tiny_element

    self.assertAllClose(expected_s, jnp.real(actual_s), rtol=_SVD_RTOL,
                        atol=1E-6)

  @jtu.sample_product(
      start=[0, 1, 64, 126, 127],
      end=[1, 2, 65, 127, 128],
  )
  @jtu.run_on_devices('tpu')  # TODO(rmlarsen: enable on other devices)
  def testSvdSubsetByIndex(self, start, end):
    if start >= end:
      return
    dtype = np.float32
    m = 256
    n = 128
    rng = jtu.rand_default(self.rng())
    tol = np.maximum(n, 80) * np.finfo(dtype).eps
    args_maker = lambda: [rng((m, n), dtype)]
    subset_by_index = (start, end)
    k = end - start
    (a,) = args_maker()

    u, s, vt = jnp.linalg.svd(
        a, full_matrices=False, subset_by_index=subset_by_index
    )
    self.assertEqual(u.shape, (m, k))
    self.assertEqual(s.shape, (k,))
    self.assertEqual(vt.shape, (k, n))

    with jax.numpy_rank_promotion('allow'):
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, vt.T) - u * s), tol * np.linalg.norm(a)
      )

    # Test that we get the approximately the same singular values when
    # slicing the full SVD.
    _, full_s, _ = jnp.linalg.svd(a, full_matrices=False)
    s_slice = full_s[start:end]
    self.assertAllClose(s_slice, s, atol=tol, rtol=tol)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
