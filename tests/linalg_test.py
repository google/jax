# Copyright 2018 The JAX Authors.
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

"""Tests for the LAPAX linear algebra module."""

from functools import partial
import itertools

import numpy as np
import scipy
import scipy.linalg
import scipy as osp

from absl.testing import absltest, parameterized

import jax
from jax import jit, grad, jvp, vmap
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax._src import config
from jax._src.lax import linalg as lax_linalg
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.numpy.util import promote_dtypes_inexact

config.parse_flags_with_absl()

scipy_version = jtu.parse_version(scipy.version.version)

T = lambda x: np.swapaxes(x, -1, -2)

float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex
int_types = jtu.dtypes.all_integer

def _is_required_cuda_version_satisfied(cuda_version):
  version = xla_bridge.get_backend().platform_version
  if version == "<unknown>" or version.split()[0] == "rocm":
    return False
  else:
    return int(version.split()[-1]) >= cuda_version

class NumpyLinalgTest(jtu.JaxTestCase):

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (2, 5, 5), (200, 200), (1000, 0, 0)],
    dtype=float_types + complex_types,
    upper=[True, False]
  )
  def testCholesky(self, shape, dtype, upper):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      factor_shape = shape[:-1] + (2 * shape[-1],)
      a = rng(factor_shape, dtype)
      return [np.matmul(a, jnp.conj(T(a)))]

    jnp_fun = partial(jnp.linalg.cholesky, upper=upper)

    def np_fun(x, upper=upper):
      # Upper argument added in NumPy 2.0.0
      if jtu.numpy_version() >= (2, 0, 0):
        return np.linalg.cholesky(x, upper=upper)
      result = np.linalg.cholesky(x)
      if upper:
        axes = list(range(x.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return np.transpose(result, axes).conj()
      return result

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            tol=1e-3)
    self._CompileAndCheck(jnp_fun, args_maker)

    if jnp.finfo(dtype).bits == 64:
      jtu.check_grads(jnp.linalg.cholesky, args_maker(), order=2)

  def testCholeskyGradPrecision(self):
    rng = jtu.rand_default(self.rng())
    a = rng((3, 3), np.float32)
    a = np.dot(a, a.T)
    jtu.assert_dot_precision(
        lax.Precision.HIGHEST, partial(jvp, jnp.linalg.cholesky), (a,), (a,))

  @jtu.sample_product(
    n=[0, 2, 3, 4, 5, 25],  # TODO(mattjj): complex64 unstable on large sizes?
    dtype=float_types + complex_types,
  )
  def testDet(self, n, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((n, n), dtype)]

    self._CheckAgainstNumpy(np.linalg.det, jnp.linalg.det, args_maker, tol=1e-3)
    self._CompileAndCheck(jnp.linalg.det, args_maker,
                          rtol={np.float64: 1e-13, np.complex128: 1e-13})

  def testDetOfSingularMatrix(self):
    x = jnp.array([[-1., 3./2], [2./3, -1.]], dtype=np.float32)
    self.assertAllClose(np.float32(0), jsp.linalg.det(x))

  @jtu.sample_product(
    shape=[(1, 1), (3, 3), (2, 4, 4)],
    dtype=float_types,
  )
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  @jtu.skip_on_devices("tpu")
  def testDetGrad(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    jtu.check_grads(jnp.linalg.det, (a,), 2, atol=1e-1, rtol=1e-1)
    # make sure there are no NaNs when a matrix is zero
    if len(shape) == 2:
      jtu.check_grads(
        jnp.linalg.det, (jnp.zeros_like(a),), 1, atol=1e-1, rtol=1e-1)
    else:
      a[0] = 0
      jtu.check_grads(jnp.linalg.det, (a,), 1, atol=1e-1, rtol=1e-1)

  def testDetGradIssue6121(self):
    f = lambda x: jnp.linalg.det(x).sum()
    x = jnp.ones((16, 1, 1))
    jax.grad(f)(x)
    jtu.check_grads(f, (x,), 2, atol=1e-1, rtol=1e-1)

  def testDetGradOfSingularMatrixCorank1(self):
    # Rank 2 matrix with nonzero gradient
    a = jnp.array([[ 50, -30,  45],
                  [-30,  90, -81],
                  [ 45, -81,  81]], dtype=jnp.float32)
    jtu.check_grads(jnp.linalg.det, (a,), 1, atol=1e-1, rtol=1e-1)

  # TODO(phawkins): Test sometimes produces NaNs on TPU.
  @jtu.skip_on_devices("tpu")
  def testDetGradOfSingularMatrixCorank2(self):
    # Rank 1 matrix with zero gradient
    b = jnp.array([[ 36, -42,  18],
                  [-42,  49, -21],
                  [ 18, -21,   9]], dtype=jnp.float32)
    jtu.check_grads(jnp.linalg.det, (b,), 1, atol=1e-1, rtol=1e-1, eps=1e-1)

  @jtu.sample_product(
    m=[1, 5, 7, 23],
    nq=zip([2, 4, 6, 36], [(1, 2), (2, 2), (1, 2, 3), (3, 3, 1, 4)]),
    dtype=float_types,
  )
  def testTensorsolve(self, m, nq, dtype):
    rng = jtu.rand_default(self.rng())

    # According to numpy docs the shapes are as follows:
    # Coefficient tensor (a), of shape b.shape + Q.
    # And prod(Q) == prod(b.shape)
    # Therefore, n = prod(q)
    n, q = nq
    b_shape = (n, m)
    # To accomplish prod(Q) == prod(b.shape) we append the m extra dim
    # to Q shape
    Q = q + (m,)
    args_maker = lambda: [
        rng(b_shape + Q, dtype), # = a
        rng(b_shape, dtype)]     # = b
    a, b = args_maker()
    result = jnp.linalg.tensorsolve(*args_maker())
    self.assertEqual(result.shape, Q)

    self._CheckAgainstNumpy(np.linalg.tensorsolve,
                            jnp.linalg.tensorsolve, args_maker,
                            tol={np.float32: 1e-2, np.float64: 1e-3})
    self._CompileAndCheck(jnp.linalg.tensorsolve,
                          args_maker,
                          rtol={np.float64: 1e-13})

  @jtu.sample_product(
    [dict(dtype=dtype, method=method)
     for dtype in float_types + complex_types
     for method in (["lu"] if jnp.issubdtype(dtype, jnp.complexfloating)
                     else ["lu", "qr"])
    ],
    shape=[(0, 0), (1, 1), (3, 3), (4, 4), (10, 10), (200, 200), (2, 2, 2),
           (2, 3, 3), (3, 2, 2)],
  )
  def testSlogdet(self, shape, dtype, method):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    slogdet = partial(jnp.linalg.slogdet, method=method)
    self._CheckAgainstNumpy(np.linalg.slogdet, slogdet, args_maker,
                            tol=1e-3)
    self._CompileAndCheck(slogdet, args_maker)

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (5, 5), (2, 7, 7)],
    dtype=float_types + complex_types,
  )
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testSlogdetGrad(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    jtu.check_grads(jnp.linalg.slogdet, (a,), 2, atol=1e-1, rtol=2e-1)

  def testIssue1213(self):
    for n in range(5):
      mat = jnp.array([np.diag(np.ones([5], dtype=np.float32))*(-.01)] * 2)
      args_maker = lambda: [mat]
      self._CheckAgainstNumpy(np.linalg.slogdet, jnp.linalg.slogdet, args_maker,
                              tol=1e-3)

  @jtu.sample_product(
    shape=[(0, 0), (4, 4), (5, 5), (50, 50), (2, 6, 6)],
    dtype=float_types + complex_types,
    compute_left_eigenvectors=[False, True],
    compute_right_eigenvectors=[False, True],
  )
  # TODO(phawkins): enable when there is an eigendecomposition implementation
  # for GPU/TPU.
  @jtu.run_on_devices("cpu")
  def testEig(self, shape, dtype, compute_left_eigenvectors,
              compute_right_eigenvectors):
    rng = jtu.rand_default(self.rng())
    n = shape[-1]
    args_maker = lambda: [rng(shape, dtype)]

    # Norm, adjusted for dimension and type.
    def norm(x):
      norm = np.linalg.norm(x, axis=(-2, -1))
      return norm / ((n + 1) * jnp.finfo(dtype).eps)

    def check_right_eigenvectors(a, w, vr):
      self.assertTrue(
        np.all(norm(np.matmul(a, vr) - w[..., None, :] * vr) < 100))

    def check_left_eigenvectors(a, w, vl):
      rank = len(a.shape)
      aH = jnp.conj(a.transpose(list(range(rank - 2)) + [rank - 1, rank - 2]))
      wC = jnp.conj(w)
      check_right_eigenvectors(aH, wC, vl)

    a, = args_maker()
    results = lax.linalg.eig(
        a, compute_left_eigenvectors=compute_left_eigenvectors,
        compute_right_eigenvectors=compute_right_eigenvectors)
    w = results[0]

    if compute_left_eigenvectors:
      check_left_eigenvectors(a, w, results[1])
    if compute_right_eigenvectors:
      check_right_eigenvectors(a, w, results[1 + compute_left_eigenvectors])

    self._CompileAndCheck(partial(jnp.linalg.eig), args_maker,
                          rtol=1e-3)

  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (50, 50), (2, 6, 6)],
    dtype=float_types + complex_types,
    compute_left_eigenvectors=[False, True],
    compute_right_eigenvectors=[False, True],
  )
  # TODO(phawkins): enable when there is an eigendecomposition implementation
  # for GPU/TPU.
  @jtu.run_on_devices("cpu")
  def testEigHandlesNanInputs(self, shape, dtype, compute_left_eigenvectors,
                              compute_right_eigenvectors):
    """Verifies that `eig` fails gracefully if given non-finite inputs."""
    a = jnp.full(shape, jnp.nan, dtype)
    results = lax.linalg.eig(
        a, compute_left_eigenvectors=compute_left_eigenvectors,
        compute_right_eigenvectors=compute_right_eigenvectors)
    for result in results:
      self.assertTrue(np.all(np.isnan(result)))


  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (8, 8), (7, 6, 6)],
    dtype=float_types + complex_types,
  )
  # TODO(phawkins): enable when there is an eigendecomposition implementation
  # for GPU/TPU.
  @jtu.run_on_devices("cpu")
  def testEigvalsGrad(self, shape, dtype):
    # This test sometimes fails for large matrices. I (@j-towns) suspect, but
    # haven't checked, that might be because of perturbations causing the
    # ordering of eigenvalues to change, which will trip up check_grads. So we
    # just test on small-ish matrices.
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    a, = args_maker()
    tol = 1e-4 if dtype in (np.float64, np.complex128) else 1e-1
    jtu.check_grads(lambda x: jnp.linalg.eigvals(x), (a,), order=1,
                    modes=['fwd', 'rev'], rtol=tol, atol=tol)

  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (50, 50)],
    dtype=float_types + complex_types,
  )
  # TODO: enable when there is an eigendecomposition implementation
  # for GPU/TPU.
  @jtu.run_on_devices("cpu")
  def testEigvals(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    a, = args_maker()
    w1, _ = jnp.linalg.eig(a)
    w2 = jnp.linalg.eigvals(a)
    self.assertAllClose(w1, w2, rtol={np.complex64: 1e-5, np.complex128: 1e-14})

  @jtu.run_on_devices("cpu")
  def testEigvalsInf(self):
    # https://github.com/google/jax/issues/2661
    x = jnp.array([[jnp.inf]])
    self.assertTrue(jnp.all(jnp.isnan(jnp.linalg.eigvals(x))))

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (5, 5)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testEigBatching(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    shape = (10,) + shape
    args = rng(shape, dtype)
    ws, vs = vmap(jnp.linalg.eig)(args)
    self.assertTrue(np.all(np.linalg.norm(
        np.matmul(args, vs) - ws[..., None, :] * vs) < 1e-3))

  @jtu.sample_product(
    n=[0, 4, 5, 50, 512],
    dtype=float_types + complex_types,
    lower=[True, False],
  )
  def testEigh(self, n, dtype, lower):
    rng = jtu.rand_default(self.rng())
    tol = 0.5 * np.maximum(n, 80) * np.finfo(dtype).eps
    args_maker = lambda: [rng((n, n), dtype)]

    uplo = "L" if lower else "U"

    a, = args_maker()
    a = (a + np.conj(a.T)) / 2
    w, v = jnp.linalg.eigh(np.tril(a) if lower else np.triu(a),
                           UPLO=uplo, symmetrize_input=False)
    w = w.astype(v.dtype)
    self.assertLessEqual(
        np.linalg.norm(np.eye(n) - np.matmul(np.conj(T(v)), v)), 4 * tol
    )
    with jax.numpy_rank_promotion('allow'):
      self.assertLessEqual(np.linalg.norm(np.matmul(a, v) - w * v),
                           tol * np.linalg.norm(a))

    self._CompileAndCheck(
        partial(jnp.linalg.eigh, UPLO=uplo), args_maker, rtol=tol
    )

  @jtu.sample_product(
      start=[0, 1, 63, 64, 65, 255],
      end=[1, 63, 64, 65, 256],
  )
  @jtu.run_on_devices("tpu")  # TODO(rmlarsen: enable on other devices)
  def testEighSubsetByIndex(self, start, end):
    if start >= end:
      return
    dtype = np.float32
    n = 256
    rng = jtu.rand_default(self.rng())
    tol = np.maximum(n, 80) * np.finfo(dtype).eps
    args_maker = lambda: [rng((n, n), dtype)]
    subset_by_index = (start, end)
    k = end - start
    (a,) = args_maker()
    a = (a + np.conj(a.T)) / 2

    v, w = lax.linalg.eigh(
        a, symmetrize_input=False, subset_by_index=subset_by_index
    )
    w = w.astype(v.dtype)

    self.assertEqual(v.shape, (n, k))
    self.assertEqual(w.shape, (k,))
    self.assertLessEqual(
        np.linalg.norm(np.eye(k) - np.matmul(np.conj(T(v)), v)), 3 * tol
    )
    with jax.numpy_rank_promotion("allow"):
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v), tol * np.linalg.norm(a)
      )

    self._CompileAndCheck(partial(jnp.linalg.eigh), args_maker, rtol=tol)

    # Compare eigenvalues against Numpy. We do not compare eigenvectors because
    # they are not uniquely defined, but the two checks above guarantee that
    # that they satisfy the conditions for being eigenvectors.
    w_np = np.linalg.eigvalsh(a)[subset_by_index[0] : subset_by_index[1]]
    self.assertAllClose(w_np, w, atol=tol, rtol=tol)

  def testEighZeroDiagonal(self):
    a = np.array([[0., -1., -1.,  1.],
                  [-1.,  0.,  1., -1.],
                  [-1.,  1.,  0., -1.],
                  [1., -1., -1.,  0.]], dtype=np.float32)
    w, v = jnp.linalg.eigh(a)
    w = w.astype(v.dtype)
    eps = jnp.finfo(a.dtype).eps
    with jax.numpy_rank_promotion('allow'):
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v), 2 * eps * np.linalg.norm(a)
      )

  def testEighTinyNorm(self):
    rng = jtu.rand_default(self.rng())
    a = rng((300, 300), dtype=np.float32)
    eps = jnp.finfo(a.dtype).eps
    a = eps * (a + np.conj(a.T))
    w, v = jnp.linalg.eigh(a)
    w = w.astype(v.dtype)
    with jax.numpy_rank_promotion("allow"):
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v), 80 * eps * np.linalg.norm(a)
      )

  @jtu.sample_product(
      rank=[1, 3, 299],
  )
  def testEighRankDeficient(self, rank):
    rng = jtu.rand_default(self.rng())
    eps = jnp.finfo(np.float32).eps
    a = rng((300, rank), dtype=np.float32)
    a = a @ np.conj(a.T)
    w, v = jnp.linalg.eigh(a)
    w = w.astype(v.dtype)
    with jax.numpy_rank_promotion("allow"):
      self.assertLessEqual(
          np.linalg.norm(np.matmul(a, v) - w * v),
          81 * eps * np.linalg.norm(a),
      )

  @jtu.sample_product(
    n=[0, 4, 5, 50, 512],
    dtype=float_types + complex_types,
    lower=[True, False],
  )
  def testEighIdentity(self, n, dtype, lower):
    tol = np.finfo(dtype).eps
    uplo = "L" if lower else "U"

    a = jnp.eye(n, dtype=dtype)
    w, v = jnp.linalg.eigh(a, UPLO=uplo, symmetrize_input=False)
    w = w.astype(v.dtype)
    self.assertLessEqual(
        np.linalg.norm(np.eye(n) - np.matmul(np.conj(T(v)), v)), tol
    )
    with jax.numpy_rank_promotion('allow'):
      self.assertLessEqual(np.linalg.norm(np.matmul(a, v) - w * v),
                           tol * np.linalg.norm(a))

  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (50, 50)],
    dtype=float_types + complex_types,
  )
  def testEigvalsh(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    n = shape[-1]
    def args_maker():
      a = rng((n, n), dtype)
      a = (a + np.conj(a.T)) / 2
      return [a]
    self._CheckAgainstNumpy(
        np.linalg.eigvalsh, jnp.linalg.eigvalsh, args_maker, tol=2e-5
    )

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (5, 5), (50, 50), (2, 10, 10)],
    dtype=float_types + complex_types,
    lower=[True, False],
  )
  def testEighGrad(self, shape, dtype, lower):
    rng = jtu.rand_default(self.rng())
    self.skipTest("Test fails with numeric errors.")
    uplo = "L" if lower else "U"
    a = rng(shape, dtype)
    a = (a + np.conj(T(a))) / 2
    ones = np.ones((a.shape[-1], a.shape[-1]), dtype=dtype)
    a *= np.tril(ones) if lower else np.triu(ones)
    # Gradient checks will fail without symmetrization as the eigh jvp rule
    # is only correct for tangents in the symmetric subspace, whereas the
    # checker checks against unconstrained (co)tangents.
    if dtype not in complex_types:
      f = partial(jnp.linalg.eigh, UPLO=uplo, symmetrize_input=True)
    else:  # only check eigenvalue grads for complex matrices
      f = lambda a: partial(jnp.linalg.eigh, UPLO=uplo, symmetrize_input=True)(a)[0]
    jtu.check_grads(f, (a,), 2, rtol=1e-5)

  @jtu.sample_product(
      shape=[(1, 1), (4, 4), (5, 5), (50, 50)],
      dtype=complex_types,
      lower=[True, False],
      eps=[1e-5],
  )
  def testEighGradVectorComplex(self, shape, dtype, lower, eps):
    rng = jtu.rand_default(self.rng())
    # Special case to test for complex eigenvector grad correctness.
    # Exact eigenvector coordinate gradients are hard to test numerically for complex
    # eigensystem solvers given the extra degrees of per-eigenvector phase freedom.
    # Instead, we numerically verify the eigensystem properties on the perturbed
    # eigenvectors.  You only ever want to optimize eigenvector directions, not coordinates!
    uplo = "L" if lower else "U"
    a = rng(shape, dtype)
    a = (a + np.conj(a.T)) / 2
    a = np.tril(a) if lower else np.triu(a)
    a_dot = eps * rng(shape, dtype)
    a_dot = (a_dot + np.conj(a_dot.T)) / 2
    a_dot = np.tril(a_dot) if lower else np.triu(a_dot)
    # evaluate eigenvector gradient and groundtruth eigensystem for perturbed input matrix
    f = partial(jnp.linalg.eigh, UPLO=uplo)
    (w, v), (dw, dv) = jvp(f, primals=(a,), tangents=(a_dot,))
    self.assertTrue(jnp.issubdtype(w.dtype, jnp.floating))
    self.assertTrue(jnp.issubdtype(dw.dtype, jnp.floating))
    new_a = a + a_dot
    new_w, new_v = f(new_a)
    new_a = (new_a + np.conj(new_a.T)) / 2
    new_w = new_w.astype(new_a.dtype)
    # Assert rtol eigenvalue delta between perturbed eigenvectors vs new true eigenvalues.
    RTOL = 1e-2
    with jax.numpy_rank_promotion('allow'):
      assert np.max(
        np.abs((np.diag(np.dot(np.conj((v+dv).T), np.dot(new_a,(v+dv)))) - new_w) / new_w)) < RTOL
      # Redundant to above, but also assert rtol for eigenvector property with new true eigenvalues.
      assert np.max(
        np.linalg.norm(np.abs(new_w*(v+dv) - np.dot(new_a, (v+dv))), axis=0) /
        np.linalg.norm(np.abs(new_w*(v+dv)), axis=0)
      ) < RTOL

  def testEighGradPrecision(self):
    rng = jtu.rand_default(self.rng())
    a = rng((3, 3), np.float32)
    jtu.assert_dot_precision(
        lax.Precision.HIGHEST, partial(jvp, jnp.linalg.eigh), (a,), (a,))

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (5, 5), (300, 300)],
    dtype=float_types + complex_types,
  )
  def testEighBatching(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    shape = (10,) + shape
    args = rng(shape, dtype)
    args = (args + np.conj(T(args))) / 2
    ws, vs = vmap(jsp.linalg.eigh)(args)
    ws = ws.astype(vs.dtype)
    norm = np.max(np.linalg.norm(np.matmul(args, vs) - ws[..., None, :] * vs))
    self.assertLess(norm, 1e-2)

  @jtu.sample_product(
    shape=[(1,), (4,), (5,)],
    dtype=(np.int32,),
  )
  def testLuPivotsToPermutation(self, shape, dtype):
    pivots_size = shape[-1]
    permutation_size = 2 * pivots_size

    pivots = jnp.arange(permutation_size - 1, pivots_size - 1, -1, dtype=dtype)
    pivots = jnp.broadcast_to(pivots, shape)
    actual = lax.linalg.lu_pivots_to_permutation(pivots, permutation_size)
    expected = jnp.arange(permutation_size - 1, -1, -1, dtype=dtype)
    expected = jnp.broadcast_to(expected, actual.shape)
    self.assertArraysEqual(actual, expected)

  @jtu.sample_product(
    shape=[(1,), (4,), (5,)],
    dtype=(np.int32,),
  )
  def testLuPivotsToPermutationBatching(self, shape, dtype):
    shape = (10,) + shape
    pivots_size = shape[-1]
    permutation_size = 2 * pivots_size

    pivots = jnp.arange(permutation_size - 1, pivots_size - 1, -1, dtype=dtype)
    pivots = jnp.broadcast_to(pivots, shape)
    batched_fn = vmap(
        lambda x: lax.linalg.lu_pivots_to_permutation(x, permutation_size))
    actual = batched_fn(pivots)
    expected = jnp.arange(permutation_size - 1, -1, -1, dtype=dtype)
    expected = jnp.broadcast_to(expected, actual.shape)
    self.assertArraysEqual(actual, expected)

  @jtu.sample_product(
    [dict(axis=axis, shape=shape, ord=ord)
     for axis, shape in [
       (None, (1,)), (None, (7,)), (None, (5, 8)),
       (0, (9,)), (0, (4, 5)), ((1,), (10, 7, 3)), ((-2,), (4, 8)),
       (-1, (6, 3)), ((0, 2), (3, 4, 5)), ((2, 0), (7, 8, 9)),
       (None, (7, 8, 11))]
     for ord in (
         [None] if axis is None and len(shape) > 2
         else [None, 0, 1, 2, 3, -1, -2, -3, jnp.inf, -jnp.inf]
         if (axis is None and len(shape) == 1) or
            isinstance(axis, int) or
            (isinstance(axis, tuple) and len(axis) == 1)
         else [None, 'fro', 1, 2, -1, -2, jnp.inf, -jnp.inf, 'nuc'])
    ],
    keepdims=[False, True],
    dtype=float_types + complex_types,
  )
  def testNorm(self, shape, dtype, ord, axis, keepdims):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fn = partial(np.linalg.norm, ord=ord, axis=axis, keepdims=keepdims)
    jnp_fn = partial(jnp.linalg.norm, ord=ord, axis=axis, keepdims=keepdims)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-3)
    self._CompileAndCheck(jnp_fn, args_maker)

  def testStringInfNorm(self):
    err, msg = ValueError, r"Invalid order 'inf' for vector norm."
    with self.assertRaisesRegex(err, msg):
      jnp.linalg.norm(jnp.array([1.0, 2.0, 3.0]), ord="inf")

  @jtu.sample_product(
      shape=[(2, 3), (4, 2, 3), (2, 3, 4, 5)],
      dtype=float_types + complex_types,
      keepdims=[True, False],
      ord=[1, -1, 2, -2, np.inf, -np.inf, 'fro', 'nuc'],
  )
  def testMatrixNorm(self, shape, dtype, keepdims, ord):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    if jtu.numpy_version() < (2, 0, 0):
      np_fn = partial(np.linalg.norm, ord=ord, keepdims=keepdims, axis=(-2, -1))
    else:
      np_fn = partial(np.linalg.matrix_norm, ord=ord, keepdims=keepdims)
    jnp_fn = partial(jnp.linalg.matrix_norm, ord=ord, keepdims=keepdims)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-3)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
      shape=[(3,), (3, 4), (2, 3, 4, 5)],
      dtype=float_types + complex_types,
      keepdims=[True, False],
      axis=[0, None],
      ord=[1, -1, 2, -2, np.inf, -np.inf],
  )
  def testVectorNorm(self, shape, dtype, keepdims, axis, ord):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    if jtu.numpy_version() < (2, 0, 0):
      def np_fn(x, *, ord, keepdims, axis):
        x = np.asarray(x)
        if axis is None:
          result = np_fn(x.ravel(), ord=ord, keepdims=False, axis=0)
          return np.reshape(result, (1,) * x.ndim) if keepdims else result
        return np.linalg.norm(x, ord=ord, keepdims=keepdims, axis=axis)
    else:
      np_fn = np.linalg.vector_norm
    np_fn = partial(np_fn, ord=ord, keepdims=keepdims, axis=axis)
    jnp_fn = partial(jnp.linalg.vector_norm, ord=ord, keepdims=keepdims, axis=axis)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-3)
    self._CompileAndCheck(jnp_fn, args_maker)

  # jnp.linalg.vecdot is an alias of jnp.vecdot; do a minimal test here.
  @jtu.sample_product(
      [
        dict(lhs_shape=(2, 2, 2), rhs_shape=(2, 2), axis=0),
        dict(lhs_shape=(2, 2, 2), rhs_shape=(2, 2), axis=1),
        dict(lhs_shape=(2, 2, 2), rhs_shape=(2, 2), axis=-1),
      ],
      dtype=int_types + float_types + complex_types
  )
  @jax.default_matmul_precision("float32")
  def testVecdot(self, lhs_shape, rhs_shape, axis, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    np_fn = jtu.numpy_vecdot if jtu.numpy_version() < (2, 0, 0) else np.linalg.vecdot
    np_fn = jtu.promote_like_jnp(partial(np_fn, axis=axis))
    jnp_fn = partial(jnp.linalg.vecdot, axis=axis)
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  # jnp.linalg.matmul is an alias of jnp.matmul; do a minimal test here.
  @jtu.sample_product(
      [
        dict(lhs_shape=(3,), rhs_shape=(3,)), # vec-vec
        dict(lhs_shape=(2, 3), rhs_shape=(3,)), # mat-vec
        dict(lhs_shape=(3,), rhs_shape=(3, 4)), # vec-mat
        dict(lhs_shape=(2, 3), rhs_shape=(3, 4)), # mat-mat
      ],
      dtype=float_types + complex_types
  )
  @jax.default_matmul_precision("float32")
  def testMatmul(self, lhs_shape, rhs_shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    np_fn = jtu.promote_like_jnp(
        np.matmul if jtu.numpy_version() < (2, 0, 0) else np.linalg.matmul)
    jnp_fn = jnp.linalg.matmul
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  # jnp.linalg.tensordot is an alias of jnp.tensordot; do a minimal test here.
  @jtu.sample_product(
      [
        dict(lhs_shape=(2, 2, 2), rhs_shape=(2, 2), axes=0),
        dict(lhs_shape=(2, 2, 2), rhs_shape=(2, 2), axes=1),
        dict(lhs_shape=(2, 2, 2), rhs_shape=(2, 2), axes=2),
      ],
      dtype=float_types + complex_types
  )
  @jax.default_matmul_precision("float32")
  def testTensordot(self, lhs_shape, rhs_shape, axes, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    np_fn = jtu.promote_like_jnp(
      partial(
        np.tensordot if jtu.numpy_version() < (2, 0, 0) else np.linalg.tensordot,
        axes=axes))
    jnp_fn = partial(jnp.linalg.tensordot, axes=axes)
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  @jtu.sample_product(
      [
          dict(m=m, n=n, full_matrices=full_matrices, hermitian=hermitian)
          for (m, n), full_matrices in (
              list(
                  itertools.product(
                      itertools.product([0, 2, 7, 29, 32, 53], repeat=2),
                      [False, True],
                  )
              )
              +
              # Test cases that ensure we are economical when computing the SVD
              # and its gradient. If we form a 400kx400k matrix explicitly we
              # will OOM.
              [((400000, 2), False), ((2, 400000), False)]
          )
          for hermitian in ([False, True] if m == n else [False])
      ],
      b=[(), (3,), (2, 3)],
      dtype=float_types + complex_types,
      compute_uv=[False, True],
  )
  @jax.default_matmul_precision("float32")
  def testSVD(self, b, m, n, dtype, full_matrices, compute_uv, hermitian):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(b + (m, n), dtype)]

    def compute_max_backward_error(operand, reconstructed_operand):
      error_norm = np.linalg.norm(operand - reconstructed_operand,
                                  axis=(-2, -1))
      backward_error = (error_norm /
                        np.linalg.norm(operand, axis=(-2, -1)))
      max_backward_error = np.amax(backward_error)
      return max_backward_error

    tol = 80 * jnp.finfo(dtype).eps
    reconstruction_tol = 2 * tol
    unitariness_tol = 3 * tol

    a, = args_maker()
    if hermitian:
      a = a + np.conj(T(a))
    out = jnp.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv,
                         hermitian=hermitian)
    if compute_uv:
      # Check the reconstructed matrices
      out = list(out)
      out[1] = out[1].astype(out[0].dtype)  # for strict dtype promotion.
      if m and n:
        if full_matrices:
          k = min(m, n)
          if m < n:
            max_backward_error = compute_max_backward_error(
                a, np.matmul(out[1][..., None, :] * out[0], out[2][..., :k, :]))
            self.assertLess(max_backward_error, reconstruction_tol)
          else:
            max_backward_error = compute_max_backward_error(
                a, np.matmul(out[1][..., None, :] * out[0][..., :, :k], out[2]))
            self.assertLess(max_backward_error, reconstruction_tol)
        else:
          max_backward_error = compute_max_backward_error(
              a, np.matmul(out[1][..., None, :] * out[0], out[2]))
          self.assertLess(max_backward_error, reconstruction_tol)

      # Check the unitary properties of the singular vector matrices.
      unitary_mat = np.real(np.matmul(np.conj(T(out[0])), out[0]))
      eye_slice = np.eye(out[0].shape[-1], dtype=unitary_mat.dtype)
      self.assertAllClose(np.broadcast_to(eye_slice, b + eye_slice.shape),
                          unitary_mat, rtol=unitariness_tol,
                          atol=unitariness_tol)
      if m >= n:
        unitary_mat = np.real(np.matmul(np.conj(T(out[2])), out[2]))
        eye_slice = np.eye(out[2].shape[-1], dtype=unitary_mat.dtype)
        self.assertAllClose(np.broadcast_to(eye_slice, b + eye_slice.shape),
                            unitary_mat, rtol=unitariness_tol,
                            atol=unitariness_tol)
      else:
        unitary_mat = np.real(np.matmul(out[2], np.conj(T(out[2]))))
        eye_slice = np.eye(out[2].shape[-2], dtype=unitary_mat.dtype)
        self.assertAllClose(np.broadcast_to(eye_slice, b + eye_slice.shape),
                            unitary_mat, rtol=unitariness_tol,
                            atol=unitariness_tol)
    else:
      self.assertTrue(np.allclose(np.linalg.svd(a, compute_uv=False),
                                  np.asarray(out), atol=1e-4, rtol=1e-4))

    self._CompileAndCheck(partial(jnp.linalg.svd, full_matrices=full_matrices,
                                  compute_uv=compute_uv),
                          args_maker)

    if not compute_uv and a.size < 100000:
      svd = partial(jnp.linalg.svd, full_matrices=full_matrices,
                    compute_uv=compute_uv)
      # TODO(phawkins): these tolerances seem very loose.
      if dtype == np.complex128:
        jtu.check_jvp(svd, partial(jvp, svd), (a,), rtol=1e-4, atol=1e-4,
                      eps=1e-8)
      else:
        jtu.check_jvp(svd, partial(jvp, svd), (a,), rtol=5e-2, atol=2e-1)

    if compute_uv and (not full_matrices):
      b, = args_maker()
      def f(x):
        u, s, v = jnp.linalg.svd(
          a + x * b,
          full_matrices=full_matrices,
          compute_uv=compute_uv)
        vdiag = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')
        return jnp.matmul(jnp.matmul(u, vdiag(s).astype(u.dtype)), v).real
      _, t_out = jvp(f, (1.,), (1.,))
      if dtype == np.complex128:
        atol = 2e-13
      else:
        atol = 5e-4
      self.assertArraysAllClose(t_out, b.real, atol=atol)

  def testJspSVDBasic(self):
    # since jax.scipy.linalg.svd is almost the same as jax.numpy.linalg.svd
    # do not check it functionality here
    jsp.linalg.svd(np.ones((2, 2), dtype=np.float32))

  @jtu.sample_product(
    shape=[(0, 2), (2, 0), (3, 4), (3, 3), (4, 3)],
    dtype=[np.float32],
    mode=["reduced", "r", "full", "complete", "raw"],
  )
  def testNumpyQrModes(self, shape, dtype, mode):
    rng = jtu.rand_default(self.rng())
    jnp_func = partial(jax.numpy.linalg.qr, mode=mode)
    np_func = partial(np.linalg.qr, mode=mode)
    if mode == "full":
      np_func = jtu.ignore_warning(category=DeprecationWarning, message="The 'full' option.*")(np_func)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_func, jnp_func, args_maker, rtol=1e-5, atol=1e-5,
                            check_dtypes=(mode != "raw"))
    self._CompileAndCheck(jnp_func, args_maker)

  @jtu.sample_product(
    shape=[(0, 0), (2, 0), (0, 2), (3, 3), (3, 4), (2, 10, 5),
           (2, 200, 100), (64, 16, 5), (33, 7, 3), (137, 9, 5), (20000, 2, 2)],
    dtype=float_types + complex_types,
    full_matrices=[False, True],
  )
  @jax.default_matmul_precision("float32")
  def testQr(self, shape, dtype, full_matrices):
    if (jtu.test_device_matches(["cuda"]) and
        _is_required_cuda_version_satisfied(12000)):
      self.skipTest("Triggers a bug in cuda-12 b/287345077")
    rng = jtu.rand_default(self.rng())
    m, n = shape[-2:]

    if full_matrices:
      mode, k = "complete", m
    else:
      mode, k = "reduced", min(m, n)

    a = rng(shape, dtype)
    lq, lr = jnp.linalg.qr(a, mode=mode)

    # np.linalg.qr doesn't support batch dimensions. But it seems like an
    # inevitable extension so we support it in our version.
    nq = np.zeros(shape[:-2] + (m, k), dtype)
    nr = np.zeros(shape[:-2] + (k, n), dtype)
    for index in np.ndindex(*shape[:-2]):
      nq[index], nr[index] = np.linalg.qr(a[index], mode=mode)

    max_rank = max(m, n)

    # Norm, adjusted for dimension and type.
    def norm(x):
      n = np.linalg.norm(x, axis=(-2, -1))
      return n / (max(1, max_rank) * jnp.finfo(dtype).eps)

    def compare_orthogonal(q1, q2):
      # Q is unique up to sign, so normalize the sign first.
      ratio = np.divide(np.where(q2 == 0, 0, q1), np.where(q2 == 0, 1, q2))
      sum_of_ratios = ratio.sum(axis=-2, keepdims=True)
      phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
      q1 *= phases
      nm = norm(q1 - q2)
      self.assertTrue(np.all(nm < 160), msg=f"norm={np.amax(nm)}")

    # Check a ~= qr
    norm_error = norm(a - np.matmul(lq, lr))
    self.assertTrue(np.all(norm_error < 60), msg=np.amax(norm_error))

    # Compare the first 'k' vectors of Q; the remainder form an arbitrary
    # orthonormal basis for the null space.
    compare_orthogonal(nq[..., :k], lq[..., :k])

    # Check that q is close to unitary.
    self.assertTrue(np.all(
        norm(np.eye(k) - np.matmul(np.conj(T(lq)), lq)) < 10))

    # This expresses identity function, which makes us robust to, e.g., the
    # tangents flipping the direction of vectors in Q.
    def qr_and_mul(a):
      q, r = jnp.linalg.qr(a, mode=mode)
      return q @ r

    if m == n or (m > n and not full_matrices):
      jtu.check_jvp(qr_and_mul, partial(jvp, qr_and_mul), (a,), atol=3e-3)

  @jtu.skip_on_devices("tpu")
  def testQrInvalidDtypeCPU(self, shape=(5, 6), dtype=np.float16):
    # Regression test for https://github.com/google/jax/issues/10530
    rng = jtu.rand_default(self.rng())
    arr = rng(shape, dtype)
    if jtu.test_device_matches(['cpu']):
      err, msg = NotImplementedError, "Unsupported dtype float16"
    else:
      err, msg = ValueError, r"Unsupported dtype dtype\('float16'\)"
    with self.assertRaisesRegex(err, msg):
      jnp.linalg.qr(arr)

  @jtu.sample_product(
    shape=[(10, 4, 5), (5, 3, 3), (7, 6, 4)],
    dtype=float_types + complex_types,
  )
  def testQrBatching(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args = rng(shape, jnp.float32)
    qs, rs = vmap(jsp.linalg.qr)(args)
    self.assertTrue(np.all(np.linalg.norm(args - np.matmul(qs, rs)) < 1e-3))

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (2, 3, 5), (5, 5, 5), (20, 20), (5, 10)],
    pnorm=[jnp.inf, -jnp.inf, 1, -1, 2, -2, 'fro'],
    dtype=float_types + complex_types,
  )
  @jtu.skip_on_devices("gpu")  # TODO(#2203): numerical errors
  def testCond(self, shape, pnorm, dtype):
    def gen_mat():
      # arr_gen = jtu.rand_some_nan(self.rng())
      arr_gen = jtu.rand_default(self.rng())
      res = arr_gen(shape, dtype)
      return res

    def args_gen(p):
      def _args_gen():
        return [gen_mat(), p]
      return _args_gen

    args_maker = args_gen(pnorm)
    if pnorm not in [2, -2] and len(set(shape[-2:])) != 1:
      with self.assertRaises(np.linalg.LinAlgError):
        jnp.linalg.cond(*args_maker())
    else:
      self._CheckAgainstNumpy(np.linalg.cond, jnp.linalg.cond, args_maker,
                              check_dtypes=False, tol=1e-3)
      partial_norm = partial(jnp.linalg.cond, p=pnorm)
      self._CompileAndCheck(partial_norm, lambda: [gen_mat()],
                            check_dtypes=False, rtol=1e-03, atol=1e-03)

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (200, 200), (7, 7, 7, 7)],
    dtype=float_types,
  )
  def testTensorinv(self, shape, dtype):
    rng = jtu.rand_default(self.rng())

    def tensor_maker():
      invertible = False
      while not invertible:
        a = rng(shape, dtype)
        try:
          np.linalg.inv(a)
          invertible = True
        except np.linalg.LinAlgError:
          pass
      return a

    args_maker = lambda: [tensor_maker(), int(np.floor(len(shape) / 2))]
    self._CheckAgainstNumpy(np.linalg.tensorinv, jnp.linalg.tensorinv, args_maker,
                            check_dtypes=False, tol=1e-3)
    partial_inv = partial(jnp.linalg.tensorinv, ind=int(np.floor(len(shape) / 2)))
    self._CompileAndCheck(partial_inv, lambda: [tensor_maker()], check_dtypes=False, rtol=1e-03, atol=1e-03)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
     for lhs_shape, rhs_shape in [
      ((1, 1), (1, 1)),
      ((4, 4), (4,)),
      ((8, 8), (8, 4)),
      ((2, 2), (3, 2, 2)),
      ((2, 1, 3, 3), (1, 4, 3, 4)),
      ((1, 0, 0), (1, 0, 2)),
     ]
    ],
    dtype=float_types + complex_types,
  )
  @jtu.ignore_warning(category=FutureWarning, message="jnp.linalg.solve: batched")
  def testSolve(self, lhs_shape, rhs_shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(np.linalg.solve, jnp.linalg.solve, args_maker,
                            tol=1e-3)
    self._CompileAndCheck(jnp.linalg.solve, args_maker)

  @jtu.sample_product(
      lhs_shape=[(2, 2), (2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2, 2)],
      rhs_shape=[(2,), (2, 2), (2, 2, 2), (2, 2, 2, 2)]
  )
  @jtu.ignore_warning(category=FutureWarning, message="jnp.linalg.solve: batched")
  def testSolveBroadcasting(self, lhs_shape, rhs_shape):
    # Batched solve can involve some ambiguities; this test checks
    # that we match NumPy's convention in all cases.
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, 'float32'), rng(rhs_shape, 'float32')]

    if jtu.numpy_version() >= (2, 0, 0):
      # TODO(jakevdp) remove this condition after solve broadcast deprecation is finalized.
      if len(rhs_shape) == 1 or (len(lhs_shape) != len(rhs_shape) + 1):
        self._CheckAgainstNumpy(np.linalg.solve, jnp.linalg.solve, args_maker, tol=1E-3)
    else:  # numpy 1.X
      # As of numpy 1.26.3, np.linalg.solve fails when this condition is not met.
      if len(lhs_shape) == 2 or len(rhs_shape) > 1:
        self._CheckAgainstNumpy(np.linalg.solve, jnp.linalg.solve, args_maker, tol=1E-3)
    self._CompileAndCheck(jnp.linalg.solve, args_maker)

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (2, 5, 5), (100, 100), (5, 5, 5), (0, 0)],
    dtype=float_types,
  )
  def testInv(self, shape, dtype):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      invertible = False
      while not invertible:
        a = rng(shape, dtype)
        try:
          np.linalg.inv(a)
          invertible = True
        except np.linalg.LinAlgError:
          pass
      return [a]

    self._CheckAgainstNumpy(np.linalg.inv, jnp.linalg.inv, args_maker,
                            tol=1e-3)
    self._CompileAndCheck(jnp.linalg.inv, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, hermitian=hermitian)
     for shape in [(1, 1), (4, 4), (3, 10, 10), (2, 70, 7), (2000, 7),
                   (7, 1000), (70, 7, 2), (2, 0, 0), (3, 0, 2), (1, 0),
                   (400000, 2), (2, 400000)]
     for hermitian in ([False, True] if shape[-1] == shape[-2] else [False])],
    dtype=float_types + complex_types,
  )
  def testPinv(self, shape, hermitian, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    jnp_fn = partial(jnp.linalg.pinv, hermitian=hermitian)
    def np_fn(a):
      # Symmetrize the input matrix to match the jnp behavior.
      if hermitian:
        a = (a + T(a.conj())) / 2
      return np.linalg.pinv(a, hermitian=hermitian)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker)

    # TODO(phawkins): 6e-2 seems like a very loose tolerance.
    jtu.check_grads(jnp_fn, args_maker(), 1, rtol=6e-2, atol=1e-3)

  def testPinvGradIssue2792(self):
    def f(p):
      a = jnp.array([[0., 0.],[-p, 1.]], jnp.float32) * 1 / (1 + p**2)
      return jnp.linalg.pinv(a)
    j = jax.jacobian(f)(jnp.float32(2.))
    self.assertAllClose(jnp.array([[0., -1.], [ 0., 0.]], jnp.float32), j)

    expected = jnp.array([[[[-1., 0.], [ 0., 0.]], [[0., -1.], [0.,  0.]]],
                         [[[0.,  0.], [-1., 0.]], [[0.,  0.], [0., -1.]]]],
                         dtype=jnp.float32)
    self.assertAllClose(
      expected, jax.jacobian(jnp.linalg.pinv)(jnp.eye(2, dtype=jnp.float32)))

  @jtu.sample_product(
    shape=[(1, 1), (2, 2), (4, 4), (5, 5), (1, 2, 2), (2, 3, 3), (2, 5, 5)],
    dtype=float_types + complex_types,
    n=[-5, -2, -1, 0, 1, 2, 3, 4, 5, 10],
  )
  @jax.default_matmul_precision("float32")
  def testMatrixPower(self, shape, dtype, n):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(partial(np.linalg.matrix_power, n=n),
                            partial(jnp.linalg.matrix_power, n=n),
                            args_maker, tol=1e-3)
    self._CompileAndCheck(partial(jnp.linalg.matrix_power, n=n), args_maker,
                          rtol=1e-3)

  @jtu.sample_product(
    shape=[(3, ), (1, 2), (8, 5), (4, 4), (5, 5), (50, 50), (3, 4, 5),
           (2, 3, 4, 5)],
    dtype=float_types + complex_types,
  )
  def testMatrixRank(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    a, = args_maker()
    self._CheckAgainstNumpy(np.linalg.matrix_rank, jnp.linalg.matrix_rank,
                            args_maker, check_dtypes=False, tol=1e-3)
    self._CompileAndCheck(jnp.linalg.matrix_rank, args_maker,
                          check_dtypes=False, rtol=1e-3)

  @jtu.sample_product(
    shapes=[
      [(3, ), (3, 1)],  # quick-out codepath
      [(1, 3), (3, 5), (5, 2)],  # multi_dot_three codepath
      [(1, 3), (3, 5), (5, 2), (2, 7), (7, )]  # dynamic programming codepath
    ],
    dtype=float_types + complex_types,
  )
  def testMultiDot(self, shapes, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [[rng(shape, dtype) for shape in shapes]]

    np_fun = np.linalg.multi_dot
    jnp_fun = partial(jnp.linalg.multi_dot, precision=lax.Precision.HIGHEST)
    tol = {np.float32: 1e-4, np.float64: 1e-10,
           np.complex64: 1e-4, np.complex128: 1e-10}

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker,
                          atol=tol, rtol=tol)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [
          ((1, 1), (1, 1)),
          ((4, 6), (4,)),
          ((6, 6), (6, 1)),
          ((8, 6), (8, 4)),
          ((0, 3), (0,)),
          ((3, 0), (3,)),
          ((3, 1), (3, 0)),
      ]
    ],
    rcond=[-1, None, 0.5],
    dtype=float_types + complex_types,
  )
  def testLstsq(self, lhs_shape, rhs_shape, dtype, rcond):
    rng = jtu.rand_default(self.rng())
    np_fun = partial(np.linalg.lstsq, rcond=rcond)
    jnp_fun = partial(jnp.linalg.lstsq, rcond=rcond)
    jnp_fun_numpy_resid = partial(jnp.linalg.lstsq, rcond=rcond, numpy_resid=True)
    tol = {np.float32: 1e-4, np.float64: 1e-12,
           np.complex64: 1e-5, np.complex128: 1e-12}
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(np_fun, jnp_fun_numpy_resid, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol)

    # Disabled because grad is flaky for low-rank inputs.
    # TODO:
    # jtu.check_grads(lambda *args: jnp_fun(*args)[0], args_maker(), order=2, atol=1e-2, rtol=1e-2)

  # Regression test for incorrect type for eigenvalues of a complex matrix.
  def testIssue669(self):
    def test(x):
      val, vec = jnp.linalg.eigh(x)
      return jnp.real(jnp.sum(val))

    grad_test_jc = jit(grad(jit(test)))
    xc = np.eye(3, dtype=np.complex64)
    self.assertAllClose(xc, grad_test_jc(xc))

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  @jtu.ignore_warning(category=FutureWarning, message="jnp.linalg.solve: batched")
  def testIssue1151(self):
    rng = self.rng()
    A = jnp.array(rng.randn(100, 3, 3), dtype=jnp.float32)
    b = jnp.array(rng.randn(100, 3), dtype=jnp.float32)
    x = jnp.linalg.solve(A, b)
    self.assertAllClose(vmap(jnp.dot)(A, x), b, atol=2e-3, rtol=1e-2)

    _ = jax.jacobian(jnp.linalg.solve, argnums=0)(A, b)
    _ = jax.jacobian(jnp.linalg.solve, argnums=1)(A, b)

    _ = jax.jacobian(jnp.linalg.solve, argnums=0)(A[0], b[0])
    _ = jax.jacobian(jnp.linalg.solve, argnums=1)(A[0], b[0])

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  @jax.legacy_prng_key("allow")
  def testIssue1383(self):
    seed = jax.random.PRNGKey(0)
    tmp = jax.random.uniform(seed, (2,2))
    a = jnp.dot(tmp, tmp.T)

    def f(inp):
      val, vec = jnp.linalg.eigh(inp)
      return jnp.dot(jnp.dot(vec, inp), vec.T)

    grad_func = jax.jacfwd(f)
    hess_func = jax.jacfwd(grad_func)
    cube_func = jax.jacfwd(hess_func)
    self.assertFalse(np.any(np.isnan(cube_func(a))))

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, axis=axis)
      for lhs_shape, rhs_shape, axis in [
          [(3,), (3,), -1],
          [(2, 3), (2, 3), -1],
          [(3, 4), (3, 4), 0],
          [(3, 5), (3, 4, 5), 0]
      ]],
    lhs_dtype=jtu.dtypes.numeric,
    rhs_dtype=jtu.dtypes.numeric,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testCross(self, lhs_shape, rhs_shape, lhs_dtype, rhs_dtype, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    lax_fun = partial(jnp.linalg.cross, axis=axis)
    np_fun = jtu.promote_like_jnp(partial(
      np.cross if jtu.numpy_version() < (2, 0, 0) else np.linalg.cross,
      axis=axis))
    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, lax_fun, args_maker)
      self._CompileAndCheck(lax_fun, args_maker)

  @jtu.sample_product(
      lhs_shape=[(0,), (3,), (5,)],
      rhs_shape=[(0,), (3,), (5,)],
      lhs_dtype=jtu.dtypes.numeric,
      rhs_dtype=jtu.dtypes.numeric,
  )
  def testOuter(self, lhs_shape, rhs_shape, lhs_dtype, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    lax_fun = jnp.linalg.outer
    np_fun = jtu.promote_like_jnp(
      np.outer if jtu.numpy_version() < (2, 0, 0) else np.linalg.outer)
    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, lax_fun, args_maker)
      self._CompileAndCheck(lax_fun, args_maker)

  @jtu.sample_product(
    shape = [(2, 3), (3, 2), (3, 3, 4), (4, 3, 3), (2, 3, 4, 5)],
    dtype = jtu.dtypes.all,
    offset=range(-2, 3)
  )
  def testDiagonal(self, shape, dtype, offset):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    lax_fun = partial(jnp.linalg.diagonal, offset=offset)
    if jtu.numpy_version() >= (2, 0, 0):
      np_fun = partial(np.linalg.diagonal, offset=offset)
    else:
      np_fun = partial(np.diagonal, offset=offset, axis1=-2, axis2=-1)
    self._CheckAgainstNumpy(np_fun, lax_fun, args_maker)
    self._CompileAndCheck(lax_fun, args_maker)


class ScipyLinalgTest(jtu.JaxTestCase):

  @jtu.sample_product(
    args=[
      (),
      (1,),
      (7, -2),
      (3, 4, 5),
      (np.ones((3, 4), dtype=float), 5,
       np.random.randn(5, 2).astype(float)),
    ]
  )
  def testBlockDiag(self, args):
    args_maker = lambda: args
    self._CheckAgainstNumpy(osp.linalg.block_diag, jsp.linalg.block_diag,
                            args_maker, check_dtypes=False)
    self._CompileAndCheck(jsp.linalg.block_diag, args_maker)


  @jtu.sample_product(
    shape=[(1, 1), (4, 5), (10, 5), (50, 50)],
    dtype=float_types + complex_types,
  )
  def testLu(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    x, = args_maker()
    p, l, u = jsp.linalg.lu(x)
    self.assertAllClose(x, np.matmul(p, np.matmul(l, u)),
                        rtol={np.float32: 1e-3, np.float64: 5e-12,
                              np.complex64: 1e-3, np.complex128: 1e-12},
                        atol={np.float32: 1e-5})
    self._CompileAndCheck(jsp.linalg.lu, args_maker)

  def testLuOfSingularMatrix(self):
    x = jnp.array([[-1., 3./2], [2./3, -1.]], dtype=np.float32)
    p, l, u = jsp.linalg.lu(x)
    self.assertAllClose(x, np.matmul(p, np.matmul(l, u)))

  @parameterized.parameters(lax_linalg.lu, lax_linalg._lu_python)
  def testLuOnZeroMatrix(self, lu):
    # Regression test for https://github.com/google/jax/issues/19076
    x = jnp.zeros((2, 2), dtype=np.float32)
    x_lu, _, _ = lu(x)
    self.assertArraysEqual(x_lu, x)

  @jtu.sample_product(
    shape=[(1, 1), (4, 5), (10, 5), (10, 10), (6, 7, 7)],
    dtype=float_types + complex_types,
  )
  def testLuGrad(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    lu = vmap(jsp.linalg.lu) if len(shape) > 2 else jsp.linalg.lu
    jtu.check_grads(lu, (a,), 2, atol=5e-2, rtol=3e-1)

  @jtu.sample_product(
    shape=[(4, 5), (6, 5)],
    dtype=[jnp.float32],
  )
  def testLuBatching(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args = [rng(shape, jnp.float32) for _ in range(10)]
    expected = [osp.linalg.lu(x) for x in args]
    ps = np.stack([out[0] for out in expected])
    ls = np.stack([out[1] for out in expected])
    us = np.stack([out[2] for out in expected])

    actual_ps, actual_ls, actual_us = vmap(jsp.linalg.lu)(jnp.stack(args))
    self.assertAllClose(ps, actual_ps)
    self.assertAllClose(ls, actual_ls, rtol=5e-6)
    self.assertAllClose(us, actual_us)

  @jtu.skip_on_devices("cpu", "tpu")
  def testLuCPUBackendOnGPU(self):
    # tests running `lu` on cpu when a gpu is present.
    jit(jsp.linalg.lu, backend="cpu")(np.ones((2, 2)))  # does not crash

  @jtu.sample_product(
    n=[1, 4, 5, 200],
    dtype=float_types + complex_types,
  )
  def testLuFactor(self, n, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((n, n), dtype)]

    x, = args_maker()
    lu, piv = jsp.linalg.lu_factor(x)
    l = np.tril(lu, -1) + np.eye(n, dtype=dtype)
    u = np.triu(lu)
    for i in range(n):
      x[[i, piv[i]],] = x[[piv[i], i],]
    self.assertAllClose(x, np.matmul(l, u), rtol=1e-3,
                        atol=1e-3)
    self._CompileAndCheck(jsp.linalg.lu_factor, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
     for lhs_shape, rhs_shape in [
       ((1, 1), (1, 1)),
       ((4, 4), (4,)),
       ((8, 8), (8, 4)),
     ]
    ],
    trans=[0, 1, 2],
    dtype=float_types + complex_types,
  )
  @jtu.skip_on_devices("cpu")  # TODO(frostig): Test fails on CPU sometimes
  def testLuSolve(self, lhs_shape, rhs_shape, dtype, trans):
    rng = jtu.rand_default(self.rng())
    osp_fun = lambda lu, piv, rhs: osp.linalg.lu_solve((lu, piv), rhs, trans=trans)
    jsp_fun = lambda lu, piv, rhs: jsp.linalg.lu_solve((lu, piv), rhs, trans=trans)

    def args_maker():
      a = rng(lhs_shape, dtype)
      lu, piv = osp.linalg.lu_factor(a)
      return [lu, piv, rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, tol=1e-3)
    self._CompileAndCheck(jsp_fun, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [
        ((1, 1), (1, 1)),
        ((4, 4), (4,)),
        ((8, 8), (8, 4)),
      ]
    ],
    [dict(assume_a=assume_a, lower=lower)
      for assume_a, lower in [
        ('gen', False),
        ('pos', False),
        ('pos', True),
      ]
    ],
    dtype=float_types + complex_types,
  )
  def testSolve(self, lhs_shape, rhs_shape, dtype, assume_a, lower):
    rng = jtu.rand_default(self.rng())
    osp_fun = lambda lhs, rhs: osp.linalg.solve(lhs, rhs, assume_a=assume_a, lower=lower)
    jsp_fun = lambda lhs, rhs: jsp.linalg.solve(lhs, rhs, assume_a=assume_a, lower=lower)

    def args_maker():
      a = rng(lhs_shape, dtype)
      if assume_a == 'pos':
        a = np.matmul(a, np.conj(T(a)))
        a = np.tril(a) if lower else np.triu(a)
      return [a, rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, tol=1e-3)
    self._CompileAndCheck(jsp_fun, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [
        ((4, 4), (4,)),
        ((4, 4), (4, 3)),
        ((2, 8, 8), (2, 8, 10)),
      ]
    ],
    lower=[False, True],
    transpose_a=[False, True],
    unit_diagonal=[False, True],
    dtype=float_types,
  )
  def testSolveTriangular(self, lower, transpose_a, unit_diagonal, lhs_shape,
                          rhs_shape, dtype):
    rng = jtu.rand_default(self.rng())
    k = rng(lhs_shape, dtype)
    l = np.linalg.cholesky(np.matmul(k, T(k))
                            + lhs_shape[-1] * np.eye(lhs_shape[-1]))
    l = l.astype(k.dtype)
    b = rng(rhs_shape, dtype)

    if unit_diagonal:
      a = np.tril(l, -1) + np.eye(lhs_shape[-1], dtype=dtype)
    else:
      a = l
    a = a if lower else T(a)

    inv = np.linalg.inv(T(a) if transpose_a else a).astype(a.dtype)
    if len(lhs_shape) == len(rhs_shape):
      np_ans = np.matmul(inv, b)
    else:
      np_ans = np.einsum("...ij,...j->...i", inv, b)

    # The standard scipy.linalg.solve_triangular doesn't support broadcasting.
    # But it seems like an inevitable extension so we support it.
    ans = jsp.linalg.solve_triangular(
        l if lower else T(l), b, trans=1 if transpose_a else 0, lower=lower,
        unit_diagonal=unit_diagonal)

    self.assertAllClose(np_ans, ans,
                        rtol={np.float32: 1e-4, np.float64: 1e-11})

  @jtu.sample_product(
    [dict(left_side=left_side, a_shape=a_shape, b_shape=b_shape)
      for left_side, a_shape, b_shape in [
          (False, (4, 4), (4,)),
          (False, (4, 4), (1, 4,)),
          (False, (3, 3), (4, 3)),
          (True, (4, 4), (4,)),
          (True, (4, 4), (4, 1)),
          (True, (4, 4), (4, 3)),
          (True, (2, 8, 8), (2, 8, 10)),
      ]
    ],
    [dict(dtype=dtype, conjugate_a=conjugate_a)
      for dtype in float_types + complex_types
      for conjugate_a in (
          [False] if jnp.issubdtype(dtype, jnp.floating) else [False, True])
    ],
    lower=[False, True],
    unit_diagonal=[False, True],
    transpose_a=[False, True],
  )
  def testTriangularSolveGrad(
      self, lower, transpose_a, conjugate_a, unit_diagonal, left_side, a_shape,
      b_shape, dtype):
    rng = jtu.rand_default(self.rng())
    # Test lax.linalg.triangular_solve instead of scipy.linalg.solve_triangular
    # because it exposes more options.
    A = jnp.tril(rng(a_shape, dtype) + 5 * np.eye(a_shape[-1], dtype=dtype))
    A = A if lower else T(A)
    B = rng(b_shape, dtype)
    f = partial(lax.linalg.triangular_solve, lower=lower, transpose_a=transpose_a,
                conjugate_a=conjugate_a, unit_diagonal=unit_diagonal,
                left_side=left_side)
    jtu.check_grads(f, (A, B), order=1, rtol=4e-2, eps=1e-3)

  @jtu.sample_product(
    [dict(left_side=left_side, a_shape=a_shape, b_shape=b_shape, bdims=bdims)
      for left_side, a_shape, b_shape, bdims in [
          (False, (4, 4), (2, 3, 4,), (None, 0)),
          (False, (2, 4, 4), (2, 2, 3, 4,), (None, 0)),
          (False, (2, 4, 4), (3, 4,), (0, None)),
          (False, (2, 4, 4), (2, 3, 4,), (0, 0)),
          (True, (2, 4, 4), (2, 4, 3), (0, 0)),
          (True, (2, 4, 4), (2, 2, 4, 3), (None, 0)),
      ]
    ],
  )
  def testTriangularSolveBatching(self, left_side, a_shape, b_shape, bdims):
    rng = jtu.rand_default(self.rng())
    A = jnp.tril(rng(a_shape, np.float32)
                + 5 * np.eye(a_shape[-1], dtype=np.float32))
    B = rng(b_shape, np.float32)
    solve = partial(lax.linalg.triangular_solve, lower=True, transpose_a=False,
                    conjugate_a=False, unit_diagonal=False, left_side=left_side)
    X = vmap(solve, bdims)(A, B)
    matmul = partial(jnp.matmul, precision=lax.Precision.HIGHEST)
    Y = matmul(A, X) if left_side else matmul(X, A)
    self.assertArraysAllClose(Y, jnp.broadcast_to(B, Y.shape), atol=1e-4)

  def testTriangularSolveGradPrecision(self):
    rng = jtu.rand_default(self.rng())
    a = jnp.tril(rng((3, 3), np.float32))
    b = rng((1, 3), np.float32)
    jtu.assert_dot_precision(
        lax.Precision.HIGHEST,
        partial(jvp, lax.linalg.triangular_solve),
        (a, b),
        (a, b))

  @jtu.sample_product(
    n=[1, 4, 5, 20, 50, 100],
    batch_size=[(), (2,), (3, 4)] if scipy_version >= (1, 9, 0) else [()],
    dtype=int_types + float_types + complex_types
  )
  def testExpm(self, n, batch_size, dtype):
    if (jtu.test_device_matches(["cuda"]) and
        _is_required_cuda_version_satisfied(12000)):
      self.skipTest("Triggers a bug in cuda-12 b/287345077")

    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng((*batch_size, n, n), dtype)]

    # Compare to numpy with JAX type promotion semantics.
    def osp_fun(A):
      return osp.linalg.expm(np.array(*promote_dtypes_inexact(A)))
    jsp_fun = jsp.linalg.expm
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker)
    self._CompileAndCheck(jsp_fun, args_maker)

    args_maker_triu = lambda: [np.triu(rng((*batch_size, n, n), dtype))]
    jsp_fun_triu = lambda a: jsp.linalg.expm(a, upper_triangular=True)
    self._CheckAgainstNumpy(osp_fun, jsp_fun_triu, args_maker_triu)
    self._CompileAndCheck(jsp_fun_triu, args_maker_triu)

  @jtu.sample_product(
    # Skip empty shapes because scipy fails: https://github.com/scipy/scipy/issues/1532
    shape=[(3, 4), (3, 3), (4, 3)],
    dtype=[np.float32],
    mode=["full", "r", "economic"],
  )
  def testScipyQrModes(self, shape, dtype, mode):
    rng = jtu.rand_default(self.rng())
    jsp_func = partial(jax.scipy.linalg.qr, mode=mode)
    sp_func = partial(scipy.linalg.qr, mode=mode)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(sp_func, jsp_func, args_maker, rtol=1E-5, atol=1E-5)
    self._CompileAndCheck(jsp_func, args_maker)

  @jtu.sample_product(
      [dict(shape=shape, k=k)
       for shape in [(1, 1), (3, 4, 4), (10, 5)]
       # TODO(phawkins): there are some test failures on GPU for k=0
       for k in range(1, shape[-1] + 1)],
      dtype=float_types + complex_types,
  )
  def testHouseholderProduct(self, shape, k, dtype):

    @partial(np.vectorize, signature='(m,n),(k)->(m,n)')
    def reference_fn(a, taus):
      if dtype == np.float32:
        q, _, info = scipy.linalg.lapack.sorgqr(a, taus)
      elif dtype == np.float64:
        q, _, info = scipy.linalg.lapack.dorgqr(a, taus)
      elif dtype == np.complex64:
        q, _, info = scipy.linalg.lapack.cungqr(a, taus)
      elif dtype == np.complex128:
        q, _, info = scipy.linalg.lapack.zungqr(a, taus)
      else:
        assert False, dtype
      assert info == 0, info
      return q

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), rng(shape[:-2] + (k,), dtype)]
    tol = {np.float32: 1e-5, np.complex64: 1e-5, np.float64: 1e-12,
           np.complex128: 1e-12}
    self._CheckAgainstNumpy(reference_fn, lax.linalg.householder_product,
                            args_maker, rtol=tol, atol=tol)
    self._CompileAndCheck(lax.linalg.householder_product, args_maker)

  @jtu.sample_product(
      shape=[(1, 1), (2, 4, 4), (0, 100, 100), (10, 10)],
      dtype=float_types + complex_types,
      calc_q=[False, True],
  )
  @jtu.run_on_devices("cpu")
  def testHessenberg(self, shape, dtype, calc_q):
    rng = jtu.rand_default(self.rng())
    jsp_func = partial(jax.scipy.linalg.hessenberg, calc_q=calc_q)
    if calc_q:
      sp_func = np.vectorize(partial(scipy.linalg.hessenberg, calc_q=True),
                             otypes=(dtype, dtype),
                             signature='(n,n)->(n,n),(n,n)')
    else:
      sp_func = np.vectorize(scipy.linalg.hessenberg, signature='(n,n)->(n,n)',
                             otypes=(dtype,))
    args_maker = lambda: [rng(shape, dtype)]
    # scipy.linalg.hessenberg sometimes returns a float Q matrix for complex
    # inputs
    self._CheckAgainstNumpy(sp_func, jsp_func, args_maker, rtol=1e-5, atol=1e-5,
                            check_dtypes=not calc_q)
    self._CompileAndCheck(jsp_func, args_maker)

  @jtu.sample_product(
      shape=[(1, 1), (2, 2, 2), (4, 4), (10, 10), (2, 5, 5)],
      dtype=float_types + complex_types,
      lower=[False, True],
  )
  @jtu.skip_on_devices("tpu","rocm")
  def testTridiagonal(self, shape, dtype, lower):
    rng = jtu.rand_default(self.rng())
    def jax_func(a):
      return lax.linalg.tridiagonal(a, lower=lower)

    real_dtype = jnp.finfo(dtype).dtype
    @partial(np.vectorize, otypes=(dtype, real_dtype, real_dtype, dtype),
            signature='(n,n)->(n,n),(n),(k),(k)')
    def sp_func(a):
      if dtype == np.float32:
        c, d, e, tau, info = scipy.linalg.lapack.ssytrd(a, lower=lower)
      elif dtype == np.float64:
        c, d, e, tau, info = scipy.linalg.lapack.dsytrd(a, lower=lower)
      elif dtype == np.complex64:
        c, d, e, tau, info = scipy.linalg.lapack.chetrd(a, lower=lower)
      elif dtype == np.complex128:
        c, d, e, tau, info = scipy.linalg.lapack.zhetrd(a, lower=lower)
      else:
        assert False, dtype
      assert info == 0
      return c, d, e, tau

    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(sp_func, jax_func, args_maker, rtol=1e-4, atol=1e-4,
                            check_dtypes=False)


  @jtu.sample_product(
    n=[1, 4, 5, 20, 50, 100],
    dtype=float_types + complex_types,
  )
  def testIssue2131(self, n, dtype):
    args_maker_zeros = lambda: [np.zeros((n, n), dtype)]
    osp_fun = lambda a: osp.linalg.expm(a)
    jsp_fun = lambda a: jsp.linalg.expm(a)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker_zeros)
    self._CompileAndCheck(jsp_fun, args_maker_zeros)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [
          [(1, 1), (1,)],
          [(4, 4), (4,)],
          [(4, 4), (4, 4)],
      ]
    ],
    dtype=float_types,
    lower=[True, False],
  )
  def testChoSolve(self, lhs_shape, rhs_shape, dtype, lower):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      b = rng(rhs_shape, dtype)
      if lower:
        L = np.tril(rng(lhs_shape, dtype))
        return [(L, lower), b]
      else:
        U = np.triu(rng(lhs_shape, dtype))
        return [(U, lower), b]
    self._CheckAgainstNumpy(osp.linalg.cho_solve, jsp.linalg.cho_solve,
                            args_maker, tol=1e-3)


  @jtu.sample_product(
    n=[1, 4, 5, 20, 50, 100],
    dtype=float_types + complex_types,
  )
  def testExpmFrechet(self, n, dtype):
    rng = jtu.rand_small(self.rng())
    if dtype == np.float64 or dtype == np.complex128:
      target_norms = [1.0e-2, 2.0e-1, 9.0e-01, 2.0, 3.0]
      # TODO(zhangqiaorjc): Reduce tol to default 1e-15.
      tol = {
        np.dtype(np.float64): 1e-14,
        np.dtype(np.complex128): 1e-14,
      }
    elif dtype == np.float32 or dtype == np.complex64:
      target_norms = [4.0e-1, 1.0, 3.0]
      tol = None
    else:
      raise TypeError(f"{dtype=} is not supported.")
    for norm in target_norms:
      def args_maker():
        a = rng((n, n), dtype)
        a = a / np.linalg.norm(a, 1) * norm
        e = rng((n, n), dtype)
        return [a, e, ]

      #compute_expm is True
      osp_fun = lambda a,e: osp.linalg.expm_frechet(a,e,compute_expm=True)
      jsp_fun = lambda a,e: jsp.linalg.expm_frechet(a,e,compute_expm=True)
      self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                              check_dtypes=False, tol=tol)
      self._CompileAndCheck(jsp_fun, args_maker, check_dtypes=False)
      #compute_expm is False
      osp_fun = lambda a,e: osp.linalg.expm_frechet(a,e,compute_expm=False)
      jsp_fun = lambda a,e: jsp.linalg.expm_frechet(a,e,compute_expm=False)
      self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                              check_dtypes=False, tol=tol)
      self._CompileAndCheck(jsp_fun, args_maker, check_dtypes=False)

  @jtu.sample_product(
    n=[1, 4, 5, 20, 50],
    dtype=float_types + complex_types,
  )
  def testExpmGrad(self, n, dtype):
    rng = jtu.rand_small(self.rng())
    a = rng((n, n), dtype)
    if dtype == np.float64 or dtype == np.complex128:
      target_norms = [1.0e-2, 2.0e-1, 9.0e-01, 2.0, 3.0]
    elif dtype == np.float32 or dtype == np.complex64:
      target_norms = [4.0e-1, 1.0, 3.0]
    else:
      raise TypeError(f"{dtype=} is not supported.")
    # TODO(zhangqiaorjc): Reduce tol to default 1e-5.
    # Lower tolerance is due to 2nd order derivative.
    tol = {
      # Note that due to inner_product, float and complex tol are coupled.
      np.dtype(np.float32): 0.02,
      np.dtype(np.complex64): 0.02,
      np.dtype(np.float64): 1e-4,
      np.dtype(np.complex128): 1e-4,
    }
    for norm in target_norms:
      a = a / np.linalg.norm(a, 1) * norm
      def expm(x):
        return jsp.linalg.expm(x, upper_triangular=False, max_squarings=16)
      jtu.check_grads(expm, (a,), modes=["fwd", "rev"], order=1, atol=tol,
                      rtol=tol)
  @jtu.sample_product(
    shape=[(4, 4), (15, 15), (50, 50), (100, 100)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testSchur(self, shape, dtype):
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(shape, dtype)]

      self._CheckAgainstNumpy(osp.linalg.schur, jsp.linalg.schur, args_maker)
      self._CompileAndCheck(jsp.linalg.schur, args_maker)

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (15, 15), (50, 50), (100, 100)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testRsf2csf(self, shape, dtype):
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(shape, dtype), rng(shape, dtype)]
      tol = 3e-5
      self._CheckAgainstNumpy(osp.linalg.rsf2csf, jsp.linalg.rsf2csf,
                              args_maker, tol=tol)
      self._CompileAndCheck(jsp.linalg.rsf2csf, args_maker)

  @jtu.sample_product(
    shape=[(1, 1), (5, 5), (20, 20), (50, 50)],
    dtype=float_types + complex_types,
    disp=[True, False],
  )
  # funm uses jax.scipy.linalg.schur which is implemented for a CPU
  # backend only, so tests on GPU and TPU backends are skipped here
  @jtu.run_on_devices("cpu")
  def testFunm(self, shape, dtype, disp):
      def func(x):
        return x**-2.718
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(shape, dtype)]
      jnp_fun = lambda arr: jsp.linalg.funm(arr, func, disp=disp)
      scp_fun = lambda arr: osp.linalg.funm(arr, func, disp=disp)
      self._CheckAgainstNumpy(jnp_fun, scp_fun, args_maker, check_dtypes=False,
                              tol={np.complex64: 1e-5, np.complex128: 1e-6})
      self._CompileAndCheck(jnp_fun, args_maker, atol=2e-5)

  @jtu.sample_product(
    shape=[(4, 4), (15, 15), (50, 50), (100, 100)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testSqrtmPSDMatrix(self, shape, dtype):
    # Checks against scipy.linalg.sqrtm when the principal square root
    # is guaranteed to be unique (i.e no negative real eigenvalue)
    rng = jtu.rand_default(self.rng())
    arg = rng(shape, dtype)
    mat = arg @ arg.T
    args_maker = lambda : [mat]
    if dtype == np.float32 or dtype == np.complex64:
        tol = 1e-4
    else:
        tol = 1e-8
    self._CheckAgainstNumpy(osp.linalg.sqrtm,
                            jsp.linalg.sqrtm,
                            args_maker,
                            tol=tol,
                            check_dtypes=False)
    self._CompileAndCheck(jsp.linalg.sqrtm, args_maker)

  @jtu.sample_product(
    shape=[(4, 4), (15, 15), (50, 50), (100, 100)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testSqrtmGenMatrix(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    arg = rng(shape, dtype)
    if dtype == np.float32 or dtype == np.complex64:
      tol = 2e-3
    else:
      tol = 1e-8
    R = jsp.linalg.sqrtm(arg)
    self.assertAllClose(R @ R, arg, atol=tol, check_dtypes=False)

  @jtu.sample_product(
    [dict(diag=diag, expected=expected)
     for diag, expected in [([1, 0, 0], [1, 0, 0]), ([0, 4, 0], [0, 2, 0]),
                            ([0, 0, 0, 9],[0, 0, 0, 3]),
                            ([0, 0, 9, 0, 0, 4], [0, 0, 3, 0, 0, 2])]
    ],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testSqrtmEdgeCase(self, diag, expected, dtype):
    """
    Tests the zero numerator condition
    """
    mat = jnp.diag(jnp.array(diag)).astype(dtype)
    expected = jnp.diag(jnp.array(expected))
    root = jsp.linalg.sqrtm(mat)

    self.assertAllClose(root, expected, check_dtypes=False)

  @jtu.sample_product(
    cshape=[(), (4,), (8,), (3, 7), (0, 5, 1)],
    cdtype=float_types + complex_types,
    rshape=[(), (3,), (7,), (2, 1, 4), (19, 0)],
    rdtype=float_types + complex_types + int_types)
  def testToeplitzConstrcution(self, rshape, rdtype, cshape, cdtype):
    if ((rdtype in [np.float64, np.complex128]
         or cdtype in [np.float64, np.complex128])
        and not config.enable_x64.value):
      self.skipTest("Only run float64 testcase when float64 is enabled.")

    int_types_excl_i8 = set(int_types) - {np.int8}
    if ((rdtype in int_types_excl_i8 or cdtype in int_types_excl_i8)
        and jtu.test_device_matches(["gpu"])):
      self.skipTest("Integer (except int8) toeplitz is not supported on GPU yet.")

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(cshape, cdtype), rng(rshape, rdtype)]
    with jtu.strict_promotion_if_dtypes_match([rdtype, cdtype]):
      self._CheckAgainstNumpy(jtu.promote_like_jnp(osp.linalg.toeplitz),
                              jsp.linalg.toeplitz, args_maker)
      self._CompileAndCheck(jsp.linalg.toeplitz, args_maker)

  @jtu.sample_product(
    shape=[(), (3,), (1, 4), (1, 5, 9), (11, 0, 13)],
    dtype=float_types + complex_types + int_types)
  @jtu.skip_on_devices("rocm")
  def testToeplitzSymmetricConstruction(self, shape, dtype):
    if (dtype in [np.float64, np.complex128]
        and not config.enable_x64.value):
      self.skipTest("Only run float64 testcase when float64 is enabled.")

    int_types_excl_i8 = set(int_types) - {np.int8}
    if (dtype in int_types_excl_i8
        and jtu.test_device_matches(["gpu"])):
      self.skipTest("Integer (except int8) toeplitz is not supported on GPU yet.")

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(jtu.promote_like_jnp(osp.linalg.toeplitz),
                            jsp.linalg.toeplitz, args_maker)
    self._CompileAndCheck(jsp.linalg.toeplitz, args_maker)

  def testToeplitzConstructionWithKnownCases(self):
    # Test with examples taken from SciPy doc for the corresponding function.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    ret = jsp.linalg.toeplitz(np.array([1.0, 2+3j, 4-1j]))
    self.assertAllClose(ret, np.array([
      [ 1.+0.j,  2.-3.j,  4.+1.j],
      [ 2.+3.j,  1.+0.j,  2.-3.j],
      [ 4.-1.j,  2.+3.j,  1.+0.j]]))
    ret = jsp.linalg.toeplitz(np.array([1, 2, 3], dtype=np.float32),
                              np.array([1, 4, 5, 6], dtype=np.float32))
    self.assertAllClose(ret, np.array([
      [1, 4, 5, 6],
      [2, 1, 4, 5],
      [3, 2, 1, 4]], dtype=np.float32))


class LaxLinalgTest(jtu.JaxTestCase):
  """Tests for lax.linalg primitives."""

  @jtu.sample_product(
    n=[0, 4, 5, 50],
    dtype=float_types + complex_types,
    lower=[True, False],
    sort_eigenvalues=[True, False],
  )
  def testEigh(self, n, dtype, lower, sort_eigenvalues):
    rng = jtu.rand_default(self.rng())
    tol = 1e-3
    args_maker = lambda: [rng((n, n), dtype)]

    a, = args_maker()
    a = (a + np.conj(a.T)) / 2
    v, w = lax.linalg.eigh(np.tril(a) if lower else np.triu(a),
                           lower=lower, symmetrize_input=False,
                           sort_eigenvalues=sort_eigenvalues)
    w = np.asarray(w)
    v = np.asarray(v)
    self.assertLessEqual(
        np.linalg.norm(np.eye(n) - np.matmul(np.conj(T(v)), v)), 1e-3)
    self.assertLessEqual(np.linalg.norm(np.matmul(a, v) - w * v),
                         tol * np.linalg.norm(a))

    w_expected, v_expected = np.linalg.eigh(np.asarray(a))
    self.assertAllClose(w_expected, w if sort_eigenvalues else np.sort(w),
                        rtol=1e-4, atol=1e-4)

  def run_eigh_tridiagonal_test(self, alpha, beta):
    n = alpha.shape[-1]
    # scipy.linalg.eigh_tridiagonal doesn't support complex inputs, so for
    # this we call the slower numpy.linalg.eigh.
    if np.issubdtype(alpha.dtype, np.complexfloating):
      tridiagonal = np.diag(alpha) + np.diag(beta, 1) + np.diag(
          np.conj(beta), -1)
      eigvals_expected, _ = np.linalg.eigh(tridiagonal)
    else:
      eigvals_expected = scipy.linalg.eigh_tridiagonal(
          alpha, beta, eigvals_only=True)
    eigvals = jax.scipy.linalg.eigh_tridiagonal(
        alpha, beta, eigvals_only=True)
    finfo = np.finfo(alpha.dtype)
    atol = 4 * np.sqrt(n) * finfo.eps * np.amax(np.abs(eigvals_expected))
    self.assertAllClose(eigvals_expected, eigvals, atol=atol, rtol=1e-4)

  @jtu.sample_product(
    n=[1, 2, 3, 7, 8, 100],
    dtype=float_types + complex_types,
  )
  def testToeplitz(self, n, dtype):
    for a, b in [[2, -1], [1, 0], [0, 1], [-1e10, 1e10], [-1e-10, 1e-10]]:
      alpha = a * np.ones([n], dtype=dtype)
      beta = b * np.ones([n - 1], dtype=dtype)
      self.run_eigh_tridiagonal_test(alpha, beta)

  @jtu.sample_product(
    n=[1, 2, 3, 7, 8, 100],
    dtype=float_types + complex_types,
  )
  def testRandomUniform(self, n, dtype):
    alpha = jtu.rand_uniform(self.rng())((n,), dtype)
    beta = jtu.rand_uniform(self.rng())((n - 1,), dtype)
    self.run_eigh_tridiagonal_test(alpha, beta)

  @jtu.sample_product(dtype=float_types + complex_types)
  def testSelect(self, dtype):
    n = 5
    alpha = jtu.rand_uniform(self.rng())((n,), dtype)
    beta = jtu.rand_uniform(self.rng())((n - 1,), dtype)
    eigvals_all = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, select="a",
                                                    eigvals_only=True)
    eps = np.finfo(alpha.dtype).eps
    atol = 2 * n * eps
    for first in range(n - 1):
      for last in range(first + 1, n - 1):
        # Check that we get the expected eigenvalues by selecting by
        # index range.
        eigvals_index = jax.scipy.linalg.eigh_tridiagonal(
            alpha, beta, select="i", select_range=(first, last),
            eigvals_only=True)
        self.assertAllClose(
            eigvals_all[first:(last + 1)], eigvals_index, atol=atol)

  @jtu.sample_product(dtype=[np.float32, np.float64])
  def test_tridiagonal_solve(self, dtype):
    dl = np.array([0.0, 2.0, 3.0], dtype=dtype)
    d = np.ones(3, dtype=dtype)
    du = np.array([1.0, 2.0, 0.0], dtype=dtype)
    m = 3
    B = np.ones([m, 1], dtype=dtype)
    X = lax.linalg.tridiagonal_solve(dl, d, du, B)
    A = np.eye(3, dtype=dtype)
    A[[1, 2], [0, 1]] = dl[1:]
    A[[0, 1], [1, 2]] = du[:-1]
    np.testing.assert_allclose(A @ X, B, rtol=1e-6, atol=1e-6)

  @jtu.sample_product(
    shape=[(4, 4), (15, 15), (50, 50), (100, 100)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testSchur(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(osp.linalg.schur, lax.linalg.schur, args_maker)
    self._CompileAndCheck(lax.linalg.schur, args_maker)

  @jtu.sample_product(
    shape=[(2, 2), (4, 4), (15, 15), (50, 50), (100, 100)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testSchurBatching(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    batch_size = 10
    shape = (batch_size,) + shape
    args = rng(shape, dtype)
    reconstruct = vmap(lambda S, T: S @ T @ jnp.conj(S.T))

    Ts, Ss = vmap(lax.linalg.schur)(args)
    self.assertAllClose(reconstruct(Ss, Ts), args, atol=1e-4)

  @jtu.sample_product(
    shape=[(2, 3), (2, 3, 4), (2, 3, 4, 5)],
    dtype=float_types + complex_types,
  )
  def testMatrixTranspose(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_fun = jnp.linalg.matrix_transpose
    if jtu.numpy_version() < (2, 0, 0):
      np_fun = lambda x: np.swapaxes(x, -1, -2)
    else:
      np_fun = np.linalg.matrix_transpose
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    n=[0, 1, 5, 10, 20],
    )
  def testHilbert(self, n):
    args_maker = lambda: []
    osp_fun = partial(osp.linalg.hilbert, n=n)
    jsp_fun = partial(jsp.linalg.hilbert, n=n)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker)
    self._CompileAndCheck(jsp_fun, args_maker)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
