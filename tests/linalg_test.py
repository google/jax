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

from functools import partial
import itertools
from collections.abc import Iterator
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
from jax._src.lib import cuda_versions
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
  if version == "<unknown>" or "rocm" in version.split():
    return False
  else:
    return int(version.split()[-1]) >= cuda_version


def _axis_for_ndim(ndim: int) -> Iterator[None | int | tuple[int, ...]]:
  """
  Generate a range of valid axis arguments for a reduction over
  an array with a given number of dimensions.
  """
  yield from (None, ())
  if ndim > 0:
    yield from (0, (-1,))
  if ndim > 1:
    yield from (1, (0, 1), (-1, 0))
  if ndim > 2:
    yield (-1, 0, 1)


def _random_invertible(rng, shape, dtype):
  """
  Generate a random invertible matrix was specified shape and dtype
  """
  while True:
    a = rng(shape, dtype)
    try:
      np.linalg.inv(a)
    except np.linalg.LinAlgError:
      pass
    else:
      return a


def osp_linalg_toeplitz(c: np.ndarray, r: np.ndarray | None = None) -> np.ndarray:
  """scipy.linalg.toeplitz with v1.17+ batching semantics."""
  # scipy 1.17 doesn't support zero batch size: https://github.com/scipy/scipy/pull/24151
  zero_batch = (0 in c.shape[:-1]) or (r is not None and 0 in r.shape[:-1])
  if scipy_version >= (1, 17, 0) and not zero_batch:
    return scipy.linalg.toeplitz(c, r)
  elif r is None:
    c = np.atleast_1d(c)
    return np.vectorize(
      scipy.linalg.toeplitz, signature="(m)->(m,m)", otypes=(c.dtype,))(c)
  else:
    c = np.atleast_1d(c)
    r = np.atleast_1d(r)
    return np.vectorize(
      scipy.linalg.toeplitz, signature="(m),(n)->(m,n)", otypes=(np.result_type(c, r),))(c, r)



def osp_linalg_hankel(c: np.ndarray, r: np.ndarray | None = None) -> np.ndarray:
  """Batched scipy hankel for testing."""
  if scipy_version >= (1, 19):
    if r is None:
      return scipy.linalg.hankel(c)
    return scipy.linalg.hankel(c, r)
  if r is None:
    r = np.zeros_like(c)
  return np.vectorize(
      scipy.linalg.hankel, signature="(m),(n)->(m,n)",
      otypes=(np.result_type(c.dtype, r.dtype),))(c, r)

def svd_algorithms():
  algorithms = [None]
  if jtu.device_under_test() in ["cpu", "gpu"]:
    algorithms.append(lax.linalg.SvdAlgorithm.QR)
  if jtu.device_under_test() == "cpu":
    algorithms.append(lax.linalg.SvdAlgorithm.DIVIDE_AND_CONQUER)
  if jtu.device_under_test() == "gpu":
    algorithms.append(lax.linalg.SvdAlgorithm.JACOBI)
    algorithms.append(lax.linalg.SvdAlgorithm.DIVIDE_AND_CONQUER)
  if jtu.device_under_test() == "tpu" or jtu.device_under_test() == "gpu":
    algorithms.append(lax.linalg.SvdAlgorithm.POLAR)
  return algorithms




# (complex) singular vectors are only unique up to an arbitrary phase. This makes the gradient
# tests based on finite differences unstable, since perturbing the input matri may cause an
# arbitrary sign flip of one or more of the singular vectors. To remedy this, we normalize the
# singular vectors such that the first component of the left singular vectors has phase 0.
def _normalizing_svd(a: np.array, full_matrices: bool):
  u, s, vt = jnp.linalg.svd(a, full_matrices=full_matrices, compute_uv=True)
  top_rows = u[..., 0:1, :]
  if np.issubdtype(a.dtype, np.complexfloating):
    angle = -jnp.angle(top_rows)
    u_phase = lax.complex(jnp.cos(angle), jnp.sin(angle))
    v_phase = lax.complex(jnp.cos(-angle), jnp.sin(-angle))
  else:
    u_phase = jnp.sign(top_rows)
    v_phase = u_phase
  u *= u_phase
  vt *= np.swapaxes(v_phase, -1, -2)
  return u, s, vt


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

    np_fun = partial(np.linalg.cholesky, upper=upper)
    jnp_fun = partial(jnp.linalg.cholesky, upper=upper, symmetrize_input=True)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=1e-3)
    self._CompileAndCheck(jnp_fun, args_maker)

    if jnp.finfo(dtype).bits == 64:
      jtu.check_grads(jnp.linalg.cholesky, args_maker(), order=2)
      jtu.check_grads(
          partial(jnp.linalg.cholesky, upper=upper, symmetrize_input=False),
          args_maker(),
          order=2,
      )
      jtu.check_grads(
          partial(jsp.linalg.cholesky, lower=not upper), args_maker(), order=2
      )

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
    shape=[(1, 1), (2, 2), (3, 3), (2, 2, 2), (2, 3, 3), (2, 4, 4), (5, 7, 7)],
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

  def testDet3x3NumericalStabilityIssue39905(self):
    a = jnp.array([[41932.215, 41927.867, 41927.867],
                   [41927.867, 41932.215, 41927.867],
                   [41927.867, 41927.867, 41932.215]], dtype=np.float32)
    expected = np.linalg.det(np.asarray(a, dtype=np.float64))
    self.assertAllClose(jnp.linalg.det(a), expected, rtol=1e-3, check_dtypes=False)

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
                          tol={np.float32: 1e-5, np.float64: 1e-13})

  def testTensorsolveAxes(self):
    a_shape = (2, 1, 3, 6)
    b_shape = (1, 6)
    axes = (0, 2)
    dtype = "float32"

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    np_fun = partial(np.linalg.tensorsolve, axes=axes)
    jnp_fun = partial(jnp.linalg.tensorsolve, axes=axes)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(dtype=dtype, method=method)
     for dtype in float_types + complex_types
     for method in (["lu"] if jnp.issubdtype(dtype, jnp.complexfloating)
                     else ["lu", "qr"])
    ],
    shape=[(0, 0), (1, 1), (3, 3), (4, 4), (10, 10), (200, 200), (2, 2, 2),
           (2, 3, 3), (3, 2, 2)],
  )
  @jtu.ignore_warning(message="(divide by zero|overflow|invalid value)", category=RuntimeWarning)
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
  @jtu.run_on_devices("cpu", "gpu")
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

    implementations = [None]

    if (
        jtu.is_device_cuda()
        and not compute_left_eigenvectors
        and cuda_versions
        and cuda_versions.cusolver_get_version() >= 11701
    ):
      implementations.append(jax.lax.linalg.EigImplementation.CUSOLVER)

    for implementation in implementations:
      results = lax.linalg.eig(
          a, compute_left_eigenvectors=compute_left_eigenvectors,
          compute_right_eigenvectors=compute_right_eigenvectors,
          implementation=implementation)
      w = results[0]

      if compute_left_eigenvectors:
        check_left_eigenvectors(a, w, results[1])
      if compute_right_eigenvectors:
        check_right_eigenvectors(a, w, results[1 + compute_left_eigenvectors])

    self._CompileAndCheck(partial(jnp.linalg.eig), args_maker, rtol=1e-3)

  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (50, 50), (2, 6, 6)],
    dtype=float_types + complex_types,
    compute_left_eigenvectors=[False, True],
    compute_right_eigenvectors=[False, True],
  )
  @jtu.run_on_devices("cpu", "gpu")
  def testEigHandlesNanInputs(self, shape, dtype, compute_left_eigenvectors,
                              compute_right_eigenvectors):
    """Verifies that `eig` fails gracefully if given non-finite inputs."""
    if jtu.is_device_cuda():
      # TODO(phawkins): CUSOLVER's implementation does not pass this test.
      implementation = jax.lax.linalg.EigImplementation.LAPACK
    else:
      implementation = None
    a = jnp.full(shape, jnp.nan, dtype)
    results = lax.linalg.eig(
        a, compute_left_eigenvectors=compute_left_eigenvectors,
        compute_right_eigenvectors=compute_right_eigenvectors,
        implementation=implementation)
    for result in results:
      self.assertTrue(np.all(np.isnan(result)))

  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (8, 8), (7, 6, 6)],
    dtype=float_types + complex_types,
 )
  @jtu.run_on_devices("cpu", "gpu")
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
    shape=[(4, 4), (5, 5)],
    dtype=complex_types,
    left_right=[(False, True), (True, False), (True, True)],
  )
  @jtu.run_on_devices("cpu", "gpu")
  def testEigGradComplexInputs(self, shape, dtype, left_right):
    # Small matrices only; the eigenvector derivative blows up like
    # 1/min|w_i - w_j| so close eigenvalues make the FD check noisy.
    left, right = left_right
    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    tol = 1e-3 if dtype == np.complex128 else 5e-1
    f = partial(lax.linalg.eig, compute_left_eigenvectors=left,
                compute_right_eigenvectors=right, enable_eigvec_derivs=True)
    jtu.check_grads(f, (a,), order=2, rtol=tol, atol=tol)

  @jtu.sample_product(
    shape=[(4, 4), (5, 5)],
    dtype=float_types,
    left_right=[(False, True), (True, False), (True, True)],
  )
  @jtu.run_on_devices("cpu", "gpu")
  def testEigGradRealInputs(self, shape, dtype, left_right):
    # dgeev does not pin the sign of complex-pair eigenvectors so the primal
    # output is itself discontinuous; compose with elementwise v -> v*v which
    # is sign-invariant but still phase-sensitive (so still exercises the
    # gauge-fixing part of the JVP).
    left, right = left_right
    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    tol = 1e-3 if dtype == np.float64 else 5e-1
    def f(a):
      out = lax.linalg.eig(a, compute_left_eigenvectors=left,
                           compute_right_eigenvectors=right,
                           enable_eigvec_derivs=True)
      return (out[0],) + tuple(v * v for v in out[1:])
    jtu.check_grads(f, (a,), order=2, rtol=tol, atol=tol)

  @jtu.sample_product(
    shape=[(4, 4), (5, 5), (50, 50)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu", "gpu")
  def testEigvals(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    a, = args_maker()
    result = jnp.linalg.eig(a)
    # Check that eig returns a namedtuple with the right fields
    self.assertTrue(hasattr(result, 'eigenvalues'))
    self.assertTrue(hasattr(result, 'eigenvectors'))
    self.assertIs(result.eigenvalues, result[0])
    self.assertIs(result.eigenvectors, result[1])
    w1 = result.eigenvalues
    w2 = jnp.linalg.eigvals(a)
    self.assertAllClose(w1, w2, rtol={np.complex64: 1e-5, np.complex128: 2e-14})

  @jtu.run_on_devices("cpu", "gpu")
  def testEigvalsInf(self):
    # https://github.com/jax-ml/jax/issues/2661
    x = jnp.array([[jnp.inf]])
    self.assertTrue(jnp.all(jnp.isnan(jnp.linalg.eigvals(x))))

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (5, 5)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu", "gpu")
  def testEigBatching(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    shape = (10,) + shape
    args = rng(shape, dtype)
    ws, vs = vmap(jnp.linalg.eig)(args)
    self.assertTrue(np.all(np.linalg.norm(
        np.matmul(args, vs) - ws[..., None, :] * vs) < 1e-3))


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
    np_fn = partial(np.linalg.matrix_norm, ord=ord, keepdims=keepdims)
    jnp_fn = partial(jnp.linalg.matrix_norm, ord=ord, keepdims=keepdims)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-3)
    self._CompileAndCheck(jnp_fn, args_maker)

  @jtu.sample_product(
      shape=[(0, 2), (2, 0), (0, 0)],
      dtype=float_types + complex_types,
      ord=[1, 2, np.inf, 'fro', 'nuc'],
  )
  def testEmptyMatrixNorm(self, shape, dtype, ord):
    x = jnp.zeros(shape, dtype)
    norm = jnp.linalg.matrix_norm(x, ord=ord)
    self.assertEqual(norm, 0)

  @jtu.sample_product(
      [
        dict(shape=shape, axis=axis)
        for shape in [(3,), (3, 4), (2, 3, 4, 5)]
        for axis in _axis_for_ndim(len(shape))
      ],
      dtype=float_types + complex_types,
      keepdims=[True, False],
      ord=[1, -1, 2, -2, np.inf, -np.inf],
  )
  def testVectorNorm(self, shape, dtype, keepdims, axis, ord):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fn = partial(np.linalg.vector_norm, ord=ord, keepdims=keepdims, axis=axis)
    jnp_fn = partial(jnp.linalg.vector_norm, ord=ord, keepdims=keepdims, axis=axis)
    tol = 1E-3 if jtu.test_device_matches(['tpu']) else None
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  @jtu.sample_product(
      dtype=float_types + complex_types,
      ord=[1, 2, np.inf],
  )
  def testEmptyVectorNorm(self, dtype, ord):
    x = jnp.zeros(0, dtype)
    norm = jnp.linalg.vector_norm(x, ord=ord)
    self.assertEqual(norm, 0)

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
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testVecdot(self, lhs_shape, rhs_shape, axis, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    np_fn = np.linalg.vecdot
    np_fn = jtu.promote_like_jnp(partial(np_fn, axis=axis))
    jnp_fn = partial(jnp.linalg.vecdot, axis=axis)
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

    # smoke-test for optional kwargs.
    jnp_fn = partial(jnp.linalg.vecdot, axis=axis,
                     precision=lax.Precision.HIGHEST,
                     preferred_element_type=dtype)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)

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
    np_fn = jtu.promote_like_jnp(np.linalg.matmul)
    jnp_fn = jnp.linalg.matmul
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

    # smoke-test for optional kwargs.
    jnp_fn = partial(jnp.linalg.matmul,
                     precision=lax.Precision.HIGHEST,
                     preferred_element_type=dtype)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)

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
    np_fn = jtu.promote_like_jnp(partial(np.linalg.tensordot, axes=axes))
    jnp_fn = partial(jnp.linalg.tensordot, axes=axes)
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

    # smoke-test for optional kwargs.
    jnp_fn = partial(jnp.linalg.tensordot, axes=axes,
                     precision=lax.Precision.HIGHEST,
                     preferred_element_type=dtype)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)

  @parameterized.product(
      jtu.sample_product_testcases(
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
      ),
      algorithm=svd_algorithms()
  )
  @jax.default_matmul_precision("float32")
  def testSVD(self, b, m, n, dtype, full_matrices, compute_uv, hermitian,
              algorithm):
    if hermitian and algorithm is not None:
      # Hermitian SVD doesn't support the algorithm parameter.
      self.skipTest("Hermitian SVD doesn't support the algorithm parameter")

    if jtu.is_device_rocm() and algorithm == lax.linalg.SvdAlgorithm.POLAR:
      self.skipTest("ROCM polar SVD not implemented")

    if jtu.is_device_oneapi() and algorithm == lax.linalg.SvdAlgorithm.JACOBI:
      self.skipTest("OneAPI does not support Jacobi SVD")

    if jtu.is_device_oneapi() and algorithm == lax.linalg.SvdAlgorithm.POLAR:
      self.skipTest("OneAPI does not support Polar SVD")

    if (not jtu.is_device_rocm() and jtu.device_under_test() == "gpu"
        and algorithm == lax.linalg.SvdAlgorithm.DIVIDE_AND_CONQUER):
      self.skipTest("Divide-and-conquer SVD only supported on AMD (ROCm) GPUs")

    if (
        jtu.test_device_matches(["cuda"])
        and (algorithm, m, n) in [
          (lax.linalg.SvdAlgorithm.POLAR, 400000, 2),
          (lax.linalg.SvdAlgorithm.POLAR, 2, 400000),
          (lax.linalg.SvdAlgorithm.JACOBI, 400000, 2),
          (lax.linalg.SvdAlgorithm.JACOBI, 2, 400000),
        ]
    ):
      # Test fails with CUDA polar and jacobi decompositions
      self.skipTest("Test fails with CUDA polar and jacobi decompositions")

    def compute_max_backward_error(operand, reconstructed_operand):
      error_norm = np.linalg.norm(operand - reconstructed_operand,
                                  axis=(-2, -1))
      backward_error = (error_norm /
                        np.linalg.norm(operand, axis=(-2, -1)))
      max_backward_error = np.amax(backward_error)
      return max_backward_error

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(b + (m, n), dtype)]

    tol = 100 * jnp.finfo(dtype).eps
    reconstruction_tol = 2 * tol
    unitariness_tol = 3 * tol

    a, = args_maker()
    if hermitian:
      a = a + np.conj(T(a))
    if algorithm is None:
      fun = partial(jnp.linalg.svd, hermitian=hermitian)
    else:
      fun = partial(lax.linalg.svd, algorithm=algorithm)
    out = fun(a, full_matrices=full_matrices, compute_uv=compute_uv)
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
      self.assertAllClose(np.linalg.svd(a, compute_uv=False), np.asarray(out),
                          atol=1e-4, rtol=3e-4)

    self._CompileAndCheck(partial(fun, full_matrices=full_matrices,
                                  compute_uv=compute_uv),
                          args_maker)

    if not compute_uv and a.size < 100000:
      svd = partial(fun, full_matrices=full_matrices, compute_uv=compute_uv)
      # TODO(phawkins): these tolerances seem very loose.
      if dtype == np.complex128:
        jtu.check_jvp(svd, partial(jvp, svd), (a,), rtol=1e-4, atol=1e-4,
                      eps=1e-8)
      else:
        jtu.check_jvp(svd, partial(jvp, svd), (a,), rtol=5e-2, atol=2e-1)

    if compute_uv and (not full_matrices):
      d, = args_maker()
      def f(x):
        u, s, v = jnp.linalg.svd(
          a + x * d,
          full_matrices=full_matrices,
          compute_uv=compute_uv)
        vdiag = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')
        return jnp.matmul(jnp.matmul(u, vdiag(s).astype(u.dtype)), v).real
      _, t_out = jvp(f, (1.,), (1.,))
      if dtype == np.complex128:
        tol = 2e-13
      else:
        tol = 6e-4
      self.assertArraysAllClose(t_out, d.real, atol=tol, rtol=tol)

  def testJspSVDBasic(self):
    # since jax.scipy.linalg.svd is almost the same as jax.numpy.linalg.svd
    # do not check it functionality here
    jsp.linalg.svd(np.ones((2, 2), dtype=np.float32))

  @jtu.sample_product(
      shape=[(1, 1), (4, 4), (2, 5), (5, 2), (5, 5), (2, 5, 5)],
      dtype=float_types + complex_types,
      full_matrices=[True, False],
      compute_uv=[True, False],
  )
  @jax.default_matmul_precision("float32")
  def testSVDGrad(self, shape, dtype, full_matrices, compute_uv):
    rng = jtu.rand_default(self.rng())
    a = rng(shape, dtype)
    if not compute_uv:
      f = partial(jnp.linalg.svd, full_matrices=False, compute_uv=False)
    else:
      f = partial(_normalizing_svd, full_matrices=full_matrices)
    if full_matrices and shape[-1] != shape[-2]:
      self.skipTest("JVP for SVD not implemented for full matrices.")

    jtu.check_grads(f, (a,), order=2, rtol=0.035, eps=1.0 / 512)

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
      max_norm = 220 if jtu.is_device_tpu(7, 'x') else 160
      if not np.all(nm < max_norm):
        self.fail(f"norm={np.amax(nm)}")

    # Check a ~= qr
    norm_error = norm(a - np.matmul(lq, lr))
    if not np.all(norm_error < 60):
      self.fail(np.amax(norm_error))

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

    jtu.check_jvp(qr_and_mul, partial(jvp, qr_and_mul), (a,), atol=3e-3)

  @jtu.skip_on_devices("tpu")
  def testQrInvalidDtypeCPU(self, shape=(5, 6), dtype=np.float16):
    # Regression test for https://github.com/jax-ml/jax/issues/10530
    rng = jtu.rand_default(self.rng())
    arr = rng(shape, dtype)
    err, msg = TypeError, "does not accept dtype float16"
    with self.assertRaisesRegex(err, msg):
      jax.block_until_ready(jnp.linalg.qr(arr))

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
  # Unskipping test for ROCm while leaving
  # original skip in place for other GPUs as per
  # commit: e81024f5053def119eddb7fb06ff6c4f7b5948a8
  #
  # Original note: TODO(#2203): numerical errors
  @jtu.skip_on_devices("cuda")
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
      with self.assertRaises(ValueError):
        jnp.linalg.cond(*args_maker())
    else:
      self._CheckAgainstNumpy(np.linalg.cond, jnp.linalg.cond, args_maker,
                              check_dtypes=False, tol=1e-3)
      partial_norm = partial(jnp.linalg.cond, p=pnorm)
      self._CompileAndCheck(partial_norm, lambda: [gen_mat()],
                            check_dtypes=False, rtol=1e-03, atol=1e-03)

  @jtu.sample_product(
    pnorm=[jnp.inf, -jnp.inf, 1, -1, 2, -2, 'fro', None],
    dtype=float_types + complex_types,
    batched=[True, False],
  )
  def testCondSingular(self, pnorm, dtype, batched):
    # Singular matrices have an infinite condition number for every norm.
    singular = np.zeros((2, 2), dtype=dtype)
    x = np.stack([singular, np.eye(2, dtype=dtype)]) if batched else singular
    args_maker = lambda: [x, pnorm]
    self._CheckAgainstNumpy(np.linalg.cond, jnp.linalg.cond, args_maker,
                            check_dtypes=False)
    partial_cond = partial(jnp.linalg.cond, p=pnorm)
    self._CompileAndCheck(partial_cond, lambda: [x], check_dtypes=False)

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (6, 2, 3), (3, 4, 2, 6)],
    dtype=float_types + complex_types,
  )
  def testTensorinv(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = partial(np.linalg.tensorinv, ind=len(shape) // 2)
    jnp_fun = partial(jnp.linalg.tensorinv, ind=len(shape) // 2)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=1E-4)
    self._CompileAndCheck(jnp_fun, args_maker)

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
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
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
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testSolveBroadcasting(self, lhs_shape, rhs_shape):
    # Batched solve can involve some ambiguities; this test checks
    # that we match NumPy's convention in all cases.
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, 'float32'), rng(rhs_shape, 'float32')]
    self._CheckAgainstNumpy(np.linalg.solve, jnp.linalg.solve, args_maker, tol=1E-3)
    self._CompileAndCheck(jnp.linalg.solve, args_maker)

  @jtu.sample_product(
    shape=[(1, 1), (4, 4), (2, 5, 5), (100, 100), (5, 5, 5), (0, 0)],
    dtype=float_types,
  )
  def testInv(self, shape, dtype):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      a = _random_invertible(rng=rng, shape=shape, dtype=dtype)
      return [a]

    self._CheckAgainstNumpy(np.linalg.inv, jnp.linalg.inv, args_maker,
                            tol=1e-3)
    self._CompileAndCheck(jnp.linalg.inv, args_maker)

  @jtu.run_on_devices('cpu')
  def testInvDtypeError(self):
    # regression test for https://github.com/jax-ml/jax/issues/38825
    x = jnp.ones((3, 3), dtype='float16')
    with self.assertRaisesRegex(TypeError,
                                "lu does not accept dtype float16"):
      jsp.linalg.inv(x)

  def testLinalgPrimitivesInvalidDtype(self):
    x = jnp.ones((3, 3), dtype=jnp.float16)
    with self.assertRaisesRegex(TypeError, "cholesky does not accept dtype float16"):
      jnp.linalg.cholesky(x)
    with self.assertRaisesRegex(TypeError, "eigh does not accept dtype float16"):
      jnp.linalg.eigh(x)
    with self.assertRaisesRegex(TypeError, "svd does not accept dtype float16"):
      jnp.linalg.svd(x)
    with self.assertRaisesRegex(TypeError, "eig does not accept dtype float16"):
      jnp.linalg.eig(x)


  @jtu.sample_product(
    [dict(shape=shape, hermitian=hermitian)
     for shape in [(1, 1), (4, 4), (3, 10, 10), (2, 70, 7), (2000, 7),
                   (7, 1000), (70, 7, 2), (2, 0, 0), (3, 0, 2), (1, 0),
                   (400000, 2), (2, 400000)]
     for hermitian in ([False, True] if shape[-1] == shape[-2] else [False])],
    dtype=float_types + complex_types,
  )
  @jtu.ignore_warning(message="invalid value", category=RuntimeWarning)
  def testPinv(self, shape, hermitian, dtype):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      a = rng(shape, dtype)
      if hermitian:
        a = (a + T(a.conj())) / 2
      return [a]

    np_fn = partial(np.linalg.pinv, hermitian=hermitian)
    jnp_fn = partial(jnp.linalg.pinv, hermitian=hermitian)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-5)

    # TODO(phawkins): 6e-2 seems like a very loose tolerance.
    jtu.check_grads(jnp_fn, args_maker(), 1, rtol=6e-2, atol=1e-3)

  def testPinvRcond(self):
    x = jnp.ones((3, 3))
    with self.assertRaisesWithLiteralMatch(
        ValueError, "pinv: only one of rtol and rcond may be specified."):
      jnp.linalg.pinv(x, rcond=1E-2, rtol=1E-2)
    self.assertArraysEqual(
        jnp.linalg.pinv(x, rcond=1E-2),
        jnp.linalg.pinv(x, rtol=1E-2)
    )

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

  def testMatrixPowerBool(self):
    # Regression test for https://github.com/jax-ml/jax/issues/28603
    mat = np.array([[True,True], [False,True]])
    np_result = np.linalg.matrix_power(mat, 2)
    jnp_result = jnp.linalg.matrix_power(mat, 2)
    self.assertArraysEqual(np_result, jnp_result)

  @jtu.sample_product(
    [dict(shape=shape, hermitian=hermitian)
     for shape in [(3, ), (1, 2), (8, 5), (4, 4), (5, 5), (50, 50), (3, 4, 5),
                   (2, 3, 4, 5)]
     for hermitian in ([False, True] if len(shape) >= 2 and shape[-1] == shape[-2] else [False])],
    dtype=float_types + complex_types,
  )
  def testMatrixRank(self, shape, dtype, hermitian):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      a = rng(shape, dtype)
      if hermitian:
        a = (a + T(a.conj())) / 2
      return [a]

    np_fn = partial(np.linalg.matrix_rank, hermitian=hermitian)
    jnp_fn = partial(jnp.linalg.matrix_rank, hermitian=hermitian)
    self._CheckAgainstNumpy(np_fn, jnp_fn,
                            args_maker, check_dtypes=False, tol=1e-3)
    self._CompileAndCheck(jnp_fn, args_maker,
                          check_dtypes=False, rtol=1e-3)

  def testMatrixRankTol(self):
    x = jnp.ones((3, 3))
    with self.assertRaisesWithLiteralMatch(
        ValueError, "matrix_rank: only one of tol or rtol may be specified."):
      jnp.linalg.matrix_rank(x, rtol=1E-2, tol=1E-2)
    self.assertArraysEqual(
        jnp.linalg.matrix_rank(x, rtol=1E-2),
        jnp.linalg.matrix_rank(x, tol=1E-2)
    )

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

  @jtu.sample_product(
      shape=[(2, 1), (2, 2), (1, 2)]
  )
  def testLstsqZeroMatrix(self, shape):
    # Regression test for https://github.com/jax-ml/jax/issues/32666
    args_maker = lambda: [np.zeros(shape), np.ones(shape)]
    np_fun = np.linalg.lstsq
    jnp_fun = partial(jnp.linalg.lstsq, numpy_resid=True)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  # Regression test for incorrect type for eigenvalues of a complex matrix.
  def testIssue669(self):
    def test(x):
      val, vec = jnp.linalg.eigh(x)
      return jnp.real(jnp.sum(val))

    grad_test_jc = jit(grad(jit(test)))
    xc = np.eye(3, dtype=np.complex64)
    self.assertAllClose(xc, grad_test_jc(xc))

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testIssue1151(self):
    rng = self.rng()
    A = jnp.array(rng.randn(100, 3, 3), dtype=jnp.float32)
    b = jnp.array(rng.randn(100, 3, 1), dtype=jnp.float32)
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
    np_fun = jtu.promote_like_jnp(partial(np.linalg.cross, axis=axis))
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
    np_fun = jtu.promote_like_jnp(np.linalg.outer)
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
    np_fun = partial(np.linalg.diagonal, offset=offset)
    self._CheckAgainstNumpy(np_fun, lax_fun, args_maker)
    self._CompileAndCheck(lax_fun, args_maker)

  def testTrace(self):
    shape, dtype, offset, out_dtype = (3, 4), "float32", 0, None
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    lax_fun = partial(jnp.linalg.trace, offset=offset, dtype=out_dtype)
    np_fun = partial(np.linalg.trace, offset=offset)
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
    # Regression test for https://github.com/jax-ml/jax/issues/19076
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
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
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
       # batched lu, unbatched b (vector)
       ((3, 4, 4), (4,)),
       # batched lu, unbatched b (matrix)
       ((3, 8, 8), (8, 4)),
       # unbatched lu, batched b (matrix) — broadcast lu -> batch
       ((4, 4), (3, 4, 2)),
     ]
    ],
    trans=[0, 1, 2],
    dtype=float_types + complex_types,
  )
  @jtu.skip_on_devices("cpu")  # TODO(frostig): Test fails on CPU sometimes
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLuSolveBroadcast(self, lhs_shape, rhs_shape, dtype, trans):
    if scipy_version < (1, 16, 0):
      self.skipTest("scipy.linalg.lu_solve batch broadcasting requires scipy >= 1.16")
    rng = jtu.rand_default(self.rng())
    osp_fun = lambda lu, piv, rhs: osp.linalg.lu_solve((lu, piv), rhs, trans=trans)
    jsp_fun = lambda lu, piv, rhs: jsp.linalg.lu_solve((lu, piv), rhs, trans=trans)
    lu_factor_vec = np.vectorize(osp.linalg.lu_factor,
                                  signature="(n,n)->(n,n),(n)")

    def args_maker():
      a = rng(lhs_shape, dtype)
      lu, piv = lu_factor_vec(a) if a.ndim > 2 else osp.linalg.lu_factor(a)
      return [lu, piv, rng(rhs_shape, dtype)]

    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, tol=1e-3)
    self._CompileAndCheck(jsp_fun, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [
        ((1, 1), (1, 1)),
        ((4, 4), (4,)),
        ((8, 8), (8, 4)),
        ((3, 4, 4), (3, 4, 4)),
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
    if scipy_version < (1, 17, 0) and len(lhs_shape) > 2:
      self.skipTest("scipy 1.17 or newer required for batched solve.")
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

    osp_fun = partial(osp.linalg.solve_triangular, trans=1 if transpose_a else 0,
                      lower=lower, unit_diagonal=unit_diagonal)
    jsp_fun = partial(jsp.linalg.solve_triangular, trans=1 if transpose_a else 0,
                      lower=lower, unit_diagonal=unit_diagonal)
    args_maker = lambda: [a, b]
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            rtol={np.float32: 1e-4, np.float64: 1e-11})
    self._CompileAndCheck(jsp_fun, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [
        # matching batch dims
        ((2, 8, 8), (2, 8, 10)),
        # batched a, unbatched b (vector) — broadcast b
        ((3, 4, 4), (4,)),
        # batched a, unbatched b (matrix)
        ((3, 4, 4), (4, 2)),
        # unbatched a, batched b (matrix) — broadcast a
        ((4, 4), (3, 4, 2)),
      ]
    ],
    lower=[False, True],
    dtype=float_types,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testSolveTriangularBroadcast(self, lower, lhs_shape, rhs_shape, dtype):
    if scipy_version < (1, 16, 0):
      self.skipTest("scipy.linalg.solve_triangular batch broadcasting requires scipy >= 1.16")
    rng = jtu.rand_default(self.rng())
    k = rng(lhs_shape, dtype)
    l = np.linalg.cholesky(
        np.matmul(k, T(k)) + lhs_shape[-1] * np.eye(lhs_shape[-1]))
    l = l.astype(k.dtype)
    a = l if lower else T(l)
    b = rng(rhs_shape, dtype)
    osp_fun = partial(osp.linalg.solve_triangular, lower=lower)
    jsp_fun = partial(jsp.linalg.solve_triangular, lower=lower)
    args_maker = lambda: [a, b]
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            rtol={np.float32: 1e-4, np.float64: 1e-11})
    self._CompileAndCheck(jsp_fun, args_maker)

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

  def testTriangularSolveSingularBatched(self):
    x = jnp.array([[1, 1], [0, 0]], dtype=np.float32)
    y = jnp.array([[1], [1.]], dtype=np.float32)
    out = jax.lax.linalg.triangular_solve(x[None], y[None], left_side=True)
    # x is singular. The triangular solve may contain either nans or infs, but
    # it should not consist of only finite values.
    self.assertFalse(np.all(np.isfinite(out)))

  @jtu.sample_product(
    n=[1, 4, 5, 20, 50, 100],
    batch_size=[(), (2,), (3, 4)],
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
    dtype=float_types + complex_types,
    mode=["full", "r", "economic"],
    pivoting=[False, True]
  )
  @jax.default_matmul_precision("float32")
  def testScipyQrModes(self, shape, dtype, mode, pivoting):
    if pivoting:
      if not jtu.test_device_matches(["cpu", "gpu"]):
        self.skipTest("Pivoting is only supported on CPU and GPU.")
    rng = jtu.rand_default(self.rng())
    jsp_func = partial(jax.scipy.linalg.qr, mode=mode, pivoting=pivoting)
    sp_func = partial(scipy.linalg.qr, mode=mode, pivoting=pivoting)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(sp_func, jsp_func, args_maker, rtol=1E-5, atol=1E-5)
    self._CompileAndCheck(jsp_func, args_maker)

    # Pivoting is unsupported by the numpy api - repeat the jvp checks performed
    # in NumpyLinalgTest::testQR for the `pivoting=True` modes here. Like in the
    # numpy test, `qr_and_mul` expresses the identity function.
    def qr_and_mul(a):
      q, r, *p = jsp_func(a)
      # To express the identity function we must "undo" the pivoting of `q @ r`.
      inverted_pivots = jnp.argsort(p[0])
      return (q @ r)[:, inverted_pivots]

    m, n = shape
    if pivoting and mode != "r" and (m == n or (m > n and mode != "full")):
      for a in args_maker():
        jtu.check_jvp(qr_and_mul, partial(jvp, qr_and_mul), (a,), atol=3e-3)

  @jtu.sample_product(
      shape=[(4, 3), (4, 4), (6, 4), (3, 6, 4)],
      dtype=float_types + complex_types,
      mode=['right', 'left'],
      pivoting=[False, True],
  )
  def testQrMultiply(self, shape, dtype, mode, pivoting):
    if pivoting and not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("Pivoting is only supported on CPU and GPU.")
    rng = jtu.rand_default(self.rng())
    m, n = shape[-2:]
    k = min(m, n)
    c_shape = shape[:-2] + (k, 2) if mode == 'left' else shape[:-2] + (2, m)
    args_maker = lambda: [rng(shape, dtype), rng(c_shape, dtype)]
    fn = partial(jax.scipy.linalg.qr_multiply, mode=mode, pivoting=pivoting)

    a, c = args_maker()
    if pivoting:
      result, r, p = fn(a, c)
    else:
      result, r = fn(a, c)

    # Compute reference using full QR
    if pivoting:
      q_full, r_full, p_ref = jax.scipy.linalg.qr(a, pivoting=True)
    else:
      q_full, r_full = jax.scipy.linalg.qr(a)
    r_ref = r_full[..., :k, :]

    with jax.default_matmul_precision("highest"):
      if mode == 'left':
        expected = q_full[..., :k] @ c  # Q_eco @ c: (M,K) @ (K,2) = (M,2)
      else:
        expected = c @ q_full[..., :k]  # c @ Q_eco: (2,M) @ (M,K) = (2,K)

    tol = {np.float32: 1e-5, np.complex64: 1e-5, np.float64: 1e-12,
           np.complex128: 1e-12}
    self.assertAllClose(result, expected, rtol=tol, atol=tol)
    # R should match
    self.assertAllClose(r, r_ref, rtol=tol, atol=tol)
    self._CompileAndCheck(fn, args_maker)

  @jtu.sample_product(
      shape=[(4, 3), (4, 4), (6, 4)],
      dtype=float_types + complex_types,
      mode=['right', 'left'],
  )
  def testQrMultiply1D(self, shape, dtype, mode):
    rng = jtu.rand_default(self.rng())
    m, n = shape[-2:]
    k = min(m, n)
    c_shape = (k,) if mode == 'left' else (m,)
    args_maker = lambda: [rng(shape, dtype), rng(c_shape, dtype)]
    fn = partial(jax.scipy.linalg.qr_multiply, mode=mode)

    a, c = args_maker()
    result, r = fn(a, c)
    self.assertEqual(result.ndim, 1)

    q_full, _ = jax.scipy.linalg.qr(a)
    with jax.default_matmul_precision("highest"):
      if mode == 'left':
        expected = (q_full[..., :k] @ c[:, None]).ravel()  # Q_eco @ c: (M,)
      else:
        expected = (c[None, :] @ q_full[..., :k]).ravel()  # c @ Q_eco: (K,)

    tol = {np.float32: 1e-5, np.complex64: 1e-5, np.float64: 1e-12,
           np.complex128: 1e-12}
    self.assertAllClose(result, expected, rtol=tol, atol=tol)
    self._CompileAndCheck(fn, args_maker)

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
      [dict(a_shape=a_shape, c_shape=c_shape, left=left)
       for a_shape, c_shape, left in [
           ((4, 3), (4, 2), True),    # Q @ C, tall A
           ((4, 4), (4, 3), True),    # Q @ C, square A
           ((6, 4), (6, 5), True),    # Q @ C, tall A
           ((4, 3), (2, 4), False),   # C @ Q, tall A
           ((4, 4), (3, 4), False),   # C @ Q, square A
           ((3, 6, 4), (3, 6, 5), True),  # batched Q @ C
       ]],
      dtype=float_types + complex_types,
      transpose=[False, True],
  )
  def testOrmqr(self, a_shape, c_shape, dtype, left, transpose):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(a_shape, dtype), rng(c_shape, dtype)]

    def fn(a, c):
      qr_result, taus = lax_linalg.geqrf(a)
      return lax_linalg.ormqr(qr_result, taus, c, left=left, transpose=transpose)

    a, c = args_maker()
    qr_result, taus = lax_linalg.geqrf(a)
    result = lax_linalg.ormqr(qr_result, taus, c, left=left, transpose=transpose)

    # Reference: build full Q from householder_product, then matmul.
    m, n = a_shape[-2:]
    if m > n:
      padded = jnp.pad(qr_result,
                        [(0, 0)] * (qr_result.ndim - 1) + [(0, m - n)])
      q = lax.linalg.householder_product(padded, taus)
    elif m < n:
      q = lax.linalg.householder_product(qr_result[..., :m, :m], taus)
    else:
      q = lax.linalg.householder_product(qr_result, taus)
    q_op = jnp.conj(jnp.swapaxes(q, -1, -2)) if transpose else q
    with jax.default_matmul_precision("highest"):
      expected = q_op @ c if left else c @ q_op

    tol = {np.float32: 1e-5, np.complex64: 1e-5, np.float64: 1e-12,
           np.complex128: 1e-12}
    self.assertAllClose(result, expected, rtol=tol, atol=tol)
    self._CompileAndCheck(fn, args_maker)

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

    if len(shape) == 3:
      args = args_maker()
      self.assertAllClose(jax.vmap(jsp_func)(*args), jsp_func(*args))

  @jtu.sample_product(
      shape=[(1, 1), (2, 2, 2), (4, 4), (10, 10), (2, 5, 5)],
      dtype=float_types + complex_types,
      lower=[False, True],
  )
  @jtu.skip_on_devices("tpu")
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

    if len(shape) == 3:
      args = args_maker()
      self.assertAllClose(jax.vmap(jax_func)(*args), jax_func(*args))

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
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [
          # batched c, unbatched b (vector)
          [(3, 4, 4), (4,)],
          # batched c, unbatched b (matrix)
          [(3, 4, 4), (4, 2)],
          # batched c, batched b
          [(3, 4, 4), (3, 4, 2)],
          # unbatched c, batched b (broadcast c -> batch)
          [(4, 4), (3, 4, 2)],
      ]
    ],
    dtype=float_types,
    lower=[True, False],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testChoSolveBroadcast(self, lhs_shape, rhs_shape, dtype, lower):
    if scipy_version < (1, 16, 0):
      self.skipTest("scipy.linalg.cho_solve batch broadcasting requires scipy >= 1.16")
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

      # compute_expm is True
      osp_fun = lambda a,e: osp.linalg.expm_frechet(a,e,compute_expm=True)
      jsp_fun = lambda a,e: jsp.linalg.expm_frechet(a,e,compute_expm=True)
      self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                              check_dtypes=False, tol=tol)
      self._CompileAndCheck(jsp_fun, args_maker, check_dtypes=False)
      # compute_expm is False
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
    self._CheckAgainstNumpy(
        osp.linalg.rsf2csf, jsp.linalg.rsf2csf, args_maker, tol=tol
    )
    self._CompileAndCheck(jsp.linalg.rsf2csf, args_maker)

  @jtu.sample_product(
    [dict(t_shape=t_shape, z_shape=z_shape)
     for t_shape, z_shape in [
       # batched T and Z
       ((3, 4, 4), (3, 4, 4)),
       # batched T, unbatched Z — broadcast Z
       ((3, 4, 4), (4, 4)),
       # unbatched T, batched Z — broadcast T
       ((4, 4), (3, 4, 4)),
     ]
    ],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testRsf2csfBroadcast(self, t_shape, z_shape, dtype):
    if scipy_version < (1, 16, 0):
      self.skipTest("scipy.linalg.rsf2csf batch broadcasting requires scipy >= 1.16")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(t_shape, dtype), rng(z_shape, dtype)]
    self._CheckAgainstNumpy(osp.linalg.rsf2csf, jsp.linalg.rsf2csf,
                            args_maker, tol=3e-5)
    self._CompileAndCheck(jsp.linalg.rsf2csf, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, side=side)
     for shape, side in [
       ((4, 4), 'right'),  # square, side='right' (m >= n)
       ((5, 3), 'right'),  # tall, side='right' (m > n)
       ((3, 5), 'left'),   # wide, side='left' (m < n, qdwh constraint)
     ]
    ],
    batch=[3],
    dtype=float_types,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testPolarBatch(self, batch, shape, side, dtype):
    if scipy_version < (1, 16, 0):
      self.skipTest("scipy.linalg.polar batch support requires scipy >= 1.16")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((batch,) + shape, dtype)]
    self._CheckAgainstNumpy(
        partial(osp.linalg.polar, side=side),
        partial(jsp.linalg.polar, side=side),
        args_maker, tol=1e-4)
    self._CompileAndCheck(partial(jsp.linalg.polar, side=side), args_maker)

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
    self._CheckAgainstNumpy(
        jnp_fun,
        scp_fun,
        args_maker,
        check_dtypes=False,
        tol={np.complex64: 1e-5, np.complex128: 1e-6},
    )
    self._CompileAndCheck(jnp_fun, args_maker, atol=2e-5)

  @jtu.sample_product(
    shape=[(4, 4), (15, 15), (50, 50), (100, 100)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  @jtu.ignore_warning(
      category=RuntimeWarning, message='invalid value encountered in matmul'
  )
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
    self._CheckAgainstNumpy(osp.linalg.sqrtm, jsp.linalg.sqrtm, args_maker,
                            tol=tol, check_dtypes=False)
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
    cshape=[(), (4,), (8,), (4, 7), (2, 1, 5)],
    cdtype=float_types + complex_types,
    rshape=[(), (3,), (7,), (4, 4), (2, 4, 0)],
    rdtype=float_types + complex_types + int_types)
  def testToeplitzConstruction(self, rshape, rdtype, cshape, cdtype):
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
    with jax.numpy_rank_promotion("allow"):
      with jtu.strict_promotion_if_dtypes_match([rdtype, cdtype]):
        self._CheckAgainstNumpy(jtu.promote_like_jnp(osp_linalg_toeplitz),
                                jsp.linalg.toeplitz, args_maker)
        self._CompileAndCheck(jsp.linalg.toeplitz, args_maker)

  @jtu.sample_product(
    shape=[(), (3,), (1, 4), (1, 5, 9), (11, 0, 13)],
    dtype=float_types + complex_types + int_types)
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
    self._CheckAgainstNumpy(osp_linalg_toeplitz, jsp.linalg.toeplitz, args_maker)
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
  def testToeplitzNaNPropagation(self):
    # Regression test for https://github.com/jax-ml/jax/issues/38636
    # NaN in column/row should only appear at the diagonal position it maps to,
    # not bleed into the whole matrix. The old conv-based implementation used
    # arithmetic (NaN + x = NaN) instead of pure indexing, poisoning all outputs.
    column = np.array([np.nan, np.inf, -0.0], dtype=np.float32)
    row = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
    expected = np.array([
        [np.nan, 2.0,    3.0 ],
        [np.inf, np.nan, 2.0 ],
        [-0.0,   np.inf, np.nan],
    ], dtype=np.float32)
    out = np.asarray(jsp.linalg.toeplitz(
        jnp.asarray(column), jnp.asarray(row)))
    self.assertTrue(
        np.array_equal(out, expected, equal_nan=True),
        msg=f"toeplitz NaN propagation: got\n{out}\nexpected\n{expected}")
    # Exactly 3 NaNs on the main diagonal, none elsewhere
    self.assertEqual(np.sum(np.isnan(out)), 3,
                     msg="Expected exactly 3 NaNs (main diagonal only)")

    # Also verify: a single NaN in c[0] only populates the main diagonal.
    c2 = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
    r2 = np.array([np.nan, 3.0, 4.0], dtype=np.float32)
    out2 = np.asarray(jsp.linalg.toeplitz(jnp.asarray(c2), jnp.asarray(r2)))
    # Non-diagonal entries must be finite (no NaN bleed)
    diag_mask = np.eye(3, dtype=bool)
    self.assertTrue(np.all(np.isfinite(out2[~diag_mask])),
                    msg=f"Off-diagonal entries should be finite, got\n{out2}")
    self.assertTrue(np.all(np.isnan(out2[diag_mask])),
                    msg=f"Diagonal entries should be NaN, got\n{out2}")

  def testHankelNaNPropagation(self):
    # Companion regression for https://github.com/jax-ml/jax/issues/38636:
    # hankel also used conv_general_dilated_patches (same root cause).
    c = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
    r = np.array([np.nan, 4.0, 5.0], dtype=np.float32)
    # hankel[i,j] = v[i+j], where v = [c[0], c[1], c[2], r[1], r[2]]
    # = [NaN, 1, 2, 4, 5]
    # Only hankel[0,0] (v[0]) should be NaN.
    expected = np.array([
        [np.nan, 1.0, 2.0],
        [1.0,    2.0, 4.0],
        [2.0,    4.0, 5.0],
    ], dtype=np.float32)
    out = np.asarray(jsp.linalg.hankel(jnp.asarray(c), jnp.asarray(r)))
    self.assertTrue(
        np.array_equal(out, expected, equal_nan=True),
        msg=f"hankel NaN propagation: got\n{out}\nexpected\n{expected}")
    self.assertEqual(np.sum(np.isnan(out)), 1,
                     msg="Expected exactly 1 NaN at [0,0] only")

  def testHankelConstructionWithKnownCases(self):
    # r=None should default to zeros_like(c)
    c = np.array([1, 2, 3], dtype=np.float32)
    expected = osp_linalg_hankel(c, None)
    actual = jsp.linalg.hankel(c)
    self.assertAllClose(expected, actual)

    # r[0] is ignored (changing it should not change output)
    r1 = np.array([999, 4, 5, 6], dtype=np.float32)
    r2 = np.array([c[-1], 4, 5, 6], dtype=np.float32)
    out1 = jsp.linalg.hankel(c, r1)
    out2 = jsp.linalg.hankel(c, r2)
    self.assertAllClose(out1, out2)

  @jtu.sample_product(
     cshape = [(3,), (0,), (2, 3), (1, 2, 3)],
     rshape = [(4,), (0,), (1, 4), (2, 4), (1, 1, 4)],
     cdtype=float_types + complex_types + int_types,
     rdtype=float_types + complex_types + int_types,
  )
  def testHankelConstruction(self, cshape, cdtype, rshape, rdtype):
    int_types_excl_i8 = set(int_types) - {np.int8}
    if ((rdtype in int_types_excl_i8 or cdtype in int_types_excl_i8)
        and jtu.test_device_matches(["gpu"])):
      self.skipTest("Integer (except int8) hankel is not supported on GPU")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(cshape, cdtype), rng(rshape, rdtype)]
    with jtu.strict_promotion_if_dtypes_match([rdtype, cdtype]):
      self._CheckAgainstNumpy(
          jtu.promote_like_jnp(osp_linalg_hankel),
          jsp.linalg.hankel,
          args_maker,
          check_dtypes=False,
      )
      self._CompileAndCheck(jsp.linalg.hankel, args_maker)

  @jtu.sample_product(
     shape=[(), (3,), (0,), (1,), (5,), (2, 3), (1, 2, 4)],
     dtype=float_types + complex_types + int_types,
  )
  def testCirculantConstruction(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(scipy.linalg.circulant, jsp.linalg.circulant, args_maker)
    self._CompileAndCheck(jsp.linalg.circulant, args_maker)

  @jtu.sample_product(
     n=[2, 3, 5, 8],
     batch=[(), (2,), (1, 3)],
     dtype=float_types + int_types,
  )
  def testLeslie(self, n, batch, dtype):
    rng = jtu.rand_default(self.rng())
    f_shape = batch + (n,)
    s_shape = batch + (n - 1,)
    args_maker = lambda: [rng(f_shape, dtype), rng(s_shape, dtype)]
    self._CheckAgainstNumpy(scipy.linalg.leslie, jsp.linalg.leslie, args_maker,
                            check_dtypes=False)
    self._CompileAndCheck(jsp.linalg.leslie, args_maker)

  def testLeslieInvalidLength(self):
    with self.assertRaisesRegex(ValueError, "length of f along the last axis"):
      jsp.linalg.leslie(jnp.array([1.0]), jnp.array([]))
    with self.assertRaisesRegex(ValueError, "Incorrect lengths for f and s"):
      jsp.linalg.leslie(jnp.array([1., 2., 3.]), jnp.array([0.5]))
    with self.assertRaisesRegex(ValueError, "Incorrect lengths for f and s"):
      jsp.linalg.leslie(jnp.array([1., 2.]), jnp.array([]))

  @jtu.sample_product(
     shape=[(2,), (3,), (5,), (8,), (2, 3), (1, 2, 4)],
     dtype=float_types + complex_types + int_types,
  )
  def testCompanion(self, shape, dtype):
    rng = jtu.rand_nonzero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    tol = {np.complex64: 1e-5, np.float32: 1e-5}
    self._CheckAgainstNumpy(scipy.linalg.companion, jsp.linalg.companion,
                            args_maker, atol=tol, rtol=tol, check_dtypes=False)
    self._CompileAndCheck(jsp.linalg.companion, args_maker)

  @jtu.sample_product(shape=[(0,), (1,), (3, 0), (3, 1)])
  def testCompanionInvalidLength(self, shape):
    a = jnp.zeros(shape)
    with self.assertRaisesRegex(ValueError, "length of `a` along the last axis"):
      jsp.linalg.companion(a)

  @jtu.sample_product(
     shape=[(), (1,), (3,), (5,), (2, 3), (1, 2, 4)],
     dtype=float_types + int_types,
  )
  def testFiedler(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(scipy.linalg.fiedler, jsp.linalg.fiedler, args_maker,
                            check_dtypes=False)
    self._CompileAndCheck(jsp.linalg.fiedler, args_maker)

  @jtu.sample_product(
     shape=[(2,), (3,), (4,), (5,), (8,), (2, 3), (1, 2, 4)],
     dtype=float_types,
  )
  def testFiedlerCompanion(self, shape, dtype):
    rng = jtu.rand_nonzero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    tol = {np.float32: 1e-5, np.float64: 1e-10}
    self._CheckAgainstNumpy(scipy.linalg.fiedler_companion,
                            jsp.linalg.fiedler_companion, args_maker,
                            atol=tol, rtol=tol, check_dtypes=False)
    self._CompileAndCheck(jsp.linalg.fiedler_companion, args_maker)

  @jtu.sample_product(shape=[(1,), (3, 1)])
  def testFiedlerCompanionShortInput(self, shape):
    # A single coefficient along the last axis yields an empty (..., 0, 0)
    # companion matrix rather than an error, per the docstring.
    a = jnp.ones(shape)
    result = jsp.linalg.fiedler_companion(a)
    self.assertEqual(result.shape, shape[:-1] + (0, 0))

  @jtu.sample_product(shape=[(0,), (3, 0)])
  def testFiedlerCompanionEmptyInput(self, shape):
    # A zero-length last axis raises a clear ValueError rather than an
    # IndexError from the implementation.
    a = jnp.ones(shape)
    with self.assertRaisesRegex(ValueError, "nonzero"):
      jsp.linalg.fiedler_companion(a)

  @jtu.sample_product(
     m=[1, 2, 3, 5],
     n=[1, 2, 3, 5, 7],
     mode=['full', 'valid', 'same'],
     batch=[(), (2,), (1, 3)],
     # Integer dtypes are skipped because the underlying Toeplitz lowers via
     # an integer convolution that CuDNN only supports for s8 inputs with f32
     # bias on GPU.
     dtype=float_types + complex_types,
  )
  def testConvolutionMatrix(self, m, n, mode, batch, dtype):
    rng = jtu.rand_default(self.rng())
    shape = (*batch, m)
    args_maker = lambda: [rng(shape, dtype)]
    osp_fun = partial(scipy.linalg.convolution_matrix, n=n, mode=mode)
    jsp_fun = partial(jsp.linalg.convolution_matrix, n=n, mode=mode)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jsp_fun, args_maker)

  def testConvolutionMatrixInvalidArgs(self):
    a = jnp.array([1., 2., 3.])
    with self.assertRaisesRegex(ValueError, "n must be a positive integer"):
      jsp.linalg.convolution_matrix(a, 0)
    with self.assertRaisesRegex(ValueError, "n must be a positive integer"):
      jsp.linalg.convolution_matrix(a, -1)
    with self.assertRaisesRegex(ValueError, "mode must be one of"):
      jsp.linalg.convolution_matrix(a, 3, mode='bad')
    with self.assertRaisesRegex(ValueError, r"len\(a\) must be at least 1"):
      jsp.linalg.convolution_matrix(jnp.zeros((0,)), 3)
    with self.assertRaisesRegex(ValueError, r"len\(a\) must be at least 1"):
      jsp.linalg.convolution_matrix(jnp.zeros((2, 0)), 3)
    with self.assertRaisesRegex(ValueError, "must be at least 1-dimensional"):
      jsp.linalg.convolution_matrix(jnp.array(5.0), 3)

  @jtu.sample_product(
    shape=[(2, 3), (4, 6), (50, 7), (100, 110)],
    dtype = float_types + complex_types,
    method = ["schur", "eigen"]
  )
  @jtu.run_on_devices("cpu", "gpu")
  @jax.default_matmul_precision("float32")
  def test_solve_sylvester(self, shape, dtype, method):
    if jtu.test_device_matches(["gpu"]) and method == "schur":
      self.skipTest("Schur not supported on GPU.")

    tol = {np.float32: 1e-1, np.float64: 1e-9, np.complex64: 5e-2, np.complex128: 1e-9}

    def args_maker():
      rng = jtu.rand_default(self.rng())
      m, n = shape

      A = rng(shape=(m, m), dtype=dtype)
      B = rng(shape=(n, n), dtype=dtype)
      X_true = rng(shape=(m, n), dtype=dtype)

      C = A @ X_true + X_true @ B
      return [A, B, C]

    jnp_fun = partial(jsp.linalg.solve_sylvester, method=method)

    self._CheckAgainstNumpy(osp.linalg.solve_sylvester, jnp_fun, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)


  @jtu.sample_product(
    n=[3, 6, 7, 100],
    dtype = float_types + complex_types,
    method = ["schur", "eigen"]
  )
  @jtu.run_on_devices("cpu", "gpu")
  def test_ill_conditioned_sylvester(self, n, dtype, method):
    """
    Test no solution case to AX + XB = C using the eigen decomposition method.
    When the sum of the eigenvalues of A and B are zero there is no solution.
    We simulate this case below by randomly selecting the eigenvalues of A and then assign the
    eigenvalues of B as negative eigenvalues of A. We say that A and B are ill-conditioned.
    """
    if jtu.test_device_matches(["gpu"]) and method == "schur":
      self.skipTest("Schur not supported on GPU.")

    rng = jtu.rand_default(self.rng())

    # Define eigenvalues that sum to zero
    eigenvalues_A = rng(shape=(n,), dtype=dtype)
    eigenvalues_B = -eigenvalues_A
    P = _random_invertible(rng=rng, shape=(n, n), dtype=dtype)

    # Construct A and B matrices using selected eigenvalues that positionally sum to zero
    D_A = np.diag(eigenvalues_A)
    D_B = np.diag(eigenvalues_B)
    P_inv = np.linalg.inv(P)
    A = P @ D_A @ P_inv
    B = P @ D_B @ P_inv

    C = rng(shape=(n, n), dtype=dtype)
    sylv_solution = jsp.linalg.solve_sylvester(A, B, C, method=method, tol=1e-5)
    self.assertArraysEqual(sylv_solution, np.full((n, n), np.nan, dtype))

  @jtu.sample_product(
    [dict(a_shape=a_shape, b_shape=b_shape, c_shape=c_shape)
     for a_shape, b_shape, c_shape in [
       # unbatched A,B; batched C — broadcast A,B to batch
       ((3, 3), (4, 4), (2, 3, 4)),
       # batched A; unbatched B, C
       ((2, 3, 3), (4, 4), (3, 4)),
       # all batched
       ((2, 3, 3), (2, 4, 4), (2, 3, 4)),
     ]
    ],
    dtype=float_types,
    method=["schur", "eigen"],
  )
  @jtu.run_on_devices("cpu", "gpu")
  @jax.default_matmul_precision("float32")
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def test_solve_sylvester_broadcast(self, a_shape, b_shape, c_shape, dtype, method):
    if scipy_version < (1, 16, 0):
      self.skipTest("scipy.linalg.solve_sylvester batch broadcasting requires scipy >= 1.16")
    if jtu.test_device_matches(["gpu"]) and method == "schur":
      self.skipTest("Schur not supported on GPU.")
    tol = {np.float32: 5e-2, np.float64: 1e-9}
    rng = jtu.rand_default(self.rng())

    def args_maker():
      A = rng(a_shape, dtype)
      B = rng(b_shape, dtype)
      # Compute the broadcast batch shape to construct a valid C
      batch = np.broadcast_shapes(a_shape[:-2], b_shape[:-2], c_shape[:-2])
      A_bc = np.broadcast_to(A, batch + a_shape[-2:])
      B_bc = np.broadcast_to(B, batch + b_shape[-2:])
      X_true = rng(batch + c_shape[-2:], dtype)
      C = np.einsum("...ij,...jk->...ik", A_bc, X_true) + \
          np.einsum("...ij,...jk->...ik", X_true, B_bc)
      # Return un-broadcast A, B so JAX must do the broadcasting itself
      return [A, B, C]

    jnp_fun = partial(jsp.linalg.solve_sylvester, method=method)
    self._CheckAgainstNumpy(osp.linalg.solve_sylvester, jnp_fun, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)


class LaxLinalgTest(jtu.JaxTestCase):
  """Tests for lax.linalg primitives."""


  @jtu.sample_product(shape=[(2,), (3,), (3, 2), (3, 4), (3, 4, 5)],
                      dtype=float_types + complex_types,
                      k_rhs=[1, 2],
                      perturb_singular=[False, True])
  def test_tridiagonal_solve(self, shape, dtype, k_rhs, perturb_singular):

    if perturb_singular and not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("perturb_singular=True only supported on CPU and GPU")
    # TODO: Add these tests back once rocSparse issues are fixed.
    if jtu.is_device_rocm() and shape in [(3, 4), (3, 4, 5)]:
      self.skipTest("Skipped on ROCm due to rocSparse numerical error.")
    rng = self.rng()
    d = 1.0 + jtu.rand_positive(rng)(shape, dtype)
    dl = jtu.rand_default(rng)(shape, dtype)
    du = jtu.rand_default(rng)(shape, dtype)
    b = jtu.rand_default(rng)(shape + (k_rhs,), dtype)
    x = lax.linalg.tridiagonal_solve(
      dl, d, du, b, perturb_singular=perturb_singular)

    def build_tri(dl, d, du):
      return jnp.diag(d) + jnp.diag(dl[1:], -1) + jnp.diag(du[:-1], 1)
    for _ in shape[:-1]:
      build_tri = jax.vmap(build_tri)

    a = build_tri(dl, d, du)
    with jax.default_matmul_precision("float32"):
      self.assertAllClose(a @ x, b, atol={
          np.float32: 1e-3, np.float64: 1e-10, np.complex64: 1e-3,
          np.complex128: 1e-10})

  @jtu.sample_product(perturb_singular=[False, True])
  def test_tridiagonal_solve_zero_matrix(self, perturb_singular):

    if perturb_singular and not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("perturb_singular=True only supported on CPU and GPU")
    dtype = np.float32
    dl = np.zeros((3,), dtype=dtype)
    d = np.zeros((3,), dtype=dtype)
    du = np.zeros((3,), dtype=dtype)
    b = np.ones((3, 1), dtype=dtype)

    x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=perturb_singular)

    if perturb_singular:
      self.assertFalse(np.any(np.isnan(x)))
      self.assertTrue(np.all(np.abs(x) > 1e6))
    else:
      self.assertTrue(np.any(np.isnan(x)) or np.any(np.isinf(x)))

  @jtu.sample_product(perturb_singular=[False, True])
  def test_tridiagonal_solve_requiring_pivoting(self, perturb_singular):
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("Pivoting not supported in fallback tridiagonal solve")

    # TODO(magaonka-amd): Re-enable once hipSPARSE gtsv2 numerical issues are resolved.
    if jtu.is_device_rocm() and not perturb_singular:
      self.skipTest(
          "Skipped on ROCm: hipsparseSgtsv2 numerical error on pivoting path.")
    dl = np.array([0.0, 2.0, -2.0, 3.0], dtype=np.float32)
    d = np.array([1.0, 4.0, 1.0, -1.0], dtype=np.float32)
    du = np.array([2.0, -1.0, 1.0, 0.0], dtype=np.float32)
    b = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

    x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=perturb_singular)
    expected = np.array([[8.0], [-3.5], [0.0], [-4.0]], dtype=np.float32)
    self.assertAllClose(x, expected)

  @jtu.sample_product(perturb_singular=[False, True])
  def test_tridiagonal_solve_requiring_pivoting_last_rows(self, perturb_singular):
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("Pivoting not supported in fallback tridiagonal solve")

    # TODO(magaonka-amd): Re-enable once hipSPARSE gtsv2 numerical issues are resolved.
    if jtu.is_device_rocm() and not perturb_singular:
      self.skipTest(
          "Skipped on ROCm: hipsparseSgtsv2 numerical error on pivoting path.")
    dl = np.array([0.0, 1.0, -6.0, 1.0], dtype=np.float32)
    d = np.array([1.0, -1.0, 2.0, 1.0], dtype=np.float32)
    du = np.array([2.0, 1.0, -1.0, 0.0], dtype=np.float32)
    b = np.array([[1.0], [2.0], [-1.0], [-2.0]], dtype=np.float32)

    x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=perturb_singular)
    expected = np.array([[5.0], [-2.0], [-5.0], [3.0]], dtype=np.float32)
    self.assertAllClose(x, expected)

  @jtu.sample_product(perturb_singular=[False, True])
  def test_tridiagonal_solve_2x2_singular(self, perturb_singular):

    if perturb_singular and not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("perturb_singular=True only supported on CPU and GPU")
    dl = np.array([0.0, 1.0], dtype=np.float32)
    d = np.array([1.0, 3.0], dtype=np.float32)
    du = np.array([3.0, 0.0], dtype=np.float32)
    b = np.array([[1.0], [4.0]], dtype=np.float32)

    x = lax.linalg.tridiagonal_solve(
      dl, d, du, b, perturb_singular=perturb_singular)
    if perturb_singular:
      self.assertFalse(np.any(np.isnan(x)))
      self.assertTrue(np.all(np.abs(x) > 1e6))
    else:
      self.assertTrue(np.any(np.isnan(x)) or np.any(np.isinf(x)))

  @jtu.sample_product(perturb_singular=[False, True])
  def test_tridiagonal_solve_endpoints(self, perturb_singular):

    if perturb_singular and not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("perturb_singular=True only supported on CPU and GPU")
    # tridagonal_solve shouldn't depend on the endpoints being explicitly zero.
    dtype = np.float32
    size = 10
    dl = np.linspace(-1.0, 1.0, size, dtype=dtype)
    dlz = np.copy(dl)
    dlz[0] = 0.0
    d = np.linspace(1.0, 2.0, size, dtype=dtype)
    du = np.linspace(1.0, -1.0, size, dtype=dtype)
    duz = np.copy(du)
    duz[-1] = 0.0
    b = np.linspace(0.1, -0.1, size, dtype=dtype)[:, None]
    self.assertAllClose(
        lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=perturb_singular),
        lax.linalg.tridiagonal_solve(dlz, d, duz, b, perturb_singular=perturb_singular),
    )

  @jtu.sample_product(shape=[(3,), (3, 4)], dtype=float_types + complex_types,
                      perturb_singular=[False, True])
  def test_tridiagonal_solve_grad(self, shape, dtype, perturb_singular):

    if perturb_singular and not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("perturb_singular=True only supported on CPU and GPU")
    # TODO: Add these tests back once rocSparse issues are fixed.
    if jtu.is_device_rocm() and shape == (3, 4):
      self.skipTest("Skipped on ROCm due to rocSparse numerical error.")
    rng = self.rng()
    d = 1.0 + jtu.rand_positive(rng)(shape, dtype)
    dl = jtu.rand_default(rng)(shape, dtype)
    du = jtu.rand_default(rng)(shape, dtype)
    b = jtu.rand_default(rng)(shape + (1,), dtype)
    args = (dl, d, du, b)
    f = lambda dl, d, du, b: lax.linalg.tridiagonal_solve(
      dl, d, du, b, perturb_singular=perturb_singular)
    jtu.check_grads(f, args, order=2, atol=1e-1, rtol=1e-1)

  @jtu.sample_product(
    shape=[(4, 4), (15, 15), (50, 50), (100, 100)],
    dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("cpu")
  def testSchur(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args = rng(shape, dtype)
    Ts, Ss = lax.linalg.schur(args)
    eps = np.finfo(dtype).eps
    self.assertAllClose(args, Ss @ Ts @ jnp.conj(Ss.T), atol=600 * eps)
    self.assertAllClose(
        np.eye(*shape, dtype=dtype), Ss @ jnp.conj(Ss.T), atol=100 * eps
    )

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

  @jtu.sample_product(
    n=[0, 1, 2, 3, 5, 8],
    )
  def testInvhilbert(self, n):
    args_maker = lambda: []
    osp_fun = lambda: osp.linalg.invhilbert(n, exact=False)
    jsp_fun = partial(jsp.linalg.invhilbert, n=n)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            atol=1e-3,
                            rtol=1e-2 if jtu.test_device_matches(['tpu']) else 1e-3,
                            check_dtypes=False)
    self._CompileAndCheck(jsp_fun, args_maker)

  @jtu.sample_product(
    n=[1, 2, 4, 8, 16],
    dtype=int_types + float_types,
  )
  def testHadamard(self, n, dtype):
    args_maker = lambda: []
    osp_fun = partial(osp.linalg.hadamard, n=n, dtype=dtype)
    jsp_fun = partial(jsp.linalg.hadamard, n=n, dtype=dtype)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker)
    self._CompileAndCheck(jsp_fun, args_maker)

  @jtu.sample_product(n=[0, -1, 3, 5, 7])
  def testHadamardInvalidN(self, n):
    with self.assertRaisesRegex(ValueError, "positive power of 2"):
      jsp.linalg.hadamard(n)

  @jtu.sample_product(
    n=[1, 2, 3, 4, 5, 8],
    full=[False, True],
  )
  def testHelmert(self, n, full):
    args_maker = lambda: []
    osp_fun = partial(osp.linalg.helmert, n=n, full=full)
    jsp_fun = partial(jsp.linalg.helmert, n=n, full=full)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            atol=1e-5, rtol=1e-5, check_dtypes=False)
    self._CompileAndCheck(jsp_fun, args_maker)

  @jtu.sample_product(n=[0, -1, -5])
  def testHelmertInvalidN(self, n):
    with self.assertRaisesRegex(ValueError, "n must be a positive integer"):
      jsp.linalg.helmert(n)

  @jtu.sample_product(
    n=[0, 1, 2, 3, 5, 8, 16],
    scale=[None, 'sqrtn', 'n'],
  )
  def testDft(self, n, scale):
    args_maker = lambda: []
    osp_fun = partial(osp.linalg.dft, n=n, scale=scale)
    jsp_fun = partial(jsp.linalg.dft, n=n, scale=scale)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            atol=1e-5, rtol=1e-5, check_dtypes=False)
    self._CompileAndCheck(jsp_fun, args_maker)

  def testDftInvalidScale(self):
    with self.assertRaisesRegex(ValueError, "scale must be"):
      jsp.linalg.dft(4, scale='bad')

  @jtu.sample_product(dtype=complex_types)
  def testDftDtype(self, dtype):
    result = jsp.linalg.dft(4, dtype=dtype)
    self.assertEqual(result.dtype, dtype)

  def testDftInvalidDtype(self):
    with self.assertRaisesRegex(ValueError, "dtype must be"):
      jsp.linalg.dft(4, dtype=jnp.float32)

  @jtu.sample_product(
      shape=[(5, 1), (10, 4), (128, 12)],
      dtype=float_types,
      symmetrize_output=[True, False],
  )
  @jtu.skip_on_devices("tpu")
  def testSymmetricProduct(self, shape, dtype, symmetrize_output):
    rng = jtu.rand_default(self.rng())
    batch_size = 10
    atol = 1e-6 if dtype == jnp.float64 else 1e-3

    a_matrix = rng((batch_size,) + shape, dtype)
    c_shape = a_matrix.shape[:-1] + (a_matrix.shape[-2],)
    c_matrix = jnp.zeros(c_shape, dtype)

    old_product = jnp.einsum("...ij,...kj->...ik", a_matrix, a_matrix,
                             precision=lax.Precision.HIGHEST)
    new_product = lax_linalg.symmetric_product(
        a_matrix, c_matrix, symmetrize_output=symmetrize_output)
    new_product_with_batching = jax.vmap(
        lambda a, c: lax_linalg.symmetric_product(
            a, c, symmetrize_output=symmetrize_output),
        in_axes=(0, 0))(a_matrix, c_matrix)

    if not symmetrize_output:
      old_product = jnp.tril(old_product)
      new_product = jnp.tril(new_product)
      new_product_with_batching = jnp.tril(new_product_with_batching)
    self.assertAllClose(new_product, old_product, atol=atol)
    self.assertAllClose(
        new_product_with_batching, old_product, atol=atol)

  @jtu.sample_product(
    n=[0, 1, 5, 10, 20],
    kind=["symmetric", "lower", "upper"],
  )
  @jax.default_matmul_precision("float32")
  def testPascal(self, n, kind):
    args_maker = lambda: []
    osp_fun = partial(osp.linalg.pascal, n=n, kind=kind, exact=False)
    jsp_fun = partial(jsp.linalg.pascal, n=n, kind=kind)
    self._CheckAgainstNumpy(osp_fun,
                            jsp_fun, args_maker,
                            atol=1e-3,
                            rtol=1e-2 if jtu.test_device_matches(['tpu']) else 1e-3,
                            check_dtypes=False)
    self._CompileAndCheck(jsp_fun, args_maker)

  @jtu.sample_product(
    n=[0, 1, 2, 5, 8],
    kind=["symmetric", "lower", "upper"],
  )
  @jax.default_matmul_precision("float32")
  def testInvpascal(self, n, kind):
    args_maker = lambda: []
    osp_fun = partial(osp.linalg.invpascal, n=n, kind=kind, exact=False)
    jsp_fun = partial(jsp.linalg.invpascal, n=n, kind=kind)
    self._CheckAgainstNumpy(osp_fun, jsp_fun, args_maker,
                            atol=1e-3,
                            rtol=1e-2 if jtu.test_device_matches(['tpu']) else 1e-3,
                            check_dtypes=False)
    self._CompileAndCheck(jsp_fun, args_maker)

  def testInvpascalInvalidKind(self):
    with self.assertRaisesRegex(ValueError, "Expected kind to be one of"):
      jsp.linalg.invpascal(3, kind='bad')


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
