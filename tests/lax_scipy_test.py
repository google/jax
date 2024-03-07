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
import unittest

from absl.testing import absltest

import numpy as np
import scipy.integrate
import scipy.special as osp_special
import scipy.cluster as osp_cluster

import jax
import jax.dtypes
from jax import numpy as jnp
from jax import lax
from jax import scipy as jsp
from jax._src.scipy import special as lsp_special_internal
from jax._src import test_util as jtu
from jax.scipy import special as lsp_special
from jax.scipy import cluster as lsp_cluster

jax.config.parse_flags_with_absl()

scipy_version = jtu.parse_version(scipy.version.version)

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]
compatible_shapes = [[(), ()],
                     [(4,), (3, 4)],
                     [(3, 1), (1, 4)],
                     [(2, 3, 4), (2, 1, 4)]]

float_dtypes = jtu.dtypes.floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.integer

# Params for the polar tests.
polar_shapes = [(16, 12), (12, 16), (128, 128)]
n_zero_svs = [0, 4]
degeneracies = [0, 4]
geometric_spectra = [False, True]
max_svs = [0.1, 10.]
nonzero_condition_numbers = [0.1, 100000]
sides = ["right", "left"]
methods = ["qdwh", "svd"]
seeds = [1, 10]


def _initialize_polar_test(rng, shape, n_zero_svs, degeneracy, geometric_spectrum,
                           max_sv, nonzero_condition_number, dtype):

  n_rows, n_cols = shape
  min_dim = min(shape)
  left_vecs = rng.randn(n_rows, min_dim).astype(np.float64)
  left_vecs, _ = np.linalg.qr(left_vecs)
  right_vecs = rng.randn(n_cols, min_dim).astype(np.float64)
  right_vecs, _ = np.linalg.qr(right_vecs)

  min_nonzero_sv = max_sv / nonzero_condition_number
  num_nonzero_svs = min_dim - n_zero_svs
  if geometric_spectrum:
    nonzero_svs = np.geomspace(min_nonzero_sv, max_sv, num=num_nonzero_svs,
                               dtype=np.float64)
  else:
    nonzero_svs = np.linspace(min_nonzero_sv, max_sv, num=num_nonzero_svs,
                              dtype=np.float64)
  half_point = n_zero_svs // 2
  for i in range(half_point, half_point + degeneracy):
    nonzero_svs[i] = nonzero_svs[half_point]
  svs = np.zeros(min(shape), dtype=np.float64)
  svs[n_zero_svs:] = nonzero_svs
  svs = svs[::-1]

  result = np.dot(left_vecs * svs, right_vecs.conj().T)
  result = jnp.array(result).astype(dtype)
  spectrum = jnp.array(svs).astype(dtype)
  return result, spectrum


class LaxBackedScipyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Scipy implementation."""

  @jtu.sample_product(
    [dict(shapes=shapes, axis=axis, use_b=use_b)
      for shape_group in compatible_shapes
      for use_b in [False, True]
      for shapes in itertools.product(*(
        (shape_group, shape_group) if use_b else (shape_group,)))
      for axis in range(-max(len(shape) for shape in shapes),
                         max(len(shape) for shape in shapes))
    ],
    dtype=float_dtypes + complex_dtypes + int_dtypes,
    keepdims=[False, True],
    return_sign=[False, True],
  )
  @jtu.ignore_warning(category=RuntimeWarning, message="invalid value encountered in .*")
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLogSumExp(self, shapes, dtype, axis,
                    keepdims, return_sign, use_b):
    if jnp.issubdtype(dtype, jnp.complexfloating) and scipy_version < (1, 13, 0):
      self.skipTest("logsumexp of complex input uses scipy 1.13.0 semantics.")
    if not jtu.test_device_matches(["cpu"]):
      rng = jtu.rand_some_inf_and_nan(self.rng())
    else:
      rng = jtu.rand_default(self.rng())
    # TODO(mattjj): test autodiff
    if use_b:
      def scipy_fun(array_to_reduce, scale_array):
        res = osp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                    return_sign=return_sign, b=scale_array)
        if dtype == np.int32:
          res = jax.tree.map(lambda x: x.astype('float32'), res)
        return res

      def lax_fun(array_to_reduce, scale_array):
        return lsp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign, b=scale_array)

      args_maker = lambda: [rng(shapes[0], dtype), rng(shapes[1], dtype)]
    else:
      def scipy_fun(array_to_reduce):
        res = osp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                    return_sign=return_sign)
        if dtype == np.int32:
          res = jax.tree.map(lambda x: x.astype('float32'), res)
        return res

      def lax_fun(array_to_reduce):
        return lsp_special.logsumexp(array_to_reduce, axis, keepdims=keepdims,
                                     return_sign=return_sign)

      args_maker = lambda: [rng(shapes[0], dtype)]
    tol = (
        {np.float32: 2e-4, np.complex64: 2e-4}
        if jtu.test_device_matches(["tpu"])
        else None
    )
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, rtol=tol, atol=tol)
    tol = {np.float32: 1E-6, np.float64: 1E-14}
    self._CompileAndCheck(lax_fun, args_maker, rtol=tol, atol=tol)

  def testLogSumExpComplexSign(self):
    # Tests behavior of complex sign, which changed in SciPy 1.13
    x = jnp.array([1 + 1j, 2 - 1j, -2 + 3j])
    logsumexp, sign = lsp_special.logsumexp(x, return_sign=True)
    expected_sumexp = jnp.exp(x).sum()
    expected_sign = expected_sumexp / abs(expected_sumexp).astype(x.dtype)
    self.assertEqual(logsumexp.dtype, sign.real.dtype)
    tol = 1E-4 if jtu.test_device_matches(['tpu']) else 1E-6
    self.assertAllClose(sign, expected_sign, rtol=tol)
    self.assertAllClose(sign * np.exp(logsumexp).astype(x.dtype), expected_sumexp, rtol=tol)

  def testLogSumExpZeros(self):
    # Regression test for https://github.com/google/jax/issues/5370
    scipy_fun = lambda a, b: osp_special.logsumexp(a, b=b)
    lax_fun = lambda a, b: lsp_special.logsumexp(a, b=b)
    args_maker = lambda: [np.array([-1000, -2]), np.array([1, 0])]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker)
    self._CompileAndCheck(lax_fun, args_maker)

  def testLogSumExpOnes(self):
    # Regression test for https://github.com/google/jax/issues/7390
    args_maker = lambda: [np.ones(4, dtype='float32')]
    with jax.debug_infs(True):
      self._CheckAgainstNumpy(osp_special.logsumexp, lsp_special.logsumexp, args_maker)
      self._CompileAndCheck(lsp_special.logsumexp, args_maker)

  def testLogSumExpNans(self):
    # Regression test for https://github.com/google/jax/issues/7634
    with jax.debug_nans(True):
      with jax.disable_jit():
        result = lsp_special.logsumexp(1.0)
        self.assertEqual(result, 1.0)

        result = lsp_special.logsumexp(1.0, b=1.0)
        self.assertEqual(result, 1.0)

  @jtu.sample_product(
    shape=[(0,), (1,), (2,), (3,), (4,), (5,)],
    dtype=float_dtypes,
  )
  def testLogSumExpWhere(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    rng = jtu.rand_bool(self.rng())
    mask = rng(shape, bool)
    y_expected = osp_special.logsumexp(x[mask]) if mask.any() else -jnp.inf
    y_actual = lsp_special.logsumexp(x, where=mask)
    self.assertAllClose(y_expected, y_actual, check_dtypes=False)

  @jtu.sample_product(
    shape=all_shapes,
    dtype=float_dtypes,
    d=[1, 2, 5],
  )
  @jax.numpy_rank_promotion('raise')
  def testMultigammaln(self, shape, dtype, d):
    def scipy_fun(a):
      return osp_special.multigammaln(a, d)

    def lax_fun(a):
      return lsp_special.multigammaln(a, d)

    rng = jtu.rand_positive(self.rng())
    args_maker = lambda: [rng(shape, dtype) + (d - 1) / 2.]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                            tol={np.float32: 1e-3, np.float64: 1e-14})
    self._CompileAndCheck(
        lax_fun, args_maker, rtol={
            np.float32: 5e-5 if jtu.test_device_matches(["tpu"]) else 1e-05,
            np.float64: 4e-15
        })

  def testIssue980(self):
    x = np.full((4,), -1e20, dtype=np.float32)
    self.assertAllClose(np.zeros((4,), dtype=np.float32),
                        lsp_special.expit(x))

  def testIssue13267(self):
    """Tests betaln(x, 1) across wide range of x."""
    xs = jnp.geomspace(1, 1e30, 1000)
    primals_out, tangents_out = jax.jvp(lsp_special.betaln, primals=[xs, 1.0], tangents=[jnp.ones_like(xs), 0.0])
    # Check that betaln(x, 1) = -log(x).
    # Betaln is still not perfect for small values, hence the atol (but it's close)
    atol = 1e-3 if jtu.test_device_matches(["tpu"]) else 1e-5
    self.assertAllClose(primals_out, -jnp.log(xs), atol=atol)
    # Check that d/dx betaln(x, 1) = d/dx -log(x) = -1/x.
    self.assertAllClose(tangents_out, -1 / xs, atol=atol)

  def testXlogyShouldReturnZero(self):
    self.assertAllClose(lsp_special.xlogy(0., 0.), 0., check_dtypes=False)

  def testGradOfXlogyAtZero(self):
    # https://github.com/google/jax/issues/15598
    x0, y0 = 0.0, 3.0
    d_xlog1py_dx = jax.grad(lsp_special.xlogy, argnums=0)(x0, y0)
    self.assertAllClose(d_xlog1py_dx, lax.log(y0))

    d_xlog1py_dy = jax.grad(lsp_special.xlogy, argnums=1)(x0, y0)
    self.assertAllClose(d_xlog1py_dy, 0.0)

    jtu.check_grads(lsp_special.xlogy, (x0, y0), order=2)

  def testXlog1pyShouldReturnZero(self):
    self.assertAllClose(lsp_special.xlog1py(0., -1.), 0., check_dtypes=False)

  def testGradOfXlog1pyAtZero(self):
    # https://github.com/google/jax/issues/15598
    x0, y0 = 0.0, 3.0
    d_xlog1py_dx = jax.grad(lsp_special.xlog1py, argnums=0)(x0, y0)
    self.assertAllClose(d_xlog1py_dx, lax.log1p(y0))

    d_xlog1py_dy = jax.grad(lsp_special.xlog1py, argnums=1)(x0, y0)
    self.assertAllClose(d_xlog1py_dy, 0.0)

    jtu.check_grads(lsp_special.xlog1py, (x0, y0), order=2)

  def testXLogX(self):
    scipy_op = lambda x: osp_special.xlogy(x, x)
    lax_op = lsp_special_internal._xlogx
    rng = jtu.rand_positive(self.rng())
    args_maker = lambda: [rng((2, 3, 4), np.float32)]
    self._CheckAgainstNumpy(
        scipy_op, lax_op, args_maker,
        rtol=5e-4 if jtu.test_device_matches(["tpu"]) else None)
    self._CompileAndCheck(lax_op, args_maker)
    jtu.check_grads(lax_op, args_maker(), order=1,
                    atol=.1 if jtu.test_device_matches(["tpu"]) else 1e-3,
                    rtol=.1, eps=1e-3)

  def testGradOfEntrAtZero(self):
    # https://github.com/google/jax/issues/15709
    self.assertEqual(jax.jacfwd(lsp_special.entr)(0.0), jnp.inf)
    self.assertEqual(jax.jacrev(lsp_special.entr)(0.0), jnp.inf)

  @jtu.sample_product(
    [dict(order=order, z=z, n_iter=n_iter)
     for order, z, n_iter in zip(
         [0, 1, 2, 3, 6], [0.01, 1.1, 11.4, 30.0, 100.6], [5, 20, 50, 80, 200]
     )],
  )
  def testBesselJn(self, order, z, n_iter):
    def lax_fun(z):
      return lsp_special.bessel_jn(z, v=order, n_iter=n_iter)

    def scipy_fun(z):
      vals = [osp_special.jv(v, z) for v in range(order+1)]
      return np.array(vals)

    args_maker = lambda : [z]
    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, rtol=1E-6)
    self._CompileAndCheck(lax_fun, args_maker, rtol=1E-8)

  @jtu.sample_product(
    order=[3, 4],
    shape=[(2,), (3,), (4,), (3, 5), (2, 2, 3)],
    dtype=float_dtypes,
  )
  def testBesselJnRandomPositiveZ(self, order, shape, dtype):
    rng = jtu.rand_default(self.rng(), scale=1)
    points = jnp.abs(rng(shape, dtype))

    args_maker = lambda: [points]

    def lax_fun(z):
      return lsp_special.bessel_jn(z, v=order, n_iter=15)

    def scipy_fun(z):
      vals = [osp_special.jv(v, z) for v in range(order+1)]
      return np.stack(vals, axis=0)

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, rtol=1E-6)
    self._CompileAndCheck(lax_fun, args_maker, rtol=1E-8)

  @jtu.sample_product(
    l_max=[1, 2, 3, 6],
    shape=[(5,), (10,)],
    dtype=float_dtypes,
  )
  def testLpmn(self, l_max, shape, dtype):
    rng = jtu.rand_uniform(self.rng(), low=-0.2, high=0.9)
    args_maker = lambda: [rng(shape, dtype)]

    lax_fun = partial(lsp_special.lpmn, l_max, l_max)

    def scipy_fun(z, m=l_max, n=l_max):
      # scipy only supports scalar inputs for z, so we must loop here.
      vals, derivs = zip(*(osp_special.lpmn(m, n, zi) for zi in z.astype('float64')))
      return np.dstack(vals).astype(z.dtype), np.dstack(derivs).astype(z.dtype)

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker, rtol=1e-5,
                            atol=3e-3, check_dtypes=False)
    self._CompileAndCheck(lax_fun, args_maker, rtol=1E-5, atol=3e-3)

  @jtu.sample_product(
    l_max=[3, 4, 6, 32],
    shape=[(2,), (3,), (4,), (64,)],
    dtype=float_dtypes,
  )
  def testNormalizedLpmnValues(self, l_max, shape, dtype):
    rng = jtu.rand_uniform(self.rng(), low=-0.2, high=0.9)
    args_maker = lambda: [rng(shape, dtype)]

    # Note: we test only the normalized values, not the derivatives.
    lax_fun = partial(lsp_special.lpmn_values, l_max, l_max, is_normalized=True)

    def scipy_fun(z, m=l_max, n=l_max):
      # scipy only supports scalar inputs for z, so we must loop here.
      vals, _ = zip(*(osp_special.lpmn(m, n, zi) for zi in z.astype('float64')))
      a = np.dstack(vals)

      # apply the normalization
      num_m, num_l, _ = a.shape
      a_normalized = np.zeros_like(a)
      for m in range(num_m):
        for l in range(num_l):
          c0 = (2.0 * l + 1.0) * osp_special.factorial(l - m)
          c1 = (4.0 * np.pi) * osp_special.factorial(l + m)
          c2 = np.sqrt(c0 / c1)
          a_normalized[m, l] = c2 * a[m, l]
      return a_normalized.astype(z.dtype)

    self._CheckAgainstNumpy(scipy_fun, lax_fun, args_maker,
                            rtol=1e-5, atol=1e-5, check_dtypes=False)
    self._CompileAndCheck(lax_fun, args_maker, rtol=1E-6, atol=1E-6)

  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises dtype promotion
  def testSphHarmAccuracy(self):
    m = jnp.arange(-3, 3)[:, None]
    n = jnp.arange(3, 6)
    n_max = 5
    theta = 0.0
    phi = jnp.pi

    expected = lsp_special.sph_harm(m, n, theta, phi, n_max)

    actual = osp_special.sph_harm(m, n, theta, phi)

    self.assertAllClose(actual, expected, rtol=1e-8, atol=9e-5)

  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises dtype promotion
  def testSphHarmOrderZeroDegreeZero(self):
    """Tests the spherical harmonics of order zero and degree zero."""
    theta = jnp.array([0.3])
    phi = jnp.array([2.3])
    n_max = 0

    expected = jnp.array([1.0 / jnp.sqrt(4.0 * np.pi)])
    actual = jnp.real(
        lsp_special.sph_harm(jnp.array([0]), jnp.array([0]), theta, phi, n_max))

    self.assertAllClose(actual, expected, rtol=1.1e-7, atol=3e-8)

  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises dtype promotion
  def testSphHarmOrderZeroDegreeOne(self):
    """Tests the spherical harmonics of order one and degree zero."""
    theta = jnp.array([2.0])
    phi = jnp.array([3.1])
    n_max = 1

    expected = jnp.sqrt(3.0 / (4.0 * np.pi)) * jnp.cos(phi)
    actual = jnp.real(
        lsp_special.sph_harm(jnp.array([0]), jnp.array([1]), theta, phi, n_max))

    self.assertAllClose(actual, expected, rtol=2e-7, atol=6e-8)

  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises dtype promotion
  def testSphHarmOrderOneDegreeOne(self):
    """Tests the spherical harmonics of order one and degree one."""
    theta = jnp.array([2.0])
    phi = jnp.array([2.5])
    n_max = 1

    expected = (-1.0 / 2.0 * jnp.sqrt(3.0 / (2.0 * np.pi)) *
                jnp.sin(phi) * jnp.exp(1j * theta))
    actual = lsp_special.sph_harm(
        jnp.array([1]), jnp.array([1]), theta, phi, n_max)

    self.assertAllClose(actual, expected, rtol=1e-8, atol=6e-8)

  @jtu.sample_product(
    [dict(l_max=l_max, num_z=num_z)
      for l_max, num_z in zip([1, 3, 8, 10], [2, 6, 7, 8])
    ],
    dtype=jtu.dtypes.all_integer,
  )
  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises dtype promotion
  def testSphHarmForJitAndAgainstNumpy(self, l_max, num_z, dtype):
    """Tests against JIT compatibility and Numpy."""
    n_max = l_max
    shape = (num_z,)
    rng = jtu.rand_int(self.rng(), -l_max, l_max + 1)

    lsp_special_fn = partial(lsp_special.sph_harm, n_max=n_max)

    def args_maker():
      m = rng(shape, dtype)
      n = abs(m)
      theta = np.linspace(-4.0, 5.0, num_z)
      phi = np.linspace(-2.0, 1.0, num_z)
      return m, n, theta, phi

    with self.subTest('Test JIT compatibility'):
      self._CompileAndCheck(lsp_special_fn, args_maker)

    with self.subTest('Test against numpy.'):
      self._CheckAgainstNumpy(osp_special.sph_harm, lsp_special_fn, args_maker)

  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises dtype promotion
  def testSphHarmCornerCaseWithWrongNmax(self):
    """Tests the corner case where `n_max` is not the maximum value of `n`."""
    m = jnp.array([2])
    n = jnp.array([10])
    n_clipped = jnp.array([6])
    n_max = 6
    theta = jnp.array([0.9])
    phi = jnp.array([0.2])

    expected = lsp_special.sph_harm(m, n, theta, phi, n_max)

    actual = lsp_special.sph_harm(m, n_clipped, theta, phi, n_max)

    self.assertAllClose(actual, expected, rtol=1e-8, atol=9e-5)

  @jtu.sample_product(
    n_zero_sv=n_zero_svs,
    degeneracy=degeneracies,
    geometric_spectrum=geometric_spectra,
    max_sv=max_svs,
    shape=polar_shapes,
    method=methods,
    side=sides,
    nonzero_condition_number=nonzero_condition_numbers,
    dtype=jtu.dtypes.inexact,
    seed=seeds,
  )
  def testPolar(
    self, n_zero_sv, degeneracy, geometric_spectrum, max_sv, shape, method,
      side, nonzero_condition_number, dtype, seed):
    """ Tests jax.scipy.linalg.polar."""
    if not jtu.test_device_matches(["cpu"]):
      if jnp.dtype(dtype).name in ("bfloat16", "float16"):
        raise unittest.SkipTest("Skip half precision off CPU.")

    m, n = shape
    if (method == "qdwh" and ((side == "left" and m >= n) or
                              (side == "right" and m < n))):
      raise unittest.SkipTest("method=qdwh does not support these sizes")

    matrix, _ = _initialize_polar_test(self.rng(),
      shape, n_zero_sv, degeneracy, geometric_spectrum, max_sv,
      nonzero_condition_number, dtype)
    if jnp.dtype(dtype).name in ("bfloat16", "float16"):
      self.assertRaises(
        NotImplementedError, jsp.linalg.polar, matrix, method=method,
        side=side)
      return

    unitary, posdef = jsp.linalg.polar(matrix, method=method, side=side)
    if shape[0] >= shape[1]:
      should_be_eye = np.matmul(unitary.conj().T, unitary)
    else:
      should_be_eye = np.matmul(unitary, unitary.conj().T)
    tol = 650 * float(jnp.finfo(matrix.dtype).eps)
    eye_mat = np.eye(should_be_eye.shape[0], dtype=should_be_eye.dtype)
    with self.subTest('Test unitarity.'):
      self.assertAllClose(
        eye_mat, should_be_eye, atol=tol * 1000 * min(shape))

    with self.subTest('Test Hermiticity.'):
      self.assertAllClose(
        posdef, posdef.conj().T, atol=tol * jnp.linalg.norm(posdef))

    ev, _ = np.linalg.eigh(posdef)
    ev = ev[np.abs(ev) > tol * np.linalg.norm(posdef)]
    negative_ev = jnp.sum(ev < 0.)
    with self.subTest('Test positive definiteness.'):
      self.assertEqual(negative_ev, 0)

    if side == "right":
      recon = jnp.matmul(unitary, posdef, precision=lax.Precision.HIGHEST)
    elif side == "left":
      recon = jnp.matmul(posdef, unitary, precision=lax.Precision.HIGHEST)
    with self.subTest('Test reconstruction.'):
      self.assertAllClose(
        matrix, recon, atol=tol * jnp.linalg.norm(matrix))

  @jtu.sample_product(
    n_obs=[1, 3, 5],
    n_codes=[1, 2, 4],
    n_feats=[()] + [(i,) for i in range(1, 3)],
    dtype=float_dtypes + int_dtypes, # scipy doesn't support complex
  )
  def test_vq(self, n_obs, n_codes, n_feats, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((n_obs, *n_feats), dtype), rng((n_codes, *n_feats), dtype)]
    self._CheckAgainstNumpy(osp_cluster.vq.vq, lsp_cluster.vq.vq, args_maker, check_dtypes=False)
    self._CompileAndCheck(lsp_cluster.vq.vq, args_maker)

  @jtu.sample_product(
    shape=all_shapes,
    dtype=float_dtypes,
  )
  def test_spence(self, shape, dtype):
    rng = jtu.rand_positive(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    with self.subTest('Test against SciPy'):
      rtol = 1e-4 if jtu.test_device_matches(["tpu"]) else 1e-8
      self._CheckAgainstNumpy(osp_special.spence, lsp_special.spence, args_maker,
                              rtol=rtol, check_dtypes=False)

    with self.subTest('Test JIT compatibility'):
      self._CompileAndCheck(lsp_special.spence, args_maker)

    # This function is not defined for negative values, this makes sure they are nan
    with self.subTest('Test Negative Values'):
      x = -rng(shape, dtype)
      nan_array = jnp.nan * jnp.ones_like(x)
      actual = lsp_special.spence(x)
      self.assertArraysEqual(actual, nan_array, check_dtypes=False)

  @jtu.sample_product(
    [dict(yshape=yshape, xshape=xshape, dx=dx, axis=axis)
      for yshape, xshape, dx, axis in [
        ((10,), None, 1.0, -1),
        ((3, 10), None, 2.0, -1),
        ((3, 10), None, 3.0, -0),
        ((10, 3), (10,), 1.0, -2),
        ((3, 10), (10,), 1.0, -1),
        ((3, 10), (3, 10), 1.0, -1),
        ((2, 3, 10), (3, 10), 1.0, -2),
      ]
    ],
    dtype=float_dtypes + int_dtypes,
  )
  @jtu.skip_on_devices("tpu")  # TODO(jakevdp): fix and reenable this test.
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testIntegrateTrapezoid(self, yshape, xshape, dtype, dx, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(yshape, dtype), rng(xshape, dtype) if xshape is not None else None]
    np_fun = partial(scipy.integrate.trapezoid, dx=dx, axis=axis)
    jnp_fun = partial(jax.scipy.integrate.trapezoid, dx=dx, axis=axis)
    tol = jtu.tolerance(dtype, {np.float16: 2e-3, np.float64: 1e-12,
                                jax.dtypes.bfloat16: 4e-2})
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol,
                            check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol,
                          check_dtypes=False)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
