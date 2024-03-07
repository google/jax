# Copyright 2020 The JAX Authors.
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


from functools import reduce, partial

from absl.testing import absltest
import numpy as np
import unittest

import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import jax.scipy.special
from jax import random
from jax import jacfwd, jit
from jax.example_libraries import stax
from jax.experimental.jet import jet, fact, zero_series
from jax import lax

jax.config.parse_flags_with_absl()

def jvp_taylor(fun, primals, series):
  # Computes the Taylor series the slow way, with nested jvp.
  order, = set(map(len, series))
  primals = tuple(jnp.asarray(p) for p in primals)
  def composition(eps):
    taylor_terms = [sum(eps ** (i+1) * terms[i] / fact(i + 1)
                         for i in range(len(terms))) for terms in series]
    nudged_args = [(x + t).astype(x.dtype) for x, t in zip(primals, taylor_terms)]
    return fun(*nudged_args)
  primal_out = fun(*primals)
  terms_out = [repeated(jacfwd, i+1)(composition)(0.) for i in range(order)]
  return primal_out, terms_out

def repeated(f, n):
  def rfun(p):
    return reduce(lambda x, _: f(x), range(n), p)
  return rfun

def transform(lims, x):
  return x * (lims[1] - lims[0]) + lims[0]

class JetTest(jtu.JaxTestCase):

  def check_jet(self, fun, primals, series, atol=1e-5, rtol=1e-5,
                check_dtypes=True):
    # Convert to jax arrays to ensure dtype canonicalization.
    primals = jax.tree.map(jnp.asarray, primals)
    series = jax.tree.map(jnp.asarray, series)

    y, terms = jet(fun, primals, series)
    expected_y, expected_terms = jvp_taylor(fun, primals, series)

    self.assertAllClose(y, expected_y, atol=atol, rtol=rtol,
                        check_dtypes=check_dtypes)

    self.assertAllClose(terms, expected_terms, atol=atol, rtol=rtol,
                        check_dtypes=check_dtypes)

  def check_jet_finite(self, fun, primals, series, atol=1e-5, rtol=1e-5,
                       check_dtypes=True):
    # Convert to jax arrays to ensure dtype canonicalization.
    primals = jax.tree.map(jnp.asarray, primals)
    series = jax.tree.map(jnp.asarray, series)

    y, terms = jet(fun, primals, series)
    expected_y, expected_terms = jvp_taylor(fun, primals, series)

    def _convert(x):
      return jnp.where(jnp.isfinite(x), x, jnp.nan)

    y = _convert(y)
    expected_y = _convert(expected_y)

    terms = _convert(jnp.asarray(terms))
    expected_terms = _convert(jnp.asarray(expected_terms))

    self.assertAllClose(y, expected_y, atol=atol, rtol=rtol,
                        check_dtypes=check_dtypes)

    self.assertAllClose(terms, expected_terms, atol=atol, rtol=rtol,
                        check_dtypes=check_dtypes)

  @jtu.skip_on_devices("tpu")
  # Default tolerance too tight on A100 after openxla/xla@a58070090
  @jax.default_matmul_precision("float32")
  def test_dot(self):
    M, K, N = 2, 3, 4
    order = 3
    rng = self.rng()
    x1 = rng.randn(M, K)
    x2 = rng.randn(K, N)
    primals = (x1, x2)
    terms_in1 = [rng.randn(*x1.shape) for _ in range(order)]
    terms_in2 = [rng.randn(*x2.shape) for _ in range(order)]
    series_in = (terms_in1, terms_in2)
    self.check_jet(jnp.dot, primals, series_in)

  @jtu.skip_on_devices("tpu")
  @jax.legacy_prng_key('allow')
  def test_conv(self):
    order = 3
    input_shape = (1, 5, 5, 1)
    key = random.PRNGKey(0)
    # TODO(duvenaud): Check all types of padding
    init_fun, apply_fun = stax.Conv(3, (2, 2), padding='VALID')
    _, (W, b) = init_fun(key, input_shape)

    rng = self.rng()

    x = rng.randn(*input_shape).astype(W.dtype)
    primals = (W, b, x)

    series_in1 = [rng.randn(*W.shape).astype(W.dtype) for _ in range(order)]
    series_in2 = [rng.randn(*b.shape).astype(W.dtype) for _ in range(order)]
    series_in3 = [rng.randn(*x.shape).astype(W.dtype) for _ in range(order)]

    series_in = (series_in1, series_in2, series_in3)

    def f(W, b, x):
      return apply_fun((W, b), x)

    self.check_jet(f, primals, series_in, check_dtypes=False)

  def unary_check(self, fun, lims=(-2, 2), order=3, dtype=None, atol=1e-3,
                  rtol=1e-3):
    dims = 2, 3
    rng = self.rng()
    if dtype is None:
      primal_in = transform(lims, rng.rand(*dims))
      terms_in = [rng.randn(*dims) for _ in range(order)]
    else:
      rng = jtu.rand_uniform(rng, *lims)
      primal_in = rng(dims, dtype)
      terms_in = [rng(dims, dtype) for _ in range(order)]
    self.check_jet(fun, (primal_in,), (terms_in,), atol, rtol)

  def binary_check(self, fun, lims=None, order=3, finite=True, dtype=None):
    lims = lims or [-2, 2]
    dims = 2, 3
    rng = self.rng()
    if isinstance(lims, tuple):
      x_lims, y_lims = lims
    else:
      x_lims, y_lims = lims, lims
    if dtype is None:
      primal_in = (transform(x_lims, rng.rand(*dims)),
                   transform(y_lims, rng.rand(*dims)))
      series_in = ([rng.randn(*dims) for _ in range(order)],
                   [rng.randn(*dims) for _ in range(order)])
    else:
      rng = jtu.rand_uniform(rng, *lims)
      primal_in = (rng(dims, dtype),
                   rng(dims, dtype))
      series_in = ([rng(dims, dtype) for _ in range(order)],
                   [rng(dims, dtype) for _ in range(order)])
    if finite:
      self.check_jet(fun, primal_in, series_in, atol=1e-4, rtol=1e-4)
    else:
      self.check_jet_finite(fun, primal_in, series_in, atol=1e-4, rtol=1e-4)

  def unary_check_float0(self, fun, lims=(-2, 2), order=3, dtype=None):
    # like unary_check but for functions that output integers (so their tangent
    # type is float0 arrays)
    raise unittest.SkipTest("jet tests must be adapted for integer-output functions")

  def binary_check_float0(self, fun, lims=(-2, 2), order=3, finite=True, dtype=None):
    # like binary_check but for functions that output integers (so their tangent
    # type is float0 arrays)
    raise unittest.SkipTest("jet tests must be adapted for integer-output functions")

  def expit_check(self, lims=(-2, 2), order=3):
    dims = 2, 3
    rng = self.rng()
    primal_in = transform(lims, rng.rand(*dims))
    terms_in = [rng.randn(*dims) for _ in range(order)]

    primals = (primal_in, )
    series = (terms_in, )

    y, terms = jax.experimental.jet._logistic_taylor(primals, series)
    expected_y, expected_terms = jvp_taylor(jax.scipy.special.expit, primals, series)

    atol = 1e-4
    rtol = 1e-4
    self.assertAllClose(y, expected_y, atol=atol, rtol=rtol)

    self.assertAllClose(terms, expected_terms, atol=atol, rtol=rtol)

  @jtu.skip_on_devices("tpu")
  def test_int_pow(self):
    for p in range(6):
      self.unary_check(lambda x: x ** p, lims=[-2, 2])
    self.unary_check(lambda x: x ** 10, lims=[0, 0])

  @jtu.skip_on_devices("tpu")
  def test_is_finite(self):  self.unary_check_float0(lax.is_finite)
  @jtu.skip_on_devices("tpu")
  def test_and(self):         self.binary_check_float0(lax.bitwise_and, dtype=np.bool_)
  @jtu.skip_on_devices("tpu")
  def test_or(self):          self.binary_check_float0(lax.bitwise_or, dtype=np.bool_)
  @jtu.skip_on_devices("tpu")
  def test_xor(self):         self.binary_check_float0(jnp.bitwise_xor, dtype=np.bool_)
  @jtu.skip_on_devices("tpu")
  def test_shift_left(self):  self.binary_check_float0(lax.shift_left, dtype=np.int32)
  @jtu.skip_on_devices("tpu")
  def test_shift_right_a(self):  self.binary_check_float0(lax.shift_right_arithmetic, dtype=np.int32)
  @jtu.skip_on_devices("tpu")
  def test_shift_right_l(self):  self.binary_check_float0(lax.shift_right_logical, dtype=np.int32)
  @jtu.skip_on_devices("tpu")
  def test_le(self):          self.binary_check_float0(lambda x, y: x <= y)
  @jtu.skip_on_devices("tpu")
  def test_gt(self):          self.binary_check_float0(lambda x, y: x > y)
  @jtu.skip_on_devices("tpu")
  def test_lt(self):          self.binary_check_float0(lambda x, y: x < y)
  @jtu.skip_on_devices("tpu")
  def test_ge(self):          self.binary_check_float0(lambda x, y: x >= y)
  @jtu.skip_on_devices("tpu")
  def test_eq(self):          self.binary_check_float0(lambda x, y: x == y)
  @jtu.skip_on_devices("tpu")
  def test_ne(self):          self.binary_check_float0(lambda x, y: x != y)
  @jtu.skip_on_devices("tpu")
  def test_not(self):        self.unary_check_float0(lax.bitwise_not, dtype=np.bool_)

  @jtu.skip_on_devices("tpu")
  def test_exp(self):        self.unary_check(jnp.exp)
  @jtu.skip_on_devices("tpu")
  def test_neg(self):        self.unary_check(jnp.negative)
  @jtu.skip_on_devices("tpu")
  def test_floor(self):      self.unary_check(jnp.floor)
  @jtu.skip_on_devices("tpu")
  def test_ceil(self):       self.unary_check(jnp.ceil)
  @jtu.skip_on_devices("tpu")
  def test_trunc(self):       self.unary_check(jnp.trunc)
  @jtu.skip_on_devices("tpu")
  def test_round(self):      self.unary_check(lax.round)
  @jtu.skip_on_devices("tpu")
  def test_sign(self):       self.unary_check(lax.sign)
  @jtu.skip_on_devices("tpu")
  def test_real(self):       self.unary_check(lax.real, dtype=np.complex64)
  @jtu.skip_on_devices("tpu")
  def test_conj(self):       self.unary_check(lax.conj, dtype=np.complex64)
  @jtu.skip_on_devices("tpu")
  def test_imag(self):       self.unary_check(lax.imag, dtype=np.complex64)
  @jtu.skip_on_devices("tpu")
  def test_log(self):        self.unary_check(jnp.log, lims=[0.8, 4.0])
  @jtu.skip_on_devices("tpu")
  def test_gather(self):     self.unary_check(lambda x: x[1:])
  @jtu.skip_on_devices("tpu")
  def test_reduce_max(self): self.unary_check(lambda x: x.max(axis=1))
  @jtu.skip_on_devices("tpu")
  def test_reduce_min(self): self.unary_check(lambda x: x.min(axis=1))
  @jtu.skip_on_devices("tpu")
  def test_all_max(self):    self.unary_check(jnp.max)
  @jtu.skip_on_devices("tpu")
  def test_all_min(self):    self.unary_check(jnp.min)
  @jtu.skip_on_devices("tpu")
  def test_stopgrad(self):   self.unary_check(lax.stop_gradient)
  @jtu.skip_on_devices("tpu")
  def test_abs(self):        self.unary_check(jnp.abs)
  @jtu.skip_on_devices("tpu")
  def test_fft(self):        self.unary_check(jnp.fft.fft)
  @jtu.skip_on_devices("tpu")
  def test_log1p(self):      self.unary_check(jnp.log1p, lims=[0, 4.])
  @jtu.skip_on_devices("tpu")
  def test_expm1(self):      self.unary_check(jnp.expm1)
  @jtu.skip_on_devices("tpu")
  def test_sin(self):        self.unary_check(jnp.sin)
  @jtu.skip_on_devices("tpu")
  def test_cos(self):        self.unary_check(jnp.cos)
  @jtu.skip_on_devices("tpu")
  def test_sinh(self):       self.unary_check(jnp.sinh)
  @jtu.skip_on_devices("tpu")
  def test_cosh(self):       self.unary_check(jnp.cosh)
  @jtu.skip_on_devices("tpu")
  def test_tanh(self):       self.unary_check(jnp.tanh, lims=[-500, 500], order=5,
                                              atol=5e-3)
  @jtu.skip_on_devices("tpu")
  def test_logistic(self):   self.unary_check(lax.logistic, lims=[-100, 100], order=5)
  @jtu.skip_on_devices("tpu")
  def test_expit2(self):     self.expit_check(lims=[-500, 500], order=5)
  @jtu.skip_on_devices("tpu")
  def test_sqrt(self):       self.unary_check(jnp.sqrt, lims=[0, 5.])
  @jtu.skip_on_devices("tpu")
  def test_rsqrt(self):      self.unary_check(lax.rsqrt, lims=[0, 5000.])
  @jtu.skip_on_devices("tpu")
  def test_asinh(self):      self.unary_check(lax.asinh, lims=[-100, 100])
  @jtu.skip_on_devices("tpu")
  def test_acosh(self):      self.unary_check(lax.acosh, lims=[-100, 100])
  @jtu.skip_on_devices("tpu")
  def test_atanh(self):      self.unary_check(lax.atanh, lims=[-1, 1])
  @jtu.skip_on_devices("tpu")
  def test_erf(self):        self.unary_check(lax.erf)
  @jtu.skip_on_devices("tpu")
  def test_erfc(self):       self.unary_check(lax.erfc)
  @jtu.skip_on_devices("tpu")
  def test_erf_inv(self):    self.unary_check(lax.erf_inv, lims=[-1, 1])
  @jtu.skip_on_devices("tpu")
  def test_cumsum(self):     self.unary_check(jnp.cumsum)
  @jtu.skip_on_devices("tpu")
  def test_cumprod(self):    self.unary_check(jnp.cumprod)
  @jtu.skip_on_devices("tpu")
  def test_cummax(self):     self.unary_check(partial(lax.cummax, axis=0))
  @jtu.skip_on_devices("tpu")
  def test_cummin(self):     self.unary_check(partial(lax.cummin, axis=0))
  @jtu.skip_on_devices("tpu")
  def test_dynamic_slice(self): self.unary_check(partial(lax.dynamic_slice, start_indices=(1,2), slice_sizes=(1,1)))
  @jtu.skip_on_devices("tpu")
  def test_dynamic_update_slice(self): self.unary_check(partial(lax.dynamic_update_slice, start_indices=(1,2), update=np.arange(6.0).reshape(2, 3)))


  @jtu.skip_on_devices("tpu")
  def test_div(self):         self.binary_check(lambda x, y: x / y, lims=[0.8, 4.0])
  @jtu.skip_on_devices("tpu")
  def test_rem(self):         self.binary_check(lax.rem, lims=[0.8, 4.0])
  @jtu.skip_on_devices("tpu")
  def test_complex(self):     self.binary_check(lax.complex)
  @jtu.skip_on_devices("tpu")
  def test_sub(self):         self.binary_check(lambda x, y: x - y)
  @jtu.skip_on_devices("tpu")
  def test_add(self):         self.binary_check(lambda x, y: x + y)
  @jtu.skip_on_devices("tpu")
  def test_mul(self):         self.binary_check(lambda x, y: x * y)
  @jtu.skip_on_devices("tpu")
  def test_max(self):         self.binary_check(lax.max)
  @jtu.skip_on_devices("tpu")
  def test_min(self):         self.binary_check(lax.min)
  @jtu.skip_on_devices("tpu")
  @jtu.ignore_warning(message="overflow encountered in power")
  def test_pow(self):         self.binary_check(lambda x, y: x ** y, lims=([0.2, 500], [-500, 500]), finite=False)
  @jtu.skip_on_devices("tpu")
  def test_atan2(self):       self.binary_check(lax.atan2, lims=[-40, 40])

  @jtu.skip_on_devices("tpu")
  def test_clamp(self):
    lims = [-1, 1]
    order = 3
    dims = 2, 3
    # TODO(jakevdp): This test is very sensitive to the inputs, so we use a known
    # working seed. We should instead use self.rng(), and make sure that the primal
    # points lie outside an epsilon ball of the two critical points in the function.
    rng = np.random.RandomState(0)
    primal_in = (transform(lims, rng.rand(*dims)),
                 transform(lims, rng.rand(*dims)),
                 transform(lims, rng.rand(*dims)))
    series_in = ([rng.randn(*dims) for _ in range(order)],
                 [rng.randn(*dims) for _ in range(order)],
                 [rng.randn(*dims) for _ in range(order)])

    self.check_jet(lax.clamp, primal_in, series_in, atol=1e-4, rtol=1e-4)

  def test_process_call(self):
    def f(x):
      return jit(lambda x: x * x)(x)
    self.unary_check(f, rtol=2e-4)

  def test_post_process_call(self):
    def f(x):
      return jit(lambda y: x * y)(2.)

    self.unary_check(f, rtol=5e-4)

  def test_select(self):
    M, K = 2, 3
    order = 3
    rng = self.rng()
    b = rng.rand(M, K) < 0.5
    x = rng.randn(M, K)
    y = rng.randn(M, K)
    primals = (b, x, y)
    terms_b = [rng.randn(*b.shape) for _ in range(order)]
    terms_x = [rng.randn(*x.shape) for _ in range(order)]
    terms_y = [rng.randn(*y.shape) for _ in range(order)]
    series_in = (terms_b, terms_x, terms_y)
    # Since this nudges bool inputs, we need to allow promotion to float.
    with jax.numpy_dtype_promotion('standard'):
      self.check_jet(jnp.where, primals, series_in, rtol=5e-4)

  def test_inst_zero(self):
    def f(x):
      return jnp.full_like(x, 2.)
    def g(x):
      return 2. + 0 * x
    x = jnp.ones(1)
    order = 3
    f_out_primals, f_out_series = jet(f, (x, ), ([jnp.ones_like(x) for _ in range(order)], ))
    assert f_out_series is not zero_series

    g_out_primals, g_out_series = jet(g, (x, ), ([jnp.ones_like(x) for _ in range(order)], ))

    self.assertArraysEqual(g_out_primals, f_out_primals)
    self.assertArraysEqual(g_out_series, f_out_series)

  def test_add_any(self):
    # https://github.com/google/jax/issues/5217
    f = lambda x, eps: x * eps + eps + x
    def g(eps):
      x = jnp.array(1.)
      return jax.grad(f)(x, eps)
    jet(g, (1.,), ([1.],))  # doesn't crash

  def test_scatter_add(self):
    # very basic test from https://github.com/google/jax/issues/5365
    def f(x):
      x0 = x[0]
      x1 = x[1]
      return (x0**5 + x1**5).sum()

    def h(eps):
      from jax import jacfwd, grad

      x = jnp.array([1., 1.])
      μ = eps * x

      def F(t):
        return f(x + t * μ)

      return grad(jacfwd(F))(0.)

    self.check_jet(h, (0.,), ([1., 2., 3.],), rtol=1e-3)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
