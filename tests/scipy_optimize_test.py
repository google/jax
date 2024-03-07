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

from absl.testing import absltest
import numpy as np
import scipy
import scipy.optimize

import jax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax import jit
import jax.scipy.optimize

jax.config.parse_flags_with_absl()


def rosenbrock(np):
  def func(x):
    return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)

  return func


def himmelblau(np):
  def func(p):
    x, y = p
    return (x ** 2 + y - 11.) ** 2 + (x + y ** 2 - 7.) ** 2

  return func


def matyas(np):
  def func(p):
    x, y = p
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

  return func


def eggholder(np):
  def func(p):
    x, y = p
    return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2. + y + 47.))) - x * np.sin(
      np.sqrt(np.abs(x - (y + 47.))))

  return func


def zakharovFromIndices(x, ii):
  sum1 = (x**2).sum()
  sum2 = (0.5*ii*x).sum()
  answer = sum1+sum2**2+sum2**4
  return answer


class TestBFGS(jtu.JaxTestCase):

  @jtu.sample_product(
    maxiter=[None],
    func_and_init=[(rosenbrock, np.zeros(2, dtype='float32')),
                   (himmelblau, np.ones(2, dtype='float32')),
                   (matyas, np.ones(2) * 6.),
                   (eggholder, np.ones(2) * 100.)],
  )
  def test_minimize(self, maxiter, func_and_init):
    # Note, cannot compare step for step with scipy BFGS because our line search is _slightly_ different.

    func, x0 = func_and_init

    @jit
    def min_op(x0):
      result = jax.scipy.optimize.minimize(
          func(jnp),
          x0,
          method='BFGS',
          options=dict(maxiter=maxiter, gtol=1e-6),
      )
      return result.x

    jax_res = min_op(x0)
    # Newer scipy versions perform poorly in float32. See
    # https://github.com/scipy/scipy/issues/19024.
    x0_f64 = x0.astype('float64')
    scipy_res = scipy.optimize.minimize(func(np), x0_f64, method='BFGS').x
    self.assertAllClose(scipy_res, jax_res, atol=2e-4, rtol=2e-4,
                        check_dtypes=False)

  def test_fixes4594(self):
    n = 2
    A = jnp.eye(n) * 1e4
    def f(x):
      return jnp.mean((A @ x) ** 2)
    results = jax.scipy.optimize.minimize(f, jnp.ones(n), method='BFGS')
    self.assertAllClose(results.x, jnp.zeros(n), atol=1e-6, rtol=1e-6)

  @jtu.skip_on_flag('jax_enable_x64', False)
  def test_zakharov(self):
    def zakharov_fn(x):
      ii = jnp.arange(1, len(x) + 1, step=1, dtype=x.dtype)
      answer = zakharovFromIndices(x=x, ii=ii)
      return answer

    x0 = jnp.array([600.0, 700.0, 200.0, 100.0, 90.0, 1e4])
    eval_func = jax.jit(zakharov_fn)
    jax_res = jax.scipy.optimize.minimize(fun=eval_func, x0=x0, method='BFGS')
    self.assertLess(jax_res.fun, 1e-6)

  def test_minimize_bad_initial_values(self):
    # This test runs deliberately "bad" initial values to test that handling
    # of failed line search, etc. is the same across implementations
    initial_value = jnp.array([92, 0.001])
    opt_fn = himmelblau(jnp)
    jax_res = jax.scipy.optimize.minimize(
        fun=opt_fn,
        x0=initial_value,
        method='BFGS',
    ).x
    scipy_res = scipy.optimize.minimize(
        fun=opt_fn,
        jac=jax.grad(opt_fn),
        method='BFGS',
        x0=initial_value
    ).x
    self.assertAllClose(scipy_res, jax_res, atol=2e-5, check_dtypes=False)


  def test_args_must_be_tuple(self):
    A = jnp.eye(2) * 1e4
    def f(x):
      return jnp.mean((A @ x) ** 2)
    with self.assertRaisesRegex(TypeError, "args .* must be a tuple"):
      jax.scipy.optimize.minimize(f, jnp.ones(2), args=45, method='BFGS')


class TestLBFGS(jtu.JaxTestCase):

  @jtu.sample_product(
    maxiter=[None],
    func_and_init=[(rosenbrock, np.zeros(2)),
                   (himmelblau, np.zeros(2)),
                   (matyas, np.ones(2) * 6.),
                   (eggholder, np.ones(2) * 100.)],
  )
  def test_minimize(self, maxiter, func_and_init):

    func, x0 = func_and_init

    @jit
    def min_op(x0):
      result = jax.scipy.optimize.minimize(
          func(jnp),
          x0,
          method='l-bfgs-experimental-do-not-rely-on-this',
          options=dict(maxiter=maxiter, gtol=1e-7),
      )
      return result.x

    jax_res = min_op(x0)

    # Newer scipy versions perform poorly in float32. See
    # https://github.com/scipy/scipy/issues/19024.
    x0_f64 = x0.astype('float64')
    # Note that without bounds, L-BFGS-B is just L-BFGS
    with jtu.ignore_warning(category=DeprecationWarning,
                            message=".*tostring.*is deprecated.*"):
      scipy_res = scipy.optimize.minimize(func(np), x0_f64, method='L-BFGS-B').x

    if func.__name__ == 'matyas':
      # scipy performs badly for Matyas, compare to true minimum instead
      self.assertAllClose(jax_res, jnp.zeros_like(jax_res), atol=1e-7)
      return

    if func.__name__ == 'eggholder':
      # L-BFGS performs poorly for the eggholder function.
      # Neither scipy nor jax find the true minimum, so we can only loosely (with high atol) compare the false results
      self.assertAllClose(jax_res, scipy_res, atol=1e-3)
      return

    self.assertAllClose(jax_res, scipy_res, atol=2e-5, check_dtypes=False)

  def test_minimize_complex_sphere(self):
    z0 = jnp.array([1., 2. - 3.j, 4., -5.j])

    def f(z):
      return jnp.real(jnp.dot(jnp.conj(z - z0), z - z0))

    @jit
    def min_op(x0):
      result = jax.scipy.optimize.minimize(
          f,
          x0,
          method='l-bfgs-experimental-do-not-rely-on-this',
          options=dict(gtol=1e-6),
      )
      return result.x

    jax_res = min_op(jnp.zeros_like(z0))

    self.assertAllClose(jax_res, z0)

  def test_complex_rosenbrock(self):
    complex_dim = 5

    f_re = rosenbrock(jnp)
    init_re = jnp.zeros((2 * complex_dim,), dtype=complex)
    expect_re = jnp.ones((2 * complex_dim,), dtype=complex)

    def f(z):
      x_re = jnp.concatenate([jnp.real(z), jnp.imag(z)])
      return f_re(x_re)

    init = init_re[:complex_dim] + 1.j * init_re[complex_dim:]
    expect = expect_re[:complex_dim] + 1.j * expect_re[complex_dim:]

    @jit
    def min_op(z0):
      result = jax.scipy.optimize.minimize(
          f,
          z0,
          method='l-bfgs-experimental-do-not-rely-on-this',
          options=dict(gtol=1e-6),
      )
      return result.x

    jax_res = min_op(init)
    self.assertAllClose(jax_res, expect, atol=2e-5)


if __name__ == "__main__":
  absltest.main()
