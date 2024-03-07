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

import re

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import lax
from jax._src import test_util as jtu
import jax.numpy as jnp  # scan tests use numpy
import jax.scipy as jsp

jax.config.parse_flags_with_absl()


def high_precision_dot(a, b):
  return lax.dot(a, b, precision=lax.Precision.HIGHEST)


# Simple optimization routine for testing custom_root
def binary_search(func, x0, low=0.0, high=100.0):
  del x0  # unused

  def cond(state):
    low, high = state
    midpoint = 0.5 * (low + high)
    return (low < midpoint) & (midpoint < high)

  def body(state):
    low, high = state
    midpoint = 0.5 * (low + high)
    update_upper = func(midpoint) > 0
    low = jnp.where(update_upper, low, midpoint)
    high = jnp.where(update_upper, midpoint, high)
    return (low, high)

  solution, _ = lax.while_loop(cond, body, (low, high))
  return solution

# Optimization routine for testing custom_root.
def newton_raphson(func, x0):
  tol = 1e-16
  max_it = 20

  fx0, dfx0 = func(x0), jax.jacobian(func)(x0)
  initial_state = (0, x0, fx0, dfx0)  # (iteration, x, f(x), grad(f)(x))

  def cond(state):
    it, _, fx, _ = state
    return (jnp.max(jnp.abs(fx)) > tol) & (it < max_it)

  def body(state):
    it, x, fx, dfx = state
    step = jnp.linalg.solve(
      dfx.reshape((-1, fx.size)), fx.ravel()
    ).reshape(fx.shape)
    x_next = x - step
    fx, dfx = func(x_next), jax.jacobian(func)(x_next)
    return (it + 1, x_next, fx, dfx)

  _, x, _, _ = lax.while_loop(cond, body, initial_state)

  return x


class CustomRootTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      {"testcase_name": "binary_search", "solve_method": binary_search},
      {"testcase_name": "newton_raphson", "solve_method": newton_raphson},
  )
  def test_custom_root_scalar(self, solve_method):

    def scalar_solve(f, y):
      return y / f(1.0)

    def sqrt_cubed(x, tangent_solve=scalar_solve):
      f = lambda y: y ** 2 - x ** 3
      # Note: Nonzero derivative at x0 required for newton_raphson
      return lax.custom_root(f, 1.0, solve_method, tangent_solve)

    value, grad = jax.value_and_grad(sqrt_cubed)(5.0)
    self.assertAllClose(value, 5 ** 1.5, check_dtypes=False, rtol=1e-6)
    rtol = 5e-6 if jtu.test_device_matches(["tpu"]) else 1e-7
    self.assertAllClose(grad, jax.grad(pow)(5.0, 1.5), check_dtypes=False,
                        rtol=rtol)
    jtu.check_grads(sqrt_cubed, (5.0,), order=2,
                    rtol={jnp.float32: 1e-2, jnp.float64: 1e-3})

    inputs = jnp.array([4.0, 5.0])
    results = jax.vmap(sqrt_cubed)(inputs)
    self.assertAllClose(
      results, inputs ** 1.5, check_dtypes=False,
      atol={jnp.float32: 1e-3, jnp.float64: 1e-6},
      rtol={jnp.float32: 1e-3, jnp.float64: 1e-6},
    )

    results = jax.jit(sqrt_cubed)(5.0)
    self.assertAllClose(
        results, 5.0**1.5, check_dtypes=False, rtol={np.float64: 1e-7})

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_custom_root_vector_with_solve_closure(self):

    def vector_solve(f, y):
      return jnp.linalg.solve(jax.jacobian(f)(y), y)

    def linear_solve(a, b):
      f = lambda y: high_precision_dot(a, y) - b
      x0 = jnp.zeros_like(b)
      solution = jnp.linalg.solve(a, b)
      oracle = lambda func, x0: solution
      return lax.custom_root(f, x0, oracle, vector_solve)

    rng = self.rng()
    a = rng.randn(2, 2)
    b = rng.randn(2)
    jtu.check_grads(linear_solve, (a, b), order=2,
                    atol={np.float32: 1e-2, np.float64: 1e-11})

    actual = jax.jit(linear_solve)(a, b)
    expected = jnp.linalg.solve(a, b)
    self.assertAllClose(expected, actual)

  def test_custom_root_vector_nonlinear(self):

    def nonlinear_func(x, y):
      # func(x, y) == 0 if and only if x == y.
      return (x - y) * (x**2 + y**2 + 1)

    def tangent_solve(g, y):
      return jnp.linalg.solve(
        jax.jacobian(g)(y).reshape(-1, y.size),
        y.ravel()
      ).reshape(y.shape)

    def nonlinear_solve(y):
      f = lambda x: nonlinear_func(x, y)
      x0 = -jnp.ones_like(y)
      return lax.custom_root(f, x0, newton_raphson, tangent_solve)

    y = self.rng().randn(3, 1)
    jtu.check_grads(nonlinear_solve, (y,), order=2,
                rtol={jnp.float32: 1e-2, jnp.float64: 1e-3})

    actual = jax.jit(nonlinear_solve)(y)
    self.assertAllClose(y, actual, rtol=1e-5, atol=1e-5)

  def test_custom_root_with_custom_linear_solve(self):

    def linear_solve(a, b):
      f = lambda x: high_precision_dot(a, x) - b
      factors = jsp.linalg.cho_factor(a)
      cho_solve = lambda f, b: jsp.linalg.cho_solve(factors, b)
      def pos_def_solve(g, b):
        return lax.custom_linear_solve(g, b, cho_solve, symmetric=True)
      return lax.custom_root(f, b, cho_solve, pos_def_solve)

    rng = self.rng()
    a = rng.randn(2, 2)
    b = rng.randn(2)

    actual = linear_solve(high_precision_dot(a, a.T), b)
    expected = jnp.linalg.solve(high_precision_dot(a, a.T), b)
    self.assertAllClose(expected, actual)

    actual = jax.jit(linear_solve)(high_precision_dot(a, a.T), b)
    expected = jnp.linalg.solve(high_precision_dot(a, a.T), b)
    self.assertAllClose(expected, actual)

    jtu.check_grads(lambda x, y: linear_solve(high_precision_dot(x, x.T), y),
                    (a, b), order=2, rtol={jnp.float32: 1e-2})

  def test_custom_root_with_aux(self):
    def root_aux(a, b):
      f = lambda x: high_precision_dot(a, x) - b
      factors = jsp.linalg.cho_factor(a)
      cho_solve = lambda f, b: (jsp.linalg.cho_solve(factors, b), orig_aux)

      def pos_def_solve(g, b):
        # prune aux to allow use as tangent_solve
        cho_solve_noaux = lambda f, b: cho_solve(f, b)[0]
        return lax.custom_linear_solve(g, b, cho_solve_noaux, symmetric=True)

      return lax.custom_root(f, b, cho_solve, pos_def_solve, has_aux=True)

    orig_aux = {"converged": np.array(1.), "nfev": np.array(12345.), "grad": np.array([1.0, 2.0, 3.0])}

    rng = self.rng()
    a = rng.randn(2, 2)
    b = rng.randn(2)

    actual, actual_aux = root_aux(high_precision_dot(a, a.T), b)
    actual_jit, actual_jit_aux = jax.jit(root_aux)(high_precision_dot(a, a.T), b)
    expected = jnp.linalg.solve(high_precision_dot(a, a.T), b)

    self.assertAllClose(expected, actual)
    self.assertAllClose(expected, actual_jit)
    jtu.check_eq(actual_jit_aux, orig_aux)

    # grad check with aux
    jtu.check_grads(lambda x, y: root_aux(high_precision_dot(x, x.T), y),
                    (a, b), order=2, rtol={jnp.float32: 1e-2, np.float64: 3e-5})

    # test vmap and jvp combined by jacfwd
    fwd = jax.jacfwd(lambda x, y: root_aux(high_precision_dot(x, x.T), y), argnums=(0, 1))
    expected_fwd = jax.jacfwd(lambda x, y: jnp.linalg.solve(high_precision_dot(x, x.T), y), argnums=(0, 1))

    fwd_val, fwd_aux = fwd(a, b)
    expected_fwd_val = expected_fwd(a, b)
    self.assertAllClose(fwd_val, expected_fwd_val, rtol={np.float32: 5E-6, np.float64: 5E-12})

    jtu.check_close(fwd_aux, jax.tree.map(jnp.zeros_like, fwd_aux))

  def test_custom_root_errors(self):
    with self.assertRaisesRegex(TypeError, re.escape("f() output pytree")):
      lax.custom_root(lambda x: (x, x), 0.0, lambda f, x: x, lambda f, x: x)
    with self.assertRaisesRegex(TypeError, re.escape("solve() output pytree")):
      lax.custom_root(lambda x: x, 0.0, lambda f, x: (x, x), lambda f, x: x)

    def dummy_root_usage(x):
      f = lambda y: x - y
      return lax.custom_root(f, 0.0, lambda f, x: x, lambda f, x: (x, x))

    with self.assertRaisesRegex(
        TypeError, re.escape("tangent_solve() output pytree")):
      jax.jvp(dummy_root_usage, (0.0,), (0.0,))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
