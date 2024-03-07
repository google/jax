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
import re
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import lax
from jax.ad_checkpoint import checkpoint
from jax._src import test_util as jtu
import jax.numpy as jnp  # scan tests use numpy
import jax.scipy as jsp

jax.config.parse_flags_with_absl()


def high_precision_dot(a, b):
  return lax.dot(a, b, precision=lax.Precision.HIGHEST)


def posify(matrix):
  return high_precision_dot(matrix, matrix.T.conj())


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


class CustomLinearSolveTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      {"testcase_name": "nonsymmetric", "symmetric": False},
      {"testcase_name": "symmetric", "symmetric": True},
  )
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_custom_linear_solve(self, symmetric):

    def explicit_jacobian_solve(matvec, b):
      return lax.stop_gradient(jnp.linalg.solve(jax.jacobian(matvec)(b), b))

    def matrix_free_solve(matvec, b):
      return lax.custom_linear_solve(
          matvec, b, explicit_jacobian_solve, explicit_jacobian_solve,
          symmetric=symmetric)

    def linear_solve(a, b):
      return matrix_free_solve(partial(high_precision_dot, a), b)

    rng = self.rng()
    a = rng.randn(3, 3)
    if symmetric:
      a = a + a.T
    b = rng.randn(3)
    jtu.check_grads(linear_solve, (a, b), order=2, rtol=3e-3)

    expected = jnp.linalg.solve(a, b)
    actual = jax.jit(linear_solve)(a, b)
    self.assertAllClose(expected, actual)

    c = rng.randn(3, 2)
    expected = jnp.linalg.solve(a, c)
    actual = jax.vmap(linear_solve, (None, 1), 1)(a, c)
    self.assertAllClose(expected, actual)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_custom_linear_solve_aux(self):
    def explicit_jacobian_solve_aux(matvec, b):
      x = lax.stop_gradient(jnp.linalg.solve(jax.jacobian(matvec)(b), b))
      return x, array_aux

    def matrix_free_solve_aux(matvec, b):
      return lax.custom_linear_solve(
        matvec, b, explicit_jacobian_solve_aux, explicit_jacobian_solve_aux,
        symmetric=True, has_aux=True)

    def linear_solve_aux(a, b):
      return matrix_free_solve_aux(partial(high_precision_dot, a), b)

    # array aux values, to be able to use jtu.check_grads
    array_aux = {"converged": np.array(1.), "nfev": np.array(12345.)}
    rng = self.rng()
    a = rng.randn(3, 3)
    a = a + a.T
    b = rng.randn(3)

    expected = jnp.linalg.solve(a, b)
    actual_nojit, nojit_aux = linear_solve_aux(a, b)
    actual_jit, jit_aux = jax.jit(linear_solve_aux)(a, b)

    self.assertAllClose(expected, actual_nojit)
    self.assertAllClose(expected, actual_jit)
    # scalar dict equality check
    self.assertDictEqual(nojit_aux, array_aux)
    self.assertDictEqual(jit_aux, array_aux)

    # jvp / vjp test
    jtu.check_grads(linear_solve_aux, (a, b), order=2, rtol=4e-3)

    # vmap test
    c = rng.randn(3, 2)
    expected = jnp.linalg.solve(a, c)
    expected_aux = jax.tree.map(partial(np.repeat, repeats=2), array_aux)
    actual_vmap, vmap_aux = jax.vmap(linear_solve_aux, (None, 1), -1)(a, c)

    self.assertAllClose(expected, actual_vmap)
    jtu.check_eq(expected_aux, vmap_aux)


  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  @unittest.skip("Test is too slow (> 1 minute at time of writing)")
  def test_custom_linear_solve_zeros(self):
    def explicit_jacobian_solve(matvec, b):
      return lax.stop_gradient(jnp.linalg.solve(jax.jacobian(matvec)(b), b))

    def matrix_free_solve(matvec, b):
      return lax.custom_linear_solve(matvec, b, explicit_jacobian_solve,
                                     explicit_jacobian_solve)

    def linear_solve(a, b):
      return matrix_free_solve(partial(high_precision_dot, a), b)

    rng = self.rng()
    a = rng.randn(3, 3)
    b = rng.randn(3)
    jtu.check_grads(lambda x: linear_solve(x, b), (a,), order=2,
                    rtol={np.float32: 5e-3})
    jtu.check_grads(lambda x: linear_solve(a, x), (b,), order=2,
                    rtol={np.float32: 5e-3})

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_custom_linear_solve_iterative(self):

    def richardson_iteration(matvec, b, omega=0.1, tolerance=1e-6):
      # Equivalent to vanilla gradient descent:
      # https://en.wikipedia.org/wiki/Modified_Richardson_iteration
      def cond(x):
        return jnp.linalg.norm(matvec(x) - b) > tolerance
      def body(x):
        return x + omega * (b - matvec(x))
      return lax.while_loop(cond, body, b)

    def matrix_free_solve(matvec, b):
      return lax.custom_linear_solve(matvec, b, richardson_iteration,
                                     richardson_iteration)

    def build_and_solve(a, b):
      # intentionally non-linear in a and b
      matvec = partial(high_precision_dot, jnp.exp(a))
      return matrix_free_solve(matvec, jnp.cos(b))

    # rng = self.rng()
    # This test is very sensitive to the inputs, so we use a known working seed.
    rng = np.random.RandomState(0)
    a = rng.randn(2, 2)
    b = rng.randn(2)
    expected = jnp.linalg.solve(jnp.exp(a), jnp.cos(b))
    actual = build_and_solve(a, b)
    self.assertAllClose(expected, actual, atol=1e-5)
    jtu.check_grads(build_and_solve, (a, b), atol=1e-5, order=2,
                    rtol={jnp.float32: 6e-2, jnp.float64: 2e-3})

    # vmap across an empty dimension
    jtu.check_grads(
        jax.vmap(build_and_solve), (a[None, :, :], b[None, :]),
        atol=1e-5,
        order=2,
        rtol={jnp.float32: 6e-2, jnp.float64: 2e-3})

  def test_custom_linear_solve_cholesky(self):

    def positive_definite_solve(a, b):
      factors = jsp.linalg.cho_factor(a)
      def solve(matvec, x):
        return jsp.linalg.cho_solve(factors, x)
      matvec = partial(high_precision_dot, a)
      return lax.custom_linear_solve(matvec, b, solve, symmetric=True)

    rng = self.rng()
    a = rng.randn(2, 2)
    b = rng.randn(2)

    tol = {np.float32: 1E-3 if jtu.test_device_matches(["tpu"]) else 2E-4,
           np.float64: 1E-12}
    expected = jnp.linalg.solve(np.asarray(posify(a)), b)
    actual = positive_definite_solve(posify(a), b)
    self.assertAllClose(expected, actual, rtol=tol, atol=tol)

    actual = jax.jit(positive_definite_solve)(posify(a), b)
    self.assertAllClose(expected, actual, rtol=tol, atol=tol)

    # numerical gradients are only well defined if ``a`` is guaranteed to be
    # positive definite.
    jtu.check_grads(
        lambda x, y: positive_definite_solve(posify(x), y),
        (a, b), order=2, rtol=0.3)

  def test_custom_linear_solve_complex(self):

    def solve(a, b):
      def solve(matvec, x):
        return jsp.linalg.solve(a, x)
      def tr_solve(matvec, x):
        return jsp.linalg.solve(a.T, x)
      matvec = partial(high_precision_dot, a)
      return lax.custom_linear_solve(matvec, b, solve, tr_solve)

    rng = self.rng()
    a = 0.5 * rng.randn(2, 2) + 0.5j * rng.randn(2, 2)
    b = 0.5 * rng.randn(2) + 0.5j * rng.randn(2)
    jtu.check_grads(solve, (a, b), order=2, rtol=1e-2)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_custom_linear_solve_lu(self):

    def linear_solve(a, b):
      a_factors = jsp.linalg.lu_factor(a)
      at_factors = jsp.linalg.lu_factor(a.T)
      def solve(matvec, x):
        return jsp.linalg.lu_solve(a_factors, x)
      def transpose_solve(vecmat, x):
        return jsp.linalg.lu_solve(at_factors, x)
      return lax.custom_linear_solve(
          partial(high_precision_dot, a), b, solve, transpose_solve)

    rng = self.rng()
    a = rng.randn(3, 3)
    b = rng.randn(3)

    expected = jnp.linalg.solve(a, b)
    actual = linear_solve(a, b)
    self.assertAllClose(expected, actual)

    jtu.check_grads(linear_solve, (a, b), order=2, rtol=2e-3)

    # regression test for https://github.com/google/jax/issues/1536
    jtu.check_grads(jax.jit(linear_solve), (a, b), order=2,
                    rtol={np.float32: 2e-3})

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_custom_linear_solve_without_transpose_solve(self):

    def explicit_jacobian_solve(matvec, b):
      return lax.stop_gradient(jnp.linalg.solve(jax.jacobian(matvec)(b), b))

    def loss(a, b):
      matvec = partial(high_precision_dot, a)
      x = lax.custom_linear_solve(matvec, b, explicit_jacobian_solve)
      return jnp.sum(x)

    rng = self.rng()
    a = rng.randn(2, 2)
    b = rng.randn(2)

    jtu.check_grads(loss, (a, b), order=2, modes=['fwd'],
                    atol={np.float32: 2e-3, np.float64: 1e-11})
    jtu.check_grads(jax.vmap(loss), (a[None,:,:], b[None,:]), order=2,
                    modes=['fwd'], atol={np.float32: 2e-3, np.float64: 1e-11})

    with self.assertRaisesRegex(TypeError, "transpose_solve required"):
      jax.grad(loss)(a, b)

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  @unittest.skip("Test is too slow (> 2 minutes at time of writing)")
  def test_custom_linear_solve_pytree(self):
    """Test custom linear solve with inputs and outputs that are pytrees."""

    def unrolled_matvec(mat, x):
      """Apply a Python list of lists of scalars to a list of scalars."""
      result = []
      for i in range(len(mat)):
        v = 0
        for j in range(len(x)):
          if mat[i][j] is not None:
            v += mat[i][j] * x[j]
        result.append(v)
      return result

    def unrolled_substitution_solve(matvec, b, lower_tri):
      """Solve a triangular unrolled system with fwd/back substitution."""
      zero = jnp.zeros(())
      one = jnp.ones(())
      x = [zero for _ in b]
      ordering = range(len(b)) if lower_tri else range(len(b) - 1, -1, -1)
      for i in ordering:
        residual = b[i] - matvec(x)[i]
        diagonal = matvec([one if i == j else zero for j in range(len(b))])[i]
        x[i] = residual / diagonal
      return x

    def custom_unrolled_lower_tri_solve(mat, b):
      return lax.custom_linear_solve(
          partial(unrolled_matvec, mat), b,
          partial(unrolled_substitution_solve, lower_tri=True),
          partial(unrolled_substitution_solve, lower_tri=False))

    mat = [[1.0, None, None, None, None, None, None],
           [1.0, 1.0, None, None, None, None, None],
           [None, 1.0, 1.0, None, None, None, None],
           [None, None, 1.0, 1.0, None, None, None],
           [None, None, None, 1.0, 1.0, None, None],
           [None, None, None, None, None, 2.0, None],
           [None, None, None, None, None, 4.0, 3.0]]

    rng = self.rng()
    b = list(rng.randn(7))

    # Non-batched
    jtu.check_grads(custom_unrolled_lower_tri_solve, (mat, b), order=2,
                    rtol={jnp.float32: 2e-2})

    # Batch one element of b (which, because of unrolling, should only affect
    # the first block of outputs)
    b_bat = list(b)
    b_bat[3] = rng.randn(3)
    jtu.check_grads(
        jax.vmap(
            custom_unrolled_lower_tri_solve,
            in_axes=(None, [None, None, None, 0, None, None, None]),
            out_axes=[0, 0, 0, 0, 0, None, None]), (mat, b_bat),
        order=2,
        rtol={jnp.float32: 1e-2})

    # Batch one element of mat (again only affecting first block)
    mat[2][1] = rng.randn(3)
    mat_axis_tree = [
        [0 if i == 2 and j == 1 else None for j in range(7)] for i in range(7)
    ]
    jtu.check_grads(
        jax.vmap(
            custom_unrolled_lower_tri_solve,
            in_axes=(mat_axis_tree, None),
            out_axes=[0, 0, 0, 0, 0, None, None]), (mat, b),
        order=2)



  def test_custom_linear_solve_pytree_with_aux(self):
    # Check that lax.custom_linear_solve handles
    # pytree inputs + has_aux=True
    # https://github.com/google/jax/pull/13093

    aux_orig = {'a': 1, 'b': 2}
    b = {'c': jnp.ones(2), 'd': jnp.ones(3)}

    def solve_with_aux(matvec, b):
      return b, aux_orig

    sol, aux = lax.custom_linear_solve(
          lambda x:x,
          b,
          solve_with_aux,
          solve_with_aux,
          has_aux=True)

    assert len(aux.keys()) == 2
    assert 'a' in aux
    assert 'b' in aux
    self.assertAllClose(aux['a'], aux_orig['a'], check_dtypes=False)
    self.assertAllClose(aux['b'], aux_orig['b'], check_dtypes=False)


  def test_custom_linear_solve_errors(self):

    solve = lambda f, x: x

    with self.assertRaisesRegex(TypeError, re.escape("matvec() output pytree")):
      lax.custom_linear_solve(lambda x: [x], 1.0, solve, solve)
    with self.assertRaisesRegex(TypeError, re.escape("solve() output pytree")):
      lax.custom_linear_solve(lambda x: x, 1.0, lambda f, x: [x], solve)
    with self.assertRaisesRegex(
        TypeError, re.escape("transpose_solve() output pytree")):
      lax.custom_linear_solve(lambda x: x, 1.0, solve, lambda f, x: [x])

    with self.assertRaisesRegex(ValueError, re.escape("solve() output shapes")):
      lax.custom_linear_solve(lambda x: x, 1.0, lambda f, x: jnp.ones(2), solve)

    def bad_matvec_usage(a):
      return lax.custom_linear_solve(
          lambda x: a * jnp.ones(2), 1.0, solve, solve)
    with self.assertRaisesRegex(ValueError, re.escape("matvec() output shapes")):
      jax.jvp(bad_matvec_usage, (1.0,), (1.0,))

  def test_custom_linear_solve_new_remat(self):

    def explicit_jacobian_solve(matvec, b):
      return lax.stop_gradient(jnp.linalg.solve(jax.jacobian(matvec)(b), b))

    def matrix_free_solve(matvec, b):
      return lax.custom_linear_solve(
          matvec, b, explicit_jacobian_solve, explicit_jacobian_solve,
          symmetric=True)

    @checkpoint
    def linear_solve(a, b):
      return matrix_free_solve(partial(high_precision_dot, a), b)

    rng = self.rng()
    a = rng.randn(3, 3)
    if True:
      a = a + a.T
    b = rng.randn(3)
    jtu.check_grads(linear_solve, (a, b), order=1, rtol=3e-3, modes=['rev'])

    @partial(checkpoint, policy=lambda *_, **__: True)
    def linear_solve(a, b):
      return matrix_free_solve(partial(high_precision_dot, a), b)
    jtu.check_grads(linear_solve, (a, b), order=1, rtol=3e-3, modes=['rev'])

  def test_custom_linear_solve_batching_with_aux(self):
    def solve(mv, b):
      aux = (np.array(1.), True, 0)
      return mv(b), aux

    def solve_aux(x):
      matvec = lambda y: jax.tree.map(partial(jnp.dot, A), y)
      return lax.custom_linear_solve(matvec, (x, x), solve, solve, symmetric=True, has_aux=True)

    rng = self.rng()
    A = rng.randn(3, 3)
    A = A + A.T
    b = rng.randn(3, 3)

    # doesn't crash
    jax.vmap(solve_aux)(b)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
