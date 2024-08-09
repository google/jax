import numpy as np

import jax.numpy as jnp

#local imports
from . import bdf_util

#Adapted from tensorflow_probabilty at
#https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/bdf_util_test.py
#Accessed 2020-06-24


def test_first_step_size_is_large_when_ode_fn_is_constant():
  initial_state_vec = jnp.array([1.], dtype=jnp.float64)
  atol = jnp.array(1e-12, dtype=jnp.float64)
  first_order_bdf_coefficient = -0.1850
  first_order_error_coefficient = first_order_bdf_coefficient + 0.5
  initial_time = jnp.array(0., dtype=jnp.float64)
  ode_fn_vec = lambda time, state: 1.
  rtol = jnp.array(1e-8, dtype=jnp.float64)
  safety_factor = jnp.array(0.9, dtype=jnp.float64)
  max_step_size = 1.
  step_size = bdf_util.first_step_size(atol,
                                       first_order_error_coefficient,
                                       initial_state_vec,
                                       initial_time,
                                       ode_fn_vec,
                                       rtol,
                                       safety_factor,
                                       max_step_size=max_step_size)
  # Step size should be maximal.
  np.testing.assert_allclose(np.asarray(max_step_size, dtype=np.float64),
                             np.asarray(step_size, dtype=np.float64),
                             err_msg='step size is not equal')


def test_interpolation_matrix_unit_step_size_ratio():
  order = jnp.array(bdf_util.MAX_ORDER, dtype=jnp.int32)
  step_size_ratio = jnp.array(1., dtype=jnp.float64)
  interpolation_matrix = bdf_util.interpolation_matrix(order, step_size_ratio)
  np.testing.assert_allclose(
      interpolation_matrix,
      jnp.array([[-1., -0., -0., -0., -0.], [-2., 1., 0., 0., 0.],
                 [-3., 3., -1., -0., -0.], [-4., 6., -4., 1., 0.],
                 [-5., 10., -10., 5., -1.]],
                dtype=jnp.float64),
      err_msg='Interpolation matrices are not equal')


def test_interpolate_backward_differences_zeroth_order_is_unchanged():
  backward_differences = jnp.array(np.random.normal(size=((bdf_util.MAX_ORDER +
                                                           3, 3))),
                                   dtype=jnp.float64)
  step_size_ratio = jnp.array(0.5, dtype=jnp.float64)
  interpolated_backward_differences = (
      bdf_util.interpolate_backward_differences(backward_differences,
                                                bdf_util.MAX_ORDER,
                                                step_size_ratio))
  np.testing.assert_allclose(
      backward_differences[0],
      interpolated_backward_differences[0],
      err_msg='Interpolated backward differences are not equal')


def test_newton_order_one():
  jacobian_mat = jnp.array([[-1.]], dtype=jnp.float64)
  bdf_coefficient = jnp.array(-0.1850, dtype=jnp.float64)
  first_order_newton_coefficient = 1. / (1. - bdf_coefficient)
  step_size = jnp.array(0.01, dtype=jnp.float64)
  unitary, upper = bdf_util.newton_qr(jacobian_mat,
                                      first_order_newton_coefficient,
                                      step_size)

  backward_differences = jnp.array([[1.], [-1.], [0.], [0.], [0.], [0.]],
                                   dtype=jnp.float64)
  ode_fn_vec = lambda time, state: -state
  order = jnp.array(1, dtype=jnp.int32)
  time = jnp.array(0., dtype=jnp.float64)
  tol = jnp.array(1e-6, dtype=jnp.float64)

  # The equation we are trying to solve with Newton's method is linear.
  # Therefore, we should observe exact convergence after one iteration. An
  # additional iteration is required to obtain an accurate error estimate,
  # making the total number of iterations 2.
  max_num_newton_iters = 2

  converged, next_backward_difference, next_state, _ = bdf_util.newton(
      backward_differences, max_num_newton_iters,
      first_order_newton_coefficient, ode_fn_vec, order, step_size, time, tol,
      unitary, upper)
  np.testing.assert_equal(np.asarray(converged), True)

  state = backward_differences[0, :]
  exact_next_state = ((1. - bdf_coefficient) * state +
                      bdf_coefficient) / (1. + step_size - bdf_coefficient)

  np.testing.assert_allclose(next_backward_difference, exact_next_state)
  np.testing.assert_allclose(next_state, exact_next_state)


if __name__ == "__main__":
  test_first_step_size_is_large_when_ode_fn_is_constant()
  test_interpolation_matrix_unit_step_size_ratio()
  test_interpolate_backward_differences_zeroth_order_is_unchanged()
  test_newton_order_one()
