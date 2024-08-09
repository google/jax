from collections import namedtuple
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp

#Adapted from tensorflow_probabilty at
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/bdf_util.py
#Accessed 2020-06-24

MAX_ORDER = 5
ORDERS = jnp.arange(0, MAX_ORDER + 1)
RECIPROCAL_SUMS = jnp.concatenate(
    [jnp.array([jnp.nan]), jnp.cumsum(1. / ORDERS[1:])])


def register_pytree_namedtuple(cls: object):
  jax.tree_util.register_pytree_node(cls, lambda xs: (tuple(xs), None),
                                     lambda _, xs: cls(*xs))


def error_ratio(backward_difference: jnp.ndarray,
                error_coefficient: jnp.ndarray,
                tol: Union[jnp.ndarray, float]) -> jnp.ndarray:
  """Computes the ratio of the error in the computed state to the tolerance."""
  error_ratio_ = jnp.linalg.norm(error_coefficient * backward_difference / tol)
  return error_ratio_


def first_step_size(atol: Union[jnp.ndarray, float],
                    first_order_error_coefficient: Union[jnp.ndarray, float],
                    initial_state_vec: jnp.ndarray,
                    initial_time: Union[jnp.ndarray, float],
                    ode_fn_vec: Callable,
                    rtol: Union[jnp.ndarray, float],
                    safety_factor: Union[jnp.ndarray, float],
                    epsilon: float = 1e-12,
                    max_step_size: float = 1.,
                    min_step_size: float = 1e-12) -> jnp.ndarray:
  """Selects the first step size to use."""
  next_time = initial_time + epsilon
  first_derivative = ode_fn_vec(initial_time, initial_state_vec)
  next_state_vec = initial_state_vec + first_derivative * epsilon
  second_derivative = (ode_fn_vec(next_time, next_state_vec) -
                       first_derivative) / epsilon
  tol = atol + (rtol * jnp.abs(initial_state_vec))
  # Local truncation error of an order one step is
  # `err(step_size) = first_order_error_coefficient * second_derivative *
  #                 * step_size**2`.
  # Choose the largest `step_size` such that `norm(err(step_size) / tol) <= 1`.
  norm = jnp.linalg.norm(first_order_error_coefficient * second_derivative /
                         tol)
  step_size = jax.lax.rsqrt(norm)
  return jnp.clip(safety_factor * step_size, min_step_size, max_step_size)


def interpolation_matrix(
    order: Union[jnp.ndarray, int],
    step_size_ratio: Union[jnp.ndarray, float, int]) -> jnp.ndarray:
  """Creates the matrix used to interpolate backward differences."""
  orders = jnp.arange(1, MAX_ORDER + 1)
  i = orders[:, jnp.newaxis]
  j = orders[jnp.newaxis, :]
  # Matrix whose (i, j)-th entry (`1 <= i, j <= order`) is
  # `1/j! (0 - i * step_size_ratio) * ... * ((j-1) - i * step_size_ratio)`.
  full_interpolation_matrix = jnp.cumprod(((j - 1) - i * step_size_ratio) / j,
                                          axis=1)
  zeros_matrix = jnp.zeros_like(full_interpolation_matrix)
  interpolation_matrix_ = jnp.where(
      jnp.arange(1, MAX_ORDER + 1) <= order,
      jnp.transpose(
          jnp.where(
              jnp.arange(1, MAX_ORDER + 1) <= order,
              jnp.transpose(full_interpolation_matrix), zeros_matrix)),
      zeros_matrix)
  return interpolation_matrix_


def interpolate_backward_differences(
    backward_differences: jnp.ndarray, order: Union[jnp.ndarray, int],
    step_size_ratio: Union[jnp.ndarray, float, int]) -> jnp.ndarray:
  """Updates backward differences when a change in the step size occurs."""
  interpolation_matrix_ = interpolation_matrix(order, step_size_ratio)
  interpolation_matrix_unit_step_size_ratio = interpolation_matrix(order, 1.)
  interpolated_backward_differences_orders_one_to_five = jnp.matmul(
      interpolation_matrix_unit_step_size_ratio,
      jnp.matmul(interpolation_matrix_, backward_differences[1:MAX_ORDER + 1]))
  interpolated_backward_differences = jnp.concatenate([
      backward_differences[0].reshape(1,
                                      jnp.shape(backward_differences)[1]),
      interpolated_backward_differences_orders_one_to_five,
      jnp.zeros(
          jnp.stack(
              jnp.array([2, jnp.shape(backward_differences)[1]],
                        dtype=jnp.int64)))
  ],
                                                      axis=0)
  return interpolated_backward_differences


class _NewtonIterand(
    namedtuple("NewtonIterand", [
        "converged", "finished", "next_backward_difference", "next_state_vec",
        "num_iters", "prev_delta_norm"
    ])):  #pylint: disable=inherit-non-class
  """
    namedtuple class to store Newton iterand
    """
  def __new__(cls, converged, finished, next_backward_difference,
              next_state_vec, num_iters, prev_delta_norm):
    return super(_NewtonIterand,
                 cls).__new__(cls, converged, finished,
                              next_backward_difference, next_state_vec,
                              num_iters, prev_delta_norm)


register_pytree_namedtuple(_NewtonIterand)  #JAX pytree


def newton_qr(
    jacobian_mat: jnp.ndarray, newton_coefficient: Union[jnp.ndarray, float,
                                                         int],
    step_size: Union[jnp.ndarray, float,
                     int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """QR factorizes the matrix used in each iteration of Newton's method."""
  identity = jnp.eye(jnp.shape(jacobian_mat)[0], dtype=jacobian_mat.dtype)
  newton_matrix = (identity - step_size * newton_coefficient * jacobian_mat)
  q, r = jnp.linalg.qr(newton_matrix)
  return q, r


def newton(
    backward_differences: jnp.ndarray, max_num_iters: Union[jnp.ndarray, float,
                                                            int],
    newton_coefficient: Union[jnp.ndarray, float, int], ode_fn_vec: Callable,
    order: Union[jnp.ndarray, float,
                 int], step_size: Union[jnp.ndarray, float,
                                        int], time: Union[jnp.ndarray, float,
                                                          int],
    tol: Union[jnp.ndarray, float], unitary: jnp.ndarray, upper: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Runs Newton's method to solve the BDF equation."""
  initial_guess = jnp.sum(jnp.where(
      jnp.arange(MAX_ORDER + 1).reshape(-1, 1) <= order,
      backward_differences[:MAX_ORDER + 1],
      jnp.zeros_like(backward_differences)[:MAX_ORDER + 1]),
                          axis=0)

  rhs_constant_term = newton_coefficient * jnp.sum(jnp.where(
      jnp.arange(1, MAX_ORDER + 1).reshape(-1, 1) <= order,
      RECIPROCAL_SUMS[1:, jnp.newaxis] * backward_differences[1:MAX_ORDER + 1],
      jnp.zeros_like(backward_differences)[1:MAX_ORDER + 1]),
                                                   axis=0)

  next_time = time + step_size

  def newton_body(iterand):
    """Performs one iteration of Newton's method."""
    next_backward_difference = iterand.next_backward_difference
    next_state_vec = iterand.next_state_vec

    rhs = newton_coefficient * step_size * ode_fn_vec(
        next_time,
        next_state_vec) - rhs_constant_term - next_backward_difference
    delta = jnp.squeeze(
        jax.scipy.linalg.solve_triangular(upper,
                                          jnp.matmul(jnp.transpose(unitary),
                                                     rhs[:, jnp.newaxis]),
                                          lower=False))
    num_iters = iterand.num_iters + 1

    next_backward_difference += delta
    next_state_vec += delta

    delta_norm = jnp.linalg.norm(delta)
    lipschitz_const = delta_norm / iterand.prev_delta_norm

    # Stop if method has converged.
    approx_dist_to_sol = lipschitz_const / (1. - lipschitz_const) * delta_norm
    close_to_sol = approx_dist_to_sol < tol
    delta_norm_is_zero = jnp.equal(delta_norm, jnp.array(0.,
                                                         dtype=jnp.float64))
    converged = close_to_sol | delta_norm_is_zero
    finished = converged

    # Stop if any of the following conditions are met:
    # (A) We have hit the maximum number of iterations.
    # (B) The method is converging too slowly.
    # (C) The method is not expected to converge.
    too_slow = lipschitz_const > 1.
    finished = finished | too_slow

    too_many_iters = jnp.equal(num_iters, max_num_iters)
    num_iters_left = max_num_iters - num_iters
    wont_converge = (approx_dist_to_sol * lipschitz_const**num_iters_left >
                     tol)
    finished = finished | too_many_iters | wont_converge

    return _NewtonIterand(converged=converged,
                          finished=finished,
                          next_backward_difference=next_backward_difference,
                          next_state_vec=next_state_vec,
                          num_iters=num_iters,
                          prev_delta_norm=delta_norm)

  iterand = _NewtonIterand(
      converged=False,
      finished=False,
      next_backward_difference=jnp.zeros_like(initial_guess),
      next_state_vec=initial_guess,
      num_iters=0,
      prev_delta_norm=(jnp.array(-0.)))
  iterand = jax.lax.while_loop(
      lambda iterand: jnp.logical_not(iterand.finished), newton_body, iterand)
  return (iterand.converged, iterand.next_backward_difference,
          iterand.next_state_vec, iterand.num_iters)


def update_backward_differences(backward_differences, next_backward_difference,
                                next_state_vec, order):
  """Returns the backward differences for the next time."""
  new_backward_differences_array = jnp.zeros(
      (MAX_ORDER + 3, jnp.shape(next_backward_difference)[0]),
      dtype=next_backward_difference.dtype)
  new_backward_differences_array = jax.ops.index_update(
      new_backward_differences_array, jax.ops.index[order + 2],
      next_backward_difference - backward_differences[order + 1])
  new_backward_differences_array = jax.ops.index_update(
      new_backward_differences_array, jax.ops.index[order + 1],
      next_backward_difference)

  def body(vals):
    k, new_backward_differences_array_ = vals
    new_backward_differences_array_k = new_backward_differences_array_[
        k + 1] + backward_differences[k]
    new_backward_differences_array_ = jax.ops.index_update(
        new_backward_differences_array_, jax.ops.index[k],
        new_backward_differences_array_k)
    return (k - 1, new_backward_differences_array_)

  def body_cond(vals):
    k, _ = vals
    return k > 0

  _, new_backward_differences_array = jax.lax.while_loop(
      body_cond, body, (order, new_backward_differences_array))
  new_backward_differences_array = jax.ops.index_update(
      new_backward_differences_array, jax.ops.index[0], next_state_vec)

  new_backward_differences = jnp.stack(new_backward_differences_array)
  return new_backward_differences


def get_ode_fn_vec(ode_fn, initial_time, initial_state): #pylint: disable=unused-variable
  initial_state_vec = initial_state.flatten()  #noqa: F841 #pylint: disable=unused-variable 

  def ode_fn_vec(initial_time, initial_state_vec):
    return ode_fn(initial_time, initial_state_vec)

  return ode_fn_vec


def next_step_size(step_size, order, error_ratio, safety_factor,
                   min_step_size_factor, max_step_size_factor):
  """Computes the next step size to use.

  Computes the next step size by applying a multiplicative factor to the current
  step size. This factor is
  ```none
  factor_unclamped = error_ratio**(-1. / (order + 1)) * safety_factor
  factor = clamp(factor_unclamped, min_step_size_factor, max_step_size_factor)
  ```

  Args:
    step_size: Scalar float `Tensor` specifying the current step size.
    order: Scalar integer `Tensor` specifying the order of the method.
    error_ratio: Scalar float `Tensor` specifying the ratio of the error in the
      computed state and the tolerance.
    safety_factor: Scalar float `Tensor`.
    min_step_size_factor: Scalar float `Tensor` specifying a lower bound on the
      multiplicative factor.
    max_step_size_factor: Scalar float `Tensor` specifying an upper bound on the
      multiplicative factor.

  Returns:
    Scalar float `Tensor` specifying the next step size.
  """
  factor = error_ratio**(-1. / (order + 1.))
  return step_size * jnp.clip(safety_factor * factor, min_step_size_factor,
                              max_step_size_factor)
