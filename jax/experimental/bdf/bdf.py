import collections
from functools import partial

import jax
import jax.numpy as jnp

from . import bdf_util

#Adapted from tensorflow_probabilty
#https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/ode/bdf.py
#Accessed 2020-06-26


def _get_coefficients(bdf_coefficients):
  newton_coefficients = 1. / (
      (1. - bdf_coefficients) * bdf_util.RECIPROCAL_SUMS)

  error_coefficients = bdf_coefficients * bdf_util.RECIPROCAL_SUMS + 1. / (
      bdf_util.ORDERS + 1)

  return newton_coefficients, error_coefficients


def _get_common_params_and_coefficients(
    ode_fn, initial_time, initial_state, atol, rtol, min_step_size_factor,
    max_step_size_factor, max_order, max_num_newton_iters, max_num_steps,
    newton_tol_factor, newton_step_size_factor, safety_factor,
    bdf_coefficients):
  #convert everything to jnp arrays
  atol, rtol = jnp.array(atol, dtype=jnp.float64), jnp.array(rtol,
                                                             dtype=jnp.float64)
  min_step_size_factor, max_step_size_factor = jnp.array(min_step_size_factor, dtype=jnp.float64),\
                                               jnp.array(max_step_size_factor, dtype=jnp.float64)
  max_order, max_num_newton_iters = jnp.array(max_order, dtype=jnp.int64),\
                                    jnp.array(max_num_newton_iters, dtype=jnp.int64)
  max_num_steps = jnp.array(max_num_steps, dtype=jnp.float64)
  newton_tol_factor, newton_step_size_factor = jnp.array(newton_tol_factor, dtype=jnp.float64), \
                                               jnp.array(newton_step_size_factor, dtype=jnp.float64)
  safety_factor = jnp.array(safety_factor, dtype=jnp.float64)
  bdf_coefficients = jnp.array(bdf_coefficients, dtype=jnp.float64)
  initial_state_vec = initial_state.flatten()
  ode_fn_vec = bdf_util.get_ode_fn_vec(ode_fn, initial_time, initial_state)
  num_odes = jnp.shape(initial_state_vec)[0]

  newton_coefficients, error_cofficients = _get_coefficients(bdf_coefficients)

  _params = SolverParams(atol=atol,
                         rtol=rtol,
                         min_step_size_factor=min_step_size_factor,
                         max_step_size_factor=max_step_size_factor,
                         safety_factor=safety_factor,
                         max_num_steps=max_num_steps,
                         max_order=max_order,
                         max_num_newton_iters=max_num_newton_iters,
                         newton_tol_factor=newton_tol_factor,
                         newton_step_size_factor=newton_step_size_factor,
                         bdf_coefficients=bdf_coefficients,
                         initial_state_vec=initial_state_vec,
                         ode_fn_vec=ode_fn_vec,
                         initial_time=initial_time,
                         num_odes=num_odes)
  _coefficients = Coefficients(newton_coefficients=newton_coefficients,
                               error_coefficients=error_cofficients)

  return _params, _coefficients


def _initialize_solver_internal_state(
    ode_fn, initial_time, initial_state, atol, rtol, min_step_size_factor,
    max_step_size_factor, max_order, max_num_newton_iters, max_num_steps,
    newton_tol_factor, newton_step_size_factor, safety_factor,
    bdf_coefficients):

  p, e = _get_common_params_and_coefficients(
      ode_fn, initial_time, initial_state, atol, rtol, min_step_size_factor,
      max_step_size_factor, max_order, max_num_newton_iters, max_num_steps,
      newton_tol_factor, newton_step_size_factor, safety_factor,
      bdf_coefficients)

  first_step_size = bdf_util.first_step_size(
      atol=atol,
      first_order_error_coefficient=e.error_coefficients[1],
      initial_state_vec=p.initial_state_vec,
      initial_time=initial_time,
      ode_fn_vec=p.ode_fn_vec,
      rtol=rtol,
      safety_factor=safety_factor)

  first_order_backward_difference = p.ode_fn_vec(
      initial_time, p.initial_state_vec) * first_step_size

  backward_differences = jnp.concatenate([
      p.initial_state_vec[jnp.newaxis, :],
      first_order_backward_difference[jnp.newaxis, :],
      jnp.zeros(
          jnp.array(jnp.stack((bdf_util.MAX_ORDER + 1, p.num_odes)),
                    dtype=jnp.int64))
  ],
                                         axis=0)
  return _BDFSolverInternalState(backward_differences=backward_differences,
                                 order=1,
                                 step_size=first_step_size)


@partial(jax.jit, static_argnums=(0, 4))
def _solve(ode_fn, initial_time, initial_state, solution_times, jacobian_fn,
           atol, rtol, min_step_size_factor, max_step_size_factor, max_order,
           max_num_newton_iters, max_num_steps, newton_tol_factor,
           newton_step_size_factor, safety_factor, bdf_coefficients):
  def advance_to_solution_time(_states):
    """Takes multiple steps to advance time to `solution_times[n]`."""
    n, diagnostics, iterand, solver_internal_state, state_vec, times = _states

    def step_cond(_states):
      next_time, diagnostics, iterand, *_ = _states
      return (iterand.time < next_time) & (jnp.equal(diagnostics.status, 0))

    nth_solution_time = solution_times[n]

    [_, diagnostics, iterand, solver_internal_state, state_vec,
     times] = jax.lax.while_loop(step_cond, step, [
         nth_solution_time, diagnostics, iterand, solver_internal_state,
         state_vec, times
     ])

    state_vec = jax.ops.index_update(
        state_vec, jax.ops.index[n],
        solver_internal_state.backward_differences[0])
    times = jax.ops.index_update(times, jax.ops.index[n], nth_solution_time)

    return (n + 1, diagnostics, iterand, solver_internal_state, state_vec,
            times)

  def step(_states):
    """Takes a single step."""
    next_time, diagnostics, iterand, solver_internal_state, state_vec, times = _states
    distance_to_next_time = next_time - iterand.time
    overstepped = iterand.new_step_size > distance_to_next_time
    iterand = iterand._replace(
        new_step_size=jnp.where(overstepped, distance_to_next_time,
                                iterand.new_step_size),
        should_update_step_size=overstepped | iterand.should_update_step_size)
    #lazy jacobian evaluation ?

    diagnostics = diagnostics._replace(
        num_jacobian_evaluations=diagnostics.num_jacobian_evaluations + 1)
    iterand = iterand._replace(jacobian_mat=jacobian_fn(
        iterand.time, solver_internal_state.backward_differences[0]),
                               jacobian_is_up_to_date=True)

    def maybe_step_cond(_states):
      accepted, diagnostics, *_ = _states
      return jnp.logical_not(accepted) & jnp.equal(diagnostics.status, 0)

    _, diagnostics, iterand, solver_internal_state = jax.lax.while_loop(
        maybe_step_cond, maybe_step,
        (False, diagnostics, iterand, solver_internal_state))
    return [
        next_time, diagnostics, iterand, solver_internal_state, state_vec,
        times
    ]

  def maybe_step(_states):
    """Takes a single step only if the outcome has a low enough error."""
    accepted, diagnostics, iterand, solver_internal_state = _states
    [
        num_jacobian_evaluations, num_matrix_factorizations,
        num_ode_fn_evaluations, status
    ] = diagnostics
    [
        jacobian_mat, jacobian_is_up_to_date, new_step_size, num_steps,
        num_steps_same_size, should_update_jacobian, should_update_step_size,
        time, unitary, upper
    ] = iterand
    [backward_differences, order, step_size] = solver_internal_state
    status = jnp.where(jnp.equal(num_steps, max_num_steps), -1, 0)
    backward_differences = jnp.where(
        should_update_step_size,
        bdf_util.interpolate_backward_differences(backward_differences, order,
                                                  new_step_size / step_size),
        backward_differences)
    step_size = jnp.where(should_update_step_size, new_step_size, step_size)
    #should_update_factorization = should_update_step_size  #pylint: disable=unused-variable
    num_steps_same_size = jnp.where(should_update_step_size, 0,
                                    num_steps_same_size)

    def update_factorization():
      return bdf_util.newton_qr(jacobian_mat, e.newton_coefficients[order],
                                step_size)

    #lazy jacobian evaluation?
    unitary, upper = update_factorization()
    num_matrix_factorizations += 1

    tol = p.atol + p.rtol * jnp.abs(backward_differences[0])
    newton_tol = newton_tol_factor * jnp.linalg.norm(tol)

    [
        newton_converged, next_backward_difference, next_state_vec,
        newton_num_iters
    ] = bdf_util.newton(backward_differences, max_num_newton_iters,
                        e.newton_coefficients[order], p.ode_fn_vec, order,
                        step_size, time, newton_tol, unitary, upper)
    num_steps += 1
    num_ode_fn_evaluations += newton_num_iters

    # If Newton's method failed and the Jacobian was up to date, decrease the
    # step size.
    newton_failed = jnp.logical_not(newton_converged)
    should_update_step_size = newton_failed & jacobian_is_up_to_date
    new_step_size = step_size * jnp.where(should_update_step_size,
                                          newton_step_size_factor, 1.)

    # If Newton's method failed and the Jacobian was NOT up to date, update
    # the Jacobian.
    should_update_jacobian = newton_failed & jnp.logical_not(
        jacobian_is_up_to_date)

    error_ratio = jnp.where(
        newton_converged,
        bdf_util.error_ratio(next_backward_difference,
                             e.error_coefficients[order], tol), jnp.nan)
    accepted = error_ratio < 1.
    converged_and_rejected = newton_converged & jnp.logical_not(accepted)

    # If Newton's method converged but the solution was NOT accepted, decrease
    # the step size.
    new_step_size = jnp.where(
        converged_and_rejected,
        bdf_util.next_step_size(step_size, order, error_ratio, p.safety_factor,
                                p.min_step_size_factor,
                                p.max_step_size_factor), new_step_size)
    should_update_step_size = should_update_step_size | converged_and_rejected

    # If Newton's method converged and the solution was accepted, update the
    # matrix of backward differences.
    time = jnp.where(accepted, time + step_size, time)
    backward_differences = jnp.where(
        accepted,
        bdf_util.update_backward_differences(backward_differences,
                                             next_backward_difference,
                                             next_state_vec, order),
        backward_differences)
    jacobian_is_up_to_date = jacobian_is_up_to_date & jnp.logical_not(accepted)
    num_steps_same_size = jnp.where(accepted, num_steps_same_size + 1,
                                    num_steps_same_size)

    # Order and step size are only updated if we have taken strictly more than
    # order + 1 steps of the same size. This is to prevent the order from
    # being throttled.
    should_update_order_and_step_size = accepted & (num_steps_same_size >
                                                    order + 1)
    new_order = order
    new_error_ratio = error_ratio
    for offset in [-1, +1]:
      proposed_order = jnp.clip(order + offset, 1, max_order)
      proposed_error_ratio = bdf_util.error_ratio(
          backward_differences[proposed_order + 1],
          e.error_coefficients[proposed_order], tol)
      proposed_error_ratio_is_lower = proposed_error_ratio < new_error_ratio
      new_order = jnp.where(
          should_update_order_and_step_size & proposed_error_ratio_is_lower,
          proposed_order, new_order)
      new_error_ratio = jnp.where(
          should_update_order_and_step_size & proposed_error_ratio_is_lower,
          proposed_error_ratio, new_error_ratio)
    order = new_order
    error_ratio = new_error_ratio

    new_step_size = jnp.where(
        should_update_order_and_step_size,
        bdf_util.next_step_size(step_size, order, error_ratio, p.safety_factor,
                                p.min_step_size_factor,
                                p.max_step_size_factor), new_step_size)
    should_update_step_size = (should_update_step_size
                               | should_update_order_and_step_size)

    diagnostics = _BDFDiagnostics(num_jacobian_evaluations,
                                  num_matrix_factorizations,
                                  num_ode_fn_evaluations, status)

    iterand = _BDFIterand(jacobian_mat, jacobian_is_up_to_date, new_step_size,
                          num_steps, num_steps_same_size,
                          should_update_jacobian, should_update_step_size,
                          time, unitary, upper)

    solver_internal_state = _BDFSolverInternalState(backward_differences,
                                                    order, step_size)
    return accepted, diagnostics, iterand, solver_internal_state

  solver_internal_state = _initialize_solver_internal_state(
      ode_fn, initial_time, initial_state, atol, rtol, min_step_size_factor,
      max_step_size_factor, max_order, max_num_newton_iters, max_num_steps,
      newton_tol_factor, newton_step_size_factor, safety_factor,
      bdf_coefficients)

  p, e = _get_common_params_and_coefficients(
      ode_fn, initial_time, initial_state, atol, rtol, min_step_size_factor,
      max_step_size_factor, max_order, max_num_newton_iters, max_num_steps,
      newton_tol_factor, newton_step_size_factor, safety_factor,
      bdf_coefficients)

  diagnostics = _BDFDiagnostics(num_jacobian_evaluations=0,
                                num_matrix_factorizations=0,
                                num_ode_fn_evaluations=0,
                                status=0)

  iterand = _BDFIterand(
      jacobian_mat=jnp.zeros([p.num_odes, p.num_odes]),
      jacobian_is_up_to_date=False,
      new_step_size=solver_internal_state.step_size,
      num_steps=0,
      num_steps_same_size=0,
      should_update_jacobian=True,
      should_update_step_size=False,
      time=p.initial_time,
      unitary=jnp.zeros([p.num_odes, p.num_odes]),
      upper=jnp.zeros([p.num_odes, p.num_odes]),
  )

  num_solution_times = jnp.shape(solution_times)[0]
  state_vec_size = jnp.shape(initial_state)[0]
  state_vec = jnp.zeros([num_solution_times, state_vec_size],
                        dtype=jnp.float64)
  times = jnp.zeros([num_solution_times], dtype=jnp.float64)

  def advance_to_solution_time_cond(_states):
    n, diagnostics, *_ = _states
    return (n < num_solution_times) & (jnp.equal(diagnostics.status, 0))

  [_, diagnostics, iterand,
   solver_internal_state, state_vec, times] = jax.lax.while_loop(
       advance_to_solution_time_cond, advance_to_solution_time,
       (0, diagnostics, iterand, solver_internal_state, state_vec, times))
  return Results(
      times=times,
      states=state_vec,
      diagnostics=diagnostics,
      solver_internal_state=solver_internal_state,
  )


@partial(jax.jit, static_argnums=(0, ))
def get_jac(ode_fn, t, state_vec):
  jac_rev = jax.jacrev(
      ode_fn, (1, ))  # rev mode jacobian w.r.t to 2nd arguement(state_vec)
  jac = jac_rev(t, state_vec)
  return jac[0]


def bdf_solve(ode_fn,
              initial_time,
              initial_state,
              solution_times,
              jacobian_fn,
              atol=1e-6,
              rtol=1e-3,
              min_step_size_factor=0.1,
              max_step_size_factor=10.,
              max_order=bdf_util.MAX_ORDER,
              max_num_newton_iters=4,
              max_num_steps=jnp.inf,
              newton_tol_factor=0.1,
              newton_step_size_factor=0.5,
              safety_factor=0.9,
              bdf_coefficients=[0., 0.1850, -1. / 9., -0.0823, -0.0415, 0.]):

  if jacobian_fn is None:
    jacobian_fn = partial(get_jac, ode_fn)

  results = _solve(ode_fn, initial_time, initial_state, solution_times,
                   jacobian_fn, atol, rtol, min_step_size_factor,
                   max_step_size_factor, max_order, max_num_newton_iters,
                   max_num_steps, newton_tol_factor, newton_step_size_factor,
                   safety_factor, bdf_coefficients)

  return results


class Results(
    collections.namedtuple(
        "Results",
        ["times", "states", "diagnostics", "solver_internal_state"])):
  """
    namedtuple class to store results from ode solver
    """
  def __new__(cls, times, states, diagnostics, solver_internal_state):
    return super(Results, cls).__new__(cls, times, states, diagnostics,
                                       solver_internal_state)


bdf_util.register_pytree_namedtuple(Results)  #JAX pytree


class _BDFDiagnostics(
    collections.namedtuple('_BDFDiagnostics', [
        'num_jacobian_evaluations',
        'num_matrix_factorizations',
        'num_ode_fn_evaluations',
        'status',
    ])):
  """
    namedtuple class to store diagnostics
    """
  def __new__(cls, num_jacobian_evaluations, num_matrix_factorizations,
              num_ode_fn_evaluations, status):
    return super(_BDFDiagnostics, cls).__new__(cls, num_jacobian_evaluations,
                                               num_matrix_factorizations,
                                               num_ode_fn_evaluations, status)


bdf_util.register_pytree_namedtuple(_BDFDiagnostics)  #JAX pytree


class _BDFIterand(
    collections.namedtuple('_BDFIterand', [
        'jacobian_mat', 'jacobian_is_up_to_date', 'new_step_size', 'num_steps',
        'num_steps_same_size', 'should_update_jacobian',
        'should_update_step_size', 'time', 'unitary', 'upper'
    ])):
  """
    namedtuple class to store iterand state
    """
  def __new__(cls, jacobian_mat, jacobian_is_up_to_date, new_step_size,
              num_steps, num_steps_same_size, should_update_jacobian,
              should_update_step_size, time, unitary, upper):
    return super(_BDFIterand,
                 cls).__new__(cls, jacobian_mat, jacobian_is_up_to_date,
                              new_step_size, num_steps, num_steps_same_size,
                              should_update_jacobian, should_update_step_size,
                              time, unitary, upper)


bdf_util.register_pytree_namedtuple(_BDFIterand)  #JAX pytree


class _BDFSolverInternalState(
    collections.namedtuple('_BDFSolverInternalState', [
        'backward_differences',
        'order',
        'step_size',
    ])):
  """
    Returned by the solver to warm start future invocations
    """
  def __new__(cls, backward_differences, order, step_size):
    return super(_BDFSolverInternalState,
                 cls).__new__(cls, backward_differences, order, step_size)


bdf_util.register_pytree_namedtuple(_BDFSolverInternalState)  #JAX pytree


class SolverParams(
    collections.namedtuple('SolverParams', [
        "rtol", "atol", "safety_factor", "min_step_size_factor",
        "max_step_size_factor", "max_num_steps", "max_order",
        "max_num_newton_iters", "newton_tol_factor", "newton_step_size_factor",
        "bdf_coefficients", "initial_state_vec", "ode_fn_vec", "initial_time",
        "num_odes"
    ])):
  def __new__(cls, rtol, atol, safety_factor, min_step_size_factor,
              max_step_size_factor, max_num_steps, max_order,
              max_num_newton_iters, newton_tol_factor, newton_step_size_factor,
              bdf_coefficients, initial_state_vec, ode_fn_vec, initial_time,
              num_odes):
    return super(SolverParams,
                 cls).__new__(cls, rtol, atol, safety_factor,
                              min_step_size_factor, max_step_size_factor,
                              max_num_steps, max_order, max_num_newton_iters,
                              newton_tol_factor, newton_step_size_factor,
                              bdf_coefficients, initial_state_vec, ode_fn_vec,
                              initial_time, num_odes)


bdf_util.register_pytree_namedtuple(SolverParams)


class Coefficients(
    collections.namedtuple('Coefficients',
                           ["newton_coefficients", "error_coefficients"])):
  def __new__(cls, newton_coefficients, error_coefficients):
    return super(Coefficients, cls).__new__(cls, newton_coefficients,
                                            error_coefficients)


bdf_util.register_pytree_namedtuple(Coefficients)
