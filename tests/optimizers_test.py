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

"""Tests for the optimizers module."""

import functools

from absl.testing import absltest
import numpy as np

import jax
import jax.numpy as jnp
import jax._src.test_util as jtu
from jax import jit, grad, jacfwd, jacrev
from jax import lax
from jax.example_libraries import optimizers

jax.config.parse_flags_with_absl()


class OptimizerTests(jtu.JaxTestCase):

  def _CheckOptimizer(self, optimizer, loss, x0, num_steps, *args, **kwargs):
    self._CheckFuns(optimizer, loss, x0, *args)
    self._CheckRun(optimizer, loss, x0, num_steps, *args, **kwargs)

  def _CheckFuns(self, optimizer, loss, x0, *args):
    init_fun, update_fun, get_params = optimizer(*args)
    opt_state = init_fun(x0)
    self.assertAllClose(x0, get_params(opt_state))
    opt_state2 = update_fun(0, grad(loss)(x0), opt_state)  # doesn't crash
    self.assertEqual(jax.tree.structure(opt_state),
                     jax.tree.structure(opt_state2))

  @jtu.skip_on_devices('gpu')
  def _CheckRun(self, optimizer, loss, x0, num_steps, *args, **kwargs):
    init_fun, update_fun, get_params = optimizer(*args)

    opt_state = init_fun(x0)
    for i in range(num_steps):
      x = get_params(opt_state)
      g = grad(loss)(x)
      opt_state = update_fun(i, g, opt_state)
    xstar = get_params(opt_state)
    self.assertLess(loss(xstar), 1e-2)

    update_fun_jitted = jit(update_fun)
    opt_state = init_fun(x0)
    for i in range(num_steps):
      x = get_params(opt_state)
      g = grad(loss)(x)
      opt_state = update_fun_jitted(i, g, opt_state)
    xstar = get_params(opt_state)
    self.assertLess(loss(xstar), 1e-2)

  def testSgdScalar(self):
    def loss(x): return x**2
    x0 = 1.
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testSgdVector(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testSgdNestedTuple(self):
    def loss(xyz):
      x, (y, z) = xyz
      return sum(jnp.dot(a, a) for a in [x, y, z])
    x0 = (jnp.ones(2), (jnp.ones(2), jnp.ones(2)))
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sgd, loss, x0, num_iters, step_size)

  def testMomentumVector(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    num_iters = 100
    step_size = 0.1
    mass = 0.
    self._CheckOptimizer(optimizers.momentum, loss, x0, num_iters, step_size, mass)

  def testMomentumDict(self):
    def loss(dct): return jnp.dot(dct['x'], dct['x'])
    x0 = {'x': jnp.ones(2)}
    num_iters = 100
    step_size = 0.1
    mass = 0.
    self._CheckOptimizer(optimizers.momentum, loss, x0, num_iters, step_size, mass)

  def testRmspropVector(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.rmsprop, loss, x0, num_iters, step_size)

  def testAdamVector(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.adam, loss, x0, num_iters, step_size)

  def testSgdClosure(self):
    def loss(y, x): return y**2 * x**2
    x0 = 1.
    y = 1.
    num_iters = 20
    step_size = 0.1
    partial_loss = functools.partial(loss, y)
    self._CheckRun(optimizers.sgd, partial_loss, x0, num_iters, step_size)

  def testAdagrad(self):

    def loss(xs):
      x1, x2 = xs
      return jnp.sum(x1**2) + jnp.sum(x2**2)

    num_iters = 100
    step_size = 0.1
    x0 = (jnp.ones(2), jnp.ones((2, 2)))
    self._CheckOptimizer(optimizers.adagrad, loss, x0, num_iters, step_size)

  def testSM3Scalar(self):
    def loss(x): return x**2
    x0 = jnp.array(1.)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.sm3, loss, x0, num_iters, step_size)

  def testSM3Vector(self):
    def loss(xs):
      x1, x2 = xs
      return jnp.sum(x1 ** 2) + jnp.sum(x2 ** 2)

    num_iters = 100
    step_size = 0.1
    x0 = (jnp.ones(2), jnp.ones((2, 2)))
    self._CheckOptimizer(optimizers.sm3, loss, x0, num_iters, step_size)

  def testAdaMaxVector(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    num_iters = 100
    step_size = 0.1
    self._CheckOptimizer(optimizers.adamax, loss, x0, num_iters, step_size)

  def testSgdVectorExponentialDecaySchedule(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    step_schedule = optimizers.exponential_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.sgd, loss, x0, step_schedule)

  def testSgdVectorInverseTimeDecaySchedule(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    step_schedule = optimizers.inverse_time_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.sgd, loss, x0, step_schedule)

  def testAdamVectorInverseTimeDecaySchedule(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    step_schedule = optimizers.inverse_time_decay(0.1, 3, 2.)
    self._CheckFuns(optimizers.adam, loss, x0, step_schedule)

  def testMomentumVectorInverseTimeDecayStaircaseSchedule(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    step_sched = optimizers.inverse_time_decay(0.1, 3, 2., staircase=True)
    mass = 0.9
    self._CheckFuns(optimizers.momentum, loss, x0, step_sched, mass)

  def testRmspropmomentumVectorPolynomialDecaySchedule(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    step_schedule = optimizers.polynomial_decay(1.0, 50, 0.1)
    self._CheckFuns(optimizers.rmsprop_momentum, loss, x0, step_schedule)

  def testRmspropVectorPiecewiseConstantSchedule(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    step_schedule = optimizers.piecewise_constant([25, 75], [1.0, 0.5, 0.1])
    self._CheckFuns(optimizers.rmsprop, loss, x0, step_schedule)

  def testTracedStepSize(self):
    def loss(x): return jnp.dot(x, x)
    x0 = jnp.ones(2)
    step_size = 0.1

    init_fun, _, _ = optimizers.sgd(step_size)
    opt_state = init_fun(x0)

    @jit
    def update(opt_state, step_size):
      _, update_fun, get_params = optimizers.sgd(step_size)
      x = get_params(opt_state)
      g = grad(loss)(x)
      return update_fun(0, g, opt_state)

    update(opt_state, 0.9)  # doesn't crash

  # TODO(mattjj): re-enable
  # def testDeviceTupleState(self):
  #   init_fun, update_fun, _ = optimizers.sgd(0.1)
  #   opt_state = init_fun(jnp.zeros(3))
  #   self.assertIsInstance(opt_state, optimizers.OptimizerState)
  #   self.assertIsInstance(opt_state.packed_state, core.JaxTuple)
  #   opt_state = jit(update_fun)(0, jnp.zeros(3), opt_state)
  #   self.assertIsInstance(opt_state, optimizers.OptimizerState)
  #   self.assertIsInstance(opt_state.packed_state, xla.DeviceTuple)

  def testUpdateFunStructureMismatchErrorMessage(self):
    @optimizers.optimizer
    def opt_maker():
      def init_fun(x0):
        return {'x': x0}
      def update_fun(i, g, opt_state):
        x = opt_state['x']
        return {'x': x - 0.1 * g, 'v': g}  # bug!
      def get_params(opt_state):
        return opt_state['x']
      return init_fun, update_fun, get_params

    init_fun, update_fun, get_params = opt_maker()
    opt_state = init_fun(jnp.zeros(3))
    self.assertRaises(TypeError, lambda: update_fun(opt_state))

  def testUtilityNorm(self):
    x0 = (jnp.ones(2), (jnp.ones(3), jnp.ones(4)))
    norm = optimizers.l2_norm(x0)
    expected = np.sqrt(np.sum(np.ones(2+3+4)**2))
    self.assertAllClose(norm, expected, check_dtypes=False)

  def testUtilityClipGrads(self):
    g = (jnp.ones(2), (jnp.ones(3), jnp.ones(4)))
    norm = optimizers.l2_norm(g)

    ans = optimizers.clip_grads(g, 1.1 * norm)
    expected = g
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = optimizers.l2_norm(optimizers.clip_grads(g, 0.9 * norm))
    expected = 0.9 * norm
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testIssue758(self):
    # code from https://github.com/google/jax/issues/758
    # this is more of a scan + jacfwd/jacrev test, but it lives here to use the
    # optimizers.py code

    def harmonic_bond(conf, params):
      return jnp.sum(conf * params)

    opt_init, opt_update, get_params = optimizers.sgd(5e-2)

    x0 = np.array([0.5], dtype=np.float64)

    def minimize_structure(test_params):
      energy_fn = functools.partial(harmonic_bond, params=test_params)
      grad_fn = grad(energy_fn, argnums=(0,))
      opt_state = opt_init(x0)

      def apply_carry(carry, _):
        i, x = carry
        g = grad_fn(get_params(x))[0]
        new_state = opt_update(i, g, x)
        new_carry = (i+1, new_state)
        return new_carry, _

      carry_final, _ = lax.scan(apply_carry, (0, opt_state), jnp.zeros((75, 0)))
      trip, opt_final = carry_final
      assert trip == 75
      return opt_final

    initial_params = jnp.array(0.5)
    minimize_structure(initial_params)

    def loss(test_params):
      opt_final = minimize_structure(test_params)
      return 1.0 - get_params(opt_final)[0]

    loss_opt_init, loss_opt_update, loss_get_params = optimizers.sgd(5e-2)

    J1 = jacrev(loss, argnums=(0,))(initial_params)
    J2 = jacfwd(loss, argnums=(0,))(initial_params)
    self.assertAllClose(J1, J2, rtol=1e-6)

  def testUnpackPackRoundTrip(self):
    opt_init, _, _ = optimizers.momentum(0.1, mass=0.9)
    params = [{'w': self.rng().randn(1, 2), 'bias': self.rng().randn(2)}]
    expected = opt_init(params)
    ans = optimizers.pack_optimizer_state(
        optimizers.unpack_optimizer_state(expected))
    self.assertEqual(ans, expected)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
