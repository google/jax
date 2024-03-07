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

"""An example of using host_callback.call to invoke on the host functions
written in Tensorflow. The interesting aspect here is how we can differentiate
through the outside computation, using tf.GradientTape on the host.

This is separate from host_callback_test because it needs a TF dependency.
"""
from collections.abc import Callable
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax.experimental import host_callback as hcb

import numpy as np

try:
  import tensorflow as tf
except ImportError:
  tf = None

config.parse_flags_with_absl()


def call_tf_no_ad(tf_fun: Callable, arg, *, result_shape):
  """The simplest implementation of calling to TF, without AD support.

  We must use hcb.call because the TF invocation must happen outside the
  JAX staged computation."""

  def tf_to_numpy(t):
    # Turn the Tensor to NumPy array without copying.
    return np.asarray(memoryview(t)) if isinstance(t, tf.Tensor) else t

  return hcb.call(lambda arg: tf.nest.map_structure(tf_to_numpy,
                                                    tf_fun(arg)),
                  arg, result_shape=result_shape,
                  callback_flavor=hcb.CallbackFlavor.DEBUG)


def call_tf_simple_ad(tf_fun: Callable, arg, *, result_shape):
  """Calls a TensorFlow function with simple support for reverse AD.

  Works only for 1st order AD and only for arguments and results being a single
  ndarray (no pytrees). Functions whose name starts with "tf_" are TensorFlow
  functions and must be called outside the JAX computation.
  """

  @jax.custom_vjp
  def make_call(arg):
    """We wrap it all in `make_call` so that we can attach custom VJP."""
    return call_tf_no_ad(tf_fun, arg, result_shape=result_shape)

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(arg):
    # Return the primal argument as the residual. Use `make_call` for the
    # primal computation to enable higher-order AD.
    return make_call(arg), arg

  def make_call_vjp_bwd(res, ct_res):
    arg = res  # residual is the primal argument

    def tf_vjp_fun(arg_and_ct_res):
      """Invoke TF gradient; used with hcb.call."""
      arg, ct_res = arg_and_ct_res
      arg_var = tf.Variable(arg)
      with tf.GradientTape(persistent=True) as tape:
        res = tf_fun(arg_var)

      dres_darg = tape.gradient(res, sources=arg_var,
                                output_gradients=ct_res,
                                unconnected_gradients=tf.UnconnectedGradients.ZERO)
      return dres_darg

    return (call_tf_simple_ad(tf_vjp_fun, (arg, ct_res),
                              result_shape=arg),)

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return make_call(arg)


def call_tf_full_ad(tf_fun: Callable, arg, *, result_shape):
  """Calls a TensorFlow function with support for reverse AD.

  Supports higher-order AD and pytree arguments.
  """

  @jax.custom_vjp
  def make_call(arg):
    """We wrap it all in `make_call` so that we can attach custom VJP."""
    return call_tf_no_ad(tf_fun, arg, result_shape=result_shape)

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(arg):
    return make_call(arg), arg  # Return the primal argument as the residual

  def make_call_vjp_bwd(res, ct_res):
    arg = res  # residual is the primal argument

    def tf_vjp_fun(arg_and_ct_res):
      """Invoke TF gradient; used with hcb.call."""
      arg, ct_res = arg_and_ct_res

      def make_var(a):
        return a if isinstance(a, tf.Variable) else tf.Variable(a)

      arg_var = tf.nest.map_structure(make_var, arg)

      with tf.GradientTape(persistent=True) as tape:
        res = tf_fun(arg_var)

      tf.nest.assert_same_structure(res, ct_res)
      accumulator = None  # Accumulate argument cotangent. Same structure as "arg"

      def acc_ct(res_, ct_res_):
        dres_darg = tape.gradient(res_, sources=arg_var,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        tf.nest.assert_same_structure(dres_darg, arg)
        scaled_dres_darg = tf.nest.map_structure(lambda d: d * ct_res_, dres_darg)
        nonlocal accumulator
        accumulator = (scaled_dres_darg if accumulator is None
                       else tf.nest.map_structure(lambda x, y: x + y,
                                                  accumulator, scaled_dres_darg))

      tf.nest.map_structure(acc_ct, res, ct_res)
      return accumulator

    return (call_tf_full_ad(tf_vjp_fun, (arg, ct_res),
                            result_shape=arg),)

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return make_call(arg)


CALL_TF_IMPLEMENTATIONS = {
    "none": call_tf_no_ad,
    "simple": call_tf_simple_ad,
    "full": call_tf_full_ad,
}


class CallToTFTest(jtu.JaxTestCase):

  def setUp(self):
    if tf is None:
      raise unittest.SkipTest("Test requires tensorflow")
    if xla_bridge.using_pjrt_c_api():
      raise unittest.SkipTest("host_callback not implemented in PJRT C API")
    super().setUp()

  def supported_only_in_legacy_mode(self):
    if not hcb._HOST_CALLBACK_LEGACY.value:
      self.skipTest("Not supported when JAX_HOST_CALLBACK_LEGACY=False")

  @parameterized.named_parameters(
      dict(
          testcase_name=f"_{ad=}",
          ad=ad)
      for ad in CALL_TF_IMPLEMENTATIONS.keys())
  def test_impl(self, ad="simple"):
    self.supported_only_in_legacy_mode()
    call_tf = CALL_TF_IMPLEMENTATIONS[ad]

    def f_jax(x):
      return jnp.sin(x)

    def f_outside(x):
      return call_tf(tf.math.sin, x,
                     result_shape=x)

    res = f_outside(3.)
    self.assertAllClose(f_jax(3.), res)
    self.assertAllClose(f_jax(3.), jax.jit(f_outside)(3.))

  @parameterized.named_parameters(
      dict(
          testcase_name=f"_{ad=}",
          ad=ad)
      for ad in CALL_TF_IMPLEMENTATIONS.keys()
      if ad != "none")
  def test_grad(self, ad="simple"):
    self.supported_only_in_legacy_mode()
    call_tf = CALL_TF_IMPLEMENTATIONS[ad]

    def f_jax(x):
      return 3. * jnp.sin(2. * x)

    def f_outside(x):
      return 3. * call_tf(
          lambda x: tf.cast(tf.math.sin(x), tf.float32), 2. * x,
          result_shape=jax.ShapeDtypeStruct((), np.float32))

    x = np.float32(4.)
    self.assertAllClose(f_jax(x), f_outside(x),
                        check_dtypes=False)

    grad_f = jax.grad(f_outside)(x)
    self.assertAllClose(jax.grad(f_jax)(x), grad_f,
                        check_dtypes=False)

  def test_grad_pytree(self):
    self.supported_only_in_legacy_mode()
    call_tf = call_tf_full_ad

    def f_jax(xy):
      dict_ab = dict(a=2. * xy[0], b=xy[0] * xy[1])
      return 3. * dict_ab["a"] + 4. * dict_ab["b"]

    def f_outside(xy):
      dict_ab = call_tf(
          lambda xy: dict(a=tf.cast(2. * xy[0], np.float32),
                          b=tf.cast(xy[0] * xy[1], np.float32)),
          xy,
          result_shape=dict(a=jax.ShapeDtypeStruct((), np.float32),
                            b=jax.ShapeDtypeStruct((), np.float32)))
      return 3. * dict_ab["a"] + 4. * dict_ab["b"]

    xy = (5., 6.)
    self.assertAllClose(f_jax(xy), f_outside(xy),
                        check_dtypes=False)
    res_jax = jax.grad(f_jax)(xy)
    self.assertAllClose(res_jax, jax.grad(f_outside)(xy),
                        check_dtypes=False)

  @parameterized.named_parameters(
      dict(
          testcase_name=f"_degree=_{degree}",
          degree=degree)
      for degree in [1, 2, 3, 4])
  def test_higher_order_grad(self, degree=4):
    self.supported_only_in_legacy_mode()
    call_tf = call_tf_full_ad

    def f_jax(x):
      return 2. * x * x * x

    def f_outside(x):
      return 2. * call_tf(lambda y: y * y * y, x,
                          result_shape=x)

    grad_jax = f_jax
    grad_outside = f_outside
    for i in range(degree):
      grad_jax = jax.grad(grad_jax)
      grad_outside = jax.grad(grad_outside)

    res_jax = grad_jax(5.)
    self.assertAllClose(res_jax, grad_outside(5.))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
