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
"""Tests for JAX2TF converted.

Specific JAX primitive conversion tests are in primitives_test."""
import collections
import os
from typing import Callable, Dict, Optional, Tuple
import unittest

from absl import logging
from absl.testing import absltest

import jax
from jax import ad_checkpoint
from jax import dtypes
from jax import lax
from jax import numpy as jnp
from jax._src import source_info_util
from jax._src import test_util as jtu
import jax._src.lib.xla_bridge
from jax.config import config
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
from jax.experimental import pjit
from jax.interpreters import mlir
from jax.sharding import PartitionSpec as P

import numpy as np
import tensorflow as tf  # type: ignore[import]
# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]
# pylint: enable=g-direct-tensorflow-import

config.parse_flags_with_absl()


class Jax2TfTest(tf_test_util.JaxToTfTestCase):

  def test_empty(self):
    f_jax = lambda x, y: x
    self.ConvertAndCompare(f_jax, 0.7, 1)

  def test_basics(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    self.ConvertAndCompare(f_jax, 0.7)

  def test_input_output_naming(self):
    @jax2tf.convert
    def f(xs, y):
      return [jnp.add(x, y) for x in xs]

    @tf.function(autograph=False)
    def u(xs, y):
      xs = tf.nest.map_structure(tf.convert_to_tensor, xs)
      with tf.GradientTape() as tape:
        tf.nest.map_structure(tape.watch, xs)
        y = f(xs, y)
        tape.gradient(y, xs)
        return y

    cf = u.get_concrete_function([1., 2., 3.], 4.)
    g = cf.graph
    g.get_operation_by_name("jax2tf_arg_0")
    g.get_operation_by_name("jax2tf_arg_1")
    g.get_operation_by_name("jax2tf_arg_2")
    g.get_operation_by_name("jax2tf_arg_3")
    g.get_operation_by_name("jax2tf_out")
    g.get_operation_by_name("jax2tf_out_1")
    g.get_operation_by_name("jax2tf_out_2")
    with self.assertRaises(KeyError):
      g.get_operation_by_name("jax2tf_arg_4")
    with self.assertRaises(KeyError):
      g.get_operation_by_name("jax2tf_out_3")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_0")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_1")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_2")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_arg_3")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_1")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_2")
    g.get_operation_by_name("jax2tf_vjp/jax2tf_out_3")

  def test_pytrees(self):
    # Take and return pytrees
    def f_jax(x: Tuple[float, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
      x_a, x_dict = x
      return x_a * 2., {k: v * 3. for k, v in x_dict.items()}

    x = (.7, {"a": .8, "b": .9})
    self.ConvertAndCompare(f_jax, x)

  def test_variable_input(self):
    f_jax = lambda x: jnp.sin(jnp.cos(x))
    f_tf = jax2tf.convert(f_jax)
    v = tf.Variable(0.7, dtype=jax2tf.dtype_of_val(0.7))
    self.assertIsInstance(f_tf(v), tf.Tensor)
    self.assertAllClose(f_jax(0.7), f_tf(v))

  def test_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, 0.7)

  def test_nested_jit(self):
    f_jax = jax.jit(lambda x: jnp.sin(jax.jit(jnp.cos)(x)))
    x = 0.7
    self.ConvertAndCompare(f_jax, x)

  def test_nested_jit_pytree(self):
    @jax.jit
    def f_jax(xy):
      x, y = xy
      return x + y
    xy = (0.7, 0.8)
    self.ConvertAndCompare(f_jax, xy)

  def test_nested_jit_is_compiled(self):
    # Check that nested jax.jit are compiled with tf.function(jit_compile=True)
    # We do this by looking for the _XlaMustCompile attribute in the function graph
    def has_xla_must_compile(f_tf, x):
      f_conc = tf.function(f_tf, autograph=True).get_concrete_function(tf.convert_to_tensor(x))
      for n in f_conc.graph._nodes_by_id.values():
        try:
          n.get_attr("_XlaMustCompile")
          return True
        except ValueError:
          continue
      return False

    x = np.array(0.7)
    f_no_jit = lambda x: x
    self.assertFalse(has_xla_must_compile(jax2tf.convert(f_no_jit), x))
    f_jit = lambda x: jax.jit(jnp.sin)(x)
    # TODO(b/207464757): TF compilation is disabled
    self.assertFalse(has_xla_must_compile(jax2tf.convert(f_jit), x))

  def test_converts_jax_arrays(self):
    f_tf = tf.function(lambda x: x)
    self.assertEqual(f_tf(jnp.zeros([])).numpy(), 0.)
    self.assertEqual(f_tf(jnp.ones([])).numpy(), 1.)
    f_tf = tf.function(lambda x: x + x)
    self.assertEqual(f_tf(jnp.ones([])).numpy(), 2.)

    # Test with ShardedDeviceArray.
    n = jax.local_device_count()
    mk_sharded = lambda f: jax.pmap(lambda x: x)(f([n]))
    f_tf = tf.function(lambda x: x)
    self.assertAllClose(f_tf(mk_sharded(jnp.zeros)).numpy(),
                        jnp.zeros([n]))
    self.assertAllClose(f_tf(mk_sharded(jnp.ones)).numpy(),
                        jnp.ones([n]))

  @jtu.skip_on_devices("gpu")
  def test_bfloat16_passed_by_tf(self):
    f_jax = lambda a, b: a + b
    f_tf = tf.function(jax2tf.convert(f_jax), autograph=False,
                       input_signature=[tf.TensorSpec([512, 512], tf.bfloat16),
                                        tf.TensorSpec([512, 512], tf.bfloat16)])
    self.assertIsNotNone(f_tf.get_concrete_function())

  @jtu.skip_on_devices("gpu")
  def test_bfloat16_returned_by_jax(self):
    f_jax = lambda a, b: (a + b).astype(jnp.bfloat16)
    f_tf = jax2tf.convert(f_jax)
    self.assertEqual(f_tf(1., 2.).dtype, tf.bfloat16)

  @jtu.skip_on_devices("gpu")
  def test_bfloat16_tf_grad(self):
    f_jax = lambda a, b: a + b

    def _tf_grad(a, b):
      with tf.GradientTape() as tape:
        tape.watch(a)
        result = jax2tf.convert(f_jax)(a, b)
      return result, tape.gradient(result, a)

    f_tf = tf.function(_tf_grad, autograph=False,
                       input_signature=[tf.TensorSpec([512, 512], tf.bfloat16),
                                        tf.TensorSpec([512, 512], tf.bfloat16)])

    self.assertIsNotNone(f_tf.get_concrete_function())

  @jtu.sample_product(
    dtype=[np.int64, np.float64],
    with_function=[True, False],
  )
  def test_converts_64bit(self, dtype=np.int64, with_function=False):
    if not config.jax_enable_x64:
      self.skipTest("requires x64 mode")
    big_const = np.full((5,), 2 ** 33, dtype=dtype)
    self.ConvertAndCompare(jnp.sin, big_const)
    f_conv = jax2tf.convert(jnp.sin)
    if with_function:
      f_conv = tf.function(f_conv, autograph=False)
    # We check also when we pass tf.Variable or tf.Tensor into the
    # converted function
    self.assertAllClose(jnp.sin(big_const),
                        f_conv(tf.Variable(big_const)))
    self.assertAllClose(jnp.sin(big_const),
                        f_conv(tf.constant(big_const)))

  def test_64bit_behavior_enable_x64_readme(self):
    # Tests some of the examples from the README
    if not config.jax_enable_x64:
      self.skipTest("requires x64 mode")

    # JAX and TF have different default float types if JAX_ENABLE_X64=1
    self.assertEqual(tf.math.sin(3.14).dtype, tf.float32)
    self.assertEqual(jnp.sin(3.14).dtype, jnp.float64)

    # jax2tf.convert has the same behavior as JAX
    self.assertEqual(jax2tf.convert(jnp.sin)(3.14).dtype, tf.float64)
    # The following will compute `sin` in float64.
    self.assertEqual(tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14, dtype=tf.float64)).dtype, tf.float64)

    # The following will compute `sin` in float32.
    self.assertEqual(tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14)).dtype, tf.float32)

  def test_64bit_behavior_not_enable_x64_readme(self):
    # Tests some of the examples from the README
    if config.jax_enable_x64:
      self.skipTest("requires not x64 mode")

    # JAX and TF have same default float types if JAX_ENABLE_X64=0
    self.assertEqual(tf.math.sin(3.14).dtype, tf.float32)
    self.assertEqual(jnp.sin(3.14).dtype, jnp.float32)

    self.assertEqual(tf.math.sin(np.float64(3.14)).dtype, tf.float64)
    # JAX forces values to 32-bit
    self.assertEqual(jnp.sin(np.float64(3.14)).dtype, jnp.float32)

    # jax2tf.convert has the same behavior as JAX
    self.assertEqual(jax2tf.convert(jnp.sin)(3.14).dtype, tf.float32)
    self.assertEqual(jax2tf.convert(jnp.sin)(np.float64(3.14)).dtype, tf.float32)
    self.assertEqual(tf.function(jax2tf.convert(jnp.sin), autograph=False)(tf.Variable(3.14, dtype=tf.float64)).dtype, tf.float32)

  def test_function(self):
    f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
    self.ConvertAndCompare(f_jax, 0.7)

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_disabled(self, with_function=False):
    f_tf = jax2tf.convert(jnp.tan, with_gradient=False)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    x = tf.ones([])

    # With tf.function the error is raised when we evaluate f_tf(x), in
    # eager mode when we evaluate tape.gradient(y, x)
    with self.assertRaisesRegex(LookupError,
                                "Gradient explicitly disabled.*The jax2tf-converted function does not support gradients"):
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = f_tf(x)
        _ = tape.gradient(y, x)

  @jtu.sample_product(with_function=[False, True])
  def test_gradients(self, with_function=True):
    def f(x, y):
      return x * x, x * y
    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    default_float_type = jax2tf.dtype_of_val(4.)
    x = tf.Variable(4., dtype=jax2tf.dtype_of_val(4.))
    y = tf.Variable(5., dtype=default_float_type)
    with tf.GradientTape(persistent=True) as tape:
      u, v = f_tf(x, y)

    self.assertAllClose(2. * 4., tape.gradient(u, x))
    self.assertAllClose(0., tape.gradient(u, y))
    self.assertAllClose(5., tape.gradient(v, x))
    self.assertAllClose(4., tape.gradient(v, y))

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_pytree(self, with_function=True):
    def f(xy: Tuple[float, float]) -> Dict[str, float]:
      x, y = xy
      return dict(one=x * x, two=x * y)

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    default_float_dtype = jax2tf.dtype_of_val(4.)
    x = tf.Variable(4., dtype=default_float_dtype)
    y = tf.Variable(5., dtype=default_float_dtype)
    with tf.GradientTape(persistent=True) as tape:
      uv = f_tf((x, y))

    self.assertAllClose(2. * 4., tape.gradient(uv["one"], x))
    self.assertAllClose(0., tape.gradient(uv["one"], y))
    self.assertAllClose(5., tape.gradient(uv["two"], x))
    self.assertAllClose(4., tape.gradient(uv["two"], y))

  def test_custom_pytree_readme(self):
    # Code examples from README.md
    class CustomPair:
      def __init__(self, a, b):
        self.a = a
        self.b = b

    jax.tree_util.register_pytree_node(CustomPair,
                                       lambda x: ((x.a, x.b), None),
                                       lambda _, ab: CustomPair(*ab))
    def f_jax(pair: CustomPair):
      return np.float32(2.) * pair.a + np.float32(3.) * pair.b
    f_tf = jax2tf.convert(f_jax)

    x = CustomPair(np.float32(4.), np.float32(5.))
    res_jax = f_jax(x)
    # TF execution works as long as JAX can flatten the arguments and results
    res_tf = f_tf(x)
    self.assertAllClose(res_jax, res_tf.numpy())
    res_tf_2 = tf.function(f_tf, autograph=False, jit_compile=True)(x)
    self.assertAllClose(res_jax, res_tf_2)

    # wrapped TF function to use only standard containers
    def f_tf_wrapped(a, b):
      return f_tf(CustomPair(a, b))

    # Try to put into SavedModel
    my_model = tf.Module()
    # Save a function that can take scalar inputs.
    my_model.f = tf.function(f_tf_wrapped, autograph=False,
                             input_signature=[tf.TensorSpec([], tf.float32),
                                              tf.TensorSpec([], tf.float32)])
    model_dir = os.path.join(absltest.get_default_test_tmpdir(), str(id(my_model)))
    tf.saved_model.save(my_model, model_dir,
                        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))

    # Restoring (note: the restored model does *not* require JAX to run, just XLA).
    restored_model = tf.saved_model.load(model_dir)
    def restored_f(pair: CustomPair):
      return restored_model.f(pair.a, pair.b)

    res_tf_3 = restored_f(x)
    self.assertAllClose(res_jax, res_tf_3)
    grad_jax = jax.grad(f_jax)(x)

    x_v = [tf.Variable(x.a), tf.Variable(x.b)]
    with tf.GradientTape() as tape:
      res = f_tf_wrapped(*x_v)

      grad_tf = tape.gradient(res, x_v)
    self.assertAllClose(grad_jax.a, grad_tf[0])
    self.assertAllClose(grad_jax.b, grad_tf[1])


  @jtu.sample_product(with_function=[False, True])
  def test_gradients_with_ordered_dict_input(self, with_function=True):
    def f(inputs):
      out = 0.0
      for v in inputs.values():
        out += jnp.sum(v)
      return out

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    default_float_type = jax2tf.dtype_of_val(4.)
    x = tf.Variable([4.], dtype=default_float_type)
    y = tf.Variable([4., 5.], dtype=default_float_type)
    inputs = collections.OrderedDict()
    inputs['r'] = x
    inputs['d'] = y
    with tf.GradientTape(persistent=True) as tape:
      u = f_tf(inputs)

    self.assertAllClose(np.array([1.]), tape.gradient(u, x).numpy())
    self.assertAllClose(np.array([1., 1.]), tape.gradient(u, y).numpy())

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_with_custom_jvp(self, with_function=True):
    """Check gradients, for a function with custom JVP."""
    @jax.custom_jvp
    def f(x):
      return x * x

    @f.defjvp
    def f_jvp(primals, tangents):
      # 3 * x * x_t
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * x_dot
      return primal_out, tangent_out

    self.assertAllClose(4. * 4., f(4.))
    self.assertAllClose(3. * 4., jax.grad(f)(4.))

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    self.assertAllClose(4. * 4., f_tf(4.))
    x = tf.Variable(4., dtype=jax2tf.dtype_of_val(4.))
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f_tf(x)

    self.assertAllClose(4. * 4., y)
    self.assertAllClose(3. * 4., tape.gradient(y, x))

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_with_custom_vjp(self, with_function=True):
    """Check gradients, for a function with custom VJP."""
    @jax.custom_vjp
    def f(x):
      return x * x

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x
    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * ct_b,

    f.defvjp(f_fwd, f_bwd)

    self.assertAllClose(4. * 4., f(4.))
    self.assertAllClose(3. * 4., jax.grad(f)(4.))

    f_tf = jax2tf.convert(f, with_gradient=True)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    self.assertAllClose(4. * 4., f_tf(4.))
    x = tf.Variable(4., dtype=jax2tf.dtype_of_val(4.))
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = f_tf(x)

    self.assertAllClose(4. * 4., y)
    self.assertAllClose(3. * 4., tape.gradient(y, x))

  def test_gradient_with_float0_intermediate(self):
    # Gradient over integer-argument functions
    def f(x, y):  # x is an int, y is a float
      return 2 * x + y

    def g(x):  # x: f32
      return 2. * f(3 * x.astype("int32"), x * 4.)

    x = 2.
    grad_g = jax.grad(g)
    self.ConvertAndCompare(grad_g, x)

  def test_gradient_with_float0_result(self):
    # Gradient over integer-argument functions, with float0 result
    def f(x, y):  # x is an int, y is a float
      return 2 * x + y

    def g(x):  # x: i32
      return jnp.sum(2. * f(3 * x, 4. * jnp.array(x, jnp.dtype("float32"))))

    grad_g = jax.grad(g, allow_int=True)
    x = 2
    d_dx_jax = grad_g(x)
    d_dx_tf = jax2tf.convert(grad_g)(x)
    self.assertEqual(d_dx_jax.dtype, dtypes.float0)
    self.assertAllClose(jnp.zeros(np.shape(d_dx_jax), np.int32),
                        d_dx_tf.numpy())

    shape = (3, 4)
    x = np.ones(shape, dtype=np.int32)
    d_dx_jax = grad_g(x)
    d_dx_tf = jax2tf.convert(grad_g)(x)
    self.assertEqual(d_dx_jax.dtype, dtypes.float0)
    self.assertAllClose(jnp.zeros(np.shape(d_dx_jax), np.int32),
                        d_dx_tf.numpy())

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_unused_argument_readme(self, with_function=False):
    # x1 and x3 are not used. x3 has integer type.
    def fn(x0, x1, x2, x3):
      return x0 * 0. + x2 * 2.

    xs = [tf.Variable(x) for x in [10., 11., 12., 13]]
    with tf.GradientTape(persistent=True) as tape:
      res = fn(*xs)

    g_tf_native = tape.gradient(res, xs)
    self.assertAllClose(g_tf_native[0].numpy(), np.float32(0.))
    self.assertIsNone(g_tf_native[1])
    self.assertAllClose(g_tf_native[2].numpy(), np.float32(2.))
    self.assertIsNone(g_tf_native[3])

    g_tf_native_0 = tape.gradient(res, xs,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertAllClose(g_tf_native_0[0].numpy(), np.float32(0.))
    self.assertAllClose(g_tf_native_0[1].numpy(), np.float32(0.))
    self.assertAllClose(g_tf_native_0[2].numpy(), np.float32(2.))
    self.assertAllClose(g_tf_native_0[3].numpy(), np.int32(0))

    # Now with jax2tf.convert
    with tf.GradientTape(persistent=True) as tape:
      conv_fn = jax2tf.convert(fn, with_gradient=True)
      if with_function:
        conv_fn = tf.function(conv_fn, autograph=False)
      res = conv_fn(*xs)

    g_jax2tf = tape.gradient(res, xs)
    # Returns: 0., 0., 2., None
    # Note that the gradient for x1 is 0.
    self.assertAllClose(g_jax2tf[0].numpy(), np.float32(0.))
    self.assertAllClose(g_jax2tf[1].numpy(), np.float32(0.))
    self.assertAllClose(g_jax2tf[2].numpy(), np.float32(2.))
    self.assertIsNone(g_jax2tf[3])

    g_jax2tf = tape.gradient(res, xs,
                               unconnected_gradients=tf.UnconnectedGradients.ZERO)
    self.assertAllClose(g_jax2tf[0].numpy(), np.float32(0.))
    self.assertAllClose(g_jax2tf[1].numpy(), np.float32(0.))
    self.assertAllClose(g_jax2tf[2].numpy(), np.float32(2.))
    self.assertAllClose(g_jax2tf[3].numpy(), np.int32(0))

  @jtu.sample_product(with_function=[False, True])
  def test_gradients_int_argument(self, with_function=False):
    # https://github.com/google/jax/issues/6975
    # Also issue #6975.
    # An expanded version of test_gradients_unused_argument
    state = dict(
        float_used=np.array([0.7, 0.9], dtype=np.float32),
        float_passthrough=np.float16(1.),
        float_unused=np.array([1.1, 2.2, 3.3], dtype=np.float32),
        int_used=np.int16(5),
        int_passthrough=np.int8(7),
        int_unused=np.array([1, 2, 3], dtype=np.uint32),
        bool_used=np.array([True, False, False, True], dtype=np.bool_),
        bool_passthrough=np.array([True, False, False, True, False], dtype=np.bool_),
        bool_unused=np.array([[True, False], [False, True]], dtype=np.bool_),
    )
    def jax_f(state):
      res = dict(state,
                 float_used=2. * state["float_used"],
                 int_used=3 * state["int_used"],
                 bool_used=(state["bool_used"] == state["bool_used"]))
      del res["float_unused"]
      del res["int_unused"]
      del res["bool_unused"]
      return res

    args = (state,)
    res_jax = jax_f(*args)
    # Native JAX AD
    vjp_jax_fun, args_vjp = tf_test_util.TransformJaxVJP(jax_f, args, res_jax)
    grad_jax, = vjp_jax_fun(*args_vjp)

    def compare_with_overrides(*, what, expected, **expected_overrides):
      what_keys = set(what.keys())
      expected_keys = set(expected.keys())
      self.assertEqual(what_keys, expected_keys)
      for k, w in what.items():
        e = expected[k]
        if k in expected_overrides:
          if expected_overrides[k] == "ZERO":
            e = np.zeros_like(w)
          elif expected_overrides[k] == "ZERO_INT32":
            e = np.zeros(np.shape(w), dtype=np.int32)
          elif expected_overrides[k] == "ONE":
            e = np.ones_like(w)
          else:
            e = expected_overrides[k]

        if e is None:
          self.assertIsNone(w, msg=k)
        else:
          self.assertIsNotNone(w, msg=k)
        w = w.numpy() if isinstance(w, tf.Tensor) else e
        e = e.numpy() if isinstance(e, tf.Tensor) else e
        try:
          self.assertAllClose(e, w, err_msg=k)
        except:
          print(f"Failed at {k}")
          raise


    # compare_with_overrides(g_jax, {},
    #   bool_passthrough=np.zeros(state["bool_passthrough"].shape, dtype=dtypes.float0),
    #   bool_unused=np.zeros(state["bool_unused"].shape, dtype=dtypes.float0),
    #   bool_used=np.zeros(state["bool_used"].shape, dtype=dtypes.float0),
    #   float_passthrough=np.ones_like(state["float_passthrough"]),
    #   float_unused=np.zeros_like(state["float_unused"]),
    #   float_used=np.ones_like(state["float_used"]) * np.array(2., dtype=state["float_used"].dtype),
    #   int_passthrough=np.zeros(state["int_passthrough"].shape, dtype=dtypes.float0),
    #   int_unused=np.zeros(state["int_unused"].shape, dtype=dtypes.float0),
    #   int_used=np.zeros(state["int_used"].shape, dtype=dtypes.float0))


    # Now native TF gradients, only to test how native TF AD works
    _, (grad_tf_0,) = tf_test_util.ComputeTfValueAndGrad(
        jax_f, args, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    compare_with_overrides(what=grad_tf_0,
                           expected=grad_jax,
                           float_unused="ZERO",
                           bool_used="ZERO", bool_passthrough="ONE", bool_unused="ZERO",
                           int_used="ZERO", int_passthrough="ONE", int_unused="ZERO")

    _, (grad_tf_None,) = tf_test_util.ComputeTfValueAndGrad(
        jax_f, args,
        unconnected_gradients=tf.UnconnectedGradients.NONE)
    compare_with_overrides(what=grad_tf_None,
                           expected=grad_tf_0,
                           float_unused=None, int_used=None, int_unused=None,
                           bool_used=None, bool_unused=None)

    f_tf_jax = jax2tf.convert(jax_f)
    if with_function:
      f_tf_jax = tf.function(f_tf_jax, autograph=False)

    _, (grad_tf_jax_0,) = tf_test_util.ComputeTfValueAndGrad(f_tf_jax, args)
    # Same results as TF native AD with tf.UnconnectedGradients.ZERO
    compare_with_overrides(what=grad_tf_jax_0,
                           expected=grad_tf_0,
                           int_passthrough="ZERO", bool_passthrough="ZERO")

    _, (grad_tf_jax_None,) = tf_test_util.ComputeTfValueAndGrad(
        f_tf_jax, args,
        unconnected_gradients=tf.UnconnectedGradients.NONE)
    compare_with_overrides(what=grad_tf_jax_None,
                           expected=grad_tf_0,
                           int_used=None, int_passthrough=None, int_unused=None,
                           bool_unused=None, bool_used=None, bool_passthrough=None)

    # Not convert the JAX gradient function
    tf_vjp_jax_fun = jax2tf.convert(vjp_jax_fun)
    grad_tf_vjp_jax, = tf_vjp_jax_fun(*args_vjp)
    compare_with_overrides(what=grad_tf_vjp_jax,
                           expected=grad_tf_0,
                           bool_passthrough="ZERO_INT32",
                           bool_unused="ZERO_INT32", bool_used="ZERO_INT32",
                           int_passthrough="ZERO_INT32", int_unused="ZERO_INT32",
                           int_used="ZERO_INT32")

  def test_readme_gradient_int(self):
    x = np.array(2, dtype=np.int16)

    def f_jax(x):  # x: int16
      return x.astype(np.float32) * 2.

    print(jax.grad(f_jax, allow_int=True)(x))
    # returns a special `float0`: array((b'',), dtype=[('float0', 'V')])

    print(jax2tf.convert(jax.grad(f_jax, allow_int=True))(x))
    # returns a 0 with same shape as x, but with dtype int32

    def f_tf(x):  # x: int16
      return tf.cast(x, tf.float32) * 2.

    xv = tf.Variable(x)
    with tf.GradientTape(persistent=True) as tape:
      print(tape.gradient(f_tf(xv), xv))
      # returns None
      print(tape.gradient(f_tf(xv), xv,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO))
      # returns 0 with the same shape and dtype as x


  def test_convert_argument_non_callable_error(self):
    with self.assertRaisesRegex(TypeError, "Expected a callable value"):
      jax2tf.convert(5.)

  def test_convert_argument_non_tensor_error(self):
    with self.assertRaisesRegex(TypeError,
                                "Argument.*should be NumPy array"):
      jax2tf.convert(lambda x: x)(lambda y: y)

  def test_argument_eager_tensor(self):
    x = jax2tf.convert(jnp.sin)(1.)
    jax2tf.convert(jnp.cos)(x)  # No error

  def test_checkpoint_wrapper_types(self):
    m = tf.Module()
    m.a = [tf.Module(), tf.Module()]
    m.b = (tf.Module(), tf.Module())
    m.c = {'a': tf.Module(), 'b': tf.Module()}
    self.assertNotEqual(type(m.a), list)
    self.assertNotEqual(type(m.b), tuple)
    self.assertNotEqual(type(m.c), dict)
    self.assertLen(jax.tree_util.tree_leaves(m.a), 2)
    self.assertLen(jax.tree_util.tree_leaves(m.b), 2)
    self.assertLen(jax.tree_util.tree_leaves(m.c), 2)

  def test_issue_10586(self):

    class JaxModule(tf.Module):
      def __init__(self):
        self._params = {'w': tf.Variable(tf.ones([784, 10]), name='w'),
                        'b': tf.Variable(tf.ones([10]), name='b')}

      def __call__(self, x):
        return jax2tf.convert(lambda p, x: x @ p['w'] + p['b'])(self._params, x)

    net = JaxModule()
    images = tf.ones([1, 784])

    with tf.GradientTape() as tape:
      loss = tf.reduce_sum(net(images))
    params = tape.watched_variables()
    grads = tape.gradient(loss, params)
    for var, grad in zip(params, grads):
      self.assertEqual(var.shape, grad.shape, msg=var.name)

  def test_custom_jvp(self):
    """Conversion of function with custom JVP"""

    @jax.custom_jvp
    def f(x):
      return x * x

    @f.defjvp
    def f_jvp(primals, tangents):
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * x_dot
      return primal_out, tangent_out

    arg = 0.7
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "jvp")
    self.TransformConvertAndCompare(f, arg, "vmap")
    self.TransformConvertAndCompare(f, arg, "jvp_vmap")
    self.TransformConvertAndCompare(f, arg, "grad")
    self.TransformConvertAndCompare(f, arg, "grad_vmap")

  def test_custom_vjp(self):
    """Conversion of function with custom VJP"""

    @jax.custom_vjp
    def f(x):
      return x * x

    # f_fwd: a -> (b, residual)
    def f_fwd(x):
      return f(x), 3. * x

    # f_bwd: (residual, CT b) -> [CT a]
    def f_bwd(residual, ct_b):
      return residual * ct_b,

    f.defvjp(f_fwd, f_bwd)
    arg = 0.7
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "vmap")
    self.TransformConvertAndCompare(f, arg, "grad")
    self.TransformConvertAndCompare(f, arg, "grad_vmap")

  def test_remat(self):
    def f(x1):
      x2 = jnp.sin(x1)
      x3 = jnp.sin(x2)
      x4 = jnp.sin(x3)
      return x4
    remat_f = ad_checkpoint.checkpoint(f)

    # The computation of grad_f computes "sin" 5 times, 3 for the forward pass
    # and then to rematerialize "x2" and "x3" in the backward pass.
    arg = np.array(3.)
    f_tf = jax2tf.convert(jax.grad(remat_f))
    f_tf_hlo = self.TfToHlo(f_tf, arg)
    self.assertRegex(f_tf_hlo, r"opt-barrier")

  def test_remat_free_var(self):
    def f(x):
      y = 2 * x

      @ad_checkpoint.checkpoint
      def g():
        return y

      return g()
    arg = 3.
    self.TransformConvertAndCompare(f, arg, None)
    self.TransformConvertAndCompare(f, arg, "grad")

  def test_checkpoint_name(self):
    def f_jax(x):
      return ad_checkpoint.checkpoint_name(jnp.sin(x), "sin")
    jax2tf.convert(f_jax)(1.)  # No error.

  def test_convert_nullary_func(self):
    # Even nullary functions are converted to TF (as opposed to constant-folded
    # in JAX prior to conversion).
    def f_jax():
      return jnp.sin(1.)
    f_tf = jax2tf.convert(f_jax)
    # for native lowering the HLO we get from TF is constant-folded, so this
    # test fails.
    if not config.jax2tf_default_experimental_native_lowering:
      self.assertIn("sine(", self.TfToHlo(f_tf))

  def test_convert_of_nested_independent_jit(self):
    def func(x):
      def inner1(y):
        return x + y
      # The JIT does not have data dependency
      return jax.jit(inner1)(1.)

    jax2tf.convert(func)(2.)

  def test_convert_of_nested_dependent_jit(self):
    def func(x):
      def inner1(y):
        return x + y
      # The JIT does have data dependency
      return jax.jit(inner1)(x)

    jax2tf.convert(func)(2.)  # No error

  def test_jit_unused(self):
    def f_jax(x, y_unused):
      return x * np.float32(2.)
    x, y_unused = np.float32(5.), np.arange(7, dtype=np.int32)
    res_tf = jax2tf.convert(jax.jit(f_jax, keep_unused=False))(x, y_unused)
    self.assertAllClose(f_jax(x, None), res_tf)

  def test_jit_unused_grad(self):
    def f_jax(x, y_unused):
      return x * np.float32(2.)

    x, y_unused = np.float32(5.), np.arange(7, dtype=np.int32)
    f_tf = jax2tf.convert(jax.jit(f_jax, keep_unused=False))
    xv, y_unused_v = tf.Variable(x), tf.Variable(y_unused)
    with tf.GradientTape() as tape:
      res_tf = f_tf(xv, y_unused_v)
      grad_tf_x, grad_tf_y = tape.gradient(res_tf, (xv, y_unused_v))

    self.assertAllClose(f_jax(x, None), res_tf)
    self.assertAllClose(np.float32(2.), grad_tf_x)
    self.assertIsNone(grad_tf_y)

  def test_nested_convert_error(self):
    def outer(y):
      return jax2tf.convert(jnp.sin)(y)  # Inner convert takes tracer args
    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      jax2tf.convert(outer)(np.ones((4, )))

  def test_nested_convert_error_non_tracer(self):
    """The inner convert takes non-tracer arguments"""
    def outer(y):
      sin_1 = jax2tf.convert(jnp.sin)(1.)  # Inner convert takes non-tracer arg
      return y + sin_1

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      jax2tf.convert(outer)(2.)

  @jtu.sample_product(transform=["jit", "jvp", "grad", "vmap"])
  def test_convert_under_transform_error(self, transform="vmap"):
    def outer(y):
      return jax2tf.convert(jnp.sin)(y)  # Inner convert takes tracer args

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      self.TransformConvertAndCompare(outer, np.ones((4,)), transform)

  @jtu.sample_product(transform=["jit", "jvp", "grad", "vmap"])
  def test_convert_under_transform_error_non_tracer(self, transform="vmap"):
    def outer(y):
      sin_1 = jax2tf.convert(jnp.sin)(1.)  # Inner convert takes non-tracer arg
      return y + sin_1

    with self.assertRaisesRegex(
        ValueError, "convert must be used outside all JAX transformations"):
      self.TransformConvertAndCompare(outer, np.ones((4,)), transform)

  def test_name_scope(self):
    def run_tf():
      @jax.named_call
      def my_test_function_jax(x):
        return x * x

      def caller_jax(x):
        return my_test_function_jax(jnp.sin(x))

      out = jax2tf.convert(caller_jax, with_gradient=False)(2.)
      return out
    if config.jax2tf_default_experimental_native_lowering:
      self.assertIn("my_test_function_jax/mul", self.TfToHlo(run_tf))
    else:
      graph_def = str(tf.function(run_tf, autograph=False).get_concrete_function().graph.as_graph_def())
      if "my_test_function_jax/pjit_fn_/Mul" not in graph_def:
        self.assertIn("my_test_function_jax/jit_fn_/Mul", graph_def)

  def test_bfloat16_constant(self):
    # Re: https://github.com/google/jax/issues/3942
    def jax_fn_scalar(x):
      x = x.astype(jnp.bfloat16)
      x *= 2.
      return x

    def jax_fn_array(x):
      x = x.astype(jnp.bfloat16)
      x *= np.array([1.5, 2.5, 3.5], jnp.bfloat16)
      return x

    tf_fn_scalar = jax2tf.convert(jax_fn_scalar)
    self.assertAllClose(tf_fn_scalar(1.375).numpy(), jnp.bfloat16(2.750))

    tf_fn_array = jax2tf.convert(jax_fn_array)
    self.assertAllClose(
        tf_fn_array(np.array([3, 4, 5])), np.array([4.5, 10, 17.5],
                                                   jnp.bfloat16))

  def test_shared_constants(self):
    # Check that the constants are shared properly in converted functions
    # See https://github.com/google/jax/issues/7992.
    if config.jax2tf_default_experimental_native_lowering:
      raise unittest.SkipTest("shared constants tests not interesting for native lowering")
    const = np.random.uniform(size=256).astype(np.float32)  # A shared constant
    def f(x):
      return x + const + const + const + const

    f_tf_consts = self.FindLargeTfConstants(jax2tf.convert(f), const)
    self.assertLen(f_tf_consts, 1)

  def test_shared_constants_under_cond(self):
    # Check that the constants are shared properly in converted functions
    # See https://github.com/google/jax/issues/7992.
    if config.jax2tf_default_experimental_native_lowering:
      raise unittest.SkipTest("shared constants tests not interesting for native lowering")
    const_size = 512
    const = np.random.uniform(size=const_size).astype(np.float32)  # A shared constant
    x = np.ones((const_size,), dtype=np.float32)
    def f1(x):
      # Ensure that we first see the constants in the inside jaxpr
      return lax.cond(x[0] >= 0., lambda x: x + const, lambda x: x * const, x) + const
    def f2(x):
      return f1(x) + const  # The extra const should not cost anything
    f1_consts = self.FindLargeTfConstants(jax2tf.convert(f1), x, at_least=const_size)
    f2_consts = self.FindLargeTfConstants(jax2tf.convert(f2), x, at_least=const_size)
    self.assertLen(f2_consts, len(f1_consts))

  def test_shared_constants_under_scan(self):
    # See https://github.com/google/jax/issues/7992.
    if config.jax2tf_default_experimental_native_lowering:
      raise unittest.SkipTest("shared constants tests not interesting for native lowering")
    const_size = 512
    const = np.random.uniform(size=const_size).astype(np.float32)  # A shared constant
    xs = np.ones((8, const_size), dtype=np.float32)
    def f1(xs):
      res, _ = lax.scan(lambda carry, x: (carry + x + const, None),
                        jnp.zeros((const_size,), dtype=np.float32), xs)
      return res

    def f2(xs):
      return f1(xs) + const  # The extra const should not be saved

    f1_consts = self.FindLargeTfConstants(jax2tf.convert(f1), xs, at_least=const_size)
    f2_consts = self.FindLargeTfConstants(jax2tf.convert(f2), xs, at_least=const_size)
    self.assertLen(f2_consts, len(f1_consts))

  def test_shared_constants_under_jit(self):
    # We do not share constants under jit.
    if config.jax2tf_default_experimental_native_lowering:
      raise unittest.SkipTest("shared constants tests not interesting for native lowering")
    const = np.random.uniform(size=(16, 16)).astype(np.float32)  # A shared constant
    @jax.jit
    def g_jit(x):
      return x * const
    def f(x):
      return g_jit(x) + const + const

    f_tf_graph_consts = self.FindLargeTfConstants(jax2tf.convert(f), const)
    self.assertLen(f_tf_graph_consts, 1)

  def test_shared_constants_randint(self):
    # randint has the property that the TF lowering of the randbits_p
    # primitive generates constants that did not exist in the Jaxpr. As such
    # it has created new errors related to the sharing of the constants.
    if config.jax2tf_default_experimental_native_lowering:
      raise unittest.SkipTest("shared constants tests not interesting for native lowering")

    key = jax.random.PRNGKey(42)

    def f_nested_jax(x):
      # Lowering this will generate a tf.constant(shape=(1,), dtype=np.int32)
      # that was not already in the Jaxpr, and hence JAX did not get a chance
      # to share.
      return x + jax.random.randint(key, shape=x.shape,
                                    minval=0, maxval=100, dtype=np.int32)
    def f_jax(x):
      res = lax.cond(x[0] >= 2, lambda: f_nested_jax(x), lambda: f_nested_jax(x))
      res += lax.while_loop(lambda x: f_nested_jax(x)[0] <= 0, f_nested_jax, x)
      # We also generate tf.while in the batching rule for cond
      res += jax.vmap(lambda x: lax.cond(x[0] >= 2,
                                         lambda: f_nested_jax(x),
                                         lambda: f_nested_jax(x)))(jnp.stack([x, x]))
      res += f_nested_jax(x)
      return res

    # Must be odd to trigger the failure
    x = np.array([123, 456, 789], dtype=np.int32)

    f_tf = tf.function(jax2tf.convert(f_jax), autograph=False)
    res_tf = f_tf(x)
    self.assertAllClose(res_tf, f_jax(x))

  def test_weak_types(self):
    mul = jax.jit(jnp.multiply)
    # The value `2` here should be weakly typed, and should not lead to
    # promotion.
    tf_fn = jax2tf.convert(lambda x: mul(x, 2.))
    self.assertAllClose(tf_fn(tf.constant(1.375, tf.bfloat16)).numpy(),
                        jnp.bfloat16(2.750))

  @jtu.sample_product(with_function=[False, True])
  def test_kwargs(self, with_function=True):
    # Re: https://github.com/google/jax/issues/6791
    def f_jax(*, x):
      return jnp.sum(x)
    f_tf = jax2tf.convert(f_jax)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    self.assertAllClose(
      f_tf(x=np.zeros(3, dtype=np.float32)),  # Call with kwargs.
      np.zeros((), dtype=np.float32))

  @jtu.sample_product(with_function=[False, True])
  def test_grad_kwargs(self, with_function=False):
    # Re: https://github.com/google/jax/issues/6791
    x = (np.zeros(3, dtype=np.float32),
         np.zeros(4, dtype=np.float32))
    def f_jax(*, x=(1., 2.)):
      return jnp.sum(x[0]) + 2. * jnp.sum(x[1])
    f_tf = jax2tf.convert(f_jax)
    if with_function:
      f_tf = tf.function(f_tf, autograph=False)
    xv = tf.nest.map_structure(tf.Variable, x)
    with tf.GradientTape() as tape:
      res = f_tf(x=xv)
    grad_tf = tape.gradient(res, xv)
    self.assertAllClose((np.full_like(x[0], fill_value=1.),
                         np.full_like(x[1], fill_value=2.)),
                        (grad_tf[0].numpy(), grad_tf[1].numpy()))


  @jtu.skip_on_flag("jax2tf_default_experimental_native_lowering", True)
  def test_enable_xla(self):
    # Tests that enable_xla flag is properly scoped to a conversion.
    def fun(x):
      # lax.reduce is unlikely to ever be convertible with enable_xla=False
      return lax.reduce(x, np.float32(0), lambda v, acc: v + acc, dimensions=(0, 1))

    tf_fun_with_xla = jax2tf.convert(fun, enable_xla=True)
    tf_fun_without_xla = jax2tf.convert(fun, enable_xla=False)
    x = np.ones((2, 3), dtype=np.float32)

    self.assertAllClose(fun(x), tf_fun_with_xla(x))
    with self.assertRaisesRegex(NotImplementedError,
                                "Call to reduce cannot be converted with enable_xla=False"):
      tf_fun_without_xla(x)

    # Now in reverse order (we had bugs with the management of enable_xla global)
    tf_fun2_without_xla = jax2tf.convert(lambda x: fun(x), enable_xla=False)
    tf_fun2_with_xla = jax2tf.convert(lambda x: fun(x), enable_xla=True)

    with self.assertRaisesRegex(NotImplementedError,
                                "Call to reduce cannot be converted with enable_xla=False"):
      tf_fun2_without_xla(x)
    self.assertAllClose(fun(x), tf_fun2_with_xla(x))

  def test_device_array_arg(self):
    self.ConvertAndCompare(jnp.sin, jnp.zeros((2, 3), jnp.float32))

  def test_randint(self):
    def randint(x):
      return jax.random.randint(
          jax.random.PRNGKey(42), shape=x.shape, minval=0, maxval=1)

    self.ConvertAndCompare(randint, np.ones((3, 4)))

  def test_op_metadata_simple(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # A simple example
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    def f_simple(x):
      return jnp.sin(x)

    x = np.ones((2, 3), np.float32)
    self.CheckOpMetadata(
        f_simple, x,
        [tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 2,
                                      op_name="jax2tf(f_simple)/sin",
                                      op_type="sin")
         ]
    )

  def test_op_metadata_sub_jit(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # Calling a jitted-function
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    def f_callee(x):
      return jnp.cos(x)
    def f_caller(x):
      y = jnp.tanh(x)
      z = jax.jit(f_callee)(y)
      return jnp.sin(z)

    x = np.ones((2, 3), np.float32)

    self.CheckOpMetadata(
        f_caller, x,
        [tf_test_util.OpMetadataGraph(tf_type="Tanh",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 4,
                                      op_name="jax2tf(f_caller)/tanh",
                                      op_type="tanh"),
         tf_test_util.OpMetadataGraph(tf_type="Cos",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 2,
                                      op_name="jax2tf(f_caller)/jit(f_callee)/cos",
                                      op_type="cos"),
         tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 6,
                                      op_name="jax2tf(f_caller)/sin",
                                      op_type="sin"),
         ]
    )

  def test_op_metadata_named(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # Calling a jax.named_call
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    def f_callee(x):
      return jnp.cos(x)
    def f_caller(x):
      y = jnp.tanh(x)
      z = jax.named_call(f_callee, name="callee")(y)
      return jnp.sin(z)

    x = np.ones((2, 3), np.float32)

    self.CheckOpMetadata(
        f_caller, x,
        [tf_test_util.OpMetadataGraph(tf_type="Tanh",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 4,
                                      op_name="jax2tf(f_caller)/tanh",
                                      op_type="tanh"),
         tf_test_util.OpMetadataGraph(tf_type="Cos",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 2,
                                      op_name="jax2tf(f_caller)/named(callee)/cos",
                                      op_type="cos"),
         tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 6,
                                      op_name="jax2tf(f_caller)/sin",
                                      op_type="sin"),
         ]
    )

  def test_op_metadata_while_and_cond(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # An example with while and cond
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    def f_while_cond(x):
      def body_fun(i_acc):
        i, acc = i_acc
        return (i + 1,
                (jnp.cos(acc) +
                 lax.cond(jnp.mod(i, 2) == 0,
                          lambda acc: jnp.sin(acc),
                          lambda acc: acc,
                          acc)))

      _, acc = lax.while_loop(
          lambda i_acc: i_acc[0] <= 5,
          body_fun, (0, x))
      return acc

    x = np.ones((2, 3), np.float32)
    self.CheckOpMetadata(
        f_while_cond, x,
        [tf_test_util.OpMetadataGraph(tf_type="Cos",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 5,
                                      op_name="jax2tf(f_while_cond)/while/body/cos",
                                      op_type="cos"),
         tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 7,
                                      op_name="jax2tf(f_while_cond)/while/body/branch_1_fun/sin",
                                      op_type="sin"),
         tf_test_util.OpMetadataGraph(tf_type="FloorMod",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 6,
                                      op_name="jax2tf(f_while_cond)/while/body/rem",
                                      op_type="rem"),
         ]
    )

  def test_op_metadata_batched_while(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    # An example with while and cond
    # The user_frame is used to compute line numbers for ops in the test.
    user_frame = source_info_util.user_frame(source_info_util.current())
    @jax.vmap
    def f_while(x):
      def body_fun(carry):
        new_carry = jnp.sin(carry)  # We look for "sin" in the graph
        return new_carry

      _, carry = lax.while_loop(
          lambda carry: jnp.all(carry <= x),  # We look for "le" in the graph
          body_fun, x)
      return carry

    shape = (3, 2)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    jax_comp = jax.xla_computation(f_while)(x)
    backend = jax._src.lib.xla_bridge.get_backend()
    modules = backend.compile(jax_comp).hlo_modules()
    jax_opt_hlo = modules[0].to_string()
    print(f"JAX OPT HLO = {jax_opt_hlo}")

    self.CheckOpMetadata(
        f_while, x,
        [tf_test_util.OpMetadataGraph(tf_type="Sin",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 4,
                                      op_name="jax2tf(f_while)/while/body/sin",
                                      op_type="sin"),
         tf_test_util.OpMetadataGraph(tf_type="LessEqual",
                                      source_file=__file__,
                                      source_line=user_frame.start_line + 8,
                                      op_name="jax2tf(f_while)/while/body_pred/le",
                                      op_type="le"),
         ]
    )

  def test_op_metadata_disabled(self):
    self.skipTest("include_xla_op_metadata not yet enabled")
    def f_simple(x):
      return jnp.sin(x)

    x = np.ones((2, 3), np.float32)
    self.CheckOpMetadata(
        f_simple, x,
        [],
        include_xla_op_metadata=False
    )

  def assertAllOperationStartWith(self, g: tf.Graph, scope_name: str):
    """Assert all operations name start with ```scope_name```.

    Also the scope_name only occur one time.
    """
    result = g.get_operations()
    if not result:
      self.fail("result is empty.")
    for op in result:
      logging.info("tf op.name = %s", op.name)
      if not op.name.startswith(scope_name):
        self.fail(f"{op.name} does not start with {scope_name}.")

  def test_name_scope_polymorphic(self):
    if config.jax2tf_default_experimental_native_lowering and not config.jax_dynamic_shapes:
      self.skipTest("shape polymorphism but --jax_dynamic_shapes is not set.")

    def func_jax(x, y):
      return jnp.sin(x) + jnp.cos(y)

    func_tf = jax2tf.convert(
        func_jax, polymorphic_shapes="(b,...)", with_gradient=True)

    outer_scope = "output_a"

    g = tf.Graph()
    with g.as_default() as g:
      with tf.name_scope(outer_scope):
        x = tf.Variable(
            tf.zeros(shape=(1, 5), dtype=tf.dtypes.float32), name="x")
        y = tf.compat.v1.placeholder(tf.dtypes.float32, (None, 5), "y")
        _ = func_tf(x, y)
    self.assertAllOperationStartWith(g, outer_scope)

    # wrap tf.function
    g2 = tf.Graph()
    with g2.as_default() as g:
      with tf.name_scope(outer_scope):
        x = tf.Variable(
            tf.zeros(shape=(1, 5), dtype=tf.dtypes.float32), name="x")
        y = tf.compat.v1.placeholder(tf.dtypes.float32, (None, 5), "y")
        _ = tf.function(func_tf, jit_compile=True, autograph=False)(x, y)
    self.assertAllOperationStartWith(g2, outer_scope)

  def test_name_scope_cond(self):

    def f(x):

      def f_pos(x):
        with jax.named_scope("jax_f_pos"):
          return lax.cond(x < 1., jnp.cos, jnp.sin, x)

      with jax.named_scope("jax_f_outer"):
        return lax.cond(x > 0., f_pos, lambda x: x, x)

    @tf.function(jit_compile=True, autograph=False)
    def outer_forward():
      with tf.name_scope("tf_outer_forward"):
        x = 0.5
        f_tf = jax2tf.convert(f)
        _ = f_tf(x)

    g = outer_forward.get_concrete_function().graph
    self.assertAllOperationStartWith(g, "tf_outer_forward")
    for func in g._functions.values():
      self.assertAllOperationStartWith(
          func.graph, "tf_outer_forward/jax2tf_f_/jax_f_outer")

    x = tf.Variable(0.5, name="tf_outer_back/x")

    @tf.function(jit_compile=True, autograph=False)
    def outer_back():
      with tf.name_scope("tf_outer_back"):
        f_tf = jax2tf.convert(f)
        with tf.GradientTape() as tape:
          res_tf = f_tf(x)
          _ = tape.gradient(res_tf, x)

    g = outer_back.get_concrete_function().graph
    self.assertAllOperationStartWith(g, "tf_outer_back")
    for func in g._functions.values():
      self.assertAllOperationStartWith(func.graph, "tf_outer_back")

  def test_name_scope_while_loop(self):

    def f(x):
      with tf.name_scope("outer_scope"):

        def condition(x):
          return jnp.sum(x, keepdims=False) < 100

        def body(x):
          return jnp.add(x, 2.0)

        result = jax.lax.while_loop(condition, body, x)
        return result

    tf_f = tf.function(jax2tf.convert(f), jit_compile=True, autograph=False)
    g = tf_f.get_concrete_function(tf.zeros((1, 3))).graph

    for func in g._functions.values():
      for op in func.graph.get_operations():
        if op.name.count(f"outer_scope/jax2tf_{f.__name__}_/while") > 1:
          self.fail(
              "tf graph has repeated name issue on when converting lax.while to tf.while."
              f"See op.name = : {op.name}")

def get_serialized_computation(
    f_jax: Callable,
    *args,
    abstracted_axes: Optional[Tuple[Dict[int, str]]] = None,
    use_pjit: bool = False,
    in_axis_resources = None,
    out_axis_resources = None) -> str:
  if use_pjit:
    assert not abstracted_axes
    lowered = pjit.pjit(
        f_jax, in_shardings=in_axis_resources, out_shardings=out_axis_resources
    ).lower(*args)
  else:
    lowered = jax.jit(f_jax, abstracted_axes=abstracted_axes).lower(*args)
  stablehlo_module_text = mlir.module_to_string(lowered._lowering.stablehlo())
  logging.info(f'Serialized ir.Module = {stablehlo_module_text}')
  return stablehlo_module_text


class XlaCallModuleTest(tf_test_util.JaxToTfTestCase):
  """Unit tests for XlaCallModule. Will move these eventually to TF."""
  def test_simple(self):

    def f_jax(x):
      return jnp.sin(x)

    x = np.ones((2, 3), dtype=np.float32)

    jax_res = f_jax(x)
    res = tfxla.call_module([x],
                            version=2,
                            module=get_serialized_computation(f_jax, x),
                            Tout=[jax_res.dtype],
                            Sout=[jax_res.shape])
    self.assertAllClose(tf.nest.map_structure(lambda t: t.numpy(), res),
                        [jax_res])

  def test_while(self):
    # With nested computation
    def f_jax(count, x):
      return lax.while_loop(lambda carry: carry[0] < count, lambda carry:
                            (carry[0] + 1, carry[1] + 1.), (0, x))[1]

    count = np.int32(5)
    x = np.ones((2, 3), dtype=np.float32)

    jax_res = f_jax(count, x)
    res = tfxla.call_module([count, x],
                            version=2,
                            module=get_serialized_computation(f_jax, count, x),
                            Tout=[jax_res.dtype],
                            Sout=[jax_res.shape])
    self.assertAllClose(tf.nest.map_structure(lambda t: t.numpy(), res),
                        [jax_res])

  def test_multiple_args_results(self):

    def f_jax(x1, x2):
      return (jnp.sin(x1), jnp.cos(x2))

    x1 = np.ones((2, 3), dtype=np.float32)
    x2 = np.ones((3, 4), dtype=np.float32)

    jax_res = f_jax(x1, x2)

    def f_tf(x1_tf, x2_tf):
      return tfxla.call_module([x1_tf, x2_tf],
                               version=2,
                               module=get_serialized_computation(f_jax, x1, x2),
                               Tout=[jax_res[0].dtype, jax_res[1].dtype],
                               Sout=[jax_res[0].shape, jax_res[1].shape])

    res = tf.function(f_tf, jit_compile=True, autograph=False)(x1, x2)
    self.assertAllClose(tf.nest.map_structure(lambda t: t.numpy(), res),
                        jax_res)

  @unittest.skip("TODO(necula): 'dynamic_iota' op can't be translated to XLA HLO")
  def test_shape_poly_arange(self):
    if not config.jax_dynamic_shapes:
      raise unittest.SkipTest("jax_dynamic_shapes must be enabled")
    def f_jax(x):  # x: f32[b]
      return jnp.arange(x.shape[0]) + x

    x1 = np.ones((5,), dtype=np.float32)
    jax_res = f_jax(x1)

    def f_tf(x1_tf):
      return tfxla.call_module([x1_tf],
                               version=2,
                               module=get_serialized_computation(
                                   f_jax, x1,
                                   abstracted_axes=({
                                       0: 'b'
                                   },)),
                               Tout=[jax_res.dtype],
                               Sout=[jax_res.shape],
                               dim_args_spec=('0.0',))

    res = tf.function(f_tf, jit_compile=True, autograph=False)(x1)
    self.assertAllClose(
        tf.nest.map_structure(lambda t: t.numpy(), res), jax_res)

  @jtu.with_mesh([("x", 2)])
  def test_pjit_basic1D(self):

    def func_jax(x, y):
      return x + y

    shape = (8, 10)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    in_axis_resources = (P("x"), P("x"))
    out_axis_resources = None
    res_jax = pjit.pjit(
        func_jax,
        in_shardings=in_axis_resources,
        out_shardings=out_axis_resources,
    )(x, x)
    module = get_serialized_computation(
        func_jax,
        x,
        x,
        use_pjit=True,
        in_axis_resources=in_axis_resources,
        out_axis_resources=out_axis_resources)

    def f_tf(x_tf, y_tf):
      return tfxla.call_module([x_tf, y_tf],
                               version=2,
                               module=module,
                               Tout=[x.dtype],
                               Sout=[x.shape])

    res_tf = tf.function(f_tf, jit_compile=True, autograph=False)(x, x)[0]
    self.assertAllClose(res_tf.numpy(), res_jax)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
