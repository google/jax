# Copyright 2022 The JAX Authors.
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

import collections
import contextlib
import functools
import logging
import textwrap
import time
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import maps
from jax._src.maps import xmap
from jax._src import test_util as jtu
from jax._src import util
from jax._src.lib import xla_client
from jax._src.lib import xla_extension_version
from jax.experimental import io_callback
from jax.experimental import pjit
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

config.parse_flags_with_absl()


def _format_multiline(text):
  return textwrap.dedent(text).lstrip()

prev_xla_flags = None


def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)


# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()

map, unsafe_map = util.safe_map, map

# Some test methods take a kwarg
# callback=[io_callback(ordered=True) | io_callback(ordered=False) | pure_callback]
io_callback_ordered = functools.partial(io_callback, ordered=True)
io_calback_unordered = functools.partial(io_callback, ordered=False)
with_pure_and_io_callbacks = parameterized.named_parameters(
  dict(testcase_name=flavor,
       callback=dict(io_unordered=io_calback_unordered,
                     io_ordered=io_callback_ordered,
                     pure=jax.pure_callback)[flavor])
  for flavor in ("io_unordered", "io_ordered", "pure")
)


class PythonCallbackTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu", "tpu"]):
      self.skipTest(f"Host callback not supported on {jtu.device_under_test()}")

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  @with_pure_and_io_callbacks
  def test_callback_with_scalar_values(self, *, callback):
    @jax.jit
    def f(x):
      return callback(lambda x: x + np.float32(1.),
                      core.ShapedArray(x.shape, x.dtype), x)

    out = f(0.)
    self.assertEqual(out, 1.)

  @parameterized.named_parameters(
    dict(testcase_name=f"{flavor}_expect_dtype_{expect_dtype}",
         callback=dict(io_unordered=io_calback_unordered,
                       io_ordered=io_callback_ordered,
                       pure=jax.pure_callback)[flavor],
         expect_dtype=expect_dtype)
    for flavor in ("io_unordered", "io_ordered", "pure")
    for expect_dtype in (np.int32, np.int64, np.float32, np.float64)
  )
  def test_callback_returning_python_literal(self, *, callback, expect_dtype):
    returned_literal = 42 if expect_dtype in (np.int32, np.int64) else 42.
    @jax.jit
    def f(x):
      return callback(lambda x: returned_literal,
                      core.ShapedArray((), expect_dtype), x)

    if not config.enable_x64.value:
      ctx = self.assertRaisesRegex(Exception, "Cannot return 64-bit values")
    elif expect_dtype in (np.int32, np.float32):
      ctx = self.assertRaisesRegex(Exception, "Incorrect output dtype")
    else:
      ctx = contextlib.nullcontext()

    with ctx:
      out = f(0.)
      jax.effects_barrier()
      self.assertEqual(out, returned_literal)

  @parameterized.named_parameters(
    dict(testcase_name=f"{flavor}_{dtype}",
         dtype=dtype,
         callback=dict(io_unordered=io_calback_unordered,
                       io_ordered=io_callback_ordered,
                       pure=jax.pure_callback)[flavor])
    for flavor in ("io_unordered", "io_ordered", "pure")
    for dtype in jtu.dtypes.all
  )
  def test_callback_works_with_all_types(self, *, callback, dtype):
    def host_func(x):
      if dtype == np.bool_:
        return ~ x
      else:
        return x + x
    _received = None
    def _cb(x):
      nonlocal _received
      _received = x
      return host_func(x)

    if dtype == np.bool_:
      x = np.array([True, False, True, True], dtype=np.bool_)
    else:
      x = np.arange(4, dtype=dtype)
    @jax.jit
    def f(x):
      return callback(_cb,
                      core.ShapedArray(x.shape, x.dtype), x)

    out = f(x)
    self.assertAllClose(out, host_func(x))
    jax.effects_barrier()
    self.assertAllClose(_received, x)

  @with_pure_and_io_callbacks
  def test_callback_with_wrong_number_of_args(self, *, callback):

    @jax.jit
    def f():
      # Calling a function that expects `x` with no arguments
      return callback(lambda x: np.ones(4, np.float32),
                      core.ShapedArray((4,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  @with_pure_and_io_callbacks
  def test_callback_with_wrong_number_of_returned_values(self, *, callback):

    @jax.jit
    def f(x):
      # Calling a function with two return values that expects one return value
      return callback(lambda x: (x, np.ones(4, np.float32)), x, x)

    with self.assertRaises(RuntimeError):
      f(2.)
      jax.effects_barrier()

    @jax.jit
    def g():
      # Specifically for io_callback, calling a function with a return value
      # that expects no return values
      return io_callback(lambda: None, (core.ShapedArray(
          (1,), np.float32), core.ShapedArray((2,), np.float32)))

    with self.assertRaises(RuntimeError):
      g()
      jax.effects_barrier()

  @with_pure_and_io_callbacks
  def test_callback_with_wrong_shape_outputs(self, *, callback):

    @jax.jit
    def f():
      # Calling a function expected a (1,) shaped return value but getting ()
      return callback(lambda: np.float32(1.), core.ShapedArray((1,),
        np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  @with_pure_and_io_callbacks
  def test_callback_with_wrong_dtype_outputs(self, *, callback):

    def _cb():
      return np.array([1], np.float64)

    @jax.jit
    def f():
      # Calling a function expected a f32 return value but getting f64
      return callback(_cb, core.ShapedArray((1,), np.float32))

    with self.assertRaises(RuntimeError):
      f()
      jax.effects_barrier()

  @with_pure_and_io_callbacks
  def test_callback_with_wrongly_specified_64_bit_dtype(self, *, callback):
    if config.enable_x64.value:
      raise unittest.SkipTest("Test only needed when 64-bit mode disabled.")

    @jax.jit
    def f():
      return callback(lambda: np.float64(1.),
                      core.ShapedArray((), np.float64))

    with self.assertRaises(ValueError):
      f()
      jax.effects_barrier()

  @with_pure_and_io_callbacks
  def test_callback_with_single_return_value(self, *, callback):

    @jax.jit
    def f():
      return callback(lambda: np.ones(4, np.float32),
                      core.ShapedArray((4,), np.float32))

    out = f()
    jax.effects_barrier()
    np.testing.assert_allclose(out, np.ones(4, np.float32))

  @with_pure_and_io_callbacks
  def test_callback_with_multiple_return_values(self, *, callback):

    @jax.jit
    def f():
      return callback(lambda: (np.ones(4, np.float32), np.ones(5, np.int32)),
                      (core.ShapedArray(
                          (4,), np.float32), core.ShapedArray((5,), np.int32)))

    x, y = f()
    jax.effects_barrier()
    np.testing.assert_allclose(x, np.ones(4, np.float32))
    np.testing.assert_allclose(y, np.ones(5, np.int32))

  @with_pure_and_io_callbacks
  def test_callback_with_multiple_arguments_and_return_values(self, *, callback):

    def _callback(x, y, z):
      return (x, y + z)

    @jax.jit
    def f(x, y, z):
      return callback(_callback, (core.ShapedArray(
          (3,), x.dtype), core.ShapedArray((3,), x.dtype)), x, y, z)

    x, y = f(jnp.ones(3), jnp.arange(3.), jnp.arange(3.) + 1.)
    jax.effects_barrier()
    np.testing.assert_allclose(x, np.ones(3))
    np.testing.assert_allclose(y, np.array([1., 3., 5]))

  @with_pure_and_io_callbacks
  def test_send_zero_dim_arrays(self, *, callback):
    result = np.full((2,), 42.0, dtype=np.float32)
    x = np.zeros((2, 0), np.float32)
    def _callback(x):  # x: f32[2, 0]
      return result

    @jax.jit
    def f(x):
      return callback(
          _callback, core.ShapedArray(result.shape, result.dtype), x)
    jax.effects_barrier()
    self.assertAllClose(f(x), result)

  @with_pure_and_io_callbacks
  def test_send_zero_dim_and_non_zero_dim_arrays(self, *, callback):
    x = np.zeros((2, 0), np.float32)
    y = np.full((2,), 42.0, dtype=np.float32)
    result = y
    def _callback(x, y):  # x: f32[2, 0]  y: f32[2]
      return y

    @jax.jit
    def f(x, y):
      return callback(
          _callback, core.ShapedArray(result.shape, result.dtype), x, y)
    jax.effects_barrier()
    self.assertAllClose(f(x, y), result)

  @with_pure_and_io_callbacks
  def test_recv_zero_dim_arrays(self, *, callback):
    result = np.full((2, 0), 42.0, dtype=np.float32)
    x = np.zeros((2,), np.float32)
    def _callback(_):  # f32[2] -> f32[2, 0]
      return result

    @jax.jit
    def f(x):
      return callback(
          _callback, core.ShapedArray(result.shape, result.dtype), x)
    jax.effects_barrier()
    self.assertAllClose(f(x), result)

  @with_pure_and_io_callbacks
  def test_recv_zero_dim_and_non_zero_dim_arrays(self, *, callback):
    x = np.full((2,), 42., dtype=np.float32)
    result0 = np.ones((2, 0), dtype=np.float32)
    result1 = x
    result2 = np.ones((3, 0), dtype=np.int32)
    result3 = np.concatenate([x, x]) + 1.
    def _callback(x):  # x: f32[2] -> (f32[2, 0], f32[2], f32[3, 0], f32[4])
      return (result0, x, result2, np.concatenate([x, x]) + 1.)

    @jax.jit
    def f(x):
      return callback(
          _callback, (core.ShapedArray(result0.shape, result0.dtype),
                      core.ShapedArray(result1.shape, result1.dtype),
                      core.ShapedArray(result2.shape, result2.dtype),
                      core.ShapedArray(result3.shape, result3.dtype)), x)
    res = f(x)
    jax.effects_barrier()
    self.assertAllClose(res, (result0, result1, result2, result3))

  @with_pure_and_io_callbacks
  def test_callback_with_pytree_arguments_and_return_values(self, *, callback):

    def _callback(x):
      return dict(y=[x])

    @jax.jit
    def f(x):
      return callback(_callback, dict(y=[core.ShapedArray((), np.float32)]),
                      [x])

    out = f(jnp.float32(2.))
    jax.effects_barrier()
    self.assertEqual(out, dict(y=[2.]))

  @with_pure_and_io_callbacks
  def test_callback_inside_of_while_loop_of_scalars(self, *, callback):

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.jit
    def f(x):
      def cond(x):
        return x < 10
      def body(x):
        return callback(_callback, core.ShapedArray((), x.dtype), x)
      return lax.while_loop(cond, body, x)

    out = f(0.)
    jax.effects_barrier()
    self.assertEqual(out, 10.)

  @with_pure_and_io_callbacks
  def test_callback_inside_of_while_loop(self, *, callback):

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.jit
    def f(x):

      def cond(x):
        return jnp.any(x < 10)

      def body(x):
        return callback(_callback, core.ShapedArray(x.shape, x.dtype), x)

      return lax.while_loop(cond, body, x)

    out = f(jnp.arange(5.))
    jax.effects_barrier()
    np.testing.assert_allclose(out, jnp.arange(10., 15.))

  @with_pure_and_io_callbacks
  def test_callback_inside_of_cond_of_scalars(self, *, callback):

    def _callback1(x):
      return (x + 1.).astype(x.dtype)

    def _callback2(x):
      return (x - 1.).astype(x.dtype)

    @jax.jit
    def f(pred, x):

      def true_fun(x):
        return callback(_callback1, core.ShapedArray((), x.dtype), x)

      def false_fun(x):
        return callback(_callback2, core.ShapedArray((), x.dtype), x)

      return lax.cond(pred, true_fun, false_fun, x)

    out = f(True, 1.)
    jax.effects_barrier()
    self.assertEqual(out, 2.)
    out = f(False, 1.)
    jax.effects_barrier()
    self.assertEqual(out, 0.)

  @with_pure_and_io_callbacks
  def test_callback_inside_of_cond(self, *, callback):

    def _callback1(x):
      return x + 1.

    def _callback2(x):
      return x - 1.

    @jax.jit
    def f(pred, x):

      def true_fun(x):
        return callback(_callback1, core.ShapedArray(x.shape, x.dtype), x)

      def false_fun(x):
        return callback(_callback2, core.ShapedArray(x.shape, x.dtype), x)

      return lax.cond(pred, true_fun, false_fun, x)

    out = f(True, jnp.ones(2))
    jax.effects_barrier()
    np.testing.assert_allclose(out, jnp.ones(2) * 2.)
    out = f(False, jnp.ones(2))
    jax.effects_barrier()
    np.testing.assert_allclose(out, jnp.zeros(2))

  @with_pure_and_io_callbacks
  def test_callback_inside_of_scan_of_scalars(self, *, callback):

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.jit
    def f(x):

      def body(x, _):
        x = callback(_callback, core.ShapedArray(x.shape, x.dtype), x)
        return x, ()

      return lax.scan(body, x, jnp.arange(10))[0]

    out = f(0.)
    jax.effects_barrier()
    self.assertEqual(out, 10.)

  @with_pure_and_io_callbacks
  def test_callback_inside_of_scan(self, *, callback):

    def _callback(x):
      return x + 1.

    @jax.jit
    def f(x):

      def body(x, _):
        x = callback(_callback, core.ShapedArray(x.shape, x.dtype), x)
        return x, ()

      return lax.scan(body, x, jnp.arange(10))[0]

    out = f(jnp.arange(2.))
    jax.effects_barrier()
    np.testing.assert_allclose(out, jnp.arange(2.) + 10.)

  @with_pure_and_io_callbacks
  def test_callback_inside_of_pmap_of_scalars(self, *, callback):
    if callback is io_callback_ordered:
      self.skipTest("N/A")

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.pmap
    def f(x):
      return callback(_callback, core.ShapedArray(x.shape, x.dtype), x)

    out = f(jnp.arange(jax.local_device_count(), dtype=jnp.float32))
    jax.effects_barrier()
    np.testing.assert_allclose(
        out, np.arange(jax.local_device_count(), dtype=np.float32) + 1.)

  @with_pure_and_io_callbacks
  def test_callback_inside_of_pmap(self, *, callback):
    if callback is io_callback_ordered:
      self.skipTest("N/A")

    def _callback(x):
      return x + 1.

    @jax.pmap
    def f(x):
      return callback(_callback, core.ShapedArray(x.shape, x.dtype), x)

    out = f(
        jnp.arange(2 * jax.local_device_count(),
                   dtype=jnp.float32).reshape([-1, 2]))
    jax.effects_barrier()
    np.testing.assert_allclose(
        out,
        np.arange(2 * jax.local_device_count()).reshape([-1, 2]) + 1.)

  @with_pure_and_io_callbacks
  def test_exception_in_callback(self, *, callback):
    def fail(x):
      raise RuntimeError("Ooops")

    @jax.jit
    def f(x):
      return callback(fail, core.ShapedArray(x.shape, x.dtype), x)

    with self.assertLogs(level="ERROR") as l:
      try:
        f(0.0).block_until_ready()
      except RuntimeError:
        pass

      api_name = (
          "pure_callback" if callback is jax.pure_callback else "io_callback"
      )
      output = "\n".join(l.output)
      self.assertIn(f"jax.{api_name} failed", output)
      self.assertIn("Traceback (most recent call last)", output)


class PureCallbackTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu", "tpu"]):
      self.skipTest(f"Host callback not supported on {jtu.device_under_test()}")

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  def test_pure_callback_passes_ndarrays_without_jit(self):

    def cb(x):
      self.assertIs(type(x), np.ndarray)
      return x

    def f(x):
      return jax.pure_callback(cb, x, x)
    f(jnp.array(2.))

  def test_can_dce_pure_callback(self):

    if jax.default_backend() == "tpu":
      raise unittest.SkipTest("DCE doesn't currently happen on TPU")

    log = []
    def _callback(x):
      # Should never happen!
      log.append("hello world")
      return (x * 2.).astype(x.dtype)

    @jax.jit
    def f(x):
      _ = jax.pure_callback(_callback, x, x)
      return x * 2.
    _ = f(2.)
    self.assertEmpty(log)

  def test_can_vmap_pure_callback(self):

    @jax.jit
    @jax.vmap
    def f(x):
      return jax.pure_callback(np.sin, x, x)
    out = f(jnp.arange(4.))
    np.testing.assert_allclose(out, np.sin(np.arange(4.)))

    @jax.jit
    def g(x):
      return jax.pure_callback(np.sin, x, x)
    out = jax.vmap(g, in_axes=1)(jnp.arange(8.).reshape((4, 2)))
    np.testing.assert_allclose(out, np.sin(np.arange(8.).reshape((4, 2))).T)

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(0, None))
    def h(x, y):
      out_shape = jax.ShapeDtypeStruct(x.shape, np.result_type(x.dtype, y.dtype))
      return jax.pure_callback(lambda x, y: np.sin(x) + y, out_shape, x, y)
    out = h(jnp.arange(4.), 4.)
    self.assertArraysAllClose(out, np.sin(np.arange(4.)) + 4.,
                              rtol=1E-7, check_dtypes=False)

    @jax.jit
    @functools.partial(jax.vmap)
    def h(x, y):
      out_shape = jax.ShapeDtypeStruct(x.shape, np.result_type(x.dtype, y.dtype))
      return jax.pure_callback(lambda x, y: np.sin(x) + y, out_shape, x, y)
    out = h(jnp.arange(4.), jnp.arange(10., 14.))
    self.assertArraysAllClose(out, np.sin(np.arange(4.)) + np.arange(10., 14.),
                              rtol=1E-7, check_dtypes=False)

    @jax.jit
    @functools.partial(jax.vmap, in_axes=1, out_axes=1)
    def h(x, y):
      out_shape = jax.ShapeDtypeStruct(x.shape, np.result_type(x.dtype, y.dtype))
      return jax.pure_callback(lambda x, y: np.sin(x) + y, out_shape, x, y)
    out = h(jnp.arange(4.)[None], jnp.arange(10., 14.)[None])
    self.assertArraysAllClose(out, np.sin(np.arange(4.)) + np.arange(10.,
      14.)[None],
                              rtol=1E-7, check_dtypes=False)

  def test_vmap_vectorized_callback(self):

    def cb(x):
      self.assertTupleEqual(x.shape, ())
      return np.sin(x)

    @jax.jit
    @jax.vmap
    def f(x):
      return jax.pure_callback(cb, x, x)

    np.testing.assert_allclose(f(jnp.arange(4.)), np.sin(np.arange(4.)))

    def cb2(x):
      self.assertTupleEqual(x.shape, (4,))
      return np.sin(x)

    @jax.jit
    @jax.vmap
    def g(x):
      return jax.pure_callback(cb2, x, x, vectorized=True)

    np.testing.assert_allclose(g(jnp.arange(4.)), np.sin(np.arange(4.)))

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(0, None))
    def h(x, y):
      return jax.pure_callback(lambda x, y: np.sin(x) + y, x, x, y,
                               vectorized=True)
    out = h(jnp.arange(4.), 4.)
    np.testing.assert_allclose(out, np.sin(np.arange(4.)) + 4.)

    @jax.jit
    @functools.partial(jax.vmap, in_axes=(1, None), out_axes=1)
    def h(x, y):
      return jax.pure_callback(lambda x, y: np.sin(x) + y, x, x, y,
                               vectorized=True)
    out = h(jnp.arange(4.)[None], 4.)
    np.testing.assert_allclose(out, np.sin(np.arange(4.)[None]) + 4.)

  def test_vmap_vectorized_callback_errors_if_returns_wrong_shape(self):

    def cb(x):
      # Reduces over all dimension when it shouldn't
      return np.sin(x).sum()

    @jax.jit
    @jax.vmap
    def f(x):
      return jax.pure_callback(cb, x, x, vectorized=True)

    with self.assertRaises(RuntimeError):
      f(jnp.arange(4.))
      jax.effects_barrier()

  def test_can_pmap_pure_callback(self):

    @jax.pmap
    def f(x):
      return jax.pure_callback(np.sin, x, x)
    out = f(jnp.arange(float(jax.local_device_count())))
    np.testing.assert_allclose(out, np.sin(np.arange(jax.local_device_count())))

  def test_can_pjit_pure_callback_under_hard_xmap(self):

    if not hasattr(xla_client.OpSharding.Type, 'MANUAL'):
      raise unittest.SkipTest('Manual partitioning needed for pure_callback')

    spmd_lowering = maps.SPMD_LOWERING.value
    spmd_manual_lowering = maps.SPMD_LOWERING_MANUAL.value
    config.update('experimental_xmap_spmd_lowering', True)
    config.update('experimental_xmap_spmd_lowering_manual', True)
    try:
      mesh = Mesh(np.array(jax.devices()), axis_names=('x',))

      spec = jax.sharding.PartitionSpec('x')

      def f(x):
        axis_resources = {v: v for v in mesh.axis_names}
        return xmap(
            lambda x: jax.pure_callback(np.sin, x, x),
            in_axes=(('x',),),
            out_axes=('x',),
            axis_resources=axis_resources,
            axis_sizes=mesh.shape,
        )(x)

      def without_xmap_f(x):
        return jax.pure_callback(np.sin, x, x)

      with mesh:
        inp = jnp.arange(float(jax.local_device_count()))
        out = pjit.pjit(f, in_shardings=spec, out_shardings=spec)(inp)
        np.testing.assert_allclose(
            out, np.sin(np.arange(jax.local_device_count()))
        )
    finally:
      config.update('experimental_xmap_spmd_lowering', spmd_lowering)
      config.update(
        'experimental_xmap_spmd_lowering_manual',
        spmd_manual_lowering,
      )

  def test_cant_take_grad_of_pure_callback(self):

    def sin(x):
      return np.sin(x)

    @jax.jit
    @jax.grad
    def f(x):
      return jax.pure_callback(sin, x, x)
    with self.assertRaisesRegex(
        ValueError, "Pure callbacks do not support JVP."):
      f(2.)

  @unittest.skipIf(xla_extension_version < 245, "jaxlib version too old")
  def test_error_propagation(self):
    def throws_error_fn(x):
      raise RuntimeError("Errors should propagate.")

    @jax.jit
    def f(x):
      return jax.pure_callback(throws_error_fn, x, x)

    with self.assertRaisesRegex(Exception, "Errors should propagate."):
      print(np.array(f(2.0)), flush=True)

  def test_can_take_grad_of_pure_callback_with_custom_jvp(self):

    @jax.custom_jvp
    def sin(x):
      return jax.pure_callback(np.sin, x, x)

    @sin.defjvp
    def sin_jvp(xs, ts):
      (x,), (t,), = xs, ts
      return sin(x), jax.pure_callback(np.cos, x, x) * t

    @jax.jit
    @jax.grad
    def f(x):
      return sin(x)
    out = f(2.)
    np.testing.assert_allclose(out, jnp.cos(2.))

  def test_callback_inside_of_cond(self):

    def _callback1(x):
      return x + 1.

    def _callback2(x):
      return x - 1.

    @jax.jit
    def f(pred, x):

      def true_fun(x):
        return jax.pure_callback(_callback1, x, x)

      def false_fun(x):
        return jax.pure_callback(_callback2, x, x)

      return lax.cond(pred, true_fun, false_fun, x)

    out = f(True, jnp.ones(2))
    np.testing.assert_allclose(out, jnp.ones(2) * 2.)
    out = f(False, jnp.ones(2))
    np.testing.assert_allclose(out, jnp.zeros(2))

  def test_callback_inside_of_scan(self):

    def _callback(x):
      return x + 1.

    @jax.jit
    def f(x):

      def body(x, _):
        x = jax.pure_callback(_callback, x, x)
        return x, ()

      return lax.scan(body, x, jnp.arange(10))[0]

    out = f(jnp.arange(2.))
    np.testing.assert_allclose(out, jnp.arange(2.) + 10.)

  def test_callback_inside_of_while_loop(self):

    def _cond_callback(x):
      return np.any(x < 10)

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    @jax.jit
    def f(x):

      def cond(x):
        return jax.pure_callback(
            _cond_callback, jax.ShapeDtypeStruct((), np.bool_), x)

      def body(x):
        return jax.pure_callback(_callback, x, x)

      return lax.while_loop(cond, body, x)

    out = f(jnp.arange(5.))
    np.testing.assert_allclose(out, jnp.arange(10., 15.))

  def test_callback_inside_of_pmap(self):

    def _callback(x):
      return x + 1.

    @jax.pmap
    def f(x):
      return jax.pure_callback(_callback, x, x)

    out = f(
        jnp.arange(2 * jax.local_device_count(),
                   dtype=jnp.float32).reshape([-1, 2]))
    np.testing.assert_allclose(
        out,
        np.arange(2 * jax.local_device_count()).reshape([-1, 2]) + 1.)

  def test_callback_with_immediate_executable_destruction(self):

    def loop_body(i, x):
      del i
      return jax.pure_callback(lambda y: y + np.ones(4, np.float32),
                               x, x)

    class AClass:
      def f(self, ys):
        return lax.fori_loop(0, 10, loop_body, jnp.ones(4, np.float32))

    num_devices = jax.local_device_count()
    c = AClass()
    out = jax.pmap(c.f)(np.ones((num_devices,), np.float32))
    # c.f is an ephemeral bound method object, and it will be destroyed
    # immediately. This test verifies that the execution itself keeps the
    # callback alive.
    np.testing.assert_allclose(out, np.full((num_devices, 4), 11, np.float32))

  def test_callback_inside_xmap(self):

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    def f(x):
      return jax.pure_callback(_callback, x, x)

    f = maps.xmap(f, in_axes=['a'], out_axes=['a'],
                  axis_resources={'a': 'dev'})
    with jax.sharding.Mesh(np.array(jax.devices()), ['dev']):
      out = f(np.arange(40.))
    np.testing.assert_allclose(out, jnp.arange(1., 41.))

  def test_vectorized_callback_inside_xmap(self):

    def _callback(x):
      return (x + 1.).astype(x.dtype)

    def f(x):
      return jax.pure_callback(_callback, x, x, vectorized=True)

    f = maps.xmap(f, in_axes=['a'], out_axes=['a'],
                  axis_resources={'a': 'dev'})
    with jax.sharding.Mesh(np.array(jax.devices()), ['dev']):
      out = f(np.arange(40.))
    np.testing.assert_allclose(out, jnp.arange(1., 41.))

  def test_array_layout_is_preserved(self):

    def g(x):
      return jax.pure_callback(lambda x: x, x, x)

    x = np.arange(6, dtype=np.int32).reshape((3, 2))
    np.testing.assert_allclose(g(x), x)

  def test_can_shard_pure_callback_maximally(self):
    mesh = Mesh(np.array(jax.devices()), axis_names=('x',))

    spec = jax.sharding.PartitionSpec('x')
    sharding = jax.sharding.NamedSharding(mesh, spec)

    def func(x):
      return x + np.arange(x.shape[0], dtype=x.dtype)

    def f(x):
      return jax.pure_callback(func, x, x)

    inp = jnp.arange(float(jax.local_device_count()))
    out = jax.jit(f, in_shardings=sharding, out_shardings=sharding)(inp)
    jax.block_until_ready(out)
    np.testing.assert_allclose(
        out, np.arange(jax.local_device_count()) * 2
    )

  def test_can_shard_pure_callback_maximally_with_sharding(self):
    mesh = Mesh(np.array(jax.devices()), axis_names=('x',))

    spec = jax.sharding.PartitionSpec('x')
    sharding = jax.sharding.NamedSharding(mesh, spec)

    def func(x):
      return x + np.arange(x.shape[0], dtype=x.dtype)

    callback_device = jax.devices()[-1]
    callback_device_index = sharding._device_assignment.index(callback_device)

    def f(x):
      sharding = jax.sharding.SingleDeviceSharding(callback_device)
      return jax.pure_callback(func, x, x, sharding=sharding)

    f_jit = jax.jit(f, in_shardings=sharding, out_shardings=sharding)

    inp = jnp.arange(float(jax.local_device_count()))
    out = f_jit(inp)
    jax.block_until_ready(out)
    np.testing.assert_allclose(
        out, np.arange(jax.local_device_count()) * 2
    )

    self.assertIn(
        f'{{maximal device={callback_device_index}}}',
        str(f_jit.lower(inp).compiler_ir(dialect='stablehlo')),
    )

  def test_can_shard_pure_callback_manually(self):

    mesh = Mesh(np.array(jax.devices()), axis_names=('x',))

    spec = jax.sharding.PartitionSpec('x')
    sharding = jax.sharding.NamedSharding(mesh, spec)

    def func(x):
      return x + np.arange(x.shape[0], dtype=x.dtype)

    def f(x):
      return jax.pure_callback(func, x, x)
    f = shard_map(f, mesh=mesh, in_specs=(spec,), out_specs=spec)

    inp = jnp.arange(float(jax.local_device_count() * 2))
    out = jax.jit(f, in_shardings=sharding, out_shardings=sharding)(inp)
    y = np.tile(np.arange(2, dtype=inp.dtype), jax.local_device_count())
    jax.block_until_ready(out)
    np.testing.assert_allclose(
        out, inp + y
    )


class IOCallbackTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu", "tpu"]):
      self.skipTest(f"Host callback not supported on {jtu.device_under_test()}")

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  def test_io_callback_can_mutate_state(self):
    x = 0
    def cb():
      nonlocal x
      x += 1
      return np.array(x, np.int32)

    def f():
      return io_callback(cb, jax.ShapeDtypeStruct((), jnp.int32))
    f()
    jax.effects_barrier()
    self.assertEqual(x, 1)
    f()
    jax.effects_barrier()
    self.assertEqual(x, 2)

  def test_io_callback_can_be_batched_if_unordered(self):
    _mut = 0
    def cb(x):
      nonlocal _mut
      _mut += 1
      return x

    x = jnp.arange(4)
    def f(x):
      return io_callback(cb, jax.ShapeDtypeStruct((), x.dtype), x)
    jax.vmap(f)(x)
    jax.effects_barrier()
    self.assertEqual(_mut, 4)
    jax.vmap(f)(x)
    jax.effects_barrier()
    self.assertEqual(_mut, 8)

  def test_cannot_call_ordered_io_in_pmap(self):
    def f(x):
      return io_callback(
          lambda x: x, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=True)
    with self.assertRaisesRegex(
        ValueError, "Ordered effects not supported in `pmap`"):
      jax.pmap(f)(jnp.arange(jax.local_device_count()))

  def test_cannot_call_ordered_io_in_xmap(self):
    def f(x):
      return io_callback(
          lambda x: x, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=True)
    with self.assertRaisesRegex(
        ValueError, "Cannot `vmap` ordered IO callback"):
      maps.xmap(f, in_axes=([0],), out_axes=[0])(jnp.arange(16))

  def test_cannot_call_ordered_io_in_vmap(self):
    def f(x):
      return io_callback(
          lambda x: x, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=True)
    with self.assertRaisesRegex(
        ValueError, "Cannot `vmap` ordered IO callback"):
      jax.vmap(f)(jnp.arange(4))

  def test_cannot_use_io_callback_in_jvp(self):
    def f(x):
      return io_callback(lambda x: x, jax.ShapeDtypeStruct((), jnp.float32), x)
    with self.assertRaisesRegex(
        ValueError, "IO callbacks do not support JVP."):
      jax.jvp(f, (0.,), (1.,))

  def test_cannot_use_io_callback_in_linearize(self):
    def f(x):
      return io_callback(lambda x: x, jax.ShapeDtypeStruct((), jnp.float32), x)
    with self.assertRaisesRegex(
        ValueError, "IO callbacks do not support JVP."):
      jax.linearize(f, 0.)

  def test_cannot_use_io_callback_in_transpose(self):
    x = jnp.array(1.)

    def f(x):
      return io_callback(lambda x: x, jax.ShapeDtypeStruct((), x.dtype), x)
    with self.assertRaisesRegex(
        ValueError, "IO callbacks do not support transpose."):
      jax.linear_transpose(f, x)(x)

  def test_cannot_vmap_of_cond_io_callback(self):
    def f(pred):
      def true_fun():
        io_callback(lambda: print("true"), None)
      def false_fun():
        io_callback(lambda: print("false"), None)
      return lax.cond(pred, false_fun, true_fun)
    with self.assertRaisesRegex(NotImplementedError,
        "IO effect not supported in vmap-of-cond."):
      jax.vmap(f)(jnp.array([True, True]))

  def test_cannot_vmap_of_while_io_callback(self):
    def check(x):
      assert np.all(x < 5)

    def f(i):
      def cond(i):
        return i < 5
      def body(i):
        io_callback(check, None, i)
        return i + 1
      return lax.while_loop(cond, body, i)
    with self.assertRaisesRegex(NotImplementedError,
        "IO effect not supported in vmap-of-while."):
      jax.vmap(f)(jnp.array([0, 4]))

  def test_cannot_use_io_callback_in_checkpoint(self):
    @jax.grad
    @jax.checkpoint
    def f(x, y):
      io_callback(lambda x: x, y, y)
      return x

    with self.assertRaisesRegex(NotImplementedError,
        "Effects not supported in partial-eval of `checkpoint`"):
      f(2., 3.)

  @parameterized.named_parameters(
      dict(
          testcase_name=f'{ordered=}_{with_sharding=}',
          ordered=ordered,
          with_sharding=with_sharding,
      )
      for ordered in [True, False]
      for with_sharding in [True, False]
  )
  def test_can_use_io_callback_in_pjit(
      self, *, ordered: bool, with_sharding: bool
  ):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), ['dev'])

    _collected: list[int] = []
    def _cb(x):
      nonlocal _collected
      _collected.append(int(x.sum()))

    io_callback_kwargs = dict(ordered=ordered)
    callback_device = devices[0]
    if with_sharding:
      callback_device = devices[-1]
      io_callback_kwargs['sharding'] = jax.sharding.SingleDeviceSharding(
          callback_device
      )

    def f(x):
      io_callback(_cb, None, x, **io_callback_kwargs)
      io_callback(_cb, None, x + 1, **io_callback_kwargs)
      return x

    in_spec = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('dev')
    )
    out_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    f = pjit.pjit(f, in_shardings=in_spec, out_shardings=out_spec)
    expected = []
    with mesh:
      x = jnp.arange(mesh.size)
      f(x)
      expected.extend([int(x.sum()), int((x + 1).sum())])
      f(x + 5)
      expected.extend([int((x + 5).sum()), int((x + 6).sum())])

    jax.effects_barrier()
    if ordered:
      self.assertAllClose(_collected, expected)
    else:
      self.assertEqual(len(_collected), len(expected))
      for v in expected:
        self.assertIn(v, _collected)

    callback_device_index = in_spec._device_assignment.index(callback_device)
    self.assertIn(
        f'{{maximal device={callback_device_index}}}',
        str(f.lower(x).compiler_ir(dialect='stablehlo')),
    )

  def test_sequence_pjit_io_callback_ordered(self):
    # A sequence of pairs of calls to pjit(io_callback(ordered=True)) with each
    # pair on a different device assignment.
    _collected: list[int] = []
    def _cb(i, x):
      nonlocal _collected
      # Sleep different amounts of time, to test the ordering.
      time.sleep([0.02, 0.03, 0.04][len(_collected) % 3])
      logging.info('Collected iteration %s: %s', i, x)
      _collected.append(int(x.sum()))

    def f_base(i, x):
      io_callback(_cb, None, i, x, ordered=True)
      io_callback(_cb, None, i, x + 1, ordered=True)

    nr_iterations = 8
    # TODO(zce): If I pin to 1 device below (jax.devices()[:1]) then this test
    # flakes. It also flakes when pinned to 2 devices. It seems that repeatedly
    # dispatching to the same device triggers the problem.
    devices = jax.devices()
    expected = []  # The expected value for _collected
    for i in range(nr_iterations):
      if len(devices) > 1:
        devices_for_iteration = [
            devices[i % len(devices)],
            devices[(i + 1) % len(devices)],
        ]
      else:
        devices_for_iteration = devices
      logging.info(
          'Running iteration %d on devices %s', i, devices_for_iteration
      )
      mesh = jax.sharding.Mesh(np.array(devices_for_iteration), ['dev'])
      in_spec = (
          jax.sharding.NamedSharding(mesh, None),
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dev')),
      )
      out_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      f = pjit.pjit(f_base, in_shardings=in_spec, out_shardings=out_spec)
      with mesh:
        x = jax.device_put(
            np.arange(len(devices_for_iteration), dtype=np.int32) + 10 * i,
            in_spec[1],
        )
        f(i, x)
        expected.extend([int(x.sum()), int((x + 1).sum())])
        f(i, x + 5)
        expected.extend([int((x + 5).sum()), int((x + 6).sum())])

    jax.effects_barrier()
    self.assertEqual(_collected, expected)

  def test_can_shard_io_callback_manually(self):

    mesh = Mesh(np.array(jax.devices()), axis_names=('x',))

    spec = jax.sharding.PartitionSpec('x')
    sharding = jax.sharding.NamedSharding(mesh, spec)

    _collected = collections.defaultdict(list)

    def func(shard_id, x):
      nonlocal _collected
      _collected[shard_id.item()].append(x)

    def f(shard_ids, x):
      io_callback(func, None, shard_ids, x, ordered=True)
      io_callback(func, None, shard_ids, x + 1, ordered=True)
    f = shard_map(f, mesh=mesh, in_specs=spec, out_specs=None)

    shard_ids = jnp.arange(mesh.devices.size)
    inp = jnp.arange(2 * jax.local_device_count())
    jax.jit(f, in_shardings=sharding, out_shardings=None)(shard_ids, inp)
    jax.effects_barrier()

    self.assertLen(_collected, mesh.devices.size)
    # Verify the partial ordering: no specified order across shards, but strict
    # ordering between the two calls in each shard.
    for shard in _collected.values():
      self.assertLen(shard, 2)
      np.testing.assert_array_equal(shard[0] + 1, shard[1])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
