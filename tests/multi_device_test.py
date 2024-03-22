# Copyright 2019 The JAX Authors.
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


import os
from unittest import SkipTest
import tracemalloc as tm

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src import test_util as jtu
from jax._src import xla_bridge

from jax import config
config.parse_flags_with_absl()

prev_xla_flags = None


# Run all tests with 8 CPU devices.
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class MultiDeviceTest(jtu.JaxTestCase):

  def get_devices(self):
    if len(jax.devices()) < 2:
      raise SkipTest("test requires multiple devices")
    return jax.devices()

  def assert_committed_to_device(self, data, device):
    """Asserts that the data is committed to the device."""
    self.assertTrue(data._committed)
    self.assertEqual(data.devices(), {device})

  def assert_uncommitted_to_device(self, data, device):
    """Asserts that the data is on the device but not committed to it."""
    self.assertFalse(data._committed)
    self.assertEqual(data.devices(), {device})

  def test_computation_follows_data(self):
    if jax.device_count() < 5:
      self.skipTest("test requires 5 devices")
    devices = self.get_devices()

    # By default, computation is placed (uncommitted) on device 0
    x = jnp.ones(2)
    self.assert_uncommitted_to_device(x, devices[0])

    # Computing only with uncommitted data, will perform the computation
    # on the same device 0, and the result is uncommitted
    y = jnp.sin(x + x * 1) + 2
    self.assert_uncommitted_to_device(y, devices[0])

    # We can use device_put to both place the data on the requested device,
    # and to commit it there.
    z = jax.device_put(1, devices[1])
    self.assert_committed_to_device(z, devices[1])

    # A computation with some committed inputs will happen on the committed
    # device. Uncommitted data may be moved to the committed device.
    u = z + x  # z is committed, and x is uncommitted
    self.assert_committed_to_device(u, devices[1])

    # A computation with inputs committed to multiple devices will result
    # in an error

    err_msg = "Received incompatible devices for jitted computation"
    with self.assertRaisesRegex(ValueError, err_msg):
      jax.device_put(x, devices[2]) + jax.device_put(x, devices[3])

    # A jitted-computation without a device specification behave like any
    # other primitive
    jit_add = jax.jit(lambda a, b: a + b)
    self.assert_uncommitted_to_device(jit_add(1, 2), devices[0])
    self.assert_committed_to_device(jit_add(1, jax.device_put(2, devices[2])),
                                    devices[2])
    with self.assertRaisesRegex(ValueError, err_msg):
      jit_add(jax.device_put(x, devices[2]), jax.device_put(x, devices[3]))

    # Even jit of trivial computations leaves the result uncommitted
    x_uncommitted = jnp.array([1, 2, 3])
    y = jax.jit(lambda x: x)(x_uncommitted)
    self.assert_uncommitted_to_device(y, devices[0])

    z1, z2 = jax.jit(lambda x: (x, x))(x_uncommitted)
    self.assert_uncommitted_to_device(z1, devices[0])
    self.assert_uncommitted_to_device(z2, devices[0])
    # trivial computation does not exist in JAX anymore.
    self.assertNotEqual(z1.unsafe_buffer_pointer(), z2.unsafe_buffer_pointer())

    x2_uncommitted = jnp.array([2, 3])
    z1, z2, z3 = jax.jit(lambda x, y: (y, 1, x))(x_uncommitted, x2_uncommitted)
    self.assert_uncommitted_to_device(z1, devices[0])
    self.assert_uncommitted_to_device(z2, devices[0])
    self.assert_uncommitted_to_device(z3, devices[0])


    # A jitted computation with an device specification behaves as if the
    # arguments are first device_put to the specified device. The result
    # will be committed on the specified.
    # The `device` parameter is experimental, and subject to change.
    jit_add_on4 = jax.jit(lambda a, b: a + b, device=devices[4])
    self.assert_committed_to_device(jit_add_on4(1, 2), devices[4])
    self.assert_committed_to_device(jit_add_on4(1, jax.device_put(2, devices[2])),
                                    devices[4])
    self.assert_committed_to_device(jit_add_on4(jax.device_put(x_uncommitted, devices[2]),
                                                jax.device_put(x_uncommitted, devices[3])),
                                    devices[4])

  @jax.legacy_prng_key('allow')
  def test_computation_follows_data_prng(self):
    _, device, *_ = self.get_devices()
    rng = jax.device_put(jax.random.PRNGKey(0), device)
    val = jax.random.normal(rng, ())
    self.assert_committed_to_device(val, device)

  def test_primitive_compilation_cache(self):
    devices = self.get_devices()

    x = jax.device_put(jnp.int32(1), devices[1])

    with jtu.count_primitive_compiles() as count:
      y = lax.add(x, x)
      z = lax.add(y, y)

    self.assertEqual(count[0], 1)
    self.assert_committed_to_device(y, devices[1])
    self.assert_committed_to_device(z, devices[1])

  def test_device_put(self):
    devices = self.get_devices()

    # test device_put on regular values
    x = jax.device_put(1, device=devices[0])
    self.assert_committed_to_device(x, devices[0])

    # test device_put on its own output
    y = jax.device_put(x, device=devices[1])
    self.assert_committed_to_device(y, devices[1])

    x = jax.device_put(jnp.zeros(2), device=devices[0])
    self.assert_committed_to_device(x, devices[0])

    y = jax.device_put(x, device=devices[1])
    self.assert_committed_to_device(y, devices[1])

    x = jax.device_put(jnp.zeros(2), device=devices[1])
    self.assert_committed_to_device(x, devices[1])

    # device_put with device=None does not change placement
    x = jax.device_put(jnp.zeros(2))
    self.assert_uncommitted_to_device(x, devices[0])

    x = jax.device_put(jax.device_put(2, device=devices[1]))
    self.assert_committed_to_device(x, devices[1])


  def test_closed_over_values_device_placement(self):
    # see https://github.com/google/jax/issues/1431
    devices = self.get_devices()

    def f(): return lax.add(3., 4.)
    self.assertIsInstance(f(), jax.Array)
    self.assert_uncommitted_to_device(f(), devices[0])
    self.assert_uncommitted_to_device(jax.jit(f)(), devices[0])
    self.assert_committed_to_device(jax.jit(f, device=devices[1])(),
                                    devices[1])

  def test_reshape(self):
    devices = self.get_devices()
    # computation follows data explicitly placed on device 1
    x = jax.device_put(1, devices[1])
    y = x.reshape((1, 1))
    self.assert_committed_to_device(y, devices[1])
    z = y.reshape((1, 1))
    self.assert_committed_to_device(z, devices[1])

  def test_broadcast(self):
    if jax.device_count() < 3:
      self.skipTest("test requires 3 devices")
    devices = self.get_devices()

    z = 1 + jnp.ones((2, 3))
    self.assert_uncommitted_to_device(z, devices[0])
    y = jax.device_put(1, devices[2]) + jnp.ones((2, 3))
    self.assert_committed_to_device(y, devices[2])

  def test_single_input_committed_multi_output(self):
    if jax.device_count() < 3:
      self.skipTest("Test requires 3 devices")
    devices = self.get_devices()

    @jax.jit
    def f(a, b, c, d, e):
      return a, b, c, d, e

    outs = f(jax.device_put(1, devices[2]), jnp.array(2), jnp.array(3),
             jnp.array(4), jnp.array(5))
    for o in outs:
      self.assert_committed_to_device(o, devices[2])

  def test_different_devices_input_error(self):
    if jax.device_count() < 2:
      self.skipTest("Test requires 2 devices")
    devices = self.get_devices()

    a = jax.device_put(1, devices[0])
    b = jax.device_put(2, devices[1])

    # Don't look for the message because the Array and non-Array path raise
    # slightly different error messages.
    with self.assertRaises(ValueError):
      _ = a + b

  def test_transpose(self):
    if jax.device_count() < 3:
      self.skipTest("test requires 3 devices")
    devices = self.get_devices()

    x = jnp.ones((2, 3))
    self.assert_uncommitted_to_device(x, devices[0])

    y = lax.transpose(x, (1, 0))
    self.assert_uncommitted_to_device(y, devices[0])
    z = lax.transpose(jax.device_put(x, devices[2]), (1, 0))
    self.assert_committed_to_device(z, devices[2])

  def test_device_put_committed_check(self):
    devices = self.get_devices()
    x = jnp.array(1.)
    y = jax.device_put(x, jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    self.assert_committed_to_device(y, devices[0])

  def test_grad_device_put_src_inference(self):
    devices = self.get_devices()
    x = jax.device_put(2.0, devices[0])
    y, x_bar = jax.value_and_grad(lambda x: jax.device_put(x, devices[1]))(x)
    self.assert_committed_to_device(y, devices[1])
    self.assert_committed_to_device(x_bar, devices[0])

  def test_lax_full_sharding(self):
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("i"))
    sharding = NamedSharding(mesh, P('i', None))
    x = lax.full((len(devices),), 1.0, sharding=sharding)
    self.assertEqual(x.sharding, sharding)

  def test_lax_full_like_sharding(self):
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("i"))
    sharding = NamedSharding(mesh, P('i'))
    x = lax.iota("float32", len(devices))
    y = lax.full_like(x, 1, sharding=sharding)
    self.assertEqual(y.sharding, sharding)

  def test_lax_full_like_same_device(self):
    devices = self.get_devices()
    x = jax.device_put(jnp.ones((100, 100)), devices[1])
    y = lax.full_like(x, 1)
    self.assertEqual(y.sharding, x.sharding)
    self.assertEqual(y.sharding.device_set, {jax.devices()[1]})


  def test_lax_full_like_jitted_preserves_sharding(self):
    devices = self.get_devices()
    x = jax.device_put(jnp.ones((100, 100)), devices[1])
    y = jax.jit(lax.full_like)(x, 1)
    self.assertEqual(y.sharding, x.sharding)
    self.assertEqual(y.sharding.device_set, {jax.devices()[1]})

  def test_lax_full_like_custom_shape_sharded(self):
    devices = [self.get_devices()]
    mesh = Mesh(devices, axis_names=('i', 'j'))
    sharding = NamedSharding(mesh, P('i', 'j'))
    x = jnp.array(jnp.arange(8).reshape((1, 8)), dtype=jnp.int32)
    x = jax.device_put(x, sharding)
    y = lax.full_like(x, fill_value=1.0, shape=())
    self.assertEqual(y.shape, ())

  def test_lax_full_like_single_device(self):
    devices = self.get_devices()
    x = jax.device_put(jnp.ones((100, 100)), devices[1])
    y = lax.full_like(x, fill_value=1.0, shape=())
    self.assertEqual(y.shape, ())
    # Currently if shape is provided the sharding will revert
    # to default. This might change in the future and this test might
    # need to be updated.
    self.assertEqual(
        y.sharding,
        jax.sharding.SingleDeviceSharding(jax.devices()[0]))


  def test_lax_full_like_efficient(self):
    devices = self.get_devices()
    if len(devices) < 4:
      self.skipTest("test requires 4 devices")
    mem_stats = devices[0].memory_stats()
    if mem_stats is None:
      self.skipTest('Only can run test on device with mem_stats')
    mesh = Mesh(devices, axis_names=("i"))
    sharding = NamedSharding(mesh, P('i'))
    available_memory = mem_stats['bytes_reservable_limit']
    array_size = available_memory // (6 * len(devices)) * len(devices)
    # Set up tracemalloc to track memory usage.
    tm.start()
    x = lax.full([array_size], sharding=sharding, fill_value=1.0,
                  dtype=jnp.float32)
    y = lax.full_like(x, fill_value=1.0, dtype=jnp.float32)

    # Wait until computation finished to ensure we are measuring the correct
    # thing.
    y.block_until_ready()
    unused_current, peak = tm.get_traced_memory()
    # Verify that we don't create large CPU arrays.
    self.assertLess(peak, array_size // len(devices))

    # Important: make sure that all jax computation in this part has finished
    # before we can stop trace_malloc.
    jax.effects_barrier()
    tm.stop()
    self.assertEqual(y.sharding, x.sharding)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
