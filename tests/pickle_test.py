# Copyright 2021 The JAX Authors.
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
"""Tests for interoperability between JAX and pickling libraries."""

import pickle
import unittest

from absl.testing import absltest
from absl.testing import parameterized

try:
  import cloudpickle
except ImportError:
  cloudpickle = None

import jax
from jax import numpy as jnp
from jax.interpreters import pxla
from jax._src import test_util as jtu
from jax._src.lib import xla_client as xc

import numpy as np

jax.config.parse_flags_with_absl()


def _get_device_by_id(device_id: int) -> xc.Device:
  for device in jax.devices():
    if device.id == device_id:
      return device
  raise ValueError(f'Device {device_id} was not found')


xc.Device.__reduce__ = lambda d: (_get_device_by_id, (d.id,))


if cloudpickle is not None:
  def _reduce_mesh(mesh):
    # Avoid including mesh._hash in the serialized bytes for Mesh. Without this
    # the Mesh would be different among the workers.
    return jax.sharding.Mesh, (mesh.devices, mesh.axis_names)

  cloudpickle.CloudPickler.dispatch_table[jax.sharding.Mesh] = _reduce_mesh


class CloudpickleTest(jtu.JaxTestCase):

  @unittest.skipIf(cloudpickle is None, "Requires cloudpickle")
  def testPickleOfJittedFunctions(self):

    @jax.jit
    def f(x, y):
      return x * y

    @jax.jit
    def g(z):
      return f(z, z + 77)  # noqa: F821

    expected = g(32)
    s = cloudpickle.dumps(g)
    del f, g

    g_unpickled = pickle.loads(s)
    actual = g_unpickled(32)
    self.assertEqual(expected, actual)

  @unittest.skipIf(cloudpickle is None, "Requires cloudpickle")
  def testPickleOfPmappedFunctions(self):

    @jax.pmap
    def f(x, y):
      return x * y

    @jax.pmap
    def g(z):
      return f(z, z + 77)  # noqa: F821

    expected = g(jnp.asarray([[32]]))
    s = cloudpickle.dumps(g)
    del f, g

    g_unpickled = pickle.loads(s)
    actual = g_unpickled(jnp.asarray([[32]]))
    self.assertEqual(expected, actual)


class PickleTest(jtu.JaxTestCase):

  def testPickleOfArray(self):
    x = jnp.arange(10.0)
    s = pickle.dumps(x)
    y = pickle.loads(s)
    self.assertArraysEqual(x, y)
    self.assertIsInstance(y, type(x))
    self.assertEqual(x.aval, y.aval)

  def testPickleOfArrayWeakType(self):
    x = jnp.array(4.0)
    self.assertEqual(x.aval.weak_type, True)
    s = pickle.dumps(x)
    y = pickle.loads(s)
    self.assertArraysEqual(x, y)
    self.assertIsInstance(y, type(x))
    self.assertEqual(x.aval, y.aval)

  @jtu.sample_product(prng_name=['threefry2x32', 'rbg', 'unsafe_rbg'])
  def testPickleOfKeyArray(self, prng_name):
    with jax.default_prng_impl(prng_name):
      k1 = jax.random.PRNGKey(72)
      s  = pickle.dumps(k1)
      k2 = pickle.loads(s)
      self.assertEqual(k1.dtype, k2.dtype)
      with jax.legacy_prng_key('allow'):
        self.assertArraysEqual(jax.random.key_data(k1),
                              jax.random.key_data(k2))

  @parameterized.parameters(
      (jax.sharding.PartitionSpec(),),
      (jax.sharding.PartitionSpec(None),),
      (jax.sharding.PartitionSpec('x', None),),
      (jax.sharding.PartitionSpec(None, 'y'),),
      (jax.sharding.PartitionSpec('x', 'y'),),
      (jax.sharding.PartitionSpec(('x', 'y'),),),
  )
  def testPickleOfPartitionSpecs(self, partition_spec):
    restored_partition_spec = pickle.loads(pickle.dumps(partition_spec))
    self.assertIsInstance(restored_partition_spec, jax.sharding.PartitionSpec)
    self.assertTupleEqual(partition_spec, restored_partition_spec)

  def testPickleX64(self):
    with jax.experimental.enable_x64():
      x = jnp.array(4.0, dtype='float64')
      s = pickle.dumps(x)

    with jax.experimental.disable_x64():
      y = pickle.loads(s)

    self.assertEqual(x.dtype, jnp.float64)
    self.assertArraysEqual(x, y, check_dtypes=False)
    self.assertEqual(y.dtype, jnp.float32)
    self.assertEqual(y.aval.dtype, jnp.float32)
    self.assertIsInstance(y, type(x))

  def testPickleTracerError(self):
    with self.assertRaises(jax.errors.ConcretizationTypeError):
      jax.jit(pickle.dumps)(0)

  def testPickleSharding(self):
    sharding = pxla.ShardingSpec((pxla.NoSharding(), pxla.Chunked(
        (2, 2)), pxla.Unstacked(3)), (pxla.ShardedAxis(0), pxla.ShardedAxis(1),
                                      pxla.ShardedAxis(2), pxla.Replicated(4)))
    self.assertEqual(pickle.loads(pickle.dumps(sharding)), sharding)

  def testPickleOpSharding(self):
    op = xc.OpSharding()
    op.type = xc.OpSharding.Type.OTHER
    op.tile_assignment_dimensions = [4, 2]
    op.tile_assignment_devices = [0, 1, 2, 3, 4, 5, 6, 7]
    self.assertTrue(
        xc.HloSharding.from_proto(pickle.loads(pickle.dumps(op))),
        xc.HloSharding.from_proto(op))

  def test_pickle_single_device_sharding(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  def test_pickle_pmap_sharding(self):
    ss = pxla.ShardingSpec(
        sharding=(pxla.Unstacked(8),),
        mesh_mapping=(pxla.ShardedAxis(0),))
    s = jax.sharding.PmapSharding(jax.devices(), ss)
    self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  def test_pickle_gspmd_sharding(self):
    s = jax.sharding.GSPMDSharding.get_replicated(jax.devices())
    self.assertEqual(s, pickle.loads(pickle.dumps(s)))

  @unittest.skipIf(cloudpickle is None, "Requires cloudpickle")
  def test_pickle_named_sharding(self):
    s = jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(np.array(jax.devices()), 'd'),
        spec=jax.sharding.PartitionSpec('d'))
    self.assertEqual(s, pickle.loads(pickle.dumps(s)))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
