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
"""Tests for serialization and deserialization of GDA."""

import asyncio
import math
from functools import partial
import os
import pathlib
import tracemalloc as tm

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax import config
from jax._src import array
from jax.sharding import NamedSharding, GSPMDSharding
from jax.sharding import PartitionSpec as P
from jax.experimental.array_serialization import serialization
import numpy as np
import tensorstore as ts
import unittest

config.parse_flags_with_absl()

prev_xla_flags = None

def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(8)

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()


class CheckpointTest(jtu.JaxTestCase):

  def _on_commit_callback(self, temp_ckpt_dir, final_ckpt_dir):
    os.rename(temp_ckpt_dir, final_ckpt_dir)

  @jtu.skip_on_devices('cpu')
  def test_memory_consumption(self):
    global_mesh = jtu.create_global_mesh((2, 4), ('x', 'y'))
    inp_shape = (2_048, 4_096)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)
    sharding = NamedSharding(global_mesh, pspec)
    src = jax.numpy.arange(num, dtype=np.int32).reshape(inp_shape)  # 8e9
    inp = array.make_array_from_callback(
        inp_shape, sharding,
        lambda idx: src[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('memprof').full_path)
    tspec = serialization.get_tensorstore_spec(str(ckpt_dir))

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [inp], [tspec],
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    async def deserialize_with_byte_limit():
      r = await serialization.async_deserialize(
        sharding, tspec, inp_shape,
        byte_limiter=serialization._LimitInFlightBytes(4_200_000))
      r.block_until_ready()

    tm.start()
    asyncio.run(deserialize_with_byte_limit())
    unused_current, peak = tm.get_traced_memory()
    # NB: some padding + tensorstore overhead. It should always be
    # less than array size (2048 * 4096 * 4 = 32M)
    self.assertLess(peak, 10_000_000)
    deserialize_wo_limit = serialization.async_deserialize(
        sharding, tspec, inp_shape)
    tm.clear_traces()
    # NB: call block_until_ready() is important here and above
    # because otherwise this leads to racing condition and segfault with
    # tensorstore attempting to dealloc using tracemalloc which is already
    # destroyed.
    asyncio.run(deserialize_wo_limit).block_until_ready()

    unused_current, peak = tm.get_traced_memory()
    # We load entire array in memory here.
    self.assertGreater(peak, 30_000_000)
    tm.stop()

  def test_checkpointing_with_path_variant(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt_variant').full_path)
    ckpt_path1 = pathlib.Path(
        self.create_tempdir(f'{ckpt_dir}/first').full_path)

    ckpt_paths = [str(ckpt_path1)]
    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize_with_paths(
        [a1], ckpt_paths,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    m1, = manager.deserialize_with_paths(
        [NamedSharding(global_mesh, pspec)], ckpt_paths)
    self.assertIsInstance(m1, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m1.addressable_shards[0].data),
                           np.array([[0], [2]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m1.addressable_shards[1].data),
                           np.array([[1], [3]], dtype=np.int32))
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

  def test_checkpointing_jax_array(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    ckpt_path1 = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/first').full_path)

    # Second Array
    global_input_data2 = np.arange(
        num, num + num, dtype=np.int32).reshape(inp_shape)
    a2 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data2[idx])
    ckpt_path2 = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/second').full_path)

    # Third Array
    def cb3(_):
      return np.array([], dtype=np.float32)
    global_mesh1d = jtu.create_global_mesh((8,), ('x',))
    a3 = array.make_array_from_callback(
        (0,), NamedSharding(global_mesh1d, P(None)), cb3)
    ckpt_path3 = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/third').full_path)

    ckpt_paths = [str(ckpt_path1), str(ckpt_path2), str(ckpt_path3)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [a1, a2, a3], tspecs,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    m1, m2, m3 = serialization.run_deserialization(
        [NamedSharding(global_mesh, pspec),
         NamedSharding(global_mesh, P('x')),
         NamedSharding(global_mesh1d, P(None))],
        tspecs)

    self.assertIsInstance(m1, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m1.addressable_shards[0].data),
                           np.array([[0], [2]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m1.addressable_shards[1].data),
                           np.array([[1], [3]], dtype=np.int32))
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

    self.assertIsInstance(m2, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m2.addressable_shards[0].data),
                           np.array([[16, 17], [18, 19]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m2.addressable_shards[1].data),
                           np.array([[16, 17], [18, 19]], dtype=np.int32))
    self.assertEqual(m2.addressable_shards[0].data.shape, (2, 2))
    self.assertEqual(m2.dtype, np.int32)

    self.assertIsInstance(m3, array.ArrayImpl)
    for i, s in enumerate(m3.addressable_shards):
      self.assertEqual(s.index, (slice(None),))
      self.assertEqual(s.replica_id, i)
      self.assertArraysEqual(np.asarray(s.data), np.array([], dtype=np.float32))
    self.assertEqual(m3.dtype, np.float32)

  def test_checkpointing_with_bigger_shape_jax_array(self):
    global_mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    num = math.prod(global_input_shape)

    global_input_data1 = np.arange(num, dtype=np.int32).reshape(global_input_shape)
    def cb1(index):
      return global_input_data1[index]
    arr = array.make_array_from_callback(
        global_input_shape, NamedSharding(global_mesh, P('x', 'y')), cb1)
    ckpt_dir = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [arr], tspecs,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    ds = NamedSharding(jtu.create_global_mesh((4, 2), ('x', 'y')), P('x', 'y'))

    m1, = serialization.run_deserialization([ds], tspecs, [(12, 2)],
                                            [np.float32])

    expected_data = {
        0: np.array([[0], [2], [4]], dtype=np.float32),
        1: np.array([[1], [3], [5]], dtype=np.float32),
        2: np.array([[6], [8], [10]], dtype=np.float32),
        3: np.array([[7], [9], [11]], dtype=np.float32),
        4: np.array([[12], [14], [0]], dtype=np.float32),
        5: np.array([[13], [15], [0]], dtype=np.float32),
        6: np.array([[0], [0], [0]], dtype=np.float32),
        7: np.array([[0], [0], [0]], dtype=np.float32),
    }

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), expected_data[l.device.id])

    new_ds = GSPMDSharding.get_replicated(list(global_mesh.devices.flat))
    m2, = serialization.run_deserialization([new_ds], tspecs, [(8, 2)], [np.float32])
    for l in m2.addressable_shards:
      self.assertArraysEqual(l.data, global_input_data1.astype('float32'))

  def test_checkpointing_scalar_jax_array(self):
    global_mesh = jtu.create_global_mesh((2,), ('x'))
    global_input_shape = ()
    data = np.array(4)
    s = NamedSharding(global_mesh, P(None))
    array1 = array.make_array_from_callback(
        global_input_shape, s, lambda idx: data[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [array1], tspecs,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    ds = NamedSharding(jtu.create_global_mesh((2,), ('x')), P(None))

    m1, = serialization.run_deserialization(
        [ds],
        tspecs,
        [()],
        [np.float32]
    )

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), data.astype(np.float32))

  def test_deserialize_tensorstore_array_jax_array(self):
    global_mesh = jtu.create_global_mesh((2,), ('x'))
    data = np.arange(1024)
    tspec = ts.array(data).spec()
    m1, = serialization.run_deserialization(
        [NamedSharding(global_mesh, P(None))],
        [tspec]
    )
    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), data)

  def test_spec_has_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
            'metadata': 3
        },
        'f': 4
    }
    self.assertTrue(serialization._spec_has_metadata(spec))
    self.assertTrue(
        serialization._spec_has_metadata({
            'driver': 'zarr',
            'kvstore': 'gfile',
            'metadata': {
                'chunks': 4,
                'shape': (32, 64)
            },
            'one_more': 'thing'
        }))

  def test_spec_has_no_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
        },
        'f': 4
    }
    self.assertFalse(serialization._spec_has_metadata(spec))

  def test_empty_spec_has_no_metadata(self):
    spec = {}
    self.assertFalse(serialization._spec_has_metadata(spec))

  @parameterized.named_parameters(
      ('gcs', 'gs://my/ckpt/dir/path'),
      ('file', '/my/ckpt/dir/path')
  )
  def test_get_tensorstore_spec_ocdbt(self, path):
    spec = serialization.get_tensorstore_spec(path, ocdbt=True)
    is_gcs_path = path.startswith('gs://')
    if is_gcs_path:
      self.assertEqual(spec['kvstore']['base'], os.path.dirname(path))
    else:
      self.assertEqual(spec['kvstore']['base'],
                       f'{serialization._DEFAULT_DRIVER}://{os.path.dirname(path)}')
    self.assertEqual(spec['kvstore']['path'], 'path')

  def test_get_tensorstore_spec_not_absolute_path(self):
    path = 'my/ckpt/path'
    with self.assertRaisesRegex(ValueError,
                                "Checkpoint path should be absolute"):
      serialization.get_tensorstore_spec(path, ocdbt=True)

  def test_maybe_cloud_storage(self):
    gs_path = 'gs://some-buck/path'
    gs_spec = serialization.get_tensorstore_spec(gs_path, ocdbt=True)
    self.assertTrue(serialization.maybe_cloud_storage(gs_spec))

    local_path = '/tmp/checkpoint'
    local_spec = serialization.get_tensorstore_spec(local_path, ocdbt=True)
    self.assertFalse(serialization.maybe_cloud_storage(local_spec))

    nested_tspec = {
        'driver': 'cast',
        'dtype': 'int32',
        'base': {
            'driver': 'zarr',
            'kvstore': {'driver': 'ocdbt', 'base': 's3://some-bucket/path'},
        },
    }
    self.assertTrue(serialization.maybe_cloud_storage(nested_tspec))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
