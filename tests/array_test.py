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
"""Tests for Array."""

import contextlib
import math
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src import deprecations
from jax._src import dispatch
from jax._src import op_shardings
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.util import safe_zip
from jax._src.sharding import common_devices_indices_map
from jax._src.sharding_impls import (_op_sharding_to_pos_sharding,
                                     pmap_sharding_devices_indices_map,
                                     NamedSharding, GSPMDSharding,
                                     PositionalSharding)
from jax.experimental.pjit import pjit
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from jax._src import array
from jax._src import prng

jax.config.parse_flags_with_absl()

with contextlib.suppress(ImportError):
  import pytest
  pytestmark = pytest.mark.multiaccelerator

# Run all tests with 8 CPU devices.
_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(8))

def tearDownModule():
  _exit_stack.close()


def create_array(shape, sharding, global_data=None):
  if global_data is None:
    global_data = np.arange(math.prod(shape)).reshape(shape)

  return array.make_array_from_callback(
      shape, sharding, lambda idx: global_data[idx]), global_data


class JaxArrayTest(jtu.JaxTestCase):

  def test_array_impl_name(self):
    self.assertEqual(array.ArrayImpl.__name__, "ArrayImpl")

  @parameterized.named_parameters(
      ("mesh_x_y", P("x", "y")),
      ("mesh_x", P("x")),
      ("mesh_y", P("y")),
      ("mesh_none_y", P(None, "y")),
      ("mesh_xy", P(("x", "y"))),
      ("mesh_fully_replicated", P()),
  )
  def test_jax_array_value(self, mesh_axes):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, global_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, mesh_axes))
    for s in arr.addressable_shards:
      self.assertTrue(dispatch.is_single_device_sharding(s.data.sharding))
      self.assertArraysEqual(s.data, global_data[s.index])
    self.assertArraysEqual(arr._value, global_data)
    self.assertArraysEqual(arr._npy_value, global_data)

  @parameterized.named_parameters(
      ("mesh_x_y", P("x", "y"),
       # There are more slices but for convenient purposes, checking for only
       # 2. The indices + shard_shape + replica_id should be unique enough.
       ((slice(0, 2), slice(0, 1)), (slice(0, 2), slice(1, 2))),
       (2, 1),
       [0, 0, 0, 0, 0, 0, 0, 0], False),
      ("mesh_x", P("x"),
       ((slice(0, 2), slice(None)), (slice(0, 2), slice(None))),
       (2, 2),
       [0, 1, 0, 1, 0, 1, 0, 1], False),
      ("mesh_y", P("y"),
       ((slice(0, 4), slice(None)), (slice(4, 8), slice(None))),
       (4, 2),
       [0, 0, 1, 1, 2, 2, 3, 3], False),
      ("mesh_none_y", P(None, "y"),
       ((slice(None), slice(0, 1)), (slice(None), slice(1, 2))),
       (8, 1),
       [0, 0, 1, 1, 2, 2, 3, 3], False),
      ("mesh_xy", P(("x", "y")),
       ((slice(0, 1), slice(None)), (slice(1, 2), slice(None))),
       (1, 2),
       [0, 0, 0, 0, 0, 0, 0, 0], False),
      ("mesh_fully_replicated", P(),
       ((slice(None), slice(None)), (slice(None), slice(None))),
       (8, 2),
       [0, 1, 2, 3, 4, 5, 6, 7], True),
  )
  def test_array_2d_shard(self, mesh_axes, expected_index, expected_shard_shape,
                          expected_replica_ids, expected_is_fully_replicated):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    s = jax.sharding.NamedSharding(global_mesh, mesh_axes)
    arr, global_input_data = create_array(global_input_shape, s)
    self.assertEqual(arr.ndim, 2)
    self.assertEqual(arr.size, 16)
    self.assertEqual(arr.addressable_shards[0].index, expected_index[0])
    self.assertEqual(arr.addressable_shards[1].index, expected_index[1])
    replica_ids = [i.replica_id for i in arr.addressable_shards]
    self.assertListEqual(replica_ids, expected_replica_ids)
    self.assertListEqual([i.device.id for i in arr.addressable_shards],
                         [0, 1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(arr.is_fully_replicated, expected_is_fully_replicated)
    for i, s in enumerate(arr.addressable_shards):
      self.assertEqual(s.data.aval,
                       core.ShapedArray(expected_shard_shape, s.data.dtype))
      self.assertArraysEqual(s.data, global_input_data[s.index])
      self.assertArraysEqual(s.data, arr.addressable_data(i))

    for g, l in safe_zip(arr.global_shards, arr.addressable_shards):
      self.assertEqual(g.device, l.device)
      self.assertEqual(g.index, l.index)
      self.assertEqual(g.replica_id, l.replica_id)
      self.assertEqual(g.data.aval, l.data.aval)
      self.assertArraysEqual(g.data, l.data)

  def test_addressable_data(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    shape = (8, 2)
    s = jax.sharding.NamedSharding(global_mesh, P(None))
    arr, inp_data = create_array(shape, s)
    for i in range(len(arr)):
      self.assertArraysEqual(inp_data, arr.addressable_data(i))

  def test_array_delete(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, _ = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))
    arr.delete()
    with self.assertRaisesRegex(RuntimeError, 'Array has been deleted.'):
      arr._check_if_deleted()
    self.assertIsNone(arr._npy_value)
    self.assertIsNone(arr._arrays)

  def test_single_device_array_usage_after_delete(self):
    x = jnp.array([1, 2, 3])
    x.delete()

    with self.assertRaisesRegex(RuntimeError, 'Array has been deleted.'):
      _ = x + 1

  def test_multi_device_array_usage_after_delete(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    shape = (8, 2)
    arr = jax.device_put(np.arange(math.prod(shape), dtype=np.int32),
                         jax.sharding.NamedSharding(global_mesh, P('x')))
    arr.delete()

    with self.assertRaisesRegex(
        RuntimeError, r'Array has been deleted with shape=int32\[16\].'):
      _ = arr + 1

  def test_device_put(self):
    numpy_array = np.array([1, 2, 3])
    arr = jax.device_put(numpy_array, jax.devices()[0])
    self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)
    self.assertArraysEqual(arr, numpy_array)
    self.assertEqual(arr._committed, True)
    for i in arr.addressable_shards:
      self.assertArraysEqual(i.data, numpy_array)
      self.assertEqual(i.device, jax.devices()[0])
      self.assertEqual(i.index, (slice(None),))
      self.assertEqual(i.replica_id, 0)

  def test_device_put_array_delete(self):
    arr = jax.device_put(np.array([1, 2, 3]), jax.devices()[0])
    arr.delete()
    with self.assertRaisesRegex(RuntimeError, 'Array has been deleted.'):
      arr._check_if_deleted()
    self.assertIsNone(arr._npy_value)
    self.assertIsNone(arr._arrays)

  def test_array_device_get(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))
    self.assertArraysEqual(jax.device_get(arr), input_data)

  def test_repr(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, _ = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))
    self.assertStartsWith(repr(arr), "Array(")

  def test_empty_repr(self):
    shape = (0, 5)
    dtype = 'float32'
    x = jnp.empty(shape, dtype)
    self.assertEqual(repr(x), f"Array([], shape={shape}, dtype={dtype})")

  def test_jnp_array(self):
    arr = jnp.array([1, 2, 3])
    self.assertIsInstance(arr, array.ArrayImpl)
    self.assertTrue(dispatch.is_single_device_sharding(arr.sharding))
    self.assertEqual(arr._committed, False)
    self.assertFalse(arr.weak_type)

  def test_jnp_array_jit_add(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    arr = jax.jit(lambda x, y: x + y)(a, b)
    self.assertIsInstance(arr, array.ArrayImpl)
    self.assertArraysEqual(arr, np.array([5, 7, 9]))
    self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)

  def test_jnp_array_jnp_add(self):
    arr = jnp.add(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    self.assertIsInstance(arr, array.ArrayImpl)
    self.assertArraysEqual(arr, np.array([5, 7, 9]))
    self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)

  def test_jnp_array_normal_add(self):
    a = jnp.array([1, 2, 3])
    b = jnp.array([4, 5, 6])
    arr = a + b
    self.assertIsInstance(arr, array.ArrayImpl)
    self.assertArraysEqual(arr, np.array([5, 7, 9]))
    self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)

  def test_array_sharded_astype(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))
    arr_float32 = arr.astype(jnp.float32)
    self.assertEqual(arr_float32.dtype, np.float32)
    self.assertArraysEqual(arr_float32, input_data.astype(np.float32))
    self.assertLen(arr_float32.addressable_shards, 8)
    for i in arr_float32.addressable_shards:
      self.assertArraysEqual(i.data, input_data[i.index].astype(np.float32))

  def test_jnp_array_astype(self):
    arr = jnp.array([1, 2, 3])
    arr_float32 = arr.astype(jnp.float32)
    self.assertEqual(arr_float32.dtype, np.float32)
    self.assertArraysEqual(arr_float32, arr.astype(np.float32))

  def test_array_delete_idempotent(self):
    mesh = jtu.create_global_mesh((2,), ('x',))
    arr = jax.device_put(np.arange(8), jax.sharding.NamedSharding(mesh, P('x')))

    arr.delete()
    self.assertTrue(arr.is_deleted())

    arr.delete()  # Run delete again to check if it's idempotent.
    self.assertTrue(arr.is_deleted())

  def test_sharded_add(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    a, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))
    b, _ = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x')))
    out = a + b
    expected = input_data + input_data
    self.assertArraysEqual(out, expected)
    self.assertLen(out.addressable_shards, 8)
    for i in out.addressable_shards:
      self.assertArraysEqual(i.data, expected[i.index])

  def test_sharded_zeros_like(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    a, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))
    out = jnp.zeros_like(a)
    expected = jnp.zeros(input_data.shape, dtype=a.dtype)
    self.assertArraysEqual(out, expected)
    self.assertLen(out.addressable_shards, 8)
    for i in out.addressable_shards:
      self.assertArraysEqual(i.data, expected[i.index])

  def test_zeros_like(self):
    a = jnp.array([1, 2, 3], dtype=np.int32)
    out = jnp.zeros_like(a)
    expected = np.zeros(a.shape, dtype=np.int32)
    self.assertArraysEqual(out, expected)
    self.assertTrue(dispatch.is_single_device_sharding(out.sharding))

  def test_wrong_num_arrays(self):
    if jax.device_count() < 4:
      self.skipTest('Requires more than 4 devices')
    shape = (8, 2)
    mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    devices = jax.local_devices()[:2]  # Taking up to 2 devices
    s = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    inp_data = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    di_map = s.devices_indices_map(shape)
    bufs = [jax.device_put(inp_data[di_map[d]], d) for d in devices]
    with self.assertRaisesRegex(
        ValueError,
        r'Expected 2 per-device arrays \(this is how many devices are addressable '
        r'by the sharding\), but got 1'):
      array.ArrayImpl(core.ShapedArray(shape, np.float32), s, bufs[:1], committed=True)

    for buf, d in zip(list(bufs), jax.local_devices()[2:4]):
      bufs.append(jax.device_put(buf, d))
    with self.assertRaisesRegex(
        ValueError,
        r'Expected 2 per-device arrays \(this is how many devices are addressable '
        r'by the sharding\), but got 4'):
      array.ArrayImpl(core.ShapedArray(shape, np.float32), s, bufs, committed=True)

  def test_arrays_not_in_device_assignment(self):
    if jax.device_count() < 4:
      self.skipTest('Requires more than 4 devices')
    shape = (8, 2)
    mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    # sharding device ids = {0, 1}
    s = jax.sharding.NamedSharding(mesh, P('x'))
    inp_data = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    # _arrays device ids = {2, 3}
    bufs = [jax.device_put(inp_data, d) for d in jax.devices()[2:4]]
    with self.assertRaisesRegex(
        ValueError,
        "Addressable devices and per-device arrays devices do not match. "
        "Sharding contains devices {0, 1} that are not present in per-device "
        "arrays. Per-device arrays contain devices {2, 3} that are not present "
        "in the sharding."):
      array.ArrayImpl(core.ShapedArray(shape, np.float32), s, bufs, committed=True)

  def test_different_devices_in_arrays_than_sharding(self):
    if jax.device_count() < 3:
      self.skipTest('Requires more than 3 devices')
    shape = (8, 2)
    mesh = jax.sharding.Mesh(np.array([jax.devices()[1], jax.devices()[2]]), ('x'))
    # sharding device ids = {1, 2}
    s = jax.sharding.NamedSharding(mesh, P('x'))
    inp_data = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    # _arrays device ids = {0, 1}
    bufs = [jax.device_put(inp_data, d) for d in jax.devices()[:2]]
    with self.assertRaisesRegex(
        ValueError,
        "Addressable devices and per-device arrays devices do not match. "
        r"Sharding contains devices \{2\} that are not present in per-device "
        r"arrays. Per-device arrays contain devices \{0\} that are not present "
        "in the sharding."):
      array.ArrayImpl(core.ShapedArray(shape, np.float32), s, bufs, committed=True)

  def test_duplicated_devices_in_arrays(self):
    if xc._version <= 274:
      self.skipTest('Test requires jaxlib version 275')
    shape = (8, 2)
    mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    # Sharding device ids = {0, 1}
    s = jax.sharding.NamedSharding(mesh, P('x'))
    inp_data = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    # _arrays device ids = {0, 2}
    bufs = [jax.device_put(inp_data, jax.devices()[0]) for _ in range(2)]
    with self.assertRaisesRegex(
        ValueError,
        'When making an array from single-device arrays, the input arrays must'
        ' be from distinct devices'):
      array.ArrayImpl(core.ShapedArray(shape, np.float32), s, bufs, committed=True)

  @parameterized.named_parameters(
      ("mesh_x_y", P("x", "y"), (2, 2)),
      ("mesh_x", P("x"), (2, 4)),
      ("mesh_y", P("y"), (4, 4)),
      ("mesh_none_y", P(None, "y"), (8, 2)),
      ("mesh_none_x", P(None, "x"), (8, 1)),
      ("mesh_xy", P(("x", "y")), (1, 4)),
      ("mesh_replicated", P(()), (8, 4)),
  )
  def test_shard_shape_mismatch_with_buffer_shape(self, pspec, expected_shard_shape):
    shape = (8, 4)
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mps = jax.sharding.NamedSharding(mesh, pspec)
    inp_data = np.arange(5)

    str_expected_shard_shape = str(expected_shard_shape).replace(
        r"(", r"\(").replace(r")", r"\)")
    with self.assertRaisesRegex(
        ValueError,
        f"Expected shard shape {str_expected_shard_shape} doesn't match the "
        "single device array shape"):
      array.make_array_from_callback(shape, mps, lambda idx: inp_data)

  def test_mismatch_dtype(self):
    shape = (8, 2)
    mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    s = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    inp_data = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    indices = s.devices_indices_map(shape)
    bufs = [jax.device_put(inp_data[indices[d]], d) for d in mesh.local_devices]
    with self.assertRaisesRegex(
        ValueError,
        "Input buffers to `Array` must have matching dtypes. "
        "Got int32, expected float32"):
      array.ArrayImpl(core.ShapedArray(shape, np.float32), s, bufs, committed=True)

  def test_array_iter_pmap_sharding(self):
    if jax.device_count() < 2:
      self.skipTest('Test requires >= 2 devices.')

    x = jnp.array([[1., 0., 0.], [0., 2., 3.]])
    y = jax.pmap(jnp.sin)(x)
    self.assertArraysEqual([list(a.devices())[0] for a in y],
                           y.sharding._device_assignment,
                           allow_object_dtype=True)

    sin_x = iter(np.sin(x))
    for i, j in zip(iter(y), sin_x):
      self.assertIsInstance(i, array.ArrayImpl)
      self.assertArraysAllClose(i, j)

  def test_array_iter_pmap_sharding_last_dim_sharded(self):
    if jax.device_count() < 2:
      self.skipTest('Test requires >= 2 devices.')

    x = jnp.array([[1., 0., 0.], [0., 2., 3.]])
    y = jax.pmap(jnp.sin, out_axes=1)(x)

    for i, j in zip(iter(y), iter(np.sin(x).T)):
      self.assertArraysAllClose(i, j)

  def test_array_iter_mesh_pspec_sharding_multi_device(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))

    for i, j in zip(iter(arr), iter(input_data)):
      self.assertIsInstance(i, array.ArrayImpl)
      self.assertArraysEqual(i, j)

  def test_array_iter_replicated_multi_device(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P(None)))

    for i, j in zip(iter(arr), iter(input_data)):
      self.assertIsInstance(i, array.ArrayImpl)
      self.assertArraysEqual(i, j)
      self.assertLen(i.sharding.device_set, 8)
      self.assertTrue(
        op_shardings.are_op_shardings_equal(
            arr.sharding._to_xla_hlo_sharding(arr.ndim),
            i.sharding._to_xla_hlo_sharding(i.ndim)))

  def test_array_getitem_mesh_pspec_sharding_multi_device(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))

    s = arr[2:4, 0:1]
    self.assertIsInstance(s, array.ArrayImpl)
    self.assertArraysEqual(s, input_data[2:4, 0:1])

    p = arr[:2]
    self.assertIsInstance(p, array.ArrayImpl)
    self.assertArraysEqual(p, input_data[:2])

  def test_array_getitem_compile_multi_device_sharding(self):
    def _check(out, inp, shard_shape):
      self.assertArraysEqual(out, inp)
      self.assertEqual(out.sharding.shard_shape(out.shape), shard_shape)
      self.assertNotIsInstance(out.sharding, jax.sharding.SingleDeviceSharding)

    global_mesh = jtu.create_global_mesh((2, 2, 2), ('x', 'y', 'z'))
    input_shape = (4, 4, 2)
    arr, np_inp = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y', 'z')))

    _check(arr[:, -1, :], np_inp[:, -1, :], (2, 1))
    _check(arr[0, 0, 0], np_inp[0, 0, 0], ())
    _check(arr[-1, -1, :], np_inp[-1, -1, :], (1,))
    _check(arr[:, 1, 0], np_inp[:, 1, 0], (2,))
    _check(arr[:, :, :], np_inp[:, :, :], (2, 2, 1))
    _check(arr[3, :, :], np_inp[3, :, :], (2, 1))
    _check(arr[-1, -1, -1], np_inp[-1, -1, -1], ())
    _check(arr[2, -1, :], np_inp[2, -1, :], (1,))
    _check(arr[2, 3, 1], np_inp[2, 3, 1], ())
    _check(arr[-1], np_inp[-1], (2, 1))
    _check(arr[:], np_inp[:], (2, 2, 1))
    _check(arr[np.array(0), :, :], np_inp[np.array(0), :, :], (2, 1))
    _check(arr[jnp.array(0), :, :], np_inp[jnp.array(0), :, :], (2, 1))
    _check(arr[0, :2, 1], np_inp[0, :2, 1], (2,))
    _check(arr[:, 1::2], np_inp[:, 1::2], (2, 2, 1))
    _check(arr[:, -1:, :], np_inp[:, -1:, :], (2, 1, 1))
    _check(arr[0:6:1], np_inp[0:6:1], (2, 2, 1))
    _check(arr[:4], np_inp[:4], (2, 2, 1))
    _check(arr[::-1], np_inp[::-1], (2, 2, 1))
    _check(arr[1], np_inp[1], (2, 1))

  def test_array_getitem_replicated_multi_device(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P(None)))

    s = arr[2:4, 0:1]
    self.assertIsInstance(s, array.ArrayImpl)
    self.assertArraysEqual(s, np.array([[4], [6]]))
    self.assertLen(s.sharding.device_set, 8)
    self.assertTrue(
        op_shardings.are_op_shardings_equal(
            arr.sharding._to_xla_hlo_sharding(arr.ndim),
            s.sharding._to_xla_hlo_sharding(s.ndim)))

    p = arr[:2]
    self.assertIsInstance(p, array.ArrayImpl)
    self.assertArraysEqual(p, input_data[:2])
    self.assertLen(s.sharding.device_set, 8)
    self.assertTrue(
        op_shardings.are_op_shardings_equal(
            arr.sharding._to_xla_hlo_sharding(arr.ndim),
            s.sharding._to_xla_hlo_sharding(s.ndim)))

  def test_array_iter_mesh_pspec_sharding_single_device(self):
    if jax.device_count() < 2:
      self.skipTest('Test requires >= 2 devices.')

    single_dev = jax.devices()[1:2]
    mesh = jax.sharding.Mesh(np.array(single_dev), ('x'))
    input_shape = (8, 2)
    arr, input_data = create_array(
        input_shape, jax.sharding.NamedSharding(mesh, P('x')))

    for i, j in zip(arr, iter(input_data)):
      self.assertArraysEqual(i, j)
      self.assertEqual(i.devices(), {single_dev[0]})

  def test_array_shards_committed(self):
    if jax.device_count() < 2:
      self.skipTest('Test requires >= 2 devices.')

    x = jnp.array([1, 2, 3])
    for s in x.addressable_shards:
      self.assertEqual(s.data._committed, x._committed)
      self.assertFalse(s.data._committed)

    y = jax.device_put(x, jax.devices()[1])
    for s in y.addressable_shards:
      self.assertEqual(s.data._committed, y._committed)
      self.assertTrue(s.data._committed)

  def test_array_jnp_array_copy_multi_device(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, _ = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))

    c_arr = jnp.array(arr, copy=True)
    self.assertArraysEqual(arr, c_arr)
    self.assertEqual(arr._committed, c_arr._committed)
    for a, c in safe_zip(arr.addressable_shards, c_arr.addressable_shards):
      self.assertArraysEqual(a.data, c.data)
      self.assertEqual(a.index, c.index)
      self.assertEqual(a.replica_id, c.replica_id)
      self.assertEqual(a.device, c.device)
      self.assertNotEqual(a.data.unsafe_buffer_pointer(),
                          c.data.unsafe_buffer_pointer())

  def test_array_addressable_shards(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    arr, _ = create_array(
        input_shape, jax.sharding.NamedSharding(global_mesh, P('x', 'y')))

    for a in arr.addressable_shards:
      self.assertIsInstance(a.data, array.ArrayImpl)

    x = jnp.array([1, 2, 3])
    self.assertIsInstance(x.addressable_data(0), array.ArrayImpl)

  def test_array_not_hashable(self):
    x = jnp.arange(4)
    with self.assertRaisesRegex(TypeError, "unhashable type"):
      hash(x)

    @jax.jit
    def check_tracer_hash(x):
      self.assertIsInstance(hash(x), int)

    if deprecations.is_accelerated('tracer-hash'):
      with self.assertRaisesRegex(TypeError, "unhashable type"):
        check_tracer_hash(x)
    else:
      with self.assertWarnsRegex(FutureWarning, "unhashable type"):
        check_tracer_hash(x)

  def test_shape_dtype_struct_sharding_jit(self):
    mesh = jtu.create_global_mesh((8,), ('x'))
    s = jax.sharding.NamedSharding(mesh, P('x'))

    x_dummy = jax.ShapeDtypeStruct(
        shape=(16,),
        dtype=jnp.dtype('float32'),
        sharding=s)

    def f(x):
      return x * 2

    c = jax.jit(f).lower(x_dummy).compile()
    input_shardings, output_shardings = c.input_shardings, c.output_shardings
    self.assertLen(input_shardings, 2)
    self.assertEqual(input_shardings[1], {})
    self.assertEqual(input_shardings[1], {})

    self.assertTrue(
        op_shardings.are_op_shardings_equal(
            input_shardings[0][0]._to_xla_hlo_sharding(x_dummy.ndim),
            s._to_xla_hlo_sharding(x_dummy.ndim)))
    self.assertTrue(
        op_shardings.are_op_shardings_equal(
            output_shardings._to_xla_hlo_sharding(x_dummy.ndim),
            s._to_xla_hlo_sharding(x_dummy.ndim)))

  def test_shape_dtype_struct_sharding_pjit(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    s = jax.sharding.NamedSharding(mesh, P('x', 'y'))

    def f(x):
      return x * 2.

    x_dummy = jax.ShapeDtypeStruct(
        shape=(8, 2),
        dtype=jnp.dtype('float32'),
        sharding=s)

    c = pjit(f).lower(x_dummy).compile()
    input_shardings, output_shardings = c.input_shardings, c.output_shardings
    self.assertTrue(
        op_shardings.are_op_shardings_equal(
            input_shardings[0][0]._to_xla_hlo_sharding(x_dummy.ndim),
            s._to_xla_hlo_sharding(x_dummy.ndim)))
    self.assertTrue(
        op_shardings.are_op_shardings_equal(
            output_shardings._to_xla_hlo_sharding(x_dummy.ndim),
            s._to_xla_hlo_sharding(x_dummy.ndim)))

  # TODO(skyewm): remove this test when we can remove the workaround manual
  # defragment API
  @jtu.skip_on_devices('cpu')  # defragment not implemented for TFRT CPU
  def test_defragment(self):
    if xb.using_pjrt_c_api():
      self.skipTest("Manual defragment not exposed via PJRT C API")

    # Create a few arrays
    global_mesh = jtu.create_global_mesh((jax.local_device_count(),), ('x',))
    shape = (8, 2)
    mpsharding = jax.sharding.NamedSharding(global_mesh, P('x',))
    arr1, data = create_array(shape, mpsharding)
    arr2, _ = create_array(shape, mpsharding, data)
    arr3, _ = create_array(shape, mpsharding, data)

    # Delete one of them
    arr2.delete()

    # Defragment
    xb.get_backend().defragment()

    # Sanity check remaining arrays
    self.assertArraysEqual(arr1, data)
    self.assertArraysEqual(arr1 + arr3, data * 2)

    # TODO(skyewm): check that defragmentation actually happened. I originally
    # thought to do this with unsafe_buffer_pointer(), but that's not always the
    # device memory address. Other ideas include causing enough fragmentation to
    # OOM, and exposing allocator stats in Python.

  def test_on_device_size_in_bytes(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    a, _ = create_array(
        (8, 2), jax.sharding.NamedSharding(global_mesh, P('x', 'y')))
    shard_size = a.addressable_shards[0].data.on_device_size_in_bytes()
    self.assertGreaterEqual(shard_size, 4 * 2)
    self.assertEqual(shard_size * len(a.global_shards),
                     a.on_device_size_in_bytes())

  def test_array_is_ready(self):
    x = jax.device_put(jnp.arange(8.), jax.devices()[0])
    x.is_ready()  # doesn't crash

  def test_process_allgather_single_host(self):
    x = jnp.arange(8.)
    out = multihost_utils.process_allgather(x)
    self.assertEqual(out.shape, x.shape)
    self.assertArraysEqual(out, x)

  @jtu.sample_product(
    dtype=jtu.dtypes.all,
    shape=[(), (10), (2, 3)],
  )
  @jtu.run_on_devices("cpu")
  def test_buffer_protocol(self, dtype, shape):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    y = jax.device_put(x)
    if dtype == jax.dtypes.bfloat16:
      with self.assertRaisesRegex(
          BufferError,
          'Buffers of type BF16 are not supported by the Python buffer '
          'protocol.'
      ):
        memoryview(y)
      return

    x_bytes = memoryview(x).tobytes()
    y_bytes = memoryview(y).tobytes()
    self.assertEqual(x_bytes, y_bytes)

  @jtu.run_on_devices("cpu")
  def test_buffer_protocol_deletion(self):
    rng = jtu.rand_default(self.rng())
    x = rng((3, 4), np.float32)
    y = jax.device_put(x)
    x_bytes = memoryview(x).tobytes()
    y_view = memoryview(y)
    # The array does not actually get deleted until any external reference is
    # dropped. Arguably we should make calling delete() in these circumstances
    # return an error instead, but that would be a behavior change for existing
    # users.
    y.delete()
    y_bytes = y_view.tobytes()
    self.assertEqual(x_bytes, y_bytes)

  def test_array_copy_to_host_async(self):
    global_mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    x = pjit(lambda: jnp.arange(8.),
             out_shardings=jax.sharding.NamedSharding(global_mesh, P(None)))()
    self.assertLen(x.sharding.device_set, 4)
    x.copy_to_host_async()  # doesn't crash
    self.assertArraysEqual(np.arange(8.), x)

  def test_array_fully_replicated_shard(self):

    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    arr, inp_data = create_array(
        inp_shape, jax.sharding.NamedSharding(global_mesh, P()))
    fs = arr._fully_replicated_shard()
    self.assertEqual(fs.shape, inp_shape)
    self.assertTrue(dispatch.is_single_device_sharding(fs.sharding))
    self.assertArraysEqual(fs, inp_data)
    self.assertArraysEqual(arr.addressable_data(0), inp_data)

  def test_shard_array_to_fully_replicated(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    sharding = jax.sharding.NamedSharding(global_mesh, P())
    arr = jnp.arange(16)
    self.assertFalse(arr._committed)
    self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)
    out = jax.jit(lambda x: x * 2, in_shardings=sharding)(arr)
    self.assertTrue(out.sharding.is_fully_replicated)
    self.assertArraysEqual(out, arr * 2)

  def test_fully_replicated_donated_array_is_deleted(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    sharding = jax.sharding.NamedSharding(global_mesh, P())
    arr = jnp.arange(16)
    arr_copy = arr.copy()
    self.assertFalse(arr._committed)
    self.assertIsInstance(arr.sharding, jax.sharding.SingleDeviceSharding)
    out = jax.jit(lambda x: x * 2, in_shardings=sharding, donate_argnums=0)(arr)
    self.assertTrue(out.sharding.is_fully_replicated)
    self.assertArraysEqual(out, arr_copy * 2)
    self.assertTrue(arr.is_deleted())

  @parameterized.product(dtype=jtu.dtypes.all + jtu.dtypes.custom_floats)
  def test_shards_have_correct_dtype(self, dtype):
    x = jnp.ones((), dtype=dtype)
    for shard in x.addressable_shards:
      self.assertEqual(shard.data.dtype, dtype)

  def test_make_array_from_callback_global_array(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    sharding = jax.sharding.NamedSharding(mesh, P())
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, sharding)

    out = jax.make_array_from_callback(np_inp.shape, sharding,
                                       lambda idx: arr[idx])
    self.assertArraysEqual(out, arr)
    self.assertEqual(out.sharding, sharding)

    sharding2 = NamedSharding(mesh, P('x', 'y'))
    arr2 = jax.device_put(np_inp, sharding2)
    out2 = jax.make_array_from_callback(np_inp.shape, sharding2,
                                       lambda idx: arr2[idx])
    self.assertArraysEqual(out2, arr2)
    self.assertEqual(out2.sharding, sharding2)

  def test_make_array_from_process_data_single_host_data_sharding(self):
    data = np.ones((1, 512))
    mesh = jtu.create_global_mesh((1, 1), ('x', 'unused'))
    sharding_spec = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('x')
    )
    global_shape = data.shape
    result = jax.make_array_from_process_local_data(
        sharding_spec, data, global_shape
    )
    self.assertIsInstance(result, jax.Array)
    self.assertEqual(result.shape, data.shape)
    self.assertEqual(result.sharding, sharding_spec)

class ShardingTest(jtu.JaxTestCase):

  def test_mesh_pspec_sharding_interface(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    pspec = P('y', 'x')
    global_shape = (8, 4)
    mp_sharding = jax.sharding.NamedSharding(mesh, pspec)
    di_map = mp_sharding.devices_indices_map(global_shape)
    hlo_sharding = mp_sharding._to_xla_hlo_sharding(len(global_shape))
    device_assignment = mp_sharding._device_assignment

    self.assertEqual(di_map[mesh.devices.flat[0]], (slice(0, 4), slice(0, 1)))
    self.assertArraysEqual(device_assignment, list(mesh.devices.flat),
                           allow_object_dtype=True)
    self.assertTrue(hlo_sharding.is_tiled())
    self.assertListEqual(hlo_sharding.tile_assignment_dimensions(), [2, 4])
    self.assertListEqual(hlo_sharding.tile_assignment_devices(),
                         [0, 2, 4, 6, 1, 3, 5, 7])

  def test_util_clear_cache(self):
    mesh = jtu.create_global_mesh((1,), ('x',))
    s = NamedSharding(mesh, P())
    s.devices_indices_map((8,))
    jax.clear_caches()
    s.devices_indices_map((8,))
    c = common_devices_indices_map.cache_info()
    self.assertEqual(c.currsize, 1)

  @parameterized.named_parameters(
      ("mesh_x_y", P("x", "y")),
      ("mesh_x", P("x")),
      ("mesh_y", P("y")),
      ("mesh_none_y", P(None, "y")),
      ("mesh_none_x", P(None, "x")),
      ("mesh_xy", P(("x", "y"))),
      ("mesh_fully_replicated", P()),
  )
  def test_op_sharding_indices(self, pspec):
    shape = (8, 4)
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mps = jax.sharding.NamedSharding(mesh, pspec)
    ops = jax.sharding.GSPMDSharding(
        list(mesh.devices.flat), mps._to_xla_hlo_sharding(len(shape)))
    self.assertDictEqual(
        ops.devices_indices_map(shape), mps.devices_indices_map(shape))

  @parameterized.named_parameters(
      ("mesh_x_y", P("x", "y"), (2, 2)),
      ("mesh_x", P("x"), (2, 4)),
      ("mesh_y", P("y"), (4, 4)),
      ("mesh_none_y", P(None, "y"), (8, 2)),
      ("mesh_none_x", P(None, "x"), (8, 1)),
      ("mesh_xy", P(("x", "y")), (1, 4)),
      ("mesh_fully_replicated", P(), (8, 4)),
  )
  def test_shard_shape(self, pspec, expected_shard_shape):
    shape = (8, 4)
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mps = jax.sharding.NamedSharding(mesh, pspec)
    self.assertEqual(mps.shard_shape(shape), expected_shard_shape)

  def test_uneven_shard_error(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mps = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    with self.assertRaisesRegex(
        ValueError,
        r"Sharding.*implies that array axis 1 is partitioned 2 times, but the "
        r"dimension size is 3 \(full shape: \(8, 3\), per-dimension tiling "
        r"factors: \[4, 2\] should evenly divide the shape\)"):
      mps.shard_shape((8, 3))

  def test_pmap_sharding_hash_eq(self):
    if jax.device_count() < 2:
      self.skipTest('Test needs >= 2 devices.')

    shape = (2, 2)
    num_elements = math.prod(shape)
    inp_data = np.arange(num_elements).reshape(shape)
    out = jax.pmap(lambda x: x)(inp_data)
    self.assertIsInstance(out.sharding, jax.sharding.PmapSharding)
    # Populate the device_indices_map cache.
    _ = out.sharding.devices_indices_map(shape)
    cache_info1 = pmap_sharding_devices_indices_map.cache_info()

    inp_data2 = np.arange(num_elements, num_elements + num_elements).reshape(shape)
    out2 = jax.pmap(lambda x: x)(inp_data2)
    # Populate the device_indices_map cache.
    _ = out2.sharding.devices_indices_map(shape)
    cache_info2 = pmap_sharding_devices_indices_map.cache_info()

    self.assertGreater(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)

  def test_is_compatible_error(self):
    shape = (8, 2)
    mesh = jtu.create_global_mesh((1, 1, 2), ('replica', 'data', 'mdl'))
    mps = jax.sharding.NamedSharding(mesh, P(None, ('mdl',), None, None))
    new_mps = jax.sharding.NamedSharding._from_parsed_pspec(
        mps.mesh, mps._parsed_pspec)

    with self.assertRaisesRegex(
        ValueError,
        r"Sharding NamedSharding\(mesh=Mesh\('replica': 1, 'data': 1, 'mdl': 2\), "
        r"spec=PartitionSpec\(None, \('mdl',\), None, None\).*\) is only "
        "valid for values of rank at least 4, but was applied to a value of rank 2"):
      new_mps.check_compatible_aval(shape)

  def test_is_subclass(self):
    # array version of api_test.py::APITest::test_is_subclass
    self.assertTrue(issubclass(array.ArrayImpl, jax.Array))
    self.assertFalse(issubclass(array.ArrayImpl, np.ndarray))

  def test_gspmd_sharding_repr(self):
    op = xc.OpSharding()
    op.type = xc.OpSharding.Type.OTHER
    op.tile_assignment_dimensions = [4, 1, 2]
    op.tile_assignment_devices = [0, 1, 2, 3, 4, 5, 6, 7]
    op.replicate_on_last_tile_dim = True
    s = jax.sharding.GSPMDSharding(jax.devices(), op)
    # memory kind also appears in the repr but only for TPU.
    self.assertIn(
        'GSPMDSharding({devices=[4,1,2]0,1,2,3,4,5,6,7 '
        'last_tile_dim_replicate}', repr(s))

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.REPLICATED
    s2 = jax.sharding.GSPMDSharding(jax.devices(), op2)
    # memory kind also appears in the repr but only for TPU.
    self.assertIn('GSPMDSharding({replicated}', repr(s2))

  def test_positional_sharding_fully_replicated(self):
    sharding = PositionalSharding(jax.devices())
    jax.device_put(jnp.array(1), sharding.replicate())  # doesn't crash

  @parameterized.named_parameters(
      ("mesh_x_y",              P("x", "y"),   (4, 2), (),   False),
      ("mesh_x",                P("x"),        (4, 2), (1,), False),
      ("mesh_y",                P("y"),        (4, 2), (0,), True),
      ("mesh_none_y",           P(None, "y"),  (4, 2), (0,), False),
      ("mesh_none_x",           P(None, "x"),  (4, 2), (1,), True),
      ("mesh_xy",               P(("x", "y")), (8, 1), (),   False),
      ("mesh_fully_replicated", P(),           (4, 2), None, False),
  )
  def test_positional_sharding_op_sharding_lowering(
      self, pspec, shape, axes, transpose):
    value_shape = (8, 4)

    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mps = jax.sharding.NamedSharding(mesh, pspec)
    devices = jax.local_devices()[:8] # Taking up to 8 devices

    devices_sharding = jax.sharding.PositionalSharding(devices)
    devices_sharding = devices_sharding.reshape(shape).replicate(axes)
    if transpose:
      devices_sharding = devices_sharding.T

    op1 = mps._to_xla_hlo_sharding(len(value_shape))
    op2 = devices_sharding._to_xla_hlo_sharding(len(value_shape))

    self.assertEqual(mps.shard_shape(value_shape),
                     devices_sharding.shard_shape(value_shape))
    self.assertTrue(op_shardings.are_op_shardings_equal(op1, op2))

  def test_positional_sharding_aval_compatible(self):
    if jax.device_count() < 2:
      self.skipTest('Requires >=2 devices')
    sharding = PositionalSharding(jax.devices()).reshape(1, jax.device_count())
    x = jax.random.uniform(jax.random.key(42), (256, 20, 1000))
    with self.assertRaisesRegex(
        ValueError,
        'Sharding PositionalSharding.*is only valid for values of rank 2, but'
        ' was applied to a value of rank 3'):
      jax.lax.with_sharding_constraint(x, sharding)

  @parameterized.named_parameters(
      ("2d_mesh_x_y",              (4, 2), P("x", "y")),
      ("2d_mesh_x",                (4, 2), P("x")),
      ("2d_mesh_y",                (4, 2), P("y")),
      ("2d_mesh_none_y",           (4, 2), P(None, "y")),
      ("2d_mesh_none_x",           (4, 2), P(None, "x")),
      ("2d_mesh_xy",               (4, 2), P(("x", "y"))),
      ("2d_mesh_none_xy",          (4, 2), P(None, ("x", "y"))),
      ("2d_mesh_x_none",           (2, 1), P(('x',), None)),
      ("2d_mesh_fully_replicated", (4, 2), P()),
      ("3d_mesh_none_none_z",      (2, 2, 2), P(None, None, 'z')),
      ("3d_mesh_none_y_none",      (2, 2, 2), P(None, 'y', None)),
      ("3d_mesh_x_y_none",         (2, 2, 2), P('x', 'y', None)),
      ("3d_mesh_none_yz",          (2, 2, 2), P(None, ('y', 'z'))),
      ("3d_mesh_x_none_yz",        (2, 2, 2), P('x', None, ('y', 'z'))),
      ("3d_mesh_none_x_yz",        (2, 2, 2), P(None, 'x', ('y', 'z'))),
      ("3d_mesh_xy_z",             (2, 2, 2), P(('x', 'y'), 'z')),
      ("3d_mesh_xy_none_z",        (2, 2, 2), P(('x', 'y'), None, 'z')),
      ("3d_mesh_x_y_z",            (2, 2, 2), P('x', 'y', 'z')),
      ("3d_mesh_xz_y",             (2, 2, 2), P(('x', 'z'), 'y')),
      ("3d_mesh_xz_none_y",        (2, 2, 2), P(('x', 'z'), None, 'y')),
      ("3d_mesh_y_none_xz",        (2, 2, 2), P('y', None, ('x', 'z'))),
      ("3d_mesh_none_y_xz",        (2, 2, 2), P(None, 'y', ('x', 'z'))),
      ("3d_mesh2_none_none_z",     (1, 2, 4), P(None, None, 'z')),
      ("3d_mesh2_x_none_none",     (1, 2, 4), P('x', None, None)),
      ("3d_mesh_x_none_none",      (2, 1, 1), P('x', None, None)),
  )
  def test_positional_sharding_from_op_sharding(self, mesh_shape, pspec):
    ndim = len(mesh_shape)
    mesh = jtu.create_global_mesh(
        mesh_shape, ('x', 'y') if ndim == 2 else ('x', 'y', 'z'))
    mps = jax.sharding.NamedSharding(mesh, pspec)
    original_op_sharding = mps._to_xla_hlo_sharding(ndim)
    ps = _op_sharding_to_pos_sharding(original_op_sharding,
                                           mps._device_assignment)
    out_op_sharding = ps._to_xla_hlo_sharding(ndim)
    self.assertTrue(op_shardings.are_op_shardings_equal(
        original_op_sharding, out_op_sharding))

  @parameterized.named_parameters(
      ("2d_mesh_x",                (1, 1), P("x", "y")),
      ("2d_mesh_x_y",              (4, 2), P("x", "y")),
      ("2d_mesh_empty",            (2, 1), P()),
      ("2d_mesh_p_none",           (2, 1), P(None)),
      ("2d_mesh_none_none",        (2, 1), P(None, None)),
      ("2d_mesh_tuple_empty",      (2, 1), P((),)),
      ("2d_mesh_x_none",           (2, 1), P(('x',), None)),
      ("2d_mesh_xy_none",          (2, 1), P(('x', 'y'), None)),
      ("2d_mesh_none",             (2, 1), None),
      ("2d_mesh_x_tuple_empty",    (2, 1), P('x', (), (), ())),
      ("2d_mesh_3_tuple_empty",    (2, 1), P((), (), ())),
      ("3d_mesh2_x_none_none",     (1, 2, 4), P('x', None, None)),
      ("3d_mesh2_x_y_none",        (1, 1, 4), P('x', 'y', None)),
      ("3d_mesh2_xy_none",         (1, 1, 4), P(('x', 'y'), None)),
  )
  def test_is_fully_replicated_named_sharding(self, mesh_shape, pspec):
    if len(mesh_shape) == 2:
      axis_names = ('x', 'y')
    elif len(mesh_shape) == 3:
      axis_names = ('x', 'y', 'z')
    else:
      axis_names = ('x',)
    mesh = jtu.create_global_mesh(mesh_shape, axis_names)
    mps = jax.sharding.NamedSharding(mesh, pspec)
    shape = (8, 2, 4)
    mps_op_sharding = mps._to_xla_hlo_sharding(len(shape))
    ops_ifr = op_shardings.is_op_sharding_replicated(mps_op_sharding)
    self.assertEqual(mps.is_fully_replicated, ops_ifr)

    ps = _op_sharding_to_pos_sharding(mps_op_sharding, mps._device_assignment)
    self.assertEqual(ps.is_fully_replicated,
                     op_shardings.is_op_sharding_replicated(
                         ps._to_xla_hlo_sharding(len(shape))))

  def test_devices_sharding_respects_init_mesh_shape(self):
    value_shape = (8, 4)

    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mps = jax.sharding.NamedSharding(mesh, P('x', 'y'))

    devices_sharding = jax.sharding.PositionalSharding(mesh.devices)

    op1 = mps._to_xla_hlo_sharding(len(value_shape))
    op2 = devices_sharding._to_xla_hlo_sharding(len(value_shape))

    self.assertEqual(mps.shard_shape(value_shape),
                     devices_sharding.shard_shape(value_shape))
    self.assertTrue(op_shardings.are_op_shardings_equal(op1, op2))

  def test_pmap_sharding_repr(self):
    if jax.device_count() < 2:
      self.skipTest('Test needs >= 2 devices.')
    out = jax.pmap(lambda x: x)(jnp.arange(2.))
    str(out.sharding)  # doesn't crash
    repr(out.sharding)  # doesn't crash

  def test_positional_sharding_repr(self):
    if jax.device_count() < 2:
      self.skipTest('Test needs >= 2 devices.')
    s = jax.sharding.PositionalSharding(jax.devices()).reshape(jax.device_count(), 1)
    repr(s)  # doesn't crash
    str(s)  # doesn't crash

  @parameterized.named_parameters(
      ('sharded_dim_0', (4, 2), 0),
      ('sharded_dim_1_0', (4, 2), 1),
      ('sharded_dim_2', (4, 2, 4), 2),
      ('sharded_dim_1_1', (2, 4), 1)
  )
  def test_default_pmap_sharding(self, shape, sharded_dim):
    if jax.device_count() < 4:
      self.skipTest('Test needs >= 4 devices.')
    ps = jax.sharding.PmapSharding.default(shape, sharded_dim)

    inp = jnp.arange(math.prod(shape)).reshape(shape)
    compiled = jax.pmap(lambda x: x, in_axes=sharded_dim).lower(inp).compile()
    pmap_in_sharding, = compiled._executable.unsafe_call.in_handler.in_shardings

    self.assertEqual(ps._device_assignment, pmap_in_sharding._device_assignment)
    self.assertEqual(ps.sharding_spec, pmap_in_sharding.sharding_spec)

  def test_default_pmap_sharding_with_devices(self):
    if jax.device_count() < 4:
      self.skipTest('Test needs >= 4 devices.')

    devs = jax.devices()
    new_order = (devs[0], devs[3], devs[2], devs[1])
    ps = jax.sharding.PmapSharding.default((4, 2), devices=new_order)
    self.assertEqual(ps._device_assignment, new_order)

  def test_mesh_repr(self):
    mesh = jtu.create_global_mesh((1, 1), ('x', 'y'))
    mesh_repr = repr(mesh)
    self.assertIn('device_ids', mesh_repr)
    self.assertIn('axis_names', mesh_repr)

  def test_are_shardings_equivalent(self):
    mesh = jtu.create_global_mesh((1,), ('x'))
    mesh2 = jtu.create_global_mesh((2, 1), ('x', 'y'))

    s1 = jax.sharding.NamedSharding(mesh, P('x'))
    s2 = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    self.assertTrue(s1.is_equivalent_to(s2, 2))

    s3 = jax.pmap(lambda x: x)(jnp.arange(jax.device_count())).sharding
    s4 = jax.pmap(lambda x: x)(jnp.arange(jax.device_count())).sharding
    self.assertTrue(s3.is_equivalent_to(s4, 2))

    self.assertFalse(s1.is_equivalent_to(s3, 2))
    self.assertFalse(s2.is_equivalent_to(s3, 2))

    s5 = jax.sharding.NamedSharding(mesh2, P('x', 'y'))

    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.REPLICATED
    s6 = jax.sharding.GSPMDSharding([jax.devices()[0]], op1)

    s7 = jax.sharding.GSPMDSharding(jax.devices(), op1)

    # The OpSharding is replicated but the Sharding itself are on different
    # devices.
    self.assertFalse(s6.is_equivalent_to(s7, 2))

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.OTHER
    op2.tile_assignment_devices = [0, 1]
    op2.tile_assignment_dimensions = [2, 1]
    s8 = jax.sharding.GSPMDSharding(list(mesh2.devices.flat), op2)

    self.assertTrue(s1.is_equivalent_to(s6, 2))
    self.assertTrue(s5.is_equivalent_to(s8, 2))
    self.assertFalse(s5.is_equivalent_to(s2, 2))

    s9 = jax.sharding.NamedSharding(mesh2, P('y'))

    op3 = xc.OpSharding()
    op3.type = xc.OpSharding.Type.OTHER
    op3.tile_assignment_devices = [0, 1]
    op3.tile_assignment_dimensions = [1, 1, 2]
    op3.replicate_on_last_tile_dim = True
    s10 = jax.sharding.GSPMDSharding(list(mesh2.devices.flat), op3)

    self.assertTrue(s9.is_equivalent_to(s10, 2))

  def test_devices_indices_map_good_error_message(self):
    shape = (1, 2)
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    s = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    with self.assertRaisesRegex(
        ValueError,
        "Sharding.*implies that array axis 0 is partitioned 2 times, but the "
        "dimension size is 1"):
      s.devices_indices_map(shape)

  def test_scalar_input_wrong_pspec(self):
    mesh = jtu.create_global_mesh((1, ), ('x'))
    shape = ()
    s = jax.sharding.NamedSharding(mesh, P('x'))
    with self.assertRaisesRegex(
        ValueError,
        r"For scalars the PartitionSpec should be P()"):
      s.check_compatible_aval(shape)

  def test_mesh_caching_during_construction(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Requires >=2 devices")
    mesh1 = jax.sharding.Mesh(jax.devices(), 'x')
    mesh2 = jax.sharding.Mesh(jax.devices(), 'x')

    self.assertIs(mesh1, mesh2)

  def test_mesh_str(self):
    mesh = jtu.create_global_mesh((2, 2, 2), ('x', 'y', 'z'))
    self.assertEqual(str(mesh), "Mesh('x': 2, 'y': 2, 'z': 2)")

  def test_make_array_from_callback_error(self):
    mesh_shape = (2, 3)
    global_shape = tuple(np.square(mesh_shape))
    mesh = jtu.create_global_mesh(mesh_shape, ('x', 'y'))
    pspec = P('x', 'y')
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    n = math.prod(global_shape)
    global_x = jnp.arange(n).astype('uint32').reshape(global_shape)

    def f(arr):
      return array.make_array_from_callback(arr.shape, sharding, lambda i: arr[i])

    out = f(global_x)
    self.assertEqual(out.shape, global_shape)

    msg = "jax.make_array_from_callback cannot be called within a traced context"
    with self.assertRaisesRegex(jax.errors.UnexpectedTracerError, msg):
      jax.jit(f)(global_x)

  def test_make_array_from_single_device_arrays_error(self):
    x = jnp.arange(10)
    sharding = x.sharding

    def f(x):
      return jax.make_array_from_single_device_arrays(x.shape, sharding, [x])

    msg = "jax.make_array_from_single_device_arrays requires a list of concrete arrays"
    with self.assertRaisesRegex(ValueError, msg):
      jax.jit(f)(x)

  def test_make_array_from_single_device_arrays_bad_inputs(self):
    x = jnp.arange(10)
    mesh = jtu.create_global_mesh((2,), ('x',))
    s = jax.sharding.NamedSharding(mesh, P('x'))
    x = jax.device_put(x, s)

    msg = ("When making an array from single-device arrays the input arrays "
           "must have one shard each. An argument array had 2 shard\\(s\\).")
    with self.assertRaisesRegex(ValueError, msg):
      jax.make_array_from_single_device_arrays(x.shape, s, [x, x])


  def test_gspmd_sharding_hash_eq(self):
    mesh = jtu.create_global_mesh((1, 1, 1), ('x', 'y', 'z'))
    ns = NamedSharding(mesh, P('x', 'y', 'z'))

    x1 = GSPMDSharding(mesh._flat_devices_tuple, ns._to_xla_hlo_sharding(3))
    x2 = GSPMDSharding.get_replicated(mesh._flat_devices_tuple)

    self.assertEqual(x1, x2)
    self.assertEqual(hash(x1), hash(x2))


class RngShardingTest(jtu.JaxTestCase):
  # tests that the PRNGs are automatically sharded as expected

  @parameterized.named_parameters(("3", 3), ("4", 4), ("5", 5))
  @jtu.skip_on_devices("gpu")
  def test_random_bits_is_pure_map_1d(self, num_devices):
    @jax.jit
    def f(x):
      bits = prng.threefry_random_bits(jnp.array([0, 0], dtype='uint32'),
                                       32, x.shape)
      return bits + x

    mesh = jtu.create_global_mesh((num_devices,), ('x',))
    s = jax.sharding.NamedSharding(mesh, P('x'))

    n = num_devices ** 2
    global_x = jnp.arange(n).astype('uint32')
    x = array.make_array_from_callback(global_x.shape, s, lambda i: global_x[i])

    # check computation is fully partitioned and without any communication
    with jax.threefry_partitionable(True):
      unopt_txt = f.lower(x).as_text(dialect='hlo')
      opt_txt = f.lower(x).compile().as_text()
      self.assertIn(   f'[{n}]', unopt_txt)
      self.assertNotIn(f'[{n}]', opt_txt)
      self.assertNotIn('all-reduce', opt_txt)
      self.assertNotIn('collective-permute', opt_txt)

      # check against single-device reference
      y = f(x)
      y_ref1 = f(jax.device_put(x, jax.devices()[0]))
      self.assertArraysEqual(y, y_ref1)

  @parameterized.named_parameters(
      {"testcase_name": f"_{mesh_shape}_{pspec}",
       "mesh_shape": mesh_shape, "pspec": pspec}
      for mesh_shape in [(3, 2), (4, 2), (2, 3)]
      for pspec in [P('x', None), P(None, 'y'), P('x', 'y')])
  @jtu.skip_on_devices("gpu")
  def test_random_bits_is_pure_map_2d(self, mesh_shape, pspec):
    @jax.jit
    def f(x):
      bits = prng.threefry_random_bits(jnp.array([0, 0], dtype='uint32'),
                                       32, x.shape)
      return bits + x

    global_shape = tuple(np.square(mesh_shape))

    mesh = jtu.create_global_mesh(mesh_shape, ('x', 'y'))
    s = jax.sharding.NamedSharding(mesh, pspec)

    n = math.prod(global_shape)
    global_x = np.arange(n).astype('uint32').reshape(global_shape)
    x = array.make_array_from_callback(global_x.shape, s, lambda i: global_x[i])

    # check computation is fully partitioned and without any communication
    with jax.threefry_partitionable(True):
      unopt_txt = f.lower(x).as_text(dialect='hlo')
      opt_txt = f.lower(x).compile().as_text()
      global_shape_fmt = ','.join(str(x) for x in global_shape)
      self.assertIn(   f'[{global_shape_fmt}]', unopt_txt)
      self.assertNotIn(f'[{global_shape_fmt}]', opt_txt)
      self.assertNotIn('all-reduce', opt_txt)
      self.assertNotIn('collective-permute', opt_txt)

      # check against single-device reference
      y = f(x)
      y_ref1 = f(jax.device_put(x, jax.devices()[0]))
      self.assertArraysEqual(y, y_ref1)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
