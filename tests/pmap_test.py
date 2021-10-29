# Copyright 2018 Google LLC
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


from concurrent.futures import ThreadPoolExecutor
from functools import partial
import itertools as it
import gc
import os
from random import shuffle
from typing import Optional, cast
import unittest
from unittest import SkipTest
import warnings
import weakref

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax import tree_util
from jax import lax
from jax._src.lax import parallel
from jax._src import api as src_api
from jax import random
from jax.core import ShapedArray
from jax import (pmap, soft_pmap, jit, vmap, jvp, grad, make_jaxpr,
                 linearize, device_put)
import jax._src.lib
from jax._src.lib import xla_bridge
from jax._src.util import prod, safe_map
from jax.interpreters import pxla
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()

prev_xla_flags = None

compatible_shapes = [[(3,)], [(3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]

def all_bdims(*shapes, pmap):
  bdims = (it.chain([cast(Optional[int], None)], range(len(shape) + 1))
           for shape in shapes)
  return (t for t in it.product(*bdims) if not all(e is None for e in t))

def out_bdims(shape, pmap):
  return (d[0] for d in all_bdims(shape, pmap=pmap) if d[0] is not None)


def add_bdim(bdim_size, bdim, shape):
  shape = list(shape)
  if bdim is not None:
    shape.insert(bdim, bdim_size)
  return tuple(shape)

def slicer(x, bdim):
  if bdim is None:
    return lambda _: x
  else:
    return lambda i: lax.index_in_dim(x, i, bdim, keepdims=False)

def args_slicer(args, bdims):
  slicers = safe_map(slicer, args, bdims)
  return lambda i: [sl(i) for sl in slicers]

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

ignore_jit_of_pmap_warning = partial(
  jtu.ignore_warning, message=".*jit-of-pmap.*")

ignore_slow_all_to_all_warning = partial(
  jtu.ignore_warning, message="all_to_all.*expect significant slowdowns.*")

ignore_xmap_warning = partial(
  jtu.ignore_warning, message=".*is an experimental.*")


class PythonPmapTest(jtu.JaxTestCase):

  @property
  def pmap(self):
    return src_api._python_pmap

  def testDeviceBufferToArray(self):
    sda = self.pmap(lambda x: x)(jnp.ones((jax.device_count(), 2)))
    buf = sda.device_buffers[-1]

    view = jnp.array(buf, copy=False)
    self.assertArraysEqual(sda[-1], view)
    self.assertEqual(buf.device(), view.device())
    self.assertEqual(buf.unsafe_buffer_pointer(), view.unsafe_buffer_pointer())

    copy = jnp.array(buf, copy=True)
    self.assertArraysEqual(sda[-1], copy)
    self.assertEqual(buf.device(), copy.device())
    self.assertNotEqual(buf.unsafe_buffer_pointer(), copy.unsafe_buffer_pointer())

  def _getMeshShape(self, device_mesh_shape):
    device_count = jax.device_count()
    if any(size == -1 for size in device_mesh_shape):
      try:
        return np.arange(device_count).reshape(device_mesh_shape).shape
      except ValueError as err:
        msg = "device mesh shape {} not compatible with device count {}"
        raise SkipTest(msg.format(device_mesh_shape, device_count)) from err
    else:
      if device_count % prod(device_mesh_shape):
        msg = "device mesh size {} does not divide available device count {}"
        raise SkipTest(msg.format(prod(device_mesh_shape), device_count))
      else:
        return device_mesh_shape

  def testBasic(self):
    f = self.pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.sum(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMean(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.broadcast_to(np.mean(x, 0), x.shape)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": f"_dtype={np.dtype(dtype).name}", "dtype": dtype}
      for dtype in [np.float32, np.bool_])
  def testGather(self, dtype):
    f = self.pmap(lambda x: lax.all_gather(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    expected = np.array([x] * jax.device_count())
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGatherTiled(self):
    f = self.pmap(lambda x: lax.all_gather(x, 'i', tiled=True), axis_name='i')

    device_count = jax.device_count()
    shape = (device_count, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = np.array([x] * device_count).reshape(device_count, -1)
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReduceScatter(self):
    f = self.pmap(lambda x: lax.psum_scatter(x, 'i'), axis_name='i')

    device_count = jax.device_count()
    shape = (device_count, device_count)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = np.sum(x, axis=0)
    ans = f(x)
    for i, actual in enumerate(ans):
      self.assertAllClose(actual, expected[i])

  def testReduceScatterTiled(self):
    f = self.pmap(lambda x: lax.psum_scatter(x, 'i', tiled=True), axis_name='i')

    device_count = jax.device_count()
    shape = (device_count, 4 * device_count)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = np.sum(x, axis=0)
    ans = f(x)
    scatter_len = len(expected) // device_count
    for i, actual in enumerate(ans):
      self.assertAllClose(actual,
                          expected[i * scatter_len:(i + 1) * scatter_len])

  def testReduceScatterReplicaGroupsTiled(self):
    replicas = jax.device_count()
    if replicas % 2 != 0:
      raise SkipTest
    axis_index_groups = [[i for i in range(jax.device_count()) if i % 2 == 0],
                         [i for i in range(jax.device_count()) if i % 2 != 0]]
    f = lambda x: lax.psum_scatter(
        x, 'i', axis_index_groups=axis_index_groups, tiled=True)
    f = self.pmap(f, axis_name='i')

    shape = (replicas, 4 * replicas)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    ans = f(x)

    group_1_result = np.sum(x[0::2,:], axis=0)
    group_2_result = np.sum(x[1::2,:], axis=0)
    # the result is scattered over (replicas // 2) devices
    scatter_len = len(group_1_result) * 2 // replicas

    for i, actual in enumerate(ans):
      expected = group_1_result if i % 2 == 0 else group_2_result
      self.assertAllClose(
          actual, expected[i // 2 * scatter_len:(i // 2 + 1) * scatter_len])

  @ignore_slow_all_to_all_warning()
  def testTrees(self):
    ptranspose = lambda x, axis_name: lax.all_to_all(x, axis_name, 0, 0)
    def protate(x, axis_name):
      n = lax.psum(1, axis_name)
      return lax.ppermute(x, axis_name, [(i, (i + 1) % n) for i in range(n)])

    tree_f = lambda f: partial(tree_util.tree_map, f)
    jax_f = lambda p: self.pmap(lambda x: p(x, 'i'), 'i')
    np_f = lambda p: tree_f(lambda x: np.broadcast_to(p(x, 0), x.shape))
    np_transpose = tree_f(np.transpose)
    np_rotate = tree_f(lambda x: np.concatenate([x[-1:], x[:-1]]))

    n = jax.device_count()
    x = {'a': np.arange(1 * n * n, 2 * n * n).reshape([n, n]),
         'b': np.arange(2 * n * n, 3 * n * n).reshape([n, n]),
         'c': np.arange(4 * n * n, 5 * n * n).reshape([n, n])}

    assert_allclose = partial(tree_util.tree_multimap,
                              partial(self.assertAllClose, check_dtypes=False))
    assert_allclose(jax_f(lax.pmax)(x), np_f(np.max)(x))
    assert_allclose(jax_f(lax.pmin)(x), np_f(np.min)(x))
    assert_allclose(jax_f(lax.psum)(x), np_f(np.sum)(x))
    assert_allclose(jax_f(lax.pmean)(x), np_f(np.mean)(x))
    assert_allclose(jax_f(ptranspose)(x), np_transpose(x))
    assert_allclose(jax_f(protate)(x), np_rotate(x))

  def testCollectivesWithTreesOfDifferentDtypes(self):
    n = len(jax.devices())
    x = {'a': np.arange(1 * n * n, 2 * n * n, dtype=np.float32).reshape([n, n]),
         'b': np.arange(2 * n * n, 3 * n * n, dtype=np.int32).reshape([n, n]),
         'c': np.arange(4 * n * n, 5 * n * n, dtype=np.float32).reshape([n, n]),
         'd': np.arange(6 * n * n, 7 * n * n, dtype=np.int32).reshape([n, n])}
    tree_f = lambda f: partial(tree_util.tree_map, f)
    jax_f = lambda p: self.pmap(lambda x: p(x, 'i'), 'i')
    np_f = lambda p: tree_f(lambda x: np.broadcast_to(p(x, 0), x.shape))
    assert_allclose = partial(tree_util.tree_multimap,
                              partial(self.assertAllClose, check_dtypes=False))
    assert_allclose(jax_f(lax.pmax)(x), np_f(np.max)(x))
    assert_allclose(jax_f(lax.pmin)(x), np_f(np.min)(x))
    assert_allclose(jax_f(lax.psum)(x), np_f(np.sum)(x))
    assert_allclose(jax_f(lax.pmean)(x), np_f(np.mean)(x))

  def testComplexPsum(self):
    f = self.pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4 * 2)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape).view(np.complex64)
    expected = x - np.sum(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}",
      "split_axis": split_axis, "concat_axis": concat_axis}
      for split_axis, concat_axis in it.product(range(2), range(2)))
  @ignore_slow_all_to_all_warning()
  def testAllToAll(self, split_axis, concat_axis):
    pmap_in_axis = 0
    shape = (jax.device_count(),) * 3
    x = np.arange(np.prod(shape)).reshape(shape)

    @partial(self.pmap, axis_name='i')
    def f(x):
      return lax.all_to_all(x, 'i', split_axis, concat_axis)
    y = f(x)
    if pmap_in_axis <= split_axis:
      split_axis += 1
    ref = jnp.moveaxis(x, (pmap_in_axis, split_axis),
                          (concat_axis + 1, 0))
    self.assertAllClose(y, ref)

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}",
       "split_axis": split_axis, "concat_axis": concat_axis}
      for split_axis, concat_axis in it.product(range(2), range(2)))
  @ignore_slow_all_to_all_warning()
  def testAllToAllSplitAxis(self, split_axis, concat_axis):
    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")
    pmap_in_axis = 0
    shape = (4, 4, 4)
    x = np.arange(np.prod(shape)).reshape(shape)

    @partial(self.pmap, axis_name='i')
    @partial(self.pmap, axis_name='j')
    def f(x):
      return lax.all_to_all(x, ('i', 'j'), split_axis, concat_axis)

    unroll_shape = (2, 2, *shape[1:])
    x_unroll = x.reshape(unroll_shape)
    y_unroll = f(x_unroll)
    y = y_unroll.reshape(shape)

    if pmap_in_axis <= split_axis:
      split_axis += 1
    ref = jnp.moveaxis(x, (pmap_in_axis, split_axis),
                          (concat_axis + 1, 0))
    self.assertAllClose(y, ref)

  def testNestedBasic(self):
    f = lambda x: lax.psum(lax.psum(x, 'i'), 'j')
    f = self.pmap(self.pmap(f, 'i'), 'j')

    def sum_and_broadcast(x, axis):
      return np.repeat(np.sum(x, axis, keepdims=True), x.shape[axis], axis)

    shape = (jax.device_count(), 1, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)
    expected = sum_and_broadcast(sum_and_broadcast(x, 0), 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMismatchedAxisSizes(self):
    n = jax.device_count()
    f = self.pmap(lambda x, y: x + y)
    self.assertRaisesRegex(
        ValueError,
        "pmap got inconsistent sizes for array axes to be mapped",
        lambda: f(np.random.randn(n), np.random.randn(n - 1)))

  @parameterized.named_parameters(
      {"testcase_name": "_mesh={}".format(device_mesh_shape).replace(" ", ""),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testNestedShardingAndStacking(self, device_mesh_shape):
    mesh_shape = self._getMeshShape(device_mesh_shape)

    f = lambda x: x
    f = self.pmap(self.pmap(f, 'i'), 'j')

    shape = mesh_shape + (4,)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)
    expected = x
    self.assertEqual(ans.shape, expected.shape)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPartiallyMapped(self):
    f = self.pmap(lambda x, y: x, in_axes=(None, 0))
    g = self.pmap(lambda x, y: x - lax.psum(y, 'i'), axis_name='i', in_axes=(None, 0))

    mesh_shape = (jax.device_count(),)
    shape = mesh_shape + (4,)
    x = np.array(3., dtype=np.float32)
    y = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    f_expected = np.broadcast_to(x, mesh_shape)
    f_ans = f(x, y)
    self.assertAllClose(f_ans, f_expected)
    self.assertIsInstance(f_ans, pxla.ShardedDeviceArray)
    # the output is actually replicated (has the same values in each device buffer)
    # but out_axes is implicitly 0, so we shouldn't have replication in the
    # sharding spec.
    self.assertEmpty([a for a in f_ans.sharding_spec.mesh_mapping
                      if isinstance(a, pxla.Replicated)])

    g_expected = np.broadcast_to(x - np.sum(y, 0, keepdims=True), shape)
    g_ans = g(x, y)
    self.assertAllClose(g_ans, g_expected)
    self.assertIsInstance(g_ans, pxla.ShardedDeviceArray)
    self.assertEmpty([a for a in g_ans.sharding_spec.mesh_mapping
                      if isinstance(a, pxla.Replicated)])

  def testReplicate(self):
    base = np.array([3.,4.], dtype=np.float32)
    num_devices = jax.device_count()
    replicated = pxla.replicate(base, num_devices, num_devices, in_axis=None)
    self.assertAllClose(base, replicated)
    self.assertEmpty([a for a in replicated.sharding_spec.mesh_mapping
                      if not isinstance(a, pxla.Replicated)])

  @parameterized.named_parameters(
      {"testcase_name": "_mesh={}".format(device_mesh_shape).replace(" ", ""),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testPartiallyMappedNested(self, device_mesh_shape):
    mesh_shape = self._getMeshShape(device_mesh_shape)

    f = self.pmap(lambda x, y: x - lax.psum(y, 'i'), axis_name='i', in_axes=(None, 0))
    f = self.pmap(f, axis_name='j', in_axes=(None, 0))

    x = 3.
    y = np.arange(prod(mesh_shape), dtype=np.float32).reshape(mesh_shape)
    expected = np.broadcast_to(x - np.sum(y, 1, keepdims=True), mesh_shape)

    ans = f(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testJvpAndPartialEval(self):
    @partial(self.pmap, axis_name='i')
    def f(x):
      return jnp.sin(x)

    def splitjvp(x):
      _, jvp = linearize(f, x)
      return jvp(jnp.ones_like(x))

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = np.cos(x)

    ans = splitjvp(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    make_jaxpr(splitjvp)(x)  # doesn't crash

  def testGradBasic(self):
    @partial(self.pmap, axis_name='i')
    def f(x):
      return jnp.sin(x)

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(jnp.sin(x)))(x)
    expected = grad(lambda x: jnp.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGradOfPsum(self):
    @partial(self.pmap, axis_name='i')
    def f(x):
      return lax.psum(x, axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    jtu.check_grads(f, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2, eps=1.)

  def testGradOfJvp(self):
    @partial(self.pmap, axis_name='i')
    def f(x):
      return jnp.sin(x)

    def splitjvp(x):
      _, jvp = linearize(f, x)
      return jvp(jnp.ones_like(x))

    fun = lambda x: jnp.sum(jvp(jnp.sin, (x,), (jnp.ones_like(x),))[1])

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(splitjvp(x)))(x)
    expected = grad(fun)(x)
    self.assertAllClose(ans, expected)

  def testTwoArgsGrad(self):
    def f(x, y):
      return lax.psum(5. * jnp.cos(x) * jnp.sin(y), 'i')
    f = self.pmap(f, 'i')

    def g(x, y):
      tot = jnp.sum(5. * jnp.cos(x) * jnp.sin(y))
      return tot * jnp.ones_like(x)  # broadcast to map like pjit does

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    y = 4 + x
    ans = grad(lambda x, y: jnp.sum(g(x, y)))(x, y)
    expected = grad(lambda x, y: jnp.sum(g(x, y)))(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_mesh={}".format(device_mesh_shape).replace(" ", ""),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testNestedWithClosure(self, device_mesh_shape):
    mesh_shape = self._getMeshShape(device_mesh_shape)

    @partial(self.pmap, axis_name='i')
    def test_fun(x):
      y = jnp.sum(jnp.sin(x))

      @partial(self.pmap, axis_name='j')
      def g(z):
        return 3. * jnp.exp(jnp.sin(x).sum() * jnp.cos(y) * jnp.tan(z))

      return grad(lambda w: jnp.sum(g(w)))(x)

    @vmap
    def baseline_fun(x):
      y = jnp.sum(jnp.sin(x))

      @vmap
      def g(z):
        return 3. * jnp.exp(jnp.sin(x).sum() * jnp.cos(y) * jnp.tan(z))

      return grad(lambda w: jnp.sum(g(w)))(x)

    shape = mesh_shape + (4,)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(test_fun(x)))(x)
    expected = grad(lambda x: jnp.sum(baseline_fun(x)))(x)
    self.assertAllClose(ans, expected, atol=1e-3)

  def testShardedDeviceArrays(self):
    f = lambda x: 2 * x
    f = self.pmap(f, axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    # test that we can pass in and out ShardedDeviceArrays
    y = f(x)
    self.assertIsInstance(y, jnp.ndarray)
    self.assertIsInstance(y, pxla.ShardedDeviceArray)
    self.assertIsInstance(y, jax.interpreters.xla.DeviceArray)
    self.assertNotIsInstance(y, np.ndarray)
    self.assertAllClose(y, 2 * x, check_dtypes=False)
    z = f(y)
    self.assertIsInstance(z, pxla.ShardedDeviceArray)
    self.assertIsInstance(z, jax.interpreters.xla.DeviceArray)
    self.assertNotIsInstance(z, np.ndarray)
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can pass in a regular DeviceArray
    y = f(device_put(x))
    self.assertIsInstance(y, pxla.ShardedDeviceArray)
    self.assertAllClose(y, 2 * x, check_dtypes=False)

    # test that we can pass a ShardedDeviceArray to a regular jit computation
    z = y + y
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can handle device movement on dispatch
    y = pxla.make_sharded_device_array(y.aval, y.sharding_spec,
                                       y.device_buffers[::-1])
    z = f(y)
    self.assertAllClose(z, 2 * 2 * x[::-1], check_dtypes=False)

    # test that the repr doesn't crash
    repr(z)

  # Tests edge cases in lax._reshape_sharded_device_array
  @parameterized.named_parameters(
      {"testcase_name": "_in={}_out={}".format(in_shape, out_shape)
       .replace(" ", ""),
       "in_shape": in_shape, "out_shape": out_shape}
      for in_shape, out_shape in [
          [(1,1), (1,)], [(1,), (1,1)], [(1,), ()], [(4,7), (2,2,7)]
      ])
  def testShardedDeviceArrayReshape(self, in_shape, out_shape):
    if jax.device_count() < max(in_shape[:1] + out_shape[:1]):
      raise SkipTest("not enough devices")

    x = np.arange(prod(in_shape)).reshape(in_shape)
    sharded_x = self.pmap(lambda x: x)(x)
    self.assertAllClose(sharded_x.reshape(out_shape), x.reshape(out_shape),
                        check_dtypes=False)

  def testPsumMultiple(self):
    f = lambda x: lax.psum(x, ('i', 'j'))
    f = self.pmap(self.pmap(f, 'i'), 'j')

    def sum_and_broadcast(x, axis):
      return np.repeat(np.sum(x, axis, keepdims=True), x.shape[axis], axis)

    device_count = jax.device_count()
    num_pairs, ragged = divmod(device_count, 2)
    if num_pairs > 1 and not ragged:
      shape = (num_pairs, 2, 4)
    else:
      shape = (device_count, 1, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)
    expected = sum_and_broadcast(sum_and_broadcast(x, 0), 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPsumConstantReplicaGroups(self):
    replicas = jax.device_count()
    if replicas % 2 != 0:
      raise SkipTest
    axis_index_groups = np.arange(replicas).reshape(
      2, replicas // 2).tolist()
    f = lambda x: x - lax.psum(2., 'i', axis_index_groups=axis_index_groups)
    f = self.pmap(f, 'i')

    shape = (replicas, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected_psum = 2. * replicas // 2
    expected = x - expected_psum

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("tpu")
  def testPsumUnevenReplicaGroups(self):
    replicas = jax.device_count()
    if replicas <= 2:
      raise SkipTest("Test expected devices greater than 2.")
    axis_index_groups = [[0,1], np.arange(2,replicas)]
    f = lambda x: x - lax.psum(x, 'i', axis_index_groups=axis_index_groups)
    f = self.pmap(f, 'i')

    shape = (replicas, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    def sum_helper(a):
      return np.broadcast_to(a.sum(0, keepdims=True),
                              (len(a), x.shape[1]))
    expected_psum_1 = sum_helper(x[0:2])
    expected_psum_2 = sum_helper(x[2:])
    expected_psum = np.concatenate([expected_psum_1, expected_psum_2], 0)
    expected = x - expected_psum

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPsumReplicaGroups(self):
    replicas = jax.device_count()
    if replicas % 2 != 0:
      raise SkipTest
    axis_index_groups = np.arange(replicas).reshape(
      2, replicas // 2).tolist()
    f = lambda x: x - lax.psum(x, 'i', axis_index_groups=axis_index_groups)
    f = self.pmap(f, 'i')

    shape = (replicas, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    def sum_helper(a):
      return np.broadcast_to(a.sum(0, keepdims=True),
                              (replicas // 2, x.shape[1]))
    expected_psum_1 = sum_helper(x[:replicas // 2])
    expected_psum_2 = sum_helper(x[replicas // 2:])
    expected_psum = np.concatenate([expected_psum_1, expected_psum_2], 0)
    expected = x - expected_psum

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGatherReplicaGroups(self):
    replicas = jax.device_count()
    if replicas % 2 != 0:
      raise SkipTest("Test expected an even number of devices greater than 1.")

    axis_index_groups = np.arange(replicas, dtype=np.int32)
    axis_index_groups = axis_index_groups.reshape((replicas // 2, 2)).T
    axis_index_groups = axis_index_groups.tolist()

    f = lambda x: lax.all_gather(x, 'i', axis_index_groups=axis_index_groups)
    f = self.pmap(f, 'i')

    shape = (replicas, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)

    group_1_result = x[0::2]
    group_2_result = x[1::2]
    expected = np.empty((replicas, replicas // 2, x.shape[1]))
    expected[0::2] = group_1_result
    expected[1::2] = group_2_result

    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGatherReplicaGroupsInterleaved(self):
    replicas = jax.device_count()
    if replicas % 2 != 0:
      raise SkipTest("Test expected an even number of devices greater than 1.")

    indexes = np.arange(replicas)
    indexes = np.concatenate([indexes[::2], indexes[1::2]])
    axis_index_groups = indexes.reshape(2, replicas // 2).tolist()

    f = lambda x: lax.all_gather(x, 'i', axis_index_groups=axis_index_groups)
    f = self.pmap(f, 'i')

    shape = (replicas, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)

    expected = np.zeros((replicas, replicas // 2, x.shape[1]))
    expected[::2] = x[::2]
    expected[1::2] = x[1::2]

    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_slow_all_to_all_warning()
  def testGradOfGather(self):
    @partial(self.pmap, axis_name='i')
    def f(x):
      return lax.all_gather(x, axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    jtu.check_grads(f, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2, eps=1.)

  def testNestedPmapReplicaGroups(self):
    replicas = jax.device_count()
    if replicas % 4 != 0:
      raise SkipTest
    axis_index_groups = np.arange(replicas // 2).reshape(
        2, replicas // 4).tolist()
    f = lambda x: x - lax.psum(x, 'i', axis_index_groups=axis_index_groups)
    f1 = self.pmap(self.pmap(f, 'i'), 'j')
    f2 = self.pmap(lambda x: self.pmap(f, 'i')(x) + 1., 'j')  # "imperfectly nested" case
    f3 = self.pmap(self.pmap(f, 'j'), 'i')

    shape = (2, replicas // 2, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    def sum_helper_f1(a):
      return np.broadcast_to(a.sum(1, keepdims=True),
                              (shape[0], shape[1] // 2, shape[2]))
    expected_psum_1 = sum_helper_f1(x[:, :replicas // 4])
    expected_psum_2 = sum_helper_f1(x[:, replicas // 4:])
    expected_psum = np.concatenate([expected_psum_1, expected_psum_2], 1)
    expected = x - expected_psum
    ans = f1(x)
    self.assertAllClose(ans, expected)

    expected = x - expected_psum + 1.
    ans = f2(x)
    self.assertAllClose(ans, expected)

    shape = (replicas // 2, 2, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    def sum_helper_f3(a):
      return np.broadcast_to(a.sum(0, keepdims=True),
                              (shape[0] // 2, shape[1], shape[2]))
    expected_psum_1 = sum_helper_f3(x[:replicas // 4])
    expected_psum_2 = sum_helper_f3(x[replicas // 4:])
    expected_psum = np.concatenate([expected_psum_1, expected_psum_2], 0)
    expected = x - expected_psum
    ans = f3(x)
    self.assertAllClose(ans, expected)

  def testAxisGroups(self):
    axis_env = xla.AxisEnv(8, ('i', 'j'), (4, 2))
    groups = xla.axis_groups(axis_env, 'i')
    self.assertEqual(groups, ((0, 2, 4, 6), (1, 3, 5, 7)))

    groups = xla.axis_groups(axis_env, 'j')
    self.assertEqual(groups, ((0, 1), (2, 3), (4, 5), (6, 7)))

    groups = xla.axis_groups(axis_env, ('i', 'j'))
    self.assertEqual(groups, ((0, 1, 2, 3, 4, 5, 6, 7,),))

    groups = xla.axis_groups(axis_env, ('j', 'i'))
    self.assertEqual(len(groups), 1)
    self.assertEqual((tuple(sorted(groups[0])),),
                     ((0, 1, 2, 3, 4, 5, 6, 7,),))  # order doesn't matter

  def testCollectivePermute(self):
    device_count = jax.device_count()
    rotation = [(i, (i + 1) % device_count) for i in range(device_count)]
    f = lambda x: lax.ppermute(x, perm=rotation, axis_name='i')
    f = self.pmap(f, 'i')

    x = jnp.arange(4 * device_count).reshape((device_count, 4))
    ans = f(x)
    expected = np.roll(x, shift=1, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu")
  def testCollectivePermuteGrad(self):
    device_count = jax.device_count()
    shift_right = [(i, (i + 1)) for i in range(device_count - 1)]
    f = lambda x: lax.ppermute(x, perm=shift_right, axis_name='i')
    y = np.pi + np.arange(device_count, dtype=np.float32)
    g = lambda x: jnp.sum(y * self.pmap(f, 'i')(x))

    x = np.arange(device_count, dtype=np.float32)
    ans = grad(g)(x)
    expected = np.concatenate([np.pi + np.arange(1, device_count), [0]])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCollectivePermuteCyclicGrad(self):
    device_count = jax.device_count()
    shift_right = [(i, (i + 1) % device_count) for i in range(device_count)]
    f = lambda x: lax.ppermute(x, perm=shift_right, axis_name='i')
    y = np.pi + np.arange(device_count, dtype=np.float32)
    g = lambda x: jnp.sum(y * self.pmap(f, 'i')(x))

    x = np.arange(device_count, dtype=np.float32)

    ans = grad(g)(x)
    expected = np.roll(np.pi + np.arange(device_count), -1)
    self.assertAllClose(ans, expected, check_dtypes=False)

    jtu.check_grads(g, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2)

  def testCollectivePermuteCyclicWithPShuffle(self):
    device_count = jax.device_count()
    values = np.arange(device_count)
    shift_right = [(i - 1) % device_count for i in range(device_count)]
    f = lambda x: lax.pshuffle(x, perm=shift_right, axis_name='i')
    expected = np.roll(values, 1)
    ans = np.asarray(self.pmap(f, "i")(values))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPShuffleWithBadPerm(self):
    device_count = jax.device_count()
    bad_perm = list(range(device_count))
    bad_perm[0] = 1
    f = lambda x: lax.pshuffle(x, perm=bad_perm, axis_name='i')
    g = lambda: self.pmap(f, "i")(np.arange(device_count))
    self.assertRaisesRegex(
      ValueError,
      "`perm` does not represent a permutation: \\[1.*\\]", g)

  def testPpermuteWithZipObject(self):
    # https://github.com/google/jax/issues/1703
    num_devices = jax.device_count()
    perm = [num_devices - 1] + list(range(num_devices - 1))
    f = self.pmap(lambda x: lax.ppermute(x, "i", zip(perm, range(num_devices))), "i")
    result = f(jnp.arange(num_devices, dtype=jnp.float32))
    expected = jnp.asarray(perm, dtype=jnp.float32)
    self.assertAllClose(result, expected)

  def testRule30(self):
    # This is a test of collective_permute implementing a simple halo exchange
    # to run a rule 30 simulation: https://en.wikipedia.org/wiki/Rule_30
    # Halo exchange should be useful in spatially-sharded convolutions and in
    # other simulations.
    device_count = jax.device_count()

    def send_right(x, axis_name):
      left_perm = [(i, (i + 1) % device_count) for i in range(device_count)]
      return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

    def send_left(x, axis_name):
      left_perm = [((i + 1) % device_count, i) for i in range(device_count)]
      return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

    def update_board(board):
      left = board[:-2]
      right = board[2:]
      center = board[1:-1]
      return lax.bitwise_xor(left, lax.bitwise_or(center, right))

    @partial(self.pmap, axis_name='i')
    def step(board_slice):
      left, right = board_slice[:1], board_slice[-1:]
      right, left = send_left(left, 'i'), send_right(right, 'i')
      enlarged_board_slice = jnp.concatenate([left, board_slice, right])
      return update_board(enlarged_board_slice)

    board = np.zeros(40, dtype=bool)
    board[board.shape[0] // 2] = True
    reshaped_board = board.reshape((device_count, -1))

    boards = []
    def print_board(board):
      boards.append(''.join('*' if x else ' ' for x in board.ravel()))

    print_board(reshaped_board)
    for _ in range(20):
      reshaped_board = step(reshaped_board)
      print_board(reshaped_board)

    ans = '\n'.join(boards)
    expected = '\n'.join((
        '                    *                   ',
        '                   ***                  ',
        '                  **  *                 ',
        '                 ** ****                ',
        '                **  *   *               ',
        '               ** **** ***              ',
        '              **  *    *  *             ',
        '             ** ****  ******            ',
        '            **  *   ***     *           ',
        '           ** **** **  *   ***          ',
        '          **  *    * **** **  *         ',
        '         ** ****  ** *    * ****        ',
        '        **  *   ***  **  ** *   *       ',
        '       ** **** **  *** ***  ** ***      ',
        '      **  *    * ***   *  ***  *  *     ',
        '     ** ****  ** *  * *****  *******    ',
        '    **  *   ***  **** *    ***      *   ',
        '   ** **** **  ***    **  **  *    ***  ',
        '  **  *    * ***  *  ** *** ****  **  * ',
        ' ** ****  ** *  ******  *   *   *** ****',
        ' *  *   ***  ****     **** *** **   *   ',
    ))

    print(ans)
    self.assertEqual(ans, expected)

  def testReduceMax(self):
    f = self.pmap(lambda x: x - lax.pmax(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.max(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReduceMin(self):
    f = self.pmap(lambda x: x - lax.pmin(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.min(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDeviceCountError(self):
    device_count = jax.device_count()

    f = self.pmap(lambda x: x)
    x = jnp.arange(device_count + 1)
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

    f = self.pmap(lambda x: x)
    x = np.ones((device_count + 1, 10))
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

    f = self.pmap(lambda x: self.pmap(lambda x: x)(x))
    x = np.ones((device_count, 2, 10))
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

  def testPmapConstant(self):
    device_count = jax.device_count()
    f = self.pmap(lambda x: 3)
    x = jnp.arange(device_count)
    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): fix this
    expected = np.repeat(3, device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

    f = self.pmap(lambda x: (x, 3))
    x = np.arange(device_count)
    with jtu.assert_num_jit_and_pmap_compilations(1):
      _, ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPmapConstantDevices(self):
    if jax.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    devices = jax.devices()[:-1]
    shuffle(devices)
    f = self.pmap(lambda x: 3, devices=devices)
    x = jnp.arange(len(devices))
    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = np.repeat(3, len(devices))
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    self.assertEqual([b.device() for b in ans.device_buffers], devices)

  def testPmapConstantError(self):
    device_count = jax.device_count()
    f = self.pmap(lambda x: 3)
    x = jnp.arange(device_count + 1)
    self.assertRaisesRegex(
        ValueError,
        (r"compiling computation that requires \d+ logical devices, "
        r"but only \d+ XLA devices are available .*"),
        lambda: f(x))

    # TODO(mattjj): test error message with explicit devices
    # f = pmap(lambda x: 3, devices=[jax.devices()[0]])
    # x = jnp.arange(2)
    # self.assertRaisesRegex(
    #     ValueError, r"Cannot replicate across \d+ replicas because only \d+ "
    #     r"local devices are available.", lambda: f(x))

  def testNestedPmapConstant(self):
    if jax.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    f = self.pmap(self.pmap(lambda x: 3))
    shape = (2, jax.device_count() // 2, 3)
    x = jnp.arange(prod(shape)).reshape(shape)
    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = 3 * np.ones(shape[:2])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    expected_sharded = self.pmap(self.pmap(lambda x: x))(expected)
    self.assertEqual([b.device() for b in ans.device_buffers],
                     [b.device() for b in expected_sharded.device_buffers])

    f = self.pmap(self.pmap(lambda x: (x, 3)))
    x_sharded, ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)
    self.assertEqual([b.device() for b in ans.device_buffers],
                     [b.device() for b in x_sharded.device_buffers])

  @unittest.skip("Nested pmaps with devices not yet implemented")
  def testNestedPmapConstantDevices(self):
    if jax.device_count() < 6:
      raise SkipTest("this test requires >= 6 devices")

    devices = jax.devices()[:-2]
    shuffle(devices)
    f = self.pmap(self.pmap(lambda x: 3), devices=devices)
    shape = (2, len(devices) // 2, 3)
    x = jnp.arange(prod(shape)).reshape(shape)
    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = 3 * np.ones(shape[:2])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    expected_sharded = self.pmap(self.pmap(lambda x: x), devices=devices)(expected)
    self.assertEqual([b.device() for b in ans.device_buffers],
                     [b.device() for b in expected_sharded.device_buffers])

  def testNestedPmapConstantError(self):
    f = self.pmap(self.pmap(lambda x: 3))
    shape = (2, jax.device_count() // 2 + 1, 3)
    x = jnp.arange(prod(shape)).reshape(shape)
    self.assertRaisesRegex(
        ValueError,
        (r"compiling computation that requires \d+ logical devices, "
        r"but only \d+ XLA devices are available .*"),
        lambda: f(x))

    # TODO(mattjj): check error message with explicit devices
    # if jax.device_count() > 1:
    #   f = pmap(pmap(lambda x: 3), devices=jax.devices()[:-1])
    #   shape = (2, jax.device_count() // 2, 3)
    #   x = jnp.arange(prod(shape)).reshape(shape)
    #   self.assertRaisesRegex(
    #       ValueError,
    #       (r"compiling computation that requires \d+ replicas, "
    #        r"but only \d+ XLA devices are available"),
    #       lambda: f(x))

  def testCollectiveConstant(self):
    device_count = jax.device_count()
    f = self.pmap(lambda x: lax.psum(1, 'i'), 'i')
    x = jnp.arange(device_count)
    ans = f(x)
    expected = np.repeat(device_count, device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCollectiveConstantNested(self):
    device_count = jax.device_count()

    @partial(self.pmap, axis_name='i')
    def f(x):
      @partial(self.pmap, axis_name='j')
      def g(y):
        a = lax.psum(1, 'i')
        b = lax.psum(1, 'j')
        c = lax.psum(1, ('i', 'j'))
        return a, b, c
      return g(x)

    shape = (device_count, 1, 4)
    x = jnp.arange(prod(shape)).reshape(shape)
    a, b, c = f(x)

    self.assertEqual(a.shape, shape[:-1])
    self.assertEqual(b.shape, shape[:-1])
    self.assertEqual(c.shape, shape[:-1])

    self.assertEqual(a.ravel()[0], device_count)
    self.assertEqual(b.ravel()[0], 1)
    self.assertEqual(c.ravel()[0], device_count * 1)

  def testAxisIndex(self):
    device_count = jax.device_count()
    f = self.pmap(lambda x: x + lax.axis_index('i'), 'i')
    x = jnp.ones(device_count)
    ans = f(x)
    expected = 1 + np.arange(device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAxisIndexNestedPmap(self):
    device_count = jax.device_count()
    if device_count < 4:
      raise SkipTest("test requires at least four devices")
    f = lambda axis: self.pmap(self.pmap(lambda x: x + lax.axis_index(axis), 'j'), 'i')
    x = jnp.ones((2, 2))
    expected_j = np.broadcast_to(1 + np.arange(2), (2, 2))
    self.assertAllClose(f('j')(x), expected_j, check_dtypes=False)
    self.assertAllClose(f('i')(x), expected_j.T, check_dtypes=False)

  def testAxisIndexNd(self):
    device_count = jax.device_count()
    if device_count < 4:
      raise SkipTest("test requires at least four devices")
    f = lambda axes: self.pmap(self.pmap(lambda x: x + lax.axis_index(axes), 'j'), 'i')
    x = jnp.ones((2, 2))
    expected = 1 + np.arange(4).reshape((2, 2))
    self.assertAllClose(f(('i', 'j'))(x), expected, check_dtypes=False)
    self.assertAllClose(f(('j', 'i'))(x), expected.T, check_dtypes=False)

  def testAxisIndexInInitialStyle(self):
    @partial(self.pmap, axis_name='i')
    def f(x):
      def body(carry, i):
        return carry + i + lax.axis_index('i'), None
      return lax.scan(body, 0, x)[0]
    device_count = jax.device_count()
    shape = (device_count, 10)
    self.assertAllClose(f(jnp.ones(shape, dtype=int)),
                        (np.arange(device_count) + 1) * 10)

  def testVmapOfPmap(self):
    device_count = jax.device_count()
    f0 = lambda x: x
    f1 = self.pmap(f0, axis_name='i')
    ax = np.random.randn(2, device_count, 50, 60)
    bx = vmap(f1)(ax)
    self.assertAllClose(ax, bx, check_dtypes=False)

  def testVmapOfPmap2(self):
    N_DEVICES = jax.device_count()
    keys = random.split(random.PRNGKey(1), 13)  # [13, 2]

    @self.pmap
    def g(key):
      _ = random.normal(key, ())
      return 0.

    @vmap
    def s(keys):
      keys = tree_util.tree_map(
          lambda x: jnp.broadcast_to(x, (N_DEVICES,) + x.shape),
          keys)
      return g(keys)

    ans = s(keys)  # doesn't crash
    self.assertEqual(ans.shape, (13, N_DEVICES))

  def testVmapOfPmap3(self):
    # https://github.com/google/jax/issues/3399
    device_count = jax.device_count()
    if device_count < 2:
      raise SkipTest("test requires at least two devices")

    def map_version(qs, pts):
      return jax.lax.map(lambda x: func(x, pts), qs)

    def vmap_version(qs, pts):
      return jax.vmap(func, in_axes=(0, None))(qs, pts)

    def func(q, pts):
      q_from_pmap = self.pmap(lambda x, y: y, in_axes=(0, None))(pts, q)
      return q, q_from_pmap

    pts = jnp.ones(device_count)
    qs = jnp.asarray(((0,0), (3,3), (2,2)))

    with ignore_jit_of_pmap_warning():
      _, expected = map_version(qs, pts)
    _, ans = vmap_version(qs, pts)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testVmapOfPmapNonLeadingAxis(self):
    device_count = jax.device_count()
    f0 = lambda x: x
    f1 = self.pmap(f0, axis_name='i')
    ax = np.random.randn(device_count, 2, 50, 60)
    bx = vmap(f1, in_axes=2, out_axes=2)(ax)
    self.assertAllClose(ax, bx, check_dtypes=False)

  def testVmapOfPmapTuple(self):
    device_count = jax.device_count()
    f0 = lambda *x: x
    f1 = self.pmap(f0, axis_name='i')

    ax = np.random.randn(device_count, 2, 50, 60)
    ay = np.random.randn(device_count, 30, 2)
    az1 = np.random.randn(device_count, 20)
    az2 = np.random.randn(2, device_count, 20)

    bx, by, bz = vmap(f1, in_axes=(1, 2, (None, 0)), out_axes=(1, 2, 0))(ax, ay, (az1, az2))

    self.assertAllClose(ax, bx, check_dtypes=False)
    self.assertAllClose(ay, by, check_dtypes=False)

    bz1, bz2 = bz
    expected_bz1 = np.broadcast_to(az1, (2,) + az1.shape)
    self.assertAllClose(expected_bz1, bz1, check_dtypes=False)
    self.assertAllClose(bz2, bz2, check_dtypes=False)

  @ignore_slow_all_to_all_warning()
  def testPswapaxes(self):
    device_count = jax.device_count()
    shape = (device_count, 3, device_count, 5)
    x = np.arange(prod(shape)).reshape(shape)

    ans = self.pmap(lambda x: lax.pswapaxes(x, 'i', 1), axis_name='i')(x)
    expected = np.swapaxes(x, 0, 2)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_slow_all_to_all_warning()
  def testGradOfPswapaxes(self):
    device_count = jax.device_count()
    shape = (device_count, 1, device_count)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    w = np.arange(device_count, dtype=np.float32)

    @partial(self.pmap, axis_name='i')
    def f(x, w):
      g = lambda x: jnp.sum(lax.pswapaxes(x, 'i', 1) * w)
      return grad(g)(x)

    ans = f(x, w)
    expected = np.tile(w, reps=device_count).reshape(shape)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_slow_all_to_all_warning()
  def testAllToAllReplicaGroups(self):
    # If num_devices = 4, these would be the inputs/outputs:
    # input = [[0, 1], [2, 3], [4, 5], [6, 7]]
    # axis_index_groups = [[0, 2], [1, 3]]
    # output = [[0, 4], [2, 6], [1, 5], [3, 7]]
    #
    # This is essentially like splitting the number of rows in the input in two
    # groups of rows, and swaping the two inner axes (axis=1 and axis=2), which
    # is exactly what the test case checks.
    device_count = jax.device_count()
    if device_count % 2 != 0:
      raise SkipTest('test requires an even number of devices')
    shape = (device_count, device_count // 2)
    x = np.arange(prod(shape)).reshape(shape)

    axis_index_groups = np.arange(device_count, dtype=np.int32)
    axis_index_groups = axis_index_groups.reshape((device_count // 2, 2)).T
    axis_index_groups = axis_index_groups.tolist()

    @partial(self.pmap, axis_name='i')
    def fn(x):
      return lax.all_to_all(x, 'i', 0, 0, axis_index_groups=axis_index_groups)

    expected = np.swapaxes(
        x.reshape((device_count // 2, 2, device_count // 2)),
        0, 2).reshape(shape)
    self.assertAllClose(fn(x), expected, check_dtypes=False)

  @ignore_slow_all_to_all_warning()
  def testGradOfAllToAllReplicaGroups(self):
    device_count = jax.device_count()
    if device_count % 2 != 0:
      raise SkipTest('test requires an even number of devices')
    shape = (device_count, device_count // 2, 1)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    w = np.arange(device_count, dtype=np.float32)

    axis_index_groups = np.arange(device_count, dtype=np.int32)
    axis_index_groups = axis_index_groups.reshape((2, device_count // 2))
    axis_index_groups = axis_index_groups.tolist()

    @partial(self.pmap, axis_name='i')
    def fn(x, w):
      g = lambda x: jnp.sum(lax.all_to_all(x, 'i', 0, 1, axis_index_groups=axis_index_groups) * w)
      return grad(g)(x)

    expected = np.ones_like(x) * w[:, np.newaxis, np.newaxis]
    expected = np.swapaxes(
        expected.reshape((2, device_count // 2, device_count // 2)),
        1, 2).reshape(shape)
    self.assertAllClose(fn(x, w), expected, check_dtypes=False)

  def testReshardInput(self):
    if jax.device_count() < 6:
      raise SkipTest("testReshardInput requires 6 devices")
    # Manually construct a ShardedDeviceArray with the wrong sharding for the
    # subsequent pmap
    shard_shape = (3,2)
    shard = jnp.arange(prod(shard_shape)).reshape(shard_shape)
    bufs = pxla.device_put(shard, jax.devices()[:4], replicate=True)
    aval = ShapedArray((6,4), shard.dtype)
    sharding_spec = pxla.ShardingSpec(
        sharding=map(pxla.Chunked, ([2], [2])),
        mesh_mapping=map(pxla.ShardedAxis, (0, 1)))
    arr = pxla.make_sharded_device_array(aval, sharding_spec, bufs)

    r = self.pmap(lambda x: x + 1)(arr)
    self.assertAllClose(r, arr + 1)
    self.assertEqual(len(r.device_buffers), 6)

  @ignore_xmap_warning()
  def testSoftPmapBatchMatmul(self):
    n = 4 * jax.device_count()
    xs = np.arange(n * 2 * 3).reshape(n, 2, 3)
    ys = np.arange(n * 3 * 4).reshape(n, 3, 4)
    ans = soft_pmap(jnp.dot, 'i')(xs, ys)
    expected = np.einsum('nij,njk->nik', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_xmap_warning()
  def testSoftPmapBatchMatmulJit(self):
    n = 4 * jax.device_count()
    xs = np.arange(n * 2 * 3).reshape(n, 2, 3)
    ys = np.arange(n * 3 * 4).reshape(n, 3, 4)
    ans = soft_pmap(jit(jnp.dot), 'i')(xs, ys)
    expected = np.einsum('nij,njk->nik', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_xmap_warning()
  def testSoftPmapPsumConstant(self):
    n = 4 * jax.device_count()
    def f(_):
      return lax.psum(1, 'i')
    ans = soft_pmap(f, 'i')(jnp.ones(n))
    expected = n * np.ones(n)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_xmap_warning()
  def testSoftPmapPsum(self):
    n = 4 * jax.device_count()
    def f(x):
      return x / lax.psum(x, 'i')
    ans = soft_pmap(f, 'i')(jnp.ones(n))
    expected = np.ones(n) / n
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_xmap_warning()
  def testSoftPmapAxisIndex(self):
    n = 4 * jax.device_count()
    def f(x):
      return x * lax.axis_index('i')
    ans = soft_pmap(f, 'i')(2 * jnp.ones(n))
    expected = 2 * np.arange(n)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_xmap_warning()
  def testSoftPmapOfJit(self):
    n = 4 * jax.device_count()
    def f(x):
      return 3 * x
    ans = soft_pmap(jit(f), 'i')(np.arange(n))
    expected = 3 * np.arange(n)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_xmap_warning()
  @unittest.skip("not implemented")  # TODO(mattjj): re-implement
  def testSoftPmapNested(self):
    n = 4 * jax.device_count()

    @partial(soft_pmap, axis_name='i')
    @partial(soft_pmap, axis_name='j')
    def f(x):
      i_size = lax.psum(1, 'i')
      return x + lax.axis_index('i') + i_size * lax.axis_index('j')

    ans = f(jnp.zeros((n, n)))
    expected = np.arange(n ** 2).reshape(n, n).T
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_xmap_warning()
  @unittest.skip("not implemented")  # TODO(mattjj): re-implement
  def testGradOfSoftPmap(self):
    n = 4 * jax.device_count()

    @partial(soft_pmap, axis_name='i')
    def f(x):
      return x * lax.axis_index('i')

    ans = grad(lambda x: jnp.sum(f(x)))(jnp.zeros((n, n)))
    expected = np.repeat(np.arange(n)[:, None], n, axis=1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_xmap_warning()
  def testSoftPmapDevicePersistence(self):
    device_count = jax.device_count()
    shape = (2 * 2 * device_count, 2, 3)

    # check that we can maintain device persistence across calls
    x = np.arange(prod(shape)).reshape(shape)
    x = soft_pmap(lambda x: x)(x)
    self.assertIsInstance(x, pxla.ShardedDeviceArray)
    x._npy_value = np.float32(np.nan)  # can't be coerced to ndarray for xfer
    x = soft_pmap(lambda x: x)(x)  # doesn't crash
    self.assertIsInstance(x, pxla.ShardedDeviceArray)

  @unittest.skip("the underlying code here is broken")  # TODO(mattjj)
  def testSoftPmapAllToAll(self):
    n = 4 * jax.device_count()
    def f(x):
      return lax.all_to_all(x, 'i', 0, 0)
    ans = soft_pmap(f, 'i')(jnp.arange(n ** 2).reshape(n, n))
    expected = np.arange(n ** 2).reshape(n, n).T
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testShardedDeviceArrayBlockUntilReady(self):
    x = np.arange(jax.device_count())
    x = self.pmap(lambda x: x)(x)
    x.block_until_ready()  # doesn't crash

  @ignore_jit_of_pmap_warning()
  def testJitPmapComposition(self):
    f = lambda x: x - lax.psum(x, 'i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.sum(x, 0)

    ans = jit(self.pmap(f, 'i'))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = self.pmap(jit(f), 'i')(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCompositionWithJitTwice(self):
    @jit
    def f(x):
      y = 2 * x

      @jit
      def g(z):
        return self.pmap(lambda x: x * y)(z)

      return g(x)

    f(np.arange(1.).reshape((1, 1)))  # doesn't crash

  @ignore_jit_of_pmap_warning()
  def testIssue1065(self):
    # from https://github.com/google/jax/issues/1065
    device_count = jax.device_count()

    def multi_step_pmap(state, count):
      @partial(self.pmap, axis_name='x')
      @jit
      def exchange_and_multi_step(state):
        return state

      @jit
      def time_evolution(state):
        return lax.fori_loop(0, count, lambda i, s: exchange_and_multi_step(s), state)

      return time_evolution(state)

    multi_step_pmap(jnp.zeros((device_count,)), count=1)

  def testShardedDeviceArrayGetItem(self):
    f = lambda x: 2 * x
    f = self.pmap(f, axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    y = f(x)
    self.assertIsInstance(y, jnp.ndarray)
    self.assertIsInstance(y, pxla.ShardedDeviceArray)

    z = y[0]  # doesn't crash
    self.assertAllClose(z, 2 * x[0], check_dtypes=False)

  # TODO(mattjj): this fails with multiple devices (unless we add a jit)
  # because we assume eager ops (like scan here) can't require more than 1
  # replica.
  @unittest.skip("need eager multi-replica support")
  def testPostProcessMap(self):
    # test came from https://github.com/google/jax/issues/1369
    nrep = jax.device_count()

    def pmvm(a, b):
      a = a.reshape((nrep, -1, a.shape[1]))
      func = self.pmap(lambda z: jnp.dot(z, b))
      return func(a).reshape(b.shape)

    n = nrep * 2
    rng = np.random.RandomState(0)
    a = rng.randn(n, n)
    b = rng.randn(n)

    iters = jnp.arange(5)
    def body(carry, i):
      return pmvm(a, carry), i
    ans, _ = lax.scan(body, b, iters)

    expected = np.linalg.matrix_power(a, 5).dot(b)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testManyArgs(self):
    @self.pmap
    def f(args_list):
      return sum(args_list)

    vals = list(range(500))
    ndevices = jax.device_count()
    self.assertAllClose(f(jnp.array([vals] * ndevices)),
                        jnp.array([sum(vals)] * ndevices))

  def testPostProcessMap2(self):
    # code from https://github.com/google/jax/issues/2787
    def vv(x, y):
      """Vector-vector multiply"""
      return jnp.dot(x, y)

    def distributed_matrix_vector(x, y):
      """Matrix vector multiply. First batch it and then row by row"""
      fv = lambda z: lax.map(lambda j: vv(j, y), z)
      res = self.pmap(fv)(x.reshape((jax.device_count(), -1) + tuple(x.shape[1:])))
      res = res.reshape(res.shape[0] * res.shape[1], *res.shape[2:])
      return res

    key = random.PRNGKey(1)
    x = random.normal(key, (80, 50))
    batched_mvm = vmap(lambda b: distributed_matrix_vector(x, b), in_axes=0)
    y = random.normal(key, (10, 50, 1))
    result = batched_mvm(y)
    expected = jnp.einsum('ij,njk->nik', x, y)
    tol = 1e-1 if jtu.device_under_test() == "tpu" else 1e-3
    self.assertAllClose(result, expected, check_dtypes=False, atol=tol, rtol=tol)

  def testAxisIndexRemat(self):
    # https://github.com/google/jax/issues/2716
    n = len(jax.devices())

    def f(key):
      key = random.fold_in(key, jax.lax.axis_index('i'))
      return random.bernoulli(key, p=0.5)

    keys = random.split(random.PRNGKey(0), n)
    self.pmap(jax.remat(f), axis_name='i')(keys)

  def testPmapMapVmapCombinations(self):
    # https://github.com/google/jax/issues/2822
    def vv(x, y):
      """Vector-vector multiply"""
      return jnp.dot(x, y)

    def matrix_vector(x, y, parallel=True):
      """Matrix vector multiply. First batch it and then row by row"""
      fv = lambda z: lax.map(lambda j: vv(j, y), z)
      if parallel:
        # split leading axis in two
        new_x = x.reshape((jax.device_count(), -1, *x.shape[1:]))
        # apply map
        new_res = self.pmap(fv)(new_x)
        # reshape back out
        res = new_res.reshape(x.shape[0], *new_res.shape[2:])
      else:
        res = fv(x)
      return res

    x = random.normal(random.PRNGKey(1), (80, 5))
    y = random.normal(random.PRNGKey(1), (10, 5))

    result1 = vmap(lambda b: matrix_vector(x, b, True))(y)       # vmap + pmap
    result2 = lax.map(lambda b: matrix_vector(x, b, False), y)   # map + map
    with ignore_jit_of_pmap_warning():
      result3 = lax.map(lambda b: matrix_vector(x, b, True), y)    # map + pmap
    result4 = jnp.stack([matrix_vector(x, b, False) for b in y])  # none + map

    self.assertAllClose(result1, result2, check_dtypes=False, atol=1e-3, rtol=1e-3)
    self.assertAllClose(result1, result3, check_dtypes=False, atol=1e-3, rtol=1e-3)
    self.assertAllClose(result1, result4, check_dtypes=False, atol=1e-3, rtol=1e-3)

  def testPmapAxisNameError(self):
    # https://github.com/google/jax/issues/3120
    a = np.arange(4)[np.newaxis,:]
    def test(x):
      return jax.lax.psum(x, axis_name='batch')

    with self.assertRaisesRegex(NameError, "unbound axis name: batch"):
      self.pmap(test)(a)

  def testPsumOnBooleanDtype(self):
    # https://github.com/google/jax/issues/3123
    n = jax.device_count()
    if n > 1:
      x = jnp.array([True, False])

      out = self.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x)
      self.assertEqual(list(out), [1, 1])

      out = self.pmap(lambda x: jax.lax.pmean(x, 'i'), 'i')(x)
      self.assertEqual(list(out), [1/2, 1/2])
    else:
      x = jnp.array([True])

      out = self.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x)
      self.assertEqual(list(out), [1])

      out = self.pmap(lambda x: jax.lax.pmean(x, 'i'), 'i')(x)
      self.assertEqual(list(out), [1])

  def testPsumWithNoAxisDoesntLeakFunctions(self):
    x = jnp.ones((1, 1024), dtype=np.float32)
    f = lambda _: x
    w = weakref.ref(f)
    g = self.pmap(f)
    g(np.ones((1,), dtype=np.float32)).block_until_ready()
    del f, g
    gc.collect()
    # 'f' should not be alive at this point; in particular the pmap cache must
    # not keep it alive.
    self.assertTrue(w() is None)

  def testJitOfPmapWarningMessage(self):
    device_count = jax.device_count()

    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    def foo(x): return x

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      jit(self.pmap(foo))(jnp.arange(device_count))

      self.assertGreaterEqual(len(w), 1)
      self.assertIn("The jitted function foo includes a pmap",
                    str(w[-1].message))

  def testPsumZeroCotangents(self):
    # https://github.com/google/jax/issues/3651
    def loss(params, meta_params):
      (net, mpo) = params
      return meta_params * mpo * net

    def inner(meta_params, params):
      grads = jax.grad(loss)(params, meta_params)
      grads = lax.psum(grads, axis_name="i")
      net_grads, mpo_grads = grads
      net = params[0] + net_grads
      mpo = params[1]
      return mpo * net

    def outer(params):
      meta_params = jnp.array(4.0)
      return jax.grad(inner)(meta_params, params)

    params = (jnp.array([2.0]), jnp.array([3.0]))
    self.pmap(outer, axis_name='i')(params)  # doesn't crash

    f = self.pmap(outer, axis_name='i')
    jtu.check_grads(f, (params,), 2, ["fwd", "rev"], 1e-3, 1e-3)

  @ignore_jit_of_pmap_warning()
  def test_issue_1062(self):
    # code from https://github.com/google/jax/issues/1062 @shoyer
    # this tests, among other things, whether ShardedDeviceTuple constants work
    device_count = jax.device_count()

    @jit
    def multi_step(state, count):
      return lax.fori_loop(0, count, lambda i, s: s, state)

    @jit
    def multi_step_pmap(state, count=2):
      @partial(self.pmap, axis_name='x')
      def pmapped_multi_step(state):
        return multi_step(state, count)

      return pmapped_multi_step(state)

    u = np.ones((device_count, 100))
    multi_step_pmap(u)  # doesn't crash

  @jtu.skip_on_devices("cpu")
  def test_replicate_backend(self):
    # TODO(skye): fix backend caching so we always have multiple CPUs available
    if jax.device_count("cpu") < 4:
      self.skipTest("test requires 4 CPU device")
    # https://github.com/google/jax/issues/4223
    def fn(indices):
      return jnp.equal(indices, jnp.arange(3)).astype(jnp.float32)
    mapped_fn = self.pmap(fn, axis_name='i', backend='cpu')
    mapped_fn = self.pmap(mapped_fn, axis_name='j', backend='cpu')
    indices = np.array([[[2], [1]], [[0], [0]]])
    mapped_fn(indices)  # doesn't crash

  @ignore_xmap_warning()
  def testPdotBasic(self):
    num_devices = jax.device_count()

    def f(x, y):
      return lax.pdot(x, y, 'i')

    x = jnp.arange(num_devices * 3).reshape(num_devices, 3)
    y = jnp.arange(num_devices * 5).reshape(num_devices, 5)
    z = self.pmap(f, axis_name='i', out_axes=None)(x, y)
    self.assertAllClose(z, jnp.dot(x.T, y))

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_collective={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis, collective.__name__.replace(" ", "")),
       "shape": shape, "dtype": dtype, "axis": axis,
       "collective": collective, "bulk_op": bulk_op}
      for collective, bulk_op in [
          (parallel.pargmax, jnp.argmax),
          (parallel.pargmin, jnp.argmin)
      ]
      for dtype in [np.float32, np.int32]
      for shape in [(4,), (2, 2), (2, 4), (4, 2)]
      for axis in range(len(shape))
  )
  def testArgAllReduce(self, shape, dtype, axis, collective, bulk_op):
    if jax.device_count() < shape[axis]:
      raise SkipTest(f"test requires at least {shape[axis]} devices")
    if (jtu.device_under_test() == 'cpu' and
        np.issubdtype(dtype, np.floating) and
        len(shape) > 1):
      raise SkipTest("skipped on cpu due to strange failures")  # TODO(mattjj)

    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    ans = self.pmap(lambda x: collective(x, 'i'), in_axes=axis, out_axes=None,
               axis_name='i')(x)
    expected = bulk_op(x, axis=axis)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_dtype={}".format(
          jtu.format_shape_dtype_string((), dtype)),
       "dtype": dtype}
      for dtype in [np.float32, np.int32]
  )
  def testPmapDtype(self, dtype):
    # Regression test for https://github.com/google/jax/issues/6022
    @partial(self.pmap, axis_name='i')
    def func(_):
      return jax.lax.psum(dtype(0), axis_name='i')
    unused_arg = jnp.arange(jax.device_count())
    out_dtype = func(unused_arg).dtype
    self.assertEqual(out_dtype, dtype)

  def test_num_replicas_with_switch(self):
    # https://github.com/google/jax/issues/7411
    def identity(x):
      return x

    def cond_of_pmap(x):
      y = lax.cond(True, jax.pmap(identity), jax.pmap(identity), x)
      return y

    with ignore_jit_of_pmap_warning():
      cond_of_pmap(jnp.zeros((jax.device_count(), 2)))

  def test_static_argnum_on_method(self):

    class A:

      @partial(self.pmap, static_broadcasted_argnums=(0,))
      def my_func_pmap(self, x):
        return x + 2

    A().my_func_pmap(jnp.asarray([3] * jax.device_count()))

  def test_pmap_error_on_non_hashable_static_argument(self):
    f = lambda x, y: x + 3
    pmapped_f = self.pmap(f, static_broadcasted_argnums=(1,))

    inputs = np.asarray([1] * jax.device_count())
    with self.assertRaisesRegex(
        ValueError, "Non-hashable static arguments are not supported.*"):
      pmapped_f(inputs, np.asarray(1))

  @parameterized.named_parameters(
      {"testcase_name": f"_axis_size={axis_size}", "axis_size": axis_size}
      for axis_size in [1, 2])
  def test_grad_of_pmap_compilation_caching(self, axis_size):
    if len(jax.local_devices()) < axis_size:
      raise SkipTest("too few devices for test")

    @jax.pmap
    def f(x):
      return jnp.sin(x)

    x = jnp.ones(axis_size)
    f(x)  # warm-up any dispatching compilations

    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      _, f_bwd  = jax.vjp(f, x)
      _ = f_bwd(x)
    self.assertEqual(count[0], 2)  # one for fwd, one for bwd

    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      _  = jax.vjp(f, x)
      _ = f_bwd(x)
    self.assertEqual(count[0], 0)  # cache hits on fwd and bwd

class CppPmapTest(PythonPmapTest):

  @property
  def pmap(self):
    if jax._src.lib._xla_extension_version >= 38:
      return src_api._cpp_pmap
    else:
      return src_api._python_pmap


class VmapOfPmapTest(jtu.JaxTestCase):

  # TODO(apaszke)
  @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
       "testcase_name": f"{shapes}_{vmap_in_axes}_{vmap_out_axes}_{pmap_in_axes}_{pmap_out_axes}",
       "shapes": shapes,
       "vmap_in_axes": vmap_in_axes, "vmap_out_axes": vmap_out_axes,
       "pmap_in_axes": pmap_in_axes, "pmap_out_axes": pmap_out_axes
    } for arg_shapes in s(compatible_shapes)
      for num_args in s(range(1, 4))
      for shapes in s(list(it.combinations_with_replacement(arg_shapes, num_args)))
      for vmap_in_axes in s(all_bdims(*shapes, pmap=False))
      for pmap_in_axes in s(all_bdims(*shapes, pmap=True))
      for vmap_out_axes in s(out_bdims(shapes[0], False))
      for pmap_out_axes in s(out_bdims(shapes[0], True))
  )))
  def testVmapOfPmap(self, shapes, vmap_in_axes, pmap_in_axes, vmap_out_axes, pmap_out_axes):
    vmapped_size = 3
    pmapped_size = jax.device_count()

    rng = jtu.rand_default(self.rng())

    def fun(*args):
      return sum(args)

    final_shapes = map(partial(add_bdim, vmapped_size), vmap_in_axes,
                       map(partial(add_bdim, pmapped_size), pmap_in_axes, shapes))

    def args_slice(vi, pi):
      return args_slicer(args_slicer(args, vmap_in_axes)(vi), pmap_in_axes)(pi)

    args = [rng(shape, jnp.float32) for shape in final_shapes]
    ans = vmap(pmap(fun, in_axes=pmap_in_axes, out_axes=pmap_out_axes),
               in_axes=vmap_in_axes,
               out_axes=vmap_out_axes)(*args)
    expected = np.stack(
      [np.stack([fun(*args_slice(vi, pi)) for pi in range(pmapped_size)], axis=pmap_out_axes)
       for vi in range(vmapped_size)],
      axis=vmap_out_axes)
    self.assertAllClose(ans, expected)

class VmapPmapCollectivesTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      {"testcase_name": "_collective={}".format(collective.__name__).replace(" ", ""),
       "collective": collective}
      for collective in [lax.psum, lax.pmean, lax.pmax, lax.pmin])
  def testCollectivesWithVmap(self, collective):
    def f(map1, map2):
      @partial(map1, axis_name='i')
      @partial(map2, axis_name='j')
      def f(x, y):
        return x + collective(x.dot(y), ('i', 'j'))
      return f

    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")
    x = jnp.ones((2, 2, 64, 64))
    y = f(jax.pmap, jax.pmap)(x, x)
    self.assertAllClose(f(jax.vmap, jax.vmap)(x, x), y)
    self.assertAllClose(f(jax.pmap, jax.vmap)(x, x), y)
    self.assertAllClose(f(jax.vmap, jax.pmap)(x, x), y)

  def testPPermuteWithVmap(self):
    perm = [(0, 1), (1, 0)]

    def f(map2):
      @partial(jax.pmap, axis_name='i')
      @partial(map2)
      def f(x, y):
        return x + jax.lax.ppermute(x.dot(y), 'i', perm)
      return f

    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")
    x = jnp.ones((2, 2, 64, 64))
    self.assertAllClose(f(jax.pmap)(x, x), f(jax.vmap)(x, x))

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}_vmap={vmap_axis}",
       "split_axis": split_axis, "concat_axis": concat_axis, "vmap_axis": vmap_axis}
      for split_axis, concat_axis, vmap_axis in it.product(range(3), range(3), range(4)))
  @ignore_slow_all_to_all_warning()
  def testAllToAllInVmap(self, split_axis, concat_axis, vmap_axis):
    def f(x):
      return lax.all_to_all(x, 'i', split_axis=split_axis, concat_axis=concat_axis)

    def adj(axis, hidden_axes):
      for hax in sorted(hidden_axes):
        if hax <= axis:
          axis += 1
      return axis

    def reference(x, split_axis, concat_axis, vmap_axis):
      pmap_axis = 0
      vmap_axis = adj(vmap_axis, [pmap_axis])
      ref = x

      # Step 1.
      # Adjust the split axis to the real tensor layout and move it to
      # position 1. Since pmap_axis is always 0 we don't have to adjust it,
      # but we do have to adjust vmap_axis.
      split_axis = adj(split_axis, [pmap_axis, vmap_axis])
      ref = jnp.moveaxis(ref, split_axis, pmap_axis + 1)
      vmap_axis = vmap_axis + (0 if split_axis < vmap_axis else 1)
      split_axis = pmap_axis + 1  # split_axes == 1

      # Step 2.
      # Now, we move pmap_axis to the position indicated by concat_axis.
      concat_axis = adj(concat_axis, [pmap_axis, split_axis, vmap_axis]) - 1
      ref = jnp.moveaxis(ref, pmap_axis, concat_axis)
      pmap_axis = 0
      vmap_axis = vmap_axis - (1 if concat_axis >= vmap_axis else 0)
      del split_axis, concat_axis

      # Step 3. vmap_axis always ends in position 1, since out_axes=0.
      ref = jnp.moveaxis(ref, vmap_axis, 1)
      return ref

    def verify_ref():
      # Both the reference and the real implementation of all_to_all batching involve
      # some pretty complicated axis arithmetic, so it would be good to verify that it's
      # not the case that the test passes because they're both incorrect. Fortunately, it
      # is quite easy to write out the shape function for this code, and we know
      # that it should be equivalent to a bunch of transposes, so the code below verifies
      # that the reference puts the right dimensions in the right places. Note that we
      # can't do the same comparison on f, since all_to_all wouldn't allow us to swap axes of
      # different sizes.
      start_shape = [2, 3, 4, 5, 6]
      instance_shape = start_shape.copy()
      pmap_dim_id = instance_shape.pop(0)
      vmap_dim_id = instance_shape.pop(vmap_axis)
      split_axis_id = instance_shape.pop(split_axis)
      instance_shape.insert(concat_axis, pmap_dim_id)
      expected_shape = (split_axis_id, vmap_dim_id, *instance_shape)

      x = np.empty(start_shape)
      self.assertEqual(reference(x, split_axis, concat_axis, vmap_axis).shape,
                       expected_shape)

    verify_ref()

    shape = (jax.device_count(),) * 5
    x = jnp.arange(np.prod(shape)).reshape(shape)
    self.assertAllClose(pmap(vmap(f, in_axes=vmap_axis), axis_name='i')(x),
                        reference(x, split_axis, concat_axis, vmap_axis))

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}",
       "split_axis": split_axis, "concat_axis": concat_axis}
      for split_axis, concat_axis in it.product(range(3), range(3)))
  @ignore_slow_all_to_all_warning()
  def testAllToAllVsVmap(self, split_axis, concat_axis):
    def f(x):
      return lax.all_to_all(x, 'i', split_axis=split_axis, concat_axis=concat_axis)

    shape = (jax.device_count(),) * 4
    x = jnp.arange(np.prod(shape)).reshape(shape)
    self.assertAllClose(pmap(f, axis_name='i')(x),
                        vmap(f, axis_name='i')(x))

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}_axes={''.join(axes)}",
       "axes": axes, "split_axis": split_axis, "concat_axis": concat_axis}
      for axes, split_axis, concat_axis
      in it.product([('i', 'j'), ('j', 'i')], range(3), range(3)))
  @ignore_slow_all_to_all_warning()
  @unittest.skip("multi-axis all_to_all broken after #4835")  # TODO(mattjj,apaszke)
  def testAllToAllMultipleAxesVsVmap(self, axes, split_axis, concat_axis):
    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")

    def f(x):
      return lax.all_to_all(x, axes, split_axis=split_axis, concat_axis=concat_axis)

    shape = (2, 2, 4, 4, 4)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    self.assertAllClose(pmap(pmap(f, axis_name='j'), axis_name='i')(x),
                        vmap(vmap(f, axis_name='j'), axis_name='i')(x))

  def testAllGatherWithVmap(self):
    def f(map2):
      @partial(jax.pmap, axis_name='i')
      @partial(map2)
      def f(x):
        return jax.lax.all_gather(x, 'i')
      return f

    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")
    x = jnp.ones((2, 2, 64, 64))
    self.assertAllClose(f(jax.pmap)(x), f(jax.vmap)(x))


class PmapWithDevicesTest(jtu.JaxTestCase):

  def testAllDevices(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i',
             devices=jax.devices())
    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.sum(x, 0)
    ans = f(x)
    self.assertAllClose(ans, expected)

  def testOneDevice(self):
    if jax.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    d0 = jax.devices()[0]
    d1 = jax.devices()[1]
    f = lambda x: jnp.dot(x, x.T)
    f0 = pmap(f, devices=[d0])
    f1 = pmap(f, devices=[d1])
    x = np.random.rand(1, 1000, 1000)
    r0 = f0(x)
    r1 = f1(x)
    expected = np.expand_dims(np.dot(x.squeeze(), x.squeeze().T), 0)
    self.assertAllClose(r0, expected, atol=1e-6, rtol=1e-3)
    self.assertAllClose(r1, expected, atol=1e-6, rtol=1e-3)

  def testNoDevicesError(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i', devices=[])
    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    with self.assertRaisesRegex(
        ValueError, "'devices' argument to pmap must be non-empty, or None."):
      f(x)

  def testBadAxisSizeError(self):
    if jax.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    f = pmap(lambda x: lax.psum(x, 'i'), axis_name='i',
             devices=jax.devices())
    with self.assertRaisesRegex(
        ValueError, r"Leading axis size of input to pmapped function must "
        r"equal the number of local devices passed to pmap. Got axis_size=1, "
        r"num_local_devices=\d."):
      f(jnp.ones(1))

    with self.assertRaisesRegex(
        ValueError, r"Leading axis size of input to pmapped function must "
        r"equal the number of local devices passed to pmap. Got axis_size=\d, "
        r"num_local_devices=\d."):
      f(jnp.ones(jax.device_count() + 1))

  def testBadAxisSizeErrorNested(self):
    f = pmap(pmap(lambda x: lax.psum(x, ('i', 'j')),
                  axis_name='j'),
             axis_name='i',
             devices=[jax.local_devices()[0]])
    with self.assertRaisesRegex(
        ValueError,
        r"pmapped function requires 4 local devices to run due to nested "
        r"pmapped or other parallel functions, but only 1 are available."):
      f(jnp.ones((1, 4)))

  def testNestedPmaps(self):
    if jax.device_count() % 2 != 0:
      raise SkipTest

    # Devices specified in outer pmap are OK
    @partial(pmap, axis_name='i', devices=jax.devices())
    def foo(x):
      @partial(pmap, axis_name='j')
      def bar(y):
        return lax.psum(y, 'j')
      return bar(x)

    x = jnp.ones((jax.device_count() // 2, 2))
    ans = foo(x)
    expected = x * 2
    self.assertAllClose(ans, expected)

  def testNestedPmapsError(self):
    # Devices specified in inner pmap not OK
    @partial(pmap, axis_name='i')
    def foo(x):
      @partial(pmap, axis_name='j', devices=jax.devices())
      def bar(y):
        return lax.psum(y, 'j')
      return bar(x)

    with self.assertRaisesRegex(
        ValueError,
        "Nested pmap with explicit devices argument."):
      foo(jnp.ones((jax.device_count(), 1)))

  def testJitInPmap(self):
    @partial(pmap, axis_name='i', devices=jax.devices())
    def foo(x):
      @jit
      def bar(y):
        return y + 1
      return lax.psum(bar(x), 'i')

    ndevices = jax.device_count()
    ans = foo(jnp.ones((ndevices, 1)))
    expected = np.ones((ndevices, 1), dtype=jnp.float_) * ndevices * 2
    self.assertAllClose(ans, expected)

  @ignore_jit_of_pmap_warning()
  def testPmapInJit(self):
    @jit
    def foo(x):
      @partial(pmap, axis_name='i', devices=jax.devices())
      def bar(y):
        return lax.psum(y, 'i')
      return bar(x)

    ndevices = jax.device_count()
    ans = foo(jnp.ones((ndevices, 1)))
    expected = np.ones((ndevices, 1), dtype=jnp.float_) * ndevices
    self.assertAllClose(ans, expected)

  def testGradBasic(self):
    @partial(pmap, axis_name='i', devices=jax.devices())
    def f(x):
      return jnp.sin(x)

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(jnp.sin(x)))(x)
    expected = grad(lambda x: jnp.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPmapStaticArgnums(self):
    @partial(pmap, axis_name='i', static_broadcasted_argnums=1)
    def f(x, y):
      return jnp.sin(x + y())
    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    y = lambda: 3.

    ans = f(x, y)
    expected = np.sin(x + 3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPmapInAxesBasic(self):
    @partial(pmap, in_axes=(1, 2))
    def f(x, y):
      return jnp.sin(x + y)
    xshape = (2, jax.device_count(), 4)
    x = np.arange(prod(xshape)).reshape(xshape)
    yshape = (2, 4, jax.device_count())
    y = np.arange(prod(yshape)).reshape(yshape)

    self.assertAllClose(f(x, y),
                        jnp.sin(x.transpose((1, 0, 2)) + y.transpose((2, 0, 1))))

  def testPmapInAxesGrad(self):
    def f(x, y, z):
      return jnp.sin(x + y + z)
    fp = pmap(f, in_axes=(1, 2, None))
    fv = vmap(f, in_axes=(1, 2, None))
    xshape = (5, jax.device_count(), 7)
    x = np.arange(prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (5, 7, jax.device_count())
    y = np.arange(prod(yshape), dtype=np.float32).reshape(yshape)
    zshape = (5, 7)
    z = np.arange(prod(zshape), dtype=np.float32).reshape(zshape)

    dx, dy, dz = jax.grad(lambda args: fp(*args).sum())((x, y, z))
    assert dx.shape == xshape
    assert dy.shape == yshape
    assert dz.shape == zshape

    self.assertAllClose(jax.grad(lambda args: fp(*args).sum())((x, y, z)),
                        jax.grad(lambda args: fv(*args).sum())((x, y, z)))

  def testPmapOutAxesBasic(self):
    @partial(pmap, in_axes=(1, None), out_axes=(2, None))
    def f(x, y):
      return jnp.sin(x + y), y * 2
    xshape = (2, jax.device_count(), 4)
    x = np.arange(prod(xshape)).reshape(xshape)
    yshape = (2, 4)
    y = np.arange(prod(yshape)).reshape(yshape)

    self.assertAllClose(f(x, y),
                        (jnp.sin(x.transpose((1, 0, 2)) + y).transpose((1, 2, 0)), y * 2))

  def testPmapDictOutAxes(self):
    # see issue #6410
    @partial(pmap, out_axes={'a': 0})
    def f(x):
      return {'a': x}
    device_count = jax.device_count()
    x = jnp.arange(device_count)
    tree_util.tree_multimap(self.assertAllClose, f(x), {'a': x})

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{in_axes}_{out_axes}",
       "in_axes": in_axes, "out_axes": out_axes}
      for in_axes in all_bdims((3, 4), (3, 1), (1, 4), pmap=True)
      for out_axes in out_bdims((3, 4), True)
  ))
  def testPmapAllAxesGrad(self, in_axes, out_axes):
    def f(x, y, z):
      return jnp.sin(x + y) * z

    pmapped_size = jax.device_count()
    mapped_shapes = [(3, 4), (3, 1), (1, 4)]
    arg_shapes = map(partial(add_bdim, pmapped_size), in_axes, mapped_shapes)
    rng = jtu.rand_default(self.rng())
    args = [rng(shape, jnp.float64) for shape in arg_shapes]
    jtu.check_grads(pmap(f, in_axes=in_axes, out_axes=out_axes), args,
                    order=2, atol=2e-2, rtol=2e-2, eps=1e-3)

  def testPmapPostProcess(self):
    def mk_case(map_fun):
      def f(x, y):
        # NOTE: Map doesn't have any arguments we differentiate wrt
        @partial(map_fun, in_axes=1, out_axes=2)
        def h(y):
          return jnp.sin(x + y)
        return h(y).sum()
      return f

    xshape = (5, 7)
    x = np.arange(prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (5, jax.device_count(), 7)
    y = np.arange(prod(yshape), dtype=np.float32).reshape(yshape)
    self.assertAllClose(jax.grad(mk_case(pmap))(x, y),
                        jax.grad(mk_case(vmap))(x, y))


class ShardedDeviceArrayTest(jtu.JaxTestCase):

  def testThreadsafeIndexing(self):
    # NOTE(skye): I picked these values to be big enough to cause interesting
    # execution overlap, but small enough to not use too much memory. YMMV.
    shape = (8, 8000, 1000)

    if jax.device_count() < shape[0]:
      raise SkipTest(f"requires {shape[0]} devices")

    x = jnp.arange(prod(shape)).reshape(shape)
    sharded_x = pmap(lambda x: x)(x)

    num_threads = 10
    futures = []
    expected = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      for i in range(num_threads):
        idx = i % shape[0]
        # Mix together different kinds of indices
        if i % 2 == 0:
          idx = slice(idx, idx + 1)
        # Use the "kwarg trick" to work around late-binding closures. See
        # https://docs.python-guide.org/writing/gotchas/#late-binding-closures.
        futures.append(executor.submit(
            lambda idx=idx: [sharded_x[idx] for _ in range(10)][0]))
        expected.append(x[idx])
      actual = [f.result() for f in futures]
    self.assertAllClose(actual, expected, check_dtypes=False)

  def testNoCopyIndexing1D(self):
    shape = (8, 4)

    if jax.device_count() < shape[0]:
      raise SkipTest(f"requires {shape[0]} devices")

    x = jnp.arange(prod(shape)).reshape(shape)
    sharded_x = pmap(lambda x: x)(x)
    self.assertIsNone(sharded_x._npy_value)
    for i in range(8):
      self.assertIsInstance(sharded_x[i], jax.interpreters.xla.DeviceArray)
    self.assertIsNone(sharded_x._npy_value)

  def test_device_put_sharded_array(self):
    devices = jax.local_devices()
    n_devices = len(devices)
    x = [np.arange(i, i + 4) for i in range(n_devices)]
    y = jax.device_put_sharded(x, devices)
    self.assertIsInstance(y, pxla.ShardedDeviceArray)
    self.assertEqual(len(y.device_buffers), len(devices))
    self.assertTrue(all(b.device() == d for b, d in zip(y.device_buffers, devices)))
    self.assertArraysEqual(y, jnp.stack(x))

  def test_device_put_sharded_pytree(self):
    devices = jax.local_devices()
    n_devices = len(devices)
    x = [(i, np.arange(i, i + 4)) for i in range(n_devices)]
    y1, y2 = jax.device_put_sharded(x, devices)
    self.assertIsInstance(y1, pxla.ShardedDeviceArray)
    self.assertArraysEqual(y1, jnp.array([a for a, _ in x]))
    self.assertTrue(all(b.device() == d for b, d in zip(y1.device_buffers, devices)))
    self.assertIsInstance(y2, pxla.ShardedDeviceArray)
    self.assertArraysEqual(y2, jnp.vstack([b for _, b in x]))
    self.assertTrue(all(b.device() == d for b, d in zip(y2.device_buffers, devices)))

  def test_device_put_replicated_array(self):
    devices = jax.local_devices()
    x = np.arange(1, 5)
    y = jax.device_put_replicated(x, devices)
    self.assertIsInstance(y, pxla.ShardedDeviceArray)
    self.assertEqual(len(y.device_buffers), len(devices))
    self.assertTrue(all(b.device() == d for b, d in zip(y.device_buffers, devices)))
    self.assertArraysEqual(y, np.stack([x for _ in devices]))

  def test_device_put_replicated_pytree(self):
    devices = jax.local_devices()
    xs = {'a': np.arange(1, 5), 'b': np.arange(3)}
    ys = jax.device_put_replicated(xs, devices)
    self.assertIsInstance(ys, dict)
    y1, y2 = ys['a'], ys['b']

    self.assertIsInstance(y1, pxla.ShardedDeviceArray)
    self.assertEqual(len(y1.device_buffers), len(devices))
    self.assertTrue(all(b.device() == d for b, d in zip(y1.device_buffers, devices)))
    self.assertArraysEqual(y1, np.stack([xs['a'] for _ in devices]))

    self.assertIsInstance(y2, pxla.ShardedDeviceArray)
    self.assertEqual(len(y2.device_buffers), len(devices))
    self.assertTrue(all(b.device() == d for b, d in zip(y2.device_buffers, devices)))
    self.assertArraysEqual(y2, np.stack([xs['b'] for _ in devices]))

  def test_repr(self):
    x = jax.device_put_replicated(1, jax.devices())
    self.assertStartsWith(repr(x), 'ShardedDeviceArray')

  def test_delete_is_idempotent(self):
    x = jax.device_put_replicated(1, jax.devices())
    x.delete()
    x.delete()

    with self.assertRaisesRegex(ValueError,
                                'ShardedDeviceArray has been deleted.'):
      _ = x[0]


class SpecToIndicesTest(jtu.JaxTestCase):

  def testShardsPerAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=map(pxla.Chunked, ([2], [2])),
                             mesh_mapping=map(pxla.ShardedAxis, (0, 1)))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((slice(0,2), slice(0,4)),
                      (slice(0,2), slice(4,8)),
                      (slice(2,4), slice(0,4)),
                      (slice(2,4), slice(4,8))))

  def testShardedAxisPermutation(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=map(pxla.Chunked, ([2], [2])),
                             mesh_mapping=map(pxla.ShardedAxis, (1, 0)))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((slice(0,2), slice(0,4)),
                      (slice(2,4), slice(0,4)),
                      (slice(0,2), slice(4,8)),
                      (slice(2,4), slice(4,8))))

  def testShardedAxisPermutationAndReplication(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=map(pxla.Chunked, ([2], [2])),
                             mesh_mapping=(pxla.Replicated(2),
                                           pxla.ShardedAxis(1),
                                           pxla.ShardedAxis(0)))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((slice(0,2), slice(0,4)),
                      (slice(2,4), slice(0,4)),
                      (slice(0,2), slice(4,8)),
                      (slice(2,4), slice(4,8))) * 2)

  def testUnshardedAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Chunked([2]), pxla.NoSharding()),
                             mesh_mapping=(pxla.ShardedAxis(0),))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((slice(0,2), slice(None)),
                      (slice(2,4), slice(None))))

  def testNoSharding(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.NoSharding(), pxla.NoSharding()),
                             mesh_mapping=())
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((slice(None), slice(None)),))

  def testUnmaterializedAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(4), pxla.NoSharding()),
                             mesh_mapping=(pxla.ShardedAxis(0),))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((0, slice(None)),
                      (1, slice(None)),
                      (2, slice(None)),
                      (3, slice(None))))

    shape = (2, 2)
    spec = pxla.ShardingSpec(sharding=(pxla.NoSharding(), pxla.Unstacked(2)),
                             mesh_mapping=(pxla.ShardedAxis(0),))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((slice(None), 0),
                      (slice(None), 1)))

  def testReplicationAfterUnsharded(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.NoSharding()),
                             mesh_mapping=(pxla.ShardedAxis(0), pxla.Replicated(3)))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     tuple([(0, slice(None))] * 3 + [(1, slice(None))] * 3))

  def testReplicationPosition2(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.Chunked([2])),
                             mesh_mapping=(pxla.ShardedAxis(0), pxla.ShardedAxis(1), pxla.Replicated(3)))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((0, slice(0, 4)), (0, slice(0, 4)), (0, slice(0, 4)),
                      (0, slice(4, 8)), (0, slice(4, 8)), (0, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(0, 4)), (1, slice(0, 4)),
                      (1, slice(4, 8)), (1, slice(4, 8)), (1, slice(4, 8))))

  def testReplicationPosition1(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.Chunked([2])),
                             mesh_mapping=(pxla.ShardedAxis(0), pxla.Replicated(3), pxla.ShardedAxis(1)))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((0, slice(0, 4)), (0, slice(4, 8)),
                      (0, slice(0, 4)), (0, slice(4, 8)),
                      (0, slice(0, 4)), (0, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(4, 8))))

  def testReplicationPosition0(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.NoSharding()),
                             mesh_mapping=(pxla.Replicated(3), pxla.ShardedAxis(0)))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     tuple([(0, slice(None)), (1, slice(None))] * 3))

  def testMultipleReplications(self):
    shape = (2, 7, 4)
    spec = pxla.ShardingSpec(
        sharding=(pxla.Unstacked(2), pxla.NoSharding(), pxla.Chunked([2])),
        mesh_mapping=(pxla.Replicated(3), pxla.Replicated(2),
                      pxla.ShardedAxis(0), pxla.Replicated(2),
                      pxla.ShardedAxis(1)))
    self.assertEqual(
        pxla.spec_to_indices(shape, spec),
        ((0, slice(None), slice(0, 2)), (0, slice(None), slice(2, 4)),
         (0, slice(None), slice(0, 2)), (0, slice(None), slice(2, 4)),
         (1, slice(None), slice(0, 2)), (1, slice(None), slice(2, 4)),
         (1, slice(None), slice(0, 2)), (1, slice(None), slice(2, 4))) * 3 * 2)

  def testReplicatedScalar(self):
    shape = ()
    spec = pxla.ShardingSpec(sharding=(),
                             mesh_mapping=(pxla.Replicated(3),))
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((), (), ()))


def _spec_str(spec):
  return (f"({spec.sharding},"
          f"{spec.mesh_mapping},)")


class ShardArgsTest(jtu.JaxTestCase):

  def numpy_array(x):
    return x

  def device_array(x):
    return jax.device_put(x)

  # TODO(skye): add coverage for ShardedDeviceArrays

  @parameterized.named_parameters(
      {"testcase_name":
       f"_shape={shape}_spec={_spec_str(spec)}_arg={make_arg.__name__}"
       .replace(" ", ""),
       "shape": shape, "spec": spec, "make_arg": make_arg}
      for make_arg in [numpy_array, device_array]
      for shape, spec in [
          # pmap(in_axes=0)
          [(4, 8), pxla.ShardingSpec(sharding=(pxla.Unstacked(4), pxla.NoSharding()),
                                     mesh_mapping=(pxla.ShardedAxis(0),))],
          # pmap(in_axes=1)
          [(2, 2), pxla.ShardingSpec(sharding=(pxla.NoSharding(), pxla.Unstacked(2)),
                                     mesh_mapping=(pxla.ShardedAxis(0),))],
          # unsharded
          [(4, 8), pxla.ShardingSpec(sharding=(pxla.NoSharding(), pxla.NoSharding()),
                                     mesh_mapping=())],
          # partitioned, 1 axis
          [(4, 8), pxla.ShardingSpec(sharding=(pxla.Chunked([2]), pxla.NoSharding()),
                                     mesh_mapping=(pxla.ShardedAxis(0),))],
          # partitioned, 2 axes
          [(4, 8), pxla.ShardingSpec(sharding=(pxla.Chunked([2]), pxla.Chunked([2])),
                                     mesh_mapping=map(pxla.ShardedAxis, (0, 1)))],
          # partitioned, 2 axes, permuted
          [(4, 8), pxla.ShardingSpec(sharding=(pxla.Chunked([2]), pxla.Chunked([2])),
                                     mesh_mapping=map(pxla.ShardedAxis, (1, 0)))],
          # partitioned + sharding
          [(2, 8), pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.Chunked([2])),
                                     mesh_mapping=map(pxla.ShardedAxis, (0, 1)))],
          # replication + sharding
          [(2, 8), pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.NoSharding()),
                                     mesh_mapping=(pxla.ShardedAxis(0), pxla.Replicated(3)))],
          # replication, no sharding
          [(2, 8), pxla.ShardingSpec(sharding=(pxla.NoSharding(), pxla.NoSharding()),
                                     mesh_mapping=(pxla.Replicated(3),))],
          # multiple replicated axes
          [(1, 8), pxla.ShardingSpec(sharding=(pxla.Unstacked(1), pxla.Chunked([2])),
                                     mesh_mapping=(pxla.Replicated(2), pxla.ShardedAxis(0),
                                                   pxla.Replicated(2), pxla.ShardedAxis(1)))],
          # replicated scalar
          [(), pxla.ShardingSpec(sharding=(),
                                 mesh_mapping=(pxla.Replicated(2), pxla.Replicated(3)))],
      ])
  def testShardArgs(self, shape, spec, make_arg):
    indices = pxla.spec_to_indices(shape, spec)
    nshards = len(indices)
    if jax.device_count() < nshards:
      raise SkipTest
    x = np.arange(prod(shape)).reshape(shape)
    arg = make_arg(x)
    bufs = pxla.shard_args(jax.devices()[:nshards],
                           [indices], [arg])
    self.assertEqual(len(bufs), 1)
    self.assertEqual(len(bufs[0]), nshards)
    for buf, idx in zip(bufs[0], indices):
      self.assertAllClose(buf.to_py(), x[idx], check_dtypes=False)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
