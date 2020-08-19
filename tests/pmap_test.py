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
from unittest import SkipTest, skipIf
import warnings
import weakref

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import test_util as jtu
from jax import tree_util
from jax import lax
from jax import mask
from jax import random
from jax.abstract_arrays import ShapedArray
from jax.api import (pmap, soft_pmap, jit, vmap, jvp, grad, make_jaxpr,
                     linearize, device_put)
from jax.lib import xla_bridge
from jax.util import prod, safe_map
from jax.interpreters import pxla
from jax.interpreters import xla

from jax.config import config
config.parse_flags_with_absl()

prev_xla_flags = None

# TODO(jakevdp): move the following to test_util.py
compatible_shapes = [[(3,)], [(3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]

def all_bdims(*shapes):
  bdims = (it.chain([cast(Optional[int], None)],
                     range(len(shape) + 1)) for shape in shapes)
  return (t for t in it.product(*bdims) if not all(e is None for e in t))

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

ignore_soft_pmap_warning = partial(
  jtu.ignore_warning, message="soft_pmap is an experimental.*")

ignore_jit_of_pmap_warning = partial(
  jtu.ignore_warning, message=".*jit-of-pmap.*")


class PmapTest(jtu.JaxTestCase):
  def _getMeshShape(self, device_mesh_shape):
    device_count = xla_bridge.device_count()
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
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.sum(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMean(self):
    f = pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.broadcast_to(np.mean(x, 0), x.shape)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGather(self):
    f = pmap(lambda x: lax.all_gather(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = np.array([x] * xla_bridge.device_count())
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTrees(self):
    ptranspose = lambda x, axis_name: lax.all_to_all(x, axis_name, 0, 0)
    def protate(x, axis_name):
      n = lax.psum(1, axis_name)
      return lax.ppermute(x, axis_name, [(i, (i + 1) % n) for i in range(n)])

    tree_f = lambda f: partial(tree_util.tree_map, f)
    jax_f = lambda p: pmap(lambda x: p(x, 'i'), 'i')
    np_f = lambda p: tree_f(lambda x: np.broadcast_to(p(x, 0), x.shape))
    np_transpose = tree_f(np.transpose)
    np_rotate = tree_f(lambda x: np.concatenate([x[-1:], x[:-1]]))

    n = xla_bridge.device_count()
    x = {'a': np.arange(1 * n * n, 2 * n * n).reshape([n, n]),
         'b': np.arange(2 * n * n, 3 * n * n).reshape([n, n]),
         'c': np.arange(4 * n * n, 5 * n * n).reshape([n, n])}

    assert_allclose = partial(tree_util.tree_multimap,
                              partial(self.assertAllClose, check_dtypes=False))
    assert_allclose(jax_f(lax.pmax)(x), np_f(np.max)(x))
    assert_allclose(jax_f(lax.pmin)(x), np_f(np.min)(x))
    assert_allclose(jax_f(lax.psum)(x), np_f(np.sum)(x))
    assert_allclose(jax_f(lax.pmean)(x), np_f(np.mean)(x))
    if jtu.device_under_test() not in ("cpu", "gpu"):
      # NOTE: all-to-all and ppermute only supported on TPU.
      assert_allclose(jax_f(ptranspose)(x), np_transpose(x))
      assert_allclose(jax_f(protate)(x), np_rotate(x))

  def testCollectivesWithTreesOfDifferentDtypes(self):
    n = len(jax.devices())
    x = {'a': np.arange(1 * n * n, 2 * n * n, dtype=np.float32).reshape([n, n]),
         'b': np.arange(2 * n * n, 3 * n * n, dtype=np.int32).reshape([n, n]),
         'c': np.arange(4 * n * n, 5 * n * n, dtype=np.float32).reshape([n, n]),
         'd': np.arange(6 * n * n, 7 * n * n, dtype=np.int32).reshape([n, n])}
    tree_f = lambda f: partial(tree_util.tree_map, f)
    jax_f = lambda p: pmap(lambda x: p(x, 'i'), 'i')
    np_f = lambda p: tree_f(lambda x: np.broadcast_to(p(x, 0), x.shape))
    assert_allclose = partial(tree_util.tree_multimap,
                              partial(self.assertAllClose, check_dtypes=False))
    assert_allclose(jax_f(lax.pmax)(x), np_f(np.max)(x))
    assert_allclose(jax_f(lax.pmin)(x), np_f(np.min)(x))
    assert_allclose(jax_f(lax.psum)(x), np_f(np.sum)(x))
    assert_allclose(jax_f(lax.pmean)(x), np_f(np.mean)(x))

  def testComplexPsum(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4 * 2)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape).view(np.complex64)
    expected = x - np.sum(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)


  def testNestedBasic(self):
    f = lambda x: lax.psum(lax.psum(x, 'i'), 'j')
    f = pmap(pmap(f, 'i'), 'j')

    def sum_and_broadcast(x, axis):
      return np.repeat(np.sum(x, axis, keepdims=True), x.shape[axis], axis)

    shape = (xla_bridge.device_count(), 1, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)
    expected = sum_and_broadcast(sum_and_broadcast(x, 0), 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMismatchedAxisSizes(self):
    n = xla_bridge.device_count()
    f = pmap(lambda x, y: x + y)
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
    f = pmap(pmap(f, 'i'), 'j')

    shape = mesh_shape + (4,)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)
    expected = x
    self.assertEqual(ans.shape, expected.shape)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPartiallyMapped(self):
    f = pmap(lambda x, y: x, in_axes=(None, 0))
    g = pmap(lambda x, y: x - lax.psum(y, 'i'), axis_name='i', in_axes=(None, 0))

    mesh_shape = (xla_bridge.device_count(),)
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
    self.assertEmpty(f_ans.sharding_spec.replication_factors)

    g_expected = np.broadcast_to(x - np.sum(y, 0, keepdims=True), shape)
    g_ans = g(x, y)
    self.assertAllClose(g_ans, g_expected)
    self.assertIsInstance(g_ans, pxla.ShardedDeviceArray)
    self.assertEmpty(g_ans.sharding_spec.replication_factors)

  @parameterized.named_parameters(
      {"testcase_name": "_mesh={}".format(device_mesh_shape).replace(" ", ""),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testPartiallyMappedNested(self, device_mesh_shape):
    mesh_shape = self._getMeshShape(device_mesh_shape)

    f = pmap(lambda x, y: x - lax.psum(y, 'i'), axis_name='i', in_axes=(None, 0))
    f = pmap(f, axis_name='j', in_axes=(None, 0))

    x = 3.
    y = np.arange(prod(mesh_shape), dtype=np.float32).reshape(mesh_shape)
    expected = np.broadcast_to(x - np.sum(y, 1, keepdims=True), mesh_shape)

    ans = f(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testJvpAndPartialEval(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return jnp.sin(x)

    def splitjvp(x):
      _, jvp = linearize(f, x)
      return jvp(jnp.ones_like(x))

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = np.cos(x)

    ans = splitjvp(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    make_jaxpr(splitjvp)(x)  # doesn't crash

  def testGradBasic(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return jnp.sin(x)

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(jnp.sin(x)))(x)
    expected = grad(lambda x: jnp.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGradOfPsum(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return lax.psum(x, axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    jtu.check_grads(f, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2, eps=1.)

  def testGradOfJvp(self):
    @partial(pmap, axis_name='i')
    def f(x):
      return jnp.sin(x)

    def splitjvp(x):
      _, jvp = linearize(f, x)
      return jvp(jnp.ones_like(x))

    fun = lambda x: jnp.sum(jvp(jnp.sin, (x,), (jnp.ones_like(x),))[1])

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(splitjvp(x)))(x)
    expected = grad(fun)(x)
    self.assertAllClose(ans, expected)

  def testTwoArgsGrad(self):
    def f(x, y):
      return lax.psum(5. * jnp.cos(x) * jnp.sin(y), 'i')
    f = pmap(f, 'i')

    def g(x, y):
      tot = jnp.sum(5. * jnp.cos(x) * jnp.sin(y))
      return tot * jnp.ones_like(x)  # broadcast to map like pjit does

    shape = (xla_bridge.device_count(), 4)
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

    @partial(pmap, axis_name='i')
    def test_fun(x):
      y = jnp.sum(jnp.sin(x))

      @partial(pmap, axis_name='j')
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
    f = pmap(f, axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    # test that we can pass in and out ShardedDeviceArrays
    y = f(x)
    self.assertIsInstance(y, jnp.ndarray)
    self.assertIsInstance(y, pxla.ShardedDeviceArray)
    self.assertAllClose(y, 2 * x, check_dtypes=False)
    z = f(y)
    self.assertIsInstance(z, pxla.ShardedDeviceArray)
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can pass in a regular DeviceArray
    y = f(device_put(x))
    self.assertIsInstance(y, pxla.ShardedDeviceArray)
    self.assertAllClose(y, 2 * x, check_dtypes=False)

    # test that we can pass a ShardedDeviceArray to a regular jit computation
    z = y + y
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can handle device movement on dispatch
    y.device_buffers = y.device_buffers[::-1]
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
    if xla_bridge.device_count() < max(in_shape[:1] + out_shape[:1]):
      raise SkipTest("not enough devices")

    x = np.arange(prod(in_shape)).reshape(in_shape)
    sharded_x = pmap(lambda x: x)(x)
    self.assertAllClose(sharded_x.reshape(out_shape), x.reshape(out_shape),
                        check_dtypes=False)

  def testPsumMultiple(self):
    f = lambda x: lax.psum(x, ('i', 'j'))
    f = pmap(pmap(f, 'i'), 'j')

    def sum_and_broadcast(x, axis):
      return np.repeat(np.sum(x, axis, keepdims=True), x.shape[axis], axis)

    device_count = xla_bridge.device_count()
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
    replicas = xla_bridge.device_count()
    if replicas % 2 != 0:
      raise SkipTest
    axis_index_groups = np.arange(replicas).reshape(
      2, replicas // 2).tolist()
    f = lambda x: x - lax.psum(2., 'i', axis_index_groups=axis_index_groups)
    f = pmap(f, 'i')

    shape = (replicas, 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected_psum = 2. * replicas // 2
    expected = x - expected_psum

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPsumReplicaGroups(self):
    replicas = xla_bridge.device_count()
    if replicas % 2 != 0:
      raise SkipTest
    axis_index_groups = np.arange(replicas).reshape(
      2, replicas // 2).tolist()
    f = lambda x: x - lax.psum(x, 'i', axis_index_groups=axis_index_groups)
    f = pmap(f, 'i')

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

  def testNestedPmapReplicaGroups(self):
    replicas = xla_bridge.device_count()
    if replicas % 4 != 0:
      raise SkipTest
    axis_index_groups = np.arange(replicas // 2).reshape(
        2, replicas // 4).tolist()
    f = lambda x: x - lax.psum(x, 'i', axis_index_groups=axis_index_groups)
    f1 = pmap(pmap(f, 'i'), 'j')
    f2 = pmap(lambda x: pmap(f, 'i')(x) + 1., 'j')  # "imperfectly nested" case
    f3 = pmap(pmap(f, 'j'), 'i')

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
    axis_env = xla.AxisEnv(8, ('i', 'j'), (4, 2), None)
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

  @jtu.skip_on_devices("cpu", "gpu")
  def testCollectivePermute(self):
    device_count = xla_bridge.device_count()
    rotation = [(i, (i + 1) % device_count) for i in range(device_count)]
    f = lambda x: lax.ppermute(x, perm=rotation, axis_name='i')
    f = pmap(f, 'i')

    x = jnp.arange(4 * device_count).reshape((device_count, 4))
    ans = f(x)
    expected = np.roll(x, shift=1, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu", "gpu")
  def testCollectivePermuteGrad(self):
    device_count = xla_bridge.device_count()
    shift_right = [(i, (i + 1)) for i in range(device_count - 1)]
    f = lambda x: lax.ppermute(x, perm=shift_right, axis_name='i')
    y = np.pi + np.arange(device_count, dtype=np.float32)
    g = lambda x: jnp.sum(y * pmap(f, 'i')(x))

    x = np.arange(device_count, dtype=np.float32)
    ans = grad(g)(x)
    expected = np.concatenate([np.pi + np.arange(1, device_count), [0]])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu", "gpu")
  def testCollectivePermuteCyclicGrad(self):
    device_count = xla_bridge.device_count()
    shift_right = [(i, (i + 1) % device_count) for i in range(device_count)]
    f = lambda x: lax.ppermute(x, perm=shift_right, axis_name='i')
    y = np.pi + np.arange(device_count, dtype=np.float32)
    g = lambda x: jnp.sum(y * pmap(f, 'i')(x))

    x = np.arange(device_count, dtype=np.float32)

    ans = grad(g)(x)
    expected = np.roll(np.pi + np.arange(device_count), -1)
    self.assertAllClose(ans, expected, check_dtypes=False)

    jtu.check_grads(g, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2)

  @jtu.skip_on_devices("cpu")
  def testCollectivePermuteCyclicWithPShuffle(self):
    device_count = xla_bridge.device_count()
    values = np.arange(device_count)
    shift_right = [(i - 1) % device_count for i in range(device_count)]
    f = lambda x: lax.pshuffle(x, perm=shift_right, axis_name='i')
    expected = np.roll(values, 1)
    ans = np.asarray(pmap(f, "i")(values))
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu")
  def testPShuffleWithBadPerm(self):
    device_count = xla_bridge.device_count()
    bad_perm = list(range(device_count))
    bad_perm[0] = 1
    f = lambda x: lax.pshuffle(x, perm=bad_perm, axis_name='i')
    g = lambda: pmap(f, "i")(np.arange(device_count))
    self.assertRaisesRegex(
      ValueError,
      "`perm` does not represent a permutation: \\[1.*\\]", g)

  @jtu.skip_on_devices("cpu", "gpu")
  def testPpermuteWithZipObject(self):
    # https://github.com/google/jax/issues/1703
    num_devices = xla_bridge.device_count()
    perm = [num_devices - 1] + list(range(num_devices - 1))
    f = pmap(lambda x: lax.ppermute(x, "i", zip(perm, range(num_devices))), "i")
    result = f(jnp.arange(num_devices, dtype=jnp.float32))
    expected = jnp.asarray(perm, dtype=jnp.float32)
    self.assertAllClose(result, expected)

  @jtu.skip_on_devices("cpu", "gpu")
  def testRule30(self):
    # This is a test of collective_permute implementing a simple halo exchange
    # to run a rule 30 simulation: https://en.wikipedia.org/wiki/Rule_30
    # Halo exchange should be useful in spatially-sharded convolutions and in
    # other simulations.
    device_count = xla_bridge.device_count()

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

    @partial(pmap, axis_name='i')
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

  @jtu.skip_on_devices("cpu", "gpu")
  def testReduceMax(self):
    f = pmap(lambda x: x - lax.pmax(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.max(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("cpu", "gpu")
  def testReduceMin(self):
    f = pmap(lambda x: x - lax.pmin(x, 'i'), axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.min(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDeviceCountError(self):
    device_count = xla_bridge.device_count()

    f = pmap(lambda x: x)
    x = jnp.arange(device_count + 1)
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

    f = pmap(lambda x: x)
    x = np.ones((device_count + 1, 10))
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

    f = pmap(lambda x: pmap(lambda x: x)(x))
    x = np.ones((device_count, 2, 10))
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

  def testPmapConstant(self):
    device_count = xla_bridge.device_count()
    f = pmap(lambda x: 3)
    x = jnp.arange(device_count)
    with jtu.count_jit_and_pmap_compiles() as count:
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): fix this
    expected = np.repeat(3, device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

    f = pmap(lambda x: (x, 3))
    x = np.arange(device_count)
    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      _, ans = f(x)
    self.assertEqual(count[0], 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPmapConstantDevices(self):
    if xla_bridge.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    devices = xla_bridge.devices()[:-1]
    shuffle(devices)
    f = pmap(lambda x: 3, devices=devices)
    x = jnp.arange(len(devices))
    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = np.repeat(3, len(devices))
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    self.assertEqual([b.device() for b in ans.device_buffers], devices)

  def testPmapConstantError(self):
    device_count = xla_bridge.device_count()
    f = pmap(lambda x: 3)
    x = jnp.arange(device_count + 1)
    if config.omnistaging_enabled:
      self.assertRaisesRegex(
          ValueError,
          (r"compiling computation that requires \d+ logical devices, "
          r"but only \d+ XLA devices are available .*"),
          lambda: f(x))

      # TODO(mattjj): test error message with explicit devices
      # f = pmap(lambda x: 3, devices=[xla_bridge.devices()[0]])
      # x = jnp.arange(2)
      # self.assertRaisesRegex(
      #     ValueError, r"Cannot replicate across \d+ replicas because only \d+ "
      #     r"local devices are available.", lambda: f(x))
    else:
      self.assertRaisesRegex(
          ValueError, r"Cannot replicate across \d+ replicas because only \d+ "
          r"local devices are available.", lambda: f(x))

      f = pmap(lambda x: 3, devices=[xla_bridge.devices()[0]])
      x = jnp.arange(2)
      self.assertRaisesRegex(
          ValueError, "Cannot replicate across 2 replicas because only 1 "
          "local devices are available.", lambda: f(x))

  def testNestedPmapConstant(self):
    if xla_bridge.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    f = pmap(pmap(lambda x: 3))
    shape = (2, xla_bridge.device_count() // 2, 3)
    x = jnp.arange(prod(shape)).reshape(shape)
    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = 3 * np.ones(shape[:2])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    expected_sharded = pmap(pmap(lambda x: x))(expected)
    self.assertEqual([b.device() for b in ans.device_buffers],
                     [b.device() for b in expected_sharded.device_buffers])

    f = pmap(pmap(lambda x: (x, 3)))
    x_sharded, ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)
    self.assertEqual([b.device() for b in ans.device_buffers],
                     [b.device() for b in x_sharded.device_buffers])

  def testNestedPmapConstantDevices(self):
    raise SkipTest("Nested pmaps with devices not yet implemented")

    if xla_bridge.device_count() < 6:
      raise SkipTest("this test requires >= 6 devices")

    devices = xla_bridge.devices()[:-2]
    shuffle(devices)
    f = pmap(pmap(lambda x: 3), devices=devices)
    shape = (2, len(devices) // 2, 3)
    x = jnp.arange(prod(shape)).reshape(shape)
    with jtu.count_jit_and_pmap_compiles() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = 3 * np.ones(shape[:2])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    expected_sharded = pmap(pmap(lambda x: x), devices=devices)(expected)
    self.assertEqual([b.device() for b in ans.device_buffers],
                     [b.device() for b in expected_sharded.device_buffers])

  def testNestedPmapConstantError(self):
    f = pmap(pmap(lambda x: 3))
    shape = (2, xla_bridge.device_count() // 2 + 1, 3)
    x = jnp.arange(prod(shape)).reshape(shape)
    if config.omnistaging_enabled:
      self.assertRaisesRegex(
          ValueError,
          (r"compiling computation that requires \d+ logical devices, "
          r"but only \d+ XLA devices are available .*"),
          lambda: f(x))

      # TODO(mattjj): check error message with explicit devices
      # if xla_bridge.device_count() > 1:
      #   f = pmap(pmap(lambda x: 3), devices=xla_bridge.devices()[:-1])
      #   shape = (2, xla_bridge.device_count() // 2, 3)
      #   x = jnp.arange(prod(shape)).reshape(shape)
      #   self.assertRaisesRegex(
      #       ValueError,
      #       (r"compiling computation that requires \d+ replicas, "
      #        r"but only \d+ XLA devices are available"),
      #       lambda: f(x))
    else:
      self.assertRaisesRegex(
          ValueError, r"Cannot replicate across \d+ replicas because only \d+ "
          r"local devices are available.", lambda: f(x))

      if xla_bridge.device_count() > 1:
        f = pmap(pmap(lambda x: 3), devices=xla_bridge.devices()[:-1])
        shape = (2, xla_bridge.device_count() // 2, 3)
        x = jnp.arange(prod(shape)).reshape(shape)
        self.assertRaisesRegex(
            ValueError, r"Cannot replicate across \d+ replicas because only \d+ "
            r"local devices are available.", lambda: f(x))

  def testCollectiveConstant(self):
    device_count = xla_bridge.device_count()
    f = pmap(lambda x: lax.psum(1, 'i'), 'i')
    x = jnp.arange(device_count)
    ans = f(x)
    expected = np.repeat(device_count, device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCollectiveConstantNested(self):
    device_count = xla_bridge.device_count()

    @partial(pmap, axis_name='i')
    def f(x):
      @partial(pmap, axis_name='j')
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
    device_count = xla_bridge.device_count()
    f = pmap(lambda x: x + lax.axis_index('i'), 'i')
    x = jnp.ones(device_count)
    ans = f(x)
    expected = 1 + np.arange(device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testVmapOfPmap(self):
    device_count = xla_bridge.device_count()
    f0 = lambda x: x
    f1 = pmap(f0, axis_name='i')
    ax = np.random.randn(2, device_count, 50, 60)
    bx = vmap(f1)(ax)
    self.assertAllClose(ax, bx, check_dtypes=False)

  def testVmapOfPmap2(self):
    N_DEVICES = xla_bridge.device_count()
    keys = random.split(random.PRNGKey(1), 13)  # [13, 2]

    @pmap
    def g(key):
      _ = random.normal(key, ())
      return 0.

    @vmap
    def s(keys):
      keys = jnp.broadcast_to(keys, (N_DEVICES,) + keys.shape)
      return g(keys)

    ans = s(keys)  # doesn't crash
    self.assertEqual(ans.shape, (13, N_DEVICES))

  def testVmapOfPmap3(self):
    # https://github.com/google/jax/issues/3399
    device_count = xla_bridge.device_count()
    if device_count < 2:
      raise SkipTest("test requires at least two devices")

    def map_version(qs, pts):
      return jax.lax.map(lambda x: func(x, pts), qs)

    def vmap_version(qs, pts):
      return jax.vmap(func, in_axes=(0, None))(qs, pts)

    def func(q, pts):
      q_from_pmap = jax.pmap(lambda x, y: y, in_axes=(0, None))(pts, q)
      return q, q_from_pmap

    pts = jnp.ones(device_count)
    qs = jnp.asarray(((0,0), (3,3), (2,2)))

    _, expected = map_version(qs, pts)
    _, ans = vmap_version(qs, pts)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testVmapOfPmapNonLeadingAxis(self):
    device_count = xla_bridge.device_count()
    f0 = lambda x: x
    f1 = pmap(f0, axis_name='i')
    ax = np.random.randn(device_count, 2, 50, 60)
    bx = vmap(f1, in_axes=2, out_axes=2)(ax)
    self.assertAllClose(ax, bx, check_dtypes=False)

  def testVmapOfPmapTuple(self):
    device_count = xla_bridge.device_count()
    f0 = lambda *x: x
    f1 = pmap(f0, axis_name='i')

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

  @jtu.skip_on_devices("gpu")
  def testPswapaxes(self):
    device_count = xla_bridge.device_count()
    # TODO: AllToAll not yet implemented on XLA:CPU
    if jtu.device_under_test() == "cpu":
      device_count = 1
    shape = (device_count, 3, device_count, 5)
    x = np.arange(prod(shape)).reshape(shape)

    ans = pmap(lambda x: lax.pswapaxes(x, 'i', 1), axis_name='i')(x)
    expected = np.swapaxes(x, 0, 2)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.skip_on_devices("gpu")
  def testGradOfPswapaxes(self):
    device_count = xla_bridge.device_count()
    # TODO: AllToAll not yet implemented on XLA:CPU
    if jtu.device_under_test() == "cpu":
      device_count = 1
    shape = (device_count, 1, device_count)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    w = np.arange(device_count, dtype=np.float32)

    @partial(pmap, axis_name='i')
    def f(x, w):
      g = lambda x: jnp.sum(lax.pswapaxes(x, 'i', 1) * w)
      return grad(g)(x)

    ans = f(x, w)
    expected = np.tile(w, reps=device_count).reshape(shape)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReshardInput(self):
    if xla_bridge.device_count() < 6:
      raise SkipTest("testReshardInput requires 6 devices")
    # Manually construct a ShardedDeviceArray with the wrong sharding for the
    # subsequent pmap
    shard_shape = (3,2)
    shard = jnp.arange(prod(shard_shape)).reshape(shard_shape)
    bufs = [xla.device_put(shard, d) for d in xla_bridge.devices()[:4]]
    aval = ShapedArray((6,4), shard.dtype)
    sharding_spec = pxla.ShardingSpec(
        shards_per_axis=(2, 2),
        is_axis_materialized=(True, True),
        replication_factors=[])
    arr = pxla.ShardedDeviceArray(aval, sharding_spec, bufs)

    r = pmap(lambda x: x + 1)(arr)
    self.assertAllClose(r, arr + 1)
    self.assertEqual(len(r.device_buffers), 6)

  @ignore_soft_pmap_warning()
  def testSoftPmapBatchMatmul(self):
    if not config.omnistaging_enabled: raise SkipTest("requires omnistaging")
    n = 4 * xla_bridge.device_count()
    xs = np.arange(n * 2 * 3).reshape(n, 2, 3)
    ys = np.arange(n * 3 * 4).reshape(n, 3, 4)
    ans = soft_pmap(jnp.dot, 'i')(xs, ys)
    expected = np.einsum('nij,njk->nik', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSoftPmapBatchMatmulJit(self):
    if not config.omnistaging_enabled: raise SkipTest("requires omnistaging")
    n = 4 * xla_bridge.device_count()
    xs = np.arange(n * 2 * 3).reshape(n, 2, 3)
    ys = np.arange(n * 3 * 4).reshape(n, 3, 4)
    ans = soft_pmap(jit(jnp.dot), 'i')(xs, ys)
    expected = np.einsum('nij,njk->nik', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSoftPmapPsumConstant(self):
    if not config.omnistaging_enabled: raise SkipTest("requires omnistaging")
    n = 4 * xla_bridge.device_count()
    def f(_):
      return lax.psum(1, 'i')
    ans = soft_pmap(f, 'i')(jnp.ones(n))
    expected = n * np.ones(n)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSoftPmapPsum(self):
    if not config.omnistaging_enabled: raise SkipTest("requires omnistaging")
    n = 4 * xla_bridge.device_count()
    def f(x):
      return x / lax.psum(x, 'i')
    ans = soft_pmap(f, 'i')(jnp.ones(n))
    expected = np.ones(n) / n
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSoftPmapAxisIndex(self):
    if not config.omnistaging_enabled: raise SkipTest("requires omnistaging")
    n = 4 * xla_bridge.device_count()
    def f(x):
      return x * lax.axis_index('i')
    ans = soft_pmap(f, 'i')(2 * jnp.ones(n))
    expected = 2 * np.arange(n)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSoftPmapOfJit(self):
    if not config.omnistaging_enabled: raise SkipTest("requires omnistaging")
    n = 4 * xla_bridge.device_count()
    def f(x):
      return 3 * x
    ans = soft_pmap(jit(f), 'i')(np.arange(n))
    expected = 3 * np.arange(n)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSoftPmapNested(self):
    raise SkipTest("not implemented")  # TODO(mattjj): re-implement
    n = 4 * xla_bridge.device_count()

    @partial(soft_pmap, axis_name='i')
    @partial(soft_pmap, axis_name='j')
    def f(x):
      i_size = lax.psum(1, 'i')
      return x + lax.axis_index('i') + i_size * lax.axis_index('j')

    ans = f(jnp.zeros((n, n)))
    expected = np.arange(n ** 2).reshape(n, n).T
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testGradOfSoftPmap(self):
    raise SkipTest("not implemented")  # TODO(mattjj): re-implement
    n = 4 * xla_bridge.device_count()

    @partial(soft_pmap, axis_name='i')
    def f(x):
      return x * lax.axis_index('i')

    ans = grad(lambda x: jnp.sum(f(x)))(jnp.zeros((n, n)))
    expected = np.repeat(np.arange(n)[:, None], n, axis=1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSoftPmapDevicePersistence(self):
    if not config.omnistaging_enabled: raise SkipTest("requires omnistaging")
    device_count = xla_bridge.device_count()
    shape = (2 * 2 * device_count, 2, 3)

    # check that we can maintain device persistence across calls
    x = np.arange(prod(shape)).reshape(shape)
    x = soft_pmap(lambda x: x)(x)
    self.assertIsInstance(x, pxla.ShardedDeviceArray)
    x._npy_value = np.float32(np.nan)  # can't be coerced to ndarray for xfer
    x = soft_pmap(lambda x: x)(x)  # doesn't crash
    self.assertIsInstance(x, pxla.ShardedDeviceArray)

  def testSoftPmapAllToAll(self):
    raise SkipTest("the underlying code here is broken")  # TODO(mattjj)
    n = 4 * xla_bridge.device_count()
    def f(x):
      return lax.all_to_all(x, 'i', 0, 0)
    ans = soft_pmap(f, 'i')(jnp.arange(n ** 2).reshape(n, n))
    expected = np.arange(n ** 2).reshape(n, n).T
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testShardedDeviceArrayBlockUntilReady(self):
    x = np.arange(xla_bridge.device_count())
    x = pmap(lambda x: x)(x)
    x.block_until_ready()  # doesn't crash

  @ignore_jit_of_pmap_warning()
  def testJitPmapComposition(self):
    f = lambda x: x - lax.psum(x, 'i')

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.sum(x, 0)

    ans = jit(pmap(f, 'i'))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = pmap(jit(f), 'i')(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMakeJaxprOfOpenSpmd(self):
    f = lambda x: x - lax.psum(x, 'i')
    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    make_jaxpr(f)(x)  # doesn't crash

  def testCompositionWithJitTwice(self):
    @jit
    def f(x):
      y = 2 * x

      @jit
      def g(z):
        return pmap(lambda x: x * y)(z)

      return g(x)

    f(np.arange(1.).reshape((1, 1)))  # doesn't crash

  @ignore_jit_of_pmap_warning()
  def testIssue1065(self):
    # from https://github.com/google/jax/issues/1065
    device_count = xla_bridge.device_count()

    def multi_step_pmap(state, count):
      @partial(pmap, axis_name='x')
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
    f = pmap(f, axis_name='i')

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    y = f(x)
    self.assertIsInstance(y, jnp.ndarray)
    self.assertIsInstance(y, pxla.ShardedDeviceArray)

    z = y[0]  # doesn't crash
    self.assertAllClose(z, 2 * x[0], check_dtypes=False)

  def testPostProcessMap(self):
    # TODO(mattjj): this fails with multiple devices (unless we add a jit)
    # because we assume eager ops (like scan here) can't require more than 1
    # replica.
    raise SkipTest("need eager multi-replica support")
    # test came from https://github.com/google/jax/issues/1369
    nrep = xla_bridge.device_count()

    def pmvm(a, b):
      a = a.reshape((nrep, -1, a.shape[1]))
      func = pmap(lambda z: jnp.dot(z, b))
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
    @pmap
    def f(args_list):
      return sum(args_list)

    vals = list(range(500))
    ndevices = xla_bridge.device_count()
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
      res = pmap(fv)(x.reshape((jax.device_count(), -1) + tuple(x.shape[1:])))
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
    jax.pmap(jax.remat(f), axis_name='i')(keys)

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
        new_res = pmap(fv)(new_x)
        # reshape back out
        res = new_res.reshape(x.shape[0], *new_res.shape[2:])
      else:
        res = fv(x)
      return res

    x = random.normal(random.PRNGKey(1), (80, 5))
    y = random.normal(random.PRNGKey(1), (10, 5))

    result1 = vmap(lambda b: matrix_vector(x, b, True))(y)       # vmap + pmap
    result2 = lax.map(lambda b: matrix_vector(x, b, False), y)   # map + map
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
      jax.pmap(test)(a)

  def testPsumOnBooleanDtype(self):
    # https://github.com/google/jax/issues/3123
    n = xla_bridge.device_count()
    if n > 1:
      x = jnp.array([True, False])

      out = pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x)
      self.assertEqual(list(out), [1, 1])

      out = pmap(lambda x: jax.lax.pmean(x, 'i'), 'i')(x)
      self.assertEqual(list(out), [1/2, 1/2])
    else:
      x = jnp.array([True])

      out = pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x)
      self.assertEqual(list(out), [1])

      out = pmap(lambda x: jax.lax.pmean(x, 'i'), 'i')(x)
      self.assertEqual(list(out), [1])

  def testPsumWithNoAxisDoesntLeakFunctions(self):
    x = jnp.ones((1, 1024), dtype=np.float32)
    f = lambda _: x
    w = weakref.ref(f)
    g = pmap(f)
    g(np.ones((1,), dtype=np.float32)).block_until_ready()
    del f, g
    gc.collect()
    # 'f' should not be alive at this point; in particular the pmap cache must
    # not keep it alive.
    self.assertTrue(w() is None)

  def testJitOfPmapWarningMessage(self):
    device_count = xla_bridge.device_count()

    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    def foo(x): return x

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      jit(pmap(foo))(jnp.arange(device_count))

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
    jax.pmap(outer, axis_name='i')(params)  # doesn't crash

    f = jax.pmap(outer, axis_name='i')
    jtu.check_grads(f, (params,), 2, ["fwd", "rev"], 1e-3, 1e-3)

  @ignore_jit_of_pmap_warning()
  def test_issue_1062(self):
    # code from https://github.com/google/jax/issues/1062 @shoyer
    # this tests, among other things, whether ShardedDeviceTuple constants work
    device_count = xla_bridge.device_count()

    @jit
    def multi_step(state, count):
      return lax.fori_loop(0, count, lambda i, s: s, state)

    @jit
    def multi_step_pmap(state, count=2):
      @partial(pmap, axis_name='x')
      def pmapped_multi_step(state):
        return multi_step(state, count)

      return pmapped_multi_step(state)

    u = np.ones((device_count, 100))
    multi_step_pmap(u)  # doesn't crash



class VmapOfPmapTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"{shapes}_{vmap_bdims}_{pmap_bdims}",
       "shapes": shapes, "vmap_bdims": vmap_bdims, "pmap_bdims": pmap_bdims}
      for shape_group in compatible_shapes
      for num_args in range(1, 4)
      for shapes in it.combinations_with_replacement(shape_group, num_args)
      for vmap_bdims in all_bdims(*shapes)
      for pmap_bdims in it.product([0, None], repeat=num_args)
      if not all(bd is None for bd in pmap_bdims)
  ))
  def testVmapOfPmap(self, shapes, vmap_bdims, pmap_bdims):
    vmapped_size = 3
    pmapped_size = xla_bridge.device_count()

    rng = jtu.rand_default(self.rng())

    def fun(*args):
      return sum(args)

    final_shapes = map(partial(add_bdim, vmapped_size), vmap_bdims,
                       map(partial(add_bdim, pmapped_size), pmap_bdims, shapes))

    args = [rng(shape, jnp.float32) for shape in final_shapes]
    args_slice = args_slicer(args, vmap_bdims)
    ans = vmap(pmap(fun, in_axes=pmap_bdims), vmap_bdims)(*args)
    expected = np.stack([fun(*args_slice(i)) for i in range(vmapped_size)])
    self.assertAllClose(ans, expected)

  @skipIf(not jax.config.omnistaging_enabled,
          "vmap collectives only supported when omnistaging is enabled")
  def testCollectivesWithVmap(self):
    def f(map1, map2):
      @partial(map1, axis_name='i')
      @partial(map2, axis_name='j')
      def f(x, y):
        return x + jax.lax.psum(x.dot(y), ('i', 'j'))
      return f

    if xla_bridge.device_count() < 4:
      raise SkipTest("test requires at least four devices")
    x = jnp.ones((2, 2, 64, 64))
    y = f(jax.pmap, jax.pmap)(x, x)
    self.assertAllClose(f(jax.vmap, jax.vmap)(x, x), y)
    self.assertAllClose(f(jax.pmap, jax.vmap)(x, x), y)
    self.assertAllClose(f(jax.vmap, jax.pmap)(x, x), y)


class MaskOfPmapTest(jtu.JaxTestCase):

  def testMaskOfPmap(self):
    device_count = xla_bridge.device_count()
    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    rng = jtu.rand_default(self.rng())

    @pmap
    def fun(x):
      return jnp.sum(x)

    in_shapes = ['(b, n)']
    out_shape = 'b'
    padded_in_shapes = [(device_count, 3)]
    logical_env = dict(b=device_count - 1, n=2)
    with ignore_jit_of_pmap_warning():
      jtu.check_masking(fun, in_shapes, out_shape, logical_env,
                        padded_in_shapes, [np.float32], rng,
                        skip_shapecheck=True)

  def testMaskOfCollectives(self):
    raise SkipTest("Not working yet - need avals with named axes for "
                   "collective masking rules.")
    device_count = xla_bridge.device_count()
    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    rng = jtu.rand_default(self.rng())

    @partial(pmap, axis_name='i')
    def fun(x):
      t1 = lax.psum(x, 'i')
      t2 = lax.pmax(x, 'i')
      t3 = lax.pmin(x, 'i')
      return jnp.sum(t1 + t2 + t3)

    in_shapes = ['(b, n)']
    out_shape = 'b'
    padded_in_shapes = [(device_count, 3)]
    logical_env = dict(b=device_count - 1, n=2)
    with ignore_jit_of_pmap_warning():
      jtu.check_masking(fun, in_shapes, out_shape, logical_env,
                        padded_in_shapes, [np.float32], rng,
                        skip_shapecheck=True)

  def testPmapOfMaskCollectives(self):
    device_count = xla_bridge.device_count()
    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    rng = jtu.rand_default(self.rng())

    in_shapes = ['n']
    out_shape = ''

    def fun(x):
      t1 = lax.psum(x, 'i')
      t2 = lax.pmax(x, 'i')
      t3 = lax.pmin(x, 'i')
      return jnp.sum(t1 + t2 + t3)

    @partial(pmap, axis_name='i')
    def masked_fun(x_padded):
      return mask(fun, in_shapes=in_shapes, out_shape=out_shape)(
          [x_padded], logical_env)

    padded_in_shapes = [(device_count, 3)]
    padded_args = [rng(shape, float) for shape in padded_in_shapes]
    logical_env = dict(n=2)

    padded_out = masked_fun(*padded_args)
    expected = pmap(fun, axis_name='i')(padded_args[0][:, :2])
    self.assertAllClose(padded_out, expected)

  def testMaskLastBatch(self):
    device_count = xla_bridge.device_count()
    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    in_shapes = ['n']
    out_shape = ''

    @partial(pmap, axis_name='batch')
    @partial(mask, in_shapes=in_shapes, out_shape=out_shape)
    def fun(x):
      return lax.psum(jnp.sum(x), 'batch')  # Sum out ragged dim first

    args = [jnp.array([[1, 2], [5, 10]])]
    logical_env = dict(n=jnp.array([2, 1])) # Batch size 3

    padded_out = fun(args, logical_env)
    self.assertAllClose(padded_out, jnp.array([8, 8]))

    logical_env = dict(n=jnp.array([1, 0])) # Batch size 1
    padded_out = fun(args, logical_env)
    self.assertAllClose(padded_out, jnp.array([1, 1]))

  def testRaggedPmapOfMaskCollectivesFailure(self):
    device_count = xla_bridge.device_count()
    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    rng = jtu.rand_default(self.rng())

    in_shapes = ['n']
    out_shape = ''

    @partial(pmap, axis_name='i')
    @partial(mask, in_shapes=in_shapes, out_shape=out_shape)
    def fun(x):
      t1 = lax.psum(x, 'i')
      t2 = lax.pmax(x, 'i')
      t3 = lax.pmin(x, 'i')
      return jnp.sum(t1 + t2 + t3)

    padded_in_shapes = [(device_count, 3)]
    padded_args = [rng(shape, float) for shape in padded_in_shapes]
    logical_env = dict(n=jnp.minimum(jnp.arange(device_count, dtype=int), 3))

    with self.assertRaisesRegex(
        ValueError, "ragged batch.*not supported"):
      fun(padded_args, logical_env)


  def testMaskPostProcessMap(self):
    device_count = xla_bridge.device_count()
    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    rng = jtu.rand_default(self.rng())

    def fun(x):
      @pmap
      def fun2(y):
        return jnp.sum(x + y) + x + y
      return fun2(jnp.array([2., 3.]))

    in_shapes = ['n']
    out_shape = '(2, n)'
    padded_in_shapes = [(3,)]
    logical_env = dict(n=2)
    with ignore_jit_of_pmap_warning():
      jtu.check_masking(fun, in_shapes, out_shape, logical_env,
                        padded_in_shapes, [np.float32], rng)

  def testGradMaskPmap(self):
    raise SkipTest("Mask of grad not working yet.")
    device_count = xla_bridge.device_count()
    if device_count == 1:
      raise SkipTest("test requires at least two devices")

    rng = jtu.rand_default(self.rng())

    @pmap
    def fun(x):
      return jnp.sum(x ** 2)

    fun_grad = grad(lambda x: jnp.sum(fun(x)))

    in_shapes = ['n']
    out_shape = 'n'
    padded_in_shapes = [(2,)]
    logical_env = dict(n=1)
    with ignore_jit_of_pmap_warning():
      jtu.check_masking(fun_grad, in_shapes, out_shape, logical_env,
                        padded_in_shapes, [np.float32], rng)

class PmapWithDevicesTest(jtu.JaxTestCase):

  def testAllDevices(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i',
             devices=xla_bridge.devices())
    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.sum(x, 0)
    ans = f(x)
    self.assertAllClose(ans, expected)

  def testOneDevice(self):
    if xla_bridge.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    d0 = xla_bridge.devices()[0]
    d1 = xla_bridge.devices()[1]
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
    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    with self.assertRaisesRegex(
        ValueError, "'devices' argument to pmap must be non-empty, or None."):
      f(x)

  def testBadAxisSizeError(self):
    if xla_bridge.device_count() == 1:
      raise SkipTest("this test requires multiple devices")

    f = pmap(lambda x: lax.psum(x, 'i'), axis_name='i',
             devices=xla_bridge.devices())
    with self.assertRaisesRegex(
        ValueError, r"Leading axis size of input to pmapped function must "
        r"equal the number of local devices passed to pmap. Got axis_size=1, "
        r"num_local_devices=\d."):
      f(jnp.ones(1))

    with self.assertRaisesRegex(
        ValueError, r"Leading axis size of input to pmapped function must "
        r"equal the number of local devices passed to pmap. Got axis_size=\d, "
        r"num_local_devices=\d."):
      f(jnp.ones(xla_bridge.device_count() + 1))

  def testNestedPmaps(self):
    if xla_bridge.device_count() % 2 != 0:
      raise SkipTest

    # Devices specified in outer pmap are OK
    @partial(pmap, axis_name='i', devices=xla_bridge.devices())
    def foo(x):
      @partial(pmap, axis_name='j')
      def bar(y):
        return lax.psum(y, 'j')
      return bar(x)

    x = jnp.ones((xla_bridge.device_count() // 2, 2))
    ans = foo(x)
    expected = x * 2
    self.assertAllClose(ans, expected)

  def testNestedPmapsError(self):
    # Devices specified in inner pmap not OK
    @partial(pmap, axis_name='i')
    def foo(x):
      @partial(pmap, axis_name='j', devices=xla_bridge.devices())
      def bar(y):
        return lax.psum(y, 'j')
      return bar(x)

    with self.assertRaisesRegex(
        ValueError,
        "Nested pmap with explicit devices argument."):
      foo(jnp.ones((xla_bridge.device_count(), 1)))

  def testJitInPmap(self):
    @partial(pmap, axis_name='i', devices=xla_bridge.devices())
    def foo(x):
      @jit
      def bar(y):
        return y + 1
      return lax.psum(bar(x), 'i')

    ndevices = xla_bridge.device_count()
    ans = foo(jnp.ones((ndevices, 1)))
    expected = np.ones((ndevices, 1), dtype=jnp.float_) * ndevices * 2
    self.assertAllClose(ans, expected)

  @ignore_jit_of_pmap_warning()
  def testPmapInJit(self):
    @jit
    def foo(x):
      @partial(pmap, axis_name='i', devices=xla_bridge.devices())
      def bar(y):
        return lax.psum(y, 'i')
      return bar(x)

    ndevices = xla_bridge.device_count()
    ans = foo(jnp.ones((ndevices, 1)))
    expected = np.ones((ndevices, 1), dtype=jnp.float_) * ndevices
    self.assertAllClose(ans, expected)

  def testGradBasic(self):
    @partial(pmap, axis_name='i', devices=xla_bridge.devices())
    def f(x):
      return jnp.sin(x)

    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(jnp.sin(x)))(x)
    expected = grad(lambda x: jnp.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPmapStaticArgnums(self):
    @partial(pmap, axis_name='i', static_broadcasted_argnums=1)
    def f(x, y):
      return jnp.sin(x + y)
    shape = (xla_bridge.device_count(), 4)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    y = np.arange(4, dtype=np.float32)

    ans = f(x, y)
    expected = np.sin(x + y[None])
    self.assertAllClose(ans, expected, check_dtypes=False)


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


class SpecToIndicesTest(jtu.JaxTestCase):

  def testShardsPerAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(shards_per_axis=(2, 2),
                             is_axis_materialized=(True, True),
                             replication_factors=[])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((slice(0,2), slice(0,4)),
                      (slice(0,2), slice(4,8)),
                      (slice(2,4), slice(0,4)),
                      (slice(2,4), slice(4,8))))

  def testUnshardedAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(shards_per_axis=(2, 1),
                             is_axis_materialized=(True, True),
                             replication_factors=[])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     (slice(0,2), (slice(2,4))))

  def testNoSharding(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(shards_per_axis=(1, 1),
                             is_axis_materialized=(True, True),
                             replication_factors=[])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     (slice(None),))

  def testUnmaterializedAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(shards_per_axis=(4, 1),
                             is_axis_materialized=(False, True),
                             replication_factors=[])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     (0, 1, 2, 3))

    shape = (2, 2)
    spec = pxla.ShardingSpec(shards_per_axis=(1, 2),
                             is_axis_materialized=(True, False),
                             replication_factors=[])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((slice(None), 0),
                      (slice(None), 1)))

  def testReplicationAfterUnsharded(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(shards_per_axis=(2, 1),
                             is_axis_materialized=(False, True),
                             replication_factors=[(3, 2)])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     (0, 0, 0, 1, 1, 1))

  def testReplicationPosition2(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(shards_per_axis=(2, 2),
                             is_axis_materialized=(False, True),
                             replication_factors=[(3, 2)])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((0, slice(0, 4)), (0, slice(0, 4)), (0, slice(0, 4)),
                      (0, slice(4, 8)), (0, slice(4, 8)), (0, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(0, 4)), (1, slice(0, 4)),
                      (1, slice(4, 8)), (1, slice(4, 8)), (1, slice(4, 8))))

  def testReplicationPosition1(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(shards_per_axis=(2, 2),
                             is_axis_materialized=(False, True),
                             replication_factors=[(3, 1)])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((0, slice(0, 4)), (0, slice(4, 8)),
                      (0, slice(0, 4)), (0, slice(4, 8)),
                      (0, slice(0, 4)), (0, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(4, 8))))

  def testReplicationPosition0(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(shards_per_axis=(2, 1),
                             is_axis_materialized=(False, True),
                             replication_factors=[(3, 0)])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     (0, 1, 0, 1, 0, 1))

  def testMultipleReplications(self):
    shape = (2, 7, 4)
    spec = pxla.ShardingSpec(shards_per_axis=(2, 1, 2),
                             is_axis_materialized=(False, True, True),
                             replication_factors=[(3, 0), (2, 0), (2, 2)])
    self.assertEqual(
        pxla.spec_to_indices(shape, spec),
        ((0, slice(None), slice(0, 2)), (0, slice(None), slice(2, 4)),
         (0, slice(None), slice(0, 2)), (0, slice(None), slice(2, 4)),
         (1, slice(None), slice(0, 2)), (1, slice(None), slice(2, 4)),
         (1, slice(None), slice(0, 2)), (1, slice(None), slice(2, 4))) * 3 * 2)

  def testReplicatedScalar(self):
    shape = ()
    spec = pxla.ShardingSpec(shards_per_axis=(),
                             is_axis_materialized=(),
                             replication_factors=[(3, 0)])
    self.assertEqual(pxla.spec_to_indices(shape, spec),
                     ((), (), ()))


def _spec_str(spec):
  return (f"({spec.shards_per_axis},"
          f"{spec.is_axis_materialized},"
          f"{spec.replication_factors})")


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
          [(4, 8), pxla.ShardingSpec(shards_per_axis=(4, 1),
                                     is_axis_materialized=(False, True),
                                     replication_factors=[])],
          # pmap(in_axes=1)
          [(2, 2), pxla.ShardingSpec(shards_per_axis=(1, 2),
                                     is_axis_materialized=(True, False),
                                     replication_factors=[])],
          # unsharded
          [(4, 8), pxla.ShardingSpec(shards_per_axis=(1, 1),
                                     is_axis_materialized=(True, True),
                                     replication_factors=[])],
          # partitioned, 1 axis
          [(4, 8), pxla.ShardingSpec(shards_per_axis=(2, 1),
                                     is_axis_materialized=(True, True),
                                     replication_factors=[])],
          # partitioned, 2 axes
          [(4, 8), pxla.ShardingSpec(shards_per_axis=(2, 2),
                                     is_axis_materialized=(True, True),
                                     replication_factors=[])],
          # partitioned + sharding
          [(2, 8), pxla.ShardingSpec(shards_per_axis=(2, 2),
                                     is_axis_materialized=(False, True),
                                     replication_factors=[])],
          # replication + sharding
          [(2, 8), pxla.ShardingSpec(shards_per_axis=(2, 1),
                                     is_axis_materialized=(False, True),
                                     replication_factors=[(3, 2)])],
          # replication, no sharding
          [(2, 8), pxla.ShardingSpec(shards_per_axis=(1, 1),
                                     is_axis_materialized=(True, True),
                                     replication_factors=[(3, 2)])],
          # multiple replicated axes
          [(1, 8), pxla.ShardingSpec(shards_per_axis=(1, 2),
                                     is_axis_materialized=(False, True),
                                     replication_factors=[(2, 0), (2, 1)])],
          # replicated scalar
          [(), pxla.ShardingSpec(shards_per_axis=(),
                                 is_axis_materialized=(),
                                 replication_factors=[(2, 0), (3, 0)])]
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
    self.assertEqual(len(bufs), nshards)
    for buf, idx in zip(bufs, indices):
      self.assertEqual(len(buf), 1)
      self.assertAllClose(buf[0].to_py(), x[idx], check_dtypes=False)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
