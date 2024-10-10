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

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
from functools import partial
import itertools as it
import gc
import math
from random import shuffle
import re
from typing import Union, cast
import unittest
from unittest import SkipTest
import weakref

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import (pmap, jit, vmap, jvp, grad, make_jaxpr,
                 linearize, device_put)
from jax import lax
import jax.scipy.linalg
from jax import random
from jax.ad_checkpoint import checkpoint as new_checkpoint
import jax.numpy as jnp
from jax._src import api as src_api
from jax._src import array
from jax._src import core
from jax._src import config
from jax._src import sharding_impls
from jax._src import sharding_specs
from jax._src import test_util as jtu
from jax._src.internal_test_util import lax_test_util
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lax import parallel
from jax._src.lib import xla_extension
from jax._src.util import safe_map, safe_zip

config.parse_flags_with_absl()

# Run all tests with 8 CPU devices.
_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(8))

def tearDownModule():
  _exit_stack.close()

compatible_shapes = [[(3,)], [(3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]

def all_bdims(*shapes, pmap):
  bdims = (it.chain([cast(Union[int, None], None)], range(len(shape) + 1))
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

ignore_jit_of_pmap_warning = partial(
  jtu.ignore_warning, message=".*jit-of-pmap.*")

def create_input_array_for_pmap(input_shape, in_axes=0, input_data=None,
                                devices=None, sharded_dim_size=None):
  if input_data is None:
    input_data = np.arange(math.prod(input_shape)).reshape(input_shape)

  sharding_spec = sharding_specs.create_pmap_sharding_spec(
      input_shape, in_axes, sharded_dim_size)

  if devices is None:
    devices = jax.devices()

  pmap_sharding = jax.sharding.PmapSharding(np.array(devices), sharding_spec)

  return array.make_array_from_callback(
      input_shape, pmap_sharding, lambda idx: input_data[idx]), input_data


@jtu.pytest_mark_if_available('multiaccelerator')
@jtu.with_config(jax_legacy_prng_key="allow")
class PythonPmapTest(jtu.JaxTestCase):

  @property
  def pmap(self):
    return src_api.pmap

  def testDeviceBufferToArray(self):
    sda = self.pmap(lambda x: x)(jnp.ones((jax.device_count(), 2)))

    # Changed in https://github.com/jax-ml/jax/pull/10584 not to access
    # sda.device_buffers, which isn't supported, and instead ensure fast slices
    # of the arrays returned by pmap are set up correctly.
    # buf = sda.device_buffers[-1]
    buf = sda[-1]

    view = jnp.array(buf, copy=False)
    self.assertArraysEqual(sda[-1], view)
    self.assertSetEqual(buf.devices(), view.devices())
    self.assertEqual(buf.unsafe_buffer_pointer(), view.unsafe_buffer_pointer())

    copy = jnp.array(buf, copy=True)
    self.assertArraysEqual(sda[-1], copy)
    self.assertSetEqual(buf.devices(), copy.devices())
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
      if device_count % math.prod(device_mesh_shape):
        msg = "device mesh size {} does not divide available device count {}"
        raise SkipTest(msg.format(math.prod(device_mesh_shape), device_count))
      else:
        return device_mesh_shape

  def testBasic(self):
    f = self.pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.sum(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDefaultDeviceOrdering(self):
    # Users rely on the fact that the default order of jax.devices() matches
    # the default order of pmap for single-host jobs.
    device_order = jax.devices()
    pmap_sharding = pmap(lambda x: x)(np.arange(jax.device_count())).sharding
    if config.pmap_shmap_merge.value:
      self.assertListEqual(device_order, pmap_sharding._device_assignment)
    else:
      self.assertListEqual(device_order, pmap_sharding.devices.tolist())

  def testLowerCompile(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = f(x)
    lowered = f.lower(x)
    compiled = lowered.compile()
    ans = compiled(x)

    self.assertAllClose(ans, expected)

    # It's a pair of: (positional args, as a tuple of their structures, kwargs).
    for obj in [lowered, compiled]:
      self.assertFalse(obj._no_kwargs)
      self.assertEqual(obj.in_tree, jax.tree.flatten(((0,), {}))[1])
      self.assertEqual(obj.in_avals, ((core.ShapedArray(x.shape, x.dtype),), {}))

  def testLowerCompileInTreeMismatch(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f_exe = f.lower(x).compile()
    self.assertRaisesRegex(
        TypeError,
        'Function compiled with input pytree does not match the input pytree it'
        ' was called with',
        lambda: f_exe([x]))

  def testLowerCompileTrivial(self):
    f = self.pmap(lambda x: x, axis_name='i')
    x = np.arange(jax.device_count(), dtype=np.float32)
    expected = f(x)
    f_exe = f.lower(x).compile()
    ans = f_exe(x)
    self.assertAllClose(ans, expected)

  def testLowerCompileTrivialInTreeMismatch(self):
    f = self.pmap(lambda x: x, axis_name='i')
    x = np.arange(jax.device_count(), dtype=np.float32)
    f_exe = f.lower(x).compile()
    self.assertRaisesRegex(
        TypeError,
        'Function compiled with input pytree does not match the input pytree it'
        ' was called with',
        lambda: f_exe([x]))

  def testLowerCompileArgTypeMismatch(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=int).reshape(shape)
    x_f32 = x.astype(jnp.float32)
    x_i32 = x.astype(jnp.int32)
    f_exe = f.lower(x_f32).compile()
    self.assertRaisesRegex(
        TypeError,
        r"Argument types differ .*"
        r"The mismatches are:\n"
        r"Argument 'x' compiled with.*float32.*and called with.*int32.*",
        lambda: f_exe(x_i32))

  def testLowerCompileMultiArg(self):
    f = self.pmap(lambda x, y: x - lax.pmean(y, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = y = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = f(x, y)
    f_exe = f.lower(x, y).compile()
    ans = f_exe(x, y)
    self.assertAllClose(ans, expected)

  def testLowerCompileTrivialMultiArg(self):
    f = self.pmap(lambda x, y: (x, y), axis_name='i')
    x = y = np.arange(jax.device_count(), dtype=np.float32)
    expected = f(x, y)
    f_exe = f.lower(x, y).compile()
    ans = f_exe(x, y)
    self.assertAllClose(ans, expected)

  def testLowerAsText(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f = f.lower(x)
    self.assertIsInstance(f.as_text(), str)
    self.assertIsInstance(f.as_text(dialect='hlo'), str)
    self.assertIsInstance(f.as_text(dialect='stablehlo'), str)

  def testLowerCompilerIR(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f = f.lower(x)
    self.assertIsNotNone(f.compiler_ir())
    self.assertIsNotNone(f.compiler_ir(dialect='hlo'))
    self.assertIsNotNone(f.compiler_ir(dialect='stablehlo'))

  def testLowerCompileCompilerIR(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f = f.lower(x).compile()
    self.assertIsNotNone(f.runtime_executable())

  def testLowerCompileAsText(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f = f.lower(x).compile()
    self.assertIsInstance(f.as_text(), (str, type(None)))

  def testLowerCostAnalysis(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f = f.lower(x)
    f.cost_analysis()  # doesn't raise

  def testLowerCompileCostAnalysis(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f = f.lower(x).compile()
    f.cost_analysis()  # doesn't raise

  def testLowerCompileMemoryAnalysis(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f = f.lower(x).compile()
    f.memory_analysis()  # doesn't raise

  def testLowerCompileExecutable(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f = f.lower(x).compile()
    self.assertIsNotNone(f.runtime_executable())

  def test_jit_lower_compile_with_compiler_options(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    lowered = f.lower(x)

    lowered.compile(            # doesn't crash
        compiler_options={"xla_embed_ir_in_executable": True})

  def test_jit_lower_compile_with_compiler_options_invalid(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    lowered = f.lower(x)

    self.assertRaisesRegex(
        xla_extension.XlaRuntimeError, "No such compile option: 'invalid_key'",
        lambda: lowered.compile(
            compiler_options={"invalid_key": "invalid_value"}))

    self.assertRaisesRegex(
        xla_extension.XlaRuntimeError, "is not a valid bool value.",
        lambda: lowered.compile(
            compiler_options={"xla_embed_ir_in_executable": "invalid_value"}))

  def test_pmap_replicated_copy(self):
    # https://github.com/jax-ml/jax/issues/17690
    inp = jnp.arange(jax.device_count())
    x = jax.pmap(lambda x: x, in_axes=0, out_axes=None)(inp)
    out = jnp.copy(x)
    self.assertIsInstance(out.sharding, jax.sharding.SingleDeviceSharding)
    self.assertArraysEqual(out, inp[0])

  def test_jit_lower_compile_with_compiler_options_multiple(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    lowered = f.lower(x)

    l1 = lowered.compile()
    l2 = lowered.compile(
        compiler_options={"xla_embed_ir_in_executable": True})
    l3 = lowered.compile(
        compiler_options={"xla_embed_ir_in_executable": False})

    # Ideally we could test that these objects are different only in
    # that they respect the different options. Object identity is a
    # heuristic proxy for that.
    self.assertTrue(l1 is not l2)
    self.assertTrue(l1 is not l3)
    self.assertTrue(l2 is not l3)

    # We should still error on invalid options after some valid compiles
    self.assertRaisesRegex(
        xla_extension.XlaRuntimeError, "No such compile option: 'invalid_key'",
        lambda: lowered.compile(
            compiler_options={"invalid_key": "invalid_value"}))

  def testLowerShapedArray(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    x_shape = core.ShapedArray(x.shape, x.dtype)
    self.assertAllClose(f.lower(x_shape).compile()(x), f(x))

  def testLowerHasReplicaAttributes(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    num_devices = jax.device_count()
    shape = (num_devices, 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    lowered = f.lower(x)
    hlo = lowered.as_text("stablehlo")
    self.assertIn(f"mhlo.num_replicas = {num_devices}", hlo)
    self.assertIn("mhlo.num_partitions = 1", hlo)

  def testMean(self):
    f = self.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.broadcast_to(np.mean(x, 0), x.shape)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGather(self):
    f = self.pmap(lambda x: lax.all_gather(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.array([x] * jax.device_count())
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGatherBool(self):
    f = self.pmap(lambda x: lax.all_gather(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    x = (x % 2).astype(np.bool_)
    expected = np.array([x] * jax.device_count())
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGatherNegativeAxis(self):
    f = self.pmap(lambda x: lax.all_gather(x, 'i', axis=-1), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.array([x.T] * jax.device_count())
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGatherTiled(self):
    f = self.pmap(lambda x: lax.all_gather(x, 'i', tiled=True), axis_name='i')

    device_count = jax.device_count()
    shape = (device_count, 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.array([x] * device_count).reshape(device_count, -1)
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGatherTiledNegativeAxis(self):
    f = self.pmap(lambda x: lax.all_gather(x, 'i', tiled=True, axis=-1),
                  axis_name='i')

    device_count = jax.device_count()
    shape = (device_count, 4, 3)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.array([x.transpose(1, 0, 2).reshape(4, -1)] * device_count)
    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters([
    ('Gather', lax.all_gather),
    ('ReduceScatter', lax.psum_scatter)
  ])
  def testVmapOf(self, prim):
    f = self.pmap(partial(prim, axis_name='i'), axis_name='i')

    device_count = jax.device_count()
    shape = (4, device_count, device_count)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    self.assertAllClose(vmap(f)(x), jnp.stack([f(xs) for xs in x], axis=0))

  def testReduceScatter(self):
    f = self.pmap(lambda x: lax.psum_scatter(x, 'i'), axis_name='i')

    device_count = jax.device_count()
    shape = (device_count, device_count)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.sum(x, axis=0)
    ans = f(x)
    for i, actual in enumerate(ans):
      self.assertAllClose(actual, expected[i])

  def testReduceScatterTiled(self):
    f = self.pmap(lambda x: lax.psum_scatter(x, 'i', tiled=True), axis_name='i')

    device_count = jax.device_count()
    shape = (device_count, 4 * device_count)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    ans = f(x)

    group_1_result = np.sum(x[0::2,:], axis=0)
    group_2_result = np.sum(x[1::2,:], axis=0)
    # the result is scattered over (replicas // 2) devices
    scatter_len = len(group_1_result) * 2 // replicas

    for i, actual in enumerate(ans):
      expected = group_1_result if i % 2 == 0 else group_2_result
      self.assertAllClose(
          actual, expected[i // 2 * scatter_len:(i // 2 + 1) * scatter_len])

  def testTrees(self):
    ptranspose = lambda x, axis_name: lax.all_to_all(x, axis_name, 0, 0)
    def protate(x, axis_name):
      n = lax.psum(1, axis_name)
      return lax.ppermute(x, axis_name, [(i, (i + 1) % n) for i in range(n)])

    tree_f = lambda f: partial(jax.tree.map, f)
    jax_f = lambda p: self.pmap(lambda x: p(x, 'i'), 'i')
    np_f = lambda p: tree_f(lambda x: np.broadcast_to(p(x, 0), x.shape))
    np_transpose = tree_f(np.transpose)
    np_rotate = tree_f(lambda x: np.concatenate([x[-1:], x[:-1]]))

    n = jax.device_count()
    x = {'a': np.arange(1 * n * n, 2 * n * n).reshape([n, n]),
         'b': np.arange(2 * n * n, 3 * n * n).reshape([n, n]),
         'c': np.arange(4 * n * n, 5 * n * n).reshape([n, n])}

    assert_allclose = partial(jax.tree.map,
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
    tree_f = lambda f: partial(jax.tree.map, f)
    jax_f = lambda p: self.pmap(lambda x: p(x, 'i'), 'i')
    np_f = lambda p: tree_f(lambda x: np.broadcast_to(p(x, 0), x.shape))
    assert_allclose = partial(jax.tree.map,
                              partial(self.assertAllClose, check_dtypes=False))
    assert_allclose(jax_f(lax.pmax)(x), np_f(np.max)(x))
    assert_allclose(jax_f(lax.pmin)(x), np_f(np.min)(x))
    assert_allclose(jax_f(lax.psum)(x), np_f(np.sum)(x))
    assert_allclose(jax_f(lax.pmean)(x), np_f(np.mean)(x))

  def testComplexPsum(self):
    f = self.pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4 * 2)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape).view(np.complex64)
    expected = x - np.sum(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.sample_product(
    split_axis=list(range(2)),
    concat_axis=list(range(2)),
    dtype=lax_test_util.all_dtypes,
  )
  def testAllToAll(self, split_axis, concat_axis, dtype):
    pmap_in_axis = 0
    shape = (jax.device_count(),) * 3
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)

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
  def testAllToAllSplitAxis(self, split_axis, concat_axis):
    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")
    pmap_in_axis = 0
    shape = (4, 4, 4)
    x = np.arange(math.prod(shape)).reshape(shape)

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

  def testNestedPmapAxisSwap(self):
    # Regression test for https://github.com/jax-ml/jax/issues/5757
    if jax.device_count() < 8:
      raise SkipTest("test requires at least 8 devices")
    f = jax.pmap(jax.pmap(lambda x: x, in_axes=1, out_axes=0), in_axes=0,
                 out_axes=0)
    A = jnp.ones((2, 4, 3))
    self.assertAllClose(A.transpose((0, 2, 1)), f(A))

  def testNestedBasic(self):
    f = lambda x: lax.psum(lax.psum(x, 'i'), 'j')
    f = self.pmap(self.pmap(f, 'i'), 'j')

    def sum_and_broadcast(x, axis):
      return np.repeat(np.sum(x, axis, keepdims=True), x.shape[axis], axis)

    shape = (jax.device_count(), 1, 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)
    expected = sum_and_broadcast(sum_and_broadcast(x, 0), 1)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMismatchedAxisSizes(self):
    n = jax.device_count()
    f = self.pmap(lambda x, y: x + y)
    self.assertRaisesRegex(
        ValueError,
        "pmap got inconsistent sizes for array axes to be mapped",
        lambda: f(self.rng().randn(n), self.rng().randn(n - 1)))

  def testInAxesPyTreePrefixMismatchError(self):
    x = jnp.array([3.14])
    f = self.pmap(lambda x, y: x, in_axes=((0, 0, 0), 0))
    with self.assertRaisesRegex(ValueError, re.escape("pmap in_axes[0][0]")):
      f((x, x), x)

  def testInAxesPyTreePrefixMismatchErrorKwargs(self):
    x = jnp.array([3.14])
    f = self.pmap(lambda x, y: x, in_axes=((0, 0), 0))
    with self.assertRaisesRegex(
        ValueError, re.escape("each argument passed by keyword is mapped")):
      f(x=(x, x), y=x)

  def testOutAxesPyTreePrefixMismatchError(self):
    x = jnp.array([3.14])
    f = jax.pmap(lambda x, y: ((x, x), x), out_axes=((0, 0, 0), 0))
    with self.assertRaisesRegex(ValueError, re.escape("pmap out_axes[0]")):
      f(x, x)

  @parameterized.named_parameters(
      {"testcase_name": f"_mesh={device_mesh_shape}".replace(" ", ""),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testNestedShardingAndStacking(self, device_mesh_shape):
    mesh_shape = self._getMeshShape(device_mesh_shape)

    f = lambda x: x
    f = self.pmap(self.pmap(f, 'i'), 'j')

    shape = mesh_shape + (4,)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

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
    y = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    f_expected = np.broadcast_to(x, mesh_shape)
    f_ans = f(x, y)
    self.assertAllClose(f_ans, f_expected)
    self.assertIsInstance(f_ans, array.ArrayImpl)
    sharding_spec = f_ans.sharding.sharding_spec
    # the output is actually replicated (has the same values in each device buffer)
    # but out_axes is implicitly 0, so we shouldn't have replication in the
    # sharding spec.
    self.assertEmpty([a for a in sharding_spec.mesh_mapping
                      if isinstance(a, pxla.Replicated)])

    g_expected = np.broadcast_to(x - np.sum(y, 0, keepdims=True), shape)
    g_ans = g(x, y)
    self.assertAllClose(g_ans, g_expected)
    self.assertIsInstance(g_ans, array.ArrayImpl)
    sharding_spec = g_ans.sharding.sharding_spec
    self.assertEmpty([a for a in sharding_spec.mesh_mapping
                      if isinstance(a, pxla.Replicated)])

  @parameterized.named_parameters(
      {"testcase_name": f"_mesh={device_mesh_shape}".replace(" ", ""),
       "device_mesh_shape": device_mesh_shape}
      for device_mesh_shape in [(1, 1), (2, -1), (-1, 2)])
  def testPartiallyMappedNested(self, device_mesh_shape):
    mesh_shape = self._getMeshShape(device_mesh_shape)

    f = self.pmap(lambda x, y: x - lax.psum(y, 'i'), axis_name='i', in_axes=(None, 0))
    f = self.pmap(f, axis_name='j', in_axes=(None, 0))

    x = 3.
    y = np.arange(math.prod(mesh_shape), dtype=np.float32).reshape(mesh_shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.cos(x)

    ans = splitjvp(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    make_jaxpr(splitjvp)(x)  # doesn't crash

  def testGradBasic(self):
    @partial(self.pmap, axis_name='i')
    def f(x):
      return jnp.sin(x)

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(jnp.sin(x)))(x)
    expected = grad(lambda x: jnp.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGradOfPsum(self):
    @partial(self.pmap, axis_name='i')
    def f(x):
      return lax.psum(x, axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    y = 4 + x
    ans = grad(lambda x, y: jnp.sum(g(x, y)))(x, y)
    expected = grad(lambda x, y: jnp.sum(g(x, y)))(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": f"_mesh={device_mesh_shape}".replace(" ", ""),
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(test_fun(x)))(x)
    expected = grad(lambda x: jnp.sum(baseline_fun(x)))(x)
    self.assertAllClose(ans, expected, atol=1e-3, rtol=1e-3)

  def testArrays(self):
    f = lambda x: 2 * x
    f = self.pmap(f, axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    # test that we can pass in and out Arrays
    y = f(x)
    self.assertIsInstance(y, jax.Array)
    self.assertIsInstance(y, array.ArrayImpl)
    self.assertNotIsInstance(y, np.ndarray)
    self.assertAllClose(y, 2 * x, check_dtypes=False)
    z = f(y)
    self.assertIsInstance(z, array.ArrayImpl)
    self.assertNotIsInstance(z, np.ndarray)
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can pass in a regular Array
    y = f(device_put(x))
    self.assertIsInstance(y, array.ArrayImpl)
    self.assertAllClose(y, 2 * x, check_dtypes=False)

    # test that we can pass an Array to a regular jit computation
    z = y + y
    self.assertAllClose(z, 2 * 2 * x, check_dtypes=False)

    # test that we can handle device movement on dispatch
    bufs = y._arrays[::-1]
    sharding = jax.sharding.PmapSharding(
        [list(b.devices())[0] for b in bufs], y.sharding.sharding_spec)
    y = jax.make_array_from_single_device_arrays(y.shape, sharding, bufs)
    z = f(y)
    self.assertAllClose(z, 2 * 2 * x[::-1], check_dtypes=False)

    # test that the repr doesn't crash
    repr(z)

    # test that we can lexically capture a sda as a constant.
    g = jit(lambda z: z + y)
    self.assertAllClose(g(7), y + 7)

  # Tests edge cases in lax._reshape_sharded_device_array
  @parameterized.named_parameters(
      {"testcase_name": f"_in={in_shape}_out={out_shape}"
       .replace(" ", ""),
       "in_shape": in_shape, "out_shape": out_shape}
      for in_shape, out_shape in [
          [(1,1), (1,)], [(1,), (1,1)], [(1,), ()], [(4,7), (2,2,7)]
      ])
  def testArrayReshape(self, in_shape, out_shape):
    if jax.device_count() < max(in_shape[:1] + out_shape[:1]):
      raise SkipTest("not enough devices")

    x = np.arange(math.prod(in_shape)).reshape(in_shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    ans = f(x)

    expected = np.zeros((replicas, replicas // 2, x.shape[1]))
    expected[::2] = x[::2]
    expected[1::2] = x[1::2]

    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(it.chain.from_iterable([
      (name, prim, False, False),
      (name + 'Tiled', prim, True, False),
      (name + 'IndexGroups', prim, False, True),
    ] for name, prim in
    (('Gather', lax.all_gather), ('ReduceScatter', lax.psum_scatter))
  ))
  def testGradOf(self, prim, tiled, use_axis_index_groups):
    axis_index_groups = None
    devices = jax.devices()

    if use_axis_index_groups:
      if len(devices) < 2:
        raise SkipTest("Need at least two devices")
      axis_index_groups = [(l.id, r.id)
                           for l, r in np.asarray(devices).reshape(-1, 2)]

    @partial(self.pmap, axis_name='i')
    def f(x):
      return prim(x, axis_name='i', tiled=tiled,
                  axis_index_groups=axis_index_groups)

    shape = (len(devices), 2 if axis_index_groups else jax.device_count())
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    axis_env = sharding_impls.AxisEnv(8, ('i', 'j'), (4, 2))
    groups = pxla.axis_groups(axis_env, 'i')
    self.assertEqual(groups, ((0, 2, 4, 6), (1, 3, 5, 7)))

    groups = pxla.axis_groups(axis_env, 'j')
    self.assertEqual(groups, ((0, 1), (2, 3), (4, 5), (6, 7)))

    groups = pxla.axis_groups(axis_env, ('i', 'j'))
    self.assertEqual(groups, ((0, 1, 2, 3, 4, 5, 6, 7,),))

    groups = pxla.axis_groups(axis_env, ('j', 'i'))
    self.assertEqual(len(groups), 1)
    self.assertEqual((tuple(sorted(groups[0])),),
                     ((0, 1, 2, 3, 4, 5, 6, 7,),))  # order doesn't matter

  @jtu.run_on_devices("gpu")
  def testCollectiveBroadcast(self):
    device_count = jax.device_count()
    f = lambda x: lax.pbroadcast(x, source=0, axis_name='i')
    f = self.pmap(f, 'i')
    x = jnp.arange(4 * device_count).reshape((device_count, 4))
    ans = f(x)
    expected = np.take(x, [0] * device_count, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.run_on_devices("gpu")
  def testCollectiveBroadcastVmap(self):
    device_count = jax.device_count()
    f = lambda x: lax.pbroadcast(x, source=0, axis_name='i')
    x = np.arange(device_count * 16, dtype=np.float32)
    x = x.reshape((device_count, 4, 4))
    ans = self.pmap(vmap(f), 'i')(x)
    expected = jnp.broadcast_to(x[0:1], x.shape)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jtu.run_on_devices("gpu")
  def testCollectiveBroadcastGrad(self):
    device_count = jax.device_count()
    f = lambda x: lax.pbroadcast(x, source=0, axis_name='i')
    x = np.arange(device_count, dtype=np.float32)
    ans = self.pmap(grad(f), 'i')(x)
    expected = np.zeros_like(x)
    expected[0] = device_count
    self.assertAllClose(ans, expected, check_dtypes=False)

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
    # https://github.com/jax-ml/jax/issues/1703
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
    for _ in range(9):
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
    ))

    print(ans)
    self.assertEqual(ans, expected)

  def testReduceMax(self):
    f = self.pmap(lambda x: x - lax.pmax(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.max(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReduceMin(self):
    f = self.pmap(lambda x: x - lax.pmin(x, 'i'), axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.min(x, 0)

    ans = f(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDeviceCountError(self):
    device_count = jax.device_count()

    f = self.pmap(lambda x: 2 * x)
    x = jnp.arange(device_count + 1)
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

    f = self.pmap(lambda x: 2 * x)
    x = np.ones((device_count + 1, 10))
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

    f = self.pmap(lambda x: self.pmap(lambda x: 2 * x)(x))
    x = np.ones((device_count, 2, 10))
    self.assertRaisesRegex(ValueError, ".*requires.*replicas", lambda: f(x))

  def testPmapConstant(self):
    device_count = jax.device_count()
    f = self.pmap(lambda x: 3)
    x = jnp.arange(device_count)
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): fix this
    expected = np.repeat(3, device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

    if not config.disable_jit.value:
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
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = np.repeat(3, len(devices))
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    ans_devices = ans.sharding._device_assignment
    # TODO(mattjj,sharadmv): fix physical layout with eager pmap, remove 'if'
    if not config.disable_jit.value:
      self.assertEqual(ans_devices, tuple(devices))

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
    x = jnp.arange(math.prod(shape)).reshape(shape)
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = 3 * np.ones(shape[:2])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    expected_sharded = self.pmap(self.pmap(lambda x: x))(expected)
    self.assertTrue(ans.sharding._device_assignment,
                    expected_sharded.sharding._device_assignment)

    f = self.pmap(self.pmap(lambda x: (x, 3)))
    x_sharded, ans = f(x)
    self.assertEqual(ans.sharding._device_assignment,
                      x_sharded.sharding._device_assignment)

  @unittest.skip("Nested pmaps with devices not yet implemented")
  def testNestedPmapConstantDevices(self):
    if jax.device_count() < 6:
      raise SkipTest("this test requires >= 6 devices")

    devices = jax.devices()[:-2]
    shuffle(devices)
    f = self.pmap(self.pmap(lambda x: 3), devices=devices)
    shape = (2, len(devices) // 2, 3)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      ans = f(x)
    # self.assertEqual(count[0], 0)  # TODO(mattjj): don't compile for constants
    expected = 3 * np.ones(shape[:2])
    self.assertAllClose(ans, expected, check_dtypes=False)

    # Test that 'ans' was properly replicated across devices.
    expected_sharded = self.pmap(self.pmap(lambda x: x), devices=devices)(expected)
    self.assertTrue(ans.sharding == expected_sharded.sharding)

  def testNestedPmapConstantError(self):
    if config.disable_jit.value:
      raise SkipTest("error test doesn't apply with disable_jit")
    f = self.pmap(self.pmap(lambda x: 3))
    shape = (2, jax.device_count() // 2 + 1, 3)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    self.assertRaisesRegex(
        ValueError,
        (r"compiling computation that requires \d+ logical devices, "
        r"but only \d+ XLA devices are available .*"),
        lambda: f(x))

    # TODO(mattjj): check error message with explicit devices
    # if jax.device_count() > 1:
    #   f = pmap(pmap(lambda x: 3), devices=jax.devices()[:-1])
    #   shape = (2, jax.device_count() // 2, 3)
    #   x = jnp.arange(math.prod(shape)).reshape(shape)
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
    x = jnp.arange(math.prod(shape)).reshape(shape)
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
    x = jnp.ones(device_count, dtype='int32')
    ans = f(x)
    expected = 1 + np.arange(device_count)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAxisIndexNestedPmap(self):
    device_count = jax.device_count()
    if device_count < 4:
      raise SkipTest("test requires at least four devices")
    f = lambda axis: self.pmap(self.pmap(lambda x: x + lax.axis_index(axis), 'j'), 'i')
    x = jnp.ones((2, 2), dtype='int32')
    expected_j = np.broadcast_to(1 + np.arange(2), (2, 2))
    self.assertAllClose(f('j')(x), expected_j, check_dtypes=False)
    self.assertAllClose(f('i')(x), expected_j.T, check_dtypes=False)

  def testAxisIndexNd(self):
    device_count = jax.device_count()
    if device_count < 4:
      raise SkipTest("test requires at least four devices")
    f = lambda axes: self.pmap(self.pmap(lambda x: x + lax.axis_index(axes), 'j'), 'i')
    x = jnp.ones((2, 2), dtype='int32')
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
    self.assertAllClose(f(jnp.ones(shape, dtype='int32')),
                        (jnp.arange(device_count, dtype='int32') + 1) * 10)

  def testVmapOfPmap(self):
    device_count = jax.device_count()
    f0 = lambda x: x
    f1 = self.pmap(f0, axis_name='i')
    ax = self.rng().randn(2, device_count, 50, 60)
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
      keys = jax.tree.map(
          lambda x: jnp.broadcast_to(x, (N_DEVICES,) + x.shape),
          keys)
      return g(keys)

    ans = s(keys)  # doesn't crash
    self.assertEqual(ans.shape, (13, N_DEVICES))

  def testVmapOfPmap3(self):
    # https://github.com/jax-ml/jax/issues/3399
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
    ax = self.rng().randn(device_count, 2, 50, 60)
    bx = vmap(f1, in_axes=2, out_axes=2)(ax)
    self.assertAllClose(ax, bx, check_dtypes=False)

  def testVmapOfPmapTuple(self):
    device_count = jax.device_count()
    f0 = lambda *x: x
    f1 = self.pmap(f0, axis_name='i')

    ax = self.rng().randn(device_count, 2, 50, 60)
    ay = self.rng().randn(device_count, 30, 2)
    az1 = self.rng().randn(device_count, 20)
    az2 = self.rng().randn(2, device_count, 20)

    bx, by, bz = vmap(f1, in_axes=(1, 2, (None, 0)), out_axes=(1, 2, 0))(ax, ay, (az1, az2))

    self.assertAllClose(ax, bx, check_dtypes=False)
    self.assertAllClose(ay, by, check_dtypes=False)

    bz1, bz2 = bz
    expected_bz1 = np.broadcast_to(az1, (2,) + az1.shape)
    self.assertAllClose(expected_bz1, bz1, check_dtypes=False)
    self.assertAllClose(bz2, bz2, check_dtypes=False)

  def testPswapaxes(self):
    device_count = jax.device_count()
    shape = (device_count, 3, device_count, 5)
    x = np.arange(math.prod(shape)).reshape(shape)

    ans = self.pmap(lambda x: lax.pswapaxes(x, 'i', 1), axis_name='i')(x)
    expected = np.swapaxes(x, 0, 2)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testGradOfPswapaxes(self):
    device_count = jax.device_count()
    shape = (device_count, 1, device_count)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    w = np.arange(device_count, dtype=np.float32)

    @partial(self.pmap, axis_name='i')
    def f(x, w):
      g = lambda x: jnp.sum(lax.pswapaxes(x, 'i', 1) * w)
      return grad(g)(x)

    ans = f(x, w)
    expected = np.tile(w, reps=device_count).reshape(shape)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAllToAllReplicaGroups(self):
    # If num_devices = 4, these would be the inputs/outputs:
    # input = [[0, 1], [2, 3], [4, 5], [6, 7]]
    # axis_index_groups = [[0, 2], [1, 3]]
    # output = [[0, 4], [2, 6], [1, 5], [3, 7]]
    #
    # This is essentially like splitting the number of rows in the input in two
    # groups of rows, and swapping the two inner axes (axis=1 and axis=2), which
    # is exactly what the test case checks.
    device_count = jax.device_count()
    if device_count % 2 != 0:
      raise SkipTest('test requires an even number of devices')
    shape = (device_count, device_count // 2)
    x = np.arange(math.prod(shape)).reshape(shape)

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

  def testGradOfAllToAllReplicaGroups(self):
    device_count = jax.device_count()
    if device_count % 2 != 0:
      raise SkipTest('test requires an even number of devices')
    shape = (device_count, device_count // 2, 1)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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

  def testArrayBlockUntilReady(self):
    x = np.arange(jax.device_count())
    x = self.pmap(lambda x: x)(x)
    x.block_until_ready()  # doesn't crash

  @ignore_jit_of_pmap_warning()
  def testJitPmapComposition(self):
    f = lambda x: x - lax.psum(x, 'i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    expected = x - np.sum(x, 0)

    ans = jit(self.pmap(f, 'i'))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = self.pmap(jit(f), 'i')(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCompositionWithJitTwice(self):
    @jit
    def f(x):
      y = jnp.float32(2) * x

      @jit
      def g(z):
        return self.pmap(lambda x: x[jnp.newaxis] * y)(z)

      return g(x)

    f(np.arange(1., dtype='float32').reshape((1, 1)))  # doesn't crash

  @ignore_jit_of_pmap_warning()
  def testIssue1065(self):
    # from https://github.com/jax-ml/jax/issues/1065
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

  def testArrayGetItem(self):
    f = lambda x: 2 * x
    f = self.pmap(f, axis_name='i')

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    y = f(x)
    self.assertIsInstance(y, jax.Array)
    self.assertIsInstance(y, array.ArrayImpl)

    z = y[0]  # doesn't crash
    self.assertAllClose(z, 2 * x[0], check_dtypes=False)

  # TODO(mattjj): this fails with multiple devices (unless we add a jit)
  # because we assume eager ops (like scan here) can't require more than 1
  # replica.
  @unittest.skip("need eager multi-replica support")
  def testPostProcessMap(self):
    # test came from https://github.com/jax-ml/jax/issues/1369
    nrep = jax.device_count()

    def pmvm(a, b):
      a = a.reshape((nrep, -1, a.shape[1]))
      func = self.pmap(lambda z: jnp.dot(z, b))
      return func(a).reshape(b.shape)

    n = nrep * 2
    rng = self.rng()
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
    self.assertAllClose(f([np.array([i] * ndevices) for i in range(500)]),
                        jnp.array([sum(vals)] * ndevices))

  @jax.default_matmul_precision("float32")
  def testPostProcessMap2(self):
    # code from https://github.com/jax-ml/jax/issues/2787
    def vv(x, y):
      """Vector-vector multiply"""
      return jnp.dot(x, y)

    def distributed_matrix_vector(x, y):
      """Matrix vector multiply. First batch it and then row by row"""
      fv = lambda z: lax.map(lambda j: vv(j, y), z)
      res = self.pmap(fv)(x.reshape((jax.device_count(), -1) + tuple(x.shape[1:])))
      res = res.reshape(res.shape[0] * res.shape[1], *res.shape[2:])
      return res

    key = lambda: random.PRNGKey(1)
    x = random.normal(key(), (80, 50))
    batched_mvm = vmap(lambda b: distributed_matrix_vector(x, b), in_axes=0)
    y = random.normal(key(), (10, 50, 1))
    result = batched_mvm(y)
    expected = jnp.einsum('ij,njk->nik', x, y)
    self.assertAllClose(result, expected, check_dtypes=False, atol=1e-3,
                        rtol=1e-3)

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def testAxisIndexRemat(self, remat):
    # https://github.com/jax-ml/jax/issues/2716
    n = len(jax.devices())

    def f(key):
      key = random.fold_in(key, jax.lax.axis_index('i'))
      return random.bernoulli(key, p=0.5)

    keys = random.split(random.PRNGKey(0), n)
    self.pmap(remat(f), axis_name='i')(keys)

  def testPmapMapVmapCombinations(self):
    # https://github.com/jax-ml/jax/issues/2822
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

    x = random.normal(random.PRNGKey(1), (40, 5))
    y = random.normal(random.PRNGKey(1), (5, 5))

    result1 = vmap(lambda b: matrix_vector(x, b, True))(y)       # vmap + pmap
    result2 = lax.map(lambda b: matrix_vector(x, b, False), y)   # map + map
    with ignore_jit_of_pmap_warning():
      result3 = lax.map(lambda b: matrix_vector(x, b, True), y)    # map + pmap
    result4 = jnp.stack([matrix_vector(x, b, False) for b in y])  # none + map

    self.assertAllClose(result1, result2, check_dtypes=False, atol=1e-3, rtol=1e-3)
    self.assertAllClose(result1, result3, check_dtypes=False, atol=1e-3, rtol=1e-3)
    self.assertAllClose(result1, result4, check_dtypes=False, atol=1e-3, rtol=1e-3)

  def testPmapAxisNameError(self):
    # https://github.com/jax-ml/jax/issues/3120
    a = np.arange(4)[np.newaxis,:]
    def test(x):
      return jax.lax.psum(x, axis_name='batch')

    with self.assertRaisesRegex(NameError, "unbound axis name: batch"):
      self.pmap(test)(a)

  def testPsumOnBooleanDtype(self):
    # https://github.com/jax-ml/jax/issues/3123
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
    self.assertIs(w(), None)

  def testJitOfPmapWarningMessage(self):
    device_count = jax.device_count()

    if device_count == 1 or config.disable_jit.value:
      raise SkipTest("test requires at least two devices")

    def foo(x): return x

    with self.assertWarnsRegex(UserWarning, "The jitted function foo includes a pmap"):
      jit(self.pmap(foo))(jnp.arange(device_count))

  def testJitOfPmapOutputSharding(self):
    device_count = jax.device_count()

    if device_count == 1 or config.disable_jit.value:
      raise SkipTest("test requires at least two devices")

    @jax.jit
    @jax.pmap
    def foo(x): return x + x

    x = np.ones((2,2,2), dtype=np.float32)
    for _ in range(10):
      # Does not crash.
      with jtu.ignore_warning(
          message=".*Using jit-of-pmap can lead to inefficient data movement"):
        x = foo(x)

  @jtu.ignore_warning(
      message=".*Using jit-of-pmap can lead to inefficient data movement")
  def testJitOfPmapLowerHasReplicaAttributes(self):
    device_count = jax.device_count()

    if device_count == 1 or config.disable_jit.value:
      raise SkipTest("test requires at least two devices")

    @jax.jit
    @jax.pmap
    def foo(x): return x + x

    x = np.ones((2,2,2), dtype=np.float32)

    hlo = foo.lower(x).as_text("stablehlo")
    self.assertIn(f"mhlo.num_replicas = {2}", hlo)
    self.assertIn("mhlo.num_partitions = 1", hlo)

  def testPsumZeroCotangents(self):
    # https://github.com/jax-ml/jax/issues/3651
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
    # code from https://github.com/jax-ml/jax/issues/1062 @shoyer
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
    # https://github.com/jax-ml/jax/issues/4223
    def fn(indices):
      return jnp.equal(indices, jnp.arange(3)).astype(jnp.float32)
    mapped_fn = self.pmap(fn, axis_name='i', backend='cpu')
    mapped_fn = self.pmap(mapped_fn, axis_name='j', backend='cpu')
    indices = np.array([[[2], [1]], [[0], [0]]])
    mapped_fn(indices)  # doesn't crash

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
    if (jtu.test_device_matches(['cpu']) and
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
    # Regression test for https://github.com/jax-ml/jax/issues/6022
    @partial(self.pmap, axis_name='i')
    def func(_):
      return jax.lax.psum(dtype(0), axis_name='i')
    unused_arg = jnp.arange(jax.device_count())
    out_dtype = func(unused_arg).dtype
    self.assertEqual(out_dtype, dtype)

  def test_num_replicas_with_switch(self):
    # https://github.com/jax-ml/jax/issues/7411
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
      {"testcase_name": f"_{axis_size=}", "axis_size": axis_size}
      for axis_size in [1, 2])
  def test_grad_of_pmap_compilation_caching(self, axis_size):
    if len(jax.local_devices()) < axis_size:
      raise SkipTest("too few devices for test")
    if config.disable_jit.value:
      raise SkipTest("caching doesn't apply with jit disabled")

    @jax.pmap
    def f(x):
      return jnp.sin(x)

    # warm-up the cache
    x = jnp.ones(axis_size)
    _, f_bwd  = jax.vjp(f, x)
    _ = f_bwd(x)

    with jtu.count_jit_and_pmap_lowerings() as count:  # noqa: F841
      _, f_bwd2  = jax.vjp(f, x)
      _ = f_bwd(x)
      _ = f_bwd2(x)
    self.assertEqual(count[0], 0)  # cache hits on fwd and bwd

  def testSizeOverflow(self):
    if config.disable_jit.value:
      # TODO(sharadmv, mattjj): investigate and fix this issue
      raise SkipTest("OOMs in eager mode")
    x = jnp.arange(1)
    x = self.pmap(lambda _: jnp.ones([8, 267736, 1024], dtype=jnp.int8))(x)
    self.assertEqual(x.size, 8 * 267736 * 1024)
    self.assertEqual(type(x.size), int)

  def test_axis_env_length(self):
    f = lambda x: jax.pmap(g)(jnp.array([x]))[0]
    def g(x):
      assert len(core.get_axis_env().axis_names()) == 1
      return x
    jax.grad(f)(3.)  # doesn't fail

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_of_pmap(self, remat):
    f = remat(jax.pmap(lambda x: jnp.sin(jnp.sin(x))))
    jtu.check_grads(f, (jnp.arange(1.),), order=2, modes=["rev"])

    x = jnp.arange(1.)
    jaxpr = jax.make_jaxpr(jax.linearize(f, x)[1])(x)
    self.assertIn(' sin ', str(jaxpr))
    self.assertIn(' cos ', str(jaxpr))

  @parameterized.named_parameters(
      {"testcase_name": f"{suffix}", "remat": remat}
      for suffix, remat in [
          ('', jax.remat),
          ('_new', new_checkpoint),
      ])
  def test_remat_of_pmap_policy(self, remat):
    g = jax.pmap(lambda x: jnp.sin(jnp.sin(x)))
    x = jnp.arange(1.)

    save_cos = lambda prim, *_, **__: str(prim) == 'cos'
    f = remat(g, policy=save_cos)
    _, f_vjp = jax.vjp(f, x)
    jaxpr = f_vjp.args[0].func.args[1]
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 0)

    save_sin = lambda prim, *_, **__: str(prim) == 'sin'
    f = remat(g, policy=save_sin)
    _, f_vjp = jax.vjp(f, x)
    jaxpr = f_vjp.args[0].func.args[1]
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 0)
    self.assertEqual(jaxpr_text.count(' cos '), 2)

    save_nothing = lambda prim, *_, **__: False
    f = remat(g, policy=save_nothing)
    _, f_vjp = jax.vjp(f, x)
    jaxpr = f_vjp.args[0].func.args[1]
    jaxpr_text = str(jaxpr)
    self.assertEqual(jaxpr_text.count(' sin '), 1)
    self.assertEqual(jaxpr_text.count(' cos '), 2)

  def test_pmap_lower_arg_info(self):
    def f(x, y, *args, **kwargs):
      return y['hi'] + args[1] + sum(kwargs.values())

    lowered = jax.pmap(f).lower(
      {'hi': jnp.array([1.])}, {'hi': jnp.array([2.])}, jnp.array([3.]),
      jnp.array([4.]), z=jnp.array([5.]), w=jnp.array([6.]))
    hlo_str = mlir.module_to_string(lowered.compiler_ir('stablehlo'))
    self.assertNotIn("\"x\"", hlo_str)
    self.assertIn("y['hi']", hlo_str)
    self.assertIn("args[0]", hlo_str)
    self.assertIn("args[1]", hlo_str)
    self.assertIn("kwargs['z']", hlo_str)
    self.assertIn("kwargs['w']", hlo_str)

  def test_pmap_lower_result_info(self):
    def f(x, y, z):
      return {'a': x, 'b': [y]}

    lowered = jax.pmap(f).lower(jnp.array([1.]), (jnp.array([2]),),
                                [jnp.array([3])])
    hlo_str = mlir.module_to_string(lowered.compiler_ir('stablehlo'))
    self.assertIn("jax.result_info = \"['a']\"", hlo_str)
    self.assertIn("jax.result_info = \"['b'][0][0]\"", hlo_str)

  def test_axis_name_shadowing_with_vmap(self):
    # vmap-of-pmap with mismatched axis sizes
    jax.vmap(jax.pmap(lambda x: 2 * x, axis_name='i'),
             axis_name='i')(jax.numpy.ones((2, 1)))  # don't crash

    # vmap-of-pmap with matched axis sizes
    jax.vmap(jax.pmap(lambda x: 2 * x, axis_name='i'),
             axis_name='i')(jax.numpy.ones((1, 1)))  # don't crash

    # vmap-of-vmap with mismatched axis sizes
    jax.vmap(jax.vmap(lambda x: 2 * x, axis_name='i'),
             axis_name='i')(jax.numpy.ones((2, 1)))  # don't crash

    # vmap-of-vmap with matched axis sizes
    jax.vmap(jax.vmap(lambda x: 2 * x, axis_name='i'),
             axis_name='i')(jax.numpy.ones((1, 1)))  # don't crash

  @jtu.run_on_devices("cpu")
  def test_pmap_stack_size(self):
    # Regression test for https://github.com/jax-ml/jax/issues/20428
    # pmap isn't particularly important here, but it guarantees that the CPU
    # client runs the computation on a threadpool rather than inline.
    if jax.device_count() < 2:
      raise SkipTest("test requires at least two devices")
    x = jnp.eye(200)
    y = jax.pmap(jax.scipy.linalg.expm)(jnp.array([x, x]))
    y.block_until_ready()  # doesn't crash

  def test_pmap_of_prng_key(self):
    # Regression test for https://github.com/jax-ml/jax/issues/20392
    keys = jax.random.split(jax.random.key(0), jax.device_count())
    result1 = jax.pmap(jax.random.bits)(keys)
    with jtu.ignore_warning(
        category=UserWarning, message="The jitted function bits includes a pmap"):
      result2 = jax.jit(jax.pmap(jax.random.bits))(keys)
    self.assertArraysEqual(result1, result2)


@jtu.pytest_mark_if_available('multiaccelerator')
class CppPmapTest(PythonPmapTest):

  @property
  def pmap(self):
    if config.pmap_shmap_merge.value:
      return src_api.pmap
    return src_api._cpp_pmap

  def pmap_fast_path_is_enabled(self):
    num_devices = jax.device_count()
    f = jax.pmap(lambda x: x+1)
    size = f._cache_size()
    f(np.zeros([num_devices], dtype=np.float32))
    self.assertEqual(f._cache_size(), size+1)

  def test_cache_hits_across_threads(self):
    f = lambda x: x+1
    inputs = np.zeros([jax.device_count()], dtype=np.float32)
    pmaped_f = self.pmap(f)
    pmaped_f(inputs)
    self.assertEqual(pmaped_f._cache_size, 1)

    # Note: We do not call jax.pmap in the other thread but we reuse the same
    # object.
    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
      futures.append(executor.submit(lambda: pmaped_f(inputs)))
      outputs = [f.result() for f in futures]

    np.testing.assert_array_equal(pmaped_f(inputs), outputs[0])
    if pmaped_f._cache_size != 1:
      print(pmaped_f._debug_cache_keys())
    self.assertEqual(pmaped_f._cache_size, 1)

  def test_cache_uses_jax_key(self):
    f = lambda x: x+1
    inputs = np.zeros([jax.device_count()], dtype=np.float32)
    pmaped_f = self.pmap(f)
    pmaped_f(inputs)
    self.assertEqual(pmaped_f._cache_size, 1)

    config.update_thread_local_jit_state()

    pmaped_f(inputs)
    self.assertEqual(pmaped_f._cache_size, 1)

  def test_constants_fallback(self):
    fn = pmap(lambda x, y: x + y, in_axes=(0, None))

    for _ in range(2):
      fn(np.zeros((jax.device_count(), 5), dtype=np.float32), 2.0)


@jtu.pytest_mark_if_available('multiaccelerator')
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


@jtu.pytest_mark_if_available('multiaccelerator')
class VmapPmapCollectivesTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      {"testcase_name": f"_collective={collective.__name__}".replace(" ", ""),
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

  @parameterized.named_parameters(
      {"testcase_name": f"_collective={collective.__name__}".replace(" ", ""),
       "collective": collective}
      for collective in [lax.psum, lax.pmean, lax.pmax, lax.pmin])
  def testCollectivesWithVmap2(self, collective):
    def f(map1, map2):
      @partial(map1, axis_name='i')
      @partial(map2, axis_name='j')
      def f(x, y):
        return x + collective(x.dot(y), ('i', 'j'))
      return f

    if jax.device_count() < 8:
      raise SkipTest("test requires at least eight devices")
    x = jnp.arange(4*2*64*64, dtype=float).reshape(4, 2, 64, 64)
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

  def testPPermuteAgreesWithVmap(self):
    if jax.device_count() < 3:
      raise SkipTest("test requires at least three devices")

    def f(x):
      return lax.ppermute(x, 'i', [[1, 0], [2, 1], [0, 2]])

    xs = jnp.arange(3) * 10
    ys = jax.pmap(f, axis_name='i')(xs)
    zs = jax.vmap(f, axis_name='i')(xs)
    self.assertAllClose(ys, zs, check_dtypes=True)

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}_vmap={vmap_axis}",
       "split_axis": split_axis, "concat_axis": concat_axis, "vmap_axis": vmap_axis}
      for split_axis, concat_axis, vmap_axis in it.product(range(3), range(3), range(4)))
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

      x = np.ones(start_shape)
      self.assertEqual(reference(x, split_axis, concat_axis, vmap_axis).shape,
                       expected_shape)

    verify_ref()

    shape = (jax.device_count(),) * 5
    x = jnp.arange(math.prod(shape)).reshape(shape)
    self.assertAllClose(pmap(vmap(f, in_axes=vmap_axis), axis_name='i')(x),
                        reference(x, split_axis, concat_axis, vmap_axis))

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}",
       "split_axis": split_axis, "concat_axis": concat_axis}
      for split_axis, concat_axis in it.product(range(3), range(3)))
  def testAllToAllVsVmap(self, split_axis, concat_axis):
    def f(x):
      return lax.all_to_all(x, 'i', split_axis=split_axis, concat_axis=concat_axis)

    shape = (jax.device_count(),) * 4
    x = jnp.arange(math.prod(shape)).reshape(shape)
    self.assertAllClose(pmap(f, axis_name='i')(x),
                        vmap(f, axis_name='i')(x))

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}_axes={''.join(axes)}",
       "axes": axes, "split_axis": split_axis, "concat_axis": concat_axis}
      for axes, split_axis, concat_axis
      in it.product([('i', 'j'), ('j', 'i')], range(3), range(3)))
  @unittest.skip("multi-axis all_to_all broken after #4835")  # TODO(mattjj,apaszke)
  def testAllToAllMultipleAxesVsVmap(self, axes, split_axis, concat_axis):
    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")

    def f(x):
      return lax.all_to_all(x, axes, split_axis=split_axis, concat_axis=concat_axis)

    shape = (2, 2, 4, 4, 4)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    self.assertAllClose(pmap(pmap(f, axis_name='j'), axis_name='i')(x),
                        vmap(vmap(f, axis_name='j'), axis_name='i')(x))

  @parameterized.named_parameters([
    ('AllGather', lax.all_gather),
    ('ReduceScatter', lax.psum_scatter),
  ])
  def testWithVmap(self, prim):
    def f(map2):
      return jax.pmap(map2(partial(prim, axis_name='i')), axis_name='i')

    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")
    x = jnp.ones((2, 2, 2, 64))
    self.assertAllClose(f(jax.pmap)(x), f(jax.vmap)(x))

  @parameterized.named_parameters(it.chain.from_iterable([
    ('AllGather' + ('Tiled' if tiled else ''), lax.all_gather, tiled),
    ('ReduceScatter' + ('Tiled' if tiled else ''), lax.psum_scatter, tiled),
  ] for tiled in (False, True)))
  def testVsVmap(self, prim, tiled):
    if jax.device_count() < 4:
      raise SkipTest("test requires at least four devices")
    shape = (4, 4, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    f = partial(prim, axis_name='i', tiled=tiled)
    self.assertAllClose(vmap(f, axis_name='i')(x), pmap(f, axis_name='i')(x))


@jtu.pytest_mark_if_available('multiaccelerator')
class PmapWithDevicesTest(jtu.JaxTestCase):

  def testAllDevices(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i',
             devices=jax.devices())
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    x = self.rng().rand(1, 500, 500)
    r0 = f0(x)
    r1 = f1(x)
    expected = np.expand_dims(np.dot(x.squeeze(), x.squeeze().T), 0)
    self.assertAllClose(r0, expected, atol=1e-6, rtol=1e-3)
    self.assertAllClose(r1, expected, atol=1e-6, rtol=1e-3)

  def testNoDevicesError(self):
    f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i', devices=[])
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
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
    if config.disable_jit.value:
      raise SkipTest("error doesn't apply when jit is disabled")
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
    if config.disable_jit.value:
      raise SkipTest("disable_jit requires num devices to equal axis size")

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

  def testNestedPmapsBools(self):
    if jax.device_count() % 2 != 0:
      raise SkipTest
    if config.disable_jit.value:
      raise SkipTest("disable_jit requires num devices to equal axis size")

    # Devices specified in outer pmap are OK
    @partial(pmap, axis_name='i', devices=jax.devices())
    def foo(x):
      @partial(pmap, axis_name='j')
      def bar(y):
        return jnp.logical_not(y)
      return bar(x)

    x = jnp.ones((jax.device_count() // 2, 2), jnp.bool_)
    ans = foo(x)
    expected = jnp.zeros((jax.device_count() // 2, 2), jnp.bool_)
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
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    ans = grad(lambda x: jnp.sum(jnp.sin(x)))(x)
    expected = grad(lambda x: jnp.sum(f(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPmapStaticArgnums(self):
    @partial(pmap, axis_name='i', static_broadcasted_argnums=1)
    def f(x, y):
      return jnp.sin(x + y())
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    y = lambda: 3.

    ans = f(x, y)
    expected = np.sin(x + 3.)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPmapInAxesBasic(self):
    @partial(pmap, in_axes=(1, 2))
    def f(x, y):
      return jnp.sin(x + y)
    xshape = (2, jax.device_count(), 4)
    x = np.arange(math.prod(xshape)).reshape(xshape)
    yshape = (2, 4, jax.device_count())
    y = np.arange(math.prod(yshape)).reshape(yshape)

    self.assertAllClose(f(x, y),
                        jnp.sin(x.transpose((1, 0, 2)) + y.transpose((2, 0, 1))))

  def testPmapInAxesGrad(self):
    def f(x, y, z):
      return jnp.sin(x + y + z)
    fp = pmap(f, in_axes=(1, 2, None))
    fv = vmap(f, in_axes=(1, 2, None))
    xshape = (5, jax.device_count(), 7)
    x = np.arange(math.prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (5, 7, jax.device_count())
    y = np.arange(math.prod(yshape), dtype=np.float32).reshape(yshape)
    zshape = (5, 7)
    z = np.arange(math.prod(zshape), dtype=np.float32).reshape(zshape)

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
    x = np.arange(math.prod(xshape)).reshape(xshape)
    yshape = (2, 4)
    y = np.arange(math.prod(yshape)).reshape(yshape)

    self.assertAllClose(f(x, y),
                        (jnp.sin(x.transpose((1, 0, 2)) + y).transpose((1, 2, 0)), y * 2))

  def testPmapDictOutAxes(self):
    # see issue #6410
    @partial(pmap, out_axes={'a': 0})
    def f(x):
      return {'a': x}
    device_count = jax.device_count()
    x = jnp.arange(device_count)
    jax.tree.map(self.assertAllClose, f(x), {'a': x})

  @jtu.sample_product(
    in_axes=all_bdims((3, 4), (3, 1), (1, 4), pmap=True),
    out_axes=out_bdims((3, 4), True),
  )
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
    x = np.arange(math.prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (5, jax.device_count(), 7)
    y = np.arange(math.prod(yshape), dtype=np.float32).reshape(yshape)
    self.assertAllClose(jax.grad(mk_case(pmap))(x, y),
                        jax.grad(mk_case(vmap))(x, y))


@jtu.pytest_mark_if_available('multiaccelerator')
class ArrayTest(jtu.JaxTestCase):

  def testThreadsafeIndexing(self):
    # NOTE(skye): I picked these values to be big enough to cause interesting
    # execution overlap, but small enough to not use too much memory. YMMV.
    shape = (8, 4000, 1000)

    if jax.device_count() < shape[0]:
      raise SkipTest(f"requires {shape[0]} devices")

    x = jnp.arange(math.prod(shape)).reshape(shape)
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

    x = jnp.arange(math.prod(shape)).reshape(shape)
    sharded_x = pmap(lambda x: x)(x)
    self.assertIsNone(sharded_x._npy_value)

    for i in range(8):
      self.assertIsInstance(sharded_x[i], array.ArrayImpl)
    self.assertIsNone(sharded_x._npy_value)

  def test_device_put_sharded(self):
    devices = jax.local_devices()
    n_devices = len(devices)
    x = [np.arange(i, i + 4) for i in range(n_devices)]
    y = jax.device_put_sharded(x, devices)
    self.assertIsInstance(y, array.ArrayImpl)
    self.assertIsInstance(y.sharding, jax.sharding.PmapSharding)
    for s in y.addressable_shards:
      self.assertArraysEqual(s.data, y[s.index])
      self.assertEqual(s.replica_id, 0)
    buffers = getattr(y, '_arrays')
    self.assertEqual(len(buffers), len(devices))
    self.assertTrue(all(b.devices() == {d} for b, d in zip(buffers, devices)))
    self.assertArraysEqual(y, jnp.stack(x))

  def test_device_put_sharded_pytree(self):
    devices = jax.local_devices()
    n_devices = len(devices)
    x = [(i, np.arange(i, i + 4)) for i in range(n_devices)]
    y1, y2 = jax.device_put_sharded(x, devices)

    self.assertIsInstance(y1, array.ArrayImpl)
    self.assertArraysEqual(y1, jnp.array([a for a, _ in x]))
    y1_buffers = getattr(y1, '_arrays')
    self.assertTrue(all(b.devices() == {d} for b, d in zip(y1_buffers, devices)))

    self.assertIsInstance(y2, array.ArrayImpl)
    self.assertArraysEqual(y2, jnp.vstack([b for _, b in x]))
    y2_buffers = getattr(y2, '_arrays')
    self.assertTrue(all(b.devices() == {d} for b, d in zip(y2_buffers, devices)))

  def test_device_put_replicated(self):
    devices = jax.local_devices()
    x = np.arange(1, 5)
    y = jax.device_put_replicated(x, devices)

    self.assertIsInstance(y, array.ArrayImpl)
    buffers = getattr(y, '_arrays')
    self.assertEqual(len(buffers), len(devices))
    self.assertTrue(all(b.devices() == {d} for b, d in zip(buffers, devices)))
    self.assertArraysEqual(y, np.stack([x for _ in devices]))

  def test_device_put_replicated_pytree(self):
    devices = jax.local_devices()
    xs = {'a': np.arange(1, 5), 'b': np.arange(3)}
    ys = jax.device_put_replicated(xs, devices)
    self.assertIsInstance(ys, dict)
    y1, y2 = ys['a'], ys['b']

    self.assertIsInstance(y1, array.ArrayImpl)
    y1_buffers = getattr(y1, '_arrays')
    self.assertEqual(len(y1_buffers), len(devices))
    self.assertTrue(all(b.devices() == {d} for b, d in zip(y1_buffers, devices)))
    self.assertArraysEqual(y1, np.stack([xs['a'] for _ in devices]))

    self.assertIsInstance(y2, array.ArrayImpl)
    y2_buffers = getattr(y2, '_arrays')
    self.assertEqual(len(y2_buffers), len(devices))
    self.assertTrue(all(b.devices() == {d} for b, d in zip(y2_buffers, devices)))
    self.assertArraysEqual(y2, np.stack([xs['b'] for _ in devices]))

  def test_repr(self):
    x = jax.device_put_replicated(1, jax.devices())
    self.assertStartsWith(repr(x), 'Array')

  def test_delete_is_idempotent(self):
    x = jax.device_put_replicated(1, jax.devices())
    x.delete()
    x.delete()

    with self.assertRaisesRegex(RuntimeError, 'Array has been deleted.'):
      _ = x[0]


class SpecToIndicesTest(jtu.JaxTestCase):

  def testShardsPerAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=map(pxla.Chunked, ([2], [2])),
                             mesh_mapping=map(pxla.ShardedAxis, (0, 1)))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     ((slice(0,2), slice(0,4)),
                      (slice(0,2), slice(4,8)),
                      (slice(2,4), slice(0,4)),
                      (slice(2,4), slice(4,8))))

  def testShardedAxisPermutation(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=map(pxla.Chunked, ([2], [2])),
                             mesh_mapping=map(pxla.ShardedAxis, (1, 0)))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
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
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     ((slice(0,2), slice(0,4)),
                      (slice(2,4), slice(0,4)),
                      (slice(0,2), slice(4,8)),
                      (slice(2,4), slice(4,8))) * 2)

  def testUnshardedAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Chunked([2]), pxla.NoSharding()),
                             mesh_mapping=(pxla.ShardedAxis(0),))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     ((slice(0,2), slice(None)),
                      (slice(2,4), slice(None))))

  def testNoSharding(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.NoSharding(), pxla.NoSharding()),
                             mesh_mapping=())
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     ((slice(None), slice(None)),))

  def testUnmaterializedAxis(self):
    shape = (4, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(4), pxla.NoSharding()),
                             mesh_mapping=(pxla.ShardedAxis(0),))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     ((0, slice(None)),
                      (1, slice(None)),
                      (2, slice(None)),
                      (3, slice(None))))

    shape = (2, 2)
    spec = pxla.ShardingSpec(sharding=(pxla.NoSharding(), pxla.Unstacked(2)),
                             mesh_mapping=(pxla.ShardedAxis(0),))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     ((slice(None), 0),
                      (slice(None), 1)))

  def testReplicationAfterUnsharded(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.NoSharding()),
                             mesh_mapping=(pxla.ShardedAxis(0), pxla.Replicated(3)))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     tuple([(0, slice(None))] * 3 + [(1, slice(None))] * 3))

  def testReplicationPosition2(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.Chunked([2])),
                             mesh_mapping=(pxla.ShardedAxis(0), pxla.ShardedAxis(1), pxla.Replicated(3)))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     ((0, slice(0, 4)), (0, slice(0, 4)), (0, slice(0, 4)),
                      (0, slice(4, 8)), (0, slice(4, 8)), (0, slice(4, 8)),
                      (1, slice(0, 4)), (1, slice(0, 4)), (1, slice(0, 4)),
                      (1, slice(4, 8)), (1, slice(4, 8)), (1, slice(4, 8))))

  def testReplicationPosition1(self):
    shape = (2, 8)
    spec = pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.Chunked([2])),
                             mesh_mapping=(pxla.ShardedAxis(0), pxla.Replicated(3), pxla.ShardedAxis(1)))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
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
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     tuple([(0, slice(None)), (1, slice(None))] * 3))

  def testMultipleReplications(self):
    shape = (2, 7, 4)
    spec = pxla.ShardingSpec(
        sharding=(pxla.Unstacked(2), pxla.NoSharding(), pxla.Chunked([2])),
        mesh_mapping=(pxla.Replicated(3), pxla.Replicated(2),
                      pxla.ShardedAxis(0), pxla.Replicated(2),
                      pxla.ShardedAxis(1)))
    self.assertEqual(
        sharding_specs.spec_to_indices(shape, spec),
        ((0, slice(None), slice(0, 2)), (0, slice(None), slice(2, 4)),
         (0, slice(None), slice(0, 2)), (0, slice(None), slice(2, 4)),
         (1, slice(None), slice(0, 2)), (1, slice(None), slice(2, 4)),
         (1, slice(None), slice(0, 2)), (1, slice(None), slice(2, 4))) * 3 * 2)

  def testReplicatedScalar(self):
    shape = ()
    spec = pxla.ShardingSpec(sharding=(),
                             mesh_mapping=(pxla.Replicated(3),))
    self.assertEqual(sharding_specs.spec_to_indices(shape, spec),
                     ((), (), ()))


def _spec_str(spec):
  return (f"({spec.sharding},"
          f"{spec.mesh_mapping},)")


@jtu.pytest_mark_if_available('multiaccelerator')
class ShardArgsTest(jtu.JaxTestCase):

  def numpy_array(x):
    return x

  def device_array(x):
    return jax.device_put(x)

  # TODO(skye): add coverage for Arrays

  @parameterized.named_parameters(
      {"testcase_name":
       f"_{shape=}_spec={_spec_str(spec)}_arg={make_arg.__name__}"
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
          # replication + sharding
          [(2, 8), pxla.ShardingSpec(sharding=(pxla.Unstacked(2), pxla.NoSharding()),
                                     mesh_mapping=(pxla.ShardedAxis(0), pxla.Replicated(3)))],
          # replication, no sharding
          [(2, 8), pxla.ShardingSpec(sharding=(pxla.NoSharding(), pxla.NoSharding()),
                                     mesh_mapping=(pxla.Replicated(3),))],
          # replicated scalar
          [(), pxla.ShardingSpec(sharding=(),
                                 mesh_mapping=(pxla.Replicated(2), pxla.Replicated(3)))],
      ])
  def testShardArgs(self, shape, spec, make_arg):
    indices = sharding_specs.spec_to_indices(shape, spec)
    nshards = len(indices)
    if jax.device_count() < nshards:
      raise SkipTest
    x = np.arange(math.prod(shape)).reshape(shape)
    arg = make_arg(x)
    sharding = jax.sharding.PmapSharding(jax.devices()[:nshards], spec)
    results = pxla.shard_args([sharding], [None], [arg])
    self.assertEqual(len(results), 1)
    if isinstance(results[0], array.ArrayImpl):
      bufs = results[0]._arrays
    else:
      bufs = results[0]
    self.assertEqual(len(bufs), nshards)
    for buf, idx in zip(bufs, indices):
      self.assertAllClose(np.asarray(buf), x[idx], check_dtypes=False)


@jtu.pytest_mark_if_available('multiaccelerator')
class ArrayPmapTest(jtu.JaxTestCase):

  def test_pmap_input_array_output_array(self):
    input_shape = (jax.device_count(), 2)
    input_array, input_data = create_input_array_for_pmap(input_shape)

    f = jax.pmap(lambda x, y: x * y)
    out = f(input_array, input_array)

    expected = input_data * input_data

    self.assertIsInstance(out, array.ArrayImpl)
    for s in out.addressable_shards:
      self.assertArraysEqual(s.data, expected[s.index])
    self.assertArraysEqual(out, expected)

  def test_pmap_double_input_array_output_array(self):
    input_shape = (jax.device_count(), 2)
    input_array, input_data = create_input_array_for_pmap(input_shape)

    def f(x, y):
      assert x.shape == (2,)
      assert y.shape == (2,)
      return x, y

    f = jax.pmap(f)
    out1, out2 = f(input_array, input_array)

    self.assertIsInstance(out1, array.ArrayImpl)
    self.assertIsInstance(out2, array.ArrayImpl)
    for s1, s2 in safe_zip(out1.addressable_shards, out2.addressable_shards):
      self.assertArraysEqual(s1.data, input_data[s1.index])
      self.assertArraysEqual(s2.data, input_data[s2.index])
    self.assertArraysEqual(out1, input_data)
    self.assertArraysEqual(out2, input_data)

  def test_pmap_array_in_axes_out_axes(self):
    dc = jax.device_count()
    input_shape = (dc, 2)
    a1, input_data = create_input_array_for_pmap(input_shape, in_axes=0)
    a2, _ = create_input_array_for_pmap(input_shape, in_axes=None,
                                        sharded_dim_size=a1.shape[0])

    def f(x, y):
      assert x.shape == (2,)
      assert y.shape == input_shape
      return x, y

    f = jax.pmap(f, in_axes=(0, None), out_axes=(None, 0))
    out1, out2 = f(a1, a2)

    self.assertIsInstance(out1, array.ArrayImpl)
    self.assertIsInstance(out2, array.ArrayImpl)
    self.assertEqual(out1.shape, (2,))
    self.assertEqual(out2.shape, (dc, dc, 2))
    for i, (s1, s2) in enumerate(safe_zip(out1.addressable_shards, out2.addressable_shards)):
      self.assertArraysEqual(s1.data, input_data[i])
      if config.pmap_no_rank_reduction.value:
        self.assertArraysEqual(s2.data, input_data[None])
      else:
        self.assertArraysEqual(s2.data, input_data)

  def test_pmap_array_sharding_mismatch(self):
    input_shape = (jax.device_count(), 2)
    a1, inp_data = create_input_array_for_pmap(input_shape, in_axes=None,
                                        sharded_dim_size=input_shape[0])

    f = jax.pmap(lambda x: x, in_axes=0, out_axes=0)
    out_array = f(a1)

    self.assertArraysEqual(out_array, inp_data)

  def test_pmap_array_devices_mismatch(self):
    if jax.device_count() <= 1:
      raise unittest.SkipTest('Skipping because this test needs more than '
                              '1 device.')
    input_shape = (jax.device_count(), 2)
    a1, inp_data = create_input_array_for_pmap(input_shape)

    f = jax.pmap(lambda x: x, devices=jax.devices()[::-1])
    out_array = f(a1)

    self.assertArraysEqual(out_array, inp_data)

  def test_amap(self):
    # Copied from an example mattjj@ posted in a chat thread.

    if jax.device_count() < 2:
      self.skipTest('Test requires >= 2 devices.')

    def amap(f, xs):
      ys = [f(jax.device_put(x, list(x.devices())[0])) for x in xs]
      return jax.device_put_sharded(ys, jax.local_devices()[:2])

    # leading axis is batch dim (i.e. mapped/parallel dim), of size 2
    x = jnp.array([[1., 0., 0.],
                   [0., 2., 3.]])

    # first pmapped computation
    y = jax.pmap(jnp.sin)(x)

    def dynamic_shape_function(y):
      nonzero_idx = y != 0
      results = y[nonzero_idx] ** 2
      return y.at[nonzero_idx].set(results)
    z = amap(dynamic_shape_function, y)

    # second pmapped computation
    w = jax.pmap(jnp.cos)(z)

    self.assertArraysEqual(w, jnp.cos(jnp.sin(x) ** 2))

  def test_same_out_sharding_id(self):
    if config.disable_jit.value:
      self.skipTest('Skip this under eager pmap mode.')
    shape = (jax.device_count(), 2)
    arr, inp_data = create_input_array_for_pmap(shape)

    f = pmap(lambda x: x)
    out1 = f(arr)
    self.assertArraysEqual(out1, inp_data)
    out1_sharding_id = id(out1.sharding)

    out2 = f(out1)
    self.assertArraysEqual(out2, inp_data)
    out2_sharding_id = id(out2.sharding)

    out3 = f(out2)
    self.assertArraysEqual(out3, inp_data)
    out3_sharding_id = id(out3.sharding)

    self.assertEqual(out1_sharding_id, out2_sharding_id)
    self.assertEqual(out1_sharding_id, out3_sharding_id)
    self.assertEqual(out2_sharding_id, out3_sharding_id)

  def test_array_with_pmap_sharding_copy_without_round_trip(self):

    def _compare_if_equal(out, out_copy):
      self.assertArraysEqual(out, out_copy)
      self.assertIsInstance(out_copy.sharding, jax.sharding.PmapSharding)
      self.assertEqual(out.sharding, out_copy.sharding)
      for o, o_copy in safe_zip(out.addressable_shards, out_copy.addressable_shards):
        self.assertArraysEqual(o.data, o_copy.data)
        self.assertEqual(o.device, o_copy.device)
        self.assertEqual(o.index, o_copy.index)
        self.assertEqual(o.replica_id, o_copy.replica_id)
        self.assertNotEqual(o.data.unsafe_buffer_pointer(),
                            o_copy.data.unsafe_buffer_pointer())

    out, _ = create_input_array_for_pmap((jax.device_count(),))
    out_copy = jnp.copy(out)
    _compare_if_equal(out, out_copy)

    out1, _ = create_input_array_for_pmap((1, jax.device_count(),), in_axes=1)
    out_copy1 = jnp.copy(out1)
    _compare_if_equal(out1, out_copy1)

  def test_device_put_sharded_transfer_guard(self):
    inp = jnp.arange(jax.device_count())
    arr_inp = [jax.device_put(i, d) for i, d in zip(inp, jax.devices())]

    with jax.transfer_guard("disallow_explicit"):
      jax.device_put_sharded(arr_inp, jax.devices())

  def test_jnp_stack(self):
    @jax.pmap
    def something(x):
      return (x + x).reshape([])

    z = something(np.arange(jax.device_count()))
    self.assertArraysEqual(jnp.stack([z[i] for i in range(jax.device_count())]),
                           np.arange(jax.device_count()) * 2)


class EagerPmapMixin:

  def setUp(self):
    super().setUp()
    self.eager_pmap_enabled = config.eager_pmap.value
    self.jit_disabled = config.disable_jit.value
    config.update('jax_disable_jit', True)
    config.update('jax_eager_pmap', True)
    self.warning_ctx = jtu.ignore_warning(
        message="Some donated buffers were not usable", category=UserWarning)
    self.warning_ctx.__enter__()

  def tearDown(self):
    self.warning_ctx.__exit__(None, None, None)
    config.update('jax_eager_pmap', self.eager_pmap_enabled)
    config.update('jax_disable_jit', self.jit_disabled)
    super().tearDown()

@jtu.pytest_mark_if_available('multiaccelerator')
class PythonPmapEagerTest(EagerPmapMixin, PythonPmapTest):

  def test_custom_jvp(self):

    @jax.custom_jvp
    def foo(x):
      return jnp.exp(x)
    @foo.defjvp
    def foo_jvp(xs, ts):
      (x,), (t,) = xs, ts
      return foo(x), t * 4.

    f = lambda x, t: jax.jvp(foo, (x,), (t,))
    x = jnp.arange(
        jax.local_device_count() * 5, dtype=jnp.dtype('float32')).reshape((
          jax.local_device_count(), 5))
    self.assertAllClose(self.pmap(f)(x, x), jax.vmap(f)(x, x))

  def test_custom_vjp(self):

    @jax.custom_vjp
    def foo(x):
      return jnp.exp(x)

    def foo_fwd(x):
      return foo(x), x
    def foo_bwd(_, g):
      return (g * 5.,)
    foo.defvjp(foo_fwd, foo_bwd)

    f = jax.grad(foo)
    x = jnp.arange(jax.local_device_count(), dtype=jnp.dtype('float32'))
    self.assertAllClose(self.pmap(f)(x), jax.vmap(f)(x))


@jtu.pytest_mark_if_available('multiaccelerator')
class CppPmapEagerTest(EagerPmapMixin, CppPmapTest):
  pass

@jtu.pytest_mark_if_available('multiaccelerator')
class PmapWithDevicesEagerTest(EagerPmapMixin, PmapWithDevicesTest):
  pass

@jtu.pytest_mark_if_available('multiaccelerator')
class VmapOfPmapEagerTest(EagerPmapMixin, VmapOfPmapTest):
  pass

@jtu.pytest_mark_if_available('multiaccelerator')
class ArrayPmapEagerTest(EagerPmapMixin, ArrayPmapTest):
  pass


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
