# Copyright 2023 The JAX Authors.
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

from functools import partial
import glob
import logging
import math
import os
import tempfile
import unittest

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import profiler
from jax._src import pjit
from jax._src import monitoring
from jax._src import test_util as jtu
from jax._src import api
from jax.experimental import profiler as exp_profiler
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from jax._src import compilation_cache as cc
import numpy as np

from jax.experimental.serialize_executable import (
    deserialize_and_load,
    serialize,
)

jax.config.parse_flags_with_absl()


@jtu.pytest_mark_if_available('multiaccelerator')
class PgleTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    cc.reset_cache()

  def tearDown(self):
    cc.reset_cache()
    super().tearDown()

  @unittest.skip("Test failing in CI")
  def testPGLEProfilerGetFDOProfile(self):
    mesh = jtu.create_global_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
    )
    def f(x, y):
      return x @ y

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1

    with config.pgle_profiling_runs(0):
      f_lowered = f.lower(x, y)
      compiled = f_lowered.compile()

    pgle_profiler = profiler.PGLEProfiler(1, 90)
    with config.enable_pgle(False):
      with profiler.PGLEProfiler.trace(pgle_profiler):
        compiled(x, y)

    fdo_profile = pgle_profiler.consume_fdo_profile()
    self.assertIsNotNone(fdo_profile)
    self.assertIn(b'custom', fdo_profile)

  @unittest.skip("Test failing in CI")
  def testPGLEProfilerGetFDOProfileLarge(self):
    mesh = jtu.create_global_mesh((2,), ('x',))
    its = 500

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
    )
    def f(x):
      agg = x
      for _ in range(its):
        agg = agg @ x
      return agg

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)

    with config.pgle_profiling_runs(0):
      f_lowered = f.lower(x)
      f_compiled = f_lowered.compile()

    pgle_profiler = profiler.PGLEProfiler(1, 90)
    with config.enable_pgle(False):
      with profiler.PGLEProfiler.trace(pgle_profiler):
        f_compiled(x)
    fdo_profile = pgle_profiler.consume_fdo_profile()
    self.assertEqual(fdo_profile.count(b'custom'), its)

  def testAutoPgle(self):
    mesh = jtu.create_global_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
    )
    def f(x):
      return x * 2

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    expected = x * 2

    with config.pgle_profiling_runs(2), config.enable_pgle(True):
      # Run 1: Module should be compiled without FDO. Two modules are expected
      # One is the funtion f, the other one is multi slice module
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      self.assertEqual(cache_miss_count[0], 2)

      # Run 2: Second PGLE run should not recompile the module
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      self.assertEqual(cache_miss_count[0], 0)

      # Run 3: The module should be recompiled with FDO profiles
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      self.assertEqual(cache_miss_count[0], 2)

      # Run 4: Fast-path should be used after PGLE is done
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      self.assertEqual(cache_miss_count[0], 0)

  def testAutoPgleWithAot(self):
    @jax.jit
    def f(x):
      return x * 2

    x = jnp.arange(1)
    expected = x * 2

    f_lowered = f.lower(x)
    serialized, in_tree, out_tree = serialize(f_lowered.compile())
    compiled = deserialize_and_load(serialized, in_tree, out_tree)

    with config.pgle_profiling_runs(1), config.enable_pgle(True):
      # Run 1
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(compiled(x), expected)
      self.assertEqual(cache_miss_count[0], 0)

      # Run 2
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(compiled(x), expected)
      self.assertEqual(cache_miss_count[0], 0)

  @unittest.skip("Test failing in CI")
  def testAutoPgleWithPersistentCache(self):

    @jax.jit
    def f(x):
      return x * 2

    x = jnp.arange(1)
    expected = x * 2

    profilers_dict = (
        pjit._most_recent_pjit_call_executable.weak_pgle_profiler_dict)
    with (config.enable_compilation_cache(True),
          config.enable_pgle(True),
          config.raise_persistent_cache_errors(True),
          config.raise_persistent_cache_errors(True),
          config.persistent_cache_min_entry_size_bytes(0),
          config.persistent_cache_min_compile_time_secs(0),
          config.pgle_profiling_runs(2),
          tempfile.TemporaryDirectory() as tmpdir):
      cc.set_cache_dir(tmpdir)
      # Run 1: Module should be compiled without FDO
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      self.assertEqual(cache_miss_count[0], 1)

      # Non-pgle profiled version of module should be saved
      non_pgle_profiled_files = os.listdir(tmpdir)
      self.assertLen(non_pgle_profiled_files, 1)

      # Run 2: Compilation should not be called
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      self.assertEqual(cache_miss_count[0], 0)

      # Run 3: Module should be compiled with FDO and stored to persistent cache
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      self.assertEqual(cache_miss_count[0], 1)

      for pgle_profiler in profilers_dict.values():
        self.assertTrue(pgle_profiler.is_enabled())
        self.assertTrue(pgle_profiler.is_fdo_consumed())
      # One module is PGLEd version another one is not PGLEd
      self.assertLen(os.listdir(tmpdir), 2)

      # Removing non-pgle profiled module from cache to check that later pgle
      # profiled version will be used.
      os.remove(os.path.join(tmpdir, non_pgle_profiled_files[0]))

      api.clear_caches()
      profilers_dict.clear()

      # Run 4: Persistent compilation cache should be hit PGLE profiler should
      # be disabled
      cache_hit = 0
      def check_if_cache_hit(event):
        nonlocal cache_hit
        if event == '/jax/compilation_cache/cache_hits':
          cache_hit += 1

      monitoring.register_event_listener(check_if_cache_hit)
      with jtu.count_cached_compilation_cache_miss() as cache_miss_count:
        self.assertArraysEqual(f(x), expected)
      monitoring._unregister_event_listener_by_callback(check_if_cache_hit)

      self.assertEqual(cache_miss_count[0], 1)
      self.assertEqual(cache_hit, 1)
      self.assertLen(profilers_dict, 1)
      for pgle_profiler in profilers_dict.values():
        self.assertFalse(pgle_profiler.is_enabled())
        self.assertFalse(pgle_profiler.is_fdo_consumed())

  def testPassingFDOProfile(self):
    mesh = jtu.create_global_mesh((2,), ('x',))

    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, PartitionSpec('x')),
        out_shardings=NamedSharding(mesh, PartitionSpec('x')),
    )
    def f(x, y):
      return x @ y

    shape = (16, 16)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1

    with config.pgle_profiling_runs(0):
      f_lowered = f.lower(x, y)
      compiled = f_lowered.compile()

    with tempfile.TemporaryDirectory() as tmpdir:
      jax.profiler.start_trace(tmpdir)
      compiled(x, y)
      jax.profiler.stop_trace()
      directories = glob.glob(os.path.join(tmpdir, 'plugins/profile/**/'))
      directories = [d for d in directories if os.path.isdir(d)]
      rundir = directories[-1]
      logging.info('rundir: %s', rundir)
      fdo_profile = exp_profiler.get_profiled_instructions_proto(rundir)

    if jtu.test_device_matches(['gpu']) and jtu.is_device_cuda():
      self.assertIn(b'custom', fdo_profile)

    logging.info('fdo_profile: %s', fdo_profile)
    # Test pass fdo_profile as compiler_options API works.
    f_lowered.compile(compiler_options={'fdo_profile': fdo_profile})


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
