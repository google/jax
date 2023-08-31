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

import hashlib
import os
import random
import sys
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import config
from jax import lax
from jax._src import cache_key
from jax._src import compiler
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.config import compilation_cache_include_metadata_in_key
from jax._src.lib import xla_client
import numpy as np


config.parse_flags_with_absl()
FLAGS = config.FLAGS


class CacheKeyTest(jtu.JaxTestCase):

  def test_compile_options(self):
    compile_options_not_filled = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    compile_options_filled = self.filled_compile_options()
    filled_hash1 = self.get_hashed_value(
        cache_key._hash_compile_options, compile_options_filled
    )
    filled_hash2 = self.get_hashed_value(
        cache_key._hash_compile_options, compile_options_filled
    )
    not_filled_hash3 = self.get_hashed_value(
        cache_key._hash_compile_options, compile_options_not_filled
    )
    self.assertEqual(filled_hash1, filled_hash2)
    self.assertNotEqual(filled_hash1, not_filled_hash3)

  def test_executable_build_options(self):
    compile_options_not_filled = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    compile_options_filled = self.filled_compile_options()
    filled_hash1 = self.get_hashed_value(
        cache_key._hash_executable_build_options,
        compile_options_filled.executable_build_options,
    )
    filled_hash2 = self.get_hashed_value(
        cache_key._hash_executable_build_options,
        compile_options_filled.executable_build_options,
    )
    not_filled_hash3 = self.get_hashed_value(
        cache_key._hash_executable_build_options,
        compile_options_not_filled.executable_build_options,
    )
    self.assertEqual(filled_hash1, filled_hash2)
    self.assertNotEqual(filled_hash1, not_filled_hash3)

  def test_debug_options(self):
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    hash1 = self.get_hashed_value(
        cache_key._hash_debug_options,
        compile_options.executable_build_options.debug_options,
    )
    hash2 = self.get_hashed_value(
        cache_key._hash_debug_options,
        compile_options.executable_build_options.debug_options,
    )
    self.assertEqual(hash1, hash2)
    new_debug_options = self.create_new_debug_options(
        compile_options.executable_build_options.debug_options
    )
    hash3 = self.get_hashed_value(
        cache_key._hash_debug_options, new_debug_options
    )
    self.assertNotEqual(hash1, hash3)

  def test_hash_platform(self):
    hash1 = self.get_hashed_value(
        cache_key._hash_platform, xla_bridge.get_backend()
    )
    hash2 = self.get_hashed_value(
        cache_key._hash_platform, xla_bridge.get_backend()
    )
    self.assertEqual(hash1, hash2)
    if xla_bridge.get_backend().platform != "cpu":
      cpu_backend = xla_bridge.get_backend("cpu")
      hash3 = self.get_hashed_value(cache_key._hash_platform, cpu_backend)
      self.assertNotEqual(hash1, hash3)

  def test_hash_int(self):
    hash1 = self.get_hashed_value(cache_key._hash_int, 90)
    hash2 = self.get_hashed_value(cache_key._hash_int, 8)
    hash3 = self.get_hashed_value(cache_key._hash_int, 8)
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  def test_hash_signed_int(self):
    hash1 = self.get_hashed_value(cache_key._hash_signed_int, 90)
    hash2 = self.get_hashed_value(cache_key._hash_signed_int, -90)
    hash3 = self.get_hashed_value(cache_key._hash_signed_int, -8)
    hash4 = self.get_hashed_value(cache_key._hash_signed_int, -8)
    self.assertEqual(hash3, hash4)
    self.assertNotEqual(hash1, hash2)
    self.assertNotEqual(hash1, hash3)

  def test_hash_bool(self):
    hash1 = self.get_hashed_value(cache_key._hash_bool, False)
    hash2 = self.get_hashed_value(cache_key._hash_bool, True)
    hash3 = self.get_hashed_value(cache_key._hash_bool, True)
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  def test_hash_string(self):
    hash1 = self.get_hashed_value(cache_key._hash_string, "foo")
    hash2 = self.get_hashed_value(cache_key._hash_string, "bar")
    hash3 = self.get_hashed_value(cache_key._hash_string, "bar")
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  @parameterized.parameters([False, True])
  def test_same_key(self, produce_original_cache_key):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    self.assertEqual(
        cache_key.get(computation, devices, compile_options, backend,
                      produce_original_cache_key=produce_original_cache_key),
        cache_key.get(computation, devices, compile_options, backend,
                      produce_original_cache_key=produce_original_cache_key),
    )

  @parameterized.parameters([False, True])
  def test_different_key(self, produce_original_cache_key):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options_not_filled = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    compile_options_filled = self.filled_compile_options()
    backend = xla_bridge.get_backend()
    self.assertNotEqual(
        cache_key.get(
            computation, devices, compile_options_not_filled, backend,
            produce_original_cache_key=produce_original_cache_key
        ),
        cache_key.get(computation, devices, compile_options_filled, backend,
                      produce_original_cache_key=produce_original_cache_key),
    )

  @parameterized.parameters([False, True])
  def test_different_computations(self, produce_original_cache_key):
    computation1 = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    computation2 = jax.jit(lambda x, y: x * y).lower(2, 2).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    self.assertNotEqual(
        cache_key.get(computation1, devices, compile_options, backend,
                      produce_original_cache_key=produce_original_cache_key),
        cache_key.get(computation2, devices, compile_options, backend,
                      produce_original_cache_key=produce_original_cache_key),
    )

  @parameterized.parameters([False, True],  # include_metadata
                            [False, True])  # produce_original_cache_key
  def test_identical_computations_different_metadata(
      self, include_metadata, produce_original_cache_key
  ):
    f = lambda x, y: lax.mul(lax.add(x, y), 2)
    g = lambda x, y: lax.mul(lax.add(x, y), 2)
    assert id(f) != id(g)
    computation1 = jax.jit(f).lower(1, 1).compiler_ir()
    computation2 = jax.jit(g).lower(2, 3).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    with compilation_cache_include_metadata_in_key(include_metadata):
      key1 = cache_key.get(
          computation1, devices, compile_options, backend,
          produce_original_cache_key=produce_original_cache_key)
      key2 = cache_key.get(
          computation2, devices, compile_options, backend,
          produce_original_cache_key=produce_original_cache_key)
    self.assertEqual(include_metadata, key1 != key2)

  @parameterized.parameters([False, True])
  def test_xla_flags(self, produce_original_cache_key):
    if jtu.is_device_tpu_v4():
      raise unittest.SkipTest("TODO(b/240151176)")

    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()

    orig_xla_flags = os.getenv("XLA_FLAGS")
    orig_argv = sys.argv
    try:
      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
      key1 = cache_key.get(
          computation, devices, compile_options, backend,
          produce_original_cache_key=produce_original_cache_key)
      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=1"
      key2 = cache_key.get(
          computation, devices, compile_options, backend,
          produce_original_cache_key=produce_original_cache_key)
      self.assertNotEqual(key1, key2)

      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
      key3 = cache_key.get(
          computation, devices, compile_options, backend,
          produce_original_cache_key=produce_original_cache_key)
      self.assertEqual(key1, key3)

      # Test flag in _xla_flags_to_exclude_from_cache_key
      os.environ["XLA_FLAGS"] = (
          "--xla_gpu_autotune_level=0 --xla_force_host_platform_device_count=8"
      )
      key4 = cache_key.get(
          computation, devices, compile_options, backend,
          produce_original_cache_key=produce_original_cache_key)
      self.assertEqual(key1, key4)

      # Test flags given on command line
      del os.environ["XLA_FLAGS"]
      sys.argv.append("--xla_gpu_autotune_level=0")
      key5 = cache_key.get(
          computation, devices, compile_options, backend,
          produce_original_cache_key=produce_original_cache_key)
      self.assertEqual(key1, key5)
      sys.argv.append("--xla_force_host_platform_device_count=8")
      self.assertEqual(key1, key5)

    finally:
      if orig_xla_flags is not None:
        os.environ["XLA_FLAGS"] = orig_xla_flags
      elif os.getenv("XLA_FLAGS") is not None:
        del os.environ["XLA_FLAGS"]
      sys.argv = orig_argv

  def create_new_debug_options(self, debug_options_obj):
    debug_options_obj.xla_cpu_enable_fast_math = False
    debug_options_obj.xla_cpu_fast_math_honor_infs = False
    debug_options_obj.xla_cpu_fast_math_honor_nans = False
    debug_options_obj.xla_cpu_fast_math_honor_division = False
    debug_options_obj.xla_cpu_fast_math_honor_functions = False
    debug_options_obj.xla_gpu_enable_fast_min_max = False
    debug_options_obj.xla_backend_optimization_level = random.randint(0, 10)
    debug_options_obj.xla_cpu_enable_xprof_traceme = False
    debug_options_obj.xla_llvm_disable_expensive_passes = False
    debug_options_obj.xla_test_all_input_layouts = False
    return debug_options_obj

  def filled_compile_options(self):
    compile_options = xla_client.CompileOptions()
    compile_options.num_replicas = 1
    compile_options.num_partitions = 1
    shape = xla_client.Shape.array_shape(np.dtype(np.float32), [2])
    shape_array = [shape, shape]
    compile_options.argument_layouts = shape_array
    compile_options.executable_build_options.result_layout = shape

    device_assignment = xla_client.DeviceAssignment.create(
        np.arange(4).reshape(2, 2)
    )
    compile_options.device_assignment = device_assignment
    compile_options.executable_build_options.device_assignment = (
        device_assignment
    )
    compile_options.executable_build_options.fdo_profile = b"test_profile"
    return compile_options

  def get_hashed_value(self, hash_function, hash_function_input):
    hash_obj = hashlib.sha256()
    hash_function(hash_obj, hash_function_input)
    return hash_obj.digest().hex()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
