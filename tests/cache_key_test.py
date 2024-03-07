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
import sys
import unittest

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import cache_key
from jax._src import compiler
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.lib import xla_client


config.parse_flags_with_absl()


class CacheKeyTest(jtu.JaxTestCase):

  def test_serialized_compile_options(self):
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    hash1 = self.get_hashed_value(
        cache_key._hash_serialized_compile_options, compile_options
    )
    debug_options = compile_options.executable_build_options.debug_options
    debug_options.xla_force_host_platform_device_count = 2
    debug_options.xla_dump_to = "foo"
    debug_options.xla_dump_hlo_module_re = "bar"
    debug_options.xla_dump_hlo_pass_re = "baz"
    debug_options.xla_dump_hlo_as_text = True
    debug_options.xla_dump_hlo_as_proto = True
    debug_options.xla_dump_hlo_as_dot = True
    debug_options.xla_dump_hlo_as_url = True
    debug_options.xla_dump_hlo_as_html = True
    debug_options.xla_dump_fusion_visualization = True
    debug_options.xla_dump_hlo_snapshots = True
    debug_options.xla_dump_max_hlo_modules = True
    debug_options.xla_dump_module_metadata = True
    debug_options.xla_dump_compress_protos = True
    debug_options.xla_dump_hlo_as_long_text = True
    debug_options.xla_dump_disable_metadata = True
    debug_options.xla_dump_hlo_pipeline_re = "xyzzy"
    hash2 = self.get_hashed_value(
        cache_key._hash_serialized_compile_options, compile_options
    )
    self.assertEqual(hash1, hash2)

  @jtu.skip_on_devices("cpu")
  def test_hash_accelerator_devices(self):
    devices = np.array([[jax.local_devices()[0]]])

    dev_hash1 = self.get_hashed_value(cache_key._hash_devices, devices)
    dev_hash2 = self.get_hashed_value(cache_key._hash_devices, devices)
    self.assertEqual(dev_hash1, dev_hash2)

    acc_hash1 = self.get_hashed_value(
        cache_key._hash_accelerator_config, devices, xla_bridge.get_backend())
    acc_hash2 = self.get_hashed_value(
        cache_key._hash_accelerator_config, devices, xla_bridge.get_backend())
    self.assertEqual(acc_hash1, acc_hash2)

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

  def test_hash_string(self):
    hash1 = self.get_hashed_value(cache_key._hash_string, "foo")
    hash2 = self.get_hashed_value(cache_key._hash_string, "bar")
    hash3 = self.get_hashed_value(cache_key._hash_string, "bar")
    self.assertEqual(hash2, hash3)
    self.assertNotEqual(hash1, hash2)

  def test_same_key(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    self.assertEqual(
        cache_key.get(computation, devices, compile_options, backend),
        cache_key.get(computation, devices, compile_options, backend),
    )

  def test_different_key(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options_not_filled = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    compile_options_filled = self.filled_compile_options()
    backend = xla_bridge.get_backend()
    self.assertNotEqual(
        cache_key.get(
            computation, devices, compile_options_not_filled, backend
        ),
        cache_key.get(computation, devices, compile_options_filled, backend),
    )

  def test_custom_hook(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    original_custom_hook = cache_key.custom_hook
    cache_key.custom_hook = lambda: "hook1"
    key1 = cache_key.get(computation, devices, compile_options, backend)
    cache_key.custom_hook = lambda: "hook2"
    key2 = cache_key.get(computation, devices, compile_options, backend)
    cache_key.custom_hook = original_custom_hook
    self.assertNotEqual(key1, key2)

  def test_different_computations(self):
    computation1 = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    computation2 = jax.jit(lambda x, y: x * y).lower(2, 2).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    self.assertNotEqual(
        cache_key.get(computation1, devices, compile_options, backend),
        cache_key.get(computation2, devices, compile_options, backend),
    )

  def test_different_device_assignment(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options_1 = compiler.get_compile_options(
        num_replicas=1, num_partitions=1, device_assignment=np.array([[0]])
    )
    compile_options_2 = compiler.get_compile_options(
        num_replicas=1, num_partitions=1, device_assignment=np.array([[1]])
    )
    backend = xla_bridge.get_backend()
    hash_1 = cache_key.get(computation, devices, compile_options_1, backend)
    hash_2 = cache_key.get(computation, devices, compile_options_2, backend)
    if backend.platform == "gpu":
      self.assertEqual(hash_1, hash_2)
    else:
      self.assertNotEqual(hash_1, hash_2)

  @parameterized.parameters([False, True])
  def test_identical_computations_different_metadata(self, include_metadata):
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
    with config.compilation_cache_include_metadata_in_key(include_metadata):
      key1 = cache_key.get(computation1, devices, compile_options, backend)
      key2 = cache_key.get(computation2, devices, compile_options, backend)
    self.assertEqual(include_metadata, key1 != key2)

  def test_xla_flags(self):
    if jtu.is_device_tpu(version=4):
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
      key1 = cache_key.get(computation, devices, compile_options, backend)
      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=1"
      key2 = cache_key.get(computation, devices, compile_options, backend)
      self.assertNotEqual(key1, key2)

      os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
      key3 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key3)

      # Test flag in _xla_flags_to_exclude_from_cache_key
      os.environ["XLA_FLAGS"] = (
          "--xla_gpu_autotune_level=0 --xla_force_host_platform_device_count=8"
      )
      key4 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key4)

      # Test flags given on command line
      del os.environ["XLA_FLAGS"]
      sys.argv.append("--xla_gpu_autotune_level=0")
      key5 = cache_key.get(computation, devices, compile_options, backend)
      self.assertEqual(key1, key5)
      sys.argv.append("--xla_force_host_platform_device_count=8")
      self.assertEqual(key1, key5)

    finally:
      if orig_xla_flags is not None:
        os.environ["XLA_FLAGS"] = orig_xla_flags
      elif os.getenv("XLA_FLAGS") is not None:
        del os.environ["XLA_FLAGS"]
      sys.argv = orig_argv

  def test_libtpu_init_args(self):
    if jtu.is_device_tpu(version=4):
      raise unittest.SkipTest("TODO(b/240151176)")

    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()

    orig_libtpu_init_args = os.getenv("LIBTPU_INIT_ARGS")
    orig_argv = sys.argv
    try:
      os.environ["LIBTPU_INIT_ARGS"] = (
          "--xla_spmd_threshold_for_windowed_einsum_mib=0"
      )
      key1 = cache_key.get(computation, devices, compile_options, backend)
      os.environ["LIBTPU_INIT_ARGS"] = (
          "--xla_spmd_threshold_for_windowed_einsum_mib=1"
      )
      key2 = cache_key.get(computation, devices, compile_options, backend)
      self.assertNotEqual(key1, key2)

    finally:
      if orig_libtpu_init_args is not None:
        os.environ["LIBTPU_INIT_ARGS"] = orig_libtpu_init_args
      elif os.getenv("LIBTPU_INIT_ARGS") is not None:
        del os.environ["LIBTPU_INIT_ARGS"]
      sys.argv = orig_argv

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

  def get_hashed_value(
      self, hash_function, hash_function_input1, hash_function_input2=None):
    hash_obj = hashlib.sha256()
    if hash_function_input2 is not None:
      hash_function(hash_obj, hash_function_input1, hash_function_input2)
    else:
      hash_function(hash_obj, hash_function_input1)
    return hash_obj.digest().hex()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
