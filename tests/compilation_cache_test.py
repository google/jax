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

from __future__ import annotations

from collections import Counter
from functools import partial
import math
import platform
import unittest
from unittest import mock
from unittest import SkipTest
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import jit
from jax import lax
from jax import pmap
from jax._src import compilation_cache as cc
from jax._src import compiler
from jax._src import config
from jax._src import distributed
from jax._src import monitoring
from jax._src import path as pathlib
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.compilation_cache_interface import CacheInterface
from jax._src.lib import xla_client
from jax._src.maps import xmap
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
import numpy as np


config.parse_flags_with_absl()

FAKE_COMPILE_TIME = 10
_counts = Counter()  # Map event name to count


def setUpModule():
  monitoring.register_event_listener(increment_event_count)


def tearDownModule():
  monitoring._unregister_event_listener_by_callback(increment_event_count)


def increment_event_count(event):
  _counts[event] += 1


class InMemoryCache(CacheInterface):
    """An in-memory cache for testing purposes."""

    # not used, but required by `CacheInterface`
    _path = pathlib.Path()

    def __init__(self):
        self._cache: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._cache.get(key)

    def put(self, key: str, value: bytes) -> None:
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


def count_cache_items() -> int:
  return 0 if cc._cache is None else len(cc._cache)


def clear_cache() -> None:
  if cc._cache is not None:
    cc._cache.clear()


class CompilationCacheTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    cc.reset_cache()
    cc._cache = InMemoryCache()

  def tearDown(self):
    cc.reset_cache()
    super().tearDown()


@jtu.with_config(
    jax_enable_compilation_cache=True,
    jax_raise_persistent_cache_errors=True,
    jax_persistent_cache_min_compile_time_secs=0,
    jax_persistent_cache_min_entry_size_bytes=0,
)
class CompilationCacheTest(CompilationCacheTestCase):
  def setUp(self):
    super().setUp()
    supported_platforms = ["tpu", "gpu", "cpu"]

    if not jtu.test_device_matches(supported_platforms):
      raise SkipTest(
          "serialize executable only works on " + ",".join(supported_platforms)
      )

  def test_get_no_executable(self):
    computation = jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir()
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    key = cc.get_cache_key(computation, devices, compile_options, backend)
    executable, compile_time = cc.get_executable_and_time(
        key, compile_options, backend)
    self.assertIsNone(executable)
    self.assertIsNone(compile_time)

  def test_diff_executables(self):
    computation1 = str(jax.jit(lambda x, y: x + y).lower(1, 1).compiler_ir())
    computation2 = str(jax.jit(lambda x, y: x * y).lower(2, 2).compiler_ir())
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    executable1 = backend.compile(computation1, compile_options)
    executable2 = backend.compile(computation2, compile_options)
    cc.put_executable_and_time(
        "key1", "computation1", executable1, backend, FAKE_COMPILE_TIME)
    cc.put_executable_and_time(
        "key2", "computation2", executable2, backend, FAKE_COMPILE_TIME)
    self.assertNotEqual(
        cc.get_executable_and_time("key1", compile_options, backend)[0],
        cc.get_executable_and_time("key2", compile_options, backend)[0]
    )

  def test_put_executable(self):
    computation = (
        jax.jit(lambda x, y: x + y)
        .lower(np.int32(1), np.int32(1))
        .compiler_ir()
    )
    devices = np.array([[jax.local_devices()[0]]])
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1
    )
    backend = xla_bridge.get_backend()
    executable = backend.compile(str(computation), compile_options)
    key = cc.get_cache_key(computation, devices, compile_options, backend)
    cc.put_executable_and_time(
        key, "alambda", executable, backend, FAKE_COMPILE_TIME)
    executable_retrieved, compile_time_retrieved = cc.get_executable_and_time(
        key, compile_options, backend)
    inputs_to_executable = (
        np.array(1, dtype=np.int32),
        np.array(2, dtype=np.int32),
    )
    expected = xla_client.execute_with_python_values(
        executable, inputs_to_executable, backend
    )
    actual = xla_client.execute_with_python_values(
        executable_retrieved, inputs_to_executable, backend
    )
    self.assertEqual(expected, actual)
    self.assertEqual(FAKE_COMPILE_TIME, compile_time_retrieved)

  def test_pmap(self):
    f = pmap(lambda x: x - lax.psum(x, "i"), axis_name="i")
    x = np.arange(jax.device_count(), dtype=np.int64)
    f(x)
    self.assertEqual(count_cache_items(), 1)
    x = np.arange(jax.device_count(), dtype=np.float32)
    f(x)
    self.assertEqual(count_cache_items(), 2)
    # TODO: create a test for calling pmap with the same input more than once

  def test_jit(self):
    f = jit(lambda x: x * x)
    f(1)
    self.assertEqual(count_cache_items(), 1)
    f(1.0)
    self.assertEqual(count_cache_items(), 2)

  def test_xla_autofdo_profile_version(self):
    original_profile_version = config.jax_xla_profile_version.value
    with config.jax_xla_profile_version(original_profile_version + 1):
      f = jit(lambda x: x * x)
      f(1)
      self.assertEqual(count_cache_items(), 1)
      # Clear the cache directory, then update the profile version and execute
      # again. The in-memory caches should be invalidated and a new persistent
      # cache entry created.
      clear_cache()
      with config.jax_xla_profile_version(original_profile_version + 2):
        f(1)
        self.assertEqual(count_cache_items(), 1)

  @jtu.with_mesh([("x", 2)])
  def test_pjit(self):
    @partial(pjit, in_shardings=(P("x"), P("x")), out_shardings=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(math.prod(shape), dtype=np.int64).reshape(shape)
    f(x, x + 1)
    self.assertEqual(count_cache_items(), 1)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    f(x, x + 1)
    self.assertEqual(count_cache_items(), 2)

  @jtu.with_mesh([("x", 2)])
  def test_xmap(self):
    def f(x):
      return x * 2

    devices = np.array(jax.local_devices()[:2])
    if devices.size < 2:
      raise SkipTest("Test requires 2 devices")
    x = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
    xmap(
        f, in_axes=["a", ...], out_axes=["a", ...], axis_resources={"a": "x"}
    )(x)
    self.assertEqual(count_cache_items(), 1)
    x = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    xmap(
        f, in_axes=["a", ...], out_axes=["a", ...], axis_resources={"a": "x"}
    )(x)
    self.assertEqual(count_cache_items(), 2)

  def test_cache_write_warning(self):
    f = jit(lambda x: x * x)

    backend = xla_bridge.get_backend()
    with (
      config.raise_persistent_cache_errors(False),
      mock.patch.object(cc._get_cache(backend).__class__, "put") as mock_put,
      warnings.catch_warnings(record=True) as w,
    ):
      mock_put.side_effect = RuntimeError("test error")
      self.assertEqual(f(2).item(), 4)
      if len(w) != 1:
        print("Warnings:", [str(w_) for w_ in w], flush=True)
      self.assertLen(w, 1)
      self.assertIn(
          (
              "Error writing persistent compilation cache entry "
              "for 'jit__lambda_': RuntimeError: test error"
          ),
          str(w[0].message),
      )

  def test_cache_read_warning(self):
    f = jit(lambda x: x * x)

    backend = xla_bridge.get_backend()
    with (
      config.raise_persistent_cache_errors(False),
      mock.patch.object(cc._get_cache(backend).__class__, "get") as mock_get,
      warnings.catch_warnings(record=True) as w,
    ):
      mock_get.side_effect = RuntimeError("test error")
      # Calling assertEqual with the jitted f will generate two PJIT
      # executables: Equal and the lambda function itself.
      self.assertEqual(f(2).item(), 4)
      if len(w) != 1:
        print("Warnings:", [str(w_) for w_ in w], flush=True)
      self.assertLen(w, 1)
      self.assertIn(
          (
              "Error reading persistent compilation cache entry "
              "for 'jit__lambda_': RuntimeError: test error"
          ),
          str(w[0].message),
      )

  def test_min_entry_size(self):
    with (
      config.persistent_cache_min_compile_time_secs(0),
      config.persistent_cache_min_entry_size_bytes(1048576),  # 1MiB
    ):
      jit(lambda x: x + 1)(1)
      self.assertEqual(count_cache_items(), 0)

  def test_min_compile_time(self):
    with (
      config.persistent_cache_min_compile_time_secs(2),
      config.persistent_cache_min_entry_size_bytes(0),
    ):
      # Mock time to progress in small intervals so compilation time is small.
      with mock.patch("time.monotonic", side_effect=np.arange(0, 10, 0.1)):
        jit(lambda x: x + 1)(1)
        self.assertEqual(count_cache_items(), 0)

      # Mock time to progress in large intervals so compilation time is large.
      with mock.patch("time.monotonic", side_effect=np.arange(0, 100, 10)):
        jit(lambda x: x + 2)(1)
        self.assertEqual(count_cache_items(), 1)

  # This is perhaps related to mocking time.monotonic?
  @unittest.skipIf(platform.system() == "Windows", "Test fails on Windows")
  def test_cache_saving_metric(self):
    with (
      config.persistent_cache_min_compile_time_secs(2),
      config.persistent_cache_min_entry_size_bytes(0),
    ):
      durations = Counter()  # Map metric name to time duration.
      def append_metric_duration(metric, duration):
        durations[metric] += duration

      with jtu.register_event_duration_listener(append_metric_duration):

        # Mock time to create a short compilation time, no cache saved, no cache
        # hit, no metric recorded.
        with mock.patch("time.monotonic", side_effect=np.arange(0, 1, 0.1)):
          jit(lambda x: x + 1)(1)

        jit(lambda x: x + 1)(1)
        self.assertNotIn(
            "/jax/compilation_cache/cache_retrieval_time_sec", durations)
        self.assertNotIn(
            "/jax/compilation_cache/compile_time_saved_sec", durations)

        # Mock time to create a long compilation time, metrics incremented with
        # a cache hit.
        with mock.patch("time.monotonic", side_effect=np.arange(0, 100, 10)):
          jit(lambda x: x + 2)(1)

        jit(lambda x: x + 2)(1)
        self.assertGreater(
            durations["/jax/compilation_cache/cache_retrieval_time_sec"], 0)
        self.assertGreater(
            durations["/jax/compilation_cache/compile_time_saved_sec"], 0)

  def test_task_using_cache_metric(self):
    count_before_first_use = _counts[
        "/jax/compilation_cache/tasks_using_cache"]
    jit(lambda x: x + 1)(1)
    count_after_first_use = _counts[
        "/jax/compilation_cache/tasks_using_cache"]
    self.assertEqual(count_after_first_use, count_before_first_use + 1)

    # Verify that the count is incremented only once per task.
    jit(lambda x: x + 3)(3)
    count_after_second_use = _counts[
        "/jax/compilation_cache/tasks_using_cache"]
    self.assertEqual(count_after_second_use, count_after_first_use)

  def test_compile_requests_use_cache_metric(self):
    previous_counts = Counter(_counts)
    jit(lambda x: x + 1)(1)
    jit(lambda x: x + 2)(1)
    jit(lambda x: x + 1)(1)

    self.assertEqual(
        _counts["/jax/compilation_cache/compile_requests_use_cache"]
        - previous_counts["/jax/compilation_cache/compile_requests_use_cache"],
        3)

  @parameterized.parameters(0, 1048576)  # 0 byte, 1 MiB
  def test_cache_misses_metric(self, min_entry_size):
    previous_counts = Counter(_counts)
    with (
      config.persistent_cache_min_compile_time_secs(2),
      config.persistent_cache_min_entry_size_bytes(min_entry_size),
    ):
      # Mock time to create a long compilation time and make cache misses.
      with mock.patch("time.monotonic", side_effect=np.arange(0, 100, 10)):
        jit(lambda x: x + 1)(1)
        jit(lambda x: x + 2)(1)

    if min_entry_size <= 0:
      self.assertEqual(
          _counts["/jax/compilation_cache/cache_misses"]
          - previous_counts["/jax/compilation_cache/cache_misses"],
          2)
    else:
      self.assertEqual(
          _counts["/jax/compilation_cache/cache_misses"]
          - previous_counts["/jax/compilation_cache/cache_misses"],
          0)

  def test_cache_hits_metric(self):
    previous_counts = Counter(_counts)
    with (
      config.persistent_cache_min_compile_time_secs(2),
      config.persistent_cache_min_entry_size_bytes(0),
    ):
      # Mock time to create a long compilation time, cache saved.
      with mock.patch("time.monotonic", side_effect=np.arange(0, 100, 10)):
        jit(lambda x: x + 1)(1)
      jit(lambda x: x + 1)(1)

    self.assertEqual(
        _counts["/jax/compilation_cache/cache_hits"]
        - previous_counts["/jax/compilation_cache/cache_hits"],
        1)

  @parameterized.parameters(0, 1)
  def test_cache_write_with_process_restriction(self, process_id):
    with (
      config.persistent_cache_min_compile_time_secs(0),
      config.persistent_cache_min_entry_size_bytes(0),
      mock.patch.object(distributed.global_state, "process_id", process_id),
    ):
      jit(lambda x: x + 1)(1)

      files_in_directory = count_cache_items()
      if process_id == 0:
        self.assertEqual(files_in_directory, 1)
      elif process_id == 1:
        self.assertEqual(files_in_directory, 0)

  def test_backend_serialization_deserialization(self):
    backend = xla_bridge.get_backend()
    executable = (
        jax.jit(lambda x, y: x + y)
        .lower(np.array(1.), np.array(1.))
        .compile()
        .runtime_executable()
    )
    serialized_executable = backend.serialize_executable(executable)
    deserialized_executable = backend.deserialize_executable(
        serialized_executable, None)
    self.assertEqual(
        executable.fingerprint, deserialized_executable.fingerprint)


@jtu.with_config(
    jax_enable_compilation_cache=False,
    jax_persistent_cache_min_compile_time_secs=0,
    jax_persistent_cache_min_entry_size_bytes=0,
)
class CompilationCacheDisabledTest(CompilationCacheTestCase):
  # If the cache is disabled, there should be no files in the cache directory.
  # A call to set_cache_dir() does not affect this.
  def test_jit(self):
    # Sequence of flag settings for config.jax_enable_compilation_cache:
    # 1. Flag is disabled by @jtu.with_config() above.
    # 2. Flag is enabled by JaxTestCase for some test configs
    #    (see test_util.py).
    # We need the flag disabled for this test, so disable it below.
    with config.enable_compilation_cache(False):
      f = jit(lambda x: x * x)
      f(1)
      self.assertEqual(count_cache_items(), 0)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
