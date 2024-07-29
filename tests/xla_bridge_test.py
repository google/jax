# Copyright 2019 The JAX Authors.
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

import os
import platform
import time
import warnings

from absl import logging
from absl.testing import absltest

from jax import version
from jax._src import compiler
from jax._src import config
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.interpreters import xla
from jax._src.lib import xla_client as xc

config.parse_flags_with_absl()

mock = absltest.mock


class XlaBridgeTest(jtu.JaxTestCase):

  def test_set_device_assignment_no_partition(self):
    compile_options = compiler.get_compile_options(
        num_replicas=4, num_partitions=1, device_assignment=[0, 1, 2, 3])
    expected_device_assignment = ("Computations: 1 Replicas: 4\nComputation 0: "
                                  "0 1 2 3 \n")
    self.assertEqual(compile_options.device_assignment.__repr__(),
                     expected_device_assignment)

  def test_set_device_assignment_with_partition(self):
    compile_options = compiler.get_compile_options(
        num_replicas=2, num_partitions=2, device_assignment=[[0, 1], [2, 3]])
    expected_device_assignment = ("Computations: 2 Replicas: 2\nComputation 0: "
                                  "0 2 \nComputation 1: 1 3 \n")
    self.assertEqual(compile_options.device_assignment.__repr__(),
                     expected_device_assignment)

  def test_set_fdo_profile(self):
    compile_options = compiler.get_compile_options(
        num_replicas=1, num_partitions=1, fdo_profile=b"test_profile"
    )
    self.assertEqual(
        compile_options.executable_build_options.fdo_profile, b"test_profile")

  def test_autofdo_profile(self):

    class _DummyBackend:
      platform: str = "tpu"

    # --jax_xla_profile_version takes precedence.
    jax_flag_profile = 1
    another_profile = 2
    with config.jax_xla_profile_version(jax_flag_profile):
      with mock.patch.object(compiler, "get_latest_profile_version",
                             side_effect=lambda _: another_profile):
        self.assertEqual(
            compiler.get_compile_options(
                num_replicas=3, num_partitions=4, backend=_DummyBackend(),
            ).profile_version,
            jax_flag_profile,
        )

    # Use whatever non-zero value the function get_latest_profile_version
    # returns if --jax_xla_profile_version is not set.
    profile_version = 1
    with mock.patch.object(compiler, "get_latest_profile_version",
                           side_effect=lambda _: profile_version):
      self.assertEqual(
          compiler.get_compile_options(
              num_replicas=3, num_partitions=4, backend=_DummyBackend(),
          ).profile_version,
          profile_version,
      )

    # If the function returns 0, something is wrong, so expect that we set
    # profile_version to -1 instead to ensure that no attempt is made to
    # retrieve the latest profile later.
    error_return = 0
    no_profile_dont_retrieve = -1
    with mock.patch.object(compiler, "get_latest_profile_version",
                           side_effect=lambda _: error_return):
      self.assertEqual(
          compiler.get_compile_options(
              num_replicas=3, num_partitions=4, backend=_DummyBackend(),
          ).profile_version,
          no_profile_dont_retrieve,
      )

  def test_deterministic_serialization(self):
    c1 = compiler.get_compile_options(
        num_replicas=2,
        num_partitions=3,
        env_options_overrides={"1": "1", "2": "2"},
    )
    c2 = compiler.get_compile_options(
        num_replicas=2,
        num_partitions=3,
        env_options_overrides={"2": "2", "1": "1"},  # order changed
    )
    c1str = c1.SerializeAsString()

    # Idempotence.
    self.assertEqual(c1str, c1.SerializeAsString())
    # Map order does not matter.
    self.assertEqual(c1str, c2.SerializeAsString())

  def test_parameter_replication_default(self):
    c = xc.XlaBuilder("test")
    _ = xla.parameter(c, 0, xc.Shape.array_shape(xc.PrimitiveType.F32, ()))
    built_c = c.Build()
    assert "replication" not in built_c.as_hlo_text()

  def test_parameter_replication(self):
    c = xc.XlaBuilder("test")
    _ = xla.parameter(c, 0, xc.Shape.array_shape(xc.PrimitiveType.F32, ()), "",
                     False)
    built_c = c.Build()
    assert "parameter_replication={false}" in built_c.as_hlo_text()

  def test_local_devices(self):
    self.assertNotEmpty(xb.local_devices())
    with self.assertRaisesRegex(ValueError, "Unknown process_index 100"):
      xb.local_devices(100)
    with self.assertRaisesRegex(RuntimeError, "Unknown backend foo"):
      xb.local_devices(backend="foo")

  def test_timer_tpu_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      def _mock_tpu_client_with_options(library_path=None,
                                        options=None,
                                        distributed_client=None,
                                        node_id=None,
                                        num_nodes=None):
        time_to_wait = 5
        start = time.time()
        while not w:
          if time.time() - start > time_to_wait:
            raise ValueError(
                "This test should not hang for more than "
                f"{time_to_wait} seconds.")
          time.sleep(0.1)

        self.assertLen(w, 1)
        msg = str(w[-1].message)
        self.assertIn("Did you run your code on all TPU hosts?", msg)

      def _mock_tpu_client(library_path=None):
        _mock_tpu_client_with_options(library_path=library_path, options=None)

      with mock.patch.object(xc, "make_tpu_client",
                            side_effect=_mock_tpu_client_with_options):
        xb.tpu_client_timer_callback(0.01)

  def test_register_plugin(self):
    with self.assertLogs(level="WARNING") as log_output:
      with mock.patch.object(xc, "load_pjrt_plugin_dynamically", autospec=True):
        if platform.system() == "Windows":
          os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = (
              "name1;path1,name2;path2,name3"
          )
        else:
          os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = (
              "name1:path1,name2:path2,name3"
          )
        with mock.patch.object(
            xc.profiler, "register_plugin_profiler", autospec=True
        ):
          xb.register_pjrt_plugin_factories_from_env()
    registration = xb._backend_factories["name1"]
    with mock.patch.object(xc, "make_c_api_client", autospec=True) as mock_make:
      with mock.patch.object(
          xc, "pjrt_plugin_initialized", autospec=True, return_vale=True
      ):
        with mock.patch.object(xc, "initialize_pjrt_plugin", autospec=True):
          registration.factory()

    self.assertRegex(
        log_output[1][0],
        r"invalid value name3 in env var PJRT_NAMES_AND_LIBRARY_PATHS"
        r" name1.path1,name2.path2,name3",
    )
    self.assertIn("name1", xb._backend_factories)
    self.assertIn("name2", xb._backend_factories)
    self.assertEqual(registration.priority, 400)
    self.assertTrue(registration.experimental)

    options = {}
    if xb.get_backend().platform == 'tpu':
      options["ml_framework_name"] = "JAX"
      options["ml_framework_version"] = version.__version__
    mock_make.assert_called_once_with("name1", options, None)

  def test_register_plugin_with_config(self):
    test_json_file_path = os.path.join(
        os.path.dirname(__file__), "testdata/example_pjrt_plugin_config.json"
    )
    os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = (
        f"name1;{test_json_file_path}"
        if platform.system() == "Windows"
        else f"name1:{test_json_file_path}"
    )
    with mock.patch.object(xc, "load_pjrt_plugin_dynamically", autospec=True):
      with mock.patch.object(
          xc.profiler, "register_plugin_profiler", autospec=True
      ):
        xb.register_pjrt_plugin_factories_from_env()
    registration = xb._backend_factories["name1"]
    with mock.patch.object(xc, "make_c_api_client", autospec=True) as mock_make:
      with mock.patch.object(
          xc, "pjrt_plugin_initialized", autospec=True, return_vale=True
      ):
        with mock.patch.object(xc, "initialize_pjrt_plugin", autospec=True):
          registration.factory()

    self.assertIn("name1", xb._backend_factories)
    self.assertEqual(registration.priority, 400)
    self.assertTrue(registration.experimental)

    # The expectation is specified in example_pjrt_plugin_config.json.
    options = {
        "int_option": 64,
        "int_list_option": [32, 64],
        "string_option": "string",
        "float_option": 1.0,
        }
    if xb.get_backend().platform == 'tpu':
      options["ml_framework_name"] = "JAX"
      options["ml_framework_version"] = version.__version__

    mock_make.assert_called_once_with("name1", options, None)


class GetBackendTest(jtu.JaxTestCase):

  class _DummyBackend:

    def __init__(self, platform, device_count):
      self.platform = platform
      self._device_count = device_count

    def device_count(self):
      return self._device_count

    def process_index(self):
      return 0

    def local_devices(self):
      return []

  def _register_factory(self, platform: str, priority, device_count=1,
                        assert_used_at_most_once=False, experimental=False):
    if assert_used_at_most_once:
      used = []
    def factory():
      if assert_used_at_most_once:
        if used:
          # We need to fail aggressively here since exceptions are caught by
          # the caller and suppressed.
          logging.fatal("Backend factory for %s was called more than once",
                        platform)
        else:
          used.append(True)
      return self._DummyBackend(platform, device_count)

    xb.register_backend_factory(platform, factory, priority=priority,
                                fail_quietly=False, experimental=experimental)

  def setUp(self):
    super().setUp()
    self._orig_factories = xb._backend_factories
    xb._backend_factories = {}
    self.enter_context(config.jax_platforms(""))
    self._save_backend_state()
    self._reset_backend_state()

    # get_backend logic assumes CPU platform is always present.
    self._register_factory("cpu", 0)

  def tearDown(self):
    super().tearDown()
    xb._backend_factories = self._orig_factories
    self._restore_backend_state()

  def _save_backend_state(self):
    self._orig_backends = xb._backends
    self._orig_backend_errors = xb._backend_errors
    self._orig_default_backend = xb._default_backend

  def _reset_backend_state(self):
    xb._backends = {}
    xb._backend_errors = {}
    xb._default_backend = None
    xb.get_backend.cache_clear()

  def _restore_backend_state(self):
    xb._backends = self._orig_backends
    xb._backend_errors = self._orig_backend_errors
    xb._default_backend = self._orig_default_backend
    xb.get_backend.cache_clear()

  def test_default(self):
    self._register_factory("platform_A", 20)
    self._register_factory("platform_B", 10)

    backend = xb.get_backend()
    self.assertEqual(backend.platform, "platform_A")
    # All backends initialized.
    self.assertEqual(len(xb._backends), len(xb._backend_factories))

  def test_specific_platform(self):
    self._register_factory("platform_A", 20)
    self._register_factory("platform_B", 10)

    backend = xb.get_backend("platform_B")
    self.assertEqual(backend.platform, "platform_B")
    # All backends initialized.
    self.assertEqual(len(xb._backends), len(xb._backend_factories))

  def test_unknown_backend_error(self):
    with self.assertRaisesRegex(RuntimeError, "Unknown backend foo"):
      xb.get_backend("foo")

  def test_backend_init_error(self):
    def factory():
      raise RuntimeError("I'm not a real backend")

    xb.register_backend_factory("error", factory, priority=10,
                                fail_quietly=False)

    with self.assertRaisesRegex(
      RuntimeError,
      "Unable to initialize backend 'error': I'm not a real backend"
    ):
      xb.get_backend("error")


  def test_no_devices(self):
    self._register_factory("no_devices", -10, device_count=0)
    with self.assertRaisesRegex(
        RuntimeError,
        "Unable to initialize backend 'no_devices': "
        "Backend 'no_devices' provides no devices."):
      xb.get_backend("no_devices")

  def test_factory_returns_none(self):
    xb.register_backend_factory("none", lambda: None, priority=10,
                                fail_quietly=False)
    with self.assertRaisesRegex(
        RuntimeError,
        "Unable to initialize backend 'none': "
        "Could not initialize backend 'none'"):
      xb.get_backend("none")

  def cpu_fallback_warning(self):
    with self.assertWarnsRegex(UserWarning, "No GPU/TPU found, falling back to CPU"):
      xb.get_backend()

  def test_jax_platforms_flag(self):
    self._register_factory("platform_A", 20, assert_used_at_most_once=True)
    self._register_factory("platform_B", 10, assert_used_at_most_once=True)

    with config.jax_platforms("cpu,platform_A"):
      backend = xb.get_backend()
      self.assertEqual(backend.platform, "cpu")
      # Only specified backends initialized.
      self.assertEqual(len(xb._backends), 2)

      backend = xb.get_backend("platform_A")
      self.assertEqual(backend.platform, "platform_A")

      with self.assertRaisesRegex(RuntimeError, "Unknown backend platform_B"):
        backend = xb.get_backend("platform_B")

  def test_experimental_warning(self):
    self._register_factory("platform_A", 20, experimental=True)

    with self.assertLogs("jax._src.xla_bridge", level="WARNING") as logs:
      _ = xb.get_backend()
    self.assertIn(
      "WARNING:jax._src.xla_bridge:Platform 'platform_A' is experimental and "
      "not all JAX functionality may be correctly supported!",
      logs.output
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
