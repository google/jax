# Copyright 2022 The JAX Authors.
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
"""Tests for transfer guards."""

import contextlib
import pickle

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp

jax.config.parse_flags_with_absl()


def _host_to_device_funcs():
  """Generates host-to-device transfer functions."""
  return [
      # (function name, is an explicit transfer?, function)
      ("host_to_device_jax_device_put", True,
       lambda: jax.device_put(np.ones(10))),
      ("host_to_device_jax_jit", False, lambda: jax.jit(lambda x: x)
       (np.ones(1))),
      ("host_to_device_jnp_one", False, lambda: jnp.ones(1)),
  ]


def _device_to_device_funcs():
  """Generates device-to-device transfer functions."""
  if len(jax.local_devices()) < 2:
    # device-to-device tests require at least 2 devices.
    return []

  with jax.transfer_guard_host_to_device("allow"):
    device_arrays = [jnp.ones(1) for _ in range(2)]
  return [
      # (function name, is an explicit transfer?, function)
      ("device_to_device_jax_device_put", True,
       lambda: jax.device_put(device_arrays[0], device=jax.local_devices()[1])),
      ("device_to_device_jax_jit", False,
       lambda: jax.jit(lambda x: x, device=jax.local_devices()[1])
       (device_arrays[1])),
  ]


def _device_to_host_funcs():
  """Generates device-to-host transfer functions."""
  if jax.default_backend() == "cpu":
    # device-to-host does not incur transfer on the CPU backend.
    return []

  with jax.transfer_guard_host_to_device("allow"):
    device_arrays = [jnp.ones(1) for _ in range(6)]
  return [
      # (function name, is an explicit transfer?, function)
      ("device_to_host_jax_device_get", True,
       lambda: jax.device_get(device_arrays[0])),
      ("device_to_host_np_asarray", False,
       lambda: np.asarray(device_arrays[1])),
      ("device_to_host_copy_to_host_async", False,
       lambda: device_arrays[2].copy_to_host_async()),
      ("device_to_host_np_add", False, lambda: np.add(device_arrays[3], 1)),
      ("device_to_host_str", False, lambda: str(device_arrays[4])),
      ("device_to_host_pickle_dumps", False,
       lambda: pickle.dumps(device_arrays[5])),
  ]


def _all_funcs():
  """Generates all transfer functions."""
  return (_host_to_device_funcs() + _device_to_device_funcs() +
          _device_to_host_funcs())


# List of test parameters shared by multiple tests.
_COMMON_TEST_PARAMETERS = [
    ("host_to_device", _host_to_device_funcs,
     jax.transfer_guard_host_to_device),
    ("device_to_device", _device_to_device_funcs,
     jax.transfer_guard_device_to_device),
    ("device_to_host", _device_to_host_funcs,
     jax.transfer_guard_device_to_host),
    ("all", _all_funcs, jax.transfer_guard),
]


# TransferGuardTest disables `--jax_enable_checks` because it
# can prematurely fetch the value of device arrays and make
# device-to-host tests to incur no transfers unexpectedly.
@jtu.with_config(jax_enable_checks=False)
class TransferGuardTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    # Nearly all test methods use the deprecated device argument to JIT.
    self.enter_context(jtu.ignore_warning(category=DeprecationWarning,
                                          message="backend and device argument"))

  @contextlib.contextmanager
  def assertAllows(self, func_name):
    """Asserts that a transfer in the context is allowed."""
    try:
      yield
    except Exception as e:  # pylint: disable=broad-except
      raise RuntimeError(
          f"Expected a transfer to be allowed while running: {func_name}"
      ) from e

  @contextlib.contextmanager
  def assertLogs(self, func_name):
    """Asserts that a transfer in the context is logged and allowed."""
    # Only check if the transfer is allowed until Abseil provides an interface
    # to capture logs.
    with self.assertAllows(func_name):
      yield

  @contextlib.contextmanager
  def assertDisallows(self, func_name):
    """Asserts that a transfer in the context is disallowed."""
    try:
      with self.assertRaises(Exception):
        yield
    except Exception as e:  # pylint: disable=broad-except
      raise RuntimeError(
          f"Expected a transfer to be disallowed while running: {func_name}"
      ) from e

  def test_simple(self):
    """Simple transfer guard tests."""
    with jax.transfer_guard("allow"):
      with self.assertAllows("host_to_device_jnp_ones"):
        jnp.ones(1)
    with jax.transfer_guard("log"):
      with self.assertLogs("host_to_device_jnp_ones"):
        jnp.ones(1)
    with jax.transfer_guard("disallow"):
      with self.assertDisallows("host_to_device_jnp_ones"):
        jnp.ones(1)

  def test_nesting(self):
    with jax.transfer_guard("disallow"):
      with jax.transfer_guard("allow"):
        with self.assertAllows("host_to_device_jnp_ones"):
          jnp.ones(1)
      with self.assertDisallows("host_to_device_jnp_ones"):
        jnp.ones(1)

  def test_mixed_nesting(self):
    with jax.transfer_guard_host_to_device("disallow"):
      with jax.transfer_guard("allow"):
        with self.assertAllows("host_to_device_jnp_ones"):
          jnp.ones(1)
      with self.assertDisallows("host_to_device_jnp_ones"):
        jnp.ones(1)

    with jax.transfer_guard("disallow"):
      with jax.transfer_guard_host_to_device("allow"):
        with self.assertAllows("host_to_device_jnp_ones"):
          jnp.ones(1)
      with self.assertDisallows("host_to_device_jnp_ones"):
        jnp.ones(1)

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_allow_by_default(self, func_generator, _):
    for func_name, _, func in func_generator():
      with self.assertAllows(func_name):
        func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_allow(self, func_generator, jax_transfer_guard):
    for func_name, _, func in func_generator():
      with jax_transfer_guard("allow"):
        with self.assertAllows(func_name):
          func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_log(self, func_generator, jax_transfer_guard):
    for func_name, explicit, func in func_generator():
      with jax_transfer_guard("log"):
        if explicit:
          with self.assertAllows(func_name):
            func()
        else:
          with self.assertLogs(func_name):
            func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_disallow(self, func_generator, jax_transfer_guard):
    for func_name, explicit, func in func_generator():
      with jax_transfer_guard("disallow"):
        if explicit:
          with self.assertAllows(func_name):
            func()
        else:
          with self.assertDisallows(func_name):
            func()

  @parameterized.named_parameters(
      ("device_to_host", _device_to_host_funcs,
       jax.transfer_guard_device_to_host),
      ("all", _device_to_host_funcs, jax.transfer_guard),
  )
  def test_disallow_ignores_arrays_on_cpu(self, func_generator,
                                          jax_transfer_guard):
    for func_name, _, func in func_generator():
      with jax_transfer_guard("allow"):
        # Transfer the device array to host.
        func()
      with jax_transfer_guard("disallow"):
        with self.assertAllows(func_name):
          # No error because the array has a value on host and no new transfer
          # will occur.
          func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_log_explicit(self, func_generator, jax_transfer_guard):
    for func_name, _, func in func_generator():
      with jax_transfer_guard("log_explicit"):
        with self.assertLogs(func_name):
          func()

  @parameterized.named_parameters(*_COMMON_TEST_PARAMETERS)
  def test_disallow_explicit(self, func_generator, jax_transfer_guard):
    for func_name, _, func in func_generator():
      with jax_transfer_guard("disallow_explicit"):
        with self.assertDisallows(func_name):
          func()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
