# Copyright 2024 The JAX Authors.
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

import contextlib
from typing import Sequence

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import xla_extension_version  # pylint: disable=g-importing-member
from jax.experimental import colocated_python
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()


def _colocated_cpu_devices(
    devices: Sequence[jax.Device],
) -> Sequence[jax.Device]:
  """Returns CPU devices colocated with the given devices."""
  # TODO(hyeontaek): Use `colocated_python.colocated_cpu_devices(devices)` once
  # PjRt-IFRT prepares CPU devices by its own.
  cpu_backend_devices = jax.local_devices(backend="cpu")
  device_index_map = {device.id: i for i, device in enumerate(jax.devices())}
  return [
      cpu_backend_devices[device_index_map[device.id]] for device in devices
  ]


class ColocatedPythonTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if xla_extension_version < 290:
      self.skipTest("Requires xla_extension_version >= 290")
    # TODO(hyeontaek): Remove provisioning "cpu" backend devices once PjRt-IFRT
    # prepares CPU devices by its own.
    self._exit_stack = contextlib.ExitStack()
    self._exit_stack.enter_context(
        jtu.set_host_platform_device_count(len(jax.devices()))
    )

  def testSimpleFunction(self):
    @colocated_python.colocated_python
    def add_one(x):
      return x + 1

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])
    self.assertEqual(add_one(x), np.array(2))

  def testSimpleFunctioWithTree(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = [np.array(1), (np.array(2), {"v": np.array(3)})]
    x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))
    self.assertEqual(
        add_one(x), [np.array(2), (np.array(3), {"v": np.array(4)})]
    )

  def testEmptyInputFailsWithoutSpecialization(self):
    @colocated_python.colocated_python
    def make_zero():
      return jnp.array(0)

    with self.assertRaisesRegex(
        ValueError,
        "No devices found. colocated_python function without input arguments"
        " must be first specialized with devices.",
    ):
      _ = make_zero()

  def testEmptyInputWithDevicesSpecialization(self):
    @colocated_python.colocated_python
    def make_zero():
      return jnp.array(0)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    make_zero = make_zero.specialize(devices=cpu_devices[:1])

    self.assertEqual(int(make_zero()), np.array(0))

  def testInputPolymorphismWithoutOutSpecs(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])
    self.assertEqual(add_one(x), np.array(2))

    x = [np.array(1), (np.array(2), {"v": np.array(3)})]
    x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))
    self.assertEqual(
        add_one(x), [np.array(2), (np.array(3), {"v": np.array(4)})]
    )

  def testInputPolymorphismAllowedWithOutSpecsFn(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])
    add_one = add_one.specialize(out_specs_fn=lambda x: x)
    self.assertEqual(add_one(x), np.array(2))

    x = [np.array(1), (np.array(2), {"v": jnp.array(3)})]
    x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))
    self.assertEqual(
        add_one(x), [np.array(2), (np.array(3), {"v": np.array(4)})]
    )

  def testInputPolymorphismDisallowedWithOutSpecsSpecialization(self):
    @colocated_python.colocated_python
    def add_one(x):
      return jax.tree.map(lambda x: x + 1, x)

    cpu_devices = _colocated_cpu_devices(jax.local_devices())
    x = np.array(1)
    x = jax.device_put(x, cpu_devices[0])
    add_one = add_one.specialize(
        out_specs=jax.ShapeDtypeStruct(
            shape=x.shape, dtype=x.dtype, sharding=x.sharding
        )
    )
    self.assertEqual(add_one(x), np.array(2))

    x = [np.array(1), (np.array(2), {"v": jnp.array(3)})]
    x = jax.device_put(x, jax.sharding.SingleDeviceSharding(cpu_devices[0]))
    with self.assertRaisesRegex(ValueError, "Input specs do not match: "):
      _ = add_one(x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
