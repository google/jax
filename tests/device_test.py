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

from absl.testing import absltest
import jax
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class DeviceTest(jtu.JaxTestCase):

  def test_repr(self):
    device = jax.devices()[0]

    # TODO(pobudzey): Add a test for rocm devices when available.
    if jtu.is_device_cuda():
      self.assertEqual(device.platform, 'gpu')
      self.assertEqual(repr(device), 'CudaDevice(id=0)')
    elif jtu.test_device_matches(['tpu']):
      self.assertEqual(device.platform, 'tpu')
      self.assertEqual(
          repr(device),
          'TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)',
      )
    elif jtu.test_device_matches(['cpu']):
      self.assertEqual(device.platform, 'cpu')
      self.assertEqual(repr(device), 'CpuDevice(id=0)')

  def test_str(self):
    device = jax.devices()[0]

    # TODO(pobudzey): Add a test for rocm devices when available.
    if jtu.is_device_cuda():
      self.assertEqual(str(device), 'cuda:0')
    elif jtu.test_device_matches(['tpu']):
      self.assertEqual(str(device), 'TPU_0(process=0,(0,0,0,0))')
    elif jtu.test_device_matches(['cpu']):
      self.assertEqual(str(device), 'TFRT_CPU_0')


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
