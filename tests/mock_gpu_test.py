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
import math

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

jax.config.parse_flags_with_absl()
NUM_SHARDS = 4


@jtu.with_config(mock_num_gpu_processes=NUM_SHARDS)
class MockGPUTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Mocking devices only works on the GPU backend.")
    super().setUp()

  @jtu.skip_under_pytest("Test must run in an isolated process")
  def testMockDeviceCount(self):
    self.assertEqual(jax.device_count(), jax.local_device_count() * NUM_SHARDS)

  @jtu.skip_under_pytest("Test must run in an isolated process")
  def testMockWithSharding(self):
    mesh = jax.sharding.Mesh(jax.devices(), ('x',))
    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, P('x',)),
        out_shardings=NamedSharding(mesh, P('x',)),
    )
    def f(x, y):
      z = x @ y
      return z @ y

    shape = (1024, 1024)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1
    f_lowered = f.lower(x, y)
    hlo = f_lowered.compiler_ir()

    mocked_count = NUM_SHARDS * jax.local_device_count()
    self.assertIn(
        f'sharding = "{{devices=[{mocked_count},1]<=[{mocked_count}]}}"',
        str(hlo)
    )

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
