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
NUM_SHARDS = 16


@jtu.with_config(use_mock_gpu_client=True, mock_num_gpus=NUM_SHARDS)
class MockGPUTest(jtu.JaxTestCase):

  def testMockWithSharding(self):
    mesh = jtu.create_global_mesh((NUM_SHARDS,), ('x',))
    @partial(
        jax.jit,
        in_shardings=NamedSharding(mesh, P('x',)),
        out_shardings=NamedSharding(mesh, P('x',)),
    )
    def f(x, y):
      z = x @ y
      return z @ y

    shape = (64, 64)
    x = jnp.arange(math.prod(shape)).reshape(shape).astype(np.float32)
    y = x + 1
    f_lowered = f.lower(x, y)
    hlo = f_lowered.compiler_ir()
    self.assertIn('sharding = "{devices=[16,1]<=[16]}"', str(hlo))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
