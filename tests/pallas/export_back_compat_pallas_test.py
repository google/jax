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
"""Tests for backwards compatibility of exporting code with Pallas custom calls.

See the export_back_compat_test_util module docstring for how to setup and
update these tests.
"""

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.internal_test_util import export_back_compat_test_util as bctu
from jax._src.internal_test_util.export_back_compat_test_data.pallas import cuda_add_one
from jax.experimental import pallas as pl

config.parse_flags_with_absl()


@jtu.with_config(jax_include_full_tracebacks_in_locations=False)
class CompatTest(bctu.CompatTestBase):

  def setUp(self):
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Only works on GPU")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPUs with capability >= sm80")
    super().setUp()

  def test_cuda_add_one(self):
    def func(x):
      def add_one(x_ref, o_ref):
        o_ref[0] = x_ref[0] + 1
      return pl.pallas_call(add_one,
                            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
                            in_specs=[pl.BlockSpec((1,), lambda i: i)],
                            out_specs=pl.BlockSpec((1,), lambda i: i),
                            grid=8)(x)
    data = self.load_testdata(cuda_add_one.data_2024_05_02)

    self.run_one_test(func, data)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
