# Copyright 2023 The JAX Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

from absl.testing import absltest
from jax._src import test_util as jax_test_util
from jax._src import xla_bridge
from jax._src.config import config
import jax.numpy as jnp


config.parse_flags_with_absl()


class PjrtGpuPluginTest(jax_test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = "gpu_plugin:"

  def test_add_using_numpy(self):
    result = jnp.add(1, 1)
    self.assertEqual(result, 2)
    self.assertEqual(os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"], "gpu_plugin:")
    platform_version = xla_bridge.get_backend().platform_version
    self.assertTrue(platform_version.startswith("PJRT C API\ncuda"))


if __name__ == "__main__":
  absltest.main(testLoader=jax_test_util.JaxTestLoader())
