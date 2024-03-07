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

import os
import unittest

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu

config.parse_flags_with_absl()


class GpuMemoryAllocationTest(absltest.TestCase):

  # This test must be run in its own subprocess.
  @jtu.skip_under_pytest("Test must run in an isolated process")
  @unittest.skipIf(
      "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ,
      "Test does not work if the python client allocator has been overriden",
  )
  def test_gpu_memory_allocation(self):
    falsey_values = ("0", "False", "false")
    preallocate = (
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "1")
        not in falsey_values
    )
    device = jax.devices()[0]
    mem_stats = device.memory_stats()
    self.assertEqual(mem_stats["pool_bytes"], 0)
    x = jax.lax.add(1, 2)

    mem_stats = device.memory_stats()
    if preallocate:
      self.assertEqual(mem_stats["pool_bytes"], mem_stats["bytes_limit"])
    else:
      self.assertLessEqual(
          mem_stats["pool_bytes"], mem_stats["bytes_limit"] // 2
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
