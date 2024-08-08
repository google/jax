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
import jax.extend as jex
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class FfiCpuTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jtu.is_running_under_pytest():
      self.skipTest(
          "These tests are only intended to be run using Bazel because they "
          "require compiled dependencies.")
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("These tests are only intended for CPU")

    # Import here so that we don't even try to import when running with pytest.
    from jax.tests.ffi import ffi_cpu  # pylint: disable=g-import-not-at-top

    for name, target in ffi_cpu.registrations().items():
      jex.ffi.register_ffi_target(name, target, platform="cpu")

  def test_add_to(self):
    def add_to(delta, x):
      return jex.ffi.ffi_call("add_to", x, x, delta=np.int32(delta))

    x = jnp.arange(10).astype(jnp.int32)
    self.assertArraysEqual(add_to(5, x), x + 5)

  def test_error(self):
    out_type = jax.ShapeDtypeStruct((), jnp.int32)
    jex.ffi.ffi_call("should_fail", out_type, should_fail=False)  # no error
    with self.assertRaisesRegex(Exception, ".* Test should error"):
      jex.ffi.ffi_call("should_fail", out_type, should_fail=True)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
