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
import jax.numpy as jnp
from jax._src import test_util as jtu

from jax_ffi_example import rms_norm

jax.config.parse_flags_with_absl()


def rms_norm_ref(x, eps=1e-5):
  scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
  return x / scale


class RmsNormTests(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu"]):
      self.skipTest("Unsupported platform")

  def test_basic(self):
    x = jnp.linspace(-0.5, 0.5, 15)
    self.assertAllClose(rms_norm.rms_norm(x), rms_norm_ref(x))

  def test_batching(self):
    x = jnp.linspace(-0.5, 0.5, 15).reshape((3, 5))
    self.assertAllClose(jax.vmap(rms_norm.rms_norm)(x), jax.vmap(rms_norm_ref)(x))

  def test_grads(self):
    x = jnp.linspace(-0.5, 0.5, 15).reshape((3, 5))
    jtu.check_grads(rms_norm.rms_norm, (x,), order=1, modes=("rev",))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
