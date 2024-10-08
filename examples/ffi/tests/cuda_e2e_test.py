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

jax.config.parse_flags_with_absl()


class CudaE2eTests(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Unsupported platform")

    # Import here to avoid trying to load the library when it's not built.
    from jax_ffi_example import cuda_e2e
    self.foo = cuda_e2e.foo

  def test_fwd_interpretable(self):
    shape = (2, 3)
    a = 2. * jnp.ones(shape)
    b = 3. * jnp.ones(shape)
    observed = jax.jit(self.foo)(a, b)
    expected = (2. * (3. + 1.))
    self.assertArraysEqual(observed, expected)

  def test_bwd_interpretable(self):
    shape = (2, 3)
    a = 2. * jnp.ones(shape)
    b = 3. * jnp.ones(shape)

    def loss(a, b):
      return jnp.sum(self.foo(a, b))

    da_observed, db_observed = jax.jit(jax.grad(loss, argnums=(0, 1)))(a, b)
    da_expected = b + 1
    db_expected = a
    self.assertArraysEqual(da_observed, da_expected)
    self.assertArraysEqual(db_observed, db_expected)

  def test_fwd_random(self):
    shape = (2, 3)
    akey, bkey = jax.random.split(jax.random.key(0))
    a = jax.random.normal(key=akey, shape=shape)
    b = jax.random.normal(key=bkey, shape=shape)
    observed = jax.jit(self.foo)(a, b)
    expected = a * (b + 1)
    self.assertAllClose(observed, expected)

  def test_bwd_random(self):
    shape = (2, 3)
    akey, bkey = jax.random.split(jax.random.key(0))
    a = jax.random.normal(key=akey, shape=shape)
    b = jax.random.normal(key=bkey, shape=shape)
    jtu.check_grads(f=jax.jit(self.foo), args=(a, b), order=1,
                    modes=("rev",))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
