# Copyright 2024 The JAX Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from absl.testing import absltest

from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import linalg as lax_linalg
import numpy as np

config.parse_flags_with_absl()

class CholeskyUpdateTest(jtu.JaxTestCase):

  @jtu.sample_product(
      shape=[
          (128, 128),
      ],
      dtype=[jnp.float32, jnp.float64],
  )
  def testUpperOnes(self, shape, dtype):
    """A test with a (mildly) ill-conditioned matrix."""
    if dtype is jnp.float64 and not config.enable_x64.value:
      self.skipTest("Test disabled for x32 mode")
    r_upper = jnp.triu(jnp.ones(shape)).astype(dtype)
    w = jnp.arange(1, shape[0] + 1).astype(dtype)
    new_matrix = r_upper.T @ r_upper + jnp.outer(w, w)
    new_cholesky = jnp.linalg.cholesky(new_matrix, upper=True)

    updated = lax_linalg.cholesky_update(r_upper, w)

    atol = 1e-6 if (dtype is jnp.float64) else 2e-2
    jtu._assert_numpy_allclose(updated, new_cholesky, atol=atol)

  @jtu.sample_product(
      shape=[
          (128, 128),
      ],
      dtype=[jnp.float32, jnp.float64],
  )
  def testRandomMatrix(self, shape, dtype):
    if dtype is jnp.float64 and not config.enable_x64.value:
      self.skipTest("Test disabled for x32 mode")
    rng = jtu.rand_default(self.rng())
    a = rng(shape, np.float64)
    pd_matrix = jnp.array(a.T @ a).astype(dtype)
    old_cholesky = jnp.linalg.cholesky(pd_matrix, upper=True)

    w = rng((shape[0],), np.float64)
    w = jnp.array(w).astype(dtype)

    new_matrix = pd_matrix + jnp.outer(w, w)
    new_cholesky = jnp.linalg.cholesky(new_matrix, upper=True)
    updated = lax_linalg.cholesky_update(old_cholesky, w)
    atol = 1e-6 if dtype == jnp.float64 else 1e-3
    jtu._assert_numpy_allclose(updated, new_cholesky, atol=atol)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
