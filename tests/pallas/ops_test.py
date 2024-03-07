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

"""Tests for common JAX operations within pallas_call."""

import functools

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl

# Import mosaic for flag definitions
from jax.experimental import mosaic as _  # noqa: F401


jax.config.parse_flags_with_absl()


class OpsTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")
    if not self.INTERPRET:
      if jtu.device_under_test() == "cpu":
        self.skipTest("Only interpreter mode supported on CPU")
      if (jtu.test_device_matches(["cuda"]) and
          not jtu.is_cuda_compute_capability_at_least("8.0")):
        self.skipTest("Only works on GPUs with capability >= sm80")

    super().setUp()

  @classmethod
  def pallas_call(cls, *args, **kwargs):
    return pl.pallas_call(*args, interpret=cls.INTERPRET, **kwargs)

  @parameterized.named_parameters(
      (fn.__name__, fn, dtype) for fn, dtype in [
          (lax.pow, jnp.float32),
          (lax.bitwise_and, jnp.int32),
          (lax.bitwise_or, jnp.int32),
          (lax.bitwise_xor, jnp.int32),
          (lax.shift_left, jnp.int32),
          (lax.shift_right_logical, jnp.int32),
      ]
  )
  def test_weak_dtype(self, fn, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct([1], dtype),
    )
    def kernel(x_ref, o_ref):
      o_ref[:] = fn(x_ref[:], y)

    x = jnp.array([4], dtype=dtype)
    y = 2 if jnp.issubdtype(dtype, jnp.integer) else 2.0
    np.testing.assert_allclose(kernel(x), fn(x, y))


class OpsInterpreterTest(OpsTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main()
