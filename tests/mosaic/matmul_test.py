# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test different parameterizations of a matmul."""

import os

from absl.testing import absltest, parameterized
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  matmul = None
else:
  from jax.experimental.mosaic.gpu.examples import matmul


config.parse_flags_with_absl()
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0")


@jtu.with_config(jax_traceback_filtering="off")
class MatmulTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if matmul is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")

  @parameterized.product(
      m=(128, 256, 512, 2048),
      n=(128, 256, 512, 2048),
      k=(128, 256, 512, 2048),
      stages=(2, 4),
      tile_m=(64, 128, 256),
      tile_n=(64, 128, 256),
      in_dtype=(jnp.float16, jnp.bfloat16),  # f32 tested separately
  )
  def test_matmul(self, m, k, n, stages, tile_m, tile_n, in_dtype):
    if stages * (128 // jnp.dtype(in_dtype).itemsize) > k:
      self.skipTest("Too many stages.")

    if m < tile_m:
      self.skipTest(f"No use in running a test with {m=} < {tile_m=}.")

    if n < tile_n:
      self.skipTest(f"No use in running a test with {n=} < {tile_n=}.")

    try:
      matmul.verify(
          m,
          k,
          n,
          stages,
          tile_m=tile_m,
          tile_n=tile_n,
          lhs_dtype=in_dtype,
          rhs_dtype=in_dtype,
          rhs_transpose=True,
      )
    except ValueError as e:
      if "Mosaic GPU kernel exceeds available shared memory" in str(e):
        self.skipTest("Not enough shared memory for test, skipping.")
      raise e

  @parameterized.product(
      m=(128, 256, 512, 2048),
      n=(128, 256, 512, 2048),
      k=(128, 256, 512, 2048),
      stages=(2, 4),
      tile_m=(64, 128, 256),
      tile_n=(64, 128, 256),
      high_precision=(False, True),
  )
  def test_matmul_f32(self, m, k, n, stages, tile_m, tile_n, high_precision):
    if stages * (128 // jnp.dtype(jnp.float32).itemsize) > k:
      self.skipTest("Too many stages.")

    if m < tile_m:
      self.skipTest(f"No use in running a test with {m=} < {tile_m=}.")

    if n < tile_n:
      self.skipTest(f"No use in running a test with {n=} < {tile_n=}.")

    try:
      matmul.verify(
          m,
          k,
          n,
          stages,
          tile_m=tile_m,
          tile_n=tile_n,
          lhs_dtype=jnp.float32,
          rhs_dtype=jnp.float32,
          rhs_transpose=True,
          precision=(
              matmul.F32Precision.TF32_X3
              if high_precision
              else matmul.F32Precision.DEFAULT
          ),
      )
    except ValueError as e:
      if "Mosaic GPU kernel exceeds available shared memory" in str(e):
        self.skipTest("Not enough shared memory for test, skipping.")
      raise e

  @parameterized.parameters(
      dict(m=55 * 128, n=95 * 128, k=48 * 128, stages=4, tile_m=128),
      dict(m=55 * 128, n=45 * 128, k=48 * 128, stages=4, tile_m=128),
      dict(m=64, n=95 * 128, k=48 * 128, stages=4, tile_m=64),
      dict(m=64, n=45 * 128, k=48 * 128, stages=4, tile_m=64),
  )
  def test_mixed_matmul(self, m, k, n, stages, tile_m):
    # RHS.element_size==1b so k_tile=128
    if stages * 128 > k:
      self.skipTest("Too many stages.")

    matmul.verify(
        m,
        k,
        n,
        stages,
        tile_m=tile_m,
        rhs_transpose=False,
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.int8,
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
