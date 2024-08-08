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

import math

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.internal_test_util import export_back_compat_test_util as bctu
from jax._src.internal_test_util.export_back_compat_test_data.pallas import mosaic_matmul
from jax._src.internal_test_util.export_back_compat_test_data.pallas import mosaic_semaphore_dma
from jax._src.internal_test_util.export_back_compat_test_data.pallas import triton_add_one
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu import matmul
import jax.numpy as jnp


config.parse_flags_with_absl()


@jtu.with_config(jax_include_full_tracebacks_in_locations=False)
class CompatTest(bctu.CompatTestBase):

  def setUp(self):
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPUs with capability >= sm80")
    super().setUp()

  def test_triton_add_one(self):
    def func(x):
      def add_one(x_ref, o_ref):
        o_ref[0] = x_ref[0] + 1
      return pl.pallas_call(add_one,
                            out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
                            in_specs=[pl.BlockSpec((1,), lambda i: i)],
                            out_specs=pl.BlockSpec((1,), lambda i: i),
                            grid=8)(x)
    data = self.load_testdata(triton_add_one.data_2024_05_02)

    self.run_one_test(func, data)

  @jax.default_matmul_precision("bfloat16")
  def test_mosaic_matmul(self):
    dtype = jnp.float32
    def func():
      # Build the inputs here, to reduce the size of the golden inputs.
      x_shape = (1024, 512)
      bias = 1.0
      scale = 1e-3
      x = bias + scale * jnp.arange(
          math.prod(x_shape), dtype=dtype).reshape(x_shape)
      y = x[:512, :256]
      res = matmul.matmul(x, y, block_shape=(256, 256))
      # Keep only slices of the output, to reduce the size of the goldens.
      return res[::16, ::16]

    data = self.load_testdata(mosaic_matmul.data_2023_09_22)
    self.run_one_test(func, data, rtol=2e-7)

  def test_mosaic_semaphore_dma(self):
    if not (jtu.test_device_matches(["tpu"]) and
            jtu.is_device_tpu_at_least(4)):
      # TODO: crashes during compilation on TPU v4
      self.skipTest("Only works on TPU v5+")

    # The signatures of TPU ops for semaphore and DMA have changed.
    # This test ensures that the new signatures are backwards compatible.
    def func():
      def dma_kernel(x, y):
        def body(dma_sem, sem):
          pltpu.async_copy(x, y, dma_sem).wait()
          pltpu.semaphore_signal(sem)
          pltpu.semaphore_wait(sem)
        pl.run_scoped(
            body, pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.REGULAR
        )
      x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
      y = pl.pallas_call(dma_kernel, out_shape=x)(x)
      return jnp.array_equal(x, y).astype(jnp.float32)

    data = self.load_testdata(
        mosaic_semaphore_dma.semaphore_and_dma_2024_04_22)
    self.run_one_test(func, data)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
