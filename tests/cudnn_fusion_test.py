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

from absl.testing import absltest, parameterized
from unittest import SkipTest
from jax._src import test_util as jtu
from jax._src.lib import cuda_versions
import jax
import jax.numpy as jnp
from jax._src.cudnn import cudnn_fusion


jax.config.parse_flags_with_absl()


class CudnnFusionTest(jtu.JaxTestCase):
  def setUp(self):
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("8.0") or
        cuda_versions.cudnn_get_version() < 9110):
      self.skipTest("Only works on >= sm80 GPUs with cuDNN 9.1.1+")
    super().setUp()

  @parameterized.parameters(["", "pmap"])
  @jtu.run_on_devices("cuda")
  def test_cudnn_fusion(self, mode):
    batch_size = 2
    if mode == "pmap" and jax.device_count() < batch_size:
        raise SkipTest("pmap test requires 2 GPUs")

    @cudnn_fusion
    def comp1(x, y, z):
        return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z

    k = jax.random.key(0)
    s = batch_size, 16, 16
    x = jnp.int8(jax.random.normal(k, shape=s))
    y = jnp.bfloat16(jax.random.normal(k, shape=s))
    z = jnp.float32(jax.random.normal(k, shape=s))

    fn = jax.pmap(comp1) if mode == "pmap" else comp1
    jitted = jax.jit(comp1)
    lowered = jitted.lower(x, y, z)
    stablehlo = lowered.as_text("stablehlo")
    self.assertIn("func.func private @comp1", stablehlo)
    self.assertIn("__cudnn$fusion", stablehlo)

    hlo = lowered.as_text("hlo")
    self.assertIn('custom_call_target="__cudnn$fusion"', hlo)
    self.assertIn("called_computations=", hlo)

    compiled = lowered.compile({"xla_gpu_cublas_fallback": False})
    hlo_after_opt = compiled.as_text()

    self.assertIn("kind=kCustom", hlo_after_opt)
    self.assertIn("plan_id", hlo_after_opt)

    self.assertAllClose(compiled(x, y, z), fn(x, y, z))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
