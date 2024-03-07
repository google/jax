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
"""Test different parameterizations of FlashAttention."""

import os

from absl.testing import absltest, parameterized
from jax._src import config
from jax._src import test_util as jtu

# pylint: disable=g-import-not-at-top
try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  flash_attention = None
else:
  from jax.experimental.mosaic.gpu.examples import flash_attention


config.parse_flags_with_absl()
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0")


@jtu.with_config(jax_traceback_filtering="off")
class FlashAttentionTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if flash_attention is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")

  @parameterized.product(
      batch_size=(1,),
      q_seq_len=(4096,),
      kv_seq_len=(4096,),
      num_q_and_kv_heads=((4, 1),    # MQA
                          (6, 3),    # GQA
                          (4, 4),),  # MHA
      head_dim=(64, 128, 256),
      # Provide a default value for exp_impl if 'flash_attention' is not
      # available. Bypasses test failures when Mosaic is not available.
      exp_impl=[*(flash_attention.ExpImplementation
                  if flash_attention is not None else (NotImplementedError,))],
  )
  def test_flash_attention(self, batch_size, q_seq_len, kv_seq_len,
                           num_q_and_kv_heads, head_dim, exp_impl):
    num_q_heads, num_kv_heads = num_q_and_kv_heads
    flash_attention.benchmark_and_verify(
        batch_size=batch_size,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        exp_impl=exp_impl,
        blocks=flash_attention.BlockSizes(stages=2, q=64, kv=64)
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
