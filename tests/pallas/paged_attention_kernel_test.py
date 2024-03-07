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
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas.ops.tpu import paged_attention
from jax.experimental.pallas.ops.tpu.paged_attention import quantization_utils
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


def _generate_qkv(
    seq_lens,
    page_size,
    max_seq_len,
    num_kv_heads,
    num_heads,
    head_dim,
    prng_key,
    dtype=jnp.float32,
    are_kv_quantized=False,
):
  assert max_seq_len % page_size == 0
  pages_per_sequence = max_seq_len // page_size
  batch_size = len(seq_lens)
  total_pages = batch_size * pages_per_sequence
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  k_pages = jax.random.normal(
      k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )
  v_pages = jax.random.normal(
      k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
  )

  if are_kv_quantized:
    k_pages = quantization_utils.quantize_to_int8(k_pages)
    v_pages = quantization_utils.quantize_to_int8(v_pages)

  page_indices = jnp.arange(batch_size * pages_per_sequence, dtype=jnp.int32)
  page_indices = jax.random.permutation(k3, page_indices, independent=True)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = jax.random.normal(k4, (batch_size, num_heads, head_dim), dtype=dtype)
  return q, k_pages, v_pages, page_indices


def _reconstruct_kv(page_indices, pages):
  if isinstance(pages, quantization_utils.QuantizedTensor):
    pages = quantization_utils.unquantize_from_int8(pages, dtype=jnp.float32)

  batch_size = page_indices.shape[0]
  num_heads, _, _, head_dim = pages.shape

  def per_sequence_page_gather(pages, page_indices):
    return jnp.take(pages, page_indices, 1)

  gathered = jax.vmap(per_sequence_page_gather, in_axes=(None, 0))(
      pages, page_indices
  )
  return gathered.reshape(batch_size, num_heads, -1, head_dim)


def _grouped_query_attention_reference(q, k, v, lengths):
  batch_size, num_heads, head_dim = q.shape
  _, num_kv_heads, max_seq_len, _ = k.shape
  assert k.shape == v.shape
  assert num_heads % num_kv_heads == 0
  q = q.reshape(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)

  if isinstance(k, quantization_utils.QuantizedTensor):
    k = quantization_utils.unquantize_from_int8(k, dtype=jnp.float32)
  if isinstance(v, quantization_utils.QuantizedTensor):
    v = quantization_utils.unquantize_from_int8(v, dtype=jnp.float32)

  logits = jnp.einsum(
      "bhgd,bhtd->bhgt", q.astype(jnp.float32), k.astype(jnp.float32)
  )
  mask = jnp.arange(max_seq_len)[None] < lengths[:, None]
  mask_value = -0.7 * float(np.finfo(np.dtype("float32")).max)
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None, None, :]
  weights = jax.nn.softmax(logits, axis=-1)
  o = jnp.einsum("bhgt,bhtd->bhgd", weights.astype(v.dtype), v)
  return o.reshape(batch_size, num_heads, head_dim)


def _megacore_enabled():
  return jax.devices()[0].device_kind == "TPU v4" or jtu.is_device_tpu(
      version=5, variant="p"
  )


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class PagedAttentionKernelTest(jtu.JaxTestCase):
  @parameterized.product(
      dtype=(jnp.float32, jnp.bfloat16),
      page_size=(16, 32, 64),
      num_kv_heads=(1, 8),
      q_kv_head_ratio=(1, 4, 8),
      head_dim=(128, 256),
      megacore_mode=("batch", "kv_head", None),
      are_kv_quantized=(
          False,
          True,
      ),
  )
  def test_paged_attention(
      self,
      dtype,
      page_size,
      num_kv_heads,
      q_kv_head_ratio,
      head_dim,
      megacore_mode,
      are_kv_quantized,
  ):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest("Only supports TPU generation 4 or above")
    if jtu.is_device_tpu(version=4) and are_kv_quantized:
      # TPU v4 has only 16MiB of VMEM which is not sufficient to store both the
      # weight and scale tensors for quantized tensors. When enabled on TPUv4,
      # the tests sometimes failed with resource exhausted error.
      self.skipTest("Quantization is not supported on TPU v4")
    if megacore_mode and not _megacore_enabled():
      self.skipTest("Megacore is only available on TPU v4 or TPU v5p")
    if num_kv_heads % 2 != 0 and megacore_mode == "kv_head":
      self.skipTest("Skip kv_head megacore mode when num_kv_heads is odd")
    max_kv_len = 2048
    block_size = 512
    seq_lens = np.asarray([0, 3, 256, 513, 1023, 2048])
    q, k_pages, v_pages, page_indices = _generate_qkv(
        seq_lens,
        page_size,
        max_kv_len,
        num_kv_heads,
        num_kv_heads * q_kv_head_ratio,
        head_dim,
        jax.random.key(0),
        dtype,
        are_kv_quantized=are_kv_quantized,
    )
    o = paged_attention.paged_attention(
        q,
        k_pages,
        v_pages,
        seq_lens,
        page_indices,
        pages_per_compute_block=block_size // page_size,
        megacore_mode=megacore_mode,
    )
    k = _reconstruct_kv(page_indices, k_pages)
    v = _reconstruct_kv(page_indices, v_pages)
    o_ref = _grouped_query_attention_reference(q, k, v, seq_lens)

    if q_kv_head_ratio > 1:
      atol, rtol = 1e-2, 2e-2
    else:
      atol, rtol = 2e-1, 1e-1
    np.testing.assert_allclose(
        o[np.where(seq_lens > 0)].astype(jnp.float32),
        o_ref[np.where(seq_lens > 0)].astype(jnp.float32),
        atol=atol,
        rtol=rtol,
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
