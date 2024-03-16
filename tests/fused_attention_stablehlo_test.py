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

from functools import partial
from absl.testing import absltest
from typing import Optional
import os

os.environ['XLA_FLAGS'] = '--xla_gpu_enable_cudnn_fmha=true --xla_gpu_fused_attention_use_cudnn_rng=true'

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec, NamedSharding
from jax._src import config
from jax._src import test_util as jtu
from jax._src.cudnn.fused_attention_stablehlo import dot_product_attention, check_is_flash_attention, check_cudnn_version

config.parse_flags_with_absl()
Array = jnp.ndarray

def sdpa_train(query: Array,
               key: Array,
               value: Array,
               grad: Array,
               bias: Optional[Array] = None,
               mask: Optional[Array] = None,
               scale: float = 0.5,
               is_causal_mask: bool = False,
               is_bnth: bool = False,
               dropout_rate: float = 0.1) -> Array:
  if mask is not None:
    # convert bool mask to dtype mask
    mask = mask.astype(query.dtype)
  out, sdpa_vjp = jax.vjp(
      partial(dot_product_attention, scale=scale, is_causal_mask=is_causal_mask,
              dropout_rate=dropout_rate,
              qkv_layout='BNTH' if is_bnth else 'BTNH',
              is_training=True),
      query, key, value, bias, mask)
  query_grad, key_grad, value_grad, _, _ = sdpa_vjp(grad)
  return out, (query_grad, key_grad, value_grad)

def sdpa_ref(query: Array,
      key: Array,
      value: Array,
      bias: Optional[Array] = None,
      mask: Optional[Array] = None,
      scale: float = 0.5,
      is_causal_mask: bool = False,
      dropout_rate: float = 0.1) -> Array:

  def get_large_negative_number(input_t):
    dtype = input_t.dtype
    if jnp.issubdtype(dtype, jnp.inexact):
      dtype_max = jnp.finfo(dtype).max
    elif jnp.issubdtype(dtype, jnp.integer):
      dtype_max = jnp.iinfo(dtype).max
    else:
      raise ValueError('Unsupported dtype for inputs.')
    large_negative_number = jnp.asarray(-0.7 * dtype_max, dtype=dtype)
    return large_negative_number

  def get_causal_mask(input_t):
    large_negative_number = get_large_negative_number(input_t)
    t = input_t.shape[2]
    col_idx = jax.lax.broadcasted_iota(np.int32, (t, t), 1)
    row_idx = jax.lax.broadcasted_iota(np.int32, (t, t), 0)
    mask = (row_idx < col_idx).astype(input_t.dtype) * large_negative_number
    return mask[jnp.newaxis, jnp.newaxis, :, :]

  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)
  if scale != 1.0:
    attn_weights = attn_weights * scale
  if is_causal_mask:
    bias = get_causal_mask(attn_weights)
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)
  if mask is not None:
    large_negative_number = get_large_negative_number(attn_weights)
    attn_weights = jax.lax.select(mask, attn_weights, jax.lax.broadcast(large_negative_number, attn_weights.shape))
  attn_weights = jax.nn.softmax(attn_weights)
  if dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    dropout_rng = jax.random.key(0)
    keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    attn_weights = jax.lax.select(keep, attn_weights / keep_prob, jnp.zeros_like(attn_weights))

  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)

def sdpa_train_ref(query: Array,
            key: Array,
            value: Array,
            grad: Array,
            bias: Optional[Array] = None,
            mask: Optional[Array] = None,
            scale: float = 0.5,
            is_causal_mask: bool = False,
            dropout_rate: float = 0.1) -> Array:
  out_ref, sdpa_vjp_ref = jax.vjp(
    partial(sdpa_ref, scale=scale, is_causal_mask=is_causal_mask, dropout_rate=dropout_rate),
    query, key, value, bias, mask)
  query_grad_ref, key_grad_ref, value_grad_ref, _, _ = sdpa_vjp_ref(grad)
  return out_ref, (query_grad_ref, key_grad_ref, value_grad_ref)

class DotProductAttentionTest(jtu.JaxTestCase):
  def setUp(self):
    # TODO(Cjkkkk): Tests fail when the cuDNN constraints check fails, or
    #  even when it passes: https://github.com/google/jax/issues/20438
    self.skipTest('Failing with various errors')
    return

    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 8904:
      self.skipTest('Requires >= cuDNN 8.9.4')

  @jtu.sample_product(
      batch_size=[4],
      seq_len=[256, 1024],
      num_heads=[8],
      head_dim=[64, 128],
      use_bias=[False, True],
      use_mask=[False, True],
      is_causal_mask=[False],
      dropout_rate=[0, 0.5],
      scale=[0.5],
      dtype=[jnp.float16, jnp.bfloat16]
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa(self, batch_size: int, seq_len: int, num_heads: int,
                head_dim: int, use_bias: bool, use_mask: bool, is_causal_mask: bool,
                dropout_rate: float, scale: float, dtype: jnp.dtype):
    if seq_len == 256 and is_causal_mask:
      self.skipTest("Fused attention does not support mask generation.")
    if seq_len == 256 and head_dim == 128:
      self.skipTest("Fused attention does not support head dim = 128.")
    if len(jax.local_devices()) <= 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")

    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.key(0), 6)
    query = jax.random.normal(
        k1, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    key = jax.random.normal(
        k2, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    value = jax.random.normal(
        k3, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    grad = jax.random.normal(
        k4, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    if use_bias:
      bias = jax.random.normal(
        k5, (batch_size, num_heads, seq_len, seq_len), dtype=dtype)
    else:
      bias = None
    if use_mask:
      mask = jax.random.bernoulli(
        k5, 0.5, (batch_size, num_heads, seq_len, seq_len))
    else:
      mask = None
    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ('dp', 'tp')) as mesh:
      qkv_spec = PartitionSpec('dp', None, 'tp', None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      if bias is not None:
        bias_spec = PartitionSpec('dp', 'tp', None, None)
      else:
        bias_spec = PartitionSpec()
      if mask is not None:
        mask_spec = PartitionSpec('dp', 'tp', None, None)
      else:
        mask_spec = PartitionSpec()
      bias_sharding = NamedSharding(mesh, bias_spec)
      mask_sharding = NamedSharding(mesh, mask_spec)
      replicated = NamedSharding(mesh, PartitionSpec())
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      if bias is not None:
        bias = jax.device_put(bias, bias_sharding)
      if mask is not None:
        mask = jax.device_put(mask, mask_sharding)
      grad = jax.device_put(grad, qkv_sharding)
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding, qkv_sharding, bias_sharding, mask_sharding)
      out_shardings = (replicated, (qkv_sharding, qkv_sharding, qkv_sharding))
      jitted_sdpa_train = jax.jit(
        partial(sdpa_train, scale=scale, is_causal_mask=is_causal_mask, dropout_rate=dropout_rate),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_train_ref = jax.jit(
        partial(sdpa_train_ref, scale=scale, is_causal_mask=is_causal_mask, dropout_rate=dropout_rate),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out, (query_grad, key_grad, value_grad) = jitted_sdpa_train(query, key, value, grad, bias, mask)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = jitted_sdpa_train_ref(query, key, value, grad, bias, mask)
      self.assertArraysAllClose(out_ref, out, rtol=1e-5, atol=1e-5)
      if seq_len > 512:
        # query_grad in flash attention is not deterministic
        self.assertArraysAllClose(query_grad_ref, query_grad, rtol=1e-2, atol=1e-2)
      else:
        self.assertArraysAllClose(query_grad_ref, query_grad, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(key_grad_ref, key_grad, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(value_grad_ref, value_grad, rtol=1e-5, atol=1e-5)

  @jtu.run_on_devices("cuda")
  def test_sdpa_inference(self):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(
        k1, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    key = jax.random.normal(
        k2, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    value = jax.random.normal(
        k3, (4, 1024, 4, 64), dtype=jnp.bfloat16)

    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ('dp', 'tp')) as mesh:
      qkv_spec = PartitionSpec('dp', None, 'tp', None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      replicated = NamedSharding(mesh, PartitionSpec())
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding, replicated, replicated)
      out_shardings = replicated
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      jitted_sdpa_inference = jax.jit(
        partial(dot_product_attention, scale=1.0, is_causal_mask=False, dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_inference_ref = jax.jit(
        partial(sdpa_ref, scale=1.0, is_causal_mask=False, dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out = jitted_sdpa_inference(query, key, value, None, None)
      out_ref = jitted_sdpa_inference_ref(query, key, value, None, None)
      self.assertArraysAllClose(out_ref, out, rtol=1e-5, atol=1e-5)

  def test_sdpa_utils(self):
    test_cases = {
      (256, 512, 64, 8905, False, False): False,
      (1, 257, 64, 8905, False, True): True,
      (1, 1024, 64, 8905, False, False): True,
      (1024, 1024, 64, 8905, False, False): True,
      (1024, 1024, 128, 8905, False, False): True,
    }

    for k, v in test_cases.items():
      sql_q, sql_v, head_dim, cudnn_version, has_bias, is_training = k
      query = jnp.empty((4, sql_q, 4, head_dim))
      key = jnp.empty((4, sql_v, 4, head_dim))
      self.assertEqual(check_is_flash_attention(query, key, cudnn_version, has_bias, is_training), v)

  def test_layouts(self):
    dtype = 'bfloat16'
    B, T, N, H = 4, 1024, 8, 128
    S = T
    k0, k1, k2, k3 = jax.random.split(jax.random.key(123), 4)
    query = jax.random.normal(k0, (B, T, N, H), dtype=dtype)
    key = jax.random.normal(k1, (B, S, N, H), dtype=dtype)
    value = jax.random.normal(k2, (B, S, N, H), dtype=dtype)
    grad = jax.random.normal(k3, (B, T, N, H), dtype=dtype)

    btnh_fn = jax.jit(partial(sdpa_train_ref, scale=.5, is_causal_mask=True,
                              dropout_rate=0.0))
    out_ref, (dq_ref, dk_ref, dv_ref) = btnh_fn(query, key, value, grad)

    def _cvt(x):
      return jnp.einsum('BTNH->BNTH', x)
    def _cvt_back(x):
      return jnp.einsum('BNTH->BTNH', x)
    bnth_fn = jax.jit(partial(sdpa_train, scale=.5, is_causal_mask=True,
                              is_bnth=True, dropout_rate=0.0))
    out, (dq, dk, dv) = bnth_fn(_cvt(query), _cvt(key), _cvt(value), _cvt(grad))

    self.assertArraysAllClose(out_ref, _cvt_back(out))
    self.assertArraysAllClose(dq_ref, _cvt_back(dq))
    self.assertArraysAllClose(dk_ref, _cvt_back(dk))
    self.assertArraysAllClose(dv_ref, _cvt_back(dv))

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
