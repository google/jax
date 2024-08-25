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
import os

os.environ["XLA_FLAGS"] = \
  "--xla_gpu_enable_cudnn_fmha=true --xla_gpu_fused_attention_use_cudnn_rng=true"

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec, NamedSharding
from jax._src import config
from jax._src import test_util as jtu
from jax._src.cudnn.fused_attention_stablehlo import (
    dot_product_attention,
    check_is_flash_attention,
    check_cudnn_version,
    get_large_negative_number,
    MaskType,
    AttentionLayout,
)
from jax._src.cudnn.fused_attention_fp8_stablehlo import (
    dot_product_attention_fp8,
)
from flax.linen.fp8_ops import (quantize_dequantize, quantize)
cast_to_representable = partial(
       quantize_dequantize, scale=jnp.ones((1,)), compute_dtype=jnp.bfloat16
)


config.parse_flags_with_absl()
Array = jnp.ndarray

def sdpa_train(query: Array,
               key: Array,
               value: Array,
               grad: Array,
               bias: Array | None = None,
               mask: Array | None = None,
               scale: float = 0.5,
               mask_type: MaskType = MaskType.NO_MASK,
               is_bnth: bool = False,
               dropout_rate: float = 0.1) -> Array:
  if mask_type == MaskType.PADDING:
    if is_bnth:
      B, _, S, _ = query.shape
    else:
      B, S, _, _ = query.shape
    q_seqlen = kv_seqlen = jnp.full((B,), S // 2, jnp.int32)
  else:
    q_seqlen = kv_seqlen = None
  out, sdpa_vjp = jax.vjp(
      partial(dot_product_attention, scale=scale, mask_type=mask_type,
              dropout_rate=dropout_rate,
              qkv_layout="BNTH" if is_bnth else "BTNH"),
      query, key, value, bias, mask, q_seqlen, kv_seqlen)
  query_grad, key_grad, value_grad, bias_grad, _, _, _ = sdpa_vjp(grad)
  if bias is not None and len(bias.shape) == 3:
    # has dbias
    return out, (query_grad, key_grad, value_grad, bias_grad)
  return out, (query_grad, key_grad, value_grad)

def sdpa_ref(query: Array,
             key: Array,
             value: Array,
             bias: Array | None = None,
             mask: Array | None = None,
             scale: float = 0.5,
             mask_type: MaskType = MaskType.NO_MASK,
             dropout_rate: float = 0.1) -> Array:

  def get_causal_mask(logits):
    large_negative_number = get_large_negative_number(logits.dtype)
    t = logits.shape[-2]
    col_idx = jax.lax.broadcasted_iota(np.int32, (t, t), 1)
    row_idx = jax.lax.broadcasted_iota(np.int32, (t, t), 0)
    mask = (row_idx < col_idx).astype(logits.dtype) * large_negative_number
    return mask[(*([jnp.newaxis]*(len(logits.shape) - 2)), ...)]

  def get_padding_mask(logits):
    S, T = logits.shape[-2:]
    large_negative_number = get_large_negative_number(logits.dtype)
    q_padding = (jax.lax.iota(np.int32, S) >= S // 2).reshape((S, 1))
    kv_padding = (jax.lax.iota(np.int32, T) >= T // 2).reshape((1, T))
    combined_padding = \
      (q_padding + kv_padding).astype(logits.dtype) * large_negative_number
    return jax.lax.broadcast(combined_padding, logits.shape[:-2])

  def get_encoded_padding_mask(encoded):
    S = encoded.shape[1]
    encoded_padding = (jax.lax.iota(np.int32, S) < S // 2).astype(encoded.dtype)
    return jax.lax.broadcast_in_dim(
      encoded_padding, encoded.shape, broadcast_dimensions=[1])

  B, T, qN, H = query.shape
  _, _, kN, _ = key.shape
  logits = jnp.einsum("bqhd,bkhd->bhqk", query, key)
  if scale != 1.0:
    logits = logits * scale
  if mask_type == MaskType.CAUSAL:
    bias = get_causal_mask(logits)
  elif mask_type == MaskType.PADDING:
    bias = get_padding_mask(logits)
  if mask is not None:
    large_negative_number = get_large_negative_number(logits.dtype)
    mask = jnp.where(mask, jnp.asarray(0, query.dtype), large_negative_number)
  if bias is None:
    bias = mask
  elif mask is not None:
    bias += mask
  if bias is not None:
    if bias.shape != logits.shape:
      bias = jnp.broadcast_to(bias, logits.shape)
    logits = logits + bias.astype(logits.dtype)
  probs = jax.nn.softmax(logits, axis=-1)
  if dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    dropout_rng = jax.random.key(0)
    keep = jax.random.bernoulli(dropout_rng, keep_prob, probs.shape)
    probs = jax.lax.select(keep, probs / keep_prob, jnp.zeros_like(probs))
  encoded = jnp.einsum("bhqk,bkhd->bqhd", probs, value)
  if mask_type == MaskType.PADDING:
    # cuDNN padding mask generation will mask out output accordingly
    # make sure the behavior is the same
    encoded_mask = get_encoded_padding_mask(encoded)
    encoded = encoded * encoded_mask
  return encoded

def sdpa_train_ref(query: Array,
                   key: Array,
                   value: Array,
                   grad: Array,
                   bias: Array | None = None,
                   mask: Array | None = None,
                   scale: float = 0.5,
                   mask_type: MaskType = MaskType.NO_MASK,
                   dropout_rate: float = 0.1) -> Array:
  out_ref, sdpa_vjp_ref = jax.vjp(
    partial(
      sdpa_ref, scale=scale, mask_type=mask_type, dropout_rate=dropout_rate),
    query, key, value, bias, mask)
  query_grad_ref, key_grad_ref, value_grad_ref, bias_grad_ref, _ = sdpa_vjp_ref(grad)
  if bias is not None and len(bias.shape) == 3:
    return out_ref, (query_grad_ref, key_grad_ref, value_grad_ref, bias_grad_ref)
  return out_ref, (query_grad_ref, key_grad_ref, value_grad_ref)


class DotProductAttentionTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 8904:
      self.skipTest("Requires >= cuDNN 8.9.4")
    if not jtu.is_cuda_compute_capability_at_least("8.0"):
      self.skipTest("Requires at least Ampere arch")

  @jtu.sample_product(
      batch_size=[4],
      seq_len=[1024],
      num_heads=[8],
      head_dim=[64, 128],
      use_mask=[False, True],
      use_bias=[False, True],
      mask_type=[MaskType.NO_MASK],
      dropout_rate=[0, 0.5],
      scale=[0.5],
      dtype=[jnp.float16, jnp.bfloat16]
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa(self, batch_size: int, seq_len: int, num_heads: int,
                head_dim: int, use_mask: bool, use_bias: bool, mask_type: MaskType,
                dropout_rate: float, scale: float, dtype: jnp.dtype):
    if len(jax.local_devices()) <= 4:
      self.skipTest("Require at least 4 devices to run sharding tests.")
    if use_mask and mask_type != MaskType.NO_MASK:
      self.skipTest("Either pass in mask or generate mask directly in cuDNN.")
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
        k6, 0.5, (batch_size, num_heads, seq_len, seq_len))
    else:
      mask = None
    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", None, "tp", None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      if bias is not None:
        bias_spec = PartitionSpec("dp", "tp", None, None)
      else:
        bias_spec = PartitionSpec()
      if mask is not None:
        mask_spec = PartitionSpec("dp", "tp", None, None)
      else:
        mask_spec = PartitionSpec()
      bias_sharding = NamedSharding(mesh, bias_spec)
      mask_sharding = NamedSharding(mesh, mask_spec)
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      if bias is not None:
        bias = jax.device_put(bias, bias_sharding)
      if mask is not None:
        mask = jax.device_put(mask, mask_sharding)
      grad = jax.device_put(grad, qkv_sharding)
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding,
                      qkv_sharding, bias_sharding, mask_sharding)
      out_shardings = (qkv_sharding, (qkv_sharding, qkv_sharding, qkv_sharding))
      jitted_sdpa_train = jax.jit(
        partial(
          sdpa_train, scale=scale, mask_type=mask_type,
          dropout_rate=dropout_rate),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_train_ref = jax.jit(
        partial(
          sdpa_train_ref, scale=scale, mask_type=mask_type,
          dropout_rate=dropout_rate),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out, (query_grad, key_grad, value_grad) = \
          jitted_sdpa_train(query, key, value, grad, bias, mask)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = \
          jitted_sdpa_train_ref(query, key, value, grad, bias, mask)
      self.assertArraysAllClose(out_ref, out, rtol=1e-5, atol=1e-5)
      if seq_len > 512:
        # query_grad in flash attention is not deterministic
        self.assertArraysAllClose(
          query_grad_ref, query_grad, rtol=1e-2, atol=1e-2)
      else:
        self.assertArraysAllClose(
          query_grad_ref, query_grad, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(
        key_grad_ref, key_grad, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(
        value_grad_ref, value_grad, rtol=1e-5, atol=1e-5)

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
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", None, "tp", None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      replicated = NamedSharding(mesh, PartitionSpec())
      in_shardings = (
        qkv_sharding, qkv_sharding, qkv_sharding, replicated, replicated)
      out_shardings = qkv_sharding
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      jitted_sdpa_inference = jax.jit(
        partial(
          dot_product_attention, scale=1.0, mask_type=MaskType.NO_MASK,
          dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_inference_ref = jax.jit(
        partial(
          sdpa_ref, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out = jitted_sdpa_inference(query, key, value, None, None)
      out_ref = jitted_sdpa_inference_ref(query, key, value, None, None)
      self.assertArraysAllClose(out_ref, out, rtol=1e-5, atol=1e-5)

  @jtu.run_on_devices("cuda")
  def test_sdpa_var_seq(self):
    self.skipTest("Skip before fixed.")
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(
        k1, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    key = jax.random.normal(
        k2, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    value = jax.random.normal(
        k3, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    grad = jax.random.normal(
        k4, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    jitted_sdpa_train = jax.jit(
      partial(
        sdpa_train, scale=1.0, mask_type=MaskType.PADDING, dropout_rate=0),
    )

    jitted_sdpa_train_ref = jax.jit(
      partial(
        sdpa_train_ref, scale=1.0, mask_type=MaskType.PADDING, dropout_rate=0),
    )

    out, (query_grad, key_grad, value_grad) = \
      jitted_sdpa_train(query, key, value, grad, None, None)
    out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = \
      jitted_sdpa_train_ref(query, key, value, grad, None, None)
    self.assertArraysAllClose(out_ref, out, rtol=1e-5, atol=1e-5)
    self.assertArraysAllClose(query_grad_ref, query_grad, rtol=1e-2, atol=1e-2)
    self.assertArraysAllClose(key_grad_ref, key_grad, rtol=1e-5, atol=1e-5)
    self.assertArraysAllClose(value_grad_ref, value_grad, rtol=1e-5, atol=1e-5)

  @jtu.run_on_devices("cuda")
  def test_sdpa_broadcast_bias_and_dbias(self):
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 8906:
      self.skipTest("Requires >= cuDNN 8.9.6")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Requires at least Hopper arch")

    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    query = jax.random.normal(
        k1, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    key = jax.random.normal(
        k2, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    value = jax.random.normal(
        k3, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    grad = jax.random.normal(
        k4, (4, 1024, 4, 64), dtype=jnp.bfloat16)
    bias = jax.random.normal(
        k5, (4, 1024, 1024), dtype=jnp.bfloat16)
    devices = np.array(jax.local_devices()[:4])
    devices = devices.reshape((2, 2))
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", None, "tp", None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      bias_spec = PartitionSpec("tp", None, None)
      bias_sharding = NamedSharding(mesh, bias_spec)
      replicated = NamedSharding(mesh, PartitionSpec())
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding,
                      qkv_sharding, bias_sharding, replicated)
      out_shardings = (qkv_sharding, (qkv_sharding, qkv_sharding, qkv_sharding, bias_sharding))
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      grad = jax.device_put(grad, qkv_sharding)
      bias = jax.device_put(bias, bias_sharding)
      jitted_sdpa_train = jax.jit(
        partial(
          sdpa_train, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_train_ref = jax.jit(
        partial(
          sdpa_train_ref, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      out, (query_grad, key_grad, value_grad, bias_grad) = \
        jitted_sdpa_train(query, key, value, grad, bias, None)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref, bias_grad_ref) = \
        jitted_sdpa_train_ref(query, key, value, grad, bias, None)
      self.assertArraysAllClose(out_ref, out, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(query_grad_ref, query_grad, rtol=1e-2, atol=1e-2)
      self.assertArraysAllClose(key_grad_ref, key_grad, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(value_grad_ref, value_grad, rtol=1e-5, atol=1e-5)
      self.assertArraysAllClose(bias_grad_ref, bias_grad, rtol=1e-5, atol=1e-5)

  @jtu.run_on_devices("cuda")
  def test_layouts(self):
    dtype = "bfloat16"
    B, T, N, H = 4, 1024, 8, 128
    S = T
    k0, k1, k2, k3 = jax.random.split(jax.random.key(123), 4)
    query = jax.random.normal(k0, (B, T, N, H), dtype=dtype)
    key = jax.random.normal(k1, (B, S, N, H), dtype=dtype)
    value = jax.random.normal(k2, (B, S, N, H), dtype=dtype)
    grad = jax.random.normal(k3, (B, T, N, H), dtype=dtype)

    btnh_fn = jax.jit(partial(sdpa_train_ref, scale=.5,
      mask_type=MaskType.CAUSAL, dropout_rate=0.0))
    out_ref, (dq_ref, dk_ref, dv_ref) = btnh_fn(query, key, value, grad)

    def _cvt(x):
      return jnp.einsum("BTNH->BNTH", x)
    def _cvt_back(x):
      return jnp.einsum("BNTH->BTNH", x)
    bnth_fn = jax.jit(partial(sdpa_train, scale=.5, mask_type=MaskType.CAUSAL,
                              is_bnth=True, dropout_rate=0.0))
    out, (dq, dk, dv) = bnth_fn(_cvt(query), _cvt(key), _cvt(value), _cvt(grad))

    self.assertArraysAllClose(out_ref, _cvt_back(out))
    self.assertArraysAllClose(dq_ref, _cvt_back(dq))
    self.assertArraysAllClose(dk_ref, _cvt_back(dk))
    self.assertArraysAllClose(dv_ref, _cvt_back(dv))

  def test_sdpa_utils(self):
    test_cases = [
      (1, 257, 64, 8905, False, True),
      (1, 1024, 64, 8905, False, False),
      (1024, 1024, 64, 8905, False, False),
      (1024, 1024, 128, 8905, False, False),
    ]

    for k in test_cases:
      sql_q, sql_v, head_dim, cudnn_version, has_bias, is_training = k
      query = jnp.empty((4, sql_q, 4, head_dim))
      key = jnp.empty((4, sql_v, 4, head_dim))
      check_is_flash_attention(
        query, key, AttentionLayout.BNTH, cudnn_version, has_bias, is_training)

@jtu.with_config(jax_numpy_dtype_promotion='standard')
class DotProductAttentionF8Test(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")
    try:
      cudnn_version = check_cudnn_version()
    except RuntimeError as e:
      self.skipTest(str(e))
      return
    if cudnn_version < 9010:
      self.skipTest("Requires >= cuDNN 9.1.0")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Requires at least Hopper arch")

  @jtu.sample_product(
      batch_size=[2, 4],
      seq_len=[128, 256],
      num_heads=[4, 8],
      head_dim=[128],
      use_causal_mask=[False],
      qkv_layout=['BNTH'],
      scale=[1.0, 0.75],
      dtype=[jnp.bfloat16, jnp.float16]
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa_fp8(self, batch_size: int, seq_len: int, num_heads: int,
                    head_dim: int, use_causal_mask: bool, qkv_layout: str,
                    scale: float, dtype: jnp.dtype):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    input_shape = (batch_size, num_heads, seq_len, head_dim)
    query_h = jax.random.normal(
        k1, input_shape, dtype=dtype)
    key_h = jax.random.normal(
        k2, input_shape, dtype=dtype)
    value_h = jax.random.normal(
        k3, input_shape, dtype=dtype)
    grad_h = jax.random.normal(
        k4, input_shape, dtype=dtype)
    mask_type = MaskType.CAUSAL if use_causal_mask else MaskType.NO_MASK

    query = cast_to_representable(query_h, jnp.float8_e4m3fn)
    key = cast_to_representable(key_h, jnp.float8_e4m3fn)
    value = cast_to_representable(value_h, jnp.float8_e4m3fn)
    grad = cast_to_representable(grad_h, jnp.float8_e4m3fn)
   
    query_quantized = quantize(query, jnp.float8_e4m3fn, jnp.ones((1,)), jnp.float32)
    key_quantized = quantize(key, jnp.float8_e4m3fn, jnp.ones((1,)), jnp.float32)
    value_quantized = quantize(value, jnp.float8_e4m3fn, jnp.ones((1,)), jnp.float32)
    grad_quantized = quantize(grad, jnp.float8_e4m3fn, jnp.ones((1,)), jnp.float32)

    variables = [
      'amax_dQ', 'amax_dK', 'amax_dV', 'amax_dP',
      'descale_q', 'descale_k', 'descale_v', 'descale_s',
      'scale_s', 'scale_o', 'descale_o', 'descale_dO',
      'descale_dP', 'scale_dQ', 'scale_dK', 'scale_dV', 'scale_dP',
    ]

    fp8_metas = {name: jnp.ones((1, 1, 1, 1), dtype=jnp.float32) for name in variables}

    def sdpa_train_fp8(query: Array,
                       key: Array,
                       value: Array,
                       grad: Array, 
                       *fp8_metas):
      out, sdpa_vjp = jax.vjp(
        partial(
          dot_product_attention_fp8,
          scale=scale, use_causal_mask=use_causal_mask, qkv_layout=qkv_layout),
        query, key, value, *fp8_metas
      )
      amax_dV, amax_dP = fp8_metas[2], fp8_metas[3]
      query_grad, key_grad, value_grad, *_ = sdpa_vjp((grad, amax_dV, amax_dP))
      return out[0], (query_grad, key_grad, value_grad)

    devices = np.array(jax.local_devices()[:2])
    devices = devices.reshape((2, 1))
  
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", "tp", None, None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      
      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      grad = jax.device_put(grad, qkv_sharding)

      query_quantized = jax.device_put(query_quantized , qkv_sharding)
      key_quantized  = jax.device_put(key_quantized , qkv_sharding)
      value_quantized  = jax.device_put(value_quantized , qkv_sharding)
      grad_quantized = jax.device_put(grad_quantized , qkv_sharding)
  
      fp8_meta_shardings = (None,) * len(fp8_metas)
      in_shardings = (qkv_sharding, qkv_sharding, qkv_sharding, qkv_sharding, *fp8_meta_shardings)
      out_shardings = (qkv_sharding, (qkv_sharding, qkv_sharding, qkv_sharding))

      in_shardings_ref = (qkv_sharding, qkv_sharding, qkv_sharding, qkv_sharding, None, None)
      out_shardings_ref = out_shardings

      args = tuple(fp8_metas.values())
      jitted_sdpa_train_fp8 = jax.jit(sdpa_train_fp8, in_shardings=in_shardings, out_shardings=out_shardings)
      jitted_sdpa_train_ref = jax.jit(
        partial(
          sdpa_train_ref, scale=scale, mask_type=mask_type, dropout_rate=0.0),
        in_shardings=in_shardings_ref,
        out_shardings=out_shardings_ref
      )

      out, (query_grad, key_grad, value_grad) = \
          jitted_sdpa_train_fp8(query_quantized, key_quantized, value_quantized, grad_quantized, *args)
      out_ref, (query_grad_ref, key_grad_ref, value_grad_ref) = \
          jitted_sdpa_train_ref(query, key, value, grad, None, None)

      self.assertArraysAllClose(out_ref, out.astype(dtype), rtol=5e-1, atol=5e-1)
      self.assertArraysAllClose(query_grad_ref, query_grad.astype(dtype), rtol=5e-1, atol=3e-0)
      self.assertArraysAllClose(key_grad_ref, key_grad.astype(dtype), rtol=5e-1, atol=3e-0)
      self.assertArraysAllClose(value_grad_ref, value_grad.astype(dtype), rtol=5e-1, atol=5e-1)

  @jtu.sample_product(
      batch_size=[4, 2],
      seq_len=[4, 16],
      num_heads=[4, 16],
      head_dim=[16, 32],
      use_causal_mask=[False],
      qkv_layout=['BNTH'],
      scale=[1.0, 0.75],
      dtype=[jnp.bfloat16, jnp.float16]
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa_fp8_inference(self, batch_size: int, seq_len: int, num_heads: int,
                              head_dim: int, use_causal_mask: bool, qkv_layout: str,
                              scale: float, dtype: jnp.dtype):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)

    input_shape = (batch_size, num_heads, seq_len, head_dim)
    query_h = jax.random.normal(k1, input_shape, dtype=dtype)
    key_h = jax.random.normal(k2, input_shape, dtype=dtype)
    value_h = jax.random.normal(k3, input_shape, dtype=dtype)
    mask_type = MaskType.CAUSAL if use_causal_mask else MaskType.NO_MASK

    query = cast_to_representable(query_h, jnp.float8_e4m3fn)
    key = cast_to_representable(key_h, jnp.float8_e4m3fn)
    value = cast_to_representable(value_h, jnp.float8_e4m3fn)
    
    query_quantized = quantize(query, jnp.float8_e4m3fn, jnp.ones((1,)), jnp.float32)
    key_quantized = quantize(key, jnp.float8_e4m3fn, jnp.ones((1,)), jnp.float32)
    value_quantized = quantize(value, jnp.float8_e4m3fn, jnp.ones((1,)), jnp.float32)
    
    variables = [
      'amax_dQ', 'amax_dK', 'amax_dV', 'amax_dP',
      'descale_q', 'descale_k', 'descale_v', 'descale_s',
      'scale_s', 'scale_o', 'descale_o', 'descale_dO',
      'descale_dP', 'scale_dQ', 'scale_dK', 'scale_dV', 'scale_dP',
    ]

    fp8_metas = {name: jnp.ones((1, 1, 1, 1), dtype=jnp.float32) for name in variables}
    devices = np.array(jax.local_devices()[:2])
    devices = devices.reshape((2, 1))
    with Mesh(devices, ("dp", "tp")) as mesh:
      qkv_spec = PartitionSpec("dp", "tp", None, None)
      qkv_sharding = NamedSharding(mesh, qkv_spec)
      fp8_meta_shardings = (None,) * len(variables)
      in_shardings = (
        qkv_sharding, qkv_sharding, qkv_sharding, *fp8_meta_shardings)
      out_shardings = (qkv_sharding, None, None)

      in_shardings_ref = (
        qkv_sharding, qkv_sharding, qkv_sharding, None, None)
      out_shardings_ref = qkv_sharding

      query = jax.device_put(query, qkv_sharding)
      key = jax.device_put(key, qkv_sharding)
      value = jax.device_put(value, qkv_sharding)
      jitted_sdpa_inference = jax.jit(
        partial(
          dot_product_attention_fp8, scale=scale, use_causal_mask=use_causal_mask, qkv_layout=qkv_layout),
        in_shardings=in_shardings,
        out_shardings=out_shardings
      )

      jitted_sdpa_inference_ref = jax.jit(
        partial(
          dot_product_attention, scale=scale),
        in_shardings=in_shardings_ref,
        out_shardings=out_shardings_ref
      )
      args = tuple(fp8_metas.values())
      out, _, _ = jitted_sdpa_inference(query_quantized, key_quantized, value_quantized, *args)
      out_ref = jitted_sdpa_inference_ref(query, key, value, None, None)
      self.assertArraysAllClose(out_ref, out.astype(dtype), rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
