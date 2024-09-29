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
               dropout_rate: float = 0.1,
               sliding_window_length: int | None = None) -> Array:
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
              qkv_layout="BNTH" if is_bnth else "BTNH",
              sliding_window_length=sliding_window_length),
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
      dropout_rate: float = 0.1,
      sliding_window_length: int | None = None) -> Array:

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

  def get_sliding_window_mask(logits, window_length):
    large_negative_number = get_large_negative_number(logits.dtype)
    T = logits.shape[-2]
    col_idx = jax.lax.broadcasted_iota(np.int32, (T, T), 1)
    row_idx = jax.lax.broadcasted_iota(np.int32, (T, T), 0)
    mask = jnp.logical_or(
      row_idx < col_idx,
      col_idx <= row_idx - window_length).astype(logits.dtype) * large_negative_number
    return mask[(*([jnp.newaxis]*(len(logits.shape) - 2)), ...)]

  B, T, qN, H = query.shape
  _, _, kN, _ = key.shape
  logits = jnp.einsum("bqhd,bkhd->bhqk", query, key)
  if scale != 1.0:
    logits = logits * scale
  if mask_type == MaskType.CAUSAL:
    bias = get_causal_mask(logits)
  elif mask_type == MaskType.PADDING:
    bias = get_padding_mask(logits)
  elif sliding_window_length is not None:
    if sliding_window_length <= 0:
      raise ValueError(
        f"Expect sliding_window_length > 0, got {sliding_window_length}.")
    bias = get_sliding_window_mask(logits, sliding_window_length)
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
            dropout_rate: float = 0.1,
            sliding_window_length: int | None = None) -> Array:
  out_ref, sdpa_vjp_ref = jax.vjp(
    partial(
      sdpa_ref, scale=scale, mask_type=mask_type, dropout_rate=dropout_rate,
      sliding_window_length=sliding_window_length),
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

  @jtu.sample_product(
      batch_size=[1, 16],
  )
  @jtu.run_on_devices("cuda")
  def test_sdpa_dbias(self, batch_size: int):
    # cuDNN only supports dbias when batch size is 1. If the batch size is
    # greater, dbias is silently set to all zeros. This test verifies this
    # behavior for both vmap and regular use cases.
    # TODO: Remove this test once cuDNN adds broader dbias support.
    dtype = jnp.bfloat16
    x_shape = (batch_size, 512, 16, 48)
    bias_shape = (batch_size, 16, 512, 512)
    mask_shape = (1, 1, 512)

    keys = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(keys[0], x_shape, dtype=dtype)
    bias = jax.random.normal(keys[1], bias_shape, dtype=dtype)
    mask = jnp.ones(mask_shape, dtype=jnp.bool_)

    def attn(x, bias, mask):
      return dot_product_attention(x, x, x, bias, mask)

    def attn_vjp(x, bias, mask, target_fn):
      _, f_vjp = jax.vjp(target_fn, x, bias, mask)
      return f_vjp(x)

    attn_vmap = jax.vmap(attn, in_axes=(0, 0, None))
    attn_ref = jax.jit(partial(attn_vjp, target_fn=attn))
    attn_ans = jax.jit(partial(attn_vjp, target_fn=attn_vmap))

    _, dbias_ref, _ = attn_ref(x, bias, mask)
    x = jnp.expand_dims(x, axis=1)
    bias = jnp.expand_dims(bias, axis=1)
    _, dbias_ans, _ = attn_ans(x, bias, mask)
    dbias_ans = jnp.squeeze(dbias_ans, axis=1)
    self.assertArraysAllClose(dbias_ans, dbias_ref)
    if batch_size != 1:
      self.assertTrue(not jnp.any(dbias_ans))

  @jtu.run_on_devices("cuda")
  def test_sdpa_sliding_window_length(self):
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
        sdpa_train, scale=1.0, mask_type=MaskType.CAUSAL, dropout_rate=0,
        sliding_window_length=64),
    )
    # for reference implementation
    # sliding_window_length option itself will setup correct mask
    jitted_sdpa_train_ref = jax.jit(
      partial(
        sdpa_train_ref, scale=1.0, mask_type=MaskType.NO_MASK, dropout_rate=0,
        sliding_window_length=64),
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
      (1, 257, 64, 8905, False, True, True),
      (1, 1024, 64, 8905, False, False, True),
      (1024, 1024, 64, 8905, False, False, True),
      (1024, 1024, 128, 8905, False, False, True),
      (1024, 1024, 127, 8905, False, False, False),
    ]

    for k in test_cases:
      sql_q, sql_v, head_dim, cudnn_version, has_bias, is_training, \
        expected_pass = k
      query = jnp.empty((4, sql_q, 4, head_dim))
      key = jnp.empty((4, sql_v, 4, head_dim))
      if expected_pass:
        check_is_flash_attention(
          query, key, AttentionLayout.BNTH.value, cudnn_version, has_bias,
          is_training)
      else:
        with self.assertRaises(NotImplementedError):
          check_is_flash_attention(
            query, key, AttentionLayout.BNTH.value, cudnn_version, has_bias,
            is_training)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
