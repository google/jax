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

"""Module containing decode attention."""
from __future__ import annotations

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp


def attn_forward_kernel(
    q_ref,  # [num_heads, head_dim]
    k_ref,  # [k_seq_len, head_dim]
    v_ref,  # [k_seq_len, head_dim]
    o_ref: Any,  # [num_heads, head_dim]
    *residual_refs: Any,  # Residual outputs: [num_heads,], [num_heads,]
    sm_scale: float,
    block_k: int,
):
  block_h, head_dim = q_ref.shape
  k_seq_len, _ = k_ref.shape
  start_q = pl.program_id(0)

  # o is the buffer where we accumulate the output on sram.
  # m_i and l_i (see FlashAttention2 paper) are updated during the k,v loop.
  m_i = jnp.zeros(block_h, dtype=jnp.float32) - float("inf")
  l_i = jnp.zeros(block_h, dtype=jnp.float32)
  o = jnp.zeros((block_h, head_dim), dtype=jnp.float32)

  # Load q: it will stay in L1 throughout. Indices form a matrix because we
  # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
  # q tile has shape [block_h, head_dim].
  curr_q_slice = pl.dslice(start_q * block_h, block_h)
  q = pl.load(q_ref, (curr_q_slice, pl.dslice(None)))

  def _dot(a, b):
    # if a.shape[0] == 1:
    #   # Use matrix vector product
    #   return (a.T * b).sum(axis=0, keepdims=True)
    return pl.dot(a, b)

  # Loop over blocks of kv to process entire kv seq_len.
  # Grid loops over q blocks over num_heads.
  def body(start_k, carry):
    o_prev, m_prev, l_prev = carry
    curr_k_slice = pl.dslice(start_k * block_k, block_k)

    k = pl.load(k_ref, (curr_k_slice, slice(None)))
    qk = _dot(q, k.T)  # [block_h, block_k]
    if sm_scale != 1.0:
      qk *= sm_scale  # [block_h, block_k]

    m_curr = qk.max(axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    correction = jnp.exp(m_prev - m_next)
    l_prev_corr = correction * l_prev
    s_curr = jnp.exp(
        qk - m_next[:, None]
    )  # Use m_next instead of m_curr to avoid a correction on l_curr
    l_curr = s_curr.sum(axis=-1)
    l_next = l_prev_corr + l_curr
    v = pl.load(v_ref, (curr_k_slice, slice(None)))
    o_curr = _dot(s_curr.astype(v.dtype), v)

    # flash2 unscaled_o
    o_next = correction[:, None] * o_prev + o_curr
    return o_next, m_next, l_next

  upper_bound = pl.cdiv(k_seq_len, block_k)
  # o is left unscaled; it will be scaled in the final reduction step
  o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

  if residual_refs:
    l_ref, m_ref = residual_refs
    pl.store(l_ref, (curr_q_slice,), l_i)
    pl.store(m_ref, (curr_q_slice,), m_i)
  # Write output to dram.
  o = o.astype(o_ref.dtype)
  pl.store(o_ref, (curr_q_slice, pl.dslice(None)), o)


def attn_unbatched(
    q,  # [num_heads, head_dim]
    k,  # [k_seq_len, head_dim]
    v,  # [k_seq_len, head_dim]
    sm_scale: float,
    block_h: int,
    block_k: int,
    k_splits: int,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
):
  num_heads, head_dim = q.shape
  k_seq_len, _ = k.shape
  # Pad num query heads to 16 if needed, and slice output at the end.
  original_num_heads = None
  if num_heads < 16:
    q = jnp.pad(q, ((0, 16 - num_heads), (0, 0)))
    original_num_heads = num_heads
    num_heads = q.shape[0]
  block_h = min(block_h, num_heads)
  head_splits = pl.cdiv(num_heads, block_h)
  grid_ = grid
  if grid_ is None:
    grid_ = (head_splits, k_splits)

  assert (
      k_seq_len % k_splits == 0
  ), f"{k_seq_len=} must be divisible by {k_splits=}"
  k = k.reshape(k_splits, k_seq_len // k_splits, head_dim)
  v = v.reshape(k_splits, k_seq_len // k_splits, head_dim)
  k_seq_len = k_seq_len // k_splits
  assert min(num_heads, head_dim, k_seq_len) >= 16, "Minimum pl.dot size is 16"
  block_k = min(block_k, k_seq_len)
  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4
  kernel = functools.partial(
      attn_forward_kernel,
      sm_scale=sm_scale,
      block_k=block_k,
  )

  o, l, m = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=[
          pl.BlockSpec((block_h, head_dim), lambda i, j: (i, 0)),
          pl.BlockSpec((None, k_seq_len, head_dim), lambda i, j: (j, 0, 0)),
          pl.BlockSpec((None, k_seq_len, head_dim), lambda i, j: (j, 0, 0)),
      ],
      out_specs=[
          pl.BlockSpec((None, block_h, head_dim), lambda i, j: (j, i, 0)),  # o
          pl.BlockSpec((None, block_h), lambda i, j: (j, i)),  # l
          pl.BlockSpec((None, block_h), lambda i, j: (j, i)),  # m
      ],
      compiler_params=dict(
          triton=dict(num_warps=num_warps_, num_stages=num_stages)
      ),
      out_shape=[
          jax.ShapeDtypeStruct(shape=(k_splits, *q.shape), dtype=q.dtype),  # o
          jax.ShapeDtypeStruct(
              shape=(k_splits, num_heads), dtype=jnp.float32
          ),  # l
          jax.ShapeDtypeStruct(
              shape=(k_splits, num_heads), dtype=jnp.float32
          ),  # m
      ],
      debug=debug,
      interpret=interpret,
      name="mha_forward",
  )(q, k, v)

  # final round of flash
  m_next = m.max(axis=0)
  correction = jnp.exp(m - m_next[None])
  o = o * correction[:, :, None]
  l_next = (l * correction).sum(axis=0)
  o = o.sum(axis=0) / l_next[:, None]

  if original_num_heads is not None:
    o = o[:original_num_heads, :]
  return o


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "block_h",
        "block_k",
        "k_splits",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
    ],
)
def mqa(
    q,  # [batch_size, num_heads, head_dim]
    k,  # [batch_size, k_seq_len, head_dim]
    v,  # [batch_size, k_seq_len, head_dim]
    sm_scale: float = 1.0,
    block_h: int = 16,
    block_k: int = 256,
    k_splits: int = 16,
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
):
  inner = functools.partial(
      attn_unbatched,
      sm_scale=sm_scale,
      block_h=block_h,
      block_k=block_k,
      k_splits=k_splits,
      num_warps=num_warps,
      num_stages=num_stages,
      grid=grid,
      interpret=interpret,
      debug=debug,
  )
  return jax.vmap(inner)(q, k, v)


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "block_h",
        "block_k",
        "k_splits",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
    ],
)
def gqa(
    q,  # [batch_size, num_q_heads, head_dim]
    k,  # [batch_size, k_seq_len, num_kv_heads, head_dim]
    v,  # [batch_size, k_seq_len, num_kv_heads, head_dim]
    sm_scale: float = 1.0,
    block_h: int = 16,
    block_k: int = 256,
    k_splits: int = 16,
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
):
  batch_size, q_heads, head_dim = q.shape
  kv_heads = k.shape[2]
  assert kv_heads == v.shape[2]
  assert q_heads % kv_heads == 0
  q_heads_per_kv_head = q_heads // kv_heads
  q_reshaped = q.reshape(batch_size, kv_heads, q_heads_per_kv_head, head_dim)
  k_transposed = jnp.swapaxes(
      k, 1, 2
  )  # [batch_size, num_kv_heads, k_seq_len, head_dim]
  v_transposed = jnp.swapaxes(
      v, 1, 2
  )  # [batch_size, num_kv_heads, k_seq_len, head_dim]
  inner = functools.partial(
      attn_unbatched,
      sm_scale=sm_scale,
      block_h=block_h,
      block_k=block_k,
      k_splits=k_splits,
      num_warps=num_warps,
      num_stages=num_stages,
      grid=grid,
      interpret=interpret,
      debug=debug,
  )
  with_kv_heads = jax.vmap(inner)
  o = jax.vmap(with_kv_heads)(q_reshaped, k_transposed, v_transposed)
  return o.reshape(batch_size, q_heads, head_dim)


@functools.partial(jax.jit, static_argnames=["sm_scale"])
def mqa_reference(
    q,  # [bs, num_q_heads, head_dim]
    k,  # [bs, k_seq_len, head_dim]
    v,  # [bs, k_seq_len, head_dim]
    sm_scale=1.0,
):
  logits = jnp.einsum("bnd,bsd->bns", q, k).astype(jnp.float32)
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum("bns,bsd->bnd", weights, v)


@functools.partial(jax.jit, static_argnames=["sm_scale"])
def mha_reference(
    q,  # [bs, num_q_heads, head_dim]
    k,  # [bs, k_seq_len, num_k_heads, head_dim]
    v,  # [bs, k_seq_len, num_v_heads, head_dim]
    sm_scale=1.0,
):
  assert q.shape[1] == k.shape[2]
  logits = jnp.einsum("bnd,bsnd->bns", q, k).astype(jnp.float32)
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum("bns,bsnd->bnd", weights, v)


@functools.partial(jax.jit, static_argnames=["sm_scale"])
def gqa_reference(
    q,  # [bs, num_q_heads, head_dim]
    k,  # [bs, k_seq_len, num_k_heads, head_dim]
    v,  # [bs, k_seq_len, num_v_heads, head_dim]
    sm_scale=1.0,
):
  bs, num_q_heads, head_dim = q.shape
  num_kv_heads = k.shape[2]
  assert num_q_heads % num_kv_heads == 0
  q_reshaped = q.reshape(
      bs, num_kv_heads, num_q_heads // num_kv_heads, head_dim
  )
  k_transposed = jnp.swapaxes(
      k, 1, 2
  )  # [batch_size, num_kv_heads, k_seq_len, head_dim]
  v_transposed = jnp.swapaxes(
      v, 1, 2
  )  # [batch_size, num_kv_heads, k_seq_len, head_dim]
  logits = jnp.einsum("bkgd,bksd->bkgs", q_reshaped, k_transposed).astype(
      jnp.float32
  )
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  o = jnp.einsum("bkgs,bksd->bkgd", weights, v_transposed)
  return o.reshape(bs, num_q_heads, head_dim)
