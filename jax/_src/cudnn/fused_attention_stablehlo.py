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

from enum import Enum
from functools import partial, reduce
import operator
from typing import Optional
import json

import jax
import jax.numpy as jnp
from jax import core
from jax import dtypes
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters.mlir import ir
from jax.interpreters.mlir import hlo
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import PartitionSpec, NamedSharding

from jax._src import dispatch
from jax._src.interpreters import batching
from jax._src.lib import cuda_versions

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray

class AttentionLayout(Enum):
  BTNH = 0
  BNTH = 1

class MaskType(Enum):
  NO_MASK = 0
  PADDING = 1
  CAUSAL = 2
  PADDING_CAUSAL = 3
  ALIBI = 4

  @property
  def has_padding(self) -> bool:
    return self == MaskType.PADDING or self == MaskType.PADDING_CAUSAL

def should_export_dbias(bias_shape, query_shape, layout) -> bool:
  b_B, b_N, _, _ = bias_shape
  if layout == AttentionLayout.BNTH.value:
    _, q_N, _, _ = query_shape
  else:
    _, _, q_N, _ = query_shape
  return b_B == 1 and b_N == q_N

def _normalize_layout(layout: str) -> AttentionLayout:
  layout_upper = layout.upper()
  if layout_upper in ["BSNH", "BNSH", "BTNH", "BNTH"]:
    return AttentionLayout[layout_upper.replace("S", "T")]
  else:
    raise ValueError(f"Unsupported qkv_layout: {layout}")

def element_type_to_backend_config_type_mapping(dtype):
  _element_type_to_backend_config_type_mapping = {
    ir.BF16Type.get(): "BF16",
    ir.F16Type.get(): "F16",
  }
  return _element_type_to_backend_config_type_mapping[dtype]

def default_layouts(*shapes):
  return [range(len(shape) - 1, -1, -1) for shape in shapes]

def create_dot_product_attention_backend_config(batch,
                                                num_heads,
                                                seq_q,
                                                seq_kv,
                                                dtype,
                                                fmha_scale,
                                                seed,
                                                dropout_rate,
                                                mask_type,
                                                layout,
                                                is_bwd):
  # Q, K, V: query, key, value in shape of BT(S)NH or BNT(S)H
  # P: BMM1 output in shape of BNTS
  # O: BMM2 output in the same shape with Q
  # BMM1: Q @ K -> P
  # BMM2: P @ V -> O
  # BMM1Grad1: dP @ Q -> dK
  # BMM1Grad2: dP @ K -> dQ
  # BMM2Grad1: P @ dO -> dV
  # BMM2Grad2: dO @ V -> dP

  cudnn_fmha_backend_config = {
    "algorithm": {
      "algo_id": "0",
      "math_type": "TENSOR_OP_MATH",
      "tuning_knobs": {"17": "1", "24": "0"},
      "is_cudnn_frontend": True,
      "workspace_size": "0",
    },
    "fmha_scale": fmha_scale,
    "dropout_rate": dropout_rate,
    "intermediate_tensor_shape": {
      "element_type": element_type_to_backend_config_type_mapping(dtype),
      "dimensions": [str(batch), str(num_heads), str(seq_q), str(seq_kv)],
      "tuple_shapes": [],
      "layout": {
        "dim_level_types": [],
        "dim_unique": [],
        "dim_ordered": [],
        "minor_to_major": ["3", "2", "1", "0"],
        "tiles": [],
        "element_size_in_bits": "0",
        "memory_space": "0",
        "index_primitive_type": "PRIMITIVE_TYPE_INVALID",
        "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID",
        "dynamic_shape_metadata_prefix_bytes": "0",
      },
      "is_dynamic_dimension": [False, False, False, False],
    },
    "seed": seed,
    "is_flash_attention": True,
    "mask_type": mask_type.name,
  }

  # We define the contracting and batch dims in the format of
  # ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
  # rhs_batch_dims)).
  if layout == AttentionLayout.BNTH.value:
    dims = [
        ((3, 3), ((0, 1), (0, 1))), # BMM1: BNTH,BNSH->BNTS
        ((3, 2), ((0, 1), (0, 1))), # BMM2: BNTS,BNSH->BNTH
        ((2, 2), ((0, 1), (0, 1))), # BMM1_grad_1: BNTS,BNTH->BNSH
        ((3, 2), ((0, 1), (0, 1))), # BMM1_grad_2: BNTS,BNSH->BNTH
        ((2, 2), ((0, 1), (0, 1))), # BMM2_grad_1: BNTS,BNTH->BNSH
        ((3, 3), ((0, 1), (0, 1))), # BMM2_grad_2: BNTH,BNSH->BNTS
    ]
  else:
    dims = [
        ((3, 3), ((0, 2), (0, 2))), # BMM1: BTNH,BSNH->BNTS
        ((3, 1), ((0, 1), (0, 2))), # BMM2: BNTS,BSNH->BTNH
        ((2, 1), ((0, 1), (0, 2))), # BMM1_grad_1: BNTS,BTNH->BSNH
        ((3, 1), ((0, 1), (0, 2))), # BMM1_grad_2: BNTS,BSNH->BTNH
        ((2, 1), ((0, 1), (0, 2))), # BMM2_grad_1: BNTS,BTNH->BSNH
        ((3, 3), ((0, 2), (0, 2))), # BMM2_grad_2: BTNH,BSNH->BNTS
    ]
  keys = [
      "bmm1_dot_dimension_numbers",
      "bmm2_dot_dimension_numbers",
      "bmm1_grad_gemm1_dot_dimension_numbers",
      "bmm1_grad_gemm2_dot_dimension_numbers",
      "bmm2_grad_gemm1_dot_dimension_numbers",
      "bmm2_grad_gemm2_dot_dimension_numbers",
  ]
  fwd_dot_number = {}
  bwd_dot_number = {}
  for idx, (key, ((lc, rc), (lb, rb))) in enumerate(zip(keys, dims)):
    dims_to_write = fwd_dot_number if idx < 2 else bwd_dot_number
    dims_to_write[key] = {
        "lhs_contracting_dimensions": [str(lc)],
        "rhs_contracting_dimensions": [str(rc)],
        "lhs_batch_dimensions": [str(i) for i in lb],
        "rhs_batch_dimensions": [str(i) for i in rb],
    }

  if is_bwd:
    cudnn_fmha_backend_config = {**cudnn_fmha_backend_config, **bwd_dot_number}
  else:
    cudnn_fmha_backend_config = {**cudnn_fmha_backend_config, **fwd_dot_number}

  backend_config = {
    "operation_queue_id":"0",
    "wait_on_operation_queues":[],
    "cudnn_fmha_backend_config": cudnn_fmha_backend_config
  }
  backend_config = json.dumps(backend_config)
  return backend_config

# mapping from (is_bwd, has_dropout, has_mask, has_bias) to custom call name
_custom_name_maps = {
  # fMHA forward call targets.
  (False, False, False): "__cudnn$fmhaSoftmax",
  (False, False, True): "__cudnn$fmhaScaleBiasSoftmax",
  (False, True, False): "__cudnn$fmhaSoftmaxDropout",
  (False, True, True): "__cudnn$fmhaScaleBiasSoftmaxDropout",
  # fMHA backward call targets.
  (True, False, False): "__cudnn$fmhaSoftmaxBackward",
  (True, False, True): "__cudnn$fmhaScaleBiasSoftmaxBackward",
  (True, True, False): "__cudnn$fmhaSoftmaxDropoutBackward",
  (True, True, True): "__cudnn$fmhaScaleBiasSoftmaxDropoutBackward",
}

def get_custom_call_name(has_bias, has_dropout, is_bwd):
  return _custom_name_maps[(is_bwd, has_dropout, has_bias)]

def check_layout(query, key, value, bias, q_seqlen, kv_seqlen, layout):
  def check_eq(a, b, c, msg):
    if not (a == b == c):
      raise ValueError(f"{msg} must be same, got {a}, {b}, {b}")

  q_rank, k_rank, v_rank = len(query.shape), len(key.shape), len(value.shape)
  if q_rank != 4:
    raise ValueError(f"Q must have a rank of 4, got {q_rank}")
  check_eq(q_rank, k_rank, v_rank, "QKV rank")

  q_dtype, k_dtype, v_dtype = query.dtype, key.dtype, value.dtype
  assert q_dtype in [jnp.float16, jnp.bfloat16], "Q must be fp16 or bf16"
  check_eq(q_dtype, k_dtype, v_dtype, "QKV dtype")

  if layout == AttentionLayout.BNTH:
    qB, qN, qT, qH = query.shape
    kB, kN, kS, kH = key.shape
    vB, vN, vS, vH = value.shape
  else:
    assert layout == AttentionLayout.BTNH
    qB, qT, qN, qH = query.shape
    kB, kS, kN, kH = key.shape
    vB, vS, vN, vH = value.shape

  check_eq(qB, kB, vB, "QKV batch")
  check_eq(qH, kH, vH, "QKV dim_per_head")
  if kN != vN:
    raise ValueError(f"KV must have same number of heads, got {kN} vs {vN}")
  if kS != vS:
    raise ValueError(f"KV must have same seq length, got {kS} vs {vS}")

  # check bias/q_seqlen/kv_seqlen
  if bias is not None:
    _, _, bT, bS = bias.shape
    if bT != qT or bS != vS:
      raise ValueError(
        f"Bias must have same seq length as QKV, got {bT} and {bS}")
  if q_seqlen is not None:
    q_seq_dtype = q_seqlen.dtype
    q_seq_rank = len(q_seqlen.shape)
    if q_seq_dtype != jnp.int32:
      raise ValueError(f"q_seqlen must have int32 datatype, got {q_seq_dtype}")
    if q_seq_rank != 1:
      raise ValueError(f"q_seqlen must have a rank of 1, got {q_seq_rank}")
    q_seq_b = q_seqlen.shape[0]
    if q_seq_b != qB:
      raise ValueError(f"q_seqlen must have same batch as Q, got {q_seq_b}")
  if kv_seqlen is not None:
    kv_seq_dtype = kv_seqlen.dtype
    kv_seq_rank = len(kv_seqlen.shape)
    if kv_seq_dtype != jnp.int32:
      raise ValueError(
        f"kv_seqlen must have int32 datatype, got {kv_seq_dtype}")
    if kv_seq_rank != 1:
      raise ValueError(f"kv_seq_rank must have a rank of 1, got {kv_seq_rank}")
    kv_seq_b = kv_seqlen.shape[0]
    if kv_seq_b != qB:
      raise ValueError(f"kv_seqlen must have same batch as Q, got {kv_seq_b}")

def check_is_flash_attention(
    query, key, layout, cudnn_version, has_bias, is_training):
  if layout == AttentionLayout.BNTH:
    _, _, T, H = query.shape
    _, _, S, _ = key.shape
  else:
    _, T, _, H = query.shape
    _, S, _, _ = key.shape

  if not ((H <= 128 and H % 8 == 0) and
        (not is_training or not has_bias or T % 2 == 0 and S % 2 == 0)):
    # check if flash attention is supported
    # for training, for patterns with bias, seqlen should be divisible by 2
    raise NotImplementedError(
      f"Unsupported sequence length Q {T}, KV {S} and head dim {H}.")
  # check if minimum cudnn version requirement is satisfied
  if cudnn_version < 8904:
    raise RuntimeError(
      "JAX requires cuDNN >= 8.9.4 to use flash cross attention.")

def check_cudnn_version():
  # check if cuDNN is installed
  if cuda_versions is None:
    raise RuntimeError("cuDNN is not detected.")
  return cuda_versions.cudnn_get_version()

def check_compute_capability(cc):
  if cuda_versions is None:
    raise RuntimeError("cuDNN is not detected.")
  d, *_ = jax.local_devices(backend="gpu")
  if d.compute_capability not in cc:
    raise RuntimeError("Require compute capability in " + str(cc))

def _dot_product_attention_fwd(
    query, key, value, bias, mask, q_seqlen, kv_seqlen, scale, seed,
    dropout_rate, variadic_args, mask_type, layout, is_training):
  outputs = _dot_product_attention_fwd_p_wrapper.bind(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, scale=scale,
      seed=seed, dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout, is_training=is_training)
  output = outputs[0]
  return output

def _dot_product_attention_fwd_rule(
    query, key, value, bias, mask, q_seqlen, kv_seqlen, scale, seed,
    dropout_rate, variadic_args, mask_type, layout, is_training):
  outputs = _dot_product_attention_fwd_p_wrapper.bind(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, scale=scale,
      seed=seed, dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout, is_training=is_training)
  res = (query, key, value, bias, mask, q_seqlen, kv_seqlen,
         outputs[1], outputs[0]) if is_training else None
  return outputs[0], res

def _dot_product_attention_bwd_rule(
    scale, seed, dropout_rate, variadic_args, mask_type, layout, is_training,
    res, grad_output):
  (query, key, value, bias, mask, q_seqlen, kv_seqlen, activation,
   fwd_output) = res
  grads = _dot_product_attention_bwd_p_wrapper.bind(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, activation,
      fwd_output, grad_output, scale=scale, seed=seed,
      dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout,
  )
  grads = (*grads,) + (None,) * (7 - len(grads))
  return grads

def _dot_product_attention_fwd_impl(
    query, key, value, bias, mask, q_seqlen, kv_seqlen, scale, seed,
    dropout_rate, variadic_args, mask_type, layout, is_training):
  # args: {Q, K, V, mask*, bias*}
  outputs = _dot_product_attention_fwd_p.bind(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, scale=scale,
      seed=seed, dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout, is_training=is_training)
  return outputs

def _dot_product_attention_bwd_impl(
    query, key, value, bias, mask, q_seqlen, kv_seqlen, activation, fwd_output,
    grad_output, scale, seed, dropout_rate, variadic_args, mask_type, layout):
  grads = _dot_product_attention_bwd_p.bind(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, activation,
      fwd_output, grad_output, scale=scale, seed=seed,
      dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout)
  return grads

def _dot_product_attention_fwd_abstract(
    query, key, value, bias, mask, q_seqlen, kv_seqlen, *, scale, seed,
    dropout_rate, variadic_args, mask_type, layout, is_training):
  query_dtype = dtypes.canonicalize_dtype(query.dtype)
  if layout == AttentionLayout.BNTH.value:
    B, N, T, _ = query.shape
    _, _, S, _ = key.shape
  else:
    B, T, N, _ = query.shape
    _, S, _, _ = key.shape
  output_shape = query.shape
  softmax_stat_shape = (B, N, T)

  if is_training:
    return (
      core.ShapedArray(output_shape, query_dtype),  # output
      core.ShapedArray(softmax_stat_shape, jnp.float32),  # softmax_stat
    )
  else:
    return (
      core.ShapedArray(output_shape, query_dtype),  # output
    )

def _dot_product_attention_bwd_abstract(
    query, key, value, bias, mask, q_seqlen, kv_seqlen, activation, fwd_output,
    grad_output, *, scale, seed, dropout_rate, variadic_args, mask_type,
    layout):
  query_dtype = dtypes.canonicalize_dtype(query.dtype)
  key_dtype = dtypes.canonicalize_dtype(key.dtype)
  value_dtype = dtypes.canonicalize_dtype(value.dtype)

  _, _, has_dbias = variadic_args
  if has_dbias:
    # cuDNN supports bias for this case
    bias_dtype = dtypes.canonicalize_dtype(bias.dtype)
    return (
      core.ShapedArray(
          query.shape, query_dtype
      ),  # grad query
      core.ShapedArray(
          key.shape, key_dtype
      ),  # grad key
      core.ShapedArray(
          value.shape, value_dtype
      ),  # grad value
      core.ShapedArray(
          bias.shape, bias_dtype
      ),  # grad bias
    )
  else:
    return (
      core.ShapedArray(
          query.shape, query_dtype
      ),  # grad query
      core.ShapedArray(
          key.shape, key_dtype
      ),  # grad key
      core.ShapedArray(
          value.shape, value_dtype
      ),  # grad value
    )

def _dot_product_attention_fwd_cuda_lowering(
    ctx, query, key, value, bias, mask, q_seqlen, kv_seqlen, scale, seed,
    dropout_rate, variadic_args, mask_type, layout, is_training):
  query_type = ir.RankedTensorType(query.type)
  query_shape = query_type.shape
  key_type = ir.RankedTensorType(key.type)
  key_shape = key_type.shape

  if layout == AttentionLayout.BNTH.value:
    B, N, T, H = query_shape
    _, _, S, _ = key_shape
    output_layout = (3, 2, 1, 0)
    output_transpose_perm = mlir.dense_int_array((0, 1, 2, 3))
  else:
    B, T, N, H = query_shape
    _, S, _, _ = key_shape
    output_layout = (3, 1, 2, 0)
    output_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))

  output_shape = (B, N, T, H)
  softmax_stat_shape = (B, N, T)
  workspace_shape = (0,)
  workspace_type = ir.IntegerType.get_unsigned(8)
  backend_config = create_dot_product_attention_backend_config(
      B, N, T, S, query_type.element_type, scale, seed, dropout_rate,
      mask_type, layout, is_bwd=False,
  )
  # {Q, K, V, mask*, bias*, q_seqlen*, kv_seqlen*}
  # {output, activation*, workspace}
  has_dropout = dropout_rate > 0
  has_bias, has_mask, _ = variadic_args
  operands = [query, key, value]
  if has_mask:
    operands.append(mask)
  if has_bias:
    operands.append(bias)
  if mask_type.has_padding:
    operands.append(q_seqlen)
    operands.append(kv_seqlen)
  custom_call_name = get_custom_call_name(has_bias, has_dropout, False)
  # create output types and layouts
  if is_training:
    result_types = [
      ir.RankedTensorType.get(output_shape, query_type.element_type),
      ir.RankedTensorType.get(softmax_stat_shape, ir.F32Type.get()),
      ir.RankedTensorType.get(workspace_shape, workspace_type),
    ]
    result_layouts = [output_layout] + default_layouts(softmax_stat_shape, workspace_shape)
  else:
    result_types = [
      ir.RankedTensorType.get(output_shape, query_type.element_type),
      ir.RankedTensorType.get(workspace_shape, workspace_type)
    ]
    result_layouts = [output_layout] + default_layouts(workspace_shape)
  # create custom call here
  out = mlir.custom_call(
    custom_call_name,
    result_types=result_types,
    operands=operands,
    backend_config=backend_config,
    operand_layouts=default_layouts(
      *[ir.RankedTensorType(operand.type).shape for operand in operands]),
    result_layouts=result_layouts,
  )
  # drop workspace memory
  # output should be (B, T, N, H) instead of (B, N, T, H)
  if is_training:
    return [hlo.transpose(out.results[0], output_transpose_perm), out.results[1]]
  else:
    return [hlo.transpose(out.results[0], output_transpose_perm)]

def _dot_product_attention_bwd_cuda_lowering(
    ctx, query, key, value, bias, mask, q_seqlen, kv_seqlen, activation,
    fwd_output, grad_output, scale, seed, dropout_rate, variadic_args,
    mask_type, layout):
  query_type = ir.RankedTensorType(query.type)
  query_shape = query_type.shape
  key_type = ir.RankedTensorType(key.type)
  key_shape = key_type.shape
  value_type = ir.RankedTensorType(value.type)

  if layout == AttentionLayout.BNTH.value:
    B, q_N, T, H = query_shape
    _, k_N, S, _ = key_shape
    grad_layout = (3, 2, 1, 0)
    grad_transpose_perm = mlir.dense_int_array((0, 1, 2, 3))
  else:
    B, T, q_N, H = query_shape
    _, S, k_N, _ = key_shape
    grad_layout = (3, 1, 2, 0)
    grad_transpose_perm = mlir.dense_int_array((0, 2, 1, 3))

  workspace_shape = (0,)
  workspace_type = ir.IntegerType.get_unsigned(8)

  grad_query_shape = (B, q_N, T, H)
  grad_key_shape = (B, k_N, S, H)
  grad_value_shape = (B, k_N, S, H)
  backend_config = create_dot_product_attention_backend_config(
      B, q_N, T, S, query_type.element_type, scale, seed, dropout_rate,
      mask_type, layout, is_bwd=True,
  )
  # {Q, K, V, activation, dO, mask*, bias*, O, q_seqlen*, kv_seqlen*}
  # {dQ, dK, dV, dbias*, workspace}
  has_dropout = dropout_rate > 0
  has_bias, has_mask, has_dbias = variadic_args
  # create operands
  operands = [query, key, value, activation, grad_output]
  if has_mask:
    operands.append(mask)
  if has_bias:
    # flash attention requires bias in the bwd for remat
    operands.append(bias)
  operands.append(fwd_output)
  if mask_type.has_padding:
    operands.append(q_seqlen)
    operands.append(kv_seqlen)
  # get custom call name
  custom_call_name = get_custom_call_name(has_bias, has_dropout, True)

  # create output types and layouts
  # grad_query, grad_key, grad_value
  result_types = [
    ir.RankedTensorType.get(grad_query_shape, query_type.element_type),
    ir.RankedTensorType.get(grad_key_shape, key_type.element_type),
    ir.RankedTensorType.get(grad_value_shape, value_type.element_type),
  ]
  result_layouts = [grad_layout, grad_layout, grad_layout]
  bias_type = ir.RankedTensorType(bias.type)
  bias_shape = bias_type.shape
  if has_dbias:
    # cuDNN supports bias for this case
    result_types.append(
      ir.RankedTensorType.get(bias_shape, bias_type.element_type))
    result_layouts = result_layouts + default_layouts(bias_shape)
  # workspace
  result_types.append(ir.RankedTensorType.get(workspace_shape, workspace_type))
  result_layouts = result_layouts + default_layouts(workspace_shape)
  out = mlir.custom_call(
    custom_call_name,
    result_types=result_types,
    operands=operands,
    backend_config=backend_config,
    operand_layouts=default_layouts(
      *[ir.RankedTensorType(operand.type).shape for operand in operands]),
    result_layouts=result_layouts,
  )
  dqkv = (hlo.transpose(out.results[0], grad_transpose_perm),
          hlo.transpose(out.results[1], grad_transpose_perm),
          hlo.transpose(out.results[2], grad_transpose_perm))
  # Only keep dQ, dK, dV and dBias here
  if has_dbias:
    return dqkv + (out.results[3],)
  else:
    return dqkv

# batcher
def _check_valid_batch_dims(bdims):
  for dim in bdims:
    if dim not in [0, None]:
      raise NotImplementedError(
        f"Currently only support batch_dim in [0, None], but got {dim=}")

def _dot_product_attention_fwd_batcher(
    batched_args, batch_dims, *, scale, seed, dropout_rate, variadic_args,
    mask_type, layout, is_training):
  _check_valid_batch_dims(batch_dims)
  query, key, value, bias, mask, q_seqlen, kv_seqlen = batched_args
  query_bdim = batch_dims[0]
  if is_training:
    out_bdims = query_bdim, query_bdim
  else:
    out_bdims = (query_bdim,)

  if layout == AttentionLayout.BNTH.value:
    *Bs, N, T, _ = query.shape
    *_, _, S, _ = key.shape
  else:
    *Bs, T, N, _ = query.shape
    *_, S, _, _ = key.shape
  B = reduce(operator.mul, Bs)
  has_bias, has_mask, _ = variadic_args
  # reshape to 4D shape
  query = jnp.reshape(query, (B,) + query.shape[-3:])
  key = jnp.reshape(key, (B,) + key.shape[-3:])
  value = jnp.reshape(value, (B,) + key.shape[-3:])
  if has_bias:
    bias = jnp.reshape(bias, (B, N, T, S))
  if has_mask:
    mask = jnp.reshape(mask, (B, N, T, S))
  if mask_type.has_padding:
    q_seqlen = jnp.reshape(q_seqlen, (B, ))
    kv_seqlen = jnp.reshape(kv_seqlen, (B, ))

  outputs = _dot_product_attention_fwd_p_wrapper.bind(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, scale=scale,
      seed=seed, dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout, is_training=is_training)

  # reshape to original shape
  output = outputs[0]
  output = jnp.reshape(output, query.shape)
  if is_training:
    activation = outputs[1]
    activation = jnp.reshape(activation, (*Bs, N, T))
    return (output, activation), out_bdims
  else:
    return (output,), out_bdims

def _dot_product_attention_bwd_batcher(
     batched_args, batch_dims, *, scale, seed, dropout_rate, variadic_args,
     mask_type, layout):
  _check_valid_batch_dims(batch_dims)
  query, key, value, bias, mask, q_seqlen, \
    kv_seqlen, activation, fwd_output, grad_output = batched_args
  query_bdim = batch_dims[0]
  out_bdims = query_bdim, query_bdim, query_bdim

  if layout == AttentionLayout.BNTH.value:
    *Bs, N, T, _ = query.shape
    *_, _, S, _ = key.shape
  else:
    *Bs, T, N, _ = query.shape
    *_, S, _, _ = key.shape
  B = reduce(operator.mul, Bs)
  has_bias, has_mask, has_dbias = variadic_args
  # reshape to 4D shape
  query = jnp.reshape(query, (B,) + query.shape[-3:])
  key = jnp.reshape(key, (B,) + key.shape[-3:])
  value = jnp.reshape(value, (B,) + key.shape[-3:])
  if has_bias:
    bias = jnp.reshape(bias, (B, N, T, S))
  if has_mask:
    mask = jnp.reshape(mask, (B, N, T, S))
  if mask_type.has_padding:
    q_seqlen = jnp.reshape(q_seqlen, (B, ))
    kv_seqlen = jnp.reshape(kv_seqlen, (B, ))

  activation = jnp.reshape(activation, (B, N, T))
  fwd_output = jnp.reshape(fwd_output, (B,) + query.shape[-3:])
  grad_output = jnp.reshape(grad_output, (B,) + query.shape[-3:])

  grads = _dot_product_attention_bwd_p_wrapper.bind(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, activation,
      fwd_output, grad_output, scale=scale, seed=seed,
      dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout,
  )

  grad_query, grad_key, grad_value = grads[:3]
  # reshape to original shape
  grad_query = jnp.reshape(grad_query, query.shape)
  grad_key = jnp.reshape(grad_key, key.shape)
  grad_value = jnp.reshape(grad_value, value.shape)
  if has_dbias:
    grad_bias = grads[3]
    grad_bias = jnp.reshape(grad_bias, bias.shape)
    return grads + (grad_bias,), out_bdims + (query_bdim,)
  return grads, out_bdims

# custom partitioning
def _get_padded_spec(arg_info):
  spec = None if arg_info.sharding is None else arg_info.sharding.spec
  ndim = arg_info.ndim
  if spec is None:
    return (None,) * ndim
  assert len(spec) <= ndim
  return spec + (None,) * (ndim - len(spec))

def _check_qkv_bias_mask_spec(
    query_spec, key_spec, value_spec, bias_spec, mask_spec):
  # check qkv spec
  if not query_spec == key_spec == value_spec:
    raise ValueError("Query, key and value should have same sharding.")
  *batch_spec, q_seq_spec, num_head_spec, head_spec = query_spec
  if q_seq_spec is not None:
    raise ValueError("Sharding on sequence dim is not allowed.")
  if head_spec is not None:
    raise ValueError("Sharding on head dim is not allowed.")
  # check bias and mask spec
  if bias_spec:
    *bias_batch_spec, bias_num_head_spec, bias_q_seq_spec, bias_kv_seq_spec = bias_spec
    if bias_batch_spec != batch_spec or bias_num_head_spec != num_head_spec:
      raise ValueError(
        "Query and bias should have same sharding on batch and num_head dim.")
    if bias_q_seq_spec is not None or bias_kv_seq_spec is not None:
      raise ValueError("Sharding on bias sequence dim is not allowed.")
  if mask_spec:
    *mask_batch_spec, mask_num_head_spec, mask_q_seq_spec, mask_kv_seq_spec = mask_spec
    if mask_batch_spec != batch_spec or mask_num_head_spec != num_head_spec:
      raise ValueError(
        "Query and mask should have same sharding on batch and num_head dim.")
    if mask_q_seq_spec is not None or mask_kv_seq_spec is not None:
      raise ValueError("Sharding on mask sequence dim is not allowed.")

# fwd custom partition
def _infer_fwd_output_sharding(mesh, arg_shapes, variadic_args, is_training):
  # only sharding on batch and num_head dim is allowed
  # (*batch, q_seq, num_head, head)
  query_spec = _get_padded_spec(arg_shapes[0])
  # (*batch, kv_seq, num_head, head)
  key_spec = _get_padded_spec(arg_shapes[1])
  value_spec = _get_padded_spec(arg_shapes[2])
  has_bias, has_mask, _ = variadic_args
  bias_spec = _get_padded_spec(arg_shapes[3]) if has_bias else None
  mask_spec = _get_padded_spec(arg_shapes[4]) if has_mask else None
  _check_qkv_bias_mask_spec(
    query_spec, key_spec, value_spec, bias_spec, mask_spec)
  # keep out sharding same as query sharding since they have same shape
  out_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
  if is_training:
    # activation sharding
    *batch_spec, q_seq_spec, num_head_spec, _ = query_spec
    activation_sharding = NamedSharding(
      mesh, PartitionSpec(*batch_spec, num_head_spec, q_seq_spec, None))
    return [out_sharding, activation_sharding]
  return [out_sharding]

_dot_product_attention_fwd_lower = custom_partitioning(
    _dot_product_attention_fwd_impl, static_argnums=(7, 8, 9, 10, 11, 12, 13))

def _dot_product_attention_fwd_infer_sharding_from_operands(
    scale, seed, dropout_rate, variadic_args, mask_type, layout, is_training,
    mesh, arg_shapes, result_shape):
  return _infer_fwd_output_sharding(mesh, arg_shapes, variadic_args, is_training)

def _dot_product_attention_fwd_partition(
    scale, seed, dropout_rate, variadic_args, mask_type, layout, is_training,
    mesh, arg_shapes, result_shape):
  # args sharding
  arg_shardings = tuple([arg_i.sharding for arg_i in arg_shapes])
  out_shardings = _infer_fwd_output_sharding(
    mesh, arg_shapes, variadic_args, is_training)
  impl = partial(
      _dot_product_attention_fwd_impl, scale=scale, seed=seed,
      dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type,
      layout=layout, is_training=is_training)
  return mesh, impl, out_shardings, arg_shardings

# bwd custom partition
def _infer_bwd_output_sharding(mesh, arg_shapes, variadic_args):
  # (*batch, q_seq, num_head, head)
  query_spec = _get_padded_spec(arg_shapes[0])
  # (*batch, kv_seq, num_head, head)
  key_spec = _get_padded_spec(arg_shapes[1])
  value_spec = _get_padded_spec(arg_shapes[2])
  has_bias, has_mask, has_dbias = variadic_args
  bias_spec = _get_padded_spec(arg_shapes[3]) if has_bias else None
  mask_spec = _get_padded_spec(arg_shapes[4]) if has_mask else None
  _check_qkv_bias_mask_spec(
    query_spec, key_spec, value_spec, bias_spec, mask_spec)
  # keep grad query sharding same as query sharding
  grad_query_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
  grad_key_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
  grad_value_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
  out_shardings = [grad_query_sharding, grad_key_sharding, grad_value_sharding]
  if has_dbias:
    grad_bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_spec))
    out_shardings = out_shardings + [grad_bias_sharding]
  return out_shardings

_dot_product_attention_bwd_lower = custom_partitioning(
    _dot_product_attention_bwd_impl, static_argnums=(10, 11, 12, 13, 14, 15)
)

def _dot_product_attention_bwd_infer_sharding_from_operands(
    scale, seed, dropout_rate, variadic_args, mask_type, layout, mesh,
    arg_shapes, result_shape):
  return _infer_bwd_output_sharding(mesh, arg_shapes, variadic_args)

def _dot_product_attention_bwd_partition(
    scale, seed, dropout_rate, variadic_args, mask_type, layout, mesh,
    arg_shapes, result_shape):
  out_shardings = _infer_bwd_output_sharding(mesh, arg_shapes, variadic_args)
  # args sharding
  arg_shardings = tuple([arg_i.sharding for arg_i in arg_shapes])
  impl = partial(
      _dot_product_attention_bwd_impl, scale=scale, seed=seed,
      dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type,
      layout=layout,
  )
  return mesh, impl, out_shardings, arg_shardings

# Create dot_product_attention_fwd_p for forward operation.
_dot_product_attention_fwd_p = core.Primitive("dot_product_attention_fwd")
_dot_product_attention_fwd_p.multiple_results = True
_dot_product_attention_fwd_p.def_impl(
    partial(xla.apply_primitive, _dot_product_attention_fwd_p)
)
_dot_product_attention_fwd_p.def_abstract_eval(
    _dot_product_attention_fwd_abstract
)

mlir.register_lowering(
  _dot_product_attention_fwd_p,
  _dot_product_attention_fwd_cuda_lowering,
  platform="cuda",
)

_dot_product_attention_fwd_p_wrapper = core.Primitive(
    "dot_product_attention_fwd_wrapper"
)
_dot_product_attention_fwd_p_wrapper.multiple_results = True
_dot_product_attention_fwd_p_wrapper.def_impl(_dot_product_attention_fwd_impl)
_dot_product_attention_fwd_p_wrapper.def_abstract_eval(
    _dot_product_attention_fwd_abstract
)

# Create dot_product_attention_bwd_p for backward operation.
_dot_product_attention_bwd_p = core.Primitive("dot_product_attention_bwd")
_dot_product_attention_bwd_p.multiple_results = True
_dot_product_attention_bwd_p.def_impl(
    partial(xla.apply_primitive, _dot_product_attention_bwd_p)
)
_dot_product_attention_bwd_p.def_abstract_eval(
    _dot_product_attention_bwd_abstract
)

mlir.register_lowering(
  _dot_product_attention_bwd_p,
  _dot_product_attention_bwd_cuda_lowering,
  platform="cuda",
)

_dot_product_attention_bwd_p_wrapper = core.Primitive(
    "dot_product_attention_bwd_wrapper"
)
_dot_product_attention_bwd_p_wrapper.multiple_results = True
_dot_product_attention_bwd_p_wrapper.def_impl(_dot_product_attention_bwd_impl)
_dot_product_attention_bwd_p_wrapper.def_abstract_eval(
    _dot_product_attention_bwd_abstract
)

batching.primitive_batchers[
    _dot_product_attention_fwd_p_wrapper
] = _dot_product_attention_fwd_batcher
batching.primitive_batchers[
    _dot_product_attention_bwd_p_wrapper
] = _dot_product_attention_bwd_batcher

_dot_product_attention_fwd_lower.def_partition(
  infer_sharding_from_operands=_dot_product_attention_fwd_infer_sharding_from_operands,
  partition=_dot_product_attention_fwd_partition)

mlir.register_lowering(_dot_product_attention_fwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_fwd_lower, multiple_results=True))

_dot_product_attention_bwd_lower.def_partition(
  infer_sharding_from_operands=_dot_product_attention_bwd_infer_sharding_from_operands,
  partition=_dot_product_attention_bwd_partition)

mlir.register_lowering(_dot_product_attention_bwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_bwd_lower, multiple_results=True))

dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fwd_p
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fwd_p_wrapper
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_bwd_p
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_bwd_p_wrapper
)

@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _dot_product_attention(query: Array,
                           key: Array,
                           value: Array,
                           bias: Array,
                           mask: Array,
                           q_seqlen: Array,
                           kv_seqlen: Array,
                           scale: float,
                           seed: int,
                           dropout_rate: float,
                           variadic_args: tuple[bool, ...],
                           mask_type: bool,
                           layout: int,
                           is_training: bool):
  output = _dot_product_attention_fwd(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, scale=scale,
      seed=seed, dropout_rate=dropout_rate, variadic_args=variadic_args,
      mask_type=mask_type, layout=layout, is_training=is_training)
  return output

# _dot_product_attention_fwd must have the same func signature as _dot_product_attention
_dot_product_attention.defvjp(_dot_product_attention_fwd_rule, _dot_product_attention_bwd_rule)

# User interface
def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          mask: Optional[Array] = None,
                          q_seqlen: Optional[Array] = None,
                          kv_seqlen: Optional[Array] = None,
                          *,
                          scale: float = 1.0,
                          mask_type: MaskType = MaskType.NO_MASK,
                          seed: int = 42,
                          dropout_rate: float = 0.,
                          qkv_layout: str = "BTNH",
                          is_training = False):
  """Computes dot-product attention given query (Q), key (K), and value (V).

  This function serves as the core operation for applying attention
  mechanisms as described in the paper [https://arxiv.org/abs/1706.03762].
  Initially, it determines the attention weights by processing Q and K,
  subsequently combining the outcomes using K. Throughout this function, we
  utilize the following uppercase letters to represent specific parameters of
  array:

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    N = number of attention heads
    H = dimensions of each attention head.

  The supported layouts for Q, K, V are either BT(S)NH or BNT(S)H, and they must
  adhere to the same layout. The output layout remains consistent with Q,
  defaulting to BT(S)NH.

  Args:
    query: Queries for attention calculation with a shape of BTNH or BNTH.
    key: Keys for attention calculation with a shape of BSNH or BNSH.
    value: Values to be used in attention with a shape of BSNH or BNSH.
    bias: Bias to be added to logits with a shape of BNTS.
    mask: Mask used to filter out logits with a shape of BNTS.
    q_seqlen: Non padded sequence length of Queries with a shape of B.
    kv_seqlen: Non padded sequence length of Keys and Values with a shape of B.
    scale: Scale for the query.
    dropout_rate: Dropout rate.
    qkv_layout: Layout string, with supported formats being BTNH, BNTH, BSNH,
                BNSH.
    is_training: choose to save activation or not.

  Returns:
    Output of the same shape as the query.
  """
  # check if cuDNN is installed
  cudnn_version = check_cudnn_version()
  # only support Ampere and Hopper for now
  check_compute_capability(("8.0", "9.0"))
  layout = _normalize_layout(qkv_layout)
  if bias is not None:
    # reshape bias to have 4D shape
    bias = bias.reshape((1,) * (4 - len(bias.shape)) + bias.shape)
  # check if input shape and data type is compatiable
  check_layout(query, key, value, bias, q_seqlen, kv_seqlen, layout)
  # check if flash attention is supported for this attention pattern
  check_is_flash_attention(
      query, key, layout, cudnn_version, bias is not None, is_training)
  if mask_type.has_padding and (q_seqlen is None or kv_seqlen is None):
    raise ValueError("Require q_seqlen and kv_seqlen to generate padding mask")
  has_bias = bias is not None
  has_mask = mask is not None
  has_dbias = has_bias and is_training and \
    should_export_dbias(bias.shape, query.shape, layout)
  variadic_args = (has_bias, has_mask, has_dbias)
  if bias is None:
    bias = jnp.zeros(0, dtype=query.dtype)
  if mask is None:
    mask = jnp.zeros(0, dtype=query.dtype)
  if q_seqlen is None:
    q_seqlen = jnp.zeros(0, dtype=query.dtype)
  if kv_seqlen is None:
    kv_seqlen = jnp.zeros(0, dtype=query.dtype)
  output = _dot_product_attention(
      query, key, value, bias, mask, q_seqlen, kv_seqlen, scale, seed,
      dropout_rate, variadic_args, mask_type, layout.value, is_training
  )
  return output
