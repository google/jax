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
ONES = jnp.ones((1,1,1,1), dtype=jnp.float32)

class AttentionLayout(Enum):
  BTNH = 0
  BNTH = 1

class MaskType(Enum):
  NO_MASK = 0
  CAUSAL = 1

def convert_mask_type_to_string(mask_type: MaskType) -> str:
  if mask_type == MaskType.NO_MASK:
    return "NO_MASK"
  elif mask_type == MaskType.CAUSAL:
    return "CAUSAL"
  else:
    raise ValueError(f"Unexpected mask type: {mask_type}")

def _normalize_layout(layout: str) -> AttentionLayout:
  layout_upper = layout.upper()
  if layout_upper in ["BSNH", "BNSH", "BTNH", "BNTH"]:
    return AttentionLayout[layout_upper.replace("S", "T")]
  else:
    raise ValueError(f"Unsupported qkv_layout: {layout}")

def element_type_to_backend_config_type_mapping(dtype):
  _element_type_to_backend_config_type_mapping = {
    ir.Float8E4M3FNType.get(): "Float8E4M3FN",
    ir.Float8E5M2Type.get(): "Float8E5M2",
    ir.BF16Type.get(): "BF16",
    ir.F16Type.get(): "F16",
    ir.F32Type.get(): "F32",
  }
  return _element_type_to_backend_config_type_mapping[dtype]

def default_layouts(*shapes):
  return [range(len(shape) - 1, -1, -1) for shape in shapes]

def check_layout(query, key, value, layout):
  def check_eq(a, b, c, msg):
    if not (a == b == c):
      raise ValueError(f"{msg} must be same, got {a}, {b}, {b}")

  q_rank, k_rank, v_rank = len(query.shape), len(key.shape), len(value.shape)
  if q_rank != 4:
    raise ValueError(f"Q must have a rank of 4, got {q_rank}")
  check_eq(q_rank, k_rank, v_rank, "QKV rank")

  q_dtype, k_dtype, v_dtype = query.dtype, key.dtype, value.dtype
  assert q_dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2], "Q must be fp8"
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


# mapping from (is_bwd, has_mask) to custom call name
_custom_name_maps = {
  # fMHA forward call targets.
  (False,): "__cudnn$fmhaSoftmaxF8",
  # fMHA backward call targets.
  (True,): "__cudnn$fmhaSoftmaxBackwardF8",
}

def get_custom_call_name(is_bwd):
  return _custom_name_maps[(is_bwd,)]

def create_dot_product_attention_fp8_backend_config(batch,
                                                    num_heads,
                                                    seq_q,
                                                    seq_kv,
                                                    dtype,
                                                    fmha_scale,
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
    "is_flash_attention": True,
    "mask_type": convert_mask_type_to_string(mask_type),
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

def check_is_flash_attention(
    query, key, layout:int, cudnn_version, is_training):
  # check if minimum cudnn version requirement is satisfied
  if cudnn_version < 9010:
    raise RuntimeError(
      "JAX requires cuDNN >= 9.1.0 to use flash cross attention.")
  
  if layout == AttentionLayout.BNTH.value:
    _, _, T, H = query.shape
    _, _, S, _ = key.shape
  else:
    _, T, _, H = query.shape
    _, S, _, _ = key.shape

  # requires that all seq_len % 128 and only H=128 for bprop. For fprop it is all H <= 256 with H % 16 == 0
  if not ((is_training and H == 128 and T % 128 == 0 and S % 128 == 0) or (not is_training and H <= 256 and H % 16 == 0)):
    raise NotImplementedError(
      f"Unsupported sequence length Q {T}, KV {S} and head dim {H}.")
    
def check_cudnn_version():
  # check if cuDNN is installed
  if cuda_versions is None:
    raise RuntimeError("cuDNN is not detected.")
  return cuda_versions.cudnn_get_version()

def check_compute_capability(cc):
  if cuda_versions is None:
    raise RuntimeError("cuDNN is not detected.")
  for i in range(jax.device_count()):
    compute_cap = cuda_versions.cuda_compute_capability(i)
    if compute_cap not in cc:
      raise RuntimeError("Require compute capability in " + str(cc))

def _dot_product_attention_fp8_fwd(
    query, key, value,
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
    scale, use_causal_mask, layout, cudnn_version):
  # check if flash attention is supported for this attention pattern
  check_is_flash_attention(
      query, key, layout, cudnn_version, False)
  outputs = _dot_product_attention_fp8_fwd_p_wrapper.bind(
      query, key, value, 
      descale_q, descale_k, descale_v, descale_s,
      scale_s, scale_o, 
      scale=scale, use_causal_mask=use_causal_mask, layout=layout, is_training=False)
  return outputs

def _dot_product_attention_fp8_fwd_rule(
    query, key, value, 
    amax_dQ, amax_dK, amax_dV, amax_dP,  # place holder for bwd output
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
    descale_o, descale_dO, descale_dP, scale_dQ, scale_dK, scale_dV, scale_dP,
    scale, use_causal_mask, layout, cudnn_version):
  # check if flash attention is supported for this attention pattern
  check_is_flash_attention(
      query, key, layout, cudnn_version, True)
  outputs = _dot_product_attention_fp8_fwd_p_wrapper.bind(
      query, key, value,
      descale_q, descale_k, descale_v, descale_s,
      scale_s, scale_o,
      scale=scale, use_causal_mask=use_causal_mask, layout=layout, is_training=True)
  res = (query, key, value, outputs[3], outputs[0],
         descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP,
         scale_s, scale_dQ, scale_dK, scale_dV, scale_dP)
  
  return (outputs[0], outputs[1], outputs[2]), res

def _dot_product_attention_fp8_bwd_rule(
    scale, use_causal_mask, layout, cudnn_version, res, g):
  (query, key, value, activation, fwd_output,
   descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP, 
   scale_s, scale_dQ, scale_dK, scale_dV, scale_dP) = res
  grad_output = g[0]
  # dQ, dK, dV, amax_dq, amax_dk ,amax_dv, amax_dp
  grads = _dot_product_attention_fp8_bwd_p_wrapper.bind(
    query,
    key,
    value,
    fwd_output,
    grad_output,
    activation,
    descale_q,
    descale_k,
    descale_v,
    descale_o,
    descale_dO,
    descale_s,
    descale_dP,
    scale_s,
    scale_dQ,
    scale_dK,
    scale_dV,
    scale_dP,
    scale=scale,
    use_causal_mask=use_causal_mask,
    layout=layout,
    )
  grads = (*grads,) + (None,) * (20 - len(grads))
  return grads

def _dot_product_attention_fp8_fwd_impl(
    query, key, value,
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
    scale, use_causal_mask, layout, is_training):
  # args: {Q, K, V, mask*}
  outputs = _dot_product_attention_fp8_fwd_p.bind(
      query,
      key,
      value,
      descale_q,
      descale_k,
      descale_v,
      descale_s,
      scale_s,
      scale_o,
      scale=scale,
      use_causal_mask=use_causal_mask,
      layout=layout,
      is_training=is_training,
  )
  return outputs

def _dot_product_attention_fp8_bwd_impl(
    query, key, value, fwd_output, grad_output, activation,
    descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s,
    descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
    scale, use_causal_mask, layout):
  grads = _dot_product_attention_fp8_bwd_p.bind(
      query, key, value, fwd_output, grad_output, activation,
      descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s,
      descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
      scale=scale, use_causal_mask=use_causal_mask, layout=layout)
  return grads  # dQ, dK, dV, amax_dq, amax_dk ,amax_dv, amax_dp


def _dot_product_attention_fp8_fwd_abstract(
    query, key, value,
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
    scale, use_causal_mask, layout, is_training):
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
      core.ShapedArray((1,1,1,1), jnp.float32),  # amax_s
      core.ShapedArray((1,1,1,1), jnp.float32),  # amax_o
      core.ShapedArray(softmax_stat_shape, jnp.float32),  # M: softmax_stat
    )
  else:
    return (
      core.ShapedArray(output_shape, query_dtype),  # output
      core.ShapedArray((1,1,1,1), jnp.float32),  # amax_s
      core.ShapedArray((1,1,1,1), jnp.float32),  # amax_o
    )

def _dot_product_attention_fp8_bwd_abstract(
    query, key, value, fwd_output, grad_output, activation,
    descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s,
    descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
    scale, use_causal_mask, layout):
  query_dtype = dtypes.canonicalize_dtype(query.dtype)
  key_dtype = dtypes.canonicalize_dtype(key.dtype)
  value_dtype = dtypes.canonicalize_dtype(value.dtype)
  
  amax_shape = (1,1,1,1)

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
        amax_shape, jnp.float32
    ),  # amax of grad of query
    core.ShapedArray(
        amax_shape, jnp.float32
    ),  # amax of grad key
    core.ShapedArray(
        amax_shape, jnp.float32
    ),  # amax of grad value
    core.ShapedArray(
        amax_shape, jnp.float32
    ),  # amax of grad of P  
  )

def _dot_product_attention_fp8_fwd_cuda_lowering(
    ctx, query, key, value,
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
    scale, use_causal_mask, layout, is_training):
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
  amax_shape = (1,1,1,1)
  workspace_type = ir.IntegerType.get_unsigned(8)
  mask_type = MaskType.CAUSAL if use_causal_mask else MaskType.NO_MASK
  backend_config = create_dot_product_attention_fp8_backend_config(
      B, N, T, S, ir.BF16Type.get(),#query_type.element_type, 
      scale, mask_type, layout, is_bwd=False,
  )
  # {Q, K, V, mask*, q_seqlen*, kv_seqlen*}
  # {output, activation*, workspace}
  operands = [query, key, value, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o]

  custom_call_name = get_custom_call_name(False)
      
  # create output types and layouts
  if is_training:
    result_types = [
      ir.RankedTensorType.get(output_shape, query_type.element_type),
      ir.RankedTensorType.get((1,1,1,1), ir.F32Type.get()),
      ir.RankedTensorType.get((1,1,1,1), ir.F32Type.get()),
      ir.RankedTensorType.get(softmax_stat_shape, ir.F32Type.get()),
      ir.RankedTensorType.get(workspace_shape, workspace_type),
    ]
    result_layouts = [output_layout] + default_layouts(amax_shape, amax_shape, softmax_stat_shape, workspace_shape)
  else:
    result_types = [
      ir.RankedTensorType.get(output_shape, query_type.element_type),
      ir.RankedTensorType.get((1,1,1,1), ir.F32Type.get()),
      ir.RankedTensorType.get((1,1,1,1), ir.F32Type.get()),
      ir.RankedTensorType.get(workspace_shape, workspace_type)
    ]
    result_layouts = [output_layout] + default_layouts(amax_shape, amax_shape, workspace_shape)

  tmp_shapes = [ir.RankedTensorType(operand.type).shape for operand in operands[:3]]
  tmp_shapes += [[1, 1, 1, 1]] * 6
  operand_layouts = default_layouts(*tmp_shapes)
  out = mlir.custom_call(
    custom_call_name,
    result_types=result_types,
    operands=operands,
    backend_config=backend_config,
    operand_layouts=operand_layouts,
    result_layouts=result_layouts,
  )
  # drop workspace memory
  # output should be (B, T, N, H) instead of (B, N, T, H)
  if is_training:
    return [hlo.transpose(out.results[0], output_transpose_perm), out.results[1], out.results[2], out.results[3]]
  else:
    return [hlo.transpose(out.results[0], output_transpose_perm), out.results[1], out.results[2]]



def _dot_product_attention_fp8_bwd_cuda_lowering(
    ctx, query, key, value, fwd_output, grad_output, activation,
    descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s,
    descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, scale,
    use_causal_mask, layout):
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
  amax_shape = (1,1,1,1)

  grad_query_shape = (B, q_N, T, H)
  grad_key_shape = (B, k_N, S, H)
  grad_value_shape = (B, k_N, S, H)
  mask_type = MaskType.CAUSAL if use_causal_mask else MaskType.NO_MASK

  backend_config = create_dot_product_attention_fp8_backend_config(
      B, q_N, T, S, ir.BF16Type.get(),
      scale, mask_type, layout, is_bwd=True,
  )

  # create operands
  operands = [
    query,
    key,
    value,
    fwd_output,
    grad_output,
    activation,
    descale_q,
    descale_k,
    descale_v,
    descale_o,
    descale_dO,
    descale_s,
    descale_dP,
    scale_s,
    scale_dQ,
    scale_dK,
    scale_dV,
    scale_dP,
  ]

  # get custom call name
  custom_call_name = get_custom_call_name(True)

  # create output types and layouts
  # grad_query, grad_key, grad_value, amax_dQ, amax_dK, amax_dV, amax_dP
  result_types = [
    ir.RankedTensorType.get(grad_query_shape, query_type.element_type),
    ir.RankedTensorType.get(grad_key_shape, key_type.element_type),
    ir.RankedTensorType.get(grad_value_shape, value_type.element_type),
    ir.RankedTensorType.get(amax_shape, ir.F32Type.get()),
    ir.RankedTensorType.get(amax_shape, ir.F32Type.get()),
    ir.RankedTensorType.get(amax_shape, ir.F32Type.get()),
    ir.RankedTensorType.get(amax_shape, ir.F32Type.get()),
  ]
  result_layouts = [grad_layout, grad_layout, grad_layout] + default_layouts(amax_shape, amax_shape, amax_shape, amax_shape)

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
  dqkv_amaxs = (hlo.transpose(out.results[0], grad_transpose_perm),
          hlo.transpose(out.results[1], grad_transpose_perm),
          hlo.transpose(out.results[2], grad_transpose_perm),
          out.results[3], out.results[4], out.results[5], out.results[6])
  # Only keep dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP here
  return dqkv_amaxs
  
# batcher
def _check_valid_batch_dims(bdims):
  for dim in bdims:
    if dim not in [0, None]:
      raise NotImplementedError(
        f"Currently only support batch_dim in [0, None], but got {dim=}")

def _dot_product_attention_fp8_fwd_batcher(
    batched_args, batch_dims, *, scale, use_causal_mask, layout, is_training):
  _check_valid_batch_dims(batch_dims)
  query, key, value,\
    descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, = batched_args
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

  # reshape to 4D shape
  query = jnp.reshape(query, (B,) + query.shape[-3:])
  key = jnp.reshape(key, (B,) + key.shape[-3:])
  value = jnp.reshape(value, (B,) + key.shape[-3:])

  outputs = _dot_product_attention_fp8_fwd_p_wrapper.bind(
      query, key, value, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
      scale=scale, use_causal_mask=use_causal_mask, layout=layout, is_training=is_training)

  # reshape to original shape
  output, amax_s, amax_o = outputs[0], outputs[1], outputs[2]
  output = jnp.reshape(output, query.shape)
  if is_training:
    activation = outputs[3]
    activation = jnp.reshape(activation, (*Bs, N, T))
    return (output, amax_s, amax_o, activation), out_bdims
  else:
    return (output, amax_s, amax_o), out_bdims

def _dot_product_attention_fp8_bwd_batcher(
    batched_args, batch_dims, *, scale, use_causal_mask, layout):
  _check_valid_batch_dims(batch_dims)
  query, key, value, fwd_output, grad_output, activation,\
    descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP,\
    scale_s, scale_dQ, scale_dK, scale_dV, scale_dP = batched_args
  query_bdim = batch_dims[0]
  out_bdims = query_bdim, query_bdim, query_bdim

  if layout == AttentionLayout.BNTH.value:
    *Bs, N, T, _ = query.shape
    *_, _, S, _ = key.shape
  else:
    *Bs, T, N, _ = query.shape
    *_, S, _, _ = key.shape
  B = reduce(operator.mul, Bs)

  # reshape to 4D shape
  query = jnp.reshape(query, (B,) + query.shape[-3:])
  key = jnp.reshape(key, (B,) + key.shape[-3:])
  value = jnp.reshape(value, (B,) + key.shape[-3:])

  activation = jnp.reshape(activation, (B, N, T))
  fwd_output = jnp.reshape(fwd_output, (B,) + query.shape[-3:])
  grad_output = jnp.reshape(grad_output, (B,) + query.shape[-3:])

  grads = _dot_product_attention_fp8_bwd_p_wrapper.bind(
      query, key, value, fwd_output, grad_output, activation, 
      descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
      scale=scale, use_causal_mask=use_causal_mask, layout=layout,
  )

  grad_query, grad_key, grad_value = grads[:3]
  # reshape to original shape
  grad_query = jnp.reshape(grad_query, query.shape)
  grad_key = jnp.reshape(grad_key, key.shape)
  grad_value = jnp.reshape(grad_value, value.shape)

  return grads, out_bdims

# custom partitioning
def _get_padded_spec(arg_info):
  spec = None if arg_info.sharding is None else arg_info.sharding.spec
  ndim = arg_info.ndim
  if spec is None:
    return (None,) * ndim
  assert len(spec) <= ndim
  return spec + (None,) * (ndim - len(spec))

def _check_qkv_spec(
    query_spec, key_spec, value_spec):
  # check qkv spec
  if not query_spec == key_spec == value_spec:
    raise ValueError("Query, key and value should have same sharding.")
  *batch_spec, q_seq_spec, num_head_spec, head_spec = query_spec
  if q_seq_spec is not None:
    raise ValueError("Sharding on sequence dim is not allowed.")
  if head_spec is not None:
    raise ValueError("Sharding on head dim is not allowed.")

# fwd custom partition
def _infer_fp8_fwd_output_sharding(mesh, arg_shapes, is_training):
  # only sharding on batch and num_head dim is allowed
  # (*batch, q_seq, num_head, head)
  query_spec = _get_padded_spec(arg_shapes[0])
  # (*batch, kv_seq, num_head, head)
  key_spec = _get_padded_spec(arg_shapes[1])
  value_spec = _get_padded_spec(arg_shapes[2])

  _check_qkv_spec(
    query_spec, key_spec, value_spec)
  # keep out sharding same as query sharding since they have same shape
  out_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
  amax_sharding = NamedSharding(mesh, PartitionSpec())
  if is_training:
    # activation sharding
    *batch_spec, q_seq_spec, num_head_spec, _ = query_spec
    activation_sharding = NamedSharding(
      mesh, PartitionSpec(*batch_spec, num_head_spec, q_seq_spec, None))
    return [out_sharding, amax_sharding, amax_sharding, activation_sharding]
  return [out_sharding, amax_sharding, amax_sharding]

_dot_product_attention_fp8_fwd_lower = custom_partitioning(
    _dot_product_attention_fp8_fwd_impl, static_argnums=(9, 10, 11, 12))

def _dot_product_attention_fp8_fwd_infer_sharding_from_operands(
    scale, use_causal_mask, layout, is_training,
    mesh, arg_shapes, result_shape):
  return _infer_fp8_fwd_output_sharding(mesh, arg_shapes, is_training)

def _dot_product_attention_fp8_fwd_partition(
    scale, use_causal_mask, layout, is_training,
    mesh, arg_shapes, result_shape):
  # args sharding
  arg_shardings = tuple([arg_i.sharding for arg_i in arg_shapes])
  out_shardings = _infer_fp8_fwd_output_sharding(
    mesh, arg_shapes, is_training)
  impl = partial(
      _dot_product_attention_fp8_fwd_impl, scale=scale, use_causal_mask=use_causal_mask,
      layout=layout, is_training=is_training)
  return mesh, impl, out_shardings, arg_shardings

# bwd custom partition
def _infer_fp8_bwd_output_sharding(mesh, arg_shapes):
  # (*batch, q_seq, num_head, head)
  query_spec = _get_padded_spec(arg_shapes[0])
  # (*batch, kv_seq, num_head, head)
  key_spec = _get_padded_spec(arg_shapes[1])
  value_spec = _get_padded_spec(arg_shapes[2])

  _check_qkv_spec(
    query_spec, key_spec, value_spec)
  # keep grad query sharding same as query sharding
  grad_query_sharding = NamedSharding(mesh, PartitionSpec(*query_spec))
  grad_key_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
  grad_value_sharding = NamedSharding(mesh, PartitionSpec(*key_spec))
  amax_sharding = NamedSharding(mesh, PartitionSpec())
  
  out_shardings = [grad_query_sharding, grad_key_sharding, grad_value_sharding,
                   amax_sharding, amax_sharding, amax_sharding, amax_sharding]
  return out_shardings

_dot_product_attention_fp8_bwd_lower = custom_partitioning(
    _dot_product_attention_fp8_bwd_impl, static_argnums=(18,19,20)
)
def _dot_product_attention_fp8_bwd_infer_sharding_from_operands(
    scale, use_causal_mask, layout, mesh,
    arg_shapes, result_shape):
  return _infer_fp8_bwd_output_sharding(mesh, arg_shapes)

def _dot_product_attention_fp8_bwd_partition(
    scale, use_causal_mask, layout, mesh,
    arg_shapes, result_shape):
  out_shardings = _infer_fp8_bwd_output_sharding(mesh, arg_shapes)
  # args sharding
  arg_shardings = tuple([arg_i.sharding for arg_i in arg_shapes])
  impl = partial(
      _dot_product_attention_fp8_bwd_impl, scale=scale,
      use_causal_mask=use_causal_mask, layout=layout
  )
  return mesh, impl, out_shardings, arg_shardings


# Create dot_product_attention_fp8_fwd_p for forward operation.
_dot_product_attention_fp8_fwd_p = core.Primitive("dot_product_attention_fp8_fwd")
_dot_product_attention_fp8_fwd_p.multiple_results = True
_dot_product_attention_fp8_fwd_p.def_impl(
    partial(xla.apply_primitive, _dot_product_attention_fp8_fwd_p)
)
_dot_product_attention_fp8_fwd_p.def_abstract_eval(
    _dot_product_attention_fp8_fwd_abstract
)

mlir.register_lowering(
  _dot_product_attention_fp8_fwd_p,
  _dot_product_attention_fp8_fwd_cuda_lowering,
  platform="cuda",
)

_dot_product_attention_fp8_fwd_p_wrapper = core.Primitive(
    "dot_product_attention_fp8_fwd_wrapper"
)
_dot_product_attention_fp8_fwd_p_wrapper.multiple_results = True
_dot_product_attention_fp8_fwd_p_wrapper.def_impl(_dot_product_attention_fp8_fwd_impl)
_dot_product_attention_fp8_fwd_p_wrapper.def_abstract_eval(
    _dot_product_attention_fp8_fwd_abstract
)

# Create dot_product_attention_bwd_p for backward operation.
_dot_product_attention_fp8_bwd_p = core.Primitive("dot_product_attention_fp8_bwd")
_dot_product_attention_fp8_bwd_p.multiple_results = True
_dot_product_attention_fp8_bwd_p.def_impl(
    partial(xla.apply_primitive, _dot_product_attention_fp8_bwd_p)
)
_dot_product_attention_fp8_bwd_p.def_abstract_eval(
    _dot_product_attention_fp8_bwd_abstract
)

mlir.register_lowering(
  _dot_product_attention_fp8_bwd_p,
  _dot_product_attention_fp8_bwd_cuda_lowering,
  platform="cuda",
)

_dot_product_attention_fp8_bwd_p_wrapper = core.Primitive(
    "dot_product_attention_fp8_bwd_wrapper"
)
_dot_product_attention_fp8_bwd_p_wrapper.multiple_results = True
_dot_product_attention_fp8_bwd_p_wrapper.def_impl(_dot_product_attention_fp8_bwd_impl)
_dot_product_attention_fp8_bwd_p_wrapper.def_abstract_eval(
    _dot_product_attention_fp8_bwd_abstract
)

batching.primitive_batchers[
    _dot_product_attention_fp8_fwd_p_wrapper
] = _dot_product_attention_fp8_fwd_batcher
batching.primitive_batchers[
    _dot_product_attention_fp8_bwd_p_wrapper
] = _dot_product_attention_fp8_bwd_batcher

_dot_product_attention_fp8_fwd_lower.def_partition(
  infer_sharding_from_operands=_dot_product_attention_fp8_fwd_infer_sharding_from_operands,
  partition=_dot_product_attention_fp8_fwd_partition)

mlir.register_lowering(_dot_product_attention_fp8_fwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_fp8_fwd_lower, multiple_results=True))

_dot_product_attention_fp8_bwd_lower.def_partition(
  infer_sharding_from_operands=_dot_product_attention_fp8_bwd_infer_sharding_from_operands,
  partition=_dot_product_attention_fp8_bwd_partition)

mlir.register_lowering(_dot_product_attention_fp8_bwd_p_wrapper,
                        mlir.lower_fun(_dot_product_attention_fp8_bwd_lower, multiple_results=True))

dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fp8_fwd_p
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fp8_fwd_p_wrapper
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fp8_bwd_p
)
dispatch.prim_requires_devices_during_lowering.add(
  _dot_product_attention_fp8_bwd_p_wrapper
)


@partial(jax.custom_vjp, nondiff_argnums=(20, 21, 22, 23))
def _dot_product_attention_fp8(query: Array,
                               key: Array,
                               value: Array,
                               amax_dQ: Array,
                               amax_dK: Array,
                               amax_dV: Array,
                               amax_dP: Array,
                               descale_q: Array,
                               descale_k: Array, 
                               descale_v: Array, 
                               descale_s: Array, 
                               scale_s: Array, 
                               scale_o: Array,
                               descale_o: Array,
                               descale_dO: Array,
                               descale_dP: Array,
                               scale_dQ: Array,
                               scale_dK: Array,
                               scale_dV: Array,
                               scale_dP: Array,                                                  
                               scale: float,
                               use_causal_mask: bool,
                               layout: int,
                               cudnn_version: int):
  output, amax_s, amax_o = _dot_product_attention_fp8_fwd(
      query, key, value,
      descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
      scale, use_causal_mask, layout, cudnn_version
  )
  return output, amax_s, amax_o

_dot_product_attention_fp8.defvjp(_dot_product_attention_fp8_fwd_rule, _dot_product_attention_fp8_bwd_rule)

# User interface for fp8
def dot_product_attention_fp8(query: Array,
                              key: Array,
                              value: Array,
                              amax_dQ: Array,
                              amax_dK: Array,
                              amax_dV: Array,
                              amax_dP: Array,
                              descale_q: Array,
                              descale_k: Array,
                              descale_v: Array,
                              descale_s: Array,
                              scale_s: Array,
                              scale_o: Array,
                              descale_o: Array,
                              descale_dO: Array,
                              descale_dP: Array,
                              scale_dQ: Array,
                              scale_dK: Array,
                              scale_dV: Array,
                              scale_dP: Array,
                              *,
                              scale: float = 1.0,
                              use_causal_mask: bool = False,
                              qkv_layout: str = "BTNH"):
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
    amax_dQ: amax of gradient of query.
    amax_dK: amax of gradient of key.
    amax_dV: amax of gradient of value.
    amax_dP: amax of gradient of state.
    descale_q: Descaling factor of query.
    descale_k: Descaling factor of key.
    descale_v: Descaling factor of value.
    descale_s: Descaling factor of attention score.
    scale_s: Scale factor for S tensor.
    scale_o: Scale factor for output.
    descale_o (bwd): Descale factor for output.
    descale_dO (bwd): Descale factor for output gradient.
    descale_dP (bwd): Descale factor for P gradient tensor.
    scale_dQ (bwd): Scale factor for query gradient.
    scale_dK (bwd): Scale factor for key gradient.
    scale_dV (bwd): Scale factor for value gradient.
    scale_dP (bwd): Scale factor for state gradient.
    scale: Scale for the query.
    use_causal_mask: To use casual mask
    qkv_layout: Layout string, with supported formats being BTNH, BNTH, BSNH,
                BNSH.

  Returns:
    output: Output of the same shape as the query.
    amax_s: amax of state.
    amax_o: amax of output.
  """
  # check if cuDNN is installed
  cudnn_version = check_cudnn_version()
  # only support Ampere and Hopper for now
  check_compute_capability((80, 90))
  layout = _normalize_layout(qkv_layout)
  # check if input shape and data type is compatiable
  check_layout(query, key, value, layout)
  output, amax_s, amax_o = _dot_product_attention_fp8(
      query, key, value, 
      amax_dQ, amax_dK, amax_dV, amax_dP,
      descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
      descale_o, descale_dO, descale_dP, scale_dQ, scale_dK, scale_dV, scale_dP, 
      scale, use_causal_mask, layout.value, cudnn_version
  )
  return output, amax_s, amax_o
