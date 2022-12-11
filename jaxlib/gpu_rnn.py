# Copyright 2022 The JAX Authors.
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

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.stablehlo as xhlo

import numpy as np

from jaxlib import xla_client

try:
  from .cuda import _rnn as _rnn
  for _name, _value in _rnn.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform='CUDA')
except ImportError:
  _rnn = None

if _rnn:
  compute_rnn_workspace_reserve_space_sizes = _rnn.compute_rnn_workspace_reserve_space_sizes


def cudnn_rnn_lowering(ctx, input, h_0, c_0, weights, seq_lengths, *,
                       input_size: int, hidden_size: int, num_layers: int,
                       dropout: bool, bidirectional: bool):
  """CuDnn RNN."""
  out_dtype = ctx.avals_out[0].dtype
  if out_dtype == np.float32:
    out_type = ir.F32Type.get()
  elif out_dtype == np.float64:
    out_type = ir.F64Type.get()
  elif out_dtype == np.complex64:
    out_type = ir.ComplexType.get(ir.F32Type.get())
  elif out_dtype == np.complex128:
    out_type = ir.ComplexType.get(ir.F64Type.get())
  else:
    raise ValueError(f'Unknown output type {out_dtype}')

  output_type = ir.RankedTensorType.get(ctx.avals_out[0].shape, out_type)
  batch_size = ctx.avals_in[0].shape[0]
  max_seq_length = ctx.avals_in[0].shape[1]
  workspace_shape = ctx.avals_out[3].shape
  reserve_space_shape = ctx.avals_out[4].shape
  workspace_type = ir.RankedTensorType.get(workspace_shape, ir.F32Type.get())
  reserve_space_type = ir.RankedTensorType.get(reserve_space_shape,
                                               ir.F32Type.get())
  opaque = _rnn.build_rnn_descriptor(input_size, hidden_size, num_layers,
                                     batch_size, max_seq_length, dropout,
                                     bidirectional, workspace_shape[0],
                                     reserve_space_shape[0])

  i32_type = ir.IntegerType.get_signless(32)

  out = xhlo.CustomCallOp(
      [
          ir.TupleType.get_tuple([
              output_type, h_0.type, c_0.type, workspace_type,
              reserve_space_type
          ])
      ],
      [input, h_0, c_0, weights, seq_lengths],
      call_target_name=ir.StringAttr.get('cudnn_rnn'),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
  )
  return [
      xhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
      for i in range(5)
  ]


def _xhlo_zeros_f32(shape):
  return xhlo.ConstantOp(
      ir.DenseElementsAttr.get(
          np.zeros(shape, dtype=np.float32), type=ir.F32Type.get())).result


def cudnn_rnn_bwd_lowering(ctx, dy, dhn, dcn, x, h0, c0, w, y, workspace,
                           reserve_space, seq_lengths, *, input_size: int,
                           hidden_size: int, num_layers: int, dropout: bool,
                           bidirectional: bool):
  """CuDnn RNN Backward pass."""
  batch_size = ctx.avals_in[3].shape[0]
  max_seq_length = ctx.avals_in[3].shape[1]
  workspace_shape = ctx.avals_in[8].shape
  reserve_space_shape = ctx.avals_in[9].shape
  opaque = _rnn.build_rnn_descriptor(input_size, hidden_size, num_layers,
                                     batch_size, max_seq_length, dropout,
                                     bidirectional, workspace_shape[0],
                                     reserve_space_shape[0])

  i32_type = ir.IntegerType.get_signless(32)
  zeroed_dw = _xhlo_zeros_f32(ctx.avals_out[3].shape)
  out = xhlo.CustomCallOp(
      [ir.TupleType.get_tuple([x.type, h0.type, c0.type, w.type])], [
          dy, dhn, dcn, x, h0, c0, w, y, workspace, reserve_space, zeroed_dw,
          seq_lengths
      ],
      call_target_name=ir.StringAttr.get('cudnn_rnn_bwd'),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(opaque),
      api_version=ir.IntegerAttr.get(i32_type, 2),
      called_computations=ir.ArrayAttr.get([]),
      output_operand_aliases=ir.ArrayAttr.get([
          xhlo.OutputOperandAlias.get(
              output_tuple_indices=[3],
              operand_index=10,
              operand_tuple_indices=[])
      ]))
  return [
      xhlo.GetTupleElementOp(out, ir.IntegerAttr.get(i32_type, i)).result
      for i in range(4)
  ]
