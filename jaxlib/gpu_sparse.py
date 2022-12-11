# Copyright 2019 The JAX Authors.
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
"""
cusparse wrappers for performing sparse matrix computations in JAX
"""

from functools import partial

import jaxlib.mlir.ir as ir

import numpy as np

from jaxlib import xla_client

from .xhlo_helpers import custom_call

try:
  from .cuda import _sparse as _cusparse
except ImportError:
  _cusparse = None
else:
  for _name, _value in _cusparse.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")

try:
  from .rocm import _sparse as _hipsparse
except ImportError:
  _hipsparse = None
else:
  for _name, _value in _hipsparse.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")


cuda_is_supported : bool = _cusparse and _cusparse.sparse_supported
rocm_is_supported : bool = _hipsparse and _hipsparse.sparse_supported


def _validate_csr_xhlo(data, indices, indptr, shape):
  data_type = ir.RankedTensorType(data.type)
  indices_type = ir.RankedTensorType(indices.type)
  indptr_type = ir.RankedTensorType(indptr.type)

  nnz, = data_type.shape
  assert indices_type.shape == [nnz]
  assert indptr_type.element_type == indices_type.element_type
  assert indptr_type.shape == [shape[0] + 1]
  return data_type.element_type, indices_type.element_type, nnz


def _validate_coo_xhlo(data, row, col):
  data_type = ir.RankedTensorType(data.type)
  row_type = ir.RankedTensorType(row.type)
  col_type = ir.RankedTensorType(col.type)

  nnz, = data_type.shape
  assert row_type.shape == [nnz]
  assert col_type.element_type == row_type.element_type
  assert col_type.shape == [nnz]
  return data_type.element_type, row_type.element_type, nnz


def _csr_todense_xhlo(platform, gpu_sparse, data, indices, indptr, *, shape,
                      data_dtype, index_dtype):
  """CSR to dense matrix."""
  data_type, index_type, nnz = _validate_csr_xhlo(data, indices, indptr, shape)
  rows, cols = shape

  buffer_size, opaque = gpu_sparse.build_csr_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = custom_call(
      f"{platform}sparse_csr_todense",
      [
          ir.RankedTensorType.get(shape, data_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [data, indices, indptr],
      backend_config=opaque,
      operand_layouts=[[0]] * 3,
      result_layouts=[[1, 0], [0]])
  return out[0]


cuda_csr_todense = partial(_csr_todense_xhlo, "cu", _cusparse)
rocm_csr_todense = partial(_csr_todense_xhlo, "hip", _hipsparse)


def _csr_fromdense_xhlo(platform, gpu_sparse, mat, *, nnz, index_dtype,
                        data_dtype, index_type):
  """CSR from dense matrix."""
  mat_type = ir.RankedTensorType(mat.type)
  rows, cols = mat_type.shape

  buffer_size, opaque = gpu_sparse.build_csr_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = custom_call(
      f"{platform}sparse_csr_fromdense",
      [
          ir.RankedTensorType.get([nnz], mat_type.element_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([rows + 1], index_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [mat],
      backend_config=opaque,
      operand_layouts=[[1, 0]],
      result_layouts=[[0]] * 4)
  return out[:3]


cuda_csr_fromdense = partial(_csr_fromdense_xhlo, "cu", _cusparse)
rocm_csr_fromdense = partial(_csr_fromdense_xhlo, "hip", _hipsparse)


def _csr_matvec_xhlo(platform,
                     gpu_sparse,
                     data,
                     indices,
                     indptr,
                     x,
                     *,
                     shape,
                     transpose=False,
                     compute_dtype=None,
                     compute_type=None,
                     data_dtype,
                     index_dtype,
                     x_dtype):
  """CSR matrix/vector multiply."""
  data_type, index_type, nnz = _validate_csr_xhlo(data, indices, indptr, shape)
  rows, cols = shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = gpu_sparse.build_csr_matvec_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  out = custom_call(
      f"{platform}sparse_csr_matvec",
      [
          ir.RankedTensorType.get([out_size], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [data, indices, indptr, x],
      backend_config=opaque,
      operand_layouts=[[0]] * 4,
      result_layouts=[[0]] * 2)
  return out[0]


cuda_csr_matvec = partial(_csr_matvec_xhlo, "cu", _cusparse)
rocm_csr_matvec = partial(_csr_matvec_xhlo, "hip", _hipsparse)


def _csr_matmat_xhlo(platform,
                     gpu_sparse,
                     data,
                     indices,
                     indptr,
                     B,
                     *,
                     shape,
                     transpose=False,
                     compute_dtype=None,
                     compute_type=None,
                     index_dtype,
                     data_dtype,
                     B_dtype):
  """CSR from dense matrix."""
  data_type, index_type, nnz = _validate_csr_xhlo(data, indices, indptr, shape)
  rows, cols = shape
  B_shape = ir.RankedTensorType(B.type).shape
  _, Ccols = B_shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = gpu_sparse.build_csr_matmat_descriptor(
      data_dtype, B_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose)
  out_size = cols if transpose else rows

  out = custom_call(
      f"{platform}sparse_csr_matmat",
      [
          ir.RankedTensorType.get([out_size, Ccols], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [data, indices, indptr, B],
      backend_config=opaque,
      operand_layouts=[[0], [0], [0], [1, 0]],
      result_layouts=[[1, 0], [0]])
  return out[0]


cuda_csr_matmat = partial(_csr_matmat_xhlo, "cu", _cusparse)
rocm_csr_matmat = partial(_csr_matmat_xhlo, "hip", _hipsparse)


def _coo_todense_xhlo(platform, gpu_sparse, data, row, col, *, shape,
                      data_dtype, index_dtype):
  """COO to dense matrix."""
  data_type, _, nnz = _validate_coo_xhlo(data, row, col)
  rows, cols = shape

  buffer_size, opaque = gpu_sparse.build_coo_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = custom_call(
      f"{platform}sparse_coo_todense",
      [
          ir.RankedTensorType.get(shape, data_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [data, row, col],
      backend_config=opaque,
      operand_layouts=[[0]] * 3,
      result_layouts=[[1, 0], [0]])
  return out[0]


cuda_coo_todense = partial(_coo_todense_xhlo, "cu", _cusparse)
rocm_coo_todense = partial(_coo_todense_xhlo, "hip", _hipsparse)


def _coo_fromdense_xhlo(platform, gpu_sparse, mat, *, nnz, data_dtype,
                        index_dtype, index_type):
  """COO from dense matrix."""
  mat_type = ir.RankedTensorType(mat.type)
  rows, cols = mat_type.shape

  buffer_size, opaque = gpu_sparse.build_coo_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = custom_call(
      f"{platform}sparse_coo_fromdense",
      [
          ir.RankedTensorType.get([nnz], mat_type.element_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([nnz], index_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [mat],
      backend_config=opaque,
      operand_layouts=[[1, 0]],
      result_layouts=[[0]] * 4)
  return out[:3]


cuda_coo_fromdense = partial(_coo_fromdense_xhlo, "cu", _cusparse)
rocm_coo_fromdense = partial(_coo_fromdense_xhlo, "hip", _hipsparse)


def _coo_matvec_xhlo(platform,
                     gpu_sparse,
                     data,
                     row,
                     col,
                     x,
                     *,
                     shape,
                     transpose=False,
                     compute_dtype=None,
                     compute_type=None,
                     index_dtype,
                     data_dtype,
                     x_dtype):
  """COO matrix/vector multiply."""
  data_type, _, nnz = _validate_coo_xhlo(data, row, col)
  rows, cols = shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  buffer_size, opaque = gpu_sparse.build_coo_matvec_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  out = custom_call(
      f"{platform}sparse_coo_matvec",
      [
          ir.RankedTensorType.get([out_size], compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [data, row, col, x],
      backend_config=opaque,
      operand_layouts=[[0]] * 4,
      result_layouts=[[0]] * 2)
  return out[0]


cuda_coo_matvec = partial(_coo_matvec_xhlo, "cu", _cusparse)
rocm_coo_matvec = partial(_coo_matvec_xhlo, "hip", _hipsparse)


def _coo_matmat_xhlo(platform,
                     gpu_sparse,
                     data,
                     row,
                     col,
                     B,
                     *,
                     shape,
                     transpose=False,
                     compute_dtype=None,
                     compute_type=None,
                     x_dtype,
                     data_dtype,
                     index_dtype):
  """COO from dense matrix."""
  data_type, _, nnz = _validate_coo_xhlo(data, row, col)
  is_batched_matmat = False
  batch_count = 1
  if len(shape) == 2:
    rows, cols = shape
  elif len(shape) == 3:
    is_batched_matmat = True
    batch_count, rows, cols = shape
    # Redefine nnz as nnz per batch.
    nnz = nnz // batch_count

  B_shape = ir.RankedTensorType(B.type).shape
  _, Ccols = B_shape

  if compute_dtype is None:
    compute_dtype = data_dtype
    compute_type = data_type

  # TODO(tianjianlu): use batch stride to trigger different mode of batch
  # computation. Currently batch_stride = 0 is not allowed because of the issue
  # in cusparse https://github.com/NVIDIA/CUDALibrarySamples/issues/81#issuecomment-1205562643
  # Set batch stride to be the matrix size for now.
  lhs_batch_stride = nnz
  B_rows = rows if transpose else cols
  rhs_batch_stride =  B_rows * Ccols

  buffer_size, opaque = gpu_sparse.build_coo_matmat_descriptor(
      data_dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose, batch_count, lhs_batch_stride,
      rhs_batch_stride)
  out_size = cols if transpose else rows

  if is_batched_matmat:
    out_shape = [batch_count, out_size, Ccols]
    out_layout = [2, 1, 0]
  else:
    out_shape = [out_size, Ccols]
    out_layout = [1, 0]

  out = custom_call(
      f"{platform}sparse_coo_matmat",
      [
          ir.RankedTensorType.get(out_shape, compute_type),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [data, row, col, B],
      backend_config=opaque,
      operand_layouts=[[0], [0], [0], [1, 0]],
      result_layouts=[out_layout, [0]])
  return out[0]


cuda_coo_matmat = partial(_coo_matmat_xhlo, "cu", _cusparse)
rocm_coo_matmat = partial(_coo_matmat_xhlo, "hip", _hipsparse)


def _gtsv2_xhlo(platform, gpu_sparse, dl, d, du, B, *, m, n, ldb, t):
  """Calls `cusparse<t>gtsv2(dl, d, du, B, m, n, ldb)`."""
  f32 = (t == np.float32)
  if f32:
    buffer_size = gpu_sparse.gtsv2_f32_buffer_size(m, n, ldb)
  else:
    buffer_size = gpu_sparse.gtsv2_f64_buffer_size(m, n, ldb)
  out = custom_call(
      f"{platform}sparse_gtsv2_" + ("f32" if f32 else "f64"),
      [
          ir.RankedTensorType.get(
              [ldb, n], ir.F32Type.get() if f32 else ir.F64Type.get()),
          ir.RankedTensorType.get([buffer_size],
                                  ir.IntegerType.get_signless(8)),
      ],
      [dl, d, du, B],
      backend_config=gpu_sparse.build_gtsv2_descriptor(m, n, ldb),
      operand_layouts=[[0]] * 3 + [[1, 0]],
      result_layouts=[[1, 0], [0]],
      operand_output_aliases={3: 0})
  return out[0]


cuda_gtsv2 = partial(_gtsv2_xhlo, "cu", _cusparse)
rocm_gtsv2 = partial(_gtsv2_xhlo, "hip", _hipsparse)
