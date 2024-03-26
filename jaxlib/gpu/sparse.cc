/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"
#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/sparse_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/tsl/python/lib/core/numpy.h"

namespace nb = nanobind;

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

gpusparseIndexType_t DtypeToCuSparseIndexType(const dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, gpusparseIndexType_t>({
          {{'u', 2}, GPUSPARSE_INDEX_16U},
          {{'i', 4}, GPUSPARSE_INDEX_32I},
          {{'i', 8}, GPUSPARSE_INDEX_64I},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    nb::str repr = nb::repr(np_type);
    throw std::invalid_argument(
        absl::StrFormat("Unsupported index dtype: %s", repr.c_str()));
  }
  return it->second;
}

gpuDataType DtypeToCudaDataType(const dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, gpuDataType>({
        {{'f', 2}, GPU_R_16F}, {{'c', 4}, GPU_C_16F}, {{'f', 4}, GPU_R_32F},
            {{'c', 8}, GPU_C_32F}, {{'f', 8}, GPU_R_64F},
            {{'c', 16}, GPU_C_64F},
#ifdef JAX_GPU_CUDA
            {{'i', 1}, CUDA_R_8I}, {{'u', 1}, CUDA_R_8U},
            {{'i', 4}, CUDA_R_32I}, {{'u', 4}, CUDA_R_32U},
#if JAX_GPU_HAVE_SPARSE
            {{'V', 2}, CUDA_R_16BF},
#endif  // JAX_GPU_HAVE_SPARSE
#endif  // JAX_GPU_CUDA
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    nb::str repr = nb::repr(np_type);
    throw std::invalid_argument(
        absl::StrFormat("Unsupported data dtype: %s", repr.c_str()));
  }
  return it->second;
}
// Returns the descriptor for a Sparse matrix.
SparseMatDescriptor BuildSparseMatDescriptor(const dtype& data_dtype,
                                             const dtype& index_dtype,
                                             int rows, int cols, int nnz,
                                             int batch_count,
                                             int batch_stride) {
  gpuDataType value_type = DtypeToCudaDataType(data_dtype);
  gpusparseIndexType_t index_type = DtypeToCuSparseIndexType(index_dtype);
  return SparseMatDescriptor{value_type, index_type,  rows,        cols,
                             nnz,        batch_count, batch_stride};
}

// Returns the descriptor for a Dense matrix.
DenseMatDescriptor BuildDenseMatDescriptor(const dtype& data_dtype,
                                           int rows, int cols, int batch_count,
                                           int batch_stride) {
  gpuDataType value_type = DtypeToCudaDataType(data_dtype);
  return DenseMatDescriptor{value_type, rows, cols, batch_count, batch_stride};
}

// Returns the descriptor for a Dense vector.
DenseVecDescriptor BuildDenseVecDescriptor(const dtype& data_dtype,
                                           int size) {
  gpuDataType value_type = DtypeToCudaDataType(data_dtype);
  return DenseVecDescriptor{value_type, size};
}

#if JAX_GPU_HAVE_SPARSE
// CsrToDense: Convert CSR matrix to dense matrix

// Returns the descriptor for a Sparse matrix.
std::pair<size_t, nb::bytes> BuildCsrToDenseDescriptor(
    const dtype& data_dtype, const dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz,
                               /*batch_count*/ 1, /*batch_stride*/ 0);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;

  // buffer_size does not reference these pointers, but does error on NULL.
  // TODO(jakevdp): check whether this is documented.
  int val alignas(16) = 0;
  void* empty = &val;

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCsr(
      &mat_a, d.rows, d.cols, d.nnz, empty, empty, empty, d.index_type,
      d.index_type, GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, empty, d.value_type, GPUSPARSE_ORDER_ROW)));
  size_t buffer_size;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseSparseToDense_bufferSize(
      handle.get(), mat_a, mat_b, GPUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));

  return {buffer_size, PackDescriptor(d)};
}

absl::Status CsrToDense_(gpuStream_t stream, void** buffers, const char* opaque,
                         size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseCreateCsr(&mat_a, d.rows, d.cols, d.nnz,
                         /*csrRowOffsets=*/buffers[2],
                         /*csrColInd=*/buffers[1],
                         /*csrValues=*/buffers[0], d.index_type, d.index_type,
                         GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, buffers[3], d.value_type, GPUSPARSE_ORDER_ROW)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseSparseToDense(handle.get(), mat_a, mat_b,
                             GPUSPARSE_SPARSETODENSE_ALG_DEFAULT, buffers[4])));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));
  return absl::OkStatus();
}

void CsrToDense(gpuStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrToDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrFromDense: Convert dense matrix to CSR matrix

// Returns the descriptor for a CsrFromDense operation.
std::pair<size_t, nb::bytes> BuildCsrFromDenseDescriptor(
    const dtype& data_dtype, const dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz,
                               /*batch_count=*/1, /*batch_stride=*/0);

  gpusparseDnMatDescr_t mat_a = 0;
  gpusparseSpMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val alignas(16) = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, empty, d.value_type, GPUSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCsr(
      &mat_b, d.rows, d.cols, d.nnz, empty, empty, empty, d.index_type,
      d.index_type, GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  size_t buffer_size;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDenseToSparse_bufferSize(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_b)));

  return {buffer_size, PackDescriptor(d)};
}

absl::Status CsrFromDense_(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SparseMatDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SparseMatDescriptor& d = **s;
  auto h = SparseHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpusparseDnMatDescr_t mat_a = 0;
  gpusparseSpMatDescr_t mat_b = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, buffers[0], d.value_type, GPUSPARSE_ORDER_ROW)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseCreateCsr(&mat_b, d.rows, d.cols, d.nnz,
                         /*csrRowOffsets=*/buffers[3],
                         /*csrColInd=*/buffers[2],
                         /*csrValues=*/buffers[1], d.index_type, d.index_type,
                         GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDenseToSparse_analysis(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDenseToSparse_convert(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      buffers[4])));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_a)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_b)));
  return absl::OkStatus();
}

void CsrFromDense(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = CsrFromDense_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// CsrMatvec: Product of CSR matrix and dense vector.

// Returns the descriptor for a CsrMatvec operation.
std::pair<size_t, nb::bytes> BuildCsrMatvecDescriptor(
    const dtype& data_dtype, const dtype& x_dtype,
    const dtype& compute_dtype, const dtype& index_dtype, int rows,
    int cols, int nnz, bool transpose) {
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz,
                               /*batch_count=*/1, /*batch_stride=*/0);
  DenseVecDescriptor x =
      BuildDenseVecDescriptor(x_dtype, transpose ? rows : cols);
  DenseVecDescriptor y =
      BuildDenseVecDescriptor(compute_dtype, transpose ? cols : rows);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnVecDescr_t vec_x = 0;
  gpusparseDnVecDescr_t vec_y = 0;
  gpusparseOperation_t op = transpose ? GPUSPARSE_OPERATION_TRANSPOSE
                                      : GPUSPARSE_OPERATION_NON_TRANSPOSE;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val alignas(16) = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCsr(
      &mat_a, A.rows, A.cols, A.nnz, empty, empty, empty, A.index_type,
      A.index_type, GPUSPARSE_INDEX_BASE_ZERO, A.value_type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnVec(&vec_x, x.size, empty, x.type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnVec(&vec_y, y.size, empty, y.type)));
  size_t buffer_size;
  SparseConst alpha = ConstOne(y.type);
  SparseConst beta = ConstZero(y.type);
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseSpMV_bufferSize(
      handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y, y.type,
      GPUSPARSE_SPMV_CSR_ALG, &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnVec(vec_x)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnVec(vec_y)));

  return {buffer_size, PackDescriptor(CsrMatvecDescriptor{A, x, y, op})};
}

// CsrMatmat: Product of CSR matrix and dense matrix.

// Returns the descriptor for a CsrMatmat operation.
std::pair<size_t, nb::bytes> BuildCsrMatmatDescriptor(
    const dtype& data_dtype, const dtype& b_dtype,
    const dtype& compute_dtype, const dtype& index_dtype, int rows,
    int cols, int BCcols, int nnz, bool transpose) {
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz,
                               /*batch_count=*/1, /*batch_stride=*/0);
  DenseMatDescriptor B =
      BuildDenseMatDescriptor(b_dtype, transpose ? rows : cols, BCcols,
                              /*batch_count=*/1, /*batch_stride=*/0);
  DenseMatDescriptor C =
      BuildDenseMatDescriptor(compute_dtype, transpose ? cols : rows, BCcols,
                              /*batch_count=*/1, /*batch_stride=*/0);
  gpusparseOperation_t op_A = transpose ? GPUSPARSE_OPERATION_TRANSPOSE
                                        : GPUSPARSE_OPERATION_NON_TRANSPOSE;

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  gpusparseDnMatDescr_t mat_c = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val alignas(16) = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCsr(
      &mat_a, A.rows, A.cols, A.nnz, empty, empty, empty, A.index_type,
      A.index_type, GPUSPARSE_INDEX_BASE_ZERO, A.value_type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnMat(&mat_b, B.rows, B.cols, /*ld=*/B.cols,
                                         empty, B.type, GPUSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnMat(&mat_c, C.rows, C.cols, /*ld=*/C.cols,
                                         empty, C.type, GPUSPARSE_ORDER_ROW)));
  size_t buffer_size;
  SparseConst alpha = ConstOne(C.type);
  SparseConst beta = ConstZero(C.type);
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseSpMM_bufferSize(
      handle.get(), op_A, GPUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a,
      mat_b, &beta, mat_c, C.type, GPUSPARSE_SPMM_CSR_ALG, &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_c)));

  return {buffer_size, PackDescriptor(CsrMatmatDescriptor{A, B, C, op_A})};
}

// CooToDense: Convert COO matrix to dense matrix

// Returns the descriptor for a CooToDense operation.
std::pair<size_t, nb::bytes> BuildCooToDenseDescriptor(
    const dtype& data_dtype, const dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz,
                               /*batch_count=*/1, /*batch_stride=*/0);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val alignas(16) = 0;
  void* empty = &val;

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCoo(
      &mat_a, d.rows, d.cols, d.nnz, empty, empty, empty, d.index_type,
      GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_b, d.rows, d.cols,
      /*ld=*/d.cols, empty, d.value_type, GPUSPARSE_ORDER_ROW)));
  size_t buffer_size;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseSparseToDense_bufferSize(
      handle.get(), mat_a, mat_b, GPUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));

  return {buffer_size, PackDescriptor(d)};
}

// CooFromDense: Convert dense matrix to COO matrix

// Returns the descriptor for a CooFromDense operation.
std::pair<size_t, nb::bytes> BuildCooFromDenseDescriptor(
    const dtype& data_dtype, const dtype& index_dtype, int rows,
    int cols, int nnz) {
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor d =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz,
                               /*batch_count=*/1, /*batch_stride=*/0);

  gpusparseDnMatDescr_t mat_a = 0;
  gpusparseSpMatDescr_t mat_b = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val alignas(16) = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateDnMat(
      &mat_a, d.rows, d.cols,
      /*ld=*/d.cols, empty, d.value_type, GPUSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCoo(
      &mat_b, d.rows, d.cols, d.nnz, empty, empty, empty, d.index_type,
      GPUSPARSE_INDEX_BASE_ZERO, d.value_type)));
  size_t buffer_size;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDenseToSparse_bufferSize(
      handle.get(), mat_a, mat_b, GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
      &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_b)));

  return {buffer_size, PackDescriptor(d)};
}

// CooMatvec: Product of COO matrix and dense vector.

// Returns the descriptor for a CooMatvec operation.
std::pair<size_t, nb::bytes> BuildCooMatvecDescriptor(
    const dtype& data_dtype, const dtype& x_dtype,
    const dtype& compute_dtype, const dtype& index_dtype, int rows,
    int cols, int nnz, bool transpose) {
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  SparseMatDescriptor A =
      BuildSparseMatDescriptor(data_dtype, index_dtype, rows, cols, nnz,
                               /*batch_count=*/1, /*batch_stride=*/0);
  DenseVecDescriptor x =
      BuildDenseVecDescriptor(x_dtype, transpose ? rows : cols);
  DenseVecDescriptor y =
      BuildDenseVecDescriptor(compute_dtype, transpose ? cols : rows);

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnVecDescr_t vec_x = 0;
  gpusparseDnVecDescr_t vec_y = 0;
  gpusparseOperation_t op = transpose ? GPUSPARSE_OPERATION_TRANSPOSE
                                      : GPUSPARSE_OPERATION_NON_TRANSPOSE;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val alignas(16) = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCoo(
      &mat_a, A.rows, A.cols, A.nnz, empty, empty, empty, A.index_type,
      GPUSPARSE_INDEX_BASE_ZERO, A.value_type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnVec(&vec_x, x.size, empty, x.type)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnVec(&vec_y, y.size, empty, y.type)));
  size_t buffer_size;
  SparseConst alpha = ConstOne(y.type);
  SparseConst beta = ConstZero(y.type);
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseSpMV_bufferSize(
      handle.get(), op, &alpha, mat_a, vec_x, &beta, vec_y, y.type,
      GPUSPARSE_SPMV_COO_ALG, &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnVec(vec_x)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnVec(vec_y)));

  return {buffer_size, PackDescriptor(CooMatvecDescriptor{A, x, y, op})};
}

// CooMatmat: Product of COO matrix and dense matrix.

// Returns the descriptor for a CooMatmat operation.
std::pair<size_t, nb::bytes> BuildCooMatmatDescriptor(
    const dtype& data_dtype, const dtype& b_dtype,
    const dtype& compute_dtype, const dtype& index_dtype, int rows,
    int cols, int BCcols, int nnz, bool transpose, int batch_count,
    int lhs_batch_stride, int rhs_batch_stride) {
  // Three batch modes are supported, C_i = A_i B, C_i = A B_i, and
  // Ci = A_i B_i, where `i` denotes the batch dimension.
  // All three matrices A, B, and C must have the same batch count.
  // Use batch stride to trigger individual mode, e.g.,
  // `rhs_batch_stride = 0` for C_i = A_i B.
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;

  SparseMatDescriptor A = BuildSparseMatDescriptor(
      data_dtype, index_dtype, rows, cols, nnz, batch_count, lhs_batch_stride);
  DenseMatDescriptor B = BuildDenseMatDescriptor(
      b_dtype, transpose ? rows : cols, BCcols, batch_count, rhs_batch_stride);
  int C_rows = (transpose == true) ? cols : rows;
  // TODO(tianjianlu): enable the selection of batch stride.
  // The issue
  // (https://github.com/NVIDIA/CUDALibrarySamples/issues/81#issuecomment-1205562643)
  // in cusparse library does not allow batch_stride = 0.
  // int C_batch_stride = (batch_count > 1)? C_rows * BCcols : 0;
  int C_batch_stride = C_rows * BCcols;
  DenseMatDescriptor C =
      BuildDenseMatDescriptor(compute_dtype, /*rows=*/C_rows, /*cols=*/BCcols,
                              batch_count, C_batch_stride);
  gpusparseOperation_t op_A = transpose ? GPUSPARSE_OPERATION_TRANSPOSE
                                        : GPUSPARSE_OPERATION_NON_TRANSPOSE;

  gpusparseSpMatDescr_t mat_a = 0;
  gpusparseDnMatDescr_t mat_b = 0;
  gpusparseDnMatDescr_t mat_c = 0;

  // bufferSize does not reference these pointers, but does error on NULL.
  int val alignas(16) = 0;
  void* empty = &val;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCreateCoo(
      &mat_a, A.rows, A.cols, A.nnz, empty, empty, empty, A.index_type,
      GPUSPARSE_INDEX_BASE_ZERO, A.value_type)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseCooSetStridedBatch(
      mat_a, /*batchCount=*/batch_count, /*batchStride=*/A.batch_stride)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnMat(&mat_b, B.rows, B.cols, /*ld=*/B.cols,
                                         empty, B.type, GPUSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDnMatSetStridedBatch(
      mat_b, /*batchCount=*/batch_count, /*batchStride=*/B.batch_stride)));
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(gpusparseCreateDnMat(&mat_c, C.rows, C.cols, /*ld=*/C.cols,
                                         empty, C.type, GPUSPARSE_ORDER_ROW)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDnMatSetStridedBatch(
      mat_c, /*batchCount=*/batch_count, /*batchStride=*/C.batch_stride)));
  size_t buffer_size;
  SparseConst alpha = ConstOne(C.type);
  SparseConst beta = ConstZero(C.type);
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseSpMM_bufferSize(
      handle.get(), op_A, GPUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a,
      mat_b, &beta, mat_c, C.type, GPUSPARSE_SPMM_COO_ALG, &buffer_size)));

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroySpMat(mat_a)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_b)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpusparseDestroyDnMat(mat_c)));

  return {buffer_size, PackDescriptor(CooMatmatDescriptor{A, B, C, op_A})};
}

#endif  // if JAX_GPU_HAVE_SPARSE

nb::bytes BuildGtsv2Descriptor(int b, int m, int n, int ldb) {
  return PackDescriptor(Gtsv2Descriptor{b, m, n, ldb});
}

template <typename F>
size_t Gtsv2BufferSize(F f, int m, int n, int ldb) {
  auto h = SparseHandlePool::Borrow(/*stream=*/nullptr);
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  size_t size;
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(f(handle.get(), m, n, /*dl=*/nullptr, /*d=*/nullptr,
                      /*du=*/nullptr, /*B=*/nullptr, ldb, &size)));
  return size;
}

size_t Gtsv2BufferSizeF32(int m, int n, int ldb) {
  return Gtsv2BufferSize(gpusparseSgtsv2_bufferSizeExt, m, n, ldb);
}

size_t Gtsv2BufferSizeF64(int m, int n, int ldb) {
  return Gtsv2BufferSize(gpusparseDgtsv2_bufferSizeExt, m, n, ldb);
}

nb::dict Registrations() {
  nb::dict dict;
#if JAX_GPU_HAVE_SPARSE
  dict[JAX_GPU_PREFIX "sparse_csr_todense"] = EncapsulateFunction(CsrToDense);
  dict[JAX_GPU_PREFIX "sparse_csr_fromdense"] =
      EncapsulateFunction(CsrFromDense);
  dict[JAX_GPU_PREFIX "sparse_csr_matvec"] = EncapsulateFunction(CsrMatvec);
  dict[JAX_GPU_PREFIX "sparse_csr_matmat"] = EncapsulateFunction(CsrMatmat);
  dict[JAX_GPU_PREFIX "sparse_coo_todense"] = EncapsulateFunction(CooToDense);
  dict[JAX_GPU_PREFIX "sparse_coo_fromdense"] =
      EncapsulateFunction(CooFromDense);
  dict[JAX_GPU_PREFIX "sparse_coo_matvec"] = EncapsulateFunction(CooMatvec);
  dict[JAX_GPU_PREFIX "sparse_coo_matmat"] = EncapsulateFunction(CooMatmat);
#endif
  dict[JAX_GPU_PREFIX "sparse_gtsv2_f32"] = EncapsulateFunction(gtsv2_f32);
  dict[JAX_GPU_PREFIX "sparse_gtsv2_f64"] = EncapsulateFunction(gtsv2_f64);
  // TODO(tomhennigan): Add support for gtsv2 complex 32/64.
  return dict;
}

NB_MODULE(_sparse, m) {
  tsl::ImportNumpy();
  m.attr("sparse_supported") = nb::cast(JAX_GPU_HAVE_SPARSE);
  m.def("registrations", &Registrations);
#if JAX_GPU_HAVE_SPARSE
  m.def("build_csr_todense_descriptor", &BuildCsrToDenseDescriptor);
  m.def("build_csr_fromdense_descriptor", &BuildCsrFromDenseDescriptor);
  m.def("build_csr_matvec_descriptor", &BuildCsrMatvecDescriptor);
  m.def("build_csr_matmat_descriptor", &BuildCsrMatmatDescriptor);
  m.def("build_coo_todense_descriptor", &BuildCooToDenseDescriptor);
  m.def("build_coo_fromdense_descriptor", &BuildCooFromDenseDescriptor);
  m.def("build_coo_matvec_descriptor", &BuildCooMatvecDescriptor);
  m.def("build_coo_matmat_descriptor", &BuildCooMatmatDescriptor);
#endif
  m.def("gtsv2_f32_buffer_size", &Gtsv2BufferSizeF32);
  m.def("gtsv2_f64_buffer_size", &Gtsv2BufferSizeF64);
  m.def("build_gtsv2_descriptor", &BuildGtsv2Descriptor);
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
