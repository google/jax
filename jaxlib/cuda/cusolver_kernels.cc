/* Copyright 2019 Google LLC

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

#include "jaxlib/cuda/cusolver_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "third_party/gpus/cuda/include/cusolverSp.h"
#include "jaxlib/cuda/cuda_gpu_kernel_helpers.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_helpers.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {

template <>
/*static*/ absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
    cudaStream_t stream) {
  SolverHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  cusolverDnHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

template <>
/*static*/ absl::StatusOr<SpSolverHandlePool::Handle>
SpSolverHandlePool::Borrow(cudaStream_t stream) {
  SpSolverHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  cusolverSpHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

static int SizeOfCusolverType(CusolverType type) {
  switch (type) {
    case CusolverType::F32:
      return sizeof(float);
    case CusolverType::F64:
      return sizeof(double);
    case CusolverType::C64:
      return sizeof(cuComplex);
    case CusolverType::C128:
      return sizeof(cuDoubleComplex);
  }
}

static int SizeOfCudaDataType(cudaDataType type) {
  switch (type) {
    case CUDA_R_32F:
      return sizeof(float);
    case CUDA_R_64F:
      return sizeof(double);
    case CUDA_C_32F:
      return sizeof(cuComplex);
    case CUDA_C_64F:
      return sizeof(cuDoubleComplex);
    default:
      return -1;
  }
}

// potrf: Cholesky decomposition

static absl::Status Potrf_(cudaStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<PotrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const PotrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
        cudaMemcpyAsync(buffers[1], buffers[0],
                        SizeOfCusolverType(d.type) * d.batch * d.n * d.n,
                        cudaMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[2]);
  void* workspace = buffers[3];
  if (d.batch == 1) {
    switch (d.type) {
      case CusolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSpotrf(handle.get(), d.uplo, d.n, a, d.n,
                             static_cast<float*>(workspace), d.lwork, info)));
        break;
      }
      case CusolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDpotrf(handle.get(), d.uplo, d.n, a, d.n,
                             static_cast<double*>(workspace), d.lwork, info)));
        break;
      }
      case CusolverType::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCpotrf(
            handle.get(), d.uplo, d.n, a, d.n,
            static_cast<cuComplex*>(workspace), d.lwork, info)));
        break;
      }
      case CusolverType::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZpotrf(
            handle.get(), d.uplo, d.n, a, d.n,
            static_cast<cuDoubleComplex*>(workspace), d.lwork, info)));
        break;
      }
    }
  } else {
    auto buffer_ptrs_host =
        MakeBatchPointers(stream, buffers[1], workspace, d.batch,
                          SizeOfCusolverType(d.type) * d.n * d.n);
    JAX_RETURN_IF_ERROR(buffer_ptrs_host.status());
    // Make sure that accesses to buffer_ptrs_host complete before we delete it.
    // TODO(phawkins): avoid synchronization here.
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaStreamSynchronize(stream)));
    switch (d.type) {
      case CusolverType::F32: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<float**>(workspace), d.n,

            info, d.batch)));
        break;
      }
      case CusolverType::F64: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<double**>(workspace), d.n,
            info, d.batch)));
        break;
      }
      case CusolverType::C64: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<cuComplex**>(workspace), d.n,
            info, d.batch)));
        break;
      }
      case CusolverType::C128: {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZpotrfBatched(
            handle.get(), d.uplo, d.n,
            static_cast<cuDoubleComplex**>(workspace), d.n, info, d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Potrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Potrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// getrf: LU decomposition

static absl::Status Getrf_(cudaStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GetrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GetrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfCusolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        cudaMemcpyDeviceToDevice, stream)));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case CusolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSgetrf(handle.get(), d.m, d.n, a, d.m,
                             static_cast<float*>(workspace), ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case CusolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDgetrf(handle.get(), d.m, d.n, a, d.m,
                             static_cast<double*>(workspace), ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case CusolverType::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnCgetrf(handle.get(), d.m, d.n, a, d.m,
                             static_cast<cuComplex*>(workspace), ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case CusolverType::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgetrf(
            handle.get(), d.m, d.n, a, d.m,
            static_cast<cuDoubleComplex*>(workspace), ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Getrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Getrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// geqrf: QR decomposition

static absl::Status Geqrf_(cudaStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GeqrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GeqrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfCusolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        cudaMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case CusolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* tau = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                             static_cast<float*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case CusolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* tau = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                             static_cast<double*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case CusolverType::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      cuComplex* tau = static_cast<cuComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<cuComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case CusolverType::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      cuDoubleComplex* tau = static_cast<cuDoubleComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<cuDoubleComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Geqrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Geqrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// csrlsvqr: Linear system solve via Sparse QR

static absl::Status Csrlsvqr_(cudaStream_t stream, void** buffers,
                              const char* opaque, size_t opaque_len,
                              int& singularity) {
  auto s = UnpackDescriptor<CsrlsvqrDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const CsrlsvqrDescriptor& d = **s;

  // This is the handle to the CUDA session. Gets a cusolverSp handle.
  auto h = SpSolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  cusparseMatDescr_t matdesc = nullptr;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusparseCreateMatDescr(&matdesc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cusparseSetMatType(matdesc, CUSPARSE_MATRIX_TYPE_GENERAL)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cusparseSetMatIndexBase(matdesc, CUSPARSE_INDEX_BASE_ZERO)));

  switch (d.type) {
    case CusolverType::F32: {
      float* csrValA = static_cast<float*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      float* b = static_cast<float*>(buffers[3]);
      float* x = static_cast<float*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpScsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA,
          b, (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
    case CusolverType::F64: {
      double* csrValA = static_cast<double*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      double* b = static_cast<double*>(buffers[3]);
      double* x = static_cast<double*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpDcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA,
          b, d.tol, d.reorder, x, &singularity)));

      break;
    }
    case CusolverType::C64: {
      cuComplex* csrValA = static_cast<cuComplex*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      cuComplex* b = static_cast<cuComplex*>(buffers[3]);
      cuComplex* x = static_cast<cuComplex*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpCcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA,
          b, (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
    case CusolverType::C128: {
      cuDoubleComplex* csrValA = static_cast<cuDoubleComplex*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      cuDoubleComplex* b = static_cast<cuDoubleComplex*>(buffers[3]);
      cuDoubleComplex* x = static_cast<cuDoubleComplex*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpZcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA,
          b, (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
  }

  cusparseDestroyMatDescr(matdesc);
  return absl::OkStatus();
}

void Csrlsvqr(cudaStream_t stream, void** buffers, const char* opaque,
              size_t opaque_len, XlaCustomCallStatus* status) {
  // Is >= 0 if A is singular.
  int singularity = -1;

  auto s = Csrlsvqr_(stream, buffers, opaque, opaque_len, singularity);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }

  if (singularity >= 0) {
    auto s = std::string("Singular matrix in linear solve.");
    XlaCustomCallStatusSetFailure(status, s.c_str(), s.length());
  }
}

// orgqr/ungqr: apply elementary Householder transformations

static absl::Status Orgqr_(cudaStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<OrgqrDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const OrgqrDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[2] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[2], buffers[0],
        SizeOfCusolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        cudaMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case CusolverType::F32: {
      float* a = static_cast<float*>(buffers[2]);
      float* tau = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                             static_cast<float*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case CusolverType::F64: {
      double* a = static_cast<double*>(buffers[2]);
      double* tau = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                             static_cast<double*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case CusolverType::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[2]);
      cuComplex* tau = static_cast<cuComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<cuComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case CusolverType::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[2]);
      cuDoubleComplex* tau = static_cast<cuDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<cuDoubleComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Orgqr(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Orgqr_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

static absl::Status Syevd_(cudaStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SyevdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SyevdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfCusolverType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.n),
      cudaMemcpyDeviceToDevice, stream)));
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[3]);
  void* work = buffers[4];
  switch (d.type) {
    case CusolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* w = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnSsyevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                             static_cast<float*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case CusolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* w = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnDsyevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                             static_cast<double*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case CusolverType::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      float* w = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnCheevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                             static_cast<cuComplex*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case CusolverType::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      double* w = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevd(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<cuDoubleComplex*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Syevd(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Syevd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
// Supports batches of matrices up to size 32.

absl::Status Syevj_(cudaStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<SyevjDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SyevjDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfCusolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.n),
        cudaMemcpyDeviceToDevice, stream)));
  }
  syevjInfo_t params;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateSyevjInfo(&params)));
  std::unique_ptr<syevjInfo, void (*)(syevjInfo*)> params_cleanup(
      params, [](syevjInfo* p) { cusolverDnDestroySyevjInfo(p); });

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[3]);
  void* work = buffers[4];
  if (d.batch == 1) {
    switch (d.type) {
      case CusolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<float*>(work), d.lwork, info, params)));
        break;
      }
      case CusolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<double*>(work), d.lwork, info, params)));
        break;
      }
      case CusolverType::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<cuComplex*>(work), d.lwork, info, params)));
        break;
      }
      case CusolverType::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<cuDoubleComplex*>(work), d.lwork, info, params)));
        break;
      }
    }
  } else {
    switch (d.type) {
      case CusolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<float*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case CusolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<double*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case CusolverType::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<cuComplex*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case CusolverType::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            cusolverDnZheevjBatched(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                                    static_cast<cuDoubleComplex*>(work),
                                    d.lwork, info, params, d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Syevj(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Syevj_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Singular value decomposition using QR algorithm: gesvd

static absl::Status Gesvd_(cudaStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GesvdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GesvdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfCusolverType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
      cudaMemcpyDeviceToDevice, stream)));
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  int64_t k = d.jobu == 'A' ? d.m : d.n;
  switch (d.type) {
    case CusolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      float* u = static_cast<float*>(buffers[3]);
      float* vt = static_cast<float*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<float*>(work), d.lwork,
            /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * k;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case CusolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      double* u = static_cast<double*>(buffers[3]);
      double* vt = static_cast<double*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<double*>(work), d.lwork,
            /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * k;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case CusolverType::C64: {
      cuComplex* a = static_cast<cuComplex*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      cuComplex* u = static_cast<cuComplex*>(buffers[3]);
      cuComplex* vt = static_cast<cuComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<cuComplex*>(work), d.lwork, /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * k;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case CusolverType::C128: {
      cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      cuDoubleComplex* u = static_cast<cuDoubleComplex*>(buffers[3]);
      cuDoubleComplex* vt = static_cast<cuDoubleComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<cuDoubleComplex*>(work), d.lwork,
            /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * k;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Gesvd(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Gesvd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Singular value decomposition using Jacobi algorithm: gesvdj

static absl::Status Gesvdj_(cudaStream_t stream, void** buffers,
                            const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GesvdjDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GesvdjDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfCusolverType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
      cudaMemcpyDeviceToDevice, stream)));
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  gesvdjInfo_t params;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateGesvdjInfo(&params)));
  std::unique_ptr<gesvdjInfo, void (*)(gesvdjInfo*)> params_cleanup(
      params, [](gesvdjInfo* p) { cusolverDnDestroyGesvdjInfo(p); });
  if (d.batch == 1) {
    switch (d.type) {
      case CusolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* v = static_cast<float*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdj(
            handle.get(), d.jobz, d.econ, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<float*>(work), d.lwork, info, params)));
        break;
      }
      case CusolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* v = static_cast<double*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdj(
            handle.get(), d.jobz, d.econ, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<double*>(work), d.lwork, info, params)));
        break;
      }
      case CusolverType::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        cuComplex* u = static_cast<cuComplex*>(buffers[3]);
        cuComplex* v = static_cast<cuComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdj(
            handle.get(), d.jobz, d.econ, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<cuComplex*>(work), d.lwork, info, params)));
        break;
      }
      case CusolverType::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        cuDoubleComplex* u = static_cast<cuDoubleComplex*>(buffers[3]);
        cuDoubleComplex* v = static_cast<cuDoubleComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdj(
            handle.get(), d.jobz, d.econ, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<cuDoubleComplex*>(work), d.lwork, info, params)));
        break;
      }
    }
  } else {
    switch (d.type) {
      case CusolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* v = static_cast<float*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<float*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case CusolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* v = static_cast<double*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<double*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case CusolverType::C64: {
        cuComplex* a = static_cast<cuComplex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        cuComplex* u = static_cast<cuComplex*>(buffers[3]);
        cuComplex* v = static_cast<cuComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<cuComplex*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case CusolverType::C128: {
        cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        cuDoubleComplex* u = static_cast<cuDoubleComplex*>(buffers[3]);
        cuDoubleComplex* v = static_cast<cuDoubleComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<cuDoubleComplex*>(work), d.lwork, info, params,
            d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Gesvdj(cudaStream_t stream, void** buffers, const char* opaque,
            size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Gesvdj_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Singular value decomposition using qdwh-based polar decomposition: gesvdp

static absl::Status Gesvdp_(cudaStream_t stream, void** buffers,
                            const char* opaque, size_t opaque_len) {
  auto descriptor = UnpackDescriptor<GesvdpDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(descriptor.status());
  const GesvdpDescriptor& d = **descriptor;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudaMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfCudaDataType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
      cudaMemcpyDeviceToDevice, stream)));
  void* bufferOnDevice = static_cast<void*>(buffers[5]);
  void* bufferOnHost = static_cast<void*>(buffers[6]);
  int* d_info = static_cast<int*>(buffers[7]);
  double* h_err_sigma = static_cast<double*>(buffers[8]);

  int k = std::min(d.m, d.n);
  int num_u_cols = d.econ ? k : d.m;
  int num_v_cols = d.econ ? k : d.n;

  if (d.type == CUDA_R_32F) {
    float* a = static_cast<float*>(buffers[1]);
    float* s = static_cast<float*>(buffers[2]);
    float* u = static_cast<float*>(buffers[3]);
    float* v = static_cast<float*>(buffers[4]);
    for (int i = 0; i < d.batch; ++i) {
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnXgesvdp(
          handle.get(), /*params=*/NULL, d.jobz, d.econ, d.m, d.n,
          /*dataTypeA=*/d.type, a, /*lda=*/d.m, /*dataTypeS=*/d.type, s,
          /*dataTypeU=*/d.type, u, /*ldu=*/d.m, /*dataTypeV=*/d.type, v,
          /*ldv=*/d.n, /*computeType=*/d.type, bufferOnDevice,
          d.workspaceInBytesOnDevice, bufferOnHost, d.workspaceInBytesOnHost,
          d_info, h_err_sigma)));
      a += d.m * d.n;
      s += k;
      u += d.m * num_u_cols;
      v += d.n * num_v_cols;
      ++d_info;
    }
  } else if (d.type == CUDA_R_64F) {
    double* a = static_cast<double*>(buffers[1]);
    double* s = static_cast<double*>(buffers[2]);
    double* u = static_cast<double*>(buffers[3]);
    double* v = static_cast<double*>(buffers[4]);
    for (int i = 0; i < d.batch; ++i) {
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnXgesvdp(
          handle.get(), /*params=*/NULL, d.jobz, d.econ, d.m, d.n,
          /*dataTypeA=*/d.type, a, /*lda=*/d.m, /*dataTypeS=*/d.type, s,
          /*dataTypeU=*/d.type, u, /*ldu=*/d.m, /*dataTypeV=*/d.type, v,
          /*ldv=*/d.n, /*computeType=*/d.type, bufferOnDevice,
          d.workspaceInBytesOnDevice, bufferOnHost, d.workspaceInBytesOnHost,
          d_info, h_err_sigma)));
      a += d.m * d.n;
      s += k;
      u += d.m * num_u_cols;
      v += d.n * num_v_cols;
      ++d_info;
    }
  } else if (d.type == CUDA_C_32F) {
    cuComplex* a = static_cast<cuComplex*>(buffers[1]);
    cuComplex* s = static_cast<cuComplex*>(buffers[2]);
    cuComplex* u = static_cast<cuComplex*>(buffers[3]);
    cuComplex* v = static_cast<cuComplex*>(buffers[4]);
    for (int i = 0; i < d.batch; ++i) {
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnXgesvdp(
          handle.get(), /*params=*/NULL, d.jobz, d.econ, d.m, d.n,
          /*dataTypeA=*/d.type, a, /*lda=*/d.m, /*dataTypeS=*/d.type, s,
          /*dataTypeU=*/d.type, u, /*ldu=*/d.m, /*dataTypeV=*/d.type, v,
          /*ldv=*/d.n, /*computeType=*/d.type, bufferOnDevice,
          d.workspaceInBytesOnDevice, bufferOnHost, d.workspaceInBytesOnHost,
          d_info, h_err_sigma)));
      a += d.m * d.n;
      s += k;
      u += d.m * num_u_cols;
      v += d.n * num_v_cols;
      ++d_info;
    }
  } else if (d.type == CUDA_C_64F) {
    cuDoubleComplex* a = static_cast<cuDoubleComplex*>(buffers[1]);
    cuDoubleComplex* s = static_cast<cuDoubleComplex*>(buffers[2]);
    cuDoubleComplex* u = static_cast<cuDoubleComplex*>(buffers[3]);
    cuDoubleComplex* v = static_cast<cuDoubleComplex*>(buffers[4]);
    for (int i = 0; i < d.batch; ++i) {
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnXgesvdp(
          handle.get(), /*params=*/NULL, d.jobz, d.econ, d.m, d.n,
          /*dataTypeA=*/d.type, a, /*lda=*/d.m, /*dataTypeS=*/d.type, s,
          /*dataTypeU=*/d.type, u, /*ldu=*/d.m, /*dataTypeV=*/d.type, v,
          /*ldv=*/d.n, /*computeType=*/d.type, bufferOnDevice,
          d.workspaceInBytesOnDevice, bufferOnHost, d.workspaceInBytesOnHost,
          d_info, h_err_sigma)));
      a += d.m * d.n;
      s += k;
      u += d.m * num_u_cols;
      v += d.n * num_v_cols;
      ++d_info;
    }
  } else {
    return absl::InvalidArgumentError("Invalid cudaDataType for gesvdp. "
      "Valid types are CUDA_R_32F, CUDA_R_64F, CUDA_C_32F, and CUDA_C_64F.");
  }
  return absl::OkStatus();
}

void Gesvdp(cudaStream_t stream, void** buffers, const char* opaque,
            size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Gesvdp_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace jax
