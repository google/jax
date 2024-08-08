/* Copyright 2019 The JAX Authors.

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

#include "jaxlib/gpu/solver_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/gpu/solver_handle_pool.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"

#ifdef JAX_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverSp.h"
#endif  // JAX_GPU_CUDA

namespace jax {

namespace JAX_GPU_NAMESPACE {

static int SizeOfSolverType(SolverType type) {
  switch (type) {
    case SolverType::F32:
      return sizeof(float);
    case SolverType::F64:
      return sizeof(double);
    case SolverType::C64:
      return sizeof(gpuComplex);
    case SolverType::C128:
      return sizeof(gpuDoubleComplex);
  }
}

// getrf: LU decomposition

static absl::Status Getrf_(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GetrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GetrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfSolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        gpuMemcpyDeviceToDevice, stream)));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case SolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnSgetrf(
            handle.get(), d.m, d.n, a, d.m, static_cast<float*>(workspace),
            d.lwork, ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case SolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnDgetrf(
            handle.get(), d.m, d.n, a, d.m, static_cast<double*>(workspace),
            d.lwork, ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case SolverType::C64: {
      gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnCgetrf(
            handle.get(), d.m, d.n, a, d.m, static_cast<gpuComplex*>(workspace),
            d.lwork, ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnZgetrf(
            handle.get(), d.m, d.n, a, d.m,
            static_cast<gpuDoubleComplex*>(workspace), d.lwork, ipiv, info)));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Getrf(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Getrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// geqrf: QR decomposition

static absl::Status Geqrf_(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GeqrfDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GeqrfDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfSolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        gpuMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case SolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* tau = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            gpusolverDnSgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                              static_cast<float*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case SolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* tau = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            gpusolverDnDgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                              static_cast<double*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case SolverType::C64: {
      gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
      gpuComplex* tau = static_cast<gpuComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnCgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<gpuComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
      gpuDoubleComplex* tau = static_cast<gpuDoubleComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnZgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<gpuDoubleComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Geqrf(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Geqrf_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

#ifdef JAX_GPU_CUDA

// csrlsvqr: Linear system solve via Sparse QR

static absl::Status Csrlsvqr_(gpuStream_t stream, void** buffers,
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
    case SolverType::F32: {
      float* csrValA = static_cast<float*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      float* b = static_cast<float*>(buffers[3]);
      float* x = static_cast<float*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpScsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA, b,
          (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
    case SolverType::F64: {
      double* csrValA = static_cast<double*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      double* b = static_cast<double*>(buffers[3]);
      double* x = static_cast<double*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpDcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA, b,
          d.tol, d.reorder, x, &singularity)));

      break;
    }
    case SolverType::C64: {
      gpuComplex* csrValA = static_cast<gpuComplex*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      gpuComplex* b = static_cast<gpuComplex*>(buffers[3]);
      gpuComplex* x = static_cast<gpuComplex*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpCcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA, b,
          (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* csrValA = static_cast<gpuDoubleComplex*>(buffers[0]);
      int* csrRowPtrA = static_cast<int*>(buffers[1]);
      int* csrColIndA = static_cast<int*>(buffers[2]);
      gpuDoubleComplex* b = static_cast<gpuDoubleComplex*>(buffers[3]);
      gpuDoubleComplex* x = static_cast<gpuDoubleComplex*>(buffers[4]);

      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpZcsrlsvqr(
          handle.get(), d.n, d.nnz, matdesc, csrValA, csrRowPtrA, csrColIndA, b,
          (float)d.tol, d.reorder, x, &singularity)));

      break;
    }
  }

  cusparseDestroyMatDescr(matdesc);
  return absl::OkStatus();
}

void Csrlsvqr(gpuStream_t stream, void** buffers, const char* opaque,
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

#endif  // JAX_GPU_CUDA

// orgqr/ungqr: apply elementary Householder transformations

static absl::Status Orgqr_(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<OrgqrDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const OrgqrDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[2] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[2], buffers[0],
        SizeOfSolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        gpuMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case SolverType::F32: {
      float* a = static_cast<float*>(buffers[2]);
      float* tau = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            gpusolverDnSorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                              static_cast<float*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case SolverType::F64: {
      double* a = static_cast<double*>(buffers[2]);
      double* tau = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            gpusolverDnDorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                              static_cast<double*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case SolverType::C64: {
      gpuComplex* a = static_cast<gpuComplex*>(buffers[2]);
      gpuComplex* tau = static_cast<gpuComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnCungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<gpuComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[2]);
      gpuDoubleComplex* tau = static_cast<gpuDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnZungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<gpuDoubleComplex*>(workspace), d.lwork, info)));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Orgqr(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Orgqr_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

static absl::Status Syevd_(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SyevdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SyevdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  std::int64_t batch = d.batch;
  int output_idx = 1;  // with static shapes buffers[1] is the first output
  if (d.batch == -1) {
    // the batch is passed as a second operand
    gpuMemcpyAsync((void*)&batch,
                   reinterpret_cast<const std::int64_t*>(buffers[1]),
                   sizeof(batch), gpuMemcpyDeviceToHost, stream);
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuStreamSynchronize(stream)));
    output_idx = 2;
  }
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
      buffers[output_idx], buffers[0],
      SizeOfSolverType(d.type) * batch * static_cast<std::int64_t>(d.n) *
          static_cast<std::int64_t>(d.n),
      gpuMemcpyDeviceToDevice, stream)));
  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[output_idx + 2]);
  void* work = buffers[output_idx + 3];
  switch (d.type) {
    case SolverType::F32: {
      float* a = static_cast<float*>(buffers[output_idx]);
      float* w = static_cast<float*>(buffers[output_idx + 1]);
      for (int i = 0; i < batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            gpusolverDnSsyevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                              static_cast<float*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case SolverType::F64: {
      double* a = static_cast<double*>(buffers[output_idx]);
      double* w = static_cast<double*>(buffers[output_idx + 1]);
      for (int i = 0; i < batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            gpusolverDnDsyevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                              static_cast<double*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case SolverType::C64: {
      gpuComplex* a = static_cast<gpuComplex*>(buffers[output_idx]);
      float* w = static_cast<float*>(buffers[output_idx + 1]);
      for (int i = 0; i < batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            gpusolverDnCheevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                              static_cast<gpuComplex*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[output_idx]);
      double* w = static_cast<double*>(buffers[output_idx + 1]);
      for (int i = 0; i < batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnZheevd(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<gpuDoubleComplex*>(work), d.lwork, info)));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Syevd(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Syevd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
// Supports batches of matrices up to size 32.

absl::Status Syevj_(gpuStream_t stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto s = UnpackDescriptor<SyevjDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SyevjDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfSolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.n),
        gpuMemcpyDeviceToDevice, stream)));
  }
  gpuSyevjInfo_t params;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnCreateSyevjInfo(&params)));
  std::unique_ptr<gpuSyevjInfo, void (*)(gpuSyevjInfo_t)> params_cleanup(
      params, [](gpuSyevjInfo_t p) { gpusolverDnDestroySyevjInfo(p); });

  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[3]);
  void* work = buffers[4];
  if (d.batch == 1) {
    switch (d.type) {
      case SolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnSsyevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<float*>(work), d.lwork, info, params)));
        break;
      }
      case SolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnDsyevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<double*>(work), d.lwork, info, params)));
        break;
      }
      case SolverType::C64: {
        gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnCheevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<gpuComplex*>(work), d.lwork, info, params)));
        break;
      }
      case SolverType::C128: {
        gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnZheevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<gpuDoubleComplex*>(work), d.lwork, info, params)));
        break;
      }
    }
  } else {
    switch (d.type) {
      case SolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnSsyevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<float*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case SolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnDsyevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<double*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case SolverType::C64: {
        gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnCheevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<gpuComplex*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case SolverType::C128: {
        gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
            gpusolverDnZheevjBatched(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                                     static_cast<gpuDoubleComplex*>(work),
                                     d.lwork, info, params, d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Syevj(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Syevj_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Singular value decomposition using QR algorithm: gesvd

static absl::Status Gesvd_(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GesvdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GesvdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfSolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        gpuMemcpyDeviceToDevice, stream)));
  }
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  int64_t k = d.jobu == 'A' ? d.m : d.n;
  switch (d.type) {
    case SolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      float* u = static_cast<float*>(buffers[3]);
      float* vt = static_cast<float*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnSgesvd(
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
    case SolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      double* u = static_cast<double*>(buffers[3]);
      double* vt = static_cast<double*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnDgesvd(
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
    case SolverType::C64: {
      gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      gpuComplex* u = static_cast<gpuComplex*>(buffers[3]);
      gpuComplex* vt = static_cast<gpuComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnCgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<gpuComplex*>(work), d.lwork, /*rwork=*/nullptr, info)));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * k;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      gpuDoubleComplex* u = static_cast<gpuDoubleComplex*>(buffers[3]);
      gpuDoubleComplex* vt = static_cast<gpuDoubleComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnZgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<gpuDoubleComplex*>(work), d.lwork,
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

void Gesvd(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Gesvd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

#ifdef JAX_GPU_CUDA

// Singular value decomposition using Jacobi algorithm: gesvdj

static absl::Status Gesvdj_(gpuStream_t stream, void** buffers,
                            const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GesvdjDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GesvdjDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfSolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        gpuMemcpyDeviceToDevice, stream)));
  }
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  gesvdjInfo_t params;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateGesvdjInfo(&params)));
  std::unique_ptr<gesvdjInfo, void (*)(gesvdjInfo*)> params_cleanup(
      params, [](gesvdjInfo* p) { cusolverDnDestroyGesvdjInfo(p); });
  if (d.batch <= 1 || d.m > 32 || d.n > 32 || d.econ) {
    int k = std::min(d.m, d.n);
    switch (d.type) {
      case SolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* v = static_cast<float*>(buffers[4]);
        for (int i = 0; i < d.batch; ++i) {
          JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdj(
              handle.get(), d.jobz, d.econ, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
              static_cast<float*>(work), d.lwork, info, params)));
          a += d.m * d.n;
          s += k;
          u += d.m * (d.econ ? k : d.m);
          v += (d.econ ? k : d.n) * d.n;
          ++info;
        }
        break;
      }
      case SolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* v = static_cast<double*>(buffers[4]);
        for (int i = 0; i < d.batch; ++i) {
          JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdj(
              handle.get(), d.jobz, d.econ, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
              static_cast<double*>(work), d.lwork, info, params)));
          a += d.m * d.n;
          s += k;
          u += d.m * (d.econ ? k : d.m);
          v += (d.econ ? k : d.n) * d.n;
          ++info;
        }
        break;
      }
      case SolverType::C64: {
        gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        gpuComplex* u = static_cast<gpuComplex*>(buffers[3]);
        gpuComplex* v = static_cast<gpuComplex*>(buffers[4]);
        for (int i = 0; i < d.batch; ++i) {
          JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdj(
              handle.get(), d.jobz, d.econ, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
              static_cast<gpuComplex*>(work), d.lwork, info, params)));
          a += d.m * d.n;
          s += k;
          u += d.m * (d.econ ? k : d.m);
          v += (d.econ ? k : d.n) * d.n;
          ++info;
        }
        break;
      }
      case SolverType::C128: {
        gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        gpuDoubleComplex* u = static_cast<gpuDoubleComplex*>(buffers[3]);
        gpuDoubleComplex* v = static_cast<gpuDoubleComplex*>(buffers[4]);
        for (int i = 0; i < d.batch; ++i) {
          JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdj(
              handle.get(), d.jobz, d.econ, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
              static_cast<gpuDoubleComplex*>(work), d.lwork, info, params)));
          a += d.m * d.n;
          s += k;
          u += d.m * (d.econ ? k : d.m);
          v += (d.econ ? k : d.n) * d.n;
          ++info;
        }
        break;
      }
    }
  } else {
    switch (d.type) {
      case SolverType::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* v = static_cast<float*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<float*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case SolverType::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* v = static_cast<double*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<double*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case SolverType::C64: {
        gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        gpuComplex* u = static_cast<gpuComplex*>(buffers[3]);
        gpuComplex* v = static_cast<gpuComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<gpuComplex*>(work), d.lwork, info, params, d.batch)));
        break;
      }
      case SolverType::C128: {
        gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        gpuDoubleComplex* u = static_cast<gpuDoubleComplex*>(buffers[3]);
        gpuDoubleComplex* v = static_cast<gpuDoubleComplex*>(buffers[4]);
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<gpuDoubleComplex*>(work), d.lwork, info, params,
            d.batch)));
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Gesvdj(gpuStream_t stream, void** buffers, const char* opaque,
            size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Gesvdj_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

#endif  // JAX_GPU_CUDA

// sytrd/hetrd: symmetric (Hermitian) tridiagonal reduction

static absl::Status Sytrd_(gpuStream_t stream, void** buffers,
                           const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<SytrdDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const SytrdDescriptor& d = **s;
  auto h = SolverHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[1] != buffers[0]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfSolverType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.lda),
        gpuMemcpyDeviceToDevice, stream)));
  }

  int* info = static_cast<int*>(buffers[5]);
  void* workspace = buffers[6];
  switch (d.type) {
    case SolverType::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* d_out = static_cast<float*>(buffers[2]);
      float* e_out = static_cast<float*>(buffers[3]);
      float* tau = static_cast<float*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnSsytrd(
            handle.get(), d.uplo, d.n, a, d.lda, d_out, e_out, tau,
            static_cast<float*>(workspace), d.lwork, info)));
        a += d.lda * d.n;
        d_out += d.n;
        e_out += d.n - 1;
        tau += d.n - 1;
        ++info;
      }
      break;
    }
    case SolverType::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* d_out = static_cast<double*>(buffers[2]);
      double* e_out = static_cast<double*>(buffers[3]);
      double* tau = static_cast<double*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnDsytrd(
            handle.get(), d.uplo, d.n, a, d.lda, d_out, e_out, tau,
            static_cast<double*>(workspace), d.lwork, info)));
        a += d.lda * d.n;
        d_out += d.n;
        e_out += d.n - 1;
        tau += d.n - 1;
        ++info;
      }
      break;
    }
    case SolverType::C64: {
      gpuComplex* a = static_cast<gpuComplex*>(buffers[1]);
      float* d_out = static_cast<float*>(buffers[2]);
      float* e_out = static_cast<float*>(buffers[3]);
      gpuComplex* tau = static_cast<gpuComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnChetrd(
            handle.get(), d.uplo, d.n, a, d.lda, d_out, e_out, tau,
            static_cast<gpuComplex*>(workspace), d.lwork, info)));
        a += d.lda * d.n;
        d_out += d.n;
        e_out += d.n - 1;
        tau += d.n - 1;
        ++info;
      }
      break;
    }
    case SolverType::C128: {
      gpuDoubleComplex* a = static_cast<gpuDoubleComplex*>(buffers[1]);
      double* d_out = static_cast<double*>(buffers[2]);
      double* e_out = static_cast<double*>(buffers[3]);
      gpuDoubleComplex* tau = static_cast<gpuDoubleComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnZhetrd(
            handle.get(), d.uplo, d.n, a, d.lda, d_out, e_out, tau,
            static_cast<gpuDoubleComplex*>(workspace), d.lwork, info)));
        a += d.lda * d.n;
        d_out += d.n;
        e_out += d.n - 1;
        tau += d.n - 1;
        ++info;
      }
      break;
    }
  }
  return absl::OkStatus();
}

void Sytrd(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = Sytrd_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
