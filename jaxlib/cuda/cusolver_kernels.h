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

#ifndef JAXLIB_CUSOLVER_KERNELS_H_
#define JAXLIB_CUSOLVER_KERNELS_H_

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "third_party/gpus/cuda/include/cusolverSp.h"
#include "jaxlib/handle_pool.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

namespace jax {

using SolverHandlePool = HandlePool<cusolverDnHandle_t, cudaStream_t>;
using SpSolverHandlePool = HandlePool<cusolverSpHandle_t, cudaStream_t>;

template <>
absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
    cudaStream_t stream);

template <>
absl::StatusOr<SpSolverHandlePool::Handle> SpSolverHandlePool::Borrow(
    cudaStream_t stream);

// Set of types known to Cusolver.
enum class CusolverType {
  F32,
  F64,
  C64,
  C128,
};

// potrf: Cholesky decomposition

struct PotrfDescriptor {
  CusolverType type;
  cublasFillMode_t uplo;
  std::int64_t batch, n;
  int lwork;
};

void Potrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);
// getrf: LU decomposition

struct GetrfDescriptor {
  CusolverType type;
  int batch, m, n;
};

void Getrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// geqrf: QR decomposition

struct GeqrfDescriptor {
  CusolverType type;
  int batch, m, n, lwork;
};

void Geqrf(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// csrlsvpr: Linear system solve via Sparse QR

struct CsrlsvqrDescriptor {
  CusolverType type;
  int n, nnz, reorder;
  double tol;
};

void Csrlsvqr(cudaStream_t stream, void** buffers, const char* opaque,
              size_t opaque_len, XlaCustomCallStatus* status);

// orgqr/ungqr: apply elementary Householder transformations

struct OrgqrDescriptor {
  CusolverType type;
  int batch, m, n, k, lwork;
};

void Orgqr(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

struct SyevdDescriptor {
  CusolverType type;
  cublasFillMode_t uplo;
  int batch, n;
  int lwork;
};

void Syevd(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
// Supports batches of matrices up to size 32.

struct SyevjDescriptor {
  CusolverType type;
  cublasFillMode_t uplo;
  int batch, n;
  int lwork;
};

void Syevj(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Singular value decomposition using QR algorithm: gesvd

struct GesvdDescriptor {
  CusolverType type;
  int batch, m, n;
  int lwork;
  signed char jobu, jobvt;
};

void Gesvd(cudaStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Singular value decomposition using Jacobi algorithm: gesvdj

struct GesvdjDescriptor {
  CusolverType type;
  int batch, m, n;
  int lwork;
  cusolverEigMode_t jobz;
  int econ;
};

void Gesvdj(cudaStream_t stream, void** buffers, const char* opaque,
            size_t opaque_len, XlaCustomCallStatus* status);

// Singular value decomposition using qdwh-based polar decomposition: gesvdp
struct GesvdpDescriptor {
  cudaDataType type;
  int batch, m, n;
  cusolverEigMode_t jobz;
  int econ;
  size_t workspaceInBytesOnDevice;
  size_t workspaceInBytesOnHost;
};

void Gesvdp(cudaStream_t stream, void** buffers, const char* opaque,
            size_t opaque_len, XlaCustomCallStatus* status);

}  // namespace jax

#endif  // JAXLIB_CUSOLVER_KERNELS_H_

