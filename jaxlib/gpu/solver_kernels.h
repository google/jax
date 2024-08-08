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

#ifndef JAXLIB_CUSOLVER_KERNELS_H_
#define JAXLIB_CUSOLVER_KERNELS_H_

#include <cstddef>

#include "jaxlib/gpu/vendor.h"
#include "xla/service/custom_call_status.h"

namespace jax {

namespace JAX_GPU_NAMESPACE {

// Set of types known to Cusolver.
enum class SolverType {
  F32,
  F64,
  C64,
  C128,
};

// getrf: LU decomposition

struct GetrfDescriptor {
  SolverType type;
  int batch, m, n, lwork;
};

void Getrf(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// geqrf: QR decomposition

struct GeqrfDescriptor {
  SolverType type;
  int batch, m, n, lwork;
};

void Geqrf(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

#ifdef JAX_GPU_CUDA

// csrlsvpr: Linear system solve via Sparse QR

struct CsrlsvqrDescriptor {
  SolverType type;
  int n, nnz, reorder;
  double tol;
};

void Csrlsvqr(gpuStream_t stream, void** buffers, const char* opaque,
              size_t opaque_len, XlaCustomCallStatus* status);

#endif  // JAX_GPU_CUDA

// orgqr/ungqr: apply elementary Householder transformations

struct OrgqrDescriptor {
  SolverType type;
  int batch, m, n, k, lwork;
};

void Orgqr(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

struct SyevdDescriptor {
  SolverType type;
  gpusolverFillMode_t uplo;
  int batch, n;  // batch may be -1 in which case it is passed as operand.
  int lwork;
};

void Syevd(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
// Supports batches of matrices up to size 32.

struct SyevjDescriptor {
  SolverType type;
  gpusolverFillMode_t uplo;
  int batch, n;
  int lwork;
};

void Syevj(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

// Singular value decomposition using QR algorithm: gesvd

struct GesvdDescriptor {
  SolverType type;
  int batch, m, n;
  int lwork;
  signed char jobu, jobvt;
};

void Gesvd(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);

#ifdef JAX_GPU_CUDA

// Singular value decomposition using Jacobi algorithm: gesvdj

struct GesvdjDescriptor {
  SolverType type;
  int batch, m, n;
  int lwork;
  gpusolverEigMode_t jobz;
  int econ;
};

void Gesvdj(gpuStream_t stream, void** buffers, const char* opaque,
            size_t opaque_len, XlaCustomCallStatus* status);
#endif  // JAX_GPU_CUDA

// sytrd/hetrd: Reduction of a symmetric (Hermitian) matrix to tridiagonal form.
struct SytrdDescriptor {
  SolverType type;
  gpusolverFillMode_t uplo;
  int batch, n, lda, lwork;
};

void Sytrd(gpuStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len, XlaCustomCallStatus* status);


}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_CUSOLVER_KERNELS_H_
