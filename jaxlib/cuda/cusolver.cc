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

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "jaxlib/cuda/cuda_gpu_kernel_helpers.h"
#include "jaxlib/cuda/cusolver_kernels.h"
#include "jaxlib/kernel_pybind11_helpers.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"

namespace jax {
namespace {
namespace py = pybind11;

// Converts a NumPy dtype to a Type.
CusolverType DtypeToCusolverType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, CusolverType>({
          {{'f', 4}, CusolverType::F32},
          {{'f', 8}, CusolverType::F64},
          {{'c', 8}, CusolverType::C64},
          {{'c', 16}, CusolverType::C128},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported dtype %s", py::repr(np_type)));
  }
  return it->second;
}

cudaDataType DtypeToCudaDataType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, cudaDataType>(
          {{{'f', 4}, CUDA_R_32F},
           {{'f', 8}, CUDA_R_64F},
           {{'c', 8}, CUDA_C_32F},
           {{'c', 16}, CUDA_C_64F}});
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported data dtype: %s", py::repr(np_type)));
  }
  return it->second;
}

// potrf: Cholesky decomposition

// Returns the workspace size and a descriptor for a potrf operation.
std::pair<int, py::bytes> BuildPotrfDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  std::int64_t workspace_size;
  cublasFillMode_t uplo =
      lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  if (b == 1) {
    switch (type) {
      case CusolverType::F32:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(cusolverDnSpotrf_bufferSize(handle.get(), uplo, n,
                                                      /*A=*/nullptr,
                                                      /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(float);
        break;
      case CusolverType::F64:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(cusolverDnDpotrf_bufferSize(handle.get(), uplo, n,
                                                      /*A=*/nullptr,
                                                      /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(double);
        break;
      case CusolverType::C64:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(cusolverDnCpotrf_bufferSize(handle.get(), uplo, n,
                                                      /*A=*/nullptr,
                                                      /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(cuComplex);
        break;
      case CusolverType::C128:
        JAX_THROW_IF_ERROR(
            JAX_AS_STATUS(cusolverDnZpotrf_bufferSize(handle.get(), uplo, n,
                                                      /*A=*/nullptr,
                                                      /*lda=*/n, &lwork)));
        workspace_size = lwork * sizeof(cuDoubleComplex);
        break;
    }
  } else {
    // We use the workspace buffer for our own scratch space.
    workspace_size = sizeof(void*) * b;
  }
  return {workspace_size,
          PackDescriptor(PotrfDescriptor{type, uplo, b, n, lwork})};
}

// getrf: LU decomposition

// Returns the workspace size and a descriptor for a getrf operation.
std::pair<int, py::bytes> BuildGetrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case CusolverType::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnSgetrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case CusolverType::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnDgetrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case CusolverType::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnCgetrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case CusolverType::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnZgetrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(GetrfDescriptor{type, b, m, n})};
}

// geqrf: QR decomposition

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, py::bytes> BuildGeqrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case CusolverType::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnSgeqrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case CusolverType::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnDgeqrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case CusolverType::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnCgeqrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
    case CusolverType::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnZgeqrf_bufferSize(handle.get(), m, n,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(GeqrfDescriptor{type, b, m, n, lwork})};
}

// csrlsvqr: Linear system solve via Sparse QR

// Returns a descriptor for a csrlsvqr operation.
py::bytes BuildCsrlsvqrDescriptor(const py::dtype& dtype, int n, int nnzA,
                                  int reorder, double tol) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SpSolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;

  return PackDescriptor(CsrlsvqrDescriptor{type, n, nnzA, reorder, tol});
}

// orgqr/ungqr: apply elementary Householder transformations

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, py::bytes> BuildOrgqrDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, int k) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case CusolverType::F32:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnSorgqr_bufferSize(handle.get(), m, n, k,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m,
                                                    /*tau=*/nullptr, &lwork)));
      break;
    case CusolverType::F64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnDorgqr_bufferSize(handle.get(), m, n, k,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m,
                                                    /*tau=*/nullptr, &lwork)));
      break;
    case CusolverType::C64:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnCungqr_bufferSize(handle.get(), m, n, k,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m,
                                                    /*tau=*/nullptr, &lwork)));
      break;
    case CusolverType::C128:
      JAX_THROW_IF_ERROR(
          JAX_AS_STATUS(cusolverDnZungqr_bufferSize(handle.get(), m, n, k,
                                                    /*A=*/nullptr,
                                                    /*lda=*/m,
                                                    /*tau=*/nullptr, &lwork)));
      break;
  }
  return {lwork, PackDescriptor(OrgqrDescriptor{type, b, m, n, k, lwork})};
}

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

// Returns the workspace size and a descriptor for a syevd operation.
std::pair<int, py::bytes> BuildSyevdDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo =
      lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  switch (type) {
    case CusolverType::F32:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
    case CusolverType::F64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
    case CusolverType::C64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
    case CusolverType::C128:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork)));
      break;
  }
  return {lwork, PackDescriptor(SyevdDescriptor{type, uplo, b, n, lwork})};
}

// Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
// Supports batches of matrices up to size 32.

// Returns the workspace size and a descriptor for a syevj_batched operation.
std::pair<int, py::bytes> BuildSyevjDescriptor(const py::dtype& dtype,
                                               bool lower, int batch, int n) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  syevjInfo_t params;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateSyevjInfo(&params)));
  std::unique_ptr<syevjInfo, void (*)(syevjInfo*)> params_cleanup(
      params, [](syevjInfo* p) { cusolverDnDestroySyevjInfo(p); });
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo =
      lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  if (batch == 1) {
    switch (type) {
      case CusolverType::F32:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params)));
        break;
      case CusolverType::F64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params)));
        break;
      case CusolverType::C64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params)));
        break;
      case CusolverType::C128:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params)));
        break;
    }
  } else {
    switch (type) {
      case CusolverType::F32:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSsyevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch)));
        break;
      case CusolverType::F64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDsyevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch)));
        break;
      case CusolverType::C64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCheevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch)));
        break;
      case CusolverType::C128:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZheevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch)));
        break;
    }
  }
  return {lwork, PackDescriptor(SyevjDescriptor{type, uplo, batch, n, lwork})};
}

// Singular value decomposition using QR algorithm: gesvd

// Returns the workspace size and a descriptor for a gesvd operation.
std::pair<int, py::bytes> BuildGesvdDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, bool compute_uv,
                                               bool full_matrices) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  switch (type) {
    case CusolverType::F32:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          cusolverDnSgesvd_bufferSize(handle.get(), m, n, &lwork)));
      break;
    case CusolverType::F64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          cusolverDnDgesvd_bufferSize(handle.get(), m, n, &lwork)));
      break;
    case CusolverType::C64:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          cusolverDnCgesvd_bufferSize(handle.get(), m, n, &lwork)));
      break;
    case CusolverType::C128:
      JAX_THROW_IF_ERROR(JAX_AS_STATUS(
          cusolverDnZgesvd_bufferSize(handle.get(), m, n, &lwork)));
      break;
  }
  signed char jobu, jobvt;
  if (compute_uv) {
    if (full_matrices) {
      jobu = jobvt = 'A';
    } else {
      jobu = jobvt = 'S';
    }
  } else {
    jobu = jobvt = 'N';
  }
  return {lwork,
          PackDescriptor(GesvdDescriptor{type, b, m, n, lwork, jobu, jobvt})};
}

// Singular value decomposition using Jacobi algorithm: gesvdj

// Returns the workspace size and a descriptor for a gesvdj operation.
std::pair<int, py::bytes> BuildGesvdjDescriptor(const py::dtype& dtype,
                                                int batch, int m, int n,
                                                bool compute_uv, int econ) {
  CusolverType type = DtypeToCusolverType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  int lwork;
  cusolverEigMode_t jobz =
      compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t params;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCreateGesvdjInfo(&params)));
  std::unique_ptr<gesvdjInfo, void (*)(gesvdjInfo*)> params_cleanup(
      params, [](gesvdjInfo* p) { cusolverDnDestroyGesvdjInfo(p); });
  if (batch == 1) {
    switch (type) {
      case CusolverType::F32:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdj_bufferSize(
            handle.get(), jobz, econ, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params)));
        break;
      case CusolverType::F64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdj_bufferSize(
            handle.get(), jobz, econ, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params)));
        break;
      case CusolverType::C64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdj_bufferSize(
            handle.get(), jobz, econ, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params)));
        break;
      case CusolverType::C128:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdj_bufferSize(
            handle.get(), jobz, econ, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params)));
        break;
    }
  } else {
    switch (type) {
      case CusolverType::F32:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnSgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch)));
        break;
      case CusolverType::F64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnDgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch)));
        break;
      case CusolverType::C64:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnCgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch)));
        break;
      case CusolverType::C128:
        JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnZgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch)));
        break;
    }
  }
  return {lwork, PackDescriptor(
                     GesvdjDescriptor{type, batch, m, n, lwork, jobz, econ})};
}

// Singular value decomposition using qdwh-based polar decomposition: gesvdp

// Returns the workspace sizes and a descriptor for a gesvdp operation.
std::pair<std::pair<size_t, size_t>, py::bytes> BuildGesvdpDescriptor(
    const py::dtype& dtype, int batch, int m, int n, bool compute_uv,
    int econ) {
  cudaDataType cuda_dtype = DtypeToCudaDataType(dtype);
  auto h = SolverHandlePool::Borrow();
  JAX_THROW_IF_ERROR(h.status());
  auto& handle = *h;
  cusolverEigMode_t jobz =
      compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  size_t workspaceInBytesOnDevice = 0;
  size_t workspaceInBytesOnHost = 0;

  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverDnXgesvdp_bufferSize(
      handle.get(), /*params=*/NULL, jobz, econ, (int64_t)m, (int64_t)n,
      /*dataTypeA=*/cuda_dtype, /*A=*/nullptr, /*lda=*/(int64_t)m,
      /*dataTypeS=*/cuda_dtype, /*S=*/nullptr, /*dataTypeU=*/cuda_dtype,
      /*U=*/nullptr, /*ldu=*/(int64_t)m, /*dataTypeV=*/cuda_dtype,
      /*V=*/nullptr,
      /*ldv=*/(int64_t)n, /*computeType=*/cuda_dtype, &workspaceInBytesOnDevice,
      &workspaceInBytesOnHost)));

  return {{workspaceInBytesOnDevice, workspaceInBytesOnHost},
          PackDescriptor(GesvdpDescriptor{cuda_dtype, batch, m, n, jobz, econ,
                                          workspaceInBytesOnDevice,
                                          workspaceInBytesOnHost})};
}

py::dict Registrations() {
  py::dict dict;
  dict["cusolver_potrf"] = EncapsulateFunction(Potrf);
  dict["cusolver_getrf"] = EncapsulateFunction(Getrf);
  dict["cusolver_geqrf"] = EncapsulateFunction(Geqrf);
  dict["cusolver_csrlsvqr"] = EncapsulateFunction(Csrlsvqr);
  dict["cusolver_orgqr"] = EncapsulateFunction(Orgqr);
  dict["cusolver_syevd"] = EncapsulateFunction(Syevd);
  dict["cusolver_syevj"] = EncapsulateFunction(Syevj);
  dict["cusolver_gesvd"] = EncapsulateFunction(Gesvd);
  dict["cusolver_gesvdj"] = EncapsulateFunction(Gesvdj);
  dict["cusolver_gesvdp"] = EncapsulateFunction(Gesvdp);
  return dict;
}

PYBIND11_MODULE(_cusolver, m) {
  m.def("registrations", &Registrations);
  m.def("build_potrf_descriptor", &BuildPotrfDescriptor);
  m.def("build_getrf_descriptor", &BuildGetrfDescriptor);
  m.def("build_geqrf_descriptor", &BuildGeqrfDescriptor);
  m.def("build_csrlsvqr_descriptor", &BuildCsrlsvqrDescriptor);
  m.def("build_orgqr_descriptor", &BuildOrgqrDescriptor);
  m.def("build_syevd_descriptor", &BuildSyevdDescriptor);
  m.def("build_syevj_descriptor", &BuildSyevjDescriptor);
  m.def("build_gesvd_descriptor", &BuildGesvdDescriptor);
  m.def("build_gesvdj_descriptor", &BuildGesvdjDescriptor);
  m.def("build_gesvdp_descriptor", &BuildGesvdpDescriptor);
}

}  // namespace
}  // namespace jax
