/* Copyright 2022 The JAX Authors.

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

// This header is a shim that manages differences between CUDA and ROCM APIs.
// Jaxlib GPU kernels can be compiled for either CUDA or ROCM by defining
// JAX_GPU_CUDA or JAX_GPU_HIP respectively.

#ifndef JAXLIB_GPU_VENDOR_H_
#define JAXLIB_GPU_VENDOR_H_

#if defined(JAX_GPU_CUDA)

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"  // IWYU pragma: export
#include "third_party/gpus/cuda/include/cuComplex.h"  // IWYU pragma: export
#include "third_party/gpus/cuda/include/cublas_v2.h"  // IWYU pragma: export
#include "third_party/gpus/cuda/include/cuda.h"       // IWYU pragma: export
#include "third_party/gpus/cuda/include/cuda_fp8.h"   // IWYU pragma: export
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"  // IWYU pragma: export
#include "third_party/gpus/cuda/include/cufft.h"       // IWYU pragma: export
#include "third_party/gpus/cuda/include/cusolverDn.h"  // IWYU pragma: export
#include "third_party/gpus/cuda/include/cusparse.h"    // IWYU pragma: export
#include "third_party/gpus/cudnn/cudnn.h"              // IWYU pragma: export

#if CUDA_VERSION < 11080
#error "JAX requires CUDA 11.8 or newer."
#endif  // CUDA_VERSION < 11080

#define JAX_GPU_HAVE_SPARSE 1

// CUDA-11.8 introduces FP8 E4M3/E5M2 types.
#define JAX_GPU_HAVE_FP8 1

#define JAX_GPU_NAMESPACE cuda
#define JAX_GPU_PREFIX "cu"

typedef cuComplex gpuComplex;
typedef cuDoubleComplex gpuDoubleComplex;

typedef cuComplex gpublasComplex;
typedef cuDoubleComplex gpublasDoubleComplex;
typedef cublasFillMode_t gpusolverFillMode_t;
typedef cublasStatus_t gpublasStatus_t;
typedef cublasHandle_t gpublasHandle_t;
typedef CUcontext gpuContext_t;
typedef CUstreamCaptureMode gpustreamCaptureMode_t;
typedef CUstreamCaptureStatus gpustreamCaptureStatus_t;
typedef cudaDataType gpuDataType;
typedef CUdevice gpuDevice_t;
typedef CUdeviceptr gpuDevicePtr_t;
typedef cudaStream_t gpuStream_t;
typedef cudaError_t gpuError_t;
typedef CUevent gpuEvent_t;
typedef CUfunction gpuFunction_t;
typedef cudnnHandle_t gpudnnHandle_t;
typedef cudnnStatus_t gpudnnStatus_t;
typedef CUmodule gpuModule_t;
typedef cusolverDnHandle_t gpusolverDnHandle_t;
typedef cusolverStatus_t gpusolverStatus_t;
typedef cusolverEigMode_t gpusolverEigMode_t;
typedef syevjInfo gpuSyevjInfo;
typedef syevjInfo_t gpuSyevjInfo_t;
typedef cusparseIndexType_t gpusparseIndexType_t;
typedef cusparseHandle_t gpusparseHandle_t;
typedef cusparseOperation_t gpusparseOperation_t;
typedef cusparseStatus_t gpusparseStatus_t;
typedef cusparseSpMatDescr_t gpusparseSpMatDescr_t;
typedef cusparseDnMatDescr_t gpusparseDnMatDescr_t;
typedef cusparseDnVecDescr_t gpusparseDnVecDescr_t;

#define GPU_C_16F CUDA_C_16F
#define GPU_R_16F CUDA_R_16F
#define GPU_C_32F CUDA_C_32F
#define GPU_R_32F CUDA_R_32F
#define GPU_C_64F CUDA_C_64F
#define GPU_R_64F CUDA_R_64F

#define gpublasCreate cublasCreate
#define gpublasSetStream cublasSetStream
#define gpublasSgeqrfBatched cublasSgeqrfBatched
#define gpublasDgeqrfBatched cublasDgeqrfBatched
#define gpublasCgeqrfBatched cublasCgeqrfBatched
#define gpublasZgeqrfBatched cublasZgeqrfBatched
#define gpublasSgetrfBatched cublasSgetrfBatched
#define gpublasDgetrfBatched cublasDgetrfBatched
#define gpublasCgetrfBatched cublasCgetrfBatched
#define gpublasZgetrfBatched cublasZgetrfBatched

#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS

#define gpudnnCreate cudnnCreate
#define gpudnnSetStream cudnnSetStream

#define GPUDNN_STATUS_SUCCESS CUDNN_STATUS_SUCCESS

#define gpusolverDnCreate cusolverDnCreate
#define gpusolverDnSetStream cusolverDnSetStream
#define gpusolverDnCreateSyevjInfo cusolverDnCreateSyevjInfo
#define gpusolverDnDestroySyevjInfo cusolverDnDestroySyevjInfo
#define gpusolverDnSgeqrf cusolverDnSgeqrf
#define gpusolverDnDgeqrf cusolverDnDgeqrf
#define gpusolverDnCgeqrf cusolverDnCgeqrf
#define gpusolverDnZgeqrf cusolverDnZgeqrf
#define gpusolverDnSgeqrf_bufferSize cusolverDnSgeqrf_bufferSize
#define gpusolverDnDgeqrf_bufferSize cusolverDnDgeqrf_bufferSize
#define gpusolverDnCgeqrf_bufferSize cusolverDnCgeqrf_bufferSize
#define gpusolverDnZgeqrf_bufferSize cusolverDnZgeqrf_bufferSize
#define gpusolverDnSgetrf(h, m, n, a, lda, work, lwork, ipiv, info) \
  cusolverDnSgetrf(h, m, n, a, lda, work, ipiv, info)
#define gpusolverDnDgetrf(h, m, n, a, lda, work, lwork, ipiv, info) \
  cusolverDnDgetrf(h, m, n, a, lda, work, ipiv, info)
#define gpusolverDnCgetrf(h, m, n, a, lda, work, lwork, ipiv, info) \
  cusolverDnCgetrf(h, m, n, a, lda, work, ipiv, info)
#define gpusolverDnZgetrf(h, m, n, a, lda, work, lwork, ipiv, info) \
  cusolverDnZgetrf(h, m, n, a, lda, work, ipiv, info)
#define gpusolverDnSgetrf_bufferSize cusolverDnSgetrf_bufferSize
#define gpusolverDnDgetrf_bufferSize cusolverDnDgetrf_bufferSize
#define gpusolverDnCgetrf_bufferSize cusolverDnCgetrf_bufferSize
#define gpusolverDnZgetrf_bufferSize cusolverDnZgetrf_bufferSize
#define gpusolverDnSorgqr cusolverDnSorgqr
#define gpusolverDnDorgqr cusolverDnDorgqr
#define gpusolverDnCungqr cusolverDnCungqr
#define gpusolverDnZungqr cusolverDnZungqr
#define gpusolverDnSorgqr_bufferSize cusolverDnSorgqr_bufferSize
#define gpusolverDnDorgqr_bufferSize cusolverDnDorgqr_bufferSize
#define gpusolverDnCungqr_bufferSize cusolverDnCungqr_bufferSize
#define gpusolverDnZungqr_bufferSize cusolverDnZungqr_bufferSize
#define gpusolverDnSsyevd cusolverDnSsyevd
#define gpusolverDnDsyevd cusolverDnDsyevd
#define gpusolverDnCheevd cusolverDnCheevd
#define gpusolverDnZheevd cusolverDnZheevd
#define gpusolverDnSsyevd_bufferSize cusolverDnSsyevd_bufferSize
#define gpusolverDnDsyevd_bufferSize cusolverDnDsyevd_bufferSize
#define gpusolverDnCheevd_bufferSize cusolverDnCheevd_bufferSize
#define gpusolverDnZheevd_bufferSize cusolverDnZheevd_bufferSize
#define gpusolverDnSsyevj cusolverDnSsyevj
#define gpusolverDnDsyevj cusolverDnDsyevj
#define gpusolverDnCheevj cusolverDnCheevj
#define gpusolverDnZheevj cusolverDnZheevj
#define gpusolverDnSsyevj_bufferSize cusolverDnSsyevj_bufferSize
#define gpusolverDnDsyevj_bufferSize cusolverDnDsyevj_bufferSize
#define gpusolverDnCheevj_bufferSize cusolverDnCheevj_bufferSize
#define gpusolverDnZheevj_bufferSize cusolverDnZheevj_bufferSize
#define gpusolverDnSsyevjBatched cusolverDnSsyevjBatched
#define gpusolverDnDsyevjBatched cusolverDnDsyevjBatched
#define gpusolverDnCheevjBatched cusolverDnCheevjBatched
#define gpusolverDnZheevjBatched cusolverDnZheevjBatched
#define gpusolverDnSsyevjBatched_bufferSize cusolverDnSsyevjBatched_bufferSize
#define gpusolverDnDsyevjBatched_bufferSize cusolverDnDsyevjBatched_bufferSize
#define gpusolverDnCheevjBatched_bufferSize cusolverDnCheevjBatched_bufferSize
#define gpusolverDnZheevjBatched_bufferSize cusolverDnZheevjBatched_bufferSize
#define gpusolverDnSgesvd cusolverDnSgesvd
#define gpusolverDnDgesvd cusolverDnDgesvd
#define gpusolverDnCgesvd cusolverDnCgesvd
#define gpusolverDnZgesvd cusolverDnZgesvd
#define gpusolverDnSgesvd_bufferSize(h, jobu, jobvt, m, n, lwork) \
  cusolverDnSgesvd_bufferSize(h, m, n, lwork)
#define gpusolverDnDgesvd_bufferSize(h, jobu, jobvt, m, n, lwork) \
  cusolverDnDgesvd_bufferSize(h, m, n, lwork)
#define gpusolverDnCgesvd_bufferSize(h, jobu, jobvt, m, n, lwork) \
  cusolverDnCgesvd_bufferSize(h, m, n, lwork)
#define gpusolverDnZgesvd_bufferSize(h, jobu, jobvt, m, n, lwork) \
  cusolverDnZgesvd_bufferSize(h, m, n, lwork)
#define gpusolverDnSsytrd_bufferSize cusolverDnSsytrd_bufferSize
#define gpusolverDnDsytrd_bufferSize cusolverDnDsytrd_bufferSize
#define gpusolverDnChetrd_bufferSize cusolverDnChetrd_bufferSize
#define gpusolverDnZhetrd_bufferSize cusolverDnZhetrd_bufferSize
#define gpusolverDnSsytrd cusolverDnSsytrd
#define gpusolverDnDsytrd cusolverDnDsytrd
#define gpusolverDnChetrd cusolverDnChetrd
#define gpusolverDnZhetrd cusolverDnZhetrd

#define GPUSOLVER_FILL_MODE_LOWER CUBLAS_FILL_MODE_LOWER
#define GPUSOLVER_FILL_MODE_UPPER CUBLAS_FILL_MODE_UPPER
#define GPUSOLVER_EIG_MODE_VECTOR CUSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_STATUS_SUCCESS CUSOLVER_STATUS_SUCCESS

#define gpusparseCooSetStridedBatch cusparseCooSetStridedBatch
#define gpusparseCreate cusparseCreate
#define gpusparseCreateCoo cusparseCreateCoo
#define gpusparseCreateCsr cusparseCreateCsr
#define gpusparseCreateDnMat cusparseCreateDnMat
#define gpusparseCreateDnVec cusparseCreateDnVec
#define gpusparseDenseToSparse_analysis cusparseDenseToSparse_analysis
#define gpusparseDenseToSparse_bufferSize cusparseDenseToSparse_bufferSize
#define gpusparseDenseToSparse_convert cusparseDenseToSparse_convert
#define gpusparseDestroySpMat cusparseDestroySpMat
#define gpusparseDestroyDnMat cusparseDestroyDnMat
#define gpusparseDestroyDnVec cusparseDestroyDnVec
#define gpusparseDnMatSetStridedBatch cusparseDnMatSetStridedBatch
#define gpusparseSetStream cusparseSetStream
#define gpusparseSparseToDense cusparseSparseToDense
#define gpusparseSparseToDense_bufferSize cusparseSparseToDense_bufferSize
#define gpusparseSpMM cusparseSpMM
#define gpusparseSpMM_bufferSize cusparseSpMM_bufferSize
#define gpusparseSpMV cusparseSpMV
#define gpusparseSpMV_bufferSize cusparseSpMV_bufferSize
#define gpusparseSgtsv2 cusparseSgtsv2
#define gpusparseDgtsv2 cusparseDgtsv2
#define gpusparseSgtsv2_bufferSizeExt cusparseSgtsv2_bufferSizeExt
#define gpusparseDgtsv2_bufferSizeExt cusparseDgtsv2_bufferSizeExt

#define GPUSPARSE_INDEX_16U CUSPARSE_INDEX_16U
#define GPUSPARSE_INDEX_32I CUSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_64I CUSPARSE_INDEX_64I
#define GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT CUSPARSE_DENSETOSPARSE_ALG_DEFAULT
#define GPUSPARSE_INDEX_BASE_ZERO CUSPARSE_INDEX_BASE_ZERO
// Use CUSPARSE_SPMV_COO_ALG2 and CUSPARSE_SPMV_CSR_ALG2 for SPMV and
// use CUSPARSE_SPMM_COO_ALG2 and CUSPARSE_SPMM_CSR_ALG3 for SPMM, which
// provide deterministic (bit-wise) results for each run. These indexing modes
// are fully supported (both row- and column-major inputs) in CUSPARSE 11.7.1
// and newer (which was released as part of CUDA 11.8)
#define GPUSPARSE_SPMV_COO_ALG CUSPARSE_SPMV_COO_ALG2
#define GPUSPARSE_SPMV_CSR_ALG CUSPARSE_SPMV_CSR_ALG2
#define GPUSPARSE_SPMM_COO_ALG CUSPARSE_SPMM_COO_ALG2
// In general Cusparse does not support a fully general deterministic CSR SpMM
// algorithm.
// In CUDA versions before 12.2.1, we used ALG3, which is deterministic, but
// does not cover all cases and silently fell back to other algorithms for cases
// it did not cover. CUDA 12.2.1 removed the fallback behavior.
#define GPUSPARSE_SPMM_CSR_ALG CUSPARSE_SPMM_ALG_DEFAULT

#define GPUSPARSE_OPERATION_NON_TRANSPOSE CUSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_OPERATION_TRANSPOSE CUSPARSE_OPERATION_TRANSPOSE
#define GPUSPARSE_ORDER_ROW CUSPARSE_ORDER_ROW
#define GPUSPARSE_SPARSETODENSE_ALG_DEFAULT CUSPARSE_SPARSETODENSE_ALG_DEFAULT
#define GPUSPARSE_STATUS_SUCCESS CUSPARSE_STATUS_SUCCESS

#define GPU_STREAM_CAPTURE_STATUS_ACTIVE CU_STREAM_CAPTURE_STATUS_ACTIVE
#define GPU_STREAM_CAPTURE_MODE_RELAXED CU_STREAM_CAPTURE_MODE_RELAXED
#define GPU_STREAM_NON_BLOCKING CU_STREAM_NON_BLOCKING

#define gpuCtxGetDevice cuCtxGetDevice
#define gpuCtxGetCurrent cuCtxGetCurrent
#define gpuCtxPopCurrent cuCtxPopCurrent
#define gpuCtxPushCurrent cuCtxPushCurrent
#define gpuDeviceGetAttribute cuDeviceGetAttribute
#define gpuEventCreate cuEventCreate
#define gpuEventRecord cuEventRecord
#define gpuEventSynchronize cuEventSynchronize
#define gpuEventElapsedTime cuEventElapsedTime
#define gpuEventDestroy cuEventDestroy
#define gpuFuncGetAttribute cuFuncGetAttribute
#define gpuFuncSetCacheConfig cuFuncSetCacheConfig
#define gpuInit cuInit
#define gpuLaunchKernel cuLaunchKernel
#define gpuMemcpyDtoHAsync cuMemcpyDtoHAsync
#define gpuMemcpyHtoDAsync cuMemcpyHtoDAsync
#define gpuMemsetD8Async cuMemsetD8Async
#define gpuModuleLoadData cuModuleLoadData
#define gpuModuleGetFunction cuModuleGetFunction
#define gpuModuleUnload cuModuleUnload
#define gpuStreamGetCtx cuStreamGetCtx
#define gpuThreadExchangeStreamCaptureMode cuThreadExchangeStreamCaptureMode
#define gpuStreamCreate cuStreamCreate
#define gpuStreamDestroy cuStreamDestroy
#define gpuStreamIsCapturing cuStreamIsCapturing

#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR \
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR \
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#define GPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN \
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
#define GPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR \
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
#define GPU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define GPU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES \
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define GPU_EVENT_DEFAULT CU_EVENT_DEFAULT

#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamWaitEvent cudaStreamWaitEvent
#define gpuSuccess cudaSuccess

namespace jax::JAX_GPU_NAMESPACE {
namespace {
constexpr uint32_t kNumThreadsPerWarp = 32;
}
}  // namespace jax::JAX_GPU_NAMESPACE

#elif defined(JAX_GPU_HIP)

#include "rocm/include/hip/hip_runtime_api.h"
#include "rocm/include/hipblas/hipblas.h"
#include "rocm/include/hipsolver/hipsolver.h"
#include "rocm/include/hipsparse/hipsparse.h"

#define JAX_GPU_NAMESPACE hip
#define JAX_GPU_PREFIX "hip"

#define JAX_GPU_HAVE_SPARSE 1
#define JAX_GPU_HAVE_FP8 0

typedef hipFloatComplex gpuComplex;
typedef hipDoubleComplex gpuDoubleComplex;

typedef hipblasComplex gpublasComplex;
typedef hipblasDoubleComplex gpublasDoubleComplex;
typedef hipsolverHandle_t gpusolverDnHandle_t;
typedef hipblasFillMode_t gpublasFillMode_t;
typedef hipsolverFillMode_t gpusolverFillMode_t;
typedef hipblasHandle_t gpublasHandle_t;
typedef hipblasStatus_t gpublasStatus_t;
typedef hipCtx_t gpuContext_t;
typedef hipStreamCaptureMode gpustreamCaptureMode_t;
typedef hipStreamCaptureStatus gpustreamCaptureStatus_t;
typedef hipDataType gpuDataType;
typedef hipDevice_t gpuDevice_t;
typedef hipDeviceptr_t gpuDevicePtr_t;
typedef hipStream_t gpuStream_t;
typedef hipError_t gpuError_t;
typedef hipEvent_t gpuEvent_t;
typedef hipFunction_t gpuFunction_t;
typedef hipModule_t gpuModule_t;
typedef void gpuSyevjInfo;
typedef hipsolverSyevjInfo_t gpuSyevjInfo_t;
typedef hipsolverEigMode_t gpusolverEigMode_t;
typedef hipsolverStatus_t gpusolverStatus_t;
typedef hipsparseIndexType_t gpusparseIndexType_t;
typedef hipsparseHandle_t gpusparseHandle_t;
typedef hipsparseOperation_t gpusparseOperation_t;
typedef hipsparseStatus_t gpusparseStatus_t;
typedef hipsparseSpMatDescr_t gpusparseSpMatDescr_t;
typedef hipsparseDnMatDescr_t gpusparseDnMatDescr_t;
typedef hipsparseDnVecDescr_t gpusparseDnVecDescr_t;

#define GPU_C_16F HIP_C_16F
#define GPU_R_16F HIP_R_16F
#define GPU_C_32F HIP_C_32F
#define GPU_R_32F HIP_R_32F
#define GPU_C_64F HIP_C_64F
#define GPU_R_64F HIP_R_64F

#define gpublasCreate hipblasCreate
#define gpublasSetStream hipblasSetStream
#define gpublasSgeqrfBatched hipblasSgeqrfBatched
#define gpublasDgeqrfBatched hipblasDgeqrfBatched
#define gpublasCgeqrfBatched hipblasCgeqrfBatched
#define gpublasZgeqrfBatched hipblasZgeqrfBatched
#define gpublasSgetrfBatched hipblasSgetrfBatched
#define gpublasDgetrfBatched hipblasDgetrfBatched
#define gpublasCgetrfBatched hipblasCgetrfBatched
#define gpublasZgetrfBatched hipblasZgetrfBatched

#define GPUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

#define gpusolverDnCreate hipsolverCreate
#define gpusolverDnSetStream hipsolverSetStream
#define gpusolverDnCreateSyevjInfo hipsolverCreateSyevjInfo
#define gpusolverDnDestroySyevjInfo hipsolverDestroySyevjInfo
#define gpusolverDnSgeqrf hipsolverSgeqrf
#define gpusolverDnDgeqrf hipsolverDgeqrf
#define gpusolverDnCgeqrf hipsolverCgeqrf
#define gpusolverDnZgeqrf hipsolverZgeqrf
#define gpusolverDnSgeqrf_bufferSize hipsolverSgeqrf_bufferSize
#define gpusolverDnDgeqrf_bufferSize hipsolverDgeqrf_bufferSize
#define gpusolverDnCgeqrf_bufferSize hipsolverCgeqrf_bufferSize
#define gpusolverDnZgeqrf_bufferSize hipsolverZgeqrf_bufferSize
#define gpusolverDnSgetrf(h, m, n, a, lda, work, lwork, ipiv, info) \
  hipsolverSgetrf(h, m, n, a, lda, work, lwork, ipiv, info)
#define gpusolverDnDgetrf(h, m, n, a, lda, work, lwork, ipiv, info) \
  hipsolverDgetrf(h, m, n, a, lda, work, lwork, ipiv, info)
#define gpusolverDnCgetrf(h, m, n, a, lda, work, lwork, ipiv, info) \
  hipsolverCgetrf(h, m, n, a, lda, work, lwork, ipiv, info)
#define gpusolverDnZgetrf(h, m, n, a, lda, work, lwork, ipiv, info) \
  hipsolverZgetrf(h, m, n, a, lda, work, lwork, ipiv, info)
#define gpusolverDnSgetrf_bufferSize hipsolverSgetrf_bufferSize
#define gpusolverDnDgetrf_bufferSize hipsolverDgetrf_bufferSize
#define gpusolverDnCgetrf_bufferSize hipsolverCgetrf_bufferSize
#define gpusolverDnZgetrf_bufferSize hipsolverZgetrf_bufferSize
#define gpusolverDnSorgqr hipsolverSorgqr
#define gpusolverDnDorgqr hipsolverDorgqr
#define gpusolverDnCungqr hipsolverCungqr
#define gpusolverDnZungqr hipsolverZungqr
#define gpusolverDnSorgqr_bufferSize hipsolverSorgqr_bufferSize
#define gpusolverDnDorgqr_bufferSize hipsolverDorgqr_bufferSize
#define gpusolverDnCungqr_bufferSize hipsolverCungqr_bufferSize
#define gpusolverDnZungqr_bufferSize hipsolverZungqr_bufferSize
#define gpusolverDnSsyevd hipsolverSsyevd
#define gpusolverDnDsyevd hipsolverDsyevd
#define gpusolverDnCheevd hipsolverCheevd
#define gpusolverDnZheevd hipsolverZheevd
#define gpusolverDnSsyevd_bufferSize hipsolverSsyevd_bufferSize
#define gpusolverDnDsyevd_bufferSize hipsolverDsyevd_bufferSize
#define gpusolverDnCheevd_bufferSize hipsolverCheevd_bufferSize
#define gpusolverDnZheevd_bufferSize hipsolverZheevd_bufferSize
#define gpusolverDnSsyevj hipsolverSsyevj
#define gpusolverDnDsyevj hipsolverDsyevj
#define gpusolverDnCheevj hipsolverCheevj
#define gpusolverDnZheevj hipsolverZheevj
#define gpusolverDnSsyevj_bufferSize hipsolverSsyevj_bufferSize
#define gpusolverDnDsyevj_bufferSize hipsolverDsyevj_bufferSize
#define gpusolverDnCheevj_bufferSize hipsolverCheevj_bufferSize
#define gpusolverDnZheevj_bufferSize hipsolverZheevj_bufferSize
#define gpusolverDnSsyevjBatched hipsolverSsyevjBatched
#define gpusolverDnDsyevjBatched hipsolverDsyevjBatched
#define gpusolverDnCheevjBatched hipsolverCheevjBatched
#define gpusolverDnZheevjBatched hipsolverZheevjBatched
#define gpusolverDnSsyevjBatched_bufferSize hipsolverSsyevjBatched_bufferSize
#define gpusolverDnDsyevjBatched_bufferSize hipsolverDsyevjBatched_bufferSize
#define gpusolverDnCheevjBatched_bufferSize hipsolverCheevjBatched_bufferSize
#define gpusolverDnZheevjBatched_bufferSize hipsolverZheevjBatched_bufferSize
#define gpusolverDnSgesvd hipsolverSgesvd
#define gpusolverDnDgesvd hipsolverDgesvd
#define gpusolverDnCgesvd hipsolverCgesvd
#define gpusolverDnZgesvd hipsolverZgesvd
#define gpusolverDnSgesvd_bufferSize(h, jobu, jobvt, m, n, lwork) \
  hipsolverSgesvd_bufferSize(h, jobu, jobvt, m, n, lwork)
#define gpusolverDnDgesvd_bufferSize(h, jobu, jobvt, m, n, lwork) \
  hipsolverDgesvd_bufferSize(h, jobu, jobvt, m, n, lwork)
#define gpusolverDnCgesvd_bufferSize(h, jobu, jobvt, m, n, lwork) \
  hipsolverCgesvd_bufferSize(h, jobu, jobvt, m, n, lwork)
#define gpusolverDnZgesvd_bufferSize(h, jobu, jobvt, m, n, lwork) \
  hipsolverZgesvd_bufferSize(h, jobu, jobvt, m, n, lwork)
#define gpusolverDnSsytrd_bufferSize hipsolverDnSsytrd_bufferSize
#define gpusolverDnDsytrd_bufferSize hipsolverDnDsytrd_bufferSize
#define gpusolverDnChetrd_bufferSize hipsolverDnChetrd_bufferSize
#define gpusolverDnZhetrd_bufferSize hipsolverDnZhetrd_bufferSize
#define gpusolverDnSsytrd hipsolverDnSsytrd
#define gpusolverDnDsytrd hipsolverDnDsytrd
#define gpusolverDnChetrd hipsolverDnChetrd
#define gpusolverDnZhetrd hipsolverDnZhetrd

#define GPUSOLVER_FILL_MODE_LOWER HIPSOLVER_FILL_MODE_LOWER
#define GPUSOLVER_FILL_MODE_UPPER HIPSOLVER_FILL_MODE_UPPER
#define GPUSOLVER_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS

#define gpusparseCooSetStridedBatch hipsparseCooSetStridedBatch
#define gpusparseCreate hipsparseCreate
#define gpusparseSetStream hipsparseSetStream
#define gpusparseCreateCoo hipsparseCreateCoo
#define gpusparseCreateCsr hipsparseCreateCsr
#define gpusparseCreateDnMat hipsparseCreateDnMat
#define gpusparseCreateDnVec hipsparseCreateDnVec
#define gpusparseDenseToSparse_analysis hipsparseDenseToSparse_analysis
#define gpusparseDenseToSparse_bufferSize hipsparseDenseToSparse_bufferSize
#define gpusparseDenseToSparse_convert hipsparseDenseToSparse_convert
#define gpusparseDestroySpMat hipsparseDestroySpMat
#define gpusparseDestroyDnMat hipsparseDestroyDnMat
#define gpusparseDestroyDnVec hipsparseDestroyDnVec
#define gpusparseDnMatSetStridedBatch hipsparseDnMatSetStridedBatch
#define gpusparseSparseToDense hipsparseSparseToDense
#define gpusparseSparseToDense_bufferSize hipsparseSparseToDense_bufferSize
#define gpusparseSpMM hipsparseSpMM
#define gpusparseSpMM_bufferSize hipsparseSpMM_bufferSize
#define gpusparseSpMV hipsparseSpMV
#define gpusparseSpMV_bufferSize hipsparseSpMV_bufferSize
#define gpusparseSgtsv2 hipsparseSgtsv2
#define gpusparseDgtsv2 hipsparseDgtsv2
#define gpusparseSgtsv2_bufferSizeExt hipsparseSgtsv2_bufferSizeExt
#define gpusparseDgtsv2_bufferSizeExt hipsparseDgtsv2_bufferSizeExt

#define GPUSPARSE_INDEX_16U HIPSPARSE_INDEX_16U
#define GPUSPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_64I HIPSPARSE_INDEX_64I
#define GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT
#define GPUSPARSE_SPMV_COO_ALG HIPSPARSE_MV_ALG_DEFAULT
#define GPUSPARSE_SPMV_CSR_ALG HIPSPARSE_MV_ALG_DEFAULT
#define GPUSPARSE_SPMM_COO_ALG HIPSPARSE_SPMM_ALG_DEFAULT
#define GPUSPARSE_SPMM_CSR_ALG HIPSPARSE_SPMM_ALG_DEFAULT
#define GPUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO
#define GPUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_OPERATION_TRANSPOSE HIPSPARSE_OPERATION_TRANSPOSE
#define GPUSPARSE_ORDER_ROW HIPSPARSE_ORDER_ROW
#define GPUSPARSE_SPARSETODENSE_ALG_DEFAULT HIPSPARSE_SPARSETODENSE_ALG_DEFAULT
#define GPUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS

#define GPU_STREAM_CAPTURE_STATUS_ACTIVE hipStreamCaptureStatusActive
#define GPU_STREAM_CAPTURE_MODE_RELAXED hipStreamCaptureModeRelaxed
#define GPU_STREAM_NON_BLOCKING hipStreamNonBlocking

#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamWaitEvent hipStreamWaitEvent
#define gpuSuccess hipSuccess

#define gpuCtxGetDevice hipCtxGetDevice
#define gpuCtxGetCurrent hipCtxGetCurrent
#define gpuCtxPopCurrent hipCtxPopCurrent
#define gpuCtxPushCurrent hipCtxPushCurrent
#define gpuDeviceGet hipDeviceGet
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuDevicePrimaryCtxRetain hipDevicePrimaryCtxRetain
#define gpuEventCreate hipEventCreateWithFlags
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventDestroy hipEventDestroy
#define gpuFuncGetAttribute hipFuncGetAttribute
#define gpuFuncSetCacheConfig hipFuncSetCacheConfig
#define gpuGetStreamDeviceId hipGetStreamDeviceId
#define gpuInit hipInit
#define gpuLaunchKernel hipModuleLaunchKernel
#define gpuModuleGetFunction hipModuleGetFunction
#define gpuModuleLoadData hipModuleLoadData
#define gpuModuleUnload hipModuleUnload
#define gpuMemsetD8Async hipMemsetD8Async
#define gpuMemcpyDtoHAsync hipMemcpyDtoHAsync
#define gpuMemcpyHtoDAsync hipMemcpyHtoDAsync
#define gpuMemsetD8Async hipMemsetD8Async
#define gpuThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define gpuStreamCreate hipStreamCreateWithFlags
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamIsCapturing hipStreamIsCapturing

#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR \
  hipDeviceAttributeComputeCapabilityMajor
#define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR \
  hipDeviceAttributeComputeCapabilityMinor
#define GPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN \
  hipDeviceAttributeMaxSharedMemoryPerBlock
#define GPU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR \
  hipDeviceAttributeMaxBlocksPerMultiProcessor
#define GPU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES \
  HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define GPU_EVENT_DEFAULT hipEventDefault

namespace jax::JAX_GPU_NAMESPACE {
namespace {
constexpr uint32_t kNumThreadsPerWarp = 64;
}
}  // namespace jax::JAX_GPU_NAMESPACE

#else  // defined(GPU vendor)
#error "Either JAX_GPU_CUDA or JAX_GPU_HIP must be defined"
#endif  // defined(GPU vendor)

#endif  // JAXLIB_GPU_VENDOR_H_
