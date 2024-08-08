/* Copyright 2024 The JAX Authors.

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

#include "jaxlib/gpu/blas_kernels_ffi.h"

#include "absl/status/status.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/blas_handle_pool.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

namespace {
#define GETRF_BATCHED_KERNEL_IMPL(type, name)                                 \
  template <>                                                                 \
  struct GetrfBatchedKernel<type> {                                           \
    static absl::Status Run(gpublasHandle_t handle, int n, type** a, int lda, \
                            int* ipiv, int* info, int batch) {                \
      return JAX_AS_STATUS(name(handle, n, a, lda, ipiv, info, batch));       \
    }                                                                         \
  }

template <typename T>
struct GetrfBatchedKernel;
GETRF_BATCHED_KERNEL_IMPL(float, gpublasSgetrfBatched);
GETRF_BATCHED_KERNEL_IMPL(double, gpublasDgetrfBatched);
GETRF_BATCHED_KERNEL_IMPL(gpublasComplex, gpublasCgetrfBatched);
GETRF_BATCHED_KERNEL_IMPL(gpublasDoubleComplex, gpublasZgetrfBatched);
#undef GETRF_BATCHED_KERNEL_IMPL

template <typename T>
ffi::Error GetrfBatchedImpl(gpuStream_t stream, ffi::ScratchAllocator& scratch,
                            ffi::AnyBuffer a, ffi::Result<ffi::AnyBuffer> out,
                            ffi::Result<ffi::Buffer<ffi::DataType::S32>> ipiv,
                            ffi::Result<ffi::Buffer<ffi::DataType::S32>> info) {
  FFI_RETURN_IF_ERROR(CheckMatrixDimensions(a.dimensions()));
  auto [batch, rows, cols] = SplitBatch2D(a.dimensions());
  if (rows != cols) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "getrf_batched only supports square matrices");
  }
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(cols));
  FFI_ASSIGN_OR_RETURN(auto handle, BlasHandlePool::Borrow(stream));

  auto maybe_workspace = scratch.Allocate(sizeof(void*) * batch);
  if (!maybe_workspace.has_value()) {
    return ffi::Error(ffi::ErrorCode::kUnknown,
                      "Unable to allocate workspace for batched getrf");
  }
  auto workspace = maybe_workspace.value();

  auto a_data = a.untyped_data();
  auto out_data = out->untyped_data();
  auto ipiv_data = ipiv->typed_data();
  auto info_data = info->typed_data();
  if (a_data != out_data) {
    FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(
        gpuMemcpyAsync(out_data, a_data, sizeof(T) * batch * cols * cols,
                       gpuMemcpyDeviceToDevice, stream)));
  }

  FFI_ASSIGN_OR_RETURN(
      auto a_ptrs_host,
      MakeBatchPointers(stream, out_data, workspace, batch, sizeof(T) * n * n));
  // TODO(phawkins, danfm): ideally we would not need to synchronize here, but
  // to avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuStreamSynchronize(stream)));

  auto batch_ptrs = static_cast<T**>(workspace);
  FFI_RETURN_IF_ERROR_STATUS(GetrfBatchedKernel<T>::Run(
      handle.get(), n, batch_ptrs, n, ipiv_data, info_data, batch));

  return ffi::Error::Success();
}

ffi::Error GetrfBatchedDispatch(
    gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer a,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> ipiv,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> info) {
  auto dataType = a.element_type();
  if (dataType != out->element_type()) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "Input and output to getrf_batched must have the same element type");
  }
  if (dataType == ffi::DataType::F32) {
    return GetrfBatchedImpl<float>(stream, scratch, a, out, ipiv, info);
  } else if (dataType == ffi::DataType::F64) {
    return GetrfBatchedImpl<double>(stream, scratch, a, out, ipiv, info);
  } else if (dataType == ffi::DataType::C64) {
    return GetrfBatchedImpl<gpublasComplex>(stream, scratch, a, out, ipiv,
                                            info);
  } else if (dataType == ffi::DataType::C128) {
    return GetrfBatchedImpl<gpublasDoubleComplex>(stream, scratch, a, out, ipiv,
                                                  info);
  }
  return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                    "Unsupported element type for getrf");
}
}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GetrfBatchedFfi, GetrfBatchedDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()                   // a
        .Ret<ffi::AnyBuffer>()                   // out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // ipiv
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // info
);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
