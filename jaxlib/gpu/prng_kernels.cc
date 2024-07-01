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

#include "jaxlib/gpu/prng_kernels.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "absl/algorithm/container.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_status.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = xla::ffi;

namespace {

// TODO(b/338022728): old custom call target, remove after 6 months
absl::Status ThreeFry2x32_(gpuStream_t stream, void** buffers,
                           const char* opaque, std::size_t opaque_len) {
  auto s = UnpackDescriptor<ThreeFry2x32Descriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  LaunchThreeFry2x32Kernel(stream, buffers, **s);
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuGetLastError()));
  return absl::OkStatus();
}

}  // namespace

// TODO(b/338022728): remove after 6 months
void ThreeFry2x32(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = ThreeFry2x32_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    std::string_view message = s.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}

XLA_FFI_Error* ThreeFry2x32Ffi(XLA_FFI_CallFrame* call_frame) {
  static const auto* kImpl =
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<gpuStream_t>>()
          .Arg<ffi::Buffer<ffi::DataType::U32>>()
          .Arg<ffi::Buffer<ffi::DataType::U32>>()
          .Arg<ffi::Buffer<ffi::DataType::U32>>()
          .Arg<ffi::Buffer<ffi::DataType::U32>>()
          .Ret<ffi::Buffer<ffi::DataType::U32>>()
          .Ret<ffi::Buffer<ffi::DataType::U32>>()
          .To([](gpuStream_t stream, auto keys0, auto keys1, auto data0,
                 auto data1, auto out0, auto out1) -> ffi::Error {
            std::int64_t n = absl::c_accumulate(out0->dimensions, 1,
                                                std::multiplies<int64_t>());
            LaunchThreeFry2x32KernelFfi(stream, n, keys0.data, keys1.data,
                                        data0.data, data1.data, out0->data,
                                        out1->data);
            if (auto status = JAX_AS_STATUS(gpuGetLastError()); !status.ok()) {
              return ffi::Error(static_cast<XLA_FFI_Error_Code>(status.code()),
                                std::string(status.message()));
            }
            return ffi::Error::Success();
          })
          .release();
  return kImpl->Call(call_frame);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
