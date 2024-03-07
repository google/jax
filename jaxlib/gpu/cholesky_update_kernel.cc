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
#include "jaxlib/gpu/cholesky_update_kernel.h"
#include <cstddef>
#include <string_view>

#include "absl/status/status.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {


absl::Status CholeskyUpdateImpl(gpuStream_t stream, void** buffers,
                                const char* opaque, std::size_t opaque_len) {
  auto s = UnpackDescriptor<CholeskyUpdateDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const CholeskyUpdateDescriptor& d = **s;
  LaunchCholeskyUpdateKernel(stream, buffers, d);
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuGetLastError()));
  return absl::OkStatus();
}

}  // namespace

void CholeskyUpdate(gpuStream_t stream, void** buffers,
                    const char* opaque, size_t opaque_len,
                    XlaCustomCallStatus* status) {
  auto s = CholeskyUpdateImpl(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    std::string_view message = s.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
