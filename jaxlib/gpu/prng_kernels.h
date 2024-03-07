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

#ifndef JAXLIB_GPU_PRNG_KERNELS_H_
#define JAXLIB_GPU_PRNG_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/service/custom_call_status.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

// TODO(b/338022728): remove after 6 months
struct ThreeFry2x32Descriptor {
  std::int64_t n;  // If -1 then the length is passed as a 5th operand
};

// TODO(b/338022728): remove after 6 months
void LaunchThreeFry2x32Kernel(gpuStream_t stream, void** buffers,
                              ThreeFry2x32Descriptor descriptor);

// TODO(b/338022728): remove after 6 months
void ThreeFry2x32(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status);

XLA_FFI_Error* ThreeFry2x32Ffi(XLA_FFI_CallFrame* call_frame);

void LaunchThreeFry2x32KernelFfi(gpuStream_t stream,
                                 std::int64_t n,
                                 std::uint32_t *keys0, std::uint32_t *keys1,
                                 std::uint32_t *data0, std::uint32_t *data1,
                                 std::uint32_t *out0, std::uint32_t *out1);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

#endif  // JAXLIB_GPU_PRNG_KERNELS_H_
