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

#include "tests/ffi/ffi_cpu_kernels.h"

#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tests {

namespace ffi = ::xla::ffi;

ffi::Error AddToImpl(std::int32_t delta, ffi::Buffer<ffi::S32> input,
                     ffi::Result<ffi::Buffer<ffi::S32>> output) {
  if (input.element_count() != output->element_count()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Input and output must have the same size");
  }
  for (auto n = 0; n < input.element_count(); ++n) {
    output->typed_data()[n] = input.typed_data()[n] + delta;
  }
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(AddTo, AddToImpl,
                              ffi::Ffi::Bind()
                                  .Attr<std::int32_t>("delta")
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Ret<ffi::Buffer<ffi::S32>>());

ffi::Error ShouldFailImpl(bool should_fail,
                          ffi::Result<ffi::BufferR0<ffi::S32>> output) {
  if (should_fail) {
    output->typed_data()[0] = 0;
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Test should error.");
  }
  output->typed_data()[0] = 1;
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ShouldFail, ShouldFailImpl,
    ffi::Ffi::Bind()
        .Attr<bool>("should_fail")
        // An output is required so that this doesn't get optimized away.
        .Ret<ffi::BufferR0<ffi::S32>>());

}  // namespace tests
}  // namespace jax
