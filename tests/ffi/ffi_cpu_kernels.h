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

#ifndef JAX_TESTS_FFI_CPU_KERNELS_H_
#define JAX_TESTS_FFI_CPU_KERNELS_H_

#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tests {

XLA_FFI_DECLARE_HANDLER_SYMBOL(AddTo);
XLA_FFI_DECLARE_HANDLER_SYMBOL(ShouldFail);

}  // namespace tests
}  // namespace jax

#endif  // JAX_TESTS_FFI_CPU_KERNELS_H_
