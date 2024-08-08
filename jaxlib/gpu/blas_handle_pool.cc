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

#include "jaxlib/gpu/blas_handle_pool.h"

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/handle_pool.h"

namespace jax {

template <>
/*static*/ absl::StatusOr<BlasHandlePool::Handle> BlasHandlePool::Borrow(
    gpuStream_t stream) {
  BlasHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  gpublasHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpublasCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpublasSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

}  // namespace jax
