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

#include "jaxlib/gpu/solver_handle_pool.h"

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/handle_pool.h"

#ifdef JAX_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverSp.h"
#endif  // JAX_GPU_CUDA

namespace jax {

template <>
/*static*/ absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
    gpuStream_t stream) {
  SolverHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  gpusolverDnHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

#ifdef JAX_GPU_CUDA

template <>
/*static*/ absl::StatusOr<SpSolverHandlePool::Handle>
SpSolverHandlePool::Borrow(gpuStream_t stream) {
  SpSolverHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  cusolverSpHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverSpSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

#endif  // JAX_GPU_CUDA

}  // namespace jax
