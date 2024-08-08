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

#ifndef JAXLIB_GPU_SOLVER_HANDLE_POOL_H_
#define JAXLIB_GPU_SOLVER_HANDLE_POOL_H_

#include "absl/status/statusor.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/handle_pool.h"

#ifdef JAX_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverSp.h"
#endif  // JAX_GPU_CUDA

namespace jax {

using SolverHandlePool = HandlePool<gpusolverDnHandle_t, gpuStream_t>;

template <>
absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
    gpuStream_t stream);

#ifdef JAX_GPU_CUDA
using SpSolverHandlePool = HandlePool<cusolverSpHandle_t, gpuStream_t>;

template <>
absl::StatusOr<SpSolverHandlePool::Handle> SpSolverHandlePool::Borrow(
    gpuStream_t stream);
#endif  // JAX_GPU_CUDA

}  // namespace jax

#endif  // JAXLIB_GPU_SOLVER_HANDLE_POOL_H_
