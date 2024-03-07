/* Copyright 2021 The JAX Authors.

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

#include "jaxlib/gpu/lu_pivot_kernels.h"

#include <array>
#include <cstdint>
#include <iostream>

#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

__device__ void ComputePermutation(const std::int32_t* pivots,
                                   std::int32_t* permutation_out,
                                   const std::int32_t pivot_size,
                                   const std::int32_t permutation_size) {
  for (int i = 0; i < permutation_size; ++i) {
    permutation_out[i] = i;
  }

  // Compute the permutation from a sequence of transpositions encoded in the
  // pivot array by applying the transpositions in order on the identity
  // permutation.
  for (int i = 0; i < pivot_size; ++i) {
    if ((pivots[i] < 0) || (pivots[i] >= permutation_size)) {
      continue;
    }
    std::int32_t swap_temporary = permutation_out[i];
    permutation_out[i] = permutation_out[pivots[i]];
    permutation_out[pivots[i]] = swap_temporary;
  }
}

__global__ void LuPivotsToPermutationKernel(
    const std::int32_t* pivots, std::int32_t* permutation_out,
    const std::int64_t batch_size, const std::int32_t pivot_size,
    const std::int32_t permutation_size) {
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < batch_size; idx += blockDim.x * gridDim.x) {
    // Fill in the output array with the identity permutation.
    ComputePermutation(pivots + idx * pivot_size,
                       permutation_out + idx * permutation_size, pivot_size,
                       permutation_size);
  }
}

}  // namespace

void LaunchLuPivotsToPermutationKernel(gpuStream_t stream,
                                       std::int64_t batch_size,
                                       std::int32_t pivot_size,
                                       std::int32_t permutation_size,
                                       const std::int32_t* pivots,
                                       std::int32_t* permutation) {
  const int block_dim = 128;
  const std::int64_t grid_dim =
      std::min<std::int64_t>(1024, (batch_size + block_dim - 1) / block_dim);

  LuPivotsToPermutationKernel<<<grid_dim, block_dim,
                                /*dynamic_shared_mem_bytes=*/0, stream>>>(
      pivots, permutation, batch_size, pivot_size, permutation_size);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
