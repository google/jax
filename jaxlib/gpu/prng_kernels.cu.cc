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

#include <array>
#include <cstddef>
#include <cstdint>

#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

__global__ void ThreeFry2x32Kernel(const std::uint32_t* key0,
                                   const std::uint32_t* key1,
                                   const std::uint32_t* data0,
                                   const std::uint32_t* data1,
                                   std::uint32_t* out0, std::uint32_t* out1,
                                   std::int64_t n, const std::int64_t* n_ptr) {
  if (n < 0) {
    // n is stored in device memory.
    n = *n_ptr;
  }

  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    // Rotation distances specified by the Threefry2x32 algorithm.
    std::uint32_t rotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};
    std::uint32_t x[2];
    std::uint32_t ks[3];

    // 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
    ks[2] = 0x1BD11BDA;

    ks[0] = key0[idx];
    x[0] = data0[idx];
    ks[2] = ks[2] ^ key0[idx];

    ks[1] = key1[idx];
    x[1] = data1[idx];
    ks[2] = ks[2] ^ key1[idx];

    auto rotate_left = [](std::uint32_t v, std::uint32_t distance) {
      return (v << distance) | (v >> (32 - distance));
    };

    // Performs a single round of the Threefry2x32 algorithm, with a rotation
    // amount 'rotation'.
    auto round = [&](std::uint32_t* v, std::uint32_t rotation) {
      v[0] += v[1];
      v[1] = rotate_left(v[1], rotation);
      v[1] ^= v[0];
    };

    // There are no known statistical flaws with 13 rounds of Threefry2x32.
    // We are conservative and use 20 rounds.
    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1];
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 1u;
    for (int i = 4; i < 8; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[2];
    x[1] = x[1] + ks[0] + 2u;
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1] + 3u;
    for (int i = 4; i < 8; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 4u;
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    out0[idx] = x[0] + ks[2];
    out1[idx] = x[1] + ks[0] + 5u;
  }
}

}  // namespace

void LaunchThreeFry2x32KernelFfi(gpuStream_t stream,
                                 std::int64_t n,
                                 std::uint32_t *keys0,
                                 std::uint32_t *keys1,
                                 std::uint32_t *data0,
                                 std::uint32_t *data1,
                                 std::uint32_t *out0,
                                 std::uint32_t *out1) {
  const int block_dim = 128;
  const std::int64_t grid_dim =
      std::min<std::int64_t>(1024, (n + block_dim - 1) / block_dim);
  ThreeFry2x32Kernel<<<grid_dim, block_dim, /*dynamic_shared_mem_bytes=*/0,
                       stream>>>(keys0, keys1, data0, data1, out0,
                                 out1, n, nullptr);
}

// TODO(b/338022728): remove after 6 months
void LaunchThreeFry2x32Kernel(gpuStream_t stream, void** buffers,
                              ThreeFry2x32Descriptor descriptor) {
  std::array<const std::uint32_t*, 2> keys;
  keys[0] = reinterpret_cast<const std::uint32_t*>(buffers[0]);
  keys[1] = reinterpret_cast<const std::uint32_t*>(buffers[1]);
  std::array<const std::uint32_t*, 2> data;
  data[0] = reinterpret_cast<const std::uint32_t*>(buffers[2]);
  data[1] = reinterpret_cast<const std::uint32_t*>(buffers[3]);
  std::int64_t n = descriptor.n;
  int output_idx = 4;
  std::int64_t* n_ptr = nullptr;
  if (n < 0) {
    // n is an operand in device memory.
    n_ptr = reinterpret_cast<std::int64_t*>(buffers[4]);
    output_idx = 5;
  }

  std::array<std::uint32_t*, 2> out;
  out[0] = reinterpret_cast<std::uint32_t*>(buffers[output_idx]);
  out[1] = reinterpret_cast<std::uint32_t*>(buffers[output_idx + 1]);
  const int block_dim = 128;
  const std::int64_t grid_dim =
      n < 0 ? 32
            : std::min<std::int64_t>(1024, (n + block_dim - 1) / block_dim);
  ThreeFry2x32Kernel<<<grid_dim, block_dim, /*dynamic_shared_mem_bytes=*/0,
                       stream>>>(keys[0], keys[1], data[0], data[1], out[0],
                                 out[1], n, n_ptr);
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
