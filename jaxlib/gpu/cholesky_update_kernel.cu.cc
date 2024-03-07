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
#include <stdio.h>

#ifdef JAX_GPU_HIP
#include "rocm/include/hip/amd_detail/amd_hip_cooperative_groups.h"
#else // JAX_GPU_CUDA
#include "third_party/gpus/cuda/include/cooperative_groups.h"
#endif

namespace cg = cooperative_groups;

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {


template <typename T>
__device__ void drotg(T* da, T* db, T* c, T* s) {
  if (*db == 0) {
    *c = 1.;
    *s = 0.;
    return;
  }
  T denominator = max(abs(*da), abs(*db));
  T a = *da / denominator;
  T b = *db / denominator;
  T rh = rhypot(a, b);
  *c = a * rh;
  *s = -(b * rh);
  return;
}

template <typename T>
__global__ void CholeskyUpdateKernel(
    T* rMatrix, T* uVector,
    int nSize) {

  cg::grid_group grid = cg::this_grid();
  int k = grid.thread_rank();

  T c, s;

  for (int step = 0; step < 2 * nSize; ++step) {
    grid.sync();

    int i = step - k;
    if (i < k || i >= nSize || k >= nSize) {
      continue;
    }
    if (i == k) {
      drotg(
          rMatrix + k * nSize + k,
          uVector + k,
          &c,
          &s);
    }
    T r_i = c * rMatrix[k * nSize + i] - s * uVector[i];
    uVector[i] = s * rMatrix[k * nSize + i] + c * uVector[i];
    rMatrix[k * nSize + i] = r_i;
  }
}
}  // namespace


template <typename T>
void LaunchCholeskyUpdateKernelBody(
    gpuStream_t stream, void** buffers,
    int grid_dim, int block_dim, int nSize) {
  T* rMatrix = reinterpret_cast<T*>(buffers[2]);
  T* uVector = reinterpret_cast<T*>(buffers[3]);

  void* arg_ptrs[3] = {
      reinterpret_cast<void*>(&rMatrix),
      reinterpret_cast<void*>(&uVector),
      reinterpret_cast<void*>(&nSize),
  };
#ifdef JAX_GPU_HIP
  hipLaunchCooperativeKernel(
      (void*) CholeskyUpdateKernel<T>, grid_dim, block_dim, arg_ptrs,
      /*dynamic_shared_mem_bytes=*/ 0, stream);
#else // JAX_GPU_CUDA
  cudaLaunchCooperativeKernel(
      (void*) CholeskyUpdateKernel<T>, grid_dim, block_dim, arg_ptrs,
      /*dynamic_shared_mem_bytes=*/ 0, stream);
#endif
}


void LaunchCholeskyUpdateKernel(
    gpuStream_t stream, void** buffers,
    CholeskyUpdateDescriptor descriptor) {

  int nSize = descriptor.matrix_size;
  LinalgType type = descriptor.linalg_type;

  int dev = 0;
#ifdef JAX_GPU_HIP
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, dev);
#else // JAX_GPU_CUDA
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
#endif

  int block_dim = deviceProp.maxThreadsPerBlock;
  int grid_dim = deviceProp.multiProcessorCount;

  switch (type) {
    case LinalgType::F64:
      LaunchCholeskyUpdateKernelBody<double>(
          stream, buffers, grid_dim, block_dim, nSize);
      break;
    case LinalgType::F32:
      LaunchCholeskyUpdateKernelBody<float>(
          stream, buffers, grid_dim, block_dim, nSize);
      break;
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
