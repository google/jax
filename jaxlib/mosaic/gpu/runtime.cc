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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "third_party/gpus/cuda/include/cuda.h"

extern "C" {

void mosaic_gpu_init_tma_desc(CUtensorMap *tma_desc, void *base_addr,
                              int64_t elem_bytewidth, int64_t rank,
                              int64_t *sizes, int64_t *strides,
                              int64_t swizzle_bytes, int64_t *window_shape) {
  CUtensorMapDataType data_type;
  if (elem_bytewidth == 1) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (elem_bytewidth == 2) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
  } else if (elem_bytewidth == 4) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
  } else if (elem_bytewidth == 8) {
    data_type = CU_TENSOR_MAP_DATA_TYPE_UINT64;
  } else {
    fprintf(stderr, "Unsupported element size: %ld\n", elem_bytewidth);
    abort();
  }
  cuuint64_t tma_sizes[5] = {1, 1, 1, 1, 1};
  for (int i = 0; i < rank; ++i) {
    tma_sizes[i] = static_cast<cuuint64_t>(sizes[rank - i - 1]);
  }
  cuuint64_t tma_strides[5] = {1, 1, 1, 1, 1};
  if (strides[rank - 1] != 1) {
    fprintf(stderr, "Minormost stride must be 1, but got %ld\n",
            strides[rank - 1]);
    abort();
  }
  for (int i = 0; i < rank - 1; ++i) {  // We skip the implicit minor stride.
    tma_strides[i] =
        static_cast<cuuint64_t>(strides[rank - i - 2] * elem_bytewidth);
  }
  cuuint32_t tma_window_shape[5] = {1, 1, 1, 1, 1};
  for (int64_t i = 0; i < rank; ++i) {
    tma_window_shape[i] = static_cast<cuuint32_t>(window_shape[rank - i - 1]);
  }
  cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
  CUtensorMapSwizzle swizzle;
  if (swizzle_bytes == 0) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  } else if (swizzle_bytes == 32) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  } else if (swizzle_bytes == 64) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (swizzle_bytes == 128) {
    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
  } else {
    fprintf(stderr, "Unsupported swizzle: %ld\n", swizzle_bytes);
    abort();
  }
  CUresult result = cuTensorMapEncodeTiled(
      tma_desc, data_type, rank, base_addr, tma_sizes, tma_strides,
      tma_window_shape, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", ptr);
    abort();
  }
}

void mosaic_gpu_memcpy_async_h2d(CUdeviceptr dst, void *src, uint64_t bytes,
                                 CUstream stream) {
  CUresult result = cuMemcpyHtoDAsync(dst, src, bytes, stream);
  if (result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuMemcpyAsync failed: %s\n", ptr);
    abort();
  }
}

void* mosaic_gpu_module_load(void *data) {
  CUmodule module = nullptr;
  if (auto result = cuModuleLoadData(&module, data); result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuModuleLoadData failed: %s\n", ptr);
    abort();
  }
  return module;
}

void *mosaic_gpu_get_function(CUmodule module, const char *name,
                              int32_t smem_bytes) {
  CUfunction function = nullptr;
  CUresult result = cuModuleGetFunction(&function, module, name);
  if (result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuModuleGetFunction failed: %s\n", ptr);
    abort();
  }
  if (smem_bytes) {
    result = cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
    if (result != CUDA_SUCCESS) {
      const char *ptr = nullptr;
      cuGetErrorString(result, &ptr);
      fprintf(stderr, "cuFuncSetAttribute failed: %s\n", ptr);
      abort();
    }
  }
  return function;
}

void mosaic_gpu_launch_kernel(CUfunction function, int64_t grid_x,
                              int64_t grid_y, int64_t grid_z, int64_t block_x,
                              int64_t block_y, int64_t block_z,
                              int32_t smem_bytes, CUstream stream,
                              void **params) {
  CUresult result =
      cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y,
                     block_z, smem_bytes, stream, params, nullptr);
  if (result != CUDA_SUCCESS) {
    const char *ptr = nullptr;
    cuGetErrorString(result, &ptr);
    fprintf(stderr, "cuLaunchKernel failed: %s\n", ptr);
    abort();
  }
}

}
