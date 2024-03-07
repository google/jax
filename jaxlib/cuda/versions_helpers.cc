/* Copyright 2023 The JAX Authors.

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

#include "jaxlib/cuda/versions_helpers.h"

#include <cstddef>
#include <stdexcept>

#include "absl/base/dynamic_annotations.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"

namespace jax::cuda {

#if CUDA_VERSION < 11080
#error "JAX requires CUDA 11.8 or newer."
#endif  // CUDA_VERSION < 11080

int CudaRuntimeGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cudaRuntimeGetVersion(&version)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}

int CudaDriverGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cudaDriverGetVersion(&version)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}

uint32_t CuptiGetVersion() {
  uint32_t version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cuptiGetVersion(&version)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}

int CufftGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cufftGetVersion(&version)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}

int CusolverGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusolverGetVersion(&version)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}

int CublasGetVersion() {
  int version;
  // NVIDIA promise that it's safe to pass a null pointer as the handle to this
  // function.
  JAX_THROW_IF_ERROR(
      JAX_AS_STATUS(cublasGetVersion(/*handle=*/nullptr, &version)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}

int CusparseGetVersion() {
  // cusparseGetVersion is unhappy if passed a null library handle. But
  // cusparseGetProperty doesn't require one.
  int major, minor, patch;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusparseGetProperty(MAJOR_VERSION, &major)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusparseGetProperty(MINOR_VERSION, &minor)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cusparseGetProperty(PATCH_LEVEL, &patch)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&major, sizeof major);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&minor, sizeof minor);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&patch, sizeof patch);
  return major * 1000 + minor * 100 + patch;
}
size_t CudnnGetVersion() {
  size_t version = ::cudnnGetVersion();
  // If the cudnn stub in TSL can't find the library, it will use a dummy stub
  // that returns 0, since cudnnGetVersion() cannot fail.
  if (version == 0) {
    throw std::runtime_error("cuDNN not found.");
  }
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}
int CudaComputeCapability(int device) {
  int major, minor;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpuInit(0)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpuDeviceGetAttribute(
      &major, GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(gpuDeviceGetAttribute(
      &minor, GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&major, sizeof major);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&minor, sizeof minor);
  return major * 10 + minor;
}

int CudaDeviceCount() {
  int device_count = 0;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cuInit(0)));
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(cuDeviceGetCount(&device_count)));

  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&device_count, sizeof device_count);
  return device_count;
}


}  // namespace jax::cuda
