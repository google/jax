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

#include "jaxlib/gpu/blas_kernels.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/gpu/blas_handle_pool.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"

namespace jax {

namespace JAX_GPU_NAMESPACE {

namespace {

int SizeOfBlasType(BlasType type) {
  switch (type) {
    case BlasType::F32:
      return sizeof(float);
    case BlasType::F64:
      return sizeof(double);
    case BlasType::C64:
      return sizeof(gpublasComplex);
    case BlasType::C128:
      return sizeof(gpublasDoubleComplex);
  }
}

}  // namespace

// Batched LU decomposition: getrfbatched

static absl::Status GetrfBatched_(gpuStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GetrfBatchedDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GetrfBatchedDescriptor& d = **s;
  auto h = BlasHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[0] != buffers[1]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0], SizeOfBlasType(d.type) * d.batch * d.n * d.n,
        gpuMemcpyDeviceToDevice, stream)));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[4], d.batch,
                                       SizeOfBlasType(d.type) * d.n * d.n);
  JAX_RETURN_IF_ERROR(a_ptrs_host.status());
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuStreamSynchronize(stream)));
  switch (d.type) {
    case BlasType::F32: {
      float** batch_ptrs = static_cast<float**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpublasSgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case BlasType::F64: {
      double** batch_ptrs = static_cast<double**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpublasDgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case BlasType::C64: {
      gpublasComplex** batch_ptrs = static_cast<gpublasComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpublasCgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
    case BlasType::C128: {
      gpublasDoubleComplex** batch_ptrs =
          static_cast<gpublasDoubleComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpublasZgetrfBatched(
          handle.get(), d.n, batch_ptrs, d.n, ipiv, info, d.batch)));
      break;
    }
  }
  return absl::OkStatus();
}

void GetrfBatched(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = GetrfBatched_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

// Batched QR decomposition: geqrfbatched

static absl::Status GeqrfBatched_(gpuStream_t stream, void** buffers,
                                  const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<GeqrfBatchedDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const GeqrfBatchedDescriptor& d = **s;
  auto h = BlasHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;
  if (buffers[0] != buffers[1]) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuMemcpyAsync(
        buffers[1], buffers[0], SizeOfBlasType(d.type) * d.batch * d.m * d.n,
        gpuMemcpyDeviceToDevice, stream)));
  }

  std::vector<int> info(d.batch);
  auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[3], d.batch,
                                       SizeOfBlasType(d.type) * d.m * d.n);
  JAX_RETURN_IF_ERROR(a_ptrs_host.status());
  auto tau_ptrs_host =
      MakeBatchPointers(stream, buffers[2], buffers[4], d.batch,
                        SizeOfBlasType(d.type) * std::min(d.m, d.n));
  JAX_RETURN_IF_ERROR(tau_ptrs_host.status());
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuStreamSynchronize(stream)));
  switch (d.type) {
    case BlasType::F32: {
      float** a_batch_ptrs = static_cast<float**>(buffers[3]);
      float** tau_batch_ptrs = static_cast<float**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
          gpublasSgeqrfBatched(handle.get(), d.m, d.n, a_batch_ptrs, d.m,
                               tau_batch_ptrs, info.data(), d.batch)));
      break;
    }
    case BlasType::F64: {
      double** a_batch_ptrs = static_cast<double**>(buffers[3]);
      double** tau_batch_ptrs = static_cast<double**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
          gpublasDgeqrfBatched(handle.get(), d.m, d.n, a_batch_ptrs, d.m,
                               tau_batch_ptrs, info.data(), d.batch)));
      break;
    }
    case BlasType::C64: {
      gpublasComplex** a_batch_ptrs = static_cast<gpublasComplex**>(buffers[3]);
      gpublasComplex** tau_batch_ptrs =
          static_cast<gpublasComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
          gpublasCgeqrfBatched(handle.get(), d.m, d.n, a_batch_ptrs, d.m,
                               tau_batch_ptrs, info.data(), d.batch)));
      break;
    }
    case BlasType::C128: {
      gpublasDoubleComplex** a_batch_ptrs =
          static_cast<gpublasDoubleComplex**>(buffers[3]);
      gpublasDoubleComplex** tau_batch_ptrs =
          static_cast<gpublasDoubleComplex**>(buffers[4]);
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
          gpublasZgeqrfBatched(handle.get(), d.m, d.n, a_batch_ptrs, d.m,
                               tau_batch_ptrs, info.data(), d.batch)));
      break;
    }
  }
  auto it =
      std::find_if(info.begin(), info.end(), [](int i) { return i != 0; });

  if (it != info.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("QR decomposition failed with status %d for batch "
                        "element %d",
                        *it, std::distance(info.begin(), it)));
  }

  return absl::OkStatus();
}

void GeqrfBatched(gpuStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = GeqrfBatched_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
