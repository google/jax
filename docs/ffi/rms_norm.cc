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

#include <cmath>
#include <cstdint>
#include <utility>
#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// This is the example "library function" that we want to expose to JAX. This
// isn't meant to be a particularly good implementation, it's just here as a
// placeholder for the purposes of this tutorial.
float ComputeRmsNorm(float eps, int64_t size, const float *x, float *y) {
  float sm = 0.0f;
  for (int64_t n = 0; n < size; ++n) {
    sm += x[n] * x[n];
  }
  float scale = 1.0f / std::sqrt(sm / float(size) + eps);
  for (int64_t n = 0; n < size; ++n) {
    y[n] = x[n] * scale;
  }
  return scale;
}

// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// In this example, we treat all leading dimensions as batch dimensions, so this
// function returns the total number of elements in the buffer, and the size of
// the last dimension.
std::pair<int64_t, int64_t> GetDims(ffi::Span<const int64_t> dims) {
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  int64_t lastDim = dims.back();
  int64_t totalSize = 1;
  for (auto n : dims) {
    totalSize *= n;
  }
  return std::make_pair(totalSize, lastDim);
}

// A wrapper function providing the interface between the XLA FFI call and our
// library function `ComputeRmsNorm` above. This function handles the batch
// dimensions by calling `ComputeRmsNorm` within a loop.
ffi::Error RmsNormImpl(float eps, ffi::Buffer<ffi::DataType::F32> x,
                       ffi::Result<ffi::Buffer<ffi::DataType::F32>> y) {
  auto [totalSize, lastDim] = GetDims(x.dimensions);
  if (lastDim == 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RmsNorm input must be an array");
  }
  for (int64_t n = 0; n < totalSize; n += lastDim) {
    ComputeRmsNorm(eps, lastDim, &(x.data[n]), &(y->data[n]));
  }
  return ffi::Error::Success();
}

// Wrap `RmsNormImpl` and specify the interface to XLA.
XLA_FFI_DEFINE_HANDLER(RmsNorm, RmsNormImpl,
                       ffi::Ffi::Bind()
                           .Attr<float>("eps")
                           .Arg<ffi::Buffer<ffi::DataType::F32>>(/* x */)
                           .Ret<ffi::Buffer<ffi::DataType::F32>>(/* y */));

ffi::Error RmsNormFwdImpl(float eps, ffi::Buffer<ffi::DataType::F32> x,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> y,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> res) {
  auto [totalSize, lastDim] = GetDims(x.dimensions);
  if (lastDim == 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RmsNormFwd input must be an array");
  }
  for (int64_t n = 0, idx = 0; n < totalSize; n += lastDim, ++idx) {
    res->data[idx] = ComputeRmsNorm(eps, lastDim, &(x.data[n]), &(y->data[n]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(RmsNormFwd, RmsNormFwdImpl,
                       ffi::Ffi::Bind()
                           .Attr<float>("eps")
                           .Arg<ffi::Buffer<ffi::DataType::F32>>(/* x */)
                           .Ret<ffi::Buffer<ffi::DataType::F32>>(/* y */)
                           .Ret<ffi::Buffer<ffi::DataType::F32>>(/* res */));

void ComputeRmsNormBwd(int64_t size, float res, const float *x,
                       const float *ct_y, float *ct_x) {
  float ct_res = 0.0f;
  for (int64_t n = 0; n < size; ++n) {
    ct_res += x[n] * ct_y[n];
  }
  float factor = ct_res * res * res * res / float(size);
  for (int64_t n = 0; n < size; ++n) {
    ct_x[n] = res * ct_y[n] - factor * x[n];
  }
}

ffi::Error RmsNormBwdImpl(ffi::Buffer<ffi::DataType::F32> res,
                          ffi::Buffer<ffi::DataType::F32> x,
                          ffi::Buffer<ffi::DataType::F32> ct_y,
                          ffi::Result<ffi::Buffer<ffi::DataType::F32>> ct_x) {
  auto [totalSize, lastDim] = GetDims(x.dimensions);
  if (lastDim == 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RmsNormBwd inputs must be arrays");
  }
  for (int64_t n = 0, idx = 0; n < totalSize; n += lastDim, ++idx) {
    ComputeRmsNormBwd(lastDim, res.data[idx], &(x.data[n]), &(ct_y.data[n]),
                      &(ct_x->data[n]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(RmsNormBwd, RmsNormBwdImpl,
                       ffi::Ffi::Bind()
                           .Arg<ffi::Buffer<ffi::DataType::F32>>(/* res */)
                           .Arg<ffi::Buffer<ffi::DataType::F32>>(/* x */)
                           .Arg<ffi::Buffer<ffi::DataType::F32>>(/* ct_y */)
                           .Ret<ffi::Buffer<ffi::DataType::F32>>(/* ct_x */));

template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

NB_MODULE(rms_norm, m) {
  m.def("rms_norm", []() { return EncapsulateFfiCall(RmsNorm); });
  m.def("rms_norm_fwd", []() { return EncapsulateFfiCall(RmsNormFwd); });
  m.def("rms_norm_bwd", []() { return EncapsulateFfiCall(RmsNormBwd); });
}
