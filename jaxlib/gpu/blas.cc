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

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "jaxlib/gpu/blas_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/tsl/python/lib/core/numpy.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace nb = nanobind;

// Converts a NumPy dtype to a Type.
BlasType DtypeToBlasType(const dtype& np_type) {
  static auto* types = new absl::flat_hash_map<std::pair<char, int>, BlasType>({
      {{'f', 4}, BlasType::F32},
      {{'f', 8}, BlasType::F64},
      {{'c', 8}, BlasType::C64},
      {{'c', 16}, BlasType::C128},
  });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    nb::str repr = nb::repr(np_type);
    throw std::invalid_argument(
        absl::StrFormat("Unsupported dtype %s", repr.c_str()));
  }
  return it->second;
}

// Returns the descriptor for a GetrfBatched operation.
std::pair<size_t, nb::bytes> BuildGetrfBatchedDescriptor(const dtype& dtype,
                                                         int b, int n) {
  BlasType type = DtypeToBlasType(dtype);
  size_t size = b * sizeof(void*);
  return {size, PackDescriptor(GetrfBatchedDescriptor{type, b, n})};
}

// Returns the descriptor for a GetrfBatched operation.
std::pair<size_t, nb::bytes> BuildGeqrfBatchedDescriptor(const dtype& dtype,
                                                         int b, int m, int n) {
  BlasType type = DtypeToBlasType(dtype);
  size_t size = b * sizeof(void*);
  return {size, PackDescriptor(GeqrfBatchedDescriptor{type, b, m, n})};
}

nb::dict Registrations() {
  nb::dict dict;
  dict[JAX_GPU_PREFIX "blas_getrf_batched"] = EncapsulateFunction(GetrfBatched);
  dict[JAX_GPU_PREFIX "blas_geqrf_batched"] = EncapsulateFunction(GeqrfBatched);
  return dict;
}

NB_MODULE(_blas, m) {
  tsl::ImportNumpy();

  m.def("registrations", &Registrations);
  m.def("build_getrf_batched_descriptor", &BuildGetrfBatchedDescriptor);
  m.def("build_geqrf_batched_descriptor", &BuildGeqrfBatchedDescriptor);
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
