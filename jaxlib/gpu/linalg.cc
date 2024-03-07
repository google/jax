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

#include "nanobind/nanobind.h"
#include "jaxlib/gpu/cholesky_update_kernel.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/lu_pivot_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/tsl/python/lib/core/numpy.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

namespace nb = nanobind;


nb::bytes BuildCholeskyUpdateDescriptor(
    dtype np_type,
    std::int64_t matrix_size) {

  LinalgType linalg_type = (
      np_type.itemsize() == 4 ? LinalgType::F32 : LinalgType::F64);

  return PackDescriptor(CholeskyUpdateDescriptor{linalg_type, matrix_size});
}

NB_MODULE(_linalg, m) {
  tsl::ImportNumpy();
  m.def("registrations", []() {
    nb::dict dict;
    dict[JAX_GPU_PREFIX "_lu_pivots_to_permutation"] =
        nb::capsule(reinterpret_cast<void*>(+LuPivotsToPermutation));
    dict["cu_cholesky_update"] = EncapsulateFunction(CholeskyUpdate);
    return dict;
  });
  m.def("build_cholesky_update_descriptor", &BuildCholeskyUpdateDescriptor);
}

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
