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

#include "nanobind/nanobind.h"
#include "tests/ffi/ffi_cpu_kernels.h"

namespace jax {
namespace nb = nanobind;

template <typename T>
nanobind::capsule EncapsulateFfiHandler(T* fn) {
  return nanobind::capsule(reinterpret_cast<void*>(fn));
}

nb::dict Registrations() {
  nb::dict dict;
  dict["add_to"] = EncapsulateFfiHandler(tests::AddTo);
  dict["should_fail"] = EncapsulateFfiHandler(tests::ShouldFail);
  return dict;
}

NB_MODULE(ffi_cpu, m) { m.def("registrations", &Registrations); }
}  // namespace jax
