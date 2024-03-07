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

#ifndef JAXLIB_KERNEL_NANOBIND_HELPERS_H_
#define JAXLIB_KERNEL_NANOBIND_HELPERS_H_

#include <string>

#include "nanobind/nanobind.h"
#include "absl/base/casts.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/tsl/python/lib/core/numpy.h"  // NOLINT

namespace jax {

// Caution: to use this type you must call tsl::ImportNumpy() in your module
// initialization function. Otherwise PyArray_DescrCheck will be nullptr.
class dtype : public nanobind::object {
 public:
  NB_OBJECT_DEFAULT(dtype, object, "dtype", PyArray_DescrCheck);  // NOLINT

  int itemsize() const { return nanobind::cast<int>(attr("itemsize")); }

  /// Single-character code for dtype's kind.
  /// For example, floating point types are 'f' and integral types are 'i'.
  char kind() const { return nanobind::cast<char>(attr("kind")); }
};

// Descriptor objects are opaque host-side objects used to pass data from JAX
// to the custom kernel launched by XLA. Currently simply treat host-side
// structures as byte-strings; this is not portable across architectures. If
// portability is needed, we could switch to using a representation such as
// protocol buffers or flatbuffers.

// Packs a descriptor object into a nanobind::bytes structure.
// UnpackDescriptor() is available in kernel_helpers.h.
template <typename T>
nanobind::bytes PackDescriptor(const T& descriptor) {
  std::string s = PackDescriptorAsString(descriptor);
  return nanobind::bytes(s.data(), s.size());
}

template <typename T>
nanobind::capsule EncapsulateFunction(T* fn) {
  return nanobind::capsule(absl::bit_cast<void*>(fn),
                           "xla._CUSTOM_CALL_TARGET");
}

}  // namespace jax

#endif  // JAXLIB_KERNEL_NANOBIND_HELPERS_H_
