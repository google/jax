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
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

ffi::Error CounterImpl(int64_t index,
                       ffi::Result<ffi::BufferR0<ffi::S32>> out) {
  static std::mutex mutex;
  static auto& cache = *new std::unordered_map<int64_t, int32_t>();
  {
    const std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(index);
    if (it != cache.end()) {
      out->typed_data()[0] = ++it->second;
    } else {
      cache.insert({index, 0});
      out->typed_data()[0] = 0;
    }
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Counter, CounterImpl,
    ffi::Ffi::Bind().Attr<int64_t>("index").Ret<ffi::BufferR0<ffi::S32>>());

NB_MODULE(_counter, m) {
  m.def("registrations", []() {
    nb::dict registrations;
    registrations["counter"] = nb::capsule(reinterpret_cast<void*>(Counter));
    return registrations;
  });
}
