/* Copyright 2022 The JAX Authors.

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

#include <iostream>

#include <pybind11/pybind11.h>
#include "jax_custom_call.h" // Header that comes from JAX

namespace py = pybind11;

struct Info {
  float n;
};

extern "C" void add_one(JaxFFIApi* api, JaxFFIStatus* status,
                        void* descriptor, size_t descriptor_size,
                        void** inputs, void** outputs) {
  assert(JaxFFIVersion(api) == 1);
  Info info;
  assert(descriptor_size == sizeof(Info));
  std::memcpy(&info, descriptor, descriptor_size);
  if (info.n < 0) {
    JaxFFIStatusSetFailure(api, status, "Info must be >= 0");
  }
  float* input = (float*) inputs[0];
  float* output = (float*) outputs[0];
  *output = *input + info.n;
}

PYBIND11_MODULE(add_one_lib, m) {
  m.def("get_function", []() {
    return py::capsule(reinterpret_cast<void*>(add_one), JaxFFICallCpu);
  });
  m.def("get_descriptor", [](float n) {
    Info info;
    info.n = n;
    // Serialize user-provided static info as a byte string.
    return py::bytes(reinterpret_cast<const char*>(&info), sizeof(Info));
  });
}
