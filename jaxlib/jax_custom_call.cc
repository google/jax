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

#include "jax_custom_call.h"

#include <map>
#include <optional>
#include <iostream>
#include <pybind11/pybind11.h>
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"

struct JaxFFIStatus {
  std::optional<std::string> message;
};

int JaxFFIVersion1 = 0;

void JaxFFIStatusSetSuccessFn(JaxFFIStatus* status) {
  status->message = std::nullopt;
}

void JaxFFIStatusSetFailureFn(JaxFFIStatus* status, const char* message) {
  status->message = std::string(message);
}

struct JaxFFIStruct {
  int version = JaxFFIVersion1;
  void* fn_table[2] = {
    (void *)&JaxFFIStatusSetSuccessFn,
    (void *)&JaxFFIStatusSetFailureFn
  };
};

JaxFFIStruct jax_ffi_struct;

namespace py = pybind11;

namespace jax {

std::map<std::string, void*>& get_registry() {
  static auto registry = new std::map<std::string, void*>();
  return *registry;
};

std::optional<std::string> JaxFFIStatusGetMessage(JaxFFIStatus* status) {
  return status->message;
}

struct Descriptor {
  void* function_ptr;
  std::string user_descriptor;
};

extern "C" void JaxFFICallWrapper(void* output, void** inputs,
                                  XlaCustomCallStatus* status) {
  auto descriptor = reinterpret_cast<Descriptor*>(*static_cast<uintptr_t*>(inputs[0]));
  inputs += 1;
  JaxFFIStatus jax_ffi_status;
  JaxFFIApi jax_ffi_api;
  jax_ffi_api.version = jax_ffi_struct.version;
  jax_ffi_api.fn_table = jax_ffi_struct.fn_table;
  auto function_ptr = reinterpret_cast<void(*)(JaxFFIApi*, JaxFFIStatus*, void*, size_t, void**, void**)>(descriptor->function_ptr);
  function_ptr(reinterpret_cast<JaxFFIApi*>(&jax_ffi_api), &jax_ffi_status,
               descriptor->user_descriptor.data(), descriptor->user_descriptor.size(),
               inputs, reinterpret_cast<void**>(output));
  if (jax_ffi_status.message) {
    XlaCustomCallStatusSetFailure(status, jax_ffi_status.message->data(),
                                  jax_ffi_status.message->size());
  }
}

PYBIND11_MODULE(_jax_custom_call, m) {
  m.def("get_jax_ffi_call_wrapper", []() {
    return py::capsule(reinterpret_cast<void*>(&JaxFFICallWrapper),
                       "xla._CUSTOM_CALL_TARGET");
  });
  m.def("make_descriptor", [](void* function_ptr, const std::string& user_descriptor_bytes) {
    Descriptor* descriptor = new Descriptor;
    descriptor->function_ptr = function_ptr;
    // Make a copy of user descriptor bytes. Possibly avoidable.
    descriptor->user_descriptor = user_descriptor_bytes;
    uint64_t ptr = reinterpret_cast<uint64_t>(descriptor);
    py::capsule keepalive = py::capsule(descriptor, [](void* ptr) {
        delete reinterpret_cast<Descriptor*>(ptr);
    });
    return std::make_pair(ptr, keepalive);
  });
}

}  // jax
