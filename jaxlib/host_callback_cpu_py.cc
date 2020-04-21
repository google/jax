/* Copyright 2020 Google LLC

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

// This exposes the host_callback_py Python module with the implementation
// of the CustomCalls for CPU and GPU.

#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/pytypes.h"
#include "jaxlib/host_callback.h"
#include "jaxlib/kernel_pybind11_helpers.h"

namespace jax {

namespace py = pybind11;

namespace {

// Returns a dictionary with CustomCall functions to register.
py::dict CustomCallRegistrations() {
  py::dict dict;
  dict["jax_print"] = EncapsulateFunction(PrintCPU);
  return dict;
}

PYBIND11_MODULE(host_callback_cpu_py, m) {
  m.doc() = "Python bindings for the host_callback runtime for CPU.";
  m.def("customcall_registrations", &CustomCallRegistrations);
  m.def("get_print_metadata_version", &GetPrintMetadataVersion);
}

}  // namespace
}  // namespace jax
