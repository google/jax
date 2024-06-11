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
#include "absl/status/status.h"
#include "xla/ffi/api/c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/status_casters.h"

namespace {

namespace nb = nanobind;

absl::Status RegisterHandler(const PJRT_Api* c_api, nb::str target_name,
                             nb::capsule handler, nb::str platform_name,
                             int api_version, XLA_FFI_Handler_Traits traits) {
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(c_api->extension_start);
  while (next != nullptr &&
         next->type != PJRT_Extension_Type::PJRT_Extension_Type_FFI) {
    next = next->next;
  }
  if (next == nullptr) {
    return absl::UnimplementedError("FFI extension is not available.");
  }

  if (traits != 0) {
    return absl::UnimplementedError(
        "FFI handler traits are not currently supported.");
  }

  PJRT_FFI_Register_Handler_Args args;
  args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
  args.target_name = target_name.c_str();
  args.target_name_size = nb::len(target_name);
  args.api_version = api_version;
  args.handler = static_cast<void*>(handler.data());
  args.platform_name = platform_name.c_str();
  args.platform_name_size = nb::len(platform_name);
  RETURN_STATUS_IF_PJRT_ERROR(
      reinterpret_cast<const PJRT_FFI_Extension*>(next)->register_handler(
          &args),
      c_api);
  return absl::OkStatus();
}

}  // namespace

NB_MODULE(ffi, m) {
  m.def(
      "register_handler",
      [](nb::capsule c_api, nb::str target_name, nb::capsule handler,
         nb::str xla_platform_name, int api_version,
         XLA_FFI_Handler_Traits traits) {
        xla::ThrowIfError(RegisterHandler(
            static_cast<const PJRT_Api*>(c_api.data()), target_name,
            std::move(handler), xla_platform_name, api_version, traits));
      },
      nb::arg("c_api"), nb::arg("target_name"), nb::arg("handler"),
      nb::arg("xla_platform_name"), nb::arg("api_version") = 0,
      nb::arg("traits") = 0);
}
