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
#include <Python.h>

#include <string>
#include <utility>

#include "nanobind/nanobind.h"
#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/py_client_gpu.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"

namespace nb = nanobind;

namespace xla {
namespace {
Status RegisterCustomCallTarget(const PJRT_Api* c_api, nb::str fn_name,
                                nb::capsule fn, int api_version,
                                XLA_FFI_Handler_Traits traits) {
  if (c_api->extension_start == nullptr) {
    return Unimplemented("The plugin does not have extension.");
  }
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(c_api->extension_start);
  while (next != nullptr &&
         next->type !=
             PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call) {
    next = next->next;
  }
  if (next == nullptr) {
    return Unimplemented("The plugin does not have a custom call extension.");
  }

  if (traits != 0) {
    return Unimplemented("The plugin does not support custom call traits.");
  }

  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = fn_name.c_str();
  args.function_name_size = nb::len(fn_name);
#if PJRT_API_GPU_EXTENSION_VERSION >= 1
  args.api_version = api_version;
#endif
  args.custom_call_function = static_cast<void*>(fn.data());
  RETURN_STATUS_IF_PJRT_ERROR(
      reinterpret_cast<const PJRT_Gpu_Custom_Call*>(next)->custom_call(&args),
      c_api);
  return OkStatus();
}

nb::dict Registrations() {
  nb::dict dict;
  dict["xla_python_gpu_callback"] =
      jax::EncapsulateFunction(xla::XlaPythonGpuCallback);
  return dict;
}

std::string ToString(hipError_t result) {
#define OSTREAM_ROCM_ERROR(__name) \
  case hipError##__name:           \
    return "HIP_ERROR_" #__name;

  switch (result) {
    OSTREAM_ROCM_ERROR(InvalidValue)
    OSTREAM_ROCM_ERROR(OutOfMemory)
    OSTREAM_ROCM_ERROR(NotInitialized)
    OSTREAM_ROCM_ERROR(Deinitialized)
    OSTREAM_ROCM_ERROR(NoDevice)
    OSTREAM_ROCM_ERROR(InvalidDevice)
    OSTREAM_ROCM_ERROR(InvalidImage)
    OSTREAM_ROCM_ERROR(InvalidContext)
    OSTREAM_ROCM_ERROR(InvalidHandle)
    OSTREAM_ROCM_ERROR(NotFound)
    OSTREAM_ROCM_ERROR(NotReady)
    OSTREAM_ROCM_ERROR(NoBinaryForGpu)

    // Encountered an uncorrectable ECC error during execution.
    OSTREAM_ROCM_ERROR(ECCNotCorrectable)

    // Load/store on an invalid address. Must reboot all context.
    case 700:
      return "ROCM_ERROR_ILLEGAL_ADDRESS";
    // Passed too many / wrong arguments, too many threads for register count.
    case 701:
      return "ROCM_ERROR_LAUNCH_OUT_OF_RESOURCES";

      OSTREAM_ROCM_ERROR(ContextAlreadyInUse)
      OSTREAM_ROCM_ERROR(PeerAccessUnsupported)
      OSTREAM_ROCM_ERROR(Unknown)  // Unknown internal error to ROCM.
    default:
      return absl::StrCat("hipError_t(", static_cast<int>(result), ")");
  }
}
}  // namespace

NB_MODULE(rocm_plugin_extension, m) {
  tsl::ImportNumpy();
  m.def(
      "register_custom_call_target",
      [](nb::capsule c_api, nb::str fn_name, nb::capsule fn,
         nb::str xla_platform_name, int api_version,
         XLA_FFI_Handler_Traits traits) {
        xla::ThrowIfError(RegisterCustomCallTarget(
            static_cast<const PJRT_Api*>(c_api.data()), fn_name, std::move(fn),
            api_version, traits));
      },
      nb::arg("c_api"), nb::arg("fn_name"), nb::arg("fn"),
      nb::arg("xla_platform_name"), nb::arg("api_version") = 0,
      nb::arg("traits") = 0);
  m.def("registrations", &Registrations);
  m.def(
      "get_device_ordinal",
      [](std::intptr_t data_value) {
        if (data_value == 0) {
          return 0;
        }
        int device_ordinal;
        void* data_ptr = reinterpret_cast<void*>(data_value);
        hipError_t result =
            hipPointerGetAttribute(static_cast<void*>(&device_ordinal),
                                  HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                  reinterpret_cast<hipDeviceptr_t>(data_ptr));
        if (result != hipSuccess) {
          LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << data_ptr
                    << ". Error: " << ToString(result);
        }
        return device_ordinal;
      },
      nb::arg("data_value"));
}
}  // namespace xla
