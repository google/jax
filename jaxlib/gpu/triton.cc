#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"
#include "absl/status/statusor.h"
#include "jaxlib/absl_status_casters.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/triton.pb.h"
#include "jaxlib/gpu/triton_kernels.h"
#include "jaxlib/gpu/triton_utils.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"

#define GPU_RETURN_IF_ERROR(expr) JAX_RETURN_IF_ERROR(JAX_AS_STATUS(expr))

namespace nb = nanobind;

namespace jax::JAX_GPU_NAMESPACE {

NB_MODULE(_triton, m) {
  nb::class_<Kernel>(m, "TritonKernel")
      .def(nb::init<std::string, uint32_t, uint32_t, std::string, std::string,
                    int, uint32_t, uint32_t, uint32_t>());

  nb::class_<KernelCall::Parameter>(m, "TritonParameter");

  m.def("create_array_parameter",
        [](size_t bytes_to_zero, size_t ptr_divisibility) {
          return KernelCall::Parameter{
              KernelCall::Parameter::Array{bytes_to_zero, ptr_divisibility}};
        });

  m.def("create_scalar_parameter",
        ValueOrThrowWrapper([](bool value, std::string_view dtype)
                                -> absl::StatusOr<KernelCall::Parameter> {
          if ((dtype == "i1") || (dtype == "B")) {
            return KernelCall::Parameter{value};
          } else {
            return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                              dtype.data());
          }
        }));

  m.def("create_scalar_parameter",
        ValueOrThrowWrapper([](nb::int_ value, std::string_view dtype)
                                -> absl::StatusOr<KernelCall::Parameter> {
          if (dtype == "i32") {
            return KernelCall::Parameter{static_cast<int32_t>(value)};
          } else if (dtype == "u32") {
            return KernelCall::Parameter{static_cast<uint32_t>(value)};
          } else if (dtype == "i64") {
            return KernelCall::Parameter{static_cast<int64_t>(value)};
          } else if (dtype == "u64") {
            return KernelCall::Parameter{static_cast<uint64_t>(value)};
          } else {
            return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                              dtype.data());
          }
        }));

  m.def("create_scalar_parameter",
        ValueOrThrowWrapper([](double value, std::string_view dtype)
                                -> absl::StatusOr<KernelCall::Parameter> {
          if (dtype == "fp32") {
            return KernelCall::Parameter{static_cast<float>(value)};
          } else if (dtype == "fp64") {
            return KernelCall::Parameter{static_cast<double>(value)};
          } else {
            return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                              dtype.data());
          }
        }));

  nb::class_<KernelCall>(m, "TritonKernelCall")
      .def(nb::init<Kernel, uint32_t, uint32_t, uint32_t,
                    std::vector<KernelCall::Parameter>>())
      .def("to_proto", [](const KernelCall& kernel_call, std::string name,
                          nb::bytes metadata) {
        jax_triton::TritonAnyKernelCall proto;
        *proto.mutable_kernel_call() = kernel_call.ToProto();
        proto.set_name(std::move(name));
        proto.set_metadata(metadata.c_str(), metadata.size());
        std::string s = proto.SerializeAsString();
        return nb::bytes(s.c_str(), s.size());
      });

  nb::class_<AutotunedKernelCall>(m, "TritonAutotunedKernelCall")
      .def("__init__",
           [](AutotunedKernelCall* call, std::string name,
              std::vector<std::pair<KernelCall, std::string>>
                  calls_and_descriptions,
              std::vector<std::tuple<size_t, size_t, size_t>>
                  input_output_aliases) {
             std::vector<AutotunedKernelCall::Config> configs;
             configs.reserve(calls_and_descriptions.size());
             for (auto& [kernel_call, desc] : calls_and_descriptions) {
               configs.push_back({std::move(kernel_call), std::move(desc)});
             }
             new (call) AutotunedKernelCall(std::move(name), std::move(configs),
                                            std::move(input_output_aliases));
           })
      .def("to_proto", [](const AutotunedKernelCall& kernel_call,
                          std::string name, nb::bytes metadata) {
        jax_triton::TritonAnyKernelCall proto;
        *proto.mutable_autotuned_kernel_call() = kernel_call.ToProto();
        proto.set_name(std::move(name));
        proto.set_metadata(metadata.c_str(), metadata.size());
        std::string s = proto.SerializeAsString();
        return nb::bytes(s.c_str(), s.size());
      });

  m.def("get_custom_call",
        [] { return EncapsulateFunction(&TritonKernelCall); });

  m.def("get_ffi_call",
        [] { return EncapsulateFunction(&TritonKernelCallFfi); });

  m.def("get_compute_capability",
        ValueOrThrowWrapper([](int device) -> absl::StatusOr<int> {
          int major, minor;
          GPU_RETURN_IF_ERROR(gpuInit(device));
          GPU_RETURN_IF_ERROR(gpuDeviceGetAttribute(
              &major, GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
          GPU_RETURN_IF_ERROR(gpuDeviceGetAttribute(
              &minor, GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
          return major * 10 + minor;
        }));

  m.def(
      "get_arch_details",
      ValueOrThrowWrapper([](int device) -> absl::StatusOr<absl::string_view> {
#ifdef JAX_GPU_HIP
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        return prop.gcnArchName;
#else
        return absl::UnimplementedError("Not a HIP GPU");
#endif
      }));

  m.def("get_serialized_metadata",
        ValueOrThrowWrapper([](nb::bytes opaque) -> absl::StatusOr<nb::bytes> {
          JAX_ASSIGN_OR_RETURN(
              std::string metadata,
              GetTritonKernelCallSerializedMetadata(
                  absl::string_view(opaque.c_str(), opaque.size())));
          return nb::bytes(metadata.c_str(), metadata.size());
        }));
}

}  // namespace jax::JAX_GPU_NAMESPACE
