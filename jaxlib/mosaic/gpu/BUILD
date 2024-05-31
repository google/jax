# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "pybind_extension")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

py_library(
    name = "mosaic_gpu",
    data = [":libmosaic_gpu_runtime.so"],
    deps = [":_mosaic_gpu_ext"],
)

cc_library(
    name = "passes",
    srcs = ["launch_lowering.cc"],
    hdrs = ["launch_lowering.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

CAPI_SOURCES = [
    "integrations/c/passes.cc",
]

CAPI_HEADERS = [
    "integrations/c/passes.h",
]

cc_library(
    name = "mlir_capi",
    srcs = CAPI_SOURCES,
    hdrs = CAPI_HEADERS,
    deps = [
        ":passes",
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "mlir_capi_headers",
    hdrs = CAPI_HEADERS,
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "mlir_capi_objects",
    srcs = CAPI_SOURCES,
    hdrs = CAPI_HEADERS,
    deps = [
        ":passes",
        "@llvm-project//mlir:CAPIIRObjects",
    ],
    alwayslink = True,
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    deps = [
        ":passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToLLVMIRTranslation",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVGPUDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMTarget",
        "@llvm-project//mlir:NVVMToLLVM",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorDialect",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/service:custom_call_target_registry",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,
)

pybind_extension(
    name = "_mosaic_gpu_ext",
    srcs = ["mosaic_gpu_ext.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../../nvidia/cuda_runtime/lib",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//jaxlib:kernel_nanobind_helpers",
        "//jaxlib/cuda:cuda_vendor",
        "@llvm-project//llvm:Support",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/tsl/cuda:cudart",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
    ],
)

cc_binary(
    name = "libmosaic_gpu_runtime.so",
    srcs = ["runtime.cc"],
    copts = ["-fvisibility=default"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../../nvidia/cuda_runtime/lib",
        ],
        "//conditions:default": [],
    }),
    linkshared = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "@xla//xla/tsl/cuda:cudart",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)
