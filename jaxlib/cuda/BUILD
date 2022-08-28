# Copyright 2018 Google LLC
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

# NVIDIA CUDA kernels

load(
    "//jaxlib:jax.bzl",
    "cuda_library",
    "pybind_extension",
)

licenses(["notice"])

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "cuda_gpu_kernel_helpers",
    srcs = ["cuda_gpu_kernel_helpers.cc"],
    hdrs = ["cuda_gpu_kernel_helpers.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cublas_kernels",
    srcs = ["cublas_kernels.cc"],
    hdrs = ["cublas_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cublas_lib",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cublas",
    srcs = ["cublas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cublas",
    deps = [
        ":cublas_kernels",
        "//jaxlib:kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cublas_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cusolver_kernels",
    srcs = ["cusolver_kernels.cc"],
    hdrs = ["cusolver_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cusolver",
    srcs = ["cusolver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cusolver",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cusolver_kernels",
        "//jaxlib:kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusolver_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cusparse_kernels",
    srcs = ["cusparse_kernels.cc"],
    hdrs = ["cusparse_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cusparse",
    srcs = ["cusparse.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cusparse",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cusparse_kernels",
        "//jaxlib:kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cusparse_lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_lu_pivot_kernels",
    srcs = [
        "cuda_lu_pivot_kernels.cc",
    ],
    hdrs = ["cuda_lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels_impl",
        "//jaxlib:kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "cuda_lu_pivot_kernels_impl",
    srcs = [
        "cuda_lu_pivot_kernels.cu.cc",
    ],
    hdrs = ["cuda_lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        "//jaxlib:kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cuda_linalg",
    srcs = ["cuda_linalg.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cuda_linalg",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels",
        ":cuda_lu_pivot_kernels_impl",
        "//jaxlib:kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_prng_kernels",
    srcs = [
        "cuda_prng_kernels.cc",
    ],
    hdrs = ["cuda_prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels_impl",
        "//jaxlib:kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "cuda_prng_kernels_impl",
    srcs = [
        "cuda_prng_kernels.cu.cc",
    ],
    hdrs = ["cuda_prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        "//jaxlib:kernel_helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_cuda_prng",
    srcs = ["cuda_prng.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cuda_prng",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels",
        "//jaxlib:kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cuda_gpu_kernels",
    srcs = ["cuda_gpu_kernels.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":cublas_kernels",
        ":cuda_lu_pivot_kernels",
        ":cuda_prng_kernels",
        ":cusolver_kernels",
        ":cusparse_kernels",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)

py_library(
    name = "cuda_gpu_support",
    deps = [
        ":_cublas",
        ":_cuda_linalg",
        ":_cuda_prng",
        ":_cusolver",
        ":_cusparse",
    ],
)
