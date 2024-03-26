# Copyright 2018 The JAX Authors.
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
    "if_cuda_is_configured",
    "pybind_extension",
)
load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

cc_library(
    name = "cuda_vendor",
    hdrs = [
        "//jaxlib/gpu:vendor.h",
    ],
    defines = ["JAX_GPU_CUDA=1"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
    ],
)

cc_library(
    name = "cuda_gpu_kernel_helpers",
    srcs = [
        "//jaxlib/gpu:gpu_kernel_helpers.cc",
    ],
    hdrs = [
        "//jaxlib/gpu:gpu_kernel_helpers.h",
    ],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":cuda_vendor",
        "@xla//xla/tsl/cuda:cupti",
        "@xla//xla/tsl/cuda:cusolver",
        "@xla//xla/tsl/cuda:cusparse",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
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
    srcs = ["//jaxlib/gpu:blas_kernels.cc"],
    hdrs = ["//jaxlib/gpu:blas_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/tsl/cuda:cublas",
        "@xla//xla/tsl/cuda:cudart",
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
    name = "_blas",
    srcs = ["//jaxlib/gpu:blas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cublas/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_blas",
    deps = [
        ":cublas_kernels",
        ":cuda_vendor",
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//xla/tsl/cuda:cublas",
        "@xla//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@nanobind",
    ],
)

cc_library(
    name = "cudnn_rnn_kernels",
    srcs = ["//jaxlib/gpu:rnn_kernels.cc"],
    hdrs = ["//jaxlib/gpu:rnn_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/cuda:cudnn",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_rnn",
    srcs = ["//jaxlib/gpu:rnn.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_rnn",
    deps = [
        ":cuda_vendor",
        ":cudnn_rnn_kernels",
        "//jaxlib:absl_status_casters",
        "//jaxlib:kernel_nanobind_helpers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@nanobind",
    ],
)

cc_library(
    name = "cusolver_kernels",
    srcs = ["//jaxlib/gpu:solver_kernels.cc"],
    hdrs = ["//jaxlib/gpu:solver_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/tsl/cuda:cusolver",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_solver",
    srcs = ["//jaxlib/gpu:solver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cusolver/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_solver",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":cusolver_kernels",
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/cuda:cusolver",
        "@xla//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_cuda//cuda:cuda_headers",
        "@nanobind",
    ],
)

cc_library(
    name = "cusparse_kernels",
    srcs = ["//jaxlib/gpu:sparse_kernels.cc"],
    hdrs = ["//jaxlib/gpu:sparse_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:handle_pool",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/cuda:cusparse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_sparse",
    srcs = ["//jaxlib/gpu:sparse.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cusparse/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_sparse",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":cusparse_kernels",
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/cuda:cusparse",
        "@xla//xla/tsl/python/lib/core:numpy",
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
        "@nanobind",
    ],
)

cc_library(
    name = "cuda_lu_pivot_kernels",
    srcs = [
        "//jaxlib/gpu:lu_pivot_kernels.cc",
    ],
    hdrs = ["//jaxlib/gpu:lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels_impl",
        ":cuda_vendor",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "cuda_lu_pivot_kernels_impl",
    srcs = [
        "//jaxlib/gpu:lu_pivot_kernels.cu.cc",
    ],
    hdrs = ["//jaxlib/gpu:lu_pivot_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_linalg",
    srcs = ["//jaxlib/gpu:linalg.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_linalg",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_lu_pivot_kernels",
        ":cuda_lu_pivot_kernels_impl",
        ":cuda_vendor",
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//xla/tsl/cuda:cudart",
        "@local_config_cuda//cuda:cuda_headers",
        "@nanobind",
    ],
)

cc_library(
    name = "cuda_prng_kernels",
    srcs = [
        "//jaxlib/gpu:prng_kernels.cc",
    ],
    hdrs = ["//jaxlib/gpu:prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels_impl",
        ":cuda_vendor",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "cuda_prng_kernels_impl",
    srcs = [
        "//jaxlib/gpu:prng_kernels.cu.cc",
    ],
    hdrs = ["//jaxlib/gpu:prng_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:kernel_helpers",
        "@xla//xla/service:custom_call_status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "_prng",
    srcs = ["//jaxlib/gpu:prng.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_prng",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_prng_kernels",
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//xla/tsl/cuda:cudart",
        "@local_config_cuda//cuda:cuda_headers",
        "@nanobind",
    ],
)

cc_library(
    name = "cuda_gpu_kernels",
    srcs = ["//jaxlib/gpu:gpu_kernels.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":cublas_kernels",
        ":cuda_lu_pivot_kernels",
        ":cuda_prng_kernels",
        ":cuda_vendor",
        ":cusolver_kernels",
        ":cusparse_kernels",
        ":triton_kernels",
        "@xla//xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)

cc_library(
    name = "triton_kernels",
    srcs = ["//jaxlib/gpu:triton_kernels.cc"],
    hdrs = ["//jaxlib/gpu:triton_kernels.h"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":triton_utils",
        "//jaxlib/gpu:triton_cc_proto",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/stream_executor/gpu:asm_compiler",
        "@xla//xla/tsl/cuda:cudart",
        "@tsl//tsl/platform:env",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "triton_utils",
    srcs = ["//jaxlib/gpu:triton_utils.cc"],
    hdrs = ["//jaxlib/gpu:triton_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib/gpu:triton_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@zlib",
    ],
)

pybind_extension(
    name = "_triton",
    srcs = ["//jaxlib/gpu:triton.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_triton",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":triton_kernels",
        ":triton_utils",
        "//jaxlib:absl_status_casters",
        "//jaxlib:kernel_nanobind_helpers",
        "//jaxlib/gpu:triton_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
)

cc_library(
    name = "versions_helpers",
    srcs = ["versions_helpers.cc"],
    hdrs = ["versions_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        "//jaxlib:absl_status_casters",
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//xla/tsl/cuda:cublas",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/cuda:cudnn",
        "@xla//xla/tsl/cuda:cufft",
        "@xla//xla/tsl/cuda:cupti",
        "@xla//xla/tsl/cuda:cusolver",
        "@xla//xla/tsl/cuda:cusparse",
    ],
)

pybind_extension(
    name = "_versions",
    srcs = ["versions.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    linkopts = select({
        "@xla//xla/python:use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_cupti/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cublas/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cufft/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cudnn/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cusolver/lib",
            "-Wl,-rpath,$$ORIGIN/../../nvidia/cusparse/lib",
        ],
        "//conditions:default": [],
    }),
    module_name = "_versions",
    deps = [
        ":cuda_gpu_kernel_helpers",
        ":cuda_vendor",
        ":versions_helpers",
        "//jaxlib:absl_status_casters",
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//xla/tsl/cuda:cublas",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/cuda:cudnn",
        "@xla//xla/tsl/cuda:cufft",
        "@xla//xla/tsl/cuda:cupti",
        "@xla//xla/tsl/cuda:cusolver",
        "@xla//xla/tsl/cuda:cusparse",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
)

py_library(
    name = "cuda_gpu_support",
    deps = [
        ":_blas",
        ":_linalg",
        ":_prng",
        ":_rnn",
        ":_solver",
        ":_sparse",
        ":_triton",
        ":_versions",
    ],
)

# We cannot nest select and if_cuda_is_configured so we introduce
# a standalone py_library target.
py_library(
    name = "gpu_only_test_deps",
    # `if_cuda_is_configured` will default to `[]`.
    deps = if_cuda_is_configured([
        ":cuda_gpu_support",
        "//jaxlib:cuda_plugin_extension",
    ]),
)
