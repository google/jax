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

# JAX is Autograd and XLA

load("//jaxlib:symlink_files.bzl", "symlink_files")
load(
    "//jaxlib:jax.bzl",
    "if_windows",
    "pybind_extension",
)

licenses(["notice"])

package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "jaxlib",
    srcs = [
        "ducc_fft.py",
        "gpu_linalg.py",
        "gpu_prng.py",
        "gpu_solver.py",
        "gpu_sparse.py",
        "init.py",
        "jax_custom_call.py",
        "lapack.py",
        "mhlo_helpers.py",
        ":version",
        ":xla_client",
    ],
    data = [
        "jax_custom_call.h",
        ":xla_extension",
    ],
    deps = [
        ":cpu_feature_guard",
        "//jaxlib/cpu:_ducc_fft",
        "//jaxlib/cpu:_lapack",
        "//jaxlib/mlir",
        "//jaxlib/mlir:builtin_dialect",
        "//jaxlib/mlir:chlo_dialect",
        "//jaxlib/mlir:func_dialect",
        "//jaxlib/mlir:ir",
        "//jaxlib/mlir:mhlo_dialect",
        "//jaxlib/mlir:ml_program_dialect",
        "//jaxlib/mlir:pass_manager",
        "//jaxlib/mlir:sparse_tensor_dialect",
        "//jaxlib/mlir:stablehlo_dialect",
    ],
)

symlink_files(
    name = "version",
    srcs = ["//jax:version.py"],
    dst = ".",
    flatten = True,
)

symlink_files(
    name = "xla_client",
    srcs = ["@org_tensorflow//tensorflow/compiler/xla/python:xla_client"],
    dst = ".",
    flatten = True,
)

symlink_files(
    name = "xla_extension",
    srcs = if_windows(
        ["@org_tensorflow//tensorflow/compiler/xla/python:xla_extension.pyd"],
        ["@org_tensorflow//tensorflow/compiler/xla/python:xla_extension.so"],
    ),
    dst = ".",
    flatten = True,
)

exports_files([
    "README.md",
    "setup.py",
    "setup.cfg",
])

cc_library(
    name = "kernel_pybind11_helpers",
    hdrs = ["kernel_pybind11_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kernel_helpers",
        "@com_google_absl//absl/base",
        "@pybind11",
    ],
)

cc_library(
    name = "kernel_helpers",
    hdrs = ["kernel_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "handle_pool",
    hdrs = ["handle_pool.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

# This isn't a CPU kernel. This exists to catch cases where jaxlib is built for the wrong
# target architecture.
pybind_extension(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.c"],
    module_name = "cpu_feature_guard",
    deps = [
        "@org_tensorflow//third_party/python_runtime:headers",
    ],
)

# CPU kernels

# TODO(phawkins): Remove this forwarding target.
cc_library(
    name = "cpu_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//jaxlib/cpu:cpu_kernels",
    ],
    alwayslink = 1,
)

# TODO(phawkins): Remove this forwarding target.
cc_library(
    name = "gpu_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//jaxlib/cuda:cuda_gpu_kernels",
    ],
    alwayslink = 1,
)

pybind_extension(
    name = "_jax_custom_call",
    srcs = ["jax_custom_call.cc"],
    hdrs = [
        "jax_custom_call.h",
        "jax_custom_call_internal.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_status",
        "@org_tensorflow//tensorflow/compiler/xla/service:custom_call_target_registry",
        "@pybind11",
    ],
)
