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
    "py_library_providing_imports_info",
    "pybind_extension",
    "pytype_library",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

# This makes xla_extension module accessible from jax._src.lib.
genrule(
    name = "xla_extension_py",
    outs = ["xla_extension.py"],
    cmd = "echo 'from xla.xla.python.xla_extension import *\n' > $@",
)

py_library_providing_imports_info(
    name = "jaxlib",
    srcs = [
        "gpu_common_utils.py",
        "gpu_linalg.py",
        "gpu_prng.py",
        "gpu_rnn.py",
        "gpu_solver.py",
        "gpu_sparse.py",
        "gpu_triton.py",
        "hlo_helpers.py",
        "init.py",
        "lapack.py",
        ":version",
        ":xla_client",
        ":xla_extension_py",
    ],
    data = [":ffi_headers"],
    lib_rule = pytype_library,
    deps = [
        ":cpu_feature_guard",
        ":utils",
        "//jaxlib/cpu:_lapack",
        "//jaxlib/mlir",
        "//jaxlib/mlir:arithmetic_dialect",
        "//jaxlib/mlir:builtin_dialect",
        "//jaxlib/mlir:chlo_dialect",
        "//jaxlib/mlir:func_dialect",
        "//jaxlib/mlir:gpu_dialect",
        "//jaxlib/mlir:ir",
        "//jaxlib/mlir:llvm_dialect",
        "//jaxlib/mlir:math_dialect",
        "//jaxlib/mlir:memref_dialect",
        "//jaxlib/mlir:mhlo_dialect",
        "//jaxlib/mlir:nvgpu_dialect",
        "//jaxlib/mlir:nvvm_dialect",
        "//jaxlib/mlir:pass_manager",
        "//jaxlib/mlir:scf_dialect",
        "//jaxlib/mlir:sparse_tensor_dialect",
        "//jaxlib/mlir:stablehlo_dialect",
        "//jaxlib/mlir:vector_dialect",
        "//jaxlib/mosaic",
        "//jaxlib/triton",
        "@xla//xla/python:xla_extension",
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
    srcs = ["@xla//xla/python:xla_client"],
    dst = ".",
    flatten = True,
)

symlink_files(
    name = "ffi_headers",
    srcs = ["@xla//xla/ffi/api:all_headers"],
    dst = "include/xla/ffi/api",
    flatten = True,
)

exports_files([
    "README.md",
    "setup.py",
])

cc_library(
    name = "absl_status_casters",
    hdrs = ["absl_status_casters.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "kernel_nanobind_helpers",
    hdrs = ["kernel_nanobind_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kernel_helpers",
        "@xla//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/base",
        "@nanobind",
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
        "@xla//third_party/python_runtime:headers",
    ],
)

pybind_extension(
    name = "utils",
    srcs = ["utils.cc"],
    module_name = "utils",
    deps = [
        "@xla//third_party/python_runtime:headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:inlined_vector",
        "@nanobind",
    ],
)

pybind_extension(
    name = "cuda_plugin_extension",
    srcs = ["cuda_plugin_extension.cc"],
    module_name = "cuda_plugin_extension",
    deps = [
        "@com_google_absl//absl/status",
        "@nanobind",
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//third_party/python_runtime:headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@xla//xla:util",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        # TODO(jieying): move to jaxlib after py_client_gpu is separated from py_client
        "@xla//xla/python:py_client_gpu",
        "@xla//xla/tsl/cuda:cublas",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

pybind_extension(
    name = "rocm_plugin_extension",
    srcs = ["rocm_plugin_extension.cc"],
    module_name = "rocm_plugin_extension",
    deps = [
        "//jaxlib:kernel_nanobind_helpers",
        "@xla//third_party/python_runtime:headers",
        "@xla//xla:status",
        "@xla//xla:util",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@xla//xla/python:py_client_gpu",
        "@xla//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/status",
        "@local_config_rocm//rocm:rocm_headers",
        "@nanobind",
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
