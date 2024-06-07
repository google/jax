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

load(
    "//jaxlib:jax.bzl",
    "pybind_extension",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

# LAPACK

cc_library(
    name = "lapack_kernels",
    srcs = ["lapack_kernels.cc"],
    hdrs = ["lapack_kernels.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/service:custom_call_status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "lapack_kernels_using_lapack",
    srcs = ["lapack_kernels_using_lapack.cc"],
    deps = [":lapack_kernels"],
    alwayslink = 1,
)

pybind_extension(
    name = "_lapack",
    srcs = ["lapack.cc"],
    hdrs = ["lapack.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    enable_stub_generation = True,
    features = ["-use_header_modules"],
    module_name = "_lapack",
    pytype_srcs = [
        "_lapack.pyi",
    ],
    deps = [
        ":lapack_kernels",
        "//jaxlib:kernel_nanobind_helpers",
        "@nanobind",
    ],
)

cc_library(
    name = "cpu_kernels",
    srcs = ["cpu_kernels.cc"],
    hdrs = ["lapack.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":lapack_kernels",
        ":lapack_kernels_using_lapack",
        "@xla//xla/service:custom_call_target_registry",
    ],
    alwayslink = 1,
)
