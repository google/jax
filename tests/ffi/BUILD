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

load(
    "//jaxlib:jax.bzl",
    "jax_generate_backend_suites",
    "jax_test",
    "py_deps",
    "pybind_extension",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:private"],
)

jax_generate_backend_suites()

cc_library(
    name = "ffi_cpu_kernels",
    srcs = ["ffi_cpu_kernels.cc"],
    hdrs = ["ffi_cpu_kernels.h"],
    deps = [
        "@xla//xla/ffi/api:ffi",
    ],
)

pybind_extension(
    name = "ffi_cpu",
    srcs = ["ffi_cpu.cc"],
    module_name = "ffi_cpu",
    deps = [
        ":ffi_cpu_kernels",
        "@xla//xla/ffi/api:ffi",
        "@nanobind",
    ],
)

jax_test(
    name = "ffi_test",
    srcs = ["ffi_test.py"],
    deps = [
        ":ffi_cpu",
        "//jax:extend",
    ] + py_deps("numpy"),
)
