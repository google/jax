# Copyright 2019 The JAX Authors.
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
load(
    "//jaxlib:jax.bzl",
    "py_deps",
    "pytype_strict_library",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "jax_to_ir",
    srcs = ["jax_to_ir.py"],
    tags = [
        "ignore_for_dep=third_party.py.jax.experimental.jax2tf",
        "ignore_for_dep=third_party.py.tensorflow",
    ],
    deps = [
        "//jax",
    ],
)

py_library(
    name = "jax_to_ir_with_tensorflow",
    srcs = ["jax_to_ir.py"],
    deps = [
        "//jax",
        "//jax/experimental/jax2tf",
    ] + py_deps("tensorflow_core"),
)

pytype_strict_library(
    name = "build_utils",
    srcs = ["build_utils.py"],
)
