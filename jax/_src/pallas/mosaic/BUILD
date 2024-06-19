# Copyright 2023 The JAX Authors.
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

# Package for Mosaic-specific Pallas extensions

load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "py_deps")

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//:__subpackages__",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    deps = [
        "//jax",
        "//jax/_src/pallas",
    ],
)

py_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        "//jax",
        "//jax:core",
        "//jax:mlir",
    ],
)

py_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":lowering",
        "//jax",
        "//jax:mosaic",
        "//jax:source_info_util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

py_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        ":core",
        ":primitives",
        "//jax",
        "//jax:ad_util",
        "//jax:core",
        "//jax:mesh",
        "//jax:mlir",
        "//jax:mosaic",
        "//jax:partial_eval",
        "//jax:source_info_util",
        "//jax:util",
        "//jax:xla",
        "//jax/_src/lib",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":core",
        ":primitives",
        "//jax",
        "//jax:api_util",
        "//jax:util",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)

py_library(
    name = "random",
    srcs = ["random.py"],
    deps = [
        ":primitives",
        "//jax",
        "//jax:typing",
    ] + py_deps("numpy"),
)
