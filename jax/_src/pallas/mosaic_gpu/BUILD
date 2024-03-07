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

# Package for Mosaic-specific Pallas extensions

load("@rules_python//python:defs.bzl", "py_library")
load(
    "//jaxlib:jax.bzl",
    "py_deps",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//:__subpackages__",
    ],
)

py_library(
    name = "mosaic_gpu",
    srcs = ["__init__.py"],
    deps = [
        ":pallas_call_registration",
    ],
)

pytype_strict_library(
    name = "pallas_call_registration",
    srcs = ["pallas_call_registration.py"],
    deps = [
        ":lowering",
        "//jax",
        "//jax:mlir",
        "//jax:mosaic_gpu",
        "//jax/_src/pallas",
    ],
)

pytype_strict_library(
    name = "lowering",
    srcs = ["lowering.py"],
    deps = [
        "//jax",
        "//jax:core",
        "//jax:mlir",
        "//jax:mosaic_gpu",
        "//jax:util",
        "//jax/_src/lib",
        "//jax/_src/pallas",
    ] + py_deps("numpy"),
)
