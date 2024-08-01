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

load(
    "//jaxlib:jax.bzl",
    "if_building_jaxlib",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:jax_extend_users"],
)

pytype_strict_library(
    name = "arithmetic_dialect",
    srcs = ["arith.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:arithmetic_dialect"]),
)

pytype_strict_library(
    name = "builtin_dialect",
    srcs = ["builtin.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:builtin_dialect"]),
)

pytype_strict_library(
    name = "chlo_dialect",
    srcs = ["chlo.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:chlo_dialect"]),
)

pytype_strict_library(
    name = "func_dialect",
    srcs = ["func.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:func_dialect"]),
)

pytype_strict_library(
    name = "math_dialect",
    srcs = ["math.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:math_dialect"]),
)

pytype_strict_library(
    name = "memref_dialect",
    srcs = ["memref.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:memref_dialect"]),
)

pytype_strict_library(
    name = "scf_dialect",
    srcs = ["scf.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:scf_dialect"]),
)

pytype_strict_library(
    name = "sparse_tensor_dialect",
    srcs = ["sparse_tensor.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:sparse_tensor_dialect"]),
)

pytype_strict_library(
    name = "sdy_dialect",
    srcs = ["sdy.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:sdy_dialect"]),
)

pytype_strict_library(
    name = "stablehlo_dialect",
    srcs = ["stablehlo.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:stablehlo_dialect"]),
)

pytype_strict_library(
    name = "vector_dialect",
    srcs = ["vector.py"],
    deps = if_building_jaxlib(["//jaxlib/mlir:vector_dialect"]),
)
