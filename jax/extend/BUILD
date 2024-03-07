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
    "py_library_providing_imports_info",
    "pytype_strict_library",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:jax_extend_users"],
)

pytype_strict_library(
    name = "extend",
    srcs = ["__init__.py"],
    deps = [
        ":backend",
        ":core",
        ":ffi",
        ":linear_util",
        ":random",
        ":source_info_util",
    ],
)

py_library_providing_imports_info(
    name = "core",
    srcs = glob(["core/**/*.py"]),
    lib_rule = pytype_strict_library,
    deps = [
        "//jax",
        "//jax:abstract_arrays",
        "//jax:ad_util",
        "//jax:core",
    ],
)

pytype_strict_library(
    name = "linear_util",
    srcs = ["linear_util.py"],
    deps = ["//jax:core"],
)

pytype_strict_library(
    name = "backend",
    srcs = ["backend.py"],
    deps = ["//jax"],
)

pytype_strict_library(
    name = "random",
    srcs = ["random.py"],
    deps = ["//jax"],
)

pytype_strict_library(
    name = "source_info_util",
    srcs = ["source_info_util.py"],
    deps = ["//jax:source_info_util"],
)

pytype_strict_library(
    name = "ffi",
    srcs = ["ffi.py"],
    deps = ["//jax"],
)
