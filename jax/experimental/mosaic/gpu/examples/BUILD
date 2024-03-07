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

load("//jaxlib:jax.bzl", "py_deps")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//third_party/py/jax:mosaic_gpu_users"],
)

exports_files(
    srcs = [
        "flash_attention.py",
        "matmul.py",
    ],
    visibility = ["//third_party/py/jax:internal"],
)

py_library(
    name = "matmul",
    srcs = ["matmul.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/jax:mosaic_gpu",
    ],
)

py_library(
    name = "flash_attention",
    srcs = ["flash_attention.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/jax:mosaic_gpu",
    ],
)

py_test(
    name = "run_matmul",
    srcs = ["matmul.py"],
    main = "matmul.py",
    tags = [
        "manual",
        "notap",
        "requires-gpu-sm90-only",
    ],
    deps = [
        "//learning/brain/research/jax:gpu_support",
        "//third_party/py/jax",
        "//third_party/py/jax:mosaic_gpu",
    ] + py_deps("numpy"),
)
