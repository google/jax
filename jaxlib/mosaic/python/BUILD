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

# Mosaic Python bindings

load("@rules_python//python:defs.bzl", "py_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_filegroup")

gentbl_filegroup(
    name = "tpu_python_gen_raw",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=tpu",
            ],
            "_tpu_gen_raw.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = ":tpu_python.td",
    deps = [
        "//jaxlib/mosaic:tpu_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

genrule(
    name = "tpu_python_gen",
    srcs = ["_tpu_gen_raw.py"],
    outs = ["_tpu_gen.py"],
    cmd = "cat $(location _tpu_gen_raw.py) | sed -e 's/^from \\./from jaxlib\\.mlir\\.dialects\\./g' > $@",
)

py_library(
    name = "tpu_dialect",
    srcs = [
        "_tpu_gen.py",
        "tpu.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//jaxlib/mlir",
        "//jaxlib/mlir/_mlir_libs:_tpu_ext_lib",
    ],
)

py_library(
    name = "layout_defs",
    srcs = [
        "layout_defs.py",
    ],
    visibility = [
        "//visibility:public",
    ],
)
