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

load("@rules_python//python:defs.bzl", "py_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = [
        "//:__subpackages__",
    ],
)

py_library(
    name = "mosaic",
    deps = [
        "//jaxlib/mosaic/python:tpu_dialect",
    ],
)

################################################################################
# TPU dialect

cc_library(
    name = "tpu_dialect",
    srcs = [
        "dialect/tpu/layout.cc",
        "dialect/tpu/tpu_dialect.cc",
        "dialect/tpu/tpu_ops.cc",
        "dialect/tpu/util.h",
    ] + glob([
        "dialect/tpu/transforms/*.cc",
    ]),
    hdrs = [
        "dialect/tpu/layout.h",
        "dialect/tpu/tpu_dialect.h",
    ] + glob([
        "dialect/tpu/transforms/*.h",
    ]),
    # compatible with libtpu
    deps = [
        ":tpu_inc_gen",
        ":tpu_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorTransforms",
        "@xla//xla:array",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

gentbl_cc_library(
    name = "tpu_ops_inc_gen",
    # compatible with libtpu
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "dialect/tpu/tpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "dialect/tpu/tpu_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/tpu/tpu_ops.td",
    deps = [":tpu_td_files"],
)

gentbl_cc_library(
    name = "tpu_inc_gen",
    # compatible with libtpu
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "dialect/tpu/tpu_dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "dialect/tpu/tpu_dialect.cc.inc",
        ),
        (
            ["-gen-enum-decls"],
            "dialect/tpu/tpu_enums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "dialect/tpu/tpu_enums.cc.inc",
        ),
        (
            ["-gen-attrdef-decls"],
            "dialect/tpu/tpu_attr_defs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "dialect/tpu/tpu_attr_defs.cc.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "dialect/tpu/tpu_type_defs.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "dialect/tpu/tpu_type_defs.cc.inc",
        ),
        (
            [
                "-gen-pass-decls",
                "-name=TPU",
            ],
            "dialect/tpu/tpu_passes.h.inc",
        ),
        (
            [
                "-gen-pass-capi-header",
                "--prefix=TPU",
            ],
            "dialect/tpu/integrations/c/tpu_passes.capi.h.inc",
        ),
        (
            [
                "-gen-pass-capi-impl",
                "--prefix=TPU",
            ],
            "dialect/tpu/integrations/c/tpu_passes.capi.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/tpu/tpu.td",
    deps = [":tpu_td_files"],
)

td_library(
    name = "tpu_td_files",
    srcs = [
        "dialect/tpu/tpu.td",
        "dialect/tpu/tpu_ops.td",
    ],
    # compatible with libtpu
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

# C API targets

TPU_CAPI_SOURCES = [
    "dialect/tpu/integrations/c/tpu_dialect.cc",
    "dialect/tpu/integrations/c/tpu_passes.capi.cc.inc",
]

TPU_CAPI_HEADERS = [
    "dialect/tpu/integrations/c/tpu_dialect.h",
    "dialect/tpu/integrations/c/tpu_passes.capi.h.inc",
]

cc_library(
    name = "tpu_dialect_capi",
    srcs = TPU_CAPI_SOURCES,
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_dialect",
        ":tpu_inc_gen",
        ":tpu_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@xla//xla:array",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "tpu_dialect_capi_headers",
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_inc_gen",
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "tpu_dialect_capi_objects",
    srcs = TPU_CAPI_SOURCES,
    hdrs = TPU_CAPI_HEADERS,
    deps = [
        ":tpu_dialect",
        ":tpu_inc_gen",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = True,
)
