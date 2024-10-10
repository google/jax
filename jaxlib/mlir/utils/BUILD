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

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//jax:internal"],
)

# TODO(mvoz):Break up into smaller targets, utils is a catchall name.
cc_library(
    name = "mlir_utils",
    hdrs = ["mlir_utils.h"],
    # compatible with libtpu
    features = ["-use_header_modules"],
    visibility = [
        "//platforms/xla/service:__subpackages__",
    ],
    deps = [
        "//jaxlib/mosaic:tpu_dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
    ],
)
