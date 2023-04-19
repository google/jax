/* Copyright 2023 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "jaxlib/mlir/_mlir_libs/passes/jax_passes.h"
#include "jaxlib/mlir/_mlir_libs/passes/jax_passes.h.inc"

// Must include the declarations as they carry important visibility attributes.
#include "jaxlib/mlir/_mlir_libs/passes/jax_passes.capi.h.inc"

using namespace jax;

extern "C" {

#include "jaxlib/mlir/_mlir_libs/passes/jax_passes.capi.cc.inc"
}