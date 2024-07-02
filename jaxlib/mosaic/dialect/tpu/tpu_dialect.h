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

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_
#define JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_enums.h.inc"
#include "xla/layout.h"

namespace mlir::tpu {
class TPUDialect;
}  // namespace mlir::tpu

#define GET_ATTRDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_attr_defs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_type_defs.h.inc"

#define GET_OP_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h.inc"
#include "jaxlib/mosaic/dialect/tpu/tpu_ops.h.inc"

namespace mlir {
namespace tpu {

std::pair<bool, bool> mightCommunicateBetweenChips(Operation* op);

std::unique_ptr<OperationPass<func::FuncOp>> createInferMemRefLayoutPass(
    int hardware_generation = -1);

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass();

std::unique_ptr<OperationPass<func::FuncOp>> createInferVectorLayoutPass(
    int lane_count = 128, int sublane_count = 8);

std::unique_ptr<OperationPass<func::FuncOp>> createApplyVectorLayoutPass(
    int hardware_generation = -1, int lane_count = 128, int sublane_count = 8,
    int mxu_contracting_size = 128, int mxu_noncontracting_size = 128,
    int max_sublanes_in_scratch = 0);

std::unique_ptr<OperationPass<func::FuncOp>>
createLogicalToPhysicalDeviceIdPass(int64_t total_devices);

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgVectorizationPass(
    bool supports_bf16_alu_instructions = false,
    bool supports_bf16_matmul = false);

std::unique_ptr<OperationPass<func::FuncOp>> createDebugAssertInsertionPass();

#define GEN_PASS_DECL_MOSAICSERDEPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

// Changes the memory space of the value and propagates it through the program.
LogicalResult specializeMemorySpace(TypedValue<MemRefType> value,
                                    MemorySpace memory_space);

// In Mosaic, we often strip tiled layouts from memrefs, for compatibility with
// vector ops. This functions inverts the layout erasure applied to the value.
MemRefType getMemRefType(Value value);

bool isGuaranteedDivisible(Value value, int64_t divisor, int64_t fuel = 8);

#define GEN_PASS_REGISTRATION
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

}  // namespace tpu
}  // namespace mlir

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_DIALECT_H_
