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

#include <cstdint>
#include <functional>
#include <memory>

#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_TYPECONVERTINSERTIONPASS
#define GEN_PASS_DEF_TYPECONVERTINSERTIONPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

using rule_type = std::function<void(Operation *)>;
static constexpr int8_t kNativeBitwidth = 32;

template <typename Op>
rule_type as_generic_rule(void (*rule)(Op)) {
  return [rule](const Operation *op) { return rule(cast<Op>(op)); };
}

void shape_cast_rule(vector::ShapeCastOp op) {
  auto src_ty = op.getSourceVectorType();
  auto res_ty = op.getResultVectorType();
  ImplicitLocOpBuilder builder(op->getLoc(), op);

  // When a vector has a packed dtype, a native packed tiling and the sublane
  // dim size is 1, it indicates that the tile has paddings on sublane dim. We
  // need to convert to x32 to do reshape.
  if (src_ty.getRank() >= 2 && res_ty.getRank() >= 2 &&
      src_ty.getElementTypeBitWidth() == res_ty.getElementTypeBitWidth() &&
      src_ty.getElementTypeBitWidth() < kNativeBitwidth &&
      *(src_ty.getShape().end() - 1) == *(res_ty.getShape().end() - 1) &&
      (*(src_ty.getShape().end() - 2) == 1 ||
       *(src_ty.getShape().end() - 2) == 1)) {
    if (src_ty.getElementType().isInteger()) {
      auto new_op = builder.create<arith::TruncIOp>(
          res_ty,
          builder.create<vector::ShapeCastOp>(
              VectorType::get(res_ty.getShape(), builder.getI32Type()),
              builder.create<arith::ExtSIOp>(
                  VectorType::get(src_ty.getShape(), builder.getI32Type()),
                  op.getOperand())));
      op->replaceAllUsesWith(new_op);
      op->erase();
    } else {
      auto new_op = builder.create<arith::TruncFOp>(
          res_ty,
          builder.create<vector::ShapeCastOp>(
              VectorType::get(res_ty.getShape(), builder.getF32Type()),
              builder.create<arith::ExtFOp>(
                  VectorType::get(src_ty.getShape(), builder.getF32Type()),
                  op.getOperand())));
      op->replaceAllUsesWith(new_op);
      op->erase();
    }
  }
}

const llvm::StringMap<rule_type> &rules() {
  static auto rules = new llvm::StringMap<rule_type>{
      // TODO: handle matmul with mixed types.
      {vector::ShapeCastOp::getOperationName(),
       as_generic_rule(shape_cast_rule)},
  };
  return *rules;
}

struct TypeConvertInsertionPass
    : public impl::TypeConvertInsertionPassBase<TypeConvertInsertionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([](Operation *op) {
      if (auto rule_it = rules().find(op->getName().getStringRef());
          rule_it != rules().end()) {
        const rule_type &rule = rule_it->getValue();
        rule(op);
      }
      return WalkResult::advance();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTypeConvertInsertionPass() {
  return std::make_unique<TypeConvertInsertionPass>();
}

}  // namespace mlir::tpu
