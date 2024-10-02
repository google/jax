/* Copyright 2024 The JAX Authors.

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

#include "jaxlib/mosaic/gpu/passes.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/StringRef.h"
#include "mlir/include/mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/include/mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/include/mlir/IR/AffineExpr.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "jaxlib/mosaic/gpu/pass_boilerplate.h"

namespace mosaic {
namespace gpu {

namespace {

class ConvertGpuToLLVMPass
    : public mosaic::gpu::Pass<ConvertGpuToLLVMPass, mlir::ModuleOp> {
 public:
  using mosaic::gpu::Pass<ConvertGpuToLLVMPass, mlir::ModuleOp>::Pass;
  static constexpr llvm::StringLiteral kArgumentName =
      "mosaic-convert-gpu-to-llvm";
  static constexpr llvm::StringLiteral kPassName = "ConvertGpuToLLVMPass";

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::LLVMTypeConverter converter(ctx);
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::gpu::GPUModuleOp>();
    target.addDynamicallyLegalOp<mlir::gpu::LaunchFuncOp>(
        [&](mlir::gpu::LaunchFuncOp op) -> bool {
          return converter.isLegal(op->getOperandTypes()) &&
                 converter.isLegal(op->getResultTypes());
        });
    auto symtab = mlir::SymbolTable(getOperation());
    mlir::populateGpuToLLVMConversionPatterns(converter, patterns, false);
    if (mlir::applyPartialConversion(getOperation(), target,
                                     std::move(patterns))
            .failed()) {
      signalPassFailure();
    }
  }
};

// Convert all array parameters to GPU kernels into byval pointers.
// NVVM backend converts them into arrays in the .param memory space.
// We only use arrays to pass in TMA descriptors, which is why we also
// require 64-byte alignment.
class ByvalInsertionPass
    : public mosaic::gpu::Pass<ByvalInsertionPass, mlir::gpu::GPUModuleOp> {
 public:
  using mosaic::gpu::Pass<ByvalInsertionPass, mlir::gpu::GPUModuleOp>::Pass;
  static constexpr llvm::StringLiteral kArgumentName = "mosaic-byval-insertion";
  static constexpr llvm::StringLiteral kPassName = "ByvalInsertionPass";

  void runOnOperation() override {
    auto result = getOperation().walk([](mlir::LLVM::LLVMFuncOp op) {
      // TODO(apaszke): op.isDeclaration() always returns false...
      if (op.getFunctionBody().empty()) {  // Skip over declarations.
        return mlir::WalkResult::advance();
      }
      auto ptr_ty = mlir::LLVM::LLVMPointerType::get(op.getContext());
      mlir::LLVM::LLVMFunctionType func_ty = op.getFunctionType();
      std::vector<mlir::Type> new_arg_types = func_ty.getParams().vec();
      for (unsigned i = 0; i < op.getNumArguments(); ++i) {
        mlir::BlockArgument arg = op.getArgument(i);
        if (!mlir::isa<mlir::LLVM::LLVMArrayType>(arg.getType())) {
          continue;
        }
        if (op.getArgAttrDict(i)) {
          op->emitOpError(
              "!llvm.array argument already has some argument attributes");
          return mlir::WalkResult::interrupt();
        }
        // It would be a lot simpler to use op.insertArgument, but the
        // impl of FunctionOpInterface for llvm.func is _completely_ broken
        new_arg_types[i] = ptr_ty;
        op.setArgAttr(i, "llvm.byval", mlir::TypeAttr::get(arg.getType()));
        op.setArgAttr(i, "nvvm.grid_constant",
                      mlir::UnitAttr::get(op.getContext()));
        op.setArgAttr(i, "llvm.align",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(op.getContext(), 32), 64));
        arg.setType(ptr_ty);
      }
      op.setFunctionType(mlir::LLVM::LLVMFunctionType::get(
          func_ty.getReturnType(), new_arg_types, func_ty.isVarArg()));
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

class LoopPeelingPass
    : public mosaic::gpu::Pass<LoopPeelingPass, mlir::ModuleOp> {
 public:
  using mosaic::gpu::Pass<LoopPeelingPass, mlir::ModuleOp>::Pass;
  static constexpr llvm::StringLiteral kArgumentName = "mosaic-loop-peeling";
  static constexpr llvm::StringLiteral kPassName = "LoopPeelingPass";

  void runOnOperation() override {
    auto peel_front_attr = getOperation()->getAttrOfType<mlir::IntegerAttr>(
        "mosaic_gpu.loop_peel_front");
    int64_t peel_front = peel_front_attr ? peel_front_attr.getInt() : 0;
    auto peel_end_attr = getOperation()->getAttrOfType<mlir::IntegerAttr>(
        "mosaic_gpu.loop_peel_end");
    int64_t peel_end = peel_end_attr ? peel_end_attr.getInt() : 0;
    if (peel_front < 0 || peel_end < 0) {
      getOperation().emitError("negative loop peel count");
      signalPassFailure();
      return;
    }
    if (peel_front != 0) {
      getOperation().emitError("front loop peeling is not supported yet");
      signalPassFailure();
      return;
    }
    if (peel_end == 0) {
      return;
    }
    getOperation().walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::ForOp op) {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), op);
      mlir::Value split_point = b.create<mlir::arith::SubIOp>(
          op.getUpperBound(),
          b.create<mlir::arith::MulIOp>(
              op.getStep(), b.create<mlir::arith::ConstantIndexOp>(peel_end)));
      b.setInsertionPointAfter(op);
      auto last_step =
          mlir::cast<mlir::scf::ForOp>(b.clone(*op.getOperation()));
      op.getUpperBoundMutable().assign(split_point);
      last_step.getLowerBoundMutable().assign(split_point);
      op.getResults().replaceAllUsesWith(last_step->getResults());
      last_step.getInitArgsMutable().assign(op->getResults());
      return mlir::WalkResult::skip();
    });
  }
};

}  // namespace

void registerConvertGpuToLLVMPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<ConvertGpuToLLVMPass>();
  });
}

void registerByvalInsertionPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<ByvalInsertionPass>();
  });
}

void registerLoopPeelingPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<LoopPeelingPass>();
  });
}

}  // namespace gpu
}  // namespace mosaic
