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
#include <memory>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/StringRef.h"
#include "mlir/include/mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/include/mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
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

}  // namespace

void registerConvertGpuToLLVMPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<ConvertGpuToLLVMPass>();
  });
}

}  // namespace gpu
}  // namespace mosaic
