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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir {
namespace tpu {

LogicalResult UnrollVectorsOp::canonicalize(UnrollVectorsOp op,
                                            PatternRewriter &rewriter) {
  RollVectorsOp roll_op =
      llvm::dyn_cast_or_null<RollVectorsOp>(op.getOperand().getDefiningOp());
  if (!roll_op) {
     return failure();
  }
  if (roll_op.getNumOperands() != op.getNumResults()) {
     return failure();
  }
  for (auto [v1, v2] :
       llvm::zip(roll_op.getOperandTypes(), op.getResultTypes())) {
    if (v1 != v2) {
       return failure();
    }
  }
  rewriter.replaceOp(op, roll_op.getOperands());
  return success();
}

LogicalResult MemRefSliceOp::verify() {
  auto source_type = getMemRefType(getMemRef());
  auto target_type = getType();
  auto target_layout = target_type.getLayout();
  auto target_memory_space = target_type.getMemorySpace();
  // TODO(apaszke): Check that the result has a smaller shape.
  // TODO(apaszke): Check that strides are equivalent.
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type. Note that MemRefType does not allow a null layout so we treat
  // the default identity affine map as an "unset" value instead.
  return success(
      (target_memory_space == nullptr ||
       target_memory_space == source_type.getMemorySpace()) &&
      ((isa<AffineMapAttr>(target_layout) && target_layout.isIdentity()) ||
       target_type.getLayout() == source_type.getLayout()));
}

LogicalResult MemRefSliceOp::canonicalize(MemRefSliceOp op,
                                          PatternRewriter &rewriter) {
  auto erase_layout = op.getMemRef().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout) {
    return failure();
  }
  // Push layout erasure through slicing. It is important we see the layout
  // for lowering and don't make it hard for other ops to query it.
  auto layout_ref = erase_layout.getOperand();
  MemRefType layout_ty = layout_ref.getType();
  auto new_result_type = MemRefType::get(
      op.getResult().getType().getShape(), layout_ty.getElementType(),
      layout_ty.getLayout(), layout_ty.getMemorySpace());
  auto slice = rewriter.create<MemRefSliceOp>(op.getLoc(), new_result_type,
                                              layout_ref, op.getBaseIdx());
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), slice);
  return success();
}

LogicalResult MemRefSqueezeOp::verify() {
  auto source_type = getMemRefType(getInput());
  auto target_type = getType();
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type.
  if (target_type.getMemorySpace() != nullptr &&
      target_type.getMemorySpace() != source_type.getMemorySpace()) {
    emitOpError("Memory spaces do not match.");
    return failure();
  }
  if (target_type.getElementType() != source_type.getElementType()) {
    this->emitOpError("Element types don't match.");
    return failure();
  }
  auto source_shape = source_type.getShape();
  auto target_shape = target_type.getShape();
  int source_index = source_shape.size() - 1;
  int target_index = target_shape.size() - 1;
  auto error_msg = llvm::formatv(
      "Target shape is not valid. "
      "Source type: {0}. Target type: {1}.",
      source_type, target_type);
  while (source_index >= 0 || target_index >= 0) {
    int target_dim = target_index < 0 ? -1 : target_shape[target_index];
    if (source_index < 0) {
       // We have run out of source shape but target shape still remains.
       emitOpError(error_msg);
       return failure();
    }
    int source_dim = source_shape[source_index];
    if (source_dim == target_dim) {
       source_index--;
       target_index--;
    } else {
       // Only the source dim can be 1 here.
       if (source_dim != 1) {
         this->emitOpError(error_msg);
         return failure();
       }
       source_index--;
    }
  }
  return success();
}

LogicalResult MemRefSqueezeOp::canonicalize(MemRefSqueezeOp op,
                                            PatternRewriter &rewriter) {
  auto source_type = getMemRefType(op.getInput());
  auto target_type = op.getType();
  auto erase_layout = op.getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout) {
    return failure();
  }
  // Push layout erasure through squeezing. It is important we see the layout
  // for lowering and don't make it hard for other ops to query it.
  auto layout_ref = erase_layout.getOperand();
  MemRefType layout_ty = layout_ref.getType();
  auto source_shape = source_type.getShape();
  auto target_shape = target_type.getShape();
  int source_index = source_shape.size() - 1;
  int target_index = target_shape.size() - 1;
  auto old_layout = dyn_cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  auto target_strides = old_layout.getTileStrides();
  llvm::SmallVector<int64_t> tile_strides(target_strides.begin(),
                                          target_strides.end());
  // We want to remove all strides that correspond to squeezed dimensions and
  // update the corresponding output layout.
  while (source_index >= 0 || target_index >= 0) {
    int target_dim = target_index < 0 ? -1 : target_shape[target_index];
    int source_dim = source_shape[source_index];
    if (source_dim == target_dim) {
       source_index--;
       target_index--;
    } else {
       // Source index must be 1 here (otherwise verification will have failed).
       // We are safe to mutate the strides vector here because we are looping
       // backwards.
       tile_strides.erase(tile_strides.begin() + source_index);
       source_index--;
    }
  }
  auto new_layout = tpu::TiledLayoutAttr::get(
      source_type.getContext(), old_layout.getTiles(), tile_strides);
  auto new_result_type = MemRefType::get(op.getResult().getType().getShape(),
                                         layout_ty.getElementType(), new_layout,
                                         layout_ty.getMemorySpace());
  auto squeeze = rewriter.create<MemRefSqueezeOp>(op.getLoc(), new_result_type,
                                                  layout_ref);
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), squeeze);
  return success();
}

LogicalResult ReinterpretCastOp::verify() {
  auto source_type = getMemRefType(getInput());
  auto target_type = getType();
  return success(
      source_type.getMemorySpace() &&  // Require memory space annotations.
      source_type.getMemorySpace() == target_type.getMemorySpace());
}

// a + matmul(l, r, 0) == matmul(l, r, a)
template <typename AddOp>
class CanonicalizeAddOfMatmul : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const {
    auto try_canonicalize = [&](Value maybe_matmul, Value maybe_acc) {
      auto matmul = dyn_cast_if_present<MatmulOp>(maybe_matmul.getDefiningOp());
      if (!matmul) {
        return failure();
      }
      if (auto const_acc = matmul.getAcc().getDefiningOp<arith::ConstantOp>();
          const_acc &&
          const_acc.getValue() == rewriter.getZeroAttr(const_acc.getType())) {
        IRMapping remap;
        remap.map(matmul.getAcc(), maybe_acc);
        Operation *new_matmul = rewriter.clone(*matmul, remap);
        rewriter.replaceOp(op, new_matmul->getResult(0));
        return success();
      }
      return failure();
    };
    return success(succeeded(try_canonicalize(op.getLhs(), op.getRhs())) ||
                   succeeded(try_canonicalize(op.getLhs(), op.getRhs())));
  }
};

void MatmulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<CanonicalizeAddOfMatmul<arith::AddFOp>,
               CanonicalizeAddOfMatmul<arith::AddIOp>>(context);
}

LogicalResult MaskCastOp::verify() {
  auto input_ty = getInput().getType();
  auto output_ty = getResult().getType();
  return success(input_ty.getElementType() == output_ty.getElementType() &&
                 output_ty.getRank() == 3 &&
                 (input_ty.getRank() == 2 ||
                  (input_ty.getRank() == 3 &&
                   input_ty.getDimSize(2) < output_ty.getDimSize(2))) &&
                 input_ty.getShape().take_front(2) ==
                     output_ty.getShape().take_front(2));
  return success();
}

LogicalResult GetBarrierSemaphoreOp::verify() {
  auto sem_type = getMemRefType(getResult());
  if (sem_type.getRank() != 0) {
    emitOpError("Barrier semaphore reference must be rank 0");
    return failure();
  }
  return success();
}

LogicalResult SemaphoreSignalOp::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    emitOpError("Semaphore reference must be rank 0");
    return failure();
  }
  return success();
}

LogicalResult EnqueueDMAOp::verify() {
  auto source_sem = getSourceSemaphore();
  if (source_sem) {
    auto source_sem_type = getMemRefType(getSourceSemaphore());
    if (source_sem_type.getRank() != 0) {
      emitOpError("DMA source semaphore reference must be rank 0");
      return failure();
    }
  }
  auto target_sem_type = getMemRefType(getTargetSemaphore());
  if (target_sem_type.getRank() != 0) {
    emitOpError("DMA target semaphore must be rank 0");
    return failure();
  }
  return success();
}

LogicalResult WaitDMAOp::verify() {
  auto sem_type = getMemRefType(getSemaphore());
  if (sem_type.getRank() != 0) {
    emitOpError("DMA wait semaphore must be rank 0");
    return failure();
  }
  return success();
}

}  // namespace tpu
}  // namespace mlir

#define GET_OP_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_ops.cc.inc"
