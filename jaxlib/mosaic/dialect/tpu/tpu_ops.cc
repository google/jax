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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"

namespace mlir {
namespace tpu {

LogicalResult UnrollVectorsOp::canonicalize(UnrollVectorsOp op,
                                            PatternRewriter &rewriter) {
  RollVectorsOp roll_op =
      dyn_cast_or_null<RollVectorsOp>(op.getOperand().getDefiningOp());
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

LogicalResult BitcastOp::verify() {
  auto in_ty = getInput().getType();
  auto out_ty = getOutput().getType();
  auto in_bitwidth = in_ty.getElementTypeBitWidth();
  auto out_bitwidth = out_ty.getElementTypeBitWidth();
  if (in_bitwidth != out_bitwidth) {
    if (in_ty.getRank() < 2 || out_ty.getRank() < 2) {
      return emitError(
          "Not implemented: bitcast between different bitwidths on a 1D "
          "vector.");
    }
    SmallVector<int64_t, 4> in_shape(in_ty.getShape());
    SmallVector<int64_t, 4> out_shape(out_ty.getShape());
    *(in_shape.end() - 2) *= in_bitwidth;
    *(out_shape.end() - 2) *= out_bitwidth;
    if (in_shape != out_shape) {
      return emitError(
          "Expected input and output shapes are the same after multiplying the "
          "second-minor dimension by the ratio of bitwidths.");
    }
  } else if (in_ty.getShape() != out_ty.getShape()) {
    return emitError(
        "Expected input and output shapes are the same when bitwidth does not "
        "change.");
  }
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
       target_type.getLayout() == source_type.getLayout()) &&
      getDynamicSizes().size() == target_type.getNumDynamicDims());
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
  auto slice =
      rewriter.create<MemRefSliceOp>(op.getLoc(), new_result_type, layout_ref,
                                     op.getBaseIdx(), op.getDynamicSizes());
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
  SmallVector<int64_t> tile_strides(target_strides.begin(),
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

LogicalResult MemRefReshapeOp::verify() {
  auto src_ty = getMemRefType(getInput());
  auto tgt_ty = getType();
  if (tgt_ty.getMemorySpace() != nullptr &&
      tgt_ty.getMemorySpace() != src_ty.getMemorySpace()) {
    return emitOpError("Memory spaces do not match.");
  }
  if (src_ty.getShape().size() < 2 || tgt_ty.getShape().size() < 2) {
    return emitError("Not implemented: 1d memref reshape.");
  }
  if (tgt_ty.getElementType() != src_ty.getElementType()) {
    return emitOpError("Element types don't match.");
  }
  auto src_elements_num = ShapedType::getNumElements(src_ty.getShape());
  auto tgt_elements_num = ShapedType::getNumElements(tgt_ty.getShape());
  if (src_elements_num != tgt_elements_num) {
    return emitOpError(
        "Number of elements doesn't match between input and output memref "
        "type.");
  }
  // Source and target attributes may be different before propagation is done by
  // the canonicalizer, so we allow this when attributes are "unset" in the
  // target type.
  auto tgt_layout = dyn_cast<tpu::TiledLayoutAttr>(tgt_ty.getLayout());
  if (!tgt_layout) {
    return success();
  }
  auto src_layout = dyn_cast<tpu::TiledLayoutAttr>(src_ty.getLayout());
  if (!src_layout || src_layout.getTiles().empty()) {
    return emitOpError("Expected a tiled layout for the input memref.");
  }
  if (src_layout.getTiles() != tgt_layout.getTiles()) {
    return emitOpError(
        "Expected the same tiling for the input and output memref.");
  }
  auto tile = src_layout.getTiles().front().dimensions();
  if (tile.size() != 2) {
    return emitOpError("Not implemented: memref reshape with 1D tiling.");
  }
  SmallVector<int64_t> src_tile_strides(src_layout.getTileStrides());
  if (ComputeTileStrides(src_ty, tile) != src_tile_strides) {
    return emitOpError("Not implemented: reshape on a non-contiguous memref.");
  }
  auto src_tiled_shape = src_ty.getShape().take_back(2);
  auto tgt_tiled_shape = tgt_ty.getShape().take_back(2);
  bool is_src_align_tile_2nd_minor = src_tiled_shape[0] % tile[0] == 0;
  bool is_src_align_tile_minor = src_tiled_shape[1] % tile[1] == 0;
  bool is_tgt_align_tile_2nd_minor = tgt_tiled_shape[0] % tile[0] == 0;
  bool is_tgt_align_tile_minor = tgt_tiled_shape[1] % tile[1] == 0;
  if (tile[0] == 1 && is_src_align_tile_minor && is_tgt_align_tile_minor) {
    // When the tiling is (1, ?) and the source and target shapes are aligned
    // to the tile, we support reshape on any dims.
  } else if (tgt_tiled_shape[1] != src_tiled_shape[1]) {
    return emitError("Expected the minormost dimension to be unchanged");
  } else if (tgt_tiled_shape[0] != src_tiled_shape[0]) {
    if (!is_src_align_tile_2nd_minor || !is_tgt_align_tile_2nd_minor) {
      return emitError(
          "Expected the 2nd minor dimension is aligned to the tile");
    }
  }
  return success();
}

LogicalResult MemRefReshapeOp::canonicalize(MemRefReshapeOp op,
                                            PatternRewriter &rewriter) {
  auto src_ty = op.getInput().getType();
  auto dst_ty = op.getType();
  auto erase_layout_op = op.getInput().getDefiningOp<tpu::EraseLayoutOp>();
  if (!erase_layout_op) {
    return failure();
  }
  auto layout_ref = erase_layout_op.getOperand();
  auto layout_ty = layout_ref.getType();
  auto layout =
      dyn_cast<tpu::TiledLayoutAttr>(layout_ty.getLayout());
  CHECK(!layout.getTiles().empty());
  auto tile = layout.getTiles().front().dimensions();
  auto new_tile_strides = ComputeTileStrides(dst_ty, tile);
  auto new_layout = tpu::TiledLayoutAttr::get(
      src_ty.getContext(), layout.getTiles(), new_tile_strides);
  auto new_result_ty =
      MemRefType::get(dst_ty.getShape(), dst_ty.getElementType(), new_layout,
                      layout_ty.getMemorySpace());
  auto reshape =
      rewriter.create<MemRefReshapeOp>(op.getLoc(), new_result_ty, layout_ref);
  rewriter.replaceOpWithNewOp<EraseLayoutOp>(op, op.getType(), reshape);
  return success();
}

template <typename Op>
LogicalResult verifyStridedOp(Op op, MemRefType memref_ty,
                              VectorType vector_ty) {
  auto indices = op.getIndices();
  auto strides = op.getStrides();
  if (memref_ty.getRank() != indices.size()) {
    op.emitError("Base memref's rank and indices size do not match: ")
        << memref_ty.getRank() << " vs " << indices.size();
    return failure();
  }
  if (memref_ty.getRank() != strides.size()) {
    op.emitError("Base memref's rank and strides size do not match: ")
        << memref_ty.getRank() << " vs " << strides.size();
    return failure();
  }
  if (memref_ty.getRank() != vector_ty.getRank()) {
    op.emitError("Base memref's rank and result's rank do not match: ")
        << memref_ty.getRank() << " vs " << vector_ty.getRank();
    return failure();
  }
  for (int64_t i = 0; i < memref_ty.getRank(); ++i) {
    if (strides[i] < 1) {
      op.emitError("Strides[") << i << "]=" << strides[i] << " must be >= 1";
      return failure();
    }
  }
  return success();
}

LogicalResult StridedLoadOp::verify() {
  return verifyStridedOp<StridedLoadOp>(*this, getMemRefType(getBase()),
                                        getType());
}

LogicalResult StridedStoreOp::verify() {
  return verifyStridedOp<StridedStoreOp>(*this, getMemRefType(getBase()),
                                         getValueToStore().getType());
}

LogicalResult ReinterpretCastOp::verify() {
  auto source_type = getMemRefType(getInput());
  auto target_type = getType();
  return success(
      source_type.getMemorySpace() &&  // Require memory space annotations.
      source_type.getMemorySpace() == target_type.getMemorySpace());
}

template <typename Op>
LogicalResult verifyRotateOp(Op op) {
  auto vty = op.getResult().getType();
  if (vty.getRank() <= op.getDimension() || op.getDimension() < 0) {
    op.emitOpError("Invalid dimension: ") << op.getDimension();
    return failure();
  }
  if (op.getStride().has_value() && op.getStride().value() < 0) {
    op.emitOpError("Rotate stride must be >= 0 if it is specified");
    return failure();
  }
  if (op.getStrideDimension().has_value() &&
      (vty.getRank() <= op.getStrideDimension().value() ||
       op.getStrideDimension().value() < 0)) {
    op.emitOpError("Invalid stride dimension: ")
        << op.getStrideDimension().value();
    return failure();
  }
  if (op.getStride().has_value() != op.getStrideDimension().has_value()) {
    op.emitOpError(
        "Expected  either none or both stride and stride dimension are "
        "present");
    return failure();
  }
  return success();
}

// TODO(b/347016737): deprecate static rotate
LogicalResult RotateOp::verify() { return verifyRotateOp<RotateOp>(*this); }

LogicalResult DynamicRotateOp::verify() {
  return verifyRotateOp<DynamicRotateOp>(*this);
}

// a + matmul(l, r, 0) == matmul(l, r, a)
template <typename AddOp>
class CanonicalizeAddOfMatmul : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const {
    auto try_canonicalize = [&](Value maybe_matmul, Value maybe_acc) {
      auto matmul = dyn_cast_if_present<MatmulOp>(maybe_matmul.getDefiningOp());
      if (!matmul || !matmul->hasOneUse()) {
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
    return emitOpError("Semaphore reference must be rank 0");
  }
  return success();
}

LogicalResult EnqueueDMAOp::verify() {
  auto source_sem = getSourceSemaphore();
  if (source_sem) {
    auto source_sem_type = getMemRefType(getSourceSemaphore());
    if (source_sem_type.getRank() != 0) {
      return emitOpError("DMA source semaphore reference must be rank 0");
    }
  }
  auto target_sem_type = getMemRefType(getTargetSemaphore());
  if (target_sem_type.getRank() != 0) {
    return emitOpError("DMA target semaphore must be rank 0");
  }
  if (getDeviceId() || getCoreId()) {
    if (!getSourceSemaphore()) {
      return emitOpError(
          "DMA source semaphore must be specified when "
          "device_id or core_id is specified");
    }
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

LogicalResult RegionOp::verify() {
  for (auto result_type : getResultTypes()) {
    if (!isa<FloatType, IntegerType, VectorType, IndexType>(result_type)) {
      return emitOpError(
          "Region result must be float, int, index or a vector type.");
    }
  }
  return success();
}

}  // namespace tpu
}  // namespace mlir

#define GET_OP_CLASSES
#include "jaxlib/mosaic/dialect/tpu/tpu_ops.cc.inc"
