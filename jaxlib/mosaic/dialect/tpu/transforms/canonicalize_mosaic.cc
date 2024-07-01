#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// TODO(mvoz): Seems to be an issue with how tpu_passes.h.inc is generated.
// It requires these headers, but does not include them.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/SCF/IR/SCF.h"
// END
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/IR/AffineExpr.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_CANONICALIZEMOSAICPASS
#define GEN_PASS_DEF_CANONICALIZEMOSAICPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

class MosaicCanonicalizer {
 public:
  MosaicCanonicalizer() {}

  LogicalResult canonicalize(func::FuncOp op) {
    if (!op.getBody().hasOneBlock()) {
      op.emitOpError("Only one block functions supported");
      return failure();
    }
    return canonicalizeBlock(op.getBody().front());
  }

  LogicalResult canonicalizeBlock(Block &block) {
    // make_early_inc_range is utilized due to op mutation.
    for (Operation &any_op : make_early_inc_range(block)) {
      if (auto op = dyn_cast<tpu::MatmulOp>(any_op)) {
        if (canonicalize(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ContractionOp>(any_op)) {
        if (canonicalize(op).failed()) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult tpu_matmul_rule(tpu::MatmulOp op) {
    ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    const VectorType lhs_ty = lhs.getType();
    const VectorType rhs_ty = rhs.getType();

    auto lhs_element_type = lhs_ty.getElementType();
    auto rhs_element_type = rhs_ty.getElementType();

    if (lhs_element_type != rhs_element_type) {
      if (lhs_element_type.isInteger() || rhs_element_type.isInteger()) {
        op->emitOpError("Mix int/float or different int/int - NYI");
        return failure();
      }
    }
    // TODO(voz): Add more invariants.
    // TODO(voz): Insert extf/ sitof/ etc ops to cast the operands to the
    // correct type for mixed matmul cases.
    return success();
  };

  LogicalResult canonicalize(tpu::MatmulOp op) { return tpu_matmul_rule(op); };

  LogicalResult canonicalize(vector::ContractionOp op) {
    // Rewrite the contraction as a matmul
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto acc = op.getAcc();
    VectorType acc_ty;
    if (!(acc_ty = dyn_cast<VectorType>(acc.getType()))) {
      op->emitOpError("Not implemented: acc must be a vector");
      return failure();
    }

    if (op.getKind() != vector::CombiningKind::ADD) {
      op->emitOpError("Only ADD supported");
      return failure();
    }

    ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());

    MLIRContext *const mlir_ctx = op->getContext();

    auto getMapAttr = [&](const unsigned first, const unsigned second) {
      return AffineMapAttr::get(
          AffineMap::get(3, 0,
                         {getAffineDimExpr(first, mlir_ctx),
                          getAffineDimExpr(second, mlir_ctx)},
                         mlir_ctx));
    };

    const ArrayAttr matmul_indexing_maps = builder.getArrayAttr(
        {getMapAttr(0, 2), getMapAttr(2, 1), getMapAttr(0, 1)});
    const ArrayAttr matmul_indexing_maps_transposed = builder.getArrayAttr(
        {getMapAttr(0, 2), getMapAttr(1, 2), getMapAttr(0, 1)});
    const auto indexing_maps = op.getIndexingMaps();
    if (indexing_maps != matmul_indexing_maps &&
        indexing_maps != matmul_indexing_maps_transposed) {
      return op->emitOpError(
          "Not implemented: Non-matmul or unsupported indexing_maps");
    }
    const bool transpose_rhs = indexing_maps == matmul_indexing_maps_transposed;

    const ArrayAttr matmul_iterator_types =
      builder.getArrayAttr({builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::reduction)});
    if (op->getAttr("iterator_types") != matmul_iterator_types) {
      return op->emitOpError(
        "Not implemented: Non-matmul iterator_types");
    }
    const tpu::ContractPrecisionAttr precision_attr =  // May be null
      op->getAttrOfType<tpu::ContractPrecisionAttr>("precision");
    auto matmul_op = builder.create<tpu::MatmulOp>(
        op->getLoc(), acc_ty, lhs, rhs, acc,
        /*transpose_lhs=*/false, transpose_rhs, precision_attr);
    op.replaceAllUsesWith(matmul_op.getResult());
    op.erase();
    auto result = tpu_matmul_rule(matmul_op);
    return result;
  }
};

struct CanonicalizeMosaicPass
    : public impl::CanonicalizeMosaicPassBase<CanonicalizeMosaicPass> {
  CanonicalizeMosaicPass() {}

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MosaicCanonicalizer vlc;
    if (vlc.canonicalize(func).failed()) {
      signalPassFailure();
    }
  };
};

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass() {
  return std::make_unique<CanonicalizeMosaicPass>();
}
}  // namespace mlir::tpu
