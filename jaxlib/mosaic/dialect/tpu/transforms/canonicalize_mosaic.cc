#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// It requires these headers, but does not include them.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/ArrayRef.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/SmallVector.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/include/mlir/IR/AffineExpr.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_CANONICALIZEMOSAICPASS
#define GEN_PASS_DEF_CANONICALIZEMOSAICPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

LogicalResult tpu_matmul_rule(tpu::MatmulOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());

  auto transpose_lhs = op.getTransposeLhs();

  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto acc = op.getAcc();

  const VectorType lhs_ty = lhs.getType();
  const VectorType rhs_ty = rhs.getType();
  const VectorType acc_ty = acc.getType();

  auto lhs_element_type = lhs_ty.getElementType();
  auto rhs_element_type = rhs_ty.getElementType();
  auto acc_element_type = acc_ty.getElementType();

  // there are a few primary paths for dimension_numbers in matmul
  // 1) empty and no transpose_lhs -> in which case, we define a default mkn or
  // use the legacy transpose_rhs path, for now.
  // 2) empty and transpose_lhs -> Illegal construction
  // 3) defined and not default -> in which case, we verify them, and apply them
  // 4) defined and default -> no op for batching and transposition
  std::optional<int64_t> batch_size = std::nullopt;
  std::optional<std::function<mlir::vector::TransposeOp(mlir::Value &)>>
      lhs_transposition = std::nullopt;
  std::optional<std::function<mlir::vector::TransposeOp(mlir::Value &)>>
      rhs_transposition = std::nullopt;
  // Vanilla mkn
  if (!op.getDimensionNumbers().has_value() && !transpose_lhs) {
    op.setDimensionNumbersAttr(
        default_dimension_numbers(builder, false, false));
    // NOTE - we always disregard transposition here - because if the user
    // set transpose, we want to fail the default dimension numbers check
    // in order to apply transposition correctly.
  } else if (!op.getDimensionNumbers().has_value() && !transpose_lhs) {
    op.emitOpError(
        "Not implemented: default dimension numbers must be set for "
        "transpose_lhs");
    return failure();
  } else if (op.getDimensionNumbers().value() !=
             default_dimension_numbers(builder, false, false)) {
    // TODO(mvoz): A bunch of these invariants can probably go to the verifier.

    auto dimension_numbers = op.getDimensionNumbers();
    auto lhs_contracting_dims = dimension_numbers->getLhsContractingDims();
    auto rhs_contracting_dims = dimension_numbers->getRhsContractingDims();

    auto transposition_fn = [&builder](auto &lhs) -> auto {
      auto lhs_ty = lhs.getType().template cast<VectorType>();
      // This function must run on 2d vectors - we rely on the rest of the
      // rule to strip away batch dimensions and handle loops.
      CHECK_EQ(lhs_ty.getShape().size(), 2);
      auto minor_perm = {1, 0};
      // swap the shape dims
      std::vector<int64_t> shape(lhs_ty.getShape());
      std::swap(shape[0], shape[1]);

      auto lhs_ty_tranposed = VectorType::get(shape, lhs_ty.getElementType());

      const SmallVector<int64_t> minor_perm_vec =
          SmallVector<int64_t>(minor_perm.begin(), minor_perm.end());
      auto new_transpose_op = builder.create<vector::TransposeOp>(
          lhs_ty_tranposed, lhs,
          DenseI64ArrayAttr::get(builder.getContext(), minor_perm_vec));

      return new_transpose_op;
    };

    if (lhs_contracting_dims.size() != 1) {
      op->emitOpError(
          "Not implemented: lhs contracting dims must be of size 1");
      return failure();
    }
    if (rhs_contracting_dims.size() != 1) {
      op->emitOpError(
          "Not implemented: rhs contracting dims must be of size 1");
      return failure();
    }

    auto lhs_contracting_dim = lhs_contracting_dims[0];
    auto rhs_contracting_dim = rhs_contracting_dims[0];

    auto lhs_batch_dims = dimension_numbers->getLhsBatchDims();
    auto rhs_batch_dims = dimension_numbers->getRhsBatchDims();

    auto lhs_non_contracting_dims =
        dimension_numbers->getLhsNonContractingDims();
    auto rhs_non_contracting_dims =
        dimension_numbers->getRhsNonContractingDims();

    // size of contracting_im + non_contracting_dims + batch_dims must be the
    // size of the shape.
    if (lhs_contracting_dims.size() + lhs_non_contracting_dims.size() +
            lhs_batch_dims.size() !=
        lhs_ty.getShape().size()) {
      op->emitOpError(
          "Not implemented: lhs contracting + non contracting + batch dims "
          "must be "
          "of the same size as the lhs shape");
      return failure();
    }
    if (rhs_contracting_dims.size() + rhs_non_contracting_dims.size() +
            rhs_batch_dims.size() !=
        rhs_ty.getShape().size()) {
      op->emitOpError(
          "Not implemented: rhs contracting + non contracting + batch dims "
          "must be "
          "of the same size as the rhs shape");
      return failure();
    }

    if (lhs_ty.getShape()[lhs_contracting_dim] !=
        rhs_ty.getShape()[rhs_contracting_dim]) {
      op->emitOpError(
          "Not implemented: lhs and rhs contracting dims must be of the same "
          "size");
      return failure();
    }

    if (lhs_batch_dims.size() != rhs_batch_dims.size()) {
      op->emitOpError("Not implemented: Up to 1 batch dim supported");
      return failure();
    }
    if (lhs_batch_dims.size() > 1) {
      op->emitOpError("Not implemented: Up to 1 batch dim supported");
      return failure();
    }

    // Ensure no overlap - batch dims cannot be found in contracting dims
    for (int64_t dim : lhs_batch_dims) {
      if (llvm::is_contained(lhs_contracting_dims, dim)) {
        op->emitOpError(
            "Illegal: batch dims cannot overlap w/ contracting dims");
        return failure();
      }
    }
    for (int64_t dim : rhs_batch_dims) {
      if (llvm::is_contained(rhs_contracting_dims, dim)) {
        op->emitOpError(
            "Illegal: batch dims cannot overlap in contracting dims");
        return failure();
      }
    }
    // Contracting dims should not overlap non-contracting dims
    for (int64_t dim : lhs_contracting_dims) {
      if (llvm::is_contained(lhs_non_contracting_dims, dim)) {
        op->emitOpError(
            "Illegal: contracting dims cannot be found in non-contracting "
            "dims");
        return failure();
      }
    }
    for (int64_t dim : rhs_contracting_dims) {
      if (llvm::is_contained(rhs_non_contracting_dims, dim)) {
        op->emitOpError() << "Illegal: contracting dims cannot be found in "
                             "non-contracting dims";
        return failure();
      }
    }

    std::vector<int64_t> lhs_all_dims;
    lhs_all_dims.insert(lhs_all_dims.end(), lhs_contracting_dims.begin(),
                        lhs_contracting_dims.end());
    lhs_all_dims.insert(lhs_all_dims.end(), lhs_non_contracting_dims.begin(),
                        lhs_non_contracting_dims.end());
    lhs_all_dims.insert(lhs_all_dims.end(), lhs_batch_dims.begin(),
                        lhs_batch_dims.end());

    std::vector<int64_t> rhs_all_dims;
    rhs_all_dims.insert(rhs_all_dims.end(), rhs_contracting_dims.begin(),
                        rhs_contracting_dims.end());
    rhs_all_dims.insert(rhs_all_dims.end(), rhs_non_contracting_dims.begin(),
                        rhs_non_contracting_dims.end());
    rhs_all_dims.insert(rhs_all_dims.end(), rhs_batch_dims.begin(),
                        rhs_batch_dims.end());

    // Create reference dimension sets (0, 1, ..., N-1)
    std::vector<int64_t> lhs_ref_dims(lhs_ty.getShape().size());
    std::iota(lhs_ref_dims.begin(), lhs_ref_dims.end(), 0);

    std::vector<int64_t> rhs_ref_dims(rhs_ty.getShape().size());
    std::iota(rhs_ref_dims.begin(), rhs_ref_dims.end(), 0);

    if (!std::is_permutation(lhs_all_dims.begin(), lhs_all_dims.end(),
                             lhs_ref_dims.begin())) {
      op->emitOpError(
          "lhs contracting + non-contracting + batch dims must be a "
          "permutation of lhs shape dims");
      return failure();
    }

    if (!std::is_permutation(rhs_all_dims.begin(), rhs_all_dims.end(),
                             rhs_ref_dims.begin())) {
      op->emitOpError(
          "rhs contracting + non-contracting + batch dims must be a "
          "permutation of rhs shape dims");
      return failure();
    }

    // At this point, we always have dimension numbers, and they are valid.
    const std::optional<int64_t> batch_dim_lhs =
        lhs_batch_dims.empty() ? std::nullopt
                               : std::optional<int64_t>(lhs_batch_dims[0]);
    const std::optional<int64_t> batch_dim_rhs =
        rhs_batch_dims.empty() ? std::nullopt
                               : std::optional<int64_t>(rhs_batch_dims[0]);
    if (batch_dim_lhs != batch_dim_rhs) {
      op->emitOpError("Not Implemented: batch dims must be equal");
      return failure();
    }
    if (batch_dim_lhs.has_value() && (batch_dim_lhs.value() != 0)) {
      op->emitOpError("Not Implemented: batch dims pos must be 0");
      return failure();
    }
    // Invariant above enforces only 1 batch dim atm, and that both are eq
    if (batch_dim_lhs.has_value()) {
      batch_size = lhs_ty.getShape()[batch_dim_lhs.value()];
      if (batch_size == 0) {
        op->emitOpError("Illegal: batch size must be > 0");
        return failure();
      }
    }
    // Lower each dim in contracting dims by size(batch_dims)
    std::vector<int64_t> batch_adjusted_lhs_contracting_dims =
        lhs_contracting_dims;
    auto lhs_adjustment = lhs_batch_dims.size();
    for (int64_t i = 0; i < batch_adjusted_lhs_contracting_dims.size(); ++i) {
      batch_adjusted_lhs_contracting_dims[i] -= lhs_adjustment;
    }

    if (batch_adjusted_lhs_contracting_dims != std::vector<int64_t>({1})) {
      lhs_transposition = transposition_fn;
    }
    std::vector<int64_t> batch_adjusted_rhs_contracting_dims =
        rhs_contracting_dims;
    auto rhs_adjustment = rhs_batch_dims.size();
    for (int64_t i = 0; i < batch_adjusted_rhs_contracting_dims.size(); ++i) {
      batch_adjusted_rhs_contracting_dims[i] -= rhs_adjustment;
    }
    if (batch_adjusted_rhs_contracting_dims != std::vector<int64_t>({0})) {
      // TODO(mvoz): The transpose rhs case is also handled in the transpose_rhs
      // branch of apply_vector_layout for now we will fold this into here
      // later. This is due to legacy API reasons. See documentation
      // on the op for more details.
      rhs_transposition = transposition_fn;
    }
    auto output_dim_order = dimension_numbers->getOutputDimOrder();
    if (output_dim_order.size() % 2 != 0) {
      op->emitOpError("Illegal: output dim order must be of size 2");
      return failure();
    }
    if (batch_size.has_value()) {
      if (output_dim_order[0] != 0 || output_dim_order[1] != 0) {
        op->emitOpError(
            "Not implemented: Output with batch size must be the lhs 0 idx for "
            "now.");
        return failure();
      }
    }
    std::vector<int64_t> batch_adjusted_output_dim_order;
    batch_adjusted_output_dim_order.reserve(output_dim_order.size());
    for (int dim_pos = 0; dim_pos < output_dim_order.size(); dim_pos += 2) {
      auto idx = output_dim_order[dim_pos];
      if (idx != 0 && idx != 1) {
        op->emitOpError("Illegal: output dim order must be 0 or 1");
        return failure();
      }
      auto is_lhs = (idx == 0);
      auto lhs_or_rhs_dim = output_dim_order[dim_pos + 1];
      if (is_lhs) {
        lhs_or_rhs_dim -= lhs_adjustment;
      } else {
        lhs_or_rhs_dim -= rhs_adjustment;
      }
      batch_adjusted_output_dim_order.emplace_back(idx);
      batch_adjusted_output_dim_order.emplace_back(lhs_or_rhs_dim);
    }
    if (batch_adjusted_output_dim_order != std::vector<int64_t>({0, 0, 1, 1})) {
      op->emitOpError(
          "Not Implemented: Only vanilla mkn output dim order is supported");
      return failure();
    }
  }

  auto extsi_sitofp = [&builder, &op](TypedValue<VectorType> element) {
    const VectorType ty = element.getType();
    auto shape = ty.getShape();
    CHECK(ty.getElementType().isInteger());
    TypedValue<VectorType> ext_ele;
    if (ty.getElementType().getIntOrFloatBitWidth() == 32) {
      ext_ele = element;
    } else {
      ext_ele = cast<TypedValue<VectorType>>(
          builder
              .create<arith::ExtSIOp>(
                  VectorType::get(shape, builder.getI32Type()), element)
              .getResult());
    }
    // TODO(mvoz): Go to bf16 when hardware supported, requires adding support
    // for 16 bitwidth in extsiop in infer/apply.
    auto ele_as_fp = builder.create<arith::SIToFPOp>(
        op.getLoc(), VectorType::get(shape, builder.getF32Type()), ext_ele);
    return ele_as_fp;
  };

  if (lhs_element_type != rhs_element_type) {
    if (lhs_element_type.isInteger() && rhs_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/int matmul.
      op->emitOpError("Mix int/int - NYI");
      return failure();
    }
    if (acc_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/float matmul with int acc.
      // Should be pretty straightforward.
      op->emitOpError("acc is int in mixed matmul. Expected float.");
      return failure();
    }
    if (lhs_element_type.isInteger()) {
      auto float_lhs = extsi_sitofp(lhs);
      op->setOperand(0, float_lhs);
      lhs = op.getLhs();
    }
    if (rhs_element_type.isInteger()) {
      auto float_rhs = extsi_sitofp(rhs);
      op->setOperand(1, float_rhs);
      rhs = op.getRhs();
    }
  }
  // TODO(mvoz): Add more invariants.
  if (acc_element_type.isInteger()) {
    if (!op.getLhs().getType().getElementType().isInteger()) {
      op->emitOpError("int acc with float lhs. Expected int lhs.");
      return failure();
    }
    if (!op.getRhs().getType().getElementType().isInteger()) {
      op->emitOpError("int acc with float rhs. Expected int rhs.");
      return failure();
    }
  } else {
    if (op.getLhs().getType().getElementType().isInteger()) {
      op->emitOpError("float acc with int lhs. Expected float lhs.");
      return failure();
    }
    if (op.getRhs().getType().getElementType().isInteger()) {
      op->emitOpError("float acc with int rhs. Expected float rhs.");
      return failure();
    }
  }

  auto transpose_applied_matmul = [&](auto lhs, auto rhs, auto acc) {
    auto precision_attr = op.getPrecisionAttr();

    if (lhs_transposition.has_value()) {
      lhs = lhs_transposition.value()(lhs);
    }
    // if rhs_tranposition is set, we ignore the legacy transpose_rhs attribute.
    // This is a rather annoying intersection of logic, which we hope to clean
    // up soon, with the aim of removing the transpose_rhs attribute entirely.
    if (rhs_transposition.has_value()) {
      rhs = rhs_transposition.value()(rhs);
    }

    auto ddn = default_dimension_numbers(builder, lhs_transposition.has_value(),
                                         rhs_transposition.has_value());
    auto matmul_res = builder.create<tpu::MatmulOp>(
        op.getLoc(), acc.getType(), lhs, rhs, acc,
        /*transpose_lhs=*/false,
        /*transpose_rhs=*/false, precision_attr, ddn);
    return matmul_res;
  };

  // If we have a batch_size, we want to slice rhs and lhs [:batch_size],
  // and then do O[i] = A[i] @ B[i]
  // Produce an output shape of [batch_size, m, n]
  if (batch_size.has_value()) {
    std::vector<Value> outputs;

    for (int64_t i = 0; i < batch_size; ++i) {
      auto lhs_shape = lhs_ty.getShape();
      SmallVector<int64_t> slice_shape(lhs_shape);
      slice_shape[0] = i;

      auto sliced_lhs = builder.create<vector::ExtractOp>(op.getLoc(), lhs,
                                                          ArrayRef<int64_t>{i});
      auto sliced_rhs = builder.create<vector::ExtractOp>(op.getLoc(), rhs,
                                                          ArrayRef<int64_t>{i});

      auto sliced_acc = builder.create<vector::ExtractOp>(op.getLoc(), acc,
                                                          ArrayRef<int64_t>{i});

      auto matmul_res = transpose_applied_matmul(sliced_lhs.getResult(),
                                                 sliced_rhs.getResult(),
                                                 sliced_acc.getResult());
      auto res_ty = matmul_res.getType().cast<VectorType>();
      auto res_shape = res_ty.getShape();
      // reshape to 1x[prior_shape]
      auto reshape_shape = llvm::to_vector(res_shape);
      reshape_shape.insert(reshape_shape.begin(), 1);
      auto shape_cast = builder.create<vector::ShapeCastOp>(
          op.getLoc(), VectorType::get(reshape_shape, res_ty.getElementType()),
          matmul_res);
      outputs.push_back(shape_cast);
    }
    auto output = builder
                      .create<tpu::ConcatenateOp>(op.getLoc(), acc_ty, outputs,
                                                  /*dimension=*/0)
                      .getResult();
    op.replaceAllUsesWith(output);
    op.erase();
  } else if (lhs_transposition.has_value() || rhs_transposition.has_value()) {
    auto matmul_res = transpose_applied_matmul(lhs, rhs, acc).getResult();
    op.replaceAllUsesWith(matmul_res);
    op.erase();
  }
  return success();
};

LogicalResult canonicalize_elementwise(int hardware_generation_,
                                       Operation &op) {
  OpBuilder builder(&op);
  auto operands = op.getOperands();
  auto res_ty = dyn_cast<VectorType>(op.getResult(0).getType());
  if (op.getNumResults() != 1) {
    op.emitOpError("Invariant violated: Unexpected number of results");
    return failure();
  }
  if (!res_ty) {
    // scalar
    // TODO(mvoz): Add canonicalization and invariants for scalar elementwise
    // ops.
    return success();
  }
  auto shape = res_ty.getShape();
  std::vector<Value> new_operands;
  new_operands.reserve(operands.size());

  bool should_rewrite_op = false;
  auto target_f32_ty = VectorType::get(shape, builder.getF32Type());
  for (int i = 0; i < operands.size(); ++i) {
    auto operand = operands[i];
    auto ty = dyn_cast<VectorType>(operand.getType());
    if (ty) {
      if (ty.getShape() != shape) {
        // Should already be checked my MLIR verification, but let's be safe.
        op.emitOpError("Mismatched shapes in elementwise op.");
        return failure();
      }
      auto element_type = ty.getElementType();
      // PowFOp and DivFOp do not seem to be supported in bf16 on later
      // hardware.
      bool needs_cast = hardware_generation_ <= 5 || isa<math::PowFOp>(op) ||
                        isa<arith::DivFOp>(op);
      if (needs_cast && element_type.isBF16()) {
        auto target_f32 =
            builder.create<arith::ExtFOp>(op.getLoc(), target_f32_ty, operand)
                .getResult();
        should_rewrite_op = true;
        new_operands.push_back(target_f32);
      } else {
        new_operands.push_back(operand);
      }
    } else {
      // Should already be checked my MLIR verification, but let's be safe.
      op.emitOpError("MLIR unsupported - mix scalar and vec elementwise ops");
      return failure();
    }
  }
  if (should_rewrite_op) {
    auto result_ty = dyn_cast<VectorType>(op.getResult(0).getType());
    if (!result_ty) {
      op.emitOpError("Not implemented: Unexpected result type");
      return failure();
    }
    auto result_element_type = result_ty.getElementType();
    if (!result_element_type.isF32() && !result_element_type.isBF16()) {
      op.emitOpError("Not implemented: Unexpected result element type");
      return failure();
    }
    // Do the new op in f32, then truncate to the original element type.
    auto new_op = builder.create(op.getLoc(), op.getName().getIdentifier(),
                                 new_operands, target_f32_ty);
    new_op = builder.create<arith::TruncFOp>(op.getLoc(), res_ty,
                                             new_op->getResult(0));
    op.replaceAllUsesWith(new_op);
    op.erase();
  }
  return success();
}

LogicalResult canonicalize_multi_dim_reduction(int hardware_generation,
                                               Operation &operation) {
  ImplicitLocOpBuilder builder(operation.getLoc(), &operation);
  auto op = cast<vector::MultiDimReductionOp>(operation);
  auto source_ty = op.getSourceVectorType();
  auto result_ty = dyn_cast<VectorType>(op.getDestType());
  if (!result_ty) {
    return op->emitOpError() << "Only vector reductions supported";
  }

  auto element_type = source_ty.getElementType();
  if (element_type.isF32()) {
    return success();
  } else if (element_type.isBF16()) {
    bool reduces_sublanes = false;
    for (int64_t dim : op.getReductionDims()) {
      if (dim == source_ty.getRank() - 2) {
        reduces_sublanes = true;
      }
    }
    if (hardware_generation <= 5 || reduces_sublanes) {
      auto new_source = builder.create<arith::ExtFOp>(
          VectorType::get(source_ty.getShape(), builder.getF32Type()),
          op.getSource());

      auto result_ty_f32 =
          VectorType::get(result_ty.getShape(), builder.getF32Type());
      auto acc_ext = builder.create<arith::ExtFOp>(result_ty_f32, op.getAcc());
      Value new_acc = acc_ext.getResult();
      // Try to constant fold.
      if (auto const_acc = op.getAcc().getDefiningOp<arith::ConstantOp>()) {
        auto result =
            acc_ext.fold(arith::ExtFOp::FoldAdaptor(const_acc.getValue()));
        if (!result.isNull() && result.is<Attribute>()) {
          acc_ext->erase();
          new_acc = builder.create<arith::ConstantOp>(
              op.getLoc(), result_ty_f32,
              cast<TypedAttr>(result.get<Attribute>()));
        }
      }
      auto new_op = builder.create<vector::MultiDimReductionOp>(
          op.getLoc(), new_acc.getType(), op.getKindAttr(), new_source, new_acc,
          DenseI64ArrayAttr::get(builder.getContext(), op.getReductionDims()));
      auto new_result = builder.create<arith::TruncFOp>(op.getLoc(), result_ty,
                                                        new_op.getResult());
      op.replaceAllUsesWith(new_result.getResult());
      op.erase();
    }
    return success();
  }
  return failure();
}

LogicalResult canonicalize_matmul(int hardware_generation, Operation &op) {
  auto matmul_op = dyn_cast<tpu::MatmulOp>(op);
  if (!matmul_op) {
    op.emitOpError("Invariant violated: Not a matmul");
    return failure();
  }
  return tpu_matmul_rule(matmul_op);
};

LogicalResult canonicalize_contraction(int hardware_generation, Operation &op) {
  auto contraction_op = dyn_cast<vector::ContractionOp>(op);
  if (!contraction_op) {
    op.emitOpError("Invariant violated: Not a contraction");
    return failure();
  }
  // Rewrite the contraction as a matmul
  auto lhs = contraction_op.getLhs();
  auto rhs = contraction_op.getRhs();
  auto acc = contraction_op.getAcc();
  VectorType acc_ty;
  if (!(acc_ty = dyn_cast<VectorType>(acc.getType()))) {
    contraction_op->emitOpError("Not implemented: acc must be a vector");
    return failure();
  }

  if (contraction_op.getKind() != vector::CombiningKind::ADD) {
    contraction_op->emitOpError("Only ADD supported");
    return failure();
  }

  ImplicitLocOpBuilder builder(contraction_op->getLoc(),
                               contraction_op.getOperation());

  MLIRContext *const mlir_ctx = contraction_op->getContext();

  auto getMapAttr = [&](const unsigned first, const unsigned second) {
    return AffineMapAttr::get(AffineMap::get(
        3, 0,
        {getAffineDimExpr(first, mlir_ctx), getAffineDimExpr(second, mlir_ctx)},
        mlir_ctx));
  };

  const ArrayAttr matmul_indexing_maps = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(2, 1), getMapAttr(0, 1)});
  const ArrayAttr matmul_indexing_maps_transposed = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(1, 2), getMapAttr(0, 1)});
  const auto indexing_maps = contraction_op.getIndexingMaps();
  if (indexing_maps != matmul_indexing_maps &&
      indexing_maps != matmul_indexing_maps_transposed) {
    return contraction_op->emitOpError(
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
  if (contraction_op->getAttr("iterator_types") != matmul_iterator_types) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul iterator_types");
  }
  const tpu::ContractPrecisionAttr precision_attr =  // May be null
      contraction_op->getAttrOfType<tpu::ContractPrecisionAttr>("precision");

  const auto dot_dimension_numbers_attr =
      default_dimension_numbers(builder, false, transpose_rhs);

  auto matmul_op = builder.create<tpu::MatmulOp>(
      contraction_op->getLoc(), acc_ty, lhs, rhs, acc,
      /*transpose_lhs=*/false, transpose_rhs, precision_attr,
      dot_dimension_numbers_attr);
  contraction_op.replaceAllUsesWith(matmul_op.getResult());
  contraction_op.erase();
  auto result = tpu_matmul_rule(matmul_op);
  return result;
}

LogicalResult canonicalize_extract(int hardware_generation, Operation &raw_op) {
  auto op = dyn_cast<vector::ExtractOp>(raw_op);
  Type result_ty = op.getResult().getType();
  if (!isa<VectorType>(result_ty)) {
    bool is_supported = result_ty.isSignlessIntOrFloat() &&
                        result_ty.getIntOrFloatBitWidth() == 32;
    if (!is_supported) {
      return op.emitOpError(
          "Only 32-bit scalar vector.extracts supported. Cast your input to a "
          "32-bit type first.");
    }
  }
  return success();
}

LogicalResult canonicalize_select(int hardware_generation, Operation &raw_op) {
  auto op = dyn_cast<arith::SelectOp>(raw_op);
  if (!isa<VectorType>(op.getType()) ||
      isa<VectorType>(op.getCondition().getType())) {
    return success();
  }
  // Canonicalize `i1 ? v1 : v2` -> `broadcast(i1) ? v1 : v2`.
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto cond_ty = VectorType::get(cast<VectorType>(op.getType()).getShape(),
                                 op.getCondition().getType());
  auto cond = builder.create<vector::BroadcastOp>(cond_ty, op.getCondition());
  auto new_op = builder.create<arith::SelectOp>(
      op.getLoc(), cond, op.getTrueValue(), op.getFalseValue());
  op.replaceAllUsesWith(new_op.getResult());
  op.erase();
  return success();
}

using canonicalize_rule_type =
    std::function<LogicalResult(int hardware_generation, Operation &op)>;

const llvm::StringMap<canonicalize_rule_type> &rules() {
  static auto rules = new llvm::StringMap<canonicalize_rule_type>{
      {tpu::MatmulOp::getOperationName(), canonicalize_matmul},
      {vector::ContractionOp::getOperationName(), canonicalize_contraction},
      {vector::ContractionOp::getOperationName(), canonicalize_extract},
      {vector::MultiDimReductionOp::getOperationName(),
       canonicalize_multi_dim_reduction},
      {arith::SelectOp::getOperationName(), canonicalize_select}};
  return *rules;
}

const llvm::StringSet<> &elementwise_convertible_ops() {
  static auto ops = new llvm::StringSet<>{arith::MulFOp::getOperationName(),
                                          arith::DivFOp::getOperationName(),
                                          arith::AddFOp::getOperationName(),
                                          arith::SubFOp::getOperationName(),
                                          arith::MaximumFOp::getOperationName(),
                                          arith::MinimumFOp::getOperationName(),
                                          math::PowFOp::getOperationName()};
  return *ops;
}

class MosaicCanonicalizer {
 public:
  MosaicCanonicalizer(int hardware_generation)
      : hardware_generation_(hardware_generation) {}

  int hardware_generation_;

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
      if (canonicalizeOp(any_op).failed()) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult canonicalizeOp(Operation &any_op) {
    // We must iterate over the op first, because canonicalization can cause
    // us to .erase() an op, and accessing getRegions on it after is not sound.
    // Invariant - top level ops with regions may never be invalidated.
    for (Region &region : any_op.getRegions()) {
      for (Block &block : region) {
        if (canonicalizeBlock(block).failed()) {
          return failure();
        }
      }
    }
    if (elementwise_convertible_ops().contains(
            any_op.getName().getStringRef())) {
      return canonicalize_elementwise(hardware_generation_, any_op);
    }
    if (auto rule_it = rules().find(any_op.getName().getStringRef());
        rule_it != rules().end()) {
      const canonicalize_rule_type &rule = rule_it->getValue();
      return rule(hardware_generation_, any_op);
    }
    return success();
  }
};

struct CanonicalizeMosaicPass
    : public impl::CanonicalizeMosaicPassBase<CanonicalizeMosaicPass> {
  CanonicalizeMosaicPass(int hardware_generation) {
    this->hardware_generation = hardware_generation;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MosaicCanonicalizer vlc(hardware_generation);
    if (vlc.canonicalize(func).failed()) {
      signalPassFailure();
    }
  };
};

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass(
    int hardware_generation) {
  return std::make_unique<CanonicalizeMosaicPass>(hardware_generation);
}

}  // namespace mlir::tpu
