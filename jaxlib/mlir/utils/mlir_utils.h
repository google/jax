#ifndef JAX_JAXLIB_MLIR_UTILS_MLIR_UTILS_H_
#define JAX_JAXLIB_MLIR_UTILS_MLIR_UTILS_H_

// Helper functions and utilities to make MLIR less verbose, easier to hold.

#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <string>

#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/AffineExpr.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/AffineMap.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Builders.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/ArrayRef.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Value.h"

namespace jax {
namespace mlir {

struct MLIRIterationBound {
  std::string name;
  int64_t bound_size;
  ::mlir::tpu::DimensionSemantics dimension_semantics;
  ::mlir::AffineExpr ae;
};

class MLIRIterationContext {
 public:
  MLIRIterationContext(::mlir::ImplicitLocOpBuilder builder,
                       ::mlir::MLIRContext* context,
                       std::vector<MLIRIterationBound> iteration_bounds)
      : context_(context) {
    std::vector<int64_t> bound_sizes;
    bound_sizes.reserve(iteration_bounds.size());

    std::vector<::mlir::Attribute> dimension_semantics;
    dimension_semantics.reserve(iteration_bounds.size());

    affine_exprs_.reserve(iteration_bounds.size());

    int i = 0;
    for (const auto& bound : iteration_bounds) {
      name_to_iteration_bound_[bound.name] = i;
      dimension_semantics.emplace_back(::mlir::tpu::DimensionSemanticsAttr::get(
          context, bound.dimension_semantics));
      bound_sizes.emplace_back(bound.bound_size);
      affine_exprs_.emplace_back(bound.ae);
      i++;
    }

    iteration_bounds_ =
        builder.getDenseI64ArrayAttr(::llvm::ArrayRef(bound_sizes));
    dimension_semantics_ = builder.getArrayAttr(
        ::llvm::ArrayRef<::mlir::Attribute>(dimension_semantics));
  }

  template <typename... Names>
  ::mlir::AffineMapAttr getAffineMapAttr(Names... names) {
    std::vector<::mlir::AffineExpr> affine_exprs;
    for (const auto& name : {names...}) {
      // TODO(mvoz): ensure the name is there
      auto idx = name_to_iteration_bound_.at(name);
      auto val = affine_exprs_[idx];
      affine_exprs.push_back(affine_exprs_[idx]);
    }
    return ::mlir::AffineMapAttr::get(::mlir::AffineMap::get(
        /*dimCount=*/iteration_bounds_.size(),
        /*symbolCount=*/0, affine_exprs, context_));
  }

  ::mlir::DenseI64ArrayAttr getIterationBounds() const {
    return iteration_bounds_;
  }

  ::mlir::ArrayAttr getDimensionSemantics() const {
    return dimension_semantics_;
  }

 private:
  ::mlir::DenseI64ArrayAttr iteration_bounds_;
  ::mlir::ArrayAttr dimension_semantics_;
  std::unordered_map<std::string, int> name_to_iteration_bound_;
  ::mlir::MLIRContext* context_;

  ::mlir::SmallVector<::mlir::AffineExpr> affine_exprs_;
};

class MLIRHelper {
 public:
  MLIRHelper(::mlir::ImplicitLocOpBuilder builder) : builder_(builder) {}

  template <typename Op, typename L>
  ::mlir::scf::IfOp if_op(Op predicate, L mlir_if_then_block) {
    return builder_.create<::mlir::scf::IfOp>(predicate.getLoc(), predicate,
                                              mlir_if_then_block);
  }

  template <typename LT, typename RT>
  ::mlir::arith::AndIOp and_op(LT lhs, RT rhs) {
    return builder_.create<::mlir::arith::AndIOp>(lhs, rhs);
  }

  ::mlir::arith::SelectOp select(const ::mlir::Value& mask, const ::mlir::Value& lhs, const ::mlir::Value& rhs) {
    return builder_.create<::mlir::arith::SelectOp>(mask, lhs, rhs);
  }

  template <typename LT, typename RT, std::enable_if_t<!std::is_scalar_v<RT>>>
  ::mlir::arith::CmpIOp eq(const LT& lhs, const RT& rhs) {
    return builder_.create<::mlir::arith::CmpIOp>(
        ::mlir::arith::CmpIPredicate::eq, lhs, rhs);
  }

  template <typename LT>
  ::mlir::arith::CmpIOp eq(const LT& lhs, int rhs_scalar) {
    // TODO: Type check the scalar, route to correct mlir attr, not always index
    auto rhs = builder_.create<::mlir::arith::ConstantOp>(
        builder_.getIndexAttr(rhs_scalar));
    return builder_.create<::mlir::arith::CmpIOp>(
        ::mlir::arith::CmpIPredicate::eq, lhs, rhs);
  }

  ::mlir::arith::CmpIOp sge(const ::mlir::Value lhs, const ::mlir::Value rhs) {
    return builder_.create<::mlir::arith::CmpIOp>(
        ::mlir::arith::CmpIPredicate::sge, lhs, rhs);
  }

  ::mlir::arith::CmpIOp sle(const ::mlir::Value lhs, const ::mlir::Value rhs) {
    return builder_.create<::mlir::arith::CmpIOp>(
        ::mlir::arith::CmpIPredicate::sle, lhs, rhs);
  }

  ::mlir::arith::CmpIOp slt(const ::mlir::Value lhs, const ::mlir::Value rhs) {
    return builder_.create<::mlir::arith::CmpIOp>(
        ::mlir::arith::CmpIPredicate::slt, lhs, rhs);
  }

 private:
  ::mlir::ImplicitLocOpBuilder builder_;
};
}  // namespace mlir
}  // namespace jax

#endif  // JAX_JAXLIB_MLIR_UTILS_MLIR_UTILS_H_
