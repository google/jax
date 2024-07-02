#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/types/span.h"
#include "tsl/platform/statusor.h"

// TODO: Instead of CHECK_EQs, can we do something like TF_RET_CHECK but with
// MLIR diagnostics?
// e.g.
// #define MLIR_RET_CHECK_EQ(a, b, diagnostic) \
//   do { \
//     const auto a_ = a; \
//     const auto b_ = b; \
//     if (LLVM_UNLIKELY(a_ != b_)) { \
//       return diagnostic << "Check failed: " << #a << " != " << #b << "(" <<
//       a_  << " vs. " << b_ << ")"; \
//     } \
//   } while (false)

#define FAILUREOR_ASSIGN_OR_RETURN_IMPL(failureor, lhs, rhs) \
  auto failureor = rhs;                                      \
  if (failed(failureor)) {                                   \
    return failure();                                        \
  }                                                          \
  lhs = std::move(failureor).value();
#define FAILUREOR_ASSIGN_OR_RETURN(lhs, rhs) \
  FAILUREOR_ASSIGN_OR_RETURN_IMPL(           \
      TF_STATUS_MACROS_CONCAT_NAME(failureor, __COUNTER__), lhs, rhs)

namespace mlir::tpu {

template <bool adjust_bool = false>
FailureOr<int8_t> getTypeBitwidth(Type ty) {
  if (auto integer_ty = dyn_cast<IntegerType>(ty)) {
    const unsigned width = integer_ty.getWidth();
    if constexpr (adjust_bool) {
      // We store only one i1 per vreg element.
      return width == 1 ? 32 : width;
    } else {
      return width;
    }
  }
  if (auto f32_ty = dyn_cast<Float32Type>(ty)) {
    return 32;
  }
  if (auto bf16_ty = dyn_cast<BFloat16Type>(ty)) {
    return 16;
  }
  if (auto f8e5m2_ty = dyn_cast<Float8E5M2Type>(ty)) {
    return 8;
  }
  if (auto f8e4m3fn_ty = dyn_cast<Float8E4M3FNType>(ty)) {
    return 8;
  }
  return emitError(UnknownLoc::get(ty.getContext()), "Unsupported type: ")
         << ty;
}

template <typename T>
ArrayRef<std::remove_const_t<T>> toArrayRef(absl::Span<T> span) {
  return ArrayRef<std::remove_const_t<T>>(span.data(), span.size());
}
template <typename T, std::size_t N>
ArrayRef<std::remove_const_t<T>> toArrayRef(std::array<T, N> array) {
  return ArrayRef<std::remove_const_t<T>>(array.data(), array.size());
}

inline arith::ConstantOp IdxConst(int64_t idx, OpBuilder &builder,
                                  Location loc) {
  return builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                           builder.getIndexAttr(idx));
}

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_
