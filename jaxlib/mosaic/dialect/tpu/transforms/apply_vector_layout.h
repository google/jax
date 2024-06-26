#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_APPLY_VECTOR_LAYOUT_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_APPLY_VECTOR_LAYOUT_H_

#include <array>
#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/array.h"

namespace mlir::tpu {

struct RewriteContext {
  func::FuncOp func;
  // TODO(tlongeri): target_shape should be determined from hardware_generation
  const int hardware_generation;
  const std::array<int64_t, 2> target_shape = {8, 128};
  const std::array<int64_t, 2> mxu_shape = {128, 128};
  const int max_sublanes_in_scratch = 0;

  MLIRContext *getMLIRContext() { return func.getContext(); }
};

// TODO(tlongeri): Remove default values for use_implicit_shape.
RollVectorsOp assemble(OpBuilder &builder, VectorType vty,
                       const VectorLayout &layout,
                       const xla::Array<Value> &vals,
                       std::array<int64_t, 2> target_shape,
                       bool use_implicit_shape = false);
FailureOr<xla::Array<Value>> disassemble(OpBuilder &builder,
                                         const VectorLayout &layout,
                                         TypedValue<VectorType> val,
                                         std::array<int64_t, 2> target_shape,
                                         bool use_implicit_shape = false);

// Rewrites the operation according to its layout annotations.
//
// Args:
//   ctx: The context used for rewriting.
//   op: An MLIR operation to be rewritten.
//
// A valid op is expected to have a layout_in attribute unless it has no
// operands. The layout_in attribute must fulfill the following:
//   - All vector operands originate from an operation (not a BlockArgument)
//   and
//     have a valid layout (Layout1D or Layout2D)
//   - All non-vector operands must have NoLayout.
LogicalResult applyLayoutOp(RewriteContext &ctx, Operation &op);

// Changes the layout of a vector value.
//
// Arguments:
//   v: The value to relayout. Must be of type VectorType.
//   src: The current layout of v.
//   dst: The target layout of v.
//
// Returns:
//   A new MLIR vector value, laid out as requested by dst.
FailureOr<TypedValue<VectorType>> relayout(OpBuilder &builder,
                                           TypedValue<VectorType> v,
                                           VectorLayout src, VectorLayout dst,
                                           std::array<int64_t, 2> target_shape);

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_APPLY_VECTOR_LAYOUT_H_
