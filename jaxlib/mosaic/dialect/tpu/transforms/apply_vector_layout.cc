#include "jaxlib/mosaic/dialect/tpu/transforms/apply_vector_layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/transforms/infer_memref_layout.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"
#include "xla/array.h"
#include "xla/layout.h"
#include "xla/util.h"

// TODO(tlongeri): Prefer returning failure over CHECKs. In particular, be more
// consistent about this for layout null checks in rules.

namespace mlir::tpu {
// TODO(tlongeri): Maybe just roll our own multi-dimensional array instead of
// using XLA's? There's too much glue for going from/to ArrayRef.

#define GEN_PASS_DECL_APPLYVECTORLAYOUTPASS
#define GEN_PASS_DEF_APPLYVECTORLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

// TPU_ASSERT_* macros should be understood as an assert, i.e. use it to check
// things that should never happen. We prefer returning failure over a CHECK
// because it's easier to debug from Python (particularly from OSS where symbols
// are removed)
#define TPU_ASSERT_IMPL(stream, cond)                    \
  if (LLVM_UNLIKELY(!(cond))) {                          \
    (stream) << "Internal error: assert failed: " #cond; \
  }
#define TPU_ASSERT_CMP_IMPL(stream, lhs, rhs, cmp)                            \
  if (LLVM_UNLIKELY(!((lhs)cmp(rhs)))) {                                      \
    (stream) << "Internal error: assert failed: " #lhs " " #cmp " " #rhs " (" \
             << (lhs) << " vs. " << (rhs) << ")";                             \
    return failure();                                                         \
  }
#define TPU_ASSERT_OP(cond) TPU_ASSERT_IMPL(op.emitOpError(), cond)
#define TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, cmp) \
  TPU_ASSERT_CMP_IMPL(op.emitOpError(), lhs, rhs, cmp)
#define TPU_ASSERT_EQ_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, ==)
#define TPU_ASSERT_GE_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, >=)
#define TPU_ASSERT_GT_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, >)
#define TPU_ASSERT_LE_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, <=)
#define TPU_ASSERT_LT_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, <)
#define TPU_ASSERT_LOC(loc, cond) TPU_ASSERT_IMPL(mlir::emitError(loc), cond)
#define TPU_ASSERT_CMP_LOC_IMPL(loc, lhs, rhs, cmp) \
  TPU_ASSERT_CMP_IMPL(loc, lhs, rhs, cmp)
#define TPU_ASSERT_EQ_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, ==)
#define TPU_ASSERT_GE_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, >=)
#define TPU_ASSERT_GT_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, >)
#define TPU_ASSERT_LT_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, <)
#define TPU_ASSERT_LE_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, <=)

// The minimum bound required to rotate with scratch space. The bound refers to
// the number of VREGs on rotation dim. This number was concluded from some cost
// analysis for comparing different dynamic rotation implementations. If
// actual bound is greater than this, dynamic rotation with internal scratch
// space is more efficient.
// TODO(jevinjiang): need to update it based on the generation.
static constexpr int kMinBoundToRotateWithScratch = 27;

LogicalResult applyLayoutBlock(RewriteContext &ctx, Block &block);
namespace {

void moveAllRegions(Operation &src, Operation &dst) {
  for (auto [src_region, dst_region] :
       llvm::zip_equal(src.getRegions(), dst.getRegions())) {
    dst_region.takeBody(src_region);
  }
}
// Masks all values outside of bounds.
//
// Arguments:
//   value: A rank 2 MLIR vector to be masked.
//   bounds: A TargetTuple of slices specifying a rectangular subregion of value
//     that should be preserved during masking.
//   neutral: A scalar attribute specifying the value that will be inserted
//     for all values outside of specified bounds.
//
// Returns:
//   An MLIR value of the same type as the value argument, with all entries
//   outside of bounds replaced by neutral.
FailureOr<Value> maskOOB(RewriteContext &ctx, OpBuilder &builder,
                         TypedValue<VectorType> value,
                         const VRegDataBounds &bounds,
                         const TypedAttr neutral) {
  TPU_ASSERT_LOC(value.getLoc(),
                 llvm::equal(value.getType().getShape(), ctx.target_shape));
  if (bounds.isComplete(ctx.target_shape)) {
    return value;
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      TypedValue<VectorType> mask,
      bounds.getVectorMask(builder, value.getLoc(), ctx.hardware_generation,
                           ctx.target_shape));
  if (cast<IntegerType>(mask.getType().getElementType()).getWidth() != 1) {
    return emitError(value.getLoc(),
                     "Not implemented: Unsupported mask bitwidth");
  }
  auto neutral_vec_ty = VectorType::get(ctx.target_shape, neutral.getType());
  auto neutral_vec = builder.create<arith::ConstantOp>(
      value.getLoc(), neutral_vec_ty,
      DenseElementsAttr::get(neutral_vec_ty, neutral));
  return builder
      .create<arith::SelectOp>(value.getLoc(), mask, value, neutral_vec)
      .getResult();
}

// Get the address of pre-allocated internal scratch space with requested shape.
//
// Arguments:
//   shape: The shape of the requested scratch space.
//   elem_ty: The type of the elements in the requested scratch space.
//
// Returns:
//   A memref of the requested shape and type.
FailureOr<Value> getInternalScratch(RewriteContext &ctx, OpBuilder &builder,
                                    Location loc, ArrayRef<int64_t> shape,
                                    Type elem_ty) {
  if (shape.empty()) {
    return failure();
  }
  if (shape.back() % ctx.target_shape[1] != 0) {
    return failure();
  }
  int sublane_count =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) /
      ctx.target_shape[1];
  if (sublane_count > ctx.max_sublanes_in_scratch) {
    return failure();
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      MemRefType scratch_ref_ty,
      inferMemref(MemRefType::get(shape, elem_ty), ctx.hardware_generation));
  return builder.create<tpu::GetInternalScratchOp>(loc, scratch_ref_ty)
      .getResult();
}

// Models Numpy's np.repeat, repeating each element `repeats` times along the
// specified axis. For example, if `src` is [1, 2], `axis` is 0 and `repeats` is
// 3, this will return [1, 1, 1, 2, 2, 2].
xla::Array<Value> repeat(const xla::Array<Value> &src, const int repeats,
                         const int64_t axis) {
  SmallVector<int64_t> dims(toArrayRef(src.dimensions()));
  dims[axis] *= repeats;
  xla::Array<Value> res(dims);
  src.Each([&](absl::Span<const int64_t> idx, const Value v) {
    SmallVector<int64_t> res_idx(toArrayRef(idx));
    res_idx[axis] *= repeats;
    for (int i = 0; i < repeats; ++i) {
      res(res_idx) = v;
      ++res_idx[axis];
    }
  });
  return res;
}

// Models Numpy's np.concatenate
xla::Array<Value> concatenate(const ArrayRef<xla::Array<Value>> arrays,
                              const int64_t axis) {
  CHECK(!arrays.empty());
  SmallVector<int64_t> dims(toArrayRef(arrays[0].dimensions()));
  CHECK(0 <= axis && axis < dims.size());
  for (size_t i = 1; i < arrays.size(); ++i) {
    CHECK_EQ(arrays[i].num_dimensions(), arrays[0].num_dimensions());
    for (size_t j = 0; j < arrays[i].num_dimensions(); ++j) {
      if (j != axis) {
        CHECK_EQ(arrays[i].dim(j), arrays[0].dim(j));
      }
    }
    dims[axis] += arrays[i].dim(axis);
  }
  xla::Array<Value> res(dims);
  int64_t offset = 0;
  for (xla::Array<Value> const& arr : arrays) {
    arr.Each([&](const absl::Span<const int64_t> idx, const Value v) {
      SmallVector<int64_t> res_idx(toArrayRef(idx));
      res_idx[axis] += offset;
      res(res_idx) = v;
    });
    offset += arr.dim(axis);
  }
  return res;
}

SmallVector<xla::Array<Value>> split(const xla::Array<Value> &vregs, int axis) {
  CHECK(axis >= 0 && axis < vregs.num_dimensions());
  SmallVector<xla::Array<Value>> chunks;
  chunks.reserve(vregs.dim(axis));
  SmallVector<int64_t> starts(vregs.num_dimensions(), 0);
  SmallVector<int64_t> limits(vregs.dimensions().begin(),
                              vregs.dimensions().end());
  for (int64_t i = 0; i < vregs.dim(axis); ++i) {
    starts[axis] = i;
    limits[axis] = i + 1;
    chunks.push_back(vregs.Slice(starts, limits));
  }
  return chunks;
};

template <typename T>
ArrayRef<T> XlaArrayToFlatArrayRef(xla::Array<T> xla_array) {
  return ArrayRef<T>(xla_array.data(), xla_array.num_elements());
}

template <typename T, typename Range>
xla::Array<T> XlaArrayFromShapeAndValues(ArrayRef<int64_t> sizes, Range vals) {
  // TODO(tlongeri): is there no way to avoid default initialization in the
  // constructor?
  xla::Array<T> arr(sizes);
  arr.SetValues(vals);
  return arr;
}

bool incrementSliceIndex(const MutableArrayRef<int64_t> idx,
                         const absl::Span<const int64_t> starts,
                         const absl::Span<const int64_t> limits) {
  const int64_t nd = idx.size();
  CHECK_EQ(nd, starts.size());
  CHECK_EQ(nd, limits.size());
  for (int64_t i = nd - 1; i >= 0; --i) {
    ++idx[i];
    if (idx[i] < limits[i]) {
      return true;
    }
    idx[i] = starts[i];
  }
  return false;
}

bool incrementIndex(const MutableArrayRef<int64_t> idx,
                    const absl::Span<const int64_t> limits) {
  const int64_t nd = idx.size();
  CHECK_EQ(nd, limits.size());
  for (int64_t i = nd - 1; i >= 0; --i) {
    ++idx[i];
    if (idx[i] < limits[i]) {
      return true;
    }
    idx[i] = 0;
  }
  return false;
}

bool sliceIsEmpty(const absl::Span<const int64_t> starts,
                  const absl::Span<const int64_t> limits) {
  for (auto [s, l] : llvm::zip_equal(starts, limits)) {
    CHECK_LE(s, l);
    if (s == l) {
      return true;
    }
  }
  return false;
}

// An alternative to xla::Array::UpdateSlice that takes a single value
template <typename T>
void updateSlice(xla::Array<T> &arr, const T &value,
                 const absl::Span<const int64_t> starts,
                 const absl::Span<const int64_t> limits) {
  if (sliceIsEmpty(starts, limits)) {
    return;
  }
  SmallVector<int64_t> idx(toArrayRef(starts));
  do {
    arr(idx) = value;
  } while (incrementSliceIndex(idx, starts, limits));
}

// An alternative to xla::Array::UpdateSlice that takes a range of data
template <typename T, typename Range>
void updateSliceFromRange(xla::Array<T> &arr, Range data,
                          const absl::Span<const int64_t> starts,
                          const absl::Span<const int64_t> limits) {
  if (sliceIsEmpty(starts, limits)) {
    return;
  }
  SmallVector<int64_t> idx(toArrayRef(starts));
  auto in_bounds = [&] {
    for (int64_t i = 0; i < idx.size(); ++i) {
      if (idx[i] >= arr.dim(i)) {
        return false;
      }
    }
    return true;
  };
  auto data_it = data.begin();
  do {
    if (in_bounds()) {
      arr(idx) = *data_it;
    }
    ++data_it;
  } while (incrementSliceIndex(idx, starts, limits));
  CHECK(data_it == data.end());
}

FailureOr<TypedAttr> getZeroIntOrFloatAttr(Type ty) {
  if (isa<FloatType>(ty)) {
    return TypedAttr(FloatAttr::get(ty, 0));
  }
  if (isa<IntegerType>(ty)) {
    return TypedAttr(IntegerAttr::get(ty, 0));
  }
  return emitError(UnknownLoc::get(ty.getContext()), "Not implemented: ") << ty;
}

FailureOr<int64_t> getIntConst(Value v, bool silent = false) {
  if (auto constant_op = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer_attr = dyn_cast<IntegerAttr>(constant_op.getValue())) {
      return integer_attr.getValue().getSExtValue();
    }
  }
  if (silent) {
    return failure();
  }
  return emitError(v.getLoc(), "Expected an integer constant");
}

FailureOr<SmallVector<int64_t>> getIntConstsFromOperandRange(
    ValueRange vals, bool silent = false) {
  SmallVector<int64_t> res(vals.size());
  for (int i = 0; i < vals.size(); ++i) {
    FAILUREOR_ASSIGN_OR_RETURN(res[i], getIntConst(vals[i], silent));
  }
  return res;
}

FailureOr<std::pair<Value, SmallVector<int64_t>>> sliceRef(
    ImplicitLocOpBuilder &builder, TypedValue<MemRefType> base_ref,
    ArrayRef<int64_t> slice_shape, ValueRange indices,
    ArrayRef<int64_t> tiling) {
  IntegerType i32 = builder.getI32Type();
  MemRefType ref_ty = base_ref.getType();

  // MemRefSliceOp only allows tile-aligned slices. We pad the shape up
  // accordingly with the padding. We don't include the static tiled indices
  // in the slice when they can be arbitrary. But we do include dynamic tiled
  // indices under the condition that they are divisible by the tile size.
  SmallVector<int64_t> pad_slice_shape(slice_shape);
  TPU_ASSERT_LE_LOC(builder.getLoc(), tiling.size(), slice_shape.size());
  for (int i = 1; i <= tiling.size(); ++i) {
    auto &dim = *(pad_slice_shape.end() - i);
    dim = xla::RoundUpTo(dim, *(tiling.end() - i));
  }

  SmallVector<Value> slice_base_indices;
  slice_base_indices.reserve(ref_ty.getRank());
  for (auto idx : indices.drop_back(tiling.size())) {
    slice_base_indices.push_back(builder.create<arith::IndexCastOp>(i32, idx));
  }

  Value c0 = nullptr;
  SmallVector<int64_t> indices_within_slice(indices.size() - tiling.size(), 0);
  for (auto tiled_idx : indices.take_back(tiling.size())) {
    if (auto cst = getIntConst(tiled_idx, /*silent=*/true); succeeded(cst)) {
      indices_within_slice.push_back(*cst);
      if (!c0) {
        c0 = builder.create<arith::ConstantOp>(i32,
                                               builder.getI32IntegerAttr(0));
      }
      slice_base_indices.push_back(c0);
    } else {
      indices_within_slice.push_back(0);
      // TODO: Check divisibility!
      slice_base_indices.push_back(
          builder.create<arith::IndexCastOp>(i32, tiled_idx));
    }
  }

  // TODO(apaszke): Allow tile-aligned dynamic slicing on tiled dimensions.
  Value sliced_ref = builder.create<tpu::MemRefSliceOp>(
      MemRefType::get(pad_slice_shape, ref_ty.getElementType(),
                      ref_ty.getLayout(), ref_ty.getMemorySpace()),
      base_ref, slice_base_indices, /*dynamic_sizes=*/ValueRange());

  return std::make_pair(sliced_ref, indices_within_slice);
}

// Returns the first-level tiling of a (packed and tiled) memref value.
FailureOr<std::array<int64_t, 2>> getMemRefTiling(
    TypedValue<MemRefType> value, const std::array<int64_t, 2> target_shape) {
  if (auto erase_layout_op =
          dyn_cast_if_present<EraseLayoutOp>(value.getDefiningOp())) {
    value = erase_layout_op.getOperand();
  }
  const MemRefType memref_ty = value.getType();
  const auto mem_layout = dyn_cast<TiledLayoutAttr>(memref_ty.getLayout());
  if (mem_layout == nullptr) {
    return emitError(value.getLoc(), "Expected a tiled memref");
  }
  FAILUREOR_ASSIGN_OR_RETURN(int8_t bitwidth,
                             getTypeBitwidth(memref_ty.getElementType()));
  const int packing = 32 / bitwidth;
  const ArrayRef<xla::Tile> tiles = mem_layout.getTiles();
  const xla::Tile &first_tile = tiles.front();
  if (first_tile.dimensions().size() == 1) {
    const int64_t tile_size = first_tile.dimension(0);
    if (tile_size % (target_shape[1] * packing) != 0) {
      return emitError(value.getLoc(), "Not implemented");
    }
    if (bitwidth == 32) {
      if (tiles.size() > 1) {
        return emitError(value.getLoc(), "Not implemented");
      }
    } else if (bitwidth < 32) {
      if (tiles.drop_front() !=
          ArrayRef<xla::Tile>{xla::Tile({target_shape[1]}),
                              xla::Tile({packing, 1})}) {
        return emitError(value.getLoc(), "Not implemented");
      }
    }
    return std::array<int64_t, 2>{1, tile_size};
  }
  if (first_tile.dimensions().size() == 2) {
    if (bitwidth == 32) {
      if (tiles.size() > 1) {
        return emitError(value.getLoc(), "Not implemented");
      }
      return std::array<int64_t, 2>{first_tile.dimension(0),
                                    first_tile.dimension(1)};
    }
    if (bitwidth < 32) {
      if (tiles.size() != 2 || tiles[1] != xla::Tile({packing, 1})) {
        return emitError(value.getLoc(), "Not implemented");
      }
      return std::array<int64_t, 2>{first_tile.dimension(0),
                                    first_tile.dimension(1)};
    }
  }
  return emitError(value.getLoc(), "Not implemented");
}

// Hoist a vector constant as an additional argument of the function.
FailureOr<BlockArgument> appendConstant(RewriteContext &ctx,
                                        DenseElementsAttr value) {
  MLIRContext *mlir_ctx = ctx.func.getContext();
  Block &entry_block = ctx.func.getBody().front();
  auto value_ty = cast<VectorType>(value.getType());
  if (value_ty.getElementType().getIntOrFloatBitWidth() != 32) {
    return ctx.func.emitOpError(
        "Not implemented: Only 32-bit constants supported");
  }
  if (ctx.func->getAttr("scratch_operands")) {
    return ctx.func.emitOpError(
        "Not implemented: function has scratch_operands");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      MemRefType arg_type,
      inferMemref(
          MemRefType::get(value_ty.getShape(), value_ty.getElementType()),
          ctx.hardware_generation));
  const BlockArgument argument =
      entry_block.insertArgument(entry_block.getNumArguments() - 1, arg_type,
                                 UnknownLoc::get(ctx.getMLIRContext()));
  const FunctionType func_ty = ctx.func.getFunctionType();
  // Adjust the function type.
  SmallVector<Type> new_arg_tys(func_ty.getInputs());
  new_arg_tys.insert(new_arg_tys.begin() + (new_arg_tys.size() - 1), arg_type);
  const auto new_func_ty =
      FunctionType::get(mlir_ctx, new_arg_tys, func_ty.getResults());
  ctx.func.setFunctionType(new_func_ty);
  // Adjust the constants attribute.
  if (auto prev_cst = ctx.func->getAttrOfType<ArrayAttr>("vector_constants")) {
    SmallVector<Attribute> vector_constants(prev_cst.getValue());
    vector_constants.push_back(value);
    ctx.func->setAttr("vector_constants",
                      ArrayAttr::get(ctx.func.getContext(), vector_constants));
  } else {
    ctx.func->setAttr("vector_constants",
                      ArrayAttr::get(ctx.func.getContext(), value));
  }
  // Adjust window params for the extra operand.
  if (auto window_params =
          ctx.func->getAttrOfType<ArrayAttr>("window_params")) {
    const auto iteration_bounds =
        ctx.func->getAttrOfType<DenseI64ArrayAttr>("iteration_bounds");
    TPU_ASSERT_LOC(UnknownLoc::get(mlir_ctx), iteration_bounds);
    const int64_t iteration_rank = iteration_bounds.getSize();
    const SmallVector<AffineExpr> zeros(
        iteration_rank, getAffineConstantExpr(0, ctx.func.getContext()));
    const auto transform_indices =
        AffineMap::get(iteration_rank, 0, zeros, ctx.func.getContext());
    const auto new_param = DictionaryAttr::get(
        ctx.func.getContext(),
        NamedAttribute(
            StringAttr::get(ctx.func.getContext(), "transform_indices"),
            AffineMapAttr::get(transform_indices)));
    SmallVector<Attribute> window_params_values(window_params.getValue());
    window_params_values.insert(window_params_values.end() - 1, new_param);
    ctx.func->setAttr("window_params", ArrayAttr::get(ctx.func.getContext(),
                                                      window_params_values));
  }
  return argument;
}

FailureOr<VectorType> getNativeVregOrVmaskTypeImpl(
    Type elem_ty, const int8_t bitwidth,
    const std::array<int64_t, 2> target_shape) {
  if (bitwidth == 32) {
    return VectorType::get(target_shape, elem_ty);
  }
  return VectorType::get({target_shape[0], target_shape[1], 32 / bitwidth},
                         elem_ty);
}

FailureOr<VectorType> getNativeVregOrVmaskType(
    Type elem_ty, const int8_t layout_bitwidth,
    const std::array<int64_t, 2> target_shape) {
  int8_t bitwidth = elem_ty.getIntOrFloatBitWidth();
  if (bitwidth == 1) {
    bitwidth = layout_bitwidth;
  } else {
    CHECK_EQ(bitwidth, layout_bitwidth);
  }
  return getNativeVregOrVmaskTypeImpl(elem_ty, bitwidth, target_shape);
}

FailureOr<VectorType> getNativeVregType(
    Type elem_ty, const std::array<int64_t, 2> target_shape) {
  return getNativeVregOrVmaskTypeImpl(elem_ty, elem_ty.getIntOrFloatBitWidth(),
                                      target_shape);
}

// Returns empty vector on null attribute
FailureOr<SmallVector<Layout>> getLayoutArrayFromAttr(const Attribute attr) {
  if (const auto array_attr = dyn_cast_if_present<ArrayAttr>(attr)) {
    SmallVector<Layout> out_layouts;
    out_layouts.reserve(array_attr.size());
    for (const Attribute a : array_attr) {
      if (auto layout_attr = dyn_cast_if_present<VectorLayoutAttr>(a)) {
        out_layouts.push_back(layout_attr.getLayout());
      } else {
        return failure();
      }
    }
    return out_layouts;
  }
  return SmallVector<Layout>{};
}

bool layoutIsValidForValue(const Layout &l, const Value v,
                           const std::array<int64_t, 2> target_shape) {
  // l must be non-null iff v is of vector type
  if (const auto vty = dyn_cast<VectorType>(v.getType())) {
    if (!l.has_value()) {
      return false;
    }

    // Vector type should have the same bitwidth as the layout, except for the
    // i1 special case, used for vmasks (see comment for VectorLayout class).
    if (!vty.getElementType().isIntOrFloat()) {
      return false;
    }
    const int8_t bitwidth = vty.getElementTypeBitWidth();
    if (bitwidth != l->bitwidth() && bitwidth != 1) {
      return false;
    }

    return l->isValid(target_shape) && l->layout_rank() <= vty.getRank();
  }
  return !l.has_value();
}

// TODO(tlongeri): Unify with infer_vector_layout.cc's getOutLayout.
FailureOr<SmallVector<Layout>> getOutLayouts(
    Operation &op, const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> out_layouts,
                             getLayoutArrayFromAttr(op.getAttr("out_layout")));
  if (out_layouts.size() != op.getNumResults()) {
    return op.emitOpError("out_layout size does not match number of results");
  }
  for (const auto [l, res] : llvm::zip_equal(out_layouts, op.getResults())) {
    if (!layoutIsValidForValue(l, res, target_shape)) {
      return op.emitOpError("Invalid output layout");
    }
  }
  return out_layouts;
}

FailureOr<SmallVector<Layout>> getInLayouts(
    Operation &op, const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> in_layouts,
                             getLayoutArrayFromAttr(op.getAttr("in_layout")));
  if (in_layouts.size() != op.getNumOperands()) {
    return op.emitOpError("in_layout size does not match number of operands");
  }
  for (const auto [l, operand] :
       llvm::zip_equal(in_layouts, op.getOperands())) {
    if (!layoutIsValidForValue(l, operand, target_shape)) {
      return op.emitOpError("Invalid input layout");
    }
  }
  return in_layouts;
}

LogicalResult elementwise_op_rule(RewriteContext &ctx, Operation &op,
                                  const ArrayRef<Layout> layouts_in,
                                  const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_OP(OpTrait::hasElementwiseMappableTraits(&op));
  if (op.getNumResults() != 1) {
    return op.emitError("Not implemented: Only ops with one result supported");
  }
  TPU_ASSERT_EQ_OP(layouts_in.size(), op.getNumOperands());
  TPU_ASSERT_GT_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  OpBuilder builder(&op);
  if (!(layouts_out.front().has_value() &&
        llvm::all_of(layouts_in,
                     [&](const Layout &l) { return l.has_value(); }))) {
    return op.emitOpError(
        "Not implemented: Null layout / non-vector operand in elementwise "
        "operation");
  }
  const auto out_ty = cast<VectorType>(op.getResult(0).getType());
  const VectorLayout &layout_out = *layouts_out.front();
  if (!llvm::all_of(layouts_in, [&](const Layout &l) {
        return l->generalizes(layout_out, out_ty.getShape(), ctx.target_shape);
      })) {
    return op.emitOpError(
        "Not implemented: Incompatible layouts in elementwise operation");
  }
  const unsigned num_operands = op.getNumOperands();
  SmallVector<xla::Array<Value>> in_vreg_arrays;
  in_vreg_arrays.reserve(num_operands);
  for (unsigned i = 0; i < num_operands; ++i) {
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> tile_array,
        disassemble(builder, *layouts_in[i],
                    cast<TypedValue<VectorType>>(op.getOperand(i)),
                    ctx.target_shape));
    in_vreg_arrays.emplace_back(std::move(tile_array));
  }

  FAILUREOR_ASSIGN_OR_RETURN(
      const VectorType out_vreg_ty,
      getNativeVregOrVmaskType(out_ty.getElementType(), layout_out.bitwidth(),
                               ctx.target_shape));

  NamedAttrList attributes(op.getAttrDictionary());
  attributes.erase("in_layout");
  attributes.erase("out_layout");

  // Note that we have to broadcast to handle replicate dimensions.
  SmallVector<int64_t> broadcasted_shape(
      toArrayRef(in_vreg_arrays[0].dimensions()));
  for (size_t i = 1; i < num_operands; ++i) {
    SmallVector<int64_t> new_broadcasted_shape;
    TPU_ASSERT_OP(OpTrait::util::getBroadcastedShape(
        broadcasted_shape, toArrayRef(in_vreg_arrays[i].dimensions()),
        new_broadcasted_shape));
    broadcasted_shape = std::move(new_broadcasted_shape);
  }
  TPU_ASSERT_OP(broadcasted_shape ==
                layout_out.tileArrayShape(out_ty.getShape(), ctx.target_shape));

  // TODO(tlongeri): Can we avoid initializing the array before filling values?
  xla::Array<Value> out_vreg_array(broadcasted_shape);
  out_vreg_array.Each([&](absl::Span<const int64_t> idx, Value *out_vreg) {
    SmallVector<Value> operands(num_operands);

    for (unsigned i = 0; i < num_operands; ++i) {
      // Handle indices for broadcasted dimensions
      SmallVector<int64_t> operand_idx(toArrayRef(idx));
      for (unsigned j = 0; j < idx.size(); ++j) {
        if (in_vreg_arrays[i].dim(j) == 1) {
          operand_idx[j] = 0;
        }
      }
      operands[i] = in_vreg_arrays[i](operand_idx);
    }
    Operation *vreg_op =
        builder.create(op.getLoc(), op.getName().getIdentifier(), operands,
                       out_vreg_ty, attributes.getAttrs());
    CHECK(vreg_op);
    CHECK_EQ(vreg_op->getNumResults(), 1);
    *out_vreg = vreg_op->getResult(0);
  });
  op.replaceAllUsesWith(assemble(builder, out_ty, layout_out,
                                 std::move(out_vreg_array), ctx.target_shape));
  op.erase();
  return success();
}

using rule_type = std::function<LogicalResult(
    RewriteContext &, Operation &, ArrayRef<Layout>, ArrayRef<Layout>)>;

template <typename OpTy>
LogicalResult ext_op_rule_impl(RewriteContext &ctx, OpTy op,
                               const VectorLayout &layout_in,
                               const VectorLayout &layout_out) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());
  const auto result_ty = cast<VectorType>(op.getResult().getType());
  auto source = cast<TypedValue<VectorType>>(op.getIn());
  const auto source_ty = source.getType();
  auto output_vregs_shape =
      layout_out.tileArrayShape(result_ty.getShape(), ctx.target_shape);
  if (layout_out.bitwidth() != 32) {
    return op.emitOpError(
        "Not implemented: Only extensions to 32-bit supported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> input_vregs,
      disassemble(builder, layout_in, source, ctx.target_shape));
  xla::Array<Value> output_vregs(output_vregs_shape);
  // TODO(jevinjiang): maybe just use tileArrayImplicitShape in disassemble?
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    input_vregs.Reshape(layout_in.tileArrayImplicitShape(source_ty.getShape(),
                                                         ctx.target_shape));
    output_vregs.Reshape(layout_out.tileArrayImplicitShape(result_ty.getShape(),
                                                           ctx.target_shape));
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const VectorType res_vreg_ty,
      getNativeVregType(result_ty.getElementType(), ctx.target_shape));
  if (layout_in.implicit_dim() != layout_out.implicit_dim()) {
    return op.emitOpError(
        "Not implemented: Change of implicit dim during the cast");
  }
  if (layout_in.offsets() != layout_out.offsets()) {
    return op.emitOpError("Not implemented: Change of offsets during the cast");
  }
  const int packing = layout_in.packing();
  if (layout_in.hasNativeTiling(ctx.target_shape) &&
      layout_out.hasNativeTiling(ctx.target_shape)) {
    output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      SmallVector<int64_t> input_vreg_idxs(toArrayRef(idxs));
      int64_t vreg_part = *(input_vreg_idxs.end() - 2) % packing;
      *(input_vreg_idxs.end() - 2) /= packing;
      *v = builder.create<UnpackSubelementsOp>(
          res_vreg_ty, input_vregs(input_vreg_idxs), vreg_part);
    });
  } else {
    if (layout_in.tiling() != layout_out.tiling()) {
      return op.emitOpError("Not implemented: Changing tiling during the cast");
    }
    auto tiling = layout_in.tiling();
    if (ctx.target_shape[0] % tiling[0] != 0 ||
        ctx.target_shape[1] != tiling[1]) {
      return op.emitOpError("Not implemented: tiling not supported");
    }
    output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      SmallVector<int64_t> input_vreg_idxs(toArrayRef(idxs));
      input_vreg_idxs.back() /= packing;
      const int64_t vreg_part = idxs.back() % packing;
      *v = builder.create<UnpackSubelementsOp>(
          res_vreg_ty, input_vregs(input_vreg_idxs), vreg_part);
    });
  }
  if (layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    output_vregs.Reshape(output_vregs_shape);
  }
  op.replaceAllUsesWith(assemble(builder, result_ty, layout_out,
                                 std::move(output_vregs), ctx.target_shape)
                            .getResult());
  op.erase();
  return success();
}

LogicalResult arith_extf_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto extf_op = cast<arith::ExtFOp>(op);
  if (layouts_in.front()->bitwidth() != 16 ||
      layouts_out.front()->bitwidth() != 32) {
    return op.emitOpError(
        "Not implemented: Only 16-bit to 32-bit conversion supported");
  }
  return ext_op_rule_impl(ctx, extf_op, *layouts_in.front(),
                          *layouts_out.front());
}

LogicalResult arith_extsi_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto extsi_op = cast<arith::ExtSIOp>(op);
  return ext_op_rule_impl(ctx, extsi_op, *layouts_in.front(),
                          *layouts_out.front());
}

template <typename OpTy>
LogicalResult trunc_op_rule_impl(RewriteContext &ctx, OpTy op,
                                 const VectorLayout &layout_in,
                                 const VectorLayout &layout_out) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());
  auto source = cast<TypedValue<VectorType>>(op.getIn());
  const auto source_ty = source.getType();
  auto result_ty = cast<VectorType>(op.getResult().getType());
  auto output_vregs_shape =
      layout_out.tileArrayShape(result_ty.getShape(), ctx.target_shape);
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> input_vregs,
      disassemble(builder, layout_in, source, ctx.target_shape));
  xla::Array<Value> output_vregs(output_vregs_shape);
  if (layout_in.bitwidth() != 32) {
    return op.emitOpError("Not implemented: Only 32-bit truncation supported");
  }
  if (layout_in.offsets() != layout_out.offsets()) {
    return op.emitOpError(
        "Not implemented: Change of offsets during the truncation");
  }
  if (layout_in.implicit_dim() != layout_out.implicit_dim()) {
    return op.emitOpError("Not implemented: Change of layout during the cast");
  }
  if (layout_in.tiling() != ctx.target_shape) {
    return op.emitOpError("Not implemented: Only (8,128) tiling supported");
  }
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    input_vregs.Reshape(layout_in.tileArrayImplicitShape(source_ty.getShape(),
                                                         ctx.target_shape));
    output_vregs.Reshape(layout_out.tileArrayImplicitShape(result_ty.getShape(),
                                                           ctx.target_shape));
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      VectorType res_vreg_ty,
      getNativeVregType(result_ty.getElementType(), ctx.target_shape));
  if (layout_out.tiling() == ctx.target_shape) {
    const int packing = layout_out.packing();
    output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      SmallVector<Value> parts;
      SmallVector<int64_t> idxs_local(toArrayRef(idxs));
      idxs_local.back() *= packing;
      for (int64_t i = 0; i < packing; ++i) {
        parts.push_back(input_vregs(idxs_local));
        // Pack any data lying around if OOB
        if (idxs_local.back() < input_vregs.dimensions().back() - 1) {
          ++idxs_local.back();
        }
      }
      *v = builder.create<PackSubelementsOp>(res_vreg_ty, parts,
                                             tpu::PackFormat::kCompressed);
    });
  } else if (layout_out.hasNativeTiling(ctx.target_shape)) {
    int packing = layout_out.packing();
    SmallVector<Value> parts;
    parts.reserve(packing);
    output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      CHECK_GE(idxs.size(), 2);
      SmallVector<int64_t> idxs_local(toArrayRef(idxs));
      idxs_local[idxs.size() - 2] *= packing;
      parts.push_back(input_vregs(idxs_local));
      idxs_local[idxs.size() - 2]++;
      while (parts.size() < packing) {
        if (*(idxs_local.end() - 2) < *(input_vregs.dimensions().end() - 2)) {
          parts.push_back(input_vregs(idxs_local));
          idxs_local[idxs.size() - 2]++;
        } else {
          // Once we run out of tiles, we can pick any one we like.
          parts.push_back(parts.back());
        }
      }
      *v = builder.create<PackSubelementsOp>(res_vreg_ty, parts,
                                             tpu::PackFormat::kCompressed);
      parts.clear();
    });
  } else {
    return op.emitOpError("Not implemented: unsupported output tiling");
  }
  if (layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    output_vregs.Reshape(output_vregs_shape);
  }
  op.replaceAllUsesWith(assemble(builder, result_ty, layout_out,
                                 std::move(output_vregs), ctx.target_shape)
                            .getResult());
  op.erase();
  return success();
}

LogicalResult arith_truncf_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto truncf_op = cast<arith::TruncFOp>(op);
  if (layouts_in.front()->bitwidth() != 32 ||
      (layouts_out.front()->bitwidth() != 16 &&
       layouts_out.front()->bitwidth() != 8)) {
    return op.emitOpError(
        "Not implemented: Only 32-bit to 16-or-8-bit conversion supported");
  }
  return trunc_op_rule_impl(ctx, truncf_op, *layouts_in.front(),
                            *layouts_out.front());
}

LogicalResult arith_trunci_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto trunci_op = cast<arith::TruncIOp>(op);
  return trunc_op_rule_impl(ctx, trunci_op, *layouts_in.front(),
                            *layouts_out.front());
}

LogicalResult func_return_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_OP(layouts_out.empty());
  for (const Layout &layout_in : layouts_in) {
    if (layout_in.has_value()) {
      return op.emitOpError("Vector-typed return values are not supported");
    }
  }
  return success();
}

LogicalResult scf_for_rule(RewriteContext &ctx, Operation &op,
                           const ArrayRef<Layout> layouts_in,
                           const ArrayRef<Layout> layouts_out) {
  scf::ForOp for_op = cast<scf::ForOp>(op);
  TPU_ASSERT_EQ_OP(layouts_in.size(), for_op->getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), for_op->getNumResults());
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<Layout> yield_in_layouts,
      getInLayouts(*for_op.getBody()->getTerminator(), ctx.target_shape));
  int out_idx = 0;
  for (auto [in_layout, yield_layout, out_layout, result] :
       llvm::zip_equal(layouts_in.drop_front(3), yield_in_layouts, layouts_out,
                       op.getResults())) {
    if (auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(in_layout.has_value());
      TPU_ASSERT_OP(yield_layout.has_value());
      TPU_ASSERT_OP(out_layout.has_value());
      if (in_layout.value() != yield_layout.value()) {
        return op.emitOpError(
                   "Not implemented: for loop input layout does not match with "
                   "yield layout ")
               << out_idx;
      }
      if (in_layout.value() != out_layout.value()) {
        return op.emitOpError(
                   "Not implemented: for loop input layout does not match with "
                   "out layout ")
               << out_idx;
      }
    } else {
      TPU_ASSERT_EQ_OP(in_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(yield_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(out_layout, kNoLayout);
    }
    ++out_idx;
  }

  if (failed(applyLayoutBlock(ctx, *for_op.getBody()))) {
    return failure();
  }

  if (op.getNumResults() == 0) {
    return success();
  }

  OpBuilder builder(&op);
  SmallVector<Value> unrolled_args;
  for (int i = 0; i < layouts_in.size(); ++i) {
    auto layout = layouts_in[i];
    auto operand = for_op.getOperand(i);
    if (i < 3) {
      if (layout.has_value()) {
        return op.emitOpError("Expected no layout for bounds and step");
      }
      continue;
    }
    if (auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand)) {
      if (!layout.has_value()) {
        return op.emitOpError("Expected layout for vector operand");
      }
      FAILUREOR_ASSIGN_OR_RETURN(
          const xla::Array<Value> tiles,
          disassemble(builder, *layout, vector_operand, ctx.target_shape));
      unrolled_args.append(tiles.begin(), tiles.end());
    } else {
      if (layout.has_value()) {
        return op.emitOpError("Expected no layout for scalar operand");
      }
      unrolled_args.push_back(operand);
    }
  }

  // Create a new scf::ForOp with unrolled args.
  auto new_op = builder.create<scf::ForOp>(
      for_op->getLoc(), for_op.getLowerBound(), for_op.getUpperBound(),
      for_op.getStep(), unrolled_args);

  int num_old_args = for_op.getBody()->getNumArguments();
  SmallVector<Location> locs(new_op.getBody()->getNumArguments(),
                             for_op.getLoc());
  for_op.getBody()->addArguments(TypeRange(new_op.getBody()->getArguments()),
                                 locs);
  builder.setInsertionPointToStart(for_op.getBody());
  auto arg_idx = num_old_args;
  // Block also has an induction variable that should have no layout,
  // which conveniently matches the in layouts.
  for (auto [old_arg, layout] : llvm::zip_equal(
           for_op.getBody()->getArguments().take_front(num_old_args),
           layouts_in.drop_front(2))) {
    if (const auto vty = dyn_cast<VectorType>(old_arg.getType())) {
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(arg_idx + num_vectors,
                       for_op.getBody()->getNumArguments());
      tiles.SetValues(llvm::make_range(
          for_op.getBody()->getArguments().begin() + arg_idx,
          for_op.getBody()->getArguments().begin() + arg_idx + num_vectors));
      arg_idx += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      old_arg.replaceUsesWithIf(rolled_op, [&](OpOperand &operand) {
        return operand.getOwner() != rolled_op;
      });
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      old_arg.replaceAllUsesWith(for_op.getBody()->getArgument(arg_idx));
      ++arg_idx;
    }
  }
  for_op.getBody()->eraseArguments(0, num_old_args);
  new_op.getRegion().takeBody(for_op.getRegion());

  // Roll the results back to the original shapes.
  builder.setInsertionPointAfter(new_op);
  int64_t res_idx = 0;
  SmallVector<Value> rolled_results;
  for (auto [result, layout] :
       llvm::zip_equal(for_op.getResults(), layouts_out)) {
    if (const auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(res_idx + num_vectors, new_op.getResults().size());
      tiles.SetValues(llvm::make_range(
          new_op.getResults().begin() + res_idx,
          new_op.getResults().begin() + res_idx + num_vectors));
      res_idx += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      rolled_results.push_back(rolled_op);
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      rolled_results.push_back(new_op.getResult(res_idx));
      ++res_idx;
    }
  }

  for_op.replaceAllUsesWith(rolled_results);
  for_op.erase();
  return success();
}

LogicalResult scf_while_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  scf::WhileOp while_op = cast<scf::WhileOp>(op);
  TPU_ASSERT_EQ_OP(layouts_in.size(), while_op->getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), while_op->getNumResults());
  TPU_ASSERT_EQ_OP(layouts_in.size(), layouts_out.size());

  // The terminator for the before region is the condition op.
  // It takes multiple arguments -- the first being the decision to execute the
  // after region or branch to the exit.
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<Layout> cond_in_layouts,
      getInLayouts(*while_op.getBeforeBody()->getTerminator(),
                   ctx.target_shape));

  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<Layout> yield_in_layouts,
      getInLayouts(*while_op.getYieldOp(), ctx.target_shape));
  int out_idx = 0;
  for (auto [in_layout, cond_layout, yield_layout, out_layout, result] :
       llvm::zip_equal(layouts_in,
                       ArrayRef<Layout>(cond_in_layouts).drop_front(1),
                       yield_in_layouts, layouts_out, op.getResults())) {
    if (auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(in_layout.has_value());
      TPU_ASSERT_OP(yield_layout.has_value());
      TPU_ASSERT_OP(out_layout.has_value());
      if (in_layout.value() != cond_layout.value()) {
        return op.emitOpError(
                   "Not implemented: while loop input layout does not match "
                   "with condition layout ")
               << out_idx;
      }
      if (in_layout.value() != yield_layout.value()) {
        return op.emitOpError(
                   "Not implemented: while loop input layout does not match "
                   "with yield layout ")
               << out_idx;
      }
      if (in_layout.value() != out_layout.value()) {
        return op.emitOpError(
                   "Not implemented: while loop input layout does not match "
                   "with output layout ")
               << out_idx;
      }
    } else {
      TPU_ASSERT_EQ_OP(in_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(cond_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(yield_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(out_layout, kNoLayout);
    }
    ++out_idx;
  }

  if (failed(applyLayoutBlock(ctx, *while_op.getBeforeBody()))) {
    return failure();
  }

  if (failed(applyLayoutBlock(ctx, *while_op.getAfterBody()))) {
    return failure();
  }

  if (op.getNumResults() == 0) {
    return success();
  }

  OpBuilder builder(&op);
  SmallVector<Value> unrolled_args;
  for (int i = 0; i < layouts_in.size(); ++i) {
    auto layout = layouts_in[i];
    auto operand = while_op.getOperand(i);
    if (auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand)) {
      if (!layout.has_value()) {
        return op.emitOpError("Expected layout for vector operand");
      }
      FAILUREOR_ASSIGN_OR_RETURN(
          const xla::Array<Value> tiles,
          disassemble(builder, *layout, vector_operand, ctx.target_shape));
      unrolled_args.append(tiles.begin(), tiles.end());
    } else {
      if (layout.has_value()) {
        return op.emitOpError("Expected no layout for scalar operand");
      }
      unrolled_args.push_back(operand);
    }
  }

  // Create a new scf::WhileOp with unrolled args.
  auto new_op = builder.create<scf::WhileOp>(
      while_op->getLoc(),
      TypeRange(while_op.getConditionOp().getOperands().drop_front(1)),
      unrolled_args, nullptr, nullptr);

  const auto tile_body_args = [&](::mlir::Block *old_body,
                                  ::mlir::Block *new_body,
                                  const ArrayRef<Layout> layouts) {
    TPU_ASSERT_OP(old_body != nullptr);
    TPU_ASSERT_OP(new_body != nullptr);
    int num_old_args = old_body->getNumArguments();
    SmallVector<Location> locs(new_body->getNumArguments(), while_op.getLoc());
    old_body->addArguments(TypeRange(new_body->getArguments()), locs);
    builder.setInsertionPointToStart(old_body);
    auto arg_idx = num_old_args;
    for (auto [old_arg, layout] : llvm::zip_equal(
             old_body->getArguments().take_front(num_old_args), layouts)) {
      if (const auto vty = dyn_cast<VectorType>(old_arg.getType())) {
        TPU_ASSERT_OP(layout.has_value());
        const SmallVector<int64_t> tiles_shape =
            layout->tileArrayShape(vty.getShape(), ctx.target_shape);
        const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
        xla::Array<Value> tiles(tiles_shape);
        TPU_ASSERT_LE_OP(arg_idx + num_vectors, old_body->getNumArguments());
        tiles.SetValues(llvm::make_range(
            old_body->getArguments().begin() + arg_idx,
            old_body->getArguments().begin() + arg_idx + num_vectors));
        arg_idx += num_vectors;
        RollVectorsOp rolled_op =
            assemble(builder, vty, *layout, tiles, ctx.target_shape);
        old_arg.replaceUsesWithIf(rolled_op, [&](OpOperand &operand) {
          return operand.getOwner() != rolled_op;
        });
      } else {
        TPU_ASSERT_OP(!layout.has_value());
        old_arg.replaceAllUsesWith(old_body->getArgument(arg_idx));
        ++arg_idx;
      }
    }
    old_body->eraseArguments(0, num_old_args);
    return success();
  };

  const auto before_status = tile_body_args(while_op.getBeforeBody(),
                                            new_op.getBeforeBody(), layouts_in);
  if (before_status.failed()) return before_status;
  new_op.getBefore().takeBody(while_op.getBefore());

  const auto after_status = tile_body_args(while_op.getAfterBody(),
                                           new_op.getAfterBody(), layouts_out);
  if (after_status.failed()) return after_status;
  new_op.getAfter().takeBody(while_op.getAfter());

  builder.setInsertionPointAfter(new_op);
  int64_t res_idx = 0;
  SmallVector<Value> rolled_results;
  for (auto [result, layout] :
       llvm::zip_equal(while_op.getResults(), layouts_out)) {
    if (const auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(res_idx + num_vectors, new_op.getResults().size());
      tiles.SetValues(llvm::make_range(
          new_op.getResults().begin() + res_idx,
          new_op.getResults().begin() + res_idx + num_vectors));
      res_idx += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      rolled_results.push_back(rolled_op);
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      rolled_results.push_back(new_op.getResult(res_idx));
      ++res_idx;
    }
  }

  while_op.replaceAllUsesWith(rolled_results);
  while_op.erase();
  return success();
}

LogicalResult scf_condition_rule(RewriteContext &ctx, Operation &op,
                                 const ArrayRef<Layout> layouts_in,
                                 const ArrayRef<Layout> layouts_out) {
  OpBuilder builder(&op);
  auto condition_op = cast<scf::ConditionOp>(op);
  TPU_ASSERT_EQ_OP(layouts_in.size(), condition_op.getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  SmallVector<Value> unrolled;

  for (auto [operand, layout] :
       llvm::zip_equal(condition_op.getOperands(), layouts_in)) {
    if (auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand)) {
      // When the operand has vector type, disassemble the operand.
      TPU_ASSERT_OP(layout.has_value());
      FAILUREOR_ASSIGN_OR_RETURN(
          const xla::Array<Value> tiles,
          disassemble(builder, *layout, vector_operand, ctx.target_shape));
      unrolled.append(tiles.begin(), tiles.end());
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      unrolled.push_back(operand);
    }
  }

  // Replace the old operands with unrolled operands.
  condition_op->setOperands(unrolled);
  return success();
}

LogicalResult scf_if_rule(RewriteContext &ctx, Operation &op,
                          const ArrayRef<Layout> layouts_in,
                          const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_OP(!layouts_in.front().has_value());
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  scf::IfOp if_op = cast<scf::IfOp>(op);
  SmallVector<Layout, 4> then_yield_in_layouts;
  SmallVector<Layout, 4> else_yield_in_layouts;
  FAILUREOR_ASSIGN_OR_RETURN(
      then_yield_in_layouts,
      getInLayouts(*if_op.thenYield(), ctx.target_shape));
  if (!if_op.getElseRegion().empty()) {
    FAILUREOR_ASSIGN_OR_RETURN(
        else_yield_in_layouts,
        getInLayouts(*if_op.elseYield(), ctx.target_shape));
  }
  int out_idx = 0;
  for (auto [then_layout, else_layout, result_layout, result] :
       llvm::zip_equal(then_yield_in_layouts, else_yield_in_layouts,
                       layouts_out, op.getResults())) {
    if (auto vty = dyn_cast<VectorType>(result.getType())) {
      TPU_ASSERT_OP(then_layout.has_value());
      TPU_ASSERT_OP(else_layout.has_value());
      TPU_ASSERT_OP(result_layout.has_value());
      if (result_layout.value() != then_layout.value()) {
        return op.emitOpError(
                   "Not implemented: yield layout from then branch does not "
                   "match with output layout ")
               << out_idx;
      }
      if (result_layout.value() != else_layout.value()) {
        return op.emitOpError(
                   "Not implemented: yield layout from else branch does not "
                   "match with output layout ")
               << out_idx;
      }
    } else {
      TPU_ASSERT_EQ_OP(then_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(else_layout, kNoLayout);
      TPU_ASSERT_EQ_OP(result_layout, kNoLayout);
    }
    ++out_idx;
  }
  if (failed(applyLayoutBlock(ctx, *if_op.thenBlock()))) {
    return failure();
  }
  if (if_op.getElseRegion().empty()) {
    TPU_ASSERT_EQ_OP(if_op->getNumResults(), 0);
    TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
    return success();
  }
  if (failed(applyLayoutBlock(ctx, *if_op.elseBlock()))) {
    return failure();
  }

  // Apply layout to results after applying layout in the true and false
  // regions.
  if (if_op.getNumResults() == 0) {
    TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
    return success();
  }
  TPU_ASSERT_EQ_OP(if_op.getNumResults(), layouts_out.size());
  // If scf.if has results, it should have both non-empty true and false
  // regions.
  TPU_ASSERT_OP(!if_op.getThenRegion().empty() &&
                !if_op.getElseRegion().empty());

  // Move true and false regions to the new if op whose result has same type and
  // layout as yield operand's.
  auto new_op = builder.create<scf::IfOp>(
      TypeRange(if_op.thenYield().getResults()), if_op.getCondition(),
      /*withElseRegion =*/true);
  moveAllRegions(*if_op, *new_op);

  int64_t index = 0;
  SmallVector<Value> rolled_results;
  for (auto [result, layout] :
       llvm::zip_equal(if_op.getResults(), layouts_out)) {
    if (const auto vty = dyn_cast<VectorType>(result.getType())) {
      // When the result has a vector type, assemble the result.
      TPU_ASSERT_OP(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      TPU_ASSERT_LE_OP(index + num_vectors, new_op.getResults().size());
      tiles.SetValues(
          llvm::make_range(new_op.getResults().begin() + index,
                           new_op.getResults().begin() + index + num_vectors));
      index += num_vectors;
      RollVectorsOp rolled_op =
          assemble(builder, vty, *layout, tiles, ctx.target_shape);
      rolled_results.push_back(rolled_op);
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      rolled_results.push_back(new_op.getResult(index));
      ++index;
    }
  }
  if_op.replaceAllUsesWith(rolled_results);
  if_op.erase();
  return success();
}

LogicalResult scf_yield_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  OpBuilder builder(&op);
  auto yield_op = cast<scf::YieldOp>(op);
  TPU_ASSERT_EQ_OP(layouts_in.size(), yield_op.getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  if (yield_op.getNumOperands() == 0) {
    return success();
  }
  SmallVector<Value> unrolled;
  for (auto [operand, layout] :
       llvm::zip_equal(yield_op.getOperands(), layouts_in)) {
    if (auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand)) {
      // When the operand has vector type, disassemble the operand.
      TPU_ASSERT_OP(layout.has_value());
      FAILUREOR_ASSIGN_OR_RETURN(
          const xla::Array<Value> tiles,
          disassemble(builder, *layout, vector_operand, ctx.target_shape));
      unrolled.append(tiles.begin(), tiles.end());
    } else {
      TPU_ASSERT_OP(!layout.has_value());
      unrolled.push_back(operand);
    }
  }

  // Replace the old operands with unrolled operands.
  yield_op->setOperands(unrolled);
  return success();
}

LogicalResult tpu_load_rule(RewriteContext &ctx, Operation &op,
                            const ArrayRef<Layout> layouts_in,
                            const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(llvm::none_of(layouts_in,
                              [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_out = *layouts_out.front();
  // We expect the result is already a native-sized vreg.
  // TODO(b/300493694): Support other bitwidths
  if (layout_out.bitwidth() != 32) {
    return op.emitOpError("Not implemented: Only 32-bit loads supported");
  }
  tpu::LoadOp load_op = cast<tpu::LoadOp>(op);
  if (layout_out != VectorLayout(32, {0, 0}, ctx.target_shape,
                                 VectorLayout::ImplicitDim::kNone)) {
    return op.emitOpError("Invalid output layout for ") << load_op->getName();
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<int64_t> indices,
      getIntConstsFromOperandRange(load_op.getIndices()));
  TPU_ASSERT_EQ_OP(indices.size(), 2);
  if (indices[1] % ctx.target_shape[1] != 0) {
    return op.emitOpError("Not implemented: Lane index is not a multiple of ")
           << ctx.target_shape[1];
  }

  OpBuilder builder(op.getContext());
  builder.setInsertionPointAfter(&op);
  const RollVectorsOp roll_vectors_op =
      assemble(builder, load_op.getResult().getType(), layout_out,
               {{load_op.getResult()}}, ctx.target_shape);
  load_op->replaceUsesWithIf(roll_vectors_op, [&](OpOperand &operand) {
    return operand.getOwner() != roll_vectors_op;
  });
  return success();
}

LogicalResult strided_op_rule_impl(RewriteContext &ctx, Operation &op,
                                   Value base_ref, ValueRange indices,
                                   const VectorType &vty,
                                   const VectorLayout &layout,
                                   const ArrayRef<int32_t> &strides) {
  if (!isa<tpu::StridedLoadOp, tpu::StridedStoreOp>(op)) {
    return op.emitOpError("Not implemented: Unsupported strided op")
           << op.getName();
  }
  if (layout != VectorLayout(32, {0, 0}, ctx.target_shape,
                             VectorLayout::ImplicitDim::kNone)) {
    return op.emitOpError("Not implemented: Unsupported vector layout in ")
           << op.getName();
  }
  const auto base_ty = getMemRefType(base_ref);
  auto rank = base_ty.getRank();
  CHECK_EQ(rank, indices.size());
  CHECK_EQ(rank, strides.size());
  CHECK_EQ(rank, vty.getShape().size());
  if (rank < 2) {
    return op.emitOpError("Not implemented: Stride on 1D vector");
  }
  auto mem_layout = dyn_cast<TiledLayoutAttr>(base_ty.getLayout());
  if (!mem_layout) {
    return op.emitOpError("Expected a tiled memref");
  }
  auto tile_strides = mem_layout.getTileStrides();

  // Currently we hold constraints that the last dim size of memref needs to be
  // exactly same as the lane size of native vreg and the memref has never
  // been sliced before on the last dim. In other words, the original base
  // memref's shape needs to be (..., target_shape[1]).
  if (base_ty.getShape()[rank - 1] != ctx.target_shape[1] ||
      tile_strides.take_back(2) != ArrayRef<int64_t>{1, 1}) {
    return op.emitOpError("Not Implemented: The last dim size is not ")
           << ctx.target_shape[1] << " in original base memref";
  }
  if (strides[rank - 1] != 1) {
    return op.emitOpError("Not Implemented: Stride on last dim is not 1");
  }
  auto last_idx = getIntConst(indices[rank - 1], /*silent=*/true);
  if (failed(last_idx)) {
    return op.emitOpError("Not Implemented: Dynamic index on last dim");
  } else if (last_idx.value() != 0) {
    return op.emitOpError("Not Implemented: Index on last dim is not 0");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);

  FAILUREOR_ASSIGN_OR_RETURN(
      VectorType vreg_ty,
      getNativeVregType(vty.getElementType(), ctx.target_shape));

  bool is_load_op = true;
  xla::Array<Value> tiles(
      layout.tileArrayShape(vty.getShape(), ctx.target_shape));
  if (auto store_op = dyn_cast<tpu::StridedStoreOp>(op)) {
    is_load_op = false;
    FAILUREOR_ASSIGN_OR_RETURN(
        tiles, disassemble(builder, layout, store_op.getValueToStore(),
                           ctx.target_shape));
  }

  tiles.Each([&](absl::Span<const int64_t> tile_idxs, Value *v) {
    CHECK_EQ(tile_idxs.size(), rank);
    SmallVector<Value> idxs(rank);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t stride = (i < rank - 2)
                           ? strides[i]
                           : (strides[i] * ctx.target_shape[i - rank + 2]);
      idxs[i] = builder.create<arith::AddIOp>(
          indices[i], IdxConst(tile_idxs[i] * stride, builder, op.getLoc()));
    }
    SmallVector<bool> sublane_mask(ctx.target_shape[0], true);
    int64_t sublane_rem = vty.getDimSize(rank - 2) % ctx.target_shape[0];
    if (sublane_rem > 0 && tile_idxs[rank - 2] == tiles.dim(rank - 2) - 1) {
      for (int64_t i = sublane_rem; i < ctx.target_shape[0]; ++i) {
        sublane_mask[i] = false;
      }
    }
    const auto sublane_mask_attr =
        DenseBoolArrayAttr::get(op.getContext(), sublane_mask);
    if (is_load_op) {
      *v = builder.create<tpu::LoadOp>(
          vreg_ty, base_ref, idxs, sublane_mask_attr,
          builder.getI32IntegerAttr(strides[rank - 2]));
    } else {
      builder.create<tpu::StoreOp>(
          *v, base_ref, idxs, sublane_mask_attr,
          /*mask=*/nullptr, builder.getI32IntegerAttr(strides[rank - 2]));
    }
  });
  if (is_load_op) {
    op.replaceAllUsesWith(
        assemble(builder, vty, layout, std::move(tiles), ctx.target_shape));
  }
  op.erase();
  return success();
}

// TODO(jevinjiang): maybe unify with vector load?
LogicalResult tpu_strided_load_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_OP(llvm::none_of(layouts_in,
                              [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_out = *layouts_out.front();
  auto load_op = cast<tpu::StridedLoadOp>(op);
  const auto vty = cast<VectorType>(load_op.getResult().getType());
  return strided_op_rule_impl(ctx, op, load_op.getBase(), load_op.getIndices(),
                              vty, layout_out, load_op.getStrides());
}

// TODO(jevinjiang): maybe unify with vector store?
LogicalResult tpu_strided_store_rule(RewriteContext &ctx, Operation &op,
                                     const ArrayRef<Layout> layouts_in,
                                     const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(llvm::none_of(layouts_in.drop_front(),
                              [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);

  const VectorLayout &to_store_layout = *layouts_in.front();
  auto store_op = cast<tpu::StridedStoreOp>(op);
  const auto vty = store_op.getValueToStore().getType();
  return strided_op_rule_impl(ctx, op, store_op.getBase(),
                              store_op.getIndices(), vty, to_store_layout,
                              store_op.getStrides());
}

LogicalResult matmul_rule_impl(RewriteContext &ctx, Operation &op,
                               const bool transpose_lhs,
                               const bool transpose_rhs,
                               const VectorLayout &layout_lhs,
                               const VectorLayout &layout_rhs,
                               const VectorLayout &layout_acc,
                               const VectorLayout &layout_out) {
  if (transpose_lhs) {
    return op.emitOpError("Not implemented: Transposed LHS");
  }
  const std::array<std::reference_wrapper<const VectorLayout>, 3> layouts_in = {
      layout_lhs, layout_rhs, layout_acc};
  const std::array<std::reference_wrapper<const VectorLayout>, 4> all_layouts =
      {layout_lhs, layout_rhs, layout_acc, layout_out};
  for (const VectorLayout &layout : all_layouts) {
    for (const LayoutOffset offset : layout.offsets()) {
      if (offset.value_or(0) != 0) {
        return op.emitOpError("Not implemented: Unaligned layout in matmul");
      }
    }
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  TypedValue<VectorType> lhs, rhs, acc, res;
  if (auto tpu_matmul_op = dyn_cast<tpu::MatmulOp>(op)) {
    lhs = tpu_matmul_op.getLhs();
    rhs = tpu_matmul_op.getRhs();
    acc = tpu_matmul_op.getAcc();
    res = tpu_matmul_op.getResult();
  } else if (auto vector_contraction_op = dyn_cast<vector::ContractionOp>(op)) {
    lhs = vector_contraction_op.getLhs();
    rhs = vector_contraction_op.getRhs();
    acc = cast<TypedValue<VectorType>>(vector_contraction_op.getAcc());
    res = cast<TypedValue<VectorType>>(vector_contraction_op.getResult());
  } else {
    LOG(FATAL) << "Unexpected op type";
  }

  for (const VectorLayout &layout : layouts_in) {
    if (layout.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
      return op.emitOpError(
          "Not implemented: Unsupported matmul operand layout");
    }
    if (!layout.hasNativeTiling(ctx.target_shape)) {
      return op.emitOpError(
          "Not implemented: Unsupported matmul operand tiling");
    }
  }
  if (acc.getType().getElementType().getIntOrFloatBitWidth() != 32) {
    return op.emitOpError("Not implemented: Non-32-bit matmul result");
  }
  const ArrayRef<int64_t> lhs_shape = lhs.getType().getShape();
  const ArrayRef<int64_t> rhs_shape = rhs.getType().getShape();
  // TODO(tlongeri): This should be part of the tpu::MatmulOp verifier
  TPU_ASSERT_EQ_OP(lhs_shape.size(), 2);
  TPU_ASSERT_EQ_OP(rhs_shape.size(), 2);
  // The code below puts no constraints on the second dimension of both lhs and
  // rhs. However, leading axis of lhs and rhs needs to be a multiple of native
  // tiling for packed types.
  if (layout_lhs.packing() != 1 && lhs_shape[0] % layout_lhs.tiling()[0] != 0) {
    return op.emitOpError(
        "Not implemented: Unsupported LHS shape with padded tiling and "
        "narrower data type");
  }
  if (layout_rhs.packing() != 1 && rhs_shape[0] % layout_rhs.tiling()[0] != 0) {
    return op.emitOpError(
        "Not implemented: Unsupported RHS shape with padded tiling and "
        "narrower data type");
  }

  const int64_t padded_lhs_rows =
      llvm::alignTo(lhs_shape[0], layout_lhs.tiling()[0]);
  const int64_t padded_lhs_cols =
      llvm::alignTo(lhs_shape[1], layout_lhs.tiling()[1]);
  const int64_t padded_rhs_rows =
      llvm::alignTo(rhs_shape[0], layout_rhs.tiling()[0]);
  const int64_t padded_rhs_cols =
      llvm::alignTo(rhs_shape[1], layout_rhs.tiling()[1]);

  if (llvm::alignTo(lhs_shape[0], layout_acc.tiling()[0]) != padded_lhs_rows) {
    return op.emitOpError(
        "Not implemented: Matmul acc requires less padding than lhs");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> lhs_vregs,
      disassemble(builder, layout_lhs, lhs, ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> acc_vregs,
      disassemble(builder, layout_acc, acc, ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> rhs_vregs,
      disassemble(builder, layout_rhs, rhs, ctx.target_shape));
  TPU_ASSERT_EQ_OP(padded_lhs_rows, lhs_vregs.dim(0) * layout_lhs.tiling()[0]);
  TPU_ASSERT_EQ_OP(padded_lhs_rows, acc_vregs.dim(0) * layout_acc.tiling()[0]);
  TPU_ASSERT_EQ_OP(padded_rhs_rows, rhs_vregs.dim(0) * layout_rhs.tiling()[0]);

  FAILUREOR_ASSIGN_OR_RETURN(
      const VectorType i32_vreg,
      getNativeVregType(builder.getI32Type(), ctx.target_shape));
  auto getVmaskByPaddingEnd = [&](int64_t dim, int64_t padding,
                                  VectorType vreg_ty) {
    CHECK(dim == 0 || dim == 1);
    CHECK(padding >= 0 && padding <= ctx.target_shape[dim]);
    auto mask = cast<TypedValue<VectorType>>(
        builder
            .create<arith::CmpIOp>(
                arith::CmpIPredicate::slt,
                builder.create<tpu::IotaOp>(i32_vreg,
                                            builder.getI32IntegerAttr(dim)),
                builder.create<arith::ConstantOp>(DenseElementsAttr::get(
                    i32_vreg, builder.getI32IntegerAttr(ctx.target_shape[dim] -
                                                        padding))))
            .getResult());
    if (vreg_ty.getShape() != mask.getType().getShape()) {
      mask = builder.create<tpu::MaskCastOp>(
          VectorType::get(vreg_ty.getShape(), builder.getI1Type()), mask);
    }
    return mask;
  };

  // We can also extend this helper function with padding_top and padding_left
  // based on the offsets in vregs.
  auto maskVregs = [&](xla::Array<Value> &vregs, Value zeros,
                       int64_t padding_bottom, int64_t padding_right) {
    auto vreg_ty = cast<VectorType>(vregs.begin()->getType());
    // Mask out the bottom.
    if (padding_bottom > 0) {
      auto mask_bottom = getVmaskByPaddingEnd(0, padding_bottom, vreg_ty);
      for (int64_t i = 0; i < vregs.dim(1); ++i) {
        Value &vreg = vregs({vregs.dim(0) - 1, i});
        vreg = builder.create<arith::SelectOp>(mask_bottom, vreg, zeros);
      }
    }
    // Mask out the right.
    if (padding_right > 0) {
      auto mask_right = getVmaskByPaddingEnd(1, padding_right, vreg_ty);
      for (int64_t i = 0; i < vregs.dim(0); ++i) {
        Value &vreg = vregs({i, vregs.dim(1) - 1});
        vreg = builder.create<arith::SelectOp>(mask_right, vreg, zeros);
      }
    }
  };

  // Create a vreg filled with zeros.
  auto getZerosVergLike =
      [&](const Value &vreg) -> FailureOr<TypedValue<VectorType>> {
    const VectorType vreg_type = cast<VectorType>(vreg.getType());
    FAILUREOR_ASSIGN_OR_RETURN(
        const Attribute zero_attr,
        getZeroIntOrFloatAttr(vreg_type.getElementType()));
    return cast<TypedValue<VectorType>>(
        builder
            .create<arith::ConstantOp>(
                op.getLoc(), DenseElementsAttr::get(vreg_type, zero_attr))
            .getResult());
  };

  FAILUREOR_ASSIGN_OR_RETURN(auto lhs_zeros_vreg,
                             getZerosVergLike(*lhs_vregs.begin()));
  FAILUREOR_ASSIGN_OR_RETURN(auto rhs_zeros_vreg,
                             getZerosVergLike(*rhs_vregs.begin()));
  FAILUREOR_ASSIGN_OR_RETURN(auto acc_zeros_vreg,
                             getZerosVergLike(*acc_vregs.begin()));

  // Only mask out the paddings on contracting dim of LHS and RHS.
  maskVregs(lhs_vregs, lhs_zeros_vreg, 0, padded_lhs_cols - lhs_shape[1]);
  if (transpose_rhs) {
    maskVregs(rhs_vregs, rhs_zeros_vreg, 0, padded_rhs_cols - rhs_shape[1]);
  } else {
    maskVregs(rhs_vregs, rhs_zeros_vreg, padded_rhs_rows - rhs_shape[0], 0);
  }

  // TODO(b/328094640): use latch 3 for short dimensions.
  // TODO(b/328093587): Skip zeros vreg matmul
  // At this point, all paddings on vregs are masked out. For now, we
  // append zero vregs to make LHS's second dim, both RHS's dims and ACC's
  // second dim to be a multiple of mxu_size.
  if (ctx.mxu_shape[0] != ctx.mxu_shape[1]) {
    return op.emitOpError(
        "Not implemented: MXU contracting size and noncontracting size are "
        "different");
  }
  int64_t mxu_size = ctx.mxu_shape[0];
  CHECK_EQ(mxu_size % ctx.target_shape[0], 0);
  CHECK_EQ(mxu_size % ctx.target_shape[1], 0);
  auto mxu_row_vregs = mxu_size / (ctx.target_shape[0] * layout_rhs.packing());
  auto mxu_col_vregs = mxu_size / ctx.target_shape[1];
  int64_t target_lhs_col_vregs = llvm::alignTo(lhs_vregs.dim(1), mxu_col_vregs);
  int64_t target_rhs_row_vregs = llvm::alignTo(rhs_vregs.dim(0), mxu_row_vregs);
  int64_t target_rhs_col_vregs = llvm::alignTo(rhs_vregs.dim(1), mxu_col_vregs);
  int64_t target_acc_col_vregs = llvm::alignTo(acc_vregs.dim(1), mxu_col_vregs);

  xla::Array<Value> target_lhs_vregs({lhs_vregs.dim(0), target_lhs_col_vregs},
                                     lhs_zeros_vreg);
  xla::Array<Value> target_rhs_vregs(
      {target_rhs_row_vregs, target_rhs_col_vregs}, rhs_zeros_vreg);
  xla::Array<Value> target_acc_vregs({acc_vregs.dim(0), target_acc_col_vregs},
                                     acc_zeros_vreg);
  target_lhs_vregs.UpdateSlice(lhs_vregs, {0, 0});
  target_rhs_vregs.UpdateSlice(rhs_vregs, {0, 0});
  target_acc_vregs.UpdateSlice(acc_vregs, {0, 0});

  // Now we can regroup vregs from target vregs.
  const auto lhs_col_ty = VectorType::get({padded_lhs_rows, mxu_size},
                                          lhs.getType().getElementType());
  const auto acc_col_ty = VectorType::get({padded_lhs_rows, mxu_size},
                                          acc.getType().getElementType());
  const ArrayAttr lhs_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_lhs)});
  const ArrayAttr rhs_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_rhs)});
  const ArrayAttr acc_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_acc)});

  int64_t nk = llvm::divideCeil(lhs_shape[1], mxu_size);
  CHECK_EQ(nk, target_lhs_vregs.dim(1) / mxu_col_vregs);
  SmallVector<tpu::RollVectorsOp> lhs_cols(nk);
  for (int64_t i = 0; i < nk; ++i) {
    const xla::Array<Value> col_vregs = target_lhs_vregs.Slice(
        {0, i * mxu_col_vregs},
        {target_lhs_vregs.dim(0), (i + 1) * mxu_col_vregs});
    lhs_cols[i] = builder.create<tpu::RollVectorsOp>(
        op.getLoc(), lhs_col_ty, XlaArrayToFlatArrayRef(col_vregs));
    lhs_cols[i]->setAttr("out_layout", lhs_layout_attr);
  }
  // Here, "tile" is used as in the context of the MXU shape (NOT as in the
  // context of tiled layouts).
  const auto rhs_tile_ty =
      VectorType::get({mxu_size, mxu_size}, rhs.getType().getElementType());
  const int64_t rhs_vregs_per_tile = mxu_row_vregs * mxu_col_vregs;

  int64_t nj;
  if (transpose_rhs) {
    nj = llvm::divideCeil(rhs_shape[0], mxu_size);
    CHECK_EQ(nk, llvm::divideCeil(rhs_shape[1], mxu_size));
    CHECK_EQ(nk, target_rhs_vregs.dim(1) / mxu_col_vregs);
    target_rhs_vregs.Reshape(
        {nj, rhs_vregs_per_tile / mxu_col_vregs, nk, mxu_col_vregs});
    target_rhs_vregs.TransposeDimensions({2, 0, 1, 3});
    target_rhs_vregs.Reshape({nk, nj, rhs_vregs_per_tile});
  } else {
    nj = llvm::divideCeil(rhs_shape[1], mxu_size);
    CHECK_EQ(nk, llvm::divideCeil(rhs_shape[0], mxu_size));
    CHECK_EQ(nk, target_rhs_vregs.dim(0) / mxu_row_vregs);
    target_rhs_vregs.Reshape(
        {nk, rhs_vregs_per_tile / mxu_col_vregs, nj, mxu_col_vregs});
    target_rhs_vregs.TransposeDimensions({0, 2, 1, 3});
    target_rhs_vregs.Reshape({nk, nj, rhs_vregs_per_tile});
  }

  const tpu::ContractPrecisionAttr precision_attr =  // May be null
      op.getAttrOfType<tpu::ContractPrecisionAttr>("precision");
  for (int64_t j = 0; j < nj; ++j) {
    for (int64_t k = 0; k < nk; ++k) {
      // TODO(tlongeri): there should be a way to slice without copying
      xla::Array<Value> rhs_tile =
          target_rhs_vregs.Slice({k, j, 0}, {k + 1, j + 1, rhs_vregs_per_tile});
      auto rhs_rolled_tile = builder.create<tpu::RollVectorsOp>(
          op.getLoc(), rhs_tile_ty, XlaArrayToFlatArrayRef(rhs_tile));
      rhs_rolled_tile->setAttr("out_layout", rhs_layout_attr);
      const xla::Array<Value> acc_col_vregs = target_acc_vregs.Slice(
          {0, j * mxu_col_vregs},
          {target_acc_vregs.dim(0), (j + 1) * mxu_col_vregs});
      auto acc_col = builder.create<tpu::RollVectorsOp>(
          op.getLoc(), acc_col_ty, XlaArrayToFlatArrayRef(acc_col_vregs));
      acc_col->setAttr("out_layout", acc_layout_attr);
      auto new_acc_col = builder.create<tpu::MatmulOp>(
          op.getLoc(), acc_col_ty, lhs_cols[k], rhs_rolled_tile, acc_col,
          transpose_lhs, transpose_rhs, precision_attr);
      auto new_acc_vregs = builder.create<tpu::UnrollVectorsOp>(
          op.getLoc(),
          TypeRange(ValueRange(XlaArrayToFlatArrayRef(acc_col_vregs))),
          new_acc_col);
      new_acc_vregs->setAttr("in_layout", acc_layout_attr);
      updateSliceFromRange(target_acc_vregs, new_acc_vregs->getResults(),
                           {0, j * mxu_col_vregs},
                           {target_acc_vregs.dim(0), (j + 1) * mxu_col_vregs});
    }
  }
  op.replaceAllUsesWith(
      assemble(
          builder, res.getType(), layout_out,
          target_acc_vregs.Slice({0, 0}, {acc_vregs.dim(0), acc_vregs.dim(1)}),
          ctx.target_shape)
          .getOperation());
  op.erase();
  return success();
}

LogicalResult tpu_matmul_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 3);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(
      llvm::all_of(layouts_in, [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  auto matmul_op = cast<tpu::MatmulOp>(op);
  return matmul_rule_impl(ctx, *matmul_op, matmul_op.getTransposeLhs(),
                          matmul_op.getTransposeRhs(), *layouts_in[0],
                          *layouts_in[1], *layouts_in[2], *layouts_out[0]);
}

LogicalResult tpu_store_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  TPU_ASSERT_OP(layouts_in.front().has_value());  // value to store layout
  TPU_ASSERT_OP(llvm::none_of(layouts_in.drop_front(),
                              [&](const Layout &l) { return l.has_value(); }));
  OpBuilder builder(&op);
  const VectorLayout &to_store_layout = *layouts_in.front();
  // We expect the value to store is already a native-sized vreg.
  if (to_store_layout.bitwidth() != 32) {
    return op.emitOpError("Not implemented: Only 32-bit loads supported");
  }
  TPU_ASSERT_OP(to_store_layout ==
                VectorLayout(32, {0, 0}, ctx.target_shape,
                             VectorLayout::ImplicitDim::kNone));
  tpu::StoreOp store_op = cast<tpu::StoreOp>(op);
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<int64_t> indices,
      getIntConstsFromOperandRange(store_op.getIndices()));
  TPU_ASSERT_EQ_OP(indices.size(), 2);
  if (indices[1] % ctx.target_shape[1] != 0) {
    return op.emitOpError("Not implemented: Lane index is not a multiple of ")
           << ctx.target_shape[1];
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> tiles,
      disassemble(builder, to_store_layout, store_op.getValueToStore(),
                  ctx.target_shape));
  TPU_ASSERT_OP((tiles.dimensions() == xla::DimensionVector{1, 1}));
  store_op.getValueToStoreMutable().assign(tiles({0, 0}));
  return success();
}

LogicalResult tpu_bitcast_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (!layout_in.hasNativeTiling(ctx.target_shape) ||
      !layout_out.hasNativeTiling(ctx.target_shape)) {
    return op.emitOpError("Not implemented: unsupported tiling");
  }
  if (layout_in.offsets() != LayoutOffsets{0, 0} ||
      layout_out.offsets() != LayoutOffsets{0, 0}) {
    return op.emitOpError("Not implemented: unsupported offsets");
  }
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    return op.emitOpError("Not implemented: unsupported implicit dim");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto bitcast_op = cast<tpu::BitcastOp>(op);
  const VectorType vty = bitcast_op.getResult().getType();
  FAILUREOR_ASSIGN_OR_RETURN(
      const auto native_vreg_ty,
      getNativeVregType(vty.getElementType(), ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> in_tiles,
      disassemble(builder, layout_in, bitcast_op.getInput(), ctx.target_shape));
  xla::Array<Value> out_tiles(in_tiles.dimensions());
  out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
    const Value in_tile = in_tiles(idxs);
    *v = builder.create<tpu::BitcastVregOp>(native_vreg_ty, in_tile);
  });
  bitcast_op.replaceAllUsesWith(
      assemble(builder, vty, layout_out, out_tiles, ctx.target_shape)
          .getOperation());
  bitcast_op.erase();
  return success();
}

LogicalResult tpu_trace_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
    return op.emitOpError(
        "Not implemented: tpu.traced_block with inputs or outputs");
  }
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  // We don't modify the op, but we do rewrite the branch bodies.
  TPU_ASSERT_EQ_OP(op.getNumRegions(), 1);
  Region &region = op.getRegion(0);
  TPU_ASSERT_OP(region.hasOneBlock());
  Block &block = region.front();
  return applyLayoutBlock(ctx, block);
}

LogicalResult tpu_assume_layout_rule(RewriteContext &ctx, Operation &op,
                                     const ArrayRef<Layout> layouts_in,
                                     const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(op.getNumOperands(), 1);
  TPU_ASSERT_EQ_OP(op.getNumResults(), 1);
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  if (layouts_in[0] != layouts_out[0]) {
    return op.emitOpError("Expected same input and output layout");
  }
  OpBuilder builder(&op);
  auto val = op.getOperand(0);
  auto layout = layouts_in[0];
  const auto vty = cast<VectorType>(val.getType());
  SmallVector<int64_t> layout_shape =
      layout->tileArrayShape(vty.getShape(), ctx.target_shape);
  const int64_t num_vectors = ShapedType::getNumElements(layout_shape);
  FAILUREOR_ASSIGN_OR_RETURN(
      VectorType vreg_ty,
      getNativeVregType(vty.getElementType(), ctx.target_shape));
  // We can not use disassemble here because the val is block argument.
  auto unrolled_op = builder.create<tpu::UnrollVectorsOp>(
      val.getLoc(), SmallVector<Type>(num_vectors, vreg_ty), val);

  op.replaceAllUsesWith(assemble(builder, vty, *layout,
                                 XlaArrayFromShapeAndValues<Value>(
                                     layout_shape, unrolled_op->getResults()),
                                 ctx.target_shape));
  op.erase();
  return success();
}

// TODO(b/347016737): Deprecate tpu.rotate and only use tpu.dynamic_rotate. So
// we do not need template for the op type and to explicitly force amount
// argument to dynamic.
template <typename OpTy>
LogicalResult rotate_rule_impl(RewriteContext &ctx, OpTy op, Value amount,
                               const VectorLayout &layout_in,
                               const VectorLayout &layout_out) {
  auto layout = VectorLayout(32, {0, 0}, ctx.target_shape,
                             VectorLayout::ImplicitDim::kNone);
  if (layout_in != layout) {
    return op.emitOpError("Not implemented: unsupported layout for input");
  }
  if (layout_out != layout) {
    return op.emitOpError("Not implemented: unsupported layout for output");
  }
  auto vty = op.getResult().getType();
  if (vty.getRank() < 2) {
    return op.emitOpError("Not implemented: unsupported 1D shape");
  }
  if (*(vty.getShape().end() - 2) % *(layout.tiling().end() - 2) != 0 ||
      *(vty.getShape().end() - 1) % *(layout.tiling().end() - 1) != 0) {
    return op.emitOpError("Not implemented: unsupported unaliged shape");
  }

  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());
  FAILUREOR_ASSIGN_OR_RETURN(
      VectorType res_vreg_ty,
      getNativeVregType(vty.getElementType(), ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> in_tiles,
      disassemble(builder, layout_in, op.getValue(), ctx.target_shape));

  FAILUREOR_ASSIGN_OR_RETURN(
      const VectorType i32_vreg,
      getNativeVregType(builder.getI32Type(), ctx.target_shape));

  // Some helper functions for math ops.
  auto mlirI32Const = [&](int d) {
    return builder.create<arith::ConstantOp>(
        builder.getIntegerAttr(builder.getI32Type(), d));
  };
  auto mlirIndexConst = [&](int d) {
    return builder.create<arith::ConstantOp>(
        builder.getIntegerAttr(builder.getIndexType(), d));
  };
  auto modI = [&](const Value &v, unsigned d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return mlirI32Const(cst.value() % d);
    }
    return builder.create<arith::RemUIOp>(v, mlirI32Const(d));
  };
  auto divI = [&](const Value &v, unsigned d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return mlirI32Const(cst.value() / d);
    }
    return builder.create<arith::DivUIOp>(v, mlirI32Const(d));
  };
  auto addI = [&](const Value &v, unsigned d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return mlirI32Const(cst.value() + d);
    }
    return builder.create<arith::AddIOp>(v, mlirI32Const(d));
  };

  // A helper function that creates a VMASK with false flags to bottom (dim = 0)
  // or right (dim = 1) where the flag count corresponds to the (dim_size -
  // padding). If stride is provided, the padding value is sequentially
  // increased by the stride value along the dim.
  //
  // For example, assume VMASK shape is (4, 8)
  //
  // getVmaskByPaddingEnd(padding=3, dim=1) creates:
  //  [T, T, T, T, T, F, F, F]
  //  [T, T, T, T, T, F, F, F]
  //  [T, T, T, T, T, F, F, F]
  //  [T, T, T, T, T, F, F, F]
  //
  // getVmaskByPaddingEnd(padding=3, dim=1, stride=1) creates:
  //  [T, T, T, T, T, F, F, F]
  //  [T, T, T, T, T, T, F, F]
  //  [T, T, T, T, T, T, T, F]
  //  [T, T, T, T, T, T, T, T]
  auto getVmaskByPaddingEnd = [&](Value padding, int dim, int stride = 0) {
    CHECK(dim == 0 || dim == 1);
    Value padding_vreg;
    if (auto padding_cst = getIntConst(padding, /*silent=*/true);
        succeeded(padding_cst)) {
      CHECK_GE(padding_cst.value(), 0);
      CHECK_LE(padding_cst.value(), ctx.target_shape[dim]);
      padding_vreg = builder.create<arith::ConstantOp>(DenseElementsAttr::get(
          i32_vreg, builder.getI32IntegerAttr(padding_cst.value())));
    } else {
      padding_vreg = builder.create<vector::BroadcastOp>(i32_vreg, padding);
    }

    if (stride > 0) {
      auto offset = builder.create<arith::MulIOp>(
          i32_vreg,
          builder.create<tpu::IotaOp>(
              i32_vreg, builder.getI32IntegerAttr(dim == 0 ? 1 : 0)),
          builder.create<arith::ConstantOp>(DenseElementsAttr::get(
              i32_vreg, builder.getI32IntegerAttr(stride))));
      padding_vreg =
          builder.create<arith::AddIOp>(i32_vreg, padding_vreg, offset);
    }
    return builder.create<arith::CmpIOp>(
        arith::CmpIPredicate::slt,
        builder.create<tpu::IotaOp>(i32_vreg, builder.getI32IntegerAttr(dim)),
        padding_vreg);
  };

  // Apply rotation on each vreg with the assumption that shift <= VREG dim size
  // and blend the data from contiguous vregs to emulate circular rotation.
  auto rotateOnTilingDim = [&](const xla::Array<Value> &vregs,
                               const Value &shift, int axis, int stride = 0) {
    if (auto shift_cst = getIntConst(shift, /*silent=*/true);
        succeeded(shift_cst)) {
      if (shift_cst.value() == 0 && stride == 0) {
        return vregs;
      }
    }
    int tiling_dim = axis - (vregs.num_dimensions() - 2);
    CHECK((tiling_dim == 0 && stride == 0) || (tiling_dim == 1 && stride >= 0));
    xla::Array<Value> result(vregs.dimensions());
    auto chunks = split(vregs, axis);
    for (int64_t i = 0; i < chunks.size(); ++i) {
      chunks[i].Each([&](absl::Span<const int64_t> idxs, Value *v) {
        auto stride_attr =
            stride > 0 ? builder.getSI32IntegerAttr(stride) : nullptr;
        auto stride_dimension_attr =
            stride > 0 ? builder.getSI32IntegerAttr(0) : nullptr;
        *v = builder.create<tpu::DynamicRotateOp>(res_vreg_ty, *v, shift,
                                                  tiling_dim, stride_attr,
                                                  stride_dimension_attr);
      });
    }
    auto mask = getVmaskByPaddingEnd(shift, tiling_dim, stride);
    xla::Array<Value> last_chunk_copy(chunks[chunks.size() - 1]);
    for (int64_t i = chunks.size() - 1; i > 0; --i) {
      chunks[i].Each([&](absl::Span<const int64_t> idxs, Value *v) {
        *v = builder.create<arith::SelectOp>(mask, chunks[i - 1](idxs), *v);
      });
    }
    chunks[0].Each([&](absl::Span<const int64_t> idxs, Value *v) {
      *v = builder.create<arith::SelectOp>(mask, last_chunk_copy(idxs), *v);
    });
    return concatenate(chunks, axis);
  };

  std::function<xla::Array<Value>(const xla::Array<Value> &, Value, int, int)>
      rotate;
  rotate = [&](const xla::Array<Value> &vregs, Value shift, int axis,
               int stride) {
    xla::Array<Value> result(vregs.dimensions());
    CHECK(axis >= 0 && axis < vregs.num_dimensions());
    int tiling_dim = axis - (vregs.num_dimensions() - 2);
    CHECK((tiling_dim != 1 && stride == 0) || (tiling_dim == 1 && stride >= 0));
    SmallVector<xla::Array<Value>, 4> chunks;
    // Handle rotation with static shift.
    if (auto shift_cst = getIntConst(shift, /*silent=*/true);
        succeeded(shift_cst)) {
      int64_t static_shift = shift_cst.value();
      if (tiling_dim >= 0) {
        shift = mlirI32Const(static_shift % ctx.target_shape[tiling_dim]);
        static_shift /= ctx.target_shape[tiling_dim];
        chunks = split(rotateOnTilingDim(vregs, shift, axis, stride), axis);
      } else {
        chunks = split(vregs, axis);
      }
      // Now we only need to shuffle vregs.
      for (int64_t i = 0; i < chunks.size(); ++i) {
        SmallVector<int64_t> starts(result.num_dimensions(), 0);
        starts[axis] = (i + static_shift) % result.dim(axis);
        result.UpdateSlice(chunks[i], starts);
      }
      return result;
    }
    // Handle rotation with dynamic shift.
    // TODO(jevinjiang): consider optimize with assume_multiple op.
    Value in_vreg_shift = tiling_dim >= 0
                              ? modI(shift, ctx.target_shape[tiling_dim])
                              : mlirI32Const(0);
    Value vreg_shift =
        tiling_dim >= 0 ? divI(shift, ctx.target_shape[tiling_dim]) : shift;
    result = tiling_dim >= 0
                 ? rotateOnTilingDim(vregs, in_vreg_shift, axis, stride)
                 : vregs;
    int bound = vregs.dim(axis);
    if (bound <= ctx.max_sublanes_in_scratch / ctx.target_shape[0] &&
        bound >= kMinBoundToRotateWithScratch) {
      // Use static store + dynamic load to implement dynamic shift.
      if (auto scratch_ref = getInternalScratch(
              ctx, builder, op.getLoc(),
              {ctx.max_sublanes_in_scratch / ctx.target_shape[0],
               ctx.target_shape[0], ctx.target_shape[1]},
              vty.getElementType());
          succeeded(scratch_ref)) {
        auto cst_0 = mlirIndexConst(0);
        SmallVector<Value, 3> scratch_indices(3, cst_0);
        SmallVector<bool> sublane_mask(ctx.target_shape[0], true);
        const auto sublane_mask_attr =
            DenseBoolArrayAttr::get(op.getContext(), sublane_mask);
        chunks = split(result, axis);
        chunks[0].Each([&](absl::Span<const int64_t> idxs, Value *v) {
          // Static store vregs.
          for (int i = 0; i < bound; ++i) {
            scratch_indices[0] = mlirIndexConst(i);
            builder.create<tpu::StoreOp>(chunks[i](idxs), scratch_ref.value(),
                                         scratch_indices, sublane_mask_attr,
                                         /*mask=*/nullptr,
                                         /*sublane_stride=*/nullptr);
          }
          // Dynamic load vregs back from a circular buffer.
          for (int i = 0; i < bound; ++i) {
            scratch_indices[0] = builder.create<arith::IndexCastOp>(
                builder.getIndexType(),
                modI(builder.create<arith::SubIOp>(mlirI32Const(bound + i),
                                                   vreg_shift),
                     bound));
            chunks[i](idxs) =
                builder.create<tpu::LoadOp>(v->getType(), scratch_ref.value(),
                                            scratch_indices, sublane_mask_attr,
                                            /*sublane_stride=*/nullptr);
          }
        });
        return concatenate(chunks, axis);
      }
    }
    // Convert dynamic shift to log(bound) static ops.
    int roll_by = 1;
    Value cst_1 = mlirI32Const(1);
    while (bound > 0) {
      auto new_result = rotate(
          result,
          mlirI32Const(tiling_dim >= 0 ? roll_by * ctx.target_shape[tiling_dim]
                                       : roll_by),
          axis, /*stride=*/0);
      auto mask = builder.create<arith::CmpIOp>(
          arith::CmpIPredicate::ne,
          builder.create<vector::BroadcastOp>(
              i32_vreg, builder.create<arith::AndIOp>(vreg_shift, cst_1)),
          builder.create<arith::ConstantOp>(
              DenseElementsAttr::get(i32_vreg, builder.getI32IntegerAttr(0))));
      result.Each([&](absl::Span<const int64_t> idxs, Value *v) {
        *v = builder.create<arith::SelectOp>(mask, new_result(idxs), *v);
      });
      roll_by *= 2;
      bound /= 2;
      vreg_shift = divI(vreg_shift, 2);
    }
    return result;
  };

  xla::Array<Value> out_tiles(in_tiles.dimensions());
  const auto dim = op.getDimension();
  amount = modI(amount, vty.getDimSize(dim));

  if (op.getStride().has_value() && op.getStrideDimension().has_value()) {
    auto stride_dim = op.getStrideDimension().value();
    auto stride = op.getStride().value() % vty.getDimSize(stride_dim);
    if (stride_dim == dim) {
      return op.emitOpError(
          "Expected rotation dimension and stride dimension are not equal");
    }
    if (stride_dim == vty.getRank() - 1) {
      return op.emitOpError(
          "Not implemented: stride dimension is the minor most");
    } else if (stride_dim == vty.getRank() - 2) {
      if (dim != vty.getRank() - 1 || ctx.hardware_generation < 5) {
        return op.emitOpError(
            "Not implemented: only supported in TPU v5+ and rotation dimension "
            "is the minor most when stride dimension is the second minor most");
      }
      CHECK_GE(stride, 0);
      auto chunks = split(in_tiles, stride_dim);
      for (int64_t i = 0; i < chunks.size(); ++i) {
        Value base_amount = modI(addI(amount, ctx.target_shape[0] * i * stride),
                                 vty.getDimSize(dim));
        // After applying stride, we expect all shifts in a vreg are less or
        // equal to the vreg's lane count for now.
        if (auto base_amount_cst = getIntConst(base_amount, /*silent=*/true);
            succeeded(base_amount_cst)) {
          int64_t static_base_amount = base_amount_cst.value();
          auto max_shift_in_vreg = static_base_amount % ctx.target_shape[1] +
                                   (ctx.target_shape[0] - 1) * stride;
          if (max_shift_in_vreg > ctx.target_shape[1]) {
            return op.emitOpError("Not implemented: the max shift in a vreg ")
                   << max_shift_in_vreg << " is larger than the vreg's width "
                   << ctx.target_shape[1];
          }
        }
        SmallVector<int64_t> starts(out_tiles.num_dimensions(), 0);
        starts[stride_dim] = i;
        out_tiles.UpdateSlice(rotate(chunks[i], base_amount, dim, stride),
                              starts);
      }
    } else {
      // Split vregs along the stride dimension.
      auto chunks = split(in_tiles, stride_dim);
      for (int64_t i = 0; i < chunks.size(); ++i) {
        SmallVector<int64_t> starts(out_tiles.num_dimensions(), 0);
        starts[stride_dim] = i;
        out_tiles.UpdateSlice(
            rotate(chunks[i], addI(amount, i * stride), dim, /*stride=*/0),
            starts);
      }
    }
  } else {  // No stride.
    out_tiles = rotate(in_tiles, amount, dim, /*stride=*/0);
  }

  const RollVectorsOp rolled_op =
      assemble(builder, op.getResult().getType(), layout_out, out_tiles,
               ctx.target_shape);
  op.replaceAllUsesWith(rolled_op);
  op.erase();
  return success();
}

// TODO(b/347016737): deprecate the static rotate.
LogicalResult tpu_rotate_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null input layout");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  auto rotate_op = cast<tpu::RotateOp>(op);
  if (rotate_op.getAmount() < 0) {
    return op.emitOpError("Not implemented: shifting by negative amount");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  Value shift = builder.create<arith::ConstantOp>(
      builder.getIntegerAttr(builder.getI32Type(), rotate_op.getAmount()));
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  return rotate_rule_impl(ctx, rotate_op, shift, layout_in, layout_out);
}

LogicalResult tpu_dynamic_rotate_rule(RewriteContext &ctx, Operation &op,
                                      const ArrayRef<Layout> layouts_in,
                                      const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 2);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null layout for the value to rotate");
  }
  if (layouts_in[1].has_value()) {
    return op.emitOpError("Expected null layout for the shift");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  auto rotate_op = cast<tpu::DynamicRotateOp>(op);
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  return rotate_rule_impl(ctx, rotate_op, rotate_op.getAmount(), layout_in,
                          layout_out);
}

LogicalResult tpu_concatenate_rule(RewriteContext &ctx, Operation &op,
                                   const ArrayRef<Layout> layouts_in,
                                   const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), op.getNumOperands());
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(
      llvm::all_of(layouts_in, [](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout = *layouts_out.front();
  for (const Layout &l : layouts_in) {
    if (l != layout) {
      return op.emitOpError("Not implemented: Inconsistent layouts");
    }
  }
  OpBuilder builder(&op);
  auto concatenate_op = cast<tpu::ConcatenateOp>(op);
  const VectorType res_ty = concatenate_op.getResult().getType();
  const uint32_t dimension = concatenate_op.getDimension();
  if (dimension - res_ty.getRank() >= -2) {
    if (!layout.hasNativeTiling(ctx.target_shape) ||
        layout.offsets() != LayoutOffsets{0, 0}) {
      return op.emitOpError(
          "Not implemented: Only native tiling with offset (0, 0) is supported "
          "when concatenation along tiling dims.");
    }
    // Check if shapes of src and res are aligned to native tiling.
    auto check_aligned = [&](const VectorType &vty) {
      return vty.getRank() >= 2 &&
             *(vty.getShape().end() - 2) % *(layout.tiling().end() - 2) == 0 &&
             *(vty.getShape().end() - 1) % *(layout.tiling().end() - 1) == 0;
    };
    bool is_aligned = check_aligned(res_ty);
    int op_idx = 0;
    while (is_aligned && op_idx < op.getNumOperands()) {
      auto vty = dyn_cast<VectorType>(op.getOperand(op_idx++).getType());
      is_aligned = check_aligned(vty);
    }
    if (!is_aligned) {
      return op.emitOpError(
          "Not implemented: Only aligned shapes are supported when "
          "concatenation along tiling dims");
    }
  }

  SmallVector<xla::Array<Value>> tiles;
  tiles.reserve(concatenate_op->getNumOperands());
  for (Value operand : concatenate_op.getOperands()) {
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> t,
        disassemble(builder, layout, cast<TypedValue<VectorType>>(operand),
                    ctx.target_shape));
    tiles.emplace_back(std::move(t));
  }
  const xla::Array<Value> res_tiles = concatenate(tiles, dimension);
  op.replaceAllUsesWith(
      assemble(builder, res_ty, layout, res_tiles, ctx.target_shape));
  op.erase();
  return success();
}

LogicalResult tpu_iota_rule(RewriteContext &ctx, Operation &op,
                            const ArrayRef<Layout> layouts_in,
                            const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  tpu::IotaOp iota_op = cast<tpu::IotaOp>(op);
  VectorType vty = iota_op.getResult().getType();
  if (const auto int_ty = dyn_cast<IntegerType>(vty.getElementType());
      int_ty == nullptr || int_ty.getWidth() != 32) {
    return iota_op.emitOpError("Not implemented: Only 32-bit Iota supported");
  }
  if (!layout_out.hasNativeTiling(ctx.target_shape)) {
    return iota_op.emitOpError("Not implemented: Only native tiling supported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const auto native_vreg_ty,
      getNativeVregType(vty.getElementType(), ctx.target_shape));
  if (layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    return op.emitOpError("Not implemented: Only 2D layouts supported");
  }
  const SmallVector<int64_t> tile_array_shape =
      layout_out.tileArrayShape(vty.getShape(), ctx.target_shape);
  const std::optional<int32_t> dimension = iota_op.getDimension();
  if (!dimension.has_value()) {
    return op.emitOpError("Not implemented: null dimension");
  }
  if (*dimension == vty.getRank() - 1) {
    if (layout_out.offsets()[1] != 0) {
      return op.emitOpError("Not implemented: Unsupported offset");
    }
    const int64_t num_tiles = tile_array_shape[tile_array_shape.size() - 1];
    SmallVector<Value> tiles(num_tiles);
    auto vreg_iota = builder.create<tpu::IotaOp>(
        native_vreg_ty,
        /*dimension =*/builder.getI32IntegerAttr(1));
    for (int64_t i = 0; i < num_tiles; ++i) {
      auto offset = builder.create<arith::ConstantOp>(
          native_vreg_ty,
          DenseElementsAttr::get(
              native_vreg_ty,
              IntegerAttr::get(vty.getElementType(),
                               i * *(native_vreg_ty.getShape().end() - 1))));
      tiles[i] = builder.create<arith::AddIOp>(vreg_iota, offset);
    }
    xla::Array<Value> broadcasted_tiles(tile_array_shape);
    broadcasted_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      *v = tiles[*(idxs.end() - 1)];
    });
    op.replaceAllUsesWith(assemble(builder, vty, layout_out, broadcasted_tiles,
                                   ctx.target_shape));
    op.erase();
    return success();
  }
  if (*dimension == vty.getRank() - 2) {
    if (layout_out.offsets()[0] != 0) {
      return op.emitOpError("Not implemented: Unsupported offset");
    }
    const int64_t num_tiles = tile_array_shape[tile_array_shape.size() - 2];
    SmallVector<Value> tiles(num_tiles);
    auto vreg_iota = builder.create<tpu::IotaOp>(
        native_vreg_ty,
        /*dimension =*/builder.getI32IntegerAttr(0));
    for (int64_t i = 0; i < num_tiles; ++i) {
      auto offset = builder.create<arith::ConstantOp>(
          native_vreg_ty,
          DenseElementsAttr::get(
              native_vreg_ty,
              IntegerAttr::get(vty.getElementType(),
                               i * *(native_vreg_ty.getShape().end() - 2))));
      tiles[i] = builder.create<arith::AddIOp>(vreg_iota, offset);
    }
    xla::Array<Value> broadcasted_tiles(tile_array_shape);
    broadcasted_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      *v = tiles[*(idxs.end() - 2)];
    });
    op.replaceAllUsesWith(assemble(builder, vty, layout_out, broadcasted_tiles,
                                   ctx.target_shape));
    op.erase();
    return success();
  }
  // We take the iota over an untiled dimension.
  CHECK_LT(*dimension, vty.getRank());
  SmallVector<Value> tiles;
  tiles.reserve(vty.getDimSize(*dimension));
  for (int64_t i = 0; i < vty.getDimSize(*dimension); ++i) {
    tiles.push_back(builder.create<arith::ConstantOp>(
        native_vreg_ty,
        DenseElementsAttr::get(native_vreg_ty,
                               IntegerAttr::get(vty.getElementType(), i))));
  }
  xla::Array<Value> out_tiles(tile_array_shape);
  out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
    *v = tiles[idxs[*dimension]];
  });
  op.replaceAllUsesWith(
      assemble(builder, vty, layout_out, out_tiles, ctx.target_shape));
  op.erase();
  return success();
}

LogicalResult tpu_gather_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_in.offsets() != layout_out.offsets() ||
      llvm::any_of(layout_in.offsets(), [&](const LayoutOffset o) {
        return o.has_value() && o != 0;
      })) {
    return op.emitOpError("Not implemented: Only 2D layouts supported");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto gather_op = cast<tpu::GatherOp>(op);
  const VectorType vty = gather_op.getResult().getType();
  const uint32_t dimension = gather_op.getDimension();
  if (dimension + 2 < vty.getRank()) {
    return op.emitOpError("Not implemented: Unsupported dimension");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> in_tiles,
      disassemble(builder, layout_in, gather_op.getSource(), ctx.target_shape));
  const int64_t width = ctx.target_shape[2 - (vty.getRank() - dimension)];
  const ArrayRef<int32_t> indices(gather_op.getIndices());
  auto [num_sections, rem] = std::div(indices.size(), width);
  SmallVector<int32_t> segment_indices;
  if (rem == 0) {
    for (int64_t i = 0; i < width; ++i) {
      const int64_t offset = i - i % width;
      if (!(offset <= indices[i] && indices[i] < offset + width)) {
        return op.emitOpError("Not implemented: Cross-segment gather");
      }
    }
    for (int64_t i = width; i < indices.size(); ++i) {
      const int64_t offset = i - i % width;
      if (indices[i] != indices[i % width] + offset) {
        return op.emitOpError(
            "Not implemented: Indices varying between segments");
      }
    }
    segment_indices.assign(indices.begin(), indices.begin() + width);
  } else if (num_sections == 0) {  // Only one vreg.
    segment_indices.assign(indices.begin(), indices.end());
    segment_indices.append(width - indices.size(), 0);
  } else {
    return op.emitOpError("Not implemented: Not a multiple of target length");
  }
  xla::Array<Value> out_tiles(in_tiles.dimensions());
  if (dimension == vty.getRank() - 1) {
    // TODO(b/265133497): Remove the broadcast once 2nd minor works.
    const auto dyn_ix_ty =
        VectorType::get(ctx.target_shape, builder.getI32Type());
    // Broadcast indices to target_shape
    SmallVector<int32_t> dyn_ix_val;
    for (int64_t i = 0; i < ctx.target_shape[0]; ++i) {  // Broadcast
      dyn_ix_val.append(segment_indices);
    }
    FAILUREOR_ASSIGN_OR_RETURN(
        const BlockArgument dyn_ix_ref,
        appendConstant(ctx, DenseIntElementsAttr::get(dyn_ix_ty, dyn_ix_val)));
    auto all_sublanes = builder.getAttr<DenseBoolArrayAttr>(
        SmallVector<bool>(ctx.target_shape[1], true));
    auto dyn_ix = builder.create<tpu::LoadOp>(
        dyn_ix_ty, dyn_ix_ref,
        SmallVector<Value>(2, IdxConst(0, builder, op.getLoc())),
        /*sublane_mask=*/all_sublanes, /*sublane_stride=*/nullptr);
    out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      const Value in_tile = in_tiles(idxs);
      *v = builder.create<tpu::DynamicGatherOp>(in_tile.getType(), in_tile,
                                                dyn_ix, 1);
    });
  } else {
    TPU_ASSERT_EQ_OP(dimension, vty.getRank() - 2);
    const auto segment_indices_attr =
        builder.getAttr<DenseI32ArrayAttr>(segment_indices);
    out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      const Value in_tile = in_tiles(idxs);
      *v = builder.create<tpu::GatherOp>(in_tile.getType(), in_tile,
                                         segment_indices_attr, 0);
    });
  }
  gather_op.replaceAllUsesWith(
      assemble(builder, vty, layout_out, out_tiles, ctx.target_shape)
          .getOperation());
  gather_op.erase();
  return success();
}

LogicalResult tpu_region_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
    return op.emitOpError(
        "Not implemented: tpu.region_block with inputs or outputs");
  }
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  auto region_op = cast<tpu::RegionOp>(op);
  // We don't modify the op, but we do rewrite the branch bodies.
  return applyLayoutBlock(ctx, region_op.getRegion().getBlocks().front());
}

LogicalResult tpu_repeat_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    return op.emitOpError("Not implemented: Only 2D layouts supported");
  }
  if (layout_in != layout_out) {
    return op.emitOpError("Not implemented: Changing layout mid-repeat");
  }
  if (!layout_in.hasNaturalTopology(ctx.target_shape) ||
      layout_in.offsets() != LayoutOffsets{0, 0}) {
    return op.emitOpError("Not implemented: Non-trivial layouts unsupported");
  }
  OpBuilder builder(&op);
  tpu::RepeatOp repeat_op = cast<tpu::RepeatOp>(op);
  VectorType src_ty = repeat_op.getSource().getType();
  const uint32_t dim = repeat_op.getDimension();
  if (dim != src_ty.getRank() - 1) {
    return op.emitOpError(
        "Not implemented: Only repeats along the last dim supported");
  }
  if (src_ty.getShape().back() % ctx.target_shape.back() != 0) {
    return op.emitOpError("Not implemented: Only free repeats are suppported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> &in_vregs,
      disassemble(builder, layout_in, repeat_op.getSource(), ctx.target_shape));
  xla::Array<Value> out_vregs = repeat(in_vregs, repeat_op.getTimes(), dim);
  repeat_op->replaceAllUsesWith(
      assemble(builder, repeat_op.getResult().getType(), layout_out, out_vregs,
               ctx.target_shape)
          .getOperation());
  repeat_op->erase();
  return success();
}

LogicalResult vector_load_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  MLIRContext *const mlir_ctx = op.getContext();
  TPU_ASSERT_OP(llvm::none_of(layouts_in,
                              [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto load_op = cast<vector::LoadOp>(op);
  const auto memref_ty = getMemRefType(load_op.getBase());
  const auto vty = cast<VectorType>(load_op.getResult().getType());
  FAILUREOR_ASSIGN_OR_RETURN(
      VectorType target_ty,
      getNativeVregType(vty.getElementType(), ctx.target_shape));
  if (vty.getRank() == 0) {
    op.emitOpError("Not implemented: scalar loads from vmem");
  }
  const bool is_1d = vty.getRank() == 1;
  VectorLayout::ImplicitDim expected_dim =
      is_1d ? VectorLayout::ImplicitDim::kSecondMinor
            : VectorLayout::ImplicitDim::kNone;
  if (layout_out.implicit_dim() != expected_dim) {
    return op.emitOpError("Not implemented: unsupported layout");
  }
  using Tiling = std::array<int64_t, 2>;  // To avoid comma in macro
  FAILUREOR_ASSIGN_OR_RETURN(
      Tiling memref_tiling,
      getMemRefTiling(load_op.getBase(), ctx.target_shape));
  if (memref_tiling != layout_out.tiling() &&
      !(memref_tiling[0] == 1 && layout_out.tiling()[0] == 1 &&
        memref_tiling[1] % layout_out.tiling()[1] == 0)) {
    // Now we can handle the case when tiling is (1, TARGET_SHAPE.lanes).
    // TODO(b/295393167): need to support strided load for bitwidth < 32.
    if (layout_out.bitwidth() != 32 ||
        layout_out.tiling() != std::array<int64_t, 2>{1, ctx.target_shape[1]}) {
      return op.emitOpError("Not implemented");
    }
  }
  // TODO(apaszke): Check that loads are from vmem!

  bool can_support_unaligned_dynamic_index = false;
  bool must_support_unaligned_dynamic_index = false;
  if (load_op.getIndices().size() > 1) {
    auto second_minor_idx = load_op.getIndices().take_back(2)[0];
    if (failed(getIntConst(second_minor_idx, /*silent=*/true)) &&
        !isGuaranteedDivisible(second_minor_idx, memref_tiling[0])) {
      must_support_unaligned_dynamic_index = true;
    }
  }
  const SmallVector<int64_t> implicit_shape =
      layout_out.implicitShape(vty.getShape());
  const int64_t ss = implicit_shape[implicit_shape.size() - 2];
  int64_t sublane_stride = 1;
  // Handle special patterns that allow us to support more flexible loads.
  if (layout_out.bitwidth() == 32 &&
      layout_out.tiling() == std::array<int64_t, 2>{1, ctx.target_shape[1]} &&
      ss == 1) {
    // Loading a single row on the 2nd minor dim into the (1, 128) layout. We
    // can use sublane striding to perform the relayout as part of the load.
    sublane_stride = memref_tiling[0];
    can_support_unaligned_dynamic_index = true;
  } else {
    // Otherwise, if the memref has a short last dimension and is contiguous
    // all the tiled layouts become equivalent, so we can handle unaligned
    // dynamic indices without any special case.
    auto mem_layout = dyn_cast<TiledLayoutAttr>(memref_ty.getLayout());
    if (!mem_layout) {
      return op.emitOpError("Expected a tiled memref");
    }
    auto tile_strides = mem_layout.getTileStrides();
    if (memref_ty.getShape().back() == ctx.target_shape[1] &&
        tile_strides.take_back(2) == ArrayRef<int64_t>{1, 1}) {
      can_support_unaligned_dynamic_index = true;
    }
  }

  auto add_idx = [&](const Value &v, int64_t d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return IdxConst(cst.value() + d, builder, op.getLoc());
    }
    return builder.create<arith::AddIOp>(v, IdxConst(d, builder, op.getLoc()));
  };

  int tiled_dims = is_1d ? 1 : 2;
  Value base_addr = load_op.getBase();
  SmallVector<Value, 4> base_indices = load_op.getIndices();

  if (must_support_unaligned_dynamic_index) {
    if (!can_support_unaligned_dynamic_index) {
      return op.emitOpError(
          "Not implemented: dynamic load with unaligned indices");
    }
  } else {
    // Convert dynamic load to dynamic slice + static load. This saves us a
    // bunch of scalar core work.
    auto slice_result =
        sliceRef(builder, load_op.getBase(), load_op.getVectorType().getShape(),
                 load_op.getIndices(),
                 ArrayRef<int64_t>(memref_tiling).take_back(tiled_dims));
    if (failed(slice_result)) {
      return failure();
    }
    base_addr = slice_result->first;
    CHECK_EQ(slice_result->second.size(), base_indices.size());
    for (int i = 0; i < base_indices.size(); ++i) {
      base_indices[i] = IdxConst(slice_result->second[i], builder, op.getLoc());
    }
  }

  // TODO(jevinjiang): ideally we should update the base addr and use static
  // indices even for the cases that can skip alignment check. This can save us
  // a bunch of scalar core work.
  auto tile_base_idxs = ArrayRef<Value>(base_indices).take_back(tiled_dims);
  auto batch_base_idxs = ArrayRef<Value>(base_indices).drop_back(tiled_dims);
  const LayoutOffsets offsets = layout_out.offsets();
  AffineMap load_map;
  arith::ConstantOp padding;
  if (offsets[1] == std::nullopt) {
    return op.emitOpError(
        "Not implemented: Load replicated along lanes is unsupported");
  }
  if (offsets[0] == std::nullopt) {
    if (ss != 1) {
      return op.emitOpError(
          "Not implemented: Sublane-replicated load with size > 1 is "
          "unsupported");
    }
    if (!layout_out.hasNativeTiling(ctx.target_shape)) {
      return op.emitOpError("Not implemented");
    }
    // affine_map<(..., j) -> (0, j)
    load_map =
        AffineMap::get(memref_ty.getRank(), 0,
                       {getAffineConstantExpr(0, mlir_ctx),
                        getAffineDimExpr(memref_ty.getRank() - 1, mlir_ctx)},
                       mlir_ctx);
    FAILUREOR_ASSIGN_OR_RETURN(const TypedAttr zero_attr,
                               getZeroIntOrFloatAttr(vty.getElementType()));
    padding =
        builder.create<arith::ConstantOp>(vty.getElementType(), zero_attr);
  }

  xla::Array<Value> tiles(
      layout_out.tileArrayShape(vty.getShape(), ctx.target_shape));
  const std::array<int64_t, 2> vreg_slice =
      layout_out.vregSlice(ctx.target_shape);
  const int64_t num_dims = vty.getRank();
  const int64_t num_batch_dims = num_dims - (is_1d ? 1 : 2);
  const absl::Status status =
      tiles.EachStatus([&](absl::Span<const int64_t> tile_idxs, Value * /*v*/) {
        CHECK_EQ(num_dims, tile_idxs.size());
        SmallVector<Value> idxs(tile_idxs.size());
        for (int64_t i = 0; i < num_batch_dims; ++i) {
          idxs[i] = add_idx(batch_base_idxs[i], tile_idxs[i]);
        }
        const auto base_l = tile_base_idxs.back();
        const int64_t lidx = tile_idxs[num_dims - 1];
        idxs[num_dims - 1] =
            add_idx(base_l, lidx * vreg_slice[1] - offsets[1].value_or(0));
        if (!is_1d) {
          CHECK_EQ(tile_base_idxs.size(), 2);
          const auto base_s = tile_base_idxs.front();
          const int64_t sidx = tile_idxs[num_dims - 2];
          idxs[num_dims - 2] =
              add_idx(base_s, sidx * vreg_slice[0] - offsets[0].value_or(0));
        }
        TPU_ASSERT_OP(tile_idxs[num_dims - 1] + ctx.target_shape[1] <=
                      memref_ty.getShape()[num_dims - 1]);
        std::unique_ptr<VRegDataBounds> bounds = layout_out.tileDataBounds(
            mlir_ctx, vty.getShape(), toArrayRef(tile_idxs), ctx.target_shape,
            /*allow_replicated =*/{true, false});
        Operation *tile;
        if (bounds->maskVariesAlong(Direction::kSublanes, ctx.target_shape)) {
          CHECK(offsets[0].has_value());
          tile = builder.create<tpu::LoadOp>(
              target_ty, base_addr, idxs,
              bounds->getSublaneMask(mlir_ctx, ctx.target_shape),
              builder.getI32IntegerAttr(sublane_stride));
        } else {
          if (load_map) {
            CHECK(padding);
            if (layout_out.bitwidth() != 32) {
              load_op.emitOpError("Not implemented");
              return absl::UnimplementedError("");
            }
            tile = builder.create<vector::TransferReadOp>(
                target_ty, base_addr, idxs, load_map, padding, nullptr,
                nullptr);
          } else {
            const SmallVector<bool> sublane_mask(ctx.target_shape[0], true);
            const auto sublane_mask_attr =
                DenseBoolArrayAttr::get(mlir_ctx, sublane_mask);
            tile = builder.create<tpu::LoadOp>(
                target_ty, base_addr, idxs, sublane_mask_attr,
                builder.getI32IntegerAttr(sublane_stride));
          }
        }
        tiles(tile_idxs) = tile->getResult(0);
        return absl::OkStatus();
      });
  if (!status.ok()) {
    return failure();
  }
  load_op->replaceAllUsesWith(
      assemble(builder, vty, layout_out, std::move(tiles), ctx.target_shape));
  load_op->erase();
  return success();
}

LogicalResult arith_constant_rule(RewriteContext &ctx, Operation &op,
                                  const ArrayRef<Layout> layouts_in,
                                  const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto constant_op = cast<arith::ConstantOp>(op);
  auto vty = dyn_cast<VectorType>(op.getResult(0).getType());
  if (vty) {
    if (!layouts_out.front().has_value()) {
      return op.emitOpError(
          "Expected non-null output layout for vector constant");
    }
    const VectorLayout &layout_out = *layouts_out.front();
    DenseElementsAttr value = cast<DenseElementsAttr>(constant_op.getValue());
    FAILUREOR_ASSIGN_OR_RETURN(
        const VectorType target_vty,
        getNativeVregType(vty.getElementType(), ctx.target_shape));
    if (value.isSplat()) {
      if (layout_out.offsets() != LayoutOffsets{std::nullopt, std::nullopt}) {
        return op.emitOpError(
            "Not implemented: Non-replicated splat constants");
      }
      auto new_value =
          DenseElementsAttr::get(target_vty, value.getSplatValue<Attribute>());
      const auto tile =
          builder.create<arith::ConstantOp>(target_vty, new_value);
      const xla::Array<Value> tiles(
          layout_out.tileArrayShape(vty.getShape(), ctx.target_shape),
          tile->getResult(0));
      op.replaceAllUsesWith(assemble(builder, vty, layout_out, std::move(tiles),
                                     ctx.target_shape));
      op.erase();
      return success();
    }
    // !value.isSplat()
    if (getTypeBitwidth<true>(vty.getElementType()) != 32) {
      return op.emitOpError(
          "Not implemented: Only 32-bit non-splat constants are supported");
    }
    FAILUREOR_ASSIGN_OR_RETURN(const BlockArgument ref,
                               appendConstant(ctx, value));
    auto load_op = builder.create<vector::LoadOp>(
        vty, ref,
        SmallVector<Value>(vty.getRank(), IdxConst(0, builder, op.getLoc())));
    op.replaceAllUsesWith(ArrayRef<Value>{load_op.getResult()});
    op.erase();
    const SmallVector<Layout> vector_load_in_layouts(vty.getRank() + 1);
    return vector_load_rule(ctx, *load_op, vector_load_in_layouts,
                            {VectorLayout(/*bitwidth=*/32, /*offsets=*/{0, 0},
                                          /*tiling=*/ctx.target_shape)});
  }
  return op.emitOpError("Not implemented: Unsupported arith.const type: ")
         << op.getResult(0).getType();
}

LogicalResult vector_broadcast_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const Layout &maybe_layout_in = layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  vector::BroadcastOp broadcast_op = cast<vector::BroadcastOp>(op);
  const VectorType dst_ty = broadcast_op.getResult().getType();
  const ArrayRef<int64_t> dst_shape = dst_ty.getShape();
  const SmallVector<int64_t> dst_tiles_shape =
      layout_out.tileArrayShape(dst_shape, ctx.target_shape);
  const SmallVector<int64_t> dst_tiles_implicit_shape =
      layout_out.tileArrayImplicitShape(dst_shape, ctx.target_shape);
  if (auto src = dyn_cast<TypedValue<VectorType>>(broadcast_op.getSource())) {
    VectorType src_ty = src.getType();
    TPU_ASSERT_OP(maybe_layout_in.has_value());
    const VectorLayout &layout_in = *maybe_layout_in;
    if (layout_in.implicit_dim() != layout_out.implicit_dim()) {
      return op.emitOpError(
          "Not implemented: Changing implicit dims mid-broadcast");
    }
    const LayoutOffsets offsets_in = layout_in.offsets();
    const LayoutOffsets offsets_out = layout_out.offsets();
    if (layout_in.tiling() != layout_out.tiling()) {
      return op.emitOpError("Not implemented: Changing tiling mid-broadcast");
    }
    auto tiling = layout_in.tiling();

    const int64_t expand_rank = dst_ty.getRank() - src_ty.getRank();
    const ArrayRef<int64_t> src_shape = src_ty.getShape();

    SmallVector<int64_t> src_implicit_shape_padded;
    // `is_logical_broadcast` stores whether each dimension of the implicit
    // shape of the result is a broadcast. E.g. if the implicit shape goes from
    // (2, 1, 3) to (4, 2, 5, 3) it's (true, false, true, false).
    SmallVector<bool> is_logical_broadcast;
    src_implicit_shape_padded.reserve(dst_shape.size() +
                                      layout_in.num_implicit_dims());
    is_logical_broadcast.reserve(dst_shape.size() +
                                 layout_in.num_implicit_dims());
    src_implicit_shape_padded.append(expand_rank, 1);
    src_implicit_shape_padded.append(src_shape.begin(), src_shape.end());
    for (auto [i, o] : llvm::zip(src_implicit_shape_padded, dst_shape)) {
      TPU_ASSERT_OP(i == o || i == 1);  // Verifier should guarantee this.
      is_logical_broadcast.push_back(i != o);
    }
    layout_in.insertImplicit<int64_t>(src_implicit_shape_padded, 1);
    layout_in.insertImplicit<bool>(is_logical_broadcast, false);

    // Verify that the offsets are valid.
    for (auto [is_logical_broadcast_on_dim, in_off, out_off] :
         llvm::zip_equal(ArrayRef(is_logical_broadcast).take_back(2),
                         offsets_in, offsets_out)) {
      if (is_logical_broadcast_on_dim) {
        if (out_off.has_value()) {
          // There's no reason to ever assign a non-replicated offset to a
          // broadcasted dimension in the output.
          return op.emitOpError(
              // TODO(tlongeri): This should never be implemented but the fuzzed
              //                 tests expect a NotImplementedError, which
              //                 is raised with a "Not implemented" (see
              //                 NotImplementedDetector in tpu_ext.cc). Fix.
              "Not implemented: Broadcast output expected to have replicated "
              "offsets.");
        }
      } else {  // !is_logical_broadcast_on_dim
        if (in_off != out_off) {
          return op.emitOpError(
              "Not implemented: Changing offsets mid-broadcast");
        }
      }
    }

    // `needs_physical_broadcast` specifies whether we need to broadcast vregs
    // vregs in the sublane and lane dimensions. We only need to do this if the
    // corresponding dimension of the implicit shape is logically broadcast and
    // if the input vregs are not already replicated along this dimension.
    const std::array<bool, 2> needs_physical_broadcast{
        *(is_logical_broadcast.end() - 2) && offsets_in[0].has_value(),
        *(is_logical_broadcast.end() - 1) && offsets_in[1].has_value()};
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> src_tiles,
        disassemble(builder, layout_in, src, ctx.target_shape,
                    /*use_implicit_shape=*/true));
    xla::Array<Value> dst_tiles(dst_tiles_implicit_shape);
    if (needs_physical_broadcast == std::array{false, false}) {  // No-op
      SmallVector<int64_t> reshape_dims(expand_rank, 1);
      const absl::Span<const int64_t> src_tiles_dims = src_tiles.dimensions();
      reshape_dims.append(src_tiles_dims.begin(), src_tiles_dims.end());
      src_tiles.Reshape(reshape_dims);
      dst_tiles.Each([&](const absl::Span<const int64_t> dst_idx, Value *tile) {
        const SmallVector<int64_t> src_idx = llvm::map_to_vector(
            llvm::zip_equal(dst_idx, is_logical_broadcast), [](auto tup) {
              auto [i, is_logical_broadcast_on_dim] = tup;
              return is_logical_broadcast_on_dim ? 0 : i;
            });
        *tile = src_tiles(src_idx);
      });
    } else {
      if (layout_in.bitwidth() != 32) {
        return op.emitOpError(
            "Not implemented: Only 32-bit broadcast supported");
      }
      if (tiling[1] != ctx.target_shape[1]) {
        return op.emitOpError("Not implemented: unsupported tiling");
      }
      int64_t num_tiles = layout_in.tilesPerVreg(ctx.target_shape);
      if (needs_physical_broadcast ==
          std::array{true, false}) {  // Sublane broadcast
        if (num_tiles != 1) {
          return op.emitOpError(
              "Not implemented: Only native tiling supported");
        }
        TPU_ASSERT_EQ_OP(*(src_tiles.dimensions().end() - 2), 1);
        TPU_ASSERT_OP(offsets_in[0].has_value());
        const int64_t offset = *offsets_in[0];
        const DenseI32ArrayAttr indices = builder.getDenseI32ArrayAttr(
            SmallVector<int32_t>(ctx.target_shape[0], offset));
        src_tiles.Each([&](const absl::Span<const int64_t> src_idx,
                           Value *const src_tile) {
          SmallVector<int64_t> dst_starts(dst_tiles_implicit_shape.size());
          SmallVector<int64_t> dst_limits(dst_tiles_implicit_shape.size());
          for (int64_t i = 0; i < dst_tiles.num_dimensions(); ++i) {
            if (i < expand_rank || is_logical_broadcast[i]) {
              dst_starts[i] = 0;
              dst_limits[i] = dst_tiles_implicit_shape[i];
            } else {
              dst_starts[i] = src_idx[i - expand_rank];
              dst_limits[i] = dst_starts[i] + 1;
            }
          }
          updateSlice<Value>(dst_tiles,
                             builder.create<tpu::GatherOp>(
                                 src_tile->getType(), *src_tile, indices, 0),
                             dst_starts, dst_limits);
        });
      } else if (needs_physical_broadcast ==
                 std::array{false, true}) {  // Lane broadcast
        TPU_ASSERT_EQ_OP(*(src_tiles.dimensions().end() - 1), 1);
        TPU_ASSERT_OP(offsets_in[1].has_value());
        const int64_t offset = *offsets_in[1];
        const auto idx_ty =
            VectorType::get(ctx.target_shape, builder.getI32Type());
        auto idx_const = builder.create<arith::ConstantOp>(
            broadcast_op.getLoc(), idx_ty,
            DenseElementsAttr::get(idx_ty, builder.getI32IntegerAttr(offset)));
        int64_t sublanes_per_tile = layout_in.sublanesPerTile(ctx.target_shape);
        DenseI32ArrayAttr sublane_pattern;
        if (num_tiles != 1) {
          SmallVector<int32_t> pattern;
          pattern.reserve(ctx.target_shape[0]);
          for (int32_t t = 0; t < num_tiles; ++t) {
            for (int32_t i = 0; i < sublanes_per_tile; ++i) {
              pattern.push_back(i);
            }
          }
          sublane_pattern = builder.getDenseI32ArrayAttr(pattern);
        }
        src_tiles.Each([&](const absl::Span<const int64_t> src_idx,
                           Value *const src_tile) {
          SmallVector<int64_t> dst_starts(dst_tiles_implicit_shape.size());
          SmallVector<int64_t> dst_limits(dst_tiles_implicit_shape.size());
          for (int64_t i = 0; i < dst_tiles.num_dimensions(); ++i) {
            if (i < expand_rank || is_logical_broadcast[i]) {
              dst_starts[i] = 0;
              dst_limits[i] = dst_tiles_implicit_shape[i];
            } else {
              dst_starts[i] = src_idx[i - expand_rank];
              dst_limits[i] = dst_starts[i] + 1;
            }
          }
          Value res_vreg = builder.create<tpu::DynamicGatherOp>(
              broadcast_op.getLoc(), src_tile->getType(), *src_tile, idx_const,
              /*dimension=*/1);
          if (num_tiles != 1) {
            res_vreg = builder.create<tpu::GatherOp>(
                broadcast_op.getLoc(), res_vreg.getType(), res_vreg,
                sublane_pattern, 0);
          }
          updateSlice<Value>(dst_tiles, res_vreg, dst_starts, dst_limits);
        });
      } else {
        TPU_ASSERT_OP((needs_physical_broadcast == std::array{true, true}));
        return op.emitOpError(
            "Not implemented: Broadcast in both sublanes and lanes");
      }
    }
    broadcast_op.replaceAllUsesWith(assemble(builder, dst_ty, layout_out,
                                             dst_tiles, ctx.target_shape,
                                             /*use_implicit_shape=*/true)
                                        .getOperation());
    broadcast_op.erase();
    return success();
  } else if (layout_out.bitwidth() == 32 &&
             broadcast_op.getSourceType().getIntOrFloatBitWidth() == 1) {
    // Broadcasting the i1 scalar involves first converting i1 to i32, followed
    // by broadcasting i32 to the target shape. Finally, the comparison with 0s
    // yields the vmask.
    auto src_i32 = builder.create<arith::ExtUIOp>(
        broadcast_op.getLoc(), builder.getI32Type(), broadcast_op.getSource());
    FAILUREOR_ASSIGN_OR_RETURN(
        const VectorType native_vreg_ty,
        getNativeVregType(src_i32.getType(), ctx.target_shape));
    auto tile_i32 =
        builder.create<vector::BroadcastOp>(native_vreg_ty, src_i32);
    auto zeros = builder.create<arith::ConstantOp>(
        broadcast_op.getLoc(), tile_i32.getType(),
        DenseElementsAttr::get(tile_i32.getType(),
                               builder.getI32IntegerAttr(0)));
    auto tile =
        builder.create<arith::CmpIOp>(arith::CmpIPredicate::ne, tile_i32, zeros)
            .getResult();
    const xla::Array<Value> dst_tiles(dst_tiles_shape, tile);
    broadcast_op.replaceAllUsesWith(
        assemble(builder, dst_ty, layout_out, dst_tiles, ctx.target_shape)
            .getOperation());
    broadcast_op.erase();
    return success();
  } else if (layout_out.bitwidth() < 32) {
    CHECK_EQ(layout_out.bitwidth(),
             broadcast_op.getSourceType().getIntOrFloatBitWidth());
    // Broadcasting the scalar with narrower type involves first packing (32 /
    // bitwidth) copies to i32, followed by broadcasting i32 to the target
    // shape. Finally, bitcast i32 vector back to the original narrower type
    // vector.
    auto loc = broadcast_op.getLoc();
    auto src_ty = broadcast_op.getSourceType();
    auto bitwidth = src_ty.getIntOrFloatBitWidth();
    auto unpacked_src = broadcast_op.getSource();
    if (!src_ty.isSignlessInteger(bitwidth)) {
      unpacked_src = builder.create<arith::BitcastOp>(
          loc, builder.getIntegerType(bitwidth), unpacked_src);
    }
    auto src_i32 =
        builder.create<arith::ExtUIOp>(loc, builder.getI32Type(), unpacked_src)
            .getResult();
    for (int i = 1; i < (32 / bitwidth); ++i) {
      auto shift_width = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(builder.getI32Type(), i * bitwidth));
      src_i32 = builder.create<arith::OrIOp>(
          loc, src_i32,
          builder.create<arith::ShLIOp>(loc, src_i32, shift_width));
    }
    FAILUREOR_ASSIGN_OR_RETURN(
        const VectorType i32_vreg_ty,
        getNativeVregType(src_i32.getType(), ctx.target_shape));
    auto tile_i32 = builder.create<vector::BroadcastOp>(i32_vreg_ty, src_i32);

    FAILUREOR_ASSIGN_OR_RETURN(const VectorType native_vreg_ty,
                               getNativeVregType(src_ty, ctx.target_shape));
    auto tile = builder.create<tpu::BitcastVregOp>(native_vreg_ty, tile_i32);

    const xla::Array<Value> dst_tiles(dst_tiles_shape, tile);
    broadcast_op.replaceAllUsesWith(
        assemble(builder, dst_ty, layout_out, dst_tiles, ctx.target_shape)
            .getOperation());
    broadcast_op.erase();
    return success();
  } else {
    FAILUREOR_ASSIGN_OR_RETURN(
        const VectorType native_vreg_ty,
        getNativeVregType(broadcast_op.getSourceType(), ctx.target_shape));
    auto tile = builder.create<vector::BroadcastOp>(native_vreg_ty,
                                                    broadcast_op.getSource());
    const xla::Array<Value> dst_tiles(dst_tiles_shape, tile);
    broadcast_op.replaceAllUsesWith(
        assemble(builder, dst_ty, layout_out, dst_tiles, ctx.target_shape)
            .getOperation());
    broadcast_op.erase();
    return success();
  }
}

// Returns slice of vregs containing a given slice of elements, obtained from
// the result of a vector.extract or vector.extract_strided_slice op.
//
// Takes offsets and sizes describing the slice of elements. If their size is
// less than the rank of the input vector, they describe a prefix i.e. they
// apply to the first (majormost) dimensions and the remaining dimensions are
// not sliced.
//
// Args:
// - ctx:        Rewrite context (for disassembling, which may create an op).
// - op:         Source vector.extract or vector.extract_strided_slice op.
// - offsets:    Prefix of offsets of slice of elements. Must have the same size
//               as sizes.
// - sizes:      Prefix of sizes of slice of elements. Must have the same size
//               as offsets.
// - layout_in:  Layout of src_vector.
// - layout_out: Layout that will be used to reassemble the slice (by caller).
//               Used only to check that the reassembling is valid.
FailureOr<xla::Array<Value>> vector_extract_slice_impl(
    RewriteContext &ctx, Operation &op, const ArrayRef<int64_t> sizes,
    const ArrayRef<int64_t> offsets, const VectorLayout &layout_in,
    const VectorLayout &layout_out) {
  if (layout_in.tiling() != layout_out.tiling() ||
      layout_in.bitwidth() != layout_out.bitwidth()) {
    return op.emitOpError(
        "Expected layout_in and layout_out tiling and packing to match");
  }

  // Both extract_strided_slice and extract have their input vector at index 0
  // and a single result.
  CHECK((isa<vector::ExtractOp, vector::ExtractStridedSliceOp>(op)));
  auto src_vector = cast<TypedValue<VectorType>>(op.getOperand(0));
  auto result = cast<TypedValue<VectorType>>(op.getResult(0));

  const VectorType dst_ty = result.getType();
  if (layout_in.implicit_dim() != layout_out.implicit_dim() &&
      !(layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
        layout_out.implicit_dim() == VectorLayout::ImplicitDim::kSecondMinor &&
        dst_ty.getRank() == 1)) {
    return op.emitOpError(
        "Unexpected change in implicit dimension that may not be a no-op");
  }

  const ArrayRef<int64_t> src_vector_shape = src_vector.getType().getShape();
  const int64_t src_vector_rank = src_vector_shape.size();
  const int64_t num_indices = offsets.size();
  TPU_ASSERT_EQ_OP(num_indices, sizes.size());

  SmallVector<int64_t> full_sizes;
  full_sizes.reserve(src_vector_rank + layout_in.num_implicit_dims());
  full_sizes.append(sizes.begin(), sizes.end());
  full_sizes.append(src_vector_shape.begin() + num_indices,
                    src_vector_shape.end());
  layout_in.insertImplicit<int64_t>(full_sizes, 1);

  SmallVector<int64_t> full_offsets;
  full_offsets.reserve(src_vector_rank + layout_in.num_implicit_dims());
  full_offsets.append(offsets.begin(), offsets.end());
  full_offsets.append(src_vector_rank - num_indices, 0);
  layout_in.insertImplicit<int64_t>(full_offsets, 0);

  // We currently only support no-op cases - that is, those where we effectively
  // just extract a slice of vregs without doing any operations (e.g. shifts) on
  // them.
  // TODO(tlongeri): VectorLayout enforces that the offsets must fall in the
  // first tile of each vreg. That means a no-op would not result in a valid
  // layout if the index offset falls within a different tile in the vreg. Do we
  // want to loosen this restriction or add shifts? This is the only non-no-op
  // that might make sense to support - otherwise we should expect
  // infer-vector-layout to assign no-op layouts and have the burden of any
  // shifts that might be needed later fall on relayout.
  for (auto [index_offset, in_offset, vreg_slice, out_offset] : llvm::zip_equal(
           ArrayRef<int64_t>(full_offsets).take_back(2), layout_in.offsets(),
           layout_in.vregSlice(ctx.target_shape), layout_out.offsets())) {
    if (in_offset.has_value() != out_offset.has_value()) {
      return op.emitOpError(
          "Unexpected mismatch in replication between input and output "
          "layouts");
    }
    if (in_offset.has_value() &&
        (index_offset + *in_offset) % vreg_slice != *out_offset) {
      return op.emitOpError("Not implemented: Only no-op tiles");
    }
  }

  const std::array<int64_t, 2> vreg_slice =
      layout_in.vregSlice(ctx.target_shape);
  SmallVector<int64_t> slice_tiled_starts(full_offsets);
  *(slice_tiled_starts.end() - 2) =
      (layout_in.offsets()[0].value_or(0) + *(full_offsets.end() - 2)) /
      vreg_slice[0];
  *(slice_tiled_starts.end() - 1) =
      (layout_in.offsets()[1].value_or(0) + *(full_offsets.end() - 1)) /
      vreg_slice[1];
  layout_in.eraseImplicit(slice_tiled_starts);
  SmallVector<int64_t> slice_tiled_limits(full_offsets);
  for (int64_t i = 0; i < full_offsets.size() - layout_in.layout_rank(); ++i) {
    slice_tiled_limits[i] += full_sizes[i];
  }
  *(slice_tiled_limits.end() - 2) =
      llvm::divideCeil(layout_in.offsets()[0].value_or(0) +
                           *(full_offsets.end() - 2) + *(full_sizes.end() - 2),
                       vreg_slice[0]);
  *(slice_tiled_limits.end() - 1) =
      llvm::divideCeil(layout_in.offsets()[1].value_or(0) +
                           *(full_offsets.end() - 1) + *(full_sizes.end() - 1),
                       vreg_slice[1]);
  layout_in.eraseImplicit(slice_tiled_limits);

  OpBuilder builder(&op);
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> input_tiles,
      disassemble(builder, layout_in, src_vector, ctx.target_shape));
  return input_tiles.Slice(slice_tiled_starts, slice_tiled_limits);
}

LogicalResult vector_extract_rule(RewriteContext &ctx, Operation &op,
                                  const ArrayRef<Layout> layouts_in,
                                  const ArrayRef<Layout> layouts_out) {
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  vector::ExtractOp extract_op = cast<vector::ExtractOp>(op);
  if (extract_op.hasDynamicPosition()) {
    return op.emitOpError("Not implemented: dynamic indices");
  }
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  if (layout_in.bitwidth() != 32) {
    return op.emitOpError(
        "Not implemented: Only 32-bit vector.extract supported");
  }
  const VectorType res_vty =
      dyn_cast<VectorType>(extract_op.getResult().getType());
  if (res_vty != nullptr) {
    TPU_ASSERT_OP(layouts_out.front().has_value());
    const VectorLayout &layout_out = *layouts_out.front();
    const int64_t num_indices = extract_op.getStaticPosition().size();
    const SmallVector<int64_t> sizes(num_indices, 1);
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> dst_vregs,
        vector_extract_slice_impl(ctx, *extract_op, sizes,
                                  extract_op.getStaticPosition(), layout_in,
                                  *layouts_out.front()));
    // Squeeze leading singleton dimensions.
    TPU_ASSERT_EQ_OP(res_vty.getRank(),
                     extract_op.getSourceVectorType().getRank() - num_indices);
    TPU_ASSERT_OP(
        llvm::all_of(toArrayRef(dst_vregs.dimensions()).take_front(num_indices),
                     [](const int64_t d) { return d == 1; }));
    // Copy dims to temporary before passing to xla::Array::Reshape - it cannot
    // take a pointer to its own data.
    dst_vregs.Reshape(SmallVector<int64_t>(
        toArrayRef(dst_vregs.dimensions()).drop_front(num_indices)));
    op.replaceAllUsesWith(
        assemble(builder, res_vty, layout_out, dst_vregs, ctx.target_shape)
            .getOperation());
    op.erase();
    return success();
  } else {
    for (int64_t i : extract_op.getStaticPosition()) {
      if (i != 0) {
        return op.emitOpError(
            "Not implemented: Only 0 indices supported for scalar results");
      }
    }
    if (layout_in.offsets() != LayoutOffsets{0, 0}) {
      return op.emitOpError("Not implemented: Unsupported layout");
    }
    FAILUREOR_ASSIGN_OR_RETURN(
        const xla::Array<Value> vregs,
        disassemble(builder, layout_in, extract_op.getVector(),
                    ctx.target_shape));
    TPU_ASSERT_GT_OP(vregs.num_elements(), 0);
    extract_op.replaceAllUsesWith(
        builder
            .create<vector::ExtractOp>(op.getLoc(), *vregs.data(),
                                       ArrayRef<int64_t>{0, 0})
            .getResult());
  }
  extract_op.erase();
  return success();
}

LogicalResult vector_contract_rule(RewriteContext &ctx, Operation &op,
                                   const ArrayRef<Layout> layouts_in,
                                   const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 3);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(
      llvm::all_of(layouts_in, [&](const Layout &l) { return l.has_value(); }));
  TPU_ASSERT_OP(layouts_out.front().has_value());
  MLIRContext *const mlir_ctx = ctx.getMLIRContext();
  Builder builder(mlir_ctx);
  auto vector_contract_op = cast<vector::ContractionOp>(op);
  // TODO(tlongeri): There is some unnecessary uniquing happening but not sure
  // if it can be avoided without sacrificing readability rather severely.
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
  const auto indexing_maps = vector_contract_op->getAttr("indexing_maps");
  if (indexing_maps != matmul_indexing_maps &&
      indexing_maps != matmul_indexing_maps_transposed) {
    return vector_contract_op->emitOpError(
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
  if (vector_contract_op->getAttr("iterator_types") != matmul_iterator_types) {
    return vector_contract_op->emitOpError(
        "Not implemented: Non-matmul iterator_types");
  }
  const bool transpose_lhs =
      false;  // TODO(apaszke): Support that in the affine maps
  return matmul_rule_impl(ctx, *vector_contract_op, transpose_lhs,
                          transpose_rhs, *layouts_in[0], *layouts_in[1],
                          *layouts_in[2], *layouts_out[0]);
}

LogicalResult vector_extract_strided_slice_rule(
    RewriteContext &ctx, Operation &op, const ArrayRef<Layout> layouts_in,
    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto extract_strided_slice_op = cast<vector::ExtractStridedSliceOp>(op);

  auto I64ArrayToSmallVector = [&](const ArrayAttr array_attr) {
    return llvm::map_to_vector(array_attr, [](Attribute attr) {
      return cast<IntegerAttr>(attr).getValue().getSExtValue();
    });
  };

  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> dst_vregs,
      vector_extract_slice_impl(
          ctx, *extract_strided_slice_op,
          I64ArrayToSmallVector(extract_strided_slice_op.getSizes()),
          I64ArrayToSmallVector(extract_strided_slice_op.getOffsets()),
          layout_in, layout_out));
  op.replaceAllUsesWith(assemble(builder,
                                 extract_strided_slice_op.getResult().getType(),
                                 layout_out, dst_vregs, ctx.target_shape)
                            .getOperation());
  op.erase();
  return success();
}

LogicalResult vector_multi_reduction_rule(RewriteContext &ctx, Operation &op,
                                          const ArrayRef<Layout> layouts_in,
                                          const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 2);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(
      llvm::all_of(layouts_in, [&](const Layout &l) { return l.has_value(); }));
  const VectorLayout &src_layout = *layouts_in[0];
  const VectorLayout &acc_layout = *layouts_in[1];
  const VectorLayout &dst_layout = *layouts_out[0];
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto multi_reduction_op = cast<vector::MultiDimReductionOp>(op);
  const VectorType src_ty = multi_reduction_op.getSourceVectorType();
  int64_t src_rank = src_ty.getRank();
  const auto res_ty = dyn_cast<VectorType>(multi_reduction_op.getDestType());
  if (res_ty == nullptr) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Can only reduce into vectors");
  }
  // Op definition enforces that accumulator type must match result type
  auto acc = cast<TypedValue<VectorType>>(multi_reduction_op.getAcc());
  TPU_ASSERT_OP(layouts_out.front().has_value());

  const ArrayAttr dim_attrs = multi_reduction_op.getReductionDims();
  SmallVector<int64_t> dims;
  dims.reserve(dim_attrs.size());
  for (const Attribute dim_attr : dim_attrs) {
    dims.push_back(cast<IntegerAttr>(dim_attr).getValue().getSExtValue());
  }
  std::sort(dims.begin(), dims.end());

  // Make sure that the accumulator is a splat of the neutral value
  if (acc_layout.offsets() != LayoutOffsets{std::nullopt, std::nullopt}) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Only replicated accumulator supported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> acc_vregs,
      disassemble(builder, acc_layout, acc, ctx.target_shape));
  auto acc_def = dyn_cast_if_present<arith::ConstantOp>(
      acc_vregs.begin()->getDefiningOp());
  if (acc_def == nullptr) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Only constant accumulator supported");
  }
  if (!src_ty.getElementType().isF32()) {
    return multi_reduction_op.emitOpError(
               "Not implemented: Only FP32 reductions supported, but got ")
           << src_ty;
  }
  // Element types of source, dest, acc match (by multi_dim reduction's
  // definition), so we expect an f32 constant
  const auto acc_def_value = dyn_cast<DenseFPElementsAttr>(acc_def.getValue());
  if (acc_def_value == nullptr || !acc_def_value.isSplat()) {
    return multi_reduction_op.emitOpError("Expected a splat constant");
  }
  TPU_ASSERT_OP(acc_def_value.getElementType().isF32());
  const auto val = acc_def_value.getSplatValue<float>();
  FloatAttr neutral;
  switch (multi_reduction_op.getKind()) {
    case vector::CombiningKind::ADD:
      neutral = builder.getF32FloatAttr(0);
      break;
    case vector::CombiningKind::MAXIMUMF: {
      // TODO(b/322836633): The semantics of maximumf don't match the lowering
      // for older TPU versions because older TPU versions don't respect the
      // -0.0 vs +0.0 ordering.
      neutral = builder.getFloatAttr(
          builder.getF32Type(),
          APFloat::getInf(APFloat::IEEEsingle(), /*Negative=*/true));
    } break;
    default:
      return multi_reduction_op.emitOpError(
          "Not implemented: unsupported kind");
  }
  if (val != neutral.getValueAsDouble()) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Only neutral accumulator supported");
  }

  std::array<bool, 2> reduces;
  switch (src_layout.implicit_dim()) {
    case VectorLayout::ImplicitDim::kNone:
      reduces = {
          std::find(dims.begin(), dims.end(), src_rank - 2) != dims.end(),
          std::find(dims.begin(), dims.end(), src_rank - 1) != dims.end()};
      break;
    case VectorLayout::ImplicitDim::kSecondMinor:
      reduces = {false, std::find(dims.begin(), dims.end(), src_rank - 1) !=
                            dims.end()};
      break;
    case VectorLayout::ImplicitDim::kMinor:
      reduces = {
          std::find(dims.begin(), dims.end(), src_rank - 1) != dims.end(),
          false};
      break;
  }
  const std::array<bool, 2> allow_replicated = {!reduces[0], !reduces[1]};

  if ((reduces[0] || reduces[1]) &&
      !src_layout.hasNativeTiling(ctx.target_shape)) {
    return multi_reduction_op.emitOpError(
               "Not implemented: Unsupported input layout: ")
           << src_layout;
  }
  if (src_layout.tiling() != dst_layout.tiling()) {
    return multi_reduction_op.emitOpError("Not implemented: Tiling change");
  }
  for (int i = 0; i < 2; ++i) {
    if (reduces[i] && src_layout.offsets()[i] == std::nullopt) {
      return multi_reduction_op.emitOpError(
          "Not implemented: Reductions over replicated axes");
    }
    // Offsets have to be equal, unless we're reducing over that dimension.
    if (src_layout.offsets()[i] != dst_layout.offsets()[i] && !reduces[i]) {
      return multi_reduction_op.emitOpError("Not implemented: Offset change");
    }
  }
  VectorLayout::ImplicitDim dst_implicit_dim;
  if ((reduces[0] && reduces[1]) ||
      (src_layout.implicit_dim() != VectorLayout::ImplicitDim::kNone &&
       (reduces[0] || reduces[1]))) {
    // This is difficult, because we'd like to make both tiling dims implicit,
    // but there is no way to do that in VectorLayout right now.
    // We use an equivalence between VectorLayouts when trailing dims are 1
    // to enable some special cases, but we should generalize this.
    if (*(res_ty.getShape().end() - 1) != 1) {
      return multi_reduction_op.emitOpError(
          "Not implemented: reductions over both trailing dimensions are only "
          "supported when the resulting value has a trailing axis of size 1");
    }
    dst_implicit_dim =
        VectorLayout::ImplicitDim::kSecondMinor;  // Anything works.
  } else if (reduces[0]) {
    TPU_ASSERT_OP(src_layout.implicit_dim() ==
                  VectorLayout::ImplicitDim::kNone);
    dst_implicit_dim = VectorLayout::ImplicitDim::kSecondMinor;
  } else if (reduces[1]) {
    TPU_ASSERT_OP(src_layout.implicit_dim() ==
                  VectorLayout::ImplicitDim::kNone);
    dst_implicit_dim = VectorLayout::ImplicitDim::kMinor;
  } else {
    dst_implicit_dim = src_layout.implicit_dim();
  }
  if (dst_layout.implicit_dim() != dst_implicit_dim) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Unsupported output implicit dimension");
  }

  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_vregs,
      disassemble(builder, src_layout, multi_reduction_op.getSource(),
                  ctx.target_shape));
  xla::Array<Value> dst_vregs(
      dst_layout.tileArrayShape(res_ty.getShape(), ctx.target_shape));
  tpu::ReductionKind tpu_kind;
  switch (multi_reduction_op.getKind()) {
    case vector::CombiningKind::ADD:
      tpu_kind = tpu::ReductionKind::SUM;
      break;
    case vector::CombiningKind::MAXIMUMF:
      tpu_kind = tpu::ReductionKind::MAX;
      break;
    default:
      return multi_reduction_op.emitOpError(
          "Not implemented: unsupported reduction kind");
  }
  const ArrayRef<int64_t> src_shape = src_ty.getShape();
  auto all_results_ok = dst_vregs.EachStatus(
      [&](const absl::Span<const int64_t> idx, Value *const dst_vreg) {
        // Extract a subset of source vregs that reduce into this result vreg.
        SmallVector<int64_t> src_slice_start;
        src_slice_start.reserve(src_rank);
        SmallVector<int64_t> src_slice_end;
        src_slice_end.reserve(src_rank);
        for (int64_t i : idx) {
          src_slice_start.push_back(i);
          src_slice_end.push_back(i + 1);
        }
        for (int64_t d : dims) {
          src_slice_start.insert(src_slice_start.begin() + d, 0);
          src_slice_end.insert(src_slice_end.begin() + d, src_vregs.dim(d));
        }
        xla::Array<Value> reduced_vregs =
            src_vregs.Slice(src_slice_start, src_slice_end);
        std::optional<Value> acc_vreg;
        auto reduction_status = reduced_vregs.EachStatus(
            [&](const absl::Span<const int64_t> red_idx,
                Value *const src_vreg) {
              SmallVector<int64_t> src_idx(red_idx.begin(), red_idx.end());
              for (int i = 0; i < src_idx.size(); ++i) {
                src_idx[i] += src_slice_start[i];
              }
              const std::unique_ptr<VRegDataBounds> data_bounds =
                  src_layout.tileDataBounds(builder.getContext(), src_shape,
                                            src_idx, ctx.target_shape,
                                            allow_replicated);
              // TODO(tlongeri): Maybe assemble/disassemble should take
              // TypedValue<VectorType> and we could save casts here and
              // elsewhere
              FailureOr<Value> failure_or_vreg =
                  maskOOB(ctx, builder, cast<TypedValue<VectorType>>(*src_vreg),
                          *data_bounds, neutral);
              if (failed(failure_or_vreg)) {
                return absl::UnknownError("");
              }
              Value vreg = failure_or_vreg.value();
              if (!acc_vreg.has_value()) {
                acc_vreg = vreg;
              } else {
                switch (tpu_kind) {
                  case tpu::ReductionKind::SUM:
                    acc_vreg = builder.create<arith::AddFOp>(vreg.getLoc(),
                                                             *acc_vreg, vreg);
                    break;
                  case tpu::ReductionKind::MAX:
                    acc_vreg = builder.create<arith::MaximumFOp>(
                        vreg.getLoc(), *acc_vreg, vreg);
                    break;
                }
              }
              return absl::OkStatus();
            });
        if (!reduction_status.ok()) {
          return reduction_status;
        }
        TPU_ASSERT_OP(acc_vreg.has_value());
        if (reduces[1]) {
          acc_vreg = builder.create<tpu::AllReduceOp>(
              multi_reduction_op->getLoc(), *acc_vreg, 1, tpu_kind);
        }
        if (reduces[0]) {
          acc_vreg = builder.create<tpu::AllReduceOp>(
              multi_reduction_op->getLoc(), *acc_vreg, 0, tpu_kind);
        }
        *dst_vreg = *acc_vreg;
        return absl::OkStatus();
      });
  if (!all_results_ok.ok()) {
    return failure();
  }
  multi_reduction_op->replaceAllUsesWith(
      assemble(builder, res_ty, dst_layout, dst_vregs, ctx.target_shape));
  multi_reduction_op->erase();
  return success();
}

LogicalResult vector_shape_cast_rule(RewriteContext &ctx, Operation &op,
                                     const ArrayRef<Layout> layouts_in,
                                     const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  using Tiling = std::array<int64_t, 2>;
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  TPU_ASSERT_EQ_OP(
      layout_in.bitwidth(),
      layout_out.bitwidth());  // This should be guaranteed through MLIR
                               // verifier plus our layoutIsValidForValue check
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto shape_cast_op = cast<vector::ShapeCastOp>(op);
  const VectorType src_ty = shape_cast_op.getSourceVectorType();
  const ArrayRef<int64_t> src_shape = src_ty.getShape();
  const VectorType dst_ty = shape_cast_op.getResultVectorType();
  const ArrayRef<int64_t> dst_shape = dst_ty.getShape();
  bool no_op = false;
  const std::array<int64_t, 2> src_tiled_dims =
      layout_in.getImplicitTiledDims(src_shape, 1);
  const std::array<int64_t, 2> dst_tiled_dims =
      layout_out.getImplicitTiledDims(dst_shape, 1);
  const std::array<int64_t, 2> src_vreg_slice =
      layout_in.vregSlice(ctx.target_shape);
  const std::array<int64_t, 2> dst_vreg_slice =
      layout_out.vregSlice(ctx.target_shape);
  if (layout_in.tiling() == layout_out.tiling() &&
      layout_in.offsets() == layout_out.offsets() &&
      src_tiled_dims == dst_tiled_dims) {
    no_op = true;
  } else if (  // Fold or unfold sublane dim, but keeping a whole number of
               // vregs.
      layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      layout_out.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      layout_in.offsets()[0] == 0 &&
      layout_in.offsets() == layout_out.offsets() &&
      layout_in.tiling() == layout_out.tiling() &&
      *(dst_shape.end() - 1) == *(src_shape.end() - 1) &&
      *(dst_shape.end() - 2) % dst_vreg_slice[0] == 0 &&
      *(src_shape.end() - 2) % src_vreg_slice[0] == 0) {
    no_op = true;
  } else if (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
             layout_out.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
             layout_in.offsets() == layout_out.offsets() &&
             layout_in.offsets() == LayoutOffsets{0, 0} &&
             layout_in.tiling()[0] == 1 &&
             layout_out.hasNativeTiling(ctx.target_shape) &&
             *(dst_shape.end() - 1) == dst_vreg_slice[1] &&
             *(dst_shape.end() - 2) % dst_vreg_slice[0] == 0 &&
             *(src_shape.end() - 1) % src_vreg_slice[1] == 0) {
    // Shapecast (..., m * 128 * packing) -> (..., 128).
    no_op = true;
  } else if (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
             layout_out.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
             layout_in.offsets() == LayoutOffsets{0, 0} &&
             layout_out.offsets() == LayoutOffsets{0, 0} &&
             layout_in.hasNativeTiling(ctx.target_shape) &&
             layout_out.tiling()[0] == 1 &&
             *(src_shape.end() - 1) == src_vreg_slice[1] &&
             *(src_shape.end() - 2) % src_vreg_slice[0] == 0 &&
             *(dst_shape.end() - 1) % dst_vreg_slice[1] == 0) {
    // Shapecast (..., 128) -> (..., m * 128 * packing).
    no_op = true;
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_vregs,
      disassemble(builder, layout_in, shape_cast_op.getSource(),
                  ctx.target_shape));
  auto getDstVregs = [&]() -> FailureOr<xla::Array<Value>> {
    if (no_op) {
      xla::Array<Value> dst_vregs_local = src_vregs;
      dst_vregs_local.Reshape(
          layout_out.tileArrayShape(dst_shape, ctx.target_shape));
      return dst_vregs_local;
    } else if (dst_shape.take_back(2) ==
                   ArrayRef<int64_t>{src_shape.back(), 1} &&
               layout_in.bitwidth() == 32 &&
               (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone ||
                layout_in.implicit_dim() ==
                    VectorLayout::ImplicitDim::kSecondMinor) &&
               layout_out.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
               layout_in.hasNativeTiling(ctx.target_shape) &&
               layout_in.tiling() == layout_out.tiling() &&
               layout_in.offsets()[0].value_or(0) == 0 &&
               layout_in.offsets()[1] == 0 && layout_out.offsets()[0] == 0
               // layout_out.offsets[1] can be anything, as we produce a
               // replicated result
    ) {
      // First, insert the new singleton lane dimension.
      SmallVector<int64_t> s(src_shape);
      s.push_back(1);
      xla::Array<Value> dst_vregs_local(
          layout_out.tileArrayShape(s, ctx.target_shape));
      if (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kSecondMinor) {
        // Make the sublane dimension explicit.
        SmallVector<int64_t> new_src_vregs_shape(
            toArrayRef(src_vregs.dimensions()));
        new_src_vregs_shape.insert(new_src_vregs_shape.end() - 1, 1);
        src_vregs.Reshape(new_src_vregs_shape);
        SmallVector<int64_t> new_dst_vregs_shape(
            toArrayRef(dst_vregs_local.dimensions()));
        new_dst_vregs_shape.insert(new_dst_vregs_shape.end() - 2, 1);
        dst_vregs_local.Reshape(new_dst_vregs_shape);
      }
      TPU_ASSERT_EQ_OP(dst_vregs_local.dimensions().back(),
                       1);  // We're inserting a singleton dimension
      dst_vregs_local.Each(
          [&](const absl::Span<const int64_t> dst_idx, Value *const dst_vreg) {
            const int64_t col_idx = *(dst_idx.end() - 2);
            const int64_t row_idx = *(dst_idx.end() - 3);
            auto [sublanes_in_lane, rem] =
                std::div(ctx.target_shape[1], ctx.target_shape[0]);
            CHECK_EQ(rem, 0);
            if (!layout_in.offsets()[0].has_value() && row_idx != 0) {
              return;  // All vregs along that dimension are the same.
            }
            SmallVector<int64_t> src_idx(toArrayRef(dst_idx));
            src_idx.pop_back();
            *(src_idx.end() - 2) /= ctx.target_shape[0];
            *(src_idx.end() - 1) /= sublanes_in_lane;
            Value col_vreg = src_vregs(src_idx);
            // BroadcastInSublanesOp requires the sublanes to be replicated.
            if (layout_in.offsets()[0].has_value()) {
              const int32_t sublane = row_idx % ctx.target_shape[0];
              col_vreg = builder.create<tpu::GatherOp>(
                  col_vreg.getType(), col_vreg,
                  /*indices=*/
                  SmallVector<int32_t>(ctx.target_shape[0], sublane),
                  /*dimension=*/0);
            }
            *dst_vreg = builder.create<BroadcastInSublanesOp>(
                col_vreg.getType(), col_vreg,
                /*lane=*/(col_idx % sublanes_in_lane) * ctx.target_shape[0]);
          });
      if (!layout_in.offsets()[0].has_value()) {
        // Broadcast the sublane vregs.
        // TODO(tlongeri): This could be done more efficiently
        dst_vregs_local.Each([&](const absl::Span<const int64_t> dst_idx,
                                 Value *const dst_vreg) {
          SmallVector<int64_t> first_row_idx(toArrayRef(dst_idx));
          *(first_row_idx.end() - 3) = 0;
          *dst_vreg = dst_vregs_local(first_row_idx);
        });
      }
      // Now, permute the major axes of the vreg array.
      dst_vregs_local.Reshape(
          layout_out.tileArrayShape(dst_shape, ctx.target_shape));
      return dst_vregs_local;
    } else {
      return shape_cast_op.emitOpError(
                 "Not implemented: Unsupported vector.shape_cast: ")
             << *shape_cast_op;
    }
  };
  FAILUREOR_ASSIGN_OR_RETURN(const xla::Array<Value> dst_vregs, getDstVregs());
  shape_cast_op->replaceAllUsesWith(
      assemble(builder, dst_ty, layout_out, dst_vregs, ctx.target_shape));
  shape_cast_op->erase();
  return success();
}
LogicalResult vector_store_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_out.size(), 0);
  MLIRContext *const mlir_ctx = op.getContext();
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(llvm::none_of(layouts_in.drop_front(),
                              [&](const Layout &l) { return l.has_value(); }));
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  vector::StoreOp store_op = cast<vector::StoreOp>(op);
  const VectorType ty = store_op.getValueToStore().getType();
  const VectorLayout &to_store_layout = *layouts_in.front();
  const auto memref_ty = getMemRefType(store_op.getBase());
  if (!ty.getRank()) {
    return op.emitOpError("Not implemented: scalar stores to vmem");
  }
  const bool is_1d = ty.getRank() == 1;
  VectorLayout::ImplicitDim expected_dim =
      is_1d ? VectorLayout::ImplicitDim::kSecondMinor
            : VectorLayout::ImplicitDim::kNone;
  if (to_store_layout.implicit_dim() != expected_dim) {
    return op.emitOpError("Not implemented: unsupported layout");
  }
  using Tiling = std::array<int64_t, 2>;
  FAILUREOR_ASSIGN_OR_RETURN(
      const Tiling memref_tiling,
      getMemRefTiling(store_op.getBase(), ctx.target_shape));
  if (memref_tiling != to_store_layout.tiling() &&
      !(memref_tiling[0] == 1 && to_store_layout.tiling()[0] == 1 &&
        memref_tiling[1] % to_store_layout.tiling()[1] == 0)) {
    // Now we can handle the case when tiling is (1, TARGET_SHAPE.lanes).
    // TODO(b/295393167): need to support strided store for bitwidth < 32.
    if (to_store_layout.bitwidth() != 32 ||
        to_store_layout.tiling() != Tiling{1, ctx.target_shape[1]}) {
      return op.emitOpError("Not implemented");
    }
  }

  bool can_support_unaligned_dynamic_index = false;
  bool must_support_unaligned_dynamic_index = false;
  if (store_op.getIndices().size() > 1) {
    auto second_minor_idx = store_op.getIndices().take_back(2)[0];
    if (failed(getIntConst(second_minor_idx, /*silent=*/true)) &&
        !isGuaranteedDivisible(second_minor_idx, memref_tiling[0])) {
      must_support_unaligned_dynamic_index = true;
    }
  }
  int64_t sublane_stride = 1;
  // Handle special patterns that allow us to support more flexible loads.
  if (to_store_layout.bitwidth() == 32 &&
      to_store_layout.tiling() == Tiling{1, ctx.target_shape[1]}) {
    // Storing a single row on the 2nd minor dim from the (1, 128) layout. We
    // can use sublane striding to perform the relayout as part of the store.
    // The stride of store should be the number of sublanes in memref tile when
    // store a single sublane.
    sublane_stride = memref_tiling[0];
    can_support_unaligned_dynamic_index = true;
  } else {
    // Otherwise, if the memref has a short last dimension and is contiguous
    // all the tiled layouts become equivalent, so we can handle unaligned
    // dynamic indices without any special case.
    auto mem_layout = dyn_cast<TiledLayoutAttr>(memref_ty.getLayout());
    if (!mem_layout) {
      return op.emitOpError("Expected a tiled memref");
    }
    auto tile_strides = mem_layout.getTileStrides();
    if (memref_ty.getShape().back() == ctx.target_shape[1] &&
        tile_strides.take_back(2) == ArrayRef<int64_t>{1, 1}) {
      can_support_unaligned_dynamic_index = true;
    }
  }

  auto add_idx = [&](const Value &v, int64_t d) -> Value {
    if (auto cst = getIntConst(v, /*silent=*/true); succeeded(cst)) {
      return IdxConst(cst.value() + d, builder, op.getLoc());
    }
    return builder.create<arith::AddIOp>(v, IdxConst(d, builder, op.getLoc()));
  };

  int tiled_dims = is_1d ? 1 : 2;
  Value base_addr = store_op.getBase();
  SmallVector<Value, 4> base_indices = store_op.getIndices();

  if (must_support_unaligned_dynamic_index) {
    if (!can_support_unaligned_dynamic_index) {
      return op.emitOpError(
          "Not implemented: dynamic store with unaligned indices");
    }
  } else {
    // Convert dynamic store to dynamic slice + static store. This saves us a
    // bunch of scalar core work.
    auto slice_result =
        sliceRef(builder, store_op.getBase(),
                 store_op.getVectorType().getShape(), store_op.getIndices(),
                 ArrayRef<int64_t>(memref_tiling).take_back(tiled_dims));
    if (failed(slice_result)) {
      return failure();
    }
    base_addr = slice_result->first;
    CHECK_EQ(slice_result->second.size(), base_indices.size());
    for (int i = 0; i < base_indices.size(); ++i) {
      base_indices[i] = IdxConst(slice_result->second[i], builder, op.getLoc());
    }
  }

  // TODO(jevinjiang): ideally we should update the base addr and use static
  // indices even for the cases that can skip alignment check. This can save
  // us a bunch of scalar core work.
  auto tile_base_idxs = ArrayRef<Value>(base_indices).take_back(tiled_dims);
  auto batch_base_idxs = ArrayRef<Value>(base_indices).drop_back(tiled_dims);

  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> tiles,
      disassemble(builder, to_store_layout, store_op.getValueToStore(),
                  ctx.target_shape));
  const int64_t ndims = ty.getRank();
  const auto base_s =
      is_1d ? IdxConst(0, builder, op.getLoc()) : tile_base_idxs.front();
  const auto base_l = tile_base_idxs.back();
  if (is_1d) {
    tiles.Reshape(
        to_store_layout.implicitShape(toArrayRef(tiles.dimensions())));
  }
  const LayoutOffset sublane_offset = to_store_layout.offsets()[0];
  const LayoutOffset lane_offset = to_store_layout.offsets()[1];
  if (!sublane_offset.has_value() || !lane_offset.has_value()) {
    return store_op.emitOpError(
        "Not implemented: Replicated layout disallowed in vector store");
  }
  const SmallVector<int64_t> stored_shape =
      to_store_layout.implicitShape(ty.getShape());
  const std::array<int64_t, 2> vreg_slice =
      to_store_layout.vregSlice(ctx.target_shape);
  const absl::Status status =
      tiles.EachStatus([&](const absl::Span<const int64_t> idx,
                           const Value tile) -> absl::Status {
        const std::unique_ptr<VRegDataBounds> bounds =
            to_store_layout.tileDataBounds(mlir_ctx, stored_shape,
                                           toArrayRef(idx), ctx.target_shape);
        const int64_t sidx = *(idx.end() - 2);
        const int64_t lidx = *(idx.end() - 1);
        SmallVector<Value> indices(ndims);
        for (int64_t i = 0; i < batch_base_idxs.size(); ++i) {
          indices[i] = add_idx(batch_base_idxs[i], idx[i]);
        }
        if (!is_1d) {
          *(indices.end() - 2) =
              add_idx(base_s, sidx * vreg_slice[0] - *sublane_offset);
        }
        *(indices.end() - 1) =
            add_idx(base_l, lidx * vreg_slice[1] - *lane_offset);
        const DenseBoolArrayAttr sublane_mask =
            bounds->getSublaneMask(store_op->getContext(), ctx.target_shape);
        const bool masks_subelements =
            bounds->maskVariesAlong(Direction::kSubelements, ctx.target_shape);
        if (bounds->maskVariesAlong(Direction::kLanes, ctx.target_shape) ||
            masks_subelements) {
          auto failure_or_mask =
              bounds->getVectorMask(builder, store_op.getLoc(),
                                    ctx.hardware_generation, ctx.target_shape);
          if (failed(failure_or_mask)) {
            return absl::UnimplementedError("Failed to get vector mask");
          }
          TypedValue<VectorType> mask = failure_or_mask.value();
          // Vmem stores don't support masking below 32-bit granularity, so we
          // need to load and blend explicitly if needed.
          if (masks_subelements) {
            auto data = builder.create<tpu::LoadOp>(tile.getType(), base_addr,
                                                    indices, sublane_mask,
                                                    /*sublane_stride=*/nullptr);
            const bool mask_is_a_bitmask =
                cast<IntegerType>(mask.getType().getElementType()).getWidth() ==
                32;
            Value updated;
            if (mask_is_a_bitmask) {
              auto ones = builder.create<arith::ConstantOp>(
                  mask.getType(),
                  DenseElementsAttr::get(
                      mask.getType(),
                      builder.getIntegerAttr(builder.getI32Type(),
                                             APInt(32, 0xFFFFFFFF))));
              auto masked_tile = builder.create<arith::AndIOp>(
                  store_op.getLoc(), mask,
                  builder.create<tpu::BitcastVregOp>(mask.getType(), tile));
              auto mask_neg = builder.create<arith::XOrIOp>(ones, mask);
              auto masked_data = builder.create<arith::AndIOp>(
                  mask_neg,
                  builder.create<tpu::BitcastVregOp>(mask.getType(), data));
              updated = builder.create<tpu::BitcastVregOp>(
                  tile.getType(),
                  builder.create<arith::OrIOp>(masked_data, masked_tile));
            } else {
              updated = builder.create<arith::SelectOp>(mask, tile, data);
            }
            builder.create<tpu::StoreOp>(
                updated, base_addr, indices, sublane_mask,
                /*mask=*/nullptr,
                /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
          } else {
            builder.create<tpu::StoreOp>(
                tile, base_addr, indices, sublane_mask,
                /*mask=*/mask,
                /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
          }
        } else {
          builder.create<tpu::StoreOp>(
              tile, base_addr, indices, sublane_mask,
              /*mask=*/nullptr,
              /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
        }
        return absl::OkStatus();
      });
  if (!status.ok()) {
    return failure();
  }
  store_op->erase();
  return success();
  }

LogicalResult vector_transpose_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 1);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_in.front().has_value());
  TPU_ASSERT_OP(layouts_out.front().has_value());
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_in != layout_out) {
    return op.emitOpError("Not implemented: Unsupported 2D layouts");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto transpose_op = cast<vector::TransposeOp>(op);
  VectorType src_ty = transpose_op.getSourceVectorType();
  VectorType dst_ty = transpose_op.getResultVectorType();
  const int64_t rank = src_ty.getRank();
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_vregs,
      disassemble(builder, layout_in, transpose_op.getVector(),
                  ctx.target_shape));
  ArrayRef<int64_t> permutation = transpose_op.getPermutation();
  const auto tile_perm = permutation.take_back(2);
  if (tile_perm != ArrayRef<int64_t>{rank - 2, rank - 1} &&
      tile_perm != ArrayRef<int64_t>{rank - 1, rank - 2}) {
    return transpose_op->emitOpError(
        "Not implemented: Unsupported permutation");
  }
  {
    SmallVector<int64_t> p(permutation);
    p[rank - 2] = rank - 2;
    p[rank - 1] = rank - 1;
    src_vregs.TransposeDimensions(p);
  }
  if (tile_perm == ArrayRef<int64_t>{rank - 2, rank - 1}) {
    transpose_op->replaceAllUsesWith(
        assemble(builder, dst_ty, layout_out, src_vregs, ctx.target_shape));
    transpose_op.erase();
    return success();
  }
  if (layout_in.offsets() != LayoutOffsets{0, 0} ||
      !layout_in.hasNativeTiling(ctx.target_shape)) {
    return transpose_op->emitOpError(
        "Not implemented: Non-native or offset layout unsupported");
  }
  const int64_t transpose_unit_size = ctx.target_shape[1];
  if (ctx.hardware_generation < 4 && layout_in.bitwidth() != 32) {
    return transpose_op->emitOpError(
        "Not implemented: TPUs before v4 only support 32-bit transposes");
  }
  xla::Array<Value> dst_vregs(
      layout_out.tileArrayShape(dst_ty.getShape(), ctx.target_shape));
  const int packing = layout_in.packing();
  // Note that we checked for native tiling above.
  const int64_t vregs_per_tile = transpose_unit_size / layout_in.tiling()[0];
  const SmallVector<int64_t> minor_perm{1, 0};
  const auto tile_ty = VectorType::get(
      {transpose_unit_size, transpose_unit_size}, src_ty.getElementType());
  const auto batch_tile_ty_in =
      VectorType::get({transpose_unit_size, transpose_unit_size * packing},
                      src_ty.getElementType());
  const auto batch_tile_ty_out =
      VectorType::get({transpose_unit_size * packing, transpose_unit_size},
                      src_ty.getElementType());
  // For packed types, we can increase the XLU throughput by batching together
  // multiple tiles. At the moment we always batch along columns, with the
  // reasoning being that if all the tiles are fed into the MXU, then it's
  // better if we end up with results that contribute to the same contraction.
  const bool can_batch = layout_in.bitwidth() == 16;
  auto doTranspose = [&](const ArrayRef<int64_t> batch_idx,
                         const int64_t src_row, const int64_t src_col,
                         const int64_t src_col_end, const VectorType tile_ty_in,
                         const VectorType tile_ty_out) {
    SmallVector<int64_t> src_slice_starts;
    src_slice_starts.reserve(rank);
    src_slice_starts.append(batch_idx.begin(), batch_idx.end());
    src_slice_starts.append({src_row * vregs_per_tile, src_col});
    SmallVector<int64_t> src_slice_ends;
    src_slice_ends.reserve(rank);
    auto incremented_batch_idx =
        map_range(batch_idx, [](int64_t i) { return i + 1; });
    src_slice_ends.append(incremented_batch_idx.begin(),
                          incremented_batch_idx.end());
    src_slice_ends.append({(src_row + 1) * vregs_per_tile, src_col_end});
    xla::Array<Value> src_tile_vregs = src_vregs.Slice(
        src_slice_starts, src_slice_ends, /*out_of_bounds_ok=*/true);
    // Drop leading singleton (batch) dimensions to have a shape that conforms
    // with the vreg array shape specified by layout_in, as expected by assemble
    src_tile_vregs.Reshape(
        ArrayRef<int64_t>{vregs_per_tile, src_col_end - src_col});
    const Value src_tile = assemble(builder, tile_ty_in, layout_in,
                                    src_tile_vregs, ctx.target_shape);
    auto new_transpose_op =
        builder.create<vector::TransposeOp>(tile_ty_out, src_tile, minor_perm);
    new_transpose_op->setAttr("out_layout",
                              builder.getAttr<VectorLayoutAttr>(layout_out));
    auto unroll_vectors_op = builder.create<tpu::UnrollVectorsOp>(
        llvm::map_to_vector(src_tile_vregs,
                            [](Value v) { return v.getType(); }),
        new_transpose_op);
    SmallVector<int64_t> dst_slice_starts;
    dst_slice_starts.reserve(rank);
    dst_slice_starts.append(batch_idx.begin(), batch_idx.end());
    dst_slice_starts.append({src_col * vregs_per_tile, src_row});
    SmallVector<int64_t> dst_slice_ends;
    dst_slice_ends.reserve(rank);
    dst_slice_ends.append(incremented_batch_idx.begin(),
                          incremented_batch_idx.end());
    dst_slice_ends.append({src_col_end * vregs_per_tile, src_row + 1});
    updateSliceFromRange(dst_vregs, unroll_vectors_op.getResults(),
                         dst_slice_starts, dst_slice_ends);
  };
  const int num_batch_dims = rank - 2;
  const ArrayRef<int64_t> batch_sizes =
      dst_ty.getShape().take_front(num_batch_dims);
  SmallVector<int64_t> batch_idx(num_batch_dims);
  const int64_t tile_rows =
      xla::CeilOfRatio(*(src_ty.getShape().end() - 2), transpose_unit_size);
  const int64_t num_col_tiles =
      xla::CeilOfRatio(*(src_ty.getShape().end() - 1), transpose_unit_size);
  do {
    for (int64_t src_row = 0; src_row < tile_rows; ++src_row) {
      if (can_batch) {
        const int64_t num_batch_tiles = num_col_tiles / 2;
        for (int64_t src_col = 0; src_col < num_batch_tiles; ++src_col) {
          doTranspose(batch_idx, src_row, src_col * 2, (src_col + 1) * 2,
                      batch_tile_ty_in, batch_tile_ty_out);
        }
        if (num_col_tiles % 2 == 1) {
          doTranspose(batch_idx, src_row, num_col_tiles - 1, num_col_tiles,
                      tile_ty, tile_ty);
        }
      } else {
        for (int64_t src_col = 0; src_col < num_col_tiles; ++src_col) {
          doTranspose(batch_idx, src_row, src_col, src_col + 1, tile_ty,
                      tile_ty);
        }
      }
    }
  } while (incrementIndex(batch_idx, batch_sizes));
  for (const Value v : dst_vregs) {
    TPU_ASSERT_OP(v != nullptr);
  }
  transpose_op->replaceAllUsesWith(
      assemble(builder, dst_ty, layout_out, dst_vregs, ctx.target_shape));
  transpose_op->erase();
  return success();
}

LogicalResult prng_random_bits_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  TPU_ASSERT_EQ_OP(layouts_in.size(), 0);
  TPU_ASSERT_EQ_OP(layouts_out.size(), 1);
  TPU_ASSERT_OP(layouts_out.front().has_value());

  const VectorLayout &layout_out = *layouts_out.front();
  tpu::PRNGRandomBitsOp rng_op = cast<tpu::PRNGRandomBitsOp>(op);
  if (layout_out != VectorLayout(32, {0, 0}, ctx.target_shape,
                                   VectorLayout::ImplicitDim::kNone)) {
    return op.emitOpError(
        "Unsupported output layout for ") << rng_op->getName();
  }
  OpBuilder builder(op.getContext());
  builder.setInsertionPointAfter(&op);

  VectorType vty = rng_op.getResult().getType();
  TPU_ASSERT_OP(vty.getElementType().isInteger());
  // Only 32-bit output supported currently.
  TPU_ASSERT_OP(vty.getElementType().getIntOrFloatBitWidth() == 32);
  xla::Array<Value> tiles(
      layout_out.tileArrayShape(vty.getShape(), ctx.target_shape));
  VectorType tile_ty = VectorType::get(ctx.target_shape, vty.getElementType());
  tiles.Each([&](absl::Span<const int64_t> tile_idxs, Value * v) {
    *v = builder.create<tpu::PRNGRandomBitsOp>(op.getLoc(), tile_ty);
  });
  const RollVectorsOp roll_vectors_op =
      assemble(builder, vty, layout_out, tiles, ctx.target_shape);
  rng_op->replaceUsesWithIf(roll_vectors_op, [&](OpOperand &operand) {
    return operand.getOwner() != roll_vectors_op;
  });
  rng_op->erase();
  return success();
}

const llvm::StringMap<rule_type> &rules() {
  static auto rules = new llvm::StringMap<rule_type>{
      {arith::ConstantOp::getOperationName(), arith_constant_rule},
      {arith::ExtFOp::getOperationName(), arith_extf_rule},
      {arith::ExtSIOp::getOperationName(), arith_extsi_rule},
      {arith::TruncFOp::getOperationName(), arith_truncf_rule},
      {arith::TruncIOp::getOperationName(), arith_trunci_rule},
      {func::ReturnOp::getOperationName(), func_return_rule},
      {scf::ForOp::getOperationName(), scf_for_rule},
      {scf::WhileOp::getOperationName(), scf_while_rule},
      {scf::ConditionOp::getOperationName(), scf_condition_rule},
      {scf::IfOp::getOperationName(), scf_if_rule},
      {scf::YieldOp::getOperationName(), scf_yield_rule},
      {tpu::RotateOp::getOperationName(), tpu_rotate_rule},
      {tpu::DynamicRotateOp::getOperationName(), tpu_dynamic_rotate_rule},
      {tpu::ConcatenateOp::getOperationName(), tpu_concatenate_rule},
      {tpu::IotaOp::getOperationName(), tpu_iota_rule},
      {tpu::GatherOp::getOperationName(), tpu_gather_rule},
      {tpu::LoadOp::getOperationName(), tpu_load_rule},
      {tpu::StoreOp::getOperationName(), tpu_store_rule},
      {tpu::StridedLoadOp::getOperationName(), tpu_strided_load_rule},
      {tpu::StridedStoreOp::getOperationName(), tpu_strided_store_rule},
      {tpu::MatmulOp::getOperationName(), tpu_matmul_rule},
      {tpu::RegionOp::getOperationName(), tpu_region_rule},
      {tpu::RepeatOp::getOperationName(), tpu_repeat_rule},
      {tpu::BitcastOp::getOperationName(), tpu_bitcast_rule},
      {tpu::TraceOp::getOperationName(), tpu_trace_rule},
      {tpu::AssumeLayoutOp::getOperationName(), tpu_assume_layout_rule},
      {tpu::PRNGRandomBitsOp::getOperationName(), prng_random_bits_rule},
      {vector::BroadcastOp::getOperationName(), vector_broadcast_rule},
      {vector::ContractionOp::getOperationName(), vector_contract_rule},
      {vector::ExtractOp::getOperationName(), vector_extract_rule},
      {vector::LoadOp::getOperationName(), vector_load_rule},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_reduction_rule},
      {vector::ExtractStridedSliceOp::getOperationName(),
       vector_extract_strided_slice_rule},
      {vector::ShapeCastOp::getOperationName(), vector_shape_cast_rule},
      {vector::StoreOp::getOperationName(), vector_store_rule},
      {vector::TransposeOp::getOperationName(), vector_transpose_rule}};
  return *rules;
}
}  // namespace

RollVectorsOp assemble(OpBuilder &builder, VectorType vty,
                       const VectorLayout &layout,
                       const xla::Array<Value> &vals,
                       const std::array<int64_t, 2> target_shape,
                       const bool use_implicit_shape) {
  // TODO(tlongeri): Maybe just add a parameter to tileArrayShape instead of
  // having `tileArrayShape` and `tileArrayImplicitShape`.
  SmallVector<int64_t> vreg_array_shape =
      layout.tileArrayImplicitShape(vty.getShape(), target_shape);
  if (!use_implicit_shape) {
    layout.eraseImplicit(vreg_array_shape);
  }
  CHECK(vals.dimensions() == vreg_array_shape);
  CHECK_GT(vals.num_elements(), 0);
  Location loc = vals.begin()->getLoc();
  auto op =
      builder.create<RollVectorsOp>(loc, vty, XlaArrayToFlatArrayRef(vals));
  op->setAttr("out_layout", builder.getAttr<ArrayAttr>(ArrayRef<Attribute>{
                                builder.getAttr<VectorLayoutAttr>(layout)}));
  return op;
}

// Disassemble an MLIR vector into an ndarray of native vectors.
//
// Args:
//   layout: The layout of val. Used to determine the unrolling into
//     native-shaped vectors.
//   val: Value to disassemble. Must be of type VectorType.
//
// Returns:
//   An ndarray of MLIR values representing the tiling of val given by layout.
FailureOr<xla::Array<Value>> disassemble(
    OpBuilder &builder, const VectorLayout &layout,
    const TypedValue<VectorType> val, const std::array<int64_t, 2> target_shape,
    const bool use_implicit_shape) {  // TODO(tlongeri): Remove default
  const auto vty = val.getType();
  const auto op_result = dyn_cast<OpResult>(val);
  if (op_result == nullptr) {
    return failure();
  }
  Operation *const op = op_result.getOwner();
  const unsigned res_idx = op_result.getResultNumber();
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                             getOutLayouts(*op, target_shape));
  const Layout def_layout = def_layouts[res_idx];
  TPU_ASSERT_LOC(val.getLoc(), def_layout.has_value());
  TPU_ASSERT_LOC(val.getLoc(),
                 def_layout->generalizes(layout, vty.getShape(), target_shape));
  // TODO(tlongeri): Maybe just add a parameter to tileArrayShape instead of
  // having `tileArrayShape` and `tileArrayImplicitShape`.
  SmallVector<int64_t> layout_shape =
      layout.tileArrayImplicitShape(vty.getShape(), target_shape);
  if (!use_implicit_shape) {
    layout.eraseImplicit(layout_shape);
  }
  if (auto roll_vectors_op = dyn_cast<RollVectorsOp>(op)) {
    return XlaArrayFromShapeAndValues<Value>(layout_shape,
                                             roll_vectors_op->getOperands());
  }
  if (auto contraction_op = dyn_cast<vector::ContractionOp>(op)) {
    const int64_t num_vectors = ShapedType::getNumElements(layout_shape);
    FAILUREOR_ASSIGN_OR_RETURN(
        VectorType vreg_ty,
        getNativeVregType(vty.getElementType(), target_shape));
    // TODO(tlongeri): nicer way of doing ValueTypeRange?
    Operation *const u = builder.create<UnrollVectorsOp>(
        val.getLoc(), SmallVector<Type>(num_vectors, vreg_ty), val);
    return XlaArrayFromShapeAndValues<Value>(layout_shape, u->getResults());
  }
  return op->emitOpError("Not implemented: ") << val;
}

// Assembles a destination tile using partial data from rotated vregs using a
// divide-and-conquer strategy.
//
// Arguments:
//   rotated_row_vregs: A row of rotated vregs, from which destination tile(s)
//     is/are to be selected to assemble a new vreg.
//   src_layout: The source layout.
//   start_src_col: The first rotated vreg in the row of rotated vregs to
//     process.
//   end_src_col: The last rotated vreg in the row of rotated vreg to process.
//   first_dst_tile_sublane_offset: Sublane offset where the first dst tile to
//   be
//     selected starts.
//   dst_layout: Destination layout, based on which retiling is being performed.
//   hw_generation: The generation of a target hardware.
//
// Returns:
//   A new vreg assembled from dst tiles stored in given rotated vregs.
Value selectTilesFromRotatedRowVregs(
    OpBuilder &builder, const ArrayRef<Value> &rotated_row_vregs,
    const int64_t start_src_col, const int64_t end_src_col,
    const int64_t first_dst_tile_sublane_offset, const VectorLayout &dst_layout,
    const std::array<int64_t, 2> target_shape) {
  CHECK_LE(start_src_col, end_src_col);
  CHECK_LE(start_src_col, end_src_col);
  if (start_src_col == end_src_col) {
    return rotated_row_vregs[start_src_col];
  }
  const int64_t mid_src_col = start_src_col + (end_src_col - start_src_col) / 2;

  Value left_partial_vreg = selectTilesFromRotatedRowVregs(
      builder, rotated_row_vregs, start_src_col, mid_src_col,
      first_dst_tile_sublane_offset, dst_layout, target_shape);
  Location loc = left_partial_vreg.getLoc();

  const int64_t left_tiles_count = mid_src_col - start_src_col + 1;
  const int64_t right_first_dst_tile_sublane_offset =
      (first_dst_tile_sublane_offset +
       left_tiles_count * dst_layout.sublanesPerTile(target_shape)) %
      target_shape[0];

  Value right_partial_vreg = selectTilesFromRotatedRowVregs(
      builder, rotated_row_vregs, mid_src_col + 1, end_src_col,
      right_first_dst_tile_sublane_offset, dst_layout, target_shape);

  const IntegerType i1 = builder.getI1Type();
  // We never need to select partial sublanes, even for packed data.
  const auto mask_vreg_ty = VectorType::get(target_shape, i1);
  auto i32_vreg = VectorType::get(target_shape, builder.getI32Type());
  auto select_32bit = [&](Value sublane_mask, Value left, Value right) {
    // Always do the selects on 32-bit granularity for maximum HW compatibility.
    Type vreg_ty = left.getType();
    if (dst_layout.packing() != 1) {
      left = builder.create<tpu::BitcastVregOp>(loc, i32_vreg, left);
      right = builder.create<tpu::BitcastVregOp>(loc, i32_vreg, right);
    }
    Value result =
        builder.create<arith::SelectOp>(loc, sublane_mask, left, right);
    if (dst_layout.packing() != 1) {
      result = builder.create<tpu::BitcastVregOp>(loc, vreg_ty, result);
    }
    return result;
  };

  auto boundIdxConst = std::bind(IdxConst, std::placeholders::_1, builder,
                                 left_partial_vreg.getLoc());
  if (first_dst_tile_sublane_offset < right_first_dst_tile_sublane_offset) {
    // The useful data sublanes in left vregs do not wrap around in vreg.
    // For e.g. consider (2,128) destination tiling and we are trying to merge
    // two vregs as follows:
    //
    //   vreg 0:        vreg 1:
    //   x x x x x     dst_tile_2
    //   x x x x x     dst_tile_3
    //   dst_tile_4    x x x x x
    //   dst_tile_5    x x x x x
    //   dst_tile_6    x x x x x
    //   dst_tile_7    x x x x x
    //   x x x x x     dst_tile_0
    //   x x x x x     dst_tile_1
    //
    // In the above case, the data we want to select from vreg 1 wraps around,
    // whereas vreg 0 useful data is contiguous. It is easier to create '1' mask
    // for vreg 0.
    auto sublanes_mask = builder.create<tpu::CreateMaskOp>(
        left_partial_vreg.getLoc(), mask_vreg_ty,
        ArrayRef<Value>{boundIdxConst(first_dst_tile_sublane_offset),
                        boundIdxConst(0)},
        ArrayRef<Value>{boundIdxConst(right_first_dst_tile_sublane_offset),
                        boundIdxConst(target_shape[1])});
    return select_32bit(sublanes_mask, left_partial_vreg, right_partial_vreg);
  }

  auto sublanes_mask = builder.create<tpu::CreateMaskOp>(
      left_partial_vreg.getLoc(), mask_vreg_ty,
      ArrayRef<Value>{boundIdxConst(right_first_dst_tile_sublane_offset),
                      boundIdxConst(0)},
      ArrayRef<Value>{boundIdxConst(first_dst_tile_sublane_offset),
                      boundIdxConst(target_shape[1])});
  return select_32bit(sublanes_mask, right_partial_vreg, left_partial_vreg);
}

// Retiles across vregs to match the destination layout when the sublane tiling
// dimension is reduced.
//
// Arguments:
//   value_shape: The shape of the value which needs to be retiled in vregs.
//   src: The source layout.
//   src_vreg_array: An array of vregs storing source tiles.
//   dst_layout: The destination layout, with reduced sublane dimension, based
//   on
//     which the retiling will be performed.
//   hw_generation: The generation of a target hardware.
//
// Returns:
//   A new array of vregs that store tiles based on the destination layout.
xla::Array<Value> retileToReducedSublanes(
    OpBuilder &builder, const ArrayRef<int64_t> value_shape,
    const VectorLayout &src_layout, const xla::Array<Value> &src_vreg_array,
    const VectorLayout &dst_layout, const std::array<int64_t, 2> target_shape) {
  const int64_t dst_tiling_sublane = dst_layout.tiling()[0];
  CHECK_LT(0, dst_tiling_sublane);
  CHECK_LT(dst_tiling_sublane, src_layout.tiling()[0]);
  CHECK(llvm::isPowerOf2_64(dst_tiling_sublane));

  xla::Array<Value> dst_vreg_array(
      dst_layout.tileArrayShape(value_shape, target_shape));

  // We need to rotate each src tile in each src vreg once so that they can
  // be merged to form new vregs. If a src vreg contains more than one src tile,
  // it will be rotated once per src tile. Consider (8,512) tensor stored with
  // layout (8,128) in a vreg array of shape (1, 4). Each src vreg
  // contains one src tile in this case. Given, the destination layout is
  // (2,128), each src tile is divided into 4 destination tiles as shown below:
  //
  //   src_vreg_0_0:     src_vreg_0_1:    src_vreg_0_2:   src_vreg_0_3:
  // dst_tile_0_0_0    dst_tile_0_0_1   dst_tile_0_0_2  dst_tile_0_0_3
  // dst_tile_1_0_0    dst_tile_1_0_1   dst_tile_1_0_2  dst_tile_1_0_3
  // dst_tile_2_0_0    dst_tile_2_0_1   dst_tile_2_0_2  dst_tile_2_0_3
  // dst_tile_3_0_0    dst_tile_3_0_1   dst_tile_3_0_2  dst_tile_3_0_3
  //
  // In this example, each src tile in the src vreg is rotated by
  // col *  sublanes_per_tile to produce the following rotated src vregs:
  //
  // rot_src_vreg_0_0: rot_src_vreg_0_1: rot_src_vreg_0_2: rot_src_vreg_0_3:
  //     dst_tile_0_0_0    dst_tile_3_0_1    dst_tile_2_0_2    dst_tile_1_0_3
  //     dst_tile_1_0_0    dst_tile_0_0_1    dst_tile_3_0_2    dst_tile_2_0_3
  //     dst_tile_2_0_0    dst_tile_1_0_1    dst_tile_0_0_2    dst_tile_3_0_3
  //     dst_tile_3_0_0    dst_tile_2_0_1    dst_tile_1_0_2    dst_tile_0_0_3

  // If there were 2 src tiles in the src vreg, we would have rotated each src
  // vreg twice, producing 2 rotated src vreg per src vreg. The rotation amount
  // is calculated from the src and the dest tiling.

  const int64_t src_tiles_per_vreg = src_layout.tilesPerVreg(target_shape);
  const int64_t dst_tiles_per_vreg = dst_layout.tilesPerVreg(target_shape);
  const int64_t src_sublanes_per_tile =
      src_layout.sublanesPerTile(target_shape);
  const int64_t dst_sublanes_per_tile =
      dst_layout.sublanesPerTile(target_shape);
  // Each vreg may store more than one src tile. We may have to rotate a vreg,
  // once for every src tile in the vreg.
  SmallVector<int64_t> rotated_src_vreg_array_shape(
      toArrayRef(src_vreg_array.dimensions()));
  rotated_src_vreg_array_shape.back() *= src_tiles_per_vreg;
  xla::Array<Value> rotated_src_vreg_array(rotated_src_vreg_array_shape);

  rotated_src_vreg_array.Each([&](const absl::Span<const int64_t> rotated_idx,
                                  Value *const rotated_src_vreg) {
    const int64_t idx = rotated_idx.back();
    const int64_t tile_idx = idx % dst_tiles_per_vreg;
    const int64_t dst_sublane = tile_idx * dst_sublanes_per_tile;
    auto [src_col, src_tile_offset] = std::div(idx, src_tiles_per_vreg);
    SmallVector<int64_t> src_vreg_idx(toArrayRef(rotated_idx));
    src_vreg_idx.back() = src_col;
    Value src_vreg = src_vreg_array(src_vreg_idx);
    const int64_t src_sublane = src_tile_offset * src_sublanes_per_tile;
    int64_t rotate_amt = dst_sublane - src_sublane;
    if (rotate_amt == 0) {
      *rotated_src_vreg = src_vreg;
      return;
    }
    if (rotate_amt < 0) {
      rotate_amt += target_shape[0];
    }
    *rotated_src_vreg = builder.create<tpu::RotateOp>(
        src_vreg.getLoc(), src_vreg, rotate_amt,
        /*dimension=*/0, /*stride=*/nullptr, /*stride_dimension=*/nullptr);
  });
  // Assemble output vregs using tiles from rotated vregs using select.
  // Given, above example, destination vregs are then assembled as follows:
  //  dst_vreg_0_0:
  // dst_tile_0_0_0
  // dst_tile_0_0_1
  // dst_tile_0_0_2
  // dst_tile_0_0_3

  //  dst_vreg_1_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_1_0_3
  // dst_tile_1_0_0
  // dst_tile_1_0_1
  // dst_tile_1_0_2

  //  dst_vreg_2_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_2_0_2
  // dst_tile_2_0_3
  // dst_tile_2_0_0
  // dst_tile_2_0_1

  //  dst_vreg_3_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_3_0_1
  // dst_tile_3_0_2
  // dst_tile_3_0_3
  // dst_tile_3_0_0

  // Each destination vreg is assembled from destination tiles in multiple
  // rotated src vregs. In the above example, if we wanted each destination tile
  // to be in correct sublane offset in a rotated vreg, say rot_src_vreg_0_1,
  // before assembling the destination tiles, we would have had to rotate
  // src_vreg_0_1 four times, creating 4 rotated vregs (instead of 1) for each
  // src vreg. In the above example, we instead rotated a src vreg src_vreg_0_1
  // only once to obtain rot_src_vreg_0_1 where the dst_tile_0_0_1 is in correct
  // final sublane offset, i.e. 2. But notice the sublane offset of
  // dst_tile_1_0_1 in the same rotated vreg. Its correct final destination
  // sublane offset is 2, but in rot_src_vreg_0_1, its offset is 4. Its sublane
  // offset is off by 2. We need to correct these sublane offsets in the final
  // assembled dst vregs. A single rotation of each assembled dst vreg is needed
  // to correct such sublane offsets. This strategy reduces the number of
  // sublane rotations required. See comments below.
  const int64_t tile_sublane_change_factor =
      src_layout.tiling()[0] / dst_layout.tiling()[0];

  dst_vreg_array.Each([&](absl::Span<const int64_t> idx,
                          Value *const dst_vreg) {
    const int64_t row = *(idx.end() - 2);
    const int64_t col = *(idx.end() - 1);
    auto [rotated_vreg_row, first_dst_tile_offset] =
        std::div(row, tile_sublane_change_factor);
    const int64_t first_dst_tile_sublane_offset =
        first_dst_tile_offset * dst_sublanes_per_tile;
    const int64_t src_vreg_array_col_start = col * dst_tiles_per_vreg;
    const int64_t src_vreg_array_col_end =
        std::min((col + 1) * dst_tiles_per_vreg,
                 rotated_src_vreg_array.dimensions().back()) -
        1;

    // TODO(tlongeri): Find a better way to slice that doesn't involve so
    // copying so many index vectors and hopefully is more concise. Probably
    // by expanding xla::Array (maybe could just expose calculate_index?).
    SmallVector<int64_t> rotated_row_starts(toArrayRef(idx));
    *(rotated_row_starts.end() - 2) = rotated_vreg_row;
    *(rotated_row_starts.end() - 1) = 0;
    SmallVector<int64_t> rotated_row_ends(idx.size());
    for (size_t i = 0; i + 1 < rotated_row_ends.size(); ++i) {
      rotated_row_ends[i] = rotated_row_starts[i] + 1;
    }
    *(rotated_row_ends.end() - 1) = rotated_src_vreg_array.dimensions().back();
    const xla::Array<Value> rotated_row_slice =
        rotated_src_vreg_array.Slice(rotated_row_starts, rotated_row_ends);
    const Value dst_tile = selectTilesFromRotatedRowVregs(
        builder, /*rotated_row_vregs=*/
        ArrayRef(rotated_row_slice.begin(), rotated_row_slice.end()),
        src_vreg_array_col_start, src_vreg_array_col_end,
        first_dst_tile_sublane_offset, dst_layout, target_shape);
    if (first_dst_tile_sublane_offset == 0) {
      // No need to rotate. First dst tile is already at offset 0, which means
      // rest of the dst tiles are also at correct sublane offset.
      *dst_vreg = dst_tile;
    } else {
      // Fix the destination tile sublane offset by rotating assembled dest vreg
      // once (See comments above). The dst vregs are fixed as follows:
      // No rotation needed.
      // dst_tile_0_0_0
      // dst_tile_0_0_1
      // dst_tile_0_0_2
      // dst_tile_0_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=1):
      // dst_tile_1_0_0
      // dst_tile_1_0_1
      // dst_tile_1_0_2
      // dst_tile_1_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=2):
      // dst_tile_2_0_0
      // dst_tile_2_0_1
      // dst_tile_2_0_2
      // dst_tile_2_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=3):
      // dst_tile_3_0_0
      // dst_tile_3_0_1
      // dst_tile_3_0_2
      // dst_tile_3_0_3
      *dst_vreg = builder.create<tpu::RotateOp>(
          dst_tile.getLoc(), dst_tile,
          target_shape[0] - first_dst_tile_sublane_offset, /*dimension=*/0,
          /*stride=*/nullptr, /*stride_dimension=*/nullptr);
    }
  });
  return dst_vreg_array;
}

// Returns true iff the layout changes involve reduced sublanes per tile.
//
// Arguments:
//  src: The existing layout.
//  dst: The new layout based on which the retiling is to be carried out.
bool isSupportedReducedSublanesRetile(
    const VectorLayout &src, const VectorLayout &dst,
    const std::array<int64_t, 2> target_shape) {
  return src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
         dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
         llvm::all_of(llvm::zip_equal(src.offsets(), dst.offsets()),
                      [](auto tup) {
                        auto [lhs, rhs] = tup;
                        return lhs.value_or(0) == rhs.value_or(0);
                      })
         // TODO (kumudbhandari): We have not tested any tile size where
         // tile[-1] != TARGET_SHAPE.lanes. It should work but needs to be
         // tested.
         && src.tiling()[1] == target_shape[1] &&
         dst.tiling()[1] == target_shape[1] &&
         dst.tiling()[0] < src.tiling()[0] &&
         src.bitwidth() == dst.bitwidth() &&
         llvm::isPowerOf2_64(src.tiling()[0]) &&
         llvm::isPowerOf2_64(dst.tiling()[0]);
}

// Copy one sublane from a vreg to another vreg.
//
// Arguments:
//  src_vreg: The source vreg to copy a sublane from.
//  src_sl_idx: The sublane index in src_vreg to copy.
//  dst_vreg: The destination vreg to copy a sublane to.
//  dst_sl_idx: The sublane index in dst_vreg to paste.
//
// Returns:
//  A new dst_vreg with the copied sublane.
Value copy_one_sublane(OpBuilder &builder, Value src_vreg, int src_sl_idx,
                       Value dst_vreg, int dst_sl_idx,
                       const std::array<int64_t, 2> target_shape) {
  if (!dst_vreg) {
    const DenseI32ArrayAttr gather_indices =
        builder.getDenseI32ArrayAttr(SmallVector<int32_t>(8, src_sl_idx));
    return builder.create<tpu::GatherOp>(src_vreg.getLoc(), src_vreg.getType(),
                                         src_vreg, gather_indices,
                                         /*dimension=*/0);
  }
  auto src_vreg_rot = builder.create<tpu::RotateOp>(
      src_vreg.getLoc(), src_vreg,
      /*amount=*/(dst_sl_idx - src_sl_idx + 8) % 8,
      /*dimension=*/0, /*stride=*/nullptr, /*stride_dimension=*/nullptr);
  auto boundIdxConst =
      std::bind(IdxConst, std::placeholders::_1, builder, src_vreg.getLoc());
  auto sublanes_mask = builder.create<tpu::CreateMaskOp>(
      src_vreg.getLoc(), VectorType::get(target_shape, builder.getI1Type()),
      ValueRange{boundIdxConst(dst_sl_idx), boundIdxConst(0)},
      ValueRange{boundIdxConst(dst_sl_idx + 1),
                 boundIdxConst(target_shape[1])});
  return builder.create<arith::SelectOp>(src_vreg.getLoc(), sublanes_mask,
                                         src_vreg_rot, dst_vreg);
}

// TODO(apaszke): Test this function properly
FailureOr<TypedValue<VectorType>> relayout(
    OpBuilder &builder, TypedValue<VectorType> v, VectorLayout src,
    const VectorLayout &dst, const std::array<int64_t, 2> target_shape) {
  const int8_t bitwidth = src.bitwidth();
  if (bitwidth != dst.bitwidth()) {
    return emitError(v.getLoc(), "Can't change bitwidth during a relayout");
  }
  const int packing = src.packing();
  VectorType vty = v.getType();
  FAILUREOR_ASSIGN_OR_RETURN(xla::Array<Value> src_tiles,
                             disassemble(builder, src, v, target_shape));
  SmallVector<int64_t> dst_tiles_shape =
      dst.tileArrayShape(vty.getShape(), target_shape);
  if (src.generalizes(dst, vty.getShape(), target_shape)) {
    return assemble(builder, vty, dst, std::move(src_tiles), target_shape)
        .getResult();
  }
  if (!src.offsets()[0].has_value() && !src.offsets()[1].has_value() &&
      src.tilesPerVreg(target_shape) == 1) {
    // A fully replicated value is always easy to relayout
    // It would be nice to be able to assert this here, but given replicated
    // values our rules can introduce equivalent expressions.
    // assert all(t is src_tiles_list[0] for t in src_tiles_list)
    xla::Array<Value> dst_tiles(
        /*sizes=*/dst.tileArrayShape(vty.getShape(), target_shape),
        /*value=*/src_tiles.data()[0]);
    return assemble(builder, vty, dst, std::move(dst_tiles), target_shape)
        .getResult();
  }
  // Try to reconcile differences in implicit dim.
  if (src.implicit_dim() != dst.implicit_dim()) {
    VectorLayout candidate(src.bitwidth(), src.offsets(), src.tiling(),
                           dst.implicit_dim());
    if (candidate.equivalentTo(src, vty.getShape(), target_shape)) {
      src = candidate;
    }
  }

  // Handle retiling from (1, 128) to (8, 128) for 32-bit data.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      src.bitwidth() == 32 && src.offsets() == LayoutOffsets{0, 0} &&
      (dst.offsets()[0] == 0 || (dst.offsets()[0] == std::nullopt &&
                                 *(src_tiles.dimensions().end() - 2) == 1)) &&
      dst.offsets()[1] == 0 && src.tiling() == std::array<int64_t, 2>{1, 128} &&
      dst.tiling() == std::array<int64_t, 2>{8, 128}) {
    xla::Array<Value> src_tiles_retiled(
        dst.tileArrayShape(vty.getShape(), target_shape));
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      for (int dst_sl_idx = 0; dst_sl_idx < 8; ++dst_sl_idx) {
        SmallVector<int64_t> src_idx(idx.begin(), idx.end());
        src_idx[src_idx.size() - 2] = 8 * idx[idx.size() - 2] + dst_sl_idx;
        if (src_idx[src_idx.size() - 2] >=
            *(src_tiles.dimensions().end() - 2)) {
          break;
        }
        src_idx[src_idx.size() - 1] = idx[idx.size() - 1] / 8;
        const int64_t src_sl_idx = idx[idx.size() - 1] % 8;
        *tile = copy_one_sublane(builder, src_tiles(src_idx), src_sl_idx, *tile,
                                 dst_sl_idx, target_shape);
      }
    });
    src = dst;
    src_tiles = std::move(src_tiles_retiled);
  } else if (  // Handle retiling from (m, 128) to (8, 128) for 32-bit data
               // where m < 8 and m is a power of 2.
               // TODO(b/306692696) Generalize relayout from tiling (m, 128) to
               // (8, 128) for any src_tiles.dimensions().
      src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      src.bitwidth() == 32 && src.offsets() == LayoutOffsets{0, 0} &&
      dst.offsets() == LayoutOffsets{0, 0} &&
      target_shape[0] % src.tiling()[0] == 0 &&
      src.tiling()[1] == target_shape[1] && dst.tiling() == target_shape &&
      *(src_tiles.dimensions().end() - 2) == 1) {
    xla::Array<Value> src_tiles_retiled(
        dst.tileArrayShape(vty.getShape(), target_shape));
    src_tiles_retiled.Each(
        [&](const absl::Span<const int64_t> idx, Value *const new_src_tile) {
          const int64_t tiles_per_vreg = src.tilesPerVreg(target_shape);
          const int64_t dst_col = idx.back();
          const int64_t src_col = dst_col / tiles_per_vreg;
          const int64_t start_slane_idx =
              src.tiling()[0] * (dst_col % tiles_per_vreg);
          SmallVector<int64_t> src_idx(toArrayRef(idx));
          src_idx.back() = src_col;
          Value src_tile = src_tiles(src_idx);
          if (start_slane_idx) {
            SmallVector<int32_t> slane_idxs;
            slane_idxs.reserve(target_shape[0]);
            for (int i = 0; i < target_shape[0]; ++i) {
              slane_idxs.push_back(start_slane_idx + (i % src.tiling()[0]));
            }
            const DenseI32ArrayAttr gather_indices =
                builder.getDenseI32ArrayAttr(slane_idxs);
            *new_src_tile = builder.create<tpu::GatherOp>(
                v.getLoc(), src_tile.getType(), src_tile, gather_indices,
                /*dimension=*/0);
          } else {
            *new_src_tile = src_tile;
          }
        });
    src = dst;
    src_tiles = std::move(src_tiles_retiled);
  } else if (  // TODO(b/265133506): Generalize retiling.
               // (8,128) -> (8 * packing,128) tiling change for packed type.
      src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone && bitwidth < 32 &&
      32 % bitwidth == 0 && src.offsets() == dst.offsets() &&
      src.tiling() == std::array<int64_t, 2>{8, 128} &&
      dst.tiling() == std::array<int64_t, 2>{8 * dst.packing(), 128}) {
    const VectorLayout new_src(src.bitwidth(), src.offsets(), dst.tiling());
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), target_shape));
    int vty_packing = dst.packing();
    VectorType vreg_x32 =
        vty.getElementType().isSignlessInteger()
            ? VectorType::get(target_shape, builder.getI32Type())
            : VectorType::get(target_shape, builder.getF32Type());
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      const int vreg_part = idx.back() % vty_packing;
      SmallVector<Value, 8> parts;
      parts.reserve(vty_packing);
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      src_idx[src_idx.size() - 2] *= vty_packing;
      src_idx[src_idx.size() - 1] /= vty_packing;
      for (int i = 0; i < vty_packing; ++i) {
        parts.push_back(builder.create<tpu::UnpackSubelementsOp>(
            v.getLoc(), vreg_x32, src_tiles(src_idx), vreg_part));
        if (src_idx[src_idx.size() - 2] <
            src_tiles.dim(src_tiles.num_dimensions() - 2) - 1) {
          ++src_idx[src_idx.size() - 2];
        }
      }
      *tile = builder.create<tpu::PackSubelementsOp>(
          v.getLoc(), src_tiles.begin()->getType(), parts,
          tpu::PackFormat::kCompressed);
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  } else if (  // Handle retiling from (1, 128 * packing) to (packing, 128) for
               // packed data.
               // We do compressed unpacking followed by interleaved packing.
               // TODO(tlongeri): This can be used as a first step before using
               // a generalized retiling where we only move sublanes around
               // (without packing/unpacking).
               // TODO(tlongeri): Interleaved unpacking followed by interleaved
               // packing (but with different pairings) might also be
               // interesting if the next step is a retile, since we can also
               // match corresponding elements without shifting. It's just that
               // the tiles are not adjacent (no contiguous vreg slice).
      src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone && bitwidth < 32 &&
      32 % bitwidth == 0 && src.offsets() == dst.offsets() &&
      src.tiling() == std::array<int64_t, 2>{1, 128 * packing} &&
      dst.tiling() == std::array<int64_t, 2>{packing, 128}) {
    // To illustrate, consider a 2 x 16 16-bit shape laid out in vregs of
    // 4 sublanes and 2 lanes (this is convenient for to keep the example small
    // yet non-trivial) with (1, 4) tiling. We will relayout to (2, 2) tiling.
    //
    // The vreg slice is 1 x 16, that is, the vreg contains the data for a
    // 1 x 16 window of the logical shape.
    //
    // [a b c d e f g h i j k l m n o p] -> vreg 1
    // [A B C D E F G H I J K L M N O P] -> vreg 2
    //
    // Note: we support multiple vregs per row of the logical shape, but we use
    //       one here just to keep the example small.
    //
    // When we do a compressed unpack, the resulting vregs effectively have a
    // tiling of (1, 2) and cover a vreg slice of 1 x 8 logical elements.
    //
    // [a b c d e f g h] -> vreg 1, part 1   [i j k l m n o p] -> vreg 1, part 2
    // [A B C D E F G H] -> vreg 2, part 1   [I J K L M N O P] -> vreg 2, part 2
    //
    // It is clear that if combine vreg 1, part 1 and vreg 2, part 1 we get data
    // that covers a 2 x 8 vreg slice. Note, however, that we will have to mind
    // the internal ordering of the vreg.
    //
    // [a b c d e f g h                      [i j k l m n o p
    //  A B C D E F G H] -> new vreg 1        I J K L M N O P] -> new vreg 2
    //
    // To see if we can get the right internal ordering that we need for (2, 2)
    // tiling, let's break new vreg 1 into (1, 2) rows, which correspond to
    // sublanes when unpacked and half-sublanes when packed.
    //
    // [(a b) (c d) (e f) (g h)
    //  (A B) (C D) (E F) (G H)]
    //
    // The sublane order for the vreg parts is [(a b) (c d) ...] for vreg 1,
    // part 1 and [(A B) (C D) ...] for vreg 2, part 1.
    //
    // The desired half-sublane order, for packed (2, 2) tiling, is
    // [(a b) (A B) (c d) (C D) ...]. That is, traverse down each column before
    // moving to the next one. This is exactly an interleaving of the sublanes
    // of the vreg parts.
    const VectorLayout new_src(src.bitwidth(), src.offsets(),
                               std::array<int64_t, 2>{packing, 128});
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), target_shape));
    const VectorType vreg_x32 =
        vty.getElementType().isSignlessInteger()
            ? VectorType::get(target_shape, builder.getI32Type())
            : VectorType::get(target_shape, builder.getF32Type());
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      SmallVector<Value> parts;
      parts.reserve(packing);
      SmallVector<int64_t> src_idx(toArrayRef(idx));
      *(src_idx.end() - 2) *= packing;
      const int64_t vreg_part = *(src_idx.end() - 1) % packing;
      *(src_idx.end() - 1) /= packing;
      for (int i = 0; i < packing; ++i) {
        parts.push_back(builder.create<tpu::UnpackSubelementsOp>(
            v.getLoc(), vreg_x32, src_tiles(src_idx), vreg_part));
        if (*(src_idx.end() - 2) < *(src_tiles.dimensions().end() - 2)) {
          ++*(src_idx.end() - 2);
        }  // The rest is padding, so just pick any of the input parts (but not
           // an arbitrary vreg so we don't add an extra dependency).
      }
      *tile = builder.create<tpu::PackSubelementsOp>(
          v.getLoc(), src_tiles.begin()->getType(), parts,
          tpu::PackFormat::kInterleaved);
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  } else if (  // Handle retiling from (8, 128, -2) to (8, 128) for 32-bit data.
               // This drops the implicit second minor dimension.
      src.implicit_dim() == VectorLayout::ImplicitDim::kSecondMinor &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      src.bitwidth() == 32 && src.offsets() == dst.offsets() &&
      src.offsets() == LayoutOffsets{0, 0} && src.tiling() == dst.tiling() &&
      src.tiling() == std::array<int64_t, 2>{8, 128}) {
    xla::Array<Value> src_tiles_retiled(
        dst.tileArrayShape(vty.getShape(), target_shape));
    src_tiles_retiled.Each(
        [&](const absl::Span<const int64_t> idx, Value *tile) {
          for (int dst_sl_idx = 0; dst_sl_idx < 8; ++dst_sl_idx) {
            SmallVector<int64_t> src_idx(idx.begin(), idx.end());
            auto second_minor_idx = idx.size() - 2;
            src_idx[second_minor_idx] = 8 * idx[second_minor_idx] + dst_sl_idx;
            if (src_idx[second_minor_idx] >= src_tiles.dim(second_minor_idx)) {
              break;
            }
            *tile = copy_one_sublane(builder, src_tiles(src_idx), 0, *tile,
                                     dst_sl_idx, target_shape);
          }
        });
    src = dst;
    src_tiles = std::move(src_tiles_retiled);
  }

  if (isSupportedReducedSublanesRetile(src, dst, target_shape)) {
    src_tiles = retileToReducedSublanes(builder, vty.getShape(), src, src_tiles,
                                        dst, target_shape);
    src = dst;
  }

  // Fix up the offsets, assuming everything else matches between src and dst.
  if (src.tiling() == dst.tiling() &&
      src.implicit_dim() == dst.implicit_dim()) {
    const auto &tiling = src.tiling();
    // TODO(apaszke): Changing an offset might add or remove one vreg.
    if (dst_tiles_shape != src_tiles.dimensions()) {
      return emitError(
          v.getLoc(), "Not implemented: Offsets changing the vreg array shape");
    }
    xla::Array<Value> dst_tiles = src_tiles;

    // Shifting rows
    int row_diff;
    if (!src.offsets()[0].has_value()) {
      row_diff = 0;
    } else if (!dst.offsets()[0].has_value()) {
      return emitError(v.getLoc(), "Not implemented: Sublane broadcast");
    } else {
      row_diff = *dst.offsets()[0] - *src.offsets()[0];
    }

    if (row_diff != 0) {  // This is an easy case, because we never need to
                          // combine multiple vregs.
      const SmallVector<int64_t> implicit_shape =
          src.implicitShape(vty.getShape());
      if (implicit_shape[implicit_shape.size() - 2] != 1) {
        return emitError(v.getLoc(),
                         "Not implemented: Row shifts for multi-row values");
      }
      const int64_t src_sublane = *src.offsets()[0] / packing;
      const int64_t dst_sublane = *dst.offsets()[0] / packing;
      if (int64_t sublane_diff = dst_sublane - src_sublane) {
        if (sublane_diff < 0) {
          sublane_diff += target_shape[0];
        }
        src_tiles.Each([&](absl::Span<const int64_t> idx, Value tile) {
          dst_tiles(idx) =
              builder
                  .create<tpu::RotateOp>(v.getLoc(), tile,
                                         /*amount=*/sublane_diff,
                                         /*dimension=*/0, /*stride=*/nullptr,
                                         /*stride_dimension=*/nullptr)
                  .getResult();
        });
      }
      const int src_subelem = *src.offsets()[0] % packing;
      const int dst_subelem = *dst.offsets()[0] % packing;
      if (src_subelem != dst_subelem) {
        const int subelem_diff = dst_subelem - src_subelem;
        const int shift_bits = bitwidth * std::abs(subelem_diff);
        VectorType bits_vreg_ty =
            VectorType::get(target_shape, builder.getI32Type());
        auto shift_vreg = builder.create<arith::ConstantOp>(
            v.getLoc(), bits_vreg_ty,
            DenseElementsAttr::get(bits_vreg_ty, shift_bits));
        dst_tiles.Each([&](absl::Span<const int64_t> /*idx*/, Value *tile) {
          auto bit_tile = builder.create<tpu::BitcastVregOp>(
              v.getLoc(), bits_vreg_ty, *tile);
          Operation *shift_tile;
          if (subelem_diff > 0) {
            shift_tile =
                builder.create<arith::ShLIOp>(v.getLoc(), bit_tile, shift_vreg);
          } else {  // subelem_diff < 0
            CHECK_LT(subelem_diff, 0);
            shift_tile = builder.create<arith::ShRUIOp>(v.getLoc(), bit_tile,
                                                        shift_vreg);
          }
          *tile = builder
                      .create<tpu::BitcastVregOp>(v.getLoc(), tile->getType(),
                                                  shift_tile->getResult(0))
                      .getResult();
          return absl::OkStatus();
        });
      }
    }
    int64_t col_diff;
    if (!src.offsets()[1].has_value()) {
      col_diff = 0;
    } else if (!dst.offsets()[1].has_value()) {
      return emitError(v.getLoc(), "Not implemented: Lane broadcast");
    } else {
      col_diff = *dst.offsets()[1] - *src.offsets()[1];
    }
    if (col_diff != 0) {
      if (row_diff != 0) {
        return emitError(v.getLoc(),
                         "Not implemented: Both columns and rows are shifted");
      }
      if (bitwidth != 32 || tiling != target_shape) {
        return emitError(v.getLoc(),
                         "Not implemented: Only 32-bit column shifts for "
                         "native layouts supported");
      }
      TPU_ASSERT_GE_LOC(v.getLoc(), src_tiles.num_dimensions(), 1);
      std::optional<tpu::CreateMaskOp> maybe_create_mask;
      if (*(src_tiles.dimensions().end() - 1) > 1) {
        int64_t lane_start, lane_end;
        if (col_diff > 0) {
          lane_start = 0;
          lane_end = col_diff;
        } else {  // col_diff < 0
          lane_start = target_shape[1] + col_diff;
          lane_end = target_shape[1];
        }
        auto boundIdxConst =
            std::bind(IdxConst, std::placeholders::_1, builder, v.getLoc());
        maybe_create_mask = builder.create<tpu::CreateMaskOp>(
            v.getLoc(), VectorType::get(target_shape, builder.getI1Type()),
            ValueRange{boundIdxConst(0), boundIdxConst(lane_start)},
            ValueRange{boundIdxConst(target_shape[0]),
                       boundIdxConst(lane_end)});
      }
      src_tiles.Each([&](absl::Span<const int64_t> idx, Value *tile) {
        *tile = builder
                    .create<tpu::RotateOp>(v.getLoc(), *tile,
                                           /*amount=*/col_diff < 0
                                               ? target_shape[1] + col_diff
                                               : col_diff,
                                           /*dimension=*/1, /*stride=*/nullptr,
                                           /*stride_dimension=*/nullptr)
                    .getResult();
      });
      src_tiles.Each([&](absl::Span<const int64_t> idx, Value rot_tile) {
        Value prev_rot_tile;
        if (col_diff > 0) {
          if (*(idx.end() - 1) != 0) {
            SmallVector<int64_t> prev_idx(idx.begin(), idx.end());
            --*(prev_idx.end() - 1);
            prev_rot_tile = src_tiles(prev_idx);
          }
        } else {  // col_diff < 0
          if (*(idx.end() - 1) != *(src_tiles.dimensions().end() - 1) - 1) {
            SmallVector<int64_t> prev_idx(idx.begin(), idx.end());
            ++*(prev_idx.end() - 1);
            prev_rot_tile = src_tiles(prev_idx);
          }
        }
        if (prev_rot_tile != nullptr) {
          rot_tile = builder.create<arith::SelectOp>(
              v.getLoc(), maybe_create_mask->getResult(), prev_rot_tile,
              rot_tile);
        }
        dst_tiles(idx) = rot_tile;
      });
    }
    return assemble(builder, vty, dst, std::move(dst_tiles), target_shape)
        .getResult();
  }
  // TODO(apaszke): Implement general relayout
  return emitError(v.getLoc(),
                   "Not implemented: Unsupported layout change for ")
         << vty << ": " << src << " -> " << dst;
}

// TODO(apaszke): Implement a debug mode that inserts additional assertions.
// For example, we should verify that ops that were supposed to generate
// replicated outputs satisfy that requirement.
LogicalResult applyLayoutOp(RewriteContext &ctx, Operation &op) {
  // When an operation does not have any operands, the layout_in tuple is empty.
  // If one of the operands is not of vector type, the corresponding entry in
  // the layout_in tuple will be None. The same applies to the results of the
  // operation and the layout_out tuple.
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layouts_out,
                             getOutLayouts(op, ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layouts_in,
                             getInLayouts(op, ctx.target_shape));
  if (!layouts_in.empty() && !isa<tpu::AssumeLayoutOp>(op)) {
    // Relayout the operands, if their requested input layouts don't match the
    // layouts in which they were produced.
    for (auto [idx, tup] :
         llvm::enumerate(llvm::zip(op.getOperands(), layouts_in))) {
      auto [operand, li] = tup;
      auto vector_operand = dyn_cast<TypedValue<VectorType>>(operand);
      TPU_ASSERT_EQ_OP(vector_operand != nullptr, li.has_value());
      if (vector_operand == nullptr) {
        continue;
      }
      auto vty = vector_operand.getType();

      // The operand should always be an Operation (and not a BlockArgument)
      // since we expect the FuncOp to have only memrefs and semaphores as
      // arguments.
      auto op_result = dyn_cast<OpResult>(vector_operand);
      if (op_result == nullptr) {
        return op.emitError("Expected operand to be an operation result");
      }
      Operation *const def_op = op_result.getOwner();
      TPU_ASSERT_OP(def_op);
      const unsigned res_idx = op_result.getResultNumber();
      FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                                 getOutLayouts(*def_op, ctx.target_shape));
      const Layout lo = def_layouts[res_idx];
      TPU_ASSERT_OP(lo.has_value());
      if (lo->generalizes(*li, vty.getShape(), ctx.target_shape)) {
        continue;
      }
      OpBuilder builder(&op);
      FAILUREOR_ASSIGN_OR_RETURN(Value new_v,
                                 relayout(builder, vector_operand, /*src=*/*lo,
                                          /*dst=*/*li, ctx.target_shape));
      op.setOperand(idx, new_v);
    }
  }

  const bool no_vector_args =
      llvm::none_of(layouts_out,
                    [](Layout layout) { return layout.has_value(); }) &&
      llvm::none_of(layouts_in,
                    [](Layout layout) { return layout.has_value(); });
  if (no_vector_args && op.getRegions().empty()) {
    // We don't need to do anything for scalar operations.
    if (!op.getOperands().empty()) {
      op.removeAttr("in_layout");
    }
    if (!op.getResults().empty()) {
      op.removeAttr("out_layout");
    }
    return success();
  }
  if (auto rule_it = rules().find(op.getName().getStringRef());
      rule_it != rules().end()) {
    const rule_type &rule = rule_it->getValue();
    return rule(ctx, op, layouts_in, layouts_out);
  }
  if (OpTrait::hasElementwiseMappableTraits(&op)) {
    return elementwise_op_rule(ctx, op, layouts_in, layouts_out);
  }
  return op.emitError("Not implemented: Unsupported operation: ")
         << op.getName();
}

LogicalResult applyLayoutBlock(RewriteContext &ctx, Block &block) {
  // We'll be modifying the block, so use early increment.
  for (Operation &op : make_early_inc_range(block)) {
    if (failed(applyLayoutOp(ctx, op))) {
      return failure();
    }
  }
  return success();
}

// Rewrites the function according to layout annotations of its operations.
//
//   Args:
//     ctx: The context used for rewriting.
//     f: An MLIR function to be rewritten.
LogicalResult applyLayoutFunc(RewriteContext &ctx, func::FuncOp f) {
  if (f->getNumRegions() != 1) {
    return f.emitError("Expected FuncOp to have a single region");
  }
  if (!f.getBody().hasOneBlock()) {
    return f.emitError("Expected FuncOp to have a single block");
  }
  return applyLayoutBlock(ctx, f.getBody().front());
}

struct ApplyVectorLayoutPass
    : public impl::ApplyVectorLayoutPassBase<ApplyVectorLayoutPass> {
  ApplyVectorLayoutPass(int hardware_generation_, int lane_count_,
                        int sublane_count_, int mxu_contracting_size_,
                        int mxu_noncontracting_size_,
                        int max_sublanes_in_scratch_) {
    hardware_generation = hardware_generation_;
    sublane_count = sublane_count_;
    lane_count = lane_count_;
    mxu_contracting_size = mxu_contracting_size_;
    mxu_noncontracting_size = mxu_noncontracting_size_;
    max_sublanes_in_scratch = max_sublanes_in_scratch_;
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      signalPassFailure();
      return;
    }
    func::FuncOp func = getOperation();
    RewriteContext ctx{func,
                       hardware_generation,
                       {sublane_count, lane_count},
                       {mxu_contracting_size, mxu_noncontracting_size},
                       max_sublanes_in_scratch};
    if (failed(applyLayoutFunc(ctx, func))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createApplyVectorLayoutPass(
    int hardware_generation, int lane_count, int sublane_count,
    int mxu_contracting_size, int mxu_noncontracting_size,
    int max_sublanes_in_scratch) {
  return std::make_unique<ApplyVectorLayoutPass>(
      hardware_generation, lane_count, sublane_count, mxu_contracting_size,
      mxu_noncontracting_size, max_sublanes_in_scratch);
}

}  // namespace mlir::tpu
