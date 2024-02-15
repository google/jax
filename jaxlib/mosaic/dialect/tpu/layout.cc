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

#include "jaxlib/mosaic/dialect/tpu/layout.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "absl/log/check.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"

namespace mlir::tpu {

bool RectangularVregBounds::maskVariesAlong(
    const Direction direction,
    const std::array<int64_t, 2> target_shape) const {
  switch (direction) {
    case Direction::kSublanes:
      return starts_[0] != 0 || ends_[0] != target_shape[0];
    case Direction::kLanes:
      return starts_[1] != 0 || ends_[1] != target_shape[1];
    case Direction::kSubelements:
      return false;
  }
}

FailureOr<TypedValue<VectorType>> RectangularVregBounds::getVectorMask(
    OpBuilder& builder, const Location loc, const int /*generation*/,
    const std::array<int64_t, 2> target_shape) const {
  auto boundIdxConst = std::bind(IdxConst, std::placeholders::_1, builder, loc);
  return cast<TypedValue<VectorType>>(
      builder
          .create<tpu::CreateMaskOp>(
              loc, VectorType::get(target_shape, builder.getI1Type()),
              /*low=*/
              ValueRange{boundIdxConst(starts_[0]), boundIdxConst(starts_[1])},
              /*high=*/
              ValueRange{boundIdxConst(ends_[0]), boundIdxConst(ends_[1])})
          .getResult());
}

DenseBoolArrayAttr RectangularVregBounds::getSublaneMask(
    MLIRContext* mlir_ctx, const std::array<int64_t, 2> target_shape) const {
  llvm::SmallVector<bool, 8> sublane_mask(target_shape[0], false);
  for (int64_t i = starts_[0]; i < ends_[0]; ++i) {
    sublane_mask[i] = true;
  }
  return DenseBoolArrayAttr::get(mlir_ctx, sublane_mask);
}

namespace {

// Represents a subset of a (packed) 1D vector register.
//
// All indices below are scaled up by the packing. That is, the maximal stop
// offset for a register containing 16-bit values is twice as large as for
// a register containing 32-bit values.
//
// Standard 1D packing is used. The values start laid out in the low half of the
// first sublane, then wrap around to the higher half of the first sublane, etc.
//
// Attributes:
//   layout: The layout used to generate the bounds.
//   start_offset: Index of the element from which the mask begins (inclusive).
//   stop_offset: Index of the element at which the mask ends (exclusive).
class SingleRowVRegBounds : public VRegDataBounds {
 public:
  SingleRowVRegBounds(const VectorLayout& layout, const int64_t start_offset,
                      const int64_t stop_offset,
                      const std::array<int64_t, 2> target_shape)
      : layout_(layout),
        start_offset_(start_offset),
        stop_offset_(stop_offset) {
    CHECK(0 <= start_offset_ && start_offset_ < stop_offset_ &&
          stop_offset_ <= getEntriesPerVreg(target_shape));
  }

  // Total number of entries contained in a vreg.
  int64_t getEntriesPerVreg(const std::array<int64_t, 2> target_shape) const {
    return target_shape[0] * target_shape[1] * layout_.packing();
  }

  // See base class.
  bool maskVariesAlong(
      const Direction direction,
      const std::array<int64_t, 2> target_shape) const override {
    if (start_offset_ == 0 && stop_offset_ == getEntriesPerVreg(target_shape)) {
      return false;
    }
    const int64_t entries_per_vreg = getEntriesPerVreg(target_shape);
    switch (direction) {
      case Direction::kSublanes:
        return start_offset_ >= target_shape[1] ||
               stop_offset_ < entries_per_vreg - target_shape[1];
      case Direction::kLanes:
        return true;
      case Direction::kSubelements:
        return start_offset_ % layout_.packing() != 0 ||
               stop_offset_ % layout_.packing() != 0;
    }
  }

  // See base class.
  FailureOr<TypedValue<VectorType>> getVectorMask(
      OpBuilder& builder, const Location loc, const int generation,
      const std::array<int64_t, 2> target_shape) const override {
    if (maskVariesAlong(Direction::kSubelements, target_shape)) {
      return emitError(loc, "Not implemented");
    }
    const auto i32_vreg = VectorType::get(target_shape, builder.getI32Type());
    const auto getI32VregConstant = [&](const int32_t v) {
      return builder.create<arith::ConstantOp>(
          loc, i32_vreg, DenseElementsAttr::get(i32_vreg, v));
    };
    if (layout_.bitwidth() != 32 &&
        (start_offset_ % (target_shape[1] * layout_.packing()) != 0 ||
         stop_offset_ % (target_shape[1] * layout_.packing()) != 0)) {
      return emitError(loc, "Not implemented");
    }
    const Value start = getI32VregConstant(start_offset_ / layout_.packing());
    const Value end = getI32VregConstant(stop_offset_ / layout_.packing());
    const Value iota = builder.create<tpu::IotaOp>(loc, i32_vreg, nullptr);
    return cast<TypedValue<VectorType>>(
        builder
            .create<arith::AndIOp>(
                loc,
                builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              iota, start),
                builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              iota, end))
            .getResult());
  }

  // See base class.
  DenseBoolArrayAttr getSublaneMask(
      MLIRContext* mlir_ctx,
      const std::array<int64_t, 2> target_shape) const override {
    const int64_t start_sublane =
        start_offset_ / layout_.packing() / target_shape[1];
    const int64_t end_sublane = llvm::divideCeil(
        llvm::divideCeil(stop_offset_, layout_.packing()), target_shape[1]);

    llvm::SmallVector<bool> sublane_mask(target_shape[0], false);
    for (int64_t i = start_sublane; i < end_sublane; ++i) {
      sublane_mask[i] = true;
    }
    return DenseBoolArrayAttr::get(mlir_ctx, sublane_mask);
  }

 private:
  VectorLayout layout_;
  int64_t start_offset_;
  int64_t stop_offset_;
};

// Represents the data bounds within a vector register with tiled and
// potentially packed data.
//
// Note that the (packed) sublane offset from start_offset and (packed) sublane
// bound from end_offsets apply to all tiles within a vreg. On the other hand,
// the lane offset from start_offset only applies to the first tile, while
// lane bound from end_offset only applies to the last used tile.
//
// Attributes:
//   layout: The layout of the value, mainly used for its bitwidth and tiling.
//     Note that the layout offsets SHOULD NOT be used.
//   num_tiles: The number of tiles at the beginning of the vreg that contain
//     actual data.
//   start_offsets: The lane and (packed) sublane offset within the first tile.
//   end_offsets: The lane and (packed) sublane offset within the last used
//   tile.
class TiledRectangularVregBounds : public VRegDataBounds {
 public:
  TiledRectangularVregBounds(const VectorLayout& layout,
                             const int64_t num_tiles,
                             const std::array<int64_t, 2> start_offsets,
                             const std::array<int64_t, 2> end_offsets,
                             const std::array<int64_t, 2> target_shape)
      : layout_(layout),
        num_tiles_(num_tiles),
        start_offsets_(start_offsets),
        end_offsets_(end_offsets) {
    CHECK(layout_.tiling()[1] == target_shape[1]);
    CHECK(0 < num_tiles_ && num_tiles_ <= layout.tilesPerVreg(target_shape));
    for (auto [o, t] : llvm::zip(start_offsets_, layout_.tiling())) {
      CHECK(0 <= o && o < t);
    }
    for (auto [o, t] : llvm::zip(end_offsets_, layout_.tiling())) {
      CHECK(0 <= o && o <= t);
    }
  }

  bool usesAllTiles(const std::array<int64_t, 2> target_shape) const {
    return num_tiles_ == layout_.tilesPerVreg(target_shape);
  }

  // See base class.
  bool maskVariesAlong(
      const Direction direction,
      const std::array<int64_t, 2> target_shape) const override {
    switch (direction) {
      case Direction::kSublanes:
        return !usesAllTiles(target_shape) || start_offsets_[0] != 0 ||
               end_offsets_[0] != layout_.tiling()[0];
      case Direction::kLanes:
        return start_offsets_[1] != 0 || end_offsets_[1] != layout_.tiling()[1];
      case Direction::kSubelements:
        return start_offsets_[0] % layout_.packing() != 0 ||
               end_offsets_[0] % layout_.packing() != 0;
    }
  }

  // See base class.
  FailureOr<TypedValue<VectorType>> getVectorMask(
      OpBuilder& builder, const Location loc, const int generation,
      const std::array<int64_t, 2> target_shape) const override {
    const IntegerType i1 = builder.getI1Type();
    FAILUREOR_ASSIGN_OR_RETURN(
        const VectorType mask_vreg_ty, [&]() -> FailureOr<VectorType> {
          // I'm pretty sure this works for all bitwidths, but it's untested.
          if (maskVariesAlong(Direction::kSubelements, target_shape)) {
            if (layout_.packing() != 2) {
              // TODO(b/300082350): Generalize this
              return emitError(loc, "Not implemented");
            }
            // For older TPUs, we virtualize masking, but only for simple cases.
            if (generation < 4) {
              if (num_tiles_ > 1) {
                return emitError(loc, "Not implemented");
              }
              return VectorType::get(target_shape, i1);
            } else {
              return VectorType::get(
                  {target_shape[0], target_shape[1], layout_.packing()}, i1);
            }
            return VectorType::get({target_shape[0], target_shape[1], 2}, i1);
          }
          return VectorType::get(target_shape, i1);
        }());
    if (isComplete(target_shape)) {
      return cast<TypedValue<VectorType>>(
          builder
              .create<arith::ConstantOp>(
                  loc, mask_vreg_ty,
                  DenseElementsAttr::get(mask_vreg_ty,
                                         builder.getBoolAttr(true)))
              .getResult());
    }
    Value mask = nullptr;
    CHECK_GE(num_tiles_, 0);
    const int packing = layout_.packing();
    const int64_t start_sub = start_offsets_[0] / packing;
    const int64_t end_sub = llvm::divideCeil(end_offsets_[0], packing);
    CHECK_LE(0, start_sub);
    CHECK_LT(start_sub, end_sub);
    CHECK_LE(end_sub, target_shape[0]);
    const int64_t sublanes_per_tile = layout_.sublanesPerTile(target_shape);
    for (int64_t tile = 0; tile < num_tiles_; ++tile) {
      const int64_t sublane_offset = sublanes_per_tile * tile;
      const int64_t row_offset = sublane_offset * layout_.packing();
      const int64_t start_lane = tile == 0 ? start_offsets_[1] : 0;
      const int64_t end_lane =
          tile == num_tiles_ - 1 ? end_offsets_[1] : target_shape[1];
      CHECK_LE(0, start_lane);
      CHECK_LT(start_lane, end_lane);
      CHECK_LE(end_lane, target_shape[1]);
      auto boundIdxConst =
          std::bind(IdxConst, std::placeholders::_1, builder, loc);
      // TODO(apaszke): For loads/stores whole sublanes are covered by the
      // sublane mask, so we can focus only on lanes and partial sublanes.
      Value tile_mask = builder.create<CreateMaskOp>(
          loc, mask_vreg_ty,
          ValueRange{boundIdxConst(sublane_offset + start_sub),
                     boundIdxConst(start_lane)},
          ValueRange{boundIdxConst(sublane_offset + end_sub),
                     boundIdxConst(end_lane)});
      if (maskVariesAlong(Direction::kSubelements, target_shape)) {
        int64_t start_row = start_offsets_[0] + row_offset;
        int64_t end_row = end_offsets_[0] + row_offset;
        if (generation >= 4) {
          // Only use non-trivial start/end if they don't fall on sublane
          // boundary. Otherwise CreateMaskOp already does the right thing. This
          // lets us use cheaper instruction sequences on TPUv4.
          if (start_offsets_[0] % layout_.packing() == 0) {
            start_row = 0;
          }
          if (end_offsets_[0] % layout_.packing() == 0) {
            end_row = target_shape[0] * layout_.packing();
          }
          auto submask = builder.create<tpu::CreateSubelementMaskOp>(
              loc, mask_vreg_ty, start_row, end_row, layout_.packing());
          tile_mask = builder.create<arith::AndIOp>(loc, tile_mask, submask);
        } else {  // generation < 4
          if (num_tiles_ > 1) {
            return emitError(loc,
                             "Not implemented: TPU generations before 4 cannot "
                             "handle all bf16 masking");
          }
          const auto getMaskCst = [&](const uint64_t v) {
            const auto int_mask_ty =
                VectorType::get(target_shape, builder.getI32Type());
            return builder.create<arith::ConstantOp>(
                loc, int_mask_ty,
                DenseElementsAttr::get(
                    int_mask_ty, builder.getIntegerAttr(builder.getI32Type(),
                                                        APInt(32, v))));
          };
          Value tile_bitmask = builder.create<arith::SelectOp>(
              loc, tile_mask, getMaskCst(0xFFFFFFFF), getMaskCst(0));
          if (start_row % 2 != 0) {
            auto row_mask = builder.create<tpu::CreateMaskOp>(
                loc, mask_vreg_ty,
                ValueRange{boundIdxConst(start_row / 2), boundIdxConst(0)},
                ValueRange{boundIdxConst(start_row / 2 + 1),
                           boundIdxConst(target_shape[1])});
            auto row_bitmask = builder.create<arith::SelectOp>(
                loc, row_mask, getMaskCst(0xFFFF0000), getMaskCst(0xFFFFFFFF));
            tile_bitmask =
                builder.create<arith::AndIOp>(loc, tile_bitmask, row_bitmask);
          }
          if (end_row % 2 != 0) {
            auto row_mask = builder.create<tpu::CreateMaskOp>(
                loc, mask_vreg_ty,
                ValueRange{boundIdxConst(end_row / 2), boundIdxConst(0)},
                ValueRange{boundIdxConst(end_row / 2 + 1),
                           boundIdxConst(target_shape[1])});
            auto row_bitmask = builder.create<arith::SelectOp>(
                loc, row_mask, getMaskCst(0xFFFF), getMaskCst(0xFFFFFFFF));
            tile_bitmask =
                builder.create<arith::AndIOp>(loc, tile_bitmask, row_bitmask);
          }
          return cast<TypedValue<VectorType>>(tile_bitmask);
        }
      }
      mask = mask == nullptr
                 ? tile_mask
                 : builder.create<arith::OrIOp>(loc, tile_mask, mask);
    }
    CHECK(mask != nullptr);
    return cast<TypedValue<VectorType>>(mask);
  }

  // See base class
  DenseBoolArrayAttr getSublaneMask(
      MLIRContext* mlir_ctx,
      const std::array<int64_t, 2> target_shape) const override {
    llvm::SmallVector<bool> mask(target_shape[0], false);
    const int64_t start = start_offsets_[0] / layout_.packing();
    const int64_t end = llvm::divideCeil(end_offsets_[0], layout_.packing());
    const int64_t sublanes_per_tile = layout_.sublanesPerTile(target_shape);
    const int64_t sublane_bound = num_tiles_ * sublanes_per_tile;
    for (int64_t sub = 0; sub < sublane_bound; sub += sublanes_per_tile) {
      for (int64_t i = sub + start; i < sub + end; ++i) {
        CHECK(!mask[i]);
        mask[i] = true;
      }
    }
    return DenseBoolArrayAttr::get(mlir_ctx, mask);
  }

 private:
  VectorLayout layout_;
  int64_t num_tiles_;
  std::array<int64_t, 2> start_offsets_;
  std::array<int64_t, 2> end_offsets_;
};

mlir::ParseResult parseOffset(llvm::StringRef* data,
                              std::optional<int64_t>* result) {
  int64_t int_result;
  if (data->consume_front("*")) {
    *result = std::nullopt;
    return success();
  }
  if (!data->consumeInteger(10, int_result)) {
    *result = int_result;
    return success();
  }
  return failure();
}

std::array<int64_t, 2> nativeTiling(const int8_t bitwidth,
                                    const std::array<int64_t, 2> target_shape) {
  const int packing = 32 / bitwidth;
  return {target_shape[0] * packing, target_shape[1]};
}

}  // namespace

std::tuple<std::optional<int64_t>, std::optional<int64_t>, int64_t, int64_t,
           int8_t, VectorLayout::ImplicitDim>
VectorLayout::as_tuple() const {
  return std::make_tuple(offsets_[0], offsets_[1], tiling_[0], tiling_[1],
                         bitwidth_, implicit_dim_);
}

bool VectorLayout::operator==(const VectorLayout& other) const {
  return as_tuple() == other.as_tuple();
}

bool VectorLayout::hasNativeTiling(
    const std::array<int64_t, 2> target_shape) const {
  return tiling_ == nativeTiling(bitwidth_, target_shape);
}

llvm::SmallVector<int64_t> VectorLayout::implicitShape(
    ArrayRef<int64_t> shape) const {
  CHECK(!shape.empty());
  switch (implicit_dim_) {
    case ImplicitDim::kNone:
      return llvm::SmallVector<int64_t>(shape);
    case ImplicitDim::kMinor: {
      llvm::SmallVector<int64_t> implicit_shape;
      implicit_shape.reserve(shape.size() + 1);
      implicit_shape.append(shape.begin(), shape.end());
      implicit_shape.push_back(1);
      return implicit_shape;
    }
    case ImplicitDim::kSecondMinor: {
      llvm::SmallVector<int64_t> implicit_shape;
      implicit_shape.reserve(shape.size() + 1);
      implicit_shape.append(shape.begin(), std::prev(shape.end()));
      implicit_shape.push_back(1);
      implicit_shape.push_back(shape.back());
      return implicit_shape;
    }
  }
}

llvm::SmallVector<int64_t> VectorLayout::tileArrayImplicitShape(
    const ArrayRef<int64_t> shape,
    const std::array<int64_t, 2> target_shape) const {
  const std::array<int64_t, 2> vreg_slice = vregSlice(target_shape);
  llvm::SmallVector<int64_t> tiles_shape = implicitShape(shape);
  tiles_shape[tiles_shape.size() - 2] = llvm::divideCeil(
      offsets_[0].value_or(0) + tiles_shape[tiles_shape.size() - 2],
      vreg_slice[0]);
  tiles_shape[tiles_shape.size() - 1] = llvm::divideCeil(
      offsets_[1].value_or(0) + tiles_shape[tiles_shape.size() - 1],
      vreg_slice[1]);
  return tiles_shape;
}

llvm::SmallVector<int64_t> VectorLayout::tileArrayShape(
    const ArrayRef<int64_t> shape,
    const std::array<int64_t, 2> target_shape) const {
  llvm::SmallVector<int64_t> tiles_shape =
      tileArrayImplicitShape(shape, target_shape);
  // Remove the implicit dimension --- it's always of size 1.
  switch (implicit_dim_) {
    case ImplicitDim::kNone:
      break;
    case ImplicitDim::kMinor:
      tiles_shape.pop_back();
      break;
    case ImplicitDim::kSecondMinor:
      tiles_shape.erase(tiles_shape.end() - 2);
      break;
  }
  return tiles_shape;
}

std::unique_ptr<VRegDataBounds> VectorLayout::tileDataBounds(
    MLIRContext* mlir_ctx, const ArrayRef<int64_t> full_shape,
    const ArrayRef<int64_t> idxs, const std::array<int64_t, 2> target_shape,
    const std::array<bool, 2> allow_replicated /*= {false, false}*/) const {
  // TODO(apaszke): allow_replicated could have been generalized to specify
  // what action should be taken when a REPLICATED offset is encountered.
  // Right now it either disallows replication, or selects the whole dimension.
  int64_t s, l;
  switch (implicit_dim_) {
    case ImplicitDim::kNone:
      s = idxs[idxs.size() - 2];
      l = idxs[idxs.size() - 1];
      break;
    case ImplicitDim::kMinor:
      s = idxs[idxs.size() - 1];
      l = 0;
      break;
    case ImplicitDim::kSecondMinor:
      s = 0;
      l = idxs[idxs.size() - 1];
      break;
  }

  const llvm::SmallVector<int64_t> tiles_implicit_shape =
      tileArrayImplicitShape(full_shape, target_shape);
  const int64_t ns = tiles_implicit_shape[tiles_implicit_shape.size() - 2];
  const int64_t nl = tiles_implicit_shape[tiles_implicit_shape.size() - 1];
  const llvm::SmallVector<int64_t> implicit_shape = implicitShape(full_shape);
  const int64_t is = implicit_shape[implicit_shape.size() - 2];
  const int64_t il = implicit_shape[implicit_shape.size() - 1];

  if (!hasNaturalTopology(target_shape)) {
    if (!offsets_[0].has_value() || !offsets_[1].has_value()) {
      emitError(UnknownLoc::get(mlir_ctx), "Not implemented");
      return nullptr;
    }
    const int64_t so = *offsets_[0];
    const int64_t lo = *offsets_[1];
    if (tiling_[0] == 1 && tiling_[1] % target_shape[1] == 0 &&
        implicit_dim_ == ImplicitDim::kSecondMinor) {
      const int64_t values_per_vreg =
          target_shape[0] * target_shape[1] * packing();
      const int64_t start_offset = l == 0 ? lo : 0;
      const int64_t end_offset =
          l == nl - 1 ? lo + il - l * values_per_vreg : values_per_vreg;
      return std::make_unique<SingleRowVRegBounds>(*this, start_offset,
                                                   end_offset, target_shape);
    }
    if (tiling_[1] != target_shape[1]) {
      emitError(UnknownLoc::get(mlir_ctx), "Not implemented");
      return nullptr;
    }
    const int64_t start_sublanes = s == 0 ? so : 0;
    const int64_t start_lanes = l == 0 ? lo : 0;
    const int64_t end_sublanes =
        s == ns - 1 ? (so + is - 1) % tiling_[0] + 1 : tiling_[0];
    const int64_t end_lanes =
        l == nl - 1 ? (lo + il - 1) % tiling_[1] + 1 : tiling_[1];
    const int64_t tiles_per_vreg = tilesPerVreg(target_shape);
    const int64_t minormost_tiles = llvm::divideCeil(lo + il, tiling_[1]);
    const int64_t num_tiles =
        l == nl - 1 && minormost_tiles % tiles_per_vreg != 0
            ? minormost_tiles % tiles_per_vreg
            : tiles_per_vreg;
    return std::make_unique<TiledRectangularVregBounds>(
        *this, num_tiles, std::array<int64_t, 2>{start_sublanes, start_lanes},
        std::array<int64_t, 2>{end_sublanes, end_lanes}, target_shape);
  }
  // TODO(apaszke): Remove this path in favor of TiledVRegBounds
  const std::array<int64_t, 2> shift = {offsets_[0].value_or(0),
                                        offsets_[1].value_or(0)};
  const int64_t sb = s == 0 ? shift[0] : 0;
  const int64_t lb = l == 0 ? shift[1] : 0;
  int64_t se = target_shape[0];
  int64_t le = target_shape[1];
  // First, deal with sublanes.
  if (!offsets_[0].has_value()) {
    if (!allow_replicated[0]) {
      emitError(UnknownLoc::get(mlir_ctx), "Unexpected replicated offset");
      return nullptr;
    }
    // Otherwise, do nothing. We take the full slice.
  } else if (s == ns - 1) {
    se = shift[0] + is - s * target_shape[0];
  }
  // Now, we deal with lanes.
  if (!offsets_[1].has_value()) {
    if (!allow_replicated[1]) {
      emitError(UnknownLoc::get(mlir_ctx), "Unexpected replicated offset");
      return nullptr;
    }
    // Otherwise, do nothing. We take the full slice.
  } else if (l == nl - 1) {
    le = shift[1] + il - l * target_shape[1];
  }
  CHECK_LT(sb, se);
  CHECK_LT(lb, le);
  return std::make_unique<RectangularVregBounds>(
      std::array<int64_t, 2>{sb, lb}, std::array<int64_t, 2>{se, le});
}

bool VectorLayout::generalizes(const VectorLayout& other,
                               ArrayRef<int64_t> shape,
                               const std::array<int64_t, 2> target_shape,
                               bool check_bitwidth) const {
  if (check_bitwidth && bitwidth_ != other.bitwidth_) {
    return false;
  }
  for (auto [s, o] : llvm::zip(offsets_, other.offsets_)) {
    if (s.has_value() && s != o) {
      return false;
    }
  }
  if (implicit_dim_ != other.implicit_dim_) {
    // Don't fail yet! implicit_dim might not matter for some shapes.
    if (shape.data() == nullptr) {
      return false;
    }
    // If the second-minor dimension is of size 1, then it does not matter
    // whether we have a second minor implicit dim or not.
    bool ok = false;
    if (((implicit_dim_ == ImplicitDim::kSecondMinor &&
          other.implicit_dim_ == ImplicitDim::kNone) ||
         (other.implicit_dim_ == ImplicitDim::kSecondMinor &&
          implicit_dim_ == ImplicitDim::kNone)) &&
        shape[shape.size() - 2] == 1) {
      ok =  true;
    }
    // If sufficiently many trailing dimensions are of size 1, then it does not
    // matter if we use implicit dims to insert more.
    int max_rank = std::max(layout_rank(), other.layout_rank());
    CHECK_GE(max_rank, 1);
    CHECK_LE(max_rank, 2);
    if (*(shape.end() - 1) == 1 && (max_rank == 1 || *(shape.end() - 2) == 1)) {
      ok = true;
    }
    if (!ok) {
      return false;
    }
  }
  if (tiling_ != other.tiling_) {
    // Don't fail yet!
    // If there is only one tile in both tilings, then they are equivalent.
    if (shape.data() == nullptr) {
      return false;
    }
    const SmallVector<int64_t> ishape = implicitShape(shape);
    if (!(tiling_[1] == other.tiling_[1] && tiling_[1] == target_shape[1] &&
          offsets_[1].value_or(0) + ishape[ishape.size() - 1] <=
              target_shape[1] &&
          offsets_[0].value_or(0) + ishape[ishape.size() - 2] <=
              std::min(tiling_[0], other.tiling_[0]))) {
      return false;
    }
  }
  return true;
}

template <typename Stream>
void VectorLayout::print(Stream& os) const {
  os << static_cast<int32_t>(bitwidth_) << ",{";
  bool first = true;
  for (auto o : offsets_) {
    if (first) {
      first = false;
    } else {
      os << ',';
    }
    if (!o) {
      os << '*';
    } else {
      os << *o;
    }
  }
  os << "},(" << tiling_[0] << ',' << tiling_[1] << ")";
  if (implicit_dim_ == ImplicitDim::kMinor) {
    os << ",-1";
  } else if (implicit_dim_ == ImplicitDim::kSecondMinor) {
    os << ",-2";
  }
}

std::optional<VectorLayout> VectorLayout::join(const VectorLayout& l,
                                               const VectorLayout& r,
                                               ArrayRef<int64_t> shape,
                                               bool check_bitwidth) {
  if (check_bitwidth &&
      (l.bitwidth_ != r.bitwidth_ || l.tiling_ != r.tiling_)) {
    return std::nullopt;
  }
  if (l.implicit_dim_ != r.implicit_dim_) {
    if (shape.size() < 2) {
      return std::nullopt;
    }
    ImplicitDim dim;
    if (l.implicit_dim_ == ImplicitDim::kNone) {
      dim = r.implicit_dim_;
    } else if (r.implicit_dim_ == ImplicitDim::kNone) {
      dim = l.implicit_dim_;
    } else {
      return std::nullopt;
    }
    if (dim == ImplicitDim::kMinor && shape[shape.size() - 1] == 1) {
      // OK, they are equivalent.
    } else if (dim == ImplicitDim::kSecondMinor &&
               shape[shape.size() - 2] == 1) {
      // OK, they are equivalent.
    } else {
      return std::nullopt;
    }
  }
  LayoutOffsets offsets;
  for (int i = 0; i < 2; ++i) {
    auto lo = l.offsets()[i];
    auto ro = r.offsets()[i];
    if (lo && ro && lo != ro) {
      return std::nullopt;
    }
    offsets[i] = lo.has_value() ? lo : ro;
  }
  return VectorLayout(l.bitwidth_, offsets, l.tiling_, l.implicit_dim_);
}

std::optional<VectorLayout> VectorLayout::parse(llvm::StringRef* data) {
  llvm::StringRef local(*data);
  int8_t bitwidth;
  LayoutOffsets offsets;
  std::array<int64_t, 2> tiling;
  ImplicitDim implicit_dim = ImplicitDim::kNone;
  if (local.consumeInteger(10, bitwidth) || !local.consume_front(",{") ||
      parseOffset(&local, &offsets[0]) || !local.consume_front(",") ||
      parseOffset(&local, &offsets[1]) || !local.consume_front("},(") ||
      local.consumeInteger(10, tiling[0]) || !local.consume_front(",") ||
      local.consumeInteger(10, tiling[1]) || !local.consume_front(")")) {
    return std::nullopt;
  }
  if (local.consume_front(",-1")) {
    implicit_dim = ImplicitDim::kMinor;
  } else if (local.consume_front(",-2")) {
    implicit_dim = ImplicitDim::kSecondMinor;
  }
  *data = local;
  return VectorLayout(bitwidth, offsets, tiling, implicit_dim);
}

namespace {
template <class>
inline constexpr bool false_v = false;

template <typename Stream>
Stream& printLayout(Stream& os, const Layout& v) {
  os << '"';
  if (v.has_value()) {
    v->print(os);
  } else {
    os << "none";
  }
  os << '"';
  return os;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Layout& v) {
  return printLayout<std::ostream>(os, v);
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Layout& v) {
  return printLayout<llvm::raw_ostream>(os, v);
}

mlir::Diagnostic& operator<<(mlir::Diagnostic& diag, const Layout& v) {
  return printLayout<mlir::Diagnostic>(diag, v);
}

llvm::hash_code hash_value(const VectorLayout& layout) {
  return llvm::hash_value(layout.as_tuple());
}

std::ostream &operator<<(std::ostream &os, VectorLayout::ImplicitDim dim) {
  switch (dim) {
    case VectorLayout::ImplicitDim::kNone:
      os << "none";
      break;
    case VectorLayout::ImplicitDim::kMinor:
      os << "-1";
      break;
    case VectorLayout::ImplicitDim::kSecondMinor:
      os << "-2";
      break;
  }
  return os;
}

std::optional<Layout> parseLayout(mlir::AsmParser& parser) {
  std::string layout_str;
  if (failed(parser.parseString(&layout_str))) {
    return std::nullopt;
  }
  if (layout_str == "none") {
    return kNoLayout;
  }
  llvm::StringRef ref(layout_str);
  if (auto layout = VectorLayout::parse(&ref); ref.empty()) {
    return *layout;
  }
  return std::nullopt;
}

const Layout kNoLayout = std::nullopt;

}  // namespace mlir::tpu
