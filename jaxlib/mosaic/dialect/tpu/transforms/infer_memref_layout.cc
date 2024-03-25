#include "jaxlib/mosaic/dialect/tpu/transforms/infer_memref_layout.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"
#include "xla/layout.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_INFERMEMREFLAYOUTPASS
#define GEN_PASS_DEF_INFERMEMREFLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

// Returns the number of 128-element groups in a tile.
//
// Arguments:
//   num_128s: A number of 128-element groups in the full operand.
//   hardware_generation: An integer indicating the target TPU generation.
//   bitwidth: The bitwidth of the element type of the operand.
int getTilingFactor(const int num_128s, const int hardware_generation,
                    const int8_t bitwidth) {
  CHECK(llvm::isPowerOf2_32(bitwidth));
  CHECK_LE(4, bitwidth);
  CHECK_LE(bitwidth, 32);
  const int packing = 32 / bitwidth;
  const int min_tiling = (1 + (hardware_generation < 4)) * packing;
  const int max_tiling = 8;
  int tiling = min_tiling;
  while (tiling < std::min(num_128s, max_tiling)) {
    tiling *= 2;
  }
  return tiling;
}

FailureOr<TiledLayoutAttr> inferLayout(MemRefType memref_ty,
                                       const int hardware_generation) {
  if (auto tiled_layout_attr =
          dyn_cast<TiledLayoutAttr>(memref_ty.getLayout())) {
    return tiled_layout_attr;
  }
  if (auto affine_map_attr = dyn_cast<AffineMapAttr>(memref_ty.getLayout())) {
    if (memref_ty.getRank() == 0) {
      return emitError(UnknownLoc::get(memref_ty.getContext()),
                       "0-rank memref not supported");
    }
    if (!affine_map_attr.isIdentity()) {
      return emitError(UnknownLoc::get(memref_ty.getContext()),
                       "Non-identity affine layout");
    }
    if (!memref_ty.getElementType().isIntOrFloat()) {
      return emitError(UnknownLoc::get(memref_ty.getContext()),
                       "Invalid element type for memref");
    }
    const int8_t bitwidth = memref_ty.getElementTypeBitWidth();
    // Infer the layout
    if (memref_ty.getRank() == 1) {
      const int64_t leading_tile =
          getTilingFactor(llvm::divideCeil(memref_ty.getShape().back(), 128),
                          hardware_generation, bitwidth) *
          128;
      SmallVector<xla::Tile> tiles{xla::Tile({leading_tile})};
      if (bitwidth != 32) {
        if (!llvm::has_single_bit<unsigned>(bitwidth) || bitwidth > 32) {
          return emitError(UnknownLoc::get(memref_ty.getContext()),
                           "Unsupported bitwidth: ")
                 << bitwidth;
        }
        tiles.append({xla::Tile({128}), xla::Tile({32 / bitwidth, 1})});
      }
      return TiledLayoutAttr::get(memref_ty.getContext(), tiles, {1});
    }
    // memref.getRank() > 1
    const ArrayRef<int64_t> shape = memref_ty.getShape();
    const int64_t second_minor = shape[shape.size() - 2];
    const int64_t leading_tile_rows =
        getTilingFactor(second_minor, hardware_generation, bitwidth);
    SmallVector<xla::Tile> tiles{xla::Tile({leading_tile_rows, 128})};
    if (bitwidth != 32) {
      if (!llvm::has_single_bit<unsigned>(bitwidth) || bitwidth > 32) {
        return emitError(UnknownLoc::get(memref_ty.getContext()),
                         "Unsupported bitwidth: ")
               << bitwidth;
      }
      tiles.push_back(xla::Tile({32 / bitwidth, 1}));
    }
    SmallVector<int64_t> tile_strides(memref_ty.getRank());
    int64_t stride = 1;
    for (int i = memref_ty.getRank() - 1; i >= 0; --i) {
      tile_strides[i] = stride;
      if (i == memref_ty.getRank() - 1) {
        stride *= (memref_ty.getShape()[i] + 127) / 128;
      } else if (i == memref_ty.getRank() - 2) {
        stride *= (memref_ty.getShape()[i] + leading_tile_rows - 1) /
                  leading_tile_rows;
      } else {
        stride *= memref_ty.getShape()[i];
      }
    }
    return TiledLayoutAttr::get(memref_ty.getContext(), tiles, tile_strides);
  }
  return emitError(UnknownLoc::get(memref_ty.getContext()),
                   "Unrecognized layout annotation");
}

// Make sure only the first tile might introduce padding.
LogicalResult checkTiles(MLIRContext *mlir_ctx,
                         const ArrayRef<xla::Tile> &tiles) {
  SmallVector<int64_t> tiled_dims(tiles.front().dimensions().begin(),
                                  tiles.front().dimensions().end());
  for (const xla::Tile &t : tiles.drop_front()) {
    const int64_t offset = tiled_dims.size() - t.dimensions().size();
    if (offset < 0) {
      return emitError(UnknownLoc::get(mlir_ctx),
                       "Not implemented: layout too complicated");
    }
    for (int i = 0; i < t.dimensions().size(); ++i) {
      auto [d, m] = std::div(tiled_dims[offset + i], t.dimension(i));
      if (m != 0) {
        return emitError(UnknownLoc::get(mlir_ctx),
                         "Not implemented: layout too complicated");
      }
      tiled_dims[offset + i] = d;
    }
    tiled_dims.append(t.dimensions().begin(), t.dimensions().end());
  }
  return success();
}

FailureOr<MemRefType> inferMemref(MemRefType memref,
                                  const int hardware_generation) {
  const Attribute semaphore_mem = tpu::MemorySpaceAttr::get(
      memref.getContext(), MemorySpace::kSemaphoreMem);
  const Attribute smem =
      tpu::MemorySpaceAttr::get(memref.getContext(), MemorySpace::smem);
  const Attribute vmem =
      tpu::MemorySpaceAttr::get(memref.getContext(), MemorySpace::vmem);
  const Attribute memory_space =
      memref.getMemorySpace() == nullptr ? vmem : memref.getMemorySpace();
  if (memory_space == semaphore_mem || memory_space == smem) {
    SmallVector<int64_t> tile_strides;
    tile_strides.reserve(memref.getRank());
    int64_t stride = 1;
    for (int i = memref.getRank() - 1; i >= 0; --i) {
      tile_strides.push_back(stride);
      stride *= memref.getDimSize(i);
    }
    std::reverse(tile_strides.begin(), tile_strides.end());
    auto layout = TiledLayoutAttr::get(memref.getContext(), {}, tile_strides);
    return MemRefType::get(memref.getShape(), memref.getElementType(), layout,
                           memory_space);
  }
  FAILUREOR_ASSIGN_OR_RETURN(const TiledLayoutAttr layout,
                             inferLayout(memref, hardware_generation));

  const ArrayRef<xla::Tile> tiles = layout.getTiles();
  if (failed(checkTiles(memref.getContext(), tiles))) {
    return failure();
  }
  const xla::Tile &first_tile = tiles.front();
  const int64_t untiled_dims =
      memref.getShape().size() - first_tile.dimensions().size();
  if (untiled_dims < 0) {
    return emitError(UnknownLoc::get(memref.getContext()), "Invalid tiling");
  }
  SmallVector<int64_t> new_shape(memref.getShape());
  for (int i = 0; i < first_tile.dimensions().size(); ++i) {
    new_shape[untiled_dims + i] =
        llvm::alignTo(new_shape[untiled_dims + i], first_tile.dimension(i));
  }
  return MemRefType::get(new_shape, memref.getElementType(), layout,
                         memory_space);
}

LogicalResult inferOp(Operation &op, const int hardware_generation) {
  if (auto alloca_op = dyn_cast<memref::AllocaOp>(op)) {
    TypedValue<MemRefType> arg = alloca_op.getResult();
    const MemRefType memref_ty = alloca_op.getResult().getType();
    FAILUREOR_ASSIGN_OR_RETURN(const MemRefType new_memref_ty,
                               inferMemref(memref_ty, hardware_generation));
    alloca_op.getResult().setType(new_memref_ty);
    if (memref_ty != new_memref_ty) {
      OpBuilder builder(alloca_op->getContext());
      builder.setInsertionPointAfter(alloca_op);
      auto erase_op = builder.create<tpu::EraseLayoutOp>(
          arg.getLoc(),
          MemRefType::get(new_memref_ty.getShape(), memref_ty.getElementType(),
                          /*layout=*/nullptr, new_memref_ty.getMemorySpace()),
          arg);
      arg.replaceAllUsesExcept(erase_op.getResult(), erase_op);
    }
  } else if (auto alloca_op = dyn_cast<tpu::AllocaSemaphoreOp>(op)) {
    TypedValue<MemRefType> arg = alloca_op.getResult();
    const MemRefType memref_ty = alloca_op.getResult().getType();
    FAILUREOR_ASSIGN_OR_RETURN(const MemRefType new_memref_ty,
                               inferMemref(memref_ty, hardware_generation));
    alloca_op.getResult().setType(new_memref_ty);
    if (memref_ty != new_memref_ty) {
      OpBuilder builder(alloca_op->getContext());
      builder.setInsertionPointAfter(alloca_op);
      auto erase_op = builder.create<tpu::EraseLayoutOp>(
          arg.getLoc(),
          MemRefType::get(new_memref_ty.getShape(), memref_ty.getElementType(),
                          /*layout=*/nullptr, new_memref_ty.getMemorySpace()),
          arg);
      arg.replaceAllUsesExcept(erase_op.getResult(), erase_op);
    }
  }
  for (Region &region : op.getRegions()) {
    for (Block &block : region) {
      for (Operation& op : block) {
        if (failed(inferOp(op, hardware_generation))) {
          return failure();
        }
      }
    }
  }
  return success();
}

LogicalResult inferFunc(func::FuncOp f, const int hardware_generation) {
  if (!f.getBody().hasOneBlock()) {
    return f.emitOpError("Functions should only have a single block");
  }
  Block &entry = f.getBody().front();
  SmallVector<Type> new_arg_types;
  auto builder = OpBuilder::atBlockBegin(&entry);
  for (BlockArgument arg : entry.getArguments()) {
    const auto memref_ty = dyn_cast<MemRefType>(arg.getType());
    if (memref_ty == nullptr) {
      new_arg_types.push_back(arg.getType());
      continue;
    }
    FAILUREOR_ASSIGN_OR_RETURN(const MemRefType new_memref_ty,
                               inferMemref(memref_ty, hardware_generation));
    arg.setType(new_memref_ty);
    new_arg_types.push_back(arg.getType());
    if (memref_ty != new_memref_ty) {
      // Some standard MLIR ops have static checks that seems unreasonable,
      // and we know they hold in the way they are used in Mosaic. Still,
      // verification with layouts likes to fail, because it can't statically
      // prove the properties.
      auto erase_op = builder.create<tpu::EraseLayoutOp>(
          arg.getLoc(),
          MemRefType::get(new_memref_ty.getShape(), memref_ty.getElementType(),
                          /*layout=*/nullptr, new_memref_ty.getMemorySpace()),
          arg);
      arg.replaceAllUsesExcept(erase_op.getResult(), erase_op);
    }
  }
  f.setFunctionType(
      builder.getAttr<FunctionType>(new_arg_types, f.getResultTypes()));
  for (Operation &op : entry.getOperations()) {
    if (failed(inferOp(op, hardware_generation))) {
      return failure();
    }
  }
  return success();
}

// Infers the layout and memory space attributes of function memref arguments.
//
// In the future we should require those annotations from Mosaic users, but it's
// best to keep them internal for as long as they are under development.
//
// Arguments:
//   module: The MLIR module on which to perform the inference.
//   hardware_generation: The TPU hardware generation to target.
LogicalResult inferModule(ModuleOp module, const int hardware_generation) {
  // TODO(apaszke): Do layout assignment for scoped allocations too.
  for (Operation &op : *module.getBody()) {
    auto f = dyn_cast<func::FuncOp>(op);
    if (f == nullptr) {
      return module.emitOpError("Expected only FuncOps but found ") << op;
    }
    if (failed(inferFunc(f, hardware_generation))) {
      return failure();
    }
  }
  return success();
}

struct InferMemRefLayoutPass
    : public impl::InferMemRefLayoutPassBase<InferMemRefLayoutPass> {
  InferMemRefLayoutPass(int hardware_generation_) {
    hardware_generation = hardware_generation_;
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      signalPassFailure();
      return;
    }
    func::FuncOp func = getOperation();
    if (failed(inferFunc(func, hardware_generation))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createInferMemRefLayoutPass(
    int hardware_generation) {
  return std::make_unique<InferMemRefLayoutPass>(hardware_generation);
}

}  // namespace mlir::tpu
