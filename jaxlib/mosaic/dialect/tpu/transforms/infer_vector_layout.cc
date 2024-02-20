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

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/layout.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_INFERVECTORLAYOUTPASS
#define GEN_PASS_DEF_INFERVECTORLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {

using ImplicitDim = VectorLayout::ImplicitDim;

static constexpr int kLayoutLog = 10;

class Print {
 public:
  explicit Print(Operation *t) : payload_(t) {}
  Operation *payload_;

 private:
  friend std::ostream &operator<<(std::ostream &, Print);
};

std::ostream &operator<<(std::ostream &os, Print p) {
  std::string s;
  llvm::raw_string_ostream tmp_os(s);
  p.payload_->print(tmp_os);
  os << tmp_os.str();
  return os;
}

bool is_fully_replicated(const Layout &layout) {
  static LayoutOffsets replicated_offsets = {std::nullopt, std::nullopt};
  return layout.has_value() && layout->offsets() == replicated_offsets;
}

TiledLayoutAttr getMemRefLayout(Value ref) {
  if (auto erase_op = ref.getDefiningOp<tpu::EraseLayoutOp>()) {
    ref = erase_op.getOperand();
  }
  return cast<TiledLayoutAttr>(cast<MemRefType>(ref.getType()).getLayout());
}

LogicalResult verifyDivisibleIndex(Value tiled_index, int64_t tiling, int dim,
                                   Operation *op) {
  if (!isGuaranteedDivisible(tiled_index, tiling)) {
    return op->emitOpError("cannot statically prove that index in dimension ")
           << dim << " is a multiple of " << tiling;
  }
  return success();
}

// TODO(apaszke): Test that this pass fills in NoLayout for all operations that
// have corresponding native instructions.
class VectorLayoutInferer {
 public:
  explicit VectorLayoutInferer(std::array<int64_t, 2> target_shape)
      : target_shape_({target_shape[0], target_shape[1]}),
        default_tiling_(target_shape) {}

#define TPU_CHECK_OP(cond, msg) \
  if (!(cond)) {                \
    op->emitOpError(msg);       \
    return failure();           \
  }

#define NYI(msg)                            \
  op->emitOpError("not implemented: " msg); \
  return failure();

  LogicalResult inferBlock(
      Block &block,
      const std::function<LogicalResult(Operation *)> &match_terminator) {
    for (Operation &any_op : block.without_terminator()) {
      VLOG(kLayoutLog) << Print(&any_op);
      if (any_op.hasAttr("in_layout") || any_op.hasAttr("out_layout")) {
        if (auto op = dyn_cast<tpu::AssumeLayoutOp>(any_op)) {
          TPU_CHECK_OP(
              any_op.hasAttr("in_layout") && any_op.hasAttr("out_layout"),
              "expect layout attributes in tpu::AssumeLayoutOp");
          continue;
        } else {
          any_op.emitOpError("layout attributes already attached");
          return failure();
        }
      }
      bool has_vector_io = false;
      for (auto op : any_op.getOperands()) {
        has_vector_io |= op.getType().isa<VectorType>();
      }
      for (auto r : any_op.getResults()) {
        has_vector_io |= r.getType().isa<VectorType>();
      }
      if (!has_vector_io && any_op.getRegions().empty()) {
        llvm::SmallVector<Layout, 4> in_layout(any_op.getNumOperands(),
                                               kNoLayout);
        if (any_op.getNumResults() == 0) {
          setInLayout(&any_op, in_layout);
        } else if (any_op.getNumResults() == 1) {
          setLayout(&any_op, in_layout, kNoLayout);
        } else {
          any_op.emitOpError("Multi-result ops not supported");
          return failure();
        }
      } else if (auto op = dyn_cast<arith::ExtFOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::TruncFOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::ExtSIOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::TruncIOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::SelectOp>(any_op)) {
        auto true_ty = dyn_cast<VectorType>(op.getTrueValue().getType());
        auto false_ty = dyn_cast<VectorType>(op.getFalseValue().getType());
        TPU_CHECK_OP(static_cast<bool>(true_ty) == static_cast<bool>(false_ty),
                     "Only one side of arith is a vector?");
        if (true_ty) {
          TPU_CHECK_OP(true_ty.getElementTypeBitWidth() == kNativeBitwidth &&
                           false_ty.getElementTypeBitWidth() == kNativeBitwidth,
                       "Only 32-bit select supported");
        }
        if (inferElementwise(&any_op, /*check_bitwidth=*/false).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::ExtUIOp>(any_op)) {
        auto in_ty = dyn_cast<VectorType>(op.getIn().getType());
        auto out_ty = dyn_cast<VectorType>(op.getType());
        TPU_CHECK_OP(static_cast<bool>(in_ty) == static_cast<bool>(out_ty),
                     "Input and output are not both vectors?");
        if (in_ty) {
          TPU_CHECK_OP(in_ty.getElementTypeBitWidth() == 1 &&
                           out_ty.getElementTypeBitWidth() == 32,
                       "Only 1 bit -> 32 bit extensison supported");
        }
        if (inferElementwise(&any_op, /*check_bitwidth=*/false).failed()) {
          return failure();
        }
      } else if (isa<arith::CmpIOp>(any_op) || isa<arith::CmpFOp>(any_op)) {
        auto lhs_ty = dyn_cast<VectorType>(any_op.getOperand(0).getType());
        auto rhs_ty = dyn_cast<VectorType>(any_op.getOperand(1).getType());
        TPU_CHECK_OP(static_cast<bool>(lhs_ty) == static_cast<bool>(rhs_ty),
                     "Only one side of cmp is a vector?");
        if (lhs_ty) {
          TPU_CHECK_OP(lhs_ty.getElementTypeBitWidth() == kNativeBitwidth &&
                           rhs_ty.getElementTypeBitWidth() == kNativeBitwidth,
                       "Only 32-bit cmp supported");
        }
        if (inferElementwise(&any_op, /*check_bitwidth=*/false).failed()) {
          return failure();
        }
      } else if (OpTrait::hasElementwiseMappableTraits(&any_op)) {
        if (inferElementwise(&any_op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<arith::ConstantOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<cf::AssertOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<memref::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<scf::IfOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<scf::ForOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
        } else if (auto op = dyn_cast<tpu::RotateOp>(any_op)) {
          if (infer(op).failed()) {
            return failure();
          }
      } else if (auto op = dyn_cast<tpu::ConcatenateOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::MatmulOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::StoreOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::EraseLayoutOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::IotaOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::GatherOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::BitcastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::RepeatOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::TraceOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<tpu::RegionOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::BroadcastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ContractionOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ExtractOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::LoadOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::MultiDimReductionOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::ShapeCastOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::StoreOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op = dyn_cast<vector::TransposeOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else if (auto op =
                     llvm::dyn_cast<vector::ExtractStridedSliceOp>(any_op)) {
        if (infer(op).failed()) {
          return failure();
        }
      } else {
        any_op.emitOpError("unsupported in vector layout inference");
        return failure();
      }
      CHECK(any_op.getNumResults() == 0 || any_op.hasAttr("out_layout"));
      CHECK(any_op.getNumOperands() == 0 || any_op.hasAttr("in_layout"));
    }
    return match_terminator(block.getTerminator());
  }

  LogicalResult infer(arith::ConstantOp op) {
    if (op.getType().isSignlessIntOrIndexOrFloat()) {
      setOutLayout(op, kNoLayout);
      return success();
    }
    if (auto ty = dyn_cast<VectorType>(op.getType())) {
      auto elems = dyn_cast<DenseElementsAttr>(op.getValue());
      TPU_CHECK_OP(ty.getElementType().isSignlessIntOrIndexOrFloat(),
                   "expected scalar element type in vector");
      TPU_CHECK_OP(ty.getRank() > 0, "rank 0 vectors unsupported");
      TPU_CHECK_OP(elems, "expected vector constants to use DenseElementsAttr");
      auto bitwidth = ty.getElementTypeBitWidth();
      if (elems.isSplat()) {
        if (ty.getRank() == 1) {
          // Here, we choose to lay out along lanes arbitrarily. It would be
          // equally valid to go with sublanes. Still, this value is so easy
          // to relayout that it shouldn't really make a difference.
          setOutLayout(op, VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                                        nativeTiling(bitwidth),
                                        ImplicitDim::kSecondMinor));
        } else {  // ty.getRank() >= 2
          setOutLayout(
              op, VectorLayout(bitwidth, {std::nullopt, std::nullopt},
                               nativeTiling(bitwidth), ImplicitDim::kNone));
        }
      } else {
        TPU_CHECK_OP(ty.getElementTypeBitWidth() == kNativeBitwidth,
                     "Only 32-bit non-splat constants supported");
        if (ty.getRank() == 1) {
          if (ty.getDimSize(0) <= target_shape_[0]) {
            // Use 2D layout with replication.
            NYI("small 1D constants");
          } else {  // NOLINT(readability-else-after-return)
            NYI("large 1D constants");
          }
        } else {  // ty.getRank() >= 2
          setOutLayout(op, VectorLayout(kNativeBitwidth, {0, 0},
                                        default_tiling_, ImplicitDim::kNone));
        }
      }
      return success();
    }
    op.emitOpError("unsupported constant type");
    return failure();
  }

  LogicalResult infer(arith::ExtFOp op) {
    auto src_ty = dyn_cast<VectorType>(op.getIn().getType());
    if (!src_ty) {
      setLayout(op, kNoLayout, kNoLayout);
      return success();
    }
    auto dst_ty = cast<VectorType>(op.getOut().getType());
    auto some_layout = getLayout(op.getIn());
    TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == 16 &&
                     dst_ty.getElementTypeBitWidth() == 32,
                 "Only 16-bit to 32-bit extensions supported");
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    auto &layout = *some_layout;
    if (layout.implicit_dim() == ImplicitDim::kNone) {
      Layout src_layout;
      Layout dst_layout;
      // All layouts that subdivide the rows of the default tiling evenly
      // can be handled uniformly with the default case, by preserving the
      // tiling through the op.
      // TODO(apaszke): Support (16,128) too.
      if (default_tiling_[0] % layout.tiling()[0] == 0 &&
          default_tiling_[1] == layout.tiling()[1]) {
        src_layout = layout;
      } else {
        src_layout = VectorLayout(16, layout.offsets(), default_tiling_,
                                  ImplicitDim::kNone);
      }
      dst_layout = VectorLayout(32, layout.offsets(), src_layout->tiling(),
                                ImplicitDim::kNone);
      setLayout(op, src_layout, dst_layout);
      return success();
    }
    if (layout.implicit_dim() == ImplicitDim::kSecondMinor) {
      TPU_CHECK_OP(layout.tiling() == nativeTiling(16), "unsupported tiling");
      auto dst_layout = VectorLayout(32, layout.offsets(), default_tiling_,
                                     layout.implicit_dim());
      setLayout(op, some_layout, dst_layout);
      return success();
    }
    op.emitOpError("unsupported extension layout");
    return failure();
  }

  LogicalResult infer(arith::TruncFOp op) {
    auto src_ty = dyn_cast<VectorType>(op.getIn().getType());
    if (!src_ty) {
      setLayout(op, kNoLayout, kNoLayout);
      return success();
    }
    auto dst_ty = cast<VectorType>(op.getOut().getType());
    auto some_layout = getLayout(op.getIn());
    TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == 32 &&
                     dst_ty.getElementTypeBitWidth() == 16,
                 "Only 32-bit to 16-bit truncation supported");
    auto &layout = *some_layout;
    if (layout.implicit_dim() == ImplicitDim::kNone) {
      bool select_native = allUsersRequireNativeTiling(op.getResult());
      auto src_layout = VectorLayout(32, layout.offsets(), default_tiling_,
                                     ImplicitDim::kNone);
      auto dst_layout =
          VectorLayout(16, layout.offsets(),
                       select_native ? nativeTiling(16) : default_tiling_,
                       ImplicitDim::kNone);
      setLayout(op, src_layout, dst_layout);
      return success();
    }
    op.emitOpError("unsupported truncation layout");
    return failure();
  }

  LogicalResult infer(arith::ExtSIOp op) {
    auto src_ty = dyn_cast<VectorType>(op.getIn().getType());
    if (!src_ty) {
      setLayout(op, kNoLayout, kNoLayout);
      return success();
    }
    auto dst_ty = cast<VectorType>(op.getOut().getType());
    auto some_layout = getLayout(op.getIn());
    TPU_CHECK_OP(dst_ty.getElementTypeBitWidth() == 32,
                 "Only extensions to 32-bit supported");
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    auto &layout = *some_layout;
    if (layout.implicit_dim() == ImplicitDim::kNone) {
      // TODO(apaszke): Support native layouts here.
      Layout src_layout;
      Layout dst_layout;
      // All layouts that subdivide the rows of the default tiling evenly
      // can be handled uniformly with the default case, by preserving the
      // tiling through the op.
      if (default_tiling_[0] % layout.tiling()[0] == 0 &&
          default_tiling_[1] == layout.tiling()[1]) {
        src_layout = layout;
      } else {
        src_layout = VectorLayout(layout.bitwidth(), layout.offsets(),
                                  default_tiling_, ImplicitDim::kNone);
      }
      dst_layout = VectorLayout(32, layout.offsets(), src_layout->tiling(),
                                ImplicitDim::kNone);
      setLayout(op, src_layout, dst_layout);
      return success();
    }
    if (layout.implicit_dim() == ImplicitDim::kSecondMinor) {
      TPU_CHECK_OP(layout.tiling() == nativeTiling(16), "unsupported tiling");
      auto dst_layout = VectorLayout(32, layout.offsets(), default_tiling_,
                                     layout.implicit_dim());
      setLayout(op, some_layout, dst_layout);
      return success();
    }
    op.emitOpError("unsupported extension layout");
    return failure();
  }

  LogicalResult infer(arith::TruncIOp op) {
    auto src_ty = dyn_cast<VectorType>(op.getIn().getType());
    if (!src_ty) {
      setLayout(op, kNoLayout, kNoLayout);
      return success();
    }
    auto dst_ty = cast<VectorType>(op.getOut().getType());
    auto some_layout = getLayout(op.getIn());
    TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == 32,
                 "Only 32-bit truncation supported");
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    auto &layout = *some_layout;
    if (layout.implicit_dim() == ImplicitDim::kNone) {
      auto src_layout = VectorLayout(32, layout.offsets(), default_tiling_,
                                     ImplicitDim::kNone);
      bool select_native = allUsersRequireNativeTiling(op.getResult());
      auto dst_layout = VectorLayout(
          dst_ty.getElementTypeBitWidth(), layout.offsets(),
          select_native ? nativeTiling(dst_ty.getElementTypeBitWidth())
                        : default_tiling_,
          ImplicitDim::kNone);
      setLayout(op, src_layout, dst_layout);
      return success();
    }
    op.emitOpError("unsupported truncation layout");
    return failure();
  }

  LogicalResult infer(cf::AssertOp op) {
    setInLayout(op, {kNoLayout});
    return success();
  }

  LogicalResult infer(func::FuncOp op) {
    if (!op.getBody().hasOneBlock()) {
      op.emitOpError("Only one block functions supported");
      return failure();
    }
    return inferBlock(
        op.getBody().front(), [this](Operation *op) -> LogicalResult {
          TPU_CHECK_OP(isa<func::ReturnOp>(op),
                       "Expected func.return terminator");
          for (Value o : op->getOperands()) {
            TPU_CHECK_OP(!isa<VectorType>(o.getType()),
                         "vector returns unsupported");
          }
          SmallVector<Layout, 4> in_layout(op->getNumOperands(), {kNoLayout});
          setInLayout(op, in_layout);
          return success();
        });
  }

  LogicalResult infer(memref::LoadOp op) {
    TPU_CHECK_OP(op.getType().isSignlessIntOrIndexOrFloat(),
                 "memref.load with non-scalar result");
    SmallVector<Layout, 5> in_layout(op.getNumOperands(), {kNoLayout});
    setLayout(op, in_layout, kNoLayout);
    return success();
  }

  LogicalResult infer(scf::IfOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 1, "expected one operand");
    setInLayout(op, {kNoLayout});
    if (inferBlock(*op.thenBlock(), match_yield).failed()) {
      op.emitOpError("failed to infer layout for then branch");
      return failure();
    }
    auto then_yield = op.thenBlock()->getTerminator();
    TPU_CHECK_OP(then_yield->getOperandTypes() == op->getResultTypes(),
                 "scf if results and then branch yield operands do not match");
    llvm::SmallVector<Layout, 4> result_layout;
    result_layout.reserve(then_yield->getNumOperands());
    for (const auto &operand : then_yield->getOperands()) {
      if (operand.getType().isSignlessIntOrIndexOrFloat()) {
        result_layout.push_back(kNoLayout);
      } else if (isa<VectorType>(operand.getType())) {
        result_layout.push_back(getLayout(operand));
      } else {
        op.emitOpError("unsupported scf.yield type");
        return failure();
      }
    }

    if (auto else_block = op.elseBlock()) {
      if (inferBlock(*else_block, match_yield).failed()) {
        op.emitOpError("failed to infer layout for else branch");
        return failure();
      }
    }
    if (op->getNumResults() == 0) {
      return success();
    }
    // If the if op has results, it should have both then and else regions with
    // yield op.
    auto else_yield = op.elseBlock()->getTerminator();
    TPU_CHECK_OP(else_yield->getOperandTypes() == op->getResultTypes(),
                 "scf if results and else branch yield operands do not match");

    // Check each layout of the yield in else branch and override the
    // result_layout if else branch's yield layout is less general. For example,
    // if we yield offset (*, *) in then branch and offset (*, 0) in else
    // branch, the result offset should be (*, 0).
    for (int i = 0; i < else_yield->getNumOperands(); ++i) {
      const auto &operand = else_yield->getOperand(i);
      if (!isa<VectorType>(operand.getType())) {
        continue;
      }
      auto shape = dyn_cast<VectorType>(operand.getType()).getShape();
      auto layout = getLayout(operand);
      CHECK(result_layout[i].has_value() && layout.has_value());
      result_layout[i] =
          VectorLayout::join(result_layout[i].value(), layout.value(), shape);
      if (!result_layout[i].has_value()) {
        op.emitOpError(
            "failed to find a compatible layout in then and else branch for "
            "output ")
            << i;
        return failure();
      }
    }
    setInLayout(then_yield, result_layout);
    setInLayout(else_yield, result_layout);
    setOutLayout(op, result_layout);
    return success();
  }

  LogicalResult infer(scf::ForOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<scf::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op.getRegion().hasOneBlock(),
                 "expected one block for scf.for");
    TPU_CHECK_OP(
        op.getNumRegionIterArgs() == op.getNumResults(),
        "expected num_region_iter_args is equal to num_results in scf.for");
    TPU_CHECK_OP(
        op->getNumOperands() == 3 + op.getNumResults(),
        "expected num_operands is equal to 3 + num_results in scf.for");

    llvm::SmallVector<Layout, 4> in_layouts;
    in_layouts.reserve(op->getNumOperands());
    in_layouts.push_back(kNoLayout);  // Lower bound.
    in_layouts.push_back(kNoLayout);  // Upper bound.
    in_layouts.push_back(kNoLayout);  // Step.
    for (const auto &arg : op.getInitArgs()) {
      if (arg.getType().isSignlessIntOrIndexOrFloat()) {
        in_layouts.push_back(kNoLayout);
      } else if (isa<VectorType>(arg.getType())) {
        auto layout = getLayout(arg);
        in_layouts.push_back(layout);
      } else {
        op.emitOpError() << "unsupported arg type " << arg.getType()
                         << " in scf::for";
        return failure();
      }
    }
    ArrayRef<Layout> out_layouts = ArrayRef<Layout>(in_layouts).drop_front(3);
    // Use tpu.assume_layout to annotate every block argument with the layout of
    // the corresponding operand in forOp and replace all uses of the block
    // argument with the result of tpu.assume_layout.
    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), op.getBody());

    // Drop the induction_variable and layouts of bounds+step (respectively).
    for (auto [iter_arg, layout] : llvm::zip_equal(
             op.getBody()->getArguments().drop_front(1), out_layouts)) {
      if (!dyn_cast<VectorType>(iter_arg.getType())) {
        continue;
      }
      auto assume_layout_op =
          builder.create<AssumeLayoutOp>(iter_arg.getType(), iter_arg);
      setLayout(assume_layout_op, layout, layout);
      iter_arg.replaceUsesWithIf(assume_layout_op, [&](OpOperand &operand) {
        return operand.getOwner() != assume_layout_op;
      });
    }

    if (inferBlock(*op.getBody(), match_yield).failed()) {
      return failure();
    }
    auto yield_op = op.getBody()->getTerminator();
    setInLayout(yield_op, out_layouts);
    setLayout(op, in_layouts, out_layouts);
    return success();
  }

  LogicalResult infer(tpu::RotateOp op) {
    auto bitwidth = op.getType().getElementTypeBitWidth();
    if (bitwidth != 32) {
      NYI("Rotate with non-32-bit data");
    }
    if (op.getType().getRank() < 2) {
      NYI("Unsupported 1D shape");
    }
    auto layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                               ImplicitDim::kNone);
    setLayout(op, layout, layout);
    return success();
  }

  LogicalResult infer(tpu::ConcatenateOp op) {
    TPU_CHECK_OP(!op.getSources().empty(),
                 "Need at least one vector to concatenate");
    auto res_rank = op.getType().getRank();
    auto dimension = op.getDimension();
    TPU_CHECK_OP(0 <= dimension && dimension < res_rank,
                 "Expect a valid concatenate dimension");
    if (res_rank == 1) {
      NYI("Support concatenation with 1D vectors");
    }
    auto res_ty = op.getResult().getType();
    int8_t bitwidth = res_ty.getElementTypeBitWidth();
    if (bitwidth != 32) {
      NYI("Support concatenation with non 32-bit data");
    }
    auto layout = (dimension >= res_rank - 2)
                      ? VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                     ImplicitDim::kNone)
                      : getLayout(op.getSources().front());
    SmallVector<Layout> in_layouts(op->getNumOperands(), layout);
    setLayout(op, in_layouts, in_layouts.back());
    return success();
  }

  LogicalResult infer(tpu::LoadOp op) {
    auto res_ty = op.getResult().getType();
    int8_t bitwidth = res_ty.getElementTypeBitWidth();

    // We expect the result is already a native-sized vreg.
    TPU_CHECK_OP(bitwidth == 32 && res_ty.getShape()[0] == target_shape_[0] &&
                     res_ty.getShape()[1] == target_shape_[1],
                 "Only 32-bit loads supported");
    SmallVector<Layout, 4> in_layout(op->getNumOperands(), kNoLayout);
    auto out_layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                   ImplicitDim::kNone);
    setLayout(op, in_layout, out_layout);
    return success();
  }

  LogicalResult infer(tpu::MatmulOp op) { return inferMatmul(op); }

  LogicalResult infer(tpu::StoreOp op) {
    auto store_ty = op.getValueToStore().getType();
    int8_t bitwidth = store_ty.getElementTypeBitWidth();

    // We expect the value to store is already a native-sized vreg.
    TPU_CHECK_OP(bitwidth == 32 && store_ty.getShape()[0] == target_shape_[0] &&
                     store_ty.getShape()[1] == target_shape_[1],
                 "Only 32-bit stores supported");
    auto store_layout = VectorLayout(bitwidth, {0, 0}, nativeTiling(bitwidth),
                                     ImplicitDim::kNone);
    SmallVector<Layout, 5> in_layout{store_layout};
    in_layout.insert(in_layout.end(), op.getIndices().size() + 1, kNoLayout);
    setInLayout(op, in_layout);
    return success();
  }

  LogicalResult infer(tpu::EraseLayoutOp op) {
    setLayout(op, kNoLayout, kNoLayout);
    return success();
  }

  LogicalResult infer(tpu::GatherOp op) {
    auto src_layout = getLayout(op.getSource());
    setLayout(op, src_layout, src_layout);
    return success();
  }

  LogicalResult infer(tpu::BitcastOp op) {
    auto src_layout = getLayout(op.getInput());
    LayoutOffsets src_offsets = src_layout->offsets();
    if (src_offsets[0].value_or(0) || src_offsets[1].value_or(0)) {
      NYI("unsupported bitcast with offsets");
    }
    if (src_layout->implicit_dim() != ImplicitDim::kNone) {
      NYI("unsupported bitcast with an implicit dim");
    }
    // Check if input and output have same bit size.
    auto in_ty = dyn_cast<VectorType>(op.getInput().getType());
    auto out_ty = dyn_cast<VectorType>(op.getOutput().getType());
    auto in_bitwidth = in_ty.getElementTypeBitWidth();
    auto out_bitwidth = out_ty.getElementTypeBitWidth();
    TPU_CHECK_OP(in_ty && out_ty && in_ty.getRank() == out_ty.getRank(),
                 "Input and output have different rank");
    if (out_ty.getRank() < 2) {
      NYI("Support bitcast with 1D vector");
    }
    for (int i = 0; i < in_ty.getRank(); ++i) {
      auto in_dim = in_ty.getDimSize(i);
      auto out_dim = out_ty.getDimSize(i);

      // The sublane dimension is scaled down by the ratio of input element
      // bitwidth to output element bitwidth when bitcasting. For example,
      // bitcasting a vector<16x128xbf16> to a vector<8x128xi32> packs every 2
      // rows in the bf16 vector into 1 row in the i32 vector. This means the
      // bit representation of one i32 element vector[i,j] is equal to
      // concatenating bf16 elements vector[2*i+1,j] and vector[2*i,j].
      if (i == in_ty.getRank() - 2) {
        in_dim *= in_bitwidth;
        out_dim *= out_bitwidth;
      }
      TPU_CHECK_OP(in_dim == out_dim,
                   "Input and output have incompatible shape");
    }
    setLayout(op,
              VectorLayout(in_bitwidth, src_offsets, nativeTiling(in_bitwidth),
                           ImplicitDim::kNone),
              VectorLayout(out_bitwidth, src_offsets,
                           nativeTiling(out_bitwidth), ImplicitDim::kNone));
    return success();
  }

  LogicalResult infer(tpu::RepeatOp op) {
    auto src_layout = getLayout(op.getSource());
    setLayout(op, src_layout, src_layout);
    return success();
  }

  LogicalResult infer(tpu::TraceOp op) {
    static LogicalResult (*match_yield)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<tpu::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 0, "expected no operands");
    TPU_CHECK_OP(op->getNumResults() == 0, "results unsupported");
    return inferBlock(*op.getBody(), match_yield);
  }

  LogicalResult infer(tpu::RegionOp op) {
    static LogicalResult (*match_region)(Operation *) = [](Operation *op) {
      TPU_CHECK_OP(isa<tpu::YieldOp>(op), "expected yield terminator");
      return success();
    };
    TPU_CHECK_OP(op->getNumOperands() == 0, "expected no operands");
    TPU_CHECK_OP(op->getNumResults() == 0, "results unsupported");
    return inferBlock((*op).getRegion(0).getBlocks().front(), match_region);
  }

  LogicalResult infer(tpu::IotaOp op) {
    auto ty = op.getResult().getType();
    TPU_CHECK_OP(ty.getElementType().isSignlessInteger(32),
                 "Only 32-bit integer iota supported");
    TPU_CHECK_OP(ty.getRank() >= 2, "iota rank below 2D unsupported");
    LayoutOffsets offsets = {0, 0};
    if (op.getDimension() == ty.getRank() - 1) {
      offsets[0] = std::nullopt;
    }
    if (op.getDimension() == ty.getRank() - 2) {
      offsets[1] = std::nullopt;
    }
    setOutLayout(op, VectorLayout(kNativeBitwidth, offsets, default_tiling_,
                                  ImplicitDim::kNone));
    return success();
  }

  LogicalResult infer(vector::BroadcastOp op) {
    auto some_src_ty = op.getSourceType();
    auto res_ty = op.getResultVectorType();
    TPU_CHECK_OP(res_ty.getRank() > 0, "rank 0 vectors unsupported");
    if (some_src_ty.isSignlessIntOrIndexOrFloat()) {
      TPU_CHECK_OP(some_src_ty.getIntOrFloatBitWidth() == kNativeBitwidth,
                   "Only 32-bit broadcasts supported");
      if (res_ty.getRank() == 1) {
        // We use a full vreg tile, because only then its layout can be changed
        // for free.
        setLayout(op, kNoLayout,
                  VectorLayout(kNativeBitwidth, {std::nullopt, std::nullopt},
                               default_tiling_, ImplicitDim::kSecondMinor));
      } else {  // rank >= 2  // NOLINT(readability-else-after-return)
        setLayout(op, kNoLayout,
                  VectorLayout(kNativeBitwidth, {std::nullopt, std::nullopt},
                               default_tiling_, ImplicitDim::kNone));
      }
      return success();
    }
    if (auto src_ty = dyn_cast<VectorType>(some_src_ty)) {
      TPU_CHECK_OP(src_ty.getRank() >= 2, "source rank below 2D unsupported");
      TPU_CHECK_OP(res_ty.getRank() >= 2, "result rank below 2D unsupported");
      auto some_layout = getLayout(op.getSource());
      TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
      // Since we can only do sublane broadcasts in the (8, 128) tiling, we
      // should always use that when sublane broadcasting is required.
      if (src_ty.getDimSize(src_ty.getRank() - 2) !=
          res_ty.getDimSize(res_ty.getRank() - 2)) {
        if (some_layout->bitwidth() != kNativeBitwidth) {
          NYI("Only 32-bit broadcasts supported");
        }
        LayoutOffsets offsets = some_layout->offsets();
        // Only when src tiling[0] is 1 we can use replicated. The other tilings
        // are padded for one row.
        if (some_layout->tiling()[0] == 1) {
          offsets[0] = std::nullopt;
        }
        *some_layout = VectorLayout(some_layout->bitwidth(), offsets,
                                   default_tiling_, some_layout->implicit_dim());
      }
      auto &layout = *some_layout;
      if (layout.implicit_dim() != ImplicitDim::kNone) {
        VectorLayout layout_2d(layout.bitwidth(), layout.offsets(),
                               layout.tiling(), ImplicitDim::kNone);
        if (layout_2d.equivalentTo(layout, src_ty.getShape(), target_shape_)) {
          layout = layout_2d;
        } else {
          op.emitOpError() << "Only 2D layouts supported";
          return failure();
        }
      }
      auto src_tiled_shape = src_ty.getShape().take_back(2);
      auto dst_tiled_shape = res_ty.getShape().take_back(2);
      LayoutOffsets offsets = layout.offsets();
      if (layout.bitwidth() == kNativeBitwidth &&
          layout.tiling() == default_tiling_) {
        for (int i = 0; i < 2; ++i) {
          if (src_tiled_shape[i] != dst_tiled_shape[i]) {
            offsets[i] = std::nullopt;
          }
        }
      }
      setLayout(op, some_layout,
                VectorLayout(layout.bitwidth(), offsets, layout.tiling(),
                             ImplicitDim::kNone));
      return success();
    }
    op.emitOpError("unsupported broadcast source type");
    return failure();
  }

  LogicalResult infer(vector::ContractionOp op) {
    // TODO(apaszke): Support layout here, at least on batch dimensions.
    TPU_CHECK_OP(op.getKind() == vector::CombiningKind::ADD,
                 "Only ADD supported");
    auto ctx = op.getContext();
    const auto matmul_iterator_types = mlir::ArrayAttr::get(
        ctx,
        {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});
    TPU_CHECK_OP(op.getIteratorTypes() == matmul_iterator_types,
                 "Not a matmul");
    const auto matmul_indexing_maps = mlir::ArrayAttr::get(
        ctx,
        {AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx)}, ctx)),
         AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(2, ctx), getAffineDimExpr(1, ctx)}, ctx)),
         AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)},
             ctx))});
    const auto matmul_indexing_maps_transposed = mlir::ArrayAttr::get(
        ctx,
        {AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx)}, ctx)),
         AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx)),
         AffineMapAttr::get(AffineMap::get(
             3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)},
             ctx))});
    TPU_CHECK_OP(op.getIndexingMaps() == matmul_indexing_maps ||
                     op.getIndexingMaps() == matmul_indexing_maps_transposed,
                 "Not a matmul");
    return inferMatmul(op);
  }

  LogicalResult infer(vector::ExtractOp op) {
    TPU_CHECK_OP(!op.hasDynamicPosition(), "dynamic indices not supported");
    TPU_CHECK_OP(
        op.getSourceVectorType().getElementTypeBitWidth() == kNativeBitwidth,
        "Only 32-bit types supported");
    auto layout = getLayout(op.getVector());
    TPU_CHECK_OP(layout.has_value(), "missing vector layout");
    setLayout(op,
              VectorLayout(kNativeBitwidth, {0, 0}, layout->tiling(),
                           layout->implicit_dim()),
              kNoLayout);
    return success();
  }

  LogicalResult infer(vector::LoadOp op) {
    auto src_ty = op.getMemRefType();
    auto res_ty = op.getVectorType();
    TPU_CHECK_OP(src_ty.getRank() == res_ty.getRank(),
                 "memref and vector rank mismatch");
    int64_t rank = res_ty.getRank();
    int8_t bitwidth = res_ty.getElementTypeBitWidth();
    auto maybe_tiling =
        verifyMemoryTiling(op, getMemRefLayout(op.getBase()).getTiles(),
                           src_ty.getRank(), src_ty.getElementTypeBitWidth());
    if (!maybe_tiling) {
      return failure();
    }
    auto tiling = *maybe_tiling;

    SmallVector<Layout, 4> in_layout(op->getNumOperands(), kNoLayout);
    CHECK_EQ(op->getNumOperands(), op.getIndices().size() + 1);
    SmallVector<int64_t, 2> tile_indices;
    for (int i = 0; i < tiling.size(); ++i) {
      int dim = rank - tiling.size() + i;
      Value tiled_index = op.getIndices()[dim];
      if (auto cst_op = tiled_index.getDefiningOp<arith::ConstantOp>()) {
        tile_indices.push_back(cast<IntegerAttr>(cst_op.getValue()).getInt());
      } else {
        if (failed(verifyDivisibleIndex(tiled_index, tiling[0], dim, op))) {
          return failure();
        }
        tile_indices.push_back(0);
      }
    }

    if (rank == 0) {
      op.emitOpError("rank 0 vectors unsupported");
      return failure();
    }
    if (rank == 1) {
      TPU_CHECK_OP(tiling.size() == 1, "Expected 1D tiling in 1D loads");
      auto tile = tiling.front();
      TPU_CHECK_OP(tile % target_shape_[1] == 0,
                   "Unsupported tiling for 1D load");
      CHECK_EQ(tile_indices.size(), 1);
      int64_t idx = tile_indices.front();
      int64_t offset = idx % kVmemAlignment32;
      // TODO(apaszke): We could generate replicated loads for short values.
      setLayout(op, in_layout,
                VectorLayout(bitwidth, {0, offset}, {1, tile},
                             ImplicitDim::kSecondMinor));
    } else {  // rank >= 2
      TPU_CHECK_OP(tiling.size() == 2, "Expected 2D tiling in 2D+ loads");
      CHECK_EQ(tile_indices.size(), 2);
      std::array<std::optional<int64_t>, 2> offsets;
      const auto tile_src_shape = src_ty.getShape().take_back(2);
      const auto tile_res_shape = res_ty.getShape().take_back(2);
      const int64_t num_sublanes = tile_res_shape[0];
      // For now, we focus on tilings that span full sublanes.
      TPU_CHECK_OP(tiling[1] == target_shape_[1],
                   "Unsupported tiling for 2d load");
      // We can load starting from any row if the source has few columns,
      // because the tiling structure degenerates to regular layout there.
      // There is also no extra need for alignment if we load a single sublane.
      // TODO(apaszke): Also no need to align if we don't exceed the base chunk!
      if (bitwidth == 32 &&
          (tile_src_shape[1] <= target_shape_[1] || num_sublanes == 1)) {
        offsets[0] = 0;
      } else {
        offsets[0] = tile_indices[0] % tiling[0];
      }
      offsets[1] = tile_indices[1] % target_shape_[1];
      std::array<int64_t, 2> layout_tiling{tiling[0], tiling[1]};
      if (num_sublanes == 1 && bitwidth == 32 &&
          tiling[1] == target_shape_[1] &&
          tile_res_shape[1] > target_shape_[1]) {
        // We can strided load sublanes if we're loading a single sublane for
        // multiple times. Enabling this helps load one entire row from memref
        // more efficiently.
        setLayout(op, in_layout,
                  VectorLayout(bitwidth, offsets, {1, layout_tiling[1]},
                               ImplicitDim::kNone));
      } else if (num_sublanes == 1 && bitwidth == 32 &&
                 tiling == target_shape_) {
        // We can use replicated loads if we're only loading a single sublane.
        setLayout(op, in_layout,
                  VectorLayout(bitwidth, {std::nullopt, offsets[1]},
                               layout_tiling, ImplicitDim::kNone));
      } else {
        setLayout(
            op, in_layout,
            VectorLayout(bitwidth, offsets, layout_tiling, ImplicitDim::kNone));
      }
    }
    return success();
  }

  LogicalResult infer(vector::ExtractStridedSliceOp op) {
    auto input_layout = getLayout(op.getVector());
    TPU_CHECK_OP(input_layout, "missing vector layout");
    TPU_CHECK_OP(input_layout->implicit_dim() == ImplicitDim::kNone,
                 "only 2D layouts supported");
    TPU_CHECK_OP(op.getType().getElementTypeBitWidth() == 32,
                 "Only 32-bit types supported");
    auto offsets = op.getOffsets().getValue();
    auto strides = op.getStrides().getValue();
    for (auto offset_attr : offsets.take_back(2)) {
      int off = offset_attr.cast<IntegerAttr>().getInt();
      TPU_CHECK_OP(off == 0, "Only zero-offset slices supported.");
    }
    for (auto stride : strides) {
      TPU_CHECK_OP(stride.cast<IntegerAttr>().getInt() == 1,
                   "Only trivial strides supported.");
    }

    setLayout(op, input_layout, input_layout);
    return success();
  }

  LogicalResult infer(vector::MultiDimReductionOp op) {
    auto src_ty = op.getSourceVectorType();
    auto dst_ty = dyn_cast<VectorType>(op.getDestType());
    TPU_CHECK_OP(dst_ty, "only reductions with vector results supported");
    SmallVector<int64_t> dims;
    dims.reserve(op.getReductionDims().size());
    for (Attribute dim_attr : op.getReductionDims()) {
      dims.push_back(cast<IntegerAttr>(dim_attr).getInt());
    }
    int64_t src_rank = src_ty.getRank();
    auto acc_layout = getLayout(op.getAcc());
    TPU_CHECK_OP(is_fully_replicated(acc_layout),
                 "only constant accumulators supported");
    TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == kNativeBitwidth,
                 "only 32-bit reductions supported");
    auto some_src_layout = getLayout(op.getSource());
    TPU_CHECK_OP(some_src_layout, "missing vector layout");
    auto &src_layout = *some_src_layout;
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
    if ((reduces[0] || reduces[1]) &&
        !src_layout.hasNativeTiling(target_shape_)) {
      src_layout = VectorLayout(kNativeBitwidth, src_layout.offsets(),
                                default_tiling_, src_layout.implicit_dim());
    }
    LayoutOffsets out_offsets = src_layout.offsets();
    for (int i = 0; i < out_offsets.size(); ++i) {
      if (reduces[i]) {
        out_offsets[i] = std::nullopt;
      }
    }
    ImplicitDim out_implicit_dim = src_layout.implicit_dim();
    if ((reduces[0] && reduces[1]) ||
        (src_layout.implicit_dim() != ImplicitDim::kNone &&
         (reduces[0] || reduces[1]))) {
      TPU_CHECK_OP(
          dst_ty.getRank() > 0 && *(dst_ty.getShape().end() - 1) == 1,
          "Not implemented: reductions over both trailing dimensions are only "
          "supported when the resulting value has a trailing axis of size 1");
      out_implicit_dim = VectorLayout::ImplicitDim::kSecondMinor;
    } else if (reduces[0]) {
      out_implicit_dim = VectorLayout::ImplicitDim::kSecondMinor;
    } else if (reduces[1]) {
      out_implicit_dim = VectorLayout::ImplicitDim::kMinor;
    }
    setLayout(op, {src_layout, acc_layout},
              VectorLayout(src_layout.bitwidth(), out_offsets,
                           src_layout.tiling(), out_implicit_dim));
    return success();
  }

  LogicalResult infer(vector::ShapeCastOp op) {
    auto src_ty = op.getSourceVectorType();
    auto src_shape = src_ty.getShape();
    int64_t src_rank = src_ty.getRank();
    auto res_ty = op.getResultVectorType();
    auto res_shape = res_ty.getShape();
    int64_t res_rank = res_ty.getRank();
    auto some_src_layout = getLayout(op.getSource());
    TPU_CHECK_OP(some_src_layout, "missing vector layout");
    auto layout = *some_src_layout;
    if (layout.implicit_dim() == ImplicitDim::kNone) {
      // Nothing changes in the last two dims.
      if (res_rank >= 2 && src_shape.take_back(2) == res_shape.take_back(2)) {
        setLayout(op, layout, layout);
        return success();
      }
      // Sublane (un)tiling.
      if (res_rank >= 2 && layout.offsets() == LayoutOffsets{0, 0} &&
          layout.tiling()[1] == target_shape_[1] &&
          src_ty.getDimSize(src_ty.getRank() - 1) ==
              res_shape[res_shape.size() - 1] &&
          src_ty.getDimSize(src_ty.getRank() - 2) % layout.tiling()[0] == 0 &&
          res_shape[res_shape.size() - 2] % layout.tiling()[0] == 0) {
        setLayout(op, layout, layout);
        return success();
      }
      // Lane (un)tiling.
      if (layout.offsets() == LayoutOffsets{0, 0} &&
          layout.tiling()[1] == target_shape_[1] &&
          src_ty.getDimSize(src_ty.getRank() - 1) !=
              res_shape[res_shape.size() - 1] &&
          src_ty.getDimSize(src_ty.getRank() - 1) % layout.tiling()[1] == 0 &&
          res_shape[res_shape.size() - 1] % layout.tiling()[1] == 0) {
        // TODO(jevinjiang): support shapecast along lane with any bitwidth.
        if (src_ty.getElementTypeBitWidth() != kNativeBitwidth) {
          NYI("Shapecast along lane dimension when bitwidth is not 32");
        }

        // When we shapecast from input shape (..., m * target_shape_[1]) to
        // output shape (..., target_shape_[1]), the reshape becomes no-op when
        // input is densely packed with tiling (1, target_shape_[1]) and
        // output has the native tiling.
        if (*(res_shape.end() - 1) == target_shape_[1] &&
            *(res_shape.end() - 2) % target_shape_[0] == 0 &&
            *(src_shape.end() - 1) % (target_shape_[0] * target_shape_[1]) ==
                0 &&
            (*(src_shape.end() - 2) == 1 ||
             *(src_shape.end() - 2) % target_shape_[0] == 0)) {
          // Inferring in_layout to have tiling (1, 128) triggers any
          // necessary relayout before shapecast.
          setLayout(op,
                    VectorLayout(layout.bitwidth(), layout.offsets(),
                                 {1, target_shape_[1]}, ImplicitDim::kNone),
                    VectorLayout(layout.bitwidth(), layout.offsets(),
                                 default_tiling_, ImplicitDim::kNone));
          return success();
        }

        // When we shapecast from input shape (..., target_shape_[1]) to
        // output shape (..., m * target_shape_[1]), the reshape becomes no-op
        // when input has the native tiling and output is densely packed with
        // tiling (1, target_shape_[1]).
        if (*(src_shape.end() - 1) == target_shape_[1] &&
            *(src_shape.end() - 2) % target_shape_[0] == 0 &&
            *(res_shape.end() - 1) % (target_shape_[0] * target_shape_[1]) ==
                0 &&
            (*(res_shape.end() - 2) == 1 ||
             *(res_shape.end() - 2) % target_shape_[0] == 0)) {
          setLayout(op,
                    VectorLayout(layout.bitwidth(), layout.offsets(),
                                 default_tiling_, ImplicitDim::kNone),
                    VectorLayout(layout.bitwidth(), layout.offsets(),
                                 {1, target_shape_[1]}, ImplicitDim::kNone));
          return success();
        }

        // TODO(b/299253805): support shapecast along lane for other cases.
        op.emitOpError("unsupported shape cast");
        return failure();
      }
      unsigned bitwidth = src_ty.getElementTypeBitWidth();
      auto native_tiling = nativeTiling(bitwidth);
      if (layout.tiling() != native_tiling) {
        layout = VectorLayout(bitwidth, layout.offsets(), native_tiling,
                              layout.implicit_dim());
      }
      TPU_CHECK_OP(src_ty.getRank() >= 2,
                   "expected 2D+ operand with 2D layout");
      ArrayRef<int64_t> layout_shape = src_ty.getShape().take_back(2);
      if (res_ty.getRank() >= 2) {
        // Squeeze out the sublane dim.
        if (layout_shape[0] == 1 &&
            res_shape.drop_back(1) == src_shape.drop_back(2) &&
            res_shape.back() == src_shape.back()) {
          setLayout(op, layout,
                    VectorLayout(bitwidth, layout.offsets(), layout.tiling(),
                                 ImplicitDim::kSecondMinor));
          return success();
        }
        // Insert a singleton lane dimension. The old lane dimension ends up
        // in the sublane dimension. Other axes can be reshaped arbitrarily.
        if (src_ty.getElementTypeBitWidth() == kNativeBitwidth &&
            src_shape.back() == res_shape[res_shape.size() - 2] &&
            res_shape.back() == 1) {
          setLayout(op, layout,
                    VectorLayout(kNativeBitwidth, {0, std::nullopt},
                                 default_tiling_, ImplicitDim::kNone));
          return success();
        }
      } else if (res_ty.getRank() == 1) {
        bool all_one = true;
        for (int64_t s : src_ty.getShape().drop_back(2)) {
          all_one &= s == 1;
        }
        // Squeeze out everything, but lanes
        if (layout_shape[0] == 1 && all_one &&
            res_ty.getShape().back() == layout_shape[1]) {
          setLayout(op, layout,
                    VectorLayout(bitwidth, layout.offsets(), layout.tiling(),
                                 ImplicitDim::kSecondMinor));
          return success();
        }
        // Squeeze out everything, but sublanes
        if (layout_shape[1] == 1 && all_one &&
            res_ty.getShape().back() == layout_shape[0]) {
          TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == kNativeBitwidth,
                       "only 32-bit shape casts supported");
          setLayout(op, layout,
                    VectorLayout(kNativeBitwidth, layout.offsets(),
                                 layout.tiling(), ImplicitDim::kMinor));
          return success();
        }
      }
    } else {
      // Nothing changes in the last dim.
      if (res_ty.getRank() >= 1 && src_shape.back() == res_shape.back()) {
        setLayout(op, layout, layout);
        return success();
      }
      TPU_CHECK_OP(src_ty.getElementTypeBitWidth() == kNativeBitwidth,
                   "only 32-bit shape casts supported");
      // Insert a singleton innermost dim.
      if (res_ty.getRank() == src_ty.getRank() + 1 &&
          src_ty.getDimSize(src_rank - 1) == res_ty.getDimSize(res_rank - 2) &&
          res_ty.getDimSize(res_rank - 1) == 1) {
        if (layout.implicit_dim() == ImplicitDim::kMinor) {
          setLayout(op, layout,
                    VectorLayout(kNativeBitwidth, layout.offsets(),
                                 default_tiling_, ImplicitDim::kNone));
        } else {
          TPU_CHECK_OP(layout.implicit_dim() == ImplicitDim::kSecondMinor,
                       "unexpected implicit dim value");
          setLayout(op, layout,
                    VectorLayout(kNativeBitwidth, {0, std::nullopt},
                                 default_tiling_, ImplicitDim::kNone));
        }
        return success();
      }
    }
    op.emitOpError("unsupported shape cast");
    return failure();
  }

  LogicalResult infer(vector::StoreOp op) {
    auto ref_ty = op.getMemRefType();
    auto store_ty = op.getValueToStore().getType();
    TPU_CHECK_OP(ref_ty.getRank() == store_ty.getRank(),
                 "memref and vector rank mismatch");
    int64_t rank = ref_ty.getRank();
    int8_t bitwidth = store_ty.getElementTypeBitWidth();
    auto maybe_tiling =
        verifyMemoryTiling(op, getMemRefLayout(op.getBase()).getTiles(),
                           ref_ty.getRank(), ref_ty.getElementTypeBitWidth());
    if (!maybe_tiling) {
      return failure();
    }
    auto tiling = *maybe_tiling;

    SmallVector<int64_t, 2> tile_indices;
    for (int i = 0; i < tiling.size(); ++i) {
      int dim = rank - tiling.size() + i;
      Value tiled_index = op.getIndices()[dim];
      if (auto cst_op = tiled_index.getDefiningOp<arith::ConstantOp>()) {
        tile_indices.push_back(cast<IntegerAttr>(cst_op.getValue()).getInt());
      } else {
        if (failed(verifyDivisibleIndex(tiled_index, tiling[0], dim, op))) {
          return failure();
        }
        tile_indices.push_back(0);
      }
    }

    Layout store_layout;
    if (rank == 0) {
      op.emitOpError("rank 0 vectors unsupported");
      return failure();
    }
    if (rank == 1) {
      TPU_CHECK_OP(tiling.size() == 1, "Expected 1D tiling in 1D store");
      auto tile = tiling.front();
      TPU_CHECK_OP(tile % target_shape_[1] == 0,
                   "Unsupported 1D tiling for 1D store");
      CHECK_EQ(tile_indices.size(), 1);
      int64_t idx = tile_indices.front();
      int64_t offset = idx % kVmemAlignment32;
      store_layout = VectorLayout(bitwidth, {0, offset}, {1, tile},
                                  ImplicitDim::kSecondMinor);
    } else {  // rank >= 2  // NOLINT(readability-else-after-return)
      TPU_CHECK_OP(tiling.size() == 2, "Expected 2D tiling in 2D+ store");
      CHECK_EQ(tile_indices.size(), 2);
      std::array<std::optional<int64_t>, 2> offsets;
      const auto tile_ref_shape = ref_ty.getShape().take_back(2);
      const auto tile_store_shape = store_ty.getShape().take_back(2);
      const int64_t num_sublanes = tile_store_shape[0];
      // For now, we focus on tilings that span full sublanes.
      TPU_CHECK_OP(tiling[1] == target_shape_[1],
                   "Unsupported tiling for 2d store");
      // We can store starting from any row if the source has few columns,
      // because the tiling structure degenerates to regular layout there.
      // There is also no extra need for alignment if we store a single sublane.
      // TODO(apaszke): Also no need to align if we don't exceed the base chunk!
      if (bitwidth == 32 &&
          (tile_ref_shape[1] <= target_shape_[1] || num_sublanes == 1)) {
        offsets[0] = 0;
      } else {
        offsets[0] = tile_indices[0] % tiling[0];
      }
      offsets[1] = tile_indices[1] % target_shape_[1];
      if (num_sublanes == 1 && bitwidth == 32 &&
          tiling[1] == target_shape_[1] &&
          tile_store_shape[1] > target_shape_[1]) {
        // We can strided store sublanes if we're storing a single sublane for
        // multiple times. Enabling this helps store one entire row to memref
        // more efficiently.
        store_layout = VectorLayout(store_ty.getElementTypeBitWidth(), offsets,
                                    {1, tiling[1]}, ImplicitDim::kNone);
      } else {
        store_layout = VectorLayout(store_ty.getElementTypeBitWidth(), offsets,
                                    {tiling[0], tiling[1]}, ImplicitDim::kNone);
      }
    }
    SmallVector<Layout, 5> in_layout{store_layout};
    in_layout.insert(in_layout.end(), op.getIndices().size() + 1, kNoLayout);
    setInLayout(op, in_layout);
    return success();
  }

  LogicalResult infer(vector::TransposeOp op) {
    auto permutation = op.getPermutation();
    auto some_layout = getLayout(op.getVector());
    TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
    auto &layout = *some_layout;
    auto src_ty = op.getSourceVectorType();
    TPU_CHECK_OP(permutation.size() == src_ty.getRank(),
                 "Transpose permutation has incorrect rank");
    if (layout.implicit_dim() == ImplicitDim::kNone) {
      TPU_CHECK_OP((layout.offsets() == LayoutOffsets{0, 0}),
                   "Padded transposes unsupported");
      auto xlu_width = target_shape_[1];
      for (int64_t s : src_ty.getShape().take_back(2)) {
        TPU_CHECK_OP(s % xlu_width == 0, "Padded transposes unsupported");
      }
      for (auto dim : permutation.drop_back(2)) {
        TPU_CHECK_OP(
            dim < src_ty.getRank() - 2,
            "Unsupported transpose permutation - minor dims into major");
      }
      for (auto dim : permutation.take_back(2)) {
        TPU_CHECK_OP(
            dim >= src_ty.getRank() - 2,
            "Unsupported transpose permutation - major dims into minor");
      }
      Layout required_layout = some_layout;
      if (permutation.size() < 2) {
        return failure();
      }
      // Require native tiling if we're going to use the XLU.
      if (permutation[permutation.size() - 1] == permutation.size() - 2) {
        auto native_tiling = nativeTiling(layout.bitwidth());
        required_layout = VectorLayout(layout.bitwidth(), layout.offsets(),
                                       native_tiling, ImplicitDim::kNone);
      }
      setLayout(op, required_layout, required_layout);
      return success();
    }
    op.emitOpError("Unsupported transpose");
    return failure();
  }

  LogicalResult inferElementwise(Operation *op, bool check_bitwidth = true) {
    TPU_CHECK_OP(op->getNumResults() == 1, "only one result supported");
    TPU_CHECK_OP(op->getNumOperands() > 0,
                 "elementwise ops with no operands unsupported");
    // Elementwise operators can be parameterized by both scalars and shaped
    // types, so make sure we infer layout based on a shaped-typed operand.
    std::optional<VectorLayout> out_layout_candidate;
    std::optional<VectorLayout> out_layout;
    SmallVector<std::optional<Layout>, 4> in_layouts;
    int64_t bit_width = -1;
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      if (auto vty = dyn_cast<VectorType>(op->getOperand(i).getType())) {
        if (bit_width == -1) {
          bit_width = vty.getElementTypeBitWidth();
        }
        TPU_CHECK_OP(
            !check_bitwidth || bit_width == vty.getElementTypeBitWidth(),
            "Generic elementwise rule only supports operands of same width");
        auto some_layout = getLayout(op->getOperand(i));
        TPU_CHECK_OP(some_layout.has_value(), "missing vector layout");
        auto &layout = *some_layout;
        // If the input is fully replicated, don't use it to commit to any
        // layout. Replicated values are easy to relayout.
        if (is_fully_replicated(some_layout)) {
          in_layouts.push_back(std::nullopt);
          out_layout_candidate = layout;
          continue;
        }
        if (!out_layout) {
          // TODO(apaszke): There are probably smarter ways to choose layout.
          out_layout = layout;
          in_layouts.push_back(some_layout);
        } else {
          if (auto new_out =
                  VectorLayout::join(layout, *out_layout, vty.getShape())) {
            out_layout = *new_out;
            in_layouts.push_back(some_layout);
          } else {
            // When we detect a layout conflict we cannot reconcile, we remove
            // any replication bits that might have been present in out_layout,
            // since there is no guarantee that the conflicting inputs could
            // even become replicated.
            out_layout =
                VectorLayout(out_layout->bitwidth(),
                             {out_layout->offsets()[0].value_or(0),
                              out_layout->offsets()[1].value_or(0)},
                             out_layout->tiling(), out_layout->implicit_dim());
            in_layouts.push_back(std::nullopt);
          }
        }
      } else {
        TPU_CHECK_OP(op->getOperand(i).getType().isSignlessIntOrIndexOrFloat(),
                     "expected only vector and scalar operands");
        in_layouts.push_back({kNoLayout});
      }
    }
    Layout final_out_layout = std::nullopt;
    if (auto out_vty = dyn_cast<VectorType>(op->getResult(0).getType())) {
      TPU_CHECK_OP(
          !check_bitwidth || bit_width == out_vty.getElementTypeBitWidth(),
          "Generic elementwise rule can't change element type width");
      if (out_layout) {
        final_out_layout = *out_layout;
      } else if (out_layout_candidate) {
        final_out_layout = *out_layout_candidate;
      } else {
        op->emitOpError(
            "Elementwise op has no vector operands but returns a vector?");
        return failure();
      }
    }
    CHECK_EQ(in_layouts.size(), op->getNumOperands()) << Print(op);
    SmallVector<Layout, 4> final_in_layouts;
    for (int i = 0; i < in_layouts.size(); ++i) {
      if (in_layouts[i]) {
        final_in_layouts.push_back(*in_layouts[i]);
      } else {
        final_in_layouts.push_back(final_out_layout);
      }
    }
    setLayout(op, final_in_layouts, final_out_layout);
    return success();
  }

  LogicalResult inferMatmul(Operation *op) {
    auto get_unpadded_layout =
        [&](Value v, std::optional<int64_t> major_multiple = std::nullopt,
            std::optional<int64_t> minor_multiple =
                std::nullopt) -> std::optional<VectorLayout> {
      auto pad = getLayout(v);
      if (!pad.has_value() || pad->implicit_dim() != ImplicitDim::kNone) {
        return std::nullopt;
      }
      auto vty = cast<VectorType>(v.getType());
      auto tiling = nativeTiling(vty.getElementTypeBitWidth());
      auto shape = vty.getShape().take_back(2);
      if (pad->offsets()[0].value_or(0) != 0 ||
          pad->offsets()[1].value_or(0) != 0 ||
          shape[0] % major_multiple.value_or(tiling[0]) != 0 ||
          shape[1] % minor_multiple.value_or(tiling[1]) != 0) {
        return std::nullopt;
      }
      // Override tiling to match the native one.
      return VectorLayout(pad->bitwidth(), pad->offsets(), tiling,
                          ImplicitDim::kNone);
    };
    auto res_ty = dyn_cast<VectorType>(op->getResult(0).getType());
    TPU_CHECK_OP(res_ty, "only vector results supported");
    TPU_CHECK_OP(res_ty.getElementTypeBitWidth() == kNativeBitwidth,
                 "only 32-bit matmul results supported");
    std::array<Layout, 3> in_layout;
    CHECK_EQ(op->getNumOperands(), 3);
    std::optional<int64_t> lhs_major_multiple;
    // We don't restrict the first lhs axis when the data is not packed.
    if (cast<VectorType>(op->getOperand(0).getType())
            .getElementTypeBitWidth() == kNativeBitwidth) {
      lhs_major_multiple = 1;
    }
    in_layout[0] =
        get_unpadded_layout(op->getOperand(0), lhs_major_multiple, 1);
    in_layout[1] = get_unpadded_layout(op->getOperand(1), 128, 1);
    in_layout[2] = get_unpadded_layout(op->getOperand(2), 1, 1);
    for (Layout &l : in_layout) {
      if (!l.has_value()) {
        op->emitOpError("unsupported operand shapes or layouts");
        return failure();
      }
    }
    setLayout(op, in_layout,
              VectorLayout(kNativeBitwidth, {0, 0}, default_tiling_,
                           ImplicitDim::kNone));
    return success();
  }

  bool allUsersRequireNativeTiling(Value x) {
    for (OpOperand &operand : x.getUses()) {
      if (isa<vector::ContractionOp, tpu::MatmulOp>(operand.getOwner())) {
        continue;
      }
      if (auto transpose = dyn_cast<vector::TransposeOp>(operand.getOwner())) {
        auto perm = transpose.getPermutation();
        auto rank = perm.size();
        // Only permutations that actually swap the last two dims need it.
        if (rank >= 2 && perm[rank - 1] == rank - 2 &&
            perm[rank - 2] == rank - 1) {
          continue;
        }
        // Fall through.
      }
      return false;
    }
    return true;
  }

  void setInLayout(Operation *op, ArrayRef<Layout> in) {
    CHECK_EQ(in.size(), op->getNumOperands()) << Print(op);
    SmallVector<Attribute, 4> in_attrs;
    in_attrs.reserve(in.size());
    for (const Layout &p : in) {
      in_attrs.push_back(VectorLayoutAttr::get(op->getContext(), p));
    }
    op->setAttr("in_layout", ArrayAttr::get(op->getContext(), in_attrs));
  }

  void setOutLayout(Operation *op, Layout out) {
    setOutLayout(op, ArrayRef<Layout>(out));
  }

  void setOutLayout(Operation *op, ArrayRef<Layout> out) {
    SmallVector<Attribute, 4> out_attrs;
    out_attrs.reserve(out.size());
    for (const Layout &p : out) {
      out_attrs.push_back(VectorLayoutAttr::get(op->getContext(), p));
    }
    op->setAttr("out_layout", ArrayAttr::get(op->getContext(), out_attrs));
  }

  void setLayout(Operation *op, Layout in, Layout out) {
    setLayout(op, ArrayRef<Layout>(in), ArrayRef<Layout>(out));
  }

  void setLayout(Operation *op, ArrayRef<Layout> in, Layout out) {
    setLayout(op, in, ArrayRef<Layout>(out));
  }

  void setLayout(Operation *op, Layout in, ArrayRef<Layout> out) {
    setLayout(op, ArrayRef<Layout>(in), out);
  }

  void setLayout(Operation *op, ArrayRef<Layout> in, ArrayRef<Layout> out) {
    setInLayout(op, in);
    setOutLayout(op, out);
  }

  SmallVector<Layout, 4> getInLayout(Operation *op) {
    CHECK(op);
    CHECK(op->getAttr("in_layout"));
    auto in_attrs = op->getAttrOfType<ArrayAttr>("in_layout").getValue();
    CHECK_EQ(in_attrs.size(), op->getNumOperands());
    SmallVector<Layout, 4> in_layouts;
    in_layouts.reserve(op->getNumOperands());
    for (int i = 0; i < op->getNumOperands(); ++i) {
      in_layouts.push_back(cast<VectorLayoutAttr>(in_attrs[i]).getLayout());
    }
    return in_layouts;
  }

  SmallVector<Layout, 4> getOutLayout(Operation *op) {
    CHECK(op);
    CHECK(op->getAttr("out_layout"));
    auto out_attrs = op->getAttrOfType<ArrayAttr>("out_layout").getValue();
    CHECK_EQ(out_attrs.size(), op->getNumResults());
    SmallVector<Layout, 4> out_layouts;
    out_layouts.reserve(op->getNumResults());
    for (int i = 0; i < op->getNumResults(); ++i) {
      out_layouts.push_back(cast<VectorLayoutAttr>(out_attrs[i]).getLayout());
    }
    return out_layouts;
  }

  Layout getLayout(Value v) {
    auto op = v.getDefiningOp();
    CHECK(op);
    auto op_result = dyn_cast<OpResult>(v);
    CHECK(op_result);
    auto result_index = op_result.getResultNumber();
    auto out_attrs = op->getAttrOfType<ArrayAttr>("out_layout").getValue();
    CHECK(out_attrs.size() > result_index);
    return cast<VectorLayoutAttr>(out_attrs[result_index]).getLayout();
  }

 private:
  std::optional<absl::Span<const int64_t>> verifyMemoryTiling(
      Operation *op, ArrayRef<xla::Tile> mem_tiling, int64_t rank,
      int8_t bitwidth) {
    if (bitwidth == 32) {
      if (mem_tiling.size() != 1) {
        op->emitOpError("Only one-level tiling supported for 32-bit loads");
        return std::nullopt;
      }
    } else if (bitwidth < 32) {
      int64_t rows_per_tile;
      if (rank == 1) {
        if (mem_tiling.size() != 3) {
          op->emitOpError(
              "Only three-level tiling supported for 1D memory ops narrower "
              "than 32-bit");
          return std::nullopt;
        }
        auto first = mem_tiling[0].dimensions();
        auto second = mem_tiling[1].dimensions();
        if (first.size() != 1 || first[0] % target_shape_[1] != 0) {
          op->emitOpError("Invalid first-level tile in 1D memory op");
          return std::nullopt;
        }
        rows_per_tile = first[0] / target_shape_[1];
        if (second.size() != 1 || second[0] != target_shape_[1]) {
          op->emitOpError("Invalid second-level tile in 1D memory op");
          return std::nullopt;
        }
      } else {
        if (mem_tiling.size() != 2) {
          op->emitOpError(
              "Only two-level tiling supported for 2D+ memory ops narrower "
              "than 32-bit");
          return std::nullopt;
        }
        auto first = mem_tiling[0].dimensions();
        rows_per_tile = first[0];
      }
      auto row_compressed = mem_tiling[mem_tiling.size() - 1].dimensions();
      if (row_compressed.size() != 2) {
        op->emitOpError("Expected 2D tiling for packed layout");
        return std::nullopt;
      }
      if (row_compressed[0] != (32 / bitwidth) || row_compressed[1] != 1) {
        op->emitOpError("Expected compressed packed layout");
        return std::nullopt;
      }
      if (row_compressed[0] > rows_per_tile) {
        op->emitOpError("Packing cannot introduce padding");
        return std::nullopt;
      }
    } else {
      op->emitOpError("Loads of types wider than 32-bit unsupported");
      return std::nullopt;
    }
    return mem_tiling[0].dimensions();
  }

  std::array<int64_t, 2> nativeTiling(int8_t bitwidth) {
    return {default_tiling_[0] * kNativeBitwidth / bitwidth,
            default_tiling_[1]};
  }

  std::array<int64_t, 2> target_shape_;
  std::array<int64_t, 2> default_tiling_;

  // Address alignment requirement, counted in 32-bit increments.
  static constexpr int64_t kVmemAlignment32 = 128;
  // TODO(apaszke): This is not really native on newer generations of TPUs.
  // Get rid of this temporary stopgap.
  static constexpr int8_t kNativeBitwidth = 32;
};

struct InferVectorLayoutPass
    : public impl::InferVectorLayoutPassBase<InferVectorLayoutPass> {
  InferVectorLayoutPass(int lane_count, int sublane_count) {
    this->sublane_count = sublane_count;
    this->lane_count = lane_count;
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    VectorLayoutInferer run({sublane_count, lane_count});
    if (run.infer(func).failed()) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInferVectorLayoutPass(
    int lane_count, int sublane_count) {
  return std::make_unique<InferVectorLayoutPass>(lane_count, sublane_count);
}

}  // namespace mlir::tpu
