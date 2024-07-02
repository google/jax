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

#include <sys/stat.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// clang-format: off
// pybind11 must be included before mlir/Bindings/Python/PybindAdaptors.h,
// otherwise this code will not build on Windows.
#include "pybind11/pybind11.h"
// clang-format: on

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "absl/log/check.h"
#include "jaxlib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"

// TODO(tlongeri): Can I add my own return type annotations to functions?
// TODO(tlongeri): I don't understand why MLIR uses the C API to implement
// Python bindings. Do we have a reason to do that?

namespace {
constexpr const char LAYOUT_DEFS_MODULE[] =
    "jax.jaxlib.mosaic.python.layout_defs";
constexpr const char IR_MODULE[] = "jaxlib.mlir.ir";
// TODO(tlongeri): Get rid of this somehow
constexpr MlirTpuI64TargetTuple TARGET_SHAPE{8, 128};

// TODO(tlongeri): Add type annotations from pybind11/typing.h once there is
// a release for it (and maybe add a custom Sequence<T> one as well).

// TODO(tlongeri): For our use-case, we don't really need C++ exceptions - just
// setting the exception object and returning NULL to Python should suffice, but
// not sure if this is possible with pybind.
class NotImplementedException : public std::runtime_error {
  using runtime_error::runtime_error;
};
}  // namespace

template <>
struct py::detail::type_caster<MlirTpuImplicitDim> {
  PYBIND11_TYPE_CASTER(MlirTpuImplicitDim, const_name("ImplicitDim | None"));

  bool load(handle src, bool) {
    if (src.is_none()) {
      value = MlirTpuImplicitDimNone;
      return true;
    }
    auto implicit_dim_cls =
        py::module_::import(LAYOUT_DEFS_MODULE).attr("ImplicitDim");
    if (!py::isinstance(src, implicit_dim_cls)) {
      return false;
    }
    if (src.is(implicit_dim_cls.attr("MINOR"))) {
      value = MlirTpuImplicitDimMinor;
    } else if (src.is(implicit_dim_cls.attr("SECOND_MINOR"))) {
      value = MlirTpuImplicitDimSecondMinor;
    } else {
      throw py::value_error();
    }
    return true;
  }

  static handle cast(MlirTpuImplicitDim implicit_dim,
                     return_value_policy /* policy */, handle /* parent */) {
    auto implicit_dim_cls =
        py::module_::import(LAYOUT_DEFS_MODULE).attr("ImplicitDim");
    switch (implicit_dim) {
      case MlirTpuImplicitDimNone:
        return py::none().release();
      case MlirTpuImplicitDimMinor:
        return static_cast<py::object>(implicit_dim_cls.attr("MINOR"))
            .release();
      case MlirTpuImplicitDimSecondMinor:
        return static_cast<py::object>(implicit_dim_cls.attr("SECOND_MINOR"))
            .release();
    }
  }
};

template <>
struct py::detail::type_caster<MlirTpuDirection> {
  PYBIND11_TYPE_CASTER(MlirTpuDirection, const_name("Direction"));

  bool load(handle src, bool) {
    auto direction_cls =
        py::module_::import(LAYOUT_DEFS_MODULE).attr("Direction");
    if (!py::isinstance(src, direction_cls)) {
      return false;
    }
    if (src.is(direction_cls.attr("LANES"))) {
      value = MlirTpuDirectionLanes;
    } else if (src.is(direction_cls.attr("SUBLANES"))) {
      value = MlirTpuDirectionSublanes;
    } else if (src.is(direction_cls.attr("SUBELEMENTS"))) {
      value = MlirTpuDirectionSubelements;
    } else {
      throw py::value_error();
    }
    return true;
  }

  static handle cast(MlirTpuDirection direction,
                     return_value_policy /* policy */, handle /* parent */) {
    auto direction_cls =
        py::module_::import(LAYOUT_DEFS_MODULE).attr("ImplicitDim");
    switch (direction) {
      case MlirTpuDirectionLanes:
        return static_cast<py::object>(direction_cls.attr("LANES")).release();
      case MlirTpuDirectionSublanes:
        return static_cast<py::object>(direction_cls.attr("SUBLANES"))
            .release();
      case MlirTpuDirectionSubelements:
        return static_cast<py::object>(direction_cls.attr("SUBELEMENTS"))
            .release();
      default:
        throw py::value_error();
    }
  }
};

namespace {
// Handler for use with MLIR C API print functions. The 2nd parameter is an
// opaque pointer to "user data" that should always be a string.
void printToString(MlirStringRef c_mlir_str, void* opaque_string) {
    std::string* str = static_cast<std::string*>(opaque_string);
    CHECK(str != nullptr);
    str->append(c_mlir_str.data, c_mlir_str.length);
}

class DiagnosticCapture {
 public:
  DiagnosticCapture(MlirContext ctx)
      : ctx_(ctx),
        id_(mlirContextAttachDiagnosticHandler(ctx, handleDiagnostic, this,
                                               nullptr)) {}

  ~DiagnosticCapture() { mlirContextDetachDiagnosticHandler(ctx_, id_); }

  void throwIfError() {
    if (error_messages_.size() == 1) {
      // Throw NotImplementedException if we got a single diagnostic that
      // contains "Not implemented".
      llvm::StringRef ref = error_messages_.front();
      constexpr llvm::StringRef not_implemented = "Not implemented";
      if (const size_t pos = ref.find(not_implemented);
          pos != llvm::StringRef::npos) {
        // We strip "Not implemented" only if it is a prefix. Sometimes it may
        // come after another prefix (e.g. op prefix), in which case we leave it
        if (pos == 0) {
          ref = ref.drop_front(not_implemented.size());
          ref.consume_front(": ");
        }
        throw NotImplementedException(ref.str());
      }
    }
    if (!error_messages_.empty()) {
      // Note that it is unusual/unexpected to get multiple diagnostics, so we
      // just forward all the error messages.
      throw std::runtime_error(llvm::join(error_messages_, "\n"));
    }
  }

 private:
  static MlirLogicalResult handleDiagnostic(MlirDiagnostic diag,
                                            void* opaque_detector) {
    DiagnosticCapture* detector =
        static_cast<DiagnosticCapture*>(opaque_detector);
    if (mlirDiagnosticGetSeverity(diag) == MlirDiagnosticError) {
      std::string& message = detector->error_messages_.emplace_back();
      mlirDiagnosticPrint(diag, printToString, &message);
    }
    return mlirLogicalResultFailure();  // Propagate to other handlers
  }
  llvm::SmallVector<std::string, 1> error_messages_;
  const MlirContext ctx_;
  const MlirDiagnosticHandlerID id_;
};

template <typename T>
class Holder {};

// Holder class for MlirTpuVectorLayout, to deal properly with destruction.
// TODO(tlongeri): It would be nice to not have a seemingly unnecessary
// "pointer-to-pointer" (MlirTpuVectorLayout is basically an opaque pointer).
// But I'm not sure if that's possible since pybind expects get() to return a
// true pointer type.
template <>
class Holder<MlirTpuVectorLayout> {
 public:
  Holder(MlirTpuVectorLayout layout) : ptr(new MlirTpuVectorLayout(layout)) {}
  Holder(MlirTpuVectorLayout* layout) : ptr(layout) {}
  Holder(Holder<MlirTpuVectorLayout>&& other) = default;
  ~Holder() { mlirTpuVectorLayoutDestroy(*ptr); }
  MlirTpuVectorLayout* get() { return ptr.get(); }

 private:
  std::unique_ptr<MlirTpuVectorLayout> ptr;
};
}  // namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, Holder<T>);

namespace {
py::object toPyLayoutOffset(int64_t offset) {
  CHECK_GE(offset, -1);
  if (offset == -1) {
    return py::module_::import(LAYOUT_DEFS_MODULE).attr("REPLICATED");
  } else {
    return py::int_(offset);
  }
}

// TODO(tlongeri): Would `type_caster`s let me avoid defining all of these
// to/from functions?
int64_t offsetFromPyOffset(py::object py_offset) {
  if (py::isinstance<py::int_>(py_offset)) {
    int64_t offset = py::cast<py::int_>(py_offset);
    if (offset < 0) {
      throw py::value_error("Invalid py layout offset");
    }
    return offset;
  } else if (py_offset.equal(
                 py::module_::import(LAYOUT_DEFS_MODULE).attr("REPLICATED"))) {
    return -1;
  } else {
    throw py::type_error("Invalid layout offset type");
  }
}

template <typename T>
llvm::SmallVector<T> sequenceToSmallVector(py::sequence seq) {
  return llvm::map_to_vector(
      seq, [](py::handle handle) { return py::cast<T>(handle); });
}

py::tuple toPyTuple(const int64_t* data, size_t count) {
  py::tuple tuple(count);
  for (size_t i = 0; i < count; ++i) {
    tuple[i] = data[i];
  }
  return tuple;
}

py::tuple toPyTuple(MlirTpuI64TargetTuple tuple) {
  return py::make_tuple(tuple.sublane, tuple.lane);
}

// Unwraps the current default insertion point
// ValueError is raised if default insertion point is not set
MlirTpuInsertionPoint getDefaultInsertionPoint() {
  py::object insertion_point =
      py::module_::import(IR_MODULE).attr("InsertionPoint").attr("current");
  py::object ref_operation = insertion_point.attr("ref_operation");
  return {py::cast<MlirBlock>(insertion_point.attr("block")),
          ref_operation.is_none()
              ? MlirOperation{nullptr}
              : py::cast<MlirOperation>(insertion_point.attr("ref_operation"))};
}

// Unwraps the current default location
// ValueError is raised if default location is not set
MlirLocation getDefaultLocation() {
  return py::cast<MlirLocation>(
      py::module_::import(IR_MODULE).attr("Location").attr("current"));
}

// Unwraps the current default MLIR context
// ValueError is raised if default context is not set
MlirContext getDefaultContext() {
  return py::cast<MlirContext>(
      py::module_::import(IR_MODULE).attr("Context").attr("current"));
}

}  // namespace

PYBIND11_MODULE(_tpu_ext, m) {
  mlirRegisterTPUPasses();  // Register all passes on load.

  py::class_<MlirTpuVregDataBounds>(m, "VRegDataBounds", py::module_local())
      .def("mask_varies_along",
           [](MlirTpuVregDataBounds self, MlirTpuDirection direction) {
             return mlirTpuVregDataBoundsMaskVariesAlong(self, direction,
                                                         TARGET_SHAPE);
           })
      .def_property_readonly("complete",
                             [](MlirTpuVregDataBounds self) {
                               return mlirTpuVregDataBoundsIsComplete(
                                   self, TARGET_SHAPE);
                             })
      .def("get_vector_mask",
           [](MlirTpuVregDataBounds self, int generation) {
             // TODO: Does this work? Test in Python
             MlirValue mask = mlirTpuVregDataBoundsGetVectorMask(
                 self, getDefaultInsertionPoint(), getDefaultLocation(),
                 generation, TARGET_SHAPE);
             if (mask.ptr == nullptr) {
               throw std::runtime_error("getVectorMask failed");
             }
             return mask;
           })
      .def("get_sublane_mask", [](MlirTpuVregDataBounds self) {
        return mlirTpuVregDataBoundsGetSublaneMask(self, getDefaultContext(),
                                                   TARGET_SHAPE);
      });

  // TODO(tlongeri): More precise argument type annotations. There currently
  // seems to be no way to define your own?
  py::class_<MlirTpuVectorLayout, Holder<MlirTpuVectorLayout>>(
      m, "VectorLayout", py::module_local())
      .def(py::init([](int bitwidth, py::tuple offsets, py::tuple tiling,
                       MlirTpuImplicitDim implicit_dim) {
             if (offsets.size() != 2) {
               throw py::value_error("offsets should be of length 2");
             }
             return mlirTpuVectorLayoutCreate(
                 bitwidth,
                 {offsetFromPyOffset(offsets[0]),
                  offsetFromPyOffset(offsets[1])},
                 {tiling[0].cast<int64_t>(), tiling[1].cast<int64_t>()},
                 implicit_dim);
           }),
           py::arg("bitwidth"), py::arg("offsets"), py::arg("tiling"),
           py::arg("implicit_dim"))
      .def_property_readonly("bitwidth", mlirTpuVectorLayoutGetBitwidth,
                             "The bitwidth of the stored values.")
      .def_property_readonly(
          "offsets",
          [](MlirTpuVectorLayout self) {
            MlirTpuLayoutOffsets offsets = mlirTpuVectorLayoutGetOffsets(self);
            return py::make_tuple(toPyLayoutOffset(offsets.sublane),
                                  toPyLayoutOffset(offsets.lane));
          },
          "The coordinates of the first valid element. If an offset is "
          "REPLICATED, then any offset is valid as the value does not vary "
          "across sublanes or lanes respectively.")
      .def_property_readonly(
          "tiling",
          [](MlirTpuVectorLayout self) {
            return toPyTuple(mlirTpuVectorLayoutGetTiling(self));
          },
          "The tiling used to lay out values (see the XLA docs). For values of "
          "bitwidth < 32, an implicit (32 // bitwidth, 1) tiling is appended "
          "to the one specified as an attribute.")
      .def_property_readonly(
          "implicit_dim", mlirTpuVectorLayoutGetImplicitDim,
          "If specified, the value has an implicit dim inserted in either "
          "minormost or second minormost position.")
      .def_property_readonly(
          "packing", mlirTpuVectorLayoutGetPacking,
          "Returns the number of values stored in a vreg entry.")
      .def_property_readonly(
          "layout_rank", mlirTpuVectorLayoutGetLayoutRank,
          "The number of minormost dimensions tiled by this layout.")
      .def_property_readonly(
          "has_natural_topology",
          [](MlirTpuVectorLayout self) {
            return mlirTpuVectorLayoutHasNaturalTopology(self, TARGET_SHAPE);
          },
          "True, if every vector register has a layout without jumps.\n"
          "\n"
          "By without jumps we mean that traversing vregs over (sub)lanes "
          "always leads to a contiguous traversal of the (second) minormost "
          "dimension of data. This is only true for 32-bit types, since "
          "narrower types use two level tiling.")
      .def_property_readonly(
          "has_native_tiling",
          [](MlirTpuVectorLayout self) {
            return mlirTpuVectorLayoutHasNativeTiling(self, TARGET_SHAPE);
          },
          "True, if every vector register has a natural \"packed\" topology.\n"
          "\n"
          "This is equivalent to has_natural_topology for 32-bit types, but "
          "generalizes it to narrower values with packed layouts too.")
      .def_property_readonly(
          "tiles_per_vreg",
          [](MlirTpuVectorLayout self) {
            return mlirTpuVectorLayoutTilesPerVreg(self, TARGET_SHAPE);
          },
          "How many tiles fit in each vector register.")
      .def_property_readonly(
          "sublanes_per_tile",
          [](MlirTpuVectorLayout self) {
            return mlirTpuVectorLayoutSublanesPerTile(self, TARGET_SHAPE);
          },
          "The number of sublanes necessary to store each tile.")
      .def_property_readonly(
          "vreg_slice",
          [](MlirTpuVectorLayout self) {
            MlirTpuI64TargetTuple vreg_slice =
                mlirTpuVectorLayoutVregSlice(self, TARGET_SHAPE);
            return py::module_::import(LAYOUT_DEFS_MODULE)
                .attr("TargetTuple")(vreg_slice.sublane, vreg_slice.lane);
          },
          "Returns the size of a window contained in a single vreg.\n"
          "\n"
          "We never reuse the same vector register to store data of multiple "
          "rows, so only the minormost dimension can increase.")
      .def(
          "implicit_shape",
          [](MlirTpuVectorLayout self, py::sequence shape) {
            llvm::SmallVector<int64_t> implicit_shape_vec =
                sequenceToSmallVector<int64_t>(shape);
            MlirTpuI64ArrayRef implicit_shape =
                mlirTpuVectorLayoutImplicitShape(
                    self,
                    {implicit_shape_vec.data(), implicit_shape_vec.size()});
            py::tuple ret = toPyTuple(implicit_shape.ptr, implicit_shape.size);
            free(implicit_shape.ptr);
            return ret;
          },
          py::arg("shape"))
      .def(
          "tile_array_shape",
          [](MlirTpuVectorLayout self, py::sequence shape) {
            llvm::SmallVector<int64_t> tile_array_shape_vec =
                sequenceToSmallVector<int64_t>(shape);
            MlirTpuI64ArrayRef tile_array_shape =
                mlirTpuVectorLayoutTileArrayShape(
                    self,
                    {tile_array_shape_vec.data(), tile_array_shape_vec.size()},
                    TARGET_SHAPE);
            py::tuple ret =
                toPyTuple(tile_array_shape.ptr, tile_array_shape.size);
            free(tile_array_shape.ptr);
            return ret;
          },
          py::arg("shape"),
          "Returns the shape of an ndarray of vregs needed to represent a "
          "value.\n"
          "\n"
          "All but the last two dimensions are unrolled over vregs. In the "
          "last two dims we need as many vregs as indicated by dividing the "
          "point at which the value ends (given by the start offset plus the "
          "dim size) divided by the respective vreg capacity in that dim (and "
          "a ceiling if non-integral). If a value is replicated, then any "
          "offset is valid and we pick 0 to minimize the number of vregs.\n"
          "\n"
          "Args:\n"
          "  shape: The shape of the ndarray to tile.")
      .def(
          "tile_data_bounds",
          [](MlirTpuVectorLayout self, py::sequence shape, py::sequence ixs,
             std::variant<bool, py::tuple> allow_replicated) {
            llvm::SmallVector<int64_t> shape_vec =
                sequenceToSmallVector<int64_t>(shape);
            llvm::SmallVector<int64_t> ixs_vec =
                sequenceToSmallVector<int64_t>(ixs);
            if (shape_vec.size() != ixs_vec.size()) {
              throw py::value_error(
                  "Expected shape and ixs to have the same size");
            }
            return std::visit(
                [&](auto ar) {
                  if constexpr (std::is_same_v<decltype(ar), bool>) {
                    return mlirTpuVectorLayoutTileDataBounds(
                        self, getDefaultContext(), shape_vec.data(),
                        ixs_vec.data(), shape_vec.size(), TARGET_SHAPE,
                        {ar, ar});
                  } else {
                    return mlirTpuVectorLayoutTileDataBounds(
                        self, getDefaultContext(), shape_vec.data(),
                        ixs_vec.data(), shape_vec.size(), TARGET_SHAPE,
                        {ar[0].template cast<bool>(),
                         ar[1].template cast<bool>()});
                  }
                },
                allow_replicated);
          },
          py::arg("shape"), py::arg("ixs"), py::arg("allow_replicated") = false,
          "Returns the bounds of the given tile that hold useful data.\n"
          "\n"
          "Arguments:\n"
          "  full_shape: The shape of the full vector this layout applies to.\n"
          "  ixs: The indices into an array of tiles representing the full "
          "vector (see tile_array_shape for bounds) selecting the tile for "
          "which the bounds are queried.\n"
          "  allow_replicated: If False, no offset is allowed to be "
          "REPLICATED. If True, offsets are allowed to be REPLICATED, but the "
          "bounds will span the full dimension of the tile (i.e. potentially "
          "multiple repeats of the actual data).\n"
          "\n"
          "Returns:\n"
          "  A TargetTuple of slices, indicating the span of useful data "
          "within the tile selected by idx.")
      .def(
          "generalizes",
          [](MlirTpuVectorLayout self, MlirTpuVectorLayout other,
             std::optional<py::sequence> shape) {
            if (shape) {
              llvm::SmallVector<int64_t> shape_vec =
                  sequenceToSmallVector<int64_t>(*shape);
              return mlirTpuVectorLayoutGeneralizes(
                  self, other, {shape_vec.data(), shape_vec.size()},
                  TARGET_SHAPE);
            }
            return mlirTpuVectorLayoutGeneralizes(self, other, {nullptr, 0},
                                                  TARGET_SHAPE);
          },
          py::arg("other"), py::arg("shape") = std::nullopt,
          "Returns True if the other layout is a special case of this one.\n"
          "\n"
          "In here, other is considered \"a special case\" when the set of "
          "vector register entries that represent a value in that layout is "
          "also the set of entries in which self stores the value. This is of "
          "course true for layouts that are equivalent, but it does not need "
          "to hold both ways. For example, a layout that implies the value "
          "does not change along an axis of the vector register is more "
          "general than the layout that picks a fixed starting point for the "
          "value and does not encode that assumption.\n"
          "\n"
          "The generalization relation is a non-strict partial order. You can "
          "think of it as a partial <= on vector layouts, but we don't "
          "overload Python operators since there's no clear way to decide "
          "where the bottom and top should be.\n"
          "\n"
          "Args:\n"
          "  other: The layout compared against self.\n"
          "  shape: An optional shape of the vector to which both layouts "
          "apply.\n"
          "    The generalization relation is larger than usual for some "
          "shapes. That is, if self.generalizes(other) then also "
          "self.generalizes(other, shape) for any shape, but that implication "
          "does not hold the other way around for some shapes.")
      .def(
          "equivalent_to",
          [](MlirTpuVectorLayout self, MlirTpuVectorLayout other,
             std::optional<py::sequence> shape) {
            if (shape) {
              llvm::SmallVector<int64_t> shape_vec =
                  sequenceToSmallVector<int64_t>(*shape);
              return mlirTpuVectorLayoutEquivalentTo(
                  self, other, {shape_vec.data(), shape_vec.size()},
                  TARGET_SHAPE);
            }
            return mlirTpuVectorLayoutEquivalentTo(self, other, {nullptr, 0},
                                                   TARGET_SHAPE);
          },
          py::arg("other"), py::arg("shape") = std::nullopt,
          "Returns True if the two layouts are equivalent.\n"
          "\n"
          "That is, when all potential vector entries where the value can be "
          "stored (there might be multiple choices for some layouts!) are "
          "equal in both self and other.\n"
          "\n"
          "Args:\n"
          "  other: The layout compared against self.\n"
          "  shape: An optional shape of the vector to which both layouts "
          "apply. More layouts are considered equivalent when the shape is "
          "specified. Also see the docstring of the generalizes method.")
      .def("__eq__", mlirTpuVectorLayoutEquals)
      .def("__repr__",
           [](MlirTpuVectorLayout self) {
             std::string str;
             mlirTpuVectorLayoutPrint(self, printToString, &str);
             return str;
           });

  // TODO(tlongeri): Can we make the first parameter a VectorType?
  m.def("assemble",
        [](const MlirType ty, MlirTpuVectorLayout layout,
           // TODO(tlongeri): Remove py::array::c_style, I only added it because
           // I couldn't find a simple way to iterate over array data, but it
           // causes yet another unnecessary copy.
           py::array_t<PyObject*, py::array::c_style> np_arr) -> MlirOperation {
          if (!mlirTypeIsAVector(ty)) {
            throw py::type_error("Expected vector type");
          }
          llvm::SmallVector<MlirValue> vals(np_arr.size());
          for (int64_t i = 0; i < np_arr.size(); ++i) {
            vals.data()[i] = py::cast<MlirValue>(py::handle(np_arr.data()[i]));
          }
          llvm::SmallVector<int64_t> shape(np_arr.ndim());
          for (int64_t i = 0; i < np_arr.ndim(); ++i) {
            shape.data()[i] = np_arr.shape()[i];
          }
          return mlirTpuAssemble(
              getDefaultInsertionPoint(), ty, layout,
              MlirTpuValueArray{MlirTpuI64ArrayRef{shape.data(), shape.size()},
                                vals.data()},
              TARGET_SHAPE);
        });
  m.def("disassemble", [](MlirTpuVectorLayout layout, MlirValue val) {
    DiagnosticCapture diag_capture(getDefaultContext());
    MlirTpuValueArray val_arr = mlirTpuDisassemble(getDefaultInsertionPoint(),
                                                   layout, val, TARGET_SHAPE);
    if (val_arr.vals == nullptr) {
      diag_capture.throwIfError();
      throw py::value_error("Failed to disassemble");
    }
    py::array_t<PyObject*> np_vals(
        llvm::ArrayRef<int64_t>{val_arr.shape.ptr, val_arr.shape.size});
    for (ssize_t i = 0; i < np_vals.size(); ++i) {
      np_vals.mutable_data()[i] = py::cast(val_arr.vals[i]).release().ptr();
    }
    free(val_arr.shape.ptr);
    free(val_arr.vals);
    return np_vals;
  });
  m.def("apply_layout_op",
        [](int hardware_generation, const MlirOperation c_op) {
          DiagnosticCapture diag_capture(getDefaultContext());
          MlirLogicalResult res =
              mlirTpuApplyLayoutOp(hardware_generation, c_op, TARGET_SHAPE);
          if (mlirLogicalResultIsFailure(res)) {
            diag_capture.throwIfError();
            throw std::runtime_error("applyLayoutOp failed");
          }
        });
  m.def("relayout",
        [](MlirValue v, MlirTpuVectorLayout src, MlirTpuVectorLayout dst) {
          DiagnosticCapture diag_capture(getDefaultContext());
          MlirValue new_v = mlirTpuRelayout(getDefaultInsertionPoint(), v, src,
                                            dst, TARGET_SHAPE);
          if (new_v.ptr == nullptr) {
            diag_capture.throwIfError();
            throw py::value_error("Failed to relayout");
          }
          return new_v;
        });
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const NotImplementedException& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    }
  });

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__tpu__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def("private_is_tiled_layout", [](MlirAttribute attr) {
    return mlirTPUAttributeIsATiledLayoutAttr(attr);
  });
  m.def("private_get_tiles", [](MlirAttribute attr) -> py::object {
    MlirAttribute encoded_tiles = mlirTPUTiledLayoutAttrGetTiles(attr);
    py::tuple py_tiles(mlirArrayAttrGetNumElements(encoded_tiles));
    for (intptr_t i = 0; i < mlirArrayAttrGetNumElements(encoded_tiles); ++i) {
      MlirAttribute tile = mlirArrayAttrGetElement(encoded_tiles, i);
      py::tuple py_tile(mlirDenseArrayGetNumElements(tile));
      for (intptr_t j = 0; j < mlirDenseArrayGetNumElements(tile); ++j) {
        py_tile[j] = mlirDenseI64ArrayGetElement(tile, j);
      }
      py_tiles[i] = py_tile;
    }
    return py_tiles;
  });
  m.def("private_has_communication", [](MlirOperation op) {
    bool has_communication;
    bool has_custom_barrier;
    mlirTPUAnalyzePotentialCommunication(op, &has_communication,
                                         &has_custom_barrier);
    return std::make_pair(has_communication, has_custom_barrier);
  });

  // TODO(apaszke): All of those should be upstreamed to MLIR Python bindings.
  m.def("private_replace_all_uses_with", [](MlirOperation op,
                                            std::vector<MlirValue> vals) {
    if (vals.size() != mlirOperationGetNumResults(op)) {
      throw py::value_error("length mismatch in replace_all_uses_with");
    }
    for (int i = 0; i < vals.size(); ++i) {
      mlirValueReplaceAllUsesOfWith(mlirOperationGetResult(op, i), vals[i]);
    }
  });
  m.def("private_replace_all_uses_except",
        [](MlirValue old, MlirValue new_val, MlirOperation except) {
          for (intptr_t i = 0; i < mlirOperationGetNumOperands(except); ++i) {
            if (mlirValueEqual(mlirOperationGetOperand(except, i), new_val)) {
              throw py::value_error("new val already used in except");
            }
          }
          mlirValueReplaceAllUsesOfWith(old, new_val);
          // Undo the replacement in the except op.
          for (intptr_t i = 0; i < mlirOperationGetNumOperands(except); ++i) {
            if (mlirValueEqual(mlirOperationGetOperand(except, i), new_val)) {
              mlirOperationSetOperand(except, i, old);
            }
          }
        });
  m.def("private_set_operand",
        [](MlirOperation op, int idx, MlirValue new_operand) {
          mlirOperationSetOperand(op, idx, new_operand);
        });
  m.def("private_set_operands", [](MlirOperation op,
                                   std::vector<MlirValue> new_operands) {
    mlirOperationSetOperands(op, new_operands.size(), new_operands.data());
  });
  m.def("private_has_no_memory_space", [](MlirType ty) {
    return mlirAttributeIsNull(mlirMemRefTypeGetMemorySpace(ty));
  });
  m.def("private_is_identity", [](MlirAttribute attr) {
    return mlirAffineMapIsIdentity(mlirAffineMapAttrGetValue(attr));
  });
  m.def("private_insert_argument",
        [](int index, MlirBlock block, MlirType type) -> MlirValue {
          return mlirBlockInsertArgument(
              block, index, type,
              mlirLocationUnknownGet(mlirTypeGetContext(type)));
        });
  m.def("private_set_arg_attr",
        [](MlirOperation op, unsigned i, std::string name, MlirAttribute attr) {
          mlirFuncSetArgAttr(
              op, i, mlirStringRefCreateFromCString(name.c_str()), attr);
        });
  m.def("private_move_all_regions", [](MlirOperation src, MlirOperation dst) {
    if (mlirOperationGetNumRegions(src) != mlirOperationGetNumRegions(dst)) {
      throw py::value_error(
          "Region counts do not match in src operation and dst operations");
    }
    for (intptr_t i = 0; i < mlirOperationGetNumRegions(src); ++i) {
      mlirRegionTakeBody(mlirOperationGetRegion(dst, i),
                         mlirOperationGetRegion(src, i));
    }
  });
}
