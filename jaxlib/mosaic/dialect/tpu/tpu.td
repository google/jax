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

#ifndef TPU_ATTRS
#define TPU_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TPU_Dialect : Dialect {
  let name = "tpu";
  let cppNamespace = "::mlir::tpu";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 0;
}

class TPU_Attr<string name, string mnemonic_, list<Trait> traits = []>
    : AttrDef<TPU_Dialect, name, traits> {
  let mnemonic = mnemonic_;
}

def TPU_Vreg : Type<
  And<[IsVectorTypePred,
       Or<[
        And<[
          CPred<"llvm::cast<::mlir::VectorType>($_self).getShape() == ArrayRef<int64_t>{8, 128}">,
          CPred<"llvm::cast<::mlir::VectorType>($_self).getElementType().getIntOrFloatBitWidth() == 32">
        ]>,
        CPred<"::llvm::cast<::mlir::VectorType>($_self).getShape() == ArrayRef<int64_t>{"
          "8, 128, 32 / ::llvm::cast<::mlir::VectorType>($_self).getElementType().getIntOrFloatBitWidth()}">,
       ]>
      ]>,
  "native-sized vreg", "::mlir::VectorType">;

class TPU_Type<string name, string mnemonic_, list<Trait> traits = []>
    : TypeDef<TPU_Dialect, name, traits> {
  let mnemonic = mnemonic_;
}

def TPU_SemaphoreType : TPU_Type<"Semaphore", "semaphore", [MemRefElementTypeInterface]>;
def TPU_DMASemaphoreType : TPU_Type<"DMASemaphore", "dma_semaphore", [MemRefElementTypeInterface]>;
def TPU_SomeSemaphoreType : AnyTypeOf<[TPU_SemaphoreType, TPU_DMASemaphoreType]>;

def TPU_DimensionSemantics : I32EnumAttr<"DimensionSemantics", "Dimension semantics", [
  I32EnumAttrCase<"parallel", 0>,
  I32EnumAttrCase<"arbitrary", 1>
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_DimensionSemanticsEnum
    : EnumAttr<TPU_Dialect, TPU_DimensionSemantics, "dimension_semantics"> {
    let assemblyFormat = "`<` $value `>`";
}

// All indices/sizes are in element-space.
// Note that the implementation will require statically provable tile alignment.
def TPU_ElementWindowAttr : TPU_Attr<"ElementWindow", "element_window"> {
  // Including low padding, to avoid backwards-incompatible changes once we add it.
  let parameters = (ins
    ArrayRefParameter<"int64_t", "">:$pad_low,
    ArrayRefParameter<"int64_t", "">:$pad_high
  );
  let assemblyFormat = "`<` `[` $pad_low `]` `,` `[` $pad_high `]` `>`";
}

def TPU_ContractPrecision : I32EnumAttr<"ContractPrecision", "Contraction precision", [
  I32EnumAttrCase<"kBF16", 0, "bf16">,
  I32EnumAttrCase<"kFP32", 1, "fp32">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ContractPrecisionEnum
    : EnumAttr<TPU_Dialect, TPU_ContractPrecision, "contract_precision"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_TiledCase   : I32EnumAttrCase<"tiled", 0>;
def TPU_LaneCase    : I32EnumAttrCase<"lanes", 1>;
def TPU_SublaneCase : I32EnumAttrCase<"sublanes", 2>;
def TPU_VectorLayoutDim : I32EnumAttr<
  "VectorLayoutDim", "", [TPU_TiledCase, TPU_LaneCase, TPU_SublaneCase]>;

def TPU_VectorLayoutAttr : TPU_Attr<"VectorLayout", "vpad"> {
  let description = [{TODO}];

  let parameters = (ins "Layout":$layout);
  let hasCustomAssemblyFormat = 1;
}

def TPU_TiledLayoutAttr
  : TPU_Attr<"TiledLayout", "tiled",
             [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface>]> {
  let description = [{TODO}];
  let parameters = (ins
    ArrayRefParameter<"::xla::Tile", "">:$tiles,
    ArrayRefParameter<"int64_t", "">:$tile_strides
  );

  let hasCustomAssemblyFormat = 1;
}

def TPU_MemorySpace : I32EnumAttr<"MemorySpace", "Memory space", [
  I32EnumAttrCase<"kAny", 4294967295, "any">,
  // TODO(apaszke): Rename to kXYZ in C++
  I32EnumAttrCase<"vmem", 0, "vmem">,
  I32EnumAttrCase<"smem", 1, "smem">,
  I32EnumAttrCase<"kHbm", 2, "hbm">,
  I32EnumAttrCase<"kCmem", 3, "cmem">,
  I32EnumAttrCase<"kSemaphoreMem", 4, "semaphore_mem">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_MemorySpaceEnum
    : EnumAttr<TPU_Dialect, TPU_MemorySpace, "memory_space"> {
    let assemblyFormat = "`<` $value `>`";
}

class TPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits> {
}

def TPU_ReductionKind : I32EnumAttr<"ReductionKind", "Reduction kind", [
  I32EnumAttrCase<"SUM", 0, "sum">,
  I32EnumAttrCase<"MAX", 1, "max">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ReductionKindAttr
    : EnumAttr<TPU_Dialect, TPU_ReductionKind, "reduction_kind"> {
    let assemblyFormat = "`<` $value `>`";
}

def DebugAssertInsertionPass : Pass<"debug-assert-insertion", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createDebugAssertInsertionPass()";
}

def MosaicSerdePass : Pass<"mosaic-serde", "::mlir::ModuleOp"> {
  let options = [Option<"serialize", "serialize", "bool", "", "">];
}

def LogicalToPhysicalDeviceIdPass : Pass<"logical-to-physical-device-id", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createLogicalToPhysicalDeviceIdPass(-1)";
  let options = [Option<"total_devices", "total-devices", "int", "", "">];
}

def InferMemRefLayoutPass : Pass<"tpu-infer-memref-layout", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
  ];
  let constructor = "::mlir::tpu::createInferMemRefLayoutPass(-1)";
  let options = [
    // If hardware_generation is not set, the default value of -1 will crash on
    // runOnOperation.
    Option<"hardware_generation", "hardware-generation", "int", /*default=*/"-1", "">,
  ];
}

def InferVectorLayoutPass : Pass<"tpu-infer-vector-layout", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createInferVectorLayoutPass()";
  let options = [
    Option<"lane_count", "lane-count", "int", /*default=*/"128", "">,
    Option<"sublane_count", "sublane-count", "int", /*default=*/"8", "">,
  ];
}

def ApplyVectorLayoutPass : Pass<"tpu-apply-vector-layout", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createApplyVectorLayoutPass(-1)";
  let options = [
    // If hardware_generation is not set, the default value of -1 will crash on
    // runOnOperation.
    Option<"hardware_generation", "hardware-generation", "int", /*default=*/"-1", "">,
    Option<"lane_count", "lane-count", "int", /*default=*/"128", "">,
    Option<"sublane_count", "sublane-count", "int", /*default=*/"8", "">,
    Option<"mxu_contracting_size", "mxu-contracting-size", "int", /*default=*/"128", "">,
    Option<"mxu_noncontracting_size", "mxu-noncontracting-size", "int", /*default=*/"128", "">,
  ];
}

def LinalgVectorizationPass : Pass<"linalg-vectorization", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createLinalgVectorizationPass()";
}

#endif  // TPU_ATTRS
