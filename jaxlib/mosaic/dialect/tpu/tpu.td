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

def TPU_AllReduceOp : TPU_Op<"all_reduce", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyVector:$input, I64Attr:$dim, TPU_ReductionKindAttr:$kind);
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def TPU_StoreOp : TPU_Op<"store", [AttrSizedOperandSegments]> {
  let arguments = (ins
    TPU_Vreg:$valueToStore,
    AnyType:$base,
    Variadic<Index>:$indices,
    DenseBoolArrayAttr:$sublane_mask,
    Optional<AnyType>:$mask,
    OptionalAttr<I32Attr>:$sublane_stride  // In sublane-sized units
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore (`masked` $mask^)? `sublanes` $sublane_mask  (`sublane_stride` $sublane_stride^)? attr-dict `:` type($base) `,` type($valueToStore) `,` type($mask)
  }];
}

def TPU_LoadOp : TPU_Op<"load"> {
  let arguments = (ins
    AnyType:$base,
    Variadic<Index>:$indices,
    DenseBoolArrayAttr:$sublane_mask,
    OptionalAttr<I32Attr>:$sublane_stride  // In sublane-sized units
  );
  let results = (outs TPU_Vreg:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` `sublanes` $sublane_mask (`sublane_stride` $sublane_stride^)? attr-dict `:` type($base) `,` type($result)
  }];
}

def TPU_StridedLoadOp : TPU_Op<"strided_load"> {
  let arguments = (ins
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseI32ArrayAttr:$strides
  );
  let results = (outs AnyVector:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` type($base) `,` type($result)
  }];
  let hasVerifier = 1;
}

def TPU_StridedStoreOp : TPU_Op<"strided_store"> {
  let arguments = (ins
    AnyVector:$valueToStore,
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseI32ArrayAttr:$strides
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore attr-dict `:` type($base) `,` type($valueToStore)
  }];
  let hasVerifier = 1;
}

def TPU_RotateOp : TPU_Op<"rotate", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins
    AnyVector:$value,
    SI32Attr:$amount,
    SI32Attr:$dimension,
    // When the stride is specified, the rotation amount for each index on the
    // stride dimension will be (amount + stride * index).
    OptionalAttr<SI32Attr>:$stride,
    OptionalAttr<SI32Attr>:$stride_dimension
  );
  let results = (outs AnyVector:$result);
  let assemblyFormat = [{
    $value `by` $amount `dim` $dimension (`stride` $stride `stride_dim` $stride_dimension^)? attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

def TPU_IotaOp : TPU_Op<"iota", [Pure]> {
  let arguments = (ins OptionalAttr<I32Attr>:$dimension);
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{ attr-dict `:` type($output) }];
}

def TPU_RepeatOp : TPU_Op<"repeat", [Pure]> {
  let arguments = (ins
    AnyVector:$source,
    I32Attr:$dimension,
    I32Attr:$times
  );
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{ $source `,` $dimension `x` $times attr-dict `:` type($source) `->` type($output) }];
}

def TPU_BroadcastInSublanesOp : TPU_Op<"broadcast_in_sublanes", [Pure]> {
  let arguments = (ins
    AnyVector:$source,  // All sublanes should be equal.
    I32Attr:$lane  // Coordinates of the first element to take.
  );
  // Output shape should be the same, except for position dim which contains
  // the newly inserted dimension.
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{
    $source `,` $lane attr-dict `:` type($source) `->` type($output)
  }];
}

// Integer unpacks are always signed at the moment.
def TPU_UnpackSubelementsOp : TPU_Op<"unpack_subelements", [Pure]> {
  let arguments = (ins
    AnyVector:$source,
    I32Attr:$index
  );
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{ $source `,` $index attr-dict `:` type($source) `->` type($output) }];
}

// Integer packs are always signed at the moment.
def TPU_PackSubelementsOp : TPU_Op<"pack_subelements", [Pure]> {
  let arguments = (ins Variadic<AnyVector>:$sources);
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{ $sources attr-dict `:` type($sources) `->` type($output) }];
}

def TPU_GatherOp : TPU_Op<"gather", [Pure]> {
  let arguments = (ins
    AnyVector:$source,
    DenseI32ArrayAttr:$indices,
    I32Attr:$dimension
  );
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{
    $source `[` $indices `]` `in` $dimension attr-dict
    `:` type($source) `->` type($output)
  }];
}

def TPU_DynamicGatherOp : TPU_Op<"dynamic_gather", [Pure]> {
  let arguments = (ins
    AnyVector:$source,
    AnyVector:$indices,  // If this is 2D, only the first row matters.
    I32Attr:$dimension
  );
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{
    $source `[` $indices `]` `in` $dimension attr-dict
    `:` type($source) `,` type($indices) `->` type($output)
  }];
}

// TODO(apaszke): Add a verifier for this op
// TODO(apaszke): Think hard about precision
def TPU_MatmulOp : TPU_Op<"matmul", [Pure]> {
  let arguments = (ins
    AnyVector:$lhs,
    AnyVector:$rhs,
    AnyVector:$acc,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs,
    OptionalAttr<TPU_ContractPrecisionEnum>:$precision
  );
  let results = (outs AnyVector:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)
  }];
  let hasCanonicalizer = 1;
}

def TPU_ConcatenateOp : TPU_Op<"concatenate", [Pure]> {
  let arguments = (ins
    Variadic<AnyVector>:$sources,
    I32Attr:$dimension
  );
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{
    $sources `in` $dimension attr-dict `:` type($sources) `->` type($output)
  }];
}

def TPU_BitcastOp : TPU_Op<"bitcast", [Pure]> {
  let arguments = (ins AnyVector:$input);
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
}

def TPU_BitcastVregOp : TPU_Op<"bitcast_vreg", [Pure]> {
  let arguments = (ins TPU_Vreg:$input);
  let results = (outs TPU_Vreg:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
}

def TPU_RollVectorsOp : TPU_Op<"roll_vectors", [Pure]> {
  let arguments = (ins Variadic<AnyVector>:$input);
  let results = (outs AnyVector:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def TPU_UnrollVectorsOp : TPU_Op<"unroll_vectors", [Pure]> {
  let arguments = (ins AnyVector:$input);
  let results = (outs Variadic<AnyVector>:$output);
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def TPU_CreateMaskOp : TPU_Op<"create_mask", [Pure, SameVariadicOperandSize]> {
  // high is exclusive
  let arguments = (ins Variadic<Index>:$low, Variadic<Index>:$high);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `[` $low `]``[` $high `]` attr-dict `:` type($output)
  }];
}

def TPU_CreateSubelementMaskOp : TPU_Op<"create_subelement_mask", [Pure]> {
  let arguments = (ins
    I32Attr:$from,  // inclusive
    I32Attr:$to,  // exclusive
    I32Attr:$num_subelems
  );
  let results = (outs AnyType:$output);  // Verify this is a vmsk with num_subelems
  let assemblyFormat = [{
    $from `,` $to `,` $num_subelems attr-dict `:` type($output)
  }];
}

def TPU_AssumeMultipleOp : TPU_Op<"assume_multiple", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins
    AnyTypeOf<[Index, AnyInteger]>:$value,
    I32Attr:$multiple
  );
  let results = (outs AnyTypeOf<[Index, AnyInteger]>:$result);
}

def TPU_MemRefSliceOp : TPU_Op<"memref_slice", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$mem_ref,
    Variadic<I32>:$base_idx,
    Variadic<I32>:$dynamic_sizes
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $mem_ref `[` $base_idx `]` (`<` $dynamic_sizes^ `>`)?
    attr-dict `:` type($mem_ref) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TPU_MemRefSqueezeOp : TPU_Op<"memref_squeeze", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TPU_ReinterpretCastOp : TPU_Op<"reinterpret_cast", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TPU_AssumeLayoutOp : TPU_Op<"assume_layout", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def TPU_EraseLayoutOp : TPU_Op<"erase_memref_layout", [Pure]> {
  let arguments = (ins AnyMemRef:$operand);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def TPU_DeviceIdOp : TPU_Op<"device_id", [Pure]> {
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def TPU_SemaphoreWaitOp : TPU_Op<"sem_wait"> {
  let arguments = (ins
    MemRefOf<[TPU_SemaphoreType]>:$semaphore,
    I32:$amount
  );
  let results = (outs);
  let assemblyFormat = [{ $semaphore `,` $amount attr-dict `:` type($semaphore)}];
}

def TPU_AllocaSemaphoreOp : TPU_Op<"sem_alloc"> {
  let arguments = (ins);
  let results = (outs MemRefOf<[TPU_SomeSemaphoreType]>:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def TPU_GetBarrierSemaphoreOp : TPU_Op<"sem_barrier"> {
  let arguments = (ins);
  let results = (outs MemRefOf<[TPU_SemaphoreType]>:$semaphore);
  let assemblyFormat = [{ attr-dict `:` type($semaphore) }];
  let hasVerifier = 1;
}

def TPU_SemaphoreSignalOp : TPU_Op<"sem_signal"> {
  let arguments = (ins
    MemRefOf<[TPU_SemaphoreType]>:$semaphore,
    I32:$amount,
    Optional<I32>:$device_id // For remote DMAs
  );
  let assemblyFormat = [{
    $semaphore `,` $amount (`,` $device_id^)? attr-dict `:` type($semaphore)
  }];
  let hasVerifier = 1;
}

def TPU_EnqueueDMAOp : TPU_Op<"enqueue_dma", [SameVariadicOperandSize]> {
  let arguments = (ins
    AnyMemRef:$source,
    Optional<MemRefOf<[TPU_DMASemaphoreType]>>:$source_semaphore, // For remote DMAs
    AnyMemRef:$target,
    MemRefOf<[TPU_DMASemaphoreType]>:$target_semaphore,
    Optional<I32>:$device_id // For remote DMAs
  );
  let hasVerifier = 1;
}

def TPU_WaitDMAOp : TPU_Op<"wait_dma"> {
  let arguments = (ins
    MemRefOf<[TPU_DMASemaphoreType]>:$semaphore,
    AnyMemRef:$ref
  );
  let hasVerifier = 1;
}

def TPU_RegionOp : TPU_Op<"region", [RecursiveMemoryEffects, SingleBlockImplicitTerminator<"tpu::YieldOp">]> {
  let arguments = (ins);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);
}

def TPU_TraceOp : TPU_Op<"trace", [RecursiveMemoryEffects, SingleBlockImplicitTerminator<"tpu::YieldOp">]> {
  let arguments = (ins StrAttr:$message, I32Attr:$level);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);
}

def TPU_TraceStartOp : TPU_Op<"trace_start", []> {
  let arguments = (ins StrAttr:$message, I32Attr:$level);
  let results = (outs);
}

def TPU_TraceStopOp : TPU_Op<"trace_stop", []> {
  let arguments = (ins);
  let results = (outs);
}

def TPU_YieldOp : TPU_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

// Expands the granularity of mask to subelements.
def TPU_MaskCastOp : TPU_Op<"mask_cast", [Pure]> {
  let arguments = (ins AnyVector:$input);
  let results = (outs AnyVector:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TPU_GetIterationBoundOp : TPU_Op<"iteration_bound"> {
  let arguments = (ins I32Attr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = [{ $dim attr-dict `:` type($result) }];
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
