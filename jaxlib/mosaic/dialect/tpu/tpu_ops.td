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

#ifndef TPU_OPS_ATTRS
#define TPU_OPS_ATTRS

// TODO(tlongeri): Fix includes
include "jaxlib/mosaic/dialect/tpu/tpu.td"

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

def TPU_MemRefSliceOp : TPU_Op<"memref_slice", [Pure]> {
  let arguments = (ins
    AnyMemRef:$mem_ref,
    Variadic<I32>:$base_idx
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $mem_ref `[` $base_idx `]` attr-dict `:` type($mem_ref) `->` type($result)
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

#endif  // TPU_OPS_ATTRS
