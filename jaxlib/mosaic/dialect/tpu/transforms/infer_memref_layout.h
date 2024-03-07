#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_MEMREF_LAYOUT_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_MEMREF_LAYOUT_H_

#include <string>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tpu {

FailureOr<MemRefType> inferMemref(MemRefType memref, int hardware_generation,
                                  int64_t leading_tile_rows = 0);

const std::string_view kLeadingTileRows = "leading_tile_rows";

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_MEMREF_LAYOUT_H_
