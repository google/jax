// Registers MLIR dialects used by JAX.
// This module is called by mlir/__init__.py during initialization.

#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/NVGPU.h"
#include "mlir-c/Dialect/NVVM.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/Vector.h"
#include "mlir-c/Transforms.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#define REGISTER_DIALECT(name) \
    MlirDialectHandle name##_dialect = mlirGetDialectHandle__##name##__(); \
    mlirDialectHandleInsertDialect(name##_dialect, registry)

PYBIND11_MODULE(register_jax_dialects, m) {
  m.doc() = "Registers upstream MLIR dialects used by JAX.";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    REGISTER_DIALECT(arith);
    REGISTER_DIALECT(func);
    REGISTER_DIALECT(math);
    REGISTER_DIALECT(memref);
    REGISTER_DIALECT(scf);
    REGISTER_DIALECT(vector);
    // For Mosaic GPU
    REGISTER_DIALECT(gpu);
    REGISTER_DIALECT(nvgpu);
    REGISTER_DIALECT(nvvm);
    REGISTER_DIALECT(llvm);
    mlirRegisterTransformsPasses();
    // Transforms used by JAX.
    mlirRegisterTransformsStripDebugInfo();
  });
}
