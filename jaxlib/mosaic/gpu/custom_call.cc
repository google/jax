/* Copyright 2021 The JAX Authors.

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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "llvm/include/llvm/Support/TargetSelect.h"
#include "mlir/include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/include/mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/include/mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/include/mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/include/mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/include/mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/include/mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/include/mlir/Conversion/Passes.h"
#include "mlir/include/mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/include/mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/include/mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/include/mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/include/mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/include/mlir/ExecutionEngine/OptUtils.h"
#include "mlir/include/mlir/IR/AsmState.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/Parser/Parser.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/include/mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/include/mlir/Transforms/Passes.h"
#include "jaxlib/mosaic/gpu/launch_lowering.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"

namespace {

using MosaicInitFunc = void(void***);
using MosaicHostFunc = void(void**);

void InitContext(mlir::MLIRContext* context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                  mlir::gpu::GPUDialect, mlir::nvgpu::NVGPUDialect,
                  mlir::NVVM::NVVMDialect, mlir::LLVM::LLVMDialect>();
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);  // Arith needs this
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

absl::StatusOr<std::unique_ptr<mlir::ExecutionEngine>> Compile(
    mlir::ModuleOp module) {
  static bool register_once = []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerCanonicalizer();
    mlir::registerCSE();
    mlir::registerStripDebugInfo();
    mlir::registerConvertNVGPUToNVVMPass();
    mlir::registerConvertVectorToSCF();
    mlir::registerSCFToControlFlow();
    mlir::registerConvertNVVMToLLVMPass();
    mlir::registerArithToLLVMConversionPass();
    mlir::registerConvertIndexToLLVMPass();
    mlir::registerConvertGpuOpsToNVVMOps();
    mlir::registerConvertMathToLLVMPass();
    mlir::registerConvertFuncToLLVMPass();
    mlir::registerConvertAffineToStandard();
    mlir::registerReconcileUnrealizedCasts();
    mlir::registerGpuToLLVMConversionPass();
    // TODO(apaszke): Only register the passes we actually use.
    mlir::memref::registerMemRefPasses();
    mlir::registerGPUPasses();
    mosaic::gpu::registerGpuLaunchLoweringPass();
    return true;
  }();
  (void)register_once;
  auto pipeline = mlir::parsePassPipeline(
      R"(
      builtin.module(
        convert-nvgpu-to-nvvm,
        gpu-kernel-outlining{data-layout-str=},
        convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1},
        convert-scf-to-cf,
        convert-nvvm-to-llvm,
        expand-strided-metadata,
        nvvm-attach-target{O=3 chip=sm_90a fast=false features=+ptx80 ftz=false  module= triple=nvptx64-nvidia-cuda},
        lower-affine,
        convert-arith-to-llvm{index-bitwidth=0},
        convert-index-to-llvm{index-bitwidth=64},
        canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
        cse,
        gpu.module(strip-debuginfo),
        gpu.module(convert-gpu-to-nvvm{has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false}),
        gpu.module(canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}),
        gpu.module(cse),
        gpu.module(reconcile-unrealized-casts),
        gpu-to-llvm{gpu-binary-annotation=gpu.binary use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false},
        gpu-module-to-binary{format=binary},
        convert-math-to-llvm{approximate-log1p=true},
        canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
        cse,
        reconcile-unrealized-casts,
        gpu-launch-lowering,
        convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}
      )
  )");
  if (mlir::failed(pipeline)) {
    return absl::InternalError("Failed to construct pass pipeline");
  }
  mlir::PassManager pm(module.getContext());
  *static_cast<mlir::OpPassManager*>(&pm) = std::move(*pipeline);
  if (mlir::failed(pm.run(module))) {
    return absl::InternalError("Pass pipeline failed");
  }

  llvm::SmallVector<llvm::StringRef> runtime_lib;
  if (const char* lib_path = getenv("MOSAIC_GPU_RUNTIME_LIB_PATH")) {
    runtime_lib.emplace_back(lib_path);
  }
  // Create a transformer to run all LLVM optimization passes at the
  // specified optimization level.
  mlir::ExecutionEngineOptions options;
  options.transformer = mlir::makeOptimizingTransformer(3, 0, nullptr);
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  options.sharedLibPaths = runtime_lib;
  auto maybe_execution_engine = mlir::ExecutionEngine::create(module, options);
  if (!maybe_execution_engine) {
    return absl::InternalError("Failed to compile kernel");
  }
  return std::move(*maybe_execution_engine);
}

class CompiledKernel {
 public:
  CompiledKernel(std::unique_ptr<mlir::ExecutionEngine> engine, void* ctx,
                 MosaicHostFunc* host_launch)
      : engine_(std::move(engine)), ctx_(ctx), host_launch_(host_launch) {}

  std::pair<void*, MosaicHostFunc*> GetHostLaunch() {
    return std::make_pair(ctx_, host_launch_);
  }

 private:
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  void* ctx_;  // TODO(apaszke): Destroy this properly
  MosaicHostFunc* host_launch_;
};

std::pair<absl::flat_hash_map<uint64_t, CompiledKernel>*, absl::Mutex*>
GetKernelCache() {
  static absl::Mutex mutex;
  static auto& context_cache =
      *new absl::flat_hash_map<uint64_t, CompiledKernel>;
  return std::make_pair(&context_cache, &mutex);
}

// Each compiled kernel has a unique init func, and each kernel is used from
// a single HLO module. So it should be safe to not include the CUDA context
// in the key.
absl::StatusOr<std::pair<void*, MosaicHostFunc*>> CompileAndInit(
    uint64_t kernel_id, const char* module) {
  auto cache_and_mutex = GetKernelCache();
  auto* cache = cache_and_mutex.first;
  auto* mutex = cache_and_mutex.second;

  {
    // Fast path uses reader lock (as hash map look-up is relatively slow).
    absl::ReaderMutexLock lock(mutex);
    auto it = cache->find(kernel_id);
    if (ABSL_PREDICT_TRUE(it != cache->end()))
      return it->second.GetHostLaunch();
  }

  absl::MutexLock lock(mutex);
  // We released the reader lock, another thread might have initialized it.
  if (cache->find(kernel_id) == cache->end()) {
    mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
    InitContext(&context);
    mlir::ParserConfig parse_config(&context);
    auto module_op =
        mlir::parseSourceString<mlir::ModuleOp>(module, parse_config);
    if (!module_op) {
      return absl::InternalError("Failed to parse module");
    }
    auto maybe_engine = Compile(*module_op);
    if (!maybe_engine.ok()) {
      return maybe_engine.status();
    }
    mlir::ExecutionEngine* execution_engine = maybe_engine->get();
    auto main = execution_engine->lookupPacked("_mlir_ciface_main");
    auto init = execution_engine->lookupPacked("_mlir_ciface_main_init");
    if (!init || !main) {
      return absl::InternalError("Failed to retrieve kernel function");
    }
    void* ctx;
    void** ptr_to_ctx = &ctx;
    reinterpret_cast<MosaicInitFunc*>(*init)(&ptr_to_ctx);
    cache->insert_or_assign(
        kernel_id, CompiledKernel(std::move(*maybe_engine), ctx,
                                  reinterpret_cast<MosaicHostFunc*>(*main)));
  }
  return cache->at(kernel_id).GetHostLaunch();
}

void MosaicGPUCustomCall(void* stream, void** buffers, char* opaque,
                         size_t opaque_len, XlaCustomCallStatus* status) {
  uint64_t kernel_id = *reinterpret_cast<uint64_t*>(opaque);
  auto ctx_and_kernel = CompileAndInit(kernel_id, opaque + sizeof(uint64_t));
  if (!ctx_and_kernel.ok()) {
    XlaCustomCallStatusSetFailure(status,
                                  ctx_and_kernel.status().message().data(),
                                  ctx_and_kernel.status().message().size());
    return;
  }
  void* args[3] = {&ctx_and_kernel->first, &stream, &buffers};
  ctx_and_kernel->second(args);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("mosaic_gpu", &MosaicGPUCustomCall,
                                         "CUDA");

}  // namespace
