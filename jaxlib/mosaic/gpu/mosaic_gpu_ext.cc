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

#include "nanobind/nanobind.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "llvm/include/llvm/Support/Error.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/service/custom_call_status.h"
namespace jax::cuda {
namespace {

namespace nb = nanobind;

void EventRecordCall(void* stream, void** buffers, char* opaque,
                     size_t opaque_len, XlaCustomCallStatus* status) {
  auto* event = reinterpret_cast<gpuEvent_t**>(opaque);
  if (gpuEventRecord(**event, reinterpret_cast<gpuStream_t>(stream)) !=
      gpuSuccess) {
    const char message[] = "Failed to record event";
    XlaCustomCallStatusSetFailure(status, message, sizeof(message));
  }
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("_gpu_event_create", []() {
    gpuEvent_t* event = new gpuEvent_t();
    gpuEventCreate(event, GPU_EVENT_DEFAULT);
    return reinterpret_cast<uintptr_t>(event);
  });
  m.def("_gpu_event_destroy", [](uintptr_t event) {
    gpuEventDestroy(*reinterpret_cast<gpuEvent_t*>(event));
  });
  m.def("_gpu_event_elapsed", [](uintptr_t start_event, uintptr_t end_event) {
    float elapsed_ms = -1;
    if (gpuEventElapsedTime(
            &elapsed_ms, *reinterpret_cast<gpuEvent_t*>(start_event),
            *reinterpret_cast<gpuEvent_t*>(end_event)) != gpuSuccess) {
      throw std::runtime_error("Failed to get elapsed time between events");
    }
    return elapsed_ms;
  });
  m.def("_record_event_capsule",
        []() { return EncapsulateFunction(EventRecordCall); });

  m.def("_enable_ptx_log", [] () {
#ifdef NDEBUG
    llvm::report_fatal_error("Built without NDEBUG, can't enable ptx logging.");
#else
    llvm::DebugFlag = true;
    llvm::setCurrentDebugType("serialize-to-llvm");
#endif
  });
}

}  // namespace
}  // namespace jax::cuda
