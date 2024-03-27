#ifndef _GPU_OPS_KERNELS_H_
#define _GPU_OPS_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gpu_ops {

enum ElementType { BF16, F16, F32, F64 };

struct RMSNormDescriptor {
  int n1;
  int n2;
  double eps;
  ElementType x_type;
  ElementType w_type;
  int part_grad_size;
};

void rms_forward_affine_mixed_dtypes(cudaStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len);
void rms_backward_affine(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len);
} // namespace gpu_ops

#endif
