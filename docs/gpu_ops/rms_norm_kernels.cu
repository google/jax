/* Copyright 2024 The JAX Authors.

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

#include "kernel_helpers.h"
#include "kernels.h"
#include "stdio.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>

namespace {

#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT,     \
                                                          NAME, ...)           \
  switch (TYPEIN) {                                                            \
  case gpu_ops::ElementType::F64: {                                            \
    using scalar_t_in = double;                                                \
    using accscalar_t = double;                                                \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F64: {                                          \
      using scalar_t_out = double;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using scalar_t_out = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using scalar_t_out = __nv_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::F32: {                                            \
    using scalar_t_in = float;                                                 \
    using accscalar_t = float;                                                 \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F64: {                                          \
      using scalar_t_out = double;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using scalar_t_out = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using scalar_t_out = __nv_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::F16: {                                            \
    using scalar_t_in = __half;                                                \
    using accscalar_t = float;                                                 \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F64: {                                          \
      using scalar_t_out = double;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using scalar_t_out = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using scalar_t_out = __nv_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::BF16: {                                           \
    using scalar_t_in = __nv_bfloat16;                                         \
    using accscalar_t = float;                                                 \
    switch (TYPEOUT) {                                                         \
    case gpu_ops::ElementType::F64: {                                          \
      using scalar_t_out = double;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F32: {                                          \
      using scalar_t_out = float;                                              \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::F16: {                                          \
      using scalar_t_out = __half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case gpu_ops::ElementType::BF16: {                                         \
      using scalar_t_out = __nv_bfloat16;                                      \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    break;                                                                     \
  }

template <typename U>
__device__ void cuWelfordOnlineSum(const U curr, U &mu, U &sigma2, U &count) {
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template <typename U>
__device__ void cuChanOnlineSum(const U muB, const U sigma2B, const U countB,
                                U &mu, U &sigma2, U &count) {
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template <typename U> __device__ void cuRMSOnlineSum(const U curr, U &sigma2) {
  sigma2 = sigma2 + curr * curr;
}

template <typename U>
__device__ void cuChanRMSOnlineSum(const U sigma2B, U &sigma2) {
  sigma2 = sigma2 + sigma2B;
}

template <typename T, typename U>
__device__ void cuWelfordMuSigma2(const T *__restrict__ vals, const int n1,
                                  const int n2, const int i1, U &mu, U &sigma2,
                                  U *buf, bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T *lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        if (!rms_only) {
          cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
        } else {
          cuRMSOnlineSum<U>(curr, sigma2);
        }
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
      } else {
        cuRMSOnlineSum<U>(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      U sigma2B = __shfl_sync(0xffffffff, sigma2, srcLaneB, warpSize);
      if (!rms_only) {
        U muB = __shfl_sync(0xffffffff, mu, srcLaneB, warpSize);
        U countB = __shfl_sync(0xffffffff, count, srcLaneB, warpSize);
        cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
      } else {
        cuChanRMSOnlineSum<U>(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U *ubuf = (U *)buf;
      U *ibuf = (U *)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          if (!rms_only) {
            ubuf[2 * wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
          ubuf[2 * wrt_y + 1] = sigma2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          if (!rms_only) {
            U muB = ubuf[2 * threadIdx.y];
            U countB = ibuf[threadIdx.y];
            cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
          } else {
            cuChanRMSOnlineSum<U>(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = __shfl_sync(0xffffffff, mu, 0, warpSize);
      }
      sigma2 = __shfl_sync(0xffffffff, sigma2 / U(n2), 0, warpSize);
    }
  }
}

template <>
__device__ void cuWelfordMuSigma2(const __half *__restrict__ vals, const int n1,
                                  const int n2, const int i1, float &mu,
                                  float &sigma2, float *buf, bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu = float(0);
  sigma2 = float(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const __half *lvals = vals + i1 * n2;
    int l = 8 * thrx;
    if ((((size_t)lvals) & 3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        if (!rms_only) {
          cuWelfordOnlineSum(curr, mu, sigma2, count);
        } else {
          cuRMSOnlineSum(curr, sigma2);
        }
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (; l + 7 < n2; l += 8 * numx) {
      for (int k = 0; k < 8; k += 2) {
        float2 curr = __half22float2(*((__half2 *)(lvals + l + k)));
        if (!rms_only) {
          cuWelfordOnlineSum(curr.x, mu, sigma2, count);
          cuWelfordOnlineSum(curr.y, mu, sigma2, count);
        } else {
          cuRMSOnlineSum(curr.x, sigma2);
          cuRMSOnlineSum(curr.y, sigma2);
        }
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      if (!rms_only) {
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      } else {
        cuRMSOnlineSum(curr, sigma2);
      }
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      float sigma2B = __shfl_sync(0xffffffff, sigma2, srcLaneB, warpSize);
      if (!rms_only) {
        float muB = __shfl_sync(0xffffffff, mu, srcLaneB, warpSize);
        float countB = __shfl_sync(0xffffffff, count, srcLaneB, warpSize);
        cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
      } else {
        cuChanRMSOnlineSum(sigma2B, sigma2);
      }
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float *ubuf = (float *)buf;
      float *ibuf = (float *)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y + 1] = sigma2;
          if (!rms_only) {
            ubuf[2 * wrt_y] = mu;
            ibuf[wrt_y] = count;
          }
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          if (!rms_only) {
            float muB = ubuf[2 * threadIdx.y];
            float countB = ibuf[threadIdx.y];
            cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
          } else {
            cuChanRMSOnlineSum(sigma2B, sigma2);
          }
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (!rms_only) {
          ubuf[0] = mu;
        }
        ubuf[1] = sigma2;
      }
      __syncthreads();
      if (!rms_only) {
        mu = ubuf[0];
      }
      sigma2 = ubuf[1] / float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      if (!rms_only) {
        mu = __shfl_sync(0xffffffff, mu, 0, warpSize);
      }
      sigma2 = __shfl_sync(0xffffffff, sigma2 / float(n2), 0, warpSize);
    }
  }
}

// This is the un-specialized struct.  Note that we prevent instantiation of
// this struct by putting an undefined symbol in the function body so it won't
// compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T> struct SharedMemory;

template <> struct SharedMemory<float> {
  __device__ float *getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <> struct SharedMemory<double> {
  __device__ double *getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

template <typename T, typename U, typename V>
__device__ void cuApplyLayerNorm_(V *__restrict__ output_vals,
                                  U *__restrict__ mean, U *__restrict__ invvar,
                                  const T *__restrict__ vals, const int n1,
                                  const int n2, const U epsilon,
                                  const V *__restrict__ gamma,
                                  const V *__restrict__ beta, bool rms_only) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U *buf = shared.getPointer();
    U mu, sigma2;
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf, rms_only);

    const T *lvals = vals + i1 * n2;
    V *ovals = output_vals + i1 * n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && (beta != NULL || rms_only)) {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] =
              gamma[i] * static_cast<V>(c_invvar * (curr - mu)) + beta[i];
        } else {
          ovals[i] = gamma[i] * static_cast<V>(c_invvar * curr);
        }
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        if (!rms_only) {
          ovals[i] = static_cast<V>(c_invvar * (curr - mu));
        } else {
          ovals[i] = static_cast<V>(c_invvar * curr);
        }
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (!rms_only) {
        mean[i1] = mu;
      }
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template <typename T, typename U, typename V = T>
__global__ void
cuApplyRMSNorm(V *__restrict__ output_vals, U *__restrict__ invvar,
               const T *__restrict__ vals, const int n1, const int n2,
               const U epsilon, const V *__restrict__ gamma) {
  cuApplyLayerNorm_<T, U, V>(output_vals, NULL, invvar, vals, n1, n2, epsilon,
                             gamma, NULL, true);
}

template <typename T, typename U, typename V = T>
void HostApplyRMSNorm(cudaStream_t stream, V *output, U *invvar, const T *input,
                      int n1, int n2, double epsilon, const V *gamma) {
  auto getMaxGridY = []() {
    int device;
    int val;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimY, device);
    return val;
  };
  const dim3 threads(32, 4, 1);
  const uint64_t maxGridY = getMaxGridY();
  const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
  int nshared =
      threads.y > 1 ? threads.y * sizeof(U) + (threads.y / 2) * sizeof(U) : 0;
  cuApplyRMSNorm<<<blocks, threads, nshared, stream>>>(
      output, invvar, input, n1, n2, U(epsilon), gamma);
}

template <typename T, typename U, typename V>
__device__ void cuLoadWriteStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const V *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar, bool rms_only) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean;
    if (!rms_only) {
      curr_mean = mean[i1];
    }
    U curr_invvar = invvar[i1];
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          warp_buf1[write_idx] = curr_dout;
          warp_buf2[write_idx] =
              curr_dout * (curr_input - curr_mean) * curr_invvar;
        } else {
          warp_buf2[write_idx] = curr_dout * (curr_input)*curr_invvar;
        }
      } else {
        if (!rms_only) {
          warp_buf1[write_idx] = U(0);
        }
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0; k < blockDim.y; ++k) {
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (!rms_only) {
        warp_buf1[write_idx] = U(0);
      }
      warp_buf2[write_idx] = U(0);
    }
  }
}

template <typename T, typename U, typename V>
__device__ void cuLoadAddStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const V *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar, bool rms_only) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean;
    if (!rms_only) {
      curr_mean = mean[i1];
    }
    U curr_invvar = invvar[i1];
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U curr_input = static_cast<U>(input[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        if (!rms_only) {
          warp_buf1[write_idx] += curr_dout;
          warp_buf2[write_idx] +=
              curr_dout * (curr_input - curr_mean) * curr_invvar;
        } else {
          warp_buf2[write_idx] += curr_dout * (curr_input)*curr_invvar;
        }
      }
    }
  }
}

template <typename T, typename U, typename V>
__global__ void cuComputePartGradGammaBeta(
    const V *__restrict__ dout, const T *__restrict__ input, const int n1,
    const int n2, const U *__restrict__ mean, const U *__restrict__ invvar,
    U epsilon, U *part_grad_gamma, U *part_grad_beta, bool rms_only) {
  const int numsegs_n1 =
      (n1 + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
  const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
  const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
  const int i1_beg_plus_one =
      (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
  const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
  const int row_stride = blockDim.x + 1;
  const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
  const int thr_load_row_off =
      (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
  const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
  SharedMemory<U> shared;
  U *buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y *
                                // blockDim.y + (blockDim.y -
                                // 1)*(blockDim.x/blockDim.y) elements
  U *warp_buf1 = (U *)buf;
  U *warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
  // compute partial sums from strided inputs
  // do this to increase number of loads in flight
  cuLoadWriteStridedInputs(i1_beg, thr_load_row_off, thr_load_col_off, i2_off,
                           row_stride, warp_buf1, warp_buf2, input, dout,
                           i1_end, n2, mean, invvar, rms_only);
  for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end;
       i1_block += blockDim.y * blockDim.y) {
    cuLoadAddStridedInputs(i1_block, thr_load_row_off, thr_load_col_off, i2_off,
                           row_stride, warp_buf1, warp_buf2, input, dout,
                           i1_end, n2, mean, invvar, rms_only);
  }
  __syncthreads();
  // inter-warp reductions
  // sum within each warp
  U acc1 = U(0);
  U acc2 = U(0);
  for (int k = 0; k < blockDim.y; ++k) {
    int row1 = threadIdx.y + k * blockDim.y;
    int idx1 = row1 * row_stride + threadIdx.x;
    if (!rms_only) {
      acc1 += warp_buf1[idx1];
    }
    acc2 += warp_buf2[idx1];
  }
  if (!rms_only) {
    warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
  }
  warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
  __syncthreads();
  // sum all warps
  for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
    if (threadIdx.y < offset) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + offset;
      int idx1 = row1 * row_stride + threadIdx.x;
      int idx2 = row2 * row_stride + threadIdx.x;
      if (!rms_only) {
        warp_buf1[idx1] += warp_buf1[idx2];
      }
      warp_buf2[idx1] += warp_buf2[idx2];
    }
    __syncthreads();
  }
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.y == 0 && i2 < n2) {
    int row1 = threadIdx.y;
    int row2 = threadIdx.y + 1;
    int idx1 = row1 * row_stride + threadIdx.x;
    int idx2 = row2 * row_stride + threadIdx.x;
    if (!rms_only) {
      part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
    }
    part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
  }
}

template <typename U, typename V>
__global__ void
cuComputeGradGammaBeta(const U *part_grad_gamma, const U *part_grad_beta,
                       const int part_size, const int n1, const int n2,
                       V *grad_gamma, V *grad_beta, bool rms_only) {
  // sum partial gradients for gamma and beta
  SharedMemory<U> shared;
  U *buf = shared.getPointer();
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i2 < n2) {
    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = part_size / blockDim.y;
    U sum_gamma = U(0);
    U sum_beta = U(0);
    const U *part_grad_gamma_ptr =
        part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
    const U *part_grad_beta_ptr =
        part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
    for (int warp_offset = 0; warp_offset < num_warp_reductions;
         ++warp_offset) {
      sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
      if (!rms_only) {
        sum_beta += part_grad_beta_ptr[warp_offset * n2];
      }
    }
    // inter-warp reductions
    const int nbsize3 = blockDim.x * blockDim.y / 2;
    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
      // top half write to shared memory
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        buf[write_idx] = sum_gamma;
        if (!rms_only) {
          buf[write_idx + nbsize3] = sum_beta;
        }
      }
      __syncthreads();
      // bottom half sums
      if (threadIdx.y < offset) {
        const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
        sum_gamma += buf[read_idx];
        if (!rms_only) {
          sum_beta += buf[read_idx + nbsize3];
        }
      }
      __syncthreads();
    }
    // write out fully summed gradients
    if (threadIdx.y == 0) {
      grad_gamma[i2] = sum_gamma;
      if (!rms_only) {
        grad_beta[i2] = sum_beta;
      }
    }
  }
}

template <typename T, typename U, typename V>
__global__ void
cuComputeGradInput(const V *__restrict__ dout, const T *__restrict__ input,
                   const int n1, const int n2, const U *__restrict__ mean,
                   const U *__restrict__ invvar, U epsilon, const V *gamma,
                   T *grad_input, bool rms_only) {
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    U c_mean;
    if (!rms_only) {
      c_mean = mean[i1];
    }
    const U c_invvar = invvar[i1];
    const T *k_input = input + i1 * n2;
    const V *k_dout = dout + i1 * n2;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          if (!rms_only) {
            sum_loss1 += c_loss * static_cast<U>(gamma[l + k]);
            sum_loss2 += c_loss * static_cast<U>(gamma[l + k]) *
                         (c_h - c_mean) * c_invvar;
          } else {
            sum_loss2 += c_loss * static_cast<U>(gamma[l + k]) * (c_h)*c_invvar;
          }
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss * static_cast<U>(gamma[l]);
          sum_loss2 +=
              c_loss * static_cast<U>(gamma[l]) * (c_h - c_mean) * c_invvar;
        } else {
          sum_loss2 += c_loss * static_cast<U>(gamma[l]) * (c_h)*c_invvar;
        }
      }
    } else {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_input[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          if (!rms_only) {
            sum_loss1 += c_loss;
            sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
          } else {
            sum_loss2 += c_loss * (c_h)*c_invvar;
          }
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        if (!rms_only) {
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        } else {
          sum_loss2 += c_loss * (c_h)*c_invvar;
        }
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
      if (!rms_only) {
        sum_loss1 += __shfl_xor_sync(0xffffffff, sum_loss1, mask, warpSize);
      }
      sum_loss2 += __shfl_xor_sync(0xffffffff, sum_loss2, mask, warpSize);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U *buf = shared.getPointer();
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          if (!rms_only) {
            buf[2 * wrt_i] = sum_loss1;
          }
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          if (!rms_only) {
            sum_loss1 += buf[2 * read_i];
          }
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        if (!rms_only) {
          buf[2 * threadIdx.x] = sum_loss1;
        }
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        if (!rms_only) {
          sum_loss1 = buf[2 * threadIdx.x];
        }
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T *k_grad_input = grad_input + i1 * n2;
    if (gamma != NULL) {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * static_cast<U>(gamma[l]);
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
          f_grad_input -= (c_h)*c_invvar * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        if (!rms_only) {
          f_grad_input -= sum_loss1;
          f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        } else {
          f_grad_input -= (c_h)*c_invvar * sum_loss2;
        }
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}

template <typename T, typename U = float, typename V = T>
void HostRMSNormGradient(cudaStream_t stream, const V *dout, const U *invvar,
                         const T *input, int n1, int n2, const V *gamma,
                         double epsilon, T *grad_input, V *grad_gamma,
                         int part_size, U *part_grad_gamma) {
  auto getMaxGridY = []() {
    int device;
    int val;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimY, device);
    return val;
  };
  const uint64_t maxGridY = getMaxGridY();
  if (gamma != NULL) {
    const dim3 threads2(32, 4, 1);
    const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
    const int nshared2_a =
        2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
    const int nshared2_b = threads2.x * threads2.y * sizeof(U);
    const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
    // note (mkozuki): I can hard code part_grad_gamma's dtype as float given
    // that the `cuda_layer_norm_gradient` doesn't support double.
    cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
        dout, input, n1, n2,
        invvar,                                               // unused
        invvar, U(epsilon), part_grad_gamma, part_grad_gamma, /* unused */
        true);

    const dim3 threads3(32, 8, 1);
    const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
    const int nshared3 = threads3.x * threads3.y * sizeof(U);
    cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
        part_grad_gamma, part_grad_gamma,          /* unused */
        part_size, n1, n2, grad_gamma, grad_gamma, /* unused */
        true);
  }

  // compute grad_input
  const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
  const dim3 threads1(32, 4, 1);
  int nshared = threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;
  cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
      dout, input, n1, n2, invvar, /* unused */
      invvar, U(epsilon), gamma, grad_input, true);
}

} // namespace

namespace gpu_ops {

void rms_forward_affine_mixed_dtypes(cudaStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len) {
  const RMSNormDescriptor &d =
      *UnpackDescriptor<RMSNormDescriptor>(opaque, opaque_len);

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      d.x_type, d.w_type, "rms_norm_cuda_kernel",
      HostApplyRMSNorm<scalar_t_in, accscalar_t, scalar_t_out>(
          stream, static_cast<scalar_t_out *>(buffers[2]),
          static_cast<accscalar_t *>(buffers[3]),
          static_cast<scalar_t_in *>(buffers[0]), d.n1, d.n2, d.eps,
          /*gamma=*/static_cast<scalar_t_out *>(buffers[1]));)
}

void rms_backward_affine(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len) {
  const RMSNormDescriptor &d =
      *UnpackDescriptor<RMSNormDescriptor>(opaque, opaque_len);

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
      d.x_type, d.w_type, "cuComputeGradInputRMS",
      HostRMSNormGradient(
          stream,
          /*dout=*/static_cast<scalar_t_out *>(buffers[0]),
          /*invvar=*/static_cast<accscalar_t *>(buffers[1]),
          /*input=*/static_cast<scalar_t_in *>(buffers[2]), d.n1, d.n2,
          // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
          // if gamma Tensor is NULL on input.
          /*gamma=*/static_cast<scalar_t_out *>(buffers[3]), d.eps,
          /*grad_input=*/static_cast<scalar_t_in *>(buffers[4]),
          /*grad_gamma=*/static_cast<scalar_t_out *>(buffers[5]),
          d.part_grad_size,
          /*part_grad_gamma=*/static_cast<accscalar_t *>(buffers[6]));)
}

} // namespace gpu_ops
