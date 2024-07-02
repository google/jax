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

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

//----------------------------------------------------------------------------//
//                            Forward pass                                    //
//----------------------------------------------------------------------------//

// c = a * (b+1)
// This strawman operation works well for demo purposes because:
// 1. it's simple enough to be quickly understood,
// 2. it's complex enough to require intermediate outputs in grad computation,
//    like many operations in practice do, and
// 3. it does not have a built-in implementation in JAX.
__global__ void FooFwdKernel(const float *a, const float *b, float *c,
                             float *b_plus_1,  // intermediate output b+1
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    b_plus_1[i] = b[i] + 1.0f;
    c[i] = a[i] * b_plus_1[i];
  }
}

// Host function wrapper that launches the kernel with hardcoded grid/block
// size. Note, it uses types from XLA FFI. The return type must be ffi::Error.
// Buffer type provides buffer dimensions, so the "n" argument here is not
// strictly necessary, but it allows us to demonstrate the use of attributes
// (.Attr in the FFI handler definition above).
ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::DataType::F32> a,
                      ffi::Buffer<ffi::DataType::F32> b,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> c,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> b_plus_1,
                      size_t n) {
  const int block_dim = 128;
  const int grid_dim = 1;
  // Note how we access regular Buffer data vs Result Buffer data:
  FooFwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      a.data, b.data, c->data, b_plus_1->data, n);
  // Check for launch time errors. Note that this function may also
  // return error codes from previous, asynchronous launches. This
  // means that an error status returned here could have been caused
  // by a different kernel previously launched by XLA.
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error(
        XLA_FFI_Error_Code_INTERNAL,
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

// Creates symbol FooFwd with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooFwd, FooFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // a
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // b
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // c
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // b_plus_1
        .Attr<size_t>("n"));

//----------------------------------------------------------------------------//
//                            Backward pass                                   //
//----------------------------------------------------------------------------//

// compute da = dc * (b+1), and
//         db = dc * a
__global__ void FooBwdKernel(const float *c_grad,    // incoming gradient wrt c
                             const float *a,         // original input a
                             const float *b_plus_1,  // intermediate output b+1
                             float *a_grad,          // outgoing gradient wrt a
                             float *b_grad,          // outgoing gradient wrt b
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    // In practice on GPUs b_plus_1 can be recomputed for practically free
    // instead of storing it out and reusing, so the reuse here is a bit
    // contrived. We do it to demonstrate residual/intermediate output passing
    // between the forward and the backward pass which becomes useful when
    // recomputation is more expensive than reuse.
    a_grad[i] = c_grad[i] * b_plus_1[i];
    b_grad[i] = c_grad[i] * a[i];
  }
}

ffi::Error FooBwdHost(cudaStream_t stream,
                      ffi::Buffer<ffi::DataType::F32> c_grad,
                      ffi::Buffer<ffi::DataType::F32> a,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> b_plus_1,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> a_grad,
                      ffi::Result<ffi::Buffer<ffi::DataType::F32>> b_grad,
                      size_t n) {
  const int block_dim = 128;
  const int grid_dim = 1;
  FooBwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      c_grad.data, a.data, b_plus_1->data, a_grad->data, b_grad->data, n);
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error(
        XLA_FFI_Error_Code_INTERNAL,
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

// Creates symbol FooBwd with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooBwd, FooBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // c_grad
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // a
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // b_plus_1
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // a_grad
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // b_grad
        .Attr<size_t>("n"));
