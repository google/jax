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

// This file is not used by JAX itself, but exists to assist with running
// JAX-generated HLO code from outside of JAX.

#include "jaxlib/gpu/blas_kernels.h"
#include "jaxlib/gpu/blas_kernels_ffi.h"
#include "jaxlib/gpu/linalg_kernels.h"
#include "jaxlib/gpu/prng_kernels.h"
#include "jaxlib/gpu/rnn_kernels.h"
#include "jaxlib/gpu/solver_kernels.h"
#include "jaxlib/gpu/solver_kernels_ffi.h"
#include "jaxlib/gpu/sparse_kernels.h"
#include "jaxlib/gpu/triton_kernels.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_target_registry.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cublas_getrf_batched", GetrfBatched,
                                         "CUDA");
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cublas_getrf_batched_ffi", "CUDA",
                         GetrfBatchedFfi);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cublas_geqrf_batched", GeqrfBatched,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cudnn_rnn", RNNForward, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cudnn_rnn_bwd", RNNBackward, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cu_cholesky_update", CholeskyUpdate,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cu_threefry2x32", ThreeFry2x32,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_getrf", Getrf, "CUDA");
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_getrf_ffi", "CUDA",
                         GetrfFfi);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_geqrf", Geqrf, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_csrlsvqr", Csrlsvqr, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_orgqr", Orgqr, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_syevd", Syevd, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_syevj", Syevj, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_sytrd", Sytrd, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_gesvd", Gesvd, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusolver_gesvdj", Gesvdj, "CUDA");

XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cu_lu_pivots_to_permutation",
                         "CUDA", LuPivotsToPermutation);
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cu_threefry2x32_ffi", "CUDA",
                         ThreeFry2x32Ffi);

#if JAX_CUSPARSE_11300
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_csr_todense", CsrToDense,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_csr_fromdense", CsrFromDense,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_csr_matvec", CsrMatvec,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_csr_matmat", CsrMatmat,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_coo_todense", CooToDense,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_coo_fromdense", CooFromDense,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_coo_matvec", CooMatvec,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_coo_matmat", CooMatmat,
                                         "CUDA");
#endif
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_gtsv2_f32", gtsv2_f32,
                                         "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cusparse_gtsv2_f64", gtsv2_f64,
                                         "CUDA");

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("triton_kernel_call", TritonKernelCall,
                                         "CUDA");

}  // namespace
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
