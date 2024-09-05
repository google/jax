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

#include <complex>

#include "jaxlib/cpu/lapack_kernels.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_target_registry.h"

#define JAX_CPU_REGISTER_HANDLER(name) \
  XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), #name, "Host", name);

namespace jax {
namespace {

// Old-style kernels
// TODO(b/344892332): To be removed after the 6M compatibility period is over.

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("blas_strsm", Trsm<float>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("blas_dtrsm", Trsm<double>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("blas_ctrsm",
                                         Trsm<std::complex<float>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("blas_ztrsm",
                                         Trsm<std::complex<double>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgetrf", Getrf<float>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgetrf", Getrf<double>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_cgetrf",
                                         Getrf<std::complex<float>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_zgetrf",
                                         Getrf<std::complex<double>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgeqrf", Geqrf<float>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgeqrf", Geqrf<double>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_cgeqrf",
                                         Geqrf<std::complex<float>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_zgeqrf",
                                         Geqrf<std::complex<double>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sorgqr", Orgqr<float>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dorgqr", Orgqr<double>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_cungqr",
                                         Orgqr<std::complex<float>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_zungqr",
                                         Orgqr<std::complex<double>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_spotrf", Potrf<float>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dpotrf", Potrf<double>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_cpotrf",
                                         Potrf<std::complex<float>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_zpotrf",
                                         Potrf<std::complex<double>>::Kernel,
                                         "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgesdd",
                                         RealGesdd<float>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgesdd",
                                         RealGesdd<double>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "lapack_cgesdd", ComplexGesdd<std::complex<float>>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "lapack_zgesdd", ComplexGesdd<std::complex<double>>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_ssyevd",
                                         RealSyevd<float>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dsyevd",
                                         RealSyevd<double>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "lapack_cheevd", ComplexHeevd<std::complex<float>>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "lapack_zheevd", ComplexHeevd<std::complex<double>>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgeev",
                                         RealGeev<float>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgeev",
                                         RealGeev<double>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "lapack_cgeev", ComplexGeev<std::complex<float>>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "lapack_zgeev", ComplexGeev<std::complex<double>>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_sgees",
                                         RealGees<float>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("lapack_dgees",
                                         RealGees<double>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "lapack_cgees", ComplexGees<std::complex<float>>::Kernel, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "lapack_zgees", ComplexGees<std::complex<double>>::Kernel, "Host");

// FFI Kernels

JAX_CPU_REGISTER_HANDLER(blas_strsm_ffi);
JAX_CPU_REGISTER_HANDLER(blas_dtrsm_ffi);
JAX_CPU_REGISTER_HANDLER(blas_ctrsm_ffi);
JAX_CPU_REGISTER_HANDLER(blas_ztrsm_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgetrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgetrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgetrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgetrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgeqrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgeqrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgeqrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgeqrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sorgqr_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dorgqr_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cungqr_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zungqr_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_spotrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dpotrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cpotrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zpotrf_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgesdd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgesdd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgesdd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgesdd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_ssyevd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dsyevd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cheevd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zheevd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_sgeev_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dgeev_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_cgeev_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zgeev_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_ssytrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_dsytrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_chetrd_ffi);
JAX_CPU_REGISTER_HANDLER(lapack_zhetrd_ffi);

#undef JAX_CPU_REGISTER_HANDLER

}  // namespace
}  // namespace jax
