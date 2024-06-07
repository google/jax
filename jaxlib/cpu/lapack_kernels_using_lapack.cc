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

#include "jaxlib/cpu/lapack_kernels.h"

#include <type_traits>

// From a Python binary, JAX obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// JAX-generated HLO from C++.

namespace ffi = xla::ffi;

extern "C" {

jax::Trsm<float>::FnType strsm_;
jax::Trsm<double>::FnType dtrsm_;
jax::Trsm<std::complex<float>>::FnType ctrsm_;
jax::Trsm<std::complex<double>>::FnType ztrsm_;

jax::Getrf<float>::FnType sgetrf_;
jax::Getrf<double>::FnType dgetrf_;
jax::Getrf<std::complex<float>>::FnType cgetrf_;
jax::Getrf<std::complex<double>>::FnType zgetrf_;

jax::Geqrf<float>::FnType sgeqrf_;
jax::Geqrf<double>::FnType dgeqrf_;
jax::Geqrf<std::complex<float>>::FnType cgeqrf_;
jax::Geqrf<std::complex<double>>::FnType zgeqrf_;

jax::Orgqr<float>::FnType sorgqr_;
jax::Orgqr<double>::FnType dorgqr_;
jax::Orgqr<std::complex<float>>::FnType cungqr_;
jax::Orgqr<std::complex<double>>::FnType zungqr_;

jax::CholeskyFactorization<ffi::DataType::F32>::FnType spotrf_;
jax::CholeskyFactorization<ffi::DataType::F64>::FnType dpotrf_;
jax::CholeskyFactorization<ffi::DataType::C64>::FnType cpotrf_;
jax::CholeskyFactorization<ffi::DataType::C128>::FnType zpotrf_;

jax::RealGesdd<float>::FnType sgesdd_;
jax::RealGesdd<double>::FnType dgesdd_;
jax::ComplexGesdd<std::complex<float>>::FnType cgesdd_;
jax::ComplexGesdd<std::complex<double>>::FnType zgesdd_;

jax::RealSyevd<float>::FnType ssyevd_;
jax::RealSyevd<double>::FnType dsyevd_;
jax::ComplexHeevd<std::complex<float>>::FnType cheevd_;
jax::ComplexHeevd<std::complex<double>>::FnType zheevd_;

jax::RealGeev<float>::FnType sgeev_;
jax::RealGeev<double>::FnType dgeev_;
jax::ComplexGeev<std::complex<float>>::FnType cgeev_;
jax::ComplexGeev<std::complex<double>>::FnType zgeev_;

jax::RealGees<float>::FnType sgees_;
jax::RealGees<double>::FnType dgees_;
jax::ComplexGees<std::complex<float>>::FnType cgees_;
jax::ComplexGees<std::complex<double>>::FnType zgees_;

jax::Gehrd<float>::FnType sgehrd_;
jax::Gehrd<double>::FnType dgehrd_;
jax::Gehrd<std::complex<float>>::FnType cgehrd_;
jax::Gehrd<std::complex<double>>::FnType zgehrd_;

jax::Sytrd<float>::FnType ssytrd_;
jax::Sytrd<double>::FnType dsytrd_;
jax::Sytrd<std::complex<float>>::FnType chetrd_;
jax::Sytrd<std::complex<double>>::FnType zhetrd_;

}  // extern "C"

namespace jax {

#define JAX_KERNEL_FNTYPE_MISMATCH_MSG "FFI Kernel FnType mismatch"

static_assert(
    std::is_same_v<jax::CholeskyFactorization<ffi::DataType::F32>::FnType,
                   jax::Potrf<float>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::CholeskyFactorization<ffi::DataType::F64>::FnType,
                   jax::Potrf<double>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::CholeskyFactorization<ffi::DataType::C64>::FnType,
                   jax::Potrf<std::complex<float>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::CholeskyFactorization<ffi::DataType::C128>::FnType,
                   jax::Potrf<std::complex<double>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);

#undef JAX_KERNEL_FNTYPE_MISMATCH_MSG

static auto init = []() -> int {
  AssignKernelFn<Trsm<float>>(strsm_);
  AssignKernelFn<Trsm<double>>(dtrsm_);
  AssignKernelFn<Trsm<std::complex<float>>>(ctrsm_);
  AssignKernelFn<Trsm<std::complex<double>>>(ztrsm_);

  AssignKernelFn<Getrf<float>>(sgetrf_);
  AssignKernelFn<Getrf<double>>(dgetrf_);
  AssignKernelFn<Getrf<std::complex<float>>>(cgetrf_);
  AssignKernelFn<Getrf<std::complex<double>>>(zgetrf_);

  AssignKernelFn<Geqrf<float>>(sgeqrf_);
  AssignKernelFn<Geqrf<double>>(dgeqrf_);
  AssignKernelFn<Geqrf<std::complex<float>>>(cgeqrf_);
  AssignKernelFn<Geqrf<std::complex<double>>>(zgeqrf_);

  AssignKernelFn<Orgqr<float>>(sorgqr_);
  AssignKernelFn<Orgqr<double>>(dorgqr_);
  AssignKernelFn<Orgqr<std::complex<float>>>(cungqr_);
  AssignKernelFn<Orgqr<std::complex<double>>>(zungqr_);

  AssignKernelFn<Potrf<float>>(spotrf_);
  AssignKernelFn<Potrf<double>>(dpotrf_);
  AssignKernelFn<Potrf<std::complex<float>>>(cpotrf_);
  AssignKernelFn<Potrf<std::complex<double>>>(zpotrf_);

  AssignKernelFn<RealGesdd<float>>(sgesdd_);
  AssignKernelFn<RealGesdd<double>>(dgesdd_);
  AssignKernelFn<ComplexGesdd<std::complex<float>>>(cgesdd_);
  AssignKernelFn<ComplexGesdd<std::complex<double>>>(zgesdd_);

  AssignKernelFn<RealSyevd<float>>(ssyevd_);
  AssignKernelFn<RealSyevd<double>>(dsyevd_);
  AssignKernelFn<ComplexHeevd<std::complex<float>>>(cheevd_);
  AssignKernelFn<ComplexHeevd<std::complex<double>>>(zheevd_);

  AssignKernelFn<RealGeev<float>>(sgeev_);
  AssignKernelFn<RealGeev<double>>(dgeev_);
  AssignKernelFn<ComplexGeev<std::complex<float>>>(cgeev_);
  AssignKernelFn<ComplexGeev<std::complex<double>>>(zgeev_);

  AssignKernelFn<RealGees<float>>(sgees_);
  AssignKernelFn<RealGees<double>>(dgees_);
  AssignKernelFn<ComplexGees<std::complex<float>>>(cgees_);
  AssignKernelFn<ComplexGees<std::complex<double>>>(zgees_);

  AssignKernelFn<Gehrd<float>>(sgehrd_);
  AssignKernelFn<Gehrd<double>>(dgehrd_);
  AssignKernelFn<Gehrd<std::complex<float>>>(cgehrd_);
  AssignKernelFn<Gehrd<std::complex<double>>>(zgehrd_);

  AssignKernelFn<Sytrd<float>>(ssytrd_);
  AssignKernelFn<Sytrd<double>>(dsytrd_);
  AssignKernelFn<Sytrd<std::complex<float>>>(chetrd_);
  AssignKernelFn<Sytrd<std::complex<double>>>(zhetrd_);

  // FFI Kernels

  AssignKernelFn<CholeskyFactorization<ffi::DataType::F32>>(spotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::F64>>(dpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C64>>(cpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C128>>(zpotrf_);

  return 0;
}();

}  // namespace jax
