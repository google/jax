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

#include <complex>
#include <type_traits>

#include "jaxlib/cpu/lapack_kernels.h"

// From a Python binary, JAX obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// JAX-generated HLO from C++.

namespace ffi = xla::ffi;

extern "C" {

jax::TriMatrixEquationSolver<ffi::DataType::F32>::FnType strsm_;
jax::TriMatrixEquationSolver<ffi::DataType::F64>::FnType dtrsm_;
jax::TriMatrixEquationSolver<ffi::DataType::C64>::FnType ctrsm_;
jax::TriMatrixEquationSolver<ffi::DataType::C128>::FnType ztrsm_;

jax::LuDecomposition<ffi::DataType::F32>::FnType sgetrf_;
jax::LuDecomposition<ffi::DataType::F64>::FnType dgetrf_;
jax::LuDecomposition<ffi::DataType::C64>::FnType cgetrf_;
jax::LuDecomposition<ffi::DataType::C128>::FnType zgetrf_;

jax::QrFactorization<ffi::DataType::F32>::FnType sgeqrf_;
jax::QrFactorization<ffi::DataType::F64>::FnType dgeqrf_;
jax::QrFactorization<ffi::DataType::C64>::FnType cgeqrf_;
jax::QrFactorization<ffi::DataType::C128>::FnType zgeqrf_;

jax::OrthogonalQr<ffi::DataType::F32>::FnType sorgqr_;
jax::OrthogonalQr<ffi::DataType::F64>::FnType dorgqr_;
jax::OrthogonalQr<ffi::DataType::C64>::FnType cungqr_;
jax::OrthogonalQr<ffi::DataType::C128>::FnType zungqr_;

jax::CholeskyFactorization<ffi::DataType::F32>::FnType spotrf_;
jax::CholeskyFactorization<ffi::DataType::F64>::FnType dpotrf_;
jax::CholeskyFactorization<ffi::DataType::C64>::FnType cpotrf_;
jax::CholeskyFactorization<ffi::DataType::C128>::FnType zpotrf_;

jax::SingularValueDecomposition<ffi::DataType::F32>::FnType sgesdd_;
jax::SingularValueDecomposition<ffi::DataType::F64>::FnType dgesdd_;
jax::SingularValueDecompositionComplex<ffi::DataType::C64>::FnType cgesdd_;
jax::SingularValueDecompositionComplex<ffi::DataType::C128>::FnType zgesdd_;

jax::EigenvalueDecompositionSymmetric<ffi::DataType::F32>::FnType ssyevd_;
jax::EigenvalueDecompositionSymmetric<ffi::DataType::F64>::FnType dsyevd_;
jax::EigenvalueDecompositionHermitian<ffi::DataType::C64>::FnType cheevd_;
jax::EigenvalueDecompositionHermitian<ffi::DataType::C128>::FnType zheevd_;

jax::EigenvalueDecomposition<ffi::DataType::F32>::FnType sgeev_;
jax::EigenvalueDecomposition<ffi::DataType::F64>::FnType dgeev_;
jax::EigenvalueDecompositionComplex<ffi::DataType::C64>::FnType cgeev_;
jax::EigenvalueDecompositionComplex<ffi::DataType::C128>::FnType zgeev_;

jax::RealGees<float>::FnType sgees_;
jax::RealGees<double>::FnType dgees_;
jax::ComplexGees<std::complex<float>>::FnType cgees_;
jax::ComplexGees<std::complex<double>>::FnType zgees_;

jax::Gehrd<float>::FnType sgehrd_;
jax::Gehrd<double>::FnType dgehrd_;
jax::Gehrd<std::complex<float>>::FnType cgehrd_;
jax::Gehrd<std::complex<double>>::FnType zgehrd_;

jax::TridiagonalReduction<ffi::DataType::F32>::FnType ssytrd_;
jax::TridiagonalReduction<ffi::DataType::F64>::FnType dsytrd_;
jax::TridiagonalReduction<ffi::DataType::C64>::FnType chetrd_;
jax::TridiagonalReduction<ffi::DataType::C128>::FnType zhetrd_;

}  // extern "C"

namespace jax {

#define JAX_KERNEL_FNTYPE_MISMATCH_MSG "FFI Kernel FnType mismatch"

static_assert(
    std::is_same_v<jax::TriMatrixEquationSolver<ffi::DataType::F32>::FnType,
                   jax::Trsm<float>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::TriMatrixEquationSolver<ffi::DataType::F64>::FnType,
                   jax::Trsm<double>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::TriMatrixEquationSolver<ffi::DataType::C64>::FnType,
                   jax::Trsm<std::complex<float>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::TriMatrixEquationSolver<ffi::DataType::C128>::FnType,
                   jax::Trsm<std::complex<double>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::LuDecomposition<ffi::DataType::F32>::FnType,
                             jax::Getrf<float>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::LuDecomposition<ffi::DataType::F64>::FnType,
                             jax::Getrf<double>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::LuDecomposition<ffi::DataType::C64>::FnType,
                             jax::Getrf<std::complex<float>>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::LuDecomposition<ffi::DataType::C128>::FnType,
                             jax::Getrf<std::complex<double>>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::QrFactorization<ffi::DataType::F32>::FnType,
                             jax::Geqrf<float>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::QrFactorization<ffi::DataType::F64>::FnType,
                             jax::Geqrf<double>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::QrFactorization<ffi::DataType::C64>::FnType,
                             jax::Geqrf<std::complex<float>>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::QrFactorization<ffi::DataType::C128>::FnType,
                             jax::Geqrf<std::complex<double>>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::OrthogonalQr<ffi::DataType::F32>::FnType,
                             jax::Orgqr<float>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::OrthogonalQr<ffi::DataType::F64>::FnType,
                             jax::Orgqr<double>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::OrthogonalQr<ffi::DataType::C64>::FnType,
                             jax::Orgqr<std::complex<float>>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(std::is_same_v<jax::OrthogonalQr<ffi::DataType::C128>::FnType,
                             jax::Orgqr<std::complex<double>>::FnType>,
              JAX_KERNEL_FNTYPE_MISMATCH_MSG);
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
static_assert(
    std::is_same_v<jax::SingularValueDecomposition<ffi::DataType::F32>::FnType,
                   jax::RealGesdd<float>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::SingularValueDecomposition<ffi::DataType::F64>::FnType,
                   jax::RealGesdd<double>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<
        jax::SingularValueDecompositionComplex<ffi::DataType::C64>::FnType,
        jax::ComplexGesdd<std::complex<float>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<
        jax::SingularValueDecompositionComplex<ffi::DataType::C128>::FnType,
        jax::ComplexGesdd<std::complex<double>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<
        jax::EigenvalueDecompositionSymmetric<ffi::DataType::F32>::FnType,
        jax::RealSyevd<float>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<
        jax::EigenvalueDecompositionSymmetric<ffi::DataType::F64>::FnType,
        jax::RealSyevd<double>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<
        jax::EigenvalueDecompositionHermitian<ffi::DataType::C64>::FnType,
        jax::ComplexHeevd<std::complex<float>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<
        jax::EigenvalueDecompositionHermitian<ffi::DataType::C128>::FnType,
        jax::ComplexHeevd<std::complex<double>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::EigenvalueDecomposition<ffi::DataType::F32>::FnType,
                   jax::RealGeev<float>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::EigenvalueDecomposition<ffi::DataType::F64>::FnType,
                   jax::RealGeev<double>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<
        jax::EigenvalueDecompositionComplex<ffi::DataType::C64>::FnType,
        jax::ComplexGeev<std::complex<float>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<
        jax::EigenvalueDecompositionComplex<ffi::DataType::C128>::FnType,
        jax::ComplexGeev<std::complex<double>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::TridiagonalReduction<ffi::DataType::F32>::FnType,
                   jax::Sytrd<float>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::TridiagonalReduction<ffi::DataType::F64>::FnType,
                   jax::Sytrd<double>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::TridiagonalReduction<ffi::DataType::C64>::FnType,
                   jax::Sytrd<std::complex<float>>::FnType>,
    JAX_KERNEL_FNTYPE_MISMATCH_MSG);
static_assert(
    std::is_same_v<jax::TridiagonalReduction<ffi::DataType::C128>::FnType,
                   jax::Sytrd<std::complex<double>>::FnType>,
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

  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F32>>(strsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::F64>>(dtrsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C64>>(ctrsm_);
  AssignKernelFn<TriMatrixEquationSolver<ffi::DataType::C128>>(ztrsm_);

  AssignKernelFn<LuDecomposition<ffi::DataType::F32>>(sgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::F64>>(dgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::C64>>(cgetrf_);
  AssignKernelFn<LuDecomposition<ffi::DataType::C128>>(zgetrf_);

  AssignKernelFn<QrFactorization<ffi::DataType::F32>>(sgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::F64>>(dgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::C64>>(cgeqrf_);
  AssignKernelFn<QrFactorization<ffi::DataType::C128>>(zgeqrf_);

  AssignKernelFn<OrthogonalQr<ffi::DataType::F32>>(sorgqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::F64>>(dorgqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::C64>>(cungqr_);
  AssignKernelFn<OrthogonalQr<ffi::DataType::C128>>(zungqr_);

  AssignKernelFn<CholeskyFactorization<ffi::DataType::F32>>(spotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::F64>>(dpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C64>>(cpotrf_);
  AssignKernelFn<CholeskyFactorization<ffi::DataType::C128>>(zpotrf_);

  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F32>>(sgesdd_);
  AssignKernelFn<SingularValueDecomposition<ffi::DataType::F64>>(dgesdd_);
  AssignKernelFn<SingularValueDecompositionComplex<ffi::DataType::C64>>(
      cgesdd_);
  AssignKernelFn<SingularValueDecompositionComplex<ffi::DataType::C128>>(
      zgesdd_);

  AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F32>>(ssyevd_);
  AssignKernelFn<EigenvalueDecompositionSymmetric<ffi::DataType::F64>>(dsyevd_);
  AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C64>>(cheevd_);
  AssignKernelFn<EigenvalueDecompositionHermitian<ffi::DataType::C128>>(
      zheevd_);

  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F32>>(sgeev_);
  AssignKernelFn<EigenvalueDecomposition<ffi::DataType::F64>>(dgeev_);
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C64>>(cgeev_);
  AssignKernelFn<EigenvalueDecompositionComplex<ffi::DataType::C128>>(zgeev_);

  AssignKernelFn<TridiagonalReduction<ffi::DataType::F32>>(ssytrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::F64>>(dsytrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C64>>(chetrd_);
  AssignKernelFn<TridiagonalReduction<ffi::DataType::C128>>(zhetrd_);

  return 0;
}();

}  // namespace jax
