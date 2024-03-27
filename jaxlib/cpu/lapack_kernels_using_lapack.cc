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

// From a Python binary, JAX obtains its LAPACK/BLAS kernels from Scipy, but
// a C++ user should link against LAPACK directly. This is needed when using
// JAX-generated HLO from C++.

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

jax::RealGeqp3<float>::FnType sgeqp3_;
jax::RealGeqp3<double>::FnType dgeqp3_;
jax::ComplexGeqp3<std::complex<float>>::FnType cgeqp3_;
jax::ComplexGeqp3<std::complex<double>>::FnType zgeqp3_;

jax::Orgqr<float>::FnType sorgqr_;
jax::Orgqr<double>::FnType dorgqr_;
jax::Orgqr<std::complex<float>>::FnType cungqr_;
jax::Orgqr<std::complex<double>>::FnType zungqr_;

jax::Potrf<float>::FnType spotrf_;
jax::Potrf<double>::FnType dpotrf_;
jax::Potrf<std::complex<float>>::FnType cpotrf_;
jax::Potrf<std::complex<double>>::FnType zpotrf_;

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

static auto init = []() -> int {
  Trsm<float>::fn = strsm_;
  Trsm<double>::fn = dtrsm_;
  Trsm<std::complex<float>>::fn = ctrsm_;
  Trsm<std::complex<double>>::fn = ztrsm_;
  Getrf<float>::fn = sgetrf_;
  Getrf<double>::fn = dgetrf_;
  Getrf<std::complex<float>>::fn = cgetrf_;
  Getrf<std::complex<double>>::fn = zgetrf_;
  Geqrf<float>::fn = sgeqrf_;
  Geqrf<double>::fn = dgeqrf_;
  Geqrf<std::complex<float>>::fn = cgeqrf_;
  Geqrf<std::complex<double>>::fn = zgeqrf_;
  Orgqr<float>::fn = sorgqr_;
  Orgqr<double>::fn = dorgqr_;
  Orgqr<std::complex<float>>::fn = cungqr_;
  Orgqr<std::complex<double>>::fn = zungqr_;
  Potrf<float>::fn = spotrf_;
  Potrf<double>::fn = dpotrf_;
  Potrf<std::complex<float>>::fn = cpotrf_;
  Potrf<std::complex<double>>::fn = zpotrf_;
  RealGesdd<float>::fn = sgesdd_;
  RealGesdd<double>::fn = dgesdd_;
  ComplexGesdd<std::complex<float>>::fn = cgesdd_;
  ComplexGesdd<std::complex<double>>::fn = zgesdd_;
  RealSyevd<float>::fn = ssyevd_;
  RealSyevd<double>::fn = dsyevd_;
  ComplexHeevd<std::complex<float>>::fn = cheevd_;
  ComplexHeevd<std::complex<double>>::fn = zheevd_;
  RealGeev<float>::fn = sgeev_;
  RealGeev<double>::fn = dgeev_;
  ComplexGeev<std::complex<float>>::fn = cgeev_;
  ComplexGeev<std::complex<double>>::fn = zgeev_;
  RealGees<float>::fn = sgees_;
  RealGees<double>::fn = dgees_;
  ComplexGees<std::complex<float>>::fn = cgees_;
  ComplexGees<std::complex<double>>::fn = zgees_;
  Gehrd<float>::fn = sgehrd_;
  Gehrd<double>::fn = dgehrd_;
  Gehrd<std::complex<float>>::fn = cgehrd_;
  Gehrd<std::complex<double>>::fn = zgehrd_;
  Sytrd<float>::fn = ssytrd_;
  Sytrd<double>::fn = dsytrd_;
  Sytrd<std::complex<float>>::fn = chetrd_;
  Sytrd<std::complex<double>>::fn = zhetrd_;

  return 0;
}();

}  // namespace jax
