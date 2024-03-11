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

#include "nanobind/nanobind.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace {

namespace nb = nanobind;

void GetLapackKernelsFromScipy() {
  static bool initialized = false;  // Protected by GIL
  if (initialized) return;
  nb::module_ cython_blas = nb::module_::import_("scipy.linalg.cython_blas");
  // Technically this is a Cython-internal API. However, it seems highly likely
  // it will remain stable because Cython itself needs API stability for
  // cross-package imports to work in the first place.
  nb::dict blas_capi = cython_blas.attr("__pyx_capi__");
  auto blas_ptr = [&](const char* name) {
    return nb::cast<nb::capsule>(blas_capi[name]).data();
  };
  Trsm<float>::fn = reinterpret_cast<Trsm<float>::FnType*>(blas_ptr("strsm"));
  Trsm<double>::fn = reinterpret_cast<Trsm<double>::FnType*>(blas_ptr("dtrsm"));
  Trsm<std::complex<float>>::fn =
      reinterpret_cast<Trsm<std::complex<float>>::FnType*>(blas_ptr("ctrsm"));
  Trsm<std::complex<double>>::fn =
      reinterpret_cast<Trsm<std::complex<double>>::FnType*>(blas_ptr("ztrsm"));

  nb::module_ cython_lapack =
      nb::module_::import_("scipy.linalg.cython_lapack");
  nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
  auto lapack_ptr = [&](const char* name) {
    return nb::cast<nb::capsule>(lapack_capi[name]).data();
  };
  Getrf<float>::fn =
      reinterpret_cast<Getrf<float>::FnType*>(lapack_ptr("sgetrf"));
  Getrf<double>::fn =
      reinterpret_cast<Getrf<double>::FnType*>(lapack_ptr("dgetrf"));
  Getrf<std::complex<float>>::fn =
      reinterpret_cast<Getrf<std::complex<float>>::FnType*>(
          lapack_ptr("cgetrf"));
  Getrf<std::complex<double>>::fn =
      reinterpret_cast<Getrf<std::complex<double>>::FnType*>(
          lapack_ptr("zgetrf"));
  Geqrf<float>::fn =
      reinterpret_cast<Geqrf<float>::FnType*>(lapack_ptr("sgeqrf"));
  Geqrf<double>::fn =
      reinterpret_cast<Geqrf<double>::FnType*>(lapack_ptr("dgeqrf"));
  Geqrf<std::complex<float>>::fn =
      reinterpret_cast<Geqrf<std::complex<float>>::FnType*>(
          lapack_ptr("cgeqrf"));
  Geqrf<std::complex<double>>::fn =
      reinterpret_cast<Geqrf<std::complex<double>>::FnType*>(
          lapack_ptr("zgeqrf"));
  RealGeqp3<float>::fn =
      reinterpret_cast<RealGeqp3<float>::FnType*>(lapack_ptr("sgeqp3"));
  RealGeqp3<double>::fn =
      reinterpret_cast<RealGeqp3<double>::FnType*>(lapack_ptr("dgeqp3"));
  ComplexGeqp3<std::complex<float>>::fn =
      reinterpret_cast<ComplexGeqp3<std::complex<float>>::FnType*>(
          lapack_ptr("cgeqp3"));
  ComplexGeqp3<std::complex<double>>::fn =
      reinterpret_cast<ComplexGeqp3<std::complex<double>>::FnType*>(
          lapack_ptr("zgeqp3"));
  Orgqr<float>::fn =
      reinterpret_cast<Orgqr<float>::FnType*>(lapack_ptr("sorgqr"));
  Orgqr<double>::fn =
      reinterpret_cast<Orgqr<double>::FnType*>(lapack_ptr("dorgqr"));
  Orgqr<std::complex<float>>::fn =
      reinterpret_cast<Orgqr<std::complex<float>>::FnType*>(
          lapack_ptr("cungqr"));
  Orgqr<std::complex<double>>::fn =
      reinterpret_cast<Orgqr<std::complex<double>>::FnType*>(
          lapack_ptr("zungqr"));
  Potrf<float>::fn =
      reinterpret_cast<Potrf<float>::FnType*>(lapack_ptr("spotrf"));
  Potrf<double>::fn =
      reinterpret_cast<Potrf<double>::FnType*>(lapack_ptr("dpotrf"));
  Potrf<std::complex<float>>::fn =
      reinterpret_cast<Potrf<std::complex<float>>::FnType*>(
          lapack_ptr("cpotrf"));
  Potrf<std::complex<double>>::fn =
      reinterpret_cast<Potrf<std::complex<double>>::FnType*>(
          lapack_ptr("zpotrf"));
  RealGesdd<float>::fn =
      reinterpret_cast<RealGesdd<float>::FnType*>(lapack_ptr("sgesdd"));
  RealGesdd<double>::fn =
      reinterpret_cast<RealGesdd<double>::FnType*>(lapack_ptr("dgesdd"));
  ComplexGesdd<std::complex<float>>::fn =
      reinterpret_cast<ComplexGesdd<std::complex<float>>::FnType*>(
          lapack_ptr("cgesdd"));
  ComplexGesdd<std::complex<double>>::fn =
      reinterpret_cast<ComplexGesdd<std::complex<double>>::FnType*>(
          lapack_ptr("zgesdd"));
  RealSyevd<float>::fn =
      reinterpret_cast<RealSyevd<float>::FnType*>(lapack_ptr("ssyevd"));
  RealSyevd<double>::fn =
      reinterpret_cast<RealSyevd<double>::FnType*>(lapack_ptr("dsyevd"));
  ComplexHeevd<std::complex<float>>::fn =
      reinterpret_cast<ComplexHeevd<std::complex<float>>::FnType*>(
          lapack_ptr("cheevd"));
  ComplexHeevd<std::complex<double>>::fn =
      reinterpret_cast<ComplexHeevd<std::complex<double>>::FnType*>(
          lapack_ptr("zheevd"));
  RealGeev<float>::fn =
      reinterpret_cast<RealGeev<float>::FnType*>(lapack_ptr("sgeev"));
  RealGeev<double>::fn =
      reinterpret_cast<RealGeev<double>::FnType*>(lapack_ptr("dgeev"));
  ComplexGeev<std::complex<float>>::fn =
      reinterpret_cast<ComplexGeev<std::complex<float>>::FnType*>(
          lapack_ptr("cgeev"));
  ComplexGeev<std::complex<double>>::fn =
      reinterpret_cast<ComplexGeev<std::complex<double>>::FnType*>(
          lapack_ptr("zgeev"));
  RealGees<float>::fn =
      reinterpret_cast<RealGees<float>::FnType*>(lapack_ptr("sgees"));
  RealGees<double>::fn =
      reinterpret_cast<RealGees<double>::FnType*>(lapack_ptr("dgees"));
  ComplexGees<std::complex<float>>::fn =
      reinterpret_cast<ComplexGees<std::complex<float>>::FnType*>(
          lapack_ptr("cgees"));
  ComplexGees<std::complex<double>>::fn =
      reinterpret_cast<ComplexGees<std::complex<double>>::FnType*>(
          lapack_ptr("zgees"));
  Gehrd<float>::fn =
      reinterpret_cast<Gehrd<float>::FnType*>(lapack_ptr("sgehrd"));
  Gehrd<double>::fn =
      reinterpret_cast<Gehrd<double>::FnType*>(lapack_ptr("dgehrd"));
  Gehrd<std::complex<float>>::fn =
      reinterpret_cast<Gehrd<std::complex<float>>::FnType*>(
          lapack_ptr("cgehrd"));
  Gehrd<std::complex<double>>::fn =
      reinterpret_cast<Gehrd<std::complex<double>>::FnType*>(
          lapack_ptr("zgehrd"));
  Sytrd<float>::fn =
      reinterpret_cast<Sytrd<float>::FnType*>(lapack_ptr("ssytrd"));
  Sytrd<double>::fn =
      reinterpret_cast<Sytrd<double>::FnType*>(lapack_ptr("dsytrd"));
  Sytrd<std::complex<float>>::fn =
      reinterpret_cast<Sytrd<std::complex<float>>::FnType*>(
          lapack_ptr("chetrd"));
  Sytrd<std::complex<double>>::fn =
      reinterpret_cast<Sytrd<std::complex<double>>::FnType*>(
          lapack_ptr("zhetrd"));

  initialized = true;
}

nb::dict Registrations() {
  nb::dict dict;
  dict["blas_strsm"] = EncapsulateFunction(Trsm<float>::Kernel);
  dict["blas_dtrsm"] = EncapsulateFunction(Trsm<double>::Kernel);
  dict["blas_ctrsm"] = EncapsulateFunction(Trsm<std::complex<float>>::Kernel);
  dict["blas_ztrsm"] = EncapsulateFunction(Trsm<std::complex<double>>::Kernel);
  dict["lapack_sgetrf"] = EncapsulateFunction(Getrf<float>::Kernel);
  dict["lapack_dgetrf"] = EncapsulateFunction(Getrf<double>::Kernel);
  dict["lapack_cgetrf"] =
      EncapsulateFunction(Getrf<std::complex<float>>::Kernel);
  dict["lapack_zgetrf"] =
      EncapsulateFunction(Getrf<std::complex<double>>::Kernel);
  dict["lapack_sgeqrf"] = EncapsulateFunction(Geqrf<float>::Kernel);
  dict["lapack_dgeqrf"] = EncapsulateFunction(Geqrf<double>::Kernel);
  dict["lapack_cgeqrf"] =
      EncapsulateFunction(Geqrf<std::complex<float>>::Kernel);
  dict["lapack_zgeqrf"] =
      EncapsulateFunction(Geqrf<std::complex<double>>::Kernel);
  dict["lapack_sgeqp3"] = EncapsulateFunction(RealGeqp3<float>::Kernel);
  dict["lapack_dgeqp3"] = EncapsulateFunction(RealGeqp3<double>::Kernel);
  dict["lapack_cgeqp3"] = 
      EncapsulateFunction(ComplexGeqp3<std::complex<float>>::Kernel);
  dict["lapack_zgeqp3"] = 
      EncapsulateFunction(ComplexGeqp3<std::complex<double>>::Kernel);
  dict["lapack_sorgqr"] = EncapsulateFunction(Orgqr<float>::Kernel);
  dict["lapack_dorgqr"] = EncapsulateFunction(Orgqr<double>::Kernel);
  dict["lapack_cungqr"] =
      EncapsulateFunction(Orgqr<std::complex<float>>::Kernel);
  dict["lapack_zungqr"] =
      EncapsulateFunction(Orgqr<std::complex<double>>::Kernel);
  dict["lapack_spotrf"] = EncapsulateFunction(Potrf<float>::Kernel);
  dict["lapack_dpotrf"] = EncapsulateFunction(Potrf<double>::Kernel);
  dict["lapack_cpotrf"] =
      EncapsulateFunction(Potrf<std::complex<float>>::Kernel);
  dict["lapack_zpotrf"] =
      EncapsulateFunction(Potrf<std::complex<double>>::Kernel);
  dict["lapack_sgesdd"] = EncapsulateFunction(RealGesdd<float>::Kernel);
  dict["lapack_dgesdd"] = EncapsulateFunction(RealGesdd<double>::Kernel);
  dict["lapack_cgesdd"] =
      EncapsulateFunction(ComplexGesdd<std::complex<float>>::Kernel);
  dict["lapack_zgesdd"] =
      EncapsulateFunction(ComplexGesdd<std::complex<double>>::Kernel);
  dict["lapack_ssyevd"] = EncapsulateFunction(RealSyevd<float>::Kernel);
  dict["lapack_dsyevd"] = EncapsulateFunction(RealSyevd<double>::Kernel);
  dict["lapack_cheevd"] =
      EncapsulateFunction(ComplexHeevd<std::complex<float>>::Kernel);
  dict["lapack_zheevd"] =
      EncapsulateFunction(ComplexHeevd<std::complex<double>>::Kernel);
  dict["lapack_sgeev"] = EncapsulateFunction(RealGeev<float>::Kernel);
  dict["lapack_dgeev"] = EncapsulateFunction(RealGeev<double>::Kernel);
  dict["lapack_cgeev"] =
      EncapsulateFunction(ComplexGeev<std::complex<float>>::Kernel);
  dict["lapack_zgeev"] =
      EncapsulateFunction(ComplexGeev<std::complex<double>>::Kernel);

  dict["lapack_sgees"] = EncapsulateFunction(RealGees<float>::Kernel);
  dict["lapack_dgees"] = EncapsulateFunction(RealGees<double>::Kernel);
  dict["lapack_cgees"] =
      EncapsulateFunction(ComplexGees<std::complex<float>>::Kernel);
  dict["lapack_zgees"] =
      EncapsulateFunction(ComplexGees<std::complex<double>>::Kernel);

  dict["lapack_sgehrd"] = EncapsulateFunction(Gehrd<float>::Kernel);
  dict["lapack_dgehrd"] = EncapsulateFunction(Gehrd<double>::Kernel);
  dict["lapack_cgehrd"] =
      EncapsulateFunction(Gehrd<std::complex<float>>::Kernel);
  dict["lapack_zgehrd"] =
      EncapsulateFunction(Gehrd<std::complex<double>>::Kernel);

  dict["lapack_ssytrd"] = EncapsulateFunction(Sytrd<float>::Kernel);
  dict["lapack_dsytrd"] = EncapsulateFunction(Sytrd<double>::Kernel);
  dict["lapack_chetrd"] =
      EncapsulateFunction(Sytrd<std::complex<float>>::Kernel);
  dict["lapack_zhetrd"] =
      EncapsulateFunction(Sytrd<std::complex<double>>::Kernel);

  return dict;
}

NB_MODULE(_lapack, m) {
  // Populates the LAPACK kernels from scipy on first call.
  m.def("initialize", GetLapackKernelsFromScipy);

  m.def("registrations", &Registrations);
  m.def("lapack_sgeqrf_workspace", &Geqrf<float>::Workspace, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_dgeqrf_workspace", &Geqrf<double>::Workspace, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_cgeqrf_workspace", &Geqrf<std::complex<float>>::Workspace,
        nb::arg("m"), nb::arg("n"));
  m.def("lapack_zgeqrf_workspace", &Geqrf<std::complex<double>>::Workspace,
        nb::arg("m"), nb::arg("n"));
  m.def("lapack_sgeqp3_workspace", &RealGeqp3<float>::Workspace, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_dgeqp3_workspace", &RealGeqp3<double>::Workspace, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_cgeqp3_workspace", &ComplexGeqp3<std::complex<float>>::Workspace,
        nb::arg("m"), nb::arg("n"));
  m.def("lapack_zgeqp3_workspace", &ComplexGeqp3<std::complex<double>>::Workspace,
        nb::arg("m"), nb::arg("n"));
  m.def("lapack_sorgqr_workspace", &Orgqr<float>::Workspace, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_dorgqr_workspace", &Orgqr<double>::Workspace, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_cungqr_workspace", &Orgqr<std::complex<float>>::Workspace,
        nb::arg("m"), nb::arg("n"), nb::arg("k"));
  m.def("lapack_zungqr_workspace", &Orgqr<std::complex<double>>::Workspace,
        nb::arg("m"), nb::arg("n"), nb::arg("k"));
  m.def("gesdd_iwork_size", &GesddIworkSize, nb::arg("m"), nb::arg("n"));
  m.def("sgesdd_work_size", &RealGesdd<float>::Workspace, nb::arg("m"),
        nb::arg("n"), nb::arg("job_opt_compute_uv"),
        nb::arg("job_opt_full_matrices"));
  m.def("dgesdd_work_size", &RealGesdd<double>::Workspace, nb::arg("m"),
        nb::arg("n"), nb::arg("job_opt_compute_uv"),
        nb::arg("job_opt_full_matrices"));
  m.def("cgeqp3_rwork_size", &ComplexGeqp3RworkSize, nb::arg("n"));
  m.def("cgesdd_rwork_size", &ComplexGesddRworkSize, nb::arg("m"), nb::arg("n"),
        nb::arg("compute_uv"));
  m.def("cgesdd_work_size", &ComplexGesdd<std::complex<float>>::Workspace,
        nb::arg("m"), nb::arg("n"), nb::arg("job_opt_compute_uv"),
        nb::arg("job_opt_full_matrices"));
  m.def("zgesdd_work_size", &ComplexGesdd<std::complex<double>>::Workspace,
        nb::arg("m"), nb::arg("n"), nb::arg("job_opt_compute_uv"),
        nb::arg("job_opt_full_matrices"));
  m.def("syevd_work_size", &SyevdWorkSize, nb::arg("n"));
  m.def("syevd_iwork_size", &SyevdIworkSize, nb::arg("n"));
  m.def("heevd_work_size", &HeevdWorkSize, nb::arg("n"));
  m.def("heevd_rwork_size", &HeevdRworkSize, nb::arg("n"));

  m.def("lapack_sgehrd_workspace", &Gehrd<float>::Workspace, nb::arg("lda"),
        nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_dgehrd_workspace", &Gehrd<double>::Workspace, nb::arg("lda"),
        nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_cgehrd_workspace", &Gehrd<std::complex<float>>::Workspace,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_zgehrd_workspace", &Gehrd<std::complex<double>>::Workspace,
        nb::arg("lda"), nb::arg("n"), nb::arg("ilo"), nb::arg("ihi"));
  m.def("lapack_ssytrd_workspace", &Sytrd<float>::Workspace, nb::arg("lda"),
        nb::arg("n"));
  m.def("lapack_dsytrd_workspace", &Sytrd<double>::Workspace, nb::arg("lda"),
        nb::arg("n"));
  m.def("lapack_chetrd_workspace", &Sytrd<std::complex<float>>::Workspace,
        nb::arg("lda"), nb::arg("n"));
  m.def("lapack_zhetrd_workspace", &Sytrd<std::complex<double>>::Workspace,
        nb::arg("lda"), nb::arg("n"));
}

}  // namespace
}  // namespace jax
