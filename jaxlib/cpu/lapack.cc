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
#include <cstdint>

#include "nanobind/nanobind.h"
#include "jaxlib/cpu/lapack_kernels.h"
#include "jaxlib/kernel_nanobind_helpers.h"

namespace jax {
namespace {

namespace nb = nanobind;

using ::xla::ffi::DataType;

svd::ComputationMode GetSvdComputationMode(bool job_opt_compute_uv,
                                           bool job_opt_full_matrices) {
  if (!job_opt_compute_uv) {
    return svd::ComputationMode::kNoComputeUVt;
  } else if (!job_opt_full_matrices) {
    return svd::ComputationMode::kComputeMinUVt;
  }
  return svd::ComputationMode::kComputeFullUVt;
}

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

  AssignKernelFn<Trsm<float>>(blas_ptr("strsm"));
  AssignKernelFn<Trsm<double>>(blas_ptr("dtrsm"));
  AssignKernelFn<Trsm<std::complex<float>>>(blas_ptr("ctrsm"));
  AssignKernelFn<Trsm<std::complex<double>>>(blas_ptr("ztrsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::F32>>(blas_ptr("strsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::F64>>(blas_ptr("dtrsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::C64>>(blas_ptr("ctrsm"));
  AssignKernelFn<TriMatrixEquationSolver<DataType::C128>>(blas_ptr("ztrsm"));

  nb::module_ cython_lapack =
      nb::module_::import_("scipy.linalg.cython_lapack");
  nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
  auto lapack_ptr = [&](const char* name) {
    return nb::cast<nb::capsule>(lapack_capi[name]).data();
  };
  AssignKernelFn<Getrf<float>>(lapack_ptr("sgetrf"));
  AssignKernelFn<Getrf<double>>(lapack_ptr("dgetrf"));
  AssignKernelFn<Getrf<std::complex<float>>>(lapack_ptr("cgetrf"));
  AssignKernelFn<Getrf<std::complex<double>>>(lapack_ptr("zgetrf"));
  AssignKernelFn<LuDecomposition<DataType::F32>>(lapack_ptr("sgetrf"));
  AssignKernelFn<LuDecomposition<DataType::F64>>(lapack_ptr("dgetrf"));
  AssignKernelFn<LuDecomposition<DataType::C64>>(lapack_ptr("cgetrf"));
  AssignKernelFn<LuDecomposition<DataType::C128>>(lapack_ptr("zgetrf"));

  AssignKernelFn<Geqrf<float>>(lapack_ptr("sgeqrf"));
  AssignKernelFn<Geqrf<double>>(lapack_ptr("dgeqrf"));
  AssignKernelFn<Geqrf<std::complex<float>>>(lapack_ptr("cgeqrf"));
  AssignKernelFn<Geqrf<std::complex<double>>>(lapack_ptr("zgeqrf"));
  AssignKernelFn<QrFactorization<DataType::F32>>(lapack_ptr("sgeqrf"));
  AssignKernelFn<QrFactorization<DataType::F64>>(lapack_ptr("dgeqrf"));
  AssignKernelFn<QrFactorization<DataType::C64>>(lapack_ptr("cgeqrf"));
  AssignKernelFn<QrFactorization<DataType::C128>>(lapack_ptr("zgeqrf"));

  AssignKernelFn<Orgqr<float>>(lapack_ptr("sorgqr"));
  AssignKernelFn<Orgqr<double>>(lapack_ptr("dorgqr"));
  AssignKernelFn<Orgqr<std::complex<float>>>(lapack_ptr("cungqr"));
  AssignKernelFn<Orgqr<std::complex<double>>>(lapack_ptr("zungqr"));
  AssignKernelFn<OrthogonalQr<DataType::F32>>(lapack_ptr("sorgqr"));
  AssignKernelFn<OrthogonalQr<DataType::F64>>(lapack_ptr("dorgqr"));
  AssignKernelFn<OrthogonalQr<DataType::C64>>(lapack_ptr("cungqr"));
  AssignKernelFn<OrthogonalQr<DataType::C128>>(lapack_ptr("zungqr"));

  AssignKernelFn<Potrf<float>>(lapack_ptr("spotrf"));
  AssignKernelFn<Potrf<double>>(lapack_ptr("dpotrf"));
  AssignKernelFn<Potrf<std::complex<float>>>(lapack_ptr("cpotrf"));
  AssignKernelFn<Potrf<std::complex<double>>>(lapack_ptr("zpotrf"));
  AssignKernelFn<CholeskyFactorization<DataType::F32>>(lapack_ptr("spotrf"));
  AssignKernelFn<CholeskyFactorization<DataType::F64>>(lapack_ptr("dpotrf"));
  AssignKernelFn<CholeskyFactorization<DataType::C64>>(lapack_ptr("cpotrf"));
  AssignKernelFn<CholeskyFactorization<DataType::C128>>(lapack_ptr("zpotrf"));

  AssignKernelFn<RealGesdd<float>>(lapack_ptr("sgesdd"));
  AssignKernelFn<RealGesdd<double>>(lapack_ptr("dgesdd"));
  AssignKernelFn<ComplexGesdd<std::complex<float>>>(lapack_ptr("cgesdd"));
  AssignKernelFn<ComplexGesdd<std::complex<double>>>(lapack_ptr("zgesdd"));
  AssignKernelFn<svd::SVDType<DataType::F32>>(lapack_ptr("sgesdd"));
  AssignKernelFn<svd::SVDType<DataType::F64>>(lapack_ptr("dgesdd"));
  AssignKernelFn<svd::SVDType<DataType::C64>>(lapack_ptr("cgesdd"));
  AssignKernelFn<svd::SVDType<DataType::C128>>(lapack_ptr("zgesdd"));

  AssignKernelFn<RealSyevd<float>>(lapack_ptr("ssyevd"));
  AssignKernelFn<RealSyevd<double>>(lapack_ptr("dsyevd"));
  AssignKernelFn<ComplexHeevd<std::complex<float>>>(lapack_ptr("cheevd"));
  AssignKernelFn<ComplexHeevd<std::complex<double>>>(lapack_ptr("zheevd"));
  AssignKernelFn<EigenvalueDecompositionSymmetric<DataType::F32>>(
      lapack_ptr("ssyevd"));
  AssignKernelFn<EigenvalueDecompositionSymmetric<DataType::F64>>(
      lapack_ptr("dsyevd"));
  AssignKernelFn<EigenvalueDecompositionHermitian<DataType::C64>>(
      lapack_ptr("cheevd"));
  AssignKernelFn<EigenvalueDecompositionHermitian<DataType::C128>>(
      lapack_ptr("zheevd"));

  AssignKernelFn<RealGeev<float>>(lapack_ptr("sgeev"));
  AssignKernelFn<RealGeev<double>>(lapack_ptr("dgeev"));
  AssignKernelFn<ComplexGeev<std::complex<float>>>(lapack_ptr("cgeev"));
  AssignKernelFn<ComplexGeev<std::complex<double>>>(lapack_ptr("zgeev"));
  AssignKernelFn<EigenvalueDecomposition<DataType::F32>>(lapack_ptr("sgeev"));
  AssignKernelFn<EigenvalueDecomposition<DataType::F64>>(lapack_ptr("dgeev"));
  AssignKernelFn<EigenvalueDecompositionComplex<DataType::C64>>(
      lapack_ptr("cgeev"));
  AssignKernelFn<EigenvalueDecompositionComplex<DataType::C128>>(
      lapack_ptr("zgeev"));

  AssignKernelFn<RealGees<float>>(lapack_ptr("sgees"));
  AssignKernelFn<RealGees<double>>(lapack_ptr("dgees"));
  AssignKernelFn<ComplexGees<std::complex<float>>>(lapack_ptr("cgees"));
  AssignKernelFn<ComplexGees<std::complex<double>>>(lapack_ptr("zgees"));

  AssignKernelFn<Gehrd<float>>(lapack_ptr("sgehrd"));
  AssignKernelFn<Gehrd<double>>(lapack_ptr("dgehrd"));
  AssignKernelFn<Gehrd<std::complex<float>>>(lapack_ptr("cgehrd"));
  AssignKernelFn<Gehrd<std::complex<double>>>(lapack_ptr("zgehrd"));

  AssignKernelFn<Sytrd<float>>(lapack_ptr("ssytrd"));
  AssignKernelFn<Sytrd<double>>(lapack_ptr("dsytrd"));
  AssignKernelFn<Sytrd<std::complex<float>>>(lapack_ptr("chetrd"));
  AssignKernelFn<Sytrd<std::complex<double>>>(lapack_ptr("zhetrd"));
  AssignKernelFn<TridiagonalReduction<DataType::F32>>(lapack_ptr("ssytrd"));
  AssignKernelFn<TridiagonalReduction<DataType::F64>>(lapack_ptr("dsytrd"));
  AssignKernelFn<TridiagonalReduction<DataType::C64>>(lapack_ptr("chetrd"));
  AssignKernelFn<TridiagonalReduction<DataType::C128>>(lapack_ptr("zhetrd"));

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

  dict["blas_strsm_ffi"] = EncapsulateFunction(blas_strsm_ffi);
  dict["blas_dtrsm_ffi"] = EncapsulateFunction(blas_dtrsm_ffi);
  dict["blas_ctrsm_ffi"] = EncapsulateFunction(blas_ctrsm_ffi);
  dict["blas_ztrsm_ffi"] = EncapsulateFunction(blas_ztrsm_ffi);
  dict["lapack_sgetrf_ffi"] = EncapsulateFunction(lapack_sgetrf_ffi);
  dict["lapack_dgetrf_ffi"] = EncapsulateFunction(lapack_dgetrf_ffi);
  dict["lapack_cgetrf_ffi"] = EncapsulateFunction(lapack_cgetrf_ffi);
  dict["lapack_zgetrf_ffi"] = EncapsulateFunction(lapack_zgetrf_ffi);
  dict["lapack_sgeqrf_ffi"] = EncapsulateFunction(lapack_sgeqrf_ffi);
  dict["lapack_dgeqrf_ffi"] = EncapsulateFunction(lapack_dgeqrf_ffi);
  dict["lapack_cgeqrf_ffi"] = EncapsulateFunction(lapack_cgeqrf_ffi);
  dict["lapack_zgeqrf_ffi"] = EncapsulateFunction(lapack_zgeqrf_ffi);
  dict["lapack_sorgqr_ffi"] = EncapsulateFunction(lapack_sorgqr_ffi);
  dict["lapack_dorgqr_ffi"] = EncapsulateFunction(lapack_dorgqr_ffi);
  dict["lapack_cungqr_ffi"] = EncapsulateFunction(lapack_cungqr_ffi);
  dict["lapack_zungqr_ffi"] = EncapsulateFunction(lapack_zungqr_ffi);
  dict["lapack_spotrf_ffi"] = EncapsulateFunction(lapack_spotrf_ffi);
  dict["lapack_dpotrf_ffi"] = EncapsulateFunction(lapack_dpotrf_ffi);
  dict["lapack_cpotrf_ffi"] = EncapsulateFunction(lapack_cpotrf_ffi);
  dict["lapack_zpotrf_ffi"] = EncapsulateFunction(lapack_zpotrf_ffi);
  dict["lapack_sgesdd_ffi"] = EncapsulateFunction(lapack_sgesdd_ffi);
  dict["lapack_dgesdd_ffi"] = EncapsulateFunction(lapack_dgesdd_ffi);
  dict["lapack_cgesdd_ffi"] = EncapsulateFunction(lapack_cgesdd_ffi);
  dict["lapack_zgesdd_ffi"] = EncapsulateFunction(lapack_zgesdd_ffi);
  dict["lapack_ssyevd_ffi"] = EncapsulateFunction(lapack_ssyevd_ffi);
  dict["lapack_dsyevd_ffi"] = EncapsulateFunction(lapack_dsyevd_ffi);
  dict["lapack_cheevd_ffi"] = EncapsulateFunction(lapack_cheevd_ffi);
  dict["lapack_zheevd_ffi"] = EncapsulateFunction(lapack_zheevd_ffi);
  dict["lapack_sgeev_ffi"] = EncapsulateFunction(lapack_sgeev_ffi);
  dict["lapack_dgeev_ffi"] = EncapsulateFunction(lapack_dgeev_ffi);
  dict["lapack_cgeev_ffi"] = EncapsulateFunction(lapack_cgeev_ffi);
  dict["lapack_zgeev_ffi"] = EncapsulateFunction(lapack_zgeev_ffi);
  dict["lapack_ssytrd_ffi"] = EncapsulateFunction(lapack_ssytrd_ffi);
  dict["lapack_dsytrd_ffi"] = EncapsulateFunction(lapack_dsytrd_ffi);
  dict["lapack_chetrd_ffi"] = EncapsulateFunction(lapack_chetrd_ffi);
  dict["lapack_zhetrd_ffi"] = EncapsulateFunction(lapack_zhetrd_ffi);

  return dict;
}

NB_MODULE(_lapack, m) {
  // Populates the LAPACK kernels from scipy on first call.
  m.def("initialize", GetLapackKernelsFromScipy);
  m.def("registrations", &Registrations);
  // Submodules
  auto svd = m.def_submodule("svd");
  auto eig = m.def_submodule("eig");
  // Enums
  nb::enum_<svd::ComputationMode>(svd, "ComputationMode")
      // kComputeVtOverwriteXPartialU is not implemented
      .value("kComputeFullUVt", svd::ComputationMode::kComputeFullUVt)
      .value("kComputeMinUVt", svd::ComputationMode::kComputeMinUVt)
      .value("kNoComputeUVt", svd::ComputationMode::kNoComputeUVt);
  nb::enum_<eig::ComputationMode>(eig, "ComputationMode")
      .value("kComputeEigenvectors", eig::ComputationMode::kComputeEigenvectors)
      .value("kNoEigenvectors", eig::ComputationMode::kNoEigenvectors);

  // Old-style LAPACK Workspace Size Queries
  m.def("lapack_sgeqrf_workspace", &Geqrf<float>::Workspace, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_dgeqrf_workspace", &Geqrf<double>::Workspace, nb::arg("m"),
        nb::arg("n"));
  m.def("lapack_cgeqrf_workspace", &Geqrf<std::complex<float>>::Workspace,
        nb::arg("m"), nb::arg("n"));
  m.def("lapack_zgeqrf_workspace", &Geqrf<std::complex<double>>::Workspace,
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
  // FFI Kernel LAPACK Workspace Size Queries
  m.def("lapack_sorgqr_workspace_ffi",
        &OrthogonalQr<DataType::F32>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_dorgqr_workspace_ffi",
        &OrthogonalQr<DataType::F64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_cungqr_workspace_ffi",
        &OrthogonalQr<DataType::C64>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
  m.def("lapack_zungqr_workspace_ffi",
        &OrthogonalQr<DataType::C128>::GetWorkspaceSize, nb::arg("m"),
        nb::arg("n"), nb::arg("k"));
}

}  // namespace
}  // namespace jax
