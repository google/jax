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

#ifndef JAXLIB_CPU_LAPACK_KERNELS_H_
#define JAXLIB_CPU_LAPACK_KERNELS_H_

#include <cstdint>
#include <optional>
#include <type_traits>

#include "absl/status/statusor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_status.h"

// Underlying function pointers (i.e., KERNEL_CLASS::Fn) are initialized either
// by the pybind wrapper that links them to an existing SciPy lapack instance,
// or using the lapack_kernels_strong.cc static initialization to link them
// directly to lapack for use in a pure C++ context.

namespace jax {

struct MatrixParams {
  enum class Side : char { kLeft = 'L', kRight = 'R' };
  enum class UpLo : char { kLower = 'L', kUpper = 'U' };
  enum class Diag : char { kNonUnit = 'N', kUnit = 'U' };
  enum class Transpose : char {
    kNoTrans = 'N',
    kTrans = 'T',
    kConjTrans = 'C'
  };
};

namespace svd {

enum class ComputationMode : char {
  kComputeFullUVt = 'A',  // Compute U and VT
  kComputeMinUVt = 'S',   // Compute min(M, N) columns of U and rows of VT
  kComputeVtOverwriteXPartialU = 'O',  // Compute VT, overwrite X
                                       // with partial U
  kNoComputeUVt = 'N',                 // Do not compute U or VT
};

inline bool ComputesUV(ComputationMode mode) {
  return mode == ComputationMode::kComputeFullUVt ||
         mode == ComputationMode::kComputeMinUVt;
}

}  // namespace svd

namespace eig {

enum class ComputationMode : char {
  kNoEigenvectors = 'N',
  kComputeEigenvectors = 'V',
};

}

template <typename KernelType>
void AssignKernelFn(void* func) {
  KernelType::fn = reinterpret_cast<typename KernelType::FnType*>(func);
}

template <typename KernelType>
void AssignKernelFn(typename KernelType::FnType* func) {
  KernelType::fn = func;
}

}  // namespace jax

#define DEFINE_CHAR_ENUM_ATTR_DECODING(ATTR)                             \
  template <>                                                            \
  struct xla::ffi::AttrDecoding<ATTR> {                                  \
    using Type = ATTR;                                                   \
    static std::optional<Type> Decode(XLA_FFI_AttrType type, void* attr, \
                                      DiagnosticEngine& diagnostic);     \
  }

// XLA needs attributes to have deserialization method specified
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Side);
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::UpLo);
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Transpose);
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Diag);
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::svd::ComputationMode);
DEFINE_CHAR_ENUM_ATTR_DECODING(jax::eig::ComputationMode);

#undef DEFINE_CHAR_ENUM_ATTR_DECODING

namespace jax {

using lapack_int = int;
inline constexpr auto LapackIntDtype = ::xla::ffi::DataType::S32;
static_assert(
    std::is_same_v<::xla::ffi::NativeType<LapackIntDtype>, lapack_int>);

//== Triangular System Solver ==//

// lapack trsm

template <typename T>
struct Trsm {
  using FnType = void(char* side, char* uplo, char* transa, char* diag,
                      lapack_int* m, lapack_int* n, T* alpha, T* a,
                      lapack_int* lda, T* b, lapack_int* ldb);

  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// FFI Kernel

template <::xla::ffi::DataType dtype>
struct TriMatrixEquationSolver {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* side, char* uplo, char* transa, char* diag,
                      lapack_int* m, lapack_int* n, ValueType* alpha,
                      ValueType* a, lapack_int* lda, ValueType* b,
                      lapack_int* ldb);

  inline static FnType* fn = nullptr;
  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::Buffer<dtype> y,
      ::xla::ffi::BufferR0<dtype> alpha, ::xla::ffi::ResultBuffer<dtype> y_out,
      MatrixParams::Side side, MatrixParams::UpLo uplo,
      MatrixParams::Transpose trans_x, MatrixParams::Diag diag);
};

//== LU Decomposition ==//

// lapack getrf

template <typename T>
struct Getrf {
  using FnType = void(lapack_int* m, lapack_int* n, T* a, lapack_int* lda,
                      lapack_int* ipiv, lapack_int* info);

  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// FFI Kernel

template <::xla::ffi::DataType dtype>
struct LuDecomposition {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(lapack_int* m, lapack_int* n, ValueType* a,
                      lapack_int* lda, lapack_int* ipiv, lapack_int* info);

  inline static FnType* fn = nullptr;
  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> ipiv,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

//== QR Factorization ==//

// lapack geqrf

template <typename T>
struct Geqrf {
  using FnType = void(lapack_int* m, lapack_int* n, T* a, lapack_int* lda,
                      T* tau, T* work, lapack_int* lwork, lapack_int* info);

  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);

  static int64_t Workspace(lapack_int m, lapack_int n);
};

// FFI Kernel

template <::xla::ffi::DataType dtype>
struct QrFactorization {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(lapack_int* m, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* tau, ValueType* work,
                      lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> tau);

  static int64_t GetWorkspaceSize(lapack_int x_rows, lapack_int x_cols);
};

//== Orthogonal QR ==//

// lapack orgqr

template <typename T>
struct Orgqr {
  using FnType = void(lapack_int* m, lapack_int* n, lapack_int* k, T* a,
                      lapack_int* lda, T* tau, T* work, lapack_int* lwork,
                      lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
  static int64_t Workspace(lapack_int m, lapack_int n, lapack_int k);
};

// FFI Kernel

template <::xla::ffi::DataType dtype>
struct OrthogonalQr {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(lapack_int* m, lapack_int* n, lapack_int* k, ValueType* a,
                      lapack_int* lda, ValueType* tau, ValueType* work,
                      lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  ::xla::ffi::Buffer<dtype> tau,
                                  ::xla::ffi::ResultBuffer<dtype> x_out);

  static int64_t GetWorkspaceSize(lapack_int x_rows, lapack_int x_cols,
                                  lapack_int tau_size);
};

//== Cholesky Factorization ==//

// lapack potrf

template <typename T>
struct Potrf {
  using FnType = void(char* uplo, lapack_int* n, T* a, lapack_int* lda,
                      lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

template <::xla::ffi::DataType dtype>
struct CholeskyFactorization {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* uplo, lapack_int* n, ValueType* a, lapack_int* lda,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);
};

//== Singular Value Decomposition (SVD) ==//

// lapack gesdd

lapack_int GesddIworkSize(int64_t m, int64_t n);

template <typename T>
struct RealGesdd {
  using FnType = void(char* jobz, lapack_int* m, lapack_int* n, T* a,
                      lapack_int* lda, T* s, T* u, lapack_int* ldu, T* vt,
                      lapack_int* ldvt, T* work, lapack_int* lwork,
                      lapack_int* iwork, lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);

  static int64_t Workspace(lapack_int m, lapack_int n, bool job_opt_compute_uv,
                           bool job_opt_full_matrices);
};

lapack_int ComplexGesddRworkSize(int64_t m, int64_t n, int compute_uv);

template <typename T>
struct ComplexGesdd {
  using FnType = void(char* jobz, lapack_int* m, lapack_int* n, T* a,
                      lapack_int* lda, typename T::value_type* s, T* u,
                      lapack_int* ldu, T* vt, lapack_int* ldvt, T* work,
                      lapack_int* lwork, typename T::value_type* rwork,
                      lapack_int* iwork, lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);

  static int64_t Workspace(lapack_int m, lapack_int n, bool job_opt_compute_uv,
                           bool job_opt_full_matrices);
};

// FFI Kernel

template <::xla::ffi::DataType dtype>
struct SingularValueDecomposition {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ValueType;
  using FnType = void(char* jobz, lapack_int* m, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* s, ValueType* u,
                      lapack_int* ldu, ValueType* vt, lapack_int* ldvt,
                      ValueType* work, lapack_int* lwork, lapack_int* iwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<dtype> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<int64_t> GetWorkspaceSize(lapack_int x_rows,
                                                  lapack_int x_cols,
                                                  svd::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
struct SingularValueDecompositionComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobz, lapack_int* m, lapack_int* n, ValueType* a,
                      lapack_int* lda, RealType* s, ValueType* u,
                      lapack_int* ldu, ValueType* vt, lapack_int* ldvt,
                      ValueType* work, lapack_int* lwork, RealType* rwork,
                      lapack_int* iwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> singular_values,
      ::xla::ffi::ResultBuffer<dtype> u, ::xla::ffi::ResultBuffer<dtype> vt,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info, svd::ComputationMode mode);

  static absl::StatusOr<int64_t> GetWorkspaceSize(lapack_int x_rows,
                                                  lapack_int x_cols,
                                                  svd::ComputationMode mode);
};

namespace svd {

template <::xla::ffi::DataType dtype>
using SVDType = std::conditional_t<::xla::ffi::IsComplexType<dtype>(),
                                   SingularValueDecompositionComplex<dtype>,
                                   SingularValueDecomposition<dtype>>;

absl::StatusOr<lapack_int> GetIntWorkspaceSize(int64_t x_rows, int64_t x_cols);
absl::StatusOr<lapack_int> GetRealWorkspaceSize(int64_t x_rows, int64_t x_cols,
                                                ComputationMode mode);

}  // namespace svd

//== Eigenvalues and eigenvectors ==//

// lapack syevd/heevd

lapack_int SyevdWorkSize(int64_t n);
lapack_int SyevdIworkSize(int64_t n);

template <typename T>
struct RealSyevd {
  using FnType = void(char* jobz, char* uplo, lapack_int* n, T* a,
                      lapack_int* lda, T* w, T* work, lapack_int* lwork,
                      lapack_int* iwork, lapack_int* liwork, lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

lapack_int HeevdWorkSize(int64_t n);
lapack_int HeevdRworkSize(int64_t n);

template <typename T>
struct ComplexHeevd {
  using FnType = void(char* jobz, char* uplo, lapack_int* n, T* a,
                      lapack_int* lda, typename T::value_type* w, T* work,
                      lapack_int* lwork, typename T::value_type* rwork,
                      lapack_int* lrwork, lapack_int* iwork, lapack_int* liwork,
                      lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// FFI Kernel

namespace eig {

// Eigenvalue Decomposition
absl::StatusOr<lapack_int> GetWorkspaceSize(int64_t x_cols,
                                            ComputationMode mode);
absl::StatusOr<lapack_int> GetIntWorkspaceSize(int64_t x_cols,
                                               ComputationMode mode);

// Hermitian Eigenvalue Decomposition
absl::StatusOr<lapack_int> GetComplexWorkspaceSize(int64_t x_cols,
                                                   ComputationMode mode);
absl::StatusOr<lapack_int> GetRealWorkspaceSize(int64_t x_cols,
                                                ComputationMode mode);

}  // namespace eig

template <::xla::ffi::DataType dtype>
struct EigenvalueDecompositionSymmetric {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* jobz, char* uplo, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* w, ValueType* work,
                      lapack_int* lwork, lapack_int* iwork, lapack_int* liwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> x,
                                  MatrixParams::UpLo uplo,
                                  ::xla::ffi::ResultBuffer<dtype> x_out,
                                  ::xla::ffi::ResultBuffer<dtype> eigenvalues,
                                  ::xla::ffi::ResultBuffer<LapackIntDtype> info,
                                  eig::ComputationMode mode);
};

template <::xla::ffi::DataType dtype>
struct EigenvalueDecompositionHermitian {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobz, char* uplo, lapack_int* n, ValueType* a,
                      lapack_int* lda, RealType* w, ValueType* work,
                      lapack_int* lwork, RealType* rwork, lapack_int* lrwork,
                      lapack_int* iwork, lapack_int* liwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> eigenvalues,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info,
      eig::ComputationMode mode);
};

// lapack geev

template <typename T>
struct RealGeev {
  using FnType = void(char* jobvl, char* jobvr, lapack_int* n, T* a,
                      lapack_int* lda, T* wr, T* wi, T* vl, lapack_int* ldvl,
                      T* vr, lapack_int* ldvr, T* work, lapack_int* lwork,
                      lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

template <typename T>
struct ComplexGeev {
  using FnType = void(char* jobvl, char* jobvr, lapack_int* n, T* a,
                      lapack_int* lda, T* w, T* vl, lapack_int* ldvl, T* vr,
                      lapack_int* ldvr, T* work, lapack_int* lwork,
                      typename T::value_type* rwork, lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

// FFI Kernel

template <::xla::ffi::DataType dtype>
struct EigenvalueDecomposition {
  static_assert(!::xla::ffi::IsComplexType<dtype>(),
                "There exists a separate implementation for Complex types");

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using FnType = void(char* jobvl, char* jobvr, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* wr, ValueType* wi,
                      ValueType* vl, lapack_int* ldvl, ValueType* vr,
                      lapack_int* ldvr, ValueType* work, lapack_int* lwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
      eig::ComputationMode compute_right,
      ::xla::ffi::ResultBuffer<dtype> eigvals_real,
      ::xla::ffi::ResultBuffer<dtype> eigvals_imag,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToComplex(dtype)> eigvecs_left,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToComplex(dtype)> eigvecs_right,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_cols,
                                  eig::ComputationMode compute_left,
                                  eig::ComputationMode compute_right);
};

template <::xla::ffi::DataType dtype>
struct EigenvalueDecompositionComplex {
  static_assert(::xla::ffi::IsComplexType<dtype>());

  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* jobvl, char* jobvr, lapack_int* n, ValueType* a,
                      lapack_int* lda, ValueType* w, ValueType* vl,
                      lapack_int* ldvl, ValueType* vr, lapack_int* ldvr,
                      ValueType* work, lapack_int* lwork, RealType* rwork,
                      lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, eig::ComputationMode compute_left,
      eig::ComputationMode compute_right,
      ::xla::ffi::ResultBuffer<dtype> eigvals,
      ::xla::ffi::ResultBuffer<dtype> eigvecs_left,
      ::xla::ffi::ResultBuffer<dtype> eigvecs_right,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_cols,
                                  eig::ComputationMode compute_left,
                                  eig::ComputationMode compute_right);
};

//== Schur Decomposition ==//

// lapack gees

template <typename T>
struct RealGees {
  using FnType = void(char* jobvs, char* sort, bool (*select)(T, T),
                      lapack_int* n, T* a, lapack_int* lda, lapack_int* sdim,
                      T* wr, T* wi, T* vs, lapack_int* ldvs, T* work,
                      lapack_int* lwork, bool* bwork, lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

template <typename T>
struct ComplexGees {
  using FnType = void(char* jobvs, char* sort, bool (*select)(T), lapack_int* n,
                      T* a, lapack_int* lda, lapack_int* sdim, T* w, T* vs,
                      lapack_int* ldvs, T* work, lapack_int* lwork,
                      typename T::value_type* rwork, bool* bwork,
                      lapack_int* info);
  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);
};

//== Hessenberg Decomposition                                       ==//
//== Reduces a non-symmetric square matrix to upper Hessenberg form ==//

// lapack gehrd

template <typename T>
struct Gehrd {
  using FnType = void(lapack_int* n, lapack_int* ilo, lapack_int* ihi, T* a,
                      lapack_int* lda, T* tau, T* work, lapack_int* lwork,
                      lapack_int* info);

  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);

  static int64_t Workspace(lapack_int lda, lapack_int n, lapack_int ilo,
                           lapack_int ihi);
};

template <typename T>
struct real_type {
  typedef T type;
};
template <typename T>
struct real_type<std::complex<T>> {
  typedef T type;
};

//== Tridiagonal Reduction                                           ==//
//== Reduces a Symmetric/Hermitian square matrix to tridiagonal form ==//

// lapack sytrd/hetrd

template <typename T>
struct Sytrd {
  using FnType = void(char* uplo, lapack_int* n, T* a, lapack_int* lda,
                      typename real_type<T>::type* d,
                      typename real_type<T>::type* e, T* tau, T* work,
                      lapack_int* lwork, lapack_int* info);

  static FnType* fn;
  static void Kernel(void* out, void** data, XlaCustomCallStatus*);

  static int64_t Workspace(lapack_int lda, lapack_int n);
};

// FFI Kernel

template <::xla::ffi::DataType dtype>
struct TridiagonalReduction {
  using ValueType = ::xla::ffi::NativeType<dtype>;
  using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
  using FnType = void(char* uplo, lapack_int* n, ValueType* a, lapack_int* lda,
                      RealType* d, RealType* e, ValueType* tau, ValueType* work,
                      lapack_int* lwork, lapack_int* info);

  inline static FnType* fn = nullptr;

  static ::xla::ffi::Error Kernel(
      ::xla::ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
      ::xla::ffi::ResultBuffer<dtype> x_out,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> diagonal,
      ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> off_diagonal,
      ::xla::ffi::ResultBuffer<dtype> tau,
      ::xla::ffi::ResultBuffer<LapackIntDtype> info);

  static int64_t GetWorkspaceSize(lapack_int x_rows, lapack_int x_cols);
};

// Declare all the handler symbols
XLA_FFI_DECLARE_HANDLER_SYMBOL(blas_strsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(blas_dtrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(blas_ctrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(blas_ztrsm_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgeqrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sorgqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dorgqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cungqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zungqr_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_spotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zpotrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgesdd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ssyevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dsyevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cheevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zheevd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_sgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_cgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zgeev_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ssytrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dsytrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_chetrd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zhetrd_ffi);

}  // namespace jax

#endif  // JAXLIB_CPU_LAPACK_KERNELS_H_
