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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "absl/base/dynamic_annotations.h"
#include "jaxlib/ffi_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/service/custom_call_status.h"

static_assert(sizeof(jax::lapack_int) == sizeof(int32_t),
              "Expected LAPACK integers to be 32-bit");

namespace ffi = xla::ffi;

#define REGISTER_CHAR_ENUM_ATTR_DECODING(type)                                \
  std::optional<type> xla::ffi::AttrDecoding<type>::Decode(                   \
      XLA_FFI_AttrType attr_type, void* attr, DiagnosticEngine& diagnostic) { \
    if (attr_type != XLA_FFI_AttrType_SCALAR) [[unlikely]] {                  \
      return diagnostic.Emit("Wrong attribute type: expected ")               \
             << XLA_FFI_AttrType_SCALAR << " but got" << attr_type;           \
    }                                                                         \
    auto* scalar = reinterpret_cast<XLA_FFI_Scalar*>(attr);                   \
    if (scalar->dtype != XLA_FFI_DataType_U8) [[unlikely]] {                  \
      return diagnostic.Emit("Wrong scalar data type: expected ")             \
             << XLA_FFI_DataType_U8 << " but got " << scalar->dtype;          \
    }                                                                         \
    auto underlying =                                                         \
        *reinterpret_cast<std::underlying_type_t<type>*>(scalar->value);      \
    return static_cast<type>(underlying);                                     \
  }

REGISTER_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Side);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Transpose);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::Diag);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::MatrixParams::UpLo);
REGISTER_CHAR_ENUM_ATTR_DECODING(jax::svd::ComputationMode);

#undef REGISTER_CHAR_ENUM_ATTR_DECODING

namespace jax {

template <typename T>
inline T CastNoOverflow(int64_t value, const std::string& source = __FILE__) {
  auto result = MaybeCastNoOverflow<T>(value, source);
  if (!result.ok()) {
    throw std::overflow_error{std::string(result.status().message())};
  }
  return result.value();
}

template <ffi::DataType dtype>
void CopyIfDiffBuffer(ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions());
  if (x.typed_data() != x_out->typed_data()) {
    const auto x_size = batch_count * x_rows * x_cols;
    std::copy_n(x.typed_data(), x_size, x_out->typed_data());
  }
}

//== Triangular System Solver ==//

// lapack trsm

template <typename T>
typename Trsm<T>::FnType* Trsm<T>::fn = nullptr;

template <typename T>
void Trsm<T>::Kernel(void* out, void** data, XlaCustomCallStatus*) {
  int32_t left_side = *reinterpret_cast<int32_t*>(data[0]);
  int32_t lower = *reinterpret_cast<int32_t*>(data[1]);
  int32_t trans_a = *reinterpret_cast<int32_t*>(data[2]);
  int32_t diag = *reinterpret_cast<int32_t*>(data[3]);
  int m = *reinterpret_cast<int32_t*>(data[4]);
  int n = *reinterpret_cast<int32_t*>(data[5]);
  int batch = *reinterpret_cast<int32_t*>(data[6]);
  T* alpha = reinterpret_cast<T*>(data[7]);
  T* a = reinterpret_cast<T*>(data[8]);
  T* b = reinterpret_cast<T*>(data[9]);

  T* x = reinterpret_cast<T*>(out);
  if (x != b) {
    std::memcpy(x, b,
                static_cast<int64_t>(batch) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  char cside = left_side ? 'L' : 'R';
  char cuplo = lower ? 'L' : 'U';
  char ctransa = 'N';
  if (trans_a == 1) {
    ctransa = 'T';
  } else if (trans_a == 2) {
    ctransa = 'C';
  }
  char cdiag = diag ? 'U' : 'N';
  int lda = left_side ? m : n;
  int ldb = m;

  int64_t x_plus = static_cast<int64_t>(m) * static_cast<int64_t>(n);
  int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(lda);

  for (int i = 0; i < batch; ++i) {
    fn(&cside, &cuplo, &ctransa, &cdiag, &m, &n, alpha, a, &lda, x, &ldb);
    x += x_plus;
    a += a_plus;
  }
}

template struct Trsm<float>;
template struct Trsm<double>;
template struct Trsm<std::complex<float>>;
template struct Trsm<std::complex<double>>;

// FFI Kernel

template <ffi::DataType dtype>
ffi::Error TriMatrixEquationSolver<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::Buffer<dtype> y, ffi::BufferR0<dtype> alpha,
    ffi::ResultBuffer<dtype> y_out, MatrixParams::Side side,
    MatrixParams::UpLo uplo, MatrixParams::Transpose trans_x,
    MatrixParams::Diag diag) {
  CopyIfDiffBuffer(y, y_out);

  auto [batch_count, y_rows, y_cols] = SplitBatch2D(y.dimensions());
  auto* y_out_data = y_out->typed_data();
  lapack_int x_leading_dim_v =
      side == MatrixParams::Side::kLeft ? y_rows : y_cols;
  lapack_int y_leading_dim_v = y_rows;

  auto side_v = static_cast<char>(side);
  auto uplo_v = static_cast<char>(uplo);
  auto trans_x_v = static_cast<char>(trans_x);
  auto diag_v = static_cast<char>(diag);
  FFI_ASSIGN_OR_RETURN(auto y_rows_v, MaybeCastNoOverflow<lapack_int>(y_rows));
  FFI_ASSIGN_OR_RETURN(auto y_cols_v, MaybeCastNoOverflow<lapack_int>(y_cols));

  auto* x_data = x.typed_data();
  const int64_t y_out_step{y_rows * y_cols};
  const int64_t x_step{x_leading_dim_v * x_leading_dim_v};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&side_v, &uplo_v, &trans_x_v, &diag_v, &y_rows_v, &y_cols_v,
       alpha.typed_data(), x_data, &x_leading_dim_v, y_out_data,
       &y_leading_dim_v);

    y_out_data += y_out_step;
    x_data += x_step;
  }
  return ffi::Error::Success();
}

template struct TriMatrixEquationSolver<ffi::DataType::F32>;
template struct TriMatrixEquationSolver<ffi::DataType::F64>;
template struct TriMatrixEquationSolver<ffi::DataType::C64>;
template struct TriMatrixEquationSolver<ffi::DataType::C128>;

//== LU Decomposition ==//

// lapack getrf

template <typename T>
typename Getrf<T>::FnType* Getrf<T>::fn = nullptr;

template <typename T>
void Getrf<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int m = *(reinterpret_cast<int32_t*>(data[1]));
  int n = *(reinterpret_cast<int32_t*>(data[2]));
  const T* a_in = reinterpret_cast<T*>(data[3]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  int* ipiv = reinterpret_cast<int*>(out[1]);
  int* info = reinterpret_cast<int*>(out[2]);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }
  for (int i = 0; i < b; ++i) {
    fn(&m, &n, a_out, &m, ipiv, info);
    a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
    ipiv += std::min(m, n);
    ++info;
  }
}

template struct Getrf<float>;
template struct Getrf<double>;
template struct Getrf<std::complex<float>>;
template struct Getrf<std::complex<double>>;

// FFI Kernel

template <ffi::DataType dtype>
ffi::Error LuDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<LapackIntDtype> ipiv,
    ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_RETURN_IF_ERROR(CheckMatrixDimensions(x.dimensions()));
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions());
  auto* x_out_data = x_out->typed_data();
  auto* ipiv_data = ipiv->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t ipiv_step{std::min(x_rows, x_cols)};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, ipiv_data,
       info_data);
    x_out_data += x_out_step;
    ipiv_data += ipiv_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct LuDecomposition<ffi::DataType::F32>;
template struct LuDecomposition<ffi::DataType::F64>;
template struct LuDecomposition<ffi::DataType::C64>;
template struct LuDecomposition<ffi::DataType::C128>;

//== QR Factorization ==//

// lapack geqrf

template <typename T>
typename Geqrf<T>::FnType* Geqrf<T>::fn = nullptr;

template <typename T>
void Geqrf<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int m = *(reinterpret_cast<int32_t*>(data[1]));
  int n = *(reinterpret_cast<int32_t*>(data[2]));
  int lwork = *(reinterpret_cast<int32_t*>(data[3]));
  const T* a_in = reinterpret_cast<T*>(data[4]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  T* tau = reinterpret_cast<T*>(out[1]);
  int* info = reinterpret_cast<int*>(out[2]);
  T* work = reinterpret_cast<T*>(out[3]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  for (int i = 0; i < b; ++i) {
    fn(&m, &n, a_out, &m, tau, work, &lwork, info);
    a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
    tau += std::min(m, n);
    ++info;
  }
}

template <typename T>
int64_t Geqrf<T>::Workspace(lapack_int m, lapack_int n) {
  T work = 0;
  lapack_int lwork = -1;
  lapack_int info = 0;
  fn(&m, &n, nullptr, &m, nullptr, &work, &lwork, &info);
  return info == 0 ? static_cast<int64_t>(std::real(work)) : -1;
}

template struct Geqrf<float>;
template struct Geqrf<double>;
template struct Geqrf<std::complex<float>>;
template struct Geqrf<std::complex<double>>;

// FFI Kernel

template <ffi::DataType dtype>
ffi::Error QrFactorization<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<dtype> tau, ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<dtype> work) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions());
  auto* x_out_data = x_out->typed_data();
  auto* tau_data = tau->typed_data();
  auto* info_data = info->typed_data();
  auto* work_data = work->typed_data();

  CopyIfDiffBuffer(x, x_out);
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v, MaybeCastNoOverflow<lapack_int>(
                                                 work->dimensions().back()));
  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t tau_step{std::min(x_rows, x_cols)};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_cols_v, x_out_data, &x_leading_dim_v, tau_data, work_data,
       &workspace_dim_v, info_data);
    x_out_data += x_out_step;
    tau_data += tau_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t QrFactorization<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                                 lapack_int x_cols) {
  ValueType optimal_size{};
  lapack_int x_leading_dim_v = x_rows;
  lapack_int info = 0;
  lapack_int workspace_query = -1;
  fn(&x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, &optimal_size,
     &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct QrFactorization<ffi::DataType::F32>;
template struct QrFactorization<ffi::DataType::F64>;
template struct QrFactorization<ffi::DataType::C64>;
template struct QrFactorization<ffi::DataType::C128>;

//== Orthogonal QR                                      ==//
//== Computes orthogonal matrix Q from QR Decomposition ==//

// lapack orgqr

template <typename T>
typename Orgqr<T>::FnType* Orgqr<T>::fn = nullptr;

template <typename T>
void Orgqr<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int m = *(reinterpret_cast<int32_t*>(data[1]));
  int n = *(reinterpret_cast<int32_t*>(data[2]));
  int k = *(reinterpret_cast<int32_t*>(data[3]));
  int lwork = *(reinterpret_cast<int32_t*>(data[4]));
  const T* a_in = reinterpret_cast<T*>(data[5]);
  T* tau = reinterpret_cast<T*>(data[6]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  int* info = reinterpret_cast<int*>(out[1]);
  T* work = reinterpret_cast<T*>(out[2]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  for (int i = 0; i < b; ++i) {
    fn(&m, &n, &k, a_out, &m, tau, work, &lwork, info);
    a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
    tau += k;
    ++info;
  }
}

template <typename T>
int64_t Orgqr<T>::Workspace(int m, int n, int k) {
  T work = 0;
  int lwork = -1;
  int info = 0;
  fn(&m, &n, &k, nullptr, &m, nullptr, &work, &lwork, &info);
  return info ? -1 : static_cast<int64_t>(std::real(work));
}

template struct Orgqr<float>;
template struct Orgqr<double>;
template struct Orgqr<std::complex<float>>;
template struct Orgqr<std::complex<double>>;

// FFI Kernel

template <ffi::DataType dtype>
ffi::Error OrthogonalQr<dtype>::Kernel(ffi::Buffer<dtype> x,
                                       ffi::Buffer<dtype> tau,
                                       ffi::ResultBuffer<dtype> x_out,
                                       ffi::ResultBuffer<LapackIntDtype> info,
                                       ffi::ResultBuffer<dtype> work) {
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions());
  auto* tau_data = tau.typed_data();
  auto* x_out_data = x_out->typed_data();
  auto* info_data = info->typed_data();
  auto* work_data = work->typed_data();

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto tau_size_v, MaybeCastNoOverflow<lapack_int>(
                                            tau.dimensions().back()));
  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v, MaybeCastNoOverflow<lapack_int>(
                                                 work->dimensions().back()));
  auto x_leading_dim_v = x_rows_v;

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t tau_step{tau_size_v};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&x_rows_v, &x_cols_v, &tau_size_v, x_out_data, &x_leading_dim_v,
       tau_data, work_data, &workspace_dim_v, info_data);
    x_out_data += x_out_step;
    tau_data += tau_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
int64_t OrthogonalQr<dtype>::GetWorkspaceSize(lapack_int x_rows,
                                              lapack_int x_cols,
                                              lapack_int tau_size) {
  ValueType optimal_size = {};
  lapack_int x_leading_dim_v = x_rows;
  lapack_int info = 0;
  lapack_int workspace_query = -1;
  fn(&x_rows, &x_cols, &tau_size, nullptr, &x_leading_dim_v, nullptr,
     &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

template struct OrthogonalQr<ffi::DataType::F32>;
template struct OrthogonalQr<ffi::DataType::F64>;
template struct OrthogonalQr<ffi::DataType::C64>;
template struct OrthogonalQr<ffi::DataType::C128>;

//== Cholesky Factorization ==//

// lapack potrf

template <typename T>
typename Potrf<T>::FnType* Potrf<T>::fn = nullptr;

template <typename T>
void Potrf<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int32_t lower = *(reinterpret_cast<int32_t*>(data[0]));
  int b = *(reinterpret_cast<int32_t*>(data[1]));
  int n = *(reinterpret_cast<int32_t*>(data[2]));
  const T* a_in = reinterpret_cast<T*>(data[3]);
  char uplo = lower ? 'L' : 'U';

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  int* info = reinterpret_cast<int*>(out[1]);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  for (int i = 0; i < b; ++i) {
    fn(&uplo, &n, a_out, &n, info);
    a_out += static_cast<int64_t>(n) * static_cast<int64_t>(n);
    ++info;
  }
}

template struct Potrf<float>;
template struct Potrf<double>;
template struct Potrf<std::complex<float>>;
template struct Potrf<std::complex<double>>;

// FFI Kernel

template <ffi::DataType dtype>
ffi::Error CholeskyFactorization<dtype>::Kernel(
    ffi::Buffer<dtype> x, MatrixParams::UpLo uplo,
    ffi::ResultBuffer<dtype> x_out, ffi::ResultBuffer<LapackIntDtype> info) {
  FFI_RETURN_IF_ERROR(CheckMatrixDimensions(x.dimensions()));
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions());
  auto* x_out_data = x_out->typed_data();
  auto* info_data = info->typed_data();

  CopyIfDiffBuffer(x, x_out);

  auto uplo_v = static_cast<char>(uplo);
  FFI_ASSIGN_OR_RETURN(auto x_order_v,
                       MaybeCastNoOverflow<lapack_int>(x.dimensions().back()));
  auto x_leading_dim_v = x_order_v;

  const int64_t x_out_step{x_rows * x_cols};
  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&uplo_v, &x_order_v, x_out_data, &x_leading_dim_v, info_data);
    x_out_data += x_out_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template struct CholeskyFactorization<ffi::DataType::F32>;
template struct CholeskyFactorization<ffi::DataType::F64>;
template struct CholeskyFactorization<ffi::DataType::C64>;
template struct CholeskyFactorization<ffi::DataType::C128>;

//== Singular Value Decomposition (SVD) ==//
//== using a divide and conquer method  ==//

// lapack gesdd

static char GesddJobz(bool job_opt_compute_uv, bool job_opt_full_matrices) {
  if (!job_opt_compute_uv) {
    return 'N';
  } else if (!job_opt_full_matrices) {
    return 'S';
  }
  return 'A';
}

lapack_int GesddIworkSize(int64_t m, int64_t n) {
  return CastNoOverflow<lapack_int>(8 * std::min(m, n), "gesdd iwork");
}

template <typename T>
typename RealGesdd<T>::FnType* RealGesdd<T>::fn = nullptr;

template <typename T>
void RealGesdd<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int32_t job_opt_full_matrices = *(reinterpret_cast<int32_t*>(data[0]));
  int32_t job_opt_compute_uv = *(reinterpret_cast<int32_t*>(data[1]));
  int b = *(reinterpret_cast<int32_t*>(data[2]));
  int m = *(reinterpret_cast<int32_t*>(data[3]));
  int n = *(reinterpret_cast<int32_t*>(data[4]));
  int lwork = *(reinterpret_cast<int32_t*>(data[5]));
  T* a_in = reinterpret_cast<T*>(data[6]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  T* s = reinterpret_cast<T*>(out[1]);
  T* u = reinterpret_cast<T*>(out[2]);
  T* vt = reinterpret_cast<T*>(out[3]);
  int* info = reinterpret_cast<int*>(out[4]);
  int* iwork = reinterpret_cast<int*>(out[5]);
  T* work = reinterpret_cast<T*>(out[6]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  char jobz = GesddJobz(job_opt_compute_uv, job_opt_full_matrices);

  int lda = m;
  int ldu = m;
  int tdu = job_opt_full_matrices ? m : std::min(m, n);
  int ldvt = job_opt_full_matrices ? n : std::min(m, n);

  for (int i = 0; i < b; ++i) {
    fn(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork,
       info);
    a_out += static_cast<int64_t>(m) * n;
    s += std::min(m, n);
    u += static_cast<int64_t>(m) * tdu;
    vt += static_cast<int64_t>(ldvt) * n;
    ++info;
  }
}

template <typename T>
int64_t RealGesdd<T>::Workspace(lapack_int m, lapack_int n,
                                bool job_opt_compute_uv,
                                bool job_opt_full_matrices) {
  T work = 0;
  int lwork = -1;
  int info = 0;
  int ldvt = job_opt_full_matrices ? n : std::min(m, n);
  char jobz = GesddJobz(job_opt_compute_uv, job_opt_full_matrices);
  fn(&jobz, &m, &n, nullptr, &m, nullptr, nullptr, &m, nullptr, &ldvt, &work,
     &lwork, nullptr, &info);
  return info ? -1 : static_cast<int>(work);
}

lapack_int ComplexGesddRworkSize(int64_t m, int64_t n, int compute_uv) {
  int64_t mn = std::min(m, n);
  if (compute_uv == 0) {
    return CastNoOverflow<lapack_int>(7 * mn, "complex gesdd rwork");
  }
  int64_t mx = std::max(m, n);
  return CastNoOverflow<lapack_int>(
      std::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn),
      "complex gesdd rwork");
}

template <typename T>
typename ComplexGesdd<T>::FnType* ComplexGesdd<T>::fn = nullptr;

template <typename T>
void ComplexGesdd<T>::Kernel(void* out_tuple, void** data,
                             XlaCustomCallStatus*) {
  int32_t job_opt_full_matrices = *(reinterpret_cast<int32_t*>(data[0]));
  int32_t job_opt_compute_uv = *(reinterpret_cast<int32_t*>(data[1]));
  int b = *(reinterpret_cast<int32_t*>(data[2]));
  int m = *(reinterpret_cast<int32_t*>(data[3]));
  int n = *(reinterpret_cast<int32_t*>(data[4]));
  int lwork = *(reinterpret_cast<int32_t*>(data[5]));
  T* a_in = reinterpret_cast<T*>(data[6]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  typename T::value_type* s = reinterpret_cast<typename T::value_type*>(out[1]);
  T* u = reinterpret_cast<T*>(out[2]);
  T* vt = reinterpret_cast<T*>(out[3]);
  int* info = reinterpret_cast<int*>(out[4]);
  int* iwork = reinterpret_cast<int*>(out[5]);
  typename T::value_type* rwork =
      reinterpret_cast<typename T::value_type*>(out[6]);
  T* work = reinterpret_cast<T*>(out[7]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  char jobz = GesddJobz(job_opt_compute_uv, job_opt_full_matrices);

  int lda = m;
  int ldu = m;
  int tdu = job_opt_full_matrices ? m : std::min(m, n);
  int ldvt = job_opt_full_matrices ? n : std::min(m, n);

  for (int i = 0; i < b; ++i) {
    fn(&jobz, &m, &n, a_out, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork,
       iwork, info);
    a_out += static_cast<int64_t>(m) * n;
    s += std::min(m, n);
    u += static_cast<int64_t>(m) * tdu;
    vt += static_cast<int64_t>(ldvt) * n;
    ++info;
  }
}

template <typename T>
int64_t ComplexGesdd<T>::Workspace(lapack_int m, lapack_int n,
                                   bool job_opt_compute_uv,
                                   bool job_opt_full_matrices) {
  T work = 0;
  int lwork = -1;
  int info = 0;
  int ldvt = job_opt_full_matrices ? n : std::min(m, n);
  char jobz = GesddJobz(job_opt_compute_uv, job_opt_full_matrices);
  fn(&jobz, &m, &n, nullptr, &m, nullptr, nullptr, &m, nullptr, &ldvt, &work,
     &lwork, nullptr, nullptr, &info);
  return info ? -1 : static_cast<int>(work.real());
}

template struct RealGesdd<float>;
template struct RealGesdd<double>;
template struct ComplexGesdd<std::complex<float>>;
template struct ComplexGesdd<std::complex<double>>;

// FFI Kernel

namespace internal {

template <ffi::DataType dtype>
using RealBufferForComplexOrNull =
    std::conditional_t<ffi::IsComplexType<dtype>(),
                       ffi::ResultBuffer<ffi::ToReal(dtype)>, std::nullptr_t>;

template <ffi::DataType dtype>
static ffi::Error SvdKernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<LapackIntDtype> iwork, ffi::ResultBuffer<dtype> work,
    svd::ComputationMode mode, RealBufferForComplexOrNull<dtype> rwork) {
  if (mode == svd::ComputationMode::kComputeVtOverwriteXPartialU) [[unlikely]] {
    return ffi::Error(
        XLA_FFI_Error_Code_UNIMPLEMENTED,
        "Current implementation does not support this computation mode");
  }
  auto [batch_count, x_rows, x_cols] = SplitBatch2D(x.dimensions());
  auto* x_out_data = x_out->typed_data();
  auto* singular_values_data = singular_values->typed_data();
  auto* u_data = u->typed_data();
  auto* vt_data = vt->typed_data();
  auto* info_data = info->typed_data();
  auto* iwork_data = iwork->typed_data();
  auto* work_data = work->typed_data();

  CopyIfDiffBuffer(x, x_out);

  FFI_ASSIGN_OR_RETURN(auto x_rows_v, MaybeCastNoOverflow<lapack_int>(x_rows));
  FFI_ASSIGN_OR_RETURN(auto x_cols_v, MaybeCastNoOverflow<lapack_int>(x_cols));
  auto mode_v = static_cast<char>(mode);
  FFI_ASSIGN_OR_RETURN(auto workspace_dim_v, MaybeCastNoOverflow<lapack_int>(
                                                 work->dimensions().back()));
  auto x_leading_dim_v = x_rows_v;
  auto u_leading_dim_v = x_rows_v;

  auto u_dims = u->dimensions().last(2);
  auto vt_dims = vt->dimensions().last(2);
  FFI_ASSIGN_OR_RETURN(auto vt_leading_dim_v,
                       MaybeCastNoOverflow<lapack_int>(vt_dims.front()));

  const int64_t x_out_step{x_rows * x_cols};
  const int64_t singular_values_step{singular_values->dimensions().back()};
  const int64_t u_step{u_dims.front() * u_dims.back()};
  const int64_t vt_step{vt_leading_dim_v * vt_dims.back()};

  for (int64_t i = 0; i < batch_count; ++i) {
    if constexpr (ffi::IsComplexType<dtype>()) {
      svd::SVDType<dtype>::fn(&mode_v, &x_rows_v, &x_cols_v, x_out_data,
                              &x_leading_dim_v, singular_values_data, u_data,
                              &u_leading_dim_v, vt_data, &vt_leading_dim_v,
                              work_data, &workspace_dim_v, rwork->typed_data(),
                              iwork_data, info_data);
    } else {
      svd::SVDType<dtype>::fn(&mode_v, &x_rows_v, &x_cols_v, x_out_data,
                              &x_leading_dim_v, singular_values_data, u_data,
                              &u_leading_dim_v, vt_data, &vt_leading_dim_v,
                              work_data, &workspace_dim_v, iwork_data,
                              info_data);
    }
    x_out_data += x_out_step;
    singular_values_data += singular_values_step;
    u_data += u_step;
    vt_data += vt_step;
    ++info_data;
  }
  return ffi::Error::Success();
}

template <ffi::DataType dtype>
static int64_t SvdGetWorkspaceSize(lapack_int x_rows, lapack_int x_cols,
                                   svd::ComputationMode mode) {
  ffi::NativeType<dtype> optimal_size = {};
  lapack_int info = 0;
  lapack_int workspace_query = -1;

  auto mode_v = static_cast<char>(mode);
  auto x_leading_dim_v = x_rows;
  auto u_leading_dim_v = x_rows;
  auto vt_leading_dim_v = mode == svd::ComputationMode::kComputeFullUVt

                              ? x_cols
                              : std::min(x_rows, x_cols);
  if constexpr (ffi::IsComplexType<dtype>()) {
    svd::SVDType<dtype>::fn(
        &mode_v, &x_rows, &x_cols, nullptr, &x_leading_dim_v, nullptr, nullptr,
        &u_leading_dim_v, nullptr, &vt_leading_dim_v, &optimal_size,
        &workspace_query, nullptr, nullptr, &info);
  } else {
    svd::SVDType<dtype>::fn(&mode_v, &x_rows, &x_cols, nullptr,
                            &x_leading_dim_v, nullptr, nullptr,
                            &u_leading_dim_v, nullptr, &vt_leading_dim_v,
                            &optimal_size, &workspace_query, nullptr, &info);
  }
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

}  // namespace internal

template <ffi::DataType dtype>
ffi::Error SingularValueDecomposition<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<dtype> singular_values, ffi::ResultBuffer<dtype> u,
    ffi::ResultBuffer<dtype> vt, ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<LapackIntDtype> iwork, ffi::ResultBuffer<dtype> work,
    svd::ComputationMode mode) {
  return internal::SvdKernel<dtype>(x, x_out, singular_values, u, vt, info,
                                    iwork, work, mode, nullptr);
}

template <ffi::DataType dtype>
ffi::Error SingularValueDecompositionComplex<dtype>::Kernel(
    ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> singular_values,
    ffi::ResultBuffer<dtype> u, ffi::ResultBuffer<dtype> vt,
    ffi::ResultBuffer<LapackIntDtype> info,
    ffi::ResultBuffer<ffi::ToReal(dtype)> rwork,
    ffi::ResultBuffer<LapackIntDtype> iwork, ffi::ResultBuffer<dtype> work,
    svd::ComputationMode mode) {
  return internal::SvdKernel<dtype>(x, x_out, singular_values, u, vt, info,
                                    iwork, work, mode, rwork);
}

template <ffi::DataType dtype>
int64_t SingularValueDecomposition<dtype>::GetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  return internal::SvdGetWorkspaceSize<dtype>(x_rows, x_cols, mode);
}

template <ffi::DataType dtype>
int64_t SingularValueDecompositionComplex<dtype>::GetWorkspaceSize(
    lapack_int x_rows, lapack_int x_cols, svd::ComputationMode mode) {
  return internal::SvdGetWorkspaceSize<dtype>(x_rows, x_cols, mode);
}

lapack_int svd::GetRealWorkspaceSize(int64_t x_rows, int64_t x_cols,
                                     svd::ComputationMode mode) {
  const auto min_dim = std::min(x_rows, x_cols);
  if (!ComputesUV(mode)) {
    return CastNoOverflow<lapack_int>(7 * min_dim);
  }
  const auto max_dim = std::max(x_rows, x_cols);
  return CastNoOverflow<lapack_int>(
      std::max(5 * min_dim * min_dim + 5 * min_dim,
               2 * max_dim * min_dim + 2 * min_dim * min_dim + min_dim));
}

lapack_int svd::GetIntWorkspaceSize(int64_t x_rows, int64_t x_cols) {
  return CastNoOverflow<lapack_int>(8 * std::min(x_rows, x_cols));
}

template struct SingularValueDecomposition<ffi::DataType::F32>;
template struct SingularValueDecomposition<ffi::DataType::F64>;
template struct SingularValueDecompositionComplex<ffi::DataType::C64>;
template struct SingularValueDecompositionComplex<ffi::DataType::C128>;

//== Eigenvalues and eigenvectors ==//

// lapack syevd/heevd

// # Workspace sizes, taken from the LAPACK documentation.
lapack_int SyevdWorkSize(int64_t n) {
  return CastNoOverflow<lapack_int>(1 + 6 * n + 2 * n * n, "syevd lwork");
}

lapack_int SyevdIworkSize(int64_t n) {
  return CastNoOverflow<lapack_int>(3 + 5 * n, "syevd iwork");
}

template <typename T>
typename RealSyevd<T>::FnType* RealSyevd<T>::fn = nullptr;

template <typename T>
void RealSyevd<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int32_t lower = *(reinterpret_cast<int32_t*>(data[0]));
  int b = *(reinterpret_cast<int32_t*>(data[1]));
  int n = *(reinterpret_cast<int32_t*>(data[2]));
  const T* a_in = reinterpret_cast<T*>(data[3]);
  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  T* w_out = reinterpret_cast<T*>(out[1]);
  int* info_out = reinterpret_cast<int*>(out[2]);
  T* work = reinterpret_cast<T*>(out[3]);
  int* iwork = reinterpret_cast<int*>(out[4]);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  char jobz = 'V';
  char uplo = lower ? 'L' : 'U';

  lapack_int lwork = SyevdWorkSize(n);
  lapack_int liwork = SyevdIworkSize(n);
  for (int i = 0; i < b; ++i) {
    fn(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, iwork, &liwork,
       info_out);
    a_out += static_cast<int64_t>(n) * n;
    w_out += n;
    ++info_out;
  }
}

// Workspace sizes, taken from the LAPACK documentation.
lapack_int HeevdWorkSize(int64_t n) {
  return CastNoOverflow<lapack_int>(1 + 2 * n + n * n, "heevd work");
}

lapack_int HeevdRworkSize(int64_t n) {
  return CastNoOverflow<lapack_int>(1 + 5 * n + 2 * n * n, "heevd rwork");
}

template <typename T>
typename ComplexHeevd<T>::FnType* ComplexHeevd<T>::fn = nullptr;

template <typename T>
void ComplexHeevd<T>::Kernel(void* out_tuple, void** data,
                             XlaCustomCallStatus*) {
  int32_t lower = *(reinterpret_cast<int32_t*>(data[0]));
  int b = *(reinterpret_cast<int32_t*>(data[1]));
  int n = *(reinterpret_cast<int32_t*>(data[2]));
  const T* a_in = reinterpret_cast<T*>(data[3]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  typename T::value_type* w_out =
      reinterpret_cast<typename T::value_type*>(out[1]);
  int* info_out = reinterpret_cast<int*>(out[2]);
  T* work = reinterpret_cast<T*>(out[3]);
  typename T::value_type* rwork =
      reinterpret_cast<typename T::value_type*>(out[4]);
  int* iwork = reinterpret_cast<int*>(out[5]);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  char jobz = 'V';
  char uplo = lower ? 'L' : 'U';

  lapack_int lwork = HeevdWorkSize(n);
  lapack_int lrwork = HeevdRworkSize(n);
  lapack_int liwork = SyevdIworkSize(n);
  for (int i = 0; i < b; ++i) {
    fn(&jobz, &uplo, &n, a_out, &n, w_out, work, &lwork, rwork, &lrwork, iwork,
       &liwork, info_out);
    a_out += static_cast<int64_t>(n) * n;
    w_out += n;
    ++info_out;
  }
}

template struct RealSyevd<float>;
template struct RealSyevd<double>;
template struct ComplexHeevd<std::complex<float>>;
template struct ComplexHeevd<std::complex<double>>;

// LAPACK uses a packed representation to represent a mixture of real
// eigenvectors and complex conjugate pairs. This helper unpacks the
// representation into regular complex matrices.
template <typename T>
static void UnpackEigenvectors(int n, const T* im_eigenvalues, const T* packed,
                               std::complex<T>* unpacked) {
  T re, im;
  int j;
  j = 0;
  while (j < n) {
    if (im_eigenvalues[j] == 0. || std::isnan(im_eigenvalues[j])) {
      for (int k = 0; k < n; ++k) {
        unpacked[j * n + k] = {packed[j * n + k], 0.};
      }
      ++j;
    } else {
      for (int k = 0; k < n; ++k) {
        re = packed[j * n + k];
        im = packed[(j + 1) * n + k];
        unpacked[j * n + k] = {re, im};
        unpacked[(j + 1) * n + k] = {re, -im};
      }
      j += 2;
    }
  }
}

// lapack geev

template <typename T>
typename RealGeev<T>::FnType* RealGeev<T>::fn = nullptr;

template <typename T>
void RealGeev<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int n_int = *(reinterpret_cast<int32_t*>(data[1]));
  int64_t n = n_int;
  char jobvl = *(reinterpret_cast<uint8_t*>(data[2]));
  char jobvr = *(reinterpret_cast<uint8_t*>(data[3]));

  const T* a_in = reinterpret_cast<T*>(data[4]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_work = reinterpret_cast<T*>(out[0]);
  T* vl_work = reinterpret_cast<T*>(out[1]);
  T* vr_work = reinterpret_cast<T*>(out[2]);

  T* wr_out = reinterpret_cast<T*>(out[3]);
  T* wi_out = reinterpret_cast<T*>(out[4]);
  std::complex<T>* vl_out = reinterpret_cast<std::complex<T>*>(out[5]);
  std::complex<T>* vr_out = reinterpret_cast<std::complex<T>*>(out[6]);
  int* info_out = reinterpret_cast<int*>(out[7]);

  // TODO(phawkins): preallocate workspace using XLA.
  T work_query;
  int lwork = -1;
  fn(&jobvl, &jobvr, &n_int, a_work, &n_int, wr_out, wi_out, vl_work, &n_int,
     vr_work, &n_int, &work_query, &lwork, info_out);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&work_query, sizeof(work_query));
  lwork = static_cast<int>(work_query);
  T* work = new T[lwork];

  auto is_finite = [](T* a_work, int64_t n) {
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t k = 0; k < n; ++k) {
        if (!std::isfinite(a_work[j * n + k])) {
          return false;
        }
      }
    }
    return true;
  };
  for (int i = 0; i < b; ++i) {
    size_t a_size = n * n * sizeof(T);
    std::memcpy(a_work, a_in, a_size);
    if (is_finite(a_work, n)) {
      fn(&jobvl, &jobvr, &n_int, a_work, &n_int, wr_out, wi_out, vl_work,
         &n_int, vr_work, &n_int, work, &lwork, info_out);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(a_work, a_size);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wr_out, sizeof(T) * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wi_out, sizeof(T) * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vl_work, sizeof(T) * n * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vr_work, sizeof(T) * n * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_out, sizeof(int));
      if (info_out[0] == 0) {
        UnpackEigenvectors(n, wi_out, vl_work, vl_out);
        UnpackEigenvectors(n, wi_out, vr_work, vr_out);
      }
    } else {
      *info_out = -4;
    }
    a_in += n * n;
    wr_out += n;
    wi_out += n;
    vl_out += n * n;
    vr_out += n * n;
    ++info_out;
  }
  delete[] work;
}

template <typename T>
typename ComplexGeev<T>::FnType* ComplexGeev<T>::fn = nullptr;

template <typename T>
void ComplexGeev<T>::Kernel(void* out_tuple, void** data,
                            XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int n_int = *(reinterpret_cast<int32_t*>(data[1]));
  int64_t n = n_int;
  char jobvl = *(reinterpret_cast<uint8_t*>(data[2]));
  char jobvr = *(reinterpret_cast<uint8_t*>(data[3]));

  const T* a_in = reinterpret_cast<T*>(data[4]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_work = reinterpret_cast<T*>(out[0]);
  typename T::value_type* r_work =
      reinterpret_cast<typename T::value_type*>(out[1]);

  T* w_out = reinterpret_cast<T*>(out[2]);
  T* vl_out = reinterpret_cast<T*>(out[3]);
  T* vr_out = reinterpret_cast<T*>(out[4]);
  int* info_out = reinterpret_cast<int*>(out[5]);

  // TODO(phawkins): preallocate workspace using XLA.
  T work_query;
  int lwork = -1;
  fn(&jobvl, &jobvr, &n_int, a_work, &n_int, w_out, vl_out, &n_int, vr_out,
     &n_int, &work_query, &lwork, r_work, info_out);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&work_query, sizeof(work_query));
  lwork = static_cast<int>(work_query.real());
  T* work = new T[lwork];

  auto is_finite = [](T* a_work, int64_t n) {
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t k = 0; k < n; ++k) {
        T v = a_work[j * n + k];
        if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) {
          return false;
        }
      }
    }
    return true;
  };

  for (int i = 0; i < b; ++i) {
    size_t a_size = n * n * sizeof(T);
    std::memcpy(a_work, a_in, a_size);
    if (is_finite(a_work, n)) {
      fn(&jobvl, &jobvr, &n_int, a_work, &n_int, w_out, vl_out, &n_int, vr_out,
         &n_int, work, &lwork, r_work, info_out);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(a_work, a_size);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(w_out, sizeof(T) * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vl_out, sizeof(T) * n * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vr_out, sizeof(T) * n * n);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_out, sizeof(int));
    } else {
      *info_out = -4;
    }
    a_in += n * n;
    w_out += n;
    vl_out += n * n;
    vr_out += n * n;
    info_out += 1;
  }
  delete[] work;
}

template struct RealGeev<float>;
template struct RealGeev<double>;
template struct ComplexGeev<std::complex<float>>;
template struct ComplexGeev<std::complex<double>>;

//== Schur Decomposition ==//

// lapack gees

template <typename T>
typename RealGees<T>::FnType* RealGees<T>::fn = nullptr;

template <typename T>
void RealGees<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int n_int = *(reinterpret_cast<int32_t*>(data[1]));
  int64_t n = n_int;
  char jobvs = *(reinterpret_cast<uint8_t*>(data[2]));
  char sort = *(reinterpret_cast<uint8_t*>(data[3]));

  const T* a_in = reinterpret_cast<T*>(data[4]);

  // bool* select (T, T) = reinterpret_cast<bool* (T, T)>(data[5]);
  bool (*select)(T, T) = nullptr;

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);

  T* wr_out = reinterpret_cast<T*>(out[1]);
  T* wi_out = reinterpret_cast<T*>(out[2]);
  T* vs_out = reinterpret_cast<T*>(out[3]);
  int* sdim_out = reinterpret_cast<int*>(out[4]);
  int* info_out = reinterpret_cast<int*>(out[5]);

  bool* b_work = (sort != 'N') ? (new bool[n]) : nullptr;

  T work_query;
  int lwork = -1;
  fn(&jobvs, &sort, select, &n_int, a_out, &n_int, sdim_out, wr_out, wi_out,
     vs_out, &n_int, &work_query, &lwork, b_work, info_out);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&work_query, sizeof(work_query));
  lwork = static_cast<int>(work_query);
  T* work = new T[lwork];

  size_t a_size = static_cast<int64_t>(n) * static_cast<int64_t>(n) * sizeof(T);
  if (a_out != a_in) {
    std::memcpy(a_out, a_in, static_cast<int64_t>(b) * a_size);
  }

  for (int i = 0; i < b; ++i) {
    fn(&jobvs, &sort, select, &n_int, a_out, &n_int, sdim_out, wr_out, wi_out,
       vs_out, &n_int, work, &lwork, b_work, info_out);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(a_out, a_size);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(sdim_out, sizeof(int));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wr_out, sizeof(T) * n);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(wi_out, sizeof(T) * n);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vs_out, sizeof(T) * n * n);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_out, sizeof(int));

    a_in += n * n;
    a_out += n * n;
    wr_out += n;
    wi_out += n;
    vs_out += n * n;
    ++sdim_out;
    ++info_out;
  }
  delete[] work;
  delete[] b_work;
}

template <typename T>
typename ComplexGees<T>::FnType* ComplexGees<T>::fn = nullptr;

template <typename T>
void ComplexGees<T>::Kernel(void* out_tuple, void** data,
                            XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int n_int = *(reinterpret_cast<int32_t*>(data[1]));
  int64_t n = n_int;
  char jobvs = *(reinterpret_cast<uint8_t*>(data[2]));
  char sort = *(reinterpret_cast<uint8_t*>(data[3]));

  const T* a_in = reinterpret_cast<T*>(data[4]);

  // bool* select (T, T) = reinterpret_cast<bool* (T, T)>(data[5]);
  bool (*select)(T) = nullptr;

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  typename T::value_type* r_work =
      reinterpret_cast<typename T::value_type*>(out[1]);
  T* w_out = reinterpret_cast<T*>(out[2]);
  T* vs_out = reinterpret_cast<T*>(out[3]);
  int* sdim_out = reinterpret_cast<int*>(out[4]);
  int* info_out = reinterpret_cast<int*>(out[5]);

  bool* b_work = (sort != 'N') ? (new bool[n]) : nullptr;

  T work_query;
  int lwork = -1;
  fn(&jobvs, &sort, select, &n_int, a_out, &n_int, sdim_out, w_out, vs_out,
     &n_int, &work_query, &lwork, r_work, b_work, info_out);
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&work_query, sizeof(work_query));
  lwork = static_cast<int>(work_query.real());
  T* work = new T[lwork];

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  for (int i = 0; i < b; ++i) {
    fn(&jobvs, &sort, select, &n_int, a_out, &n_int, sdim_out, w_out, vs_out,
       &n_int, work, &lwork, r_work, b_work, info_out);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(w_out, sizeof(T) * n);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(vs_out, sizeof(T) * n * n);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(info_out, sizeof(int));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(sdim_out, sizeof(int));

    a_in += n * n;
    a_out += n * n;
    w_out += n;
    vs_out += n * n;
    ++info_out;
    ++sdim_out;
  }
  delete[] work;
  delete[] b_work;
}

template struct RealGees<float>;
template struct RealGees<double>;
template struct ComplexGees<std::complex<float>>;
template struct ComplexGees<std::complex<double>>;

//== Hessenberg Decomposition ==//

// lapack gehrd

template <typename T>
typename Gehrd<T>::FnType* Gehrd<T>::fn = nullptr;

template <typename T>
void Gehrd<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int32_t n = *reinterpret_cast<int32_t*>(data[0]);
  int32_t ilo = *reinterpret_cast<int32_t*>(data[1]);
  int32_t ihi = *reinterpret_cast<int32_t*>(data[2]);
  int32_t lda = *reinterpret_cast<int32_t*>(data[3]);
  int32_t batch = *reinterpret_cast<int32_t*>(data[4]);
  int32_t lwork = *reinterpret_cast<int32_t*>(data[5]);
  T* a = reinterpret_cast<T*>(data[6]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  T* tau = reinterpret_cast<T*>(out[1]);
  int* info = reinterpret_cast<int*>(out[2]);
  T* work = reinterpret_cast<T*>(out[3]);

  if (a_out != a) {
    std::memcpy(a_out, a,
                static_cast<int64_t>(batch) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(n);

  for (int i = 0; i < batch; ++i) {
    fn(&n, &ilo, &ihi, a_out, &lda, tau, work, &lwork, info);
    a_out += a_plus;
    tau += n - 1;
    ++info;
  }
}

template <typename T>
int64_t Gehrd<T>::Workspace(lapack_int lda, lapack_int n, lapack_int ilo,
                            lapack_int ihi) {
  T work = 0;
  lapack_int lwork = -1;
  lapack_int info = 0;
  fn(&n, &ilo, &ihi, nullptr, &lda, nullptr, &work, &lwork, &info);
  return info == 0 ? static_cast<int64_t>(std::real(work)) : -1;
}

template struct Gehrd<float>;
template struct Gehrd<double>;
template struct Gehrd<std::complex<float>>;
template struct Gehrd<std::complex<double>>;

//== Tridiagonal Reduction ==//

// lapack sytrd/hetrd

template <typename T>
typename Sytrd<T>::FnType* Sytrd<T>::fn = nullptr;

template <typename T>
void Sytrd<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int32_t n = *reinterpret_cast<int32_t*>(data[0]);
  int32_t lower = *reinterpret_cast<int32_t*>(data[1]);
  int32_t lda = *reinterpret_cast<int32_t*>(data[2]);
  int32_t batch = *reinterpret_cast<int32_t*>(data[3]);
  int32_t lwork = *reinterpret_cast<int32_t*>(data[4]);
  T* a = reinterpret_cast<T*>(data[5]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  typedef typename real_type<T>::type Real;
  Real* d = reinterpret_cast<Real*>(out[1]);
  Real* e = reinterpret_cast<Real*>(out[2]);
  T* tau = reinterpret_cast<T*>(out[3]);
  int* info = reinterpret_cast<int*>(out[4]);
  T* work = reinterpret_cast<T*>(out[5]);

  if (a_out != a) {
    std::memcpy(a_out, a,
                static_cast<int64_t>(batch) * static_cast<int64_t>(n) *
                    static_cast<int64_t>(n) * sizeof(T));
  }

  char cuplo = lower ? 'L' : 'U';

  int64_t a_plus = static_cast<int64_t>(lda) * static_cast<int64_t>(n);

  for (int i = 0; i < batch; ++i) {
    fn(&cuplo, &n, a_out, &lda, d, e, tau, work, &lwork, info);
    a_out += a_plus;
    d += n;
    e += n - 1;
    tau += n - 1;
    ++info;
  }
}

template <typename T>
int64_t Sytrd<T>::Workspace(lapack_int lda, lapack_int n) {
  char cuplo = 'L';
  T work = 0;
  lapack_int lwork = -1;
  lapack_int info = 0;
  fn(&cuplo, &n, nullptr, &lda, nullptr, nullptr, nullptr, &work, &lwork,
     &info);
  return info == 0 ? static_cast<int64_t>(std::real(work)) : -1;
}

template struct Sytrd<float>;
template struct Sytrd<double>;
template struct Sytrd<std::complex<float>>;
template struct Sytrd<std::complex<double>>;

// FFI Definition Macros (by DataType)

#define JAX_CPU_DEFINE_TRSM(name, data_type)               \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                           \
      name, TriMatrixEquationSolver<data_type>::Kernel,    \
      ::xla::ffi::Ffi::Bind()                              \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)       \
          .Arg<::xla::ffi::Buffer<data_type>>(/*y*/)       \
          .Arg<::xla::ffi::BufferR0<data_type>>(/*alpha*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*y_out*/)   \
          .Attr<MatrixParams::Side>("side")                \
          .Attr<MatrixParams::UpLo>("uplo")                \
          .Attr<MatrixParams::Transpose>("trans_x")        \
          .Attr<MatrixParams::Diag>("diag"))

#define JAX_CPU_DEFINE_GETRF(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, LuDecomposition<data_type>::Kernel,              \
      ::xla::ffi::Ffi::Bind()                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*ipiv*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GEQRF(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, QrFactorization<data_type>::Kernel,              \
      ::xla::ffi::Ffi::Bind()                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*tau*/)       \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*work*/))

#define JAX_CPU_DEFINE_ORGQR(name, data_type)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                             \
      name, OrthogonalQr<data_type>::Kernel,                 \
      ::xla::ffi::Ffi::Bind()                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)         \
          .Arg<::xla::ffi::Buffer<data_type>>(/*tau*/)       \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)     \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*work*/))

#define JAX_CPU_DEFINE_POTRF(name, data_type)            \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                         \
      name, CholeskyFactorization<data_type>::Kernel,    \
      ::xla::ffi::Ffi::Bind()                            \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)     \
          .Attr<MatrixParams::UpLo>("uplo")              \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/))

#define JAX_CPU_DEFINE_GESDD(name, data_type)                 \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                              \
      name, SingularValueDecomposition<data_type>::Kernel,    \
      ::xla::ffi::Ffi::Bind()                                 \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)          \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)      \
          .Ret<::xla::ffi::Buffer<data_type>>(/*s*/)          \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)          \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)         \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)  \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*iwork*/) \
          .Ret<::xla::ffi::Buffer<data_type>>(/*work*/)       \
          .Attr<svd::ComputationMode>("mode"))

#define JAX_CPU_DEFINE_GESDD_COMPLEX(name, data_type)                        \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      name, SingularValueDecompositionComplex<data_type>::Kernel,            \
      ::xla::ffi::Ffi::Bind()                                                \
          .Arg<::xla::ffi::Buffer<data_type>>(/*x*/)                         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*x_out*/)                     \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>(/*s*/)     \
          .Ret<::xla::ffi::Buffer<data_type>>(/*u*/)                         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*vt*/)                        \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)                 \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>(/*rwork*/) \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*iwork*/)                \
          .Ret<::xla::ffi::Buffer<data_type>>(/*work*/)                      \
          .Attr<svd::ComputationMode>("mode"))

// FFI Handlers

JAX_CPU_DEFINE_TRSM(blas_strsm_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_TRSM(blas_dtrsm_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_TRSM(blas_ctrsm_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_TRSM(blas_ztrsm_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GETRF(lapack_sgetrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GETRF(lapack_dgetrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GETRF(lapack_cgetrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GETRF(lapack_zgetrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GEQRF(lapack_sgeqrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GEQRF(lapack_dgeqrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GEQRF(lapack_cgeqrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GEQRF(lapack_zgeqrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_ORGQR(lapack_sorgqr_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_ORGQR(lapack_dorgqr_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_ORGQR(lapack_cungqr_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_ORGQR(lapack_zungqr_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_POTRF(lapack_spotrf_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_POTRF(lapack_dpotrf_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_POTRF(lapack_cpotrf_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_POTRF(lapack_zpotrf_ffi, ::xla::ffi::DataType::C128);

JAX_CPU_DEFINE_GESDD(lapack_sgesdd_ffi, ::xla::ffi::DataType::F32);
JAX_CPU_DEFINE_GESDD(lapack_dgesdd_ffi, ::xla::ffi::DataType::F64);
JAX_CPU_DEFINE_GESDD_COMPLEX(lapack_cgesdd_ffi, ::xla::ffi::DataType::C64);
JAX_CPU_DEFINE_GESDD_COMPLEX(lapack_zgesdd_ffi, ::xla::ffi::DataType::C128);

#undef JAX_CPU_DEFINE_TRSM
#undef JAX_CPU_DEFINE_GETRF
#undef JAX_CPU_DEFINE_GEQRF
#undef JAX_CPU_DEFINE_ORGQR
#undef JAX_CPU_DEFINE_POTRF
#undef JAX_CPU_DEFINE_GESDD
#undef JAX_CPU_DEFINE_GESDD_COMPLEX

}  // namespace jax
