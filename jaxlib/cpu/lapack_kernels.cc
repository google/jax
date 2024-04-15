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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "absl/base/dynamic_annotations.h"
#include "lapack_kernels.h"

namespace {

inline int64_t catch_lapack_int_overflow(const std::string& source, int64_t value) {
  if constexpr (sizeof(jax::lapack_int) == sizeof(int64_t)) {
    return value;
  } else {
    if (value > std::numeric_limits<jax::lapack_int>::max()) {
      throw std::overflow_error(source + "(=" + std::to_string(value) + ") exceeds maximum value of jax::lapack_int");
    }
    return value;
  }
}

}

namespace jax {

static_assert(sizeof(lapack_int) == sizeof(int32_t),
              "Expected LAPACK integers to be 32-bit");

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

// Getrf

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

// Geqrf

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

// Geqp3

template <typename T>
typename RealGeqp3<T>::FnType* RealGeqp3<T>::fn = nullptr;

template <typename T>
void RealGeqp3<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int m = *(reinterpret_cast<int32_t*>(data[1]));
  int n = *(reinterpret_cast<int32_t*>(data[2]));
  int lwork = *(reinterpret_cast<int32_t*>(data[3]));
  const T* a_in = reinterpret_cast<T*>(data[4]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  int* jpvt = reinterpret_cast<int*>(out[1]);
  T* tau = reinterpret_cast<T*>(out[2]);
  int* info = reinterpret_cast<int*>(out[3]);
  T* work = reinterpret_cast<T*>(out[4]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }
  
  for (int i = 0; i < b; ++i) {
    fn(&m, &n, a_out, &m, jpvt, tau, work, &lwork, info);
    a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
    jpvt += static_cast<int64_t>(n);
    tau += std::min(m, n);
    ++info;
  }
}

template <typename T>
int64_t RealGeqp3<T>::Workspace(lapack_int m, lapack_int n) {
  T work = 0;
  lapack_int lwork = -1;
  lapack_int info = 0;
  fn(&m, &n, nullptr, &m, nullptr, nullptr, &work, &lwork, &info);
  return info == 0 ? static_cast<int64_t>(std::real(work)) : -1;
}

lapack_int ComplexGeqp3RworkSize(int64_t n) {
  return catch_lapack_int_overflow("complex geqp3 rwork", 2 * n);
}

template <typename T>
typename ComplexGeqp3<T>::FnType* ComplexGeqp3<T>::fn = nullptr;

template <typename T>
void ComplexGeqp3<T>::Kernel(void* out_tuple, void** data, XlaCustomCallStatus*) {
  int b = *(reinterpret_cast<int32_t*>(data[0]));
  int m = *(reinterpret_cast<int32_t*>(data[1]));
  int n = *(reinterpret_cast<int32_t*>(data[2]));
  int lwork = *(reinterpret_cast<int32_t*>(data[3]));
  const T* a_in = reinterpret_cast<T*>(data[4]);

  void** out = reinterpret_cast<void**>(out_tuple);
  T* a_out = reinterpret_cast<T*>(out[0]);
  int* jpvt = reinterpret_cast<int*>(out[1]);
  T* tau = reinterpret_cast<T*>(out[2]);
  int* info = reinterpret_cast<int*>(out[3]);
  T* work = reinterpret_cast<T*>(out[4]);
  typename T::value_type* rwork =
      reinterpret_cast<typename T::value_type*>(out[5]);

  if (a_out != a_in) {
    std::memcpy(a_out, a_in,
                static_cast<int64_t>(b) * static_cast<int64_t>(m) *
                    static_cast<int64_t>(n) * sizeof(T));
  }
  
  for (int i = 0; i < b; ++i) {
    fn(&m, &n, a_out, &m, jpvt, tau, work, &lwork, rwork, info);
    a_out += static_cast<int64_t>(m) * static_cast<int64_t>(n);
    jpvt += static_cast<int64_t>(n);
    tau += std::min(m, n);
    ++info;
  }
}

template <typename T>
int64_t ComplexGeqp3<T>::Workspace(lapack_int m, lapack_int n) {
  T work = 0;
  lapack_int lwork = -1;
  lapack_int info = 0;
  fn(&m, &n, nullptr, &m, nullptr, nullptr, &work, &lwork, nullptr, &info);
  return info == 0 ? static_cast<int64_t>(std::real(work)) : -1;
}

template struct RealGeqp3<float>;
template struct RealGeqp3<double>;
template struct ComplexGeqp3<std::complex<float>>;
template struct ComplexGeqp3<std::complex<double>>;

// Orgqr

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

// Potrf

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

// Gesdd

static char GesddJobz(bool job_opt_compute_uv, bool job_opt_full_matrices) {
  if (!job_opt_compute_uv) {
    return 'N';
  } else if (!job_opt_full_matrices) {
    return 'S';
  }
  return 'A';
}

lapack_int GesddIworkSize(int64_t m, int64_t n) {
  return catch_lapack_int_overflow("gesdd iwork", 8 * std::min(m, n));
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
    return catch_lapack_int_overflow("complex gesdd rwork", 7 * mn);
  }
  int64_t mx = std::max(m, n);
  return catch_lapack_int_overflow("complex gesdd rwork",
                                   std::max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn));
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

// # Workspace sizes, taken from the LAPACK documentation.
lapack_int SyevdWorkSize(int64_t n) {
  return catch_lapack_int_overflow("syevd lwork", 1 + 6 * n + 2 * n * n);
}

lapack_int SyevdIworkSize(int64_t n) {
  return catch_lapack_int_overflow("syevd iwork", 3 + 5 * n);
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
  return catch_lapack_int_overflow("heevd work", 1 + 2 * n + n * n);
}

lapack_int HeevdRworkSize(int64_t n) {
  return catch_lapack_int_overflow("heevd rwork", 1 + 5 * n + 2 * n * n);
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

// Gees

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

}  // namespace jax
