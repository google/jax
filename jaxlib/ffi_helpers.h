#ifndef JAXLIB_FFI_HELPERS_H_
#define JAXLIB_FFI_HELPERS_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <tuple>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace jax {

#define FFI_ASSIGN_OR_RETURN(lhs, rhs)      \
  if (ABSL_PREDICT_FALSE(!rhs.ok())) {      \
    return ::jax::AsFfiError(rhs.status()); \
  }                                         \
  lhs = rhs.value()

#define FFI_RETURN_IF_ERROR(...)             \
  do {                                       \
    ::xla::ffi::Error err = (__VA_ARGS__);   \
    if (ABSL_PREDICT_FALSE(err.failure())) { \
      return err;                            \
    }                                        \
  } while (0)

#define FFI_RETURN_IF_ERROR_STATUS(...)     \
  do {                                      \
    ::absl::Status status = (__VA_ARGS__);  \
    if (ABSL_PREDICT_FALSE(!status.ok())) { \
      return ::jax::AsFfiError(status);     \
    }                                       \
  } while (0)

template <typename T>
inline absl::StatusOr<T> MaybeCastNoOverflow(
    std::int64_t value, const std::string& source = __FILE__) {
  if constexpr (sizeof(T) == sizeof(std::int64_t)) {
    return value;
  } else {
    if (value > std::numeric_limits<T>::max()) [[unlikely]] {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s: Value (=%d) exceeds the maximum representable value of the "
          "desired type",
          source, value));
    }
    return static_cast<T>(value);
  }
}

inline xla::ffi::Error AsFfiError(const absl::Status& status) {
  if (ABSL_PREDICT_FALSE(!status.ok())) {
    return xla::ffi::Error(static_cast<XLA_FFI_Error_Code>(status.code()),
                           std::string(status.message()));
  } else {
    return xla::ffi::Error::Success();
  }
}

template <typename T>
xla::ffi::Error CheckMatrixDimensions(xla::ffi::Span<T> dims) {
  if (dims.size() < 2) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "Matrix must have at least 2 dimensions");
  }
  return xla::ffi::Error::Success();
}

template <typename T>
std::tuple<int64_t, int64_t, int64_t> SplitBatch2D(xla::ffi::Span<T> dims) {
  auto matrix_dims = dims.last(2);
  return std::make_tuple(absl::c_accumulate(dims.first(dims.size() - 2), 1,
                                            std::multiplies<int64_t>()),
                         matrix_dims.front(), matrix_dims.back());
}

}  // namespace jax

#endif  // JAXLIB_FFI_HELPERS_H_
