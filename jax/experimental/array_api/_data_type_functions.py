# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import functools
from typing import NamedTuple
import jax
import jax.numpy as jnp


from jax.dtypes import canonicalize_dtype
from jax.experimental.array_api._dtypes import (
  bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
  float32, float64, complex64, complex128
)

_valid_dtypes = {
    bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64, complex64, complex128
}

_promotion_table = {
  (bool, bool): bool,
  (int8, int8): int8,
  (int8, int16): int16,
  (int8, int32): int32,
  (int8, int64): int64,
  (int8, uint8): int16,
  (int8, uint16): int32,
  (int8, uint32): int64,
  (int16, int8): int16,
  (int16, int16): int16,
  (int16, int32): int32,
  (int16, int64): int64,
  (int16, uint8): int16,
  (int16, uint16): int32,
  (int16, uint32): int64,
  (int32, int8): int32,
  (int32, int16): int32,
  (int32, int32): int32,
  (int32, int64): int64,
  (int32, uint8): int32,
  (int32, uint16): int32,
  (int32, uint32): int64,
  (int64, int8): int64,
  (int64, int16): int64,
  (int64, int32): int64,
  (int64, int64): int64,
  (int64, uint8): int64,
  (int64, uint16): int64,
  (int64, uint32): int64,
  (uint8, int8): int16,
  (uint8, int16): int16,
  (uint8, int32): int32,
  (uint8, int64): int64,
  (uint8, uint8): uint8,
  (uint8, uint16): uint16,
  (uint8, uint32): uint32,
  (uint8, uint64): uint64,
  (uint16, int8): int32,
  (uint16, int16): int32,
  (uint16, int32): int32,
  (uint16, int64): int64,
  (uint16, uint8): uint16,
  (uint16, uint16): uint16,
  (uint16, uint32): uint32,
  (uint16, uint64): uint64,
  (uint32, int8): int64,
  (uint32, int16): int64,
  (uint32, int32): int64,
  (uint32, int64): int64,
  (uint32, uint8): uint32,
  (uint32, uint16): uint32,
  (uint32, uint32): uint32,
  (uint32, uint64): uint64,
  (uint64, uint8): uint64,
  (uint64, uint16): uint64,
  (uint64, uint32): uint64,
  (uint64, uint64): uint64,
  (float32, float32): float32,
  (float32, float64): float64,
  (float32, complex64): complex64,
  (float32, complex128): complex128,
  (float64, float32): float64,
  (float64, float64): float64,
  (float64, complex64): complex128,
  (float64, complex128): complex128,
  (complex64, float32): complex64,
  (complex64, float64): complex128,
  (complex64, complex64): complex64,
  (complex64, complex128): complex128,
  (complex128, float32): complex128,
  (complex128, float64): complex128,
  (complex128, complex64): complex128,
  (complex128, complex128): complex128,
}


def _is_valid_dtype(t):
  try:
    return t in _valid_dtypes
  except TypeError:
    return False


def _promote_types(t1, t2):
  if not _is_valid_dtype(t1):
    raise ValueError(f"{t1} is not a valid dtype")
  if not _is_valid_dtype(t2):
    raise ValueError(f"{t2} is not a valid dtype")
  if result := _promotion_table.get((t1, t2), None):
    return result
  else:
    raise ValueError("No promotion path for {t1} & {t2}")


def astype(x, dtype, /, *, copy=True):
  return jnp.array(x, dtype=dtype, copy=copy)


def can_cast(from_, to, /):
  if isinstance(from_, jax.Array):
    from_ = from_.dtype
  if not _is_valid_dtype(from_):
    raise ValueError(f"{from_} is not a valid dtype")
  if not _is_valid_dtype(to):
    raise ValueError(f"{to} is not a valid dtype")
  try:
    result = _promote_types(from_, to)
  except ValueError:
    return False
  else:
    return result == to


class FInfo(NamedTuple):
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: jnp.dtype


class IInfo(NamedTuple):
    bits: int
    max: int
    min: int
    dtype: jnp.dtype


def finfo(type, /) -> FInfo:
  info = jnp.finfo(type)
  return FInfo(
    bits=info.bits,
    eps=float(info.eps),
    max=float(info.max),
    min=float(info.min),
    smallest_normal=float(info.smallest_normal),
    dtype=jnp.dtype(type)
  )


def iinfo(type, /) -> IInfo:
  info = jnp.iinfo(type)
  return IInfo(bits=info.bits, max=info.max, min=info.min, dtype=jnp.dtype(type))


def isdtype(dtype, kind):
  return jax.numpy.isdtype(dtype, kind)


def result_type(*arrays_and_dtypes):
  dtypes = []
  for val in arrays_and_dtypes:
    if isinstance(val, (builtins.bool, int, float, complex)):
      val = jax.numpy.array(val)
    if isinstance(val, jax.Array):
      val = val.dtype
    if _is_valid_dtype(val):
      dtypes.append(val)
    else:
      raise ValueError(f"{val} is not a valid dtype")
  if len(dtypes) == 0:
    raise ValueError("result_type requires at least one argument")
  if len(dtypes) == 1:
    return dtypes[0]
  return functools.reduce(_promote_types, dtypes)


def _promote_to_default_dtype(x):
  if x.dtype.kind == 'b':
    return x
  elif x.dtype.kind == 'i':
    return x.astype(jnp.int_)
  elif x.dtype.kind == 'u':
    return x.astype(jnp.uint)
  elif x.dtype.kind == 'f':
    return x.astype(canonicalize_dtype(float))
  elif x.dtype.kind == 'c':
    return x.astype(canonicalize_dtype(complex))
  else:
    raise ValueError(f"Unrecognized {x.dtype=}")
