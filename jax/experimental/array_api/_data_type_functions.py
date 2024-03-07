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

from __future__ import annotations

import builtins
from typing import NamedTuple
import numpy as np

import jax.numpy as jnp

from jax._src.lib import xla_client as xc
from jax._src.sharding import Sharding
from jax._src import dtypes as _dtypes

# TODO(micky774): Update jax.numpy dtypes to dtype *objects*
bool = np.dtype('bool')
int8 = np.dtype('int8')
int16 = np.dtype('int16')
int32 = np.dtype('int32')
int64 = np.dtype('int64')
uint8 = np.dtype('uint8')
uint16 = np.dtype('uint16')
uint32 = np.dtype('uint32')
uint64 = np.dtype('uint64')
float32 = np.dtype('float32')
float64 = np.dtype('float64')
complex64 = np.dtype('complex64')
complex128 = np.dtype('complex128')


# TODO(micky774): Remove when jax.numpy.astype is deprecation is completed
def astype(x, dtype, /, *, copy: builtins.bool = True, device: xc.Device | Sharding | None = None):
  src_dtype = x.dtype if hasattr(x, "dtype") else _dtypes.dtype(x)
  if (
    src_dtype is not None
    and _dtypes.isdtype(src_dtype, "complex floating")
    and _dtypes.isdtype(dtype, ("integral", "real floating"))
  ):
    raise ValueError(
      "Casting from complex to non-complex dtypes is not permitted. Please "
      "first use jnp.real or jnp.imag to take the real/imaginary component of "
      "your input."
    )
  return jnp.astype(x, dtype, copy=copy, device=device)


class FInfo(NamedTuple):
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: jnp.dtype

# TODO(micky774): Update jax.numpy.finfo so that its attributes are python
# floats
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
