# Copyright 2022 The JAX Authors.
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

"""
`jax._src.typing`: JAX type annotations
---------------------------------------

This submodule is a work in progress; when we finalize the contents here, it will be
exported at `jax.typing`. Until then, the contents here should be considered unstable
and may change without notice.

To see the proposal that led to the development of these tools, see
https://github.com/google/jax/pull/11859/.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Union
import numpy as np

from jax._src.basearray import (
    Array as Array,
    ArrayLike as ArrayLike,
)

DType = np.dtype

# TODO(jakevdp, froystig): make ExtendedDType a protocol
ExtendedDType = Any

class SupportsDType(Protocol):
  @property
  def dtype(self) -> DType: ...

# DTypeLike is meant to annotate inputs to np.dtype that return
# a valid JAX dtype. It's different than numpy.typing.DTypeLike
# because JAX doesn't support objects or structured dtypes.
# Unlike np.typing.DTypeLike, we exclude None, and instead require
# explicit annotations when None is acceptable.
# TODO(jakevdp): consider whether to add ExtendedDtype to the union.
DTypeLike = Union[
  str,            # like 'float32', 'int32'
  type[Any],      # like np.float32, np.int32, float, int
  np.dtype,       # like np.dtype('float32'), np.dtype('int32')
  SupportsDType,  # like jnp.float32, jnp.int32
]

# Shapes are tuples of dimension sizes, which are normally integers. We allow
# modules to extend the set of dimension sizes to contain other types, e.g.,
# symbolic dimensions in jax2tf.shape_poly.DimVar and masking.Poly.
DimSize = Union[int, Any]  # extensible
Shape = Sequence[DimSize]

class DuckTypedArray(Protocol):
  @property
  def dtype(self) -> DType: ...
  @property
  def shape(self) -> Shape: ...

# Array is a type annotation for standard JAX arrays and tracers produced by
# core functions in jax.lax and jax.numpy; it is not meant to include
# future non-standard array types like KeyArray and BInt. It is imported above.

# ArrayLike is a Union of all objects that can be implicitly converted to a standard
# JAX array (i.e. not including future non-standard array types like KeyArray and BInt).
# It's different than np.typing.ArrayLike in that it doesn't accept arbitrary sequences,
# nor does it accept string data.

class ScalarType(Protocol):
  """Protocol for annotating jax.numpy scalar types

  jax.numpy scalar types are analogs of numpy scalar
  types, and include :class:`jax.numpy.int32`,
  :class:`jax.numpy.float32`, etc.

  Example::

    import jax
    import jax.numpy as jnp

    typ: ScalarType = jnp.float32

    # These objects are convertible to numpy dtypes
    # ... either explicitly:
    dt: np.dtype = np.dtype(typ)

    # ... or implicitly
    arr: jax.Array = jnp.arange(10, dtype=typ)

    # They are callable, returning a jax.Array
    out: jax.Array = typ(1)

    # They are hashable
    typedict: dict[ScalarType, np.dtype] = {
      jnp.float32: np.dtype('float32'),
      jnp.int32: np.dtype('int32')
    }
  """
  @property
  def dtype(self) -> DType: ...
  def __call__(self, x: Any) -> Array: ...
  def __hash__(self) -> int: ...
