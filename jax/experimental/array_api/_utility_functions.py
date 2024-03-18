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

import jax
from typing import Tuple
from jax._src.sharding import Sharding
from jax._src.lib import xla_client as xc
from jax._src import dtypes as _dtypes, config

def all(x, /, *, axis=None, keepdims=False):
  """Tests whether all input array elements evaluate to True along a specified axis."""
  return jax.numpy.all(x, axis=axis, keepdims=keepdims)


def any(x, /, *, axis=None, keepdims=False):
  """Tests whether any input array element evaluates to True along a specified axis."""
  return jax.numpy.any(x, axis=axis, keepdims=keepdims)

class __array_namespace_info__:

  def __init__(self):
    self._default_dtypes = self._build_default_dtype_dict()
    self._capabilities = {
      "boolean indexing": True,
      "data-dependent shapes": False,
    }
    self._data_types = self._build_dtype_dict()

  def _build_default_dtype_dict(self):
    default_dtypes = {
      "real floating": "f",
      "complex floating": "c",
      "integral": "i",
      "indexing": "i",
    }
    for dtype_name, kind in default_dtypes.items():
      dtype = _dtypes._default_types.get(kind)
      dtype = _dtypes.canonicalize_dtype(dtype)
      default_dtypes[dtype_name] = dtype
    return default_dtypes

  def _build_dtype_dict(self):
    data_types = {
      "signed integer": ["int8", "int16", "int32", "int64"],
      "unsigned integer": ["uint8", "uint16", "uint32", "uint64"],
      "real floating": ["float32", "float64"],
      "complex floating": ["complex64", "complex128"],
    }
    if not config.enable_x64.value:
      for category in data_types:
        data_types[category] = data_types[category][:-1]

    data_types["bool"] = ["bool"]

    for category in data_types:
      _dtype_dict = {}
      for name in data_types[category]:
        _dtype_dict[name] = _dtypes.dtype(name)
      data_types[category] = _dtype_dict
    data_types["integral"] = (
      data_types["signed integer"] | data_types["unsigned integer"]
    )
    data_types["numeric"] = (
      data_types["integral"]
      | data_types["real floating"]
      | data_types["complex floating"]
    )
    return data_types

  def default_device(self):
    # By default JAX arrays are uncommitted (device=None), meaning that
    # JAX is free to choose the most efficient device placement.
    return None

  def devices(self):
    return jax.devices()

  def capabilities(self):
    return self._capabilities

  def default_dtypes(self):
    return self._default_dtypes

  def dtypes(self, *, device: xc.Device | Sharding | None = None, kind: str | Tuple[str, ...] | None = None):
    # Array API supported dtypes are device-independent in JAX
    if kind is None:
      out_dict = self._data_types["numeric"] | self._data_types["bool"]
    elif isinstance(kind, tuple):
      out_dict = {}
      for _kind in kind:
        out_dict |= self._data_types[_kind]
    else:
      out_dict = self._data_types[kind]
    return out_dict
