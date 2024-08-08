# Copyright 2018 The JAX Authors.
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

from functools import partial

import numpy as np

from jax._src import core
from jax._src import dtypes

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

UnshapedArray = core.UnshapedArray
ShapedArray = core.ShapedArray
ConcreteArray = core.ConcreteArray
AbstractToken = core.AbstractToken
abstract_token = core.abstract_token
canonicalize_shape = core.canonicalize_shape
raise_to_shaped = core.raise_to_shaped

numpy_scalar_types: set[type] = {  # pylint: disable=g-bare-generic
    dtypes.int4, np.int8, np.int16, np.int32, np.int64,
    dtypes.uint4, np.uint8, np.uint16, np.uint32, np.uint64,
    np.complex64, np.complex128,
    np.bool_, np.longlong, np.intc,
} | {np.dtype(dt).type for dt in dtypes._float_types}

if dtypes.int2 is not None:
  assert dtypes.uint2 is not None
  numpy_scalar_types.add(dtypes.int2)
  numpy_scalar_types.add(dtypes.uint2)

array_types: set[type] = {np.ndarray} | numpy_scalar_types  # pylint: disable=g-bare-generic

def canonical_concrete_aval(val, weak_type=None):
  return ConcreteArray(dtypes.canonicalize_dtype(np.result_type(val)), val,
                       weak_type=weak_type)

def masked_array_error(*args, **kwargs):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")

core.pytype_aval_mappings[np.ma.MaskedArray] = masked_array_error

for t in array_types:
  core.pytype_aval_mappings[t] = canonical_concrete_aval

core.literalable_types.update(array_types)

def _make_concrete_python_scalar(t, x):
  dtype = dtypes._scalar_type_to_dtype(t, x)
  weak_type = dtypes.is_weakly_typed(x)
  return canonical_concrete_aval(np.array(x, dtype=dtype), weak_type=weak_type)

for t in dtypes.python_scalar_dtypes:
  core.pytype_aval_mappings[t] = partial(_make_concrete_python_scalar, t)

core.literalable_types.update(dtypes.python_scalar_dtypes.keys())
