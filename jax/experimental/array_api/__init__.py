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

"""
This module includes experimental JAX support for the `Python array API standard`_.
Support for this is currently experimental and not fully complete.

Example Usage::

  >>> from jax.experimental import array_api as xp

  >>> xp.__array_api_version__
  '2022.12'

  >>> arr = xp.arange(1000)

  >>> arr.sum()
  Array(499500, dtype=int32)

The ``xp`` namespace is the array API compliant analog of :mod:`jax.numpy`,
and implements most of the API listed in the standard.

.. _Python array API standard: https://data-apis.org/array-api/latest/
"""

from __future__ import annotations

from jax.experimental.array_api._version import __array_api_version__ as __array_api_version__

from jax.experimental.array_api import (
    fft as fft,
    linalg as linalg,
)

from jax.experimental.array_api._constants import (
    e as e,
    inf as inf,
    nan as nan,
    newaxis as newaxis,
    pi as pi,
)

from jax.experimental.array_api._creation_functions import (
    arange as arange,
    asarray as asarray,
    empty as empty,
    empty_like as empty_like,
    eye as eye,
    from_dlpack as from_dlpack,
    full as full,
    full_like as full_like,
    linspace as linspace,
    meshgrid as meshgrid,
    ones as ones,
    ones_like as ones_like,
    tril as tril,
    triu as triu,
    zeros as zeros,
    zeros_like as zeros_like,
)

from jax.experimental.array_api._data_type_functions import (
    astype as astype,
    can_cast as can_cast,
    finfo as finfo,
    iinfo as iinfo,
    isdtype as isdtype,
    result_type as result_type,
)

from jax.experimental.array_api._dtypes import (
    bool as bool,
    int8 as int8,
    int16 as int16,
    int32 as int32,
    int64 as int64,
    uint8 as uint8,
    uint16 as uint16,
    uint32 as uint32,
    uint64 as uint64,
    float32 as float32,
    float64 as float64,
    complex64 as complex64,
    complex128 as complex128,
)

from jax.experimental.array_api._elementwise_functions import (
    abs as abs,
    acos as acos,
    acosh as acosh,
    add as add,
    asin as asin,
    asinh as asinh,
    atan as atan,
    atan2 as atan2,
    atanh as atanh,
    bitwise_and as bitwise_and,
    bitwise_invert as bitwise_invert,
    bitwise_left_shift as bitwise_left_shift,
    bitwise_or as bitwise_or,
    bitwise_right_shift as bitwise_right_shift,
    bitwise_xor as bitwise_xor,
    ceil as ceil,
    conj as conj,
    cos as cos,
    cosh as cosh,
    divide as divide,
    equal as equal,
    exp as exp,
    expm1 as expm1,
    floor as floor,
    floor_divide as floor_divide,
    greater as greater,
    greater_equal as greater_equal,
    imag as imag,
    isfinite as isfinite,
    isinf as isinf,
    isnan as isnan,
    less as less,
    less_equal as less_equal,
    log as log,
    log10 as log10,
    log1p as log1p,
    log2 as log2,
    logaddexp as logaddexp,
    logical_and as logical_and,
    logical_not as logical_not,
    logical_or as logical_or,
    logical_xor as logical_xor,
    multiply as multiply,
    negative as negative,
    not_equal as not_equal,
    positive as positive,
    pow as pow,
    real as real,
    remainder as remainder,
    round as round,
    sign as sign,
    sin as sin,
    sinh as sinh,
    sqrt as sqrt,
    square as square,
    subtract as subtract,
    tan as tan,
    tanh as tanh,
    trunc as trunc,
)

from jax.experimental.array_api._indexing_functions import (
    take as take,
)

from jax.experimental.array_api._manipulation_functions import (
    broadcast_arrays as broadcast_arrays,
    broadcast_to as broadcast_to,
    concat as concat,
    expand_dims as expand_dims,
    flip as flip,
    permute_dims as permute_dims,
    reshape as reshape,
    roll as roll,
    squeeze as squeeze,
    stack as stack,
)

from jax.experimental.array_api._searching_functions import (
    argmax as argmax,
    argmin as argmin,
    nonzero as nonzero,
    where as where,
)

from jax.experimental.array_api._set_functions import (
    unique_all as unique_all,
    unique_counts as unique_counts,
    unique_inverse as unique_inverse,
    unique_values as unique_values,
)

from jax.experimental.array_api._sorting_functions import (
    argsort as argsort,
    sort as sort,
)

from jax.experimental.array_api._statistical_functions import (
    max as max,
    mean as mean,
    min as min,
    prod as prod,
    std as std,
    sum as sum,
    var as var
)

from jax.experimental.array_api._utility_functions import (
    __array_namespace_info__ as __array_namespace_info__,
    all as all,
    any as any,
)

from jax.experimental.array_api._linear_algebra_functions import (
    matmul as matmul,
    matrix_transpose as matrix_transpose,
    tensordot as tensordot,
    vecdot as vecdot,
)

from jax.experimental.array_api import _array_methods
_array_methods.add_array_object_methods()
del _array_methods
