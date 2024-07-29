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
  '2023.12'

  >>> arr = xp.arange(1000)

  >>> arr.sum()
  Array(499500, dtype=int32)

The ``xp`` namespace is the array API compliant analog of :mod:`jax.numpy`,
and implements most of the API listed in the standard.

.. _Python array API standard: https://data-apis.org/array-api/latest/
"""

from __future__ import annotations

from jax.experimental.array_api._version import __array_api_version__ as __array_api_version__

from jax.experimental.array_api import fft as fft
from jax.experimental.array_api import linalg as linalg

from jax.numpy import (
    abs as abs,
    acos as acos,
    acosh as acosh,
    add as add,
    all as all,
    any as any,
    arange as arange,
    argmax as argmax,
    argmin as argmin,
    argsort as argsort,
    asarray as asarray,
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
    bool as bool,
    broadcast_arrays as broadcast_arrays,
    broadcast_to as broadcast_to,
    can_cast as can_cast,
    ceil as ceil,
    complex128 as complex128,
    complex64 as complex64,
    concat as concat,
    conj as conj,
    copysign as copysign,
    cos as cos,
    cosh as cosh,
    cumulative_sum as cumulative_sum,
    divide as divide,
    e as e,
    empty as empty,
    empty_like as empty_like,
    equal as equal,
    exp as exp,
    expand_dims as expand_dims,
    expm1 as expm1,
    eye as eye,
    finfo as finfo,
    flip as flip,
    float32 as float32,
    float64 as float64,
    floor as floor,
    floor_divide as floor_divide,
    from_dlpack as from_dlpack,
    full as full,
    full_like as full_like,
    greater as greater,
    greater_equal as greater_equal,
    iinfo as iinfo,
    imag as imag,
    inf as inf,
    int16 as int16,
    int32 as int32,
    int64 as int64,
    int8 as int8,
    isdtype as isdtype,
    isfinite as isfinite,
    isinf as isinf,
    isnan as isnan,
    less as less,
    less_equal as less_equal,
    linspace as linspace,
    log as log,
    log10 as log10,
    log1p as log1p,
    log2 as log2,
    logaddexp as logaddexp,
    logical_and as logical_and,
    logical_not as logical_not,
    logical_or as logical_or,
    logical_xor as logical_xor,
    matmul as matmul,
    matrix_transpose as matrix_transpose,
    max as max,
    maximum as maximum,
    mean as mean,
    meshgrid as meshgrid,
    min as min,
    minimum as minimum,
    moveaxis as moveaxis,
    multiply as multiply,
    nan as nan,
    negative as negative,
    newaxis as newaxis,
    nonzero as nonzero,
    not_equal as not_equal,
    ones as ones,
    ones_like as ones_like,
    permute_dims as permute_dims,
    pi as pi,
    positive as positive,
    pow as pow,
    prod as prod,
    real as real,
    remainder as remainder,
    repeat as repeat,
    result_type as result_type,
    roll as roll,
    round as round,
    searchsorted as searchsorted,
    sign as sign,
    signbit as signbit,
    sin as sin,
    sinh as sinh,
    sort as sort,
    sqrt as sqrt,
    square as square,
    squeeze as squeeze,
    stack as stack,
    std as std,
    subtract as subtract,
    sum as sum,
    take as take,
    tan as tan,
    tanh as tanh,
    tensordot as tensordot,
    tile as tile,
    tril as tril,
    triu as triu,
    trunc as trunc,
    uint16 as uint16,
    uint32 as uint32,
    uint64 as uint64,
    uint8 as uint8,
    unique_all as unique_all,
    unique_counts as unique_counts,
    unique_inverse as unique_inverse,
    unique_values as unique_values,
    unstack as unstack,
    var as var,
    vecdot as vecdot,
    where as where,
    zeros as zeros,
    zeros_like as zeros_like,
)

from jax.experimental.array_api._manipulation_functions import (
    reshape as reshape,
)

from jax.experimental.array_api._data_type_functions import (
    astype as astype,
)

from jax.experimental.array_api._elementwise_functions import (
    clip as clip,
    hypot as hypot,
)

from jax.experimental.array_api._utility_functions import (
    __array_namespace_info__ as __array_namespace_info__,
)

from jax.experimental.array_api import _array_methods
_array_methods.add_array_object_methods()
del _array_methods
