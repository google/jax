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

# pytype: skip-file
"""
Implements the NumPy API, using the primitives in :mod:`jax.lax`.

NumPy operations are implemented in Python in terms of the primitive operations
in :mod:`jax.lax`. Since NumPy operations are not primitive and instead are
implemented in terms of :mod:`jax.lax` operations, we do not need to define
transformation rules such as gradient or batching rules. Instead,
transformations for NumPy primitives can be derived from the transformation
rules for the underlying :code:`lax` primitives.
"""
from __future__ import annotations

import builtins
import collections
from collections.abc import Callable, Sequence
from functools import partial
import importlib
import math
import operator
import types
from typing import (overload, Any, Literal, NamedTuple,
                    Protocol, TypeVar, Union)
from textwrap import dedent as _dedent
import warnings

import numpy as np
import opt_einsum

import jax
from jax import jit
from jax import errors
from jax import lax
from jax.sharding import Sharding, SingleDeviceSharding
from jax.tree_util import tree_leaves, tree_flatten, tree_map

from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src.custom_derivatives import custom_jvp
from jax._src import deprecations
from jax._src import dispatch
from jax._src import dtypes
from jax._src import xla_bridge
from jax._src.api_util import _ensure_index_tuple
from jax._src.array import ArrayImpl
from jax._src.core import ShapedArray, ConcreteArray
from jax._src.lax.lax import (_array_copy, _sort_lt_comparator,
                              _sort_le_comparator, PrecisionLike)
from jax._src.lax import lax as lax_internal
from jax._src.lib import xla_client as xc
from jax._src.numpy import reductions
from jax._src.numpy import ufuncs
from jax._src.numpy import util
from jax._src.numpy.vectorize import vectorize
from jax._src.typing import (
  Array, ArrayLike, DeprecatedArg, DimSize, DuckTypedArray,
  DType, DTypeLike, Shape, StaticScalar,
)
from jax._src.util import (unzip2, subvals, safe_zip,
                           ceil_of_ratio, partition_list,
                           canonicalize_axis as _canonicalize_axis,
                           NumpyComplexWarning)

for pkg_name in ['jax_cuda12_plugin', 'jax.jaxlib']:
  try:
    cuda_plugin_extension = importlib.import_module(
        f'{pkg_name}.cuda_plugin_extension'
    )
  except ImportError:
    cuda_plugin_extension = None  # type: ignore
  else:
    break

newaxis = None
T = TypeVar('T')


# Like core.canonicalize_shape, but also accept int-like (non-sequence)
# arguments for `shape`.
def canonicalize_shape(shape: Any, context: str="") -> core.Shape:
  if (not isinstance(shape, (tuple, list)) and
      (getattr(shape, 'ndim', None) == 0 or ndim(shape) == 0)):
    return core.canonicalize_shape((shape,), context)
  else:
    return core.canonicalize_shape(shape, context)

# Common docstring additions:

_PRECISION_DOC = """\
In addition to the original NumPy arguments listed below, also supports
``precision`` for extra control over matrix-multiplication precision
on supported devices. ``precision`` may be set to ``None``, which means
default precision for the backend, a :class:`~jax.lax.Precision` enum value
(``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple
of two :class:`~jax.lax.Precision` enums indicating separate precision for each argument.
"""

# Some objects below rewrite their __module__ attribute to this name.
_PUBLIC_MODULE_NAME = "jax.numpy"

# NumPy constants

pi = np.pi
e = np.e
euler_gamma = np.euler_gamma
inf = np.inf
nan = np.nan

# NumPy utility functions

get_printoptions = np.get_printoptions
printoptions = np.printoptions
set_printoptions = np.set_printoptions

def iscomplexobj(x: Any) -> bool:
  """Check if the input is a complex number or an array containing complex elements.

  JAX implementation of :func:`numpy.iscomplexobj`.

  The function evaluates based on input type rather than value.
  Inputs with zero imaginary parts are still considered complex.

  Args:
    x: input object to check.

  Returns:
    True if ``x`` is a complex number or an array containing at least one complex element,
    False otherwise.

  See Also:
    - :func:`jax.numpy.isrealobj`
    - :func:`jax.numpy.iscomplex`

  Examples:
    >>> jnp.iscomplexobj(True)
    False
    >>> jnp.iscomplexobj(0)
    False
    >>> jnp.iscomplexobj(jnp.array([1, 2]))
    False
    >>> jnp.iscomplexobj(1+2j)
    True
    >>> jnp.iscomplexobj(jnp.array([0, 1+2j]))
    True
  """
  if x is None:
    return False
  try:
    typ = x.dtype.type
  except AttributeError:
    typ = asarray(x).dtype.type
  return issubdtype(typ, complexfloating)

shape = _shape = np.shape
ndim = _ndim = np.ndim
size = np.size

def _dtype(x: Any) -> DType:
  return dtypes.dtype(x, canonicalize=True)

# At present JAX doesn't have a reason to distinguish between scalars and arrays
# in its object system. Further, we want JAX scalars to have the same type
# promotion behaviors as JAX arrays. Rather than introducing a new type of JAX
# scalar object with JAX promotion behaviors, instead we make the JAX scalar
# types return JAX arrays when instantiated.

class _ScalarMeta(type):
  dtype: np.dtype

  def __hash__(self) -> int:
    return hash(self.dtype.type)

  def __eq__(self, other: Any) -> bool:
    return id(self) == id(other) or self.dtype.type == other

  def __ne__(self, other: Any) -> bool:
    return not (self == other)

  def __call__(self, x: Any) -> Array:
    return asarray(x, dtype=self.dtype)

  def __instancecheck__(self, instance: Any) -> bool:
    return isinstance(instance, self.dtype.type)

def _abstractify_scalar_meta(x):
  raise TypeError(f"JAX scalar type {x} cannot be interpreted as a JAX array.")
api_util._shaped_abstractify_handlers[_ScalarMeta] = _abstractify_scalar_meta

def _make_scalar_type(np_scalar_type: type) -> _ScalarMeta:
  meta = _ScalarMeta(np_scalar_type.__name__, (object,),
                     {"dtype": np.dtype(np_scalar_type)})
  meta.__module__ = _PUBLIC_MODULE_NAME
  return meta

bool_ = _make_scalar_type(np.bool_)
if dtypes.uint2 is not None:
  uint2 = _make_scalar_type(dtypes.uint2)
uint4 = _make_scalar_type(dtypes.uint4)
uint8 = _make_scalar_type(np.uint8)
uint16 = _make_scalar_type(np.uint16)
uint32 = _make_scalar_type(np.uint32)
uint64 = _make_scalar_type(np.uint64)
if dtypes.int2 is not None:
  int2 = _make_scalar_type(dtypes.int2)
int4 = _make_scalar_type(dtypes.int4)
int8 = _make_scalar_type(np.int8)
int16 = _make_scalar_type(np.int16)
int32 = _make_scalar_type(np.int32)
int64 = _make_scalar_type(np.int64)
float8_e4m3fn = _make_scalar_type(dtypes.float8_e4m3fn)
float8_e4m3fnuz = _make_scalar_type(dtypes.float8_e4m3fnuz)
float8_e5m2 = _make_scalar_type(dtypes.float8_e5m2)
float8_e5m2fnuz = _make_scalar_type(dtypes.float8_e5m2fnuz)
float8_e4m3b11fnuz = _make_scalar_type(dtypes.float8_e4m3b11fnuz)
bfloat16 = _make_scalar_type(dtypes.bfloat16)
float16 = _make_scalar_type(np.float16)
float32 = single = _make_scalar_type(np.float32)
float64 = double = _make_scalar_type(np.float64)
complex64 = csingle = _make_scalar_type(np.complex64)
complex128 = cdouble = _make_scalar_type(np.complex128)

int_ = int32 if dtypes.int_ == np.int32 else int64
uint = uint32 if dtypes.uint == np.uint32 else uint64
float_: Any = float32 if dtypes.float_ == np.float32 else float64
complex_ = complex64 if dtypes.complex_ == np.complex64 else complex128

generic = np.generic
number = np.number
inexact = np.inexact
complexfloating = np.complexfloating
floating = np.floating
integer = np.integer
signedinteger = np.signedinteger
unsignedinteger = np.unsignedinteger

flexible = np.flexible
character = np.character
object_ = np.object_

iinfo = dtypes.iinfo
finfo = dtypes.finfo

dtype = np.dtype
can_cast = dtypes.can_cast
promote_types = dtypes.promote_types

ComplexWarning = NumpyComplexWarning

array_str = np.array_str
array_repr = np.array_repr

save = np.save
savez = np.savez

@util.implements(np.dtype)
def _jnp_dtype(obj: DTypeLike | None, *, align: bool = False,
               copy: bool = False) -> DType:
  """Similar to np.dtype, but respects JAX dtype defaults."""
  if dtypes.issubdtype(obj, dtypes.extended):
    return obj  # type: ignore[return-value]
  if obj is None:
    obj = dtypes.float_
  elif isinstance(obj, type) and obj in dtypes.python_scalar_dtypes:
    obj = _DEFAULT_TYPEMAP[obj]
  return np.dtype(obj, align=align, copy=copy)

### utility functions

_DEFAULT_TYPEMAP: dict[type, _ScalarMeta] = {
  bool: bool_,
  int: int_,
  float: float_,
  complex: complex_,
}

_lax_const = lax_internal._const


def _convert_and_clip_integer(val: ArrayLike, dtype: DType) -> Array:
  """
  Convert integer-typed val to specified integer dtype, clipping to dtype
  range rather than wrapping.

  Args:
    val: value to be converted
    dtype: dtype of output

  Returns:
    equivalent of val in new dtype

  Examples
  --------
  Normal integer type conversion will wrap:

  >>> val = jnp.uint32(0xFFFFFFFF)
  >>> val.astype('int32')
  Array(-1, dtype=int32)

  This function clips to the values representable in the new type:

  >>> _convert_and_clip_integer(val, 'int32')
  Array(2147483647, dtype=int32)
  """
  val = val if isinstance(val, Array) else asarray(val)
  dtype = dtypes.canonicalize_dtype(dtype)
  if not (issubdtype(dtype, integer) and issubdtype(val.dtype, integer)):
    raise TypeError("_convert_and_clip_integer only accepts integer dtypes.")

  val_dtype = dtypes.canonicalize_dtype(val.dtype)
  if val_dtype != val.dtype:
    # TODO(jakevdp): this is a weird corner case; need to figure out how to handle it.
    # This happens in X32 mode and can either come from a jax value created in another
    # context, or a Python integer converted to int64.
    pass
  min_val = _lax_const(val, max(iinfo(dtype).min, iinfo(val_dtype).min))
  max_val = _lax_const(val, min(iinfo(dtype).max, iinfo(val_dtype).max))
  return clip(val, min_val, max_val).astype(dtype)


@util.implements(np.load, update_doc=False)
def load(*args: Any, **kwargs: Any) -> Array:
  # The main purpose of this wrapper is to recover bfloat16 data types.
  # Note: this will only work for files created via np.save(), not np.savez().
  out = np.load(*args, **kwargs)
  if isinstance(out, np.ndarray):
    # numpy does not recognize bfloat16, so arrays are serialized as void16
    if out.dtype == 'V2':
      out = out.view(bfloat16)
    try:
      out = asarray(out)
    except (TypeError, AssertionError):  # Unsupported dtype
      pass
  return out

### implementations of numpy functions in terms of lax

@util.implements(np.fmin, module='numpy')
@jit
def fmin(x1: ArrayLike, x2: ArrayLike) -> Array:
  return where(ufuncs.less(x1, x2) | ufuncs.isnan(x2), x1, x2)

@util.implements(np.fmax, module='numpy')
@jit
def fmax(x1: ArrayLike, x2: ArrayLike) -> Array:
  return where(ufuncs.greater(x1, x2) | ufuncs.isnan(x2), x1, x2)

@util.implements(np.issubdtype)
def issubdtype(arg1: DTypeLike, arg2: DTypeLike) -> bool:
  return dtypes.issubdtype(arg1, arg2)

@util.implements(np.isscalar)
def isscalar(element: Any) -> bool:
  if hasattr(element, '__jax_array__'):
    element = element.__jax_array__()
  return dtypes.is_python_scalar(element) or np.isscalar(element)

iterable = np.iterable

@util.implements(np.result_type)
def result_type(*args: Any) -> DType:
  return dtypes.result_type(*args)


@util.implements(np.trunc, module='numpy')
@jit
def trunc(x: ArrayLike) -> Array:
  util.check_arraylike('trunc', x)
  if dtypes.isdtype(dtypes.dtype(x), ('integral', 'bool')):
    return lax_internal.asarray(x)
  return where(lax.lt(x, _lax_const(x, 0)), ufuncs.ceil(x), ufuncs.floor(x))


_CONV_PREFERRED_ELEMENT_TYPE_DESCRIPTION = """
preferred_element_type : dtype, optional
    If specified, accumulate results and return a result of the given data type.
    If not specified, the function instead follows the numpy convention of always
    accumulating results and returning an inexact dtype.
"""

@partial(jit, static_argnames=['mode', 'op', 'precision', 'preferred_element_type'])
def _conv(x: Array, y: Array, mode: str, op: str, precision: PrecisionLike,
          preferred_element_type: DTypeLike | None = None) -> Array:
  if ndim(x) != 1 or ndim(y) != 1:
    raise ValueError(f"{op}() only support 1-dimensional inputs.")
  if preferred_element_type is None:
    # if unspecified, promote to inexact following NumPy's default for convolutions.
    x, y = util.promote_dtypes_inexact(x, y)
  else:
    # otherwise cast to same type but otherwise preserve input dtypes
    x, y = util.promote_dtypes(x, y)
  if len(x) == 0 or len(y) == 0:
    raise ValueError(f"{op}: inputs cannot be empty, got shapes {x.shape} and {y.shape}.")

  out_order = slice(None)
  if op == 'correlate':
    y = ufuncs.conj(y)
    if len(x) < len(y):
      x, y = y, x
      out_order = slice(None, None, -1)
  elif op == 'convolve':
    if len(x) < len(y):
      x, y = y, x
    y = flip(y)

  if mode == 'valid':
    padding = [(0, 0)]
  elif mode == 'same':
    padding = [(y.shape[0] // 2, y.shape[0] - y.shape[0] // 2 - 1)]
  elif mode == 'full':
    padding = [(y.shape[0] - 1, y.shape[0] - 1)]
  else:
    raise ValueError("mode must be one of ['full', 'same', 'valid']")

  result = lax.conv_general_dilated(x[None, None, :], y[None, None, :], (1,),
                                    padding, precision=precision,
                                    preferred_element_type=preferred_element_type)
  return result[0, 0, out_order]


@partial(jit, static_argnames=('mode', 'precision', 'preferred_element_type'))
def convolve(a: ArrayLike, v: ArrayLike, mode: str = 'full', *,
             precision: PrecisionLike = None,
             preferred_element_type: DTypeLike | None = None) -> Array:
  r"""Convolution of two one dimensional arrays.

  JAX implementation of :func:`numpy.convolve`.

  Convolution of one dimensional arrays is defined as:

  .. math::

     c_k = \sum_j a_{k - j} v_j

  Args:
    a: left-hand input to the convolution. Must have ``a.ndim == 1``.
    v: right-hand input to the convolution. Must have ``v.ndim == 1``.
    mode: controls the size of the output. Available operations are:

      * ``"full"``: (default) output the full convolution of the inputs.
      * ``"same"``: return a centered portion of the ``"full"`` output which
        is the same size as ``a``.
      * ``"valid"``: return the portion of the ``"full"`` output which do not
        depend on padding at the array edges.

    precision: Specify the precision of the computation. Refer to
      :class:`jax.lax.Precision` for a description of available values.

    preferred_element_type: A datatype, indicating to accumulate results to and
      return a result with that datatype. Default is ``None``, which means the
      default accumulation type for the input types.

  Returns:
    Array containing the convolved result.

  See Also:
    - :func:`jax.scipy.signal.convolve`: ND convolution
    - :func:`jax.numpy.correlate`: 1D correlation

  Examples:
    A few 1D convolution examples:

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([4, 1, 2])

    ``jax.numpy.convolve``, by default, returns full convolution using implicit
    zero-padding at the edges:

    >>> jnp.convolve(x, y)
    Array([ 4.,  9., 16., 15., 12.,  5.,  2.], dtype=float32)

    Specifying ``mode = 'same'`` returns a centered convolution the same size
    as the first input:

    >>> jnp.convolve(x, y, mode='same')
    Array([ 9., 16., 15., 12.,  5.], dtype=float32)

    Specifying ``mode = 'valid'`` returns only the portion where the two arrays
    fully overlap:

    >>> jnp.convolve(x, y, mode='valid')
    Array([16., 15., 12.], dtype=float32)

    For complex-valued inputs:

    >>> x1 = jnp.array([3+1j, 2, 4-3j])
    >>> y1 = jnp.array([1, 2-3j, 4+5j])
    >>> jnp.convolve(x1, y1)
    Array([ 3. +1.j, 11. -7.j, 15.+10.j,  7. -8.j, 31. +8.j], dtype=complex64)
  """
  util.check_arraylike("convolve", a, v)
  return _conv(asarray(a), asarray(v), mode=mode, op='convolve',
               precision=precision, preferred_element_type=preferred_element_type)


@partial(jit, static_argnames=('mode', 'precision', 'preferred_element_type'))
def correlate(a: ArrayLike, v: ArrayLike, mode: str = 'valid', *,
              precision: PrecisionLike = None,
              preferred_element_type: DTypeLike | None = None) -> Array:
  r"""Correlation of two one dimensional arrays.

  JAX implementation of :func:`numpy.correlate`.

  Correlation of one dimensional arrays is defined as:

  .. math::

     c_k = \sum_j a_{k + j} \overline{v_j}

  where :math:`\overline{v_j}` is the complex conjugate of :math:`v_j`.

  Args:
    a: left-hand input to the correlation. Must have ``a.ndim == 1``.
    v: right-hand input to the correlation. Must have ``v.ndim == 1``.
    mode: controls the size of the output. Available operations are:

      * ``"full"``: output the full correlation of the inputs.
      * ``"same"``: return a centered portion of the ``"full"`` output which
        is the same size as ``a``.
      * ``"valid"``: (default) return the portion of the ``"full"`` output which do not
        depend on padding at the array edges.

    precision: Specify the precision of the computation. Refer to
      :class:`jax.lax.Precision` for a description of available values.

    preferred_element_type: A datatype, indicating to accumulate results to and
      return a result with that datatype. Default is ``None``, which means the
      default accumulation type for the input types.

  Returns:
    Array containing the cross-correlation result.

  See Also:
    - :func:`jax.scipy.signal.correlate`: ND correlation
    - :func:`jax.numpy.convolve`: 1D convolution

  Examples:
    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([4, 5, 6])

    Since default ``mode = 'valid'``, ``jax.numpy.correlate`` returns only the
    portion of correlation where the two arrays fully overlap:

    >>> jnp.correlate(x, y)
    Array([32., 35., 28.], dtype=float32)

    Specifying ``mode = 'full'`` returns full correlation using implicit
    zero-padding at the edges.

    >>> jnp.correlate(x, y, mode='full')
    Array([ 6., 17., 32., 35., 28., 13.,  4.], dtype=float32)

    Specifying ``mode = 'same'`` returns a centered correlation the same size
    as the first input:

    >>> jnp.correlate(x, y, mode='same')
    Array([17., 32., 35., 28., 13.], dtype=float32)

    If both the inputs arrays are real-valued and symmetric then the result will
    also be symmetric and will be equal to the result of ``jax.numpy.convolve``.

    >>> x1 = jnp.array([1, 2, 3, 2, 1])
    >>> y1 = jnp.array([4, 5, 4])
    >>> jnp.correlate(x1, y1, mode='full')
    Array([ 4., 13., 26., 31., 26., 13.,  4.], dtype=float32)
    >>> jnp.convolve(x1, y1, mode='full')
    Array([ 4., 13., 26., 31., 26., 13.,  4.], dtype=float32)

    For complex-valued inputs:

    >>> x2 = jnp.array([3+1j, 2, 2-3j])
    >>> y2 = jnp.array([4, 2-5j, 1])
    >>> jnp.correlate(x2, y2, mode='full')
    Array([ 3. +1.j,  3.+17.j, 18.+11.j, 27. +4.j,  8.-12.j], dtype=complex64)
  """
  util.check_arraylike("correlate", a, v)
  return _conv(asarray(a), asarray(v), mode=mode, op='correlate',
               precision=precision, preferred_element_type=preferred_element_type)


@util.implements(np.histogram_bin_edges)
def histogram_bin_edges(a: ArrayLike, bins: ArrayLike = 10,
                        range: None | Array | Sequence[ArrayLike] = None,
                        weights: ArrayLike | None = None) -> Array:
  del weights  # unused, because string bins is not supported.
  if isinstance(bins, str):
    raise NotImplementedError("string values for `bins` not implemented.")
  util.check_arraylike("histogram_bin_edges", a, bins)
  arr = asarray(a)
  dtype = dtypes.to_inexact_dtype(arr.dtype)
  if _ndim(bins) == 1:
    return asarray(bins, dtype=dtype)

  bins_int = core.concrete_or_error(operator.index, bins,
                                    "bins argument of histogram_bin_edges")
  if range is None:
    range = [arr.min(), arr.max()]
  range = asarray(range, dtype=dtype)
  if shape(range) != (2,):
    raise ValueError(f"`range` must be either None or a sequence of scalars, got {range}")
  range = (where(reductions.ptp(range) == 0, range[0] - 0.5, range[0]),
           where(reductions.ptp(range) == 0, range[1] + 0.5, range[1]))
  assert range is not None
  return linspace(range[0], range[1], bins_int + 1, dtype=dtype)


@util.implements(np.histogram)
def histogram(a: ArrayLike, bins: ArrayLike = 10,
              range: Sequence[ArrayLike] | None = None,
              weights: ArrayLike | None = None,
              density: bool | None = None) -> tuple[Array, Array]:
  if weights is None:
    util.check_arraylike("histogram", a, bins)
    a, = util.promote_dtypes_inexact(a)
    weights = ones_like(a)
  else:
    util.check_arraylike("histogram", a, bins, weights)
    if shape(a) != shape(weights):
      raise ValueError("weights should have the same shape as a.")
    a, weights = util.promote_dtypes_inexact(a, weights)

  bin_edges = histogram_bin_edges(a, bins, range, weights)
  bin_idx = searchsorted(bin_edges, a, side='right')
  bin_idx = where(a == bin_edges[-1], len(bin_edges) - 1, bin_idx)
  counts = zeros(len(bin_edges), weights.dtype).at[bin_idx].add(weights)[1:]
  if density:
    bin_widths = diff(bin_edges)
    counts = counts / bin_widths / counts.sum()
  return counts, bin_edges

@util.implements(np.histogram2d)
def histogram2d(x: ArrayLike, y: ArrayLike, bins: ArrayLike | list[ArrayLike] = 10,
                range: Sequence[None | Array | Sequence[ArrayLike]] | None = None,
                weights: ArrayLike | None = None,
                density: bool | None = None) -> tuple[Array, Array, Array]:
  util.check_arraylike("histogram2d", x, y)
  try:
    N = len(bins)  # type: ignore[arg-type]
  except TypeError:
    N = 1

  if N != 1 and N != 2:
    x_edges = y_edges = asarray(bins)
    bins = [x_edges, y_edges]

  sample = transpose(asarray([x, y]))
  hist, edges = histogramdd(sample, bins, range, weights, density)
  return hist, edges[0], edges[1]

@util.implements(np.histogramdd)
def histogramdd(sample: ArrayLike, bins: ArrayLike | list[ArrayLike] = 10,
                range: Sequence[None | Array | Sequence[ArrayLike]] | None = None,
                weights: ArrayLike | None = None,
                density: bool | None = None) -> tuple[Array, list[Array]]:
  if weights is None:
    util.check_arraylike("histogramdd", sample)
    sample, = util.promote_dtypes_inexact(sample)
  else:
    util.check_arraylike("histogramdd", sample, weights)
    if shape(weights) != shape(sample)[:1]:
      raise ValueError("should have one weight for each sample.")
    sample, weights = util.promote_dtypes_inexact(sample, weights)
  N, D = shape(sample)

  if range is not None and (
      len(range) != D or any(r is not None and shape(r)[0] != 2 for r in range)):  # type: ignore[arg-type]
    raise ValueError(f"For sample.shape={(N, D)}, range must be a sequence "
                     f"of {D} pairs or Nones; got {range=}")

  try:
    num_bins = len(bins)  # type: ignore[arg-type]
  except TypeError:
    # when bin_size is integer, the same bin is used for each dimension
    bins_per_dimension: list[ArrayLike] = D * [bins]  # type: ignore[assignment]
  else:
    if num_bins != D:
      raise ValueError("should be a bin for each dimension.")
    bins_per_dimension = list(bins)  # type: ignore[arg-type]

  bin_idx_by_dim: list[Array] = []
  bin_edges_by_dim: list[Array] = []

  for i in builtins.range(D):
    range_i = None if range is None else range[i]
    bin_edges = histogram_bin_edges(sample[:, i], bins_per_dimension[i], range_i, weights)
    bin_idx = searchsorted(bin_edges, sample[:, i], side='right')
    bin_idx = where(sample[:, i] == bin_edges[-1], bin_idx - 1, bin_idx)
    bin_idx_by_dim.append(bin_idx)
    bin_edges_by_dim.append(bin_edges)

  nbins = tuple(len(bin_edges) + 1 for bin_edges in bin_edges_by_dim)
  dedges = [diff(bin_edges) for bin_edges in bin_edges_by_dim]

  xy = ravel_multi_index(tuple(bin_idx_by_dim), nbins, mode='clip')
  hist = bincount(xy, weights, length=math.prod(nbins))
  hist = reshape(hist, nbins)
  core = D*(slice(1, -1),)
  hist = hist[core]

  if density:
    hist = hist.astype(sample.dtype)
    hist /= hist.sum()
    for norm in ix_(*dedges):
      hist /= norm

  return hist, bin_edges_by_dim


_ARRAY_VIEW_DOC = """
The JAX version of this function may in some cases return a copy rather than a
view of the input.
"""

def transpose(a: ArrayLike, axes: Sequence[int] | None = None) -> Array:
  """Return a transposed version of an N-dimensional array.

  JAX implementation of :func:`numpy.transpose`, implemented in terms of
  :func:`jax.lax.transpose`.

  Args:
    a: input array
    axes: optionally specify the permutation using a length-`a.ndim` sequence of integers
      ``i`` satisfying ``0 <= i < a.ndim``. Defaults to ``range(a.ndim)[::-1]``, i.e.
      reverses the order of all axes.

  Returns:
    transposed copy of the array.

  See Also:
    - :func:`jax.Array.transpose`: equivalent function via an :class:`~jax.Array` method.
    - :attr:`jax.Array.T`: equivalent function via an :class:`~jax.Array`  property.
    - :func:`jax.numpy.matrix_transpose`: transpose the last two axes of an array. This is
      suitable for working with batched 2D matrices.
    - :func:`jax.numpy.swapaxes`: swap any two axes in an array.
    - :func:`jax.numpy.moveaxis`: move an axis to another position in the array.

  Note:
    Unlike :func:`numpy.transpose`, :func:`jax.numpy.transpose` will return a copy rather
    than a view of the input array. However, under JIT, the compiler will optimize-away
    such copies when possible, so this doesn't have performance impacts in practice.

  Examples:
    For a 1D array, the transpose is the identity:

    >>> x = jnp.array([1, 2, 3, 4])
    >>> jnp.transpose(x)
    Array([1, 2, 3, 4], dtype=int32)

    For a 2D array, the transpose is a matrix transpose:

    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.transpose(x)
    Array([[1, 3],
           [2, 4]], dtype=int32)

    For an N-dimensional array, the transpose reverses the order of the axes:

    >>> x = jnp.zeros(shape=(3, 4, 5))
    >>> jnp.transpose(x).shape
    (5, 4, 3)

    The ``axes`` argument can be specified to change this default behavior:

    >>> jnp.transpose(x, (0, 2, 1)).shape
    (3, 5, 4)

    Since swapping the last two axes is a common operation, it can be done
    via its own API, :func:`jax.numpy.matrix_transpose`:

    >>> jnp.matrix_transpose(x).shape
    (3, 5, 4)

    For convenience, transposes may also be performed using the :meth:`jax.Array.transpose`
    method or the :attr:`jax.Array.T` property:

    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> x.transpose()
    Array([[1, 3],
           [2, 4]], dtype=int32)
    >>> x.T
    Array([[1, 3],
           [2, 4]], dtype=int32)
  """
  util.check_arraylike("transpose", a)
  axes_ = list(range(ndim(a))[::-1]) if axes is None else axes
  axes_ = [_canonicalize_axis(i, ndim(a)) for i in axes_]
  return lax.transpose(a, axes_)


@util.implements(getattr(np, "permute_dims", None))
def permute_dims(a: ArrayLike, /, axes: tuple[int, ...]) -> Array:
  util.check_arraylike("permute_dims", a)
  return lax.transpose(a, axes)


def matrix_transpose(x: ArrayLike, /) -> Array:
  """Transpose the last two dimensions of an array.

  JAX implementation of :func:`numpy.matrix_transpose`, implemented in terms of
  :func:`jax.lax.transpose`.

  Args:
    x: input array, Must have ``x.ndim >= 2``

  Returns:
    matrix-transposed copy of the array.

  See Also:
    - :attr:`jax.Array.mT`: same operation accessed via an :func:`~jax.Array` property.
    - :func:`jax.numpy.transpose`: general multi-axis transpose

  Note:
    Unlike :func:`numpy.matrix_transpose`, :func:`jax.numpy.matrix_transpose` will return a
    copy rather than a view of the input array. However, under JIT, the compiler will
    optimize-away such copies when possible, so this doesn't have performance impacts in practice.

  Examples:
    Here is a 2x2x2 matrix representing a batched 2x2 matrix:

    >>> x = jnp.array([[[1, 2],
    ...                 [3, 4]],
    ...                [[5, 6],
    ...                 [7, 8]]])
    >>> jnp.matrix_transpose(x)
    Array([[[1, 3],
            [2, 4]],
    <BLANKLINE>
           [[5, 7],
            [6, 8]]], dtype=int32)

    For convenience, you can perform the same transpose via the :attr:`~jax.Array.mT`
    property of :class:`jax.Array`:

    >>> x.mT
    Array([[[1, 3],
            [2, 4]],
    <BLANKLINE>
           [[5, 7],
            [6, 8]]], dtype=int32)
  """
  util.check_arraylike("matrix_transpose", x)
  ndim = np.ndim(x)
  if ndim < 2:
    raise ValueError(f"x must be at least two-dimensional for matrix_transpose; got {ndim=}")
  axes = (*range(ndim - 2), ndim - 1, ndim - 2)
  return lax.transpose(x, axes)


@util.implements(np.rot90, lax_description=_ARRAY_VIEW_DOC)
@partial(jit, static_argnames=('k', 'axes'))
def rot90(m: ArrayLike, k: int = 1, axes: tuple[int, int] = (0, 1)) -> Array:
  util.check_arraylike("rot90", m)
  if np.ndim(m) < 2:
    raise ValueError("rot90 requires its first argument to have ndim at least "
                     f"two, but got first argument of shape {np.shape(m)}, "
                     f"which has ndim {np.ndim(m)}")
  ax1, ax2 = axes
  ax1 = _canonicalize_axis(ax1, ndim(m))
  ax2 = _canonicalize_axis(ax2, ndim(m))
  if ax1 == ax2:
    raise ValueError("Axes must be different")  # same as numpy error
  k = k % 4
  if k == 0:
    return asarray(m)
  elif k == 2:
    return flip(flip(m, ax1), ax2)
  else:
    perm = list(range(ndim(m)))
    perm[ax1], perm[ax2] = perm[ax2], perm[ax1]
    if k == 1:
      return transpose(flip(m, ax2), perm)
    else:
      return flip(transpose(m, perm), ax2)


def flip(m: ArrayLike, axis: int | Sequence[int] | None = None) -> Array:
  """Reverse the order of elements of an array along the given axis.

  JAX implementation of :func:`numpy.flip`.

  Args:
    m: Array.
    axis: integer or sequence of integers. Specifies along which axis or axes
      should the array elements be reversed. Default is ``None``, which flips
      along all axes.

  Returns:
    An array with the elements in reverse order along ``axis``.

  See Also:
    - :func:`jax.numpy.fliplr`: reverse the order along axis 1 (left/right)
    - :func:`jax.numpy.flipud`: reverse the order along axis 0 (up/down)

  Examples:
    >>> x1 = jnp.array([[1, 2],
    ...                 [3, 4]])
    >>> jnp.flip(x1)
    Array([[4, 3],
           [2, 1]], dtype=int32)

    If ``axis`` is specified with an integer, then ``jax.numpy.flip`` reverses
    the array along that particular axis only.

    >>> jnp.flip(x1, axis=1)
    Array([[2, 1],
           [4, 3]], dtype=int32)

    >>> x2 = jnp.arange(1, 9).reshape(2, 2, 2)
    >>> x2
    Array([[[1, 2],
            [3, 4]],
    <BLANKLINE>
           [[5, 6],
            [7, 8]]], dtype=int32)
    >>> jnp.flip(x2)
    Array([[[8, 7],
            [6, 5]],
    <BLANKLINE>
           [[4, 3],
            [2, 1]]], dtype=int32)

    When ``axis`` is specified with a sequence of integers, then
    ``jax.numpy.flip`` reverses the array along the specified axes.

    >>> jnp.flip(x2, axis=[1, 2])
    Array([[[4, 3],
            [2, 1]],
    <BLANKLINE>
           [[8, 7],
            [6, 5]]], dtype=int32)
  """
  util.check_arraylike("flip", m)
  return _flip(asarray(m), reductions._ensure_optional_axes(axis))

@partial(jit, static_argnames=('axis',))
def _flip(m: Array, axis: int | tuple[int, ...] | None = None) -> Array:
  if axis is None:
    return lax.rev(m, list(range(len(shape(m)))))
  axis = _ensure_index_tuple(axis)
  return lax.rev(m, [_canonicalize_axis(ax, ndim(m)) for ax in axis])


def fliplr(m: ArrayLike) -> Array:
  """Reverse the order of elements of an array along axis 1.

  JAX implementation of :func:`numpy.fliplr`.

  Args:
    m: Array with at least two dimensions.

  Returns:
    An array with the elements in reverse order along axis 1.

  See Also:
    - :func:`jax.numpy.flip`: reverse the order along the given axis
    - :func:`jax.numpy.flipud`: reverse the order along axis 0

  Examples:
    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.fliplr(x)
    Array([[2, 1],
           [4, 3]], dtype=int32)
  """
  util.check_arraylike("fliplr", m)
  return _flip(asarray(m), 1)


def flipud(m: ArrayLike) -> Array:
  """Reverse the order of elements of an array along axis 0.

  JAX implementation of :func:`numpy.flipud`.

  Args:
    m: Array with at least one dimension.

  Returns:
    An array with the elements in reverse order along axis 0.

  See Also:
    - :func:`jax.numpy.flip`: reverse the order along the given axis
    - :func:`jax.numpy.fliplr`: reverse the order along axis 1

  Examples:
    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.flipud(x)
    Array([[3, 4],
           [1, 2]], dtype=int32)
  """
  util.check_arraylike("flipud", m)
  return _flip(asarray(m), 0)

@jit
def iscomplex(x: ArrayLike) -> Array:
  """Return boolean array showing where the input is complex.

  JAX implementation of :func:`numpy.iscomplex`.

  Args:
    x: Input array to check.

  Returns:
    A new array containing boolean values indicating complex elements.

  See Also:
    - :func:`jax.numpy.iscomplexobj`
    - :func:`jax.numpy.isrealobj`

  Examples:
    >>> jnp.iscomplex(jnp.array([True, 0, 1, 2j, 1+2j]))
    Array([False, False, False, True, True], dtype=bool)
  """
  i = ufuncs.imag(x)
  return lax.ne(i, _lax_const(i, 0))

@jit
def isreal(x: ArrayLike) -> Array:
  """Return boolean array showing where the input is real.

  JAX implementation of :func:`numpy.isreal`.

  Args:
    x: input array to check.

  Returns:
    A new array containing boolean values indicating real elements.

  See Also:
    - :func:`jax.numpy.iscomplex`
    - :func:`jax.numpy.isrealobj`

  Examples:
    >>> jnp.isreal(jnp.array([False, 0j, 1, 2.1, 1+2j]))
    Array([ True,  True,  True,  True, False], dtype=bool)
  """
  i = ufuncs.imag(x)
  return lax.eq(i, _lax_const(i, 0))


@partial(jit, static_argnames=['deg'])
def angle(z: ArrayLike, deg: bool = False) -> Array:
  """Return the angle of a complex valued number or array.

  JAX implementation of :func:`numpy.angle`.

  Args:
    z: A complex number or an array of complex numbers.
    deg: Boolean. If ``True``, returns the result in degrees else returns
      in radians. Default is ``False``.

  Returns:
    An array of counterclockwise angle of each element of ``z``, with the same
    shape as ``z`` of dtype float.

  Examples:

    If ``z`` is a number

    >>> z1 = 2+3j
    >>> jnp.angle(z1)
    Array(0.98279375, dtype=float32, weak_type=True)

    If ``z`` is an array

    >>> z2 = jnp.array([[1+3j, 2-5j],
    ...                 [4-3j, 3+2j]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(jnp.angle(z2))
    [[ 1.25 -1.19]
     [-0.64  0.59]]

    If ``deg=True``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(jnp.angle(z2, deg=True))
    [[ 71.57 -68.2 ]
     [-36.87  33.69]]
  """
  re = ufuncs.real(z)
  im = ufuncs.imag(z)
  dtype = _dtype(re)
  if not issubdtype(dtype, inexact) or (
      issubdtype(_dtype(z), floating) and ndim(z) == 0):
    dtype = dtypes.canonicalize_dtype(float_)
    re = lax.convert_element_type(re, dtype)
    im = lax.convert_element_type(im, dtype)
  result = lax.atan2(im, re)
  return ufuncs.degrees(result) if deg else result


@util.implements(np.diff)
@partial(jit, static_argnames=('n', 'axis'))
def diff(a: ArrayLike, n: int = 1, axis: int = -1,
         prepend: ArrayLike | None = None,
         append: ArrayLike | None = None) -> Array:
  util.check_arraylike("diff", a)
  arr = asarray(a)
  n = core.concrete_or_error(operator.index, n, "'n' argument of jnp.diff")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.diff")
  if n == 0:
    return arr
  if n < 0:
    raise ValueError(f"order must be non-negative but got {n}")
  if arr.ndim == 0:
    raise ValueError(f"diff requires input that is at least one dimensional; got {a}")

  nd = arr.ndim
  axis = _canonicalize_axis(axis, nd)

  combined: list[Array] = []
  if prepend is not None:
    util.check_arraylike("diff", prepend)
    if not ndim(prepend):
      shape = list(arr.shape)
      shape[axis] = 1
      prepend = broadcast_to(prepend, tuple(shape))
    combined.append(asarray(prepend))

  combined.append(arr)

  if append is not None:
    util.check_arraylike("diff", append)
    if not ndim(append):
      shape = list(arr.shape)
      shape[axis] = 1
      append = broadcast_to(append, tuple(shape))
    combined.append(asarray(append))

  if len(combined) > 1:
    arr = concatenate(combined, axis)

  slice1 = [slice(None)] * nd
  slice2 = [slice(None)] * nd
  slice1[axis] = slice(1, None)
  slice2[axis] = slice(None, -1)
  slice1_tuple = tuple(slice1)
  slice2_tuple = tuple(slice2)

  op = ufuncs.not_equal if arr.dtype == np.bool_ else ufuncs.subtract
  for _ in range(n):
    arr = op(arr[slice1_tuple], arr[slice2_tuple])

  return arr

_EDIFF1D_DOC = """\
Unlike NumPy's implementation of ediff1d, :py:func:`jax.numpy.ediff1d` will not
issue an error if casting ``to_end`` or ``to_begin`` to the type of ``ary``
loses precision.
"""

@util.implements(np.ediff1d, lax_description=_EDIFF1D_DOC)
@jit
def ediff1d(ary: ArrayLike, to_end: ArrayLike | None = None,
            to_begin: ArrayLike | None = None) -> Array:
  util.check_arraylike("ediff1d", ary)
  arr = ravel(ary)
  result = lax.sub(arr[1:], arr[:-1])
  if to_begin is not None:
    util.check_arraylike("ediff1d", to_begin)
    result = concatenate((ravel(asarray(to_begin, dtype=arr.dtype)), result))
  if to_end is not None:
    util.check_arraylike("ediff1d", to_end)
    result = concatenate((result, ravel(asarray(to_end, dtype=arr.dtype))))
  return result

@util.implements(np.gradient, skip_params=['edge_order'])
@partial(jit, static_argnames=("axis", "edge_order"))
def gradient(
    f: ArrayLike,
    *varargs: ArrayLike,
    axis: int | Sequence[int] | None = None,
    edge_order: int | None = None,
) -> Array | list[Array]:

  if edge_order is not None:
    raise NotImplementedError(
        "The 'edge_order' argument to jnp.gradient is not supported."
    )
  a, *spacing = util.promote_dtypes_inexact(f, *varargs)

  def gradient_along_axis(a, h, axis):
    sliced = partial(lax.slice_in_dim, a, axis=axis)
    upper_edge = sliced(1, 2) - sliced(0, 1)
    lower_edge = sliced(-1, None) - sliced(-2, -1)

    if ndim(h) == 0:
      inner = (sliced(2, None) - sliced(None, -2)) * 0.5 / h
      lower_edge /= h
      upper_edge /= h

    elif ndim(h) == 1:
      if len(h) != a.shape[axis]:
        raise ValueError(
            "Spacing arrays must have the same length as the "
            "dimension along which the gradient is calculated."
        )
      h_shape = [1] * a.ndim
      h_shape[axis] = len(h)
      h = h.reshape(h_shape)
      sliced_x = partial(lax.slice_in_dim, h, axis=axis)

      upper_edge /= sliced_x(1, 2) - sliced_x(0, 1)
      lower_edge /= sliced_x(-1, None) - sliced_x(-2, -1)
      dx1 = sliced_x(1, -1) - sliced_x(0, -2)
      dx2 = sliced_x(2, None) - sliced_x(1, -1)
      a = -(dx2) / (dx1 * (dx1 + dx2))
      b = (dx2 - dx1) / (dx1 * dx2)
      c = dx1 / (dx2 * (dx1 + dx2))
      inner = a * sliced(0, -2) + b * sliced(1, -1) + c * sliced(2, None)
    else:
      raise ValueError("Spacing arrays must be 1D arrays or scalars.")

    return concatenate((upper_edge, inner, lower_edge), axis=axis)

  if axis is None:
    axis_tuple = tuple(range(a.ndim))
  else:
    axis_tuple = tuple(_canonicalize_axis(i, a.ndim) for i in _ensure_index_tuple(axis))
  if len(axis_tuple) == 0:
    return []

  if min(s for i, s in enumerate(a.shape) if i in axis_tuple) < 2:
    raise ValueError("Shape of array too small to calculate "
                     "a numerical gradient, "
                     "at least 2 elements are required.")
  if len(spacing) == 0:
    dx: Sequence[ArrayLike] = [1.0] * len(axis_tuple)
  elif len(spacing) == 1:
    dx = list(spacing) * len(axis_tuple)
  elif len(spacing) == len(axis_tuple):
    dx = list(spacing)
  else:
    TypeError(f"Invalid number of spacing arguments {len(spacing)} for {axis=}")

  a_grad = [gradient_along_axis(a, h, ax) for ax, h in zip(axis_tuple, dx)]
  return a_grad[0] if len(axis_tuple) == 1 else a_grad


def isrealobj(x: Any) -> bool:
  """Check if the input is not a complex number or an array containing complex elements.

  JAX implementation of :func:`numpy.isrealobj`.

  The function evaluates based on input type rather than value.
  Inputs with zero imaginary parts are still considered complex.

  Args:
    x: input object to check.

  Returns:
    False if ``x`` is a complex number or an array containing at least one complex element,
    True otherwise.

  See Also:
    - :func:`jax.numpy.iscomplexobj`
    - :func:`jax.numpy.isreal`

  Examples:
    >>> jnp.isrealobj(0)
    True
    >>> jnp.isrealobj(1.2)
    True
    >>> jnp.isrealobj(jnp.array([1, 2]))
    True
    >>> jnp.isrealobj(1+2j)
    False
    >>> jnp.isrealobj(jnp.array([0, 1+2j]))
    False
  """
  return not iscomplexobj(x)


def reshape(
    a: ArrayLike, shape: DimSize | Shape | None = None, order: str = "C", *,
    newshape: DimSize | Shape | DeprecatedArg = DeprecatedArg()) -> Array:
  """Return a reshaped copy of an array.

  JAX implementation of :func:`numpy.reshape`, implemented in terms of
  :func:`jax.lax.reshape`.

  Args:
    a: input array to reshape
    shape: integer or sequence of integers giving the new shape, which must match the
      size of the input array. If any single dimension is given size ``-1``, it will be
      replaced with a value such that the output has the correct size.
    order: ``'F'`` or ``'C'``, specifies whether the reshape should apply column-major
      (fortran-style, ``"F"``) or row-major (C-style, ``"C"``) order; default is ``"C"``.
      JAX does not support ``order="A"``.

  Returns:
    reshaped copy of input array with the specified shape.

  Notes:
    Unlike :func:`numpy.reshape`, :func:`jax.numpy.reshape` will return a copy rather
    than a view of the input array. However, under JIT, the compiler will optimize-away
    such copies when possible, so this doesn't have performance impacts in practice.

  See Also:
    - :meth:`jax.Array.reshape`: equivalent functionality via an array method.
    - :func:`jax.numpy.ravel`: flatten an array into a 1D shape.
    - :func:`jax.numpy.squeeze`: remove one or more length-1 axes from an array's shape.

  Examples:
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.reshape(x, 6)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)
    >>> jnp.reshape(x, (3, 2))
    Array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)

    You can use ``-1`` to automatically compute a shape that is consistent with
    the input size:

    >>> jnp.reshape(x, -1)  # -1 is inferred to be 6
    Array([1, 2, 3, 4, 5, 6], dtype=int32)
    >>> jnp.reshape(x, (-1, 2))  # -1 is inferred to be 3
    Array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)

    The default ordering of axes in the reshape is C-style row-major ordering.
    To use Fortran-style column-major ordering, specify ``order='F'``:

    >>> jnp.reshape(x, 6, order='F')
    Array([1, 4, 2, 5, 3, 6], dtype=int32)
    >>> jnp.reshape(x, (3, 2), order='F')
    Array([[1, 5],
           [4, 3],
           [2, 6]], dtype=int32)

    For convenience, this functionality is also available via the
    :meth:`jax.Array.reshape` method:

    >>> x.reshape(3, 2)
    Array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)
  """
  __tracebackhide__ = True
  util.check_arraylike("reshape", a)

  # TODO(micky774): deprecated 2024-5-9, remove after deprecation expires.
  if not isinstance(newshape, DeprecatedArg):
    if shape is not None:
      raise ValueError(
        "jnp.reshape received both `shape` and `newshape` arguments. Note that "
        "using `newshape` is deprecated, please only use `shape` instead."
      )
    warnings.warn(
      "The newshape argument of jax.numpy.reshape is deprecated and setting it "
      "will soon raise an error. To avoid an error in the future, and to "
      "suppress this warning, please use the shape argument instead.",
      DeprecationWarning, stacklevel=2)
    shape = newshape
    del newshape
  elif shape is None:
    raise TypeError(
      "jnp.shape requires passing a `shape` argument, but none was given."
    )
  try:
    # forward to method for ndarrays
    return a.reshape(shape, order=order)  # type: ignore[call-overload,union-attr]
  except AttributeError:
    pass
  return asarray(a).reshape(shape, order=order)


@partial(jit, static_argnames=('order',), inline=True)
def ravel(a: ArrayLike, order: str = "C") -> Array:
  """Flatten array into a 1-dimensional shape.

  JAX implementation of :func:`numpy.ravel`, implemented in terms of
  :func:`jax.lax.reshape`.

  ``ravel(arr, order=order)`` is equivalent to ``reshape(arr, -1, order=order)``.

  Args:
    a: array to be flattened.
    order: ``'F'`` or ``'C'``, specifies whether the reshape should apply column-major
      (fortran-style, ``"F"``) or row-major (C-style, ``"C"``) order; default is ``"C"``.
      JAX does not support `order="A"` or `order="K"`.

  Returns:
    flattened copy of input array.

  Notes:
    Unlike :func:`numpy.ravel`, :func:`jax.numpy.ravel` will return a copy rather
    than a view of the input array. However, under JIT, the compiler will optimize-away
    such copies when possible, so this doesn't have performance impacts in practice.

  See Also:
    - :meth:`jax.Array.ravel`: equivalent functionality via an array method.
    - :func:`jax.numpy.reshape`: general array reshape.

  Examples:
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])

    By default, ravel in C-style, row-major order

    >>> jnp.ravel(x)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

    Optionally ravel in Fortran-style, column-major:

    >>> jnp.ravel(x, order='F')
    Array([1, 4, 2, 5, 3, 6], dtype=int32)

    For convenience, the same functionality is available via the :meth:`jax.Array.ravel`
    method:

    >>> x.ravel()
    Array([1, 2, 3, 4, 5, 6], dtype=int32)
  """
  util.check_arraylike("ravel", a)
  if order == "K":
    raise NotImplementedError("Ravel not implemented for order='K'.")
  return reshape(a, (size(a),), order)


def ravel_multi_index(multi_index: Sequence[ArrayLike], dims: Sequence[int],
                      mode: str = 'raise', order: str = 'C') -> Array:
  """Convert multi-dimensional indices into flat indices.

  JAX implementation of :func:`numpy.ravel_multi_index`

  Args:
    multi_index: sequence of integer arrays containing indices in each dimension.
    dims: sequence of integer sizes; must have ``len(dims) == len(multi_index)``
    mode: how to handle out-of bound indices. Options are

      - ``"raise"`` (default): raise a ValueError. This mode is incompatible
        with :func:`~jax.jit` or other JAX transformations.
      - ``"clip"``: clip out-of-bound indices to valid range.
      - ``"wrap"``: wrap out-of-bound indices to valid range.

    order: ``"C"`` (default) or ``"F"``, specify whether to assume C-style
      row-major order or Fortran-style column-major order.

  Returns:
    array of flattened indices

  See also:
    :func:`jax.numpy.unravel_index`: inverse of this function.

  Examples:
    Define a 2-dimensional array and a sequence of indices of even values:

    >>> x = jnp.array([[2., 3., 4.],
    ...                [5., 6., 7.]])
    >>> indices = jnp.where(x % 2 == 0)
    >>> indices
    (Array([0, 0, 1], dtype=int32), Array([0, 2, 1], dtype=int32))
    >>> x[indices]
    Array([2., 4., 6.], dtype=float32)

    Compute the flattened indices:

    >>> indices_flat = jnp.ravel_multi_index(indices, x.shape)
    >>> indices_flat
    Array([0, 2, 4], dtype=int32)

    These flattened indices can be used to extract the same values from the
    flattened ``x`` array:

    >>> x_flat = x.ravel()
    >>> x_flat
    Array([2., 3., 4., 5., 6., 7.], dtype=float32)
    >>> x_flat[indices_flat]
    Array([2., 4., 6.], dtype=float32)

    The original indices can be recovered with :func:`~jax.numpy.unravel_index`:

    >>> jnp.unravel_index(indices_flat, x.shape)
    (Array([0, 0, 1], dtype=int32), Array([0, 2, 1], dtype=int32))
  """
  assert len(multi_index) == len(dims), f"len(multi_index)={len(multi_index)} != len(dims)={len(dims)}"
  dims = tuple(core.concrete_or_error(operator.index, d, "in `dims` argument of ravel_multi_index().") for d in dims)
  util.check_arraylike("ravel_multi_index", *multi_index)
  multi_index_arr = [asarray(i) for i in multi_index]
  for index in multi_index_arr:
    if mode == 'raise':
      core.concrete_or_error(array, index,
        "The error occurred because ravel_multi_index was jit-compiled"
        " with mode='raise'. Use mode='wrap' or mode='clip' instead.")
    if not issubdtype(_dtype(index), integer):
      raise TypeError("only int indices permitted")
  if mode == "raise":
    if any(reductions.any((i < 0) | (i >= d)) for i, d in zip(multi_index_arr, dims)):
      raise ValueError("invalid entry in coordinates array")
  elif mode == "clip":
    multi_index_arr = [clip(i, 0, d - 1) for i, d in zip(multi_index_arr, dims)]
  elif mode == "wrap":
    multi_index_arr = [i % d for i, d in zip(multi_index_arr, dims)]
  else:
    raise ValueError(f"invalid mode={mode!r}. Expected 'raise', 'wrap', or 'clip'")

  if order == "F":
    strides = np.cumprod((1,) + dims[:-1])
  elif order == "C":
    strides = np.cumprod((1,) + dims[1:][::-1])[::-1]
  else:
    raise ValueError(f"invalid order={order!r}. Expected 'C' or 'F'")

  result = array(0, dtype=(multi_index_arr[0].dtype if multi_index_arr
                           else dtypes.canonicalize_dtype(int_)))
  for i, s in zip(multi_index_arr, strides):
    result = result + i * int(s)
  return result


def unravel_index(indices: ArrayLike, shape: Shape) -> tuple[Array, ...]:
  """Convert flat indices into multi-dimensional indices.

  JAX implementation of :func:`numpy.unravel_index`. The JAX version differs in
  its treatment of out-of-bound indices: unlike NumPy, negative indices are
  supported, and out-of-bound indices are clipped to the nearest valid value.

  Args:
    indices: integer array of flat indices
    shape: shape of multidimensional array to index into

  Returns:
    Tuple of unraveled indices

  See also:
    :func:`jax.numpy.ravel_multi_index`: Inverse of this function.

  Examples:
    Start with a 1D array values and indices:

    >>> x = jnp.array([2., 3., 4., 5., 6., 7.])
    >>> indices = jnp.array([1, 3, 5])
    >>> print(x[indices])
    [3. 5. 7.]

    Now if ``x`` is reshaped, ``unravel_indices`` can be used to convert
    the flat indices into a tuple of indices that access the same entries:

    >>> shape = (2, 3)
    >>> x_2D = x.reshape(shape)
    >>> indices_2D = jnp.unravel_index(indices, shape)
    >>> indices_2D
    (Array([0, 1, 1], dtype=int32), Array([1, 0, 2], dtype=int32))
    >>> print(x_2D[indices_2D])
    [3. 5. 7.]

    The inverse function, ``ravel_multi_index``, can be used to obtain the
    original indices:

    >>> jnp.ravel_multi_index(indices_2D, shape)
    Array([1, 3, 5], dtype=int32)
  """
  util.check_arraylike("unravel_index", indices)
  indices_arr = asarray(indices)
  # Note: we do not convert shape to an array, because it may be passed as a
  # tuple of weakly-typed values, and asarray() would strip these weak types.
  try:
    shape = list(shape)
  except TypeError:
    # TODO: Consider warning here since shape is supposed to be a sequence, so
    # this should not happen.
    shape = [shape]
  if any(ndim(s) != 0 for s in shape):
    raise ValueError("unravel_index: shape should be a scalar or 1D sequence.")
  out_indices: list[ArrayLike] = [0] * len(shape)
  for i, s in reversed(list(enumerate(shape))):
    indices_arr, out_indices[i] = ufuncs.divmod(indices_arr, s)
  oob_pos = indices_arr > 0
  oob_neg = indices_arr < -1
  return tuple(where(oob_pos, s - 1, where(oob_neg, 0, i))
               for s, i in safe_zip(shape, out_indices))

@util.implements(np.resize)
@partial(jit, static_argnames=('new_shape',))
def resize(a: ArrayLike, new_shape: Shape) -> Array:
  util.check_arraylike("resize", a)
  new_shape = _ensure_index_tuple(new_shape)

  if any(dim_length < 0 for dim_length in new_shape):
    raise ValueError("all elements of `new_shape` must be non-negative")

  arr = ravel(a)

  new_size = math.prod(new_shape)
  if arr.size == 0 or new_size == 0:
    return zeros_like(arr, shape=new_shape)

  repeats = ceil_of_ratio(new_size, arr.size)
  arr = tile(arr, repeats)[:new_size]

  return reshape(arr, new_shape)


def squeeze(a: ArrayLike, axis: int | Sequence[int] | None = None) -> Array:
  """Remove one or more length-1 axes from array

  JAX implementation of :func:`numpy.sqeeze`, implemented via :func:`jax.lax.squeeze`.

  Args:
    a: input array
    axis: integer or sequence of integers specifying axes to remove. If any specified
      axis does not have a length of 1, an error is raised. If not specified, squeeze
      all length-1 axes in ``a``.

  Returns:
    copy of ``a`` with length-1 axes removed.

  Notes:
    Unlike :func:`numpy.squeeze`, :func:`jax.numpy.squeeze` will return a copy rather
    than a view of the input array. However, under JIT, the compiler will optimize-away
    such copies when possible, so this doesn't have performance impacts in practice.

  See Also:
    - :func:`jax.numpy.expand_dims`: the inverse of ``squeeze``: add dimensions of length 1.
    - :meth:`jax.Array.squeeze`: equivalent functionality via an array method.
    - :func:`jax.lax.squeeze`: equivalent XLA API.
    - :func:`jax.numpy.ravel`: flatten an array into a 1D shape.
    - :func:`jax.numpy.reshape`: general array reshape.

  Examples:
    >>> x = jnp.array([[[0]], [[1]], [[2]]])
    >>> x.shape
    (3, 1, 1)

    Squeeze all length-1 dimensions:

    >>> jnp.squeeze(x)
    Array([0, 1, 2], dtype=int32)
    >>> _.shape
    (3,)

    Equivalent while specifying the axes explicitly:

    >>> jnp.squeeze(x, axis=(1, 2))
    Array([0, 1, 2], dtype=int32)

    Attempting to squeeze a non-unit axis results in an error:

    >>> jnp.squeeze(x, axis=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one, got shape=(3, 1, 1) and dimensions=(0,)

    For convenience, this functionality is also available via the
    :meth:`jax.Array.squeeze` method:

    >>> x.squeeze()
    Array([0, 1, 2], dtype=int32)
  """
  util.check_arraylike("squeeze", a)
  return _squeeze(asarray(a), _ensure_index_tuple(axis) if axis is not None else None)

@partial(jit, static_argnames=('axis',), inline=True)
def _squeeze(a: Array, axis: tuple[int, ...]) -> Array:
  if axis is None:
    a_shape = shape(a)
    if not core.is_constant_shape(a_shape):
      # We do not even know the rank of the output if the input shape is not known
      raise ValueError("jnp.squeeze with axis=None is not supported with shape polymorphism")
    axis = tuple(i for i, d in enumerate(a_shape) if d == 1)
  return lax.squeeze(a, axis)


def expand_dims(a: ArrayLike, axis: int | Sequence[int]) -> Array:
  """Insert dimensions of length 1 into array

  JAX implementation of :func:`numpy.expand_dims`, implemented via
  :func:`jax.lax.expand_dims`.

  Args:
    a: input array
    axis: integer or sequence of integers specifying positions of axes to add.

  Returns:
    Copy of ``a`` with added dimensions.

  Notes:
    Unlike :func:`numpy.expand_dims`, :func:`jax.numpy.expand_dims` will return a copy
    rather than a view of the input array. However, under JIT, the compiler will optimize
    away such copies when possible, so this doesn't have performance impacts in practice.

  See Also:
    - :func:`jax.numpy.squeeze`: inverse of this operation, i.e. remove length-1 dimensions.
    - :func:`jax.lax.expand_dims`: XLA version of this functionality.

  Examples:
    >>> x = jnp.array([1, 2, 3])
    >>> x.shape
    (3,)

    Expand the leading dimension:

    >>> jnp.expand_dims(x, 0)
    Array([[1, 2, 3]], dtype=int32)
    >>> _.shape
    (1, 3)

    Expand the trailing dimension:

    >>> jnp.expand_dims(x, 1)
    Array([[1],
           [2],
           [3]], dtype=int32)
    >>> _.shape
    (3, 1)

    Expand multiple dimensions:

    >>> jnp.expand_dims(x, (0, 1, 3))
    Array([[[[1],
             [2],
             [3]]]], dtype=int32)
    >>> _.shape
    (1, 1, 3, 1)

    Dimensions can also be expanded more succinctly by indexing with ``None``:

    >>> x[None]  # equivalent to jnp.expand_dims(x, 0)
    Array([[1, 2, 3]], dtype=int32)
    >>> x[:, None]  # equivalent to jnp.expand_dims(x, 1)
    Array([[1],
           [2],
           [3]], dtype=int32)
    >>> x[None, None, :, None]  # equivalent to jnp.expand_dims(x, (0, 1, 3))
    Array([[[[1],
             [2],
             [3]]]], dtype=int32)
  """
  util.check_arraylike("expand_dims", a)
  axis = _ensure_index_tuple(axis)
  return lax.expand_dims(a, axis)


@partial(jit, static_argnames=('axis1', 'axis2'), inline=True)
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> Array:
  """Swap two axes of an array.

  JAX implementation of :func:`numpy.swapaxes`, implemented in terms of
  :func:`jax.lax.transpose`.

  Args:
    a: input array
    axis1: index of first axis
    axis2: index of second axis

  Returns:
    Copy of ``a`` with specified axes swapped.

  Notes:
    Unlike :func:`numpy.swapaxes`, :func:`jax.numpy.swapaxes` will return a copy rather
    than a view of the input array. However, under JIT, the compiler will optimize away
    such copies when possible, so this doesn't have performance impacts in practice.

  See Also:
    - :func:`jax.numpy.moveaxis`: move a single axis of an array.
    - :func:`jax.numpy.rollaxis`: older API for ``moveaxis``.
    - :func:`jax.lax.transpose`: more general axes permutations.
    - :meth:`jax.Array.swapaxes`: same functionality via an array method.

  Examples:
    >>> a = jnp.ones((2, 3, 4, 5))
    >>> jnp.swapaxes(a, 1, 3).shape
    (2, 5, 4, 3)

    Equivalent output via the ``swapaxes`` array method:

    >>> a.swapaxes(1, 3).shape
    (2, 5, 4, 3)

    Equivalent output via :func:`~jax.numpy.transpose`:

    >>> a.transpose(0, 3, 2, 1).shape
    (2, 5, 4, 3)
  """
  util.check_arraylike("swapaxes", a)
  perm = np.arange(ndim(a))
  perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
  return lax.transpose(a, list(perm))


def moveaxis(a: ArrayLike, source: int | Sequence[int],
             destination: int | Sequence[int]) -> Array:
  """Move an array axis to a new position

  JAX implementation of :func:`numpy.moveaxis`, implemented in terms of
  :func:`jax.lax.transpose`.

  Args:
    a: input array
    source: index or indices of the axes to move.
    destination: index or indices of the axes destinations

  Returns:
    Copy of ``a`` with axes moved from ``source`` to ``destination``.

  Notes:
    Unlike :func:`numpy.moveaxis`, :func:`jax.numpy.moveaxis` will return a copy rather
    than a view of the input array. However, under JIT, the compiler will optimize away
    such copies when possible, so this doesn't have performance impacts in practice.

  See also:
    - :func:`jax.numpy.swapaxes`: swap two axes.
    - :func:`jax.numpy.rollaxis`: older API for moving an axis.
    - :func:`jax.numpy.transpose`: general axes permutation.

  Examples:
    >>> a = jnp.ones((2, 3, 4, 5))

    Move axis ``1`` to the end of the array:

    >>> jnp.moveaxis(a, 1, -1).shape
    (2, 4, 5, 3)

    Move the last axis to position 1:

    >>> jnp.moveaxis(a, -1, 1).shape
    (2, 5, 3, 4)

    Move multiple axes:

    >>> jnp.moveaxis(a, (0, 1), (-1, -2)).shape
    (4, 5, 3, 2)

    This can also be accomplished via :func:`~jax.numpy.transpose`:

    >>> a.transpose(2, 3, 1, 0).shape
    (4, 5, 3, 2)
  """
  util.check_arraylike("moveaxis", a)
  return _moveaxis(asarray(a), _ensure_index_tuple(source),
                   _ensure_index_tuple(destination))

@partial(jit, static_argnames=('source', 'destination'), inline=True)
def _moveaxis(a: Array, source: tuple[int, ...], destination: tuple[int, ...]) -> Array:
  source = tuple(_canonicalize_axis(i, ndim(a)) for i in source)
  destination = tuple(_canonicalize_axis(i, ndim(a)) for i in destination)
  if len(source) != len(destination):
    raise ValueError("Inconsistent number of elements: {} vs {}"
                     .format(len(source), len(destination)))
  perm = [i for i in range(ndim(a)) if i not in source]
  for dest, src in sorted(zip(destination, source)):
    perm.insert(dest, src)
  return lax.transpose(a, perm)


@partial(jit, static_argnames=('equal_nan',))
def isclose(a: ArrayLike, b: ArrayLike, rtol: ArrayLike = 1e-05, atol: ArrayLike = 1e-08,
            equal_nan: bool = False) -> Array:
  r"""Check if the elements of two arrays are approximately equal within a tolerance.

  JAX implementation of :func:`numpy.allclose`.

  Essentially this function evaluates the following condition:

  .. math::

     |a - b| \le \mathtt{atol} + \mathtt{rtol} * |b|

  ``jnp.inf`` in ``a`` will be considered equal to ``jnp.inf`` in ``b``.

  Args:
    a: first input array to compare.
    b: second input array to compare.
    rtol: relative tolerance used for approximate equality. Default = 1e-05.
    atol: absolute tolerance used for approximate equality. Default = 1e-08.
    equal_nan: Boolean. If ``True``, NaNs in ``a`` will be considered
      equal to NaNs in ``b``. Default is ``False``.

  Returns:
    A new array containing boolean values indicating whether the input arrays
    are element-wise approximately equal within the specified tolerances.

  See Also:
    - :func:`jax.numpy.allclose`
    - :func:`jax.numpy.equal`

  Examples:
    >>> jnp.isclose(jnp.array([1e6, 2e6, jnp.inf]), jnp.array([1e6, 2e7, jnp.inf]))
    Array([ True, False,  True], dtype=bool)
    >>> jnp.isclose(jnp.array([1e6, 2e6, 3e6]),
    ...              jnp.array([1.00008e6, 2.00008e7, 3.00008e8]), rtol=1e3)
    Array([ True,  True,  True], dtype=bool)
    >>> jnp.isclose(jnp.array([1e6, 2e6, 3e6]),
    ...              jnp.array([1.00001e6, 2.00002e6, 3.00009e6]), atol=1e3)
    Array([ True,  True,  True], dtype=bool)
    >>> jnp.isclose(jnp.array([jnp.nan, 1, 2]),
    ...              jnp.array([jnp.nan, 1, 2]), equal_nan=True)
    Array([ True,  True,  True], dtype=bool)
  """
  a, b = util.promote_args("isclose", a, b)
  dtype = _dtype(a)
  if dtypes.issubdtype(dtype, dtypes.extended):
    return lax.eq(a, b)

  a, b = util.promote_args_inexact("isclose", a, b)
  dtype = _dtype(a)
  if issubdtype(dtype, complexfloating):
    dtype = util._complex_elem_type(dtype)
  rtol = lax.convert_element_type(rtol, dtype)
  atol = lax.convert_element_type(atol, dtype)
  out = lax.le(
    lax.abs(lax.sub(a, b)),
    lax.add(atol, lax.mul(rtol, lax.abs(b))))
  # This corrects the comparisons for infinite and nan values
  a_inf = ufuncs.isinf(a)
  b_inf = ufuncs.isinf(b)
  any_inf = ufuncs.logical_or(a_inf, b_inf)
  both_inf = ufuncs.logical_and(a_inf, b_inf)
  # Make all elements where either a or b are infinite to False
  out = ufuncs.logical_and(out, ufuncs.logical_not(any_inf))
  # Make all elements where both a or b are the same inf to True
  same_value = lax.eq(a, b)
  same_inf = ufuncs.logical_and(both_inf, same_value)
  out = ufuncs.logical_or(out, same_inf)

  # Make all elements where either a or b is NaN to False
  a_nan = ufuncs.isnan(a)
  b_nan = ufuncs.isnan(b)
  any_nan = ufuncs.logical_or(a_nan, b_nan)
  out = ufuncs.logical_and(out, ufuncs.logical_not(any_nan))
  if equal_nan:
    # Make all elements where both a and b is NaN to True
    both_nan = ufuncs.logical_and(a_nan, b_nan)
    out = ufuncs.logical_or(out, both_nan)
  return out


def _interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike,
           left: ArrayLike | str | None = None,
           right: ArrayLike | str | None = None,
           period: ArrayLike | None = None) -> Array:
  util.check_arraylike("interp", x, xp, fp)
  if shape(xp) != shape(fp) or ndim(xp) != 1:
    raise ValueError("xp and fp must be one-dimensional arrays of equal size")
  x_arr, xp_arr = util.promote_dtypes_inexact(x, xp)
  fp_arr, = util.promote_dtypes_inexact(fp)
  del x, xp, fp

  if isinstance(left, str):
    if left != 'extrapolate':
      raise ValueError("the only valid string value of `left` is "
                       f"'extrapolate', but got: {left!r}")
    extrapolate_left = True
  else:
    extrapolate_left = False
  if isinstance(right, str):
    if right != 'extrapolate':
      raise ValueError("the only valid string value of `right` is "
                       f"'extrapolate', but got: {right!r}")
    extrapolate_right = True
  else:
    extrapolate_right = False

  if dtypes.issubdtype(x_arr.dtype, np.complexfloating):
    raise ValueError("jnp.interp: complex x values not supported.")

  if period is not None:
    if ndim(period) != 0:
      raise ValueError(f"period must be a scalar; got {period}")
    period = ufuncs.abs(period)
    x_arr = x_arr % period
    xp_arr = xp_arr % period
    xp_arr, fp_arr = lax.sort_key_val(xp_arr, fp_arr)
    xp_arr = concatenate([xp_arr[-1:] - period, xp_arr, xp_arr[:1] + period])
    fp_arr = concatenate([fp_arr[-1:], fp_arr, fp_arr[:1]])

  i = clip(searchsorted(xp_arr, x_arr, side='right'), 1, len(xp_arr) - 1)
  df = fp_arr[i] - fp_arr[i - 1]
  dx = xp_arr[i] - xp_arr[i - 1]
  delta = x_arr - xp_arr[i - 1]

  epsilon = np.spacing(np.finfo(xp_arr.dtype).eps)
  dx0 = lax.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
  f = where(dx0, fp_arr[i - 1], fp_arr[i - 1] + (delta / where(dx0, 1, dx)) * df)

  if not extrapolate_left:
    assert not isinstance(left, str)
    left_arr: ArrayLike = fp_arr[0] if left is None else left
    if period is None:
      f = where(x_arr < xp_arr[0], left_arr, f)
  if not extrapolate_right:
    assert not isinstance(right, str)
    right_arr: ArrayLike = fp_arr[-1] if right is None else right
    if period is None:
      f = where(x_arr > xp_arr[-1], right_arr, f)

  return f


@util.implements(np.interp,
  lax_description=_dedent("""
    In addition to constant interpolation supported by NumPy, jnp.interp also
    supports left='extrapolate' and right='extrapolate' to indicate linear
    extrapolation instead."""))
def interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike,
           left: ArrayLike | str | None = None,
           right: ArrayLike | str | None = None,
           period: ArrayLike | None = None) -> Array:
  static_argnames = []
  if isinstance(left, str) or left is None:
    static_argnames.append('left')
  if isinstance(right, str) or right is None:
    static_argnames.append('right')
  if period is None:
    static_argnames.append('period')
  jitted_interp = jit(_interp, static_argnames=static_argnames)
  return jitted_interp(x, xp, fp, left, right, period)


@overload
def where(condition: ArrayLike, x: Literal[None] = None,
          y: Literal[None] = None, /, *, size: int | None = None,
          fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None
          ) -> tuple[Array, ...]: ...

@overload
def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike, / ,*,
          size: int | None = None,
          fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None
          ) -> Array: ...

@overload
def where(condition: ArrayLike, x: ArrayLike | None = None,
          y: ArrayLike | None = None, /, *, size: int | None = None,
          fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None
          ) -> Array | tuple[Array, ...]: ...


def where(condition, x=None, y=None, /, *, size=None, fill_value=None):
  """Select elements from two arrays based on a condition.

  JAX implementation of :func:`numpy.where`.

  .. note::
     when only ``condition`` is provided, ``jnp.where(condition)`` is equivalent
     to ``jnp.nonzero(condition)``. For that case, refer to the documentation of
     :func:`jax.numpy.nonzero`. The docstring below focuses on the case where
     ``x`` and ``y`` are specified.

  The three-term version of ``jnp.where`` lowers to :func:`jax.lax.select`.

  Args:
    condition: boolean array. Must be broadcast-compatible with ``x`` and ``y`` when
      they are specified.
    x: arraylike. Should be broadcast-compatible with ``condition`` and ``y``, and
      typecast-compatible with ``y``.
    y: arraylike. Should be broadcast-compatible with ``condition`` and ``x``, and
      typecast-compatible with ``x``.
    size: integer, only referenced when ``x`` and ``y`` are ``None``. For details,
      see :func:`jax.numpy.nonzero`.
    fill_value: only referenced when ``x`` and ``y`` are ``None``. For details,
      see :func:`jax.numpy.nonzero`.

  Returns:
    An array of dtype ``jnp.result_type(x, y)`` with values drawn from ``x`` where ``condition``
    is True, and from ``y`` where condition is ``False. If ``x`` and ``y`` are ``None``, the
    function behaves differently; see `:func:`jax.numpy.nonzero` for a description of the return
    type.

  See Also:
    - :func:`jax.numpy.nonzero`
    - :func:`jax.numpy.argwhere`
    - :func:`jax.lax.select`

  Notes:
    Special care is needed when the ``x`` or ``y`` input to :func:`jax.numpy.where` could
    have a value of NaN. Specifically, when a gradient is taken with :func:`jax.grad`
    (reverse-mode differentiation), a NaN in either ``x`` or ``y`` will propagate into the
    gradient, regardless of the value of ``condition``.  More information on this behavior
    and workarounds is available in the `JAX FAQ
    <https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where>`_.

  Examples:
    When ``x`` and ``y`` are not provided, ``where`` behaves equivalently to
    :func:`jax.numpy.nonzero`:

    >>> x = jnp.arange(10)
    >>> jnp.where(x > 4)
    (Array([5, 6, 7, 8, 9], dtype=int32),)
    >>> jnp.nonzero(x > 4)
    (Array([5, 6, 7, 8, 9], dtype=int32),)

    When ``x`` and ``y`` are provided, ``where`` selects between them based on
    the specified condition:

    >>> jnp.where(x > 4, x, 0)
    Array([0, 0, 0, 0, 0, 5, 6, 7, 8, 9], dtype=int32)
  """
  if x is None and y is None:
    util.check_arraylike("where", condition)
    return nonzero(condition, size=size, fill_value=fill_value)
  else:
    util.check_arraylike("where", condition, x, y)
    if size is not None or fill_value is not None:
      raise ValueError("size and fill_value arguments cannot be used in "
                       "three-term where function.")
    if x is None or y is None:
      raise ValueError("Either both or neither of the x and y arguments "
                       "should be provided to jax.numpy.where, got "
                       f"{x} and {y}.")
    return util._where(condition, x, y)


@util.implements(np.select)
def select(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    default: ArrayLike = 0,
) -> Array:
  if len(condlist) != len(choicelist):
    msg = "condlist must have length equal to choicelist ({} vs {})"
    raise ValueError(msg.format(len(condlist), len(choicelist)))
  if len(condlist) == 0:
    raise ValueError("condlist must be non-empty")
  # Put the default at front with condition False because
  # argmax returns zero for an array of False values.
  choicelist = util.promote_dtypes(default, *choicelist)
  conditions = stack(broadcast_arrays(False, *condlist))
  idx = argmax(conditions.astype(bool), axis=0)
  return lax.select_n(*broadcast_arrays(idx, *choicelist))


def bincount(x: ArrayLike, weights: ArrayLike | None = None,
             minlength: int = 0, *, length: int | None = None
             ) -> Array:
  """Count the number of occurrences of each value in an integer array.

  JAX implementation of :func:`numpy.bincount`.

  For an array of positive integers ``x``, this function returns an array ``counts``
  of size ``x.max() + 1``, such that ``counts[i]`` contains the number of occurrences
  of the value ``i`` in ``x``.

  The JAX version has a few differences from the NumPy version:

  - In NumPy, passing an array ``x`` with negative entries will result in an error.
    In JAX, negative values are clipped to zero.
  - JAX adds an optional ``length`` parameter which can be used to statically specify
    the length of the output array so that this function can be used with transformations
    like :func:`jax.jit`. In this case, items larger than `length + 1` will be dropped.

  Args:
    x : N-dimensional array of positive integers
    weights: optional array of weights associated with ``x``. If not specified, the
      weight for each entry will be ``1``.
    minlength: the minimum length of the output counts array.
    length: the length of the output counts array. Must be specified statically for
      ``bincount`` to be used with :func:`jax.jit` and other JAX transformations.

  Returns:
    An array of counts or summed weights reflecting the number of occurrences of values
    in ``x``.

  See Also:
    - :func:`jax.numpy.histogram`
    - :func:`jax.numpy.digitize`
    - :func:`jax.numpy.unique_counts`

  Examples:
    Basic bincount:

    >>> x = jnp.array([1, 1, 2, 3, 3, 3])
    >>> jnp.bincount(x)
    Array([0, 2, 1, 3], dtype=int32)

    Weighted bincount:

    >>> weights = jnp.array([1, 2, 3, 4, 5, 6])
    >>> jnp.bincount(x, weights)
    Array([ 0,  3,  3, 15], dtype=int32)

    Specifying a static ``length`` makes this jit-compatible:

    >>> jit_bincount = jax.jit(jnp.bincount, static_argnames=['length'])
    >>> jit_bincount(x, length=5)
    Array([0, 2, 1, 3, 0], dtype=int32)

    Any negative numbers are clipped to the first bin, and numbers beyond the
    specified ``length`` are dropped:

    >>> x = jnp.array([-1, -1, 1, 3, 10])
    >>> jnp.bincount(x, length=5)
    Array([2, 1, 0, 1, 0], dtype=int32)
  """
  util.check_arraylike("bincount", x)
  if not issubdtype(_dtype(x), integer):
    raise TypeError(f"x argument to bincount must have an integer type; got {_dtype(x)}")
  if ndim(x) != 1:
    raise ValueError("only 1-dimensional input supported.")
  minlength = core.concrete_or_error(operator.index, minlength,
      "The error occurred because of argument 'minlength' of jnp.bincount.")
  if length is None:
    x_arr = core.concrete_or_error(asarray, x,
      "The error occurred because of argument 'x' of jnp.bincount. "
      "To avoid this error, pass a static `length` argument.")
    length = max(minlength, x_arr.size and int(x_arr.max()) + 1)
  else:
    length = core.concrete_dim_or_error(length,
        "The error occurred because of argument 'length' of jnp.bincount.")
  if weights is None:
    weights = np.array(1, dtype=int_)
  elif shape(x) != shape(weights):
    raise ValueError("shape of weights must match shape of x.")
  return zeros(length, _dtype(weights)).at[clip(x, 0)].add(weights)

@overload
def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]: ...

@overload
def broadcast_shapes(*shapes: Sequence[int | core.Tracer]
                     ) -> tuple[int | core.Tracer, ...]: ...

@util.implements(getattr(np, "broadcast_shapes", None))
def broadcast_shapes(*shapes):
  if not shapes:
    return ()
  shapes = [(shape,) if np.ndim(shape) == 0 else tuple(shape) for shape in shapes]
  return lax.broadcast_shapes(*shapes)


@util.implements(np.broadcast_arrays, lax_description="""\
The JAX version does not necessarily return a view of the input.
""")
def broadcast_arrays(*args: ArrayLike) -> list[Array]:
  return util._broadcast_arrays(*args)


@util.implements(np.broadcast_to, lax_description="""\
The JAX version does not necessarily return a view of the input.
""")
def broadcast_to(array: ArrayLike, shape: DimSize | Shape) -> Array:
  return util._broadcast_to(array, shape)


def _split(op: str, ary: ArrayLike,
           indices_or_sections: int | Sequence[int] | ArrayLike,
           axis: int = 0) -> list[Array]:
  util.check_arraylike(op, ary)
  ary = asarray(ary)
  axis = core.concrete_or_error(operator.index, axis, f"in jax.numpy.{op} argument `axis`")
  size = ary.shape[axis]
  if (isinstance(indices_or_sections, (tuple, list)) or
      isinstance(indices_or_sections, (np.ndarray, Array)) and
      indices_or_sections.ndim > 0):
    indices_or_sections = [
        core.concrete_dim_or_error(i_s, f"in jax.numpy.{op} argument 1")
        for i_s in indices_or_sections]
    split_indices = [0] + list(indices_or_sections) + [size]
  else:
    if core.is_symbolic_dim(indices_or_sections):
      raise ValueError(f"jax.numpy.{op} with a symbolic number of sections is "
                       "not supported")
    num_sections: int = core.concrete_or_error(int, indices_or_sections,
                                               f"in jax.numpy.{op} argument 1")
    part_size, r = divmod(size, num_sections)
    if r == 0:
      split_indices = [i * part_size
                       for i in range(num_sections + 1)]
    elif op == "array_split":
      split_indices = (
        [i * (part_size + 1) for i in range(r + 1)] +
        [i * part_size + ((r + 1) * (part_size + 1) - 1)
         for i in range(num_sections - r)])
    else:
      raise ValueError(f"array split does not result in an equal division: rest is {r}")
  split_indices = [i if core.is_symbolic_dim(i) else np.int64(i)  # type: ignore[misc]
                   for i in split_indices]
  starts, ends = [0] * ndim(ary), shape(ary)
  _subval = lambda x, i, v: subvals(x, [(i, v)])
  return [lax.slice(ary, _subval(starts, axis, start), _subval(ends, axis, end))
          for start, end in zip(split_indices[:-1], split_indices[1:])]

@util.implements(np.split, lax_description=_ARRAY_VIEW_DOC)
def split(ary: ArrayLike, indices_or_sections: int | Sequence[int] | ArrayLike,
          axis: int = 0) -> list[Array]:
  return _split("split", ary, indices_or_sections, axis=axis)

def _split_on_axis(op: str, axis: int) -> Callable[[ArrayLike, int | ArrayLike], list[Array]]:
  @util.implements(getattr(np, op), update_doc=False)
  def f(ary: ArrayLike, indices_or_sections: int | Sequence[int] | ArrayLike) -> list[Array]:
    # for 1-D array, hsplit becomes vsplit
    nonlocal axis
    util.check_arraylike(op, ary)
    a = asarray(ary)
    if axis == 1 and len(a.shape) == 1:
      axis = 0
    return _split(op, ary, indices_or_sections, axis=axis)
  return f

vsplit = _split_on_axis("vsplit", axis=0)
hsplit = _split_on_axis("hsplit", axis=1)
dsplit = _split_on_axis("dsplit", axis=2)

@util.implements(np.array_split)
def array_split(ary: ArrayLike, indices_or_sections: int | Sequence[int] | ArrayLike,
                axis: int = 0) -> list[Array]:
  return _split("array_split", ary, indices_or_sections, axis=axis)

deprecations.register("jax-numpy-clip-complex")

@jit
def clip(
  arr: ArrayLike | None = None,
  /,
  min: ArrayLike | None = None,
  max: ArrayLike | None = None,
  *,
  a: ArrayLike | DeprecatedArg = DeprecatedArg(),
  a_min: ArrayLike | None | DeprecatedArg = DeprecatedArg(),
  a_max: ArrayLike | None | DeprecatedArg = DeprecatedArg()
) -> Array:
  """Clip array values to a specified range.

  JAX implementation of :func:`numpy.clip`.

  Args:
    arr: N-dimensional array to be clipped.
    min: optional minimum value of the clipped range; if ``None`` (default) then
      result will not be clipped to any minimum value. If specified, it should be
      broadcast-compatible with ``arr`` and ``max``.
    max: optional maximum value of the clipped range; if ``None`` (default) then
      result will not be clipped to any maximum value. If specified, it should be
      broadcast-compatible with ``arr`` and ``min``.
    a: deprecated alias of the ``arr`` argument.  Will result in a
      :class:`DeprecationWarning` if used.
    a_min: deprecated alias of the ``min`` argument. Will result in a
      :class:`DeprecationWarning` if used.
    a_max: deprecated alias of the ``max`` argument. Will result in a
      :class:`DeprecationWarning` if used.

  Returns:
    An array containing values from ``arr``, with values smaller than ``min`` set
    to ``min``, and values larger than ``max`` set to ``max``.

  See also:
    - :func:`jax.numpy.minimum`: Compute the element-wise minimum value of two arrays.
    - :func:`jax.numpy.maximum`: Compute the element-wise maximum value of two arrays.

  Examples:
    >>> arr = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> jnp.clip(arr, 2, 5)
    Array([2, 2, 2, 3, 4, 5, 5, 5], dtype=int32)
  """
  # TODO(micky774): deprecated 2024-4-2, remove after deprecation expires.
  arr = a if not isinstance(a, DeprecatedArg) else arr
  if arr is None:
    raise ValueError("No input was provided to the clip function.")
  min = a_min if not isinstance(a_min, DeprecatedArg) else min
  max = a_max if not isinstance(a_max, DeprecatedArg) else max
  if any(not isinstance(t, DeprecatedArg) for t in (a, a_min, a_max)):
    warnings.warn(
      "Passing arguments 'a', 'a_min' or 'a_max' to jax.numpy.clip is "
      "deprecated. Please use 'arr', 'min' or 'max' respectively instead.",
      DeprecationWarning,
      stacklevel=2,
    )

  util.check_arraylike("clip", arr)
  if any(jax.numpy.iscomplexobj(t) for t in (arr, min, max)):
    # TODO(micky774): Deprecated 2024-4-2, remove after deprecation expires.
    deprecations.warn(
      "jax-numpy-clip-complex",
      "Clip received a complex value either through the input or the min/max "
      "keywords. Complex values have no ordering and cannot be clipped. "
      "Attempting to clip using complex numbers is deprecated and will soon "
      "raise a ValueError. Please convert to a real value or array by taking "
      "the real or imaginary components via jax.numpy.real/imag respectively.",
      stacklevel=2)
  if min is not None:
    arr = ufuncs.maximum(min, arr)
  if max is not None:
    arr = ufuncs.minimum(max, arr)
  return asarray(arr)

@util.implements(np.around, skip_params=['out'])
@partial(jit, static_argnames=('decimals',))
def round(a: ArrayLike, decimals: int = 0, out: None = None) -> Array:
  util.check_arraylike("round", a)
  decimals = core.concrete_or_error(operator.index, decimals, "'decimals' argument of jnp.round")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.round is not supported.")
  dtype = _dtype(a)
  if issubdtype(dtype, integer):
    if decimals < 0:
      raise NotImplementedError(
        "integer np.round not implemented for decimals < 0")
    return asarray(a)  # no-op on integer types

  def _round_float(x: ArrayLike) -> Array:
    if decimals == 0:
      return lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)

    # TODO(phawkins): the strategy of rescaling the value isn't necessarily a
    # good one since we may be left with an incorrectly rounded value at the
    # end due to precision problems. As a workaround for float16, convert to
    # float32,
    x = lax.convert_element_type(x, np.float32) if dtype == np.float16 else x
    factor = _lax_const(x, 10 ** decimals)
    out = lax.div(lax.round(lax.mul(x, factor),
                            lax.RoundingMethod.TO_NEAREST_EVEN), factor)
    return lax.convert_element_type(out, dtype) if dtype == np.float16 else out

  if issubdtype(dtype, complexfloating):
    return lax.complex(_round_float(lax.real(a)), _round_float(lax.imag(a)))
  else:
    return _round_float(a)
around = round
round_ = round


@util.implements(np.fix, skip_params=['out'])
@jit
def fix(x: ArrayLike, out: None = None) -> Array:
  util.check_arraylike("fix", x)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.fix is not supported.")
  zero = _lax_const(x, 0)
  return where(lax.ge(x, zero), ufuncs.floor(x), ufuncs.ceil(x))


@util.implements(np.nan_to_num)
@jit
def nan_to_num(x: ArrayLike, copy: bool = True, nan: ArrayLike = 0.0,
               posinf: ArrayLike | None = None,
               neginf: ArrayLike | None = None) -> Array:
  del copy
  util.check_arraylike("nan_to_num", x)
  dtype = _dtype(x)
  if not issubdtype(dtype, inexact):
    return asarray(x)
  if issubdtype(dtype, complexfloating):
    return lax.complex(
      nan_to_num(lax.real(x), nan=nan, posinf=posinf, neginf=neginf),
      nan_to_num(lax.imag(x), nan=nan, posinf=posinf, neginf=neginf))
  info = finfo(dtypes.canonicalize_dtype(dtype))
  posinf = info.max if posinf is None else posinf
  neginf = info.min if neginf is None else neginf
  out = where(ufuncs.isnan(x), asarray(nan, dtype=dtype), x)
  out = where(ufuncs.isposinf(out), asarray(posinf, dtype=dtype), out)
  out = where(ufuncs.isneginf(out), asarray(neginf, dtype=dtype), out)
  return out


@partial(jit, static_argnames=('equal_nan',))
def allclose(a: ArrayLike, b: ArrayLike, rtol: ArrayLike = 1e-05,
             atol: ArrayLike = 1e-08, equal_nan: bool = False) -> Array:
  r"""Check if two arrays are element-wise approximately equal within a tolerance.

  JAX implementation of :func:`numpy.allclose`.

  Essentially this function evaluates the following condition:

  .. math::

     |a - b| \le \mathtt{atol} + \mathtt{rtol} * |b|

  ``jnp.inf`` in ``a`` will be considered equal to ``jnp.inf`` in ``b``.

  Args:
    a: first input array to compare.
    b: second input array to compare.
    rtol: relative tolerance used for approximate equality. Default = 1e-05.
    atol: absolute tolerance used for approximate equality. Default = 1e-08.
    equal_nan: Boolean. If ``True``, NaNs in ``a`` will be considered
      equal to NaNs in ``b``. Default is ``False``.

  Returns:
    Boolean scalar array indicating whether the input arrays are element-wise
    approximately equal within the specified tolerances.

  See Also:
    - :func:`jax.numpy.isclose`
    - :func:`jax.numpy.equal`

  Examples:
    >>> jnp.allclose(jnp.array([1e6, 2e6, 3e6]), jnp.array([1e6, 2e6, 3e7]))
    Array(False, dtype=bool)
    >>> jnp.allclose(jnp.array([1e6, 2e6, 3e6]),
    ...              jnp.array([1.00008e6, 2.00008e7, 3.00008e8]), rtol=1e3)
    Array(True, dtype=bool)
    >>> jnp.allclose(jnp.array([1e6, 2e6, 3e6]),
    ...              jnp.array([1.00001e6, 2.00002e6, 3.00009e6]), atol=1e3)
    Array(True, dtype=bool)
    >>> jnp.allclose(jnp.array([jnp.nan, 1, 2]),
    ...              jnp.array([jnp.nan, 1, 2]), equal_nan=True)
    Array(True, dtype=bool)
  """
  util.check_arraylike("allclose", a, b)
  return reductions.all(isclose(a, b, rtol, atol, equal_nan))


def nonzero(a: ArrayLike, *, size: int | None = None,
            fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None
    ) -> tuple[Array, ...]:
  """Return indices of nonzero elements of an array.

  JAX implementation of :func:`numpy.nonzero`.

  Because the size of the output of ``nonzero`` is data-dependent, the function
  is not compatible with JIT and other transformations. The JAX version adds
  the optional ``size`` argument which must be specified statically for
  ``jnp.nonzero`` to be used within JAX's transformations.

  Args:
    a: N-dimensional array.
    size: optional static integer specifying the number of nonzero entries to
      return. If there are more nonzero elements than the specified ``size``,
      then indices will be truncated at the end. If there are fewer nonzero
      elements than the specified size, then indices will be padded with
      ``fill_value``, which defaults to zero.
    fill_value: optional padding value when ``size`` is specified. Defaults to 0.

  Returns:
    Tuple of JAX Arrays of length ``a.ndim``, containing the indices of each
    nonzero value.

  See also:
    - :func:`jax.numpy.flatnonzero`
    - :func:`jax.numpy.where`

  Examples:

    One-dimensional array returns a length-1 tuple of indices:

    >>> x = jnp.array([0, 5, 0, 6, 0, 7])
    >>> jnp.nonzero(x)
    (Array([1, 3, 5], dtype=int32),)

    Two-dimensional array returns a length-2 tuple of indices:

    >>> x = jnp.array([[0, 5, 0],
    ...                [6, 0, 7]])
    >>> jnp.nonzero(x)
    (Array([0, 1, 1], dtype=int32), Array([1, 0, 2], dtype=int32))

    In either case, the resulting tuple of indices can be used directly to extract
    the nonzero values:

    >>> indices = jnp.nonzero(x)
    >>> x[indices]
    Array([5, 6, 7], dtype=int32)

    The output of ``nonzero`` has a dynamic shape, because the number of returned
    indices depends on the contents of the input array. As such, it is incompatible
    with JIT and other JAX transformations:

    >>> x = jnp.array([0, 5, 0, 6, 0, 7])
    >>> jax.jit(jnp.nonzero)(x)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[].
    The size argument of jnp.nonzero must be statically specified to use jnp.nonzero within JAX transformations.

    This can be addressed by passing a static ``size`` parameter to specify the
    desired output shape:

    >>> nonzero_jit = jax.jit(jnp.nonzero, static_argnames='size')
    >>> nonzero_jit(x, size=3)
    (Array([1, 3, 5], dtype=int32),)

    If ``size`` does not match the true size, the result will be either truncated or padded:

    >>> nonzero_jit(x, size=2)  # size < 3: indices are truncated
    (Array([1, 3], dtype=int32),)
    >>> nonzero_jit(x, size=5)  # size > 3: indices are padded with zeros.
    (Array([1, 3, 5, 0, 0], dtype=int32),)

    You can specify a custom fill value for the padding using the ``fill_value`` argument:

    >>> nonzero_jit(x, size=5, fill_value=len(x))
    (Array([1, 3, 5, 6, 6], dtype=int32),)
  """
  util.check_arraylike("nonzero", a)
  arr = asarray(a)
  del a
  if ndim(arr) == 0:
    raise ValueError("Calling nonzero on 0d arrays is not allowed. "
                     "Use jnp.atleast_1d(scalar).nonzero() instead.")
  mask = arr if arr.dtype == bool else (arr != 0)
  calculated_size_ = mask.sum() if size is None else size
  calculated_size: int = core.concrete_dim_or_error(calculated_size_,
    "The size argument of jnp.nonzero must be statically specified "
    "to use jnp.nonzero within JAX transformations.")
  if arr.size == 0 or calculated_size == 0:
    return tuple(zeros(calculated_size, int) for dim in arr.shape)
  flat_indices = reductions.cumsum(
      bincount(reductions.cumsum(mask), length=calculated_size))
  strides: np.ndarray = (np.cumprod(arr.shape[::-1])[::-1] // arr.shape).astype(int_)
  out = tuple((flat_indices // stride) % size for stride, size in zip(strides, arr.shape))
  if fill_value is not None:
    fill_value_tup = fill_value if isinstance(fill_value, tuple) else arr.ndim * (fill_value,)
    if any(_shape(val) != () for val in fill_value_tup):
      raise ValueError(f"fill_value must be a scalar or a tuple of length {arr.ndim}; got {fill_value}")
    fill_mask = arange(calculated_size) >= mask.sum()
    out = tuple(where(fill_mask, fval, entry) for fval, entry in safe_zip(fill_value_tup, out))
  return out


def flatnonzero(a: ArrayLike, *, size: int | None = None,
                fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None) -> Array:
  """Return indices of nonzero elements in a flattened array

  JAX implementation of :func:`numpy.flatnonzero`.

  ``jnp.flatnonzero(x)`` is equivalent to ``nonzero(ravel(a))[0]``. For a full
  discussion of the parameters to this function, refer to :func:`jax.numpy.nonzero`.

  Args:
    a: N-dimensional array.
    size: optional static integer specifying the number of nonzero entries to
      return. See :func:`jax.numpy.nonzero` for more discussion of this parameter.
    fill_value: optional padding value when ``size`` is specified. Defaults to 0.
      See :func:`jax.numpy.nonzero` for more discussion of this parameter.

  Returns:
    Array containing the indices of each nonzero value in the flattened array.

  See Also:
    - :func:`jax.numpy.nonzero`
    - :func:`jax.numpy.where`

  Examples:
    >>> x = jnp.array([[0, 5, 0],
    ...                [6, 0, 8]])
    >>> jnp.flatnonzero(x)
    Array([1, 3, 5], dtype=int32)

    This is equivalent to calling :func:`~jax.numpy.nonzero` on the flattened
    array, and extracting the first entry in the resulting tuple:

    >>> jnp.nonzero(x.ravel())[0]
    Array([1, 3, 5], dtype=int32)

    The returned indices can be used to extract nonzero entries from the
    flattened array:

    >>> indices = jnp.flatnonzero(x)
    >>> x.ravel()[indices]
    Array([5, 6, 8], dtype=int32)
  """
  return nonzero(ravel(a), size=size, fill_value=fill_value)[0]


@util.implements(np.unwrap)
@partial(jit, static_argnames=('axis',))
def unwrap(p: ArrayLike, discont: ArrayLike | None = None,
           axis: int = -1, period: ArrayLike = 2 * pi) -> Array:
  util.check_arraylike("unwrap", p)
  p = asarray(p)
  if issubdtype(p.dtype, np.complexfloating):
    raise ValueError("jnp.unwrap does not support complex inputs.")
  if p.shape[axis] == 0:
    return util.promote_dtypes_inexact(p)[0]
  if discont is None:
    discont = period / 2
  interval = period / 2
  dd = diff(p, axis=axis)
  ddmod = ufuncs.mod(dd + interval, period) - interval
  ddmod = where((ddmod == -interval) & (dd > 0), interval, ddmod)

  ph_correct = where(ufuncs.abs(dd) < discont, 0, ddmod - dd)

  up = concatenate((
    lax.slice_in_dim(p, 0, 1, axis=axis),
    lax.slice_in_dim(p, 1, None, axis=axis) + reductions.cumsum(ph_correct, axis=axis)
  ), axis=axis)

  return up


### Padding

PadValueLike = Union[T, Sequence[T], Sequence[Sequence[T]]]
PadValue = tuple[tuple[T, T], ...]

class PadStatFunc(Protocol):
  def __call__(self, array: ArrayLike, /, *,
               axis: int | None = None,
               keepdims: bool = False) -> Array: ...


def _broadcast_to_pairs(nvals: PadValueLike, nd: int, name: str) -> PadValue:
  try:
    nvals = np.asarray(tree_map(
      lambda x: core.concrete_or_error(None, x, context=f"{name} argument of jnp.pad"),
      nvals))
  except ValueError as e:
    # In numpy 1.24
    if "array has an inhomogeneous shape" in str(e):
      raise TypeError(f'`{name}` entries must be the same shape: {nvals}') from e
    raise

  def as_scalar_dim(v):
    if core.is_dim(v) or not np.shape(v):
      return v
    else:
      raise TypeError(f'`{name}` entries must be the same shape: {nvals}')

  if nvals.shape == (nd, 2):
    # ((before_1, after_1), ..., (before_N, after_N))
    return tuple((as_scalar_dim(nval[0]), as_scalar_dim(nval[1])) for nval in nvals)
  elif nvals.shape == (1, 2):
    # ((before, after),)
    v1_2 = as_scalar_dim(nvals[0, 0]), as_scalar_dim(nvals[0, 1])
    return tuple(v1_2 for i in range(nd))
  elif nvals.shape == (2,):
    # (before, after)  (not in the numpy docstring but works anyway)
    v1_2 = as_scalar_dim(nvals[0]), as_scalar_dim(nvals[1])
    return tuple(v1_2 for i in range(nd))
  elif nvals.shape == (1,):
    # (pad,)
    v = as_scalar_dim(nvals[0])
    return tuple((v, v) for i in range(nd))
  elif nvals.shape == ():
    # pad
    v = as_scalar_dim(nvals.flat[0])
    return tuple((v, v) for i in range(nd))
  else:
    raise ValueError(f"jnp.pad: {name} with {nd=} has unsupported shape {nvals.shape}. "
                     f"Valid shapes are ({nd}, 2), (1, 2), (2,), (1,), or ().")


def _check_no_padding(axis_padding: tuple[Any, Any], mode: str):
  if (axis_padding[0] > 0 or axis_padding[1] > 0):
    msg = "Cannot apply '{}' padding to empty axis"
    raise ValueError(msg.format(mode))


def _pad_constant(array: Array, pad_width: PadValue[int], constant_values: Array) -> Array:
  nd = ndim(array)
  constant_values = broadcast_to(constant_values, (nd, 2))
  constant_values = lax_internal._convert_element_type(
      constant_values, array.dtype, dtypes.is_weakly_typed(array))
  for i in range(nd):
    widths = [(0, 0, 0)] * nd
    widths[i] = (pad_width[i][0], 0, 0)
    array = lax.pad(array, constant_values[i, 0], widths)
    widths[i] = (0, pad_width[i][1], 0)
    array = lax.pad(array, constant_values[i, 1], widths)
  return array


def _pad_wrap(array: Array, pad_width: PadValue[int]) -> Array:
  for i in range(ndim(array)):
    if array.shape[i] == 0:
      _check_no_padding(pad_width[i], "wrap")
      continue
    size = array.shape[i]
    left_repeats, left_remainder = divmod(pad_width[i][0], size)
    right_repeats, right_remainder = divmod(pad_width[i][1], size)
    total_repeats = left_repeats + right_repeats + 1
    parts = []
    if left_remainder > 0:
      parts += [lax.slice_in_dim(array, size - left_remainder, size, axis=i)]
    parts += total_repeats * [array]
    if right_remainder > 0:
      parts += [lax.slice_in_dim(array, 0, right_remainder, axis=i)]
    array = lax.concatenate(parts, dimension=i)
  return array


def _pad_symmetric_or_reflect(array: Array, pad_width: PadValue[int],
                              mode: str, reflect_type: str) -> Array:
  assert mode in ("symmetric", "reflect")
  assert reflect_type in ("even", "odd")

  for i in range(ndim(array)):
    if array.shape[i] == 0:
      _check_no_padding(pad_width[i], mode)
      continue

    axis_size = array.shape[i]

    def build_padding(array, padding, before):
      if before:
        edge = lax.slice_in_dim(array, 0, 1, axis=i)
      else:
        edge = lax.slice_in_dim(array, -1, None, axis=i)

      # Try to give nicer error messages for unsupported shape polymorphic uses
      shape_poly_error_msg = lambda: (
          "Shape polymorphism is supported for jnp.pad with 'reflect' or "
          "'symmetric' padding mode only when it is possible to determine "
          f"at lowering time that the axis size (= {axis_size}) is larger than 1 "
          f"and larger or equal than the padding length (= {padding}). "
          f"Error while handling {'left' if before else 'right'} padding on axis {i}.")
      try:
        # We check that we can determine all comparisions.
        offset = 1 if (mode == "reflect" and axis_size > 1) else 0
        has_poly_dim = not core.is_constant_shape((axis_size, padding))
        # For shape polymorphism, ensure the loop below ends after 1 iteration
        if has_poly_dim and not (axis_size > 1 and axis_size - offset >= padding):
          raise ValueError(shape_poly_error_msg())
      except core.InconclusiveDimensionOperation as e:
        raise ValueError(shape_poly_error_msg()) from e

      while padding > 0:
        curr_pad = min(padding, axis_size - offset)
        padding -= curr_pad
        if has_poly_dim: assert padding == 0

        if before:
          start = offset
          stop = offset + curr_pad
        else:
          start = -(curr_pad + offset)
          stop = None if (mode == "symmetric" or axis_size == 1) else -1

        x = lax.slice_in_dim(array, start, stop, axis=i)
        x = flip(x, axis=i)

        if reflect_type == 'odd':
          x = 2 * edge - x
          if axis_size > 1:
            if before:
              edge = lax.slice_in_dim(x, 0, 1, axis=i)
            else:
              edge = lax.slice_in_dim(x, -1, None, axis=i)

        if before:
          array = lax.concatenate([x, array], dimension=i)
        else:
          array = lax.concatenate([array, x], dimension=i)
      return array

    array = build_padding(array, pad_width[i][0], before=True)
    array = build_padding(array, pad_width[i][1], before=False)
  return array


def _pad_edge(array: Array, pad_width: PadValue[int]) -> Array:
  nd = ndim(array)
  for i in range(nd):
    if array.shape[i] == 0:
      _check_no_padding(pad_width[i], "edge")
      continue

    n = array.shape[i]
    npad_before, npad_after = pad_width[i]

    edge_before = lax.slice_in_dim(array, 0, 1, axis=i)
    pad_before = repeat(edge_before, npad_before, axis=i)

    edge_after = lax.slice_in_dim(array, n-1, n, axis=i)
    pad_after = repeat(edge_after, npad_after, axis=i)

    array = lax.concatenate([pad_before, array, pad_after], dimension=i)
  return array


def _pad_linear_ramp(array: Array, pad_width: PadValue[int],
                     end_values: PadValue[ArrayLike]) -> Array:
  for axis in range(ndim(array)):
    edge_before = lax.slice_in_dim(array, 0, 1, axis=axis)
    edge_after = lax.slice_in_dim(array, -1, None, axis=axis)
    ramp_before = linspace(
        start=end_values[axis][0],
        stop=edge_before.squeeze(axis), # Dimension is replaced by linspace
        num=pad_width[axis][0],
        endpoint=False,
        dtype=array.dtype,
        axis=axis
    )
    ramp_before = lax_internal._convert_element_type(
        ramp_before, weak_type=dtypes.is_weakly_typed(array))
    ramp_after = linspace(
        start=end_values[axis][1],
        stop=edge_after.squeeze(axis), # Dimension is replaced by linspace
        num=pad_width[axis][1],
        endpoint=False,
        dtype=array.dtype,
        axis=axis
    )
    ramp_after = lax_internal._convert_element_type(
        ramp_after, weak_type=dtypes.is_weakly_typed(array))

    # Reverse linear space in appropriate dimension
    ramp_after = flip(ramp_after, axis)

    array = lax.concatenate([ramp_before, array, ramp_after], dimension=axis)
  return array


def _pad_stats(array: Array, pad_width: PadValue[int],
               stat_length: PadValue[int] | None,
               stat_func: PadStatFunc) -> Array:
  nd = ndim(array)
  for i in range(nd):
    if stat_length is None:
      stat_before = stat_func(array, axis=i, keepdims=True)
      stat_after = stat_before
    else:
      array_length = array.shape[i]
      length_before, length_after = stat_length[i]
      if length_before == 0 or length_after == 0:
        raise ValueError("stat_length of 0 yields no value for padding")

      # Limit stat_length to length of array.
      length_before = min(length_before, array_length)
      length_after = min(length_after, array_length)

      slice_before = lax.slice_in_dim(array, 0, length_before, axis=i)
      slice_after = lax.slice_in_dim(array, -length_after, None, axis=i)
      stat_before = stat_func(slice_before, axis=i, keepdims=True)
      stat_after = stat_func(slice_after, axis=i, keepdims=True)

    if np.issubdtype(array.dtype, np.integer):
      stat_before = round(stat_before)
      stat_after = round(stat_after)

    stat_before = lax_internal._convert_element_type(
        stat_before, array.dtype, dtypes.is_weakly_typed(array))
    stat_after = lax_internal._convert_element_type(
        stat_after, array.dtype, dtypes.is_weakly_typed(array))

    npad_before, npad_after = pad_width[i]
    pad_before = repeat(stat_before, npad_before, axis=i)
    pad_after = repeat(stat_after, npad_after, axis=i)

    array = lax.concatenate([pad_before, array, pad_after], dimension=i)
  return array


def _pad_empty(array: Array, pad_width: PadValue[int]) -> Array:
  # Note: jax.numpy.empty = jax.numpy.zeros
  for i in range(ndim(array)):
    shape_before = array.shape[:i] + (pad_width[i][0],) + array.shape[i + 1:]
    pad_before = empty_like(array, shape=shape_before)

    shape_after = array.shape[:i] + (pad_width[i][1],) + array.shape[i + 1:]
    pad_after = empty_like(array, shape=shape_after)
    array = lax.concatenate([pad_before, array, pad_after], dimension=i)
  return array


def _pad_func(array: Array, pad_width: PadValue[int], func: Callable[..., Any], **kwargs) -> Array:
  pad_width = _broadcast_to_pairs(pad_width, ndim(array), "pad_width")
  padded = _pad_constant(array, pad_width, asarray(0))
  for axis in range(ndim(padded)):
    padded = apply_along_axis(func, axis, padded, pad_width[axis], axis, kwargs)
  return padded


@partial(jit, static_argnums=(1, 2, 4, 5, 6))
def _pad(array: ArrayLike, pad_width: PadValueLike[int], mode: str,
         constant_values: ArrayLike, stat_length: PadValueLike[int],
         end_values: PadValueLike[ArrayLike], reflect_type: str):
  array = asarray(array)
  nd = ndim(array)

  if nd == 0:
    return array

  stat_funcs: dict[str, PadStatFunc] = {
      "maximum": reductions.amax,
      "minimum": reductions.amin,
      "mean": reductions.mean,
      "median": reductions.median
  }

  pad_width = _broadcast_to_pairs(pad_width, nd, "pad_width")
  pad_width_arr = np.array(pad_width)
  if pad_width_arr.shape != (nd, 2):
    raise ValueError(f"Expected pad_width to have shape {(nd, 2)}; got {pad_width_arr.shape}.")

  if np.any(pad_width_arr < 0):
    raise ValueError("index can't contain negative values")

  if mode == "constant":
    return _pad_constant(array, pad_width, asarray(constant_values))

  elif mode == "wrap":
    return _pad_wrap(array, pad_width)

  elif mode in ("symmetric", "reflect"):
    return _pad_symmetric_or_reflect(array, pad_width, str(mode), reflect_type)

  elif mode == "edge":
    return _pad_edge(array, pad_width)

  elif mode == "linear_ramp":
    end_values = _broadcast_to_pairs(end_values, nd, "end_values")
    return _pad_linear_ramp(array, pad_width, end_values)

  elif mode in stat_funcs:
    if stat_length is not None:
      stat_length = _broadcast_to_pairs(stat_length, nd, "stat_length")
    return _pad_stats(array, pad_width, stat_length, stat_funcs[str(mode)])

  elif mode == "empty":
    return _pad_empty(array, pad_width)

  else:
    assert False, ("Should not be reached since pad already handled unsupported and"
                   "not implemented modes")


@util.implements(np.pad, lax_description="""\
Unlike numpy, JAX "function" mode's argument (which is another function) should return
the modified array. This is because Jax arrays are immutable.
(In numpy, "function" mode's argument should modify a rank 1 array in-place.)
""")
def pad(array: ArrayLike, pad_width: PadValueLike[int | Array | np.ndarray],
        mode: str | Callable[..., Any] = "constant", **kwargs) -> Array:
  util.check_arraylike("pad", array)
  pad_width = _broadcast_to_pairs(pad_width, ndim(array), "pad_width")
  if pad_width and not all(core.is_dim(p[0]) and core.is_dim(p[1])
                           for p in pad_width):
    raise TypeError('`pad_width` must be of integral type.')

  if callable(mode):
    return _pad_func(asarray(array), pad_width, mode, **kwargs)

  allowed_kwargs = {
      'empty': [], 'edge': [], 'wrap': [],
      'constant': ['constant_values'],
      'linear_ramp': ['end_values'],
      'maximum': ['stat_length'],
      'mean': ['stat_length'],
      'median': ['stat_length'],
      'minimum': ['stat_length'],
      'reflect': ['reflect_type'],
      'symmetric': ['reflect_type'],
  }
  try:
    unsupported_kwargs = set(kwargs) - set(allowed_kwargs[mode])
  except KeyError:
    msg = "Unimplemented padding mode '{}' for np.pad."
    raise NotImplementedError(msg.format(mode))
  if unsupported_kwargs:
    raise ValueError("unsupported keyword arguments for mode '{}': {}"
                     .format(mode, unsupported_kwargs))
  # Set default value if not given.
  constant_values = kwargs.get('constant_values', 0)
  stat_length = kwargs.get('stat_length', None)
  end_values = kwargs.get('end_values', 0)
  reflect_type = kwargs.get('reflect_type', "even")

  return _pad(array, pad_width, mode, constant_values, stat_length, end_values, reflect_type)

### Array-creation functions


@util.implements(np.stack, skip_params=['out'])
def stack(arrays: np.ndarray | Array | Sequence[ArrayLike],
          axis: int = 0, out: None = None, dtype: DTypeLike | None = None) -> Array:
  if not len(arrays):
    raise ValueError("Need at least one array to stack.")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.stack is not supported.")
  if isinstance(arrays, (np.ndarray, Array)):
    axis = _canonicalize_axis(axis, arrays.ndim)
    return concatenate(expand_dims(arrays, axis + 1), axis=axis, dtype=dtype)
  else:
    util.check_arraylike("stack", *arrays)
    shape0 = shape(arrays[0])
    axis = _canonicalize_axis(axis, len(shape0) + 1)
    new_arrays = []
    for a in arrays:
      if shape(a) != shape0:
        raise ValueError("All input arrays must have the same shape.")
      new_arrays.append(expand_dims(a, axis))
    return concatenate(new_arrays, axis=axis, dtype=dtype)

@util.implements(getattr(np, 'unstack', None))
@partial(jit, static_argnames="axis")
def unstack(x: ArrayLike, /, *, axis: int = 0) -> tuple[Array, ...]:
  util.check_arraylike("unstack", x)
  x = asarray(x)
  if x.ndim == 0:
    raise ValueError(
      "Unstack requires arrays with rank > 0, however a scalar array was "
      "passed."
    )
  return tuple(moveaxis(x, axis, 0))

@util.implements(np.tile)
def tile(A: ArrayLike, reps: DimSize | Sequence[DimSize]) -> Array:
  util.check_arraylike("tile", A)
  try:
    iter(reps)  # type: ignore[arg-type]
  except TypeError:
    reps_tup: tuple[DimSize, ...] = (reps,)
  else:
    reps_tup = tuple(reps)  # type: ignore[arg-type]
  reps_tup = tuple(operator.index(rep) if core.is_constant_dim(rep) else rep
                   for rep in reps_tup)
  A_shape = (1,) * (len(reps_tup) - ndim(A)) + shape(A)
  reps_tup = (1,) * (len(A_shape) - len(reps_tup)) + reps_tup
  result = broadcast_to(reshape(A, [j for i in A_shape for j in [1, i]]),
                        [k for pair in zip(reps_tup, A_shape) for k in pair])
  return reshape(result, tuple(np.multiply(A_shape, reps_tup)))

def _concatenate_array(arr: ArrayLike, axis: int | None,
                       dtype: DTypeLike | None = None) -> Array:
  # Fast path for concatenation when the input is an ndarray rather than a list.
  arr = asarray(arr, dtype=dtype)
  if arr.ndim == 0 or arr.shape[0] == 0:
    raise ValueError("Need at least one array to concatenate.")
  if axis is None:
    return lax.reshape(arr, (arr.size,))
  if arr.ndim == 1:
    raise ValueError("Zero-dimensional arrays cannot be concatenated.")
  axis = _canonicalize_axis(axis, arr.ndim - 1)
  shape = arr.shape[1:axis + 1] + (arr.shape[0] * arr.shape[axis + 1],) + arr.shape[axis + 2:]
  dimensions = [*range(1, axis + 1), 0, *range(axis + 1, arr.ndim)]
  return lax.reshape(arr, shape, dimensions)

@util.implements(np.concatenate)
def concatenate(arrays: np.ndarray | Array | Sequence[ArrayLike],
                axis: int | None = 0, dtype: DTypeLike | None = None) -> Array:
  if isinstance(arrays, (np.ndarray, Array)):
    return _concatenate_array(arrays, axis, dtype=dtype)
  util.check_arraylike("concatenate", *arrays)
  if not len(arrays):
    raise ValueError("Need at least one array to concatenate.")
  if axis is None:
    return concatenate([ravel(a) for a in arrays], axis=0, dtype=dtype)
  if ndim(arrays[0]) == 0:
    raise ValueError("Zero-dimensional arrays cannot be concatenated.")
  axis = _canonicalize_axis(axis, ndim(arrays[0]))
  if dtype is None:
    arrays_out = util.promote_dtypes(*arrays)
  else:
    arrays_out = [asarray(arr, dtype=dtype) for arr in arrays]
  # lax.concatenate can be slow to compile for wide concatenations, so form a
  # tree of concatenations as a workaround especially for op-by-op mode.
  # (https://github.com/google/jax/issues/653).
  k = 16
  while len(arrays_out) > 1:
    arrays_out = [lax.concatenate(arrays_out[i:i+k], axis)
                  for i in range(0, len(arrays_out), k)]
  return arrays_out[0]


@util.implements(getattr(np, "concat", None))
def concat(arrays: Sequence[ArrayLike], /, *, axis: int | None = 0) -> Array:
  util.check_arraylike("concat", *arrays)
  return jax.numpy.concatenate(arrays, axis=axis)


@util.implements(np.vstack)
def vstack(tup: np.ndarray | Array | Sequence[ArrayLike],
           dtype: DTypeLike | None = None) -> Array:
  arrs: Array | list[Array]
  if isinstance(tup, (np.ndarray, Array)):
    arrs = jax.vmap(atleast_2d)(tup)
  else:
    # TODO(jakevdp): Non-array input deprecated 2023-09-22; change to error.
    util.check_arraylike("vstack", *tup, emit_warning=True)
    arrs = [atleast_2d(m) for m in tup]
  return concatenate(arrs, axis=0, dtype=dtype)


@util.implements(np.hstack)
def hstack(tup: np.ndarray | Array | Sequence[ArrayLike],
           dtype: DTypeLike | None = None) -> Array:
  arrs: Array | list[Array]
  if isinstance(tup, (np.ndarray, Array)):
    arrs = jax.vmap(atleast_1d)(tup)
    arr0_ndim = arrs.ndim - 1
  else:
    # TODO(jakevdp): Non-array input deprecated 2023-09-22; change to error.
    util.check_arraylike("hstack", *tup, emit_warning=True)
    arrs = [atleast_1d(m) for m in tup]
    arr0_ndim = arrs[0].ndim
  return concatenate(arrs, axis=0 if arr0_ndim == 1 else 1, dtype=dtype)


@util.implements(np.dstack)
def dstack(tup: np.ndarray | Array | Sequence[ArrayLike],
           dtype: DTypeLike | None = None) -> Array:
  arrs: Array | list[Array]
  if isinstance(tup, (np.ndarray, Array)):
    arrs = jax.vmap(atleast_3d)(tup)
  else:
    # TODO(jakevdp): Non-array input deprecated 2023-09-22; change to error.
    util.check_arraylike("dstack", *tup, emit_warning=True)
    arrs = [atleast_3d(m) for m in tup]
  return concatenate(arrs, axis=2, dtype=dtype)


@util.implements(np.column_stack)
def column_stack(tup: np.ndarray | Array | Sequence[ArrayLike]) -> Array:
  arrs: Array | list[Array] | np.ndarray
  if isinstance(tup, (np.ndarray, Array)):
    arrs = jax.vmap(lambda x: atleast_2d(x).T)(tup) if tup.ndim < 3 else tup
  else:
    # TODO(jakevdp): Non-array input deprecated 2023-09-22; change to error.
    util.check_arraylike("column_stack", *tup, emit_warning=True)
    arrs = [atleast_2d(arr).T if arr.ndim < 2 else arr for arr in map(asarray, tup)]
  return concatenate(arrs, 1)


@util.implements(np.choose, skip_params=['out'])
def choose(a: ArrayLike, choices: Sequence[ArrayLike],
           out: None = None, mode: str = 'raise') -> Array:
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.choose is not supported.")
  util.check_arraylike('choose', a, *choices)
  if not issubdtype(_dtype(a), integer):
    raise ValueError("`a` array must be integer typed")
  N = len(choices)

  if mode == 'raise':
    arr: Array = core.concrete_or_error(asarray, a,
      "The error occurred because jnp.choose was jit-compiled"
      " with mode='raise'. Use mode='wrap' or mode='clip' instead.")
    if reductions.any((arr < 0) | (arr >= N)):
      raise ValueError("invalid entry in choice array")
  elif mode == 'wrap':
    arr = asarray(a) % N
  elif mode == 'clip':
    arr = clip(a, 0, N - 1)
  else:
    raise ValueError(f"mode={mode!r} not understood. Must be 'raise', 'wrap', or 'clip'")

  arr, *choices = broadcast_arrays(arr, *choices)
  return array(choices)[(arr,) + indices(arr.shape, sparse=True)]


def _atleast_nd(x: ArrayLike, n: int) -> Array:
  m = ndim(x)
  return lax.broadcast(x, (1,) * (n - m)) if m < n else asarray(x)

def _block(xs: ArrayLike | list[ArrayLike]) -> tuple[Array, int]:
  if isinstance(xs, tuple):
    raise ValueError("jax.numpy.block does not allow tuples, got {}"
                     .format(xs))
  elif isinstance(xs, list):
    if len(xs) == 0:
      raise ValueError("jax.numpy.block does not allow empty list arguments")
    xs_tup, depths = unzip2([_block(x) for x in xs])
    if any(d != depths[0] for d in depths[1:]):
      raise ValueError("Mismatched list depths in jax.numpy.block")
    rank = max(depths[0], max(ndim(x) for x in xs_tup))
    xs_tup = tuple(_atleast_nd(x, rank) for x in xs_tup)
    return concatenate(xs_tup, axis=-depths[0]), depths[0] + 1
  else:
    return asarray(xs), 1

@util.implements(np.block)
@jit
def block(arrays: ArrayLike | list[ArrayLike]) -> Array:
  out, _ = _block(arrays)
  return out


@overload
def atleast_1d() -> list[Array]:
  ...
@overload
def atleast_1d(x: ArrayLike, /) -> Array:
  ...
@overload
def atleast_1d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]:
  ...
@util.implements(np.atleast_1d, update_doc=False, lax_description=_ARRAY_VIEW_DOC)
@jit
def atleast_1d(*arys: ArrayLike) -> Array | list[Array]:
  # TODO(jakevdp): Non-array input deprecated 2023-09-22; change to error.
  util.check_arraylike("atleast_1d", *arys, emit_warning=True)
  if len(arys) == 1:
    return array(arys[0], copy=False, ndmin=1)
  else:
    return [array(arr, copy=False, ndmin=1) for arr in arys]


@overload
def atleast_2d() -> list[Array]:
  ...
@overload
def atleast_2d(x: ArrayLike, /) -> Array:
  ...
@overload
def atleast_2d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]:
  ...
@util.implements(np.atleast_2d, update_doc=False, lax_description=_ARRAY_VIEW_DOC)
@jit
def atleast_2d(*arys: ArrayLike) -> Array | list[Array]:
  # TODO(jakevdp): Non-array input deprecated 2023-09-22; change to error.
  util.check_arraylike("atleast_2d", *arys, emit_warning=True)
  if len(arys) == 1:
    return array(arys[0], copy=False, ndmin=2)
  else:
    return [array(arr, copy=False, ndmin=2) for arr in arys]


@overload
def atleast_3d() -> list[Array]:
  ...
@overload
def atleast_3d(x: ArrayLike, /) -> Array:
  ...
@overload
def atleast_3d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]:
  ...
@util.implements(np.atleast_3d, update_doc=False, lax_description=_ARRAY_VIEW_DOC)
@jit
def atleast_3d(*arys: ArrayLike) -> Array | list[Array]:
  # TODO(jakevdp): Non-array input deprecated 2023-09-22; change to error.
  util.check_arraylike("atleast_3d", *arys, emit_warning=True)
  if len(arys) == 1:
    arr = asarray(arys[0])
    if arr.ndim == 0:
      arr = lax.expand_dims(arr, dimensions=(0, 1, 2))
    elif arr.ndim == 1:
      arr = lax.expand_dims(arr, dimensions=(0, 2))
    elif arr.ndim == 2:
      arr = lax.expand_dims(arr, dimensions=(2,))
    return arr
  else:
    return [atleast_3d(arr) for arr in arys]


def _supports_buffer_protocol(obj):
  try:
    view = memoryview(obj)
  except TypeError:
    return False
  else:
    return True


_ARRAY_DOC = """
This function will create arrays on JAX's default device. For control of the
device placement of data, see :func:`jax.device_put`. More information is
available in the JAX FAQ at :ref:`faq-data-placement` (full FAQ at
https://jax.readthedocs.io/en/latest/faq.html).
"""

deprecations.register("jax-numpy-array-none")

@util.implements(np.array, lax_description=_ARRAY_DOC, extra_params="""
device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
  to which the created array will be committed.
""")
def array(object: Any, dtype: DTypeLike | None = None, copy: bool = True,
          order: str | None = "K", ndmin: int = 0,
          *, device: xc.Device | Sharding | None = None) -> Array:
  if order is not None and order != "K":
    raise NotImplementedError("Only implemented for order='K'")

  # check if the given dtype is compatible with JAX
  dtypes.check_user_dtype_supported(dtype, "array")

  # Here we make a judgment call: we only return a weakly-typed array when the
  # input object itself is weakly typed. That ensures asarray(x) is a no-op
  # whenever x is weak, but avoids introducing weak types with something like
  # array([1, 2, 3])
  weak_type = dtype is None and dtypes.is_weakly_typed(object)
  sharding = canonicalize_device_to_sharding(device)

  # Use device_put to avoid a copy for ndarray inputs.
  if (not copy and isinstance(object, np.ndarray) and
      (dtype is None or dtype == object.dtype) and (ndmin <= object.ndim) and
      device is None):
    # Keep the output uncommitted.
    return jax.device_put(object)

  # For Python scalar literals, call coerce_to_array to catch any overflow
  # errors. We don't use dtypes.is_python_scalar because we don't want this
  # triggering for traced values. We do this here because it matters whether or
  # not dtype is None. We don't assign the result because we want the raw object
  # to be used for type inference below.
  if isinstance(object, (bool, int, float, complex)):
    _ = dtypes.coerce_to_array(object, dtype)
  elif not isinstance(object, Array):
    # Check if object supports any of the data exchange protocols
    # (except dlpack, see data-apis/array-api#301). If it does,
    # consume the object as jax array and continue (but not return) so
    # that other array() arguments get processed against the input
    # object.
    #
    # Notice that data exchange protocols define dtype in the
    # corresponding data structures and it may not be available as
    # object.dtype. So, we'll resolve the protocols here before
    # evaluating object.dtype.
    if hasattr(object, '__jax_array__'):
      object = object.__jax_array__()
    elif hasattr(object, '__cuda_array_interface__'):
      cai = object.__cuda_array_interface__
      backend = xla_bridge.get_backend("cuda")
      if cuda_plugin_extension is None:
        device_id = None
      else:
        device_id = cuda_plugin_extension.get_device_ordinal(cai["data"][0])
      object = xc._xla.cuda_array_interface_to_buffer(
          cai=cai, gpu_backend=backend, device_id=device_id)

  object = tree_map(lambda leaf: leaf.__jax_array__()
                    if hasattr(leaf, "__jax_array__") else leaf, object)
  leaves = tree_leaves(object, is_leaf=lambda x: x is None)
  if any(leaf is None for leaf in leaves):
    # Added Nov 16 2023
    if deprecations.is_accelerated("jax-numpy-array-none"):
      raise TypeError("None is not a valid value for jnp.array")
    warnings.warn(
      "None encountered in jnp.array(); this is currently treated as NaN. "
      "In the future this will result in an error.",
      FutureWarning, stacklevel=2)
    leaves = tree_leaves(object)
  if dtype is None:
    # Use lattice_result_type rather than result_type to avoid canonicalization.
    # Otherwise, weakly-typed inputs would have their dtypes canonicalized.
    try:
      dtype = dtypes._lattice_result_type(*leaves)[0] if leaves else dtypes.float_
    except TypeError:
      # This happens if, e.g. one of the entries is a memoryview object.
      # This is rare, so we only handle it if the normal path fails.
      leaves = [_convert_to_array_if_dtype_fails(leaf) for leaf in leaves]
      dtype = dtypes._lattice_result_type(*leaves)[0]

  if not weak_type:
    dtype = dtypes.canonicalize_dtype(dtype, allow_extended_dtype=True)  # type: ignore[assignment]

  out: ArrayLike

  if all(not isinstance(leaf, Array) for leaf in leaves):
    # TODO(jakevdp): falling back to numpy here fails to overflow for lists
    # containing large integers; see discussion in
    # https://github.com/google/jax/pull/6047. More correct would be to call
    # coerce_to_array on each leaf, but this may have performance implications.
    out = np.asarray(object, dtype=dtype)
  elif isinstance(object, Array):
    assert object.aval is not None
    out = _array_copy(object) if copy else object
  elif isinstance(object, (list, tuple)):
    if object:
      out = stack([asarray(elt, dtype=dtype) for elt in object])
    else:
      out = np.array([], dtype=dtype)
  elif _supports_buffer_protocol(object):
    object = memoryview(object)
    # TODO(jakevdp): update this once we support NumPy 2.0 semantics for the copy arg.
    out = np.array(object) if copy else np.asarray(object)
  else:
    raise TypeError(f"Unexpected input type for array: {type(object)}")
  out_array: Array = lax_internal._convert_element_type(
      out, dtype, weak_type=weak_type, sharding=sharding)
  if ndmin > ndim(out_array):
    out_array = lax.expand_dims(out_array, range(ndmin - ndim(out_array)))
  return out_array


def canonicalize_device_to_sharding(device: xc.Device | Sharding | None
                                    ) -> Sharding | None:
  if isinstance(device, xc.Device):
    return SingleDeviceSharding(device)
  return device


def _convert_to_array_if_dtype_fails(x: ArrayLike) -> ArrayLike:
  try:
    dtypes.dtype(x)
  except TypeError:
    return np.asarray(x)
  else:
    return x


deprecations.register("jax-numpy-astype-complex-to-real")

def astype(x: ArrayLike, dtype: DTypeLike | None,
           /, *, copy: bool = False,
           device: xc.Device | Sharding | None = None) -> Array:
  """Convert an array to a specified dtype.

  JAX imlementation of :func:`numpy.astype`.

  This is implemented via :func:`jax.lax.convert_element_type`, which may
  have slightly different behavior than :func:`numpy.astype` in some cases.
  In particular, the details of float-to-int and int-to-float casts are
  implementation dependent.

  Args:
    x: input array to convert
    dtype: output dtype
    copy: if True, then always return a copy. If False (default) then only
      return a copy if necessary.
    device: optionally specify the device to which the output will be committed.

  Returns:
    An array with the same shape as ``x``, containing values of the specified
    dtype.

  See Also:
    - :func:`jax.lax.convert_element_type`: lower-level function for XLA-style
      dtype conversions.

  Examples:
    >>> x = jnp.array([0, 1, 2, 3])
    >>> x
    Array([0, 1, 2, 3], dtype=int32)
    >>> x.astype('float32')
    Array([0.0, 1.0, 2.0, 3.0], dtype=float32)

    >>> y = jnp.array([0.0, 0.5, 1.0])
    >>> y.astype(int)  # truncates fractional values
    Array([0, 0, 1], dtype=int32)
  """
  util.check_arraylike("astype", x)
  x_arr = asarray(x)

  if dtype is None:
    dtype = dtypes.canonicalize_dtype(float_)
  dtypes.check_user_dtype_supported(dtype, "astype")
  if issubdtype(x_arr.dtype, complexfloating):
    if dtypes.isdtype(dtype, ("integral", "real floating")):
      deprecations.warn(
        "jax-numpy-astype-complex-to-real",
        "Casting from complex to real dtypes will soon raise a ValueError. "
        "Please first use jnp.real or jnp.imag to take the real/imaginary "
        "component of your input.",
        stacklevel=2)
    elif np.dtype(dtype) == bool:
      # convert_element_type(complex, bool) has the wrong semantics.
      x_arr = (x_arr != _lax_const(x_arr, 0))

  # We offer a more specific warning than the usual ComplexWarning so we prefer
  # to issue our warning.
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", ComplexWarning)
    result = lax_internal._convert_element_type(
      x_arr, dtype, sharding=_normalize_to_sharding(device))
  return _array_copy(result) if copy else result


@util.implements(np.asarray, lax_description=_ARRAY_DOC, extra_params="""
device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
  to which the created array will be committed.
""")
def asarray(a: Any, dtype: DTypeLike | None = None, order: str | None = None,
            *, copy: bool | None = None,
            device: xc.Device | Sharding | None = None) -> Array:
  # For copy=False, the array API specifies that we raise a ValueError if the input supports
  # the buffer protocol but a copy is required. Since array() supports the buffer protocol
  # via numpy, this is only the case when the default device is not 'cpu'
  if (copy is False and not isinstance(a, Array)
      and jax.default_backend() != 'cpu'
      and _supports_buffer_protocol(a)):
    raise ValueError(f"jnp.asarray: cannot convert object of type {type(a)} to JAX Array "
                     f"on backend={jax.default_backend()!r} with copy=False. "
                      "Consider using copy=None or copy=True instead.")
  dtypes.check_user_dtype_supported(dtype, "asarray")
  if dtype is not None:
    dtype = dtypes.canonicalize_dtype(dtype, allow_extended_dtype=True)  # type: ignore[assignment]
  return array(a, dtype=dtype, copy=bool(copy), order=order, device=device)


@util.implements(np.copy, lax_description=_ARRAY_DOC)
def copy(a: ArrayLike, order: str | None = None) -> Array:
  util.check_arraylike("copy", a)
  return array(a, copy=True, order=order)


def zeros_like(a: ArrayLike | DuckTypedArray,
               dtype: DTypeLike | None = None,
               shape: Any = None, *,
               device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of zeros with the same shape and dtype as an array.

  JAX implementation of :func:`numpy.zeros_like`.

  Args:
    a: Array-like object with ``shape`` and ``dtype`` attributes.
    shape: optionally override the shape of the created array.
    dtype: optionally override the dtype of the created array.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.zeros`
    - :func:`jax.numpy.empty_like`
    - :func:`jax.numpy.ones_like`
    - :func:`jax.numpy.full_like`

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.zeros_like(x)
    Array([0, 0, 0, 0], dtype=int32)
    >>> jnp.zeros_like(x, dtype=bool)
    Array([False, False, False, False], dtype=bool)
    >>> jnp.zeros_like(x, shape=(2, 3))
    Array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)
  """
  if not (hasattr(a, 'dtype') and hasattr(a, 'shape')):  # support duck typing
    util.check_arraylike("zeros_like", a)
  dtypes.check_user_dtype_supported(dtype, "zeros_like")
  if shape is not None:
    shape = canonicalize_shape(shape)
  return lax.full_like(a, 0, dtype, shape, sharding=_normalize_to_sharding(device))


def ones_like(a: ArrayLike | DuckTypedArray,
              dtype: DTypeLike | None = None,
              shape: Any = None, *,
              device: xc.Device | Sharding | None = None) -> Array:
  """Create an array of ones with the same shape and dtype as an array.

  JAX implementation of :func:`numpy.ones_like`.

  Args:
    a: Array-like object with ``shape`` and ``dtype`` attributes.
    shape: optionally override the shape of the created array.
    dtype: optionally override the dtype of the created array.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.zeros_like`
    - :func:`jax.numpy.ones_like`
    - :func:`jax.numpy.full_like`

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.ones_like(x)
    Array([1, 1, 1, 1], dtype=int32)
    >>> jnp.ones_like(x, dtype=bool)
    Array([ True,  True,  True,  True], dtype=bool)
    >>> jnp.ones_like(x, shape=(2, 3))
    Array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
  """
  if not (hasattr(a, 'dtype') and hasattr(a, 'shape')):  # support duck typing
    util.check_arraylike("ones_like", a)
  dtypes.check_user_dtype_supported(dtype, "ones_like")
  if shape is not None:
    shape = canonicalize_shape(shape)
  return lax.full_like(a, 1, dtype, shape, sharding=_normalize_to_sharding(device))


def empty_like(prototype: ArrayLike | DuckTypedArray,
               dtype: DTypeLike | None = None,
               shape: Any = None, *,
               device: xc.Device | Sharding | None = None) -> Array:
  """Create an empty array with the same shape and dtype as an array.

  JAX implementation of :func:`numpy.empty_like`. Because XLA cannot create
  an un-initialized array, :func:`jax.numpy.empty` will always return an
  array full of zeros.

  Args:
    a: Array-like object with ``shape`` and ``dtype`` attributes.
    shape: optionally override the shape of the created array.
    dtype: optionally override the dtype of the created array.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.zeros_like`
    - :func:`jax.numpy.ones_like`
    - :func:`jax.numpy.full_like`

  Examples:
    >>> x = jnp.arange(4)
    >>> jnp.empty_like(x)
    Array([0, 0, 0, 0], dtype=int32)
    >>> jnp.empty_like(x, dtype=bool)
    Array([False, False, False, False], dtype=bool)
    >>> jnp.empty_like(x, shape=(2, 3))
    Array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)
  """
  if not (hasattr(prototype, 'dtype') and hasattr(prototype, 'shape')):  # support duck typing
    util.check_arraylike("empty_like", prototype)
  dtypes.check_user_dtype_supported(dtype, "empty_like")
  return zeros_like(prototype, dtype=dtype, shape=shape, device=device)


def _normalize_to_sharding(device: xc.Device | Sharding | None) -> Sharding | None:
  if isinstance(device, xc.Device):
    return SingleDeviceSharding(device)
  else:
    return device


def full(shape: Any, fill_value: ArrayLike,
         dtype: DTypeLike | None = None, *,
         device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of a specified value.

  JAX implementation of :func:`numpy.full`.

  Args:
    shape: int or sequence of ints specifying the shape of the created array.
    fill_value: scalar or array with which to fill the created array.
    dtype: optional dtype for the created array; defaults to the dtype of the
      fill value.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.full_like`
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.zeros`
    - :func:`jax.numpy.ones`

  Examples:
    >>> jnp.full(4, 2, dtype=float)
    Array([2., 2., 2., 2.], dtype=float32)
    >>> jnp.full((2, 3), 0, dtype=bool)
    Array([[False, False, False],
           [False, False, False]], dtype=bool)

    `fill_value` may also be an array that is broadcast to the specified shape:

    >>> jnp.full((2, 3), fill_value=jnp.arange(3))
    Array([[0, 1, 2],
           [0, 1, 2]], dtype=int32)
  """
  dtypes.check_user_dtype_supported(dtype, "full")
  util.check_arraylike("full", fill_value)

  if ndim(fill_value) == 0:
    shape = canonicalize_shape(shape)
    return lax.full(shape, fill_value, dtype, sharding=_normalize_to_sharding(device))
  else:
    return jax.device_put(
        broadcast_to(asarray(fill_value, dtype=dtype), shape), device)


def full_like(a: ArrayLike | DuckTypedArray,
              fill_value: ArrayLike, dtype: DTypeLike | None = None,
              shape: Any = None, *,
              device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of a specified value with the same shape and dtype as an array.

  JAX implementation of :func:`numpy.full_like`.

  Args:
    a: Array-like object with ``shape`` and ``dtype`` attributes.
    fill_value: scalar or array with which to fill the created array.
    shape: optionally override the shape of the created array.
    dtype: optionally override the dtype of the created array.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.full`
    - :func:`jax.numpy.empty_like`
    - :func:`jax.numpy.zeros_like`
    - :func:`jax.numpy.ones_like`

  Examples:
    >>> x = jnp.arange(4.0)
    >>> jnp.full_like(x, 2)
    Array([2., 2., 2., 2.], dtype=float32)
    >>> jnp.full_like(x, 0, shape=(2, 3))
    Array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

    `fill_value` may also be an array that is broadcast to the specified shape:

    >>> x = jnp.arange(6).reshape(2, 3)
    >>> jnp.full_like(x, fill_value=jnp.array([[1], [2]]))
    Array([[1, 1, 1],
           [2, 2, 2]], dtype=int32)
  """
  if hasattr(a, 'dtype') and hasattr(a, 'shape'):  # support duck typing
    util.check_arraylike("full_like", 0, fill_value)
  else:
    util.check_arraylike("full_like", a, fill_value)
  dtypes.check_user_dtype_supported(dtype, "full_like")
  if shape is not None:
    shape = canonicalize_shape(shape)
  if ndim(fill_value) == 0:
    return lax.full_like(a, fill_value, dtype, shape, sharding=_normalize_to_sharding(device))
  else:
    shape = np.shape(a) if shape is None else shape  # type: ignore[arg-type]
    dtype = result_type(a) if dtype is None else dtype
    return jax.device_put(
        broadcast_to(asarray(fill_value, dtype=dtype), shape), device)


def zeros(shape: Any, dtype: DTypeLike | None = None, *,
          device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of zeros.

  JAX implementation of :func:`numpy.zeros`.

  Args:
    shape: int or sequence of ints specifying the shape of the created array.
    dtype: optional dtype for the created array; defaults to floating point.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.zeros_like`
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.ones`
    - :func:`jax.numpy.full`

  Examples:
    >>> jnp.zeros(4)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> jnp.zeros((2, 3), dtype=bool)
    Array([[False, False, False],
           [False, False, False]], dtype=bool)
  """
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  if (m := _check_forgot_shape_tuple("zeros", shape, dtype)): raise TypeError(m)
  dtypes.check_user_dtype_supported(dtype, "zeros")
  shape = canonicalize_shape(shape)
  return lax.full(shape, 0, _jnp_dtype(dtype), sharding=_normalize_to_sharding(device))


def ones(shape: Any, dtype: DTypeLike | None = None, *,
         device: xc.Device | Sharding | None = None) -> Array:
  """Create an array full of ones.

  JAX implementation of :func:`numpy.ones`.

  Args:
    shape: int or sequence of ints specifying the shape of the created array.
    dtype: optional dtype for the created array; defaults to floating point.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.ones_like`
    - :func:`jax.numpy.empty`
    - :func:`jax.numpy.zeros`
    - :func:`jax.numpy.full`

  Examples:
    >>> jnp.ones(4)
    Array([1., 1., 1., 1.], dtype=float32)
    >>> jnp.ones((2, 3), dtype=bool)
    Array([[ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
  """
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  if (m := _check_forgot_shape_tuple("ones", shape, dtype)): raise TypeError(m)
  shape = canonicalize_shape(shape)
  dtypes.check_user_dtype_supported(dtype, "ones")
  return lax.full(shape, 1, _jnp_dtype(dtype), sharding=_normalize_to_sharding(device))


def empty(shape: Any, dtype: DTypeLike | None = None, *,
          device: xc.Device | Sharding | None = None) -> Array:
  """Create an empty array.

  JAX implementation of :func:`numpy.empty`. Because XLA cannot create an
  un-initialized array, :func:`jax.numpy.empty` will always return an array
  full of zeros.

  Args:
    shape: int or sequence of ints specifying the shape of the created array.
    dtype: optional dtype for the created array; defaults to floating point.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of the specified shape and dtype, on the specified device if specified.

  See also:
    - :func:`jax.numpy.empty_like`
    - :func:`jax.numpy.zeros`
    - :func:`jax.numpy.ones`
    - :func:`jax.numpy.full`

  Examples:
    >>> jnp.empty(4)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> jnp.empty((2, 3), dtype=bool)
    Array([[False, False, False],
           [False, False, False]], dtype=bool)
  """
  if (m := _check_forgot_shape_tuple("empty", shape, dtype)): raise TypeError(m)
  dtypes.check_user_dtype_supported(dtype, "empty")
  return zeros(shape, dtype, device=device)

def _check_forgot_shape_tuple(name, shape, dtype) -> str | None:  # type: ignore
  if isinstance(dtype, int) and isinstance(shape, int):
    return (f"Cannot interpret '{dtype}' as a data type."
            f"\n\nDid you accidentally write "
            f"`jax.numpy.{name}({shape}, {dtype})` "
            f"when you meant `jax.numpy.{name}(({shape}, {dtype}))`, i.e. "
            "with a single tuple argument for the shape?")


def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: bool = False) -> Array:
  """Check if two arrays are element-wise equal.

  JAX implementation of :func:`numpy.array_equal`.

  Args:
    a1: first input array to compare.
    a2: second input array to compare.
    equal_nan: Boolean. If ``True``, NaNs in ``a1`` will be considered
      equal to NaNs in ``a2``. Default is ``False``.

  Returns:
    Boolean scalar array indicating whether the input arrays are element-wise equal.

  See Also:
    - :func:`jax.numpy.allclose`
    - :func:`jax.numpy.array_equiv`

  Examples:
    >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    Array(True, dtype=bool)
    >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2]))
    Array(False, dtype=bool)
    >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))
    Array(False, dtype=bool)
    >>> jnp.array_equal(jnp.array([1, 2, float('nan')]),
    ...                 jnp.array([1, 2, float('nan')]))
    Array(False, dtype=bool)
    >>> jnp.array_equal(jnp.array([1, 2, float('nan')]),
    ...                 jnp.array([1, 2, float('nan')]), equal_nan=True)
    Array(True, dtype=bool)
  """
  a1, a2 = asarray(a1), asarray(a2)
  if shape(a1) != shape(a2):
    return bool_(False)
  eq = asarray(a1 == a2)
  if equal_nan:
    eq = ufuncs.logical_or(eq, ufuncs.logical_and(ufuncs.isnan(a1), ufuncs.isnan(a2)))
  return reductions.all(eq)


def array_equiv(a1: ArrayLike, a2: ArrayLike) -> Array:
  """Check if two arrays are element-wise equal.

  JAX implementation of :func:`numpy.array_equiv`.

  This function will return ``False`` if the input arrays cannot be broadcasted
  to the same shape.

  Args:
    a1: first input array to compare.
    a2: second input array to compare.

  Returns:
    Boolean scalar array indicating whether the input arrays are
    element-wise equal after broadcasting.

  See Also:
    - :func:`jax.numpy.allclose`
    - :func:`jax.numpy.array_equal`

  Examples:
    >>> jnp.array_equiv(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    Array(True, dtype=bool)
    >>> jnp.array_equiv(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))
    Array(False, dtype=bool)
    >>> jnp.array_equiv(jnp.array([[1, 2, 3], [1, 2, 3]]),
    ...                 jnp.array([1, 2, 3]))
    Array(True, dtype=bool)
  """
  a1, a2 = asarray(a1), asarray(a2)
  try:
    eq = ufuncs.equal(a1, a2)
  except ValueError:
    # shapes are not broadcastable
    return bool_(False)
  return reductions.all(eq)


# General np.from* style functions mostly delegate to numpy.

@util.implements(np.frombuffer)
def frombuffer(buffer: bytes | Any, dtype: DTypeLike = float,
               count: int = -1, offset: int = 0) -> Array:
  return asarray(np.frombuffer(buffer=buffer, dtype=dtype, count=count, offset=offset))


def fromfile(*args, **kwargs):
  """Unimplemented JAX wrapper for jnp.fromfile.

  This function is left deliberately unimplemented because it may be non-pure and thus
  unsafe for use with JIT and other JAX transformations. Consider using
  ``jnp.asarray(np.fromfile(...))`` instead, although care should be taken if ``np.fromfile``
  is used within jax transformations because of its potential side-effect of consuming the
  file object; for more information see `Common Gotchas: Pure Functions
  <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_.
  """
  raise NotImplementedError(
    "jnp.fromfile() is not implemented because it may be non-pure and thus unsafe for use "
    "with JIT and other JAX transformations. Consider using jnp.asarray(np.fromfile(...)) "
    "instead, although care should be taken if np.fromfile is used within a jax transformations "
    "because of its potential side-effect of consuming the file object; for more information see "
    "https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions")


def fromiter(*args, **kwargs):
  """Unimplemented JAX wrapper for jnp.fromiter.

  This function is left deliberately unimplemented because it may be non-pure and thus
  unsafe for use with JIT and other JAX transformations. Consider using
  ``jnp.asarray(np.fromiter(...))`` instead, although care should be taken if ``np.fromiter``
  is used within jax transformations because of its potential side-effect of consuming the
  iterable object; for more information see `Common Gotchas: Pure Functions
  <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_.
  """
  raise NotImplementedError(
    "jnp.fromiter() is not implemented because it may be non-pure and thus unsafe for use "
    "with JIT and other JAX transformations. Consider using jnp.asarray(np.fromiter(...)) "
    "instead, although care should be taken if np.fromiter is used within a jax transformations "
    "because of its potential side-effect of consuming the iterable object; for more information see "
    "https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions")

@util.implements(getattr(np, "from_dlpack", None), lax_description="""
.. note::

   While JAX arrays are always immutable, dlpack buffers cannot be marked as
   immutable, and it is possible for processes external to JAX to mutate them
   in-place. If a jax Array is constructed from a dlpack buffer and the buffer
   is later modified in-place, it may lead to undefined behavior when using
   the associated JAX array.
""")
def from_dlpack(x: Any, /, *, device: xc.Device | Sharding | None = None,
                copy: bool | None = None) -> Array:
  from jax.dlpack import from_dlpack  # pylint: disable=g-import-not-at-top
  return from_dlpack(x, device=device, copy=copy)

@util.implements(np.fromfunction)
def fromfunction(function: Callable[..., Array], shape: Any,
                 *, dtype: DTypeLike = float, **kwargs) -> Array:
  shape = core.canonicalize_shape(shape, context="shape argument of jnp.fromfunction()")
  for i in range(len(shape)):
    in_axes = [0 if i == j else None for j in range(len(shape))]
    function = jax.vmap(function, in_axes=tuple(in_axes[::-1]))
  return function(*(arange(s, dtype=dtype) for s in shape), **kwargs)


@util.implements(np.fromstring)
def fromstring(string: str, dtype: DTypeLike = float, count: int = -1, *, sep: str) -> Array:
  return asarray(np.fromstring(string=string, dtype=dtype, count=count, sep=sep))


def eye(N: DimSize, M: DimSize | None = None,
        k: int | ArrayLike = 0,
        dtype: DTypeLike | None = None,
        *, device: xc.Device | Sharding | None = None) -> Array:
  """Create a square or rectangular identity matrix

  JAX implementation of :func:`numpy.eye`.

  Args:
    N: integer specifying the first dimension of the array.
    M: optional integer specifying the second dimension of the array;
      defaults to the same value as ``N``.
    k: optional integer specifying the offset of the diagonal. Use positive
      values for upper diagonals, and negative values for lower diagonals.
      Default is zero.
    dtype: optional dtype; defaults to floating point.
    device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Identity array of shape ``(N, M)``, or ``(N, N)`` if ``M`` is not specified.

  See also:
    :func:`jax.numpy.identity`: Simpler API for generating square identity matrices.

  Examples:
    A simple 3x3 identity matrix:

    >>> jnp.eye(3)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

    Integer identity matrices with offset diagonals:

    >>> jnp.eye(3, k=1, dtype=int)
    Array([[0, 1, 0],
           [0, 0, 1],
           [0, 0, 0]], dtype=int32)
    >>> jnp.eye(3, k=-1, dtype=int)
    Array([[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0]], dtype=int32)

    Non-square identity matrix:

    >>> jnp.eye(3, 5, k=1)
    Array([[0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.]], dtype=float32)
  """
  # TODO(vfdev-5): optimize putting the array directly on the device specified
  # instead of putting it on default device and then on the specific device
  output = _eye(N, M=M, k=k, dtype=dtype)
  if device is not None:
    return jax.device_put(output, device=device)
  return output


def _eye(N: DimSize, M: DimSize | None = None,
        k: int | ArrayLike = 0,
        dtype: DTypeLike | None = None) -> Array:
  dtypes.check_user_dtype_supported(dtype, "eye")
  if isinstance(k, int):
    k = lax_internal._clip_int_to_valid_range(k, np.int32)
  util.check_arraylike("eye", k)
  offset = asarray(k)
  if not (offset.shape == () and dtypes.issubdtype(offset.dtype, np.integer)):
    raise ValueError(f"k must be a scalar integer; got {k}")
  N_int = core.canonicalize_dim(N, "'N' argument of jnp.eye()")
  M_int = N_int if M is None else core.canonicalize_dim(M, "'M' argument of jnp.eye()")
  if N_int < 0 or M_int < 0:
    raise ValueError(f"negative dimensions are not allowed, got {N} and {M}")
  i = lax.broadcasted_iota(offset.dtype, (N_int, M_int), 0)
  j = lax.broadcasted_iota(offset.dtype, (N_int, M_int), 1)
  return (i + offset == j).astype(dtype)


def identity(n: DimSize, dtype: DTypeLike | None = None) -> Array:
  """Create a square identity matrix

  JAX implementation of :func:`numpy.identity`.

  Args:
    n: integer specifying the size of each array dimension.
    dtype: optional dtype; defaults to floating point.

  Returns:
    Identity array of shape ``(n, n)``.

  See also:
    :func:`jax.numpy.eye`: non-square and/or offset identity matrices.

  Examples:
    A simple 3x3 identity matrix:

    >>> jnp.identity(3)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

    A 2x2 integer identity matrix:

    >>> jnp.identity(2, dtype=int)
    Array([[1, 0],
           [0, 1]], dtype=int32)
  """
  dtypes.check_user_dtype_supported(dtype, "identity")
  return eye(n, dtype=dtype)


def arange(start: ArrayLike | DimSize, stop: ArrayLike | DimSize | None = None,
           step: ArrayLike | None = None, dtype: DTypeLike | None = None,
           *, device: xc.Device | Sharding | None = None) -> Array:
  """Create an array of evenly-spaced values.

  JAX implementation of :func:`numpy.arange`, implemented in terms of
  :func:`jax.lax.iota`.

  Similar to Python's :func:`range` function, this can be called with a few
  different positional signatures:

  - ``jnp.arange(stop)``: generate values from 0 to ``stop``, stepping by 1.
  - ``jnp.arange(start, stop)``: generate values from ``start`` to ``stop``,
    stepping by 1.
  - ``jnp.arange(start, stop, step)``: generate values from ``start`` to ``stop``,
    stepping by ``step``.

  Like with Python's :func:`range` function, the starting value is inclusive,
  and the stop value is exclusive.

  Args:
    start: start of the interval, inclusive.
    stop: optional end of the interval, exclusive. If not specified, then
      ``(start, stop) = (0, start)``
    step: optional step size for the interval. Default = 1.
    dtype: optional dtype for the returned array; if not specified it will
      be determined via type promotion of `start`, `stop`, and `step`.
    device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of evenly-spaced values from ``start`` to ``stop``, separated by ``step``.

  Note:
    Using ``arange`` with a floating-point ``step`` argument can lead to unexpected
    results due to accumulation of floating-point errors, especially with
    lower-precision data types like ``float8_*`` and ``bfloat16``.
    To avoid precision errors, consider generating a range of integers, and scaling
    it to the desired range. For example, instead of this::

       jnp.arange(-1, 1, 0.01, dtype='bfloat16')

    it can be more accurate to generate a sequence of integers, and scale them::

       (jnp.arange(-100, 100) * 0.01).astype('bfloat16')

  Examples:
    Single-argument version specifies only the ``stop`` value:

    >>> jnp.arange(4)
    Array([0, 1, 2, 3], dtype=int32)

    Passing a floating-point ``stop`` value leads to a floating-point result:

    >>> jnp.arange(4.0)
    Array([0., 1., 2., 3.], dtype=float32)

    Two-argument version specifies ``start`` and ``stop``, with ``step=1``:

    >>> jnp.arange(1, 6)
    Array([1, 2, 3, 4, 5], dtype=int32)

    Three-argument version specifies ``start``, ``stop``, and ``step``:

    >>> jnp.arange(0, 2, 0.5)
    Array([0. , 0.5, 1. , 1.5], dtype=float32)

  See Also:
    - :func:`jax.numpy.linspace`: generate a fixed number of evenly-spaced values.
    - :func:`jax.lax.iota`: directly generate integer sequences in XLA.
  """
  # TODO(vfdev-5): optimize putting the array directly on the device specified
  # instead of putting it on default device and then on the specific device
  output = _arange(start, stop=stop, step=step, dtype=dtype)
  if device is not None:
    return jax.device_put(output, device=device)
  return output


def _arange(start: ArrayLike | DimSize, stop: ArrayLike | DimSize | None = None,
            step: ArrayLike | None = None, dtype: DTypeLike | None = None) -> Array:
  dtypes.check_user_dtype_supported(dtype, "arange")
  if not config.dynamic_shapes.value:
    util.check_arraylike("arange", start)
    if stop is None and step is None:
      start = core.concrete_or_error(None, start, "It arose in the jnp.arange argument 'stop'")
    else:
      start = core.concrete_or_error(None, start, "It arose in the jnp.arange argument 'start'")
  util.check_arraylike_or_none("arange", None, stop, step)
  stop = core.concrete_or_error(None, stop, "It arose in the jnp.arange argument 'stop'")
  step = core.concrete_or_error(None, step, "It arose in the jnp.arange argument 'step'")
  start_name = "stop" if stop is None and step is None else "start"
  for name, val in [(start_name, start), ("stop", stop), ("step", step)]:
    if val is not None and np.ndim(val) != 0:
      raise ValueError(f"jax.numpy.arange: arguments must be scalars; got {name}={val}")
  if any(core.is_symbolic_dim(v) for v in (start, stop, step)):
    # Some dynamic shapes
    if stop is None and step is None:
      stop = start
      start = 0
      step = 1
    elif stop is not None and step is None:
      step = 1
    return _arange_dynamic(start, stop, step, dtype or dtypes.canonicalize_dtype(np.int64))
  if dtype is None:
    dtype = result_type(start, *(x for x in [stop, step] if x is not None))
  dtype = _jnp_dtype(dtype)
  if stop is None and step is None:
    start_dtype = _dtype(start)
    if (not dtypes.issubdtype(start_dtype, np.integer) and
        not dtypes.issubdtype(start_dtype, dtypes.extended)):
      ceil_ = ufuncs.ceil if isinstance(start, core.Tracer) else np.ceil
      start = ceil_(start).astype(int)  # type: ignore[operator]
    return lax.iota(dtype, start)  # type: ignore[arg-type]
  else:
    if step is None and start == 0 and stop is not None:
      return lax.iota(dtype, np.ceil(stop).astype(int))  # type: ignore[arg-type]
    return array(np.arange(start, stop=stop, step=step, dtype=dtype))


def _arange_dynamic(
    start: DimSize, stop: DimSize, step: DimSize, dtype: DTypeLike) -> Array:
  # Here if at least one of start, stop, step are dynamic.
  if any(not core.is_dim(v) for v in (start, stop, step)):
    raise ValueError(
        "In arange with non-constant arguments all of start, stop, and step "
        f"must be either dimension expressions or integers: start={start}, "
        f"stop={stop}, step={step}")
  # Must resolve statically if step is {<0, ==0, >0}
  try:
    if step == 0:
      raise ValueError("arange has step == 0")
    step_gt_0 = (step > 0)
  except core.InconclusiveDimensionOperation as e:
    raise core.InconclusiveDimensionOperation(
        f"In arange with non-constant arguments the step ({step}) must " +
        f"be resolved statically if it is > 0 or < 0.\nDetails: {e}")
  gap = step if step_gt_0 else - step
  distance = (stop - start) if step_gt_0 else (start - stop)
  size = core.max_dim(0, distance + gap - 1) // gap
  return (array(start, dtype=dtype) +
          array(step, dtype=dtype) * lax.iota(dtype, size))

@overload
def linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, retstep: Literal[False] = False,
             dtype: DTypeLike | None = None,
             axis: int = 0,
             *, device: xc.Device | Sharding | None = None) -> Array: ...
@overload
def linspace(start: ArrayLike, stop: ArrayLike, num: int,
             endpoint: bool, retstep: Literal[True],
             dtype: DTypeLike | None = None,
             axis: int = 0,
             *, device: xc.Device | Sharding | None = None) -> tuple[Array, Array]: ...
@overload
def linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, *, retstep: Literal[True],
             dtype: DTypeLike | None = None,
             axis: int = 0,
             device: xc.Device | Sharding | None = None) -> tuple[Array, Array]: ...
@overload
def linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, retstep: bool = False,
             dtype: DTypeLike | None = None,
             axis: int = 0,
             *, device: xc.Device | Sharding | None = None) -> Array | tuple[Array, Array]: ...
@util.implements(np.linspace, extra_params="""
device: (optional) :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
  to which the created array will be committed.
""")
def linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, retstep: bool = False,
             dtype: DTypeLike | None = None,
             axis: int = 0,
             *, device: xc.Device | Sharding | None = None) -> Array | tuple[Array, Array]:
  num = core.concrete_dim_or_error(num, "'num' argument of jnp.linspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.linspace")
  return _linspace(start, stop, num, endpoint, retstep, dtype, axis, device=device)

@partial(jit, static_argnames=('num', 'endpoint', 'retstep', 'dtype', 'axis', 'device'))
def _linspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
              endpoint: bool = True, retstep: bool = False,
              dtype: DTypeLike | None = None,
              axis: int = 0,
              *, device: xc.Device | Sharding | None = None) -> Array | tuple[Array, Array]:
  """Implementation of linspace differentiable in start and stop args."""
  dtypes.check_user_dtype_supported(dtype, "linspace")
  if num < 0:
    raise ValueError(f"Number of samples, {num}, must be non-negative.")
  util.check_arraylike("linspace", start, stop)

  if dtype is None:
    dtype = dtypes.to_inexact_dtype(result_type(start, stop))
  dtype = _jnp_dtype(dtype)
  computation_dtype = dtypes.to_inexact_dtype(dtype)
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)

  bounds_shape = list(lax.broadcast_shapes(shape(start), shape(stop)))
  broadcast_start = broadcast_to(start, bounds_shape)
  broadcast_stop = broadcast_to(stop, bounds_shape)
  axis = len(bounds_shape) + axis + 1 if axis < 0 else axis
  bounds_shape.insert(axis, 1)
  div = (num - 1) if endpoint else num
  if num > 1:
    delta: Array = lax.convert_element_type(stop - start, computation_dtype) / array(div, dtype=computation_dtype)
    iota_shape = [1,] * len(bounds_shape)
    iota_shape[axis] = div
    # This approach recovers the endpoints with float32 arithmetic,
    # but can lead to rounding errors for integer outputs.
    real_dtype = finfo(computation_dtype).dtype
    step = reshape(lax.iota(real_dtype, div), iota_shape) / array(div, real_dtype)
    step = step.astype(computation_dtype)
    out = (reshape(broadcast_start, bounds_shape) * (1 - step) +
      reshape(broadcast_stop, bounds_shape) * step)

    if endpoint:
      out = lax.concatenate([out, lax.expand_dims(broadcast_stop, (axis,))],
                            _canonicalize_axis(axis, out.ndim))

  elif num == 1:
    delta = asarray(nan if endpoint else stop - start, dtype=computation_dtype)
    out = reshape(broadcast_start, bounds_shape)
  else:  # num == 0 degenerate case, match numpy behavior
    empty_shape = list(lax.broadcast_shapes(shape(start), shape(stop)))
    empty_shape.insert(axis, 0)
    delta = asarray(nan, dtype=computation_dtype)
    out = reshape(array([], dtype=dtype), empty_shape)

  if issubdtype(dtype, integer) and not issubdtype(out.dtype, integer):
    out = lax.floor(out)

  sharding = canonicalize_device_to_sharding(device)
  result = lax_internal._convert_element_type(out, dtype, sharding=sharding)
  return (result, delta) if retstep else result


@util.implements(np.logspace)
def logspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
             endpoint: bool = True, base: ArrayLike = 10.0,
             dtype: DTypeLike | None = None, axis: int = 0) -> Array:
  num = core.concrete_or_error(operator.index, num, "'num' argument of jnp.logspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.logspace")
  return _logspace(start, stop, num, endpoint, base, dtype, axis)

@partial(jit, static_argnames=('num', 'endpoint', 'dtype', 'axis'))
def _logspace(start: ArrayLike, stop: ArrayLike, num: int = 50,
              endpoint: bool = True, base: ArrayLike = 10.0,
              dtype: DTypeLike | None = None, axis: int = 0) -> Array:
  """Implementation of logspace differentiable in start and stop args."""
  dtypes.check_user_dtype_supported(dtype, "logspace")
  if dtype is None:
    dtype = dtypes.to_inexact_dtype(result_type(start, stop))
  dtype = _jnp_dtype(dtype)
  computation_dtype = dtypes.to_inexact_dtype(dtype)
  util.check_arraylike("logspace", start, stop)
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)
  lin = linspace(start, stop, num,
                 endpoint=endpoint, retstep=False, dtype=None, axis=axis)
  return lax.convert_element_type(ufuncs.power(base, lin), dtype)


@util.implements(np.geomspace)
def geomspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True,
              dtype: DTypeLike | None = None, axis: int = 0) -> Array:
  num = core.concrete_or_error(operator.index, num, "'num' argument of jnp.geomspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.geomspace")
  return _geomspace(start, stop, num, endpoint, dtype, axis)

@partial(jit, static_argnames=('num', 'endpoint', 'dtype', 'axis'))
def _geomspace(start: ArrayLike, stop: ArrayLike, num: int = 50, endpoint: bool = True,
               dtype: DTypeLike | None = None, axis: int = 0) -> Array:
  """Implementation of geomspace differentiable in start and stop args."""
  dtypes.check_user_dtype_supported(dtype, "geomspace")
  if dtype is None:
    dtype = dtypes.to_inexact_dtype(result_type(start, stop))
  dtype = _jnp_dtype(dtype)
  computation_dtype = dtypes.to_inexact_dtype(dtype)
  util.check_arraylike("geomspace", start, stop)
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)

  sign = ufuncs.sign(start)
  res = sign * logspace(ufuncs.log10(start / sign), ufuncs.log10(stop / sign),
                        num, endpoint=endpoint, base=10.0,
                        dtype=computation_dtype, axis=0)
  if axis != 0:
    res = moveaxis(res, 0, axis)
  return lax.convert_element_type(res, dtype)


@util.implements(np.meshgrid, lax_description=_ARRAY_VIEW_DOC)
def meshgrid(*xi: ArrayLike, copy: bool = True, sparse: bool = False,
             indexing: str = 'xy') -> list[Array]:
  util.check_arraylike("meshgrid", *xi)
  args = [asarray(x) for x in xi]
  if not copy:
    raise ValueError("jax.numpy.meshgrid only supports copy=True")
  if indexing not in ["xy", "ij"]:
    raise ValueError(f"Valid values for indexing are 'xy' and 'ij', got {indexing}")
  if any(a.ndim != 1 for a in args):
    raise ValueError("Arguments to jax.numpy.meshgrid must be 1D, got shapes "
                     f"{[a.shape for a in args]}")
  if indexing == "xy" and len(args) >= 2:
    args[0], args[1] = args[1], args[0]
  shape = [1 if sparse else a.shape[0] for a in args]
  _a_shape = lambda i, a: [*shape[:i], a.shape[0], *shape[i + 1:]] if sparse else shape
  output = [lax.broadcast_in_dim(a, _a_shape(i, a), (i,)) for i, a, in enumerate(args)]
  if indexing == "xy" and len(args) >= 2:
    output[0], output[1] = output[1], output[0]
  return output


@custom_jvp
@util.implements(np.i0)
@jit
def i0(x: ArrayLike) -> Array:
  x_arr, = util.promote_args_inexact("i0", x)
  if not issubdtype(x_arr.dtype, np.floating):
    raise ValueError(f"Unsupported input type to jax.numpy.i0: {_dtype(x)}")
  x_arr = lax.abs(x_arr)
  return lax.mul(lax.exp(x_arr), lax.bessel_i0e(x_arr))

@i0.defjvp
def _i0_jvp(primals, tangents):
  primal_out, tangent_out = jax.jvp(i0.fun, primals, tangents)
  return primal_out, where(primals[0] == 0, 0.0, tangent_out)

def ix_(*args: ArrayLike) -> tuple[Array, ...]:
  """Return a multi-dimensional grid (open mesh) from N one-dimensional sequences.

  JAX implementation of :func:`numpy.ix_`.

  Args:
    *args: N one-dimensional arrays

  Returns:
    Tuple of Jax arrays forming an open mesh, each with N dimensions.

  See Also:
    - :obj:`jax.numpy.ogrid`
    - :obj:`jax.numpy.mgrid`
    - :func:`jax.numpy.meshgrid`

  Examples:
    >>> rows = jnp.array([0, 2])
    >>> cols = jnp.array([1, 3])
    >>> open_mesh = jnp.ix_(rows, cols)
    >>> open_mesh
    (Array([[0],
          [2]], dtype=int32), Array([[1, 3]], dtype=int32))
    >>> [grid.shape for grid in open_mesh]
    [(2, 1), (1, 2)]
    >>> x = jnp.array([[10, 20, 30, 40],
    ...                [50, 60, 70, 80],
    ...                [90, 100, 110, 120],
    ...                [130, 140, 150, 160]])
    >>> x[open_mesh]
    Array([[ 20,  40],
           [100, 120]], dtype=int32)
  """
  util.check_arraylike("ix", *args)
  n = len(args)
  output = []
  for i, a in enumerate(args):
    a = asarray(a)
    if len(a.shape) != 1:
      msg = "Arguments to jax.numpy.ix_ must be 1-dimensional, got shape {}"
      raise ValueError(msg.format(a.shape))
    if _dtype(a) == bool_:
      raise NotImplementedError(
        "Boolean arguments to jax.numpy.ix_ are not implemented")
    shape = [1] * n
    shape[i] = a.shape[0]
    if a.size == 0:
      # Numpy uses an integer index type for empty arrays.
      output.append(lax.full(shape, np.zeros((), np.intp)))
    else:
      output.append(lax.broadcast_in_dim(a, shape, (i,)))
  return tuple(output)


@overload
def indices(dimensions: Sequence[int], dtype: DTypeLike | None = None,
            sparse: Literal[False] = False) -> Array: ...
@overload
def indices(dimensions: Sequence[int], dtype: DTypeLike | None = None,
            *, sparse: Literal[True]) -> tuple[Array, ...]: ...
@overload
def indices(dimensions: Sequence[int], dtype: DTypeLike | None = None,
            sparse: bool = False) -> Array | tuple[Array, ...]: ...
@util.implements(np.indices)
def indices(dimensions: Sequence[int], dtype: DTypeLike | None = None,
            sparse: bool = False) -> Array | tuple[Array, ...]:
  dtypes.check_user_dtype_supported(dtype, "indices")
  dtype = dtype or dtypes.canonicalize_dtype(int_)
  dimensions = tuple(
      core.concrete_or_error(operator.index, d, "dimensions argument of jnp.indices")
      for d in dimensions)
  N = len(dimensions)
  output = []
  s = dimensions
  for i, dim in enumerate(dimensions):
    idx = lax.iota(dtype, dim)
    if sparse:
      s = (1,)*i + (dim,) + (1,)*(N - i - 1)
    output.append(lax.broadcast_in_dim(idx, s, (i,)))
  if sparse:
    return tuple(output)
  return stack(output, 0) if output else array([], dtype=dtype)


def repeat(a: ArrayLike, repeats: ArrayLike, axis: int | None = None, *,
           total_repeat_length: int | None = None) -> Array:
  """Construct an array from repeated elements.

  JAX implementation of :func:`numpy.repeat`.

  Args:
    a: N-dimensional array
    repeats: 1D integer array specifying the number of repeats. Must match the
      length of the repeated axis.
    axis: integer specifying the axis of ``a`` along which to construct the
      repeated array. If None (default) then ``a`` is first flattened.
    total_repeat_length: this must be specified statically for ``jnp.repeat``
      to be compatible with :func:`~jax.jit` and other JAX transformations.
      If ``sum(repeats)`` is larger than the specified ``total_repeat_length``,
      the remaining values will be discarded. If ``sum(repeats)`` is smaller
      than ``total_repeat_length``, the final value will be repeated.

  Returns:
    an array constructed from repeated values of ``a``.

  See Also:
    - :func:`jax.numpy.tile`: repeat a full array rather than individual values.

  Examples:
    Repeat each value twice along the last axis:

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.repeat(a, 2, axis=-1)
    Array([[1, 1, 2, 2],
           [3, 3, 4, 4]], dtype=int32)

    If ``axis`` is not specified, the input array will be flattened:

    >>> jnp.repeat(a, 2)
    Array([1, 1, 2, 2, 3, 3, 4, 4], dtype=int32)

    Pass an array to ``repeats`` to repeat each value a different number of times:

    >>> repeats = jnp.array([2, 3])
    >>> jnp.repeat(a, repeats, axis=1)
    Array([[1, 1, 2, 2, 2],
           [3, 3, 4, 4, 4]], dtype=int32)

    In order to use ``repeat`` within ``jit`` and other JAX transformations, the
    size of the output must be specified statically using ``total_repeat_length``:

    >>> jit_repeat = jax.jit(jnp.repeat, static_argnames=['axis', 'total_repeat_length'])
    >>> jit_repeat(a, repeats, axis=1, total_repeat_length=5)
    Array([[1, 1, 2, 2, 2],
           [3, 3, 4, 4, 4]], dtype=int32)

    If `total_repeat_length` is smaller than ``sum(repeats)``, the result will be truncated:

    >>> jit_repeat(a, repeats, axis=1, total_repeat_length=4)
    Array([[1, 1, 2, 2],
           [3, 3, 4, 4]], dtype=int32)

    If it is larger, then the additional entries will be filled with the final value:

    >>> jit_repeat(a, repeats, axis=1, total_repeat_length=7)
    Array([[1, 1, 2, 2, 2, 2, 2],
           [3, 3, 4, 4, 4, 4, 4]], dtype=int32)
  """
  util.check_arraylike("repeat", a)
  core.is_dim(repeats) or util.check_arraylike("repeat", repeats)

  if axis is None:
    a = ravel(a)
    axis = 0
  else:
    a = asarray(a)

  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.repeat()")
  assert isinstance(axis, int)  # to appease mypy

  if core.is_symbolic_dim(repeats):
    if total_repeat_length is not None:
      raise ValueError("jnp.repeat with a non-constant `repeats` is supported only "
                       f"when `total_repeat_length` is None. ({repeats=} {total_repeat_length=})")

  # If total_repeat_length is not given, use a default.
  if total_repeat_length is None:
    repeats = core.concrete_or_error(None, repeats,
      "When jit-compiling jnp.repeat, the total number of repeats must be static. "
      "To fix this, either specify a static value for `repeats`, or pass a static "
      "value to `total_repeat_length`.")

    # Fast path for when repeats is a scalar.
    if np.ndim(repeats) == 0 and ndim(a) != 0:
      input_shape = shape(a)
      axis = _canonicalize_axis(axis, len(input_shape))
      aux_axis = axis + 1
      aux_shape: list[DimSize] = list(input_shape)
      aux_shape.insert(aux_axis, operator.index(repeats) if core.is_constant_dim(repeats) else repeats)  # type: ignore
      a = lax.broadcast_in_dim(
        a, aux_shape, [i for i in range(len(aux_shape)) if i != aux_axis])
      result_shape: list[DimSize] = list(input_shape)
      result_shape[axis] *= repeats
      return reshape(a, result_shape)

    repeats = np.ravel(repeats)
    if ndim(a) != 0:
      repeats = np.broadcast_to(repeats, [shape(a)[axis]])
    total_repeat_length = np.sum(repeats)
  else:
    repeats = ravel(repeats)
    if ndim(a) != 0:
      repeats = broadcast_to(repeats, [shape(a)[axis]])

  # Special case when a is a scalar.
  if ndim(a) == 0:
    if shape(repeats) == (1,):
      return full([total_repeat_length], a)
    else:
      raise ValueError('`repeat` with a scalar parameter `a` is only '
      'implemented for scalar values of the parameter `repeats`.')

  # Special case if total_repeat_length is zero.
  if total_repeat_length == 0:
    result_shape = list(shape(a))
    result_shape[axis] = 0
    return reshape(array([], dtype=_dtype(a)), result_shape)

  # If repeats is on a zero sized axis, then return the array.
  if shape(a)[axis] == 0:
    return asarray(a)

  # This implementation of repeat avoid having to instantiate a large.
  # intermediate tensor.

  # Modify repeats from e.g. [1,2,0,5] -> [0,1,2,0] for exclusive repeat.
  exclusive_repeats = roll(repeats, shift=1).at[0].set(0)
  # Cumsum to get indices of new number in repeated tensor, e.g. [0, 1, 3, 3]
  scatter_indices = reductions.cumsum(exclusive_repeats)
  # Scatter these onto a zero buffer, e.g. [1,1,0,2,0,0,0,0]
  block_split_indicators = zeros([total_repeat_length], dtype=int32)
  block_split_indicators = block_split_indicators.at[scatter_indices].add(1)
  # Cumsum again to get scatter indices for repeat, e.g. [0,1,1,3,3,3,3,3]
  gather_indices = reductions.cumsum(block_split_indicators) - 1
  return take(a, gather_indices, axis=axis)


@util.implements(getattr(np, "trapezoid", getattr(np, "trapz", None)))
@partial(jit, static_argnames=('axis',))
def trapezoid(y: ArrayLike, x: ArrayLike | None = None, dx: ArrayLike = 1.0,
              axis: int = -1) -> Array:
  # TODO(phawkins): remove this annotation after fixing jnp types.
  dx_array: Array
  if x is None:
    util.check_arraylike('trapezoid', y)
    y_arr, = util.promote_dtypes_inexact(y)
    dx_array = asarray(dx)
  else:
    util.check_arraylike('trapezoid', y, x)
    y_arr, x_arr = util.promote_dtypes_inexact(y, x)
    if x_arr.ndim == 1:
      dx_array = diff(x_arr)
    else:
      dx_array = moveaxis(diff(x_arr, axis=axis), axis, -1)
  y_arr = moveaxis(y_arr, axis, -1)
  return 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).sum(-1)


def tri(N: int, M: int | None = None, k: int = 0, dtype: DTypeLike | None = None) -> Array:
  r"""Return an array with ones on and below the diagonal and zeros elsewhere.

  JAX implementation of :func:`numpy.tri`

  Args:
    N: int. Dimension of the rows of the returned array.
    M: optional, int. Dimension of the columns of the returned array. If not
      specified, then ``M = N``.
    k: optional, int, default=0. Specifies the sub-diagonal on and below which
      the array is filled with ones. ``k=0`` refers to main diagonal, ``k<0``
      refers to sub-diagonal below the main diagonal and ``k>0`` refers to
      sub-diagonal above the main diagonal.
    dtype: optional, data type of the returned array. The default type is float.

  Returns:
    An array of shape ``(N, M)`` containing the lower triangle with elements
    below the sub-diagonal specified by ``k`` are set to one and zero elsewhere.

  See also:
    - :func:`jax.numpy.tril`: Returns a lower triangle of an array.
    - :func:`jax.numpy.triu`: Returns an upper triangle of an array.

  Examples:
    >>> jnp.tri(3)
    Array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]], dtype=float32)

    When ``M`` is not equal to ``N``:

    >>> jnp.tri(3, 4)
    Array([[1., 0., 0., 0.],
           [1., 1., 0., 0.],
           [1., 1., 1., 0.]], dtype=float32)

    when ``k>0``:

    >>> jnp.tri(3, k=1)
    Array([[1., 1., 0.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)

    When ``k<0``:

    >>> jnp.tri(3, 4, k=-1)
    Array([[0., 0., 0., 0.],
           [1., 0., 0., 0.],
           [1., 1., 0., 0.]], dtype=float32)
  """
  dtypes.check_user_dtype_supported(dtype, "tri")
  M = M if M is not None else N
  dtype = dtype or float32
  return lax_internal._tri(dtype, (N, M), k)


@partial(jit, static_argnames=('k',))
def tril(m: ArrayLike, k: int = 0) -> Array:
  r"""Return lower triangle of an array.

  JAX implementation of :func:`numpy.tril`

  Args:
    m: input array. Must have ``m.ndim >= 2``.
    k: k: optional, int, default=0. Specifies the sub-diagonal above which the
      elements of the array are set to zero. ``k=0`` refers to main diagonal,
      ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers
      to sub-diagonal above the main diagonal.

  Returns:
    An array with same shape as input containing the upper triangle of the given
    array with elements below the sub-diagonal specified by ``k`` are set to zero.

  See also:
    - :func:`jax.numpy.triu`: Returns an upper triangle of an array.
    - :func:`jax.numpy.tri`: Returns an array with ones on and below the diagonal
      and zeros elsewhere.

  Examples:
    >>> x = jnp.array([[1, 2, 3, 4],
    ...                [5, 6, 7, 8],
    ...                [9, 10, 11, 12]])
    >>> jnp.tril(x)
    Array([[ 1,  0,  0,  0],
           [ 5,  6,  0,  0],
           [ 9, 10, 11,  0]], dtype=int32)
    >>> jnp.tril(x, k=1)
    Array([[ 1,  2,  0,  0],
           [ 5,  6,  7,  0],
           [ 9, 10, 11, 12]], dtype=int32)
    >>> jnp.tril(x, k=-1)
    Array([[ 0,  0,  0,  0],
           [ 5,  0,  0,  0],
           [ 9, 10,  0,  0]], dtype=int32)

    When ``m.ndim > 2``, ``jnp.tril`` operates batch-wise on the trailing axes.

    >>> x1 = jnp.array([[[1, 2],
    ...                  [3, 4]],
    ...                 [[5, 6],
    ...                  [7, 8]]])
    >>> jnp.tril(x1)
    Array([[[1, 0],
            [3, 4]],
    <BLANKLINE>
           [[5, 0],
            [7, 8]]], dtype=int32)
  """
  util.check_arraylike("tril", m)
  m_shape = shape(m)
  if len(m_shape) < 2:
    raise ValueError("Argument to jax.numpy.tril must be at least 2D")
  N, M = m_shape[-2:]
  mask = tri(N, M, k=k, dtype=bool)
  return lax.select(lax.broadcast(mask, m_shape[:-2]), m, zeros_like(m))


@partial(jit, static_argnames=('k',))
def triu(m: ArrayLike, k: int = 0) -> Array:
  r"""Return upper triangle of an array.

  JAX implementation of :func:`numpy.triu`

  Args:
    m: input array. Must have ``m.ndim >= 2``.
    k: optional, int, default=0. Specifies the sub-diagonal below which the
      elements of the array are set to zero. ``k=0`` refers to main diagonal,
      ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers
      to sub-diagonal above the main diagonal.

  Returns:
    An array with same shape as input containing the lower triangle of the given
    array with elements above the sub-diagonal specified by ``k`` are set to zero.

  See also:
    - :func:`jax.numpy.tril`: Returns a lower triangle of an array.
    - :func:`jax.numpy.tri`: Returns an array with ones on and below the diagonal
      and zeros elsewhere.

  Examples:
    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9],
    ...                [10, 11, 12]])
    >>> jnp.triu(x)
    Array([[1, 2, 3],
           [0, 5, 6],
           [0, 0, 9],
           [0, 0, 0]], dtype=int32)
    >>> jnp.triu(x, k=1)
    Array([[0, 2, 3],
           [0, 0, 6],
           [0, 0, 0],
           [0, 0, 0]], dtype=int32)
    >>> jnp.triu(x, k=-1)
    Array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]], dtype=int32)

    When ``m.ndim > 2``, ``jnp.triu`` operates batch-wise on the trailing axes.

    >>> x1 = jnp.array([[[1, 2],
    ...                  [3, 4]],
    ...                 [[5, 6],
    ...                  [7, 8]]])
    >>> jnp.triu(x1)
    Array([[[1, 2],
            [0, 4]],
    <BLANKLINE>
           [[5, 6],
            [0, 8]]], dtype=int32)
  """
  util.check_arraylike("triu", m)
  m_shape = shape(m)
  if len(m_shape) < 2:
    raise ValueError("Argument to jax.numpy.triu must be at least 2D")
  N, M = m_shape[-2:]
  mask = tri(N, M, k=k - 1, dtype=bool)
  return lax.select(lax.broadcast(mask, m_shape[:-2]), zeros_like(m), m)


@util.implements(np.trace, skip_params=['out'])
@partial(jit, static_argnames=('axis1', 'axis2', 'dtype'))
def trace(a: ArrayLike, offset: int | ArrayLike = 0, axis1: int = 0, axis2: int = 1,
          dtype: DTypeLike | None = None, out: None = None) -> Array:
  util.check_arraylike("trace", a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.trace is not supported.")
  dtypes.check_user_dtype_supported(dtype, "trace")

  a_shape = shape(a)
  a = moveaxis(a, (axis1, axis2), (-2, -1))

  # Mask out the diagonal and reduce.
  a = where(eye(a_shape[axis1], a_shape[axis2], k=offset, dtype=bool),
            a, zeros_like(a))
  return reductions.sum(a, axis=(-2, -1), dtype=dtype)


def _wrap_indices_function(f):
  @util.implements(f, update_doc=False)
  def wrapper(*args, **kwargs):
    args = [core.concrete_or_error(
              None, arg, f"argument {i} of jnp.{f.__name__}()")
            for i, arg in enumerate(args)]
    kwargs = {key: core.concrete_or_error(
                None, val, f"argument '{key}' of jnp.{f.__name__}()")
              for key, val in kwargs.items()}
    return tuple(asarray(x) for x in f(*args, **kwargs))
  return wrapper

mask_indices = _wrap_indices_function(np.mask_indices)


def _triu_size(n, m, k):
  if k < 0:
    return n * m - _triu_size(m, n, (1 - k))
  elif k >= m:
    return 0
  else:
    mk = min(n, m - k)
    return mk * (mk + 1) // 2 + mk * (m - k - mk)


def triu_indices(n: int, k: int = 0, m: int | None = None) -> tuple[Array, Array]:
  """Return the indices of upper triangle of an array of size ``(n, m)``.

  JAX implementation of :func:`numpy.triu_indices`.

  Args:
    n: int. Number of rows of the array for which the indices are returned.
    k: optional, int, default=0. Specifies the sub-diagonal on and above which
      the indices of upper triangle are returned. ``k=0`` refers to main diagonal,
      ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers
      to sub-diagonal above the main diagonal.
    m: optional, int. Number of columns of the array for which the indices are
      returned. If not specified, then ``m = n``.

  Returns:
    A tuple of two arrays containing the indices of the upper triangle, one along
    each axis.

  See also:
    - :func:`jax.numpy.tril_indices`: Returns the indices of lower triangle of an
      array of size ``(n, m)``.
    - :func:`jax.numpy.triu_indices_from`: Returns the indices of upper triangle
      of a given array.
    - :func:`jax.numpy.tril_indices_from`: Returns the indices of lower triangle
      of a given array.

  Examples:
    If only ``n`` is provided in input, the indices of upper triangle of an array
    of size ``(n, n)`` array are returned.

    >>> jnp.triu_indices(3)
    (Array([0, 0, 0, 1, 1, 2], dtype=int32), Array([0, 1, 2, 1, 2, 2], dtype=int32))

    If both ``n`` and ``m`` are provided in input, the indices of upper triangle
    of an ``(n, m)`` array are returned.

    >>> jnp.triu_indices(3, m=2)
    (Array([0, 0, 1], dtype=int32), Array([0, 1, 1], dtype=int32))

    If ``k = 1``, the indices on and above the first sub-diagonal above the main
    diagonal are returned.

    >>> jnp.triu_indices(3, k=1)
    (Array([0, 0, 1], dtype=int32), Array([1, 2, 2], dtype=int32))

    If ``k = -1``, the indices on and above the first sub-diagonal below the main
    diagonal are returned.

    >>> jnp.triu_indices(3, k=-1)
    (Array([0, 0, 0, 1, 1, 1, 2, 2], dtype=int32), Array([0, 1, 2, 0, 1, 2, 1, 2], dtype=int32))
  """
  n = core.concrete_or_error(operator.index, n, "n argument of jnp.triu_indices")
  k = core.concrete_or_error(operator.index, k, "k argument of jnp.triu_indices")
  m = n if m is None else core.concrete_or_error(operator.index, m, "m argument of jnp.triu_indices")
  i, j = nonzero(triu(ones((n, m)), k=k), size=_triu_size(n, m, k))
  return i, j


def tril_indices(n: int, k: int = 0, m: int | None = None) -> tuple[Array, Array]:
  """Return the indices of lower triangle of an array of size ``(n, m)``.

  JAX implementation of :func:`numpy.tril_indices`.

  Args:
    n: int. Number of rows of the array for which the indices are returned.
    k: optional, int, default=0. Specifies the sub-diagonal on and below which
      the indices of lower triangle are returned. ``k=0`` refers to main diagonal,
      ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers
      to sub-diagonal above the main diagonal.
    m: optional, int. Number of columns of the array for which the indices are
      returned. If not specified, then ``m = n``.

  Returns:
    A tuple of two arrays containing the indices of the lower triangle, one along
    each axis.

  See also:
    - :func:`jax.numpy.triu_indices`: Returns the indices of upper triangle of an
      array of size ``(n, m)``.
    - :func:`jax.numpy.triu_indices_from`: Returns the indices of upper triangle
      of a given array.
    - :func:`jax.numpy.tril_indices_from`: Returns the indices of lower triangle
      of a given array.

  Examples:
    If only ``n`` is provided in input, the indices of lower triangle of an array
    of size ``(n, n)`` array are returned.

    >>> jnp.tril_indices(3)
    (Array([0, 1, 1, 2, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1, 2], dtype=int32))

    If both ``n`` and ``m`` are provided in input, the indices of lower triangle
    of an ``(n, m)`` array are returned.

    >>> jnp.tril_indices(3, m=2)
    (Array([0, 1, 1, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1], dtype=int32))

    If ``k = 1``, the indices on and below the first sub-diagonal above the main
    diagonal are returned.

    >>> jnp.tril_indices(3, k=1)
    (Array([0, 0, 1, 1, 1, 2, 2, 2], dtype=int32), Array([0, 1, 0, 1, 2, 0, 1, 2], dtype=int32))

    If ``k = -1``, the indices on and below the first sub-diagonal below the main
    diagonal are returned.

    >>> jnp.tril_indices(3, k=-1)
    (Array([1, 2, 2], dtype=int32), Array([0, 0, 1], dtype=int32))
  """
  n = core.concrete_or_error(operator.index, n, "n argument of jnp.triu_indices")
  k = core.concrete_or_error(operator.index, k, "k argument of jnp.triu_indices")
  m = n if m is None else core.concrete_or_error(operator.index, m, "m argument of jnp.triu_indices")
  i, j = nonzero(tril(ones((n, m)), k=k), size=_triu_size(m, n, -k))
  return i, j


def triu_indices_from(arr: ArrayLike, k: int = 0) -> tuple[Array, Array]:
  """Return the indices of upper triangle of a given array.

  JAX implementation of :func:`numpy.triu_indices_from`.

  Args:
    arr: input array. Must have ``arr.ndim == 2``.
    k: optional, int, default=0. Specifies the sub-diagonal on and above which
      the indices of upper triangle are returned. ``k=0`` refers to main diagonal,
      ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers
      to sub-diagonal above the main diagonal.

  Returns:
    A tuple of two arrays containing the indices of the upper triangle, one along
    each axis.

  See also:
    - :func:`jax.numpy.tril_indices_from`: Returns the indices of lower triangle
      of a given array.
    - :func:`jax.numpy.triu_indices`: Returns the indices of upper triangle of an
      array of size ``(n, m)``.
    - :func:`jax.numpy.triu`: Return an upper triangle of an array.

  Examples:
    >>> arr = jnp.array([[1, 2, 3],
    ...                  [4, 5, 6],
    ...                  [7, 8, 9]])
    >>> jnp.triu_indices_from(arr)
    (Array([0, 0, 0, 1, 1, 2], dtype=int32), Array([0, 1, 2, 1, 2, 2], dtype=int32))

    Elements indexed by ``jnp.triu_indices_from`` correspond to those in the
    output of ``jnp.triu``.

    >>> ind = jnp.triu_indices_from(arr)
    >>> arr[ind]
    Array([1, 2, 3, 5, 6, 9], dtype=int32)
    >>> jnp.triu(arr)
    Array([[1, 2, 3],
           [0, 5, 6],
           [0, 0, 9]], dtype=int32)

    When ``k > 0``:

    >>> jnp.triu_indices_from(arr, k=1)
    (Array([0, 0, 1], dtype=int32), Array([1, 2, 2], dtype=int32))

    When ``k < 0``:

    >>> jnp.triu_indices_from(arr, k=-1)
    (Array([0, 0, 0, 1, 1, 1, 2, 2], dtype=int32), Array([0, 1, 2, 0, 1, 2, 1, 2], dtype=int32))
  """
  arr_shape = shape(arr)
  if len(arr_shape) != 2:
    raise ValueError("Only 2-D inputs are accepted")
  return triu_indices(arr_shape[0], k=k, m=arr_shape[1])


def tril_indices_from(arr: ArrayLike, k: int = 0) -> tuple[Array, Array]:
  """Return the indices of lower triangle of a given array.

  JAX implementation of :func:`numpy.tril_indices_from`.

  Args:
    arr: input array. Must have ``arr.ndim == 2``.
    k: optional, int, default=0. Specifies the sub-diagonal on and below which
      the indices of upper triangle are returned. ``k=0`` refers to main diagonal,
      ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers
      to sub-diagonal above the main diagonal.

  Returns:
    A tuple of two arrays containing the indices of the lower triangle, one along
    each axis.

  See also:
    - :func:`jax.numpy.triu_indices_from`: Returns the indices of upper triangle
      of a given array.
    - :func:`jax.numpy.tril_indices`: Returns the indices of lower triangle of an
      array of size ``(n, m)``.
    - :func:`jax.numpy.tril`: Returns a lower triangle of an array

  Examples:
    >>> arr = jnp.array([[1, 2, 3],
    ...                  [4, 5, 6],
    ...                  [7, 8, 9]])
    >>> jnp.tril_indices_from(arr)
    (Array([0, 1, 1, 2, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1, 2], dtype=int32))

    Elements indexed by ``jnp.tril_indices_from`` correspond to those in the
    output of ``jnp.tril``.

    >>> ind = jnp.tril_indices_from(arr)
    >>> arr[ind]
    Array([1, 4, 5, 7, 8, 9], dtype=int32)
    >>> jnp.tril(arr)
    Array([[1, 0, 0],
           [4, 5, 0],
           [7, 8, 9]], dtype=int32)

    When ``k > 0``:

    >>> jnp.tril_indices_from(arr, k=1)
    (Array([0, 0, 1, 1, 1, 2, 2, 2], dtype=int32), Array([0, 1, 0, 1, 2, 0, 1, 2], dtype=int32))

    When ``k < 0``:

    >>> jnp.tril_indices_from(arr, k=-1)
    (Array([1, 2, 2], dtype=int32), Array([0, 0, 1], dtype=int32))
  """
  arr_shape = shape(arr)
  if len(arr_shape) != 2:
    raise ValueError("Only 2-D inputs are accepted")
  return tril_indices(arr_shape[0], k=k, m=arr_shape[1])


def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap: bool = False, *,
                  inplace: bool = True) -> Array:
  """Return a copy of the array with the diagonal overwritten.

  JAX implementation of :func:`numpy.fill_diagonal`.

  The semantics of :func:`numpy.fill_diagonal` are to modify arrays in-place, which
  is not possible for JAX's immutable arrays. The JAX version returns a modified
  copy of the input, and adds the ``inplace`` parameter which must be set to
  `False`` by the user as a reminder of this API difference.

  Args:
    a: input array. Must have ``a.ndim >= 2``. If ``a.ndim >= 3``, then all
      dimensions must be the same size.
    val: scalar or array with which to fill the diagonal. If an array, it will
      be flattened and repeated to fill the diagonal entries.
    inplace: must be set to False to indicate that the input is not modified
      in-place, but rather a modified copy is returned.

  Returns:
    A copy of ``a`` with the diagonal set to ``val``.

  Examples:
    >>> x = jnp.zeros((3, 3), dtype=int)
    >>> jnp.fill_diagonal(x, jnp.array([1, 2, 3]), inplace=False)
    Array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]], dtype=int32)

    Unlike :func:`numpy.fill_diagonal`, the input ``x`` is not modified.

    If the diagonal value has too many entries, it will be truncated

    >>> jnp.fill_diagonal(x, jnp.arange(100, 200), inplace=False)
    Array([[100,   0,   0],
           [  0, 101,   0],
           [  0,   0, 102]], dtype=int32)

    If the diagonal has too few entries, it will be repeated:

    >>> x = jnp.zeros((4, 4), dtype=int)
    >>> jnp.fill_diagonal(x, jnp.array([3, 4]), inplace=False)
    Array([[3, 0, 0, 0],
           [0, 4, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]], dtype=int32)

    For non-square arrays, the diagonal of the leading square slice is filled:

    >>> x = jnp.zeros((3, 5), dtype=int)
    >>> jnp.fill_diagonal(x, 1, inplace=False)
    Array([[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0]], dtype=int32)

    And for square N-dimensional arrays, the N-dimensional diagonal is filled:

    >>> y = jnp.zeros((2, 2, 2))
    >>> jnp.fill_diagonal(y, 1, inplace=False)
    Array([[[1., 0.],
            [0., 0.]],
    <BLANKLINE>
           [[0., 0.],
            [0., 1.]]], dtype=float32)
  """
  if inplace:
    raise NotImplementedError("JAX arrays are immutable, must use inplace=False")
  if wrap:
    raise NotImplementedError("wrap=True is not implemented, must use wrap=False")
  util.check_arraylike("fill_diagonal", a, val)
  a = asarray(a)
  val = asarray(val)
  if a.ndim < 2:
    raise ValueError("array must be at least 2-d")
  if a.ndim > 2 and not all(n == a.shape[0] for n in a.shape[1:]):
    raise ValueError("All dimensions of input must be of equal length")
  n = min(a.shape)
  idx = diag_indices(n, a.ndim)
  return a.at[idx].set(val if val.ndim == 0 else _tile_to_size(val.ravel(), n))


@util.implements(np.diag_indices)
def diag_indices(n: int, ndim: int = 2) -> tuple[Array, ...]:
  n = core.concrete_or_error(operator.index, n, "'n' argument of jnp.diag_indices()")
  ndim = core.concrete_or_error(operator.index, ndim, "'ndim' argument of jnp.diag_indices()")
  if n < 0:
    raise ValueError("n argument to diag_indices must be nonnegative, got {}"
                     .format(n))
  if ndim < 0:
    raise ValueError("ndim argument to diag_indices must be nonnegative, got {}"
                     .format(ndim))
  return (lax.iota(int_, n),) * ndim

@util.implements(np.diag_indices_from)
def diag_indices_from(arr: ArrayLike) -> tuple[Array, ...]:
  util.check_arraylike("diag_indices_from", arr)
  nd = ndim(arr)
  if not ndim(arr) >= 2:
    raise ValueError("input array must be at least 2-d")

  s = shape(arr)
  if len(set(shape(arr))) != 1:
    raise ValueError("All dimensions of input must be of equal length")

  return diag_indices(s[0], ndim=nd)

@util.implements(np.diagonal, lax_description=_ARRAY_VIEW_DOC)
@partial(jit, static_argnames=('offset', 'axis1', 'axis2'))
def diagonal(a: ArrayLike, offset: int = 0, axis1: int = 0,
             axis2: int = 1) -> Array:
  util.check_arraylike("diagonal", a)
  a_shape = shape(a)
  if ndim(a) < 2:
    raise ValueError("diagonal requires an array of at least two dimensions.")
  offset = core.concrete_or_error(operator.index, offset, "'offset' argument of jnp.diagonal()")

  a = moveaxis(a, (axis1, axis2), (-2, -1))

  diag_size = max(0, min(a_shape[axis1] + min(offset, 0),
                         a_shape[axis2] - max(offset, 0)))
  i = arange(diag_size)
  j = arange(abs(offset), abs(offset) + diag_size)
  return a[..., i, j] if offset >= 0 else a[..., j, i]


@util.implements(np.diag, lax_description=_ARRAY_VIEW_DOC)
def diag(v: ArrayLike, k: int = 0) -> Array:
  return _diag(v, operator.index(k))

@partial(jit, static_argnames=('k',))
def _diag(v, k):
  util.check_arraylike("diag", v)
  v_shape = shape(v)
  if len(v_shape) == 1:
    zero = lambda x: lax.full_like(x, shape=(), fill_value=0)
    n = v_shape[0] + abs(k)
    v = lax.pad(v, zero(v), ((max(0, k), max(0, -k), 0),))
    return where(eye(n, k=k, dtype=bool), v, zeros_like(v))
  elif len(v_shape) == 2:
    return diagonal(v, offset=k)
  else:
    raise ValueError("diag input must be 1d or 2d")

_SCALAR_VALUE_DOC = """\
This differs from np.diagflat for some scalar values of v,
jax always returns a two-dimensional array, whereas numpy may
return a scalar depending on the type of v.
"""

@util.implements(np.diagflat, lax_description=_SCALAR_VALUE_DOC)
def diagflat(v: ArrayLike, k: int = 0) -> Array:
  util.check_arraylike("diagflat", v)
  v_ravel = ravel(v)
  v_length = len(v_ravel)
  adj_length = v_length + abs(k)
  res = zeros(adj_length*adj_length, dtype=v_ravel.dtype)
  i = arange(0, adj_length-abs(k))
  if (k >= 0):
    fi = i+k+i*adj_length
  else:
    fi = i+(i-k)*adj_length
  res = res.at[fi].set(v_ravel)
  res = res.reshape(adj_length, adj_length)
  return res


@util.implements(np.trim_zeros)
def trim_zeros(filt, trim='fb'):
  filt = core.concrete_or_error(asarray, filt,
    "Error arose in the `filt` argument of trim_zeros()")
  nz = (filt == 0)
  if reductions.all(nz):
    return empty(0, _dtype(filt))
  start = argmin(nz) if 'f' in trim.lower() else 0
  end = argmin(nz[::-1]) if 'b' in trim.lower() else 0
  return filt[start:len(filt) - end]


def trim_zeros_tol(filt, tol, trim='fb'):
  filt = core.concrete_or_error(asarray, filt,
    "Error arose in the `filt` argument of trim_zeros_tol()")
  nz = (ufuncs.abs(filt) < tol)
  if reductions.all(nz):
    return empty(0, _dtype(filt))
  start = argmin(nz) if 'f' in trim.lower() else 0
  end = argmin(nz[::-1]) if 'b' in trim.lower() else 0
  return filt[start:len(filt) - end]

@partial(jit, static_argnames=('axis',))
def append(
    arr: ArrayLike, values: ArrayLike, axis: int | None = None
) -> Array:
  """Return a new array with values appended to the end of the original array.

  JAX implementation of :func:`numpy.append`.

  Args:
    arr: original array.
    values: values to be appended to the array. The ``values`` must have
      the same number of dimensions as ``arr``, and all dimensions must
      match except in the specified axis.
    axis: axis along which to append values. If None (default), both ``arr``
      and ``values`` will be flattened before appending.

  Returns:
    A new array with values appended to ``arr``.

  See also:
    - :func:`jax.numpy.insert`
    - :func:`jax.numpy.delete`

  Examples:
    >>> a = jnp.array([1, 2, 3])
    >>> b = jnp.array([4, 5, 6])
    >>> jnp.append(a, b)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

    Appending along a specific axis:

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> b = jnp.array([[5, 6]])
    >>> jnp.append(a, b, axis=0)
    Array([[1, 2],
           [3, 4],
           [5, 6]], dtype=int32)

    Appending along a trailing axis:

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> b = jnp.array([[7], [8]])
    >>> jnp.append(a, b, axis=1)
    Array([[1, 2, 3, 7],
           [4, 5, 6, 8]], dtype=int32)
  """
  if axis is None:
    return concatenate([ravel(arr), ravel(values)], 0)
  else:
    return concatenate([arr, values], axis=axis)


def delete(
    arr: ArrayLike,
    obj: ArrayLike | slice,
    axis: int | None = None,
    *,
    assume_unique_indices: bool = False,
) -> Array:
  """Delete entry or entries from an array.

  JAX implementation of :func:`numpy.delete`.

  Args:
    arr: array from which entries will be deleted.
    obj: index, indices, or slice to be deleted.
    axis: axis along which entries will be deleted.
    assume_unique_indices: In case of array-like integer (not boolean) indices,
      assume the indices are unique, and perform the deletion in a way that is
      compatible with JIT and other JAX transformations.

  Returns:
    Copy of ``arr`` with specified indices deleted.

  Note:
    ``delete()`` usually requires the index specification to be static. If the
    index is an integer array that is guaranteed to contain unique entries, you
    may specify ``assume_unique_indices=True`` to perform the operation in a
    manner that does not require static indices.

  Examples:
    Delete entries from a 1D array:

    >>> a = jnp.array([4, 5, 6, 7, 8, 9])
    >>> jnp.delete(a, 2)
    Array([4, 5, 7, 8, 9], dtype=int32)
    >>> jnp.delete(a, slice(1, 4))  # delete a[1:4]
    Array([4, 8, 9], dtype=int32)
    >>> jnp.delete(a, slice(None, None, 2))  # delete a[::2]
    Array([5, 7, 9], dtype=int32)

    Delete entries from a 2D array along a specified axis:

    >>> a2 = jnp.array([[4, 5, 6],
    ...                 [7, 8, 9]])
    >>> jnp.delete(a2, 1, axis=1)
    Array([[4, 6],
           [7, 9]], dtype=int32)

    Delete multiple entries via a sequence of indices:

    >>> indices = jnp.array([0, 1, 3])
    >>> jnp.delete(a, indices)
    Array([6, 8, 9], dtype=int32)

    This will fail under :func:`~jax.jit` and other transformations, because
    the output shape cannot be known with the possibility of duplicate indices:

    >>> jax.jit(jnp.delete)(a, indices)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[3].

    If you can ensure that the indices are unique, pass ``assume_unique_indices``
    to allow this to be executed under JIT:

    >>> jit_delete = jax.jit(jnp.delete, static_argnames=['assume_unique_indices'])
    >>> jit_delete(a, indices, assume_unique_indices=True)
    Array([6, 8, 9], dtype=int32)
  """
  util.check_arraylike("delete", arr)
  if axis is None:
    arr = ravel(arr)
    axis = 0
  a = asarray(arr)
  axis = _canonicalize_axis(axis, a.ndim)

  # Case 1: obj is a static integer.
  try:
    obj = operator.index(obj)  # type: ignore[arg-type]
    obj = _canonicalize_axis(obj, a.shape[axis])
  except TypeError:
    pass
  else:
    idx = tuple(slice(None) for i in range(axis))
    return concatenate([a[idx + (slice(0, obj),)], a[idx + (slice(obj + 1, None),)]], axis=axis)

  # Case 2: obj is a static slice.
  if isinstance(obj, slice):
    obj = arange(a.shape[axis])[obj]
    assume_unique_indices = True

  # Case 3: obj is an array
  # NB: pass both arrays to check for appropriate error message.
  util.check_arraylike("delete", a, obj)

  # Case 3a: unique integer indices; delete in a JIT-compatible way
  if issubdtype(_dtype(obj), integer) and assume_unique_indices:
    obj = asarray(obj).ravel()
    obj = clip(where(obj < 0, obj + a.shape[axis], obj), 0, a.shape[axis])
    obj = sort(obj)
    obj -= arange(len(obj))  # type: ignore[arg-type,operator]
    i = arange(a.shape[axis] - obj.size)
    i += (i[None, :] >= obj[:, None]).sum(0)
    return a[(slice(None),) * axis + (i,)]

  # Case 3b: non-unique indices: must be static.
  obj_array = core.concrete_or_error(np.asarray, obj, "'obj' array argument of jnp.delete()")
  if issubdtype(obj_array.dtype, integer):
    # TODO(jakevdp): in theory this could be done dynamically if obj has no duplicates,
    # but this would require the complement of lax.gather.
    mask = np.ones(a.shape[axis], dtype=bool)
    mask[obj_array] = False
  elif obj_array.dtype == bool:
    if obj_array.shape != (a.shape[axis],):
      raise ValueError("np.delete(arr, obj): for boolean indices, obj must be one-dimensional "
                       "with length matching specified axis.")
    mask = ~obj_array
  else:
    raise ValueError(f"np.delete(arr, obj): got obj.dtype={obj_array.dtype}; must be integer or bool.")
  return a[tuple(slice(None) for i in range(axis)) + (mask,)]


@util.implements(np.insert)
def insert(arr: ArrayLike, obj: ArrayLike | slice, values: ArrayLike,
           axis: int | None = None) -> Array:
  util.check_arraylike("insert", arr, 0 if isinstance(obj, slice) else obj, values)
  a = asarray(arr)
  values_arr = asarray(values)

  if axis is None:
    a = ravel(a)
    axis = 0
  axis = core.concrete_or_error(None, axis, "axis argument of jnp.insert()")
  axis = _canonicalize_axis(axis, a.ndim)
  if isinstance(obj, slice):
    indices = arange(*obj.indices(a.shape[axis]))
  else:
    indices = asarray(obj)

  if indices.ndim > 1:
    raise ValueError("jnp.insert(): obj must be a slice, a one-dimensional "
                     f"array, or a scalar; got {obj}")
  if not np.issubdtype(indices.dtype, np.integer):
    if indices.size == 0 and not isinstance(obj, Array):
      indices = indices.astype(int)
    else:
      # Note: np.insert allows boolean inputs but the behavior is deprecated.
      raise ValueError("jnp.insert(): index array must be "
                       f"integer typed; got {obj}")
  values_arr = array(values_arr, ndmin=a.ndim, dtype=a.dtype, copy=False)

  if indices.size == 1:
    index = ravel(indices)[0]
    if indices.ndim == 0:
      values_arr = moveaxis(values_arr, 0, axis)
    indices = full(values_arr.shape[axis], index)
  n_input = a.shape[axis]
  n_insert = broadcast_shapes(indices.shape, (values_arr.shape[axis],))[0]
  out_shape = list(a.shape)
  out_shape[axis] += n_insert
  out = zeros_like(a, shape=tuple(out_shape))

  indices = where(indices < 0, indices + n_input, indices)
  indices = clip(indices, 0, n_input)

  values_ind = indices.at[argsort(indices)].add(arange(n_insert, dtype=indices.dtype))
  arr_mask = ones(n_input + n_insert, dtype=bool).at[values_ind].set(False)
  arr_ind = where(arr_mask, size=n_input)[0]

  out = out.at[(slice(None),) * axis + (values_ind,)].set(values_arr)
  out = out.at[(slice(None),) * axis + (arr_ind,)].set(a)

  return out


@util.implements(np.apply_along_axis)
def apply_along_axis(
    func1d: Callable, axis: int, arr: ArrayLike, *args, **kwargs
) -> Array:
  util.check_arraylike("apply_along_axis", arr)
  num_dims = ndim(arr)
  axis = _canonicalize_axis(axis, num_dims)
  func = lambda arr: func1d(arr, *args, **kwargs)
  for i in range(1, num_dims - axis):
    func = jax.vmap(func, in_axes=i, out_axes=-1)
  for i in range(axis):
    func = jax.vmap(func, in_axes=0, out_axes=0)
  return func(arr)


@util.implements(np.apply_over_axes)
def apply_over_axes(func: Callable[[ArrayLike, int], Array], a: ArrayLike,
                    axes: Sequence[int]) -> Array:
  util.check_arraylike("apply_over_axes", a)
  a_arr = asarray(a)
  for axis in axes:
    b = func(a_arr, axis)
    if b.ndim == a_arr.ndim:
      a_arr = b
    elif b.ndim == a_arr.ndim - 1:
      a_arr = expand_dims(b, axis)
    else:
      raise ValueError("function is not returning an array of the correct shape")
  return a_arr


### Tensor contraction operations

@partial(jit, static_argnames=('precision', 'preferred_element_type'), inline=True)
def dot(a: ArrayLike, b: ArrayLike, *,
        precision: PrecisionLike = None,
        preferred_element_type: DTypeLike | None = None) -> Array:
  """Compute the dot product of two arrays.

  JAX implementation of :func:`numpy.dot`.

  This differs from :func:`jax.numpy.matmul` in two respects:

  - if either ``a`` or ``b`` is a scalar, the result of ``dot`` is equivalent to
    :func:`jax.numpy.multiply`, while the result of ``matmul`` is an error.
  - if ``a`` and ``b`` have more than 2 dimensions, the batch indices are
    stacked rather than broadcast.

  Args:
    a: first input array, of shape ``(..., N)``.
    b: second input array. Must have shape ``(N,)`` or ``(..., N, M)``.
      In the multi-dimensional case, leading dimensions must be broadcast-compatible
      with the leading dimensions of ``a``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the dot product of the inputs, with batch dimensions of
    ``a`` and ``b`` stacked rather than broadcast.

  See also:
    - :func:`jax.numpy.matmul`: broadcasted batched matmul.
    - :func:`jax.lax.dot_general`: general batched matrix multiplication.

  Examples:
    For scalar inputs, ``dot`` computes the element-wise product:

    >>> x = jnp.array([1, 2, 3])
    >>> jnp.dot(x, 2)
    Array([2, 4, 6], dtype=int32)

    For vector or matrix inputs, ``dot`` computes the vector or matrix product:

    >>> M = jnp.array([[2, 3, 4],
    ...                [5, 6, 7],
    ...                [8, 9, 0]])
    >>> jnp.dot(M, x)
    Array([20, 38, 26], dtype=int32)
    >>> jnp.dot(M, M)
    Array([[ 51,  60,  29],
           [ 96, 114,  62],
           [ 61,  78,  95]], dtype=int32)

    For higher-dimensional matrix products, batch dimensions are stacked, whereas
    in :func:`~jax.numpy.matmul` they are broadcast. For example:

    >>> a = jnp.zeros((3, 2, 4))
    >>> b = jnp.zeros((3, 4, 1))
    >>> jnp.dot(a, b).shape
    (3, 2, 3, 1)
    >>> jnp.matmul(a, b).shape
    (3, 2, 1)
  """
  util.check_arraylike("dot", a, b)
  dtypes.check_user_dtype_supported(preferred_element_type, "dot")
  a, b = asarray(a), asarray(b)
  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(a, b, return_weak_type_flag=True)
  else:
    output_weak_type = False

  batch_dims = ((), ())
  a_ndim, b_ndim = ndim(a), ndim(b)
  if a_ndim == 0 or b_ndim == 0:
    # TODO(jakevdp): lower this case to dot_general as well?
    # Currently, doing so causes issues in remat tests due to #16805
    if preferred_element_type is not None:
      a = a.astype(preferred_element_type)
      b = b.astype(preferred_element_type)
    result = lax.mul(a, b)
  else:
    if b_ndim == 1:
      contract_dims = ((a_ndim - 1,), (0,))
    else:
      contract_dims = ((a_ndim - 1,), (b_ndim - 2,))
    result = lax.dot_general(a, b, dimension_numbers=(contract_dims, batch_dims),
                             precision=precision, preferred_element_type=preferred_element_type)
  return lax_internal._convert_element_type(result, preferred_element_type, output_weak_type)


@partial(jit, static_argnames=('precision', 'preferred_element_type'), inline=True)
def matmul(a: ArrayLike, b: ArrayLike, *,
           precision: PrecisionLike = None,
           preferred_element_type: DTypeLike | None = None,
           ) -> Array:
  """Perform a matrix multiplication.

  JAX implementation of :func:`numpy.matmul`.

  Args:
    a: first input array, of shape ``(..., N)``.
    b: second input array. Must have shape ``(N,)`` or ``(..., N, M)``.
      In the multi-dimensional case, leading dimensions must be broadcast-compatible
      with the leading dimensions of ``a``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the matrix product of the inputs. Shape is ``a.shape[:-1]``
    if ``b.ndim == 1``, otherwise the shape is ``(..., M)``, where leading
    dimensions of ``a`` and ``b`` are broadcast together.

  See Also:
    - :func:`jax.numpy.linalg.vecdot`: batched vector product.
    - :func:`jax.numpy.linalg.tensordot`: batched tensor product.
    - :func:`jax.lax.dot_general`: general N-dimensional batched dot product.

  Examples:
    Vector dot products:

    >>> a = jnp.array([1, 2, 3])
    >>> b = jnp.array([4, 5, 6])
    >>> jnp.matmul(a, b)
    Array(32, dtype=int32)

    Matrix dot product:

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> b = jnp.array([[1, 2],
    ...                [3, 4],
    ...                [5, 6]])
    >>> jnp.matmul(a, b)
    Array([[22, 28],
           [49, 64]], dtype=int32)

    For convenience, in all cases you can do the same computation using
    the ``@`` operator:

    >>> a @ b
    Array([[22, 28],
           [49, 64]], dtype=int32)
  """
  util.check_arraylike("matmul", a, b)
  dtypes.check_user_dtype_supported(preferred_element_type, "matmul")
  a, b = asarray(a), asarray(b)
  for i, x in enumerate((a, b)):
    if ndim(x) < 1:
      msg = (f"matmul input operand {i} must have ndim at least 1, "
             f"but it has ndim {ndim(x)}")
      raise ValueError(msg)
  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(a, b, return_weak_type_flag=True)
  else:
    output_weak_type = False

  a_is_mat, b_is_mat = (ndim(a) > 1), (ndim(b) > 1)
  a_batch_dims: tuple[int | None, ...] = shape(a)[:-2] if a_is_mat else ()
  b_batch_dims: tuple[int | None, ...] = shape(b)[:-2] if b_is_mat else ()
  num_batch_dims = max(len(a_batch_dims), len(b_batch_dims))
  a_batch_dims = (None,) * (num_batch_dims - len(a_batch_dims)) + a_batch_dims
  b_batch_dims = (None,) * (num_batch_dims - len(b_batch_dims)) + b_batch_dims

  # Dimensions to squeeze from the inputs.
  a_squeeze: list[int] = []
  b_squeeze: list[int] = []

  # Positions of batch dimensions in squeezed inputs.
  a_batch = []
  b_batch = []

  # Desired index in final output of each kind of dimension, in the order that
  # lax.dot_general will emit them.
  idx_batch: list[int] = []
  idx_a_other: list[int] = []  # other = non-batch, non-contracting.
  idx_b_other: list[int] = []
  for i, (ba, bb) in enumerate(zip(a_batch_dims, b_batch_dims)):
    if ba is None:
      idx_b_other.append(i)
    elif bb is None:
      idx_a_other.append(i)
    elif core.definitely_equal(ba, 1):
      idx_b_other.append(i)
      a_squeeze.append(len(idx_batch) + len(idx_a_other) + len(a_squeeze))
    elif core.definitely_equal(bb, 1):
      idx_a_other.append(i)
      b_squeeze.append(len(idx_batch) + len(idx_b_other) + len(b_squeeze))
    elif core.definitely_equal(ba, bb):
      a_batch.append(len(idx_batch) + len(idx_a_other))
      b_batch.append(len(idx_batch) + len(idx_b_other))
      idx_batch.append(i)
    else:
      raise ValueError("Incompatible shapes for matmul arguments: {} and {}"
                       .format(shape(a), shape(b)))

  if a_is_mat: idx_a_other.append(num_batch_dims)
  if b_is_mat: idx_b_other.append(num_batch_dims + a_is_mat)
  perm = np.argsort(np.concatenate([idx_batch, idx_a_other, idx_b_other]))

  a = lax.squeeze(a, tuple(a_squeeze))
  b = lax.squeeze(b, tuple(b_squeeze))
  out = lax.dot_general(
    a, b, (((ndim(a) - 1,), (ndim(b) - 1 - b_is_mat,)), (a_batch, b_batch)),
    precision=precision, preferred_element_type=preferred_element_type)
  result = lax.transpose(out, perm)
  return lax_internal._convert_element_type(result, preferred_element_type, output_weak_type)


@partial(jit, static_argnames=('precision', 'preferred_element_type'), inline=True)
def vdot(
    a: ArrayLike, b: ArrayLike, *,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array:
  """Perform a conjugate multiplication of two 1D vectors.

  JAX implementation of :func:`numpy.vdot`.

  Args:
    a: first input array, if not 1D it will be flattened.
    b: second input array, if not 1D it will be flattened. Must have ``a.size == b.size``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    Scalar array (shape ``()``) containing the conjugate vector product of the inputs.

  See Also:
    - :func:`jax.numpy.vecdot`: batched vector product.
    - :func:`jax.numpy.matmul`: general matrix multiplication.
    - :func:`jax.lax.dot_general`: general N-dimensional batched dot product.

  Examples:
    >>> x = jnp.array([1j, 2j, 3j])
    >>> y = jnp.array([1., 2., 3.])
    >>> jnp.vdot(x, y)
    Array(0.-14.j, dtype=complex64)

    Note the difference between this and :func:`~jax.numpy.dot`, which does not
    conjugate the first input when complex:

    >>> jnp.dot(x, y)
    Array(0.+14.j, dtype=complex64)
  """
  util.check_arraylike("vdot", a, b)
  if issubdtype(_dtype(a), complexfloating):
    a = ufuncs.conj(a)
  return dot(ravel(a), ravel(b), precision=precision,
             preferred_element_type=preferred_element_type)


def vecdot(x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1,
           precision: PrecisionLike = None,
           preferred_element_type: DTypeLike | None = None) -> Array:
  """Perform a conjugate multiplication of two batched vectors.

  JAX implementation of :func:`numpy.vecdot`.

  Args:
    a: left-hand side array.
    b: right-hand side array. Size of ``b[axis]`` must match size of ``a[axis]``,
      and remaining dimensions must be broadcast-compatible.
    axis: axis along which to compute the dot product (default: -1)
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the conjugate dot product of ``a`` and ``b`` along ``axis``.
    The non-contracted dimensions are broadcast together.

  See Also:
    - :func:`jax.numpy.vdot`: flattened vector product.
    - :func:`jax.numpy.matmul`: general matrix multiplication.
    - :func:`jax.lax.dot_general`: general N-dimensional batched dot product.

  Examples:
    Vector conjugate-dot product of two 1D arrays:

    >>> a = jnp.array([1j, 2j, 3j])
    >>> b = jnp.array([4., 5., 6.])
    >>> jnp.linalg.vecdot(a, b)
    Array(0.-32.j, dtype=complex64)

    Batched vector dot product of two 2D arrays:

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> b = jnp.array([[2, 3, 4]])
    >>> jnp.linalg.vecdot(a, b, axis=-1)
    Array([20, 47], dtype=int32)
  """
  util.check_arraylike("jnp.vecdot", x1, x2)
  x1_arr, x2_arr = asarray(x1), asarray(x2)
  if x1_arr.shape[axis] != x2_arr.shape[axis]:
    raise ValueError(f"axes must match; got shapes {x1_arr.shape} and {x2_arr.shape} with {axis=}")
  x1_arr = jax.numpy.moveaxis(x1_arr, axis, -1)
  x2_arr = jax.numpy.moveaxis(x2_arr, axis, -1)
  return vectorize(partial(vdot, precision=precision, preferred_element_type=preferred_element_type),
                   signature="(n),(n)->()")(x1_arr, x2_arr)


def tensordot(a: ArrayLike, b: ArrayLike,
              axes: int | Sequence[int] | Sequence[Sequence[int]] = 2,
              *, precision: PrecisionLike = None,
              preferred_element_type: DTypeLike | None = None) -> Array:
  """Compute the tensor dot product of two N-dimensional arrays.

  JAX implementation of :func:`numpy.linalg.tensordot`.

  Args:
    a: N-dimensional array
    b: M-dimensional array
    axes: integer or tuple of sequences of integers. If an integer `k`, then
      sum over the last `k` axes of ``a`` and the first `k` axes of ``b``,
      in order. If a tuple, then ``axes[0]`` specifies the axes of ``a`` and
      ``axes[1]`` specifies the axes of ``b``.
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array containing the tensor dot product of the inputs

  See also:
    - :func:`jax.numpy.einsum`: NumPy API for more general tensor contractions.
    - :func:`jax.lax.dot_general`: XLA API for more general tensor contractions.

  Examples:
    >>> x1 = jnp.arange(24.).reshape(2, 3, 4)
    >>> x2 = jnp.ones((3, 4, 5))
    >>> jnp.tensordot(x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Equivalent result when specifying the axes as explicit sequences:

    >>> jnp.tensordot(x1, x2, axes=([1, 2], [0, 1]))
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Equivalent result via :func:`~jax.numpy.einsum`:

    >>> jnp.einsum('ijk,jkm->im', x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

    Setting ``axes=1`` for two-dimensional inputs is equivalent to a matrix
    multiplication:

    >>> x1 = jnp.array([[1, 2],
    ...                 [3, 4]])
    >>> x2 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> jnp.linalg.tensordot(x1, x2, axes=1)
    Array([[ 9, 12, 15],
           [19, 26, 33]], dtype=int32)
    >>> x1 @ x2
    Array([[ 9, 12, 15],
           [19, 26, 33]], dtype=int32)

    Setting ``axes=0`` for one-dimensional inputs is equivalent to
    :func:`~jax.numpy.outer`:

    >>> x1 = jnp.array([1, 2])
    >>> x2 = jnp.array([1, 2, 3])
    >>> jnp.linalg.tensordot(x1, x2, axes=0)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)
    >>> jnp.outer(x1, x2)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)
  """
  util.check_arraylike("tensordot", a, b)
  dtypes.check_user_dtype_supported(preferred_element_type, "tensordot")
  a, b = asarray(a), asarray(b)
  a_ndim = ndim(a)
  b_ndim = ndim(b)

  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(a, b, return_weak_type_flag=True)
  else:
    output_weak_type = False

  if type(axes) is int:
    if axes > min(a_ndim, b_ndim):
      msg = "Number of tensordot axes (axes {}) exceeds input ranks ({} and {})"
      raise TypeError(msg.format(axes, a.shape, b.shape))
    contracting_dims = tuple(range(a_ndim - axes, a_ndim)), tuple(range(axes))
  elif isinstance(axes, (tuple, list)) and len(axes) == 2:
    ax1, ax2 = axes
    if type(ax1) == type(ax2) == int:
      contracting_dims = ((_canonicalize_axis(ax1, a_ndim),),
                          (_canonicalize_axis(ax2, b_ndim),))
    elif isinstance(ax1, (tuple, list)) and isinstance(ax2, (tuple, list)):
      if len(ax1) != len(ax2):
        msg = "tensordot requires axes lists to have equal length, got {} and {}."
        raise TypeError(msg.format(ax1, ax2))
      contracting_dims = (tuple(_canonicalize_axis(i, a_ndim) for i in ax1),
                          tuple(_canonicalize_axis(i, b_ndim) for i in ax2))
    else:
      msg = ("tensordot requires both axes lists to be either ints, tuples or "
             "lists, got {} and {}")
      raise TypeError(msg.format(ax1, ax2))
  else:
    msg = ("tensordot axes argument must be an int, a pair of ints, or a pair "
           "of lists/tuples of ints.")
    raise TypeError(msg)
  result = lax.dot_general(a, b, (contracting_dims, ((), ())),
                           precision=precision, preferred_element_type=preferred_element_type)
  return lax_internal._convert_element_type(result, preferred_element_type, output_weak_type)


class Unoptimized(opt_einsum.paths.PathOptimizer):
  """Unoptimized path for einsum."""
  def __call__(self, inputs, *args, **kwargs):
    return [(0, 1)] * (len(inputs) - 1)

@overload
def einsum(
    subscript: str, /,
    *operands: ArrayLike,
    out: None = None,
    optimize: str | bool | list[tuple[int, ...]] = "optimal",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = lax.dot_general,
) -> Array: ...

@overload
def einsum(
    arr: ArrayLike,
    axes: Sequence[Any], /,
    *operands: ArrayLike | Sequence[Any],
    out: None = None,
    optimize: str | bool | list[tuple[int, ...]] = "optimal",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = lax.dot_general,
) -> Array: ...

def einsum(
    subscripts, /,
    *operands,
    out: None = None,
    optimize: str | bool | list[tuple[int, ...]] = "optimal",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = lax.dot_general,
) -> Array:
  """Einstein summation

  JAX implementation of :func:`numpy.einsum`.

  ``einsum`` is a powerful and generic API for computing various reductions,
  inner products, outer products, axis reorderings, and combinations thereof
  across one or more input arrays. It has a somewhat complicated overloaded API;
  the arguments below reflect the most common calling convention. The Examples
  section below demonstrates some of the alternative calling conventions.

  Args:
    subscripts: string containing axes names separated by commas.
    *operands: sequence of one or more arrays corresponding to the subscripts.
    optimize: specify how to optimize the order of computation. In JAX this defaults
      to ``"optimal"`` which produces optimized expressions via the opt_einsum_
      package. Other options are ``True`` (same as ``"optimal"``), ``False``
      (unoptimized), or any string supported by ``opt_einsum``, which
      includes ``"auto"``, ``"greedy"``, ``"eager"``, and others. It may also
      be a pre-computed path (see :func:`~jax.numpy.einsum_path`).
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``).
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.
    out: unsupported by JAX
    _dot_general: optionally override the ``dot_general`` callable used by ``einsum``.
      This parameter is experimental, and may be removed without warning at any time.

  Returns:
    array containing the result of the einstein summation.

  See also:
    :func:`jax.numpy.einsum_path`

  Examples:
    The mechanics of ``einsum`` are perhaps best demonstrated by example. Here we
    show how to use ``einsum`` to compute a number of quantities from one or more
    arrays. For more discussion and examples of ``einsum``, see the documentation
    of :func:`numpy.einsum`.

    >>> M = jnp.arange(16).reshape(4, 4)
    >>> x = jnp.arange(4)
    >>> y = jnp.array([5, 4, 3, 2])

    **Vector product**

    >>> jnp.einsum('i,i', x, y)
    Array(16, dtype=int32)
    >>> jnp.vecdot(x, y)
    Array(16, dtype=int32)

    Here are some alternative ``einsum`` calling conventions to compute the same
    result:

    >>> jnp.einsum('i,i->', x, y)  # explicit form
    Array(16, dtype=int32)
    >>> jnp.einsum(x, (0,), y, (0,))  # implicit form via indices
    Array(16, dtype=int32)
    >>> jnp.einsum(x, (0,), y, (0,), ())  # explicit form via indices
    Array(16, dtype=int32)

    **Matrix product**

    >>> jnp.einsum('ij,j->i', M, x)  # explicit form
    Array([14, 38, 62, 86], dtype=int32)
    >>> jnp.matmul(M, x)
    Array([14, 38, 62, 86], dtype=int32)

    Here are some alternative ``einsum`` calling conventions to compute the same
    result:

    >>> jnp.einsum('ij,j', M, x) # implicit form
    Array([14, 38, 62, 86], dtype=int32)
    >>> jnp.einsum(M, (0, 1), x, (1,), (0,)) # explicit form via indices
    Array([14, 38, 62, 86], dtype=int32)
    >>> jnp.einsum(M, (0, 1), x, (1,))  # implicit form via indices
    Array([14, 38, 62, 86], dtype=int32)

    **Outer product**

    >>> jnp.einsum("i,j->ij", x, y)
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)
    >>> jnp.outer(x, y)
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)

    Some other ways of computing outer products:

    >>> jnp.einsum("i,j", x, y)  # implicit form
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)
    >>> jnp.einsum(x, (0,), y, (1,), (0, 1))  # explicit form via indices
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)
    >>> jnp.einsum(x, (0,), y, (1,))  # implicit form via indices
    Array([[ 0,  0,  0,  0],
           [ 5,  4,  3,  2],
           [10,  8,  6,  4],
           [15, 12,  9,  6]], dtype=int32)

    **1D array sum**

    >>> jnp.einsum("i->", x)  # requires explicit form
    Array(6, dtype=int32)
    >>> jnp.einsum(x, (0,), ())  # explicit form via indices
    Array(6, dtype=int32)
    >>> jnp.sum(x)
    Array(6, dtype=int32)

    **Sum along an axis**

    >>> jnp.einsum("...j->...", M)  # requires explicit form
    Array([ 6, 22, 38, 54], dtype=int32)
    >>> jnp.einsum(M, (..., 0), (...,))  # explicit form via indices
    Array([ 6, 22, 38, 54], dtype=int32)
    >>> M.sum(-1)
    Array([ 6, 22, 38, 54], dtype=int32)

    **Matrix transpose**

    >>> y = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.einsum("ij->ji", y)  # explicit form
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
    >>> jnp.einsum("ji", y)  # implicit form
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
    >>> jnp.einsum(y, (1, 0))  # implicit form via indices
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
    >>> jnp.einsum(y, (0, 1), (1, 0))  # explicit form via indices
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)
    >>> jnp.transpose(y)
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)

    **Matrix diagonal**

    >>> jnp.einsum("ii->i", M)
    Array([ 0,  5, 10, 15], dtype=int32)
    >>> jnp.diagonal(M)
    Array([ 0,  5, 10, 15], dtype=int32)

    **Matrix trace**

    >>> jnp.einsum("ii", M)
    Array(30, dtype=int32)
    >>> jnp.trace(M)
    Array(30, dtype=int32)

    **Tensor products**

    >>> x = jnp.arange(30).reshape(2, 3, 5)
    >>> y = jnp.arange(60).reshape(3, 4, 5)
    >>> jnp.einsum('ijk,jlk->il', x, y)  # explicit form
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)
    >>> jnp.tensordot(x, y, axes=[(1, 2), (0, 2)])
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)
    >>> jnp.einsum('ijk,jlk', x, y)  # implicit form
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)
    >>> jnp.einsum(x, (0, 1, 2), y, (1, 3, 2), (0, 3))  # explicit form via indices
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)
    >>> jnp.einsum(x, (0, 1, 2), y, (1, 3, 2))  # implicit form via indices
    Array([[ 3340,  3865,  4390,  4915],
           [ 8290,  9940, 11590, 13240]], dtype=int32)

    **Chained dot products**

    >>> w = jnp.arange(5, 9).reshape(2, 2)
    >>> x = jnp.arange(6).reshape(2, 3)
    >>> y = jnp.arange(-2, 4).reshape(3, 2)
    >>> z = jnp.array([[2, 4, 6], [3, 5, 7]])
    >>> jnp.einsum('ij,jk,kl,lm->im', w, x, y, z)
    Array([[ 481,  831, 1181],
           [ 651, 1125, 1599]], dtype=int32)
    >>> jnp.einsum(w, (0, 1), x, (1, 2), y, (2, 3), z, (3, 4))  # implicit, via indices
    Array([[ 481,  831, 1181],
           [ 651, 1125, 1599]], dtype=int32)
    >>> w @ x @ y @ z  # direct chain of matmuls
    Array([[ 481,  831, 1181],
           [ 651, 1125, 1599]], dtype=int32)
    >>> jnp.linalg.multi_dot([w, x, y, z])
    Array([[ 481,  831, 1181],
           [ 651, 1125, 1599]], dtype=int32)

  .. _opt_einsum: https://github.com/dgasmith/opt_einsum
  """
  operands = (subscripts, *operands)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.einsum is not supported.")
  spec = operands[0] if isinstance(operands[0], str) else None
  path_type = 'optimal' if optimize is True else Unoptimized() if optimize is False else optimize

  # Allow handling of shape polymorphism
  non_constant_dim_types = {
      type(d) for op in operands if not isinstance(op, str)
      for d in np.shape(op) if not core.is_constant_dim(d)
  }
  if not non_constant_dim_types:
    contract_path = opt_einsum.contract_path
  else:
    ty = next(iter(non_constant_dim_types))
    contract_path = _poly_einsum_handlers.get(ty, _default_poly_einsum_handler)
  # using einsum_call=True here is an internal api for opt_einsum... sorry
  operands, contractions = contract_path(
        *operands, einsum_call=True, use_blas=True, optimize=path_type)

  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)

  einsum = jit(_einsum, static_argnums=(1, 2, 3, 4), inline=True)
  if spec is not None:
    einsum = jax.named_call(einsum, name=spec)
  return einsum(operands, contractions, precision,
                preferred_element_type, _dot_general)


# Enable other modules to override einsum_contact_path.
# Indexed by the type of the non constant dimension
_poly_einsum_handlers = {}  # type: ignore

def _default_poly_einsum_handler(*operands, **kwargs):
  dummy = collections.namedtuple('dummy', ['shape', 'dtype'])
  dummies = [dummy(tuple(d if type(d) is int else 8 for d in x.shape), x.dtype)
             if hasattr(x, 'dtype') else x for x in operands]
  mapping = {id(d): i for i, d in enumerate(dummies)}
  out_dummies, contractions = opt_einsum.contract_path(*dummies, **kwargs)
  contract_operands = [operands[mapping[id(d)]] for d in out_dummies]
  return contract_operands, contractions

@overload
def einsum_path(
    subscripts: str, /,
    *operands: ArrayLike,
    optimize: bool | str | list[tuple[int, ...]] =  ...,
) -> tuple[list[tuple[int, ...]], Any]: ...

@overload
def einsum_path(
    arr: ArrayLike,
    axes: Sequence[Any], /,
    *operands: ArrayLike | Sequence[Any],
    optimize: bool | str | list[tuple[int, ...]] =  ...,
) -> tuple[list[tuple[int, ...]], Any]: ...

def einsum_path(
    subscripts, /,
    *operands,
    optimize: bool | str | list[tuple[int, ...]] = 'auto'
  ) -> tuple[list[tuple[int, ...]], Any]:
  """Evaluates the optimal contraction path without evaluating the einsum.

  JAX implementation of :func:`numpy.einsum_path`. This function calls into
  the opt_einsum_ package, and makes use of its optimization routines.

  Args:
    subscripts: string containing axes names separated by commas.
    *operands: sequence of one or more arrays corresponding to the subscripts.
    optimize: specify how to optimize the order of computation. In JAX this defaults
      to ``"auto"``. Other options are ``True`` (same as ``"optimize"``), ``False``
      (unoptimized), or any string supported by ``opt_einsum``, which
      includes ``"optimize"``,, ``"greedy"``, ``"eager"``, and others.

  Returns:
    A tuple containing the path that may be passed to :func:`~jax.numpy.einsum`, and a
    printable object representing this optimal path.

  Examples:
    >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
    >>> x = jax.random.randint(key1, minval=-5, maxval=5, shape=(2, 3))
    >>> y = jax.random.randint(key2, minval=-5, maxval=5, shape=(3, 100))
    >>> z = jax.random.randint(key3, minval=-5, maxval=5, shape=(100, 5))
    >>> path, path_info = jnp.einsum_path("ij,jk,kl", x, y, z, optimize="optimal")
    >>> print(path)
    [(1, 2), (0, 1)]
    >>> print(path_info)
          Complete contraction:  ij,jk,kl->il
                Naive scaling:  4
            Optimized scaling:  3
              Naive FLOP count:  9.000e+3
          Optimized FLOP count:  3.060e+3
          Theoretical speedup:  2.941e+0
          Largest intermediate:  1.500e+1 elements
        --------------------------------------------------------------------------------
        scaling        BLAS                current                             remaining
        --------------------------------------------------------------------------------
          3           GEMM              kl,jk->lj                             ij,lj->il
          3           GEMM              lj,ij->il                                il->il

    Use the computed path in :func:`~jax.numpy.einsum`:

    >>> jnp.einsum("ij,jk,kl", x, y, z, optimize=path)
    Array([[-539,  216,   95,  592,  209],
           [ 527,   76,  285, -436, -529]], dtype=int32)

  .. _opt_einsum: https://github.com/dgasmith/opt_einsum
  """
  if optimize is True:
    optimize = 'optimal'
  elif optimize is False:
    optimize = Unoptimized()
  return opt_einsum.contract_path(subscripts, *operands, optimize=optimize)

def _removechars(s, chars):
  return s.translate(str.maketrans(dict.fromkeys(chars)))


def _einsum(
    operands: Sequence,
    contractions: Sequence[tuple[tuple[int, ...], frozenset[str], str]],
    precision,
    preferred_element_type,
    _dot_general=lax.dot_general,
):
  dtypes.check_user_dtype_supported(preferred_element_type, "einsum")
  operands = list(map(asarray, operands))
  if preferred_element_type is None:
    preferred_element_type, output_weak_type = dtypes.result_type(*operands, return_weak_type_flag=True)
  else:
    output_weak_type = False

  def sum(x, axes):
    if dtypes.result_type(x, preferred_element_type) != x.dtype:
      x = x.astype(preferred_element_type)
    return lax.reduce(x, np.array(0, x.dtype),
                      lax.add if x.dtype != bool_ else lax.bitwise_or, axes)

  def sum_uniques(operand, names, uniques):
    if uniques:
      axes = [names.index(name) for name in uniques]
      operand = sum(operand, axes)
      names = _removechars(names, uniques)
    return operand, names

  def sum_repeats(operand, names, counts, keep_names):
    for name, count in counts.items():
      if count > 1:
        axes = [i for i, n in enumerate(names) if n == name]
        eye = lax_internal._delta(np.dtype('bool'), operand.shape, axes)
        operand = lax.select(eye, operand, zeros_like(operand))
        if name not in keep_names:
          operand = sum(operand, axes)
          names = names.replace(name, '')
        else:
          operand = sum(operand, axes[:-1])
          names = names.replace(name, '', count - 1)
    return operand, names

  def filter_singleton_dims(operand, names, other_shape, other_names):
    eq = core.definitely_equal
    keep = [not eq(operand.shape[i], 1) or j == -1 or eq(other_shape[j], 1)
            for i, j in enumerate(map(other_names.find, names))]
    sqez_axes, keep_axes = partition_list(keep, list(range(operand.ndim)))
    return lax.squeeze(operand, sqez_axes), "".join(names[i] for i in keep_axes)

  for operand_indices, contracted_names_set, einstr in contractions:
    contracted_names = sorted(contracted_names_set)
    input_str, result_names = einstr.split('->')
    input_names = input_str.split(',')

    # switch on the number of operands to be processed in this loop iteration.
    # every case here sets 'operand' and 'names'.
    if len(operand_indices) == 1:
      operand = operands.pop(operand_indices[0])
      names, = input_names
      counts = collections.Counter(names)

      # sum out unique contracted indices with a single reduce-sum
      uniques = [name for name in contracted_names if counts[name] == 1]
      operand, names = sum_uniques(operand, names, uniques)

      # for every repeated index, do a contraction against an identity matrix
      operand, names = sum_repeats(operand, names, counts, result_names)

    elif len(operand_indices) == 2:
      lhs, rhs = map(operands.pop, operand_indices)
      lhs_names, rhs_names = input_names

      # handle cases where one side of a contracting or batch dimension is 1
      # but its counterpart is not.
      lhs, lhs_names = filter_singleton_dims(lhs, lhs_names, shape(rhs),
                                             rhs_names)
      rhs, rhs_names = filter_singleton_dims(rhs, rhs_names, shape(lhs),
                                             lhs_names)

      lhs_counts = collections.Counter(lhs_names)
      rhs_counts = collections.Counter(rhs_names)

      # sum out unique contracted indices in lhs and rhs
      lhs_uniques = [name for name in contracted_names
                     if lhs_counts[name] == 1 and rhs_counts[name] == 0]
      lhs, lhs_names = sum_uniques(lhs, lhs_names, lhs_uniques)

      rhs_uniques = [name for name in contracted_names
                     if rhs_counts[name] == 1 and lhs_counts[name] == 0]
      rhs, rhs_names = sum_uniques(rhs, rhs_names, rhs_uniques)

      # for every repeated index, contract against an identity matrix
      lhs, lhs_names = sum_repeats(lhs, lhs_names, lhs_counts,
                                   result_names + rhs_names)
      rhs, rhs_names = sum_repeats(rhs, rhs_names, rhs_counts,
                                   result_names + lhs_names)

      lhs_or_rhs_names = set(lhs_names) | set(rhs_names)
      contracted_names = [x for x in contracted_names if x in lhs_or_rhs_names]
      lhs_and_rhs_names = set(lhs_names) & set(rhs_names)
      batch_names = [x for x in result_names if x in lhs_and_rhs_names]

      lhs_batch, rhs_batch = unzip2((lhs_names.find(n), rhs_names.find(n))
                                    for n in batch_names)

      # NOTE(mattjj): this can fail non-deterministically in python3, maybe
      # due to opt_einsum
      assert config.dynamic_shapes.value or all(
        name in lhs_names and name in rhs_names and
        lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
        for name in contracted_names), (
          "Incompatible reduction dimensions: "
          f"lhs.shape={lhs.shape} lhs_names={lhs_names} "
          f"rhs.shape={rhs.shape} rhs_names={rhs_names}")

      # contract using dot_general
      batch_names_str = ''.join(batch_names)
      lhs_cont, rhs_cont = unzip2((lhs_names.index(n), rhs_names.index(n))
                                  for n in contracted_names)
      deleted_names = batch_names_str + ''.join(contracted_names)
      remaining_lhs_names = _removechars(lhs_names, deleted_names)
      remaining_rhs_names = _removechars(rhs_names, deleted_names)
      # Try both orders of lhs and rhs, in the hope that one of them means we
      # don't need an explicit transpose. opt_einsum likes to contract from
      # right to left, so we expect (rhs,lhs) to have the best chance of not
      # needing a transpose.
      names = batch_names_str + remaining_rhs_names + remaining_lhs_names
      if names == result_names:
        dimension_numbers = ((rhs_cont, lhs_cont), (rhs_batch, lhs_batch))
        operand = _dot_general(rhs, lhs, dimension_numbers, precision,
                               preferred_element_type=preferred_element_type)
      else:
        names = batch_names_str + remaining_lhs_names + remaining_rhs_names
        dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
        operand = _dot_general(lhs, rhs, dimension_numbers, precision,
                               preferred_element_type=preferred_element_type)
    else:
      raise NotImplementedError  # if this is actually reachable, open an issue!

    # the resulting 'operand' with axis labels 'names' should be a permutation
    # of the desired result
    assert len(names) == len(result_names) == len(set(names))
    assert set(names) == set(result_names)
    if names != result_names:
      perm = tuple(names.index(name) for name in result_names)
      operand = lax.transpose(operand, perm)
    operands.append(operand)  # used in next iteration

  return lax_internal._convert_element_type(operands[0], preferred_element_type, output_weak_type)


@partial(jit, static_argnames=('precision', 'preferred_element_type'), inline=True)
def inner(
    a: ArrayLike, b: ArrayLike, *, precision: PrecisionLike = None,
    preferred_element_type: DType | None = None,
) -> Array:
  """Compute the inner product of two arrays.

  JAX implementation of :func:`numpy.inner`.

  Unlike :func:`jax.numpy.matmul` or :func:`jax.numpy.dot`, this always performs
  a contraction along the last dimension of each input.

  Args:
    a: array of shape ``(..., N)``
    b: array of shape ``(..., N)``
    precision: either ``None`` (default), which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      such values indicating precision of ``a`` and ``b``.
    preferred_element_type: either ``None`` (default), which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    array of shape ``(*a.shape[:-1], *b.shape[:-1])`` containing the batched vector
    product of the inputs.

  See also:
    - :func:`jax.numpy.vecdot`: conjugate multiplication along a specified axis.
    - :func:`jax.numpy.tensordot`: general tensor multiplication.
    - :func:`jax.numpy.matmul`: general batched matrix & vector multiplication.

  Examples:
    For 1D inputs, this implements standard (non-conjugate) vector multiplication:

    >>> a = jnp.array([1j, 3j, 4j])
    >>> b = jnp.array([4., 2., 5.])
    >>> jnp.inner(a, b)
    Array(0.+30.j, dtype=complex64)

    For multi-dimensional inputs, batch dimensions are stacked rather than broadcast:

    >>> a = jnp.ones((2, 3))
    >>> b = jnp.ones((5, 3))
    >>> jnp.inner(a, b).shape
    (2, 5)
  """
  util.check_arraylike("inner", a, b)
  if ndim(a) == 0 or ndim(b) == 0:
    a = asarray(a, dtype=preferred_element_type)
    b = asarray(b, dtype=preferred_element_type)
    return a * b
  return tensordot(a, b, (-1, -1), precision=precision,
                   preferred_element_type=preferred_element_type)


@util.implements(np.outer, skip_params=['out'])
@partial(jit, inline=True)
def outer(a: ArrayLike, b: ArrayLike, out: None = None) -> Array:
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.outer is not supported.")
  util.check_arraylike("outer", a, b)
  a, b = util.promote_dtypes(a, b)
  return ravel(a)[:, None] * ravel(b)[None, :]

@util.implements(np.cross)
@partial(jit, static_argnames=('axisa', 'axisb', 'axisc', 'axis'))
def cross(a, b, axisa: int = -1, axisb: int = -1, axisc: int = -1,
          axis: int | None = None):
  # TODO(jakevdp): NumPy 2.0 deprecates 2D inputs. Follow suit here.
  util.check_arraylike("cross", a, b)
  if axis is not None:
    axisa = axis
    axisb = axis
    axisc = axis
  a = moveaxis(a, axisa, -1)
  b = moveaxis(b, axisb, -1)

  if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
    raise ValueError("Dimension must be either 2 or 3 for cross product")

  if a.shape[-1] == 2 and b.shape[-1] == 2:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

  a0 = a[..., 0]
  a1 = a[..., 1]
  a2 = a[..., 2] if a.shape[-1] == 3 else zeros_like(a0)
  b0 = b[..., 0]
  b1 = b[..., 1]
  b2 = b[..., 2] if b.shape[-1] == 3 else zeros_like(b0)
  c = array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])
  return moveaxis(c, 0, axisc)


@util.implements(np.kron)
@jit
def kron(a: ArrayLike, b: ArrayLike) -> Array:
  util.check_arraylike("kron", a, b)
  a, b = util.promote_dtypes(a, b)
  if ndim(a) < ndim(b):
    a = expand_dims(a, range(ndim(b) - ndim(a)))
  elif ndim(b) < ndim(a):
    b = expand_dims(b, range(ndim(a) - ndim(b)))
  a_reshaped = expand_dims(a, range(1, 2 * ndim(a), 2))
  b_reshaped = expand_dims(b, range(0, 2 * ndim(b), 2))
  out_shape = tuple(np.multiply(shape(a), shape(b)))
  return reshape(lax.mul(a_reshaped, b_reshaped), out_shape)


@util.implements(np.vander)
@partial(jit, static_argnames=('N', 'increasing'))
def vander(
    x: ArrayLike, N: int | None = None, increasing: bool = False
) -> Array:
  util.check_arraylike("vander", x)
  x = asarray(x)
  if x.ndim != 1:
    raise ValueError("x must be a one-dimensional array")
  N = x.shape[0] if N is None else core.concrete_or_error(
    operator.index, N, "'N' argument of jnp.vander()")
  if N < 0:
    raise ValueError("N must be nonnegative")

  iota = lax.iota(x.dtype, N)
  if not increasing:
    iota = lax.sub(_lax_const(iota, N - 1), iota)

  return ufuncs.power(x[..., None], expand_dims(iota, tuple(range(x.ndim))))


### Misc

def argwhere(
    a: ArrayLike,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> Array:
  """Find the indices of nonzero array elements

  JAX implementation of :func:`numpy.argwhere`.

  ``jnp.argwhere(x)`` is essentially equivalent to ``jnp.column_stack(jnp.nonzero(x))``
  with special handling for zero-dimensional (i.e. scalar) inputs.

  Because the size of the output of ``argwhere`` is data-dependent, the function is not
  typically compatible with JIT. The JAX version adds the optional ``size`` argument, which
  specifies the size of the leading dimension of the output - it must be specified statically
  for ``jnp.argwhere`` to be compiled with non-static operands. See :func:`jax.numpy.nonzero`
  for a full discussion of ``size`` and its semantics.

  Args:
    a: array for which to find nonzero elements
    size: optional integer specifying statically the number of expected nonzero elements.
      This must be specified in order to use ``argwhere`` within JAX transformations like
      :func:`jax.jit`. See :func:`jax.numpy.nonzero` for more information.
    fill_value: optional array specifying the fill value when ``size`` is specified.
      See :func:`jax.numpy.nonzero` for more information.

  Returns:
    a two-dimensional array of shape ``[size, x.ndim]``. If ``size`` is not specified as
    an argument, it is equal to the number of nonzero elements in ``x``.

  See Also:
    - :func:`jax.numpy.where`
    - :func:`jax.numpy.nonzero`

  Examples:
    Two-dimensional array:

    >>> x = jnp.array([[1, 0, 2],
    ...                [0, 3, 0]])
    >>> jnp.argwhere(x)
    Array([[0, 0],
           [0, 2],
           [1, 1]], dtype=int32)

    Equivalent computation using :func:`jax.numpy.column_stack` and :func:`jax.numpy.nonzero`:

    >>> jnp.column_stack(jnp.nonzero(x))
    Array([[0, 0],
           [0, 2],
           [1, 1]], dtype=int32)

    Special case for zero-dimensional (i.e. scalar) inputs:

    >>> jnp.argwhere(1)
    Array([], shape=(1, 0), dtype=int32)
    >>> jnp.argwhere(0)
    Array([], shape=(0, 0), dtype=int32)
  """
  result = transpose(vstack(nonzero(atleast_1d(a), size=size, fill_value=fill_value)))
  if ndim(a) == 0:
    return result[:0].reshape(result.shape[0], 0)
  return result.reshape(result.shape[0], ndim(a))


@util.implements(np.argmax, skip_params=['out'])
def argmax(a: ArrayLike, axis: int | None = None, out: None = None,
           keepdims: bool | None = None) -> Array:
  util.check_arraylike("argmax", a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.argmax is not supported.")
  return _argmax(asarray(a), None if axis is None else operator.index(axis),
                 keepdims=bool(keepdims))

@partial(jit, static_argnames=('axis', 'keepdims'), inline=True)
def _argmax(a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
  if axis is None:
    dims = list(range(ndim(a)))
    a = ravel(a)
    axis = 0
  else:
    dims = [axis]
  if a.shape[axis] == 0:
    raise ValueError("attempt to get argmax of an empty sequence")
  result = lax.argmax(a, _canonicalize_axis(axis, a.ndim), dtypes.canonicalize_dtype(int_))
  return expand_dims(result, dims) if keepdims else result

@util.implements(np.argmin, skip_params=['out'])
def argmin(a: ArrayLike, axis: int | None = None, out: None = None,
           keepdims: bool | None = None) -> Array:
  util.check_arraylike("argmin", a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.argmin is not supported.")
  return _argmin(asarray(a), None if axis is None else operator.index(axis),
                 keepdims=bool(keepdims))

@partial(jit, static_argnames=('axis', 'keepdims'), inline=True)
def _argmin(a: Array, axis: int | None = None, keepdims: bool = False) -> Array:
  if axis is None:
    dims = list(range(ndim(a)))
    a = ravel(a)
    axis = 0
  else:
    dims = [axis]
  if a.shape[axis] == 0:
    raise ValueError("attempt to get argmin of an empty sequence")
  result = lax.argmin(a, _canonicalize_axis(axis, a.ndim), dtypes.canonicalize_dtype(int_))
  return expand_dims(result, dims) if keepdims else result


_NANARG_DOC = """\
Warning: jax.numpy.arg{} returns -1 for all-NaN slices and does not raise
an error.
"""


@util.implements(np.nanargmax, lax_description=_NANARG_DOC.format("max"), skip_params=['out'])
def nanargmax(
    a: ArrayLike,
    axis: int | None = None,
    out: None = None,
    keepdims: bool | None = None,
) -> Array:
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.nanargmax is not supported.")
  return _nanargmax(a, None if axis is None else operator.index(axis), keepdims=bool(keepdims))


@partial(jit, static_argnames=('axis', 'keepdims'))
def _nanargmax(a, axis: int | None = None, keepdims: bool = False):
  util.check_arraylike("nanargmax", a)
  if not issubdtype(_dtype(a), inexact):
    return argmax(a, axis=axis, keepdims=keepdims)
  nan_mask = ufuncs.isnan(a)
  a = where(nan_mask, -inf, a)
  res = argmax(a, axis=axis, keepdims=keepdims)
  return where(reductions.all(nan_mask, axis=axis, keepdims=keepdims), -1, res)


@util.implements(np.nanargmin, lax_description=_NANARG_DOC.format("min"),  skip_params=['out'])
def nanargmin(
    a: ArrayLike,
    axis: int | None = None,
    out: None = None,
    keepdims: bool | None = None,
) -> Array:
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.nanargmin is not supported.")
  return _nanargmin(a, None if axis is None else operator.index(axis), keepdims=bool(keepdims))


@partial(jit, static_argnames=('axis', 'keepdims'))
def _nanargmin(a, axis: int | None = None, keepdims : bool = False):
  util.check_arraylike("nanargmin", a)
  if not issubdtype(_dtype(a), inexact):
    return argmin(a, axis=axis, keepdims=keepdims)
  nan_mask = ufuncs.isnan(a)
  a = where(nan_mask, inf, a)
  res = argmin(a, axis=axis, keepdims=keepdims)
  return where(reductions.all(nan_mask, axis=axis, keepdims=keepdims), -1, res)


@partial(jit, static_argnames=('axis', 'kind', 'order', 'stable', 'descending'))
def sort(
    a: ArrayLike,
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array:
  """Return a sorted copy of an array.

  JAX implementation of :func:`numpy.sort`.

  Args:
    a: array to sort
    axis: integer axis along which to sort. Defaults to ``-1``, i.e. the last
      axis. If ``None``, then ``a`` is flattened before being sorted.
    stable: boolean specifying whether a stable sort should be used. Default=True.
    descending: boolean specifying whether to sort in descending order. Default=False.
    kind: deprecated; instead specify sort algorithm using stable=True or stable=False.
    order: not supported by JAX

  Returns:
    Sorted array of shape ``a.shape`` (if ``axis`` is an integer) or of shape
    ``(a.size,)`` (if ``axis`` is None).

  Examples:
    Simple 1-dimensional sort

    >>> x = jnp.array([1, 3, 5, 4, 2, 1])
    >>> jnp.sort(x)
    Array([1, 1, 2, 3, 4, 5], dtype=int32)

    Sort along the last axis of an array:

    >>> x = jnp.array([[2, 1, 3],
    ...                [4, 3, 6]])
    >>> jnp.sort(x, axis=1)
    Array([[1, 2, 3],
           [3, 4, 6]], dtype=int32)

  See also:
    - :func:`jax.numpy.argsort`: return indices of sorted values.
    - :func:`jax.numpy.lexsort`: lexicographical sort of multiple arrays.
    - :func:`jax.lax.sort`: lower-level function wrapping XLA's Sort operator.
  """
  util.check_arraylike("sort", a)
  if kind is not None:
    raise TypeError("'kind' argument to sort is not supported. Use"
                    " stable=True or stable=False to specify sort stability.")
  if order is not None:
    raise TypeError("'order' argument to sort is not supported.")
  if axis is None:
    arr = ravel(a)
    axis = 0
  else:
    arr = asarray(a)
  dimension = _canonicalize_axis(axis, arr.ndim)
  result = lax.sort(arr, dimension=dimension, is_stable=stable)
  return lax.rev(result, dimensions=[dimension]) if descending else result


@util.implements(np.sort_complex)
@jit
def sort_complex(a: ArrayLike) -> Array:
  util.check_arraylike("sort_complex", a)
  a = lax.sort(asarray(a), dimension=0)
  return lax.convert_element_type(a, dtypes.to_complex_dtype(a.dtype))

@util.implements(np.lexsort)
@partial(jit, static_argnames=('axis',))
def lexsort(keys: Array | np.ndarray | Sequence[ArrayLike], axis: int = -1) -> Array:
  key_tuple = tuple(keys)
  util.check_arraylike("lexsort", *key_tuple)
  key_arrays = tuple(asarray(k) for k in key_tuple)
  if len(key_arrays) == 0:
    raise TypeError("need sequence of keys with len > 0 in lexsort")
  if len({shape(key) for key in key_arrays}) > 1:
    raise ValueError("all keys need to be the same shape")
  if ndim(key_arrays[0]) == 0:
    return array(0, dtype=dtypes.canonicalize_dtype(int_))
  axis = _canonicalize_axis(axis, ndim(key_arrays[0]))
  use_64bit_index = key_arrays[0].shape[axis] >= (1 << 31)
  iota = lax.broadcasted_iota(int64 if use_64bit_index else int_, shape(key_arrays[0]), axis)
  return lax.sort((*key_arrays[::-1], iota), dimension=axis, num_keys=len(key_arrays))[-1]


@partial(jit, static_argnames=('axis', 'kind', 'order', 'stable', 'descending'))
def argsort(
    a: ArrayLike,
    axis: int | None = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Array:
  """Return indices that sort an array.

  JAX implementation of :func:`numpy.argsort`.

  Args:
    a: array to sort
    axis: integer axis along which to sort. Defaults to ``-1``, i.e. the last
      axis. If ``None``, then ``a`` is flattened before being sorted.
    stable: boolean specifying whether a stable sort should be used. Default=True.
    descending: boolean specifying whether to sort in descending order. Default=False.
    kind: deprecated; instead specify sort algorithm using stable=True or stable=False.
    order: not supported by JAX

  Returns:
    Array of indices that sort an array. Returned array will be of shape ``a.shape``
    (if ``axis`` is an integer) or of shape ``(a.size,)`` (if ``axis`` is None).

  Examples:
    Simple 1-dimensional sort

    >>> x = jnp.array([1, 3, 5, 4, 2, 1])
    >>> indices = jnp.argsort(x)
    >>> indices
    Array([0, 5, 4, 1, 3, 2], dtype=int32)
    >>> x[indices]
    Array([1, 1, 2, 3, 4, 5], dtype=int32)

    Sort along the last axis of an array:

    >>> x = jnp.array([[2, 1, 3],
    ...                [6, 4, 3]])
    >>> indices = jnp.argsort(x, axis=1)
    >>> indices
    Array([[1, 0, 2],
           [2, 1, 0]], dtype=int32)
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[1, 2, 3],
           [3, 4, 6]], dtype=int32)


  See also:
    - :func:`jax.numpy.sort`: return sorted values directly.
    - :func:`jax.numpy.lexsort`: lexicographical sort of multiple arrays.
    - :func:`jax.lax.sort`: lower-level function wrapping XLA's Sort operator.
  """
  util.check_arraylike("argsort", a)
  arr = asarray(a)
  if kind is not None:
    raise TypeError("'kind' argument to argsort is not supported. Use"
                    " stable=True or stable=False to specify sort stability.")
  if order is not None:
    raise TypeError("'order' argument to argsort is not supported.")
  if axis is None:
    arr = ravel(arr)
    axis = 0
  else:
    arr = asarray(a)
  dimension = _canonicalize_axis(axis, arr.ndim)
  use_64bit_index = not core.is_constant_dim(arr.shape[dimension]) or arr.shape[dimension] >= (1 << 31)
  iota = lax.broadcasted_iota(int64 if use_64bit_index else int_, arr.shape, dimension)
  # For stable descending sort, we reverse the array and indices to ensure that
  # duplicates remain in their original order when the final indices are reversed.
  # For non-stable descending sort, we can avoid these extra operations.
  if descending and stable:
    arr = lax.rev(arr, dimensions=[dimension])
    iota = lax.rev(iota, dimensions=[dimension])
  _, indices = lax.sort_key_val(arr, iota, dimension=dimension, is_stable=stable)
  return lax.rev(indices, dimensions=[dimension]) if descending else indices


@partial(jit, static_argnames=['kth', 'axis'])
def partition(a: ArrayLike, kth: int, axis: int = -1) -> Array:
  """Returns a partially-sorted copy of an array.

  JAX implementation of :func:`numpy.partition`. The JAX version differs from
  NumPy in the treatment of NaN entries: NaNs which have the negative bit set
  are sorted to the beginning of the array.

  Args:
    a: array to be partitioned.
    kth: static integer index about which to partition the array.
    axis: static integer axis along which to partition the array; default is -1.

  Returns:
    A copy of ``a`` partitioned at the ``kth`` value along ``axis``. The entries
    before ``kth`` are values smaller than ``take(a, kth, axis)``, and entries
    after ``kth`` are indices of values larger than ``take(a, kth, axis)``

  Note:
    The JAX version requires the ``kth`` argument to be a static integer rather than
    a general array. This is implemented via two calls to :func:`jax.lax.top_k`. If
    you're only accessing the top or bottom k values of the output, it may be more
    efficient to call :func:`jax.lax.top_k` directly.

  See Also:
    - :func:`jax.numpy.sort`: full sort
    - :func:`jax.numpy.argpartition`: indirect partial sort
    - :func:`jax.lax.top_k`: directly find the top k entries
    - :func:`jax.lax.approx_max_k`: compute the approximate top k entries
    - :func:`jax.lax.approx_min_k`: compute the approximate bottom k entries

  Examples:
    >>> x = jnp.array([6, 8, 4, 3, 1, 9, 7, 5, 2, 3])
    >>> kth = 4
    >>> x_partitioned = jnp.partition(x, kth)
    >>> x_partitioned
    Array([1, 2, 3, 3, 4, 9, 8, 7, 6, 5], dtype=int32)

    The result is a partially-sorted copy of the input. All values before ``kth``
    are of smaller than the pivot value, and all values after ``kth`` are larger
    than the pivot value:

    >>> smallest_values = x_partitioned[:kth]
    >>> pivot_value = x_partitioned[kth]
    >>> largest_values = x_partitioned[kth + 1:]
    >>> print(smallest_values, pivot_value, largest_values)
    [1 2 3 3] 4 [9 8 7 6 5]

    Notice that among ``smallest_values`` and ``largest_values``, the returned
    order is arbitrary and implementation-dependent.
  """
  # TODO(jakevdp): handle NaN values like numpy.
  util.check_arraylike("partition", a)
  arr = asarray(a)
  if issubdtype(arr.dtype, np.complexfloating):
    raise NotImplementedError("jnp.partition for complex dtype is not implemented.")
  axis = _canonicalize_axis(axis, arr.ndim)
  kth = _canonicalize_axis(kth, arr.shape[axis])

  arr = swapaxes(arr, axis, -1)
  if dtypes.isdtype(arr.dtype, "unsigned integer"):
    # Here, we apply a trick to handle correctly 0 values for unsigned integers
    bottom = -lax.top_k(-(arr + 1), kth + 1)[0] - 1
  else:
    bottom = -lax.top_k(-arr, kth + 1)[0]
  top = lax.top_k(arr, arr.shape[-1] - kth - 1)[0]
  out = lax.concatenate([bottom, top], dimension=arr.ndim - 1)
  return swapaxes(out, -1, axis)


@partial(jit, static_argnames=['kth', 'axis'])
def argpartition(a: ArrayLike, kth: int, axis: int = -1) -> Array:
  """Returns indices that partially sort an array.

  JAX implementation of :func:`numpy.argpartition`. The JAX version differs from
  NumPy in the treatment of NaN entries: NaNs which have the negative bit set are
  sorted to the beginning of the array.

  Args:
    a: array to be partitioned.
    kth: static integer index about which to partition the array.
    axis: static integer axis along which to partition the array; default is -1.

  Returns:
    Indices which partition ``a`` at the ``kth`` value along ``axis``. The entries
    before ``kth`` are indices of values smaller than ``take(a, kth, axis)``, and
    entries after ``kth`` are indices of values larger than ``take(a, kth, axis)``

  Note:
    The JAX version requires the ``kth`` argument to be a static integer rather than
    a general array. This is implemented via two calls to :func:`jax.lax.top_k`. If
    you're only accessing the top or bottom k values of the output, it may be more
    efficient to call :func:`jax.lax.top_k` directly.

  See Also:
    - :func:`jax.numpy.partition`: direct partial sort
    - :func:`jax.numpy.argsort`: full indirect sort
    - :func:`jax.lax.top_k`: directly find the top k entries
    - :func:`jax.lax.approx_max_k`: compute the approximate top k entries
    - :func:`jax.lax.approx_min_k`: compute the approximate bottom k entries

  Examples:
    >>> x = jnp.array([6, 8, 4, 3, 1, 9, 7, 5, 2, 3])
    >>> kth = 4
    >>> idx = jnp.argpartition(x, kth)
    >>> idx
    Array([4, 8, 3, 9, 2, 0, 1, 5, 6, 7], dtype=int32)

    The result is a sequence of indices that partially sort the input. All indices
    before ``kth`` are of values smaller than the pivot value, and all indices
    after ``kth`` are of values larger than the pivot value:

    >>> x_partitioned = x[idx]
    >>> smallest_values = x_partitioned[:kth]
    >>> pivot_value = x_partitioned[kth]
    >>> largest_values = x_partitioned[kth + 1:]
    >>> print(smallest_values, pivot_value, largest_values)
    [1 2 3 3] 4 [6 8 9 7 5]

    Notice that among ``smallest_values`` and ``largest_values``, the returned
    order is arbitrary and implementation-dependent.
  """
  # TODO(jakevdp): handle NaN values like numpy.
  util.check_arraylike("partition", a)
  arr = asarray(a)
  if issubdtype(arr.dtype, np.complexfloating):
    raise NotImplementedError("jnp.argpartition for complex dtype is not implemented.")
  axis = _canonicalize_axis(axis, arr.ndim)
  kth = _canonicalize_axis(kth, arr.shape[axis])

  arr = swapaxes(arr, axis, -1)
  if dtypes.isdtype(arr.dtype, "unsigned integer"):
    # Here, we apply a trick to handle correctly 0 values for unsigned integers
    bottom_ind = lax.top_k(-(arr + 1), kth + 1)[1]
  else:
    bottom_ind = lax.top_k(-arr, kth + 1)[1]

  # To avoid issues with duplicate values, we compute the top indices via a proxy
  set_to_zero = lambda a, i: a.at[i].set(0)
  for _ in range(arr.ndim - 1):
    set_to_zero = jax.vmap(set_to_zero)
  proxy = set_to_zero(ones(arr.shape), bottom_ind)
  top_ind = lax.top_k(proxy, arr.shape[-1] - kth - 1)[1]
  out = lax.concatenate([bottom_ind, top_ind], dimension=arr.ndim - 1)
  return swapaxes(out, -1, axis)


@partial(jit, static_argnums=(2,))
def _roll_dynamic(a: Array, shift: Array, axis: Sequence[int]) -> Array:
  b_shape = lax.broadcast_shapes(shift.shape, np.shape(axis))
  if len(b_shape) != 1:
    msg = "'shift' and 'axis' arguments to roll must be scalars or 1D arrays"
    raise ValueError(msg)

  for x, i in zip(broadcast_to(shift, b_shape),
                  np.broadcast_to(axis, b_shape)):
    a_shape_i = array(a.shape[i], dtype=np.int32)
    x = ufuncs.remainder(lax.convert_element_type(x, np.int32),
                         lax.max(a_shape_i, np.int32(1)))
    a_concat = lax.concatenate((a, a), i)
    a = lax.dynamic_slice_in_dim(a_concat, a_shape_i - x, a.shape[i], axis=i)
  return a

@partial(jit, static_argnums=(1, 2))
def _roll_static(a: Array, shift: Sequence[int], axis: Sequence[int]) -> Array:
  for ax, s in zip(*np.broadcast_arrays(axis, shift)):
    if a.shape[ax] == 0:
      continue
    i = (-s) % a.shape[ax]
    a = lax.concatenate([lax.slice_in_dim(a, i, a.shape[ax], axis=ax),
                         lax.slice_in_dim(a, 0, i, axis=ax)],
                        dimension=ax)
  return a

@util.implements(np.roll)
def roll(a: ArrayLike, shift: ArrayLike | Sequence[int],
         axis: int | Sequence[int] | None = None) -> Array:
  util.check_arraylike("roll", a)
  arr = asarray(a)
  if axis is None:
    return roll(arr.ravel(), shift, 0).reshape(arr.shape)
  axis = _ensure_index_tuple(axis)
  axis = tuple(_canonicalize_axis(ax, arr.ndim) for ax in axis)
  try:
    shift = _ensure_index_tuple(shift)
  except TypeError:
    return _roll_dynamic(arr, asarray(shift), axis)
  else:
    return _roll_static(arr, shift, axis)


@partial(jit, static_argnames=('axis', 'start'))
def rollaxis(a: ArrayLike, axis: int, start: int = 0) -> Array:
  """Roll the specified axis to a given position.

  JAX implementation of :func:`numpy.rollaxis`.

  This function exists for compatibility with NumPy, but in most cases the newer
  :func:`jax.numpy.moveaxis` instead, because the meaning of its arguments is
  more intuitive.

  Args:
    a: input array.
    axis: index of the axis to roll forward.
    start: index toward which the axis will be rolled (default = 0). After
      normalizing negative axes, if ``start <= axis``, the axis is rolled to
      the ``start`` index; if ``start > axis``, the axis is rolled until the
      position before ``start``.

  Returns:
    Copy of ``a`` with rolled axis.

  Notes:
    Unlike :func:`numpy.rollaxis`, :func:`jax.numpy.rollaxis` will return a copy rather
    than a view of the input array. However, under JIT, the compiler will optimize away
    such copies when possible, so this doesn't have performance impacts in practice.

  See also:
    - :func:`jax.numpy.moveaxis`: newer API with clearer semantics than ``rollaxis``;
      this should be preferred to ``rollaxis`` in most cases.
    - :func:`jax.numpy.swapaxes`: swap two axes.
    - :func:`jax.numpy.transpose`: general permutation of axes.

  Examples:
    >>> a = jnp.ones((2, 3, 4, 5))

    Roll axis 2 to the start of the array:

    >>> jnp.rollaxis(a, 2).shape
    (4, 2, 3, 5)

    Roll axis 1 to the end of the array:

    >>> jnp.rollaxis(a, 1, a.ndim).shape
    (2, 4, 5, 3)

    Equivalent of these two with :func:`~jax.numpy.moveaxis`

    >>> jnp.moveaxis(a, 2, 0).shape
    (4, 2, 3, 5)
    >>> jnp.moveaxis(a, 1, -1).shape
    (2, 4, 5, 3)
  """
  util.check_arraylike("rollaxis", a)
  start = core.concrete_or_error(operator.index, start, "'start' argument of jnp.rollaxis()")
  a_ndim = ndim(a)
  axis = _canonicalize_axis(axis, a_ndim)
  if not (-a_ndim <= start <= a_ndim):
    raise ValueError(f"{start=} must satisfy {-a_ndim}<=start<={a_ndim}")
  if start < 0:
    start += a_ndim
  if start > axis:
    start -= 1
  return moveaxis(a, axis, start)


@util.implements(np.packbits)
@partial(jit, static_argnames=('axis', 'bitorder'))
def packbits(
    a: ArrayLike, axis: int | None = None, bitorder: str = "big"
) -> Array:
  util.check_arraylike("packbits", a)
  arr = asarray(a)
  if not (issubdtype(arr.dtype, integer) or issubdtype(arr.dtype, bool_)):
    raise TypeError('Expected an input array of integer or boolean data type')
  if bitorder not in ['little', 'big']:
    raise ValueError("'order' must be either 'little' or 'big'")
  arr = lax.gt(arr, _lax_const(a, 0)).astype('uint8')
  bits = arange(8, dtype='uint8')
  if bitorder == 'big':
    bits = bits[::-1]
  if axis is None:
    arr = ravel(arr)
    axis = 0
  arr = swapaxes(arr, axis, -1)

  remainder = arr.shape[-1] % 8
  if remainder:
    arr = lax.pad(arr, np.uint8(0),
                  (arr.ndim - 1) * [(0, 0, 0)] + [(0, 8 - remainder, 0)])

  arr = arr.reshape(arr.shape[:-1] + (arr.shape[-1] // 8, 8))
  bits = expand_dims(bits, tuple(range(arr.ndim - 1)))
  packed = (arr << bits).sum(-1).astype('uint8')
  return swapaxes(packed, axis, -1)


@util.implements(np.unpackbits)
@partial(jit, static_argnames=('axis', 'count', 'bitorder'))
def unpackbits(
    a: ArrayLike,
    axis: int | None = None,
    count: int | None = None,
    bitorder: str = "big",
) -> Array:
  util.check_arraylike("unpackbits", a)
  arr = asarray(a)
  if _dtype(a) != uint8:
    raise TypeError("Expected an input array of unsigned byte data type")
  if bitorder not in ['little', 'big']:
    raise ValueError("'order' must be either 'little' or 'big'")
  bits = asarray(1) << arange(8, dtype='uint8')
  if bitorder == 'big':
    bits = bits[::-1]
  if axis is None:
    arr = ravel(arr)
    axis = 0
  arr = swapaxes(arr, axis, -1)
  unpacked = ((arr[..., None] & expand_dims(bits, tuple(range(arr.ndim)))) > 0).astype('uint8')
  unpacked = unpacked.reshape(unpacked.shape[:-2] + (-1,))
  if count is not None:
    if count > unpacked.shape[-1]:
      unpacked = pad(unpacked, [(0, 0)] * (unpacked.ndim - 1) + [(0, count - unpacked.shape[-1])])
    else:
      unpacked = unpacked[..., :count]
  return swapaxes(unpacked, axis, -1)


def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis: int | None = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: StaticScalar | None = None,
) -> Array:
  """Take elements from an array.

  JAX implementation of :func:`numpy.take`, implemented in terms of
  :func:`jax.lax.gather`. JAX's behavior differs from NumPy in the case
  of out-of-bound indices; see the ``mode`` parameter below.

  Args:
    a: array from which to take values.
    indices: N-dimensional array of integer indices of values to take from the array.
    axis: the axis along which to take values. If not specified, the array will
      be flattened before indexing is applied.
    mode: Out-of-bounds indexing mode, either ``"fill"`` or ``"clip"``. The default
      ``mode="fill"`` returns invalid values (e.g. NaN) for out-of bounds indices;
      the ``fill_value`` argument gives control over this value. For more discussion
      of ``mode`` options, see :attr:`jax.numpy.ndarray.at`.
    fill_value: The fill value to return for out-of-bounds slices when mode is 'fill'.
      Ignored otherwise. Defaults to NaN for inexact types, the largest negative value for
      signed types, the largest positive value for unsigned types, and True for booleans.
    unique_indices: If True, the implementation will assume that the indices are unique,
      which can result in more efficient execution on some backends. If set to True and
      indices are not unique, the output is undefined.
    indices_are_sorted : If True, the implementation will assume that the indices are
      sorted in ascending order, which can lead to more efficient execution on some
      backends. If set to True and indices are not sorted, the output is undefined.

  Returns:
    Array of values extracted from ``a``.

  See also:
    - :attr:`jax.numpy.ndarray.at`: take values via indexing syntax.
    - :func:`jax.numpy.take_along_axis`: take values along an axis

  Examples:
    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 6.]])
    >>> indices = jnp.array([2, 0])

    Passing no axis results in indexing into the flattened array:

    >>> jnp.take(x, indices)
    Array([3., 1.], dtype=float32)
    >>> x.ravel()[indices]  # equivalent indexing syntax
    Array([3., 1.], dtype=float32)

    Passing an axis results ind applying the index to every subarray along the axis:

    >>> jnp.take(x, indices, axis=1)
    Array([[3., 1.],
           [6., 4.]], dtype=float32)
    >>> x[:, indices]  # equivalent indexing syntax
    Array([[3., 1.],
           [6., 4.]], dtype=float32)

    Out-of-bound indices fill with invalid values. For float inputs, this is `NaN`:

    >>> jnp.take(x, indices, axis=0)
    Array([[nan, nan, nan],
           [ 1.,  2.,  3.]], dtype=float32)
    >>> x.at[indices].get(mode='fill', fill_value=jnp.nan)  # equivalent indexing syntax
    Array([[nan, nan, nan],
           [ 1.,  2.,  3.]], dtype=float32)

    This default out-of-bound behavior can be adjusted using the ``mode`` parameter, for
    example, we can instead clip to the last valid value:

    >>> jnp.take(x, indices, axis=0, mode='clip')
    Array([[4., 5., 6.],
           [1., 2., 3.]], dtype=float32)
    >>> x.at[indices].get(mode='clip')  # equivalent indexing syntax
    Array([[4., 5., 6.],
           [1., 2., 3.]], dtype=float32)
  """
  return _take(a, indices, None if axis is None else operator.index(axis), out,
               mode, unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
               fill_value=fill_value)


@partial(jit, static_argnames=('axis', 'mode', 'unique_indices', 'indices_are_sorted', 'fill_value'))
def _take(a, indices, axis: int | None = None, out=None, mode=None,
          unique_indices=False, indices_are_sorted=False, fill_value=None):
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.take is not supported.")
  util.check_arraylike("take", a, indices)
  a = asarray(a)
  indices = asarray(indices)

  if axis is None:
    a = ravel(a)
    axis_idx = 0
  else:
    axis_idx = _canonicalize_axis(axis, ndim(a))

  if mode is None or mode == "fill":
    gather_mode = lax.GatherScatterMode.FILL_OR_DROP
    # lax.gather() does not support negative indices, so we wrap them here
    indices = where(indices < 0, indices + a.shape[axis_idx], indices)
  elif mode == "raise":
    # TODO(phawkins): we have no way to report out of bounds errors yet.
    raise NotImplementedError("The 'raise' mode to jnp.take is not supported.")
  elif mode == "wrap":
    indices = ufuncs.mod(indices, _lax_const(indices, a.shape[axis_idx]))
    gather_mode = lax.GatherScatterMode.PROMISE_IN_BOUNDS
  elif mode == "clip":
    gather_mode = lax.GatherScatterMode.CLIP
  else:
    raise ValueError(f"Invalid mode '{mode}' for np.take")

  index_dims = len(shape(indices))
  slice_sizes = list(shape(a))
  if slice_sizes[axis_idx] == 0:
    if indices.size != 0:
      raise IndexError("Cannot do a non-empty jnp.take() from an empty axis.")
    return a

  if indices.size == 0:
    out_shape = (slice_sizes[:axis_idx] + list(indices.shape) +
                 slice_sizes[axis_idx + 1:])
    return full_like(a, 0, shape=out_shape)

  slice_sizes[axis_idx] = 1
  dnums = lax.GatherDimensionNumbers(
    offset_dims=tuple(
      list(range(axis_idx)) +
      list(range(axis_idx + index_dims, len(a.shape) + index_dims - 1))),
    collapsed_slice_dims=(axis_idx,),
    start_index_map=(axis_idx,))
  return lax.gather(a, indices[..., None], dimension_numbers=dnums,
                    slice_sizes=tuple(slice_sizes),
                    mode=gather_mode, unique_indices=unique_indices,
                    indices_are_sorted=indices_are_sorted, fill_value=fill_value)


def _normalize_index(index, axis_size):
  """Normalizes an index value in the range [-N, N) to the range [0, N)."""
  if issubdtype(_dtype(index), np.unsignedinteger):
    return index
  if core.is_constant_dim(axis_size):
    axis_size_val = _lax_const(index, axis_size)
  else:
    axis_size_val = lax.convert_element_type(core.dimension_as_value(axis_size),
                                             _dtype(index))
  if isinstance(index, (int, np.integer)):
    return lax.add(index, axis_size_val) if index < 0 else index
  else:
    return lax.select(index < 0, lax.add(index, axis_size_val), index)


@partial(jit, static_argnames=('axis', 'mode', 'fill_value'))
def take_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    axis: int | None,
    mode: str | lax.GatherScatterMode | None = None,
    fill_value: StaticScalar | None = None,
) -> Array:
  """Take elements from an array.

  JAX implementation of :func:`numpy.take_along_axis`, implemented in
  terms of :func:`jax.lax.gather`. JAX's behavior differs from NumPy
  in the case of out-of-bound indices; see the ``mode`` parameter below.

  Args:
    a: array from which to take values.
    indices: array of integer indices. If ``axis`` is ``None``, must be one-dimensional.
      If ``axis`` is not None, must have ``a.ndim == indices.ndim``, and ``a`` must be
      broadcast-compatible with ``indices`` along dimensions other than ``axis``.
    axis: the axis along which to take values. If not specified, the array will
      be flattened before indexing is applied.
    mode: Out-of-bounds indexing mode, either ``"fill"`` or ``"clip"``. The default
      ``mode="fill"`` returns invalid values (e.g. NaN) for out-of bounds indices.
      For more discussion of ``mode`` options, see :attr:`jax.numpy.ndarray.at`.

  Returns:
    Array of values extracted from ``a``.

  See also:
    - :attr:`jax.numpy.ndarray.at`: take values via indexing syntax.
    - :func:`jax.numpy.take`: take the same indices along every axis slice.

  Examples:
    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 6.]])
    >>> indices = jnp.array([[0, 2],
    ...                      [1, 0]])
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[1., 3.],
           [5., 4.]], dtype=float32)
    >>> x[jnp.arange(2)[:, None], indices]  # equivalent via indexing syntax
    Array([[1., 3.],
           [5., 4.]], dtype=float32)

    Out-of-bound indices fill with invalid values. For float inputs, this is `NaN`:

    >>> indices = jnp.array([[1, 0, 2]])
    >>> jnp.take_along_axis(x, indices, axis=0)
    Array([[ 4.,  2., nan]], dtype=float32)
    >>> x.at[indices, jnp.arange(3)].get(
    ...     mode='fill', fill_value=jnp.nan)  # equivalent via indexing syntax
    Array([[ 4.,  2., nan]], dtype=float32)

    ``take_along_axis`` is helpful for extracting values from multi-dimensional
    argsorts and arg reductions. For, here we compute :func:`~jax.numpy.argsort`
    indices along an axis, and use ``take_along_axis`` to construct the sorted
    array:

    >>> x = jnp.array([[5, 3, 4],
    ...                [2, 7, 6]])
    >>> indices = jnp.argsort(x, axis=1)
    >>> indices
    Array([[1, 2, 0],
           [0, 2, 1]], dtype=int32)
    >>> jnp.take_along_axis(x, indices, axis=1)
    Array([[3, 4, 5],
           [2, 6, 7]], dtype=int32)

    Similarly, we can use :func:`~jax.numpy.argmin` with ``keepdims=True`` and
    use ``take_along_axis`` to extract the minimum value:

    >>> idx = jnp.argmin(x, axis=1, keepdims=True)
    >>> idx
    Array([[1],
           [0]], dtype=int32)
    >>> jnp.take_along_axis(x, idx, axis=1)
    Array([[3],
           [2]], dtype=int32)
  """
  util.check_arraylike("take_along_axis", arr, indices)
  a = asarray(arr)
  index_dtype = dtypes.dtype(indices)
  idx_shape = shape(indices)
  if not dtypes.issubdtype(index_dtype, integer):
    raise TypeError("take_along_axis indices must be of integer type, got "
                    f"{index_dtype}")
  if axis is None:
    if ndim(indices) != 1:
      msg = "take_along_axis indices must be 1D if axis=None, got shape {}"
      raise ValueError(msg.format(idx_shape))
    a = a.ravel()
    axis = 0
  rank = a.ndim
  if rank != ndim(indices):
    msg = "indices and arr must have the same number of dimensions; {} vs. {}"
    raise ValueError(msg.format(ndim(indices), a.ndim))
  axis_int = _canonicalize_axis(axis, rank)

  def replace(tup, val):
    lst = list(tup)
    lst[axis_int] = val
    return tuple(lst)

  use_64bit_index = any(not core.is_constant_dim(d) or d >= (1 << 31) for d in a.shape)
  index_dtype = dtype(int64 if use_64bit_index else int32)
  indices = lax.convert_element_type(indices, index_dtype)

  axis_size = a.shape[axis_int]
  arr_shape = replace(a.shape, 1)
  out_shape = lax.broadcast_shapes(idx_shape, arr_shape)
  if axis_size == 0:
    return zeros(out_shape, a.dtype)
  index_dims = [i for i, idx in enumerate(idx_shape) if i == axis_int or not core.definitely_equal(idx, 1)]

  gather_index_shape = tuple(np.array(out_shape)[index_dims]) + (1,)
  gather_indices = []
  slice_sizes = []
  offset_dims = []
  start_index_map = []
  collapsed_slice_dims = []
  j = 0
  for i in range(rank):
    if i == axis_int:
      indices = _normalize_index(indices, axis_size)
      gather_indices.append(lax.reshape(indices, gather_index_shape))
      slice_sizes.append(1)
      start_index_map.append(i)
      collapsed_slice_dims.append(i)
      j += 1
    elif core.definitely_equal(idx_shape[i], 1):
      # If idx_shape[i] == 1, we can just take the entirety of the arr's axis
      # and avoid forming an iota index.
      offset_dims.append(i)
      slice_sizes.append(arr_shape[i])
    elif core.definitely_equal(arr_shape[i], 1):
      # If the array dimension is 1 but the index dimension is not, we
      # broadcast the array dimension to the index dimension by repeatedly
      # gathering the first element.
      gather_indices.append(zeros(gather_index_shape, dtype=index_dtype))
      slice_sizes.append(1)
      start_index_map.append(i)
      collapsed_slice_dims.append(i)
      j += 1
    else:
      # Otherwise, idx_shape[i] == arr_shape[i]. Use an iota index so
      # corresponding elements of array and index are gathered.
      # TODO(mattjj): next line needs updating for dynamic shapes
      iota = lax.broadcasted_iota(index_dtype, gather_index_shape, j)
      gather_indices.append(iota)
      slice_sizes.append(1)
      start_index_map.append(i)
      collapsed_slice_dims.append(i)
      j += 1

  gather_indices_arr = lax.concatenate(gather_indices, dimension=j)
  dnums = lax.GatherDimensionNumbers(
    offset_dims=tuple(offset_dims),
    collapsed_slice_dims=tuple(collapsed_slice_dims),
    start_index_map=tuple(start_index_map))
  return lax.gather(a, gather_indices_arr, dnums, tuple(slice_sizes),
                    mode="fill" if mode is None else mode, fill_value=fill_value)


### Indexing

def _is_integer_index(idx: Any) -> bool:
  return isinstance(idx, (int, np.integer)) and not isinstance(idx, (bool, np.bool_))

def _is_simple_reverse_slice(idx: Any) -> bool:
  return (isinstance(idx, slice) and
          idx.start is idx.stop is None and
          isinstance(idx.step, int) and idx.step == -1)

def _is_valid_integer_index_for_slice(idx, size, mode):
  if size == 0:
    return False
  if _is_integer_index(idx):
    return -size <= idx < size
  try:
    shape, dtype = np.shape(idx), _dtype(idx)
  except:
    return False
  if shape == () and np.issubdtype(dtype, np.integer):
    # For dynamic integer indices, dynamic_slice semantics require index clipping:
    return mode in [None, 'promise_inbounds', 'clip']
  return False

def _is_contiguous_slice(idx):
  return (isinstance(idx, slice) and
          (idx.start is None or _is_integer_index(idx.start)) and
          (idx.stop is None or _is_integer_index(idx.stop)) and
          (idx.step is None or (_is_integer_index(idx.step) and idx.step == 1)))

def _attempt_rewriting_take_via_slice(arr: Array, idx: Any, mode: str | None) -> Array | None:
  # attempt to compute _rewriting_take via lax.slice(); return None if not possible.
  idx = idx if isinstance(idx, tuple) else (idx,)

  if not all(isinstance(i, int) for i in arr.shape):
    return None
  if len(idx) > arr.ndim:
    return None
  if any(i is None for i in idx):
    return None  # TODO(jakevdp): handle newaxis case
  # For symbolic dimensions fallback to gather
  if any(core.is_symbolic_dim(elt)
         for i in idx if isinstance(i, slice)
         for elt in (i.start, i.stop, i.step)):
    return None

  if any(i is Ellipsis for i in idx):
    # Remove ellipses and add trailing `slice(None)`.
    idx = _canonicalize_tuple_index(arr.ndim, idx=idx)

  simple_revs = {i for i, ind in enumerate(idx) if _is_simple_reverse_slice(ind)}
  int_indices = {i for i, (ind, size) in enumerate(zip(idx, arr.shape))
                 if _is_valid_integer_index_for_slice(ind, size, mode)}
  contiguous_slices = {i for i, ind in enumerate(idx) if _is_contiguous_slice(ind)}

  # For sharded inputs, indexing (like x[0]) and partial slices (like x[:2] as
  # opposed to x[:]) lead to incorrect sharding semantics when computed via
  # dynamic_slice, so we fall back to gather.
  # TODO(yashkatariya): fix dynamic_slice with sharding
  is_sharded = (isinstance(arr, ArrayImpl) and
                not dispatch.is_single_device_sharding(arr.sharding))
  has_partial_slices = any(idx[i].indices(arr.shape[i]) != (0, arr.shape[i], 1)
                           for i in contiguous_slices)
  if is_sharded and (int_indices or has_partial_slices):
    return None

  if len(simple_revs) + len(int_indices) + len(contiguous_slices) != len(idx):
    return None

  if simple_revs:
    arr = lax.rev(arr, tuple(simple_revs))
    idx = tuple(slice(None) if i in simple_revs else ind
                for i, ind in enumerate(idx))
    contiguous_slices |= simple_revs

  if not (int_indices or has_partial_slices):
    return arr

  idx += (arr.ndim - len(idx)) * (slice(None),)
  start_indices: Sequence[ArrayLike] = []
  slice_sizes: Sequence[int] = []

  for ind, size in safe_zip(idx, arr.shape):
    if isinstance(ind, slice):
      start, stop, step = ind.indices(size)
      assert step == 1  # checked above
      start_indices.append(start)
      slice_sizes.append(max(0, stop - start))
    else:
      assert np.issubdtype(_dtype(ind), np.integer)  # checked above
      assert np.shape(ind) == ()  # checked above
      start_indices.append(ind)
      slice_sizes.append(1)
  # Try to use static slicing when possible.
  if all(isinstance(i, (int, np.integer)) and i >= 0 for i in start_indices):
    int_start_indices = [int(i) for i in start_indices]  # type: ignore
    int_limit_indices = [i + s for i, s in zip(int_start_indices, slice_sizes)]
    arr = lax.slice(
        arr, start_indices=int_start_indices, limit_indices=int_limit_indices)
  else:
    # We must be careful with dtypes because dynamic_slice requires all
    # start indices to have matching types.
    if len(start_indices) > 1:
      start_indices = util.promote_dtypes(*start_indices)
    arr = lax.dynamic_slice(
        arr, start_indices=start_indices, slice_sizes=slice_sizes)
  if int_indices:
    arr = lax.squeeze(arr, tuple(int_indices))
  return arr


def _rewriting_take(arr, idx, indices_are_sorted=False, unique_indices=False,
                    mode=None, fill_value=None):
  # Computes arr[idx].
  # All supported cases of indexing can be implemented as an XLA gather,
  # followed by an optional reverse and broadcast_in_dim.

  # For simplicity of generated primitives, we call lax.dynamic_slice in the
  # simplest cases: i.e. non-dynamic arrays indexed with integers and slices.

  if (result := _attempt_rewriting_take_via_slice(arr, idx, mode)) is not None:
    return result

  # TODO(mattjj,dougalm): expand dynamic shape indexing support
  if config.dynamic_shapes.value and arr.ndim > 0:
    try: aval = core.get_aval(idx)
    except: pass
    else:
      if (isinstance(aval, core.DShapedArray) and aval.shape == () and
          dtypes.issubdtype(aval.dtype, np.integer) and
          not dtypes.issubdtype(aval.dtype, dtypes.bool_) and
          isinstance(arr.shape[0], int)):
        return lax.dynamic_index_in_dim(arr, idx, keepdims=False)

  treedef, static_idx, dynamic_idx = _split_index_for_jit(idx, arr.shape)
  return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
                 unique_indices, mode, fill_value)

# TODO(phawkins): re-enable jit after fixing excessive recompilation for
# slice indexes (e.g., slice(0, 5, None), slice(10, 15, None), etc.).
# @partial(jit, static_argnums=(1, 2))
def _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
            unique_indices, mode, fill_value):
  idx = _merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx)
  indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
  y = arr

  if fill_value is not None:
    core.concrete_or_error(None, fill_value,
                           "fill_value argument to indexed get()")
    if np.ndim(fill_value) != 0:
      raise ValueError("fill_value argument to indexed get() must be a scalar")
    if isinstance(fill_value, np.ndarray):
      fill_value = fill_value.item()

  if indexer.scalar_bool_dims:
    y = lax.expand_dims(y, indexer.scalar_bool_dims)

  # Avoid calling gather if the slice shape is empty, both as a fast path and to
  # handle cases like zeros(0)[array([], int32)].
  if core.is_empty_shape(indexer.slice_shape):
    return zeros_like(y, shape=indexer.slice_shape)

  # We avoid generating a gather when indexer.gather_indices.size is empty.
  if not core.is_empty_shape(indexer.gather_indices.shape):
    y = lax.gather(
      y, indexer.gather_indices, indexer.dnums, indexer.gather_slice_shape,
      unique_indices=unique_indices or indexer.unique_indices,
      indices_are_sorted=indices_are_sorted or indexer.indices_are_sorted,
      mode=mode, fill_value=fill_value)

  # Reverses axes with negative strides.
  if indexer.reversed_y_dims:
    y = lax.rev(y, indexer.reversed_y_dims)

  # This adds np.newaxis/None dimensions.
  return expand_dims(y, indexer.newaxis_dims)

class _Indexer(NamedTuple):
  # The expected shape of the slice output.
  slice_shape: Sequence[int]
  # The slice shape to pass to lax.gather().
  gather_slice_shape: Sequence[int]
  # The gather indices to use.
  gather_indices: ArrayLike
  # A GatherDimensionNumbers object describing the gather to perform.
  dnums: lax.GatherDimensionNumbers

  # Are the gather_indices known to be non-overlapping and/or sorted?
  # (In practice, these translate to "there no advanced indices", because
  # only advanced indices could lead to index repetition.)
  unique_indices: bool
  indices_are_sorted: bool

  # Slice dimensions that have negative strides, and so must be reversed after
  # the gather.
  reversed_y_dims: Sequence[int]

  # Keep track of any axes created by `newaxis`. These must be inserted for
  # gathers and eliminated for scatters.
  newaxis_dims: Sequence[int]

  # Keep track of dimensions with scalar bool indices. These must be inserted
  # for gathers before performing other index operations.
  scalar_bool_dims: Sequence[int]


def _split_index_for_jit(idx, shape):
  """Splits indices into necessarily-static and dynamic parts.

  Used to pass indices into `jit`-ted function.
  """
  # Convert list indices to tuples in cases (deprecated by NumPy.)
  idx = _eliminate_deprecated_list_indexing(idx)
  if any(isinstance(i, str) for i in idx):
    raise TypeError(f"JAX does not support string indexing; got {idx=}")

  # Expand any (concrete) boolean indices. We can then use advanced integer
  # indexing logic to handle them.
  idx = _expand_bool_indices(idx, shape)

  leaves, treedef = tree_flatten(idx)
  dynamic = [None] * len(leaves)
  static = [None] * len(leaves)
  for i, x in enumerate(leaves):
    if x is Ellipsis:
      static[i] = x
    elif isinstance(x, slice):
      # slice objects aren't hashable.
      static[i] = (x.start, x.stop, x.step)
    else:
      dynamic[i] = x
  return treedef, tuple(static), dynamic

def _merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx):
  """Recombines indices that were split by _split_index_for_jit."""
  idx = []
  for s, d in zip(static_idx, dynamic_idx):
    if d is not None:
      idx.append(d)
    elif isinstance(s, tuple):
      idx.append(slice(s[0], s[1], s[2]))
    else:
      idx.append(s)
  return treedef.unflatten(idx)

def _int(aval):
  return not aval.shape and issubdtype(aval.dtype, integer)

def _index_to_gather(x_shape: Sequence[int], idx: Sequence[Any],
                     normalize_indices: bool = True) -> _Indexer:
  # Remove ellipses and add trailing slice(None)s.
  idx = _canonicalize_tuple_index(len(x_shape), idx)

  # Check for scalar boolean indexing: this requires inserting extra dimensions
  # before performing the rest of the logic.
  scalar_bool_dims: Sequence[int] = [n for n, i in enumerate(idx) if isinstance(i, bool)]
  if scalar_bool_dims:
    idx = tuple(np.arange(int(i)) if isinstance(i, bool) else i for i in idx)
    x_shape = list(x_shape)
    for i in sorted(scalar_bool_dims):
      x_shape.insert(i, 1)
    x_shape = tuple(x_shape)

  # Check for advanced indexing:
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

  # Do the advanced indexing axes appear contiguously? If not, NumPy semantics
  # move the advanced axes to the front.
  advanced_axes_are_contiguous = False

  advanced_indexes: Sequence[Array | np.ndarray] | None = None

  # The positions of the advanced indexing axes in `idx`.
  idx_advanced_axes: Sequence[int] = []

  # The positions of the advanced indexes in x's shape.
  # collapsed, after None axes have been removed. See below.
  x_advanced_axes: Sequence[int] | None = None

  if _is_advanced_int_indexer(idx):
    idx_no_nones = [(i, d) for i, d in enumerate(idx) if d is not None]
    advanced_pairs = (
      (asarray(e), i, j) for j, (i, e) in enumerate(idx_no_nones)
      if isscalar(e) or isinstance(e, (Sequence, Array, np.ndarray)))
    if normalize_indices:
      advanced_pairs = ((_normalize_index(e, x_shape[j]), i, j)
                        for e, i, j in advanced_pairs)
    advanced_indexes, idx_advanced_axes, x_advanced_axes = zip(*advanced_pairs)
    advanced_axes_are_contiguous = bool(np.all(np.diff(idx_advanced_axes) == 1))

  x_axis = 0  # Current axis in x.
  y_axis = 0  # Current axis in y, before collapsing. See below.
  collapsed_y_axis = 0  # Current axis in y, after collapsing.

  # Scatter dimension numbers.
  offset_dims: Sequence[int] = []
  collapsed_slice_dims: Sequence[int] = []
  start_index_map: Sequence[int] = []

  use_64bit_index = (
    any(not core.is_constant_dim(d) or d >= (1 << 31) for d in x_shape) and
    config.enable_x64.value)
  index_dtype = int64 if use_64bit_index else int32

  # Gather indices.
  # Pairs of (array, start_dim) values. These will be broadcast into
  # gather_indices_shape, with the array dimensions aligned to start_dim, and
  # then concatenated.
  gather_indices: list[tuple[Array, int]] = []
  gather_indices_shape: list[int] = []

  # We perform three transformations to y before the scatter op, in order:
  # First, y is broadcast to slice_shape. In general `y` only need broadcast to
  # the right shape.
  slice_shape: Sequence[int] = []

  # Next, y is squeezed to remove newaxis_dims. This removes np.newaxis/`None`
  # indices, which the scatter cannot remove itself.
  newaxis_dims: Sequence[int] = []

  # Finally, we reverse reversed_y_dims to handle slices with negative strides.
  reversed_y_dims: Sequence[int] = []

  gather_slice_shape: Sequence[int] = []

  for idx_pos, i in enumerate(idx):
    # Handle the advanced indices here if:
    # * the advanced indices were not contiguous and we are the start.
    # * we are at the position of the first advanced index.
    if (advanced_indexes is not None and
        (advanced_axes_are_contiguous and idx_pos == idx_advanced_axes[0] or
         not advanced_axes_are_contiguous and idx_pos == 0)):
      advanced_indexes = broadcast_arrays(*advanced_indexes)
      shape = advanced_indexes[0].shape
      ndim = len(shape)

      start_dim = len(gather_indices_shape)
      gather_indices += ((lax.convert_element_type(a, index_dtype), start_dim)
                         for a in advanced_indexes)
      gather_indices_shape += shape

      start_index_map.extend(x_advanced_axes)
      collapsed_slice_dims.extend(x_advanced_axes)
      slice_shape.extend(shape)
      y_axis += ndim
      collapsed_y_axis += ndim

    # Per-index bookkeeping for advanced indexes.
    if idx_pos in idx_advanced_axes:
      x_axis += 1
      gather_slice_shape.append(1)
      continue

    try:
      abstract_i = core.get_aval(i)
    except TypeError:
      abstract_i = None
    # Handle basic int indexes.
    if isinstance(abstract_i, (ConcreteArray, ShapedArray)) and _int(abstract_i):
      if core.definitely_equal(x_shape[x_axis], 0):
        # XLA gives error when indexing into an axis of size 0
        raise IndexError(f"index is out of bounds for axis {x_axis} with size 0")
      i = _normalize_index(i, x_shape[x_axis]) if normalize_indices else i
      i_converted = lax.convert_element_type(i, index_dtype)
      gather_indices.append((i_converted, len(gather_indices_shape)))
      collapsed_slice_dims.append(x_axis)
      gather_slice_shape.append(1)
      start_index_map.append(x_axis)
      x_axis += 1
    # Handle np.newaxis (None)
    elif i is None:
      slice_shape.append(1)
      newaxis_dims.append(y_axis)
      y_axis += 1

    elif isinstance(i, slice):
      # Handle slice index (only static, otherwise an error is raised)
      if not all(_is_slice_element_none_or_constant_or_symbolic(elt)
                 for elt in (i.start, i.stop, i.step)):
        msg = ("Array slice indices must have static start/stop/step to be used "
               "with NumPy indexing syntax. "
               f"Found slice({i.start}, {i.stop}, {i.step}). "
               "To index a statically sized "
               "array at a dynamic position, try lax.dynamic_slice/"
               "dynamic_update_slice (JAX does not support dynamically sized "
               "arrays within JIT compiled functions).")
        raise IndexError(msg)

      start, step, slice_size = _preprocess_slice(i, x_shape[x_axis])
      slice_shape.append(slice_size)

      if core.definitely_equal(step, 1):
        # Avoid generating trivial gather (an optimization)
        if not core.definitely_equal(slice_size, x_shape[x_axis]):
          gather_indices.append((lax.convert_element_type(start, index_dtype),
                                len(gather_indices_shape)))
          start_index_map.append(x_axis)
        gather_slice_shape.append(slice_size)
        offset_dims.append(collapsed_y_axis)
      else:
        indices = (array(start, dtype=index_dtype) +
                   array(step, dtype=index_dtype) * lax.iota(index_dtype, slice_size))
        if step < 0:
          reversed_y_dims.append(collapsed_y_axis)
          indices = lax.rev(indices, dimensions=(0,))

        gather_slice_shape.append(1)
        gather_indices.append((indices, len(gather_indices_shape)))
        start_index_map.append(x_axis)
        gather_indices_shape.append(slice_size)
        collapsed_slice_dims.append(x_axis)

      collapsed_y_axis += 1
      y_axis += 1
      x_axis += 1
    else:
      if (abstract_i is not None and
          not (issubdtype(abstract_i.dtype, integer) or issubdtype(abstract_i.dtype, bool_))):
        msg = ("Indexer must have integer or boolean type, got indexer "
               "with type {} at position {}, indexer value {}")
        raise TypeError(msg.format(abstract_i.dtype.name, idx_pos, i))

      raise IndexError("Indexing mode not yet supported. Got unsupported indexer "
                      f"at position {idx_pos}: {i!r}")

  if len(gather_indices) == 0:
    gather_indices_array: ArrayLike = np.zeros((0,), dtype=index_dtype)
  elif len(gather_indices) == 1:
    g, _ = gather_indices[0]
    gather_indices_array = lax.expand_dims(g, (g.ndim,))
  else:
    last_dim = len(gather_indices_shape)
    gather_indices_shape.append(1)
    gather_indices_array = lax.concatenate([
      lax.broadcast_in_dim(g, gather_indices_shape, tuple(range(i, i + g.ndim)))
      for g, i in gather_indices],
      last_dim)

  dnums = lax.GatherDimensionNumbers(
    offset_dims = tuple(offset_dims),
    collapsed_slice_dims = tuple(sorted(collapsed_slice_dims)),
    start_index_map = tuple(start_index_map)
  )
  return _Indexer(
    slice_shape=slice_shape,
    newaxis_dims=tuple(newaxis_dims),
    gather_slice_shape=gather_slice_shape,
    reversed_y_dims=reversed_y_dims,
    dnums=dnums,
    gather_indices=gather_indices_array,
    unique_indices=advanced_indexes is None,
    indices_are_sorted=advanced_indexes is None,
    scalar_bool_dims=scalar_bool_dims)

def _should_unpack_list_index(x):
  """Helper for _eliminate_deprecated_list_indexing."""
  return (isinstance(x, (np.ndarray, Array)) and np.ndim(x) != 0
          or isinstance(x, (Sequence, slice))
          or x is Ellipsis or x is None)

def _eliminate_deprecated_list_indexing(idx):
  # "Basic slicing is initiated if the selection object is a non-array,
  # non-tuple sequence containing slice objects, [Ellipses, or newaxis
  # objects]". Detects this and raises a TypeError.
  if not isinstance(idx, tuple):
    if isinstance(idx, Sequence) and not isinstance(idx, (Array, np.ndarray, str)):
      # As of numpy 1.16, some non-tuple sequences of indices result in a warning, while
      # others are converted to arrays, based on a set of somewhat convoluted heuristics
      # (See https://github.com/numpy/numpy/blob/v1.19.2/numpy/core/src/multiarray/mapping.c#L179-L343)
      # In JAX, we raise an informative TypeError for *all* non-tuple sequences.
      if any(_should_unpack_list_index(i) for i in idx):
        msg = ("Using a non-tuple sequence for multidimensional indexing is not allowed; "
               "use `arr[tuple(seq)]` instead of `arr[seq]`. "
               "See https://github.com/google/jax/issues/4564 for more information.")
      else:
        msg = ("Using a non-tuple sequence for multidimensional indexing is not allowed; "
               "use `arr[array(seq)]` instead of `arr[seq]`. "
               "See https://github.com/google/jax/issues/4564 for more information.")
      raise TypeError(msg)
    else:
      idx = (idx,)
  return idx

def _is_boolean_index(i):
  try:
    abstract_i = core.get_aval(i)
  except TypeError:
    abstract_i = None
  return (isinstance(abstract_i, ShapedArray) and issubdtype(abstract_i.dtype, bool_)
          or isinstance(i, list) and i and all(_is_scalar(e)
          and issubdtype(_dtype(e), np.bool_) for e in i))

def _expand_bool_indices(idx, shape):
  """Converts concrete bool indexes into advanced integer indexes."""
  out = []
  total_dims = len(shape)
  num_ellipsis = sum(e is Ellipsis for e in idx)
  if num_ellipsis > 1:
    raise IndexError("an index can only have a single ellipsis ('...')")
  elif num_ellipsis == 1:
    total_dims = sum(_ndim(e) if _is_boolean_index(e) else 1 for e in idx
                     if e is not None and e is not Ellipsis)
  ellipsis_offset = 0
  newaxis_offset = 0
  for dim_number, i in enumerate(idx):
    try:
      abstract_i = core.get_aval(i)
    except TypeError:
      abstract_i = None
    if _is_boolean_index(i):
      if isinstance(i, list):
        i = array(i)
        abstract_i = core.get_aval(i)

      if not type(abstract_i) is ConcreteArray:
        # TODO(mattjj): improve this error by tracking _why_ the indices are not concrete
        raise errors.NonConcreteBooleanIndexError(abstract_i)
      elif _ndim(i) == 0:
        out.append(bool(i))
      else:
        i_shape = _shape(i)
        start = len(out) + ellipsis_offset - newaxis_offset
        expected_shape = shape[start: start + _ndim(i)]
        if i_shape != expected_shape:
          raise IndexError("boolean index did not match shape of indexed array in index "
                           f"{dim_number}: got {i_shape}, expected {expected_shape}")
        out.extend(np.where(i))
    else:
      out.append(i)
    if i is Ellipsis:
      ellipsis_offset = len(shape) - total_dims - 1
    if i is None:
      newaxis_offset += 1
  return tuple(out)


def _is_slice_element_none_or_constant_or_symbolic(elt):
  """Return True if elt is a constant or None."""
  if elt is None: return True
  if core.is_symbolic_dim(elt): return True
  try:
    return type(core.get_aval(elt)) is ConcreteArray
  except TypeError:
    return False

# TODO(mattjj): clean up this logic
def _is_advanced_int_indexer(idx):
  """Returns True if idx should trigger int array indexing, False otherwise."""
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
  assert isinstance(idx, tuple)
  if all(e is None or e is Ellipsis or isinstance(e, slice)
         or _is_scalar(e) and issubdtype(_dtype(e), np.integer) for e in idx):
    return False
  return all(e is None or e is Ellipsis or isinstance(e, slice)
             or _is_int_arraylike(e) for e in idx)

def _is_int_arraylike(x):
  """Returns True if x is array-like with integer dtype, False otherwise."""
  return (isinstance(x, int) and not isinstance(x, bool)
          or issubdtype(getattr(x, "dtype", None), np.integer)
          or isinstance(x, (list, tuple)) and all(_is_int_arraylike(e) for e in x))

def _is_scalar(x):
  """Checks if a Python or NumPy scalar."""
  return  np.isscalar(x) or (isinstance(x, (np.ndarray, Array))
                             and np.ndim(x) == 0)

def _canonicalize_tuple_index(arr_ndim, idx, array_name='array'):
  """Helper to remove Ellipsis and add in the implicit trailing slice(None)."""
  num_dimensions_consumed = sum(not (e is None or e is Ellipsis or isinstance(e, bool)) for e in idx)
  if num_dimensions_consumed > arr_ndim:
    raise IndexError(
        f"Too many indices for {array_name}: {num_dimensions_consumed} "
        f"non-None/Ellipsis indices for dim {arr_ndim}.")
  ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
  ellipsis_index = next(ellipses, None)
  if ellipsis_index is not None:
    if next(ellipses, None) is not None:
      raise IndexError(
          f"Multiple ellipses (...) not supported: {list(map(type, idx))}.")
    colons = (slice(None),) * (arr_ndim - num_dimensions_consumed)
    idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1:]
  elif num_dimensions_consumed < arr_ndim:
    colons = (slice(None),) * (arr_ndim - num_dimensions_consumed)
    idx = tuple(idx) + colons
  return idx

def _preprocess_slice(
    s: slice,
    axis_size: core.DimSize
  ) -> tuple[core.DimSize, core.DimSize, core.DimSize]:
  """Computes the start index, step, and size of the slice `x[s]`."""
  # See https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
  # "this is harder to get right than you may think"
  # (from https://github.com/python/cpython/blob/939fc6d6eab9b7ea8c244d513610dbdd556503a7/Objects/sliceobject.c#L275)
  def convert_to_index(d: DimSize) -> DimSize:
    # Convert np.array and jax.Array to int, leave symbolic dimensions alone
    try:
      return operator.index(d)
    except:
      return d

  # Must resolve statically if step is {<0, ==0, >0}
  step = convert_to_index(s.step) if s.step is not None else 1
  try:
    if step == 0:
      raise ValueError("slice step cannot be zero")
    step_gt_0 = (step > 0)
  except core.InconclusiveDimensionOperation as e:
    raise core.InconclusiveDimensionOperation(
        f"In slice with non-constant elements the step ({step}) must " +
        f"be resolved statically if it is > 0 or < 0.\nDetails: {e}")

  def clamp_index(i: DimSize, which: str):
    try:
      i_ge_0 = (i >= 0)
    except core.InconclusiveDimensionOperation as e:
      raise core.InconclusiveDimensionOperation(
          f"In slice with non-constant elements the {which} ({i}) must " +
          f"be resolved statically if it is >= 0.\nDetails: {e}")
    if i_ge_0:
      if step_gt_0:
        return core.min_dim(axis_size, i)
      else:
        return core.min_dim(axis_size - 1, i)
    else:
      if step_gt_0:
        return core.max_dim(0, axis_size + i)
      else:
        return core.max_dim(-1, axis_size + i)

  if s.start is None:
    start = 0 if step_gt_0 else axis_size - 1
  else:
    start = clamp_index(convert_to_index(s.start), "start")

  if s.stop is None:
    stop = axis_size if step_gt_0 else -1
  else:
    stop = clamp_index(convert_to_index(s.stop), "stop")

  gap = step if step_gt_0 else - step
  distance = (stop - start) if step_gt_0 else (start - stop)
  slice_size = core.max_dim(0, distance + gap - 1) // gap
  return start, step, slice_size


@util.implements(np.blackman)
def blackman(M: int) -> Array:
  M = core.concrete_or_error(int, M, "M argument of jnp.blackman")
  dtype = dtypes.canonicalize_dtype(float_)
  if M <= 1:
    return ones(M, dtype)
  n = lax.iota(dtype, M)
  return 0.42 - 0.5 * ufuncs.cos(2 * pi * n / (M - 1)) + 0.08 * ufuncs.cos(4 * pi * n / (M - 1))


@util.implements(np.bartlett)
def bartlett(M: int) -> Array:
  M = core.concrete_or_error(int, M, "M argument of jnp.bartlett")
  dtype = dtypes.canonicalize_dtype(float_)
  if M <= 1:
    return ones(M, dtype)
  n = lax.iota(dtype, M)
  return 1 - ufuncs.abs(2 * n + 1 - M) / (M - 1)


@util.implements(np.hamming)
def hamming(M: int) -> Array:
  M = core.concrete_or_error(int, M, "M argument of jnp.hamming")
  dtype = dtypes.canonicalize_dtype(float_)
  if M <= 1:
    return ones(M, dtype)
  n = lax.iota(dtype, M)
  return 0.54 - 0.46 * ufuncs.cos(2 * pi * n / (M - 1))


@util.implements(np.hanning)
def hanning(M: int) -> Array:
  M = core.concrete_or_error(int, M, "M argument of jnp.hanning")
  dtype = dtypes.canonicalize_dtype(float_)
  if M <= 1:
    return ones(M, dtype)
  n = lax.iota(dtype, M)
  return 0.5 * (1 - ufuncs.cos(2 * pi * n / (M - 1)))


@util.implements(np.kaiser)
def kaiser(M: int, beta: ArrayLike) -> Array:
  M = core.concrete_or_error(int, M, "M argument of jnp.kaiser")
  dtype = dtypes.canonicalize_dtype(float_)
  if M <= 1:
    return ones(M, dtype)
  n = lax.iota(dtype, M)
  alpha = 0.5 * (M - 1)
  return i0(beta * ufuncs.sqrt(1 - ((n - alpha) / alpha) ** 2)) / i0(beta)


def _gcd_cond_fn(xs: tuple[Array, Array]) -> Array:
  x1, x2 = xs
  return reductions.any(x2 != 0)

def _gcd_body_fn(xs: tuple[Array, Array]) -> tuple[Array, Array]:
  x1, x2 = xs
  x1, x2 = (where(x2 != 0, x2, x1),
            where(x2 != 0, lax.rem(x1, x2), _lax_const(x2, 0)))
  return (where(x1 < x2, x2, x1), where(x1 < x2, x1, x2))

@util.implements(np.gcd, module='numpy')
@jit
def gcd(x1: ArrayLike, x2: ArrayLike) -> Array:
  util.check_arraylike("gcd", x1, x2)
  x1, x2 = util.promote_dtypes(x1, x2)
  if not issubdtype(_dtype(x1), integer):
    raise ValueError("Arguments to jax.numpy.gcd must be integers.")
  x1, x2 = broadcast_arrays(x1, x2)
  gcd, _ = lax.while_loop(_gcd_cond_fn, _gcd_body_fn, (ufuncs.abs(x1), ufuncs.abs(x2)))
  return gcd


@util.implements(np.lcm, module='numpy')
@jit
def lcm(x1: ArrayLike, x2: ArrayLike) -> Array:
  util.check_arraylike("lcm", x1, x2)
  x1, x2 = util.promote_dtypes(x1, x2)
  x1, x2 = ufuncs.abs(x1), ufuncs.abs(x2)
  if not issubdtype(_dtype(x1), integer):
    raise ValueError("Arguments to jax.numpy.lcm must be integers.")
  d = gcd(x1, x2)
  return where(d == 0, _lax_const(d, 0),
               ufuncs.multiply(x1, ufuncs.floor_divide(x2, d)))


def extract(condition: ArrayLike, arr: ArrayLike,
            *, size: int | None = None, fill_value: ArrayLike = 0) -> Array:
  """Return the elements of an array that satisfy a condition.

  JAX implementation of :func:`numpy.extract`.

  Args:
    condition: array of conditions. Will be converted to boolean and flattened to 1D.
    arr: array of values to extract. Will be flattened to 1D.
    size: optional static size for output. Must be specified in order for ``extract``
      to be compatible with JAX transformations like :func:`~jax.jit` or :func:`~jax.vmap`.
    fill_value: if ``size`` is specified, fill padded entries with this value (default: 0).

  Returns:
    1D array of extracted entries . If ``size`` is specified, the result will have shape
    ``(size,)`` and be right-padded with ``fill_value``. If ``size`` is not specified,
    the output shape will depend on the number of True entries in ``condition``.

  Notes:
    This function does not require strict shape agreement between ``condition`` and ``arr``.
    If ``condition.size > arr.size``, then ``condition`` will be truncated, and if
    ``arr.size > condition.size``, then ``arr`` will be truncated.

  See also:
    :func:`jax.numpy.compress`: multi-dimensional version of ``extract``.

  Examples:
     Extract values from a 1D array:

     >>> x = jnp.array([1, 2, 3, 4, 5, 6])
     >>> mask = (x % 2 == 0)
     >>> jnp.extract(mask, x)
     Array([2, 4, 6], dtype=int32)

     In the simplest case, this is equivalent to boolean indexing:

     >>> x[mask]
     Array([2, 4, 6], dtype=int32)

     For use with JAX transformations, you can pass the ``size`` argument to
     specify a static shape for the output, along with an optional ``fill_value``
     that defaults to zero:

     >>> jnp.extract(mask, x, size=len(x), fill_value=0)
     Array([2, 4, 6, 0, 0, 0], dtype=int32)

     Notice that unlike with boolean indexing, ``extract`` does not require strict
     agreement between the sizes of the array and condition, and will effectively
     truncate both to the minimum size:

     >>> short_mask = jnp.array([False, True])
     >>> jnp.extract(short_mask, x)
     Array([2], dtype=int32)
     >>> long_mask = jnp.array([True, False, True, False, False, False, False, False])
     >>> jnp.extract(long_mask, x)
     Array([1, 3], dtype=int32)
  """
  util.check_arraylike("extreact", condition, arr, fill_value)
  return compress(ravel(condition), ravel(arr), size=size, fill_value=fill_value)


def compress(condition: ArrayLike, a: ArrayLike, axis: int | None = None,
             *, size: int | None = None, fill_value: ArrayLike = 0, out: None = None) -> Array:
  """Compress an array along a given axis using a boolean condition.

  JAX implementation of :func:`numpy.compress`.

  Args:
    condition: 1-dimensional array of conditions. Will be converted to boolean.
    a: N-dimensional array of values.
    axis: axis along which to compress. If None (default) then ``a`` will be
      flattened, and axis will be set to 0.
    size: optional static size for output. Must be specified in order for ``compress``
      to be compatible with JAX transformations like :func:`~jax.jit` or :func:`~jax.vmap`.
    fill_value: if ``size`` is specified, fill padded entries with this value (default: 0).
    out: not implemented by JAX.

  Returns:
    An array of dimension ``a.ndim``, compressed along the specified axis.

  See also:
    - :func:`jax.numpy.extract`: 1D version of ``compress``.
    - :meth:`jax.Array.compress`: equivalent functionality as an array method.

  Notes:
    This function does not require strict shape agreement between ``condition`` and ``a``.
    If ``condition.size > a.shape[axis]``, then ``condition`` will be truncated, and if
    ``a.shape[axis] > condition.size``, then ``a`` will be truncated.

  Examples:
    Compressing along the rows of a 2D array:

    >>> a = jnp.array([[1,  2,  3,  4],
    ...                [5,  6,  7,  8],
    ...                [9,  10, 11, 12]])
    >>> condition = jnp.array([True, False, True])
    >>> jnp.compress(condition, a, axis=0)
    Array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]], dtype=int32)

    For convenience, you can equivalently use the :meth:`~jax.Array.compress`
    method of JAX arrays:

    >>> a.compress(condition, axis=0)
    Array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]], dtype=int32)

    Note that the condition need not match the shape of the specified axis;
    here we compress the columns with the length-3 condition. Values beyond
    the size of the condition are ignored:

    >>> jnp.compress(condition, a, axis=1)
    Array([[ 1,  3],
           [ 5,  7],
           [ 9, 11]], dtype=int32)

    The optional ``size`` argument lets you specify a static output size so
    that the output is statically-shaped, and so this function can be used
    with transformations like :func:`~jax.jit` and :func:`~jax.vmap`:

    >>> f = lambda c, a: jnp.extract(c, a, size=len(a), fill_value=0)
    >>> mask = (a % 3 == 0)
    >>> jax.vmap(f)(mask, a)
    Array([[ 3,  0,  0,  0],
           [ 6,  0,  0,  0],
           [ 9, 12,  0,  0]], dtype=int32)
  """
  util.check_arraylike("compress", condition, a, fill_value)
  condition_arr = asarray(condition).astype(bool)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.compress is not supported.")
  if condition_arr.ndim != 1:
    raise ValueError("condition must be a 1D array")
  if axis is None:
    axis = 0
    arr = ravel(a)
  else:
    arr = moveaxis(a, axis, 0)
  condition_arr, extra = condition_arr[:arr.shape[0]], condition_arr[arr.shape[0]:]
  arr = arr[:condition_arr.shape[0]]

  if size is None:
    if reductions.any(extra):
      raise ValueError("condition contains entries that are out of bounds")
    result = arr[condition_arr]
  elif not 0 <= size <= arr.shape[0]:
    raise ValueError("size must be positive and not greater than the size of the array axis;"
                     f" got {size=} for a.shape[axis]={arr.shape[0]}")
  else:
    mask = expand_dims(condition_arr, range(1, arr.ndim))
    arr = where(mask, arr, array(fill_value, dtype=arr.dtype))
    result = arr[argsort(condition_arr, stable=True, descending=True)][:size]
  return moveaxis(result, 0, axis)


@util.implements(np.cov)
@partial(jit, static_argnames=('rowvar', 'bias', 'ddof'))
def cov(m: ArrayLike, y: ArrayLike | None = None, rowvar: bool = True,
        bias: bool = False, ddof: int | None = None,
        fweights: ArrayLike | None = None,
        aweights: ArrayLike | None = None) -> Array:
  if y is not None:
    m, y = util.promote_args_inexact("cov", m, y)
    if y.ndim > 2:
      raise ValueError("y has more than 2 dimensions")
  else:
    m, = util.promote_args_inexact("cov", m)

  if m.ndim > 2:
    raise ValueError("m has more than 2 dimensions")  # same as numpy error

  X = atleast_2d(m)
  if not rowvar and X.shape[0] != 1:
    X = X.T
  if X.shape[0] == 0:
    return array([]).reshape(0, 0)

  if y is not None:
    y_arr = atleast_2d(y)
    if not rowvar and y_arr.shape[0] != 1:
      y_arr = y_arr.T
    X = concatenate((X, y_arr), axis=0)
  if ddof is None:
    ddof = 1 if bias == 0 else 0

  w: Array | None = None
  if fweights is not None:
    util.check_arraylike("cov", fweights)
    if ndim(fweights) > 1:
      raise RuntimeError("cannot handle multidimensional fweights")
    if shape(fweights)[0] != X.shape[1]:
      raise RuntimeError("incompatible numbers of samples and fweights")
    if not issubdtype(_dtype(fweights), integer):
      raise TypeError("fweights must be integer.")
    # Ensure positive fweights; note that numpy raises an error on negative fweights.
    w = asarray(ufuncs.abs(fweights))
  if aweights is not None:
    util.check_arraylike("cov", aweights)
    if ndim(aweights) > 1:
      raise RuntimeError("cannot handle multidimensional aweights")
    if shape(aweights)[0] != X.shape[1]:
      raise RuntimeError("incompatible numbers of samples and aweights")
    # Ensure positive aweights: note that numpy raises an error for negative aweights.
    aweights = ufuncs.abs(aweights)
    w = asarray(aweights) if w is None else w * asarray(aweights)

  avg, w_sum = reductions.average(X, axis=1, weights=w, returned=True)
  w_sum = w_sum[0]

  if w is None:
    f = X.shape[1] - ddof
  elif ddof == 0:
    f = w_sum
  elif aweights is None:
    f = w_sum - ddof
  else:
    f = w_sum - ddof * reductions.sum(w * aweights) / w_sum

  X = X - avg[:, None]
  X_T = X.T if w is None else (X * lax.broadcast_to_rank(w, X.ndim)).T
  return ufuncs.true_divide(dot(X, X_T.conj()), f).squeeze()


@util.implements(np.corrcoef)
@partial(jit, static_argnames=('rowvar',))
def corrcoef(x: ArrayLike, y: ArrayLike | None = None, rowvar: bool = True) -> Array:
  util.check_arraylike("corrcoef", x)
  c = cov(x, y, rowvar)
  if len(shape(c)) == 0:
    # scalar - this should yield nan for values (nan/nan, inf/inf, 0/0), 1 otherwise
    return ufuncs.divide(c, c)
  d = diag(c)
  stddev = ufuncs.sqrt(ufuncs.real(d)).astype(c.dtype)
  c = c / stddev[:, None] / stddev[None, :]

  real_part = clip(ufuncs.real(c), -1, 1)
  if iscomplexobj(c):
    complex_part = clip(ufuncs.imag(c), -1, 1)
    c = lax.complex(real_part, complex_part)
  else:
    c = real_part
  return c


@partial(vectorize, excluded={0, 1, 3, 4})
def _searchsorted_via_scan(unrolled: bool, sorted_arr: Array, query: Array, side: str, dtype: type) -> Array:
  op = _sort_le_comparator if side == 'left' else _sort_lt_comparator
  def body_fun(state, _):
    low, high = state
    mid = (low + high) // 2
    go_left = op(query, sorted_arr[mid])
    return (where(go_left, low, mid), where(go_left, mid, high)), ()
  n_levels = int(np.ceil(np.log2(len(sorted_arr) + 1)))
  init = (dtype(0), dtype(len(sorted_arr)))
  carry, _ = lax.scan(body_fun, init, (), length=n_levels,
                      unroll=n_levels if unrolled else 1)
  return carry[1]


def _searchsorted_via_sort(sorted_arr: Array, query: Array, side: str, dtype: type) -> Array:
  working_dtype = int32 if sorted_arr.size + query.size < np.iinfo(np.int32).max else int64
  def _rank(x):
    idx = lax.iota(working_dtype, len(x))
    return zeros_like(idx).at[argsort(x)].set(idx)
  query_flat = query.ravel()
  if side == 'left':
    index = _rank(lax.concatenate([query_flat, sorted_arr], 0))[:query.size]
  else:
    index = _rank(lax.concatenate([sorted_arr, query_flat], 0))[sorted_arr.size:]
  return lax.reshape(lax.sub(index, _rank(query_flat)), np.shape(query)).astype(dtype)


def _searchsorted_via_compare_all(sorted_arr: Array, query: Array, side: str, dtype: type) -> Array:
  op = _sort_lt_comparator if side == 'left' else _sort_le_comparator
  comparisons = jax.vmap(op, in_axes=(0, None))(sorted_arr, query)
  return comparisons.sum(dtype=dtype, axis=0)


@partial(jit, static_argnames=('side', 'method'))
def searchsorted(a: ArrayLike, v: ArrayLike, side: str = 'left',
                 sorter: ArrayLike | None = None, *, method: str = 'scan') -> Array:
  """Perform a binary search within a sorted array.

  JAX implementation of :func:`numpy.searchsorted`.

  This will return the indices within a sorted array ``a`` where values in ``v``
  can be inserted to maintain its sort order.

  Args:
    a: one-dimensional array, assumed to be in sorted order unless ``sorter`` is specified.
    v: N-dimensional array of query values
    side: ``'left'`` (default) or ``'right'``; specifies whether insertion indices will be
      to the left or the right in case of ties.
    sorter: optional array of indices specifying the sort order of ``a``. If specified,
      then the algorithm assumes that ``a[sorter]`` is in sorted order.
    method: one of ``'scan'`` (default), ``'scan_unrolled'``, ``'sort'`` or ``'compare_all'``.
      See *Note* below.

  Returns:
    Array of insertion indices of shape ``v.shape``.

  Note:
    The ``method`` argument controls the algorithm used to compute the insertion indices.

    - ``'scan'`` (the default) tends to be more performant on CPU, particularly when ``a`` is
      very large.
    - ``'scan_unrolled'`` is more performant on GPU at the expense of additional compile time.
    - ``'sort'`` is often more performant on accelerator backends like GPU and TPU, particularly
      when ``v`` is very large.
    - ``'compare_all'`` tends to be the most performant when ``a`` is very small.

  Examples:
    Searching for a single value:

    >>> a = jnp.array([1, 2, 2, 3, 4, 5, 5])
    >>> jnp.searchsorted(a, 2)
    Array(1, dtype=int32)
    >>> jnp.searchsorted(a, 2, side='right')
    Array(3, dtype=int32)

    Searching for a batch of values:

    >>> vals = jnp.array([0, 3, 8, 1.5, 2])
    >>> jnp.searchsorted(a, vals)
    Array([0, 3, 7, 1, 1], dtype=int32)

    Optionally, the ``sorter`` argument can be used to find insertion indices into
    an array sorted via :func:`jax.numpy.argsort`:

    >>> a = jnp.array([4, 3, 5, 1, 2])
    >>> sorter = jnp.argsort(a)
    >>> jnp.searchsorted(a, vals, sorter=sorter)
    Array([0, 2, 5, 1, 1], dtype=int32)

    The result is equivalent to passing the sorted array:

    >>> jnp.searchsorted(jnp.sort(a), vals)
    Array([0, 2, 5, 1, 1], dtype=int32)
  """
  if sorter is None:
    util.check_arraylike("searchsorted", a, v)
  else:
    util.check_arraylike("searchsorted", a, v, sorter)
  if side not in ['left', 'right']:
    raise ValueError(f"{side!r} is an invalid value for keyword 'side'. "
                     "Expected one of ['left', 'right'].")
  if method not in ['scan', 'scan_unrolled', 'sort', 'compare_all']:
    raise ValueError(
        f"{method!r} is an invalid value for keyword 'method'. "
        "Expected one of ['sort', 'scan', 'scan_unrolled', 'compare_all'].")
  if ndim(a) != 1:
    raise ValueError("a should be 1-dimensional")
  a, v = util.promote_dtypes(a, v)
  if sorter is not None:
    a = a[sorter]
  dtype = int32 if len(a) <= np.iinfo(np.int32).max else int64
  if len(a) == 0:
    return zeros_like(v, dtype=dtype)
  impl = {
      'scan': partial(_searchsorted_via_scan, False),
      'scan_unrolled': partial(_searchsorted_via_scan, True),
      'sort': _searchsorted_via_sort,
      'compare_all': _searchsorted_via_compare_all,
  }[method]
  return impl(asarray(a), asarray(v), side, dtype)  # type: ignore

@util.implements(np.digitize)
@partial(jit, static_argnames=('right',))
def digitize(x: ArrayLike, bins: ArrayLike, right: bool = False) -> Array:
  util.check_arraylike("digitize", x, bins)
  right = core.concrete_or_error(bool, right, "right argument of jnp.digitize()")
  bins_arr = asarray(bins)
  if bins_arr.ndim != 1:
    raise ValueError(f"digitize: bins must be a 1-dimensional array; got {bins=}")
  if bins_arr.shape[0] == 0:
    return zeros_like(x, dtype=int32)
  side = 'right' if not right else 'left'
  return where(
    bins_arr[-1] >= bins_arr[0],
    searchsorted(bins_arr, x, side=side),
    len(bins_arr) - searchsorted(bins_arr[::-1], x, side=side)
  )

_PIECEWISE_DOC = """\
Unlike `np.piecewise`, :py:func:`jax.numpy.piecewise` requires functions in
`funclist` to be traceable by JAX, as it is implemented via :func:`jax.lax.switch`.
See the :func:`jax.lax.switch` documentation for more information.
"""

@util.implements(np.piecewise, lax_description=_PIECEWISE_DOC)
def piecewise(x: ArrayLike, condlist: Array | Sequence[ArrayLike],
              funclist: list[ArrayLike | Callable[..., Array]],
              *args, **kw) -> Array:
  util.check_arraylike("piecewise", x)
  nc, nf = len(condlist), len(funclist)
  if nf == nc + 1:
    funclist = funclist[-1:] + funclist[:-1]
  elif nf == nc:
    funclist = [0] + list(funclist)
  else:
    raise ValueError(f"with {nc} condition(s), either {nc} or {nc+1} functions are expected; got {nf}")
  consts = {i: c for i, c in enumerate(funclist) if not callable(c)}
  funcs = {i: f for i, f in enumerate(funclist) if callable(f)}
  return _piecewise(asarray(x), asarray(condlist, dtype=bool_), consts,
                    frozenset(funcs.items()),  # dict is not hashable.
                    *args, **kw)

@partial(jit, static_argnames=['funcs'])
def _piecewise(x: Array, condlist: Array, consts: dict[int, ArrayLike],
               funcs: frozenset[tuple[int, Callable[..., Array]]],
               *args, **kw) -> Array:
  funcdict = dict(funcs)
  funclist = [consts.get(i, funcdict.get(i)) for i in range(len(condlist) + 1)]
  indices = argmax(reductions.cumsum(concatenate([zeros_like(condlist[:1]), condlist], 0), 0), 0)
  dtype = _dtype(x)
  def _call(f):
    return lambda x: f(x, *args, **kw).astype(dtype)
  def _const(v):
    return lambda x: array(v, dtype=dtype)
  funclist = [_call(f) if callable(f) else _const(f) for f in funclist]
  return vectorize(lax.switch, excluded=(1,))(indices, funclist, x)


def _tile_to_size(arr: Array, size: int) -> Array:
  assert arr.ndim == 1
  if arr.size < size:
    arr = tile(arr, int(np.ceil(size / arr.size)))
  assert arr.size >= size
  return arr[:size] if arr.size > size else arr


def place(arr: ArrayLike, mask: ArrayLike, vals: ArrayLike, *,
          inplace: bool = True) -> Array:
  """Update array elements based on a mask.

  JAX implementation of :func:`numpy.place`.

  The semantics of :func:`numpy.place` are to modify arrays in-place, which
  is not possible for JAX's immutable arrays. The JAX version returns a modified
  copy of the input, and adds the ``inplace`` parameter which must be set to
  `False`` by the user as a reminder of this API difference.

  Args:
    arr: array into which values will be placed.
    mask: boolean mask with the same size as ``arr``.
    vals: values to be inserted into ``arr`` at the locations indicated
      by mask. If too many values are supplied, they will be truncated.
      If not enough values are supplied, they will be repeated.
    inplace: must be set to False to indicate that the input is not modified
      in-place, but rather a modified copy is returned.

  Returns:
    A copy of ``arr`` with masked values set to entries from `vals`.

  See Also:
    - :func:`jax.numpy.put`: put elements into an array at numerical indices.
    - :func:`jax.numpy.ndarray.at`: array updates using NumPy-style indexing

  Examples:
    >>> x = jnp.zeros((3, 5), dtype=int)
    >>> mask = (jnp.arange(x.size) % 3 == 0).reshape(x.shape)
    >>> mask
    Array([[ True, False, False,  True, False],
           [False,  True, False, False,  True],
           [False, False,  True, False, False]], dtype=bool)

    Placing a scalar value:

    >>> jnp.place(x, mask, 1, inplace=False)
    Array([[1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0]], dtype=int32)

    In this case, ``jnp.place`` is similar to the masked array update syntax:

    >>> x.at[mask].set(1)
    Array([[1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0]], dtype=int32)

    ``place`` differs when placing values from an array. The array is repeated
    to fill the masked entries:

    >>> vals = jnp.array([1, 3, 5])
    >>> jnp.place(x, mask, vals, inplace=False)
    Array([[1, 0, 0, 3, 0],
           [0, 5, 0, 0, 1],
           [0, 0, 3, 0, 0]], dtype=int32)
  """
  util.check_arraylike("place", arr, mask, vals)
  data, mask_arr, vals_arr = asarray(arr), asarray(mask), ravel(vals)
  if inplace:
    raise ValueError(
      "jax.numpy.place cannot modify arrays in-place, because JAX arrays are immutable. "
      "Pass inplace=False to instead return an updated array.")
  if data.size != mask_arr.size:
    raise ValueError("place: arr and mask must be the same size")
  if not vals_arr.size:
    raise ValueError("Cannot place values from an empty array")
  if not data.size:
    return data
  indices = where(mask_arr.ravel(), size=mask_arr.size, fill_value=mask_arr.size)[0]
  vals_arr = _tile_to_size(vals_arr, len(indices))
  return data.ravel().at[indices].set(vals_arr, mode='drop').reshape(data.shape)


def put(a: ArrayLike, ind: ArrayLike, v: ArrayLike,
        mode: str | None = None, *, inplace: bool = True) -> Array:
  """Put elements into an array at given indices.

  JAX implementation of :func:`numpy.put`.

  The semantics of :func:`numpy.put` are to modify arrays in-place, which
  is not possible for JAX's immutable arrays. The JAX version returns a modified
  copy of the input, and adds the ``inplace`` parameter which must be set to
  `False`` by the user as a reminder of this API difference.

  Args:
    a: array into which values will be placed.
    ind: array of indices over the flattened array at which to put values.
    v: array of values to put into the array.
    mode: string specifying how to handle out-of-bound indices. Supported values:

      - ``"clip"`` (default): clip out-of-bound indices to the final index.
      - ``"wrap"``: wrap out-of-bound indices to the beginning of the array.

    inplace: must be set to False to indicate that the input is not modified
      in-place, but rather a modified copy is returned.

  Returns:
    A copy of ``a`` with specified entries updated.

  See Also:
    - :func:`jax.numpy.place`: place elements into an array via boolean mask.
    - :func:`jax.numpy.ndarray.at`: array updates using NumPy-style indexing.
    - :func:`jax.numpy.take`: extract values from an array at given indices.

  Examples:
    >>> x = jnp.zeros(5, dtype=int)
    >>> indices = jnp.array([0, 2, 4])
    >>> values = jnp.array([10, 20, 30])
    >>> jnp.put(x, indices, values, inplace=False)
    Array([10,  0, 20,  0, 30], dtype=int32)

    This is equivalent to the following :attr:`jax.numpy.ndarray.at` indexing syntax:

    >>> x.at[indices].set(values)
    Array([10,  0, 20,  0, 30], dtype=int32)

    There are two modes for handling out-of-bound indices. By default they are
    clipped:

    >>> indices = jnp.array([0, 2, 6])
    >>> jnp.put(x, indices, values, inplace=False, mode='clip')
    Array([10,  0, 20,  0, 30], dtype=int32)

    Alternatively, they can be wrapped to the beginning of the array:

    >>> jnp.put(x, indices, values, inplace=False, mode='wrap')
    Array([10,  30, 20,  0, 0], dtype=int32)

    For N-dimensional inputs, the indices refer to the flattened array:

    >>> x = jnp.zeros((3, 5), dtype=int)
    >>> indices = jnp.array([0, 7, 14])
    >>> jnp.put(x, indices, values, inplace=False)
    Array([[10,  0,  0,  0,  0],
           [ 0,  0, 20,  0,  0],
           [ 0,  0,  0,  0, 30]], dtype=int32)
  """
  util.check_arraylike("put", a, ind, v)
  arr, ind_arr, v_arr = asarray(a), ravel(ind), ravel(v)
  if not arr.size or not ind_arr.size or not v_arr.size:
    return arr
  v_arr = _tile_to_size(v_arr, len(ind_arr))
  if inplace:
    raise ValueError(
      "jax.numpy.put cannot modify arrays in-place, because JAX arrays are immutable. "
      "Pass inplace=False to instead return an updated array.")
  if mode is None:
    scatter_mode = "drop"
  elif mode == "clip":
    ind_arr = clip(ind_arr, 0, arr.size - 1)
    scatter_mode = "promise_in_bounds"
  elif mode == "wrap":
    ind_arr = ind_arr % arr.size
    scatter_mode = "promise_in_bounds"
  elif mode == "raise":
    raise NotImplementedError("The 'raise' mode to jnp.put is not supported.")
  else:
    raise ValueError(f"mode should be one of 'wrap' or 'clip'; got {mode=}")
  return arr.at[unravel_index(ind_arr, arr.shape)].set(v_arr, mode=scatter_mode)
