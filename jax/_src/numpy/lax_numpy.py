# Copyright 2018 Google LLC
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

import builtins
import collections
from functools import partial
import operator
import types
from typing import Any, Sequence, FrozenSet, Optional, Tuple, Union
from textwrap import dedent as _dedent
import warnings

import numpy as np
import opt_einsum

import jax
from jax import jit
from jax import core
from jax import errors
from jax import lax
from jax.core import ShapedArray, DShapedArray, ConcreteArray, canonicalize_shape
from jax.interpreters import pxla
from jax.tree_util import tree_leaves, tree_flatten, tree_map

from jax._src import device_array
from jax._src import dtypes
from jax._src.api_util import _ensure_index_tuple
from jax._src.lax.lax import _array_copy, _sort_lt_comparator, _sort_le_comparator
from jax._src.lax import lax as lax_internal
from jax._src.numpy.ndarray import ndarray
from jax._src.numpy.reductions import (  # noqa: F401
  _ensure_optional_axes, _reduction_dims,
  alltrue, amin, amax, any, all, average, count_nonzero, cumsum, cumprod, cumproduct,
  max, mean, min, nancumsum, nancumprod, nanmax, nanmean, nanmin, nanprod, nanstd,
  nansum, nanvar, prod, product, ptp, sometrue, std, sum, var,
)
from jax._src.numpy.ufuncs import (  # noqa: F401
  abs, absolute, add, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
  bitwise_and, bitwise_not, bitwise_or, bitwise_xor, cbrt, ceil, conj, conjugate,
  copysign, cos, cosh, deg2rad, degrees, divide, divmod, equal, exp, exp2, expm1,
  fabs, float_power, floor, floor_divide, fmod, frexp, greater, greater_equal,
  heaviside, hypot, imag, invert, isfinite, isinf, isnan, isneginf, isposinf,
  ldexp, left_shift, less, less_equal, log, log10, log1p, log2, logaddexp, logaddexp2,
  logical_and, logical_not, logical_or, logical_xor, maximum, minimum, mod, modf,
  multiply, negative, nextafter, not_equal, positive, power, rad2deg, radians, real,
  reciprocal, remainder, right_shift, rint, sign, signbit, sin, sinc, sinh, sqrt,
  square, subtract, tan, tanh, true_divide)
from jax._src.numpy.util import (  # noqa: F401
  _arraylike, _broadcast_arrays, _broadcast_to, _check_arraylike, _complex_elem_type, _promote_args,
  _promote_args_inexact, _promote_dtypes, _promote_dtypes_inexact, _promote_shapes, _register_stackable,
  _stackable, _where, _wraps)
from jax._src.numpy.vectorize import vectorize
from jax._src.ops import scatter
from jax._src.util import (unzip2, prod as _prod, subvals, safe_zip, ceil_of_ratio,
                           canonicalize_axis as _canonicalize_axis)

newaxis = None

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

# We replace some builtin names to follow Numpy's API, so we capture here.
_abs = builtins.abs
_all = builtins.all
_any = builtins.any
_max = builtins.max
_min = builtins.min
_sum = builtins.sum
_divmod = builtins.divmod

# NumPy constants

pi = np.pi
e = np.e
euler_gamma = np.euler_gamma
inf = np.inf
NINF = np.NINF
PZERO = np.PZERO
NZERO = np.NZERO
nan = np.nan

# NumPy utility functions

get_printoptions = np.get_printoptions
printoptions = np.printoptions
set_printoptions = np.set_printoptions

iscomplexobj = np.iscomplexobj

shape = _shape = np.shape
ndim = _ndim = np.ndim
size = np.size
_dtype = partial(dtypes.dtype, canonicalize=True)

# At present JAX doesn't have a reason to distinguish between scalars and arrays
# in its object system. Further, we want JAX scalars to have the same type
# promotion behaviors as JAX arrays. Rather than introducing a new type of JAX
# scalar object with JAX promotion behaviors, instead we make the JAX scalar
# types return JAX arrays when instantiated.

class _ScalarMeta(type):
  def __hash__(self):
    return hash(self.dtype.type)

  def __eq__(self, other):
    return id(self) == id(other) or self.dtype.type == other

  def __ne__(self, other):
    return not (self == other)

  def __call__(self, x):
    return array(x, dtype=self.dtype)

  def __instancecheck__(self, instance):
    return isinstance(instance, self.dtype.type)

def _make_scalar_type(np_scalar_type):
  meta = _ScalarMeta(np_scalar_type.__name__, (object,),
                     {"dtype": np.dtype(np_scalar_type)})
  meta.__module__ = _PUBLIC_MODULE_NAME
  return meta

bool_ = _make_scalar_type(np.bool_)
uint8 = _make_scalar_type(np.uint8)
uint16 = _make_scalar_type(np.uint16)
uint32 = _make_scalar_type(np.uint32)
uint64 = _make_scalar_type(np.uint64)
int8 = _make_scalar_type(np.int8)
int16 = _make_scalar_type(np.int16)
int32 = _make_scalar_type(np.int32)
int64 = _make_scalar_type(np.int64)
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
issubsctype = dtypes.issubsctype
promote_types = dtypes.promote_types

ComplexWarning = np.ComplexWarning

array_str = np.array_str
array_repr = np.array_repr

save = np.save
savez = np.savez

@_wraps(np.dtype)
def _jnp_dtype(obj, align=False, copy=False):
  """Similar to np.dtype, but respects JAX dtype defaults."""
  if obj is None:
    obj = dtypes.float_
  elif isinstance(obj, type) and obj in dtypes.python_scalar_dtypes:
    obj = _DEFAULT_TYPEMAP[np.dtype(obj, align=align, copy=copy).type]
  return np.dtype(obj, align=align, copy=copy)

### utility functions

_DEFAULT_TYPEMAP = {
  np.bool_: bool_,
  np.int_: int_,
  np.float_: float_,
  np.complex_: complex_
}

_lax_const = lax_internal._const


def _result_dtype(op, *args):
  """Compute result dtype of applying op to arguments with given dtypes."""
  args = [np.ones((0,) * ndim(arg), _dtype(arg)) for arg in args]
  return _dtype(op(*args))


def _convert_and_clip_integer(val, dtype):
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
  DeviceArray(-1, dtype=int32)

  This function clips to the values representable in the new type:

  >>> _convert_and_clip_integer(val, 'int32')
  DeviceArray(2147483647, dtype=int32)
  """
  val = val if isinstance(val, ndarray) else asarray(val)
  dtype = dtypes.canonicalize_dtype(dtype)
  if not (issubdtype(dtype, integer) and issubdtype(val.dtype, integer)):
    raise TypeError("_convert_and_clip_integer only accepts integer dtypes.")

  val_dtype = dtypes.canonicalize_dtype(val.dtype)
  if val_dtype != val.dtype:
    # TODO(jakevdp): this is a weird corner case; need to figure out how to handle it.
    # This happens in X32 mode and can either come from a jax value created in another
    # context, or a Python integer converted to int64.
    pass
  min_val = _lax_const(val, _max(iinfo(dtype).min, iinfo(val_dtype).min))
  max_val = _lax_const(val, _min(iinfo(dtype).max, iinfo(val_dtype).max))
  return clip(val, min_val, max_val).astype(dtype)


@_wraps(np.load, update_doc=False)
def load(*args, **kwargs):
  # The main purpose of this wrapper is to recover bfloat16 data types.
  # Note: this will only work for files created via np.save(), not np.savez().
  out = np.load(*args, **kwargs)
  if isinstance(out, np.ndarray):
    # numpy does not recognize bfloat16, so arrays are serialized as void16
    if out.dtype == 'V2':
      out = out.view(bfloat16)
    try:
      out = asarray(out)
    except TypeError:  # Unsupported dtype
      pass
  return out

### implementations of numpy functions in terms of lax

@_wraps(np.fmin)
@jit
def fmin(x1, x2):
  return where((x1 < x2) | isnan(x2), x1, x2)

@_wraps(np.fmax)
@jit
def fmax(x1, x2):
  return where((x1 > x2) | isnan(x2), x1, x2)

@_wraps(np.issubdtype)
def issubdtype(arg1, arg2):
  return dtypes.issubdtype(arg1, arg2)

@_wraps(np.isscalar)
def isscalar(element):
  if hasattr(element, '__jax_array__'):
    element = element.__jax_array__()
  return dtypes.is_python_scalar(element) or np.isscalar(element)

iterable = np.iterable

@_wraps(np.result_type)
def result_type(*args):
  return dtypes.result_type(*args)


@_wraps(np.trapz)
@partial(jit, static_argnames=('axis',))
def trapz(y, x=None, dx=1.0, axis: int = -1):
  _check_arraylike('trapz', y)
  y = moveaxis(y, axis, -1)
  if x is not None:
    if ndim(x) == 1:
      dx = diff(x)
    else:
      dx = moveaxis(diff(x, axis=axis), axis, -1)
  return 0.5 * (dx * (y[..., 1:] + y[..., :-1])).sum(-1)


@_wraps(np.trunc)
@jit
def trunc(x):
  _check_arraylike('trunc', x)
  return where(lax.lt(x, _lax_const(x, 0)), ceil(x), floor(x))


@partial(jit, static_argnums=(2, 3, 4))
def _conv(x, y, mode, op, precision):
  if ndim(x) != 1 or ndim(y) != 1:
    raise ValueError(f"{op}() only support 1-dimensional inputs.")
  x, y = _promote_dtypes_inexact(x, y)
  if len(x) == 0 or len(y) == 0:
    raise ValueError(f"{op}: inputs cannot be empty, got shapes {x.shape} and {y.shape}.")

  out_order = slice(None)
  if op == 'correlate':
    y = conj(y)
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
                                    padding, precision=precision)
  return result[0, 0, out_order]


@_wraps(np.convolve, lax_description=_PRECISION_DOC)
@partial(jit, static_argnames=('mode', 'precision'))
def convolve(a, v, mode='full', *, precision=None):
  _check_arraylike("convolve", a, v)
  return _conv(a, v, mode, 'convolve', precision)


@_wraps(np.correlate, lax_description=_PRECISION_DOC)
@partial(jit, static_argnames=('mode', 'precision'))
def correlate(a, v, mode='valid', *, precision=None):
  _check_arraylike("correlate", a, v)
  return _conv(a, v, mode, 'correlate', precision)


@_wraps(np.histogram_bin_edges)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
  if isinstance(bins, str):
    raise NotImplementedError("string values for `bins` not implemented.")
  _check_arraylike("histogram_bin_edges", a, bins)
  a = ravel(a)
  b = asarray(bins)
  if b.ndim == 1:
    return b
  if range is None:
    range = [a.min(), a.max()]
  assert len(range) == 2
  range = asarray(range)
  range = (where(ptp(range) == 0, range[0] - 0.5, range[0]),
           where(ptp(range) == 0, range[1] + 0.5, range[1]))
  dtype = _dtype(a)
  if issubdtype(dtype, integer):
    dtype = promote_types(dtype, float32)
  return linspace(range[0], range[1], bins + 1, dtype=dtype)


@_wraps(np.histogram)
def histogram(a, bins=10, range=None, weights=None, density=None):
  _check_arraylike("histogram", a, bins)
  if weights is not None and a.shape != weights.shape:
    raise ValueError("weights should have the same shape as a.")
  a = ravel(a)
  if weights is not None:
    weights = ravel(weights)
  else:
    weights = ones_like(a)
  bin_edges = histogram_bin_edges(a, bins, range, weights)
  bin_idx = searchsorted(bin_edges, a, side='right')
  bin_idx = where(a == bin_edges[-1], len(bin_edges) - 1, bin_idx)
  counts = bincount(bin_idx, weights, length=len(bin_edges))[1:]
  if density:
    bin_widths = diff(bin_edges)
    counts = counts / bin_widths / counts.sum()
  return counts, bin_edges

@_wraps(np.histogram2d)
def histogram2d(x, y, bins=10, range=None, weights=None, density=None):
  _check_arraylike("histogram2d", x, y)
  try:
    N = len(bins)
  except TypeError:
    N = 1

  if N != 1 and N != 2:
    x_edges = y_edges = asarray(bins)
    bins = [x_edges, y_edges]

  sample = transpose(asarray([x, y]))
  hist, edges = histogramdd(sample, bins, range, weights, density)
  return hist, edges[0], edges[1]

@_wraps(np.histogramdd)
def histogramdd(sample, bins=10, range=None, weights=None, density=None):
  _check_arraylike("histogramdd", sample)
  N, D = shape(sample)

  if weights is not None and weights.shape != (N,):
    raise ValueError("should have one weight for each sample.")

  if range is not None and (
      len(range) != D or _any(r is not None and len(r) != 2 for r in range)):
    raise ValueError(f"For sample.shape={(N, D)}, range must be a sequence "
                     f"of {D} pairs or Nones; got range={range}")

  try:
    num_bins = len(bins)
    if num_bins != D:
      raise ValueError("should be a bin for each dimension.")
  except TypeError:
    # when bin_size is integer, the same bin is used for each dimension
    bins = D * [bins]

  bin_idx_by_dim = D*[None]
  nbins = np.empty(D, int)
  bin_edges_by_dim = D*[None]
  dedges = D*[None]

  for i in builtins.range(D):
    range_i = None if range is None else range[i]
    bin_edges = histogram_bin_edges(sample[:, i], bins[i], range_i, weights)
    bin_idx = searchsorted(bin_edges, sample[:, i], side='right')
    bin_idx = where(sample[:, i] == bin_edges[-1], bin_idx - 1, bin_idx)
    bin_idx_by_dim[i] = bin_idx
    nbins[i] = len(bin_edges) + 1
    bin_edges_by_dim[i] = bin_edges
    dedges[i] = diff(bin_edges_by_dim[i])

  xy = ravel_multi_index(bin_idx_by_dim, nbins, mode='clip')
  hist = bincount(xy, weights, length=nbins.prod())
  hist = reshape(hist, nbins)
  core = D*(slice(1, -1),)
  hist = hist[core]

  if density:
    hist /= hist.sum()
    for norm in ix_(*dedges):
      hist /= norm

  return hist, bin_edges_by_dim


_ARRAY_VIEW_DOC = """
The JAX version of this function may in some cases return a copy rather than a
view of the input.
"""

@_wraps(np.transpose, lax_description=_ARRAY_VIEW_DOC)
def transpose(a, axes=None):
  _check_arraylike("transpose", a)
  axes = np.arange(ndim(a))[::-1] if axes is None else axes
  return lax.transpose(a, axes)


@_wraps(np.rot90, lax_description=_ARRAY_VIEW_DOC)
@partial(jit, static_argnames=('k', 'axes'))
def rot90(m, k=1, axes=(0, 1)):
  _check_arraylike("rot90", m)
  ax1, ax2 = axes
  ax1 = _canonicalize_axis(ax1, ndim(m))
  ax2 = _canonicalize_axis(ax2, ndim(m))
  if ax1 == ax2:
    raise ValueError("Axes must be different")  # same as numpy error
  k = k % 4
  if k == 0:
    return m
  elif k == 2:
    return flip(flip(m, ax1), ax2)
  else:
    perm = list(range(m.ndim))
    perm[ax1], perm[ax2] = perm[ax2], perm[ax1]
    if k == 1:
      return transpose(flip(m, ax2), perm)
    else:
      return flip(transpose(m, perm), ax2)


@_wraps(np.flip, lax_description=_ARRAY_VIEW_DOC)
def flip(m, axis: Optional[Union[int, Tuple[int, ...]]] = None):
  return _flip(m, _ensure_optional_axes(axis))

@partial(jit, static_argnames=('axis',))
def _flip(m, axis: Optional[Union[int, Tuple[int, ...]]] = None):
  _check_arraylike("flip", m)
  if axis is None:
    return lax.rev(m, list(range(len(shape(m)))))
  axis = _ensure_index_tuple(axis)
  return lax.rev(m, [_canonicalize_axis(ax, ndim(m)) for ax in axis])


@_wraps(np.fliplr, lax_description=_ARRAY_VIEW_DOC)
def fliplr(m):
  return _flip(m, 1)


@_wraps(np.flipud, lax_description=_ARRAY_VIEW_DOC)
def flipud(m):
  return _flip(m, 0)

@_wraps(np.iscomplex)
@jit
def iscomplex(x):
  i = imag(x)
  return lax.ne(i, _lax_const(i, 0))

@_wraps(np.isreal)
@jit
def isreal(x):
  i = imag(x)
  return lax.eq(i, _lax_const(i, 0))

@_wraps(np.angle)
@partial(jit, static_argnames=['deg'])
def angle(z, deg=False):
  re = real(z)
  im = imag(z)
  dtype = _dtype(re)
  if not issubdtype(dtype, inexact) or (
      issubdtype(_dtype(z), floating) and ndim(z) == 0):
    dtype = dtypes.canonicalize_dtype(float_)
    re = lax.convert_element_type(re, dtype)
    im = lax.convert_element_type(im, dtype)
  result = lax.atan2(im, re)
  return degrees(result) if deg else result


@_wraps(np.diff)
@partial(jit, static_argnames=('n', 'axis'))
def diff(a, n=1, axis: int = -1, prepend=None, append=None):
  _check_arraylike("diff", a)
  n = core.concrete_or_error(operator.index, n, "'n' argument of jnp.diff")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.diff")
  if n == 0:
    return a
  if n < 0:
    raise ValueError(f"order must be non-negative but got {n}")
  if ndim(a) == 0:
    raise ValueError(f"diff requires input that is at least one dimensional; got {a}")

  nd = a.ndim
  axis = _canonicalize_axis(axis, nd)

  combined = []
  if prepend is not None:
    _check_arraylike("diff", prepend)
    if isscalar(prepend):
      shape = list(a.shape)
      shape[axis] = 1
      prepend = broadcast_to(prepend, tuple(shape))
    combined.append(prepend)

  combined.append(a)

  if append is not None:
    _check_arraylike("diff", append)
    if isscalar(append):
      shape = list(a.shape)
      shape[axis] = 1
      append = broadcast_to(append, tuple(shape))
    combined.append(append)

  if len(combined) > 1:
    a = concatenate(combined, axis)

  slice1 = [slice(None)] * nd
  slice2 = [slice(None)] * nd
  slice1[axis] = slice(1, None)
  slice2[axis] = slice(None, -1)
  slice1_tuple = tuple(slice1)
  slice2_tuple = tuple(slice2)

  op = not_equal if a.dtype == np.bool_ else subtract
  for _ in range(n):
    a = op(a[slice1_tuple], a[slice2_tuple])

  return a

_EDIFF1D_DOC = """\
Unlike NumPy's implementation of ediff1d, :py:func:`jax.numpy.ediff1d` will not
issue an error if casting ``to_end`` or ``to_begin`` to the type of ``ary``
loses precision.
"""

@_wraps(np.ediff1d, lax_description=_EDIFF1D_DOC)
@jit
def ediff1d(ary, to_end=None, to_begin=None):
  _check_arraylike("ediff1d", ary)
  ary = ravel(ary)
  result = lax.sub(ary[1:], ary[:-1])
  if to_begin is not None:
    _check_arraylike("ediff1d", to_begin)
    result = concatenate((ravel(asarray(to_begin, dtype=ary.dtype)), result))
  if to_end is not None:
    _check_arraylike("ediff1d", to_end)
    result = concatenate((result, ravel(asarray(to_end, dtype=ary.dtype))))
  return result


@_wraps(np.gradient, skip_params=['edge_order'])
@partial(jit, static_argnames=('axis', 'edge_order'))
def gradient(f, *varargs, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             edge_order=None):
  if edge_order is not None:
    raise NotImplementedError("The 'edge_order' argument to jnp.gradient is not supported.")

  def gradient_along_axis(a, h, axis):
    sliced = partial(lax.slice_in_dim, a, axis=axis)
    a_grad = concatenate((
      (sliced(1, 2) - sliced(0, 1)),  # upper edge
      (sliced(2, None) - sliced(None, -2)) * 0.5,  # inner
      (sliced(-1, None) - sliced(-2, -1)),  # lower edge
    ), axis)
    return a_grad / h

  a = f
  axis_tuple: Tuple[int, ...]
  if axis is None:
    axis_tuple = tuple(range(a.ndim))
  else:
    if isinstance(axis, int):
      axis = (axis,)
    elif not isinstance(axis, tuple) and not isinstance(axis, list):
      raise ValueError("Give `axis` either as int or iterable")
    elif len(axis) == 0:
      return []
    axis_tuple = tuple(_canonicalize_axis(i, a.ndim) for i in axis)

  if _min([s for i, s in enumerate(a.shape) if i in axis_tuple]) < 2:
    raise ValueError("Shape of array too small to calculate "
                     "a numerical gradient, "
                     "at least 2 elements are required.")
  len_axes = len(axis_tuple)
  n = len(varargs)
  if n == 0 or varargs is None:
    # no spacing
    dx = [1.0] * len_axes
  elif n == 1:
    # single value for all axes
    dx = list(varargs) * len_axes
  elif n == len_axes:
    dx = list(varargs)
  else:
    TypeError("Invalid number of spacing arguments %d" % n)

  if ndim(dx[0]) != 0:
    raise NotImplementedError("Non-constant spacing not implemented")

  # TODO: use jax.lax loop tools if possible
  a_grad = [gradient_along_axis(a, h, ax) for ax, h in zip(axis_tuple, dx)]

  if len(axis_tuple) == 1:
    a_grad = a_grad[0]

  return a_grad


@_wraps(np.isrealobj)
def isrealobj(x):
  return not iscomplexobj(x)



@_wraps(np.reshape, lax_description=_ARRAY_VIEW_DOC)
def reshape(a, newshape, order="C"):
  _stackable(a) or _check_arraylike("reshape", a)
  try:
    return a.reshape(newshape, order=order)  # forward to method for ndarrays
  except AttributeError:
    return _reshape(a, newshape, order=order)

def _compute_newshape(a, newshape):
  """Fixes a -1 value in newshape, if present."""
  # other errors, like having more than one -1, are caught downstream, in
  # reshape_shape_rule.
  try: iter(newshape)
  except: iterable = False
  else: iterable = True
  newshape = core.canonicalize_shape(newshape if iterable else [newshape])
  return tuple(- core.divide_shape_sizes(np.shape(a), newshape)
               if core.symbolic_equal_dim(d, -1) else d
               for d in newshape)


def _reshape(a, *args, order="C"):
  newshape = _compute_newshape(a, args[0] if len(args) == 1 else args)
  if order == "C":
    return lax.reshape(a, newshape, None)
  elif order == "F":
    dims = np.arange(ndim(a))[::-1]
    return lax.reshape(a, newshape[::-1], dims).T
  elif order == "A":
    raise NotImplementedError("np.reshape order=A is not implemented.")
  else:
    raise ValueError("Unexpected value for 'order' argument: {}.".format(order))

def _transpose(a, *args):
  if not args:
    axis = None
  elif len(args) == 1:
    axis = args[0] if args[0] is None else _ensure_index_tuple(args[0])
  else:
    axis = _ensure_index_tuple(args)
  return transpose(a, axis)

@_wraps(np.ravel, lax_description=_ARRAY_VIEW_DOC)
@partial(jit, static_argnames=('order',), inline=True)
def ravel(a, order="C"):
  _stackable(a) or _check_arraylike("ravel", a)
  if order == "K":
    raise NotImplementedError("Ravel not implemented for order='K'.")
  return reshape(a, (size(a),), order)


@_wraps(np.ravel_multi_index)
def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
  assert len(multi_index) == len(dims), f"len(multi_index)={len(multi_index)} != len(dims)={len(dims)}"
  dims = tuple(core.concrete_or_error(int, d, "in `dims` argument of ravel_multi_index().") for d in dims)
  _check_arraylike("ravel_multi_index", *multi_index)
  for index in multi_index:
    if mode == 'raise':
      core.concrete_or_error(array, index,
        "The error occurred because ravel_multi_index was jit-compiled"
        " with mode='raise'. Use mode='wrap' or mode='clip' instead.")
    if not issubdtype(_dtype(index), integer):
      raise TypeError("only int indices permitted")
  if mode == "raise":
    if _any(any((i < 0) | (i >= d)) for i, d in zip(multi_index, dims)):
      raise ValueError("invalid entry in coordinates array")
  elif mode == "clip":
    multi_index = [clip(i, 0, d - 1) for i, d in zip(multi_index, dims)]
  elif mode == "wrap":
    multi_index = [i % d for i, d in zip(multi_index, dims)]
  else:
    raise ValueError(f"invalid mode={mode!r}. Expected 'raise', 'wrap', or 'clip'")

  if order == "F":
    strides = np.cumprod((1,) + dims[:-1])
  elif order == "C":
    strides = np.cumprod((1,) + dims[1:][::-1])[::-1]
  else:
    raise ValueError(f"invalid order={order!r}. Expected 'C' or 'F'")

  result = array(0, dtype=dtypes.canonicalize_dtype(int_))
  for i, s in zip(multi_index, strides):
    result = result + i * s
  return result


_UNRAVEL_INDEX_DOC = """\
Unlike numpy's implementation of unravel_index, negative indices are accepted
and out-of-bounds indices are clipped.
"""

@_wraps(np.unravel_index, lax_description=_UNRAVEL_INDEX_DOC)
def unravel_index(indices, shape):
  _check_arraylike("unravel_index", indices)
  sizes = append(array(shape), 1)
  cumulative_sizes = cumprod(sizes[::-1])[::-1]
  total_size = cumulative_sizes[0]
  # Clip so raveling and unraveling an oob index will not change the behavior
  clipped_indices = clip(indices, -total_size, total_size - 1)
  # Add enough trailing dims to avoid conflict with clipped_indices
  cumulative_sizes = expand_dims(cumulative_sizes, range(1, 1 + _ndim(indices)))
  clipped_indices = expand_dims(clipped_indices, axis=0)
  idx = clipped_indices % cumulative_sizes[:-1] // cumulative_sizes[1:]
  return tuple(idx)

@_wraps(np.resize)
@partial(jit, static_argnames=('new_shape',))
def resize(a, new_shape):
  _check_arraylike("resize", a)
  new_shape = _ensure_index_tuple(new_shape)

  if _any(dim_length < 0 for dim_length in new_shape):
    raise ValueError("all elements of `new_shape` must be non-negative")

  a = ravel(a)

  new_size = _prod(new_shape)
  if a.size == 0 or new_size == 0:
    return zeros_like(a, shape=new_shape)

  repeats = ceil_of_ratio(new_size, a.size)
  a = tile(a, repeats)[:new_size]

  return reshape(a, new_shape)

@_wraps(np.squeeze, lax_description=_ARRAY_VIEW_DOC)
def squeeze(a, axis: Optional[Union[int, Tuple[int, ...]]] = None):
  return _squeeze(a, _ensure_index_tuple(axis) if axis is not None else None)

@partial(jit, static_argnames=('axis',), inline=True)
def _squeeze(a, axis):
  _check_arraylike("squeeze", a)
  if axis is None:
    a_shape = shape(a)
    axis = tuple(i for i, d in enumerate(a_shape) if d == 1)
  return lax.squeeze(a, axis)


@_wraps(np.expand_dims)
def expand_dims(a, axis: Union[int, Sequence[int]]):
  _stackable(a) or _check_arraylike("expand_dims", a)
  axis = _ensure_index_tuple(axis)
  if hasattr(a, "expand_dims"):
    return a.expand_dims(axis)
  return lax.expand_dims(a, axis)


@_wraps(np.swapaxes, lax_description=_ARRAY_VIEW_DOC)
@partial(jit, static_argnames=('axis1', 'axis2'), inline=True)
def swapaxes(a, axis1: int, axis2: int):
  _check_arraylike("swapaxes", a)
  perm = np.arange(ndim(a))
  perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
  return lax.transpose(a, list(perm))


@_wraps(np.moveaxis, lax_description=_ARRAY_VIEW_DOC)
def moveaxis(a, source: Union[int, Sequence[int]],
             destination: Union[int, Sequence[int]]):
  return _moveaxis(a, _ensure_index_tuple(source),
                   _ensure_index_tuple(destination))

@partial(jit, static_argnames=('source', 'destination'), inline=True)
def _moveaxis(a, source: Tuple[int, ...], destination: Tuple[int, ...]):
  _check_arraylike("moveaxis", a)
  source = tuple(_canonicalize_axis(i, ndim(a)) for i in source)
  destination = tuple(_canonicalize_axis(i, ndim(a)) for i in destination)
  if len(source) != len(destination):
    raise ValueError("Inconsistent number of elements: {} vs {}"
                     .format(len(source), len(destination)))
  perm = [i for i in range(ndim(a)) if i not in source]
  for dest, src in sorted(zip(destination, source)):
    perm.insert(dest, src)
  return lax.transpose(a, perm)


@_wraps(np.isclose)
@partial(jit, static_argnames=('equal_nan',))
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  a, b = _promote_args("isclose", a, b)
  dtype = _dtype(a)
  if issubdtype(dtype, inexact):
    if issubdtype(dtype, complexfloating):
      dtype = _complex_elem_type(dtype)
    rtol = lax.convert_element_type(rtol, dtype)
    atol = lax.convert_element_type(atol, dtype)
    out = lax.le(
      lax.abs(lax.sub(a, b)),
      lax.add(atol, lax.mul(rtol, lax.abs(b))))
    # This corrects the comparisons for infinite and nan values
    a_inf = isinf(a)
    b_inf = isinf(b)
    any_inf = logical_or(a_inf, b_inf)
    both_inf = logical_and(a_inf, b_inf)
    # Make all elements where either a or b are infinite to False
    out = logical_and(out, logical_not(any_inf))
    # Make all elements where both a or b are the same inf to True
    same_value = lax.eq(a, b)
    same_inf = logical_and(both_inf, same_value)
    out = logical_or(out, same_inf)

    # Make all elements where either a or b is NaN to False
    a_nan = isnan(a)
    b_nan = isnan(b)
    any_nan = logical_or(a_nan, b_nan)
    out = logical_and(out, logical_not(any_nan))
    if equal_nan:
      # Make all elements where both a and b is NaN to True
      both_nan = logical_and(a_nan, b_nan)
      out = logical_or(out, both_nan)
    return out
  else:
    return lax.eq(a, b)


@_wraps(np.interp)
@partial(jit, static_argnames=('period',))
def interp(x, xp, fp, left=None, right=None, period=None):
  if shape(xp) != shape(fp) or ndim(xp) != 1:
    raise ValueError("xp and fp must be one-dimensional arrays of equal size")
  x, xp, fp = _promote_dtypes_inexact(x, xp, fp)
  if period is not None:
    if period == 0:
      raise ValueError(f"period must be a non-zero value; got {period}")
    period = abs(period)
    x = x % period
    xp = xp % period
    xp, fp = lax.sort_key_val(xp, fp)
    xp = concatenate([xp[-1:] - period, xp, xp[:1] + period])
    fp = concatenate([fp[-1:], fp, fp[:1]])

  i = clip(searchsorted(xp, x, side='right'), 1, len(xp) - 1)
  df = fp[i] - fp[i - 1]
  dx = xp[i] - xp[i - 1]
  delta = x - xp[i - 1]
  f = where((dx == 0), fp[i], fp[i - 1] + (delta / dx) * df)

  if period is None:
    f = where(x < xp[0], fp[0] if left is None else left, f)
    f = where(x > xp[-1], fp[-1] if right is None else right, f)
  return f


@_wraps(np.where,
  lax_description=_dedent("""
    At present, JAX does not support JIT-compilation of the single-argument form
    of :py:func:`jax.numpy.where` because its output shape is data-dependent. The
    three-argument form does not have a data-dependent shape and can be JIT-compiled
    successfully. Alternatively, you can use the optional ``size`` keyword to
    statically specify the expected size of the output."""),
  extra_params=_dedent("""
    size : int, optional
        Only referenced when ``x`` and ``y`` are ``None``. If specified, the indices of the first
        ``size`` elements of the result will be returned. If there are fewer elements than ``size``
        indicates, the return value will be padded with ``fill_value``.
    fill_value : array_like, optional
        When ``size`` is specified and there are fewer than the indicated number of elements, the
        remaining elements will be filled with ``fill_value``, which defaults to zero."""))
def where(condition, x=None, y=None, *, size=None, fill_value=None):
  if x is None and y is None:
    _check_arraylike("where", condition)
    return nonzero(condition, size=size, fill_value=fill_value)
  else:
    _check_arraylike("where", condition, x, y)
    if size is not None or fill_value is not None:
      raise ValueError("size and fill_value arguments cannot be used in three-term where function.")
    return _where(condition, x, y)


@_wraps(np.select)
def select(condlist, choicelist, default=0):
  if len(condlist) != len(choicelist):
    msg = "condlist must have length equal to choicelist ({} vs {})"
    raise ValueError(msg.format(len(condlist), len(choicelist)))
  if len(condlist) == 0:
    raise ValueError("condlist must be non-empty")
  choices = _promote_dtypes(default, *choicelist)
  choicelist = choices[1:]
  output = choices[0]
  for cond, choice in zip(condlist[::-1], choicelist[::-1]):
    output = where(cond, choice, output)
  return output


@_wraps(np.bincount, lax_description="""\
Jax adds the optional `length` parameter which specifies the output length, and
defaults to ``x.max() + 1``. It must be specified for bincount to be compiled
with non-static operands. Values larger than the specified length will be discarded.
If `length` is specified, `minlength` will be ignored.

Additionally, while ``np.bincount`` raises an error if the input array contains
negative values, ``jax.numpy.bincount`` clips negative values to zero.
""")
def bincount(x, weights=None, minlength=0, *, length=None):
  _check_arraylike("bincount", x)
  if not issubdtype(_dtype(x), integer):
    msg = f"x argument to bincount must have an integer type; got {x.dtype}"
    raise TypeError(msg)
  if ndim(x) != 1:
    raise ValueError("only 1-dimensional input supported.")
  minlength = core.concrete_or_error(operator.index, minlength,
      "The error occurred because of argument 'minlength' of jnp.bincount.")
  if length is None:
    x = core.concrete_or_error(asarray, x,
      "The error occured because of argument 'x' of jnp.bincount. "
      "To avoid this error, pass a static `length` argument.")
    length = _max(minlength, x.size and x.max() + 1)
  else:
    length = core.concrete_or_error(operator.index, length,
        "The error occurred because of argument 'length' of jnp.bincount.")
  if weights is None:
    weights = np.array(1, dtype=int_)
  elif shape(x) != shape(weights):
    raise ValueError("shape of weights must match shape of x.")
  return zeros(length, _dtype(weights)).at[clip(x, 0)].add(weights)

@_wraps(getattr(np, "broadcast_shapes", None))
def broadcast_shapes(*shapes):
  if not shapes:
    return ()
  shapes = [(shape,) if np.ndim(shape) == 0 else tuple(shape) for shape in shapes]
  return lax.broadcast_shapes(*shapes)


broadcast_arrays = _wraps(np.broadcast_arrays, lax_description="""\
The JAX version does not necessarily return a view of the input.
""")(_broadcast_arrays)


broadcast_to = _wraps(np.broadcast_to, lax_description="""\
The JAX version does not necessarily return a view of the input.
""")(_broadcast_to)


def _split(op, ary, indices_or_sections, axis=0):
  _check_arraylike(op, ary)
  ary = asarray(ary)
  axis = core.concrete_or_error(int, axis, f"in jax.numpy.{op} argument `axis`")
  size = ary.shape[axis]
  if isinstance(indices_or_sections, (tuple, list)):
    indices_or_sections = np.array(
        [core.concrete_or_error(np.int64, i_s, f"in jax.numpy.{op} argument 1")
         for i_s in indices_or_sections], np.int64)
    split_indices = np.concatenate([[np.int64(0)], indices_or_sections,
                                    [np.int64(size)]])
  elif (isinstance(indices_or_sections, (np.ndarray, ndarray)) and
        indices_or_sections.ndim > 0):
    indices_or_sections = np.array(
        [core.concrete_or_error(np.int64, i_s, f"in jax.numpy.{op} argument 1")
         for i_s in indices_or_sections], np.int64)
    split_indices = np.concatenate([[np.int64(0)], indices_or_sections,
                                    [np.int64(size)]])
  else:
    indices_or_sections = core.concrete_or_error(np.int64, indices_or_sections,
                                                 f"in jax.numpy.{op} argument 1")
    part_size, r = _divmod(size, indices_or_sections)
    if r == 0:
      split_indices = np.arange(indices_or_sections + 1,
                                dtype=np.int64) * part_size
    elif op == "array_split":
      split_indices = np.concatenate(
          [np.arange(r + 1, dtype=np.int64) * (part_size + 1),
           np.arange(indices_or_sections - r, dtype=np.int64) * part_size
           + ((r + 1) * (part_size + 1) - 1)])
    else:
      raise ValueError("array split does not result in an equal division")
  starts, ends = [0] * ndim(ary), shape(ary)
  _subval = lambda x, i, v: subvals(x, [(i, v)])
  return [lax.slice(ary, _subval(starts, axis, start), _subval(ends, axis, end))
          for start, end in zip(split_indices[:-1], split_indices[1:])]

@_wraps(np.split, lax_description=_ARRAY_VIEW_DOC)
def split(ary, indices_or_sections, axis: int = 0):
  return _split("split", ary, indices_or_sections, axis=axis)

def _split_on_axis(op, axis):
  @_wraps(getattr(np, op), update_doc=False)
  def f(ary, indices_or_sections):
    return _split(op, ary, indices_or_sections, axis=axis)
  return f

vsplit = _split_on_axis("vsplit", axis=0)
hsplit = _split_on_axis("hsplit", axis=1)
dsplit = _split_on_axis("dsplit", axis=2)

@_wraps(np.array_split)
def array_split(ary, indices_or_sections, axis: int = 0):
  return _split("array_split", ary, indices_or_sections, axis=axis)

@_wraps(np.clip, skip_params=['out'])
@jit
def clip(a, a_min=None, a_max=None, out=None):
  _check_arraylike("clip", a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.clip is not supported.")
  if a_min is None and a_max is None:
    raise ValueError("At most one of a_min and a_max may be None")
  if a_min is not None:
    a = maximum(a_min, a)
  if a_max is not None:
    a = minimum(a_max, a)
  return a

@_wraps(np.around, skip_params=['out'])
@partial(jit, static_argnames=('decimals',))
def round(a, decimals=0, out=None):
  _check_arraylike("round", a)
  decimals = core.concrete_or_error(operator.index, decimals, "'decimals' argument of jnp.round")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.round is not supported.")
  dtype = _dtype(a)
  if issubdtype(dtype, integer):
    if decimals < 0:
      raise NotImplementedError(
        "integer np.round not implemented for decimals < 0")
    return a  # no-op on integer types

  def _round_float(x):
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


@_wraps(np.fix, skip_params=['out'])
@jit
def fix(x, out=None):
  _check_arraylike("fix", x)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.fix is not supported.")
  zero = _lax_const(x, 0)
  return where(lax.ge(x, zero), floor(x), ceil(x))


@_wraps(np.nan_to_num)
@jit
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
  del copy
  _check_arraylike("nan_to_num", x)
  dtype = _dtype(x)
  if issubdtype(dtype, complexfloating):
    return lax.complex(
      nan_to_num(lax.real(x), nan=nan, posinf=posinf, neginf=neginf),
      nan_to_num(lax.imag(x), nan=nan, posinf=posinf, neginf=neginf))
  info = finfo(dtypes.canonicalize_dtype(dtype))
  posinf = info.max if posinf is None else posinf
  neginf = info.min if neginf is None else neginf
  x = where(isnan(x), array(nan, dtype=x.dtype), x)
  x = where(isposinf(x), array(posinf, dtype=x.dtype), x)
  x = where(isneginf(x), array(neginf, dtype=x.dtype), x)
  return x


@_wraps(np.allclose)
@partial(jit, static_argnames=('equal_nan',))
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  _check_arraylike("allclose", a, b)
  return all(isclose(a, b, rtol, atol, equal_nan))


_NONZERO_DOC = """\
Because the size of the output of ``nonzero`` is data-dependent, the function is not
typically compatible with JIT. The JAX version adds the optional ``size`` argument which
must be specified statically for ``jnp.nonzero`` to be used within some of JAX's
transformations.
"""
_NONZERO_EXTRA_PARAMS = """
size : int, optional
    If specified, the indices of the first ``size`` True elements will be returned. If there are
    fewer unique elements than ``size`` indicates, the return value will be padded with ``fill_value``.
fill_value : array_like, optional
    When ``size`` is specified and there are fewer than the indicated number of elements, the
    remaining elements will be filled with ``fill_value``, which defaults to zero.
"""

@_wraps(np.nonzero, lax_description=_NONZERO_DOC, extra_params=_NONZERO_EXTRA_PARAMS)
def nonzero(a, *, size=None, fill_value=None):
  a = atleast_1d(a)
  mask = a != 0
  if size is None:
    size = mask.sum()
  size = core.concrete_or_error(int, size,
    "The size argument of jnp.nonzero must be statically specified "
    "to use jnp.nonzero within JAX transformations.")
  if a.size == 0 or size == 0:
    return tuple(zeros(size, int) for dim in a.shape)
  flat_indices = cumsum(bincount(cumsum(mask), length=size))
  strides = (np.cumprod(a.shape[::-1])[::-1] // a.shape).astype(int_)
  out = tuple((flat_indices // stride) % size for stride, size in zip(strides, a.shape))
  if size is not None and fill_value is not None:
    if not isinstance(fill_value, tuple):
      fill_value = a.ndim * (fill_value,)
    if _shape(fill_value) != (a.ndim,):
      raise ValueError(f"fill_value must be a scalar or a tuple of length {a.ndim}; got {fill_value}")
    fill_mask = arange(size) >= mask.sum()
    out = tuple(where(fill_mask, fval, entry) for fval, entry in safe_zip(fill_value, out))
  return out

@_wraps(np.flatnonzero, lax_description=_NONZERO_DOC, extra_params=_NONZERO_EXTRA_PARAMS)
def flatnonzero(a, *, size=None, fill_value=None):
  return nonzero(ravel(a), size=size, fill_value=fill_value)[0]


@_wraps(np.unwrap)
@partial(jit, static_argnames=('axis',))
def unwrap(p, discont=pi, axis: int = -1):
  _check_arraylike("unwrap", p)
  dd = diff(p, axis=axis)
  ddmod = mod(dd + pi, 2 * pi) - pi
  ddmod = where((ddmod == -pi) & (dd > 0), pi, ddmod)

  ph_correct = where(abs(dd) < discont, 0, ddmod - dd)

  up = concatenate((
    lax.slice_in_dim(p, 0, 1, axis=axis),
    lax.slice_in_dim(p, 1, None, axis=axis) + cumsum(ph_correct, axis=axis)
  ), axis=axis)

  return up


### Array-creation functions

def _check_no_padding(axis_padding, mode):
  if (axis_padding[0] > 0 or axis_padding[1] > 0):
    msg = "Cannot apply '{}' padding to empty axis"
    raise ValueError(msg.format(mode))


def _pad_constant(array, pad_width, constant_values):
  nd = ndim(array)
  constant_values = broadcast_to(asarray(constant_values), (nd, 2))
  constant_values = lax_internal._convert_element_type(
      constant_values, array.dtype, dtypes.is_weakly_typed(array))
  for i in range(nd):
    widths = [(0, 0, 0)] * nd
    widths[i] = (pad_width[i, 0], 0, 0)
    array = lax.pad(array, constant_values[i, 0], widths)
    widths[i] = (0, pad_width[i, 1], 0)
    array = lax.pad(array, constant_values[i, 1], widths)
  return array


def _pad_wrap(array, pad_width):
  for i in range(ndim(array)):
    if array.shape[i] == 0:
      _check_no_padding(pad_width[i], "wrap")
      continue
    size = array.shape[i]
    repeats, (left_remainder, right_remainder) = _divmod(pad_width[i], size)
    total_repeats = repeats.sum() + 1
    parts = []
    if left_remainder:
      parts += [lax.slice_in_dim(array, size - left_remainder, size, axis=i)]
    parts += total_repeats * [array]
    if right_remainder:
      parts += [lax.slice_in_dim(array, 0, right_remainder, axis=i)]
    array = lax.concatenate(parts, dimension=i)
  return array


def _pad_symmetric_or_reflect(array, pad_width, mode, reflect_type):
  assert mode in ("symmetric", "reflect")
  assert reflect_type in ("even", "odd")

  for i in range(ndim(array)):
    if array.shape[i] == 0:
      _check_no_padding(pad_width[i], mode)
      continue

    n = array.shape[i]
    offset = 1 if (mode == "reflect" and n > 1) else 0

    def build_padding(array, padding, before):
      if before:
        edge = lax.slice_in_dim(array, 0, 1, axis=i)
      else:
        edge = lax.slice_in_dim(array, -1, None, axis=i)

      while padding > 0:
        curr_pad = _min(padding, n - offset)
        padding -= curr_pad

        if before:
          start = offset
          stop = offset + curr_pad
        else:
          start = -(curr_pad + offset)
          stop = None if (mode == "symmetric" or n == 1) else -1

        x = lax.slice_in_dim(array, start, stop, axis=i)
        x = flip(x, axis=i)

        if reflect_type == 'odd':
          x = 2 * edge - x
          if n > 1:
            if before:
              edge = lax.slice_in_dim(x, 0, 1, axis=i)
            else:
              edge = lax.slice_in_dim(x, -1, None, axis=i)

        if before:
          array = lax.concatenate([x, array], dimension=i)
        else:
          array = lax.concatenate([array, x], dimension=i)
      return array

    array = build_padding(array, pad_width[i, 0], before=True)
    array = build_padding(array, pad_width[i, 1], before=False)
  return array


def _pad_edge(array, pad_width):
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


def _pad_linear_ramp(array, pad_width, end_values):
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


def _pad_stats(array, pad_width, stat_length, stat_func):
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
      length_before = _min(length_before, array_length)
      length_after = _min(length_after, array_length)

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


def _pad_empty(array, pad_width):
  # Note: jax.numpy.empty = jax.numpy.zeros
  for i in range(ndim(array)):
    shape_before = array.shape[:i] + (pad_width[i][0],) + array.shape[i + 1:]
    pad_before = empty_like(array, shape=shape_before)

    shape_after = array.shape[:i] + (pad_width[i][1],) + array.shape[i + 1:]
    pad_after = empty_like(array, shape=shape_after)
    array = lax.concatenate([pad_before, array, pad_after], dimension=i)
  return array


def _pad_func(array, pad_width, func, **kwargs):
  pad_width = _broadcast_to_pairs(pad_width, ndim(array), "pad_width")
  padded = _pad_constant(array, np.array(pad_width), 0)
  for axis in range(ndim(padded)):
    padded = apply_along_axis(func, axis, padded, pad_width[axis], axis, kwargs)
  return padded


def _broadcast_to_pairs(nvals, nd, name):
  nvals = np.asarray(tree_map(
    lambda x: core.concrete_or_error(np.array, x, context=f"{name} argument of jnp.pad"),
    nvals))
  if nvals.dtype.kind == 'O':
    raise TypeError(f'`{name}` entries must be the same shape.')

  if nvals.shape == (nd, 2):
    # ((before_1, after_1), ..., (before_N, after_N))
    return tuple(tuple(nval) for nval in nvals)
  elif nvals.shape == (1, 2):
    # ((before, after),)
    return tuple(tuple(nvals[0]) for i in range(nd))
  elif nvals.shape == (2,):
    # (before, after)  (not in the numpy docstring but works anyway)
    return tuple(tuple(nvals) for i in range(nd))
  elif nvals.shape == (1,):
    # (pad,)
    return tuple((nvals[0], nvals[0]) for i in range(nd))
  elif nvals.shape == ():
    # pad
    return tuple((nvals.flat[0], nvals.flat[0]) for i in range(nd))
  else:
    raise ValueError(f"jnp.pad: {name} with nd={nd} has unsupported shape {nvals.shape}. "
                     f"Valid shapes are ({nd}, 2), (1, 2), (2,), (1,), or ().")


@partial(jit, static_argnums=(1, 2, 4, 5, 6))
def _pad(array, pad_width, mode, constant_values, stat_length, end_values, reflect_type):
  array = asarray(array)
  nd = ndim(array)

  if nd == 0:
    return array

  stat_funcs = {"maximum": amax, "minimum": amin,
                "mean": mean, "median": median}

  pad_width = _broadcast_to_pairs(pad_width, nd, "pad_width")
  pad_width = np.array(pad_width)
  assert pad_width.shape == (nd, 2), pad_width

  if np.any(pad_width < 0):
    raise ValueError("index can't contain negative values")

  if mode == "constant":
    return _pad_constant(array, pad_width, constant_values)

  elif mode == "wrap":
    return _pad_wrap(array, pad_width)

  elif mode in ("symmetric", "reflect"):
    return _pad_symmetric_or_reflect(array, pad_width, mode, reflect_type)

  elif mode == "edge":
    return _pad_edge(array, pad_width)

  elif mode == "linear_ramp":
    end_values = _broadcast_to_pairs(end_values, nd, "end_values")
    return _pad_linear_ramp(array, pad_width, end_values)

  elif mode in stat_funcs:
    if stat_length is not None:
      stat_length = _broadcast_to_pairs(stat_length, nd, "stat_length")
    return _pad_stats(array, pad_width, stat_length, stat_funcs[mode])

  elif mode == "empty":
    return _pad_empty(array, pad_width)

  else:
    assert False, ("Should not be reached since pad already handled unsupported and"
                   "not implemented modes")


@_wraps(np.pad, lax_description="""\
Unlike numpy, JAX "function" mode's argument (which is another function) should return
the modified array. This is because Jax arrays are immutable.
(In numpy, "function" mode's argument should modify a rank 1 array in-place.)
""")
def pad(array, pad_width, mode="constant", **kwargs):
  _check_arraylike("pad", array)
  pad_width = _broadcast_to_pairs(pad_width, ndim(array), "pad_width")
  if pad_width and np.array(pad_width).dtype.kind != 'i':
    raise TypeError('`pad_width` must be of integral type.')

  if callable(mode):
    return _pad_func(array, pad_width, mode, **kwargs)

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


@_wraps(np.stack, skip_params=['out'])
def stack(arrays, axis: int = 0, out=None):
  if not len(arrays):
    raise ValueError("Need at least one array to stack.")
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.stack is not supported.")
  if isinstance(arrays, (np.ndarray, ndarray)):
    axis = _canonicalize_axis(axis, arrays.ndim)
    return concatenate(expand_dims(arrays, axis + 1), axis=axis)
  else:
    _stackable(*arrays) or _check_arraylike("stack", *arrays)
    shape0 = shape(arrays[0])
    axis = _canonicalize_axis(axis, len(shape0) + 1)
    new_arrays = []
    for a in arrays:
      if shape(a) != shape0:
        raise ValueError("All input arrays must have the same shape.")
      new_arrays.append(expand_dims(a, axis))
    return concatenate(new_arrays, axis=axis)

@_wraps(np.tile)
def tile(A, reps):
  _stackable(A) or _check_arraylike("tile", A)
  try:
    iter(reps)
  except TypeError:
    reps = (reps,)
  reps = tuple(operator.index(rep) if core.is_constant_dim(rep) else rep
               for rep in reps)
  A_shape = (1,) * (len(reps) - ndim(A)) + shape(A)
  reps = (1,) * (len(A_shape) - len(reps)) + reps
  result = broadcast_to(reshape(A, [j for i in A_shape for j in [1, i]]),
                        [k for pair in zip(reps, A_shape) for k in pair])
  return reshape(result, tuple(np.multiply(A_shape, reps)))

def _concatenate_array(arr, axis: int):
  # Fast path for concatenation when the input is an ndarray rather than a list.
  arr = asarray(arr)
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

@_wraps(np.concatenate)
def concatenate(arrays, axis: int = 0):
  if isinstance(arrays, (np.ndarray, ndarray)):
    return _concatenate_array(arrays, axis)
  _stackable(*arrays) or _check_arraylike("concatenate", *arrays)
  if not len(arrays):
    raise ValueError("Need at least one array to concatenate.")
  if ndim(arrays[0]) == 0:
    raise ValueError("Zero-dimensional arrays cannot be concatenated.")
  if axis is None:
    return concatenate([ravel(a) for a in arrays], axis=0)
  if hasattr(arrays[0], "concatenate"):
    return arrays[0].concatenate(arrays[1:], axis)
  axis = _canonicalize_axis(axis, ndim(arrays[0]))
  arrays = _promote_dtypes(*arrays)
  # lax.concatenate can be slow to compile for wide concatenations, so form a
  # tree of concatenations as a workaround especially for op-by-op mode.
  # (https://github.com/google/jax/issues/653).
  k = 16
  if len(arrays) == 1:
    return asarray(arrays[0])
  else:
    while len(arrays) > 1:
      arrays = [lax.concatenate(arrays[i:i+k], axis)
                for i in range(0, len(arrays), k)]
    return arrays[0]


@_wraps(np.vstack)
def vstack(tup):
  if isinstance(tup, (np.ndarray, ndarray)):
    arrs = jax.vmap(atleast_2d)(tup)
  else:
    arrs = [atleast_2d(m) for m in tup]
  return concatenate(arrs, axis=0)
row_stack = vstack


@_wraps(np.hstack)
def hstack(tup):
  if isinstance(tup, (np.ndarray, ndarray)):
    arrs = jax.vmap(atleast_1d)(tup)
    arr0_ndim = arrs.ndim - 1
  else:
    arrs = [atleast_1d(m) for m in tup]
    arr0_ndim = arrs[0].ndim
  return concatenate(arrs, axis=0 if arr0_ndim == 1 else 1)


@_wraps(np.dstack)
def dstack(tup):
  if isinstance(tup, (np.ndarray, ndarray)):
    arrs = jax.vmap(atleast_3d)(tup)
  else:
    arrs = [atleast_3d(m) for m in tup]
  return concatenate(arrs, axis=2)


@_wraps(np.column_stack)
def column_stack(tup):
  if isinstance(tup, (np.ndarray, ndarray)):
    arrs = jax.vmap(lambda x: atleast_2d(x).T)(tup) if tup.ndim < 3 else tup
  else:
    arrs = [atleast_2d(arr).T if arr.ndim < 2 else arr for arr in map(asarray, tup)]
  return concatenate(arrs, 1)


@_wraps(np.choose, skip_params=['out'])
def choose(a, choices, out=None, mode='raise'):
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.choose is not supported.")
  _check_arraylike('choose', a, *choices)
  if not issubdtype(_dtype(a), integer):
    raise ValueError("`a` array must be integer typed")
  N = len(choices)

  if mode == 'raise':
    a = core.concrete_or_error(asarray, a,
      "The error occurred because jnp.choose was jit-compiled"
      " with mode='raise'. Use mode='wrap' or mode='clip' instead.")
    if any((a < 0) | (a >= N)):
      raise ValueError("invalid entry in choice array")
  elif mode == 'wrap':
    a = a % N
  elif mode == 'clip':
    a = clip(a, 0, N - 1)
  else:
    raise ValueError(f"mode={mode!r} not understood. Must be 'raise', 'wrap', or 'clip'")

  a, *choices = broadcast_arrays(a, *choices)
  return array(choices)[(a,) + indices(a.shape, sparse=True)]


def _atleast_nd(x, n):
  m = ndim(x)
  return lax.broadcast(x, (1,) * (n - m)) if m < n else x

def _block(xs):
  if isinstance(xs, tuple):
    raise ValueError("jax.numpy.block does not allow tuples, got {}"
                     .format(xs))
  elif isinstance(xs, list):
    if len(xs) == 0:
      raise ValueError("jax.numpy.block does not allow empty list arguments")
    xs, depths = unzip2([_block(x) for x in xs])
    if _any(d != depths[0] for d in depths[1:]):
      raise ValueError("Mismatched list depths in jax.numpy.block")
    rank = _max(depths[0], _max(ndim(x) for x in xs))
    xs = [_atleast_nd(x, rank) for x in xs]
    return concatenate(xs, axis=-depths[0]), depths[0] + 1
  else:
    return asarray(xs), 1

@_wraps(np.block)
@jit
def block(arrays):
  out, _ = _block(arrays)
  return out

@_wraps(np.atleast_1d, update_doc=False, lax_description=_ARRAY_VIEW_DOC)
@jit
def atleast_1d(*arys):
  if len(arys) == 1:
    arr = asarray(arys[0])
    return arr if ndim(arr) >= 1 else reshape(arr, -1)
  else:
    return [atleast_1d(arr) for arr in arys]


@_wraps(np.atleast_2d, update_doc=False, lax_description=_ARRAY_VIEW_DOC)
@jit
def atleast_2d(*arys):
  if len(arys) == 1:
    arr = asarray(arys[0])
    if ndim(arr) >= 2:
      return arr
    elif ndim(arr) == 1:
      return expand_dims(arr, axis=0)
    else:
      return expand_dims(arr, axis=(0, 1))
  else:
    return [atleast_2d(arr) for arr in arys]


@_wraps(np.atleast_3d, update_doc=False, lax_description=_ARRAY_VIEW_DOC)
@jit
def atleast_3d(*arys):
  if len(arys) == 1:
    arr = asarray(arys[0])
    if ndim(arr) == 0:
      arr = expand_dims(arr, axis=(0, 1, 2))
    elif ndim(arr) == 1:
      arr = expand_dims(arr, axis=(0, 2))
    elif ndim(arr) == 2:
      arr = expand_dims(arr, axis=2)
    return arr
  else:
    return [atleast_3d(arr) for arr in arys]


_ARRAY_DOC = """
This function will create arrays on JAX's default device. For control of the
device placement of data, see :func:`jax.device_put`. More information is
available in the JAX FAQ at :ref:`faq-data-placement` (full FAQ at
https://jax.readthedocs.io/en/latest/faq.html).
"""

@_wraps(np.array, lax_description=_ARRAY_DOC)
def array(object, dtype=None, copy=True, order="K", ndmin=0):
  if order is not None and order != "K":
    raise NotImplementedError("Only implemented for order='K'")

  # check if the given dtype is compatible with JAX
  lax_internal._check_user_dtype_supported(dtype, "array")

  # Here we make a judgment call: we only return a weakly-typed array when the
  # input object itself is weakly typed. That ensures asarray(x) is a no-op whenever
  # x is weak, but avoids introducing weak types with something like array([1, 2, 3])
  # weak_type = dtype is None and dtypes.is_weakly_typed(object)
  weak_type = False

  # For Python scalar literals, call coerce_to_array to catch any overflow errors.
  # We don't use dtypes.is_python_scalar because we don't want this triggering for
  # traced values. We do this here because it matters whether or not dtype is None.
  # We don't assign the result because we want the raw object to be used for type
  # inference below.
  if isinstance(object, (bool, int, float, complex)):
    _ = dtypes.coerce_to_array(object, dtype)

  leaves = tree_leaves(object)
  if dtype is None:
    # Use lattice_result_type rather than result_type to avoid canonicalization.
    # Otherwise, weakly-typed inputs would have their dtypes canonicalized.
    if not leaves:
      dtype = dtypes.float_
    else:
      try:
        dtype, weak_type = dtypes._lattice_result_type(*leaves)
      except TypeError:
        # This happens if, e.g. one of the entries is a memoryview object.
        # This is rare, so we only handle it if the normal path fails.
        leaves = [_convert_to_array_if_dtype_fails(leaf) for leaf in leaves]
        dtype, weak_type = dtypes._lattice_result_type(*leaves)
      if weak_type:
        dtype = dtypes._default_types['f' if dtype == dtypes._bfloat16_dtype
                                      else dtype.kind]
        weak_type = False
  assert not weak_type

  if not weak_type:
    dtype = dtypes.canonicalize_dtype(dtype)

  # We can't use the ndarray class because we need to handle internal buffers
  # (See https://github.com/google/jax/issues/8950)
  ndarray_types = (device_array.DeviceArray, core.Tracer)

  if not _any(isinstance(leaf, ndarray_types) for leaf in leaves):
    # TODO(jakevdp): falling back to numpy here fails to overflow for lists containing
    # large integers; see discussion in https://github.com/google/jax/pull/6047.
    # More correct would be to call coerce_to_array on each leaf, but this may have
    # performance implications.
    out = np.array(object, dtype=dtype, ndmin=ndmin, copy=False)
  elif isinstance(object, ndarray_types):
    if object.aval is None:
      # object is a raw buffer; convert to device array on its current device.
      aval = ShapedArray(object.xla_shape().dimensions(), object.dtype,
                         weak_type=bool(getattr(object, "weak_type", False)))
      object = device_array.make_device_array(aval, object.device(), object)
    out = _array_copy(object) if copy else object
  elif isinstance(object, (list, tuple)):
    if object:
      out = stack([asarray(elt, dtype=dtype) for elt in object])
    else:
      out = np.array([], dtype=dtype)
  else:
    try:
      view = memoryview(object)
    except TypeError:
      pass  # `object` does not support the buffer interface.
    else:
      return array(np.asarray(view), dtype, copy, ndmin=ndmin)

    raise TypeError("Unexpected input type for array: {}".format(type(object)))

  out = lax_internal._convert_element_type(out, dtype, weak_type=weak_type)
  if ndmin > ndim(out):
    out = lax.expand_dims(out, range(ndmin - ndim(out)))
  return out


def _convert_to_array_if_dtype_fails(x):
  try:
    dtypes.dtype(x)
  except TypeError:
    return np.asarray(x)
  else:
    return x


@_wraps(np.asarray, lax_description=_ARRAY_DOC)
def asarray(a, dtype=None, order=None):
  lax_internal._check_user_dtype_supported(dtype, "asarray")
  dtype = dtypes.canonicalize_dtype(dtype) if dtype is not None else dtype
  return array(a, dtype=dtype, copy=False, order=order)


@_wraps(np.copy, lax_description=_ARRAY_DOC)
def copy(a, order=None):
  return array(a, copy=True, order=order)


@_wraps(np.zeros_like)
def zeros_like(a, dtype=None, shape=None):
  _check_arraylike("zeros_like", a)
  lax_internal._check_user_dtype_supported(dtype, "zeros_like")
  if np.isscalar(shape):
    shape = (shape,)
  return lax.full_like(a, 0, dtype, shape)


@_wraps(np.ones_like)
def ones_like(a, dtype=None, shape=None):
  _check_arraylike("ones_like", a)
  lax_internal._check_user_dtype_supported(dtype, "ones_like")
  if np.isscalar(shape):
    shape = (shape,)
  return lax.full_like(a, 1, dtype, shape)


@_wraps(np.full)
def full(shape, fill_value, dtype=None):
  lax_internal._check_user_dtype_supported(dtype, "full")
  _check_arraylike("full", fill_value)
  if ndim(fill_value) == 0:
    shape = (shape,) if ndim(shape) == 0 else shape
    return lax.full(shape, fill_value, dtype)
  else:
    return broadcast_to(asarray(fill_value, dtype=dtype), shape)


@_wraps(np.full_like)
def full_like(a, fill_value, dtype=None, shape=None):
  lax_internal._check_user_dtype_supported(dtype, "full_like")
  _check_arraylike("full_like", a, fill_value)
  if shape is not None:
    shape = (shape,) if ndim(shape) == 0 else shape
  if ndim(fill_value) == 0:
    return lax.full_like(a, fill_value, dtype, shape)
  else:
    shape = np.shape(a) if shape is None else shape
    dtype = result_type(a) if dtype is None else dtype
    return broadcast_to(asarray(fill_value, dtype=dtype), shape)


@_wraps(np.zeros)
def zeros(shape, dtype=None):
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  lax_internal._check_user_dtype_supported(dtype, "zeros")
  shape = canonicalize_shape((shape,) if ndim(shape) == 0 else shape)
  return lax.full(shape, 0, _jnp_dtype(dtype))

@_wraps(np.ones)
def ones(shape, dtype=None):
  if isinstance(shape, types.GeneratorType):
    raise TypeError("expected sequence object with len >= 0 or a single integer")
  lax_internal._check_user_dtype_supported(dtype, "ones")
  shape = canonicalize_shape((shape,) if ndim(shape) == 0 else shape)
  return lax.full(shape, 1, _jnp_dtype(dtype))


@_wraps(np.array_equal)
def array_equal(a1, a2, equal_nan=False):
  try:
    a1, a2 = asarray(a1), asarray(a2)
  except Exception:
    return False
  if shape(a1) != shape(a2):
    return False
  eq = asarray(a1 == a2)
  if equal_nan:
    eq = logical_or(eq, logical_and(isnan(a1), isnan(a2)))
  return all(eq)


@_wraps(np.array_equiv)
def array_equiv(a1, a2):
  try:
    a1, a2 = asarray(a1), asarray(a2)
  except Exception:
    return False
  try:
    eq = equal(a1, a2)
  except ValueError:
    # shapes are not broadcastable
    return False
  return all(eq)


# We can't create uninitialized arrays in XLA; use zeros for empty.
empty_like = zeros_like
empty = zeros


# General np.from* style functions mostly delegate to numpy.

@_wraps(np.frombuffer)
def frombuffer(buffer, dtype=float, count=-1, offset=0):
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


@_wraps(np.fromfunction)
def fromfunction(function, shape, *, dtype=float, **kwargs):
  shape = core.canonicalize_shape(shape, context="shape argument of jnp.fromfunction()")
  for i in range(len(shape)):
    in_axes = [0 if i == j else None for j in range(len(shape))]
    function = jax.vmap(function, in_axes=tuple(in_axes[::-1]))
  return function(*(arange(s, dtype=dtype) for s in shape), **kwargs)


@_wraps(np.fromstring)
def fromstring(string, dtype=float, count=-1, *, sep):
  return asarray(np.fromstring(string=string, dtype=dtype, count=count, sep=sep))


@_wraps(np.eye)
def eye(N, M=None, k=0, dtype=None):
  lax_internal._check_user_dtype_supported(dtype, "eye")
  N = core.canonicalize_dim(N, "'N' argument of jnp.eye()")
  M = N if M is None else core.canonicalize_dim(M, "'M' argument of jnp.eye()")
  if N < 0 or M < 0:
    raise ValueError(f"negative dimensions are not allowed, got {N} and {M}")
  k = operator.index(k)
  return lax_internal._eye(_jnp_dtype(dtype), (N, M), k)


@_wraps(np.identity)
def identity(n, dtype=None):
  lax_internal._check_user_dtype_supported(dtype, "identity")
  return eye(n, dtype=dtype)


@_wraps(np.arange)
def arange(start: core.DimSize, stop: Optional[core.DimSize]=None,
           step: Optional[core.DimSize]=None, dtype=None):
  lax_internal._check_user_dtype_supported(dtype, "arange")
  require = partial(core.concrete_or_error, None)
  msg = "It arose in jax.numpy.arange argument `{}`.".format
  if _any(core.is_special_dim_size(d) for d in (start, stop, step)):
    if stop is not None or step is not None:
      raise ValueError(
          "jax.numpy.arange supports non-constant arguments only in single-argument form. "
          f"Found jax.numpy.arange(start={start}, stop={stop}, step={step})")
    return lax.iota(int_, start)
  if dtype is None:
    dtype = result_type(start, *(x for x in [stop, step] if x is not None))
  dtype = _jnp_dtype(dtype)
  if stop is None and step is None:
    start = require(start, msg("stop"))
    start = np.ceil(start).astype(int)
    return lax.iota(dtype, start)
  else:
    start = require(start, msg("start"))
    stop = None if stop is None else require(stop, msg("stop"))
    step = None if step is None else require(step, msg("step"))
    return array(np.arange(start, stop=stop, step=step, dtype=dtype))


def _wrap_numpy_nullary_function(f):
  """Adapts `f` to return a DeviceArray instead of an np.ndarray.

  `f` cannot have any non-static array arguments.
  """
  @_wraps(f, update_doc=False)
  def wrapper(*args, **kwargs):
    args = [core.concrete_or_error(None, arg, f"the error occured in argument {i} jnp.{f.__name__}()")
            for i, arg in enumerate(args)]
    kwargs = {key: core.concrete_or_error(None, val, f"the error occured in argument '{key}' jnp.{f.__name__}()")
              for key, val in kwargs.items()}
    return asarray(f(*args, **kwargs))
  return wrapper


@_wraps(np.linspace)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis: int = 0):
  num = core.concrete_or_error(operator.index, num, "'num' argument of jnp.linspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.linspace")
  return _linspace(start, stop, int(num), endpoint, retstep, dtype,
                   operator.index(axis))

@partial(jit, static_argnames=('num', 'endpoint', 'retstep', 'dtype', 'axis'))
def _linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
              axis: int = 0):
  """Implementation of linspace differentiable in start and stop args."""
  lax_internal._check_user_dtype_supported(dtype, "linspace")
  if num < 0:
    raise ValueError(f"Number of samples, {num}, must be non-negative.")
  _check_arraylike("linspace", start, stop)

  if dtype is None:
    dtype = result_type(start, stop, dtypes.canonicalize_dtype(float_))
  dtype = _jnp_dtype(dtype)
  computation_dtype = promote_types(dtype, dtypes.canonicalize_dtype(float_))
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)

  bounds_shape = list(lax.broadcast_shapes(shape(start), shape(stop)))
  broadcast_start = broadcast_to(start, bounds_shape)
  broadcast_stop = broadcast_to(stop, bounds_shape)
  axis = len(bounds_shape) + axis + 1 if axis < 0 else axis
  bounds_shape.insert(axis, 1)
  div = (num - 1) if endpoint else num
  if num > 1:
    delta = lax.convert_element_type(stop - start, computation_dtype) / div
    iota_shape = [1,] * len(bounds_shape)
    iota_shape[axis] = div
    # This approach recovers the endpoints with float32 arithmetic,
    # but can lead to rounding errors for integer outputs.
    real_dtype = finfo(computation_dtype).dtype
    step = reshape(lax.iota(real_dtype, div), iota_shape) / div
    out = (reshape(broadcast_start, bounds_shape) * (1 - step) +
      reshape(broadcast_stop, bounds_shape) * step)

    if endpoint:
      out = lax.concatenate([out, lax.expand_dims(broadcast_stop, (axis,))],
                            _canonicalize_axis(axis, out.ndim))

  elif num == 1:
    delta = nan if endpoint else stop - start
    out = reshape(broadcast_start, bounds_shape)
  else: # num == 0 degenerate case, match numpy behavior
    empty_shape = list(lax.broadcast_shapes(shape(start), shape(stop)))
    empty_shape.insert(axis, 0)
    delta = nan
    out = reshape(array([], dtype=dtype), empty_shape)

  if issubdtype(dtype, integer) and not issubdtype(out.dtype, integer):
    out = lax.floor(out)

  if retstep:
    return lax.convert_element_type(out, dtype), delta
  else:
    return lax.convert_element_type(out, dtype)


@_wraps(np.logspace)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None,
             axis: int = 0):
  num = core.concrete_or_error(operator.index, num, "'num' argument of jnp.logspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.logspace")
  return _logspace(start, stop, int(num), endpoint, base, dtype,
                   operator.index(axis))

@partial(jit, static_argnames=('num', 'endpoint', 'dtype', 'axis'))
def _logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None,
             axis: int = 0):
  """Implementation of logspace differentiable in start and stop args."""
  lax_internal._check_user_dtype_supported(dtype, "logspace")
  if dtype is None:
    dtype = result_type(start, stop, dtypes.canonicalize_dtype(float_))
  dtype = _jnp_dtype(dtype)
  computation_dtype = promote_types(dtype, dtypes.canonicalize_dtype(float_))
  _check_arraylike("logspace", start, stop)
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)
  lin = linspace(start, stop, num,
                 endpoint=endpoint, retstep=False, dtype=None, axis=axis)
  return lax.convert_element_type(power(base, lin), dtype)


@_wraps(np.geomspace)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis: int = 0):
  num = core.concrete_or_error(operator.index, num, "'num' argument of jnp.geomspace")
  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.geomspace")
  return _geomspace(start, stop, int(num), endpoint, dtype,
                    operator.index(axis))

@partial(jit, static_argnames=('num', 'endpoint', 'dtype', 'axis'))
def _geomspace(start, stop, num=50, endpoint=True, dtype=None, axis: int = 0):
  """Implementation of geomspace differentiable in start and stop args."""
  lax_internal._check_user_dtype_supported(dtype, "geomspace")
  if dtype is None:
    dtype = result_type(start, stop, dtypes.canonicalize_dtype(float_))
  dtype = _jnp_dtype(dtype)
  computation_dtype = promote_types(dtype, dtypes.canonicalize_dtype(float_))
  _check_arraylike("geomspace", start, stop)
  start = asarray(start, dtype=computation_dtype)
  stop = asarray(stop, dtype=computation_dtype)
  # follow the numpy geomspace convention for negative and complex endpoints
  signflip = 1 - (1 - sign(real(start))) * (1 - sign(real(stop))) // 2
  res = signflip * logspace(log10(signflip * start),
                            log10(signflip * stop), num,
                            endpoint=endpoint, base=10.0,
                            dtype=computation_dtype, axis=0)
  if axis != 0:
    res = moveaxis(res, 0, axis)
  return lax.convert_element_type(res, dtype)


@_wraps(np.meshgrid, lax_description=_ARRAY_VIEW_DOC)
def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
  _check_arraylike("meshgrid", *xi)
  args = [asarray(x) for x in xi]
  if not copy:
    raise ValueError("jax.numpy.meshgrid only supports copy=True")
  if indexing not in ["xy", "ij"]:
    raise ValueError(f"Valid values for indexing are 'xy' and 'ij', got {indexing}")
  if _any(a.ndim != 1 for a in args):
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


@_wraps(np.i0)
@jit
def i0(x):
  x_orig = x
  x, = _promote_args_inexact("i0", x)
  if not issubdtype(x.dtype, np.floating):
    raise ValueError(f"Unsupported input type to jax.numpy.i0: {_dtype(x_orig)}")
  x = lax.abs(x)
  return lax.mul(lax.exp(x), lax.bessel_i0e(x))


@_wraps(np.ix_)
def ix_(*args):
  _check_arraylike("ix", *args)
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


@_wraps(np.indices)
def indices(dimensions, dtype=int32, sparse=False):
  dimensions = tuple(
      core.concrete_or_error(int, d, "dimensions argument of jnp.indices")
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


_TOTAL_REPEAT_LENGTH_DOC = """\
Jax adds the optional `total_repeat_length` parameter which specifies the total
number of repeat, and defaults to sum(repeats). It must be specified for repeat
to be compilable. If `sum(repeats)` is larger than the specified
`total_repeat_length` the remaining values will be discarded. In the case of
`sum(repeats)` being smaller than the specified target length, the final value
will be repeated.
"""


@_wraps(np.repeat, lax_description=_TOTAL_REPEAT_LENGTH_DOC)
def repeat(a, repeats, axis: Optional[int] = None, *, total_repeat_length=None):
  _check_arraylike("repeat", a, repeats)

  if axis is None:
    a = ravel(a)
    axis = 0

  axis = core.concrete_or_error(operator.index, axis, "'axis' argument of jnp.repeat()")
  assert isinstance(axis, int)  # to appease mypy

  # If total_repeat_length is not given, can't compile, use a default.
  if total_repeat_length is None:
    repeats = core.concrete_or_error(np.array, repeats,
      "When jit-compiling jnp.repeat, the total number of repeats must be static. "
      "To fix this, either specify a static value for `repeats`, or pass a static "
      "value to `total_repeat_length`.")

    # Fast path for when repeats is a scalar.
    if np.ndim(repeats) == 0 and ndim(a) != 0:
      input_shape = a.shape
      aux_axis = axis if axis < 0 else axis + 1
      a = expand_dims(a, aux_axis)
      reps = [1] * len(a.shape)
      reps[aux_axis] = repeats
      a = tile(a, reps)
      result_shape = list(input_shape)
      result_shape[axis] *= repeats
      return reshape(a, result_shape)

    repeats = np.ravel(repeats)
    if ndim(a) != 0:
      repeats = np.broadcast_to(repeats, [a.shape[axis]])
    total_repeat_length = np.sum(repeats)
  else:
    repeats = ravel(repeats)
    if ndim(a) != 0:
      repeats = broadcast_to(repeats, [a.shape[axis]])

  # Special case when a is a scalar.
  if ndim(a) == 0:
    if repeats.shape == (1,):
      return full([total_repeat_length], a)
    else:
      raise ValueError('`repeat` with a scalar parameter `a` is only '
      'implemented for scalar values of the parameter `repeats`.')

  # Special case if total_repeat_length is zero.
  if total_repeat_length == 0:
    result_shape = list(a.shape)
    result_shape[axis] = 0
    return reshape(array([], dtype=a.dtype), result_shape)

  # If repeats is on a zero sized axis, then return the array.
  if a.shape[axis] == 0:
    return a

  # This implementation of repeat avoid having to instantiate a large.
  # intermediate tensor.

  # Modify repeats from e.g. [1,2,0,5] -> [0,1,2,0] for exclusive repeat.
  exclusive_repeats = roll(repeats, shift=1).at[0].set(0)
  # Cumsum to get indices of new number in repeated tensor, e.g. [0, 1, 3, 3]
  scatter_indices = cumsum(exclusive_repeats)
  # Scatter these onto a zero buffer, e.g. [1,1,0,2,0,0,0,0]
  block_split_indicators = zeros([total_repeat_length], dtype=int32)
  block_split_indicators = block_split_indicators.at[scatter_indices].add(1)
  # Cumsum again to get scatter indices for repeat, e.g. [0,1,1,3,3,3,3,3]
  gather_indices = cumsum(block_split_indicators) - 1
  return take(a, gather_indices, axis=axis)


@_wraps(np.tri)
def tri(N, M=None, k=0, dtype=None):
  lax_internal._check_user_dtype_supported(dtype, "tri")
  M = M if M is not None else N
  dtype = dtype or float32
  return lax_internal._tri(dtype, (N, M), k)


@_wraps(np.tril)
@partial(jit, static_argnames=('k',))
def tril(m, k=0):
  _check_arraylike("tril", m)
  m_shape = shape(m)
  if len(m_shape) < 2:
    raise ValueError("Argument to jax.numpy.tril must be at least 2D")
  mask = tri(*m_shape[-2:], k=k, dtype=bool)
  return lax.select(lax.broadcast(mask, m_shape[:-2]), m, zeros_like(m))


@_wraps(np.triu, update_doc=False)
@partial(jit, static_argnames=('k',))
def triu(m, k=0):
  _check_arraylike("triu", m)
  m_shape = shape(m)
  if len(m_shape) < 2:
    raise ValueError("Argument to jax.numpy.triu must be at least 2D")
  mask = tri(*m_shape[-2:], k=k - 1, dtype=bool)
  return lax.select(lax.broadcast(mask, m_shape[:-2]), zeros_like(m), m)


@_wraps(np.trace, skip_params=['out'])
@partial(jit, static_argnames=('offset', 'axis1', 'axis2', 'dtype'))
def trace(a, offset=0, axis1: int = 0, axis2: int = 1, dtype=None, out=None):
  _check_arraylike("trace", a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.trace is not supported.")
  lax_internal._check_user_dtype_supported(dtype, "trace")

  axis1 = _canonicalize_axis(axis1, ndim(a))
  axis2 = _canonicalize_axis(axis2, ndim(a))

  a_shape = shape(a)
  if dtype is None:
    dtype = _dtype(a)
    if issubdtype(dtype, integer):
      default_int = dtypes.canonicalize_dtype(np.int_)
      if iinfo(dtype).bits < iinfo(default_int).bits:
        dtype = default_int

  # Move the axis? dimensions to the end.
  perm = [i for i in range(len(a_shape)) if i != axis1 and i != axis2]
  perm = perm + [axis1, axis2]
  a = lax.transpose(a, perm)

  # Mask out the diagonal and reduce.
  a = where(eye(a_shape[axis1], a_shape[axis2], k=offset, dtype=bool),
            a, zeros_like(a))
  return sum(a, axis=(-2, -1), dtype=dtype)


def _wrap_indices_function(f):
  @_wraps(f, update_doc=False)
  def wrapper(*args, **kwargs):
    args = [core.concrete_or_error(
              None, arg, f"argument {i} of jnp.{f.__name__}()")
            for i, arg in enumerate(args)]
    kwargs = {key: core.concrete_or_error(
                None, val, f"argument '{key}' of jnp.{f.__name__}()")
              for key, val in kwargs.items()}
    return tuple(asarray(x) for x in f(*args, **kwargs))
  return wrapper

tril_indices = _wrap_indices_function(np.tril_indices)
triu_indices = _wrap_indices_function(np.triu_indices)
mask_indices = _wrap_indices_function(np.mask_indices)


@_wraps(np.triu_indices_from)
def triu_indices_from(arr, k=0):
  return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])


@_wraps(np.tril_indices_from)
def tril_indices_from(arr, k=0):
  return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])


@_wraps(np.diag_indices)
def diag_indices(n, ndim=2):
  n = core.concrete_or_error(operator.index, n, "'n' argument of jnp.diag_indices()")
  ndim = core.concrete_or_error(operator.index, ndim, "'ndim' argument of jnp.diag_indices()")
  if n < 0:
    raise ValueError("n argument to diag_indices must be nonnegative, got {}"
                     .format(n))
  if ndim < 0:
    raise ValueError("ndim argument to diag_indices must be nonnegative, got {}"
                     .format(ndim))
  return (lax.iota(int_, n),) * ndim

@_wraps(np.diag_indices_from)
def diag_indices_from(arr):
  _check_arraylike("diag_indices_from", arr)
  if not arr.ndim >= 2:
    raise ValueError("input array must be at least 2-d")

  if len(set(arr.shape)) != 1:
    raise ValueError("All dimensions of input must be of equal length")

  return diag_indices(arr.shape[0], ndim=arr.ndim)

@_wraps(np.diagonal, lax_description=_ARRAY_VIEW_DOC)
@partial(jit, static_argnames=('offset', 'axis1', 'axis2'))
def diagonal(a, offset=0, axis1: int = 0, axis2: int = 1):
  _check_arraylike("diagonal", a)
  a_shape = shape(a)
  a_ndims = len(a_shape)
  offset = core.concrete_or_error(operator.index, offset, "'offset' argument of jnp.diagonal()")

  # Move the two dimensions to the end.
  axis1 = _canonicalize_axis(axis1, a_ndims)
  axis2 = _canonicalize_axis(axis2, a_ndims)
  perm = [i for i in range(a_ndims) if i != axis1 and i != axis2]
  perm = perm + [axis1, axis2]
  a = lax.transpose(a, perm)

  diag_size = _max(0, _min(a_shape[axis1] + _min(offset, 0),
                           a_shape[axis2] - _max(offset, 0)))
  i = arange(diag_size)
  j = arange(_abs(offset), _abs(offset) + diag_size)
  return a[..., i, j] if offset >= 0 else a[..., j, i]


@_wraps(np.diag, lax_description=_ARRAY_VIEW_DOC)
def diag(v, k=0):
  return _diag(v, int(k))

@partial(jit, static_argnames=('k',))
def _diag(v, k):
  _check_arraylike("diag", v)
  v_shape = shape(v)
  if len(v_shape) == 1:
    zero = lambda x: lax.full_like(x, shape=(), fill_value=0)
    n = v_shape[0] + _abs(k)
    v = lax.pad(v, zero(v), ((_max(0, k), _max(0, -k), 0),))
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

@_wraps(np.diagflat, lax_description=_SCALAR_VALUE_DOC)
def diagflat(v, k=0):
  _check_arraylike("diagflat", v)
  v = ravel(v)
  v_length = len(v)
  adj_length = v_length + _abs(k)
  res = zeros(adj_length*adj_length, dtype=v.dtype)
  i = arange(0, adj_length-_abs(k))
  if (k >= 0):
    fi = i+k+i*adj_length
  else:
    fi = i+(i-k)*adj_length
  res = res.at[fi].set(v)
  res = res.reshape(adj_length, adj_length)
  return res


@_wraps(np.trim_zeros)
def trim_zeros(filt, trim='fb'):
  filt = core.concrete_or_error(asarray, filt,
    "Error arose in the `filt` argument of trim_zeros()")
  nz = (filt == 0)
  if all(nz):
    return empty(0, _dtype(filt))
  start = argmin(nz) if 'f' in trim.lower() else 0
  end = argmin(nz[::-1]) if 'b' in trim.lower() else 0
  return filt[start:len(filt) - end]


@_wraps(np.append)
@partial(jit, static_argnames=('axis',))
def append(arr, values, axis: Optional[int] = None):
  if axis is None:
    return concatenate([ravel(arr), ravel(values)], 0)
  else:
    return concatenate([arr, values], axis=axis)


@_wraps(np.delete)
def delete(arr, obj, axis=None):
  _check_arraylike("delete", arr)
  if axis is None:
    arr = ravel(arr)
    axis = 0
  axis = _canonicalize_axis(axis, arr.ndim)

  # Case 1: obj is a static integer.
  try:
    obj = operator.index(obj)
    obj = _canonicalize_axis(obj, arr.shape[axis])
  except TypeError:
    pass
  else:
    idx = tuple(slice(None) for i in range(axis))
    return concatenate([arr[idx + (slice(0, obj),)], arr[idx + (slice(obj + 1, None),)]], axis=axis)

  # Case 2: obj is a static slice.
  if isinstance(obj, slice):
    # TODO(jakevdp): we should be able to do this dynamically with care.
    indices = np.delete(np.arange(arr.shape[axis]), obj)
    return take(arr, indices, axis=axis)

  # Case 3: obj is an array
  # NB: pass both arrays to check for appropriate error message.
  _check_arraylike("delete", arr, obj)
  obj = core.concrete_or_error(np.asarray, obj, "'obj' array argument of jnp.delete()")

  if issubdtype(obj.dtype, integer):
    # TODO(jakevdp): in theory this could be done dynamically if obj has no duplicates,
    # but this would require the complement of lax.gather.
    mask = np.ones(arr.shape[axis], dtype=bool)
    mask[obj] = False
  elif obj.dtype == bool:
    if obj.shape != (arr.shape[axis],):
      raise ValueError("np.delete(arr, obj): for boolean indices, obj must be one-dimensional "
                       "with length matching specified axis.")
    mask = ~obj
  else:
    raise ValueError(f"np.delete(arr, obj): got obj.dtype={obj.dtype}; must be integer or bool.")
  return arr[tuple(slice(None) for i in range(axis)) + (mask,)]

@_wraps(np.insert)
def insert(arr, obj, values, axis=None):
  _check_arraylike("insert", arr, 0 if isinstance(obj, slice) else obj, values)
  arr = asarray(arr)
  values = asarray(values)

  if axis is None:
    arr = ravel(arr)
    axis = 0
  axis = core.concrete_or_error(None, axis, "axis argument of jnp.insert()")
  axis = _canonicalize_axis(axis, arr.ndim)
  if isinstance(obj, slice):
    indices = arange(*obj.indices(arr.shape[axis]))
  else:
    indices = asarray(obj)

  if indices.ndim > 1:
    raise ValueError("jnp.insert(): obj must be a slice, a one-dimensional "
                     f"array, or a scalar; got {obj}")
  if not np.issubdtype(indices.dtype, np.integer):
    if indices.size == 0 and not isinstance(obj, ndarray):
      indices = indices.astype(int)
    else:
      # Note: np.insert allows boolean inputs but the behavior is deprecated.
      raise ValueError("jnp.insert(): index array must be "
                       f"integer typed; got {obj}")
  values = array(values, ndmin=arr.ndim, dtype=arr.dtype, copy=False)

  if indices.size == 1:
    index = ravel(indices)[0]
    if indices.ndim == 0:
      values = moveaxis(values, 0, axis)
    indices = full(values.shape[axis], index)
  n_input = arr.shape[axis]
  n_insert = broadcast_shapes(indices.shape, values.shape[axis])[0]
  out_shape = list(arr.shape)
  out_shape[axis] += n_insert
  out = zeros_like(arr, shape=tuple(out_shape))

  indices = where(indices < 0, indices + n_input, indices)
  indices = clip(indices, 0, n_input)

  values_ind = indices.at[argsort(indices)].add(arange(n_insert))
  arr_mask = ones(n_input + n_insert, dtype=bool).at[values_ind].set(False)
  arr_ind = where(arr_mask, size=n_input)[0]

  out = out.at[(slice(None),) * axis + (values_ind,)].set(values)
  out = out.at[(slice(None),) * axis + (arr_ind,)].set(arr)

  return out


@_wraps(np.apply_along_axis)
def apply_along_axis(func1d, axis: int, arr, *args, **kwargs):
  num_dims = ndim(arr)
  axis = _canonicalize_axis(axis, num_dims)
  func = lambda arr: func1d(arr, *args, **kwargs)
  for i in range(1, num_dims - axis):
    func = jax.vmap(func, in_axes=i, out_axes=-1)
  for i in range(axis):
    func = jax.vmap(func, in_axes=0, out_axes=0)
  return func(arr)


@_wraps(np.apply_over_axes)
def apply_over_axes(func, a, axes):
  for axis in axes:
    b = func(a, axis=axis)
    if b.ndim == a.ndim:
      a = b
    elif b.ndim == a.ndim - 1:
      a = expand_dims(b, axis)
    else:
      raise ValueError("function is not returning an array of the correct shape")
  return a


### Tensor contraction operations


@_wraps(np.dot, lax_description=_PRECISION_DOC)
@partial(jit, static_argnames=('precision',), inline=True)
def dot(a, b, *, precision=None):  # pylint: disable=missing-docstring
  _check_arraylike("dot", a, b)
  a, b = _promote_dtypes(a, b)
  a_ndim, b_ndim = ndim(a), ndim(b)
  if a_ndim == 0 or b_ndim == 0:
    return lax.mul(a, b)
  if _max(a_ndim, b_ndim) <= 2:
    return lax.dot(a, b, precision=precision)

  if b_ndim == 1:
    contract_dims = ((a_ndim - 1,), (0,))
  else:
    contract_dims = ((a_ndim - 1,), (b_ndim - 2,))
  batch_dims = ((), ())
  return lax.dot_general(a, b, (contract_dims, batch_dims), precision)


@_wraps(np.matmul, lax_description=_PRECISION_DOC)
@partial(jit, static_argnames=('precision',), inline=True)
def matmul(a, b, *, precision=None):  # pylint: disable=missing-docstring
  _check_arraylike("matmul", a, b)
  for i, x in enumerate((a, b)):
    if ndim(x) < 1:
      msg = (f"matmul input operand {i} must have ndim at least 1, "
             f"but it has ndim {ndim(x)}")
      raise ValueError(msg)

  a, b = _promote_dtypes(a, b)

  a_is_mat, b_is_mat = (ndim(a) > 1), (ndim(b) > 1)
  a_batch_dims = shape(a)[:-2] if a_is_mat else ()
  b_batch_dims = shape(b)[:-2] if b_is_mat else ()
  num_batch_dims = _max(len(a_batch_dims), len(b_batch_dims))
  a_batch_dims = (None,) * (num_batch_dims - len(a_batch_dims)) + a_batch_dims
  b_batch_dims = (None,) * (num_batch_dims - len(b_batch_dims)) + b_batch_dims

  # Dimensions to squeeze from the inputs.
  a_squeeze = []
  b_squeeze = []

  # Positions of batch dimensions in squeezed inputs.
  a_batch = []
  b_batch = []

  # Desired index in final output of each kind of dimension, in the order that
  # lax.dot_general will emit them.
  idx_batch = []
  idx_a_other = []  # other = non-batch, non-contracting.
  idx_b_other = []
  for i, (ba, bb) in enumerate(zip(a_batch_dims, b_batch_dims)):
    if ba is None:
      idx_b_other.append(i)
    elif bb is None:
      idx_a_other.append(i)
    elif core.symbolic_equal_dim(ba, 1):
      idx_b_other.append(i)
      a_squeeze.append(len(idx_batch) + len(idx_a_other) + len(a_squeeze))
    elif core.symbolic_equal_dim(bb, 1):
      idx_a_other.append(i)
      b_squeeze.append(len(idx_batch) + len(idx_b_other) + len(b_squeeze))
    elif core.symbolic_equal_dim(ba, bb):
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
    precision=precision)
  return lax.transpose(out, perm)


@_wraps(np.vdot, lax_description=_PRECISION_DOC)
@partial(jit, static_argnames=('precision',), inline=True)
def vdot(a, b, *, precision=None):
  _check_arraylike("vdot", a, b)
  if issubdtype(_dtype(a), complexfloating):
    a = conj(a)
  return dot(a.ravel(), b.ravel(), precision=precision)


@_wraps(np.tensordot, lax_description=_PRECISION_DOC)
def tensordot(a, b, axes=2, *, precision=None):
  _check_arraylike("tensordot", a, b)
  a_ndim = ndim(a)
  b_ndim = ndim(b)

  a, b = _promote_dtypes(a, b)
  if type(axes) is int:
    if axes > _min(a_ndim, b_ndim):
      msg = "Number of tensordot axes (axes {}) exceeds input ranks ({} and {})"
      raise TypeError(msg.format(axes, a.shape, b.shape))
    contracting_dims = tuple(range(a_ndim - axes, a_ndim)), tuple(range(axes))
  elif type(axes) in (list, tuple) and len(axes) == 2:
    ax1, ax2 = axes
    if type(ax1) == type(ax2) == int:
      contracting_dims = ((_canonicalize_axis(ax1, a_ndim),),
                          (_canonicalize_axis(ax2, b_ndim),))
    elif type(ax1) in (list, tuple) and type(ax2) in (list, tuple):
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
  return lax.dot_general(a, b, (contracting_dims, ((), ())),
                         precision=precision)


_EINSUM_DOC = _PRECISION_DOC + """\
A tuple ``precision`` does not necessarily map to multiple arguments of ``einsum()``;
rather, the specified ``precision`` is forwarded to each ``dot_general`` call used in
the implementation.
"""

@_wraps(np.einsum, lax_description=_EINSUM_DOC, skip_params=['out'])
def einsum(*operands, out=None, optimize='optimal', precision=None,
           _use_xeinsum=False):
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.einsum is not supported.")

  if (_use_xeinsum or isinstance(operands[0], str) and '{' in operands[0] and
      len(operands[1:]) == 2):
    return lax.xeinsum(*operands)

  optimize = 'optimal' if optimize is True else optimize
  # using einsum_call=True here is an internal api for opt_einsum

  # Allow handling of shape polymorphism
  non_constant_dim_types = {
      type(d) for op in operands if not isinstance(op, str)
      for d in np.shape(op) if not core.is_constant_dim(d)
  }
  if not non_constant_dim_types:
    einsum_contract_path_fn = opt_einsum.contract_path
  else:
    einsum_contract_path_fn = _polymorphic_einsum_contract_path_handlers[next(iter(non_constant_dim_types))]
  operands, contractions = einsum_contract_path_fn(
        *operands, einsum_call=True, use_blas=True, optimize=optimize)

  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)
  return _einsum(operands, contractions, precision)

# Enable other modules to override einsum_contact_path.
# Indexed by the type of the non constant dimension
_polymorphic_einsum_contract_path_handlers = {}  # type: ignore

@_wraps(np.einsum_path)
def einsum_path(subscripts, *operands, optimize='greedy'):
  # using einsum_call=True here is an internal api for opt_einsum
  return opt_einsum.contract_path(subscripts, *operands, optimize=optimize)

def _removechars(s, chars):
  return s.translate(str.maketrans(dict.fromkeys(chars)))

@partial(jit, static_argnums=(1, 2))
def _einsum(operands: Sequence,
            contractions: Sequence[Tuple[Tuple[int, ...], FrozenSet[str], str]],
            precision):
  operands = list(_promote_dtypes(*operands))
  def sum(x, axes):
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
        eye = lax_internal._delta(operand.dtype, operand.shape, axes)
        if name not in keep_names:
          operand = sum(operand * eye, axes)
          names = names.replace(name, '')
        else:
          operand = sum(operand * eye, axes[:-1])
          names = names.replace(name, '', count - 1)
    return operand, names

  def filter_singleton_dims(operand, names, other_shape, other_names):
    s = shape(operand)
    new_shape = []
    new_names = []
    for i, d in enumerate(names):
      other_i = other_names.find(d)
      if not core.symbolic_equal_dim(s[i], 1) or other_i == -1 or core.symbolic_equal_dim(other_shape[other_i], 1):
        new_shape.append(s[i])
        new_names.append(d)
    return reshape(operand, tuple(new_shape)), "".join(new_names)

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
      assert _all(
        name in lhs_names and name in rhs_names and
        lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
        for name in contracted_names)

      # contract using lax.dot_general
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
        operand = lax.dot_general(rhs, lhs, dimension_numbers, precision)
      else:
        names = batch_names_str + remaining_lhs_names + remaining_rhs_names
        dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
        operand = lax.dot_general(lhs, rhs, dimension_numbers, precision)
    else:
      raise NotImplementedError  # if this is actually reachable, open an issue!

    # the resulting 'operand' with axis labels 'names' should be a permutation
    # of the desired result
    assert len(names) == len(result_names) == len(set(names))
    assert set(names) == set(result_names)
    if names != result_names:
      perm = tuple([names.index(name) for name in result_names])
      operand = lax.transpose(operand, perm)
    operands.append(operand)  # used in next iteration

  return operands[0]


def _movechars(s, src, dst):
  """Helper for einsum string munging, like moveaxis on identifier strings."""
  chars = [c for i, c in enumerate(s) if i not in src]
  for i, j in sorted(zip(dst, src)):
    chars.insert(i, s[j])
  return ''.join(chars)


@_wraps(np.inner, lax_description=_PRECISION_DOC)
@partial(jit, static_argnames=('precision',), inline=True)
def inner(a, b, *, precision=None):
  if ndim(a) == 0 or ndim(b) == 0:
    return a * b
  return tensordot(a, b, (-1, -1), precision=precision)


@_wraps(np.outer, skip_params=['out'])
@partial(jit, inline=True)
def outer(a, b, out=None):
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.outer is not supported.")
  a, b = _promote_dtypes(a, b)
  return ravel(a)[:, None] * ravel(b)[None, :]

@_wraps(np.cross)
@partial(jit, static_argnames=('axisa', 'axisb', 'axisc', 'axis'))
def cross(a, b, axisa: int = -1, axisb: int = -1, axisc: int = -1,
          axis: Optional[int] = None):
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


@_wraps(np.kron)
@jit
def kron(a, b):
  a, b = _promote_dtypes(a, b)
  if ndim(a) < ndim(b):
    a = expand_dims(a, range(ndim(b) - ndim(a)))
  elif ndim(b) < ndim(a):
    b = expand_dims(b, range(ndim(a) - ndim(b)))
  a_reshaped = expand_dims(a, range(1, 2 * ndim(a), 2))
  b_reshaped = expand_dims(b, range(0, 2 * ndim(b), 2))
  out_shape = tuple(np.multiply(shape(a), shape(b)))
  return reshape(lax.mul(a_reshaped, b_reshaped), out_shape)


@_wraps(np.vander)
@partial(jit, static_argnames=('N', 'increasing'))
def vander(x, N=None, increasing=False):
  _check_arraylike("vander", x)
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

  return power(x[..., None], expand_dims(iota, tuple(range(x.ndim))))


### Misc

_ARGWHERE_DOC = """\
Because the size of the output of ``argwhere`` is data-dependent, the function is not
typically compatible with JIT. The JAX version adds the optional ``size`` argument, which
specifies the size of the leading dimension of the output - it must be specified statically
for ``jnp.argwhere`` to be compiled with non-static operands. If ``size`` is specified,
the indices of the first ``size`` True elements will be returned; if there are fewer
nonzero elements than `size` indicates, the index arrays will be zero-padded.
"""

@_wraps(np.argwhere,
  lax_description=_dedent("""
    Because the size of the output of ``argwhere`` is data-dependent, the function is not
    typically compatible with JIT. The JAX version adds the optional ``size`` argument which
    must be specified statically for ``jnp.argwhere`` to be used within some of JAX's
    transformations."""),
  extra_params=_dedent("""
    size : int, optional
        If specified, the indices of the first ``size`` True elements will be returned. If there
        are fewer results than ``size`` indicates, the return value will be padded with ``fill_value``.
    fill_value : array_like, optional
        When ``size`` is specified and there are fewer than the indicated number of elements, the
        remaining elements will be filled with ``fill_value``, which defaults to zero."""))
def argwhere(a, *, size=None, fill_value=None):
  result = transpose(vstack(nonzero(a, size=size, fill_value=fill_value)))
  if ndim(a) == 0:
    return result[:0].reshape(result.shape[0], 0)
  return result.reshape(result.shape[0], ndim(a))


@_wraps(np.argmax, skip_params=['out'])
def argmax(a, axis: Optional[int] = None, out=None, keepdims=None):
  return _argmax(a, None if axis is None else operator.index(axis), keepdims=bool(keepdims))

@partial(jit, static_argnames=('axis', 'keepdims'), inline=True)
def _argmax(a, axis: Optional[int] = None, out=None, keepdims=False):
  _check_arraylike("argmax", a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.argmax is not supported.")
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

@_wraps(np.argmin, skip_params=['out'])
def argmin(a, axis: Optional[int] = None, out=None, keepdims=None):
  return _argmin(a, None if axis is None else operator.index(axis), keepdims=bool(keepdims))

@partial(jit, static_argnames=('axis', 'keepdims'), inline=True)
def _argmin(a, axis: Optional[int] = None, out=None, keepdims=False):
  _check_arraylike("argmin", a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.argmin is not supported.")
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

@_wraps(np.nanargmax, lax_description=_NANARG_DOC.format("max"), skip_params=['out'])
def nanargmax(a, axis: Optional[int] = None, out : Any = None, keepdims : Optional[bool] = None):
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.nanargmax is not supported.")
  return _nanargmax(a, None if axis is None else operator.index(axis), keepdims=bool(keepdims))

@partial(jit, static_argnames=('axis', 'keepdims'))
def _nanargmax(a, axis: Optional[int] = None, keepdims: bool = False):
  _check_arraylike("nanargmax", a)
  if not issubdtype(_dtype(a), inexact):
    return argmax(a, axis=axis, keepdims=keepdims)
  nan_mask = isnan(a)
  a = where(nan_mask, -inf, a)
  res = argmax(a, axis=axis, keepdims=keepdims)
  return where(all(nan_mask, axis=axis, keepdims=keepdims), -1, res)

@_wraps(np.nanargmin, lax_description=_NANARG_DOC.format("min"),  skip_params=['out'])
def nanargmin(a, axis: Optional[int] = None, out : Any = None, keepdims : Optional[bool] = None):
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.nanargmin is not supported.")
  return _nanargmin(a, None if axis is None else operator.index(axis), keepdims=bool(keepdims))

@partial(jit, static_argnames=('axis', 'keepdims'))
def _nanargmin(a, axis: Optional[int] = None, keepdims : bool = False):
  _check_arraylike("nanargmin", a)
  if not issubdtype(_dtype(a), inexact):
    return argmin(a, axis=axis, keepdims=keepdims)
  nan_mask = isnan(a)
  a = where(nan_mask, inf, a)
  res = argmin(a, axis=axis, keepdims=keepdims)
  return where(all(nan_mask, axis=axis, keepdims=keepdims), -1, res)


@_wraps(np.sort)
@partial(jit, static_argnames=('axis', 'kind', 'order'))
def sort(a, axis: Optional[int] = -1, kind='quicksort', order=None):
  _check_arraylike("sort", a)
  if kind != 'quicksort':
    warnings.warn("'kind' argument to sort is ignored.")
  if order is not None:
    raise ValueError("'order' argument to sort is not supported.")

  if axis is None:
    return lax.sort(a.ravel(), dimension=0)
  else:
    return lax.sort(a, dimension=_canonicalize_axis(axis, ndim(a)))

@_wraps(np.sort_complex)
@jit
def sort_complex(a):
  _check_arraylike("sort_complex", a)
  a = lax.sort(a, dimension=0)
  return lax.convert_element_type(a, result_type(a, dtypes.canonicalize_dtype(complex_)))

@_wraps(np.lexsort)
@partial(jit, static_argnames=('axis',))
def lexsort(keys, axis=-1):
  keys = tuple(keys)
  if len(keys) == 0:
    raise TypeError("need sequence of keys with len > 0 in lexsort")
  if len({shape(key) for key in keys}) > 1:
    raise ValueError("all keys need to be the same shape")
  if ndim(keys[0]) == 0:
    return array(0, dtype=dtypes.canonicalize_dtype(int_))
  axis = _canonicalize_axis(axis, ndim(keys[0]))
  use_64bit_index = keys[0].shape[axis] >= (1 << 31)
  iota = lax.broadcasted_iota(int64 if use_64bit_index else int_, shape(keys[0]), axis)
  return lax.sort((*keys[::-1], iota), dimension=axis, num_keys=len(keys))[-1]


_ARGSORT_DOC = """
Only :code:`kind='stable'` is supported. Other :code:`kind` values will produce
a warning and be treated as if they were :code:`'stable'`.
"""

@_wraps(np.argsort, lax_description=_ARGSORT_DOC)
@partial(jit, static_argnames=('axis', 'kind', 'order'))
def argsort(a, axis: Optional[int] = -1, kind='stable', order=None):
  _check_arraylike("argsort", a)
  if kind != 'stable':
    warnings.warn("'kind' argument to argsort is ignored; only 'stable' sorts "
                  "are supported.")
  if order is not None:
    raise ValueError("'order' argument to argsort is not supported.")

  if axis is None:
    return argsort(a.ravel(), 0)
  else:
    axis_num = _canonicalize_axis(axis, ndim(a))
    use_64bit_index = a.shape[axis_num] >= (1 << 31)
    iota = lax.broadcasted_iota(int64 if use_64bit_index else int_, shape(a), axis_num)
    _, perm = lax.sort_key_val(a, iota, dimension=axis_num)
    return perm


@_wraps(np.msort)
def msort(a):
  return sort(a, axis=0)


@partial(jit, static_argnums=(2,))
def _roll(a, shift, axis):
  a_shape = shape(a)
  if axis is None:
    return lax.reshape(_roll(ravel(a), shift, axis=0), a_shape)
  shift = asarray(shift)
  a_ndim = len(a_shape)
  axis = np.asarray(axis)
  b_shape = lax.broadcast_shapes(shift.shape, axis.shape, (1,))
  if len(b_shape) != 1:
    msg = "'shift' and 'axis' arguments to roll must be scalars or 1D arrays"
    raise ValueError(msg)

  for x, i in zip(broadcast_to(shift, b_shape),
                  np.broadcast_to(axis, b_shape)):
    i = _canonicalize_axis(i, a_ndim)
    x = remainder(x, (a_shape[i] or 1))
    a = lax.concatenate((a, a), i)
    a = lax.dynamic_slice_in_dim(a, a_shape[i] - x, a_shape[i], axis=i)
  return a


@_wraps(np.roll)
def roll(a, shift, axis: Optional[Union[int, Sequence[int]]] = None):
  _check_arraylike("roll", a,)
  if isinstance(axis, list):
    axis = tuple(axis)
  return _roll(a, shift, axis)


@_wraps(np.rollaxis, lax_description=_ARRAY_VIEW_DOC)
@partial(jit, static_argnames=('axis', 'start'))
def rollaxis(a, axis: int, start=0):
  _check_arraylike("rollaxis", a)
  start = core.concrete_or_error(operator.index, start, "'start' argument of jnp.rollaxis()")
  a_ndim = ndim(a)
  axis = _canonicalize_axis(axis, a_ndim)
  if not (-a_ndim <= start <= a_ndim):
    raise ValueError(f"start={start} must satisfy {-a_ndim}<=start<={a_ndim}")
  if start < 0:
    start += a_ndim
  if start > axis:
    start -= 1
  return moveaxis(a, axis, start)


@_wraps(np.packbits)
@partial(jit, static_argnames=('axis', 'bitorder'))
def packbits(a, axis: Optional[int] = None, bitorder='big'):
  _check_arraylike("packbits", a)
  if not (issubdtype(_dtype(a), integer) or issubdtype(_dtype(a), bool_)):
    raise TypeError('Expected an input array of integer or boolean data type')
  if bitorder not in ['little', 'big']:
    raise ValueError("'order' must be either 'little' or 'big'")
  a = greater(a, 0).astype('uint8')
  bits = arange(8, dtype='uint8')
  if bitorder == 'big':
    bits = bits[::-1]
  if axis is None:
    a = ravel(a)
    axis = 0
  a = swapaxes(a, axis, -1)

  remainder = a.shape[-1] % 8
  if remainder:
    a = lax.pad(a, np.uint8(0),
                (a.ndim - 1) * [(0, 0, 0)] + [(0, 8 - remainder, 0)])

  a = a.reshape(a.shape[:-1] + (a.shape[-1] // 8, 8))
  bits = expand_dims(bits, tuple(range(a.ndim - 1)))
  packed = (a << bits).sum(-1).astype('uint8')
  return swapaxes(packed, axis, -1)


@_wraps(np.unpackbits)
@partial(jit, static_argnames=('axis', 'count', 'bitorder'))
def unpackbits(a, axis: Optional[int] = None, count=None, bitorder='big'):
  _check_arraylike("unpackbits", a)
  if _dtype(a) != uint8:
    raise TypeError("Expected an input array of unsigned byte data type")
  if bitorder not in ['little', 'big']:
    raise ValueError("'order' must be either 'little' or 'big'")
  bits = asarray(1) << arange(8, dtype='uint8')
  if bitorder == 'big':
    bits = bits[::-1]
  if axis is None:
    a = ravel(a)
    axis = 0
  a = swapaxes(a, axis, -1)
  unpacked = ((a[..., None] & expand_dims(bits, tuple(range(a.ndim)))) > 0).astype('uint8')
  unpacked = unpacked.reshape(unpacked.shape[:-2] + (-1,))[..., :count]
  return swapaxes(unpacked, axis, -1)


@_wraps(np.take, skip_params=['out'])
def take(a, indices, axis: Optional[int] = None, out=None, mode=None):
  return _take(a, indices, None if axis is None else operator.index(axis), out,
               mode)

@partial(jit, static_argnames=('axis', 'mode'))
def _take(a, indices, axis: Optional[int] = None, out=None, mode=None):
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.take is not supported.")
  _check_arraylike("take", a, indices)
  a = asarray(a)
  indices = asarray(indices)

  if axis is None:
    a = ravel(a)
    axis_idx = 0
  else:
    axis_idx = _canonicalize_axis(axis, ndim(a))

  if mode is None:
    # TODO(phawkins): change default mode to "fill" and delete this case.
    # lax.gather() does not support negative indices, so we wrap them here
    indices = where(indices < 0, indices + a.shape[axis_idx], indices)
    gather_mode = lax.GatherScatterMode.CLIP
  elif mode == "raise":
    # TODO(phawkins): we have no way to report out of bounds errors yet.
    raise NotImplementedError("The 'raise' mode to jnp.take is not supported.")
  elif mode == "wrap":
    indices = mod(indices, _lax_const(indices, a.shape[axis_idx]))
    gather_mode = lax.GatherScatterMode.PROMISE_IN_BOUNDS
  elif mode == "fill":
    # Undocumented non-standard mode corresponding to the fill_or_drop mode on
    # lax.gather()
    gather_mode = lax.GatherScatterMode.FILL_OR_DROP
    # lax.gather() does not support negative indices, so we wrap them here
    indices = where(indices < 0, indices + a.shape[axis_idx], indices)
  elif mode == "clip":
    gather_mode = lax.GatherScatterMode.CLIP
  else:
    raise ValueError("Invalid mode '{}' for np.take".format(mode))

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
                    mode=gather_mode)


def _normalize_index(index, axis_size):
  """Normalizes an index value in the range [-N, N) to the range [0, N)."""
  if core.is_constant_dim(axis_size):
    axis_size_val = _lax_const(index, axis_size)
  else:
    axis_size_val = lax.convert_element_type(core.dimension_as_value(axis_size),
                                             _dtype(index))
  return lax.select(
    lax.lt(index, _lax_const(index, 0)),
    lax.add(index, axis_size_val),
    index)

@_wraps(np.take_along_axis, update_doc=False)
@partial(jit, static_argnames=('axis',))
def take_along_axis(arr, indices, axis: Optional[int]):
  _check_arraylike("take_along_axis", arr, indices)
  if axis is None:
    if ndim(indices) != 1:
      msg = "take_along_axis indices must be 1D if axis=None, got shape {}"
      raise ValueError(msg.format(indices.shape))
    return take_along_axis(arr.ravel(), indices, 0)
  rank = ndim(arr)
  if rank != ndim(indices):
    msg = "indices and arr must have the same number of dimensions; {} vs. {}"
    raise ValueError(msg.format(ndim(indices), ndim(arr)))
  axis = _canonicalize_axis(axis, rank)

  def replace(tup, val):
    lst = list(tup)
    lst[axis] = val
    return tuple(lst)

  use_64bit_index = _any([not core.is_constant_dim(d) or d >= (1 << 31) for d in arr.shape])
  index_dtype = int64 if use_64bit_index else int32
  indices = lax.convert_element_type(indices, index_dtype)

  bcast_shape = lax.broadcast_shapes(replace(arr.shape, 1), replace(indices.shape, 1))
  indices = broadcast_to(indices, replace(bcast_shape, indices.shape[axis]))
  arr     = broadcast_to(arr,     replace(bcast_shape, arr.shape[axis]))

  axis_size = arr.shape[axis]
  arr_shape = replace(arr.shape, 1)
  idx_shape = indices.shape
  out_shape = lax.broadcast_shapes(idx_shape, arr_shape)

  index_dims = [i for i, idx in enumerate(idx_shape) if i == axis or not core.symbolic_equal_dim(idx, 1)]

  gather_index_shape = tuple(np.array(out_shape)[index_dims]) + (1,)
  gather_indices = []
  slice_sizes = []
  offset_dims = []
  start_index_map = []
  collapsed_slice_dims = []
  j = 0
  for i in range(rank):
    if i == axis:
      indices = _normalize_index(indices, axis_size)
      gather_indices.append(lax.reshape(indices, gather_index_shape))
      slice_sizes.append(1)
      start_index_map.append(i)
      collapsed_slice_dims.append(i)
      j += 1
    elif not core.symbolic_equal_dim(idx_shape[i], 1):
      # TODO(mattjj): next line needs updating for dynamic shapes
      iota = lax.iota(_dtype(indices), out_shape[i])  # type: ignore
      iota = lax.broadcast_in_dim(iota, gather_index_shape, (j,))
      gather_indices.append(iota)
      slice_sizes.append(1)
      start_index_map.append(i)
      collapsed_slice_dims.append(i)
      j += 1
    else:
      # If idx_shape[i] == 1, we can just take the entirety of the arr's axis
      # and avoid forming an iota index.
      offset_dims.append(i)
      slice_sizes.append(arr_shape[i])

  gather_indices = lax.concatenate(gather_indices, dimension=j)
  dnums = lax.GatherDimensionNumbers(
    offset_dims=tuple(offset_dims),
    collapsed_slice_dims=tuple(collapsed_slice_dims),
    start_index_map=tuple(start_index_map))
  return lax.gather(arr, gather_indices, dnums, tuple(slice_sizes))

### Indexing

def _rewriting_take(arr, idx, indices_are_sorted=False, unique_indices=False,
                    mode=None, fill_value=None):
  # Computes arr[idx].
  # All supported cases of indexing can be implemented as an XLA gather,
  # followed by an optional reverse and broadcast_in_dim.
  arr = asarray(arr) if not isinstance(arr, ndarray) else arr
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

_Indexer = collections.namedtuple("_Indexer", [
  # The expected shape of the slice output.
  "slice_shape",

  # The slice shape to pass to lax.gather().
  "gather_slice_shape",

  # The gather indices to use.
  "gather_indices",

  # A GatherDimensionNumbers object describing the gather to perform.
  "dnums",

  # Are the gather_indices known to be non-overlapping and/or sorted?
  # (In practice, these translate to "there no advanced indices", because
  # only advanced indices could lead to index repetition.)
  "unique_indices",
  "indices_are_sorted",

  # Slice dimensions that have negative strides, and so must be reversed after
  # the gather.
  "reversed_y_dims",

  # Keep track of any axes created by `newaxis`. These must be inserted for
  # gathers and eliminated for scatters.
  "newaxis_dims",
])

def _split_index_for_jit(idx, shape):
  """Splits indices into necessarily-static and dynamic parts.

  Used to pass indices into `jit`-ted function.
  """
  # Convert list indices to tuples in cases (deprecated by NumPy.)
  idx = _eliminate_deprecated_list_indexing(idx)

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

def _index_to_gather(x_shape, idx, normalize_indices=True):
  # Remove ellipses and add trailing slice(None)s.
  idx = _canonicalize_tuple_index(len(x_shape), idx)

  # Check for advanced indexing:
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing

  # Do the advanced indexing axes appear contiguously? If not, NumPy semantics
  # move the advanced axes to the front.
  advanced_axes_are_contiguous = False

  advanced_indexes = None

  # The positions of the advanced indexing axes in `idx`.
  idx_advanced_axes = []

  # The positions of the advanced indexes in x's shape.
  # collapsed, after None axes have been removed. See below.
  x_advanced_axes = None

  if _is_advanced_int_indexer(idx):
    idx_no_nones = [(i, d) for i, d in enumerate(idx) if d is not None]
    advanced_pairs = (
      (asarray(e), i, j) for j, (i, e) in enumerate(idx_no_nones)
      if isscalar(e) or isinstance(e, (Sequence, ndarray, np.ndarray)))
    if normalize_indices:
      advanced_pairs = ((_normalize_index(e, x_shape[j]), i, j)
                        for e, i, j in advanced_pairs)
    advanced_indexes, idx_advanced_axes, x_advanced_axes = zip(*advanced_pairs)
    advanced_axes_are_contiguous = np.all(np.diff(idx_advanced_axes) == 1)

  x_axis = 0  # Current axis in x.
  y_axis = 0  # Current axis in y, before collapsing. See below.
  collapsed_y_axis = 0  # Current axis in y, after collapsing.

  # Scatter dimension numbers.
  offset_dims = []
  collapsed_slice_dims = []
  start_index_map = []

  use_64bit_index = _any([not core.is_constant_dim(d) or d >= (1 << 31) for d in x_shape])
  index_dtype = int64 if use_64bit_index else int32

  # Gather indices.
  # Pairs of (array, start_dim) values. These will be broadcast into
  # gather_indices_shape, with the array dimensions aligned to start_dim, and
  # then concatenated.
  gather_indices = []
  gather_indices_shape = []

  # We perform three transformations to y before the scatter op, in order:
  # First, y is broadcast to slice_shape. In general `y` only need broadcast to
  # the right shape.
  slice_shape = []

  # Next, y is squeezed to remove newaxis_dims. This removes np.newaxis/`None`
  # indices, which the scatter cannot remove itself.
  newaxis_dims = []

  # Finally, we reverse reversed_y_dims to handle slices with negative strides.
  reversed_y_dims = []

  gather_slice_shape = []

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
      if core.symbolic_equal_dim(x_shape[x_axis], 0):
        # XLA gives error when indexing into an axis of size 0
        raise IndexError(f"index is out of bounds for axis {x_axis} with size 0")
      i = _normalize_index(i, x_shape[x_axis]) if normalize_indices else i
      i = lax.convert_element_type(i, index_dtype)
      gather_indices.append((i, len(gather_indices_shape)))
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
      # Normalize the slice to use None when possible
      start, stop, step = i.start, i.stop, i.step
      try:
        if ((step is None or core.symbolic_equal_dim(step, 1)) and
            stop is not None and core.symbolic_equal_dim(stop, x_shape[x_axis])):
          # The following is a useful special case with shape polymorphism
          stop = None
      except TypeError:
        pass

      # Handle slice(None)
      if start is None and stop is None and step is None:
        slice_shape.append(x_shape[x_axis])
        gather_slice_shape.append(x_shape[x_axis])
        offset_dims.append(collapsed_y_axis)
        collapsed_y_axis += 1
        y_axis += 1
        x_axis += 1
      # Handle slice index (only static, otherwise an error is raised)
      else:
        if not _all(_is_slice_element_none_or_constant(elt)
                    for elt in (start, stop, step)):
          msg = ("Array slice indices must have static start/stop/step to be used "
                 "with NumPy indexing syntax. "
                 f"Found slice({start}, {stop}, {step}). "
                 "To index a statically sized "
                 "array at a dynamic position, try lax.dynamic_slice/"
                 "dynamic_update_slice (JAX does not support dynamically sized "
                 "arrays within JIT compiled functions).")
          raise IndexError(msg)
        if not core.is_constant_dim(x_shape[x_axis]):
          msg = ("Cannot use NumPy slice indexing on an array dimension whose "
                 f"size is not statically known ({x_shape[x_axis]}). "
                 "Try using lax.dynamic_slice/dynamic_update_slice")
          raise IndexError(msg)
        start, limit, stride, needs_rev = _static_idx(slice(start, stop, step),
                                                      x_shape[x_axis])
        if needs_rev:
          reversed_y_dims.append(collapsed_y_axis)
        if stride == 1:
          i = lax.convert_element_type(start, index_dtype)
          gather_indices.append((i, len(gather_indices_shape)))
          slice_shape.append(limit - start)
          gather_slice_shape.append(limit - start)
          offset_dims.append(collapsed_y_axis)
          start_index_map.append(x_axis)
        else:
          i = arange(start, limit, stride, dtype=index_dtype)
          size = i.shape[0]
          slice_shape.append(size)
          gather_slice_shape.append(1)
          gather_indices.append((i, len(gather_indices_shape)))
          gather_indices_shape.append(size)

          start_index_map.append(x_axis)
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

      msg = "Indexing mode not yet supported. Open a feature request!\n{}"
      raise IndexError(msg.format(idx))

  if len(gather_indices) == 0:
    gather_indices_array = np.zeros((0,), dtype=index_dtype)
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
    indices_are_sorted=advanced_indexes is None)

def _should_unpack_list_index(x):
  """Helper for _eliminate_deprecated_list_indexing."""
  return (isinstance(x, (np.ndarray, ndarray)) and np.ndim(x) != 0
          or isinstance(x, (Sequence, slice))
          or x is Ellipsis or x is None)

def _eliminate_deprecated_list_indexing(idx):
  # "Basic slicing is initiated if the selection object is a non-array,
  # non-tuple sequence containing slice objects, [Ellipses, or newaxis
  # objects]". Detects this and raises a TypeError.
  if not isinstance(idx, tuple):
    if isinstance(idx, Sequence) and not isinstance(idx, (ndarray, np.ndarray)):
      # As of numpy 1.16, some non-tuple sequences of indices result in a warning, while
      # others are converted to arrays, based on a set of somewhat convoluted heuristics
      # (See https://github.com/numpy/numpy/blob/v1.19.2/numpy/core/src/multiarray/mapping.c#L179-L343)
      # In JAX, we raise an informative TypeError for *all* non-tuple sequences.
      if _any(_should_unpack_list_index(i) for i in idx):
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
          or isinstance(i, list) and i and _all(_is_scalar(e)
          and issubdtype(_dtype(e), np.bool_) for e in i))

def _expand_bool_indices(idx, shape):
  """Converts concrete bool indexes into advanced integer indexes."""
  out = []
  total_dims = len(shape)
  num_ellipsis = _sum(e is Ellipsis for e in idx)
  if num_ellipsis > 1:
    raise IndexError("an index can only have a single ellipsis ('...')")
  elif num_ellipsis == 1:
    total_dims = _sum(_ndim(e) if _is_boolean_index(e) else 1 for e in idx
                      if e is not None and e is not Ellipsis)
  ellipsis_offset = 0
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
        raise TypeError("JAX arrays do not support boolean scalar indices")
      else:
        i_shape = _shape(i)
        start = len(out) + ellipsis_offset
        expected_shape = shape[start: start + _ndim(i)]
        if i_shape != expected_shape:
          raise IndexError("boolean index did not match shape of indexed array in index "
                           f"{dim_number}: got {i_shape}, expected {expected_shape}")
        out.extend(np.where(i))
    else:
      out.append(i)
    if i is Ellipsis:
      ellipsis_offset = len(shape) - total_dims - 1
  return tuple(out)


def _is_slice_element_none_or_constant(elt):
  """Return True if elt is a constant or None."""
  if elt is None: return True
  try:
    return type(core.get_aval(elt)) is ConcreteArray
  except TypeError:
    return False

# TODO(mattjj): clean up this logic
def _is_advanced_int_indexer(idx):
  """Returns True if idx should trigger int array indexing, False otherwise."""
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
  assert isinstance(idx, tuple)
  if _all(e is None or e is Ellipsis or isinstance(e, slice)
          or _is_scalar(e) and issubdtype(_dtype(e), np.integer) for e in idx):
    return False
  return _all(e is None or e is Ellipsis or isinstance(e, slice)
              or _is_int_arraylike(e) for e in idx)

def _is_int_arraylike(x):
  """Returns True if x is array-like with integer dtype, False otherwise."""
  return (isinstance(x, int) and not isinstance(x, bool)
          or issubdtype(getattr(x, "dtype", None), np.integer)
          or isinstance(x, (list, tuple)) and _all(_is_int_arraylike(e) for e in x))

def _is_scalar(x):
  """Checks if a Python or NumPy scalar."""
  return  np.isscalar(x) or (isinstance(x, (np.ndarray, ndarray))
                             and np.ndim(x) == 0)

def _canonicalize_tuple_index(arr_ndim, idx, array_name='array'):
  """Helper to remove Ellipsis and add in the implicit trailing slice(None)."""
  len_without_none = _sum(1 for e in idx if e is not None and e is not Ellipsis)
  if len_without_none > arr_ndim:
    raise IndexError(
        f"Too many indices for {array_name}: {len_without_none} "
        f"non-None/Ellipsis indices for dim {arr_ndim}.")
  ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
  ellipsis_index = next(ellipses, None)
  if ellipsis_index is not None:
    if next(ellipses, None) is not None:
      raise IndexError(
          f"Multiple ellipses (...) not supported: {list(map(type, idx))}.")
    colons = (slice(None),) * (arr_ndim - len_without_none)
    idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1:]
  elif len_without_none < arr_ndim:
    colons = (slice(None),) * (arr_ndim - len_without_none)
    idx = tuple(idx) + colons
  return idx

def _static_idx(idx: slice, size: core.DimSize):
  """Helper function to compute the static slice start/limit/stride values."""
  if isinstance(size, int):
    start, stop, step = idx.indices(size)
  else:
    raise TypeError(size)

  if (step < 0 and stop >= start) or (step > 0 and start >= stop):
    return 0, 0, 1, False  # sliced to size zero

  if step > 0:
    return start, stop, step, False
  else:
    k  = (start - stop - 1) % (-step)
    return stop + k + 1, start + 1, -step, True


blackman = _wrap_numpy_nullary_function(np.blackman)
bartlett = _wrap_numpy_nullary_function(np.bartlett)
hamming = _wrap_numpy_nullary_function(np.hamming)
hanning = _wrap_numpy_nullary_function(np.hanning)
# TODO: lower `kaiser` via lax to allow non-constant beta values.
kaiser = _wrap_numpy_nullary_function(np.kaiser)

def _gcd_cond_fn(xs):
  x1, x2 = xs
  return any(x2 != 0)

def _gcd_body_fn(xs):
  x1, x2 = xs
  x1, x2 = (where(x2 != 0, x2, x1),
            where(x2 != 0, lax.rem(x1, x2), _lax_const(x2, 0)))
  return (where(x1 < x2, x2, x1), where(x1 < x2, x1, x2))

@_wraps(np.gcd)
@jit
def gcd(x1, x2):
  _check_arraylike("gcd", x1, x2)
  if (not issubdtype(_dtype(x1), integer) or
      not issubdtype(_dtype(x2), integer)):
    raise ValueError("Arguments to jax.numpy.gcd must be integers.")
  x1, x2 = _promote_dtypes(x1, x2)
  x1, x2 = broadcast_arrays(x1, x2)
  gcd, _ = lax.while_loop(_gcd_cond_fn, _gcd_body_fn, (abs(x1), abs(x2)))
  return gcd


@_wraps(np.lcm)
@jit
def lcm(x1, x2):
  _check_arraylike("lcm", x1, x2)
  x1, x2 = _promote_dtypes(x1, x2)
  d = gcd(x1, x2)
  return where(d == 0, _lax_const(d, 0),
               abs(multiply(x1, floor_divide(x2, d))))


@_wraps(np.extract)
def extract(condition, arr):
  return compress(ravel(condition), ravel(arr))


@_wraps(np.compress, skip_params=['out'])
def compress(condition, a, axis: Optional[int] = None, out=None):
  _check_arraylike("compress", condition, a)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.compress is not supported.")
  if ndim(condition) != 1:
    raise ValueError("condition must be a 1D array")
  condition = asarray(condition).astype(bool)
  if axis is None:
    axis = 0
    a = ravel(a)
  else:
    a = moveaxis(a, axis, 0)
  condition, extra = condition[:a.shape[0]], condition[a.shape[0]:]
  if any(extra):
    raise ValueError("condition contains entries that are out of bounds")
  a = a[:condition.shape[0]]
  return moveaxis(a[condition], 0, axis)


@_wraps(np.cov)
@partial(jit, static_argnames=('rowvar', 'bias', 'ddof'))
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None):
  if y is not None:
    m, y = _promote_args_inexact("cov", m, y)
    if y.ndim > 2:
      raise ValueError("y has more than 2 dimensions")
  else:
    m, = _promote_args_inexact("cov", m)

  if m.ndim > 2:
    raise ValueError("m has more than 2 dimensions")  # same as numpy error

  X = atleast_2d(m)
  if not rowvar and X.shape[0] != 1:
    X = X.T
  if X.shape[0] == 0:
    return array([]).reshape(0, 0)

  if y is not None:
    y = atleast_2d(y)
    if not rowvar and y.shape[0] != 1:
      y = y.T
    X = concatenate((X, y), axis=0)
  if ddof is None:
    ddof = 1 if bias == 0 else 0

  w = None
  if fweights is not None:
    _check_arraylike("cov", fweights)
    if ndim(fweights) > 1:
      raise RuntimeError("cannot handle multidimensional fweights")
    if shape(fweights)[0] != X.shape[1]:
      raise RuntimeError("incompatible numbers of samples and fweights")
    if not issubdtype(_dtype(fweights), integer):
      raise TypeError("fweights must be integer.")
    # Ensure positive fweights; note that numpy raises an error on negative fweights.
    w = asarray(abs(fweights))
  if aweights is not None:
    _check_arraylike("cov", aweights)
    if ndim(aweights) > 1:
      raise RuntimeError("cannot handle multidimensional aweights")
    if shape(aweights)[0] != X.shape[1]:
      raise RuntimeError("incompatible numbers of samples and aweights")
    # Ensure positive aweights: note that numpy raises an error for negative aweights.
    aweights = abs(aweights)
    w = aweights if w is None else w * aweights

  avg, w_sum = average(X, axis=1, weights=w, returned=True)
  w_sum = w_sum[0]

  if w is None:
    f = X.shape[1] - ddof
  elif ddof == 0:
    f = w_sum
  elif aweights is None:
    f = w_sum - ddof
  else:
    f = w_sum - ddof * sum(w * aweights) / w_sum

  X = X - avg[:, None]
  X_T = X.T if w is None else (X * lax.broadcast_to_rank(w, X.ndim)).T
  return true_divide(dot(X, X_T.conj()), f).squeeze()


@_wraps(np.corrcoef)
@partial(jit, static_argnames=('rowvar',))
def corrcoef(x, y=None, rowvar=True):
  _check_arraylike("corrcoef", x)
  c = cov(x, y, rowvar)
  if len(shape(c)) == 0:
    # scalar - this should yield nan for values (nan/nan, inf/inf, 0/0), 1 otherwise
    return divide(c, c)
  d = diag(c)
  stddev = sqrt(real(d))
  c = divide(c, stddev[:,None])
  c = divide(c, stddev[None,:])

  real_part = clip(real(c), -1, 1)
  if iscomplexobj(c):
    complex_part = clip(imag(c), -1, 1)
    c = lax.complex(real_part, complex_part)
  else:
    c = real_part
  return c


@_wraps(np.quantile, skip_params=['out', 'overwrite_input'])
@partial(jit, static_argnames=('axis', 'overwrite_input', 'interpolation',
                               'keepdims', 'method'))
def quantile(a, q, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
             overwrite_input=False, method="linear", keepdims=False,
             interpolation=None):
  _check_arraylike("quantile", a, q)
  if overwrite_input or out is not None:
    msg = ("jax.numpy.quantile does not support overwrite_input=True or "
           "out != None")
    raise ValueError(msg)
  if interpolation is not None:
    warnings.warn("The interpolation= argument to 'quantile' is deprecated. "
                  "Use 'method=' instead.", DeprecationWarning)
  return _quantile(a, q, axis, interpolation or method, keepdims, False)

@_wraps(np.nanquantile, skip_params=['out', 'overwrite_input'])
@partial(jit, static_argnames=('axis', 'overwrite_input', 'interpolation',
                               'keepdims', 'method'))
def nanquantile(a, q, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                out=None, overwrite_input=False, method="linear",
                keepdims=False, interpolation=None):
  _check_arraylike("nanquantile", a, q)
  if overwrite_input or out is not None:
    msg = ("jax.numpy.nanquantile does not support overwrite_input=True or "
           "out != None")
    raise ValueError(msg)
  if interpolation is not None:
    warnings.warn("The interpolation= argument to 'nanquantile' is deprecated. "
                  "Use 'method=' instead.", DeprecationWarning)
  return _quantile(a, q, axis, interpolation or method, keepdims, True)

def _quantile(a, q, axis, interpolation, keepdims, squash_nans):
  if interpolation not in ["linear", "lower", "higher", "midpoint", "nearest"]:
    raise ValueError("interpolation can only be 'linear', 'lower', 'higher', "
                     "'midpoint', or 'nearest'")
  a, q = _promote_dtypes_inexact(a, q)
  keepdim = []
  if issubdtype(a.dtype, np.complexfloating):
    raise ValueError("quantile does not support complex input, as the operation is poorly defined.")
  if axis is None:
    a = ravel(a)
    axis = 0
  elif isinstance(axis, tuple):
    keepdim = list(shape(a))
    nd = ndim(a)
    axis = tuple([_canonicalize_axis(ax, nd) for ax in axis])
    if len(set(axis)) != len(axis):
        raise ValueError('repeated axis')
    for ax in axis:
      keepdim[ax] = 1

    keep = set(range(nd)) - set(axis)
    # prepare permutation
    dimensions = list(range(nd))
    for i, s in enumerate(sorted(keep)):
      dimensions[i], dimensions[s] = dimensions[s], dimensions[i]
    do_not_touch_shape = tuple(x for idx,x in enumerate(shape(a)) if idx not in axis)
    touch_shape = tuple(x for idx,x in enumerate(shape(a)) if idx in axis)
    a = lax.reshape(a, do_not_touch_shape + (int(np.prod(touch_shape)),), dimensions)
    keepdim = tuple(keepdim)
    axis = _canonicalize_axis(-1, ndim(a))
  else:
    axis = _canonicalize_axis(axis, ndim(a))

  q_shape = shape(q)
  q_ndim = ndim(q)
  if q_ndim > 1:
    raise ValueError("q must be have rank <= 1, got shape {}".format(shape(q)))

  a_shape = shape(a)

  if squash_nans:
    a = where(isnan(a), nan, a) # Ensure nans are positive so they sort to the end.
    a = lax.sort(a, dimension=axis)
    counts = sum(logical_not(isnan(a)), axis=axis, dtype=q.dtype,
                 keepdims=keepdims)
    shape_after_reduction = counts.shape
    q = lax.expand_dims(
      q, tuple(range(q_ndim, len(shape_after_reduction) + q_ndim)))
    counts = lax.expand_dims(counts, tuple(range(q_ndim)))
    q = lax.mul(q, lax.sub(counts, _lax_const(q, 1)))
    low = lax.floor(q)
    high = lax.ceil(q)
    high_weight = lax.sub(q, low)
    low_weight = lax.sub(_lax_const(high_weight, 1), high_weight)

    low = lax.max(_lax_const(low, 0), lax.min(low, counts - 1))
    high = lax.max(_lax_const(high, 0), lax.min(high, counts - 1))
    low = lax.convert_element_type(low, int64)
    high = lax.convert_element_type(high, int64)
    out_shape = q_shape + shape_after_reduction
    index = [lax.broadcasted_iota(int64, out_shape, dim + q_ndim)
             for dim in range(len(shape_after_reduction))]
    if keepdims:
      index[axis] = low
    else:
      index.insert(axis, low)
    low_value = a[tuple(index)]
    index[axis] = high
    high_value = a[tuple(index)]
  else:
    a = where(any(isnan(a), axis=axis, keepdims=True), nan, a)
    a = lax.sort(a, dimension=axis)
    n = a_shape[axis]
    q = lax.mul(q, _lax_const(q, n - 1))
    low = lax.floor(q)
    high = lax.ceil(q)
    high_weight = lax.sub(q, low)
    low_weight = lax.sub(_lax_const(high_weight, 1), high_weight)

    low = lax.clamp(_lax_const(low, 0), low, _lax_const(low, n - 1))
    high = lax.clamp(_lax_const(high, 0), high, _lax_const(high, n - 1))
    low = lax.convert_element_type(low, int64)
    high = lax.convert_element_type(high, int64)

    slice_sizes = list(a_shape)
    slice_sizes[axis] = 1
    dnums = lax.GatherDimensionNumbers(
      offset_dims=tuple(range(
        q_ndim,
        len(a_shape) + q_ndim if keepdims else len(a_shape) + q_ndim - 1)),
      collapsed_slice_dims=() if keepdims else (axis,),
      start_index_map=(axis,))
    low_value = lax.gather(a, low[..., None], dimension_numbers=dnums,
                           slice_sizes=slice_sizes)
    high_value = lax.gather(a, high[..., None], dimension_numbers=dnums,
                            slice_sizes=slice_sizes)
    if q_ndim == 1:
      low_weight = lax.broadcast_in_dim(low_weight, low_value.shape,
                                        broadcast_dimensions=(0,))
      high_weight = lax.broadcast_in_dim(high_weight, high_value.shape,
                                        broadcast_dimensions=(0,))

  if interpolation == "linear":
    result = lax.add(lax.mul(low_value.astype(q.dtype), low_weight),
                     lax.mul(high_value.astype(q.dtype), high_weight))
  elif interpolation == "lower":
    result = low_value
  elif interpolation == "higher":
    result = high_value
  elif interpolation == "nearest":
    pred = lax.le(high_weight, _lax_const(high_weight, 0.5))
    result = lax.select(pred, low_value, high_value)
  elif interpolation == "midpoint":
    result = lax.mul(lax.add(low_value, high_value), _lax_const(low_value, 0.5))
  else:
    raise ValueError(f"interpolation={interpolation!r} not recognized")
  if keepdims and keepdim:
    if q_ndim > 0:
        keepdim = (shape(q)[0],) + keepdim
    result = reshape(result,  keepdim)
  return lax.convert_element_type(result, a.dtype)


@partial(vectorize, excluded={0, 2})
def _searchsorted(a, v, side):
  if len(a) == 0:
    return 0
  op = _sort_le_comparator if side == 'left' else _sort_lt_comparator
  a, v = _promote_dtypes(a, v)
  def body_fun(i, state):
    low, high = state
    mid = (low + high) // 2
    go_left = op(v, a[mid])
    return (where(go_left, low, mid), where(go_left, mid, high))

  n_levels = int(np.ceil(np.log2(len(a) + 1)))
  return lax.fori_loop(0, n_levels, body_fun, (int_(0), int_(len(a))))[1]


@_wraps(np.searchsorted, skip_params=['sorter'])
@partial(jit, static_argnames=('side', 'sorter'))
def searchsorted(a, v, side='left', sorter=None):
  _check_arraylike("searchsorted", a, v)
  if side not in ['left', 'right']:
    raise ValueError(f"{side!r} is an invalid value for keyword 'side'")
  if sorter is not None:
    raise NotImplementedError("sorter is not implemented")
  if ndim(a) != 1:
    raise ValueError("a should be 1-dimensional")
  return _searchsorted(a, v, side)


@_wraps(np.digitize)
@partial(jit, static_argnames=('right',))
def digitize(x, bins, right=False):
  _check_arraylike("digitize", x, bins)
  right = core.concrete_or_error(bool, right, "right argument of jnp.digitize()")
  if ndim(bins) != 1:
    raise ValueError(f"digitize: bins must be a 1-dimensional array; got bins={bins}")
  if len(bins) == 0:
    return zeros(x, dtype=dtypes.canonicalize_dtype(int_))
  side = 'right' if not right else 'left'
  return where(
    bins[-1] >= bins[0],
    searchsorted(bins, x, side=side),
    len(bins) - searchsorted(bins[::-1], x, side=side)
  )

_PIECEWISE_DOC = """\
Unlike `np.piecewise`, :py:func:`jax.numpy.piecewise` requires functions in
`funclist` to be traceable by JAX, as it is implemented via :func:`jax.lax.switch`.
See the :func:`jax.lax.switch` documentation for more information.
"""

@_wraps(np.piecewise, lax_description=_PIECEWISE_DOC)
def piecewise(x, condlist, funclist, *args, **kw):
  _check_arraylike("piecewise", x)
  condlist = array(condlist, dtype=bool_)
  nc, nf = len(condlist), len(funclist)
  if nf == nc + 1:
    funclist = funclist[-1:] + funclist[:-1]
  elif nf == nc:
    funclist = [0] + list(funclist)
  else:
    raise ValueError(f"with {nc} condition(s), either {nc} or {nc+1} functions are expected; got {nf}")
  consts = {i: c for i, c in enumerate(funclist) if not callable(c)}
  funcs = {i: f for i, f in enumerate(funclist) if callable(f)}
  return _piecewise(x, condlist, consts,
                    frozenset(funcs.items()),  # dict is not hashable.
                    *args, **kw)

@partial(jit, static_argnames=['funcs'])
def _piecewise(x, condlist, consts, funcs, *args, **kw):
  funcs = dict(funcs)
  funclist = [consts.get(i, funcs.get(i)) for i in range(len(condlist) + 1)]
  indices = argmax(cumsum(concatenate([zeros_like(condlist[:1]), condlist], 0), 0), 0)
  dtype = _dtype(x)
  def _call(f):
    return lambda x: f(x, *args, **kw).astype(dtype)
  def _const(v):
    return lambda x: array(v, dtype=dtype)
  funclist = [_call(f) if callable(f) else _const(f) for f in funclist]
  return vectorize(lax.switch, excluded=(1,))(indices, funclist, x)


@_wraps(np.percentile, skip_params=['out', 'overwrite_input'])
@partial(jit, static_argnames=('axis', 'overwrite_input', 'interpolation',
                               'keepdims', 'method'))
def percentile(a, q, axis: Optional[Union[int, Tuple[int, ...]]] = None,
               out=None, overwrite_input=False, method="linear",
               keepdims=False, interpolation=None):
  _check_arraylike("percentile", a, q)
  a, q = _promote_dtypes_inexact(a, q)
  q = true_divide(q, 100.0)
  return quantile(a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                  interpolation=interpolation, method=method, keepdims=keepdims)

@_wraps(np.nanpercentile, skip_params=['out', 'overwrite_input'])
@partial(jit, static_argnames=('axis', 'overwrite_input', 'interpolation',
                               'keepdims', 'method'))
def nanpercentile(a, q, axis: Optional[Union[int, Tuple[int, ...]]] = None,
                  out=None, overwrite_input=False, method="linear",
                  keepdims=False, interpolation=None):
  _check_arraylike("nanpercentile", a, q)
  q = true_divide(q, float32(100.0))
  return nanquantile(a, q, axis=axis, out=out, overwrite_input=overwrite_input,
                     interpolation=interpolation, method=method,
                     keepdims=keepdims)

@_wraps(np.median, skip_params=['out', 'overwrite_input'])
@partial(jit, static_argnames=('axis', 'overwrite_input', 'keepdims'))
def median(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
           overwrite_input=False, keepdims=False):
  _check_arraylike("median", a)
  return quantile(a, 0.5, axis=axis, out=out, overwrite_input=overwrite_input,
                  keepdims=keepdims, method='midpoint')

@_wraps(np.nanmedian, skip_params=['out', 'overwrite_input'])
@partial(jit, static_argnames=('axis', 'overwrite_input', 'keepdims'))
def nanmedian(a, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
              overwrite_input=False, keepdims=False):
  _check_arraylike("nanmedian", a)
  return nanquantile(a, 0.5, axis=axis, out=out,
                     overwrite_input=overwrite_input, keepdims=keepdims,
                     method='midpoint')


def _astype(arr, dtype):
  """Copy the array and cast to a specified dtype.

  This is implemeted via :func:`jax.lax.convert_element_type`, which may
  have slightly different behavior than :meth:`numpy.ndarray.astype` in
  some cases. In particular, the details of float-to-int and int-to-float
  casts are implementation dependent.
  """
  if dtype is None:
    dtype = dtypes.canonicalize_dtype(float_)
  lax_internal._check_user_dtype_supported(dtype, "astype")
  return lax.convert_element_type(arr, dtype)


def _nbytes(arr):
  return size(arr) * _dtype(arr).itemsize


def _itemsize(arr):
    return _dtype(arr).itemsize


def _clip(number, min=None, max=None, out=None, *, a_min=None, a_max=None):  # noqa: F811
  # ndarray.clip has a slightly different API from clip (min -> a_min, max -> a_max)
  # TODO: remove after deprecation window
  if a_min is not None or a_max is not None:
    warnings.warn('`a_min` and `a_max` keyword arguments to ndarray.clip are deprecated '
                  'in favor of `min` and `max` for compatibility with numpy. '
                  'They will be removed in JAX 0.22.2', FutureWarning)
  if min is None and a_min is not None:
    min = a_min
  if max is None and a_max is not None:
    max = a_max
  return clip(number, a_min=min, a_max=max, out=out)


def _view(arr, dtype=None, type=None):
  lax_internal._check_user_dtype_supported(dtype, "view")
  if type is not None:
    raise NotImplementedError("`type` argument of array.view()")
  if dtype is None:
    return arr
  arr_dtype = _dtype(arr)
  if arr_dtype == dtype:
    return arr
  # bool is implemented as lax:PRED, which is not compatible with lax.bitcast_convert_type.
  # We work around this by casting bool to uint8.
  if arr_dtype == bool_:
    arr = arr.astype(uint8)
  nbits_in = 8 * arr_dtype.itemsize
  nbits_out = 8 * np.dtype(dtype).itemsize
  if nbits_in == nbits_out:
    if dtype == bool_:
      return lax.bitcast_convert_type(arr, uint8).astype(dtype)
    return lax.bitcast_convert_type(arr, dtype)
  if nbits_out > nbits_in and (shape(arr)[-1] * nbits_in) % nbits_out != 0:
    raise ValueError("When changing to a larger dtype, its size must be a divisor "
                     "of the total size in bytes of the last axis of the array.")
  byte_dtypes = {8: uint8, 16: uint16, 32: uint32, 64: uint64}
  if nbits_in not in byte_dtypes:
    raise NotImplementedError(f"arr.view() for arr.dtype={arr_dtype}")
  if nbits_out not in byte_dtypes:
    raise NotImplementedError(f"arr.view(dtype) for dtype={dtype}")
  dt_in = byte_dtypes[nbits_in]
  dt_out = byte_dtypes[nbits_out]
  arr_bytes = lax.bitcast_convert_type(arr, dt_in)
  if nbits_in < nbits_out:
    arr_bytes = arr_bytes.reshape(arr.shape[:-1] + (-1, nbits_out // nbits_in)).astype(dt_out)
    shifts = expand_dims(arange(0, nbits_out, nbits_in, dtype=dt_out), tuple(range(arr_bytes.ndim - 1)))
    arr_bytes = (arr_bytes << shifts).sum(-1).astype(dt_out)
  else:
    shifts = lax.expand_dims(arange(0, nbits_in, nbits_out, dtype=dt_in), tuple(range(arr_bytes.ndim)))
    arr_bytes = ((arr_bytes[..., newaxis] >> shifts) & iinfo(dt_out).max).astype(dt_out)
    arr_bytes = arr_bytes.reshape(arr_bytes.shape[:-2] + (-1,))
  if dtype == bool_:
    return lax.bitcast_convert_type(arr_bytes, uint8).astype(dtype)
  return lax.bitcast_convert_type(arr_bytes, dtype)

### track unimplemented functions

_NOT_IMPLEMENTED_DESC = """
*** This function is not yet implemented by jax.numpy, and will raise NotImplementedError ***
"""

def _not_implemented(fun):
  @_wraps(fun, update_doc=False, lax_description=_NOT_IMPLEMENTED_DESC)
  def wrapped(*args, **kwargs):
    msg = "Numpy function {} not yet implemented"
    raise NotImplementedError(msg.format(fun))
  return wrapped


### add method and operator overloads to arraylike classes

# We add operator overloads to DeviceArray and ShapedArray. These method and
# operator overloads mainly just forward calls to the corresponding lax_numpy
# functions, which can themselves handle instances from any of these classes.

_scalar_types = (int, float, complex, np.generic)
_accepted_binop_types = (int, float, complex, np.generic, np.ndarray, ndarray)

def _defer_to_unrecognized_arg(binary_op):
  # Ensure that other array types have the chance to override arithmetic.
  def deferring_binary_op(self, other):
    if hasattr(other, '__jax_array__'):
      other = other.__jax_array__()
    if not isinstance(other, _accepted_binop_types):
      return NotImplemented
    return binary_op(self, other)
  return deferring_binary_op

def _swap_args(f):
  return lambda x, y: f(y, x)

def _unimplemented_setitem(self, i, x):
  msg = ("'{}' object does not support item assignment. JAX arrays are "
         "immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` "
         "or another .at[] method: "
         "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html")
  raise TypeError(msg.format(type(self)))

def _operator_round(number, ndigits=None):
  out = round(number, decimals=ndigits or 0)
  # If `ndigits` is None, for a builtin float round(7.5) returns an integer.
  return out.astype(int) if ndigits is None else out

_operators = {
    "getitem": _rewriting_take,
    "setitem": _unimplemented_setitem,
    "neg": negative,
    "pos": positive,
    "eq": _defer_to_unrecognized_arg(equal),
    "ne": _defer_to_unrecognized_arg(not_equal),
    "lt": _defer_to_unrecognized_arg(less),
    "le": _defer_to_unrecognized_arg(less_equal),
    "gt": _defer_to_unrecognized_arg(greater),
    "ge": _defer_to_unrecognized_arg(greater_equal),
    "abs": abs,
    "add": _defer_to_unrecognized_arg(add),
    "radd": _defer_to_unrecognized_arg(add),
    "sub": _defer_to_unrecognized_arg(subtract),
    "rsub": _defer_to_unrecognized_arg(_swap_args(subtract)),
    "mul": _defer_to_unrecognized_arg(multiply),
    "rmul": _defer_to_unrecognized_arg(multiply),
    "div": _defer_to_unrecognized_arg(divide),
    "rdiv": _defer_to_unrecognized_arg(_swap_args(divide)),
    "truediv": _defer_to_unrecognized_arg(true_divide),
    "rtruediv": _defer_to_unrecognized_arg(_swap_args(true_divide)),
    "floordiv": _defer_to_unrecognized_arg(floor_divide),
    "rfloordiv": _defer_to_unrecognized_arg(_swap_args(floor_divide)),
    "divmod": _defer_to_unrecognized_arg(divmod),
    "rdivmod": _defer_to_unrecognized_arg(_swap_args(divmod)),
    "mod": _defer_to_unrecognized_arg(mod),
    "rmod": _defer_to_unrecognized_arg(_swap_args(mod)),
    "pow": _defer_to_unrecognized_arg(power),
    "rpow": _defer_to_unrecognized_arg(_swap_args(power)),
    "matmul": _defer_to_unrecognized_arg(matmul),
    "rmatmul": _defer_to_unrecognized_arg(_swap_args(matmul)),
    "and": _defer_to_unrecognized_arg(bitwise_and),
    "rand": _defer_to_unrecognized_arg(bitwise_and),
    "or": _defer_to_unrecognized_arg(bitwise_or),
    "ror": _defer_to_unrecognized_arg(bitwise_or),
    "xor": _defer_to_unrecognized_arg(bitwise_xor),
    "rxor": _defer_to_unrecognized_arg(bitwise_xor),
    "invert": bitwise_not,
    "lshift": _defer_to_unrecognized_arg(left_shift),
    "rshift": _defer_to_unrecognized_arg(right_shift),
    "rlshift": _defer_to_unrecognized_arg(_swap_args(left_shift)),
    "rrshift": _defer_to_unrecognized_arg(_swap_args(right_shift)),
    "round": _operator_round,
}

# These numpy.ndarray methods are just refs to an equivalent numpy function
_nondiff_methods = ["all", "any", "argmax", "argmin", "argpartition", "argsort",
                    "nonzero", "searchsorted", "round"]
_diff_methods = ["choose", "conj", "conjugate", "copy", "cumprod", "cumsum",
                 "diagonal", "dot", "max", "mean", "min", "prod", "ptp",
                 "ravel", "repeat", "sort", "squeeze", "std", "sum",
                 "swapaxes", "take", "tile", "trace", "var"]

# These methods are mentioned explicitly by nondiff_methods, so we create
# _not_implemented implementations of them here rather than in __init__.py.
# TODO(phawkins): implement these.
argpartition = _not_implemented(np.argpartition)
_NOT_IMPLEMENTED = ['argpartition']


# Experimental support for NumPy's module dispatch with NEP-37.
# Currently requires https://github.com/seberg/numpy-dispatch
_JAX_ARRAY_TYPES = (device_array.DeviceArray, core.Tracer)
_HANDLED_ARRAY_TYPES = _JAX_ARRAY_TYPES + (np.ndarray,)

def __array_module__(self, types):
  if builtins.all(issubclass(t, _HANDLED_ARRAY_TYPES) for t in types):
    return jax.numpy
  else:
    return NotImplemented


def _compress_method(a, condition, axis=None, out=None):
  return compress(condition, a, axis, out)


@partial(jit, static_argnums=(1,2,3))
def _multi_slice(arr,
                 start_indices: Tuple[Tuple[int, ...]],
                 limit_indices: Tuple[Tuple[int, ...]],
                 removed_dims: Tuple[Tuple[int, ...]]):
  """Extracts multiple slices from `arr`.

  This is used to shard DeviceArray arguments to pmap. It's implemented as a
  DeviceArray method here to avoid circular imports.
  """
  results = []
  for starts, limits, removed in safe_zip(start_indices, limit_indices, removed_dims):
    sliced = lax.slice(arr, starts, limits)
    if removed:
      sliced = lax.squeeze(sliced, removed)
    results.append(sliced)
  return results

# The next two functions are related to iter(device_array), implemented here to
# avoid circular imports.
@jit
def _unstack(x):
  return [lax.index_in_dim(x, i, keepdims=False) for i in range(x.shape[0])]
setattr(device_array.DeviceArray, "_unstack", _unstack)
def _chunk_iter(x, size):
  if size > x.shape[0]:
    yield x
  else:
    num_chunks, tail = divmod(x.shape[0], size)
    for i in range(num_chunks):
      yield lax.dynamic_slice_in_dim(x, i * size, size)
    if tail:
      yield lax.dynamic_slice_in_dim(x, num_chunks * size, tail)
setattr(device_array.DeviceArray, "_chunk_iter", _chunk_iter)

# Syntactic sugar for scatter operations.
class _IndexUpdateHelper:
  # Note: this docstring will appear as the docstring for the `at` property.
  """Helper property for index update functionality.

  The ``at`` property provides a functionally pure equivalent of in-place
  array modificatons.

  In particular:

  ==============================  ================================
  Alternate syntax                Equivalent In-place expression
  ==============================  ================================
  ``x = x.at[idx].set(y)``        ``x[idx] = y``
  ``x = x.at[idx].add(y)``        ``x[idx] += y``
  ``x = x.at[idx].multiply(y)``   ``x[idx] *= y``
  ``x = x.at[idx].divide(y)``     ``x[idx] /= y``
  ``x = x.at[idx].power(y)``      ``x[idx] **= y``
  ``x = x.at[idx].min(y)``        ``x[idx] = minimum(x[idx], y)``
  ``x = x.at[idx].max(y)``        ``x[idx] = maximum(x[idx], y)``
  ``x = x.at[idx].apply(ufunc)``  ``ufunc.at(x, idx)``
  ``x = x.at[idx].get()``         ``x = x[idx]``
  ==============================  ================================

  None of the ``x.at`` expressions modify the original ``x``; instead they return
  a modified copy of ``x``. However, inside a :py:func:`~jax.jit` compiled function,
  expressions like :code:`x = x.at[idx].set(y)` are guaranteed to be applied in-place.

  Unlike NumPy in-place operations such as :code:`x[idx] += y`, if multiple
  indices refer to the same location, all updates will be applied (NumPy would
  only apply the last update, rather than applying all updates.) The order
  in which conflicting updates are applied is implementation-defined and may be
  nondeterministic (e.g., due to concurrency on some hardware platforms).

  By default, JAX assumes that all indices are in-bounds. There is experimental
  support for giving more precise semantics to out-of-bounds indexed accesses,
  via the ``mode`` parameter (see below).

  Arguments
  ---------
  mode : str
      Specify out-of-bound indexing mode. Options are:

      - ``"promise_in_bounds"``: (default) The user promises that indices are in bounds.
        No additional checking will be performed. In practice, this means that
        out-of-bounds indices in ``get()`` will be clipped, and out-of-bounds indices
        in ``set()``, ``add()``, etc. will be dropped.
      - ``"clip"``: clamp out of bounds indices into valid range.
      - ``"drop"``: ignore out-of-bound indices.
      - ``"fill"``: alias for ``"drop"``.  For `get()`, the optional ``fill_value``
        argument specifies the value that will be returned.

  indices_are_sorted : bool
      If True, the implementation will assume that the indices passed to ``at[]``
      are sorted in ascending order, which can lead to more efficient execution
      on some backends.
  unique_indices : bool
      If True, the implementation will assume that the indices passed to ``at[]``
      are unique, which can result in more efficient execution on some backends.
  fill_value : Any
      Only applies to the ``get()`` method: the fill value to return for out-of-bounds
      slices when `mode` is ``'fill'``. Ignored otherwise. Defaults to ``NaN`` for
      inexact types, the largest negative value for signed types, the largest positive
      value for unsigned types, and ``True`` for booleans.

  Examples
  --------
  >>> x = jnp.arange(5.0)
  >>> x
  DeviceArray([0., 1., 2., 3., 4.], dtype=float32)
  >>> x.at[2].add(10)
  DeviceArray([ 0.,  1., 12.,  3.,  4.], dtype=float32)
  >>> x.at[10].add(10)  # out-of-bounds indices are ignored
  DeviceArray([0., 1., 2., 3., 4.], dtype=float32)
  >>> x.at[20].add(10, mode='clip')
  DeviceArray([ 0.,  1.,  2.,  3., 14.], dtype=float32)
  >>> x.at[2].get()
  DeviceArray(2., dtype=float32)
  >>> x.at[20].get()  # out-of-bounds indices clipped
  DeviceArray(4., dtype=float32)
  >>> x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
  DeviceArray(nan, dtype=float32)
  >>> x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
  DeviceArray(-1., dtype=float32)
  """
  __slots__ = ("array",)

  def __init__(self, array):
    self.array = array

  def __getitem__(self, index):
    return _IndexUpdateRef(self.array, index)

  def __repr__(self):
    return f"_IndexUpdateHelper({repr(self.array)})"
ndarray.at.__doc__ = _IndexUpdateHelper.__doc__

_power_fn = power
_divide_fn = divide

class _IndexUpdateRef:
  """Helper object to call indexed update functions for an (advanced) index.

  This object references a source array and a specific indexer into that array.
  Methods on this object return copies of the source array that have been
  modified at the positions specified by the indexer.
  """
  __slots__ = ("array", "index")

  def __init__(self, array, index):
    self.array = array
    self.index = index

  def __repr__(self):
    return f"_IndexUpdateRef({repr(self.array)}, {repr(self.index)})"

  def get(self, indices_are_sorted=False, unique_indices=False,
          mode=None, fill_value=None):
    """Equivalent to ``x[idx]``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexing <numpy.doc.indexing>` ``x[idx]``. This function differs from
    the usual array indexing syntax in that it allows additional keyword
    arguments ``indices_are_sorted`` and ``unique_indices`` to be passed.

    See :mod:`jax.ops` for details.
    """
    return _rewriting_take(self.array, self.index,
                           indices_are_sorted=indices_are_sorted,
                           unique_indices=unique_indices, mode=mode,
                           fill_value=fill_value)

  def set(self, values, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:`indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values, lax.scatter,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def apply(self, func, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``func.at(x, idx)`` for a unary ufunc ``func``.

    Returns the value of ``x`` that would result from applying the unary
    function ``func`` to ``x`` at the given indices. This is similar to
    ``x.at[idx].set(func(x[idx]))``, but differs in the case of repeated indices:
    in ``x.at[idx].apply(func)``, repeated indices result in the function being
    applied multiple times.

    Note that in the current implementation, ``scatter_apply`` is not compatible
    with automatic differentiation.

    See :mod:`jax.ops` for details.
    """
    def _scatter_apply(x, indices, _, dims, **kwargs):
      return lax.scatter_apply(x, indices, func, dims, **kwargs)
    return scatter._scatter_update(self.array, self.index,
                                   lax_internal._zero(self.array.dtype),
                                   _scatter_apply,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def add(self, values, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] += y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] += y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_add,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def multiply(self, values, indices_are_sorted=False, unique_indices=False,
               mode=None):
    """Pure equivalent of ``x[idx] *= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_mul,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices,
                                   mode=mode)
  mul = multiply

  def divide(self, values, indices_are_sorted=False, unique_indices=False,
             mode=None):
    """Pure equivalent of ``x[idx] /= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] /= y``.

    See :mod:`jax.ops` for details.
    """
    return _divide_fn(
      self.array,
      scatter._scatter_update(ones_like(self.array), self.index, values,
                              lax.scatter_mul,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices, mode=mode))

  def power(self, values, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``x[idx] **= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] **= y``.

    See :mod:`jax.ops` for details.
    """
    return _power_fn(
      self.array,
      scatter._scatter_update(ones_like(self.array), self.index, values,
                              lax.scatter_mul,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices, mode=mode))

  def min(self, values, indices_are_sorted=False, unique_indices=False,  # noqa: F811
          mode=None):
    """Pure equivalent of ``x[idx] = minimum(x[idx], y)``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = minimum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_min,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def max(self, values, indices_are_sorted=False, unique_indices=False,  # noqa: F811
          mode=None):
    """Pure equivalent of ``x[idx] = maximum(x[idx], y)``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = maximum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_max,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)


def _set_shaped_array_attributes(shaped_array):
  # Set up operator, method, and property forwarding on Tracer instances
  # containing
  # ShapedArray avals by following the forwarding conventions for Tracer.
  # Forward operators using a single-underscore-prefix naming convention:
  for operator_name, function in _operators.items():
    setattr(shaped_array, "_{}".format(operator_name), staticmethod(function))
  # Forward methods and properties using core.{aval_method, aval_property}:
  for method_name in _nondiff_methods + _diff_methods:
    setattr(shaped_array, method_name, core.aval_method(globals()[method_name]))
  setattr(shaped_array, "reshape", core.aval_method(_reshape))
  setattr(shaped_array, "transpose", core.aval_method(_transpose))
  setattr(shaped_array, "flatten", core.aval_method(ravel))
  setattr(shaped_array, "T", core.aval_property(transpose))
  setattr(shaped_array, "real", core.aval_property(real))
  setattr(shaped_array, "imag", core.aval_property(imag))
  setattr(shaped_array, "astype", core.aval_method(_astype))
  setattr(shaped_array, "view", core.aval_method(_view))
  setattr(shaped_array, "nbytes", core.aval_property(_nbytes))
  setattr(shaped_array, "itemsize", core.aval_property(_itemsize))
  setattr(shaped_array, "clip", core.aval_method(_clip))

  setattr(shaped_array, "_array_module", staticmethod(__array_module__))
  setattr(shaped_array, "broadcast", core.aval_method(lax.broadcast))
  setattr(shaped_array, "broadcast_in_dim", core.aval_method(lax.broadcast_in_dim))
  setattr(shaped_array, "split", core.aval_method(split))
  setattr(shaped_array, "compress", _compress_method)
  setattr(shaped_array, "at", core.aval_property(_IndexUpdateHelper))
  setattr(shaped_array, "item", core.aval_method(device_array.DeviceArray.item))

_set_shaped_array_attributes(ShapedArray)
_set_shaped_array_attributes(DShapedArray)


def _set_device_array_base_attributes(device_array):
  # Forward operators, methods, and properties on DeviceArray to lax_numpy
  # functions (with no Tracers involved; this forwarding is direct)
  for operator_name, function in _operators.items():
    setattr(device_array, "__{}__".format(operator_name), function)
  for method_name in _nondiff_methods + _diff_methods:
    setattr(device_array, method_name, globals()[method_name])
  setattr(device_array, "reshape", _reshape)
  setattr(device_array, "transpose", _transpose)
  setattr(device_array, "flatten", ravel)
  setattr(device_array, "T", property(transpose))
  setattr(device_array, "real", property(real))
  setattr(device_array, "imag", property(imag))
  setattr(device_array, "astype", _astype)
  setattr(device_array, "view", _view)
  setattr(device_array, "nbytes", property(_nbytes))
  setattr(device_array, "itemsize", property(_itemsize))
  setattr(device_array, "clip", _clip)

_set_device_array_base_attributes(device_array.DeviceArray)


def _set_device_array_attributes(device_array):
  setattr(device_array, "__array_module__", __array_module__)
  # Extra methods that are handy
  setattr(device_array, "broadcast", lax.broadcast)
  setattr(device_array, "broadcast_in_dim", lax.broadcast_in_dim)
  setattr(device_array, "split", split)
  setattr(device_array, "compress", _compress_method)
  setattr(device_array, "_multi_slice", _multi_slice)
  setattr(device_array, "at", property(_IndexUpdateHelper))

for t in device_array.device_array_types:
  _set_device_array_attributes(t)
_set_device_array_attributes(pxla._ShardedDeviceArray)
_set_device_array_attributes(pxla.pmap_lib.ShardedDeviceArray)
