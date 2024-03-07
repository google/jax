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

# pytype: skip-file
# mypy: disable-error-code=has-type
"""Define methods which are dynamically added to JAX's Arrays and Tracers.

This is done dynamically in order to avoid circular imports.
"""

from __future__ import annotations

__all__ = ['register_jax_array_methods']

import abc
from functools import partial, wraps
import math
from typing import Any

import numpy as np
import jax
from jax import lax
from jax.sharding import Sharding
from jax._src import core
from jax._src import dtypes
from jax._src.api_util import _ensure_index_tuple
from jax._src.array import ArrayImpl
from jax._src.lax import lax as lax_internal
from jax._src.lib import xla_client as xc
from jax._src.numpy import lax_numpy
from jax._src.numpy import reductions
from jax._src.numpy import ufuncs
from jax._src.numpy import util
from jax._src.ops import scatter
from jax._src.typing import Array, ArrayLike, DimSize, DTypeLike, Shape
from jax._src.util import safe_zip, safe_map

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


### add method and operator overloads to arraylike classes

# We add operator overloads to Array and ShapedArray. These method and
# operator overloads mainly just forward calls to the corresponding lax_numpy
# functions, which can themselves handle instances from any of these classes.


def _astype(arr: ArrayLike, dtype: DTypeLike, copy: bool = False, device: xc.Device | Sharding | None = None) -> Array:
  """Copy the array and cast to a specified dtype.

  This is implemented via :func:`jax.lax.convert_element_type`, which may
  have slightly different behavior than :meth:`numpy.ndarray.astype` in
  some cases. In particular, the details of float-to-int and int-to-float
  casts are implementation dependent.
  """
  return lax_numpy.astype(arr, dtype, copy=copy, device=device)


def _nbytes(arr: ArrayLike) -> int:
  """Total bytes consumed by the elements of the array."""
  return np.size(arr) * dtypes.dtype(arr, canonicalize=True).itemsize


def _item(a: Array, *args) -> bool | int | float | complex:
  """Copy an element of an array to a standard Python scalar and return it."""
  arr = core.concrete_or_error(np.asarray, a, context="This occurred in the item() method of jax.Array")
  if dtypes.issubdtype(a.dtype, dtypes.extended):
    raise TypeError(f"No Python scalar type for {a.dtype=}")
  return arr.item(*args)

def _itemsize(arr: ArrayLike) -> int:
  """Length of one array element in bytes."""
  return dtypes.dtype(arr, canonicalize=True).itemsize


def _clip(number: ArrayLike,
          min: ArrayLike | None = None, max: ArrayLike | None = None) -> Array:
  """Return an array whose values are limited to a specified range.

  Refer to :func:`jax.numpy.clip` for full documentation."""
  return lax_numpy.clip(number, min=min, max=max)


def _transpose(a: Array, *args: Any) -> Array:
  """Returns a view of the array with axes transposed.

  Refer to :func:`jax.numpy.transpose` for full documentation.
  """
  if not args:
    axis = None
  elif len(args) == 1:
    axis = args[0] if args[0] is None else _ensure_index_tuple(args[0])
  else:
    axis = _ensure_index_tuple(args)
  return lax_numpy.transpose(a, axis)


def _compute_newshape(a: ArrayLike, newshape: DimSize | Shape) -> Shape:
  """Fixes a -1 value in newshape, if present."""
  orig_newshape = newshape  # for error messages
  try:
    iter(newshape)  # type: ignore[arg-type]
  except:
    newshape = [newshape]
  newshape = core.canonicalize_shape(newshape)  # type: ignore[arg-type]
  neg1s = [i for i, d in enumerate(newshape) if type(d) is int and d == -1]
  if len(neg1s) > 1:
    raise TypeError("can only specify one unknown axis size with a `-1` value, "
                    f"got {orig_newshape}")
  if neg1s:
    i, = neg1s
    other_sizes = (*newshape[:i], *newshape[i+1:])
    if (all(isinstance(d, int) for d in (*np.shape(a), *other_sizes)) and
        np.size(a) % math.prod(other_sizes) != 0):
      raise TypeError(f"cannot reshape array of shape {np.shape(a)} (size {np.size(a)}) "
                      f"into shape {orig_newshape} because the product of "
                      f"specified axis sizes ({math.prod(other_sizes)}) does "
                      f"not evenly divide {np.size(a)}")
    sz = core.cancel_divide_tracers(np.shape(a), other_sizes)
    if sz is not None:
      return (*newshape[:i], sz, *newshape[i+1:])
  else:
    if (all(isinstance(d, int) for d in (*np.shape(a), *newshape)) and
        np.size(a) != math.prod(newshape)):
      raise TypeError(f"cannot reshape array of shape {np.shape(a)} (size {np.size(a)}) "
                      f"into shape {orig_newshape} (size {math.prod(newshape)})")
  return tuple(-core.divide_shape_sizes(np.shape(a), newshape)
               if core.definitely_equal(d, -1) else d for d in newshape)


def _reshape(a: Array, *args: Any, order: str = "C") -> Array:
  """Returns an array containing the same data with a new shape.

  Refer to :func:`jax.numpy.reshape` for full documentation.
  """
  __tracebackhide__ = True
  newshape = _compute_newshape(a, args[0] if len(args) == 1 else args)
  if order == "C":
    return lax.reshape(a, newshape, None)
  elif order == "F":
    dims = list(range(a.ndim)[::-1])
    return lax.reshape(a, newshape[::-1], dims).T
  elif order == "A":
    raise NotImplementedError("np.reshape order=A is not implemented.")
  else:
    raise ValueError(f"Unexpected value for 'order' argument: {order}.")


def _view(arr: Array, dtype: DTypeLike | None = None, type: None = None) -> Array:
  """Return a bitwise copy of the array, viewed as a new dtype.

  This is fuller-featured wrapper around :func:`jax.lax.bitcast_convert_type`.

  If the source and target dtype have the same bitwidth, the result has the same
  shape as the input array. If the bitwidth of the target dtype is different
  from the source, the size of the last axis of the result is adjusted
  accordingly.

  >>> jnp.zeros([1,2,3], dtype=jnp.int16).view(jnp.int8).shape
  (1, 2, 6)
  >>> jnp.zeros([1,2,4], dtype=jnp.int8).view(jnp.int16).shape
  (1, 2, 2)

  Conversions involving booleans are not well-defined in all situations. With
  regards to the shape of result as explained above, booleans are treated as
  having a bitwidth of 8. However, when converting to a boolean array, the input
  should only contain 0 or 1 bytes. Otherwise, results may be unpredictable or
  may change depending on how the result is used.

  This conversion is guaranteed and safe:
  >>> jnp.array([1, 0, 1], dtype=jnp.int8).view(jnp.bool_)
  Array([ True, False,  True], dtype=bool)

  However, there are no guarantees about the results of any expression involving
  a view such as this: `jnp.array([1, 2, 3], dtype=jnp.int8).view(jnp.bool_)`.
  In particular, the results may change between JAX releases and depending on
  the platform. To safely convert such an array to a boolean array, compare it
  with `0`:

  >>> jnp.array([1, 2, 0], dtype=jnp.int8) != 0
  Array([ True,  True, False], dtype=bool)
  """
  if type is not None:
    raise NotImplementedError("`type` argument of array.view() is not supported.")

  util.check_arraylike("view", arr)
  arr = lax_numpy.asarray(arr)

  dtypes.check_user_dtype_supported(dtype, "view")
  dtype = dtypes.canonicalize_dtype(dtype)

  if arr.ndim == 0:
    if arr.dtype.itemsize != dtype.itemsize:
      raise ValueError("view() of a 0d array is only supported if the itemsize is unchanged.")
    return _view(lax.expand_dims(arr, (0,)), dtype).squeeze()

  if (arr.shape[-1] * arr.dtype.itemsize) % dtype.itemsize != 0:
    raise ValueError("When changing to a larger dtype, its size must be a divisor "
                     "of the total size in bytes of the last axis of the array.")

  if arr.dtype == dtype:
    return arr

  # lax.bitcast_convert_type does not support bool or complex; in these cases we
  # cast to a compatible type and recursively call _view for simplicity.
  if arr.dtype == bool:
    return _view(arr.astype('uint8'), dtype)

  if lax_numpy.issubdtype(arr.dtype, np.complexfloating):
    new_shape = (*arr.shape[:-1], arr.shape[-1] * 2)
    new_dtype = lax_numpy.finfo(arr.dtype).dtype
    arr = (lax_numpy.zeros(new_shape, new_dtype)
             .at[..., 0::2].set(arr.real)
             .at[..., 1::2].set(arr.imag))
    return _view(arr, dtype)

  if dtype == bool:
    return _view(arr, np.uint8).astype(bool)

  if lax_numpy.issubdtype(dtype, np.complexfloating):
    out = _view(arr, lax_numpy.finfo(dtype).dtype).astype(dtype)
    return out[..., 0::2] + 1j * out[..., 1::2]

  # lax.bitcast_convert_type adds or subtracts dimensions depending on the
  # relative bitwidths of the dtypes; we account for that with reshapes.
  if arr.dtype.itemsize < dtype.itemsize:
    factor = dtype.itemsize // arr.dtype.itemsize
    arr = arr.reshape(*arr.shape[:-1], arr.shape[-1] // factor, factor)
    return lax.bitcast_convert_type(arr, dtype)

  if arr.dtype.itemsize > dtype.itemsize:
    out = lax.bitcast_convert_type(arr, dtype)
    return out.reshape(*out.shape[:-2], out.shape[-2] * out.shape[-1])

  return lax.bitcast_convert_type(arr, dtype)


def _notimplemented_flat(self):
  """Not implemented: Use :meth:`~jax.Array.flatten` instead."""
  raise NotImplementedError("JAX Arrays do not implement the arr.flat property: "
                            "consider arr.flatten() instead.")

_accepted_binop_types = (int, float, complex, np.generic, np.ndarray, Array)
_rejected_binop_types = (list, tuple, set, dict)

def _defer_to_unrecognized_arg(opchar, binary_op, swap=False):
  # Ensure that other array types have the chance to override arithmetic.
  def deferring_binary_op(self, other):
    if hasattr(other, '__jax_array__'):
      other = other.__jax_array__()
    args = (other, self) if swap else (self, other)
    if isinstance(other, _accepted_binop_types):
      return binary_op(*args)
    # Note: don't use isinstance here, because we don't want to raise for
    # subclasses, e.g. NamedTuple objects that may override operators.
    if type(other) in _rejected_binop_types:
      raise TypeError(f"unsupported operand type(s) for {opchar}: "
                      f"{type(args[0]).__name__!r} and {type(args[1]).__name__!r}")
    return NotImplemented
  return deferring_binary_op

def _unimplemented_setitem(self, i, x):
  msg = ("'{}' object does not support item assignment. JAX arrays are "
         "immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` "
         "or another .at[] method: "
         "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html")
  raise TypeError(msg.format(type(self)))

def _operator_round(number: ArrayLike, ndigits: int | None = None) -> Array:
  out = lax_numpy.round(number, decimals=ndigits or 0)
  # If `ndigits` is None, for a builtin float round(7.5) returns an integer.
  return out.astype(int) if ndigits is None else out

def _copy(self: Array) -> Array:
  return self.copy()

def _deepcopy(self: Array, memo: Any) -> Array:
  del memo  # unused
  return self.copy()


# Experimental support for NumPy's module dispatch with NEP-37.
# Currently requires https://github.com/seberg/numpy-dispatch
_JAX_ARRAY_TYPES = (core.Tracer, ArrayImpl)
_HANDLED_ARRAY_TYPES = _JAX_ARRAY_TYPES + (np.ndarray,)

def __array_module__(self, types):
  if all(issubclass(t, _HANDLED_ARRAY_TYPES) for t in types):
    return jax.numpy
  else:
    return NotImplemented


def _compress_method(a: ArrayLike, condition: ArrayLike,
                     axis: int | None = None, *, out: None = None,
                     size: int | None = None, fill_value: ArrayLike = 0) -> Array:
  """Return selected slices of this array along given axis.

  Refer to :func:`jax.numpy.compress` for full documentation."""
  return lax_numpy.compress(condition, a, axis=axis, out=out,
                            size=size, fill_value=fill_value)


@core.stash_axis_env()
@partial(jax.jit, static_argnums=(1,2,3))
def _multi_slice(arr: ArrayLike,
                 start_indices: tuple[tuple[int, ...]],
                 limit_indices: tuple[tuple[int, ...]],
                 removed_dims: tuple[tuple[int, ...]]) -> list[Array]:
  """Extracts multiple slices from `arr`.

  This is used to shard Array arguments to pmap. It's implemented as a
  Array method here to avoid circular imports.
  """
  results: list[Array] = []
  for starts, limits, removed in zip(start_indices, limit_indices, removed_dims):
    sliced = lax.slice(arr, starts, limits)
    if removed:
      sliced = lax.squeeze(sliced, removed)
    results.append(sliced)
  return results

# The next two functions are related to iter(device_array), implemented here to
# avoid circular imports.
@jax.jit
def _unstack(x: Array) -> list[Array]:
  return [lax.index_in_dim(x, i, keepdims=False) for i in range(x.shape[0])]

def _chunk_iter(x, size):
  if size > x.shape[0]:
    yield x
  else:
    num_chunks, tail = ufuncs.divmod(x.shape[0], size)
    for i in range(num_chunks):
      yield lax.dynamic_slice_in_dim(x, i * size, size)
    if tail:
      yield lax.dynamic_slice_in_dim(x, num_chunks * size, tail)

def _getitem(self, item):
  return lax_numpy._rewriting_take(self, item)

# Syntactic sugar for scatter operations.
class _IndexUpdateHelper:
  # Note: this docstring will appear as the docstring for the `at` property.
  """Helper property for index update functionality.

  The ``at`` property provides a functionally pure equivalent of in-place
  array modifications.

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

  By default, JAX assumes that all indices are in-bounds. Alternative out-of-bound
  index semantics can be specified via the ``mode`` parameter (see below).

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

        See :class:`jax.lax.GatherScatterMode` for more details.

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
  Array([0., 1., 2., 3., 4.], dtype=float32)
  >>> x.at[2].add(10)
  Array([ 0.,  1., 12.,  3.,  4.], dtype=float32)
  >>> x.at[10].add(10)  # out-of-bounds indices are ignored
  Array([0., 1., 2., 3., 4.], dtype=float32)
  >>> x.at[20].add(10, mode='clip')
  Array([ 0.,  1.,  2.,  3., 14.], dtype=float32)
  >>> x.at[2].get()
  Array(2., dtype=float32)
  >>> x.at[20].get()  # out-of-bounds indices clipped
  Array(4., dtype=float32)
  >>> x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
  Array(nan, dtype=float32)
  >>> x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
  Array(-1., dtype=float32)
  """
  __slots__ = ("array",)

  def __init__(self, array):
    self.array = array

  def __getitem__(self, index):
    return _IndexUpdateRef(self.array, index)

  def __repr__(self):
    return f"_IndexUpdateHelper({self.array!r})"


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
    return f"_IndexUpdateRef({self.array!r}, {self.index!r})"

  def get(self, *, indices_are_sorted=False, unique_indices=False,
          mode=None, fill_value=None):
    """Equivalent to ``x[idx]``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexing <numpy.doc.indexing>` ``x[idx]``. This function differs from
    the usual array indexing syntax in that it allows additional keyword
    arguments ``indices_are_sorted`` and ``unique_indices`` to be passed.

    See :mod:`jax.ops` for details.
    """
    return lax_numpy._rewriting_take(self.array, self.index,
                                     indices_are_sorted=indices_are_sorted,
                                     unique_indices=unique_indices, mode=mode,
                                     fill_value=fill_value)

  def set(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:`indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values, lax.scatter,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def apply(self, func, *, indices_are_sorted=False, unique_indices=False,
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
    def _scatter_apply(x, indices, y, dims, **kwargs):
      return lax.scatter_apply(x, indices, func, dims, update_shape=y.shape, **kwargs)
    return scatter._scatter_update(self.array, self.index,
                                   lax_internal._zero(self.array.dtype),
                                   _scatter_apply,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  def add(self, values, *, indices_are_sorted=False, unique_indices=False,
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

  def multiply(self, values, *, indices_are_sorted=False, unique_indices=False,
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

  def divide(self, values, *, indices_are_sorted=False, unique_indices=False,
             mode=None):
    """Pure equivalent of ``x[idx] /= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] /= y``.

    See :mod:`jax.ops` for details.
    """
    return ufuncs.divide(
      self.array,
      scatter._scatter_update(lax_numpy.ones_like(self.array), self.index, values,
                              lax.scatter_mul,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices, mode=mode))

  def power(self, values, *, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``x[idx] **= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] **= y``.

    See :mod:`jax.ops` for details.
    """
    return ufuncs.power(
      self.array,
      scatter._scatter_update(lax_numpy.ones_like(self.array), self.index, values,
                              lax.scatter_mul,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices, mode=mode))

  def min(self, values, *, indices_are_sorted=False, unique_indices=False,
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

  def max(self, values, *, indices_are_sorted=False, unique_indices=False,
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

_array_operators = {
  "getitem": _getitem,
  "setitem": _unimplemented_setitem,
  "copy": _copy,
  "deepcopy": _deepcopy,
  "neg": ufuncs.negative,
  "pos": ufuncs.positive,
  "eq": _defer_to_unrecognized_arg("==", ufuncs.equal),
  "ne": _defer_to_unrecognized_arg("!=", ufuncs.not_equal),
  "lt": _defer_to_unrecognized_arg("<", ufuncs.less),
  "le": _defer_to_unrecognized_arg("<=", ufuncs.less_equal),
  "gt": _defer_to_unrecognized_arg(">", ufuncs.greater),
  "ge": _defer_to_unrecognized_arg(">=", ufuncs.greater_equal),
  "abs": ufuncs.abs,
  "add": _defer_to_unrecognized_arg("+", ufuncs.add),
  "radd": _defer_to_unrecognized_arg("+", ufuncs.add, swap=True),
  "sub": _defer_to_unrecognized_arg("-", ufuncs.subtract),
  "rsub": _defer_to_unrecognized_arg("-", ufuncs.subtract, swap=True),
  "mul": _defer_to_unrecognized_arg("*", ufuncs.multiply),
  "rmul": _defer_to_unrecognized_arg("*", ufuncs.multiply, swap=True),
  "div": _defer_to_unrecognized_arg("/", ufuncs.divide),
  "rdiv": _defer_to_unrecognized_arg("/", ufuncs.divide, swap=True),
  "truediv": _defer_to_unrecognized_arg("/", ufuncs.true_divide),
  "rtruediv": _defer_to_unrecognized_arg("/", ufuncs.true_divide, swap=True),
  "floordiv": _defer_to_unrecognized_arg("//", ufuncs.floor_divide),
  "rfloordiv": _defer_to_unrecognized_arg("//", ufuncs.floor_divide, swap=True),
  "divmod": _defer_to_unrecognized_arg("divmod", ufuncs.divmod),
  "rdivmod": _defer_to_unrecognized_arg("divmod", ufuncs.divmod, swap=True),
  "mod": _defer_to_unrecognized_arg("%", ufuncs.mod),
  "rmod": _defer_to_unrecognized_arg("%", ufuncs.mod, swap=True),
  "pow": _defer_to_unrecognized_arg("**", ufuncs.power),
  "rpow": _defer_to_unrecognized_arg("**", ufuncs.power, swap=True),
  "matmul": _defer_to_unrecognized_arg("@", lax_numpy.matmul),
  "rmatmul": _defer_to_unrecognized_arg("@", lax_numpy.matmul, swap=True),
  "and": _defer_to_unrecognized_arg("&", ufuncs.bitwise_and),
  "rand": _defer_to_unrecognized_arg("&", ufuncs.bitwise_and, swap=True),
  "or": _defer_to_unrecognized_arg("|", ufuncs.bitwise_or),
  "ror": _defer_to_unrecognized_arg("|", ufuncs.bitwise_or, swap=True),
  "xor": _defer_to_unrecognized_arg("^", ufuncs.bitwise_xor),
  "rxor": _defer_to_unrecognized_arg("^", ufuncs.bitwise_xor, swap=True),
  "invert": ufuncs.bitwise_not,
  "lshift": _defer_to_unrecognized_arg("<<", ufuncs.left_shift),
  "rshift": _defer_to_unrecognized_arg(">>", ufuncs.right_shift),
  "rlshift": _defer_to_unrecognized_arg("<<", ufuncs.left_shift, swap=True),
  "rrshift": _defer_to_unrecognized_arg(">>", ufuncs.right_shift, swap=True),
  "round": _operator_round,
}

_array_methods = {
  "all": reductions.all,
  "any": reductions.any,
  "argmax": lax_numpy.argmax,
  "argmin": lax_numpy.argmin,
  "argpartition": lax_numpy.argpartition,
  "argsort": lax_numpy.argsort,
  "astype": _astype,
  "choose": lax_numpy.choose,
  "clip": _clip,
  "conj": ufuncs.conj,
  "conjugate": ufuncs.conjugate,
  "compress": _compress_method,
  "copy": lax_numpy.copy,
  "cumprod": reductions.cumprod,
  "cumsum": reductions.cumsum,
  "diagonal": lax_numpy.diagonal,
  "dot": lax_numpy.dot,
  "flatten": lax_numpy.ravel,
  "item": _item,
  "max": reductions.max,
  "mean": reductions.mean,
  "min": reductions.min,
  "nonzero": lax_numpy.nonzero,
  "prod": reductions.prod,
  "ptp": reductions.ptp,
  "ravel": lax_numpy.ravel,
  "repeat": lax_numpy.repeat,
  "reshape": _reshape,
  "round": lax_numpy.round,
  "searchsorted": lax_numpy.searchsorted,
  "sort": lax_numpy.sort,
  "squeeze": lax_numpy.squeeze,
  "std": reductions.std,
  "sum": reductions.sum,
  "swapaxes": lax_numpy.swapaxes,
  "take": lax_numpy.take,
  "trace": lax_numpy.trace,
  "transpose": _transpose,
  "var": reductions.var,
  "view": _view,

  # Methods exposed in order to avoid circular imports
  "_split": lax_numpy.split,  # used in jacfwd/jacrev
  "_multi_slice": _multi_slice,  # used in pxla for sharding
}

_impl_only_array_methods = {
  "_chunk_iter": _chunk_iter,
  "_unstack": _unstack,
}

_array_properties = {
  "flat": _notimplemented_flat,
  "T": lax_numpy.transpose,
  "mT": lax_numpy.matrix_transpose,
  "real": ufuncs.real,
  "imag": ufuncs.imag,
  "nbytes": _nbytes,
  "itemsize": _itemsize,
  "at": _IndexUpdateHelper,
}

def _set_shaped_array_attributes(shaped_array):
  # Set up operator, method, and property forwarding on Tracer instances
  # containing
  # ShapedArray avals by following the forwarding conventions for Tracer.
  # Forward operators using a single-underscore-prefix naming convention:
  for operator_name, function in _array_operators.items():
    setattr(shaped_array, f"_{operator_name}", staticmethod(function))
  # Forward methods and properties using core.{aval_method, aval_property}:
  for method_name, method in _array_methods.items():
    setattr(shaped_array, method_name, core.aval_method(method))
  for prop_name, prop in _array_properties.items():
    setattr(shaped_array, prop_name, core.aval_property(prop))
  setattr(shaped_array, "_array_module", staticmethod(__array_module__))

def _forward_operator_to_aval(name):
  def op(self, *args):
    return getattr(self.aval, f"_{name}")(self, *args)
  return op

def _forward_method_to_aval(name):
  def meth(self, *args, **kwargs):
    __tracebackhide__ = True
    return getattr(self.aval, name).fun(self, *args, **kwargs)
  return meth

def _forward_property_to_aval(name):
  @property
  def prop(self):
    return getattr(self.aval, name).fget(self)
  return prop

def _set_tracer_aval_forwarding(tracer, exclude=()):
  for operator_name in _array_operators:
    if operator_name not in exclude:
      setattr(tracer, f"__{operator_name}__", _forward_operator_to_aval(operator_name))
  for method_name in _array_methods:
    if method_name not in exclude:
      setattr(tracer, method_name, _forward_method_to_aval(method_name))
  for prop_name in _array_properties:
    if prop_name not in exclude:
      setattr(tracer, prop_name, _forward_property_to_aval(prop_name))

def _set_array_base_attributes(device_array, include=None, exclude=None):
  # Forward operators, methods, and properties on Array to lax_numpy
  # functions (with no Tracers involved; this forwarding is direct)
  def maybe_setattr(attr_name, target):
    if exclude is not None and attr_name in exclude:
      return
    if not include or attr_name in include:
      setattr(device_array, attr_name, target)

  for operator_name, function in _array_operators.items():
    maybe_setattr(f"__{operator_name}__", function)
  for method_name, method in _array_methods.items():
    maybe_setattr(method_name, method)
  for prop_name, prop in _array_properties.items():
    maybe_setattr(prop_name, property(prop))

  for name, func in _impl_only_array_methods.items():
    setattr(device_array, name, func)

def _set_array_attributes(device_array):
  setattr(device_array, "__array_module__", __array_module__)

def _make_abstract_method(name, func):
  @abc.abstractmethod
  @wraps(func)
  def method(*args, **kwargs):
    raise NotImplementedError(f"Cannot call abstract method {name}")
  return method

def _set_array_abstract_methods(basearray):
  for operator_name, function in _array_operators.items():
    setattr(basearray, f"__{operator_name}__",
            _make_abstract_method(f"__{operator_name}__", function))
  for method_name, method in _array_methods.items():
    setattr(basearray, method_name,
            _make_abstract_method(method_name, method))
  for prop_name, prop in _array_properties.items():
    setattr(basearray, prop_name,
            property(_make_abstract_method(prop_name, prop)))

def register_jax_array_methods():
  """Call this function once to register methods of JAX arrays"""
  _set_shaped_array_attributes(core.ShapedArray)
  _set_shaped_array_attributes(core.DShapedArray)

  _set_array_base_attributes(ArrayImpl, exclude={'__getitem__'})
  _set_tracer_aval_forwarding(core.Tracer, exclude={*_impl_only_array_methods, "at"})
  _set_array_attributes(ArrayImpl)

  _set_array_abstract_methods(Array)
