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

import enum
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Sequence, Union
import weakref

import numpy as np

import jax
from jax import core
from jax._src import ad_util
from jax._src import dtypes
from jax._src import source_info_util
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax._src.lax.utils import (
    _argnum_weak_type,
    _input_dtype,
    standard_primitive,
)
from jax._src.lax import lax
from jax._src import util
from jax._src.util import safe_map, safe_zip
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client

xb = xla_bridge
xc = xla_client

Array = Any
Shape = core.Shape

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

_dtype = partial(dtypes.dtype, canonicalize=True)


def slice(operand: Array, start_indices: Sequence[int],
          limit_indices: Sequence[int],
          strides: Optional[Sequence[int]] = None) -> Array:
  """Wraps XLA's `Slice
  <https://www.tensorflow.org/xla/operation_semantics#slice>`_
  operator.
  """
  return slice_p.bind(operand, start_indices=tuple(start_indices),
                      limit_indices=tuple(limit_indices),
                      strides=None if strides is None else tuple(strides))

def dynamic_slice(operand: Array, start_indices: Sequence[Array],
                  slice_sizes: Shape) -> Array:
  """Wraps XLA's `DynamicSlice
  <https://www.tensorflow.org/xla/operation_semantics#dynamicslice>`_
  operator.

  Args:
    operand: an array to slice.
    start_indices: a list of scalar indices, one per dimension. These values
      may be dynamic.
    slice_sizes: the size of the slice. Must be a sequence of non-negative
      integers with length equal to `ndim(operand)`. Inside a JIT compiled
      function, only static values are supported (all JAX arrays inside JIT
      must have statically known size).

  Returns:
    An array containing the slice.

  Examples:
    Here is a simple two-dimensional dynamic slice:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    DeviceArray([[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_slice(x, (1, 1), (2, 3))
    DeviceArray([[ 5,  6,  7],
                 [ 9, 10, 11]], dtype=int32)

    Note the potentially surprising behavior for the case where the requested slice
    overruns the bounds of the array; in this case the start index is adjusted to
    return a slice of the requested size:

    >>> dynamic_slice(x, (1, 1), (2, 4))
    DeviceArray([[ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]], dtype=int32)
  """
  start_indices = _dynamic_slice_indices(operand, start_indices)
  return dynamic_slice_p.bind(operand, *start_indices,
                              slice_sizes=core.canonicalize_shape(slice_sizes))

def dynamic_update_slice(operand: Array, update: Array,
                         start_indices: Array) -> Array:
  """Wraps XLA's `DynamicUpdateSlice
  <https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice>`_
  operator.

  Args:
    operand: an array to slice.
    update: an array containing the new values to write onto `operand`.
    start_indices: a list of scalar indices, one per dimension.

  Returns:
    An array containing the slice.

  Examples:
    Here is an example of updating a one-dimensional slice update:

    >>> x = jnp.zeros(6)
    >>> y = jnp.ones(3)
    >>> dynamic_update_slice(x, y, (2,))
    DeviceArray([0., 0., 1., 1., 1., 0.], dtype=float32)

    If the update slice is too large to fit in the array, the start
    index will be adjusted to make it fit

    >>> dynamic_update_slice(x, y, (3,))
    DeviceArray([0., 0., 0., 1., 1., 1.], dtype=float32)
    >>> dynamic_update_slice(x, y, (5,))
    DeviceArray([0., 0., 0., 1., 1., 1.], dtype=float32)

    Here is an example of a two-dimensional slice update:

    >>> x = jnp.zeros((4, 4))
    >>> y = jnp.ones((2, 2))
    >>> dynamic_update_slice(x, y, (1, 2))
    DeviceArray([[0., 0., 0., 0.],
                 [0., 0., 1., 1.],
                 [0., 0., 1., 1.],
                 [0., 0., 0., 0.]], dtype=float32)
  """
  start_indices = _dynamic_slice_indices(operand, start_indices)
  return dynamic_update_slice_p.bind(operand, update, *start_indices)


class GatherDimensionNumbers(NamedTuple):
  """
  Describes the dimension number arguments to an `XLA's Gather operator
  <https://www.tensorflow.org/xla/operation_semantics#gather>`_. See the XLA
  documentation for more details of what the dimension numbers mean.

  Args:
    offset_dims: the set of dimensions in the `gather` output that offset into
      an array sliced from `operand`. Must be a tuple of integers in ascending
      order, each representing a dimension number of the output.
    collapsed_slice_dims: the set of dimensions `i` in `operand` that have
      `slice_sizes[i] == 1` and that should not have a corresponding dimension
      in the output of the gather. Must be a tuple of integers in ascending
      order.
    start_index_map: for each dimension in `start_indices`, gives the
      corresponding dimension in `operand` that is to be sliced. Must be a
      tuple of integers with size equal to `start_indices.shape[-1]`.

  Unlike XLA's `GatherDimensionNumbers` structure, `index_vector_dim` is
  implicit; there is always an index vector dimension and it must always be the
  last dimension. To gather scalar indices, add a trailing dimension of size 1.
  """
  offset_dims: Sequence[int]
  collapsed_slice_dims: Sequence[int]
  start_index_map: Sequence[int]


class GatherScatterMode(enum.Enum):
  """
  Describes how to handle out-of-bounds indices in a gather or scatter.

  Possible values are:

  CLIP:
    Indices will be clamped to the nearest in-range value, i.e., such that the
    entire window to be gathered is in-range.
  FILL_OR_DROP:
    If any part of a gathered window is out of bounds, the entire window
    that is returned, even those elements that were otherwise in-bounds, will be
    filled with a constant.
    If any part of a scattered window is out of bounds, the entire window
    will be discarded.
  PROMISE_IN_BOUNDS:
    The user promises that indices are in bounds. No additional checking will be
    performed. In practice, with the current XLA  implementation this means
    that, out-of-bounds gathers will be clamped but out-of-bounds scatters will
    be discarded. Gradients will not be correct if indices are out-of-bounds.
  """
  CLIP = enum.auto()
  FILL_OR_DROP = enum.auto()
  PROMISE_IN_BOUNDS = enum.auto()

  @staticmethod
  def from_any(s: Optional[Union[str, 'GatherScatterMode']]):
    if isinstance(s, GatherScatterMode):
      return s
    if s == "clip":
      return GatherScatterMode.CLIP
    if s is None or s == "fill" or s == "drop":
      return GatherScatterMode.FILL_OR_DROP
    if s == "promise_in_bounds":
      return GatherScatterMode.PROMISE_IN_BOUNDS
    else:
      raise ValueError(f'Unknown gather mode "{s}"')


def gather(operand: Array, start_indices: Array,
           dimension_numbers: GatherDimensionNumbers,
           slice_sizes: Shape,
           *,
           unique_indices: bool = False,
           indices_are_sorted: bool = False,
           mode: Optional[Union[str, GatherScatterMode]] = None,
           fill_value = None) -> Array:
  """Gather operator.

  Wraps `XLA's Gather operator
  <https://www.tensorflow.org/xla/operation_semantics#gather>`_.

  The semantics of gather are complicated, and its API might change in the
  future. For most use cases, you should prefer `Numpy-style indexing
  <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_
  (e.g., `x[:, (1,4,7), ...]`), rather than using `gather` directly.

  Args:
    operand: an array from which slices should be taken
    start_indices: the indices at which slices should be taken
    dimension_numbers: a `lax.GatherDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices` and the output relate.
    slice_sizes: the size of each slice. Must be a sequence of non-negative
      integers with length equal to `ndim(operand)`.
    indices_are_sorted: whether `indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve
      performance on some backends.
    mode: how to handle indices that are out of bounds: when set to ``'clip'``,
      indices are clamped so that the slice is within bounds, and when
      set to ``'fill'`` or ``'drop'`` gather returns a slice full of
      ``fill_value`` for the affected slice. The behavior for out-of-bounds
      indices when set to ``'promise_in_bounds'`` is implementation-defined.
    fill_value: the fill value to return for out-of-bounds slices when `mode`
      is ``'fill'``. Ignored otherwise. Defaults to ``NaN`` for inexact types,
      the largest negative value for signed types, the largest positive value
      for unsigned types, and ``True`` for booleans.

  Returns:
    An array containing the gather output.
  """
  if mode is None:
    mode = GatherScatterMode.PROMISE_IN_BOUNDS
  parsed_mode = GatherScatterMode.from_any(mode)
  if parsed_mode == GatherScatterMode.FILL_OR_DROP:
    if fill_value is None:
      dtype = _dtype(operand)
      if dtypes.issubdtype(dtype, np.inexact):
        fill_value = np.nan
      elif dtypes.issubdtype(dtype, np.signedinteger):
        fill_value = dtypes.iinfo(dtype).min
      elif dtypes.issubdtype(dtype, np.unsignedinteger):
        fill_value = dtypes.iinfo(dtype).max
      elif dtype == dtypes.bool_:
        fill_value = True
      else:
        raise ValueError(f"Unsupported dtype for gather fill_value {dtype}")
  else:
    fill_value = None
  return gather_p.bind(
      operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=core.canonicalize_shape(slice_sizes),
      unique_indices=bool(unique_indices),
      indices_are_sorted=bool(indices_are_sorted),
      mode=parsed_mode,
      fill_value=fill_value)



class ScatterDimensionNumbers(NamedTuple):
  """
  Describes the dimension number arguments to an `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_. See the XLA
  documentation for more details of what the dimension numbers mean.

  Args:
    update_window_dims: the set of dimensions in the `updates` that are window
      dimensions. Must be a tuple of integers in ascending
      order, each representing a dimension number.
    inserted_window_dims: the set of size 1 window dimensions that must be
      inserted into the shape of `updates`. Must be a tuple of integers in
      ascending order, each representing a dimension number of the output. These
      are the mirror image of `collapsed_slice_dims` in the case of `gather`.
    scatter_dims_to_operand_dims: for each dimension in `scatter_indices`, gives
      the corresponding dimension in `operand`. Must be a sequence of integers
      with size equal to indices.shape[-1].

  Unlike XLA's `ScatterDimensionNumbers` structure, `index_vector_dim` is
  implicit; there is always an index vector dimension and it must always be the
  last dimension. To scatter scalar indices, add a trailing dimension of size 1.
  """
  update_window_dims: Sequence[int]
  inserted_window_dims: Sequence[int]
  scatter_dims_to_operand_dims: Sequence[int]

def scatter_add(
  operand: Array, scatter_indices: Array, updates: Array,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: Optional[Union[str, GatherScatterMode]] = None) -> Array:
  """Scatter-add operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  addition is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(lax.add,
                                       lax._abstractify(lax._const(operand, 0)))
  return scatter_add_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

def scatter_mul(
  operand: Array, scatter_indices: Array, updates: Array,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: Optional[Union[str, GatherScatterMode]] = None) -> Array:
  """Scatter-multiply operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  multiplication is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(lax.mul,
                                       lax._abstractify(lax._const(operand, 1)))
  return scatter_mul_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

def scatter_min(
  operand: Array, scatter_indices: Array, updates: Array,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: Optional[Union[str, GatherScatterMode]] = None) -> Array:
  """Scatter-min operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  the `min` function is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(lax.min,
                                       lax._abstractify(lax._const(operand, 0)))
  return scatter_min_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

def scatter_max(
  operand: Array, scatter_indices: Array, updates: Array,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: Optional[Union[str, GatherScatterMode]] = None) -> Array:
  """Scatter-max operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
  the `max` function is used to combine updates and values from `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(lax.max,
                                       lax._abstractify(lax._const(operand, 0)))
  return scatter_max_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

# To avoid recompilation, we store a dict of weak references to funcs.
_scatter_apply_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

def scatter_apply(
  operand: Array, scatter_indices: Array,
  func: Callable[[Array], Array],
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: Optional[Union[str, GatherScatterMode]] = None) -> Array:
  """Scatter-apply operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where values
  from ``operand`` are replaced with ``func(operand)``, with duplicate indices
  resulting in multiple applications of ``func``.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Note that in the current implementation, ``scatter_apply`` is not compatible
  with automatic differentiation.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    func: unary function that will be applied at each index.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the result of applying `func` to `operand` at the given indices.
  """
  # TODO: can we implement this without a placeholder?
  unused = lax.full(scatter_indices.shape[:1], 0, operand.dtype)
  _apply = lambda x, _: func(x)
  try:
    _apply = _scatter_apply_cache.setdefault(func, _apply)
  except TypeError:  # func is not weak referenceable
    pass
  jaxpr, consts = lax._reduction_jaxpr(_apply, lax._abstractify(lax._zero(operand)))
  # TODO: implement this via its own primitive so we can define appropriate autodiff rules.
  return scatter_p.bind(
      operand, scatter_indices, unused, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

# Define this outside of scatter to ensure cache hits.
_scatter_reduction_computation = lambda x, y: y

def scatter(
  operand: Array, scatter_indices: Array, updates: Array,
  dimension_numbers: ScatterDimensionNumbers, *,
  indices_are_sorted: bool = False, unique_indices: bool = False,
  mode: Optional[Union[str, GatherScatterMode]] = None) -> Array:
  """Scatter-update operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where updates
  replace values from `operand`.

  If multiple updates are performed to the same index of operand, they may be
  applied in any order.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
      how dimensions of `operand`, `start_indices`, `updates` and the output
      relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve performance on
      some backends.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when
      set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
      for out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(_scatter_reduction_computation,
                                       lax._abstractify(lax._const(operand, 0)))
  return scatter_p.bind(
      operand, scatter_indices, updates, update_jaxpr=jaxpr,
      update_consts=consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))


def scatter_reduce(
    operand: Array,
    scatter_indices: Array,
    updates: Array,
    computation: Callable[[Array, Array], Array],
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: Optional[Union[str, GatherScatterMode]] = None) -> Array:
  """Scatter-reduce operator.

  Wraps `XLA's Scatter operator
  <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where the
  custom `computation` function is used to combine updates and values from
  `operand`.

  The semantics of scatter are complicated, and its API might change in the
  future. For most use cases, you should prefer the
  :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
  the familiar NumPy indexing syntax.

  Args:
    operand: an array to which the scatter should be applied
    scatter_indices: an array that gives the indices in `operand` to which each
      update in `updates` should be applied.
    updates: the updates that should be scattered onto `operand`.
    computation: the reduction function for computing a new value given a
      current value from `operand` and a corresponding update value from
      `updates`.
    dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes how
      dimensions of `operand`, `start_indices`, `updates` and the output relate.
    indices_are_sorted: whether `scatter_indices` is known to be sorted. If
      true, may improve performance on some backends.
    unique_indices: whether the indices to be updated in ``operand`` are
      guaranteed to not overlap with each other. If true, may improve
      performance on some backends.
    mode: how to handle indices that are out of bounds: when set to 'clip',
      indices are clamped so that the slice is within bounds, and when set to
      'fill' or 'drop' out-of-bounds updates are dropped. The behavior for
      out-of-bounds indices when set to 'promise_in_bounds' is
      implementation-defined.

  Returns:
    An array containing the sum of `operand` and the scattered updates.
  """
  jaxpr, consts = lax._reduction_jaxpr(computation,
                                       lax._abstractify(lax._const(operand, 0)))
  return scatter_reduce_p.bind(
      operand,
      scatter_indices,
      updates,
      update_jaxpr=jaxpr,
      update_consts=consts,
      dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      mode=GatherScatterMode.from_any(mode))

def index_take(src: Array, idxs: Array, axes: Sequence[int]) -> Array:
  indices = lax.concatenate([lax.expand_dims(i, (1,)) for i in idxs], 1)
  indices = indices % np.array([src.shape[ax] for ax in axes])
  slice_sizes = list(src.shape)
  for ax in axes:
    slice_sizes[ax] = 1
  offset_dims = tuple(range(1, src.ndim - indices.shape[1] + 1))
  dnums = GatherDimensionNumbers(
      offset_dims=offset_dims,
      collapsed_slice_dims=axes,
      start_index_map=axes)
  return gather(src, indices, dimension_numbers=dnums,
                slice_sizes=tuple(slice_sizes))


### convenience wrappers around traceables

def slice_in_dim(operand: Array, start_index: Optional[int],
                 limit_index: Optional[int],
                 stride: int = 1, axis: int = 0) -> Array:
  """Convenience wrapper around slice applying to only one dimension."""
  start_indices = [0] * operand.ndim
  limit_indices = list(operand.shape)
  strides = [1] * operand.ndim

  # translate `None`
  len_axis = operand.shape[axis]
  start_index_int = (core._canonicalize_dimension(start_index)
                     if start_index is not None else 0)
  limit_index_int = (core._canonicalize_dimension(limit_index)
                     if limit_index is not None else len_axis)

  # translate negative indices
  if start_index_int < 0:
    start_index_int = start_index_int + len_axis
  if limit_index_int < 0:
    limit_index_int = limit_index_int + len_axis

  axis = int(axis)
  start_indices[axis] = start_index_int
  limit_indices[axis] = limit_index_int
  strides[axis] = int(stride)

  return slice(operand, start_indices, limit_indices, strides)


def index_in_dim(operand: Array, index: int, axis: int = 0,
                 keepdims: bool = True) -> Array:
  """Convenience wrapper around slice to perform int indexing."""
  index, axis = core._canonicalize_dimension(index), int(axis)
  axis_size = operand.shape[axis]
  wrapped_index = index + axis_size if index < 0 else index
  if not 0 <= wrapped_index < axis_size:
    msg = 'index {} is out of bounds for axis {} with size {}'
    raise IndexError(msg.format(index, axis, axis_size))
  result = slice_in_dim(operand, wrapped_index, wrapped_index + 1, 1, axis)
  if keepdims:
    return result
  else:
    return lax.squeeze(result, (axis,))


def dynamic_slice_in_dim(operand: Array, start_index: Array,
                         slice_size: int, axis: int = 0) -> Array:
  """Convenience wrapper around dynamic_slice applying to one dimension."""
  start_indices = [lax._zero(start_index)] * operand.ndim
  slice_sizes = list(operand.shape)

  axis = int(axis)
  start_indices[axis] = start_index
  slice_sizes[axis] = core._canonicalize_dimension(slice_size)
  return dynamic_slice(operand, start_indices, slice_sizes)


def dynamic_index_in_dim(operand: Array, index: Array, axis: int = 0,
                         keepdims: bool = True) -> Array:
  """Convenience wrapper around dynamic_slice to perform int indexing."""
  result = dynamic_slice_in_dim(operand, index, 1, axis)
  if keepdims:
    return result
  else:
    return lax.squeeze(result, (axis,))


def dynamic_update_slice_in_dim(operand: Array, update: Array,
                                start_index: Array, axis: int) -> Array:
  """Convenience wrapper around :func:`dynamic_update_slice` to update a slice
     in a single ``axis``.
  """
  axis = int(axis)
  start_indices = [lax._zero(start_index)] * lax._ndim(operand)
  start_indices[axis] = start_index
  return dynamic_update_slice(operand, update, start_indices)


def dynamic_update_index_in_dim(operand: Array, update: Array, index: Array,
                                axis: int) -> Array:
  """Convenience wrapper around :func:`dynamic_update_slice` to update a slice
     of size 1 in a single ``axis``.
  """
  axis = int(axis)
  if lax._ndim(update) != lax._ndim(operand):
    assert lax._ndim(update) + 1 == lax._ndim(operand)
    update = lax.expand_dims(update, (axis,))
  return dynamic_update_slice_in_dim(operand, update, index, axis)



def _slice_shape_rule(operand, *, start_indices, limit_indices, strides):
  lax._check_shapelike("slice", "start_indices", start_indices)
  lax._check_shapelike("slice", "limit_indices", limit_indices)
  if operand.ndim != len(start_indices):
    msg = ("slice start_indices must have length equal to the number of "
           "dimensions of the operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if len(start_indices) != len(limit_indices):
    msg = ("slice limit_indices must have the same length as start_indices, "
           "got start_indices {} and limit_indices {}.")
    raise TypeError(msg.format(start_indices, limit_indices))
  if not core.greater_equal_shape(operand.shape, limit_indices):
    msg = ("slice limit_indices must be less than or equal to operand shape, "
           "got limit_indices {} for operand shape {}.")
    raise TypeError(msg.format(limit_indices, operand.shape))
  if not all(core.greater_equal_dim(si, 0) for si in start_indices):
    msg = ("slice start_indices must be greater than or equal to zero, "
           "got start_indices of {}.")
    raise TypeError(msg.format(start_indices))
  if not core.greater_equal_shape(limit_indices, start_indices):
    msg = ("slice limit_indices must be greater than or equal to start_indices,"
           " got start_indices {} and limit_indices {}.")
    raise TypeError(msg.format(start_indices, limit_indices))
  if strides is None:
    strides = np.ones(operand.ndim, np.int32)
  else:
    lax._check_shapelike("slice", "strides", strides)
    if len(strides) != operand.ndim:
      msg = ("slice strides must have length equal to the number of dimensions "
             "of the operand, got strides {} for operand shape {}.")
      raise TypeError(msg.format(strides, operand.shape))
    if not core.greater_equal_shape(strides, (0,) * len(strides)):
      msg = "slice strides must be positive, got {}"
      raise TypeError(msg.format(strides))

  diff = core.diff_shape(limit_indices, start_indices)
  return core.stride_shape(diff, (1,) * len(diff), strides)

def _slice_transpose_rule(t, operand, *, start_indices, limit_indices, strides):
  assert ad.is_undefined_primal(operand)
  operand_shape = operand.aval.shape
  if strides is None or np.all(np.equal(strides, 1)):
    pads = zip(start_indices, np.subtract(operand_shape, limit_indices),
               (0,) * len(start_indices))
  else:
    real_limits = np.add(
      start_indices,
      np.where(np.array(t.shape) == 0, 0,
               np.add(1, np.multiply(np.subtract(t.shape, 1), strides))))
    pads = zip(start_indices, np.subtract(operand_shape, real_limits),
               np.subtract(strides, 1))
  result = lax.pad(t, lax._const(t, 0), pads)
  assert result.shape == operand_shape, (
    f"result.shape={result.shape} operand_shape={operand_shape}")
  return [result]


def _slice_batching_rule(batched_args, batch_dims, *, start_indices,
                         limit_indices, strides):
  operand, = batched_args
  bdim, = batch_dims

  new_start_indices = list(start_indices)
  new_start_indices.insert(bdim, 0)

  new_limit_indices = list(limit_indices)
  new_limit_indices.insert(bdim, operand.shape[bdim])

  if strides is None:
    new_strides = None
  else:
    new_strides = list(strides)
    new_strides.insert(bdim, 1)

  out = slice(operand, new_start_indices, new_limit_indices, new_strides)
  return out, bdim

slice_p = standard_primitive(_slice_shape_rule, _input_dtype, 'slice')
ad.deflinear2(slice_p, _slice_transpose_rule)
batching.primitive_batchers[slice_p] = _slice_batching_rule

def _slice_lower(ctx, x, *, start_indices, limit_indices, strides):
  strides = strides or [1] * len(start_indices)
  aval_out, = ctx.avals_out
  if core.is_opaque_dtype(aval_out.dtype):
    return aval_out.dtype._rules.slice_mlir(
        ctx, x, start_indices, limit_indices, strides)
  return mhlo.SliceOp(x,
                      mlir.dense_int_elements(start_indices),
                      mlir.dense_int_elements(limit_indices),
                      mlir.dense_int_elements(strides)).results

mlir.register_lowering(slice_p, _slice_lower)


def _dynamic_slice_shape_rule(operand, *start_indices, slice_sizes):
  if operand.ndim != len(start_indices):
    msg = ("dynamic_slice start_indices must have length equal to the number "
           "of dimensions of the operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if len(start_indices) != len(slice_sizes):
    msg = ("dynamic_slice slice_sizes must have the same length as "
           "start_indices, got start_indices length {} and slice_sizes {}.")
    raise TypeError(msg.format(len(start_indices), slice_sizes))
  if not core.greater_equal_shape(operand.shape, slice_sizes):
    msg = ("slice slice_sizes must be less than or equal to operand shape, "
           "got slice_sizes {} for operand shape {}.")
    raise TypeError(msg.format(slice_sizes, operand.shape))
  if not all(core.greater_equal_dim(ssz, 0) for ssz in slice_sizes):
    msg = ("slice slice_sizes must be greater than or equal to zero, "
           "got slice_sizes of {}.")
    raise TypeError(msg.format(slice_sizes))
  if any(idx.ndim != 0 for idx in start_indices):
    raise TypeError("start_indices arguments to dynamic_slice must be scalars, "
                    f" got indices {start_indices}")
  return tuple(slice_sizes)

def _dynamic_slice_dtype_rule(operand, *start_indices, slice_sizes):
  if any(i.dtype != start_indices[0].dtype or
         not dtypes.issubdtype(i.dtype, np.integer) for i in start_indices):
    msg = ("index arguments to dynamic_slice must be integers of the same "
           "type, got: {}")
    raise TypeError(msg.format(", ".join(i.dtype.name for i in start_indices)))
  return operand.dtype

def _dynamic_slice_jvp(primals, tangents, *, slice_sizes):
  tangent_out = tangents[0]
  if type(tangent_out) is not ad_util.Zero:
    tangent_out = dynamic_slice_p.bind(tangent_out, *primals[1:], slice_sizes=slice_sizes)
  return dynamic_slice_p.bind(primals[0], *primals[1:], slice_sizes=slice_sizes), tangent_out

def _dynamic_slice_transpose_rule(t, operand, *start_indices, slice_sizes):
  assert ad.is_undefined_primal(operand)
  assert all(not ad.is_undefined_primal(s) for s in start_indices)
  operand_shape, operand_dtype = operand.aval.shape, operand.aval.dtype
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)] + [None] * len(start_indices)
  else:
    zeros = lax.full(operand_shape, 0, operand_dtype)
    return ([dynamic_update_slice_p.bind(zeros, t, *start_indices)] +
            [None] * len(start_indices))

def _batch_dynamic_slice_indices(indices, bdims):
  if len(indices) == 0:
    return np.array([], 'int32'), None
  empty_marker = object()
  size = next((x.shape[i] for x, i in zip(indices, bdims) if i is not None),
              empty_marker)
  if size is empty_marker:
    return lax.concatenate([lax.broadcast(i, (1,)) for i in indices], 0), None
  indices = lax.concatenate(
    [lax.broadcast_in_dim(x, (size, 1),
                          broadcast_dimensions=((0,) if i is not None else ()))
     for x, i in zip(indices, bdims)],
    dimension=1)
  return indices, 0

def _dynamic_slice_batching_rule(batched_args, batch_dims, *, slice_sizes):
  # A dynamic slice is a special case of gather; we can delegate to the gather
  # batching rule.
  # TODO(phawkins): consider removing dynamic_slice entirely and using gather
  # always.
  operand, *start_indices = batched_args
  operand_bd, *start_idx_bds = batch_dims
  operand_shape = (operand.shape if operand_bd is batching.not_mapped
                   else tuple(np.delete(operand.shape, operand_bd)))
  dims = tuple(range(len(operand_shape)))
  dnums = GatherDimensionNumbers(offset_dims=dims, collapsed_slice_dims=(),
                                 start_index_map=dims)
  index, index_bdim = _batch_dynamic_slice_indices(start_indices, start_idx_bds)
  return _gather_batching_rule(
    [operand, index], [operand_bd, index_bdim], dimension_numbers=dnums,
    slice_sizes=slice_sizes, unique_indices=True, indices_are_sorted=True,
    mode=GatherScatterMode.PROMISE_IN_BOUNDS, fill_value=None)


dynamic_slice_p = standard_primitive(
    _dynamic_slice_shape_rule, _dynamic_slice_dtype_rule, 'dynamic_slice',
    weak_type_rule=_argnum_weak_type(0))
ad.primitive_jvps[dynamic_slice_p] = _dynamic_slice_jvp
ad.primitive_transposes[dynamic_slice_p] = _dynamic_slice_transpose_rule
batching.primitive_batchers[dynamic_slice_p] = _dynamic_slice_batching_rule

def _dynamic_slice_lower(ctx, x, *start_indices, slice_sizes):
  aval_out, = ctx.avals_out
  if core.is_opaque_dtype(aval_out.dtype):
    return aval_out.dtype._rules.dynamic_slice_mlir(
        ctx, x, start_indices, slice_sizes)
  return mhlo.DynamicSliceOp(x, start_indices,
                             mlir.dense_int_elements(slice_sizes)).results

mlir.register_lowering(dynamic_slice_p, _dynamic_slice_lower)


def _dynamic_update_slice_shape_rule(operand, update, *start_indices):
  if operand.ndim != update.ndim:
    msg = ("dynamic_update_slice update must have the same rank as operand, "
           "got update shape {} for operand shape {}.")
    raise TypeError(msg.format(update.shape, operand.shape))
  if operand.ndim != len(start_indices):
    msg = ("dynamic_update_slice start_indices must have length equal to the "
           "rank of operand, got indices {} for operand shape {}.")
    raise TypeError(msg.format(start_indices, operand.shape))
  if not core.greater_equal_shape(operand.shape, update.shape):
    msg = ("dynamic_update_slice update shape must be smaller than operand "
           "shape, got update shape {} for operand shape {}.")
    raise TypeError(msg.format(update.shape, operand.shape))
  if any(idx.ndim != 0 for idx in start_indices):
    raise TypeError("start_indices arguments to dynamic_update_slice must be "
                    f"scalars, got indices {start_indices}")
  return operand.shape

def _dynamic_update_slice_dtype_rule(operand, update, *start_indices):
  lax._check_same_dtypes("dynamic_update_slice", False, operand.dtype,
                         update.dtype)
  if any(i.dtype != start_indices[0].dtype or
         not dtypes.issubdtype(i.dtype, np.integer) for i in start_indices):
    msg = ("index arguments to dynamic_update_slice must be integers of the "
           "same type, got {}")
    raise TypeError(msg.format(", ".join(i.dtype.name for i in start_indices)))
  return operand.dtype

def _dynamic_update_slice_jvp(primals, tangents):
  operand, update = primals[:2]
  start_indices = primals[2:]
  g_operand, g_update = tangents[:2]
  val_out = dynamic_update_slice_p.bind(operand, update, *start_indices)
  if type(g_operand) is ad_util.Zero and type(g_update) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_update = ad.instantiate_zeros(g_update)
    tangent_out = dynamic_update_slice_p.bind(g_operand, g_update, *start_indices)
  return val_out, tangent_out

def _dynamic_update_slice_transpose_rule(t, operand, update, *start_indices):
  assert all(not ad.is_undefined_primal(x) for x in start_indices)
  if ad.is_undefined_primal(update):
    update_shape = update.aval.shape
  else:
    update_shape = update.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(update.aval) if ad.is_undefined_primal(update) else None
  else:
    dus = dynamic_update_slice_p.bind
    ds = dynamic_slice_p.bind
    zeros = lax._zeros(t, shape=update_shape)
    operand_t = dus(t, zeros, *start_indices) if ad.is_undefined_primal(operand) else None
    update_t = ds(t, *start_indices, slice_sizes=update_shape) if ad.is_undefined_primal(update) else None
  return [operand_t, update_t] + [None] * len(start_indices)

def _dynamic_update_slice_batching_rule(batched_args, batch_dims):
  # A dynamic update slice is a special case of scatter; we can delegate to the
  # scatter batching rule.
  # TODO(phawkins): consider removing dynamic_update_slice entirely and using
  # scatter always.
  operand, update, *start_idx = batched_args
  operand_bd, update_bd, *start_idx_bd = batch_dims
  update_shape = (np.shape(update) if update_bd is batching.not_mapped
                  else tuple(np.delete(np.shape(update), update_bd)))
  dims = tuple(range(len(update_shape)))
  dnums = ScatterDimensionNumbers(update_window_dims=dims,
                                  inserted_window_dims=(),
                                  scatter_dims_to_operand_dims=dims)
  index, index_bdim = _batch_dynamic_slice_indices(start_idx, start_idx_bd)
  return _scatter_batching_rule(
    scatter, (operand, index, update), (operand_bd, index_bdim, update_bd),
    update_jaxpr=None, update_consts=None, dimension_numbers=dnums,
    indices_are_sorted=True, unique_indices=True,
    mode=GatherScatterMode.CLIP)


dynamic_update_slice_p = standard_primitive(
    _dynamic_update_slice_shape_rule, _dynamic_update_slice_dtype_rule,
    'dynamic_update_slice')
ad.primitive_jvps[dynamic_update_slice_p] = _dynamic_update_slice_jvp
ad.primitive_transposes[dynamic_update_slice_p] = \
    _dynamic_update_slice_transpose_rule
batching.primitive_batchers[dynamic_update_slice_p] = \
    _dynamic_update_slice_batching_rule

def _dynamic_update_slice_lower(ctx, x, update, *start_indices):
  aval_out, = ctx.avals_out
  if core.is_opaque_dtype(aval_out.dtype):
    return aval_out.dtype._rules.dynamic_update_slice_mlir(
        ctx, x, update, *start_indices)
  return mhlo.DynamicUpdateSliceOp(mlir.aval_to_ir_type(aval_out), x, update,
                                   start_indices).results

mlir.register_lowering(dynamic_update_slice_p, _dynamic_update_slice_lower)


def _gather_dimensions_proto(
    indices_shape: Sequence[int], dimension_numbers: GatherDimensionNumbers
) -> xla_client.GatherDimensionNumbers:
  assert type(dimension_numbers) is GatherDimensionNumbers
  proto = xla_client.GatherDimensionNumbers()
  proto.offset_dims.extend(dimension_numbers.offset_dims)
  proto.collapsed_slice_dims.extend(dimension_numbers.collapsed_slice_dims)
  proto.start_index_map.extend(dimension_numbers.start_index_map)
  assert len(indices_shape) > 0, indices_shape
  proto.index_vector_dim = len(indices_shape) - 1
  return proto

def _gather_dtype_rule(operand, indices, *, fill_value, **kwargs):
  if not dtypes.issubdtype(indices.dtype, np.integer):
    raise ValueError("indices must have an integer type")
  return dtypes.canonicalize_dtype(operand.dtype)

_rank = lambda arr: len(arr.shape)

def _is_sorted(dims, op_name, name):
  for i in range(1, len(dims)):
    if dims[i] < dims[i - 1]:
      raise TypeError(f"{name} in {op_name} op must be sorted; got {dims}")

def _sorted_dims_in_range(dims, rank, op_name, name):
  if len(dims) == 0:
    return
  invalid_dim = None
  if dims[0] < 0:
    invalid_dim = dims[0]
  elif dims[-1] >= rank:
    invalid_dim = dims[-1]
  if invalid_dim:
    raise TypeError(f"Invalid {name} set in {op_name} op; valid range is "
                    f"[0, {rank}); got: {invalid_dim}.")

def _no_duplicate_dims(dims, op_name, name):
  if len(set(dims)) != len(dims):
    raise TypeError(f"{name} in {op_name} op must not repeat; got: {dims}.")

def _gather_shape_rule(operand, indices, *, dimension_numbers,
                       slice_sizes, unique_indices, indices_are_sorted,
                       mode, fill_value):
  """Validates the well-formedness of the arguments to Gather.

  The code implements the checks based on the detailed operation semantics of
  XLA's `Gather <https://www.tensorflow.org/xla/operation_semantics#gather>`_
  operator and following the outline of the implementation of
  ShapeInference::InferGatherShape in TensorFlow.
  """

  offset_dims = dimension_numbers.offset_dims
  collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
  start_index_map = dimension_numbers.start_index_map

  # Note: in JAX, index_vector_dim is always computed as below, cf. the
  # documentation of the GatherDimensionNumbers class.
  index_vector_dim = _rank(indices) - 1

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if _rank(indices) < index_vector_dim or index_vector_dim < 0:
    raise TypeError(f"Gather index leaf dimension must be within [0, rank("
                    f"indices) + 1). rank(indices) is {_rank(indices)} and "
                    f"gather index leaf dimension is {index_vector_dim}.")

  expanded_indices_shape = list(indices.shape)

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if len(expanded_indices_shape) == index_vector_dim:
    expanded_indices_shape.append(1)

  # Start ValidateGatherDimensions
  # In the error messages output by XLA, "offset_dims" is called "Output window
  # dimensions" in error messages. For consistency's sake, our error messages
  # stick to "offset_dims".
  _is_sorted(offset_dims, "gather", "offset_dims")
  _no_duplicate_dims(offset_dims, "gather", "offset_dims")

  output_offset_dim_count = len(offset_dims)
  output_shape_rank = len(offset_dims) + _rank(indices) - 1

  for i in range(output_offset_dim_count):
    offset_dim = offset_dims[i]
    if offset_dim < 0 or offset_dim >= output_shape_rank:
      raise TypeError(f"Offset dimension {i} in gather op is out of bounds; "
                      f"got {offset_dim}, but should have been in "
                      f"[0, {output_shape_rank})")

  if len(start_index_map) != indices.shape[index_vector_dim]:
    raise TypeError(f"Gather op has {len(start_index_map)} elements in "
                    f"start_index_map and the bound of dimension "
                    f"index_vector_dim={index_vector_dim} of indices is "
                    f"{indices.shape[index_vector_dim]}. These two "
                    f"numbers must be equal.")

  for i in range(len(start_index_map)):
    operand_dim_for_start_index_i = start_index_map[i]
    if (operand_dim_for_start_index_i < 0 or
        operand_dim_for_start_index_i >= _rank(operand)):
      raise TypeError(f"Invalid start_index_map; domain is "
                      f"[0, {_rank(operand)}), got: "
                      f"{i}->{operand_dim_for_start_index_i}.")

  _no_duplicate_dims(start_index_map, "gather", "start_index_map")

  # _is_sorted and _sorted_dims_in_range are checked in the opposite order
  # compared to the XLA implementation. In cases when the input is not sorted
  # AND there are problematic collapsed_slice_dims, the error message will thus
  # be different.
  _is_sorted(collapsed_slice_dims, "gather", "collapsed_slice_dims")
  _sorted_dims_in_range(collapsed_slice_dims, _rank(operand), "gather",
                        "collapsed_slice_dims")
  _no_duplicate_dims(collapsed_slice_dims, "gather", "collapsed_slice_dims")
  # End ValidateGatherDimensions

  if _rank(operand) != len(slice_sizes):
    raise TypeError(f"Gather op must have one slice size for every input "
                    f"dimension; got: len(slice_sizes)={len(slice_sizes)}, "
                    f"input_shape.rank={_rank(operand)}")

  if len(slice_sizes) != len(offset_dims) + len(collapsed_slice_dims):
    raise TypeError(f"All components of the offset index in a gather op must "
                    f"either be a offset dimension or explicitly collapsed; "
                    f"got len(slice_sizes)={len(slice_sizes)}, "
                    f"output_slice_sizes={offset_dims}, collapsed_slice_dims="
                    f"{collapsed_slice_dims}.")

  for i in range(len(slice_sizes)):
    slice_size = slice_sizes[i]
    corresponding_input_size = operand.shape[i]

    if not (core.greater_equal_dim(slice_size, 0) and
            core.greater_equal_dim(corresponding_input_size, slice_size)):
      raise TypeError(f"Slice size at index {i} in gather op is out of range, "
                      f"must be within [0, {corresponding_input_size} + 1), "
                      f"got {slice_size}.")

  for i in range(len(collapsed_slice_dims)):
    bound = slice_sizes[collapsed_slice_dims[i]]
    if bound != 1:
      raise TypeError(f"Gather op can only collapse slice dims with bound 1, "
                      f"but bound is {bound} for index "
                      f"{collapsed_slice_dims[i]} at position {i}.")

  expanded_indices_shape.pop(index_vector_dim)
  indices_shape = iter(expanded_indices_shape)

  slice_sizes = iter(np.delete(slice_sizes, collapsed_slice_dims))
  return tuple(next(slice_sizes) if i in offset_dims
               else next(indices_shape) for i in range(output_shape_rank))


def _gather_fill(operand, indices, *, dimension_numbers, slice_sizes,
                 unique_indices, indices_are_sorted, fill_value,
                 output_shape):
  """Lowers a FILL_OR_DROP gather as a PROMISE_IN_BOUNDS gather with masking."""
  dnums = dimension_numbers
  intarray = partial(np.array, dtype=np.int64)
  operand_dims = lax.shape_as_value(operand.shape)
  indices = lax.convert_element_type(indices, np.int64)
  num_batch_dims = len(indices.shape) - 1

  upper_bound = (
      operand_dims[intarray(dnums.start_index_map)] -
      lax.shape_as_value(slice_sizes)[intarray(dnums.start_index_map)])
  mask = lax.bitwise_and(
      lax.ge(indices, np.int64(0)),
      lax.le(indices, lax.expand_dims(upper_bound, tuple(range(num_batch_dims)))))
  mask = lax._reduce_and(mask, [num_batch_dims])

  # Computes the output shape and the positions of the batch dimensions in the
  # output
  output_ndims = num_batch_dims + len(dnums.offset_dims)
  batch_dims_in_output = np.delete(np.arange(output_ndims),
                                   dnums.offset_dims)

  # We don't consume unique_indices directly in gather(), only in its transpose
  # (scatter).
  gather_out = gather(operand, indices, dnums, slice_sizes,
                      indices_are_sorted=indices_are_sorted,
                      mode=GatherScatterMode.PROMISE_IN_BOUNDS)
  return lax.select(
    lax.broadcast_in_dim(mask, output_shape, batch_dims_in_output),
    gather_out, lax.full_like(gather_out, fill_value=fill_value))


def _gather_jvp_rule(g, operand, indices, *, dimension_numbers,
                     slice_sizes, unique_indices, indices_are_sorted, mode,
                     fill_value):
  return gather(g, indices, dimension_numbers, slice_sizes,
                unique_indices=unique_indices,
                indices_are_sorted=indices_are_sorted, mode=mode,
                fill_value=0)

def _gather_transpose_rule(t, operand, indices, *, dimension_numbers,
                           slice_sizes, unique_indices, indices_are_sorted,
                           mode, fill_value):
  assert ad.is_undefined_primal(operand)
  operand_shape = operand.aval.shape
  if type(t) is ad_util.Zero:
    out = ad_util.Zero(operand.aval)
  else:
    zeros = lax.full(operand_shape, lax._zero(t))
    scatter_dnums = ScatterDimensionNumbers(
      update_window_dims=dimension_numbers.offset_dims,
      inserted_window_dims=dimension_numbers.collapsed_slice_dims,
      scatter_dims_to_operand_dims=dimension_numbers.start_index_map)
    out = scatter_add(zeros, indices, t, scatter_dnums,
                      unique_indices=unique_indices,
                      indices_are_sorted=indices_are_sorted,
                      mode=mode)
  return [out, None]

def _gather_batching_rule(batched_args, batch_dims, *, dimension_numbers,
                          slice_sizes, unique_indices, indices_are_sorted,
                          mode, fill_value):
  operand, indices = batched_args
  operand_bdim, indices_bdim = batch_dims

  if operand_bdim is not None and indices_bdim is None:
    operand = batching.moveaxis(operand, operand_bdim, 0)
    slice_sizes = (operand.shape[0],) + slice_sizes
    offset_dims = (0,) + tuple(np.add(1, dimension_numbers.offset_dims))
    collapsed_slice_dims = tuple(np.add(1, dimension_numbers.collapsed_slice_dims))
    start_index_map = tuple(np.add(1, dimension_numbers.start_index_map))
    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map)
    return gather(operand, indices, dimension_numbers=dnums,
                  slice_sizes=slice_sizes, unique_indices=unique_indices,
                  indices_are_sorted=indices_are_sorted, mode=mode,
                  fill_value=fill_value), 0

  elif operand_bdim is None and indices_bdim is not None:
    indices = batching.moveaxis(indices, indices_bdim, 0)
    offset_dims = tuple(np.add(1, dimension_numbers.offset_dims))
    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=dimension_numbers.collapsed_slice_dims,
        start_index_map=dimension_numbers.start_index_map)
    # If batching indexed accesses into the same array, the batched gather may
    # no longer have sorted or unique indices.
    return gather(operand, indices, dimension_numbers=dnums,
                  slice_sizes=slice_sizes, unique_indices=False,
                  indices_are_sorted=False, mode=mode, fill_value=fill_value), 0

  else:
    # move batch dimensions to the front to simplify logic
    operand = batching.moveaxis(operand, operand_bdim, 0)
    indices = batching.moveaxis(indices, indices_bdim, 0)

    # This slightly awkward special case is needed because the shape rule for
    # gather does not allow size-1 slices out of a size-0 dimension, even if
    # the number of slices is zero. Likely the best fix would be to change the
    # definition of gather() so it can be batched without the construction of
    # an explicit iota of size-1 slices.
    if core.symbolic_equal_dim(operand.shape[0], 0):
      output_shape = _gather_shape_rule(
          core.ShapedArray(operand.shape[1:], operand.dtype),
          core.ShapedArray(indices.shape[1:],
                           dtypes.canonicalize_dtype(indices.dtype)),
          dimension_numbers=dimension_numbers, slice_sizes=slice_sizes,
          unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
          mode=mode, fill_value=fill_value)
      return lax.full((0,) + output_shape, lax._zero(operand)), 0

    # Example: user code had indices shape (3, 4, 5), and we have to deal with
    # indices shape (7, 3, 4, 5). We transform that to indices of shape
    # (7, 3, 4, 6) where we concatenated an iota that counts along our batch
    # dimension to the front of the ndindex.
    count_shape = list(indices.shape)
    count_shape[-1] = 1
    counts = lax.broadcasted_iota(indices.dtype, tuple(count_shape), 0)
    indices = lax.concatenate([counts, indices], len(count_shape) - 1)

    slice_sizes = (1,) + slice_sizes
    collapsed_slice_dims = (0,) + tuple(np.add(1, dimension_numbers.collapsed_slice_dims))
    offset_dims = tuple(np.add(1, dimension_numbers.offset_dims))
    start_index_map = (0,) + tuple(np.add(1, dimension_numbers.start_index_map))

    dnums = GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map)
    return gather(operand, indices, dimension_numbers=dnums,
                  slice_sizes=slice_sizes, unique_indices=unique_indices,
                  indices_are_sorted=indices_are_sorted, mode=mode,
                  fill_value=fill_value), 0

gather_p = standard_primitive(
    _gather_shape_rule, _gather_dtype_rule, 'gather',
    weak_type_rule=_argnum_weak_type(0))
ad.defjvp(gather_p, _gather_jvp_rule, None)

ad.primitive_transposes[gather_p] = _gather_transpose_rule
batching.primitive_batchers[gather_p] = _gather_batching_rule



def _gather_lower(ctx, operand, indices, *,
                  dimension_numbers, slice_sizes, unique_indices,
                  indices_are_sorted, mode, fill_value):
  aval_out, = ctx.avals_out
  if core.is_opaque_dtype(aval_out.dtype):
    return aval_out.dtype._rules.gather_mlir(
        ctx, operand, indices, dimension_numbers=dimension_numbers,
        slice_sizes=slice_sizes, unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted, mode=mode, fill_value=fill_value)

  if mode == GatherScatterMode.FILL_OR_DROP:
    gather_fill_fn = mlir.lower_fun(_gather_fill, multiple_results=False)
    return gather_fill_fn(
        ctx, operand, indices,
        dimension_numbers=dimension_numbers, slice_sizes=slice_sizes,
        unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
        fill_value=fill_value, output_shape=aval_out.shape)

  assert mode in (GatherScatterMode.PROMISE_IN_BOUNDS,
                  GatherScatterMode.CLIP), mode
  dnums = mhlo.GatherDimensionNumbers.get(
    collapsed_slice_dims=list(dimension_numbers.collapsed_slice_dims),
    index_vector_dim=len(ctx.avals_in[1].shape) - 1,
    offset_dims=list(dimension_numbers.offset_dims),
    start_index_map=list(dimension_numbers.start_index_map))
  return mhlo.GatherOp(
      operand,
      indices,
      dnums,
      mlir.dense_int_elements(slice_sizes),
      indices_are_sorted=ir.BoolAttr.get(indices_are_sorted)).results

mlir.register_lowering(gather_p, _gather_lower)

def _scatter_dimensions_proto(
    indices_shape: Sequence[int], dimension_numbers: ScatterDimensionNumbers
) -> xla_client.ScatterDimensionNumbers:
  assert type(dimension_numbers) is ScatterDimensionNumbers
  proto = xla_client.ScatterDimensionNumbers()
  proto.update_window_dims.extend(dimension_numbers.update_window_dims)
  proto.inserted_window_dims.extend(dimension_numbers.inserted_window_dims)
  proto.scatter_dims_to_operand_dims.extend(
      dimension_numbers.scatter_dims_to_operand_dims)
  assert len(indices_shape) > 0, indices_shape
  proto.index_vector_dim = len(indices_shape) - 1
  return proto

def _scatter_dtype_rule(operand, indices, updates, **kwargs):
  if not dtypes.issubdtype(indices.dtype, np.integer):
    raise ValueError("indices must have an integer type")
  lax._check_same_dtypes("scatter", False, operand.dtype, updates.dtype)
  return dtypes.canonicalize_dtype(operand.dtype)

def _scatter_shape_rule(operand, indices, updates, *, update_jaxpr,
                        update_consts, dimension_numbers, indices_are_sorted,
                        unique_indices, mode):
  """Validates the well-formedness of the ``dimension_numbers`` argument to
  Scatter.

  The code implements the checks based on the detailed operation semantics of
  XLA's `Scatter <https://www.tensorflow.org/xla/operation_semantics#scatter>`_
  operator and following the outline of the implementation of
  ShapeInference::InferScatterShape in TensorFlow.
  """

  update_window_dims = dimension_numbers.update_window_dims
  inserted_window_dims = dimension_numbers.inserted_window_dims
  scatter_dims_to_operand_dims = dimension_numbers.scatter_dims_to_operand_dims
  # Note: in JAX, index_vector_dim is always computed as below, cf. the
  # documentation of the ScatterDimensionNumbers class.
  index_vector_dim = _rank(indices) - 1

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if _rank(indices) < index_vector_dim or index_vector_dim < 0:
    raise TypeError(f"Scatter index leaf dimension must be within [0, "
                    f"rank(indices) + 1). rank(indices) is {_rank(indices)} "
                    f"and scatter index leaf dimension is {index_vector_dim}.")

  expanded_indices_shape = list(indices.shape)
  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
  if len(expanded_indices_shape) == index_vector_dim:
    expanded_indices_shape.append(1)

  expected_updates_rank = (len(expanded_indices_shape) - 1 +
                           len(update_window_dims))

  if _rank(updates) != expected_updates_rank:
    raise TypeError(f"Updates tensor must be of rank {expected_updates_rank}; "
                    f"got {_rank(updates)}.")

  # Validate update_window_dims
  _is_sorted(update_window_dims, "scatter", "update_window_dims")
  _no_duplicate_dims(update_window_dims, "scatter", "update_window_dims")
  _sorted_dims_in_range(update_window_dims, _rank(updates), "scatter",
                        "update_window_dims")

  # Validate inserted_window_dims
  _is_sorted(inserted_window_dims, "scatter", "inserted_window_dims")
  _no_duplicate_dims(inserted_window_dims, "scatter", "inserted_window_dims")
  _sorted_dims_in_range(inserted_window_dims, _rank(operand), "scatter",
                        "inserted_window_dims")

  # Validate window_size
  window_size = len(update_window_dims) + len(inserted_window_dims)
  if _rank(operand) != window_size:
    raise TypeError(f"Scatter op has window of size {window_size}; doesn't "
                    f"match operand of rank {_rank(operand)}.")

  # Validate scatter_dims_to_operand_dims
  if (len(scatter_dims_to_operand_dims) !=
      indices.shape[index_vector_dim]):
    raise TypeError(f"Scatter op has {len(scatter_dims_to_operand_dims)} "
                    f"elements in scatter_dims_to_operand_dims and the bound "
                    f"of dimension index_vector_dim={index_vector_dim} of "
                    f"indices is {indices.shape[index_vector_dim]}. These two "
                    f"numbers must be equal")

  for i in range(len(scatter_dims_to_operand_dims)):
    dim = scatter_dims_to_operand_dims[i]
    if dim < 0 or dim >= _rank(operand):
      raise TypeError(f"Invalid scatter_dims_to_operand_dims mapping; domain "
                      f"is [0, {_rank(operand)}), got: {i}->{dim}.")

  _no_duplicate_dims(scatter_dims_to_operand_dims, "scatter",
                     "scatter_dims_to_operand_dims")

  max_update_slice_sizes = [operand.shape[i] for i in range(len(operand.shape))
                            if not i in set(inserted_window_dims)]

  for i in range(len(update_window_dims)):
    update_window_dim = update_window_dims[i]
    if not core.greater_equal_dim(max_update_slice_sizes[i], updates.shape[update_window_dim]):
      raise TypeError(f"Bounds of the window dimensions of updates must not "
                      f"exceed the bounds of the corresponding dimensions of "
                      f"operand. For dimension {update_window_dim}, updates "
                      f"bound is {updates.shape[update_window_dim]}, operand "
                      f"bound is {max_update_slice_sizes[i]}.")

  update_scatter_dims = [dim for dim in range(_rank(updates)) if dim not in
                         set(update_window_dims)]

  scatter_dims_seen = 0
  for i in update_scatter_dims:
    if scatter_dims_seen == index_vector_dim:
      scatter_dims_seen += 1
    if not core.symbolic_equal_dim(updates.shape[i], expanded_indices_shape[scatter_dims_seen]):
      raise TypeError(f"Bounds of the scatter dimensions of updates must be "
                      f"the same as the bounds of the corresponding dimensions "
                      f"of scatter indices. For scatter dimension {i}, updates "
                      f"bound is {updates.shape[i]}, indices bound is "
                      f"{expanded_indices_shape[scatter_dims_seen]}.")
    scatter_dims_seen += 1

  return operand.shape


def _clamp_scatter_indices(operand, indices, updates, *, dnums):
  """Clamps `indices` to be in-range for a scatter."""
  slice_sizes = []
  pos = 0
  for i in range(len(operand.shape)):
    if i in dnums.inserted_window_dims:
      slice_sizes.append(1)
    else:
      slice_sizes.append(updates.shape[dnums.update_window_dims[pos]])
      pos += 1

  upper_bounds: core.Shape = tuple(operand.shape[i] - slice_sizes[i]
                                   for i in dnums.scatter_dims_to_operand_dims)
  # Stack upper_bounds into a DeviceArray[n]
  upper_bound = lax.shape_as_value(upper_bounds)
  upper_bound = lax.min(upper_bound, np.iinfo(indices.dtype).max)
  upper_bound = lax.broadcast_in_dim(upper_bound, indices.shape,
                                     (len(indices.shape) - 1,))
  return lax.clamp(np.int64(0), lax.convert_element_type(indices, np.int64),
                   upper_bound)

def _scatter_add_jvp(primals, tangents, *, update_jaxpr, update_consts,
                     dimension_numbers, indices_are_sorted, unique_indices,
                     mode):
  operand, indices, updates = primals
  g_operand, g_indices, g_updates = tangents
  del g_indices  # ignored
  val_out = scatter_add_p.bind(
      operand, indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode)
  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_updates = ad.instantiate_zeros(g_updates)
    tangent_out = scatter_add_p.bind(
        g_operand, indices, g_updates, update_jaxpr=update_jaxpr,
        update_consts=update_consts, dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
        mode=mode)
  return val_out, tangent_out

def _scatter_add_transpose_rule(t, operand, indices, updates, *,
                                update_jaxpr, update_consts, dimension_numbers,
                                indices_are_sorted, unique_indices, mode):
  assert not ad.is_undefined_primal(indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      operand_t = t

    if ad.is_undefined_primal(updates):
      gather_dnums = GatherDimensionNumbers(
        offset_dims=dimension_numbers.update_window_dims,
        collapsed_slice_dims=dimension_numbers.inserted_window_dims,
        start_index_map=dimension_numbers.scatter_dims_to_operand_dims)
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if i in dimension_numbers.inserted_window_dims:
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(t, indices, dimension_numbers=gather_dnums,
                        slice_sizes=slice_sizes, mode=mode, fill_value=0)
  return [operand_t, None, update_t]

def _scatter_mul_transpose_rule(t, operand, indices, updates, *,
                                update_jaxpr, update_consts, dimension_numbers,
                                indices_are_sorted, unique_indices, mode):
  assert not ad.is_undefined_primal(indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      operand_t = scatter_mul(
          t, indices, updates, dimension_numbers=dimension_numbers,
          indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
          mode=mode)
    if ad.is_undefined_primal(updates):
      if not unique_indices:
        raise NotImplementedError(
          "scatter_mul gradients are only implemented if `unique_indices=True`")
      gather_dnums = GatherDimensionNumbers(
        offset_dims=dimension_numbers.update_window_dims,
        collapsed_slice_dims=dimension_numbers.inserted_window_dims,
        start_index_map=dimension_numbers.scatter_dims_to_operand_dims)
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if i in dimension_numbers.inserted_window_dims:
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(lax.mul(t, operand), indices,
                        dimension_numbers=gather_dnums, slice_sizes=slice_sizes,
                        mode=mode, fill_value=0)
  return [operand_t, None, update_t]


def _scatter_batching_rule(scatter_op, batched_args, batch_dims, *,
                           update_jaxpr, update_consts, dimension_numbers,
                           indices_are_sorted, unique_indices, mode):
  operand, indices, updates = batched_args
  operand_bdim, indices_bdim, updates_bdim = batch_dims
  del update_jaxpr, update_consts  # Unused.

  # move the operand batch dim to the front if it is not None, otherwise create
  # it at the front (so that we can scatter into it)
  size = next(x.shape[ax] for x, ax in zip(batched_args, batch_dims)
              if ax is not None)
  operand = batching.bdim_at_front(operand, operand_bdim, size)
  operand_bdim = 0

  updates = batching.bdim_at_front(updates, updates_bdim, size)

  if indices_bdim is None:
    inserted_window_dims = tuple(np.add(1, dimension_numbers.inserted_window_dims))
    update_window_dims = (0,) + tuple(np.add(1, dimension_numbers.update_window_dims))
    scatter_dims_to_operand_dims = tuple(np.add(1, dimension_numbers.scatter_dims_to_operand_dims))
    dnums = ScatterDimensionNumbers(
        update_window_dims=update_window_dims,
        inserted_window_dims=inserted_window_dims,
        scatter_dims_to_operand_dims=scatter_dims_to_operand_dims)
    return scatter_op(
      operand, indices, updates, dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode), 0


  # see the third case in _gather_batching_rule for comparison and comments
  indices = batching.bdim_at_front(indices, indices_bdim, size)

  count_shape = list(indices.shape)
  count_shape[-1] = 1
  counts = lax.broadcasted_iota(indices.dtype, tuple(count_shape), 0)
  indices = lax.concatenate([counts, indices], len(count_shape) - 1)

  update_window_dims = tuple(np.add(1, dimension_numbers.update_window_dims))
  inserted_window_dims = (0,) + tuple(np.add(1, dimension_numbers.inserted_window_dims))
  scatter_dims_to_operand_dims = (0,) + tuple(np.add(1, dimension_numbers.scatter_dims_to_operand_dims))

  dnums = ScatterDimensionNumbers(
      update_window_dims=update_window_dims,
      inserted_window_dims=inserted_window_dims,
      scatter_dims_to_operand_dims=scatter_dims_to_operand_dims)
  return scatter_op(
      operand, indices, updates, dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode), 0

scatter_add_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-add',
    weak_type_rule=_argnum_weak_type(0))
ad.primitive_jvps[scatter_add_p] = _scatter_add_jvp
ad.primitive_transposes[scatter_add_p] = _scatter_add_transpose_rule
batching.primitive_batchers[scatter_add_p] = (
  partial(_scatter_batching_rule, scatter_add))

scatter_mul_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-mul',
    weak_type_rule=_argnum_weak_type(0))

def _scatter_mul_jvp_rhs(g, x, i, y, *, dimension_numbers,
                         indices_are_sorted, unique_indices, mode, **kw):
  if not unique_indices:
    raise NotImplementedError(
      "scatter_mul gradients are only implemented if `unique_indices=True`")
  return lax.mul(x, scatter_add(
      lax.zeros_like_array(x), i, g, dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode))

ad.defjvp(scatter_mul_p,
          lambda g, x, i, y, **kw: scatter_mul_p.bind(g, i, y, **kw),
          None,
          _scatter_mul_jvp_rhs)
ad.primitive_transposes[scatter_mul_p] = _scatter_mul_transpose_rule
batching.primitive_batchers[scatter_mul_p] = (
  partial(_scatter_batching_rule, scatter_mul))

def _scatter_extremal_jvp(scatter_op, primals, tangents, update_jaxpr,
                          update_consts, dimension_numbers,
                          indices_are_sorted, unique_indices, mode):
  operand, indices, updates = primals
  g_operand, g_indices, g_updates = tangents

  scatter_dnums = dimension_numbers
  updates_shape = updates.shape

  val_out = scatter_op.bind(
      operand, indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=scatter_dnums,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices, mode=mode)

  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(val_out)
  else:
    g_operand = ad.instantiate_zeros(g_operand)
    g_updates = ad.instantiate_zeros(g_updates)

    # gather_dnums and slice_sizes define the gather op that is the inverse of
    # the scatter op specified by scatter_dnums
    gather_dnums = GatherDimensionNumbers(
        offset_dims=scatter_dnums.update_window_dims,
        collapsed_slice_dims=scatter_dnums.inserted_window_dims,
        start_index_map=scatter_dnums.scatter_dims_to_operand_dims)

    slice_sizes = []
    pos = 0
    for i in range(len(operand.shape)):
      if i in scatter_dnums.inserted_window_dims:
        slice_sizes.append(1)
      else:
        slice_sizes.append(updates_shape[scatter_dnums.update_window_dims[pos]])
        pos += 1

    # For consistency with other max operations, if there are two or more values
    # in updates that are contending to replace the same index location, the
    # resulting tangent at that location will be the average of the associated
    # tangents for the values in updates.

    initial_vals = gather(
        operand, indices, gather_dnums, np.array(slice_sizes))

    target_vals = gather(
        val_out, indices, gather_dnums, np.array(slice_sizes))

    successful_updates = (updates == target_vals)
    retained_values = (initial_vals == target_vals)

    num_updates = gather(
        scatter_add(
            lax._zeros(operand), indices,
            lax.select(successful_updates, lax._ones(updates),
                       lax._zeros(updates)),
            scatter_dnums),
        indices,
        gather_dnums,
        np.array(slice_sizes))

    num_refs = gather(
        scatter_add(lax._zeros(operand),
                    indices,
                    lax._ones(updates),
                    scatter_dnums),
        indices,
        gather_dnums,
        np.array(slice_sizes))

    updates_normalizer = lax.select(retained_values,
                                    1.0 / (num_updates + 1),
                                    1.0 / num_updates)

    updates_coef = lax.select(successful_updates,
                              updates_normalizer,
                              lax._zeros(updates))

    operand_normalizer = lax.select(retained_values,
                                    1.0 / (num_updates + 1),
                                    lax._zeros(num_updates))

    operand_coef = (-1.0 + operand_normalizer) / num_refs

    # This can be simplified once scatter has transpose implemented
    target_tangents = gather(
        g_operand, indices, gather_dnums, np.array(slice_sizes))

    tangent_updates = (target_tangents * operand_coef +
                       g_updates * updates_coef)

    tangent_out = scatter_add(g_operand,
                              indices,
                              tangent_updates,
                              scatter_dnums,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices,
                              mode=mode)

  return val_out, tangent_out

scatter_min_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-min',
    weak_type_rule=_argnum_weak_type(0))
batching.primitive_batchers[scatter_min_p] = (
  partial(_scatter_batching_rule, scatter_min))
ad.primitive_jvps[scatter_min_p] = partial(_scatter_extremal_jvp, scatter_min_p)

scatter_max_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter-max',
    weak_type_rule=_argnum_weak_type(0))
batching.primitive_batchers[scatter_max_p] = (
  partial(_scatter_batching_rule, scatter_max))
ad.primitive_jvps[scatter_max_p] = partial(_scatter_extremal_jvp, scatter_max_p)

def _scatter_jvp(primals, tangents, *, update_jaxpr, update_consts,
                 dimension_numbers, indices_are_sorted, unique_indices,
                 mode):
  operand, indices, updates = primals
  g_operand, g_indices, g_updates = tangents
  dnums = dimension_numbers

  if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
    val_out = scatter_p.bind(
      operand, indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
      mode=mode)
    return val_out, ad_util.Zero.from_value(val_out)

  g_operand = ad.instantiate_zeros(g_operand)
  g_updates = ad.instantiate_zeros(g_updates)

  if unique_indices:
    # If the user has promised that the updates don't overlap, we can use a much
    # simpler JVP.
    val_out = scatter_p.bind(
      operand, indices, updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=True, mode=mode)
    tangent_out = scatter_p.bind(
      g_operand, indices, g_updates, update_jaxpr=update_jaxpr,
      update_consts=update_consts, dimension_numbers=dnums,
      indices_are_sorted=indices_are_sorted, unique_indices=True, mode=mode)
    return val_out, tangent_out

  # If there are overlapping indices in the scatter, it is unspecified which
  # update "wins". So we use the following perhaps surprising scheme:
  # a) attach a positive ID to each update in updates, and perform the scatter
  #    on the IDs
  # b) perform the inverse gather on the scattered IDs (similar to
  #    _scatter_add_transpose).
  # c) use the gathered IDs to mask the primal and tangent values.
  # d) perform a scatter-add on the masked primal and tangent values. A benefit
  #    of using scatter-add here is that we don't need a `scatter` transpose
  #    rule.


  # a) attach a positive ID to each update in `updates`, and perform a scatter
  #    on the IDs.
  ids_shape = np.array(updates.shape, dtype=np.int64)
  ids_shape[dnums.update_window_dims,] = 1
  num_ids = np.prod(ids_shape)
  id_dtype = np.uint32 if (num_ids + 1) < np.iinfo(np.uint32).max else np.uint64
  update_ids = lax.add(lax.reshape(lax.iota(id_dtype, num_ids), ids_shape),
                       lax._ones(updates, dtype=id_dtype))

  scattered_ids = scatter(lax.full(operand.shape, 0, id_dtype),
                          indices, update_ids, dnums,
                          indices_are_sorted=indices_are_sorted,
                          unique_indices=unique_indices, mode=mode)

  # b) compute the inverse gather that "undoes" the scatter on the id values.
  gather_dnums = GatherDimensionNumbers(
    offset_dims=dnums.update_window_dims,
    collapsed_slice_dims=dnums.inserted_window_dims,
    start_index_map=dnums.scatter_dims_to_operand_dims)
  slice_sizes = []
  pos = 0
  for i in range(len(scattered_ids.shape)):
    if i in dnums.inserted_window_dims:
      slice_sizes.append(1)
    else:
      slice_sizes.append(updates.shape[dnums.update_window_dims[pos]])
      pos += 1
  gathered_update_ids = gather(scattered_ids, indices,
                               dimension_numbers=gather_dnums,
                               slice_sizes=slice_sizes)

  # c) mask off input elements that do not correspond to a primal output.
  masked_operand = lax.select(lax.eq(scattered_ids, lax._zeros(scattered_ids)),
                              operand, lax._zeros(operand))
  masked_updates = lax.select(lax.eq(update_ids,  gathered_update_ids),
                              updates, lax._zeros(updates))
  masked_g_operand = lax.select(lax.eq(scattered_ids, lax._zeros(scattered_ids)),
                                g_operand, lax._zeros(g_operand))
  masked_g_updates = lax.select(lax.eq(update_ids, gathered_update_ids),
                                g_updates, lax._zeros(g_updates))

  # d) perform scatter-adds to compute the primal and tangent outputs.
  val_out = scatter_add(masked_operand, indices, masked_updates,
                        dimension_numbers=dnums,
                        indices_are_sorted=indices_are_sorted,
                        unique_indices=unique_indices, mode=mode)
  tangent_out = scatter_add(masked_g_operand, indices, masked_g_updates,
                            dimension_numbers=dnums,
                            indices_are_sorted=indices_are_sorted,
                            unique_indices=unique_indices, mode=mode)
  return val_out, tangent_out

def _scatter_transpose_rule(t, operand, indices, updates, *,
                            update_jaxpr, update_consts, dimension_numbers,
                            indices_are_sorted, unique_indices, mode):
  if not unique_indices:
    raise NotImplementedError("scatter transpose is only implemented where"
                              "unique_indices=True")
  assert not ad.is_undefined_primal(indices)
  if ad.is_undefined_primal(updates):
    updates_shape = updates.aval.shape
  else:
    updates_shape = updates.shape
  if type(t) is ad_util.Zero:
    operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
  else:
    operand_t = update_t = None
    if ad.is_undefined_primal(operand):
      # Zero out gradient entries that correspond to updated indices.
      mask = scatter(lax._ones(t, dtype=np.bool_), indices,
                     lax.full(updates_shape, False),
                     dimension_numbers=dimension_numbers,
                     indices_are_sorted=indices_are_sorted,
                     unique_indices=True, mode=mode)
      operand_t = lax.select(mask, t, lax._zeros(t))

    if ad.is_undefined_primal(updates):
      gather_dnums = GatherDimensionNumbers(
        offset_dims=dimension_numbers.update_window_dims,
        collapsed_slice_dims=dimension_numbers.inserted_window_dims,
        start_index_map=dimension_numbers.scatter_dims_to_operand_dims)
      slice_sizes = []
      pos = 0
      for i in range(len(t.shape)):
        if i in dimension_numbers.inserted_window_dims:
          slice_sizes.append(1)
        else:
          slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
          pos += 1
      update_t = gather(t, indices, dimension_numbers=gather_dnums,
                        slice_sizes=slice_sizes, mode=mode,
                        fill_value=0)

  return [operand_t, None, update_t]

scatter_p = standard_primitive(
    _scatter_shape_rule, _scatter_dtype_rule, 'scatter',
    weak_type_rule=_argnum_weak_type(0))
ad.primitive_jvps[scatter_p] = _scatter_jvp
ad.primitive_transposes[scatter_p] = _scatter_transpose_rule
batching.primitive_batchers[scatter_p] = (
  partial(_scatter_batching_rule, scatter))


# TODO(wuke): Implement _scatter_reduce_jvp_rule.
def _scatter_reduce_batching_rule(batched_args, batch_dims, *, update_jaxpr,
                                  update_consts, dimension_numbers,
                                  indices_are_sorted, unique_indices, mode):

  def scatter_op(operand, scatter_indices, updates, dimension_numbers, *,
                 indices_are_sorted, unique_indices, mode):
    return scatter_reduce_p.bind(
        operand,
        scatter_indices,
        updates,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=GatherScatterMode.from_any(mode))

  return _scatter_batching_rule(
      scatter_op,
      batched_args,
      batch_dims,
      update_jaxpr=update_jaxpr,
      update_consts=update_consts,
      dimension_numbers=dimension_numbers,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      mode=mode)


scatter_reduce_p = standard_primitive(
    _scatter_shape_rule,
    _scatter_dtype_rule,
    "scatter-reduce",
    weak_type_rule=_argnum_weak_type(0))
batching.primitive_batchers[scatter_reduce_p] = _scatter_reduce_batching_rule


def _scatter_lower(ctx, operand, indices, updates, *,
                   update_jaxpr, update_consts, dimension_numbers,
                   indices_are_sorted, unique_indices, mode):
  if mode == GatherScatterMode.CLIP:
    clip_fn = mlir.lower_fun(_clamp_scatter_indices, multiple_results=False)
    (indices,), = clip_fn(ctx.replace(avals_out=None), operand, indices,
                          updates, dnums=dimension_numbers)

  aval_out, = ctx.avals_out
  dnums = dimension_numbers
  scatter_dnums = mhlo.ScatterDimensionNumbers.get(
    update_window_dims=list(dnums.update_window_dims),
    inserted_window_dims=list(dnums.inserted_window_dims),
    scattered_dims_to_operand_dims=list(dnums.scatter_dims_to_operand_dims),
    index_vector_dim=len(ctx.avals_in[1].shape) - 1)
  result = mlir.aval_to_ir_types(aval_out)
  operand = [operand]
  updates = [updates]
  op = mhlo.ScatterOp(
      result,
      operand,
      indices,
      updates,
      scatter_dnums,
      indices_are_sorted=ir.BoolAttr.get(indices_are_sorted),
      unique_indices=ir.BoolAttr.get(unique_indices))
  scalar_type = mlir.aval_to_ir_type(core.ShapedArray((), aval_out.dtype))
  update = op.update_computation.blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(update):
    update_ctx = ctx.module_context.replace(name_stack=util.new_name_stack())
    if update_jaxpr.effects:
      raise NotImplementedError('Cannot lower effectful `scatter`.')
    out_nodes, _ = mlir.jaxpr_subcomp(
        update_ctx, update_jaxpr, mlir.TokenSet(), update_consts,
        (update.arguments[0],), (update.arguments[1],))
    mhlo.ReturnOp(util.flatten(out_nodes))
  return op.results

mlir.register_lowering(scatter_p, _scatter_lower)
mlir.register_lowering(scatter_add_p, _scatter_lower)
mlir.register_lowering(scatter_mul_p, _scatter_lower)
mlir.register_lowering(scatter_min_p, _scatter_lower)
mlir.register_lowering(scatter_max_p, _scatter_lower)
mlir.register_lowering(scatter_reduce_p, _scatter_lower)


def _real_dtype(dtype): return np.finfo(dtype).dtype

def _scatter_add_lower_gpu(ctx, operand, indices, updates,
                           *, update_jaxpr, update_consts, dimension_numbers,
                           indices_are_sorted, unique_indices, mode):
  operand_aval_in, _, updates_aval_in = ctx.avals_in
  if operand_aval_in.dtype != np.complex128:
    return _scatter_lower(ctx, operand, indices, updates,
                          update_jaxpr=update_jaxpr,
                          update_consts=update_consts,
                          dimension_numbers=dimension_numbers,
                          indices_are_sorted=indices_are_sorted,
                          unique_indices=unique_indices, mode=mode)

  if mode == GatherScatterMode.CLIP:
    clip_fn = mlir.lower_fun(_clamp_scatter_indices, multiple_results=False)
    (indices,), = clip_fn(ctx.replace(avals_out=None), operand, indices, updates,
                          dnums=dimension_numbers)

  aval_out, = ctx.avals_out
  dnums = dimension_numbers
  scatter_dnums = mhlo.ScatterDimensionNumbers.get(
    update_window_dims=list(dnums.update_window_dims),
    inserted_window_dims=list(dnums.inserted_window_dims),
    scattered_dims_to_operand_dims=list(dnums.scatter_dims_to_operand_dims),
    index_vector_dim=len(ctx.avals_in[1].shape) - 1)
  real_dtype = _real_dtype(aval_out.dtype)
  operand_type_part = mlir.aval_to_ir_types(
      core.ShapedArray(aval_out.shape, real_dtype))

  def _scatter(operand_part, updates_part):
    operand_part = [operand_part]
    updates_part = [updates_part]

    scatter = mhlo.ScatterOp(
        operand_type_part,
        operand_part,
        indices,
        updates_part,
        scatter_dnums,
        indices_are_sorted=ir.BoolAttr.get(indices_are_sorted),
        unique_indices=ir.BoolAttr.get(unique_indices))
    scalar_type = mlir.aval_to_ir_type(core.ShapedArray((), real_dtype))
    reducer = scatter.regions[0].blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(reducer):
      add = mhlo.AddOp(*reducer.arguments).result
      mhlo.ReturnOp([add])
    return scatter.result

  real = _scatter(mhlo.RealOp(operand).result, mhlo.RealOp(updates).result)
  imag = _scatter(mhlo.ImagOp(operand).result, mhlo.ImagOp(updates).result)
  return mhlo.ComplexOp(real, imag).results

mlir.register_lowering(scatter_add_p, _scatter_add_lower_gpu, platform="gpu")


def _dynamic_slice_indices(operand, start_indices: Any):
  # Normalize the start_indices w.r.t. operand.shape
  if len(start_indices) != operand.ndim:
    msg = ("Length of slice indices must match number of operand dimensions ({} "
          "vs {})")
    raise ValueError(msg.format(len(start_indices), operand.shape))
  if not isinstance(start_indices, (tuple, list)):
    if start_indices.ndim != 1:
      raise ValueError("Slice indices must be a 1D sequence, got {}"
                       .format(start_indices.shape))
    start_indices = list(start_indices)
  result = []
  for i, d in zip(start_indices, operand.shape):
    if isinstance(i, (int, np.integer)) and core.is_constant_dim(d):
      result.append(lax.convert_element_type(i + d, _dtype(i)) if i < 0 else i)
    else:
      d = lax.convert_element_type(core.dimension_as_value(d), _dtype(i))
      result.append(lax.select(i < 0, i + d, i))
  return result


# TODO(mattjj): getslice is a prototype for dynamic shapes, revise or remove it
def _getslice(x, lo, hi):
  return getslice_p.bind(x, lo, hi)

getslice_p = core.Primitive('getslice')

@getslice_p.def_impl
def getslice_impl(x, lo, hi):
  return x[lo:hi]

def _getslice_staging_rule(trace, x, lo, hi):
  size = lax.make_bint(lax.clamp(0, hi - lo, x.shape[0]), x.shape[0])
  aval = core.DShapedArray((size,), x.dtype, x.weak_type)
  source_info = source_info_util.current()
  out_tracer = pe.DynamicJaxprTracer(trace, aval, source_info)
  invars = map(trace.getvar, [x, lo, hi])
  eqn = pe.new_jaxpr_eqn(invars, [trace.makevar(out_tracer)],
                         getslice_p, {}, source_info)
  trace.frame.eqns.append(eqn)
  return out_tracer
pe.custom_staging_rules[getslice_p] = _getslice_staging_rule

def _getslice_padding_rule(in_avals, out_avals, x, lo, hi):
  xx = lax.concatenate([x, x], 0)
  return [dynamic_slice_in_dim(xx, lo, x.shape[0])]
pe.padding_rules[getslice_p] = _getslice_padding_rule

def _getslice_lower(ctx, x, lo, hi):
  aval_out, = ctx.avals_out
  return mhlo.RealDynamicSliceOp(
      mlir.aval_to_ir_type(aval_out), x,
      mlir.shape_tensor([lo]), mlir.shape_tensor([hi]), mlir.shape_tensor([1])
  ).results
mlir.register_lowering(getslice_p, _getslice_lower)
