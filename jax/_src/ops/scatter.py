# Copyright 2019 Google LLC
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

# Helpers for indexed updates.

import sys
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from jax import core
from jax import lax

from jax._src import dtypes
from jax._src import util
from jax._src.lax import lax as lax_internal
from jax._src.numpy import lax_numpy as jnp


Array = Any
if sys.version_info >= (3, 10):
  from types import EllipsisType
  SingleIndex = Union[None, int, slice, Sequence[int], Array, EllipsisType]
else:
  SingleIndex = Union[None, int, slice, Sequence[int], Array]
Index = Union[SingleIndex, Tuple[SingleIndex, ...]]
Scalar = Union[complex, float, int, np.number]
Numeric = Union[Array, Scalar]


def _scatter_update(x, idx, y, scatter_op, indices_are_sorted,
                    unique_indices, mode=None, normalize_indices=True):
  """Helper for indexed updates.

  Computes the value of x that would result from computing::
    x[idx] op= y
  except in a pure functional way, with no in-place updating.

  Args:
    x: ndarray to be updated.
    idx: None, an integer, a slice, an ellipsis, an ndarray with integer dtype,
      or a tuple of those indicating the locations of `x` into which to scatter-
      update the values in `y`.
    y: values to be scattered.
    scatter_op: callable, one of lax.scatter, lax.scatter_add, lax.scatter_min,
      or lax_scatter_max.
    indices_are_sorted: whether `idx` is known to be sorted
    unique_indices: whether `idx` is known to be free of duplicates

  Returns:
    An ndarray representing an updated `x` after performing the scatter-update.
  """

  x = jnp.asarray(x)
  y = jnp.asarray(y)
  # XLA gathers and scatters are very similar in structure; the scatter logic
  # is more or less a transpose of the gather equivalent.
  treedef, static_idx, dynamic_idx = jnp._split_index_for_jit(idx, x.shape)
  return _scatter_impl(x, y, scatter_op, treedef, static_idx, dynamic_idx,
                       indices_are_sorted, unique_indices, mode,
                       normalize_indices)


# TODO(phawkins): re-enable jit after fixing excessive recompilation for
# slice indexes (e.g., slice(0, 5, None), slice(10, 15, None), etc.).
# @partial(jit, static_argnums=(2, 3, 4))
def _scatter_impl(x, y, scatter_op, treedef, static_idx, dynamic_idx,
                  indices_are_sorted, unique_indices, mode,
                  normalize_indices):
  dtype = lax.dtype(x)
  weak_type = dtypes.is_weakly_typed(x)

  if dtype != dtypes.result_type(x, y):
    # TODO(jakevdp): change this to an error after the deprecation period.
    warnings.warn("scatter inputs have incompatible types: cannot safely cast "
                  f"value from dtype={lax.dtype(y)} to dtype={lax.dtype(x)}. "
                  "In future JAX releases this will result in an error.",
                  FutureWarning)

  idx = jnp._merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx)
  indexer = jnp._index_to_gather(jnp.shape(x), idx,
                                 normalize_indices=normalize_indices)

  # Avoid calling scatter if the slice shape is empty, both as a fast path and
  # to handle cases like zeros(0)[array([], int32)].
  if core.is_empty_shape(indexer.slice_shape):
    return x

  x, y = jnp._promote_dtypes(x, y)

  # Broadcast `y` to the slice output shape.
  y = jnp.broadcast_to(y, tuple(indexer.slice_shape))
  # Collapse any `None`/`jnp.newaxis` dimensions.
  y = jnp.squeeze(y, axis=indexer.newaxis_dims)
  if indexer.reversed_y_dims:
    y = lax.rev(y, indexer.reversed_y_dims)

  # Transpose the gather dimensions into scatter dimensions (cf.
  # lax._gather_transpose_rule)
  dnums = lax.ScatterDimensionNumbers(
    update_window_dims=indexer.dnums.offset_dims,
    inserted_window_dims=indexer.dnums.collapsed_slice_dims,
    scatter_dims_to_operand_dims=indexer.dnums.start_index_map
  )
  out = scatter_op(
    x, indexer.gather_indices, y, dnums,
    indices_are_sorted=indexer.indices_are_sorted or indices_are_sorted,
    unique_indices=indexer.unique_indices or unique_indices,
    mode=mode)
  return lax_internal._convert_element_type(out, dtype, weak_type)



def _get_identity(op, dtype):
  """Get an appropriate identity for a given operation in a given dtype."""
  if op is lax.scatter_add:
    return 0
  elif op is lax.scatter_mul:
    return 1
  elif op is lax.scatter_min:
    if dtype == dtypes.bool_:
      return True
    elif jnp.issubdtype(dtype, jnp.integer):
      return jnp.iinfo(dtype).max
    return float('inf')
  elif op is lax.scatter_max:
    if dtype == dtypes.bool_:
      return False
    elif jnp.issubdtype(dtype, jnp.integer):
      return jnp.iinfo(dtype).min
    return -float('inf')
  else:
    raise ValueError(f"Unrecognized op: {op}")


def _segment_update(name: str,
                    data: Array,
                    segment_ids: Array,
                    scatter_op: Callable,
                    *,
                    identity: Optional[Array] = None,
                    num_segments: Optional[int] = None,
                    indices_are_sorted: bool = False,
                    unique_indices: bool = False,
                    bucket_size: Optional[int] = None,
                    reducer: Optional[Callable] = None,
                    mode: Optional[lax.GatherScatterMode] = None) -> Array:
  jnp._check_arraylike(name, data, segment_ids)
  mode = lax.GatherScatterMode.FILL_OR_DROP if mode is None else mode
  data = jnp.asarray(data)
  segment_ids = jnp.asarray(segment_ids)
  dtype = data.dtype
  if num_segments is None:
    num_segments = jnp.max(segment_ids) + 1
  num_segments = core.concrete_or_error(int, num_segments, "segment_sum() `num_segments` argument.")
  if num_segments is not None and num_segments < 0:
    raise ValueError("num_segments must be non-negative.")

  if identity is None:
    identity = _get_identity(scatter_op, dtype)

  num_buckets = 1 if bucket_size is None \
                  else util.ceil_of_ratio(segment_ids.size, bucket_size)
  if num_buckets == 1:
    out = jnp.full((num_segments,) + data.shape[1:], identity, dtype=dtype)
    return _scatter_update(
      out, segment_ids, data, scatter_op, indices_are_sorted,
      unique_indices, normalize_indices=False, mode=mode)

  # Bucketize indices and perform segment_update on each bucket to improve
  # numerical stability for operations like product and sum.
  assert reducer is not None
  out = jnp.full(
      (num_buckets, num_segments) + data.shape[1:], identity, dtype=dtype)
  out = _scatter_update(
    out, np.index_exp[lax.div(jnp.arange(segment_ids.shape[0]), bucket_size),
                      segment_ids[None, :]],
    data, scatter_op, indices_are_sorted,
    unique_indices, normalize_indices=False, mode=mode)
  return reducer(out, axis=0).astype(dtype)


def segment_sum(data: Array,
                segment_ids: Array,
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """Computes the sum within segments of an array.

  Similar to TensorFlow's `segment_sum
  <https://www.tensorflow.org/api_docs/python/tf/math/segment_sum>`_

  Args:
    data: an array with the values to be summed.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be summed. Values can be repeated and
      need not be sorted.
    num_segments: optional, an int with nonnegative value indicating the number
      of segments. The default is set to be the minimum number of segments that
      would support all indices in ``segment_ids``, calculated as
      ``max(segment_ids) + 1``.
      Since `num_segments` determines the size of the output, a static value
      must be provided to use ``segment_sum`` in a ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether `segment_ids` is known to be free of duplicates.
    bucket_size: size of bucket to group indices into. ``segment_sum`` is
      performed on each bucket separately to improve numerical stability of
      addition. Default ``None`` means no bucketing.
    mode: a :class:`jax.lax.GatherScatterMode` value describing how
      out-of-bounds indices should be handled. By default, values outside of the
      range [0, num_segments) are dropped and do not contribute to the sum.

  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.

  Examples:
    Simple 1D segment sum:

    >>> data = jnp.arange(5)
    >>> segment_ids = jnp.array([0, 0, 1, 1, 2])
    >>> segment_sum(data, segment_ids)
    DeviceArray([1, 5, 4], dtype=int32)

    Using JIT requires static `num_segments`:

    >>> from jax import jit
    >>> jit(segment_sum, static_argnums=2)(data, segment_ids, 3)
    DeviceArray([1, 5, 4], dtype=int32)
  """
  return _segment_update(
      "segment_sum",
      data,
      segment_ids,
      lax.scatter_add,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      bucket_size=bucket_size,
      reducer=jnp.sum,
      mode=mode)


def segment_prod(data: Array,
                 segment_ids: Array,
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False,
                 bucket_size: Optional[int] = None,
                 mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """Computes the product within segments of an array.

  Similar to TensorFlow's `segment_prod
  <https://www.tensorflow.org/api_docs/python/tf/math/segment_prod>`_

  Args:
    data: an array with the values to be reduced.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be reduced. Values can be repeated and
      need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with nonnegative value indicating the number
      of segments. The default is set to be the minimum number of segments that
      would support all indices in ``segment_ids``, calculated as
      ``max(segment_ids) + 1``.
      Since `num_segments` determines the size of the output, a static value
      must be provided to use ``segment_prod`` in a ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether `segment_ids` is known to be free of duplicates.
    bucket_size: size of bucket to group indices into. ``segment_prod`` is
      performed on each bucket separately to improve numerical stability of
      addition. Default ``None`` means no bucketing.
    mode: a :class:`jax.lax.GatherScatterMode` value describing how
      out-of-bounds indices should be handled. By default, values outside of the
      range [0, num_segments) are dropped and do not contribute to the sum.

  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment products.

  Examples:
    Simple 1D segment product:

    >>> data = jnp.arange(6)
    >>> segment_ids = jnp.array([0, 0, 1, 1, 2, 2])
    >>> segment_prod(data, segment_ids)
    DeviceArray([ 0,  6, 20], dtype=int32)

    Using JIT requires static `num_segments`:

    >>> from jax import jit
    >>> jit(segment_prod, static_argnums=2)(data, segment_ids, 3)
    DeviceArray([ 0,  6, 20], dtype=int32)
  """
  return _segment_update(
      "segment_prod",
      data,
      segment_ids,
      lax.scatter_mul,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      bucket_size=bucket_size,
      reducer=jnp.prod,
      mode=mode)


def segment_max(data: Array,
                segment_ids: Array,
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """Computes the maximum within segments of an array.

  Similar to TensorFlow's `segment_max
  <https://www.tensorflow.org/api_docs/python/tf/math/segment_max>`_

  Args:
    data: an array with the values to be reduced.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be reduced. Values can be repeated and
      need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with nonnegative value indicating the number
      of segments. The default is set to be the minimum number of segments that
      would support all indices in ``segment_ids``, calculated as
      ``max(segment_ids) + 1``.
      Since `num_segments` determines the size of the output, a static value
      must be provided to use ``segment_max`` in a ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether `segment_ids` is known to be free of duplicates.
    bucket_size: size of bucket to group indices into. ``segment_max`` is
      performed on each bucket separately. Default ``None`` means no bucketing.
    mode: a :class:`jax.lax.GatherScatterMode` value describing how
      out-of-bounds indices should be handled. By default, values outside of the
      range [0, num_segments) are dropped and do not contribute to the sum.

  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment maximums.

  Examples:
    Simple 1D segment max:

    >>> data = jnp.arange(6)
    >>> segment_ids = jnp.array([0, 0, 1, 1, 2, 2])
    >>> segment_max(data, segment_ids)
    DeviceArray([1, 3, 5], dtype=int32)

    Using JIT requires static `num_segments`:

    >>> from jax import jit
    >>> jit(segment_max, static_argnums=2)(data, segment_ids, 3)
    DeviceArray([1, 3, 5], dtype=int32)
  """
  return _segment_update(
      "segment_max",
      data,
      segment_ids,
      lax.scatter_max,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      bucket_size=bucket_size,
      reducer=jnp.max,
      mode=mode)


def segment_min(data: Array,
                segment_ids: Array,
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """Computes the minimum within segments of an array.

  Similar to TensorFlow's `segment_min
  <https://www.tensorflow.org/api_docs/python/tf/math/segment_min>`_

  Args:
    data: an array with the values to be reduced.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be reduced. Values can be repeated and
      need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with nonnegative value indicating the number
      of segments. The default is set to be the minimum number of segments that
      would support all indices in ``segment_ids``, calculated as
      ``max(segment_ids) + 1``.
      Since `num_segments` determines the size of the output, a static value
      must be provided to use ``segment_min`` in a ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether `segment_ids` is known to be free of duplicates.
    bucket_size: size of bucket to group indices into. ``segment_min`` is
      performed on each bucket separately. Default ``None`` means no bucketing.
    mode: a :class:`jax.lax.GatherScatterMode` value describing how
      out-of-bounds indices should be handled. By default, values outside of the
      range [0, num_segments) are dropped and do not contribute to the sum.

  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment minimums.

  Examples:
    Simple 1D segment min:

    >>> data = jnp.arange(6)
    >>> segment_ids = jnp.array([0, 0, 1, 1, 2, 2])
    >>> segment_min(data, segment_ids)
    DeviceArray([0, 2, 4], dtype=int32)

    Using JIT requires static `num_segments`:

    >>> from jax import jit
    >>> jit(segment_min, static_argnums=2)(data, segment_ids, 3)
    DeviceArray([0, 2, 4], dtype=int32)
  """
  return _segment_update(
      "segment_min",
      data,
      segment_ids,
      lax.scatter_min,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      bucket_size=bucket_size,
      reducer=jnp.min,
      mode=mode)


def segment_reduce(data: Array,
                   segment_ids: Array,
                   init_value: Array,
                   computation: Callable[[Array, Array], Array],
                   num_segments: Optional[int] = None,
                   indices_are_sorted: bool = False,
                   unique_indices: bool = False,
                   bucket_size: Optional[int] = None,
                   reducer: Optional[Callable] = None,
                   mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """Computes a custom reduction within segments of an array.

  ``init_value`` and ``computation`` together must form a `monoid
  <https://en.wikipedia.org/wiki/Monoid>`_
  for correctness. That is ``init_value`` must be an identity of
  ``computation``, and ``computation`` must be associative. If either is
  violated the result is undefined.

  Args:
    data: an array with the values to be reduced.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be reduced. Values can be repeated and
      need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    init_value: identity value for `computation`, i.e. default value for
      zero-sized segments.
    computation: reduction computation.
    num_segments: optional, an int with nonnegative value indicating the number
      of segments. The default is set to be the minimum number of segments that
      would support all indices in ``segment_ids``, calculated as
      ``max(segment_ids) + 1``. Since `num_segments` determines the size of the
      output, a static value must be provided to use ``segment_min`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether `segment_ids` is known to be free of duplicates.
    bucket_size: size of bucket to group indices into. ``segment_min`` is
      performed on each bucket separately. Default ``None`` means no bucketing.
      If not None, `reducer` must be specified.
    reducer: reducer function for combining bucketing outputs, e.g.
      ``jax.nn.logsumexp``. Should take as input the output and an ``axis``
      keyword argument.
    mode: a :class:`jax.lax.GatherScatterMode` value describing how
      out-of-bounds indices should be handled. By default, values outside of the
      range [0, num_segments) are dropped and do not contribute to the sum.

  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing
    the segment minimums.

  Examples:
    Simple 1D segment logsumexp:

    >>> data = jnp.array([-jnp.inf, 1, 2])
    >>> segment_ids = jnp.array([0, 0, 2])
    >>> segment_reduce(data, segment_ids, -jnp.inf, jnp.logaddexp)
    DeviceArray([  1., -inf,   2.], dtype=float32)


    Using JIT requires static `computation` and `num_segments`:

    >>> from jax import jit
    >>> jit(segment_reduce, static_argnums=(3, 4))(data, segment_ids, -jnp.inf, jnp.logaddexp, 3)
    DeviceArray([  1., -inf,   2.], dtype=float32)
  """

  def scatter_op(operand, scatter_indices, updates, dimension_numbers, *,
                 indices_are_sorted, unique_indices, mode):
    return lax.scatter_reduce(
        operand,
        scatter_indices,
        updates,
        computation,
        dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode)

  return _segment_update(
      "segment_reduce",
      data,
      segment_ids,
      scatter_op,
      identity=init_value,
      num_segments=num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices,
      bucket_size=bucket_size,
      reducer=reducer,
      mode=mode)
