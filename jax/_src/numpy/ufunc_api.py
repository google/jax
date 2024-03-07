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

"""Tools to create numpy-style ufuncs."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
import operator
from typing import Any

import jax
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.lax import lax as lax_internal
from jax._src.numpy import reductions
from jax._src.numpy.lax_numpy import _eliminate_deprecated_list_indexing, append, take
from jax._src.numpy.reductions import _moveaxis
from jax._src.numpy.util import implements, check_arraylike, _broadcast_to, _where
from jax._src.numpy.vectorize import vectorize
from jax._src.util import canonicalize_axis, set_module
from jax._src import pjit
import numpy as np


_AT_INPLACE_WARNING = """\
Because JAX arrays are immutable, jnp.ufunc.at() cannot operate inplace like
np.ufunc.at(). Instead, you can pass inplace=False and capture the result; e.g.
>>> arr = jnp.add.at(arr, ind, val, inplace=False)
"""


def get_if_single_primitive(fun: Callable[..., Any], *args: Any) -> jax.core.Primitive | None:
  """
  If fun(*args) lowers to a single primitive with inputs and outputs matching
  function inputs and outputs, return that primitive. Otherwise return None.
  """
  try:
    jaxpr = jax.make_jaxpr(fun)(*args)
  except:
    return None
  while len(jaxpr.eqns) == 1:
    eqn = jaxpr.eqns[0]
    if (eqn.invars, eqn.outvars) != (jaxpr.jaxpr.invars, jaxpr.jaxpr.outvars):
      return None
    elif (eqn.primitive == pjit.pjit_p and
          all(pjit.is_unspecified(sharding) for sharding in
              (*eqn.params['in_shardings'], *eqn.params['out_shardings']))):
      jaxpr = jaxpr.eqns[0].params['jaxpr']
    else:
      return jaxpr.eqns[0].primitive
  return None


_primitive_reducers: dict[jax.core.Primitive, Callable[..., Any]] = {
  lax_internal.add_p: reductions.sum,
  lax_internal.mul_p: reductions.prod,
}


_primitive_accumulators: dict[jax.core.Primitive, Callable[..., Any]] = {
  lax_internal.add_p: reductions.cumsum,
  lax_internal.mul_p: reductions.cumprod,
}


@set_module('jax.numpy')
class ufunc:
  """Functions that operate element-by-element on whole arrays.

  This is a class for LAX-backed implementations of numpy ufuncs.
  """
  def __init__(self, func: Callable[..., Any], /,
               nin: int, nout: int, *,
               name: str | None = None,
               nargs: int | None = None,
               identity: Any = None, update_doc=False):
    # We want ufunc instances to work properly when marked as static,
    # and for this reason it's important that their properties not be
    # mutated. We prevent this by storing them in a dunder attribute,
    # and accessing them via read-only properties.
    if update_doc:
      self.__doc__ = func.__doc__
    self.__name__ = name or func.__name__
    self.__static_props = {
      'func': func,
      'call': vectorize(func),
      'nin': operator.index(nin),
      'nout': operator.index(nout),
      'nargs': operator.index(nargs or nin),
      'identity': identity
    }

  _func = property(lambda self: self.__static_props['func'])
  _call = property(lambda self: self.__static_props['call'])
  nin = property(lambda self: self.__static_props['nin'])
  nout = property(lambda self: self.__static_props['nout'])
  nargs = property(lambda self: self.__static_props['nargs'])
  identity = property(lambda self: self.__static_props['identity'])

  def __hash__(self) -> int:
    # Do not include _call, because it is computed from _func.
    return hash((self._func, self.__name__, self.identity,
                 self.nin, self.nout, self.nargs))

  def __eq__(self, other: Any) -> bool:
    # Do not include _call, because it is computed from _func.
    return isinstance(other, ufunc) and (
      (self._func, self.__name__, self.identity, self.nin, self.nout, self.nargs) ==
      (other._func, other.__name__, other.identity, other.nin, other.nout, other.nargs))

  def __repr__(self) -> str:
    return f"<jnp.ufunc '{self.__name__}'>"

  def __call__(self, *args: ArrayLike,
               out: None = None, where: None = None,
               **kwargs: Any) -> Any:
    if out is not None:
      raise NotImplementedError(f"out argument of {self}")
    if where is not None:
      raise NotImplementedError(f"where argument of {self}")
    return self._call(*args, **kwargs)

  @implements(np.ufunc.reduce, module="numpy.ufunc")
  @partial(jax.jit, static_argnames=['self', 'axis', 'dtype', 'out', 'keepdims'])
  def reduce(self, a: ArrayLike, axis: int = 0, dtype: DTypeLike | None = None,
             out: None = None, keepdims: bool = False, initial: ArrayLike | None = None,
             where: ArrayLike | None = None) -> Array:
    check_arraylike(f"{self.__name__}.reduce", a)
    if self.nin != 2:
      raise ValueError("reduce only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("reduce only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.reduce()")
    if initial is not None:
      check_arraylike(f"{self.__name__}.reduce", initial)
    if where is not None:
      check_arraylike(f"{self.__name__}.reduce", where)
      if self.identity is None and initial is None:
        raise ValueError(f"reduction operation {self.__name__!r} does not have an identity, "
                         "so to use a where mask one has to specify 'initial'.")
      if lax_internal._dtype(where) != bool:
        raise ValueError(f"where argument must have dtype=bool; got dtype={lax_internal._dtype(where)}")
    primitive = get_if_single_primitive(self._call, *(self.nin * [lax_internal._one(a)]))
    if primitive is None:
      reducer = self._reduce_via_scan
    else:
      reducer = _primitive_reducers.get(primitive, self._reduce_via_scan)
    return reducer(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)

  def _reduce_via_scan(self, arr: ArrayLike, axis: int = 0, dtype: DTypeLike | None = None,
                       keepdims: bool = False, initial: ArrayLike | None = None,
                       where: ArrayLike | None = None) -> Array:
    assert self.nin == 2 and self.nout == 1
    arr = lax_internal.asarray(arr)
    if initial is None:
      initial = self.identity
    if dtype is None:
      dtype = jax.eval_shape(self._func, lax_internal._one(arr), lax_internal._one(arr)).dtype
    if where is not None:
      where = _broadcast_to(where, arr.shape)
    if isinstance(axis, tuple):
      axis = tuple(canonicalize_axis(a, arr.ndim) for a in axis)
      raise NotImplementedError("tuple of axes")
    elif axis is None:
      if keepdims:
        final_shape = (1,) * arr.ndim
      else:
        final_shape = ()
      arr = arr.ravel()
      if where is not None:
        where = where.ravel()
      axis = 0
    else:
      axis = canonicalize_axis(axis, arr.ndim)
      if keepdims:
        final_shape = (*arr.shape[:axis], 1, *arr.shape[axis + 1:])
      else:
        final_shape = (*arr.shape[:axis], *arr.shape[axis + 1:])

    # TODO: handle without transpose?
    if axis != 0:
      arr = _moveaxis(arr, axis, 0)
      if where is not None:
        where = _moveaxis(where, axis, 0)

    if initial is None and arr.shape[0] == 0:
      raise ValueError("zero-size array to reduction operation {self.__name__} which has no ideneity")

    def body_fun(i, val):
      if where is None:
        return self._call(val, arr[i].astype(dtype))
      else:
        return _where(where[i], self._call(val, arr[i].astype(dtype)), val)

    start_value: ArrayLike
    if initial is None:
      start_index = 1
      start_value = arr[0]
    else:
      start_index = 0
      start_value = initial
    start_value = _broadcast_to(lax_internal.asarray(start_value).astype(dtype), arr.shape[1:])

    result = jax.lax.fori_loop(start_index, arr.shape[0], body_fun, start_value)

    if keepdims:
      result = result.reshape(final_shape)
    return result

  @implements(np.ufunc.accumulate, module="numpy.ufunc")
  @partial(jax.jit, static_argnames=['self', 'axis', 'dtype'])
  def accumulate(self, a: ArrayLike, axis: int = 0, dtype: DTypeLike | None = None,
                 out: None = None) -> Array:
    if self.nin != 2:
      raise ValueError("accumulate only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("accumulate only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.accumulate()")
    primitive = get_if_single_primitive(self._call, *(self.nin * [lax_internal._one(a)]))
    if primitive is None:
      accumulator = self._accumulate_via_scan
    else:
      accumulator = _primitive_accumulators.get(primitive, self._accumulate_via_scan)
    return accumulator(a, axis=axis, dtype=dtype)

  def _accumulate_via_scan(self, arr: ArrayLike, axis: int = 0,
                           dtype: DTypeLike | None = None) -> Array:
    assert self.nin == 2 and self.nout == 1
    check_arraylike(f"{self.__name__}.accumulate", arr)
    arr = lax_internal.asarray(arr)

    if dtype is None:
      dtype = jax.eval_shape(self._func, lax_internal._one(arr), lax_internal._one(arr)).dtype

    if axis is None or isinstance(axis, tuple):
      raise ValueError("accumulate does not allow multiple axes")
    axis = canonicalize_axis(axis, np.ndim(arr))

    arr = _moveaxis(arr, axis, 0)
    def scan_fun(carry, _):
      i, x = carry
      y = _where(i == 0, arr[0].astype(dtype), self._call(x.astype(dtype), arr[i].astype(dtype)))
      return (i + 1, y), y
    _, result = jax.lax.scan(scan_fun, (0, arr[0].astype(dtype)), None, length=arr.shape[0])
    return _moveaxis(result, 0, axis)

  @implements(np.ufunc.at, module="numpy.ufunc")
  @partial(jax.jit, static_argnums=[0], static_argnames=['inplace'])
  def at(self, a: ArrayLike, indices: Any, b: ArrayLike | None = None, /, *,
         inplace: bool = True) -> Array:
    if inplace:
      raise NotImplementedError(_AT_INPLACE_WARNING)
    if b is None:
      return self._at_via_scan(a, indices)
    else:
      return self._at_via_scan(a, indices, b)

  def _at_via_scan(self, a: ArrayLike, indices: Any, *args: Any) -> Array:
    assert len(args) in {0, 1}
    check_arraylike(f"{self.__name__}.at", a, *args)
    dtype = jax.eval_shape(self._func, lax_internal._one(a), *(lax_internal._one(arg) for arg in args)).dtype
    a = lax_internal.asarray(a).astype(dtype)
    args = tuple(lax_internal.asarray(arg).astype(dtype) for arg in args)
    indices = _eliminate_deprecated_list_indexing(indices)
    if not indices:
      return a

    shapes = [np.shape(i) for i in indices if not isinstance(i, slice)]
    shape = shapes and jax.lax.broadcast_shapes(*shapes)
    if not shape:
      return a.at[indices].set(self._call(a.at[indices].get(), *args))

    if args:
      arg = _broadcast_to(args[0], (*shape, *args[0].shape[len(shape):]))
      args = (arg.reshape(math.prod(shape), *args[0].shape[len(shape):]),)
    indices = [idx if isinstance(idx, slice) else _broadcast_to(idx, shape).ravel() for idx in indices]

    def scan_fun(carry, x):
      i, a = carry
      idx = tuple(ind if isinstance(ind, slice) else ind[i] for ind in indices)
      a = a.at[idx].set(self._call(a.at[idx].get(), *(arg[i] for arg in args)))
      return (i + 1, a), x
    carry, _ = jax.lax.scan(scan_fun, (0, a), None, len(indices[0]))
    return carry[1]

  @implements(np.ufunc.reduceat, module="numpy.ufunc")
  @partial(jax.jit, static_argnames=['self', 'axis', 'dtype'])
  def reduceat(self, a: ArrayLike, indices: Any, axis: int = 0,
               dtype: DTypeLike | None = None, out: None = None) -> Array:
    if self.nin != 2:
      raise ValueError("reduceat only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("reduceat only supported for functions returning a single value")
    if out is not None:
      raise NotImplementedError(f"out argument of {self.__name__}.reduceat()")
    return self._reduceat_via_scan(a, indices, axis=axis, dtype=dtype)

  def _reduceat_via_scan(self, a: ArrayLike, indices: Any, axis: int = 0,
                         dtype: DTypeLike | None = None) -> Array:
    check_arraylike(f"{self.__name__}.reduceat", a, indices)
    a = lax_internal.asarray(a)
    idx_tuple = _eliminate_deprecated_list_indexing(indices)
    assert len(idx_tuple) == 1
    indices = idx_tuple[0]
    if a.ndim == 0:
      raise ValueError(f"reduceat: a must have 1 or more dimension, got {a.shape=}")
    if indices.ndim != 1:
      raise ValueError(f"reduceat: indices must be one-dimensional, got {indices.shape=}")
    if dtype is None:
      dtype = a.dtype
    if axis is None or isinstance(axis, (tuple, list)):
      raise ValueError("reduceat requires a single integer axis.")
    axis = canonicalize_axis(axis, a.ndim)
    out = take(a, indices, axis=axis)
    ind = jax.lax.expand_dims(append(indices, a.shape[axis]),
                              list(np.delete(np.arange(out.ndim), axis)))
    ind_start = jax.lax.slice_in_dim(ind, 0, ind.shape[axis] - 1, axis=axis)
    ind_end = jax.lax.slice_in_dim(ind, 1, ind.shape[axis], axis=axis)
    def loop_body(i, out):
      return _where((i > ind_start) & (i < ind_end),
                    self._call(out, take(a, jax.lax.expand_dims(i, (0,)), axis=axis)),
                    out)
    return jax.lax.fori_loop(0, a.shape[axis], loop_body, out)

  @implements(np.ufunc.outer, module="numpy.ufunc")
  @partial(jax.jit, static_argnums=[0])
  def outer(self, A: ArrayLike, B: ArrayLike, /, **kwargs) -> Array:
    if self.nin != 2:
      raise ValueError("outer only supported for binary ufuncs")
    if self.nout != 1:
      raise ValueError("outer only supported for functions returning a single value")
    check_arraylike(f"{self.__name__}.outer", A, B)
    _ravel = lambda A: jax.lax.reshape(A, (np.size(A),))
    result = jax.vmap(jax.vmap(partial(self._call, **kwargs), (None, 0)), (0, None))(_ravel(A), _ravel(B))
    return result.reshape(*np.shape(A), *np.shape(B))


def frompyfunc(func: Callable[..., Any], /, nin: int, nout: int,
               *, identity: Any = None) -> ufunc:
  """Create a JAX ufunc from an arbitrary JAX-compatible scalar function.

  Args:
    func : a callable that takes `nin` scalar arguments and returns `nout` outputs.
    nin: integer specifying the number of scalar inputs
    nout: integer specifying the number of scalar outputs
    identity: (optional) a scalar specifying the identity of the operation, if any.

  Returns:
    wrapped : jax.numpy.ufunc wrapper of func.
  """
  return ufunc(func, nin, nout, identity=identity, update_doc=True)
