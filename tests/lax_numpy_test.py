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

from array import array as make_python_array
import collections
from collections.abc import Iterator
import copy
from functools import partial
import inspect
import io
import itertools
import math
import platform
from typing import cast
import unittest
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
try:
  import numpy_dispatch
except ImportError:
  numpy_dispatch = None

import jax
from jax import lax
from jax import numpy as jnp
from jax.test_util import check_grads

from jax._src import array
from jax._src.sharding_impls import make_single_device_sharding
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.lax import lax as lax_internal
from jax._src.numpy import ufuncs as jnp_ufuncs
from jax._src.util import safe_zip, tuple_update

config.parse_flags_with_absl()

numpy_version = jtu.numpy_version()

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
one_dim_array_shapes = [(1,), (6,), (12,)]
empty_array_shapes = [(0,), (0, 4), (3, 0),]
broadcast_compatible_shapes = [(), (1,), (3,), (1, 3), (4, 1), (4, 3)]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE, jtu.PYTHON_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes + empty_array_shapes
nonzerodim_shapes = nonempty_nonscalar_array_shapes + empty_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes = scalar_shapes + array_shapes

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
unsigned_dtypes = jtu.dtypes.all_unsigned
bool_dtypes = jtu.dtypes.boolean
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes + unsigned_dtypes
all_dtypes = number_dtypes + bool_dtypes

NO_VALUE = object()

python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

# uint64 is problematic because with any uint type it promotes to float:
int_dtypes_no_uint64 = [d for d in int_dtypes + unsigned_dtypes if d != np.uint64]

def _bitcast_uint4_to_uint8(operand):
  # Note: assumes little-endian byte order.
  assert operand.dtype == 'uint4'
  operand = operand.astype('uint8')
  return operand[..., ::2] + (operand[..., 1::2] << 4)

def _bitcast_uint8_to_uint4(operand):
  # Note: assumes little-endian byte order.
  assert operand.dtype == 'uint8'
  result = np.zeros((*operand.shape[:-1], operand.shape[-1] * 2), dtype='uint4')
  result[..., ::2] = (operand & 0b00001111).astype('uint4')
  result[..., 1::2] = ((operand & 0b11110000) >> 4).astype('uint4')
  return result

def np_view(arr: np.ndarray, dtype) -> np.ndarray:
  # Implementation of np.ndarray.view() that works for int4/uint4
  if dtype is None:
    return arr
  dtype = np.dtype(dtype)
  nbits_in = dtypes.itemsize_bits(arr.dtype)
  nbits_out = dtypes.itemsize_bits(dtype)
  if nbits_in == 4:
    arr = _bitcast_uint4_to_uint8(arr.view('uint4'))
  if nbits_out == 4:
    arr = _bitcast_uint8_to_uint4(arr.view('uint8'))
  return arr.view(dtype)


def _indexer_with_default_outputs(indexer, use_defaults=True):
  """Like jtu.with_jax_dtype_defaults, but for __getitem__ APIs"""
  class Indexer:
    @partial(jtu.with_jax_dtype_defaults, use_defaults=use_defaults)
    def __getitem__(self, *args):
      return indexer.__getitem__(*args)
  return Indexer()

def _valid_dtypes_for_shape(shape, dtypes):
  # Not all (shape, dtype) pairs are valid. In particular, Python scalars only
  # have one type in each category (float, bool, etc.)
  if shape is jtu.PYTHON_SCALAR_SHAPE:
    return [t for t in dtypes if t in python_scalar_dtypes]
  return dtypes

def _shape_and_dtypes(shapes, dtypes):
  for shape in shapes:
    for dtype in _valid_dtypes_for_shape(shape, dtypes):
      yield (shape, dtype)

def _compatible_shapes(shape):
  if np.ndim(shape) == 0 or shape in scalar_shapes:
    return [shape]
  return (shape[n:] for n in range(len(shape) + 1))

OpRecord = collections.namedtuple(
  "OpRecord",
  ["name", "nargs", "dtypes", "shapes", "rng_factory", "diff_modes",
   "test_name", "check_dtypes", "tolerance", "inexact", "kwargs"])

def op_record(name, nargs, dtypes, shapes, rng_factory, diff_modes,
              test_name=None, check_dtypes=True,
              tolerance=None, inexact=False, kwargs=None):
  test_name = test_name or name
  return OpRecord(name, nargs, dtypes, shapes, rng_factory, diff_modes,
                  test_name, check_dtypes, tolerance, inexact, kwargs)


JAX_ARGMINMAX_RECORDS = [
    op_record("argmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("argmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("nanargmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanargmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
]

def _shapes_are_broadcast_compatible(shapes):
  try:
    lax.broadcast_shapes(*(() if s in scalar_shapes else s for s in shapes))
  except ValueError:
    return False
  else:
    return True

def _shapes_are_equal_length(shapes):
  return all(len(shape) == len(shapes[0]) for shape in shapes[1:])


def ref_top_k(arr, k, /, *, axis, mode, sorted):
  """Reference implementation of numpy.top_k, added in NumPy v2.6.0"""
  # TODO(jakevdp): remove this helper when minimum required NumPy is version 2.6.0.
  if jtu.numpy_version() >= (2, 6, 0):
    return np.top_k(arr, k, axis=axis, mode=mode, sorted=sorted)
  if mode == 'largest':
    sorted_indices = np.argsort(
        -arr.astype(np.float64) if np.issubdtype(arr.dtype, np.floating)
        else ~arr if arr.dtype == np.bool_
        else -arr.astype(np.int64) if not np.issubdtype(arr.dtype, np.unsignedinteger)
        else -(arr.astype(np.int64) + 1),
        axis=axis, kind='stable')
  else:
    sorted_indices = np.argsort(arr, axis=axis, kind='stable')
  slc = [slice(None)] * arr.ndim
  slc[axis] = slice(0, k)
  indices = sorted_indices[tuple(slc)]
  values = np.take_along_axis(arr, indices, axis=axis)
  return values, indices


class LaxBackedNumpyTests(jtu.JaxTestCase):
  """Tests for LAX-backed Numpy implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
    def f():
      out = [rng(shape, dtype or jnp.float_)
             for shape, dtype in zip(shapes, dtypes)]
      if np_arrays:
        return out
      return [jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
              for a in out]
    return f

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in list(range(-len(shape), len(shape)))],
    dtype=all_dtypes,
  )
  def testUnstack(self, shape, axis, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    if jnp.asarray(x).ndim == 0:
      with self.assertRaisesRegex(ValueError, "Unstack requires arrays with"):
        jnp.unstack(x, axis=axis)
      return
    y = jnp.unstack(x, axis=axis)
    if shape[axis] == 0:
      self.assertEqual(y, ())
    else:
      self.assertArraysEqual(jnp.moveaxis(jnp.array(y), 0, axis), x)

  @parameterized.parameters(
      [dtype for dtype in [
          jnp.bool,
          jnp.uint4, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64,
          jnp.int4, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
          jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64,
          jnp.complex64, jnp.complex128]
       if dtype == dtypes.canonicalize_dtype(dtype)])
  def testDtypeWrappers(self, dtype):
    arr = dtype(0)
    self.assertIsInstance(arr, jax.Array)
    self.assertEqual(arr.dtype, np.dtype(dtype))
    self.assertArraysEqual(arr, 0, check_dtypes=False)

    # No copy primitive is generated
    jaxpr = jax.make_jaxpr(dtype)(0)
    prims = [eqn.primitive for eqn in jaxpr.eqns]
    self.assertEqual(prims, [lax.convert_element_type_p])  # No copy generated.

  def testBoolDtypeAlias(self):
    self.assertIs(jnp.bool, jnp.bool_)

  @jtu.sample_product(
      dtype=float_dtypes + [object],
      allow_pickle=[True, False],
  )
  def testLoad(self, dtype, allow_pickle):
    if dtype == object and not allow_pickle:
      self.skipTest("dtype=object requires allow_pickle=True")
    rng = jtu.rand_default(self.rng())
    arr = rng((10), dtype)
    with io.BytesIO() as f:
      jnp.save(f, arr)
      f.seek(0)
      arr_out = jnp.load(f, allow_pickle=allow_pickle)
    self.assertArraysEqual(arr, arr_out, allow_object_dtype=True)

  def testArrayEqualExamples(self):
    # examples from the array_equal() docstring.
    self.assertTrue(jnp.array_equal([1, 2], [1, 2]))
    self.assertTrue(jnp.array_equal(np.array([1, 2]), np.array([1, 2])))
    self.assertFalse(jnp.array_equal([1, 2], [1, 2, 3]))
    self.assertFalse(jnp.array_equal([1, 2], [1, 4]))

    a = np.array([1, np.nan])
    self.assertFalse(jnp.array_equal(a, a))
    self.assertTrue(jnp.array_equal(a, a, equal_nan=True))

    a = np.array([1 + 1j])
    b = a.copy()
    a.real = np.nan
    b.imag = np.nan
    self.assertTrue(jnp.array_equal(a, b, equal_nan=True))

  def testArrayEquivExamples(self):
    # examples from the array_equiv() docstring.
    self.assertTrue(jnp.array_equiv([1, 2], [1, 2]))
    self.assertFalse(jnp.array_equiv([1, 2], [1, 3]))
    with jax.numpy_rank_promotion('allow'):
      self.assertTrue(jnp.array_equiv([1, 2], [[1, 2], [1, 2]]))
      self.assertFalse(jnp.array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]]))
      self.assertFalse(jnp.array_equiv([1, 2], [[1, 2], [1, 3]]))

  def testArrayModule(self):
    if numpy_dispatch is None:
      raise SkipTest('requires https://github.com/seberg/numpy-dispatch')

    jnp_array = jnp.array(1.0)
    np_array = np.array(1.0)

    module = numpy_dispatch.get_array_module(jnp_array)
    self.assertIs(module, jnp)

    module = numpy_dispatch.get_array_module(jnp_array, np_array)
    self.assertIs(module, jnp)

    def f(x):
      module = numpy_dispatch.get_array_module(x)
      self.assertIs(module, jnp)
      return x
    jax.jit(f)(jnp_array)
    jax.grad(f)(jnp_array)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in list(range(-len(shape), len(shape)))],
    discont=[None, "pi", 2],
    period=["2pi", "pi", 2, 4],
    dtype=default_dtypes,
  )
  def testUnwrap(self, shape, dtype, axis, discont, period):
    special_vals = {"pi": np.pi, "2pi": 2 * np.pi}
    period = special_vals.get(period, period)
    discont = special_vals.get(discont, discont)

    rng = jtu.rand_default(self.rng())

    def np_fun(x):
      dtype = None
      if x.dtype == dtypes.bfloat16:
        dtype = x.dtype
        x = x.astype(np.float32)
      out = np.unwrap(x, axis=axis, discont=discont, period=period)
      return out if dtype is None else out.astype(dtype)

    jnp_fun = partial(jnp.unwrap, axis=axis, discont=discont, period=period)
    if not dtypes.issubdtype(dtype, np.inexact):
      # This case requires implicit dtype promotion
      jnp_fun = jax.numpy_dtype_promotion('standard')(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True,
                            atol={dtypes.bfloat16: 1e-1, np.float16: 1e-2})
    self._CompileAndCheck(jnp_fun, args_maker, atol={dtypes.bfloat16: 1e-1})

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in list(range(-len(shape), len(shape))) + [None]],
    dtype=all_dtypes,
  )
  def testCountNonzero(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.count_nonzero(x, axis)
    jnp_fun = lambda x: jnp.count_nonzero(x, axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(shape=nonzerodim_shapes, dtype=all_dtypes)
  def testNonzero(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np.nonzero, jnp.nonzero, args_maker, check_dtypes=False)

  @jtu.sample_product(
    [dict(shape=shape, fill_value=fill_value)
      for shape in nonempty_nonscalar_array_shapes
      for fill_value in [None, -1, shape or (1,)]
     ],
    dtype=all_dtypes,
    size=[1, 5, 10],
  )
  def testNonzeroSize(self, shape, dtype, size, fill_value):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    def np_fun(x):
      result = np.nonzero(x)
      if size <= len(result[0]):
        return tuple(arg[:size] for arg in result)
      else:
        fillvals = fill_value if np.ndim(fill_value) else len(result) * [fill_value or 0]
        return tuple(np.concatenate([arg, np.full(size - len(arg), fval, arg.dtype)])
                     for fval, arg in safe_zip(fillvals, result))
    jnp_fun = lambda x: jnp.nonzero(x, size=size, fill_value=fill_value)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(shape=nonzerodim_shapes, dtype=all_dtypes)
  def testFlatNonzero(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = np.flatnonzero
    jnp_fun = jnp.flatnonzero
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

    # JIT compilation requires specifying the size statically:
    jnp_fun = lambda x: jnp.flatnonzero(x, size=np.size(x) // 2)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=nonempty_nonscalar_array_shapes,
    dtype=all_dtypes,
    fill_value=[None, -1, 10, (-1,), (10,)],
    size=[1, 5, 10],
  )
  def testFlatNonzeroSize(self, shape, dtype, size, fill_value):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    def np_fun(x):
      result = np.flatnonzero(x)
      if size <= len(result):
        return result[:size]
      else:
        fill_val = fill_value or 0
        return np.concatenate([result, np.full(size - len(result), fill_val, result.dtype)])
    jnp_fun = lambda x: jnp.flatnonzero(x, size=size, fill_value=fill_value)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(shape=nonzerodim_shapes, dtype=all_dtypes)
  def testArgWhere(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np.argwhere, jnp.argwhere, args_maker, check_dtypes=False)

    # JIT compilation requires specifying a size statically. Full test of this
    # behavior is in testNonzeroSize().
    jnp_fun = lambda x: jnp.argwhere(x, size=np.size(x) // 2)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, fill_value=fill_value)
      for shape in nonempty_nonscalar_array_shapes
      for fill_value in [None, -1, shape or (1,)]
     ],
    dtype=all_dtypes,
    size=[1, 5, 10],
  )
  def testArgWhereSize(self, shape, dtype, size, fill_value):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    def np_fun(x):
      result = np.argwhere(x)
      if size <= len(result):
        return result[:size]
      else:
        fillvals = fill_value if np.ndim(fill_value) else result.shape[-1] * [fill_value or 0]
        return np.empty((size, 0), dtype=int) if np.ndim(x) == 0 else np.stack([np.concatenate([arg, np.full(size - len(arg), fval, arg.dtype)])
                        for fval, arg in safe_zip(fillvals, result.T)]).T
    jnp_fun = lambda x: jnp.argwhere(x, size=size, fill_value=fill_value)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(np_op=getattr(np, rec.name), jnp_op=getattr(jnp, rec.name),
          shape=shape, dtype=dtype, axis=axis, rng_factory=rec.rng_factory)
      for rec in JAX_ARGMINMAX_RECORDS
      for shape, dtype in _shape_and_dtypes(rec.shapes, rec.dtypes)
      for axis in range(-len(shape), len(shape))],
    keepdims=[False, True],
  )
  def testArgMinMax(self, np_op, jnp_op, rng_factory, shape, dtype, axis, keepdims):
    rng = rng_factory(self.rng())
    if dtype == np.complex128 and jtu.test_device_matches(["gpu"]):
      raise unittest.SkipTest("complex128 reductions not supported on GPU")
    if "nan" in np_op.__name__ and dtype == jnp.bfloat16:
      raise unittest.SkipTest("NumPy doesn't correctly handle bfloat16 arrays")
    kwds = {"keepdims": True} if keepdims else {}

    np_fun = jtu.with_jax_dtype_defaults(partial(np_op, axis=axis, **kwds))
    jnp_fun = partial(jnp_op, axis=axis, **kwds)

    args_maker = lambda: [rng(shape, dtype)]
    try:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    except ValueError as e:
      if str(e) == "All-NaN slice encountered":
        self.skipTest("JAX doesn't support checking for all-NaN slices")
      else:
        raise
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(name=rec.name, np_op=getattr(np, rec.name),
          jnp_op=getattr(jnp, rec.name))
      for rec in JAX_ARGMINMAX_RECORDS],
  )
  def testArgMinMaxEmpty(self, name, np_op, jnp_op):
    name = name[3:] if name.startswith("nan") else name
    msg = f"attempt to get {name} of an empty sequence"
    with self.assertRaisesRegex(ValueError, msg):
      jnp_op(np.array([]))
    with self.assertRaisesRegex(ValueError, msg):
      jnp_op(np.zeros((2, 0)), axis=1)
    np_fun = jtu.with_jax_dtype_defaults(partial(np_op, axis=0))
    jnp_fun = partial(jnp_op, axis=0)
    args_maker = lambda: [np.zeros((2, 0))]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, axes=axes)
      for lhs_shape, rhs_shape, axes in [
          [(2,), (2,), (-1, -1, -1, None)], # scalar output
          [(2, 4), (2, 4), (-1, -1, -1, 0)], # 2D vectors
          [(3, 4), (3, 4), (-1, -1, -1, 0)], # 3D vectors
          [(3, 4), (3, 6, 5, 4), (-1, -1, -1, 0)], # broadcasting
          [(4, 3), (3, 6, 5, 4), (1, 0, -1, None)], # different axes
          [(6, 1, 3), (5, 3), (-1, -1, -1, None)], # more broadcasting
          [(6, 1, 2), (5, 3), (-1, -1, -1, None)], # mixed 2D and 3D vectors
          [(10, 5, 2, 8), (1, 5, 1, 3), (-2, -1, -3, None)], # axes/broadcasting
          [(4, 5, 2), (4, 5, 2), (-1, -1, 0, None)], # axisc should do nothing
          [(4, 5, 2), (4, 5, 2), (-1, -1, -1, None)] # same as before
      ]],
    lhs_dtype=number_dtypes,
    rhs_dtype=number_dtypes,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="Support for 2-dimensional vectors in jnp.cross is deprecated")
  def testCross(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    axisa, axisb, axisc, axis = axes
    jnp_fun = lambda a, b: jnp.cross(a, b, axisa, axisb, axisc, axis)

    def np_fun(a, b):
      effective_axisa = axisa if axis is None else axis
      effective_axisb = axisb if axis is None else axis
      effective_axisc = axisc if axis is None else axis

      a = a.astype(np.float32) if lhs_dtype == jnp.bfloat16 else a
      b = b.astype(np.float32) if rhs_dtype == jnp.bfloat16 else b

      a_m = np.moveaxis(a, effective_axisa, -1)
      b_m = np.moveaxis(b, effective_axisb, -1)

      if a_m.shape[-1] == 2 and b_m.shape[-1] == 2:
        out = a_m[..., 0] * b_m[..., 1] - a_m[..., 1] * b_m[..., 0]
      else:
        if a_m.shape[-1] == 2:
          a_m = np.pad(a_m, [(0, 0)] * (a_m.ndim - 1) + [(0, 1)])
        if b_m.shape[-1] == 2:
          b_m = np.pad(b_m, [(0, 0)] * (b_m.ndim - 1) + [(0, 1)])
        out = np.cross(a_m, b_m, axisc=effective_axisc)

      return out.astype(jnp.promote_types(lhs_dtype, rhs_dtype))

    tol_spec = {dtypes.bfloat16: 3e-1, np.float16: 0.15}
    tol = max(jtu.tolerance(lhs_dtype, tol_spec),
              jtu.tolerance(rhs_dtype, tol_spec))
    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol)

  def testCrossDeprecationWarning(self):
    with jtu.test_warning_util.record_warnings() as w:
      jnp.cross(jnp.ones(2), jnp.ones(2))
    self.assertTrue(any("Support for 2-dimensional vectors" in str(warn.message) for warn in w))

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [
          ((3, 3), ()),
          ((), (3, 3)),
          ((4, 5), (5,)),
          ((6,), (6, 4)),
          ((3, 4), (4, 5)),
          ((4, 3, 2), (2,)),
          ((2,), (3, 2, 4)),
          ((4, 3, 2), (2, 5)),
          ((5, 2), (3, 2, 4)),
          ((2, 3, 4), (5, 4, 1))]],
    lhs_dtype=number_dtypes,
    rhs_dtype=number_dtypes,
  )
  @jax.default_matmul_precision("float32")
  def testDot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {np.float16: 1e-2, jnp.bfloat16: 1e-1, np.float32: 2e-5,
           np.float64: 1e-14, np.complex128: 1e-14}
    def np_dot(x, y):
      x = x.astype(np.float32) if lhs_dtype == jnp.bfloat16 else x
      y = y.astype(np.float32) if rhs_dtype == jnp.bfloat16 else y
      return np.dot(x, y).astype(jnp.promote_types(lhs_dtype, rhs_dtype))
    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_dot, jnp.dot, args_maker, tol=tol)
      self._CompileAndCheck(jnp.dot, args_maker, atol=tol, rtol=tol)

  @jtu.sample_product(
      lhs_dtype=number_dtypes,
      rhs_dtype=number_dtypes,
  )
  @jax.numpy_dtype_promotion('standard')
  def testMixedPrecisionDot(self, lhs_dtype, rhs_dtype):
    # This test confirms that jnp.dot lowers to a single dot_general call,
    # avoiding explicit type casting of inputs and outputs.
    lhs = jax.ShapeDtypeStruct((5,), lhs_dtype)
    rhs = jax.ShapeDtypeStruct((5,), rhs_dtype)
    jaxpr = jax.make_jaxpr(jnp.dot)(lhs, rhs)
    prims = [eqn.primitive for eqn in jaxpr.eqns]
    self.assertIn(prims, [
      [lax.dot_general_p],
      [lax.dot_general_p, lax.convert_element_type_p]
    ])

  @jtu.sample_product(
    [dict(name=name, lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for name, lhs_shape, rhs_shape in [
          ("vector-vector", (3,), (3,)),
          ("matrix-vector", (3, 3), (3,)),
          ("vector-matrix", (3,), (3, 3)),
          ("matrix-matrix", (3, 3), (3, 3)),
          ("vector-tensor", (3,), (5, 3, 2)),
          ("tensor-vector", (5, 3, 2), (2,)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-matrix", (5, 2, 3), (3, 2)),
          ("tensor-tensor", (5, 3, 4), (5, 4, 1)),
          ("tensor-tensor-broadcast", (3, 1, 3, 4), (5, 4, 1)),
          ("tensor-tensor-both-1", (1, 5, 4), (1, 4, 3))]],
    lhs_dtype=number_dtypes,
    rhs_dtype=number_dtypes,
  )
  @jax.default_matmul_precision("float32")
  def testMatmul(self, name, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    def np_fun(x, y):
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.matmul(x, y).astype(dtype)
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12, jnp.bfloat16: 1e-1}

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, jnp.matmul, args_maker, tol=tol)
      self._CompileAndCheck(jnp.matmul, args_maker, atol=tol, rtol=tol)

  def testMatmulBothSize1BatchDimsSqueezedJaxpr(self):
    x = jax.ShapeDtypeStruct((1, 5, 4), jnp.float32)
    y = jax.ShapeDtypeStruct((1, 4, 3), jnp.float32)
    jaxpr = jax.make_jaxpr(jnp.matmul)(x, y)
    [dot_eqn] = (eqn for eqn in jaxpr.eqns if eqn.primitive == lax.dot_general_p)
    self.assertEqual(dot_eqn.invars[0].aval.shape, (5, 4))
    self.assertEqual(dot_eqn.invars[1].aval.shape, (4, 3))
    self.assertEqual(dot_eqn.params['dimension_numbers'], (((1,), (0,)), ((), ())))

  @jtu.sample_product(
      lhs_batch=broadcast_compatible_shapes,
      rhs_batch=broadcast_compatible_shapes,
      axis_size=[2, 4],
      axis=range(-2, 2),
      dtype=number_dtypes,
  )
  @jax.default_matmul_precision("float32")
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testVecdot(self, lhs_batch, rhs_batch, axis_size, axis, dtype):
    # Construct vecdot-compatible shapes.
    size = min(len(lhs_batch), len(rhs_batch))
    axis = int(np.clip(axis, -size - 1, size))
    if axis >= 0:
      lhs_shape = (*lhs_batch[:axis], axis_size, *lhs_batch[axis:])
      rhs_shape = (*rhs_batch[:axis], axis_size, *rhs_batch[axis:])
    else:
      laxis = axis + len(lhs_batch) + 1
      lhs_shape = (*lhs_batch[:laxis], axis_size, *lhs_batch[laxis:])
      raxis = axis + len(rhs_batch) + 1
      rhs_shape = (*rhs_batch[:raxis], axis_size, *rhs_batch[raxis:])

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    @jtu.promote_like_jnp
    def np_fn(x, y, axis=axis):
      return np.vecdot(x, y, axis=axis).astype(x.dtype)
    jnp_fn = partial(jnp.vecdot, axis=axis)
    tol = {np.float16: 1e-2, np.float32: 1E-3, np.float64: 1e-12,
           np.complex64: 1E-3, np.complex128: 1e-12, jnp.bfloat16: 1e-1}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  @jtu.sample_product(
      lhs_batch=broadcast_compatible_shapes,
      rhs_batch=broadcast_compatible_shapes,
      mat_size=[1, 2, 3],
      vec_size=[2, 3, 4],
      dtype=number_dtypes,
  )
  @jax.default_matmul_precision("float32")
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testMatvec(self, lhs_batch, rhs_batch, mat_size, vec_size, dtype):
    rng = jtu.rand_default(self.rng())
    lhs_shape = (*lhs_batch, mat_size, vec_size)
    rhs_shape = (*rhs_batch, vec_size)
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    jnp_fn = jnp.matvec
    @jtu.promote_like_jnp
    def np_fn(x, y):
      f = (np.vectorize(np.matmul, signature="(m,n),(n)->(m)")
           if jtu.numpy_version() < (2, 2, 0) else np.matvec)
      return f(x, y).astype(x.dtype)
    tol = {np.float16: 1e-2, np.float32: 1E-3, np.float64: 1e-12,
           np.complex64: 1E-3, np.complex128: 1e-12, jnp.bfloat16: 1e-1}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  @jtu.sample_product(
      lhs_batch=broadcast_compatible_shapes,
      rhs_batch=broadcast_compatible_shapes,
      mat_size=[1, 2, 3],
      vec_size=[2, 3, 4],
      dtype=number_dtypes,
  )
  @jax.default_matmul_precision("float32")
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testVecmat(self, lhs_batch, rhs_batch, mat_size, vec_size, dtype):
    rng = jtu.rand_default(self.rng())
    lhs_shape = (*lhs_batch, vec_size)
    rhs_shape = (*rhs_batch, vec_size, mat_size)
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    jnp_fn = jnp.vecmat
    @jtu.promote_like_jnp
    def np_fn(x, y):
      f = (np.vectorize(lambda x, y: np.matmul(np.conj(x), y),
                        signature="(m),(m,n)->(n)")
           if jtu.numpy_version() < (2, 2, 0) else np.vecmat)
      return f(x, y).astype(x.dtype)
    tol = {np.float16: 1e-2, np.float32: 1E-3, np.float64: 1e-12,
           np.complex64: 1E-3, np.complex128: 1e-12, jnp.bfloat16: 1e-1}
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fn, args_maker, tol=tol)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, axes=axes)
      for lhs_shape, rhs_shape, axes in [
          [(3,), (), 0],
          [(2, 3, 4), (5, 6, 7), 0],  # from issue #740
          [(2, 3, 4), (3, 4, 5, 6), 2],
          [(2, 3, 4), (5, 4, 3, 6), [1, 2]],
          [(2, 3, 4), (5, 4, 3, 6), [[1, 2], [2, 1]]],
          [(1, 2, 3, 4), (4, 5, 3, 6), [[2, 3], [2, 0]]],
      ]],
    lhs_dtype=number_dtypes,
    rhs_dtype=number_dtypes,
  )
  @jax.default_matmul_precision("float32")
  def testTensordot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    jnp_fun = lambda a, b: jnp.tensordot(a, b, axes)
    def np_fun(a, b):
      a = a if lhs_dtype != jnp.bfloat16 else a.astype(np.float32)
      b = b if rhs_dtype != jnp.bfloat16 else b.astype(np.float32)
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.tensordot(a, b, axes).astype(dtype)
    tol = {np.float16: 1e-1, np.float32: 1e-3, np.float64: 1e-12,
           np.complex64: 1e-3, np.complex128: 1e-12}

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker, tol=tol)

  def testTensordotErrors(self):
    a = self.rng().random((3, 2, 2))
    b = self.rng().random((2,))
    self.assertRaisesRegex(
      TypeError, "Number of tensordot axes.*exceeds input ranks.*",
      lambda: jnp.tensordot(a, b, axes=2))

    self.assertRaisesRegex(
      TypeError, "tensordot requires axes lists to have equal length.*",
      lambda: jnp.tensordot(a, b, axes=([0], [0, 1])))

    self.assertRaisesRegex(
      TypeError, "tensordot requires both axes lists to be either ints, tuples or lists.*",
      lambda: jnp.tensordot(a, b, axes=('bad', 'axes')))

    self.assertRaisesRegex(
      TypeError, "tensordot axes argument must be an int, a pair of ints, or a pair of lists.*",
      lambda: jnp.tensordot(a, b, axes='badaxes'))

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, lhs_dtype=lhs_dtype,
          rhs_shape=rhs_shape, rhs_dtype=rhs_dtype)
      # TODO(phawkins): support integer dtypes too.
      for lhs_shape, lhs_dtype in _shape_and_dtypes(all_shapes, inexact_dtypes)
      for rhs_shape, rhs_dtype in _shape_and_dtypes(all_shapes, inexact_dtypes)
      if len(jtu._dims_of_shape(lhs_shape)) == 0
      or len(jtu._dims_of_shape(rhs_shape)) == 0
      or lhs_shape[-1] == rhs_shape[-1]],
  )
  @jax.default_matmul_precision("float32")
  def testInner(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    def np_fun(lhs, rhs):
      lhs = lhs if lhs_dtype != jnp.bfloat16 else lhs.astype(np.float32)
      rhs = rhs if rhs_dtype != jnp.bfloat16 else rhs.astype(np.float32)
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.inner(lhs, rhs).astype(dtype)
    jnp_fun = lambda lhs, rhs: jnp.inner(lhs, rhs)
    tol_spec = {np.float16: 1e-2, np.float32: 1e-5, np.float64: 1e-13,
                np.complex64: 1e-5}
    tol = max(jtu.tolerance(lhs_dtype, tol_spec),
              jtu.tolerance(rhs_dtype, tol_spec))
    # TODO(phawkins): there are float32/float64 disagreements for some inputs.
    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=False, atol=tol, rtol=tol)

  @jtu.sample_product(
    dtype=[dt for dt in float_dtypes if dt not in [jnp.float16, jnp.bfloat16]],
    shape=[shape for shape in one_dim_array_shapes if shape != (1,)],
    num_rhs=[1, 5],
    deg=[1, 2, 3],
    rcond=[None, -1, 10e-3, 10e-5, 10e-10],
    full=[False, True],
    w=[False, True],
    cov=[False, True, "unscaled"],
  )
  @jax.default_matmul_precision("float32")
  def testPolyfit(self, shape, num_rhs, dtype, deg, rcond, full, w, cov):
    rng = jtu.rand_default(self.rng())
    tol_spec = {np.float32: 1e-3, np.float64: 1e-13, np.complex64: 1e-5}
    tol = jtu.tolerance(dtype, tol_spec)
    _w = lambda a: abs(a) if w else None
    rhs_shape = shape + (num_rhs,) if num_rhs > 1 else shape
    args_maker = lambda: [rng(shape, dtype), rng(rhs_shape, dtype), rng(shape, dtype)]
    jnp_fun = lambda x, y, a: jnp.polyfit(x, y, deg=deg, rcond=rcond, full=full, w=_w(a), cov=cov)
    np_fun = jtu.ignore_warning(
      message="Polyfit may be poorly conditioned*")(lambda x, y, a: np.polyfit(x, y, deg=deg, rcond=rcond, full=full, w=_w(a), cov=cov))

    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=False, atol=tol, rtol=tol)

    args = args_maker()
    if not full:
      args = args_maker()
      try:
        np_out = np_fun(*args)
      except ValueError:
        return  # https://github.com/numpy/numpy/issues/22380
      jnp_out = jnp_fun(*args)
      self.assertAllClose(np_out, jnp_out, atol=tol, rtol=tol,
                          check_dtypes=False)
    else:
      # Don't compare the residuals because jnp.linalg.lstsq acts slightly
      # differently to remain `jit`-compatible.
      np_p, _, nrank, nsingular_values, nrcond = np_fun(*args)
      jp_p, _, jrank, jsingular_values, jrcond = jnp_fun(*args)
      self.assertAllClose(
        (np_p, nrank, nsingular_values, nrcond),
        (jp_p, jrank, jsingular_values, jrcond),
        atol=tol, rtol=tol, check_dtypes=False)

  @jtu.sample_product(
    [dict(a_min=a_min, a_max=a_max)
      for a_min, a_max in [(-1, None), (None, 1), (-0.9, 1),
                           (-np.ones(1), None),
                           (None, np.ones(1)),
                           (np.full(1, -0.9), np.ones(1))]
    ],
    shape=all_shapes,
    dtype=float_dtypes + int_dtypes + unsigned_dtypes,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises mixed type promotion
  def testClipStaticBounds(self, shape, dtype, a_min, a_max):
    if np.issubdtype(dtype, np.unsignedinteger):
      a_min = None if a_min is None else abs(a_min)
      a_max = None if a_max is None else abs(a_max)
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.clip(x, a_min=a_min, a_max=a_max)
    jnp_fun = lambda x: jnp.clip(x, min=a_min, max=a_max)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=all_shapes,
    dtype=default_dtypes + unsigned_dtypes,
  )
  def testClipNone(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    self.assertArraysEqual(jnp.clip(x), x)

  def testClipComplexInputError(self):
    rng = jtu.rand_default(self.rng())
    x = rng((5,), dtype=jnp.complex64)
    msg = ".*Complex values have no ordering and cannot be clipped.*"
    # jit is disabled so we don't miss warnings due to caching.
    with jax.disable_jit():
      with self.assertRaisesRegex(ValueError, msg):
        jnp.clip(x)

      with self.assertRaisesRegex(ValueError, msg):
        jnp.clip(x, max=x)

      x = rng((5,), dtype=jnp.int32)
      with self.assertRaisesRegex(ValueError, msg):
        jnp.clip(x, min=-1+5j)

      with self.assertRaisesRegex(ValueError, msg):
        jnp.clip(x, max=jnp.array([-1+5j]))

  def testClipUpperPrecedence(self):
    a_min = 3 * np.ones(1)
    a_max = 2 * np.ones(1)
    x = 4 * np.ones(1)
    y = jnp.clip(x, min=a_min, max=a_max)
    assert y == a_max, f"Expected {y} to equal {a_max} when a_min > a_max."
    assert y == jnp.asarray(np.clip(x, a_min=a_min, a_max=a_max))

  def testHypotComplexInputError(self):
    rng = jtu.rand_default(self.rng())
    x = rng((5,), dtype=jnp.complex64)
    msg = "jnp.hypot is not well defined for complex-valued inputs.*"
    # jit is disabled so we don't miss warnings due to caching.
    with jax.disable_jit():
      with self.assertRaisesRegex(ValueError, msg):
        jnp.hypot(x, x)

      y = jnp.ones_like(x)
      with self.assertRaisesRegex(ValueError, msg):
        jnp.hypot(x, y)

  @jtu.sample_product(
    [dict(shape=shape, dtype=dtype)
      for shape, dtype in _shape_and_dtypes(all_shapes, number_dtypes)],
    decimals=[0, 1, -2],
  )
  def testRoundStaticDecimals(self, shape, dtype, decimals):
    rng = jtu.rand_default(self.rng())
    if jnp.issubdtype(dtype, np.integer) and decimals < 0:
      self.skipTest("Integer rounding with decimals < 0 not implemented")
    np_fun = lambda x: np.round(x, decimals=decimals)
    jnp_fun = lambda x: jnp.round(x, decimals=decimals)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 5e-2, np.float16: 1e-2}
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=check_dtypes, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=check_dtypes,
                          atol=tol, rtol=tol)

  @jtu.sample_product(
    dtype=number_dtypes,
    decimals=[1, 10, 100, 1000],
  )
  def testRoundLargeDecimals(self, dtype, decimals):
    # Regression test for https://github.com/jax-ml/jax/issues/31689.
    # Avoid testing against NumPy here because it returns NaN for large decimals.
    rng = jtu.rand_default(self.rng())
    x = rng((10,), dtype)
    result = jnp.round(x, decimals)
    self.assertArraysAllClose(x, result, atol=2 * 10. ** -decimals)

  @jtu.sample_product(jit=[False, True])
  def testOperatorRound(self, jit):
    jround = jax.jit(round, static_argnums=1) if jit else round
    self.assertAllClose(round(np.float32(7.532), 1),
                        jround(jnp.float32(7.5), 1))
    self.assertAllClose(round(np.float32(1.234), 2),
                        jround(jnp.float32(1.234), 2))
    self.assertAllClose(round(np.float32(1.234)),
                        jround(jnp.float32(1.234)), check_dtypes=False)
    self.assertAllClose(round(np.float32(7.532), 1),
                        jround(jnp.array(7.5, jnp.float32), 1))
    self.assertAllClose(round(np.float32(1.234), 2),
                        jround(jnp.array(1.234, jnp.float32), 2))
    self.assertAllClose(round(np.float32(1.234)),
                        jround(jnp.array(1.234, jnp.float32)),
                        check_dtypes=False)

  def testRoundMethod(self):
    # https://github.com/jax-ml/jax/issues/15190
    (jnp.arange(3.) / 5.).round()  # doesn't crash

  @jtu.sample_product(shape=[(5,), (5, 2)])
  def testOperatorReversed(self, shape):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, 'float32')]
    np_fun = lambda x: np.array(list(reversed(x)))
    jnp_fun = lambda x: jnp.array(list(reversed(x)))

    self._CompileAndCheck(jnp_fun, args_maker)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(mode=mode, shape=shape, dtype=dtype,
          pad_width=pad_width, constant_values=constant_values)
      for mode, shapes in [
          ('constant', all_shapes),
          ('wrap', nonempty_shapes),
          ('edge', nonempty_shapes),
      ]
      for shape, dtype in _shape_and_dtypes(shapes, all_dtypes)
      for constant_values in [
          # None is used for modes other than 'constant'
          None,
          # constant
          0, 1,
          # (constant,)
          (0,), (2.718,),
          # ((before_const, after_const),)
          (0, 2),
          (-1, 3.14),
          # ((before_1, after_1), ..., (before_N, after_N))
          tuple((i / 2, -3.14 * i) for i in range(len(shape))),
      ]
      for pad_width in [
        # ((before_1, after_1), ..., (before_N, after_N))
        tuple((i % 3, (i + 1) % 3) for i in range(len(shape))),
        # ((before, after),)
        ((1, 2),), ((2, 0),),
        # (before, after)  (not in the docstring but works in numpy)
        (2, 0), (0, 0),
        # (pad,)
        (1,), (2,),
        # pad
        0, 1,
      ]
      if (pad_width != () and constant_values != () and
          ((mode == 'constant' and constant_values is not None) or
           (mode != 'constant' and constant_values is None)))],
  )
  def testPad(self, shape, dtype, mode, pad_width, constant_values):
    if np.issubdtype(dtype, np.unsignedinteger):
      constant_values = jax.tree.map(abs, constant_values)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    if constant_values is None:
      np_fun = partial(np.pad, pad_width=pad_width, mode=mode)
      jnp_fun = partial(jnp.pad, pad_width=pad_width, mode=mode)
    else:
      np_fun = partial(np.pad, pad_width=pad_width, mode=mode,
                       constant_values=constant_values)
      jnp_fun = partial(jnp.pad, pad_width=pad_width, mode=mode,
                        constant_values=constant_values)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(mode=mode, shape=shape, dtype=dtype,
          pad_width=pad_width, stat_length=stat_length)
      for mode in ['maximum', 'minimum', 'mean', 'median']
      for shape, dtype in _shape_and_dtypes(nonempty_shapes, all_dtypes)
      for pad_width in [
          # ((before_1, after_1), ..., (before_N, after_N))
          tuple((i % 3, (i + 1) % 3) for i in range(len(shape))),
          # ((before, after),)
          ((1, 2),), ((2, 0),),
          # (before, after)  (not in the docstring but works in numpy)
          (2, 0), (0, 0),
          # (pad,)
          (1,), (2,),
          # pad
          0, 1,
      ]
      for stat_length in [
          None,
          # ((before_1, after_1), ..., (before_N, after_N))
          tuple(((i % 3 + 1), ((i + 1) % 3) + 1) for i in range(len(shape))),
          # ((before, after),)
          ((1, 2),), ((2, 2),),
          # (before, after)  (not in the docstring but works in numpy)
          (1, 1), (3, 4),
          # (pad,)
          (1,), (2,),
          # pad
          1, 2
      ]
      if (pad_width != () and stat_length != () and
          not (dtype in bool_dtypes and mode == 'mean'))],
  )
  def testPadStatValues(self, shape, dtype, mode, pad_width, stat_length):
    if mode == 'median' and np.issubdtype(dtype, np.complexfloating):
      self.skipTest("median statistic is not supported for dtype=complex.")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    np_fun = partial(np.pad, pad_width=pad_width, mode=mode, stat_length=stat_length)
    jnp_fun = partial(jnp.pad, pad_width=pad_width, mode=mode, stat_length=stat_length)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, dtype=dtype,
          pad_width=pad_width, reflect_type=reflect_type)
      for shape, dtype in _shape_and_dtypes(nonempty_shapes, all_dtypes)
      for pad_width in [
          # ((before_1, after_1), ..., (before_N, after_N))
          tuple((i % 3, (i + 1) % 3) for i in range(len(shape))),
          # ((before, after),)
          ((1, 2),), ((2, 3),),
          # (before, after)  (not in the docstring but works in numpy)
          (2, 1), (1, 2),
          # (pad,)
          (1,), (2,), (3,),
          # pad
          0, 5, 7, 10
      ]
      for reflect_type in ['even', 'odd']
      if (pad_width != () and
          # following types lack precision when calculating odd values
          (reflect_type != 'odd' or dtype not in [np.bool_, np.float16, jnp.bfloat16]))],
    mode=['symmetric', 'reflect']
  )
  def testPadSymmetricAndReflect(self, shape, dtype, mode, pad_width, reflect_type):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    np_fun = partial(np.pad, pad_width=pad_width, mode=mode, reflect_type=reflect_type)
    jnp_fun = partial(jnp.pad, pad_width=pad_width, mode=mode, reflect_type=reflect_type)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE,
                            tol={np.float32: 1e-3, np.complex64: 1e-3})
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, dtype=dtype, pad_width=pad_width, end_values=end_values)
      for shape, dtype in _shape_and_dtypes(nonempty_shapes, default_dtypes + complex_dtypes)
      for pad_width in [
        # ((before_1, after_1), ..., (before_N, after_N))
        tuple((i % 3, (i + 1) % 3) for i in range(len(shape))),
        # ((before, after),)
        ((1, 2),), ((2, 0),),
        # (before, after)  (not in the docstring but works in numpy)
        (2, 0), (0, 0),
        # (pad,)
        (1,), (2,),
        # pad
        0, 1,
      ]
      for end_values in [
        # ((before_1, after_1), ..., (before_N, after_N))
        tuple((i % 3, (i + 1) % 3) for i in range(len(shape))),
        # ((before, after),)
        ((1, 2),), ((2.0, 3.14),),
        # (before, after)  (not in the docstring but works in numpy)
        (0, 0), (-8.0, 2.0),
        # (end_values,)
        (1,), (2,),
        # end_values
        0, 1, 100, 10.0, 3.5, 4.2, -5, -3
      ]
      if (pad_width != () and end_values != () and
          # following types lack precision
          dtype not in [np.int8, np.int16, np.float16, jnp.bfloat16])],
  )
  def testPadLinearRamp(self, shape, dtype, pad_width, end_values):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    np_fun = partial(np.pad, pad_width=pad_width, mode="linear_ramp",
                     end_values=end_values)
    jnp_fun = partial(jnp.pad, pad_width=pad_width, mode="linear_ramp",
                      end_values=end_values)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testPadEmpty(self):
    arr = np.arange(6).reshape(2, 3)

    pad_width = ((2, 3), (3, 1))
    np_res = np.pad(arr, pad_width=pad_width, mode="empty")
    jnp_res = jnp.pad(arr, pad_width=pad_width, mode="empty")

    np.testing.assert_equal(np_res.shape, jnp_res.shape)
    np.testing.assert_equal(arr, np_res[2:-3, 3:-1])
    np.testing.assert_equal(arr, jnp_res[2:-3, 3:-1])
    np.testing.assert_equal(np_res[2:-3, 3:-1], jnp_res[2:-3, 3:-1])

  def testPadKwargs(self):
    modes = {
        'constant': {'constant_values': 0},
        'edge': {},
        'linear_ramp': {'end_values': 0},
        'maximum': {'stat_length': None},
        'mean': {'stat_length': None},
        'median': {'stat_length': None},
        'minimum': {'stat_length': None},
        'reflect': {'reflect_type': 'even'},
        'symmetric': {'reflect_type': 'even'},
        'wrap': {},
        'empty': {}
    }
    arr = jnp.array([1, 2, 3])
    pad_width = 1

    for mode in modes.keys():
      allowed = modes[mode]
      not_allowed = {}
      for kwargs in modes.values():
        if kwargs != allowed:
          not_allowed.update(kwargs)

      # Test if allowed keyword arguments pass
      jnp.pad(arr, pad_width, mode, **allowed)
      # Test if prohibited keyword arguments of other modes raise an error
      match = f"unsupported keyword arguments for mode '{mode}'"
      for key, value in not_allowed.items():
        with self.assertRaisesRegex(ValueError, match):
          jnp.pad(arr, pad_width, mode, **{key: value})

    # Test if unsupported mode raise error.
    unsupported_modes = [1, None, "foo"]
    for mode in unsupported_modes:
      match = f"Unimplemented padding mode '{mode}' for np.pad."
      with self.assertRaisesRegex(NotImplementedError, match):
        jnp.pad(arr, pad_width, mode)

  def testPadFunction(self):
    def np_pad_with(vector, pad_width, iaxis, kwargs):
      pad_value = kwargs.get('padder', 10)
      vector[:pad_width[0]] = pad_value
      vector[-pad_width[1]:] = pad_value

    def jnp_pad_with(vector, pad_width, iaxis, kwargs):
      pad_value = kwargs.get('padder', 10)
      vector = vector.at[:pad_width[0]].set(pad_value)
      vector = vector.at[-pad_width[1]:].set(pad_value)
      return vector

    arr = np.arange(6).reshape(2, 3)
    np_res = np.pad(arr, 2, np_pad_with)
    jnp_res = jnp.pad(arr, 2, jnp_pad_with)
    np.testing.assert_equal(np_res, jnp_res)

    arr = np.arange(24).reshape(2, 3, 4)
    np_res = np.pad(arr, 1, np_pad_with, padder=100)
    jnp_res = jnp.pad(arr, 1, jnp_pad_with, padder=100)
    np.testing.assert_equal(np_res, jnp_res)

    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(arr.shape, arr.dtype)]
    jnp_fun = partial(jnp.pad, pad_width=1, mode=jnp_pad_with)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testPadWithNumpyPadWidth(self):
    a = jnp.array([1, 2, 3, 4, 5])
    f = jax.jit(
        partial(
            jnp.pad,
            pad_width=np.asarray((2, 3)),
            mode="constant",
            constant_values=(4, 6)))

    np.testing.assert_array_equal(
        f(a),
        np.pad(
            a,
            pad_width=np.asarray((2, 3)),
            mode="constant",
            constant_values=(4, 6)))

  def testPadWeakType(self):
    x = jnp.array(1.0)[None]
    for mode in ['constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median',
                 'minimum', 'reflect', 'symmetric', 'wrap', 'empty']:
      y = jnp.pad(x, 0, mode=mode)
      self.assertTrue(dtypes.is_weakly_typed(y))

  @jtu.sample_product(
    [dict(shape=shape, dtype=dtype)
      for shape, dtype in _shape_and_dtypes(all_shapes, default_dtypes)],
    reps=[(), (2,), (3, 4), (2, 3, 4), (1, 0, 2)],
  )
  def testTile(self, shape, dtype, reps):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np.tile(arg, reps)
    jnp_fun = lambda arg: jnp.tile(arg, reps)

    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(shape=all_shapes, dtype=all_dtypes)
  def testExtract(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, jnp.float32), rng(shape, dtype)]
    self._CheckAgainstNumpy(np.extract, jnp.extract, args_maker)

  @jtu.sample_product(shape=nonempty_array_shapes, dtype=all_dtypes)
  def testExtractSize(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, jnp.float32), rng(shape, dtype)]
    def jnp_fun(condition, arr):
      return jnp.extract(condition, arr, size=jnp.size(arr) - 1)
    def np_fun(condition, arr):
      size = jnp.size(arr) - 1
      out = np.extract(condition, arr)
      result = np.zeros(np.size(arr) - 1, dtype=dtype)
      size = min(len(out), size)
      result[:size] = out[:size]
      return result
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(ncond=ncond, nfunc=nfunc)
      for ncond in [1, 2, 3]
      for nfunc in [ncond, ncond + 1]
    ],
    shape=all_shapes,
    dtype=all_dtypes)
  def testPiecewise(self, shape, dtype, ncond, nfunc):
    rng = jtu.rand_default(self.rng())
    rng_bool = jtu.rand_int(self.rng(), 0, 2)
    funclist = [lambda x: x - 1, 1, lambda x: x, 0][:nfunc]
    args_maker = lambda: (rng(shape, dtype), [rng_bool(shape, bool) for i in range(ncond)])
    np_fun = partial(np.piecewise, funclist=funclist)
    jnp_fun = partial(jnp.piecewise, funclist=funclist)

    if dtype == np.bool_:
      # The `x - 1` above uses type promotion.
      jnp_fun = jax.numpy_dtype_promotion('standard')(jnp_fun)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    # This is a higher-order function, so the cache miss check will fail.
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True, check_cache_misses=False)

  def testPiecewiseRecompile(self):
    def g(x):
      g.num_traces += 1
      return x
    g.num_traces = 0
    x = jnp.arange(10.0)
    for i in range(5):
      jnp.piecewise(x, [x < 0], [g, 0.])
    self.assertEqual(g.num_traces, 1)

  @jtu.sample_product(
    [dict(shape=shape, perm=perm)
      for shape in array_shapes
      for perm in [
        None,
        tuple(np.random.RandomState(0).permutation(np.zeros(shape).ndim)),
        tuple(np.random.RandomState(0).permutation(
          np.zeros(shape).ndim) - np.zeros(shape).ndim)
      ]
    ],
    dtype=default_dtypes,
    arg_type=["splat", "value"],
  )
  def testTransposeTuple(self, shape, dtype, perm, arg_type):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    if arg_type == "value":
      np_fun = lambda x: x.transpose(perm)
      jnp_fun = lambda x: jnp.array(x).transpose(perm)
    else:
      np_fun = lambda x: x.transpose(*(perm or ()))
      jnp_fun = lambda x: jnp.array(x).transpose(*(perm or ()))

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    shape=array_shapes,
    dtype=default_dtypes,
  )
  def testPermuteDims(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    # Generate a permutation of axes, with some positive and some negative
    axes = self.rng().permutation(len(shape))
    neg = self.rng().randint(0, 2, size=len(shape), dtype=bool)
    axes = np.where(neg, axes - len(shape), axes)

    np_fun = partial(getattr(np, "permute_dims", np.transpose), axes=axes)
    jnp_fun = partial(jnp.permute_dims, axes=axes)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    shape=[s for s in array_shapes if len(s) >= 2],
    dtype=default_dtypes,
    use_property=[True, False]
  )
  def testMatrixTranspose(self, shape, dtype, use_property):
    if use_property:
      jnp_fun = lambda x: jnp.asarray(x).mT
    else:
      jnp_fun = jnp.matrix_transpose
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np.matrix_transpose, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtype=default_dtypes,
    shape=one_dim_array_shapes if jtu.numpy_version() < (2, 2, 0) else all_shapes,
    trim=["f", "b", "fb"],
  )
  def testTrimZeros(self, shape, dtype, trim):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg1: np.trim_zeros(np.asarray(arg1), trim)
    jnp_fun = lambda arg1: jnp.trim_zeros(arg1, trim)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    dtype=default_dtypes,
    shape=[(2, 3), (3, 4)],
    trim=["f", "b", "fb"],
    axis=[None, 0, -1]  # note: contrary to its docs, NumPy errors for multiple axes.
  )
  @unittest.skipIf(jtu.numpy_version() < (2, 2, 0), "n-dimensional trim_zeros requires NumPy 2.2")
  def testTrimZerosAxis(self, shape, dtype, trim, axis):
    print(shape, trim, axis)
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg1: np.trim_zeros(arg1, trim, axis=axis)
    jnp_fun = lambda arg1: jnp.trim_zeros(arg1, trim, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    rank=(1, 2),
    dtype=default_dtypes,
    a_shape=one_dim_array_shapes,
  )
  @jax.default_matmul_precision("float32")
  def testPoly(self, a_shape, dtype, rank):
    if dtype in (np.float16, jnp.bfloat16, np.int16):
      self.skipTest(f"{dtype} gets promoted to {np.float16}, which is not supported.")
    elif rank == 2 and not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("Nonsymmetric eigendecomposition is only implemented on the CPU and GPU backends.")
    rng = jtu.rand_default(self.rng())
    tol = { np.int8: 2e-3, np.int32: 1e-3, np.float32: 1e-3, np.float64: 1e-6 }
    if jtu.test_device_matches(["tpu"]):
      tol[np.int32] = tol[np.float32] = 1e-1
    tol = jtu.tolerance(dtype, tol)
    args_maker = lambda: [rng(a_shape * rank, dtype)]
    self._CheckAgainstNumpy(np.poly, jnp.poly, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp.poly, args_maker, check_dtypes=True, rtol=tol, atol=tol)

  @jtu.sample_product(
    dtype=default_dtypes,
    a_shape=one_dim_array_shapes,
    b_shape=one_dim_array_shapes,
  )
  def testPolyAdd(self, a_shape, b_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polyadd(arg1, arg2)
    jnp_fun = lambda arg1, arg2: jnp.polyadd(arg1, arg2)
    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    dtype=default_dtypes,
    a_shape=one_dim_array_shapes,
    b_shape=one_dim_array_shapes,
  )
  def testPolySub(self, a_shape, b_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polysub(arg1, arg2)
    jnp_fun = lambda arg1, arg2: jnp.polysub(arg1, arg2)
    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    [dict(order=order, k=k, dtype=dtype)
      for dtype in default_dtypes
      for order in range(5)
      for k in [np.arange(order, dtype=dtype), np.ones(1, dtype), None]],
    a_shape=one_dim_array_shapes,
  )
  def testPolyInt(self, a_shape, order, k, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1: np.polyint(arg1, m=order, k=k)
    jnp_fun = lambda arg1: jnp.polyint(arg1, m=order, k=k)
    args_maker = lambda: [rng(a_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    dtype=default_dtypes,
    a_shape=one_dim_array_shapes,
    order=list(range(5)),
  )
  def testPolyDer(self, a_shape, order, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1: np.polyder(arg1, m=order)
    jnp_fun = lambda arg1: jnp.polyder(arg1, m=order)
    args_maker = lambda: [rng(a_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @parameterized.parameters(['int', 'np.int', 'jnp.int'])
  def testIntegerPower(self, ptype):
    p = {'int': 2, 'np.int': np.int32(2), 'jnp.int': jnp.int32(2)}[ptype]
    jaxpr = jax.make_jaxpr(lambda x1: jnp.power(x1, p))(1)
    eqns = jaxpr.eqns
    self.assertLen(eqns, 1)
    self.assertEqual(eqns[0].primitive, lax.integer_pow_p)

  @jtu.sample_product(
    x=[-1, 0, 1],
    y=[0, 32, 64, 128],
  )
  def testIntegerPowerOverflow(self, x, y):
    # Regression test for https://github.com/jax-ml/jax/issues/5987
    args_maker = lambda: [x, y]
    check_dtypes = platform.system() != 'Windows'
    self._CheckAgainstNumpy(
        np.power, jnp.power, args_maker, check_dtypes=check_dtypes
    )
    self._CompileAndCheck(jnp.power, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in [None] + list(range(len(shape)))
    ],
    dtype=all_dtypes,
  )
  def testCompress(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    if shape in scalar_shapes or len(shape) == 0:
      cond_shape = (0,)
    elif axis is None:
      cond_shape = (math.prod(shape),)
    else:
      cond_shape = (shape[axis],)

    args_maker = lambda: [rng(cond_shape, jnp.float32), rng(shape, dtype)]

    np_fun = partial(np.compress, axis=axis)
    jnp_fun = partial(jnp.compress, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in list(range(len(shape)))
    ],
    dtype=all_dtypes,
  )
  def testCompressSize(self, shape, dtype, axis):
    rng = jtu.rand_default(self.rng())
    if shape in scalar_shapes or len(shape) == 0:
      cond_shape = (0,)
    elif axis is None:
      cond_shape = (math.prod(shape),)
    else:
      cond_shape = (shape[axis],)
    args_maker = lambda: [rng(cond_shape, bool), rng(shape, dtype)]

    def np_fun(condition, a, axis=axis, fill_value=1):
      # assuming size = a.shape[axis]
      out = np.compress(condition, a, axis=axis)
      result = np.full_like(a, fill_value)
      result[tuple(slice(s) for s in out.shape)] = out
      return result

    def jnp_fun(condition, a, axis=axis, fill_value=1):
      return jnp.compress(condition, a, axis=axis,
                          size=a.shape[axis], fill_value=fill_value)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=[(2, 3)],
    dtype=int_dtypes,
    # condition entries beyond axis size must be zero.
    condition=[[1], [1, 0, 0, 0, 0, 0, 0]],
    axis=[None, 0, 1],
  )
  def testCompressMismatchedShapes(self, shape, dtype, condition, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [np.array(condition), rng(shape, dtype)]
    np_fun = partial(np.compress, axis=axis)
    jnp_fun = partial(jnp.compress, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in array_shapes
      for axis in [None] + list(range(len(shape)))
    ],
    dtype=all_dtypes,
  )
  def testCompressMethod(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    if shape in scalar_shapes or len(shape) == 0:
      cond_shape = (0,)
    elif axis is None:
      cond_shape = (math.prod(shape),)
    else:
      cond_shape = (shape[axis],)

    args_maker = lambda: [rng(cond_shape, jnp.float32), rng(shape, dtype)]

    np_fun = lambda condition, x: np.asarray(x).compress(condition, axis=axis)
    jnp_fun = lambda condition, x: jnp.asarray(x).compress(condition, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  def testCompressExtractJitErrorMessage(self):
    # Without a static `size`, the output shape of compress/extract is
    # data-dependent, so they cannot be used within transformations like jit.
    # In that case we should raise a clear ConcretizationTypeError pointing the
    # user at the `size` argument, rather than a confusing internal error.
    # https://github.com/jax-ml/jax/issues/38603
    x = jnp.ones((4, 4), dtype=jnp.float32)
    msg = "The size argument of jnp.compress must be specified"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      jax.jit(jnp.compress)(x.ravel() > 0, x.ravel())
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      jax.jit(jnp.extract)(x > 0, x)

  @jtu.sample_product(
    [dict(base_shape=base_shape, axis=axis)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in (None, *range(-len(base_shape)+1, len(base_shape)))
    ],
    arg_dtypes=[
      arg_dtypes
      for num_arrs in [3]
      for arg_dtypes in itertools.combinations_with_replacement(default_dtypes, num_arrs)
    ],
    dtype=[None] + default_dtypes,
  )
  def testConcatenate(self, axis, dtype, base_shape, arg_dtypes):
    rng = jtu.rand_default(self.rng())
    wrapped_axis = 0 if axis is None else axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), arg_dtypes)]
    @jtu.promote_like_jnp
    def np_fun(*args, dtype=dtype):
      dtype = dtype or args[0].dtype
      args = [x if x.dtype != jnp.bfloat16 else x.astype(np.float32)
              for x in args]
      return np.concatenate(args, axis=axis, dtype=dtype, casting='unsafe')
    jnp_fun = lambda *args: jnp.concatenate(args, axis=axis, dtype=dtype)

    def args_maker():
      return [rng(shape, dtype) for shape, dtype in zip(shapes, arg_dtypes)]

    with jtu.strict_promotion_if_dtypes_match(arg_dtypes):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in [(4, 1), (4, 3), (4, 5, 6)]
      for axis in [None] + list(range(1 - len(shape), len(shape) - 1))
    ],
    dtype=all_dtypes,
  )
  def testConcatenateArray(self, shape, dtype, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda x: np.concatenate(x, axis=axis)
    jnp_fun = lambda x: jnp.concatenate(x, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testConcatenateAxisNone(self):
    # https://github.com/jax-ml/jax/issues/3419
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5]])
    jnp.concatenate((a, b), axis=None)

  def testConcatenateScalarAxisNone(self):
    arrays = [np.int32(0), np.int32(1)]
    self.assertArraysEqual(jnp.concatenate(arrays, axis=None),
                           np.concatenate(arrays, axis=None))

  @jtu.sample_product(
    [dict(base_shape=base_shape, axis=axis)
      for base_shape in [(), (4,), (3, 4), (2, 3, 4)]
      for axis in (None, *range(-len(base_shape)+1, len(base_shape)))
    ],
    dtype=default_dtypes,
  )
  def testConcat(self, axis, base_shape, dtype):
    rng = jtu.rand_default(self.rng())
    wrapped_axis = 0 if axis is None else axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size in [3, 1, 4]]
    @jtu.promote_like_jnp
    def np_fun(*args):
      return np.concat(args, axis=axis)
    jnp_fun = lambda *args: jnp.concat(args, axis=axis)
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(base_shape=base_shape, axis=axis)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))],
    arg_dtypes=itertools.combinations_with_replacement(default_dtypes, 2)
  )
  def testAppend(self, axis, base_shape, arg_dtypes):
    rng = jtu.rand_default(self.rng())
    wrapped_axis = axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), arg_dtypes)]
    def np_fun(arr, values):
      arr = arr.astype(np.float32) if arr.dtype == jnp.bfloat16 else arr
      values = (values.astype(np.float32) if values.dtype == jnp.bfloat16
                else values)
      out = np.append(arr, values, axis=axis)
      return out.astype(jnp.promote_types(*arg_dtypes))
    jnp_fun = lambda arr, values: jnp.append(arr, values, axis=axis)

    def args_maker():
      return [rng(shape, dtype) for shape, dtype in zip(shapes, arg_dtypes)]

    with jtu.strict_promotion_if_dtypes_match(arg_dtypes):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis, idx=idx)
      for shape in nonempty_nonscalar_array_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))
      for idx in (range(-math.prod(shape), math.prod(shape))
                  if axis is None else
                  range(-shape[axis], shape[axis]))],
    dtype=all_dtypes,
  )
  def testDeleteInteger(self, shape, dtype, idx, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg: np.delete(arg, idx, axis=axis)
    jnp_fun = lambda arg: jnp.delete(arg, idx, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_nonscalar_array_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))
     ],
    dtype=all_dtypes,
    slc=[slice(None), slice(1, 3), slice(1, 5, 2)],
  )
  def testDeleteSlice(self, shape, dtype, axis, slc):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg: np.delete(arg, slc, axis=axis)
    jnp_fun = lambda arg: jnp.delete(arg, slc, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_nonscalar_array_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))
    ],
    dtype=all_dtypes,
    idx_shape=all_shapes,
  )
  def testDeleteIndexArray(self, shape, dtype, axis, idx_shape):
    rng = jtu.rand_default(self.rng())
    max_idx = np.zeros(shape).size if axis is None else np.zeros(shape).shape[axis]
    idx = jtu.rand_int(self.rng(), low=-max_idx, high=max_idx)(idx_shape, int)
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg: np.delete(arg, idx, axis=axis)
    jnp_fun = lambda arg: jnp.delete(arg, idx, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_nonscalar_array_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))
    ],
    dtype=all_dtypes,
    idx_shape=all_shapes,
  )
  def testDeleteUniqueIndices(self, shape, dtype, axis, idx_shape):
    rng = jtu.rand_default(self.rng())
    max_idx = np.zeros(shape).size if axis is None else np.zeros(shape).shape[axis]
    idx_size = np.zeros(idx_shape).size
    if idx_size > max_idx:
      self.skipTest("Too many indices to be unique")
    def args_maker():
      x = rng(shape, dtype)
      idx = self.rng().choice(max_idx, idx_shape, replace=False)
      return x, idx
    np_fun = partial(np.delete, axis=axis)
    jnp_fun = partial(jnp.delete, axis=axis, assume_unique_indices=True)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_nonscalar_array_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))
    ],
    dtype=all_dtypes,
  )
  def testDeleteMaskArray(self, shape, dtype, axis):
    rng = jtu.rand_default(self.rng())
    mask_size = np.zeros(shape).size if axis is None else np.zeros(shape).shape[axis]
    mask = jtu.rand_int(self.rng(), low=0, high=2)(mask_size, bool)
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg: np.delete(arg, mask, axis=axis)
    jnp_fun = lambda arg: jnp.delete(arg, mask, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_nonscalar_array_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))
    ],
    dtype=all_dtypes,
  )
  def testInsertInteger(self, shape, dtype, axis):
    x = jnp.empty(shape)
    max_ind = x.size if axis is None else x.shape[axis]
    rng = jtu.rand_default(self.rng())
    i_rng = jtu.rand_int(self.rng(), -max_ind, max_ind)
    args_maker = lambda: [rng(shape, dtype), i_rng((), np.int32), rng((), dtype)]
    np_fun = lambda *args: np.insert(*args, axis=axis)
    jnp_fun = lambda *args: jnp.insert(*args, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_nonscalar_array_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))
    ],
    dtype=all_dtypes,
  )
  def testInsertSlice(self, shape, dtype, axis):
    x = jnp.empty(shape)
    max_ind = x.size if axis is None else x.shape[axis]
    rng = jtu.rand_default(self.rng())
    i_rng = jtu.rand_int(self.rng(), -max_ind, max_ind)
    slc = slice(i_rng((), jnp.int32).item(), i_rng((), jnp.int32).item())
    args_maker = lambda: [rng(shape, dtype), rng((), dtype)]
    np_fun = lambda x, val: np.insert(x, slc, val, axis=axis)
    jnp_fun = lambda x, val: jnp.insert(x, slc, val, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.parameters([
    [[[1, 1], [2, 2], [3, 3]], 1, 5, None],
    [[[1, 1], [2, 2], [3, 3]], 1, 5, 1],
    [[[1, 1], [2, 2], [3, 3]], 1, [1, 2, 3], 1],
    [[[1, 1], [2, 2], [3, 3]], [1], [[1],[2],[3]], 1],
    [[1, 1, 2, 2, 3, 3], [2, 2], [5, 6], None],
    [[1, 1, 2, 2, 3, 3], slice(2, 4), [5, 6], None],
    [[1, 1, 2, 2, 3, 3], [2, 2], [7.13, False], None],
    [[[0, 1, 2, 3], [4, 5, 6, 7]], (1, 3), 999, 1]
  ])
  def testInsertExamples(self, arr, index, values, axis):
    # Test examples from the np.insert docstring
    args_maker = lambda: (
      np.asarray(arr), index if isinstance(index, slice) else np.array(index),
      np.asarray(values), axis)
    self._CheckAgainstNumpy(np.insert, jnp.insert, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_array_shapes
      for axis in range(-len(shape), len(shape))
    ],
    dtype=default_dtypes,
    out_dims=[0, 1, 2],
  )
  def testApplyAlongAxis(self, shape, dtype, axis, out_dims):
    def func(x, out_dims):
      if out_dims == 0:
        return x.sum(dtype=x.dtype)
      elif out_dims == 1:
        return x * x[0]
      elif out_dims == 2:
        return x[:, None] + x[None, :]
      else:
        raise NotImplementedError(f"{out_dims=}")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arr: np.apply_along_axis(func, axis, arr, out_dims=out_dims)
    jnp_fun = lambda arr: jnp.apply_along_axis(func, axis, arr, out_dims=out_dims)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            atol={dtypes.bfloat16: 2e-2})
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axes=axes)
      for shape in nonempty_shapes
      for axes in itertools.combinations(range(len(shape)), 2)
    ],
    func=["sum"],
    keepdims=[True, False],
    # Avoid low-precision types in sum()
    dtype=[dtype for dtype in default_dtypes
           if dtype not in [np.float16, jnp.bfloat16]],
  )
  def testApplyOverAxes(self, shape, dtype, func, keepdims, axes):
    f = lambda x, axis: getattr(x, func)(axis=axis, keepdims=keepdims, dtype=dtype)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    np_fun = lambda a: np.apply_over_axes(f, a, axes)
    jnp_fun = lambda a: jnp.apply_over_axes(f, a, axes)
    self._CompileAndCheck(jnp_fun, args_maker)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, dtype=dtype, axis=axis)
      for shape, dtype in _shape_and_dtypes(all_shapes, default_dtypes)
      for axis in [None] + list(range(-len(shape), max(1, len(shape))))
    ],
    repeats=[0, 1, 2],
    fixed_size=[False, True],
  )
  def testRepeat(self, axis, shape, dtype, repeats, fixed_size):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np.repeat(arg, repeats=repeats, axis=axis)
    np_fun = jtu.promote_like_jnp(np_fun)
    if fixed_size:
      total_repeat_length = np.repeat(np.zeros(shape), repeats, axis).shape[axis or 0]
      jnp_fun = lambda arg, rep: jnp.repeat(arg, repeats=rep, axis=axis,
                                     total_repeat_length=total_repeat_length)
      jnp_args_maker = lambda: [rng(shape, dtype), repeats]
      clo_fun = lambda arg: jnp.repeat(arg, repeats=repeats, axis=axis,
                                       total_repeat_length=total_repeat_length)
      clo_fun_args_maker = lambda: [rng(shape, dtype)]
      self._CompileAndCheck(jnp_fun, jnp_args_maker)
      self._CheckAgainstNumpy(np_fun, clo_fun, clo_fun_args_maker)
    else:
      # Now repeats is in a closure, so a constant.
      jnp_fun = lambda arg: jnp.repeat(arg, repeats=repeats, axis=axis)
      args_maker = lambda: [rng(shape, dtype)]
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  def testRepeatScalarFastPath(self):
    a = jnp.array([1,2,3,4])
    f = lambda a: jnp.repeat(a, repeats=2)
    jaxpr = jax.make_jaxpr(f)(a)
    self.assertLessEqual(len(jaxpr.eqns), 6)

  @jtu.sample_product(fixed_size=[False, True])
  def testNonScalarRepeats(self, fixed_size):
    '''
    Following numpy test suite from `test_repeat` at
    https://github.com/numpy/numpy/blob/main/numpy/core/tests/test_multiarray.py
    '''
    tol = 1e-5

    def test_single(m, args_maker, repeats, axis):
      lax_ans = jnp.repeat(m, repeats, axis)
      numpy_ans = np.repeat(m, repeats, axis)

      self.assertAllClose(lax_ans, numpy_ans, rtol=tol, atol=tol)
      if fixed_size:

        # Calculate expected size of the repeated axis.
        rep_length = np.repeat(np.zeros_like(m), repeats, axis).shape[axis or 0]
        jnp_fun = lambda arg, rep: jnp.repeat(
            arg, repeats=rep, axis=axis, total_repeat_length=rep_length)
      else:
        jnp_fun = lambda arg: jnp.repeat(arg, repeats = repeats, axis=axis)
      self._CompileAndCheck(jnp_fun, args_maker)

    m = jnp.array([1,2,3,4,5,6])
    if fixed_size:
      args_maker = lambda: [m, repeats]
    else:
      args_maker = lambda: [m]

    for repeats in [2, jnp.array([1,3,0,1,1,2]), jnp.array([1,3,2,1,1,2]), jnp.array([2])]:
      test_single(m, args_maker, repeats, axis=None)
      test_single(m, args_maker, repeats, axis=0)

    m_rect = m.reshape((2,3))
    if fixed_size:
      args_maker = lambda: [m_rect, repeats]
    else:
      args_maker = lambda: [m_rect]

    for repeats in [2, jnp.array([2,1]), jnp.array([2])]:
      test_single(m_rect, args_maker, repeats, axis=0)

    for repeats in [2, jnp.array([1,3,2]), jnp.array([2])]:
      test_single(m_rect, args_maker, repeats, axis=1)

  def testIssue2330(self):
    '''
    Make sure return value of jnp.concatenate is a jax.ndarray and is side-effect save
    '''
    def attempt_sideeffect(x):
      x = [x]
      x = jnp.concatenate(x)
      x -= 1.
      return x

    np_input = np.ones(1)
    jnp_input = jnp.ones(1)
    expected_np_input_after_call = np.ones(1)
    expected_jnp_input_after_call = jnp.ones(1)

    out = jnp.concatenate([np_input])
    self.assertIs(type(out), array.ArrayImpl)

    attempt_sideeffect(np_input)
    attempt_sideeffect(jnp_input)

    self.assertAllClose(np_input, expected_np_input_after_call)
    self.assertAllClose(jnp_input, expected_jnp_input_after_call)

  @jtu.sample_product(
    mode=['full', 'same', 'valid'],
    op=['convolve', 'correlate'],
    dtype=number_dtypes,
    xshape=one_dim_array_shapes,
    yshape=one_dim_array_shapes,
  )
  def testConvolutions(self, xshape, yshape, dtype, mode, op):
    jnp_op = getattr(jnp, op)
    np_op = getattr(np, op)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    precision = lax.Precision.HIGHEST if jtu.test_device_matches(["tpu"]) else None
    jnp_fun = partial(jnp_op, mode=mode, precision=precision)
    def np_fun(x, y):
      return np_op(x, y, mode=mode).astype(dtypes.to_inexact_dtype(dtype))
    tol = {
        np.float16: 2e-1,
        np.float32: 1e-2,
        np.complex64: 1e-2,
        np.float64: 1e-14,
        np.complex128: 1e-14,
    }
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    mode=['full', 'same', 'valid'],
    op=['convolve', 'correlate'],
    dtype=number_dtypes,
    xshape=one_dim_array_shapes,
    yshape=one_dim_array_shapes,
  )
  @jtu.skip_on_devices("cuda")  # backends don't support all dtypes.
  def testConvolutionsPreferredElementType(self, xshape, yshape, dtype, mode, op):
    if jtu.test_device_matches(["rocm"]) and not dtypes.issubdtype(dtype, np.inexact):
      self.skipTest(f"preferred_element_type={dtype} unsupported for ROCm GPU convolutions")
    jnp_op = getattr(jnp, op)
    np_op = getattr(np, op)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    precision = lax.Precision.HIGHEST if jtu.test_device_matches(["tpu"]) else None
    jnp_fun = partial(jnp_op, mode=mode, precision=precision,
                      preferred_element_type=dtype)
    def np_fun(x, y):
      return np_op(x, y, mode=mode).astype(dtype)
    tol = {np.float16: 2e-1, np.float32: 1e-2, np.float64: 1e-14,
           np.complex128: 1e-14}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))],
    op=["cumsum", "cumprod"],
    dtype=all_dtypes,
    out_dtype=[dtype for dtype in default_dtypes if dtype != np.float16],
  )
  def testCumSumProd(self, axis, shape, dtype, out_dtype, op):
    jnp_op = getattr(jnp, op)
    np_op = getattr(np, op)
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np_op(arg, axis=axis, dtype=out_dtype)
    np_fun = jtu.ignore_warning(category=np.exceptions.ComplexWarning)(np_fun)
    np_fun = jtu.ignore_warning(category=RuntimeWarning,
                                message="overflow encountered.*")(np_fun)
    jnp_fun = lambda arg: jnp_op(arg, axis=axis, dtype=out_dtype)
    jnp_fun = jtu.ignore_warning(category=np.exceptions.ComplexWarning)(jnp_fun)

    args_maker = lambda: [rng(shape, dtype)]

    tol_thresholds = {dtypes.bfloat16: 4e-2}
    tol = max(jtu.tolerance(dtype, tol_thresholds),
              jtu.tolerance(out_dtype, tol_thresholds))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))],
    op=["nancumsum", "nancumprod"],
    dtype=all_dtypes,
    out_dtype=default_dtypes,
  )
  def testNanCumSumProd(self, axis, shape, dtype, out_dtype, op):
    jnp_op = getattr(jnp, op)
    np_op = getattr(np, op)
    rng = jtu.rand_some_nan(self.rng())
    np_fun = partial(np_op, axis=axis, dtype=out_dtype)
    np_fun = jtu.ignore_warning(category=np.exceptions.ComplexWarning)(np_fun)
    np_fun = jtu.ignore_warning(category=RuntimeWarning,
                                message="overflow encountered.*")(np_fun)
    jnp_fun = partial(jnp_op, axis=axis, dtype=out_dtype)
    jnp_fun = jtu.ignore_warning(category=np.exceptions.ComplexWarning)(jnp_fun)

    args_maker = lambda: [rng(shape, dtype)]

    tol_thresholds = {dtypes.bfloat16: 4e-2, np.float16: 3e-3}
    tol = max(jtu.tolerance(dtype, tol_thresholds),
              jtu.tolerance(out_dtype, tol_thresholds))
    if dtype != jnp.bfloat16:
      # numpy functions do not properly handle bfloat16
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True,
                              tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    dtype=default_dtypes,
    n=[0, 4],
    m=[None, 0, 1, 3, 4],
    k=[*range(-4, 4), -2**100, 2**100],
  )
  def testEye(self, n, m, k, dtype):
    np_fun = lambda: np.eye(n, M=m, k=k, dtype=dtype)
    jnp_fun = lambda: jnp.eye(n, M=m, k=k, dtype=dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtype=default_dtypes,
    n=[0, 4],
    m=[None, 0, 1, 3, 4],
    k=range(-4, 4),
  )
  def testEyeDynamicK(self, n, m, k, dtype):
    np_fun = lambda k: np.eye(n, M=m, k=k, dtype=dtype)
    jnp_fun = lambda k: jnp.eye(n, M=m, k=k, dtype=dtype)
    args_maker = lambda: [k]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtype=default_dtypes,
    n=[0, 4],
    m=[None, 0, 1, 3, 4],
    k=[*range(-4, 4), -2**33, 2**33],
  )
  def testTri(self, m, n, k, dtype):
    np_fun = lambda k: np.tri(n, M=m, k=k, dtype=dtype)
    jnp_fun = lambda k: jnp.tri(n, M=m, k=k, dtype=dtype)
    args_maker = lambda: [k]
    if not config.enable_x64.value and (
        k < np.iinfo(np.int32).min or k > np.iinfo(np.int32).max
    ):
      with self.assertRaises(OverflowError):
        jnp_fun(k)
    else:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  def test_tri_bug_22751(self):
    with self.assertRaisesRegex(
        TypeError,
        'Shapes must be 1D sequences of concrete values of integer type',
    ):
      jax.jit(jnp.tri)(3, M=3, k=0)

  @jtu.sample_product(
    dtype=default_dtypes,
    shape=[shape for shape in all_shapes if len(shape) >= 2],
    op=["tril", "triu"],
    k=list(range(-3, 3)),
  )
  def testTriLU(self, dtype, shape, op, k):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: getattr(np, op)(arg, k=k)
    jnp_fun = lambda arg: getattr(jnp, op)(arg, k=k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    n=range(5),
    k=range(-3, 3),
    m=[None, *range(5)],
  )
  def testTrilIndices(self, n, k, m):
    np_fun = lambda n, k, m: np.tril_indices(n, k=k, m=m)
    jnp_fun = lambda n, k, m: jnp.tril_indices(n, k=k, m=m)
    args_maker = lambda: [n, k, m]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
    n=range(5),
    k=range(-3, 3),
    m=[None, *range(5)],
  )
  def testTriuIndices(self, n, k, m):
    np_fun = lambda n, k, m: np.triu_indices(n, k=k, m=m)
    jnp_fun = lambda n, k, m: jnp.triu_indices(n, k=k, m=m)
    args_maker = lambda: [n, k, m]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
    dtype=default_dtypes,
    shape=[(1,1), (1,2), (2,2), (2,3), (3,2), (3,3), (4,4)],
    k=[-1, 0, 1],
  )
  def testTriuIndicesFrom(self, shape, dtype, k):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arr, k: np.triu_indices_from(arr, k=k)
    jnp_fun = lambda arr, k: jnp.triu_indices_from(arr, k=k)
    args_maker = lambda: [rng(shape, dtype), k]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
    dtype=default_dtypes,
    shape=[(1,1), (1,2), (2,2), (2,3), (3,2), (3,3), (4,4)],
    k=[-1, 0, 1],
  )
  def testTrilIndicesFrom(self, shape, dtype, k):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arr, k: np.tril_indices_from(arr, k=k)
    jnp_fun = lambda arr, k: jnp.tril_indices_from(arr, k=k)
    args_maker = lambda: [rng(shape, dtype), k]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @jtu.sample_product(
      n = [2, 3, 4],
      k = [None, -1, 0, 1],
      funcname = ['triu', 'tril']
  )
  def testMaskIndices(self, n, k, funcname):
    kwds = {} if k is None else {'k': k}
    jnp_result = jnp.mask_indices(n, getattr(jnp, funcname), **kwds)
    np_result = np.mask_indices(n, getattr(np, funcname), **kwds)
    self.assertArraysEqual(jnp_result, np_result, check_dtypes=False)

  @jtu.sample_product(
    dtype=default_dtypes,
    a_shape=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (2, 3), (2, 2, 2), (2, 2, 2, 2)],
    val_shape=[(), (1,), (2,), (1, 2), (3, 2)],
  )
  def testFillDiagonal(self, dtype, a_shape, val_shape):
    rng = jtu.rand_default(self.rng())

    def np_fun(a, val):
      a_copy = a.copy()
      np.fill_diagonal(a_copy, val)
      return a_copy

    jnp_fun = partial(jnp.fill_diagonal, inplace=False)
    args_maker = lambda : [rng(a_shape, dtype), rng(val_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    ndim=[0, 1, 4],
    n=[0, 1, 7],
  )
  def testDiagIndices(self, ndim, n):
    np.testing.assert_equal(jtu.with_jax_dtype_defaults(np.diag_indices)(n, ndim),
                            jnp.diag_indices(n, ndim))

  @jtu.sample_product(
    dtype=default_dtypes,
    shape=[(1,1), (2,2), (3,3), (4,4), (5,5)],
  )
  def testDiagIndicesFrom(self, dtype, shape):
    rng = jtu.rand_default(self.rng())
    np_fun = jtu.with_jax_dtype_defaults(np.diag_indices_from)
    jnp_fun = jnp.diag_indices_from
    args_maker = lambda : [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtype=default_dtypes,
    shape=[shape for shape in all_shapes if len(shape) in (1, 2)],
    k=list(range(-4, 4)),
  )
  def testDiag(self, shape, dtype, k):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np.diag(arg, k)
    jnp_fun = lambda arg: jnp.diag(arg, k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtype=default_dtypes,
    shape=all_shapes,
    k=list(range(-4, 4)),
  )
  def testDiagFlat(self, shape, dtype, k):
    rng = jtu.rand_default(self.rng())
    # numpy has inconsistencies for scalar values
    # https://github.com/numpy/numpy/issues/16477
    # jax differs in that it treats scalars values as length-1 arrays
    np_fun = lambda arg: np.diagflat(np.atleast_1d(arg), k)
    jnp_fun = lambda arg: jnp.diagflat(arg, k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @jtu.sample_product(
    dtype=default_dtypes,
    a1_shape=one_dim_array_shapes,
    a2_shape=one_dim_array_shapes,
  )
  def testPolyMul(self, a1_shape, a2_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polymul(arg1, arg2)
    jnp_fun_np = lambda arg1, arg2: jnp.polymul(arg1, arg2, trim_leading_zeros=True)
    jnp_fun_co = lambda arg1, arg2: jnp.polymul(arg1, arg2)
    args_maker = lambda: [rng(a1_shape, dtype), rng(a2_shape, dtype)]
    tol = {np.float16: 2e-1, np.float32: 5e-2, np.float64: 1e-13}
    self._CheckAgainstNumpy(np_fun, jnp_fun_np, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun_co, args_maker, check_dtypes=False)

  @jtu.sample_product(
    dtype=[dtype for dtype in default_dtypes
           if dtype not in (np.float16, jnp.bfloat16)],
    a_shape=one_dim_array_shapes,
    b_shape=one_dim_array_shapes,
  )
  def testPolyDiv(self, a_shape, b_shape, dtype):
    rng = jtu.rand_default(self.rng())

    @jtu.ignore_warning(category=RuntimeWarning, message="divide by zero.*")
    @jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")
    @jtu.ignore_warning(category=RuntimeWarning, message="overflow encountered.*")
    def np_fun(arg1, arg2):
      q, r = np.polydiv(arg1, arg2)
      while r.size < max(arg1.size, arg2.size):  # Pad residual to same size
        r = np.pad(r, (1, 0), 'constant')
      return q, r

    def jnp_fun(arg1, arg2):
      q, r = jnp.polydiv(arg1, arg2, trim_leading_zeros=True)
      while r.size < max(arg1.size, arg2.size):  # Pad residual to same size
        r = jnp.pad(r, (1, 0), 'constant')
      return q, r

    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    tol = {
        dtypes.bfloat16: 2e-1,
        np.float16: 2e-1,
        np.float32: 5e-2,
        np.float64: 5e-7
    }

    jnp_compile = jnp.polydiv # Without trim_leading_zeros (trim_zeros make it unable to be compiled by XLA)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_compile, args_maker, check_dtypes=True, atol=tol, rtol=tol)

  @jtu.sample_product(
    [dict(shape=shape, axis1=axis1, axis2=axis2)
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in [a for a in range(-len(shape), len(shape))
                    if a % len(shape) != axis1 % len(shape)]
    ],
    dtype=default_dtypes,
    offset=list(range(-4, 4)),
  )
  def testDiagonal(self, shape, dtype, offset, axis1, axis2):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np.diagonal(arg, offset, axis1, axis2)
    jnp_fun = lambda arg: jnp.diagonal(arg, offset, axis1, axis2)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtype=default_dtypes,
    n=list(range(4)),
  )
  def testIdentity(self, n, dtype):
    np_fun = lambda: np.identity(n, dtype)
    jnp_fun = lambda: jnp.identity(n, dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=nonempty_shapes,
    period=[None, 0.59],
    left=[None, 0],
    right=[None, 1],
    # Note: skip 8-bit and 16-bit types due to insufficient precision.
    dtype=jtu.dtypes.integer + jtu.dtypes.floating,
    target_dtype=jtu.dtypes.inexact,
  )
  def testInterp(self, shape, dtype, period, left, right, target_dtype):
    rng = jtu.rand_default(self.rng(), scale=10)
    kwds = dict(period=period, left=left, right=right)
    np_fun = partial(np.interp, **kwds)
    jnp_fun = partial(jnp.interp, **kwds)

    args_maker = lambda: [rng(shape, dtype), np.unique(rng((100,), dtype))[:20],
                          rng((20,), target_dtype)]

    with jtu.strict_promotion_if_dtypes_match([dtype, target_dtype]):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                              rtol=3e-3, atol=1e-3)
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product([
    dict(x=0.5, left='extrapolate', expected=5),
    dict(x=1.5, left='extrapolate', expected=15),
    dict(x=3.5, left='extrapolate', expected=30),
    dict(x=3.9, right='extrapolate', expected=39),
  ])
  def testInterpExtrapoate(self, x, expected, **kwargs):
    xp = jnp.array([1.0, 2.0, 3.0])
    fp = jnp.array([10.0, 20.0, 30.0])
    actual = jnp.interp(x, xp, fp, **kwargs)
    self.assertAlmostEqual(actual, expected)

  def testInterpErrors(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'xp and fp must be one-dimensional arrays of equal size'
    ):
      jnp.interp(0.0, jnp.arange(2.0), jnp.arange(3.0))
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "the only valid string value of `left` is 'extrapolate', but got: 'interpolate'"
    ):
      jnp.interp(0.0, jnp.arange(3.0), jnp.arange(3.0), left='interpolate')
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "the only valid string value of `right` is 'extrapolate', but got: 'interpolate'"
    ):
      jnp.interp(0.0, jnp.arange(3.0), jnp.arange(3.0), right='interpolate')
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "jnp.interp: complex x values not supported."
    ):
      jnp.interp(1j, 1j * np.arange(3.0), np.arange(3.0))
    with self.assertRaisesRegex(
        ValueError,
        "period must be a scalar; got"
    ):
      jnp.interp(0.0, jnp.arange(3.0), jnp.arange(3.0), period=np.array([1.0]))

  @jtu.sample_product(
    period=[None, 0.59],
    left=[None, 0],
    right=[None, 1],
    dtype=jtu.dtypes.floating,
  )
  def testInterpGradNan(self, dtype, period, left, right):
    kwds = dict(period=period, left=left, right=right)
    jnp_fun = partial(jnp.interp, **kwds)
    # Probe values of x and xp that are close to zero and close together.
    x = dtype(np.exp(np.linspace(-90, -20, 1000)))
    g = jax.grad(lambda z: jnp.sum(jnp_fun(z, z, jnp.ones_like(z))))(x)
    np.testing.assert_equal(np.all(np.isfinite(g)), True)

  @jtu.sample_product(
    [dict(x1_shape=x1_shape, x2_shape=x2_shape)
     for x1_shape, x2_shape in filter(_shapes_are_broadcast_compatible,
                                      itertools.combinations_with_replacement(array_shapes, 2))
    ],
    x1_rng_factory=[jtu.rand_some_inf_and_nan, jtu.rand_some_zero],
    x2_rng_factory=[partial(jtu.rand_int, low=-1075, high=1024)],
    x1_dtype=default_dtypes,
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLdexp(self, x1_shape, x1_dtype, x2_shape, x1_rng_factory, x2_rng_factory):
    x1_rng = x1_rng_factory(self.rng())
    x2_rng = x2_rng_factory(self.rng())

    @jtu.ignore_warning(category=RuntimeWarning, message="overflow.*")
    def np_fun(x1, x2):
      out_dtype = dtypes.to_inexact_dtype(x1.dtype)
      return np.ldexp(x1.astype(out_dtype), x2)

    jnp_fun = jnp.ldexp
    args_maker = lambda: [x1_rng(x1_shape, x1_dtype),
                          x2_rng(x2_shape, np.int32)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.parameters(*float_dtypes)
  def testLdexpOverflow(self, dtype):
    # Regression test for https://github.com/jax-ml/jax/issues/28040
    args_maker = lambda: [np.array(0.5, dtype), 1 << (dtypes.finfo(dtype).nexp - 1)]
    def np_ldexp(x1, x2):
      return np.ldexp(x1, x2).astype(x1.dtype)
    self._CheckAgainstNumpy(np_ldexp, jnp.ldexp, args_maker)
    self._CompileAndCheck(jnp.ldexp, args_maker)

  @parameterized.parameters(*float_dtypes)
  def testLdexpExtremeValues(self, dtype):
    # Regression test for https://github.com/jax-ml/jax/issues/28040
    def args_maker():
      info = dtypes.finfo(dtype)
      span = int(np.log2(float(info.max)) - np.log2(float(info.tiny))) - 1
      return [np.array([info.tiny, info.max], dtype=dtype),
              np.array([span, -span])]
    def np_ldexp(x1, x2):
      return np.ldexp(x1, x2).astype(x1.dtype)
    self._CheckAgainstNumpy(np_ldexp, jnp.ldexp, args_maker)
    self._CompileAndCheck(jnp.ldexp, args_maker)

  @jtu.sample_product(
      rng_factory=[
          jtu.rand_some_inf_and_nan,
          jtu.rand_some_zero,
          partial(jtu.rand_not_small, offset=1e8),
      ],
      shape=all_shapes,
      dtype=default_dtypes,
  )
  @jtu.ignore_warning(category=RuntimeWarning, message="overflow")
  @jtu.ignore_warning(category=RuntimeWarning,
                      message="invalid value encountered in isinf")
  @unittest.skipIf(
      platform.system() == "Windows",
      "TODO (ybaturina): Test fails on Windows b/435663064."
  )
  def testFrexp(self, shape, dtype, rng_factory):
    # integer types are converted to float64 in numpy's implementation
    if (dtype not in [jnp.bfloat16, np.float16, np.float32]
        and not config.enable_x64.value):
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    def np_frexp(x):
      mantissa, exponent = np.frexp(x)
      # NumPy is inconsistent between Windows and Linux/Mac on what the
      # value of exponent is if the input is infinite. Normalize to the Linux
      # behavior.
      exponent = np.where(np.isinf(mantissa), np.zeros_like(exponent), exponent)
      return mantissa, exponent
    self._CheckAgainstNumpy(np_frexp, jnp.frexp, args_maker,
                            check_dtypes=np.issubdtype(dtype, np.inexact))
    self._CompileAndCheck(jnp.frexp, args_maker)

  def _np_ftz(self, x):
    info = dtypes.finfo(x.dtype)
    with jtu.ignore_warning(category=RuntimeWarning), np.errstate(all="ignore"):
      zero = np.copysign(np.float32(0.0), x.astype(np.float32)).astype(x.dtype)
      return np.where(np.abs(x) < info.tiny, zero, x)

  @parameterized.parameters(*float_dtypes, jnp.float8_e5m2)
  @jtu.ignore_warning(category=RuntimeWarning)
  def testFlushSubnormals(self, dtype):
    if dtype == np.float64 and not config.enable_x64.value:
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    info = dtypes.finfo(dtype)
    x = self._sample_float_inputs(dtype)
    out = np.asarray(jax.jit(jnp_ufuncs._flush_subnormals)(x))
    expected = self._np_ftz(x)
    not_nan = ~np.isnan(expected)
    uint_dtype = f"uint{info.bits}"
    self.assertArraysEqual(
        out[not_nan].view(uint_dtype), expected[not_nan].view(uint_dtype)
    )
    self.assertTrue(np.all(np.isnan(out[~not_nan])))

  def _sample_float_inputs(self, dtype):
    info = dtypes.finfo(dtype)
    if info.bits <= 16:
      return np.arange(1 << info.bits, dtype=f"uint{info.bits}").view(dtype)
    tiny = info.tiny
    return np.array([
        0.0, -0.0, tiny * 0.25, -tiny * 0.25, tiny, -tiny,
        0.5, -0.5, 1.0, -1.5, 3.14159, info.max, -info.max,
        np.inf, -np.inf, np.nan,
    ], dtype=dtype)

  @parameterized.parameters(*float_dtypes, jnp.float8_e5m2)
  @jtu.ignore_warning(category=RuntimeWarning)
  def testFrexpBitExact(self, dtype):
    if dtype == np.float64 and not config.enable_x64.value:
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    info = dtypes.finfo(dtype)
    x = self._sample_float_inputs(dtype)
    m_jax, e_jax = jax.jit(jnp.frexp)(x)

    x_ftz = self._np_ftz(x)
    work_dtype = (
        np.float32 if dtype in (jnp.bfloat16, jnp.float8_e5m2) else dtype
    )
    with np.errstate(all="ignore"):
      m_np, e_np = np.frexp(x_ftz.astype(work_dtype))
      m_np = m_np.astype(dtype)

    not_nan = ~np.isnan(m_np)
    uint_dtype = f"uint{info.bits}"
    self.assertArraysEqual(
        m_jax[not_nan].view(uint_dtype), m_np[not_nan].view(uint_dtype)
    )
    self.assertTrue(np.all(np.isnan(m_jax[~not_nan])))
    self.assertArraysEqual(e_jax, e_np.astype(np.int32))

  @parameterized.parameters(*float_dtypes, jnp.float8_e5m2)
  @jtu.ignore_warning(category=RuntimeWarning)
  def testLdexpBitExact(self, dtype):
    if dtype == np.float64 and not config.enable_x64.value:
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    info = dtypes.finfo(dtype)
    x = self._sample_float_inputs(dtype)
    shifts = np.array(
        [-3000, -300, -35, -15, -1, 0, 1, 15, 35, 300, 3000], dtype=np.int32
    )
    x_grid, n_grid = np.meshgrid(x, shifts, indexing="ij")
    out_jax = np.asarray(jax.jit(jnp.ldexp)(x_grid, n_grid))

    x_ftz = self._np_ftz(x_grid)
    with np.errstate(all="ignore"):
      val_f64 = np.ldexp(x_ftz.astype(np.float64), n_grid)
      zero_f64 = np.copysign(0.0, val_f64)
      out_np = np.where(
          np.abs(val_f64) < float(info.tiny), zero_f64, val_f64
      ).astype(dtype)

    not_nan = ~np.isnan(out_np)
    uint_dtype = f"uint{info.bits}"
    self.assertArraysEqual(
        out_jax[not_nan].view(uint_dtype), out_np[not_nan].view(uint_dtype)
    )
    self.assertTrue(np.all(np.isnan(out_jax[~not_nan])))

    # Verify boolean, narrow, and wide integer exponent dtypes.
    for exp_dtype in [
        np.bool_, np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
    ]:
      n_val = exp_dtype(1 if exp_dtype == np.bool_ else 3)
      expected = np.array(3.0 if exp_dtype == np.bool_ else 12.0, dtype=dtype)
      res = jax.jit(jnp.ldexp)(np.array(1.5, dtype=dtype), n_val)
      self.assertArraysEqual(res, expected)

    with self.assertRaises(ValueError):
      jnp.ldexp(np.array(1.0, dtype=dtype), 1.0)

    x_b = jnp.array([[1.5, 2.0], [3.0, 4.0]], dtype=dtype)
    n_b = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    for in_axes, expected in [
        ((0, 0), jnp.ldexp(x_b, n_b)),
        ((0, None), jnp.ldexp(x_b, n_b[0:1])),
        ((None, 0), jnp.ldexp(x_b[0:1], n_b)),
        ((1, 0), jnp.ldexp(x_b.T, n_b)),
    ]:
      self.assertAllClose(jax.vmap(jnp.ldexp, in_axes=in_axes)(
          x_b if in_axes[0] is not None else x_b[0],
          n_b if in_axes[1] is not None else n_b[0],
      ), expected)

  @jtu.sample_product(
    [dict(shape=shape, axis1=axis1, axis2=axis2)
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in range(-len(shape), len(shape))
      if (axis1 % len(shape)) != (axis2 % len(shape))
    ],
    dtype=default_dtypes,
    out_dtype=[None] + number_dtypes,
    offset=list(range(-4, 4)),
  )
  def testTrace(self, shape, dtype, out_dtype, offset, axis1, axis2):
    rng = jtu.rand_default(self.rng())
    def np_fun(arg):
      if out_dtype == jnp.bfloat16:
        return np.trace(arg, offset, axis1, axis2, np.float32).astype(jnp.bfloat16)
      else:
        return np.trace(arg, offset, axis1, axis2, out_dtype)
    jnp_fun = lambda arg: jnp.trace(arg, offset, axis1, axis2, out_dtype)
    args_maker = lambda: [rng(shape, dtype)]
    # TODO: Fails with uint8/uint16 output dtypes (integer overflow?)
    if out_dtype not in (np.uint8, np.uint16, np.uint32):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testTraceSameAxesError(self):
    a = jnp.arange(1, 13).reshape(2, 3, 2)
    with self.assertRaisesRegex(ValueError, r"axis1 and axis2 can not be same"):
      jnp.trace(a, axis1=1, axis2=-2)

  @jtu.sample_product(
    ashape=[(15,), (16,), (17,)],
    vshape=[(), (5,), (5, 5)],
    side=['left', 'right'],
    dtype=number_dtypes,
    method=['sort', 'scan', 'scan_unrolled', 'compare_all'],
    use_sorter=[True, False],
    use_method=[True, False],
  )
  def testSearchsorted(self, ashape, vshape, side, dtype, method, use_sorter, use_method):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      a = rng(ashape, dtype)
      v = rng(vshape, dtype)
      return (a, v, np.argsort(a)) if use_sorter else (np.sort(a), v)
    def np_fun(a, v, sorter=None):
      return np.searchsorted(a, v, side=side, sorter=sorter).astype('int32')
    def jnp_fun(a, v, sorter=None):
      if use_method:
        return jnp.asarray(a).searchsorted(v, side=side, method=method, sorter=sorter)
      else:
        return jnp.searchsorted(a, v, side=side, method=method, sorter=sorter)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @unittest.skipIf(
    platform.system() == "Windows",
    "Under Windows, NumPy throws if 2**32 is converted to an int32"
  )
  def testSearchsortedDtype(self):
    # Test that for large arrays, int64 indices are used. We test this
    # via abstract evaluation to avoid allocating a large array in tests.
    a_int32 = core.ShapedArray((np.iinfo(np.int32).max,), np.float32)
    a_int64 = core.ShapedArray((np.iinfo(np.int32).max + 1,), np.float32)
    v = core.ShapedArray((), np.float32)

    out_int32 = jax.eval_shape(jnp.searchsorted, a_int32, v)
    self.assertEqual(out_int32.dtype, np.int32)

    if (
        config.enable_x64.value
        or config.explicit_x64_dtypes.value == config.ExplicitX64Mode.ALLOW
    ):
      out_int64 = jax.eval_shape(jnp.searchsorted, a_int64, v)
      self.assertEqual(out_int64.dtype, np.int64)
    elif config.explicit_x64_dtypes.value == config.ExplicitX64Mode.WARN:
      with self.assertWarnsRegex(UserWarning, "Explicitly requested dtype.*int64"):
        with self.assertRaisesRegex(OverflowError, "Python integer 2147483648 out of bounds.*"):
          jax.eval_shape(jnp.searchsorted, a_int64, v)

  @jtu.sample_product(
    dtype=inexact_dtypes,
    side=['left', 'right'],
    method=['sort', 'scan', 'compare_all'],
  )
  def testSearchsortedNans(self, dtype, side, method):
    if np.issubdtype(dtype, np.complexfloating):
      raise SkipTest("Known failure for complex inputs; see #9107")
    x = np.array([-np.inf, -1.0, 0.0, -0.0, 1.0, np.inf, np.nan, -np.nan], dtype=dtype)
    # The sign bit should not matter for 0.0 or NaN, so argsorting the above should be
    # equivalent to argsorting the following:
    x_equiv = np.array([0, 1, 2, 2, 3, 4, 5, 5])

    if jnp.issubdtype(dtype, jnp.complexfloating):
      x = np.array([complex(r, c) for r, c in itertools.product(x, repeat=2)])
      x_equiv = np.array([complex(r, c) for r, c in itertools.product(x_equiv, repeat=2)])

    fun = partial(jnp.searchsorted, side=side, method=method)
    self.assertArraysEqual(fun(x, x), fun(x_equiv, x_equiv))
    self.assertArraysEqual(jax.jit(fun)(x, x), fun(x_equiv, x_equiv))

  @jtu.sample_product(
    xshape=[(20,), (5, 4)],
    binshape=[(0,), (1,), (5,)],
    right=[True, False],
    reverse=[True, False],
    dtype=default_dtypes,
  )
  def testDigitize(self, xshape, binshape, right, reverse, dtype):
    order = jnp.index_exp[::-1] if reverse else jnp.index_exp[:]
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), jnp.sort(rng(binshape, dtype))[order]]
    np_fun = lambda x, bins: np.digitize(x, bins, right=right).astype('int32')
    jnp_fun = lambda x, bins: jnp.digitize(x, bins, right=right)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    xshape=[(20,), (5, 4)],
    binshape=[(0,), (1,), (5,)],
    right=[True, False],
    method=['scan', 'scan_unrolled', 'sort', 'compare_all'],
    reverse=[True, False],
    dtype=default_dtypes,
  )
  def testDigitizeMethod(self, xshape, binshape, right, method, reverse, dtype):
    order = jnp.index_exp[::-1] if reverse else jnp.index_exp[:]
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), jnp.sort(rng(binshape, dtype))[order]]
    np_fun = lambda x, bins: np.digitize(x, bins, right=right).astype('int32')
    jnp_fun = lambda x, bins: jnp.digitize(x, bins, right=right, method=method)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    dtypes=[
      [np.float32],
      [np.float32, np.float32],
      [np.float32, np.int32, np.float32],
      [np.float32, np.int64, np.float32],
      [np.float32, np.int32, np.float64],
    ],
    shape=[(), (2,), (3, 4), (1, 5)],
    array_input=[True, False],
  )
  def testColumnStack(self, shape, dtypes, array_input):
    rng = jtu.rand_default(self.rng())
    if array_input:
      args_maker = lambda: [np.array([rng(shape, dtype) for dtype in dtypes])]
    else:
      args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    np_fun = jtu.promote_like_jnp(np.column_stack)
    jnp_fun = jnp.column_stack
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in [(), (2,), (3, 4), (1, 100)]
      for axis in range(-len(shape), len(shape) + 1)
    ],
    dtypes=[
      [np.float32],
      [np.float32, np.float32],
      [np.float32, np.int32, np.float32],
      [np.float32, np.int64, np.float32],
      [np.float32, np.int32, np.float64],
    ],
    array_input=[True, False],
    out_dtype=[np.float32, np.int32],
  )
  def testStack(self, shape, axis, dtypes, array_input, out_dtype):
    rng = jtu.rand_default(self.rng())
    if array_input:
      args_maker = lambda: [np.array([rng(shape, dtype) for dtype in dtypes])]
    else:
      args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]

    np_fun = jtu.promote_like_jnp(partial(np.stack, axis=axis, dtype=out_dtype, casting='unsafe'))

    jnp_fun = partial(jnp.stack, axis=axis, dtype=out_dtype)
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    op=["hstack", "vstack", "dstack"],
    dtypes=[
      [np.float32],
      [np.float32, np.float32],
      [np.float32, np.int32, np.float32],
      [np.float32, np.int64, np.float32],
      [np.float32, np.int32, np.float64],
    ],
    shape=[(), (2,), (3, 4), (1, 100), (2, 3, 4)],
    array_input=[True, False],
    out_dtype=[np.float32, np.int32],
  )
  def testHVDStack(self, shape, op, dtypes, array_input, out_dtype):
    rng = jtu.rand_default(self.rng())
    if array_input:
      args_maker = lambda: [np.array([rng(shape, dtype) for dtype in dtypes])]
    else:
      args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]

    if op == "dstack":
      np_fun = jtu.promote_like_jnp(lambda *args: getattr(np, op)(*args).astype(out_dtype))
    else:
      np_fun = partial(jtu.promote_like_jnp(getattr(np, op)), dtype=out_dtype,
                       casting='unsafe')

    jnp_fun = partial(getattr(jnp, op), dtype=out_dtype)
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(name=name, **kwds)
      for name in ['blackman', 'bartlett', 'hamming', 'hanning', 'kaiser']
      for kwds in ([dict(beta=1), dict(beta=0.5)] if name == 'kaiser' else [{}])
    ],
    size = [0, 1, 5, 10],
  )
  def testWindowFunction(self, name, size, **kwds):
    jnp_fun = partial(getattr(jnp, name), size, **kwds)
    np_fun = jtu.with_jax_dtype_defaults(partial(getattr(np, name), size, **kwds))
    args_maker = lambda: []
    tol = (
        5e-6 if jtu.test_device_matches(['tpu']) and name == 'kaiser' else None
    )
    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker, atol=tol, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, fill_value_shape=fill_value_shape)
      for shape in array_shapes + [3, np.array(7, dtype=np.int32)]
      for fill_value_shape in _compatible_shapes(shape)],
    fill_value_dtype=default_dtypes,
    out_dtype=[None] + default_dtypes,
  )
  def testFull(self, shape, fill_value_dtype, fill_value_shape, out_dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda fill_value: np.full(shape, fill_value, dtype=out_dtype)
    jnp_fun = lambda fill_value: jnp.full(shape, fill_value, dtype=out_dtype)
    args_maker = lambda: [rng(fill_value_shape, fill_value_dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, dtype=dtype, axis=axis)
      for shape, dtype in _shape_and_dtypes(nonempty_nonscalar_array_shapes, default_dtypes)
      for axis in list(range(-len(shape), max(1, len(shape))))
    ],
    prepend=[None, 1, 0],
    append=[None, 1, 0],
    n=[0, 1, 2],
  )
  def testDiff(self, shape, dtype, n, axis, prepend, append):
    prepend = np.zeros(shape, dtype=dtype) if prepend == 0 else prepend
    append = np.zeros(shape, dtype=dtype) if append == 0 else append
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    def np_fun(x, n=n, axis=axis, prepend=prepend, append=append):
      if prepend is None:
        prepend = np._NoValue
      elif not np.isscalar(prepend) and prepend.dtype == jnp.bfloat16:
        prepend = prepend.astype(np.float32)

      if append is None:
        append = np._NoValue
      elif not np.isscalar(append) and append.dtype == jnp.bfloat16:
        append = append.astype(np.float32)

      if x.dtype == jnp.bfloat16:
        return np.diff(x.astype(np.float32), n=n, axis=axis, prepend=prepend, append=append).astype(jnp.bfloat16)
      else:
        return np.diff(x, n=n, axis=axis, prepend=prepend, append=append)

    jnp_fun = lambda x: jnp.diff(x, n=n, axis=axis, prepend=prepend, append=append)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testDiffBool(self):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((10,), bool)]
    self._CheckAgainstNumpy(np.diff, jnp.diff, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp.diff, args_maker)

  def testDiffPrepoendScalar(self):
    # Regression test for https://github.com/jax-ml/jax/issues/19362
    x = jnp.arange(10)
    result_jax = jnp.diff(x, prepend=x[0], append=x[-1])

    x = np.array(x)
    result_numpy = np.diff(x, prepend=x[0], append=x[-1])

    self.assertArraysEqual(result_jax, result_numpy)

  @jtu.sample_product(
    op=["zeros", "ones"],
    shape=[2, (), (2,), (3, 0), np.array((4, 5, 6), dtype=np.int32),
           np.array(4, dtype=np.int32)],
    dtype=all_dtypes,
  )
  def testZerosOnes(self, op, shape, dtype):
    np_op = getattr(np, op)
    jnp_op = getattr(jnp, op)
    args_maker = lambda: []
    np_op = partial(np_op, shape, dtype)
    jnp_op = partial(jnp_op, shape, dtype)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  def testOnesWithInvalidShape(self):
    with self.assertRaises(TypeError):
      jnp.ones((-1, 1))

  def test_full_like_committed(self):
    x = jnp.array((1, 2, 3), dtype=np.int32)
    self.assertFalse(x._committed)
    self.assertFalse(lax.full_like(x, 1.1)._committed)
    x = jax.device_put(x, jax.devices()[-1])
    self.assertTrue(x._committed)
    y = lax.full_like(x, 1.1)
    self.assertTrue(y._committed)
    self.assertEqual(x.sharding, y.sharding)

  def test_zeros_like_with_explicit_device_and_jitted(self):
    x = jnp.array((1, 2, 3), dtype=np.int32)
    x = jax.device_put(x, jax.devices()[0])
    zeros_like_with_device = partial(jnp.zeros_like, device=jax.devices()[0])
    y = jax.jit(zeros_like_with_device)(x)
    self.assertEqual(x.shape, y.shape)
    self.assertEqual(y.sharding, make_single_device_sharding(jax.devices()[0]))

  @jtu.sample_product(
    [dict(shape=shape, out_shape=out_shape, fill_value_shape=fill_value_shape)
      for shape in array_shapes
      for out_shape in [None] + array_shapes
      for fill_value_shape in _compatible_shapes(shape if out_shape is None else out_shape)
    ],
    in_dtype=default_dtypes,
    fill_value_dtype=default_dtypes,
    out_dtype=default_dtypes,
  )
  def testFullLike(self, shape, in_dtype, fill_value_dtype, fill_value_shape, out_dtype, out_shape):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x, fill_value: np.full_like(
      x, fill_value, dtype=out_dtype, shape=out_shape)
    jnp_fun = lambda x, fill_value: jnp.full_like(
      x, fill_value, dtype=out_dtype, shape=out_shape)
    args_maker = lambda: [rng(shape, in_dtype), rng(fill_value_shape, fill_value_dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=array_shapes,
    out_shape=[None] + array_shapes,
    in_dtype=default_dtypes,
    func=["ones_like", "zeros_like"],
    out_dtype=default_dtypes,
  )
  def testZerosOnesLike(self, func, shape, in_dtype, out_shape, out_dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: getattr(np, func)(x, dtype=out_dtype, shape=out_shape)
    jnp_fun = lambda x: getattr(jnp, func)(x, dtype=out_dtype, shape=out_shape)
    args_maker = lambda: [rng(shape, in_dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      func=[jnp.empty, jnp.zeros, jnp.ones, jnp.full],
      shape=array_shapes,
      dtype=default_dtypes,
  )
  def testArrayCreationWithDevice(self, func, shape, dtype):
    device = jax.devices()[-1]
    kwds = {'fill_value': 1} if func is jnp.full else {}
    out = func(**kwds, shape=shape, dtype=dtype, device=device)
    self.assertEqual(out.devices(), {device})

  @jtu.sample_product(
      func=[jnp.empty, jnp.zeros, jnp.ones, jnp.full],
      shape=array_shapes,
      dtype=default_dtypes,
  )
  def testArrayCreationWithSharding(self, func, shape, dtype):
    sharding = make_single_device_sharding(jax.devices()[-1])
    kwds = {'fill_value': 1} if func is jnp.full else {}
    out = func(**kwds, shape=shape, dtype=dtype, device=sharding)
    self.assertEqual(out.sharding, sharding)

  @jtu.sample_product(
      func=[
        lambda dtype, device: jnp.arange(5, dtype=dtype, device=device),
        lambda dtype, device: jnp.eye(5, 6, dtype=dtype, device=device),
        lambda dtype, device: jnp.linspace(5, 6, 7, dtype=dtype, device=device),
        lambda dtype, device: jnp.linspace(5, 6, 7, retstep=True, dtype=dtype, device=device),
        lambda dtype, device: jnp.array([1, 2, 3, 4, 5], dtype=dtype, device=device),
      ],
      dtype=default_dtypes,
  )
  def testArangeEyeLinspaceArrayWithDevice(self, func, dtype):
    device = jax.devices()[-1]
    output = func(dtype=dtype, device=device)
    if isinstance(output, tuple):
      self.assertEqual(output[0].devices(), {device})
    else:
      self.assertEqual(output.devices(), {device})

  @jtu.sample_product(
      func=[
        lambda dtype, device: jnp.arange(5, dtype=dtype, device=device),
        lambda dtype, device: jnp.eye(5, 6, dtype=dtype, device=device),
        lambda dtype, device: jnp.linspace(5, 6, 7, dtype=dtype, device=device),
        lambda dtype, device: jnp.linspace(5, 6, 7, retstep=True, dtype=dtype, device=device),
        lambda dtype, device: jnp.array([1, 2, 3, 4, 5], dtype=dtype, device=device),
      ],
      dtype=default_dtypes,
  )
  def testArangeEyeLinspaceArrayWithSharding(self, func, dtype):
    sharding = make_single_device_sharding(jax.devices()[-1])
    output = func(dtype=dtype, device=sharding)
    if isinstance(output, tuple):
      self.assertEqual(output[0].sharding, sharding)
    else:
      self.assertEqual(output.sharding, sharding)

  @jtu.sample_product(
      func=[jnp.empty_like, jnp.zeros_like, jnp.ones_like, jnp.full_like],
      shape=array_shapes,
      dtype=default_dtypes,
  )
  def testFullLikeWithDevice(self, func, shape, dtype):
    device = jax.devices()[-1]
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    kwds = {'fill_value': 1} if func is jnp.full_like else {}

    with self.subTest('device from keyword'):
      out = func(x, **kwds, device=device)
      self.assertEqual(out.devices(), {device})

    with self.subTest('device from input array'):
      out2 = func(out, **kwds)
      self.assertEqual(out2.devices(), out.devices())

  @jtu.sample_product(
      func=[jnp.empty_like, jnp.zeros_like, jnp.ones_like, jnp.full_like],
      shape=array_shapes,
      dtype=default_dtypes,
  )
  def testFullLikeWithSharding(self, func, shape, dtype):
    sharding = make_single_device_sharding(jax.devices()[-1])
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    kwds = {'fill_value': 1} if func is jnp.full_like else {}

    with self.subTest('device from keyword'):
      out = func(x, **kwds, device=sharding)
      self.assertEqual(out.sharding, sharding)

    with self.subTest('device from input array'):
      out2 = func(out, **kwds)
      self.assertEqual(out2.devices(), out.devices())

  def testDuckTypedLike(self):
    x = jax.ShapeDtypeStruct((1, 2, 3), np.dtype("int32"))
    self.assertArraysEqual(jnp.zeros_like(x), jnp.zeros(x.shape, x.dtype))
    self.assertArraysEqual(jnp.ones_like(x), jnp.ones(x.shape, x.dtype))
    self.assertArraysEqual(jnp.full_like(x, 2), jnp.full(x.shape, 2, x.dtype))

    self.assertEqual(jnp.empty_like(x).shape, jnp.empty(x.shape, x.dtype).shape)
    self.assertEqual(jnp.empty_like(x).dtype, jnp.empty(x.shape, x.dtype).dtype)

  @jtu.sample_product(
    [dict(func=func, args=args)
     for func, args in [("full_like", (-100,)), ("ones_like", ()), ("zeros_like", ())]
    ],
    shape=array_shapes,
    in_dtype=[np.int32, np.float32, np.complex64],
    weak_type=[True, False],
    out_shape=[None, (), (10,)],
    out_dtype=[None, float],
  )
  def testZerosOnesFullLikeWeakType(self, func, args, shape, in_dtype, weak_type, out_shape, out_dtype):
    rng = jtu.rand_default(self.rng())
    x = lax_internal._convert_element_type(rng(shape, in_dtype),
                                           weak_type=weak_type)
    fun = lambda x: getattr(jnp, func)(x, *args, dtype=out_dtype, shape=out_shape)
    expected_weak_type = weak_type and (out_dtype is None)
    self.assertEqual(dtypes.is_weakly_typed(fun(x)), expected_weak_type)
    self.assertEqual(dtypes.is_weakly_typed(jax.jit(fun)(x)), expected_weak_type)

  @jtu.sample_product(
    funcname=["array", "asarray"],
    dtype=[int, float, None],
    val=[0, 1],
    input_type=[int, float, np.int32, np.float32],
  )
  def testArrayWeakType(self, funcname, input_type, val, dtype):
    func = lambda x: getattr(jnp, funcname)(x, dtype=dtype)
    fjit = jax.jit(func)
    val = input_type(val)
    expected_weak_type = dtype is None and input_type in set(dtypes._weak_types)
    self.assertEqual(dtypes.is_weakly_typed(func(val)), expected_weak_type)
    self.assertEqual(dtypes.is_weakly_typed(fjit(val)), expected_weak_type)

  @jtu.sample_product(
    shape=nonempty_nonscalar_array_shapes,
    dtype=[int, float, complex],
    weak_type=[True, False],
    slc=[slice(None), slice(0), slice(3), 0, ...],
  )
  def testSliceWeakTypes(self, shape, dtype, weak_type, slc):
    rng = jtu.rand_default(self.rng())
    x = lax_internal._convert_element_type(rng(shape, dtype),
                                           weak_type=weak_type)
    op = lambda x: x[slc]
    self.assertEqual(op(x).aval.weak_type, weak_type)
    self.assertEqual(jax.jit(op)(x).aval.weak_type, weak_type)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis, num_sections=num_sections)
      for shape, axis, num_sections in [
          ((3,), 0, 3), ((12,), 0, 3), ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), -1, 2), ((2, 3, 4), -2, 3)]
    ],
    dtype=default_dtypes,
  )
  def testSplitStaticInt(self, shape, num_sections, axis, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.split(x, num_sections, axis=axis)
    jnp_fun = lambda x: jnp.split(x, num_sections, axis=axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis, num_sections=num_sections)
      # All testcases split the specified axis unequally
      for shape, axis, num_sections in [
          ((3,), 0, 2), ((12,), 0, 5), ((12, 4), 0, 7), ((12, 4), 1, 3),
          ((2, 3, 5), -1, 2), ((2, 4, 4), -2, 3), ((7, 2, 2), 0, 3)]
    ],
    dtype=default_dtypes,
  )
  def testArraySplitStaticInt(self, shape, num_sections, axis, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.array_split(x, num_sections, axis=axis)
    jnp_fun = lambda x: jnp.array_split(x, num_sections, axis=axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(op=op, shape=shape, axis=axis, indices=indices)
      # Includes negative and out-of-bound indices, which numpy resolves
      # relative to the axis size and then clips to [0, size].
      for op, shape, axis, indices in [
          ('split', (3,), 0, [-1]), ('split', (12,), 0, [2, -1]),
          ('split', (12, 4), 0, [2, -2]), ('split', (12, 4), 1, [-3, -1]),
          ('split', (2, 3, 4), -1, [1, -1]), ('split', (2, 3, 4), -2, [-2]),
          ('split', (3,), 0, [-5]), ('split', (3,), 0, [5]),
          ('split', (3,), 0, [-5, 5]), ('split', (12,), 0, [0, 12]),
          ('split', (12,), 0, [20, 30]),
          ('array_split', (12,), 0, [5, -2]),
          ('array_split', (2, 3, 5), -1, [1, -1]),
          ('array_split', (7, 2, 2), 0, [-5]),
          ('vsplit', (12, 4), 0, [1, -2]), ('hsplit', (12, 4), 1, [-1]),
          ('dsplit', (2, 3, 4), 2, [-2]), ('vsplit', (4, 3, 4), 0, [-1, 5])]
    ],
    dtype=default_dtypes,
  )
  def testSplitIndices(self, op, shape, indices, axis, dtype):
    rng = jtu.rand_default(self.rng())
    # vsplit, hsplit and dsplit split along a fixed axis, so take no axis arg.
    kwds = {} if op in ['vsplit', 'hsplit', 'dsplit'] else {'axis': axis}
    np_fun = lambda x: getattr(np, op)(x, indices, **kwds)
    jnp_fun = lambda x: getattr(jnp, op)(x, indices, **kwds)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testSplitTypeError(self):
    # If we pass an ndarray for indices_or_sections -> no error
    self.assertEqual(3, len(jnp.split(jnp.zeros(3), jnp.array([1, 2]))))

    CONCRETIZATION_MSG = "Abstract tracer value encountered where concrete value is expected."
    with self.assertRaisesRegex(TypeError, CONCRETIZATION_MSG):
      # An abstract tracer for idx
      jax.jit(lambda idx: jnp.split(jnp.zeros((12, 2)), idx))(2.)
    with self.assertRaisesRegex(TypeError, CONCRETIZATION_MSG):
      # A list including an abstract tracer
      jax.jit(lambda idx: jnp.split(jnp.zeros((12, 2)), [2, idx]))(2.)

    # A concrete tracer -> no error
    jax.jvp(lambda idx: jnp.split(jnp.zeros((12, 2)), idx),
            (2.,), (1.,))
    # A tuple including a concrete tracer -> no error
    jax.jvp(lambda idx: jnp.split(jnp.zeros((12, 2)), (1, idx.astype(np.int32))),
            (2.,), (1.,))

  @jtu.sample_product(
    shape=[(5,), (5, 5)],
    dtype=number_dtypes,
    bins=[10, np.arange(-5, 6), np.array([-5, 0, 3])],
    range=[None, (0, 0), (0, 10)],
    weights=[True, False],
  )
  def testHistogramBinEdges(self, shape, dtype, bins, range, weights):
    rng = jtu.rand_default(self.rng())
    _weights = lambda w: abs(w) if weights else None
    np_fun = lambda a, w, r: np.histogram_bin_edges(a, bins=bins, range=r,
                                                    weights=_weights(w))
    jnp_fun = lambda a, w, r: jnp.histogram_bin_edges(a, bins=bins, range=r,
                                                      weights=_weights(w))
    args_maker = lambda: [rng(shape, dtype), rng(shape, dtype), range]
    tol = {jnp.bfloat16: 2E-2, np.float16: 1E-2}
    # linspace() compares poorly to numpy when using bfloat16
    if dtype != jnp.bfloat16:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker,
                          atol=tol, rtol=tol)

  @jtu.sample_product(
    shape=[(5,), (4, 5)],
    dtype=default_dtypes,
    # We only test explicit integer-valued bin edges because in other cases
    # rounding errors lead to flaky tests.
    bins=[np.arange(-5, 6), np.array([-5, 0, 3])],
    density=[True, False],
    weights=[True, False],
  )
  def testHistogram(self, shape, dtype, bins, density, weights):
    rng = jtu.rand_default(self.rng())
    _weights = lambda w: abs(w) if weights else None
    def np_fun(a, w):
      # Numpy can't handle bfloat16
      a = a.astype('float32') if a.dtype == jnp.bfloat16 else a
      w = w.astype('float32') if w.dtype == jnp.bfloat16 else w
      return np.histogram(a, bins=bins, density=density, weights=_weights(w))
    jnp_fun = lambda a, w: jnp.histogram(a, bins=bins, density=density,
                                         weights=_weights(w))
    args_maker = lambda: [rng(shape, dtype), rng(shape, dtype)]
    tol = {jnp.bfloat16: 2E-2, np.float16: 1E-1}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(bins=[1, 3, 10])
  def testHistogramEmptyInput(self, bins):
    # The range cannot be inferred from an empty array; NumPy falls back to
    # (0, 1) rather than failing inside min/max.
    np_fun = lambda a: np.histogram(a, bins=bins)
    jnp_fun = lambda a: jnp.histogram(a, bins=bins)
    args_maker = lambda: [jnp.array([], dtype='float32')]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(bins=[1, 3, 10])
  def testHistogramBinEdgesEmptyInput(self, bins):
    np_fun = lambda a: np.histogram_bin_edges(a, bins=bins)
    jnp_fun = lambda a: jnp.histogram_bin_edges(a, bins=bins)
    args_maker = lambda: [jnp.array([], dtype='float32')]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=[(5,), (12,)],
    dtype=int_dtypes,
    bins=[2, [2, 2], [np.array([0, 1, 3, 5]), np.array([0, 2, 3, 4, 6])]],
    weights=[False, True],
    density=[False, True],
    range=[None, [(-1, 1), None], [(-1, 1), (-2, 2)]],
  )
  def testHistogram2d(self, shape, dtype, bins, weights, density, range):
    rng = jtu.rand_default(self.rng())
    _weights = lambda w: abs(w) if weights else None
    np_fun = jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")(
        lambda a, b, w: np.histogram2d(a, b, bins=bins, weights=_weights(w), density=density, range=range))
    jnp_fun = lambda a, b, w: jnp.histogram2d(a, b, bins=bins, weights=_weights(w), density=density, range=range)
    args_maker = lambda: [rng(shape, dtype), rng(shape, dtype), rng(shape, dtype)]
    tol = {jnp.bfloat16: 2E-2, np.float16: 1E-1}
    # np.searchsorted errors on bfloat16 with
    # "TypeError: invalid type promotion with custom data type"
    with np.errstate(divide='ignore', invalid='ignore'):
      if dtype != jnp.bfloat16:
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                          tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=[(5, 3), (10, 3)],
    dtype=int_dtypes,
    bins=[(2, 2, 2), [np.array([-5, 0, 4]), np.array([-4, -1, 2]), np.array([-6, -1, 4])]],
    weights=[False, True],
    density=[False, True],
    range=[None, [(-1, 1), None, None], [(-1, 1), (-2, 2), (-3, 3)]],
  )
  def testHistogramdd(self, shape, dtype, bins, weights, density, range):
    rng = jtu.rand_default(self.rng())
    _weights = lambda w: abs(w) if weights else None
    np_fun = jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")(
        lambda a, w: np.histogramdd(a, bins=bins, weights=_weights(w), density=density, range=range))
    jnp_fun = lambda a, w: jnp.histogramdd(a, bins=bins, weights=_weights(w), density=density, range=range)
    args_maker = lambda: [rng(shape, dtype), rng((shape[0],), dtype)]
    tol = {jnp.bfloat16: 2E-2, np.float16: 1E-1}
    # np.searchsorted errors on bfloat16 with
    # "TypeError: invalid type promotion with custom data type"
    if dtype != jnp.bfloat16:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis, num_sections=num_sections)
      for shape, axis, num_sections in [
          ((12, 4), 0, 4), ((12,), 1, 2),
          ((2, 3, 4), 2, 2), ((4, 3, 4), 0, 2)]],
    dtype=default_dtypes,
  )
  def testHVDSplit(self, shape, num_sections, axis, dtype):
    rng = jtu.rand_default(self.rng())
    def fn(module, axis):
      if axis == 0:
        return module.vsplit
      elif axis == 1:
        return module.hsplit
      else:
        assert axis == 2
        return module.dsplit

    np_fun = lambda x: fn(np, axis)(x, num_sections)
    jnp_fun = lambda x: fn(jnp, axis)(x, num_sections)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, out_shape=out_shape)
      for arg_shape, out_shape in [
          (jtu.NUMPY_SCALAR_SHAPE, (1, 1, 1)),
          ((), (1, 1, 1)),
          ((7, 0), (0, 42, 101)),
          ((3, 4), 12),
          ((3, 4), (12,)),
          ((3, 4), -1),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]
    ],
    dtype=default_dtypes,
    order=["C", "F"],
  )
  def testReshape(self, arg_shape, out_shape, dtype, order):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.reshape(x, out_shape, order=order)
    jnp_fun = lambda x: jnp.reshape(x, out_shape, order=order)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, out_shape=out_shape)
      for arg_shape, out_shape in [
          ((7, 0), (0, 42, 101)),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]
    ],
    dtype=default_dtypes,
  )
  def testReshapeMethod(self, arg_shape, out_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.reshape(x, out_shape)
    jnp_fun = lambda x: x.reshape(*out_shape)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, out_shape=out_shape)
      for arg_shape, out_shape in itertools.product(all_shapes, array_shapes)],
    dtype=default_dtypes,
  )
  def testResize(self, arg_shape, out_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.resize(x, out_shape)
    jnp_fun = lambda x: jnp.resize(x, out_shape)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, dim=dim)
      for arg_shape in [(), (3,), (3, 4)]
      for dim in (list(range(-len(arg_shape)+1, len(arg_shape)))
                  + [np.array(0), np.array(-1), (0,), [np.array(0)],
                     (len(arg_shape), len(arg_shape) + 1)])
    ],
    dtype=default_dtypes,
  )
  def testExpandDimsStaticDim(self, arg_shape, dtype, dim):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.expand_dims(x, dim)
    jnp_fun = lambda x: jnp.expand_dims(x, dim)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CompileAndCheck(jnp_fun, args_maker)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  def testExpandDimsRepeatedAxisError(self):
    x = jnp.ones((2, 3))
    self.assertRaisesRegex(
        ValueError, 'repeated axis.*',
        lambda: jnp.expand_dims(x, [1, 1]))
    self.assertRaisesRegex(
        ValueError, 'repeated axis.*',
        lambda: jnp.expand_dims(x, [3, -1]))

    # ensure this is numpy's behavior too, so that we remain consistent
    x = np.ones((2, 3))
    self.assertRaisesRegex(
        ValueError, 'repeated axis.*',
        lambda: np.expand_dims(x, [1, 1]))
    self.assertRaisesRegex(
        ValueError, 'repeated axis.*',
        lambda: np.expand_dims(x, [3, -1]))

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, ax1=ax1, ax2=ax2)
      for arg_shape, ax1, ax2 in [
          ((3, 4), 0, 1), ((3, 4), 1, 0), ((3, 4, 5), 1, 2),
          ((3, 4, 5), -1, -2), ((3, 4, 5), 0, 1)]
    ],
    dtype=default_dtypes,
  )
  def testSwapAxesStaticAxes(self, arg_shape, dtype, ax1, ax2):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.swapaxes(x, ax1, ax2)
    jnp_fun = lambda x: jnp.swapaxes(x, ax1, ax2)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, ax=ax)
      for arg_shape, ax in [
          ((3, 1), None),
          ((3, 1), 1),
          ((3, 1), -1),
          ((3, 1), np.array(1)),
          ((1, 3, 1), (0, 2)),
          ((1, 3, 1), (0,)),
          ((1, 4, 1), (np.array(0),))]
    ],
    dtype=default_dtypes,
  )
  def testSqueeze(self, arg_shape, dtype, ax):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.squeeze(x, ax)
    jnp_fun = lambda x: jnp.squeeze(x, ax)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testArrayFromMasked(self):
    args_maker = lambda: [np.ma.array([1, 2], mask=[True, False])]
    # Like np.array, jnp.array strips the mask from masked array inputs.
    self._CheckAgainstNumpy(np.array, jnp.array, args_maker)
    # Under JIT, masked arrays are flagged as invalid.
    with self.assertRaisesRegex(ValueError, "numpy masked arrays are not supported"):
      jax.jit(jnp.asarray)(*args_maker())

  @jtu.sample_product(
    [dict(arg=arg, dtype=dtype, ndmin=ndmin)
      for arg, dtypes in [
          ([True, False, True], all_dtypes),
          (3., all_dtypes),
          ([1, 2, 3], all_dtypes),
          (np.array([1, 2, 3], dtype=np.int64), all_dtypes),
          ([1., 2., 3.], all_dtypes),
          ([[1, 2], [3, 4], [5, 6]], all_dtypes),
          ([[1, 2.], [3, 4], [5, 6]], all_dtypes),
          ([[1., 2j], [3., 4.], [5., 6.]], complex_dtypes),
          ([[3, np.array(2, dtype=jnp.float_), 1],
           np.arange(3., dtype=jnp.float_)], all_dtypes),
      ]
      for dtype in [None] + dtypes
      for ndmin in [None, np.ndim(arg), np.ndim(arg) + 1, np.ndim(arg) + 2]
    ],
  )
  def testArray(self, arg, ndmin, dtype):
    args_maker = lambda: [arg]
    canonical_dtype = dtypes.canonicalize_dtype(dtype or np.array(arg).dtype)
    if ndmin is not None:
      np_fun = partial(np.array, ndmin=ndmin, dtype=canonical_dtype)
      jnp_fun = partial(jnp.array, ndmin=ndmin, dtype=dtype)
    else:
      np_fun = partial(np.array, dtype=canonical_dtype)
      jnp_fun = partial(jnp.array, dtype=dtype)

    # We are testing correct canonicalization behavior here, so we turn off the
    # permissive canonicalization logic in the test harness.
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            canonicalize_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testArrayPositionalDeprecation(self):
    x = jnp.array([1, 2, 3])

    with self.assertDeprecationWarnsOrRaises(
        'jax-array-positional-args',
        'Passing the copy, order, and ndmin arguments to jnp.array',
    ):
      out = jnp.array(x, None, False)
    self.assertIs(x, out)

    with self.assertDeprecationWarnsOrRaises(
        'jax-array-positional-args',
        'Passing the copy, order, and ndmin arguments to jnp.array',
    ):
      out = jnp.array(x, None, True)
    self.assertIsNot(x, out)

    with self.assertRaisesRegex(TypeError, "array\\(\\) got multiple values for argument 'copy'"):
      jnp.array(x, None, True, copy=False)

    with self.assertRaisesRegex(TypeError, "array\\(\\) takes at most 5 positional arguments"):
      jnp.array(x, None, True, "K", 0, "extra")

  @jtu.sample_product(copy=[None, True, False])
  def testAsarrayCopy(self, copy):
    x_jax = jnp.arange(4)
    x_np = np.arange(4)
    x_list = [0, 1, 2, 3]
    x_buf = make_python_array('l', x_list)

    func = partial(jnp.asarray, copy=copy)
    self.assertArraysEqual(x_jax, func(x_jax))
    self.assertArraysEqual(x_jax, func(x_list), check_dtypes=False)

    if copy is False and jax.default_backend() != 'cpu':
      # copy=False is strict: it must raise if the input supports the buffer protocol
      # but a copy is still required.
      self.assertRaises(ValueError, func, x_np)
      self.assertRaises(ValueError, func, x_buf)
    else:
      self.assertArraysEqual(x_jax, func(x_np), check_dtypes=False)
      self.assertArraysEqual(x_jax, func(x_buf), check_dtypes=False)

  @jtu.sample_product(numpy_array=[True, False])
  def testAsarrayWithCopyFalse(self, numpy_array):
    x_jax = jnp.arange(4)
    if numpy_array:
      x = np.arange(4)
    else:
      x = make_python_array('l', [0, 1, 2, 3])
    device_error_msg = ('jnp.asarray: cannot convert object of type .* to JAX'
                        ' Array on platform={} with copy=False. Consider using'
                        ' copy=None or copy=True instead.')

    if jax.default_backend() != 'cpu':
      # test accelerator devices - no support for copy=False
      expected_platform = jax.local_devices()[0].platform
      with self.assertRaisesRegex(
          ValueError, device_error_msg.format(expected_platform)):
        jnp.asarray(x, copy=False, device=jax.local_devices()[0])
      sharding = make_single_device_sharding(jax.local_devices()[0])
      with self.assertRaisesRegex(
          ValueError, device_error_msg.format(expected_platform)):
        jnp.asarray(x, copy=False, device=sharding)

      # test None defaults to default backend - no support for copy=False
      with self.assertRaisesRegex(
          ValueError, device_error_msg.format(expected_platform)):
        jnp.asarray(x, copy=False, device=None)
    else:
      self.assertArraysEqual(jnp.asarray(x, copy=False, device=None), x_jax,
                             check_dtypes=False)

    # test explicit CPU device or default CPU device context managers overwrite the default backend
    x = make_python_array('l', [0, 1, 2, 3])
    for device in [jax.local_devices(backend='cpu')[0],
                   make_single_device_sharding(
                       jax.local_devices(backend='cpu')[0])]:
      self.assertArraysEqual(jnp.asarray(x, copy=False, device=device),
                             x_jax, check_dtypes=False)
    with jax.default_device('cpu'):
      self.assertArraysEqual(jnp.asarray(x, copy=False), x_jax,
                             check_dtypes=False)
      self.assertArraysEqual(jnp.asarray(x, copy=False, device=None), x_jax,
                             check_dtypes=False)
    with jax.default_device(jax.local_devices(backend='cpu')[0]):
      self.assertArraysEqual(jnp.asarray(x, copy=False), x_jax,
                             check_dtypes=False)
      self.assertArraysEqual(jnp.asarray(x, copy=False, device=None), x_jax,
                             check_dtypes=False)

  @jtu.ignore_warning(category=UserWarning, message="Explicitly requested dtype.*")
  def testArrayDtypeInference(self):
    def _check(obj, out_dtype, weak_type):
      dtype_reference = np.array(obj, dtype=out_dtype)

      out = jnp.array(obj)
      self.assertDtypesMatch(out, dtype_reference)
      self.assertEqual(dtypes.is_weakly_typed(out), weak_type)

      out_jit = jax.jit(jnp.array)(obj)
      self.assertDtypesMatch(out_jit, dtype_reference)
      self.assertEqual(dtypes.is_weakly_typed(out_jit), weak_type)

    # Python scalars become 64-bit weak types.
    _check(1, np.int64, True)
    _check(1.0, np.float64, True)
    _check(1.0j, np.complex128, True)

    # Lists become strongly-typed defaults.
    _check([1], jnp.int64, False)
    _check([1.0], jnp.float64, False)
    _check([1.0j], jnp.complex128, False)

    # Lists of weakly-typed objects become strongly-typed defaults.
    _check([jnp.array(1)], jnp.int64, False)
    _check([jnp.array(1.0)], jnp.float64, False)
    _check([jnp.array(1.0j)], jnp.complex128, False)

    # Lists of strongly-typed objects maintain their strong type.
    if config.explicit_x64_dtypes.value != config.ExplicitX64Mode.ERROR:
      _check([jnp.int64(1)], np.int64, False)
      _check([jnp.float64(1)], np.float64, False)
      _check([jnp.complex128(1)], np.complex128, False)

    # Mixed inputs use JAX-style promotion.
    # (regression test for https://github.com/jax-ml/jax/issues/8945)
    _check([0, np.int16(1)], np.int16, False)
    _check([0.0, np.float16(1)], np.float16, False)

  @jtu.sample_product(
    dtype=all_dtypes,
    func=["array", "copy", "copy.copy", "copy.deepcopy"],
  )
  def testArrayCopy(self, dtype, func):
    x = jnp.ones(10, dtype=dtype)
    if func == "copy.deepcopy":
      copy_func = copy.deepcopy
    elif func == "copy.copy":
      copy_func = copy.copy
    else:
      copy_func = getattr(jnp, func)

    x_view = jnp.asarray(x)
    x_view_jit = jax.jit(jnp.asarray)(x)
    x_copy = copy_func(x)
    x_copy_jit = jax.jit(copy_func)(x)

    _ptr = lambda x: x.unsafe_buffer_pointer()

    self.assertEqual(_ptr(x), _ptr(x_view))
    self.assertNotEqual(_ptr(x), _ptr(x_view_jit))
    self.assertNotEqual(_ptr(x), _ptr(x_copy))
    self.assertNotEqual(_ptr(x), _ptr(x_copy_jit))

    x.delete()

    self.assertTrue(x_view.is_deleted())
    self.assertFalse(x_view_jit.is_deleted())

    self.assertFalse(x_copy.is_deleted())
    self.assertFalse(x_copy_jit.is_deleted())

  def testArrayCopyAutodiff(self):
    f = lambda x: jnp.array(x, copy=True)

    x = jnp.ones(10)
    xdot = jnp.ones(10)
    y, ydot = jax.jvp(f, (x,), (xdot,))
    self.assertIsNot(x, y)
    self.assertIsNot(xdot, ydot)

    ybar = jnp.ones(10)
    y, f_vjp = jax.vjp(f, x)
    xbar, = f_vjp(ybar)
    self.assertIsNot(x, y)
    self.assertIsNot(xbar, ybar)

  def testArrayCopyVmap(self):
    f = lambda x: jnp.array(x, copy=True)
    x = jnp.ones(10)
    y = jax.vmap(f)(x)
    self.assertIsNot(x, y)

  def testArrayContains(self):
    self.assertIn(1, jnp.arange(4))
    self.assertNotIn(100, jnp.arange(4))

    with self.assertRaisesRegex(
        ValueError, "Array.__contains__: search array must be one-dimensional"):
      _ = 1 in jnp.array(1)

    with self.assertRaisesRegex(
        ValueError, "Array.__contains__: query value must be a scalar"):
      _ = jnp.arange(2) in jnp.arange(2)

    with self.assertRaisesRegex(
        TypeError, "Array.__contains__: unsupported operand type.*"
    ):
      _ = "abc" in jnp.arange(2)

    with self.assertRaisesRegex(
        TypeError, "Array.__contains__: unsupported operand type.*"
    ):
      _ = None in jnp.arange(2)

  def testArrayUnsupportedDtypeError(self):
    with self.assertRaisesRegex(
        TypeError, 'JAX only supports number, bool, and string dtypes.*'
    ):
      jnp.array(3, [('a','<i4'),('b','<i4')])

  def testArrayFromInteger(self):
    int_dtype = dtypes.default_int_dtype()
    int_max = jnp.iinfo(int_dtype).max
    int_min = jnp.iinfo(int_dtype).min

    # Values at extremes are converted correctly.
    for val in [int_min, 0, int_max]:
      self.assertEqual(jnp.array(val).dtype, int_dtype)

    # out of bounds leads to an OverflowError
    val = int_max + 1
    with self.assertRaisesRegex(OverflowError, f"Python int {val} too large to convert to {int_dtype.name}"):
      jnp.array(val)

    # explicit uint64 should work
    if config.enable_x64.value:
      self.assertEqual(np.uint64(val), jnp.array(val, dtype='uint64'))

  def testArrayFromList(self):
    dtype = dtypes.canonicalize_dtype('int64')
    int_max = jnp.iinfo(dtype).max
    int_min = jnp.iinfo(dtype).min

    # Values at extremes are converted correctly.
    for val in [int_min, 0, int_max]:
      self.assertEqual(jnp.array([val]).dtype, dtype)

    # list of values results in promoted type.
    with jax.numpy_dtype_promotion('standard'):
      self.assertEqual(jnp.array([0, np.float16(1)]).dtype, jnp.result_type('int32', 'float16'))

    # out of bounds leads to an OverflowError
    val = jnp.iinfo(jnp.int64).min - 1
    expected_regex = '(Python int too large.*)|(int too big to convert)'
    with self.assertRaisesRegex(OverflowError, expected_regex):
      jnp.array([0, val])

  def testArrayNone(self):
    with self.assertRaisesRegex(
        ValueError, 'None is not a valid value for jnp.array'
    ):
      jnp.array([0.0, None])

  def testIssue121(self):
    assert not np.isscalar(jnp.array(3))

  @config.explicit_x64_dtypes('allow')
  def testArrayExplicitDtypes(self):
    self.assertEqual(jnp.array(1, dtype=jnp.int64).dtype, jnp.int64)
    self.assertEqual(jnp.array(1.0, dtype=jnp.float64).dtype, jnp.float64)
    if not jtu.test_device_matches(['tpu']):
      self.assertEqual(jnp.array(1j, dtype=jnp.complex128).dtype, jnp.complex128)

  def testArrayOutputsArrays(self):
    assert type(jnp.array([])) is array.ArrayImpl
    assert type(jnp.array(np.array([]))) is array.ArrayImpl

    class NDArrayLike:
      def __array__(self, dtype=None, copy=None):
        return np.array([], dtype=dtype)
    assert type(jnp.array(NDArrayLike())) is array.ArrayImpl

    # NOTE(mattjj): disabled b/c __array__ must produce ndarrays
    # class ArrayLike:
    #     def __array__(self, dtype=None):
    #         return jnp.array([], dtype=dtype)
    # assert  xla.type_is_device_array(jnp.array(ArrayLike()))

  def testArrayMethod(self):
    class arraylike:
      dtype = np.dtype('float32')
      def __array__(self, dtype=None, copy=None):
        return np.array(3., dtype=dtype)
    a = arraylike()
    ans = jnp.array(a)
    self.assertEqual(ans, 3.)

  def testJaxArrayOps(self):
    class arraylike:
      def __jax_array__(self):
        return jnp.array(3.)
    self.assertArraysEqual(arraylike() * jnp.arange(10.), jnp.array(3.) * jnp.arange(10.))

  def testMemoryView(self):
    self.assertAllClose(
        jnp.array(bytearray(b'\x2a')),
        np.array(bytearray(b'\x2a'))
    )
    self.assertAllClose(
        jnp.array(bytearray(b'\x2a\xf3'), ndmin=2),
        np.array(bytearray(b'\x2a\xf3'), ndmin=2)
    )

  @jtu.sample_product(value=[False, 1, 1.0, np.int32(5), np.array(16)])
  def testIsScalar(self, value):
    self.assertTrue(jnp.isscalar(value))

  @jtu.sample_product(value=[None, [1], slice(4), (), np.array([0])])
  def testIsNotScalar(self, value):
    self.assertFalse(jnp.isscalar(value))

  @jtu.sample_product(val=[1+1j, [1+1j], jnp.pi, np.arange(2)])
  def testIsComplexObj(self, val):
    args_maker = lambda: [val]
    self._CheckAgainstNumpy(np.iscomplexobj, jnp.iscomplexobj, args_maker)
    self._CompileAndCheck(jnp.iscomplexobj, args_maker)

  @parameterized.parameters(
      None, bool(1), int(1), float(1), complex(1),
      np.int32(0), np.float32(1), np.complex64(1),
      (np.arange(5),)
  )
  def testIsComplexObjTransferGuard(self, val):
    with jax.transfer_guard("disallow"):
      jnp.iscomplexobj(val)

  def testIsClose(self):
    c_isclose = jax.jit(jnp.isclose)
    c_isclose_nan = jax.jit(partial(jnp.isclose, equal_nan=True))
    n = 2

    rng = self.rng()
    x = rng.randn(n, 1)
    y = rng.randn(n, 1)
    inf = np.asarray(n * [np.inf]).reshape([n, 1])
    nan = np.asarray(n * [np.nan]).reshape([n, 1])
    args = [x, y, inf, -inf, nan]

    for arg0 in args:
      for arg1 in args:
        result_np = np.isclose(arg0, arg1)
        result_jax = jnp.isclose(arg0, arg1)
        result_jit = c_isclose(arg0, arg1)
        self.assertTrue(jnp.all(jnp.equal(result_np, result_jax)))
        self.assertTrue(jnp.all(jnp.equal(result_np, result_jit)))
        result_np = np.isclose(arg0, arg1, equal_nan=True)
        result_jax = jnp.isclose(arg0, arg1, equal_nan=True)
        result_jit = c_isclose_nan(arg0, arg1)
        self.assertTrue(jnp.all(jnp.equal(result_np, result_jax)))
        self.assertTrue(jnp.all(jnp.equal(result_np, result_jit)))

    self.assertEqual(np.isclose(6, 10, rtol=0.5), jnp.isclose(6, 10, rtol=0.5))
    key = jax.random.key(0)
    self.assertTrue(jnp.isclose(key, key))

  @jtu.sample_product(
    atol=[0.0, 1E-4, np.inf],
    rtol=[0.0, 1E-4, np.inf],
    equal_nan=[True, False]
  )
  def testIsCloseCornerCases(self, atol, rtol, equal_nan):
    if jtu.numpy_version() >= (2, 3, 0) and (np.isinf(atol) or np.isinf(rtol)):
      self.skipTest("NumPy 2.3.0 now throws warnings for inf atol/rtol")
    vals = np.array([-np.nan, -np.inf, -1.00001, -1.0, -0.00001, -0.0,
                     0.0, 0.00001, 1.0, 1.00001, np.inf, np.nan])
    x, y = np.meshgrid(vals, vals)
    self.assertArraysEqual(
      np.isclose(x, y, atol=atol, rtol=rtol, equal_nan=equal_nan),
      jnp.isclose(x, y, atol=atol, rtol=rtol, equal_nan=equal_nan)
    )

  @jtu.sample_product(
    x=[1, [1], [1, 1 + 1E-4], [1, np.nan]],
    y=[1, [1], [1, 1 + 1E-4], [1, np.nan]],
    equal_nan=[True, False],
  )
  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises mixed type promotion
  def testAllClose(self, x, y, equal_nan):
    jnp_fun = partial(jnp.allclose, equal_nan=equal_nan, rtol=1E-3)
    np_fun = partial(np.allclose, equal_nan=equal_nan, rtol=1E-3)
    args_maker = lambda: [np.array(x), np.array(y)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testZeroStridesConstantHandler(self):
    raw_const = self.rng().randn(1, 2, 1, 1, 5, 1)
    const = np.broadcast_to(raw_const, (3, 2, 3, 4, 5, 6))

    def fun(x):
      return x * const

    fun = jax.jit(fun)
    out_val = fun(3.)
    self.assertAllClose(out_val, 3. * const, check_dtypes=False)

  def testIsInstanceNdarrayDuringTracing(self):
    arr = np.ones(3)

    @jax.jit
    def f(x):
      self.assertIsInstance(x, jax.Array)
      return jnp.sum(x)

    f(arr)

  def testNonArrayErrorMessage(self):
    x = [1., 2.]
    y = np.array([3., 4.])

    def g(x, y):
      return jnp.add(x, y)

    def f(x, y):
      return jnp.dot(x, y)

    self.assertRaises(TypeError, lambda: g(x, y))
    self.assertRaises(TypeError, lambda: f(x, y))
    self.assertRaises(TypeError, lambda: jax.jit(g)(x, y))
    self.assertRaises(TypeError, lambda: jax.jit(f)(x, y))

  def testAbstractionErrorMessage(self):

    @jax.jit
    def f(x, n):
      for _ in range(n):
        x = x * x
      return x

    self.assertRaises(jax.errors.TracerIntegerConversionError, lambda: f(3., 3))

    @jax.jit
    def g(x):
      if x > 0.:
        return x * 2
      else:
        return x + 2

    self.assertRaises(jax.errors.ConcretizationTypeError, lambda: g(3.))

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in [(3,), (2, 3)]
      for axis in list(range(-len(shape), len(shape))) + [None] + [tuple(range(len(shape)))]  # Test negative axes and tuples
    ],
    dtype=default_dtypes,
  )
  def testFlip(self, shape, dtype, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.flip(x, axis)
    np_op = lambda x: np.flip(x, axis)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    shape=[(3,), (2, 3), (3, 2, 4)],
    dtype=default_dtypes,
  )
  def testFlipud(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.flipud(x)
    np_op = lambda x: np.flipud(x)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    shape=[(3, 2), (2, 3), (3, 2, 4)],
    dtype=default_dtypes,
  )
  def testFliplr(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.fliplr(x)
    np_op = lambda x: np.fliplr(x)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axes=axes)
      for shape, axes in [
          [(2, 3), (0, 1)],
          [(2, 3), (1, 0)],
          [(4, 3, 2), (0, 2)],
          [(4, 3, 2), (2, 1)],
      ]
    ],
    k=range(-3, 4),
    dtype=default_dtypes,
  )
  def testRot90(self, shape, dtype, k, axes):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.rot90(x, k, axes)
    np_op = lambda x: np.rot90(x, k, axes)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  # TODO(mattjj): test infix operator overrides

  def testRavel(self):
    rng = self.rng()
    args_maker = lambda: [rng.randn(3, 4).astype("float32")]
    self._CompileAndCheck(lambda x: x.ravel(), args_maker)

  @jtu.sample_product(
    shape=nonempty_nonscalar_array_shapes,
    order=['C', 'F'],
    mode=['wrap', 'clip', 'raise'],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testRavelMultiIndex(self, shape, order, mode):
    # generate indices in each dimension with a few out of bounds.
    rngs = [jtu.rand_int(self.rng(), low=-1, high=dim + 1)
            for dim in shape]
    # generate multi_indices of different dimensions that broadcast.
    args_maker = lambda: [tuple(rng(ndim * (3,), jnp.int_)
                                for ndim, rng in enumerate(rngs))]
    def np_fun(x):
      try:
        return np.ravel_multi_index(x, shape, order=order, mode=mode)
      except ValueError as err:
        if str(err).startswith('invalid entry'):
          # sentinel indicating expected error.
          return -999
        else:
          raise
    def jnp_fun(x):
      try:
        return jnp.ravel_multi_index(x, shape, order=order, mode=mode)
      except ValueError as err:
        if str(err).startswith('invalid entry'):
          # sentinel indicating expected error.
          return -999
        else:
          raise
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    if mode == 'raise':
      msg = ("The error occurred because ravel_multi_index was jit-compiled "
             "with mode='raise'. Use mode='wrap' or mode='clip' instead.")
      with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
        jax.jit(jnp_fun)(*args_maker())
    else:
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    in_dtype=[jnp.int8, jnp.int16, jnp.int32, jnp.uint8, jnp.uint16],
    out_dtype=[None, jnp.int32],
    use_jit=[True, False],
  )
  def testRavelMultiIndexDtype(self, in_dtype, out_dtype, use_jit):
    dims = (2, 3)
    idx = jnp.array([0, 1], dtype=in_dtype)
    indices = (idx, idx)
    ravel_multi_index = partial(jnp.ravel_multi_index, dims=dims, mode='ignore', dtype=out_dtype)
    if use_jit:
      ravel_multi_index = jax.jit(ravel_multi_index)

    expected_dtype = out_dtype or in_dtype
    res = ravel_multi_index(indices)
    self.assertEqual(res.dtype, expected_dtype)
    self.assertFalse(res.weak_type)

  @jtu.sample_product(
    ashape=((), (4,), (3, 4)),
    cshapes=[
      [(), (4,)],
      [(3, 4), (4,), (3, 1)]
    ],
    adtype=int_dtypes,
    cdtype=default_dtypes,
    mode=['wrap', 'clip', 'raise'],
  )
  def testChoose(self, ashape, adtype, cshapes, cdtype, mode):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(ashape, adtype), [rng(s, cdtype) for s in cshapes]]
    def np_fun(a, c):
      try:
        return np.choose(a, c, mode=mode)
      except ValueError as err:
        if mode == 'raise' and str(err).startswith('invalid entry'):
          return -999  # sentinel indicating expected error.
        else:
          raise
    def jnp_fun(a, c):
      try:
        return jnp.choose(a, c, mode=mode)
      except ValueError as err:
        if mode == 'raise' and str(err).startswith('invalid entry'):
          return -999  # sentinel indicating expected error.
        else:
          raise
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    if mode == 'raise':
      msg = ("The error occurred because jnp.choose was jit-compiled"
             " with mode='raise'. Use mode='wrap' or mode='clip' instead.")
      with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
        jax.jit(jnp_fun)(*args_maker())
    else:
      self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=nonempty_nonscalar_array_shapes,
    dtype=int_dtypes + unsigned_dtypes,
    idx_shape=all_shapes,
  )
  def testUnravelIndex(self, shape, idx_shape, dtype):
    if not jtu.is_optimized_build():
      self.skipTest("Test fails under debug mode. See https://github.com/numpy/numpy/issues/29690.")
    size = math.prod(shape)
    unsigned = dtypes.issubdtype(dtype, np.unsignedinteger)
    rng = jtu.rand_int(
      self.rng(), low=0 if unsigned else  -((2 * size) // 3), high=(2 * size) // 3)

    def np_fun(index, shape):
      # JAX's version outputs the same dtype as the input in the typical case
      # where shape is weakly-typed.
      out_dtype = index.dtype
      # Adjust out-of-bounds behavior to match jax's documented behavior.
      index = np.clip(index, 0 if unsigned else -size, size - 1)
      index = np.where(index < 0, index + size, index)
      return [i.astype(out_dtype) for i in np.unravel_index(index, shape)]

    jnp_fun = jnp.unravel_index
    args_maker = lambda: [rng(idx_shape, dtype), shape]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    from_dtype=['int32', 'float32'],
    to_dtype=['int32', 'float32', None],
    use_method=[True, False],
  )
  def testAstype(self, from_dtype, to_dtype, use_method):
    rng = self.rng()
    args_maker = lambda: [rng.randn(3, 4).astype(from_dtype)]
    if not use_method:
      np_op = lambda x: np.astype(x, to_dtype)
    else:
      np_op = lambda x: np.asarray(x).astype(to_dtype)
    if use_method:
      jnp_op = lambda x: jnp.asarray(x).astype(to_dtype)
    else:
      jnp_op = lambda x: jnp.astype(x, to_dtype)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    from_dtype=['int32', 'float32', 'complex64'],
    use_method=[True, False],
  )
  def testAstypeBool(self, from_dtype, use_method, to_dtype='bool'):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng((3, 4), from_dtype)]
    if not use_method:
      np_op = lambda x: np.astype(x, to_dtype)
    else:
      np_op = lambda x: np.asarray(x).astype(to_dtype)
    if use_method:
      jnp_op = lambda x: jnp.asarray(x).astype(to_dtype)
    else:
      jnp_op = lambda x: jnp.astype(x, to_dtype)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    change_dtype=[True, False],
    copy=[True, False],
  )
  def testAstypeCopy(self, change_dtype, copy):
    dtype = 'float32' if change_dtype else 'int32'
    expect_copy = change_dtype or copy
    x = jnp.arange(5, dtype='int32')
    y = x.astype(dtype, copy=copy)

    self.assertEqual(y.dtype, dtype)
    y.delete()
    self.assertNotEqual(x.is_deleted(), expect_copy)

  def testAstypeComplexDowncast(self):
    x = jnp.array(2.0+1.5j, dtype='complex64')
    with self.assertDeprecationWarnsOrRaises("jax-numpy-astype-complex-to-real",
                                             "Casting from complex to real dtypes.*"):
      x.astype('float32')

  @parameterized.parameters('int1', 'int2', 'int4')
  def testAstypeIntN(self, dtype):
    if dtype in ['int1', 'int2']:
      self.skipTest('XLA support for int1/int2 is incomplete.')

    # Test converting from intN to int8
    x = np.array([1, -2, -3, 4, -8, 7], dtype=dtype)
    args_maker = lambda: [x]
    np_op = lambda x: np.asarray(x).astype(jnp.int8)
    jnp_op = lambda x: jnp.asarray(x).astype(jnp.int8)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

    # Test converting from int8 to intN
    x = np.array([1, -2, -3, 4, -8, 7], dtype=jnp.int8)
    args_maker = lambda: [x]
    np_op = lambda x: np.asarray(x).astype(dtype)
    jnp_op = lambda x: jnp.asarray(x).astype(dtype)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    shape=array_shapes,
    dtype=all_dtypes,
  )
  def testByteSwap(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_op = lambda x: np.asarray(x).byteswap()
    jnp_op = lambda x: jnp.asarray(x).byteswap()
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    shape=array_shapes,
    dtype=all_dtypes,
  )
  def testNbytes(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_op = lambda x: np.asarray(x).nbytes
    jnp_op = lambda x: jnp.asarray(x).nbytes
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    shape=array_shapes,
    dtype=all_dtypes,
  )
  def testItemsize(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_op = lambda x: np.asarray(x).itemsize
    jnp_op = lambda x: jnp.asarray(x).itemsize
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    shape=nonempty_array_shapes,
    dtype=all_dtypes,
    num_args=[0, 1, "all"],
    use_tuple=[True, False]
  )
  def testItem(self, shape, dtype, num_args, use_tuple):
    rng = jtu.rand_default(self.rng())
    size = math.prod(shape)

    if num_args == 0:
      args = ()
    elif num_args == 1:
      args = (self.rng().randint(0, size),)
    else:
      args = tuple(self.rng().randint(0, s) for s in shape)
    args = (args,) if use_tuple else args

    np_op = lambda x: np.asarray(x).item(*args)
    jnp_op = lambda x: jnp.asarray(x).item(*args)
    args_maker = lambda: [rng(shape, dtype)]

    if size != 1 and num_args == 0:
      with self.assertRaises(ValueError):
        jnp_op(*args_maker())
    else:
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker)

  @jtu.sample_product(
    # Final dimension must be a multiple of 16 to ensure compatibility of all dtype pairs.
    shape=[(0,), (64,), (2, 32)],
    a_dtype=(jnp.int4, jnp.uint4, *all_dtypes),
    dtype=((jnp.int4, jnp.uint4, *all_dtypes, None)
           if config.enable_x64.value else (jnp.int4, jnp.uint4, *all_dtypes)),
  )
  def testView(self, shape, a_dtype, dtype):
    if jtu.test_device_matches(["tpu"]):
      if jnp.dtype(a_dtype).itemsize in [1, 2] or jnp.dtype(dtype).itemsize in [1, 2]:
        self.skipTest("arr.view() not supported on TPU for 8- or 16-bit types.")
    # It is possible to fill bool arrays with arbitrary bits (not just 0/1
    # bytes), but the behavior is implementation-defined. We therefore only test
    # the well-defined case.
    rng = (jtu.rand_bool if a_dtype == np.bool_ else jtu.rand_fullrange)(
        self.rng()
    )
    args_maker = lambda: [rng(shape, a_dtype)]
    np_op = lambda x: np_view(x, dtype)
    jnp_op = lambda x: jnp.asarray(x).view(dtype)
    # Above may produce signaling nans; ignore warnings from invalid values.
    with np.errstate(invalid='ignore'):
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
      self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product([
    {'a_dtype': a_dtype, 'dtype': dtype}
    for a_dtype in [jnp.int4, jnp.uint4, *all_dtypes]
    for dtype in [jnp.int4, jnp.uint4, *all_dtypes]
    if dtypes.itemsize_bits(a_dtype) == dtypes.itemsize_bits(dtype)
  ])
  def testViewScalar(self, a_dtype, dtype):
    if jtu.test_device_matches(["tpu"]):
      if jnp.dtype(a_dtype).itemsize in [1, 2] or jnp.dtype(dtype).itemsize in [1, 2]:
        self.skipTest("arr.view() not supported on TPU for 8- or 16-bit types.")
    rng = jtu.rand_fullrange(self.rng())
    args_maker = lambda: [jnp.array(rng((), a_dtype))]
    np_op = lambda x: np.asarray(x).view(dtype)
    jnp_op = lambda x: jnp.asarray(x).view(dtype)
    # Above may produce signaling nans; ignore warnings from invalid values.
    with np.errstate(invalid='ignore'):
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
      self._CompileAndCheck(jnp_op, args_maker)

  def testPathologicalFloats(self):
    args_maker = lambda: [np.array([
      0b_0111_1111_1000_0000_0000_0000_0000_0000, # inf
      0b_1111_1111_1000_0000_0000_0000_0000_0000, # -inf
      0b_0111_1111_1100_0000_0000_0000_0000_0000, # qnan
      0b_1111_1111_1100_0000_0000_0000_0000_0000, # -qnan
      0b_0111_1111_1000_0000_0000_0000_0000_0001, # snan
      0b_1111_1111_1000_0000_0000_0000_0000_0001, # -snan
      0b_0111_1111_1000_0000_0000_1100_0000_0000, # nonstandard nan
      0b_1111_1111_1000_0000_0000_1100_0000_0000, # -nonstandard nan
      0b_0000_0000_0000_0000_0000_0000_0000_0000, # zero
      0b_1000_0000_0000_0000_0000_0000_0000_0000, # -zero
    ], dtype='uint32')]

    np_op = lambda x: np.asarray(x).view('float32').view('uint32')
    jnp_op = lambda x: jnp.asarray(x).view('float32').view('uint32')

    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  # TODO(mattjj): test other ndarray-like method overrides

  def testNpMean(self):
    # from https://github.com/jax-ml/jax/issues/125
    x = jnp.eye(3, dtype=float) + 0.
    ans = np.mean(x)
    self.assertAllClose(ans, np.array(1./3), check_dtypes=False)

  def testArangeOnFloats(self):
    np_arange = jtu.with_jax_dtype_defaults(np.arange)
    # from https://github.com/jax-ml/jax/issues/145
    self.assertAllClose(np_arange(0.0, 1.0, 0.1),
                        jnp.arange(0.0, 1.0, 0.1))
    # from https://github.com/jax-ml/jax/issues/3450
    self.assertAllClose(np_arange(2.5),
                        jnp.arange(2.5))
    self.assertAllClose(np_arange(0., 2.5),
                        jnp.arange(0., 2.5))

  def testArangeTypes(self):
    # Test that arange() output type is equal to the default types.
    int_ = dtypes.default_int_dtype()
    float_ = dtypes.default_float_dtype()

    self.assertEqual(jnp.arange(10).dtype, int_)
    self.assertEqual(jnp.arange(10.).dtype, float_)
    self.assertEqual(jnp.arange(10, dtype='uint16').dtype, np.uint16)
    self.assertEqual(jnp.arange(10, dtype='bfloat16').dtype, jnp.bfloat16)

    self.assertEqual(jnp.arange(0, 10, 1).dtype, int_)
    with jax.numpy_dtype_promotion('standard'):
      self.assertEqual(jnp.arange(0, 10, 1.).dtype, float_)
      self.assertEqual(jnp.arange(0., 10, 1).dtype, float_)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonzerodim_shapes
      for axis in (NO_VALUE, None, *range(-len(shape), len(shape)))
    ],
    stable=[True, False],
    dtype=all_dtypes,
    use_method=[True, False]
  )
  def testSort(self, dtype, shape, axis, stable, use_method):
    rng = jtu.rand_some_equal(self.rng()) if stable else jtu.rand_some_inf_and_nan(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    kwds = {} if axis is NO_VALUE else {'axis': axis}

    def np_fun(arr):
      # Note: numpy sort fails on NaN and Inf values with bfloat16
      dtype = arr.dtype
      if arr.dtype == jnp.bfloat16:
        arr = arr.astype('float32')
      # TODO(jakevdp): switch to stable=stable when supported by numpy.
      result = np.sort(arr, kind='stable' if stable else None, **kwds)
      with jtu.ignore_warning(category=RuntimeWarning, message='invalid value'):
        return result.astype(dtype)
    if use_method:
      jnp_fun = lambda x: jnp.asarray(x).sort(stable=stable, **kwds)
    else:
      jnp_fun = partial(jnp.sort, stable=stable, **kwds)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testSortStableDescending(self):
    # TODO(jakevdp): test directly against np.sort when descending is supported.
    x = jnp.array([0, 1, jnp.nan, 0, 2, jnp.nan, -jnp.inf, jnp.inf])
    x_sorted = jnp.array([-jnp.inf, 0, 0, 1, 2, jnp.inf, jnp.nan, jnp.nan])
    argsorted_stable = jnp.array([6, 0, 3, 1, 4, 7, 2, 5])
    argsorted_rev_stable = jnp.array([2, 5, 7, 4, 1, 0, 3, 6])

    self.assertArraysEqual(jnp.sort(x), x_sorted)
    self.assertArraysEqual(jnp.sort(x, descending=True), lax.rev(x_sorted, [0]))
    self.assertArraysEqual(jnp.argsort(x), argsorted_stable)
    self.assertArraysEqual(jnp.argsort(x, descending=True), argsorted_rev_stable)

  @jtu.sample_product(shape=nonzerodim_shapes, dtype=all_dtypes)
  def testSortComplex(self, shape, dtype):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np.sort_complex, jnp.sort_complex, args_maker,
                            check_dtypes=False)
    self._CompileAndCheck(jnp.sort_complex, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_nonscalar_array_shapes
      for axis in (-1, *range(len(shape) - 1))
    ],
    dtype=all_dtypes,
    input_type=[np.array, tuple],
  )
  def testLexsort(self, dtype, shape, input_type, axis):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [input_type(rng(shape, dtype))]
    jnp_op = lambda x: jnp.lexsort(x, axis=axis)
    np_op = jtu.with_jax_dtype_defaults(lambda x: np.lexsort(x, axis=axis))
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonzerodim_shapes
      for axis in (NO_VALUE, None, *range(-len(shape), len(shape)))
    ],
    dtype=all_dtypes,
    use_method=[True, False]
  )
  def testArgsort(self, dtype, shape, axis, use_method):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    kwds = {'stable': True}
    if axis is not NO_VALUE:
      kwds['axis'] = axis

    @jtu.with_jax_dtype_defaults
    def np_fun(arr):
      # Note: numpy sort fails on NaN and Inf values with bfloat16
      if arr.dtype == jnp.bfloat16:
        arr = arr.astype('float32')
      return np.argsort(arr, **kwds)
    if use_method:
      jnp_fun = lambda x: jnp.asarray(x).argsort(**kwds)
    else:
      jnp_fun = partial(jnp.argsort, **kwds)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(dtype=[np.int16, np.uint32])
  def testArgsortDtype(self, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((10,), 'float32')]
    np_fun = lambda x: np.argsort(x).astype(dtype)
    jnp_fun = partial(jnp.argsort, dtype=dtype)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in nonempty_nonscalar_array_shapes
      for axis in (NO_VALUE, None, *range(-len(shape), len(shape)))
    ],
    descending=[True, False],
    dtype=all_dtypes,
  )
  def testArgsortUnstable(self, dtype, shape, axis, descending):
    # We cannot directly compare unstable argsorts, so instead check that indexed values match.
    rng = jtu.rand_some_equal(self.rng())
    x = rng(shape, dtype)
    kwds = {} if axis is NO_VALUE else {'axis': axis}
    expected = jnp.sort(x, descending=descending, stable=False, **kwds)
    indices = jnp.argsort(x, descending=descending, stable=False, **kwds)
    if axis is None:
      actual = jnp.ravel(x)[indices]
    else:
      actual = jnp.take_along_axis(x, indices, axis=-1 if axis is NO_VALUE else axis)
    self.assertArraysEqual(actual, expected)

  def _assertSamePartitionedArrays(self, jnp_output, np_output, axis, kth, shape):
    # Assert that pivot point is equal:
    self.assertArraysEqual(
      lax.index_in_dim(jnp_output, axis=axis, index=kth),
      lax.index_in_dim(np_output, axis=axis, index=kth))

    # Assert remaining values are correctly partitioned:
    self.assertArraysEqual(
      lax.sort(lax.slice_in_dim(jnp_output, start_index=0, limit_index=kth, axis=axis), dimension=axis),
      lax.sort(lax.slice_in_dim(np_output, start_index=0, limit_index=kth, axis=axis), dimension=axis))
    self.assertArraysEqual(
      lax.sort(lax.slice_in_dim(jnp_output, start_index=kth + 1, limit_index=shape[axis], axis=axis), dimension=axis),
      lax.sort(lax.slice_in_dim(np_output, start_index=kth + 1, limit_index=shape[axis], axis=axis), dimension=axis))

  @jtu.sample_product(
    [{'shape': shape, 'axis': axis, 'kth': kth}
     for shape in nonzerodim_shapes
     for axis in range(-len(shape), len(shape))
     for kth in range(-shape[axis], shape[axis])],
    dtype=default_dtypes,
  )
  def testPartition(self, shape, dtype, axis, kth):
    rng = jtu.rand_default(self.rng())
    arg = rng(shape, dtype)
    jnp_output = jnp.partition(arg, axis=axis, kth=kth)
    np_output = np.partition(arg, axis=axis, kth=kth)
    self._assertSamePartitionedArrays(jnp_output, np_output, axis, kth, shape)

  @jtu.sample_product(
    kth=range(10),
    dtype=unsigned_dtypes,
  )
  def testPartitionUnsignedWithZeros(self, kth, dtype):
    # https://github.com/jax-ml/jax/issues/22137
    max_val = np.iinfo(dtype).max
    arg = jnp.array([[6, max_val, 0, 4, 3, 1, 0, 7, 5, 2]], dtype=dtype)
    axis = -1
    shape = arg.shape
    jnp_output = jnp.partition(arg, axis=axis, kth=kth)
    np_output = np.partition(arg, axis=axis, kth=kth)
    self._assertSamePartitionedArrays(jnp_output, np_output, axis, kth, shape)

  @jtu.sample_product(
    [{'shape': shape, 'axis': axis, 'kth': kth}
     for shape in nonzerodim_shapes
     for axis in range(-len(shape), len(shape))
     for kth in range(-shape[axis], shape[axis])],
    dtype=default_dtypes,
    use_method=[True, False]
  )
  def testArgpartition(self, shape, dtype, axis, kth, use_method):
    rng = jtu.rand_default(self.rng())
    arg = rng(shape, dtype)

    np_output = np.argpartition(arg, axis=axis, kth=kth)
    if use_method:
      jnp_output = jnp.asarray(arg).argpartition(axis=axis, kth=kth)
    else:
      jnp_output = jnp.argpartition(arg, axis=axis, kth=kth)

    # Assert that all indices are present
    self.assertArraysEqual(jnp.sort(jnp_output, axis), np.sort(np_output, axis), check_dtypes=False)

    # Because JAX & numpy may treat duplicates differently, we must compare values
    # rather than indices.
    getvals = lambda x, ind: x[ind]
    for ax in range(arg.ndim):
      if ax != range(arg.ndim)[axis]:
        getvals = jax.vmap(getvals, in_axes=ax, out_axes=ax)
    jnp_values = getvals(arg, jnp_output)
    np_values = getvals(arg, np_output)
    self._assertSamePartitionedArrays(jnp_values, np_values, axis, kth, shape)

  @jtu.sample_product(
    kth=range(10),
    dtype=unsigned_dtypes,
  )
  def testArgpartitionUnsignedWithZeros(self, kth, dtype):
    # https://github.com/jax-ml/jax/issues/22137
    max_val = np.iinfo(dtype).max
    arg = jnp.array([[6, max_val, 0, 4, 3, 1, 0, 7, 5, 2, 3]], dtype=dtype)
    axis = -1
    shape = arg.shape
    jnp_output = jnp.argpartition(arg, axis=axis, kth=kth)
    np_output = np.argpartition(arg, axis=axis, kth=kth)

    # Assert that all indices are present
    self.assertArraysEqual(jnp.sort(jnp_output, axis), np.sort(np_output, axis), check_dtypes=False)

    # Because JAX & numpy may treat duplicates differently, we must compare values
    # rather than indices.
    getvals = lambda x, ind: x[ind]
    for ax in range(arg.ndim):
      if ax != range(arg.ndim)[axis]:
        getvals = jax.vmap(getvals, in_axes=ax, out_axes=ax)
    jnp_values = getvals(arg, jnp_output)
    np_values = getvals(arg, np_output)
    self._assertSamePartitionedArrays(jnp_values, np_values, axis, kth, shape)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis, k=k)
     for shape in [(20,), (5, 8), (3, 4, 5)]
     for axis in range(-len(shape), len(shape))
     for k in [0, 1, 3, shape[axis]]],
    mode=['largest', 'smallest'],
    sorted=[True, False],
    dtype=[np.float32, np.int32, np.uint32, np.bool_],
  )
  def testTopK(self, shape, dtype, axis, k, mode, sorted):
    rng = jtu.rand_some_equal(self.rng())
    arg = rng(shape, dtype)
    args_maker = lambda: [arg]

    jnp_fun = lambda x: jnp.top_k(x, k, axis=axis, mode=mode, sorted=sorted)

    jnp_vals, jnp_indices = jnp_fun(arg)
    ref_vals, _ = ref_top_k(arg, k, axis=axis, mode=mode, sorted=sorted)

    if sorted:
      self.assertArraysEqual(jnp_vals, ref_vals)
    else:
      self.assertArraysEqual(
          jnp.sort(jnp_vals, axis=axis),
          jnp.sort(ref_vals, axis=axis),
      )

    gathered = jnp.take_along_axis(arg, jnp_indices, axis=axis)
    self.assertArraysEqual(gathered, jnp_vals)

    self._CompileAndCheck(jnp_fun, args_maker)

  def testTopKErrors(self):
    x = jnp.arange(10)
    with self.assertRaisesRegex(ValueError, "k argument to top_k must be nonnegative"):
      jnp.top_k(x, -1)
    with self.assertRaisesRegex(ValueError, "k argument to top_k must be no larger than size"):
      jnp.top_k(x, 15)
    with self.assertRaisesRegex(ValueError, "axis 5 is out of bounds"):
      jnp.top_k(x, 2, axis=5)
    with self.assertRaisesRegex(ValueError, "mode must be 'largest' or 'smallest'"):
      jnp.top_k(x, 2, mode="middle")
    with self.assertRaisesRegex(ValueError, "top_k is not compatible with complex inputs"):
      jnp.top_k(jnp.array([1+2j, 3+4j]), 1)

  @jtu.sample_product(
    [dict(shifts=shifts, axis=axis)
      for shifts, axis in [
        (3, None),
        (1, 1),
        ((3,), (0,)),
        ((-2,), (-2,)),
        ((1, 2), (0, -1)),
        ((4, 2, 5, 5, 2, 4), None),
        (100, None),
      ]
    ],
    dtype=all_dtypes,
    shape=[(3, 4), (3, 4, 5), (7, 4, 0)],
  )
  def testRoll(self, shape, dtype, shifts, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), np.array(shifts)]
    jnp_op = partial(jnp.roll, axis=axis)
    np_op = partial(np.roll, axis=axis)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    dtype=all_dtypes,
    shape=[(1, 2, 3, 4)],
    axis=[-3, 0, 2, 3],
    start=[-4, -1, 2, 4],
  )
  def testRollaxis(self, shape, dtype, start, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.rollaxis, axis=axis, start=start)
    np_op = partial(np.rollaxis, axis=axis, start=start)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    dtype=int_dtypes + unsigned_dtypes + bool_dtypes,
    bitorder=['big', 'little'],
    shape=[(1, 2, 3, 4)],
    axis=[None, 0, 1, -2, -1],
  )
  def testPackbits(self, shape, dtype, axis, bitorder):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.packbits, axis=axis, bitorder=bitorder)
    np_op = partial(np.packbits, axis=axis, bitorder=bitorder)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    dtype=[np.uint8],
    bitorder=['big', 'little'],
    shape=[(1, 2, 3, 4)],
    axis=[None, 0, 1, -2, -1],
    count=[None, 20],
  )
  def testUnpackbits(self, shape, dtype, axis, bitorder, count):
    rng = jtu.rand_int(self.rng(), 0, 256)
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.unpackbits, axis=axis, bitorder=bitorder, count=count)
    np_op = partial(np.unpackbits, axis=axis, bitorder=bitorder, count=count)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in [(3,), (3, 4), (3, 4, 5)]
      for axis in itertools.chain(range(-len(shape), len(shape)),
                                  [cast(int | None, None)])
    ],
    index_shape=scalar_shapes + [(3,), (2, 1, 3)],
    dtype=all_dtypes,
    index_dtype=int_dtypes,
    mode=[None, 'wrap', 'clip'],
  )
  def testTake(self, shape, dtype, index_shape, index_dtype, axis, mode):
    def args_maker():
      x = rng(shape, dtype)
      i = rng_indices(index_shape, index_dtype)
      return x, i

    rng = jtu.rand_default(self.rng())
    if mode is None:
      rng_indices = jtu.rand_int(self.rng(), -shape[axis or 0], shape[axis or 0])
    else:
      rng_indices = jtu.rand_int(self.rng(), -5, 5)
    jnp_op = lambda x, i: jnp.take(x, i, axis=axis, mode=mode)
    np_op = lambda x, i: np.take(x, i, axis=axis, mode=mode)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  def testTakeEmpty(self):
    np.testing.assert_array_equal(
      jnp.array([], dtype=jnp.float32),
      jnp.take(jnp.array([], jnp.float32), jnp.array([], jnp.int32)))

    np.testing.assert_array_equal(
      jnp.ones((2, 0, 4), dtype=jnp.float32),
      jnp.take(jnp.ones((2, 0, 4), dtype=jnp.float32), jnp.array([], jnp.int32),
               axis=1))

    with self.assertRaisesRegex(IndexError, "non-empty jnp.take"):
      jnp.take(jnp.ones((2, 0, 4), dtype=jnp.float32),
               jnp.array([0], jnp.int32), axis=1)

  def testTakeOptionalArgs(self):
    x = jnp.arange(5.0)
    ind = jnp.array([0, 2, 4, 6])
    expected = jnp.array([0.0, 2.0, 4.0, 10.0], dtype=x.dtype)
    actual = jnp.take(x, ind, unique_indices=True,
                      indices_are_sorted=True, fill_value=10.0)
    self.assertArraysEqual(expected, actual)

  @jtu.sample_product(
    [dict(x_shape=x_shape, i_shape=i_shape, axis=axis)
      for x_shape, i_shape in filter(
        _shapes_are_equal_length,
        filter(_shapes_are_broadcast_compatible,
               itertools.combinations_with_replacement(nonempty_nonscalar_array_shapes, 2)))
      for axis in itertools.chain(range(len(x_shape)), [-1],
                                  [cast(int | None, None)])
    ],
    dtype=default_dtypes,
    index_dtype=int_dtypes,
  )
  def testTakeAlongAxis(self, x_shape, i_shape, dtype, index_dtype, axis):
    rng = jtu.rand_default(self.rng())

    i_shape = list(i_shape)
    if axis is None:
      i_shape = [math.prod(i_shape)]
    else:
      # Test the case where the size of the axis doesn't necessarily broadcast.
      i_shape[axis] *= 3
    def args_maker():
      x = rng(x_shape, dtype)
      n = math.prod(x_shape) if axis is None else x_shape[axis]
      if np.issubdtype(index_dtype, np.unsignedinteger):
        index_rng = jtu.rand_int(self.rng(), 0, n)
      else:
        index_rng = jtu.rand_int(self.rng(), -n, n)
      i = index_rng(i_shape, index_dtype)
      return x, i

    jnp_op = lambda x, i: jnp.take_along_axis(x, i, axis=axis)
    jnp_one_hot_op = lambda x, i: jnp.take_along_axis(
        x, i, axis=axis, mode='one_hot'
    )
    np_op = lambda x, i: np.take_along_axis(x, i, axis=axis)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CheckAgainstNumpy(np_op, jnp_one_hot_op, args_maker,
                            atol=1e-2, rtol=1e-2)
    self._CompileAndCheck(jnp_op, args_maker)
    self._CompileAndCheck(jnp_one_hot_op, args_maker)

  def testTakeAlongAxisDefaultAxis(self):
    arr = jtu.rand_default(self.rng())((10, 20), np.float32)
    indices = jtu.rand_int(self.rng(), 0, 100)((10, 30), np.uint8)
    q0 = jnp.take_along_axis(arr, indices, axis=-1)
    q1 = jnp.take_along_axis(arr, indices)
    np.testing.assert_array_equal(q0, q1)

  def testTakeAlongAxisPromiseInBoundsNoWrappingRetainsDtype(self):
    arr = jnp.arange(200, dtype=jnp.float32)
    indices = jnp.array([1, 2, 3], dtype=jnp.int8)
    jaxpr = jax.make_jaxpr(
        lambda i: jnp.take_along_axis(
            arr,
            i,
            axis=0,
            mode='promise_in_bounds',
            wrap_negative_indices=False,
        )
    )(indices)
    [jit_eqn] = (eqn for eqn in jaxpr.eqns if eqn.primitive.name == 'jit')
    nested_jaxpr = jit_eqn.params['jaxpr']
    [gather_eqn] = (
        eqn for eqn in nested_jaxpr.eqns if eqn.primitive == lax.gather_p
    )
    indices_var = gather_eqn.invars[1]
    self.assertEqual(indices_var.aval.dtype, jnp.int8)

  def testTakeAlongAxisPromiseInBoundsWrappingUpcastsDtype(self):
    arr = jnp.arange(200, dtype=jnp.float32)
    indices = jnp.array([1, 2, 3], dtype=jnp.int8)
    jaxpr = jax.make_jaxpr(
        lambda i: jnp.take_along_axis(
            arr, i, axis=0, mode='promise_in_bounds', wrap_negative_indices=True
        )
    )(indices)
    [jit_eqn] = (eqn for eqn in jaxpr.eqns if eqn.primitive.name == 'jit')
    nested_jaxpr = jit_eqn.params['jaxpr']
    [gather_eqn] = (
        eqn for eqn in nested_jaxpr.eqns if eqn.primitive == lax.gather_p
    )
    indices_var = gather_eqn.invars[1]
    self.assertEqual(indices_var.aval.dtype, jnp.int32)

  def testTakeAlongAxisWithInt8Indices(self):
    h = jtu.rand_default(self.rng())((256, 256, 100), np.float32)
    g = jtu.rand_int(self.rng(), -100, 99)((256, 256, 1), np.int8)
    q0 = jnp.take_along_axis(h, g, axis=-1)
    q1 = np.take_along_axis(h, g, axis=-1)
    np.testing.assert_equal(q0, q1)

  def testTakeAlongAxisPromiseInBoundsWithInt8Indices(self):
    h = jtu.rand_default(self.rng())((256, 256, 100), np.float32)
    g = jtu.rand_int(self.rng(), -100, 99)((256, 256, 1), np.int8)
    q0 = jnp.take_along_axis(
        h, g, axis=-1, mode='promise_in_bounds', wrap_negative_indices=True
    )
    q1 = np.take_along_axis(h, g, axis=-1)
    np.testing.assert_equal(q0, q1)

  def testTakeAlongAxisWithUint8IndicesDoesNotOverflow(self):
    # https://github.com/jax-ml/jax/issues/5088
    h = jtu.rand_default(self.rng())((256, 256, 100), np.float32)
    g = jtu.rand_int(self.rng(), 0, 100)((256, 256, 1), np.uint8)
    q0 = jnp.take_along_axis(h, g, axis=-1)
    q1 = np.take_along_axis( h, g, axis=-1)
    np.testing.assert_equal(q0, q1)

  def testTakeAlongAxisInputTensorHasSingletonDimension(self):
    h = jtu.rand_default(self.rng())((2, 1, 5, 7, 13), np.float32)
    g = jtu.rand_int(self.rng(), 0, 7)((2, 3, 5, 11, 1), np.uint8)
    q0 = jnp.take_along_axis(h, g, axis=-2)
    q1 = np.take_along_axis( h, g, axis=-2)
    np.testing.assert_equal(q0, q1)

  @parameterized.named_parameters(
      ('default', None, None),
      ('clip', 'clip', True),
      ('promise_in_bounds', 'promise_in_bounds', True),
  )
  def testTakeAlongAxisWrapNegativeIndices(self, mode, wrap_negative_indices):
    x = jnp.array([10, 20, 30])
    idx = jnp.array([-1, -2])
    kwargs = {}
    if mode is not None:
      kwargs['mode'] = mode
    if wrap_negative_indices is not None:
      kwargs['wrap_negative_indices'] = wrap_negative_indices
    np.testing.assert_array_equal(
        jnp.take_along_axis(x, idx, axis=0, **kwargs), np.array([30, 20])
    )

  def testTakeAlongAxisClipNoWrapNegativeIndices(self):
    x = jnp.array([10, 20, 30])
    idx = jnp.array([-1, -2])
    np.testing.assert_array_equal(
        jnp.take_along_axis(
            x, idx, axis=0, mode='clip', wrap_negative_indices=False
        ),
        np.array([10, 10]),
    )

  def testTakeAlongAxisPromiseInBoundsNoWrapNegativeIndices(self):
    x = jnp.array([10, 20, 30])
    idx = jnp.array([-1, -2])
    # Verify no jaxpr wrapping operations are generated.
    jaxpr_wrap = jax.make_jaxpr(
        lambda x, i: jnp.take_along_axis(
            x, i, axis=0, mode='promise_in_bounds', wrap_negative_indices=True
        )
    )(x, idx)
    jaxpr_no_wrap = jax.make_jaxpr(
        lambda x, i: jnp.take_along_axis(
            x, i, axis=0, mode='promise_in_bounds', wrap_negative_indices=False
        )
    )(x, idx)
    jaxpr_default = jax.make_jaxpr(
        lambda x, i: jnp.take_along_axis(
            x, i, axis=0, mode='promise_in_bounds'  # should wrap by default
        )
    )(x, idx)

    def get_all_primitives(jaxpr):
      prims = set()
      for eqn in jaxpr.eqns:
        prims.add(eqn.primitive.name)
        if eqn.primitive.name == 'jit':
          prims.update(get_all_primitives(eqn.params['jaxpr']))
      return prims

    wrap_prims = get_all_primitives(jaxpr_wrap)
    no_wrap_prims = get_all_primitives(jaxpr_no_wrap)
    default_prims = get_all_primitives(jaxpr_default)

    self.assertTrue(
        'select' in wrap_prims or 'lt' in wrap_prims or 'add' in wrap_prims
    )
    self.assertFalse(
        'select' in no_wrap_prims
        or 'lt' in no_wrap_prims
        or 'add' in no_wrap_prims
    )
    self.assertTrue(
        'select' in default_prims
        or 'lt' in default_prims
        or 'add' in default_prims
    )

  def testTakeAlongAxisOutOfBounds(self):
    x = jnp.arange(10, dtype=jnp.float32)
    idx = jnp.array([-11, -10, -9, -5, -1, 0, 1, 5, 9, 10, 11])
    out = jnp.take_along_axis(x, idx, axis=0)
    expected_fill = np.array([jnp.nan, 0, 1, 5, 9, 0, 1, 5, 9, jnp.nan,
                              jnp.nan], np.float32)
    np.testing.assert_array_equal(expected_fill, out)
    out = jnp.take_along_axis(x, idx, axis=0, mode="fill")
    np.testing.assert_array_equal(expected_fill, out)

    expected_clip = np.array([0, 0, 1, 5, 9, 0, 1, 5, 9, 9, 9], np.float32)
    out = jnp.take_along_axis(x, idx, axis=0, mode="clip")
    np.testing.assert_array_equal(expected_clip, out)

  def testTakeAlongAxisRequiresIntIndices(self):
    x = jnp.arange(5)
    idx = jnp.array([3.], jnp.float32)
    with self.assertRaisesRegex(
        TypeError,
        "take_along_axis indices must be of integer type, got float32"):
      jnp.take_along_axis(x, idx, axis=0)

  def testTakeAlongAxisWithEmptyArgs(self):
    # take_along_axis should allow us to gather an empty list of indices from
    # an empty input axis without raising a shape error.
    x = jnp.ones((4, 0, 3), dtype=jnp.int32)
    np.testing.assert_array_equal(x, jnp.take_along_axis(x, x, axis=1))

  def testTakeAlongAxisOptionalArgs(self):
    x = jnp.arange(5.0)
    ind = jnp.array([0, 2, 4, 6])
    expected = jnp.array([0.0, 2.0, 4.0, 10.0], dtype=x.dtype)
    actual = jnp.take_along_axis(x, ind, axis=None, mode='fill', fill_value=10.0)
    self.assertArraysEqual(expected, actual)

  @jtu.sample_product(
    dtype=inexact_dtypes,
    shape=[0, 5],
    n=[2, 4],
    increasing=[False, True],
  )
  def testVander(self, shape, dtype, n, increasing):
    rng = jtu.rand_default(self.rng())
    def np_fun(arg):
      arg = arg.astype(np.float32) if dtype == jnp.bfloat16 else arg
      return np.vander(arg, N=n, increasing=increasing)
    jnp_fun = lambda arg: jnp.vander(arg, N=n, increasing=increasing)
    args_maker = lambda: [rng([shape], dtype)]
    # np.vander seems to return float64 for all floating types. We could obey
    # those semantics, but they seem like a bug.
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol={np.float32: 1e-3, np.complex64: 1e-3})
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=False)

  @jtu.sample_product(
    shape=array_shapes,
    dtype=all_dtypes,
  )
  def testNanToNum(self, shape, dtype):
    rng = jtu.rand_some_inf_and_nan(self.rng())
    dtype = np.dtype(dtypes.canonicalize_dtype(dtype)).type
    def np_fun(x):
      if dtype == jnp.bfloat16:
        x = np.where(np.isnan(x), dtype(0), x)
        x = np.where(np.isposinf(x), jnp.finfo(dtype).max, x)
        x = np.where(np.isneginf(x), jnp.finfo(dtype).min, x)
        return x
      else:
        return np.nan_to_num(x).astype(dtype)

    args_maker = lambda: [rng(shape, dtype)]
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    self._CheckAgainstNumpy(np_fun, jnp.nan_to_num, args_maker,
                            check_dtypes=check_dtypes)
    self._CompileAndCheck(jnp.nan_to_num, args_maker,
                          check_dtypes=check_dtypes)

  @jtu.sample_product(
    [dict(shapes=shapes, dtypes=dtypes)
        for shapes, dtypes in (
          ((), ()),
          (((7,),), (np.int32,)),
          (((3,), (4,)), (np.int32, np.int32)),
          (((3,), (1,), (4,)), (np.int32, np.int32, np.int32)),
        )
    ],
  )
  def testIx_(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)
                          for shape, dtype in zip(shapes, dtypes)]
    self._CheckAgainstNumpy(np.ix_, jnp.ix_, args_maker)
    self._CompileAndCheck(jnp.ix_, args_maker)

  @jtu.sample_product(
    dimensions=[(), (2,), (3, 0), (4, 5, 6)],
    dtype=number_dtypes,
    sparse=[True, False],
  )
  def testIndices(self, dimensions, dtype, sparse):
    def args_maker(): return []
    np_fun = partial(np.indices, dimensions=dimensions,
                     dtype=dtype, sparse=sparse)
    jnp_fun = partial(jnp.indices, dimensions=dimensions,
                      dtype=dtype, sparse=sparse)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testIndicesDefaultDtype(self):
    self.assertEqual(jnp.indices((2, 3)).dtype,
                     dtypes.default_int_dtype())

  @jtu.sample_product(
    shape=nonzerodim_shapes,
    dtype=all_dtypes,
  )
  def testWhereOneArgument(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(np.where, jnp.where, args_maker, check_dtypes=False)

    # JIT compilation requires specifying a size statically. Full test of
    # this behavior is in testNonzeroSize().
    jnp_fun = lambda x: jnp.where(x, size=np.size(x) // 2)

    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shapes=filter(_shapes_are_broadcast_compatible,
                  itertools.combinations_with_replacement(all_shapes, 3)),
    dtypes=itertools.combinations_with_replacement(all_dtypes, 3),
  )
  def testWhereThreeArgument(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    def np_fun(cond, x, y):
      return jtu.promote_like_jnp(partial(np.where, cond))(x, y)
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(np_fun, jnp.where, args_maker)
      self._CompileAndCheck(jnp.where, args_maker)

  def testWhereExtraCode(self):
    def f(x):
      return jnp.where(x > 0, x, -x)

    jaxpr = jax.make_jaxpr(jax.grad(f))(3.)
    # Test no comparison literal True/False in jaxpr, and hence no comparison to
    # literals
    self.assertNotIn('False', str(jaxpr))
    self.assertNotIn('True', str(jaxpr))

  def testWhereScalarPromotion(self):
    x = jnp.where(jnp.array([True, False]), 3,
                  jnp.ones((2,), dtype=jnp.float32))
    self.assertEqual(x.dtype, np.dtype(np.float32))

  @jtu.sample_product(
    [dict(n=n, shapes=shapes)
      for n in range(1, 3)
      for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(all_shapes, 2 * n + 1))
    ],
    # To avoid forming the full product of shapes and dtypes we always sample
    # maximal set of dtypes.
    dtypes=itertools.combinations_with_replacement(all_dtypes, 3),
  )
  @jax.numpy_rank_promotion('allow')
  def testSelect(self, n, shapes, dtypes):
    dtypes = dtypes[:n+1]
    rng = jtu.rand_default(self.rng())
    n = len(dtypes) - 1
    def args_maker():
      condlist = [rng(shape, np.bool_) for shape in shapes[:n]]
      choicelist = [rng(shape, dtype)
                    for shape, dtype in zip(shapes[n:-1], dtypes[:n])]
      default = rng(shapes[-1], dtypes[-1])
      return condlist, choicelist, default
    # TODO(phawkins): float32/float64 type mismatches
    @jax.numpy_dtype_promotion('standard')
    def np_fun(condlist, choicelist, default):
      choicelist = [x if jnp.result_type(x) != jnp.bfloat16
                    else x.astype(np.float32) for x in choicelist]
      dtype = jnp.result_type(default, *choicelist)
      return np.select(condlist,
                        [np.asarray(x).astype(dtype) for x in choicelist],
                        np.asarray(default, dtype=dtype))
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(np_fun, jnp.select, args_maker,
                              check_dtypes=False)
      self._CompileAndCheck(jnp.select, args_maker,
                            rtol={np.float64: 1e-7, np.complex128: 1e-7})

  def testIssue330(self):
    x = jnp.full((1, 1), jnp.array([1])[0])  # doesn't crash
    self.assertEqual(x[0, 0], 1)

  def testScalarDtypePromotion(self):
    orig_numpy_result = (1 + np.eye(1, dtype=np.float32)).dtype
    jax_numpy_result = (1 + jnp.eye(1, dtype=jnp.float32)).dtype
    self.assertEqual(orig_numpy_result, jax_numpy_result)

  def testSymmetrizeDtypePromotion(self):
    x = np.eye(3, dtype=np.float32)
    orig_numpy_result = ((x + x.T) / 2).dtype

    x = jnp.eye(3, dtype=jnp.float32)
    jax_numpy_result = ((x + x.T) / 2).dtype
    self.assertEqual(orig_numpy_result, jax_numpy_result)

  # NOTE(mattjj): I disabled this test when removing lax._safe_mul because
  # introducing the convention 0 * inf = 0 leads to silently wrong results in
  # some cases. See this comment for details:
  # https://github.com/jax-ml/jax/issues/1052#issuecomment-514083352
  # def testIssue347(self):
  #   # https://github.com/jax-ml/jax/issues/347
  #   def test_fail(x):
  #     x = jnp.sqrt(jnp.sum(x ** 2, axis=1))
  #     ones = jnp.ones_like(x)
  #     x = jnp.where(x > 0.5, x, ones)
  #     return jnp.sum(x)
  #   x = jnp.array([[1, 2], [3, 4], [0, 0]], dtype=jnp.float64)
  #   result = jax.grad(test_fail)(x)
  #   assert not np.any(np.isnan(result))

  def testIssue453(self):
    # https://github.com/jax-ml/jax/issues/453
    a = np.arange(6) + 1
    ans = jnp.reshape(a, (3, 2), order='F')
    expected = np.reshape(a, (3, 2), order='F')
    self.assertAllClose(ans, expected)

  @jtu.sample_product(
    scalar_type=[int, float, bool, complex],
    op=["atleast_1d", "atleast_2d", "atleast_3d"],
  )
  def testAtLeastNdLiterals(self, scalar_type, op):
    # Fixes: https://github.com/jax-ml/jax/issues/634
    np_fun = lambda arg: getattr(np, op)(arg).astype(
        dtypes.python_scalar_types_to_dtypes[scalar_type]
    )
    jnp_fun = lambda arg: getattr(jnp, op)(arg)
    args_maker = lambda: [scalar_type(2)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shape=[(0,), (5,), (10,)],
    dtype=int_dtypes + bool_dtypes,
    weights=[True, False],
    minlength=[0, 20],
    length=[None, 8],
  )
  def testBincount(self, shape, dtype, weights, minlength, length):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype), (rng(shape, 'float32') if weights else None))

    def np_fun(x, *args):
      x = np.clip(x, 0, None)  # jnp.bincount clips negative values to zero.
      out = np.bincount(x, *args, minlength=minlength)
      if length and length > out.size:
        return np.pad(out, (0, length - out.size))
      return out[:length]
    jnp_fun = partial(jnp.bincount, minlength=minlength, length=length)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    if length is not None:
      self._CompileAndCheck(jnp_fun, args_maker)

  def testBincountNegative(self):
    # Test that jnp.bincount ignores negative values.
    x_rng = jtu.rand_int(self.rng(), -100, 100)
    w_rng = jtu.rand_uniform(self.rng())
    shape = (1000,)
    x = x_rng(shape, 'int32')
    w = w_rng(shape, 'float32')

    xn = np.array(x)
    xn[xn < 0] = 0
    wn = np.array(w)
    np_result = np.bincount(xn[xn >= 0], wn[xn >= 0])
    jnp_result = jnp.bincount(x, w)
    self.assertAllClose(np_result, jnp_result, check_dtypes=False)

  @jtu.sample_product(
    input=[
      3,
      [3],
      [np.array(3)],
      [np.array([3])],
      [[np.array(3)]],
      [[np.array([3])]],
      [3, 4, 5],
      [
        [np.eye(2, dtype=np.int32) * 2, np.zeros((2, 3), dtype=np.int32)],
        [np.ones((3, 2), dtype=np.int32), np.eye(3, dtype=np.int32) * 3],
      ],
      [np.array([1, 2, 3]), np.array([2, 3, 4]), 10],
      [np.ones((2, 2), dtype=np.int32), np.zeros((2, 2), dtype=np.int32)],
      [[np.array([1, 2, 3])], [np.array([2, 3, 4])]],
    ],
  )
  def testBlock(self, input):
    args_maker = lambda: [input]
    self._CheckAgainstNumpy(np.block, jnp.block, args_maker)
    self._CompileAndCheck(jnp.block, args_maker)

  def testLongLong(self):
    self.assertAllClose(np.int64(7), jax.jit(lambda x: x)(np.longlong(7)))

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype.*")
  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises mixed type promotion
  def testArange(self):
    # test cases inspired by dask tests at
    # https://github.com/dask/dask/blob/main/dask/array/tests/test_creation.py#L92
    np_arange = jtu.with_jax_dtype_defaults(np.arange)
    self.assertAllClose(jnp.arange(77),
                        np_arange(77))
    self.assertAllClose(jnp.arange(2, 13),
                        np_arange(2, 13))
    self.assertAllClose(jnp.arange(4, 21, 9),
                        np_arange(4, 21, 9))
    self.assertAllClose(jnp.arange(53, 5, -3),
                        np_arange(53, 5, -3))
    self.assertAllClose(jnp.arange(77, dtype=float),
                        np_arange(77, dtype=float))
    self.assertAllClose(jnp.arange(2, 13, dtype=int),
                        np_arange(2, 13, dtype=int))
    self.assertAllClose(jnp.arange(0, 1, -0.5),
                        np_arange(0, 1, -0.5))

    self.assertRaises(TypeError, lambda: jnp.arange())

    # test that jnp.arange(N) doesn't instantiate an ndarray
    self.assertNotEqual(type(jnp.arange(77)), type(np.arange(77)))
    self.assertEqual(type(jnp.arange(77)), type(lax.iota(np.int32, 77)))

    # test that jnp.arange(N, dtype=int32) doesn't instantiate an ndarray
    self.assertNotEqual(type(jnp.arange(77, dtype=jnp.int32)),
                        type(np.arange(77, dtype=np.int32)))
    self.assertEqual(type(jnp.arange(77, dtype=jnp.int32)),
                     type(lax.iota(np.int32, 77)))

  def testArangeJit(self):
    ans = jax.jit(lambda: jnp.arange(5))()
    expected = jtu.with_jax_dtype_defaults(np.arange)(5)
    self.assertAllClose(ans, expected)

  @jtu.sample_product(
    args=[(5,), (0, 5)],
    specify_device=[True, False],
  )
  def testArangeJaxpr(self, args, specify_device):
    device = jax.devices()[-1] if specify_device else None
    kwargs = {"device": device}
    jaxpr = jax.make_jaxpr(lambda: jnp.arange(*args, **kwargs))()
    # We have 2 statements in jaxpr:
    # [a:i32[5] = iota[dimension=0 dtype=int32 shape=(5,)],
    #  a:i32[5] = device_put[devices=[None] srcs=[None]] b]
    num_eqs = 2 if device is not None else 1
    self.assertEqual(len(jaxpr.eqns), num_eqs)
    self.assertEqual(jaxpr.eqns[0].primitive, lax.iota_p)

  @jtu.sample_product(specify_device=[True, False])
  def testArangeJaxprNonZeroStart(self, specify_device):
    device = jax.devices()[-1] if specify_device else None
    jaxpr = jax.make_jaxpr(lambda: jnp.arange(1, 5, device=device))()
    # Non-zero start should produce iota + add (+ device_put if device specified)
    num_eqs = 3 if device is not None else 2
    self.assertEqual(len(jaxpr.eqns), num_eqs)
    self.assertEqual(jaxpr.eqns[0].primitive, lax.iota_p)
    self.assertEqual(jaxpr.eqns[1].primitive, lax.add_p)

  @jtu.sample_product(
      dtype=[np.int32, np.float32],
      iteration=range(10)
  )
  def testArangeRandomValues(self, dtype, iteration):
    del iteration  # not needed: each test case gets its own random seed.
    rng = jtu.rand_default(self.rng())
    start = rng((), dtype)
    stop = rng((), dtype)
    jax_result = jnp.arange(start, stop, dtype=dtype)
    np_result = np.arange(start, stop, dtype=dtype)
    self.assertAllClose(jax_result, np_result)

  @parameterized.parameters(
      (1+2j, 5+3j),
      (0+0j, 5+0j),
      (1.0+0j, 5.0+0j),
      (0, 5, 1+1j),
  )
  def testArangeComplex(self, *args):
    msg = "Passing complex start/stop/step to jnp.arange is no longer supported"
    with self.assertRaisesRegex(ValueError, msg):
      _ = jnp.arange(*args)

  @parameterized.parameters(int, float, np.int32, np.float32)
  def testArangeTransferGuard(self, typ):
    # Ensure that simple arange calls avoid host-to-device transfer.
    with jax.transfer_guard("disallow"):
      jnp.arange(typ(5))

  def testIssue830(self):
    a = jnp.arange(4, dtype=jnp.complex64)
    self.assertEqual(a.dtype, jnp.complex64)

  def testIssue728(self):
    np_eye = jtu.with_jax_dtype_defaults(np.eye)
    self.assertAllClose(jnp.eye(5000), np_eye(5000))
    self.assertEqual(0, np.sum(jnp.eye(1050) - np_eye(1050)))

  def testIssue746(self):
    jnp.arange(12).reshape(3, 4)  # doesn't crash

  def testIssue764(self):
    x = jnp.linspace(190, 200, 4)
    f = jax.grad(lambda x: jnp.sum(jnp.tanh(x)))
    # tanh(190) and tanh(200) are both 1, so the gradient is 0 in f64.
    expected = np.array([0,0,0,0], np.float64)
    self.assertAllClose(f(x), expected, check_dtypes=False, atol=1e-14, rtol=0)

  # NOTE(mattjj): I disabled this test when removing lax._safe_mul because this
  # is a numerical stability issue that should be solved with a custom jvp rule
  # of the sigmoid function being differentiated here, not by safe_mul.
  # def testIssue777(self):
  #   x = jnp.linspace(-200, 0, 4, dtype=np.float32)
  #   f = jax.grad(lambda x: jnp.sum(1 / (1 + jnp.exp(-x))))
  #   self.assertAllClose(f(x), np.array([0., 0., 0., 0.25], dtype=np.float32))

  @jtu.sample_product(
    dtype=float_dtypes,
    op=("sqrt", "arccos", "arcsin", "arctan", "sin", "cos", "tan",
        "sinh", "cosh", "tanh", "arccosh", "arcsinh", "arctanh", "exp",
        "log", "expm1", "log1p"),
  )
  def testMathSpecialFloatValues(self, op, dtype):
    np_op = getattr(np, op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                                 message="invalid value.*")(np_op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                                 message="divide by zero.*")(np_op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                                 message="overflow.*")(np_op)

    jnp_op = getattr(jnp, op)
    dtype = np.dtype(dtypes.canonicalize_dtype(dtype)).type

    for x in (np.nan, -np.inf, -100., -2., -1., 0., 1., 2., 100., np.inf,
              jnp.finfo(dtype).max, np.sqrt(jnp.finfo(dtype).max),
              np.sqrt(jnp.finfo(dtype).max) * 2.):
      x = dtype(x)
      expected = np_op(x)
      actual = jnp_op(x)
      tol = jtu.tolerance(dtype, {np.float32: 1e-3, np.float64: 1e-7})
      self.assertAllClose(expected, actual, atol=tol,
                          rtol=tol)

  def testIssue956(self):
    self.assertRaises(TypeError, lambda: jnp.ndarray((1, 1)))

  def testIssue967(self):
    self.assertRaises(TypeError, lambda: jnp.zeros(1.5))

  @jtu.sample_product(
    shape=[(5,), (10, 5), (4, 10)],
    dtype=number_dtypes,
    rowvar=[True, False],
  )
  @jax.default_matmul_precision("float32")
  def testCorrCoef(self, shape, dtype, rowvar):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      ok = False
      while not ok:
        x = rng(shape, dtype)
        ok = not np.any(np.isclose(np.std(x), 0.0))
      return (x,)
    np_fun = partial(np.corrcoef, rowvar=rowvar)
    np_fun = jtu.ignore_warning(
      category=RuntimeWarning, message="invalid value encountered.*")(np_fun)
    jnp_fun = partial(jnp.corrcoef, rowvar=rowvar)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @unittest.skipIf(numpy_version < (2, 2, 0), "test covers NumPy 2.2+ behavior.")
  @jtu.sample_product(
      shape=[(1, 3), (3, 1)],
      rowvar=[True, False],
  )
  @jax.default_matmul_precision("float32")
  def testCorrCoefTransposeBehavior(self, shape, rowvar):
    # Regression test for https://github.com/jax-ml/jax/issues/29571. The
    # NumPy 2.2 update to jnp.cov (see
    # https://github.com/numpy/numpy/pull/27661) flows through to corrcoef
    # for single-row design matrices with rowvar=False, so the output shape
    # and NaN propagation should now match NumPy.
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, np.float32)]

    @jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")
    @jtu.ignore_warning(category=RuntimeWarning, message="Degrees of freedom.*")
    @jtu.ignore_warning(category=RuntimeWarning, message="divide by zero.*")
    def np_fun(x):
      return np.corrcoef(x, rowvar=rowvar)

    jnp_fun = partial(jnp.corrcoef, rowvar=rowvar)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testCorrCoefDtype(self):
    x = jnp.arange(5)
    result_bf16 = jnp.corrcoef(x, x, dtype='bfloat16')
    self.assertEqual(result_bf16.dtype, np.dtype('bfloat16'))

    with self.assertRaisesRegex(ValueError, "corrcoef: dtype must be a subclass of float or complex"):
      jnp.corrcoef(x, x, dtype=int)

  @jtu.sample_product(
    [dict(dtype=dtype, end_dtype=end_dtype, begin_dtype=begin_dtype,
          shape=shape, begin_shape=begin_shape, end_shape=end_shape)
      for dtype in number_dtypes
      for end_dtype in [None] + [dtype]
      for begin_dtype in [None] + [dtype]
      for shape in [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE]
      for begin_shape in (
        [None] if begin_dtype is None
        else [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE])
      for end_shape in (
        [None] if end_dtype is None
        else [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE])
    ],
  )
  def testEDiff1d(self, shape, dtype, end_shape, end_dtype, begin_shape,
                  begin_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype),
            (None if end_dtype is None else rng(end_shape, end_dtype)),
            (None if begin_dtype is None else rng(begin_shape, begin_dtype))]
    np_fun = lambda x, to_end, to_begin: np.ediff1d(x, to_end, to_begin)
    jnp_fun = lambda x, to_end, to_begin: jnp.ediff1d(x, to_end, to_begin)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testEDiff1dWithDtypeCast(self):
    rng = jtu.rand_default(self.rng())
    shape = jtu.NUMPY_SCALAR_SHAPE
    dtype = jnp.float32
    end_dtype = jnp.int32
    args_maker = lambda: [rng(shape, dtype), rng(shape, end_dtype), rng(shape, dtype)]
    np_fun = lambda x, to_end, to_begin: np.ediff1d(x, to_end, to_begin)
    jnp_fun = lambda x, to_end, to_begin: jnp.ediff1d(x, to_end, to_begin)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    shapes=[(), (5,), (5, 3)],
    dtype=number_dtypes,
    indexing=['xy', 'ij'],
    sparse=[True, False],
  )
  def testMeshGrid(self, shapes, dtype, indexing, sparse):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [(x,) for x in shapes],
                                    [dtype] * len(shapes))
    np_fun = partial(np.meshgrid, indexing=indexing, sparse=sparse)
    jnp_fun = partial(jnp.meshgrid, indexing=indexing, sparse=sparse)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testMgrid(self):
    # wrap indexer for appropriate dtype defaults.
    np_mgrid = _indexer_with_default_outputs(np.mgrid)
    assertAllEqual = partial(self.assertAllClose, atol=0, rtol=0)
    assertAllEqual(np_mgrid[()], jnp.mgrid[()])
    assertAllEqual(np_mgrid[:4], jnp.mgrid[:4])
    assertAllEqual(np_mgrid[:4,], jnp.mgrid[:4,])
    assertAllEqual(np_mgrid[:4], jax.jit(lambda: jnp.mgrid[:4])())
    assertAllEqual(np_mgrid[:5, :5], jnp.mgrid[:5, :5])
    assertAllEqual(np_mgrid[:3, :2], jnp.mgrid[:3, :2])
    assertAllEqual(np_mgrid[1:4:2], jnp.mgrid[1:4:2])
    assertAllEqual(np_mgrid[1:5:3, :5], jnp.mgrid[1:5:3, :5])
    assertAllEqual(np_mgrid[:3, :2, :5], jnp.mgrid[:3, :2, :5])
    assertAllEqual(np_mgrid[:3:2, :2, :5], jnp.mgrid[:3:2, :2, :5])
    # Corner cases
    assertAllEqual(np_mgrid[:], jnp.mgrid[:])
    # When the step length is a complex number, because of float calculation,
    # the values between jnp and np might slightly different.
    atol = 1e-6
    rtol = 1e-6
    self.assertAllClose(np_mgrid[-1:1:5j],
                        jnp.mgrid[-1:1:5j],
                        atol=atol,
                        rtol=rtol)
    self.assertAllClose(np_mgrid[3:4:7j],
                        jnp.mgrid[3:4:7j],
                        atol=atol,
                        rtol=rtol)
    self.assertAllClose(np_mgrid[1:6:8j, 2:4],
                        jnp.mgrid[1:6:8j, 2:4],
                        atol=atol,
                        rtol=rtol)
    # Non-integer steps
    self.assertAllClose(np_mgrid[0:3.5:0.5],
                        jnp.mgrid[0:3.5:0.5],
                        atol=atol,
                        rtol=rtol)
    self.assertAllClose(np_mgrid[1.3:4.2:0.3],
                        jnp.mgrid[1.3:4.2:0.3],
                        atol=atol,
                        rtol=rtol)
    # abstract tracer value for jnp.mgrid slice
    with self.assertRaisesRegex(core.ConcretizationTypeError,
                                "slice start of jnp.mgrid"):
      jax.jit(lambda a, b: jnp.mgrid[a:b])(0, 2)

  def testOgrid(self):
    # wrap indexer for appropriate dtype defaults.
    np_ogrid = _indexer_with_default_outputs(np.ogrid)
    def assertSequenceOfArraysEqual(xs, ys):
      self.assertIsInstance(xs, (list, tuple))
      self.assertIsInstance(ys, (list, tuple))
      self.assertEqual(len(xs), len(ys))
      for x, y in zip(xs, ys):
        self.assertArraysEqual(x, y)

    self.assertArraysEqual(np_ogrid[:5], jnp.ogrid[:5])
    self.assertArraysEqual(np_ogrid[:5], jax.jit(lambda: jnp.ogrid[:5])())
    self.assertArraysEqual(np_ogrid[1:7:2], jnp.ogrid[1:7:2])
    # List of arrays
    assertSequenceOfArraysEqual(np_ogrid[:5,], jnp.ogrid[:5,])
    assertSequenceOfArraysEqual(np_ogrid[0:5, 1:3], jnp.ogrid[0:5, 1:3])
    assertSequenceOfArraysEqual(np_ogrid[1:3:2, 2:9:3], jnp.ogrid[1:3:2, 2:9:3])
    assertSequenceOfArraysEqual(np_ogrid[:5, :9, :11], jnp.ogrid[:5, :9, :11])
    # Corner cases
    self.assertArraysEqual(np_ogrid[:], jnp.ogrid[:])
    # Complex number steps
    atol = 1e-6
    rtol = 1e-6
    self.assertAllClose(np_ogrid[-1:1:5j],
                        jnp.ogrid[-1:1:5j],
                        atol=atol,
                        rtol=rtol)
    # Non-integer steps
    self.assertAllClose(np_ogrid[0:3.5:0.3],
                        jnp.ogrid[0:3.5:0.3],
                        atol=atol,
                        rtol=rtol)
    self.assertAllClose(np_ogrid[1.2:4.8:0.24],
                        jnp.ogrid[1.2:4.8:0.24],
                        atol=atol,
                        rtol=rtol)
    # abstract tracer value for ogrid slice
    with self.assertRaisesRegex(core.ConcretizationTypeError,
                                "slice start of jnp.ogrid"):
      jax.jit(lambda a, b: jnp.ogrid[a:b])(0, 2)

  def testR_(self):
    a = np.arange(6).reshape((2,3))
    self.assertArraysEqual(np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])],
                           jnp.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])])
    self.assertArraysEqual(np.r_['-1', a, a], jnp.r_['-1', a, a])

    self.assertArraysEqual(np.r_['0,2', [1,2,3], [4,5,6]], jnp.r_['0,2', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.r_['0,2,0', [1,2,3], [4,5,6]], jnp.r_['0,2,0', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.r_['1,2,0', [1,2,3], [4,5,6]], jnp.r_['1,2,0', [1,2,3], [4,5,6]])
    # negative 1d axis start
    self.assertArraysEqual(np.r_['0,4,-1', [1,2,3], [4,5,6]], jnp.r_['0,4,-1', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.r_['0,4,-2', [1,2,3], [4,5,6]], jnp.r_['0,4,-2', [1,2,3], [4,5,6]])

    # matrix directives
    with jtu.ignore_warning(category=PendingDeprecationWarning):
      self.assertArraysEqual(np.r_['r',[1,2,3], [4,5,6]], jnp.r_['r',[1,2,3], [4,5,6]])
      self.assertArraysEqual(np.r_['c', [1, 2, 3], [4, 5, 6]], jnp.r_['c', [1, 2, 3], [4, 5, 6]])

    # bad directive
    with self.assertRaisesRegex(ValueError, "could not understand directive.*"):
      jnp.r_["asdfgh",[1,2,3]]
    # abstract tracer value for r_ slice
    with self.assertRaisesRegex(core.ConcretizationTypeError,
                                "slice start of jnp.r_"):
      jax.jit(lambda a, b: jnp.r_[a:b])(0, 2)

    # wrap indexer for appropriate dtype defaults.
    np_r_ = _indexer_with_default_outputs(np.r_)

    # Complex number steps
    atol = 1e-6
    rtol = 1e-6
    self.assertAllClose(np_r_[-1:1:6j],
                        jnp.r_[-1:1:6j],
                        atol=atol,
                        rtol=rtol)
    with jax.numpy_dtype_promotion('standard'):  # Requires dtype promotion.
      self.assertAllClose(np_r_[-1:1:6j, [0]*3, 5, 6],
                          jnp.r_[-1:1:6j, [0]*3, 5, 6],
                          atol=atol,
                          rtol=rtol)
    # Non-integer steps
    self.assertAllClose(np_r_[1.2:4.8:0.24],
                        jnp.r_[1.2:4.8:0.24],
                        atol=atol,
                        rtol=rtol)

  def testC_(self):
    a = np.arange(6).reshape((2, 3))
    self.assertArraysEqual(np.c_[np.array([1,2,3]), np.array([4,5,6])],
                           jnp.c_[np.array([1,2,3]), np.array([4,5,6])])
    self.assertArraysEqual(np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])],
                           jnp.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])])
    self.assertArraysEqual(np.c_['-1', a, a], jnp.c_['-1', a, a])

    self.assertArraysEqual(np.c_['0,2', [1,2,3], [4,5,6]], jnp.c_['0,2', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.c_['0,2,0', [1,2,3], [4,5,6]], jnp.c_['0,2,0', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.c_['1,2,0', [1,2,3], [4,5,6]], jnp.c_['1,2,0', [1,2,3], [4,5,6]])
    # negative 1d axis start
    self.assertArraysEqual(np.c_['0,4,-1', [1,2,3], [4,5,6]], jnp.c_['0,4,-1', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np.c_['0,4,-2', [1,2,3], [4,5,6]], jnp.c_['0,4,-2', [1,2,3], [4,5,6]])
    # matrix directives, avoid numpy deprecation warning
    with jtu.ignore_warning(category=PendingDeprecationWarning):
      self.assertArraysEqual(np.c_['r',[1,2,3], [4,5,6]], jnp.c_['r',[1,2,3], [4,5,6]])
      self.assertArraysEqual(np.c_['c', [1, 2, 3], [4, 5, 6]], jnp.c_['c', [1, 2, 3], [4, 5, 6]])

    # bad directive
    with self.assertRaisesRegex(ValueError, "could not understand directive.*"):
      jnp.c_["asdfgh",[1,2,3]]
    # abstract tracer value for c_ slice
    with self.assertRaisesRegex(core.ConcretizationTypeError,
                                "slice start of jnp.c_"):
      jax.jit(lambda a, b: jnp.c_[a:b])(0, 2)

    # wrap indexer for appropriate dtype defaults.
    np_c_ = _indexer_with_default_outputs(np.c_)

    # Complex number steps
    atol = 1e-6
    rtol = 1e-6
    self.assertAllClose(np_c_[-1:1:6j],
                        jnp.c_[-1:1:6j],
                        atol=atol,
                        rtol=rtol)

    # Non-integer steps
    self.assertAllClose(np_c_[1.2:4.8:0.24],
                        jnp.c_[1.2:4.8:0.24],
                        atol=atol,
                        rtol=rtol)

  def testS_(self):
    self.assertEqual(np.s_[1:2:20],jnp.s_[1:2:20])

  def testIndex_exp(self):
    self.assertEqual(np.index_exp[5:3:2j],jnp.index_exp[5:3:2j])

  @jtu.sample_product(
    start_shape=[(), (2,), (2, 2)],
    stop_shape=[(), (2,), (2, 2)],
    num=[0, 1, 2, 5, 20],
    endpoint=[True, False],
    retstep=[True, False],
    # floating-point compute between jitted platforms and non-jit + rounding
    # cause unavoidable variation in integer truncation for some inputs, so
    # we currently only test inexact 'dtype' arguments.
    dtype=inexact_dtypes + [None,],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLinspace(self, start_shape, stop_shape, num, endpoint, retstep, dtype):
    rng = jtu.rand_default(self.rng())
    # relax default tolerances slightly
    tol = jtu.tolerance(dtype if dtype else np.float32) * 10
    args_maker = self._GetArgsMaker(rng,
                                    [start_shape, stop_shape],
                                    [dtype, dtype])
    start, stop = args_maker()
    ndim = len(np.shape(start + stop))
    for axis in range(-ndim, ndim):
      jnp_op = lambda start, stop: jnp.linspace(
        start, stop, num,
        endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)
      np_op = lambda start, stop: np.linspace(
        start, stop, num,
        endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)

      self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_op, args_maker,
                            check_dtypes=False, atol=tol, rtol=tol)

  @jtu.sample_product(dtype=number_dtypes)
  def testLinspaceEndpoints(self, dtype):
    """Regression test for Issue #3014."""
    rng = jtu.rand_default(self.rng())
    endpoints = rng((2,), dtype)
    out = jnp.linspace(*endpoints, 10, dtype=dtype)
    self.assertAllClose(out[np.array([0, -1])], endpoints, rtol=0, atol=0)

  def testLinspaceArrayNum(self):
    """Regression test for Issue #22405."""
    rng = jtu.rand_default(self.rng())
    endpoints = rng((2,), np.float32)
    # The num parameter is an np.array.
    out = jnp.linspace(*endpoints, np.array(10, dtype=np.int32),
                       dtype=np.float32)
    self.assertAllClose(out[np.array([0, -1])], endpoints, rtol=0, atol=0)

  @jtu.sample_product(
    start_shape=[(), (2,), (2, 2)],
    stop_shape=[(), (2,), (2, 2)],
    num=[0, 1, 2, 5, 20],
    endpoint=[True, False],
    base=[10.0, 2, np.e],
    # skip 16-bit floats due to insufficient precision for the test.
    dtype=jtu.dtypes.inexact + [None,],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLogspace(self, start_shape, stop_shape, num,
                   endpoint, base, dtype):
    if (dtype in int_dtypes and
        jtu.test_device_matches(["gpu", "tpu"]) and
        not config.enable_x64.value):
      raise unittest.SkipTest("GPUx32 truncated exponentiation"
                              " doesn't exactly match other platforms.")
    rng = jtu.rand_default(self.rng())
    # relax default tolerances slightly
    tol = {np.float32: 1e-2, np.float64: 1e-6, np.complex64: 1e-3, np.complex128: 1e-6}
    args_maker = self._GetArgsMaker(rng,
                                    [start_shape, stop_shape],
                                    [dtype, dtype])
    start, stop = args_maker()
    ndim = len(np.shape(start + stop))
    for axis in range(-ndim, ndim):
      jnp_op = lambda start, stop: jnp.logspace(
        start, stop, num, endpoint=endpoint, base=base, dtype=dtype, axis=axis)
      @jtu.ignore_warning(category=RuntimeWarning,
                          message="overflow encountered in power")
      def np_op(start, stop):
        return np.logspace(start, stop, num, endpoint=endpoint,
                           base=base, dtype=dtype, axis=axis)
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      if dtype in (inexact_dtypes + [None,]):
        # Why do compiled and op-by-op float16 np.power numbers differ
        # slightly more than expected?
        atol = {np.float16: 1e-2}
        self._CompileAndCheck(jnp_op, args_maker,
                              check_dtypes=False, atol=atol, rtol=tol)

  @jtu.sample_product(
    [dict(start_shape=start_shape, stop_shape=stop_shape, axis=axis)
      for start_shape in [(), (2,), (2, 2)]
      for stop_shape in [(), (2,), (2, 2)]
      for axis in range(-max(len(start_shape), len(stop_shape)),
                         max(len(start_shape), len(stop_shape)))
    ],
    num=[0, 1, 2, 5, 20],
    endpoint=[True, False],
    # NB: numpy's geomspace gives nonsense results on integer types
    dtype=inexact_dtypes + [None,],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testGeomspace(self, start_shape, stop_shape, num,
                    endpoint, dtype, axis):
    rng = jtu.rand_default(self.rng())
    # relax default tolerances slightly
    tol = {dtypes.bfloat16: 2e-2, np.float16: 4e-3, np.float32: 2e-3,
           np.float64: 1e-14, np.complex64: 2e-3, np.complex128: 1e-14}
    def args_maker():
      """Test the set of inputs np.geomspace is well-defined on."""
      start, stop = self._GetArgsMaker(rng,
                                [start_shape, stop_shape],
                                [dtype, dtype])()
      # np.geomspace can't handle differently ranked tensors
      # w. negative numbers!
      start, stop = jnp.broadcast_arrays(start, stop)
      if dtype in complex_dtypes:
        return start, stop
      # to avoid NaNs, non-complex start and stop cannot
      # differ in sign, elementwise
      start = start * jnp.sign(start) * jnp.sign(stop)
      return start, stop
    start, stop = args_maker()
    def jnp_op(start, stop):
      return jnp.geomspace(start, stop, num, endpoint=endpoint, dtype=dtype,
                           axis=axis)
    def np_op(start, stop):
      start = start.astype(np.float32) if dtype == jnp.bfloat16 else start
      stop = stop.astype(np.float32) if dtype == jnp.bfloat16 else stop
      return np.geomspace(
        start, stop, num, endpoint=endpoint,
        dtype=dtype if dtype != jnp.bfloat16 else np.float32,
        axis=axis).astype(dtype)

    # JAX follows NumPy 2.0 semantics for complex geomspace.
    if not dtypes.issubdtype(dtype, jnp.complexfloating):
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                              check_dtypes=False, tol=tol)
    if dtype in (inexact_dtypes + [None,]):
      self._CompileAndCheck(jnp_op, args_maker,
                            check_dtypes=False, atol=tol, rtol=tol)

  def testDisableNumpyRankPromotionBroadcasting(self):
    with jax.numpy_rank_promotion('allow'):
      jnp.ones(2) + jnp.ones((1, 2))  # works just fine

    with jax.numpy_rank_promotion('raise'):
      self.assertRaises(ValueError, lambda: jnp.ones(2) + jnp.ones((1, 2)))
      jnp.ones(2) + 3  # don't want to raise for scalars

    with jax.numpy_rank_promotion('warn'):
      with self.assertWarnsRegex(
        UserWarning,
        "Following NumPy automatic rank promotion for add on shapes "
        r"\(2,\) \(1, 2\).*"
      ):
        jnp.ones(2) + jnp.ones((1, 2))
      jnp.ones(2) + 3  # don't want to warn for scalars

  @unittest.skip("Test fails on CI, perhaps due to JIT caching")
  def testDisableNumpyRankPromotionBroadcastingDecorator(self):
    with jax.numpy_rank_promotion("allow"):
      jnp.ones(2) + jnp.ones((1, 2))  # works just fine

    with jax.numpy_rank_promotion("raise"):
      self.assertRaises(ValueError, lambda: jnp.ones(2) + jnp.ones((1, 2)))
      jnp.ones(2) + 3  # don't want to raise for scalars

    with jax.numpy_rank_promotion("warn"):
      self.assertWarnsRegex(UserWarning, "Following NumPy automatic rank promotion for add on "
                            r"shapes \(2,\) \(1, 2\).*", lambda: jnp.ones(2) + jnp.ones((1, 2)))
      jnp.ones(2) + 3  # don't want to warn for scalars

  def testStackArrayArgument(self):
    # tests https://github.com/jax-ml/jax/issues/1271
    @jax.jit
    def foo(x):
      return jnp.stack(x)
    foo(np.zeros(2))  # doesn't crash

    @jax.jit
    def foo(x):
      return jnp.concatenate(x)
    foo(np.zeros((2, 2)))  # doesn't crash

  def testReluGradientConstants(self):
    # This is a regression test that verifies that constants associated with the
    # gradient of np.maximum (from lax._balanced_eq) aren't hoisted into the
    # outermost jaxpr. This was producing some large materialized constants for
    # every relu activation in a model.
    def body(i, xy):
      x, y = xy
      y = y + jax.grad(lambda z: jnp.sum(jnp.maximum(z, 0.)))(x)
      return x, y

    f = lambda y: lax.fori_loop(0, 5, body, (y, y))
    jaxpr = jax.make_jaxpr(f)(np.zeros((3, 4), np.float32))
    self.assertFalse(
      any(np.array_equal(x, np.full((3, 4), 2., dtype=np.float32))
          for x in jaxpr.consts))

  @jtu.sample_product(
    [dict(from_shape=from_shape, to_shape=to_shape)
      for from_shape, to_shape in [
          [(1, 3), (4, 3)],
          [(3,), (2, 1, 3)],
          [(3,), (3, 3)],
          [(1,), (3,)],
          [(1,), 3],
      ]
    ],
  )
  def testBroadcastTo(self, from_shape, to_shape):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [from_shape], [np.float32])
    np_op = lambda x: np.broadcast_to(x, to_shape)
    jnp_op = lambda x: jnp.broadcast_to(x, to_shape)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  def testBroadcastToInvalidShape(self):
    # Regression test for https://github.com/jax-ml/jax/issues/20533
    x = jnp.zeros((3, 4, 5))
    with self.assertRaisesRegex(
        ValueError, "Cannot broadcast to shape with fewer dimensions"):
      jnp.broadcast_to(x, (4, 5))

  @jtu.sample_product(
    [dict(shapes=shapes, broadcasted_shape=broadcasted_shape)
      for shapes, broadcasted_shape in [
        [[], ()],
        [[()], ()],
        [[(1, 3), (4, 3)], (4, 3)],
        [[(3,), (2, 1, 3)], (2, 1, 3)],
        [[(3,), (3, 3)], (3, 3)],
        [[(1,), (3,)], (3,)],
        [[(1,), 3], (3,)],
        [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],
        [[[1], [0, 1]], (0, 1)],
        [[(1,), np.array([0, 1])], (0, 1)],
      ]
    ],
  )
  def testBroadcastShapes(self, shapes, broadcasted_shape):
    # Test against np.broadcast_shapes once numpy 1.20 is minimum required version
    np.testing.assert_equal(jnp.broadcast_shapes(*shapes), broadcasted_shape)

  def testBroadcastToIssue1522(self):
    self.assertRaisesRegex(
        ValueError, "Incompatible.*broadcasting: .*",
        lambda: jnp.broadcast_to(np.ones((2, 3)), (1, 3)))

  def testBroadcastToIntIssue1548(self):
    self.assertAllClose(jnp.broadcast_to(1, (3, 2)), np.ones((3, 2)),
                        check_dtypes=False)

  def testBroadcastToOnScalar(self):
    self.assertIsInstance(jnp.broadcast_to(10.0, ()), jax.Array)
    self.assertIsInstance(np.broadcast_to(10.0, ()), np.ndarray)

  def testPrecision(self):

    ones_1d = np.ones((2,))
    ones_2d = np.ones((2, 2))
    ones_3d = np.ones((2, 2, 2))
    HIGHEST = lax.Precision.HIGHEST

    jtu.assert_dot_precision(None, jnp.dot, ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.dot, precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.dot, precision=HIGHEST),
        ones_3d, ones_3d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.matmul, precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.vdot, precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.vecdot, precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.tensordot, axes=2, precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.tensordot, axes=(0, 0), precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.tensordot, axes=((0,), (0,)), precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.einsum, 'i,i', precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.einsum, 'ij,ij', precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(jnp.inner, precision=HIGHEST),
        ones_1d, ones_1d)

  @jtu.sample_product(
      funcname=['inner', 'matmul', 'dot', 'vdot', 'tensordot', 'vecdot']
  )
  def testPreferredElementType(self, funcname):
    func = getattr(jnp, funcname)
    kwargs = dict(axes=0) if funcname == 'tensordot' else {}

    ones_i32 = np.ones(2, dtype='int32')
    ones_f32 = np.ones(2, dtype='float32')

    with jax.numpy_dtype_promotion('strict'):
      jtu.assert_dot_preferred_element_type('int32', func, ones_i32, ones_i32, **kwargs)
      jtu.assert_dot_preferred_element_type('float32', func, ones_f32, ones_f32, **kwargs)
      jtu.assert_dot_preferred_element_type('bfloat16', func, ones_f32, ones_f32, **kwargs,
                                            preferred_element_type='bfloat16')
    with jax.numpy_dtype_promotion('standard'):
      jtu.assert_dot_preferred_element_type('float32', func, ones_i32, ones_f32, **kwargs)

  @jtu.sample_product(
    [dict(shape=shape, varargs=varargs, axis=axis)
        for shape in [(10,), (10, 15), (10, 15, 20)]
        for _num_axes in range(len(shape))
        for varargs in itertools.combinations(range(1, len(shape) + 1), _num_axes)
        for axis in itertools.combinations(range(len(shape)), _num_axes)
    ],
    dtype=inexact_dtypes,
  )
  def testGradient(self, shape, varargs, axis, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_fun = lambda y: jnp.gradient(y, *varargs, axis=axis)
    np_fun = lambda y: np.gradient(y, *varargs, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      shape=[(5,), (5, 7), (5, 10, 3)],
      dtype=inexact_dtypes,
  )
  def testGradientNonConstant(self, shape, dtype):
    rng = jtu.rand_default(self.rng())

    varargs = [(s,) for s in shape]
    args = [shape] + varargs
    args_maker = self._GetArgsMaker(rng, args, [dtype] * len(args))
    atol = jtu.tolerance(
        dtype, {np.float16: 4e-2, jax.dtypes.bfloat16: 4e-1, np.float32: 2e-5}
    )
    rtol = jtu.tolerance(dtype, {jax.dtypes.bfloat16: 5e-1})
    self._CheckAgainstNumpy(
        np.gradient,
        jnp.gradient,
        args_maker,
        check_dtypes=False,
        atol=atol,
        rtol=rtol,
    )
    self._CompileAndCheck(jnp.gradient, args_maker)

  def testZerosShapeErrors(self):
    # see https://github.com/jax-ml/jax/issues/1822
    self.assertRaisesRegex(
        TypeError,
        "Shapes must be 1D sequences of concrete values of integer type.*",
        lambda: jnp.zeros(1.))
    self.assertRaisesRegex(
        TypeError,
        r"Shapes must be 1D sequences of concrete values of integer type.*\n"
        "If using `jit`, try using `static_argnums` or applying `jit` to "
        "smaller subfunctions.",
        lambda: jax.jit(jnp.zeros)(2))

  def testTraceMethod(self):
    x = self.rng().randn(3, 4).astype(float)
    self.assertAllClose(x.trace(), jnp.array(x).trace())
    self.assertAllClose(x.trace(), jax.jit(lambda y: y.trace())(x))

  @jtu.ignore_warning(category=RuntimeWarning, message="divide by zero")
  def testIntegerPowersArePrecise(self):
    # See https://github.com/jax-ml/jax/pull/3036
    # Checks if the squares of float32 integers have no numerical errors.
    # It should be satisfied with all integers less than sqrt(2**24).
    x = jnp.arange(-2**12, 2**12, dtype=jnp.int32)
    np.testing.assert_array_equal(jnp.square(x.astype(jnp.float32)), x * x)
    np.testing.assert_array_equal(x.astype(jnp.float32) ** 2, x * x)

    # Similarly for cubes.
    x = jnp.arange(-2**8, 2**8, dtype=jnp.int32)
    np.testing.assert_array_equal(x.astype(jnp.float32) ** 3, x * x * x)

    x = np.arange(10, dtype=np.float32)
    for i in range(10):
      self.assertAllClose(x.astype(jnp.float32) ** i, x ** i,
                          check_dtypes=False)

  def testToBytes(self):
    v = np.arange(12, dtype=np.int32).reshape(3, 4)
    for order in ['C', 'F']:
      self.assertEqual(jnp.asarray(v).tobytes(order), v.tobytes(order))

  def testToBytesJitError(self):
    v = np.arange(12, dtype=np.int32).reshape(3, 4)
    f = jax.jit(lambda x: x.tobytes())
    msg = r".*The tobytes\(\) method was called on traced array"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(v)

  def testToList(self):
    v = np.arange(12, dtype=np.int32).reshape(3, 4)
    self.assertEqual(jnp.asarray(v).tolist(), v.tolist())

  def testToListJitError(self):
    v = np.arange(12, dtype=np.int32).reshape(3, 4)
    f = jax.jit(lambda x: x.tolist())
    msg = r".*The tolist\(\) method was called on traced array"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(v)

  def testArangeConcretizationError(self):
    msg = r"It arose in the jnp.arange argument '{}'".format
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg('stop')):
      jax.jit(jnp.arange)(3)

    with self.assertRaisesRegex(core.ConcretizationTypeError, msg('start')):
      jax.jit(lambda start: jnp.arange(start, 3))(0)

    with self.assertRaisesRegex(core.ConcretizationTypeError, msg('stop')):
      jax.jit(lambda stop: jnp.arange(0, stop))(3)

  def testIssue2347(self):
    # https://github.com/jax-ml/jax/issues/2347
    object_list = list[tuple[jnp.array, float, float, jnp.array, bool]]
    self.assertRaises(TypeError, jnp.array, object_list)

    np_object_list = np.array(object_list)
    self.assertRaises(TypeError, jnp.array, np_object_list)

  @jtu.sample_product(
    [dict(shapes=shapes, dtypes=dtypes)
      for shapes in filter(
        _shapes_are_broadcast_compatible,
        itertools.combinations_with_replacement(all_shapes, 2))
      for dtypes in itertools.product(
        *(_valid_dtypes_for_shape(s, complex_dtypes) for s in shapes))
    ],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLogaddexpComplex(self, shapes, dtypes):
    @jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")
    def np_op(x1, x2):
      return np.log(np.exp(x1) + np.exp(x2))

    rng = jtu.rand_some_nan(self.rng())
    args_maker = lambda: tuple(rng(shape, dtype) for shape, dtype in zip(shapes, dtypes))
    if jtu.test_device_matches(["tpu"]):
      tol = {np.complex64: 1e-3, np.complex128: 1e-10}
    else:
      tol = {np.complex64: 1e-5, np.complex128: 1e-14}

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(jtu.promote_like_jnp(np_op), jnp.logaddexp, args_maker, tol=tol)
      self._CompileAndCheck(jnp.logaddexp, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(
    [dict(shapes=shapes, dtypes=dtypes)
      for shapes in filter(
        _shapes_are_broadcast_compatible,
        itertools.combinations_with_replacement(all_shapes, 2))
      for dtypes in itertools.product(
        *(_valid_dtypes_for_shape(s, complex_dtypes) for s in shapes))
    ],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLogaddexp2Complex(self, shapes, dtypes):
    @jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")
    def np_op(x1, x2):
      return np.log2(np.exp2(x1) + np.exp2(x2))

    rng = jtu.rand_some_nan(self.rng())
    args_maker = lambda: tuple(rng(shape, dtype) for shape, dtype in zip(shapes, dtypes))
    if jtu.test_device_matches(["tpu"]):
      tol = {np.complex64: 1e-3, np.complex128: 1e-10}
    else:
      tol = {np.complex64: 1e-5, np.complex128: 1e-14}

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(jtu.promote_like_jnp(np_op), jnp.logaddexp2, args_maker, tol=tol)
      self._CompileAndCheck(jnp.logaddexp2, args_maker, rtol=tol, atol=tol)

  def testDefaultDtypes(self):
    self.assertEqual(jnp.bool_, np.bool_)
    self.assertEqual(jnp.int_, np.int64)
    self.assertEqual(jnp.uint, np.uint64)
    self.assertEqual(jnp.float_, np.float64)
    self.assertEqual(jnp.complex_, np.complex128)

  def testFromBuffer(self):
    buf = b'\x01\x02\x03'
    expected = np.frombuffer(buf, dtype='uint8')
    actual = jnp.frombuffer(buf, dtype='uint8')
    self.assertArraysEqual(expected, actual)

  def testFromFunction(self):
    def f(x, y, z):
      return x + 2 * y + 3 * z
    shape = (3, 4, 5)
    expected = np.fromfunction(f, shape=shape)
    actual = jnp.fromfunction(f, shape=shape)
    self.assertArraysEqual(expected, actual, check_dtypes=False)

  def testFromString(self):
    s = "1,2,3"
    expected = np.fromstring(s, sep=',', dtype=int)
    actual = jnp.fromstring(s, sep=',', dtype=int)
    self.assertArraysEqual(expected, actual)

  @jtu.sample_product(
      a_shape=nonempty_nonscalar_array_shapes,
      v_shape=nonempty_shapes,
      dtype=jtu.dtypes.all,
  )
  def testPlace(self, a_shape, v_shape, dtype):
    rng = jtu.rand_default(self.rng())
    mask_rng = jtu.rand_bool(self.rng())

    def args_maker():
      a = rng(a_shape, dtype)
      m = mask_rng(a_shape, bool)
      v = rng(v_shape, dtype)
      return a, m, v

    def np_fun(a, m, v):
      a_copy = a.copy()
      np.place(a_copy, m, v)
      return a_copy

    jnp_fun = partial(jnp.place, inplace=False)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
      a_shape=nonempty_nonscalar_array_shapes,
      i_shape=all_shapes,
      v_shape=all_shapes,
      dtype=jtu.dtypes.all,
      mode=[None, 'wrap', 'clip'],
  )
  def testPut(self, mode, a_shape, i_shape, v_shape, dtype):
    size = math.prod(a_shape)
    if math.prod(i_shape) > size:
      self.skipTest("too many indices")
    rng = jtu.rand_default(self.rng())
    # Must test unique integers, because overlapping updates in
    # JAX have implementation-defined order
    idx_rng = jtu.rand_unique_int(self.rng(), size)

    def args_maker():
      a = rng(a_shape, dtype)
      i = idx_rng(i_shape, np.int32)
      v = rng(v_shape, dtype)
      # put some indices out of range without duplicating indices
      if mode == "clip" and i.size:
        np.put(i, np.argmax(i), size + 2)
        np.put(i, np.argmin(i), -2)
      if mode == "wrap" and i.size:
        np.put(i, 0, np.take(i, 0) + size)
      return a, i, v

    def np_fun(a, i, v):
      a_copy = a.copy()
      np.put(a_copy, i, v, mode=mode)
      return a_copy

    jnp_fun = partial(jnp.put, mode=mode, inplace=False)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @jtu.sample_product(
    [
      dict(a_shape=a_shape, i_shape=i_shape, v_shape=v_shape, axis=axis)
      for a_shape in nonempty_array_shapes
      for axis in list(range(-len(a_shape), len(a_shape)))
      for i_shape in [
        tuple_update(a_shape, axis if axis >= 0 else axis + len(a_shape), J)
        for J in range(a_shape[axis] + 1)
      ]
      for v_shape in [(), (1,), i_shape]
    ] + [
      dict(a_shape=a_shape, i_shape=i_shape, v_shape=v_shape, axis=None)
      for a_shape in nonempty_array_shapes
      for i_shape in [(J,) for J in range(math.prod(a_shape) + 1)]
      for v_shape in [(), (1,), i_shape]
    ],
    dtype=jtu.dtypes.all,
    mode=[None, "promise_in_bounds", "clip"],
  )
  def testPutAlongAxis(self, a_shape, i_shape, v_shape, axis, dtype, mode):
    a_rng = jtu.rand_default(self.rng())
    if axis is None:
      size = math.prod(a_shape)
    else:
      size = a_shape[axis]
    i_rng = jtu.rand_indices_unique_along_axis(self.rng())

    def args_maker():
      a = a_rng(a_shape, dtype)
      i = i_rng(dim=size, shape=i_shape, axis=0 if axis is None else axis)
      v = a_rng(v_shape, dtype)
      return a, i, v

    def np_fun(a, i, v):
      a_copy = a.copy()
      np.put_along_axis(a_copy, i, v, axis=axis)
      return a_copy

    jnp_fun = partial(jnp.put_along_axis, axis=axis, inplace=False, mode=mode)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def test_rot90_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "rot90 requires its first argument to have ndim at least two, "
        "but got first argument of"):
      jnp.rot90(jnp.ones(2))

  @parameterized.named_parameters(
      ('ones', jnp.ones),
      ('zeros', jnp.zeros),
      ('empty', jnp.empty))
  def test_error_hint(self, fn):
    with self.assertRaisesRegex(
        TypeError,
        r"Did you accidentally write `jax\.numpy\..*?\(2, 3\)` "
        r"when you meant `jax\.numpy\..*?\(\(2, 3\)\)`"):
      fn(2, 3)

  @jtu.sample_product(
      dtype=jtu.dtypes.all,
      kind=['bool', 'signed integer', 'unsigned integer', 'integral',
            'real floating', 'complex floating', 'numeric']
  )
  def test_isdtype(self, dtype, kind):
    # Full tests also in dtypes_test.py; here we just compare against numpy
    jax_result = jnp.isdtype(dtype, kind)
    if dtype == dtypes.bfloat16:
      # just a smoke test
      self.assertIsInstance(jax_result, bool)
    else:
      numpy_result = np.isdtype(dtype, kind)
      self.assertEqual(jax_result, numpy_result)

  @jtu.sample_product(
    [dict(yshape=yshape, xshape=xshape, dx=dx, axis=axis)
      for yshape, xshape, dx, axis in [
        ((10,), None, 1.0, -1),
        ((3, 10), None, 2.0, -1),
        ((3, 10), None, 3.0, -0),
        ((10, 3), (10,), 1.0, -2),
        ((3, 10), (10,), 1.0, -1),
        ((3, 10), (3, 10), 1.0, -1),
        ((2, 3, 10), (3, 10), 1.0, -2),
      ]
    ],
    dtype=float_dtypes + int_dtypes,
  )
  @jtu.skip_on_devices("tpu")  # TODO(jakevdp): fix and re-enable this test.
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def test_trapezoid(self, yshape, xshape, dtype, dx, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(yshape, dtype), rng(xshape, dtype) if xshape is not None else None]
    np_fun = partial(np.trapezoid, dx=dx, axis=axis)
    jnp_fun = partial(jnp.trapezoid, dx=dx, axis=axis)
    tol = jtu.tolerance(
        dtype,
        {
            np.float16: 4e-3,
            np.float32: 5e-6,
            np.float64: 1e-12,
            jax.dtypes.bfloat16: 4e-2,
        },
    )
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol,
                            check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol,
                          check_dtypes=False)

  @jtu.sample_product(
    [dict(yshape=yshape, dxshape=dxshape, axis=axis)
      for yshape, dxshape, axis in [
        ((3, 10), (3, 1), -1),
        ((10,), (1,), -1),
        ((2, 3, 4, 10), (2, 3, 4, 1), -1),
        ((2, 3), (3,), 0),
        ((3, 4, 5), (4, 5), 0),
        ((3, 4, 5), (3, 5), 1),
        ((3, 4, 5), (5,), 1),
        ((2, 3, 4, 5), (2, 4, 5), 1),
        ((2, 3, 4, 5), (2, 1, 3, 5), 2),
        # slow path: varies along integration axis
        ((3, 4, 5), (2, 4, 5), 0),
        ((3, 4, 5), (3, 3, 5), 1),
      ]
    ],
  )
  @jtu.skip_on_devices("tpu")
  def test_trapezoid_array_dx(self, yshape, dxshape, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(yshape, np.float32), rng(dxshape, np.float32)]
    np_fun = lambda y, dx: np.trapezoid(y, dx=dx, axis=axis)
    jnp_fun = lambda y, dx: jnp.trapezoid(y, dx=dx, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=1e-5,
                            check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, atol=1e-5, rtol=1e-5,
                          check_dtypes=False)

  @jtu.sample_product(
      shape=all_shapes,
      dtype=default_dtypes,
      op=['ndim', 'shape', 'size'],
  )
  def testNdimShapeSize(self, shape, dtype, op):
    rng = jtu.rand_default(self.rng())
    jnp_op = getattr(jnp, op)
    np_op = getattr(np, op)
    x = rng(shape, dtype)
    expected = np_op(x)
    self.assertEqual(expected, jnp_op(x))  # np.ndarray or scalar input.
    self.assertEqual(expected, jnp_op(jnp.asarray(x)))  # jax.Array input.
    self.assertEqual(expected, jax.jit(jnp_op)(x))  # Traced input.

  @jtu.sample_product(
      shape=nonzerodim_shapes,
      dtype=default_dtypes,
  )
  def testSizeAlongAxis(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    axis = self.rng().randint(-len(shape), len(shape))
    np_op = partial(np.size, axis=axis)
    jnp_op = partial(jnp.size, axis=axis)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
      tuple_size=[0, 1, 2, 3, 4]
  )
  def testSizeAlongAxisTuple(self, tuple_size):
    rng = self.rng()

    ndim = tuple_size + rng.randint(10)

    shape = rng.randint(0, 10, ndim)
    tuples = list(itertools.combinations(range(ndim), tuple_size))
    axis = tuples[rng.randint(len(tuples))]

    array = jnp.zeros(shape)
    output = jnp.size(array, axis)
    expected = math.prod(shape[i] for i in axis)
    assert output == expected

  @jtu.sample_product(
    axis=[(0, 0), (0, -3), (1, 1), (1, -2), (2, 2), (2, -1)],
  )
  def testSizeAlongAxisDuplicate(self, axis):
    shape = (2, 3, 4)
    array = jnp.zeros(shape)
    msg = "repeated axis"
    with self.assertRaisesRegex(ValueError, msg):
      jnp.size(array, axis)

  @jtu.sample_product(
      op=[jnp.ndim, jnp.shape, jnp.size],
  )
  def testNdimShapeSizeNonArrayInput(self, op):
    msg = f"{op.__name__} requires ndarray or scalar arguments"
    with self.assertWarnsRegex(DeprecationWarning, msg):
      op([1, 2, 3])

  @jtu.run_on_devices("cpu")
  def testLog2Accuracy(self):
    # Regression test for https://github.com/jax-ml/jax/issues/40419
    exponents_f32 = np.arange(-10, 11)
    values_f32 = np.ldexp(1.0, exponents_f32).astype(jnp.float32)
    self.assertArraysEqual(jnp.log2(values_f32), exponents_f32.astype(jnp.float32))

    exponents_bf16 = np.array([-5, 5, 10, 19, 20, 38, 40, 53, 57, 67, 76, 80, 89, 93, 106, 114, 119, 127])
    values_bf16 = np.ldexp(1.0, exponents_bf16).astype(jnp.bfloat16)
    self.assertArraysEqual(jnp.log2(values_bf16), exponents_bf16.astype(jnp.bfloat16))

  @jtu.run_on_devices("cpu")
  def testLog10Accuracy(self):
    # Using precomputed 1/log(10) instead of dividing by log(10) improves accuracy.
    exponents_f32 = np.array([-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
    values_f32 = np.power(10.0, exponents_f32).astype(np.float32)
    self.assertArraysEqual(jnp.log10(values_f32), exponents_f32.astype(np.float32))

    exponents_bf16 = np.array([-7, 7, 9, 11, 14, 17, 18, 22, 23, 27, 28, 29, 31, 34, 36])
    values_bf16 = np.power(10.0, exponents_bf16).astype(jnp.bfloat16)
    self.assertArraysEqual(jnp.log10(values_bf16), exponents_bf16.astype(jnp.bfloat16))


# Most grad tests are at the lax level (see lax_test.py), but we add some here
# as needed for e.g. particular compound ops of interest.

GradTestSpec = collections.namedtuple(
    "GradTestSpec",
    ["op", "nargs", "order", "rng_factory", "dtypes", "name", "tol"])
def grad_test_spec(op, nargs, order, rng_factory, dtypes, name=None, tol=None):
  return GradTestSpec(
      op, nargs, order, rng_factory, dtypes, name or op.__name__, tol)

GRAD_TEST_RECORDS = [
    grad_test_spec(jnp.arcsinh, nargs=1, order=2,
                   rng_factory=jtu.rand_positive,
                   dtypes=[np.float64, np.complex64],
                   tol={np.complex64: 2e-2}),
    grad_test_spec(jnp.arccosh, nargs=1, order=1,
                   rng_factory=jtu.rand_positive,
                   dtypes=[np.float64, np.complex64],
                   tol={np.complex64: 2e-2}),
    grad_test_spec(jnp.arctanh, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-0.9, high=0.9),
                   dtypes=[np.float64, np.complex64],
                   tol={np.complex64: 2e-2}),
    grad_test_spec(jnp.logaddexp, nargs=2, order=1,
                   rng_factory=partial(jtu.rand_uniform, low=-0.9, high=0.9),
                   dtypes=[np.float64], tol=1e-4),
    grad_test_spec(jnp.logaddexp2, nargs=2, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-0.9, high=0.9),
                   dtypes=[np.float64], tol=1e-4),
]

GradSpecialValuesTestSpec = collections.namedtuple(
    "GradSpecialValuesTestSpec", ["op", "values", "order"])

GRAD_SPECIAL_VALUE_TEST_RECORDS = [
    GradSpecialValuesTestSpec(jnp.arcsinh, [0., 1000.], 2),
    GradSpecialValuesTestSpec(jnp.arccosh, [1000.], 2),
    GradSpecialValuesTestSpec(jnp.arctanh, [0.], 2),
    GradSpecialValuesTestSpec(jnp.sinc, [0.], 1),
]


class NumpyGradTests(jtu.JaxTestCase):

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(op=rec.op, rng_factory=rec.rng_factory, tol=rec.tol,
            order=rec.order)],
      shapes=itertools.combinations_with_replacement(nonempty_shapes, rec.nargs),
      dtype=rec.dtypes)
    for rec in GRAD_TEST_RECORDS))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises mixed type promotion
  def testOpGrad(self, op, rng_factory, shapes, dtype, order, tol):
    rng = rng_factory(self.rng())
    tol = jtu.join_tolerance(tol, {np.float32: 1e-1, np.float64: 1e-3,
                                   np.complex64: 1e-1, np.complex128: 1e-3})
    if jtu.test_device_matches(["tpu"]) and op == jnp.arctanh:
      tol = jtu.join_tolerance(tol, {np.float32: 2e-1})

    args = tuple(rng(shape, dtype) for shape in shapes)
    check_grads(op, args, order, ["fwd", "rev"], tol, tol)

  @parameterized.parameters(itertools.chain.from_iterable(
      jtu.sample_product_testcases(
        [dict(op=rec.op, order=rec.order)],
        special_value=rec.values
      )
      for rec in GRAD_SPECIAL_VALUE_TEST_RECORDS))
  def testOpGradSpecialValue(self, op, special_value, order):
    check_grads(
        op, (special_value,), order, ['fwd', 'rev'], atol={np.float32: 3.4e-3}
    )

  def testSincAtZero(self):
    # Some manual tests for sinc at zero, since it doesn't have well-behaved
    # numerical derivatives at zero
    def deriv(f):
      return lambda x: jax.jvp(f, (x,), (1.,))[1]

    def apply_all(fns, x):
      for f in fns:
        x = f(x)
      return x

    d1 = 0.
    for ops in itertools.combinations_with_replacement([deriv, jax.grad], 1):
      self.assertAllClose(apply_all(ops, jnp.sinc)(0.), d1)

    d2 = -np.pi ** 2 / 3
    for ops in itertools.combinations_with_replacement([deriv, jax.grad], 2):
      self.assertAllClose(apply_all(ops, jnp.sinc)(0.), d2)

    d3 = 0.
    for ops in itertools.combinations_with_replacement([deriv, jax.grad], 3):
      self.assertAllClose(apply_all(ops, jnp.sinc)(0.), d3)

    d4 = np.pi ** 4 / 5
    for ops in itertools.combinations_with_replacement([deriv, jax.grad], 4):
      self.assertAllClose(apply_all(ops, jnp.sinc)(0.), d4)

  def testSincGradArrayInput(self):
    # tests for a bug almost introduced in #5077
    jax.grad(lambda x: jnp.sinc(x).sum())(jnp.arange(10.))  # doesn't crash

  def testTakeAlongAxisIssue1521(self):
    # https://github.com/jax-ml/jax/issues/1521
    idx = jnp.repeat(jnp.arange(3), 10).reshape((30, 1))

    def f(x):
      y = x * jnp.arange(3.).reshape((1, 3))
      return jnp.take_along_axis(y, idx, -1).sum()

    check_grads(f, (1.,), order=1)

  @jtu.sample_product(
    shapes=filter(_shapes_are_broadcast_compatible,
                  itertools.combinations_with_replacement(nonempty_shapes, 2)),
    dtype=(np.complex128,),
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testGradLogaddexpComplex(self, shapes, dtype):
    rng = jtu.rand_default(self.rng())
    args = tuple(jnp.array(rng(shape, dtype)) for shape in shapes)
    if jtu.test_device_matches(["tpu"]):
      tol = 5e-2
    else:
      tol = 3e-2
    check_grads(jnp.logaddexp, args, 1, ["fwd", "rev"], tol, tol)

  @jtu.sample_product(
    shapes=filter(_shapes_are_broadcast_compatible,
                  itertools.combinations_with_replacement(nonempty_shapes, 2)),
    dtype=(np.complex128,),
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testGradLogaddexp2Complex(self, shapes, dtype):
    rng = jtu.rand_default(self.rng())
    args = tuple(jnp.array(rng(shape, dtype)) for shape in shapes)
    if jtu.test_device_matches(["tpu"]):
      tol = 5e-2
    else:
      tol = 3e-2
    check_grads(jnp.logaddexp2, args, 1, ["fwd", "rev"], tol, tol)

  @jtu.sample_product(
    n=range(-4, 5),
    dtype=[jnp.float32, jnp.float64],
  )
  def testGradLdexp(self, n, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng((10,), dtype)
    check_grads(lambda x: jnp.ldexp(x, n), (x,), 2, ["fwd", "rev"])
    n_arr = jnp.full((10,), n, dtype=jnp.int32)
    check_grads(lambda x: jnp.ldexp(x, n_arr), (x,), 2, ["fwd", "rev"])
    _, f_vjp = jax.vjp(jnp.ldexp, x, n_arr)
    x_bar, n_bar = f_vjp(jnp.ones_like(x))
    self.assertAllClose(x_bar, jnp.full_like(x, 2.0 ** n))
    self.assertEqual(n_bar.dtype, dtypes.float0)

  @parameterized.parameters(jnp.float32, jnp.float64)
  def testGradLdexpZero(self, dtype):
    if dtype == jnp.float64 and not config.enable_x64.value:
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    # Derivative of x * 2^n at x = 0.0 must be 2^n (previously returned 1.0).
    for n in [-3, 0, 3, 10]:
      g = jax.grad(lambda x: jnp.ldexp(x, n))(dtype(0.0))
      expected = dtype(2.0 ** n)
      self.assertEqual(g, expected)
      check_grads(lambda x: jnp.ldexp(x, n), (dtype(0.0),), 2, ["fwd", "rev"])

  @jtu.sample_product(
    n=range(-4, 5),
    dtype=[jnp.float32, jnp.float64],
  )
  def testGradFrexp(self, n, dtype):
    rng = jtu.rand_default(self.rng())
    x = rng((10,), dtype) * 2 ** n
    check_grads(lambda x: jnp.frexp(x)[0], (x,), 2, ["fwd", "rev"])
    (m, e), f_vjp = jax.vjp(jnp.frexp, x)
    x_bar, = f_vjp((jnp.ones_like(m), jnp.zeros_like(e, dtype=dtypes.float0)))
    self.assertAllClose(x_bar, jnp.ldexp(jnp.ones_like(m), -e))


class NumpySignaturesTest(jtu.JaxTestCase):

  def testWrappedSignaturesMatch(self):
    """Test that jax.numpy function signatures match numpy."""
    # NumPy functions explicitly not implemented in JAX:
    skip = {'array2string',
            'asanyarray',
            'asarray_chkfinite',
            'ascontiguousarray',
            'asfortranarray',
            'asmatrix',
            'base_repr',
            'binary_repr',
            'bmat',
            'broadcast',
            'busday_count',
            'busday_offset',
            'busdaycalendar',
            'common_type',
            'copyto',
            'datetime_as_string',
            'datetime_data',
            'errstate',
            'fix',
            'flatiter',
            'format_float_positional',
            'format_float_scientific',
            'fromregex',
            'genfromtxt',
            'get_include',
            'getbufsize',
            'geterr',
            'geterrcall',
            'in1d',
            'info',
            'is_busday',
            'isfortran',
            'isnat',
            'loadtxt',
            'matrix',
            'may_share_memory',
            'memmap',
            'min_scalar_type',
            'mintypecode',
            'ndenumerate',
            'ndindex',
            'nditer',
            'nested_iters',
            'poly1d',
            'putmask',
            'real_if_close',
            'recarray',
            'record',
            'require',
            'row_stack',
            'savetxt',
            'savez_compressed',
            'setbufsize',
            'seterr',
            'seterrcall',
            'shares_memory',
            'show_config',
            'show_runtime',
            'test',
            'trapz',
            'typename'}

    self.assertEmpty(skip.intersection(dir(jnp)))

    names = (name for name in dir(np) if not (name.startswith('_') or name in skip))
    names = (name for name in names if callable(getattr(np, name)))
    names = {name for name in names if not isinstance(getattr(np, name), type)}
    self.assertEmpty(names.difference(dir(jnp)))

    self.assertNotEmpty(names)

    # TODO(jakevdp): fix some of the following signatures. Some are due to wrong argument names.
    unsupported_params = {
      'arange': ['start_or_stop', 'like'],
      'array': ['ndmax', 'like', 'subok'],
      'argpartition': ['kind', 'order'],
      'asarray': ['like'],
      'broadcast_to': ['subok'],
      'clip': ['kwargs', 'out'],
      'concat': ['out', 'dtype', 'casting'],
      'concatenate': ['out', 'casting'],
      'copy': ['subok'],
      'corrcoef': ['ddof', 'bias'],
      'cumulative_prod': ['out'],
      'cumulative_sum': ['out'],
      'dot': ['out'],
      'empty_like': ['subok', 'order'],
      'einsum': ['kwargs'],
      'einsum_path': ['einsum_call'],
      'empty': ['order', 'like'],
      'eye': ['order', 'like'],
      'hstack': ['casting'],
      'identity': ['like'],
      'isin': ['kind'],
      'full': ['order', 'like'],
      'full_like': ['subok', 'order'],
      'frombuffer': ['like'],
      'fromfunction': ['like'],
      'frompyfunc': ['kwargs'],
      'fromstring': ['like'],
      'load': ['mmap_mode', 'allow_pickle', 'fix_imports', 'encoding', 'max_header_size'],
      'nanpercentile': ['interpolation'],
      'nanquantile': ['interpolation'],
      'nanstd': ['correction'],
      'nanvar': ['correction'],
      'ones': ['order', 'like'],
      'ones_like': ['subok', 'order'],
      'partition': ['kind', 'order'],
      'percentile': ['interpolation'],
      'promote_types': ['type1', 'type2'],
      'quantile': ['interpolation'],
      'row_stack': ['casting'],
      'stack': ['casting'],
      'tri': ['like'],
      'unravel_index': ['order'],
      'vstack': ['casting'],
      'zeros': ['order', 'like'],
      'zeros_like': ['subok', 'order']
    }

    extra_params = {
      'arange': ['start'],
      'array': ['args', 'device', 'out_sharding'],
      'compress': ['size', 'fill_value'],
      'einsum': ['subscripts', 'precision'],
      'einsum_path': ['subscripts'],
      'fill_diagonal': ['inplace'],
      'load': ['args', 'kwargs'],
      'take_along_axis': ['mode', 'fill_value'],
      'unique': ['size', 'fill_value'],
    }

    mismatches = {}

    for name in names:
      jnp_fun = getattr(jnp, name)
      np_fun = getattr(np, name)
      if isinstance(getattr(np, name), np.ufunc):
        # Skip all `np.ufunc`s since many of the missing ufunc keywords may not
        # be relevant for JAX. However, args such as `axis` and `keepdims` may
        # be useful to `matmul` and others.
        continue
      if name in ['histogram', 'histogram2d', 'histogramdd']:
        # numpy 1.24 re-orders the density and weights arguments.
        # TODO(jakevdp): migrate histogram APIs to match newer numpy versions.
        continue
      if name == "clip":
        # JAX's support of the Array API spec for clip, and the way it handles
        # backwards compatibility was introduced in
        # https://github.com/jax-ml/jax/pull/20550 with a different signature
        # from the one in numpy, introduced in
        # https://github.com/numpy/numpy/pull/26724
        # TODO(dfm): After our deprecation period for the clip arguments ends
        # it should be possible to reintroduce the check.
        continue
      if name == "reshape":
        # Similar issue to clip: we'd need logic specific to the NumPy version
        # because of the change in argument name from `newshape` to `shape`.
        continue
      if name == "asarray":
        # The order of the `device` and `copy` kwargs are swapped between jnp
        # and np.
        # jnp.asarray: a, dtype, order, copy, device, out_sharding
        # np.asarray: a, dtype, order, device, copy, like
        continue
      # Note: can't use inspect.getfullargspec for some functions due to numpy issue
      # https://github.com/numpy/numpy/issues/12225
      try:
        np_params = inspect.signature(np_fun).parameters
      except ValueError:
        continue
      jnp_params = inspect.signature(jnp_fun).parameters
      extra = set(extra_params.get(name, []))
      unsupported = set(unsupported_params.get(name, []))

      # Checks to prevent tests from becoming out-of-date. If these fail,
      # it means that extra_params or unsupported_params need to be updated.
      assert extra.issubset(jnp_params), f"{name}: {extra=} is not a subset of jnp_params={set(jnp_params)}."
      assert not unsupported.intersection(jnp_params), f"{name}: {unsupported=} overlaps with jnp_params={set(jnp_params)}."

      # Skip functions that only have *args and **kwargs; we can't introspect these further.
      var_args = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
      if all(p.kind in var_args for p in jnp_params.values()):
        continue
      if all(p.kind in var_args for p in np_params.values()):
        continue

      # Remove known extra parameters.
      jnp_params = {a: p for a, p in jnp_params.items() if a not in extra}

      # Remove known unsupported parameters.
      np_params = {a: p for a, p in np_params.items() if a not in unsupported}

      # Older versions of numpy may have fewer parameters; to avoid extraneous errors on older numpy
      # versions, we allow for jnp to have more parameters.
      if list(jnp_params)[:len(np_params)] != list(np_params):
        mismatches[name] = {'np_params': list(np_params), 'jnp_params': list(jnp_params)}

    self.assertEqual(mismatches, {})


_available_numpy_dtypes: list[str] = [dtype.__name__ for dtype in jtu.dtypes.all
                                      if dtype != dtypes.bfloat16]

# TODO(jakevdp): implement missing ufuncs.
UNIMPLEMENTED_UFUNCS = {'spacing', 'matvec', 'vecmat'}


def _all_numpy_ufuncs() -> Iterator[str]:
  """Generate the names of all ufuncs in the top-level numpy namespace."""
  for name in dir(np):
    f = getattr(np, name)
    if isinstance(f, np.ufunc) and name not in UNIMPLEMENTED_UFUNCS:
      yield name


def _dtypes_for_ufunc(name: str) -> Iterator[tuple[str, ...]]:
  """Generate valid dtypes of inputs to the given numpy ufunc."""
  func = getattr(np, name)
  for arg_dtypes in itertools.product(_available_numpy_dtypes, repeat=func.nin):
    args = (np.ones(1, dtype=dtype) for dtype in arg_dtypes)
    try:
      with jtu.ignore_warning(
          category=RuntimeWarning, message="(divide by zero|invalid value)"):
        _ = func(*args)
    except TypeError:
      pass
    else:
      yield arg_dtypes


class NumpyUfuncTests(jtu.JaxTestCase):

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases([dict(name=name)],
                                 arg_dtypes=_dtypes_for_ufunc(name))
    for name in _all_numpy_ufuncs()
  ))
  def testUfuncInputTypes(self, name, arg_dtypes):
    if name in ['arctanh', 'atanh'] and jnp.issubdtype(arg_dtypes[0], jnp.complexfloating):
      self.skipTest("np.arctanh & jnp.arctanh have mismatched NaNs for complex input.")

    jnp_op = getattr(jnp, name)
    np_op = getattr(np, name)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="(divide by zero|invalid value)")(np_op)
    args_maker = lambda: tuple(np.ones(1, dtype=dtype) for dtype in arg_dtypes)

    with jtu.strict_promotion_if_dtypes_match(arg_dtypes):
      # large tol comes from the fact that numpy returns float16 in places
      # that jnp returns float32. e.g. np.cos(np.uint8(0))
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker, check_dtypes=False, tol=1E-2)


class NumpyDocTests(jtu.JaxTestCase):

  def test_lax_numpy_docstrings(self):
    unimplemented = ['fromfile', 'fromiter']
    aliases = ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'atan2',
               'amax', 'amin', 'around', 'bitwise_invert', 'bitwise_left_shift',
               'bitwise_not','bitwise_right_shift', 'conj', 'degrees', 'divide',
               'get_printoptions', 'mod', 'pow', 'printoptions', 'radians', 'round_',
               'set_printoptions']
    skip_args_check = ['vsplit', 'hsplit', 'dsplit', 'array_split']

    for name in dir(jnp):
      if name.startswith('_') or name in unimplemented:
        continue

      obj = getattr(jnp, name)

      if isinstance(obj, type) or not callable(obj):
        # Skip docstring checks for non-functions
        pass
      elif hasattr(np, name) and obj is getattr(np, name):
        # Some APIs are imported directly from NumPy; we don't check these.
        pass
      elif name in aliases:
        assert "Alias of" in obj.__doc__
      elif name not in skip_args_check:
        # Other functions should have nontrivial docs including "Args" and "Returns".
        doc = obj.__doc__
        self.assertNotEmpty(doc)
        self.assertIn("Args:", doc, msg=f"'Args:' not found in docstring of jnp.{name}")
        self.assertIn("Returns:", doc, msg=f"'Returns:' not found in docstring of jnp.{name}")
        if name not in ["frompyfunc", "isdtype", "promote_types"]:
          self.assertIn("Examples:", doc, msg=f"'Examples:' not found in docstring of jnp.{name}")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
