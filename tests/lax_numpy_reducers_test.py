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


import collections
from functools import partial
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import numpy as jnp

from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.util import NumpyComplexWarning

config.parse_flags_with_absl()

numpy_version = jtu.numpy_version()

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
one_dim_array_shapes = [(1,), (6,), (12,)]
empty_array_shapes = [(0,), (0, 4), (3, 0),]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE, jtu.PYTHON_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes + empty_array_shapes
nonzerodim_shapes = nonempty_nonscalar_array_shapes + empty_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes = scalar_shapes + array_shapes

custom_float_dtypes = jtu.dtypes.custom_floats
float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
unsigned_dtypes = jtu.dtypes.all_unsigned
bool_dtypes = jtu.dtypes.boolean
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes + unsigned_dtypes
all_dtypes = number_dtypes + bool_dtypes

python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

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

def _get_y_shapes(y_dtype, shape, rowvar):
  # Helper function for testCov.
  if y_dtype is None:
    return [None]
  if len(shape) == 1:
    return [shape]
  elif rowvar or shape[0] == 1:
    return [(1, shape[-1]), (2, shape[-1]), (5, shape[-1])]
  return [(shape[0], 1), (shape[0], 2), (shape[0], 5)]


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

JAX_REDUCER_RECORDS = [
    op_record("mean", 1, number_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("nanmean", 1, inexact_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
    op_record("nanprod", 1, all_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("nansum", 1, number_dtypes, all_shapes, jtu.rand_some_nan, []),
]

JAX_REDUCER_INITIAL_RECORDS = [
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, [],
              tolerance={jnp.bfloat16: 2e-2}),
    op_record("max", 1, all_dtypes + custom_float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("min", 1, all_dtypes + custom_float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("nanprod", 1, inexact_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("nansum", 1, inexact_dtypes, all_shapes, jtu.rand_default, [],
              tolerance={jnp.bfloat16: 3e-2}),
    op_record("nanmax", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
    op_record("nanmin", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
]

JAX_REDUCER_WHERE_NO_INITIAL_RECORDS = [
    op_record("all", 1, bool_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("any", 1, bool_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("mean", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("var", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("std", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("nanmean", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True, tolerance={np.float16: 3e-3}),
    op_record("nanvar", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True, tolerance={np.float16: 3e-3}),
    op_record("nanstd", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True, tolerance={np.float16: 1e-3}),
]

JAX_REDUCER_NO_DTYPE_RECORDS = [
    op_record("all", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("any", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("max", 1, all_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("min", 1, all_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("var", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True, tolerance={jnp.bfloat16: 2e-2}),
    op_record("std", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("nanmax", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanmin", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanvar", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
    op_record("nanstd", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan,
              [], inexact=True),
    op_record("ptp", 1, number_dtypes, nonempty_shapes, jtu.rand_default, []),
]

JAX_REDUCER_PROMOTE_INT_RECORDS = [
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
]

def _reducer_output_dtype(name: str, input_dtype: np.dtype, promote_integers: bool = True) -> np.dtype:
  if name in ['sum', 'prod', 'nansum', 'nanprod']:
    if input_dtype == bool:
      input_dtype = dtypes.to_numeric_dtype(input_dtype)
    if promote_integers:
      if dtypes.issubdtype(input_dtype, np.integer):
        default_int = dtypes.canonicalize_dtype(
            dtypes.uint if dtypes.issubdtype(input_dtype, np.unsignedinteger) else dtypes.int_)
        if np.iinfo(input_dtype).bits < np.iinfo(default_int).bits:
          return default_int
  return input_dtype


class JaxNumpyReducerTests(jtu.JaxTestCase):
  """Tests for LAX-backed Numpy reduction operations."""

  def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
    def f():
      out = [rng(shape, dtype or jnp.float_)
             for shape, dtype in zip(shapes, dtypes)]
      if np_arrays:
        return out
      return [jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
              for a in out]
    return f

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, inexact=rec.inexact)],
      [dict(shape=shape, axis=axis, dtype=dtype)
        for shape in rec.shapes
        for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        if jtu.is_valid_shape(shape, dtype)
      ],
      out_dtype=[out_dtype for out_dtype in [None] + rec.dtypes
                  if out_dtype not in unsigned_dtypes],
      keepdims=[False, True],
    )
    for rec in JAX_REDUCER_RECORDS
  ))
  def testReducer(self, name, rng_factory, shape, dtype, out_dtype,
                  axis, keepdims, inexact):
    np_op = getattr(np, name)
    jnp_op = getattr(jnp, name)
    rng = rng_factory(self.rng())
    @jtu.ignore_warning(category=NumpyComplexWarning)
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="mean of empty slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="overflow encountered.*")
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if dtype != jnp.bfloat16 else x.astype(np.float32)
      t = out_dtype if out_dtype != jnp.bfloat16 else np.float32
      if t is None:
        t = _reducer_output_dtype(name, x_cast.dtype)
      return np_op(x_cast, axis, dtype=t, keepdims=keepdims)

    jnp_fun = lambda x: jnp_op(x, axis, dtype=out_dtype, keepdims=keepdims)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol_spec = {np.float16: 1e-2, np.int16: 2e-7, np.int32: 1E-3,
                np.uint32: 3e-7, np.float32: 1e-3, np.complex64: 1e-3,
                np.float64: 1e-5, np.complex128: 1e-5}
    tol = jtu.tolerance(dtype, tol_spec)
    tol = max(tol, jtu.tolerance(out_dtype, tol_spec)) if out_dtype else tol
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=jnp.bfloat16 not in (dtype, out_dtype),
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, inexact=rec.inexact,
            tolerance=rec.tolerance)],
      [dict(shape=shape, axis=axis, dtype=dtype)
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        if jtu.is_valid_shape(shape, dtype)
      ],
      keepdims=[False, True],
    )
    for rec in JAX_REDUCER_NO_DTYPE_RECORDS
  ))
  def testReducerNoDtype(self, name, rng_factory, shape, dtype, axis,
                         keepdims, inexact, tolerance):
    np_op = getattr(np, name)
    jnp_op = getattr(jnp, name)
    rng = rng_factory(self.rng())
    is_bf16_nan_test = (dtype == jnp.bfloat16 and
                        rng_factory.__name__ == 'rand_some_nan')
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="All-NaN (slice|axis) encountered.*")
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims)
    args_maker = lambda: [rng(shape, dtype)]
    tol = jtu.join_tolerance({np.float16: 0.002},
                             tolerance or jtu.default_tolerance())
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=tol, atol=tol)

  @jtu.sample_product(rec = JAX_REDUCER_INITIAL_RECORDS)
  def testReducerBadInitial(self, rec):
    jnp_op = getattr(jnp, rec.name)
    arr = jnp.ones((2, 3, 4))
    initial = jnp.zeros((1, 2, 3))
    msg = r"initial value must be a scalar. Got array of shape \(1, 2, 3\)"
    with self.assertRaisesRegex(ValueError, msg):
      jnp_op(arr, axis=-1, initial=initial)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, inexact=rec.inexact)],
      [dict(shape=shape, axis=axis, dtype=dtype)
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        if jtu.is_valid_shape(shape, dtype)
      ],
      initial=[0, 1],
      keepdims=[False, True],
    )
    for rec in JAX_REDUCER_INITIAL_RECORDS
  ))
  def testReducerInitial(self, name, rng_factory, shape, dtype, axis,
                         keepdims, initial, inexact):
    np_op = getattr(np, name)
    jnp_op = getattr(jnp, name)
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=NumpyComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, initial=initial)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res.astype(_reducer_output_dtype(name, x.dtype))

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 3E-2}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=tol, atol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, inexact=rec.inexact)],
      [dict(shape=shape, axis=axis, dtype=dtype)
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        if jtu.is_valid_shape(shape, dtype)
      ],
      initial=[0, 1],
      keepdims=[False, True],
      promote_integers=[False, True],
    )
    for rec in JAX_REDUCER_PROMOTE_INT_RECORDS
  ))
  def testReducerPromoteInt(self, name, rng_factory, shape, dtype, axis,
                            keepdims, initial, inexact, promote_integers):
    np_op = getattr(np, name)
    jnp_op = getattr(jnp, name)
    rng = rng_factory(self.rng())
    is_bf16_nan_test = (dtype == jnp.bfloat16 and
                        rng_factory.__name__ == 'rand_some_nan')
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=NumpyComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, initial=initial)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res.astype(_reducer_output_dtype(name, x.dtype, promote_integers))

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial, promote_integers=promote_integers)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 3E-2}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, inexact=rec.inexact)],
      [dict(shape=shape, axis=axis)
        for shape in rec.shapes if np.prod(shape) == 0
        for axis in range(-len(shape), len(shape)) if shape[axis] >= 1
      ],
      dtype=rec.dtypes,
      keepdims=[False, True],
    )
    for rec in JAX_REDUCER_INITIAL_RECORDS
  ))
  def testReducerNoInitialZeroDims(self, name, rng_factory, shape, dtype, axis,
                                   keepdims, inexact):
    np_op = getattr(np, name)
    jnp_op = getattr(jnp, name)
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=NumpyComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res.astype(_reducer_output_dtype(name, x.dtype))

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 3E-2}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, inexact=rec.inexact,
            tol=rec.tolerance)],
      [dict(shape=shape, axis=axis, dtype=dtype, whereshape=whereshape)
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        if jtu.is_valid_shape(shape, dtype)
        for whereshape in _compatible_shapes(shape)
      ],
      initial=[0, 1],
      keepdims=[False, True],
    )
    for rec in JAX_REDUCER_INITIAL_RECORDS
  ))
  def testReducerWhere(self, name, rng_factory, shape, dtype, axis,
                       keepdims, initial, inexact, whereshape, tol):
    np_op = getattr(np, name)
    jnp_op = getattr(jnp, name)
    if (shape in [()] + scalar_shapes and
        dtype in [jnp.int16, jnp.uint16] and
        jnp_op in [jnp.min, jnp.max]):
      self.skipTest("Known XLA failure; see https://github.com/google/jax/issues/4971.")
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    # Do not pass where via args_maker as that is incompatible with _promote_like_jnp.
    where = jtu.rand_bool(self.rng())(whereshape, np.bool_)
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=NumpyComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, initial=initial, where=where)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res.astype(_reducer_output_dtype(name, x.dtype))

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial, where=where)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, atol=tol, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, inexact=rec.inexact,
            tol=rec.tolerance)],
      [dict(shape=shape, axis=axis, dtype=dtype, whereshape=whereshape)
        for shape in rec.shapes for dtype in rec.dtypes
        for whereshape in _compatible_shapes(shape)
        for axis in list(range(-len(shape), len(shape))) + [None]
        if jtu.is_valid_shape(shape, dtype)
      ],
      keepdims=[False, True],
    ) for rec in JAX_REDUCER_WHERE_NO_INITIAL_RECORDS
  ))
  def testReducerWhereNoInitial(self, name, rng_factory, shape, dtype, axis,
                                keepdims, inexact, whereshape, tol):
    np_op = getattr(np, name)
    jnp_op = getattr(jnp, name)
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16
    # Do not pass where via args_maker as that is incompatible with _promote_like_jnp.
    where = jtu.rand_bool(self.rng())(whereshape, np.bool_)
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Mean of empty slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="invalid value encountered.*")
    @jtu.ignore_warning(category=NumpyComplexWarning)
    def np_fun(x):
      x = np.asarray(x)
      if inexact:
        x = x.astype(dtypes.to_inexact_dtype(x.dtype))
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, where=where)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res

    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, where=where)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, atol=tol, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testReductionOfOutOfBoundsAxis(self):  # Issue 888
    x = jnp.ones((3, 4))
    self.assertRaises(ValueError, lambda: jnp.sum(x, axis=2))

  def testReductionWithRepeatedAxisError(self):
    with self.assertRaisesRegex(ValueError, r"duplicate value in 'axis': \(0, 0\)"):
      jnp.sum(jnp.arange(3), (0, 0))

  @jtu.sample_product(
    [dict(shape=shape, dtype=dtype, axis=axis, weights_shape=weights_shape)
      for shape, dtype in _shape_and_dtypes(nonempty_shapes, number_dtypes)
      for axis in list(range(-len(shape), len(shape))) + [None] + [tuple(range(len(shape)))]
      # `weights_shape` is either `None`, same as the averaged axis, or same as
      # that of the input
      for weights_shape in ([None, shape] if axis is None or len(shape) == 1 or isinstance(axis, tuple)
                            else [None, (shape[axis],), shape])
    ],
    keepdims=[False, True],
    returned=[False, True],
  )
  def testAverage(self, shape, dtype, axis, weights_shape, returned, keepdims):
    rng = jtu.rand_default(self.rng())
    kwds = dict(returned=returned)
    if keepdims is not None:
      kwds['keepdims'] = keepdims
    if weights_shape is None:
      np_fun = lambda x: np.average(x, axis, **kwds)
      jnp_fun = lambda x: jnp.average(x, axis, **kwds)
      args_maker = lambda: [rng(shape, dtype)]
    else:
      np_fun = lambda x, weights: np.average(x, axis, weights, **kwds)
      jnp_fun = lambda x, weights: jnp.average(x, axis, weights, **kwds)
      args_maker = lambda: [rng(shape, dtype), rng(weights_shape, dtype)]
    np_fun = jtu.promote_like_jnp(np_fun, inexact=True)
    tol = {dtypes.bfloat16: 2e-1, np.float16: 1e-2, np.float32: 1e-5,
           np.float64: 1e-12, np.complex64: 1e-5}
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    if numpy_version == (1, 23, 0) and keepdims and weights_shape is not None and axis is not None:
      # Known failure: https://github.com/numpy/numpy/issues/21850
      pass
    else:
      try:
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                                check_dtypes=check_dtypes, tol=tol)
      except ZeroDivisionError:
        self.skipTest("don't support checking for ZeroDivisionError")
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=check_dtypes,
                          rtol=tol, atol=tol)

  @jtu.sample_product(
    shape=[(5,), (10, 5)],
    dtype=all_dtypes,
    out_dtype=inexact_dtypes,
    axis=[None, 0, -1],
    ddof=[0, 1, 2],
    keepdims=[False, True],
  )
  def testVar(self, shape, dtype, out_dtype, axis, ddof, keepdims):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.")
    @jtu.ignore_warning(category=NumpyComplexWarning)
    def np_fun(x):
      # Numpy fails with bfloat16 inputs
      out = np.var(x.astype(np.float32 if dtype == dtypes.bfloat16 else dtype),
                   dtype=np.float32 if out_dtype == dtypes.bfloat16 else out_dtype,
                   axis=axis, ddof=ddof, keepdims=keepdims)
      return out.astype(out_dtype)
    jnp_fun = partial(jnp.var, dtype=out_dtype, axis=axis, ddof=ddof, keepdims=keepdims)
    tol = jtu.tolerance(out_dtype, {np.float16: 1e-1, np.float32: 1e-3,
                                    np.float64: 1e-3, np.complex128: 1e-6})
    if (jnp.issubdtype(dtype, jnp.complexfloating) and
        not jnp.issubdtype(out_dtype, jnp.complexfloating)):
      self.assertRaises(ValueError, lambda: jnp_fun(*args_maker()))
    else:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                              tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker, rtol=tol,
                            atol=tol)

  @jtu.sample_product(
    shape=[(5,), (10, 5)],
    dtype=all_dtypes,
    out_dtype=inexact_dtypes,
    axis=[None, 0, -1],
    ddof=[0, 1, 2],
    keepdims=[False, True],
  )
  def testNanVar(self, shape, dtype, out_dtype, axis, ddof, keepdims):
    rng = jtu.rand_some_nan(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.")
    @jtu.ignore_warning(category=NumpyComplexWarning)
    def np_fun(x):
      # Numpy fails with bfloat16 inputs
      out = np.nanvar(x.astype(np.float32 if dtype == dtypes.bfloat16 else dtype),
                      dtype=np.float32 if out_dtype == dtypes.bfloat16 else out_dtype,
                      axis=axis, ddof=ddof, keepdims=keepdims)
      return out.astype(out_dtype)
    jnp_fun = partial(jnp.nanvar, dtype=out_dtype, axis=axis, ddof=ddof, keepdims=keepdims)
    tol = jtu.tolerance(out_dtype, {np.float16: 1e-1, np.float32: 1e-3,
                                    np.float64: 1e-3, np.complex64: 1e-3,
                                    np.complex128: 5e-4})
    if (jnp.issubdtype(dtype, jnp.complexfloating) and
        not jnp.issubdtype(out_dtype, jnp.complexfloating)):
      self.assertRaises(ValueError, lambda: jnp_fun(*args_maker()))
    else:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                              tol=tol)
      self._CompileAndCheck(jnp_fun, args_maker, rtol=tol,
                            atol=tol)

  def testNanStdGrad(self):
    # Regression test for https://github.com/google/jax/issues/8128
    x = jnp.arange(5.0).at[0].set(jnp.nan)
    y = jax.grad(jnp.nanvar)(x)
    self.assertAllClose(y, jnp.array([0.0, -0.75, -0.25, 0.25, 0.75]), check_dtypes=False)

    z = jax.grad(jnp.nanstd)(x)
    self.assertEqual(jnp.isnan(z).sum(), 0)

  @jtu.sample_product(
    [dict(shape=shape, dtype=dtype, y_dtype=y_dtype, rowvar=rowvar,
          y_shape=y_shape)
      for shape in [(5,), (10, 5), (5, 10)]
      for dtype in all_dtypes
      for y_dtype in [None, dtype]
      for rowvar in [True, False]
      for y_shape in _get_y_shapes(y_dtype, shape, rowvar)
    ],
    bias=[True, False],
    ddof=[None, 2, 3],
    fweights=[True, False],
    aweights=[True, False],
  )
  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises mixed type promotion
  @jax.default_matmul_precision('float32')
  def testCov(self, shape, dtype, y_shape, y_dtype, rowvar, ddof, bias, fweights, aweights):
    rng = jtu.rand_default(self.rng())
    wrng = jtu.rand_positive(self.rng())
    wdtype = np.real(dtype(0)).dtype
    wshape = shape[-1:] if rowvar or shape[0] == 1 else shape[:1]

    args_maker = lambda: [rng(shape, dtype),
                          rng(y_shape, y_dtype) if y_dtype else None,
                          wrng(wshape, int) if fweights else None,
                          wrng(wshape, wdtype) if aweights else None]
    kwargs = dict(rowvar=rowvar, ddof=ddof, bias=bias)
    np_fun = lambda m, y, f, a: np.cov(m, y, fweights=f, aweights=a, **kwargs)
    jnp_fun = lambda m, y, f, a: jnp.cov(m, y, fweights=f, aweights=a, **kwargs)
    tol = {jnp.bfloat16: 5E-2, np.float16: 1E-2, np.float32: 1e-5,
           np.float64: 1e-13, np.complex64: 1e-5, np.complex128: 1e-13}
    tol = jtu.join_tolerance(tol, jtu.tolerance(dtype))
    self._CheckAgainstNumpy(
        np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)

  @jtu.sample_product(
    [dict(op=op, q_rng=q_rng)
      for (op, q_rng) in (
        ("percentile", partial(jtu.rand_uniform, low=0., high=100.)),
        ("quantile", partial(jtu.rand_uniform, low=0., high=1.)),
        ("nanpercentile", partial(jtu.rand_uniform, low=0., high=100.)),
        ("nanquantile", partial(jtu.rand_uniform, low=0., high=1.)),
      )
    ],
    [dict(a_shape=a_shape, axis=axis)
      for a_shape, axis in (
        ((7,), None),
        ((47, 7), 0),
        ((47, 7), ()),
        ((4, 101), 1),
        ((4, 47, 7), (1, 2)),
        ((4, 47, 7), (0, 2)),
        ((4, 47, 7), (1, 0, 2)),
      )
    ],
    a_dtype=default_dtypes,
    q_dtype=[np.float32],
    q_shape=scalar_shapes + [(1,), (4,)],
    keepdims=[False, True],
    method=['linear', 'lower', 'higher', 'nearest', 'midpoint'],
  )
  def testQuantile(self, op, q_rng, a_shape, a_dtype, q_shape, q_dtype,
                   axis, keepdims, method):
    a_rng = jtu.rand_some_nan(self.rng())
    q_rng = q_rng(self.rng())
    if "median" in op:
      args_maker = lambda: [a_rng(a_shape, a_dtype)]
    else:
      args_maker = lambda: [a_rng(a_shape, a_dtype), q_rng(q_shape, q_dtype)]

    @jtu.ignore_warning(category=RuntimeWarning,
                        message="All-NaN slice encountered")
    def np_fun(*args):
      args = [x if jnp.result_type(x) != jnp.bfloat16 else
              np.asarray(x, np.float32) for x in args]
      return getattr(np, op)(*args, axis=axis, keepdims=keepdims,
                             method=method)
    jnp_fun = partial(getattr(jnp, op), axis=axis, keepdims=keepdims,
                      method=method)

    # TODO(phawkins): we currently set dtype=False because we aren't as
    # aggressive about promoting to float64. It's not clear we want to mimic
    # Numpy here.
    tol_spec = {np.float16: 1E-2, np.float32: 2e-4, np.float64: 5e-6}
    tol = max(jtu.tolerance(a_dtype, tol_spec),
              jtu.tolerance(q_dtype, tol_spec))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=tol)

  @unittest.skipIf(not config.enable_x64.value, "test requires X64")
  @jtu.run_on_devices("cpu")  # test is for CPU float64 precision
  def testPercentilePrecision(self):
    # Regression test for https://github.com/google/jax/issues/8513
    x = jnp.float64([1, 2, 3, 4, 7, 10])
    self.assertEqual(jnp.percentile(x, 50), 3.5)

  @jtu.sample_product(
    [dict(a_shape=a_shape, axis=axis)
      for a_shape, axis in (
        ((7,), None),
        ((47, 7), 0),
        ((4, 101), 1),
      )
    ],
    a_dtype=default_dtypes,
    keepdims=[False, True],
    op=["median", "nanmedian"],
  )
  def testMedian(self, op, a_shape, a_dtype, axis, keepdims):
    if op == "median":
      a_rng = jtu.rand_default(self.rng())
    else:
      a_rng = jtu.rand_some_nan(self.rng())
    args_maker = lambda: [a_rng(a_shape, a_dtype)]
    def np_fun(*args):
      args = [x if jnp.result_type(x) != jnp.bfloat16 else
              np.asarray(x, np.float32) for x in args]
      return getattr(np, op)(*args, axis=axis, keepdims=keepdims)
    jnp_fun = partial(getattr(jnp, op), axis=axis, keepdims=keepdims)
    # TODO(phawkins): we currently set dtype=False because we aren't as
    # aggressive about promoting to float64. It's not clear we want to mimic
    # Numpy here.
    tol_spec = {np.float32: 2e-4, np.float64: 5e-6}
    tol = jtu.tolerance(a_dtype, tol_spec)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=tol)

  def testMeanLargeArray(self):
    # https://github.com/google/jax/issues/15068
    raise unittest.SkipTest("test is slow, but it passes!")
    x = jnp.ones((16, 32, 1280, 4096), dtype='int8')
    self.assertEqual(1.0, jnp.mean(x))
    self.assertEqual(1.0, jnp.mean(x, where=True))

  def testStdLargeArray(self):
    # https://github.com/google/jax/issues/15068
    raise unittest.SkipTest("test is slow, but it passes!")
    x = jnp.ones((16, 32, 1280, 4096), dtype='int8')
    self.assertEqual(0.0, jnp.std(x))
    self.assertEqual(0.0, jnp.std(x, where=True))

  @jtu.sample_product(
      dtype=[np.dtype(np.float16), np.dtype(dtypes.bfloat16)],
  )
  def test_f16_mean(self, dtype):
    x = np.full(100_000, 1E-5, dtype=dtype)
    expected = np.mean(x.astype('float64')).astype(dtype)
    actual = jnp.mean(x)
    self.assertAllClose(expected, actual, atol=0)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
