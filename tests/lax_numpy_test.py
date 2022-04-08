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


import collections
import functools
from functools import partial
import inspect
import io
import itertools
import operator
from typing import cast, Iterator, Optional, List, Tuple
import unittest
from unittest import SkipTest
import warnings

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
try:
  import numpy_dispatch
except ImportError:
  numpy_dispatch = None

import jax
import jax.ops
from jax import lax
from jax import numpy as jnp
from jax import tree_util
from jax.test_util import check_grads

from jax._src import device_array
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.lax import lax as lax_internal
from jax._src.numpy.lax_numpy import _promote_dtypes, _promote_dtypes_inexact
from jax._src.numpy.util import _parse_numpydoc, ParsedDoc, _wraps
from jax._src.util import prod, safe_zip

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

numpy_version = tuple(map(int, np.__version__.split('.')[:3]))

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
one_dim_array_shapes = [(1,), (6,), (12,)]
empty_array_shapes = [(0,), (0, 4), (3, 0),]

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


python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

# uint64 is problematic because with any uint type it promotes to float:
int_dtypes_no_uint64 = [d for d in int_dtypes + unsigned_dtypes if d != np.uint64]

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
  if shape in scalar_shapes or np.ndim(shape) == 0:
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

JAX_ONE_TO_ONE_OP_RECORDS = [
    op_record("abs", 1, all_dtypes,
              all_shapes, jtu.rand_default, ["rev"]),
    op_record("add", 2, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("ceil", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("ceil", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_default, [], check_dtypes=False),
    op_record("conj", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("exp", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("fabs", 1, float_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("float_power", 2, inexact_dtypes, all_shapes,
              partial(jtu.rand_default, scale=1), ["rev"],
              tolerance={jnp.bfloat16: 1e-2, np.float32: 1e-3,
                         np.float64: 1e-12, np.complex64: 2e-4,
                         np.complex128: 1e-12}, check_dtypes=False),
    op_record("floor", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("floor", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_default, [], check_dtypes=False),
    op_record("greater", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("greater_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("i0", 1, float_dtypes, all_shapes, jtu.rand_default, [],
              check_dtypes=False),
    op_record("ldexp", 2, int_dtypes, all_shapes, jtu.rand_default, [], check_dtypes=False),
    op_record("less", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("less_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("log", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("logical_and", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_not", 1, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_or", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_xor", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("maximum", 2, all_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("minimum", 2, all_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("multiply", 2, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("negative", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("nextafter", 2, [f for f in float_dtypes if f != jnp.bfloat16],
              all_shapes, jtu.rand_default, ["rev"], inexact=True, tolerance=0),
    op_record("not_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, ["rev"]),
    op_record("array_equal", 2, number_dtypes, all_shapes, jtu.rand_some_equal, ["rev"]),
    op_record("array_equiv", 2, number_dtypes, all_shapes, jtu.rand_some_equal, ["rev"]),
    op_record("reciprocal", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
    op_record("subtract", 2, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("signbit", 1, default_dtypes + bool_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"]),
    op_record("trunc", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("trunc", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, [], check_dtypes=False),
    op_record("sin", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("cos", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("tan", 1, number_dtypes, all_shapes,
              partial(jtu.rand_uniform, low=-1.5, high=1.5), ["rev"],
              inexact=True),
    op_record("sinh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("cosh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    # TODO(b/142975473): on CPU, tanh for complex128 is only accurate to
    # ~float32 precision.
    # TODO(b/143135720): on GPU, tanh has only ~float32 precision.
    op_record("tanh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              tolerance={np.float64: 1e-7, np.complex128: 1e-7},
              inexact=True),
    op_record("arcsin", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arccos", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arctan", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arctan2", 2, float_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arcsinh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True, tolerance={np.complex64: 2E-4, np.complex128: 2E-14}),
    op_record("arccosh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True, tolerance={np.complex64: 2E-2, np.complex128: 2E-12}),
    op_record("arctanh", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, tolerance={np.float64: 1e-9}),
]

JAX_COMPOUND_OP_RECORDS = [
    # angle has inconsistent 32/64-bit return types across numpy versions.
    op_record("angle", 1, number_dtypes, all_shapes, jtu.rand_default, [],
              check_dtypes=False, inexact=True),
    op_record("angle", 1, number_dtypes, all_shapes, jtu.rand_default, [],
              check_dtypes=False, inexact=True, test_name="angle_deg", kwargs={'deg': True}),
    op_record("atleast_1d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("atleast_2d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("atleast_3d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("cbrt", 1, default_dtypes, all_shapes, jtu.rand_some_inf, ["rev"],
              inexact=True),
    op_record("conjugate", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("deg2rad", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("divide", 2, number_dtypes, all_shapes, jtu.rand_nonzero, ["rev"],
              inexact=True),
    op_record("divmod", 2, int_dtypes + float_dtypes, all_shapes,
              jtu.rand_nonzero, []),
    op_record("exp2", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              tolerance={jnp.bfloat16: 4e-2, np.float16: 1e-2}, inexact=True),
    # TODO(b/142975473): on CPU, expm1 for float64 is only accurate to ~float32
    # precision.
    op_record("expm1", 1, number_dtypes, all_shapes, jtu.rand_positive, [],
              test_name="expm1_large", tolerance={np.float64: 1e-8}, inexact=True),
    op_record("expm1", 1, number_dtypes, all_shapes, jtu.rand_small_positive,
              [], tolerance={np.float64: 1e-8}, inexact=True),
    op_record("fix", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("fix", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_default, [], check_dtypes=False),
    op_record("floor_divide", 2, default_dtypes + unsigned_dtypes,
              all_shapes, jtu.rand_nonzero, ["rev"]),
    op_record("fmin", 2, number_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("fmax", 2, number_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("fmod", 2, default_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("heaviside", 2, default_dtypes, all_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("hypot", 2, default_dtypes, all_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("kron", 2, number_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("outer", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("imag", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("iscomplex", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("isfinite", 1, inexact_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isinf", 1, inexact_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isnan", 1, inexact_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isneginf", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isposinf", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isreal", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("isrealobj", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("log2", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("log10", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("log1p", 1, number_dtypes, all_shapes, jtu.rand_positive, [],
              test_name="log1p_large", tolerance={np.float64: 1e-12},
              inexact=True),
    op_record("log1p", 1, number_dtypes, all_shapes, jtu.rand_small_positive, [],
              tolerance={np.float64: 1e-12}, inexact=True),
    op_record("logaddexp", 2, float_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"],
              tolerance={np.float64: 1e-12}, inexact=True),
    op_record("logaddexp2", 2, float_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"],
              tolerance={np.float16: 1e-2, np.float64: 2e-14}, inexact=True),
    op_record("polyval", 2, number_dtypes, nonempty_nonscalar_array_shapes,
              jtu.rand_default, [], check_dtypes=False,
              tolerance={dtypes.bfloat16: 4e-2, np.float16: 1e-2,
                         np.float64: 1e-12}),
    op_record("positive", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("power", 2, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              tolerance={np.complex128: 1e-14}, check_dtypes=False),
    op_record("rad2deg", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("ravel", 1, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("real", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("remainder", 2, default_dtypes, all_shapes, jtu.rand_nonzero, [],
              tolerance={np.float16: 1e-2}),
    op_record("mod", 2, default_dtypes, all_shapes, jtu.rand_nonzero, []),
    op_record("modf", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("modf", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_default, [], check_dtypes=False),
    op_record("rint", 1, inexact_dtypes, all_shapes, jtu.rand_some_inf_and_nan,
              []),
    op_record("rint", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_default, [], check_dtypes=False),
    op_record("sign", 1, number_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    # numpy 1.16 has trouble mixing uint and bfloat16, so we test these separately.
    op_record("copysign", 2, default_dtypes + unsigned_dtypes,
              all_shapes, jtu.rand_some_inf_and_nan, [], check_dtypes=False),
    op_record("sinc", 1, [t for t in number_dtypes if t != jnp.bfloat16],
              all_shapes, jtu.rand_default, ["rev"],
              tolerance={np.complex64: 1e-5}, inexact=True,
              check_dtypes=False),
    op_record("square", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("sqrt", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("transpose", 1, all_dtypes, all_shapes, jtu.rand_default, ["rev"],
              check_dtypes=False),
    op_record("true_divide", 2, all_dtypes, all_shapes, jtu.rand_nonzero,
              ["rev"], inexact=True),
    op_record("ediff1d", 3, [np.int32], all_shapes, jtu.rand_default, [], check_dtypes=False),
    # TODO(phawkins): np.unwrap does not correctly promote its default period
    # argument under NumPy 1.21 for bfloat16 inputs. It works fine if we
    # explicitly pass a bfloat16 value that does not need promition. We should
    # probably add a custom test harness for unwrap that tests the period
    # argument anyway.
    op_record("unwrap", 1, [t for t in float_dtypes if t != dtypes.bfloat16],
              nonempty_nonscalar_array_shapes,
              jtu.rand_default, ["rev"],
              # numpy.unwrap always returns float64
              check_dtypes=False,
              # numpy cumsum is inaccurate, see issue #3517
              tolerance={dtypes.bfloat16: 1e-1, np.float16: 1e-1}),
    op_record("isclose", 2, [t for t in all_dtypes if t != jnp.bfloat16],
              all_shapes, jtu.rand_small_positive, []),
    op_record("gcd", 2, int_dtypes_no_uint64, all_shapes, jtu.rand_default, []),
    op_record("lcm", 2, int_dtypes_no_uint64, all_shapes, jtu.rand_default, []),
]

JAX_BITWISE_OP_RECORDS = [
    op_record("bitwise_and", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
    op_record("bitwise_not", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
    op_record("invert", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
    op_record("bitwise_or", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
    op_record("bitwise_xor", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_bool, []),
]

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
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("max", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("min", 1, all_dtypes, all_shapes, jtu.rand_default, []),
]
if numpy_version >= (1, 22):  # initial & where keywords added in numpy 1.22
  JAX_REDUCER_INITIAL_RECORDS += [
      op_record("nanprod", 1, inexact_dtypes, all_shapes, jtu.rand_small_positive, []),
      op_record("nansum", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
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
]
if numpy_version >= (1, 22):  # where keyword added in numpy 1.22
  JAX_REDUCER_WHERE_NO_INITIAL_RECORDS += [
      op_record("nanmean", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
                inexact=True),
      op_record("nanvar", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
                inexact=True),
      op_record("nanstd", 1, inexact_dtypes, nonempty_shapes, jtu.rand_default, [],
                inexact=True),
  ]

JAX_REDUCER_NO_DTYPE_RECORDS = [
    op_record("all", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("any", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("max", 1, all_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("min", 1, all_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("var", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
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

JAX_ARGMINMAX_RECORDS = [
    op_record("argmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("argmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("nanargmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanargmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
]

JAX_OPERATOR_OVERLOADS = [
    op_record("__add__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__sub__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__mul__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__eq__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__ne__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__lt__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__le__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__gt__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__ge__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__pos__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__neg__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__pow__", 2, inexact_dtypes, all_shapes, jtu.rand_positive, [],
              tolerance={np.float32: 2e-4, np.complex64: 2e-4, np.complex128: 1e-14}),
    op_record("__mod__", 2, default_dtypes, all_shapes, jtu.rand_nonzero, [],
              tolerance={np.float16: 1e-1}),
    op_record("__floordiv__", 2, default_dtypes, all_shapes,
              jtu.rand_nonzero, []),
    op_record("__truediv__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, [],
              inexact=True),
    op_record("__abs__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    # TODO(mattjj): __invert__ fails on bool dtypes because ~True == -2
    op_record("__invert__", 1, int_dtypes, all_shapes, jtu.rand_default, []),
    # TODO(mattjj): investigate these failures
    # op_record("__or__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__and__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    # op_record("__xor__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__divmod__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, []),
    op_record("__lshift__", 2, int_dtypes_no_uint64, all_shapes, partial(jtu.rand_int, high=8), []),
    op_record("__rshift__", 2, int_dtypes_no_uint64, all_shapes, partial(jtu.rand_int, high=8), []),
]

JAX_RIGHT_OPERATOR_OVERLOADS = [
    op_record("__radd__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rsub__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rmul__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rpow__", 2, inexact_dtypes, all_shapes, jtu.rand_positive, [],
              tolerance={np.float32: 2e-4, np.complex64: 1e-3}),
    op_record("__rmod__", 2, default_dtypes, all_shapes, jtu.rand_nonzero, [],
              tolerance={np.float16: 1e-1}),
    op_record("__rfloordiv__", 2, default_dtypes, all_shapes,
              jtu.rand_nonzero, []),
    op_record("__rtruediv__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, [],
              inexact=True),
    # op_record("__ror__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__rand__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    # op_record("__rxor__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__rdivmod__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, []),
    op_record("__rlshift__", 2, int_dtypes_no_uint64, all_shapes, partial(jtu.rand_int, high=8), []),
    op_record("__rrshift__", 2, int_dtypes_no_uint64, all_shapes, partial(jtu.rand_int, high=8), [])
]

class _OverrideEverything(object):
  pass

for rec in JAX_OPERATOR_OVERLOADS + JAX_RIGHT_OPERATOR_OVERLOADS:
  if rec.nargs == 2:
    setattr(_OverrideEverything, rec.name, lambda self, other: self)

class _OverrideNothing(object):
  pass

for rec in JAX_OPERATOR_OVERLOADS + JAX_RIGHT_OPERATOR_OVERLOADS:
  if rec.nargs == 2:
    setattr(_OverrideNothing, rec.name, lambda self, other: NotImplemented)


def _dtypes_are_compatible_for_bitwise_ops(args):
  if len(args) <= 1:
    return True
  is_signed = lambda dtype: jnp.issubdtype(dtype, np.signedinteger)
  width = lambda dtype: jnp.iinfo(dtype).bits
  x, y = args
  if width(x) > width(y):
    x, y = y, x
  # The following condition seems a little ad hoc, but seems to capture what
  # numpy actually implements.
  return (
      is_signed(x) == is_signed(y)
      or (width(x) == 32 and width(y) == 32)
      or (width(x) == 32 and width(y) == 64 and is_signed(y)))

def _shapes_are_broadcast_compatible(shapes):
  try:
    lax.broadcast_shapes(*(() if s in scalar_shapes else s for s in shapes))
  except ValueError:
    return False
  else:
    return True

def _shapes_are_equal_length(shapes):
  return all(len(shape) == len(shapes[0]) for shape in shapes[1:])


def _promote_like_jnp(fun, inexact=False):
  """Decorator that promotes the arguments of `fun` to `jnp.result_type(*args)`.

  jnp and np have different type promotion semantics; this decorator allows
  tests make an np reference implementation act more like an jnp
  implementation.
  """
  _promote = _promote_dtypes_inexact if inexact else _promote_dtypes
  def wrapper(*args, **kw):
    flat_args, tree = tree_util.tree_flatten(args)
    args = tree_util.tree_unflatten(tree, _promote(*flat_args))
    return fun(*args, **kw)
  return wrapper


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

  def testNotImplemented(self):
    for name in jnp._NOT_IMPLEMENTED:
      func = getattr(jnp, name)
      with self.assertRaises(NotImplementedError):
        func()

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_allow_picke={}".format(dtype, allow_pickle),
       "dtype": dtype, "allow_pickle": allow_pickle}
      for dtype in float_dtypes + [object]
      for allow_pickle in [True, False]))
  def testLoad(self, dtype, allow_pickle):
    if dtype == object and not allow_pickle:
      self.skipTest("dtype=object requires allow_pickle=True")
    rng = jtu.rand_default(self.rng())
    arr = rng((10), dtype)
    with io.BytesIO() as f:
      jnp.save(f, arr)
      f.seek(0)
      arr_out = jnp.load(f, allow_pickle=allow_pickle)
    self.assertArraysEqual(arr, arr_out)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
         "check_dtypes": rec.check_dtypes, "tolerance": rec.tolerance,
         "inexact": rec.inexact, "kwargs": rec.kwargs or {}}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in itertools.chain(JAX_ONE_TO_ONE_OP_RECORDS,
                                 JAX_COMPOUND_OP_RECORDS)))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testOp(self, np_op, jnp_op, rng_factory, shapes, dtypes, check_dtypes,
             tolerance, inexact, kwargs):
    np_op = partial(np_op, **kwargs)
    jnp_op = partial(jnp_op, **kwargs)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="invalid value.*")(np_op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="divide by zero.*")(np_op)

    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in dtypes)
    tol = functools.reduce(jtu.join_tolerance,
                           [tolerance, tol, jtu.default_tolerance()])
    self._CheckAgainstNumpy(_promote_like_jnp(np_op, inexact), jnp_op,
                            args_maker, check_dtypes=check_dtypes, tol=tol)
    self._CompileAndCheck(jnp_op, args_maker, check_dtypes=check_dtypes,
                          atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes, "name": rec.name,
         "tol": rec.tolerance}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in JAX_OPERATOR_OVERLOADS))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testOperatorOverload(self, name, rng_factory, shapes, dtypes, tol):
    rng = rng_factory(self.rng())
    # np and jnp arrays have different type promotion rules; force the use of
    # jnp arrays.
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    fun = lambda *xs: getattr(operator, name.strip('_'))(*xs)
    self._CompileAndCheck(fun, args_maker, atol=tol, rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes, "name": rec.name,
         "op_tolerance": rec.tolerance}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in JAX_RIGHT_OPERATOR_OVERLOADS))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testRightOperatorOverload(self, name, rng_factory, shapes, dtypes,
                                op_tolerance):
    if shapes[1] is jtu.PYTHON_SCALAR_SHAPE:
      raise SkipTest("scalars not implemented")  # TODO(mattjj): clean up
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    fun = lambda fst, snd: getattr(snd, name)(fst)
    tol = max(jtu.tolerance(dtype, op_tolerance) for dtype in dtypes)
    self._CompileAndCheck( fun, args_maker, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": rec.test_name + "_{}".format(dtype),
       "rng_factory": rec.rng_factory,
       "op_name": rec.name, "dtype": dtype}
      for rec in JAX_OPERATOR_OVERLOADS if rec.nargs == 2
      for dtype in rec.dtypes))
  def testBinaryOperatorDefers(self, op_name, rng_factory, dtype):
    rng = rng_factory(self.rng())
    arg = jax.device_put(rng((), dtype))
    op = getattr(operator, op_name)

    other = _OverrideEverything()
    assert op(other, arg) is other
    assert op(arg, other) is other

    other = _OverrideNothing()
    if op_name == "__eq__":
      assert op(other, arg) is False
      assert op(arg, other) is False
    elif op_name == "__ne__":
      assert op(other, arg) is True
      assert op(arg, other) is True
    else:
      with self.assertRaises(TypeError):
        op(other, arg)
      with self.assertRaises(TypeError):
        op(arg, other)

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

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.test_name, shapes, dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name)}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in filter(
          _dtypes_are_compatible_for_bitwise_ops,
          itertools.combinations_with_replacement(rec.dtypes, rec.nargs)))
      for rec in JAX_BITWISE_OP_RECORDS))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testBitwiseOp(self, np_op, jnp_op, rng_factory, shapes, dtypes):
    rng = rng_factory(self.rng())
    if not config.x64_enabled and any(
        jnp.iinfo(dtype).bits == 64 for dtype in dtypes):
      self.skipTest("x64 types are disabled by jax_enable_x64")
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                            check_dtypes=jtu.PYTHON_SCALAR_SHAPE not in shapes)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": jtu.format_test_name_suffix(op.__name__, shapes, dtypes),
     "op": op, "dtypes": dtypes, "shapes": shapes}
    for op in [jnp.left_shift, jnp.right_shift]
    for shapes in filter(
      _shapes_are_broadcast_compatible,
      # TODO numpy always promotes to shift dtype for zero-dim shapes:
      itertools.combinations_with_replacement(nonzerodim_shapes, 2))
    for dtypes in itertools.product(
      *(_valid_dtypes_for_shape(s, int_dtypes_no_uint64) for s in shapes))))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testShiftOpAgainstNumpy(self, op, dtypes, shapes):
    dtype, shift_dtype = dtypes
    signed_mix = np.issubdtype(dtype, np.signedinteger) != \
                 np.issubdtype(shift_dtype, np.signedinteger)
    has_32 = any(np.iinfo(d).bits == 32 for d in dtypes)
    promoting_to_64 = has_32 and signed_mix
    if promoting_to_64 and not config.x64_enabled:
      self.skipTest("np.right_shift/left_shift promoting to int64"
                    "differs from jnp in 32 bit mode.")

    info, shift_info = map(np.iinfo, dtypes)
    x_rng = jtu.rand_int(self.rng(), low=info.min, high=info.max + 1)
    # NumPy requires shifts to be non-negative and below the bit width:
    shift_rng = jtu.rand_int(self.rng(), high=max(info.bits, shift_info.bits))
    args_maker = lambda: (x_rng(shapes[0], dtype), shift_rng(shapes[1], shift_dtype))
    self._CompileAndCheck(op, args_maker)
    np_op = getattr(np, op.__name__)
    self._CheckAgainstNumpy(np_op, op, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_dtype={}_keepdims={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis,
            "None" if out_dtype is None else np.dtype(out_dtype).name, keepdims),
         "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
         "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
         "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for out_dtype in [None] + rec.dtypes if out_dtype not in unsigned_dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        for keepdims in [False, True])
      for rec in JAX_REDUCER_RECORDS))
  def testReducer(self, np_op, jnp_op, rng_factory, shape, dtype, out_dtype,
                  axis, keepdims, inexact):
    rng = rng_factory(self.rng())
    @jtu.ignore_warning(category=np.ComplexWarning)
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="mean of empty slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="overflow encountered.*")
    def np_fun(x):
      x_cast = x if dtype != jnp.bfloat16 else x.astype(np.float32)
      t = out_dtype if out_dtype != jnp.bfloat16 else np.float32
      return np_op(x_cast, axis, dtype=t, keepdims=keepdims)
    np_fun = _promote_like_jnp(np_fun, inexact)
    jnp_fun = lambda x: jnp_op(x, axis, dtype=out_dtype, keepdims=keepdims)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol_spec = {np.float16: 1e-2, np.int32: 1E-3, np.float32: 1e-3,
                np.complex64: 1e-3, np.float64: 1e-5, np.complex128: 1e-5}
    tol = jtu.tolerance(dtype, tol_spec)
    tol = max(tol, jtu.tolerance(out_dtype, tol_spec)) if out_dtype else tol
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=jnp.bfloat16 not in (dtype, out_dtype),
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_keepdims={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
        "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        for keepdims in [False, True])
      for rec in JAX_REDUCER_NO_DTYPE_RECORDS))
  def testReducerNoDtype(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                         keepdims, inexact):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="All-NaN (slice|axis) encountered.*")
    def np_fun(x):
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res
    np_fun = _promote_like_jnp(np_fun, inexact)
    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {np.float16: 0.002}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=tol, atol=tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_keepdims={}_initial={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis, keepdims, initial),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
        "initial": initial, "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for axis in list(range(-len(shape), len(shape))) + [None]
        for initial in [0, 1] for keepdims in [False, True])
      for rec in JAX_REDUCER_INITIAL_RECORDS))
  def testReducerInitial(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                         keepdims, initial, inexact):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16 and rng_factory.__name__ == 'rand_some_nan'
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    def np_fun(x):
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, initial=initial)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res
    np_fun = _promote_like_jnp(np_fun, inexact)
    np_fun = jtu.ignore_warning(category=np.ComplexWarning)(np_fun)
    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {jnp.bfloat16: 3E-2}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_keepdims={}_initial={}_whereshape={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis, keepdims, initial,
            jtu.format_shape_dtype_string(whereshape, bool)),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name), "whereshape": whereshape,
        "initial": initial, "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for whereshape in _compatible_shapes(shape)
        for axis in list(range(-len(shape), len(shape))) + [None]
        for initial in [0, 1] for keepdims in [False, True])
      for rec in JAX_REDUCER_INITIAL_RECORDS))
  def testReducerWhere(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                       keepdims, initial, inexact, whereshape):
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
    def np_fun(x):
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, initial=initial, where=where)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res
    np_fun = _promote_like_jnp(np_fun, inexact)
    np_fun = jtu.ignore_warning(category=np.ComplexWarning)(np_fun)
    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial, where=where)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @unittest.skipIf(numpy_version < (1, 20), "where parameter not supported in older numpy")
  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}_keepdims={}_whereshape={}".format(
        rec.test_name.capitalize(),
        jtu.format_shape_dtype_string(shape, dtype), axis, keepdims,
        jtu.format_shape_dtype_string(whereshape, bool)),
        "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
        "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name), "whereshape": whereshape,
        "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
      for shape in rec.shapes for dtype in rec.dtypes
      for whereshape in _compatible_shapes(shape)
      for axis in list(range(-len(shape), len(shape))) + [None]
      for keepdims in [False, True])
    for rec in JAX_REDUCER_WHERE_NO_INITIAL_RECORDS))
  def testReducerWhereNoInitial(self, np_op, jnp_op, rng_factory, shape, dtype, axis,
                                keepdims, inexact, whereshape):
    rng = rng_factory(self.rng())
    is_bf16_nan_test = dtype == jnp.bfloat16
    # Do not pass where via args_maker as that is incompatible with _promote_like_jnp.
    where = jtu.rand_bool(self.rng())(whereshape, np.bool_)
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Mean of empty slice.*")
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="invalid value encountered in true_divide*")
    def np_fun(x):
      x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
      res = np_op(x_cast, axis, keepdims=keepdims, where=where)
      res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
      return res

    np_fun = _promote_like_jnp(np_fun, inexact)
    np_fun = jtu.ignore_warning(category=np.ComplexWarning)(np_fun)
    jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, where=where)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
    args_maker = lambda: [rng(shape, dtype)]
    if numpy_version >= (1, 20, 2) or np_op.__name__ in ("all", "any"):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in all_shapes for dtype in all_dtypes
      for axis in list(range(-len(shape), len(shape))) + [None]))
  def testCountNonzero(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.count_nonzero(x, axis)
    jnp_fun = lambda x: jnp.count_nonzero(x, axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testNonzero(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.nonzero(x)
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np_fun)
    jnp_fun = lambda x: jnp.nonzero(x)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_size={}_fill_value={}".format(
          jtu.format_shape_dtype_string(shape, dtype), size, fill_value),
       "shape": shape, "dtype": dtype, "size": size, "fill_value": fill_value}
      for shape in nonempty_array_shapes
      for dtype in all_dtypes
      for fill_value in [None, -1, shape or (1,)]
      for size in [1, 5, 10]))
  def testNonzeroSize(self, shape, dtype, size, fill_value):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    @jtu.ignore_warning(category=DeprecationWarning, message="Calling nonzero on 0d arrays.*")
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testFlatNonzero(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np.flatnonzero)
    jnp_fun = jnp.flatnonzero
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

    # JIT compilation requires specifying the size statically:
    jnp_fun = lambda x: jnp.flatnonzero(x, size=np.size(x) // 2)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_size={}_fill_value={}".format(
          jtu.format_shape_dtype_string(shape, dtype), size, fill_value),
       "shape": shape, "dtype": dtype, "size": size, "fill_value": fill_value}
      for shape in nonempty_array_shapes
      for dtype in all_dtypes
      for fill_value in [None, -1, 10, (-1,), (10,)]
      for size in [1, 5, 10]))
  def testFlatNonzeroSize(self, shape, dtype, size, fill_value):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    @jtu.ignore_warning(category=DeprecationWarning, message="Calling nonzero on 0d arrays.*")
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testArgWhere(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np.argwhere)
    jnp_fun = jnp.argwhere
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

    # JIT compilation requires specifying a size statically. Full test of this
    # behavior is in testNonzeroSize().
    jnp_fun = lambda x: jnp.argwhere(x, size=np.size(x) // 2)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_size={}_fill_value={}".format(
          jtu.format_shape_dtype_string(shape, dtype), size, fill_value),
       "shape": shape, "dtype": dtype, "size": size, "fill_value": fill_value}
      for shape in nonempty_array_shapes
      for dtype in all_dtypes
      for fill_value in [None, -1, shape or (1,)]
      for size in [1, 5, 10]))
  def testArgWhereSize(self, shape, dtype, size, fill_value):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    @jtu.ignore_warning(category=DeprecationWarning, message="Calling nonzero on 0d arrays.*")
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}_keepdims={}".format(
          rec.test_name.capitalize(),
          jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
       "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
       "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name),
       "axis": axis, "keepdims": keepdims}
      for rec in JAX_ARGMINMAX_RECORDS
      for shape, dtype in _shape_and_dtypes(rec.shapes, rec.dtypes)
      for axis in range(-len(shape), len(shape))
      for keepdims in [True, False]))
  def testArgMinMax(self, np_op, jnp_op, rng_factory, shape, dtype, axis, keepdims):
    rng = rng_factory(self.rng())
    if dtype == np.complex128 and jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("complex128 reductions not supported on GPU")
    if "nan" in np_op.__name__ and dtype == jnp.bfloat16:
      raise unittest.SkipTest("NumPy doesn't correctly handle bfloat16 arrays")
    if numpy_version < (1, 22) and keepdims:
      raise unittest.SkipTest("NumPy < 1.22 does not support keepdims argument to argmin/argmax")
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": rec.test_name.capitalize(), "name": rec.name,
       "np_op": getattr(np, rec.name), "jnp_op": getattr(jnp, rec.name)}
      for rec in JAX_ARGMINMAX_RECORDS))
  def testArgMinMaxEmpty(self, name, np_op, jnp_op):
    name = name[3:] if name.startswith("nan") else name
    msg = "attempt to get {} of an empty sequence".format(name)
    with self.assertRaises(ValueError, msg=msg):
      jnp_op(np.array([]))
    with self.assertRaises(ValueError, msg=msg):
      jnp_op(np.zeros((2, 0)), axis=1)
    np_fun = jtu.with_jax_dtype_defaults(partial(np_op, axis=0))
    jnp_fun = partial(jnp_op, axis=0)
    args_maker = lambda: [np.zeros((2, 0))]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype),
          axes),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "axes": axes}
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
      ]
      for lhs_dtype, rhs_dtype in itertools.combinations_with_replacement(number_dtypes, 2)))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testCross(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    axisa, axisb, axisc, axis = axes
    jnp_fun = lambda a, b: jnp.cross(a, b, axisa, axisb, axisc, axis)
    def np_fun(a, b):
      a = a.astype(np.float32) if lhs_dtype == jnp.bfloat16 else a
      b = b.astype(np.float32) if rhs_dtype == jnp.bfloat16 else b
      out = np.cross(a, b, axisa, axisb, axisc, axis)
      return out.astype(jnp.promote_types(lhs_dtype, rhs_dtype))
    tol_spec = {dtypes.bfloat16: 3e-1, np.float16: 0.15}
    tol = max(jtu.tolerance(lhs_dtype, tol_spec),
              jtu.tolerance(rhs_dtype, tol_spec))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype}
      for name, lhs_shape, rhs_shape in [
          ("matrix-scalar", (3, 3), ()),
          ("scalar-matrix", (), (3, 3)),
          ("matrix-vector", (4, 5), (5,)),
          ("vector-matrix", (6,), (6, 4)),
          ("matrix-matrix", (3, 4), (4, 5)),
          ("tensor-vector", (4, 3, 2), (2,)),
          ("vector-tensor", (2,), (3, 2, 4)),
          ("tensor-matrix", (4, 3, 2), (2, 5)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-tensor", (2, 3, 4), (5, 4, 1))]
      for lhs_dtype, rhs_dtype in itertools.combinations_with_replacement(number_dtypes, 2)))
  def testDot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {np.float16: 1e-2, np.float32: 1e-5, np.float64: 1e-14,
           np.complex128: 1e-14}
    if jtu.device_under_test() == "tpu":
      tol[np.float16] = tol[np.float32] = tol[np.complex64] = 2e-1
    def np_dot(x, y):
      x = x.astype(np.float32) if lhs_dtype == jnp.bfloat16 else x
      y = y.astype(np.float32) if rhs_dtype == jnp.bfloat16 else y
      return np.dot(x, y).astype(jnp.promote_types(lhs_dtype, rhs_dtype))
    self._CheckAgainstNumpy(np_dot, jnp.dot, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp.dot, args_maker, atol=tol,
                          rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype}
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
          ("tensor-tensor-broadcast", (3, 1, 3, 4), (5, 4, 1))]
      for lhs_dtype, rhs_dtype in itertools.combinations_with_replacement(number_dtypes, 2)))
  def testMatmul(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    def np_fun(x, y):
      dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
      return np.matmul(x, y).astype(dtype)
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {np.float16: 1e-2, np.float32: 2e-2, np.float64: 1e-12,
           np.complex128: 1e-12}
    if jtu.device_under_test() == "tpu":
      tol[np.float16] = tol[np.float32] = tol[np.complex64] = 4e-2
    self._CheckAgainstNumpy(np_fun, jnp.matmul, args_maker, tol=tol)
    self._CompileAndCheck(jnp.matmul, args_maker, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype),
          axes),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "axes": axes}
      for lhs_shape, rhs_shape, axes in [
          [(3,), (), 0],
          [(2, 3, 4), (5, 6, 7), 0],  # from issue #740
          [(2, 3, 4), (3, 4, 5, 6), 2],
          [(2, 3, 4), (5, 4, 3, 6), [1, 2]],
          [(2, 3, 4), (5, 4, 3, 6), [[1, 2], [2, 1]]],
          [(1, 2, 3, 4), (4, 5, 3, 6), [[2, 3], [2, 0]]],
      ]
      for lhs_dtype, rhs_dtype in itertools.combinations_with_replacement(number_dtypes, 2)))
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
    if jtu.device_under_test() == "tpu":
      tol[np.float16] = tol[np.float32] = tol[np.complex64] = 2e-1
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

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


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_invert={}".format(
          jtu.format_shape_dtype_string(element_shape, dtype),
          jtu.format_shape_dtype_string(test_shape, dtype), invert),
       "element_shape": element_shape, "test_shape": test_shape,
       "dtype": dtype, "invert": invert}
      for element_shape in all_shapes
      for test_shape in all_shapes
      for dtype in default_dtypes
      for invert in [True, False]))
  def testIsin(self, element_shape, test_shape, dtype, invert):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(element_shape, dtype), rng(test_shape, dtype)]
    jnp_fun = lambda e, t: jnp.isin(e, t, invert=invert)
    np_fun = lambda e, t: np.isin(e, t, invert=invert)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_invert={}".format(
          jtu.format_shape_dtype_string(element_shape, dtype),
          jtu.format_shape_dtype_string(test_shape, dtype), invert),
       "element_shape": element_shape, "test_shape": test_shape,
       "dtype": dtype, "invert": invert}
      for element_shape in all_shapes
      for test_shape in all_shapes
      for dtype in default_dtypes
      for invert in [True, False]))
  def testIn1d(self, element_shape, test_shape, dtype, invert):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(element_shape, dtype), rng(test_shape, dtype)]
    jnp_fun = lambda e, t: jnp.in1d(e, t, invert=invert)
    np_fun = lambda e, t: np.in1d(e, t, invert=invert)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
       jtu.format_shape_dtype_string(shape1, dtype1),
       jtu.format_shape_dtype_string(shape2, dtype2)),
       "shape1": shape1, "shape2": shape2, "dtype1": dtype1, "dtype2": dtype2}
      for dtype1 in [s for s in default_dtypes if s != jnp.bfloat16]
      for dtype2 in [s for s in default_dtypes if s != jnp.bfloat16]
      for shape1 in all_shapes
      for shape2 in all_shapes))
  def testSetdiff1d(self, shape1, shape2, dtype1, dtype2):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
    self._CheckAgainstNumpy(np.setdiff1d, jnp.setdiff1d, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_size={}_fill_value={}".format(
       jtu.format_shape_dtype_string(shape1, dtype1),
       jtu.format_shape_dtype_string(shape2, dtype2),
       size, fill_value),
       "shape1": shape1, "shape2": shape2, "dtype1": dtype1, "dtype2": dtype2,
       "size": size, "fill_value": fill_value}
      for dtype1 in [s for s in default_dtypes if s != jnp.bfloat16]
      for dtype2 in [s for s in default_dtypes if s != jnp.bfloat16]
      for shape1 in all_shapes
      for shape2 in all_shapes
      for size in [1, 5, 10]
      for fill_value in [None, -1]))
  def testSetdiff1dSize(self, shape1, shape2, dtype1, dtype2, size, fill_value):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
    def np_fun(arg1, arg2):
      result = np.setdiff1d(arg1, arg2)
      if size <= len(result):
        return result[:size]
      else:
        return np.pad(result, (0, size-len(result)), constant_values=fill_value or 0)
    def jnp_fun(arg1, arg2):
      return jnp.setdiff1d(arg1, arg2, size=size, fill_value=fill_value)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
       jtu.format_shape_dtype_string(shape1, dtype1),
       jtu.format_shape_dtype_string(shape2, dtype2)),
       "shape1": shape1, "shape2": shape2, "dtype1": dtype1, "dtype2": dtype2}
      for dtype1 in [s for s in default_dtypes if s != jnp.bfloat16]
      for dtype2 in [s for s in default_dtypes if s != jnp.bfloat16]
      for shape1 in nonempty_nonscalar_array_shapes
      for shape2 in nonempty_nonscalar_array_shapes))
  def testUnion1d(self, shape1, shape2, dtype1, dtype2):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
    def np_fun(arg1, arg2):
      dtype = jnp.promote_types(arg1.dtype, arg2.dtype)
      return np.union1d(arg1, arg2).astype(dtype)
    self._CheckAgainstNumpy(np_fun, jnp.union1d, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_size={}_fill_value={}".format(
       jtu.format_shape_dtype_string(shape1, dtype1),
       jtu.format_shape_dtype_string(shape2, dtype2), size, fill_value),
       "shape1": shape1, "shape2": shape2, "dtype1": dtype1, "dtype2": dtype2,
       "size": size, "fill_value": fill_value}
      for dtype1 in [s for s in default_dtypes if s != jnp.bfloat16]
      for dtype2 in [s for s in default_dtypes if s != jnp.bfloat16]
      for shape1 in nonempty_nonscalar_array_shapes
      for shape2 in nonempty_nonscalar_array_shapes
      for size in [1, 5, 10]
      for fill_value in [None, -1]))
  def testUnion1dSize(self, shape1, shape2, dtype1, dtype2, size, fill_value):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
    def np_fun(arg1, arg2):
      dtype = jnp.promote_types(arg1.dtype, arg2.dtype)
      result = np.union1d(arg1, arg2).astype(dtype)
      fv = result.min() if fill_value is None else fill_value
      if size <= len(result):
        return result[:size]
      else:
        return np.concatenate([result, np.full(size - len(result), fv, result.dtype)])
    def jnp_fun(arg1, arg2):
      return jnp.union1d(arg1, arg2, size=size, fill_value=fill_value)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_assume_unique={}".format(
       jtu.format_shape_dtype_string(shape1, dtype1),
       jtu.format_shape_dtype_string(shape2, dtype2),
       assume_unique),
       "shape1": shape1, "dtype1": dtype1, "shape2": shape2, "dtype2": dtype2,
       "assume_unique": assume_unique}
      for dtype1 in [s for s in default_dtypes if s != jnp.bfloat16]
      for dtype2 in [s for s in default_dtypes if s != jnp.bfloat16]
      for shape1 in all_shapes
      for shape2 in all_shapes
      for assume_unique in [False, True]))
  def testSetxor1d(self, shape1, dtype1, shape2, dtype2, assume_unique):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
    jnp_fun = lambda ar1, ar2: jnp.setxor1d(ar1, ar2, assume_unique=assume_unique)
    def np_fun(ar1, ar2):
      if assume_unique:
        # pre-flatten the arrays to match with jax implementation
        ar1 = np.ravel(ar1)
        ar2 = np.ravel(ar2)
      return np.setxor1d(ar1, ar2, assume_unique)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_assume_unique={}_return_indices={}".format(
       jtu.format_shape_dtype_string(shape1, dtype1),
       jtu.format_shape_dtype_string(shape2, dtype2),
       assume_unique,
       return_indices),
       "shape1": shape1, "dtype1": dtype1, "shape2": shape2, "dtype2": dtype2,
       "assume_unique": assume_unique, "return_indices": return_indices}
      for dtype1 in [s for s in default_dtypes if s != jnp.bfloat16]
      for dtype2 in [s for s in default_dtypes if s != jnp.bfloat16]
      for shape1 in all_shapes
      for shape2 in all_shapes
      for assume_unique in [False, True]
      for return_indices in [False, True]))
  def testIntersect1d(self, shape1, dtype1, shape2, dtype2, assume_unique, return_indices):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
    jnp_fun = lambda ar1, ar2: jnp.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
    np_fun = lambda ar1, ar2: np.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype}
      # TODO(phawkins): support integer dtypes too.
      for lhs_shape, lhs_dtype in _shape_and_dtypes(all_shapes, inexact_dtypes)
      for rhs_shape, rhs_dtype in _shape_and_dtypes(all_shapes, inexact_dtypes)
      if len(jtu._dims_of_shape(lhs_shape)) == 0
      or len(jtu._dims_of_shape(rhs_shape)) == 0
      or lhs_shape[-1] == rhs_shape[-1]))
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
    if jtu.device_under_test() == "tpu":
      tol_spec[np.float32] = tol_spec[np.complex64] = 2e-1
    tol = max(jtu.tolerance(lhs_dtype, tol_spec),
              jtu.tolerance(rhs_dtype, tol_spec))
    # TODO(phawkins): there are float32/float64 disagreements for some inputs.
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=False, atol=tol, rtol=tol)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_deg={}_rcond={}_full={}_w={}_cov={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          deg,
          rcond,
          full,
          w,
          cov),
      "shape": shape, "dtype": dtype, "deg": deg,
      "rcond": rcond, "full": full, "w":w, "cov":cov}
      for dtype in [dt for dt in float_dtypes if dt not in [jnp.float16, jnp.bfloat16]]
      for shape in [shape for shape in one_dim_array_shapes if shape != (1,)]
      for deg in [1, 2, 3]
      for rcond in [None, -1, 10e-3, 10e-5, 10e-10]
      for full in [False, True]
      for w in [False, True]
      for cov in [False, True, "unscaled"]))
  @jtu.skip_on_devices("rocm")  # will be fixed in ROCm-5.1
  def testPolyfit(self, shape, dtype, deg, rcond, full, w, cov):
    rng = jtu.rand_default(self.rng())
    tol_spec = {np.float32: 1e-3, np.float64: 1e-13, np.complex64: 1e-5}
    if jtu.device_under_test() == "tpu":
      tol_spec[np.float32] = tol_spec[np.complex64] = 2e-1
    tol = jtu.tolerance(dtype, tol_spec)
    _w = lambda a: abs(a) if w else None
    args_maker = lambda: [rng(shape, dtype), rng(shape, dtype), rng(shape, dtype)]
    jnp_fun = lambda x, y, a: jnp.polyfit(x, y, deg=deg, rcond=rcond, full=full, w=_w(a), cov=cov)
    np_fun = jtu.ignore_warning(
      message="Polyfit may be poorly conditioned*")(lambda x, y, a: np.polyfit(x, y, deg=deg, rcond=rcond, full=full, w=_w(a), cov=cov))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=False, atol=tol, rtol=tol)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_amin={}_amax={}".format(
          jtu.format_shape_dtype_string(shape, dtype), a_min, a_max),
       "shape": shape, "dtype": dtype, "a_min": a_min, "a_max": a_max}
      for shape in all_shapes for dtype in number_dtypes
      for a_min, a_max in [(-1, None), (None, 1), (-0.9, 1),
                           (-np.ones(1), None),
                           (None, np.ones(1)),
                           (np.full(1, -0.9), np.ones(1))]))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testClipStaticBounds(self, shape, dtype, a_min, a_max):
    if np.issubdtype(dtype, np.unsignedinteger):
      a_min = None if a_min is None else abs(a_min)
      a_max = None if a_max is None else abs(a_max)
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.clip(x, a_min=a_min, a_max=a_max)
    jnp_fun = lambda x: jnp.clip(x, a_min=a_min, a_max=a_max)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testClipError(self):
    with self.assertRaisesRegex(ValueError, "At most one of a_min and a_max.*"):
      jnp.clip(jnp.zeros((3,)))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_decimals={}".format(
          jtu.format_shape_dtype_string(shape, dtype), decimals),
       "shape": shape, "dtype": dtype, "decimals": decimals}
      for shape, dtype in _shape_and_dtypes(all_shapes, number_dtypes)
      for decimals in [0, 1, -2]))
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

  def testOperatorRound(self):
    self.assertAllClose(round(np.float32(7.532), 1),
                        round(jnp.float32(7.5), 1))
    self.assertAllClose(round(np.float32(1.234), 2),
                        round(jnp.float32(1.234), 2))
    self.assertAllClose(round(np.float32(1.234)),
                        round(jnp.float32(1.234)), check_dtypes=False)
    self.assertAllClose(round(np.float32(7.532), 1),
                        round(jnp.array(7.5, jnp.float32), 1))
    self.assertAllClose(round(np.float32(1.234), 2),
                        round(jnp.array(1.234, jnp.float32), 2))
    self.assertAllClose(round(np.float32(1.234)),
                        round(jnp.array(1.234, jnp.float32)),
                        check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_mode={}_padwidth={}_constantvalues={}".format(
          jtu.format_shape_dtype_string(shape, dtype), mode, pad_width,
          constant_values),
       "shape": shape, "dtype": dtype, "mode": mode,
       "pad_width": pad_width, "constant_values": constant_values}
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
          ((0, 2),), ((-1, 3.14),),
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
           (mode != 'constant' and constant_values is None)))))
  def testPad(self, shape, dtype, mode, pad_width, constant_values):
    if np.issubdtype(dtype, np.unsignedinteger):
      constant_values = tree_util.tree_map(abs, constant_values)
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_mode={}_pad_width={}_stat_length={}".format(
          jtu.format_shape_dtype_string(shape, dtype), mode, pad_width, stat_length),
       "shape": shape, "dtype": dtype, "mode": mode, "pad_width": pad_width,
       "stat_length": stat_length}
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
          not (dtype in bool_dtypes and mode == 'mean'))))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_mode={}_pad_width={}_reflect_type={}".format(
          jtu.format_shape_dtype_string(shape, dtype), mode, pad_width, reflect_type),
       "shape": shape, "dtype": dtype, "mode": mode, "pad_width": pad_width,
       "reflect_type": reflect_type}
      for mode in ['symmetric', 'reflect']
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
          (reflect_type != 'odd' or dtype not in [np.bool_, np.float16, jnp.bfloat16]))))
  def testPadSymmetricAndReflect(self, shape, dtype, mode, pad_width, reflect_type):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]

    np_fun = partial(np.pad, pad_width=pad_width, mode=mode, reflect_type=reflect_type)
    jnp_fun = partial(jnp.pad, pad_width=pad_width, mode=mode, reflect_type=reflect_type)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE,
                            tol={np.float32: 1e-3, np.complex64: 1e-3})
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_mode={}_pad_width={}_end_values={}".format(
          jtu.format_shape_dtype_string(shape, dtype), "linear_ramp", pad_width, end_values),
       "shape": shape, "dtype": dtype, "pad_width": pad_width,
       "end_values": end_values}
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
          dtype not in [np.int8, np.int16, np.float16, jnp.bfloat16])))
  def testPadLinearRamp(self, shape, dtype, pad_width, end_values):
    if numpy_version < (1, 20) and np.issubdtype(dtype, np.integer):
      raise unittest.SkipTest("NumPy 1.20 changed the semantics of np.linspace")
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
      match = "unsupported keyword arguments for mode '{}'".format(mode)
      for key, value in not_allowed.items():
        with self.assertRaisesRegex(ValueError, match):
          jnp.pad(arr, pad_width, mode, **{key: value})

    # Test if unsupported mode raise error.
    unsupported_modes = [1, None, "foo"]
    for mode in unsupported_modes:
      match = "Unimplemented padding mode '{}' for np.pad.".format(mode)
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

  # we think we can skip this behavior...
  # def testPadWeakType(self):
  #   x = jnp.array(1.0)[None]
  #   for mode in ['constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median',
  #                'minimum', 'reflect', 'symmetric', 'wrap', 'empty']:
  #     y = jnp.pad(x, 0, mode=mode)
  #     self.assertTrue(dtypes.is_weakly_typed(y))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape=[{}]_reps={}".format(
          jtu.format_shape_dtype_string(shape, dtype), reps),
       "shape": shape, "dtype": dtype, "reps": reps}
      for reps in [(), (2,), (3, 4), (2, 3, 4), (1, 0, 2)]
      for shape, dtype in _shape_and_dtypes(all_shapes, default_dtypes)
      ))
  def testTile(self, shape, dtype, reps):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np.tile(arg, reps)
    jnp_fun = lambda arg: jnp.tile(arg, reps)

    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes
      for dtype in all_dtypes))
  def testExtract(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, jnp.float32), rng(shape, dtype)]
    self._CheckAgainstNumpy(np.extract, jnp.extract, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_ncond={}_nfunc={}".format(
          jtu.format_shape_dtype_string(shape, dtype), ncond, nfunc),
       "shape": shape, "dtype": dtype, "ncond": ncond, "nfunc": nfunc}
      for ncond in [1, 2, 3]
      for nfunc in [ncond, ncond + 1]
      for shape in all_shapes
      for dtype in all_dtypes))
  def testPiecewise(self, shape, dtype, ncond, nfunc):
    rng = jtu.rand_default(self.rng())
    rng_bool = jtu.rand_int(self.rng(), 0, 2)
    funclist = [lambda x: x - 1, 1, lambda x: x, 0][:nfunc]
    args_maker = lambda: (rng(shape, dtype), [rng_bool(shape, bool) for i in range(ncond)])
    np_fun = partial(np.piecewise, funclist=funclist)
    jnp_fun = partial(jnp.piecewise, funclist=funclist)
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

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "{}_perm={}_{}".format(
      jtu.format_shape_dtype_string(shape, dtype), perm, arg_type),
     "dtype": dtype, "shape": shape, "perm": perm, "arg_type": arg_type}
    for dtype in default_dtypes
    for shape in array_shapes
    for arg_type in ["splat", "value"]
    for perm in [None, tuple(np.random.RandomState(0).permutation(np.zeros(shape).ndim))]))
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


  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "{}_trim={}".format(
      jtu.format_shape_dtype_string(a_shape, dtype), trim),
     "dtype": dtype, "a_shape": a_shape, "trim": trim}
    for dtype in default_dtypes
    for a_shape in one_dim_array_shapes
    for trim in ["f", "b", "fb"]))
  def testTrimZeros(self, a_shape, dtype, trim):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(a_shape, dtype)]
    np_fun = lambda arg1: np.trim_zeros(arg1, trim)
    jnp_fun = lambda arg1: jnp.trim_zeros(arg1, trim)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_rank{}".format(
        jtu.format_shape_dtype_string(a_shape, dtype), rank),
       "dtype": dtype, "a_shape": a_shape, "rank": rank}
      for rank in (1, 2)
      for dtype in default_dtypes
      for a_shape in one_dim_array_shapes))
  def testPoly(self, a_shape, dtype, rank):
    if dtype in (np.float16, jnp.bfloat16, np.int16):
      self.skipTest(f"{dtype} gets promoted to {np.float16}, which is not supported.")
    elif rank == 2 and jtu.device_under_test() in ("tpu", "gpu"):
      self.skipTest("Nonsymmetric eigendecomposition is only implemented on the CPU backend.")
    rng = jtu.rand_default(self.rng())
    tol = { np.int8: 1e-3, np.int32: 1e-3, np.float32: 1e-3, np.float64: 1e-6 }
    if jtu.device_under_test() == "tpu":
      tol[np.int32] = tol[np.float32] = 1e-1
    tol = jtu.tolerance(dtype, tol)
    args_maker = lambda: [rng(a_shape * rank, dtype)]
    self._CheckAgainstNumpy(np.poly, jnp.poly, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp.poly, args_maker, check_dtypes=True, rtol=tol, atol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "a_shape={} , b_shape={}".format(
          jtu.format_shape_dtype_string(a_shape, dtype),
          jtu.format_shape_dtype_string(b_shape, dtype)),
       "dtype": dtype, "a_shape": a_shape, "b_shape" : b_shape}
      for dtype in default_dtypes
      for a_shape in one_dim_array_shapes
      for b_shape in one_dim_array_shapes))
  def testPolyAdd(self, a_shape, b_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polyadd(arg1, arg2)
    jnp_fun = lambda arg1, arg2: jnp.polyadd(arg1, arg2)
    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "a_shape={} , b_shape={}".format(
          jtu.format_shape_dtype_string(a_shape, dtype),
          jtu.format_shape_dtype_string(b_shape, dtype)),
       "dtype": dtype, "a_shape": a_shape, "b_shape" : b_shape}
      for dtype in default_dtypes
      for a_shape in one_dim_array_shapes
      for b_shape in one_dim_array_shapes))
  def testPolySub(self, a_shape, b_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polysub(arg1, arg2)
    jnp_fun = lambda arg1, arg2: jnp.polysub(arg1, arg2)
    args_maker = lambda: [rng(a_shape, dtype), rng(b_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_order={}_k={}".format(
          jtu.format_shape_dtype_string(a_shape, dtype),
          order, k),
       "dtype": dtype, "a_shape": a_shape, "order" : order, "k": k}
      for dtype in default_dtypes
      for a_shape in one_dim_array_shapes
      for order in range(5)
      for k in [np.arange(order, dtype=dtype), np.ones(1, dtype), None]))
  def testPolyInt(self, a_shape, order, k, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1: np.polyint(arg1, m=order, k=k)
    jnp_fun = lambda arg1: jnp.polyint(arg1, m=order, k=k)
    args_maker = lambda: [rng(a_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_order={}".format(
          jtu.format_shape_dtype_string(a_shape, dtype),
          order),
       "dtype": dtype, "a_shape": a_shape, "order" : order}
      for dtype in default_dtypes
      for a_shape in one_dim_array_shapes
      for order in range(5)))
  def testPolyDer(self, a_shape, order, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1: np.polyder(arg1, m=order)
    jnp_fun = lambda arg1: jnp.polyder(arg1, m=order)
    args_maker = lambda: [rng(a_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_ptype={}".format(ptype), "ptype": ptype}
      for ptype in ['int', 'np.int', 'jnp.int']))
  def testIntegerPower(self, ptype):
    p = {'int': 2, 'np.int': np.int32(2), 'jnp.int': jnp.int32(2)}[ptype]
    jaxpr = jax.make_jaxpr(partial(jnp.power, x2=p))(1)
    eqns = jaxpr.jaxpr.eqns
    self.assertLen(eqns, 1)
    self.assertEqual(eqns[0].primitive, lax.integer_pow_p)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_x={}_y={}".format(x, y), "x": x, "y": y}
      for x in [-1, 0, 1]
      for y in [0, 32, 64, 128]))
  def testIntegerPowerOverflow(self, x, y):
    # Regression test for https://github.com/google/jax/issues/5987
    args_maker = lambda: [x, y]
    self._CheckAgainstNumpy(np.power, jnp.power, args_maker)
    self._CompileAndCheck(jnp.power, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in all_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(len(shape)))))
  def testCompress(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    if shape in scalar_shapes or len(shape) == 0:
      cond_shape = (0,)
    elif axis is None:
      cond_shape = (prod(shape),)
    else:
      cond_shape = (shape[axis],)

    args_maker = lambda: [rng(cond_shape, jnp.float32), rng(shape, dtype)]

    np_fun = partial(np.compress, axis=axis)
    jnp_fun = partial(jnp.compress, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_condition=array[{}]_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), len(condition), axis),
       "shape": shape, "dtype": dtype, "condition": condition, "axis": axis}
      for shape in [(2, 3)]
      for dtype in int_dtypes
      # condition entries beyond axis size must be zero.
      for condition in [[1], [1, 0, 0, 0, 0, 0, 0]]
      for axis in [None, 0, 1]))
  def testCompressMismatchedShapes(self, shape, dtype, condition, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [np.array(condition), rng(shape, dtype)]
    np_fun = partial(np.compress, axis=axis)
    jnp_fun = partial(jnp.compress, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in array_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(len(shape)))))
  def testCompressMethod(self, shape, dtype, axis):
    rng = jtu.rand_some_zero(self.rng())
    if shape in scalar_shapes or len(shape) == 0:
      cond_shape = (0,)
    elif axis is None:
      cond_shape = (prod(shape),)
    else:
      cond_shape = (shape[axis],)

    args_maker = lambda: [rng(cond_shape, jnp.float32), rng(shape, dtype)]

    np_fun = lambda condition, x: np.compress(condition, x, axis=axis)
    jnp_fun = lambda condition, x: x.compress(condition, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_axis={}_baseshape=[{}]_dtypes=[{}]".format(
          axis, ",".join(str(d) for d in base_shape),
          ",".join(np.dtype(dtype).name for dtype in arg_dtypes)),
       "axis": axis, "base_shape": base_shape, "arg_dtypes": arg_dtypes}
      for num_arrs in [3]
      for arg_dtypes in itertools.combinations_with_replacement(default_dtypes, num_arrs)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))))
  def testConcatenate(self, axis, base_shape, arg_dtypes):
    rng = jtu.rand_default(self.rng())
    wrapped_axis = axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), arg_dtypes)]
    def np_fun(*args):
      args = [x if x.dtype != jnp.bfloat16 else x.astype(np.float32)
              for x in args]
      dtype = functools.reduce(jnp.promote_types, arg_dtypes)
      return np.concatenate(args, axis=axis).astype(dtype)
    jnp_fun = lambda *args: jnp.concatenate(args, axis=axis)

    def args_maker():
      return [rng(shape, dtype) for shape, dtype in zip(shapes, arg_dtypes)]

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
        "shape": shape, "dtype": dtype, "axis": axis}
      for shape in [(4, 1), (4, 3), (4, 5, 6)]
      for dtype in all_dtypes
      for axis in [None] + list(range(1 - len(shape), len(shape) - 1))))
  def testConcatenateArray(self, shape, dtype, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda x: np.concatenate(x, axis=axis)
    jnp_fun = lambda x: jnp.concatenate(x, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testConcatenateAxisNone(self):
    # https://github.com/google/jax/issues/3419
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5]])
    jnp.concatenate((a, b), axis=None)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_axis={}_baseshape=[{}]_dtypes=[{}]".format(
          axis, ",".join(str(d) for d in base_shape),
          ",".join(np.dtype(dtype).name for dtype in arg_dtypes)),
       "axis": axis, "base_shape": base_shape, "arg_dtypes": arg_dtypes}
      for arg_dtypes in itertools.combinations_with_replacement(default_dtypes, 2)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))))
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

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_idx={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idx),
       "dtype": dtype, "shape": shape, "axis": axis, "idx": idx}
      for shape in nonempty_nonscalar_array_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(-len(shape), len(shape)))
      for idx in (range(-prod(shape), prod(shape))
                  if axis is None else
                  range(-shape[axis], shape[axis]))))
  def testDeleteInteger(self, shape, dtype, idx, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg: np.delete(arg, idx, axis=axis)
    jnp_fun = lambda arg: jnp.delete(arg, idx, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_slc={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, slc),
       "dtype": dtype, "shape": shape, "axis": axis, "slc": slc}
      for shape in nonempty_nonscalar_array_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(-len(shape), len(shape)))
      for slc in [slice(None), slice(1, 3), slice(1, 5, 2)]))
  def testDeleteSlice(self, shape, dtype, axis, slc):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg: np.delete(arg, slc, axis=axis)
    jnp_fun = lambda arg: jnp.delete(arg, slc, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_idx={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis,
          jtu.format_shape_dtype_string(idx_shape, int)),
       "dtype": dtype, "shape": shape, "axis": axis, "idx_shape": idx_shape}
      for shape in nonempty_nonscalar_array_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(-len(shape), len(shape)))
      for idx_shape in all_shapes))
  def testDeleteIndexArray(self, shape, dtype, axis, idx_shape):
    rng = jtu.rand_default(self.rng())
    max_idx = np.zeros(shape).size if axis is None else np.zeros(shape).shape[axis]
    # Previous to numpy 1.19, negative indices were ignored so we don't test this.
    low = 0 if numpy_version < (1, 19, 0) else -max_idx
    idx = jtu.rand_int(self.rng(), low=low, high=max_idx)(idx_shape, int)
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg: np.delete(arg, idx, axis=axis)
    jnp_fun = lambda arg: jnp.delete(arg, idx, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @unittest.skipIf(numpy_version < (1, 19), "boolean mask not supported in numpy < 1.19.0")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "dtype": dtype, "shape": shape, "axis": axis}
      for shape in nonempty_nonscalar_array_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testDeleteMaskArray(self, shape, dtype, axis):
    rng = jtu.rand_default(self.rng())
    mask_size = np.zeros(shape).size if axis is None else np.zeros(shape).shape[axis]
    mask = jtu.rand_int(self.rng(), low=0, high=2)(mask_size, bool)
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arg: np.delete(arg, mask, axis=axis)
    jnp_fun = lambda arg: jnp.delete(arg, mask, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "dtype": dtype, "shape": shape, "axis": axis}
      for shape in nonempty_nonscalar_array_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(-len(shape), len(shape)))))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "dtype": dtype, "shape": shape, "axis": axis}
      for shape in nonempty_nonscalar_array_shapes
      for dtype in all_dtypes
      for axis in [None] + list(range(-len(shape), len(shape)))))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_out_dims={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis, out_dims),
        "shape": shape, "dtype": dtype, "axis": axis, "out_dims": out_dims}
      for shape in nonempty_array_shapes
      for dtype in default_dtypes
      for axis in range(-len(shape), len(shape))
      for out_dims in [0, 1, 2]))
  def testApplyAlongAxis(self, shape, dtype, axis, out_dims):
    def func(x, out_dims):
      if out_dims == 0:
        return x.sum()
      elif out_dims == 1:
        return x * x[0]
      elif out_dims == 2:
        return x[:, None] + x[None, :]
      else:
        raise NotImplementedError(f"out_dims={out_dims}")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    np_fun = lambda arr: np.apply_along_axis(func, axis, arr, out_dims=out_dims)
    jnp_fun = lambda arr: jnp.apply_along_axis(func, axis, arr, out_dims=out_dims)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_func={}_keepdims={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          func, keepdims, axes),
        "shape": shape, "dtype": dtype, "func": func, "keepdims": keepdims, "axes": axes}
      for shape in nonempty_shapes
      for func in ["sum"]
      for keepdims in [True, False]
      for axes in itertools.combinations(range(len(shape)), 2)
      # Avoid low-precision types in sum()
      for dtype in default_dtypes if dtype not in [np.float16, jnp.bfloat16]))
  def testApplyOverAxes(self, shape, dtype, func, keepdims, axes):
    f = lambda x, axis: getattr(x, func)(axis=axis, keepdims=keepdims)
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    np_fun = lambda a: np.apply_over_axes(f, a, axes)
    jnp_fun = lambda a: jnp.apply_over_axes(f, a, axes)
    self._CompileAndCheck(jnp_fun, args_maker)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape=[{}]_axis={}_repeats={}_fixed_size={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis, repeats, fixed_size),
       "axis": axis, "shape": shape, "dtype": dtype, "repeats": repeats,
       'fixed_size': fixed_size}
      for repeats in [0, 1, 2]
      for shape, dtype in _shape_and_dtypes(all_shapes, default_dtypes)
      for axis in [None] + list(range(-len(shape), max(1, len(shape))))
      for fixed_size in [True, False]))
  def testRepeat(self, axis, shape, dtype, repeats, fixed_size):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np.repeat(arg, repeats=repeats, axis=axis)
    np_fun = _promote_like_jnp(np_fun)
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
    self.assertLessEqual(len(jaxpr.jaxpr.eqns), 6)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_ind={}_inv={}_count={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis,
          return_index, return_inverse, return_counts),
       "shape": shape, "dtype": dtype, "axis": axis,
       "return_index": return_index, "return_inverse": return_inverse,
       "return_counts": return_counts}
      for dtype in number_dtypes
      for shape in all_shapes
      for axis in [None] + list(range(len(shape)))
      for return_index in [False, True]
      for return_inverse in [False, True]
      for return_counts in [False, True]))
  def testUnique(self, shape, dtype, axis, return_index, return_inverse, return_counts):
    if axis is not None and numpy_version < (1, 19) and np.empty(shape).size == 0:
      self.skipTest("zero-sized axis in unique leads to error in older numpy.")
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    extra_args = (return_index, return_inverse, return_counts)
    use_defaults =  (False, *(True for arg in extra_args if arg)) if any(extra_args) else False
    np_fun = jtu.with_jax_dtype_defaults(lambda x: np.unique(x, *extra_args, axis=axis), use_defaults)
    jnp_fun = lambda x: jnp.unique(x, *extra_args, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_size={}_fill_value={}".format(
         jtu.format_shape_dtype_string(shape, dtype), axis, size, fill_value),
       "shape": shape, "dtype": dtype, "axis": axis,
       "size": size, "fill_value": fill_value}
      for dtype in number_dtypes
      for size in [1, 5, 10]
      for fill_value in [None, -1.0, "slice"]
      for shape in nonempty_array_shapes
      for axis in [None] + list(range(len(shape)))))
  def testUniqueSize(self, shape, dtype, axis, size, fill_value):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    kwds = dict(axis=axis, return_index=True, return_inverse=True, return_counts=True)

    if fill_value == "slice":
      if axis is None:
        fill_value = rng((), dtype)
      else:
        fill_value = rng(shape[:axis] + shape[axis + 1:], dtype)

    @partial(jtu.with_jax_dtype_defaults, use_defaults=(False, True, True, True))
    def np_fun(x, fill_value=fill_value):
      u, ind, inv, counts = np.unique(x, **kwds)
      axis = kwds['axis']
      if axis is None:
        x = x.ravel()
        axis = 0

      n_unique = u.shape[axis]
      if size <= u.shape[axis]:
        slc = (slice(None),) * axis + (slice(size),)
        u, ind, counts = u[slc], ind[:size], counts[:size]
      else:
        extra = (0, size - n_unique)
        pads = [(0, 0)] * u.ndim
        pads[axis] = extra
        u = np.pad(u, pads, constant_values=0)
        slices = [slice(None)] * u.ndim
        slices[axis] = slice(1)
        if fill_value is None:
          fill_value = u[tuple(slices)]
        elif np.ndim(fill_value):
          fill_value = lax.expand_dims(fill_value, (axis,))
        slices[axis] = slice(n_unique, None)
        u[tuple(slices)] = fill_value
        ind = np.pad(ind, extra, constant_values=ind[0])
        counts = np.pad(counts, extra, constant_values=0)
      return u, ind, inv, counts

    jnp_fun = lambda x: jnp.unique(x, size=size, fill_value=fill_value, **kwds)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @unittest.skipIf(numpy_version < (1, 21), "Numpy < 1.21 does not properly handle NaN values in unique.")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{dtype.__name__}", "dtype": dtype}
      for dtype in inexact_dtypes))
  def testUniqueNans(self, dtype):
    def args_maker():
      x = [-0.0, 0.0, 1.0, 1.0, np.nan, -np.nan]
      if np.issubdtype(dtype, np.complexfloating):
        x = [complex(i, j) for i, j in itertools.product(x, repeat=2)]
      return [np.array(x, dtype=dtype)]

    kwds = dict(return_index=True, return_inverse=True, return_counts=True)
    jnp_fun = partial(jnp.unique, **kwds)
    def np_fun(x):
      dtype = x.dtype
      # numpy unique fails for bfloat16 NaNs, so we cast to float64
      if x.dtype == jnp.bfloat16:
        x = x.astype('float64')
      u, *rest = np.unique(x, **kwds)
      return (u.astype(dtype), *rest)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_fixed_size={}".format(fixed_size),
      "fixed_size": fixed_size}
      for fixed_size in [True, False]))
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

    np_input = np.ones((1))
    jnp_input = jnp.ones((1))
    expected_np_input_after_call = np.ones((1))
    expected_jnp_input_after_call = jnp.ones((1))

    self.assertTrue(device_array.type_is_device_array(jnp.concatenate([np_input])))

    attempt_sideeffect(np_input)
    attempt_sideeffect(jnp_input)

    self.assertAllClose(np_input, expected_np_input_after_call)
    self.assertAllClose(jnp_input, expected_jnp_input_after_call)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_xshape=[{}]_yshape=[{}]_mode={}".format(
          op,
          jtu.format_shape_dtype_string(xshape, dtype),
          jtu.format_shape_dtype_string(yshape, dtype),
          mode),
       "xshape": xshape, "yshape": yshape, "dtype": dtype, "mode": mode,
       "jnp_op": getattr(jnp, op),
       "np_op": getattr(np, op)}
      for mode in ['full', 'same', 'valid']
      for op in ['convolve', 'correlate']
      for dtype in number_dtypes
      for xshape in one_dim_array_shapes
      for yshape in one_dim_array_shapes))
  def testConvolutions(self, xshape, yshape, dtype, mode, jnp_op, np_op):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
    precision = lax.Precision.HIGHEST if jtu.device_under_test() == "tpu" else None
    np_fun = partial(np_op, mode=mode)
    jnp_fun = partial(jnp_op, mode=mode, precision=precision)
    tol = {np.float16: 2e-1, np.float32: 1e-2, np.float64: 1e-14,
           np.complex128: 1e-14}
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_shape=[{}]_axis={}_out_dtype={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), axis,
          out_dtype.__name__),
       "axis": axis, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
       "jnp_op": getattr(jnp, op), "np_op": getattr(np, op)}
      for op in ["cumsum", "cumprod"]
      for dtype in all_dtypes
      for out_dtype in default_dtypes
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testCumSumProd(self, axis, shape, dtype, out_dtype, np_op, jnp_op):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np_op(arg, axis=axis, dtype=out_dtype)
    np_fun = jtu.ignore_warning(category=np.ComplexWarning)(np_fun)
    jnp_fun = lambda arg: jnp_op(arg, axis=axis, dtype=out_dtype)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)

    args_maker = lambda: [rng(shape, dtype)]

    tol_thresholds = {dtypes.bfloat16: 4e-2}
    tol = max(jtu.tolerance(dtype, tol_thresholds),
              jtu.tolerance(out_dtype, tol_thresholds))
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_shape=[{}]_axis={}_out_dtype={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), axis,
          out_dtype.__name__),
       "axis": axis, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
       "jnp_op": getattr(jnp, op), "np_op": getattr(np, op)}
      for op in ["nancumsum", "nancumprod"]
      for dtype in all_dtypes
      for out_dtype in default_dtypes
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testNanCumSumProd(self, axis, shape, dtype, out_dtype, np_op, jnp_op):
    rng = jtu.rand_some_nan(self.rng())
    np_fun = partial(np_op, axis=axis, dtype=out_dtype)
    np_fun = jtu.ignore_warning(category=np.ComplexWarning)(np_fun)
    jnp_fun = partial(jnp_op, axis=axis, dtype=out_dtype)
    jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)

    args_maker = lambda: [rng(shape, dtype)]

    tol_thresholds = {dtypes.bfloat16: 4e-2}
    tol = max(jtu.tolerance(dtype, tol_thresholds),
              jtu.tolerance(out_dtype, tol_thresholds))
    if dtype != jnp.bfloat16:
      # numpy functions do not properly handle bfloat16
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True,
                              tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_yshape={}_xshape={}_dx={}_axis={}".format(
        jtu.format_shape_dtype_string(yshape, dtype),
        jtu.format_shape_dtype_string(xshape, dtype) if xshape is not None else None,
        dx, axis),
        "yshape": yshape, "xshape": xshape, "dtype": dtype, "dx": dx, "axis": axis}
        for dtype in default_dtypes
        for yshape, xshape, dx, axis in [
          ((10,), None, 1.0, -1),
          ((3, 10), None, 2.0, -1),
          ((3, 10), None, 3.0, -0),
          ((10, 3), (10,), 1.0, -2),
          ((3, 10), (10,), 1.0, -1),
          ((3, 10), (3, 10), 1.0, -1),
          ((2, 3, 10), (3, 10), 1.0, -2),
        ]))
  @jtu.skip_on_devices("tpu")  # TODO(jakevdp): fix and reenable this test.
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testTrapz(self, yshape, xshape, dtype, dx, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(yshape, dtype), rng(xshape, dtype) if xshape is not None else None]
    np_fun = partial(np.trapz, dx=dx, axis=axis)
    jnp_fun = partial(jnp.trapz, dx=dx, axis=axis)
    tol = jtu.tolerance(dtype, {np.float64: 1e-12,
                                dtypes.bfloat16: 4e-2})
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol,
                            check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol,
                          check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}_m={}_n={}_k={}".format(
          np.dtype(dtype).name, m, n, k),
       "m": m, "n": n, "k": k, "dtype": dtype}
      for dtype in default_dtypes
      for n in [0, 4]
      for m in [None, 0, 1, 3, 4]
      for k in list(range(-4, 4))))
  def testTri(self, m, n, k, dtype):
    np_fun = lambda: np.tri(n, M=m, k=k, dtype=dtype)
    jnp_fun = lambda: jnp.tri(n, M=m, k=k, dtype=dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_shape={}_k={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "op": op, "k": k}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for op in ["tril", "triu"]
      for k in list(range(-3, 3))))
  def testTriLU(self, dtype, shape, op, k):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: getattr(np, op)(arg, k=k)
    jnp_fun = lambda arg: getattr(jnp, op)(arg, k=k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "n={}_k={}_m={}".format(n, k, m),
      "n": n, "k": k, "m": m}
    for n in range(1, 5)
    for k in [-1, 0, 1]
    for m in range(1, 5)))
  def testTrilIndices(self, n, k, m):
    np_fun = lambda n, k, m: np.tril_indices(n, k=k, m=m)
    jnp_fun = lambda n, k, m: jnp.tril_indices(n, k=k, m=m)
    args_maker = lambda: [n, k, m]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "n={}_k={}_m={}".format(n, k, m),
      "n": n, "k": k, "m": m}
    for n in range(1, 5)
    for k in [-1, 0, 1]
    for m in range(1, 5)))
  def testTriuIndices(self, n, k, m):
    np_fun = lambda n, k, m: np.triu_indices(n, k=k, m=m)
    jnp_fun = lambda n, k, m: jnp.triu_indices(n, k=k, m=m)
    args_maker = lambda: [n, k, m]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_shape={}_k={}".format(
      jtu.format_shape_dtype_string(shape, dtype), k),
      "dtype": dtype, "shape": shape, "k": k}
    for dtype in default_dtypes
    for shape in [(1,1), (1,2), (2,2), (2,3), (3,2), (3,3), (4,4)]
    for k in [-1, 0, 1]))
  def testTriuIndicesFrom(self, shape, dtype, k):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arr, k: np.triu_indices_from(arr, k=k)
    jnp_fun = lambda arr, k: jnp.triu_indices_from(arr, k=k)
    args_maker = lambda: [rng(shape, dtype), k]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_shape={}_k={}".format(
      jtu.format_shape_dtype_string(shape, dtype), k),
      "dtype": dtype, "shape": shape, "k": k}
    for dtype in default_dtypes
    for shape in [(1,1), (1,2), (2,2), (2,3), (3,2), (3,3), (4,4)]
    for k in [-1, 0, 1]))
  def testTrilIndicesFrom(self, shape, dtype, k):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arr, k: np.tril_indices_from(arr, k=k)
    jnp_fun = lambda arr, k: jnp.tril_indices_from(arr, k=k)
    args_maker = lambda: [rng(shape, dtype), k]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_ndim={}_n={}".format(ndim, n),
       "ndim": ndim, "n": n}
      for ndim in [0, 1, 4]
      for n in [0, 1, 7]))
  def testDiagIndices(self, ndim, n):
    np.testing.assert_equal(jtu.with_jax_dtype_defaults(np.diag_indices)(n, ndim),
                            jnp.diag_indices(n, ndim))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "arr_shape={}".format(
        jtu.format_shape_dtype_string(shape, dtype)
      ),
       "dtype": dtype, "shape": shape}
      for dtype in default_dtypes
      for shape in [(1,1), (2,2), (3,3), (4,4), (5,5)]))
  def testDiagIndicesFrom(self, dtype, shape):
    rng = jtu.rand_default(self.rng())
    np_fun = jtu.with_jax_dtype_defaults(np.diag_indices_from)
    jnp_fun = jnp.diag_indices_from
    args_maker = lambda : [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "k": k}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) in (1, 2)]
      for k in list(range(-4, 4))))
  def testDiag(self, shape, dtype, k):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np.diag(arg, k)
    jnp_fun = lambda arg: jnp.diag(arg, k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "k": k}
      for dtype in default_dtypes
      for shape in all_shapes
      for k in range(-4, 4)))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_a1_shape={}_a2_shape2={}".format(
          jtu.format_shape_dtype_string(a1_shape, dtype),
          jtu.format_shape_dtype_string(a2_shape, dtype)),
       "dtype": dtype, "a1_shape": a1_shape, "a2_shape": a2_shape}
      for dtype in default_dtypes
      for a1_shape in one_dim_array_shapes
      for a2_shape in one_dim_array_shapes))
  def testPolyMul(self, a1_shape, a2_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg1, arg2: np.polymul(arg1, arg2)
    jnp_fun_np = lambda arg1, arg2: jnp.polymul(arg1, arg2, trim_leading_zeros=True)
    jnp_fun_co = lambda arg1, arg2: jnp.polymul(arg1, arg2)
    args_maker = lambda: [rng(a1_shape, dtype), rng(a2_shape, dtype)]
    tol = {np.float16: 2e-1, np.float32: 5e-2, np.float64: 1e-13}
    self._CheckAgainstNumpy(np_fun, jnp_fun_np, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun_co, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_offset={}_axis1={}_axis2={}".format(
          jtu.format_shape_dtype_string(shape, dtype), offset, axis1, axis2),
       "dtype": dtype, "shape": shape, "offset": offset, "axis1": axis1,
       "axis2": axis2}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in [a for a in range(-len(shape), len(shape))
                    if a % len(shape) != axis1 % len(shape)]
      for offset in list(range(-4, 4))))
  def testDiagonal(self, shape, dtype, offset, axis1, axis2):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda arg: np.diagonal(arg, offset, axis1, axis2)
    jnp_fun = lambda arg: jnp.diagonal(arg, offset, axis1, axis2)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}".format(np.dtype(dtype).name, n),
       "dtype": dtype, "n": n}
      for dtype in default_dtypes
      for n in list(range(4))))
  def testIdentity(self, n, dtype):
    np_fun = lambda: np.identity(n, dtype)
    jnp_fun = lambda: jnp.identity(n, dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_period={}_left={}_right={}".format(
       jtu.format_shape_dtype_string(shape, dtype), period, left, right),
       "shape": shape, "dtype": dtype,
       "period": period, "left": left, "right": right}
      for shape in nonempty_shapes
      for period in [None, 0.59]
      for left in [None, 0]
      for right in [None, 1]
      for dtype in default_dtypes
      # following types lack precision for meaningful tests
      if dtype not in [np.int8, np.int16, np.float16, jnp.bfloat16]
  ))
  def testInterp(self, shape, dtype, period, left, right):
    rng = jtu.rand_default(self.rng(), scale=10)
    kwds = dict(period=period, left=left, right=right)
    np_fun = partial(np.interp, **kwds)
    jnp_fun = partial(jnp.interp, **kwds)
    args_maker = lambda: [rng(shape, dtype), np.sort(rng((20,), dtype)), np.linspace(0, 1, 20)]

    # skip numpy comparison for integer types with period specified, because numpy
    # uses an unstable sort and so results differ for duplicate values.
    if not (period and np.issubdtype(dtype, np.integer)):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol={np.float32: 2E-4})
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_x1={}_x2={}_x1_rng={}".format(
          jtu.format_shape_dtype_string(x1_shape, x1_dtype),
          jtu.format_shape_dtype_string(x2_shape, np.int32),
          x1_rng_factory_id),
       "x1_shape": x1_shape, "x1_dtype": x1_dtype,
       "x2_shape": x2_shape, "x1_rng_factory": x1_rng_factory,
       "x2_rng_factory": x2_rng_factory}
      for x1_rng_factory_id, x1_rng_factory in
        enumerate([jtu.rand_some_inf_and_nan, jtu.rand_some_zero])
      for x2_rng_factory in [partial(jtu.rand_int, low=-1075, high=1024)]
      for x1_shape, x2_shape in filter(_shapes_are_broadcast_compatible,
                                       itertools.combinations_with_replacement(array_shapes, 2))
      for x1_dtype in default_dtypes))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLdexp(self, x1_shape, x1_dtype, x2_shape, x1_rng_factory, x2_rng_factory):
    # integer types are converted to float64 in numpy's implementation
    if (x1_dtype not in [jnp.bfloat16, np.float16, np.float32]
        and not config.x64_enabled):
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    x1_rng = x1_rng_factory(self.rng())
    x2_rng = x2_rng_factory(self.rng())
    np_fun = lambda x1, x2: np.ldexp(x1, x2)
    np_fun = jtu.ignore_warning(category=RuntimeWarning,
                                 message="overflow.*")(np_fun)
    jnp_fun = lambda x1, x2: jnp.ldexp(x1, x2)
    args_maker = lambda: [x1_rng(x1_shape, x1_dtype),
                          x2_rng(x2_shape, np.int32)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_x={}_rng_factory={}".format(
          jtu.format_shape_dtype_string(shape, dtype), rng_factory_id),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory}
      for rng_factory_id, rng_factory in enumerate([
          jtu.rand_some_inf_and_nan,
          jtu.rand_some_zero,
          partial(jtu.rand_not_small, offset=1e8),
      ])
      for shape in all_shapes
      for dtype in default_dtypes))
  def testFrexp(self, shape, dtype, rng_factory):
    # integer types are converted to float64 in numpy's implementation
    if (dtype not in [jnp.bfloat16, np.float16, np.float32]
        and not config.x64_enabled):
      self.skipTest("Only run float64 testcase when float64 is enabled.")
    rng = rng_factory(self.rng())
    np_fun = lambda x: np.frexp(x)
    jnp_fun = lambda x: jnp.frexp(x)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                            check_dtypes=np.issubdtype(dtype, np.inexact))
    self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype_{}_offset={}_axis1={}_axis2={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          out_dtype, offset, axis1, axis2),
       "dtype": dtype, "out_dtype": out_dtype, "shape": shape, "offset": offset,
       "axis1": axis1, "axis2": axis2}
      for dtype in default_dtypes
      for out_dtype in [None] + number_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in range(-len(shape), len(shape))
      if (axis1 % len(shape)) != (axis2 % len(shape))
      for offset in list(range(-4, 4))))
  def testTrace(self, shape, dtype, out_dtype, offset, axis1, axis2):
    rng = jtu.rand_default(self.rng())
    def np_fun(arg):
      if out_dtype == jnp.bfloat16:
        return np.trace(arg, offset, axis1, axis2, np.float32).astype(jnp.bfloat16)
      else:
        return np.trace(arg, offset, axis1, axis2, out_dtype)
    jnp_fun = lambda arg: jnp.trace(arg, offset, axis1, axis2, out_dtype)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_a={}_v={}_side={}".format(
      jtu.format_shape_dtype_string(ashape, dtype),
      jtu.format_shape_dtype_string(vshape, dtype),
      side), "ashape": ashape, "vshape": vshape, "side": side,
     "dtype": dtype}
    for ashape in [(15,), (16,), (17,)]
    for vshape in [(), (5,), (5, 5)]
    for side in ['left', 'right']
    for dtype in number_dtypes
  ))
  def testSearchsorted(self, ashape, vshape, side, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [np.sort(rng(ashape, dtype)), rng(vshape, dtype)]
    np_fun = jtu.with_jax_dtype_defaults(
        lambda a, v: np.searchsorted(a, v, side=side))
    jnp_fun = lambda a, v: jnp.searchsorted(a, v, side=side)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": f"_dtype={dtype.__name__}_side={side}", "dtype": dtype, "side": side}
    for dtype in inexact_dtypes
    for side in ['left', 'right']))
  def testSearchsortedNans(self, dtype, side):
    if np.issubdtype(dtype, np.complexfloating):
      raise SkipTest("Known failure for complex inputs; see #9107")
    x = np.array([-np.inf, -1.0, 0.0, -0.0, 1.0, np.inf, np.nan, -np.nan], dtype=dtype)
    # The sign bit should not matter for 0.0 or NaN, so argsorting the above should be
    # equivalent to argsorting the following:
    x_equiv = np.array([0, 1, 2, 2, 3, 4, 5, 5])

    if jnp.issubdtype(dtype, jnp.complexfloating):
      x = np.array([complex(r, c) for r, c in itertools.product(x, repeat=2)])
      x_equiv = np.array([complex(r, c) for r, c in itertools.product(x_equiv, repeat=2)])

    fun = partial(jnp.searchsorted, side=side)
    self.assertArraysEqual(fun(x, x), fun(x_equiv, x_equiv))
    self.assertArraysEqual(jax.jit(fun)(x, x), fun(x_equiv, x_equiv))

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_x={}_bins={}_right={}_reverse={}".format(
      jtu.format_shape_dtype_string(xshape, dtype),
      jtu.format_shape_dtype_string(binshape, dtype),
      right, reverse), "xshape": xshape, "binshape": binshape,
      "right": right, "reverse": reverse, "dtype": dtype}
    for xshape in [(20,), (5, 4)]
    for binshape in [(1,), (5,)]
    for right in [True, False]
    for reverse in [True, False]
    for dtype in default_dtypes
  ))
  def testDigitize(self, xshape, binshape, right, reverse, dtype):
    order = jnp.index_exp[::-1] if reverse else jnp.index_exp[:]
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(xshape, dtype), jnp.sort(rng(binshape, dtype))[order]]
    np_fun = lambda x, bins: np.digitize(x, bins, right=right)
    jnp_fun = lambda x, bins: jnp.digitize(x, bins, right=right)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_array={}".format(
          jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes), array_input),
       "shape": shape, "dtypes": dtypes, "array_input": array_input}
      for dtypes in [
        [np.float32],
        [np.float32, np.float32],
        [np.float32, np.int32, np.float32],
        [np.float32, np.int64, np.float32],
        [np.float32, np.int32, np.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 5)]
      for array_input in [True, False]))
  def testColumnStack(self, shape, dtypes, array_input):
    rng = jtu.rand_default(self.rng())
    if array_input:
      args_maker = lambda: [np.array([rng(shape, dtype) for dtype in dtypes])]
    else:
      args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    np_fun = _promote_like_jnp(np.column_stack)
    jnp_fun = jnp.column_stack
    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_array={}".format(
          jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes), axis, array_input),
       "shape": shape, "axis": axis, "dtypes": dtypes, "array_input": array_input}
      for dtypes in [
        [np.float32],
        [np.float32, np.float32],
        [np.float32, np.int32, np.float32],
        [np.float32, np.int64, np.float32],
        [np.float32, np.int32, np.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 100)]
      for axis in range(-len(shape), len(shape) + 1)
      for array_input in [True, False]))
  def testStack(self, shape, axis, dtypes, array_input):
    rng = jtu.rand_default(self.rng())
    if array_input:
      args_maker = lambda: [np.array([rng(shape, dtype) for dtype in dtypes])]
    else:
      args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    np_fun = _promote_like_jnp(partial(np.stack, axis=axis))
    jnp_fun = partial(jnp.stack, axis=axis)
    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_{}_array={}".format(
          op, jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes), array_input),
       "shape": shape, "op": op, "dtypes": dtypes, "array_input": array_input}
      for op in ["hstack", "vstack", "dstack"]
      for dtypes in [
        [np.float32],
        [np.float32, np.float32],
        [np.float32, np.int32, np.float32],
        [np.float32, np.int64, np.float32],
        [np.float32, np.int32, np.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 100), (2, 3, 4)]
      for array_input in [True, False]))
  def testHVDStack(self, shape, op, dtypes, array_input):
    rng = jtu.rand_default(self.rng())
    if array_input:
      args_maker = lambda: [np.array([rng(shape, dtype) for dtype in dtypes])]
    else:
      args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    np_fun = _promote_like_jnp(getattr(np, op))
    jnp_fun = getattr(jnp, op)
    self._CheckAgainstNumpy(jnp_fun, np_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outdtype={}_fillshape={}".format(
          jtu.format_shape_dtype_string(shape, fill_value_dtype),
          np.dtype(out_dtype).name if out_dtype else "None",
          fill_value_shape),
       "fill_value_dtype": fill_value_dtype, "fill_value_shape": fill_value_shape,
       "shape": shape, "out_dtype": out_dtype}
      for shape in array_shapes + [3, np.array(7, dtype=np.int32)]
      for fill_value_dtype in default_dtypes
      for fill_value_shape in _compatible_shapes(shape)
      for out_dtype in [None] + default_dtypes))
  def testFull(self, shape, fill_value_dtype, fill_value_shape, out_dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda fill_value: np.full(shape, fill_value, dtype=out_dtype)
    jnp_fun = lambda fill_value: jnp.full(shape, fill_value, dtype=out_dtype)
    args_maker = lambda: [rng(fill_value_shape, fill_value_dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
      "testcase_name": "_shape={}_n={}_axis={}_prepend={}_append={}".format(
           jtu.format_shape_dtype_string(shape, dtype),
           n, axis, prepend, append),
        "shape": shape, "dtype": dtype, "n": n, "axis": axis,
        "prepend": prepend, "append": append
    } for shape, dtype in s(_shape_and_dtypes(nonempty_nonscalar_array_shapes, default_dtypes))
      for n in s([0, 1, 2])
      for axis in s(list(range(-len(shape), max(1, len(shape)))))
      for prepend in s([None, 1, np.zeros(shape, dtype=dtype)])
      for append in s([None, 1, np.zeros(shape, dtype=dtype)])
      )))
  def testDiff(self, shape, dtype, n, axis, prepend, append):
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

  @parameterized.named_parameters(
    jtu.cases_from_list(
      {"testcase_name": ("_op={}_shape={}_dtype={}").format(op, shape, dtype),
       "np_op": getattr(np, op), "jnp_op": getattr(jnp, op),
       "shape": shape, "dtype": dtype}
      for op in ["zeros", "ones"]
      for shape in [2, (), (2,), (3, 0), np.array((4, 5, 6), dtype=np.int32),
                    np.array(4, dtype=np.int32)]
      for dtype in all_dtypes))
  def testZerosOnes(self, np_op, jnp_op, shape, dtype):
    args_maker = lambda: []
    np_op = partial(np_op, shape, dtype)
    jnp_op = partial(jnp_op, shape, dtype)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  def testOnesWithInvalidShape(self):
    with self.assertRaises(TypeError):
      jnp.ones((-1, 1))

  @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
       "testcase_name": "_inshape={}_filldtype={}_fillshape={}_outdtype={}_outshape={}".format(
          jtu.format_shape_dtype_string(shape, in_dtype),
          np.dtype(fill_value_dtype).name, fill_value_shape,
          np.dtype(out_dtype).name, out_shape),
       "shape": shape, "in_dtype": in_dtype,
       "fill_value_dtype": fill_value_dtype, "fill_value_shape": fill_value_shape,
       "out_dtype": out_dtype, "out_shape": out_shape
    } for shape in s(array_shapes)
      for out_shape in s([None] + array_shapes)
      for in_dtype in s(default_dtypes)
      for fill_value_dtype in s(default_dtypes)
      for fill_value_shape in s(_compatible_shapes(shape if out_shape is None else out_shape))
      for out_dtype in s(default_dtypes))))
  def testFullLike(self, shape, in_dtype, fill_value_dtype, fill_value_shape, out_dtype, out_shape):
    if numpy_version < (1, 19) and out_shape == ():
      raise SkipTest("Numpy < 1.19 treats out_shape=() like out_shape=None")
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x, fill_value: np.full_like(
      x, fill_value, dtype=out_dtype, shape=out_shape)
    jnp_fun = lambda x, fill_value: jnp.full_like(
      x, fill_value, dtype=out_dtype, shape=out_shape)
    args_maker = lambda: [rng(shape, in_dtype), rng(fill_value_shape, fill_value_dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_func={}_inshape={}_outshape={}_outdtype={}".format(
          func, jtu.format_shape_dtype_string(shape, in_dtype),
          out_shape, out_dtype),
       "func": func, "shape": shape, "in_dtype": in_dtype,
       "out_shape": out_shape, "out_dtype": out_dtype}
      for shape in array_shapes
      for out_shape in [None] + array_shapes
      for in_dtype in default_dtypes
      for func in ["ones_like", "zeros_like"]
      for out_dtype in default_dtypes))
  def testZerosOnesLike(self, func, shape, in_dtype, out_shape, out_dtype):
    if numpy_version < (1, 19) and out_shape == ():
      raise SkipTest("Numpy < 1.19 treats out_shape=() like out_shape=None")
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: getattr(np, func)(x, dtype=out_dtype, shape=out_shape)
    jnp_fun = lambda x: getattr(jnp, func)(x, dtype=out_dtype, shape=out_shape)
    args_maker = lambda: [rng(shape, in_dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_func={}_inshape={}_weak_type={}_outshape={}_outdtype={}".format(
          func, jtu.format_shape_dtype_string(shape, in_dtype),
          weak_type, out_shape, out_dtype),
       "func": func, "args": args,
       "shape": shape, "in_dtype": in_dtype, "weak_type": weak_type,
       "out_shape": out_shape, "out_dtype": out_dtype}
      for shape in array_shapes
      for in_dtype in [np.int32, np.float32, np.complex64]
      for weak_type in [True, False]
      for out_shape in [None, (), (10,)]
      for func, args in [("full_like", (-100,)), ("ones_like", ()), ("zeros_like", ())]
      for out_dtype in [None, float]))
  def testZerosOnesFullLikeWeakType(self, func, args, shape, in_dtype, weak_type, out_shape, out_dtype):
    if numpy_version < (1, 19) and out_shape == ():
      raise SkipTest("Numpy < 1.19 treats out_shape=() like out_shape=None")
    rng = jtu.rand_default(self.rng())
    x = lax_internal._convert_element_type(rng(shape, in_dtype),
                                           weak_type=weak_type)
    fun = lambda x: getattr(jnp, func)(x, *args, dtype=out_dtype, shape=out_shape)
    expected_weak_type = weak_type and (out_dtype is None)
    self.assertEqual(dtypes.is_weakly_typed(fun(x)), expected_weak_type)
    self.assertEqual(dtypes.is_weakly_typed(jax.jit(fun)(x)), expected_weak_type)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_funcname={}_input_type={}_val={}_dtype={}".format(
          funcname, input_type, val, dtype),
       "funcname": funcname, "input_type": input_type, "val": val, "dtype": dtype}
      for funcname in ["array", "asarray"]
      for dtype in [int, float, None]
      for val in [0, 1]
      for input_type in [int, float, np.int32, np.float32]))
  def testArrayWeakType(self, funcname, input_type, val, dtype):
    func = lambda x: getattr(jnp, funcname)(x, dtype=dtype)
    fjit = jax.jit(func)
    val = input_type(val)
    expected_weak_type = dtype is None and input_type in set(dtypes._weak_types)
    self.assertEqual(dtypes.is_weakly_typed(func(val)), expected_weak_type)
    self.assertEqual(dtypes.is_weakly_typed(fjit(val)), expected_weak_type)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_weak_type={}_slc={}".format(
        jtu.format_shape_dtype_string(shape, dtype), weak_type, slc),
       "shape": shape, "dtype": dtype, "weak_type": weak_type, "slc": slc}
      for shape in nonempty_nonscalar_array_shapes
      for dtype in [int, float, complex]
      for weak_type in [True, False]
      for slc in [slice(None), slice(0), slice(3), 0, ...]))
  def testSliceWeakTypes(self, shape, dtype, weak_type, slc):
    rng = jtu.rand_default(self.rng())
    x = lax_internal._convert_element_type(rng(shape, dtype),
                                           weak_type=weak_type)
    op = lambda x: x[slc]
    self.assertEqual(op(x).aval.weak_type, weak_type)
    self.assertEqual(jax.jit(op)(x).aval.weak_type, weak_type)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis,
       "dtype": dtype}
      for shape, axis, num_sections in [
          ((3,), 0, 3), ((12,), 0, 3), ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), -1, 2), ((2, 3, 4), -2, 3)]
      for dtype in default_dtypes))
  def testSplitStaticInt(self, shape, num_sections, axis, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.split(x, num_sections, axis=axis)
    jnp_fun = lambda x: jnp.split(x, num_sections, axis=axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis, "dtype": dtype}
      # All testcases split the specified axis unequally
      for shape, axis, num_sections in [
          ((3,), 0, 2), ((12,), 0, 5), ((12, 4), 0, 7), ((12, 4), 1, 3),
          ((2, 3, 5), -1, 2), ((2, 4, 4), -2, 3), ((7, 2, 2), 0, 3)]
      for dtype in default_dtypes))
  def testArraySplitStaticInt(self, shape, num_sections, axis, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.array_split(x, num_sections, axis=axis)
    jnp_fun = lambda x: jnp.array_split(x, num_sections, axis=axis)
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
    jax.jvp(lambda idx: jnp.split(jnp.zeros((12, 2)), (1, idx)),
            (2.,), (1.,))

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_bins={}_range={}_weights={}".format(
      jtu.format_shape_dtype_string(shape, dtype), bins, range, weights),
      "shape": shape,
      "dtype": dtype,
      "bins": bins,
      "range": range,
      "weights": weights,
    }
    for shape in [(5,), (5, 5)]
    for dtype in number_dtypes
    for bins in [10, np.arange(-5, 6), np.array([-5, 0, 3])]
    for range in [None, (0, 0), (0, 10)]
    for weights in [True, False]
  ))
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

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_bins={}_density={}_weights={}".format(
      jtu.format_shape_dtype_string(shape, dtype), bins, density, weights),
      "shape": shape,
      "dtype": dtype,
      "bins": bins,
      "density": density,
      "weights": weights,
    }
    for shape in [(5,), (5, 5)]
    for dtype in default_dtypes
    # We only test explicit integer-valued bin edges because in other cases
    # rounding errors lead to flaky tests.
    for bins in [np.arange(-5, 6), np.array([-5, 0, 3])]
    for density in [True, False]
    for weights in [True, False]
  ))
  def testHistogram(self, shape, dtype, bins, density, weights):
    rng = jtu.rand_default(self.rng())
    _weights = lambda w: abs(w) if weights else None
    np_fun = lambda a, w: np.histogram(a, bins=bins, density=density,
                                         weights=_weights(w))
    jnp_fun = lambda a, w: jnp.histogram(a, bins=bins, density=density,
                                         weights=_weights(w))
    args_maker = lambda: [rng(shape, dtype), rng(shape, dtype)]
    tol = {jnp.bfloat16: 2E-2, np.float16: 1E-1}
    # np.searchsorted errors on bfloat16 with
    # "TypeError: invalid type promotion with custom data type"
    if dtype != jnp.bfloat16:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False,
                              tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_bins={}_weights={}_density={}_range={}".format(
      jtu.format_shape_dtype_string(shape, dtype), bins, weights, density, range),
      "shape": shape, "dtype": dtype, "bins": bins, "weights": weights, "density": density, "range": range,
    }
    for shape in [(5,), (12,)]
    for dtype in int_dtypes
    for bins in [2, [2, 2], [np.array([0, 1, 3, 5]), np.array([0, 2, 3, 4, 6])]]
    for weights in [False, True]
    for density in [False, True]
    for range in [None, [(-1, 1), None], [(-1, 1), (-2, 2)]]
  ))
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

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}_bins={}_weights={}_density={}_range={}".format(
      jtu.format_shape_dtype_string(shape, dtype), bins, weights, density, range),
      "shape": shape, "dtype": dtype, "bins": bins, "weights": weights, "density": density, "range": range,
    }
    for shape in [(5, 3), (10, 3)]
    for dtype in int_dtypes
    for bins in [(2, 2, 2), [np.array([-5, 0, 4]), np.array([-4, -1, 2]), np.array([-6, -1, 4])]]
    for weights in [False, True]
    for density in [False, True]
    for range in [None, [(-1, 1), None, None], [(-1, 1), (-2, 2), (-3, 3)]]
  ))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis,
       "dtype": dtype}
      for shape, axis, num_sections in [
          ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), 2, 2), ((4, 3, 4), 0, 2)]
      for dtype in default_dtypes))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_order={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype),
          order),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "order": order}
      for dtype in default_dtypes
      for order in ["C", "F"]
      for arg_shape, out_shape in [
          (jtu.NUMPY_SCALAR_SHAPE, (1, 1, 1)),
          ((), (1, 1, 1)),
          ((7, 0), (0, 42, 101)),
          ((3, 4), 12),
          ((3, 4), (12,)),
          ((3, 4), -1),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]))
  def testReshape(self, arg_shape, out_shape, dtype, order):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.reshape(x, out_shape, order=order)
    jnp_fun = lambda x: jnp.reshape(x, out_shape, order=order)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype}
      for dtype in default_dtypes
      for arg_shape, out_shape in [
          ((7, 0), (0, 42, 101)),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]))
  def testReshapeMethod(self, arg_shape, out_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.reshape(x, out_shape)
    jnp_fun = lambda x: x.reshape(*out_shape)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype}
      for dtype in default_dtypes
      for arg_shape, out_shape in itertools.product(all_shapes, array_shapes)))
  def testResize(self, arg_shape, out_shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.resize(x, out_shape)
    jnp_fun = lambda x: jnp.resize(x, out_shape)
    args_maker = lambda: [rng(arg_shape, dtype)]
    if len(out_shape) > 0 or numpy_version >= (1, 20, 0):
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_expanddim={!r}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), dim),
       "arg_shape": arg_shape, "dtype": dtype, "dim": dim}
      for arg_shape in [(), (3,), (3, 4)]
      for dtype in default_dtypes
      for dim in (list(range(-len(arg_shape)+1, len(arg_shape)))
                  + [np.array(0), np.array(-1), (0,), [np.array(0)],
                     (len(arg_shape), len(arg_shape) + 1)])))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axes=({},{})".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), ax1, ax2),
       "arg_shape": arg_shape, "dtype": dtype, "ax1": ax1, "ax2": ax2}
      for arg_shape, ax1, ax2 in [
          ((3, 4), 0, 1), ((3, 4), 1, 0), ((3, 4, 5), 1, 2),
          ((3, 4, 5), -1, -2), ((3, 4, 5), 0, 1)]
      for dtype in default_dtypes))
  def testSwapAxesStaticAxes(self, arg_shape, dtype, ax1, ax2):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.swapaxes(x, ax1, ax2)
    jnp_fun = lambda x: jnp.swapaxes(x, ax1, ax2)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axis={!r}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), ax),
       "arg_shape": arg_shape, "dtype": dtype, "ax": ax}
      for arg_shape, ax in [
          ((3, 1), None),
          ((3, 1), 1),
          ((3, 1), -1),
          ((3, 1), np.array(1)),
          ((1, 3, 1), (0, 2)),
          ((1, 3, 1), (0,)),
          ((1, 4, 1), (np.array(0),))]
      for dtype in default_dtypes))
  def testSqueeze(self, arg_shape, dtype, ax):
    rng = jtu.rand_default(self.rng())
    np_fun = lambda x: np.squeeze(x, ax)
    jnp_fun = lambda x: jnp.squeeze(x, ax)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}_weights={}_returned={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis,
          (None if weights_shape is None else jtu.format_shape_dtype_string(weights_shape, dtype)),
          returned),
       "shape": shape, "dtype": dtype, "axis": axis,
       "weights_shape": weights_shape, "returned": returned}
      for shape, dtype in _shape_and_dtypes(nonempty_shapes, number_dtypes)
      for axis in list(range(-len(shape), len(shape))) + [None]
      # `weights_shape` is either `None`, same as the averaged axis, or same as
      # that of the input
      for weights_shape in ([None, shape] if axis is None or len(shape) == 1
                            else [None, (shape[axis],), shape])
      for returned in [False, True]))
  def testAverage(self, shape, dtype, axis, weights_shape, returned):
    rng = jtu.rand_default(self.rng())
    if weights_shape is None:
      np_fun = lambda x: np.average(x, axis, returned=returned)
      jnp_fun = lambda x: jnp.average(x, axis, returned=returned)
      args_maker = lambda: [rng(shape, dtype)]
    else:
      np_fun = lambda x, weights: np.average(x, axis, weights, returned)
      jnp_fun = lambda x, weights: jnp.average(x, axis, weights, returned)
      args_maker = lambda: [rng(shape, dtype), rng(weights_shape, dtype)]
    np_fun = _promote_like_jnp(np_fun, inexact=True)
    tol = {dtypes.bfloat16: 2e-1, np.float16: 1e-2, np.float32: 1e-5,
           np.float64: 1e-12, np.complex64: 1e-5}
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    try:
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker,
                              check_dtypes=check_dtypes, tol=tol)
    except ZeroDivisionError:
      self.skipTest("don't support checking for ZeroDivisionError")
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=check_dtypes,
                          rtol=tol, atol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       f"_arg{i}_ndmin={ndmin}_dtype={np.dtype(dtype) if dtype else None}",
       "arg": arg, "ndmin": ndmin, "dtype": dtype}
      for i, (arg, dtypes) in enumerate([
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
      ])
      for dtype in [None] + dtypes
      for ndmin in [None, np.ndim(arg), np.ndim(arg) + 1, np.ndim(arg) + 2]))
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
    _check([1], jnp.int_, False)
    _check([1.0], jnp.float_, False)
    _check([1.0j], jnp.complex_, False)

    # Lists of weakly-typed objects become strongly-typed defaults.
    _check([jnp.array(1)], jnp.int_, False)
    _check([jnp.array(1.0)], jnp.float_, False)
    _check([jnp.array(1.0j)], jnp.complex_, False)

    # Lists of strongly-typed objects maintain their strong type.
    _check([jnp.int64(1)], np.int64, False)
    _check([jnp.float64(1)], np.float64, False)
    _check([jnp.complex128(1)], np.complex128, False)

    # Mixed inputs use JAX-style promotion.
    # (regression test for https://github.com/google/jax/issues/8945)
    _check([0, np.int16(1)], np.int16, False)
    _check([0.0, np.float16(1)], np.float16, False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_dtype={np.dtype(dtype)}_func={func}",
       "dtype": dtype, "func": func}
      for dtype in all_dtypes
      for func in ["array", "copy"]))
  def testArrayCopy(self, dtype, func):
    x = jnp.ones(10, dtype=dtype)
    copy_func = getattr(jnp, func)

    x_view = jnp.asarray(x)
    x_view_jit = jax.jit(jnp.asarray)(x)
    x_copy = copy_func(x)
    x_copy_jit = jax.jit(copy_func)(x)

    _ptr = lambda x: x.device_buffer.unsafe_buffer_pointer()

    self.assertEqual(_ptr(x), _ptr(x_view))
    self.assertEqual(_ptr(x), _ptr(x_view_jit))
    self.assertNotEqual(_ptr(x), _ptr(x_copy))
    self.assertNotEqual(_ptr(x), _ptr(x_copy_jit))

    x.delete()

    self.assertTrue(x_view.is_deleted())
    self.assertTrue(x_view_jit.is_deleted())

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

  def testArrayUnsupportedDtypeError(self):
    with self.assertRaisesRegex(TypeError,
                                "JAX only supports number and bool dtypes.*"):
      jnp.array(3, [('a','<i4'),('b','<i4')])

  def testArrayFromInteger(self):
    int_dtype = dtypes.canonicalize_dtype(jnp.int64)
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
    if config.x64_enabled:
      self.assertEqual(np.uint64(val), jnp.array(val, dtype='uint64'))

  def testArrayFromList(self):
    int_max = jnp.iinfo(jnp.int64).max
    int_min = jnp.iinfo(jnp.int64).min

    # Values at extremes are converted correctly.
    for val in [int_min, 0, int_max]:
      self.assertEqual(jnp.array([val]).dtype, dtypes.canonicalize_dtype('int64'))

    # list of values results in promoted type.
    self.assertEqual(jnp.array([0, np.float16(1)]).dtype, jnp.result_type('int64', 'float16'))

    # out of bounds leads to an OverflowError
    val = int_min - 1
    with self.assertRaisesRegex(OverflowError, "Python int too large.*"):
      jnp.array([0, val])

  def testIssue121(self):
    assert not np.isscalar(jnp.array(3))

  def testArrayOutputsDeviceArrays(self):
    assert device_array.type_is_device_array(jnp.array([]))
    assert device_array.type_is_device_array(jnp.array(np.array([])))

    class NDArrayLike:
      def __array__(self, dtype=None):
        return np.array([], dtype=dtype)
    assert device_array.type_is_device_array(jnp.array(NDArrayLike()))

    # NOTE(mattjj): disabled b/c __array__ must produce ndarrays
    # class DeviceArrayLike:
    #     def __array__(self, dtype=None):
    #         return jnp.array([], dtype=dtype)
    # assert  xla.type_is_device_array(jnp.array(DeviceArrayLike()))

  def testArrayMethod(self):
    class arraylike(object):
      dtype = np.dtype('float32')
      def __array__(self, dtype=None):
        return np.array(3., dtype=dtype)
    a = arraylike()
    ans = jnp.array(a)
    self.assertEqual(ans, 3.)

  def testJaxArrayOps(self):
    class arraylike:
      def __jax_array__(self):
        return jnp.array(3.)
    self.assertArraysEqual(arraylike() * jnp.arange(10), jnp.array(3.) * jnp.arange(10))

  def testMemoryView(self):
    self.assertAllClose(
        jnp.array(bytearray(b'\x2a')),
        np.array(bytearray(b'\x2a'))
    )
    self.assertAllClose(
        jnp.array(bytearray(b'\x2a\xf3'), ndmin=2),
        np.array(bytearray(b'\x2a\xf3'), ndmin=2)
    )

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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_x={}_y={}_equal_nan={}".format(x, y, equal_nan),
       "x": x, "y": y, "equal_nan": equal_nan}
      for x, y in itertools.product([
         1, [1], [1, 1 + 1E-4], [1, np.nan]], repeat=2)
      for equal_nan in [True, False]))
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
      self.assertIsInstance(x, jnp.ndarray)
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

  def testTracingPrimitiveWithNoTranslationErrorMessage(self):
    # TODO(mattjj): update this for jax3
    self.skipTest("test needs jax3 update")
    foo = jnp._not_implemented(lambda x: x)

    # No error if there's no tracing.
    foo(np.arange(3))

    cfoo = jax.jit(foo)
    self.assertRaises(NotImplementedError, lambda: cfoo(np.arange(3)))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in [(3,), (2, 3)]
      for dtype in default_dtypes
      for axis in list(range(-len(shape), len(shape))) + [None] + [tuple(range(len(shape)))]  # Test negative axes and tuples
    ))
  def testFlip(self, shape, dtype, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.flip(x, axis)
    np_op = lambda x: np.flip(x, axis)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(3,), (2, 3), (3, 2, 4)]
      for dtype in default_dtypes))
  def testFlipud(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.flipud(x)
    np_op = lambda x: np.flipud(x)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(3, 2), (2, 3), (3, 2, 4)]
      for dtype in default_dtypes))
  def testFliplr(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_op = lambda x: jnp.fliplr(x)
    np_op = lambda x: np.fliplr(x)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_k={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k, axes),
       "shape": shape, "dtype": dtype, "k": k, "axes": axes}
      for shape, axes in [
          [(2, 3), (0, 1)],
          [(2, 3), (1, 0)],
          [(4, 3, 2), (0, 2)],
          [(4, 3, 2), (2, 1)],
      ]
      for k in range(-3, 4)
      for dtype in default_dtypes))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_order={}_mode={}".format(
          shape, order, mode),
        "shape": shape, "order": order, "mode": mode}
      for shape in nonempty_nonscalar_array_shapes
      for order in ['C', 'F']
      for mode in ['wrap', 'clip', 'raise']))
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
      with self.assertRaisesRegex(jax.core.ConcretizationTypeError, msg):
        jax.jit(jnp_fun)(*args_maker())
    else:
      self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_ashape={}{}_cshapes={}{}_mode={}".format(
          adtype.__name__, ashape, cdtype.__name__, cshapes, mode),
        "ashape": ashape, "adtype": adtype, "cshapes": cshapes, "cdtype": cdtype, "mode": mode}
      for ashape in ((), (4,), (3, 4))
      for cshapes in [
        [(), (4,)],
        [(3, 4), (4,), (3, 1)]
      ]
      for adtype in int_dtypes
      for cdtype in default_dtypes
      for mode in ['wrap', 'clip', 'raise']))
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
      with self.assertRaisesRegex(jax.core.ConcretizationTypeError, msg):
        jax.jit(jnp_fun)(*args_maker())
    else:
      self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.parameters(
    (0, (2, 1, 3)),
    (5, (2, 1, 3)),
    (0, ()),
    (np.array([0, 1, 2]), (2, 2)),
    (np.array([[[0, 1], [2, 3]]]), (2, 2)))
  def testUnravelIndex(self, flat_index, shape):
    args_maker = lambda: (flat_index, shape)
    np_fun = jtu.with_jax_dtype_defaults(np.unravel_index, use_defaults=not hasattr(flat_index, 'dtype'))
    self._CheckAgainstNumpy(np_fun, jnp.unravel_index, args_maker)
    self._CompileAndCheck(jnp.unravel_index, args_maker)

  def testUnravelIndexOOB(self):
    self.assertEqual(jnp.unravel_index(2, (2,)), (1,))
    self.assertEqual(jnp.unravel_index(-2, (2, 1, 3,)), (1, 0, 1))
    self.assertEqual(jnp.unravel_index(-3, (2,)), (0,))

  def testAstype(self):
    rng = self.rng()
    args_maker = lambda: [rng.randn(3, 4).astype("float32")]
    np_op = lambda x: np.asarray(x).astype(jnp.int32)
    jnp_op = lambda x: jnp.asarray(x).astype(jnp.int32)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  def testAstypeNone(self):
    rng = self.rng()
    args_maker = lambda: [rng.randn(3, 4).astype("int32")]
    np_op = jtu.with_jax_dtype_defaults(lambda x: np.asarray(x).astype(None))
    jnp_op = lambda x: jnp.asarray(x).astype(None)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
      "shape": shape, "dtype": dtype}
      for shape in array_shapes
      for dtype in all_dtypes))
  def testNbytes(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_op = lambda x: np.asarray(x).nbytes
    jnp_op = lambda x: jnp.asarray(x).nbytes
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
      "shape": shape, "dtype": dtype}
      for shape in array_shapes
      for dtype in all_dtypes))
  def testItemsize(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    np_op = lambda x: np.asarray(x).itemsize
    jnp_op = lambda x: jnp.asarray(x).itemsize
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_dtype={}".format(
          jtu.format_shape_dtype_string(shape, a_dtype), dtype),
      "shape": shape, "a_dtype": a_dtype, "dtype": dtype}
      for shape in [(8,), (3, 8)]  # last dim = 8 to ensure shape compatibility
      for a_dtype in (default_dtypes + unsigned_dtypes + bool_dtypes)
      for dtype in (default_dtypes + unsigned_dtypes + bool_dtypes)))
  def testView(self, shape, a_dtype, dtype):
    if jtu.device_under_test() == 'tpu':
      if jnp.dtype(a_dtype).itemsize in [1, 2] or jnp.dtype(dtype).itemsize in [1, 2]:
        self.skipTest("arr.view() not supported on TPU for 8- or 16-bit types.")
    if not config.x64_enabled:
      if jnp.dtype(a_dtype).itemsize == 8 or jnp.dtype(dtype).itemsize == 8:
        self.skipTest("x64 types are disabled by jax_enable_x64")
    rng = jtu.rand_fullrange(self.rng())
    args_maker = lambda: [rng(shape, a_dtype)]
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
    # from https://github.com/google/jax/issues/125
    x = jnp.eye(3, dtype=float) + 0.
    ans = np.mean(x)
    self.assertAllClose(ans, np.array(1./3), check_dtypes=False)

  def testArangeOnFloats(self):
    np_arange = jtu.with_jax_dtype_defaults(np.arange)
    # from https://github.com/google/jax/issues/145
    self.assertAllClose(np_arange(0.0, 1.0, 0.1),
                        jnp.arange(0.0, 1.0, 0.1))
    # from https://github.com/google/jax/issues/3450
    self.assertAllClose(np_arange(2.5),
                        jnp.arange(2.5))

  def testArangeTypes(self):
    # Test that arange() output type is equal to the default types.
    int_ = dtypes.canonicalize_dtype(jnp.int_)
    float_ = dtypes.canonicalize_dtype(jnp.float_)

    self.assertEqual(jnp.arange(10).dtype, int_)
    self.assertEqual(jnp.arange(10.).dtype, float_)
    self.assertEqual(jnp.arange(10, dtype='uint16').dtype, np.uint16)
    self.assertEqual(jnp.arange(10, dtype='bfloat16').dtype, jnp.bfloat16)

    self.assertEqual(jnp.arange(0, 10, 1).dtype, int_)
    self.assertEqual(jnp.arange(0, 10, 1.).dtype, float_)
    self.assertEqual(jnp.arange(0., 10, 1).dtype, float_)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in all_dtypes
      for shape in nonzerodim_shapes
      for axis in (None, *range(len(shape)))))
  def testSort(self, dtype, shape, axis):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_fun = jnp.sort
    np_fun = np.sort
    if axis is not None:
      jnp_fun = partial(jnp_fun, axis=axis)
      np_fun = partial(np_fun, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
        "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in all_dtypes
      for shape in one_dim_array_shapes
      for axis in [None]))
  def testSortComplex(self, dtype, shape, axis):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np.sort_complex, jnp.sort_complex, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp.sort_complex, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_input_type={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          input_type.__name__, axis),
       "shape": shape, "dtype": dtype, "input_type": input_type, "axis": axis}
      for dtype in all_dtypes
      for shape in nonempty_nonscalar_array_shapes
      for input_type in [np.array, tuple]
      for axis in (-1, *range(len(shape) - 1))))
  def testLexsort(self, dtype, shape, input_type, axis):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [input_type(rng(shape, dtype))]
    jnp_op = lambda x: jnp.lexsort(x, axis=axis)
    np_op = jtu.with_jax_dtype_defaults(lambda x: np.lexsort(x, axis=axis))
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for dtype in all_dtypes
      for shape in nonzerodim_shapes
      for axis in (None, *range(len(shape)))))
  def testArgsort(self, dtype, shape, axis):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_fun = jnp.argsort
    np_fun = jtu.with_jax_dtype_defaults(np.argsort)
    if axis is not None:
      jnp_fun = partial(jnp_fun, axis=axis)
      np_fun = partial(np_fun, axis=axis)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for dtype in all_dtypes
      for shape in nonzerodim_shapes))
  def testMsort(self, dtype, shape):
    rng = jtu.rand_some_equal(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np.msort, jnp.msort, args_maker)
    self._CompileAndCheck(jnp.msort, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_shifts={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          shifts, axis),
       "shape": shape, "dtype": dtype, "shifts": shifts, "axis": axis}
      for dtype in all_dtypes
      for shape in [(3, 4), (3, 4, 5), (7, 4, 0)]
      for shifts, axis in [
        (3, None),
        (1, 1),
        ((3,), (0,)),
        ((-2,), (-2,)),
        ((1, 2), (0, -1)),
        ((4, 2, 5, 5, 2, 4), None),
        (100, None),
      ]))
  def testRoll(self, shape, dtype, shifts, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), np.array(shifts)]
    jnp_op = partial(jnp.roll, axis=axis)
    np_op = partial(np.roll, axis=axis)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_start={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis, start),
       "shape": shape, "dtype": dtype, "axis": axis,
       "start": start}
      for dtype in all_dtypes
      for shape in [(1, 2, 3, 4)]
      for axis in [-3, 0, 2, 3]
      for start in [-4, -1, 2, 4]))
  def testRollaxis(self, shape, dtype, start, axis):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.rollaxis, axis=axis, start=start)
    np_op = partial(np.rollaxis, axis=axis, start=start)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_bitorder={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, bitorder),
       "shape": shape, "dtype": dtype, "axis": axis,
       "bitorder": bitorder}
      for dtype in [np.uint8, np.bool_]
      for bitorder in ['big', 'little']
      for shape in [(1, 2, 3, 4)]
      for axis in [None, 0, 1, -2, -1]))
  def testPackbits(self, shape, dtype, axis, bitorder):
    rng = jtu.rand_some_zero(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.packbits, axis=axis, bitorder=bitorder)
    np_op = partial(np.packbits, axis=axis, bitorder=bitorder)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_bitorder={}_count={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, bitorder, count),
      "shape": shape, "dtype": dtype, "axis": axis, "bitorder": bitorder,
      "count": count}
      for dtype in [np.uint8]
      for bitorder in ['big', 'little']
      for shape in [(1, 2, 3, 4)]
      for axis in [None, 0, 1, -2, -1]
      for count in [None, 20]))
  def testUnpackbits(self, shape, dtype, axis, bitorder, count):
    rng = jtu.rand_int(self.rng(), 0, 256)
    args_maker = lambda: [rng(shape, dtype)]
    jnp_op = partial(jnp.unpackbits, axis=axis, bitorder=bitorder)
    np_op = partial(np.unpackbits, axis=axis, bitorder=bitorder)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_index={}_axis={}_mode={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          jtu.format_shape_dtype_string(index_shape, index_dtype),
          axis, mode),
       "shape": shape, "index_shape": index_shape, "dtype": dtype,
       "index_dtype": index_dtype, "axis": axis, "mode": mode}
      for shape in [(3,), (3, 4), (3, 4, 5)]
      for index_shape in scalar_shapes + [(3,), (2, 1, 3)]
      for axis in itertools.chain(range(-len(shape), len(shape)),
                                  [cast(Optional[int], None)])
      for dtype in all_dtypes
      for index_dtype in int_dtypes
      for mode in [None, 'wrap', 'clip']))
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

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_index={}_axis={}".format(
          jtu.format_shape_dtype_string(x_shape, dtype),
          jtu.format_shape_dtype_string(i_shape, index_dtype), axis),
       "x_shape": x_shape, "i_shape": i_shape, "dtype": dtype,
       "index_dtype": index_dtype, "axis": axis}
      for x_shape, i_shape in filter(
        _shapes_are_equal_length,
        filter(_shapes_are_broadcast_compatible,
               itertools.combinations_with_replacement(nonempty_nonscalar_array_shapes, 2)))
      for axis in itertools.chain(range(len(x_shape)), [-1],
                                  [cast(Optional[int], None)])
      for dtype in default_dtypes
      for index_dtype in int_dtypes))
  def testTakeAlongAxis(self, x_shape, i_shape, dtype, index_dtype, axis):
    rng = jtu.rand_default(self.rng())

    i_shape = np.array(i_shape)
    if axis is None:
      i_shape = [np.prod(i_shape, dtype=np.int64)]
    else:
      # Test the case where the size of the axis doesn't necessarily broadcast.
      i_shape[axis] *= 3
      i_shape = list(i_shape)
    def args_maker():
      x = rng(x_shape, dtype)
      n = np.prod(x_shape, dtype=np.int32) if axis is None else x_shape[axis]
      if np.issubdtype(index_dtype, np.unsignedinteger):
        index_rng = jtu.rand_int(self.rng(), 0, n)
      else:
        index_rng = jtu.rand_int(self.rng(), -n, n)
      i = index_rng(i_shape, index_dtype)
      return x, i

    jnp_op = lambda x, i: jnp.take_along_axis(x, i, axis=axis)

    if hasattr(np, "take_along_axis"):
      np_op = lambda x, i: np.take_along_axis(x, i, axis=axis)
      self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  def testTakeAlongAxisWithUint8IndicesDoesNotOverflow(self):
    # https://github.com/google/jax/issues/5088
    h = jtu.rand_default(self.rng())((256, 256, 100), np.float32)
    g = jtu.rand_int(self.rng(), 0, 100)((256, 256, 1), np.uint8)
    q0 = jnp.take_along_axis(h, g, axis=-1)
    q1 = np.take_along_axis( h, g, axis=-1)
    np.testing.assert_equal(q0, q1)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}_increasing={}".format(
          jtu.format_shape_dtype_string([shape], dtype),
          n, increasing),
       "dtype": dtype, "shape": shape, "n": n, "increasing": increasing}
      for dtype in inexact_dtypes
      for shape in [0, 5]
      for n in [2, 4]
      for increasing in [False, True]))
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
                            tol={np.float32: 1e-3})
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            "nan_to_num", [shape], [dtype]),
         "shape": shape, "dtype": dtype}
        for shape in array_shapes
        for dtype in inexact_dtypes))
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

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix("ix_", shapes, dtypes),
         "shapes": shapes, "dtypes": dtypes}
        for shapes, dtypes in (
          ((), ()),
          (((7,),), (np.int32,)),
          (((3,), (4,)), (np.int32, np.int32)),
          (((3,), (1,), (4,)), (np.int32, np.int32, np.int32)),
        )))
  def testIx_(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)
                          for shape, dtype in zip(shapes, dtypes)]
    self._CheckAgainstNumpy(np.ix_, jnp.ix_, args_maker)
    self._CompileAndCheck(jnp.ix_, args_maker)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_dimensions={}_dtype={}_sparse={}".format(
            dimensions, dtype, sparse),
         "dimensions": dimensions, "dtype": dtype, "sparse": sparse}
        for dimensions in [(), (2,), (3, 0), (4, 5, 6)]
        for dtype in number_dtypes
        for sparse in [True, False]))
  def testIndices(self, dimensions, dtype, sparse):
    def args_maker(): return []
    np_fun = partial(np.indices, dimensions=dimensions,
                     dtype=dtype, sparse=sparse)
    jnp_fun = partial(jnp.indices, dimensions=dimensions,
                      dtype=dtype, sparse=sparse)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name":
           "_op={}_a_shape={}_q_shape={}_axis={}_keepdims={}_method={}".format(
             op,
             jtu.format_shape_dtype_string(a_shape, a_dtype),
             jtu.format_shape_dtype_string(q_shape, q_dtype),
             axis, keepdims, method),
         "a_rng": jtu.rand_some_nan,
         "q_rng": q_rng, "op": op,
         "a_shape": a_shape, "a_dtype": a_dtype,
         "q_shape": q_shape, "q_dtype": q_dtype, "axis": axis,
         "keepdims": keepdims,
         "method": method}
        for (op, q_rng) in (
          ("percentile", partial(jtu.rand_uniform, low=0., high=100.)),
          ("quantile", partial(jtu.rand_uniform, low=0., high=1.)),
          ("nanpercentile", partial(jtu.rand_uniform, low=0., high=100.)),
          ("nanquantile", partial(jtu.rand_uniform, low=0., high=1.)),
        )
        for a_dtype in default_dtypes
        for a_shape, axis in (
          ((7,), None),
          ((47, 7), 0),
          ((47, 7), ()),
          ((4, 101), 1),
          ((4, 47, 7), (1, 2)),
          ((4, 47, 7), (0, 2)),
          ((4, 47, 7), (1, 0, 2)),
        )
        for q_dtype in [np.float32]
        for q_shape in scalar_shapes + [(1,), (4,)]
        for keepdims in [False, True]
        for method in ['linear', 'lower', 'higher', 'nearest', 'midpoint']))
  def testQuantile(self, op, a_rng, q_rng, a_shape, a_dtype, q_shape, q_dtype,
                   axis, keepdims, method):
    a_rng = a_rng(self.rng())
    q_rng = q_rng(self.rng())
    if "median" in op:
      args_maker = lambda: [a_rng(a_shape, a_dtype)]
    else:
      args_maker = lambda: [a_rng(a_shape, a_dtype), q_rng(q_shape, q_dtype)]

    def np_fun(*args):
      args = [x if jnp.result_type(x) != jnp.bfloat16 else
              np.asarray(x, np.float32) for x in args]
      if numpy_version <= (1, 22):
        return getattr(np, op)(*args, axis=axis, keepdims=keepdims,
                               interpolation=method)
      else:
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

  @unittest.skipIf(not config.jax_enable_x64, "test requires X64")
  @unittest.skipIf(jtu.device_under_test() != 'cpu', "test is for CPU float64 precision")
  def testPercentilePrecision(self):
    # Regression test for https://github.com/google/jax/issues/8513
    x = jnp.float64([1, 2, 3, 4, 7, 10])
    self.assertEqual(jnp.percentile(x, 50), 3.5)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name":
           "_{}_a_shape={}_axis={}_keepdims={}".format(
             op, jtu.format_shape_dtype_string(a_shape, a_dtype),
             axis, keepdims),
         "op": op, "a_shape": a_shape, "a_dtype": a_dtype,
         "axis": axis,
         "keepdims": keepdims}
        for a_dtype in default_dtypes
        for a_shape, axis in (
          ((7,), None),
          ((47, 7), 0),
          ((4, 101), 1),
        )
        for keepdims in [False, True]
        for op in ["median", "nanmedian"]))
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


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testWhereOneArgument(self, shape, dtype):
    rng = jtu.rand_some_zero(self.rng())
    np_fun = lambda x: np.where(x)
    np_fun = jtu.ignore_warning(
      category=DeprecationWarning,
      message="Calling nonzero on 0d arrays.*")(np_fun)
    jnp_fun = lambda x: jnp.where(x)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

    # JIT compilation requires specifying a size statically. Full test of
    # this behavior is in testNonzeroSize().
    jnp_fun = lambda x: jnp.where(x, size=np.size(x) // 2)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
      "testcase_name": "_{}".format("_".join(
        jtu.format_shape_dtype_string(shape, dtype)
        for shape, dtype in zip(shapes, dtypes))),
      "shapes": shapes, "dtypes": dtypes
    } for shapes in s(filter(_shapes_are_broadcast_compatible,
                         itertools.combinations_with_replacement(all_shapes, 3)))
      for dtypes in s(itertools.combinations_with_replacement(all_dtypes, 3)))))
  def testWhereThreeArgument(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    def np_fun(cond, x, y):
      return _promote_like_jnp(partial(np.where, cond))(x, y)
    self._CheckAgainstNumpy(np_fun, jnp.where, args_maker)
    self._CompileAndCheck(jnp.where, args_maker)

  def testWhereScalarPromotion(self):
    x = jnp.where(jnp.array([True, False]), 3,
                  jnp.ones((2,), dtype=jnp.float32))
    self.assertEqual(x.dtype, np.dtype(np.float32))

  @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
      "testcase_name": jtu.format_test_name_suffix("", shapes, (np.bool_,) * n + dtypes),
      "shapes": shapes, "dtypes": dtypes
    } for n in s(range(1, 3))
      for shapes in s(filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(all_shapes, 2 * n + 1)))
      for dtypes in s(itertools.combinations_with_replacement(all_dtypes, n + 1)))))
  def testSelect(self, shapes, dtypes):
    rng = jtu.rand_default(self.rng())
    n = len(dtypes) - 1
    def args_maker():
      condlist = [rng(shape, np.bool_) for shape in shapes[:n]]
      choicelist = [rng(shape, dtype)
                    for shape, dtype in zip(shapes[n:-1], dtypes[:n])]
      default = rng(shapes[-1], dtypes[-1])
      return condlist, choicelist, default
    # TODO(phawkins): float32/float64 type mismatches
    def np_fun(condlist, choicelist, default):
      choicelist = [x if jnp.result_type(x) != jnp.bfloat16
                    else x.astype(np.float32) for x in choicelist]
      dtype = jnp.result_type(default, *choicelist)
      return np.select(condlist,
                        [np.asarray(x, dtype=dtype) for x in choicelist],
                        np.asarray(default, dtype=dtype))
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
  # https://github.com/google/jax/issues/1052#issuecomment-514083352
  # def testIssue347(self):
  #   # https://github.com/google/jax/issues/347
  #   def test_fail(x):
  #     x = jnp.sqrt(jnp.sum(x ** 2, axis=1))
  #     ones = jnp.ones_like(x)
  #     x = jnp.where(x > 0.5, x, ones)
  #     return jnp.sum(x)
  #   x = jnp.array([[1, 2], [3, 4], [0, 0]], dtype=jnp.float64)
  #   result = jax.grad(test_fail)(x)
  #   assert not np.any(np.isnan(result))

  def testIssue453(self):
    # https://github.com/google/jax/issues/453
    a = np.arange(6) + 1
    ans = jnp.reshape(a, (3, 2), order='F')
    expected = np.reshape(a, (3, 2), order='F')
    self.assertAllClose(ans, expected)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_dtype={}".format(op, dtype.__name__),
       "dtype": dtype, "op": op}
      for dtype in [int, float, bool, complex]
      for op in ["atleast_1d", "atleast_2d", "atleast_3d"]))
  def testAtLeastNdLiterals(self, dtype, op):
    # Fixes: https://github.com/google/jax/issues/634
    np_fun = lambda arg: getattr(np, op)(arg).astype(dtypes.python_scalar_dtypes[dtype])
    jnp_fun = lambda arg: getattr(jnp, op)(arg)
    args_maker = lambda: [dtype(2)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
    {
      "testcase_name": "_shape={}_dtype={}_weights={}_minlength={}_length={}".format(
        shape, dtype, weights, minlength, length
      ),
      "shape": shape,
      "dtype": dtype,
      "weights": weights,
      "minlength": minlength,
      "length": length}
    for shape in [(0,), (5,), (10,)]
    for dtype in int_dtypes
    for weights in [True, False]
    for minlength in [0, 20]
    for length in [None, 8]
  ))
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


  @parameterized.named_parameters(*jtu.cases_from_list(
      {"testcase_name": "_case={}".format(i),
       "input": input}
       for i, input in enumerate([
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
       ])))
  def testBlock(self, input):
    args_maker = lambda: [input]
    self._CheckAgainstNumpy(np.block, jnp.block, args_maker)
    self._CompileAndCheck(jnp.block, args_maker)

  def testLongLong(self):
    self.assertAllClose(np.int64(7), jax.jit(lambda x: x)(np.longlong(7)))

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype.*")
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

  def testIssue830(self):
    a = jnp.arange(4, dtype=jnp.complex64)
    self.assertEqual(a.dtype, jnp.complex64)

  def testIssue728(self):
    assert jnp.allclose(jnp.eye(5000), np.eye(5000))
    self.assertEqual(0, np.sum(jnp.eye(1050) - np.eye(1050)))

  def testIssue746(self):
    jnp.arange(12).reshape(3, 4)  # doesn't crash

  def testIssue764(self):
    x = jnp.linspace(190, 200, 4)
    f = jax.grad(lambda x: jnp.sum(jnp.tanh(x)))
    # Expected values computed with autograd in float64 precision.
    expected = np.array([3.71669453e-165, 4.72999108e-168, 6.01954653e-171,
                          7.66067839e-174], np.float64)
    self.assertAllClose(f(x), expected, check_dtypes=False)

  def testIssue776(self):
    """Tests that the scatter-add transpose rule instantiates symbolic zeros."""
    def f(u):
      y = jnp.ones(10).at[np.array([2, 4, 5])].add(u)
      # The transpose rule for lax.tie_in returns a symbolic zero for its first
      # argument.
      return lax.tie_in(y, 7.)

    self.assertAllClose(np.zeros(3,), jax.grad(f)(np.ones(3,)))

  # NOTE(mattjj): I disabled this test when removing lax._safe_mul because this
  # is a numerical stability issue that should be solved with a custom jvp rule
  # of the sigmoid function being differentiated here, not by safe_mul.
  # def testIssue777(self):
  #   x = jnp.linspace(-200, 0, 4, dtype=np.float32)
  #   f = jax.grad(lambda x: jnp.sum(1 / (1 + jnp.exp(-x))))
  #   self.assertAllClose(f(x), np.array([0., 0., 0., 0.25], dtype=np.float32))

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(op, [()], [dtype]),
         "dtype": dtype, "op": op}
      for dtype in float_dtypes
      for op in ("sqrt", "arccos", "arcsin", "arctan", "sin", "cos", "tan",
                 "sinh", "cosh", "tanh", "arccosh", "arcsinh", "arctanh", "exp",
                 "log", "expm1", "log1p")))
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
      if (op in ("sin", "cos", "tan") and
          jtu.device_under_test() == "tpu"):
        continue  # TODO(b/132196789): fix and reenable.
      x = dtype(x)
      expected = np_op(x)
      actual = jnp_op(x)
      tol = jtu.tolerance(dtype, {np.float32: 1e-3, np.float64: 1e-7})
      self.assertAllClose(expected, actual, atol=tol,
                          rtol=tol)


  def testReductionOfOutOfBoundsAxis(self):  # Issue 888
    x = jnp.ones((3, 4))
    self.assertRaises(ValueError, lambda: jnp.sum(x, axis=2))

  def testIssue956(self):
    self.assertRaises(TypeError, lambda: jnp.ndarray((1, 1)))

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name":
         "_shape={}_dtype={}_out_dtype={}_axis={}_ddof={}_keepdims={}"
         .format(shape, dtype.__name__, out_dtype.__name__, axis, ddof, keepdims),
         "shape": shape, "dtype": dtype, "out_dtype": out_dtype, "axis": axis,
         "ddof": ddof, "keepdims": keepdims}
        for shape in [(5,), (10, 5)]
        for dtype in all_dtypes
        for out_dtype in inexact_dtypes
        for axis in [None, 0, -1]
        for ddof in [0, 1, 2]
        for keepdims in [False, True]))
  def testVar(self, shape, dtype, out_dtype, axis, ddof, keepdims):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.")
    def np_fun(x):
      out = np.var(x.astype(jnp.promote_types(np.float32, dtype)),
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name":
         "_shape={}_dtype={}_out_dtype={}_axis={}_ddof={}_keepdims={}"
         .format(shape, dtype, out_dtype, axis, ddof, keepdims),
         "shape": shape, "dtype": dtype, "out_dtype": out_dtype, "axis": axis,
         "ddof": ddof, "keepdims": keepdims}
        for shape in [(5,), (10, 5)]
        for dtype in all_dtypes
        for out_dtype in inexact_dtypes
        for axis in [None, 0, -1]
        for ddof in [0, 1, 2]
        for keepdims in [False, True]))
  def testNanVar(self, shape, dtype, out_dtype, axis, ddof, keepdims):
    rng = jtu.rand_some_nan(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    @jtu.ignore_warning(category=RuntimeWarning,
                        message="Degrees of freedom <= 0 for slice.")
    def np_fun(x):
      out = np.nanvar(x.astype(jnp.promote_types(np.float32, dtype)),
                    axis=axis, ddof=ddof, keepdims=keepdims)
      return out.astype(out_dtype)
    jnp_fun = partial(jnp.nanvar, dtype=out_dtype, axis=axis, ddof=ddof, keepdims=keepdims)
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

  def testNanStdGrad(self):
    # Regression test for https://github.com/google/jax/issues/8128
    x = jnp.arange(5.0).at[0].set(jnp.nan)
    y = jax.grad(jnp.nanvar)(x)
    self.assertAllClose(y, jnp.array([0.0, -0.75, -0.25, 0.25, 0.75]))

    z = jax.grad(jnp.nanstd)(x)
    self.assertEqual(jnp.isnan(z).sum(), 0)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name":
          "_shape={}_dtype={}_y_shape={}_y_dtype={}_rowvar={}_ddof={}_bias={}_fweights={}_aweights={}".format(
            shape, dtype, y_shape, y_dtype, rowvar, ddof, bias, fweights, aweights),
         "shape": shape, "y_shape": y_shape, "dtype": dtype, "y_dtype": y_dtype,"rowvar": rowvar, "ddof": ddof,
         "bias": bias, "fweights": fweights, "aweights": aweights}
        for shape in [(5,), (10, 5), (5, 10)]
        for dtype in all_dtypes
        for y_dtype in [None, dtype]
        for rowvar in [True, False]
        for y_shape in _get_y_shapes(y_dtype, shape, rowvar)
        for bias in [True, False]
        for ddof in [None, 2, 3]
        for fweights in [True, False]
        for aweights in [True, False]))
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
    tol = 7e-2 if jtu.device_under_test() == "tpu" else tol
    tol = jtu.join_tolerance(tol, jtu.tolerance(dtype))
    self._CheckAgainstNumpy(
        np_fun, jnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol,
                          rtol=tol)

  def testIssue967(self):
    self.assertRaises(TypeError, lambda: jnp.zeros(1.5))

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_shape={}_dtype={}_rowvar={}".format(
            shape, dtype.__name__, rowvar),
         "shape": shape, "dtype": dtype, "rowvar": rowvar}
        for shape in [(5,), (10, 5), (3, 10)]
        for dtype in number_dtypes
        for rowvar in [True, False]))
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
    tol = 1e-2 if jtu.device_under_test() == "tpu" else None
    self._CheckAgainstNumpy(
        np_fun, jnp_fun, args_maker, check_dtypes=False,
        tol=tol)
    self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(jtu.format_shape_dtype_string(shape, dtype),
       "None" if end_dtype is None else jtu.format_shape_dtype_string(end_shape, end_dtype),
       "None" if begin_dtype is None else jtu.format_shape_dtype_string(begin_shape, begin_dtype)),
       "shape": shape, "dtype": dtype, "end_shape": end_shape,
       "end_dtype": end_dtype, "begin_shape": begin_shape,
       "begin_dtype": begin_dtype}
      for dtype in number_dtypes
      for end_dtype in [None] + [dtype]
      for begin_dtype in [None] + [dtype]
      for shape in [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE]
      for begin_shape in (
        [None] if begin_dtype is None
        else [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE])
      for end_shape in (
        [None] if end_dtype is None
        else [s for s in all_shapes if s != jtu.PYTHON_SCALAR_SHAPE])))
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_shapes={}_dtype={}_indexing={}_sparse={}".format(
            shapes, dtype, indexing, sparse),
         "shapes": shapes, "dtype": dtype, "indexing": indexing,
         "sparse": sparse}
        for shapes in [(), (5,), (5, 3)]
        for dtype in number_dtypes
        for indexing in ['xy', 'ij']
        for sparse in [True, False]))
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
    with self.assertRaisesRegex(jax.core.ConcretizationTypeError,
                                "slice start of jnp.mgrid"):
      jax.jit(lambda a, b: jnp.mgrid[a:b])(0, 2)

  def testOgrid(self):
    # wrap indexer for appropriate dtype defaults.
    np_ogrid = _indexer_with_default_outputs(np.ogrid)
    def assertListOfArraysEqual(xs, ys):
      self.assertIsInstance(xs, list)
      self.assertIsInstance(ys, list)
      self.assertEqual(len(xs), len(ys))
      for x, y in zip(xs, ys):
        self.assertArraysEqual(x, y)

    self.assertArraysEqual(np_ogrid[:5], jnp.ogrid[:5])
    self.assertArraysEqual(np_ogrid[:5], jax.jit(lambda: jnp.ogrid[:5])())
    self.assertArraysEqual(np_ogrid[1:7:2], jnp.ogrid[1:7:2])
    # List of arrays
    assertListOfArraysEqual(np_ogrid[:5,], jnp.ogrid[:5,])
    assertListOfArraysEqual(np_ogrid[0:5, 1:3], jnp.ogrid[0:5, 1:3])
    assertListOfArraysEqual(np_ogrid[1:3:2, 2:9:3], jnp.ogrid[1:3:2, 2:9:3])
    assertListOfArraysEqual(np_ogrid[:5, :9, :11], jnp.ogrid[:5, :9, :11])
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
    with self.assertRaisesRegex(jax.core.ConcretizationTypeError,
                                "slice start of jnp.ogrid"):
      jax.jit(lambda a, b: jnp.ogrid[a:b])(0, 2)

  def testR_(self):
    a = np.arange(6).reshape((2,3))
    self.assertArraysEqual(np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])],
                           jnp.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])])
    self.assertArraysEqual(np.r_['-1', a, a], jnp.r_['-1', a, a])

    # wrap indexer for appropriate dtype defaults.
    np_r_ = _indexer_with_default_outputs(np.r_)
    self.assertArraysEqual(np_r_['0,2', [1,2,3], [4,5,6]], jnp.r_['0,2', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np_r_['0,2,0', [1,2,3], [4,5,6]], jnp.r_['0,2,0', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np_r_['1,2,0', [1,2,3], [4,5,6]], jnp.r_['1,2,0', [1,2,3], [4,5,6]])
    # negative 1d axis start
    self.assertArraysEqual(np_r_['0,4,-1', [1,2,3], [4,5,6]], jnp.r_['0,4,-1', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np_r_['0,4,-2', [1,2,3], [4,5,6]], jnp.r_['0,4,-2', [1,2,3], [4,5,6]])

    # matrix directives
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
      self.assertArraysEqual(np_r_['r',[1,2,3], [4,5,6]], jnp.r_['r',[1,2,3], [4,5,6]])
      self.assertArraysEqual(np_r_['c', [1, 2, 3], [4, 5, 6]], jnp.r_['c', [1, 2, 3], [4, 5, 6]])

    # bad directive
    with self.assertRaisesRegex(ValueError, "could not understand directive.*"):
      jnp.r_["asdfgh",[1,2,3]]
    # abstract tracer value for r_ slice
    with self.assertRaisesRegex(jax.core.ConcretizationTypeError,
                                "slice start of jnp.r_"):
      jax.jit(lambda a, b: jnp.r_[a:b])(0, 2)

    # Complex number steps
    atol = 1e-6
    rtol = 1e-6
    self.assertAllClose(np_r_[-1:1:6j],
                        jnp.r_[-1:1:6j],
                        atol=atol,
                        rtol=rtol)
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

    # wrap indexer for appropriate dtype defaults.
    np_c_ = _indexer_with_default_outputs(np.c_)
    self.assertArraysEqual(np_c_['0,2', [1,2,3], [4,5,6]], jnp.c_['0,2', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np_c_['0,2,0', [1,2,3], [4,5,6]], jnp.c_['0,2,0', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np_c_['1,2,0', [1,2,3], [4,5,6]], jnp.c_['1,2,0', [1,2,3], [4,5,6]])
    # negative 1d axis start
    self.assertArraysEqual(np_c_['0,4,-1', [1,2,3], [4,5,6]], jnp.c_['0,4,-1', [1,2,3], [4,5,6]])
    self.assertArraysEqual(np_c_['0,4,-2', [1,2,3], [4,5,6]], jnp.c_['0,4,-2', [1,2,3], [4,5,6]])
    # matrix directives, avoid numpy deprecation warning
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
      self.assertArraysEqual(np_c_['r',[1,2,3], [4,5,6]], jnp.c_['r',[1,2,3], [4,5,6]])
      self.assertArraysEqual(np_c_['c', [1, 2, 3], [4, 5, 6]], jnp.c_['c', [1, 2, 3], [4, 5, 6]])

    # bad directive
    with self.assertRaisesRegex(ValueError, "could not understand directive.*"):
      jnp.c_["asdfgh",[1,2,3]]
    # abstract tracer value for c_ slice
    with self.assertRaisesRegex(jax.core.ConcretizationTypeError,
                                "slice start of jnp.c_"):
      jax.jit(lambda a, b: jnp.c_[a:b])(0, 2)

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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": f"_start_shape={start_shape}_stop_shape={stop_shape}"
                          f"_num={num}_endpoint={endpoint}_retstep={retstep}"
                          f"_dtype={dtype.__name__ if dtype else 'None'}",
         "start_shape": start_shape, "stop_shape": stop_shape,
         "num": num, "endpoint": endpoint, "retstep": retstep,
         "dtype": dtype}
        for start_shape in [(), (2,), (2, 2)]
        for stop_shape in [(), (2,), (2, 2)]
        for num in [0, 1, 2, 5, 20]
        for endpoint in [True, False]
        for retstep in [True, False]
        # floating-point compute between jitted platforms and non-jit + rounding
        # cause unavoidable variation in integer truncation for some inputs, so
        # we currently only test inexact 'dtype' arguments.
        for dtype in inexact_dtypes + [None,]))
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
      # NumPy 1.20.0 changed the semantics of linspace to floor for integer
      # dtypes.
      if numpy_version >= (1, 20) or not np.issubdtype(dtype, np.integer):
        np_op = lambda start, stop: np.linspace(
          start, stop, num,
          endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)
      else:
        def np_op(start, stop):
          out = np.linspace(start, stop, num, endpoint=endpoint,
                            retstep=retstep, axis=axis)
          if retstep:
            return np.floor(out[0]).astype(dtype), out[1]
          else:
            return np.floor(out).astype(dtype)

      self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      self._CompileAndCheck(jnp_op, args_maker,
                            check_dtypes=False, atol=tol, rtol=tol)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": f"_dtype={dtype.__name__}", "dtype": dtype}
        for dtype in number_dtypes))
  def testLinspaceEndpoints(self, dtype):
    """Regression test for Issue #3014."""
    rng = jtu.rand_default(self.rng())
    endpoints = rng((2,), dtype)
    out = jnp.linspace(*endpoints, 10, dtype=dtype)
    self.assertAllClose(out[np.array([0, -1])], endpoints, rtol=0, atol=0)

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": ("_start_shape={}_stop_shape={}_num={}_endpoint={}"
                           "_base={}_dtype={}").format(
            start_shape, stop_shape, num, endpoint, base,
            dtype.__name__ if dtype else "None"),
         "start_shape": start_shape,
         "stop_shape": stop_shape,
         "num": num, "endpoint": endpoint, "base": base,
         "dtype": dtype}
        for start_shape in [(), (2,), (2, 2)]
        for stop_shape in [(), (2,), (2, 2)]
        for num in [0, 1, 2, 5, 20]
        for endpoint in [True, False]
        for base in [10.0, 2, np.e]
        for dtype in inexact_dtypes + [None,]))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLogspace(self, start_shape, stop_shape, num,
                   endpoint, base, dtype):
    if (dtype in int_dtypes and
        jtu.device_under_test() in ("gpu", "tpu") and
        not config.x64_enabled):
      raise unittest.SkipTest("GPUx32 truncated exponentiation"
                              " doesn't exactly match other platforms.")
    rng = jtu.rand_default(self.rng())
    # relax default tolerances slightly
    tol = {np.float16: 2e-2, np.float32: 1e-2, np.float64: 1e-6,
           np.complex64: 1e-3, np.complex128: 1e-6}
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": ("_start_shape={}_stop_shape={}_num={}_endpoint={}"
                           "_dtype={}_axis={}").format(
            start_shape, stop_shape, num, endpoint,
            dtype.__name__ if dtype else "None", axis),
         "start_shape": start_shape,
         "stop_shape": stop_shape,
         "num": num, "endpoint": endpoint,
         "dtype": dtype, "axis": axis}
        for start_shape in [(), (2,), (2, 2)]
        for stop_shape in [(), (2,), (2, 2)]
        for num in [0, 1, 2, 5, 20]
        for endpoint in [True, False]
        # NB: numpy's geomspace gives nonsense results on integer types
        for dtype in inexact_dtypes + [None,]
        for axis in range(-max(len(start_shape), len(stop_shape)),
                          max(len(start_shape), len(stop_shape)))))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testGeomspace(self, start_shape, stop_shape, num,
                    endpoint, dtype, axis):
    rng = jtu.rand_default(self.rng())
    # relax default tolerances slightly
    tol = {np.float16: 4e-3, np.float32: 2e-3, np.float64: 1e-14,
           np.complex128: 1e-14}
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
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker,
                            check_dtypes=False, tol=tol)
    if dtype in (inexact_dtypes + [None,]):
      self._CompileAndCheck(jnp_op, args_maker,
                            check_dtypes=False, atol=tol, rtol=tol)

  def testDisableNumpyRankPromotionBroadcasting(self):
    try:
      prev_flag = config._read('jax_numpy_rank_promotion')
      FLAGS.jax_numpy_rank_promotion = "allow"
      jnp.ones(2) + jnp.ones((1, 2))  # works just fine
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

    try:
      prev_flag = config._read('jax_numpy_rank_promotion')
      FLAGS.jax_numpy_rank_promotion = "raise"
      self.assertRaises(ValueError, lambda: jnp.ones(2) + jnp.ones((1, 2)))
      jnp.ones(2) + 3  # don't want to raise for scalars
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

    try:
      prev_flag = config._read('jax_numpy_rank_promotion')
      FLAGS.jax_numpy_rank_promotion = "warn"
      self.assertWarnsRegex(UserWarning, "Following NumPy automatic rank promotion for add on "
                            r"shapes \(2,\) \(1, 2\).*", lambda: jnp.ones(2) + jnp.ones((1, 2)))
      jnp.ones(2) + 3  # don't want to warn for scalars
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

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
    # tests https://github.com/google/jax/issues/1271
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

  @parameterized.named_parameters(
      {"testcase_name": "_from={}_to={}".format(from_shape, to_shape),
       "from_shape": from_shape, "to_shape": to_shape}
      for from_shape, to_shape in [
          [(1, 3), (4, 3)],
          [(3,), (2, 1, 3)],
          [(3,), (3, 3)],
          [(1,), (3,)],
          [(1,), 3],
      ])
  def testBroadcastTo(self, from_shape, to_shape):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [from_shape], [np.float32])
    np_op = lambda x: np.broadcast_to(x, to_shape)
    jnp_op = lambda x: jnp.broadcast_to(x, to_shape)
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
    self._CompileAndCheck(jnp_op, args_maker)

  @parameterized.named_parameters(
      {"testcase_name": f"_{shapes}", "shapes": shapes, "broadcasted_shape": broadcasted_shape}
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
    ])
  def testBroadcastShapes(self, shapes, broadcasted_shape):
    # Test against np.broadcast_shapes once numpy 1.20 is minimum required version
    np.testing.assert_equal(jnp.broadcast_shapes(*shapes), broadcasted_shape)

  def testBroadcastToIssue1522(self):
    self.assertRaisesRegex(
        ValueError, "Incompatible shapes for broadcasting: .*",
        lambda: jnp.broadcast_to(np.ones((2, 3)), (1, 3)))

  def testBroadcastToIntIssue1548(self):
    self.assertAllClose(jnp.broadcast_to(1, (3, 2)), np.ones((3, 2)),
                        check_dtypes=False)

  def testBroadcastToOnScalar(self):
    self.assertIsInstance(jnp.broadcast_to(10.0, ()), jnp.ndarray)
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

  @parameterized.named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_shape={}_varargs={} axis={}_dtype={}".format(
            shape, varargs, axis, dtype),
         "shape": shape, "varargs": varargs, "axis": axis, "dtype": dtype}
        for shape in [(10,), (10, 15), (10, 15, 20)]
        for _num_axes in range(len(shape))
        for varargs in itertools.combinations(range(1, len(shape) + 1), _num_axes)
        for axis in itertools.combinations(range(len(shape)), _num_axes)
        for dtype in inexact_dtypes))
  def testGradient(self, shape, varargs, axis, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    jnp_fun = lambda y: jnp.gradient(y, *varargs, axis=axis)
    np_fun = lambda y: np.gradient(y, *varargs, axis=axis)
    self._CheckAgainstNumpy(
        np_fun, jnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testZerosShapeErrors(self):
    # see https://github.com/google/jax/issues/1822
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
    x = self.rng().randn(3, 4).astype(jnp.float_)
    self.assertAllClose(x.trace(), jnp.array(x).trace())
    self.assertAllClose(x.trace(), jax.jit(lambda y: y.trace())(x))

  def testIntegerPowersArePrecise(self):
    # See https://github.com/google/jax/pull/3036
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

  def testToList(self):
    v = np.arange(12, dtype=np.int32).reshape(3, 4)
    self.assertEqual(jnp.asarray(v).tolist(), v.tolist())

  def testReductionWithRepeatedAxisError(self):
    with self.assertRaisesRegex(ValueError, r"duplicate value in 'axis': \(0, 0\)"):
      jnp.sum(jnp.arange(3), (0, 0))

  def testArangeConcretizationError(self):
    msg = r"It arose in jax.numpy.arange argument `{}`".format
    with self.assertRaisesRegex(jax.core.ConcretizationTypeError, msg('stop')):
      jax.jit(jnp.arange)(3)

    with self.assertRaisesRegex(jax.core.ConcretizationTypeError, msg('start')):
      jax.jit(lambda start: jnp.arange(start, 3))(0)

    with self.assertRaisesRegex(jax.core.ConcretizationTypeError, msg('stop')):
      jax.jit(lambda stop: jnp.arange(0, stop))(3)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": str(dtype), "dtype": dtype}
    for dtype in [None] + float_dtypes))
  def testArange64Bit(self, dtype):
    # Test that jnp.arange uses 64-bit arithmetic to define its range, even if the
    # output has another dtype. The issue here is that if python scalar inputs to
    # jnp.arange are cast to float32 before the range is computed, it changes the
    # number of elements output by the range.  It's unclear whether this was deliberate
    # behavior in the initial implementation, but it's behavior that downstream users
    # have come to rely on.
    args = (1.2, 4.8, 0.24)

    # Ensure that this test case leads to differing lengths if cast to float32.
    self.assertLen(np.arange(*args), 15)
    self.assertLen(np.arange(*map(np.float32, args)), 16)

    jnp_fun = lambda: jnp.arange(*args, dtype=dtype)
    np_fun = jtu.with_jax_dtype_defaults(lambda: np.arange(*args, dtype=dtype), dtype is None)
    args_maker = lambda: []
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testIssue2347(self):
    # https://github.com/google/jax/issues/2347
    object_list = List[Tuple[jnp.array, float, float, jnp.array, bool]]
    self.assertRaises(TypeError, jnp.array, object_list)

    np_object_list = np.array(object_list)
    self.assertRaises(TypeError, jnp.array, np_object_list)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": jtu.format_test_name_suffix("", shapes, dtypes),
     "shapes": shapes, "dtypes": dtypes}
    for shapes in filter(
      _shapes_are_broadcast_compatible,
      itertools.combinations_with_replacement(all_shapes, 2))
    for dtypes in itertools.product(
      *(_valid_dtypes_for_shape(s, complex_dtypes) for s in shapes))))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLogaddexpComplex(self, shapes, dtypes):
    @jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")
    def np_op(x1, x2):
      return np.log(np.exp(x1) + np.exp(x2))

    rng = jtu.rand_some_nan(self.rng())
    args_maker = lambda: tuple(rng(shape, dtype) for shape, dtype in zip(shapes, dtypes))
    if jtu.device_under_test() == 'tpu':
      tol = {np.complex64: 1e-3, np.complex128: 1e-10}
    else:
      tol = {np.complex64: 1e-5, np.complex128: 1e-14}
    self._CheckAgainstNumpy(_promote_like_jnp(np_op), jnp.logaddexp, args_maker, tol=tol)
    self._CompileAndCheck(jnp.logaddexp, args_maker, rtol=tol, atol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": jtu.format_test_name_suffix("", shapes, dtypes),
     "shapes": shapes, "dtypes": dtypes}
    for shapes in filter(
      _shapes_are_broadcast_compatible,
      itertools.combinations_with_replacement(all_shapes, 2))
    for dtypes in itertools.product(
      *(_valid_dtypes_for_shape(s, complex_dtypes) for s in shapes))))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testLogaddexp2Complex(self, shapes, dtypes):
    @jtu.ignore_warning(category=RuntimeWarning, message="invalid value.*")
    def np_op(x1, x2):
      return np.log2(np.exp2(x1) + np.exp2(x2))

    rng = jtu.rand_some_nan(self.rng())
    args_maker = lambda: tuple(rng(shape, dtype) for shape, dtype in zip(shapes, dtypes))
    if jtu.device_under_test() == 'tpu':
      tol = {np.complex64: 1e-3, np.complex128: 1e-10}
    else:
      tol = {np.complex64: 1e-5, np.complex128: 1e-14}
    self._CheckAgainstNumpy(_promote_like_jnp(np_op), jnp.logaddexp2, args_maker, tol=tol)
    self._CompileAndCheck(jnp.logaddexp2, args_maker, rtol=tol, atol=tol)

  def testDefaultDtypes(self):
    precision = config.jax_default_dtype_bits
    assert precision in ['32', '64']
    self.assertEqual(jnp.bool_, np.bool_)
    self.assertEqual(jnp.int_, np.int32 if precision == '32' else np.int64)
    self.assertEqual(jnp.uint, np.uint32 if precision == '32' else np.uint64)
    self.assertEqual(jnp.float_, np.float32 if precision == '32' else np.float64)
    self.assertEqual(jnp.complex_, np.complex64 if precision == '32' else np.complex128)

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
    self.assertArraysEqual(expected, actual)

  def testFromString(self):
    s = "1,2,3"
    expected = np.fromstring(s, sep=',', dtype=int)
    actual = jnp.fromstring(s, sep=',', dtype=int)
    self.assertArraysEqual(expected, actual)


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
    grad_test_spec(jnp.arccosh, nargs=1, order=2,
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

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.name, shapes, itertools.repeat(dtype)),
         "op": rec.op, "rng_factory": rec.rng_factory, "shapes": shapes, "dtype": dtype,
         "order": rec.order, "tol": rec.tol}
        for shapes in itertools.combinations_with_replacement(nonempty_shapes, rec.nargs)
        for dtype in rec.dtypes)
      for rec in GRAD_TEST_RECORDS))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testOpGrad(self, op, rng_factory, shapes, dtype, order, tol):
    rng = rng_factory(self.rng())
    tol = jtu.join_tolerance(tol, {np.float32: 1e-1, np.float64: 1e-3,
                                   np.complex64: 1e-1, np.complex128: 1e-3})
    args = tuple(rng(shape, dtype) for shape in shapes)
    check_grads(op, args, order, ["fwd", "rev"], tol, tol)

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
          {"testcase_name": "_{}_{}".format(rec.op.__name__, special_value),
           "op": rec.op, "special_value": special_value, "order": rec.order}
          for special_value in rec.values)
      for rec in GRAD_SPECIAL_VALUE_TEST_RECORDS))
  def testOpGradSpecialValue(self, op, special_value, order):
    check_grads(op, (special_value,), order, ["fwd", "rev"],
                atol={np.float32: 3e-3})

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
    # https://github.com/google/jax/issues/1521
    idx = jnp.repeat(jnp.arange(3), 10).reshape((30, 1))

    def f(x):
      y = x * jnp.arange(3.).reshape((1, 3))
      return jnp.take_along_axis(y, idx, -1).sum()

    check_grads(f, (1.,), order=1)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": jtu.format_test_name_suffix("", shapes, itertools.repeat(dtype)),
     "shapes": shapes, "dtype": dtype}
    for shapes in filter(
      _shapes_are_broadcast_compatible,
      itertools.combinations_with_replacement(nonempty_shapes, 2))
    for dtype in (np.complex128, )))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testGradLogaddexpComplex(self, shapes, dtype):
    rng = jtu.rand_default(self.rng())
    args = tuple(rng(shape, dtype) for shape in shapes)
    if jtu.device_under_test() == "tpu":
      tol = 5e-2
    else:
      tol = 3e-2
    check_grads(jnp.logaddexp, args, 1, ["fwd", "rev"], tol, tol)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": jtu.format_test_name_suffix("", shapes, itertools.repeat(dtype)),
     "shapes": shapes, "dtype": dtype}
    for shapes in filter(
      _shapes_are_broadcast_compatible,
      itertools.combinations_with_replacement(nonempty_shapes, 2))
    for dtype in (np.complex128, )))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testGradLogaddexp2Complex(self, shapes, dtype):
    rng = jtu.rand_default(self.rng())
    args = tuple(rng(shape, dtype) for shape in shapes)
    if jtu.device_under_test() == "tpu":
      tol = 5e-2
    else:
      tol = 3e-2
    check_grads(jnp.logaddexp2, args, 1, ["fwd", "rev"], tol, tol)


class NumpySignaturesTest(jtu.JaxTestCase):

  def testWrappedSignaturesMatch(self):
    """Test that jax.numpy function signatures match numpy."""
    jnp_funcs = {name: getattr(jnp, name) for name in dir(jnp)}
    func_pairs = {name: (fun, fun.__np_wrapped__) for name, fun in jnp_funcs.items()
                  if hasattr(fun, '__np_wrapped__')}
    assert len(func_pairs) > 0

    # TODO(jakevdp): fix some of the following signatures. Some are due to wrong argument names.
    unsupported_params = {
      'asarray': ['like'],
      'broadcast_to': ['subok', 'array'],
      'clip': ['kwargs'],
      'copy': ['subok'],
      'corrcoef': ['ddof', 'bias', 'dtype'],
      'cov': ['dtype'],
      'empty_like': ['subok', 'order'],
      'einsum': ['kwargs'],
      'einsum_path': ['einsum_call'],
      'eye': ['order', 'like'],
      'identity': ['like'],
      'full': ['order', 'like'],
      'full_like': ['subok', 'order'],
      'fromfunction': ['like'],
      'histogram': ['normed'],
      'histogram2d': ['normed'],
      'histogramdd': ['normed'],
      'ones': ['order', 'like'],
      'ones_like': ['subok', 'order'],
      'tri': ['like'],
      'unwrap': ['period'],
      'zeros_like': ['subok', 'order']
    }

    extra_params = {
      'broadcast_to': ['arr'],
      'einsum': ['precision'],
      'einsum_path': ['subscripts'],
    }

    mismatches = {}

    for name, (jnp_fun, np_fun) in func_pairs.items():
      # broadcast_shapes is not available in numpy < 1.20
      if numpy_version < (1, 20) and name == "broadcast_shapes":
        continue
      # Some signatures have changed; skip for older numpy versions.
      if numpy_version < (1, 19) and name in ['einsum_path', 'gradient', 'isscalar']:
        continue
      if numpy_version < (1, 22) and name in ['quantile', 'nanquantile',
                                              'percentile', 'nanpercentile']:
        continue
      # Note: can't use inspect.getfullargspec due to numpy issue
      # https://github.com/numpy/numpy/issues/12225
      try:
        np_params = inspect.signature(np_fun).parameters
      except ValueError:
        # Some functions cannot be inspected
        continue
      jnp_params = inspect.signature(jnp_fun).parameters
      extra = set(extra_params.get(name, []))
      unsupported = set(unsupported_params.get(name, []))

      # Checks to prevent tests from becoming out-of-date. If these fail,
      # it means that extra_params or unsupported_params need to be updated.
      assert extra.issubset(jnp_params), f"{name}: extra={extra} is not a subset of jnp_params={set(jnp_params)}."
      assert not unsupported.intersection(jnp_params), f"{name}: unsupported={unsupported} overlaps with jnp_params={set(jnp_params)}."

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


_available_numpy_dtypes: List[str] = [dtype.__name__ for dtype in jtu.dtypes.all
                                      if dtype != dtypes.bfloat16]


def _all_numpy_ufuncs() -> Iterator[str]:
  """Generate the names of all ufuncs in the top-level numpy namespace."""
  for name in dir(np):
    f = getattr(np, name)
    if isinstance(f, np.ufunc):
      yield name


def _dtypes_for_ufunc(name: str) -> Iterator[Tuple[str, ...]]:
  """Generate valid dtypes of inputs to the given numpy ufunc."""
  func = getattr(np, name)
  for arg_dtypes in itertools.product(_available_numpy_dtypes, repeat=func.nin):
    args = (np.ones(1, dtype=dtype) for dtype in arg_dtypes)
    try:
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
        _ = func(*args)
    except TypeError:
      pass
    else:
      yield arg_dtypes


class NumpyUfuncTests(jtu.JaxTestCase):

  @parameterized.named_parameters(
    {"testcase_name": f"_{name}_{','.join(arg_dtypes)}",
     "name": name, "arg_dtypes": arg_dtypes}
    for name in _all_numpy_ufuncs()
    for arg_dtypes in jtu.cases_from_list(_dtypes_for_ufunc(name)))
  def testUfuncInputTypes(self, name, arg_dtypes):
    # TODO(jakevdp): fix following failures and remove from this exception list.
    if (name in ['divmod', 'floor_divide', 'fmod', 'gcd', 'left_shift', 'mod',
                 'power', 'remainder', 'right_shift', 'rint', 'square']
        and 'bool_' in arg_dtypes):
      self.skipTest(f"jax.numpy does not support {name}{tuple(arg_dtypes)}")
    if name == 'arctanh' and jnp.issubdtype(arg_dtypes[0], jnp.complexfloating):
      self.skipTest("np.arctanh & jnp.arctanh have mismatched NaNs for complex input.")

    jnp_op = getattr(jnp, name)
    np_op = getattr(np, name)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="divide by zero.*")(np_op)
    args_maker = lambda: tuple(np.ones(1, dtype=dtype) for dtype in arg_dtypes)

    try:
      jnp_op(*args_maker())
    except NotImplementedError:
      self.skipTest(f"jtu.{name} is not yet implemented.")

    # large tol comes from the fact that numpy returns float16 in places
    # that jnp returns float32. e.g. np.cos(np.uint8(0))
    self._CheckAgainstNumpy(np_op, jnp_op, args_maker, check_dtypes=False, tol=1E-2)


class NumpyDocTests(jtu.JaxTestCase):

  def test_lax_numpy_docstrings(self):
    # Test that docstring wrapping & transformation didn't fail.

    # Functions that have their own docstrings & don't wrap numpy.
    known_exceptions = {'broadcast_arrays', 'fromfile', 'fromiter', 'vectorize'}

    for name in dir(jnp):
      if name in known_exceptions or name.startswith('_'):
        continue

      # We only check signatures of functions.
      obj = getattr(jnp, name)
      if isinstance(obj, type) or not callable(obj):
        continue

      # Some jnp functions are imported from numpy or jax.dtypes directly.
      if any(obj is getattr(mod, obj.__name__, None) for mod in [np, dtypes]):
        continue

      wrapped_fun = obj.__np_wrapped__

      # If the wrapped function has a docstring, obj should too
      if wrapped_fun.__doc__ and not obj.__doc__:
        raise Exception(f"jnp.{name} does not contain wrapped docstring.")

      if obj.__doc__ and "*Original docstring below.*" not in obj.__doc__:
        raise Exception(f"jnp.{name} does not have a wrapped docstring.")

  @parameterized.named_parameters(
    {"testcase_name": "_jit" if jit else "", "jit": jit} for jit in [True, False])
  def test_wrapped_function_parameters(self, jit):
    def orig(x):
      """Example Docstring

      Parameters
      ----------
      x : array_like
        Input Data

        .. versionadded:: 1.8.0
      out : array_like, optional
        Output to overwrite
      other_arg : Any
        not used

      Returns
      -------
      x : input
      """
      return x

    def wrapped(x, out=None):
      return x

    if jit:
      wrapped = jax.jit(wrapped)

    wrapped = _wraps(orig, skip_params=['out'])(wrapped)
    doc = wrapped.__doc__

    self.assertStartsWith(doc, "Example Docstring")
    self.assertIn("Original docstring below", doc)
    self.assertIn("Parameters", doc)
    self.assertIn("Returns", doc)
    self.assertNotIn('out', doc)
    self.assertNotIn('other_arg', doc)
    self.assertNotIn('versionadded', doc)


  def test_parse_numpydoc(self):
    # Unit test ensuring that _parse_numpydoc correctly parses docstrings for all
    # functions in NumPy's top-level namespace.
    section_titles = {'Attributes', 'Examples', 'Notes',
                      'Parameters', 'Raises', 'References',
                      'Returns', 'See also', 'See Also', 'Warnings', 'Warns'}
    headings = [title + '\n' + '-'*len(title) for title in section_titles]

    for name in dir(np):
      if name.startswith('_'):
        continue
      obj = getattr(np, name)
      if isinstance(obj, type):
        continue
      if not callable(obj):
        continue
      if 'built-in function' in repr(obj):
        continue
      parsed = _parse_numpydoc(obj.__doc__)

      # Check that no docstring is handled gracefully.
      if not obj.__doc__:
        self.assertEqual(parsed, ParsedDoc(obj.__doc__))
        continue

      # Check that no unexpected section names are found.
      extra_keys = parsed.sections.keys() - section_titles
      if extra_keys:
        raise ValueError(f"Extra section headers found in np.{name}: {extra_keys}")

      # Check that every docstring has a summary.
      if not parsed.summary:
        raise ValueError(f"No summary found for np.{name}")

      # Check that no expected headings are missed.
      for heading in headings:
        assert heading not in parsed.front_matter


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
