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
import functools
from functools import partial
import itertools
import operator
from typing import NamedTuple
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
import jax.ops
from jax import lax
from jax import numpy as jnp

from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu

config.parse_flags_with_absl()

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
real_dtypes = float_dtypes + int_dtypes + unsigned_dtypes
all_dtypes = number_dtypes + bool_dtypes


python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

# uint64 is problematic because with any uint type it promotes to float:
int_dtypes_no_uint64 = [d for d in int_dtypes + unsigned_dtypes if d != np.uint64]


def _valid_dtypes_for_shape(shape, dtypes):
  # Not all (shape, dtype) pairs are valid. In particular, Python scalars only
  # have one type in each category (float, bool, etc.)
  if shape is jtu.PYTHON_SCALAR_SHAPE:
    return [t for t in dtypes if t in python_scalar_dtypes]
  return dtypes


OpRecord = collections.namedtuple(
  "OpRecord",
  ["name", "nargs", "dtypes", "shapes", "rng_factory", "diff_modes",
   "test_name", "check_dtypes", "tolerance", "inexact", "kwargs", "alias"])

def op_record(name, nargs, dtypes, shapes, rng_factory, diff_modes,
              test_name=None, check_dtypes=True,
              tolerance=None, inexact=False, kwargs=None,
              alias=None):
  test_name = test_name or name
  return OpRecord(name, nargs, dtypes, shapes, rng_factory, diff_modes,
                  test_name, check_dtypes, tolerance, inexact, kwargs, alias)

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
              check_dtypes=False, tolerance={np.float16: 3e-3}),
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
    op_record("tan", 1, inexact_dtypes + int_dtypes, all_shapes,
              partial(jtu.rand_uniform, low=-1.5, high=1.5), ["rev"],
              inexact=True),
    op_record("tan", 1, unsigned_dtypes, all_shapes,
              partial(jtu.rand_uniform, low=0, high=1.5), ["rev"],
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
              inexact=True, tolerance={np.complex128: 2e-15}),
    op_record("arccos", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arctan", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arctan2", 2, float_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, check_dtypes=False),
    op_record("arcsinh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True, tolerance={np.complex64: 2E-4, np.complex128: 2E-14}),
    op_record("arccosh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True, tolerance={np.complex64: 2E-2, np.complex128: 2E-12}),
    op_record("arctanh", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, tolerance={np.float64: 1e-9}),
    op_record("asin", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, tolerance={np.complex128: 2e-15}, alias="arcsin"),
    op_record("acos", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, alias="arccos"),
    op_record("atan", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, alias="arctan"),
    op_record("atan2", 2, float_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, check_dtypes=False, alias="arctan2"),
    op_record("asinh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True, tolerance={np.complex64: 2E-4, np.complex128: 2E-14}, alias="arcsinh"),
    op_record("acosh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True, tolerance={np.complex64: 2E-2, np.complex128: 2E-12}, alias="arccosh"),
    op_record("atanh", 1, number_dtypes, all_shapes, jtu.rand_small, ["rev"],
              inexact=True, tolerance={np.float64: 1e-9}, alias="arctanh"),
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
    op_record("hypot", 2, real_dtypes, all_shapes, jtu.rand_default, [],
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
    op_record("polyval", 2,
              [d for d in number_dtypes if d not in (np.int8, np.uint8)],
              nonempty_nonscalar_array_shapes,
              jtu.rand_default, [], check_dtypes=False,
              tolerance={dtypes.bfloat16: 4e-2, np.float16: 2e-2,
                         np.float64: 1e-12}),
    op_record("positive", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("pow", 2, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              tolerance={np.complex128: 1e-14}, check_dtypes=False, alias="power"),
    op_record("power", 2, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              tolerance={np.complex128: 1e-14}, check_dtypes=False),
    op_record("rad2deg", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("ravel", 1, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("real", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("remainder", 2, default_dtypes, all_shapes, jtu.rand_some_zero, [],
              tolerance={np.float16: 1e-2}),
    op_record("mod", 2, default_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("modf", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("modf", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_default, [], check_dtypes=False),
    op_record("rint", 1, inexact_dtypes, all_shapes, jtu.rand_some_inf_and_nan,
              []),
    op_record("rint", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_default, [], check_dtypes=False),
    # numpy < 2.0.0 has a different convention for complex sign.
    op_record("sign", 1, real_dtypes if jtu.numpy_version() < (2, 0, 0) else number_dtypes,
              all_shapes, jtu.rand_some_inf_and_nan, []),
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
    op_record("lcm", 2, [np.int8], all_shapes, jtu.rand_not_small, [])
]

JAX_BITWISE_OP_RECORDS = [
    op_record("bitwise_and", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_fullrange, []),
    op_record("bitwise_invert", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_fullrange, [], alias='bitwise_not'),
    op_record("bitwise_not", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_fullrange, []),
    op_record("invert", 1, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_fullrange, []),
    op_record("bitwise_or", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_fullrange, []),
    op_record("bitwise_xor", 2, int_dtypes + unsigned_dtypes, all_shapes,
              jtu.rand_fullrange, []),
]
if hasattr(np, "bitwise_count"):
  # Numpy versions after 1.26
  JAX_BITWISE_OP_RECORDS.append(
    op_record("bitwise_count", 1, int_dtypes, all_shapes, jtu.rand_fullrange, []))

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

class _OverrideEverything:
  pass

for rec in JAX_OPERATOR_OVERLOADS + JAX_RIGHT_OPERATOR_OVERLOADS:
  if rec.nargs == 2:
    setattr(_OverrideEverything, rec.name, lambda self, other: self)

class _OverrideNothing:
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


def _get_testcase_name(index, params):
  dtypes = "_".join(str(dt.__name__) for dt in  params['dtypes'])
  name = params['op_name'] if "op_name" in params else params["name"]
  return f"{index}_{name}_{dtypes}"


def _create_named_parameters(iter_params):
  for i, params in enumerate(iter_params):
    yield dict(params, **{'testcase_name': _get_testcase_name(i, params)})


class JaxNumpyOperatorTests(jtu.JaxTestCase):
  """Tests for LAX-backed Numpy operators."""

  def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
    def f():
      out = [rng(shape, dtype or jnp.float_)
             for shape, dtype in zip(shapes, dtypes)]
      if np_arrays:
        return out
      return [jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
              for a in out]
    return f

  @parameterized.named_parameters(_create_named_parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(op_name=rec.name, rng_factory=rec.rng_factory,
            check_dtypes=rec.check_dtypes, tolerance=rec.tolerance,
            inexact=rec.inexact, kwargs=rec.kwargs or {}, alias=rec.alias)],
      [dict(shapes=shapes, dtypes=dtypes)
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes))],
    )
    for rec in itertools.chain(JAX_ONE_TO_ONE_OP_RECORDS,
                              JAX_COMPOUND_OP_RECORDS))))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testOp(self, op_name, rng_factory, shapes, dtypes, check_dtypes,
             tolerance, inexact, kwargs, alias):
    np_op = partial(getattr(np, op_name) if hasattr(np, op_name) else getattr(np, alias), **kwargs)
    jnp_op = partial(getattr(jnp, op_name), **kwargs)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="invalid value.*")(np_op)
    np_op = jtu.ignore_warning(category=RuntimeWarning,
                               message="divide by zero.*")(np_op)

    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in dtypes)
    if jtu.test_device_matches(["tpu"]) and op_name in (
        "arccosh", "arcsinh", "sinh", "cosh", "tanh", "sin", "cos", "tan",
        "log", "log1p", "log2", "log10", "exp", "expm1", "exp2", "pow",
        "power", "logaddexp", "logaddexp2", "i0", "acosh", "asinh"):
      tol = jtu.join_tolerance(tol, 1e-4)
    tol = functools.reduce(jtu.join_tolerance,
                           [tolerance, tol, jtu.default_tolerance()])

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(jtu.promote_like_jnp(np_op, inexact), jnp_op,
                              args_maker, check_dtypes=check_dtypes, tol=tol)
      self._CompileAndCheck(jnp_op, args_maker, check_dtypes=check_dtypes,
                            atol=tol, rtol=tol)

  @parameterized.named_parameters(_create_named_parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, tol=rec.tolerance)],
      [dict(shapes=shapes, dtypes=dtypes)
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes))],
    )
    for rec in JAX_OPERATOR_OVERLOADS)))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testOperatorOverload(self, name, rng_factory, shapes, dtypes, tol):
    rng = rng_factory(self.rng())
    # np and jnp arrays have different type promotion rules; force the use of
    # jnp arrays.
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    fun = lambda *xs: getattr(operator, name.strip('_'))(*xs)
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CompileAndCheck(fun, args_maker, atol=tol, rtol=tol)

  @parameterized.named_parameters(_create_named_parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory,
            op_tolerance=rec.tolerance)],
      [dict(shapes=shapes, dtypes=dtypes)
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          itertools.combinations_with_replacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes))],
    )
    for rec in JAX_RIGHT_OPERATOR_OVERLOADS)))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testRightOperatorOverload(self, name, rng_factory, shapes, dtypes,
                                op_tolerance):
    if shapes[1] is jtu.PYTHON_SCALAR_SHAPE:
      raise SkipTest("scalars not implemented")  # TODO(mattjj): clean up
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, np_arrays=False)
    fun = lambda fst, snd: getattr(snd, name)(fst)
    tol = max(jtu.tolerance(dtype, op_tolerance) for dtype in dtypes)
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CompileAndCheck( fun, args_maker, atol=tol, rtol=tol)

  @jtu.sample_product(
    name=[rec.name for rec in JAX_OPERATOR_OVERLOADS if rec.nargs == 2],
    othertype=[dict, list, tuple, set],
  )
  def testOperatorOverloadErrors(self, name, othertype):
    # Test that binary operators with builtin collections raise a TypeError
    # and report the types in the correct order.
    data = [(1, 2), (2, 3)]
    arr = jnp.array(data)
    other = othertype(data)

    msg = f"unsupported operand type.* 'ArrayImpl' and '{othertype.__name__}'"
    with self.assertRaisesRegex(TypeError, msg):
      getattr(arr, name)(other)

  @jtu.sample_product(
    name=[rec.name for rec in JAX_RIGHT_OPERATOR_OVERLOADS if rec.nargs == 2],
    othertype=[dict, list, tuple, set],
  )
  def testRightOperatorOverloadErrors(self, name, othertype):
    # Test that binary operators with builtin collections raise a TypeError
    # and report the types in the correct order.
    data = [(1, 2), (2, 3)]
    arr = jnp.array(data)
    other = othertype(data)

    msg = f"unsupported operand type.* '{othertype.__name__}' and 'ArrayImpl'"
    with self.assertRaisesRegex(TypeError, msg):
      getattr(arr, name)(other)

  @jtu.sample_product(
    [dict(op_name=rec.name, rng_factory=rec.rng_factory, dtype=dtype)
     for rec in JAX_OPERATOR_OVERLOADS if rec.nargs == 2
     for dtype in rec.dtypes],
  )
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

  @parameterized.named_parameters(_create_named_parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(name=rec.name, rng_factory=rec.rng_factory, alias=rec.alias)],
      shapes=filter(
        _shapes_are_broadcast_compatible,
        itertools.combinations_with_replacement(rec.shapes, rec.nargs)),
      dtypes=filter(
        _dtypes_are_compatible_for_bitwise_ops,
        itertools.combinations_with_replacement(rec.dtypes, rec.nargs)),
    )
    for rec in JAX_BITWISE_OP_RECORDS)))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testBitwiseOp(self, name, rng_factory, shapes, dtypes, alias):
    np_op = getattr(np, name) if hasattr(np, name) else getattr(np, alias)
    jnp_op = getattr(jnp, name)
    rng = rng_factory(self.rng())
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CheckAgainstNumpy(jtu.promote_like_jnp(np_op), jnp_op, args_maker)
      self._CompileAndCheck(jnp_op, args_maker)

  @jtu.sample_product(
    shape=array_shapes,
    dtype=int_dtypes + unsigned_dtypes,
  )
  def testBitwiseCount(self, shape, dtype):
    # np.bitwise_count added after numpy 1.26, but
    # np_scalar.bit_count() is available before that.
    np_fun = getattr(
      np, "bitwise_count",
      np.vectorize(lambda x: np.ravel(x)[0].bit_count(), otypes=['uint8']))
    rng = jtu.rand_fullrange(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp.bitwise_count, args_maker)
    self._CompileAndCheck(jnp.bitwise_count, args_maker)

  @jtu.sample_product(
    [dict(dtypes=dtypes, shapes=shapes)
      for shapes in filter(
        _shapes_are_broadcast_compatible,
        # TODO numpy always promotes to shift dtype for zero-dim shapes:
        itertools.combinations_with_replacement(nonzerodim_shapes, 2))
      for dtypes in itertools.product(
        *(_valid_dtypes_for_shape(s, int_dtypes_no_uint64) for s in shapes))
    ],
    op=[jnp.left_shift, jnp.bitwise_left_shift, jnp.right_shift, jnp.bitwise_right_shift],
  )
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def testShiftOpAgainstNumpy(self, op, dtypes, shapes):
    dtype, shift_dtype = dtypes
    signed_mix = np.issubdtype(dtype, np.signedinteger) != \
                 np.issubdtype(shift_dtype, np.signedinteger)
    has_32 = any(np.iinfo(d).bits == 32 for d in dtypes)
    promoting_to_64 = has_32 and signed_mix
    if promoting_to_64 and not config.enable_x64.value:
      self.skipTest("np.right_shift/left_shift promoting to int64"
                    "differs from jnp in 32 bit mode.")

    info, shift_info = map(np.iinfo, dtypes)
    x_rng = jtu.rand_int(self.rng(), low=info.min, high=info.max + 1)
    # NumPy requires shifts to be non-negative and below the bit width:
    shift_rng = jtu.rand_int(self.rng(), high=max(info.bits, shift_info.bits))
    args_maker = lambda: (x_rng(shapes[0], dtype), shift_rng(shapes[1], shift_dtype))

    np_op = getattr(np, op.__name__)

    with jtu.strict_promotion_if_dtypes_match(dtypes):
      self._CompileAndCheck(op, args_maker)
      self._CheckAgainstNumpy(np_op, op, args_maker)

  # This test can be deleted once we test against NumPy 2.0.
  @jtu.sample_product(
      shape=all_shapes,
      dtype=complex_dtypes
  )
  def testSignComplex(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    if jtu.numpy_version() >= (2, 0, 0):
      np_fun = np.sign
    else:
      np_fun = lambda x: (x / np.where(x == 0, 1, abs(x))).astype(np.result_type(x))
    jnp_fun = jnp.sign
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
    self._CompileAndCheck(jnp_fun, args_maker)

  def testDeferToNamedTuple(self):
    class MyArray(NamedTuple):
      arr: jax.Array
      def __mul__(self, other):
        return MyArray(self.arr * other)
      def __rmul__(self, other):
        return MyArray(other * self.arr)
    a = MyArray(jnp.ones(4))
    b = jnp.ones(4)
    self.assertIsInstance(a * b, MyArray)
    self.assertIsInstance(jax.jit(operator.mul)(a, b), MyArray)
    self.assertIsInstance(b * a, MyArray)
    self.assertIsInstance(jax.jit(operator.mul)(b, a), MyArray)

  def testI0Grad(self):
    # Regression test for https://github.com/google/jax/issues/11479
    dx = jax.grad(jax.numpy.i0)(0.0)
    self.assertArraysEqual(dx, 0.0)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
