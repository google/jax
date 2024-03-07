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

"""
Implements ufuncs for jax.numpy.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import operator

import warnings

import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src.api import jit
from jax._src.custom_derivatives import custom_jvp
from jax._src.lax import lax
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import (
   check_arraylike, promote_args, promote_args_inexact,
   promote_args_numeric, promote_dtypes_inexact, promote_dtypes_numeric,
   promote_shapes, _where, implements, check_no_float0s)

_lax_const = lax._const

_INT_DTYPES = {
  16: np.int16,
  32: np.int32,
  64: np.int64,
}

def _constant_like(x, const):
  return np.array(const, dtype=dtypes.dtype(x))

def _replace_inf(x: ArrayLike) -> Array:
  return lax.select(isposinf(real(x)), lax._zeros(x), x)

def _to_bool(x: Array) -> Array:
  return x if x.dtype == bool else lax.ne(x, _lax_const(x, 0))

@implements(np.fabs, module='numpy')
@partial(jit, inline=True)
def fabs(x: ArrayLike, /) -> Array:
  return lax.abs(*promote_args_inexact('fabs', x))

@implements(getattr(np, 'bitwise_invert', np.invert), module='numpy')
@partial(jit, inline=True)
def bitwise_invert(x: ArrayLike, /) -> Array:
  return lax.bitwise_not(*promote_args('bitwise_invert', x))

@implements(np.bitwise_not, module='numpy')
@partial(jit, inline=True)
def bitwise_not(x: ArrayLike, /) -> Array:
  return lax.bitwise_not(*promote_args('bitwise_not', x))

@implements(np.invert, module='numpy')
@partial(jit, inline=True)
def invert(x: ArrayLike, /) -> Array:
  return lax.bitwise_not(*promote_args('invert', x))

@implements(np.negative, module='numpy')
@partial(jit, inline=True)
def negative(x: ArrayLike, /) -> Array:
  return lax.neg(*promote_args('negative', x))

@implements(np.positive, module='numpy')
@partial(jit, inline=True)
def positive(x: ArrayLike, /) -> Array:
  return lax.asarray(*promote_args('positive', x))

@implements(np.sign, module='numpy')
@partial(jit, inline=True)
def sign(x: ArrayLike, /) -> Array:
  return lax.sign(*promote_args('sign', x))

@implements(np.floor, module='numpy')
@partial(jit, inline=True)
def floor(x: ArrayLike, /) -> Array:
  check_arraylike('floor', x)
  if dtypes.isdtype(dtypes.dtype(x), ('integral', 'bool')):
    return lax.asarray(x)
  return lax.floor(*promote_args_inexact('floor', x))

@implements(np.ceil, module='numpy')
@partial(jit, inline=True)
def ceil(x: ArrayLike, /) -> Array:
  check_arraylike('ceil', x)
  if dtypes.isdtype(dtypes.dtype(x), ('integral', 'bool')):
    return lax.asarray(x)
  return lax.ceil(*promote_args_inexact('ceil', x))

@implements(np.exp, module='numpy')
@partial(jit, inline=True)
def exp(x: ArrayLike, /) -> Array:
  return lax.exp(*promote_args_inexact('exp', x))

@implements(np.log, module='numpy')
@partial(jit, inline=True)
def log(x: ArrayLike, /) -> Array:
  return lax.log(*promote_args_inexact('log', x))

@implements(np.expm1, module='numpy')
@partial(jit, inline=True)
def expm1(x: ArrayLike, /) -> Array:
  return lax.expm1(*promote_args_inexact('expm1', x))

@implements(np.log1p, module='numpy')
@partial(jit, inline=True)
def log1p(x: ArrayLike, /) -> Array:
  return lax.log1p(*promote_args_inexact('log1p', x))

@implements(np.sin, module='numpy')
@partial(jit, inline=True)
def sin(x: ArrayLike, /) -> Array:
  return lax.sin(*promote_args_inexact('sin', x))

@implements(np.cos, module='numpy')
@partial(jit, inline=True)
def cos(x: ArrayLike, /) -> Array:
  return lax.cos(*promote_args_inexact('cos', x))

@implements(np.tan, module='numpy')
@partial(jit, inline=True)
def tan(x: ArrayLike, /) -> Array:
  return lax.tan(*promote_args_inexact('tan', x))

@implements(np.arcsin, module='numpy')
@partial(jit, inline=True)
def arcsin(x: ArrayLike, /) -> Array:
  return lax.asin(*promote_args_inexact('arcsin', x))

@implements(np.arccos, module='numpy')
@partial(jit, inline=True)
def arccos(x: ArrayLike, /) -> Array:
  return lax.acos(*promote_args_inexact('arccos', x))

@implements(np.arctan, module='numpy')
@partial(jit, inline=True)
def arctan(x: ArrayLike, /) -> Array:
  return lax.atan(*promote_args_inexact('arctan', x))

@implements(np.sinh, module='numpy')
@partial(jit, inline=True)
def sinh(x: ArrayLike, /) -> Array:
  return lax.sinh(*promote_args_inexact('sinh', x))

@implements(np.cosh, module='numpy')
@partial(jit, inline=True)
def cosh(x: ArrayLike, /) -> Array:
  return lax.cosh(*promote_args_inexact('cosh', x))

@implements(np.arcsinh, module='numpy')
@partial(jit, inline=True)
def arcsinh(x: ArrayLike, /) -> Array:
  return lax.asinh(*promote_args_inexact('arcsinh', x))

@implements(np.arccosh, module='numpy')
@jit
def arccosh(x: ArrayLike, /) -> Array:
  # Note: arccosh is multi-valued for complex input, and lax.acosh
  # uses a different convention than np.arccosh.
  result = lax.acosh(*promote_args_inexact("arccosh", x))
  if dtypes.issubdtype(result.dtype, np.complexfloating):
    result = _where(real(result) < 0, lax.neg(result), result)
  return result

@implements(np.tanh, module='numpy')
@partial(jit, inline=True)
def tanh(x: ArrayLike, /) -> Array:
  return lax.tanh(*promote_args_inexact('tanh', x))

@implements(np.arctanh, module='numpy')
@partial(jit, inline=True)
def arctanh(x: ArrayLike, /) -> Array:
  return lax.atanh(*promote_args_inexact('arctanh', x))

@implements(np.sqrt, module='numpy')
@partial(jit, inline=True)
def sqrt(x: ArrayLike, /) -> Array:
  return lax.sqrt(*promote_args_inexact('sqrt', x))

@implements(np.cbrt, module='numpy')
@partial(jit, inline=True)
def cbrt(x: ArrayLike, /) -> Array:
  return lax.cbrt(*promote_args_inexact('cbrt', x))

@implements(np.add, module='numpy')
@partial(jit, inline=True)
def add(x: ArrayLike, y: ArrayLike, /) -> Array:
  x, y = promote_args("add", x, y)
  return lax.add(x, y) if x.dtype != bool else lax.bitwise_or(x, y)

@implements(np.multiply, module='numpy')
@partial(jit, inline=True)
def multiply(x: ArrayLike, y: ArrayLike, /) -> Array:
  x, y = promote_args("multiply", x, y)
  return lax.mul(x, y) if x.dtype != bool else lax.bitwise_and(x, y)

@implements(np.bitwise_and, module='numpy')
@partial(jit, inline=True)
def bitwise_and(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.bitwise_and(*promote_args("bitwise_and", x, y))

@implements(np.bitwise_or, module='numpy')
@partial(jit, inline=True)
def bitwise_or(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.bitwise_or(*promote_args("bitwise_or", x, y))

@implements(np.bitwise_xor, module='numpy')
@partial(jit, inline=True)
def bitwise_xor(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.bitwise_xor(*promote_args("bitwise_xor", x, y))

@implements(np.left_shift, module='numpy')
@partial(jit, inline=True)
def left_shift(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.shift_left(*promote_args_numeric("left_shift", x, y))

@implements(getattr(np, "bitwise_left_shift", np.left_shift), module='numpy')
@partial(jit, inline=True)
def bitwise_left_shift(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.shift_left(*promote_args_numeric("bitwise_left_shift", x, y))

@implements(np.equal, module='numpy')
@partial(jit, inline=True)
def equal(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.eq(*promote_args("equal", x, y))

@implements(np.not_equal, module='numpy')
@partial(jit, inline=True)
def not_equal(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.ne(*promote_args("not_equal", x, y))

@implements(np.subtract, module='numpy')
@partial(jit, inline=True)
def subtract(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.sub(*promote_args("subtract", x, y))

@implements(np.arctan2, module='numpy')
@partial(jit, inline=True)
def arctan2(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.atan2(*promote_args_inexact("arctan2", x, y))

@implements(np.minimum, module='numpy')
@partial(jit, inline=True)
def minimum(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.min(*promote_args("minimum", x, y))

@implements(np.maximum, module='numpy')
@partial(jit, inline=True)
def maximum(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.max(*promote_args("maximum", x, y))

@implements(np.float_power, module='numpy')
@partial(jit, inline=True)
def float_power(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.pow(*promote_args_inexact("float_power", x, y))

@implements(np.nextafter, module='numpy')
@partial(jit, inline=True)
def nextafter(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.nextafter(*promote_args_inexact("nextafter", x, y))

# Logical ops
@implements(np.logical_and, module='numpy')
@partial(jit, inline=True)
def logical_and(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.bitwise_and(*map(_to_bool, promote_args("logical_and", x, y)))

@implements(np.logical_or, module='numpy')
@partial(jit, inline=True)
def logical_or(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.bitwise_or(*map(_to_bool, promote_args("logical_or", x, y)))

@implements(np.logical_xor, module='numpy')
@partial(jit, inline=True)
def logical_xor(x: ArrayLike, y: ArrayLike, /) -> Array:
  return lax.bitwise_xor(*map(_to_bool, promote_args("logical_xor", x, y)))

@implements(np.logical_not, module='numpy')
@partial(jit, inline=True)
def logical_not(x: ArrayLike, /) -> Array:
  return lax.bitwise_not(*map(_to_bool, promote_args("logical_not", x)))

# Comparison ops
def _complex_comparison(lax_op: Callable[[ArrayLike, ArrayLike], Array],
                        x: Array, y: Array):
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return lax.select(lax.eq(x.real, y.real),
                      lax_op(x.imag, y.imag),
                      lax_op(x.real, y.real))
  return lax_op(x, y)

@implements(np.greater_equal, module='numpy')
@partial(jit, inline=True)
def greater_equal(x: ArrayLike, y: ArrayLike, /) -> Array:
  return _complex_comparison(lax.ge, *promote_args("greater_equal", x, y))

@implements(np.greater, module='numpy')
@partial(jit, inline=True)
def greater(x: ArrayLike, y: ArrayLike, /) -> Array:
  return _complex_comparison(lax.gt, *promote_args("greater", x, y))

@implements(np.less_equal, module='numpy')
@partial(jit, inline=True)
def less_equal(x: ArrayLike, y: ArrayLike, /) -> Array:
  return _complex_comparison(lax.le, *promote_args("less_equal", x, y))

@implements(np.less, module='numpy')
@partial(jit, inline=True)
def less(x: ArrayLike, y: ArrayLike, /) -> Array:
  return _complex_comparison(lax.lt, *promote_args("less", x, y))

# Array API aliases
@partial(jit, inline=True)
def acos(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arccos`"""
  return arccos(*promote_args('acos', x))

@partial(jit, inline=True)
def acosh(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arccosh`"""
  return arccosh(*promote_args('acosh', x))

@partial(jit, inline=True)
def asin(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arcsin`"""
  return arcsin(*promote_args('asin', x))

@partial(jit, inline=True)
def asinh(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arcsinh`"""
  return arcsinh(*promote_args('asinh', x))

@partial(jit, inline=True)
def atan(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arctan`"""
  return arctan(*promote_args('atan', x))

@partial(jit, inline=True)
def atanh(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arctanh`"""
  return arctanh(*promote_args('atanh', x))

@partial(jit, inline=True)
def atan2(x: ArrayLike, y: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.arctan2`"""
  return arctan2(*promote_args('atan2', x, y))

@jit
def bitwise_count(x: ArrayLike, /) -> Array:
  r"""Counts the number of 1 bits in the binary representation of the absolute value
  of each element of ``x``.

  LAX-backend implementation of :func:`numpy.bitwise_count`.

  Args:
    x: Input array, only accepts integer subtypes

  Returns:
    An array-like object containing the binary 1 bit counts of the absolute value of
    each element in ``x``, with the same shape as ``x`` of dtype uint8.

  Examples:
    >>> x1 = jnp.array([64, 32, 31, 20])
    >>> # 64 = 0b1000000, 32 = 0b100000, 31 = 0b11111, 20 = 0b10100
    >>> jnp.bitwise_count(x1)
    Array([1, 1, 5, 2], dtype=uint8)

    >>> x2 = jnp.array([-16, -7, 7])
    >>> # |-16| = 0b10000, |-7| = 0b111, 7 = 0b111
    >>> jnp.bitwise_count(x2)
    Array([1, 3, 3], dtype=uint8)

    >>> x3 = jnp.array([[2, -7],[-9, 7]])
    >>> # 2 = 0b10, |-7| = 0b111, |-9| = 0b1001, 7 = 0b111
    >>> jnp.bitwise_count(x3)
    Array([[1, 3],
           [2, 3]], dtype=uint8)
  """
  x, = promote_args_numeric("bitwise_count", x)
  # Following numpy we take the absolute value and return uint8.
  return lax.population_count(abs(x)).astype('uint8')

@partial(jit, inline=True)
def right_shift(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  r"""Right shift the bits of ``x1`` to the amount specified in ``x2``.

  LAX-backend implementation of :func:`numpy.right_shift`.

  Args:
    x1: Input array, only accepts unsigned integer subtypes
    x2: The amount of bits to shift each element in ``x1`` to the right, only accepts
      integer subtypes

  Returns:
    An array-like object containing the right shifted elements of ``x1`` by the
    amount specified in ``x2``, with the same shape as the broadcasted shape of
    ``x1`` and ``x2``.

  Note:
    If ``x1.shape != x2.shape``, they must be compatible for broadcasting to a
    shared shape, this shared shape will also be the shape of the output. Right shifting
    a scalar x1 by scalar x2 is equivalent to ``x1 // 2**x2``.

  Examples:
    >>> def print_binary(x):
    ...   return [bin(int(val)) for val in x]

    >>> x1 = jnp.array([1, 2, 4, 8])
    >>> print_binary(x1)
    ['0b1', '0b10', '0b100', '0b1000']
    >>> x2 = 1
    >>> result = jnp.right_shift(x1, x2)
    >>> result
    Array([0, 1, 2, 4], dtype=int32)
    >>> print_binary(result)
    ['0b0', '0b1', '0b10', '0b100']

    >>> x1 = 16
    >>> print_binary([x1])
    ['0b10000']
    >>> x2 = jnp.array([1, 2, 3, 4])
    >>> result = jnp.right_shift(x1, x2)
    >>> result
    Array([8, 4, 2, 1], dtype=int32)
    >>> print_binary(result)
    ['0b1000', '0b100', '0b10', '0b1']
  """
  x1, x2 = promote_args_numeric(np.right_shift.__name__, x1, x2)
  lax_fn = lax.shift_right_logical if \
    np.issubdtype(x1.dtype, np.unsignedinteger) else lax.shift_right_arithmetic
  return lax_fn(x1, x2)

@implements(getattr(np, "bitwise_right_shift", np.right_shift), module='numpy')
@partial(jit, inline=True)
def bitwise_right_shift(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_numeric("bitwise_right_shift", x1, x2)
  lax_fn = lax.shift_right_logical if \
    np.issubdtype(x1.dtype, np.unsignedinteger) else lax.shift_right_arithmetic
  return lax_fn(x1, x2)


@partial(jit, inline=True)
def absolute(x: ArrayLike, /) -> Array:
  r"""Calculate the absolute value element-wise.

  LAX-backend implementation of :func:`numpy.absolute`.

  This is the same function as :func:`jax.numpy.abs`.

  Args:
    x: Input array

  Returns:
    An array-like object containing the absolute value of each element in ``x``,
    with the same shape as ``x``. For complex valued input, :math:`a + ib`,
    the absolute value is :math:`\sqrt{a^2+b^2}`.

  Examples:
    >>> x1 = jnp.array([5, -2, 0, 12])
    >>> jnp.absolute(x1)
    Array([ 5,  2,  0, 12], dtype=int32)

    >>> x2 = jnp.array([[ 8, -3, 1],[ 0, 9, -6]])
    >>> jnp.absolute(x2)
    Array([[8, 3, 1],
           [0, 9, 6]], dtype=int32)

    >>> x3 = jnp.array([8 + 15j, 3 - 4j, -5 + 0j])
    >>> jnp.absolute(x3)
    Array([17.,  5.,  5.], dtype=float32)
  """
  check_arraylike('absolute', x)
  dt = dtypes.dtype(x)
  return lax.asarray(x) if dt == np.bool_ or dtypes.issubdtype(dt, np.unsignedinteger) else lax.abs(x)


@partial(jit, inline=True)
def abs(x: ArrayLike, /) -> Array:
  """Alias of :func:`jax.numpy.absolute`."""
  return absolute(x)


@jit
def rint(x: ArrayLike, /) -> Array:
  """Rounds the elements of x to the nearest integer

  LAX-backend implementation of :func:`numpy.rint`.

  Args:
    x: Input array

  Returns:
    An array-like object containing the rounded elements of ``x``. Always promotes
    to inexact.

  Note:
    If an element of x is exactly half way, e.g. ``0.5`` or ``1.5``, rint will round
    to the nearest even integer.

  Example:
    >>> x1 = jnp.array([5, 4, 7])
    >>> jnp.rint(x1)
    Array([5., 4., 7.], dtype=float32)

    >>> x2 = jnp.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    >>> jnp.rint(x2)
    Array([-2., -2., -0.,  0.,  2.,  2.,  4.,  4.], dtype=float32)

    >>> x3 = jnp.array([-2.5+3.5j, 4.5-0.5j])
    >>> jnp.rint(x3)
    Array([-2.+4.j,  4.-0.j], dtype=complex64)
  """
  check_arraylike('rint', x)
  dtype = dtypes.dtype(x)
  if dtype == bool or dtypes.issubdtype(dtype, np.integer):
    return lax.convert_element_type(x, dtypes.float_)
  if dtypes.issubdtype(dtype, np.complexfloating):
    return lax.complex(rint(lax.real(x)), rint(lax.imag(x)))
  return lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)


@jit
def copysign(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Copies the sign of each element in ``x2`` to the corresponding element in ``x1``.

  LAX-backend implementation of :func:`numpy.copysign`.

  Args:
    x1: Input array
    x2: The array whose elements will be used to determine the sign, must be
      broadcast-compatible with ``x1``

  Returns:
    An array object containing the potentially changed elements of ``x1``, always promotes
    to inexact dtype, and has a shape of ``jnp.broadcast_shapes(x1.shape, x2.shape)``

  Examples:
    >>> x1 = jnp.array([5, 2, 0])
    >>> x2 = -1
    >>> jnp.copysign(x1, x2)
    Array([-5., -2., -0.], dtype=float32)

    >>> x1 = jnp.array([6, 8, 0])
    >>> x2 = 2
    >>> jnp.copysign(x1, x2)
    Array([6., 8., 0.], dtype=float32)

    >>> x1 = jnp.array([2, -3])
    >>> x2 = jnp.array([[1],[-4], [5]])
    >>> jnp.copysign(x1, x2)
    Array([[ 2.,  3.],
           [-2., -3.],
           [ 2.,  3.]], dtype=float32)
  """
  x1, x2 = promote_args_inexact("copysign", x1, x2)
  if dtypes.issubdtype(dtypes.dtype(x1), np.complexfloating):
    raise TypeError("copysign does not support complex-valued inputs")
  return _where(signbit(x2).astype(bool), -lax.abs(x1), lax.abs(x1))


@implements(np.true_divide, module='numpy')
@partial(jit, inline=True)
def true_divide(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_inexact("true_divide", x1, x2)
  return lax.div(x1, x2)

divide = true_divide


@jit
def floor_divide(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  """Calculates the floor division of x1 by x2 element-wise

  LAX-backend implementation of :func:`numpy.floor_divide`.

  Args:
    x1: Input array, the dividend
    x2: Input array, the divisor

  Returns:
    An array-like object containing each of the quotients rounded down
    to the nearest integer towards negative infinity. This is equivalent
    to ``x1 // x2`` in Python.

  Examples:
    >>> x1 = jnp.array([10, 20, 30])
    >>> x2 = jnp.array([3, 4, 7])
    >>> jnp.floor_divide(x1, x2)
    Array([3, 5, 4], dtype=int32)

    >>> x1 = jnp.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    >>> x2 = 3
    >>> jnp.floor_divide(x1, x2)
    Array([-2, -2, -1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=int32)

    >>> x1 = jnp.array([6, 6, 6], dtype=jnp.int32)
    >>> x2 = jnp.array([2.0, 2.5, 3.0], dtype=jnp.float32)
    >>> jnp.floor_divide(x1, x2)
    Array([3., 2., 2.], dtype=float32)

  Note:
    ``x1 // x2`` is equivalent to ``jnp.floor_divide(x1, x2)`` for arrays ``x1`` and ``x2``

  See Also:
    :func:`jnp.divide` and :func:`jnp.true_divide` for floating point division
  """
  x1, x2 = promote_args_numeric("floor_divide", x1, x2)
  dtype = dtypes.dtype(x1)
  if dtypes.issubdtype(dtype, np.unsignedinteger):
    return lax.div(x1, x2)
  elif dtypes.issubdtype(dtype, np.integer):
    quotient = lax.div(x1, x2)
    select = logical_and(lax.sign(x1) != lax.sign(x2), lax.rem(x1, x2) != 0)
    # TODO(mattjj): investigate why subtracting a scalar was causing promotion
    return _where(select, quotient - 1, quotient)
  elif dtypes.issubdtype(dtype, np.complexfloating):
    raise TypeError("floor_divide does not support complex-valued inputs")
  else:
    return _float_divmod(x1, x2)[0]


@jit
def divmod(x1: ArrayLike, x2: ArrayLike, /) -> tuple[Array, Array]:
  """Calculates the integer quotient and remainder of x1 by x2 element-wise

  LAX-backend implementation of :func:`numpy.divmod`.

  Args:
    x1: Input array, the dividend
    x2: Input array, the divisor

  Returns:
    A tuple of arrays ``(x1 // x2, x1 % x2)``.

  Examples:
    >>> x1 = jnp.array([10, 20, 30])
    >>> x2 = jnp.array([3, 4, 7])
    >>> jnp.divmod(x1, x2)
    (Array([3, 5, 4], dtype=int32), Array([1, 0, 2], dtype=int32))

    >>> x1 = jnp.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    >>> x2 = 3
    >>> jnp.divmod(x1, x2)
    (Array([-2, -2, -1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=int32),
     Array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=int32))

    >>> x1 = jnp.array([6, 6, 6], dtype=jnp.int32)
    >>> x2 = jnp.array([1.9, 2.5, 3.1], dtype=jnp.float32)
    >>> jnp.divmod(x1, x2)
    (Array([3., 2., 1.], dtype=float32),
     Array([0.30000007, 1.        , 2.9       ], dtype=float32))

  See Also:
    - :func:`jax.numpy.floor_divide`: floor division function
    - :func:`jax.numpy.remainder`: remainder function
  """
  x1, x2 = promote_args_numeric("divmod", x1, x2)
  if dtypes.issubdtype(dtypes.dtype(x1), np.integer):
    return floor_divide(x1, x2), remainder(x1, x2)
  else:
    return _float_divmod(x1, x2)


def _float_divmod(x1: ArrayLike, x2: ArrayLike) -> tuple[Array, Array]:
  # see float_divmod in floatobject.c of CPython
  mod = lax.rem(x1, x2)
  div = lax.div(lax.sub(x1, mod), x2)

  ind = lax.bitwise_and(mod != 0, lax.sign(x2) != lax.sign(mod))
  mod = lax.select(ind, mod + x2, mod)
  div = lax.select(ind, div - _constant_like(div, 1), div)

  return lax.round(div), mod


@implements(np.power, module='numpy')
def power(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike("power", x1, x2)
  check_no_float0s("power", x1, x2)

  # We apply special cases, both for algorithmic and autodiff reasons:
  #  1. for *concrete* integer scalar powers (and arbitrary bases), we use
  #     unrolled binary exponentiation specialized on the exponent, which is
  #     more precise for e.g. x ** 2 when x is a float (algorithmic reason!);
  #  2. for integer bases and integer powers, use unrolled binary exponentiation
  #     where the number of steps is determined by a max bit width of 64
  #     (algorithmic reason!);
  #  3. for integer powers and float/complex bases, we apply the lax primitive
  #     without any promotion of input types because in this case we want the
  #     function to be differentiable wrt its first argument at 0;
  #  3. for other cases, perform jnp dtype promotion on the arguments then apply
  #     lax.pow.

  # Case 1: concrete integer scalar powers:
  if isinstance(core.get_aval(x2), core.ConcreteArray):
    try:
      x2 = operator.index(x2)  # type: ignore[arg-type]
    except TypeError:
      pass
    else:
      x1, = promote_dtypes_numeric(x1)
      return lax.integer_pow(x1, x2)

  # Handle cases #2 and #3 under a jit:
  return _power(x1, x2)

# Array API alias
pow = power

@partial(jit, inline=True)
def _power(x1: ArrayLike, x2: ArrayLike) -> Array:
  x1, x2 = promote_shapes("power", x1, x2)  # not dtypes

  # Case 2: bool/integer result
  x1_, x2_ = promote_args_numeric("power", x1, x2)
  if (dtypes.issubdtype(dtypes.dtype(x1_), np.integer) or
      dtypes.issubdtype(dtypes.dtype(x1_), np.bool_)):
    assert np.iinfo(dtypes.dtype(x1_)).bits <= 64  # _pow_int_int assumes <=64bit
    return _pow_int_int(x1_, x2_)

  # Case 3: float/complex base with integer power (special autodiff behavior)
  d1, d2 = dtypes.dtype(x1), dtypes.dtype(x2)
  if dtypes.issubdtype(d1, np.inexact) and dtypes.issubdtype(d2, np.integer):
    return lax.pow(x1, x2)


  # Case 4: do promotion first
  return lax.pow(x1_, x2_)

# TODO(phawkins): add integer pow support to XLA.
def _pow_int_int(x1, x2):
  # Integer power => use binary exponentiation.
  bits = 6  # Anything more would overflow for any x1 > 1
  zero = _constant_like(x2, 0)
  one = _constant_like(x2, 1)
  # Initialize acc carefully such that pow(0, x2) is zero for x2 != 0
  acc = _where(lax.bitwise_and(lax.eq(x1, zero), lax.ne(x2, zero)), zero, one)
  for _ in range(bits):
    acc = _where(lax.bitwise_and(x2, one), lax.mul(acc, x1), acc)
    x1 = lax.mul(x1, x1)
    x2 = lax.shift_right_logical(x2, one)
  return acc


@custom_jvp
@implements(np.logaddexp, module='numpy')
@jit
def logaddexp(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_inexact("logaddexp", x1, x2)
  amax = lax.max(x1, x2)
  if dtypes.issubdtype(x1.dtype, np.floating):
    delta = lax.sub(x1, x2)
    return lax.select(lax._isnan(delta),
                      lax.add(x1, x2),  # NaNs or infinities of the same sign.
                      lax.add(amax, lax.log1p(lax.exp(lax.neg(lax.abs(delta))))))
  else:
    delta = lax.sub(lax.add(x1, x2), lax.mul(amax, _constant_like(amax, 2)))
    out = lax.add(amax, lax.log1p(lax.exp(delta)))
    return lax.complex(lax.real(out), _wrap_between(lax.imag(out), np.pi))


def _wrap_between(x, _a):
  """Wraps `x` between `[-a, a]`."""
  a = _constant_like(x, _a)
  two_a = _constant_like(x, 2 * _a)
  zero = _constant_like(x, 0)
  rem = lax.rem(lax.add(x, a), two_a)
  rem = lax.select(lax.lt(rem, zero), lax.add(rem, two_a), rem)
  return lax.sub(rem, a)


@logaddexp.defjvp
def _logaddexp_jvp(primals, tangents):
  x1, x2 = primals
  t1, t2 = tangents
  x1, x2, t1, t2 = promote_args_inexact("logaddexp_jvp", x1, x2, t1, t2)
  primal_out = logaddexp(x1, x2)
  tangent_out = lax.add(lax.mul(t1, exp(lax.sub(_replace_inf(x1), _replace_inf(primal_out)))),
                        lax.mul(t2, exp(lax.sub(_replace_inf(x2), _replace_inf(primal_out)))))
  return primal_out, tangent_out


@custom_jvp
@implements(np.logaddexp2, module='numpy')
@jit
def logaddexp2(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_inexact("logaddexp2", x1, x2)
  amax = lax.max(x1, x2)
  if dtypes.issubdtype(x1.dtype, np.floating):
    delta = lax.sub(x1, x2)
    return lax.select(lax._isnan(delta),
                      lax.add(x1, x2),  # NaNs or infinities of the same sign.
                      lax.add(amax, lax.div(lax.log1p(exp2(lax.neg(lax.abs(delta)))),
                                            _constant_like(x1, np.log(2)))))
  else:
    delta = lax.sub(lax.add(x1, x2), lax.mul(amax, _constant_like(amax, 2)))
    out = lax.add(amax, lax.div(lax.log1p(exp2(delta)), _constant_like(x1, np.log(2))))
    return lax.complex(lax.real(out), _wrap_between(lax.imag(out), np.pi / np.log(2)))


@logaddexp2.defjvp
def _logaddexp2_jvp(primals, tangents):
  x1, x2 = primals
  t1, t2 = tangents
  x1, x2, t1, t2 = promote_args_inexact("logaddexp2_jvp", x1, x2, t1, t2)
  primal_out = logaddexp2(x1, x2)
  tangent_out = lax.add(lax.mul(t1, exp2(lax.sub(_replace_inf(x1), _replace_inf(primal_out)))),
                        lax.mul(t2, exp2(lax.sub(_replace_inf(x2), _replace_inf(primal_out)))))
  return primal_out, tangent_out


@implements(np.log2, module='numpy')
@partial(jit, inline=True)
def log2(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("log2", x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 2)))


@implements(np.log10, module='numpy')
@partial(jit, inline=True)
def log10(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("log10", x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 10)))


@implements(np.exp2, module='numpy')
@partial(jit, inline=True)
def exp2(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("exp2", x)
  return lax.exp2(x)


@implements(np.signbit, module='numpy')
@jit
def signbit(x: ArrayLike, /) -> Array:
  x, = promote_args("signbit", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.integer):
    return lax.lt(x, _constant_like(x, 0))
  elif dtypes.issubdtype(dtype, np.bool_):
    return lax.full_like(x, False, dtype=np.bool_)
  elif not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(
        "jax.numpy.signbit is not well defined for %s" % dtype)

  info = dtypes.finfo(dtype)
  if info.bits not in _INT_DTYPES:
    raise NotImplementedError(
        "jax.numpy.signbit only supports 16, 32, and 64-bit types.")
  int_type = _INT_DTYPES[info.bits]
  x = lax.bitcast_convert_type(x, int_type)
  return lax.convert_element_type(x >> (info.nexp + info.nmant), np.bool_)


def _normalize_float(x):
  info = dtypes.finfo(dtypes.dtype(x))
  int_type = _INT_DTYPES[info.bits]
  cond = lax.abs(x) < info.tiny
  x1 = _where(cond, x * _lax_const(x, 1 << info.nmant), x)
  x2 = _where(cond, int_type(-info.nmant), int_type(0))
  return lax.bitcast_convert_type(x1, int_type), x2


@implements(np.ldexp, module='numpy')
@jit
def ldexp(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike("ldexp", x1, x2)
  x1_dtype = dtypes.dtype(x1)
  x2_dtype = dtypes.dtype(x2)
  if (dtypes.issubdtype(x1_dtype, np.complexfloating)
      or dtypes.issubdtype(x2_dtype, np.inexact)):
    raise ValueError(f"ldexp not supported for input types {(x1_dtype, x2_dtype)}")

  x1, x2 = promote_shapes("ldexp", x1, x2)

  dtype = dtypes.canonicalize_dtype(dtypes.to_inexact_dtype(x1_dtype))
  info = dtypes.finfo(dtype)
  int_type = _INT_DTYPES[info.bits]

  x1 = lax.convert_element_type(x1, dtype)
  x2 = lax.convert_element_type(x2, int_type)

  mask = (1 << info.nexp) - 1
  bias = 1 - info.minexp
  x, e = _normalize_float(x1)
  x2 += e + ((x >> info.nmant) & mask) - bias

  # find underflow/overflow before denormalization
  underflow_cond = less(x2, -(bias + info.nmant))
  overflow_cond = greater(x2, bias)

  m = lax.full_like(x, 1, dtype=dtype)

  # denormals
  cond = less(x2, -bias + 1)
  x2 = _where(cond, x2 + info.nmant, x2)
  m = _where(cond, m / (1 << info.nmant), m)

  x2 = lax.convert_element_type(x2, np.int32)
  x &= ~(mask << info.nmant)
  x |= ((lax.convert_element_type(x2, int_type) + bias) << info.nmant)

  x = lax.convert_element_type(m, dtype) * lax.bitcast_convert_type(x, dtype)

  # underflow
  x = _where(underflow_cond, lax.full_like(x, 0, dtype=dtype), x)
  # overflow
  x = _where(overflow_cond, lax.sign(x1) * lax.full_like(x, np.inf), x)
  # ldexp(x1, x2) = x1 for x1 = inf, -inf, nan, 0
  return _where(isinf(x1) | isnan(x1) | (x1 == 0), x1, x)


@implements(np.frexp, module='numpy')
@jit
def frexp(x: ArrayLike, /) -> tuple[Array, Array]:
  check_arraylike("frexp", x)
  x, = promote_dtypes_inexact(x)
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    raise TypeError("frexp does not support complex-valued inputs")

  dtype = dtypes.dtype(x)
  info = dtypes.finfo(dtype)
  mask = (1 << info.nexp) - 1
  bias = 1 - info.minexp

  x1, x2 = _normalize_float(x)
  x2 += ((x1 >> info.nmant) & mask) - bias + 1
  x1 &= ~(mask << info.nmant)
  x1 |= (bias - 1) << info.nmant
  x1 = lax.bitcast_convert_type(x1, dtype)

  cond = isinf(x) | isnan(x) | (x == 0)
  x2 = _where(cond, lax._zeros(x2), x2)
  return _where(cond, x, x1), lax.convert_element_type(x2, np.int32)


@implements(np.remainder, module='numpy')
@jit
def remainder(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_numeric("remainder", x1, x2)
  zero = _constant_like(x1, 0)
  if dtypes.issubdtype(x2.dtype, np.integer):
    x2 = _where(x2 == 0, lax._ones(x2), x2)
  trunc_mod = lax.rem(x1, x2)
  trunc_mod_not_zero = lax.ne(trunc_mod, zero)
  do_plus = lax.bitwise_and(
      lax.ne(lax.lt(trunc_mod, zero), lax.lt(x2, zero)), trunc_mod_not_zero)
  return lax.select(do_plus, lax.add(trunc_mod, x2), trunc_mod)
mod = implements(np.mod, module='numpy')(remainder)


@implements(np.fmod, module='numpy')
@jit
def fmod(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike("fmod", x1, x2)
  if dtypes.issubdtype(dtypes.result_type(x1, x2), np.integer):
    x2 = _where(x2 == 0, lax._ones(x2), x2)
  return lax.rem(*promote_args_numeric("fmod", x1, x2))


@implements(np.square, module='numpy')
@partial(jit, inline=True)
def square(x: ArrayLike, /) -> Array:
  check_arraylike("square", x)
  x, = promote_dtypes_numeric(x)
  return lax.integer_pow(x, 2)


@implements(np.deg2rad, module='numpy')
@partial(jit, inline=True)
def deg2rad(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("deg2rad", x)
  return lax.mul(x, _lax_const(x, np.pi / 180))


@implements(np.rad2deg, module='numpy')
@partial(jit, inline=True)
def rad2deg(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("rad2deg", x)
  return lax.mul(x, _lax_const(x, 180 / np.pi))


degrees = rad2deg
radians = deg2rad


@implements(np.conjugate, module='numpy')
@partial(jit, inline=True)
def conjugate(x: ArrayLike, /) -> Array:
  check_arraylike("conjugate", x)
  return lax.conj(x) if np.iscomplexobj(x) else lax.asarray(x)
conj = conjugate


@implements(np.imag)
@partial(jit, inline=True)
def imag(val: ArrayLike, /) -> Array:
  check_arraylike("imag", val)
  return lax.imag(val) if np.iscomplexobj(val) else lax.full_like(val, 0)


@implements(np.real)
@partial(jit, inline=True)
def real(val: ArrayLike, /) -> Array:
  check_arraylike("real", val)
  return lax.real(val) if np.iscomplexobj(val) else lax.asarray(val)

@implements(np.modf, module='numpy', skip_params=['out'])
@jit
def modf(x: ArrayLike, /, out=None) -> tuple[Array, Array]:
  check_arraylike("modf", x)
  x, = promote_dtypes_inexact(x)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.modf is not supported.")
  whole = _where(lax.ge(x, lax._zero(x)), floor(x), ceil(x))
  return x - whole, whole


@implements(np.isfinite, module='numpy')
@partial(jit, inline=True)
def isfinite(x: ArrayLike, /) -> Array:
  check_arraylike("isfinite", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.is_finite(x)
  elif dtypes.issubdtype(dtype, np.complexfloating):
    return lax.bitwise_and(lax.is_finite(real(x)), lax.is_finite(imag(x)))
  else:
    return lax.full_like(x, True, dtype=np.bool_)


@implements(np.isinf, module='numpy')
@jit
def isinf(x: ArrayLike, /) -> Array:
  check_arraylike("isinf", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.eq(lax.abs(x), _constant_like(x, np.inf))
  elif dtypes.issubdtype(dtype, np.complexfloating):
    re = lax.real(x)
    im = lax.imag(x)
    return lax.bitwise_or(lax.eq(lax.abs(re), _constant_like(re, np.inf)),
                          lax.eq(lax.abs(im), _constant_like(im, np.inf)))
  else:
    return lax.full_like(x, False, dtype=np.bool_)


def _isposneginf(infinity: float, x: ArrayLike, out) -> Array:
  if out is not None:
    raise NotImplementedError("The 'out' argument to isneginf/isposinf is not supported.")
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.eq(x, _constant_like(x, infinity))
  elif dtypes.issubdtype(dtype, np.complexfloating):
    raise ValueError("isposinf/isneginf are not well defined for complex types")
  else:
    return lax.full_like(x, False, dtype=np.bool_)


@implements(np.isposinf, module='numpy')
def isposinf(x, /, out=None):
  return _isposneginf(np.inf, x, out)


@implements(np.isposinf, module='numpy')
def isneginf(x, /, out=None):
  return _isposneginf(-np.inf, x, out)


@implements(np.isnan, module='numpy')
@partial(jit, inline=True)
def isnan(x: ArrayLike, /) -> Array:
  check_arraylike("isnan", x)
  return lax.ne(x, x)


@implements(np.heaviside, module='numpy')
@jit
def heaviside(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike("heaviside", x1, x2)
  x1, x2 = promote_dtypes_inexact(x1, x2)
  zero = _lax_const(x1, 0)
  return _where(lax.lt(x1, zero), zero,
                _where(lax.gt(x1, zero), _lax_const(x1, 1), x2))


@implements(np.hypot, module='numpy')
@jit
def hypot(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_inexact("hypot", x1, x2)

  # TODO(micky774): Promote to ValueError when deprecation is complete
  # (began 2024-4-14).
  if dtypes.issubdtype(x1.dtype, np.complexfloating):
    warnings.warn(
      "Passing complex-valued inputs to hypot is deprecated and will raise a "
      "ValueError in the future. Please convert to real values first, such as "
      "by using jnp.real or jnp.imag to take the real or imaginary components "
      "respectively.",
      DeprecationWarning, stacklevel=2)
  x1, x2 = lax.abs(x1), lax.abs(x2)
  idx_inf = lax.bitwise_or(isposinf(x1), isposinf(x2))
  x1, x2 = maximum(x1, x2), minimum(x1, x2)
  x = _where(x1 == 0, x1, x1 * lax.sqrt(1 + lax.square(lax.div(x2, _where(x1 == 0, lax._ones(x1), x1)))))
  return _where(idx_inf, _lax_const(x, np.inf), x)


@implements(np.reciprocal, module='numpy')
@partial(jit, inline=True)
def reciprocal(x: ArrayLike, /) -> Array:
  check_arraylike("reciprocal", x)
  x, = promote_dtypes_inexact(x)
  return lax.integer_pow(x, -1)


@implements(np.sinc, update_doc=False)
@jit
def sinc(x: ArrayLike, /) -> Array:
  check_arraylike("sinc", x)
  x, = promote_dtypes_inexact(x)
  eq_zero = lax.eq(x, _lax_const(x, 0))
  pi_x = lax.mul(_lax_const(x, np.pi), x)
  safe_pi_x = _where(eq_zero, _lax_const(x, 1), pi_x)
  return _where(eq_zero, _sinc_maclaurin(0, pi_x),
                lax.div(lax.sin(safe_pi_x), safe_pi_x))


@partial(custom_jvp, nondiff_argnums=(0,))
def _sinc_maclaurin(k, x):
  # compute the kth derivative of x -> sin(x)/x evaluated at zero (since we
  # compute the monomial term in the jvp rule)
  # TODO(mattjj): see https://github.com/google/jax/issues/10750
  if k % 2:
    return x * 0
  else:
    return x * 0 + _lax_const(x, (-1) ** (k // 2) / (k + 1))

@_sinc_maclaurin.defjvp
def _sinc_maclaurin_jvp(k, primals, tangents):
  (x,), (t,) = primals, tangents
  return _sinc_maclaurin(k, x), _sinc_maclaurin(k + 1, x) * t
