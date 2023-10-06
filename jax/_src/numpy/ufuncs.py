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

from functools import partial
import operator
from textwrap import dedent
from typing import Any, Callable, Union, overload

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
   promote_shapes, _where, _wraps, check_no_float0s)

_lax_const = lax._const

_INT_DTYPES = {
  16: np.int16,
  32: np.int32,
  64: np.int64,
}

UnOp = Callable[[ArrayLike], Array]
BinOp = Callable[[ArrayLike, ArrayLike], Array]


def _constant_like(x, const):
  return np.array(const, dtype=dtypes.dtype(x))


def _replace_inf(x: ArrayLike) -> Array:
  return lax.select(isposinf(real(x)), lax._zeros(x), x)


def _one_to_one_unop(
    numpy_fn: Callable[..., Any], lax_fn: UnOp,
    promote_to_inexact: bool = False, lax_doc: bool = False) -> UnOp:
  if promote_to_inexact:
    fn = lambda x, /: lax_fn(*promote_args_inexact(numpy_fn.__name__, x))
  else:
    fn = lambda x, /: lax_fn(*promote_args(numpy_fn.__name__, x))
  fn.__qualname__ = f"jax.numpy.{numpy_fn.__name__}"
  fn = jit(fn, inline=True)
  if lax_doc:
    doc = dedent('\n\n'.join(lax_fn.__doc__.split('\n\n')[1:])).strip()  # type: ignore[union-attr]
    return _wraps(numpy_fn, lax_description=doc, module='numpy')(fn)
  else:
    return _wraps(numpy_fn, module='numpy')(fn)


def _one_to_one_binop(
    numpy_fn: Callable[..., Any], lax_fn: BinOp,
    promote_to_inexact: bool = False, lax_doc: bool = False,
    promote_to_numeric: bool = False) -> BinOp:
  if promote_to_inexact:
    fn = lambda x1, x2, /: lax_fn(*promote_args_inexact(numpy_fn.__name__, x1, x2))
  elif promote_to_numeric:
    fn = lambda x1, x2, /: lax_fn(*promote_args_numeric(numpy_fn.__name__, x1, x2))
  else:
    fn = lambda x1, x2, /: lax_fn(*promote_args(numpy_fn.__name__, x1, x2))
  fn.__qualname__ = f"jax.numpy.{numpy_fn.__name__}"
  fn = jit(fn, inline=True)
  if lax_doc:
    doc = dedent('\n\n'.join(lax_fn.__doc__.split('\n\n')[1:])).strip()  # type: ignore[union-attr]
    return _wraps(numpy_fn, lax_description=doc, module='numpy')(fn)
  else:
    return _wraps(numpy_fn, module='numpy')(fn)


def _maybe_bool_binop(
    numpy_fn: Callable[..., Any], lax_fn: BinOp, bool_lax_fn: BinOp,
    lax_doc: bool = False) -> BinOp:
  def fn(x1, x2, /):
    x1, x2 = promote_args(numpy_fn.__name__, x1, x2)
    return lax_fn(x1, x2) if x1.dtype != np.bool_ else bool_lax_fn(x1, x2)
  fn.__qualname__ = f"jax.numpy.{numpy_fn.__name__}"
  fn = jit(fn, inline=True)
  if lax_doc:
    doc = dedent('\n\n'.join(lax_fn.__doc__.split('\n\n')[1:])).strip()  # type: ignore[union-attr]
    return _wraps(numpy_fn, lax_description=doc, module='numpy')(fn)
  else:
    return _wraps(numpy_fn, module='numpy')(fn)


def _comparison_op(numpy_fn: Callable[..., Any], lax_fn: BinOp) -> BinOp:
  def fn(x1, x2, /):
    x1, x2 =  promote_args(numpy_fn.__name__, x1, x2)
    # Comparison on complex types are defined as a lexicographic ordering on
    # the (real, imag) pair.
    if dtypes.issubdtype(dtypes.dtype(x1), np.complexfloating):
      rx = lax.real(x1)
      ry = lax.real(x2)
      return lax.select(lax.eq(rx, ry), lax_fn(lax.imag(x1), lax.imag(x2)),
                        lax_fn(rx, ry))
    return lax_fn(x1, x2)
  fn.__qualname__ = f"jax.numpy.{numpy_fn.__name__}"
  fn = jit(fn, inline=True)
  return _wraps(numpy_fn, module='numpy')(fn)

@overload
def _logical_op(np_op: Callable[..., Any], bitwise_op: UnOp) -> UnOp: ...
@overload
def _logical_op(np_op: Callable[..., Any], bitwise_op: BinOp) -> BinOp: ...
@overload
def _logical_op(np_op: Callable[..., Any], bitwise_op: Union[UnOp, BinOp]) -> Union[UnOp, BinOp]: ...

def _logical_op(np_op: Callable[..., Any], bitwise_op: Union[UnOp, BinOp]) -> Union[UnOp, BinOp]:
  @_wraps(np_op, update_doc=False, module='numpy')
  @partial(jit, inline=True)
  def op(*args):
    zero = lambda x: lax.full_like(x, shape=(), fill_value=0)
    args = (x if dtypes.issubdtype(dtypes.dtype(x), np.bool_) else lax.ne(x, zero(x))
            for x in args)
    return bitwise_op(*promote_args(np_op.__name__, *args))
  return op


fabs = _one_to_one_unop(np.fabs, lax.abs, True)
bitwise_not = _one_to_one_unop(np.bitwise_not, lax.bitwise_not)
invert = _one_to_one_unop(np.invert, lax.bitwise_not)
negative = _one_to_one_unop(np.negative, lax.neg)
positive = _one_to_one_unop(np.positive, lambda x: lax.asarray(x))
floor = _one_to_one_unop(np.floor, lax.floor, True)
ceil = _one_to_one_unop(np.ceil, lax.ceil, True)
exp = _one_to_one_unop(np.exp, lax.exp, True)
log = _one_to_one_unop(np.log, lax.log, True)
expm1 = _one_to_one_unop(np.expm1, lax.expm1, True)
log1p = _one_to_one_unop(np.log1p, lax.log1p, True)
sin = _one_to_one_unop(np.sin, lax.sin, True)
cos = _one_to_one_unop(np.cos, lax.cos, True)
tan = _one_to_one_unop(np.tan, lax.tan, True)
arcsin = _one_to_one_unop(np.arcsin, lax.asin, True)
arccos = _one_to_one_unop(np.arccos, lax.acos, True)
arctan = _one_to_one_unop(np.arctan, lax.atan, True)
sinh = _one_to_one_unop(np.sinh, lax.sinh, True)
cosh = _one_to_one_unop(np.cosh, lax.cosh, True)
arcsinh = _one_to_one_unop(np.arcsinh, lax.asinh, True)
tanh = _one_to_one_unop(np.tanh, lax.tanh, True)
arctanh = _one_to_one_unop(np.arctanh, lax.atanh, True)
sqrt = _one_to_one_unop(np.sqrt, lax.sqrt, True)
cbrt = _one_to_one_unop(np.cbrt, lax.cbrt, True)

add = _maybe_bool_binop(np.add, lax.add, lax.bitwise_or)
bitwise_and = _one_to_one_binop(np.bitwise_and, lax.bitwise_and)
bitwise_or = _one_to_one_binop(np.bitwise_or, lax.bitwise_or)
bitwise_xor = _one_to_one_binop(np.bitwise_xor, lax.bitwise_xor)
left_shift = _one_to_one_binop(np.left_shift, lax.shift_left, promote_to_numeric=True)
equal = _one_to_one_binop(np.equal, lax.eq)
multiply = _maybe_bool_binop(np.multiply, lax.mul, lax.bitwise_and)
not_equal = _one_to_one_binop(np.not_equal, lax.ne)
subtract = _one_to_one_binop(np.subtract, lax.sub)
arctan2 = _one_to_one_binop(np.arctan2, lax.atan2, True)
minimum = _one_to_one_binop(np.minimum, lax.min)
maximum = _one_to_one_binop(np.maximum, lax.max)
float_power = _one_to_one_binop(np.float_power, lax.pow, True)
nextafter = _one_to_one_binop(np.nextafter, lax.nextafter, True, True)

greater_equal = _comparison_op(np.greater_equal, lax.ge)
greater = _comparison_op(np.greater, lax.gt)
less_equal = _comparison_op(np.less_equal, lax.le)
less = _comparison_op(np.less, lax.lt)

logical_and: BinOp = _logical_op(np.logical_and, lax.bitwise_and)
logical_not: UnOp = _logical_op(np.logical_not, lax.bitwise_not)
logical_or: BinOp = _logical_op(np.logical_or, lax.bitwise_or)
logical_xor: BinOp = _logical_op(np.logical_xor, lax.bitwise_xor)

@_wraps(np.arccosh, module='numpy')
@jit
def arccosh(x: ArrayLike, /) -> Array:
  # Note: arccosh is multi-valued for complex input, and lax.acosh uses a different
  # convention than np.arccosh.
  out = lax.acosh(*promote_args_inexact("arccosh", x))
  if dtypes.issubdtype(out.dtype, np.complexfloating):
    out = _where(real(out) < 0, lax.neg(out), out)
  return out

@_wraps(getattr(np, 'bitwise_count', None), module='numpy')
@jit
def bitwise_count(x: ArrayLike, /) -> Array:
  # Ref: https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
  check_arraylike('bitwise_count', x)
  a = lax.asarray(x)
  if not dtypes.issubdtype(a.dtype, np.integer):
    raise ValueError('bitwise_count is implemented only for integers inputs.')
  nbits = np.iinfo(a.dtype).bits
  if nbits not in [8, 16, 32, 64]:
    raise ValueError("bitwise_count is implemented only for 8, 16, 32, and 64-bit")
  if not dtypes.issubdtype(a.dtype, np.unsignedinteger):
    # For signed integers, follow numpy's convention of taking the abs value.
    a = abs(a).astype(f'uint{nbits}')
  B0, B1, B2, B3 = np.array(
    [0x5555555555555555, 0x3333333333333333, 0x0f0f0f0f0f0f0f0f, 0x0101010101010101],
    dtype='uint64').astype(a.dtype)
  a = a - ((a >> 1) & B0)
  a = (a & B1) + ((a >> 2) & B1)
  a = (a + (a >> 4)) & B2
  count = a if nbits == 8 else (a * B3) >> (nbits - 8)
  # Following numpy's convention, we always return uint8.
  return count.astype('uint8')

@_wraps(np.right_shift, module='numpy')
@partial(jit, inline=True)
def right_shift(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_numeric(np.right_shift.__name__, x1, x2)
  lax_fn = lax.shift_right_logical if \
    np.issubdtype(x1.dtype, np.unsignedinteger) else lax.shift_right_arithmetic
  return lax_fn(x1, x2)


@_wraps(np.absolute, module='numpy')
@partial(jit, inline=True)
def absolute(x: ArrayLike, /) -> Array:
  check_arraylike('absolute', x)
  dt = dtypes.dtype(x)
  return lax.asarray(x) if dt == np.bool_ or dtypes.issubdtype(dt, np.unsignedinteger) else lax.abs(x)
abs = _wraps(np.abs, module='numpy')(absolute)


@_wraps(np.rint, module='numpy')
@jit
def rint(x: ArrayLike, /) -> Array:
  check_arraylike('rint', x)
  dtype = dtypes.dtype(x)
  if dtype == bool or dtypes.issubdtype(dtype, np.integer):
    return lax.convert_element_type(x, dtypes.float_)
  if dtypes.issubdtype(dtype, np.complexfloating):
    return lax.complex(rint(lax.real(x)), rint(lax.imag(x)))
  return lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)


@_wraps(np.sign, module='numpy')
@jit
def sign(x: ArrayLike, /) -> Array:
  check_arraylike('sign', x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.complexfloating):
    re = lax.real(x)
    return lax.complex(
      lax.sign(_where(re != 0, re, lax.imag(x))), _constant_like(re, 0))
  return lax.sign(x)


@_wraps(np.copysign, module='numpy')
@jit
def copysign(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_inexact("copysign", x1, x2)
  if dtypes.issubdtype(dtypes.dtype(x1), np.complexfloating):
    raise TypeError("copysign does not support complex-valued inputs")
  return _where(signbit(x2).astype(bool), -lax.abs(x1), lax.abs(x1))


@_wraps(np.true_divide, module='numpy')
@partial(jit, inline=True)
def true_divide(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_inexact("true_divide", x1, x2)
  return lax.div(x1, x2)

divide = true_divide


@_wraps(np.floor_divide, module='numpy')
@jit
def floor_divide(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  x1, x2 = promote_args_numeric("floor_divide", x1, x2)
  dtype = dtypes.dtype(x1)
  if dtypes.issubdtype(dtype, np.integer):
    quotient = lax.div(x1, x2)
    select = logical_and(lax.sign(x1) != lax.sign(x2), lax.rem(x1, x2) != 0)
    # TODO(mattjj): investigate why subtracting a scalar was causing promotion
    return _where(select, quotient - 1, quotient)
  elif dtypes.issubdtype(dtype, np.complexfloating):
    x1r = lax.real(x1)
    x1i = lax.imag(x1)
    x2r = lax.real(x2)
    x2i = lax.imag(x2)
    which = lax.ge(lax.abs(x2r), lax.abs(x2i))
    rat1 = _where(which, lax.full_like(x2i, 1), lax.div(x2r, x2i))
    rat2 = _where(which, lax.div(x2i, x2r), _lax_const(x2i, 1))
    out = lax.floor(lax.div(lax.add(lax.mul(x1r, rat1), lax.mul(x1i, rat2)),
                            lax.add(lax.mul(x2r, rat1), lax.mul(x2i, rat2))))
    return lax.convert_element_type(out, dtype)
  else:
    return _float_divmod(x1, x2)[0]


@_wraps(np.divmod, module='numpy')
@jit
def divmod(x1: ArrayLike, x2: ArrayLike, /) -> tuple[Array, Array]:
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


@_wraps(np.power, module='numpy')
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
@_wraps(np.logaddexp, module='numpy')
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
@_wraps(np.logaddexp2, module='numpy')
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


@_wraps(np.log2, module='numpy')
@partial(jit, inline=True)
def log2(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("log2", x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 2)))


@_wraps(np.log10, module='numpy')
@partial(jit, inline=True)
def log10(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("log10", x)
  return lax.div(lax.log(x), lax.log(_constant_like(x, 10)))


@_wraps(np.exp2, module='numpy')
@partial(jit, inline=True)
def exp2(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("exp2", x)
  return lax.exp2(x)


@_wraps(np.signbit, module='numpy')
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

  # TPU supports BF16 but not S16 types, so as a workaround, convert BF16 to
  # F32.
  if dtype == dtypes.bfloat16:
    dtype = np.dtype('float32')
    x = lax.convert_element_type(x, dtype)

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


@_wraps(np.ldexp, module='numpy')
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
  bias = ((1 << info.nexp) - 1) >> 1
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


@_wraps(np.frexp, module='numpy')
@jit
def frexp(x: ArrayLike, /) -> tuple[Array, Array]:
  check_arraylike("frexp", x)
  x, = promote_dtypes_inexact(x)
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    raise TypeError("frexp does not support complex-valued inputs")

  dtype = dtypes.dtype(x)
  info = dtypes.finfo(dtype)
  mask = (1 << info.nexp) - 1
  bias = ((1 << info.nexp) - 1) >> 1

  x1, x2 = _normalize_float(x)
  x2 += ((x1 >> info.nmant) & mask) - bias + 1
  x1 &= ~(mask << info.nmant)
  x1 |= (bias - 1) << info.nmant
  x1 = lax.bitcast_convert_type(x1, dtype)

  cond = isinf(x) | isnan(x) | (x == 0)
  x2 = _where(cond, lax._zeros(x2), x2)
  return _where(cond, x, x1), lax.convert_element_type(x2, np.int32)


@_wraps(np.remainder, module='numpy')
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
mod = _wraps(np.mod, module='numpy')(remainder)


@_wraps(np.fmod, module='numpy')
@jit
def fmod(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike("fmod", x1, x2)
  if dtypes.issubdtype(dtypes.result_type(x1, x2), np.integer):
    x2 = _where(x2 == 0, lax._ones(x2), x2)
  return lax.rem(*promote_args_numeric("fmod", x1, x2))


@_wraps(np.square, module='numpy')
@partial(jit, inline=True)
def square(x: ArrayLike, /) -> Array:
  check_arraylike("square", x)
  x, = promote_dtypes_numeric(x)
  return lax.integer_pow(x, 2)


@_wraps(np.deg2rad, module='numpy')
@partial(jit, inline=True)
def deg2rad(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("deg2rad", x)
  return lax.mul(x, _lax_const(x, np.pi / 180))


@_wraps(np.rad2deg, module='numpy')
@partial(jit, inline=True)
def rad2deg(x: ArrayLike, /) -> Array:
  x, = promote_args_inexact("rad2deg", x)
  return lax.mul(x, _lax_const(x, 180 / np.pi))


degrees = rad2deg
radians = deg2rad


@_wraps(np.conjugate, module='numpy')
@partial(jit, inline=True)
def conjugate(x: ArrayLike, /) -> Array:
  check_arraylike("conjugate", x)
  return lax.conj(x) if np.iscomplexobj(x) else lax.asarray(x)
conj = conjugate


@_wraps(np.imag)
@partial(jit, inline=True)
def imag(val: ArrayLike, /) -> Array:
  check_arraylike("imag", val)
  return lax.imag(val) if np.iscomplexobj(val) else lax.full_like(val, 0)


@_wraps(np.real)
@partial(jit, inline=True)
def real(val: ArrayLike, /) -> Array:
  check_arraylike("real", val)
  return lax.real(val) if np.iscomplexobj(val) else lax.asarray(val)

@_wraps(np.modf, module='numpy', skip_params=['out'])
@jit
def modf(x: ArrayLike, /, out=None) -> tuple[Array, Array]:
  check_arraylike("modf", x)
  x, = promote_dtypes_inexact(x)
  if out is not None:
    raise NotImplementedError("The 'out' argument to jnp.modf is not supported.")
  whole = _where(lax.ge(x, lax._zero(x)), floor(x), ceil(x))
  return x - whole, whole


@_wraps(np.isfinite, module='numpy')
@jit
def isfinite(x: ArrayLike, /) -> Array:
  check_arraylike("isfinite", x)
  dtype = dtypes.dtype(x)
  if dtypes.issubdtype(dtype, np.floating):
    return lax.is_finite(x)
  elif dtypes.issubdtype(dtype, np.complexfloating):
    return lax.bitwise_and(lax.is_finite(real(x)), lax.is_finite(imag(x)))
  else:
    return lax.full_like(x, True, dtype=np.bool_)


@_wraps(np.isinf, module='numpy')
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


isposinf: UnOp = _wraps(np.isposinf, skip_params=['out'])(
  lambda x, /, out=None: _isposneginf(np.inf, x, out)
)


isneginf: UnOp = _wraps(np.isneginf, skip_params=['out'])(
  lambda x, /, out=None: _isposneginf(-np.inf, x, out)
)


@_wraps(np.isnan, module='numpy')
@jit
def isnan(x: ArrayLike, /) -> Array:
  check_arraylike("isnan", x)
  return lax.ne(x, x)


@_wraps(np.heaviside, module='numpy')
@jit
def heaviside(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike("heaviside", x1, x2)
  x1, x2 = promote_dtypes_inexact(x1, x2)
  zero = _lax_const(x1, 0)
  return _where(lax.lt(x1, zero), zero,
                _where(lax.gt(x1, zero), _lax_const(x1, 1), x2))


@_wraps(np.hypot, module='numpy')
@jit
def hypot(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike("hypot", x1, x2)
  x1, x2 = promote_dtypes_inexact(x1, x2)
  x1 = lax.abs(x1)
  x2 = lax.abs(x2)
  x1, x2 = maximum(x1, x2), minimum(x1, x2)
  return lax.select(x1 == 0, x1, x1 * lax.sqrt(1 + lax.square(lax.div(x2, lax.select(x1 == 0, lax._ones(x1), x1)))))


@_wraps(np.reciprocal, module='numpy')
@partial(jit, inline=True)
def reciprocal(x: ArrayLike, /) -> Array:
  check_arraylike("reciprocal", x)
  x, = promote_dtypes_inexact(x)
  return lax.integer_pow(x, -1)


@_wraps(np.sinc, update_doc=False)
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
