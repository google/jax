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

import builtins
import enum
import functools
from functools import partial
import itertools
import operator
from typing import (Any, Callable, Optional, Sequence, Tuple, List, Dict,
                    TypeVar, Union, cast as type_cast)
import warnings

import numpy as np

import jax
from jax import core
from jax._src import ad_util
from jax._src import api
from jax._src import api_util
from jax._src import device_array
from jax._src import dispatch
from jax import linear_util as lu
from jax._src import dtypes
from jax import tree_util
from jax._src import source_info_util
from jax._src.config import config
from jax.core import (Primitive, UnshapedArray, ShapedArray, ConcreteArray,
                      raise_to_shaped, abstract_token, canonicalize_shape)
from jax._src.abstract_arrays import array_types
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.interpreters import ad
from jax.interpreters import batching
import jax._src.pretty_printer as pp
from jax._src import util
from jax._src.util import (cache, prod, safe_zip, safe_map, canonicalize_axis,
                           split_list)
from jax.tree_util import tree_map
from jax._src.lib import mlir_api_version
from jax._src.lib import pytree
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lax.utils import (
  _input_dtype,
  standard_abstract_eval,
  standard_multi_result_abstract_eval,
  standard_named_shape_rule,
  standard_primitive,
  standard_translate,
)
from jax._src.lax import slicing

xb = xla_bridge
xc = xla_client
xops = xla_client.ops
xe = xla_client._xla

_max = builtins.max
_min = builtins.min
_reduce = functools.reduce

Array = Any
DType = Any
Shape = core.Shape

T = TypeVar("T")

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def _validate_shapes(shapes: Sequence[Shape]):
  def _check_static_shape(shape: Shape):
    checked = canonicalize_shape(shape)
    if not all(idx >= 0 for idx in checked):
      msg = f"Only non-negative indices are allowed when broadcasting" \
            f" static shapes, but got shape {shape!r}."
      raise TypeError(msg)

  assert shapes
  if config.jax_dynamic_shapes:
    # pass dynamic shapes through unchecked
    return
  else:
    map(_check_static_shape, shapes)

def _try_broadcast_shapes(
    shapes: Sequence[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
  if len(shapes) == 1: return shapes[0]
  rank, *others = {len(shape) for shape in shapes}
  if others: return None  # must have consistent rank
  if not rank: return ()  # scalar case
  result_shape = []
  for ds in unsafe_zip(*shapes):
    if all(core.same_referent(d, ds[0]) for d in ds[1:]):
      # if all axes are identical objects, the resulting size is the object
      result_shape.append(ds[0])
    else:
      # if all dims are equal (or 1), the result is the non-1 size (or 1)
      non_1s = [d for d in ds if not core.symbolic_equal_dim(d, 1)]
      if not non_1s:
        result_shape.append(1)
      elif all(core.symbolic_equal_dim(non_1s[0], d) for d in non_1s[1:]):
        result_shape.append(non_1s[0])
      else:
        return None
  return tuple(result_shape)

def broadcast_shapes(*shapes: Tuple[Union[int, core.Tracer], ...]
                     ) -> Tuple[Union[int, core.Tracer], ...]:
  """Returns the shape that results from NumPy broadcasting of `shapes`."""
  # NOTE: We have both cached and uncached versions to handle Tracers in shapes.
  try:
    return _broadcast_shapes_cached(*shapes)
  except:
    return _broadcast_shapes_uncached(*shapes)

@cache()
def _broadcast_shapes_cached(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
  return _broadcast_shapes_uncached(*shapes)

def _broadcast_shapes_uncached(*shapes):
  _validate_shapes(shapes)
  fst, *rst = shapes
  if not rst: return fst

  # First check if we need only rank promotion (and not singleton-broadcasting).
  try: return _reduce(_broadcast_ranks, rst, fst)
  except ValueError: pass

  # Next try singleton-broadcasting, padding out ranks using singletons.
  ndim = _max(len(shape) for shape in shapes)
  shape_list = [(1,) * (ndim - len(shape)) + shape for shape in shapes]
  result_shape = _try_broadcast_shapes(shape_list)
  if result_shape is None:
    raise ValueError("Incompatible shapes for broadcasting: {}"
                     .format(tuple(shape_list)))
  return result_shape

def _broadcast_ranks(s1, s2):
  if len(s1) > len(s2):
    s1, s2 = s2, s1
  assert len(s1) <= len(s2)
  s1_ = s2[len(s2) - len(s1):]
  if core.symbolic_equal_shape(s1_, s1): return s2
  else: raise ValueError

def _identity(x): return x

def _extract_tracers_dyn_shape(
    shape: Sequence[Union[int, core.Tracer]]
  ) -> Tuple[List[core.Tracer], List[Optional[int]]]:
  # Given a sequence representing a shape, pull out Tracers, replacing with None
  if config.jax_dynamic_shapes:
    # We must gate this behavior under a flag because otherwise the errors
    # raised are different (and have worse source provenance information).
    dyn_shape = [d for d in shape if isinstance(d, core.Tracer)]
    static_shape = [None if isinstance(d, core.Tracer) else d for d in shape]
    return dyn_shape, static_shape
  else:
    return [], list(shape)  # type: ignore

def _nr_dynamic_dimensions(static_shape: Sequence[Optional[int]]) -> int:
  return len([d for d in static_shape if d is None])

def _merge_dyn_shape(
    static_shape: Sequence[Optional[int]],
    dyn_shape: Sequence[Any],
  ) -> Tuple[Union[int, mlir.Value], ...]:
  # Replace Nones in static_shape with elements of dyn_shape, in order
  dyn_shape_it = iter(dyn_shape)
  shape = tuple(next(dyn_shape_it) if d is None else d for d in static_shape)
  assert next(dyn_shape_it, None) is None
  return shape

def _dyn_shape_staging_rule(trace, prim, out_aval, *args, **params):
  source_info = source_info_util.current()
  out_tracer = pe.DynamicJaxprTracer(trace, out_aval, source_info)
  eqn = pe.new_jaxpr_eqn([trace.getvar(x) for x in args],
                         [trace.makevar(out_tracer)],
                         prim, params, core.no_effects, source_info)
  trace.frame.add_eqn(eqn)
  return out_tracer


### traceables

def neg(x: Array) -> Array:
  r"""Elementwise negation: :math:`-x`."""
  return neg_p.bind(x)

def sign(x: Array) -> Array:
  r"""Elementwise sign.

  For floating-point inputs, returns
  :math:`\mathrm{sign}(x) = \begin{cases}
  -1 & x < 0\\
  -0 & x = -0\\
  \mathit{NaN} & x = \mathit{NaN}\\
  +0 & x = +0\\
  1 & x > 0
  \end{cases}`

  For signed integer inputs, returns
  :math:`\mathrm{sign}(x) = \begin{cases}
  -1 & x < 0\\
  0 & x = 0\\
  1 & x > 0
  \end{cases}`

  For complex inputs, returns the complex phase, i.e.
  :math:`\mathrm{sign}(x) = \frac{x}{|x|}`.
  """
  return sign_p.bind(x)

def nextafter(x1: Array, x2: Array) -> Array:
  r"""Returns the next representable value after `x1` in the direction of `x2`.

  Note that in some environments flush-denormal-to-zero semantics is used.
  This means that, around zero, this function returns strictly non-zero
  values which appear as zero in any operations. Consider this example::

    >>> jnp.nextafter(0, 1)  # denormal numbers are representable
    DeviceArray(1.e-45, dtype=float32, weak_type=True)
    >>> jnp.nextafter(0, 1) * 1  # but are flushed to zero
    DeviceArray(0., dtype=float32, weak_type=True)

  For the smallest usable (i.e. normal) float, use ``tiny`` of ``jnp.finfo``.
  """
  return nextafter_p.bind(x1, x2)

def floor(x: Array) -> Array:
  r"""Elementwise floor: :math:`\left\lfloor x \right\rfloor`."""
  return floor_p.bind(x)

def ceil(x: Array) -> Array:
  r"""Elementwise ceiling: :math:`\left\lceil x \right\rceil`."""
  return ceil_p.bind(x)

class RoundingMethod(enum.IntEnum):
  AWAY_FROM_ZERO = 0
  TO_NEAREST_EVEN = 1

def round(x: Array,
          rounding_method: RoundingMethod = RoundingMethod.AWAY_FROM_ZERO
          ) -> Array:
  r"""Elementwise round.

  Rounds values to the nearest integer.

  Args:
    x: an array or scalar value to round.
    rounding_method: the method to use when rounding halfway values
      (e.g., `0.5`). See ``lax.RoundingMethod`` for the list of possible
      values.

  Returns:
    An array containing the elementwise rounding of x.
  """
  rounding_method = RoundingMethod(rounding_method)
  return round_p.bind(x, rounding_method=rounding_method)

def is_finite(x: Array) -> Array:
  r"""Elementwise :math:`\mathrm{isfinite}`.

  For each element x returns `True` if and only if x is not :math:`\pm\infty` or
  :math:`\mathit{NaN}`.
  """
  return is_finite_p.bind(x)

def exp(x: Array) -> Array:
  r"""Elementwise exponential: :math:`e^x`."""
  return exp_p.bind(x)

def expm1(x: Array) -> Array:
  r"""Elementwise :math:`e^{x} - 1`."""
  return expm1_p.bind(x)

def log(x: Array) -> Array:
  r"""Elementwise natural logarithm: :math:`\mathrm{log}(x)`."""
  return log_p.bind(x)

def log1p(x: Array) -> Array:
  r"""Elementwise :math:`\mathrm{log}(1 + x)`."""
  return log1p_p.bind(x)

def tanh(x: Array) -> Array:
  r"""Elementwise hyperbolic tangent: :math:`\mathrm{tanh}(x)`."""
  return tanh_p.bind(x)

def logistic(x: Array) -> Array:
  r"""Elementwise logistic (sigmoid) function: :math:`\frac{1}{1 + e^{-x}}`."""
  return logistic_p.bind(x)

def sin(x: Array) -> Array:
  r"""Elementwise sine: :math:`\mathrm{sin}(x)`."""
  return sin_p.bind(x)

def cos(x: Array) -> Array:
  r"""Elementwise cosine: :math:`\mathrm{cos}(x)`."""
  return cos_p.bind(x)

def atan2(x: Array, y: Array) -> Array:
  r"""Elementwise arc tangent of two variables:
    :math:`\mathrm{atan}({x \over y})`."""
  return atan2_p.bind(x, y)

def betainc(a: Array, b: Array, x: Array) -> Array:
  r"""Elementwise regularized incomplete beta integral."""
  return regularized_incomplete_beta_p.bind(a, b, x)

def lgamma(x: Array) -> Array:
  r"""Elementwise log gamma: :math:`\mathrm{log}(\Gamma(x))`."""
  return lgamma_p.bind(x)

def digamma(x: Array) -> Array:
  r"""Elementwise digamma: :math:`\psi(x)`."""
  return digamma_p.bind(x)

def igamma(a: Array, x: Array) -> Array:
  r"""Elementwise regularized incomplete gamma function."""
  return igamma_p.bind(a, x)

def igammac(a: Array, x: Array) -> Array:
  r"""Elementwise complementary regularized incomplete gamma function."""
  return igammac_p.bind(a, x)

def igamma_grad_a(a: Array, x: Array) -> Array:
  r"""Elementwise derivative of the regularized incomplete gamma function."""
  return igamma_grad_a_p.bind(a, x)

def random_gamma_grad(a: Array, x: Array) -> Array:
  r"""Elementwise derivative of samples from `Gamma(a, 1)`."""
  return random_gamma_grad_p.bind(a, x)

def bessel_i0e(x: Array) -> Array:
  r"""Exponentially scaled modified Bessel function of order 0:
  :math:`\mathrm{i0e}(x) = e^{-|x|} \mathrm{i0}(x)`
  """
  return bessel_i0e_p.bind(x)

def bessel_i1e(x: Array) -> Array:
  r"""Exponentially scaled modified Bessel function of order 1:
  :math:`\mathrm{i1e}(x) = e^{-|x|} \mathrm{i1}(x)`
  """
  return bessel_i1e_p.bind(x)

def erf(x: Array) -> Array:
  r"""Elementwise error function: :math:`\mathrm{erf}(x)`."""
  return erf_p.bind(x)

def erfc(x: Array) -> Array:
  r"""Elementwise complementary error function:
    :math:`\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)`."""
  return erfc_p.bind(x)

def erf_inv(x: Array) -> Array:
  r"""Elementwise inverse error function: :math:`\mathrm{erf}^{-1}(x)`."""
  return erf_inv_p.bind(x)

def real(x: Array) -> Array:
  r"""Elementwise extract real part: :math:`\mathrm{Re}(x)`.

  Returns the real part of a complex number.
  """
  return real_p.bind(x)

def imag(x: Array) -> Array:
  r"""Elementwise extract imaginary part: :math:`\mathrm{Im}(x)`.

  Returns the imaginary part of a complex number.
  """
  return imag_p.bind(x)

def complex(x: Array, y: Array) -> Array:
  r"""Elementwise make complex number: :math:`x + jy`.

  Builds a complex number from real and imaginary parts.
  """
  return complex_p.bind(x, y)

def conj(x: Array) -> Array:
  r"""Elementwise complex conjugate function: :math:`\overline{x}`."""
  return conj_p.bind(x, input_dtype=_dtype(x))

def abs(x: Array) -> Array:
  r"""Elementwise absolute value: :math:`|x|`."""
  return abs_p.bind(x)

def pow(x: Array, y: Array) -> Array:
  r"""Elementwise power: :math:`x^y`."""
  return pow_p.bind(x, y)

def integer_pow(x: Array, y: int) -> Array:
  r"""Elementwise power: :math:`x^y`, where :math:`y` is a fixed integer."""
  return integer_pow_p.bind(x, y=y)

def sqrt(x: Array) -> Array:
  r"""Elementwise square root: :math:`\sqrt{x}`."""
  return sqrt_p.bind(x)

def rsqrt(x: Array) -> Array:
  r"""Elementwise reciprocal square root:  :math:`1 \over \sqrt{x}`."""
  return rsqrt_p.bind(x)

def cbrt(x: Array) -> Array:
  r"""Elementwise cube root: :math:`\sqrt[3]{x}`."""
  return cbrt_p.bind(x)

def bitwise_not(x: Array) -> Array:
  r"""Elementwise NOT: :math:`\neg x`."""
  return not_p.bind(x)

def bitwise_and(x: Array, y: Array) -> Array:
  r"""Elementwise AND: :math:`x \wedge y`."""
  return and_p.bind(x, y)

def bitwise_or(x: Array, y: Array) -> Array:
  r"""Elementwise OR: :math:`x \vee y`."""
  return or_p.bind(x, y)

def bitwise_xor(x: Array, y: Array) -> Array:
  r"""Elementwise exclusive OR: :math:`x \oplus y`."""
  return xor_p.bind(x, y)

def population_count(x: Array) -> Array:
  r"""Elementwise popcount, count the number of set bits in each element."""
  return population_count_p.bind(x)

def clz(x: Array) -> Array:
  r"""Elementwise count-leading-zeros."""
  return clz_p.bind(x)

def add(x: Array, y: Array) -> Array:
  r"""Elementwise addition: :math:`x + y`."""
  return add_p.bind(x, y)

def sub(x: Array, y: Array) -> Array:
  r"""Elementwise subtraction: :math:`x - y`."""
  return sub_p.bind(x, y)

def mul(x: Array, y: Array) -> Array:
  r"""Elementwise multiplication: :math:`x \times y`."""
  return mul_p.bind(x, y)

def div(x: Array, y: Array) -> Array:
  r"""Elementwise division: :math:`x \over y`.

  Integer division overflow
  (division by zero or signed division of INT_SMIN with -1)
  produces an implementation defined value.
  """
  return div_p.bind(x, y)

def rem(x: Array, y: Array) -> Array:
  r"""Elementwise remainder: :math:`x \bmod y`.

  The sign of the result is taken from the dividend,
  and the absolute value of the result is always
  less than the divisor's absolute value.

  Integer division overflow
  (remainder by zero or remainder of INT_SMIN with -1)
  produces an implementation defined value.
  """
  return rem_p.bind(x, y)

def max(x: Array, y: Array) -> Array:
  r"""Elementwise maximum: :math:`\mathrm{max}(x, y)`

  For complex numbers, uses a lexicographic comparison on the
  `(real, imaginary)` pairs."""
  return max_p.bind(x, y)

def min(x: Array, y: Array) -> Array:
  r"""Elementwise minimum:  :math:`\mathrm{min}(x, y)`

  For complex numbers, uses a lexicographic comparison on the
  `(real, imaginary)` pairs."""
  return min_p.bind(x, y)

def shift_left(x: Array, y: Array) -> Array:
  r"""Elementwise left shift: :math:`x \ll y`."""
  return shift_left_p.bind(x, y)

def shift_right_arithmetic(x: Array, y: Array) -> Array:
  r"""Elementwise arithmetic right shift: :math:`x \gg y`."""
  return shift_right_arithmetic_p.bind(x, y)

def shift_right_logical(x: Array, y: Array) -> Array:
  r"""Elementwise logical right shift: :math:`x \gg y`."""
  return shift_right_logical_p.bind(x, y)

def eq(x: Array, y: Array) -> Array:
  r"""Elementwise equals: :math:`x = y`."""
  return eq_p.bind(x, y)

def ne(x: Array, y: Array) -> Array:
  r"""Elementwise not-equals: :math:`x \neq y`."""
  return ne_p.bind(x, y)

def ge(x: Array, y: Array) -> Array:
  r"""Elementwise greater-than-or-equals: :math:`x \geq y`."""
  return ge_p.bind(x, y)

def gt(x: Array, y: Array) -> Array:
  r"""Elementwise greater-than: :math:`x > y`."""
  return gt_p.bind(x, y)

def le(x: Array, y: Array) -> Array:
  r"""Elementwise less-than-or-equals: :math:`x \leq y`."""
  return le_p.bind(x, y)

def lt(x: Array, y: Array) -> Array:
  r"""Elementwise less-than: :math:`x < y`."""
  return lt_p.bind(x, y)

def convert_element_type(operand: Array, new_dtype: DType) -> Array:
  """Elementwise cast.

  Wraps XLA's `ConvertElementType
  <https://www.tensorflow.org/xla/operation_semantics#convertelementtype>`_
  operator, which performs an elementwise conversion from one type to another.
  Similar to a C++ `static_cast`.

  Args:
    operand: an array or scalar value to be cast
    new_dtype: a NumPy dtype representing the target type.

  Returns:
    An array with the same shape as `operand`, cast elementwise to `new_dtype`.
  """
  if hasattr(operand, '__jax_array__'):
    operand = operand.__jax_array__()
  return _convert_element_type(operand, new_dtype, weak_type=False)

def _convert_element_type(operand: Array, new_dtype: Optional[DType] = None,
                          weak_type: bool = False):
  from jax.experimental import array

  # Don't canonicalize old_dtype because x64 context might cause
  # un-canonicalized operands to be passed in.
  old_dtype = dtypes.dtype(operand, canonicalize=False)
  old_weak_type = dtypes.is_weakly_typed(operand)

  if new_dtype is None:
    new_dtype = old_dtype
  else:
    new_dtype = np.dtype(new_dtype)
  new_dtype = dtypes.dtype(new_dtype, canonicalize=True)
  new_weak_type = bool(weak_type)

  if (dtypes.issubdtype(old_dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    msg = "Casting complex values to real discards the imaginary part"
    warnings.warn(msg, np.ComplexWarning, stacklevel=2)

  # Python has big integers, but convert_element_type(2 ** 100, np.float32) need
  # not be an error since the target dtype fits the value. Handle this case by
  # converting to a NumPy array before calling bind. Without this step, we'd
  # first canonicalize the input to a value of dtype int32 or int64, leading to
  # an overflow error.
  if type(operand) is int:
    operand = np.asarray(operand, new_dtype)
    old_weak_type = False

  if ((old_dtype, old_weak_type) == (new_dtype, new_weak_type)
      and isinstance(operand, (core.Tracer, device_array.DeviceArray, array.Array))):
    return operand
  else:
    return convert_element_type_p.bind(operand, new_dtype=new_dtype,
                                       weak_type=new_weak_type)

def bitcast_convert_type(operand: Array, new_dtype: DType) -> Array:
  """Elementwise bitcast.

  Wraps XLA's `BitcastConvertType
  <https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype>`_
  operator, which performs a bit cast from one type to another. The bitwidth
  of the source and destination types must match.

  Args:
    operand: an array or scalar value to be cast
    new_dtype: the new type. Should be a NumPy type.

  Returns:
    An array with the same shape as `operand`, bitcast elementwise to
    `new_dtype`.
  """
  new_dtype = dtypes.canonicalize_dtype(new_dtype)
  return bitcast_convert_type_p.bind(operand, new_dtype=new_dtype)

def clamp(min: Array, x: Array, max: Array) -> Array:
  r"""Elementwise clamp.

  Returns :math:`\mathrm{clamp}(x) = \begin{cases}
  \mathit{min} & \text{if } x < \mathit{min},\\
  \mathit{max} & \text{if } x > \mathit{max},\\
  x & \text{otherwise}
  \end{cases}`.
  """
  return clamp_p.bind(min, x, max)

def concatenate(operands: Sequence[Array], dimension: int) -> Array:
  """Concatenates a sequence of arrays along `dimension`.

  Wraps XLA's `Concatenate
  <https://www.tensorflow.org/xla/operation_semantics#concatenate>`_
  operator.

  Args:
    operands: a sequence of arrays to concatenate. The arrays must have equal
      shapes, except in the `dimension` axis.
    dimension: the dimension along which to concatenate the arrays.

  Returns:
    An array containing the concatenation.
  """
  if not operands:
    msg = "concatenate expects at least one operand, got 0."
    raise TypeError(msg)
  if len({operand.ndim for operand in operands}) != 1:
    msg = "Cannot concatenate arrays with different numbers of dimensions: got {}."
    raise TypeError(msg.format(", ".join(str(o.shape) for o in operands)))
  if not 0 <= dimension < operands[0].ndim:
    msg = "concatenate dimension out of bounds: dimension {} for shapes {}."
    raise TypeError(msg.format(dimension, ", ".join([str(o.shape) for o in operands])))
  shapes = [operand.shape[:dimension] + operand.shape[dimension+1:]
            for operand in operands]
  if not shapes[:-1] == shapes[1:]:
    msg = ("Cannot concatenate arrays with shapes that differ in dimensions "
           "other than the one being concatenated: concatenating along "
           "dimension {} for shapes {}.")
    shapes = [operand.shape for operand in operands]
    raise TypeError(msg.format(dimension, ", ".join(map(str, shapes))))

  concat_size = sum(o.shape[dimension] for o in operands)
  ex_shape = operands[0].shape
  shape = ex_shape[:dimension] + (concat_size,) + ex_shape[dimension+1:]
  dynamic_shape = [d for d in shape if isinstance(d, core.Tracer)]
  static_shape = [None if isinstance(d, core.Tracer) else d for d in shape]
  return concatenate_p.bind(*operands, *dynamic_shape,
                            dimension=dimension, shape=tuple(static_shape))

class _enum_descriptor:
  def __init__(self, val):
    self.val = val
  def __get__(self, _, owner):
    return owner(self.val)


class Precision(xla_client.PrecisionConfig.Precision):  # type: ignore
  """Precision enum for lax functions

  The `precision` argument to JAX functions generally controls the tradeoff
  between speed and accuracy for array computations on accelerator backends,
  (i.e. TPU and GPU). Members are:

  DEFAULT:
    Fastest mode, but least accurate. Performs computations in bfloat16.
    Aliases: ``'default'``, ``'fastest'``, ``'bfloat16'``.
  HIGH:
    Slower but more accurate. Performs float32 computations in 3 bfloat16
    passes, or using tensorfloat32 where available. Aliases: ``'high'``,
    ``'bfloat16_3x'``, ``'tensorfloat32'``.
  HIGHEST:
    Slowest but most accurate. Performs computations in float32 or float64
    as applicable. Aliases: ``'highest'``, ``'float32'``.
  """
  # Wrap enum values with this class.
  DEFAULT = _enum_descriptor('default')
  HIGH = _enum_descriptor('high')
  HIGHEST = _enum_descriptor('highest')

  _strings = {
      'highest':       xla_client.PrecisionConfig.Precision.HIGHEST,
      'float32':       xla_client.PrecisionConfig.Precision.HIGHEST,
      'high':          xla_client.PrecisionConfig.Precision.HIGH,
      'bfloat16_3x':   xla_client.PrecisionConfig.Precision.HIGH,
      'tensorfloat32': xla_client.PrecisionConfig.Precision.HIGH,
      'default':       xla_client.PrecisionConfig.Precision.DEFAULT,
      'bfloat16':      xla_client.PrecisionConfig.Precision.DEFAULT,
      'fastest':       xla_client.PrecisionConfig.Precision.DEFAULT,
      None:            xla_client.PrecisionConfig.Precision.DEFAULT,
  }
  def __init__(self, arg0):
    arg0 = self._strings.get(arg0, arg0)
    super().__init__(arg0)

  def __str__(self) -> str:
    return self.name


PrecisionType = Precision
PrecisionLike = Union[None, str, PrecisionType, Tuple[str, str],
                      Tuple[PrecisionType, PrecisionType]]

def dot(lhs: Array, rhs: Array, precision: PrecisionLike = None,
        preferred_element_type: Optional[DType] = None) -> Array:
  """Vector/vector, matrix/vector, and matrix/matrix multiplication.

  Wraps XLA's `Dot
  <https://www.tensorflow.org/xla/operation_semantics#dot>`_
  operator.

  For more general contraction, see the `dot_general` operator.

  Args:
    lhs: an array of dimension 1 or 2.
    rhs: an array of dimension 1 or 2.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    An array containing the product.
  """
  if 1 <= lhs.ndim <= 2 and 1 <= rhs.ndim <= 2 and core.symbolic_equal_dim(lhs.shape[-1], rhs.shape[0]):
    return dot_general(lhs, rhs, (((lhs.ndim - 1,), (0,)), ((), ())),
                       precision=precision,
                       preferred_element_type=preferred_element_type)
  else:
    raise TypeError("Incompatible shapes for dot: got {} and {}.".format(
        lhs.shape, rhs.shape))


DotDimensionNumbers = Tuple[Tuple[Sequence[int], Sequence[int]],
                            Tuple[Sequence[int], Sequence[int]]]

def dot_general(lhs: Array, rhs: Array, dimension_numbers: DotDimensionNumbers,
                precision: PrecisionLike = None,
                preferred_element_type: Optional[DType] = None) -> Array:
  """More general contraction operator.

  Wraps XLA's `DotGeneral
  <https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`_
  operator.

  Args:
    lhs: an array
    rhs: an array
    dimension_numbers: a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims),
      (lhs_batch_dims, rhs_batch_dims))`
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

  Returns:
    An array containing the result.
  """
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  preferred_element_type = (
      None if preferred_element_type is None else
      dtypes.canonicalize_dtype(np.dtype(preferred_element_type)))
  return dot_general_p.bind(lhs, rhs,
                            dimension_numbers=(cdims, bdims),
                            precision=canonicalize_precision(precision),
                            preferred_element_type=preferred_element_type)

def broadcast(operand: Array, sizes: Sequence[int]) -> Array:
  """Broadcasts an array, adding new leading dimensions

  Args:
    operand: an array
    sizes: a sequence of integers, giving the sizes of new leading dimensions
      to add to the front of the array.

  Returns:
    An array containing the result.

  See Also:
    jax.lax.broadcast_in_dim : add new dimensions at any location in the array shape.
  """
  dims = tuple(range(len(sizes), len(sizes) + np.ndim(operand)))
  return broadcast_in_dim(operand, tuple(sizes) + np.shape(operand), dims)

def broadcast_in_dim(operand: Array, shape: Shape,
                     broadcast_dimensions: Sequence[int]) -> Array:
  """Wraps XLA's `BroadcastInDim
  <https://www.tensorflow.org/xla/operation_semantics#broadcastindim>`_
  operator.

  Args:
    operand: an array
    shape: the shape of the target array
    broadcast_dimensions: to which dimension in the target shape each dimension
      of the operand shape corresponds to

  Returns:
    An array containing the result.

  See Also:
    jax.lax.broadcast : simpler interface to add new leading dimensions.
  """
  from jax.experimental import array

  if (np.ndim(operand) == len(shape) and not len(broadcast_dimensions)
      and isinstance(operand, (device_array.DeviceArray, core.Tracer, array.Array))):
    return operand
  if config.jax_dynamic_shapes:
    # We must gate this behavior under a flag because otherwise the errors
    # raised are different (and have worse source provenance information).
    dyn_shape, static_shape = _extract_tracers_dyn_shape(shape)
  else:
    dyn_shape, static_shape = [], shape  # type: ignore
  return broadcast_in_dim_p.bind(
      operand, *dyn_shape, shape=tuple(static_shape),
      broadcast_dimensions=tuple(broadcast_dimensions))

def broadcast_to_rank(x: Array, rank: int) -> Array:
  """Adds leading dimensions of ``1`` to give ``x`` rank ``rank``."""
  return broadcast(x, (1,) * (rank - x.ndim))

def reshape(operand: Array, new_sizes: Shape,
            dimensions: Optional[Sequence[int]] = None) -> Array:
  """Wraps XLA's `Reshape
  <https://www.tensorflow.org/xla/operation_semantics#reshape>`_
  operator.

  For inserting/removing dimensions of size 1, prefer using ``lax.squeeze`` /
  ``lax.expand_dims``. These preserve information about axis identity that may
  be useful for advanced transformation rules.

  Args:
    operand: array to be reshaped.
    new_sizes: sequence of integers specifying the resulting shape. The size
      of the final array must match the size of the input.
    dimensions: optional sequence of integers specifying the permutation order of
      the input shape. If specified, the length must match ``operand.shape``.

  Returns:
    out: reshaped array.

  Examples:
    Simple reshaping from one to two dimensions:

    >>> x = jnp.arange(6)
    >>> y = reshape(x, (2, 3))
    >>> y
    DeviceArray([[0, 1, 2],
                 [3, 4, 5]], dtype=int32)

    Reshaping back to one dimension:

    >>> reshape(y, (6,))
    DeviceArray([0, 1, 2, 3, 4, 5], dtype=int32)

    Reshaping to one dimension with permutation of dimensions:

    >>> reshape(y, (6,), (1, 0))
    DeviceArray([0, 3, 1, 4, 2, 5], dtype=int32)
  """
  from jax.experimental import array

  new_sizes = canonicalize_shape(new_sizes)  # TODO
  new_sizes = tuple(new_sizes)
  same_shape = core.symbolic_equal_shape(np.shape(operand), new_sizes)
  if dimensions is None:
    same_dims = True
    dims = None
  else:
    dims = api_util._ensure_index_tuple(dimensions)
    same_dims = tuple(dims) == tuple(range(np.ndim(operand)))
  if (np.shape(operand) and same_shape and same_dims
      and isinstance(operand, (core.Tracer, device_array.DeviceArray, array.Array))):
    return operand
  else:
    dyn_shape, static_new_sizes = _extract_tracers_dyn_shape(new_sizes)

    return reshape_p.bind(
      operand, *dyn_shape, new_sizes=tuple(static_new_sizes),
      dimensions=None if dims is None or same_dims else dims)

def pad(operand: Array, padding_value: Array,
        padding_config: Sequence[Tuple[int, int, int]]) -> Array:
  """Applies low, high, and/or interior padding to an array.

  Wraps XLA's `Pad
  <https://www.tensorflow.org/xla/operation_semantics#pad>`_
  operator.

  Args:
    operand: an array to be padded.
    padding_value: the value to be inserted as padding. Must have the same dtype
      as ``operand``.
    padding_config: a sequence of ``(low, high, interior)`` tuples of integers,
      giving the amount of low, high, and interior (dilation) padding to insert
      in each dimension.

  Returns:
    The ``operand`` array with padding value ``padding_value`` inserted in each
    dimension according to the ``padding_config``.
  """
  return pad_p.bind(operand, padding_value, padding_config=tuple(padding_config))

def rev(operand: Array, dimensions: Sequence[int]) -> Array:
  """Wraps XLA's `Rev
  <https://www.tensorflow.org/xla/operation_semantics#rev_reverse>`_
  operator.
  """
  return rev_p.bind(operand, dimensions=tuple(dimensions))

def select(pred: Array, on_true: Array, on_false: Array) -> Array:
  """Wraps XLA's `Select
  <https://www.tensorflow.org/xla/operation_semantics#select>`_
  operator.
  """
  # Caution! The select_n_p primitive has the *opposite* order of arguments to
  # select(). This is because it implements `select_n`.
  return select_n_p.bind(pred, on_false, on_true)

def select_n(which: Array, *cases: Array) -> Array:
  """Selects array values from multiple cases.

  Generalizes XLA's `Select
  <https://www.tensorflow.org/xla/operation_semantics#select>`_
  operator. Unlike XLA's version, the operator is variadic and can select
  from many cases using an integer `pred`.

  Args:
    which: determines which case should be returned. Must be an array containing
      either a boolean or integer values. May either be a scalar or have
      shape matching ``cases``. For each array element, the value of ``which``
      determines which of ``cases`` is taken. ``which`` must be in the range
      ``[0 .. len(cases))``; for values outside that range the behavior is
      implementation-defined.
    *cases: a non-empty list of array cases. All must have equal dtypes and
      equal shapes.
  Returns:
    An array with shape and dtype equal to the cases, whose values are chosen
    according to ``which``.
  """
  if len(cases) == 0:
    raise ValueError("select_n() must have at least one case")
  return select_n_p.bind(which, *cases)


def transpose(operand: Array, permutation: Sequence[int]) -> Array:
  """Wraps XLA's `Transpose
  <https://www.tensorflow.org/xla/operation_semantics#transpose>`_
  operator.
  """
  permutation = tuple(operator.index(d) for d in permutation)
  if (permutation == tuple(range(np.ndim(operand)))
      and isinstance(operand, (core.Tracer, device_array.DeviceArray))):
    return operand
  else:
    return transpose_p.bind(operand, permutation=permutation)

def argmin(operand: Array, axis: int,
           index_dtype: DType) -> Tuple[Array, Array]:
  """Computes the index of the minimum element along ``axis``."""
  return argmin_p.bind(operand, axes=(axis,),
                       index_dtype=dtypes.canonicalize_dtype(index_dtype))

def argmax(operand: Array, axis: int,
           index_dtype: DType) -> Tuple[Array, Array]:
  """Computes the index of the maximum element along ``axis``."""
  return argmax_p.bind(operand, axes=(axis,),
                       index_dtype=dtypes.canonicalize_dtype(index_dtype))

def reduce(operands: Any,
           init_values: Any,
           computation: Callable[[Any, Any], Any],
           dimensions: Sequence[int]) -> Any:
  """Wraps XLA's `Reduce
  <https://www.tensorflow.org/xla/operation_semantics#reduce>`_
  operator.

  ``init_values`` and ``computation`` together must form a `monoid
  <https://en.wikipedia.org/wiki/Monoid>`_
  for correctness. That is ``init_values`` must be an identity of
  ``computation``, and ``computation`` must be associative. XLA may exploit both
  of these properties during code generation; if either is violated the result
  is undefined.
  """
  flat_operands, operand_tree = tree_util.tree_flatten(operands)
  flat_init_values, init_value_tree = tree_util.tree_flatten(init_values)
  if operand_tree != init_value_tree:
    raise ValueError('Operands must have the same tree structure as init_values:'
                     f' {operand_tree} vs. {init_value_tree}')
  if len(flat_operands) != len(flat_init_values):
    raise ValueError('Must have same total number of operands as init_values: '
                     f' {len(flat_operands)} vs. {len(flat_init_values)}')
  monoid_reducer = _get_monoid_reducer(computation, flat_init_values)
  if monoid_reducer:
    # monoid reducers bypass the weak_type_rule, so we set it explicitly.
    weak_type = dtypes.is_weakly_typed(*flat_operands) and dtypes.is_weakly_typed(*flat_init_values)
    return _convert_element_type(monoid_reducer(*flat_operands, dimensions),
                                 weak_type=weak_type)
  else:
    flat_init_avals = safe_map(_abstractify, flat_init_values)
    jaxpr, consts, out_tree = _variadic_reduction_jaxpr(
        computation, tuple(flat_init_avals), init_value_tree)
    out = reduce_p.bind(*(flat_operands + flat_init_values), computation=computation,
                         jaxpr=jaxpr, consts=consts, dimensions=tuple(dimensions))
    return tree_util.tree_unflatten(out_tree, out)

@cache()
def _reduction_jaxpr(computation, aval):
  @lu.wrap_init
  def comp(x, y):
    result = computation(x, y)
    if not (isinstance(result, core.Tracer) or core.valid_jaxtype(result)):
      raise ValueError(
          f"Invalid return type from reduction function: {type(result)}\n"
          f"Reduction functions should only return an array.\n"
          f"Full return value: {result}")
    return (result,)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(comp, (aval, aval))
  if any(isinstance(c, core.Tracer) for c in consts):
    raise NotImplementedError(
        "Reduction computations can't close over Tracers. Please open an issue "
        "at https://github.com/google/jax.")
  return jaxpr, tuple(consts)

@cache()
def _variadic_reduction_jaxpr(computation, flat_avals, aval_tree):
  avals = tree_util.tree_unflatten(aval_tree, flat_avals)
  flat_in_avals, in_tree = tree_util.tree_flatten((avals, avals))
  comp = lu.wrap_init(computation)
  flat_comp, out_tree = api_util.flatten_fun_nokwargs(comp, in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_comp, tuple(flat_in_avals))
  if any(isinstance(c, core.Tracer) for c in consts):
    raise NotImplementedError(
        "Reduction computations can't close over Tracers. Please open an issue "
        "at https://github.com/google/jax.")
  return jaxpr, tuple(consts), out_tree()

def _get_monoid_reducer(monoid_op: Callable,
                        xs: Sequence[Array]) -> Optional[Callable]:
  if len(xs) != 1:
    return None
  x, = xs
  aval = core.get_aval(x)
  dtype = _dtype(x)
  if (type(aval) is ConcreteArray) and aval.shape == ():
    # allow bitwise reductions for boolean and integer types
    _is_intlike = dtype == np.bool_ or dtypes.issubdtype(dtype, np.integer)
    if monoid_op is add:
      return _reduce_sum if np.equal(aval.val, 0) else None
    elif monoid_op is mul:
      return _reduce_prod if np.equal(aval.val, 1) else None
    elif monoid_op is bitwise_or and _is_intlike:
      return _reduce_or if np.equal(aval.val, _get_bitwise_or_identity(dtype)) else None
    elif monoid_op is bitwise_and and _is_intlike:
      return _reduce_and if np.equal(aval.val, _get_bitwise_and_identity(dtype)) else None
    elif monoid_op is bitwise_xor and _is_intlike:
      return _reduce_xor if np.equal(aval.val, _get_bitwise_or_identity(dtype)) else None
    elif monoid_op is max:
      return _reduce_max if np.equal(aval.val, _get_max_identity(dtype)) else None
    elif monoid_op is min:
      return _reduce_min if np.equal(aval.val, _get_min_identity(dtype)) else None
  return None

def _get_bitwise_and_identity(dtype: DType) -> Array:
  return np.array(-1, dtype)

def _get_bitwise_or_identity(dtype: DType) -> Array:
  return np.array(0, dtype)

def _get_max_identity(dtype: DType) -> Array:
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(-np.inf, dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).min, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(False, np.bool_)

def _get_min_identity(dtype: DType) -> Array:
  if dtypes.issubdtype(dtype, np.inexact):
    return np.array(np.inf, dtype)
  elif dtypes.issubdtype(dtype, np.integer):
    return np.array(dtypes.iinfo(dtype).max, dtype)
  elif dtypes.issubdtype(dtype, np.bool_):
    return np.array(True, np.bool_)

def _reduce_sum(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_sum_p.bind(operand, axes=tuple(axes))

def _reduce_prod(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_prod_p.bind(operand, axes=tuple(axes))

def _reduce_max(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_max_p.bind(operand, axes=tuple(axes))

def _reduce_min(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_min_p.bind(operand, axes=tuple(axes))

def _reduce_or(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_or_p.bind(operand, axes=tuple(axes))

def _reduce_and(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_and_p.bind(operand, axes=tuple(axes))

def _reduce_xor(operand: Array, axes: Sequence[int]) -> Array:
  return reduce_xor_p.bind(operand, axes=tuple(axes))

def sort(operand: Union[Array, Sequence[Array]], dimension: int = -1,
         is_stable: bool = True, num_keys: int = 1) -> Union[Array, Tuple[Array, ...]]:
  """Wraps XLA's `Sort
  <https://www.tensorflow.org/xla/operation_semantics#sort>`_ operator.

  For floating point inputs, -0.0 and 0.0 are treated as equivalent, and NaN values
  are sorted to the end of the array. For complex inputs, the sort order is
  lexicographic over the real and imaginary parts, with the real part primary.

  Args:
    operand : Array or sequence of arrays
    dimension : integer dimension along which to sort. Default: -1.
    is_stable : boolean specifying whether to use a stable sort. Default: True.
    num_keys : number of operands to treat as sort keys. Default: 1.
      For num_keys > 1, the sort order will be determined lexicographically using
      the first `num_keys` arrays, with the first key being primary.
      The remaining operands will be returned with the same permutation.

  Returns:
    operand : sorted version of the input or inputs.
  """
  if isinstance(operand, Sequence):
    if len(operand) == 0:
      raise TypeError("Sort requires at least one operand")
    if not (1 <= num_keys <= len(operand)):
      raise ValueError(f"num_keys={num_keys} must be between 1 and len(operand)={len(operand)}")
    dimension = canonicalize_axis(dimension, len(operand[0].shape))
    return tuple(sort_p.bind(*operand, dimension=dimension,
                             is_stable=is_stable,
                             num_keys=num_keys))
  else:
    if num_keys != 1:
      raise ValueError(f"num_keys={num_keys} must equal 1 for a single operand.")
    dimension = canonicalize_axis(dimension, len(operand.shape))
    return sort_p.bind(operand, dimension=dimension, is_stable=is_stable, num_keys=1)[0]

def sort_key_val(keys: Array, values: Array, dimension: int = -1,
                 is_stable: bool = True) -> Tuple[Array, Array]:
  """Sorts ``keys`` along ``dimension`` and applies the same permutation to ``values``."""
  dimension = canonicalize_axis(dimension, len(keys.shape))
  k, v = sort_p.bind(keys, values, dimension=dimension, is_stable=is_stable, num_keys=1)
  return k, v

def top_k(operand: Array, k: int) -> Tuple[Array, Array]:
  """Returns top ``k`` values and their indices along the last axis of ``operand``."""
  k = int(k)
  if k < 0:
    raise ValueError(f"k argument to top_k must be nonnegative, got {k}")
  return top_k_p.bind(operand, k=k)

def tie_in(x: Array, y: Array) -> Array:
  """Deprecated. Ignores ``x`` and returns ``y``."""
  return y

def full(shape: Shape, fill_value: Array, dtype: Optional[DType] = None) -> Array:
  """Returns an array of `shape` filled with `fill_value`.

  Args:
    shape: sequence of integers, describing the shape of the output array.
    fill_value: the value to fill the new array with.
    dtype: the type of the output array, or `None`. If not `None`, `fill_value`
      will be cast to `dtype`.
  """
  shape = canonicalize_shape(shape)
  if np.shape(fill_value):
    msg = "full must be called with scalar fill_value, got fill_value.shape {}."
    raise TypeError(msg.format(np.shape(fill_value)))
  weak_type = dtype is None and dtypes.is_weakly_typed(fill_value)
  dtype = dtypes.canonicalize_dtype(dtype or _dtype(fill_value))
  fill_value = _convert_element_type(fill_value, dtype, weak_type)
  return broadcast(fill_value, shape)

def zeros_like_shaped_array(aval: Array) -> Array:
  assert isinstance(aval, ShapedArray)
  if aval.dtype == dtypes.float0:
    scalar_zero = np.zeros((), dtype=aval.dtype)
  else:
    scalar_zero = _convert_element_type(0, aval.dtype, aval.weak_type)
  return broadcast(scalar_zero, aval.shape)

ad_util.aval_zeros_likers[ShapedArray] = zeros_like_shaped_array

def iota(dtype: DType, size: int) -> Array:
  """Wraps XLA's `Iota
  <https://www.tensorflow.org/xla/operation_semantics#iota>`_
  operator.
  """
  return broadcasted_iota(dtype, (size,), 0)

def broadcasted_iota(dtype: DType, shape: Shape, dimension: int) -> Array:
  """Convenience wrapper around ``iota``."""
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = canonicalize_shape(shape)
  dynamic_shape = [d for d in shape if isinstance(d, core.Tracer)]
  static_shape = [None if isinstance(d, core.Tracer) else d for d in shape]
  dimension = core.concrete_or_error(
      int, dimension, "dimension argument of lax.broadcasted_iota")
  return iota_p.bind(*dynamic_shape, dtype=dtype, shape=tuple(static_shape),
                     dimension=dimension)

def _eye(dtype: DType, shape: Shape, offset: int) -> Array:
  """Like numpy.eye, create a 2D array with ones on a diagonal."""
  offset = int(offset)
  dtype = dtypes.canonicalize_dtype(dtype)
  bool_eye = eq(add(broadcasted_iota(np.int32, shape, 0), np.int32(offset)),
                broadcasted_iota(np.int32, shape, 1))
  return convert_element_type_p.bind(bool_eye, new_dtype=dtype, weak_type=False)

def _delta(dtype: DType, shape: Shape, axes: Sequence[int]) -> Array:
  """This utility function exists for creating Kronecker delta arrays."""
  axes = map(int, axes)
  dtype = dtypes.canonicalize_dtype(dtype)
  base_shape = tuple(np.take(shape, axes))  # type: ignore[arg-type]
  iotas = [broadcasted_iota(np.uint32, base_shape, i)
           for i in range(len(base_shape))]
  eyes = [eq(i1, i2) for i1, i2 in zip(iotas[:-1], iotas[1:])]
  result = convert_element_type_p.bind(_reduce(operator.and_, eyes),
                                       new_dtype=dtype, weak_type=False)
  return broadcast_in_dim(result, shape, axes)

def _tri(dtype: DType, shape: Shape, offset: int) -> Array:
  """Like numpy.tri, create a 2D array with ones below a diagonal."""
  offset = int(offset)
  dtype = dtypes.canonicalize_dtype(dtype)
  bool_tri = ge(add(broadcasted_iota(np.int32, shape, 0), np.int32(offset)),
                broadcasted_iota(np.int32, shape, 1))
  return convert_element_type_p.bind(bool_tri, new_dtype=dtype, weak_type=False)

def stop_gradient(x: T) -> T:
  """Stops gradient computation.

  Operationally ``stop_gradient`` is the identity function, that is, it returns
  argument `x` unchanged. However, ``stop_gradient`` prevents the flow of
  gradients during forward or reverse-mode automatic differentiation. If there
  are multiple nested gradient computations, ``stop_gradient`` stops gradients
  for all of them.

  For example:

  >>> jax.grad(lambda x: x**2)(3.)
  DeviceArray(6., dtype=float32, weak_type=True)
  >>> jax.grad(lambda x: jax.lax.stop_gradient(x)**2)(3.)
  DeviceArray(0., dtype=float32, weak_type=True)
  >>> jax.grad(jax.grad(lambda x: x**2))(3.)
  DeviceArray(2., dtype=float32, weak_type=True)
  >>> jax.grad(jax.grad(lambda x: jax.lax.stop_gradient(x)**2))(3.)
  DeviceArray(0., dtype=float32, weak_type=True)
  """
  def stop(x):
    # only bind primitive on inexact dtypes, to avoid some staging
    if core.has_opaque_dtype(x):
      return x
    elif (dtypes.issubdtype(_dtype(x), np.floating) or
        dtypes.issubdtype(_dtype(x), np.complexfloating)):
      return ad_util.stop_gradient_p.bind(x)
    else:
      return x
  return tree_map(stop, x)

def reduce_precision(operand: Union[float, Array],
                     exponent_bits: int,
                     mantissa_bits: int) -> Array:
  """Wraps XLA's `ReducePrecision
  <https://www.tensorflow.org/xla/operation_semantics#reduceprecision>`_
  operator.
  """
  exponent_bits = core.concrete_or_error(
    operator.index, exponent_bits, "exponent_bits argument of lax.reduce_precision")
  mantissa_bits = core.concrete_or_error(
    operator.index, mantissa_bits, "mantissa_bits argument of lax.reduce_precision")
  return reduce_precision_p.bind(operand, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)

def squeeze(array: Array, dimensions: Sequence[int]) -> Array:
  """Squeeze any number of size 1 dimensions from an array."""
  ndim = np.ndim(array)
  dimensions = tuple(sorted(canonicalize_axis(i, ndim) for i in dimensions))
  if not dimensions and isinstance(array, (core.Tracer, device_array.DeviceArray)):
    return array
  return squeeze_p.bind(array, dimensions=dimensions)

def expand_dims(array: Array, dimensions: Sequence[int]) -> Array:
  """Insert any number of size 1 dimensions into an array."""
  if len(set(dimensions)) != len(dimensions):
    raise ValueError(f'repeated axis in lax.expand_dims: {dimensions}')
  ndim_out = np.ndim(array) + len(dimensions)
  dims = [canonicalize_axis(i, ndim_out) for i in dimensions]
  if len(set(dims)) != len(dims):  # check again after canonicalizing
    raise ValueError(f'repeated axis in lax.expand_dims: {dims}')
  dims_set = frozenset(dims)
  result_shape = list(np.shape(array))
  for i in sorted(dims_set):
    result_shape.insert(i, 1)
  broadcast_dims = [i for i in range(ndim_out) if i not in dims_set]
  return broadcast_in_dim(array, result_shape, broadcast_dims)


### convenience wrappers around traceables

def full_like(x: Array, fill_value: Array, dtype: Optional[DType] = None,
              shape: Optional[Shape] = None) -> Array:
  """Create a full array like np.full based on the example array `x`.

  Args:
    x: example array-like, used for shape and dtype information.
    fill_value: a scalar value to fill the entries of the output array.
    dtype: optional, a dtype parameter for the output ndarray.
    shape: optional, a shape parameter for the output ndarray.

  Returns:
    An ndarray with the same shape as `x` with its entries set equal to
    `fill_value`, similar to the output of np.full.
  """
  from jax.experimental import sharding, array

  fill_shape = np.shape(x) if shape is None else canonicalize_shape(shape)
  weak_type = dtype is None and dtypes.is_weakly_typed(x)
  dtype = dtype or _dtype(x)
  val = full(fill_shape, _convert_element_type(fill_value, dtype, weak_type))
  # If the sharding is SingleDeviceSharding then don't take the `if` branch
  # because `val` is already an array with SingleDeviceSharding making this an
  # optimization.
  # TODO(yashkatariya,mattjj): `x` and `val` should have the same sharding,
  # probably in the form of a primitive like `val = match_sharding_p.bind(x, val)`
  # (so it works in staged-out code as well as 'eager' code). Related to
  # equi-sharding.
  if (config.jax_array and hasattr(x, 'sharding') and
      not dispatch.is_single_device_sharding(x.sharding)):
    return array.make_array_from_callback(
        fill_shape, x.sharding, lambda idx: val[idx])  # type: ignore[arg-type]
  return val


def collapse(operand: Array, start_dimension: int,
             stop_dimension: int) -> Array:
  """Collapses dimensions of an array into a single dimension.

  For example, if ``operand`` is an array with shape ``[2, 3, 4]``,
  ``collapse(operand, 0, 2).shape == [6, 4]``. The elements of the collapsed
  dimension are laid out major-to-minor, i.e., with the lowest-numbered
  dimension as the slowest varying dimension.

  Args:
    operand: an input array.
    start_dimension: the start of the dimensions to collapse (inclusive).
    stop_dimension: the end of the dimensions to collapse (exclusive).

  Returns:
    An array where dimensions ``[start_dimension, stop_dimension)`` have been
    collapsed (raveled) into a single dimension.
  """
  lo, hi = start_dimension, stop_dimension
  size = prod(operand.shape[lo:hi])
  new_shape = operand.shape[:lo] + (size,) + operand.shape[hi:]
  return reshape(operand, new_shape)


def batch_matmul(lhs: Array, rhs: Array,
                 precision: PrecisionLike = None) -> Array:
  """Batch matrix multiplication."""
  if _min(lhs.ndim, rhs.ndim) < 2:
    raise ValueError('Arguments to batch_matmul must be at least 2D, got {}, {}'
                     .format(lhs.ndim, rhs.ndim))
  if lhs.ndim != rhs.ndim:
    raise ValueError('Arguments to batch_matmul must have same ndim, got {}, {}'
                     .format(lhs.ndim, rhs.ndim))
  lhs_contract = (lhs.ndim - 1,)
  rhs_contract = (rhs.ndim - 2,)
  batch = tuple(range(lhs.ndim - 2))
  return dot_general(lhs, rhs, ((lhs_contract, rhs_contract), (batch, batch)),
                     precision=precision)


# These functions also exist in the XLA client library, but we treat them
# as non-primitive to maintain a smaller set of autodiff primitives.

def square(x: Array) -> Array:
  r"""Elementwise square: :math:`x^2`."""
  return integer_pow(x, 2)

def reciprocal(x: Array) -> Array:
  r"""Elementwise reciprocal: :math:`1 \over x`."""
  return integer_pow(x, -1)

def _upcast_fp16_for_computation(f):
  @functools.wraps(f)
  def f_wrapped(x):
    dtype = _dtype(x)
    if dtype == np.float16 or dtype == dtypes.bfloat16:
      return convert_element_type(
        f(convert_element_type(x, np.float32)), dtype)
    return f(x)

  return f_wrapped

def tan(x: Array) -> Array:
  r"""Elementwise tangent: :math:`\mathrm{tan}(x)`."""
  return tan_p.bind(x)

def asin(x: Array) -> Array:
  r"""Elementwise arc sine: :math:`\mathrm{asin}(x)`."""
  return asin_p.bind(x)

def acos(x: Array) -> Array:
  r"""Elementwise arc cosine: :math:`\mathrm{acos}(x)`."""
  return acos_p.bind(x)

def atan(x: Array) -> Array:
  r"""Elementwise arc tangent: :math:`\mathrm{atan}(x)`."""
  return atan_p.bind(x)

def sinh(x: Array) -> Array:
  r"""Elementwise hyperbolic sine: :math:`\mathrm{sinh}(x)`."""
  return sinh_p.bind(x)

def cosh(x: Array) -> Array:
  r"""Elementwise hyperbolic cosine: :math:`\mathrm{cosh}(x)`."""
  return cosh_p.bind(x)

def asinh(x: Array) -> Array:
  r"""Elementwise inverse hyperbolic sine: :math:`\mathrm{asinh}(x)`."""
  return asinh_p.bind(x)

def acosh(x: Array) -> Array:
  r"""Elementwise inverse hyperbolic cosine: :math:`\mathrm{acosh}(x)`."""
  return acosh_p.bind(x)

def atanh(x: Array) -> Array:
  r"""Elementwise inverse hyperbolic tangent: :math:`\mathrm{atanh}(x)`."""
  return atanh_p.bind(x)


# Add some methods to ShapedArray that rely on lax primitives

ShapedArray.broadcast = core.aval_method(broadcast)
ShapedArray.transpose = core.aval_method(transpose)  # clobbered by lax_numpy
ShapedArray.reshape = core.aval_method(reshape)      # clobbered by lax_numpy

def _iter(tracer):
  if tracer.ndim == 0:
    raise TypeError("iteration over a 0-d array")  # same as numpy error
  else:
    n = int(tracer.shape[0])
    # return (index_in_dim(tracer, i, keepdims=False) for i in range(n))
    return iter([slicing.index_in_dim(tracer, i, keepdims=False)
                 for i in range(n)])
ShapedArray._iter = staticmethod(_iter)

# Add some ad handlers that use (or could use) lax primitives

def zeros_like_array(x: Array) -> Array:
  return full_like(x, 0)

for t in itertools.chain(
    dtypes.python_scalar_dtypes.keys(), array_types,
    device_array.device_array_types,
    [pxla.ShardedDeviceArray, pxla._ShardedDeviceArray,
     pxla.pmap_lib.ShardedDeviceArray]):
  ad_util.jaxval_adders[t] = add
ad_util.jaxval_zeros_likers[device_array._DeviceArray] = zeros_like_array
ad_util.jaxval_zeros_likers[device_array.Buffer] = zeros_like_array
ad_util.jaxval_zeros_likers[pxla.ShardedDeviceArray] = zeros_like_array
ad_util.jaxval_zeros_likers[pxla.pmap_lib.ShardedDeviceArray] = zeros_like_array


### primitives


_fixed_dtype = lambda dtype: lambda *args, **kwargs: dtypes.canonicalize_dtype(dtype)
_complex_basetype = lambda dtype: np.abs(np.zeros((), dtype)).dtype

_strip_weak_type = lambda *args, **_: False


def unop_dtype_rule(result_dtype, accepted_dtypes, name, aval, **kwargs):
  if not any(dtypes.issubdtype(aval.dtype, t) for t in accepted_dtypes):
    msg = '{} does not accept dtype {}. Accepted dtypes are subtypes of {}.'
    typename = str(np.dtype(aval.dtype).name)
    accepted_typenames = (t.__name__ for t in accepted_dtypes)
    raise TypeError(msg.format(name, typename, ', '.join(accepted_typenames)))
  return result_dtype(aval.dtype)


def unop(result_dtype, accepted_dtypes, name):
  dtype_rule = partial(unop_dtype_rule, result_dtype, accepted_dtypes, name)
  weak_type_rule = partial(_naryop_weak_type_rule, name)
  prim = standard_primitive(_attrgetter('shape'), dtype_rule, name,
                            weak_type_rule=weak_type_rule)
  batching.defvectorized(prim)
  pe.def_trivial_padding(prim)
  return prim
standard_unop = partial(unop, _identity)
_attrgetter = lambda name: lambda x, **kwargs: getattr(x, name)


def naryop_dtype_rule(result_dtype, accepted_dtypes, name, *avals, **kwargs):
  aval_dtypes = [aval.dtype for aval in avals]
  for i, (aval_dtype, types) in enumerate(zip(aval_dtypes, accepted_dtypes)):
    if not any(dtypes.issubdtype(aval_dtype, t) for t in types):
      if aval_dtype == dtypes.float0:
        raise TypeError(
            f"Called {name} with a float0 at position {i}. "
            "float0s do not support any operations by design, because they "
            "are not compatible with non-trivial vector spaces. No implicit dtype "
            "conversion is done. You can use np.zeros_like(arr, dtype=np.float) "
            "to cast a float0 array to a regular zeros array. \n"
            "If you didn't expect to get a float0 you might have accidentally "
            "taken a gradient with respect to an integer argument.")
      else:
        msg = ('{} does not accept dtype {} at position {}. '
               'Accepted dtypes at position {} are subtypes of {}.')
        typename = str(np.dtype(aval_dtype).name)
        typenames = ', '.join(t.__name__ for t in types)
        raise TypeError(msg.format(name, typename, i, i, typenames))
  _check_same_dtypes(name, False, *aval_dtypes)
  return result_dtype(*avals)


def broadcasting_shape_rule(name, *avals):
  shapes = [aval.shape for aval in avals if aval.shape]
  if not shapes:
    return ()
  if len({len(shape) for shape in shapes}) != 1:
    msg = '{}: arrays must have same number of dimensions, got {}.'
    raise TypeError(msg.format(name, ', '.join(map(str, map(tuple, shapes)))))
  # TODO(mattjj): de-duplicate with _try_broadcast_shapes
  result_shape = []
  for ds in zip(*shapes):
    if all(core.same_referent(d, ds[0]) for d in ds[1:]):
      # if all axes are identical objects, the resulting size is the object
      result_shape.append(ds[0])
    else:
      # if all dims are equal (or 1), the result is the non-1 size
      non_1s = [d for d in ds if not core.symbolic_equal_dim(d, 1)]
      if not non_1s:
        result_shape.append(1)
      elif all(core.symbolic_equal_dim(non_1s[0], d) for d in non_1s[1:]):
        result_shape.append(non_1s[0])
      else:
        raise TypeError(f'{name} got incompatible shapes for broadcasting: '
                        f'{", ".join(map(str, map(tuple, shapes)))}.')

  return tuple(result_shape)

def _naryop_weak_type_rule(name, *avals, **kwargs):
  if any(aval.dtype == dtypes.float0 for aval in avals):
    pos = next(i for i, aval in enumerate(avals) if aval.dtype == dtypes.float0)
    raise TypeError(
        f"Called {name} with a float0 at position {pos}. "
        "float0s do not support any operations by design, because they "
        "are not compatible with non-trivial vector spaces. No implicit dtype "
        "conversion is done. You can use np.zeros_like(arr, dtype=np.float) "
        "to cast a float0 array to a regular zeros array. \n"
        "If you didn't expect to get a float0 you might have accidentally "
        "taken a gradient with respect to an integer argument.")
  return all(aval.weak_type for aval in avals)

def naryop(result_dtype, accepted_dtypes, name):
  dtype_rule = partial(naryop_dtype_rule, result_dtype, accepted_dtypes, name)
  shape_rule = partial(broadcasting_shape_rule, name)
  weak_type_rule = partial(_naryop_weak_type_rule, name)
  prim = standard_primitive(shape_rule, dtype_rule, name,
                            weak_type_rule=weak_type_rule)
  batching.defbroadcasting(prim)
  pe.def_trivial_padding(prim)
  return prim
standard_naryop = partial(naryop, _input_dtype)


def _broadcast_translate(op, ctx, avals_in, avals_out, *args):
  """Variant of _standard_translate that performs explicit broadcasting.

  Not all XLA library functions perform their own broadcasting."""
  aval_out, = avals_out
  broadcasted_args = []
  for aval_in, arg in zip(avals_in, args):
    if aval_out.shape != aval_in.shape:
      bcast_dims = tuple(range(len(aval_out.shape) - len(aval_in.shape),
                               len(aval_out.shape)))
      arg = xops.BroadcastInDim(arg, aval_out.shape, bcast_dims)
    broadcasted_args.append(arg)
  return [op(*broadcasted_args)]


# Like autograd.numpy.numpy_vjps.unbroadcast, this utility handles transposition
# involving linear primitives with implicit broadcasting.
def _unbroadcast(aval, x):
  if not isinstance(aval, (core.DShapedArray, ShapedArray)):
    raise TypeError("transpose with implicit broadcasting of unshaped values")
  x_shape = np.shape(x)
  if core.symbolic_equal_shape(aval.shape, x_shape):
    return x
  assert not aval.shape or len(x_shape) == len(aval.shape)
  if not aval.shape:
    return _reduce_sum(x, list(range(len(x_shape))))
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, aval.shape)) if not core.symbolic_equal_dim(a, b)]
    if config.jax_enable_checks: assert all(aval.shape[i] == 1 for i in dims)
    return reshape(_reduce_sum(x, dims), aval.shape)

def _maybe_broadcast(target_shape, x):
  x_shape = np.shape(x)
  if core.symbolic_equal_shape(x_shape, target_shape):
    return x
  elif not x_shape:
    return broadcast_in_dim(x, target_shape, ())
  else:
    dims = [i for i, (a, b) in enumerate(zip(x_shape, target_shape))
            if core.symbolic_equal_dim(a, b)]
    squeeze_shape = [x_shape[i] for i in dims]
    return broadcast_in_dim(reshape(x, squeeze_shape), target_shape, dims)

def broadcast_mhlo(
    aval_out: core.ShapedArray, avals: Sequence[core.ShapedArray],
    args: Sequence[ir.Value]) -> Sequence[ir.Value]:
  """Broadcasts MHLO values with broadcast-compatible shapes to the same shape.
  """
  out = []
  for aval, arg in zip(avals, args):
    if aval.shape != aval_out.shape:
      assert len(aval.shape) <= len(aval_out.shape), (aval, aval_out)
      dims = mlir.dense_int_elements(
          range(len(aval_out.shape) - len(aval.shape), len(aval_out.shape)))
      if any(isinstance(d, ir.Value) for d in aval_out.shape):
        arg = mhlo.DynamicBroadcastInDimOp(
            mlir.aval_to_ir_type(aval_out), arg,
            mlir.shape_tensor(aval_out.shape), dims).result
      else:
        arg = mhlo.BroadcastInDimOp(
            mlir.aval_to_ir_type(aval.update(shape=aval_out.shape)), arg,
            dims).result
    out.append(arg)
  return out

def _nary_lower_mhlo(op: Callable, ctx,
                     *args: Union[ir.Value, Sequence[ir.Value]],
                     explicit_type=False, **params):
  """Lowers an elementwise operator to its MHLO/CHLO equivalent.

  Args:
    explicit_type: does the MHLO/CHLO operator require its output type to be
      provided?
  """
  del params
  avals_in, (aval_out,) = ctx.avals_in, ctx.avals_out
  if config.jax_dynamic_shapes:
    substitute = partial(_substitute_axis_sizes_in_aval, ctx.axis_size_env)
    avals_in = map(substitute, avals_in)
    aval_out = substitute(aval_out)
  broadcasted_args = broadcast_mhlo(aval_out, avals_in, args)
  if explicit_type:
    return op(mlir.aval_to_ir_type(aval_out), *broadcasted_args).results
  else:
    return op(*broadcasted_args).results

def _substitute_axis_sizes_in_aval(
    env: Dict[core.Var, ir.Value], a: core.AbstractValue) -> core.AbstractValue:
  if isinstance(a, core.DShapedArray):
    return a.update(shape=tuple(env.get(d, d) for d in a.shape))  # type: ignore
  return a


_float = {np.floating}
_complex = {np.complexfloating}
_complex_elem_types = {np.float32, np.float64}
_int = {np.integer}
_bool = {np.bool_}

_num = _int | _float | _complex
_any = _int | _float | _complex | _bool
_bool_or_int = _int | _bool
_ordered = _int | _float | _bool

neg_p = standard_unop(_num, 'neg')
ad.deflinear2(neg_p, lambda t, operand: [neg(t)])
mlir.register_lowering(neg_p, partial(_nary_lower_mhlo, mhlo.NegOp))

sign_p = standard_unop(_num, 'sign')
ad.defjvp_zero(sign_p)

def _sign_lower_mhlo(ctx, x):
  x_aval, = ctx.avals_in
  if dtypes.issubdtype(x_aval.dtype, np.unsignedinteger):
    return mhlo.SelectOp(
        mlir.compare_mhlo(x, mlir.full_like_aval(0, x_aval), 'EQ',
                          'UNSIGNED').result,
        mlir.full_like_aval(0, x_aval),
        mlir.full_like_aval(1, x_aval)).results
  return mhlo.SignOp(x).results

mlir.register_lowering(sign_p, _sign_lower_mhlo)

nextafter_p = standard_naryop([_float, _float], 'nextafter')
mlir.register_lowering(nextafter_p, partial(_nary_lower_mhlo, chlo.NextAfterOp))

floor_p = standard_unop(_float, 'floor')
ad.defjvp_zero(floor_p)
mlir.register_lowering(floor_p, partial(_nary_lower_mhlo, mhlo.FloorOp))

ceil_p = standard_unop(_float, 'ceil')
ad.defjvp_zero(ceil_p)
mlir.register_lowering(ceil_p, partial(_nary_lower_mhlo, mhlo.CeilOp))

round_p = standard_unop(_float, 'round')
ad.defjvp_zero(round_p)

def _round_lower(ctx, x, *, rounding_method):
  if rounding_method is RoundingMethod.AWAY_FROM_ZERO:
    return mhlo.RoundOp(x).results
  else:
    assert rounding_method is RoundingMethod.TO_NEAREST_EVEN
    return mhlo.RoundNearestEvenOp(x).results
mlir.register_lowering(round_p, _round_lower)

is_finite_p = unop(_fixed_dtype(np.bool_), _float, 'is_finite')
ad.defjvp_zero(is_finite_p)
mlir.register_lowering(is_finite_p, partial(_nary_lower_mhlo, mhlo.IsFiniteOp))

exp_p = standard_unop(_float | _complex, 'exp')
ad.defjvp2(exp_p, lambda g, ans, x: mul(g, ans))
# For exp_p it is more efficient to use the reconstructed output for the vjp
# rule instead of computing it again from the input.
mlir.register_lowering(exp_p, partial(_nary_lower_mhlo, mhlo.ExpOp))

log_p = standard_unop(_float | _complex, 'log')
ad.defjvp(log_p, lambda g, x: div(g, x))
mlir.register_lowering(log_p, partial(_nary_lower_mhlo, mhlo.LogOp))

expm1_p = standard_unop(_float | _complex, 'expm1')
ad.defjvp2(expm1_p, lambda g, ans, x: mul(g, add(ans, _one(ans))))
mlir.register_lowering(expm1_p, partial(_nary_lower_mhlo, mhlo.Expm1Op))

log1p_p = standard_unop(_float | _complex, 'log1p')
ad.defjvp(log1p_p, lambda g, x: div(g, add(x, _one(x))))
mlir.register_lowering(log1p_p, partial(_nary_lower_mhlo, mhlo.Log1pOp))

tanh_p = standard_unop(_float | _complex, 'tanh')
ad.defjvp2(tanh_p, lambda g, ans, x: mul(add(g, mul(g, ans)),
                                         sub(_one(x), ans)))
mlir.register_lowering(tanh_p, partial(_nary_lower_mhlo, mhlo.TanhOp))

logistic_p = standard_unop(_float | _complex, 'logistic')
ad.defjvp2(logistic_p, lambda g, ans, x: mul(g, mul(ans, sub(_one(ans), ans))))
# TODO(phawkins): switch to mhlo.logistic lowering; debug numerical problems.
# mlir.register_lowering(logistic_p, partial(_nary_lower_mhlo, mhlo.LogisticOp))

def logistic_impl(x):
  one = _const(x, 1)
  return div(one, add(one, exp(neg(x))))

mlir.register_lowering(logistic_p,
                       mlir.lower_fun(logistic_impl, multiple_results=False))

sin_p = standard_unop(_float | _complex, 'sin')
ad.defjvp(sin_p, lambda g, x: mul(g, cos(x)))
if mlir_api_version < 27:
  mlir.register_lowering(sin_p, partial(_nary_lower_mhlo, mhlo.SinOp))
else:
  mlir.register_lowering(sin_p, partial(_nary_lower_mhlo, mhlo.SineOp))

cos_p = standard_unop(_float | _complex, 'cos')
ad.defjvp(cos_p, lambda g, x: neg(mul(g, sin(x))))
if mlir_api_version < 28:
  mlir.register_lowering(cos_p, partial(_nary_lower_mhlo, mhlo.CosOp))
else:
  mlir.register_lowering(cos_p, partial(_nary_lower_mhlo, mhlo.CosineOp))

@_upcast_fp16_for_computation
def _tan_impl(x):
  return div(sin(x), cos(x))

tan_p = standard_unop(_float | _complex, 'tan')
ad.defjvp2(tan_p, lambda g, ans, x: mul(g, _const(x, 1) + square(ans)))
mlir.register_lowering(tan_p, partial(_nary_lower_mhlo, chlo.TanOp))

def asin_impl(x):
  if dtypes.issubdtype(_dtype(x), np.complexfloating):
    return mul(_const(x, -1j), asinh(mul(_const(x, 1j), x)))
  else:
    return mul(_const(x, 2),
               atan2(x, add(_const(x, 1), sqrt(sub(_const(x, 1), square(x))))))

asin_p = standard_unop(_float | _complex, 'asin')
ad.defjvp(asin_p, lambda g, x: mul(g, rsqrt(_const(x, 1) - square(x))))
if mlir_api_version < 31:
  mlir.register_lowering(asin_p, mlir.lower_fun(asin_impl,
                                                multiple_results=False))
else:
  mlir.register_lowering(asin_p, partial(_nary_lower_mhlo, chlo.AsinOp))

def acos_impl(x):
  if dtypes.issubdtype(_dtype(x), np.complexfloating):
    result = mul(_const(x, 1j), acosh(x))
    # By convention, numpy chooses the branch with positive real part.
    rpart = real(result)
    return select(
      gt(rpart, _const(rpart, 0)),
      result,
      neg(result)
    )
  else:
    return select(
        ne(x, _const(x, -1.0)),
        mul(_const(x, 2),
            atan2(sqrt(sub(_const(x, 1), square(x))), add(_const(x, 1), x))),
        full_like(x, np.pi))

acos_p = standard_unop(_float | _complex, 'acos')
ad.defjvp(acos_p, lambda g, x: mul(g, -rsqrt(_const(x, 1) - square(x))))
mlir.register_lowering(acos_p,
                       mlir.lower_fun(acos_impl, multiple_results=False))

def atan_impl(x):
  return atan2(x, _const(x, 1))

atan_p = standard_unop(_float | _complex, 'atan')
ad.defjvp(atan_p, lambda g, x: div(g, _const(x, 1) + square(x)))
if mlir_api_version < 31:
  mlir.register_lowering(atan_p, mlir.lower_fun(atan_impl,
                                                multiple_results=False))
else:
  mlir.register_lowering(atan_p, partial(_nary_lower_mhlo, chlo.AtanOp))

atan2_p = standard_naryop([_float | _complex, _float | _complex], 'atan2')
ad.defjvp(atan2_p,
          lambda g, x, y: g * (y / (square(x) + square(y))),
          lambda g, x, y: g * -x / (square(x) + square(y)))
mlir.register_lowering(atan2_p, partial(_nary_lower_mhlo, mhlo.Atan2Op))

sinh_p = standard_unop(_float | _complex, 'sinh')
ad.defjvp(sinh_p, lambda g, x: mul(g, cosh(x)))
mlir.register_lowering(sinh_p, partial(_nary_lower_mhlo, chlo.SinhOp))

cosh_p = standard_unop(_float | _complex, 'cosh')
ad.defjvp(cosh_p, lambda g, x: mul(g, sinh(x)))
mlir.register_lowering(cosh_p, partial(_nary_lower_mhlo, chlo.CoshOp))

asinh_p = standard_unop(_float | _complex, 'asinh')
ad.defjvp(asinh_p, lambda g, x: mul(g, rsqrt(square(x) + _one(x))))
mlir.register_lowering(asinh_p, partial(_nary_lower_mhlo, chlo.AsinhOp))

acosh_p = standard_unop(_float | _complex, 'acosh')
ad.defjvp(acosh_p,
          lambda g, x: mul(g, rsqrt((x - _one(x)) * (x + _one(x)))))
mlir.register_lowering(acosh_p, partial(_nary_lower_mhlo, chlo.AcoshOp))

atanh_p = standard_unop(_float | _complex, 'atanh')
ad.defjvp(atanh_p,
          lambda g, x: mul(reciprocal(_one(x) + x), div(g, (_one(x) - x))))
mlir.register_lowering(atanh_p, partial(_nary_lower_mhlo, chlo.AtanhOp))

regularized_incomplete_beta_p = standard_naryop(
    [_float, _float, _float], 'regularized_incomplete_beta')
xla.register_translation(
    regularized_incomplete_beta_p,
    partial(_broadcast_translate, xops.RegularizedIncompleteBeta))

def betainc_gradx(g, a, b, x):
  lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
  partial_x = exp((b - 1) * log1p(-x) +
                  (a - 1) * log(x) - lbeta)
  return partial_x * g

def betainc_grad_not_implemented(g, a, b, x):
  raise ValueError("Betainc gradient with respect to a and b not supported.")

ad.defjvp(regularized_incomplete_beta_p,
  betainc_grad_not_implemented,
  betainc_grad_not_implemented,
  betainc_gradx)

lgamma_p = standard_unop(_float, 'lgamma')
ad.defjvp(lgamma_p, lambda g, x: mul(g, digamma(x)))
mlir.register_lowering(lgamma_p, partial(_nary_lower_mhlo, chlo.LgammaOp))

digamma_p = standard_unop(_float, 'digamma')
mlir.register_lowering(digamma_p, partial(_nary_lower_mhlo, chlo.DigammaOp))

igamma_p = standard_naryop([_float, _float], 'igamma')
xla.register_translation(igamma_p, partial(_broadcast_translate, xops.Igamma))
igamma_grad_a_p = standard_naryop([_float, _float], 'igamma_grad_a')
xla.register_translation(igamma_grad_a_p,
                         partial(_broadcast_translate, xops.IgammaGradA))

def igamma_gradx(g, a, x):
  return g * exp(-x + (a - _ones(a)) * log(x) - lgamma(a))

def igamma_grada(g, a, x):
  return g * igamma_grad_a(a, x)

ad.defjvp(igamma_p, igamma_grada, igamma_gradx)

igammac_p = standard_naryop([_float, _float], 'igammac')
xla.register_translation(igammac_p, partial(_broadcast_translate, xops.Igammac))

def igammac_gradx(g, a, x):
  return -igamma_gradx(g, a, x)

def igammac_grada(g, a, x):
  return -igamma_grada(g, a, x)

ad.defjvp(igammac_p, igammac_grada, igammac_gradx)

random_gamma_grad_p = standard_naryop([_float, _float], 'random_gamma_grad')
xla.register_translation(random_gamma_grad_p,
                         partial(_broadcast_translate, xops.RandomGammaGrad))
bessel_i0e_p = standard_unop(_float, 'bessel_i0e')
xla.register_translation(bessel_i0e_p, standard_translate(bessel_i0e_p))
ad.defjvp2(bessel_i0e_p, lambda g, y, x: g * (bessel_i1e(x) - sign(x) * y))

bessel_i1e_p = standard_unop(_float, 'bessel_i1e')
xla.register_translation(bessel_i1e_p, standard_translate(bessel_i1e_p))
if mlir_api_version < 32:
  xla.register_translation(bessel_i1e_p, standard_translate(bessel_i1e_p))
else:
  mlir.register_lowering(bessel_i1e_p,
                         partial(_nary_lower_mhlo, chlo.BesselI1eOp))
def _bessel_i1e_jvp(g, y, x):
  eps = dtypes.finfo(_dtype(x)).eps
  x_is_not_tiny = abs(x) > eps
  safe_x = select(x_is_not_tiny, x, full_like(x, eps))
  dy_dx = bessel_i0e(safe_x) - y * (sign(safe_x) + reciprocal(safe_x))
  dy_dx = select(x_is_not_tiny, dy_dx, full_like(x, 0.5))
  return g * dy_dx
ad.defjvp2(bessel_i1e_p, _bessel_i1e_jvp)

erf_p = standard_unop(_float, 'erf')
ad.defjvp(erf_p, lambda g, x: mul(_const(x, 2. / np.sqrt(np.pi)),
                                  mul(g, exp(neg(square(x))))))
mlir.register_lowering(erf_p, partial(_nary_lower_mhlo, chlo.ErfOp))

erfc_p = standard_unop(_float, 'erfc')
ad.defjvp(erfc_p, lambda g, x: mul(_const(x, -2. / np.sqrt(np.pi)),
                                   mul(g, exp(neg(square(x))))))
mlir.register_lowering(erfc_p, partial(_nary_lower_mhlo, chlo.ErfcOp))

erf_inv_p = standard_unop(_float, 'erf_inv')
ad.defjvp2(erf_inv_p, lambda g, ans, x: mul(_const(x, np.sqrt(np.pi) / 2.),
                                            mul(g, exp(square(ans)))))
xla.register_translation(erf_inv_p, standard_translate(erf_inv_p))

real_p = unop(_complex_basetype, _complex, 'real')
ad.deflinear2(real_p, lambda t, _: [complex(t, np.zeros((), _dtype(t)))])
mlir.register_lowering(real_p, partial(_nary_lower_mhlo, mhlo.RealOp))

imag_p = unop(_complex_basetype, _complex, 'imag')
ad.deflinear2(imag_p, lambda t, _: [complex(np.zeros((), _dtype(t)), neg(t))])
mlir.register_lowering(imag_p, partial(_nary_lower_mhlo, mhlo.ImagOp))


def _complex_transpose_rule(t, x, y):
  assert ad.is_undefined_primal(x) or ad.is_undefined_primal(y)
  if ad.is_undefined_primal(x) and ad.is_undefined_primal(y):
    if type(t) is ad_util.Zero:
      return [ad_util.Zero(x.aval), ad_util.Zero(y.aval)]
    else:
      return [_unbroadcast(x.aval, real(t)), _unbroadcast(y.aval, imag(neg(t)))]
  elif ad.is_undefined_primal(x):
    if type(t) is ad_util.Zero:
      return [ad_util.Zero(x.aval), None]
    else:
      return [_unbroadcast(x.aval, real(t)), None]
  else:
    if type(t) is ad_util.Zero:
      return [None, ad_util.Zero(y.aval)]
    else:
      return [None, _unbroadcast(y.aval, imag(neg(t)))]

_complex_dtype = lambda dtype, *args: (np.zeros((), dtype) + np.zeros((), np.complex64)).dtype
complex_p = naryop(_complex_dtype, [_complex_elem_types, _complex_elem_types],
                  'complex')
ad.deflinear2(complex_p, _complex_transpose_rule)
mlir.register_lowering(complex_p, partial(_nary_lower_mhlo, mhlo.ComplexOp))

conj_p = unop(_complex_dtype, _complex_elem_types | _complex, 'conj')

def _conj_impl(x, **kw):
  if dtypes.issubdtype(x.dtype, np.complexfloating):
    return complex(real(x), -imag(x))
  else:
    return complex(x, _zeros(x))

mlir.register_lowering(conj_p,
                       mlir.lower_fun(_conj_impl, multiple_results=False))


def _conj_transpose_rule(t, x, *, input_dtype):
  assert ad.is_undefined_primal(x)
  if dtypes.issubdtype(input_dtype, np.complexfloating):
    return [conj(t)]
  else:
    return [real(t)]

ad.primitive_jvps[conj_p] = partial(ad.linear_jvp, conj_p)
ad.primitive_transposes[conj_p] = _conj_transpose_rule

abs_p = unop(_complex_basetype, _num, 'abs')
mlir.register_lowering(abs_p, partial(_nary_lower_mhlo, mhlo.AbsOp))

def _abs_jvp_rule(g, ans, x):
  if _iscomplex(x):
    return _maybe_real(mul(g, div(_maybe_conj(x),
           _replace_zero(convert_element_type(ans, _dtype(x))))))
  else:
    return select(ge(x, _zero(x)), g, neg(g))
ad.defjvp2(abs_p, _abs_jvp_rule)
_maybe_conj = lambda x: conj(x) if _iscomplex(x) else x
_maybe_real = lambda x: real(x) if _iscomplex(x) else x

sqrt_p = standard_unop(_float | _complex, 'sqrt')
ad.defjvp2(sqrt_p, lambda g, ans, x: mul(g, div(_const(x, 0.5), ans)))
mlir.register_lowering(sqrt_p, partial(_nary_lower_mhlo, mhlo.SqrtOp))

rsqrt_p = standard_unop(_float | _complex, 'rsqrt')
ad.defjvp2(rsqrt_p,
           lambda g, ans, x:
           mul(g, mul(_const(x, -0.5), div(ans, x))))
mlir.register_lowering(rsqrt_p, partial(_nary_lower_mhlo, mhlo.RsqrtOp))

cbrt_p = standard_unop(_float, 'cbrt')
ad.defjvp2(cbrt_p,
           lambda g, ans, x: mul(g, mul(_const(x, 1/3), integer_pow(ans, -2))))
mlir.register_lowering(cbrt_p, partial(_nary_lower_mhlo, mhlo.CbrtOp))

pow_p = standard_naryop([_float | _complex, _float | _complex], 'pow')

def _pow_jvp_lhs(g, ans, x, y):
  return mul(g, mul(y, pow(x, sub(y, _ones(y)))))

def _pow_jvp_rhs(g, ans, x, y):
  return mul(g, mul(log(_replace_zero(x)), ans))

ad.defjvp2(pow_p, _pow_jvp_lhs, _pow_jvp_rhs)
mlir.register_lowering(pow_p, partial(_nary_lower_mhlo, mhlo.PowOp))


def _integer_pow_dtype_rule(x, *, y):
  dtype = unop_dtype_rule(_identity, _int | _float | _complex, 'integer_pow', x)
  if y < 0 and dtypes.issubdtype(dtype, np.integer):
    raise TypeError("Integers cannot be raised to negative powers, got "
                    f"integer_pow({x}, {y})")
  return dtype

def _integer_pow_jvp(g, x, *, y):
  return _zeros(g) if y == 0 else mul(g, mul(_const(x, y), integer_pow(x, y - 1)))

integer_pow_p = standard_primitive(
  _attrgetter('shape'), _integer_pow_dtype_rule, 'integer_pow')
batching.defvectorized(integer_pow_p)
ad.defjvp(integer_pow_p, _integer_pow_jvp)
pe.def_trivial_padding(integer_pow_p)

def _integer_pow(x, *, y):
  # This should be kept in sync with the jax2tf translation rule.
  if y == 0:
    return full_like(x, 1)
  is_reciprocal = y < 0
  if is_reciprocal:
    y = -y
  acc = None
  while y > 0:
    if y & 1:
      acc = x if acc is None else mul(acc, x)
    y >>= 1
    if y > 0:
      # We don't call square because it calls integer_pow.
      x = mul(x, x)
  return div(full_like(acc, 1), acc) if is_reciprocal else acc


def _integer_pow_lowering(ctx, x, *, y):
  lowering = mlir.lower_fun(_integer_pow, multiple_results=False)
  # TODO(b/217551391): emitting an out-of-line call leads to a large
  # expansion when the MHLO is lowered to HLO, because the HLO lowering
  # clones the callee. Consider unconditionally caching when the MHLO->HLO
  # lowering doesn't expand the program.
  if y >= 4:
    lowering = mlir.cache_lowering(lowering)
  return lowering(ctx, x, y=y)

mlir.register_lowering(integer_pow_p, _integer_pow_lowering)

_replace_zero = lambda x: select(eq(x, _const(x, 0)), _ones(x), x)

not_p = standard_unop(_bool_or_int, 'not')
ad.defjvp_zero(not_p)
mlir.register_lowering(not_p, partial(_nary_lower_mhlo, mhlo.NotOp))

and_p = standard_naryop([_bool_or_int, _bool_or_int], 'and')
ad.defjvp_zero(and_p)
mlir.register_lowering(and_p, partial(_nary_lower_mhlo, mhlo.AndOp))

or_p = standard_naryop([_bool_or_int, _bool_or_int], 'or')
ad.defjvp_zero(or_p)
mlir.register_lowering(or_p, partial(_nary_lower_mhlo, mhlo.OrOp))

xor_p = standard_naryop([_bool_or_int, _bool_or_int], 'xor')
ad.defjvp_zero(xor_p)
mlir.register_lowering(xor_p, partial(_nary_lower_mhlo, mhlo.XorOp))

population_count_p = standard_unop(_int, 'population_count')
mlir.register_lowering(population_count_p,
                       partial(_nary_lower_mhlo, mhlo.PopulationCountOp))

clz_p = standard_unop(_int, 'clz')
mlir.register_lowering(clz_p, partial(_nary_lower_mhlo, mhlo.ClzOp))

def _add_jvp(primals, tangents):
  x, y = primals
  xdot, ydot = tangents
  primal_out = add(x, y)
  if type(xdot) is type(ydot) is ad_util.Zero:
    return primal_out, ad_util.Zero.from_value(primal_out)
  if type(xdot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, ydot)
  elif type(ydot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, xdot)
  else:
    return primal_out, add(xdot, ydot)

def _add_transpose(t, x, y):
  # Morally the following assertion is true, but because we instantiate zeros in
  # some places (e.g. in custom_jvp) it may not always hold. For example, see
  # api_test.py's CustomJVPTest.test_jaxpr_zeros.
  # assert ad.is_undefined_primal(x) and ad.is_undefined_primal(y)
  x_aval = x.aval if ad.is_undefined_primal(x) else _abstractify(x)
  y_aval = y.aval if ad.is_undefined_primal(y) else _abstractify(y)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(x_aval), ad_util.Zero(y_aval)]
  else:
    return [_unbroadcast(x_aval, t), _unbroadcast(y_aval, t)]

def _add_inverse(r, x, y):
  xr = r - y
  yr = r - x
  return xr, yr

# TODO(slebedev): Why does mypy fail to infer the type here?
add_p: Primitive = standard_naryop([_num, _num], 'add')
ad.primitive_jvps[add_p] = _add_jvp
ad.primitive_transposes[add_p] = _add_transpose
mlir.register_lowering(add_p, partial(_nary_lower_mhlo, mhlo.AddOp))

def _sub_jvp(primals, tangents):
  x, y = primals
  xdot, ydot = tangents
  primal_out = sub(x, y)
  if type(xdot) is type(ydot) is ad_util.Zero:
    return primal_out, ad_util.Zero.from_value(primal_out)
  if type(xdot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, neg(ydot))
  elif type(ydot) is ad_util.Zero:
    return primal_out, _maybe_broadcast(primal_out.shape, xdot)
  else:
    return primal_out, sub(xdot, ydot)

def _sub_transpose(t, x, y):
  # Morally the following assertion is true, but see the comment in add_p's
  # transpose rule.
  # assert ad.is_undefined_primal(x) and ad.is_undefined_primal(y)
  x_aval = x.aval if ad.is_undefined_primal(x) else _abstractify(x)
  y_aval = y.aval if ad.is_undefined_primal(y) else _abstractify(y)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(x_aval), ad_util.Zero(y_aval)]
  else:
    return [_unbroadcast(x_aval, t), _unbroadcast(y_aval, neg(t))]

sub_p = standard_naryop([_num, _num], 'sub')
ad.primitive_jvps[sub_p] = _sub_jvp
ad.primitive_transposes[sub_p] = _sub_transpose
if mlir_api_version < 29:
  mlir.register_lowering(sub_p, partial(_nary_lower_mhlo, mhlo.SubOp))
else:
  mlir.register_lowering(sub_p, partial(_nary_lower_mhlo, mhlo.SubtractOp))


def _mul_transpose(ct, x, y):
  assert ad.is_undefined_primal(x) ^ ad.is_undefined_primal(y)
  if ad.is_undefined_primal(x):
    if type(ct) is ad_util.Zero:
      return [ad_util.Zero(x.aval), None]
    else:
      return [_unbroadcast(x.aval, mul(ct, y)), None]
  else:
    if type(ct) is ad_util.Zero:
      return [None, ad_util.Zero(y.aval)]
    else:
      return [None, _unbroadcast(y.aval, mul(x, ct))]

def _mul_inverse(r, x, y):
  xr = r / y
  yr = r / x
  return xr, yr

mul_p = standard_naryop([_num, _num], 'mul')
ad.defjvp(mul_p,
          lambda xdot, x, y: mul(xdot, y),
          lambda ydot, x, y: mul(x, ydot))
ad.primitive_transposes[mul_p] = _mul_transpose
mlir.register_lowering(mul_p, partial(_nary_lower_mhlo, mhlo.MulOp))

def _div_transpose_rule(cotangent, x, y):
  assert ad.is_undefined_primal(x) and not ad.is_undefined_primal(y)
  if type(cotangent) is ad_util.Zero:
    return [ad_util.Zero(x.aval), None]
  else:
    return [_unbroadcast(x.aval, div(cotangent, y)), None]
div_p = standard_naryop([_num, _num], 'div')
ad.defjvp(div_p,
          lambda g, x, y: div(g, y),
          lambda g, x, y: mul(mul(neg(g), x), integer_pow(y, -2)))
ad.primitive_transposes[div_p] = _div_transpose_rule
mlir.register_lowering(div_p, partial(_nary_lower_mhlo, mhlo.DivOp))

rem_p = standard_naryop([_int | _float, _int | _float], 'rem')
ad.defjvp(
    rem_p,
    lambda g, x, y: _maybe_broadcast(broadcast_shapes(np.shape(x), np.shape(y)), g),
    lambda g, x, y: mul(neg(g), mul(sign(div(x, y)), floor(abs(div(x, y))))))
mlir.register_lowering(rem_p, partial(_nary_lower_mhlo, mhlo.RemOp))

def _minmax_complex_lowering(x, y, *, lax_cmp_pick_x):
  result_shape = broadcast_shapes(np.shape(x), np.shape(y))
  x = _maybe_broadcast(result_shape, x)
  y = _maybe_broadcast(result_shape, y)
  rx = real(x)
  ry = real(y)
  pick_x = select(eq(rx, ry), lax_cmp_pick_x(imag(x), imag(y)),
                  lax_cmp_pick_x(rx, ry))
  return select(pick_x, x, y)

max_p: core.Primitive = standard_naryop([_any, _any], 'max')
ad.defjvp2(max_p,
           lambda g, ans, x, y: mul(g, _balanced_eq(x, ans, y)),
           lambda g, ans, x, y: mul(g, _balanced_eq(y, ans, x)))
mlir.register_lowering(max_p, partial(_nary_lower_mhlo, mlir.max_mhlo))

min_p: core.Primitive = standard_naryop([_any, _any], 'min')
ad.defjvp2(min_p,
           lambda g, ans, x, y: mul(g, _balanced_eq(x, ans, y)),
           lambda g, ans, x, y: mul(g, _balanced_eq(y, ans, x)))
mlir.register_lowering(min_p, partial(_nary_lower_mhlo, mlir.min_mhlo))

shift_left_p = standard_naryop([_int, _int], 'shift_left')
ad.defjvp_zero(shift_left_p)
mlir.register_lowering(shift_left_p, partial(_nary_lower_mhlo, mhlo.ShiftLeftOp))

shift_right_arithmetic_p = standard_naryop([_int, _int], 'shift_right_arithmetic')
ad.defjvp_zero(shift_right_arithmetic_p)
mlir.register_lowering(shift_right_arithmetic_p,
                       partial(_nary_lower_mhlo, mhlo.ShiftRightArithmeticOp))

shift_right_logical_p = standard_naryop([_int, _int], 'shift_right_logical')
ad.defjvp_zero(shift_right_logical_p)
mlir.register_lowering(shift_right_logical_p,
                       partial(_nary_lower_mhlo, mhlo.ShiftRightLogicalOp))

def _compare_lower_mhlo(direction: str, ctx, x, y):
  x_aval, y_aval = ctx.avals_in
  aval_out, = ctx.avals_out
  x, y = broadcast_mhlo(aval_out.update(dtype=x_aval.dtype), ctx.avals_in,
                        (x, y))
  if dtypes.issubdtype(x_aval.dtype, np.inexact):
    compare_type = "FLOAT"
  elif dtypes.issubdtype(x_aval.dtype, np.signedinteger):
    compare_type = "SIGNED"
  else:
    compare_type = "UNSIGNED"
  return mlir.compare_mhlo(x, y, direction, compare_type).results

eq_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'eq')
ad.defjvp_zero(eq_p)
mlir.register_lowering(eq_p, partial(_compare_lower_mhlo, "EQ"))

ne_p = naryop(_fixed_dtype(np.bool_), [_any, _any], 'ne')
ad.defjvp_zero(ne_p)
mlir.register_lowering(ne_p, partial(_compare_lower_mhlo, "NE"))

ge_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'ge')
ad.defjvp_zero(ge_p)
mlir.register_lowering(ge_p, partial(_compare_lower_mhlo, "GE"))

gt_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'gt')
ad.defjvp_zero(gt_p)
mlir.register_lowering(gt_p, partial(_compare_lower_mhlo, "GT"))

le_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'le')
ad.defjvp_zero(le_p)
mlir.register_lowering(le_p, partial(_compare_lower_mhlo, "LE"))

lt_p = naryop(_fixed_dtype(np.bool_), [_ordered, _ordered], 'lt')
ad.defjvp_zero(lt_p)
mlir.register_lowering(lt_p, partial(_compare_lower_mhlo, "LT"))


def _convert_element_type_shape_rule(operand, *, new_dtype, weak_type):
  return operand.shape

def _convert_element_type_dtype_rule(operand, *, new_dtype, weak_type):
  return new_dtype

def _convert_element_type_weak_type_rule(operand, *, new_dtype, weak_type):
  return weak_type

def _convert_element_type_transpose_rule(ct, operand, *, new_dtype, weak_type):
  assert ad.is_undefined_primal(operand)
  old_dtype = operand.aval.dtype
  old_weak_type = dtypes.is_weakly_typed(operand)
  if type(ct) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)]
  elif core.primal_dtype_to_tangent_dtype(old_dtype) == dtypes.float0:
    return [ad_util.Zero(operand.aval.update(dtype=dtypes.float0, weak_type=False))]
  else:
    return [convert_element_type_p.bind(ct, new_dtype=old_dtype,
                                        weak_type=old_weak_type)]

def _convert_element_type_jvp_rule(tangent, operand , *, new_dtype, weak_type):
  if core.primal_dtype_to_tangent_dtype(new_dtype) == dtypes.float0:
    return ad_util.Zero(tangent.aval.update(dtype=dtypes.float0, weak_type=False))
  else:
    return convert_element_type_p.bind(tangent, new_dtype=new_dtype,
                                       weak_type=weak_type)

def _convert_elt_type_folding_rule(consts, eqn):
  # We constant-fold convert_element_types applied to constants if those
  # constants are Python builtin numeric types or numpy.ndarrays (so as not
  # to perform any device operations when constant-folding) and if the output
  # type can be faithfully represented by a Python builtin numeric type or
  # numpy.ndarray. If those conditions are met, we output a numpy.ndarray
  # constant if the output type is not weak, and if the output type is weak then
  # we output a Python builtin numeric type.
  # TODO(mattjj): allow constant-folding CPU-backed JAX arrays
  c, = consts
  o, = eqn.outvars
  if (type(c) in {np.ndarray, *dtypes.python_scalar_dtypes} and
      isinstance(o.aval, core.UnshapedArray) and not np.shape(c)):
    out = np.array(c, eqn.params['new_dtype'])
    if not o.aval.weak_type:
      return [out], None
    out = out.item()
    if core.get_aval(out).dtype is o.aval.dtype:
      return [out], None
  return [None], eqn

def _convert_elt_type_fwd_rule(eqn):
  v, = eqn.invars
  if (v.aval.dtype == eqn.params['new_dtype'] and
      v.aval.weak_type == eqn.params['weak_type']):
    return [v], None
  else:
    return [None], eqn

def _convert_elt_type_pp_rule(eqn, context, settings):
  # don't print new_dtype because the output binder shows it, don't print
  # weak_type when false
  printed_params = {}
  if eqn.params['weak_type']:
    printed_params['weak_type'] = True
  lhs = core.pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name),
         core.pp_kv_pairs(sorted(printed_params.items()), context, settings),
         pp.text(" ") + core.pp_vars(eqn.invars, context)]
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  return [lhs, pp.text(" = ", annotation=annotation), *rhs]


convert_element_type_p = Primitive('convert_element_type')
convert_element_type_p.def_impl(partial(xla.apply_primitive, convert_element_type_p))
convert_element_type_p.def_abstract_eval(
    partial(standard_abstract_eval, convert_element_type_p,
            _convert_element_type_shape_rule, _convert_element_type_dtype_rule,
            _convert_element_type_weak_type_rule, standard_named_shape_rule))
ad.defjvp(convert_element_type_p, _convert_element_type_jvp_rule)
ad.primitive_transposes[convert_element_type_p] = _convert_element_type_transpose_rule
batching.defvectorized(convert_element_type_p)
pe.const_fold_rules[convert_element_type_p] = _convert_elt_type_folding_rule
pe.forwarding_rules[convert_element_type_p] = _convert_elt_type_fwd_rule
pe.def_trivial_padding(convert_element_type_p)
# TODO(mattjj): un-comment the next line (see #9456)
# core.pp_eqn_rules[convert_element_type_p] = _convert_elt_type_pp_rule

def _real_dtype(dtype): return np.finfo(dtype).dtype

def _convert_element_type_lower(ctx, operand, *, new_dtype, weak_type):
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  if (dtypes.issubdtype(aval_in.dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    operand = mhlo.RealOp(operand).result
    aval_in = aval_in.update(dtype=_real_dtype(aval_in.dtype))
  return [mlir.convert_mhlo(operand, aval_in, aval_out)]

mlir.register_lowering(convert_element_type_p, _convert_element_type_lower)


def _bitcast_convert_type_shape_rule(operand, *, new_dtype):
  return operand.shape

def _bitcast_convert_type_dtype_rule(operand, *, new_dtype):
  old_dtype = dtypes.canonicalize_dtype(operand.dtype)
  if dtypes.issubdtype(old_dtype, np.bool_) or dtypes.issubdtype(old_dtype, np.complexfloating):
    if old_dtype != new_dtype:
      raise TypeError(f"`bitcast_convert_type` for operand type ({old_dtype}) cannot have different destination type ({new_dtype})")
  if np.dtype(old_dtype).itemsize != np.dtype(new_dtype).itemsize:
    raise TypeError(f"`bitcast_convert_type` for operand type ({old_dtype}) must have destination type ({new_dtype}) of same size.")
  return new_dtype

bitcast_convert_type_p = standard_primitive(
    _bitcast_convert_type_shape_rule, _bitcast_convert_type_dtype_rule,
    'bitcast_convert_type', weak_type_rule=_strip_weak_type)
ad.defjvp_zero(bitcast_convert_type_p)
batching.defvectorized(bitcast_convert_type_p)

def _bitcast_convert_type_lower(ctx, operand, *, new_dtype):
  aval_out, = ctx.avals_out
  return mhlo.BitcastConvertOp(mlir.aval_to_ir_type(aval_out), operand).results

mlir.register_lowering(bitcast_convert_type_p, _bitcast_convert_type_lower)


def _validate_preferred_element_type(input_dtype, preferred_element_type):

  if dtypes.issubdtype(input_dtype, np.integer) and dtypes.issubdtype(preferred_element_type, np.floating):
    # Special-case integer->float multiply. This is allowed, and also allows
    # different signedness between input and output.
    pass
  else:
    allowed_types = (np.integer, np.floating, np.complexfloating)
    if any(dtypes.issubdtype(input_dtype, t) and not dtypes.issubdtype(preferred_element_type, t) for t in allowed_types):
      raise TypeError("Input type is incompatible with `preferred_element_type`. The compatible combinations of "
                      "(input_type, preferred_element_type) are (integral, integral), (integral, floating), "
                      "(floating, floating), (complex, complex.")
    if dtypes.issubdtype(input_dtype, np.signedinteger) and not dtypes.issubdtype(preferred_element_type, np.signedinteger):
      raise TypeError("`preferred_element_type` must have the same signedness as the original type.")
  input_bitwidth = np.dtype(input_dtype).itemsize
  preferred_bitwidth = np.dtype(preferred_element_type).itemsize
  if preferred_bitwidth < input_bitwidth:
    raise TypeError("`preferred_element_type` must not be narrower than the original type.")

def _precision_config(precision):
  if precision is not None:
    config = xla_client.PrecisionConfig()
    if isinstance(precision, tuple):
      config.operand_precision.extend(precision)
    else:
      config.operand_precision.extend((precision, precision))
    return config
  return None

def _masked(padded_value, logical_shape, dimensions, value=0):
  """
  Sets all padding to the given value (default is 0) in the given dimensions.
  All values outside the logical shape are considered padding.
  """
  if len(dimensions) == 0:
    return padded_value

  masks = [broadcasted_iota(np.int32, padded_value.shape, d) < logical_shape[d]
           for d in dimensions]
  mask_intersection = masks[0]
  for mask in masks[1:]:
    mask_intersection &= mask
  return select(mask_intersection, padded_value, full_like(padded_value, value))


def _dot_general_shape_rule(lhs, rhs, *, dimension_numbers, precision,
                            preferred_element_type: Optional[DType]):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  if not all(np.all(np.greater_equal(d, 0)) and np.all(np.less(d, lhs.ndim))
             for d in (lhs_contracting, lhs_batch)):
    msg = ("dot_general requires lhs dimension numbers to be nonnegative and "
           "less than the number of axes of the lhs value, got "
           f"lhs_batch of {lhs_batch} and lhs_contracting of {lhs_contracting} "
           f"for lhs of rank {lhs.ndim}")
    raise TypeError(msg)
  if not all(np.all(np.greater_equal(d, 0)) and np.all(np.less(d, rhs.ndim))
             for d in (rhs_contracting, rhs_batch)):
    msg = ("dot_general requires rhs dimension numbers to be nonnegative and "
           "less than the number of axes of the rhs value, got "
           f"rhs_batch of {rhs_batch} and rhs_contracting of {rhs_contracting} "
           f"for rhs of rank {rhs.ndim}")
    raise TypeError(msg)
  if len(lhs_batch) != len(rhs_batch):
    msg = ("dot_general requires equal numbers of lhs_batch and rhs_batch "
           "dimensions, got lhs_batch {} and rhs_batch {}.")
    raise TypeError(msg.format(lhs_batch, rhs_batch))
  lhs_contracting_set, lhs_batch_set = set(lhs_contracting), set(lhs_batch)
  rhs_contracting_set, rhs_batch_set = set(rhs_contracting), set(rhs_batch)
  if len(lhs_batch_set) != len(lhs_batch):
    msg = ("dot_general requires lhs batch dimensions to be distinct, got "
           f"lhs_batch {lhs_batch}.")
    raise TypeError(msg)
  if len(rhs_batch_set) != len(rhs_batch):
    msg = ("dot_general requires rhs batch dimensions to be distinct, got "
           f"rhs_batch {rhs_batch}.")
    raise TypeError(msg)
  if len(lhs_contracting_set) != len(lhs_contracting):
    msg = ("dot_general requires lhs contracting dimensions to be distinct, "
           f"got lhs_contracting {lhs_contracting}.")
    raise TypeError(msg)
  if len(rhs_contracting_set) != len(rhs_contracting):
    msg = ("dot_general requires rhs contracting dimensions to be distinct, "
           f"got rhs_contracting {rhs_contracting}.")
    raise TypeError(msg)
  if lhs_contracting_set & lhs_batch_set:
    msg = ("dot_general requires lhs batch dimensions to be disjoint from "
           "contracting dimensions, got lhs_batch {} and lhs_contracting {}.")
    raise TypeError(msg.format(lhs_batch, lhs_contracting))
  if rhs_contracting_set & rhs_batch_set:
    msg = ("dot_general requires rhs batch dimensions to be disjoint from "
           "contracting dimensions, got rhs_batch {} and rhs_contracting {}.")
    raise TypeError(msg.format(rhs_batch, rhs_contracting))
  lhs_batch_shape = tuple(lhs.shape[i] for i in lhs_batch)
  rhs_batch_shape = tuple(rhs.shape[i] for i in rhs_batch)
  if not core.symbolic_equal_shape(lhs_batch_shape, rhs_batch_shape):
    msg = ("dot_general requires lhs batch dimensions and rhs batch dimensions "
           "to have the same shape, got {} and {}.")
    raise TypeError(msg.format(lhs_batch_shape, rhs_batch_shape))
  lhs_contracting_shape = tuple(lhs.shape[i] for i in lhs_contracting)
  rhs_contracting_shape = tuple(rhs.shape[i] for i in rhs_contracting)
  if not core.symbolic_equal_shape(lhs_contracting_shape, rhs_contracting_shape):
    msg = ("dot_general requires contracting dimensions to have the same "
           "shape, got {} and {}.")
    raise TypeError(msg.format(lhs_contracting_shape, rhs_contracting_shape))

  return _dot_general_shape_computation(lhs.shape, rhs.shape, dimension_numbers)

def _dot_general_shape_computation(lhs_shape, rhs_shape, dimension_numbers):
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  batch_shape = tuple(lhs_shape[i] for i in lhs_batch)
  lhs_contract_or_batch = tuple(sorted(tuple(lhs_contracting) + tuple(lhs_batch)))
  lhs_tensored_shape = tuple_delete(lhs_shape, lhs_contract_or_batch)
  rhs_contract_or_batch = tuple(sorted(tuple(rhs_contracting) + tuple(rhs_batch)))
  rhs_tensored_shape = tuple_delete(rhs_shape, rhs_contract_or_batch)
  return batch_shape + lhs_tensored_shape + rhs_tensored_shape

def tuple_delete(tup, idx):
  idx_ = set(idx)
  return tuple(tup[i] for i in range(len(tup)) if i not in idx_)


def _dot_general_dtype_rule(lhs, rhs, *, dimension_numbers, precision,
                            preferred_element_type: Optional[DType]):
  input_dtype = naryop_dtype_rule(_input_dtype, [_any, _any], 'dot_general', lhs, rhs)
  if preferred_element_type is None:
    return input_dtype
  _validate_preferred_element_type(input_dtype, preferred_element_type)
  return preferred_element_type

def _dot_general_transpose_lhs(g, y, *, dimension_numbers, precision,
                               preferred_element_type: Optional[DType],
                               swap_ans=False):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  x_ndim = g.ndim - y.ndim + len(x_batch) + 2 * len(x_contract)
  x_kept = remaining(range(x_ndim), x_contract, x_batch)
  y_kept = remaining(range(y.ndim), y_contract, y_batch)
  if swap_ans:
    ans_batch, ans_y, _ = ranges_like(x_batch, y_kept, x_kept)
  else:
    ans_batch, _, ans_y = ranges_like(x_batch, x_kept, y_kept)
  dims = ((ans_y, y_kept), (ans_batch, y_batch))
  x_contract_sorted_by_y = list(np.take(x_contract, np.argsort(y_contract)))  # type: ignore[arg-type]
  out_axes = np.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)
  return transpose(dot_general(g, y, dims, precision=precision, preferred_element_type=preferred_element_type),
                   tuple(out_axes))

def _dot_general_transpose_rhs(g, x, *, dimension_numbers, precision,
                               preferred_element_type: Optional[DType]):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
  return _dot_general_transpose_lhs(
    g, x, dimension_numbers=swapped_dimension_numbers, precision=precision,
    preferred_element_type=preferred_element_type,
    swap_ans=True)


def _dot_general_batch_rule(batched_args, batch_dims, *, dimension_numbers,
                            precision,
                            preferred_element_type: Optional[DType]):
  lhs, rhs = batched_args
  new_dimension_numbers, result_batch_dim = _dot_general_batch_dim_nums(
      (lhs.ndim, rhs.ndim), batch_dims, dimension_numbers)
  batched_out = dot_general(lhs, rhs, new_dimension_numbers,
                            precision=precision,
                            preferred_element_type=preferred_element_type)
  return batched_out, result_batch_dim

def _dot_general_batch_dim_nums(ndims, batch_dims, dimension_numbers):
  # there are three kinds of dimensions in a dot_general:
  # - contraction dimensions appear in lhs and rhs but not the result
  # - batch dimensions appear in lhs, rhs, and result
  # - tensor product dimensions appear in the result and one of lhs or rhs
  lhs_ndim, rhs_ndim = ndims
  lbd, rbd = batch_dims
  assert lbd is not None or rbd is not None
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  def bump_dims(dims, b):
    return tuple(np.add(dims, np.greater_equal(dims, b)))

  if lbd is not None and rbd is not None:
    # adding a batch dimension
    lhs_batch = (lbd,) + bump_dims(lhs_batch, lbd)
    rhs_batch = (rbd,) + bump_dims(rhs_batch, rbd)
    lhs_contract = bump_dims(lhs_contract, lbd)
    rhs_contract = bump_dims(rhs_contract, rbd)
    result_batch_dim = 0
  else:
    # adding a tensor product dimension
    if lbd is not None:
      other = tuple(d for d in range(lhs_ndim)
                    if d not in lhs_batch and d not in lhs_contract)
      result_batch_dim = (len(lhs_batch) + sum(np.less(other, lbd)))
      lhs_batch = bump_dims(lhs_batch, lbd)
      lhs_contract = bump_dims(lhs_contract, lbd)
    else:
      other = tuple(d for d in range(rhs_ndim)
                    if d not in rhs_batch and d not in rhs_contract)
      result_batch_dim = (lhs_ndim - len(lhs_contract) +
                          sum(np.less(other, rbd)))
      rhs_batch = bump_dims(rhs_batch, rbd)
      rhs_contract = bump_dims(rhs_contract, rbd)

  new_dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
  return new_dimension_numbers, int(result_batch_dim)

def _dot_general_padding_rule(in_avals, out_avals, lhs, rhs, *,
                              dimension_numbers, **params):
  lhs_aval, _ = in_avals
  (lhs_contract, _), _ = dimension_numbers
  padded_axes = [(i, lhs_aval.shape[i].val) for i in lhs_contract
                 if isinstance(lhs_aval.shape[i], pe.BoundedAxisSize)]
  lhs_ = _replace_masked_values(lhs, 0, padded_axes)
  return [dot_general(lhs_, rhs, dimension_numbers=dimension_numbers, **params)]

def _dot_general_pp_rule(eqn, context, settings):
  # suppress printing precision or preferred_element_type when None.
  # print dimension_numbers as list-of-lists to be shorter.
  printed_params = {k: v for k, v in eqn.params.items() if v is not None}
  (lhs_cont, rhs_cont), (lhs_batch, rhs_batch) = eqn.params['dimension_numbers']
  printed_params['dimension_numbers'] = (
      (list(lhs_cont), list(rhs_cont)), (list(lhs_batch), list(rhs_batch)))
  lhs = core.pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name),
         core.pp_kv_pairs(sorted(printed_params.items()), context, settings),
         pp.text(" ") + core.pp_vars(eqn.invars, context)]
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  return [lhs, pp.text(" = ", annotation=annotation), *rhs]


dot_general_p = standard_primitive(_dot_general_shape_rule,
                                   _dot_general_dtype_rule, 'dot_general')
ad.defbilinear(dot_general_p,
               _dot_general_transpose_lhs, _dot_general_transpose_rhs)
batching.primitive_batchers[dot_general_p] = _dot_general_batch_rule
pe.padding_rules[dot_general_p] = _dot_general_padding_rule
# TODO(mattjj): un-comment the next line
# core.pp_eqn_rules[dot_general_p] = _dot_general_pp_rule

def precision_attr(precision: PrecisionType) -> ir.ArrayAttr:
  if precision is None:
    full_precision = (Precision.DEFAULT, Precision.DEFAULT)
  elif not isinstance(precision, tuple):
    full_precision = (precision, precision)
  else:
    full_precision = precision
  return ir.ArrayAttr.get(
      [mhlo.PrecisionAttr.get(str(p)) for p in full_precision])



def _dot_general_lower(ctx, lhs, rhs, *, dimension_numbers,
                       precision, preferred_element_type: Optional[np.dtype]):
  del preferred_element_type  # Implied by the output aval
  lhs_aval, rhs_aval = ctx.avals_in
  aval_out, = ctx.avals_out
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

  # TODO(b/195364460): Work around slow XLA/CPU implementation of float16 matmul
  if ctx.module_context.platform == "cpu":
    if lhs_aval.dtype == np.float16:
      f32 = mlir.dtype_to_ir_type(np.dtype(np.float32))
      lhs = mhlo.ConvertOp(ir.RankedTensorType.get(lhs_aval.shape, f32),
                           lhs).result
    if rhs_aval.dtype == np.float16:
      f32 = mlir.dtype_to_ir_type(np.dtype(np.float32))
      rhs = mhlo.ConvertOp(ir.RankedTensorType.get(rhs_aval.shape, f32),
                           rhs).result
  dot_dnums = mhlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=list(lhs_batch),
      rhs_batching_dimensions=list(rhs_batch),
      lhs_contracting_dimensions=list(lhs_contracting),
      rhs_contracting_dimensions=list(rhs_contracting))
  return [
      mhlo.DotGeneralOp(
          mlir.aval_to_ir_type(aval_out),
          lhs,
          rhs,
          dot_dnums,
          precision_config=precision_attr(precision)).result
  ]

mlir.register_lowering(dot_general_p, _dot_general_lower)


def _broadcast_in_dim_shape_rule(operand, *, shape, broadcast_dimensions):
  _check_shapelike('broadcast_in_dim', 'shape', shape)
  _check_shapelike('broadcast_in_dim', 'broadcast_dimensions',
                   broadcast_dimensions)
  operand_ndim = np.ndim(operand)
  if operand_ndim != len(broadcast_dimensions):
    msg = ('broadcast_in_dim broadcast_dimensions must have length equal to '
           'operand ndim; got broadcast_dimensions {} for operand ndim {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand_ndim))
  if len(shape) < operand_ndim:
    msg = ('broadcast_in_dim target broadcast shape must have equal or higher rank '
           'to the operand shape; got operand ndim {} and target broadcast ndim {}.')
    raise TypeError(msg.format(operand_ndim, len(shape)))
  if not set(broadcast_dimensions).issubset(set(range(len(shape)))):
    msg = ('broadcast_in_dim broadcast_dimensions must be a subset of output '
           'dimensions, got {} for operand ndim {} and shape {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand_ndim, shape))
  if not all(core.symbolic_equal_one_of_dim(operand.shape[i],
                                            [1, shape[broadcast_dimensions[i]]])
             for i in range(operand_ndim)):
    msg = (
        "broadcast_in_dim operand dimension sizes must either be 1, or be "
        "equal to their corresponding dimensions in the target broadcast "
        "shape; got operand of shape {}, target broadcast shape {}, "
        "broadcast_dimensions {} ")
    raise TypeError(msg.format(operand.shape, shape, broadcast_dimensions))
  if (len(broadcast_dimensions) != len(set(broadcast_dimensions)) or
      tuple(broadcast_dimensions) != tuple(sorted(broadcast_dimensions))):
    msg = ("broadcast_in_dim broadcast_dimensions must be strictly increasing; "
           "got broadcast_dimensions {}")
    raise TypeError(msg.format(broadcast_dimensions))

  return shape

def _broadcast_in_dim_typecheck_rule(
    operand, *dyn_shape, shape, broadcast_dimensions):
  if not dyn_shape:
    out_aval, effects = broadcast_in_dim_p.abstract_eval(
        operand.aval, shape=shape, broadcast_dimensions=broadcast_dimensions)
    return [out_aval], effects
  else:
    # TODO(mattjj): perform more checks like _broadcast_in_dim_shape_rule
    out_shape = _merge_dyn_shape(shape, dyn_shape)
    out_shape = [x.val if type(x) is core.Literal else x for x in out_shape]
    out_aval = core.DShapedArray(tuple(out_shape), operand.aval.dtype,
                                 operand.aval.weak_type)
    return [out_aval], core.no_effects

def _broadcast_in_dim_transpose_rule(ct, operand, *dyn_shape,
                                     shape, broadcast_dimensions):
  if type(ct) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)]
  unit_dims = [i for i, s in enumerate(operand.aval.shape)
               if core.symbolic_equal_dim(s,  1)]
  bdims = tuple(np.delete(broadcast_dimensions, unit_dims))
  axes = tuple(np.delete(range(len(shape)), bdims))
  return ([expand_dims(_reduce_sum(ct, axes), unit_dims)] +
          [None] * len(dyn_shape))

def _broadcast_in_dim_batch_rule(batched_args, batch_dims, *, shape,
                                 broadcast_dimensions):
  operand, = batched_args
  bdim, = batch_dims
  new_operand = batching.moveaxis(operand, bdim, 0)
  new_shape = (operand.shape[bdim],) + shape
  new_broadcast_dimensions = (0,) + tuple(np.add(1, broadcast_dimensions))
  return broadcast_in_dim(new_operand, new_shape, new_broadcast_dimensions), 0

def _broadcast_in_dim_fwd_rule(eqn):
  v, *dyn = eqn.invars
  if not dyn and core.symbolic_equal_shape(eqn.params['shape'], v.aval.shape):
    return [v], None
  else:
    return [None], eqn

def _broadcast_in_dim_staging_rule(
    trace, x, *dyn, shape, broadcast_dimensions):
  params = dict(shape=shape, broadcast_dimensions=broadcast_dimensions)
  if not dyn:
    return trace.default_process_primitive(broadcast_in_dim_p, (x,), params)
  aval = core.DShapedArray(_merge_dyn_shape(shape, dyn), x.dtype, x.weak_type)
  return _dyn_shape_staging_rule(trace, broadcast_in_dim_p, aval, x, *dyn,
                                 **params)

def _broadcast_in_dim_padding_rule(in_avals, out_avals, x, *dyn_shape,
                                   shape, broadcast_dimensions):
  del in_avals, dyn_shape
  out_aval, = out_avals
  new_shape = []
  new_dyn_shape = []
  for d in out_aval.shape:
    if type(d) is pe.BoundedAxisSize:
      new_shape.append(d.bound)
    elif type(d) is int:
      new_shape.append(d)
    else:
      assert isinstance(d, core.Tracer)
      new_shape.append(None)
      new_dyn_shape.append(d)
  return [broadcast_in_dim_p.bind(x, *new_dyn_shape, shape=tuple(new_shape),
                                  broadcast_dimensions=broadcast_dimensions)]

def _broadcast_in_dim_jvp_rule(primals, tangents, *, shape, broadcast_dimensions):
  operand, *dyn_shape = primals
  operand_dot, *_ = tangents
  y = broadcast_in_dim_p.bind(operand, *dyn_shape, shape=shape,
                              broadcast_dimensions=broadcast_dimensions)
  if type(operand_dot) is ad_util.Zero:
    y_dot = ad_util.Zero.from_value(y)
  else:
    y_dot = broadcast_in_dim_p.bind(operand_dot, *dyn_shape, shape=shape,
                                    broadcast_dimensions=broadcast_dimensions)
  return y, y_dot

def _broadcast_in_dim_partial_eval(
    trace, operand, *dyn_shape, shape, broadcast_dimensions):
  if not dyn_shape:
    return trace.default_process_primitive(
        broadcast_in_dim_p, (operand, *dyn_shape),
        dict(shape=shape, broadcast_dimensions=broadcast_dimensions))
  assert all(t.pval.is_known() for t in dyn_shape)
  operand_tracer = trace.instantiate_const(operand)
  dyn_shape_tracers = map(trace.instantiate_const, dyn_shape)
  dyn_shape_tracers_ = iter(dyn_shape_tracers)
  shape_ = [next(dyn_shape_tracers_) if d is None else d for d in shape]
  out_aval = core.DShapedArray(tuple(shape_), operand.dtype, operand.weak_type)
  out_tracer = pe.JaxprTracer(trace, pe.PartialVal.unknown(out_aval), None)
  eqn = pe.new_eqn_recipe(
      [operand_tracer, *dyn_shape_tracers], [out_tracer], broadcast_in_dim_p,
      dict(shape=shape, broadcast_dimensions=broadcast_dimensions),
      core.no_effects, source_info_util.current())
  out_tracer.recipe = eqn
  return out_tracer

def _broadcast_in_dim_lower(ctx, x, *dyn_shape, shape, broadcast_dimensions):
  aval_out, = ctx.avals_out
  if core.is_opaque_dtype(aval_out.dtype):
    return aval_out.dtype._rules.broadcast_in_dim_mlir(
        ctx, x, *dyn_shape, shape=shape,
        broadcast_dimensions=broadcast_dimensions)
  if dyn_shape:
    shape = _merge_dyn_shape(shape, dyn_shape)
    return mhlo.DynamicBroadcastInDimOp(
        mlir.aval_to_ir_type(aval_out), x,
        mlir.shape_tensor(shape),
        mlir.dense_int_elements(broadcast_dimensions),
    ).results
  else:
    return mhlo.BroadcastInDimOp(
        mlir.aval_to_ir_type(aval_out), x,
        mlir.dense_int_elements(broadcast_dimensions)
    ).results

def _broadcast_in_dim_pp_rule(eqn, context, settings):
  # Don't print shape or trivial broadcast_dimensions in params, since it can be
  # inferred from the let-binder's type annotation.
  printed_params = {}
  if eqn.params['broadcast_dimensions']:
    printed_params['broadcast_dimensions'] = eqn.params['broadcast_dimensions']
  lhs = core.pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name),
         core.pp_kv_pairs(sorted(printed_params.items()), context, settings),
         pp.text(" ") + core.pp_vars(eqn.invars[:1], context)]
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  return [lhs, pp.text(" = ", annotation=annotation), *rhs]

def _broadcast_in_dim_abstract_eval(x, *dyn_shape, shape, broadcast_dimensions):
  if not any(isinstance(d, core.BInt) for d in shape):
    shape = _broadcast_in_dim_shape_rule(  # error checking
        x, shape=shape, broadcast_dimensions=broadcast_dimensions)
    return core.ShapedArray(shape, x.dtype, x.weak_type, x.named_shape)
  # If any BInts in shape, produce a DShapedArray (even if x is a ShapedArray)
  # TODO(mattjj): unify DShapedArray with ShapedArray, and remove this code
  return core.DShapedArray(shape, x.dtype, x.weak_type)

broadcast_in_dim_p = standard_primitive(
    _broadcast_in_dim_shape_rule, _input_dtype, 'broadcast_in_dim')
broadcast_in_dim_p.def_abstract_eval(_broadcast_in_dim_abstract_eval)
ad.primitive_jvps[broadcast_in_dim_p] = _broadcast_in_dim_jvp_rule
ad.primitive_transposes[broadcast_in_dim_p] = _broadcast_in_dim_transpose_rule
batching.primitive_batchers[broadcast_in_dim_p] = _broadcast_in_dim_batch_rule
pe.forwarding_rules[broadcast_in_dim_p] = _broadcast_in_dim_fwd_rule
pe.custom_partial_eval_rules[broadcast_in_dim_p] = _broadcast_in_dim_partial_eval
pe.custom_staging_rules[broadcast_in_dim_p] = _broadcast_in_dim_staging_rule
pe.padding_rules[broadcast_in_dim_p] = _broadcast_in_dim_padding_rule
core.custom_typechecks[broadcast_in_dim_p] = _broadcast_in_dim_typecheck_rule
mlir.register_lowering(broadcast_in_dim_p, _broadcast_in_dim_lower)
# TODO(mattjj): un-comment the next line
# core.pp_eqn_rules[broadcast_in_dim_p] = _broadcast_in_dim_pp_rule


def _clamp_shape_rule(min, operand, max):
  if min.shape and min.shape != operand.shape:
    raise TypeError("clamp requires min.shape == operand.shape or min.shape == "
                    f"(), got min.shape={min.shape}, "
                    f"operand.shape={operand.shape}.")
  if max.shape and max.shape != operand.shape:
    raise TypeError("clamp requires max.shape == operand.shape or max.shape == "
                    f"(), got max.shape={max.shape}, "
                    f"operand.shape={operand.shape}.")
  return operand.shape

_clamp_dtype_rule = partial(naryop_dtype_rule, _input_dtype, [_any, _any, _any],
                            'clamp')

def _clamp_batch_rule(batched_args, batch_dims, **params):
  min, x, max = batched_args
  min_bdim, x_bdim, max_bdim = batch_dims
  size = next(x.shape[i] for x, i in zip(batched_args, batch_dims)
              if i is not None)

  # avoid transposes and some broadcasts in special cases
  if min_bdim == x_bdim == max_bdim:
    if np.shape(min) == np.shape(x) == np.shape(max):
      return clamp_p.bind(min, x, max), x_bdim
    elif np.ndim(min) == np.ndim(max) == 0:
      return clamp_p.bind(min, x, max), x_bdim
    elif np.ndim(min) == np.ndim(max) == 1:
      min = broadcast_in_dim(min, x.shape, [min_bdim])
      max = broadcast_in_dim(max, x.shape, [max_bdim])
      return clamp_p.bind(min, x, max), x_bdim
  elif np.ndim(min) == 0 and np.ndim(max) == 0 and x_bdim is not None:
    return clamp_p.bind(min, x, max), x_bdim

  min = batching.bdim_at_front(min, min_bdim, size) if np.shape(min) else min
  max = batching.bdim_at_front(max, max_bdim, size) if np.shape(max) else max
  x = batching.bdim_at_front(x, x_bdim, size) if np.shape(x) else x
  if np.ndim(min) == 0 and np.ndim(x) > 0:
    min = broadcast(min, x.shape)
  if np.ndim(max) == 0 and np.ndim(x) > 0:
    max = broadcast(max, x.shape)
  if 0 < np.ndim(min) < np.ndim(x):
    assert np.ndim(min) == 1, np.ndim(min)
    min = broadcast_in_dim(min, x.shape, [0])
  if 0 < np.ndim(max) < np.ndim(x):
    assert np.ndim(max) == 1, np.ndim(max)
    max = broadcast_in_dim(max, x.shape, [0])
  if np.ndim(min) > np.ndim(x):
    assert np.ndim(x) == 0, np.ndim(x)
    x = broadcast(x, min.shape)
  return clamp_p.bind(min, x, max), 0

clamp_p = standard_primitive(_clamp_shape_rule, _clamp_dtype_rule, 'clamp')
ad.defjvp(clamp_p,
          lambda g, min, operand, max:
          select(bitwise_and(gt(min, operand), lt(min, max)),
                 g, _zeros(operand)),
          lambda g, min, operand, max:
          select(bitwise_and(gt(operand, min), lt(operand, max)),
                 g, _zeros(operand)),
          lambda g, min, operand, max:
          select(lt(max, operand), g, _zeros(operand)))
batching.primitive_batchers[clamp_p] = _clamp_batch_rule
if mlir_api_version < 30:
  mlir.register_lowering(
      clamp_p, partial(_nary_lower_mhlo, mhlo.ClampOp, explicit_type=True))
else:
  mlir.register_lowering(
      clamp_p, partial(_nary_lower_mhlo, mhlo.ClampOp))
pe.def_trivial_padding(clamp_p)

def _concatenate_shape_rule(*operands, shape, **kwargs):
  return shape

def _concatenate_dtype_rule(*operands, **kwargs):
  _check_same_dtypes('concatenate', False, *(o.dtype for o in operands))
  return operands[0].dtype

def _concatenate_transpose_rule(t, *operands, dimension, shape):
  if _nr_dynamic_dimensions(shape) > 0:
    raise NotImplementedError()
  operand_shapes = [o.aval.shape if ad.is_undefined_primal(o) else o.shape
                    for o in operands]
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(o.aval) if ad.is_undefined_primal(o) else None
            for o in operands]
  else:
    limit_points = np.cumsum(
        [shape[dimension] for shape in operand_shapes]).tolist()
    starts = np.zeros((len(operands), t.ndim), dtype=int).tolist()
    limits = np.tile(t.shape, (len(operands), 1)).tolist()

    for i, s in enumerate(starts[1:]):
      s[dimension] = limit_points[:-1][i]
    for i, l in enumerate(limits):
      l[dimension] = limit_points[i]

    return [slicing.slice(t, start, limit) if ad.is_undefined_primal(o)
            else None for o, start, limit in zip(operands, starts, limits)]

def _concatenate_batch_rule(batched_args, batch_dims, *, dimension, shape):
  if _nr_dynamic_dimensions(shape) > 0:
    raise NotImplementedError()
  size = next(op.shape[bdim] for op, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  operands = [batching.moveaxis(op, bdim, 0) if bdim is not None
              else broadcast(op, (size,))
              for op, bdim in zip(batched_args, batch_dims)]
  return concatenate(operands, dimension + 1), 0

concatenate_p = standard_primitive(
    _concatenate_shape_rule, _concatenate_dtype_rule, 'concatenate')
ad.deflinear2(concatenate_p, _concatenate_transpose_rule)
ad.primitive_transposes[concatenate_p] = _concatenate_transpose_rule
batching.primitive_batchers[concatenate_p] = _concatenate_batch_rule

def _concatenate_staging_rule(
    trace, *args, dimension, shape):
  dyn = args[len(args) - _nr_dynamic_dimensions(shape):]
  params = dict(dimension=dimension, shape=shape)
  if not dyn:
    return trace.default_process_primitive(concatenate_p, args, params)
  aval = core.DShapedArray(_merge_dyn_shape(shape, dyn), args[0].dtype, args[0].weak_type)
  return _dyn_shape_staging_rule(trace, concatenate_p, aval, *args, **params)
pe.custom_staging_rules[concatenate_p] = _concatenate_staging_rule

def _concatenate_typecheck_rule(*args, dimension, shape):
  nr_dynamic_dimensions = _nr_dynamic_dimensions(shape)
  if not nr_dynamic_dimensions:
    out_aval, effects = concatenate_p.abstract_eval(
        *(a.aval for a in args), dimension=dimension, shape=shape)
    return [out_aval], effects
  else:
    dyn_shape = args[len(args) - nr_dynamic_dimensions:]
    out_shape = _merge_dyn_shape(shape, dyn_shape)
    out_shape = [x.val if type(x) is core.Literal else x for x in out_shape]
    out_aval = core.DShapedArray(tuple(out_shape), args[0].aval.dtype, args[0].aval.weak_type)
    return [out_aval], core.no_effects
core.custom_typechecks[concatenate_p] = _concatenate_typecheck_rule

def _concatenate_lower(ctx, *xs, dimension, shape):
  array_xs = xs[:len(xs) - _nr_dynamic_dimensions(shape)]
  return mhlo.ConcatenateOp(array_xs, mlir.i64_attr(dimension)).results

mlir.register_lowering(concatenate_p, _concatenate_lower)


def _pad_dtype_rule(operand, padding_value, *, padding_config):
  if operand.dtype != padding_value.dtype:
    msg = "pad operand and padding_value must be same dtype: got {} and {}."
    raise TypeError(msg.format(operand.dtype, padding_value.dtype))

  return _input_dtype(operand, padding_value)

def _pad_shape_rule(operand, padding_value, *, padding_config):
  del padding_value
  op_shape = np.shape(operand)
  if not len(padding_config) == np.ndim(operand):
    raise ValueError("length of padding_config must equal the number of axes "
                     f"of operand, got padding_config {padding_config} "
                     f"for operand shape {op_shape}")
  if not all(i >= 0 for _, _, i in padding_config):
    raise ValueError("interior padding in padding_config must be nonnegative, "
                     f"got padding_config {padding_config}")
  result = tuple(core.sum_dim(l, h, core.dilate_dim(d, i + 1))
                 for (l, h, i), d in zip(padding_config, op_shape))
  if not all(core.greater_equal_dim(d, 0) for d in result):
    msg = (f"Dimension size after padding is not at least 0, "
           f"got result shape {result}, for padding_config {padding_config}"
           f" and operand shape {op_shape}")
    raise ValueError(msg)
  return result

def _pad_transpose(t, operand, padding_value, *, padding_config):
  if type(t) is ad_util.Zero:
    t_operand = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    t_padv = ad_util.Zero(padding_value.aval) if ad.is_undefined_primal(padding_value) else None
  else:
    lo, hi, interior = util.unzip3(padding_config)
    total = lambda x: _reduce_sum(x, list(range(t.ndim)))

    def t_op():
      unpad_config = safe_zip(np.negative(lo), np.negative(hi),
                              np.zeros_like(interior))
      unpadded = pad(t, np.array(0., t.dtype), unpad_config)
      return slicing.slice(unpadded, np.zeros_like(lo), unpadded.shape,
                           np.add(interior, 1))

    t_operand = t_op() if ad.is_undefined_primal(operand) else None
    t_padv = sub(total(t), total(t_operand)) if ad.is_undefined_primal(padding_value) else None
  return [t_operand, t_padv]

def _pad_batch_rule(batched_args, batch_dims, *, padding_config):
  operand, padding_value = batched_args
  operand_bdim, padding_value_bdim = batch_dims
  if operand_bdim is None:
    operand_bdim = 0
    operand = broadcast(operand, (padding_value.shape[padding_value_bdim],))

  padding_config = list(padding_config)
  padding_config.insert(operand_bdim, (0, 0, 0))
  if padding_value_bdim is None:
    return pad(operand, padding_value, padding_config), operand_bdim

  assert padding_value_bdim == 0, padding_value_bdim

  x = pad(operand, _zero(operand), padding_config)
  mask = pad(full_like(operand, True, np.bool_), False, padding_config)
  broadcasted_padding = broadcast_in_dim(padding_value, x.shape,
                                         (operand_bdim,))
  return select(mask, x, broadcasted_padding), operand_bdim

pad_p = standard_primitive(_pad_shape_rule, _pad_dtype_rule, 'pad')
ad.deflinear2(pad_p, _pad_transpose)
batching.primitive_batchers[pad_p] = _pad_batch_rule

def _pad_lower(ctx, x, padding_value, *, padding_config):
  low, high, interior = util.unzip3(padding_config)
  return mhlo.PadOp(x, padding_value,
                    mlir.dense_int_elements(low),
                    mlir.dense_int_elements(high),
                    mlir.dense_int_elements(interior)).results
mlir.register_lowering(pad_p, _pad_lower)


# The squeeze primitive exists for the benefit of masking and other
# transformations that need to keep track of axis identity.
# For example, consider reshaping a 2D array with shape (1, N) into a 1D array
# with shape (N,). This results in the following JAXpr:
#   reshape[ dimension=None new_sizes=(N,) ]
# For N > 1, we can match up the output array axis with the second axis of the
# input. But for N = 1, it is not clear how axes match up: all we know from the
# JAXpr is that we are reshaping from (1, 1) to (1,).
# In constrast, squeeze[ dimensions=(0,) ] is unambiguous.


def _squeeze_dtype_rule(operand, *, dimensions):
  return operand.dtype

def _squeeze_shape_rule(operand, *, dimensions):
  return _compute_squeeze_shape(np.shape(operand), dimensions)

def _compute_squeeze_shape(shape, dimensions):
  dims_set = set(dimensions)
  if len(dims_set) != len(dimensions):
    raise ValueError(f"dimensions are not unique: {dimensions}")
  if not all(0 <= d < len(shape) for d in dims_set):
    raise ValueError(f"dimensions outside range [0, ndim): {dimensions}")
  if any(not core.symbolic_equal_dim(shape[d], 1) for d in dimensions):
    raise ValueError(
        "cannot select an axis to squeeze out which has size not equal to "
        f"one, got shape={shape} and dimensions={dimensions}")
  return tuple(s for i, s in enumerate(shape) if i not in dims_set)

def _squeeze_transpose_rule(t, operand, *, dimensions):
  assert ad.is_undefined_primal(operand)
  return [expand_dims(t, dimensions)]

def _squeeze_batch_rule(batched_args, batch_dims, *, dimensions):
  operand, = batched_args
  bdim, = batch_dims
  operand = batching.moveaxis(operand, bdim, 0)
  dimensions = tuple(np.add(1, dimensions))
  return squeeze(operand, dimensions=dimensions), 0

squeeze_p = standard_primitive(_squeeze_shape_rule, _squeeze_dtype_rule,
                               'squeeze')
ad.deflinear2(squeeze_p, _squeeze_transpose_rule)
batching.primitive_batchers[squeeze_p] = _squeeze_batch_rule

def _squeeze_lower(ctx, operand, *, dimensions):
  del dimensions  # Implied by the output aval.
  aval_out, = ctx.avals_out
  return mhlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), operand).results

mlir.register_lowering(squeeze_p, _squeeze_lower)



def shape_as_value(shape: core.Shape):
  """Converts a shape that may contain Poly values into a JAX value."""
  if len(shape) == 0:
    return full((0,), np.array(0, np.int64))
  dims = [
      expand_dims(convert_element_type(core.dimension_as_value(d), np.int64),
                  (0,))
      for d in shape
  ]
  return concatenate(dims, dimension=0)

def _is_singleton_reshape(old, new):
  # A singleton reshape is one where only singleton dimensions are added. We
  # want to detect them because they can be expressed as (lazy) broadcasts.
  old, new = iter(old), iter(new)
  d1, d2 = next(old, None), next(new, None)
  bcast_dims = []
  i = 0
  while True:
    if d1 is d2 is None:
      return bcast_dims
    elif d1 == d2:
      bcast_dims.append(i)
      i += 1
      d1, d2 = next(old, None), next(new, None)
    elif d2 == 1:
      i += 1
      d2 = next(new, None)
    else:
      return None

def _reshape_shape_rule(operand, *, new_sizes, dimensions):
  if not all(core.greater_equal_dim(d, 0) for d in new_sizes):
    msg = 'reshape new_sizes must all be positive, got {}.'
    raise TypeError(msg.format(new_sizes))
  # TODO(necula): re-enable this check
  if (not config.jax_dynamic_shapes and
      not core.same_shape_sizes(np.shape(operand), new_sizes)):
    msg = 'reshape total size must be unchanged, got new_sizes {} for shape {}.'
    raise TypeError(msg.format(new_sizes, np.shape(operand)))
  if dimensions is not None:
    if set(dimensions) != set(range(np.ndim(operand))):
      msg = ('reshape dimensions must be a permutation of operand dimensions, '
             'got dimensions {} for shape {}.')
      raise TypeError(msg.format(dimensions, np.shape(operand)))
  return tuple(new_sizes)

def _reshape_typecheck_rule(operand, *dyn_shape, new_sizes, dimensions):
  if not dyn_shape:
    out_aval, effects = reshape_p.abstract_eval(
        operand.aval, new_sizes=new_sizes, dimensions=dimensions)
    return [out_aval], effects
  else:
    # TODO(mattjj, necula): perform more checks like _reshape_shape_rule
    out_shape = _merge_dyn_shape(new_sizes, dyn_shape)
    out_shape = [x.val if type(x) is core.Literal else x for x in out_shape]
    out_aval = core.DShapedArray(tuple(out_shape), operand.aval.dtype,
                                 operand.aval.weak_type)
    return [out_aval], core.no_effects


def _reshape_dtype_rule(operand, *, new_sizes, dimensions):
  return operand.dtype

def _reshape_transpose_rule(t, operand, *, new_sizes, dimensions):
  assert ad.is_undefined_primal(operand)
  if dimensions is None:
    return [reshape(t, operand.aval.shape)]
  else:
    return [transpose(reshape(t, np.take(operand.aval.shape, dimensions)),
                      np.argsort(dimensions))]

def _reshape_batch_rule(batched_args, batch_dims, *, new_sizes, dimensions):
  operand, = batched_args
  bdim, = batch_dims
  operand = batching.moveaxis(operand, bdim, 0)
  if dimensions is not None:
    dimensions = (0,) + tuple(np.add(1, dimensions))
  return reshape(operand, operand.shape[:1] + new_sizes, dimensions), 0


def _reshape_lower(ctx, x, *dyn_shape, new_sizes, dimensions):
  aval_out, = ctx.avals_out
  if dimensions is not None:
    x = mhlo.TransposeOp(x, mlir.dense_int_elements(dimensions)).result
  if dyn_shape:
    shape = _merge_dyn_shape(new_sizes, dyn_shape)
    return mhlo.DynamicReshapeOp(
        mlir.aval_to_ir_type(aval_out), x,
        mlir.shape_tensor(shape),
    ).results
  else:
    return mhlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), x).results

def _reshape_staging_rule(
    trace, x, *dyn, new_sizes, dimensions):
  params = dict(new_sizes=new_sizes, dimensions=dimensions)
  if not dyn:
    return trace.default_process_primitive(reshape_p, (x,), params)
  av = core.DShapedArray(_merge_dyn_shape(new_sizes, dyn), x.dtype, x.weak_type)
  return _dyn_shape_staging_rule(trace, reshape_p, av, x, *dyn, **params)

reshape_p = standard_primitive(_reshape_shape_rule, _reshape_dtype_rule,
                               'reshape')
ad.deflinear2(reshape_p, _reshape_transpose_rule)
batching.primitive_batchers[reshape_p] = _reshape_batch_rule
mlir.register_lowering(reshape_p, _reshape_lower)
core.custom_typechecks[reshape_p] = _reshape_typecheck_rule
pe.custom_staging_rules[reshape_p] = _reshape_staging_rule


def _rev_shape_rule(operand, *, dimensions):
  _check_shapelike('rev', 'dimensions', dimensions)
  if len(set(dimensions)) != len(dimensions):
    msg = 'rev dimensions must be unique, got {}.'
    raise TypeError(msg.format(dimensions))
  if dimensions and not _max(dimensions) < operand.ndim:
    msg = ('rev dimensions must all be less than operand ndim, got dimensions '
           '{} for operand ndim {}.')
    raise TypeError(msg.format(dimensions, operand.ndim))
  return operand.shape

def _rev_batch_rule(batched_args, batch_dims, *, dimensions):
  operand, = batched_args
  bdim, = batch_dims
  new_dimensions = [i + 1 if i >= bdim else i for i in dimensions]
  return rev(operand, new_dimensions), bdim

rev_p = standard_primitive(_rev_shape_rule, _input_dtype, 'rev')
ad.deflinear2(rev_p, lambda t, _, dimensions: [rev(t, dimensions)])
batching.primitive_batchers[rev_p] = _rev_batch_rule

def _rev_lower(ctx, x, *, dimensions):
  return mhlo.ReverseOp(x, mlir.dense_int_elements(dimensions)).results
mlir.register_lowering(rev_p, _rev_lower)


def _transpose_shape_rule(operand, *, permutation):
  if not isinstance(permutation, (tuple, list, np.ndarray)):
    msg = "transpose permutation must be a tuple/list/ndarray, got {}."
    raise TypeError(msg.format(type(permutation)))
  if tuple(sorted(permutation)) != tuple(range(operand.ndim)):
    msg = ("transpose permutation isn't a permutation of operand dimensions, "
           "got permutation {} for operand shape {}.")
    raise TypeError(msg.format(permutation, operand.shape))
  return tuple(operand.shape[old_idx] for old_idx in permutation)

def _transpose_batch_rule(batched_args, batch_dims, *, permutation):
  operand, = batched_args
  bdim, = batch_dims
  perm = (bdim,) + tuple(i if i < bdim else i+1 for i in permutation)
  return transpose(operand, perm), 0

def _transpose_lower(ctx, x, *, permutation):
  aval_out, = ctx.avals_out
  if core.is_opaque_dtype(aval_out.dtype):
    return aval_out.dtype._rules.transpose_mlir(ctx, x, permutation=permutation)
  return mhlo.TransposeOp(x, mlir.dense_int_elements(permutation)).results

transpose_p = standard_primitive(_transpose_shape_rule, _input_dtype,
                                 'transpose')
ad.deflinear2(transpose_p,
              lambda t, _, permutation: [transpose(t, np.argsort(permutation))])  # type: ignore[arg-type]
batching.primitive_batchers[transpose_p] = _transpose_batch_rule
mlir.register_lowering(transpose_p, _transpose_lower)
pe.def_trivial_padding(transpose_p)


def _select_shape_rule(which, *cases):
  if len(cases) == 0:
    raise TypeError("select must have at least one case")
  if any(case.shape != cases[0].shape for case in cases[1:]):
    msg = "select cases must have the same shapes, got [{}]."
    raise TypeError(msg.format(", ".join([str(c.shape) for c in cases])))
  if which.shape and which.shape != cases[0].shape:
    msg = ("select `which` must be scalar or have the same shape as cases, "
           "got `which` shape {} but case shape {}.")
    raise TypeError(msg.format(which.shape, cases[0].shape))
  return cases[0].shape

def _select_dtype_rule(which, *cases):
  _check_same_dtypes("select", False, *(c.dtype for c in cases))
  if (not dtypes.issubdtype(which.dtype, np.bool_) and
      not dtypes.issubdtype(which.dtype, np.integer)):
    raise TypeError("select `which` must be boolean or integer type, got "
                    f"{which.dtype}.")
  if dtypes.issubdtype(which.dtype, np.bool_) and len(cases) > 2:
    raise TypeError("select with boolean `which` cannot have > 2 cases.")
  return cases[0].dtype

def _select_weak_type_rule(which, *cases):
  return all(c.weak_type for c in cases)

def _select_transpose_rule(t, which, *cases):
  assert not ad.is_undefined_primal(which)
  if type(t) is ad_util.Zero:
    return [None] + [ad_util.Zero(c.aval) if ad.is_undefined_primal(c) else None
                     for c in cases]
  else:
    zeros = full_like(t, 0)
    return [None] + [
        select(eq(which, _const(which, i)), t, zeros)
        if ad.is_undefined_primal(case) else None
        for i, case in enumerate(cases)
    ]

def _select_batch_rule(batched_args, batch_dims, **unused_kwargs):
  which, *cases = batched_args
  which_bdim, *case_bdims = batch_dims
  size = next(x.shape[i] for x, i in zip(batched_args, batch_dims)
              if i is not None)

  # avoid transposes and some broadcasts in special cases
  if all(which_bdim == bdim for bdim in case_bdims):
    if np.shape(which) == np.shape(cases[0]):
      return select_n(which, *cases), which_bdim
    else:
      # vmapped function had a scalar which with nonscalar args
      assert np.ndim(which) == 1
      which = broadcast_in_dim(which, cases[0].shape, [which_bdim])
      return select_n(which, *cases), which_bdim
  elif np.ndim(which) == 0 and all(bdim is not None for bdim in case_bdims):
    if all(case_bdims[0] == bdim for bdim in case_bdims[1:]):
      return select_n(which, *cases), case_bdims[0]
    elif all(np.shape(cases[0]) == np.shape(c) for c in cases):
      bdim = case_bdims[0]
      other_cases = [batching.moveaxis(c, c_bdim, bdim)
                     for c, c_bdim in zip(cases[1:], case_bdims[1:])]
      return select_n(which, cases[0], *other_cases), bdim

  which = (batching.bdim_at_front(which, which_bdim, size) if np.shape(which)
           else which)
  if not all(() == np.shape(c) for c in cases):
    cases = [batching.bdim_at_front(c, bdim, size)
             for c, bdim in zip(cases, case_bdims)]
  assert all(np.shape(cases[0]) == np.shape(c) for c in cases[1:])
  if 0 < np.ndim(which) < np.ndim(cases[0]):
    # vmapped function had a scalar which with nonscalar args
    assert np.ndim(which) == 1
    which = broadcast_in_dim(which, cases[0].shape, [0])
  if np.ndim(which) > np.ndim(cases[0]):
    assert np.ndim(cases[0]) == 0
    cases = [broadcast(c, which.shape) for c in cases]
  return select_n(which, *cases), 0

def _select_jvp(primals, tangents):
  which, *case_primals = primals
  case_tangents = tangents[1:]
  out = select_n(which, *case_primals)
  if all(type(t) is ad_util.Zero for t in case_tangents):
    out_dot = ad_util.Zero(case_tangents[0].aval)
  else:
    z = _zeros(next(t for t in case_tangents if type(t) is not ad_util.Zero))
    case_tangents = [z if type(t) is ad_util.Zero else t for t in case_tangents]
    out_dot = select_n(which, *case_tangents)
  return out, out_dot

def _select_mhlo_lowering(ctx, which, *cases):
  which_aval = ctx.avals_in[0]
  if which_aval.dtype == np.dtype(np.bool_):
    assert len(cases) <= 2
    if len(cases) == 1: return cases
    return mhlo.SelectOp(which, cases[1], cases[0]).results

  if dtypes.issubdtype(which_aval.dtype, np.signedinteger):
    compare_type = 'SIGNED'
  else:
    compare_type = 'UNSIGNED'
  lt = 'LT'

  def _select(offset, cases):
    assert len(cases) > 0
    if len(cases) == 1:
      return cases[0]
    mid = len(cases) // 2
    pred = mlir.compare_mhlo(which,
                             mlir.full_like_aval(offset + mid, which_aval),
                             lt, compare_type)
    return mhlo.SelectOp(pred, _select(offset, cases[:mid]),
                         _select(offset + mid, cases[mid:])).result

  return [_select(0, cases)]

select_n_p = standard_primitive(
    _select_shape_rule, _select_dtype_rule, 'select_n',
    weak_type_rule=_select_weak_type_rule)
ad.primitive_jvps[select_n_p] = _select_jvp
ad.primitive_transposes[select_n_p] = _select_transpose_rule
batching.primitive_batchers[select_n_p] = _select_batch_rule
mlir.register_lowering(select_n_p, _select_mhlo_lowering)
pe.def_trivial_padding(select_n_p)


def _reduce_shape_rule(*avals, computation, jaxpr, consts, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  if any(arg.shape != () for arg in init_val_avals):
    init_val_shapes = [a.shape for a in init_val_avals]
    raise ValueError(f'reduce found non-scalar initial value: {init_val_shapes}')
  return [tuple(np.delete(op.shape, dimensions)) for op in operand_avals]

def _reduce_dtype_rule(*avals, computation, jaxpr, consts, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  operand_dtypes = [dtypes.canonicalize_dtype(op.dtype) for op in operand_avals]
  init_val_dtypes = [dtypes.canonicalize_dtype(init.dtype) for init in init_val_avals]
  if operand_dtypes != init_val_dtypes:
    raise TypeError(
        "reduce operand dtypes should match corresponding initial value dtypes, "
        f"got operands={operand_avals} and initial_values={init_val_avals}")
  return operand_dtypes

def _reduce_weak_type_rule(*avals, computation, jaxpr, consts, dimensions):
  operand_avals, init_val_avals = split_list(avals, [len(avals) // 2])
  return [op.weak_type and init_val.weak_type
          for op, init_val in safe_zip(operand_avals, init_val_avals)]

def _reduce_batch_rule(batched_args, batch_dims, *, computation, jaxpr,
                       consts, dimensions):
  # TODO(mattjj,frostig): use batch_jaxpr, delete computation (assumes poly??)
  num_operands = len(batched_args) // 2
  operands, init_values = split_list(batched_args, [num_operands])
  operand_bdims, init_value_bdims = split_list(batch_dims, [num_operands])
  if all(init_value_bdim is batching.not_mapped
         for init_value_bdim in init_value_bdims):
    size = next(x.shape[ax] for x, ax in zip(batched_args, batch_dims)
                if ax is not None)
    operands = [batching.bdim_at_front(arg, bdim, size)
                for arg, bdim in zip(operands, operand_bdims)]
    new_dimensions = [d + 1 for d in dimensions]
    new_operand_bdims = [0] * num_operands
    return reduce_p.bind(*(operands + init_values),
                         computation=computation,
                         dimensions=tuple(new_dimensions),
                         consts=consts,
                         jaxpr=jaxpr), new_operand_bdims
  else:
    raise NotImplementedError  # loop and stack

def _reduce_jvp(reducer, init_values, primals, tangents, axes):
  input_shape = np.array(primals[0].shape, dtype=np.int_)

  n = np.prod(input_shape[list(axes)])
  non_axes = np.delete(np.arange(len(input_shape)), axes)

  # Move the reduced axes to the front, and flatten them to 1D.
  permutation = axes + tuple(non_axes)
  new_shape = (n,) + tuple(input_shape[non_axes])
  primals = tuple(reshape(x, new_shape, permutation) for x in primals)
  tangents = tuple(reshape(t, new_shape, permutation) for t in tangents)

  for d in range(len(non_axes) + 1):
    reducer = api.vmap(reducer)
  def _reduce_tree(*xs, axis=0):
    """Reduce by repeatedly splitting the array and multiplying."""
    while xs[0].shape[axis] > 1:
      n = xs[0].shape[axis]
      n1 = (n + 1) // 2
      n2 = n - n1
      xs1 = [slicing.slice_in_dim(x, 0, n1) for x in xs]
      xs2 = [slicing.slice_in_dim(x, n1, None) for x in xs]
      if n2 != n1:
        paddings = [(0, 0, 0)] * len(xs[0].shape)
        paddings[axis] = (0, 1, 0)
        xs2 = [pad(x2, i, paddings) for x2, i in zip(xs2, init_values)]
      xs = reducer(*(xs1 + xs2))
    if xs[0].shape[axis] == 0:
      return [full(input_shape[non_axes], i) for i in init_values]
    return tuple(squeeze(x, (axis,)) for x in xs)

  return api.jvp(_reduce_tree, primals, tangents)

def _reduce_jvp_rule(primals, tangents, *, computation, jaxpr,
                     consts, dimensions):
  primal_xs, init_values = split_list(primals, [len(primals) // 2])
  tangent_xs, tangent_init = split_list(tangents, [len(tangents) // 2])
  # This test may be too strict, if a value is actually zero but we cannot prove
  # it is symbolically zero.
  if any(type(t) is not ad_util.Zero for t in tangent_init):
    raise NotImplementedError(
      "Gradient of general lax.reduce with non-zero tangents for "
      "initial values to reduction not implemented")
  reducer = core.jaxpr_as_fun(core.ClosedJaxpr(jaxpr, consts))
  return _reduce_jvp(reducer, init_values, primal_xs, tangent_xs, dimensions)

def _reduce_named_shape_rule(*avals, computation, jaxpr, consts, dimensions):
  # TODO(mattjj,frostig): see the TODOs noting limitations/assumptions in
  # _reduce_batching_rule. We're making the same assumptions here for now.
  num_operands = len(avals) // 2
  operand_avals, init_avals = split_list(avals, [num_operands])
  if any(a.named_shape for a in init_avals):
    raise NotImplementedError
  named_shapes = [a.named_shape for a in operand_avals]
  join = core.join_named_shapes(*(a.named_shape for a in operand_avals))
  return [join] * len(named_shapes)


reduce_p = core.Primitive('reduce')
reduce_p.multiple_results = True
reduce_p.def_impl(partial(xla.apply_primitive, reduce_p))
reduce_p.def_abstract_eval(
    partial(standard_multi_result_abstract_eval, reduce_p, _reduce_shape_rule,
            _reduce_dtype_rule, _reduce_weak_type_rule,
            _reduce_named_shape_rule))
batching.primitive_batchers[reduce_p] = _reduce_batch_rule
ad.primitive_jvps[reduce_p] = _reduce_jvp_rule

def _reduce_lower(ctx, *values, computation, jaxpr, consts, dimensions):
  assert all(isinstance(x, core.ShapedArray) for x in ctx.avals_in), ctx.avals_in
  operands, init_values = util.split_list(values, [len(values) // 2])
  init_value_avals = ctx.avals_in[len(values) // 2:]
  op = mhlo.ReduceOp([mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
                     operands, init_values, mlir.dense_int_elements(dimensions))
  ir_types = [mlir.aval_to_ir_type(aval) for aval in init_value_avals]
  reducer = op.regions[0].blocks.append(*(ir_types + ir_types))
  with ir.InsertionPoint(reducer):
    reducer_ctx = ctx.module_context.replace(name_stack=util.new_name_stack())
    if jaxpr.effects:
      raise NotImplementedError('Cannot lower effectful `reduce`.')
    out_nodes, _ = mlir.jaxpr_subcomp(reducer_ctx, jaxpr, mlir.TokenSet(), consts,
                                      *([a] for a in reducer.arguments))
    mhlo.ReturnOp(util.flatten(out_nodes))
  return op.results

mlir.register_lowering(reduce_p, _reduce_lower)


def _reduce_number_dtype_rule(name, operand, *args, **kw):
  if not dtypes.issubdtype(operand.dtype, np.number):
    raise TypeError("{} does not accept dtype {}. Accepted dtypes are subtypes "
                    "of number.".format(name, np.dtype(operand.dtype).name))
  return dtypes.canonicalize_dtype(operand.dtype)

def _reduce_sum_shape_rule(operand, *, axes):
  return _reduce_op_shape_rule(operand, axes=axes)

def _reduce_sum_transpose_rule(cotangent, operand, *, axes):
  assert ad.is_undefined_primal(operand)
  input_shape = operand.aval.shape
  broadcast_dimensions = tuple(np.delete(np.arange(len(input_shape)), axes))
  result = broadcast_in_dim(cotangent, input_shape, broadcast_dimensions)
  assert result.shape == input_shape
  return [result]

def _reduce_sum_padding_rule(in_avals, out_avals, operand, *, axes):
  del out_avals
  aval, = in_avals
  padded_axes = [(i, d.val) for i, d in enumerate(aval.shape)
                 if isinstance(d, pe.BoundedAxisSize)]
  masked_operand = _replace_masked_values(operand, 0, padded_axes)
  return [_reduce_sum(masked_operand, axes)]

def _replace_masked_values(x, val, padded_axes):
  if not padded_axes: return x
  dtype = dtypes._scalar_type_to_dtype(int)
  masks = [broadcasted_iota(dtype, x.shape, i) < d for i, d in padded_axes]
  return select(_reduce(operator.and_, masks), x, full_like(x, val))


reduce_sum_p = standard_primitive(
  _reduce_sum_shape_rule, partial(_reduce_number_dtype_rule, 'reduce_sum'),
  'reduce_sum')
ad.deflinear2(reduce_sum_p, _reduce_sum_transpose_rule)
batching.defreducer(reduce_sum_p)
pe.padding_rules[reduce_sum_p] = _reduce_sum_padding_rule


def _reduce_op_shape_rule(operand, *, axes, input_shape=None):
  del input_shape  # Unused.
  if len(axes) != len(set(axes)):
    raise ValueError(f"duplicate value in 'axes' of reduction: {axes}")
  if not all(0 <= a < operand.ndim for a in axes):
    raise ValueError(f"reduction axes {axes} contains out-of-bounds indices for {operand}.")
  axes = frozenset(axes)
  return tuple(d for i, d in enumerate(operand.shape) if i not in axes)

def _reduce_prod_jvp_rule(primals, tangents, *, axes):
  reducer = lambda x, y: [mul(x, y)]
  primals_out, tangents_out = _reduce_jvp(reducer, [_const(primals[0], 1)],
                                          primals, tangents, axes)
  return primals_out[0], tangents_out[0]

reduce_prod_p = standard_primitive(
  _reduce_op_shape_rule, partial(_reduce_number_dtype_rule, 'reduce_prod'),
  'reduce_prod')
ad.primitive_jvps[reduce_prod_p] = _reduce_prod_jvp_rule
batching.defreducer(reduce_prod_p)


def _reduce_chooser_shape_rule(operand, *, axes):
  return tuple(np.delete(operand.shape, axes))

def _reduce_chooser_jvp_rule(g, ans, operand, *, axes):
  # TODO(mattjj): an alternative is to use variadic reduce to compute the chosen
  # locations in a single pass (rather than comparing equality) and use a
  # gather, and/or even push along the chosen elements of g (b/112040122)
  shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
  location_indicators = convert_element_type(
      _eq_meet(operand, reshape(ans, shape)), g.dtype)
  counts = _reduce_sum(location_indicators, axes)
  return div(_reduce_sum(mul(g, location_indicators), axes), counts)

reduce_max_p = standard_primitive(_reduce_op_shape_rule, _input_dtype,
                                  'reduce_max')
ad.defjvp2(reduce_max_p, _reduce_chooser_jvp_rule)
batching.defreducer(reduce_max_p)

reduce_min_p = standard_primitive(_reduce_op_shape_rule, _input_dtype,
                                  'reduce_min')
ad.defjvp2(reduce_min_p, _reduce_chooser_jvp_rule)
batching.defreducer(reduce_min_p)


def _argminmax_shape_rule(operand, *, axes, index_dtype):
  axis, = axes
  if not (0 <= axis < len(operand.shape)):
    raise ValueError(f"Invalid axis {axis} for operand shape {operand.shape}")
  if not core.greater_equal_dim(operand.shape[axis], 1):
    raise ValueError("argmin and argmax require non-empty reduced dimension. "
                     f"operand.shape={operand.shape} axis={axis}")
  return tuple(np.delete(operand.shape, axis))

def _argminmax_dtype_rule(operand, *, axes, index_dtype):
  if not dtypes.issubdtype(index_dtype, np.integer):
    raise TypeError("index_dtype must be an integer type, but got {}"
                    .format(np.dtype(index_dtype).name))
  return index_dtype

def _compute_argminmax(value_comparator, get_identity,
                       operand, *, index_dtype, axes):
  # value_comparator is either lax.lt (for argmin) or lax.gt
  # get_identity(operand.dtype) is inf for argmin or -inf for argmax
  axis, = axes
  indices = broadcasted_iota(index_dtype, np.shape(operand), axis)
  def reducer_fn(op_val_index, acc_val_index):
    op_val, op_index = op_val_index
    acc_val, acc_index = acc_val_index
    # Pick op_val if Lt (for argmin) or if NaN
    pick_op_val = bitwise_or(value_comparator(op_val, acc_val),
                             ne(op_val, op_val))
    # If x and y are not NaN and x = y, then pick the first
    pick_op_index = bitwise_or(pick_op_val,
                               bitwise_and(eq(op_val, acc_val),
                                           lt(op_index, acc_index)))
    return (select(pick_op_val, op_val, acc_val),
            select(pick_op_index, op_index, acc_index))
  res = reduce([operand, indices],
               [get_identity(operand.dtype), np.array(0, index_dtype)],
               reducer_fn,
               axes)
  return res[1]

argmin_p = standard_primitive(_argminmax_shape_rule, _argminmax_dtype_rule,
                              'argmin', weak_type_rule=_strip_weak_type)
batching.defreducer(argmin_p)
ad.defjvp_zero(argmin_p)

argmax_p = standard_primitive(_argminmax_shape_rule, _argminmax_dtype_rule,
                              'argmax', weak_type_rule=_strip_weak_type)
batching.defreducer(argmax_p)
ad.defjvp_zero(argmax_p)

mlir.register_lowering(argmin_p, mlir.cache_lowering(mlir.lower_fun(
  partial(_compute_argminmax, lt, _get_min_identity),
  multiple_results=False)))

mlir.register_lowering(argmax_p, mlir.cache_lowering(mlir.lower_fun(
  partial(_compute_argminmax, gt, _get_max_identity),
  multiple_results=False)))


def _reduce_logical_shape_rule(operand, *, axes):
  if operand.dtype != np.bool_ and not np.issubdtype(operand.dtype, np.integer):
    raise TypeError(f"logical reduction requires operand dtype bool or int, got {operand.dtype}.")
  return tuple(np.delete(operand.shape, axes))

reduce_or_p = standard_primitive(
    _reduce_logical_shape_rule, _input_dtype, 'reduce_or',
    weak_type_rule=_strip_weak_type)
batching.defreducer(reduce_or_p)


reduce_and_p = standard_primitive(
    _reduce_logical_shape_rule, _input_dtype, 'reduce_and',
    weak_type_rule=_strip_weak_type)
batching.defreducer(reduce_and_p)


reduce_xor_p = standard_primitive(
    _reduce_logical_shape_rule, _input_dtype, 'reduce_xor',
    weak_type_rule=_strip_weak_type)
batching.defreducer(reduce_xor_p)


def _unary_reduce_lower(reducer, unit_factory, ctx, x, *, axes):
  aval_out, = ctx.avals_out
  dtype = aval_out.dtype
  op = mhlo.ReduceOp([mlir.aval_to_ir_type(aval_out)], [x],
                     mlir.ir_constants(unit_factory(aval_out.dtype)),
                     mlir.dense_int_elements(axes))
  scalar_type = mlir.aval_to_ir_type(core.ShapedArray((), dtype))
  reducer_region = op.regions[0].blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(reducer_region):
    add = reducer(*reducer_region.arguments)
    mhlo.ReturnOp(add.results)
  return op.results

mlir.register_lowering(reduce_sum_p, partial(_unary_reduce_lower, mhlo.AddOp,
                                         lambda dtype: np.array(0, dtype)))
mlir.register_lowering(reduce_prod_p, partial(_unary_reduce_lower, mhlo.MulOp,
                                          lambda dtype: np.array(1, dtype)))
mlir.register_lowering(reduce_or_p, partial(_unary_reduce_lower, mhlo.OrOp,
                                          _get_bitwise_or_identity))
mlir.register_lowering(reduce_and_p, partial(_unary_reduce_lower, mhlo.AndOp,
                                          _get_bitwise_and_identity))
mlir.register_lowering(reduce_xor_p, partial(_unary_reduce_lower, mhlo.XorOp,
                                          _get_bitwise_or_identity))
mlir.register_lowering(reduce_min_p, partial(_unary_reduce_lower, mlir.min_mhlo,
                                         _get_min_identity))
mlir.register_lowering(reduce_max_p, partial(_unary_reduce_lower, mlir.max_mhlo,
                                         _get_max_identity))




def _reduce_precision_shape_rule(operand, *, exponent_bits, mantissa_bits):
  exponent_bits = operator.index(exponent_bits)
  mantissa_bits = operator.index(mantissa_bits)
  if exponent_bits < 1:
    raise ValueError(f"reduce_precision: exponent_bits must be positive; got {exponent_bits}")
  if mantissa_bits < 0:
    raise ValueError(f"reduce_precision: mantissa_bits must be non-negative; got {mantissa_bits}")
  return operand.shape


reduce_precision_p = standard_primitive(
    _reduce_precision_shape_rule,
    partial(unop_dtype_rule, _identity, _float, 'reduce_precision'),
    name='reduce_precision')
batching.defvectorized(reduce_precision_p)

def _reduce_precision_lower(ctx, operand, *, exponent_bits, mantissa_bits):
  aval_out, = ctx.avals_out
  return mhlo.ReducePrecisionOp(operand, mlir.i32_attr(exponent_bits),
                                mlir.i32_attr(mantissa_bits)).results

mlir.register_lowering(reduce_precision_p, _reduce_precision_lower)



_UINT_DTYPES = {
  16: np.dtype(np.uint16),
  32: np.dtype(np.uint32),
  64: np.dtype(np.uint64),
}

_INT_DTYPES = {
  16: np.dtype(np.int16),
  32: np.dtype(np.int32),
  64: np.dtype(np.int64),
}


def _sort_abstract_eval(*args, **kwargs):
  args = tuple(raise_to_shaped(arg) for arg in args)
  if any(arg.shape != args[0].shape for arg in args[1:]):
    shapes = " ".join(str(a.shape) for a in args)
    raise TypeError(f"Arguments to sort must have equal shapes, got: {shapes}")
  return args


def _float_to_int_for_sort(x):
  # Switch from a floating point value to a integer value in such a way that
  # when using the integer value to compare, we get the same result for normal
  # values, and -nan is treated as the smallest value, and nan is treated as
  # the largest value.
  # If f is a float, and
  # x = bit_cast<int32>(f);
  # y = x < 0 ? int32_max - x : x;
  # then y is ordered as an int32 such that finite values have the obvious
  # order. In this scheme, -0 would be before 0, and -NaN and NaN appear at
  # the beginning and end of the ordering. This causes issues for stable
  # sorts, so we avoid this by standardizing the representation of zeros
  # and NaNs in the output.
  # Note that in order to avoid -x to overflow, we calculate
  # int32_max - x as unsigned, and then convert back to signed.
  if x.dtype == dtypes.bfloat16:
    x = convert_element_type(x, np.float32)
  nbits = np.finfo(x).bits
  signed_dtype = _INT_DTYPES[nbits]
  unsigned_dtype = _UINT_DTYPES[nbits]

  signed = bitcast_convert_type(x, signed_dtype)
  unsigned = bitcast_convert_type(x, unsigned_dtype)

  # We cannot standardize zeros in x because XLA elides this is some cases.
  # We cannot standardize NaNs in x because it triggers jax.debug_nans
  # So instead we do these replacements in the signed integer representation.

  # Standardize zeros:
  signed = select(eq(x, _zero(x)), _zeros(signed), signed)
  # Standardize nans:
  signed_nan = x.dtype.type(np.nan).view(signed_dtype)
  signed = select(_isnan(x), full_like(signed, signed_nan), signed)

  flipped = bitcast_convert_type(
    sub(unsigned_dtype.type(np.iinfo(signed_dtype).max), unsigned), signed_dtype)
  return select(lt(signed, _zero(signed)), flipped, signed)

# Default comparator that sorts the operands lexicographically on the
# first `num_keys` arguments.
# For floating point types, a total order is created where
# -infinity < ... < 0 < ... < infinity < NaN.
# 0.0 and -0.0 are treated as equivalent, as are all NaN representations.
# For complex types, the (real, imag) pairs are sorted lexicographically
# (following NumPy's semantics).
# This code adds complex-number support and lexicographic ordering to the algorithm from:
# https://github.com/tensorflow/tensorflow/blob/ba43780830f09da72081fe5061c436f1c6203a92/tensorflow/compiler/xla/client/lib/comparators.h#L33
def _sort_lt_comparator(*operands, num_keys=1):
  x_keys, y_keys = _operands_to_keys(*operands, num_keys=num_keys)
  p = None
  for xk, yk in zip(x_keys[::-1], y_keys[::-1]):
    p = (bitwise_or(lt(xk, yk), bitwise_and(eq(xk, yk), p)) if p is not None
         else lt(xk, yk))
  return p

# Similar to sort_lt_comparator, but implements less than or equal. Used by
# the searchsorted() implementation.
def _sort_le_comparator(*operands, num_keys=1):
  x_keys, y_keys = _operands_to_keys(*operands, num_keys=num_keys)
  p = None
  for xk, yk in zip(x_keys[::-1], y_keys[::-1]):
    p = (bitwise_or(lt(xk, yk), bitwise_and(eq(xk, yk), p)) if p is not None
         else le(xk, yk))
  return p

def _operands_to_keys(*operands, num_keys=1):
  assert len(operands) >= 2 and len(operands) % 2 == 0, operands
  assert len(operands) // 2 >= num_keys, (operands, num_keys)
  x_keys, y_keys = [], []
  for x, y in zip(operands[:2*num_keys:2], operands[1:2*num_keys:2]):
    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    if dtypes.issubdtype(x.dtype, np.complexfloating):
      x_keys.extend([_float_to_int_for_sort(real(x)), _float_to_int_for_sort(imag(x))])
      y_keys.extend([_float_to_int_for_sort(real(y)), _float_to_int_for_sort(imag(y))])
    elif dtypes.issubdtype(x.dtype, np.floating):
      x_keys.append(_float_to_int_for_sort(x))
      y_keys.append(_float_to_int_for_sort(y))
    else:
      x_keys.append(x)
      y_keys.append(y)
  return x_keys, y_keys


def _sort_jvp(primals, tangents, *, dimension, is_stable, num_keys):
  shape = primals[0].shape
  iotas = []
  for dim, size in enumerate(shape):
    dtype = np.int32 if size < np.iinfo(np.int32).max else np.int64
    iotas.append(broadcasted_iota(dtype, shape, dim))
  primals = sort_p.bind(*(primals + (iotas[dimension],)), dimension=dimension,
                        is_stable=is_stable, num_keys=num_keys)
  idx = tuple(primals[-1] if i == dimension else iotas[i]
              for i in range(len(shape)))
  tangents_out = tuple(t if type(t) is ad_util.Zero else t[idx] for t in tangents)
  return tuple(primals[:-1]), tangents_out

def _sort_batch_rule(batched_args, batch_dims, *, dimension, is_stable, num_keys):
  prototype_arg, new_bdim = next(
    (a, b) for a, b in zip(batched_args, batch_dims) if b is not None)
  new_args = []
  for arg, bdim in zip(batched_args, batch_dims):
    if bdim is None:
      dims = np.delete(np.arange(prototype_arg.ndim), new_bdim)
      new_args.append(broadcast_in_dim(arg, prototype_arg.shape, dims))
    else:
      new_args.append(batching.moveaxis(arg, bdim, new_bdim))
  new_dimension = dimension + (new_bdim <= dimension)
  bdims = (new_bdim,) * len(new_args)
  return (sort_p.bind(*new_args, dimension=new_dimension, is_stable=is_stable, num_keys=num_keys),
          bdims)


sort_p = Primitive('sort')
sort_p.multiple_results = True
sort_p.def_impl(partial(xla.apply_primitive, sort_p))
sort_p.def_abstract_eval(_sort_abstract_eval)
ad.primitive_jvps[sort_p] = _sort_jvp
batching.primitive_batchers[sort_p] = _sort_batch_rule


def _sort_lower(ctx, *operands, dimension, is_stable, num_keys):
  assert all(isinstance(x, core.ShapedArray) for x in ctx.avals_in), ctx.avals_in
  sort = mhlo.SortOp([mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
                     mlir.flatten_lowering_ir_args(operands),
                     dimension=mlir.i64_attr(dimension),
                     is_stable=ir.BoolAttr.get(is_stable))
  scalar_avals = [aval.update(shape=()) for aval in ctx.avals_in]
  scalar_types = safe_map(mlir.aval_to_ir_type, scalar_avals)
  comparator = sort.comparator.blocks.append(
      *util.flatten(zip(scalar_types, scalar_types)))
  with ir.InsertionPoint(comparator):
    lower_comparator = mlir.lower_fun(partial(_sort_lt_comparator),
                                      multiple_results=False)
    sub_ctx = ctx.replace(primitive=None,
                          avals_in=util.flatten(zip(scalar_avals, scalar_avals)),
                          avals_out=[core.ShapedArray((), np.bool_)])

    out = lower_comparator(sub_ctx, *[[a] for a in comparator.arguments],
                           num_keys=num_keys)
    mhlo.ReturnOp(util.flatten(out))
  return sort.results

mlir.register_lowering(sort_p, _sort_lower)


def _top_k_abstract_eval(operand, *, k):
  if k < 0:
    raise ValueError(f"k argument to top_k must be nonnegative, got {k}")
  if len(operand.shape) == 0:
    raise TypeError("top_k operand must have >= 1 dimension, got {}"
                    .format(operand.shape))
  shape = list(operand.shape)
  if shape[-1] < k:
    msg = "k argument to top_k must be no larger than minor dimension; {} vs {}"
    raise ValueError(msg.format(k, shape))
  shape[-1] = k
  return (operand.update(shape=shape, dtype=operand.dtype,
                         weak_type=operand.weak_type),
          operand.update(shape=shape, dtype=np.dtype(np.int32)))

def _top_k_jvp(primals, tangents, *, k):
  operand, = primals
  tangent, = tangents
  primals_out = top_k(operand, k)
  if type(tangent) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_value(primals_out[0])
  else:
    _, k_idxs = primals_out
    idx_shape = k_idxs.shape
    rank = len(idx_shape)
    gather_index_shape = idx_shape + (1,)
    gather_indices = []
    for i in range(rank-1):
      _iota = iota(k_idxs.dtype, idx_shape[i])
      _iota = broadcast_in_dim(_iota, gather_index_shape, (i,))
      gather_indices.append(_iota)
    gather_indices.append(reshape(k_idxs, gather_index_shape))
    gather_indices = concatenate(gather_indices, dimension=rank)
    slice_sizes = (1,) * rank
    dnums = slicing.GatherDimensionNumbers(
      offset_dims=(),
      collapsed_slice_dims=tuple(range(rank)),
      start_index_map=tuple(range(rank)))
    tangent_out = slicing.gather(tangent, gather_indices, dnums, slice_sizes)
  return primals_out, (tangent_out, ad_util.Zero.from_value(primals_out[1]))

def _top_k_batch_rule(batched_args, batch_dims, *, k):
  operand, = batched_args
  bdim, = batch_dims
  if bdim == operand.ndim-1:
    perm = np.arange(operand.ndim)
    perm[bdim-1], perm[bdim] = perm[bdim], perm[bdim-1]
    top_k_v, top_k_i = top_k(transpose(operand, perm), k=k)
    return (transpose(top_k_v, perm),
            transpose(top_k_i, perm)), (bdim, bdim)
  else:
    return top_k(operand, k=k), (bdim, bdim)

def _top_k_translation_rule(ctx, avals_in, avals_out, x, *, k):
  return xla.xla_destructure(ctx.builder, xops.TopK(x, k))

top_k_p = Primitive('top_k')
top_k_p.multiple_results = True
top_k_p.def_impl(partial(xla.apply_primitive, top_k_p))
top_k_p.def_abstract_eval(_top_k_abstract_eval)
def _top_k_lower(ctx, operand, k):
  return chlo.TopKOp(operand, mlir.i64_attr(k)).results
mlir.register_lowering(top_k_p, _top_k_lower)
ad.primitive_jvps[top_k_p] = _top_k_jvp
batching.primitive_batchers[top_k_p] = _top_k_batch_rule

def _stop_gradient_jvp_rule(primals, tangents):
  # if we don't call stop_gradient here, we'd only peel off one autodiff tracer
  x, = primals
  return stop_gradient(x), ad_util.Zero.from_value(x)

def _stop_gradient_batch_rule(batched_args, batch_dims):
  x, = batched_args
  dim, = batch_dims
  return stop_gradient(x), dim

ad.primitive_jvps[ad_util.stop_gradient_p] = _stop_gradient_jvp_rule
batching.primitive_batchers[ad_util.stop_gradient_p] = _stop_gradient_batch_rule


def create_token(_=None):
  """Creates an XLA token value with no preconditions for sequencing effects.

  Experimental.

  The argument is ignored. It exists for backward compatibility.
  """
  return create_token_p.bind()

create_token_p = Primitive("create_token")
create_token_p.def_impl(partial(xla.apply_primitive, create_token_p))
create_token_p.def_abstract_eval(lambda *_: abstract_token)

def _create_token_lowering(ctx, *operands):
  aval_out, = ctx.avals_out
  return mhlo.CreateTokenOp(mlir.aval_to_ir_type(aval_out)).results

mlir.register_lowering(create_token_p, _create_token_lowering)


def after_all(*operands):
  """Merges one or more XLA token values. Experimental.

  Wraps the XLA AfterAll operator."""
  return after_all_p.bind(*operands)

def _after_all_abstract_eval(*operands):
  if any(x is not abstract_token for x in operands):
    raise TypeError("Arguments to after_all must be tokens")
  return abstract_token


after_all_p = Primitive("after_all")
after_all_p.def_impl(partial(xla.apply_primitive, after_all_p))
after_all_p.def_abstract_eval(_after_all_abstract_eval)

def _after_all_lowering(ctx, *operands):
  aval_out, = ctx.avals_out
  return mhlo.AfterAllOp(mlir.aval_to_ir_type(aval_out), operands).results

mlir.register_lowering(after_all_p, _after_all_lowering)


InOutFeedEffect = enum.Enum('InOutFeedEffect', ['Infeed', 'Outfeed'])


def infeed(token, shape=None, partitions=None):
  """Consumes an infeed value of `shape` from the host. Experimental.

  `token` is used to sequence infeed and outfeed effects.
  `partitions` may be specified inside a `sharded_jit` function.
  """
  flat_shapes, treedef = pytree.flatten(shape)
  for shape in flat_shapes:
    if not isinstance(shape, ShapedArray):
      raise TypeError("shape argument to infeed must be a pytree of "
                      "ShapedArray values, got {}".format(shape))
  if partitions is not None:
    # Always replicate token.
    # We specifically use type() to raise an error for PartitionSpecs.
    if type(partitions) != tuple:  # pylint: disable=unidiomatic-typecheck
      raise ValueError(f"'partitions' argument to infeed should be a tuple, "
                       f"got {partitions}")
    partitions = partitions + (None,)
  xs_and_token = infeed_p.bind(token, shapes=tuple(flat_shapes),
                               partitions=partitions)
  return (treedef.unflatten(xs_and_token[:-1]), xs_and_token[-1])

def _infeed_abstract_eval(token, *, shapes, partitions):
  if token is not abstract_token:
    raise TypeError("First argument to infeed must be a token")
  return (*shapes, abstract_token), {InOutFeedEffect.Infeed}


infeed_p = Primitive("infeed")
infeed_p.multiple_results = True
infeed_p.def_impl(partial(xla.apply_primitive, infeed_p))
infeed_p.def_effectful_abstract_eval(_infeed_abstract_eval)
mlir.lowerable_effects.add(InOutFeedEffect.Infeed)


def _infeed_lowering(ctx, token, *, shapes, partitions):
  output_types = safe_map(mlir.aval_to_ir_types, ctx.avals_out[:-1])
  flat_output_types = util.flatten(output_types)
  # TODO(phawkins): verify `shapes` have a major-to-minor layout.
  layouts = ir.ArrayAttr.get([
      ir.ArrayAttr.get(
          [mlir.i64_attr(i)
           for i in range(len(aval.shape) - 1, -1, -1)])
      for aval in shapes
  ])
  infeed = mhlo.InfeedOp(
      flat_output_types + [mhlo.TokenType.get()],
      token,
      infeed_config=ir.StringAttr.get(''),
      layout=layouts)
  if partitions is not None:
    mlir.set_sharding(infeed, xla.sharding_to_proto(partitions))
  token = infeed.results[-1]
  outs = infeed.results[:-1]
  return util.unflatten(outs, safe_map(len, output_types)) + [[
      token,
  ]]

mlir.register_lowering(infeed_p, _infeed_lowering)


def outfeed(token, xs, partitions = None):
  """Outfeeds value `xs` to the host. Experimental.

  `token` is used to sequence infeed and outfeed effects.
  `partitions` may be specified inside a `sharded_jit` or `pjit` function.
  """
  if partitions is not None:
    # We specifically use type() to raise an error for PartitionSpecs.
    if type(partitions) != tuple:  # pylint: disable=unidiomatic-typecheck
      raise ValueError(f"'partitions' argument to outfeed should be a tuple, "
                       f"got {partitions}")
  flat_xs, _ = pytree.flatten(xs)
  return outfeed_p.bind(token, *flat_xs, partitions=partitions)

def _outfeed_abstract_eval(token, *xs, partitions):
  if token is not abstract_token:
    raise TypeError("First argument to outfeed must be a token")
  return abstract_token, {InOutFeedEffect.Outfeed}

outfeed_p = Primitive("outfeed")
outfeed_p.def_impl(partial(xla.apply_primitive, outfeed_p))
outfeed_p.def_effectful_abstract_eval(_outfeed_abstract_eval)
mlir.lowerable_effects.add(InOutFeedEffect.Outfeed)


def _outfeed_lowering(ctx, token, *xs, partitions):
  token_aval = ctx.avals_in[0]
  outfeed = mhlo.OutfeedOp(
      mlir.aval_to_ir_type(token_aval),
      mlir.flatten_lowering_ir_args(xs),
      token,
      outfeed_config=ir.StringAttr.get(''))
  if partitions is not None:
    mlir.set_sharding(outfeed, xla.sharding_to_proto(partitions))
  return outfeed.results

mlir.register_lowering(outfeed_p, _outfeed_lowering)


def rng_uniform(a, b, shape):
  """Stateful PRNG generator. Experimental and its use is discouraged.

  Returns uniformly distributed random numbers in the range [a, b)

  You should use jax.random for most purposes; this function exists only for
  niche use cases with special performance requirements.

  This API may be removed at any time.
  """
  return rng_uniform_p.bind(a, b, shape=tuple(shape))

def _rng_uniform_abstract_eval(a, b, *, shape):
  if a.dtype != b.dtype:
    raise ValueError(
      "Arguments to rng_uniform must have identical dtypes, got {} "
      "and {}.".format(a.dtype, b.dtype))
  if a.shape != () or b.shape != ():
    raise ValueError(
      "Arguments to rng_uniform must be scalars; got shapes {} and {}."
      .format(a.shape, b.shape))
  return a.update(shape=shape, dtype=a.dtype,
                  weak_type=(a.weak_type and b.weak_type))

rng_uniform_p = Primitive("rng_uniform")
rng_uniform_p.def_impl(partial(xla.apply_primitive, rng_uniform_p))
rng_uniform_p.def_abstract_eval(_rng_uniform_abstract_eval)

def _rng_uniform_lowering(ctx, a, b, *, shape):
  aval_out, = ctx.avals_out
  shape, = mlir.ir_constants(np.array(aval_out.shape, np.int64),
                             canonicalize_types=False)
  if mlir_api_version <= 22:
    return mhlo.RngUniformOp(a, b, shape).results
  else:
    return mhlo.RngOp(a, b, shape,
                      mhlo.RngDistributionAttr.get('UNIFORM')).results

mlir.register_lowering(rng_uniform_p, _rng_uniform_lowering)


def _rng_bit_generator_shape_rule(key, *, shape, dtype, algorithm):
  del dtype, algorithm
  return (key.shape, tuple(shape))

def _rng_bit_generator_dtype_rule(key, *, shape, dtype, algorithm):
  del shape, algorithm
  return (key.dtype, dtype)

def _rng_bit_generator_weak_type_rule(key, *, shape, dtype, algorithm):
  del shape, dtype, algorithm
  return (key.weak_type, False)

RandomAlgorithm = xops.RandomAlgorithm
RandomAlgorithm.__str__ = lambda algorithm: algorithm.name  # type: ignore[assignment]

def _rng_algorithm(algorithm: RandomAlgorithm):
  if algorithm == RandomAlgorithm.RNG_THREE_FRY:
    return mhlo.RngAlgorithmAttr.get("THREE_FRY")
  elif algorithm == RandomAlgorithm.RNG_PHILOX:
    return mhlo.RngAlgorithmAttr.get("PHILOX")
  elif algorithm == RandomAlgorithm.RNG_DEFAULT:
    return mhlo.RngAlgorithmAttr.get("DEFAULT")
  else:
    assert False

def _rng_bit_generator_lowering(
    ctx, key, *, shape, dtype, algorithm):
  key_type = ir.RankedTensorType(key.type)
  key_shape, key_etype = key_type.shape, key_type.element_type
  # While the RngBitGenerator HLO accepts a u64[2] key on all backends, we
  # typically represent the key argument to this primitive as a u32[4] so as to
  # sidestep issues with the jax_enable_x64=False configuration. As a result, we
  # need to convert u32[4] -> u64[2] here in the translation rule. However, we
  # also polymorphically allow a u64[2] for backward compatibility.
  #
  # Separately, xops.RngBitGenerator doesn't support generating u8 or
  # u16, so we request u32 and truncate in that case.
  u32_type = ir.IntegerType.get_unsigned(32)
  u64_type = ir.IntegerType.get_unsigned(64)
  assert ((key_shape == [4] and key_etype == u32_type) or
          (key_shape == [2] and key_etype == u64_type)), (key_shape, key_etype)
  dtype = np.dtype(dtype)
  etype = mlir.dtype_to_ir_type(dtype)
  if dtype == np.dtype('uint32') or dtype == np.dtype('uint64'):
    rbg_etype = etype
  else:
    rbg_etype = u32_type
  if key_etype == u32_type:
    key = mhlo.BitcastConvertOp(
        ir.RankedTensorType.get([2], u64_type),
        mhlo.ReshapeOp(ir.RankedTensorType.get([2, 2], u32_type), key)).result
  algorithm_attr = _rng_algorithm(algorithm)
  out_key, out_vals = mhlo.RngBitGeneratorOp(
      key.type,
      ir.RankedTensorType.get(shape, rbg_etype),
      algorithm_attr, key).results
  if key_etype == u32_type:
    out_key = mhlo.ReshapeOp(
        ir.RankedTensorType.get([4], u32_type),
        mhlo.BitcastConvertOp(
            ir.RankedTensorType.get([2, 2], u32_type), out_key)).result
  if rbg_etype != etype:
    out_vals = mhlo.ConvertOp(
      ir.RankedTensorType.get(ir.RankedTensorType(out_vals.type).shape, etype),
      out_vals).result
  return [out_key, out_vals]


def _rng_bit_generator_named_shape_rule(key, *, shape, dtype, algorithm):
  return [key.named_shape, key.named_shape]

rng_bit_generator_p = Primitive("rng_bit_generator")
rng_bit_generator_p.multiple_results = True
rng_bit_generator_p.def_impl(
    partial(xla.apply_primitive, rng_bit_generator_p))
rng_bit_generator_p.def_abstract_eval(
    partial(standard_multi_result_abstract_eval, rng_bit_generator_p,
            _rng_bit_generator_shape_rule, _rng_bit_generator_dtype_rule,
            _rng_bit_generator_weak_type_rule,
            _rng_bit_generator_named_shape_rule))
mlir.register_lowering(rng_bit_generator_p,
                       _rng_bit_generator_lowering)


def _array_copy(arr):
  return copy_p.bind(arr)

# The copy_p primitive exists for expressing making copies of runtime arrays.
# For that reason we don't simplify it out of jaxprs (e.g. for jit invariance).
# It's used in jnp.array(x, copy=True), which is the user-facing API.
copy_p = core.Primitive('copy')
copy_p.def_impl(partial(xla.apply_primitive, copy_p))
copy_p.def_abstract_eval(lambda x: x)
mlir.register_lowering(copy_p, lambda ctx, x: [x])
ad.deflinear(copy_p, lambda t: [copy_p.bind(t)])
pe.def_trivial_padding(copy_p)
batching.defvectorized(copy_p)


def rng_bit_generator(key, shape, dtype=np.uint32,
                      algorithm=RandomAlgorithm.RNG_DEFAULT):
  """Stateless PRNG bit generator. Experimental and its use is discouraged.

  Returns uniformly distributed random bits with the specified shape and dtype
  (what is required to be an integer type) using the platform specific
  default algorithm or the one specified.

  It provides direct access to the RngBitGenerator primitive exposed by XLA
  (https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator) for low
  level API access.

  Most users should use `jax.random` instead for a stable and more user
  friendly API.
  """
  shape = jax.core.canonicalize_shape(shape)
  dtype = dtypes.canonicalize_dtype(dtype)
  if np.dtype(dtype) not in {np.dtype('uint8'), np.dtype('uint16'),
                             np.dtype('uint32'), np.dtype('uint64')}:
    raise TypeError(f'rng_bit_generator: unsupported dtype {dtype}')
  return tuple(
      rng_bit_generator_p.bind(
          key, shape=shape, dtype=dtype, algorithm=algorithm))


def _iota_abstract_eval(*, dtype, shape, dimension):
  _check_shapelike("iota", "shape", shape)
  if not any(dtypes.issubdtype(dtype, t) for t in _num):
    msg = 'iota does not accept dtype {}. Accepted dtypes are subtypes of {}.'
    typename = str(np.dtype(dtype).name)
    accepted_typenames = (t.__name__ for t in _num)
    raise TypeError(msg.format(typename, ', '.join(accepted_typenames)))
  if not 0 <= dimension < len(shape):
    raise ValueError("iota dimension must be between 0 and len(shape), got "
                     f"dimension={dimension} for shape {shape}")
  if not any(isinstance(d, core.BInt) for d in shape):
    return ShapedArray(shape, dtype)
  # TODO(mattjj): unify DShapedArray with ShapedArray, and remove this code
  return core.DShapedArray(shape, dtype, False)

iota_p = Primitive('iota')
iota_p.def_impl(partial(xla.apply_primitive, iota_p))
iota_p.def_abstract_eval(_iota_abstract_eval)

def _iota_staging_rule(trace, *dyn_shape, dtype, shape, dimension):
  params = dict(dtype=dtype, shape=shape, dimension=dimension)
  if not dyn_shape:
    return trace.default_process_primitive(iota_p, (), params)
  aval = core.DShapedArray(_merge_dyn_shape(shape, dyn_shape), dtype, False)
  return _dyn_shape_staging_rule(trace, iota_p, aval, *dyn_shape, **params)
pe.custom_staging_rules[iota_p] = _iota_staging_rule

def _iota_typecheck_rule(*dyn_shape, dtype, shape, dimension):
  if not dyn_shape:
    out_aval, effects = iota_p.abstract_eval(
        dtype=dtype, shape=shape, dimension=dimension)
    return [out_aval], effects
  else:
    out_shape = _merge_dyn_shape(shape, dyn_shape)
    out_shape = [x.val if type(x) is core.Literal else x for x in out_shape]
    out_aval = core.DShapedArray(tuple(out_shape), dtype, False)
    return [out_aval], core.no_effects
core.custom_typechecks[iota_p] = _iota_typecheck_rule

def _iota_lower(ctx, *dyn_shape, dtype, shape, dimension):
  del dtype
  aval_out, = ctx.avals_out
  if dyn_shape:
    shape = _merge_dyn_shape(shape, dyn_shape)
    return mhlo.DynamicIotaOp(
        mlir.aval_to_ir_type(aval_out),
        mlir.shape_tensor(shape),
        mlir.i64_attr(dimension),
    ).results
  else:
    return mhlo.IotaOp(mlir.aval_to_ir_type(aval_out),
                       mlir.i64_attr(dimension)).results
mlir.register_lowering(iota_p, _iota_lower)

def _iota_pp_rule(eqn, context, settings):
  printed_params = {}
  if len(eqn.params['shape']) > 1:
    printed_params['dimension'] = eqn.params['dimension']
  lhs = core.pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name),
         core.pp_kv_pairs(sorted(printed_params.items()), context, settings),
         pp.text(" ") + core.pp_vars(eqn.invars, context)]
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  return [lhs, pp.text(" = ", annotation=annotation), *rhs]
# core.pp_eqn_rules[iota_p] = _iota_pp_rule

def _iota_padding_rule(in_avals, out_avals, *dyn_shape, dtype, shape, dimension):
  out_aval, = out_avals
  new_shape = []
  new_dyn_shape = []
  for d in out_aval.shape:
    if type(d) is pe.BoundedAxisSize:
      new_shape.append(d.bound)
    elif type(d) is int:
      new_shape.append(d)
    else:
      assert isinstance(d, core.Tracer)
      new_shape.append(None)
      new_dyn_shape.append(d)
  return [iota_p.bind(*new_dyn_shape, shape=tuple(new_shape),
                      dtype=dtype, dimension=dimension)]
pe.padding_rules[iota_p] = _iota_padding_rule


def make_bint(i, bd: int):
  return bint_p.bind(i, bd=bd)

bint_p = core.Primitive('bint')

@bint_p.def_impl
def _bint_impl(i, *, bd):
  return core.BInt(i, bd)

@bint_p.def_abstract_eval
def bint_abstract_eval(_, *, bd: int):
  return core.AbstractBInt(bound=bd)

pe.padding_rules[bint_p] = lambda _, __, i, bd: [i]
mlir.register_lowering(bint_p, lambda ctx, x, bd: [x])


### util

_ndim = np.ndim


def _dilate_shape(shape, dilation):
  """Utility function for computing the shape resulting from a dilation."""
  if not np.all(np.greater(dilation, 0)):
    msg = "All dilations must be positive, got {}."
    raise TypeError(msg.format(dilation))
  dilation = (1,) * (len(shape) - len(dilation)) + tuple(dilation)
  return core.dilate_shape(shape, dilation)

def _ceil_divide(x1, x2):
  return -np.floor_divide(np.negative(x1), x2)

def padtype_to_pads(in_shape, window_shape, window_strides, padding):
  """Convert padding string to list of pairs of pad values."""
  PaddingType = xla_client.PaddingType

  if isinstance(padding, str):
    mapping = {'VALID': PaddingType.VALID, 'SAME': PaddingType.SAME}
    try:
      padding = mapping[padding.upper()]
    except KeyError as err:
      msg = "Unrecognized padding type: expected 'VALID' or 'SAME', got {}."
      raise RuntimeError(msg.format(padding)) from err

  if padding == PaddingType.SAME:
    out_shape = _ceil_divide(in_shape, window_strides)
    pad_sizes = np.maximum(0, (out_shape - 1) * window_strides +
                                window_shape - in_shape)
    return [(pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes]
  elif padding == PaddingType.VALID:
    return [(0, 0)] * len(in_shape)
  else:
    msg = "Unknown padding type: {}."
    raise TypeError(msg.format(padding))


# Map of lax function to equivalent jax.numpy function for use in error string below.
_JNP_FUNCTION_EQUIVALENTS = {
  'abs': 'fabs',
  'acos': 'arccos',
  'acosh': 'arccosh',
  'add': 'add',
  'asin': 'arcsin',
  'asinh': 'arcsinh',
  'atan': 'arctan',
  'atan2': 'arctan2',
  'atanh': 'arctanh',
  'bitwise_and': 'bitwise_and',
  'bitwise_not': 'bitwise_not',
  'bitwise_or': 'bitwise_or',
  'bitwise_xor': 'bitwise_xor',
  'cbrt': 'cbrt',
  'ceil': 'ceil',
  'concatenate': 'concatenate',
  'cos': 'cos',
  'cosh': 'cosh',
  'div': 'divide',
  'eq': 'equal',
  'exp': 'exp',
  'expm1': 'expm1',
  'floor': 'floor',
  'greater': 'greater',
  'greater_equal': 'greater_equal',
  'less': 'less',
  'less_equal': 'less_equal',
  'log': 'log',
  'logical_and': 'logical_and',
  'logical_not': 'logical_not',
  'logical_or': 'logical_or',
  'logical_xor': 'logical_xor',
  'log1p': 'log1p',
  'max': 'maximum',
  'min': 'minimum',
  'mul': 'multiply',
  'ne': 'not_equal',
  'neg': 'negative',
  'nextafter': 'nextafter',
  'pow': 'float_power',
  'rount': 'rount',
  'select': 'where',
  'shift_left': 'left_shift',
  'shift_right_logical': 'right_shift',
  'shift_right_arithmetic': 'right_shift',
  'sign': 'sign',
  'sin': 'sin',
  'sinh': 'sinh',
  'sqrt': 'sqrt',
  'sub': 'subtract',
  'tan': 'tan',
  'tanh': 'tanh'
}

def _check_same_dtypes(name, ignore_fp_precision, *ttypes):
  """Check that dtypes agree, possibly ignoring float precision."""
  # the `ignore_fp_precision` flag exists because the XLA shape inference logic
  # allows mixed floating point precision, but the HLO verifier often rejects it
  if any(core.is_opaque_dtype(t) for t in ttypes):
    return  # TODO(mattjj,frostig): do some checking, friend
  types = map(np.dtype, ttypes)  # canonicalize
  if ignore_fp_precision:
    types = [
        np.floating if dtypes.issubdtype(dtype, np.floating)
        else np.complexfloating if dtypes.issubdtype(dtype, np.complexfloating)
        else dtype for dtype in types]
  if len({dtypes.canonicalize_dtype(t) for t in types}) != 1:
    if ignore_fp_precision:
      msg = ("lax.{} requires arguments to have same dtypes up to floating point "
             "precision, got {}.")
    else:
      msg = "lax.{} requires arguments to have the same dtypes, got {}."
    if name in _JNP_FUNCTION_EQUIVALENTS:
      equiv = _JNP_FUNCTION_EQUIVALENTS[name]
      msg += f" (Tip: jnp.{equiv} is a similar function that does automatic type promotion on inputs)."
    raise TypeError(msg.format(name, ", ".join(map(str, types))))


def _check_shapelike(fun_name, arg_name, obj, non_zero_shape=False):
  """Check that `obj` is a shape-like value (e.g. tuple of nonnegative ints)."""
  if not isinstance(obj, (tuple, list, np.ndarray)):
    msg = "{} {} must be of type tuple/list/ndarray, got {}."
    raise TypeError(msg.format(fun_name, arg_name, type(obj)))
  # bool(obj) for an ndarray raises an error, so we check len
  if not len(obj):  # pylint: disable=g-explicit-length-test
    return
  if (config.jax_dynamic_shapes and isinstance(obj, (tuple, list)) and
      any(isinstance(d, (core.Tracer, core.BInt)) for d in obj)):
    return  # TODO(mattjj): handle more checks in the dynamic shape case
  obj_arr = np.array(obj)
  if obj_arr.ndim != 1:
    msg = "{} {} must be 1-dimensional, got {}."
    raise TypeError(msg.format(obj_arr.ndim))
  try:
    canonicalize_shape(obj_arr)
  except TypeError as err:
    msg = "{} {} must have every element be an integer type, got {}."
    raise TypeError(msg.format(fun_name, arg_name, tuple(map(type, obj)))) from err
  lower_bound, bound_error = (
      (1, "strictly positive") if non_zero_shape else (0, "nonnegative"))
  if not all(core.greater_equal_dim(d, lower_bound) for d in obj_arr):
    msg = "{} {} must have every element be {}, got {}."
    raise TypeError(msg.format(fun_name, arg_name, bound_error, obj))


def _const(example, val):
  dtype = _dtype(example)
  if dtypes.is_python_scalar(example):
    val = dtypes.scalar_type_of(example)(val)
    return val if dtype == _dtype(val) else np.array(val, dtype)
  return np.array(val, dtype)

_zeros: Callable = partial(full_like, fill_value=0)
_zero: Callable = partial(full_like, shape=(), fill_value=0)
_ones: Callable = partial(full_like, fill_value=1)
_one: Callable = partial(full_like, shape=(), fill_value=1)
_twos: Callable = partial(full_like, fill_value=2)
_two: Callable = partial(full_like, shape=(), fill_value=2)

dtype: Callable = partial(dtypes.dtype, canonicalize=True)
_dtype: Callable = partial(dtypes.dtype, canonicalize=True)

def _isnan(x) -> bool:
  return ne(x, x)

def _iscomplex(x) -> bool:
  return dtypes.issubdtype(_dtype(x), np.complexfloating)


def ranges_like(*xs):
  start = 0
  for x in xs:
    x_len = len(x)
    yield range(start, start + x_len)
    start += x_len


def remaining(original, *removed_lists):
  removed = set(itertools.chain(*removed_lists))
  return [i for i in original if i not in removed]


def canonicalize_precision(precision: PrecisionLike) -> Optional[Tuple[PrecisionType, PrecisionType]]:
  """Turns an API precision specification, into a pair of enumeration values.

  The API can take the precision as a string, or int, and either as a single
  value to apply to both operands, or as a sequence of two values.
  """
  if precision is None:
    if config.jax_default_matmul_precision is None:
      return None
    try:
      return type_cast(
          Tuple[PrecisionType, PrecisionType],
          (Precision(config.jax_default_matmul_precision),
           Precision(config.jax_default_matmul_precision)))
    except TypeError:
      raise ValueError(
          "jax_default_matmul_precision flag must be set to None or a value in "
          f"{list(Precision._strings)}, but got {config.jax_default_matmul_precision}"
      ) from None
  elif isinstance(precision, str) and precision in Precision._strings:
    return type_cast(Tuple[PrecisionType, PrecisionType],
                     (Precision(precision), Precision(precision)))
  elif isinstance(precision, xla_client.PrecisionConfig.Precision):
    return type_cast(Tuple[PrecisionType, PrecisionType], (precision, precision))
  elif (isinstance(precision, (list, tuple)) and len(precision) == 2 and
        all(isinstance(p, xla_client.PrecisionConfig.Precision) for p in precision)):
    return type_cast(Tuple[PrecisionType, PrecisionType], precision)
  elif (isinstance(precision, (list, tuple)) and len(precision) == 2 and
        all(isinstance(s, str) for s in precision)):
    s1, s2 = precision
    p1 = type_cast(Tuple[PrecisionType, PrecisionType], canonicalize_precision(s1))[0]
    p2 = type_cast(Tuple[PrecisionType, PrecisionType], canonicalize_precision(s2))[0]
    return (p1, p2)
  else:
    raise ValueError(
        f"Precision argument must be None, a string in {list(Precision._strings)}, "
        "a lax.Precision value or a tuple of two lax.Precision values or "
        f"strings; got {precision}.")

def _balanced_eq(x, z, y):
  return div(select(_eq_meet(x, z), _ones(z), _zeros(z)),
             select(_eq_meet(y, z), _twos(z), _ones(z)))


def _eq_meet(a, b):
  a_dtype, b_dtype = _dtype(a), _dtype(b)
  if a_dtype != b_dtype:
    higher_dtype = dtypes.promote_types(a_dtype, b_dtype)
    if higher_dtype == a_dtype:
      a = convert_element_type(a, b_dtype)
    else:
      b = convert_element_type(b, a_dtype)
  return eq(a, b)


def _abstractify(x):
  return raise_to_shaped(core.get_aval(x))


def _check_user_dtype_supported(dtype, fun_name=None):
  # Avoid using `dtype in [...]` because of numpy dtype equality overloading.
  if isinstance(dtype, type) and dtype in {bool, int, float, builtins.complex}:
    return
  np_dtype = np.dtype(dtype)
  if np_dtype.kind not in "biufc" and np_dtype.type != dtypes.bfloat16:
    msg = f"JAX only supports number and bool dtypes, got dtype {dtype}"
    msg += f" in {fun_name}" if fun_name else ""
    raise TypeError(msg)
  if dtype is not None and np_dtype != dtypes.canonicalize_dtype(dtype):
    msg = ("Explicitly requested dtype {} {} is not available, "
           "and will be truncated to dtype {}. To enable more dtypes, set the "
           "jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell "
           "environment variable. "
           "See https://github.com/google/jax#current-gotchas for more.")
    fun_name = f"requested in {fun_name}" if fun_name else ""
    truncated_dtype = dtypes.canonicalize_dtype(dtype).name
    warnings.warn(msg.format(dtype, fun_name , truncated_dtype), stacklevel=3)


def empty(dtype):
  return empty_p.bind(dtype=dtype)
empty_p = core.Primitive('empty')
empty_p.def_abstract_eval(lambda *, dtype: core.ShapedArray((), dtype))
def _empty_lower(ctx, *, dtype):
  if core.is_opaque_dtype(dtype):
    return dtype._rules.empty_mlir(ctx)
  return mlir.ir_constants(np.zeros((), np.dtype(dtype)))
mlir.register_lowering(empty_p, _empty_lower)
