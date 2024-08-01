# Copyright 2019 The JAX Authors.
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

"""Shared neural network activations and other functions."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import operator
import numpy as np
from typing import Any, Literal
import warnings

import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import util
from jax._src.core import AxisName
from jax._src.cudnn.fused_attention_stablehlo import (
    dot_product_attention as cudnn_dot_product_attention, MaskType)
from jax._src.numpy import util as numpy_util
from jax._src.typing import Array, ArrayLike
from jax._src.ops.special import logsumexp as _logsumexp


class Unspecified:
  def __repr__(self):
    return "_UNSPECIFIED"
_UNSPECIFIED = Unspecified()


# activations

@custom_jvp
@jax.jit
def relu(x: ArrayLike) -> Array:
  r"""Rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{relu}(x) = \max(x, 0)

  except under differentiation, we take:

  .. math::
    \nabla \mathrm{relu}(0) = 0

  For more information see
  `Numerical influence of ReLU’(0) on backpropagation
  <https://openreview.net/forum?id=urrcVI-_jRm>`_.

  Args:
    x : input array

  Returns:
    An array.

  Examples:
    >>> jax.nn.relu(jax.numpy.array([-2., -1., -0.5, 0, 0.5, 1., 2.]))
    Array([0. , 0. , 0. , 0. , 0.5, 1. , 2. ], dtype=float32)

  See also:
    :func:`relu6`

  """
  return jnp.maximum(x, 0)
# For behavior at 0, see https://openreview.net/forum?id=urrcVI-_jRm
relu.defjvps(lambda g, ans, x: lax.select(x > 0, g, lax.full_like(g, 0)))

@jax.jit
def squareplus(x: ArrayLike, b: ArrayLike = 4) -> Array:
  r"""Squareplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{squareplus}(x) = \frac{x + \sqrt{x^2 + b}}{2}

  as described in https://arxiv.org/abs/2112.11687.

  Args:
    x : input array
    b : smoothness parameter
  """
  numpy_util.check_arraylike("squareplus", x)
  numpy_util.check_arraylike("squareplus", b)
  x = jnp.asarray(x)
  b = jnp.asarray(b)
  y = x + jnp.sqrt(jnp.square(x) + b)
  return y / 2

@jax.jit
def softplus(x: ArrayLike) -> Array:
  r"""Softplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{softplus}(x) = \log(1 + e^x)

  Args:
    x : input array
  """
  return jnp.logaddexp(x, 0)

@jax.jit
def sparse_plus(x: ArrayLike) -> Array:
  r"""Sparse plus function.

  Computes the function:

  .. math::

    \mathrm{sparse\_plus}(x) = \begin{cases}
      0, & x \leq -1\\
      \frac{1}{4}(x+1)^2, & -1 < x < 1 \\
      x, & 1 \leq x
    \end{cases}

  This is the twin function of the softplus activation ensuring a zero output
  for inputs less than -1 and a linear output for inputs greater than 1,
  while remaining smooth, convex, monotonic by an adequate definition between
  -1 and 1.

  Args:
    x: input (float)
  """
  numpy_util.check_arraylike("sparse_plus", x)
  x = jnp.asarray(x)
  return jnp.where(x <= -1.0, 0.0, jnp.where(x >= 1.0, x, (x + 1.0)**2/4))

@jax.jit
def soft_sign(x: ArrayLike) -> Array:
  r"""Soft-sign activation function.

  Computes the element-wise function

  .. math::
    \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}

  Args:
    x : input array
  """
  numpy_util.check_arraylike("soft_sign", x)
  x_arr = jnp.asarray(x)
  return x_arr / (jnp.abs(x_arr) + 1)

@partial(jax.jit, inline=True)
def sigmoid(x: ArrayLike) -> Array:
  r"""Sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`log_sigmoid`

  """
  return lax.logistic(x)

@jax.jit
def sparse_sigmoid(x: ArrayLike) -> Array:
  r"""Sparse sigmoid activation function.

  Computes the function:

  .. math::

    \mathrm{sparse\_sigmoid}(x) = \begin{cases}
      0, & x \leq -1\\
      \frac{1}{2}(x+1), & -1 < x < 1 \\
      1, & 1 \leq x
    \end{cases}

  This is the twin function of the ``sigmoid`` activation ensuring a zero output
  for inputs less than -1, a 1 output for inputs greater than 1, and a linear
  output for inputs between -1 and 1. It is the derivative of ``sparse_plus``.

  For more information, see `Learning with Fenchel-Young Losses (section 6.2)
  <https://arxiv.org/abs/1901.02324>`_.

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`sigmoid`
  """
  return 0.5 * jnp.clip(x + 1.0, 0.0, 2.0)

@jax.jit
def silu(x: ArrayLike) -> Array:
  r"""SiLU (aka swish) activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

  :func:`swish` and :func:`silu` are both aliases for the same function.

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`sigmoid`
  """
  numpy_util.check_arraylike("silu", x)
  x_arr = jnp.asarray(x)
  return x_arr * sigmoid(x_arr)

swish = silu

@jax.jit
def mish(x: ArrayLike) -> Array:
  r"""Mish activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{mish}(x) = x \cdot \mathrm{tanh}(\mathrm{softplus}(x))

  For more information, see
  `Mish: A Self Regularized Non-Monotonic Activation Function
  <https://arxiv.org/abs/1908.08681>`_.

  Args:
    x : input array

  Returns:
    An array.
  """
  numpy_util.check_arraylike("mish", x)
  x_arr = jnp.asarray(x)
  return x_arr * jnp.tanh(softplus(x_arr))

@jax.jit
def log_sigmoid(x: ArrayLike) -> Array:
  r"""Log-sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`sigmoid`
  """
  numpy_util.check_arraylike("log_sigmoid", x)
  x_arr = jnp.asarray(x)
  return -softplus(-x_arr)

@jax.jit
def elu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Array:
  r"""Exponential linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  Args:
    x : input array
    alpha : scalar or array of alpha values (default: 1.0)

  Returns:
    An array.

  See also:
    :func:`selu`
  """
  numpy_util.check_arraylike("elu", x)
  x_arr = jnp.asarray(x)
  return jnp.where(x_arr > 0,
                   x_arr,
                   alpha * jnp.expm1(jnp.where(x_arr > 0, 0., x_arr)))

@jax.jit
def leaky_relu(x: ArrayLike, negative_slope: ArrayLike = 1e-2) -> Array:
  r"""Leaky rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{leaky\_relu}(x) = \begin{cases}
      x, & x \ge 0\\
      \alpha x, & x < 0
    \end{cases}

  where :math:`\alpha` = :code:`negative_slope`.

  Args:
    x : input array
    negative_slope : array or scalar specifying the negative slope (default: 0.01)

  Returns:
    An array.

  See also:
    :func:`relu`
  """
  numpy_util.check_arraylike("leaky_relu", x)
  x_arr = jnp.asarray(x)
  return jnp.where(x_arr >= 0, x_arr, negative_slope * x_arr)

@jax.jit
def hard_tanh(x: ArrayLike) -> Array:
  r"""Hard :math:`\mathrm{tanh}` activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{hard\_tanh}(x) = \begin{cases}
      -1, & x < -1\\
      x, & -1 \le x \le 1\\
      1, & 1 < x
    \end{cases}

  Args:
    x : input array

  Returns:
    An array.
  """
  numpy_util.check_arraylike("hard_tanh", x)
  x_arr = jnp.asarray(x)
  return jnp.where(x_arr > 1, 1, jnp.where(x_arr < -1, -1, x_arr))

@jax.jit
def celu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Array:
  r"""Continuously-differentiable exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{celu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
    \end{cases}

  For more information, see
  `Continuously Differentiable Exponential Linear Units
  <https://arxiv.org/abs/1704.07483>`_.

  Args:
    x : input array
    alpha : array or scalar (default: 1.0)

  Returns:
    An array.
  """
  return jnp.maximum(x, 0.0) + alpha * jnp.expm1(jnp.minimum(x, 0.0) / alpha)

@jax.jit
def selu(x: ArrayLike) -> Array:
  r"""Scaled exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{selu}(x) = \lambda \begin{cases}
      x, & x > 0\\
      \alpha e^x - \alpha, & x \le 0
    \end{cases}

  where :math:`\lambda = 1.0507009873554804934193349852946` and
  :math:`\alpha = 1.6732632423543772848170429916717`.

  For more information, see
  `Self-Normalizing Neural Networks
  <https://arxiv.org/abs/1706.02515>`_.

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`elu`
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * elu(x, alpha)

# TODO(phawkins): this jit was found to change numerics in a test. Debug this.
# @partial(jax.jit, static_argnames=("approximate",))
def gelu(x: ArrayLike, approximate: bool = True) -> Array:
  r"""Gaussian error linear unit activation function.

  If ``approximate=False``, computes the element-wise function:

  .. math::
    \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{erf} \left(
      \frac{x}{\sqrt{2}} \right) \right)

  If ``approximate=True``, uses the approximate formulation of GELU:

  .. math::
    \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
      \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

  For more information, see `Gaussian Error Linear Units (GELUs)
  <https://arxiv.org/abs/1606.08415>`_, section 2.

  Args:
    x : input array
    approximate: whether to use the approximate or exact formulation.
  """
  [x_arr] = numpy_util.promote_args_inexact("gelu", x)

  if approximate:
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x_arr.dtype)
    cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x_arr + 0.044715 * (x_arr ** 3))))
    return x_arr * cdf
  else:
    sqrt_2 = np.sqrt(2).astype(x_arr.dtype)
    return jnp.array(x_arr * (lax.erf(x_arr / sqrt_2) + 1) / 2, dtype=x_arr.dtype)

@partial(jax.jit, static_argnames=("axis",))
def glu(x: ArrayLike, axis: int = -1) -> Array:
  r"""Gated linear unit activation function.

  Computes the function:

  .. math::
    \mathrm{glu}(x) =  x\left[\ldots, 0:\frac{n}{2}, \ldots\right] \cdot
      \mathrm{sigmoid} \left( x\left[\ldots, \frac{n}{2}:n, \ldots\right]
        \right)

  where the array is split into two along ``axis``. The size of the ``axis``
  dimension must be divisible by two.

  Args:
    x : input array
    axis: the axis along which the split should be computed (default: -1)

  Returns:
    An array.

  See also:
    :func:`sigmoid`
  """
  numpy_util.check_arraylike("glu", x)
  x_arr = jnp.asarray(x)
  size = x_arr.shape[axis]
  assert size % 2 == 0, "axis size must be divisible by 2"
  x1, x2 = jnp.split(x_arr, 2, axis)
  return x1 * sigmoid(x2)

# other functions

logsumexp = _logsumexp


@partial(jax.jit, static_argnames=("axis",))
def log_softmax(x: ArrayLike,
                axis: int | tuple[int, ...] | None = -1,
                where: ArrayLike | None = None,
                initial: ArrayLike | None | Unspecified = _UNSPECIFIED) -> Array:
  r"""Log-Softmax function.

  Computes the logarithm of the :code:`softmax` function, which rescales
  elements to the range :math:`[-\infty, 0)`.

  .. math ::
    \mathrm{log\_softmax}(x)_i = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    \right)

  Args:
    x : input array
    axis: the axis or axes along which the :code:`log_softmax` should be
      computed. Either an integer or a tuple of integers.
    where: Elements to include in the :code:`log_softmax`.

  Returns:
    An array.

  Note:
    If any input values are ``+inf``, the result will be all ``NaN``: this reflects the
    fact that ``inf / inf`` is not well-defined in the context of floating-point math.

  See also:
    :func:`softmax`
  """
  if initial is not _UNSPECIFIED:
    # Added 2024-4-10
    warnings.warn("The initial argument to log_softmax is deprecated, and no longer has any effect.",
                  DeprecationWarning, stacklevel=2)
  del initial
  numpy_util.check_arraylike("log_softmax", x)
  x_arr = jnp.asarray(x)
  x_max = jnp.max(x_arr, axis, where=where, initial=-jnp.inf, keepdims=True)
  x_safe = x_arr if where is None else jnp.where(where, x_arr, -jnp.inf)
  shifted = x_safe - lax.stop_gradient(x_max)
  shifted_logsumexp = jnp.log(
      jnp.sum(jnp.exp(shifted), axis, where=where, keepdims=True))
  result = shifted - shifted_logsumexp
  if where is not None:
    return jnp.where(where, result, -jnp.inf)
  return result


# TODO(phawkins): this jit was found to change numerics in a test. Debug this.
#@partial(jax.jit, static_argnames=("axis",))
def softmax(x: ArrayLike,
            axis: int | tuple[int, ...] | None = -1,
            where: ArrayLike | None = None,
            initial: ArrayLike | None | Unspecified = _UNSPECIFIED) -> Array:
  r"""Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x : input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer or a tuple of integers.
    where: Elements to include in the :code:`softmax`.

  Returns:
    An array.

  Note:
    If any input values are ``+inf``, the result will be all ``NaN``: this reflects the
    fact that ``inf / inf`` is not well-defined in the context of floating-point math.

  See also:
    :func:`log_softmax`
  """
  if initial is not _UNSPECIFIED:
    # Added 2024-4-10
    warnings.warn("The initial argument to softmax is deprecated, and no longer has any effect.",
                  DeprecationWarning, stacklevel=2)
  del initial
  if config.softmax_custom_jvp.value:
    # mypy is confused by the `functools.partial` application in the definition
    # of `_softmax` and incorrectly concludes that `_softmax` returns
    # `ReturnValue` -- the unsubstituted type parameter of `custom_jvp`.
    return _softmax(x, axis, where)
  else:
    return _softmax_deprecated(x, axis, where)

# TODO(mattjj): replace softmax with _softmax when deprecation flag is removed
@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _softmax(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
    initial: ArrayLike | None = -jnp.inf) -> Array:
  x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
  x_safe = x if where is None else jnp.where(where, x, initial)
  unnormalized = jnp.exp(x_safe - x_max)
  result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)
  if where is not None:
    result = jnp.where(where, result, 0)
  return result

@_softmax.defjvp
def _softmax_jvp(axis, primals, tangents):
  (x, where, initial), (x_dot, _, _) = primals, tangents
  y = _softmax(x, axis, where, initial)
  return y, y * (x_dot - (y * x_dot).sum(axis, where=where, keepdims=True))

def _softmax_deprecated(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
    initial: ArrayLike | None = -jnp.inf) -> Array:
  x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
  x_safe = x if where is None else jnp.where(where, x, initial)
  unnormalized = jnp.exp(x_safe - lax.stop_gradient(x_max))
  result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)
  if where is not None:
    result = jnp.where(where, result, 0)
  return result


@partial(jax.jit, static_argnames=("axis",))
def standardize(x: ArrayLike,
              axis: int | tuple[int, ...] | None = -1,
              mean: ArrayLike | None = None,
              variance: ArrayLike | None = None,
              epsilon: ArrayLike = 1e-5,
              where: ArrayLike | None = None) -> Array:
  r"""Normalizes an array by subtracting ``mean`` and dividing by :math:`\sqrt{\mathrm{variance}}`."""
  numpy_util.check_arraylike("standardize", x)
  numpy_util.check_arraylike_or_none("standardize", mean, variance, where)
  if mean is None:
    mean = jnp.mean(x, axis, keepdims=True, where=where)
  if variance is None:
    # this definition is traditionally seen as less accurate than jnp.var's
    # mean((x - mean(x))**2) but may be faster and even, given typical
    # activation distributions and low-precision arithmetic, more accurate
    # when used in neural network normalization layers
    variance = jnp.mean(
        jnp.square(x), axis, keepdims=True, where=where) - jnp.square(mean)
  return jnp.subtract(x, jnp.asarray(mean)) * lax.rsqrt(jnp.asarray(variance) + epsilon)

# TODO(slebedev): Change the type of `x` to `ArrayLike`.
@partial(jax.jit, static_argnames=("num_classes", "dtype", "axis"))
def _one_hot(x: Any, num_classes: int, *,
             dtype: Any, axis: int | AxisName) -> Array:
  num_classes = core.concrete_dim_or_error(
      num_classes,
      "The error arose in jax.nn.one_hot argument `num_classes`.")
  dtype = dtypes.canonicalize_dtype(dtype)
  x_arr = jnp.asarray(x)
  try:
    output_pos_axis = util.canonicalize_axis(axis, x_arr.ndim + 1)
  except TypeError:
    axis_size = lax.psum(1, axis)
    if num_classes != axis_size:
      raise ValueError(f"Expected num_classes to match the size of axis {axis}, "
                       f"but {num_classes} != {axis_size}") from None
    axis_idx = lax.axis_index(axis)
    return jnp.asarray(x_arr == axis_idx, dtype=dtype)
  axis = operator.index(axis)  # type: ignore[arg-type]
  lhs = lax.expand_dims(x_arr, (axis,))
  rhs_shape = [1] * x_arr.ndim
  rhs_shape.insert(output_pos_axis, num_classes)
  rhs = lax.broadcasted_iota(x_arr.dtype, rhs_shape, output_pos_axis)
  return jnp.asarray(lhs == rhs, dtype=dtype)

# TODO(slebedev): Change the type of `x` to `ArrayLike`.
def one_hot(x: Any, num_classes: int, *,
            dtype: Any = jnp.float_, axis: int | AxisName = -1) -> Array:
  """One-hot encodes the given indices.

  Each index in the input ``x`` is encoded as a vector of zeros of length
  ``num_classes`` with the element at ``index`` set to one::

    >>> jax.nn.one_hot(jnp.array([0, 1, 2]), 3)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

  Indices outside the range [0, num_classes) will be encoded as zeros::

    >>> jax.nn.one_hot(jnp.array([-1, 3]), 3)
    Array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

  Args:
    x: A tensor of indices.
    num_classes: Number of classes in the one-hot dimension.
    dtype: optional, a float dtype for the returned values (default :obj:`jnp.float_`).
    axis: the axis or axes along which the function should be
      computed.
  """
  num_classes = core.concrete_dim_or_error(
      num_classes,
      "The error arose in jax.nn.one_hot argument `num_classes`.")
  return _one_hot(x, num_classes, dtype=dtype, axis=axis)


@jax.custom_jvp
@jax.jit
def relu6(x: ArrayLike) -> Array:
  r"""Rectified Linear Unit 6 activation function.

  Computes the element-wise function

  .. math::
    \mathrm{relu6}(x) = \min(\max(x, 0), 6)

  except under differentiation, we take:

  .. math::
    \nabla \mathrm{relu}(0) = 0

  and

  .. math::
    \nabla \mathrm{relu}(6) = 0

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`relu`
  """
  return jnp.minimum(jnp.maximum(x, 0), 6.)
relu6.defjvps(lambda g, ans, x:
              lax.select((x > 0) & (x < 6), g, lax.full_like(g, 0)))

@jax.jit
def hard_sigmoid(x: ArrayLike) -> Array:
  r"""Hard Sigmoid activation function.

  Computes the element-wise function

  .. math::
    \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`relu6`
  """
  return relu6(x + 3.) / 6.

@jax.jit
def hard_silu(x: ArrayLike) -> Array:
  r"""Hard SiLU (swish) activation function

  Computes the element-wise function

  .. math::
    \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)

  Both :func:`hard_silu` and :func:`hard_swish` are aliases for the same
  function.

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`hard_sigmoid`
  """
  numpy_util.check_arraylike("hard_silu", x)
  x_arr = jnp.asarray(x)
  return x_arr * hard_sigmoid(x_arr)

hard_swish = hard_silu

def _get_large_negative(dtype):
  dtype_max = jnp.finfo(dtype).max
  return jnp.asarray(-0.7 * dtype_max, dtype=dtype)

def _get_causal_mask(T, S, dtype):
  pred = jnp.tril(jnp.ones((T, S), dtype=jnp.bool_))
  mask = jnp.where(pred, jnp.asarray(0.0, dtype), _get_large_negative(dtype))
  return mask

def _dot_product_attention_core(query, key, value, bias, mask, is_causal,
                                scale):
  logits_dtype = jnp.promote_types(query.dtype, jnp.float32)
  logits = jnp.einsum('BTNH,BSNH->BNTS', query, key,
                      preferred_element_type=logits_dtype)

  logits *= jnp.array(scale, dtype=logits.dtype)

  if bias is not None:
    logits = (logits + bias).astype(logits.dtype)

  if mask is not None:
    assert mask.dtype == jnp.bool_
    large_negative_number = _get_large_negative(logits.dtype)
    padded_logits = jnp.where(mask, logits, large_negative_number)
  else:
    padded_logits = logits

  if is_causal:
    T, S = query.shape[1], key.shape[1]
    mask = jnp.broadcast_to(_get_causal_mask(T, S, logits.dtype),
                            padded_logits.shape)
    padded_logits = padded_logits + mask

  # Softmax and it is always carried out in fp32.
  padded_logits = padded_logits.astype(jnp.float32)
  probs = jax.nn.softmax(padded_logits, axis=-1).astype(key.dtype)

  encoded = jnp.einsum('BNTS,BSNH->BTNH', probs, value)
  return encoded

def _dot_product_attention_xla(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None,
    mask: Array | None,
    is_causal: bool,
    scale: float):

  B, T, N, H = query.shape
  _, S, K, _ = key.shape
  G = N // K

  query = jnp.reshape(query, (B, T, K, G, H))
  def _reshape_to_grouped(t):
    if t is not None:
      tB, tN, tT, tS = t.shape
      if tN == 1:
        t = jnp.broadcast_to(t[:, :, None, :, :], (tB, tN, G, tT, tS))
      else:
        assert tN == N
        t = jnp.reshape(t, (tB, K, G, tT, tS))
    return t
  bias = _reshape_to_grouped(bias)
  mask = _reshape_to_grouped(mask)
  vmapped_fn = jax.vmap(_dot_product_attention_core,
                        in_axes=(3, None, None, 2, 2, None, None),
                        out_axes=3)
  encoded = vmapped_fn(query, key, value, bias, mask, is_causal, scale)
  encoded = jnp.reshape(encoded, (B, T, N, H))
  return encoded

def dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    *,
    bias: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    implementation: Literal['xla', 'cudnn'] | None = None) -> Array:
  r"""Scaled dot product attention function.

  Computes the attention function on Query, Key, and Value tensors:

  .. math::

    \mathrm{Attention}(Q, K, V)=\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V

  If we define :code:`logits` as the output of :math:`QK^T` and the
  :code:`probs` as the output of :math:`softmax`.

  Throughout this function, we utilize the following uppercase letters to
  represent the shape of array::

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of key/value heads
    G = number of groups, which equals to N // K

  Args:
    query: query array; shape :code:`(BTNH)`
    key: key array: shape :code:`(BSKH)`. When `K` equals `N`, multi-headed
    attention (MHA: https://arxiv.org/abs/1706.03762) is performed. Otherwise,
    grouped query attention (GQA: https://arxiv.org/abs/2305.13245) is performed
    if `N` is a multiple of `K`, and multi-query attention (MQA:
    https://arxiv.org/abs/1911.02150) is performed if `K == 1` (a special case
    of GQA).
    value: value array, should have the same shape as the `key` array.
    bias: optional, bias array to be added to logits; The shape must be 4D and
      be broadcastable to :code:`(BNTS)`.
    mask: optional, mask array used to filter out logits. It is a boolean mask
      where `True` indicates the element should take part in attention. For an
      additive mask, users should pass it to `bias`. The shape must be 4D and be
      broadcastable to :code:`(BNTS)`.
    scale: scale for the logits. If None, the scale will be set to 1 divided by
      the square root of query's head dimension (i.e. H).
    is_causal: If true, causal attention will be applied. Note, some
      implementations like `xla` will generate a mask tensor and apply it to the
      logits to mask out the non-causal parts of the attention matrix, but other
      implementations like `cudnn` will avoid computing the non-causal regions,
      providing speedups.
    implementation: A string to control which implementation backend to use.
      Supported strings are `xla`, `cudnn` (cuDNN flash attention). It defaults
      to `None`, which will automatically select the best available backend.
      Note, `cudnn` supports only a subset of shapes/dtypes, and an exception
      will be thrown if its not supported.

  Returns:
    An array of the attention output with the same shape as :code:`query`.
  """
  def _check_has_shape(t: Array, shape: Sequence[int], name: str) -> None:
    if t.ndim != len(shape):
      raise ValueError(f"{name} ndim should be {len(shape)}, but got {t.ndim}")
    for i in range(t.ndim):
      if shape[i] != -1 and t.shape[i] != shape[i]:
        raise ValueError(f"{name} shape should be {shape}: but got {t.shape}")

  query = jnp.asarray(query)
  key = jnp.asarray(key)
  value = jnp.asarray(value)
  bias = bias if bias is None else jnp.asarray(bias)
  mask = mask if mask is None else jnp.asarray(mask)

  B, S, K, H = key.shape
  _check_has_shape(value, [B, S, K, H], 'value')
  _check_has_shape(query, [B, -1, -1, H], 'query')
  if query.shape[-2] % K != 0:
    raise ValueError(f"The number of query heads must to a multiple of "
                     f"key/value heads, but got {query.shape[-2]} vs {K}")
  if not (query.dtype == key.dtype == value.dtype):
    raise ValueError(f"query/key/value should have the same shape, but got "
                     f"{query.shape} vs {key.shape} vs {value.shape}.")
  if mask is not None and mask.dtype != jnp.bool_ and mask.ndim != 4:
    raise ValueError(f"Mask must be a 4D boolean tensor, but got "
                     f"rank={mask.ndim}, dtype={mask.dtype}.")
  if bias is not None and bias.ndim != 4:
    raise ValueError(f"Bias must be a 4D tensor, but got rank={bias.ndim}.")

  scale_val = (1.0 / np.sqrt(H)) if scale is None else scale

  match implementation:
    case 'xla':
      return _dot_product_attention_xla(
          query, key, value, bias, mask, is_causal=is_causal, scale=scale_val,
      )
    case 'cudnn':
      mask_type = MaskType.CAUSAL if is_causal else MaskType.NO_MASK
      return cudnn_dot_product_attention(
          query, key, value, bias, mask, scale=scale_val, mask_type=mask_type
      )
    case None:
      # TODO(kaixih@nvidia) Defaults to XLA for now. Will automatically select
      # best backend.
      return _dot_product_attention_xla(
          query, key, value, bias, mask, is_causal=is_causal, scale=scale_val,
      )
    case _:
      raise ValueError(f"Unsupported implementation option: {implementation}")
