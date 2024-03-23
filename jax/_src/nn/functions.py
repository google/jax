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

from functools import partial
import operator
from typing import Any, Tuple

import jax
from jax import custom_jvp
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import util
from jax._src.core import AxisName
from jax._src.numpy import util as numpy_util
from jax._src.ops.special import logsumexp as _logsumexp
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp
import numpy as np


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

  Example:
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
  <https://arxiv.org/pdf/1704.07483.pdf>`_.

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
  <https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf>`_.

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


@jax.custom_jvp
def _approximate_normal_cdf_for_gelu(x_arr: Array) -> Array:
  r"""Approximate version of normal CDF for use in GELU.

  Approximates the normal distribution CDF for use in GELU. This function has a
  custom gradient since the approximation's computed gradient is complicated and
  slow to execute, and the real gradient of the function is much simpler.

  Args:
    x_arr: Input tensor.

  Returns: Approximate version of erf(x).
  """
  sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x_arr.dtype)
  cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x_arr + 0.044715 * (x_arr**3))))
  return cdf


@_approximate_normal_cdf_for_gelu.defjvp
def _approximate_normal_cdf_for_gelu_jvp(
    primals: Tuple[Array], tangents: Tuple[Array]
) -> Tuple[Array, Array]:
  (x_arr,) = primals
  (x_arr_dot,) = tangents
  sqrt_one_over_two_pi = np.sqrt(0.5 / np.pi).astype(x_arr.dtype)
  deriv = sqrt_one_over_two_pi * jnp.exp(-(x_arr**2) * 0.5) * x_arr_dot
  return _approximate_normal_cdf_for_gelu(x_arr), deriv


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
    cdf = _approximate_normal_cdf_for_gelu(x_arr)
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
                initial: ArrayLike | None = None) -> Array:
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
    initial: The minimum value used to shift the input array. Must be present
      when :code:`where` is not None.

  Returns:
    An array.

  See also:
    :func:`softmax`
  """
  numpy_util.check_arraylike("log_softmax", x)
  x_arr = jnp.asarray(x)
  x_max = jnp.max(x_arr, axis, where=where, initial=initial, keepdims=True)
  x_safe = x_arr if where is None else jnp.where(where, x_arr, initial)
  shifted = x_safe - lax.stop_gradient(x_max)
  shifted_logsumexp = jnp.log(
      jnp.sum(jnp.exp(shifted), axis, where=where, keepdims=True))
  result = shifted - shifted_logsumexp
  if where is not None:
    return jnp.where(where, result, -jnp.inf)
  return result


# TODO(phawkins): this jit was found to change numerics in a test. Debug this.
# @partial(jax.jit, static_argnames=("axis",))
def softmax(x: ArrayLike,
            axis: int | tuple[int, ...] | None = -1,
            where: ArrayLike | None = None,
            initial: ArrayLike | None = None) -> Array:
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
    initial: The minimum value used to shift the input array. Must be present
      when :code:`where` is not None.

  Returns:
    An array.

  See also:
    :func:`log_softmax`
  """
  if config.softmax_custom_jvp.value:
    # mypy is confused by the `functools.partial` application in the definition
    # of `_softmax` and incorrectly concludes that `_softmax` returns
    # `ReturnValue` -- the unsubstituted type parameter of `custom_jvp`.
    return _softmax(x, axis, where, initial)  # type: ignore[return-value]
  else:
    return _softmax_deprecated(x, axis, where, initial)

# TODO(mattjj): replace softmax with _softmax when deprecation flag is removed
@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _softmax(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
    initial: ArrayLike | None = None) -> Array:
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
    initial: ArrayLike | None = None) -> Array:
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
