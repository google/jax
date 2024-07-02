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

from functools import partial
import operator

import numpy as np

from jax import jit
from jax import lax
from jax._src import dtypes
from jax._src import core
from jax._src.numpy.lax_numpy import (
    arange, argmin, array, asarray, atleast_1d, concatenate, convolve,
    diag, dot, finfo, full, ones, outer, roll, trim_zeros,
    trim_zeros_tol, vander, zeros)
from jax._src.numpy.ufuncs import maximum, true_divide, sqrt
from jax._src.numpy.reductions import all
from jax._src.numpy import linalg
from jax._src.numpy.util import (
    check_arraylike, promote_dtypes, promote_dtypes_inexact, _where, implements)
from jax._src.typing import Array, ArrayLike


@jit
def _roots_no_zeros(p: Array) -> Array:
  # build companion matrix and find its eigenvalues (the roots)
  if p.size < 2:
    return array([], dtype=dtypes.to_complex_dtype(p.dtype))
  A = diag(ones((p.size - 2,), p.dtype), -1)
  A = A.at[0, :].set(-p[1:] / p[0])
  return linalg.eigvals(A)


@jit
def _roots_with_zeros(p: Array, num_leading_zeros: Array | int) -> Array:
  # Avoid lapack errors when p is all zero
  p = _where(len(p) == num_leading_zeros, 1.0, p)
  # Roll any leading zeros to the end & compute the roots
  roots = _roots_no_zeros(roll(p, -num_leading_zeros))
  # Sort zero roots to the end.
  roots = lax.sort_key_val(roots == 0, roots)[1]
  # Set roots associated with num_leading_zeros to NaN
  return _where(arange(roots.size) < roots.size - num_leading_zeros, roots, complex(np.nan, np.nan))


def roots(p: ArrayLike, *, strip_zeros: bool = True) -> Array:
  r"""Returns the roots of a polynomial given the coefficients ``p``.

  JAX implementations of :func:`numpy.roots`.

  Args:
    p: Array of polynomial coefficients having rank-1.
    strip_zeros : bool, default=True. If True, then leading zeros in the
      coefficients will be stripped, similar to :func:`numpy.roots`. If set to
      False, leading zeros will not be stripped, and undefined roots will be
      represented by NaN values in the function output. ``strip_zeros`` must be
      set to ``False`` for the function to be compatible with :func:`jax.jit` and
      other JAX transformations.

  Returns:
    An array containing the roots of the polynomial.

  Note:
    Unlike ``np.roots`` of this function, the ``jnp.roots`` returns the roots
    in a complex array regardless of the values of the roots.

  See Also:
    - :func:`jax.numpy.poly`: Finds the polynomial coefficients of the given
      sequence of roots.
    - :func:`jax.numpy.polyfit`: Least squares polynomial fit to data.
    - :func:`jax.numpy.polyval`: Evaluate a polynomial at specific values.

  Examples:
    >>> coeffs = jnp.array([0, 1, 2])

    The default behavior matches numpy and strips leading zeros:

    >>> jnp.roots(coeffs)
    Array([-2.+0.j], dtype=complex64)

    With ``strip_zeros=False``, extra roots are set to NaN:

    >>> jnp.roots(coeffs, strip_zeros=False)
    Array([-2. +0.j, nan+nanj], dtype=complex64)
  """
  check_arraylike("roots", p)
  p_arr = atleast_1d(promote_dtypes_inexact(p)[0])
  del p
  if p_arr.ndim != 1:
    raise ValueError("Input must be a rank-1 array.")
  if p_arr.size < 2:
    return array([], dtype=dtypes.to_complex_dtype(p_arr.dtype))
  num_leading_zeros = _where(all(p_arr == 0), len(p_arr), argmin(p_arr == 0))

  if strip_zeros:
    num_leading_zeros = core.concrete_or_error(int, num_leading_zeros,
      "The error occurred in the jnp.roots() function. To use this within a "
      "JIT-compiled context, pass strip_zeros=False, but be aware that leading zeros "
      "will result in some returned roots being set to NaN.")
    return _roots_no_zeros(p_arr[num_leading_zeros:])
  else:
    return _roots_with_zeros(p_arr, num_leading_zeros)


@partial(jit, static_argnames=('deg', 'rcond', 'full', 'cov'))
def polyfit(x: ArrayLike, y: ArrayLike, deg: int, rcond: float | None = None,
            full: bool = False, w: ArrayLike | None = None, cov: bool = False
            ) -> Array | tuple[Array, ...]:
  r"""Least squares polynomial fit to data.

  Jax implementation of :func:`numpy.polyfit`.

  Given a set of data points ``(x, y)`` and degree of polynomial ``deg``, the
  function finds a polynomial equation of the form:

  .. math::

	   y = p(x) = p[0] x^{deg} + p[1] x^{deg - 1} + ... + p[deg]

  Args:
    x: Array of data points of shape ``(M,)``.
    y: Array of data points of shape ``(M,)`` or ``(M, K)``.
    deg: Degree of the polynomials. It must be specified statically.
    rcond: Relative condition number of the fit. Default value is ``len(x) * eps``.
       It must be specified statically.
    full: Switch that controls the return value. Default is ``False`` which
      restricts the return value to the array of polynomail coefficients ``p``.
      If ``True``, the function returns a tuple ``(p, resids, rank, s, rcond)``.
      It must be specified statically.
    w: Array of weights of shape ``(M,)``. If None, all data points are considered
      to have equal weight. If not None, the weight :math:`w_i` is applied to the
      unsquared residual of :math:`y_i - \widehat{y}_i` at :math:`x_i`, where
      :math:`\widehat{y}_i` is the fitted value of :math:`y_i`. Default is None.
    cov: Boolean or string. If ``True``, returns the covariance matrix scaled
      by ``resids/(M-deg-1)`` along with ploynomial coefficients. If
      ``cov='unscaled'``, returns the unscaaled version of covariance matrix.
      Default is ``False``. ``cov`` is ignored if ``full=True``. It must be
      specified statically.

  Returns:
    - An array polynomial coefficients ``p`` if ``full=False`` and ``cov=False``.

    - A tuple of arrays ``(p, resids, rank, s, rcond)`` if ``full=True``. Where

      - ``p`` is an array of shape ``(M,)`` or ``(M, K)`` containing the polynomial
        coefficients.
      - ``resids`` is the sum of squared residual of shape () or (K,).
      - ``rank`` is the rank of the matrix ``x``.
      - ``s`` is the singular values of the matrix ``x``.
      - ``rcond`` as the array.
    - A tuple of arrays ``(p, C)`` if ``full=False`` and ``cov=True``. Where

      - ``p`` is an array of shape ``(M,)`` or ``(M, K)`` containing the polynomial
        coefficients.
      - ``C`` is the covariance matrix of polynomial coefficients of shape
        ``(deg + 1, deg + 1)`` or ``(deg + 1, deg + 1, 1)``.

  Note:
    Unlike :func:`numpy.polyfit` implementation of polyfit, :func:`jax.numpy.polyfit`
    will not warn on rank reduction, which indicates an ill conditioned matrix.

  See Also:
    - :func:`jax.numpy.poly`: Finds the polynomial coefficients of the given
      sequence of roots.
    - :func:`jax.numpy.polyval`: Evaluate a polynomial at specific values.
    - :func:`jax.numpy.roots`: Computes the roots of a polynomial for given
      coefficients.

  Examples:
    >>> x = jnp.array([3., 6., 9., 4.])
    >>> y = jnp.array([[0, 1, 2],
    ...                [2, 5, 7],
    ...                [8, 4, 9],
    ...                [1, 6, 3]])
    >>> p = jnp.polyfit(x, y, 2)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(p)
    [[ 0.2  -0.35 -0.14]
     [-1.17  4.47  2.96]
     [ 1.95 -8.21 -5.93]]

    If ``full=True``, returns a tuple of arrays as follows:

    >>> p, resids, rank, s, rcond = jnp.polyfit(x, y, 2, full=True)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print("Polynomial Coefficients:", "\n", p, "\n",
    ...         "Residuals:", resids, "\n",
    ...         "Rank:", rank, "\n",
    ...         "s:", s, "\n",
    ...         "rcond:", rcond)
    Polynomial Coefficients:
    [[ 0.2  -0.35 -0.14]
    [-1.17  4.47  2.96]
    [ 1.95 -8.21 -5.93]]
    Residuals: [0.37 5.94 0.61]
    Rank: 3
    s: [1.67 0.47 0.04]
    rcond: 4.7683716e-07

    If ``cov=True`` and ``full=False``, returns a tuple of arrays having
    polynomial coefficients and covariance matrix.

    >>> p, C = jnp.polyfit(x, y, 2, cov=True)
    >>> p.shape, C.shape
    ((3, 3), (3, 3, 1))
  """
  if w is None:
    check_arraylike("polyfit", x, y)
  else:
    check_arraylike("polyfit", x, y, w)
  deg = core.concrete_or_error(int, deg, "deg must be int")
  order = deg + 1
  # check arguments
  x_arr, y_arr = asarray(x), asarray(y)
  del x, y
  if deg < 0:
    raise ValueError("expected deg >= 0")
  if x_arr.ndim != 1:
    raise TypeError("expected 1D vector for x")
  if x_arr.size == 0:
    raise TypeError("expected non-empty vector for x")
  if y_arr.ndim < 1 or y_arr.ndim > 2:
    raise TypeError("expected 1D or 2D array for y")
  if x_arr.shape[0] != y_arr.shape[0]:
    raise TypeError("expected x and y to have same length")

  # set rcond
  if rcond is None:
    rcond = len(x_arr) * finfo(x_arr.dtype).eps
  rcond = core.concrete_or_error(float, rcond, "rcond must be float")
  # set up least squares equation for powers of x
  lhs = vander(x_arr, order)
  rhs = y_arr

  # apply weighting
  if w is not None:
    w, = promote_dtypes_inexact(w)
    w_arr = asarray(w)
    if w_arr.ndim != 1:
      raise TypeError("expected a 1-d array for weights")
    if w_arr.shape[0] != y_arr.shape[0]:
      raise TypeError("expected w and y to have the same length")
    lhs *= w_arr[:, np.newaxis]
    if rhs.ndim == 2:
      rhs *= w_arr[:, np.newaxis]
    else:
      rhs *= w_arr

  # scale lhs to improve condition number and solve
  scale = sqrt((lhs*lhs).sum(axis=0))
  lhs /= scale[np.newaxis,:]
  c, resids, rank, s = linalg.lstsq(lhs, rhs, rcond)
  c = (c.T/scale).T  # broadcast scale coefficients

  if full:
    return c, resids, rank, s, asarray(rcond)
  elif cov:
    Vbase = linalg.inv(dot(lhs.T, lhs))
    Vbase /= outer(scale, scale)
    if cov == "unscaled":
      fac = 1
    else:
      if len(x_arr) <= order:
        raise ValueError("the number of data points must exceed order "
                            "to scale the covariance matrix")
      fac = resids / (len(x_arr) - order)
      fac = fac[0] #making np.array() of shape (1,) to int
    if y_arr.ndim == 1:
      return c, Vbase * fac
    else:
      return c, Vbase[:, :, np.newaxis] * fac
  else:
    return c


@jit
def poly(seq_of_zeros: ArrayLike) -> Array:
  r"""Returns the coefficients of a polynomial for the given sequence of roots.

  JAX implementation of :func:`numpy.poly`.

  Args:
    seq_of_zeros: A scalar or an array of roots of the polynomial of shape ``(M,)``
      or ``(M, M)``.

  Returns:
    An array containing the coefficients of the polynomial. The dtype of the
    output is always promoted to inexact.

  Note:

    :func:`jax.numpy.poly` differs from :func:`numpy.poly`:

    - When the input is a scalar, ``np.poly`` raises a ``TypeError``, whereas
      ``jnp.poly`` treats scalars the same as length-1 arrays.
    - For complex-valued or square-shaped inputs, ``jnp.poly`` always returns
      complex coefficients, whereas ``np.poly`` may return real or complex
      depending on their values.

  See also:
    - :func:`jax.numpy.polyfit`: Least squares polynomial fit.
    - :func:`jax.numpy.polyval`: Evaluate a polynomial at specific values.
    - :func:`jax.numpy.roots`: Computes the roots of a polynomial for given
      coefficients.

  Example:

    Scalar inputs:

    >>> jnp.poly(1)
    Array([ 1., -1.], dtype=float32)

    Input array with integer values:

    >>> x = jnp.array([1, 2, 3])
    >>> jnp.poly(x)
    Array([ 1., -6., 11., -6.], dtype=float32)

    Input array with complex conjugates:

    >>> x = jnp.array([2, 1+2j, 1-2j])
    >>> jnp.poly(x)
    Array([  1.+0.j,  -4.+0.j,   9.+0.j, -10.+0.j], dtype=complex64)

    Input array as square matrix with real valued inputs:

    >>> x = jnp.array([[2, 1, 5],
    ...                [3, 4, 7],
    ...                [1, 3, 5]])
    >>> jnp.round(jnp.poly(x))
    Array([  1.+0.j, -11.-0.j,   9.+0.j, -15.+0.j], dtype=complex64)
  """
  check_arraylike('poly', seq_of_zeros)
  seq_of_zeros, = promote_dtypes_inexact(seq_of_zeros)
  seq_of_zeros_arr = atleast_1d(seq_of_zeros)
  del seq_of_zeros

  sh = seq_of_zeros_arr.shape
  if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
    # import at runtime to avoid circular import
    from jax._src.numpy import linalg
    seq_of_zeros_arr = linalg.eigvals(seq_of_zeros_arr)

  if seq_of_zeros_arr.ndim != 1:
    raise ValueError("input must be 1d or non-empty square 2d array.")

  dt = seq_of_zeros_arr.dtype
  if len(seq_of_zeros_arr) == 0:
    return ones((), dtype=dt)

  a = ones((1,), dtype=dt)
  for k in range(len(seq_of_zeros_arr)):
    a = convolve(a, array([1, -seq_of_zeros_arr[k]], dtype=dt), mode='full')

  return a


@partial(jit, static_argnames=['unroll'])
def polyval(p: ArrayLike, x: ArrayLike, *, unroll: int = 16) -> Array:
  r"""Evaluates the polynomial at specific values.

  JAX implementations of :func:`numpy.polyval`.

  For the 1D-polynomial coefficients ``p`` of length ``M``, the function returns
  the value:

  .. math::

    p_0 x^{M - 1} + p_1 x^{M - 2} + ... + p_{M - 1}

  Args:
    p: An array of polynomial coefficients of shape ``(M,)``.
    x: A number or an array of numbers.
    unroll: A number used to control the number of unrolled steps with
      ``lax.scan``. It must be specified statically.

  Returns:
    An array of same shape as ``x``.

  Note:

    The ``unroll`` parameter is JAX specific. It does not affect correctness but
    can have a major impact on performance for evaluating high-order polynomials.
    The parameter controls the number of unrolled steps with ``lax.scan`` inside
    the ``jnp.polyval`` implementation. Consider setting ``unroll=128`` (or even
    higher) to improve runtime performance on accelerators, at the cost of
    increased compilation time.

  See also:
    - :func:`jax.numpy.polyfit`: Least squares polynomial fit.
    - :func:`jax.numpy.poly`: Finds the coefficients of a polynomial with given
      roots.
    - :func:`jax.numpy.roots`: Computes the roots of a polynomial for given
      coefficients.

  Example:
    >>> p = jnp.array([2, 5, 1])
    >>> jnp.polyval(p, 3)
    Array(34., dtype=float32)

    If ``x`` is a 2D array, ``polyval`` returns 2D-array with same shape as
    that of ``x``:

    >>> x = jnp.array([[2, 1, 5],
    ...                [3, 4, 7],
    ...                [1, 3, 5]])
    >>> jnp.polyval(p, x)
    Array([[ 19.,   8.,  76.],
           [ 34.,  53., 134.],
           [  8.,  34.,  76.]], dtype=float32)
  """
  check_arraylike("polyval", p, x)
  p_arr, x_arr = promote_dtypes_inexact(p, x)
  del p, x
  shape = lax.broadcast_shapes(p_arr.shape[1:], x_arr.shape)
  y = lax.full_like(x_arr, 0, shape=shape, dtype=x_arr.dtype)
  y, _ = lax.scan(lambda y, p: (y * x_arr + p, None), y, p_arr, unroll=unroll)
  return y

@implements(np.polyadd)
@jit
def polyadd(a1: ArrayLike, a2: ArrayLike) -> Array:
  check_arraylike("polyadd", a1, a2)
  a1_arr, a2_arr = promote_dtypes(a1, a2)
  del a1, a2
  if a2_arr.shape[0] <= a1_arr.shape[0]:
    return a1_arr.at[-a2_arr.shape[0]:].add(a2_arr)
  else:
    return a2_arr.at[-a1_arr.shape[0]:].add(a1_arr)


@partial(jit, static_argnames=('m',))
def polyint(p: ArrayLike, m: int = 1, k: int | ArrayLike | None = None) -> Array:
  r"""Returns the coefficients of the integration of specified order of a polynomial.

  JAX implementation of :func:`numpy.polyint`.

  Args:
    p: An array of polynomial coefficients.
    m: Order of integration. Default is 1. It must be specified statically.
    k: Scalar or array of ``m`` integration constant (s).

  Returns:
    An array of coefficients of integrated polynomial.

  See also:
    - :func:`jax.numpy.polyder`: Computes the coefficients of the derivative of
      a polynomial.
    - :func:`jax.numpy.polyval`: Evaluates a polynomial at specific values.

  Examples:

    The first order integration of the polynomial :math:`12 x^2 + 12 x + 6` is
    :math:`4 x^3 + 6 x^2 + 6 x`.

    >>> p = jnp.array([12, 12, 6])
    >>> jnp.polyint(p)
    Array([4., 6., 6., 0.], dtype=float32)

    Since the constant ``k`` is not provided, the result included ``0`` at the end.
    If the constant ``k`` is provided:

    >>> jnp.polyint(p, k=4)
    Array([4., 6., 6., 4.], dtype=float32)

    and the second order integration is :math:`x^4 + 2 x^3 + 3 x`:

    >>> jnp.polyint(p, m=2)
    Array([1., 2., 3., 0., 0.], dtype=float32)

    When ``m>=2``, the constants ``k`` should be provided as an array having
    ``m`` elements. The second order integration of the polynomial
    :math:`12 x^2 + 12 x + 6` with the constants ``k=[4, 5]`` is
    :math:`x^4 + 2 x^3 + 3 x^2 + 4 x + 5`:

    >>> jnp.polyint(p, m=2, k=jnp.array([4, 5]))
    Array([1., 2., 3., 4., 5.], dtype=float32)
  """
  m = core.concrete_or_error(operator.index, m, "'m' argument of jnp.polyint")
  k = 0 if k is None else k
  check_arraylike("polyint", p, k)
  p_arr, k_arr = promote_dtypes_inexact(p, k)
  del p, k
  if m < 0:
    raise ValueError("Order of integral must be positive (see polyder)")
  k_arr = atleast_1d(k_arr)
  if len(k_arr) == 1:
    k_arr = full((m,), k_arr[0])
  if k_arr.shape != (m,):
    raise ValueError("k must be a scalar or a rank-1 array of length 1 or m.")
  if m == 0:
    return p_arr
  else:
    grid = (arange(len(p_arr) + m, dtype=p_arr.dtype)[np.newaxis]
            - arange(m, dtype=p_arr.dtype)[:, np.newaxis])
    coeff = maximum(1, grid).prod(0)[::-1]
    return true_divide(concatenate((p_arr, k_arr)), coeff)


@partial(jit, static_argnames=('m',))
def polyder(p: ArrayLike, m: int = 1) -> Array:
  r"""Returns the coefficients of the derivative of specified order of a polynomial.

  JAX implementation of :func:`numpy.polyder`.

  Args:
    p: Array of polynomials coefficients.
    m: Order of differentiation (positive integer). Default is 1. It must be
      specified statically.

  Returns:
    An array of polynomial coefficients representing the derivative.

  Note:
    :func:`jax.numpy.polyder` differs from :func:`numpy.polyder` when an integer
    array is given. NumPy returns the result with dtype ``int`` whereas JAX
    returns the result with dtype ``float``.

  See also:
    - :func:`jax.numpy.polyint`: Computes the integral of polynomial.
    - :func:`jax.numpy.polyval`: Evaluates a polynomial at specific values.

  Examples:

    The first order derivative of the polynomial :math:`2 x^3 - 5 x^2 + 3 x - 1`
    is :math:`6 x^2 - 10 x +3`:

    >>> p = jnp.array([2, -5, 3, -1])
    >>> jnp.polyder(p)
    Array([  6., -10.,   3.], dtype=float32)

    and its second order derivative is :math:`12 x - 10`:

    >>> jnp.polyder(p, m=2)
    Array([ 12., -10.], dtype=float32)
  """
  check_arraylike("polyder", p)
  m = core.concrete_or_error(operator.index, m, "'m' argument of jnp.polyder")
  p_arr, = promote_dtypes_inexact(p)
  del p
  if m < 0:
    raise ValueError("Order of derivative must be positive")
  if m == 0:
    return p_arr
  coeff = (arange(m, len(p_arr), dtype=p_arr.dtype)[np.newaxis]
          - arange(m, dtype=p_arr.dtype)[:, np.newaxis]).prod(0)
  return p_arr[:-m] * coeff[::-1]


_LEADING_ZEROS_DOC = """\
Setting trim_leading_zeros=True makes the output match that of numpy.
But prevents the function from being able to be used in compiled code.
Due to differences in accumulation of floating point arithmetic errors, the cutoff for values to be
considered zero may lead to inconsistent results between NumPy and JAX, and even between different
JAX backends. The result may lead to inconsistent output shapes when trim_leading_zeros=True.
"""

@implements(np.polymul, lax_description=_LEADING_ZEROS_DOC)
def polymul(a1: ArrayLike, a2: ArrayLike, *, trim_leading_zeros: bool = False) -> Array:
  check_arraylike("polymul", a1, a2)
  a1_arr, a2_arr = promote_dtypes_inexact(a1, a2)
  del a1, a2
  if trim_leading_zeros and (len(a1_arr) > 1 or len(a2_arr) > 1):
    a1_arr, a2_arr = trim_zeros(a1_arr, trim='f'), trim_zeros(a2_arr, trim='f')
  if len(a1_arr) == 0:
    a1_arr = asarray([0], dtype=a2_arr.dtype)
  if len(a2_arr) == 0:
    a2_arr = asarray([0], dtype=a1_arr.dtype)
  return convolve(a1_arr, a2_arr, mode='full')

@implements(np.polydiv, lax_description=_LEADING_ZEROS_DOC)
def polydiv(u: ArrayLike, v: ArrayLike, *, trim_leading_zeros: bool = False) -> tuple[Array, Array]:
  check_arraylike("polydiv", u, v)
  u_arr, v_arr = promote_dtypes_inexact(u, v)
  del u, v
  m = len(u_arr) - 1
  n = len(v_arr) - 1
  scale = 1. / v_arr[0]
  q: Array = zeros(max(m - n + 1, 1), dtype = u_arr.dtype) # force same dtype
  for k in range(0, m-n+1):
    d = scale * u_arr[k]
    q = q.at[k].set(d)
    u_arr = u_arr.at[k:k+n+1].add(-d*v_arr)
  if trim_leading_zeros:
    # use the square root of finfo(dtype) to approximate the absolute tolerance used in numpy
    u_arr = trim_zeros_tol(u_arr, tol=sqrt(finfo(u_arr.dtype).eps), trim='f')
  return q, u_arr

@implements(np.polysub)
@jit
def polysub(a1: ArrayLike, a2: ArrayLike) -> Array:
  check_arraylike("polysub", a1, a2)
  a1, a2 = promote_dtypes(a1, a2)
  return polyadd(a1, -a2)
