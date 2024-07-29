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

from collections.abc import Sequence
import operator
import numpy as np

from jax import dtypes
from jax import lax
from jax._src.lib import xla_client
from jax._src.util import safe_zip
from jax._src.numpy.util import check_arraylike, implements, promote_dtypes_inexact
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import ufuncs, reductions
from jax._src.sharding import Sharding
from jax._src.typing import Array, ArrayLike, DTypeLike

Shape = Sequence[int]

def _fft_norm(s: Array, func_name: str, norm: str) -> Array:
  if norm == "backward":
    return jnp.array(1)

  # Avoid potential integer overflow
  s, = promote_dtypes_inexact(s)

  if norm == "ortho":
    return ufuncs.sqrt(reductions.prod(s)) if func_name.startswith('i') else 1/ufuncs.sqrt(reductions.prod(s))
  elif norm == "forward":
    return reductions.prod(s) if func_name.startswith('i') else 1/reductions.prod(s)
  raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                    '"ortho" or "forward".')


def _fft_core(func_name: str, fft_type: xla_client.FftType, a: ArrayLike,
              s: Shape | None, axes: Sequence[int] | None,
              norm: str | None) -> Array:
  full_name = f"jax.numpy.fft.{func_name}"
  check_arraylike(full_name, a)
  arr = jnp.asarray(a)

  if s is not None:
    s = tuple(map(operator.index, s))
    if np.any(np.less(s, 0)):
      raise ValueError("Shape should be non-negative.")

  if s is not None and axes is not None and len(s) != len(axes):
    # Same error as numpy.
    raise ValueError("Shape and axes have different lengths.")

  orig_axes = axes
  if axes is None:
    if s is None:
      axes = range(arr.ndim)
    else:
      axes = range(arr.ndim - len(s), arr.ndim)

  if len(axes) != len(set(axes)):
    raise ValueError(
        f"{full_name} does not support repeated axes. Got axes {axes}.")

  if len(axes) > 3:
    # XLA does not support FFTs over more than 3 dimensions
    raise ValueError(
        "%s only supports 1D, 2D, and 3D FFTs. "
        "Got axes %s with input rank %s." % (full_name, orig_axes, arr.ndim))

  # XLA only supports FFTs over the innermost axes, so rearrange if necessary.
  if orig_axes is not None:
    axes = tuple(range(arr.ndim - len(axes), arr.ndim))
    arr = jnp.moveaxis(arr, orig_axes, axes)

  if s is not None:
    in_s = list(arr.shape)
    for axis, x in safe_zip(axes, s):
      in_s[axis] = x
    if fft_type == xla_client.FftType.IRFFT:
      in_s[-1] = (in_s[-1] // 2 + 1)
    # Cropping
    arr = arr[tuple(map(slice, in_s))]
    # Padding
    arr = jnp.pad(arr, [(0, x-y) for x, y in zip(in_s, arr.shape)])
  else:
    if fft_type == xla_client.FftType.IRFFT:
      s = [arr.shape[axis] for axis in axes[:-1]]
      if axes:
        s += [max(0, 2 * (arr.shape[axes[-1]] - 1))]
    else:
      s = [arr.shape[axis] for axis in axes]
  transformed = lax.fft(arr, fft_type, tuple(s))
  if norm is not None:
    transformed *= _fft_norm(
        jnp.array(s, dtype=transformed.dtype), func_name, norm)

  if orig_axes is not None:
    transformed = jnp.moveaxis(transformed, axes, orig_axes)
  return transformed


def fftn(a: ArrayLike, s: Shape | None = None,
         axes: Sequence[int] | None = None,
         norm: str | None = None) -> Array:
  r"""Compute a multidimensional discrete Fourier transform along given axes.

  JAX implementation of :func:`numpy.fft.fftn`.

  Args:
    a: input array
    s: sequence of integers. Specifies the shape of the result. If not specified,
      it will default to the shape of ``a`` along the specified ``axes``.
    axes: sequence of integers, default=None. Specifies the axes along which the
      transform is computed.
    norm: string. The normalization mode. "backward", "ortho" and "forward" are
      supported.

  Returns:
    An array containing the multidimensional discrete Fourier transform of ``a``.

  See also:
    - :func:`jax.numpy.fft.fft`: Computes a one-dimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.ifft`: Computes a one-dimensional inverse discrete
      Fourier transform.
    - :func:`jax.numpy.fft.ifftn`: Computes a multidimensional inverse discrete
      Fourier transform.

  Examples:
    ``jnp.fft.fftn`` computes the transform along all the axes by default when
    ``axes`` argument is ``None``.

    >>> x = jnp.array([[1, 2, 5, 6],
    ...                [4, 1, 3, 7],
    ...                [5, 9, 2, 1]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.fftn(x)
    Array([[ 46.  +0.j  ,   0.  +2.j  ,  -6.  +0.j  ,   0.  -2.j  ],
           [ -2.  +1.73j,   6.12+6.73j,   0.  -1.73j, -18.12-3.27j],
           [ -2.  -1.73j, -18.12+3.27j,   0.  +1.73j,   6.12-6.73j]],      dtype=complex64)

    When ``s=[2]``, dimension of the transform along ``axis -1`` will be ``2``
    and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.numpy.fft.fftn(x, s=[2]))
    [[ 3.+0.j -1.+0.j]
     [ 5.+0.j  3.+0.j]
     [14.+0.j -4.+0.j]]

    When ``s=[2]`` and ``axes=[0]``, dimension of the transform along ``axis 0``
    will be ``2`` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.numpy.fft.fftn(x, s=[2], axes=[0]))
    [[ 5.+0.j  3.+0.j  8.+0.j 13.+0.j]
     [-3.+0.j  1.+0.j  2.+0.j -1.+0.j]]

    When ``s=[2, 3]``, shape of the transform will be ``(2, 3)``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.numpy.fft.fftn(x, s=[2, 3]))
    [[16. +0.j   -0.5+4.33j -0.5-4.33j]
     [ 0. +0.j   -4.5+0.87j -4.5-0.87j]]

    ``jnp.fft.ifftn`` can be used to reconstruct ``x`` from the result of
    ``jnp.fft.fftn``.

    >>> x_fftn = jnp.fft.fftn(x)
    >>> jnp.allclose(x, jnp.fft.ifftn(x_fftn))
    Array(True, dtype=bool)
  """
  return _fft_core('fftn', xla_client.FftType.FFT, a, s, axes, norm)


def ifftn(a: ArrayLike, s: Shape | None = None,
          axes: Sequence[int] | None = None,
          norm: str | None = None) -> Array:
  r"""Compute a multidimensional inverse discrete Fourier transform.

  JAX implementation of :func:`numpy.fft.ifftn`.

  Args:
    a: input array
    s: sequence of integers. Specifies the shape of the result. If not specified,
      it will default to the shape of ``a`` along the specified ``axes``.
    axes: sequence of integers, default=None. Specifies the axes along which the
      transform is computed. If None, computes the transform along all the axes.
    norm: string. The normalization mode. "backward", "ortho" and "forward" are
      supported.

  Returns:
    An array containing the multidimensional inverse discrete Fourier transform
    of ``a``.

  See also:
    - :func:`jax.numpy.fft.fftn`: Computes a multidimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.fft`: Computes a one-dimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.ifft`: Computes a one-dimensional inverse discrete
      Fourier transform.

  Examples:
    ``jnp.fft.ifftn`` computes the transform along all the axes by default when
    ``axes`` argument is ``None``.

    >>> x = jnp.array([[1, 2, 5, 3],
    ...                [4, 1, 2, 6],
    ...                [5, 3, 2, 1]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifftn(x))
    [[ 2.92+0.j    0.08-0.33j  0.25+0.j    0.08+0.33j]
     [-0.08+0.14j -0.04-0.03j  0.  -0.29j -1.05-0.11j]
     [-0.08-0.14j -1.05+0.11j  0.  +0.29j -0.04+0.03j]]

    When ``s=[3]``, dimension of the transform along ``axis -1`` will be ``3``
    and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifftn(x, s=[3]))
    [[ 2.67+0.j   -0.83-0.87j -0.83+0.87j]
     [ 2.33+0.j    0.83-0.29j  0.83+0.29j]
     [ 3.33+0.j    0.83+0.29j  0.83-0.29j]]

    When ``s=[2]`` and ``axes=[0]``, dimension of the transform along ``axis 0``
    will be ``2`` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifftn(x, s=[2], axes=[0]))
    [[ 2.5+0.j  1.5+0.j  3.5+0.j  4.5+0.j]
     [-1.5+0.j  0.5+0.j  1.5+0.j -1.5+0.j]]

    When ``s=[2, 3]``, shape of the transform will be ``(2, 3)``.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifftn(x, s=[2, 3]))
    [[ 2.5 +0.j    0.  -0.58j  0.  +0.58j]
     [ 0.17+0.j   -0.83-0.29j -0.83+0.29j]]
  """
  return _fft_core('ifftn', xla_client.FftType.IFFT, a, s, axes, norm)


@implements(np.fft.rfftn)
def rfftn(a: ArrayLike, s: Shape | None = None,
          axes: Sequence[int] | None = None,
          norm: str | None = None) -> Array:
  return _fft_core('rfftn', xla_client.FftType.RFFT, a, s, axes, norm)


@implements(np.fft.irfftn)
def irfftn(a: ArrayLike, s: Shape | None = None,
           axes: Sequence[int] | None = None,
           norm: str | None = None) -> Array:
  return _fft_core('irfftn', xla_client.FftType.IRFFT, a, s, axes, norm)


def _axis_check_1d(func_name: str, axis: int | None):
  full_name = f"jax.numpy.fft.{func_name}"
  if isinstance(axis, (list, tuple)):
    raise ValueError(
        "%s does not support multiple axes. Please use %sn. "
        "Got axis = %r." % (full_name, full_name, axis)
    )

def _fft_core_1d(func_name: str, fft_type: xla_client.FftType,
                 a: ArrayLike, n: int | None, axis: int | None,
                 norm: str | None) -> Array:
  _axis_check_1d(func_name, axis)
  axes = None if axis is None else [axis]
  s = None if n is None else [n]
  return _fft_core(func_name, fft_type, a, s, axes, norm)


def fft(a: ArrayLike, n: int | None = None,
        axis: int = -1, norm: str | None = None) -> Array:
  r"""Compute a one-dimensional discrete Fourier transform along a given axis.

  JAX implementation of :func:`numpy.fft.fft`.

  Args:
    a: input array
    n: int. Specifies the dimension of the result along ``axis``. If not specified,
      it will default to the dimension of ``a`` along ``axis``.
    axis: int, default=-1. Specifies the axis along which the transform is computed.
      If not specified, the transform is computed along axis -1.
    norm: string. The normalization mode. "backward", "ortho" and "forward" are
      supported.

  Returns:
    An array containing the one-dimensional discrete Fourier transform of ``a``.

  See also:
    - :func:`jax.numpy.fft.ifft`: Computes a one-dimensional inverse discrete
      Fourier transform.
    - :func:`jax.numpy.fft.fftn`: Computes a multidimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.ifftn`: Computes a multidimensional inverse discrete
      Fourier transform.

  Examples:
    ``jnp.fft.fft`` computes the transform along ``axis -1`` by default.

    >>> x = jnp.array([[1, 2, 4, 7],
    ...                [5, 3, 1, 9]])
    >>> jnp.fft.fft(x)
    Array([[14.+0.j, -3.+5.j, -4.+0.j, -3.-5.j],
           [18.+0.j,  4.+6.j, -6.+0.j,  4.-6.j]], dtype=complex64)

    When ``n=3``, dimension of the transform along axis -1 will be ``3`` and
    dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.fft(x, n=3))
    [[ 7.+0.j   -2.+1.73j -2.-1.73j]
     [ 9.+0.j    3.-1.73j  3.+1.73j]]

    When ``n=3`` and ``axis=0``, dimension of the transform along ``axis 0`` will
    be ``3`` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.fft(x, n=3, axis=0))
    [[ 6. +0.j    5. +0.j    5. +0.j   16. +0.j  ]
     [-1.5-4.33j  0.5-2.6j   3.5-0.87j  2.5-7.79j]
     [-1.5+4.33j  0.5+2.6j   3.5+0.87j  2.5+7.79j]]

    ``jnp.fft.ifft`` can be used to reconstruct ``x`` from the result of
    ``jnp.fft.fft``.

    >>> x_fft = jnp.fft.fft(x)
    >>> jnp.allclose(x, jnp.fft.ifft(x_fft))
    Array(True, dtype=bool)
  """
  return _fft_core_1d('fft', xla_client.FftType.FFT, a, n=n, axis=axis,
                      norm=norm)


def ifft(a: ArrayLike, n: int | None = None,
         axis: int = -1, norm: str | None = None) -> Array:
  r"""Compute a one-dimensional inverse discrete Fourier transform.

  JAX implementation of :func:`numpy.fft.ifft`.

  Args:
    a: input array
    n: int. Specifies the dimension of the result along ``axis``. If not specified,
      it will default to the dimension of ``a`` along ``axis``.
    axis: int, default=-1. Specifies the axis along which the transform is computed.
      If not specified, the transform is computed along axis -1.
    norm: string. The normalization mode. "backward", "ortho" and "forward" are
      supported.

  Returns:
    An array containing the one-dimensional discrete Fourier transform of ``a``.

  See also:
    - :func:`jax.numpy.fft.fft`: Computes a one-dimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.fftn`: Computes a multidimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.ifftn`: Computes a multidimensional inverse of discrete
      Fourier transform.

  Examples:
    ``jnp.fft.ifft`` computes the transform along ``axis -1`` by default.

    >>> x = jnp.array([[3, 1, 4, 6],
    ...                [2, 5, 7, 1]])
    >>> jnp.fft.ifft(x)
    Array([[ 3.5 +0.j  , -0.25-1.25j,  0.  +0.j  , -0.25+1.25j],
          [ 3.75+0.j  , -1.25+1.j  ,  0.75+0.j  , -1.25-1.j  ]],      dtype=complex64)

    When ``n=5``, dimension of the transform along axis -1 will be ``5`` and
    dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifft(x, n=5))
    [[ 2.8 +0.j   -0.96-0.04j  1.06+0.5j   1.06-0.5j  -0.96+0.04j]
     [ 3.  +0.j   -0.59+1.66j  0.09-0.55j  0.09+0.55j -0.59-1.66j]]

    When ``n=3`` and ``axis=0``, dimension of the transform along ``axis 0`` will
    be ``3`` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifft(x, n=3, axis=0))
    [[ 1.67+0.j    2.  +0.j    3.67+0.j    2.33+0.j  ]
     [ 0.67+0.58j -0.5 +1.44j  0.17+2.02j  1.83+0.29j]
     [ 0.67-0.58j -0.5 -1.44j  0.17-2.02j  1.83-0.29j]]
  """
  return _fft_core_1d('ifft', xla_client.FftType.IFFT, a, n=n, axis=axis,
                      norm=norm)


def rfft(a: ArrayLike, n: int | None = None,
         axis: int = -1, norm: str | None = None) -> Array:
  r"""Compute a one-dimensional discrete Fourier transform of a real-valued array.

  JAX implementation of :func:`numpy.fft.rfft`.

  Args:
    a: real-valued input array.
    n: int. Specifies the effective dimension of the input along ``axis``. If not
      specified, it will default to the dimension of input along ``axis``.
    axis: int, default=-1. Specifies the axis along which the transform is computed.
      If not specified, the transform is computed along axis -1.
    norm: string. The normalization mode. "backward", "ortho" and "forward" are
      supported.

  Returns:
    An array containing the one-dimensional discrete Fourier transform of ``a``.
    The dimension of the array along ``axis`` is ``(n/2)+1``, if ``n`` is even and
    ``(n+1)/2``, if ``n`` is odd.

  See also:
    - :func:`jax.numpy.fft.fft`: Computes a one-dimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.irfft`: Computes a one-dimensional inverse discrete
      Fourier transform for real input.
    - :func:`jax.numpy.fft.rfftn`: Computes a multidimensional discrete Fourier
      transform for real input.
    - :func:`jax.numpy.fft.irfftn`: Computes a multidimensional inverse discrete
      Fourier transform for real input.

  Examples:
    ``jnp.fft.rfft`` computes the transform along ``axis -1`` by default.

    >>> x = jnp.array([[1, 3, 5],
    ...                [2, 4, 6]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.rfft(x)
    Array([[ 9.+0.j  , -3.+1.73j],
           [12.+0.j  , -3.+1.73j]], dtype=complex64)

    When ``n=5``, dimension of the transform along axis -1 will be ``(5+1)/2 =3``
    and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.rfft(x, n=5)
    Array([[ 9.  +0.j  , -2.12-5.79j,  0.12+2.99j],
           [12.  +0.j  , -1.62-7.33j,  0.62+3.36j]], dtype=complex64)

    When ``n=4`` and ``axis=0``, dimension of the transform along ``axis 0`` will
    be ``(4/2)+1 =3`` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.rfft(x, n=4, axis=0)
    Array([[ 3.+0.j,  7.+0.j, 11.+0.j],
           [ 1.-2.j,  3.-4.j,  5.-6.j],
           [-1.+0.j, -1.+0.j, -1.+0.j]], dtype=complex64)
  """
  return _fft_core_1d('rfft', xla_client.FftType.RFFT, a, n=n, axis=axis,
                      norm=norm)


def irfft(a: ArrayLike, n: int | None = None,
          axis: int = -1, norm: str | None = None) -> Array:
  r"""Compute a one-dimensional inverse discrete Fourier transform for real input.

  JAX implementation of :func:`numpy.fft.irfft`.

  Args:
    a: real-valued input array.
    n: int. Specifies the dimension of the result along ``axis``. If not specified,
      ``n = 2*(m-1)``, where ``m`` is the dimension of ``a`` along ``axis``.
    axis: int, default=-1. Specifies the axis along which the transform is computed.
      If not specified, the transform is computed along axis -1.
    norm: string. The normalization mode. "backward", "ortho" and "forward" are
      supported.

  Returns:
    An array containing the one-dimensional inverse discrete Fourier transform
    of ``a``, with a dimension of ``n`` along ``axis``.

  See also:
    - :func:`jax.numpy.fft.ifft`: Computes a one-dimensional inverse discrete
      Fourier transform.
    - :func:`jax.numpy.fft.irfft`: Computes a one-dimensional inverse discrete
      Fourier transform for real input.
    - :func:`jax.numpy.fft.rfftn`: Computes a multidimensional discrete Fourier
      transform for real input.
    - :func:`jax.numpy.fft.irfftn`: Computes a multidimensional inverse discrete
      Fourier transform for real input.

  Examples:
    ``jnp.fft.rfft`` computes the transform along ``axis -1`` by default.

    >>> x = jnp.array([[1, 3, 5],
    ...                [2, 4, 6]])
    >>> jnp.fft.irfft(x)
    Array([[ 3., -1.,  0., -1.],
           [ 4., -1.,  0., -1.]], dtype=float32)

    When ``n=3``, dimension of the transform along axis -1 will be ``3`` and
    dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.irfft(x, n=3)
    Array([[ 2.33, -0.67, -0.67],
           [ 3.33, -0.67, -0.67]], dtype=float32)

    When ``n=4`` and ``axis=0``, dimension of the transform along ``axis 0`` will
    be ``4`` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.irfft(x, n=4, axis=0)
    Array([[ 1.25,  2.75,  4.25],
           [ 0.25,  0.75,  1.25],
           [-0.75, -1.25, -1.75],
           [ 0.25,  0.75,  1.25]], dtype=float32)
  """
  return _fft_core_1d('irfft', xla_client.FftType.IRFFT, a, n=n, axis=axis,
                      norm=norm)

@implements(np.fft.hfft)
def hfft(a: ArrayLike, n: int | None = None,
         axis: int = -1, norm: str | None = None) -> Array:
  conj_a = ufuncs.conj(a)
  _axis_check_1d('hfft', axis)
  nn = (conj_a.shape[axis] - 1) * 2 if n is None else n
  return _fft_core_1d('hfft', xla_client.FftType.IRFFT, conj_a, n=n, axis=axis,
                      norm=norm) * nn

@implements(np.fft.ihfft)
def ihfft(a: ArrayLike, n: int | None = None,
          axis: int = -1, norm: str | None = None) -> Array:
  _axis_check_1d('ihfft', axis)
  arr = jnp.asarray(a)
  nn = arr.shape[axis] if n is None else n
  output = _fft_core_1d('ihfft', xla_client.FftType.RFFT, arr, n=n, axis=axis,
                        norm=norm)
  return ufuncs.conj(output) * (1 / nn)


def _fft_core_2d(func_name: str, fft_type: xla_client.FftType, a: ArrayLike,
                 s: Shape | None, axes: Sequence[int],
                 norm: str | None) -> Array:
  full_name = f"jax.numpy.fft.{func_name}"
  if len(axes) != 2:
    raise ValueError(
        "%s only supports 2 axes. Got axes = %r."
        % (full_name, axes)
    )
  return _fft_core(func_name, fft_type, a, s, axes, norm)


def fft2(a: ArrayLike, s: Shape | None = None, axes: Sequence[int] = (-2,-1),
         norm: str | None = None) -> Array:
  """Compute a two-dimensional discrete Fourier transform along given axes.

  JAX implementation of :func:`numpy.fft.fft2`.

  Args:
    a: input array. Must have ``a.ndim >= 2``.
    s: optional length-2 sequence of integers. Specifies the size of the output
      along each specified axis. If not specified, it will default to the size
      of ``a`` along the specified ``axes``.
    axes: optional length-2 sequence of integers, default=(-2,-1). Specifies the
      axes along which the transform is computed.
    norm: string, default="backward". The normalization mode. "backward", "ortho"
      and "forward" are supported.

  Returns:
    An array containing the two-dimensional discrete Fourier transform of ``a``
    along given ``axes``.

  See also:
    - :func:`jax.numpy.fft.fft`: Computes a one-dimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.fftn`: Computes a multidimensional discrete Fourier
      transform.
    - :func:`jax.numpy.fft.ifft2`: Computes a two-dimensional inverse discrete
      Fourier transform.

  Examples:
    ``jnp.fft.fft2`` computes the transform along the last two axes by default.

    >>> x = jnp.array([[[1, 3],
    ...                 [2, 4]],
    ...                [[5, 7],
    ...                 [6, 8]]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.fft2(x)
    Array([[[10.+0.j, -4.+0.j],
            [-2.+0.j,  0.+0.j]],
    <BLANKLINE>
           [[26.+0.j, -4.+0.j],
            [-2.+0.j,  0.+0.j]]], dtype=complex64)

    When ``s=[2, 3]``, dimension of the transform along ``axes (-2, -1)`` will be
    ``(2, 3)`` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.fft2(x, s=[2, 3])
    Array([[[10.  +0.j  , -0.5 -6.06j, -0.5 +6.06j],
            [-2.  +0.j  , -0.5 +0.87j, -0.5 -0.87j]],
    <BLANKLINE>
           [[26.  +0.j  ,  3.5-12.99j,  3.5+12.99j],
            [-2.  +0.j  , -0.5 +0.87j, -0.5 -0.87j]]], dtype=complex64)

    When ``s=[2, 3]`` and ``axes=(0, 1)``, shape of the transform along
    ``axes (0, 1)`` will be ``(2, 3)`` and dimension along other axes will be
    same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.fft2(x, s=[2, 3], axes=(0, 1))
    Array([[[14. +0.j  , 22. +0.j  ],
            [ 2. -6.93j,  4.-10.39j],
            [ 2. +6.93j,  4.+10.39j]],
    <BLANKLINE>
           [[-8. +0.j  , -8. +0.j  ],
            [-2. +3.46j, -2. +3.46j],
            [-2. -3.46j, -2. -3.46j]]], dtype=complex64)

    ``jnp.fft.ifft2`` can be used to reconstruct ``x`` from the result of
    ``jnp.fft.fft2``.

    >>> x_fft2 = jnp.fft.fft2(x)
    >>> jnp.allclose(x, jnp.fft.ifft2(x_fft2))
    Array(True, dtype=bool)
  """
  return _fft_core_2d('fft2', xla_client.FftType.FFT, a, s=s, axes=axes,
                      norm=norm)


def ifft2(a: ArrayLike, s: Shape | None = None, axes: Sequence[int] = (-2,-1),
          norm: str | None = None) -> Array:
  """Compute a two-dimensional inverse discrete Fourier transform.

  JAX implementation of :func:`numpy.fft.ifft2`.

  Args:
    a: input array. Must have ``a.ndim >= 2``.
    s: optional length-2 sequence of integers. Specifies the size of the output
      in each specified axis. If not specified, it will default to the size of
      ``a`` along the specified ``axes``.
    axes: optional length-2 sequence of integers, default=(-2,-1). Specifies the
      axes along which the transform is computed.
    norm: string, default="backward". The normalization mode. "backward", "ortho"
      and "forward" are supported.

  Returns:
    An array containing the two-dimensional inverse discrete Fourier transform
    of ``a`` along given ``axes``.

  See also:
    - :func:`jax.numpy.fft.ifft`: Computes a one-dimensional inverse discrete
      Fourier transform.
    - :func:`jax.numpy.fft.ifftn`: Computes a multidimensional inverse discrete
      Fourier transform.
    - :func:`jax.numpy.fft.fft2`: Computes a two-dimensional discrete Fourier
      transform.

  Examples:
    ``jnp.fft.ifft2`` computes the transform along the last two axes by default.

    >>> x = jnp.array([[[1, 3],
    ...                 [2, 4]],
    ...                [[5, 7],
    ...                 [6, 8]]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.ifft2(x)
    Array([[[ 2.5+0.j, -1. +0.j],
            [-0.5+0.j,  0. +0.j]],
    <BLANKLINE>
           [[ 6.5+0.j, -1. +0.j],
            [-0.5+0.j,  0. +0.j]]], dtype=complex64)

    When ``s=[2, 3]``, dimension of the transform along ``axes (-2, -1)`` will be
    ``(2, 3)`` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.ifft2(x, s=[2, 3])
    Array([[[ 1.67+0.j  , -0.08+1.01j, -0.08-1.01j],
            [-0.33+0.j  , -0.08-0.14j, -0.08+0.14j]],
    <BLANKLINE>
           [[ 4.33+0.j  ,  0.58+2.17j,  0.58-2.17j],
            [-0.33+0.j  , -0.08-0.14j, -0.08+0.14j]]], dtype=complex64)

    When ``s=[2, 3]`` and ``axes=(0, 1)``, shape of the transform along
    ``axes (0, 1)`` will be ``(2, 3)`` and dimension along other axes will be
    same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.ifft2(x, s=[2, 3], axes=(0, 1))
    Array([[[ 2.33+0.j  ,  3.67+0.j  ],
            [ 0.33+1.15j,  0.67+1.73j],
            [ 0.33-1.15j,  0.67-1.73j]],
    <BLANKLINE>
           [[-1.33+0.j  , -1.33+0.j  ],
            [-0.33-0.58j, -0.33-0.58j],
            [-0.33+0.58j, -0.33+0.58j]]], dtype=complex64)
  """
  return _fft_core_2d('ifft2', xla_client.FftType.IFFT, a, s=s, axes=axes,
                      norm=norm)

@implements(np.fft.rfft2)
def rfft2(a: ArrayLike, s: Shape | None = None, axes: Sequence[int] = (-2,-1),
          norm: str | None = None) -> Array:
  return _fft_core_2d('rfft2', xla_client.FftType.RFFT, a, s=s, axes=axes,
                      norm=norm)

@implements(np.fft.irfft2)
def irfft2(a: ArrayLike, s: Shape | None = None, axes: Sequence[int] = (-2,-1),
           norm: str | None = None) -> Array:
  return _fft_core_2d('irfft2', xla_client.FftType.IRFFT, a, s=s, axes=axes,
                      norm=norm)


def fftfreq(n: int, d: ArrayLike = 1.0, *, dtype: DTypeLike | None = None,
            device: xla_client.Device | Sharding | None = None) -> Array:
  """Return sample frequencies for the discrete Fourier transform.

  JAX implementation of :func:`numpy.fft.fftfreq`. Returns frequencies appropriate
  for use with the outputs of :func:`~jax.numpy.fft` and :func:`~jax.numpy.ifft`.

  Args:
    n: length of the FFT window
    d: optional scalar sample spacing (default: 1.0)
    dtype: optional dtype of returned frequencies. If not specified, JAX's default
      floating point dtype will be used.
    device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of sample frequencies, length ``n``.

  See also:
    - :func:`jax.numpy.fft.rfftfreq`: frequencies for use with :func:`~jax.numpy.rfft`
      and :func:`~jax.numpy.irfft`.
  """
  dtype = dtype or dtypes.canonicalize_dtype(jnp.float_)
  if isinstance(n, (list, tuple)):
    raise ValueError(
          "The n argument of jax.numpy.fft.fftfreq only takes an int. "
          "Got n = %s." % list(n))

  elif isinstance(d, (list, tuple)):
    raise ValueError(
          "The d argument of jax.numpy.fft.fftfreq only takes a single value. "
          "Got d = %s." % list(d))

  k = jnp.zeros(n, dtype=dtype)
  if n % 2 == 0:
    # k[0: n // 2 - 1] = jnp.arange(0, n // 2 - 1)
    k = k.at[0: n // 2].set( jnp.arange(0, n // 2, dtype=dtype))

    # k[n // 2:] = jnp.arange(-n // 2, -1)
    k = k.at[n // 2:].set( jnp.arange(-n // 2, 0, dtype=dtype))

  else:
    # k[0: (n - 1) // 2] = jnp.arange(0, (n - 1) // 2)
    k = k.at[0: (n - 1) // 2 + 1].set(jnp.arange(0, (n - 1) // 2 + 1, dtype=dtype))

    # k[(n - 1) // 2 + 1:] = jnp.arange(-(n - 1) // 2, -1)
    k = k.at[(n - 1) // 2 + 1:].set(jnp.arange(-(n - 1) // 2, 0, dtype=dtype))

  result = k / jnp.array(d * n, dtype=dtype)

  if device is not None:
    return result.to_device(device)
  return result


def rfftfreq(n: int, d: ArrayLike = 1.0, *, dtype: DTypeLike | None = None,
             device: xla_client.Device | Sharding | None = None) -> Array:
  """Return sample frequencies for the discrete Fourier transform.

  JAX implementation of :func:`numpy.fft.fftfreq`. Returns frequencies appropriate
  for use with the outputs of :func:`~jax.numpy.rfft` and :func:`~jax.numpy.irfft`.

  Args:
    n: length of the FFT window
    d: optional scalar sample spacing (default: 1.0)
    dtype: optional dtype of returned frequencies. If not specified, JAX's default
      floating point dtype will be used.
    device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
      to which the created array will be committed.

  Returns:
    Array of sample frequencies, length ``n // 2 + 1``.

  See also:
    - :func:`jax.numpy.fft.rfftfreq`: frequencies for use with :func:`~jax.numpy.fft`
      and :func:`~jax.numpy.ifft`.
  """
  dtype = dtype or dtypes.canonicalize_dtype(jnp.float_)
  if isinstance(n, (list, tuple)):
    raise ValueError(
          "The n argument of jax.numpy.fft.rfftfreq only takes an int. "
          "Got n = %s." % list(n))

  elif isinstance(d, (list, tuple)):
    raise ValueError(
          "The d argument of jax.numpy.fft.rfftfreq only takes a single value. "
          "Got d = %s." % list(d))

  if n % 2 == 0:
    k = jnp.arange(0, n // 2 + 1, dtype=dtype)

  else:
    k = jnp.arange(0, (n - 1) // 2 + 1, dtype=dtype)

  result = k / jnp.array(d * n, dtype=dtype)

  if device is not None:
    return result.to_device(device)
  return result


@implements(np.fft.fftshift)
def fftshift(x: ArrayLike, axes: None | int | Sequence[int] = None) -> Array:
  check_arraylike("fftshift", x)
  x = jnp.asarray(x)
  shift: int | Sequence[int]
  if axes is None:
    axes = tuple(range(x.ndim))
    shift = [dim // 2 for dim in x.shape]
  elif isinstance(axes, int):
    shift = x.shape[axes] // 2
  else:
    shift = [x.shape[ax] // 2 for ax in axes]

  return jnp.roll(x, shift, axes)


@implements(np.fft.ifftshift)
def ifftshift(x: ArrayLike, axes: None | int | Sequence[int] = None) -> Array:
  check_arraylike("ifftshift", x)
  x = jnp.asarray(x)
  shift: int | Sequence[int]
  if axes is None:
    axes = tuple(range(x.ndim))
    shift = [-(dim // 2) for dim in x.shape]
  elif isinstance(axes, int):
    shift = -(x.shape[axes] // 2)
  else:
    shift = [-(x.shape[ax] // 2) for ax in axes]

  return jnp.roll(x, shift, axes)
