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
from jax.sharding import Sharding
from jax._src.lib import xla_client as xc
from jax._src.util import safe_zip
from jax._src.numpy.util import check_arraylike, implements, promote_dtypes_inexact
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import ufuncs, reductions
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


def _fft_core(func_name: str, fft_type: xc.FftType, a: ArrayLike,
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
    if fft_type == xc.FftType.IRFFT:
      in_s[-1] = (in_s[-1] // 2 + 1)
    # Cropping
    arr = arr[tuple(map(slice, in_s))]
    # Padding
    arr = jnp.pad(arr, [(0, x-y) for x, y in zip(in_s, arr.shape)])
  else:
    if fft_type == xc.FftType.IRFFT:
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


@implements(np.fft.fftn)
def fftn(a: ArrayLike, s: Shape | None = None,
         axes: Sequence[int] | None = None,
         norm: str | None = None) -> Array:
  return _fft_core('fftn', xc.FftType.FFT, a, s, axes, norm)


@implements(np.fft.ifftn)
def ifftn(a: ArrayLike, s: Shape | None = None,
          axes: Sequence[int] | None = None,
          norm: str | None = None) -> Array:
  return _fft_core('ifftn', xc.FftType.IFFT, a, s, axes, norm)


@implements(np.fft.rfftn)
def rfftn(a: ArrayLike, s: Shape | None = None,
          axes: Sequence[int] | None = None,
          norm: str | None = None) -> Array:
  return _fft_core('rfftn', xc.FftType.RFFT, a, s, axes, norm)


@implements(np.fft.irfftn)
def irfftn(a: ArrayLike, s: Shape | None = None,
           axes: Sequence[int] | None = None,
           norm: str | None = None) -> Array:
  return _fft_core('irfftn', xc.FftType.IRFFT, a, s, axes, norm)


def _axis_check_1d(func_name: str, axis: int | None):
  full_name = f"jax.numpy.fft.{func_name}"
  if isinstance(axis, (list, tuple)):
    raise ValueError(
        "%s does not support multiple axes. Please use %sn. "
        "Got axis = %r." % (full_name, full_name, axis)
    )

def _fft_core_1d(func_name: str, fft_type: xc.FftType,
                 a: ArrayLike, n: int | None, axis: int | None,
                 norm: str | None) -> Array:
  _axis_check_1d(func_name, axis)
  axes = None if axis is None else [axis]
  s = None if n is None else [n]
  return _fft_core(func_name, fft_type, a, s, axes, norm)


@implements(np.fft.fft)
def fft(a: ArrayLike, n: int | None = None,
        axis: int = -1, norm: str | None = None) -> Array:
  return _fft_core_1d('fft', xc.FftType.FFT, a, n=n, axis=axis,
                      norm=norm)

@implements(np.fft.ifft)
def ifft(a: ArrayLike, n: int | None = None,
         axis: int = -1, norm: str | None = None) -> Array:
  return _fft_core_1d('ifft', xc.FftType.IFFT, a, n=n, axis=axis,
                      norm=norm)

@implements(np.fft.rfft)
def rfft(a: ArrayLike, n: int | None = None,
         axis: int = -1, norm: str | None = None) -> Array:
  return _fft_core_1d('rfft', xc.FftType.RFFT, a, n=n, axis=axis,
                      norm=norm)

@implements(np.fft.irfft)
def irfft(a: ArrayLike, n: int | None = None,
          axis: int = -1, norm: str | None = None) -> Array:
  return _fft_core_1d('irfft', xc.FftType.IRFFT, a, n=n, axis=axis,
                      norm=norm)

@implements(np.fft.hfft)
def hfft(a: ArrayLike, n: int | None = None,
         axis: int = -1, norm: str | None = None) -> Array:
  conj_a = ufuncs.conj(a)
  _axis_check_1d('hfft', axis)
  nn = (conj_a.shape[axis] - 1) * 2 if n is None else n
  return _fft_core_1d('hfft', xc.FftType.IRFFT, conj_a, n=n, axis=axis,
                      norm=norm) * nn

@implements(np.fft.ihfft)
def ihfft(a: ArrayLike, n: int | None = None,
          axis: int = -1, norm: str | None = None) -> Array:
  _axis_check_1d('ihfft', axis)
  arr = jnp.asarray(a)
  nn = arr.shape[axis] if n is None else n
  output = _fft_core_1d('ihfft', xc.FftType.RFFT, arr, n=n, axis=axis,
                        norm=norm)
  return ufuncs.conj(output) * (1 / nn)


def _fft_core_2d(func_name: str, fft_type: xc.FftType, a: ArrayLike,
                 s: Shape | None, axes: Sequence[int],
                 norm: str | None) -> Array:
  full_name = f"jax.numpy.fft.{func_name}"
  if len(axes) != 2:
    raise ValueError(
        "%s only supports 2 axes. Got axes = %r."
        % (full_name, axes)
    )
  return _fft_core(func_name, fft_type, a, s, axes, norm)


@implements(np.fft.fft2)
def fft2(a: ArrayLike, s: Shape | None = None, axes: Sequence[int] = (-2,-1),
         norm: str | None = None) -> Array:
  return _fft_core_2d('fft2', xc.FftType.FFT, a, s=s, axes=axes,
                      norm=norm)

@implements(np.fft.ifft2)
def ifft2(a: ArrayLike, s: Shape | None = None, axes: Sequence[int] = (-2,-1),
          norm: str | None = None) -> Array:
  return _fft_core_2d('ifft2', xc.FftType.IFFT, a, s=s, axes=axes,
                      norm=norm)

@implements(np.fft.rfft2)
def rfft2(a: ArrayLike, s: Shape | None = None, axes: Sequence[int] = (-2,-1),
          norm: str | None = None) -> Array:
  return _fft_core_2d('rfft2', xc.FftType.RFFT, a, s=s, axes=axes,
                      norm=norm)

@implements(np.fft.irfft2)
def irfft2(a: ArrayLike, s: Shape | None = None, axes: Sequence[int] = (-2,-1),
           norm: str | None = None) -> Array:
  return _fft_core_2d('irfft2', xc.FftType.IRFFT, a, s=s, axes=axes,
                      norm=norm)


@implements(np.fft.fftfreq, extra_params="""
dtype : dtype, optional
    The dtype of the returned frequencies. If not specified, JAX's default
    floating point dtype will be used.
device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
  to which the created array will be committed.
""")
def fftfreq(n: int, d: ArrayLike = 1.0, *, dtype: DTypeLike | None = None,
            device: xc.Device | Sharding | None = None) -> Array:
  dtype = dtype or dtypes.canonicalize_dtype(jnp.float_)
  if isinstance(n, (list, tuple)):
    raise ValueError(
          "The n argument of jax.numpy.fft.fftfreq only takes an int. "
          "Got n = %s." % list(n))

  elif isinstance(d, (list, tuple)):
    raise ValueError(
          "The d argument of jax.numpy.fft.fftfreq only takes a single value. "
          "Got d = %s." % list(d))

  k = jnp.zeros(n, dtype=dtype, device=device)
  if n % 2 == 0:
    # k[0: n // 2 - 1] = jnp.arange(0, n // 2 - 1)
    k = k.at[0: n // 2].set(jnp.arange(0, n // 2, dtype=dtype))

    # k[n // 2:] = jnp.arange(-n // 2, -1)
    k = k.at[n // 2:].set(jnp.arange(-n // 2, 0, dtype=dtype))

  else:
    # k[0: (n - 1) // 2] = jnp.arange(0, (n - 1) // 2)
    k = k.at[0: (n - 1) // 2 + 1].set(jnp.arange(0, (n - 1) // 2 + 1, dtype=dtype))

    # k[(n - 1) // 2 + 1:] = jnp.arange(-(n - 1) // 2, -1)
    k = k.at[(n - 1) // 2 + 1:].set(jnp.arange(-(n - 1) // 2, 0, dtype=dtype))

  return k / jnp.array(d * n, dtype=dtype, device=device)


@implements(np.fft.rfftfreq, extra_params="""
dtype : dtype, optional
    The dtype of the returned frequencies. If not specified, JAX's default
    floating point dtype will be used.
device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
  to which the created array will be committed.
""")
def rfftfreq(n: int, d: ArrayLike = 1.0, *, dtype: DTypeLike | None = None,
             device: xc.Device | Sharding | None = None) -> Array:
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
    k = jnp.arange(0, n // 2 + 1, dtype=dtype, device=device)

  else:
    k = jnp.arange(0, (n - 1) // 2 + 1, dtype=dtype, device=device)

  return k / jnp.array(d * n, dtype=dtype, device=device)


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
