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


import functools
import itertools
import operator
import textwrap
from typing import Callable, Dict, List, Sequence, Tuple

import scipy.ndimage

from jax._src import api
from jax._src import util
from jax import lax
from jax._src.lax.lax import PrecisionLike
import jax.numpy as jnp
import jax.scipy as jsp
from jax._src.numpy.util import _wraps
from jax._src.typing import ArrayLike, Array
from jax._src.util import safe_zip as zip


def _nonempty_prod(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.mul, arrs)


def _nonempty_sum(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.add, arrs)


def _mirror_index_fixer(index: Array, size: int) -> Array:
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return jnp.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index: Array, size: int) -> Array:
    return jnp.floor_divide(_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2)


_INDEX_FIXERS: Dict[str, Callable[[Array, int], Array]] = {
    'constant': lambda index, size: index,
    'nearest': lambda index, size: jnp.clip(index, 0, size - 1),
    'wrap': lambda index, size: index % size,
    'mirror': _mirror_index_fixer,
    'reflect': _reflect_index_fixer,
}


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _nearest_indices_and_weights(coordinate: Array) -> List[Tuple[Array, ArrayLike]]:
    index = _round_half_away_from_zero(coordinate).astype(jnp.int32)
    weight = coordinate.dtype.type(1)
    return [(index, weight)]


def _linear_indices_and_weights(coordinate: Array) -> List[Tuple[Array, ArrayLike]]:
    lower = jnp.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(jnp.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


@functools.partial(api.jit, static_argnums=(2, 3, 4))
def _map_coordinates(input: ArrayLike, coordinates: Sequence[ArrayLike],
                     order: int, mode: str, cval: ArrayLike) -> Array:
    input_arr = jnp.asarray(input)
    coordinate_arrs = [jnp.asarray(c) for c in coordinates]
    cval = jnp.asarray(cval, input_arr.dtype)

    if len(coordinates) != input_arr.ndim:
        raise ValueError('coordinates must be a sequence of length input.ndim, but '
                         '{} != {}'.format(len(coordinates), input_arr.ndim))

    index_fixer = _INDEX_FIXERS.get(mode)
    if index_fixer is None:
        raise NotImplementedError(
            'jax.scipy.ndimage.map_coordinates does not yet support mode {}. '
            'Currently supported modes are {}.'.format(mode, set(_INDEX_FIXERS)))

    if mode == 'constant':
        is_valid = lambda index, size: (0 <= index) & (index < size)
    else:
        is_valid = lambda index, size: True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError(
            'jax.scipy.ndimage.map_coordinates currently requires order<=1')

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = util.unzip3(items)
        if all(valid is True for valid in validities):
            # fast path
            contribution = input_arr[indices]
        else:
            all_valid = functools.reduce(operator.and_, validities)
            contribution = jnp.where(all_valid, input_arr[indices], cval)
        outputs.append(_nonempty_prod(weights) * contribution)
    result = _nonempty_sum(outputs)
    if jnp.issubdtype(input_arr.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)


@_wraps(scipy.ndimage.map_coordinates, lax_description=textwrap.dedent("""\
    Only nearest neighbor (``order=0``), linear interpolation (``order=1``) and
    modes ``'constant'``, ``'nearest'``, ``'wrap'`` ``'mirror'`` and ``'reflect'`` are currently supported.
    Note that interpolation near boundaries differs from the scipy function,
    because we fixed an outstanding bug (https://github.com/scipy/scipy/issues/2640);
    this function interprets the ``mode`` argument as documented by SciPy, but
    not as implemented by SciPy.
    """))
def map_coordinates(
        input: ArrayLike, coordinates: Sequence[ArrayLike], order: int, mode: str = 'constant', cval: ArrayLike = 0.0,
):
    return _map_coordinates(input, coordinates, order, mode, cval)


def _gaussian_kernel1d(sigma: float,
                       order: int,
                       radius: int) -> Array:
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = jnp.arange(order + 1)
    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = jnp.zeros(order + 1)
        q = q.at[0].set(1)
        D = jnp.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = jnp.diag(jnp.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


@_wraps(scipy.ndimage.gaussian_filter1d, lax_description=textwrap.dedent("""\
    Only works using standard convolution since JAX does not support FFT convolution.
    """))
def gaussian_filter1d(input: Array, sigma: float, axis=-1, order=0,
                      truncate=4.0, *,
                      radius: int = 0,
                      mode: str = 'constant',
                      cval: float = 0.0,
                      precision: PrecisionLike = None) -> Array:
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)

    if mode != 'constant' or cval != 0.:
        raise NotImplementedError('Other modes than "constant" with 0. fill value are not'
                                  'supported yet.')

    if radius > 0.:
        lw = radius
    if lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')

    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]

    # Be careful that modes in signal.convolve refer to the 'same' 'full' 'valid' modes
    # while in gaussian_filter1d refers to the way the padding is done 'constant' 'reflect' etc.
    # We should change the convolve backend for further features
    return jnp.apply_along_axis(jsp.signal.convolve, axis, input, weights,
                                mode='same',
                                method='auto',
                                precision=precision)


@_wraps(scipy.ndimage.gaussian_filter, lax_description=textwrap.dedent("""\
    Only works using standard convolution since JAX does not support FFT convolution. Does not support varying sigma
    across axes.
    """))
def gaussian_filter(input: Array, sigma: float,
                    order: int = 0,
                    truncate: float = 4.0, *,
                    radius: int = 0,
                    mode: str = 'constant',
                    cval: float = 0.0,
                    precision: PrecisionLike = None) -> Array:

    input = jnp.asarray(input)

    for axis in range(input.ndim):
        input = gaussian_filter1d(input, sigma, axis=axis, order=order,
                                  truncate=truncate, radius=radius, mode=mode, precision=precision, cval=cval)

    return input
