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


from functools import partial
import math
from operator import index
from typing import Optional, Sequence, Union
import warnings

import numpy as np

import jax.numpy as jnp
from jax import lax
from jax.numpy.linalg import cholesky, svd, eigh

from jax._src import config as config_lib
from jax._src import core
from jax._src import dtypes
from jax._src import prng
from jax._src import xla_bridge
from jax._src.api import jit, vmap
from jax._src.config import config
from jax._src.core import NamedShape
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax as lax_internal
from jax._src.numpy.lax_numpy import _convert_and_clip_integer
from jax._src.numpy.util import _arraylike, check_arraylike, promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike, DTypeLike
from jax._src.util import canonicalize_axis


RealArray = ArrayLike
IntegerArray = ArrayLike
# TODO: Import or define these to match
# https://github.com/numpy/numpy/blob/main/numpy/typing/_dtype_like.py.
DTypeLikeInt = DTypeLike
DTypeLikeFloat = DTypeLike
Shape = Sequence[int]

# TODO(frostig): simplify once we always enable_custom_prng
KeyArray = Union[Array, prng.PRNGKeyArray]

UINT_DTYPES = prng.UINT_DTYPES


### utilities

_lax_const = lax_internal._const

def _isnan(x: ArrayLike) -> Array:
  return lax.ne(x, x)


def _check_prng_key(key):
  # TODO(frostig): remove once we always enable_custom_prng
  if isinstance(key, prng.PRNGKeyArray):
    return key, False
  elif _arraylike(key):
    if config.jax_enable_custom_prng:
      warnings.warn(
          'Raw arrays as random keys to jax.random functions are deprecated. '
          'Assuming valid threefry2x32 key for now.',
          FutureWarning)
    return prng.random_wrap(key, impl=default_prng_impl()), True
  else:
    raise TypeError(f'unexpected PRNG key type {type(key)}')


def _return_prng_keys(was_wrapped, key):
  # TODO(frostig): remove once we always enable_custom_prng
  assert isinstance(key, prng.PRNGKeyArray)
  if config.jax_enable_custom_prng:
    return key
  else:
    return prng.random_unwrap(key) if was_wrapped else key


def _random_bits(key: prng.PRNGKeyArray, bit_width, shape) -> Array:
  assert isinstance(key, prng.PRNGKeyArray)
  return prng.random_bits(key, bit_width=bit_width, shape=shape)


PRNG_IMPLS = {
    'threefry2x32': prng.threefry_prng_impl,
    'rbg': prng.rbg_prng_impl,
    'unsafe_rbg': prng.unsafe_rbg_prng_impl,
}

def default_prng_impl():
  """Get the default PRNG implementation.

  The default implementation is determined by ``config.jax_default_prng_impl``,
  which specifies it by name. This function returns the corresponding
  ``jax.prng.PRNGImpl`` instance.
  """
  impl_name = config.jax_default_prng_impl
  assert impl_name in PRNG_IMPLS, impl_name
  return PRNG_IMPLS[impl_name]


### key operations


def PRNGKey(seed: Union[int, Array]) -> KeyArray:
  """Create a pseudo-random number generator (PRNG) key given an integer seed.

  The resulting key carries the default PRNG implementation, as
  determined by the ``jax_default_prng_impl`` config flag.

  Args:
    seed: a 64- or 32-bit integer used as the value of the key.

  Returns:
    A PRNG key, consumable by random functions as well as ``split``
    and ``fold_in``.

  """
  impl = default_prng_impl()
  if np.ndim(seed):
    raise TypeError("PRNGKey accepts a scalar seed, but was given an array of"
                    f"shape {np.shape(seed)} != (). Use jax.vmap for batching")
  key = prng.seed_with_impl(impl, seed)
  return _return_prng_keys(True, key)

# TODO(frostig): remove once we always enable_custom_prng
def _check_default_impl_with_no_custom_prng(impl, name):
  default_impl = default_prng_impl()
  default_name = config.jax_default_prng_impl
  if not config.jax_enable_custom_prng and default_impl is not impl:
    raise RuntimeError('jax_enable_custom_prng must be enabled in order '
                       f'to seed an RNG with an implementation "f{name}" '
                       f'differing from the default "f{default_name}".')

def threefry2x32_key(seed: int) -> KeyArray:
  """Creates a threefry2x32 PRNG key from an integer seed."""
  impl = prng.threefry_prng_impl
  _check_default_impl_with_no_custom_prng(impl, 'threefry2x32')
  key = prng.seed_with_impl(impl, seed)
  return _return_prng_keys(True, key)

def rbg_key(seed: int) -> KeyArray:
  """Creates an RBG PRNG key from an integer seed."""
  impl = prng.rbg_prng_impl
  _check_default_impl_with_no_custom_prng(impl, 'rbg')
  key = prng.seed_with_impl(impl, seed)
  return _return_prng_keys(True, key)

def unsafe_rbg_key(seed: int) -> KeyArray:
  """Creates an unsafe RBG PRNG key from an integer seed."""
  impl = prng.unsafe_rbg_prng_impl
  _check_default_impl_with_no_custom_prng(impl, 'unsafe_rbg')
  key = prng.seed_with_impl(impl, seed)
  return _return_prng_keys(True, key)

def _fold_in(key: KeyArray, data: int) -> KeyArray:
  # Alternative to fold_in() to use within random samplers.
  # TODO(frostig): remove and use fold_in() once we always enable_custom_prng
  assert isinstance(key, prng.PRNGKeyArray)
  if key.ndim:
    raise TypeError("fold_in accepts a single key, but was given a key array of"
                    f"shape {key.shape} != (). Use jax.vmap for batching.")
  if np.ndim(data):
    raise TypeError("fold_in accepts a scalar, but was given an array of"
                    f"shape {np.shape(data)} != (). Use jax.vmap for batching.")
  return prng.random_fold_in(key, jnp.uint32(data))

def fold_in(key: KeyArray, data: int) -> KeyArray:
  """Folds in data to a PRNG key to form a new PRNG key.

  Args:
    key: a PRNG key (from ``PRNGKey``, ``split``, ``fold_in``).
    data: a 32bit integer representing data to be folded in to the key.

  Returns:
    A new PRNG key that is a deterministic function of the inputs and is
    statistically safe for producing a stream of new pseudo-random values.
  """
  key, wrapped = _check_prng_key(key)
  return _return_prng_keys(wrapped, _fold_in(key, data))

def _split(key: KeyArray, num: int = 2) -> KeyArray:
  # Alternative to split() to use within random samplers.
  # TODO(frostig): remove and use split(); we no longer need to wait
  # to always enable_custom_prng
  assert isinstance(key, prng.PRNGKeyArray)
  if key.ndim:
    raise TypeError("split accepts a single key, but was given a key array of"
                    f"shape {key.shape} != (). Use jax.vmap for batching.")
  return prng.random_split(key, count=num)

def split(key: KeyArray, num: int = 2) -> KeyArray:
  """Splits a PRNG key into `num` new keys by adding a leading axis.

  Args:
    key: a PRNG key (from ``PRNGKey``, ``split``, ``fold_in``).
    num: optional, a positive integer indicating the number of keys to produce
      (default 2).

  Returns:
    An array-like object of `num` new PRNG keys.
  """
  key, wrapped = _check_prng_key(key)
  return _return_prng_keys(wrapped, _split(key, num))

def _key_data(keys: KeyArray) -> Array:
  assert isinstance(keys, prng.PRNGKeyArray)
  return prng.random_unwrap(keys)

def key_data(keys: KeyArray) -> Array:
  keys, _ = _check_prng_key(keys)
  return _key_data(keys)


### random samplers


def _check_shape(name: str, shape: Union[Shape, NamedShape], *param_shapes) -> None:
  shape = core.as_named_shape(shape)

  if param_shapes:
    shape_ = lax.broadcast_shapes(shape.positional, *param_shapes)
    if shape.positional != shape_:
      msg = ("{} parameter shapes must be broadcast-compatible with shape "
             "argument, and the result of broadcasting the shapes must equal "
             "the shape argument, but got result {} for shape argument {}.")
      raise ValueError(msg.format(name, shape_, shape))


def _check_dtype(name: str, dtype: Optional[DTypeLike], *params,
                 default_to_int: bool = False,
                 require_inexact: bool = False,
                 require_float: bool = False,
                 require_int: bool = False) -> np.dtype:
  if require_float:
    require_inexact = True
  if require_int:
    default_to_int = True
  dtypes.check_user_dtype_supported(dtype, name)
  if dtype is None:
    if params:
      dtype = dtypes.result_type(*params)
      if require_int:
        dtype = dtypes.to_numeric_dtype(dtype)
      if require_inexact:
        dtype = dtypes.to_inexact_dtype(dtype)
    elif default_to_int and not require_inexact:
      dtype = dtypes.default_int()
    else:
      dtype = dtypes.default_float()
  dtype = dtypes.canonicalize_dtype(dtype)
  if require_int and not dtypes.issubdtype(dtype, np.integer):
    raise ValueError(f"dtype argument to `{name}` must be an int dtype, got {dtype}")
  if require_float and not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `{name}` must be a float dtype, got {dtype}")
  if require_inexact and not dtypes.issubdtype(dtype, np.inexact):
    raise ValueError(f"dtype argument to `{name}` must be an inexact dtype, got {dtype}")
  return np.dtype(dtype)


def uniform(key: KeyArray,
            shape: Union[Shape, NamedShape] = (),
            dtype: Optional[DTypeLikeFloat] = None,
            minval: RealArray = 0.,
            maxval: RealArray = 1.) -> Array:
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    minval: optional, a minimum (inclusive) value broadcast-compatible with shape for the range (default 0).
    maxval: optional, a maximum (exclusive) value broadcast-compatible with shape for the range (default 1).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("uniform", dtype, minval, maxval, require_float=True)
  shape = core.as_named_shape(shape)
  return _uniform(key, shape, dtype, minval, maxval)  # type: ignore

@partial(jit, static_argnums=(1, 2), inline=True)
def _uniform(key, shape, dtype, minval, maxval) -> Array:
  _check_shape("uniform", shape)
  minval = lax.convert_element_type(minval, dtype)
  maxval = lax.convert_element_type(maxval, dtype)
  minval = lax.broadcast_to_rank(minval, shape.positional_rank)
  maxval = lax.broadcast_to_rank(maxval, shape.positional_rank)

  finfo = jnp.finfo(dtype)
  nbits, nmant = finfo.bits, finfo.nmant

  if nbits not in (16, 32, 64):
    raise TypeError("uniform only accepts 32- or 64-bit dtypes.")

  rng_bits = nbits
  if nmant < 8:
    rng_bits = 8
  bits = _random_bits(key, rng_bits, shape)
  uint_dtype = UINT_DTYPES[nbits]
  if rng_bits != nbits:
    bits = lax.convert_element_type(bits, uint_dtype)

  # The strategy here is to randomize only the mantissa bits with an exponent of
  # 1 (after applying the bias), then shift and scale to the desired range. The
  # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
  # equivalent float representations, which might not be true on all platforms.
  float_bits = lax.bitwise_or(
      lax.shift_right_logical(bits, np.array(rng_bits - nmant, uint_dtype)),
      np.array(1.0, dtype).view(uint_dtype),
  )
  floats = lax.bitcast_convert_type(float_bits, dtype) - np.array(1., dtype)
  return lax.max(
      minval,
      lax.reshape(floats * (maxval - minval) + minval, shape.positional))


def randint(key: KeyArray,
            shape: Shape,
            minval: IntegerArray,
            maxval: IntegerArray,
            dtype: Optional[DTypeLikeInt] = None) -> Array:
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNG key used as the random key.
    shape: a tuple of nonnegative integers representing the shape.
    minval: int or array of ints broadcast-compatible with ``shape``, a minimum
      (inclusive) value for the range.
    maxval: int or array of ints broadcast-compatible with ``shape``, a maximum
      (exclusive) value for the range.
    dtype: optional, an int dtype for the returned values (default int64 if
      jax_enable_x64 is true, otherwise int32).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("randint", dtype, minval, maxval, require_int=True)
  shape = core.canonicalize_shape(shape)
  return _randint(key, shape, minval, maxval, dtype)

@partial(jit, static_argnums=(1, 4), inline=True)
def _randint(key, shape, minval, maxval, dtype) -> Array:
  _check_shape("randint", shape, np.shape(minval), np.shape(maxval))
  check_arraylike("randint", minval, maxval)
  minval = jnp.asarray(minval)
  maxval = jnp.asarray(maxval)
  if not jnp.issubdtype(minval.dtype, np.integer):
    minval = minval.astype(int)
  if not jnp.issubdtype(maxval.dtype, np.integer):
    maxval = maxval.astype(int)

  # Flag where maxval is greater than the maximum value of dtype
  # in order to handle cases like randint(key, shape, 0, 256, 'uint8')
  maxval_out_of_range = lax.gt(
    maxval, _convert_and_clip_integer(jnp.array(jnp.iinfo(dtype).max, dtype), maxval.dtype))

  minval = _convert_and_clip_integer(minval, dtype)
  maxval = _convert_and_clip_integer(maxval, dtype)
  minval = lax.broadcast_to_rank(minval, len(shape))
  maxval = lax.broadcast_to_rank(maxval, len(shape))
  nbits = jnp.iinfo(dtype).bits

  if nbits not in (8, 16, 32, 64):
    raise TypeError(f"randint only accepts 8-, 16-, 32-, or 64-bit dtypes, got {dtype}")

  # This algorithm is biased whenever (maxval - minval) is not a power of 2.
  # We generate double the number of random bits required by the dtype so as to
  # reduce that bias.
  k1, k2 = _split(key)
  rbits = lambda key: _random_bits(key, nbits, shape)
  higher_bits, lower_bits = rbits(k1), rbits(k2)

  unsigned_dtype = UINT_DTYPES[nbits]
  span = lax.convert_element_type(maxval - minval, unsigned_dtype)

  # Ensure that span=1 when maxval <= minval, so minval is always returned;
  # https://github.com/google/jax/issues/222
  span = lax.select(maxval <= minval, lax.full_like(span, 1), span)

  # When maxval is out of range, the span has to be one larger.
  # If span is already the maximum representable value, this will wrap to zero,
  # causing remainders below to have no effect, which is the correct semantics.
  span = lax.select(
    maxval_out_of_range & (maxval > minval),
    lax.add(span, _lax_const(span, 1)),
    span)

  # To compute a remainder operation on an integer that might have twice as many
  # bits as we can represent in the native unsigned dtype, we compute a
  # multiplier equal to 2**nbits % span. To avoid overflow, we use the identity:
  #  (a * b) % N = [(a % N) * (b % N)] % N
  multiplier = lax.rem(_lax_const(span, 2 ** (nbits // 2)), span)
  multiplier = lax.rem(lax.mul(multiplier, multiplier), span)

  random_offset = lax.add(lax.mul(lax.rem(higher_bits, span), multiplier),
                          lax.rem(lower_bits, span))
  random_offset = lax.rem(random_offset, span)
  return lax.add(minval, lax.convert_element_type(random_offset, dtype))


def shuffle(key: KeyArray, x: ArrayLike, axis: int = 0) -> Array:
  """Shuffle the elements of an array uniformly at random along an axis.

  Args:
    key: a PRNG key used as the random key.
    x: the array to be shuffled.
    axis: optional, an int axis along which to shuffle (default 0).

  Returns:
    A shuffled version of x.
  """
  msg = ("jax.random.shuffle is deprecated and will be removed in a future release. "
         "Use jax.random.permutation with independent=True.")
  warnings.warn(msg, FutureWarning)
  key, _ = _check_prng_key(key)
  return _shuffle(key, x, axis)  # type: ignore


def permutation(key: KeyArray,
                x: Union[int, ArrayLike],
                axis: int = 0,
                independent: bool = False) -> Array:
  """Returns a randomly permuted array or range.

  Args:
    key: a PRNG key used as the random key.
    x: int or array. If x is an integer, randomly shuffle np.arange(x).
      If x is an array, randomly shuffle its elements.
    axis: int, optional. The axis which x is shuffled along. Default is 0.
    independent: bool, optional. If set to True, each individual vector along
      the given axis is shuffled independently. Default is False.

  Returns:
    A shuffled version of x or array range
  """
  key, _ = _check_prng_key(key)
  check_arraylike("permutation", x)
  axis = canonicalize_axis(axis, np.ndim(x) or 1)
  if not np.ndim(x):
    if not np.issubdtype(lax.dtype(x), np.integer):
      raise TypeError("x must be an integer or at least 1-dimensional")
    r = core.concrete_or_error(int, x, 'argument x of jax.random.permutation()')
    return _shuffle(key, jnp.arange(r), axis)
  if independent or np.ndim(x) == 1:
    return _shuffle(key, x, axis)
  ind = _shuffle(key, jnp.arange(x.shape[axis]), 0)  # type: ignore[union-attr]
  return jnp.take(x, ind, axis, unique_indices=True)


@partial(jit, static_argnums=(2,), inline=True)
def _shuffle(key, x, axis) -> Array:
  # On parallel architectures, Fisher-Yates is more expensive than doing
  # multiple sorts. This algorithm is based on one developed and analyzed by
  # tjablin@. We sort according to randomly-generated 32bit keys, but those keys
  # may have collisions. If we repeat the process, using fresh 32bit keys for
  # each sort, then whenever all pairs of elements have been assigned distinct
  # keys at some iteration (or equivalently when the strings formed by
  # concatenating the successive keys for each element are all distinct) then we
  # are guaranteed to have a perfect sample (assuming that either the sort is
  # stable or that any bias is not value-dependent). Since checking uniqueness
  # at runtime may be expensive, we use a heuristic static stop criterion
  # developed by tjablin@. See tensorflow/compiler/tf2xla/random_ops.cc for more
  # info, and for the original implementation of this algorithm. See also
  # Section 2 of http://people.csail.mit.edu/costis/6896sp11/lec5s.pdf for
  # another analysis (where the keys are generated one bit at a time).
  exponent = 3  # see tjablin@'s analysis for explanation of this parameter
  uint32max = jnp.iinfo(np.uint32).max
  num_rounds = int(np.ceil(exponent * np.log(max(1, x.size)) / np.log(uint32max)))

  for _ in range(num_rounds):
    key, subkey = _split(key)
    sort_keys = _random_bits(subkey, 32, x.shape)
    _, x = lax.sort_key_val(sort_keys, x, axis)

  return x


def choice(key: KeyArray,
           a: Union[int, ArrayLike],
           shape: Shape = (),
           replace: bool = True,
           p: Optional[RealArray] = None,
           axis: int = 0) -> Array:
  """Generates a random sample from a given array.

  .. warning::
    If ``p`` has fewer non-zero elements than the requested number of samples,
    as specified in ``shape``, and ``replace=False``, the output of this
    function is ill-defined. Please make sure to use appropriate inputs.

  Args:
    key: a PRNG key used as the random key.
    a : array or int. If an ndarray, a random sample is generated from
      its elements. If an int, the random sample is generated as if a were
      arange(a).
    shape : tuple of ints, optional. Output shape.  If the given shape is,
      e.g., ``(m, n)``, then ``m * n`` samples are drawn.  Default is (),
      in which case a single value is returned.
    replace : boolean.  Whether the sample is with or without replacement.
      default is True.
    p : 1-D array-like, The probabilities associated with each entry in a.
      If not given the sample assumes a uniform distribution over all
      entries in a.
    axis: int, optional. The axis along which the selection is performed.
      The default, 0, selects by row.

  Returns:
    An array of shape `shape` containing samples from `a`.
  """
  key, _ = _check_prng_key(key)
  if not isinstance(shape, Sequence):
    raise TypeError("shape argument of jax.random.choice must be a sequence, "
                    f"got {shape}")
  check_arraylike("choice", a)
  arr = jnp.asarray(a)
  if arr.ndim == 0:
    n_inputs = core.concrete_or_error(int, a, "The error occurred in jax.random.choice()")
  else:
    axis = canonicalize_axis(axis, arr.ndim)
    n_inputs = arr.shape[axis]
  n_draws = math.prod(shape)
  if n_draws == 0:
    return jnp.zeros(shape, dtype=arr.dtype)
  if n_inputs <= 0:
    raise ValueError("a must be greater than 0 unless no samples are taken")
  if not replace and n_draws > n_inputs:
    raise ValueError("Cannot take a larger sample than population when 'replace=False'")

  if p is None:
    if replace:
      ind = randint(key, shape, 0, n_inputs)
      result = ind if arr.ndim == 0 else jnp.take(arr, ind, axis)
    else:
      slices = (slice(None),) * axis + (slice(n_draws),)
      result = permutation(key, n_inputs if arr.ndim == 0 else arr, axis)[slices]
  else:
    check_arraylike("choice", p)
    p_arr, = promote_dtypes_inexact(p)
    if p_arr.shape != (n_inputs,):
      raise ValueError("p must be None or match the shape of a")
    if replace:
      p_cuml = jnp.cumsum(p_arr)
      r = p_cuml[-1] * (1 - uniform(key, shape, dtype=p_cuml.dtype))
      ind = jnp.searchsorted(p_cuml, r)
    else:
      # Gumbel top-k trick: https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
      g = -gumbel(key, (n_inputs,), dtype=p_arr.dtype) - jnp.log(p_arr)
      ind = jnp.argsort(g)[:n_draws]
    result = ind if arr.ndim == 0 else jnp.take(arr, ind, axis)

  return result.reshape(shape if arr.ndim == 0 else
                        np.insert(np.delete(arr.shape, axis), axis, shape))


def normal(key: KeyArray,
           shape: Union[Shape, NamedShape] = (),
           dtype: Optional[DTypeLike] = None) -> Array:
  r"""Sample standard normal random values with given shape and float dtype.

  The values are returned according to the probability density function:

  .. math::
     f(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}

  on the domain :math:`-\infty < x < \infty`

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("normal", dtype, require_inexact=True)
  shape = core.as_named_shape(shape)
  return _normal(key, shape, dtype)  # type: ignore

@partial(jit, static_argnums=(1, 2), inline=True)
def _normal(key, shape, dtype) -> Array:
  if dtypes.issubdtype(dtype, np.complexfloating):
    sqrt2 = np.array(np.sqrt(2), dtype)

    key_re, key_im = _split(key)
    real_dtype = np.array(0, dtype).real.dtype
    _re = _normal_real(key_re, shape, real_dtype).astype(dtype)
    _im = _normal_real(key_im, shape, real_dtype).astype(dtype)
    return (_re + 1j * _im) / sqrt2
  else:
    return _normal_real(key, shape, dtype) # type: ignore

@partial(jit, static_argnums=(1, 2), inline=True)
def _normal_real(key, shape, dtype) -> Array:
  _check_shape("normal", shape)
  lo = np.nextafter(np.array(-1., dtype), np.array(0., dtype), dtype=dtype)
  hi = np.array(1., dtype)
  u = uniform(key, shape, dtype, lo, hi)  # type: ignore[arg-type]
  return lax.mul(np.array(np.sqrt(2), dtype), lax.erf_inv(u))


def multivariate_normal(key: KeyArray,
                        mean: RealArray,
                        cov: RealArray,
                        shape: Optional[Shape] = None,
                        dtype: Optional[DTypeLike] = None,
                        method: str = 'cholesky') -> Array:
  r"""Sample multivariate normal random values with given mean and covariance.

  The values are returned according to the probability density function:

  .. math::
     f(x;\mu, \Sigma) = (2\pi)^{-k/2} \det(\Sigma)^{-1}e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)}

  where :math:`k` is the dimension, :math:`\mu` is the mean (given by ``mean``) and
  :math:`\Sigma` is the covariance matrix (given by ``cov``).

  Args:
    key: a PRNG key used as the random key.
    mean: a mean vector of shape ``(..., n)``.
    cov: a positive definite covariance matrix of shape ``(..., n, n)``. The
      batch shape ``...`` must be broadcast-compatible with that of ``mean``.
    shape: optional, a tuple of nonnegative integers specifying the result
      batch shape; that is, the prefix of the result shape excluding the last
      axis. Must be broadcast-compatible with ``mean.shape[:-1]`` and
      ``cov.shape[:-2]``. The default (None) produces a result batch shape by
      broadcasting together the batch shapes of ``mean`` and ``cov``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    method: optional, a method to compute the factor of ``cov``.
      Must be one of 'svd', 'eigh', and 'cholesky'. Default 'cholesky'. For
      singular covariance matrices, use 'svd' or 'eigh'.
  Returns:
    A random array with the specified dtype and shape given by
    ``shape + mean.shape[-1:]`` if ``shape`` is not None, or else
    ``broadcast_shapes(mean.shape[:-1], cov.shape[:-2]) + mean.shape[-1:]``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("multivariate_normal", dtype, mean, cov, require_float=True)
  if method not in {'svd', 'eigh', 'cholesky'}:
    raise ValueError("method must be one of {'svd', 'eigh', 'cholesky'}")
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _multivariate_normal(key, mean, cov, shape, dtype, method)  # type: ignore

@partial(jit, static_argnums=(3, 4, 5), inline=True)
def _multivariate_normal(key, mean, cov, shape, dtype, method) -> Array:
  if not np.ndim(mean) >= 1:
    msg = "multivariate_normal requires mean.ndim >= 1, got mean.ndim == {}"
    raise ValueError(msg.format(np.ndim(mean)))
  if not np.ndim(cov) >= 2:
    msg = "multivariate_normal requires cov.ndim >= 2, got cov.ndim == {}"
    raise ValueError(msg.format(np.ndim(cov)))
  n = mean.shape[-1]
  if np.shape(cov)[-2:] != (n, n):
    msg = ("multivariate_normal requires cov.shape == (..., n, n) for n={n}, "
           "but got cov.shape == {shape}.")
    raise ValueError(msg.format(n=n, shape=np.shape(cov)))

  if shape is None:
    shape = lax.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
  else:
    _check_shape("normal", shape, mean.shape[:-1], cov.shape[:-2])

  if method == 'svd':
    (u, s, _) = svd(cov)
    factor = u * jnp.sqrt(s[..., None, :])
  elif method == 'eigh':
    (w, v) = eigh(cov)
    factor = v * jnp.sqrt(w[..., None, :])
  else: # 'cholesky'
    factor = cholesky(cov)
  normal_samples = normal(key, shape + mean.shape[-1:], dtype)
  with config_lib.numpy_rank_promotion('allow'):
    result = mean + jnp.einsum('...ij,...j->...i', factor, normal_samples)
  return result


def truncated_normal(key: KeyArray,
                     lower: RealArray,
                     upper: RealArray,
                     shape: Optional[Union[Shape, NamedShape]] = None,
                     dtype: Optional[DTypeLike] = None) -> Array:
  r"""Sample truncated standard normal random values with given shape and dtype.

  The values are returned according to the probability density function:

  .. math::
     f(x) \propto e^{-x^2/2}

  on the domain :math:`\rm{lower} < x < \rm{upper}`.

  Args:
    key: a PRNG key used as the random key.
    lower: a float or array of floats representing the lower bound for
      truncation. Must be broadcast-compatible with ``upper``.
    upper: a float or array of floats representing the  upper bound for
      truncation. Must be broadcast-compatible with ``lower``.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``lower`` and ``upper``. The
      default (None) produces a result shape by broadcasting ``lower`` and
      ``upper``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``lower`` and ``upper``.
    Returns values in the open interval ``(lower, upper)``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("truncated_normal", dtype, lower, upper, require_float=True)
  if shape is not None:
    shape = core.as_named_shape(shape)
  return _truncated_normal(key, lower, upper, shape, dtype)  # type: ignore

@partial(jit, static_argnums=(3, 4), inline=True)
def _truncated_normal(key, lower, upper, shape, dtype) -> Array:
  if shape is None:
    shape = lax.broadcast_shapes(np.shape(lower), np.shape(upper))
  else:
    _check_shape("truncated_normal", shape, np.shape(lower), np.shape(upper))

  sqrt2 = np.array(np.sqrt(2), dtype)
  lower = lax.convert_element_type(lower, dtype)
  upper = lax.convert_element_type(upper, dtype)
  a = lax.erf(lower / sqrt2)
  b = lax.erf(upper / sqrt2)
  u = uniform(key, shape, dtype, minval=a, maxval=b)
  out = sqrt2 * lax.erf_inv(u)
  # Clamp the value to the open interval (lower, upper) to make sure that
  # rounding (or if we chose `a` for `u`) doesn't push us outside of the range.
  return jnp.clip(
      out,
      lax.nextafter(lax.stop_gradient(lower), np.array(np.inf, dtype=dtype)),
      lax.nextafter(lax.stop_gradient(upper), np.array(-np.inf, dtype=dtype)))


def bernoulli(key: KeyArray,
              p: RealArray = np.float32(0.5),
              shape: Optional[Union[Shape, NamedShape]] = None,
              dtype: Optional[DTypeLike] = None) -> Array:
  r"""Sample Bernoulli random values with given shape and mean.

  The values are distributed according to the probability mass function:

  .. math::
     f(k; p) = p^k(1 - p)^{1 - k}

  where :math:`k \in \{0, 1\}` and :math:`0 \le p \le 1`.

  Args:
    key: a PRNG key used as the random key.
    p: optional, a float or array of floats for the mean of the random
      variables. Must be broadcast-compatible with ``shape``. Default 0.5.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Must be broadcast-compatible with ``p.shape``. The default (None)
      produces a result shape equal to ``p.shape``.

  Returns:
    A random array with boolean dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``p.shape``.
  """
  key, _ = _check_prng_key(key)
  dtype = dtypes.canonicalize_dtype(lax.dtype(p))
  if shape is not None:
    shape = core.as_named_shape(shape)
  if not jnp.issubdtype(dtype, np.floating):
    msg = "bernoulli probability `p` must have a floating dtype, got {}."
    raise TypeError(msg.format(dtype))
  p = lax.convert_element_type(p, dtype)
  return _bernoulli(key, p, shape)  # type: ignore

@partial(jit, static_argnums=(2,), inline=True)
def _bernoulli(key, p, shape) -> Array:
  if shape is None:
    # TODO: Use the named part of `p` as well
    shape = np.shape(p)
  else:
    _check_shape("bernoulli", shape, np.shape(p))

  return uniform(key, shape, lax.dtype(p)) < p


def beta(key: KeyArray,
         a: RealArray,
         b: RealArray,
         shape: Optional[Shape] = None,
         dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Beta random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x;a,b) \propto x^{a - 1}(1 - x)^{b - 1}

  on the domain :math:`0 \le x \le 1`.

  Args:
    key: a PRNG key used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the first parameter "alpha".
    b: a float or array of floats broadcast-compatible with ``shape``
      representing the second parameter "beta".
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``a`` and ``b``. The default
      (None) produces a result shape by broadcasting ``a`` and ``b``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``a`` and ``b``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("bernoulli", dtype, a, b, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _beta(key, a, b, shape, dtype)


def _beta(key, a, b, shape, dtype) -> Array:
  if shape is None:
    shape = lax.broadcast_shapes(np.shape(a), np.shape(b))
  else:
    _check_shape("beta", shape, np.shape(a), np.shape(b))

  a = lax.convert_element_type(a, dtype)
  b = lax.convert_element_type(b, dtype)
  key_a, key_b = _split(key)
  a = jnp.broadcast_to(a, shape)
  b = jnp.broadcast_to(b, shape)
  log_gamma_a = loggamma(key_a, a, shape, dtype)
  log_gamma_b = loggamma(key_b, b, shape, dtype)
  # Compute gamma_a / (gamma_a + gamma_b) without losing precision.
  log_max = lax.max(log_gamma_a, log_gamma_b)
  gamma_a_scaled = jnp.exp(log_gamma_a - log_max)
  gamma_b_scaled = jnp.exp(log_gamma_b - log_max)
  return gamma_a_scaled / (gamma_a_scaled + gamma_b_scaled)


def cauchy(key: KeyArray,
           shape: Shape = (),
           dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Cauchy random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x) \propto \frac{1}{x^2 + 1}

  on the domain :math:`-\infty < x < \infty`

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("cauchy", dtype, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _cauchy(key, shape, dtype)

@partial(jit, static_argnums=(1, 2), inline=True)
def _cauchy(key, shape, dtype) -> Array:
  _check_shape("cauchy", shape)
  u = uniform(key, shape, dtype, minval=jnp.finfo(dtype).eps, maxval=1.)
  pi = _lax_const(u, np.pi)
  return lax.tan(lax.mul(pi, lax.sub(u, _lax_const(u, 0.5))))


def dirichlet(key: KeyArray,
              alpha: RealArray,
              shape: Optional[Shape] = None,
              dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Dirichlet random values with given shape and float dtype.

  The values are distributed according the the probability density function:

  .. math::
     f(\{x_i\}; \{\alpha_i\}) = \propto \prod_{i=1}^k x_i^{\alpha_i}

  Where :math:`k` is the dimension, and :math:`\{x_i\}` satisfies

  .. math::
     \sum_{i=1}^k x_i = 1

  and :math:`0 \le x_i \le 1` for all :math:`x_i`.

  Args:
    key: a PRNG key used as the random key.
    alpha: an array of shape ``(..., n)`` used as the concentration
      parameter of the random variables.
    shape: optional, a tuple of nonnegative integers specifying the result
      batch shape; that is, the prefix of the result shape excluding the last
      element of value ``n``. Must be broadcast-compatible with
      ``alpha.shape[:-1]``. The default (None) produces a result shape equal to
      ``alpha.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and shape given by
    ``shape + (alpha.shape[-1],)`` if ``shape`` is not None, or else
    ``alpha.shape``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("dirichlet", dtype, alpha, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _dirichlet(key, alpha, shape, dtype)

@partial(jit, static_argnums=(2, 3), inline=True)
def _dirichlet(key, alpha, shape, dtype) -> Array:
  if not np.ndim(alpha) >= 1:
    msg = "dirichlet requires alpha.ndim >= 1, got alpha.ndim == {}"
    raise ValueError(msg.format(np.ndim(alpha)))

  if shape is None:
    shape = np.shape(alpha)[:-1]
  else:
    _check_shape("dirichlet", shape, np.shape(alpha)[:-1])

  alpha = lax.convert_element_type(alpha, dtype)

  # Compute gamma in log space, otherwise small alpha can lead to poor behavior.
  log_gamma_samples = loggamma(key, alpha, shape + np.shape(alpha)[-1:], dtype)
  return _softmax(log_gamma_samples, -1)


def _softmax(x, axis) -> Array:
  """Utility to compute the softmax of x along a given axis."""
  if not dtypes.issubdtype(x.dtype, np.floating):
    raise TypeError(f"_softmax only accepts floating dtypes, got {x.dtype}")
  x_max = jnp.max(x, axis, keepdims=True)
  unnormalized = jnp.exp(x - lax.stop_gradient(x_max))
  return unnormalized / unnormalized.sum(axis, keepdims=True)


def exponential(key: KeyArray,
                shape: Shape = (),
                dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Exponential random values with given shape and float dtype.

  The values are distributed according the the probability density function:

  .. math::
     f(x) = e^{-x}

  on the domain :math:`0 \le x < \infty`.

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("exponential", dtype, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _exponential(key, shape, dtype)

@partial(jit, static_argnums=(1, 2), inline=True)
def _exponential(key, shape, dtype) -> Array:
  _check_shape("exponential", shape)
  u = uniform(key, shape, dtype)
  # taking 1 - u to move the domain of log to (0, 1] instead of [0, 1)
  return lax.neg(lax.log1p(lax.neg(u)))


def _gamma_one(key: KeyArray, alpha, log_space) -> Array:
  # Ref: A simple method for generating gamma variables, George Marsaglia and Wai Wan Tsang
  # The algorithm can also be founded in:
  # https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables
  zero = _lax_const(alpha, 0)
  one = _lax_const(alpha, 1)
  minus_one = _lax_const(alpha, -1)
  one_over_two = _lax_const(alpha, 0.5)
  one_over_three = _lax_const(alpha, 1. / 3.)
  squeeze_const = _lax_const(alpha, 0.0331)
  dtype = lax.dtype(alpha)

  # for alpha < 1, we boost alpha to alpha + 1 and get a sample according to
  #   Gamma(alpha) ~ Gamma(alpha+1) * Uniform()^(1 / alpha)
  # When alpha is very small, this boost can be problematic because it may result
  # in floating point underflow; for this reason we compute it in log space if
  # specified by the `log_space` argument:
  #   log[Gamma(alpha)] ~ log[Gamma(alpha + 1)] + log[Uniform()] / alpha
  # Note that log[Uniform()] ~ Exponential(), but the exponential() function is
  # computed via log[1 - Uniform()] to avoid taking log(0). We want the generated
  # sequence to match between log_space=True and log_space=False, so we avoid this
  # for now to maintain backward compatibility with the original implementation.
  # TODO(jakevdp) should we change the convention to avoid -inf in log-space?
  boost_mask = lax.ge(alpha, one)
  alpha_orig = alpha
  alpha = lax.select(boost_mask, alpha, lax.add(alpha, one))

  d = lax.sub(alpha, one_over_three)
  c = lax.div(one_over_three, lax.sqrt(d))

  def _cond_fn(kXVU):
    _, X, V, U = kXVU
    # TODO: use lax.cond when its batching rule is supported
    # The reason is to avoid evaluating second condition which involves log+log
    # if the first condition is satisfied
    cond = lax.bitwise_and(lax.ge(U, lax.sub(one, lax.mul(squeeze_const, lax.mul(X, X)))),
                           lax.ge(lax.log(U), lax.add(lax.mul(X, one_over_two),
                                                      lax.mul(d, lax.add(lax.sub(one, V),
                                                                         lax.log(V))))))
    return cond

  def _body_fn(kXVU):
    def _next_kxv(kxv):
      key = kxv[0]
      key, subkey = _split(key)
      x = normal(subkey, (), dtype=dtype)
      v = lax.add(one, lax.mul(x, c))
      return key, x, v

    key = kXVU[0]
    key, x_key, U_key = _split(key, 3)
    _, x, v = lax.while_loop(lambda kxv: lax.le(kxv[2], zero), _next_kxv, (x_key, zero, minus_one))
    X = lax.mul(x, x)
    V = lax.mul(lax.mul(v, v), v)
    U = uniform(U_key, (), dtype=dtype)
    return key, X, V, U

  # initial state is chosen such that _cond_fn will return True
  key, subkey = _split(key)
  u_boost = uniform(subkey, (), dtype=dtype)
  _, _, V, _ = lax.while_loop(_cond_fn, _body_fn, (key, zero, one, _lax_const(alpha, 2)))
  if log_space:
    # TODO(jakevdp): there are negative infinities here due to issues mentioned above. How should
    # we handle those?
    log_boost = lax.select(boost_mask, zero, lax.mul(lax.log(u_boost), lax.div(one, alpha_orig)))
    return lax.add(lax.add(lax.log(d), lax.log(V)), log_boost)
  else:
    boost = lax.select(boost_mask, one, lax.pow(u_boost, lax.div(one, alpha_orig)))
    z = lax.mul(lax.mul(d, V), boost)
    return lax.select(lax.eq(z, zero), jnp.finfo(z.dtype).tiny, z)


def _gamma_grad(sample, a, *, log_space):
  samples = jnp.reshape(sample, -1)
  alphas = jnp.reshape(a, -1)
  if log_space:
    # d[log(sample)] = d[sample] / sample
    # This requires computing exp(log_sample), which may be zero due to float roundoff.
    # In this case, we use the same zero-correction used in gamma() above.
    samples = lax.exp(samples)
    zero = lax_internal._const(sample, 0)
    tiny = lax.full_like(samples, jnp.finfo(samples.dtype).tiny)
    samples = lax.select(lax.eq(samples, zero), tiny, samples)
    gamma_grad = lambda alpha, sample: lax.random_gamma_grad(alpha, sample) / sample
  else:
    gamma_grad = lax.random_gamma_grad
  if xla_bridge.get_backend().platform == 'cpu':
    grads = lax.map(lambda args: gamma_grad(*args), (alphas, samples))
  else:
    grads = vmap(gamma_grad)(alphas, samples)
  return grads.reshape(np.shape(a))

def _gamma_impl(key, a, *, log_space, use_vmap=False):
  # split key to match the shape of a
  a_shape = jnp.shape(a)
  split_count = math.prod(a_shape[key.ndim:])
  keys = key.flatten()
  keys = vmap(_split, in_axes=(0, None))(keys, split_count)
  keys = keys.flatten()
  alphas = a.flatten()

  if use_vmap:
    samples = vmap(partial(_gamma_one, log_space=log_space))(keys, alphas)
  else:
    samples = lax.map(
        lambda args: _gamma_one(*args, log_space=log_space), (keys, alphas))

  return jnp.reshape(samples, a_shape)

def _gamma_batching_rule(batched_args, batch_dims, *, log_space):
  k, a = batched_args
  bk, ba = batch_dims
  size = next(
      t.shape[i] for t, i in zip(batched_args, batch_dims) if i is not None)
  k = batching.bdim_at_front(k, bk, size)
  a = batching.bdim_at_front(a, ba, size)
  return random_gamma_p.bind(k, a, log_space=log_space), 0

random_gamma_p = core.Primitive('random_gamma')
random_gamma_p.def_impl(_gamma_impl)
random_gamma_p.def_abstract_eval(lambda key, a, **_: core.raise_to_shaped(a))
ad.defjvp2(
    random_gamma_p, None,
    lambda tangent, ans, key, a, **kwds: tangent * _gamma_grad(ans, a, **kwds))
mlir.register_lowering(random_gamma_p, mlir.lower_fun(
    partial(_gamma_impl, use_vmap=True),
    multiple_results=False))
mlir.register_lowering(random_gamma_p, mlir.lower_fun(
    partial(_gamma_impl, use_vmap=False),
    multiple_results=False), platform='cpu')
batching.primitive_batchers[random_gamma_p] = _gamma_batching_rule

def gamma(key: KeyArray,
          a: RealArray,
          shape: Optional[Shape] = None,
          dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Gamma random values with given shape and float dtype.

  The values are distributed according the the probability density function:

  .. math::
     f(x;a) \propto x^{a - 1} e^{-x}

  on the domain :math:`0 \le x < \infty`, with :math:`a > 0`.

  This is the standard gamma density, with a unit scale/rate parameter.
  Dividing the sample output by the rate is equivalent to sampling from
  *gamma(a, rate)*, and multiplying the sample output by the scale is equivalent
  to sampling from *gamma(a, scale)*.

  Args:
    key: a PRNG key used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``a``. The default (None)
      produces a result shape equal to ``a.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``a.shape``.

  See Also:
    loggamma : sample gamma values in log-space, which can provide improved
      accuracy for small values of ``a``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("gamma", dtype, a, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _gamma(key, a, shape=shape, dtype=dtype)


def loggamma(key: KeyArray,
             a: RealArray,
             shape: Optional[Shape] = None,
             dtype: Optional[DTypeLikeFloat] = None) -> Array:
  """Sample log-gamma random values with given shape and float dtype.

  This function is implemented such that the following will hold for a
  dtype-appropriate tolerance::

    np.testing.assert_allclose(jnp.exp(loggamma(*args)), gamma(*args), rtol=rtol)

  The benefit of log-gamma is that for samples very close to zero (which occur frequently
  when `a << 1`) sampling in log space provides better precision.

  Args:
    key: a PRNG key used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``a``. The default (None)
      produces a result shape equal to ``a.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``a.shape``.

  See Also:
    gamma : standard gamma sampler.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("loggamma", dtype, a, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _gamma(key, a, shape=shape, dtype=dtype, log_space=True)


@partial(jit, static_argnames=('shape', 'dtype', 'log_space'), inline=True)
def _gamma(key, a, shape, dtype, log_space=False) -> Array:
  if shape is None:
    shape = np.shape(a)
  else:
    _check_shape("gamma", shape, np.shape(a))

  a = lax.convert_element_type(a, dtype)
  if np.shape(a) != shape:
    a = jnp.broadcast_to(a, shape)
  return random_gamma_p.bind(key, a, log_space=log_space)


@partial(jit, static_argnums=(2, 3, 4), inline=True)
def _poisson_knuth(key, lam, shape, dtype, max_iters) -> Array:
  # Knuth's algorithm for generating Poisson random variates.
  # Reference:
  # https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables

  def body_fn(carry):
    i, k, rng, log_prod = carry
    rng, subkey = _split(rng)
    k = lax.select(log_prod > -lam, k + 1, k)
    u = uniform(subkey, shape, np.float32)
    return i + 1, k, rng, log_prod + jnp.log(u)

  def cond_fn(carry):
    i, log_prod = carry[0], carry[3]
    return (log_prod > -lam).any() & (i < max_iters)

  k_init = lax.full_like(lam, 0, dtype, shape)
  log_rate_init = lax.full_like(lam, 0, np.float32, shape)
  k = lax.while_loop(cond_fn, body_fn, (0, k_init, key, log_rate_init))[1]
  return (k - 1).astype(dtype)


@partial(jit, static_argnums=(2, 3, 4), inline=True)
def _poisson_rejection(key, lam, shape, dtype, max_iters) -> Array:
  # Transformed rejection due to Hormann.
  # Reference:
  # http://citeseer.ist.psu.edu/viewdoc/citations;jsessionid=1BEB35946CC807879F55D42512E5490C?doi=10.1.1.48.3054.
  log_lam = lax.log(lam)
  b = 0.931 + 2.53 * lax.sqrt(lam)
  a = -0.059 + 0.02483 * b
  inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
  v_r = 0.9277 - 3.6224 / (b - 2)

  def body_fn(carry):
    i, k_out, accepted, key = carry
    key, subkey_0, subkey_1 = _split(key, 3)

    u = uniform(subkey_0, shape, lam.dtype) - 0.5
    v = uniform(subkey_1, shape, lam.dtype)
    u_shifted = 0.5 - abs(u)

    k = lax.floor((2 * a / u_shifted + b) * u + lam + 0.43)
    s = lax.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
    t = -lam + k * log_lam - lax.lgamma(k + 1)

    accept1 = (u_shifted >= 0.07) & (v <= v_r)
    reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
    accept2 = s <= t
    accept = accept1 | (~reject & accept2)

    k_out = lax.select(accept, k, k_out)
    accepted |= accept

    return i + 1, k_out, accepted, key

  def cond_fn(carry):
    i, k_out, accepted, key = carry
    return (~accepted).any() & (i < max_iters)

  k_init = lax.full_like(lam, -1, lam.dtype, shape)
  accepted = lax.full_like(lam, False, jnp.bool_, shape)
  k = lax.while_loop(cond_fn, body_fn, (0, k_init, accepted, key))[1]
  return k.astype(dtype)


@partial(jit, static_argnums=(2, 3), inline=True)
def _poisson(key, lam, shape, dtype) -> Array:
  # The implementation matches TensorFlow and NumPy:
  # https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/core/kernels/random_poisson_op.cc
  # https://github.com/numpy/numpy/blob/v1.18.3/numpy/random/src/distributions/distributions.c#L574
  # For lambda < 10, we use the Knuth algorithm; otherwise, we use transformed
  # rejection sampling.
  use_knuth = _isnan(lam) | (lam < 10)
  lam_knuth = lax.select(use_knuth, lam, lax.full_like(lam, 0.0))
  # The acceptance probability for rejection sampling maxes out at 89% as
  # λ -> ∞, so pick some arbitrary large value.
  lam_rejection = lax.select(use_knuth, lax.full_like(lam, 1e5), lam)
  max_iters = dtype.type(jnp.iinfo(dtype).max)  # insanely conservative
  result = lax.select(
    use_knuth,
    _poisson_knuth(key, lam_knuth, shape, dtype, max_iters),
    _poisson_rejection(key, lam_rejection, shape, dtype, max_iters),
  )
  return lax.select(lam == 0, jnp.zeros_like(result), result)


def poisson(key: KeyArray,
            lam: RealArray,
            shape: Optional[Shape] = None,
            dtype: Optional[DTypeLikeInt] = None) -> Array:
  r"""Sample Poisson random values with given shape and integer dtype.

  The values are distributed according to the probability mass function:

  .. math::
     f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

  Where `k` is a non-negative integer and :math:`\lambda > 0`.

  Args:
    key: a PRNG key used as the random key.
    lam: rate parameter (mean of the distribution), must be >= 0. Must be broadcast-compatible with ``shape``
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default (None) produces a result shape equal to ``lam.shape``.
    dtype: optional, a integer dtype for the returned values (default int64 if
      jax_enable_x64 is true, otherwise int32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape is not None, or else by ``lam.shape``.
  """
  key, _ = _check_prng_key(key)
  # TODO(frostig): generalize underlying poisson implementation and
  # remove this check
  key_impl = key.dtype.impl  # type: ignore[union-attr]
  if key_impl is not prng.threefry_prng_impl:
    raise NotImplementedError(
        '`poisson` is only implemented for the threefry2x32 RNG, '
        f'not {key_impl}')
  dtype = _check_dtype("poisson", dtype, require_int=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  else:
    shape = np.shape(lam)
  lam = jnp.broadcast_to(lam, shape)
  lam = lax.convert_element_type(lam, np.float32)
  return _poisson(key, lam, shape, dtype)


def gumbel(key: KeyArray,
           shape: Shape = (),
           dtype: Optional[DTypeLikeFloat] = None) -> Array:
  """Sample Gumbel random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x) = e^{-(x + e^{-x})}

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("gumbel", dtype, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _gumbel(key, shape, dtype)

@partial(jit, static_argnums=(1, 2), inline=True)
def _gumbel(key, shape, dtype) -> Array:
  _check_shape("gumbel", shape)
  return -jnp.log(-jnp.log(
      uniform(key, shape, dtype, minval=jnp.finfo(dtype).tiny, maxval=1.)))


def categorical(key: KeyArray,
                logits: RealArray,
                axis: int = -1,
                shape: Optional[Shape] = None) -> Array:
  """Sample random values from categorical distributions.

  Args:
    key: a PRNG key used as the random key.
    logits: Unnormalized log probabilities of the categorical distribution(s) to sample from,
      so that `softmax(logits, axis)` gives the corresponding probabilities.
    axis: Axis along which logits belong to the same categorical distribution.
    shape: Optional, a tuple of nonnegative integers representing the result shape.
      Must be broadcast-compatible with ``np.delete(logits.shape, axis)``.
      The default (None) produces a result shape equal to ``np.delete(logits.shape, axis)``.

  Returns:
    A random array with int dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``np.delete(logits.shape, axis)``.
  """
  key, _ = _check_prng_key(key)
  check_arraylike("categorical", logits)
  logits_arr = jnp.asarray(logits)

  if axis >= 0:
    axis -= len(logits_arr.shape)

  batch_shape = tuple(np.delete(logits_arr.shape, axis))
  if shape is None:
    shape = batch_shape
  else:
    shape = tuple(shape)
    _check_shape("categorical", shape, batch_shape)

  shape_prefix = shape[:len(shape)-len(batch_shape)]
  logits_shape = list(shape[len(shape) - len(batch_shape):])
  logits_shape.insert(axis % len(logits_arr.shape), logits_arr.shape[axis])
  return jnp.argmax(
      gumbel(key, (*shape_prefix, *logits_shape), logits_arr.dtype) +
      lax.expand_dims(logits_arr, tuple(range(len(shape_prefix)))),
      axis=axis)


def laplace(key: KeyArray,
            shape: Shape = (),
            dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Laplace random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
    f(x) = \frac{1}{2}e^{-|x|}

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("laplace", dtype, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _laplace(key, shape, dtype)

@partial(jit, static_argnums=(1, 2), inline=True)
def _laplace(key, shape, dtype) -> Array:
  _check_shape("laplace", shape)
  u = uniform(
      key, shape, dtype, minval=-1. + jnp.finfo(dtype).epsneg, maxval=1.)
  return lax.mul(lax.sign(u), lax.log1p(lax.neg(lax.abs(u))))


def logistic(key: KeyArray,
             shape: Shape = (),
             dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample logistic random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x) = \frac{e^{-x}}{(1 + e^{-x})^2}

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("logistic", dtype, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _logistic(key, shape, dtype)

@partial(jit, static_argnums=(1, 2), inline=True)
def _logistic(key, shape, dtype):
  _check_shape("logistic", shape)
  x = uniform(key, shape, dtype, minval=jnp.finfo(dtype).eps, maxval=1.)
  return lax.log(lax.div(x, lax.sub(_lax_const(x, 1), x)))


def pareto(key: KeyArray,
           b: RealArray,
           shape: Optional[Shape] = None,
           dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Pareto random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x; b) = b / x^{b + 1}

  on the domain :math:`1 \le x < \infty` with :math:`b > 0`

  Args:
    key: a PRNG key used as the random key.
    b: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``b``. The default (None)
      produces a result shape equal to ``b.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``b.shape``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("pareto", dtype, b, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _pareto(key, b, shape, dtype)

@partial(jit, static_argnums=(2, 3), inline=True)
def _pareto(key, b, shape, dtype) -> Array:
  if shape is None:
    shape = np.shape(b)
  else:
    _check_shape("pareto", shape)

  b = lax.convert_element_type(b, dtype)
  e = exponential(key, shape, dtype)
  return lax.exp(e / b)


def t(key: KeyArray,
      df: RealArray,
      shape: Shape = (),
      dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Student's t random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(t; \nu) \propto \left(1 + \frac{t^2}{\nu}\right)^{-(\nu + 1)/2}

  Where :math:`\nu > 0` is the degrees of freedom, given by the parameter ``df``.

  Args:
    key: a PRNG key used as the random key.
    df: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``df``. The default (None)
      produces a result shape equal to ``df.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``df.shape``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("t", dtype, df, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _t(key, df, shape, dtype)

@partial(jit, static_argnums=(2, 3), inline=True)
def _t(key, df, shape, dtype) -> Array:
  if shape is None:
    shape = np.shape(df)
  else:
    _check_shape("t", shape, np.shape(df))

  df = lax.convert_element_type(df, dtype)
  key_n, key_g = _split(key)
  n = normal(key_n, shape, dtype)
  two = _lax_const(n, 2)
  half_df = lax.div(df, two)
  g = gamma(key_g, half_df, shape, dtype)
  return n * jnp.sqrt(half_df / g)


def chisquare(key: KeyArray,
              df: RealArray,
              shape: Optional[Shape] = None,
              dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Chisquare random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x; \nu) \propto x^{k/2 - 1}e^{-x/2}

  on the domain :math:`0 < x < \infty`, where :math:`\nu > 0` represents the
  degrees of freedom, given by the parameter ``df``.

  Args:
    key: a PRNG key used as the random key.
    df: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``df``. The default (None)
      produces a result shape equal to ``df.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``df.shape``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("chisquare", dtype, df, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _chisquare(key, df, shape, dtype)

@partial(jit, static_argnums=(2, 3), inline=True)
def _chisquare(key, df, shape, dtype) -> Array:
  if shape is None:
    shape = np.shape(df)
  else:
    _check_shape("chisquare", shape, np.shape(df))
  df = lax.convert_element_type(df, dtype)
  two = _lax_const(df, 2)
  half_df = lax.div(df, two)
  log_g = loggamma(key, a=half_df, shape=shape, dtype=dtype)
  chi2 = lax.mul(jnp.exp(log_g), two)
  return chi2


def f(key: KeyArray,
      dfnum: RealArray,
      dfden: RealArray,
      shape: Optional[Shape] = None,
      dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample F-distribution random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x; \nu) \propto x^{\nu_1/2 - 1}\left(1 + \frac{\nu_1}{\nu_2}x\right)^{
      -(\nu_1 + \nu_2) / 2}

  on the domain :math:`0 < x < \infty`. Here :math:`\nu_1` is the degrees of
  freedom of the numerator (``dfnum``), and :math:`\nu_2` is the degrees of
  freedom of the denominator (``dfden``).

  Args:
    key: a PRNG key used as the random key.
    dfnum: a float or array of floats broadcast-compatible with ``shape``
      representing the numerator's ``df`` of the distribution.
    dfden: a float or array of floats broadcast-compatible with ``shape``
      representing the denominator's ``df`` of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``dfnum`` and ``dfden``.
      The default (None) produces a result shape equal to ``dfnum.shape``,
      and ``dfden.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``df.shape``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("f", dtype, dfnum, dfden, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _f(key, dfnum, dfden, shape, dtype)

@partial(jit, static_argnums=(3, 4), inline=True)
def _f(key, dfnum, dfden, shape, dtype) -> Array:
  if shape is None:
    shape =  lax.broadcast_shapes(np.shape(dfden), np.shape(dfnum))
  else:
    _check_shape("f", shape, np.shape(dfden), np.shape(dfnum))
  dfden = lax.convert_element_type(dfden, dtype)
  dfnum = lax.convert_element_type(dfnum, dtype)
  key_dfd, key_dfn = _split(key)
  chi2_dfn = chisquare(key_dfn, dfnum, shape, dtype)
  chi2_dfd = chisquare(key_dfd, dfden, shape, dtype)
  # broadcast dfden and dfnum to do div operation
  dfden = jnp.broadcast_to(dfden, shape)
  dfnum = jnp.broadcast_to(dfnum, shape)
  num = lax.div(chi2_dfn, dfnum)
  den = lax.div(chi2_dfd ,dfden)
  f = lax.div(num, den)
  return f


def rademacher(key: KeyArray,
               shape: Shape,
               dtype: Optional[DTypeLike] = None) -> Array:
  r"""Sample from a Rademacher distribution.

  The values are distributed according to the probability mass function:

  .. math::
     f(k) = \frac{1}{2}(\delta(k - 1) + \delta(k + 1))

  on the domain :math:`k \in \{-1, 1}`, where `\delta(x)` is the dirac delta function.

  Args:
    key: a PRNG key.
    shape: The shape of the returned samples.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples, of shape `shape`. Each element in the output has
    a 50% change of being 1 or -1.

  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("rademacher", dtype, default_to_int=True)
  shape = core.canonicalize_shape(shape)
  return _rademacher(key, shape, dtype)


@partial(jit, static_argnums=(1, 2), inline=True)
def _rademacher(key, shape, dtype) -> Array:
  bernoulli_samples = bernoulli(key=key, p=0.5, shape=shape).astype(dtype)
  return (2 * bernoulli_samples - 1).astype(dtype)


def maxwell(key: KeyArray,
            shape: Shape = (),
            dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample from a one sided Maxwell distribution.

  The values are distributed according to the probability density function:

  .. math::
     f(x) \propto x^2 e^{-x^2 / 2}

  on the domain :math:`0 \le x < \infty`.

  Args:
    key: a PRNG key.
    shape: The shape of the returned samples.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples, of shape `shape`.

  """
  # Generate samples using:
  # sqrt(X^2 + Y^2 + Z^2), X,Y,Z ~N(0,1)
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("maxwell", dtype, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _maxwell(key, shape, dtype)


@partial(jit, static_argnums=(1, 2), inline=True)
def _maxwell(key, shape, dtype) -> Array:
  shape = shape + (3,)
  norm_rvs = normal(key=key, shape=shape, dtype=dtype)
  return jnp.linalg.norm(norm_rvs, axis=-1)


def double_sided_maxwell(key: KeyArray,
                         loc: RealArray,
                         scale: RealArray,
                         shape: Shape = (),
                         dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample from a double sided Maxwell distribution.

  The values are distributed according to the probability density function:

  .. math::
     f(x;\mu,\sigma) \propto z^2 e^{-z^2 / 2}

  where :math:`z = (x - \mu) / \sigma`, with the center :math:`\mu` specified by
  ``loc`` and the scale :math:`\sigma` specified by ``scale``.

  Args:
    key: a PRNG key.
    loc: The location parameter of the distribution.
    scale: The scale parameter of the distribution.
    shape: The shape added to the parameters loc and scale broadcastable shape.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples.

  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("double_sided_maxwell", dtype, loc, scale, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _double_sided_maxwell(key, loc, scale, shape, dtype)


@partial(jit, static_argnums=(3, 4), inline=True)
def _double_sided_maxwell(key, loc, scale, shape, dtype) -> Array:
  params_shapes = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
  if not shape:
    shape = params_shapes

  shape = shape + params_shapes
  maxwell_key, rademacher_key = _split(key)
  maxwell_rvs = maxwell(maxwell_key, shape=shape, dtype=dtype)
  # Generate random signs for the symmetric variates.
  random_sign = rademacher(rademacher_key, shape=shape, dtype=dtype)
  assert random_sign.shape == maxwell_rvs.shape

  return random_sign * maxwell_rvs * scale + loc


def weibull_min(key: KeyArray,
                scale: RealArray,
                concentration: RealArray,
                shape: Shape = (),
                dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample from a Weibull distribution.

  The values are distributed according to the probability density function:

  .. math::
     f(x;\sigma,c) \propto x^{c - 1} \exp(-(x / \sigma)^c)

  on the domain :math:`0 < x < \infty`, where :math:`c > 0` is the concentration
  parameter, and :math:`\sigma > 0` is the scale parameter.

  Args:
    key: a PRNG key.
    scale: The scale parameter of the distribution.
    concentration: The concentration parameter of the distribution.
    shape: The shape added to the parameters loc and scale broadcastable shape.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples.

  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("weibull_min", dtype, scale, concentration, require_float=True)
  shape = core.canonicalize_shape(shape)
  return _weibull_min(key, scale, concentration, shape, dtype)


@partial(jit, static_argnums=(3, 4), inline=True)
def _weibull_min(key, scale, concentration, shape, dtype) -> Array:
  random_uniform = uniform(
      key=key, shape=shape, minval=0, maxval=1, dtype=dtype)

  # Inverse weibull CDF.
  return jnp.power(-jnp.log1p(-random_uniform), 1.0/concentration) * scale


# TODO(frostig): remove these aliases

threefry2x32_p = prng.threefry2x32_p

def threefry_2x32(keypair, count):
  warnings.warn('jax.random.threefry_2x32 has moved to jax.prng.threefry_2x32 '
                'and will be removed from `random` module.', FutureWarning)
  return prng.threefry_2x32(keypair, count)

def orthogonal(
  key: KeyArray,
  n: int,
  shape: Shape = (),
  dtype: Optional[DTypeLike] = None
) -> Array:
  """Sample uniformly from the orthogonal group O(n).

  If the dtype is complex, sample uniformly from the unitary group U(n).

  Args:
    key: a PRNG key used as the random key.
    n: an integer indicating the resulting dimension.
    shape: optional, the batch dimensions of the result. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array of shape `(*shape, n, n)` and specified dtype.
  """
  key, _ = _check_prng_key(key)
  _check_shape("orthogonal", shape)
  n = core.concrete_or_error(index, n, "The error occurred in jax.random.orthogonal()")
  dtype = _check_dtype("orthogonal", dtype, require_inexact=True)
  z = normal(key, (*shape, n, n), dtype)
  q, r = jnp.linalg.qr(z)
  d = jnp.diagonal(r, 0, -2, -1)
  return lax.mul(q, lax.expand_dims(lax.div(d, abs(d).astype(d.dtype)), [-2]))

def generalized_normal(
  key: KeyArray,
  p: float,
  shape: Shape = (),
  dtype: Optional[DTypeLikeFloat] = None
) -> Array:
  r"""Sample from the generalized normal distribution.

  The values are returned according to the probability density function:

  .. math::
     f(x;p) \propto e^{-|x|^p}

  on the domain :math:`-\infty < x < \infty`, where :math:`p > 0` is the
  shape parameter.

  Args:
    key: a PRNG key used as the random key.
    p: a float representing the shape parameter.
    shape: optional, the batch dimensions of the result. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key(key)
  _check_shape("generalized_normal", shape)
  dtype = _check_dtype("generalized_normal", dtype, p, require_float=True)
  keys = split(key)
  g = gamma(keys[0], 1/p, shape, dtype)
  r = rademacher(keys[1], shape, dtype)
  return r * g ** (1 / p)

def ball(
  key: KeyArray,
  d: int,
  p: float = 2,
  shape: Shape = (),
  dtype: Optional[DTypeLikeFloat] = None
):
  """Sample uniformly from the unit Lp ball.

  Reference: https://arxiv.org/abs/math/0503650.

  Args:
    key: a PRNG key used as the random key.
    d: a nonnegative int representing the dimensionality of the ball.
    p: a float representing the p parameter of the Lp norm.
    shape: optional, the batch dimensions of the result. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array of shape `(*shape, d)` and specified dtype.
  """
  key, _ = _check_prng_key(key)
  _check_shape("ball", shape)
  dtype = _check_dtype("ball", dtype, p, require_float=True)
  d = core.concrete_or_error(index, d, "The error occurred in jax.random.ball()")
  k1, k2 = split(key)
  g = generalized_normal(k1, p, (*shape, d), dtype)
  e = exponential(k2, shape, dtype)
  return g / (((jnp.abs(g) ** p).sum(-1) + e) ** (1 / p))[..., None]


def rayleigh(key: KeyArray,
             scale: RealArray,
             shape: Optional[Shape] = None,
             dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Rayleigh random values with given shape and float dtype.

  The values are returned according to the probability density function:

  .. math::
     f(x;\sigma) \propto xe^{-x^2/(2\sigma^2)}

  on the domain :math:`-\infty < x < \infty`, and where `\sigma > 0` is the scale
  parameter of the distribution.

  Args:
    key: a PRNG key used as the random key.
    scale: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``scale``. The default (None)
      produces a result shape equal to ``scale.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``scale.shape``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("rayleigh", dtype, scale, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _rayleigh(key, scale, shape, dtype)

@partial(jit, static_argnums=(2, 3), inline=True)
def _rayleigh(key, scale, shape, dtype) -> Array:
  if shape is None:
    shape = np.shape(scale)
  else:
    _check_shape("rayleigh", shape, np.shape(scale))
  u = uniform(key, shape, dtype)
  scale = scale.astype(dtype)
  scale = jnp.broadcast_to(scale, shape)
  log_u = lax.log(u)
  n_two = _lax_const(scale, -2)
  sqrt_u = lax.sqrt(lax.mul(log_u, n_two))
  ray = lax.mul(scale, sqrt_u)
  return ray

def wald(key: KeyArray,
         mean: RealArray,
         shape: Optional[Shape] = None,
         dtype: Optional[DTypeLikeFloat] = None) -> Array:
  r"""Sample Wald random values with given shape and float dtype.

  The values are returned according to the probability density function:

  .. math::
     f(x;\mu) = \frac{1}{\sqrt{2\pi x^3}} \exp\left(-\frac{(x - \mu)^2}{2\mu^2 x}\right)

  on the domain :math:`-\infty < x < \infty`, and where :math:`\mu > 0` is the location
  parameter of the distribution.


  Args:
    key: a PRNG key used as the random key.
    mean: a float or array of floats broadcast-compatible with ``shape``
      representing the mean parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``mean``. The default
      (None) produces a result shape equal to ``np.shape(mean)``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``mean.shape``.
  """
  key, _ = _check_prng_key(key)
  dtype = _check_dtype("wald", dtype, mean, require_float=True)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _wald(key, mean, shape, dtype)

@partial(jit, static_argnums=(2, 3), inline=True)
def _wald(key, mean, shape, dtype) -> Array:
  if shape is None:
    shape =  np.shape(mean)
  else:
    _check_shape("wald", shape, np.shape(mean))
  k1, k2 = _split(key, 2)
  mean = mean.astype(dtype)
  mean = jnp.broadcast_to(mean, shape)
  v = normal(k1, shape, dtype)
  z = uniform(k2, shape, dtype)
  y = lax.integer_pow(v, 2)
  y_sq = lax.integer_pow(y, 2)
  mean_sq = lax.integer_pow(mean, 2)
  sqrt_term = lax.sqrt(4 * mean * y + mean_sq * y_sq)
  x = mean + mean_sq * y / 2 - mean / 2 * sqrt_term
  w = lax.select(lax.le(z,  mean / (mean + x)), x, mean_sq / x)
  return w

def geometric(key: KeyArray,
              p: RealArray,
              shape: Optional[Shape] = None,
              dtype: DTypeLikeInt = dtypes.int_) -> Array:
  r"""Sample Geometric random values with given shape and float dtype.

  The values are returned according to the probability mass function:

  .. math::
      f(k;p) = p(1-p)^{k-1}

  on the domain :math:`0 < p < 1`.

  Args:
    key: a PRNG key used as the random key.
    p: a float or array of floats broadcast-compatible with ``shape``
      representing the the probability of success of an individual trial.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``p``. The default
      (None) produces a result shape equal to ``np.shape(p)``.
    dtype: optional, a int dtype for the returned values (default int64 if
      jax_enable_x64 is true, otherwise int32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``p.shape``.
  """
  key, _ = _check_prng_key(key)
  if not dtypes.issubdtype(dtype, np.integer):
    raise ValueError("dtype argument to `geometric` must be an int "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _geometric(key, p, shape, dtype)

@partial(jit, static_argnums=(2, 3), inline=True)
def _geometric(key, p, shape, dtype) -> Array:
  if shape is None:
    shape =  np.shape(p)
  else:
    _check_shape("geometric", shape, np.shape(p))
  check_arraylike("geometric", p)
  p, = promote_dtypes_inexact(p)
  u = uniform(key, shape, p.dtype)
  log_u = lax.log(u)
  log_one_minus_p = lax.log1p(-p)
  log_one_minus_p = jnp.broadcast_to(log_one_minus_p, shape)
  g = lax.floor(lax.div(log_u, log_one_minus_p)) + 1
  return g.astype(dtype)
