# Copyright 2021 Google LLC
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
from typing import Callable, Iterator, NamedTuple, Sequence, Optional
import warnings

import numpy as np

from jax import lax
from jax import core
from jax import numpy as jnp
from jax import tree_util
from jax.config import config
from jax.dtypes import float0
from jax.errors import KeyReuseError
from jax.interpreters import batching
from jax.interpreters import xla
from jax._src.api import jit, vmap
from jax._src.lib import xla_client
from jax._src.lib import cuda_prng
from jax._src.numpy.lax_numpy import (
    _canonicalize_tuple_index, _eliminate_deprecated_list_indexing,
    _expand_bool_indices, _register_stackable)
import jax._src.pretty_printer as pp
from jax._src import source_info_util
from jax._src.util import prod


UINT_DTYPES = {
    8: jnp.uint8, 16: jnp.uint16, 32: jnp.uint32, 64: jnp.uint64}  # type: ignore[has-type]

# -- PRNG implementation interface --

class PRNGImpl(NamedTuple):
  """Specifies PRNG key shape and operations.

  A PRNG implementation is determined by a key type ``K`` and a
  collection of functions that operate on such keys. The key type
  ``K`` is an array type with element type uint32 and shape specified
  by ``key_shape``. The type signature of each operations is::

    seed :: int[] -> K
    fold_in :: K -> int[] -> K
    split[n] :: K -> K[n]
    random_bits[shape, bit_width] :: K -> uint<bit_width>[shape]

  A PRNG implementation is adapted to an array-like object of keys
  ``K`` by the ``PRNGKeyArray`` class, which should be created via the
  ``seed_with_impl`` function.
  """
  key_shape: core.Shape
  seed: Callable
  split: Callable
  random_bits: Callable
  fold_in: Callable

  def pprint(self):
    return (pp.text(f"{self.__class__.__name__}:") +
            pp.nest(2, pp.group(pp.brk() + pp.join(pp.brk(), [
              pp.text(f"{k} = {v}") for k, v in self._asdict().items()
            ]))))


# -- PRNG key arrays --

def _is_prng_key_data(impl, keys: jnp.ndarray) -> bool:
  ndim = len(impl.key_shape)
  try:
    return (keys.ndim >= 1 and
            keys.shape[-ndim:] == impl.key_shape and
            (keys.dtype == np.uint32 or keys.dtype == float0))
  except AttributeError:
    return False

@tree_util.register_pytree_node_class
class PRNGKeyArray:
  """An array whose elements are PRNG keys.

  This class lifts the definition of a PRNG, provided in the form of a
  ``PRNGImpl``, into an array-like pytree class. Instances of this
  class behave like an array whose base elements are keys, hiding the
  fact that keys are typically arrays (of ``uint32`` dtype) themselves.

  PRNGKeyArrays are also restricted relative to JAX arrays in that
  they do not expose arithmetic operations. They instead expose
  wrapper methods around the PRNG implementation functions (``split``,
  ``random_bits``, ``fold_in``).
  """

  impl: PRNGImpl
  _keys: jnp.ndarray
  _consumed: bool
  _consumed_source_info: Optional[source_info_util.SourceInfo]

  def __init__(self, impl, key_data: jnp.ndarray):
    # key_data might be a placeholder python `object` or `bool`
    # instead of a jnp.ndarray due to tree_unflatten
    if (type(key_data) not in [object, bool] and
        not _is_prng_key_data(impl, key_data)):
      raise TypeError(
          f'Invalid PRNG key data {key_data} for PRNG implementation {impl}')
    self.impl = impl
    self._keys = key_data
    self._consumed = False

  def tree_flatten(self):
    return (self._keys,), self.impl

  def unsafe_raw_array(self):
    """Access the raw numerical array that carries underlying key data.

    Returns:
      A uint32 JAX array whose leading dimensions are ``self.shape``.
    """
    return self._keys

  @classmethod
  def tree_unflatten(cls, impl, keys):
    keys, = keys
    return cls(impl, keys)

  @property
  def dtype(self):
    # TODO(frostig): remove after deprecation window
    if config.jax_enable_custom_prng:
      raise AttributeError("'PRNGKeyArray' has no attribute 'dtype'")
    else:
      warnings.warn(
          'deprecated `dtype` attribute of PRNG key arrays', FutureWarning)
      return np.uint32

  @property
  def shape(self):
    # TODO(frostig): simplify once we always enable_custom_prng
    if config.jax_enable_custom_prng:
      return self._shape
    else:
      warnings.warn(
          'deprecated `shape` attribute of PRNG key arrays. In a future version '
          'of JAX this attribute will be removed or its value may change.',
          FutureWarning)
      return self._keys.shape

  @property
  def _shape(self):
    base_ndim = len(self.impl.key_shape)
    return self._keys.shape[:-base_ndim]

  def _is_scalar(self):
    base_ndim = len(self.impl.key_shape)
    return self._keys.ndim == base_ndim

  def __len__(self):
    if self._is_scalar():
      raise TypeError('len() of unsized object')
    return len(self._keys)

  def __iter__(self) -> Iterator['PRNGKeyArray']:
    if self._is_scalar():
      raise TypeError('iteration over a 0-d single PRNG key')
    return (PRNGKeyArray(self.impl, k) for k in iter(self._keys))

  def __getitem__(self, idx) -> 'PRNGKeyArray':
    base_ndim = len(self.impl.key_shape)
    ndim = self._keys.ndim - base_ndim
    indexable_shape = self.impl.key_shape[:ndim]
    idx = _eliminate_deprecated_list_indexing(idx)
    idx = _expand_bool_indices(idx, indexable_shape)
    idx = _canonicalize_tuple_index(ndim, idx, array_name='PRNGKeyArray')
    return PRNGKeyArray(self.impl, self._keys[idx])

  def consume(self):
    if self._consumed and config.jax_debug_prng_key_reuse:
      summary = source_info_util.summarize(self._consumed_source_info)
      raise KeyReuseError('Re-used a key, the key was previously used'
                          f' at {summary}.')
    self._consumed = True
    self._consumed_source_info = source_info_util.current()
    return self

  def _fold_in(self, data: int) -> 'PRNGKeyArray':
    return PRNGKeyArray(self.impl, self.impl.fold_in(self._keys, data))

  def _random_bits(self, bit_width, shape) -> jnp.ndarray:
    return self.impl.random_bits(self._keys, bit_width, shape)

  def _split(self, num: int) -> 'PRNGKeyArray':
    self.consume()
    return PRNGKeyArray(self.impl, self.impl.split(self._keys, num))

  def reshape(self, newshape, order=None):
    reshaped_keys = jnp.reshape(self._keys, (*newshape, -1), order=order)
    return PRNGKeyArray(self.impl, reshaped_keys)

  def concatenate(self, key_arrs, axis):
    axis = axis % len(self.shape)
    arrs = [self._keys, *[k._keys for k in key_arrs]]
    return PRNGKeyArray(self.impl, jnp.stack(arrs, axis))

  def broadcast_to(self, shape):
    new_shape = tuple(shape)+(self._keys.shape[-1],)
    return PRNGKeyArray(self.impl, jnp.broadcast_to(self._keys, new_shape))

  def __repr__(self):
    arr_shape = self._shape
    pp_keys = pp.text('shape = ') + pp.text(str(arr_shape))
    pp_impl = pp.text('impl = ') + self.impl.pprint()
    return str(pp.group(
      pp.text('PRNGKeyArray:') +
      pp.nest(2, pp.brk() + pp_keys + pp.brk() + pp_impl)))


def seed_with_impl(impl: PRNGImpl, seed: int) -> PRNGKeyArray:
  return PRNGKeyArray(impl, impl.seed(seed))

_register_stackable(PRNGKeyArray)

# -- threefry2x32 PRNG implementation --


def _is_threefry_prng_key(key: jnp.ndarray) -> bool:
  try:
    return key.shape == (2,) and key.dtype == np.uint32
  except AttributeError:
    return False


def threefry_seed(seed: int) -> jnp.ndarray:
  """Create a single raw threefry PRNG key given an integer seed.

  Args:
    seed: a 64- or 32-bit integer used as the value of the key.

  Returns:
    The PRNG key contents, modeled as an array of shape (2,) and dtype
    uint32. The key is constructed from a 64-bit seed by effectively
    bit-casting to a pair of uint32 values (or from a 32-bit seed by
    first padding out with zeros).
  """
  # Avoid overflowerror in X32 mode by first converting ints to int64.
  # This breaks JIT invariance for large ints, but supports the common
  # use-case of instantiating with Python hashes in X32 mode.
  if isinstance(seed, int):
    seed_arr = jnp.asarray(np.int64(seed))
  else:
    seed_arr = jnp.asarray(seed)
  if seed_arr.shape:
    raise TypeError(f"PRNG key seed must be a scalar; got {seed!r}.")
  if not np.issubdtype(seed_arr.dtype, np.integer):
    raise TypeError(f"PRNG key seed must be an integer; got {seed!r}")

  convert = lambda k: lax.reshape(lax.convert_element_type(k, np.uint32), [1])
  k1 = convert(lax.shift_right_logical(seed_arr, lax._const(seed_arr, 32)))
  k2 = convert(jnp.bitwise_and(seed_arr, np.uint32(0xFFFFFFFF)))
  return lax.concatenate([k1, k2], 0)


def _make_rotate_left(dtype):
  if not jnp.issubdtype(dtype, np.integer):
    raise TypeError("_rotate_left only accepts integer dtypes.")
  nbits = np.array(jnp.iinfo(dtype).bits, dtype)

  def _rotate_left(x, d):
    if lax.dtype(d) != dtype:
      d = lax.convert_element_type(d, dtype)
    if lax.dtype(x) != dtype:
      x = lax.convert_element_type(x, dtype)
    return lax.shift_left(x, d) | lax.shift_right_logical(x, nbits - d)
  return _rotate_left


def _bit_stats(bits):
  """This is a debugging function to compute the statistics of bit fields."""
  return np.array([list(map(int, np.binary_repr(x, 64))) for x in bits]).mean(0)


### hash function and split

def _threefry2x32_abstract_eval(*args):
  if any(a.dtype != jnp.uint32 for a in args):
    raise TypeError("Arguments to threefry2x32 must have uint32 type, got {}"
                    .format(args))
  if all(isinstance(arg, core.ShapedArray) for arg in args):
    shape = lax._broadcasting_shape_rule(*args)
    named_shape = core.join_named_shapes(*(a.named_shape for a in args))
    aval = core.ShapedArray(shape, jnp.dtype(jnp.uint32), named_shape=named_shape)
  else:
    aval = core.UnshapedArray(jnp.dtype(jnp.uint32))
  return (aval,) * 2


rotate_left = _make_rotate_left(np.uint32)


def apply_round(v, rot):
  v = v[:]
  v[0] = v[0] + v[1]
  v[1] = rotate_left(v[1], rot)
  v[1] = v[0] ^ v[1]
  return v


def rotate_list(xs):
  return xs[1:] + xs[:1]


def rolled_loop_step(i, state):
  x, ks, rotations = state
  for r in rotations[0]:
    x = apply_round(x, r)
  new_x = [x[0] + ks[0], x[1] + ks[1] + jnp.asarray(i + 1, dtype=np.uint32)]
  return new_x, rotate_list(ks), rotate_list(rotations)


def _threefry2x32_lowering(key1, key2, x1, x2, use_rolled_loops=True):
  """Apply the Threefry 2x32 hash.

  Args:
    keypair: a pair of 32bit unsigned integers used for the key.
    count: an array of dtype uint32 used for the counts.

  Returns:
    An array of dtype uint32 with the same shape as `count`.
  """
  x = [x1, x2]

  rotations = [np.array([13, 15, 26, 6], dtype=np.uint32),
               np.array([17, 29, 16, 24], dtype=np.uint32)]
  ks = [key1, key2, key1 ^ key2 ^ np.uint32(0x1BD11BDA)]

  x[0] = x[0] + ks[0]
  x[1] = x[1] + ks[1]

  if use_rolled_loops:
    x, _, _ = lax.fori_loop(0, 5, rolled_loop_step, (x, rotate_list(ks), rotations))

  else:
    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[1]
    x[1] = x[1] + ks[2] + np.uint32(1)

    for r in rotations[1]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[2]
    x[1] = x[1] + ks[0] + np.uint32(2)

    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[0]
    x[1] = x[1] + ks[1] + np.uint32(3)

    for r in rotations[1]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[1]
    x[1] = x[1] + ks[2] + np.uint32(4)

    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[2]
    x[1] = x[1] + ks[0] + np.uint32(5)

  return tuple(x)


def _threefry2x32_gpu_translation_rule(ctx, avals_in, avals_out, k1, k2, x1,
                                       x2):
  aval_out, _ = avals_out
  k1_aval, k2_aval, x1_aval, x2_aval = avals_in
  rank = len(aval_out.shape)
  if 0 in aval_out.shape:
    zeros = xla_client.ops.Broadcast(
        xla_client.ops.Constant(ctx.builder, np.array(0, np.uint32)),
        aval_out.shape)
    return [zeros, zeros]
  def _broadcast(x, aval):
    return xla_client.ops.BroadcastInDim(
        x, aval_out.shape, tuple(range(rank - len(aval.shape), rank)))
  return xla.xla_destructure(
      ctx.builder,
      cuda_prng.threefry2x32(
          ctx.builder, (_broadcast(k1, k1_aval), _broadcast(k2, k2_aval)),
          (_broadcast(x1, x1_aval), _broadcast(x2, x2_aval))))


threefry2x32_p = core.Primitive("threefry2x32")
threefry2x32_p.multiple_results = True
threefry2x32_p.def_impl(partial(xla.apply_primitive, threefry2x32_p))
threefry2x32_p.def_abstract_eval(_threefry2x32_abstract_eval)
batching.defbroadcasting(threefry2x32_p)
xla.register_translation(threefry2x32_p, xla.lower_fun(
    partial(_threefry2x32_lowering, use_rolled_loops=False),
    multiple_results=True, new_style=True))
xla.register_translation(threefry2x32_p, xla.lower_fun(
    partial(_threefry2x32_lowering, use_rolled_loops=True),
    multiple_results=True, new_style=True), platform='cpu')
if cuda_prng:
  xla.register_translation(threefry2x32_p, _threefry2x32_gpu_translation_rule,
                           platform='gpu')


@partial(jit, inline=True)
def threefry_2x32(keypair, count):
  """Apply the Threefry 2x32 hash.

  Args:
    keypair: a pair of 32bit unsigned integers used for the key.
    count: an array of dtype uint32 used for the counts.

  Returns:
    An array of dtype uint32 with the same shape as `count`.
  """
  key1, key2 = keypair
  if not lax.dtype(key1) == lax.dtype(key2) == lax.dtype(count) == np.uint32:
    msg = "threefry_2x32 requires uint32 arguments, got {}"
    raise TypeError(msg.format([lax.dtype(x) for x in [key1, key2, count]]))

  try:
    odd_size = count.size % 2
  except core.InconclusiveDimensionOperation as e:
    msg = ("jax.random functions have limited support for shape polymorphism. "
           "In particular, the product of the known dimensions must be even.")
    raise core.InconclusiveDimensionOperation(msg) from e

  if odd_size:
    x = list(jnp.split(jnp.concatenate([count.ravel(), np.uint32([0])]), 2))
  else:
    x = list(jnp.split(count.ravel(), 2))

  x = threefry2x32_p.bind(key1, key2, x[0], x[1])
  out = jnp.concatenate(x)
  assert out.dtype == np.uint32
  return lax.reshape(out[:-1] if odd_size else out, count.shape)


def threefry_split(key: jnp.ndarray, num: int) -> jnp.ndarray:
  return _threefry_split(key, int(num))  # type: ignore

@partial(jit, static_argnums=(1,), inline=True)
def _threefry_split(key, num) -> jnp.ndarray:
  counts = lax.iota(np.uint32, num * 2)
  return lax.reshape(threefry_2x32(key, counts), (num, 2))


def threefry_fold_in(key: jnp.ndarray, data: int) -> jnp.ndarray:
  return _threefry_fold_in(key, jnp.uint32(data))

@partial(jit, inline=True)
def _threefry_fold_in(key, data):
  return threefry_2x32(key, threefry_seed(data))


@partial(jit, static_argnums=(1, 2), inline=True)
def threefry_random_bits(key: jnp.ndarray, bit_width, shape):
  """Sample uniform random bits of given width and shape using PRNG key."""
  if not _is_threefry_prng_key(key):
    raise TypeError("threefry_random_bits got invalid prng key.")
  if bit_width not in (8, 16, 32, 64):
    raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
  shape = core.as_named_shape(shape)
  for name, size in shape.named_items:
    real_size = lax.psum(1, name)
    if real_size != size:
      raise ValueError(f"The shape of axis {name} was specified as {size}, "
                       f"but it really is {real_size}")
    axis_index = lax.axis_index(name)
    key = threefry_fold_in(key, axis_index)
  size = prod(shape.positional)
  # Compute ceil(bit_width * size / 32) in a way that is friendly to shape
  # polymorphism
  max_count, r = divmod(bit_width * size, 32)
  if r > 0:
    max_count += 1

  if core.is_constant_dim(max_count):
    nblocks, rem = divmod(max_count, jnp.iinfo(np.uint32).max)
  else:
    nblocks, rem = 0, max_count

  if not nblocks:
    bits = threefry_2x32(key, lax.iota(np.uint32, rem))
  else:
    keys = threefry_split(key, nblocks + 1)
    subkeys, last_key = keys[:-1], keys[-1]
    blocks = vmap(threefry_2x32, in_axes=(0, None))(subkeys, lax.iota(np.uint32, jnp.iinfo(np.uint32).max))
    last = threefry_2x32(last_key, lax.iota(np.uint32, rem))
    bits = lax.concatenate([blocks.ravel(), last], 0)

  dtype = UINT_DTYPES[bit_width]
  if bit_width == 64:
    bits = [lax.convert_element_type(x, dtype) for x in jnp.split(bits, 2)]
    bits = lax.shift_left(bits[0], dtype(32)) | bits[1]
  elif bit_width in [8, 16]:
    # this is essentially bits.view(dtype)[:size]
    bits = lax.bitwise_and(
      np.uint32(np.iinfo(dtype).max),
      lax.shift_right_logical(
        lax.broadcast(bits, (1,)),
        lax.mul(
          np.uint32(bit_width),
          lax.broadcasted_iota(np.uint32, (32 // bit_width, 1), 0)
        )
      )
    )
    bits = lax.reshape(bits, (np.uint32(max_count * 32 // bit_width),), (1, 0))
    bits = lax.convert_element_type(bits, dtype)[:size]
  return lax.reshape(bits, shape)


threefry_prng_impl = PRNGImpl(
    key_shape=(2,),
    seed=threefry_seed,
    split=threefry_split,
    random_bits=threefry_random_bits,
    fold_in=threefry_fold_in)


# -- RngBitGenerator PRNG implementation --

# This code is experimental!
# https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator
# Notice that the RngBitGenerator operations are not guaranteed to be
# stable/deterministic across backends or compiler versions. Correspondingly, we
# reserve the right to change any of these implementations at any time!

def _rbg_seed(seed: int) -> jnp.ndarray:
  halfkey = threefry_seed(seed)
  return jnp.concatenate([halfkey, halfkey])

def _rbg_split(key: jnp.ndarray, num: int) -> jnp.ndarray:
  return vmap(_threefry_split, (0, None), 1)(key.reshape(2, 2), num).reshape(num, 4)

def _rbg_fold_in(key: jnp.ndarray, data: int) -> jnp.ndarray:
  return vmap(_threefry_fold_in, (0, None), 0)(key.reshape(2, 2), data).reshape(4)

def _rbg_random_bits(key: jnp.ndarray, bit_width: int, shape: Sequence[int]
                     ) -> jnp.ndarray:
  if not key.shape == (4,) and key.dtype == jnp.dtype('uint32'):
    raise TypeError("_rbg_random_bits got invalid prng key.")
  if bit_width not in (8, 16, 32, 64):
    raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
  _, bits = lax.rng_bit_generator(key, shape, dtype=UINT_DTYPES[bit_width])
  return bits

rbg_prng_impl = PRNGImpl(
    key_shape=(4,),
    seed=_rbg_seed,
    split=_rbg_split,
    random_bits=_rbg_random_bits,
    fold_in=_rbg_fold_in)

def _unsafe_rbg_split(key: jnp.ndarray, num: int) -> jnp.ndarray:
  # treat 10 iterations of random bits as a 'hash function'
  _, keys = lax.rng_bit_generator(key, (10 * num, 4), dtype='uint32')
  return keys[::10]

def _unsafe_rbg_fold_in(key: jnp.ndarray, data: int) -> jnp.ndarray:
  _, random_bits = lax.rng_bit_generator(_rbg_seed(data), (10, 4), dtype='uint32')
  return key ^ random_bits[-1]

unsafe_rbg_prng_impl = PRNGImpl(
    key_shape=(4,),
    seed=_rbg_seed,
    split=_unsafe_rbg_split,
    random_bits=_rbg_random_bits,
    fold_in=_unsafe_rbg_fold_in)
