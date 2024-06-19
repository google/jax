# Copyright 2024 The JAX Authors.
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
from typing import Any, Callable

import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jax_api_random
from jax._src import typing
from jax._src.pallas.mosaic.primitives import prng_seed
from jax._src.pallas.mosaic.primitives import prng_random_bits
from jax._src import prng as jax_prng


Shape = jax_prng.Shape
SampleFnType = Any
KeylessSampleFnType = Callable[..., jax.Array]

set_seed = prng_seed

FOLD_IN_ROUNDS = 128
SUPPORTED_CONVERSION_KEYS = ["rbg", "unsafe_rbg", "pallas_tpu"]

def to_pallas_key(key: jax_prng.PRNGKeyArray) -> jax_prng.PRNGKeyArray:
  """Helper function for converting non-Pallas PRNG keys into Pallas keys."""

  # Only allow conversion from RBG -> Pallas keys.
  # There is no technical reason why we cannot support Threefry here, but
  # this reduces the chance of unintended behavior where the pallas PRNG
  # produces different random bits than Threefry. RBG has fewer guarantees
  # so users of RBG should be more aware of the consequences.
  if key._impl.name not in SUPPORTED_CONVERSION_KEYS:
    raise ValueError(f"Unsupported key type: {key._impl.name}"
                     f"Supported keys are: {SUPPORTED_CONVERSION_KEYS}")

  key_data = jax_api_random.key_data(key)
  pallas_key_size = np.prod(tpu_key_impl.key_shape)
  if key_data.size < pallas_key_size:
    raise ValueError(f"Key data must be at least {pallas_key_size} bytes.")
  pallas_key_data = jnp.ravel(key_data)[:pallas_key_size]
  pallas_key_data = jnp.reshape(pallas_key_data, tpu_key_impl.key_shape)
  return jax_api_random.wrap_key_data(pallas_key_data, impl="pallas_tpu")

def _seed_func(seed: jnp.int32):
  seed_data = jnp.zeros(tpu_key_impl.key_shape, dtype=jnp.int32)
  return (seed_data + seed).astype(jnp.uint32)

def _random_bits(key: typing.Array, bit_width: int, shape: Shape):
  if bit_width != 32:
    raise ValueError("Bit width must be 32")
  prng_seed(key)
  return prng_random_bits(shape)

def _fold_in(key: jax_prng.PRNGKeyArray, data: typing.Array):
  # Roughly, we compute the new key as follows:
  # new_key = random_bits(data)[..., 127] ^ random_bits(old_key)[..., 127]
  # Because the TPU generates random numbers in (8, 128) blocks at once, we
  # can generate that many values without additional cost which will reduce
  # correlation between the old and new keys.
  key_shape = tpu_key_impl.key_shape

  prng_seed(data)
  data_bits = prng_random_bits(
      key_shape + (FOLD_IN_ROUNDS,)).astype(jnp.uint32)
  prng_seed(key)
  key_bits = prng_random_bits(
      key_shape + (FOLD_IN_ROUNDS,)).astype(jnp.uint32)

  mixed = key_bits[..., FOLD_IN_ROUNDS-1] ^ data_bits[..., FOLD_IN_ROUNDS-1]
  assert mixed.shape == key_shape
  impl: jax_prng.PRNGSpec = jax.random.key_impl(key)  # type: ignore
  return jax.random.wrap_key_data(mixed, impl=impl)

def _split(key: typing.Array, shape: Shape):
  del key, shape
  raise NotImplementedError()

tpu_key_impl = jax_prng.PRNGImpl(
   key_shape=(1,),
   seed=_seed_func,
   split=_split,
   random_bits=_random_bits,
   fold_in=_fold_in,
   name="pallas_tpu",
   tag="pl"
)
jax_prng.register_prng(tpu_key_impl)

# Implementation of the stateful Pallas PRNG API.
# Users should set the seed using the `set_seed` function,
# and call the appropriate stateful sampling functions.
# The actual key impl should never be used. The impl
# serves as internal boilerplate code because JAX's existing
# random functions expect a key as an argument, and
# the keys are only generated as part of unused arguments.

def _pl_stateful_seed_func(seed: jnp.int32):
  del seed
  # Unused. Return the correct shape and dtype.
  return jnp.empty((), dtype=jnp.int32)

def _pl_stateful_random_bits(key: typing.Array, bit_width: int, shape: Shape):
  del key
  assert bit_width == 32, "Bit width must be 32"
  return prng_random_bits(shape)

def _pl_stateful_fold_in(key: typing.Array, data: typing.Array):
  del key, data
  raise NotImplementedError()

def _pl_stateful_split(key: typing.Array, shape: Shape):
  del key, shape
  raise NotImplementedError()


tpu_internal_stateful_impl = jax_prng.PRNGImpl(
   key_shape=(),
   seed=_pl_stateful_seed_func,
   split=_pl_stateful_split,
   random_bits=_pl_stateful_random_bits,
   fold_in=_pl_stateful_fold_in,
   name="_pallas_internal_stateful",
   tag="_pl_stateful"
)
jax_prng.register_prng(tpu_internal_stateful_impl)

def _make_stateful_sampler(sampler: SampleFnType) -> KeylessSampleFnType:
  """Converts a jax.random sampling function to a stateful version.

  Args:
    sampler: A sampling function that consumes a key and returns
      random samples.

  Returns:
    A stateful sampling function with the key argument removed.
  """
  def new_sampler(*args, **kwargs):
    # Pass in a placeholder key into the sampling function.
    # The key is ignored by the stateful random_bits function, but all jax
    # sampling functions expect a key as input so we must pass one in here.
    placeholder_key = jax_api_random.key(0, impl=tpu_internal_stateful_impl)
    return sampler(placeholder_key, *args, **kwargs)
  # Remove key argument from docstring.
  doc_lines = filter(
      lambda line: "key:" not in line, sampler.__doc__.split("\n"))
  new_sampler.__doc__ = "\n".join(doc_lines)
  return new_sampler

bits = _make_stateful_sampler(jax_api_random.bits)
uniform = _make_stateful_sampler(jax_api_random.uniform)
bernoulli = _make_stateful_sampler(jax_api_random.bernoulli)
