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
from unittest import SkipTest, skipIf
from typing import Any, Tuple, NamedTuple, Optional
import zlib

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats

import jax
from jax import grad
from jax import lax
from jax import numpy as jnp
from jax import prng
from jax import random
from jax import tree_util
from jax._src import core
from jax._src import dtypes
from jax._src import test_util as jtu
from jax import vmap
from jax.interpreters import xla

from jax._src import random as jax_random
from jax._src import prng as prng_internal

from jax import config
config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
uint_dtypes = jtu.dtypes.all_unsigned


def _prng_key_as_array(key):
  # TODO(frostig): remove once we upgrade to always enable_custom_prng
  return key.unsafe_raw_array() if config.jax_enable_custom_prng else key

def _maybe_unwrap(key):
  # TODO(frostig): remove once we upgrade to always enable_custom_prng
  unwrap = prng_internal.random_unwrap
  return unwrap(key) if config.jax_enable_custom_prng else key


PRNG_IMPLS = [('threefry2x32', prng.threefry_prng_impl),
              ('rbg', prng.rbg_prng_impl),
              ('unsafe_rbg', prng.unsafe_rbg_prng_impl)]


class RandomValuesCase(NamedTuple):
  name: str
  prng_impl: str
  shape: Tuple[int, ...]
  dtype: Any
  params: dict
  expected: np.ndarray
  skip_on_x64: bool = False
  atol: Optional[float] = None
  rtol: Optional[float] = None

  def _testname(self):
    if self.dtype is None:
      shape_dtype = str(self.shape)
    else:
      shape_dtype = jtu.format_shape_dtype_string(self.shape, self.dtype)
    name = f"_{self.name}_{self.prng_impl}_{shape_dtype}"
    if self.params:
      fmt = lambda x: str(x).replace(' ', '').replace('\n', '')
      name += "_" + "_".join(f"{k}={fmt(v)}" for k, v in self.params.items())
    return name

  def _seed(self):
    # Generate a deterministic unique 32-bit seed given the name and prng impl
    return zlib.adler32((self.name + self.prng_impl).encode())


_RANDOM_VALUES_CASES = [
  # TODO(jakevdp) add coverage for other distributions.
  RandomValuesCase("bernoulli", "threefry2x32", (5,), None, {'p': 0.5},
    np.array([False, True, True, True, False]), skip_on_x64=True),
  RandomValuesCase("bernoulli", "rbg", (5,), None, {'p': 0.5},
    np.array([True, True, True, True, True]), skip_on_x64=True),
  RandomValuesCase("beta", "threefry2x32", (5,), np.float32, {'a': 0.8, 'b': 0.9},
    np.array([0.533685, 0.843179, 0.063495, 0.573444, 0.459514], dtype='float32')),
  RandomValuesCase("beta", "rbg", (5,), np.float32, {'a': 0.8, 'b': 0.9},
    np.array([0.841308, 0.669989, 0.731763, 0.985127, 0.022745], dtype='float32')),
  RandomValuesCase("cauchy", "threefry2x32", (5,), np.float32, {},
    np.array([ -0.088416, -10.169713, 3.49677, -1.18056, 0.34556], dtype='float32'), rtol=1E-5),
  RandomValuesCase("cauchy", "rbg", (5,), np.float32, {},
    np.array([0.008389, 0.108793, -0.031826, -0.01876, 0.963218], dtype='float32')),
  RandomValuesCase("dirichlet", "threefry2x32", (2,), np.float32, {'alpha': np.array([0.5, 0.6, 0.7], dtype='float32')},
    np.array([[0.556287, 0.304219, 0.139494], [0.15221 , 0.632251, 0.21554]], dtype='float32')),
  RandomValuesCase("dirichlet", "rbg", (2,), np.float32, {'alpha': np.array([0.5, 0.6, 0.7], dtype='float32')},
    np.array([[0.024769, 0.002189, 0.973041], [0.326, 0.00244, 0.67156]], dtype='float32')),
  RandomValuesCase("double_sided_maxwell", "threefry2x32", (5,), np.float32, {"loc": 1, "scale": 2},
    np.array([-2.408914, -3.370437, 3.235352, -0.907734, -1.708732], dtype='float32'), skip_on_x64=True),
  RandomValuesCase("double_sided_maxwell", "rbg", (5,), np.float32, {"loc": 1, "scale": 2},
    np.array([4.957495, 3.003086, 5.33935, 2.942878, -1.203524], dtype='float32'), skip_on_x64=True),
  RandomValuesCase("exponential", "threefry2x32", (5,), np.float32, {},
    np.array([0.526067, 0.043046, 0.039932, 0.46427 , 0.123886], dtype='float32')),
  RandomValuesCase("exponential", "rbg", (5,), np.float32, {},
    np.array([0.231303, 0.684814, 0.017181, 0.089552, 0.345087], dtype='float32')),
  RandomValuesCase("gamma", "threefry2x32", (5,), np.float32, {'a': 0.8},
    np.array([0.332641, 0.10187 , 1.816109, 0.023457, 0.487853], dtype='float32')),
  RandomValuesCase("gamma", "rbg", (5,), np.float32, {'a': 0.8},
    np.array([0.235293, 0.446747, 0.146372, 0.79252 , 0.294762], dtype='float32')),
  RandomValuesCase("gumbel", "threefry2x32", (5,), np.float32, {},
    np.array([2.06701, 0.911726, 0.145736, 0.185427, -0.00711], dtype='float32')),
  RandomValuesCase("gumbel", "rbg", (5,), np.float32, {},
    np.array([-0.099308, -1.123809, 1.007618, -0.077968, 3.421349], dtype='float32')),
  RandomValuesCase("laplace", "threefry2x32", (5,), np.float32, {},
    np.array([0.578939, -0.204902, 0.555733, 0.911053, -0.96456], dtype='float32')),
  RandomValuesCase("laplace", "rbg", (5,), np.float32, {},
    np.array([-2.970422, 1.925082, -0.757887, -4.444797, 0.561983], dtype='float32')),
  RandomValuesCase("loggamma", "threefry2x32", (5,), np.float32, {'a': 0.8},
    np.array([-0.899633, -0.424083, 0.631593, 0.102374, -1.07189], dtype='float32')),
  RandomValuesCase("loggamma", "rbg", (5,), np.float32, {'a': 0.8},
    np.array([-1.333825, 0.287259, -0.343074, -0.998258, -0.773598], dtype='float32')),
  RandomValuesCase("logistic", "threefry2x32", (5,), np.float32, {},
    np.array([0.19611, -1.709053, -0.274093, -0.208322, -1.675489], dtype='float32')),
  RandomValuesCase("logistic", "rbg", (5,), np.float32, {},
    np.array([-0.234923, -0.545184,  0.700992, -0.708609, -1.474884], dtype='float32')),
  RandomValuesCase("maxwell", "threefry2x32", (5,), np.float32, {},
    np.array([3.070779, 0.908479, 1.521317, 0.875551, 1.306137], dtype='float32')),
  RandomValuesCase("maxwell", "rbg", (5,), np.float32, {},
    np.array([2.048746, 0.470027, 1.053105, 1.01969, 2.710645], dtype='float32')),
  RandomValuesCase("multivariate_normal", "threefry2x32", (2,), np.float32, {"mean": np.ones((1, 3)), "cov": np.eye(3)},
    np.array([[ 1.067826,  1.215599,  0.234166], [-0.237534,  1.32591, 1.413987]], dtype='float32'), skip_on_x64=True),
  RandomValuesCase("multivariate_normal", "rbg", (2,), np.float32, {"mean": np.ones((1, 3)), "cov": np.eye(3)},
    np.array([[-0.036897, 0.770969, 0.756959], [1.755091, 2.350553, 0.627142]], dtype='float32'), skip_on_x64=True),
  RandomValuesCase("normal", "threefry2x32", (5,), np.float32, {},
    np.array([-1.173234, -1.511662, 0.070593, -0.099764, 1.052845], dtype='float32')),
  RandomValuesCase("normal", "rbg", (5,), np.float32, {},
    np.array([-0.479658, 0.565747, -1.065106, 0.997962, -1.478002], dtype='float32')),
  RandomValuesCase("pareto", "threefry2x32", (5,), np.float32, {"b": 0.5},
    np.array([2.751398, 1.281863, 87.85448, 1.254542, 2.824487], dtype='float32')),
  RandomValuesCase("pareto", "rbg", (5,), np.float32, {"b": 0.5},
    np.array([1.241914, 1.521864, 5.615384, 1911.502, 1.816702], dtype='float32')),
  RandomValuesCase("poisson", "threefry2x32", (5,), np.int32, {"lam": 5},
    np.array([7, 3, 6, 11, 6], dtype='int32')),
  # Note: poisson not implemented for rbg sampler.
  RandomValuesCase("rademacher", "threefry2x32", (5,), np.int32, {},
    np.array([-1, -1, -1, -1, 1], dtype='int32'), skip_on_x64=True),
  RandomValuesCase("rademacher", "rbg", (5,), np.int32, {},
    np.array([1, 1, 1, -1, -1], dtype='int32'), skip_on_x64=True),
  RandomValuesCase("randint", "threefry2x32", (5,), np.int32, {"minval": 0, "maxval": 10},
    np.array([0, 5, 7, 7, 5], dtype='int32')),
  RandomValuesCase("randint", "rbg", (5,), np.int32, {"minval": 0, "maxval": 10},
    np.array([7, 1, 8, 5, 8], dtype='int32')),
  RandomValuesCase("truncated_normal", "threefry2x32", (5,), np.float32, {"lower": 0, "upper": 2},
    np.array([0.582807, 1.709771, 0.159513, 0.861376, 0.36148], dtype='float32')),
  RandomValuesCase("truncated_normal", "rbg", (5,), np.float32, {"lower": 0, "upper": 2},
    np.array([0.770068, 1.516464, 0.710406, 0.762801, 1.305324], dtype='float32')),
  RandomValuesCase("uniform", "threefry2x32", (5,), np.float32, {},
    np.array([0.298671, 0.073213, 0.873356, 0.260549, 0.412797], dtype='float32')),
  RandomValuesCase("uniform", "rbg", (5,), np.float32, {},
    np.array([0.477161, 0.706508, 0.656261, 0.432547, 0.057772], dtype='float32')),
  RandomValuesCase("weibull_min", "threefry2x32", (5,), np.float32, {"scale": 1, "concentration": 1},
    np.array([1.605863, 0.841809, 0.224218, 0.4826  , 0.027901], dtype='float32')),
  RandomValuesCase("weibull_min", "rbg", (5,), np.float32, {"scale": 1, "concentration": 1},
    np.array([1.370903, 0.086532, 0.061688, 3.407599, 0.215077], dtype='float32')),
]


class PrngTest(jtu.JaxTestCase):

  def testThreefry2x32(self):
    # We test the hash by comparing to known values provided in the test code of
    # the original reference implementation of Threefry. For the values, see
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32
    def result_to_hex(result):
      return tuple(hex(x.copy()).rstrip("L") for x in result)

    expected = ("0x6b200159", "0x99ba4efe")
    result = prng.threefry_2x32(np.uint32([0, 0]), np.uint32([0, 0]))

    self.assertEqual(expected, result_to_hex(result))

    expected = ("0x1cb996fc", "0xbb002be7")
    u32_max = np.iinfo(np.uint32).max
    result = prng.threefry_2x32(np.uint32([u32_max, u32_max]), np.uint32([u32_max, u32_max]))
    self.assertEqual(expected, result_to_hex(result))

    expected = ("0xc4923a9c", "0x483df7a0")
    result = prng.threefry_2x32(
        np.uint32([0x13198a2e, 0x03707344]),
        np.uint32([0x243f6a88, 0x85a308d3]))
    self.assertEqual(expected, result_to_hex(result))

  def testThreefry2x32Large(self):
    n = 10000000
    result = prng.threefry_2x32(
      (np.uint32(0x13198a2e), np.uint32(0x03707344)),
      jnp.concatenate([
        jnp.full((n,), 0x243f6a88, jnp.uint32),
        jnp.full((n,), 0x85a308d3, jnp.uint32)
      ]))
    np.testing.assert_equal(result[:n], np.full((n,), 0xc4923a9c, dtype=np.uint32))
    np.testing.assert_equal(result[n:], np.full((n,), 0x483df7a0, dtype=np.uint32))

  def testThreefry2x32Empty(self):
    # Regression test for an op-by-op crash for empty arrays in CUDA mode.
    with jax.disable_jit():
      result = prng.threefry_2x32(
        (np.uint32(0x13198a2e), np.uint32(0x03707344)),
        jnp.ones((10, 0,), jnp.uint32))
    np.testing.assert_equal(result, np.zeros((10, 0,), dtype=np.uint32))

  def testNoOpByOpUnderHash(self):
    def fail(*args, **kwargs): assert False
    apply_primitive, xla.apply_primitive = xla.apply_primitive, fail
    try:
      _ = prng.threefry_2x32(np.zeros(2, np.uint32), np.arange(10, dtype=np.uint32))
    finally:
      xla.apply_primitive = apply_primitive

  @skipIf(config.jax_threefry_partitionable, 'changed random bit values')
  def testRngRandomBits(self):
    # Test specific outputs to ensure consistent random values between JAX versions.

    # TODO(frostig): remove once we always enable_custom_prng
    def random_bits(key, *args):
      key, _ = jax_random._check_prng_key(key)
      return jax_random._random_bits(key, *args)

    key = random.PRNGKey(1701)

    bits8 = random_bits(key, 8, (3,))
    expected8 = np.array([216, 115,  43], dtype=np.uint8)
    self.assertArraysEqual(bits8, expected8)

    bits16 = random_bits(key, 16, (3,))
    expected16 = np.array([41682,  1300, 55017], dtype=np.uint16)
    self.assertArraysEqual(bits16, expected16)

    bits32 = random_bits(key, 32, (3,))
    expected32 = np.array([56197195, 4200222568, 961309823], dtype=np.uint32)
    self.assertArraysEqual(bits32, expected32)

    with jtu.ignore_warning(category=UserWarning, message="Explicitly requested dtype.*"):
      bits64 = random_bits(key, 64, (3,))
    if config.x64_enabled:
      expected64 = np.array([3982329540505020460, 16822122385914693683,
                             7882654074788531506], dtype=np.uint64)
    else:
      expected64 = np.array([676898860, 3164047411, 4010691890], dtype=np.uint32)
    self.assertArraysEqual(bits64, expected64)

  @jtu.sample_product(prng_name=[name for name, _ in PRNG_IMPLS])
  def testRngRandomBitsShapeDtype(self, prng_name):
    # Like testRngRandomBits, but only meant to exercise random_bits
    # on every PRNG implementation. Instead of values, only checks
    # that shapes/dtypes are as expected.

    # TODO(frostig): remove once we always enable_custom_prng
    def random_bits(key, *args):
      key, _ = jax_random._check_prng_key(key)
      return jax_random._random_bits(key, *args)

    with jax.default_prng_impl(prng_name):
      key = random.PRNGKey(1701)

      bits8 = random_bits(key, 8, (3,))
      self.assertEqual(bits8.shape, (3,))
      self.assertEqual(bits8.dtype, np.dtype('uint8'))

      bits16 = random_bits(key, 16, (3,))
      self.assertEqual(bits16.shape, (3,))
      self.assertEqual(bits16.dtype, np.dtype('uint16'))

      bits32 = random_bits(key, 32, (3,))
      self.assertEqual(bits32.shape, (3,))
      self.assertEqual(bits32.dtype, np.dtype('uint32'))

      with jtu.ignore_warning(category=UserWarning, message="Explicitly requested dtype.*"):
        bits64 = random_bits(key, 64, (3,))
      expected_dtype = np.dtype('uint64' if config.x64_enabled else 'uint32')
      self.assertEqual(bits64.shape, (3,))
      self.assertEqual(bits64.dtype, expected_dtype)

  @skipIf(config.jax_threefry_partitionable, 'changed random bit values')
  def testRngRandomBitsViewProperty(self):
    # TODO: add 64-bit if it ever supports this property.
    # TODO: will this property hold across endian-ness?

    # TODO(frostig): remove once we always enable_custom_prng
    def random_bits(key, *args):
      key, _ = jax_random._check_prng_key(key)
      return jax_random._random_bits(key, *args)

    N = 10
    key = random.PRNGKey(1701)
    nbits = [8, 16, 32]
    rand_bits = [random_bits(key, n, (N * 64 // n,)) for n in nbits]
    rand_bits_32 = np.array([np.array(r).view(np.uint32) for r in rand_bits])
    assert np.all(rand_bits_32 == rand_bits_32[0])


  @jtu.sample_product(case=_RANDOM_VALUES_CASES)
  @skipIf(config.jax_threefry_partitionable, 'changed random bit values')
  @jtu.skip_on_devices("tpu")  # TPU precision causes issues.
  def testRandomDistributionValues(self, case):
    """
    Tests values output by various distributions. This will catch any unintentional
    changes to the implementations that could result in different random sequences.

    Any refactoring of random distributions that leads to non-trivial differences in
    this test should involve a deprecation cycle following the procedures outlined at
    https://jax.readthedocs.io/en/latest/api_compatibility.html
    """
    if config.x64_enabled and case.skip_on_x64:
      self.skipTest("test produces different values when jax_enable_x64=True")
    with jax.default_prng_impl(case.prng_impl):
      func = getattr(random, case.name)
      key = random.PRNGKey(case._seed())
      if case.dtype:
        actual = func(key, **case.params, shape=case.shape, dtype=case.dtype)
      else:
        actual = func(key, **case.params, shape=case.shape)
      self.assertAllClose(actual, case.expected, atol=case.atol, rtol=case.rtol)

  @skipIf(config.jax_threefry_partitionable, 'changed random bit values')
  def testPRNGValues(self):
    # Test to ensure consistent random values between JAX versions
    k = random.PRNGKey(0)

    self.assertEqual(random.randint(k, (3, 3), 0, 8).dtype,
                     dtypes.canonicalize_dtype(jnp.int_))
    if config.x64_enabled:
        self.assertAllClose(
            random.randint(k, (3, 3), 0, 8, dtype='int64'),
            np.array([[7, 2, 6],
                       [2, 1, 0],
                       [6, 7, 7]], dtype='int64'))
    self.assertAllClose(
        random.randint(k, (3, 3), 0, 8, dtype='int32'),
        np.array([[2, 1, 3],
                  [6, 1, 5],
                  [6, 3, 4]], dtype='int32'))

    self.assertAllClose(
        _prng_key_as_array(random.split(k, 4)),
        np.array([[2285895361, 1501764800],
                  [1518642379, 4090693311],
                  [ 433833334, 4221794875],
                  [ 839183663, 3740430601]], dtype='uint32'))

    self.assertAllClose(
        _prng_key_as_array(random.fold_in(k, 4)),
        np.array([2285895361,  433833334], dtype='uint32'))

  @skipIf(not config.jax_threefry_partitionable, 'enable after upgrade')
  def test_threefry_split_fold_in_symmetry(self):
    with jax.default_prng_impl('threefry2x32'):
      key = random.PRNGKey(72)
      f1, f2, f3 = [random.fold_in(key, i) for i in range(3)]
      s1, s2, s3 = random.split(key, 3)
      f1, f2, f3 = map(_prng_key_as_array, [f1, f2, f3])
      s1, s2, s3 = map(_prng_key_as_array, [s1, s2, s3])
      self.assertArraysEqual(f1, s1)
      self.assertArraysEqual(f2, s2)
      self.assertArraysEqual(f3, s3)

  @skipIf(not config.jax_threefry_partitionable, 'enable after upgrade')
  def test_threefry_split_vmapped_fold_in_symmetry(self):
    # See https://github.com/google/jax/issues/7708
    with jax.default_prng_impl('threefry2x32'):
      key = random.PRNGKey(72)
      f1, f2, f3 = vmap(lambda k, _: random.fold_in(k, lax.axis_index('batch')),
                        in_axes=(None, 0), axis_name='batch')(key, jnp.ones(3))
      s1, s2, s3 = random.split(key, 3)
      f1, f2, f3 = map(_prng_key_as_array, [f1, f2, f3])
      s1, s2, s3 = map(_prng_key_as_array, [s1, s2, s3])
      self.assertArraysEqual(f1, s1)
      self.assertArraysEqual(f2, s2)
      self.assertArraysEqual(f3, s3)

  @parameterized.parameters([
      {"seed": 0, "typ": int, "jit": True, "key": [0, 0]},
      {"seed": 0, "typ": int, "jit": False, "key": [0, 0]},
      {"seed": 1, "typ": np.int32, "jit": True, "key": [0, 1]},
      {"seed": 1, "typ": np.int32, "jit": False, "key": [0, 1]},
      {"seed": 2, "typ": np.uint32, "jit": True, "key": [0, 2]},
      {"seed": 2, "typ": np.uint32, "jit": False, "key": [0, 2]},
      {"seed": 3, "typ": np.int64, "jit": True, "key": [0, 3]},
      {"seed": 3, "typ": np.int64, "jit": False, "key": [0, 3]},
      {"seed": -1, "typ": int, "jit": True, "key": [4294967295, 4294967295] if config.x64_enabled else [0, 4294967295]},
      {"seed": -1, "typ": int, "jit": False, "key": [4294967295, 4294967295] if config.x64_enabled else [0, 4294967295]},
      {"seed": -2, "typ": np.int32, "jit": True, "key": [0, 4294967294]},
      {"seed": -2, "typ": np.int32, "jit": False, "key": [0, 4294967294]},
      {"seed": -3, "typ": np.int64, "jit": True, "key": [4294967295, 4294967293] if config.x64_enabled else [0, 4294967293]},
      {"seed": -3, "typ": np.int64, "jit": False, "key": [4294967295, 4294967293] if config.x64_enabled else [0, 4294967293]},
      {"seed": np.iinfo(np.int32).max + 100, "typ": int, "jit": True, "key": [0, 2147483747]},
      {"seed": np.iinfo(np.int32).max + 100, "typ": int, "jit": False, "key": [0, 2147483747]},
      {"seed": np.iinfo(np.int32).max + 101, "typ": np.uint32, "jit": True, "key": [0, 2147483748]},
      {"seed": np.iinfo(np.int32).max + 101, "typ": np.uint32, "jit": False, "key": [0, 2147483748]},
      {"seed": np.iinfo(np.int32).min - 100, "typ": int, "jit": True, "key": [4294967295, 2147483548] if config.x64_enabled else [0, 2147483548]},
      {"seed": np.iinfo(np.int32).min - 100, "typ": int, "jit": False, "key": [4294967295, 2147483548] if config.x64_enabled else [0, 2147483548]},
      {"seed": np.iinfo(np.int32).min - 101, "typ": np.int64, "jit": True, "key": [4294967295, 2147483547] if config.x64_enabled else [0, 2147483547]},
      {"seed": np.iinfo(np.int32).min - 101, "typ": np.int64, "jit": False, "key": [4294967295, 2147483547] if config.x64_enabled else [0, 2147483547]},
  ])
  def test_prng_seeds_and_keys(self, seed, typ, jit, key):
    seed = typ(seed)
    if jit:
      maker = lambda k: _prng_key_as_array(jax.jit(random.PRNGKey)(k))
    else:
      maker = lambda k: _prng_key_as_array(random.PRNGKey(k))
    if (jit and typ is int and not config.x64_enabled and
        (seed < np.iinfo('int32').min or seed > np.iinfo('int32').max)):
      # We expect an error to be raised.
      # NOTE: we check 'if jit' because some people rely on builtin int seeds
      # (e.g. from PRNGKey(hash("altair is best plotting library"))) outside jit

      # First check with no cache entry (note lambda above).
      with self.assertRaises(OverflowError):
        maker(seed)

      # Then populate a cache entry.
      maker(typ(0)).block_until_ready()

      # Then check now that we have a cache entry.
      with self.assertRaises(OverflowError):
        maker(seed)

    else:
      # Otherwise we expect no error.
      actual = maker(seed)
      expected = jnp.array(key, dtype=jnp.uint32)
      self.assertArraysEqual(actual, expected)

  def test_default_prng_selection(self):
    if not config.jax_enable_custom_prng:
      self.skipTest("test requires config.jax_enable_custom_prng")
    for name, impl in PRNG_IMPLS:
      with jax.default_prng_impl(name):
        self.assertIs(random.default_prng_impl(), impl)
        key = random.PRNGKey(42)
        self.assertIs(key.impl, impl)
        k1, k2 = random.split(key, 2)
        self.assertIs(k1.impl, impl)
        self.assertIs(k2.impl, impl)

  def test_default_prng_selection_without_custom_prng_mode(self):
    if config.jax_enable_custom_prng:
      self.skipTest("test requires that config.jax_enable_custom_prng is False")
    for name, impl in PRNG_IMPLS:
      with jax.default_prng_impl(name):
        self.assertIs(random.default_prng_impl(), impl)
        key = random.PRNGKey(42)
        self.assertEqual(key.shape, impl.key_shape)
        k1, k2 = random.split(key, 2)
        self.assertEqual(k1.shape, impl.key_shape)
        self.assertEqual(k2.shape, impl.key_shape)


  def test_explicit_threefry2x32_key(self):
    if not config.jax_enable_custom_prng:
      self.skipTest("test requires config.jax_enable_custom_prng")
    key = random.threefry2x32_key(42)
    self.assertIs(key.impl, prng.threefry_prng_impl)

  def test_explicit_rbg_key(self):
    if not config.jax_enable_custom_prng:
      self.skipTest("test requires config.jax_enable_custom_prng")
    key = random.rbg_key(42)
    self.assertIs(key.impl, prng.rbg_prng_impl)

  def test_explicit_unsafe_rbg_key(self):
    if not config.jax_enable_custom_prng:
      self.skipTest("test requires config.jax_enable_custom_prng")
    key = random.unsafe_rbg_key(42)
    self.assertIs(key.impl, prng.unsafe_rbg_prng_impl)

  def test_key_array_indexing_0d(self):
    if not config.jax_enable_custom_prng:
      self.skipTest("test requires config.jax_enable_custom_prng")
    key = random.PRNGKey(1701)
    self.assertEqual(key.shape, ())
    self.assertEqual(key[None].shape, (1,))
    self.assertRaisesRegex(IndexError, 'Too many indices.*', lambda: key[0])

  def test_key_array_indexing_nd(self):
    if not config.jax_enable_custom_prng:
      self.skipTest("test requires config.jax_enable_custom_prng")
    keys = vmap(vmap(random.PRNGKey))(jnp.arange(6).reshape((2, 3)))
    self.assertEqual(keys.shape, (2, 3))
    self.assertEqual(keys[0, 0].shape, ())
    self.assertEqual(keys[0, 1].shape, ())
    self.assertEqual(keys[0].shape, (3,))
    self.assertEqual(keys[1, :].shape, (3,))
    self.assertEqual(keys[:, 1].shape, (2,))
    self.assertEqual(keys[None].shape, (1, 2, 3))
    self.assertEqual(keys[None, None].shape, (1, 1, 2, 3))
    self.assertEqual(keys[None, :, None].shape, (1, 2, 1, 3))
    self.assertEqual(keys[None, None, None, 0, None, None, None, 1].shape,
                      (1,) * 6)
    self.assertEqual(keys[..., 1:, None].shape, (2, 2, 1))
    self.assertEqual(keys[None, 0, ..., 1, None].shape, (1, 1))
    self.assertRaisesRegex(IndexError, 'Too many indices.*',
                           lambda: keys[0, 1, 2])
    self.assertRaisesRegex(IndexError, 'Too many indices.*',
                           lambda: keys[0, 1, None, 2])

  def test_isinstance(self):
    if not config.jax_enable_custom_prng:
      self.skipTest("test requires config.jax_enable_custom_prng")
    key = random.PRNGKey(0)
    self.assertIsInstance(key, jax.Array)

  def test_key_output_vjp(self):
    # See https://github.com/google/jax/issues/14856
    def f(seed): return random.PRNGKey(seed)
    jax.vjp(f, 1)  # doesn't crash


class ThreefryPrngTest(jtu.JaxTestCase):
  def test_seed_no_implicit_transfers(self):
    # See https://github.com/google/jax/issues/15613
    with jax.transfer_guard('disallow'):
      random.threefry2x32_key(jax.device_put(42))  # doesn't crash


class LaxRandomTest(jtu.JaxTestCase):

  def _CheckCollisions(self, samples, nbits):
    fail_prob = 0.01  # conservative bound on statistical fail prob by Chebyshev
    nitems = len(samples)
    nbins = 2 ** nbits
    nexpected = nbins * (1 - ((nbins - 1) / nbins) ** nitems)
    ncollisions = len(np.unique(samples))
    sq_percent_deviation = ((ncollisions - nexpected) / nexpected) ** 2
    self.assertLess(sq_percent_deviation, 1 / np.sqrt(nexpected * fail_prob))

  def _CheckKolmogorovSmirnovCDF(self, samples, cdf):
    # conservative bound on statistical fail prob by Kolmo CDF
    # bfloat16 quantization creates much lower p-values in large distributions
    fail_prob = 0.003 if samples.dtype == jnp.bfloat16 else 0.01
    if config.jax_enable_custom_prng and samples.dtype == jnp.bfloat16:
      return
    self.assertGreater(scipy.stats.kstest(samples, cdf).pvalue, fail_prob)

  def _CheckChiSquared(self, samples, pmf):
    if samples.dtype == bool:
      samples = samples.astype(int)
    alpha = 0.01  # significance level, threshold for p-value

    # scipy.stats.chisquare requires the sum of expected and actual to
    # match; this is only the case if we compute the expected frequency
    # at *all* nonzero values of the pmf. We don't know this a priori,
    # so we add extra values past the largest observed value. The number
    # below is empirically enough to get full coverage for the current set
    # of tests. If a new test is added where this is not enough, chisquare()
    # below will error due to the sums of the inputs not matching.
    extra_values = 100
    actual_freq = np.bincount(samples, minlength=samples.max() + extra_values)
    values = np.arange(len(actual_freq))

    expected_freq = pmf(values) * samples.size

    valid = expected_freq > 0
    actual_freq = actual_freq[valid]
    expected_freq = expected_freq[valid]

    _, p_value = scipy.stats.chisquare(actual_freq, expected_freq)
    self.assertGreater(
        p_value, alpha,
        msg=f'Failed chi-squared test with p={p_value}.\n'
            'Expected vs. actual frequencies:\n'
            f'{expected_freq}\n{actual_freq}')

  def seed_prng(self, seed):
    return random.threefry2x32_key(seed)

  @jtu.sample_product(dtype=jtu.dtypes.floating)
  def testNumpyAndXLAAgreeOnFloatEndianness(self, dtype):
    bits_dtype = np.uint32 if jnp.finfo(dtype).bits == 32 else np.uint64
    numpy_bits = np.array(1., dtype).view(bits_dtype)
    xla_bits = jax.jit(
        lambda: lax.bitcast_convert_type(np.array(1., dtype), bits_dtype))()
    self.assertEqual(numpy_bits, xla_bits)

  @jtu.sample_product(dtype=float_dtypes)
  def testRngUniform(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.uniform(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckCollisions(samples, jnp.finfo(dtype).nmant)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.uniform().cdf)

  @jtu.sample_product(dtype=int_dtypes + uint_dtypes)
  def testRngRandint(self, dtype):
    lo = 5
    hi = 10

    key = self.seed_prng(0)
    rand = lambda key: random.randint(key, (10000,), lo, hi, dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertTrue(np.all(lo <= samples))
      self.assertTrue(np.all(samples < hi))

  @jtu.sample_product(dtype=float_dtypes)
  def testNormal(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.normal(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.norm().cdf)

  def testNormalBfloat16(self):
    # Passing bfloat16 as dtype string.
    # https://github.com/google/jax/issues/6813
    res_bfloat16_str = random.normal(self.seed_prng(0), dtype='bfloat16')
    res_bfloat16 = random.normal(self.seed_prng(0), dtype=jnp.bfloat16)
    self.assertAllClose(res_bfloat16, res_bfloat16_str)

  @jtu.sample_product(dtype=complex_dtypes)
  def testNormalComplex(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.normal(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(jnp.real(samples), scipy.stats.norm(scale=1/np.sqrt(2)).cdf)
      self._CheckKolmogorovSmirnovCDF(jnp.imag(samples), scipy.stats.norm(scale=1/np.sqrt(2)).cdf)
      self.assertEqual(dtype, samples.dtype)

  @jtu.sample_product(dtype=float_dtypes)
  def testTruncatedNormal(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.truncated_normal(key, -0.3, 0.3, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    min_val = np.min(uncompiled_samples)
    max_val = np.max(uncompiled_samples)
    self.assertTrue(min_val > -0.3)
    self.assertTrue(max_val < 0.3)
    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.truncnorm(-0.3, 0.3).cdf)

  @jtu.sample_product(dtype=jtu.dtypes.floating + jtu.dtypes.integer)
  def testShuffle(self, dtype):
    key = self.seed_prng(0)
    x = np.arange(100).astype(dtype)
    rand = lambda key: random.shuffle(key, x)
    crand = jax.jit(rand)

    with self.assertWarns(FutureWarning):
      perm1 = rand(key)
    with self.assertWarns(FutureWarning):
      perm2 = crand(key)

    self.assertAllClose(perm1, perm2)
    self.assertFalse(np.all(perm1 == x))  # seems unlikely!
    self.assertAllClose(np.sort(perm1), x, check_dtypes=False)

  @jtu.sample_product(
    [dict(shape=shape, replace=replace, axis=axis,
          input_range_or_shape=input_range_or_shape)
      for shape in [(), (5,), (4, 5)]
      for replace in [True, False]
      for input_range_or_shape in [100, (10, 10), (10, 5, 2), 1, (1, 5)]
      for is_range in [type(input_range_or_shape) is int]
      for ndim in [1 if is_range else len(input_range_or_shape)]
      for axis in range(-ndim, ndim or 1)
      for ninputs in [input_range_or_shape if is_range else input_range_or_shape[axis]]
      if replace or math.prod(shape) <= ninputs
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.integer,
    weighted=[True, False],
  )
  def testChoice(self, dtype, input_range_or_shape, shape, replace, weighted, axis):
    # This is the function API that we test against (note that self.rng().choice differs)
    np_choice = np.random.default_rng(0).choice
    p_dtype = dtypes.to_inexact_dtype(dtype)

    key = self.seed_prng(0)
    is_range = type(input_range_or_shape) is int
    x = (input_range_or_shape if is_range else
         self.rng().permutation(np.arange(math.prod(
           input_range_or_shape), dtype=dtype)).reshape(input_range_or_shape))
    N = x if is_range else x.shape[axis]
    if weighted:
      p = np.arange(N, dtype=p_dtype) + 1
      p /= p.sum()
    else:
      p = None
    rand = lambda key, x: random.choice(key, x, shape, replace, p, axis)
    sample = rand(key, x)
    if not is_range:
      self.assertEqual(dtype, sample.dtype)
    np_shape = np.shape(np_choice(x, shape or None, replace, p, axis))
    self.assertEqual(np_shape, sample.shape)
    if not replace and shape:
      def lsort(x):
        if not math.prod(x.shape): return x
        ind = np.lexsort(np.swapaxes(x, axis, -1).reshape((-1, x.shape[axis])))
        return jnp.take(x, ind, axis)
      self.assertArraysEqual(lsort(sample), lsort(np.unique(sample, axis=axis)))
    self.assertArraysEqual(sample, rand(key, np.array(x)))
    self.assertArraysEqual(sample, jax.jit(rand, static_argnames=
      'x' if is_range else None)(key, x))

  @jtu.sample_product(
    [dict(range_or_shape=range_or_shape, axis=axis)
      for range_or_shape in [0, 1, 100, (0,), (1,), (100,),
                             (10, 10), (10, 5, 2), (0, 5), (1, 5)]
      for ndim in [1 if type(range_or_shape) is int else len(range_or_shape)]
      for axis in range(-ndim, ndim or 1)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.integer,
    independent=[True, False],
  )
  def testPermutation(self, dtype, range_or_shape, axis, independent):
    key = self.seed_prng(0)
    is_range = type(range_or_shape) is int
    x = (range_or_shape if is_range else
         self.rng().permutation(np.arange(
           math.prod(range_or_shape), dtype=dtype)).reshape(range_or_shape))
    shape = ((range_or_shape,) if is_range else range_or_shape)
    x_ = np.copy(x)
    rand = lambda key, x: random.permutation(key, x, axis, independent=independent)
    perm = rand(key, x)
    if shape[axis] >= 10:
      self.assertFalse(np.all(perm == x))  # seems unlikely!
    arr = np.arange(x) if is_range else x
    def lsort(x):
      if not math.prod(x.shape): return x
      ind = np.lexsort(np.swapaxes(x, axis, -1).reshape((-1, x.shape[axis])))
      return jnp.take(x, ind, axis)
    if not independent:
      self.assertArraysEqual(lsort(arr), lsort(perm), check_dtypes=not is_range)
    if independent and (arr.shape[axis] > 4) and (arr.size // arr.shape[axis] > 4):
      # Check for independent shuffling if there are >4 vectors of size >4.
      # Chance of false positive is 1 in (5!)^4
      with self.assertRaises(AssertionError):
        self.assertArraysEqual(lsort(arr), lsort(perm), check_dtypes=not is_range)
    self.assertArraysEqual(x_, x)
    self.assertArraysEqual(perm, rand(key, np.array(x)))
    self.assertArraysEqual(perm, jax.jit(rand, static_argnames=
      'x' if is_range else None)(key, x))

  def testPermutationErrors(self):
    key = self.seed_prng(0)
    with self.assertRaises(ValueError):
      random.permutation(key, 10, axis=3)
    with self.assertRaises(TypeError):
      random.permutation(key, 10.)
    with self.assertRaises(core.ConcretizationTypeError):
      jax.jit(random.permutation)(key, 10)

  @jtu.sample_product(
    p=[0.1, 0.5, 0.9],
    dtype=jtu.dtypes.floating,
  )
  def testBernoulli(self, p, dtype):
    key = self.seed_prng(0)
    p = np.array(p, dtype=dtype)
    rand = lambda key, p: random.bernoulli(key, p, (10000,))
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.bernoulli(p).pmf)

  @jtu.sample_product(
    [dict(p=p, axis=axis)
      for (p, axis) in [
        ([.25] * 4, -1),
        ([.1, .2, .3, .4], -1),
        ([[.5, .5], [.1, .9]], 1),
        ([[.5, .1], [.5, .9]], 0),
      ]
    ],
    sample_shape=[(10000,), (5000, 2)],
    dtype=jtu.dtypes.floating,
  )
  def testCategorical(self, p, axis, dtype, sample_shape):
    key = self.seed_prng(0)
    p = np.array(p, dtype=dtype)
    logits = np.log(p) - 42 # test unnormalized
    out_shape = tuple(np.delete(logits.shape, axis))
    shape = sample_shape + out_shape
    rand = partial(random.categorical, shape=shape, axis=axis)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, logits)
    compiled_samples = crand(key, logits)

    if axis < 0:
      axis += len(logits.shape)

    for samples in [uncompiled_samples, compiled_samples]:
      assert samples.shape == shape
      samples = jnp.reshape(samples, (10000,) + out_shape)
      if len(p.shape[:-1]) > 0:
        ps = np.transpose(p, (1, 0)) if axis == 0 else p
        for cat_samples, cat_p in zip(samples.transpose(), ps):
          pmf = lambda x: np.where(x < len(cat_p), cat_p[np.minimum(len(cat_p) - 1, x)], 0.0)
          self._CheckChiSquared(cat_samples, pmf=pmf)
      else:
        pmf = lambda x: np.where(x < len(p), p[np.minimum(len(p) - 1, x)], 0.0)
        self._CheckChiSquared(samples, pmf=pmf)

  def testBernoulliShape(self):
    key = self.seed_prng(0)
    with jax.numpy_rank_promotion('allow'):
      x = random.bernoulli(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @jtu.sample_product(
    a=[0.2, 5.],
    b=[0.2, 5.],
    dtype=[np.float64],  # NOTE: KS test fails with float32
  )
  def testBeta(self, a, b, dtype):
    if not config.x64_enabled:
      raise SkipTest("skip test except on X64")
    key = self.seed_prng(0)
    rand = lambda key, a, b: random.beta(key, a, b, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, a, b)
    compiled_samples = crand(key, a, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.beta(a, b).cdf)

  def testBetaSmallParameters(self, dtype=np.float32):
    # Regression test for beta version of https://github.com/google/jax/issues/9896
    key = self.seed_prng(0)
    a, b = 0.0001, 0.0002
    samples = random.beta(key, a, b, shape=(100,), dtype=dtype)

    # With such small parameters, all samples should be exactly zero or one.
    tol = 5E-2 if jtu.device_under_test() == "tpu" else 1E-3

    zeros = samples[samples < 0.5]
    self.assertAllClose(zeros, jnp.zeros_like(zeros), atol=tol)

    ones = samples[samples >= 0.5]
    self.assertAllClose(ones, jnp.ones_like(ones), atol=tol)

  @jtu.sample_product(dtype=float_dtypes)
  def testCauchy(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.cauchy(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.cauchy().cdf)

  @jtu.sample_product(
    alpha=[np.array([0.2, 1., 5.]),],
    dtype=jtu.dtypes.floating,
  )
  @jtu.skip_on_devices("tpu")  # TODO(mattjj): slow compilation times
  def testDirichlet(self, alpha, dtype):
    key = self.seed_prng(0)
    rand = lambda key, alpha: random.dirichlet(key, alpha, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, alpha)
    compiled_samples = crand(key, alpha)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertAllClose(samples.sum(-1), np.ones(10000, dtype=dtype))
      alpha_sum = sum(alpha)
      for i, a in enumerate(alpha):
        self._CheckKolmogorovSmirnovCDF(samples[..., i], scipy.stats.beta(a, alpha_sum - a).cdf)

  def testDirichletSmallAlpha(self, dtype=np.float32):
    # Regression test for https://github.com/google/jax/issues/9896
    key = self.seed_prng(0)
    alpha = 0.0001 * jnp.ones(3)
    samples = random.dirichlet(key, alpha, shape=(100,), dtype=dtype)

    # Check that results lie on the simplex.
    self.assertAllClose(samples.sum(1), jnp.ones(samples.shape[0]),
                        check_dtypes=False, rtol=1E-5)

    # Check that results contain 1 in one of the dimensions:
    # this is highly likely to be true when alpha is small.
    self.assertAllClose(samples.max(1), jnp.ones(samples.shape[0]),
                        check_dtypes=False, rtol=1E-4)

  @jtu.sample_product(dtype=float_dtypes)
  def testExponential(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.exponential(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.expon().cdf)

  @jtu.sample_product(
    a=[0.1, 1., 10.],
    dtype=jtu.dtypes.floating,
  )
  def testGammaVsLogGamma(self, a, dtype):
    key = self.seed_prng(0)
    rand_gamma = lambda key, a: random.gamma(key, a, (10000,), dtype)
    rand_loggamma = lambda key, a: random.loggamma(key, a, (10000,), dtype)
    crand_loggamma = jax.jit(rand_loggamma)

    self.assertAllClose(rand_gamma(key, a), jnp.exp(rand_loggamma(key, a)))
    self.assertAllClose(rand_gamma(key, a), jnp.exp(crand_loggamma(key, a)))

  @jtu.sample_product(
    a=[0.1, 1., 10.],
    dtype=jtu.dtypes.floating,
  )
  def testGamma(self, a, dtype):
    key = self.seed_prng(0)
    rand = lambda key, a: random.gamma(key, a, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, a)
    compiled_samples = crand(key, a)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gamma(a).cdf)

  def testGammaShape(self):
    key = self.seed_prng(0)
    x = random.gamma(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @jtu.sample_product(
    log_space=[True, False],
    alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
  )
  def testGammaGrad(self, log_space, alpha):
    rng = self.seed_prng(0)
    alphas = np.full((100,), alpha)
    z = random.gamma(rng, alphas)
    if log_space:
      actual_grad = jax.grad(lambda x: lax.exp(random.loggamma(rng, x)).sum())(alphas)
      # TODO(jakevdp): this NaN correction is required because we generate negative infinities
      # in the log-space computation; see related TODO in the source of random._gamma_one().
      actual_grad = jnp.where(jnp.isnan(actual_grad), 0.0, actual_grad)
    else:
      actual_grad = jax.grad(lambda x: random.gamma(rng, x).sum())(alphas)

    eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
    cdf_dot = (scipy.stats.gamma.cdf(z, alpha + eps)
               - scipy.stats.gamma.cdf(z, alpha - eps)) / (2 * eps)
    with np.errstate(over='ignore'):
      pdf = scipy.stats.gamma.pdf(z, alpha)
    expected_grad = -cdf_dot / pdf

    rtol = 2e-2 if jtu.device_under_test() == "tpu" else 7e-4
    self.assertAllClose(actual_grad, expected_grad, check_dtypes=True,
                        rtol=rtol)

  def testGammaGradType(self):
    # Regression test for https://github.com/google/jax/issues/2130
    key = self.seed_prng(0)
    a = jnp.array(1., dtype=jnp.float32)
    b = jnp.array(3., dtype=jnp.float32)
    f = lambda x, y: random.gamma(key=key, a=x, dtype=jnp.float32) / y
    # Should not crash with a type error.
    jax.vjp(f, a, b)

  @jtu.sample_product(
    lam=[0.5, 3, 9, 11, 50, 500],
    dtype=jtu.dtypes.integer,
  )
  def testPoisson(self, lam, dtype):
    key = self.seed_prng(0)
    rand = lambda key, lam: random.poisson(key, lam, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, lam)
    compiled_samples = crand(key, lam)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.poisson(lam).pmf)
      # TODO(shoyer): determine error bounds for moments more rigorously (e.g.,
      # based on the central limit theorem).
      self.assertAllClose(samples.mean(), lam, rtol=0.02, check_dtypes=False)
      self.assertAllClose(samples.var(), lam, rtol=0.03, check_dtypes=False)

  def testPoissonBatched(self):
    key = self.seed_prng(1)
    lam = jnp.concatenate([2 * jnp.ones(10000), 20 * jnp.ones(10000)])
    samples = random.poisson(key, lam, shape=(20000,))
    self._CheckChiSquared(samples[:10000], scipy.stats.poisson(2.0).pmf)
    self._CheckChiSquared(samples[10000:], scipy.stats.poisson(20.0).pmf)

  def testPoissonWithoutShape(self):
    key = self.seed_prng(1)
    lam = 2 * jnp.ones(10000)
    samples = random.poisson(key, lam)
    self._CheckChiSquared(samples, scipy.stats.poisson(2.0).pmf)

  def testPoissonShape(self):
    key = self.seed_prng(0)
    x = random.poisson(key, np.array([2.0, 20.0]), shape=(3, 2))
    assert x.shape == (3, 2)

  def testPoissonZeros(self):
    key = self.seed_prng(0)
    lam = jnp.concatenate([jnp.zeros(10), 20 * jnp.ones(10)])
    samples = random.poisson(key, lam, shape=(2, 20))
    self.assertArraysEqual(samples[:, :10], jnp.zeros_like(samples[:, :10]))

  def testPoissonCornerCases(self):
    key = self.seed_prng(0)
    lam = jnp.array([-1, 0, jnp.nan])
    samples = random.poisson(key, lam, shape=(3,))
    self.assertArraysEqual(samples, jnp.array([-1, 0, -1]), check_dtypes=False)

  @jtu.sample_product(dtype=jtu.dtypes.floating)
  def testGumbel(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.gumbel(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gumbel_r().cdf)

  @jtu.sample_product(dtype=float_dtypes)
  def testLaplace(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.laplace(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.laplace().cdf)

  @jtu.sample_product(dtype=float_dtypes)
  def testLogistic(self, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.logistic(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.logistic().cdf)

  @jtu.sample_product(
    n=range(1, 5),
    shape=[(), (5,), (10, 5)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def testOrthogonal(self, n, shape, dtype):
    key = self.seed_prng(0)
    q = random.orthogonal(key, n, shape, dtype)
    self.assertEqual(q.shape, (*shape, n, n))
    self.assertEqual(q.dtype, dtype)
    with jax.numpy_rank_promotion('allow'):
      self.assertAllClose(
        jnp.einsum('...ij,...jk->...ik', q, jnp.conj(q).swapaxes(-2, -1)),
        jnp.broadcast_to(jnp.eye(n, dtype=dtype), (*shape, n, n))
      )

  @jtu.sample_product(
    p=[.5, 1., 1.5, 2., 2.5],
    shape=[(), (5,), (10, 5)],
    dtype=jtu.dtypes.floating,
  )
  def testGeneralizedNormal(self, p, shape, dtype):
    key = self.seed_prng(0)
    rand = lambda key, p: random.generalized_normal(key, p, shape, dtype)
    crand = jax.jit(rand)
    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)
    for samples in [uncompiled_samples, compiled_samples]:
      self.assertEqual(samples.shape, shape)
      self.assertEqual(samples.dtype, dtype)
      self._CheckKolmogorovSmirnovCDF(samples.ravel(), scipy.stats.gennorm(p).cdf)

  @jtu.sample_product(
    d=range(1, 5),
    p=[.5, 1., 1.5, 2., 2.5],
    shape=[(), (5,), (10, 5)],
    dtype=jtu.dtypes.floating,
  )
  def testBall(self, d, p, shape, dtype):
    key = self.seed_prng(0)
    rand = lambda key, p: random.ball(key, d, p, shape, dtype)
    crand = jax.jit(rand)
    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)
    for samples in [uncompiled_samples, compiled_samples]:
      self.assertEqual(samples.shape, (*shape, d))
      self.assertEqual(samples.dtype, dtype)
      self.assertTrue(((jnp.abs(samples) ** p).sum(-1) <= 1).all())
      norms = (jnp.abs(samples) ** p).sum(-1) ** (d / p)
      self._CheckKolmogorovSmirnovCDF(norms.ravel(), scipy.stats.uniform().cdf)

  @jtu.sample_product(
    b=[0.1, 1., 10.],
    dtype=jtu.dtypes.floating,
  )
  def testPareto(self, b, dtype):
    key = self.seed_prng(0)
    rand = lambda key, b: random.pareto(key, b, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, b)
    compiled_samples = crand(key, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.pareto(b).cdf)

  def testParetoShape(self):
    key = self.seed_prng(0)
    with jax.numpy_rank_promotion('allow'):
      x = random.pareto(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @jtu.sample_product(
    df=[0.1, 1., 10.],
    dtype=jtu.dtypes.floating,
  )
  @jtu.skip_on_devices("cpu", "tpu")  # TODO(phawkins): slow compilation times
  def testT(self, df, dtype):
    key = self.seed_prng(1)
    rand = lambda key, df: random.t(key, df, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, df)
    compiled_samples = crand(key, df)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.t(df).cdf)

  @jtu.sample_product(
    dim=[1, 3, 5],
    dtype=float_dtypes,
    method=['svd', 'eigh', 'cholesky'],
  )
  def testMultivariateNormal(self, dim, dtype, method):
    r = self.rng()
    mean = r.randn(dim)
    cov_factor = r.randn(dim, dim)
    cov = np.dot(cov_factor, cov_factor.T) + dim * np.eye(dim)

    key = self.seed_prng(0)
    rand = partial(random.multivariate_normal, mean=mean, cov=cov,
                   shape=(10000,), method=method)
    crand = jax.jit(rand)

    with jax.numpy_rank_promotion('allow'):
      uncompiled_samples = np.asarray(rand(key), np.float64)
      compiled_samples = np.asarray(crand(key), np.float64)

    inv_scale = scipy.linalg.lapack.dtrtri(np.linalg.cholesky(cov), lower=True)[0]
    for samples in [uncompiled_samples, compiled_samples]:
      centered = samples - mean
      whitened = np.einsum('nj,ij->ni', centered, inv_scale)

      # This is a quick-and-dirty multivariate normality check that tests that a
      # uniform mixture of the marginals along the covariance matrix's
      # eigenvectors follow a standard normal distribution.
      self._CheckKolmogorovSmirnovCDF(whitened.ravel(), scipy.stats.norm().cdf)

  @jtu.sample_product(
    dim=[1, 2, 4],
    mean_batch_size=[(), (3,), (2, 3)],
    cov_batch_size=[(), (3,), (2, 3)],
    shape=[(), (1,), (5,)],
    method=['cholesky', 'svd', 'eigh'],
  )
  def testMultivariateNormalShapes(self, dim, mean_batch_size, cov_batch_size,
                                   shape, method):
    r = self.rng()
    key = self.seed_prng(0)
    eff_batch_size = mean_batch_size \
      if len(mean_batch_size) > len(cov_batch_size) else cov_batch_size
    mean = r.randn(*(mean_batch_size + (dim,)))
    cov_factor = r.randn(*(cov_batch_size + (dim, dim)))
    cov = np.einsum('...ij,...kj->...ik', cov_factor, cov_factor)
    cov += 1e-3 * np.eye(dim)
    shape = shape + eff_batch_size
    with jax.numpy_rank_promotion('allow'):
      samples = random.multivariate_normal(key, mean, cov, shape=shape, method=method)
    assert samples.shape == shape + (dim,)

  def testMultivariateNormalCovariance(self):
    # test code based on https://github.com/google/jax/issues/1869
    N = 100000
    mean = jnp.zeros(4)
    cov = jnp.array([[  0.19,  0.00, -0.13,  0.00],
                     [  0.00,  0.29,  0.00, -0.23],
                     [ -0.13,  0.00,  0.39,  0.00],
                     [  0.00, -0.23,  0.00,  0.49]], dtype=mean.dtype)

    out_np = self.rng().multivariate_normal(mean, cov, N)

    key = self.seed_prng(0)
    with jax.numpy_rank_promotion('allow'):
      out_jnp = random.multivariate_normal(key, mean=mean, cov=cov, shape=(N,))

    var_np = out_np.var(axis=0)
    var_jnp = out_jnp.var(axis=0)
    self.assertAllClose(var_np, var_jnp, rtol=1e-2, atol=1e-2,
                        check_dtypes=False)

    var_np = np.cov(out_np, rowvar=False)
    var_jnp = np.cov(out_jnp, rowvar=False)
    self.assertAllClose(var_np, var_jnp, rtol=1e-2, atol=1e-2,
                        check_dtypes=False)

  @jtu.sample_product(method=['cholesky', 'eigh', 'svd'])
  @jtu.skip_on_devices('gpu', 'tpu')  # Some NaNs on accelerators.
  def testMultivariateNormalSingularCovariance(self, method):
    # Singular covariance matrix https://github.com/google/jax/discussions/13293
    mu = jnp.zeros((2,))
    sigma = jnp.ones((2, 2))
    key = jax.random.PRNGKey(0)
    result = jax.random.multivariate_normal(key, mean=mu, cov=sigma, shape=(10,), method=method)
    self.assertAllClose(result[:, 0], result[:, 1], atol=1e-3, rtol=1e-3)

    # Cholesky fails for singular inputs.
    if method == 'cholesky':
      self.assertTrue(np.all(np.isnan(result)))
    else:
      self.assertFalse(np.any(np.isnan(result)))

  def testIssue222(self):
    x = random.randint(self.seed_prng(10003), (), 0, 0)
    assert x == 0

  def testFoldIn(self):
    key = self.seed_prng(0)
    keys = [_prng_key_as_array(random.fold_in(key, i)) for i in range(10)]
    assert np.unique(keys, axis=0).shape[0] == 10

  def testFoldInBig(self):
    key = self.seed_prng(0)
    seeds = [2 ** 32 - 2, 2 ** 32 - 1]
    keys = [_prng_key_as_array(random.fold_in(key, seed)) for seed in seeds]
    assert np.unique(keys, axis=0).shape[0] == 2

  def testStaticShapeErrors(self):
    if config.jax_disable_jit:
      raise SkipTest("test only relevant when jit enabled")

    @jax.jit
    def feature_map(n, d, sigma=1.0, seed=123):
      key = self.seed_prng(seed)
      W = random.normal(key, (d, n)) / sigma
      w = random.normal(key, (d, )) / sigma
      b = 2 * jnp.pi * random.uniform(key, (d, ))

      phi = lambda x, t: jnp.sqrt(2.0 / d) * jnp.cos(jnp.matmul(W, x) + w*t + b)
      return phi

    self.assertRaisesRegex(TypeError, 'Shapes must be 1D.*',
                           lambda: feature_map(5, 3))

  def testIssue756(self):
    key = self.seed_prng(0)
    w = random.normal(key, ())
    self.assertEqual(w.dtype, dtypes.canonicalize_dtype(jnp.float_))

  def testIssue1789(self):
    def f(x):
      return random.gamma(self.seed_prng(0), x)

    grad(lambda x: jnp.sum(vmap(f)(x)))(jnp.ones(2))

  def testDtypeErrorMessage(self):
    with self.assertRaisesRegex(ValueError, r"dtype argument to.*"):
      random.normal(self.seed_prng(0), (), dtype=jnp.int32)

  def testRandomBroadcast(self):
    """Issue 4033"""
    # test for broadcast issue in https://github.com/google/jax/issues/4033
    key = self.seed_prng(0)
    shape = (10, 2)
    with jax.numpy_rank_promotion('allow'):
      x1 = random.uniform(key, shape, minval=jnp.zeros(2), maxval=jnp.ones(2))
      x2 = random.randint(key, shape, jnp.array([0, 1]), jnp.array([1, 2]))
    assert x1.shape == shape
    assert x2.shape == shape

  def testMaxwellSample(self):
    num_samples = 10**5
    rng = self.seed_prng(0)

    rand = lambda x: random.maxwell(x, (num_samples, ))
    crand = jax.jit(rand)

    loc = jtu.to_default_dtype(scipy.stats.maxwell.mean())
    std = jtu.to_default_dtype(scipy.stats.maxwell.std())

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(np.mean(samples), loc, atol=0., rtol=0.1)
      self.assertAllClose(np.std(samples), std, atol=0., rtol=0.1)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.maxwell().cdf)

  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testWeibullSample(self, concentration, scale):
    num_samples = 10**5
    rng = self.seed_prng(0)

    rand = lambda x: random.weibull_min(x, scale, concentration, (num_samples,))
    crand = jax.jit(rand)

    loc = jtu.to_default_dtype(scipy.stats.weibull_min.mean(c=concentration, scale=scale))
    std = jtu.to_default_dtype(scipy.stats.weibull_min.std(c=concentration, scale=scale))

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(np.mean(samples), loc, atol=0., rtol=0.1)
      self.assertAllClose(np.std(samples), std, atol=0., rtol=0.1)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.weibull_min(
          c=concentration, scale=scale).cdf)

  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testDoublesidedMaxwellSample(self, loc, scale):
    num_samples = 10**4
    rng = self.seed_prng(0)

    rand = lambda key: random.double_sided_maxwell(
        rng, loc, scale, (num_samples,))
    crand = jax.jit(rand)

    mean = loc
    std = np.sqrt(3.) * scale

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    # Compute the double sided maxwell CDF through the one sided maxwell cdf.
    # This is done as follows:
    # P(DSM <= x) = P (loc + scale * radamacher_sample * one_sided_sample <=x) =
    # P (radamacher_sample * one_sided_sample <= (x - loc) / scale) =
    # 1/2 P(one_sided_sample <= (x - loc) / scale)
    #    + 1/2 P( - one_sided_sample <= (x - loc) / scale) =
    #  1/2 P(one_sided_sample <= (x - loc) / scale)
    #    + 1/2 P(one_sided_sample >= - (x - loc) / scale) =
    # 1/2 CDF_one_maxwell((x - loc) / scale))
    #   + 1/2 (1 - CDF_one_maxwell(- (x - loc) / scale)))
    def double_sided_maxwell_cdf(x, loc, scale):
      pos = scipy.stats.maxwell().cdf((x - loc) / scale)
      neg = (1 - scipy.stats.maxwell().cdf((-x + loc) / scale))
      return (pos + neg) / 2

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(samples.mean(), jtu.to_default_dtype(mean), atol=0., rtol=0.1)
      self.assertAllClose(samples.std(), jtu.to_default_dtype(std), atol=0., rtol=0.1)

      self._CheckKolmogorovSmirnovCDF(
          samples, lambda x: double_sided_maxwell_cdf(x, loc, scale))

  def testRadamacher(self):
    rng = self.seed_prng(0)
    num_samples = 10**5

    rand = lambda x: random.rademacher(x, (num_samples,))
    crand = jax.jit(rand)

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      unique_values, counts = np.unique(samples, return_counts=True)
      assert len(unique_values) == 2
      assert len(counts) == 2

      self.assertAllClose(
          counts[0] / num_samples, 0.5, rtol=1e-02, atol=1e-02)
      self.assertAllClose(
          counts[1] / num_samples, 0.5, rtol=1e-02, atol=1e-02)

  def testChoiceShapeIsNotSequenceError(self):
    key = self.seed_prng(0)
    with self.assertRaises(TypeError):
      random.choice(key, 5, 2, replace=False)
    with self.assertRaises(TypeError):
      random.choice(key, 5, 2, replace=True)

  def test_eval_shape_big_random_array(self):
    def f(x):
      return random.normal(self.seed_prng(x), (int(1e12),))
    with jax.enable_checks(False):  # check_jaxpr will materialize array
      jax.eval_shape(f, 0)  # doesn't error

  @jtu.sample_product(
    type_=["int", "np.array", "jnp.array"],
    seed=[-1, 0, 1, (1 << 32) - 1, (1 << 63) - 1, np.uint64((1 << 64) - 1)],
  )
  def test_prng_jit_invariance(self, seed, type_):
    if type_ == "int" and seed == (1 << 64) - 1:
      self.skipTest("Expected failure: Python int too large.")
    if not config.x64_enabled and seed > np.iinfo(np.int32).max:
      self.skipTest("Expected failure: Python int too large.")
    type_ = {"int": int, "np.array": np.array, "jnp.array": jnp.array}[type_]
    args_maker = lambda: [type_(seed)]
    f = lambda s: _maybe_unwrap(self.seed_prng(s))
    self._CompileAndCheck(f, args_maker)

  def test_prng_errors(self):
    seed = np.iinfo(np.int64).max + 1
    with self.assertRaises(OverflowError):
      self.seed_prng(seed)
    with self.assertRaises(OverflowError):
      jax.jit(self.seed_prng)(seed)

  def test_random_split_doesnt_device_put_during_tracing(self):
    key = self.seed_prng(1).block_until_ready()
    with jtu.count_device_put() as count:
      jax.jit(random.split)(key)
    self.assertLessEqual(count[0], 1)  # 1 for the argument device_put

  @jtu.sample_product(dtype=int_dtypes + uint_dtypes)
  def test_randint_bounds(self, dtype):
    min = np.iinfo(dtype).min
    max = np.iinfo(dtype).max
    key = self.seed_prng(1701)
    shape = (10,)
    if np.iinfo(dtype).bits < np.iinfo(dtypes.canonicalize_dtype(int)).bits:
      expected = random.randint(key, shape, min, max, dtype)
      self.assertArraysEqual(expected, random.randint(key, shape, min - 12345, max + 12345, dtype))
    else:
      self.assertRaises(OverflowError, random.randint, key, shape, min - 12345, max + 12345, dtype)

  def test_randint_out_of_range(self):
    key = self.seed_prng(0)

    r = random.randint(key, (10,), 255, 256, np.uint8)
    self.assertAllClose(r, jnp.full_like(r, 255))

    r = random.randint(key, (1000,), -128, 128, np.int8)
    self.assertGreater((r == -128).sum(), 0)
    self.assertGreater((r == 127).sum(), 0)

    r = random.randint(key, (1000,), -1000, 1000, np.uint8)
    self.assertGreater((r == 0).sum(), 0)
    self.assertGreater((r == 255).sum(), 0)

  def test_large_prng(self):
    # https://github.com/google/jax/issues/11010
    def f():
      return jax.random.uniform(jax.random.PRNGKey(3), (308000000, 128), dtype=jnp.bfloat16)

    # just lower, don't run, takes too long
    jax.jit(f).lower()

  @jtu.sample_product(shape=[(3, 4)],
                      logits_shape_base=[(3, 4), (3, 1), (1, 4)],
                      axis=[-3, -2, -1, 0, 1, 2])
  def test_categorical_shape_argument(self, shape, logits_shape_base, axis):
    # https://github.com/google/jax/issues/13124
    logits_shape = list(logits_shape_base)
    logits_shape.insert(axis % (len(logits_shape_base) + 1), 10)
    assert logits_shape[axis] == 10
    logits = jnp.ones(logits_shape)
    samples = jax.random.categorical(jax.random.PRNGKey(0), logits=logits,
                                     axis=axis, shape=shape)
    self.assertEqual(samples.shape, shape)

  @jtu.sample_product(
      df = [0.2, 1., 10., 100.],
      dtype=jtu.dtypes.floating)
  def testChisquare(self, df, dtype):
    key = self.seed_prng(0)

    rand = lambda key, df: random.chisquare(key, df, shape=(10000, ), dtype=dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, df)
    compiled_samples = crand(key, df)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.chi2(df).cdf)

  @jtu.sample_product(
      dfnum = [1., 2., 10. ,100.],
      dfden = [1. ,2., 10., 100.],
      dtype=jtu.dtypes.floating)
  def testF(self, dfnum, dfden, dtype):
    key = self.seed_prng(1)
    rand = lambda key: random.f(key, dfnum, dfden, shape = (10000, ), dtype = dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.f(dfnum, dfden).cdf)

  @jtu.sample_product(
      scale= [0.2, 1., 2., 10. ,100.],
      dtype=jtu.dtypes.floating)
  def testRayleigh(self, scale, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.rayleigh(key, scale, shape = (10000, ), dtype = dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.rayleigh(scale=scale).cdf)

  @jtu.sample_product(
      mean= [0.2, 1., 2., 10. ,100.],
      dtype=jtu.dtypes.floating)
  def testWald(self, mean, dtype):
    key = self.seed_prng(0)
    rand = lambda key: random.wald(key, mean, shape=(10000, ), dtype=dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.invgauss(mu=mean).cdf)

  @jtu.sample_product(
      p= [0.2, 0.3, 0.4, 0.5 ,0.6],
      dtype= [np.int16, np.int32, np.int64])
  def testGeometric(self, p, dtype):
    key = self.seed_prng(1)
    rand = lambda key: random.geometric(key, p, shape=(10000, ), dtype=dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.geom(p).pmf)
      self.assertAllClose(samples.mean(), 1 / p, rtol=0.02, check_dtypes=False)
      self.assertAllClose(samples.var(), (1 - p) / (p * p) , rtol=0.05, check_dtypes=False)

class KeyArrayTest(jtu.JaxTestCase):
  # Key arrays involve:
  # * a Python key array type, backed by an underlying uint32 "base" array,
  # * an abstract shaped array with key element type,
  # * primitives that return or operate on such shaped arrays,
  # * compiler lowerings,
  # * a device-side data representation...
  # Test it all!
  #
  # A handful of these tests follow CustomElementTypesTest in
  # lax_tests.py as an example. If you add a test here (e.g. testing
  # lowering of an key-dtyped shaped array), consider whether it
  # might also be a more general test of opaque element types. If
  # so, add a corresponding test to to CustomElementTypesTest as well.

  def make_keys(self, *shape, seed=None):
    if seed is None:
      seed = 28
    seeds = seed + jnp.arange(math.prod(shape), dtype=jnp.uint32)
    make_key = partial(prng.seed_with_impl, prng.threefry_prng_impl)
    return jnp.reshape(jax.vmap(make_key)(seeds), shape)

  def test_dtype_property(self):
    k1, k2 = self.make_keys(), self.make_keys()
    self.assertEqual(k1.dtype, k2.dtype)

    k3, k4 = jax.random.split(k1, 2)
    self.assertEqual(k1.dtype, k3.dtype)
    self.assertEqual(k3.dtype, k4.dtype)

    g = []
    def f(k):
      g.append(k.dtype)
      return jax.random.split(k)
    _ = jax.jit(f)(k1)
    self.assertEqual(g[0], k1.dtype)
    self.assertEqual(g[0], k2.dtype)

  def test_key_dtype_attributes(self):
    key = self.make_keys()
    key_raw = key.unsafe_raw_array()

    self.assertStartsWith(key.dtype.name, "key")
    self.assertEqual(key.size * key.dtype.itemsize,
                     key_raw.size * key_raw.dtype.itemsize)

  def test_isinstance(self):
    @jax.jit
    def f(k):
      self.assertIsInstance(k, random.KeyArray)
      return k

    k1 = self.make_keys()
    k2 = f(k1)
    self.assertIsInstance(k1, random.KeyArray)
    self.assertIsInstance(k2, random.KeyArray)

  # -- prng primitives

  def test_random_wrap_vmap(self):
    f = partial(prng_internal.random_wrap, impl=prng.threefry_prng_impl)
    base_arr = jnp.arange(6, dtype=jnp.uint32).reshape(3, 2)
    keys = jax.vmap(f, in_axes=0)(base_arr)
    self.assertIsInstance(keys, random.KeyArray)
    self.assertEqual(keys.shape, (3,))
    keys = jax.vmap(f, in_axes=1)(base_arr.T)
    self.assertIsInstance(keys, random.KeyArray)
    self.assertEqual(keys.shape, (3,))

  @jtu.sample_product(use_internal=[False, True])
  def test_random_unwrap(self, use_internal):
    unwrap = prng_internal.random_unwrap if use_internal else random.key_data
    def f(k): return unwrap(k)
    k = self.make_keys(3, 4)
    out = f(k)
    self.assertEqual(out.dtype, np.dtype('uint32'))
    self.assertEqual(out.shape[:2], (3, 4))
    out = jax.jit(f)(k)
    self.assertEqual(out.dtype, np.dtype('uint32'))
    self.assertEqual(out.shape[:2], (3, 4))
    out = jax.vmap(f)(k)
    self.assertEqual(out.dtype, np.dtype('uint32'))
    self.assertEqual(out.shape[:2], (3, 4))
    out = jax.vmap(jax.jit(f))(k)
    self.assertEqual(out.dtype, np.dtype('uint32'))
    self.assertEqual(out.shape[:2], (3, 4))

    # TODO(frostig): simplify when we always enable_custom_prng
    if not (config.jax_enable_custom_prng and use_internal):
      return

    x = jnp.arange(12, dtype=np.dtype('uint32')).reshape(3, 4)
    self.assertRaisesRegex(
        TypeError, 'random_unwrap takes key array operand, got .*',
        lambda: f(x))
    self.assertRaisesRegex(
        TypeError, 'random_unwrap takes key array operand, got .*',
        lambda: jax.jit(f)(x))
    self.assertRaisesRegex(
        TypeError, 'random_unwrap takes key array operand, got .*',
        lambda: jax.vmap(f)(x))

  def test_eval_shape_keys_in(self):
    def f(key):
      return prng_internal.random_bits(key, bit_width=32, shape=(5,))
    out = jax.eval_shape(f, self.make_keys())
    self.assertEqual(out.shape, (5,))
    self.assertEqual(out.dtype, np.dtype('uint32'))

    def f(key):
      return prng_internal.random_bits(key, bit_width=16, shape=(5,))
    out = jax.eval_shape(f, self.make_keys())
    self.assertEqual(out.shape, (5,))
    self.assertEqual(out.dtype, np.dtype('uint16'))

  def test_eval_shape_keys_out(self):
    def f(seed):
      return self.make_keys(seed=seed)
    out = jax.eval_shape(f, 28)
    self.assertEqual(out.shape, ())
    # TODO(frostig): check dtype too when available

  def test_eval_shape_keys_in_out(self):
    def f(key):
      return jax.random.split(key)
    out = jax.eval_shape(f, self.make_keys())
    self.assertEqual(out.shape, (2,))
    # TODO(frostig): check dtype too when available

  def test_vmap(self):
    ks = self.make_keys(3, 4, 5)
    ys = jax.vmap(jax.jit(lambda k: k.T))(ks)
    self.assertEqual(ys.shape, (3, 5, 4))

  # -- dtype-polymorphic operation (esp. lowerings)

  def test_scan_jaxpr(self):
    ks = self.make_keys(3, 4, 5)
    f = lambda ks: jax.lax.scan(lambda _, k: (None, k.T), None, ks)
    jaxpr = jax.make_jaxpr(f)(ks).jaxpr
    # { lambda ; a:key<fry>[3,4,5]. let
    #     b:key<fry>[3,5,4] = scan[
    #       jaxpr={ lambda ; c:key<fry>[4,5]. let
    #           d:key<fry>[5,4] = transpose[permutation=(1, 0)] c
    #         in (d,) }
    #     ] a
    #   in (b,) }
    self.assertLen(jaxpr.invars, 1)
    a, = jaxpr.invars
    self.assertIsInstance(a.aval, core.ShapedArray)
    self.assertEqual(a.aval.shape, (3, 4, 5))
    self.assertIs(type(a.aval.dtype), prng_internal.KeyTy)
    self.assertLen(jaxpr.eqns, 1)
    e, = jaxpr.eqns
    self.assertLen(e.outvars, 1)
    b, = e.outvars
    self.assertIsInstance(b.aval, core.ShapedArray)
    self.assertEqual(b.aval.shape, (3, 5, 4))
    self.assertIs(type(b.aval.dtype), prng_internal.KeyTy)

  def test_scan_lowering(self):
    ks = self.make_keys(3, 4)
    f = lambda ks: jax.lax.scan(lambda _, k: (None, k.T), None, ks)
    _, out = jax.jit(f)(ks)  # doesn't crash
    self.assertIsInstance(out, random.KeyArray)
    self.assertEqual(out.shape, (3, 4))

  def test_slice(self):
    ks = self.make_keys(3, 4)
    ys = jax.jit(lambda x: lax.slice_in_dim(x, 1, 3))(ks)
    self.assertIsInstance(ys, random.KeyArray)
    self.assertEqual(ys.shape, (2, 4))

  def test_dynamic_slice(self):
    ks = self.make_keys(3, 4)
    ys = jax.jit(lambda x, i: lax.dynamic_slice_in_dim(x, i, 2))(ks, 1)
    self.assertIsInstance(ys, random.KeyArray)
    self.assertEqual(ys.shape, (2, 4))

  def test_transpose(self):
    ks = self.make_keys(3, 4)
    ys = jax.jit(lambda x: x.T)(ks)
    self.assertIsInstance(ys, random.KeyArray)
    self.assertEqual(ys.shape, (4, 3))

  def test_gather(self):
    ks = self.make_keys(3, 4)
    ys = jax.jit(lambda x: x[1])(ks)
    self.assertIsInstance(ys, random.KeyArray)
    self.assertEqual(ys.shape, (4,))

    ks = self.make_keys(3, 4, 5)

    ys = jax.jit(lambda x: x[1])(ks)
    self.assertIsInstance(ys, random.KeyArray)
    self.assertEqual(ys.shape, (4, 5))

    ys = jax.jit(lambda x: x[1, 2:4])(ks)
    self.assertIsInstance(ys, random.KeyArray)
    self.assertEqual(ys.shape, (2, 5))

    ys = jax.jit(lambda x: x[1, 2:4, 3])(ks)
    self.assertIsInstance(ys, random.KeyArray)
    self.assertEqual(ys.shape, (2,))

    ys = jax.jit(lambda x: x[:, 2:4, 3:4])(ks)
    self.assertIsInstance(ys, random.KeyArray)
    self.assertEqual(ys.shape, (3, 2, 1))

  def test_device_put(self):
    device = jax.devices()[0]
    keys = self.make_keys(4)
    keys_on_device = jax.device_put(keys, device)
    self.assertArraysEqual(keys, keys_on_device)

  def test_device_put_sharded(self):
    devices = jax.devices()
    keys = self.make_keys(len(devices))
    keys_on_device = jax.device_put_sharded(list(keys), devices)
    self.assertArraysEqual(keys, keys_on_device)

  def test_device_put_replicated(self):
    devices = jax.devices()
    key = self.make_keys()
    keys_on_device = jax.device_put_replicated(key, devices)
    self.assertArraysEqual(jnp.broadcast_to(key, keys_on_device.shape), keys_on_device)

  # TODO(frostig,mattjj): more polymorphic primitives tests


threefry_seed = prng_internal.threefry_seed
threefry_split = prng_internal.threefry_split
threefry_random_bits = prng_internal.threefry_random_bits
threefry_fold_in = prng_internal.threefry_fold_in

def _double_threefry_seed(seed):
  int_t = seed.dtype.type if hasattr(seed, 'dtype') else type(seed)
  s1, s2 = seed ^ int_t(1), seed ^ int_t(3)
  return jnp.vstack([threefry_seed(s1),
                     threefry_seed(s2)])

def _double_threefry_split(key, num):
  split0 = threefry_split(key[0], num)
  split1 = threefry_split(key[1], num)
  merge = jnp.vstack([jnp.expand_dims(split0.T, axis=0),
                      jnp.expand_dims(split1.T, axis=0)])
  return merge.transpose((2, 0, 1))

def _double_threefry_random_bits(key, bit_width, shape):
  bits0 = threefry_random_bits(key[0], bit_width, shape)
  bits1 = threefry_random_bits(key[1], bit_width, shape)
  return bits0 * bits1

def _double_threefry_fold_in(key, data):
  return jnp.vstack([threefry_fold_in(key[0], data),
                     threefry_fold_in(key[1], data)])

double_threefry_prng_impl = prng.PRNGImpl(
    key_shape=(2, 2),
    seed=_double_threefry_seed,
    split=_double_threefry_split,
    random_bits=_double_threefry_random_bits,
    fold_in=_double_threefry_fold_in,
    tag='fry2')

@skipIf(not config.jax_enable_custom_prng,
        'custom PRNG tests require config.jax_enable_custom_prng')
class LaxRandomWithCustomPRNGTest(LaxRandomTest):
  def seed_prng(self, seed):
    return prng.seed_with_impl(double_threefry_prng_impl, seed)

  def test_split_shape(self):
    key = self.seed_prng(73)
    keys = random.split(key, 10)
    self.assertEqual(keys.shape, (10,))

  def test_vmap_fold_in_shape(self):
    # broadcast with scalar
    keys = random.split(self.seed_prng(73), 2)
    msgs = jnp.arange(3)
    out = vmap(lambda i: random.fold_in(keys[0], i))(msgs)
    self.assertEqual(out.shape, (3,))
    out = vmap(lambda k: random.fold_in(k, msgs[0]))(keys)
    self.assertEqual(out.shape, (2,))
    out = vmap(random.fold_in, in_axes=(None, 0))(keys[0], msgs)
    self.assertEqual(out.shape, (3,))
    out = vmap(random.fold_in, in_axes=(0, None))(keys, msgs[0])
    self.assertEqual(out.shape, (2,))

    # vmap all
    msgs = jnp.arange(2)
    out = vmap(random.fold_in)(keys, msgs)
    self.assertEqual(out.shape, (2,))

    # nested vmap
    keys = random.split(self.seed_prng(73), 2 * 3).reshape((2, 3))
    msgs = jnp.arange(2 * 3).reshape((2, 3))
    out = vmap(vmap(random.fold_in), in_axes=(0, 1))(keys, msgs.T)
    self.assertEqual(out.shape, (2, 3))
    out = vmap(vmap(random.fold_in), in_axes=(1, 0))(keys, msgs.T)
    self.assertEqual(out.shape, (3, 2))

  def test_vmap_split_mapped_key(self):
    key = self.seed_prng(73)
    mapped_keys = random.split(key, num=3)
    forloop_keys = [random.split(k) for k in mapped_keys]
    vmapped_keys = vmap(random.split)(mapped_keys)
    self.assertEqual(vmapped_keys.shape, (3, 2))
    for fk, vk in zip(forloop_keys, vmapped_keys):
      self.assertArraysEqual(fk.unsafe_raw_array(),
                             vk.unsafe_raw_array())

  def test_cannot_add(self):
    key = self.seed_prng(73)
    self.assertRaisesRegex(
        ValueError, r'dtype=key<.*> is not a valid dtype for JAX type promotion.',
        lambda: key + 47)

  @skipIf(np.__version__ == "1.21.0",
          "https://github.com/numpy/numpy/issues/19305")
  def test_grad_of_prng_key(self):
    key = self.seed_prng(73)
    with self.assertRaisesRegex(TypeError, 'input element type key<fry2>'):
      jax.grad(lambda x: 1., allow_int=True)(key)


# TODO(frostig): remove `with_config` we always enable_custom_prng
@jtu.with_config(jax_default_prng_impl='rbg')
class LaxRandomWithRBGPRNGTest(LaxRandomTest):
  def seed_prng(self, seed):
    return random.rbg_key(seed)

  @skipIf(not config.jax_enable_custom_prng, 'relies on typed key arrays')
  def test_split_shape(self):
    key = self.seed_prng(73)
    keys = random.split(key, 10)
    self.assertEqual(keys.shape, (10,))

  @skipIf(not config.jax_enable_custom_prng, 'relies on typed key arrays')
  def test_vmap_fold_in_shape(self):
    LaxRandomWithCustomPRNGTest.test_vmap_fold_in_shape(self)

  @skipIf(not config.jax_enable_custom_prng, 'relies on typed key arrays')
  def test_vmap_split_not_mapped_key(self):
    key = self.seed_prng(73)
    single_split_key = random.split(key)
    vmapped_keys = vmap(lambda _: random.split(key))(jnp.zeros(3,))
    self.assertEqual(vmapped_keys.shape, (3, 2))
    for vk in vmapped_keys:
      self.assertArraysEqual(vk.unsafe_raw_array(),
                             single_split_key.unsafe_raw_array())

  @skipIf(not config.jax_enable_custom_prng, 'relies on typed key arrays')
  def test_vmap_split_mapped_key(self):
    key = self.seed_prng(73)
    mapped_keys = random.split(key, num=3)
    forloop_keys = [random.split(k) for k in mapped_keys]
    vmapped_keys = vmap(random.split)(mapped_keys)
    self.assertEqual(vmapped_keys.shape, (3, 2))
    for fk, vk in zip(forloop_keys, vmapped_keys):
      self.assertArraysEqual(fk.unsafe_raw_array(),
                             vk.unsafe_raw_array())

  def test_vmap_random_bits(self):
    rand_fun = lambda key: random.randint(key, (), 0, 100)
    key = self.seed_prng(73)
    mapped_keys = random.split(key, num=3)
    forloop_rand_nums = [rand_fun(k) for k in mapped_keys]
    rand_nums = vmap(rand_fun)(mapped_keys)
    self.assertEqual(rand_nums.shape, (3,))
    self.assertArraysEqual(rand_nums, jnp.array(forloop_rand_nums))

  @skipIf(not config.jax_enable_custom_prng, 'relies on typed key arrays')
  def test_cannot_add(self):
    key = self.seed_prng(73)
    self.assertRaisesRegex(
        ValueError, r'dtype=key<.*> is not a valid dtype for JAX type promotion.',
        lambda: key + 47)

  @skipIf(np.__version__ == "1.21.0",
          "https://github.com/numpy/numpy/issues/19305")
  @skipIf(not config.jax_enable_custom_prng, 'relies on typed key arrays')
  def test_grad_of_prng_key(self):
    key = self.seed_prng(73)
    with self.assertRaisesRegex(TypeError, 'input element type key<.?rbg>'):
      jax.grad(lambda x: 1., allow_int=True)(key)

  def test_random_split_doesnt_device_put_during_tracing(self):
    return  # this test doesn't apply to the RBG PRNG

  def test_randint_out_of_range(self):
    # TODO(mattjj): enable this test if/when RngBitGenerator supports it
    raise SkipTest('8-bit types not supported with RBG PRNG')


# TODO(frostig): remove `with_config` we always enable_custom_prng
@jtu.with_config(jax_default_prng_impl='unsafe_rbg')
class LaxRandomWithUnsafeRBGPRNGTest(LaxRandomWithRBGPRNGTest):
  def seed_prng(self, seed):
    return random.unsafe_rbg_key(seed)

def like(keys):
  return jnp.ones(keys.shape)

@skipIf(not config.jax_enable_custom_prng,
        'custom PRNG tests require config.jax_enable_custom_prng')
class JnpWithKeyArrayTest(jtu.JaxTestCase):
  def check_shape(self, func, *args):
    out_key = func(*args)
    self.assertIsInstance(out_key, random.KeyArray)
    out_like_key = func(*tree_util.tree_map(like, args))
    self.assertIsInstance(out_like_key, jax.Array)
    self.assertEqual(out_key.shape, out_like_key.shape)

  def check_against_reference(self, key_func, arr_func, *key_args):
    out_arr = arr_func(*tree_util.tree_map(lambda x: x.unsafe_raw_array(), key_args))
    self.assertIsInstance(out_arr, jax.Array)

    out_key = key_func(*key_args)
    self.assertIsInstance(out_key, random.KeyArray)
    self.assertArraysEqual(out_key.unsafe_raw_array(), out_arr)

    out_key = jax.jit(key_func)(*key_args)
    self.assertIsInstance(out_key, random.KeyArray)
    self.assertArraysEqual(out_key.unsafe_raw_array(), out_arr)

  @parameterized.parameters([
    [(2, 3), 'shape', (2, 3)],
    [(2, 3), 'size', 6],
    [(2, 3), 'ndim', 2]
  ])
  def test_properties(self, shape, prop, expected):
    get_prop = lambda x: getattr(x, prop)
    key = random.split(random.PRNGKey(0), math.prod(shape)).reshape(shape)
    self.assertEqual(get_prop(key), expected)
    self.assertEqual(jax.jit(get_prop)(key), expected)

  def test_reshape(self):
    key = random.PRNGKey(123)
    keys = random.split(key, 4)

    newshape = (2, 2)
    key_func = partial(jnp.reshape, newshape=newshape)
    arr_func = partial(jnp.reshape, newshape=(*newshape, *key.impl.key_shape))

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_tile(self):
    key = random.PRNGKey(123)

    reps = 3
    key_func = partial(jnp.tile, reps=reps)
    arr_func = lambda x: jnp.tile(x[None], reps=(reps, *(1 for _ in key.impl.key_shape)))

    self.check_shape(key_func, key)
    self.check_against_reference(key_func, arr_func, key)

  def test_concatenate(self):
    key = random.PRNGKey(123)
    args = [random.split(k, 2) for k in random.split(key, 3)]

    key_func = arr_func = partial(jnp.concatenate, axis=0)

    self.check_shape(key_func, args)
    self.check_against_reference(key_func, arr_func, args)

  def test_broadcast_to(self):
    key = random.PRNGKey(123)

    shape = (3,)
    key_func = partial(jnp.broadcast_to, shape=shape)
    arr_func = partial(jnp.broadcast_to, shape=(*shape, *key.impl.key_shape))

    self.check_shape(key_func, key)
    self.check_against_reference(key_func, arr_func, key)

  def test_expand_dims(self):
    key = random.PRNGKey(123)
    keys = random.split(key, 6).reshape(2, 3)

    key_func = arr_func = partial(jnp.expand_dims, axis=1)

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_broadcast_arrays(self):
    key = random.PRNGKey(123)
    keys = jax.random.split(key, 3)

    key_func = arr_func = lambda *args: jnp.broadcast_arrays(*args)[0]

    self.check_shape(key_func, key, keys)
    self.check_against_reference(key_func, arr_func, key, keys)

  def test_append(self):
    key = random.PRNGKey(123)
    keys = random.split(key, 4)

    key_func = jnp.append
    arr_func = lambda keys, key: jnp.append(keys, key[None], axis=0)

    self.check_shape(key_func, keys, key)
    self.check_against_reference(key_func, arr_func, keys, key)

  def test_ravel(self):
    key = random.PRNGKey(123)
    keys = jax.random.split(key, 4).reshape(2, 2)

    key_func = jnp.ravel
    arr_func = partial(jnp.reshape, newshape=(4, *key.impl.key_shape))

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_stack(self):
    key = random.PRNGKey(123)
    keys = jax.random.split(key, 2)

    key_func = arr_func = partial(jnp.stack, axis=0)

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_array(self):
    key = random.PRNGKey(123)
    self.assertArraysEqual(key, jnp.array(key))
    self.assertArraysEqual(key, jnp.asarray(key))
    self.assertArraysEqual(key, jax.jit(jnp.array)(key))
    self.assertArraysEqual(key, jax.jit(jnp.asarray)(key))

  def test_array_user_dtype(self):
    key = random.PRNGKey(123)
    self.assertArraysEqual(key, jnp.array(key, dtype=key.dtype))
    self.assertArraysEqual(key, jnp.asarray(key, dtype=key.dtype))

  @parameterized.parameters([
    (0,),
    (slice(1),),
    (np.array([0, 2]),),
    (np.array([False, True, True]),)
  ])
  def test_getitem(self, idx):
    key = random.PRNGKey(123)
    keys = jax.random.split(key, 3)

    key_func = arr_func = lambda x: x[idx]

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  @parameterized.parameters([
    (0,),
    (slice(1),),
    (np.array([0, 2]),),
    (np.array([False, True, True]),)
  ])
  def test_gather(self, idx):
    key = random.PRNGKey(123)
    keys = jax.random.split(key, 3)

    key_func = arr_func = lambda x: x.at[idx].get()

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_equality(self):
    key = random.PRNGKey(123)
    key2 = random.PRNGKey(456)

    self.assertTrue(key == key)
    self.assertFalse(key == key2)

    self.assertTrue(key != key2)
    self.assertFalse(key != key)

    size = 5
    idx = slice(2, 4)
    key_arr = random.split(key, size).at[idx].set(key)
    expected = jnp.zeros(size, dtype=bool).at[idx].set(True)

    self.assertArraysEqual(key == key_arr, expected)
    self.assertArraysEqual(key != key_arr, ~expected)

  @parameterized.parameters([
    (0,),
    (slice(1),),
    (np.array([0, 2]),),
    (np.array([False, True, True]),)
  ])
  def test_scatter(self, idx):
    key = random.PRNGKey(123)
    keys = jax.random.split(key, 3)

    key_func = arr_func = lambda x, y: x.at[idx].set(y)

    self.check_shape(key_func, keys, key)
    self.check_against_reference(key_func, arr_func, keys, key)



  def test_errors(self):
    key = random.PRNGKey(123)
    with self.assertRaisesRegex(ValueError, "dtype=key<fry> is not a valid dtype"):
      jnp.add(key, 1)
    with self.assertRaisesRegex(ValueError, "dtype=key<fry> is not a valid dtype"):
      key + 1
    with self.assertRaisesRegex(TypeError, "add does not accept dtype key<fry>"):
      jnp.add(key, key)
    with self.assertRaisesRegex(TypeError, "add does not accept dtype key<fry>"):
      key + key
    with self.assertRaisesRegex(TypeError, "neg does not accept dtype key<fry>"):
      jnp.negative(key)
    with self.assertRaisesRegex(TypeError, "neg does not accept dtype key<fry>"):
      -key
    with self.assertRaisesRegex(ValueError, "Cannot call convert_element_type on dtype key<fry>"):
      lax.convert_element_type(key, int)

  def test_eval_shape(self):
    key = jax.random.PRNGKey(1701)
    shapedtype = jax.ShapeDtypeStruct(key.shape, key.dtype)
    out = jax.eval_shape(lambda x: x, shapedtype)
    self.assertEqual(out, shapedtype)

  def test_result_type(self):
    key = jax.random.PRNGKey(123456)
    self.assertEqual(jnp.result_type(key), key.dtype)


def _sampler_unimplemented_with_custom_prng(*args, **kwargs):
  raise SkipTest('sampler only implemented for default RNG')

for test_prefix in [
    'testPoisson',
    'testPoissonBatched',
    'testPoissonShape',
    'testPoissonZeros',
]:
  for attr in dir(LaxRandomTest):
    if attr.startswith(test_prefix):
      setattr(LaxRandomWithCustomPRNGTest, attr,
              _sampler_unimplemented_with_custom_prng)
      setattr(LaxRandomWithRBGPRNGTest, attr,
              _sampler_unimplemented_with_custom_prng)
      setattr(LaxRandomWithUnsafeRBGPRNGTest, attr,
              _sampler_unimplemented_with_custom_prng)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
