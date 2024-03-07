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

import copy
import enum
from functools import partial
import math
from unittest import skipIf
from typing import Any, NamedTuple
import zlib

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import test_util as jtu
from jax import vmap
from jax.interpreters import xla

from jax._src import random as jax_random
from jax._src import prng as prng_internal

config.parse_flags_with_absl()


PRNG_IMPLS = list(prng_internal.prngs.items())
# Remove Pallas keys from this test, which do not run in XLA.
PRNG_IMPLS = [
    (name, impl) for (name, impl) in PRNG_IMPLS if "pallas" not in name]


class OnX64(enum.Enum):
  ALSO = enum.auto()
  SKIP = enum.auto()
  ONLY = enum.auto()

class RandomValuesCase(NamedTuple):
  name: str
  prng_impl: str
  shape: tuple[int, ...]
  dtype: Any
  params: dict
  expected: np.ndarray
  on_x64: OnX64 = OnX64.ALSO
  atol: float | None = None
  rtol: float | None = None

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
    np.array([False, True, True, True, False]), on_x64=OnX64.SKIP),
  RandomValuesCase("bernoulli", "rbg", (5,), None, {'p': 0.5},
    np.array([True, True, True, True, True]), on_x64=OnX64.SKIP),
  RandomValuesCase("beta", "threefry2x32", (5,), np.float32, {'a': 0.8, 'b': 0.9},
    np.array([0.13259 , 0.824893, 0.948363, 0.964155, 0.235448], dtype='float32')),
  RandomValuesCase("beta", "rbg", (5,), np.float32, {'a': 0.8, 'b': 0.9},
    np.array([0.93215 , 0.833959, 0.121902, 0.270003, 0.429541], dtype='float32')),
  # TODO(frostig,jakevdp) add coverage for non-threefry bits
  RandomValuesCase("bits", "threefry2x32", (5,), np.uint8, {},
    np.array([10, 158, 82, 54, 158], dtype='uint8')),
  RandomValuesCase("bits", "threefry2x32", (5,), np.uint16, {},
    np.array([6738, 38161, 50695, 57337, 61600], dtype='uint16')),
  RandomValuesCase("bits", "threefry2x32", (5,), np.uint32, {},
    np.array([1978747883, 4134381225, 3628107870,  689687174, 2788938207], dtype='uint32')),
  RandomValuesCase("bits", "threefry2x32", (5,), np.uint64, {},
    np.array([17649965731882839947, 1415307058040849897, 8282622628079774249,
              14024425113645909402, 2012979996110532418], dtype='uint64'),
    on_x64=OnX64.ONLY),
  RandomValuesCase("cauchy", "threefry2x32", (5,), np.float32, {},
    np.array([ -0.088416, -10.169713, 3.49677, -1.18056, 0.34556], dtype='float32'), rtol=1E-5),
  RandomValuesCase("cauchy", "rbg", (5,), np.float32, {},
    np.array([0.008389, 0.108793, -0.031826, -0.01876, 0.963218], dtype='float32')),
  RandomValuesCase("dirichlet", "threefry2x32", (2,), np.float32, {'alpha': np.array([0.5, 0.6, 0.7], dtype='float32')},
    np.array([[0.003128, 0.009694, 0.987178], [0.025938, 0.479091, 0.494971]], dtype='float32')),
  RandomValuesCase("dirichlet", "rbg", (2,), np.float32, {'alpha': np.array([0.5, 0.6, 0.7], dtype='float32')},
    np.array([[0.080742, 0.525493, 0.393765], [0.006837, 0.804796, 0.188366]], dtype='float32')),
  RandomValuesCase("double_sided_maxwell", "threefry2x32", (5,), np.float32, {"loc": 1, "scale": 2},
    np.array([-2.408914, -3.370437, 3.235352, -0.907734, -1.708732], dtype='float32'), on_x64=OnX64.SKIP),
  RandomValuesCase("double_sided_maxwell", "rbg", (5,), np.float32, {"loc": 1, "scale": 2},
    np.array([4.957495, 3.003086, 5.33935, 2.942878, -1.203524], dtype='float32'), on_x64=OnX64.SKIP),
  RandomValuesCase("exponential", "threefry2x32", (5,), np.float32, {},
    np.array([0.526067, 0.043046, 0.039932, 0.46427 , 0.123886], dtype='float32')),
  RandomValuesCase("exponential", "rbg", (5,), np.float32, {},
    np.array([0.231303, 0.684814, 0.017181, 0.089552, 0.345087], dtype='float32')),
  RandomValuesCase("gamma", "threefry2x32", (5,), np.float32, {'a': 0.8},
    np.array([0.824221, 1.724476, 0.502882, 5.386132, 0.685543], dtype='float32')),
  RandomValuesCase("gamma", "rbg", (5,), np.float32, {'a': 0.8},
    np.array([0.994946, 0.519941, 1.754347, 0.479223, 1.16932 ], dtype='float32')),
  RandomValuesCase("gumbel", "threefry2x32", (5,), np.float32, {},
    np.array([2.06701, 0.911726, 0.145736, 0.185427, -0.00711], dtype='float32')),
  RandomValuesCase("gumbel", "rbg", (5,), np.float32, {},
    np.array([-0.099308, -1.123809, 1.007618, -0.077968, 3.421349], dtype='float32')),
  RandomValuesCase("laplace", "threefry2x32", (5,), np.float32, {},
    np.array([0.578939, -0.204902, 0.555733, 0.911053, -0.96456], dtype='float32')),
  RandomValuesCase("laplace", "rbg", (5,), np.float32, {},
    np.array([-2.970422, 1.925082, -0.757887, -4.444797, 0.561983], dtype='float32')),
  RandomValuesCase("loggamma", "threefry2x32", (5,), np.float32, {'a': 0.8},
    np.array([ 0.240559, -3.575443, -0.450946, -2.161372, -2.943277], dtype='float32')),
  RandomValuesCase("loggamma", "rbg", (5,), np.float32, {'a': 0.8},
    np.array([-0.107021, -0.809968, -0.25546 , -1.212273, -1.946579], dtype='float32')),
  RandomValuesCase("logistic", "threefry2x32", (5,), np.float32, {},
    np.array([0.19611, -1.709053, -0.274093, -0.208322, -1.675489], dtype='float32')),
  RandomValuesCase("logistic", "rbg", (5,), np.float32, {},
    np.array([-0.234923, -0.545184,  0.700992, -0.708609, -1.474884], dtype='float32')),
  RandomValuesCase("maxwell", "threefry2x32", (5,), np.float32, {},
    np.array([3.070779, 0.908479, 1.521317, 0.875551, 1.306137], dtype='float32')),
  RandomValuesCase("maxwell", "rbg", (5,), np.float32, {},
    np.array([2.048746, 0.470027, 1.053105, 1.01969, 2.710645], dtype='float32')),
  RandomValuesCase("multivariate_normal", "threefry2x32", (2,), np.float32, {"mean": np.ones((1, 3)), "cov": np.eye(3)},
    np.array([[ 1.067826,  1.215599,  0.234166], [-0.237534,  1.32591, 1.413987]], dtype='float32'), on_x64=OnX64.SKIP),
  RandomValuesCase("multivariate_normal", "rbg", (2,), np.float32, {"mean": np.ones((1, 3)), "cov": np.eye(3)},
    np.array([[-0.036897, 0.770969, 0.756959], [1.755091, 2.350553, 0.627142]], dtype='float32'), on_x64=OnX64.SKIP),
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
    np.array([-1, -1, -1, -1, 1], dtype='int32'), on_x64=OnX64.SKIP),
  RandomValuesCase("rademacher", "rbg", (5,), np.int32, {},
    np.array([1, 1, 1, -1, -1], dtype='int32'), on_x64=OnX64.SKIP),
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


KEY_CTORS = [random.key, random.PRNGKey]

@jtu.with_config(jax_legacy_prng_key='allow')
class PrngTest(jtu.JaxTestCase):

  def check_key_has_impl(self, key, impl):
    if jnp.issubdtype(key.dtype, dtypes.prng_key):
      self.assertIs(key._impl, impl)
    else:
      self.assertEqual(key.dtype, jnp.dtype('uint32'))
      self.assertEqual(key.shape, impl.key_shape)

  def test_config_prngs_registered(self):
    # TODO(frostig): pull these string values somehow from the
    # jax_default_prng_impl config enum state definition directly,
    # rather than copying manually here?
    self.assertIn('threefry2x32', prng_internal.prngs)
    self.assertIn('rbg',          prng_internal.prngs)
    self.assertIn('unsafe_rbg',   prng_internal.prngs)

  def testThreefry2x32(self):
    # We test the hash by comparing to known values provided in the test code of
    # the original reference implementation of Threefry. For the values, see
    # https://github.com/DEShawResearch/Random123-Boost/blob/65e3d874b67aa7b3e02d5ad8306462f52d2079c0/libs/random/test/test_threefry.cpp#L30-L32
    def result_to_hex(result):
      return tuple(hex(x.copy()).rstrip("L") for x in result)

    expected = ("0x6b200159", "0x99ba4efe")
    result = prng_internal.threefry_2x32(np.uint32([0, 0]), np.uint32([0, 0]))

    self.assertEqual(expected, result_to_hex(result))

    expected = ("0x1cb996fc", "0xbb002be7")
    u32_max = np.iinfo(np.uint32).max
    result = prng_internal.threefry_2x32(np.uint32([u32_max, u32_max]), np.uint32([u32_max, u32_max]))
    self.assertEqual(expected, result_to_hex(result))

    expected = ("0xc4923a9c", "0x483df7a0")
    result = prng_internal.threefry_2x32(
        np.uint32([0x13198a2e, 0x03707344]),
        np.uint32([0x243f6a88, 0x85a308d3]))
    self.assertEqual(expected, result_to_hex(result))

  def testThreefry2x32Large(self):
    n = 10000000
    result = prng_internal.threefry_2x32(
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
      result = prng_internal.threefry_2x32(
        (np.uint32(0x13198a2e), np.uint32(0x03707344)),
        jnp.ones((10, 0,), jnp.uint32))
    np.testing.assert_equal(result, np.zeros((10, 0,), dtype=np.uint32))

  def testNoOpByOpUnderHash(self):
    def fail(*args, **kwargs): assert False
    apply_primitive, xla.apply_primitive = xla.apply_primitive, fail
    try:
      _ = prng_internal.threefry_2x32(np.zeros(2, np.uint32), np.arange(10, dtype=np.uint32))
    finally:
      xla.apply_primitive = apply_primitive

  @skipIf(config.threefry_partitionable.value, 'changed random bit values')
  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def testRngRandomBits(self, make_key):
    # Test specific outputs to ensure consistent random values between JAX versions.
    seed = 1701

    bits8 = random.bits(make_key(seed), (3,), 'uint8')
    expected8 = np.array([216, 115,  43], dtype=np.uint8)
    self.assertArraysEqual(bits8, expected8)

    bits16 = random.bits(make_key(seed), (3,), 'uint16')
    expected16 = np.array([41682,  1300, 55017], dtype=np.uint16)
    self.assertArraysEqual(bits16, expected16)

    bits32 = random.bits(make_key(seed), (3,), 'uint32')
    expected32 = np.array([56197195, 4200222568, 961309823], dtype=np.uint32)
    self.assertArraysEqual(bits32, expected32)

    with jtu.ignore_warning(category=UserWarning, message="Explicitly requested dtype.*"):
      bits64 = random.bits(make_key(seed), (3,), 'uint64')
    if config.enable_x64.value:
      expected64 = np.array([3982329540505020460, 16822122385914693683,
                             7882654074788531506], dtype=np.uint64)
    else:
      expected64 = np.array([56197195, 4200222568, 961309823], dtype=np.uint32)
    self.assertArraysEqual(bits64, expected64)

  @jtu.sample_product(prng_name=[name for name, _ in PRNG_IMPLS],
                      make_key=KEY_CTORS)
  def testRngRandomBitsShapeDtype(self, prng_name, make_key):
    # Like testRngRandomBits, but only meant to exercise random_bits
    # on every PRNG implementation. Instead of values, only checks
    # that shapes/dtypes are as expected.

    def random_bits(key, width, shape):
      dtype = jnp.dtype(f'uint{width}')
      return jax.random.bits(key, shape, dtype)

    seed = 1701

    with jax.default_prng_impl(prng_name):
      bits8 = random_bits(make_key(seed), 8, (3,))
      self.assertEqual(bits8.shape, (3,))
      self.assertEqual(bits8.dtype, np.dtype('uint8'))

      bits16 = random_bits(make_key(seed), 16, (3,))
      self.assertEqual(bits16.shape, (3,))
      self.assertEqual(bits16.dtype, np.dtype('uint16'))

      bits32 = random_bits(make_key(seed), 32, (3,))
      self.assertEqual(bits32.shape, (3,))
      self.assertEqual(bits32.dtype, np.dtype('uint32'))

      with jtu.ignore_warning(category=UserWarning, message="Explicitly requested dtype.*"):
        bits64 = random_bits(make_key(seed), 64, (3,))
      expected_dtype = np.dtype('uint64' if config.enable_x64.value else 'uint32')
      self.assertEqual(bits64.shape, (3,))
      self.assertEqual(bits64.dtype, expected_dtype)

  @skipIf(config.threefry_partitionable.value, 'changed random bit values')
  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def testRngRandomBitsViewProperty(self, make_key):
    # TODO: add 64-bit if it ever supports this property.
    # TODO: will this property hold across endian-ness?

    def random_bits(key, width, shape):
      dtype = jnp.dtype(f'uint{width}')
      return jax.random.bits(key, shape, dtype)

    N = 10
    nbits = [8, 16, 32]
    rand_bits = [random_bits(make_key(1701), n, (N * 64 // n,)) for n in nbits]
    rand_bits_32 = np.array([np.array(r).view(np.uint32) for r in rand_bits])
    assert np.all(rand_bits_32 == rand_bits_32[0])


  @jtu.sample_product(case=_RANDOM_VALUES_CASES, make_key=KEY_CTORS)
  @skipIf(config.threefry_partitionable.value, 'changed random bit values')
  @jtu.skip_on_devices("tpu")  # TPU precision causes issues.
  def testRandomDistributionValues(self, case, make_key):
    """
    Tests values output by various distributions. This will catch any
    unintentional changes to the implementations that could result in
    different random sequences.

    Any refactoring of random distributions that leads to non-trivial
    differences in this test should follow the procedure outlined at
    https://jax.readthedocs.io/en/latest/api_compatibility.html#numerics-and-randomness

    This includes:
    * Announcing the change in the CHANGELOG.md
    * Considering adding a flag that reverts the new behavior, made
      available for a deprecation window's amount of time.
    """
    if config.enable_x64.value:
      self.skipTest("test produces different values when jax_enable_x64=True")
    if not config.enable_x64.value:
      self.skipTest("test only valid when jax_enable_x64=True")
    with jax.default_prng_impl(case.prng_impl):
      func = getattr(random, case.name)
      key = make_key(case._seed())
      if case.dtype:
        actual = func(key, **case.params, shape=case.shape, dtype=case.dtype)
      else:
        actual = func(key, **case.params, shape=case.shape)
      self.assertAllClose(actual, case.expected, atol=case.atol, rtol=case.rtol)

  @skipIf(config.threefry_partitionable.value, 'changed random bit values')
  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def testPRNGValues(self, make_key):
    # Test to ensure consistent random values between JAX versions
    seed = 0
    self.assertEqual(random.randint(make_key(seed), (3, 3), 0, 8).dtype,
                     dtypes.canonicalize_dtype(jnp.int_))
    if config.enable_x64.value:
        self.assertAllClose(
            random.randint(make_key(seed), (3, 3), 0, 8, dtype='int64'),
            np.array([[7, 2, 6],
                       [2, 1, 0],
                       [6, 7, 7]], dtype='int64'))
    self.assertAllClose(
        random.randint(make_key(seed), (3, 3), 0, 8, dtype='int32'),
        np.array([[2, 1, 3],
                  [6, 1, 5],
                  [6, 3, 4]], dtype='int32'))

    self.assertAllClose(
        random.key_data(random.split(make_key(seed), 4)),
        np.array([[2285895361, 1501764800],
                  [1518642379, 4090693311],
                  [ 433833334, 4221794875],
                  [ 839183663, 3740430601]], dtype='uint32'))

    self.assertAllClose(
        random.key_data(random.fold_in(make_key(seed), 4)),
        np.array([2285895361,  433833334], dtype='uint32'))

  @jtu.run_on_devices("gpu")
  def test_threefry_gpu_kernel_lowering(self):
    f = lambda key: jax.random.uniform(key, (1,))
    with jax._src.config.threefry_gpu_kernel_lowering(False):
      hlo_text = jax.jit(f).lower(jax.random.key(17)).as_text()
      if jtu.is_device_rocm():
        self.assertNotIn("hip_threefry2x32", hlo_text)
      else:
        self.assertNotIn("cu_threefry2x32", hlo_text)
    with jax._src.config.threefry_gpu_kernel_lowering(True):
      hlo_text = jax.jit(f).lower(jax.random.key(17)).as_text()
      if jtu.is_device_rocm():
        self.assertIn("hip_threefry2x32", hlo_text)
      else:
        self.assertIn("cu_threefry2x32", hlo_text)

  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def test_random_seed_offset(self, make_key):
    k1 = make_key(17)
    with config.random_seed_offset(3):
      k2 = make_key(17)
    eq = k1 == k2 if k2.ndim == 0 else all(k1 == k2)
    self.assertFalse(eq)

  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def test_random_bits_error(self, make_key):
    msg = 'dtype argument .* must be an unsigned int dtype'
    with self.assertRaisesRegex(ValueError, msg):
      random.bits(make_key(0), (3, 4), np.dtype('int8'))
    with self.assertRaisesRegex(ValueError, msg):
      random.bits(make_key(0), (3, 4), np.dtype('float16'))

  @skipIf(not config.threefry_partitionable.value, 'enable after upgrade')
  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def test_threefry_split_fold_in_symmetry(self, make_key):
    with jax.default_prng_impl('threefry2x32'):
      key = make_key(72)
      f1, f2, f3 = (random.fold_in(key, i) for i in range(3))
      s1, s2, s3 = random.split(key, 3)
      f1, f2, f3 = map(random.key_data, [f1, f2, f3])
      s1, s2, s3 = map(random.key_data, [s1, s2, s3])
      self.assertArraysEqual(f1, s1)
      self.assertArraysEqual(f2, s2)
      self.assertArraysEqual(f3, s3)

  @skipIf(not config.threefry_partitionable.value, 'enable after upgrade')
  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def test_threefry_split_vmapped_fold_in_symmetry(self, make_key):
    # See https://github.com/google/jax/issues/7708
    with jax.default_prng_impl('threefry2x32'):
      key = make_key(72)
      f1, f2, f3 = vmap(lambda k, _: random.fold_in(k, lax.axis_index('batch')),
                        in_axes=(None, 0), axis_name='batch')(key, jnp.ones(3))
      s1, s2, s3 = random.split(key, 3)
      f1, f2, f3 = map(random.key_data, [f1, f2, f3])
      s1, s2, s3 = map(random.key_data, [s1, s2, s3])
      self.assertArraysEqual(f1, s1)
      self.assertArraysEqual(f2, s2)
      self.assertArraysEqual(f3, s3)

  @skipIf(config.threefry_partitionable.value, 'changed random bit values')
  def test_loggamma_nan_corner_case(self):
    # regression test for https://github.com/google/jax/issues/17922
    # This particular key previously led to NaN output.
    # If the underlying implementation ever changes, this test will no longer
    # exercise this corner case, so we compare to a particular output value
    # rather than just checking for lack of NaNs.
    expected = jnp.float32(-4.595436)
    key = random.wrap_key_data(
      jnp.array([3200590325, 713258242], dtype='uint32'))
    actual = random.loggamma(key, 0.0, dtype='float32')
    rtol = 1E-4 if jtu.test_device_matches(["tpu"]) else 1E-6
    self.assertAllClose(expected, actual, rtol=rtol)

  @parameterized.parameters([params
      for d in [
          {"seed": 0, "typ": int, "jit": True, "key": [0, 0]},
          {"seed": 0, "typ": int, "jit": False, "key": [0, 0]},
          {"seed": 1, "typ": np.int32, "jit": True, "key": [0, 1]},
          {"seed": 1, "typ": np.int32, "jit": False, "key": [0, 1]},
          {"seed": 2, "typ": np.uint32, "jit": True, "key": [0, 2]},
          {"seed": 2, "typ": np.uint32, "jit": False, "key": [0, 2]},
          {"seed": 3, "typ": np.int64, "jit": True, "key": [0, 3]},
          {"seed": 3, "typ": np.int64, "jit": False, "key": [0, 3]},
          {"seed": -1, "typ": int, "jit": True, "key": [4294967295, 4294967295] if config.enable_x64.value else [0, 4294967295]},
          {"seed": -1, "typ": int, "jit": False, "key": [4294967295, 4294967295] if config.enable_x64.value else [0, 4294967295]},
          {"seed": -2, "typ": np.int32, "jit": True, "key": [0, 4294967294]},
          {"seed": -2, "typ": np.int32, "jit": False, "key": [0, 4294967294]},
          {"seed": -3, "typ": np.int64, "jit": True, "key": [4294967295, 4294967293] if config.enable_x64.value else [0, 4294967293]},
          {"seed": -3, "typ": np.int64, "jit": False, "key": [4294967295, 4294967293] if config.enable_x64.value else [0, 4294967293]},
          {"seed": np.iinfo(np.int32).max + 100, "typ": int, "jit": True, "key": [0, 2147483747]},
          {"seed": np.iinfo(np.int32).max + 100, "typ": int, "jit": False, "key": [0, 2147483747]},
          {"seed": np.iinfo(np.int32).max + 101, "typ": np.uint32, "jit": True, "key": [0, 2147483748]},
          {"seed": np.iinfo(np.int32).max + 101, "typ": np.uint32, "jit": False, "key": [0, 2147483748]},
          {"seed": np.iinfo(np.int32).min - 100, "typ": int, "jit": True, "key": [4294967295, 2147483548] if config.enable_x64.value else [0, 2147483548]},
          {"seed": np.iinfo(np.int32).min - 100, "typ": int, "jit": False, "key": [4294967295, 2147483548] if config.enable_x64.value else [0, 2147483548]},
          {"seed": np.iinfo(np.int32).min - 101, "typ": np.int64, "jit": True, "key": [4294967295, 2147483547] if config.enable_x64.value else [0, 2147483547]},
          {"seed": np.iinfo(np.int32).min - 101, "typ": np.int64, "jit": False, "key": [4294967295, 2147483547] if config.enable_x64.value else [0, 2147483547]},
      ]
      for params in [dict(**d, make_key=ctor) for ctor in KEY_CTORS]
  ])
  def test_prng_seeds_and_keys(self, seed, typ, jit, key, make_key):
    seed = typ(seed)
    if jit:
      maker = lambda k: random.key_data(jax.jit(make_key)(k))
    else:
      maker = lambda k: random.key_data(make_key(k))
    if (jit and typ is int and not config.enable_x64.value and
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

  @parameterized.parameters([
      {'make_key': ctor, 'name': name, 'impl': impl}
      for ctor in KEY_CTORS
      for name, impl in PRNG_IMPLS])
  def test_default_prng_selection(self, make_key, name, impl):
    with jax.default_prng_impl(name):
      self.assertIs(jax_random.default_prng_impl(), impl)
      key = make_key(42)
      self.check_key_has_impl(key, impl)
      k1, k2 = random.split(key, 2)
      self.check_key_has_impl(k1, impl)
      self.check_key_has_impl(k2, impl)

  @parameterized.parameters([{'make_key': ctor, 'name': name, 'impl': impl}
                             for ctor in KEY_CTORS
                             for name, impl in PRNG_IMPLS])
  def test_key_construction_with_explicit_impl_name(self, make_key, name, impl):
    key = make_key(42, impl=name)
    self.check_key_has_impl(key, impl)

  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def test_isinstance(self, make_key):
    key = make_key(0)
    self.assertIsInstance(key, jax.Array)

  @parameterized.parameters([{'make_key': ctor} for ctor in KEY_CTORS])
  def test_key_output_vjp(self, make_key):
    # See https://github.com/google/jax/issues/14856
    def f(seed): return make_key(seed)
    jax.vjp(f, 1)  # doesn't crash

  def test_legacy_prng_key_flag(self):
    raw_key = jnp.zeros(2, dtype='uint32')
    invalid_key = jnp.zeros(1, dtype='float32')
    msg = "Legacy uint32 key array passed as key to jax.random function."

    with jax.legacy_prng_key('allow'):
      # TODO(jakevdp): remove when enable_custom_prng no longer issues warnings
      with jax.enable_custom_prng(False):
        with self.assertNoWarnings():
          random.uniform(raw_key)

    with jax.legacy_prng_key('warn'):
      with self.assertWarnsRegex(UserWarning, msg):
        random.uniform(raw_key)

    with jax.legacy_prng_key('error'):
      with self.assertRaisesRegex(ValueError, msg):
        random.uniform(raw_key)

      # Invalid key error should take precedence.
      with self.assertRaisesRegex(TypeError, "JAX encountered invalid PRNG key data"):
        random.uniform(invalid_key)


class ThreefryPrngTest(jtu.JaxTestCase):
  @parameterized.parameters([{'make_key': ctor} for ctor in [
      partial(random.PRNGKey, impl='threefry2x32'),
      partial(random.key, impl='threefry2x32')]])
  def test_seed_no_implicit_transfers(self, make_key):
    # See https://github.com/google/jax/issues/15613
    with jax.transfer_guard('disallow'):
      make_key(jax.device_put(42))  # doesn't crash


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
  # lowering of a key-dtyped shaped array), consider whether it
  # might also be a more general test of opaque element types. If
  # so, add a corresponding test to CustomElementTypesTest as well.

  def assertKeysEqual(self, key1, key2):
    self.assertEqual(key1.dtype, key2.dtype)
    self.assertArraysEqual(random.key_data(key1), random.key_data(key2))

  def test_construction(self):
    key = random.key(42)
    self.assertIsInstance(key, prng_internal.PRNGKeyArray)

  def test_random_clone(self):
    # Here we test value semantics and compatibility with jit/vmap
    # key reuse semantics are tested in key_reuse_test.py
    keys = jax.random.split(jax.random.key(0), 5)
    self.assertKeysEqual(keys, jax.random.clone(keys))
    self.assertKeysEqual(keys, jax.jit(jax.random.clone)(keys))
    self.assertKeysEqual(keys, jax.vmap(jax.random.clone)(keys))

  def test_issubdtype(self):
    key = random.key(42)

    self.assertTrue(jnp.issubdtype(key.dtype, key.dtype))
    self.assertTrue(jnp.issubdtype(key.dtype, dtypes.prng_key))
    self.assertTrue(jnp.issubdtype(key.dtype, dtypes.extended))
    self.assertTrue(jnp.issubdtype(key.dtype, np.generic))

    self.assertFalse(jnp.issubdtype(key.dtype, np.integer))
    self.assertFalse(jnp.issubdtype(key.dtype, np.number))
    with self.assertRaisesRegex(TypeError, "Cannot interpret"):
      jnp.issubdtype(key, dtypes.prng_key)

  @skipIf(not config.enable_custom_prng.value, 'relies on typed key upgrade flag')
  def test_construction_upgrade_flag(self):
    key = random.PRNGKey(42)
    self.assertIsInstance(key, prng_internal.PRNGKeyArray)

  def make_keys(self, *shape, seed=28):
    seeds = seed + jnp.arange(math.prod(shape), dtype=jnp.uint32)
    return jax.vmap(random.key)(seeds).reshape(shape)

  def test_key_as_seed(self):
    key = self.make_keys()
    with self.assertRaisesRegex(TypeError, "PRNGKey accepts a scalar seed"):
      random.PRNGKey(key)
    with self.assertRaisesRegex(TypeError, "key accepts a scalar seed"):
      random.key(key)

  def test_non_scalar_seed(self):
    seed_arr = np.arange(4)
    with self.assertRaisesRegex(TypeError, "PRNGKey accepts a scalar seed"):
      random.PRNGKey(seed_arr)
    with self.assertRaisesRegex(TypeError, "key accepts a scalar seed"):
      random.key(seed_arr)

  def test_non_integer_seed(self):
    seed = np.pi
    with self.assertRaisesRegex(TypeError, "PRNG key seed must be an integer"):
      random.PRNGKey(seed)
    with self.assertRaisesRegex(TypeError, "PRNG key seed must be an integer"):
      random.key(seed)

  def test_dtype_property(self):
    k1, k2 = self.make_keys(), self.make_keys()
    self.assertEqual(k1.dtype, k2.dtype)

    k3, k4 = random.split(k1, 2)
    self.assertEqual(k1.dtype, k3.dtype)
    self.assertEqual(k3.dtype, k4.dtype)

    g = []
    def f(k):
      g.append(k.dtype)
      return random.split(k)
    _ = jax.jit(f)(self.make_keys())
    self.assertEqual(g[0], k1.dtype)
    self.assertEqual(g[0], k2.dtype)

  def test_key_dtype_attributes(self):
    key = self.make_keys()
    key_raw = random.key_data(key)

    self.assertStartsWith(key.dtype.name, "key")
    self.assertEqual(key.size * key.dtype.itemsize,
                     key_raw.size * key_raw.dtype.itemsize)

  def test_key_attributes(self):
    key = self.make_keys()
    self.assertEqual(key.itemsize, key.dtype.itemsize)
    self.assertEqual(key.size, math.prod(key.shape))
    self.assertEqual(key.ndim, len(key.shape))

  def test_key_copy(self):
    key = self.make_keys()
    self.assertKeysEqual(key, key.copy())
    self.assertKeysEqual(key, copy.copy(key))
    self.assertKeysEqual(key, copy.deepcopy(key))
    self.assertKeysEqual(key, jax.jit(lambda k: k.copy())(key))

  # TODO(jakevdp) remove this decorator when reuse checks move to C++
  @jax.debug_key_reuse(False)
  def test_cpp_dispatch_normal(self):
    # Ensure we stay on the C++ dispatch path when calling a jitted
    # function with a key array as an argument.

    @jax.jit
    def f(key):
      return jax.random.normal(key)

    key = self.make_keys()
    with jtu.count_pjit_cpp_cache_miss() as count:
      f(key).block_until_ready()
      f(key).block_until_ready()

    self.assertEqual(count[0], 1)

  # TODO(jakevdp) remove this decorator when reuse checks move to C++
  @jax.debug_key_reuse(False)
  def test_cpp_dispatch_split(self):
    # Ensure we stay on the C++ dispatch path when calling a jitted
    # function with a key arrays as inputs and as outputs.

    @jax.jit
    def f(key):
      return jax.random.split(key)

    key = self.make_keys()
    with jtu.count_pjit_cpp_cache_miss() as count:
      f(key).block_until_ready()
      f(key).block_until_ready()

    self.assertEqual(count[0], 1)

  def test_cpp_dispatch_aot_normal(self):
    # Ensure we stay on the C++ dispatch path when calling an
    # AOT-compiled function with a key array as an argument.

    key = self.make_keys()
    f = jax.jit(lambda key: jax.random.normal(key)).lower(key).compile()

    with jtu.count_aot_jit_cpp_cache_miss() as count:
      f(key).block_until_ready()
      f(key).block_until_ready()

    self.assertEqual(count[0], 1)

  def test_cpp_dispatch_aot_split(self):
    # Ensure we stay on the C++ dispatch path when calling an
    # AOT-compiled function with a key arrays as inputs and as
    # outputs.

    key = self.make_keys()
    f = jax.jit(lambda key: jax.random.split(key)).lower(key).compile()

    with jtu.count_aot_jit_cpp_cache_miss() as count:
      f(key).block_until_ready()
      f(key).block_until_ready()

    self.assertEqual(count[0], 1)

  # -- prng primitives

  def test_random_wrap_vmap(self):
    f = partial(prng_internal.random_wrap, impl=prng_internal.threefry_prng_impl)
    base_arr = jnp.arange(6, dtype=jnp.uint32).reshape(3, 2)
    keys = jax.vmap(f, in_axes=0)(base_arr)
    self.assertIsInstance(keys, prng_internal.PRNGKeyArray)
    self.assertEqual(keys.shape, (3,))
    keys = jax.vmap(f, in_axes=1)(base_arr.T)
    self.assertIsInstance(keys, prng_internal.PRNGKeyArray)
    self.assertEqual(keys.shape, (3,))

  @jtu.sample_product(use_internal=[False, True])
  def test_random_unwrap(self, use_internal):
    unwrap = prng_internal.random_unwrap if use_internal else random.key_data
    def f(k): return unwrap(k)
    keys = lambda: self.make_keys(3, 4)
    out = f(keys())
    self.assertEqual(out.dtype, np.dtype('uint32'))
    self.assertEqual(out.shape[:2], (3, 4))
    out = jax.jit(f)(keys())
    self.assertEqual(out.dtype, np.dtype('uint32'))
    self.assertEqual(out.shape[:2], (3, 4))
    out = jax.vmap(f)(keys())
    self.assertEqual(out.dtype, np.dtype('uint32'))
    self.assertEqual(out.shape[:2], (3, 4))
    out = jax.vmap(jax.jit(f))(keys())
    self.assertEqual(out.dtype, np.dtype('uint32'))
    self.assertEqual(out.shape[:2], (3, 4))

    if not use_internal:
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
      return random.split(key)
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
    self.assertIsInstance(out, prng_internal.PRNGKeyArray)
    self.assertEqual(out.shape, (3, 4))

  def test_slice(self):
    ks = self.make_keys(3, 4)
    ys = jax.jit(lambda x: lax.slice_in_dim(x, 1, 3))(ks)
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (2, 4))

  def test_dynamic_slice(self):
    ks = self.make_keys(3, 4)
    index = np.int16(1)  # non-default int type to catch type errors.
    ys = jax.jit(partial(lax.dynamic_slice_in_dim, slice_size=2))(ks, index)
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (2, 4))

  def test_dynamic_update_slice(self):
    ks = self.make_keys(3, 4)
    k = self.make_keys(1, 4)
    index = np.int16(1)  # non-default int type to catch type errors.
    ys = jax.jit(partial(lax.dynamic_update_slice_in_dim, axis=0))(ks, k, index)
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (3, 4))

  def test_transpose(self):
    ks = self.make_keys(3, 4)
    ys = jax.jit(lambda x: x.T)(ks)
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (4, 3))

  def test_gather(self):
    keys = self.make_keys(3, 4)
    ys = jax.jit(lambda x: x[1])(keys)
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (4,))

    keys = lambda: self.make_keys(3, 4, 5)
    ys = jax.jit(lambda x: x[1])(keys())
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (4, 5))

    ys = jax.jit(lambda x: x[1, 2:4])(keys())
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (2, 5))

    ys = jax.jit(lambda x: x[1, 2:4, 3])(keys())
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (2,))

    ys = jax.jit(lambda x: x[:, 2:4, 3:4])(keys())
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (3, 2, 1))

  def test_select(self):
    ks = self.make_keys(3, 2)
    cs = jnp.array([True, False, False, True, False, True]).reshape(3, 2)
    ys = jax.jit(lax.select)(cs, ks, ks)
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (3, 2))

  def test_select_scalar_cond(self):
    # regression test for https://github.com/google/jax/issues/16422
    ks = self.make_keys(3)
    ys = lax.select(True, ks, ks)
    self.assertIsInstance(ys, prng_internal.PRNGKeyArray)
    self.assertEqual(ys.shape, (3,))

  def test_vmap_of_cond(self):
    # See https://github.com/google/jax/issues/15869
    def f(x):
      keys = self.make_keys(*x.shape)
      return lax.select(x, keys, keys)
    x = jnp.array([True, False, False])
    f(x)  # doesn't crash

  def test_device_put(self):
    device = jax.devices()[0]
    keys = self.make_keys(4)
    keys_on_device = jax.device_put(keys, device)
    self.assertKeysEqual(keys, keys_on_device)

  def test_device_put_sharded(self):
    devices = jax.devices()
    keys = self.make_keys(len(devices))
    keys_on_device = jax.device_put_sharded(list(keys), devices)
    self.assertKeysEqual(keys, keys_on_device)

  def test_device_put_replicated(self):
    devices = jax.devices()
    key = self.make_keys()
    keys_on_device = jax.device_put_replicated(key, devices)
    self.assertKeysEqual(jnp.broadcast_to(key, keys_on_device.shape), keys_on_device)

  def test_make_array_from_callback(self):
    devices = jax.devices()
    shape = (len(devices),)
    mesh = jtu.create_global_mesh((len(devices),), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    def callback(index):
      i = jnp.arange(len(devices))[index[0]]
      return jax.vmap(random.key)(i)
    result = jax.make_array_from_callback(shape, sharding, callback)
    expected = jax.vmap(random.key)(jnp.arange(len(devices)))
    self.assertKeysEqual(result, expected)

  def test_make_array_from_single_device_arrays(self):
    devices = jax.devices()
    shape = (len(devices),)
    mesh = jtu.create_global_mesh((len(devices),), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    keys = random.split(random.key(0), len(devices))
    arrays = [jax.device_put(keys[i:i + 1], device) for i, device in enumerate(devices)]
    result = jax.make_array_from_single_device_arrays(shape, sharding, arrays)
    self.assertKeysEqual(result, keys)

  def test_key_array_custom_jvp(self):
    def f_raw(x, key):
        return x * random.normal(key, ())

    f = jax.custom_jvp(f_raw)

    @f.defjvp
    def f_jvp(primals, tangents):
      nonlocal key_dot
      x, key = primals
      x_dot, key_dot = tangents
      rand = random.normal(key, ())
      tangent_out = x_dot * rand
      primal_out = x * rand
      return primal_out, tangent_out

    key_dot = None
    default_result = jax.grad(f_raw)(0.0, self.make_keys())
    custom_result = jax.grad(f)(0.0, self.make_keys())

    self.assertAllClose(default_result, custom_result)
    self.assertEqual(key_dot.dtype, dtypes.float0)

  def test_key_array_indexing_0d(self):
    key = self.make_keys()
    self.assertEqual(key.shape, ())
    self.assertEqual(key[None].shape, (1,))
    key = self.make_keys()
    self.assertRaisesRegex(IndexError, 'Too many indices.*', lambda: key[0])

  def test_key_array_indexing_nd(self):
    keys = lambda: self.make_keys(2, 3)
    self.assertEqual(keys().shape, (2, 3))
    self.assertEqual(keys()[0, 0].shape, ())
    self.assertEqual(keys()[0, 1].shape, ())
    self.assertEqual(keys()[0].shape, (3,))
    self.assertEqual(keys()[1, :].shape, (3,))
    self.assertEqual(keys()[:, 1].shape, (2,))
    self.assertEqual(keys()[None].shape, (1, 2, 3))
    self.assertEqual(keys()[None, None].shape, (1, 1, 2, 3))
    self.assertEqual(keys()[None, :, None].shape, (1, 2, 1, 3))
    self.assertEqual(keys()[None, None, None, 0, None, None, None, 1].shape,
                      (1,) * 6)
    self.assertEqual(keys()[..., 1:, None].shape, (2, 2, 1))
    self.assertEqual(keys()[None, 0, ..., 1, None].shape, (1, 1))
    self.assertRaisesRegex(IndexError, 'Too many indices.*',
                           lambda: keys()[0, 1, 2])
    self.assertRaisesRegex(IndexError, 'Too many indices.*',
                           lambda: keys()[0, 1, None, 2])

  def test_not_hashable(self):
    key = self.make_keys()
    with self.assertRaisesRegex(TypeError, "unhashable type"):
      hash(key)

  def test_array_impl_attributes(self):
    # Test a number of ArrayImpl attributes
    key = self.make_keys(10)

    self.assertEqual(key.is_fully_addressable, key._base_array.is_fully_addressable)
    self.assertEqual(key.is_fully_replicated, key._base_array.is_fully_replicated)
    self.assertEqual(key.devices(), key._base_array.devices())
    self.assertEqual(key.on_device_size_in_bytes(),
                     key._base_array.on_device_size_in_bytes())
    self.assertEqual(key.unsafe_buffer_pointer(),
                     key._base_array.unsafe_buffer_pointer())
    self.assertArraysEqual(key.addressable_data(0)._base_array,
                           key._base_array.addressable_data(0))
    self.assertLen(key.addressable_shards, len(key._base_array.addressable_shards))
    self.assertLen(key.global_shards, len(key._base_array.global_shards))

  def test_delete(self):
    key = self.make_keys(10)

    self.assertFalse(key.is_deleted())
    key.delete()
    self.assertTrue(key.is_deleted())
    self.assertTrue(key._base_array.is_deleted())

  def test_async(self):
    key = self.make_keys(10)

    self.assertKeysEqual(key, key.block_until_ready())
    self.assertIsNone(key.copy_to_host_async())

  # -- key construction and un/wrapping with impls

  def test_wrap_key_default(self):
    key1 = jax.random.key(17)
    data = jax.random.key_data(key1)
    key2 = jax.random.wrap_key_data(data)
    self.assertEqual(key1.dtype, key2.dtype)
    self.assertArraysEqual(jax.random.key_data(key1),
                           jax.random.key_data(key2))

    impl = config.default_prng_impl.value
    key3 = jax.random.wrap_key_data(data, impl=impl)
    self.assertEqual(key1.dtype, key3.dtype)
    self.assertArraysEqual(jax.random.key_data(key1),
                           jax.random.key_data(key3))

  def test_wrap_key_explicit(self):
    key1 = jax.random.key(17, impl='rbg')
    data = jax.random.key_data(key1)
    key2 = jax.random.wrap_key_data(data, impl='rbg')
    self.assertEqual(key1.dtype, key2.dtype)
    self.assertArraysEqual(jax.random.key_data(key1),
                           jax.random.key_data(key2))

    key3 = jax.random.wrap_key_data(data, impl='unsafe_rbg')
    self.assertNotEqual(key1.dtype, key3.dtype)

  @jtu.sample_product(prng_name=[name for name, _ in PRNG_IMPLS])
  def test_key_make_like_other_key(self, prng_name):
    # start by specifying the implementation by string name, then
    # round trip via whatever `key_impl` outputs
    k1 = jax.random.key(42, impl=prng_name)
    impl = jax.random.key_impl(k1)
    k2 = jax.random.key(42, impl=impl)
    self.assertKeysEqual(k1, k2)
    self.assertEqual(k1.dtype, k2.dtype)

  @jtu.sample_product(prng_name=[name for name, _ in PRNG_IMPLS])
  def test_key_wrap_like_other_key(self, prng_name):
    # start by specifying the implementation by string name, then
    # round trip via whatever `key_impl` outputs
    k1 = jax.random.key(42, impl=prng_name)
    data = jax.random.key_data(k1)
    impl = jax.random.key_impl(k1)
    k2 = jax.random.wrap_key_data(data, impl=impl)
    self.assertKeysEqual(k1, k2)
    self.assertEqual(k1.dtype, k2.dtype)

  def test_key_impl_from_string_error(self):
    with self.assertRaisesRegex(ValueError, 'unrecognized PRNG implementation'):
      jax.random.key(42, impl='unlikely name')

  def test_key_impl_from_object_error(self):
    class A: pass

    with self.assertRaisesRegex(TypeError, 'unrecognized type .* PRNG'):
      jax.random.key(42, impl=A())

  def test_keyarray_custom_vjp(self):
    # Regression test for https://github.com/google/jax/issues/18442
    @jax.custom_vjp
    def f(_, state):
      return state
    def _f_fwd(_, state):
      return state, None
    def _f_bwd(_, state_bar):
      assert state_bar[1].dtype == dtypes.float0  # key tangent type
      return state_bar[0], state_bar
    f.defvjp(_f_fwd, _f_bwd)
    state = (8.0, jax.random.key(123))
    result = jax.grad(lambda theta: f(theta, state)[0])(3.0)
    self.assertEqual(result, 1.0)

  def test_keyarray_custom_vjp_symbolic_zeros(self):
    @jax.custom_vjp
    def f(_, state):
      return state
    def _f_fwd(_, state):
      return jax.tree.map(lambda x: x.value, state), None
    def _f_bwd(_, state_bar):
      self.assertTrue(state_bar[1].dtype == dtypes.float0)
      self.assertIsInstance(state_bar[1], jax.custom_derivatives.SymbolicZero)
      return state_bar[0], state_bar
    f.defvjp(_f_fwd, _f_bwd, symbolic_zeros=True)
    state = (8.0, jax.random.key(123))
    result = jax.grad(lambda theta: f(theta, state)[0])(3.0)
    self.assertEqual(result, 1.0)

  # TODO(frostig,mattjj): more polymorphic primitives tests


threefry_seed = prng_internal.threefry_seed
threefry_split = prng_internal.threefry_split
threefry_random_bits = prng_internal.threefry_random_bits
threefry_fold_in = prng_internal.threefry_fold_in

def _double_threefry_seed(seed):
  int_t = seed.dtype.type if hasattr(seed, 'dtype') else type(seed)
  s1, s2 = seed, seed ^ int_t(3)
  return jnp.vstack([threefry_seed(s1),
                     threefry_seed(s2)])

def _double_threefry_split(key, shape):
  return vmap(
      threefry_split, (0, None), len(shape))(key, shape)

def _double_threefry_random_bits(key, bit_width, shape):
  bits0 = threefry_random_bits(key[0], bit_width, shape)
  bits1 = threefry_random_bits(key[1], bit_width, shape)
  del bits1
  # TODO(frostig): Currently this behaves like normal threefry, to
  # avoid a few probabilistic test failures. Ideally we might want to
  # test different generation behavior here (e.g. `bits0 ^ bits1`).
  return bits0

def _double_threefry_fold_in(key, data):
  return jnp.vstack([threefry_fold_in(key[0], data),
                     threefry_fold_in(key[1], data)])

double_threefry_prng_impl = prng_internal.PRNGImpl(
    key_shape=(2, 2),
    seed=_double_threefry_seed,
    split=_double_threefry_split,
    random_bits=_double_threefry_random_bits,
    fold_in=_double_threefry_fold_in,
    tag='fry2')


class JnpWithKeyArrayTest(jtu.JaxTestCase):
  def assertKeysEqual(self, key1, key2):
    self.assertEqual(key1.dtype, key2.dtype)
    self.assertArraysEqual(random.key_data(key1), random.key_data(key2))

  def check_shape(self, func, *args):
    like = lambda keys: jnp.ones(keys.shape)
    out_key = func(*args)
    self.assertIsInstance(out_key, prng_internal.PRNGKeyArray)
    out_like_key = func(*jax.tree.map(like, args))
    self.assertIsInstance(out_like_key, jax.Array)
    self.assertEqual(out_key.shape, out_like_key.shape)

  def check_against_reference(self, key_func, arr_func, *key_args):
    out_arr = arr_func(*jax.tree.map(lambda x: random.key_data(x),
                                           key_args))
    self.assertIsInstance(out_arr, jax.Array)

    out_key = key_func(*key_args)
    self.assertIsInstance(out_key, prng_internal.PRNGKeyArray)
    self.assertArraysEqual(random.key_data(out_key), out_arr)

    out_key = jax.jit(key_func)(*key_args)
    self.assertIsInstance(out_key, prng_internal.PRNGKeyArray)
    self.assertArraysEqual(random.key_data(out_key), out_arr)

  @parameterized.parameters([
    [(2, 3), 'shape', (2, 3)],
    [(2, 3), 'size', 6],
    [(2, 3), 'ndim', 2]
  ])
  def test_properties(self, shape, prop, expected):
    get_prop = lambda x: getattr(x, prop)
    key = random.split(random.key(0), math.prod(shape)).reshape(shape)
    self.assertEqual(get_prop(key), expected)
    self.assertEqual(jax.jit(get_prop)(key), expected)

  def test_reshape(self):
    key = random.key(123)
    keys = random.split(key, 4)

    shape = (2, 2)
    key_func = partial(jnp.reshape, shape=shape)
    arr_func = partial(jnp.reshape, shape=(*shape, *key._impl.key_shape))

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_tile(self):
    key = random.key(123)

    reps = 3
    key_func = partial(jnp.tile, reps=reps)
    arr_func = lambda x: jnp.tile(x[None], reps=(reps, *(1 for _ in key._impl.key_shape)))

    self.check_shape(key_func, key)
    self.check_against_reference(key_func, arr_func, key)

  def test_concatenate(self):
    args = lambda: [random.split(k, 2) for k in random.split(random.key(123), 3)]

    key_func = arr_func = partial(jnp.concatenate, axis=0)

    self.check_shape(key_func, args())
    self.check_against_reference(key_func, arr_func, args())

  def test_broadcast_to(self):
    key = random.key(123)

    shape = (3,)
    key_func = partial(jnp.broadcast_to, shape=shape)
    arr_func = partial(jnp.broadcast_to, shape=(*shape, *key._impl.key_shape))

    self.check_shape(key_func, key)
    self.check_against_reference(key_func, arr_func, key)

  def test_expand_dims(self):
    key = random.key(123)
    keys = random.split(key, 6).reshape(2, 3)

    key_func = arr_func = partial(jnp.expand_dims, axis=1)

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_broadcast_arrays(self):
    key = random.key(123)
    keys = random.split(key, 3)

    key_func = arr_func = lambda *args: jnp.broadcast_arrays(*args)[0]

    self.check_shape(key_func, key, keys)
    self.check_against_reference(key_func, arr_func, key, keys)

  def test_append(self):
    key = lambda: random.key(123)
    keys = lambda: random.split(random.key(123), 4)

    key_func = jnp.append
    arr_func = lambda keys, key: jnp.append(keys, key[None], axis=0)

    self.check_shape(key_func, keys(), key())
    self.check_shape(arr_func, keys(), key())
    with jax.debug_key_reuse(False):
      self.check_against_reference(key_func, arr_func, keys(), key())

  def test_ravel(self):
    key = random.key(123)
    keys = random.split(key, 4).reshape(2, 2)

    key_func = jnp.ravel
    arr_func = partial(jnp.reshape, shape=(4, *key._impl.key_shape))

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_stack(self):
    key = random.key(123)
    keys = random.split(key, 2)

    key_func = arr_func = partial(jnp.stack, axis=0)

    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_array(self):
    key = random.key(123)
    self.assertKeysEqual(key, jnp.array(key))
    self.assertKeysEqual(key, jnp.asarray(key))
    self.assertKeysEqual(key, jax.jit(jnp.array)(key))
    self.assertKeysEqual(key, jax.jit(jnp.asarray)(key))

  def test_array_user_dtype(self):
    key = random.key(123)
    self.assertKeysEqual(key, jnp.array(key, dtype=key.dtype))
    self.assertKeysEqual(key, jnp.asarray(key, dtype=key.dtype))

  @parameterized.parameters([
    (0,),
    (slice(1),),
    (np.array([0, 2]),),
    (np.array([False, True, True]),)
  ])
  def test_getitem(self, idx):
    keys = lambda: random.split(random.key(123), 3)
    key_func = arr_func = lambda x: x[idx]

    self.check_shape(key_func, keys())
    with jax.debug_key_reuse(False):
      self.check_against_reference(key_func, arr_func, keys())

  @parameterized.parameters([
    (0,),
    (slice(1),),
    (np.array([0, 2]),),
    (np.array([False, True, True]),)
  ])
  def test_gather(self, idx):
    keys = lambda: random.split(random.key(123), 3)
    key_func = arr_func = lambda key: key.at[idx].get()

    self.check_shape(key_func, keys())
    with jax.debug_key_reuse(False):
      self.check_against_reference(key_func, arr_func, keys())

  @jax.debug_key_reuse(False)
  def test_equality(self):
    key = random.key(123)
    key2 = random.key(456)

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
    key = lambda: random.key(123)
    keys = lambda: random.split(key(), 3)

    key_func = arr_func = lambda k1, k2: k1.at[idx].set(k2)

    self.check_shape(key_func, keys(), key())
    self.check_against_reference(key_func, arr_func, keys(), key())

  def test_errors(self):
    key = random.key(123)
    with self.assertRaisesRegex(TypeError, "add does not accept dtypes key<fry>, int.*"):
      jnp.add(key, 1)
    with self.assertRaisesRegex(TypeError, "add does not accept dtypes key<fry>, int.*"):
      key + 1
    with self.assertRaisesRegex(TypeError, "add does not accept dtype key<fry>"):
      jnp.add(key, key)
    with self.assertRaisesRegex(TypeError, "add does not accept dtype key<fry>"):
      key + key
    with self.assertRaisesRegex(TypeError, "neg does not accept dtype key<fry>"):
      jnp.negative(key)
    with self.assertRaisesRegex(TypeError, "neg does not accept dtype key<fry>"):
      -key
    with self.assertRaisesRegex(ValueError, "Cannot convert_element_type from key<fry> to int(32|64)"):
      lax.convert_element_type(key, int)
    with self.assertRaisesRegex(ValueError, "Cannot convert_element_type from int32 to key<fry>"):
      lax.convert_element_type(np.int32(0), key.dtype)

  def test_eval_shape(self):
    key = random.key(1701)
    shapedtype = jax.ShapeDtypeStruct(key.shape, key.dtype)
    out = jax.eval_shape(lambda x: x, shapedtype)
    self.assertEqual(out, shapedtype)

  def test_result_type(self):
    key = random.key(123456)
    self.assertEqual(jnp.result_type(key), key.dtype)

  @parameterized.parameters([
    (jnp.empty_like, ()),
    (jnp.zeros_like, ()),
    (jnp.ones_like, ()),
    (jnp.full_like, (100,)),
  ])
  def test_full_like(self, func, args):
    keys = random.split(random.key(789543))

    key_func = arr_func = lambda x: func(x, *args)
    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_full_like_with_key_fillvalue(self):
    keys = random.split(random.key(789543))
    fill_value = random.key(42)

    self.check_shape(jnp.full_like, keys, fill_value)
    self.check_against_reference(jnp.full_like, jnp.full_like, keys, fill_value)

  @parameterized.parameters([
    (jnp.empty, {}),
    (jnp.zeros, {}),
    (jnp.ones, {}),
    (jnp.full, {'fill_value': 100}),
  ])
  def test_full(self, func, kwds):
    keys = random.split(random.key(789543))

    key_func = arr_func = lambda x: func(x.shape, dtype=x.dtype, **kwds)
    self.check_shape(key_func, keys)
    self.check_against_reference(key_func, arr_func, keys)

  def test_full_with_key_fillvalue(self):
    keys = random.split(random.key(789543))
    fill_value = random.key(42)
    func = lambda x, val: jnp.full(x.shape, val, dtype=x.dtype)

    self.check_shape(func, keys, fill_value)
    self.check_against_reference(func, func, keys, fill_value)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
