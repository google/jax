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

"""Tests for --debug_nans."""

from absl.testing import absltest

import jax
import numpy as np
from unittest import SkipTest

from jax._src import api
from jax._src import test_util as jtu
from jax import numpy as jnp
from jax.experimental import pjit
from jax._src.maps import xmap

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_debug_nans=True)
class DebugNaNsTest(jtu.JaxTestCase):

  def testSinc(self):
    # Regression test for #6936
    self.assertEqual(jnp.sinc(0.0), 1.0)

  def testSingleResultPrimitiveNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jnp.tanh(A)
    ans.block_until_ready()

  def testMultipleResultPrimitiveNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans, _ = jnp.linalg.eigh(A)
    ans.block_until_ready()

  def testJitComputationNoNaN(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jax.jit(jnp.tanh)(A)
    ans.block_until_ready()

  def testJitComputationNaN(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = jax.jit(lambda x: 0. / x)(A)
      ans.block_until_ready()

  @jax.debug_nans(False)
  def testJitComputationNaNContextManager(self):
    A = jnp.array(0.)
    f = jax.jit(lambda x: 0. / x)
    ans = f(A)
    ans = f(A)
    with self.assertRaises(FloatingPointError):
      with jax.debug_nans(True):
        ans = f(A)
      ans.block_until_ready()

  def testSingleResultPrimitiveNaN(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = 0. / A
      ans.block_until_ready()

  @jtu.sample_product(jit=jtu.JIT_IMPLEMENTATION)
  def testCallDeoptimized(self, jit):
    @jit
    def f(x):
      return jax.lax.cond(
          x == 1, lambda _: np.nan, lambda _: 2., operand=None)

    # This makes sure, when using the C++ jit, that the Python code has been
    # run to compile, and the next call won't go through `cache_miss`.
    f(2)
    # 'cond' not 'xla_call'
    msg = r"invalid value \(nan\) encountered in .*cond.*"
    with self.assertRaisesRegex(FloatingPointError, msg):
      f(1)

  def testPmap(self):
    pmap_funcs = [api._cpp_pmap]

    for pmap in pmap_funcs:
      f = pmap(lambda x: 0. / x)
      # For the Cpp pmap, the first execution always goes through Python.
      f(jnp.array([1.]))

      with self.assertRaisesRegex(
          FloatingPointError,
          r"invalid value \(nan\) encountered in parallel computation"):
        ans = f(jnp.array([0.]))
        ans.block_until_ready()

      if jax.device_count() >= 2:
        with self.assertRaisesRegex(
            FloatingPointError,
            r"invalid value \(nan\) encountered in parallel computation"):
          ans = f(jnp.array([1., 0.]))
          ans.block_until_ready()

  def testPmapNoNaN(self):
    ans = jax.pmap(lambda x: 0. / x)(jnp.array([1.]))
    ans.block_until_ready()

  @jtu.ignore_warning(message=".*is an experimental.*")
  def testXmap(self):

    f = xmap(
        lambda x: 0. / x,
        in_axes=["i"],
        out_axes=["i"],
        axis_resources={"i": "x"})

    with jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ('x',)):
      with self.assertRaisesRegex(
          FloatingPointError,
          r"invalid value \(nan\) encountered in xmap"):
        ans = f(jnp.array([0.]))
        ans.block_until_ready()

    if jax.device_count() >= 2:
      with jax.sharding.Mesh(np.array(jax.local_devices()[:2]), ('x',)):
        with self.assertRaises(FloatingPointError):
          ans = f(jnp.array([1., 0.]))
          ans.block_until_ready()

  @jtu.ignore_warning(message=".*is an experimental.*")
  def testPjit(self):
    if jax.device_count() < 2:
      raise SkipTest("test requires >=2 devices")

    p = jax.sharding.PartitionSpec('x')
    f = pjit.pjit(lambda x: 0. / x, in_shardings=p, out_shardings=p)

    with jax.sharding.Mesh(np.array(jax.local_devices()[:2]), ('x',)):
      with self.assertRaises(FloatingPointError):
        ans = f(jnp.array([0., 1.]))
        ans.block_until_ready()

  def testDebugNansJitWithDonation(self):
    # https://github.com/google/jax/issues/12514
    a = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = jax.jit(lambda x: 0. / x, donate_argnums=(0,))(a)
      ans.block_until_ready()

  def testDebugNansPmapWithDonation(self):
    a = jnp.zeros((1,))
    with self.assertRaises(FloatingPointError):
      ans = jax.pmap(lambda x: 0. / x, donate_argnums=(0,))(a)
      ans.block_until_ready()

  @jtu.ignore_warning(message=".*is an experimental.*")
  def testDebugNansPjitWithDonation(self):
    if jax.device_count() < 2:
      raise SkipTest("test requires >=2 devices")

    p = jax.sharding.PartitionSpec('x')
    f = pjit.pjit(lambda x: 0. / x,
                  in_shardings=p,
                  out_shardings=p,
                  donate_argnums=(0,))

    with jax.sharding.Mesh(np.array(jax.local_devices()[:2]), ('x',)):
      with self.assertRaises(FloatingPointError):
        ans = f(jnp.array([0., 1.]))
        ans.block_until_ready()

  def testDebugNansZeroDiv(self):
    inp = jnp.zeros(())
    def f(x, y):
      return x / y

    with self.assertRaisesRegex(
        FloatingPointError,
        r"invalid value \(nan\) encountered in jit\(true_divide\)"):
      f(inp, inp)

    # TODO(yashkatariya): Fix this and make true_divide appear in the name again.
    # Instead of `f` showing up in the error, the name should be of the
    # primitive (true_divide) in this case.
    with self.assertRaisesRegex(
        FloatingPointError,
        r"invalid value \(nan\) encountered in jit\(f\)"):
      jax.jit(f)(inp, inp)


@jtu.with_config(jax_debug_infs=True)
class DebugInfsTest(jtu.JaxTestCase):

  def testSingleResultPrimitiveNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jnp.tanh(A)
    ans.block_until_ready()

  def testMultipleResultPrimitiveNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans, _ = jnp.linalg.eigh(A)
    ans.block_until_ready()

  def testJitComputationNoInf(self):
    A = jnp.array([[1., 2.], [2., 3.]])
    ans = jax.jit(jnp.tanh)(A)
    ans.block_until_ready()

  def testSingleResultPrimitiveInf(self):
    A = jnp.array(0.)
    with self.assertRaises(FloatingPointError):
      ans = 1. / A
      ans.block_until_ready()

  @jtu.sample_product(jit=jtu.JIT_IMPLEMENTATION)
  def testCallDeoptimized(self, jit):
    @jit
    def f(x):
      return jax.lax.cond(
          x == 1, lambda _: np.inf, lambda _: 2., operand=None)

    # This makes sure, when using the C++ jit, that the Python code has been
    # run to compile, and the next call won't go through `cache_miss`.
    f(2)
    # 'cond' not 'xla_call'
    msg = r"invalid value \(inf\) encountered in .*cond.*"
    with self.assertRaisesRegex(FloatingPointError, msg):
      f(1)

  def testDebugNansDoesntCorruptCaches(self):
    # https://github.com/google/jax/issues/6614
    @jax.jit
    def f(x):
      return jnp.divide(x, x)

    for _ in range(2):
      try:
        with jax.debug_nans(True):
          jax.grad(f)(0.)
      except FloatingPointError:
        pass

  def testDebugNansDoesntReturnDeoptimizedResult(self):
    @jax.jit
    def f(x):
      y = x + 2  # avoid trivial dispatch path by adding some eqn
      return jnp.nan, y

    with self.assertRaisesRegex(FloatingPointError, "de-optimized"):
      with jax.debug_nans(True):
        f(3)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
