# Copyright 2023 The JAX Authors.
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

import os

import numpy as np
import unittest
from absl.testing import absltest, parameterized

import jax
from jax import lax
import jax.extend as jex
import jax.numpy as jnp

from jax._src import abstract_arrays
from jax._src import api
from jax._src import core
from jax._src import linear_util
from jax._src import prng
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib import xla_extension_version
from jax._src.lib.mlir import ir
from jax._src.extend import ffi

jax.config.parse_flags_with_absl()


class ExtendTest(jtu.JaxTestCase):

  def test_symbols(self):
    # Assume these are tested in random_test.py, only check equivalence
    self.assertIs(jex.random.seed_with_impl, prng.seed_with_impl)
    self.assertIs(jex.random.threefry2x32_p, prng.threefry2x32_p)
    self.assertIs(jex.random.threefry_2x32, prng.threefry_2x32)
    self.assertIs(jex.random.threefry_prng_impl, prng.threefry_prng_impl)
    self.assertIs(jex.random.rbg_prng_impl, prng.rbg_prng_impl)
    self.assertIs(jex.random.unsafe_rbg_prng_impl, prng.unsafe_rbg_prng_impl)

    # Assume these are tested elsewhere, only check equivalence
    self.assertIs(jex.backend.clear_backends, api.clear_backends)
    self.assertIs(jex.core.array_types, abstract_arrays.array_types)
    self.assertIs(jex.linear_util.StoreException, linear_util.StoreException)
    self.assertIs(jex.linear_util.WrappedFun, linear_util.WrappedFun)
    self.assertIs(jex.linear_util.cache, linear_util.cache)
    self.assertIs(jex.linear_util.merge_linear_aux, linear_util.merge_linear_aux)
    self.assertIs(jex.linear_util.transformation, linear_util.transformation)
    self.assertIs(jex.linear_util.transformation_with_aux, linear_util.transformation_with_aux)
    self.assertIs(jex.linear_util.wrap_init, linear_util.wrap_init)


class RandomTest(jtu.JaxTestCase):

  def test_key_make_with_custom_impl(self):
    shape = (4, 2, 7)

    def seed_rule(_):
      return jnp.ones(shape, dtype=jnp.dtype('uint32'))

    def no_rule(*args, **kwargs):
      assert False, 'unreachable'

    impl = jex.random.define_prng_impl(
        key_shape=shape, seed=seed_rule, split=no_rule, fold_in=no_rule,
        random_bits=no_rule)
    k = jax.random.key(42, impl=impl)
    self.assertEqual(k.shape, ())
    self.assertEqual(impl, jax.random.key_impl(k))

  def test_key_wrap_with_custom_impl(self):
    def no_rule(*args, **kwargs):
      assert False, 'unreachable'

    shape = (4, 2, 7)
    impl = jex.random.define_prng_impl(
        key_shape=shape, seed=no_rule, split=no_rule, fold_in=no_rule,
        random_bits=no_rule)
    data = jnp.ones((3, *shape), dtype=jnp.dtype('uint32'))
    k = jax.random.wrap_key_data(data, impl=impl)
    self.assertEqual(k.shape, (3,))
    self.assertEqual(impl, jax.random.key_impl(k))


class FfiTest(jtu.JaxTestCase):

  def testHeadersExist(self):
    base_dir = os.path.join(jex.ffi.include_dir(), "xla", "ffi", "api")
    for header in ["c_api.h", "api.h", "ffi.h"]:
      self.assertTrue(os.path.exists(os.path.join(base_dir, header)))

  @parameterized.parameters(
      [True, int(1), float(5.0),
       np.int32(-5), np.float32(0.5)])
  def testIrAttribute(self, value):
    with mlir.make_ir_context(), ir.Location.unknown():
      const = mlir.ir_constant(value)
      attr = ffi._ir_attribute(value)
      assert const.type.element_type == attr.type

  @parameterized.parameters([True, 1, 5.0, "param", np.float32(0.5)])
  def testParams(self, param):
    prim = core.Primitive("test_ffi")
    prim.def_abstract_eval(lambda *args, **kwargs: args[0])
    mlir.register_lowering(prim, jex.ffi.ffi_lowering("test_ffi"))

    # TODO(dfm): Currently testing that lowering works with different types of
    # parameters, but we should probably actually check the emitted HLO.
    func = jax.jit(lambda *args: prim.bind(*args, param=param))
    func.lower(jnp.linspace(0, 5, 10))

  @jtu.sample_product(
    shape=[(1,), (4,), (5,)],
    dtype=(np.int32,),
  )
  @jtu.run_on_devices("gpu")
  @unittest.skipIf(xla_extension_version < 272, "requires jaxlib 0.4.31")
  def testFfiCall(self, shape, dtype):
    pivots_size = shape[-1]
    permutation_size = 2 * pivots_size
    pivots = jnp.arange(permutation_size - 1, pivots_size - 1, -1, dtype=dtype)
    pivots = jnp.broadcast_to(pivots, shape)
    expected = lax.linalg.lu_pivots_to_permutation(pivots, permutation_size)
    actual = ffi_call_lu_pivots_to_permutation(pivots, permutation_size)
    self.assertArraysEqual(actual, expected)

  @jtu.sample_product(
      shape=[(1,), (4,), (5,)],
      dtype=(np.int32,),
      vectorized=(False, True),
  )
  @jtu.run_on_devices("gpu")
  @unittest.skipIf(xla_extension_version < 272, "requires jaxlib 0.4.31")
  def testFfiCallBatching(self, shape, dtype, vectorized):
    shape = (10,) + shape
    pivots_size = shape[-1]
    permutation_size = 2 * pivots_size
    pivots = jnp.arange(permutation_size - 1, pivots_size - 1, -1, dtype=dtype)
    pivots = jnp.broadcast_to(pivots, shape)
    expected = lax.linalg.lu_pivots_to_permutation(pivots, permutation_size)
    actual = jax.vmap(lambda x: ffi_call_lu_pivots_to_permutation(
        x, permutation_size, vectorized=vectorized))(pivots)
    self.assertArraysEqual(actual, expected)


# TODO(dfm): For now this test uses the `cu_lu_pivots_to_permutation`
# custom call target because that's the only one in jaxlib that uses the
# new FFI interface. Once more are available, consider using something that
# can be run on multiple platforms.
def ffi_call_lu_pivots_to_permutation(pivots, permutation_size, vectorized=True):
  return jex.ffi.ffi_call(
      "cu_lu_pivots_to_permutation",
      jax.ShapeDtypeStruct(
          shape=pivots.shape[:-1] + (permutation_size,),
          dtype=pivots.dtype,
      ),
      pivots,
      permutation_size=np.int32(permutation_size),
      vectorized=vectorized,
  )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
