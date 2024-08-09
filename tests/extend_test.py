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
from absl.testing import absltest
from absl.testing import parameterized

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
from jax._src import xla_bridge
from jax._src.interpreters import mlir

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
    self.assertIs(jex.backend.backends, xla_bridge.backends)
    self.assertIs(jex.backend.backend_xla_version, xla_bridge.backend_xla_version)
    self.assertIs(jex.backend.clear_backends, api.clear_backends)
    self.assertIs(jex.backend.get_backend, xla_bridge.get_backend)
    self.assertIs(jex.backend.register_backend_factory, xla_bridge.register_backend_factory)
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

  @parameterized.parameters([
      (True, mlir.ir.BoolAttr.get),
      (1, mlir.i64_attr),
      (5.0, lambda x: mlir.ir.FloatAttr.get(mlir.ir.F64Type.get(), x)),
      ("param", mlir.ir.StringAttr.get),
      (np.float32(0.5),
       lambda x: mlir.ir.FloatAttr.get(mlir.ir.F32Type.get(), x)),
  ])
  def testParams(self, param, expected_builder):
    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x, x, param=param)

    # Here we inspect the lowered IR to test that the parameter has been
    # serialized with the appropriate type.
    module = jax.jit(fun).lower(0.5).compiler_ir("stablehlo")
    for func in module.body.operations:
      for block in func.body.blocks:
        for op in block.operations:
          if op.OPERATION_NAME == "stablehlo.custom_call":
            config = op.attributes["mhlo.backend_config"]
            assert isinstance(config, mlir.ir.DictAttr)
            assert "param" in config
            with mlir.make_ir_context(), mlir.ir.Location.unknown():
              expected = expected_builder(param)
            assert type(config["param"]) == type(expected)
            assert expected.type.isinstance(config["param"].type)
            return
    self.fail("No custom_call found in the lowered IR")

  @jtu.sample_product(
    shape=[(1,), (4,), (5,)],
    dtype=(np.int32,),
  )
  @jtu.run_on_devices("gpu")
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
      vectorized=vectorized,
  )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
