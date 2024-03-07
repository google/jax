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

from absl.testing import absltest, parameterized
from functools import partial
import operator

import numpy as np
import jax
from jax import core
import jax.numpy as jnp
from jax._src import prng
from jax._src import random
from jax._src import test_util as jtu
from jax.errors import KeyReuseError
from jax.experimental.key_reuse._core import (
  assert_consumed, assert_unconsumed, consume, consume_p,
  Source, Sink, Forward, KeyReuseSignature)
from jax.experimental.key_reuse import _core

jax.config.parse_flags_with_absl()


key = jax.eval_shape(jax.random.key, 0)
key1D = jax.eval_shape(lambda key: key[None], key)


primitives_with_static_signatures = {
  consume_p: (consume, key),
  random.random_clone_p: (random.clone, key),
  prng.random_bits_p: (jax.random.bits, key),
  # prng.random_fold_in_p: (jax.random.fold_in, key, 2),
  prng.random_seed_p: (jax.random.key, 0),
  prng.random_split_p: (jax.random.split, key),
  prng.random_wrap_p: (jax.random.wrap_key_data, np.uint32([0, 0])),
  # prng.random_unwrap_p: (jax.random.key_data, key),
  jax.random.random_gamma_p: (jax.random.gamma, key, 1.0),
  jax.lax.broadcast_in_dim_p: (lambda key: key[None], key),
  jax.lax.copy_p: (jnp.array, key),
  jax.lax.convert_element_type_p: (lambda key: jnp.array(key, dtype=key.dtype), key),
  jax.lax.device_put_p: (jax.device_put, key),
  jax.lax.reshape_p: (lambda key: key.reshape((1,)), key),
  jax.lax.squeeze_p: (jnp.squeeze, key1D),
  jax.lax.dynamic_slice_p: (partial(jax.lax.dynamic_slice, slice_sizes=(1,)), key1D, (0,)),
  jax.lax.dynamic_update_slice_p: (jax.lax.dynamic_update_slice, key1D, key1D, (0,)),
}

# Primitive that is unknown to the key reuse machinery
unknown_p = core.Primitive("unknown")
unknown_p.def_abstract_eval(lambda x: x)
unknown_p.def_impl(lambda x: x)
def apply_unknown_primitive(key):
  return unknown_p.bind(key)


@jtu.with_config(
  jax_enable_custom_prng=False,
  jax_debug_key_reuse=False)
class KeyReuseUnitTestWithForwarding(jtu.JaxTestCase):
  def check_key_reuse(self, *args):
    return _core.check_key_reuse(*args)

  def test_assertions(self):
    key = jax.random.key(0)
    self.check_key_reuse(assert_unconsumed, key)
    with self.assertRaises(AssertionError):
      self.check_key_reuse(assert_consumed, key)

  def test_unknown(self):
    def f(key):
      assert_unconsumed(key)
      key2 = apply_unknown_primitive(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_consume(self):
    def f(key):
      assert_unconsumed(key)
      key2 = consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_random_clone(self):
    def f(key):
      assert_unconsumed(key)
      consume(key)
      assert_consumed(key)
      key2 = jax.random.clone(key)
      assert_unconsumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_seed(self):
    def f():
      key = jax.random.key(0)
      assert_unconsumed(key)
    self.check_key_reuse(f)

  def test_split(self):
    def f(key):
      assert_unconsumed(key)
      key2 = jax.random.split(key)
      assert_unconsumed(key2)
      assert_consumed(key)
    self.check_key_reuse(f, jax.random.key(0))

  def test_fold_in(self):
    def f(key):
      assert_unconsumed(key)
      key2 = jax.random.fold_in(key, 2)
      assert_unconsumed(key)
      assert_unconsumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_bits(self):
    def f(key):
      assert_unconsumed(key)
      bits = jax.random.bits(key, (), 'uint32')
      assert_consumed(key)
      return bits
    self.check_key_reuse(f, jax.random.key(0))

  def test_wrap(self):
    def f(key_data):
      key = jax.random.wrap_key_data(key_data)
      assert_unconsumed(key)
    self.check_key_reuse(f, jax.random.PRNGKey(0))

  def test_unwrap(self):
    def f(key):
      assert_unconsumed(key)
      key_data = jax.random.key_data(key)
      assert_unconsumed(key)
    self.check_key_reuse(f, jax.random.key(0))

  def test_gamma(self):
    def f(key):
      assert_unconsumed(key)
      values = jax.random.gamma(key, 1.0)
      assert_consumed(key)
      return values
    self.check_key_reuse(f, jax.random.key(0))

  def test_broadcast_in_dim(self):
    def f(key):
      assert_unconsumed(key)
      key2 = key[None]
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_copy(self):
    def f(key):
      assert_unconsumed(key)
      key2 = jnp.array(key, copy=True)
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_device_put(self):
    def f(key):
      assert_unconsumed(key)
      key2 = jax.device_put(key)
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_squeeze(self):
    def f(key):
      assert_unconsumed(key)
      key2 = jax.lax.squeeze(key, (0,))
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0)[None])

  def test_reshape(self):
    def f(key):
      assert_unconsumed(key)
      key2 = key.reshape(1, *key.shape)
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_concatenate(self):
    def f(key1, key2):
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      keys = jax.lax.concatenate([key1, key2], dimension=0)
      assert_consumed(key1)
      assert_consumed(key2)
      assert_unconsumed(keys)
    key1 = jax.random.split(jax.random.key(0))
    key2 = jax.random.split(jax.random.key(1))
    self.check_key_reuse(f, key1, key2)

  def test_slice(self):
    def f(keys):
      assert_unconsumed(keys)

      assert_unconsumed(keys[0])
      assert_consumed(keys, np.array([True, False]))

      assert_unconsumed(keys[1])
      assert_consumed(keys, np.array([True, True]))
    self.check_key_reuse(f, jax.random.split(jax.random.key(0)))

  @parameterized.parameters(operator.eq, operator.ne)
  def test_equality_checks(self, op):
    def f(key1, key2):
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      result = op(key1, key2)
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      return result
    self.check_key_reuse(f, jax.random.key(0), jax.random.key(1))

  def test_jit_can_consume_input(self):
    def f(key):
      assert_unconsumed(key)
      jax.jit(jax.random.bits)(key)
      assert_consumed(key)
    self.check_key_reuse(f, jax.random.key(0))

  def test_jit_can_return_consumed_output(self):
    def f():
      def g():
        key = jax.random.key(0)
        assert_unconsumed(key)
        bits = jax.random.bits(key)
        assert_consumed(key)
        return bits, key
      _, key = jax.jit(g)()
      assert_consumed(key)
    self.check_key_reuse(f)

  def test_jit_duplicate_inputs(self):
    def f(key):
      assert_unconsumed(key)
      def g(key1, key2):
        assert_unconsumed(key1)
        assert_unconsumed(key2)
        return jax.random.bits(key1)
      _ = jax.jit(g)(key, key)
      assert_consumed(key)
    self.check_key_reuse(f, jax.random.key(0))

  def test_jit_propagates_consumption_bit(self):
    def f(key):
      assert_unconsumed(key)
      g = jax.jit(lambda: key)
      key2 = g()
      assert_unconsumed(key)
      assert_unconsumed(key2)
      consume(key)
      assert_consumed(key)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_jit_duplicate_outputs(self):
    # TODO(jakevdp): implement this case
    def f(key):
      assert_unconsumed(key)
      def g(key):
        return key, key
      key1, key2 = jax.jit(g)(key)
      assert_unconsumed(key)
      assert_unconsumed(key1)
      assert_unconsumed(key2)
      _ = jax.random.bits(key1)
      assert_consumed(key)
      assert_consumed(key1)
      assert_consumed(key2)
    self.check_key_reuse(f, jax.random.key(0))

  def test_cond_both_consumed(self):
    @jax.jit
    def f(flag, key):
      assert_unconsumed(key)
      _ = jax.lax.cond(
        flag, jax.random.uniform, jax.random.normal, key)
      assert_consumed(key)
    self.check_key_reuse(f, True, jax.random.key(0))

  def test_cond_one_consumed(self):
    @jax.jit
    def f(flag, key):
      assert_unconsumed(key)
      _ = jax.lax.cond(
        flag, jax.random.uniform, lambda k: 1.0, key)
      assert_consumed(key)
    self.check_key_reuse(f, True, jax.random.key(0))

  def test_cond_neither_consumed(self):
    @jax.jit
    def f(flag, key):
      assert_unconsumed(key)
      _ = jax.lax.cond(
        flag, lambda k: 0.0, lambda k: 1.0, key)
      assert_unconsumed(key)
    self.check_key_reuse(f, True, jax.random.key(0))

  def test_simple_vmap(self):
    @jax.jit
    def f(seed):
      key = jax.random.key(seed)
      assert_unconsumed(key)
      result = jax.random.uniform(key)
      assert_consumed(key)
      return result
    self.check_key_reuse(f, 0)
    self.check_key_reuse(jax.vmap(f), jnp.arange(4))

  @parameterized.parameters(*primitives_with_static_signatures)
  def test_jaxpr_type_signature(self, primitive):
    func, *args = primitives_with_static_signatures[primitive]
    signature = _core.key_reuse_signatures[primitive]
    jaxpr = jax.make_jaxpr(func)(*args)
    self.assertEqual(signature, _core.jaxpr_type_signature(jaxpr.jaxpr))

  @parameterized.parameters(*primitives_with_static_signatures)
  def test_function_type_signature(self, primitive):
    func, *args = primitives_with_static_signatures[primitive]
    signature = _core.key_reuse_signatures[primitive]
    self.assertEqual(signature, _core.function_type_signature(func, *args))


@jtu.with_config(jax_debug_key_reuse=False)
class KeyReuseIntegrationTest(jtu.JaxTestCase):
  random_bits_error = "In random_bits, argument [0-9]+ is already consumed.*"
  random_split_error = "In random_split, argument [0-9]+ is already consumed.*"
  generic_error = ".*argument [0-9]+ is already consumed.*"
  pjit_error = "In pjit, argument 0 is already consumed."

  def check_key_reuse(self, f, *args):
    return _core.check_key_reuse(f, *args)

  def test_reuse(self):
    def f():
      key = jax.random.key(0)
      return jax.random.uniform(key) + jax.random.uniform(key)

    with self.assertRaisesRegex(KeyReuseError, self.pjit_error):
      self.check_key_reuse(f)

  def test_reuse_after_split(self):
    def f_good():
      key = jax.random.key(0)
      key1, key2 = jax.random.split(key)
      return jax.random.uniform(key1) + jax.random.uniform(key2)
    self.check_key_reuse(f_good)

    def f_bad():
      key = jax.random.key(0)
      _ = jax.random.split(key)
      return jax.random.uniform(key)

    with self.assertRaisesRegex(KeyReuseError, self.pjit_error):
      self.check_key_reuse(f_bad)

    def f_bad_2():
      key = jax.random.key(0)
      _ = jax.random.split(key)
      key1, _ = jax.random.split(key)
      return jax.random.uniform(key1)

    with self.assertRaisesRegex(KeyReuseError, self.random_split_error):
      self.check_key_reuse(f_bad_2)

  def test_repeated_fold_ins(self):
    # TODO(jakevdp): should we allow repeated fold-ins?
    def f():
      key = jax.random.key(0)
      keys = [jax.random.fold_in(key, i)
              for i in range(10)]
      return [jax.random.uniform(k) for k in keys]
    self.check_key_reuse(f)

  def test_reuse_after_fold_in(self):
    def f():
      key = jax.random.key(0)
      _ = jax.random.fold_in(key, 1)
      return jax.random.uniform(key)

    self.check_key_reuse(f)

  def test_reuse_after_broadcast(self):
    def f():
      key = jax.random.key(0)
      key2 = key[None]
      return jax.random.bits(key) + jax.vmap(jax.random.bits)(key2)

    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(f)

  def test_reuse_after_reshape(self):
    def f():
      key = jax.random.key(0)
      key2 = key.reshape((1,))
      return jax.random.bits(key) + jax.random.bits(key2.squeeze())

    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(f)

  def test_reuse_after_squeeze(self):
    def f():
      key = jax.random.split(jax.random.key(0), 1)
      key2 = jax.lax.squeeze(key, (0,))
      return jax.random.bits(key.squeeze()) + jax.random.bits(key2)

    with self.assertRaisesRegex(KeyReuseError, self.generic_error):
      self.check_key_reuse(f)

  def test_reuse_after_cond(self):
    def f_good(key, condition):
      return jax.lax.cond(condition, jax.random.uniform, jax.random.normal, key)
    key = jax.random.key(0)
    self.check_key_reuse(f_good, key, True)
    self.check_key_reuse(f_good, key, False)

    # Check where both branches consume the key
    def f_bad(key, condition):
      r1 = jax.lax.cond(condition, jax.random.uniform, jax.random.normal, key)
      return r1 + jax.random.uniform(key)

    with self.assertRaisesRegex(KeyReuseError, self.pjit_error):
      self.check_key_reuse(f_bad, key, True)

    # Check where only one branch consumes the key
    def f_bad_2(key, condition):
      r1 = jax.lax.cond(condition, jax.random.uniform, lambda key: 1.0, key)
      return r1 + jax.random.uniform(key)

    with self.assertRaisesRegex(KeyReuseError, self.pjit_error):
      self.check_key_reuse(f_bad_2, key, True)

  def test_simple_scan(self):
    def f_good(key):
      def body_fun(key, _):
        key, subkey = jax.random.split(key)
        return key, jax.random.bits(subkey)
      return jax.lax.scan(body_fun, key, xs=jnp.arange(10))
    self.check_key_reuse(f_good, jax.random.key(0))

  def test_scan_sink_on_consts(self):
    def f(key):
      def body_fun(carry, _):
        return carry, jax.random.uniform(key)
      return jax.lax.scan(body_fun, None, xs=jnp.arange(10))
    with self.assertRaisesRegex(KeyReuseError,  "scan body function leads to key reuse"):
      self.check_key_reuse(f, jax.random.key(0))

  def test_scan_reuse_in_body(self):
    def f_bad(key):
      def body_fun(key, _):
        return key, jax.random.bits(key)
      return jax.lax.scan(body_fun, key, xs=jnp.arange(10))
    with self.assertRaisesRegex(KeyReuseError, "scan body function leads to key reuse"):
      self.check_key_reuse(f_bad, jax.random.key(0))

  def test_scan_good_over_keys(self):
    def f_scan_over_keys(key):
      keys = jax.random.split(key, 5)
      return jax.lax.map(jax.random.bits, keys)
    self.check_key_reuse(f_scan_over_keys, jax.random.key(0))

  def test_scan_consume_one(self):
    def f_scan_over_keys(*keys):
      def body_func(keys, x):
        return tuple(jax.random.split(keys[0])), x
      return jax.lax.scan(body_func, keys, xs=jnp.arange(10))
    self.check_key_reuse(f_scan_over_keys, jax.random.key(0), jax.random.key(1))

  def test_vmap(self):
    @jax.vmap
    def f_good(seed):
      key = jax.random.key(seed)
      return jax.random.bits(key)
    self.check_key_reuse(f_good, jnp.arange(4))

    @jax.vmap
    def f_bad(seed):
      key = jax.random.key(0)
      return jax.random.bits(key) + jax.random.bits(key)
    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(f_bad, jnp.arange(4))

  def test_while_simple(self):
    def f(seed):
      key = jax.random.key(seed)
      def cond_fun(carry):
        return carry[1] < 10
      def body_fun(carry):
        key, subkey = jax.random.split(carry[0])
        return key, carry[1] + jax.random.uniform(subkey)
      return jax.lax.while_loop(cond_fun, body_fun, (key, 0))
    self.check_key_reuse(f, 0)

  def test_while_bad_cond(self):
    def f(seed):
      key = jax.random.key(seed)
      def cond_fun(carry):
        i, key = carry
        return i < jax.random.uniform(key)
      def body_fun(carry):
        i, key = carry
        return i + 1, key
      return jax.lax.while_loop(cond_fun, body_fun, (0, key))
    with self.assertRaisesRegex(KeyReuseError, "while_loop cond"):
      self.check_key_reuse(f, 0)

  def test_while_bad_body(self):
    def f(seed):
      key = jax.random.key(seed)
      def cond_fun(carry):
        key, i = carry
        return i < 5
      def body_fun(carry):
        key, i = carry
        return key, i + jax.random.randint(key, (), 1, 3)
      return jax.lax.while_loop(cond_fun, body_fun, (key, 0))
    with self.assertRaisesRegex(KeyReuseError, "while_loop body function leads to key reuse"):
      self.check_key_reuse(f, 0)

  def test_while_sink_on_body_consts(self):
    def f(seed):
      key = jax.random.key(seed)
      def cond_fun(i):
        return i < 5
      def body_fun(i):
        return i + jax.random.randint(key, (), 1, 3)
      return jax.lax.while_loop(cond_fun, body_fun, 0)
    with self.assertRaisesRegex(KeyReuseError, "while_loop body function leads to key reuse"):
      self.check_key_reuse(f, 0)

  def test_while_sink_on_cond_consts(self):
    def f(seed):
      key = jax.random.key(seed)
      def cond_fun(i):
        return i < jax.random.uniform(key)
      def body_fun(i):
        return i + 1
      return jax.lax.while_loop(cond_fun, body_fun, 0)
    with self.assertRaisesRegex(KeyReuseError, "while_loop cond function leads to key reuse"):
      self.check_key_reuse(f, 0)

  def test_pjit_consumed_input(self):
    @jax.jit
    def g(key, x):  # doesn't consume key
      return x

    def f(seed):
      key = jax.random.key(seed)
      x = jax.random.bits(key)
      return g(key, x)

    self.check_key_reuse(f, 0)

  @jax.numpy_dtype_promotion('standard')
  def test_remat(self):
    @jax.checkpoint
    def f_bad(x, key):
      return x * jax.random.bits(key) + jax.random.bits(key)

    @jax.checkpoint
    def f_good(x, key):
      return x * jax.random.bits(key)

    x = jnp.float32(1.0)
    key = jax.random.key(0)

    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(f_bad, x, key)

    with self.assertRaisesRegex(KeyReuseError, self.random_bits_error):
      self.check_key_reuse(jax.grad(f_bad), x, key)

    self.check_key_reuse(f_good, x, key)
    self.check_key_reuse(jax.grad(f_good), x, key)


@jtu.with_config(jax_debug_key_reuse=True)
class KeyReuseEagerTest(jtu.JaxTestCase):
  jit_msg = "Previously-consumed key passed to jit-compiled function at index 0"
  eager_bits_msg = "Previously-consumed key passed to random_bits at index 0"
  traced_bits_msg = "In random_bits, argument 0 is already consumed."

  def test_clone_eager(self):
    key = jax.random.key(0)
    key2 = jax.random.clone(key)
    self.assertIsNot(key, key2)

    _ = jax.random.uniform(key)
    self.assertTrue(key._consumed)
    self.assertFalse(key2._consumed)

  def test_simple_reuse_nojit(self):
    key = jax.random.key(0)
    with jax.disable_jit():
      _ = jax.random.bits(key)
      with self.assertRaisesRegex(KeyReuseError, self.eager_bits_msg):
        _ = jax.random.bits(key)

  def test_simple_key_reuse_jit(self):
    key = jax.random.key(0)
    _ = jax.jit(jax.random.bits)(key)
    with self.assertRaisesRegex(KeyReuseError, self.jit_msg):
      _ = jax.jit(jax.random.bits)(key)

  def test_closed_over_key_reuse_jit(self):
    key = jax.random.key(0)
    @jax.jit
    def f():
      return jax.random.uniform(key)
    _ = f()
    with self.assertRaisesRegex(KeyReuseError, self.jit_msg):
      _ = f()

  def test_key_reuse_within_jit(self):
    @jax.jit
    def f():
      key = jax.random.key(0)
      return jax.random.bits(key) + jax.random.bits(key)
    with self.assertRaisesRegex(KeyReuseError, self.traced_bits_msg):
      f()


class KeyReuseImplementationTest(jtu.JaxTestCase):

  def assertEquivalent(self, a, b):
    self.assertEqual(a, b)
    self.assertEqual(hash(a), hash(b))

  def assertNotEquivalent(self, a, b):
    self.assertNotEqual(a, b)
    self.assertNotEqual(hash(a), hash(b))

  def test_source_sink_immutability(self):
    mask = np.array([True, False])
    orig_mask_writeable = mask.flags.writeable

    sink = Sink(0, mask)
    source = Source(0, mask)

    self.assertFalse(sink.mask.flags.writeable)
    self.assertFalse(source.mask.flags.writeable)
    self.assertEqual(mask.flags.writeable, orig_mask_writeable)

    with self.assertRaises(ValueError):
      sink.idx = 1
    with self.assertRaises(ValueError):
      sink.mask = True
    with self.assertRaises(ValueError):
      source.idx = 1
    with self.assertRaises(ValueError):
      source.mask = True

  def test_source_sink_forward_equivalence_semantics(self):

    true_mask = np.array([True, True])
    false_mask = np.array([False, False])
    mixed_mask = np.array([True, False])

    self.assertEquivalent(Source(0), Source(0, True))
    self.assertEquivalent(Source(0, True), Source(0, true_mask))
    self.assertEquivalent(Source(0, False), Source(0, false_mask))
    self.assertEquivalent(Source(0, mixed_mask), Source(0, mixed_mask))
    self.assertNotEquivalent(Source(0), Source(1))
    self.assertNotEquivalent(Source(0), Source(0, False))
    self.assertNotEquivalent(Source(0), Source(0, mixed_mask))

    self.assertEquivalent(Sink(0), Sink(0, True))
    self.assertEquivalent(Sink(0, True), Sink(0, true_mask))
    self.assertEquivalent(Sink(0, False), Sink(0, false_mask))
    self.assertEquivalent(Sink(0, mixed_mask), Sink(0, mixed_mask))
    self.assertNotEquivalent(Sink(0), Sink(1))
    self.assertNotEquivalent(Sink(0), Sink(0, False))
    self.assertNotEquivalent(Sink(0), Sink(0, mixed_mask))

    self.assertNotEquivalent(Source(0), Sink(0))

    self.assertEquivalent(Forward(0, 1), Forward(0, 1))
    self.assertNotEquivalent(Forward(0, 1), Forward(1, 0))

  def test_signature_equality_semantics(self):
    self.assertEquivalent(
      KeyReuseSignature(Sink(0), Source(1), Forward(1, 0)),
      KeyReuseSignature(Forward(1, 0), Source(1), Sink(0)))
    self.assertEquivalent(
      KeyReuseSignature(), KeyReuseSignature())
    self.assertNotEquivalent(
      KeyReuseSignature(Source(0)), KeyReuseSignature(Sink(0)))

  def test_reprs(self):
    self.assertEqual(repr(Sink(0)), "Sink(0)")
    self.assertEqual(repr(Source(0)), "Source(0)")
    self.assertEqual(repr(Forward(0, 1)), "Forward(0, 1)")
    self.assertEqual(repr(KeyReuseSignature(Sink(1), Source(0))),
                     "KeyReuseSignature(Sink(1), Source(0))")
    self.assertEqual(repr(KeyReuseSignature(Sink(1), Sink(0))),
                     "KeyReuseSignature(Sink(0), Sink(1))")



@jtu.with_config(jax_enable_checks=False)
class KeyReuseGlobalFlagsTest(jtu.JaxTestCase):
  def test_key_reuse_flag(self):

    @jax.jit
    def f_bad(key):
      return jax.random.bits(key) + jax.random.bits(key)

    @jax.jit
    def f_good(key):
      return jax.random.bits(key)

    key = jax.random.key(0)

    with jax.debug_key_reuse(False):
      f_good(key)
      f_bad(key)  # No failure

    f_bad.clear_cache()
    f_good.clear_cache()

    with jax.debug_key_reuse(True):
      f_good(key)
      with self.assertRaisesRegex(KeyReuseError, "In random_bits.*"):
        f_bad(key)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
