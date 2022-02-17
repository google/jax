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

"""Tests for PRNG key re-use checking, enabled by `--jax_debug_prng_key_reuse`."""

import functools
import unittest

from absl.testing import absltest, parameterized
import jax
from jax._src import test_util as jtu
from jax import random
from jax.errors import KeyReuseError
import jax.numpy as jnp
import numpy as np

from jax.config import config
config.parse_flags_with_absl()

RANDOM_FUNCS = (
    ('bernoulli', random.bernoulli),
    ('beta', lambda k: random.beta(k, 1, 2)),
    ('categorical', lambda k: random.categorical(k, jnp.ones((2,)))),
    ('cauchy', random.cauchy),
    ('choice', lambda k: random.choice(k, np.ones(()))),
    ('dirichlet', lambda k: random.dirichlet(k, jnp.ones((1,)))),
    ('double_sided_maxwell', lambda k: random.double_sided_maxwell(k, 1, 2)),
    ('exponential', random.exponential),
    ('gamma', lambda k: random.gamma(k, 1)),
    ('gumbel', random.gumbel),
    ('laplace', random.laplace),
    ('logistic', random.logistic),
    ('maxwell', random.maxwell),
    ('multivariate_normal', lambda k: random.multivariate_normal(k, jnp.ones((1,)), jnp.ones((1, 1)))),
    ('normal', lambda k: random.normal(k, (), jnp.float32)),
    ('pareto', lambda k: random.pareto(k, 1)),
    ('permutation', lambda k: random.permutation(k, 1)),
    ('poisson', lambda k: random.poisson(k, 1, (), jnp.int32)),
    ('rademarcher', lambda k: random.rademacher(k, (), jnp.float32)),
    ('randint', lambda k: random.randint(k, (), 0, 10)),
    ('shuffle', lambda k: random.shuffle(k, jnp.ones(()))),
    ('split', random.split),
    ('t', lambda k: random.t(k, 1, (), jnp.float32)),
    ('truncated_normal', lambda k: random.truncated_normal(k, 0, 10)),
    ('uniform', random.uniform),
    ('weibull_min', lambda k: random.weibull_min(k, 1, 1, (), jnp.float32)),
)

JAX_FUNCS = (
    ('eval_shape', lambda f: functools.partial(jax.eval_shape, f)),
    ('grad', lambda f: jax.grad(f, allow_int=True)),
    ('jit', jax.jit),
    )


class KeyReuseTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._custom_prng_enabled = config.jax_enable_custom_prng
    config.update('jax_enable_custom_prng', True)
    self._debug_prng_keys_enabled = config.jax_debug_prng_key_reuse
    config.update('jax_debug_prng_key_reuse', True)

  def tearDown(self):
    super().tearDown()
    config.update('jax_enable_custom_prng', self._custom_prng_enabled)
    config.update('jax_debug_prng_key_reuse', self._debug_prng_keys_enabled)

  # ---- cases which error (correctly) ----
  @parameterized.named_parameters(*RANDOM_FUNCS)
  @jtu.ignore_warning(message="jax.random.shuffle is deprecated*")
  def test_double_use_errors(self, rand_function):
    key = random.PRNGKey(666)
    rand_function(key)
    with self.assertRaises(KeyReuseError):
      rand_function(key)

  @parameterized.named_parameters(*((rname+'_'+jname, rf, jf)
                                    for rname, rf in RANDOM_FUNCS
                                    for jname, jf in JAX_FUNCS))
  @jtu.ignore_warning(message="jax.random.shuffle is deprecated*")
  def test_double_use_while_tracing_errors(self, rand_function, jax_function):
    def double_use(key):
      rand_function(key)
      rand_function(key)
      return 0.

    key = random.PRNGKey(666)
    with self.assertRaises(KeyReuseError):
      jax_function(double_use)(key)

  def test_scan_double_use_in_one_iteration_errors(self):
    def body(key, _):
      random.split(key)
      random.split(key)
    key = random.PRNGKey(666)
    with self.assertRaises(KeyReuseError):
      jax.lax.scan(body, key, jnp.ones((1,)))

  def test_scan_closed_over_key_double_use_errors(self):
    key = random.PRNGKey(666)
    def body(carry, x):
      random.split(key)
      random.split(key)
      return carry, x
    with self.assertRaises(KeyReuseError):
      jax.lax.scan(body, None, jnp.ones((1,)))

  def test_vmap_double_use_in_one_batch_errors(self):
    key = random.PRNGKey(666)
    keys = jnp.broadcast_to(key, (2,))
    def f(k):
      random.split(k)
      random.split(k)

    with self.assertRaises(KeyReuseError):
      jax.vmap(f)(keys)

  def test_vmap_broadcast_key_double_use_errors(self):
    key = random.PRNGKey(666)
    def f(k, _):
      random.split(k)
      random.split(k)

    with self.assertRaises(KeyReuseError):
      jax.vmap(f, in_axes=(None, 0))(key, jnp.ones((2,)))

  def test_scan_double_use_zero_iteration_errors(self):
    # This case is rejected, even if this is a 0 iteration loop and the body
    # technically does not execute,
    def body(key, _):
      random.split(key)
      random.split(key)
    key = random.PRNGKey(666)
    with self.assertRaises(KeyReuseError):
      jax.lax.scan(body, key, None, length=0)

  def test_different_n_splits_errors(self):
    key = random.PRNGKey(666)
    random.split(key, num=2)
    with self.assertRaises(KeyReuseError):
      random.split(key, num=3)

  def test_rand_bits_and_split_errors(self):
    key = random.PRNGKey(666)
    random.randint(key, (), 0, 10)
    with self.assertRaises(KeyReuseError):
      random.split(key, num=2)

  # ---- does not error (correctly) ----
  @parameterized.named_parameters(*RANDOM_FUNCS)
  @jtu.ignore_warning(message="jax.random.shuffle is deprecated*")
  def test_single_use_while_tracing(self, rand_function):
    def single_use(key):
      rand_function(key)
      return 0.

    key = random.PRNGKey(666)
    jax.jit(single_use)(key)

  def test_cond_uses_on_both_branches(self):
    # This does not error because only one branch is taken.
    def f(x, key):
      return jax.lax.cond(x, random.split, random.split, key)
    key = random.PRNGKey(777)
    jax.jit(f)(True, key)

  def test_consuming_slice_different_index(self):
    # If a broadcast is a copy, this should not error.
    # This can be thought of as a scan/map which maps over a broadcasted key
    key = random.PRNGKey(666)
    keys = jnp.broadcast_to(key, (2,))
    random.split(keys[0])
    random.split(keys[1])

  def test_consuming_reference(self):
    key = random.PRNGKey(666)
    keys = jnp.broadcast_to(key, (2,))
    random.split(key)
    random.split(keys[0])

  def test_scan_reuses_inp(self):
    def body(carry, key):
      random.split(key)
      return carry, None
    key = random.PRNGKey(666)
    keys = jnp.broadcast_to(key, (2,))
    jax.lax.scan(body, None, keys)

  def test_vmap_broadcast_key_cross_batch(self):
    key = random.PRNGKey(666)
    def f(k, _):
      random.split(k)  # Splitting same key across 2 batches.

    # TODO(lenamartens): Is in_axes=(None,) an explicit broadcast?
    # Should this behave similar to scan_reusing_inp?
    jax.vmap(f, in_axes=(None, 0))(key, jnp.ones((2,)))

  # TODO(lenamartens) fix false positives and false negatives.
  # ---- false positives ----
  @unittest.expectedFailure
  def test_cond_uses_on_both_branches_closed(self):
    # TODO(lenamartens): Is this a false positive? Yes, if we allow re-use on
    # two branches of cond. However, maybe we always want to disallow closing
    # over keys?
    def f(x, key):
      return jax.lax.cond(x,
                          lambda _: random.split(key),  # key is closed over
                          lambda _: random.split(key),
                          operand=None)
    key = random.PRNGKey(777)
    f(True, key)

  # ---- false negatives ----
  @unittest.expectedFailure
  def test_single_use_while_tracing_call_twice_errors(self):
    def single_use(key):
      random.split(key)
      return 0.

    key = random.PRNGKey(666)
    jax.jit(single_use)(key)
    with self.assertRaises(KeyReuseError):
      jax.jit(single_use)(key)

  @unittest.expectedFailure
  def test_single_use_multiple_keys_errors(self):
    def single_use(key, unused_key):
      random.split(key)
      return unused_key

    key = random.PRNGKey(666)
    unused_key = random.PRNGKey(777)
    jax.jit(single_use)(key, unused_key)
    random.split(unused_key)  # does not error
    with self.assertRaises(KeyReuseError):
      random.split(key)

  @unittest.expectedFailure
  def test_scan_reuses_carry_errors(self):
    def body(key, x):
      random.split(key)
      return key, x
    key = random.PRNGKey(666)
    with self.assertRaises(KeyReuseError):
      jax.lax.scan(body, key, jnp.ones((2,)))

  @unittest.expectedFailure
  def test_while_loop_reuse_cond_body(self):
    def cond(x):
      i, key = x
      random.split(key)
      return i < 3
    def body(x):
      i, key = x
      new_key, _ = random.split(key)
      return i+1, new_key

    key = random.PRNGKey(666)
    with self.assertRaises(KeyReuseError):
      jax.lax.while_loop(cond, body, (0, key))

  @unittest.expectedFailure
  def test_consuming_slice_same_index(self):
    key = random.PRNGKey(666)
    keys = jnp.broadcast_to(key, (2,))
    # TODO(lenamartens): how to track across references? mutable cell/tensor?
    random.split(keys[0])
    with self.assertRaises(KeyReuseError):
      random.split(keys[0])

  @unittest.expectedFailure
  def test_consuming_while_iterating(self):
    key = random.PRNGKey(666)
    keys = jnp.broadcast_to(key, (2,))
    _ = [random.split(k) for k in keys]
    with self.assertRaises(KeyReuseError):
      _ = [random.split(k) for k in keys]

  @unittest.expectedFailure
  def test_jit_marks_key_as_consumed(self):
    key = random.PRNGKey(666)
    jax.jit(random.split)(key)
    with self.assertRaises(KeyReuseError):
      random.uniform(key, ())


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
