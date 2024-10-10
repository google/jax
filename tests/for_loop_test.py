# Copyright 2022 The JAX Authors.
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

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import random
from jax._src import test_util as jtu
from jax._src.lax.control_flow import for_loop
import jax.numpy as jnp

jax.config.parse_flags_with_absl()

def remat_of_for_loop(nsteps, body, state, **kwargs):
  return jax.remat(lambda state: for_loop.for_loop(nsteps, body, state,
                                                   **kwargs))(state)

def nested_for_loop(nsteps, body, state, **kwargs):
  def outer_body(_, refs):
    def inner_body(i, _):
      body(i, refs)
      return
    for_loop.for_loop(nsteps, inner_body, ())
  return for_loop.for_loop(1, outer_body, state)

FOR_LOOP_IMPLS = [
    (for_loop.for_loop, 'for_loop'),
    (jax.jit(for_loop.for_loop, static_argnums=(0, 1)), 'jit_for_loop'),
    (remat_of_for_loop, 'remat_for_loop'),
    (nested_for_loop, 'nested_for_loop'),
    (partial(for_loop.for_loop, unroll=3), 'unrolled_for_loop'),
]


def _for_loop_impls(f):
  return parameterized.named_parameters(
      dict(testcase_name=impl_name, for_impl=for_impl)
      for for_impl, impl_name in FOR_LOOP_IMPLS
      )(f)


class ForLoopTest(jtu.JaxTestCase):

  @_for_loop_impls
  def test_for_loop_impl_trivial(self, for_impl):
    out = for_impl(5, lambda i, _: None, None)
    self.assertIsNone(out)

  @_for_loop_impls
  def test_for_loop_can_write_to_ref(self, for_impl):
    def body(_, x_ref):
      x_ref[()] = jnp.float32(1.)
    out = for_impl(1, body, jnp.float32(0.))
    self.assertEqual(out, 1.)

    def body2(i, x_ref):
      x_ref[()] = jnp.float32(i)
    out = for_impl(2, body2, jnp.float32(0.))
    self.assertEqual(out, 1.)

    def body3(i, x_ref):
      x_ref[()] = jnp.float32(i) * 2.
    out = for_impl(2, body3, jnp.float32(0.))
    self.assertEqual(out, 2.)

  @_for_loop_impls
  def test_for_loop_can_write_to_multiple_refs(self, for_impl):
    def body(_, refs):
      x_ref, y_ref = refs
      x_ref[()] = jnp.float32(1.)
      y_ref[()] = jnp.float32(2.)
    x, y = for_impl(1, body, (jnp.float32(0.), jnp.float32(0.)))
    self.assertEqual(x, 1.)
    self.assertEqual(y, 2.)

  @_for_loop_impls
  def test_for_loop_can_read_from_ref(self, for_impl):
    def body(_, x_ref):
      x_ref[()]  # pylint: disable=pointless-statement
    x = for_impl(1, body, jnp.float32(0.))
    self.assertEqual(x, 0.)

  @_for_loop_impls
  def test_for_loop_can_read_from_and_write_to_ref(self, for_impl):
    def body(_, x_ref):
      x = x_ref[()]
      x_ref[()] = x + jnp.float32(1.)
    x = for_impl(5, body, jnp.float32(0.))
    self.assertEqual(x, 5.)

  @_for_loop_impls
  def test_for_loop_can_read_from_and_write_to_refs(self, for_impl):
    def body2(_, refs):
      x_ref, y_ref = refs
      x = x_ref[()]
      y_ref[()] = x + 1.
      x_ref[()] = x + 1.
    x, y = for_impl(5, body2, (0., 0.))
    self.assertEqual(x, 5.)
    self.assertEqual(y, 5.)

  @_for_loop_impls
  def test_for_loop_can_read_from_and_write_to_ref_slice(self, for_impl):
    def body(i, x_ref):
      x = x_ref[i]
      x_ref[i] = x + jnp.float32(1.)
    x = for_impl(4, body, jnp.ones(4, jnp.float32))
    np.testing.assert_allclose(x, 2 * jnp.ones(4, jnp.float32))

    def body2(i, x_ref):
      x = x_ref[i, 0]
      x_ref[i, 1] = x + x_ref[i, 1]
    x = for_impl(4, body2, jnp.arange(8.).reshape((4, 2)))
    np.testing.assert_allclose(
        x, jnp.array([[0., 1.], [2., 5.], [4., 9.], [6., 13.]]))

  @_for_loop_impls
  @jax.legacy_prng_key('allow')
  def test_for_loop_can_implement_cumsum(self, for_impl):
    def cumsum(x):
      def body(i, refs):
        x_ref, accum_ref = refs
        accum_ref[i + 1] = accum_ref[i] + x_ref[i]
      accum = jnp.zeros(x.shape[0] + 1, x.dtype)
      _, accum_out = for_impl(x.shape[0], body, (x, accum))
      return accum_out[1:]

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8,))
    np.testing.assert_allclose(cumsum(x), jnp.cumsum(x), rtol=1e-6)

def for_body_swap(i, refs):
  a_ref, b_ref = refs
  a, b = a_ref[i], b_ref[i]
  b_ref[i] = a
  a_ref[i] = b

def swap_ref(a, b):
  return b, a

def for_body_swap_swap(i, refs):
  for_body_swap(i, refs)
  for_body_swap(i, refs)

swap_swap_ref = lambda a, b: (a, b)

def for_body_sincos(i, refs):
  a_ref, b_ref = refs
  a = a_ref[i]
  b_ref[i] = jnp.sin(jnp.cos(a))

sincos_ref = lambda x, y: (x, jnp.sin(jnp.cos(x)))

def for_body_sincostan(i, refs):
  a_ref, b_ref = refs
  a = a_ref[i]
  b_ref[i] = jnp.tan(jnp.sin(jnp.cos(a)))

sincostan_ref = lambda x, y: (x, jnp.tan(jnp.sin(jnp.cos(x))))

def for_body_accum(i, refs):
  x_ref, accum_ref = refs
  accum_ref[i + 1] = accum_ref[i] + x_ref[i]

def accum_ref(x, accum):
  for i in range(x.shape[0] - 1):
    accum = accum.at[i + 1].set(accum[i] + x[i])
  return x, accum

def for_body_sin_sq(i, refs):
  x_ref, y_ref = refs
  x = x_ref[i]
  y = x
  y_ref[i] = y
  y = y_ref[i]
  y_ref[i] = jnp.sin(y * y)

sin_sq_ref = lambda x, y: (x, jnp.sin(x * x))

def for_body_reverse(i, refs):
  x_ref, y_ref = refs
  j = y_ref.shape[0] - i - 1
  y_ref[i] = x_ref[j]

reverse_ref = lambda x, y: (x, x[::-1])

def for_body_noop(i, refs):
  pass
noop_ref = lambda x, y: (x, y)
for_reference = for_loop.discharged_for_loop


class ForLoopTransformationTest(jtu.JaxTestCase):

  @jtu.sample_product(
    [dict(for_body_name=for_body_name, f=for_body, ref=ref,
          body_shapes=body_shapes, n=nsteps)
      for for_body_name, for_body, ref, body_shapes, nsteps in [
        ("swap", for_body_swap, swap_ref, [(4,), (4,)], 4),
        ("swap_swap", for_body_swap_swap, swap_swap_ref, [(4,), (4,)], 4),
        ("sincos", for_body_sincos, sincos_ref, [(4,), (4,)], 4),
        ("sincostan", for_body_sincostan, sincostan_ref, [(4,), (4,)], 4),
        ("accum", for_body_accum, accum_ref, [(4,), (4,)], 3),
        ("sin_sq", for_body_sin_sq, sin_sq_ref, [(4,), (4,)], 4),
        ("reverse", for_body_reverse, reverse_ref, [(4,), (4,)], 4),
      ]
    ],
    [dict(for_impl=for_impl, impl_name=impl_name)
     for for_impl, impl_name in FOR_LOOP_IMPLS],
  )
  @jtu.skip_on_devices("gpu", "cpu")  # TODO(mattjj,sharadmv, dougalm): timeouts?
  def test_for_jvp(self, f, ref, body_shapes, n, for_impl, for_body_name,
                   impl_name):
    for_ = for_impl
    rng = self.rng()

    args = [rng.randn(*s) for s in body_shapes]

    tol = {np.float64: 1e-12, np.float32: 1e-4}
    ans = jax.jvp(     lambda *args: for_(         n, f, args), args, args)
    ans_discharged = jax.jvp(lambda *args: for_reference(n, f, args), args, args)
    expected = jax.jvp(ref, args, args)
    self.assertAllClose(ans, ans_discharged, check_dtypes=True, rtol=tol, atol=tol)
    self.assertAllClose(ans, expected, check_dtypes=True, rtol=tol, atol=tol)
    jtu.check_grads(partial(for_, n, f), (args,), order=2, modes=["fwd"])

  @jtu.sample_product(
    [dict(for_body_name=for_body_name, f=for_body, ref=ref,
          body_shapes=body_shapes, n=nsteps)
      for for_body_name, for_body, ref, body_shapes, nsteps in [
        ("swap", for_body_swap, swap_ref, [(4,), (4,)], 4),
        ("swap_swap", for_body_swap_swap, swap_swap_ref, [(4,), (4,)], 4),
        ("sincos", for_body_sincos, sincos_ref, [(4,), (4,)], 4),
        ("sincostan", for_body_sincostan, sincostan_ref, [(4,), (4,)], 4),
        ("accum", for_body_accum, accum_ref, [(4,), (4,)], 3),
        ("sin_sq", for_body_sin_sq, sin_sq_ref, [(4,), (4,)], 4),
        ("reverse", for_body_reverse, reverse_ref, [(4,), (4,)], 4),
      ]
    ],
    [dict(for_impl=for_impl, impl_name=impl_name)
     for for_impl, impl_name in FOR_LOOP_IMPLS],
  )
  @jtu.skip_on_devices("gpu", "cpu")  # TODO(mattjj,sharadmv, dougalm): timeouts?
  def test_for_linearize(self, f, ref, body_shapes, n, for_impl, for_body_name,
                         impl_name):
    for_ = for_impl
    rng = self.rng()

    args = [rng.randn(*s) for s in body_shapes]

    tol = {np.float64: 1e-12, np.float32: 1e-4}
    ans = jax.linearize(lambda *args: for_(         n, f, args), *args)[1](*args)
    ans_discharged = jax.linearize(lambda *args: for_reference(n, f, args),
                                   *args)[1](*args)
    expected = jax.linearize(ref, *args)[1](*args)
    self.assertAllClose(ans, ans_discharged, check_dtypes=True, rtol=tol, atol=tol)
    self.assertAllClose(ans, expected, check_dtypes=True, rtol=tol, atol=tol)

  def test_for_loop_invar(self):
    def f(x):
      s = jnp.ones((2, 32), x.dtype)
      def body(i, refs):
        x_ref, y_ref = refs
        y_ref[i] = s * x_ref[i] * jnp.cos(s)
        # We should save `s` and `jnp.cos(s)` as residuals and not broadcast
        # them.
      return for_loop.for_loop(x.shape[0], body, (x, jnp.zeros_like(x)))
    _, f_vjp = jax.linearize(f, jnp.ones((5, 2, 32)))
    jaxpr = jax.make_jaxpr(f_vjp)(jnp.ones((5, 2, 32)))
    consts = [v.aval for v in jaxpr.jaxpr.constvars
              if v.aval.shape == (2, 32)]
    self.assertLen(consts, 2)

    def loss(A):
      def step(x, _):
        return jnp.matmul(A, x), None
      init_x = jnp.zeros(A.shape[-1:])
      last_x, _ = for_loop.scan(step, init_x, jnp.arange(10))
      return jnp.sum(last_x)

    A = jnp.zeros((3, 3))
    # The second DUS was unnecessarily replicating A across time.
    # We check XLA because _scan_impl is "underneath" the jaxpr language.
    s = jax.jit(jax.grad(loss)).lower(A).as_text('hlo')
    assert s.count("dynamic-update-slice(") < 2

  @_for_loop_impls
  def test_for_loop_fixpoint_correctly_identifies_loop_varying_residuals(
    self, for_impl):

    def body(i, refs):
      a_ref, b_ref, c_ref = refs
      a = a_ref[i]
      b = b_ref[()]
      x = jnp.sin(a)
      b_ref[()] = jnp.sin(b * x)
      c_ref[i] = x * b
    def f(a, b):
      c = jnp.zeros_like(a)
      _, b, c = for_impl(5, body, (a, b, c))
      return b, c
    a = jnp.arange(5.) + 1.
    b = jnp.ones_like(a[0])
    _, f_lin = jax.linearize(f, a, b)
    expected_tangents = f_lin(a, b)
    _, actual_tangents = jax.jvp(f, (a, b), (a, b))
    np.testing.assert_allclose(actual_tangents[0], expected_tangents[0],
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_tangents[1], expected_tangents[1],
                               rtol=1e-6, atol=1e-6)

    def body2(_, refs):
      # Here we use `i_ref` as a loop counter
      a_ref, b_ref, c_ref, i_ref = refs
      i = i_ref[()]
      a = a_ref[i]
      b = b_ref[()]
      x = jnp.sin(a)
      b_ref[()] = jnp.sin(b * x)
      c_ref[i] = x * b
      i_ref[()] = i + 1

    def g(a, b):
      c = jnp.zeros_like(a)
      _, b, c, _ = for_impl(5, body2, (a, b, c, 0))
      return b, c
    a = jnp.arange(5.) + 1.
    b = jnp.ones_like(a[0])
    _, g_lin = jax.linearize(f, a, b)
    expected_tangents = g_lin(a, b)
    _, actual_tangents = jax.jvp(g, (a, b), (a, b))
    np.testing.assert_allclose(actual_tangents[0], expected_tangents[0])
    np.testing.assert_allclose(actual_tangents[1], expected_tangents[1],
                               rtol=1e-6)

  @jtu.sample_product(
    [dict(for_body_name=for_body_name, f=for_body, ref=ref,
          body_shapes=body_shapes, n=nsteps)
      for for_body_name, for_body, ref, body_shapes, nsteps in [
        ("noop", for_body_noop, noop_ref, [(4,), (4,)], 4),
        ("swap", for_body_swap, swap_ref, [(4,), (4,)], 4),
        ("swap_swap", for_body_swap_swap, swap_swap_ref, [(4,), (4,)], 4),
        ("sincos", for_body_sincos, sincos_ref, [(4,), (4,)], 4),
        ("sincostan", for_body_sincostan, sincostan_ref, [(4,), (4,)], 4),
        ("accum", for_body_accum, accum_ref, [(4,), (4,)], 3),
        ("sin_sq", for_body_sin_sq, sin_sq_ref, [(4,), (4,)], 4),
        ("reverse", for_body_reverse, reverse_ref, [(4,), (4,)], 4),
      ]
    ],
    [dict(for_impl=for_impl, impl_name=impl_name)
     for for_impl, impl_name in FOR_LOOP_IMPLS],
  )
  @jtu.skip_on_devices("gpu", "cpu")  # TODO(mattjj,sharadmv, dougalm): timeouts?
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_for_grad(self, f, ref, body_shapes, n, for_impl, for_body_name,
                    impl_name):
    for_ = for_impl
    rng = self.rng()

    args = [rng.randn(*s) for s in body_shapes]

    tol = {np.float64: 1e-12, np.float32: 1e-4}
    ans = jax.grad(lambda args: for_(         n, f, args)[1].sum())(args)
    ans_discharged = jax.grad(
        lambda args: for_reference(n, f, args)[1].sum())(args)
    expected = jax.grad(lambda args: ref(*args)[1].sum())(args)
    self.assertAllClose(ans, ans_discharged, check_dtypes=True, rtol=tol,
                        atol=tol)
    self.assertAllClose(ans, expected, check_dtypes=True, rtol=tol, atol=tol)
    jtu.check_grads(lambda *args: for_(n, f, args)[1].sum(), args, order=2,
                    rtol=7e-3, atol=1e-2)

  @jtu.skip_on_devices("gpu", "cpu")  # TODO(mattjj,sharadmv, dougalm): timeouts?
  @jax.legacy_prng_key('allow')
  def test_grad_of_triple_nested_for_loop(self):

    func = lambda x: jnp.sin(x) + 1.

    @jax.jit
    def f(x):
      out = jnp.zeros_like(x)
      def body(i, j, k, refs):
        x_ref, out_ref = refs
        y = func(x_ref[i, j, k])
        out_ref[i, j, k] += y
      return for_loop.for_loop(x.shape, body, (x, out))[1].sum()

    x = random.normal(random.PRNGKey(0), (5, 4, 3))
    ref = lambda x: jax.vmap(jax.vmap(jax.vmap(func)))(x).sum()
    self.assertAllClose(f(x), ref(x))
    jtu.check_grads(f, (x,), order=2, atol=0.1, rtol=0.1)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
