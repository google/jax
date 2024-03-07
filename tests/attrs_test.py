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

from __future__ import annotations

from dataclasses import dataclass

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
import jax.numpy as jnp

from jax._src import config
from jax._src import test_util as jtu
from jax._src.util import safe_zip, safe_map

from jax.experimental import attrs
from jax.experimental.attrs import jax_setattr, jax_getattr

config.parse_flags_with_absl()

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

@dataclass
class Thing:
  x: float
  __hash__ = object.__hash__
  __eq__ = object.__eq__

attrs.register(Thing)  # enables passing as arg into jitted function

class AttrsTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_jit_basic(self, jit: bool):
    thing = Thing(1.0)

    def double_it() -> None:
      cur_x = jax_getattr(thing, "x")
      jax_setattr(thing, "x", cur_x * 2)

    if jit:
      double_it = jax.jit(double_it)

    self.assertEqual(thing.x, 1.0)
    double_it()
    self.assertEqual(thing.x, 2.0)
    double_it()
    self.assertEqual(thing.x, 4.0)
    double_it()
    self.assertEqual(thing.x, 8.0)
    double_it()
    self.assertEqual(thing.x, 16.0)

  @parameterized.parameters([True, False])
  def test_jit_basic_tree(self, jit: bool):
    thing = Thing((1.0, 2.0))

    def double_it() -> None:
      (cur_x, cur_y) = jax_getattr(thing, "x")
      jax_setattr(thing, "x", (cur_x * 2, cur_y * 2))

    if jit:
      double_it = jax.jit(double_it)

    self.assertEqual(thing.x, (1.0, 2.0))
    double_it()
    self.assertEqual(thing.x, (2.0, 4.0))
    double_it()
    self.assertEqual(thing.x, (4.0, 8.0))
    double_it()
    self.assertEqual(thing.x, (8.0, 16.0))
    double_it()
    self.assertEqual(thing.x, (16.0, 32.0))

  @parameterized.parameters([True, False])
  def test_jit_basic_tree_changes(self, jit: bool):
    thing = Thing(None)
    count = 0

    def double_it() -> None:
      nonlocal count
      count += 1
      maybe_x = jax_getattr(thing, "x")
      x = 1.0 if maybe_x is None else maybe_x
      jax_setattr(thing, "x", 2 * x)

    if jit:
      double_it = jax.jit(double_it)

    self.assertEqual(thing.x, None)
    double_it()
    self.assertEqual(thing.x, 2.0)
    self.assertEqual(count, 1)
    double_it()
    self.assertEqual(thing.x, 4.0)
    self.assertEqual(count, 2)
    double_it()
    self.assertEqual(thing.x, 8.0)
    self.assertEqual(count, 2 + (not jit))

  def test_jit_basic_tree_changes_multiple(self):
    thing1 = Thing(None)
    thing2 = Thing(0)
    count = 0

    @jax.jit
    def double_it() -> None:
      nonlocal count
      count += 1

      x1 = jax_getattr(thing1, "x")
      if x1 is None:
        jax_setattr(thing1, 'x', (None,))
      elif isinstance(x1, tuple):
        # depend on a new value
        jax_setattr(thing1, 'x', jax_getattr(thing2, 'x') + 1)
      else:
        jax_setattr(thing2, 'x', jax_getattr(thing1, 'x'))
        jax_setattr(thing1, 'x', None)

    self.assertEqual(thing1.x, None)
    self.assertEqual(thing2.x, 0)
    double_it()
    self.assertEqual(thing1.x, (None,))
    self.assertEqual(thing2.x, 0)
    self.assertEqual(count, 1)
    double_it()
    self.assertEqual(thing1.x, 1)
    self.assertEqual(thing2.x, 0)
    self.assertEqual(count, 2)
    double_it()
    self.assertEqual(thing1.x, None)
    self.assertEqual(thing2.x, 1)
    self.assertEqual(count, 3)
    double_it()
    self.assertEqual(thing1.x, (None,))
    self.assertEqual(thing2.x, 1)
    self.assertEqual(count, 3)
    double_it()
    self.assertEqual(thing1.x, 2)
    self.assertEqual(thing2.x, 1)
    self.assertEqual(count, 3)
    double_it()
    self.assertEqual(thing1.x, None)
    self.assertEqual(thing2.x, 2)
    self.assertEqual(count, 3)

  def test_jit_nesting_basic(self):
    thing = Thing(1.0)

    @jax.jit
    @jax.jit
    def double_it() -> None:
      cur_x = jax_getattr(thing, "x")
      jax_setattr(thing, "x", cur_x * 2)

    self.assertEqual(thing.x, 1.0)
    double_it()
    self.assertEqual(thing.x, 2.0)
    double_it()
    self.assertEqual(thing.x, 4.0)
    double_it()
    self.assertEqual(thing.x, 8.0)
    double_it()
    self.assertEqual(thing.x, 16.0)

  def test_jit_consts_and_args(self):
    thing = Thing(1.0)

    @jax.jit
    def double_it(y) -> None:
      cur_x = jax_getattr(thing, "x")
      jax_setattr(thing, "x", cur_x * 2)
      return jnp.cos(np.arange(3.) * cur_x * y)

    self.assertEqual(thing.x, 1.0)
    double_it(2.)
    self.assertEqual(thing.x, 2.0)
    double_it(2.)
    self.assertEqual(thing.x, 4.0)
    double_it(2.)
    self.assertEqual(thing.x, 8.0)
    double_it(2.)
    self.assertEqual(thing.x, 16.0)

  def test_jit_transpose_basic(self):
    thing = Thing(jnp.array(2.0))

    @jax.custom_vjp
    def foo(x):
      return x

    def foo_fwd(x):
      return x, None

    def foo_bwd(x, g):
      jax_setattr(thing, 'x', g)
      return g,

    foo.defvjp(foo_fwd, foo_bwd)

    foo(3.14)
    self.assertEqual(thing.x, 2.0)

    jax.grad(foo)(3.14)
    self.assertEqual(thing.x, 1.0)

    thing.x = jnp.array(3.14)
    self.assertEqual(thing.x, 3.14)

    jax.jit(jax.grad(foo))(3.14)
    self.assertEqual(thing.x, 1.0)

    thing.x = jnp.array(2.718)
    self.assertEqual(thing.x, 2.718)

    jax.grad(jax.jit(lambda x: jnp.sin(foo(x))))(3.0)
    self.assertAllClose(thing.x, -0.9899925, atol=1e-5, rtol=1e-5, check_dtypes=False)

    thing.x = jnp.array(3.14)
    self.assertEqual(thing.x, 3.14)

    def bar(x):
      out = jnp.sin(foo(x))
      jax_setattr(thing, 'x', 5.0)
      return out

    jax.grad(jax.jit(bar))(3.0)
    self.assertAllClose(thing.x, -0.9899925, atol=1e-5, rtol=1e-5, check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_scan_basic(self, jit: bool):
    thing = Thing(1.0)

    def double_it_10():
      def body(_, __):
        cur_x = jax_getattr(thing ,"x")
        jax_setattr(thing, "x", cur_x * 2.0)
        return None, None
      _, _ = jax.lax.scan(body, None, None, length=10)

    if jit:
      double_it_10 = jax.jit(double_it_10)

    double_it_10()
    self.assertAllClose(thing.x, 1024., check_dtypes=False)

  def test_scan_basic_consts_and_args(self):
    thing = Thing(1.0)

    def double_it_10(y):
      def body(i, x):
        cur_x = jax_getattr(thing ,"x")
        jax_setattr(thing, "x", cur_x * 2.0)
        return i + 1, (y, y)
      _, _ = jax.lax.scan(body, 0, jnp.arange(10))

    jax.jit(double_it_10)(jnp.arange(3.))
    self.assertAllClose(thing.x, 1024., check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_scan_transpose_basic(self, jit: bool):
    thing = Thing(1.0)

    @jax.custom_vjp
    def foo(x):
      return x

    def foo_fwd(x):
      return x, None

    def foo_bwd(x, g):
      jax_setattr(thing, 'x', 2 * jax_getattr(thing, 'x') * g)
      return g,

    foo.defvjp(foo_fwd, foo_bwd)


    def double_it_10(x):
      def body(x, __):
        return foo(x), None
      x, _ = jax.lax.scan(body, x, None, length=10)
      return x

    if jit:
      double_it_10 = jax.jit(double_it_10)

    double_it_10(1.0)
    self.assertAllClose(thing.x, 1., check_dtypes=False)

    jax.grad(double_it_10)(1.0)
    self.assertAllClose(thing.x, 1024., check_dtypes=False)

  def test_arg_to_jit(self):
    thing = Thing(1.0)
    count = 0

    @jax.jit
    def f(obj, x):
      nonlocal count
      count += 1
      jax_setattr(obj, 'x', x)

    f(thing, 2.0)  # don't crash!
    self.assertAllClose(thing.x, 2.0, check_dtypes=False)
    f(thing, 3.0)
    self.assertAllClose(thing.x, 3.0, check_dtypes=False)
    self.assertEqual(count, 1)

  def test_tracer_lifetime_bug(self):
    # regression test for https://github.com/google/jax/issues/20082
    class StatefulRNG:
      key: jax.Array

      def __init__(self, key: jax.Array):
        self.key = key

      def split(self) -> jax.Array:
        key = jax_getattr(self, "key")
        new_key, returned_key = jax.random.split(key)
        jax_setattr(self, "key", new_key)
        return returned_key

    rng = StatefulRNG(jax.random.key(0))

    def jitted():
      rng.split()
      rng.split()

    jax.jit(jitted)()  # don't crash


class AttrsJVPTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_jvp_basic(self, jit):
    thing = Thing(2.0)

    def f():
      x = jax_getattr(thing, 'x')
      x = jnp.sin(x)
      jax_setattr(thing, 'x', x)

    if jit:
      f = jax.jit(f)

    _, _, attr_tangents = attrs.jvp(f, (), (), [(thing, 'x', 1.0)])
    self.assertAllClose(thing.x, jnp.sin(2.0), check_dtypes=False)
    (thing_, attr_, tangent_), = attr_tangents
    self.assertIs(thing, thing_)
    self.assertEqual(attr_, 'x')
    self.assertAllClose(tangent_, jnp.cos(2.0), check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_jvp_clobber(self, jit):
    thing = Thing(2.0)

    def f():
      x = jax_getattr(thing, 'x')
      x = jnp.sin(2.0)
      jax_setattr(thing, 'x', x)

    if jit:
      f = jax.jit(f)

    _, _, attr_tangents = attrs.jvp(f, (), (), [(thing, 'x', 1.0)])
    self.assertAllClose(thing.x, jnp.sin(2.0), check_dtypes=False)
    self.assertEmpty(attr_tangents)

  @parameterized.parameters([True, False])
  def test_jvp_nowrite(self, jit):
    thing = Thing(2.0)

    def f():
      x = jax_getattr(thing, 'x')

    if jit:
      f = jax.jit(f)

    _, _, attr_tangents = attrs.jvp(f, (), (), [(thing, 'x', 1.0)])
    self.assertAllClose(thing.x, 2.0, check_dtypes=False)
    (thing_, attr_, tangent_), = attr_tangents
    self.assertIs(thing, thing_)
    self.assertEqual(attr_, 'x')
    self.assertAllClose(tangent_, 1.0, check_dtypes=False)

  def test_jit_of_jvp(self):
    thing = Thing(2.0)

    def f():
      x = jax_getattr(thing, 'x')
      x = jnp.sin(x)
      jax_setattr(thing, 'x', x)

    @jax.jit
    def g():
      _, _, attr_tangents = attrs.jvp(f, (), (), [(thing, 'x', 1.0)])
      (thing_, attr_, tangent_), = attr_tangents
      self.assertIs(thing, thing_)
      self.assertEqual(attr_, 'x')
      return jax_getattr(thing, 'x'), tangent_

    x, tangent = g()
    self.assertAllClose(x, jnp.sin(2.0), check_dtypes=False)
    self.assertAllClose(tangent, jnp.cos(2.0), check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_jvp_higher_order(self, jit):
    thing = Thing(2.0)

    def f(y):
      x = jax_getattr(thing, 'x')
      w = jnp.tan(jnp.sin(y) * jnp.cos(x))
      z = jnp.tan(jnp.cos(y) * jnp.sin(x))
      jax_setattr(thing, 'x', z)
      return w
    if jit:
      f = jax.jit(f)

    def f_ref(x, y):
      w = jnp.tan(jnp.sin(y) * jnp.cos(x))
      z = jnp.tan(jnp.cos(y) * jnp.sin(x))
      return w, z

    x     = jax.random.normal(jax.random.key(0), (3,))
    x_dot = jax.random.normal(jax.random.key(1), (3,))
    y     = jax.random.normal(jax.random.key(2), (3,))
    y_dot = jax.random.normal(jax.random.key(3), (3,))

    setattr(thing, 'x', x)
    w, w_dot, [(_, _, z_dot)] = attrs.jvp(f, (y,), (y_dot,), [(thing, 'x', x_dot)])
    z = getattr(thing, 'x')

    (w_, z_), (w_dot_, z_dot_) = jax.jvp(f_ref, (x, y), (x_dot, y_dot))

    self.assertAllClose(w, w_, check_dtypes=False)
    self.assertAllClose(z, z_, check_dtypes=False)
    self.assertAllClose(w_dot, w_dot_, check_dtypes=False)
    self.assertAllClose(z_dot, z_dot_, check_dtypes=False)

    def g(x_dot, y, y_dot):
      w, w_dot, [(_, _, z_dot)] = attrs.jvp(f, (y,), (y_dot,), [(thing, 'x', x_dot)])
      return w, w_dot, z_dot

    def g_ref(x, x_dot, y, y_dot):
      (w, z), (w_dot, z_dot) = jax.jvp(f_ref, (x, y), (x_dot, y_dot))
      return w, w_dot, z, z_dot

    x_dot2    = jax.random.normal(jax.random.key(3), (3,))
    x_ddot    = jax.random.normal(jax.random.key(4), (3,))
    y_dot2    = jax.random.normal(jax.random.key(5), (3,))
    y_ddot    = jax.random.normal(jax.random.key(6), (3,))

    setattr(thing, 'x', x)
    (w, w_dot, z_dot), (w_dot2, w_ddot, z_ddot), [(_, _, z_dot2)] = \
        attrs.jvp(g, (x_dot, y, y_dot), (x_ddot, y_dot2, y_ddot),
                  [(thing, 'x', x_dot2)])
    z = getattr(thing, 'x')

    (w_, w_dot_, z_, z_dot_), (w_dot2_, w_ddot_, z_dot2_, z_ddot_) = \
        jax.jvp(g_ref, (x, x_dot, y, y_dot), (x_dot2, x_ddot, y_dot2, y_ddot))

    self.assertAllClose(     w,      w_, check_dtypes=False)
    self.assertAllClose(     z,      z_, check_dtypes=False)
    self.assertAllClose( w_dot,  w_dot_, check_dtypes=False)
    self.assertAllClose( z_dot,  z_dot_, check_dtypes=False)
    self.assertAllClose(w_dot2, w_dot2_, check_dtypes=False)
    self.assertAllClose(z_dot2, z_dot2_, check_dtypes=False)
    self.assertAllClose(w_ddot, w_ddot_, check_dtypes=False)
    self.assertAllClose(z_ddot, z_ddot_, check_dtypes=False)

class AttrsLinTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_attr_output(self, jit):
    thing = Thing(1.0)

    def f(x, _):
      y = jnp.sin(x)
      jax_setattr(thing, 'x', y)

    if jit:
      f = jax.jit(f)

    out, f_lin = attrs.linearize(f, 3.0, 4.0)
    self.assertIsNone(out)
    self.assertAllClose(thing.x, jnp.sin(3.0), check_dtypes=False)

    out_dot, attr_tangents = f_lin(1.0, 2.0, attr_tangents={})
    self.assertIsNone(out_dot)
    self.assertAllClose(thing.x, jnp.sin(3.0))  # didn't change
    self.assertLen(attr_tangents, 1)
    self.assertAllClose(attr_tangents[(thing, 'x')], jnp.cos(3.0),
                        check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_attr_input(self, jit):
    thing = Thing(1.0)

    def f():
      x = jax_getattr(thing, 'x')
      return jnp.sin(x)

    if jit:
      f = jax.jit(f)

    out, f_lin = attrs.linearize(f, attrs=[(thing, 'x')])
    self.assertAllClose(out, jnp.sin(1.0), check_dtypes=False)

    out_dot, attr_tangents = f_lin(attr_tangents={(thing, 'x'): 2.0})
    self.assertAllClose(out_dot, 2. * jnp.cos(1.0), check_dtypes=False)
    self.assertLen(attr_tangents, 1)
    self.assertAllClose(attr_tangents[(thing, 'x')], 2.0, check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_attr_inout(self, jit):
    thing1 = Thing(1.0)
    thing2 = Thing(2.0)

    def f(x, y):
      z = jax_getattr(thing1, 'x')
      w = jax_getattr(thing2, 'x')
      out = jnp.sin(x * y * z * w)
      jax_setattr(thing1, 'x', out)
      jax_setattr(thing2, 'x', 2 * out)
      return 3 * out, 4 * out

    if jit:
      f = jax.jit(f)

    def f_ref(x, y, z, w):
      out = jnp.sin(x * y * z * w)
      return (3 * out, 4 * out), (out, 2 * out)

    out, f_lin = attrs.linearize(f, 3., 4., attrs=[(thing1, 'x'), (thing2, 'x')])
    expected = (3 * jnp.sin(1. * 2. * 3. * 4.),
                4 * jnp.sin(1. * 2. * 3. * 4.))
    self.assertAllClose(out, expected, check_dtypes=False)
    self.assertAllClose(thing1.x, jnp.sin(1. * 2. * 3. * 4.))
    self.assertAllClose(thing2.x, 2 * jnp.sin(1. * 2. * 3. * 4.))

    (out_ref, state_out_ref), f_lin_ref = jax.linearize(f_ref, 3., 4., 1., 2.)
    self.assertAllClose(out, out_ref, check_dtypes=False)
    self.assertAllClose((thing1.x, thing2.x), state_out_ref, check_dtypes=False)

    out_dot, attr_tangents = f_lin(1., 2.,
                                   attr_tangents={(thing1, 'x'): 5.,
                                                  (thing2, 'x'): 6.})
    self.assertAllClose(thing1.x, jnp.sin(1. * 2. * 3. * 4.))
    self.assertAllClose(thing2.x, 2 * jnp.sin(1. * 2. * 3. * 4.))
    (out_dot_ref, state_dot_ref) = f_lin_ref(1., 2., 5., 6.)
    self.assertAllClose(out_dot, out_dot_ref, check_dtypes=False)
    self.assertLen(attr_tangents, 2)
    self.assertAllClose(attr_tangents[(thing1, 'x')], state_dot_ref[0],
                        check_dtypes=False)
    self.assertAllClose(attr_tangents[(thing2, 'x')], state_dot_ref[1],
                        check_dtypes=False)

class AttrsVJPTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_attr_input(self, jit):
    thing = Thing(1.0)

    def f():
      x = jax_getattr(thing, 'x')
      return jnp.sin(x)

    if jit:
      f = jax.jit(f)

    out, f_vjp = attrs.vjp(f, attrs=[(thing, 'x')])
    self.assertAllClose(out, jnp.sin(1.0), check_dtypes=False)

    arg_cts, attr_cotangents = f_vjp(1.0)
    self.assertEqual(arg_cts, ())
    self.assertLen(attr_cotangents, 1)
    self.assertAllClose(attr_cotangents[(thing, 'x')], jnp.cos(1.0),
                        check_dtypes=False)

  @parameterized.parameters([True, False])
  def test_attr_output(self, jit):
    thing = Thing(1.0)

    def f(x, _):
      y = jnp.sin(x)
      jax_setattr(thing, 'x', y)

    if jit:
      f = jax.jit(f)

    out, f_vjp = attrs.vjp(f, 3.0, 4.0)
    self.assertIsNone(out)
    self.assertAllClose(thing.x, jnp.sin(3.0), check_dtypes=False)

    arg_cts, attr_cotangents = f_vjp(None, attr_cotangents={(thing, 'x'): 2.0})
    self.assertAllClose(arg_cts, (2 * jnp.cos(3.0), 0.), check_dtypes=False)
    self.assertLen(attr_cotangents, 0)

  @parameterized.parameters([True, False])
  def test_attr_inout(self, jit):
    thing1 = Thing(1.0)
    thing2 = Thing(2.0)

    def f(x, y):
      z = jax_getattr(thing1, 'x')
      w = jax_getattr(thing2, 'x')
      out = jnp.sin(x * y * z * w)
      jax_setattr(thing1, 'x', out)
      jax_setattr(thing2, 'x', 2 * out)
      return 3 * out, 4 * out

    if jit:
      f = jax.jit(f)

    def f_ref(x, y, z, w):
      out = jnp.sin(x * y * z * w)
      return (3 * out, 4 * out), (out, 2 * out)

    out, f_vjp = attrs.vjp(f, 3., 4., attrs=[(thing1, 'x'), (thing2, 'x')])
    (out_ref, state_out_ref), f_vjp_ref = jax.vjp(f_ref, 3., 4., 1., 2.)
    self.assertAllClose(out, out_ref, check_dtypes=False)
    self.assertAllClose((thing1.x, thing2.x), state_out_ref, check_dtypes=False)

    in_bar, attr_cotangents = f_vjp((1., 2.),
                                    attr_cotangents={(thing1, 'x'): 5.,
                                                     (thing2, 'x'): 6.})
    in_bar_ref_ = f_vjp_ref(((1., 2.), (5., 6.)))
    in_bar_ref, attr_cotangents_ref = in_bar_ref_[:2], in_bar_ref_[2:]
    self.assertAllClose(in_bar, in_bar_ref, check_dtypes=False)
    self.assertLen(attr_cotangents, 2)
    self.assertAllClose(attr_cotangents[(thing1, 'x')], attr_cotangents_ref[0],
                        check_dtypes=False)
    self.assertAllClose(attr_cotangents[(thing2, 'x')], attr_cotangents_ref[1],
                        check_dtypes=False)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
