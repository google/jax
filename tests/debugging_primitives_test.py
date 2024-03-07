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
import collections
import contextlib
import functools
import textwrap
import unittest

from absl.testing import absltest
import jax
from jax import lax
from jax.experimental import pjit
from jax.interpreters import pxla
from jax._src import ad_checkpoint
from jax._src import debugging
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src.maps import xmap
import jax.numpy as jnp
import numpy as np

try:
  import rich
except ModuleNotFoundError:
  rich = None

jax.config.parse_flags_with_absl()

debug_print = debugging.debug_print

def _format_multiline(text):
  return textwrap.dedent(text).lstrip()

_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(2))

def tearDownModule():
  _exit_stack.close()

class DummyDevice:
  def __init__(self, platform, id):
    self.platform = platform
    self.id = id


class DebugCallbackTest(jtu.JaxTestCase):

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  def test_error_with_non_callable(self):
    with self.assertRaisesRegex(TypeError, "callable"):
      jax.debug.callback("this is not debug.print!")


class DebugPrintTest(jtu.JaxTestCase):

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()

  def test_simple_debug_print_works_in_eager_mode(self):
    def f(x):
      debug_print('x: {}', x)
    with jtu.capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  def test_debug_print_works_with_named_format_strings(self):
    def f(x):
      debug_print('x: {x}', x=x)
    with jtu.capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  def test_multiple_debug_prints_should_print_multiple_values(self):
    def f(x):
      debug_print('x: {x}', x=x)
      debug_print('y: {y}', y=x + 1)
    with jtu.capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\ny: 3\n")

  def test_can_stage_out_debug_print(self):
    @jax.jit
    def f(x):
      debug_print('x: {x}', x=x)
    with jtu.capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  @jtu.device_supports_buffer_donation()
  def test_can_stage_out_debug_print_with_donate_argnums(self):
    def f(x, y):
      debug_print('x: {x}', x=x)
      return x + y
    f = jax.jit(f, donate_argnums=0)
    with jtu.capture_stdout() as output:
      f(2, 3)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  def test_can_stage_out_ordered_print(self):
    @jax.jit
    def f(x):
      debug_print('x: {x}', x=x, ordered=True)
    with jtu.capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  @jtu.device_supports_buffer_donation()
  def test_can_stage_out_ordered_print_with_donate_argnums(self):
    def f(x, y):
      debug_print('x: {x}', x=x, ordered=True)
      return x + y
    f = jax.jit(f, donate_argnums=0)
    with jtu.capture_stdout() as output:
      f(2, 3)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  @jtu.device_supports_buffer_donation()
  def test_can_stage_out_prints_with_donate_argnums(self):
    def f(x, y):
      debug_print('x: {x}', x=x, ordered=True)
      debug_print('x: {x}', x=x)
      return x + y
    f = jax.jit(f, donate_argnums=0)
    with jtu.capture_stdout() as output:
      f(2, 3)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\nx: 2\n")

  def test_can_double_stage_out_ordered_print(self):
    @jax.jit
    @jax.jit
    def f(x):
      debug_print('x: {x}', x=x, ordered=True)
    with jtu.capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 2\n")

  def test_can_stage_out_ordered_print_with_pytree(self):
    @jax.jit
    def f(x):
      struct = dict(foo=x)
      debug_print('x: {}', struct, ordered=True)
    with jtu.capture_stdout() as output:
      f(np.array(2, np.int32))
      jax.effects_barrier()
    self.assertEqual(output(), f"x: {str(dict(foo=jnp.array(2, np.int32)))}\n")

  def test_debug_print_should_use_default_layout(self):
    data = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]], dtype=np.int32)
    @jax.jit
    def f(x):
      jax.debug.print("{}", x)

    with jtu.capture_stdout() as output:
      f(data)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
        [[ 1  2  3  4  5  6  7  8  9 10 12 13 14]
         [ 1  2  3  4  5  6  7  8  9 10 12 13 14]
         [ 1  2  3  4  5  6  7  8  9 10 12 13 14]
         [ 1  2  3  4  5  6  7  8  9 10 12 13 14]
         [ 1  2  3  4  5  6  7  8  9 10 12 13 14]
         [ 1  2  3  4  5  6  7  8  9 10 12 13 14]
         [ 1  2  3  4  5  6  7  8  9 10 12 13 14]
         [ 1  2  3  4  5  6  7  8  9 10 12 13 14]]
    """))


class DebugPrintTransformationTest(jtu.JaxTestCase):

  def test_debug_print_batching(self):
    @jax.vmap
    def f(x):
      debug_print('hello: {}', x)
    with jtu.capture_stdout() as output:
      f(jnp.arange(2))
      jax.effects_barrier()
    self.assertEqual(output(), "hello: 0\nhello: 1\n")

  def test_debug_print_batching_with_diff_axes(self):
    @functools.partial(jax.vmap, in_axes=(0, 1))
    def f(x, y):
      debug_print('hello: {} {}', x, y)
    with jtu.capture_stdout() as output:
      f(jnp.arange(2), jnp.arange(2)[None])
      jax.effects_barrier()
    self.assertEqual(output(), "hello: 0 [0]\nhello: 1 [1]\n")

  def tested_debug_print_with_nested_vmap(self):
    def f(x):
      debug_print('hello: {}', x)
    # Call with
    # [[0, 1],
    #  [2, 3],
    #  [4, 5]]
    with jtu.capture_stdout() as output:
      # Should print over 0-axis then 1-axis
      jax.vmap(jax.vmap(f))(jnp.arange(6).reshape((3, 2)))
      jax.effects_barrier()
    self.assertEqual(
        output(),
        "hello: 0\nhello: 2\nhello: 4\nhello: 1\nhello: 3\nhello: 5\n")
    with jtu.capture_stdout() as output:
      # Should print over 1-axis then 0-axis
      jax.vmap(jax.vmap(f, in_axes=0), in_axes=1)(jnp.arange(6).reshape((3, 2)))
      jax.effects_barrier()
    self.assertEqual(
        output(),
        "hello: 0\nhello: 1\nhello: 2\nhello: 3\nhello: 4\nhello: 5\n")

  def test_debug_print_jvp_rule(self):
    def f(x):
      debug_print('x: {}', x)
    with jtu.capture_stdout() as output:
      jax.jvp(f, (1.,), (1.,))
      jax.effects_barrier()
    self.assertEqual(output(), "x: 1.0\n")

  def test_debug_print_vjp_rule(self):
    def f(x):
      debug_print('x: {}', x)
    with jtu.capture_stdout() as output:
      jax.vjp(f, 1.)
      jax.effects_barrier()
    self.assertEqual(output(), "x: 1.0\n")

  def test_debug_print_in_custom_jvp(self):

    @jax.custom_jvp
    def print_tangent(x):
      return x

    @print_tangent.defjvp
    def _(primals, tangents):
      (x,), (t,) = primals, tangents
      debug_print("x_tangent: {}", t)
      return x, t

    def f(x):
      x = jnp.square(x)
      x = print_tangent(x)
      return x

    with jtu.capture_stdout() as output:
      x = jnp.array(1., jnp.float32)
      jax.jvp(f, (x,), (x,))
      jax.effects_barrier()
    expected = jnp.array(2., jnp.float32)
    self.assertEqual(output(), f"x_tangent: {expected}\n")

  @unittest.skip("doesn't work yet!")  # TODO(mattjj,sharadmv)
  def test_debug_print_in_custom_jvp_linearize(self):

    @jax.custom_jvp
    def print_tangent(x):
      return x

    @print_tangent.defjvp
    def _(primals, tangents):
      (x,), (t,) = primals, tangents
      debug_print("x_tangent: {}", t)
      return x, t

    def f(x):
      x = jnp.sin(x)
      x = print_tangent(x)
      return x

    with jtu.capture_stdout() as output:
      x = jnp.array(1., jnp.float32)
      y, f_lin = jax.linearize(f, x)
      jax.effects_barrier()
    self.assertEqual(output(), "")

    with jtu.capture_stdout() as output:
      _ = f_lin(x)
      jax.effects_barrier()
    expected = jnp.cos(jnp.array(1., jnp.float32))
    self.assertEqual(output(), f"x_tangent: {expected}\n")

  def test_debug_print_grad_with_custom_vjp_rule(self):
    @jax.custom_vjp
    def print_grad(x):
      return x

    def print_grad_fwd(x):
      return x, None

    def print_grad_bwd(_, x_grad):
      debug_print("x_grad: {}", x_grad)
      return (x_grad,)

    print_grad.defvjp(print_grad_fwd, print_grad_bwd)
    def f(x):
      debug_print("x: {}", x)
      x = print_grad(x)
      return jnp.square(x)

    with jtu.capture_stdout() as output:
      jax.grad(f)(jnp.array(1., jnp.float32))
      jax.effects_barrier()
    expected = jnp.array(2., jnp.float32)
    self.assertEqual(output(), f"x: 1.0\nx_grad: {expected}\n")

  def test_debug_print_transpose_rule(self):
    def f(x):
      debug_print('should never be called: {}', x)
      return x
    with jtu.capture_stdout() as output:
      jax.linear_transpose(f, 1.)(1.)
      jax.effects_barrier()
    # `debug_print` should be dropped by `partial_eval` because of no
    # output data-dependence.
    self.assertEqual(output(), "")

  @jtu.sample_product(ordered=[False, True])
  def test_remat_of_debug_print(self, ordered):
    def f_(x):
      y = ad_checkpoint.checkpoint_name(x + 1., "y")
      z = ad_checkpoint.checkpoint_name(y * 2., "z")
      debug_print('y: {}, z: {}', y, z, ordered=ordered)
      return ad_checkpoint.checkpoint_name(jnp.exp(z), "w")

    # Policy that saves everything so the debug callback will be saved
    f = ad_checkpoint.checkpoint(f_, policy=ad_checkpoint.everything_saveable)

    with jtu.capture_stdout() as output:
      jax.grad(f)(2.)
      jax.effects_barrier()
    # We expect the print to happen once since it gets saved and isn't
    # rematerialized.
    self.assertEqual(output(), "y: 3.0, z: 6.0\n")

    # Policy that saves nothing so everything gets rematerialized, including the
    # debug callback
    f = ad_checkpoint.checkpoint(f_, policy=ad_checkpoint.nothing_saveable)

    with jtu.capture_stdout() as output:
      jax.grad(f)(2.)
      jax.effects_barrier()
    # We expect the print to happen twice since it is rematerialized.
    self.assertEqual(output(), "y: 3.0, z: 6.0\n" * 2)

    # Policy that does not save `z` so we will need to rematerialize the print
    f = ad_checkpoint.checkpoint(
        f_, policy=ad_checkpoint.save_any_names_but_these("z"))

    with jtu.capture_stdout() as output:
      jax.grad(f)(2.)
      jax.effects_barrier()
    # We expect the print to happen twice since it is rematerialized.
    self.assertEqual(output(), "y: 3.0, z: 6.0\n" * 2)

    def save_everything_but_these_names(*names_not_to_save):
      names_not_to_save = frozenset(names_not_to_save)
      def policy(prim, *_, **params):
        if prim is ad_checkpoint.name_p:
          return params['name'] not in names_not_to_save
        return True # Save everything else
      return policy

    # Policy that saves everything but `y`
    f = ad_checkpoint.checkpoint(
        f_, policy=save_everything_but_these_names("y"))

    with jtu.capture_stdout() as output:
      jax.grad(f)(2.)
      jax.effects_barrier()
    # We expect the print to happen once because `y` is not rematerialized and
    # we won't do extra materialization.
    self.assertEqual(output(), "y: 3.0, z: 6.0\n")

    # Policy that saves everything but `y` and `z`
    f = ad_checkpoint.checkpoint(
        f_, policy=save_everything_but_these_names("y", "z"))

    with jtu.capture_stdout() as output:
      jax.grad(f)(2.)
      jax.effects_barrier()
    # We expect the print to happen twice because both `y` and `z` have been
    # rematerialized and we don't have to do any extra rematerialization to
    # print.
    self.assertEqual(output(), "y: 3.0, z: 6.0\n" * 2)

  def test_debug_print_in_staged_out_custom_jvp(self):
    @jax.jit
    def f(x):
      @jax.custom_jvp
      def g(x):
        debug_print("hello: {x}", x=x)
        return x
      def g_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        debug_print("goodbye: {x} {t}", x=x, t=t)
        return x, t
      g.defjvp(g_jvp)
      return g(x)

    with jtu.capture_stdout() as output:
      f(2.)
      jax.effects_barrier()
    self.assertEqual(output(), "hello: 2.0\n")

    with jtu.capture_stdout() as output:
      jax.jvp(f, (2.,), (3.,))
      jax.effects_barrier()
    self.assertEqual(output(), "goodbye: 2.0 3.0\n")

  def test_debug_print_in_staged_out_custom_vjp(self):
    @jax.jit
    def f(x):
      @jax.custom_vjp
      def g(x):
        debug_print("hello: {x}", x=x)
        return x
      def g_fwd(x):
        debug_print("hello fwd: {x}", x=x)
        return x, x
      def g_bwd(x, g):
        debug_print("hello bwd: {x} {g}", x=x, g=g)
        return (g,)
      g.defvjp(fwd=g_fwd, bwd=g_bwd)
      return g(x)

    with jtu.capture_stdout() as output:
      f(2.)
      jax.effects_barrier()
    self.assertEqual(output(), "hello: 2.0\n")

    with jtu.capture_stdout() as output:
      _, f_vjp = jax.vjp(f, 2.)
      jax.effects_barrier()
    self.assertEqual(output(), "hello fwd: 2.0\n")

    with jtu.capture_stdout() as output:
      f_vjp(3.0)
      jax.effects_barrier()
    self.assertEqual(output(), "hello bwd: 2.0 3.0\n")

class DebugPrintControlFlowTest(jtu.JaxTestCase):

  def _assertLinesEqual(self, text1, text2):

    def _count(lines):
      return collections.Counter(lines)

    self.assertDictEqual(_count(text1.split("\n")), _count(text2.split("\n")))

  @jtu.sample_product(ordered=[False, True])
  def test_can_print_inside_scan(self, ordered):
    def f(xs):
      def _body(carry, x):
        debug_print("carry: {carry}, x: {x}", carry=carry, x=x, ordered=ordered)
        return carry + 1, x + 1
      return lax.scan(_body, 2, xs)
    with jtu.capture_stdout() as output:
      f(jnp.arange(2))
      jax.effects_barrier()
    self.assertEqual(
        output(),
        _format_multiline("""
      carry: 2, x: 0
      carry: 3, x: 1
      """))

  @jtu.sample_product(ordered=[False, True])
  def test_can_print_inside_for_loop(self, ordered):
    def f(x):
      def _body(i, x):
        debug_print("i: {i}", i=i, ordered=ordered)
        debug_print("x: {x}", x=x, ordered=ordered)
        return x + 1
      return lax.fori_loop(0, 5, _body, x)
    with jtu.capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    expected = _format_multiline("""
      i: 0
      x: 2
      i: 1
      x: 3
      i: 2
      x: 4
      i: 3
      x: 5
      i: 4
      x: 6
      """)
    if ordered:
      self.assertEqual(output(), expected)
    else:
      self._assertLinesEqual(output(), expected)

  @jtu.sample_product(ordered=[False, True])
  def test_can_print_inside_while_loop_body(self, ordered):
    def f(x):
      def _cond(x):
        return x < 10
      def _body(x):
        debug_print("x: {x}", x=x, ordered=ordered)
        return x + 1
      return lax.while_loop(_cond, _body, x)
    with jtu.capture_stdout() as output:
      f(5)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      x: 5
      x: 6
      x: 7
      x: 8
      x: 9
      """))

  @jtu.sample_product(ordered=[False, True])
  def test_can_print_inside_while_loop_cond(self, ordered):
    def f(x):
      def _cond(x):
        debug_print("x: {x}", x=x, ordered=ordered)
        return x < 10
      def _body(x):
        return x + 1
      return lax.while_loop(_cond, _body, x)
    with jtu.capture_stdout() as output:
      f(5)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      x: 5
      x: 6
      x: 7
      x: 8
      x: 9
      x: 10
      """))

    with jtu.capture_stdout() as output:
      f(10)
      jax.effects_barrier()
    # Should run the cond once
    self.assertEqual(output(), _format_multiline("""
      x: 10
      """))

  @jtu.sample_product(ordered=[False, True])
  def test_can_print_in_batched_while_cond(self, ordered):
    def f(x):
      def _cond(x):
        debug_print("x: {x}", x=x, ordered=ordered)
        return x < 5
      def _body(x):
        return x + 1
      return lax.while_loop(_cond, _body, x)
    with jtu.capture_stdout() as output:
      jax.vmap(f)(jnp.arange(2))
      jax.effects_barrier()
    if ordered:
      expected = _format_multiline("""
      x: 0
      x: 1
      x: 1
      x: 2
      x: 2
      x: 3
      x: 3
      x: 4
      x: 4
      x: 5
      x: 5
      x: 6
      """)
      self.assertEqual(output(), expected)
    else:
      # When the print is unordered, the `cond` is called an additional time
      # after the `_body` runs, so we get more prints.
      expected = _format_multiline("""
      x: 0
      x: 1
      x: 0
      x: 1
      x: 1
      x: 2
      x: 1
      x: 2
      x: 2
      x: 3
      x: 2
      x: 3
      x: 3
      x: 4
      x: 3
      x: 4
      x: 4
      x: 5
      x: 4
      x: 5
      x: 5
      x: 5
      """)
      self._assertLinesEqual(output(), expected)

  @jtu.sample_product(ordered=[False, True])
  def test_can_print_inside_cond(self, ordered):
    def f(x):
      def true_fun(x):
        debug_print("true: {}", x, ordered=ordered)
        return x
      def false_fun(x):
        debug_print("false: {}", x, ordered=ordered)
        return x
      return lax.cond(x < 5, true_fun, false_fun, x)
    with jtu.capture_stdout() as output:
      f(5)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      false: 5
      """))
    with jtu.capture_stdout() as output:
      f(4)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      true: 4
      """))

  @jtu.sample_product(ordered=[False, True])
  def test_can_print_inside_switch(self, ordered):
    def f(x):
      def b1(x):
        debug_print("b1: {}", x, ordered=ordered)
        return x
      def b2(x):
        debug_print("b2: {}", x, ordered=ordered)
        return x
      def b3(x):
        debug_print("b3: {}", x, ordered=ordered)
        return x
      return lax.switch(x, (b1, b2, b3), x)
    with jtu.capture_stdout() as output:
      f(0)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      b1: 0
      """))
    with jtu.capture_stdout() as output:
      f(1)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      b2: 1
      """))
    with jtu.capture_stdout() as output:
      f(2)
      jax.effects_barrier()
    self.assertEqual(output(), _format_multiline("""
      b3: 2
      """))

class DebugPrintParallelTest(jtu.JaxTestCase):

  def _assertLinesEqual(self, text1, text2):

    def _count(lines):
      return collections.Counter(lines)

    self.assertDictEqual(_count(text1.split("\n")), _count(text2.split("\n")))

  def test_ordered_print_not_supported_in_pmap(self):

    @jax.pmap
    def f(x):
      debug_print("{}", x, ordered=True)
    with self.assertRaisesRegex(
        ValueError, "Ordered effects not supported in `pmap`."):
      f(jnp.arange(jax.local_device_count()))

  def test_unordered_print_works_in_pmap(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")

    @jax.pmap
    def f(x):
      debug_print("hello: {}", x, ordered=False)
    with jtu.capture_stdout() as output:
      f(jnp.arange(jax.local_device_count()))
      jax.effects_barrier()
    lines = [f"hello: {i}\n" for i in range(jax.local_device_count())]
    self._assertLinesEqual(output(), "".join(lines))

    @jax.pmap
    def f2(x):
      debug_print('hello: {}', x)
      debug_print('hello: {}', x + 2)
    with jtu.capture_stdout() as output:
      f2(jnp.arange(2))
      jax.effects_barrier()
    self._assertLinesEqual(output(), "hello: 0\nhello: 1\nhello: 2\nhello: 3\n")

  def test_unordered_print_with_pjit(self):
    def f(x):
      debug_print("{}", x, ordered=False)
      return x
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ['dev'])
    spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dev'))
    out_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    f = pjit.pjit(f, in_shardings=spec, out_shardings=out_spec)
    with mesh:
      with jtu.capture_stdout() as output:
        f(np.arange(8, dtype=jnp.int32))
        jax.effects_barrier()
      self.assertEqual(output(), "[0 1 2 3 4 5 6 7]\n")

    def f2(x):
      y = x.dot(x)
      debug_print("{}", y, ordered=False)
      return y
    f2 = pjit.pjit(f2, in_shardings=spec, out_shardings=out_spec)
    with jax.sharding.Mesh(np.array(jax.devices()), ['dev']):
      with jtu.capture_stdout() as output:
        f2(np.arange(8, dtype=jnp.int32))
        jax.effects_barrier()
      self.assertEqual(output(), "140\n")

  def test_nested_pjit_debug_print(self):
    def f(x):
      debug_print("{}", x)
      return x

    with jtu.capture_stdout() as output:
      pjit.pjit(pjit.pjit(f))(jnp.arange(8))
      jax.effects_barrier()
    self.assertEqual(output(), "[0 1 2 3 4 5 6 7]\n")

  def test_unordered_print_of_pjit_of_while(self):
    def f(x):
      def cond(carry):
        i, *_ = carry
        return i < 5
      def body(carry):
        i, x = carry
        debug_print("{}", x, ordered=False)
        x = x + 1
        return (i + 1, x)
      return lax.while_loop(cond, body, (0, x))[1]

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ['dev'])
    spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dev'))
    f = pjit.pjit(f, in_shardings=spec, out_shardings=spec)
    with mesh:
      with jtu.capture_stdout() as output:
        f(np.arange(8, dtype=jnp.int32))
        jax.effects_barrier()
      self.assertEqual(output(),
          "[0 1 2 3 4 5 6 7]\n"
          "[1 2 3 4 5 6 7 8]\n"
          "[2 3 4 5 6 7 8 9]\n"
          "[ 3  4  5  6  7  8  9 10]\n"
          "[ 4  5  6  7  8  9 10 11]\n")

  def test_unordered_print_of_pjit_of_xmap(self):
    def f(x):
      def foo(x):
        idx = lax.axis_index('foo')
        debug_print("{idx}: {x}", idx=idx, x=x)
        return jnp.mean(x, axis=['foo'])
      out = xmap(foo, in_axes=['foo'], out_axes=[...])(x)
      debug_print("Out: {}", out)
      return out
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ['dev'])
    in_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dev'))
    out_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    f = pjit.pjit(f, in_shardings=in_spec, out_shardings=out_spec)
    with mesh:
      with jtu.capture_stdout() as output:
        f(jnp.arange(8, dtype=jnp.int32) * 2)
        lines = ["0: 0", "1: 2", "2: 4", "3: 6", "4: 8", "5: 10", "6: 12",
                 "7: 14", "Out: 7.0", ""]
        jax.effects_barrier()

      self._assertLinesEqual(output(), "\n".join(lines))

  def test_unordered_print_with_xmap(self):
    def f(x):
      debug_print("{}", x, ordered=False)
    f = xmap(f, in_axes=['a'], out_axes=None, backend='cpu',
             axis_resources={'a': 'dev'})
    with jax.sharding.Mesh(np.array(jax.devices()), ['dev']):
      with jtu.capture_stdout() as output:
        f(np.arange(40))
        jax.effects_barrier()
      lines = [f"{i}\n" for i in range(40)]
      self._assertLinesEqual(output(), "".join(lines))

  def test_unordered_print_works_in_pmap_of_while(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")

    @jax.pmap
    def f(x):
      def cond(x):
        return x < 3
      def body(x):
        debug_print("hello: {}", x, ordered=False)
        return x + 1
      return lax.while_loop(cond, body, x)

    with jtu.capture_stdout() as output:
      f(jnp.arange(2))
      jax.effects_barrier()

    self._assertLinesEqual(
        output(), "hello: 0\nhello: 1\nhello: 2\n"
        "hello: 1\nhello: 2\n")

  def test_incorrectly_formatted_string(self):

    @jax.jit
    def f(x):
      debug_print("hello: {x}", x)
      return x

    with self.assertRaises(KeyError):
      f(jnp.arange(2))
      jax.effects_barrier()

    @jax.jit
    def f(x):
      debug_print("hello: {}", x=x)
      return x

    with self.assertRaises(IndexError):
      f(jnp.arange(2))
      jax.effects_barrier()

  def test_format_string_errors_with_unused_args(self):

    @jax.jit
    def f(x):
      debug_print("hello: {x}", x=x, y=x)
      return x

    with self.assertRaisesRegex(ValueError, "Unused keyword arguments"):
      f(jnp.arange(2))
      jax.effects_barrier()

    @jax.jit
    def g(x):
      debug_print("hello", x)
      return x

    with self.assertRaisesRegex(ValueError, "Unused positional arguments"):
      g(jnp.arange(2))
      jax.effects_barrier()

  def test_accidental_fstring(self):

    @jax.jit
    def f(x):
      debug_print(f"hello: {x}", x=x)
      return x

    with self.assertRaisesRegex(ValueError, "You may be passing an f-string"):
      f(jnp.arange(2))
      jax.effects_barrier()

class VisualizeShardingTest(jtu.JaxTestCase):

  def _create_devices(self, shape):
    num_devices = np.prod(shape)
    devices = [DummyDevice("CPU", i) for i in range(num_devices)]
    return np.array(devices).reshape(shape)

  def test_trivial_sharding(self):
    mesh = jax.sharding.Mesh(self._create_devices(1), ['x'])
    pspec = jax.sharding.PartitionSpec('x')
    sd = jax.sharding.NamedSharding(mesh, pspec)
    shape = (5,)
    with jtu.capture_stdout() as output:
      debugging.visualize_sharding(shape, sd)
    self.assertEqual(output(), _format_multiline("""
    ┌───────┐
    │ CPU 0 │
    └───────┘
    """))

  def test_trivial_sharding_with_scale(self):
    mesh = jax.sharding.Mesh(self._create_devices(1), ['x'])
    pspec = jax.sharding.PartitionSpec('x')
    sd = jax.sharding.NamedSharding(mesh, pspec)
    shape = (5,)
    with jtu.capture_stdout() as output:
      debugging.visualize_sharding(shape, sd, scale=8.)
    self.assertEqual(output(), _format_multiline("""
    ┌──────────────────────────────────────┐
    │                CPU 0                 │
    └──────────────────────────────────────┘
    """))

  def test_full_sharding(self):
    mesh = jax.sharding.Mesh(self._create_devices((8, 4)), ['x', 'y'])
    pspec = jax.sharding.PartitionSpec('x', 'y')
    sd = jax.sharding.NamedSharding(mesh, pspec)
    shape = (8, 8)
    with jtu.capture_stdout() as output:
      debugging.visualize_sharding(shape, sd)
    expected = _format_multiline("""
    ┌───────┬───────┬───────┬───────┐
    │ CPU 0 │ CPU 1 │ CPU 2 │ CPU 3 │
    ├───────┼───────┼───────┼───────┤
    │ CPU 4 │ CPU 5 │ CPU 6 │ CPU 7 │
    ├───────┼───────┼───────┼───────┤
    │ CPU 8 │ CPU 9 │CPU 10 │CPU 11 │
    ├───────┼───────┼───────┼───────┤
    │CPU 12 │CPU 13 │CPU 14 │CPU 15 │
    ├───────┼───────┼───────┼───────┤
    │CPU 16 │CPU 17 │CPU 18 │CPU 19 │
    ├───────┼───────┼───────┼───────┤
    │CPU 20 │CPU 21 │CPU 22 │CPU 23 │
    ├───────┼───────┼───────┼───────┤
    │CPU 24 │CPU 25 │CPU 26 │CPU 27 │
    ├───────┼───────┼───────┼───────┤
    │CPU 28 │CPU 29 │CPU 30 │CPU 31 │
    └───────┴───────┴───────┴───────┘
    """)
    self.assertEqual(output(), expected)

  def test_sharding_with_replication(self):
    shape = (8, 8)
    mesh = jax.sharding.Mesh(self._create_devices((8, 4)), ['x', 'y'])

    pspec = jax.sharding.PartitionSpec('x', None)
    sd = jax.sharding.NamedSharding(mesh, pspec)
    with jtu.capture_stdout() as output:
      debugging.visualize_sharding(shape, sd)
    expected = _format_multiline("""
    ┌───────────────────────┐
    │      CPU 0,1,2,3      │
    ├───────────────────────┤
    │      CPU 4,5,6,7      │
    ├───────────────────────┤
    │     CPU 8,9,10,11     │
    ├───────────────────────┤
    │    CPU 12,13,14,15    │
    ├───────────────────────┤
    │    CPU 16,17,18,19    │
    ├───────────────────────┤
    │    CPU 20,21,22,23    │
    ├───────────────────────┤
    │    CPU 24,25,26,27    │
    ├───────────────────────┤
    │    CPU 28,29,30,31    │
    └───────────────────────┘
    """)
    self.assertEqual(output(), expected)

    mesh = jax.sharding.Mesh(self._create_devices((4, 2)), ['x', 'y'])
    pspec = jax.sharding.PartitionSpec(None, 'y')
    sd = jax.sharding.NamedSharding(mesh, pspec)
    with jtu.capture_stdout() as output:
      debugging.visualize_sharding(shape, sd)
    expected = _format_multiline("""
    ┌───────────┬───────────┐
    │           │           │
    │           │           │
    │           │           │
    │           │           │
    │CPU 0,2,4,6│CPU 1,3,5,7│
    │           │           │
    │           │           │
    │           │           │
    │           │           │
    └───────────┴───────────┘
    """)
    self.assertEqual(output(), expected)

  def test_visualize_wide_array(self):
    shape = (128, 10000)
    mesh = jax.sharding.Mesh(self._create_devices((8, 4)), ['x', 'y'])

    pspec = jax.sharding.PartitionSpec('x', None)
    sd = jax.sharding.NamedSharding(mesh, pspec)
    with jtu.capture_stdout() as output:
      debugging.visualize_sharding(shape, sd)
    expected = _format_multiline("""
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                                 CPU 0,1,2,3                                  │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │                                 CPU 4,5,6,7                                  │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │                                CPU 8,9,10,11                                 │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │                               CPU 12,13,14,15                                │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │                               CPU 16,17,18,19                                │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │                               CPU 20,21,22,23                                │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │                               CPU 24,25,26,27                                │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │                               CPU 28,29,30,31                                │
    └──────────────────────────────────────────────────────────────────────────────┘
    """)
    self.assertEqual(output(), expected)

  def test_visualize_pmap_sharding(self):
    ss = pxla.ShardingSpec(
        sharding=(pxla.Unstacked(8),),
        mesh_mapping=(pxla.ShardedAxis(0),))
    sd = jax.sharding.PmapSharding(self._create_devices(8), ss)
    shape = (8,)
    with jtu.capture_stdout() as output:
      debugging.visualize_sharding(shape, sd)
    expected = _format_multiline("""
    ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
    │ CPU 0 │ CPU 1 │ CPU 2 │ CPU 3 │ CPU 4 │ CPU 5 │ CPU 6 │ CPU 7 │
    └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
    """)
    self.assertEqual(output(), expected)

    ss = pxla.ShardingSpec(
        sharding=(pxla.Unstacked(8), pxla.NoSharding()),
        mesh_mapping=(pxla.ShardedAxis(0),))
    sd = jax.sharding.PmapSharding(self._create_devices(8), ss)
    shape = (8, 2)
    with jtu.capture_stdout() as output:
      debugging.visualize_sharding(shape, sd)
    expected = _format_multiline("""
    ┌───────┐
    │ CPU 0 │
    ├───────┤
    │ CPU 1 │
    ├───────┤
    │ CPU 2 │
    ├───────┤
    │ CPU 3 │
    ├───────┤
    │ CPU 4 │
    ├───────┤
    │ CPU 5 │
    ├───────┤
    │ CPU 6 │
    ├───────┤
    │ CPU 7 │
    └───────┘
    """)
    self.assertEqual(output(), expected)

class InspectShardingTest(jtu.JaxTestCase):

  def test_inspect_sharding_is_called_in_pjit(self):

    if jtu.is_cloud_tpu():
      raise unittest.SkipTest("Inspect sharding is not supported on libtpu.")

    is_called = False
    def _cb(sd):
      nonlocal is_called
      is_called = True
      self.assertIsInstance(sd, jax.sharding.Sharding)
      self.assertLen(sd.device_set, len(jax.devices()))

    def f(x):
      debugging.inspect_array_sharding(x, callback=_cb)
      return jnp.square(x)

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ['dev'])
    spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dev'))
    out_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    f = pjit.pjit(f, in_shardings=spec, out_shardings=out_spec)
    with mesh:
      f(np.arange(8, dtype=jnp.int32))
    self.assertTrue(is_called)

  def test_inspect_sharding_is_called_in_jit(self):

    is_called = False
    def _cb(sd):
      nonlocal is_called
      is_called = True
      self.assertIsInstance(sd, jax.sharding.Sharding)
      self.assertLen(sd.device_set, 1)

    def f_(x):
      debugging.inspect_array_sharding(x, callback=_cb)
      return jnp.square(x)

    f = jax.jit(f_)
    f(np.arange(8, dtype=jnp.int32))
    self.assertTrue(is_called)

    # Test in grad
    is_called = False
    f = jax.jit(jax.grad(lambda x: f_(x).sum()))
    f(np.arange(8, dtype=jnp.float32))
    self.assertTrue(is_called)

  def test_inspect_sharding_3d_input_pos_sharding(self):
    def _cb(sd):
      self.assertIsInstance(sd, jax.sharding.PositionalSharding)
      self.assertLen(sd.device_set, 2)

    def f_(x):
      debugging.inspect_array_sharding(x, callback=_cb)
      return jnp.square(x)

    f = jax.jit(f_)
    mesh = jtu.create_global_mesh((2,), ('x'))
    s = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    arr = jax.device_put(np.arange(8).reshape(2, 2, 2), s)

    f(arr)

  def test_inspect_sharding_3d_input_named_sharding(self):
    def _cb(sd):
      self.assertIsInstance(sd, jax.sharding.NamedSharding)
      self.assertLen(sd.device_set, 2)

    def f_(x):
      debugging.inspect_array_sharding(x, callback=_cb)
      return jnp.square(x)

    f = pjit.pjit(f_)
    mesh = jtu.create_global_mesh((2,), ('x'))
    s = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    arr = jax.device_put(np.arange(8).reshape(2, 2, 2), s)

    with mesh:
      f(arr)


if not rich:
  del VisualizeShardingTest

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
