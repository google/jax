# Copyright 2021 The JAX Authors.
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
import functools

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax._src import core
from jax import lax
from jax._src.pjit import pjit
from jax._src import linear_util as lu
from jax._src import test_util as jtu
from jax._src.lib import xla_client
from jax._src import ad_checkpoint

jax.config.parse_flags_with_absl()

def _get_hlo(f):
  def wrapped(*args, **kwargs):
    c = jax.jit(f).lower(*args, **kwargs).compiler_ir('hlo')
    print_opts = xla_client._xla.HloPrintOptions.short_parsable()
    print_opts.print_metadata = True
    return c.as_hlo_module().to_string(print_opts)
  return wrapped


class NameStackTest(jtu.JaxTestCase):

  def test_trivial_name_stack(self):

    def f(x):
      return x + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), '')

  def test_name_call_name_stack(self):

    @jax.named_call
    def f(x):
      return x + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), 'f')

  def test_manual_name_stack(self):

    @jax.named_scope('foo')
    def f(x):
      return x + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), 'foo')

  def test_nested_name_stack(self):

    @jax.named_scope('foo')
    def f(x):
      with jax.named_scope('bar'):
        return x + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), 'foo/bar')

  def test_multiple_name_stack(self):

    def f(x):
      with jax.named_scope('foo'):
        y = x + 1
      with jax.named_scope('bar'):
        with jax.named_scope('baz'):
          return y + 1
    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'bar/baz')

  def test_call_primitive_jaxpr_should_not_store_outer_name_stack(self):
    @jax.named_scope('foo')
    def f(x):
      @lu.wrap_init
      @jax.named_scope('bar')
      def _f(x):
        return [x + 1]
      return core.call(_f, x)[0]

    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].params['call_jaxpr'].eqns[0].source_info.name_stack), 'bar')

    hlo_text = _get_hlo(f)(2)
    self.assertIn('foo/jit(core_call)/bar', hlo_text)

  def test_jit_jaxpr_should_not_store_outer_name_stack(self):
    @jax.named_scope('foo')
    def f(x):
      @jax.jit
      @jax.named_scope('bar')
      def _f(x):
        return x + 1
      return _f(x)

    jaxpr = jax.make_jaxpr(f)(2).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    jaxpr_param = 'jaxpr'
    self.assertEqual(
        str(jaxpr.eqns[0].params[jaxpr_param].eqns[0].source_info.name_stack),
        'bar')

    hlo_text = _get_hlo(f)(2)
    self.assertIn('foo/jit(_f)/bar', hlo_text)

  def test_pmap_call_primitive_jaxpr_should_not_store_outer_name_stack(self):
    @jax.named_scope('foo')
    @jax.pmap
    def f(x):
      with jax.named_scope('bar'):
        return x + 1
    jaxpr = jax.make_jaxpr(f)(jnp.ones(1)).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(jaxpr.eqns[0].params['call_jaxpr'].eqns[0].source_info.name_stack), 'bar')


class NameStackTransformationTest(jtu.JaxTestCase):

  def test_vmap_should_transform_name_stack(self):
    @jax.vmap
    def f(x):
      with jax.named_scope('foo'):
        return x + 1
    jaxpr = jax.make_jaxpr(f)(jnp.ones(2)).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'vmap(foo)')

  def test_vmap_should_transform_inner_name_stacks(self):
    @jax.named_scope('foo')
    @jax.vmap
    def f(x):
      with jax.named_scope('bar'):
        with jax.named_scope('baz'):
          return x + 1
    jaxpr = jax.make_jaxpr(f)(jnp.ones(2)).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo/vmap(bar)/baz')

  def test_vmap_should_apply_to_call_jaxpr(self):
    @jax.named_scope('foo')
    @jax.vmap
    def f(x):
      @jax.jit
      @jax.named_scope('bar')
      def _f(x):
        return x + 1
      return _f(x)

    jaxpr = jax.make_jaxpr(f)(jnp.ones(2)).jaxpr
    jaxpr_param = 'jaxpr'
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(
        str(jaxpr.eqns[0].params[jaxpr_param].eqns[0].source_info.name_stack),
        'bar')

    hlo_text = _get_hlo(f)(jnp.ones(2))
    self.assertIn('foo/vmap(jit(_f))/bar', hlo_text)

  def test_jvp_should_transform_stacks(self):
    def f(x):
      with jax.named_scope('bar'):
        with jax.named_scope('baz'):
          return jnp.square(x)
    g = jax.named_scope('foo')(lambda x, t: jax.jvp(f, (x,), (t,)))
    jaxpr = jax.make_jaxpr(g)(1., 1.).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack),
                     'foo/jvp(bar)/baz')

  def test_jvp_should_apply_to_call_jaxpr(self):
    @jax.jit
    def f(x):
      with jax.named_scope('bar'):
        with jax.named_scope('baz'):
          return jnp.square(x)
    g = jax.named_scope('foo')(lambda x, t: jax.jvp(f, (x,), (t,)))
    jaxpr = jax.make_jaxpr(g)(1., 1.).jaxpr
    jaxpr_param = 'jaxpr'
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(
        str(jaxpr.eqns[0].params[jaxpr_param].eqns[0].source_info.name_stack),
        'bar/baz')

    hlo_text = _get_hlo(g)(1., 1.)
    self.assertIn('foo/jvp(jit(f))/bar/baz/mul', hlo_text)

  def test_grad_should_add_jvp_and_transpose_to_name_stack(self):
    @jax.value_and_grad
    def f(x):
      with jax.named_scope('foo'):
        return 2 * jnp.sin(x)
    jaxpr = jax.make_jaxpr(f)(1.).jaxpr
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[2].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[4].source_info.name_stack),
        'transpose(jvp(foo))')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('jvp(foo)/sin', hlo_text)
    self.assertIn('jvp(foo)/cos', hlo_text)
    self.assertIn('transpose(jvp(foo))/mul', hlo_text)

  def test_grad_should_add_jvp_and_transpose_to_call_jaxpr(self):
    @jax.value_and_grad
    @jax.named_scope('foo')
    @jax.jit
    def f(x):
      with jax.named_scope('bar'):
        return jnp.sin(x)
    jaxpr = jax.make_jaxpr(f)(1.).jaxpr
    jaxpr_param = 'jaxpr'

    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'transpose(jvp(foo))')
    self.assertEqual(str(
      jaxpr.eqns[0].params[jaxpr_param].eqns[0].source_info.name_stack), 'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params[jaxpr_param].eqns[1].source_info.name_stack), 'bar')
    self.assertEqual(str(
      jaxpr.eqns[1].params[jaxpr_param].eqns[0].source_info.name_stack), 'bar')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('jvp(foo)/jit(f)/bar/sin', hlo_text)
    self.assertIn('jvp(foo)/jit(f)/bar/cos', hlo_text)
    self.assertIn('transpose(jvp(foo))/jit(f)/bar/mul', hlo_text)

  def test_nested_jit_stack(self):

    @jax.value_and_grad
    @jax.jit
    def f(x):
      @jax.jit
      def g(y):
        return jnp.sin(y)
      return g(x)

    hlo_text = _get_hlo(f)(2.)
    self.assertIn('jvp(jit(f))/jit(g)/sin', hlo_text)
    self.assertIn('jvp(jit(f))/jit(g)/cos', hlo_text)
    self.assertIn('transpose(jvp(jit(f)))/jit(g)/mul', hlo_text)

  def test_nested_pjit_stack(self):
    @jax.value_and_grad
    @pjit
    def f(x):
      @pjit
      def g(y):
        return jnp.sin(y)
      return g(x)

    hlo_text = _get_hlo(f)(2.)
    self.assertIn('jvp(pjit(f))/pjit(g)/sin', hlo_text)
    self.assertIn('jvp(pjit(f))/pjit(g)/cos', hlo_text)
    self.assertIn('transpose(jvp(pjit(f)))/pjit(g)/mul', hlo_text)

  def test_remat_appears_in_hlo(self):
    @ad_checkpoint.remat
    def f(x):
      return jnp.sin(x)

    hlo_text = _get_hlo(f)(2.)
    hlo_text_grad = _get_hlo(jax.grad(f))(2.)
    self.assertNotIn('rematted_computation', hlo_text)
    self.assertNotIn('remat', hlo_text)
    self.assertIn('checkpoint', hlo_text)
    self.assertIn('rematted_computation', hlo_text_grad)


class NameStackControlFlowTest(jtu.JaxTestCase):

  def test_while_loop_body_should_not_have_name_stack(self):

    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('bar')
      def body(x):
        return x + 1
      @jax.named_scope('bar_cond')
      def cond(x):
        return x < 5
      return lax.while_loop(cond, body, x)
    jaxpr = jax.make_jaxpr(f)(0)
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(
      jaxpr.eqns[0].params['body_jaxpr'].eqns[0].source_info.name_stack),
      'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['cond_jaxpr'].eqns[0].source_info.name_stack),
      'bar_cond')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('foo/while/body/bar', hlo_text)
    self.assertIn('foo/while/cond/bar_cond', hlo_text)

  def test_vmap_of_while_loop_should_transform_name_stack(self):

    @jax.vmap
    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('bar')
      def body(x):
        return x + 1
      @jax.named_scope('bar_cond')
      def cond(x):
        return x < 5
      return lax.while_loop(cond, body, x)
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2))
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'vmap(foo)')
    self.assertEqual(str(
      jaxpr.eqns[0].params['body_jaxpr'].eqns[0].source_info.name_stack),
      'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['cond_jaxpr'].eqns[0].source_info.name_stack),
      'bar_cond')

    hlo_text = _get_hlo(f)(jnp.arange(2.))
    self.assertIn('vmap(foo)/while/body/bar/add', hlo_text)
    self.assertIn('vmap(foo)/while/cond/bar_cond/lt', hlo_text)

  def test_jvp_of_while_loop_transforms_name_stack(self):

    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('bar')
      def body(x):
        return x + 1.
      @jax.named_scope('bar_cond')
      def cond(x):
        return x < 5.
      return lax.while_loop(cond, body, x)
    g = lambda x, t: jax.jvp(f, (x,), (t,))
    jaxpr = jax.make_jaxpr(g)(1., 1.)
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(
      jaxpr.eqns[0].params['body_jaxpr'].eqns[0].source_info.name_stack),
      'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['cond_jaxpr'].eqns[0].source_info.name_stack),
      'bar_cond')

    hlo_text = _get_hlo(g)(1., 1.)
    self.assertIn('jvp(foo)/while/body/bar/add', hlo_text)
    self.assertIn('jvp(foo)/while/cond/bar_cond/lt', hlo_text)

  def test_vmap_of_jvp_of_while_loop_transforms_name_stack(self):

    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('bar')
      def body(x):
        return x + 1.
      @jax.named_scope('bar_cond')
      def cond(x):
        return x < 5.
      return lax.while_loop(cond, body, x)
    g = jax.vmap(lambda x, t: jax.jvp(f, (x,), (t,)))
    jaxpr = jax.make_jaxpr(g)(jnp.arange(2.), jnp.ones(2))
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'vmap(jvp(foo))')
    self.assertEqual(str(
      jaxpr.eqns[0].params['body_jaxpr'].eqns[0].source_info.name_stack),
      'bar')
    self.assertEqual(str(
      jaxpr.eqns[0].params['cond_jaxpr'].eqns[0].source_info.name_stack),
      'bar_cond')

    hlo_text = _get_hlo(g)(jnp.arange(2.), jnp.ones(2))
    self.assertIn('vmap(jvp(foo))/while/body/bar/add', hlo_text)
    self.assertIn('vmap(jvp(foo))/while/body_pred/bar_cond', hlo_text)


  def test_cond_body_should_not_have_name_stack(self):

    @jax.named_scope('foo')
    def f(x, y):
      @jax.named_scope('true')
      def true_fn(x):
        return x + 1
      @jax.named_scope('false')
      def false_fn(x):
        return x - 1
      return lax.cond(y, true_fn, false_fn, x)
    jaxpr = jax.make_jaxpr(f)(0, True)
    for eqn in jaxpr.eqns:
      self.assertEqual(str(eqn.source_info.name_stack), 'foo')
      if eqn.primitive is lax.cond_p:
        self.assertEqual(str(
          eqn.params['branches'][0].eqns[0].source_info.name_stack),
          'false')
        self.assertEqual(str(
          eqn.params['branches'][1].eqns[0].source_info.name_stack),
          'true')

    hlo_text = _get_hlo(f)(1, True)
    self.assertIn('foo/cond/branch_0_fun/false/sub', hlo_text)
    self.assertIn('foo/cond/branch_1_fun/true/add', hlo_text)

  def test_vmap_of_cond_should_transform_name_stack(self):

    @jax.named_scope('foo')
    @functools.partial(jax.vmap, in_axes=(0, None))
    def f(x, y):
      @jax.named_scope('true')
      def true_fn(x):
        return x + 1
      @jax.named_scope('false')
      def false_fn(x):
        return x - 1
      return lax.cond(y, true_fn, false_fn, x)
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2), True)
    for eqn in jaxpr.eqns:
      self.assertIn('foo', str(eqn.source_info.name_stack))
      if eqn.primitive is lax.cond_p:
        self.assertEqual(str(
          eqn.params['branches'][0].eqns[0].source_info.name_stack),
          'false')
        self.assertEqual(str(
          eqn.params['branches'][1].eqns[0].source_info.name_stack),
          'true')

    hlo_text = _get_hlo(f)(jnp.arange(2.), True)
    self.assertIn('foo/vmap(cond)/branch_0_fun/false/sub', hlo_text)
    self.assertIn('foo/vmap(cond)/branch_1_fun/true/add', hlo_text)

  def test_jvp_of_cond_transforms_name_stack(self):

    @jax.named_scope('foo')
    def f(x, y):
      @jax.named_scope('true')
      def true_fn(x):
        return x + 1
      @jax.named_scope('false')
      def false_fn(x):
        return x - 1
      return lax.cond(y, true_fn, false_fn, x)
    f_ = lambda x: jax.jit(f)(x, True)
    g = lambda x, t: jax.jvp(f_, (x,), (t,))
    jaxpr = jax.make_jaxpr(g)(jnp.arange(2.), jnp.ones(2))
    jaxpr_param = 'jaxpr'
    call_jaxpr = jaxpr.jaxpr.eqns[0].params[jaxpr_param]
    self.assertEqual(str(call_jaxpr.eqns[1].source_info.name_stack), 'foo')
    self.assertEqual(str(
      call_jaxpr.eqns[1].params['branches'][0].eqns[0].source_info.name_stack),
      'false')
    self.assertEqual(str(
      call_jaxpr.eqns[1].params['branches'][1].eqns[0].source_info.name_stack),
      'true')

    hlo_text = _get_hlo(g)(jnp.arange(2.), jnp.ones(2))
    self.assertIn('jvp(jit(f))/foo/cond/branch_0_fun/false/sub', hlo_text)
    self.assertIn('jvp(jit(f))/foo/cond/branch_1_fun/true/add', hlo_text)

  def test_vmap_of_jvp_of_cond_transforms_name_stack(self):

    @jax.named_scope('foo')
    def f(x, y):
      @jax.named_scope('true')
      def true_fn(x):
        return x + 1
      @jax.named_scope('false')
      def false_fn(x):
        return x - 1
      return lax.cond(y, true_fn, false_fn, x)
    f_ = lambda x: jax.jit(f)(x, True)
    g = jax.vmap(lambda x, t: jax.jvp(f_, (x,), (t,)))
    jaxpr = jax.make_jaxpr(g)(jnp.arange(2.), jnp.ones(2))
    jaxpr_param = 'jaxpr'

    call_jaxpr = jaxpr.jaxpr.eqns[0].params[jaxpr_param]
    self.assertEqual(str(
      call_jaxpr.eqns[1].params['branches'][0].eqns[0].source_info.name_stack),
      'false')
    self.assertEqual(str(
      call_jaxpr.eqns[1].params['branches'][1].eqns[0].source_info.name_stack),
      'true')

    hlo_text = _get_hlo(g)(jnp.arange(2.), jnp.ones(2))
    self.assertIn(
        'vmap(jvp(jit(f)))/foo/cond/branch_0_fun/false/sub"',
        hlo_text)
    self.assertIn(
        'vmap(jvp(jit(f)))/foo/cond/branch_1_fun/true/add"',
        hlo_text)

  def test_grad_of_cond_transforms_name_stack(self):

    @jax.value_and_grad
    @jax.named_scope('foo')
    def f(x, y):
      @jax.named_scope('true')
      def true_fn(x):
        return x * x * 2.

      @jax.named_scope('false')
      def false_fn(x):
        return x / jnp.square(x)

      return lax.cond(y, true_fn, false_fn, x)

    jaxpr = jax.make_jaxpr(f)(1., True)
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[2].source_info.name_stack),
        'transpose(jvp(foo))')

    hlo_text = _get_hlo(f)(1., True)
    self.assertIn(
        'jvp(foo)/cond/branch_0_fun/false/div',
        hlo_text)
    self.assertIn(
        'jvp(foo)/cond/branch_1_fun/true/mul',
        hlo_text)
    self.assertIn(
        'transpose(jvp(foo))/cond/branch_0_fun/false/div',
        hlo_text)
    self.assertIn(
        'transpose(jvp(foo))/cond/branch_1_fun/true/mul',
        hlo_text)

  def test_vmap_of_grad_of_cond_transforms_name_stack(self):

    @functools.partial(jax.vmap, in_axes=(0, None))
    @jax.value_and_grad
    @jax.named_scope('foo')
    def f(x, y):
      @jax.named_scope('true')
      def true_fn(x):
        return x * x * 2.
      @jax.named_scope('false')
      def false_fn(x):
        return x / x / 2.
      return lax.cond(y, true_fn, false_fn, x)
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2.), True)
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'vmap(jvp(foo))')
    self.assertEqual(str(jaxpr.eqns[2].source_info.name_stack),
        'vmap(transpose(jvp(foo)))')

    hlo_text = _get_hlo(f)(jnp.arange(2.), True)
    self.assertIn(
        'vmap(jvp(foo))/cond/branch_0_fun/false/div',
        hlo_text)
    self.assertIn(
        'vmap(jvp(foo))/cond/branch_1_fun/true/mul',
        hlo_text)
    self.assertIn(
        'vmap(transpose(jvp(foo)))/cond/branch_0_fun/false/div',
        hlo_text)
    self.assertIn(
        'vmap(transpose(jvp(foo)))/cond/branch_1_fun/true/mul',
        hlo_text)

  def test_scan_body_should_not_have_name_stack(self):

    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('scan_body')
      def body(carry, x):
        return carry + x, carry + x
      return lax.scan(body, x, jnp.arange(5, dtype='float32'))
    jaxpr = jax.make_jaxpr(f)(jnp.float32(1))
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'foo')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('foo/while/body/scan_body', hlo_text)

  def test_vmap_of_scan_should_transform_stack(self):

    @jax.vmap
    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('scan_body')
      def body(carry, x):
        return carry + x, carry + x
      return lax.scan(body, x, jnp.arange(8.))
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2.))
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'vmap(foo)')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(f)(jnp.arange(2.))
    self.assertIn('vmap(foo)/while/body/scan_body/add', hlo_text)

  def test_jvp_of_scan_should_transform_stack(self):

    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('scan_body')
      def body(carry, x):
        return carry + x, carry + x
      return lax.scan(body, x, jnp.arange(8, dtype='float32'))
    g = lambda x, t: jax.jvp(f, (x,), (t,))
    jaxpr = jax.make_jaxpr(g)(jnp.float32(1), jnp.float32(1))
    self.assertEqual(str(jaxpr.eqns[0].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(g)(1., 1.)
    self.assertIn('jvp(foo)/while/body/scan_body/add', hlo_text)

  def test_grad_of_scan_should_transform_stack(self):

    @jax.value_and_grad
    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('scan_body')
      def body(carry, x):
        return 2 * carry * x, carry + x
      return lax.scan(body, x, jnp.arange(8., dtype='float32'))[0]
    jaxpr = jax.make_jaxpr(f)(jnp.float32(2))
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'jvp(foo)')
    self.assertEqual(str(jaxpr.eqns[2].source_info.name_stack),
        'transpose(jvp(foo))')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(f)(1.)
    self.assertIn('jvp(foo)/while/body/scan_body/mul', hlo_text)
    self.assertIn('transpose(jvp(foo))/while/body/scan_body/mul', hlo_text)

  def test_vmap_of_grad_of_scan_should_transform_stack(self):

    @jax.vmap
    @jax.value_and_grad
    @jax.named_scope('foo')
    def f(x):
      @jax.named_scope('scan_body')
      def body(carry, x):
        return carry * x, carry + x
      return lax.scan(body, x, jnp.arange(8.))[0]
    jaxpr = jax.make_jaxpr(f)(jnp.arange(2.))
    self.assertEqual(str(jaxpr.eqns[1].source_info.name_stack), 'vmap(jvp(foo))')
    self.assertEqual(str(jaxpr.eqns[2].source_info.name_stack),
        'vmap(transpose(jvp(foo)))')
    self.assertEqual(str(
      jaxpr.eqns[1].params['jaxpr'].eqns[0].source_info.name_stack),
      'scan_body')

    hlo_text = _get_hlo(f)(jnp.arange(2.))
    self.assertIn('vmap(jvp(foo))/while/body/scan_body/mul', hlo_text)
    self.assertIn('vmap(transpose(jvp(foo)))/while/body/scan_body/mul', hlo_text)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
