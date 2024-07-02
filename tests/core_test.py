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

import unittest
from collections import namedtuple
from functools import partial
import gc
import operator

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
from jax import numpy as jnp
from jax import jvp, linearize, vjp, jit, make_jaxpr
from jax.api_util import flatten_fun_nokwargs
from jax._src import config

from jax._src import core
from jax._src import linear_util as lu
from jax._src import util
from jax._src import test_util as jtu
from jax._src.core import UnshapedArray, ShapedArray, DBIdx
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.lax import control_flow as lax_control_flow

config.parse_flags_with_absl()

_ = pe.PartialVal.unknown(UnshapedArray(np.float32))
__ = pe.PartialVal.unknown(ShapedArray((), np.float32))

def call(f, *args):
  return jit(f)(*args)

@util.curry
def core_call(f, *args):
  args, in_tree = jax.tree.flatten(args)
  f, out_tree = flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
  out = core.call_p.bind(f, *args)
  return jax.tree.unflatten(out_tree(), out)

@util.curry
def core_closed_call(f, *args):
  args, in_tree = jax.tree.flatten(args)
  f, out_tree = flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
  out = core.closed_call_p.bind(f, *args)
  return jax.tree.unflatten(out_tree(), out)

def simple_fun(x, y):
  return jnp.sin(x * y)

def simple_fun_fanout(x, y):
  return jnp.sin(x * y) * x

def fun_with_call(x):
  return call(jnp.sin, x)

def fun_with_nested_calls(x):
  def f(y):
    y2 = jnp.sin(y) + 1.0 + (2.0 * x)

    @jit
    def g(z):
      return y2 * z * x + (x * y)

    return call(g, y)

  return call(f, x)

def error(*args):
  def f(*args):
    assert False
  return f

def fun_with_nested_calls_2(x):
  def bar(y):
    def baz(w):
      q = call(lambda x: y, x)
      q = q + call(lambda: y)
      q = q + call(lambda y: w + y, y)
      q = call(lambda w: call(jnp.sin, x) * y, 1.0) + q
      return q
    p, t = jvp(baz, (x + 1.0,), (y,))
    return t + (x * p)
  return call(bar, x)

def fun_call_jitted(x):
  @jit
  def g(z):
    return x * z

  return call(g, x)

def fun_with_two_calls(x):
  return call(jnp.sin, x) + call(jnp.cos, x)

def fun_with_call_closure(x):
  def foo(y, z):
    return (x * x) * jnp.sin(y) * z

  return call(foo, x, jnp.cos(x)) + x

def product_io_fun(x, y):
  xa = x['a']
  xb = x['b']
  y1, (y2, y3) = y
  return jnp.sin(xa + y2), [xb, (y1, y3)]


_rng = np.random.RandomState(42)
R = _rng.randn
CallSpec = namedtuple('CallSpec', ['fun', 'args'])
test_specs_base = [
    CallSpec(simple_fun, (R(3, 2), R(3, 2))),
    CallSpec(simple_fun_fanout, (R(3, 2), R(3, 2))),
    CallSpec(product_io_fun, ({'a': R(2, 2), 'b': R(2, 2)},
                              (R(2, 2), (R(2, 2), R(2, 2))))),
    CallSpec(fun_with_call, (R(3, 2),)),
    CallSpec(fun_with_two_calls, (R(3, 2),)),
    CallSpec(fun_with_call_closure, (R(3, 2),)),
    CallSpec(fun_call_jitted, (R(1,),)),
    CallSpec(fun_with_nested_calls, (R(),)),
    CallSpec(fun_with_nested_calls, (R(3, 2),)),
    CallSpec(fun_with_nested_calls_2, (R(1, 2),)),
]

def jvp_unlinearized(f, primals, tangents):
  out, jvp = linearize(f, *primals)
  return out, jvp(*tangents)

test_specs = []
for ts in test_specs_base:
  test_specs.append(ts)
  test_specs.append(CallSpec(partial(jvp, ts.fun), (ts.args, ts.args)))
  test_specs.append(CallSpec(jit(ts.fun), ts.args))
  test_specs.append(CallSpec(jit(jit(ts.fun)), ts.args))
  test_specs.append(CallSpec(core_call(ts.fun), ts.args))
  test_specs.append(CallSpec(core_call(jit(ts.fun)), ts.args))
  test_specs.append(CallSpec(core_call(core_call(ts.fun)), ts.args))
  test_specs.append(CallSpec(core_closed_call(ts.fun), ts.args))
  test_specs.append(CallSpec(core_closed_call(jit(ts.fun)), ts.args))
  test_specs.append(CallSpec(core_closed_call(core_closed_call(ts.fun)), ts.args))
  test_specs.append(CallSpec(partial(jvp_unlinearized, ts.fun),
                             (ts.args, ts.args)))


def fwd_deriv(f):
  def df(x):
    return jvp(f, (x,), (1.0,))[1]

  return df


class CoreTest(jtu.JaxTestCase):

  def test_tree_map(self):
    xs = ({'a': 1}, [2, 3])
    ys = ({'a': 10}, [20, 30])
    ys_bad = ({'a': 10, 'b': 10}, [20, 30])
    zs = ({'a': 11}, [22, 33])

    f = lambda x, y: x + y
    assert jax.tree.map(f, xs, ys) == zs
    try:
      jax.tree.map(f, xs, ys_bad)
      assert False
    except (TypeError, ValueError):
      pass

  def test_tree_flatten(self):
    flat, _ = jax.tree.flatten(({'a': 1}, [2, 3], 4))
    assert flat == [1, 2, 3, 4]

  def test_tree_unflatten(self):
    tree = [(1, 2), {"roy": (3, [4, 5, ()])}]
    flat, treedef = jax.tree.flatten(tree)
    assert flat == [1, 2, 3, 4, 5]
    tree2 = jax.tree.unflatten(treedef, flat)
    nodes_equal = jax.tree.map(operator.eq, tree, tree2)
    assert jax.tree.reduce(operator.and_, nodes_equal)

  @jtu.sample_product(
      dtype=[*jtu.dtypes.all, object, [('i', 'i4'), ('f', 'f4')]]
  )
  def test_is_valid_jaxtype(self, dtype):
    arr = np.zeros(10, dtype=dtype)
    if dtype in jtu.dtypes.all:
      self.assertTrue(core.valid_jaxtype(arr))
    else:
      self.assertFalse(core.valid_jaxtype(arr))

  @parameterized.named_parameters(
      (str(i), *spec) for i, spec in enumerate(test_specs))
  def test_jit(self, f, args):
    jtu.check_close(jit(f)(*args), f(*args))

  @parameterized.named_parameters(
      (str(i), *spec) for i, spec in enumerate(test_specs))
  def test_jvp(self, f, args):
    jtu.check_jvp(f, partial(jvp, f), args, rtol={np.float32: 3e-2})

  def test_jvp_zeros(self):
    def foo(x):
      def bar(y):
        return jnp.sin(x * y)
      return jvp(bar, (3 * x,), (2 * x,))

    jtu.check_eq(jit(foo)(0.5), foo(0.5))

  @parameterized.parameters(test_specs)
  def test_jvp_linearized(self, f, args):
    jtu.check_jvp(f, partial(jvp_unlinearized, f), args,
                  rtol={np.float32: 3e-2})

  @parameterized.named_parameters(
      (str(i), *spec) for i, spec in enumerate(test_specs))
  def test_vjp(self, f, args):
    jtu.check_vjp(f, partial(vjp, f), args,
                  rtol={np.float32: 3e-1, np.float64: 1e-5},
                  atol={np.float32: 1e-2, np.float64: 1e-5})

  def test_jvp_closure(self):
    def foo(x):
      def bar(y):
        return jnp.multiply(x, y)
      return jvp(bar, (3.0,), (1.0,))[1]
    ans = jvp(foo, (1.0,), (2.0,))
    assert ans == (1.0, 2.0), ans

  def test_jit_closure(self):
    def foo(x):
      @jit
      def bar(y):
        return x + y
      return bar(0.0)
    assert jvp(foo, (1.0,), (2.0,)) == (1.0, 2.0)

  def test_simple_jit(self):
    def foo(x):
      if x.shape == ():
        return x + 1.
      else:
        return x + 2.

    foo2 = jit(foo)
    foo3 = jit(foo2)

    x1, y1 = np.array(1.0), np.array(2.0)
    assert foo(x1) == y1
    assert foo2(x1) == y1
    assert foo3(x1) == y1

    x2, y2 = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    assert np.all(foo(x2) == y2)
    assert np.all(foo2(x2) == y2)
    assert np.all(foo3(x2) == y2)

  def test_product_jit(self):
    def foo(x, tup):
      y, z = tup
      w = x + z
      return (w, {'x': y}), z

    foo2 = jit(foo)
    foo3 = jit(foo2)

    args = (1.0, (2.0, 3.0))
    expected_output = ((4.0, {'x': 2.0}), 3.0)

    assert foo(*args) == expected_output
    assert foo2(*args) == expected_output
    assert foo3(*args) == foo(*args)

  def test_jvp_repeated_fwd(self):
    d_sin = fwd_deriv(jnp.sin)
    d2_sin = fwd_deriv(d_sin)
    d3_sin = fwd_deriv(d2_sin)

    assert d_sin(0.0) == 1.0
    assert d2_sin(0.0) == 0.0
    assert d3_sin(0.0) == -1.0

  def test_reference_cycles(self):
    gc.collect()

    def f(x):
      return x.sum()

    fn = partial(linearize, f)
    params = jnp.zeros([])

    debug = gc.get_debug()
    try:
      fn(params)
      gc.set_debug(gc.DEBUG_SAVEALL)
      self.assertEqual(gc.collect(), 0, msg=str(gc.garbage))
    finally:
      gc.set_debug(debug)

  def test_reference_cycles_jit(self):
    gc.collect()

    def f(x):
      return x.sum()

    fn = jit(f)
    params = jnp.zeros([])

    debug = gc.get_debug()
    try:
      fn(params).block_until_ready()
      gc.set_debug(gc.DEBUG_SAVEALL)
      self.assertEqual(gc.collect(), 0, msg=str(gc.garbage))
    finally:
      gc.set_debug(debug)

  def test_invalid_shape_error_with_jit_tracer_passed(self):
    @jax.jit
    def g_jit(x):
      return jnp.zeros(shape=(2, x))

    @jax.vmap
    def g_vmap(x):
      return jnp.zeros(shape=(2, x))

    with self.assertRaisesRegex(
        TypeError,
        'This concrete value was not available in'
        + ' Python because it depends on',
    ):
      g_jit(1)

    with self.assertRaisesRegex(TypeError,
          'This BatchTracer with object id'):
      g_vmap(jnp.ones((1, )))

  def test_concrete_array_string_representation(self):
    # https://github.com/google/jax/issues/5364
    self.assertEqual(
        str(core.ConcreteArray(np.dtype(np.int32),
                               np.array([1], dtype=np.int32))),
        'ConcreteArray([1], dtype=int32)')

  def test_dropvar_avals(self):
    def f(x):
      def body(c, _):
        return c, None
      (x1, x2), _ = jax.lax.scan(body, (x, x), None, length=1)
      return [x2]

    aval = core.ShapedArray((), jnp.dtype('int32'))
    pval = pe.PartialVal.unknown(aval)
    jaxpr, _, _ = pe.trace_to_jaxpr_nounits(lu.wrap_init(f), [pval], False)
    dropvar, b = jaxpr.eqns[0].outvars
    self.assertEqual(dropvar.aval, aval)

  def test_input_residual_forwarding(self):
    # https://github.com/google/jax/pull/11151
    x = jnp.arange(3 * 4.).reshape(3, 4)
    y = jnp.arange(4 * 3.).reshape(4, 3)

    g = jax.jit(jnp.dot)

    def f(y):
      z, g_lin = jax.linearize(lambda y: g(x, y), y)
      zdot = g_lin(y)
      return z, zdot

    jaxpr = jax.make_jaxpr(f)(y)
    e1, e2 = jaxpr.jaxpr.eqns
    self.assertLen(e1.outvars, 1)  # only primal out, no residuals
    self.assertEqual(e1.outvars[0].aval.shape, (3, 3))  # only primal out shape


@jtu.with_config(jax_pprint_use_color=False)
class JaxprTypeChecks(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    lax_control_flow._initial_style_open_jaxpr.cache_clear()
    lax_control_flow._initial_style_jaxpr.cache_clear()
    lax_control_flow._initial_style_jaxprs_with_common_consts.cache_clear()

  def test_check_jaxpr_correct(self):
    jaxpr = make_jaxpr(lambda x: jnp.sin(x) + jnp.cos(x))(1.).jaxpr
    core.check_jaxpr(jaxpr)

  def test_check_jaxpr_cond_correct(self):
    jaxpr = make_jaxpr(lambda x: lax.switch(0, [jnp.sin, jnp.cos], x))(1.).jaxpr
    core.check_jaxpr(jaxpr)

  def test_check_jaxpr_jit_invalid(self):
    jaxpr = make_jaxpr(jax.jit(lambda x, y: x + 1))(1., 2.).jaxpr
    pjit_eqn, = jaxpr.eqns
    jaxpr._eqns[0] = pjit_eqn.replace(invars=())
    self.assertRaisesRegex(
        core.JaxprTypeError,
        '0 operands cannot call jaxpr with 2 inputs',
        lambda: core.check_jaxpr(jaxpr))

  def test_check_jaxpr_cond_invalid(self):
    jaxpr = make_jaxpr(lambda x: lax.switch(0, [jnp.sin, jnp.cos], x))(1.).jaxpr
    cond = next(eqn for eqn in jaxpr.eqns if eqn.primitive.name == 'cond')
    cond.params['branches'][0].jaxpr._invars = ()
    self.assertRaisesRegex(
        core.JaxprTypeError,
        'cond branch 0 takes 0 inputs, branch 1 takes 1',
        lambda: core.check_jaxpr(jaxpr))

  def test_check_jaxpr_scan_correct(self):
    def f(c, x):
      b = jnp.cos(jnp.sum(jnp.sin(x)) + jnp.sum(jnp.cos(c)))
      c = jnp.sin(c * b)
      return c, b
    xs = jnp.ones((5, 3))
    c = jnp.ones(4)
    jaxpr = make_jaxpr(partial(lax.scan, f))(c, xs).jaxpr
    core.check_jaxpr(jaxpr)

  def test_check_jaxpr_invalid_long(self):
    # jaxprs can be large, and this tests that when large ones are printed for
    # context in jaxpr typechecking errors, they're not printed entirely

    def enlarge(f, n):
      def g(x):
        for _ in range(n):
          x = x + x
        x = f(x)
        for _ in range(n):
          x = x + x
        return x
      return g

    jaxpr = make_jaxpr(enlarge(
        lambda x: lax.switch(0, [jnp.sin, jnp.cos], x), 100))(1.).jaxpr

    cond = next(eqn for eqn in jaxpr.eqns if eqn.primitive.name == 'cond')
    cond.params['branches'][0].jaxpr._invars = ()
    msg = ''
    try:
      core.check_jaxpr(jaxpr)
    except core.JaxprTypeError as e:
      msg, = e.args

    self.assertIn('cond branch 0 takes 0 inputs, branch 1 takes 1', msg)
    self.assertIn('in equation:', msg)
    self.assertIn('from source:', msg)
    self.assertIn('while checking jaxpr:', msg)
    self.assertLess(msg.count('\n'), 200)

  def test_check_jaxpr_eqn_mismatch(self):
    def f(x):
      return jnp.sin(x) + jnp.cos(x)

    def new_jaxpr():
      return make_jaxpr(f)(jnp.float32(1.)).jaxpr

    # jaxpr is:
    #
    # { lambda  ; a.
    #   let b = sin a
    #       c = cos a
    #       d = add b c
    #   in (d,) }
    #
    # NB: eqns[0].outvars[0] and eqns[2].invars[0] are both 'b'

    jaxpr = new_jaxpr()
    # int, not float!
    jaxpr.eqns[0].outvars[0].aval = core.ShapedArray((), jnp.dtype(jnp.int32))
    self.assertRaisesRegex(
        core.JaxprTypeError,
        r"Value for variable 'b' inconsistently typed as f32\[\] "
        r"for let-binder of type i32\[\]\n\nin equation:\n\nb:i32\[\] = sin a",
        lambda: core.check_jaxpr(jaxpr))

    jaxpr = new_jaxpr()
    jaxpr.eqns[0].outvars[0].aval = core.ShapedArray((2, 3),
                                                     jnp.dtype(jnp.float32))
    self.assertRaisesRegex(
        core.JaxprTypeError,
        r"Value for variable 'b' inconsistently typed as f32\[\] "
        r"for let-binder of type f32\[2,3\]\n\nin equation:\n\nb:f32\[2,3\] = sin a",
        lambda: core.check_jaxpr(jaxpr))

  def test_jaxpr_dropvar_from_jit_call(self):
    def inner(x):
      return x + 1, x + 2

    def f(x):
      _, y = jit(inner)(x)
      return y + 3

    jaxpr = make_jaxpr(f)(1).jaxpr
    assert isinstance(jaxpr.eqns[0].outvars[0], core.DropVar)
    core.check_jaxpr(jaxpr)

  def test_jaxpr_dropvar_from_loop(self):
    def f(x):
      _, y = lax.while_loop(lambda s: s[0] < 0.,
                            lambda s: (jnp.sin(s[0]), jnp.cos(s[1])),
                            (x, x))
      return y + 1.

    jaxpr = make_jaxpr(f)(1.).jaxpr
    assert isinstance(jaxpr.eqns[0].outvars[0], core.DropVar)
    core.check_jaxpr(jaxpr)

  def test_jaxpr_dropvar_from_cond(self):
    def f(x):
      _, y = lax.cond(x < 0.,
                      lambda x: (jnp.sin(x), x + 1.),
                      lambda x: (jnp.cos(x), x + 2.),
                      x)
      return y

    jaxpr = make_jaxpr(f)(1.).jaxpr
    assert isinstance(jaxpr.eqns[-1].outvars[0], core.DropVar)
    core.check_jaxpr(jaxpr)

  def test_jaxpr_undefined_eqn_invar(self):
    jaxpr = make_jaxpr(lambda x: jnp.sin(x) + jnp.cos(x))(1.).jaxpr
    cos = next(eqn for eqn in jaxpr.eqns if eqn.primitive.name == 'cos')
    cos.invars[0] = core.gensym(suffix='_test')(cos.invars[0].aval)
    self.assertRaisesRegex(
        core.JaxprTypeError,
        r"Variable '.+_test' not defined\n\nin equation:",
        lambda: core.check_jaxpr(jaxpr))

  @parameterized.parameters(
    {'value': 0, 'weak_type': True},
    {'value': np.int32(0), 'weak_type': False},
    {'value': np.array([0]), 'weak_type': False}
  )
  def test_raise_to_shaped_weak_type(self, value, weak_type):
    aval = core.raise_to_shaped(core.get_aval(value))
    self.assertEqual(aval.weak_type, weak_type)

  def test_lattice_join_named_shape(self):
    aval1 = core.ShapedArray((2, 3), np.float32, False, {'i': 10})
    self.assertEqual(core.lattice_join(aval1, aval1), aval1)

    aval2 = core.ShapedArray((2, 3), np.float32, False, {'j': 5})
    expected = core.ShapedArray((2, 3), np.float32, False, {'i': 10, 'j': 5})
    self.assertEqual(core.lattice_join(aval1, aval2), expected)

    aval3 = core.ShapedArray((2, 3), np.float32, False, {'i': 5})
    self.assertRaises(TypeError, lambda: core.lattice_join(aval1, aval3))

  def test_typecompat_named_shape(self):
    aval1 = core.ShapedArray((2, 3), np.float32, False, {'i': 10})
    aval2 = core.ShapedArray((2, 3), np.float32, False, {'j': 5})
    self.assertTrue(core.typecompat(aval1, aval2))

    aval3 = core.ShapedArray((2, 3), np.float32, False, {'i': 5})
    self.assertFalse(core.typecompat(aval1, aval3))

  def test_named_shape_comparision(self):
    self.assertTrue(core.NamedShape(2, 3) == (2, 3))
    self.assertFalse(core.NamedShape(2, i=3) == (2,))
    self.assertFalse(core.NamedShape(2, i=3) == (2, 3))
    self.assertFalse(core.NamedShape(2, i=3) == None)
    self.assertFalse(core.NamedShape() == [])


@jtu.with_config(jax_dynamic_shapes=True)
class DynamicShapesTest(jtu.JaxTestCase):

  def test_staging_basic(self):
    n = core.ShapedArray((), jnp.dtype('int32'), weak_type=False)
    a = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)
    b = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)

    @lu.wrap_init
    def f(x, y):
      return x, y

    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      f, [n, a, b], keep_inputs=[False, True, True])

    self.assertLen(jaxpr.invars, 3)
    self.assertEqual((jaxpr.invars[0],), jaxpr.invars[1].aval.shape)
    self.assertEqual((jaxpr.invars[0],), jaxpr.invars[2].aval.shape)

    self.assertLen(jaxpr.outvars, 2)
    self.assertEqual((jaxpr.invars[0],), jaxpr.outvars[0].aval.shape)
    self.assertEqual((jaxpr.invars[0],), jaxpr.outvars[1].aval.shape)

  @unittest.skip('This test does not work with nested pjit and DShapedArray')
  def test_staging_nested(self):
    n = core.ShapedArray((), jnp.dtype('int32'), weak_type=False)
    a = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)
    b = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)

    @lu.wrap_init
    def f(x, y):
      @jax.jit
      def g(x, y, z, w):
        return (x, w)
      return g(x, y, x, y)

    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      f, [n, a, b], keep_inputs=[False, True, True])

    self.assertLen(jaxpr.invars, 1 + 2)  # one axis size var, two other inputs
    self.assertEqual((jaxpr.invars[0],), jaxpr.invars[1].aval.shape)
    self.assertEqual((jaxpr.invars[0],), jaxpr.invars[2].aval.shape)

    self.assertLen(jaxpr.outvars, 2)
    self.assertEqual((jaxpr.invars[0],), jaxpr.outvars[0].aval.shape)
    self.assertEqual((jaxpr.invars[0],), jaxpr.outvars[1].aval.shape)

    self.assertLen(jaxpr.eqns, 1)
    eqn = jaxpr.eqns[0]
    self.assertIsInstance(eqn.primitive, core.CallPrimitive)
    inner_jaxpr = eqn.params['call_jaxpr']
    self.assertIsInstance(inner_jaxpr, core.Jaxpr)

    self.assertLen(inner_jaxpr.invars, 1 + 4)  # one axis size var
    self.assertEqual((inner_jaxpr.invars[0],), inner_jaxpr.invars[1].aval.shape)
    self.assertEqual((inner_jaxpr.invars[0],), inner_jaxpr.invars[2].aval.shape)
    self.assertEqual((inner_jaxpr.invars[0],), inner_jaxpr.invars[3].aval.shape)
    self.assertEqual((inner_jaxpr.invars[0],), inner_jaxpr.invars[4].aval.shape)

  @unittest.skip('This test does not work with nested pjit and DShapedArray')
  def test_staging_nested_including_shape_arg(self):
    n = core.ShapedArray((), jnp.dtype('int32'), weak_type=False)
    a = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)
    b = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)

    @lu.wrap_init
    def f(x, y):
      @jax.jit
      def g(_, x, y, z, w):
        return (x, w)
      return g(x.shape[0], x, y, x, y)

    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      f, [n, a, b], keep_inputs=[False, True, True])

    # { lambda ; a:i32[] b:f32[a] c:f32[a]. let
    #     d:f32[a] e:f32[a] = xla_call[
    #       call_jaxpr={ lambda ; f:i32[] g:i32[] h:f32[f] i:f32[f] j:f32[f] k:f32[f]. let
    #
    #         in (h, k) }
    #       name=g
    #     ] a a b c b c
    #   in (d, e) }

    self.assertLen(jaxpr.eqns, 1)
    eqn = jaxpr.eqns[0]
    self.assertIsInstance(eqn.primitive, core.CallPrimitive)
    inner_jaxpr = eqn.params['call_jaxpr']
    self.assertIsInstance(inner_jaxpr, core.Jaxpr)

    self.assertLen(inner_jaxpr.invars, 1 + 4)  # one axis size var
    self.assertEqual((inner_jaxpr.invars[0],), inner_jaxpr.invars[1].aval.shape)
    self.assertEqual((inner_jaxpr.invars[0],), inner_jaxpr.invars[2].aval.shape)
    self.assertEqual((inner_jaxpr.invars[0],), inner_jaxpr.invars[3].aval.shape)
    self.assertEqual((inner_jaxpr.invars[0],), inner_jaxpr.invars[4].aval.shape)

  def test_staging_primitive_applications(self):
    n = core.ShapedArray((), jnp.dtype('int32'), weak_type=False)
    a = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)
    b = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)

    @lu.wrap_init
    def f(x, y):
      z = lax.mul(x, y)
      w = lax.sin(z)
      u = lax_internal._reduce_sum(w, [0])
      return (u,)

    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      f, [n, a, b], keep_inputs=[False, True, True])

    self.assertLen(jaxpr.invars, 1 + 2)  # one axis size var, two other inputs
    self.assertLen(jaxpr.eqns, 3)
    self.assertLen(jaxpr.eqns[0].outvars, 1)
    self.assertEqual(jaxpr.eqns[0].outvars[0].aval.shape,
                     jaxpr.invars[1].aval.shape)

    self.assertLen(jaxpr.outvars, 1)
    self.assertEqual(jaxpr.outvars[0].aval.shape, ())

  @unittest.skip('This test does not work with nested pjit and DShapedArray')
  def test_typecheck_staging_nested(self):
    n = core.ShapedArray((), jnp.dtype('int32'), weak_type=False)
    m = core.ShapedArray((), jnp.dtype('int32'), weak_type=False)
    a = core.DShapedArray((DBIdx(0),), jnp.dtype('float32'), weak_type=False)
    b = core.DShapedArray((DBIdx(1),), jnp.dtype('float32'), weak_type=False)

    @lu.wrap_init
    def f(a, b):
      @jax.jit
      def g(x): return x
      return g(a),

    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      f, [n, m, a, b], keep_inputs=[False, False, True, True])
    # { lambda ; a:i32[] b:i32[] c:f32[a] d:f32[b]. let
    #     e:f32[a] = xla_call[
    #       call_jaxpr={ lambda ; f:i32[] g:f32[f]. let  in (g,) }
    #       name=g
    #     ] a c
    #   in (e,) }
    core.check_jaxpr(jaxpr)  # no problems here...

    # Let's introduce a type error by applying the called jaxpr to arguments
    # with types which aren't consistent with its input binders:
    _, _, c, d = jaxpr.invars
    jaxpr.eqns[0].invars[1] = d
    # { lambda ; a:i32[] b:i32[] c:f32[a] d:f32[b]. let
    #     e:f32[a] = xla_call[
    #       call_jaxpr={ lambda ; f:i32[] g:f32[f]. let  in (g,) }
    #       name=g
    #     ] a d   !!! type error here !!!
    #   in (e,) }
    with self.assertRaisesRegex(TypeError, "passes operand"):
      core.check_jaxpr(jaxpr)

    # Restore the original jaxpr:
    jaxpr.eqns[0].invars[1] = c
    core.check_jaxpr(jaxpr)  # no problems here...

    # Let's introduce another type error by setting the call result let binders
    # to have the wrong type:
    jaxpr.eqns[0].outvars[0] = core.Var('', d.aval)
    # { lambda ; a:i32[] b:i32[] c:f32[a] d:f32[b]. let
    #     e:f32[b] = xla_call[   !!! type error here !!!
    #       call_jaxpr={ lambda ; f:i32[] g:f32[f]. let  in (g,) }
    #       name=g
    #     ] a c
    #   in (h,) }
    with self.assertRaisesRegex(TypeError, "inconsistently typed as"):
      core.check_jaxpr(jaxpr)

  def test_check_jaxpr_key_reuse(self):
    with config.debug_key_reuse(True):
      def f(seed):
        key = jax.random.key(seed)
        return jax.random.uniform(key) + jax.random.normal(key)
      with jax.enable_checks(True):
        with self.assertRaises(jax.errors.KeyReuseError):
          jax.jit(f)(0)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
