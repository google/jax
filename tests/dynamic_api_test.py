# Copyright 2018 Google LLC
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
import re
import sys
import unittest
import numpy as np

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax import core, lax
import jax._src.lib
from jax._src import test_util as jtu
import jax._src.util

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


python_version = (sys.version_info[0], sys.version_info[1])
numpy_version = tuple(map(int, np.__version__.split('.')[:3]))


@jtu.with_config(jax_dynamic_shapes=True, jax_numpy_rank_promotion="allow")
class DynamicShapeTest(jtu.JaxTestCase):
  def test_basic_staging(self):
    def f(x, _):
      return x

    x = jnp.arange(3)
    y = jnp.ones((3, 4))
    jaxpr = jax.make_jaxpr(f, abstracted_axes={0: 'n'})(x, y)

    # { lambda ; a:i32[] b:i32[a] c:f32[a,4]. let  in (b,) }
    self.assertLen(jaxpr.in_avals, 3)
    self.assertLen(jaxpr.in_avals[0].shape, 0)
    self.assertLen(jaxpr.in_avals[1].shape, 1)
    self.assertLen(jaxpr.in_avals[2].shape, 2)
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[1].shape[0])
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[2].shape[0])
    self.assertLen(jaxpr.out_avals, 1)
    self.assertLen(jaxpr.out_avals[0].shape, 1)
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.out_avals[0].shape[0])

  def test_basic_staging_repeated(self):
    def f(x, _):
      return x

    x = jnp.arange(3)
    y = jnp.ones((3, 3))
    jaxpr = jax.make_jaxpr(f, abstracted_axes=(('n',), ('n', 'n')))(x, y)

    # { lambda ; a:i32[] b:i32[a] c:f32[a,a]. let  in (b,) }
    self.assertLen(jaxpr.in_avals, 3)
    self.assertLen(jaxpr.in_avals[0].shape, 0)
    self.assertLen(jaxpr.in_avals[1].shape, 1)
    self.assertLen(jaxpr.in_avals[2].shape, 2)
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[1].shape[0])
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[2].shape[0])
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[2].shape[1])
    self.assertLen(jaxpr.out_avals, 1)
    self.assertLen(jaxpr.out_avals[0].shape, 1)
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.out_avals[0].shape[0])

  def test_basic_staging_multiple_shape_vars(self):
    def f(x, _):
      return x

    x = jnp.arange(3)
    y = jnp.ones((4, 3))
    jaxpr = jax.make_jaxpr(f, abstracted_axes=(('n',), ('m', 'n')))(x, y)

    # { lambda ; a:i32[] b: i32[] c:i32[a] d:f32[b,a]. let  in (c,) }
    self.assertLen(jaxpr.in_avals, 4)
    self.assertLen(jaxpr.in_avals[0].shape, 0)
    self.assertLen(jaxpr.in_avals[1].shape, 0)
    self.assertLen(jaxpr.in_avals[2].shape, 1)
    self.assertLen(jaxpr.in_avals[3].shape, 2)
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[2].shape[0])
    self.assertIs(jaxpr.jaxpr.invars[1], jaxpr.in_avals[3].shape[0])
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[3].shape[1])
    self.assertLen(jaxpr.out_avals, 1)
    self.assertLen(jaxpr.out_avals[0].shape, 1)
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.out_avals[0].shape[0])

  def test_basic_add(self):
    def f(x, y):
      return x + y

    x = jnp.arange(3)
    y = jnp.arange(1, 4)
    jaxpr = jax.make_jaxpr(f, abstracted_axes={0: 'n'})(x, y)

    # { lambda ; a:i32[] b:i32[a] c:i32[a]. let d:i32[a] = add b c in (d,) }
    self.assertLen(jaxpr.eqns, 1)
    self.assertLen(jaxpr.out_avals, 1)
    self.assertLen(jaxpr.out_avals[0].shape, 1)
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.out_avals[0].shape[0])

  def test_basic_jnp(self):
    def f(x):
      y = x + jnp.sin(x)
      return y.sum()

    x = jnp.ones((3, 4))
    jaxpr = jax.make_jaxpr(f, abstracted_axes={0: 'n'})(x)

    # { lambda ; a:i32[] b:f32[a,4]. let
    #     c:f32[a,4] = sin b
    #     d:f32[a,4] = add b c
    #     e:f32[] = reduce_sum[axes=(0, 1)] d
    #   in (e,) }
    self.assertLen(jaxpr.in_avals, 2)
    self.assertLen(jaxpr.eqns, 3)  # sin, add, and reduce_sum
    self.assertLen(jaxpr.out_avals, 1)
    self.assertLen(jaxpr.out_avals[0].shape, 0)

  def test_shape_errors_var_and_lit(self):
    def f(x, y):
      return jnp.sin(x) + y

    x = np.ones(3)
    y = np.ones(3)
    with self.assertRaisesRegex(
        Exception, '[Ii]ncompatible shapes for broadcasting'):
      _ = jax.make_jaxpr(f, abstracted_axes=({0: 'n'}, {}))(x, y)

  def test_shape_errors_distinct_vars(self):
    def f(x, y):
      return jnp.sin(x) + y

    x = np.ones(3)
    y = np.ones(3)
    with self.assertRaisesRegex(
        Exception, '[Ii]ncompatible shapes for broadcasting'):
      _ = jax.make_jaxpr(f, abstracted_axes=({0: 'n'}, {0: 'm'}))(x, y)

  def test_basic_dot(self):
    A = jnp.ones((3, 4))
    x = jnp.ones(4)
    jaxpr = jax.make_jaxpr(jnp.dot, abstracted_axes=(('m', 'n'), ('n',)))(A, x)

    # { lambda ; a:i32[] b:i32[] c:f32[a,b] d:f32[b]. let
    #     e:f32[a] = dot_general[dimension_numbers=(((1,), (0,)), ((), ()))] c d
    #   in (e,) }
    self.assertLen(jaxpr.in_avals, 4)
    self.assertLen(jaxpr.in_avals[0].shape, 0)  # two shape vars
    self.assertLen(jaxpr.in_avals[1].shape, 0)
    self.assertLen(jaxpr.in_avals[2].shape, 2)  # one matrix
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[2].shape[0])
    self.assertIs(jaxpr.jaxpr.invars[1], jaxpr.in_avals[2].shape[1])
    self.assertLen(jaxpr.in_avals[3].shape, 1)  # one vector
    self.assertIs(jaxpr.jaxpr.invars[1], jaxpr.in_avals[3].shape[0])
    self.assertLen(jaxpr.eqns, 1)
    self.assertLen(jaxpr.out_avals, 1)
    self.assertLen(jaxpr.out_avals[0].shape, 1)  # output vector
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.out_avals[0].shape[0])

  def test_basic_broadcast(self):
    def f(x, n):
      return lax.broadcast(x, (n,))

    jaxpr = jax.make_jaxpr(f)(jnp.ones(4), 3)

    # { lambda ; a:f32[4] b:i32[]. let
    #     c:f32[b,4] = broadcast_in_dim[bcast_dims=(1,) shape=(None, 4)] a b
    #   in (c,) }
    self.assertLen(jaxpr.in_avals, 2)
    self.assertLen(jaxpr.eqns, 1)
    self.assertLen(jaxpr.out_avals, 1)
    self.assertLen(jaxpr.out_avals[0].shape, 2)
    self.assertIs(jaxpr.jaxpr.invars[1], jaxpr.out_avals[0].shape[0])
    self.assertEqual(4, jaxpr.out_avals[0].shape[1])

  def test_basic_batchpoly_neuralnet(self):
    def predict(params, inputs):
      for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)
      return outputs

    def loss(params, batch):
      inputs, targets = batch
      preds = predict(params, inputs)
      return jnp.sum((preds - targets) ** 2)

    sizes = [784, 128, 128, 10]
    params = [(jnp.ones((input_dim, output_dim)), jnp.ones(output_dim))
              for input_dim, output_dim in zip(sizes[:-1], sizes[1:])]
    batch = (jnp.ones((32, 784)), jnp.ones((32, 10)))

    # Mainly we want to test that make_jaxpr doesn't crash here.
    jaxpr = jax.make_jaxpr(loss, abstracted_axes=({}, {0: 'n'}))(params, batch)
    self.assertLen(jaxpr.in_avals, 9)
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[-2].shape[0])
    self.assertIs(jaxpr.jaxpr.invars[0], jaxpr.in_avals[-1].shape[0])
    self.assertLen(jaxpr.out_avals, 1)
    self.assertLen(jaxpr.out_avals[0].shape, 0)

  def test_closing_over_polymorphic_shape(self):
    def f(n):
      x = jnp.zeros(n)
      return jax.jit(lambda: x)()

    jaxpr = jax.make_jaxpr(f)(3)

    # { lambda ; a:i32[]. let
    #     b:f32[a] = bcast[dims=() shape=(None,)] 0.0 a
    #     c:f32[a] = xla_call[
    #       call_jaxpr={ lambda ; d:i32[] e:f32[d]. let  in (e,) }
    #       name=<lambda>
    #     ] a b
    #   in (c,) }
    a, = jaxpr.jaxpr.invars
    c, = jaxpr.jaxpr.outvars
    self.assertLen(c.aval.shape, 1)
    self.assertIs(a, c.aval.shape[0])

  def test_closing_over_dynamic_shape(self):
    def f(n):
      m = 2 * n
      x = jnp.zeros(m)
      return jax.jit(lambda: x)()

    # { lambda ; a:i32[]. let
    #     b:i32[] = mul a 2
    #     c:f32[b] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 0.0 b
    #     d:f32[b] = xla_call[
    #       call_jaxpr={ lambda ; e:i32[] f:f32[e]. let  in (f,) }
    #       name=<lambda>
    #     ] b c
    #   in (d,) }
    jaxpr = jax.make_jaxpr(f)(3)
    b, = jaxpr.jaxpr.eqns[0].outvars
    c, = jaxpr.jaxpr.eqns[1].outvars
    d, = jaxpr.jaxpr.eqns[2].outvars
    self.assertLen(c.aval.shape, 1)
    self.assertIs(b, c.aval.shape[0])
    self.assertLen(d.aval.shape, 1)
    self.assertIs(b, d.aval.shape[0])

  def test_closing_over_polymorphic_shape_and_adding(self):
    def f(n):
      x = jnp.zeros(n)
      y = jnp.zeros(n)

      @jax.jit
      def g():
        return x + y
      return g()

    # { lambda ; a:i32[]. let
    #     b:f32[a] = bcast[broadcast_dimensions=() shape=(None,)] 0.0 a
    #     c:f32[a] = bcast[broadcast_dimensions=() shape=(None,)] 0.0 a
    #     d:f32[a] = xla_call[
    #       call_jaxpr={ lambda ; e:i32[] f:f32[e] g:f32[e]. let
    #           h:f32[e] = add f g
    #         in (h,) }
    #       name=g
    #     ] a b c
    #   in (d,) }
    jaxpr = jax.make_jaxpr(f)(3)  # doesn't fail on the addition!
    a, = jaxpr.jaxpr.invars
    b, = jaxpr.jaxpr.eqns[0].outvars
    c, = jaxpr.jaxpr.eqns[1].outvars
    d, = jaxpr.jaxpr.eqns[2].outvars
    self.assertIs(a, b.aval.shape[0])
    self.assertIs(a, c.aval.shape[0])
    self.assertIs(a, d.aval.shape[0])

  def test_passing_in_equal_polymorphic_shapes_and_adding(self):
    def f(n):
      x = jnp.zeros(n)

      @jax.jit
      def g(x, y):
        return x + y
      return g(x, x)

    # { lambda ; a:i32[]. let
    #     b:f32[a] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 0.0 a
    #     c:f32[a] = xla_call[
    #       call_jaxpr={ lambda ; d:i32[] e:f32[d] f:f32[d]. let
    #           g:f32[d] = add e f
    #         in (g,) }
    #       name=g
    #     ] a b b
    #   in (c,) }
    jaxpr = jax.make_jaxpr(f)(3)
    a, = jaxpr.jaxpr.invars
    c, = jaxpr.jaxpr.outvars
    self.assertLen(c.aval.shape, 1)
    self.assertIs(a, c.aval.shape[0])

  @unittest.skip("doesn't work yet: shape error b/c we don't notice x and y same")
  def test_closing_over_and_passing_arg_addition(self):
    # TODO(mattjj,dougalm): currently fails to notice equal shapes, fix!
    def f(n):
      x = jnp.zeros(n)

      @jax.jit
      def g(y):
        return x + y
      return g(x)

    _ = jax.make_jaxpr(f)(3)

  @unittest.skip("doesn't work yet: shape error b/c we don't notice x and jnp.zeros(m) same")
  def test_closing_over_and_passing_size_addition(self):
    # TODO(mattjj,dougalm): currently fails to notice equal shapes, fix!
    def f(n):
      x = jnp.zeros(n)

      @jax.jit
      def g(m):
        return jnp.zeros(m) + x
      return g(n)

    _ = jax.make_jaxpr(f)(3)

  def test_closing_over_and_broadcasting_polymorphic_shape(self):
    def f(n):
      x = jnp.zeros(n)
      @jax.jit
      def g():
        return jnp.zeros(n) + x
      return g()

    # { lambda ; a:i32[]. let
    #     b:f32[a] = bcast[broadcast_dimensions=() shape=(None,)] 0.0 a
    #     c:f32[a] = xla_call[
    #       call_jaxpr={ lambda ; d:i32[] e:f32[d]. let
    #           f:f32[d] = bcast[broadcast_dimensions=() shape=(None,)] 0.0 d
    #           g:f32[d] = add f e
    #         in (g,) }
    #       name=g
    #     ] a b
    #   in (c,) }
    jaxpr = jax.make_jaxpr(f)(3)

    a, = jaxpr.jaxpr.invars
    c, = jaxpr.jaxpr.outvars
    self.assertLen(c.aval.shape, 1)
    self.assertIs(a, c.aval.shape[0])

  def test_closing_over_repeated_shapes(self):
    def zeros(shape):
      if not isinstance(shape, (tuple, list)):
        shape = shape,
      return lax.broadcast(0., shape)

    def f(n):
      m = 2 * n
      x = zeros((m, m))
      return jax.jit(lambda: x.sum(0))()

    # { lambda ; a:i32[]. let
    #     b:i32[] = mul a 2
    #     c:f32[b,b] = broadcast_in_dim[broadcast_dimensions=() shape=(None, None)] 0.0
    #       b b
    #     d:f32[b] = xla_call[
    #       call_jaxpr={ lambda ; e:i32[] f:f32[e,e]. let
    #           g:f32[e] = reduce_sum[axes=(0,)] f
    #         in (g,) }
    #       name=<lambda>
    #     ] b c
    #   in (d,) }
    jaxpr = jax.make_jaxpr(f)(3)
    a, = jaxpr.jaxpr.invars
    b, = jaxpr.jaxpr.eqns[0].outvars
    c, = jaxpr.jaxpr.eqns[1].outvars
    d, = jaxpr.jaxpr.eqns[2].outvars
    b_, c_ = jaxpr.jaxpr.eqns[2].invars
    self.assertLen(c.aval.shape, 2)
    self.assertIs(c.aval.shape[0], b)
    self.assertIs(c.aval.shape[1], b)
    self.assertIs(b, b_)
    self.assertIs(c, c_)
    self.assertLen(d.aval.shape, 1)
    self.assertIs(d.aval.shape[0], b)

  def test_staging_repeated_nested(self):
    def zeros(shape):
      if not isinstance(shape, (tuple, list)):
        shape = shape,
      return lax.broadcast(jnp.float32(0.), shape)

    def f(n):
      m = 2 * n
      x = zeros((m, n))
      y = zeros(m)
      return jax.jit(lambda x, y: x.sum(1) + y)(x, y)

    # { lambda ; a:i32[]. let
    #     b:i32[] = mul a 2
    #     c:f32[b,a] = broadcast_in_dim[broadcast_dimensions=() shape=(None, None)] 0.0
    #       b a
    #     d:f32[b] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 b
    #     e:f32[b] = xla_call[
    #       call_jaxpr={ lambda ; f:i32[] g:i32[] h:f32[f,g] i:f32[f]. let
    #           j:f32[f] = reduce_sum[axes=(1,)] h
    #           k:f32[f] = add j i
    #         in (k,) }
    #       name=<lambda>
    #     ] b a c d
    #   in (e,) }
    jaxpr = jax.make_jaxpr(f)(jnp.int32(3))
    a, = jaxpr.jaxpr.invars
    b, = jaxpr.jaxpr.eqns[0].outvars
    c, = jaxpr.jaxpr.eqns[1].outvars
    d, = jaxpr.jaxpr.eqns[2].outvars
    e, = jaxpr.jaxpr.eqns[3].outvars
    b_, a_, c_, d_ = jaxpr.jaxpr.eqns[3].invars
    self.assertLen(c.aval.shape, 2)
    self.assertIs(c.aval.shape[0], b)
    self.assertIs(c.aval.shape[1], a)
    self.assertLen(e.aval.shape, 1)
    self.assertIs(e.aval.shape[0], b)
    self.assertIs(a, a_)
    self.assertIs(b, b_)
    self.assertIs(c, c_)
    self.assertIs(d, d_)

  def test_jit_abstracted_axes_staging(self):
    # We just test make_jaxpr-of-jit because dynamic shape compilation/execution
    # may not be supported.
    @partial(jax.jit, abstracted_axes=('n',))
    def f(x):
      return jnp.sum(x)
    jaxpr = jax.make_jaxpr(f)(jnp.ones(3, jnp.dtype('float32')))
    # { lambda ; a:f32[3]. let
    #     b:f32[] = xla_call[
    #       call_jaxpr={ lambda ; c:i32[] d:f32[c]. let
    #           e:f32[] = reduce_sum[axes=(0,)] d
    #         in (e,) }
    #       name=f
    #     ] 3 a
    #   in (b,) }
    a, = jaxpr.jaxpr.invars
    e, = jaxpr.jaxpr.eqns
    self.assertLen(e.invars, 2)
    self.assertIsInstance(e.invars[0], core.Literal)
    self.assertIs(e.invars[1], a)
    b, = e.outvars
    self.assertLen(b.aval.shape, 0)

    subjaxpr = e.params['call_jaxpr']
    c, d = subjaxpr.invars
    self.assertLen(c.aval.shape, 0)
    self.assertLen(d.aval.shape, 1)
    self.assertIs(d.aval.shape[0], c)

  def test_jit_abstracted_axes_staging2(self):
    @partial(jax.jit, abstracted_axes=('n',))
    def fun(x):
      return jnp.sum(x)
    jaxpr = jax.make_jaxpr(lambda n: fun(jnp.ones(n + n, jnp.dtype('float32')))
                           )(3)
    # { lambda ; a:i32[]. let
    #     b:i32[] = add a a
    #     c:f32[b] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0 b
    #     d:f32[] = xla_call[
    #       call_jaxpr={ lambda ; e:i32[] f:f32[e]. let
    #           g:f32[] = reduce_sum[axes=(0,)] f
    #         in (g,) }
    #       name=f
    #     ] b c
    #   in (d,) }
    a, = jaxpr.jaxpr.invars
    e1, e2, e3 = jaxpr.jaxpr.eqns
    b, = e1.outvars
    c, = e2.outvars
    b_, c_ = e3.invars
    self.assertIs(b, b_)
    self.assertIs(c, c_)

    subjaxpr = e3.params['call_jaxpr']
    e, f = subjaxpr.invars
    self.assertLen(e.aval.shape, 0)
    self.assertLen(f.aval.shape, 1)
    self.assertIs(f.aval.shape[0], e)

  def test_jit_abstracted_axes_staging3(self):
    f = jax.jit(jnp.sum, abstracted_axes=('n',))
    jaxpr = jax.make_jaxpr(f, abstracted_axes=('n',))(jnp.arange(3.))
    # { lambda ; a:i32[] b:f32[a]. let
    #     c:f32[] = xla_call[
    #       call_jaxpr={ lambda ; d:i32[] e:f32[d]. let
    #           f:f32[] = reduce_sum[axes=(0,)] e
    #         in (f,) }
    #       name=sum
    #     ] a b
    #   in (c,) }
    a, b = jaxpr.jaxpr.invars
    e, = jaxpr.jaxpr.eqns
    self.assertIs(e.invars[0], a)
    self.assertIs(e.invars[1], b)
    c, = e.outvars
    self.assertLen(c.aval.shape, 0)

    subjaxpr = e.params['call_jaxpr']
    d, e = subjaxpr.invars
    self.assertLen(d.aval.shape, 0)
    self.assertLen(e.aval.shape, 1)
    self.assertIs(e.aval.shape[0], d)

  def test_jit_abstracted_axes_return_polymorphic_shape(self):
    f = jax.jit(lambda x: x, abstracted_axes=('n',))
    jaxpr = jax.make_jaxpr(f)(jnp.arange(3))  # doesn't crash
    # { lambda ; a:i32[3]. let
    #     b:i32[3] = xla_call[
    #       call_jaxpr={ lambda ; c:i32[] d:i32[c]. let  in (d,) }
    #       name=<lambda>
    #     ] 3 a
    #   in (b,) }
    a, = jaxpr.jaxpr.invars
    e, = jaxpr.jaxpr.eqns
    three, a_ = e.invars
    b, = e.outvars
    self.assertIsInstance(three, core.Literal)
    self.assertEqual(three.val, 3)
    self.assertIs(a_, a)
    self.assertLen(b.aval.shape, 1)
    self.assertEqual(b.aval.shape[0], 3)

  def test_jit_abstracted_axes_return_polymorphic_shape2(self):
    f = jax.jit(lambda n: jnp.ones(n))
    # TODO(mattjj,dougalm): support dynamic shapes in type checker
    with jax.enable_checks(False):
      jaxpr = jax.make_jaxpr(f)(3)
    # { lambda ; a:i32[]. let
    #     b:f32[a] = xla_call[
    #       call_jaxpr={ lambda ; c:i32[]. let
    #           d:f32[c] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0
    #             c
    #         in (d,) }
    #       name=<lambda>
    #     ] a
    #   in (b,) }
    a, = jaxpr.jaxpr.invars
    e, = jaxpr.jaxpr.eqns
    a_, = e.invars
    self.assertIs(a, a_)
    b, = e.outvars
    a__, = b.aval.shape
    self.assertIs(a, a__)

    with jax.enable_checks(False):
      jaxpr = jax.make_jaxpr(lambda: f(3))()
    # { lambda ; . let
    #     a:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; b:i32[]. let
    #           c:f32[b] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] 1.0
    #             b
    #         in (c,) }
    #       name=<lambda>
    #     ] 3
    #   in (a,) }
    () = jaxpr.jaxpr.invars
    e, = jaxpr.jaxpr.eqns
    three, = e.invars
    self.assertIsInstance(three, core.Literal)
    self.assertEqual(three.val, 3)
    b, = e.outvars
    three_, = b.aval.shape
    self.assertIsInstance(three_, int)
    self.assertEqual(three_, 3)

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_jit_basic_iree(self):
    @jax.jit
    def f(i):
      return jnp.sum(jnp.ones(i, dtype='float32'))
    self.assertAllClose(f(3), jnp.array(3., dtype='float32'), check_dtypes=True)

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_jit_basic_iree_2(self):
    count = 0

    @partial(jax.jit, abstracted_axes=('n',))
    def f(x):
      nonlocal count
      count += 1
      return jnp.sum(x)

    x = f(np.arange(3))
    y = f(np.arange(4))
    self.assertAllClose(x, 3., check_dtypes=False)
    self.assertAllClose(y, 6., check_dtypes=False)
    self.assertEqual(count, 1)

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_jit_polymorphic_output_iree(self):
    # like test_jit_basic_iree, but without the jnp.sum!
    count = 0

    @jax.jit
    def f(i):
      nonlocal count
      count += 1
      return jnp.ones(i, dtype='float32')

    self.assertAllClose(f(3), np.ones(3, dtype='float32'), check_dtypes=True)
    self.assertAllClose(f(4), np.ones(4, dtype='float32'), check_dtypes=True)
    self.assertEqual(count, 1)

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_concatenate(self):
    @partial(jax.jit, abstracted_axes=({0: 'n'},))
    def f(x):  # x: f32[n, 4]
      return jnp.concatenate([x, x, x], axis=0)

    f(np.ones((5, 4), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_reshape(self):
    @partial(jax.jit, abstracted_axes=({0: 'n'},))
    def f(x):  # x: f32[n, 4]
      return jnp.reshape(x, (2, -1))

    f(np.ones((5, 4), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_nested(self):
    @jax.jit
    def nested_f(x):  # f32[h, v] -> f32[h, v]
      # A nested call that needs shape variables
      return jnp.sin(x)

    @partial(jax.jit, abstracted_axes=({0: 'h', 1: 'v'},))
    def f(x):  # f32[h, w] -> f32[h, w]
      return jnp.sin(x) + jax.jit(nested_f)(x)
    f(np.ones((3, 5), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_nested_arange(self):
    def nested_f(x):  # f32[h, v] -> f32[h, v]
      # A nested call that needs to compute with shapes
      return jnp.arange(x.shape[0] * x.shape[1], dtype=x.dtype).reshape(x.shape)

    @partial(jax.jit, abstracted_axes=({0: 'h', 1: 'w'},))
    def f(x):  # f32[h, w] -> f32[h, w]
      return x + jax.jit(nested_f)(x)
    f(np.ones((3, 5), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', 'iree test')
  def test_transpose(self):
    @partial(jax.jit, abstracted_axes=({0: 'h', 1: 'w'},))
    def f(x):  # f32[h, w] -> f32[w, h]
      return x.T

    f(np.ones((3, 5), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', 'iree test')
  def test_matmul(self):
    @partial(jax.jit, abstracted_axes=({0: 'w', 1: 'w'},))
    def f(x):  # f32[w, w] -> f32[w, w]
      return jnp.matmul(x, x)

    f(np.ones((5, 5), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', 'iree test')
  def test_matmul_shape_error(self):
    @partial(jax.jit, abstracted_axes=({0: 'h', 1: 'w'},))
    def f(x):  # f32[h, w] -> error
      return jnp.matmul(x, x)

    # TODO(necula): improve error message, print actual shapes
    with self.assertRaisesRegex(TypeError,
                                re.escape("dot_general requires contracting dimensions to have the same shape, got")):
      f(np.ones((5, 5), dtype=np.float32))

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  @unittest.skip("TODO: investigate failure")
  def test_cond(self):
    @partial(jax.jit, abstracted_axes=({0: 'w', 1: 'w'},))
    def f(x):  # f32[w, w] -> f32[w, w]
      return lax.cond(True,
                      lambda x: jnp.sin(x),
                      lambda x: jnp.matmul(x, x), x)
    f(np.ones((5, 5), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_arange(self):
    @partial(jax.jit, abstracted_axes=({0: 'w'},))
    def f(x):  # f32[w] -> f32[w]
      return jnp.arange(x.shape[0], dtype=x.dtype) + x
    f(np.ones((5,), dtype=np.float32))
    # TODO: add assertions

  @unittest.skip('failing w/ iree error')
  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_broadcast(self):
    @partial(jax.jit, abstracted_axes=({0: 'w'},))
    def f(x):  # f32[w] -> f32[w, w]
      return jnp.broadcast_to(x, (x.shape[0], x.shape[0]))
    f(np.ones((5,), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_zeros(self):
    @partial(jax.jit, abstracted_axes=({0: 'w'},))
    def f(x):  # f32[w] -> f32[w]
      return jnp.zeros(x.shape[0], dtype=x.dtype) + x
    f(np.ones((5,), dtype=np.float32))
    # TODO: add assertions

  @unittest.skip('failing w/ iree error')
  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_stack(self):
    @partial(jax.jit, abstracted_axes=({0: 'w'},))
    def f(x):
      return jnp.stack([jnp.sin(x), jnp.cos(x)])

    f(np.ones((5,), dtype=np.float32))
    # TODO: add assertions

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_jit_dependent_pair_output_iree(self):
    # Like the above 'polymorhpic output' test, but now with a `2 * n`!
    count = 0

    @jax.jit
    def f(n):
      nonlocal count
      count += 1
      return jnp.arange(2 * n)

    x = f(3)
    y = f(4)
    self.assertAllClose(x, jnp.arange(2 * 3), check_dtypes=False)
    self.assertAllClose(y, jnp.arange(2 * 4), check_dtypes=False)
    self.assertEqual(count, 1)

  @jtu.skip_on_devices('iree')  # TODO(mattjj): update getslice, no bints
  def test_slicing_basic(self):
    f = jax.jit(lambda x, n: jnp.sum(x[:n]))
    # TODO(mattjj): revise getslice, add typecheck rule for it, enable checks
    with jax.enable_checks(False):
      ans = f(jnp.arange(10), 3)
    expected = jnp.sum(jnp.arange(10)[:3])
    self.assertAllClose(ans, expected, check_dtypes=True)

  # TODO(mattjj,dougalm,phawkins): debug iree failure, "failed to legalize
  # operation 'mhlo.while' that was explicitly marked illegal"
  @jtu.skip_on_devices('iree')
  def test_scan_basic(self):
    def cumsum(x):
      def body(i, _):
        return i + 1, jnp.sum(x[:i+1])
      _, ans = lax.scan(body, 0, None, length=len(x))
      return ans
    x = jnp.array([3, 1, 4, 1, 5, 9])
    with jax.enable_checks(False):
      ans = cumsum(x)
    expected = jnp.cumsum(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_jit_of_broadcast(self):
    x = jax.jit(jnp.ones)(3)
    self.assertAllClose(x, jnp.ones(3))

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_jit_of_broadcast2(self):
    x = jax.jit(lambda n: jnp.ones(2 * n))(3)
    self.assertAllClose(x, jnp.ones(2 * 3))

  def test_jvp_broadcast(self):
    @jax.jit
    def fn(n, x):
      return lax.broadcast_in_dim(x, (n,), ())

    outer_jaxpr = jax.make_jaxpr(
        lambda x, t: jax.jvp(lambda y: fn(3, y), (x,), (t,))
    )(3., 4.)
    # { lambda ; a:f32[] b:f32[]. let
    #     c:f32[3] d:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; e:i32[] f:f32[] g:f32[]. let
    #           h:f32[e] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] f e
    #           i:f32[e] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] g e
    #         in (h, i) }
    #       name=f
    #     ] 3 a b
    #   in (c, d) }
    self.assertLen(outer_jaxpr.jaxpr.eqns, 1)
    eqn, = outer_jaxpr.jaxpr.eqns
    self.assertIn('call_jaxpr', eqn.params)
    jaxpr = eqn.params['call_jaxpr']
    self.assertLen(jaxpr.invars, 3)
    e, f, g = jaxpr.invars
    self.assertEqual(e.aval.shape, ())
    self.assertEqual(f.aval.shape, ())
    self.assertEqual(g.aval.shape, ())
    self.assertLen(jaxpr.outvars, 2)
    h, i = jaxpr.outvars
    self.assertEqual(h.aval.shape, (e,))
    self.assertEqual(i.aval.shape, (e,))
    self.assertLen(eqn.outvars, 2)
    c, d = eqn.outvars
    self.assertEqual(c.aval.shape, (3,))
    self.assertEqual(d.aval.shape, (3,))

  def test_jvp_basic(self):
    @partial(jax.jit, abstracted_axes=('n',))
    def foo(x):
      return jnp.sin(x)

    x = t = jnp.arange(3.)
    outer_jaxpr = jax.make_jaxpr(lambda x, t: jax.jvp(foo, (x,), (t,)))(x, t)
    # { lambda ; a:f32[3] b:f32[3]. let
    #     c:f32[3] d:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; e:i32[] f:f32[e] g:f32[e]. let
    #           h:f32[e] = sin f
    #           i:f32[e] = cos f
    #           j:f32[e] = mul g i
    #         in (h, j) }
    #       name=f
    #     ] 3 a b
    #   in (c, d) }
    self.assertLen(outer_jaxpr.jaxpr.eqns, 1)
    eqn, = outer_jaxpr.eqns
    self.assertIn('call_jaxpr', eqn.params)
    jaxpr = eqn.params['call_jaxpr']
    self.assertLen(jaxpr.invars, 3)
    e, f, g = jaxpr.invars
    self.assertEqual(e.aval.shape, ())
    self.assertEqual(f.aval.shape, (e,))
    self.assertEqual(g.aval.shape, (e,))
    self.assertLen(jaxpr.outvars, 2)
    self.assertLen(eqn.outvars, 2)
    c, d = eqn.outvars
    self.assertEqual(c.aval.shape, (3,))
    self.assertEqual(d.aval.shape, (3,))

  def test_linearize_basic(self):
    @partial(jax.jit, abstracted_axes=('n',))
    def foo(x):
      return jax.lax.sin(x)

    x = jnp.arange(3.)

    # primal computation
    outer_jaxpr = jax.make_jaxpr(lambda x: jax.linearize(foo, x))(x)
    # { lambda ; a:f32[3]. let
    #     b:f32[3] c:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; d:i32[] e:f32[d]. let
    #           f:f32[d] = sin e
    #           g:f32[d] = cos e
    #         in (f, g) }
    #       name=foo
    #     ] 3 a
    #   in (b, c) }
    self.assertLen(outer_jaxpr.jaxpr.eqns, 1)
    eqn, = outer_jaxpr.jaxpr.eqns
    self.assertIn('call_jaxpr', eqn.params)
    jaxpr = eqn.params['call_jaxpr']
    self.assertLen(jaxpr.invars, 2)
    d, e = jaxpr.invars
    self.assertEqual(d.aval.shape, ())
    self.assertEqual(e.aval.shape, (d,))
    self.assertLen(jaxpr.eqns, 2)
    self.assertLen(jaxpr.outvars, 2)
    f, g = jaxpr.outvars
    self.assertEqual(jaxpr.eqns[0].outvars, [f])
    self.assertEqual(jaxpr.eqns[1].outvars, [g])
    self.assertLen(eqn.outvars, 2)
    b, c = eqn.outvars
    self.assertEqual(b.aval.shape, (3,))
    self.assertEqual(c.aval.shape, (3,))

    # primal and tangent computation
    outer_jaxpr = jax.make_jaxpr(
        lambda x, xdot: jax.linearize(foo, x)[1](xdot))(x, x)
    # { lambda ; a:f32[3] b:f32[3]. let
    #     _:f32[3] c:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; d:i32[] e:f32[d]. let
    #           f:f32[d] = sin e
    #           g:f32[d] = cos e
    #         in (f, g) }
    #       name=foo
    #     ] 3 a
    #     h:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; i:i32[] j:f32[i] k:f32[i]. let
    #           l:f32[i] = mul k j
    #         in (l,) }
    #       name=foo
    #     ] 3 c b
    #   in (h,) }
    self.assertLen(outer_jaxpr.jaxpr.eqns, 2)
    _, eqn = outer_jaxpr.jaxpr.eqns
    self.assertIn('call_jaxpr', eqn.params)
    jaxpr = eqn.params['call_jaxpr']
    self.assertLen(jaxpr.invars, 3)
    i, j, k = jaxpr.invars
    self.assertEqual(i.aval.shape, ())
    self.assertEqual(j.aval.shape, (i,))
    self.assertEqual(k.aval.shape, (i,))
    self.assertLen(eqn.outvars, 1)
    h, = eqn.outvars
    self.assertEqual(h.aval.shape, (3,))

  def test_linearize_basic2(self):
    @partial(jax.jit, abstracted_axes=('n',))
    def foo(x):
      return jax.jit(jax.lax.sin)(x)

    x = jnp.arange(3.)
    outer_jaxpr = jax.make_jaxpr(lambda x: jax.linearize(foo, x))(x)
    # { lambda ; a:f32[3]. let
    #     b:f32[3] c:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; d:i32[] e:f32[d]. let
    #           f:f32[d] g:f32[d] = xla_call[
    #             call_jaxpr={ lambda ; h:i32[] i:f32[h]. let
    #                 j:f32[h] = sin i
    #                 k:f32[h] = cos i
    #               in (j, k) }
    #             name=sin
    #           ] d e
    #         in (f, g) }
    #       name=foo
    #     ] 3 a
    #   in (b, c) }
    self.assertLen(outer_jaxpr.jaxpr.eqns, 1)
    eqn, = outer_jaxpr.jaxpr.eqns
    self.assertLen(eqn.outvars, 2)
    b, c = eqn.outvars
    self.assertEqual(b.aval.shape, (3,))
    self.assertEqual(c.aval.shape, (3,))

  def test_grad_basic(self):
    @partial(jax.jit, abstracted_axes=('n',))
    def foo(x):
      y = jax.lax.sin(x)
      return y.sum()

    x = jnp.arange(3.)
    outer_jaxpr = jax.make_jaxpr(jax.grad(foo))(x)
    # { lambda ; a:f32[3]. let
    #     _:f32[] b:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; c:i32[] d:f32[c]. let
    #           e:f32[c] = sin d
    #           f:f32[c] = cos d
    #           g:f32[] = reduce_sum[axes=(0,)] e
    #         in (g, f) }
    #       name=foo
    #     ] 3 a
    #     h:f32[3] = xla_call[
    #       call_jaxpr={ lambda ; i:i32[] j:f32[i] k:f32[]. let
    #           l:f32[i] = broadcast_in_dim[broadcast_dimensions=() shape=(None,)] k i
    #           m:f32[i] = mul l j
    #         in (m,) }
    #       name=foo
    #     ] 3 b 1.0
    #   in (h,) }
    self.assertLen(outer_jaxpr.jaxpr.eqns, 2)
    fwd_eqn, bwd_eqn = outer_jaxpr.jaxpr.eqns
    self.assertIn('call_jaxpr', fwd_eqn.params)
    fwd_jaxpr = fwd_eqn.params['call_jaxpr']
    self.assertLen(fwd_jaxpr.invars, 2)
    c, d = fwd_jaxpr.invars
    self.assertEqual(c.aval.shape, ())
    self.assertEqual(d.aval.shape, (c,))
    self.assertLen(fwd_jaxpr.outvars, 2)
    g, f = fwd_jaxpr.outvars
    self.assertEqual(g.aval.shape, ())
    self.assertEqual(f.aval.shape, (c,))
    self.assertLen(fwd_eqn.outvars, 2)
    _, b = fwd_eqn.outvars
    self.assertEqual(b.aval.shape, (3,))
    self.assertIn('call_jaxpr', bwd_eqn.params)
    bwd_jaxpr = bwd_eqn.params['call_jaxpr']
    self.assertLen(bwd_jaxpr.invars, 3)
    i, j, k = bwd_jaxpr.invars
    self.assertEqual(i.aval.shape, ())
    self.assertEqual(j.aval.shape, (i,))
    self.assertEqual(k.aval.shape, ())
    self.assertLen(bwd_jaxpr.outvars, 1)
    m, = bwd_jaxpr.outvars
    self.assertEqual(m.aval.shape, (i,))
    self.assertLen(bwd_eqn.outvars, 1)
    h, = bwd_eqn.outvars
    self.assertEqual(h.aval.shape, (3,))

  def test_mlp_autodiff_dynamic_batch_toplevel(self):
    def predict(params, inputs):
      for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(0, outputs)
      return outputs

    def loss(params, batch):
      inputs, targets = batch
      predictions = predict(params, inputs)
      return jnp.sum((predictions - targets) ** 2)

    batch = (inputs, targets) = (jnp.ones((128, 784)), jnp.ones((128, 10)))
    params = [(jnp.ones((784, 256)), jnp.ones(256)),
              (jnp.ones((256, 256)), jnp.ones(256)),
              (jnp.ones((256,  10)), jnp.ones( 10))]

    # jvp
    def loss_jvp(params, batch):
      return jax.jvp(loss, (params, batch), (params, batch))
    jaxpr = jax.make_jaxpr(loss_jvp, abstracted_axes=({}, {0: 'n'}))(params, batch)
    core.check_jaxpr(jaxpr.jaxpr)

    # linearize
    def loss_lin(params, batch):
      y, f_lin = jax.linearize(loss, params, batch)
      y_dot = f_lin(params, batch)
      return y, y_dot
    jaxpr = jax.make_jaxpr(loss_lin, abstracted_axes=({}, {0: 'n'}))(params, batch)
    core.check_jaxpr(jaxpr.jaxpr)

    # grad
    jaxpr = jax.make_jaxpr(jax.grad(loss), abstracted_axes=({}, {0: 'n'}))(params, batch)
    core.check_jaxpr(jaxpr.jaxpr)

  def test_mlp_autodiff_dynamic_batch_inner(self):
    # This is like the above 'toplevel' test, but instead of introducing
    # abstracted axes on the make_jaxpr call, we do it on a jit.

    @partial(jax.jit, abstracted_axes=({}, {0: 'n'}))
    def predict(params, inputs):
      for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(0, outputs)
      return outputs

    def loss(params, batch):
      inputs, targets = batch
      predictions = predict(params, inputs)
      return jnp.sum((predictions - targets) ** 2)

    batch = (inputs, targets) = (jnp.ones((128, 784)), jnp.ones((128, 10)))
    params = [(jnp.ones((784, 256)), jnp.ones(256)),
              (jnp.ones((256, 256)), jnp.ones(256)),
              (jnp.ones((256,  10)), jnp.ones( 10))]

    # jvp
    def loss_jvp(params, batch):
      return jax.jvp(loss, (params, batch), (params, batch))
    jaxpr = jax.make_jaxpr(loss_jvp)(params, batch)
    core.check_jaxpr(jaxpr.jaxpr)

    # linearize
    def loss_lin(params, batch):
      y, f_lin = jax.linearize(loss, params, batch)
      y_dot = f_lin(params, batch)
      return y, y_dot
    jaxpr = jax.make_jaxpr(loss_lin)(params, batch)
    core.check_jaxpr(jaxpr.jaxpr)

    # grad
    jaxpr = jax.make_jaxpr(jax.grad(loss))(params, batch)
    core.check_jaxpr(jaxpr.jaxpr)

  @unittest.skipIf(jtu.device_under_test() != 'iree', "iree test")
  def test_mlp_autodiff_dynamic_batch_iree(self):
    count = 0

    def predict(params, inputs):
      for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(0, outputs)
      return outputs

    def loss_ref(params, batch):
      nonlocal count
      count += 1  # count retraces
      inputs, targets = batch
      predictions = predict(params, inputs)
      return jnp.sum((predictions - targets) ** 2)

    loss = jax.jit(loss_ref, abstracted_axes=({}, {0: 'n'}))

    params = [(jnp.ones((784, 256)), jnp.ones(256)),
              (jnp.ones((256,  10)), jnp.ones( 10))]

    # two different size batches
    batch1 = (inputs, targets) = (jnp.ones((128, 784)), jnp.ones((128, 10)))
    batch2 = (inputs, targets) = (jnp.ones((32, 784)), jnp.ones((32, 10)))

    _ = loss(params, batch1)
    _ = loss(params, batch2)
    self.assertEqual(count, 1)

    _ = jax.grad(loss)(params, batch1)
    _ = jax.grad(loss)(params, batch2)
    self.assertEqual(count, 2)

    ans      = loss(    params, batch1)
    expected = loss_ref(params, batch1)
    self.assertAllClose(ans, expected)

    ans      = jax.grad(loss    )(params, batch1)
    expected = jax.grad(loss_ref)(params, batch1)
    self.assertAllClose(ans, expected)

  @jax.enable_checks(False)  # TODO(mattjj): upgrade typecompat to handle bints
  def test_mlp_autodiff_dynamic_batch_bint(self):
    count = 0

    def predict(params, inputs):
      for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(0, outputs)
      return outputs

    def loss_ref(params, batch):
      nonlocal count
      count += 1  # count traces
      inputs, targets = batch
      predictions = predict(params, inputs)
      return jnp.sum((predictions - targets) ** 2)

    loss = jax.jit(loss_ref, abstracted_axes=({}, {0: 'n'}))

    params = [(jnp.ones((784, 256)), jnp.ones(256)),
              (jnp.ones((256,  10)), jnp.ones( 10))]

    # two different batch sizes *with bints*
    bs1 = jax.lax.make_bint(128, 128)
    batch1 = (jnp.ones((bs1, 784)), jnp.ones((bs1, 10)))

    bs2 = jax.lax.make_bint(32, 128)
    batch2 = (jnp.ones((bs2, 784)), jnp.ones((bs2, 10)))

    # count retraces (and don't crash)
    self.assertEqual(count, 0)
    _ = jax.grad(loss)(params, batch1)
    self.assertEqual(count, 1)
    g2 = jax.grad(loss)(params, batch2)
    self.assertEqual(count, 1)  # cache hit!

    # check the numbers make sense
    batch = (jnp.ones((32, 784)), jnp.ones((32, 10)))
    g2_expected = jax.grad(loss_ref)(params, batch)
    self.assertAllClose(g2, g2_expected, check_dtypes=False,
                        atol=1e-3, rtol=1e-3)

  def test_bint_basic(self):
    d = lax.make_bint(3, 5)
    self.assertEqual(str(d), '3{≤5}')

    @jax.jit
    def f(d):
      jnp.sin(3.)  # don't have an empty jaxpr
      return d
    f(d)  # doesn't crash

  def test_bint_broadcast(self):
    d = lax.make_bint(3, 5)

    x = lax.broadcast_in_dim(0, (d,), ())  # doesn't crash
    self.assertIsInstance(x, core.PaddedArray)
    self.assertAllClose(x._data, np.zeros(5, dtype='int32'), check_dtypes=False)
    self.assertEqual(
        x._aval, core.DShapedArray((core.BInt(3, 5),), x._data.dtype, True))

    def f(n):
      return jnp.zeros(n)
    x = jax.jit(f)(d)
    self.assertIsInstance(x, core.PaddedArray)
    self.assertAllClose(x._data, np.zeros(5, dtype='int32'), check_dtypes=False)
    self.assertEqual(
        x._aval, core.DShapedArray((core.BInt(3, 5),), x._data.dtype, False))

    jaxpr = jax.make_jaxpr(f)(d).jaxpr
    # { lambda ; a:bint{≤5}[]. let
    #     b:f32[a] = broadcast_in_dim[...] 0.0 a
    #   in (b,) }
    self.assertLen(jaxpr.invars, 1)
    a, = jaxpr.invars
    self.assertEqual(a.aval, core.AbstractBInt(5))
    self.assertLen(jaxpr.eqns, 1)
    eqn, = jaxpr.eqns
    self.assertLen(eqn.outvars, 1)
    b, = eqn.outvars
    self.assertEqual(b.aval.shape, (a,))

  def test_bint_iota(self):
    def f(d):
      return jnp.arange(d, dtype='int32')

    y = f(lax.make_bint(3, 5))
    self.assertIsInstance(y, core.PaddedArray)
    self.assertAllClose(y._data, np.arange(5), check_dtypes=False)

    d = lax.make_bint(3, 5)
    y = jax.jit(f)(d)
    self.assertIsInstance(y, core.PaddedArray)
    self.assertAllClose(y._data, np.arange(5), check_dtypes=False)

  def test_bint_compilation_cache(self):
    count = 0

    @jax.jit
    def f(n):
      nonlocal count
      count += 1
      return jnp.zeros(n)
    f(lax.make_bint(3, 5))
    f(lax.make_bint(4, 5))
    self.assertEqual(count, 1)

  def test_bint_compilation_cache2(self):
    count = 0

    @partial(jax.jit, abstracted_axes=('n',))
    def f(x):
      nonlocal count
      count += 1
      return x.sum()

    d = lax.make_bint(3, 5)
    x = jnp.arange(d)
    y = f(x)
    self.assertEqual(y, 3)
    self.assertEqual(count, 1)

    d = lax.make_bint(4, 5)
    x = jnp.arange(d)
    y = f(x)
    self.assertEqual(y, 6)
    self.assertEqual(count, 1)

    d = lax.make_bint(4, 6)
    x = jnp.arange(d)
    y = f(x)
    self.assertEqual(y, 6)
    self.assertEqual(count, 2)

  def test_bint_add(self):
    d = lax.make_bint(4, 6)
    x = jnp.arange(d)

    @jax.jit
    def f(x):
      return x + x

    f(x)  # doesn't crash

  def test_lower_abstracted_axes(self):
    @partial(jax.jit, abstracted_axes=('n',))
    def f(x):
      return x.sum()

    f_lowered = f.lower(np.arange(3, dtype='int32'))
    mhlo = f_lowered.compiler_ir('mhlo')
    self.assertIn('tensor<?xi32>', str(mhlo))

  def test_lower_abstracted_axes_shapedtypestruct(self):
    @partial(jax.jit, abstracted_axes=('n',))
    def f(x):
      return x.sum()

    f_lowered = f.lower(jax.ShapeDtypeStruct((3,), np.int32))
    mhlo = f_lowered.compiler_ir('mhlo')
    self.assertIn('tensor<?xi32>', str(mhlo))

  def test_vmap_abstracted_axis(self):
    def foo(x, y):
      z = jax.vmap(jnp.sin)(x) * y
      return jax.vmap(jnp.add)(x, z)

    x = jnp.arange(3.)
    jaxpr = jax.make_jaxpr(foo, abstracted_axes=('n',))(x, x).jaxpr
    self.assertLen(jaxpr.invars, 3)
    a, b, c = jaxpr.invars
    self.assertEqual(a.aval.shape, ())
    self.assertEqual(b.aval.shape, (a,))
    self.assertEqual(c.aval.shape, (a,))
    self.assertLen(jaxpr.eqns, 3)
    self.assertLen(jaxpr.outvars, 1)
    f, = jaxpr.outvars
    self.assertEqual(f.aval.shape, (a,))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
