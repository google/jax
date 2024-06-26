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

"""Tests whether the frontend attributes added by the context manager are

correctly propagated to the jaxpr and mlir.
"""

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src.lax import lax
from jax.experimental import attributes
import jax.numpy as jnp

from jax._src import core

config.parse_flags_with_absl()

class FrontendAttributesTest(jtu.JaxTestCase):

  def test_no_attributes(self):
    @jax.jit
    def f(a, b):
      return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertNotIn("mhlo.frontend_attributes = {}", f_lowered_text)

  def test_f_jitted_jaxpr(self):
    @jax.jit
    def f(a, b):
      with attributes(a="b"):
        return a + b

    f_jaxpr = jax.make_jaxpr(f)(1, 2)
    eqns = f_jaxpr.eqns
    for eq in eqns[1:]:
      self.assertDictEqual(eq.ctx.attributes, {'a': 'b'})

  def test_f_jitted_mlir(self):
    @jax.jit
    def f(a, b):
      with attributes(a="b"):
        return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {a = "b"}', f_lowered_text)

  def test_f_jitted_mlir_bool_attributes(self):
    @jax.jit
    def f(a, b):
      with attributes(a=True):
        return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {a = "true"}', f_lowered_text)

  def test_f_jitted_mlir_int_attributes(self):
    @jax.jit
    def f(a, b):
      with attributes(a=10):
        return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {a = "10"}', f_lowered_text)

  def test_f_nonjitted_mlir(self):
    def f_add(a, b):
      return dispatch.apply_primitive(lax.add_p, a, b)

    arg1 = jnp.arange(2)
    with attributes(a="b"):
      self.assertIn(
          'mhlo.frontend_attributes = {a = "b"}',
          jax.jit(f_add).lower(arg1, arg1).as_text(),
      )

  def test_f_attributes_scope(self):
    with attributes(a="b"):

      @jax.jit
      def f(a, b):
        return a + b

    # Expect no attributes
    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertNotIn("mhlo.frontend_attributes = {}", f_lowered_text)

  def test_f_attributes_overwrite(self):
    @jax.jit
    def g(a, b):
      return a*b
    with attributes(a="b"):

      @jax.jit
      def f(a, b):
        with attributes(a="c"):
          return a + b
      f_lowered_text = f.lower(1.0, 2.0).as_text()
      self.assertIn('mhlo.frontend_attributes = {a = "c"}', f_lowered_text)
      self.assertIn('mhlo.frontend_attributes = {a = "b"}', g.lower(1.0, 2.0).as_text())
    self.assertNotIn('mhlo.frontend_attributes', g.lower(1.0, 2.0).as_text())

  def test_f_attributes_merge(self):
    with attributes(key1="val1"):

      @jax.jit
      def f(a, b):
        with attributes(key2="val2"):
          return a + b
      f_lowered_text = f.lower(1.0, 2.0).as_text()
      self.assertIn(
          'mhlo.frontend_attributes = {key1 = "val1", key2 = "val2"}',
          f_lowered_text,
      )

  def test_attr_caching_jit_mlir(self):
    @jax.jit
    def f_add_jit(a, b):
      return a + b

    with attributes(b="c"):
      f_add_lowered1 = f_add_jit.lower(2.0, 3.0).as_text()
    # Expect no attributes in the mlir.
    f_add_lowered2 = f_add_jit.lower(1.0, 2.0).as_text()
    with attributes(c="d"):
      f_add_lowered3 = f_add_jit.lower(4.0, 5.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {b = "c"}', f_add_lowered1)
    self.assertNotIn("mhlo.frontend_attributes = {}", f_add_lowered2)
    self.assertNotIn('mhlo.frontend_attributes = {b = "c"}', f_add_lowered2)
    self.assertNotIn('mhlo.frontend_attributes = {c = "d"}', f_add_lowered2)
    self.assertIn('mhlo.frontend_attributes = {c = "d"}', f_add_lowered3)

  def test_attributes_call(self):
    @jax.jit
    def f(a, b):
      return a + b
    @jax.jit
    def g(a, b):
      with attributes(a="b"):
        return a*f(a, b)
    for line in g.lower(1.0, 2.0).as_text().split("\n"):
      if "call @f" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)

  def test_attr_caching_nonjit_mlir(self):
    def f_add(a, b):
      return dispatch.apply_primitive(lax.add_p, a, b)

    arg1 = jnp.arange(2)
    arg2 = jnp.arange(2) + 1
    arg3 = jnp.arange(2) + 2
    with attributes(b="c"):
      self.assertIn(
          'mhlo.frontend_attributes = {b = "c"}',
          jax.jit(f_add).lower(arg1, arg1).as_text(),
      )
    # Expect no attributes in the jaxpr.
    self.assertNotIn(
        "mhlo.frontend_attributes",
        jax.jit(f_add).lower(arg2, arg2).as_text(),
    )

    with attributes(c="d"):
      self.assertIn(
          'mhlo.frontend_attributes = {c = "d"}',
          jax.jit(f_add).lower(arg3, arg3).as_text(),
      )

  def test_axpy(self):
    @jax.jit
    def axpy(a, x, y):
      with attributes(a="b"):
        return a * x + y

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}',
        axpy.lower(1.0, 2.0, 3.0).as_text(),
    )

  def test_while(self):
    @jax.jit
    def f(a):
      with attributes(a="b"):
        return jax.lax.while_loop(lambda x: x < 10, lambda x: x + 1, a)

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}', f.lower(1.0).as_text()
    )

  def test_while_condition_body(self):
    @jax.jit
    def f_condition(x):
      with attributes(a="b"):
        return x < 10
    @jax.jit
    def f_body(x):
      with attributes(a="c"):
        return x + 1
    @jax.jit
    def while_fn(a):
      return jax.lax.while_loop(f_condition, f_body, a)
    for line in while_fn.lower(1.0).as_text().split("\n"):
      if "stablehlo.compare" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)
      if "stablehlo.add" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "c"}', line)

  def test_nested_jit(self):
    @jax.jit
    def f(x, y):
      with attributes(a="b"):
        z = x * y

        @jax.jit
        def g(z):
          with attributes(c="d"):
            return z**2 + 1

        return g(z)

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b", c = "d"}',
        f.lower(1.0, 2.0).as_text(),
    )

  def test_grad_jaxpr(self):
    @jax.jit
    def f(x, y):
      with attributes(a="b"):
        return jax.grad(lambda x: x**3 + y**2 + jnp.sin(x))(x)

    f_jaxpr = jax.make_jaxpr(f)(1.0, 2.0)
    eqns = f_jaxpr.eqns
    for eq in eqns[1:]:
      self.assertDictEqual(eq.ctx.attributes, {'a': 'b'})

  def test_grad_mlir(self):
    @jax.jit
    def f(x):
      with attributes(a="b"):
        return jax.grad(lambda x: x**3 + x**2 + jnp.sin(x))(x)
    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}', f.lower(1.0).as_text()
    )

  def test_grad_outside_ctx(self):
    @jax.jit
    def f(x):
      with attributes(a="b"):
        return x**3 + x**2 + jnp.sin(x)
    grad_fn = jax.jit(jax.grad(f))
    for line in grad_fn.lower(1.0).as_text().split("\n"):
      if "stablehlo.cosine" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)
      if "call @integer_pow" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)

  def test_pmap_mlir(self):
    def f(x):
      with attributes(a="b"):
        return x / jax.lax.psum(x, "i")

    with attributes(c="d"):
      f_pmap = jax.pmap(f, axis_name="i")
      self.assertIn(
          'mhlo.frontend_attributes = {a = "b", c = "d"}',
          f_pmap.lower(jnp.arange(5)).as_text(),
      )

  def test_vmap_jaxpr(self):
    dct = {"a": 0.0, "b": jnp.arange(5.0)}
    @jax.jit
    def f(dct, x):
      with attributes(a="b"):
        return dct["a"] + dct["b"] + x

    with attributes(a="d"):
      f_vmap = jax.vmap(f, in_axes=({"a": None, "b": 0}, None))
      f_jaxpr = jax.make_jaxpr(f_vmap)(dct, 1.0)
      eqns = f_jaxpr.eqns
      for eq in eqns[1:]:
        self.assertDictEqual(eq.ctx.attributes, {'a': 'd'})

  def test_vmap_mlir(self):
    @jax.jit
    def f(x, y):
      with attributes(a="b"):
        return (x + y, y * 2.0)

    f_vmap_jaxpr = jax.make_jaxpr(jax.vmap(f, in_axes=(0, None)))
    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}',
        f_vmap_jaxpr.lower(jnp.arange(5.0), 1.0).as_text(),
    )

  def test_multiple_instructions(self):
    @jax.jit
    def f(x, a):
      y = jnp.matmul(x, x)
      with attributes(a="b"):
        return y + a
    for line in f.lower(jnp.arange(5.0), 1.0).as_text().split("\n"):
      # matmul doesn't have attributes
      if "stablehlo.dot_general" in line:
        self.assertNotIn('mhlo.frontend_attributes = {a = "b"}', line)
      if "stablehlo.add" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
