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

# ruff: noqa: F401

import functools
import math
import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas.pallas_call import _trace_to_jaxpr
from jax._src import tpu_custom_call  # For configuration values
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import export
import numpy as np


config.update("jax_traceback_filtering", "off")
config.parse_flags_with_absl()


# TODO(necula): support an activation
def matmul_kernel(x_ref, y_ref, o_ref):
  # x shape: (m, l), y shape (l, n), o shape: (m ,n)
  block_m, block_l = x_ref.shape
  block_l2, block_n = y_ref.shape
  assert block_l2 == block_l
  assert o_ref.shape == (block_m, block_n)
  @pl.when(pl.program_id(axis=2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)

  o_ref[...] += x_ref[...] @ y_ref[...]


@functools.partial(jax.jit, static_argnames=['block_shape'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    block_shape=(128, 128, 128)
):
  m, l = x.shape
  l2, n = y.shape
  assert l2 == l
  block_m, block_n, block_l = block_shape
  assert l % block_l == 0, f"{l=}, {block_l=}"
  assert m % block_m == 0, f"{m=}, {block_m=}"
  assert n % block_n == 0, f"{n=}, {block_n=}"
  grid = (m // block_m, n // block_n, l // block_l)
  fused_matmul = pl.pallas_call(
      functools.partial(matmul_kernel),
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      in_specs=[
          pl.BlockSpec(lambda i, j, k: (i, k), (block_m, block_l)),
          pl.BlockSpec(lambda i, j, k: (k, j), (block_l, block_n)),
      ],
      out_specs=pl.BlockSpec(lambda i, j, k: (i, j), (block_m, block_n)),
      grid=grid,
      interpret=jtu.test_device_matches(["cpu"]),
  )
  return fused_matmul(x, y)


class ShapePolyTest(jtu.JaxTestCase,
                    parameterized.TestCase):

  def setUp(self):
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPU with capability >= sm80")
    if sys.platform == "win32":
      self.skipTest("Only works on non-Windows platforms")
    super().setUp()
    _trace_to_jaxpr.cache_clear()

  def test_copy(self):
    # The blocks are static, but the input and the grid are of polymorphic
    # dimensions.
    block_shape = (8, 128)
    def f(x, *, eager=False):  # x: i32[w, h]
      def copy_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      # Use both pl.cdiv and // for specifying the grid
      grid = (pl.cdiv(x.shape[0], block_shape[0]),
              (x.shape[1] + 1) // block_shape[1])
      return pl.pallas_call(
          copy_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[pl.BlockSpec(lambda i, j: (i, j), block_shape)],
          out_specs=pl.BlockSpec(lambda i, j: (i, j), block_shape),
          grid=grid,
          interpret=eager and jtu.test_device_matches(["cpu"]))(x)

    shape1 = (128, 256)
    x1 = jnp.arange(math.prod(shape1), dtype=np.int32).reshape(shape1)
    res = f(x1, eager=True)
    self.assertAllClose(res, x1)

    w, h = export.symbolic_shape("w, h")
    exp = export.export(
        jax.jit(f),
        platforms=["tpu"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

    if jtu.test_device_matches(["tpu"]):
      res_exp_1 = exp.call(x1)
      self.assertAllClose(res_exp_1, x1)

      shape2 = block_shape
      x2 = jnp.arange(math.prod(shape2), dtype=np.int32).reshape(shape2)
      res_exp_2 = exp.call(x2)
      self.assertAllClose(res_exp_2, x2)

    # TODO(necula): support shape polymorphism for GPU
    with self.assertRaisesRegex(
        NotImplementedError,
        "dynamic grid bounds not supported in the Triton backend"):
      export.export(
          jax.jit(f),
          platforms=["cuda"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

  def test_block_sizes_must_be_static_no_grid(self):
    def f(x, *, eager=False):  # x: f32[w, h]
      def copy_one(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      return pl.pallas_call(
          copy_one,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          interpret=eager and jtu.test_device_matches(["cpu"]))(x)
    shape1 = (128, 256)
    x1 = jnp.arange(math.prod(shape1), dtype=np.int32).reshape(shape1)
    res = f(x1, eager=True)
    self.assertAllClose(res, x1)

    w, h = export.symbolic_shape("w, h")
    with self.assertRaisesRegex(
        ValueError,
        "shape polymorphism for Pallas does not support dynamically-shaped blocks"):
      export.export(
          jax.jit(f),
          platforms=["tpu"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

  def test_block_sizes_must_be_static(self):
    def f(x, *, eager=False):  # x: f32[w, h]
      def copy_one(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      grid = (2, 2)
      block_shape = (x.shape[0] // grid[0], x.shape[1] // grid[1])
      return pl.pallas_call(
          copy_one,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[pl.BlockSpec(lambda i, j: (i, j), block_shape)],
          out_specs=pl.BlockSpec(lambda i, j: (i, j), block_shape),
          grid=grid,
          interpret=eager and jtu.test_device_matches(["cpu"]))(x)
    shape1 = (128, 256)
    x1 = jnp.arange(math.prod(shape1), dtype=np.int32).reshape(shape1)
    res = f(x1, eager=True)
    self.assertAllClose(res, x1)

    w, h = export.symbolic_shape("w, h")
    with self.assertRaisesRegex(
        ValueError,
        "shape polymorphism for Pallas does not support dynamically-shaped blocks"):

      export.export(
          jax.jit(f),
          platforms=["tpu"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

  @jtu.run_on_devices("tpu")
  def test_matmul(self):
    x_shape = (1024, 256)
    y_shape = (256, 2048)

    key = jax.random.key(42)
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, x_shape, dtype=np.float32)
    y = jax.random.normal(key2, y_shape, dtype=np.float32)

    res = matmul(x, y)
    self.assertAllClose(res, x @ y, atol=1e-4)

    m, n, l = export.symbolic_shape("m, n, l",
                                    constraints=["mod(m, 128) == 0",
                                                 "mod(n, 128) == 0",
                                                 "mod(l, 128) == 0"])
    exp = export.export(
        matmul,
        platforms=["tpu"])(
            jax.ShapeDtypeStruct((m, l), jnp.float32),
            jax.ShapeDtypeStruct((l, n), jnp.float32))
    if jtu.test_device_matches(["tpu"]):
      res_exp = exp.call(x, y)
      self.assertAllClose(res_exp, x @ y, atol=1e-4)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
