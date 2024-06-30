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


class ShapePolyTest(jtu.JaxTestCase,
                    parameterized.TestCase):

  def setUp(self):
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")
    if sys.platform == "win32":
      self.skipTest("Only works on non-Windows platforms")
    super().setUp()
    _trace_to_jaxpr.cache_clear()

  def test_grid_static_input_shape_poly(self):
    # The grid and blocks are static, but the input is of polymorphic input
    # size.
    def f(x, *, eager=False):  # x: f32[w, h]
      def copy_one(x_ref, o_ref):  # x_ref, o_ref: Ref[f32, 2x2]
        o_ref[:, :] = x_ref[:, :]
      return pl.pallas_call(copy_one,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                            in_specs=[pl.BlockSpec(lambda i: (i, i), (2, 2))],
                            out_specs=pl.BlockSpec(lambda i: (i, i), (2, 2)),
                            grid=(4,),
                            interpret=eager and jtu.test_device_matches(["cpu"]))(x)

    x_10x10 = jnp.arange(100, dtype=np.int32).reshape((10, 10))
    res = f(x_10x10, eager=True)

    def check_expected(res, x):
      for i in range(4):
        self.assertAllClose(res[i * 2:i * 2 + 2, i * 2:i * 2 + 2],
                            x[i * 2:i * 2 + 2, i * 2:i * 2 + 2])

    check_expected(res, x_10x10)

    w, h = export.symbolic_shape("w, h")
    # TODO(necula): support shape polymorphism for GPU
    exp = export.export(
        jax.jit(f),
        platforms=["tpu"])(jax.ShapeDtypeStruct((w, h), jnp.int32))
    if jtu.test_device_matches(["tpu"]):
      res_exp_10x10 = exp.call(x_10x10)
      check_expected(res_exp_10x10, x_10x10)

      x_3x3 = x_10x10[:3, :3]
      res_exp_3x3 = f(x_3x3)
      check_expected(res_exp_3x3, x_3x3)

  def test_grid_poly_input_shape_poly(self):
    # The blocks are static, but the input and the grid are of polymorphic
    # dimensions.
    def f(x, *, eager=False):  # x: f32[w, h]
      def copy_one(x_ref, o_ref):
        o_ref[:, :] = x_ref[:, :]
      # Use both pl.cdiv and // for specifying the grid
      grid = (pl.cdiv(x.shape[0], 2), (x.shape[1] + 1) // 2)
      return pl.pallas_call(copy_one,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                            in_specs=[pl.BlockSpec(lambda i, j: (i, j), (2, 2))],
                            out_specs=pl.BlockSpec(lambda i, j: (i, j), (2, 2)),
                            grid=grid,
                            interpret=eager and jtu.test_device_matches(["cpu"]))(x)

    def check_expected(res, x):
      for i in range(pl.cdiv(x.shape[0], 2)):
        for j in range(pl.cdiv(x.shape[1], 2)):
          self.assertAllClose(res[i * 2:i * 2 + 2, j * 2:j * 2 + 2],
                              x[i * 2:i * 2 + 2, j * 2:j * 2 + 2])

    x_10x10 = jnp.arange(100, dtype=np.int32).reshape((10, 10))
    res = f(x_10x10, eager=True)
    check_expected(res, x_10x10)

    w, h = export.symbolic_shape("w, h")
    exp = export.export(
        jax.jit(f),
        platforms=["tpu"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

    if jtu.test_device_matches(["tpu"]):
      res_exp_10x10 = exp.call(x_10x10)
      check_expected(res_exp_10x10, x_10x10)

      x_3x3 = x_10x10[:3, :3]
      res_exp_3x3 = f(x_3x3)
      check_expected(res_exp_3x3, x_3x3)

    # TODO(necula): support shape polymorphism for GPU
    with self.assertRaisesRegex(
        NotImplementedError,
        "dynamic grid bounds not supported in the Triton backend"):
      export.export(
          jax.jit(f),
          platforms=["cuda"])(jax.ShapeDtypeStruct((w, h), jnp.int32))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
