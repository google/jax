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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
import jax._src.pallas.mosaic_gpu.core as plgpu
import jax._src.pallas.mosaic_gpu.primitives as plgpu_primitives
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


class PallasTest(jtu.JaxTestCase):

  def setUp(self):
    if config.enable_x64.value:
      self.skipTest("Only works on x32 at the moment")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Only works on a GPU with capability >= sm90")

    super().setUp()


class PallasCallTest(PallasTest):

  def test_add_one(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_grid(self):
    @functools.partial(
        pl.pallas_call,
        in_specs=[pl.BlockSpec((128,), lambda *i: i)],
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.float32),
        grid=2,
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(128 * 2).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_doubled_sum(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + jnp.sum(x_ref[...]) + jnp.sum(x_ref[...])

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + x.sum()*2)

  @parameterized.product(input_factor=[0.001, 1, 10, 100, 100])
  def test_layer_norm(self, input_factor):
    eps = 1e-5
    gamma = 1.0
    beta = 1.0

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
        compiler_params={"smem_scratch_bytes": 4 * 4},
    )
    def layer_norm(x_ref, o_ref):
      x_mean = jnp.mean(x_ref[...])
      x_centered = x_ref[...] - x_mean
      o_ref[...] = (
          x_centered * jax.lax.rsqrt(jnp.mean(x_centered**2) + eps) * gamma
          + beta
      )

    def layer_norm_np(x):
      x_mean = np.mean(x)
      x_centered = x - x_mean
      return (x_centered / np.sqrt(np.mean(x_centered**2) + eps) * gamma) + beta

    # Ones are always fully precise
    x = jnp.ones((256,)).astype(jnp.float32) * input_factor
    np.testing.assert_allclose(layer_norm(x), layer_norm_np(x))

    # random (and anything else is not)
    x = (
        jax.random.uniform(jax.random.key(42), shape=(256,), dtype=jnp.float32)
        * input_factor
    )
    # TODO(cperivol): find out why in this particular case we have a small-ish error.
    rtol = 1e-07 if input_factor > 10 else 5e-5
    np.testing.assert_allclose(layer_norm(x), layer_norm_np(x), rtol=rtol)

  def test_print(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      del x_ref, o_ref
      pl.debug_print("It works!")

    x = jnp.arange(256).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertEqual(output(), "It works!\n")

  def test_print_with_values(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x[0] = {}", x_ref[0])

    x = jnp.arange(256).astype(jnp.float32)
    with self.assertRaises(Exception):
      # TODO(slebedev): Remove assertRaises() once we support indexing.
      kernel(x)

  def test_scoped_allocation(self):
    def kernel(x_ref, o_ref):
      def body(tmp_ref):
        self.assertEqual(tmp_ref.shape, (8, 128))
        tmp_ref[...] = x_ref[...] + 1.0
        return tmp_ref[...]

      tmp = pl.run_scoped(body, plgpu.SMEM((8, 128), jnp.float32))
      self.assertEqual(tmp.shape, (8, 128))
      o_ref[...] = tmp

    inp = np.ones((8, 128))
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    o = f(inp)
    np.testing.assert_array_equal(o, inp + 1.0)

  def test_program_id(self):
    @functools.partial(
        pl.pallas_call,
        in_specs=(),
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.int32),
        grid=2,
    )
    def kernel(o_ref):
      o_ref[...] = jnp.full(o_ref.shape, pl.program_id(0))

    np.testing.assert_array_equal(
        kernel(),
        jnp.array([0] * 128 + [1] * 128, dtype=jnp.int32),
    )

  def test_num_programs(self):
    @functools.partial(
        pl.pallas_call,
        in_specs=(),
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.int32),
        grid=2,
    )
    def kernel(o_ref):
      o_ref[...] = jnp.full(o_ref.shape, pl.num_programs(0))

    np.testing.assert_array_equal(
        kernel(),
        jnp.full([256], 2, dtype=jnp.int32),
    )

  def test_wgmma(self):
    k = 128

    # The configurateion retains all information related to the
    # wgmma. Particularly the swizzling and whether the memory order
    # of the arguments. This information is low level, is unambiguous
    # given the dtype and swizzle, and needs to be consisten at TMA
    # and wgmma so we define it once here.
    wgmma_config = plgpu.WGMMAConfig(jnp.float16, swizzle=min(128, k))
    lhs_smem = wgmma_config.lhs_smem_config((64, k))
    rhs_smem = wgmma_config.rhs_smem_config((k, 128))
    out_smem = wgmma_config.out_smem_config((64, 128))

    def kernel(a_ref, b_ref, o_ref):
      def body(acc):
        # Effectful call to the wgmma pipeline. Information like
        # swizzling is encapsulated in the operands.
        plgpu_primitives.wgmma(acc, a_ref, b_ref)
        # Flush the pipeline. An argument of N would allow up to N
        # wgmma calls in the pipeline.
        plgpu_primitives.wgmma_wait(0)
        # Acc is an abstract array reference that we don't want to
        # dereference before the wgmma pipeline is flished.
        return acc[...]

      # Create a mutable and scoped accumulator for wgmma.
      acc_arr = pl.run_scoped(body, wgmma_config.accumulator_config(64, 128))
      o_ref[...] = acc_arr

    a = np.ones((64, k), jnp.float16)
    b = np.ones((k, 128), jnp.float16)
    _blockspec = lambda smem_config: pl.BlockSpec(smem_config.shape, None, memory_space=smem_config.memory_space)
    res = pl.pallas_call(
        kernel,
        in_specs=[_blockspec(lhs_smem), _blockspec(rhs_smem)],
        out_specs=_blockspec(out_smem),
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
    )(a, b)
    np.testing.assert_allclose(res, a @ b)

if __name__ == "__main__":
  absltest.main()
