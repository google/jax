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
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
import jax._src.pallas.mosaic_gpu as plgpu
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

  def test_add_xy(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[...]

    x = jnp.arange(256).astype(jnp.float32)
    y = x + 1
    np.testing.assert_array_equal(kernel(x, y), x + y)

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

  def test_add_one_grid_with_scratch(self):

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.float32),
        in_specs=[pl.BlockSpec((128,), lambda *i: i)],
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        scratch_shapes=[plgpu.SMEM((128,), jnp.float32)],
        grid=2,
    )
    def kernel(x_ref, o_ref, scratch_ref):
      scratch_ref[...] = x_ref[...] + 1
      o_ref[...] = scratch_ref[...]

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  @parameterized.product(max_concurrent_steps=[1, 2, 3, 4])
  def test_add_one_grid_pipelined(self, max_concurrent_steps):

    @functools.partial(
        pl.pallas_call,
        in_specs=[pl.BlockSpec((128, 16), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((128, 16), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct([128 * 2, 64], jnp.float32),
        compiler_params=plgpu.GPUCompilerParams(
            dimension_semantics=["parallel", "sequential"],
            max_concurrent_steps=max_concurrent_steps,
        ),
        grid=(2, 4),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(128 * 2 * 64).reshape((128 * 2, 64)).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_with_async_copy_smem_to_gmem(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[plgpu.SMEM((128,), jnp.float32)],
    )
    def kernel(x_ref, o_ref_gmem, scratch_ref):
      scratch_ref[...] = x_ref[...] + 1
      plgpu.async_copy_smem_to_gmem(scratch_ref, o_ref_gmem)
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_with_async_copy_gmem_to_smem(self):

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM((128,), jnp.float32),
            plgpu.Barrier(num_arrivals=1),
        ],
    )
    def kernel(x_ref_gmem, o_ref, scratch_ref, barrier_ref):
      plgpu.async_copy_gmem_to_smem(
          x_ref_gmem, scratch_ref, barrier=barrier_ref
      )
      plgpu.wait_barrier(barrier_ref)
      o_ref[...] = scratch_ref[...] + 1

    x = jnp.arange(128).astype(jnp.float32)
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

  @parameterized.parameters(False, True)
  def test_rsqrt(self, approx_math):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        compiler_params=plgpu.GPUCompilerParams(approx_math=approx_math),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jax.lax.rsqrt(x_ref[...])

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_allclose(kernel(x), jax.lax.rsqrt(x))

  @parameterized.product(input_factor=[0.001, 1, 10, 100, 100])
  def test_layer_norm(self, input_factor):
    eps = 1e-5
    gamma = 1.0
    beta = 1.0

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
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

  def test_print_scalar(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x.sum() = {}", x_ref[...].sum())

    x = jnp.arange(256)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x.sum() = {x.sum()}", output())

  def test_print_scalar_array(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x.sum() = {}", x_ref[...].sum() + 1)

    x = jnp.arange(256)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x.sum() = {x.sum() + 1}", output())

  def test_print_array(self):
    in_shape = [2, 1, 64, 64]

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(in_shape, jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x: {}", x_ref[...])

    x = jnp.arange(math.prod(in_shape)).reshape(in_shape)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x: [1, 0, 43, 23]/{in_shape}: 6871\n", output())

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

  def test_swizzled_blockspec_shapes(self):

    @functools.partial(
        pl.pallas_call,
        in_specs=[
            plgpu.GPUBlockSpec(
                (128, 64),
                lambda *i: i,
                transforms=plgpu.TilingTransform((64, 64)),
                swizzle=128,
            ),
        ],
        out_specs=pl.BlockSpec((2, 1, 64, 64), lambda i, j: (i, j, 64, 64)),
        out_shape=jax.ShapeDtypeStruct((4, 2, 64, 64), jnp.float16),
        grid=(2, 2),
    )
    def kernel(x_ref, o_ref):
      assert x_ref.shape == (2, 1, 64, 64), x_ref.shape
      o_ref[...] = x_ref[...]

    x = jnp.zeros((256, 128), dtype=jnp.float16)
    result = kernel(x)
    self.assertEqual(result.shape, (4, 2, 64, 64))

  def test_fori_loop_array(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      # Equivalent to x_ref[...] + 2 + 3.
      o_ref[...] = jax.lax.fori_loop(2, 4, lambda i, x: x + i, x_ref[...])

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 2.0 + 3.0)

  def test_fori_loop_scalar(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(o_ref):
      # Equivalent to 2 + 3.
      o_ref[...] = jax.lax.broadcast(
          jax.lax.fori_loop(2, 4, lambda i, x: x + i, 0.0), o_ref.shape
      )

    np.testing.assert_array_equal(
        kernel(), jnp.full([256], 5.0, dtype=jnp.float32)
    )

  def test_cond(self):

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      acc = x_ref[...].sum()
      jax.lax.cond(
          acc % 2 == 0,
          lambda: pl.debug_print("acc * 2: {}", acc * 2),
          lambda: pl.debug_print("acc: {}", acc),
      )
      o_ref[...] = jnp.broadcast_to(acc, o_ref.shape)

    x = jnp.arange(256)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn("acc * 2:", output())

  @parameterized.parameters(jnp.float16, jnp.float32)
  def test_wgmma(self, dtype):
    # TensorCores can only fuse transposes of 16-bit values, and RHS
    # is expected to be column major by default.
    rhs_transpose = jnp.dtype(dtype).itemsize != 2
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref, b_ref, rhs_transpose=rhs_transpose)
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 128), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=dtype)
    b = jax.random.uniform(key2, shape=(128, 128), dtype=dtype)

    rhs_transforms = (plgpu.TilingTransform((elems_128b, elems_128b)),)
    if rhs_transpose:
      rhs_transforms += (plgpu.TransposeTransform((1, 0, 2, 3)),)
    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec(
                (64, 128),
                lambda i, j: (i, j),
                transforms=plgpu.TilingTransform((64, elems_128b)),
                swizzle=128,
            ),
            plgpu.GPUBlockSpec(
                (128, 128),
                lambda *i: i,
                transforms=rhs_transforms,
                swizzle=128,
            ),
        ],
        out_specs=plgpu.GPUBlockSpec((64, 128), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
        grid=(1, 1),
    )(a, b)
    np.testing.assert_allclose(
        res, a @ (b.T if rhs_transpose else b), rtol=1e-3
    )

  def test_input_output_aliases(self):
    # Note that we're writing to the input pointer, which should alias b_ptr.
    def kernel(a_ref, b_ref):
      del b_ref
      a_ref[...] = jnp.ones_like(a_ref)

    a = np.zeros((64, 64), dtype=jnp.float32)
    b = pl.pallas_call(
        kernel,
        in_specs=[plgpu.GPUBlockSpec(memory_space=plgpu.GPUMemorySpace.GMEM)],
        out_specs=plgpu.GPUBlockSpec(memory_space=plgpu.GPUMemorySpace.GMEM),
        input_output_aliases={0: 0},
        out_shape=a,
    )(a)
    np.testing.assert_array_equal(b, np.ones_like(a))

  def test_realistic_matmul(self):
    dtype = jnp.float16
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    grid_m, grid_k, grid_n = 132, 10, 4
    tile_m = tile_n = 128
    tile_k = elems_128b
    m, k, n = grid_m * tile_m, grid_k * tile_k, grid_n * tile_n
    def kernel(a_ref, b_ref, o_ref, acc_ref):
      plgpu.wgmma(acc_ref, a_ref, b_ref)
      plgpu.wgmma_wait(0)  # TODO(apaszke): Delay the pipeline to avoid memory races
      # TODO(apaszke): Only store in the last step. It doesn't work because we
      # don't have partial discharge for control flow.
      # is_last_step = pl.program_id(2) == grid_k - 1
      # @pl.when(is_last_step)
      # def _epilogue():
      # pl.debug_print("{}", acc_ref[...])
      # TODO(apaszke): This is an untiled store! It's slow!!
      o_ref[...] = acc_ref[...]

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(m, k), dtype=dtype)
    b = jax.random.uniform(key2, shape=(k, n), dtype=dtype)

    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec(
                (tile_m, tile_k),
                lambda m, n, k: (m, k),
                transforms=plgpu.TilingTransform((64, elems_128b)),
                swizzle=128,
            ),
            plgpu.GPUBlockSpec(
                (tile_k, tile_n),
                lambda m, n, k: (k, n),
                transforms=plgpu.TilingTransform((elems_128b, elems_128b)),
                swizzle=128,
            ),
        ],
        out_specs=plgpu.GPUBlockSpec((tile_m, tile_n), lambda m, n, k: (m, n)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        scratch_shapes=[plgpu.ACC((tile_m, tile_n), jnp.float32)],
        grid=(grid_m, grid_n, grid_k),
        compiler_params=plgpu.GPUCompilerParams(
            dimension_semantics=["parallel", "parallel", "sequential"],
            max_concurrent_steps=2,
        ),
    )(a, b)
    np.testing.assert_allclose(res, a @ b, rtol=1e-3)


if __name__ == "__main__":
  absltest.main()
