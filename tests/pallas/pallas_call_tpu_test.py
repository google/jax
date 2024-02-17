# Copyright 2023 The JAX Authors.
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

"""Test TPU-specific extensions to pallas_call."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import state
from jax._src import test_util as jtu
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import xla_extension
from jax.experimental import mesh_utils
from jax.experimental import mosaic
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu import example_kernel
from jax.extend import linear_util as lu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

partial = functools.partial


class PallasTPUTest(jtu.JaxTestCase):
  interpret: bool = False

  def setUp(self):
    super().setUp()
    if not self.interpret and jtu.device_under_test() != 'tpu':
      self.skipTest('Only interpret mode supported on non-TPU')

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.interpret)


class PallasCallScalarPrefetchTest(PallasTPUTest):

  def test_trivial_scalar_prefetch(self):
    def body(_, x_ref, o_ref):
      o_ref[...] = x_ref[...]

    s = jnp.array([4, 3, 2, 5, 3, 5, 2, 7], jnp.int32)
    x = jnp.arange(8 * 8 * 128, dtype=jnp.int32).reshape((8 * 8, 128))

    def _x_transform(i, s_ref):
      s = pl.load(s_ref, (i,))
      return (s, 0)

    out = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(_x_transform, (x.shape[0] // 8, x.shape[1])),
            ],
            out_specs=pl.BlockSpec(lambda i, _: (i, 0),
                                   (x.shape[0] // 8, x.shape[1])),
            grid=8,
        ),
        interpret=self.interpret,
    )(s, x)
    np.testing.assert_allclose(out, x.reshape((8, 8, -1))[s].reshape(x.shape))

  def test_trivial_scalar_prefetch_with_windowless_args(self):
    def body(_, x_ref, o_ref):
      o_ref[...] = x_ref[...]

    s = jnp.array([4, 3, 2, 5, 3, 5, 2, 7], jnp.int32)
    x = jnp.arange(8 * 8 * 128, dtype=jnp.int32).reshape((8 * 8, 128))

    out = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
        ),
        interpret=self.interpret,
    )(s, x)
    np.testing.assert_array_equal(out, x)

  def test_vmap_scalar_prefetch(self):
    def body(_, x_ref, o_ref):
      o_ref[...] = x_ref[...]

    s = jnp.array([4, 3, 2, 5, 3, 5, 2, 7], jnp.int32)
    x = jnp.arange(2 * 8 * 8 * 128, dtype=jnp.int32).reshape((2, 8 * 8, 128))

    def _x_transform(i, s_ref):
      s = pl.load(s_ref, (i,))
      return (s, 0)

    def f(x):
      return pl.pallas_call(
          body,
          out_shape=jax.ShapeDtypeStruct(x.shape, jnp.int32),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[
                  pl.BlockSpec(_x_transform, (x.shape[0] // 8, x.shape[1])),
              ],
              out_specs=pl.BlockSpec(lambda i, _: (i, 0),
                                     (x.shape[0] // 8, x.shape[1])),
              grid=8,
          ),
          interpret=self.interpret,
      )(s, x)
    np.testing.assert_allclose(
        jax.vmap(f)(x), x.reshape((2, 8, 8, -1))[:, s].reshape(x.shape)
    )

  def test_multiple_scalar_prefetch(self):
    def body(s1_ref, s2_ref, x_ref, o_ref):
      del s1_ref, s2_ref
      o_ref[...] = x_ref[...]

    s1 = jnp.array([4, 3, 2, 5, 3, 5, 2, 7], jnp.int32)
    s2 = jnp.array([7, 6, 5, 4, 3, 2, 1, 0], jnp.int32)
    x = jnp.arange(64 * 128, dtype=jnp.int32).reshape((64, 128))

    def _x_transform(i, s1_ref, _):
      return s1_ref[i], 0

    def _o_transform(i, _, s2_ref):
      return s2_ref[i], 0

    out = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(_x_transform, (8, 128)),
            ],
            out_specs=pl.BlockSpec(_o_transform, (8, 128)),
            grid=8,
        ),
        interpret=self.interpret,
    )(s1, s2, x)
    out_ref = x.reshape((8, 8, -1))[s1][::-1].reshape((64, 128))
    np.testing.assert_allclose(out, out_ref)

  def test_scalar_interpreter(self):
    program = jnp.array([0, 0, 1, 0, 1, 1], jnp.int32)
    x = jnp.arange(8 * 8 * 128.0, dtype=jnp.float32).reshape(8 * 8, 128)

    def body(sprogram_ref, x_ref, o_ref, state_ref):
      x = x_ref[...]

      def add_branch_fn(j):
        state_ref[...] += jnp.float32(j)
        return ()

      def mult_branch_fn(j):
        state_ref[...] *= jnp.float32(j)
        return ()

      def single_inst(i, _):
        _ = jax.lax.switch(
            sprogram_ref[i],
            (
                add_branch_fn,
                mult_branch_fn,
            ),
            i,
        )

      # We can't use for loop state right now, because Pallas functionalizes it,
      # and Mosaic support for returning values form scf.if is incomplete.
      state_ref[...] = x
      lax.fori_loop(0, sprogram_ref.shape[0], single_inst, None, unroll=True)
      o_ref[...] = state_ref[...]

    # Ignore the scratch output.
    out, _ = pl.pallas_call(
        body,
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, jnp.float32),
            jax.ShapeDtypeStruct((8, 128), jnp.float32),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(lambda i, *_: (i, 0), (8, 128))],
            out_specs=[
                pl.BlockSpec(lambda i, *_: (i, 0), (8, 128)),
                pl.BlockSpec(lambda *_: (0, 0), (8, 128)),
            ],
            grid=8,
        ),
        interpret=self.interpret,
        debug=False,
    )(program, x)

    expected = x
    for i, p in enumerate(program):
      if p == 0:
        expected += i
      elif p == 1:
        expected *= i

    np.testing.assert_allclose(out, expected)

  def test_scalar_interpreter_dynamic_loop(self):
    loop_end = jnp.array([5], jnp.int32)

    def body(loop_end_ref, out_ref):
      out_ref[...] = jnp.zeros_like(out_ref)

      def loop_body(i, carry):
        del i, carry
        out_ref[...] += 1

      lax.fori_loop(0, loop_end_ref[0], loop_body, None)

    out = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            out_specs=pl.BlockSpec(lambda *_: (0, 0), (8, 128)),
            grid=1,
        ),
        interpret=self.interpret,
        debug=False,
    )(loop_end)

    expected_out = jnp.ones((8, 128), jnp.float32) * 5
    np.testing.assert_allclose(out, expected_out)

  def test_vmap_scalar_prefetch_1sized(self):
    def body(_, x_ref, o_ref):
      o_ref[...] = x_ref[...]

    s = jnp.array([4, 3, 2, 5, 3, 5, 2, 7], jnp.int32)
    x = jnp.arange(8 * 8 * 128, dtype=jnp.int32).reshape((8 * 8, 128))

    def _x_transform(i, s_ref):
      s = pl.load(s_ref, (i,))
      return (s, 0)

    s = s[None]
    x = x[None]

    out = jax.vmap(pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(_x_transform, (x.shape[1] // 8, x.shape[2])),
            ],
            out_specs=pl.BlockSpec(lambda i, _: (i, 0),
                                   (x.shape[1] // 8, x.shape[2])),
            grid=8,
        ),
        interpret=self.interpret,
    ))(s, x)
    np.testing.assert_allclose(
        out, x.reshape((1, 8, 8, -1))[:, s].reshape(x.shape)
    )

  def test_nontrivial_vmap_scalar_prefetch(self):
    def body(_, x_ref, o_ref):
      o_ref[...] = x_ref[...]

    s = jnp.array([4, 3, 2, 5, 3, 5, 2, 7], jnp.int32)
    x = jnp.arange(8 * 8 * 128, dtype=jnp.int32).reshape((8 * 8, 128))

    def _x_transform(i, s_ref):
      s = pl.load(s_ref, (i,))
      return (s, 0)

    s = jnp.tile(s[None], [2, 1])
    x = jnp.tile(x[None], [2, 1, 1])

    with self.assertRaises(NotImplementedError):
      jax.vmap(
          pl.pallas_call(
              body,
              out_shape=jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
              grid_spec=pltpu.PrefetchScalarGridSpec(
                  num_scalar_prefetch=1,
                  in_specs=[
                      pl.BlockSpec(_x_transform, (x.shape[1] // 8, x.shape[2])),
                  ],
                  out_specs=pl.BlockSpec(
                      lambda i, _: (i, 0), (x.shape[1] // 8, x.shape[2])
                  ),
                  grid=8,
              ),
              interpret=self.interpret,
          )
      )(s, x)


class PallasCallScalarPrefetchInterpretTest(PallasCallScalarPrefetchTest):
  interpret: bool = True


class PallasCallDynamicGridTest(PallasTPUTest):

  def test_dynamic_grid(self):
    shape = (8, 128)
    result_ty = jax.ShapeDtypeStruct(shape, jnp.float32)

    def kernel(y_ref):
      @pl.when(pl.program_id(0) == 0)
      def _init():
        y_ref[...] = jnp.zeros_like(y_ref)
      y_ref[...] += 1

    @jax.jit
    def dynamic_kernel(steps):
      return self.pallas_call(
          kernel,
          grid=(steps * 2,),
          out_specs=pl.BlockSpec(lambda i: (0, 0), shape),
          out_shape=result_ty,
      )()
    np.testing.assert_array_equal(
        dynamic_kernel(jnp.int32(4)), np.full(shape, 8.0, np.float32)
    )

  # TODO(apaszke): Add tests for scalar_prefetch too
  def test_dynamic_grid_scalar_input(self):
    shape = (8, 128)
    result_ty = jax.ShapeDtypeStruct(shape, jnp.float32)

    def kernel(scalar_input_ref, output_ref):
      output_ref[...] = jnp.full_like(output_ref, scalar_input_ref[0, 0])

    @jax.jit
    def dynamic_kernel(steps):
      return self.pallas_call(
          kernel,
          out_shape=result_ty,
          in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
          out_specs=pl.BlockSpec(lambda i: (0, 0), shape),
          grid=(steps * 2,),
      )(jnp.array([[42]], dtype=jnp.int32))

    np.testing.assert_array_equal(
        dynamic_kernel(jnp.int32(4)), np.full(shape, 42.0, np.float32)
    )

  def test_vmap_trivial_dynamic_grid(self):
    shape = (8, 128)
    result_ty = jax.ShapeDtypeStruct(shape, jnp.float32)

    def kernel(x_ref, y_ref):
      @pl.when(pl.program_id(0) == 0)
      def _init():
        y_ref[...] = x_ref[...]
      y_ref[...] += 1

    @jax.jit
    @jax.vmap
    def dynamic_kernel(steps, x):
      return self.pallas_call(
          kernel,
          grid=(steps * 2,),
          out_specs=pl.BlockSpec(lambda i: (0, 0), shape),
          out_shape=result_ty,
      )(x)
    x = jnp.arange(8 * 128., dtype=jnp.float32).reshape((1, *shape))
    np.testing.assert_array_equal(
        dynamic_kernel(jnp.array([4], jnp.int32), x), x + 8.0
    )

  def test_vmap_nontrivial_dynamic_grid(self):
    # Dynamic grid doesn't support vmapping over multiple distinct grid values
    # at the moment.
    shape = (8, 128)
    result_ty = jax.ShapeDtypeStruct(shape, jnp.float32)

    def kernel(y_ref):
      @pl.when(pl.program_id(0) == 0)
      def _init():
        y_ref[...] = jnp.zeros_like(y_ref)
      y_ref[...] += 1

    @jax.jit
    @jax.vmap
    def dynamic_kernel(steps):
      return self.pallas_call(
          kernel,
          grid=(steps * 2,),
          out_specs=pl.BlockSpec(lambda i: (0, 0), shape),
          out_shape=result_ty,
      )()
    with self.assertRaises(NotImplementedError):
      dynamic_kernel(jnp.array([4, 8], jnp.int32))

  def test_vmap_dynamic_grid(self):
    shape = (8, 128)
    result_ty = jax.ShapeDtypeStruct(shape, jnp.float32)

    def kernel(x_ref, y_ref):
      @pl.when(pl.program_id(0) == 0)
      def _init():
        y_ref[...] = x_ref[...]
      y_ref[...] += jnp.float32(1.)

    @jax.jit
    def dynamic_kernel(x, steps):
      return self.pallas_call(
          kernel,
          grid=(steps * 2,),
          out_specs=pl.BlockSpec(lambda i: (0, 0), shape),
          out_shape=result_ty,
      )(x)
    x = jnp.arange(4 * 8 * 128., dtype=jnp.float32).reshape((4, *shape))
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(dynamic_kernel, in_axes=(0, None)))(x, jnp.int32(4)),
        x + 8,
    )


class PallasCallInterpretDynamicGridTest(PallasCallDynamicGridTest):
  interpret: bool = True


class PallasCallDMATest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('DMAs not supported on TPU generations <= 3')

  def test_can_have_unspecified_memory_spaces(self):
    def kernel(x_ref, y_ref):
      # Just test whether things compile
      del x_ref, y_ref

    x = jnp.ones((8, 128), dtype=jnp.float32)
    y = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(None, None, pltpu.TPUMemorySpace.ANY)],
        out_specs=pl.BlockSpec(None, None, pltpu.TPUMemorySpace.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    jax.block_until_ready(y)

  def test_run_scoped_tracks_effects(self):
    def kernel(x_ref, y_ref):
      def body(temp_ref):
        temp_ref[...] = jnp.ones_like(temp_ref)
        x_ref[...] = 4 * y_ref[...] + temp_ref[...]

      pltpu.run_scoped(body, pltpu.VMEM((8,), jnp.float32))
      return []

    jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(kernel),
        [
            state.shaped_array_ref((8,), jnp.float32),
            state.shaped_array_ref((8,), jnp.float32),
        ],
    )
    expected_effects = {state.ReadEffect(1), state.WriteEffect(0)}
    self.assertSetEqual(jaxpr.effects, expected_effects)

  def test_scoped_allocation(self):
    def kernel(y_ref):
      def body(x_ref):
        x_ref[...] = jnp.ones_like(x_ref)
        y_ref[...] = 4 * x_ref[...]

      pltpu.run_scoped(body, pltpu.VMEM((8, 128), jnp.float32))

    o = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )()
    np.testing.assert_allclose(o, 4 * np.ones_like(o))

  def test_nested_scoped_allocation(self):
    def kernel(y_ref):
      def body(x_ref):
        x_ref[...] = jnp.zeros_like(x_ref)
        def inner_body(z_ref):
          z_ref[...] = jnp.ones_like(z_ref)
          x_ref[...] = z_ref[...]
        pltpu.run_scoped(inner_body, pltpu.VMEM((8, 128), jnp.float32))
        y_ref[...] = 4 * x_ref[...]
      pltpu.run_scoped(body, pltpu.VMEM((8, 128), jnp.float32))

    o = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )()
    np.testing.assert_allclose(o, 4 * np.ones_like(o))

  def test_can_allocate_semaphore(self):
    def kernel(y_ref):
      def body(sem1):
        pass
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA)

    jax.block_until_ready(pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )())

  def test_can_allocate_multiple_semaphores(self):
    def kernel(y_ref):
      def body(sem1, sem2):
        pass
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA,
                       pltpu.SemaphoreType.REGULAR)

    jax.block_until_ready(pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )())

  def test_can_allocate_semaphore_array(self):
    def kernel(y_ref):
      def body(dma_sems, sems):
        self.assertTupleEqual(dma_sems.shape, (4,))
        self.assertTupleEqual(sems.shape, (3,))
        self.assertTrue(jnp.issubdtype(dma_sems.dtype, pltpu.dma_semaphore))
        self.assertTrue(jnp.issubdtype(sems.dtype, pltpu.semaphore))
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA((4,)),
                       pltpu.SemaphoreType.REGULAR((3,)))

    jax.block_until_ready(pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )())

  def test_can_allocate_scratch_semaphore_array(self):
    def kernel(y_ref, dma_sems, sems):
      self.assertTupleEqual(dma_sems.shape, (4,))
      self.assertTupleEqual(sems.shape, (3,))
      self.assertTrue(jnp.issubdtype(dma_sems.dtype, pltpu.dma_semaphore))
      self.assertTrue(jnp.issubdtype(sems.dtype, pltpu.semaphore))

    jax.block_until_ready(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                scratch_shapes=[
                    pltpu.SemaphoreType.DMA((4,)),
                    pltpu.SemaphoreType.REGULAR((3,)),
                ],
            ),
        )()
    )

  def test_can_wait_on_semaphore(self):
    def kernel(y_ref):
      def body(sem):
        pltpu.semaphore_signal(sem)
        pltpu.semaphore_wait(sem)
      pltpu.run_scoped(body, pltpu.SemaphoreType.REGULAR)
      def body2(sem):
        pltpu.semaphore_signal(sem, 2)
        pltpu.semaphore_wait(sem)
        pltpu.semaphore_wait(sem)
      pltpu.run_scoped(body2, pltpu.SemaphoreType.REGULAR)
      def body3(sem):
        pltpu.semaphore_signal(sem)
        pltpu.semaphore_signal(sem)
        pltpu.semaphore_signal(sem)
        pltpu.semaphore_wait(sem)
        pltpu.semaphore_wait(sem)
        pltpu.semaphore_wait(sem)
      pltpu.run_scoped(body3, pltpu.SemaphoreType.REGULAR)

    jax.block_until_ready(pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )())

  def test_can_wait_on_semaphore_array(self):
    def kernel(y_ref):
      def body(sems):
        pltpu.semaphore_signal(sems.at[0])
        pltpu.semaphore_wait(sems.at[0])

        pltpu.semaphore_signal(sems.at[1], 2)
        pltpu.semaphore_wait(sems.at[1])
        pltpu.semaphore_wait(sems.at[1])

        pltpu.semaphore_signal(sems.at[2])
        pltpu.semaphore_signal(sems.at[2])
        pltpu.semaphore_signal(sems.at[2])
        pltpu.semaphore_wait(sems.at[2])
        pltpu.semaphore_wait(sems.at[2])
        pltpu.semaphore_wait(sems.at[2])
      pltpu.run_scoped(body, pltpu.SemaphoreType.REGULAR((3,)))

    jax.block_until_ready(pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )())

  def test_can_wait_on_semaphore_array_with_dynamic_index(self):
    def kernel(y_ref):
      i = pl.program_id(0)
      def body(sems):
        pltpu.semaphore_signal(sems.at[i, 0])
        pltpu.semaphore_wait(sems.at[i, 0])

        pltpu.semaphore_signal(sems.at[i, 1], 2)
        pltpu.semaphore_wait(sems.at[i, 1])
        pltpu.semaphore_wait(sems.at[i, 1])

        pltpu.semaphore_signal(sems.at[i, 2])
        pltpu.semaphore_signal(sems.at[i, 2])
        pltpu.semaphore_signal(sems.at[i, 2])
        pltpu.semaphore_wait(sems.at[i, 2])
        pltpu.semaphore_wait(sems.at[i, 2])
        pltpu.semaphore_wait(sems.at[i, 2])
      pltpu.run_scoped(body, pltpu.SemaphoreType.REGULAR((4, 3)))

    jax.block_until_ready(pl.pallas_call(
        kernel,
        in_specs=[],
        out_specs=pl.BlockSpec(lambda i: (0, 0), (8, 128)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid=4,
        debug=True,
    )())

  def test_hbm_hbm_dma(self):
    def kernel(x_hbm_ref, y_hbm_ref):
      def body(sem):
        pltpu.async_copy(x_hbm_ref.at[pl.ds(8), :], y_hbm_ref.at[:, pl.ds(128)],
                         sem).wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA)
    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_array_equal(y, x)

  def test_cannot_dma_with_nonscalar_semaphore_ref(self):
    def kernel(x_hbm_ref, y_hbm_ref):
      def body(sem):
        pltpu.async_copy(x_hbm_ref.at[pl.ds(8), :], y_hbm_ref.at[:, pl.ds(128)],
                         sem).wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA((1,)))
    with self.assertRaisesRegex(ValueError, 'Cannot signal'):
      x = jnp.arange(8 * 128.).reshape((8, 128))
      pl.pallas_call(
          kernel,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
      )(x)

  def test_dma_with_scalar_semaphore_ref(self):
    def kernel(x_hbm_ref, y_hbm_ref):
      def body(sem):
        pltpu.async_copy(x_hbm_ref.at[pl.ds(8), :], y_hbm_ref.at[:, pl.ds(128)],
                         sem.at[0]).wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA((1,)))
    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_array_equal(y, x)

  def test_hbm_hbm_grid_dma(self):
    # When using the grid, we have to emit Mosaic window_params. Test that they
    # work correctly with ANY memory space operands.
    def kernel(x_hbm_ref, y_hbm_ref):
      i = pl.program_id(0)
      def body(sem):
        pltpu.async_copy(
            x_hbm_ref.at[pl.ds(i, 1)], y_hbm_ref.at[pl.ds(i, 1)], sem
        ).wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA)
    x = jnp.arange(2 * 8 * 128.).reshape((2, 8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        out_shape=jax.ShapeDtypeStruct((2, 8, 128), jnp.float32),
        grid=(2,),
    )(x)
    np.testing.assert_allclose(y, x)

  def test_hbm_vmem_dma(self):
    def kernel(x_hbm_ref, y_ref):
      def body(x_ref, sem):
        pltpu.async_copy(x_hbm_ref.at[pl.ds(8), :], x_ref.at[:, pl.ds(128)],
                         sem).wait()
        y_ref[...] = x_ref[...]
      pltpu.run_scoped(body, pltpu.VMEM((8, 128), jnp.float32),
                       pltpu.SemaphoreType.DMA)
    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_allclose(y, x)

  def test_vmem_hbm_dma(self):
    def kernel(x_ref, y_hbm_ref):
      def body(y_ref, sem):
        y_ref[...] = x_ref[...]
        pltpu.async_copy(y_hbm_ref, y_ref, sem).wait()
      pltpu.run_scoped(body, pltpu.VMEM((8, 128), jnp.float32),
                       pltpu.SemaphoreType.DMA)
    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_allclose(y, x)

  def test_vmem_hbm_vmem_dma(self):
    def kernel(x_hbm_ref, y_hbm_ref):
      def body(x_ref, y_ref, sem):
        pltpu.async_copy(x_hbm_ref, x_ref, sem).wait()
        y_ref[...] = x_ref[...]
        pltpu.async_copy(y_ref, y_hbm_ref, sem).wait()
      pltpu.run_scoped(body,
                       pltpu.VMEM((8, 128), jnp.float32),
                       pltpu.VMEM((8, 128), jnp.float32),
                       pltpu.SemaphoreType.DMA)
    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_allclose(y, x)

  def test_hbm_smem_dma(self):
    def kernel(x_hbm_ref, y_ref):
      def body(x_ref, sem):
        pltpu.async_copy(x_hbm_ref, x_ref, sem).wait()
        y_ref[...] = x_ref[0, 0] * jnp.ones_like(y_ref)
      pltpu.run_scoped(body, pltpu.SMEM((8, 128), jnp.float32),
                       pltpu.SemaphoreType.DMA)
    x = 4 * jnp.ones((8, 128), jnp.float32)
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_allclose(y, x)

  def test_smem_hbm_dma(self):
    def kernel(x_ref, y_hbm_ref):
      def body(y_ref, sem):
        y_ref[0, 0] = x_ref[4, 4]
        pltpu.async_copy(y_ref, y_hbm_ref, sem).wait()
      pltpu.run_scoped(body, pltpu.SMEM((8, 128), jnp.float32),
                       pltpu.SemaphoreType.DMA)
    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.SMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    expected = jnp.zeros_like(x).at[0, 0].set(x[4, 4])
    np.testing.assert_allclose(y, expected)

  def test_vmem_vmem_dma(self):
    def kernel(x_ref, y_ref):
      def body(sem):
        pltpu.async_copy(x_ref, y_ref, sem).wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA)
    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_allclose(y, x)

  def test_hbm_vmem_dma_slicing(self):
    def kernel(x_hbm_ref, y_ref):
      def body(sem):
        dma1 = pltpu.async_copy(
            x_hbm_ref.at[pl.ds(0, 8)], y_ref.at[pl.ds(0, 8)], sem
        )
        dma2 = pltpu.async_copy(
            x_hbm_ref.at[pl.ds(8, 8)], y_ref.at[pl.ds(8, 8)], sem
        )
        dma1.wait()
        dma2.wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA)
    x = jnp.arange(2 * 8 * 128.).reshape((16, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
        out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
    )(x)
    np.testing.assert_allclose(y, x)

  def test_hbm_vmem_dma_indexing(self):
    def kernel(x_hbm_ref, y_ref):
      def body(sem):
        dma1 = pltpu.async_copy(
            x_hbm_ref.at[0], y_ref.at[pl.ds(0, 8)], sem
        )
        dma2 = pltpu.async_copy(
            x_hbm_ref.at[1], y_ref.at[pl.ds(8, 8)], sem
        )
        dma1.wait()
        dma2.wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA)
    x = jnp.arange(2 * 8 * 128.).reshape((2, 8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
        out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
    )(x)
    np.testing.assert_allclose(y, x.reshape((16, 128)))

  def test_hbm_vmem_dma_multiple_indexing(self):
    def kernel(x_hbm_ref, y_ref):
      def body(sem):
        for i in range(3):
          dma1 = pltpu.async_copy(
              x_hbm_ref.at[pl.ds(i, 1)].at[0, 0], y_ref.at[i].at[pl.ds(0, 8)],
              sem
          )
          dma2 = pltpu.async_copy(
              x_hbm_ref.at[pl.ds(i, 1)].at[0, 1], y_ref.at[i].at[pl.ds(8, 8)],
              sem
          )
          dma1.wait()
          dma2.wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA)
    x = jnp.arange(3 * 2 * 8 * 128.).reshape((3, 2, 8, 128))
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
        out_shape=jax.ShapeDtypeStruct((3, 16, 128), jnp.float32),
    )(x)
    np.testing.assert_allclose(y, x.reshape((3, 16, 128)))

  def test_cannot_squeeze_lane_sublane(self):
    def kernel(x_hbm_ref, y_ref):
      def body(sem):
        dma1 = pltpu.async_copy(
            x_hbm_ref.at[:, :, 0], y_ref.at[pl.ds(0, 8)], sem
        )
        dma2 = pltpu.async_copy(
            x_hbm_ref.at[:, :, 1], y_ref.at[pl.ds(8, 8)], sem
        )
        dma1.wait()
        dma2.wait()
      pltpu.run_scoped(body, pltpu.SemaphoreType.DMA)
    x = jnp.arange(2 * 8 * 128.).reshape((2, 8, 128))
    with self.assertRaises(Exception):
      _ = pl.pallas_call(
          kernel,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
          out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
      )(x)

  @parameterized.named_parameters(
      ('', False),
      ('_interpret', True),
  )
  def test_hoisted_scratch_space(self, interpret):
    def kernel(x_ref, y_ref, scratch_ref):
      i = pl.program_id(0)
      @pl.when(i == 0)
      def _():
        scratch_ref[...] = x_ref[...]
      scratch_ref[...] += jnp.ones_like(scratch_ref)

      @pl.when(i == 2)
      def _():
        y_ref[...] = scratch_ref[...]

    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(lambda i: (0, 0), (8, 128)),
            ],
            scratch_shapes=[pltpu.VMEM((8, 128), jnp.float32)],
            out_specs=pl.BlockSpec(lambda i: (0, 0), (8, 128)),
            grid=(3,),
        ),
        interpret=interpret,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_array_equal(y, x + 3)

  def test_hoisted_smem_space(self):
    # TODO(sharadmv,apaszke): enable SMEM scratch spaces
    # TODO(sharadmv,apaszke): add support for ()-shaped SMEM refs
    self.skipTest('Currently doesn\'t work')
    def kernel(y_ref, scratch_ref):
      scratch_ref[0, 0] = pl.program_id(0)
      y_ref[...] = jnp.broadcast_to(scratch_ref[0, 0], y_ref.shape)

    y = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[],
            scratch_shapes=[pltpu.SMEM((1, 1), jnp.int32)],
            out_specs=pl.BlockSpec(lambda i: (i, 0, 0), (None, 8, 128)),
            grid=(2,),
        ),
        debug=True,
        out_shape=jax.ShapeDtypeStruct((2, 8, 128), jnp.int32),
    )()
    expected = jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32)[..., None, None],
                                (2, 8, 128))
    np.testing.assert_array_equal(y, expected)

  def test_hoisted_semaphore(self):
    def kernel(x_bbm_ref, y_ref, sem, dma_sem):
      pltpu.semaphore_signal(sem)
      pltpu.semaphore_wait(sem)
      pltpu.async_copy(x_bbm_ref, y_ref, dma_sem).wait()

    x = jnp.arange(8 * 128.).reshape((8, 128))
    y = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
            ],
            scratch_shapes=[pltpu.SemaphoreType.REGULAR,
                            pltpu.SemaphoreType.DMA],
            out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
        ),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    np.testing.assert_array_equal(y, x)


class PallasCallRemoteDMATest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest('Only >=2 devices are supported.')
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5')

  @parameterized.named_parameters(
      ('vmem', pltpu.TPUMemorySpace.VMEM),
      ('hbm', pltpu.TPUMemorySpace.ANY),
  )
  def test_basic_remote_vmem_dma(self, mem):
    # Implements very simple collective permute
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        dev_id = pltpu.device_id()
        other_dev_id = 1 - dev_id
        pltpu.semaphore_signal(ready_sem, device_id=other_dev_id,
                               device_id_type=pltpu.DeviceIdType.LOGICAL)
        pltpu.semaphore_wait(ready_sem)
        copy_done = pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, other_dev_id,
            device_id_type=pltpu.DeviceIdType.LOGICAL,
        )
        copy_done.wait_send()
        copy_done.wait_recv()

      pltpu.run_scoped(body, pltpu.SemaphoreType.REGULAR,
                       pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA)

    x = jnp.arange(2 * 8 * 128.0).reshape((2 * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=mem)],
          out_specs=pl.BlockSpec(memory_space=mem),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh, in_specs=P('x'), out_specs=P('x'), check_rep=False
        )
    )(x)
    expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(y, expected)

  @parameterized.named_parameters(
      ('left', 'left'),
      ('right', 'right')
  )
  def test_pallas_call_axis_index(self, direction):
    # Implements very simple collective permute
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        my_id = lax.axis_index('x')
        num_devices = lax.psum(1, 'x')
        if direction == 'right':
          neighbor = lax.rem(my_id + 1, num_devices)
        else:
          neighbor = lax.rem(my_id - 1, num_devices)
          # Neighbor might be negative here so we add num_devices in case
          neighbor = jnp.where(neighbor < 0, neighbor + num_devices, neighbor)
        pltpu.semaphore_signal(ready_sem, device_id=neighbor)
        pltpu.semaphore_wait(ready_sem)
        copy_done = pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=neighbor
        )
        copy_done.wait_send()
        copy_done.wait_recv()

      pltpu.run_scoped(body, pltpu.SemaphoreType.REGULAR,
                       pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA)

    num_devices = jax.local_device_count()
    x = jnp.arange(num_devices * 8 * 128).reshape((num_devices * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
          out_shape=x,
      )(x)

    device_mesh = mesh_utils.create_device_mesh(
        (jax.device_count(),), jax.devices())
    mesh = jax.sharding.Mesh(device_mesh, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh, in_specs=P('x'), out_specs=P('x'), check_rep=False
        )
    )(x)
    if direction == 'right':
      expected = jnp.concatenate([x[-8:], x[:-8]])
    else:
      expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(y, expected)

  @parameterized.named_parameters(('left', 'left'), ('right', 'right'))
  def test_pallas_call_axis_index_2d_mesh(self, direction):
    # Implements very simple collective permute in a 2D mesh.
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        my_id = lax.axis_index('x')
        my_other_id = lax.axis_index('y')
        axis_size = lax.psum(1, 'x')
        if direction == 'right':
          neighbor = lax.rem(my_id + 1, axis_size)
        else:
          neighbor = lax.rem(my_id - 1, axis_size)
          # Neighbor might be negative here so we add num_devices in case
          neighbor = jnp.where(neighbor < 0, neighbor + axis_size, neighbor)
        pltpu.semaphore_signal(ready_sem, device_id=(my_other_id, neighbor))
        pltpu.semaphore_wait(ready_sem)
        copy_done = pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=(my_other_id, neighbor)
        )
        copy_done.wait_send()
        copy_done.wait_recv()

      pltpu.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

    axis_size = jax.device_count() // 2
    x = jnp.arange(axis_size * 8 * 128).reshape((axis_size * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
          out_shape=x,
      )(x)

    device_mesh = mesh_utils.create_device_mesh(
        (2, axis_size), jax.devices()
    )
    mesh = jax.sharding.Mesh(device_mesh, ['y', 'x'])
    y = jax.jit(
        shard_map.shard_map(
            body,
            mesh,
            in_specs=P('x', None),
            out_specs=P('x', None),
            check_rep=False,
        )
    )(x)
    if direction == 'right':
      expected = jnp.concatenate([x[-8:], x[:-8]])
    else:
      expected = jnp.concatenate([x[8:], x[:8]])
    np.testing.assert_allclose(y, expected)

  def test_barrier_semaphore(self):
    def kernel(x_ref, y_ref):
      def body(ready_sem, send_sem, recv_sem):
        my_id = lax.axis_index('x')
        num_devices = lax.psum(1, 'x')
        neighbor = lax.rem(my_id + 1, num_devices)
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(barrier_sem, device_id=neighbor)
        pltpu.semaphore_wait(barrier_sem)
        pltpu.semaphore_signal(ready_sem, device_id=neighbor)
        pltpu.semaphore_wait(ready_sem)
        pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=neighbor
        ).wait()

      pltpu.run_scoped(body, pltpu.SemaphoreType.REGULAR,
                       pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA)

    num_devices = jax.local_device_count()
    x = jnp.arange(num_devices * 8 * 128).reshape((num_devices * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
          out_shape=x,
          mosaic_params=dict(collective_id=0)
      )(x)

    device_mesh = mesh_utils.create_device_mesh(
        (jax.device_count(),), jax.devices())
    mesh = jax.sharding.Mesh(device_mesh, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh, in_specs=P('x'), out_specs=P('x'), check_rep=False
        )
    )(x)
    expected = jnp.concatenate([x[-8:], x[:-8]])
    np.testing.assert_allclose(y, expected)


class PallasCallTest(PallasTPUTest):

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() != 'tpu':
      self.skipTest('Test only works on TPU')

  def test_cost_analysis(self):
    def kernel(x, y):
      y[:] = x[:]
    x = jnp.arange(1024.).reshape(8, 128)
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        mosaic_params=dict(
            cost_estimate=pltpu.CostEstimate(
                flops=1234, transcendentals=21, bytes_accessed=12345
            )
        ),
    )
    (analysis_result,) = jax.jit(f).lower(x).compile().cost_analysis()
    self.assertEqual(analysis_result['flops'], 1234)
    self.assertEqual(analysis_result['transcendentals'], 21)
    self.assertEqual(analysis_result['bytes accessed'], 12345)

  def test_vmem_limit(self):
    shape = (128, 128)

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...]

    x = jnp.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with self.assertRaises(xla_extension.XlaRuntimeError):
      pl.pallas_call(
          kernel, out_shape=x, mosaic_params=dict(vmem_limit_bytes=256)
      )(x)
    pl.pallas_call(
        kernel, out_shape=x, mosaic_params=dict(vmem_limit_bytes=int(2**18))
    )(x)


class PallasCallUnblockedIndexingTest(PallasTPUTest):

  def setUp(self):
    super().setUp()
    if not self.interpret and jtu.device_under_test() != 'tpu':
      self.skipTest('Only interpret mode supported on non-TPU')

  def test_unblocked_indexing(self):
    shape = (16 * 8, 128)
    result_ty = jax.ShapeDtypeStruct((15 * 8, 128), jnp.float32)

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[pl.ds(0, 8)] + x_ref[pl.ds(8, 8)]

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y = pl.pallas_call(
        kernel,
        grid=(15,),
        in_specs=(
            pl.BlockSpec(
                lambda i: (i * 8, 0), (2 * 8, 128), indexing_mode=pl.unblocked
            ),
        ),
        out_specs=pl.BlockSpec(lambda i: (i, 0), (8, 128)),
        out_shape=result_ty,
        interpret=self.interpret,
    )(x)
    ref = []
    for i in range(15):
      ref.append(x[i * 8:(i + 1) * 8] + x[(i + 1) * 8:(i + 2) * 8])
    ref = np.concatenate(ref, axis=0)
    np.testing.assert_array_equal(y, ref)

  def test_unblocked_indexing_with_padding(self):
    shape = (8, 128)
    result_ty = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[pl.ds(0, 8)]

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y = pl.pallas_call(
        kernel,
        grid=(1,),
        in_specs=(
            pl.BlockSpec(
                lambda i: (0, 0),
                (2 * 8, 128),
                indexing_mode=pl.Unblocked(((0, 8), (0, 0))),
            ),
        ),
        out_specs=pl.BlockSpec(lambda i: (0, 0), (8, 128)),
        out_shape=result_ty,
        interpret=self.interpret,
    )(x)
    np.testing.assert_array_equal(y, x)


class PallasCallInterpreterUnblockedIndexingTest(
    PallasCallUnblockedIndexingTest
):
  interpret = True


class PallasUXTest(PallasTPUTest):

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() != 'tpu':
      self.skipTest('Test only works on TPU')

  def test_mlir_location(self):
    # Make sure that MLIR locations are correctly propagated to primitives.
    args = (jax.ShapeDtypeStruct((8, 128), jnp.float32),)
    f = example_kernel.double
    as_tpu_kernel = mosaic.as_tpu_kernel
    def capture_as_tpu_kernel(module, *args, **kwargs):
      asm = module.operation.get_asm(enable_debug_info=True)
      self.assertIn('example_kernel.py":25', asm)
      return as_tpu_kernel(module, *args, **kwargs)
    mosaic.as_tpu_kernel = capture_as_tpu_kernel
    try:
      jax.jit(f).lower(*args)
    finally:
      mosaic.as_tpu_kernel = as_tpu_kernel


class PallasCallInputOutputAliasingTest(PallasTPUTest):

  def setUp(self):
    super().setUp()
    if not self.interpret and jtu.device_under_test() != 'tpu':
      self.skipTest('Only interpret mode supported on non-TPU')

  def test_basic_input_output_aliasing(self):
    # Input needs to be big so it doesn't fit in VMEM
    x = jnp.ones((32, 1024, 1024))
    expected = x + 1

    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...] + 1.
    @partial(jax.jit, donate_argnums=(0,))
    def f(x):
      return pl.pallas_call(
          kernel,
          out_shape=x,
          in_specs=[pl.BlockSpec(lambda i: (i, 0, 0), (None, 1024, 1024))],
          out_specs=pl.BlockSpec(lambda i: (i, 0, 0), (None, 1024, 1024)),
          grid=(x.shape[0],),
          input_output_aliases={0: 0},
          interpret=self.interpret,
      )(x)
    o = f(x)
    np.testing.assert_array_equal(o, expected)
    compiled = f.lower(x).compile()
    mem_analysis = compiled.memory_analysis()
    expected_num_bytes = np.prod(x.shape) * x.dtype.itemsize
    self.assertEqual(mem_analysis.alias_size_in_bytes, expected_num_bytes)
    self.assertEqual(mem_analysis.temp_size_in_bytes, 0)

  def test_input_output_aliasing_with_scalar_prefetch(self):
    x = jnp.ones((32, 1024, 1024))
    expected = x + 1

    def kernel(_, x_ref, y_ref):
      y_ref[...] = x_ref[...] + 1.
    @partial(jax.jit, donate_argnums=(0,))
    def f(x):
      return pl.pallas_call(
          kernel,
          out_shape=x,
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(lambda i, _: (i, 0, 0), (None, 1024, 1024))],
              out_specs=pl.BlockSpec(lambda i, _: (i, 0, 0), (None, 1024, 1024)),
              grid=(x.shape[0],),
          ),
          input_output_aliases={1: 0},
          interpret=self.interpret,
      )(jnp.array([1,2,3]), x)
    o = f(x)
    np.testing.assert_array_equal(o, expected)
    compiled = f.lower(x).compile()
    mem_analysis = compiled.memory_analysis()
    expected_num_bytes = np.prod(x.shape) * x.dtype.itemsize
    self.assertEqual(mem_analysis.alias_size_in_bytes, expected_num_bytes)
    self.assertEqual(mem_analysis.temp_size_in_bytes, 0)


class PallasCallInterpreterInputOutputAliasingTest(PallasTPUTest):
  interpret: bool = True


class PallasMegacoreTest(PallasTPUTest):

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() != 'tpu':
      self.skipTest('Test only works on TPU')

  def test_megacore_splitting(self):
    # We want to make sure a 3-sized dimension is split across megacore
    # correctly, and if we combine the (3, 3) dimensions together it is still
    # correct.

    def matmul_kernel(x_ref, y_ref, z_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        z_ref[...] = jnp.zeros_like(z_ref)
      z_ref[...] += x_ref[...] @ y_ref[...]

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, (3, 3, 512, 512))
    y = jax.random.uniform(k2, (3, 3, 512, 512))

    z = jax.vmap(jax.vmap(
        pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((512, 512), jnp.float32),
            grid=(4, 4, 4),
            in_specs=[
                pl.BlockSpec(lambda i, j, k: (i, k), (128, 128)),
                pl.BlockSpec(lambda i, j, k: (k, j), (128, 128)),
            ],
            out_specs=pl.BlockSpec(lambda i, j, k: (i, j), (128, 128)),
            debug=True,
        )
    ))(x, y)
    np.testing.assert_allclose(z, jax.vmap(jax.vmap(jnp.dot))(x, y))


class PallasCallVmapTest(PallasTPUTest):

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() != 'tpu':
      self.skipTest('Test only works on TPU')

  def test_scratch_input_vmap(self):
    """Test that vmapp-ing a kernel with scratch inputs works correctly."""

    # Scratch inputs are only available for PallasTPU. This is why this test
    # does not live with the other vmap tests in:
    # jax/tests/pallas/pallas_test.py
    def add_one_with_scratch(x_ref, o_ref, scratch_ref):
      scratch_ref[...] = jnp.ones_like(scratch_ref[...])
      o_ref[...] = x_ref[...] + scratch_ref[...]

    tile_size = 128
    tile_shape = (tile_size, tile_size)
    array_shape = (2 * tile_size, 2 * tile_size)
    vmapped_add_one_with_scratch = jax.vmap(
        pl.pallas_call(
            add_one_with_scratch,
            out_shape=jax.ShapeDtypeStruct(array_shape, jnp.int32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(lambda i, j: (i, j), tile_shape)],
                out_specs=pl.BlockSpec(lambda i, j: (i, j), tile_shape),
                scratch_shapes=[pltpu.VMEM(tile_shape, dtype=jnp.int32)],
                grid=(2, 2),
            ),
        )
    )

    x = jnp.broadcast_to(jnp.arange(array_shape[0]), (10, *array_shape))

    out = vmapped_add_one_with_scratch(x)
    out_ref = x + 1

    np.testing.assert_array_equal(out, out_ref, strict=True)


class PallasCallControlFlowTest(PallasTPUTest):

  def setUp(self):
    super().setUp()
    if jtu.device_under_test() != 'tpu':
      self.skipTest('Test only works on TPU')

  def test_nested_conds(self):
    def kernel(y_ref):
      def select(pred, x, y, nesting=0):
        def _true():
          if nesting == 0:
            return x + 1
          return select(x == nesting, x, y, nesting=nesting - 1)

        def _false():
          if nesting == 0:
            return y + 1
          return select(y == nesting, x, y, nesting=nesting - 1)

        return jax.lax.cond(pred, _true, _false)

      j = pl.program_id(0)
      j = select(j == 0, j, j, nesting=4)
      y_ref[...] = j * jnp.ones_like(y_ref)

    pl.pallas_call(
        kernel,
        grid=(1,),
        out_specs=pl.BlockSpec(lambda i: (0, 0), (8, 128)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
    )()
    return


class PallasCallPipelineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest('Only >=2 devices are supported.')
    if not jtu.is_device_tpu_at_least(5):
      self.skipTest('Only works with TPU v5')

  @parameterized.named_parameters(
      ('vmem', pltpu.TPUMemorySpace.VMEM),
      ('hbm', pltpu.TPUMemorySpace.ANY),
  )
  def test_pipeline_matmul(self, memory_space):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, (512, 512))
    y = jax.random.uniform(k2, (512, 512))

    def matmul_pipeline(x_ref, y_ref, z_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        z_ref[...] = jnp.zeros(z_ref.shape, jnp.float32)

      z_ref[...] += x_ref[...] @ y_ref[...]

    def matmul_kernel(x_ref, y_ref, z_ref):
      pltpu.emit_pipeline(
          matmul_pipeline,
          grid=(4, 4, 4),
          in_specs=[
              pl.BlockSpec(lambda i, j, k: (i, k), (128, 128)),
              pl.BlockSpec(lambda i, j, k: (k, j), (128, 128)),
          ],
          out_specs=pl.BlockSpec(lambda i, j, k: (i, j), (128, 128)),
      )(x_ref, y_ref, z_ref)

    z = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((512, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=memory_space),
            pl.BlockSpec(memory_space=memory_space),
        ],
        out_specs=pl.BlockSpec(memory_space=memory_space),
    )

    jax.block_until_ready(z(x, y))
    jax.block_until_ready(jnp.dot(x, y))

    out = jax.block_until_ready(z(x, y))
    expected_out = jax.block_until_ready(jnp.dot(x, y))

    np.testing.assert_allclose(out, expected_out)

  @parameterized.named_parameters(
      ('vmem', pltpu.TPUMemorySpace.VMEM),
      ('hbm', pltpu.TPUMemorySpace.ANY),
  )
  def test_double_pipeline_matmul(self, memory_space):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, (512, 512))
    y = jax.random.uniform(k2, (512, 512))

    def matmul_pipeline(x_ref, y_ref, z_ref):
      @pl.when(pl.program_id(2) == 0)
      def _():
        z_ref[...] = jnp.zeros(z_ref.shape, jnp.float32)

      z_ref[...] += x_ref[...] @ y_ref[...]

    def matmul_kernel(x_ref, y_ref, z_ref):

      def emit_pipeline(should_accumulate_out):
        pltpu.emit_pipeline(
            matmul_pipeline,
            grid=(4, 4, 4),
            in_specs=[
                pl.BlockSpec(lambda i, j, k: (i, k), (128, 128)),
                pl.BlockSpec(lambda i, j, k: (k, j), (128, 128)),
            ],
            out_specs=pl.BlockSpec(lambda i, j, k: (i, j), (128, 128)),
            should_accumulate_out=should_accumulate_out,
        )(x_ref, y_ref, z_ref)

      emit_pipeline(False)
      emit_pipeline(True)

    z = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((512, 512), jnp.float32),
        in_specs=[
            pl.BlockSpec(memory_space=memory_space),
            pl.BlockSpec(memory_space=memory_space),
        ],
        out_specs=pl.BlockSpec(memory_space=memory_space),
    )(x, y)

    np.testing.assert_allclose(z, jnp.dot(x, y) + jnp.dot(x, y))

  @parameterized.named_parameters(
      ('vmem', pltpu.TPUMemorySpace.VMEM, jnp.bfloat16),
      ('hbm', pltpu.TPUMemorySpace.ANY, jnp.bfloat16),
      ('hbm_float32', pltpu.TPUMemorySpace.ANY, jnp.float32),
  )
  def test_pipeline_all_gather_matmul(self, memory_space, out_dtype):
    num_devices = jax.device_count()
    if num_devices < 2:
      self.skipTest('Only >=2 devices are supported.')
    steps = num_devices // 2

    tm = 1024
    tk = 768
    tn = 2048

    m = 1024
    k = 6144
    n = 6144 * 8

    sharded_k = k // num_devices
    sharded_n = n // num_devices

    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, (m, k), dtype=jnp.bfloat16, minval=-1, maxval=1)
    y = jax.random.uniform(
        k2, (k, sharded_n), dtype=jnp.bfloat16, minval=-1, maxval=1
    )

    def existing_matmul_kernel(
        lhs_ref, rhs_ref, out_ref, acc_scratch_ref, *, acc_steps
    ):
      @pl.when(pl.program_id(2) == 0)
      def _zero_acc():
        acc_scratch_ref[...] = jnp.zeros(
            acc_scratch_ref.shape, acc_scratch_ref.dtype
        )

      acc_scratch_ref[...] += jnp.dot(
          lhs_ref[...],
          rhs_ref[...],
          preferred_element_type=acc_scratch_ref.dtype,
      )

      @pl.when(pl.program_id(2) == acc_steps - 1)
      def _store_acc():
        out_ref[...] = acc_scratch_ref[...].astype(out_ref.dtype)

    grid_k = sharded_k // tk
    pipeline, make_pipeline_allocations = pltpu.emit_pipeline_with_allocations(
        partial(existing_matmul_kernel, acc_steps=grid_k),
        grid=(sharded_n // tn, m // tm, grid_k),
        in_specs=[
            pl.BlockSpec(lambda n, m, k: (m, k), (tm, tk)),
            pl.BlockSpec(lambda n, m, k: (k, n), (tk, tn)),
        ],
        out_specs=pl.BlockSpec(lambda n, m, k: (m, n), (tm, tn)),
        should_accumulate_out=True,
    )

    # Given shapes:
    # lhs: A 2d, jnp.ndarray with shape [m, k // lax.psum(1,
    #   collective_axes.axes)].
    # rhs: A wd, jnp.ndarray with shape [k, n].

    # We start with a prologue that gets us the lhs chunk that our left neighbor
    # will send backward for us to send forward. After that at every step we do
    # compute on our local chunks while overlapping the backward and forward
    # collective permutes of lhs. We add to the same accumulator at every step.
    # Effectively, this permute + compute pattern achieves an all-gather of lhs
    # that is overlapped with the matmul.

    # We wait for the permutes in the pipeline epilogues so we can fuse the
    # inner compute pipeline across matmul steps and avoid bubbles.
    def all_gather_lhs_matmul_kernel(
        lhs_ref,  # [m, sharded_k]
        rhs_ref,  # [k, n]
        out_ref,  # [m, n]
        # Fwd/bwd, and double buffered.
        lhs_scratch_ref,  # [2, 2, m, sharded_k]
        acc_scratch_ref,  # [tm, tn]
        bwd_recv_sem,
        bwd_send_sem,
        fwd_recv_sem,
        fwd_send_sem,
        pipeline_allocations,
    ):
      step = pl.program_id(0)
      fwd_bwd = pl.program_id(1)
      is_first_step = step == 0
      is_not_last_step = step != steps - 1
      is_start_of_step = fwd_bwd == 0
      is_end_of_step = jnp.logical_not(is_start_of_step)
      is_start = jnp.logical_and(is_first_step, is_start_of_step)
      is_end = jnp.logical_and(step == steps - 1, is_end_of_step)
      compute_buffer = lax.rem(step, 2)
      send_buffer = 1 - compute_buffer
      my_id = lax.axis_index('x')
      right_neighbor = lax.rem(my_id + 1, num_devices)
      left_neighbor = lax.rem(my_id - 1, num_devices)
      left_neighbor = jnp.where(
          left_neighbor < 0, left_neighbor + num_devices, left_neighbor
      )

      prologue_fwd_copy = pltpu.make_async_remote_copy(
          lhs_ref,
          lhs_scratch_ref.at[1, compute_buffer],
          fwd_send_sem,
          fwd_recv_sem,
          device_id=right_neighbor,
      )

      @pl.when(is_start)
      @pltpu.trace('sync_and_bwd_prologue')
      def _sync_and_bwd_prologue():
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor)
        pltpu.semaphore_signal(barrier_sem, device_id=right_neighbor)
        pltpu.semaphore_wait(barrier_sem, 2)
        prologue_bwd_copy = pltpu.make_async_copy(
            lhs_ref,
            lhs_scratch_ref.at[0, compute_buffer],
            bwd_send_sem,
        )
        prologue_bwd_copy.start()
        prologue_fwd_copy.start()
        prologue_bwd_copy.wait()

      bwd_kwargs, fwd_kwargs = [
          {
              'src_ref': scratch_ref.at[compute_buffer],
              'dst_ref': scratch_ref.at[send_buffer],
              'send_sem': send_sem,
              'recv_sem': recv_sem,
              'device_id': device_id,
          }
          for scratch_ref, send_sem, recv_sem, device_id in [
              (
                  lhs_scratch_ref.at[0],
                  bwd_send_sem,
                  bwd_recv_sem,
                  left_neighbor,
              ),
              (
                  lhs_scratch_ref.at[1],
                  fwd_send_sem,
                  fwd_recv_sem,
                  right_neighbor,
              ),
          ]
      ]

      @pl.when(jnp.logical_and(is_not_last_step, is_start_of_step))
      @pltpu.trace('send_next_dma')
      def _send_next_dma():
        pltpu.make_async_remote_copy(**bwd_kwargs).start()

        @pl.when(jnp.logical_not(is_start))
        def _send_next_fwd_dma():
          pltpu.make_async_remote_copy(**fwd_kwargs).start()

      def get_rhs_slice(step, is_start_of_step=is_start_of_step):
        bwd_rhs_offset = lax.rem(my_id + step, num_devices)
        fwd_rhs_offset = lax.rem(my_id - step - 1, num_devices)
        fwd_rhs_offset = jnp.where(
            fwd_rhs_offset < 0, fwd_rhs_offset + num_devices, fwd_rhs_offset
        )
        offset = jnp.where(is_start_of_step, bwd_rhs_offset, fwd_rhs_offset)
        return pl.ds(
            pl.multiple_of(offset * sharded_k, sharded_k),
            sharded_k,
        )

      with pltpu.trace('dots'):

        def epilogue(epilogue_args: pltpu.PipelineCallbackArgs):

          @pl.when(is_start)
          @pltpu.trace('fwd_prologue')
          def _fwd_prologue():
            prologue_fwd_copy.wait()
            pltpu.make_async_remote_copy(**fwd_kwargs).start()

          @pl.when(jnp.logical_and(is_not_last_step, is_end_of_step))
          @pltpu.trace('wait_on_prev_dma')
          def _wait_on_prev_dma():
            pltpu.make_async_remote_copy(**bwd_kwargs).wait()
            pltpu.make_async_remote_copy(**fwd_kwargs).wait()

          def prefetch_pipeline_inputs():
            prefetch_compute_buffer = jnp.where(
                is_start_of_step, compute_buffer, send_buffer
            )
            prefetch_fwd_bwd = lax.rem(fwd_bwd + 1, 2)
            prefetch_pipeline_refs = epilogue_args.make_pipeline_refs(
                lhs_scratch_ref.at[prefetch_fwd_bwd, prefetch_compute_buffer],
                rhs_ref.at[
                    get_rhs_slice(
                        jnp.where(is_start_of_step, step, step + 1),
                        jnp.logical_not(is_start_of_step),
                    )
                ],
                out_ref,
            )
            return epilogue_args.start_pipeline_prefetch(
                pltpu.PipelinePrefetchArgs(
                    prefetch_pipeline_refs,
                    epilogue_args.pipeline_allocations,
                    epilogue_args.pipeline_buffers,
                ),
                # Force copy lhs because we just permuted it.
                # Force copy rhs because we need a different slice.
                force_copy=([True, True], False),
            )

          return lax.cond(
              jnp.logical_not(is_end),
              prefetch_pipeline_inputs,
              lambda: (
                  epilogue_args.pipeline_buffers.input,
                  epilogue_args.pipeline_buffers.in_out,
              ),
          )

        pipeline(
            lhs_scratch_ref.at[fwd_bwd, compute_buffer],
            rhs_ref.at[get_rhs_slice(step)],
            out_ref,
            scratchs=[acc_scratch_ref],
            allocations=pipeline_allocations,
            init_allocations=is_start,
            prologue=lambda _: (
                # Input and accum prologue input copy start skip conditions.
                (
                    jnp.logical_not(is_start),
                    jnp.logical_not(is_start),
                ),
                # Force input and accum input copy wait.
                ([True, True], False),
            ),
            epilogue=epilogue,
            # Only skip prologue output copy wait if starting and there is no
            # previous output.
            out_prologue=lambda _: is_start,
            # Skip epilogue output copy wait unless it's the end.
            out_epilogue=lambda _: jnp.logical_not(is_end),
        )

    kernel = pl.pallas_call(
        all_gather_lhs_matmul_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((m, sharded_n), out_dtype),
            jax.ShapeDtypeStruct((2, 2, m, sharded_k), x.dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=memory_space),
                pl.BlockSpec(memory_space=memory_space),
            ],
            out_specs=[pl.BlockSpec(memory_space=memory_space)] * 2,
            grid=(steps, 2),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)]
            + [pltpu.SemaphoreType.DMA] * 4
            + [
                make_pipeline_allocations(
                    memory_space((), x.dtype),
                    memory_space((), y.dtype),
                    memory_space((), out_dtype),
                )
            ],
        ),
        mosaic_params=dict(
            collective_id=0, vmem_limit_bytes=int(134217728 * 0.9)
        ),
    )

    shard = partial(
        shard_map.shard_map,
        mesh=jax.sharding.Mesh(
            mesh_utils.create_device_mesh((num_devices,), jax.devices()),
            ['x'],
        ),
        in_specs=(P(None, 'x'), P(None, None)),
        out_specs=P(None, None),
        check_rep=False,
    )

    test = jax.jit(shard(kernel))

    @jax.jit
    @shard
    def reference(x, y):
      x = jax.lax.all_gather(x, 'x', axis=1, tiled=True)
      return jnp.dot(x, y, preferred_element_type=out_dtype)

    jax.block_until_ready(test(x, y))
    jax.block_until_ready(reference(x, y))

    out = jax.block_until_ready(test(x, y)[0])
    expected_out = jax.block_until_ready(reference(x, y))

    np.testing.assert_allclose(
        out.astype(jnp.float32),
        expected_out.astype(jnp.float32),
        atol=1 if out_dtype == jnp.float32 else 5,
    )


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
