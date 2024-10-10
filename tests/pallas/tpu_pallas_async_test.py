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

"""Test TPU-specific uses of Pallas async APIs."""

import functools
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import core as jax_core
from jax._src import test_util as jtu
from jax._src.state import discharge as state_discharge
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()
P = jax.sharding.PartitionSpec
partial = functools.partial

Future = Any


def make_async_copy(target_memory_space=None):
  if target_memory_space is None:
    target_memory_space = pltpu.ANY
  @jax.named_call
  def copy_start(x: jax.Array) -> tuple[jax.Array, Future]:

    def copy_start_kernel(x_ref, aliased_x_ref, o_ref, sem):
      del aliased_x_ref
      pltpu.make_async_copy(x_ref, o_ref, sem).start()

    x, out, sem = pl.pallas_call(
        copy_start_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),  # aliased x
            target_memory_space(x.shape, x.dtype),  # out
            pltpu.SemaphoreType.DMA(()),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=target_memory_space),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ),
        input_output_aliases={0: 0},
    )(x)
    return x, (out, sem)

  @jax.named_call
  def copy_done(x: jax.Array, future: Future) -> jax.Array:
    out, sem = future

    def copy_done_kernel(x_ref, o_ref, sem, aliased_o_ref):
      del aliased_o_ref
      pltpu.make_async_copy(x_ref, o_ref, sem).wait()

    out = pl.pallas_call(
        copy_done_kernel,
        out_shape=target_memory_space(x.shape, x.dtype),  # out
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=target_memory_space),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=target_memory_space),
        input_output_aliases={1: 0},
    )(x, out, sem)
    return out

  return copy_start, copy_done


def make_async_slice(index: int):

  def async_slice_start_kernel(x_ref, aliased_x_ref, o_ref, sem):
    del aliased_x_ref
    pltpu.make_async_copy(x_ref.at[index], o_ref, sem).start()

  def async_slice_done_kernel(x_ref, o_ref, sem, aliased_o_ref):
    del aliased_o_ref
    pltpu.make_async_copy(x_ref.at[index], o_ref, sem).wait()

  @jax.named_call
  def async_slice_start(x: jax.Array) -> tuple[jax.Array, Future]:

    x, out, sem = pl.pallas_call(
        async_slice_start_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),  # aliased x
            jax.ShapeDtypeStruct(x.shape[1:], x.dtype),  # out
            pltpu.SemaphoreType.DMA(()),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ),
        input_output_aliases={0: 0},
    )(x)
    return x, (out, sem)

  @jax.named_call
  def async_slice_done(
      x: jax.Array, future: Future
  ) -> tuple[jax.Array, Future]:
    out, sem = future
    out = pl.pallas_call(
        async_slice_done_kernel,
        out_shape=(jax.ShapeDtypeStruct(x.shape[1:], x.dtype)),  # out
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=(pl.BlockSpec(memory_space=pltpu.ANY)),
        input_output_aliases={1: 0},
    )(x, out, sem)
    return out

  return async_slice_start, async_slice_done


def make_async_dynamic_slice(index: jax.Array):

  def async_dslice_start_kernel(index_ref, x_ref, aliased_x_ref, o_ref, sem):
    del aliased_x_ref
    pltpu.make_async_copy(x_ref.at[index_ref[0]], o_ref, sem).start()

  def async_dslice_done_kernel(x_ref, o_ref, sem, aliased_o_ref):
    del aliased_o_ref
    pltpu.make_async_copy(x_ref.at[0], o_ref, sem).wait()

  @jax.named_call
  def async_dslice_start(x: jax.Array) -> tuple[jax.Array, Future]:

    x, out, sem = pl.pallas_call(
        async_dslice_start_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),  # aliased x
            jax.ShapeDtypeStruct(x.shape[1:], x.dtype),  # out
            pltpu.SemaphoreType.DMA(()),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.ANY),
            ],
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.ANY),
                pl.BlockSpec(memory_space=pltpu.ANY),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            ),
        ),
        input_output_aliases={1: 0},
    )(index[None], x)
    return x, (out, sem)

  @jax.named_call
  def async_dslice_done(
      x: jax.Array, future: Future
  ) -> tuple[jax.Array, Future]:
    out, sem = future
    out = pl.pallas_call(
        async_dslice_done_kernel,
        out_shape=(jax.ShapeDtypeStruct(x.shape[1:], x.dtype)),  # out
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=(pl.BlockSpec(memory_space=pltpu.ANY)),
        input_output_aliases={1: 0},
    )(x, out, sem)
    return out

  return async_dslice_start, async_dslice_done


class PallasCallAsyncCopyTest(parameterized.TestCase):
  # TODO(b/368123537): add more tests

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('DMAs only guaranteed to work ou TPU v4+')

  def test_basic_async_copy(self):
    @jax.jit
    def f(x):
      copy_start, copy_done = make_async_copy()
      x, fut = copy_start(x)
      y = copy_done(x, fut)
      return y

    x = jax.random.normal(jax.random.key(0), (8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_multiple_async_copy(self):
    @jax.jit
    def f(x):
      copy_start, copy_done = make_async_copy()
      x, fut = copy_start(x)
      x2, fut2 = copy_start(x)
      y = copy_done(x, fut)
      y2 = copy_done(x2, fut2)
      return y, y2

    x = jax.random.normal(jax.random.key(0), (8, 128), dtype=jnp.float32)
    y, y2 = f(x)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(y2, x)

  def test_async_slice(self):
    @jax.jit
    def f(x):
      async_slice_start, async_slice_done = make_async_slice(2)
      x, fut = async_slice_start(x)
      y = async_slice_done(x, fut)
      return y

    x = jax.random.normal(jax.random.key(0), (4, 8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x[2])

  def test_async_dynamic_slice(self):
    @jax.jit
    def f(x, i):
      async_slice_start, async_slice_done = make_async_dynamic_slice(i)
      x, fut = async_slice_start(x)
      y = async_slice_done(x, fut)
      return y

    x = jax.random.normal(jax.random.key(0), (4, 8, 128), dtype=jnp.float32)
    y = f(x, 2)
    np.testing.assert_array_equal(y, x[2])

  def test_multi_async_dynamic_slice(self):
    @jax.jit
    def f(x, i, j):
      async_slice_start, async_slice_done = make_async_dynamic_slice(i)
      async_slice_start2, async_slice_done2 = make_async_dynamic_slice(j)
      x, fut = async_slice_start(x)
      x2, fut2 = async_slice_start2(x)
      y = async_slice_done(x, fut)
      y2 = async_slice_done2(x2, fut2)
      return y, y2

    x = jax.random.normal(jax.random.key(0), (4, 8, 128), dtype=jnp.float32)
    y, y2 = f(x, 2, 3)
    np.testing.assert_array_equal(y, x[2])
    np.testing.assert_array_equal(y2, x[3])

  def test_basic_async_copy_into_vmem(self):
    @jax.jit
    def f(x):
      copy_start, copy_done = make_async_copy(pltpu.VMEM)
      x, fut = copy_start(x)
      y = copy_done(x, fut)
      return y

    x = jax.random.normal(jax.random.key(0), (8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_multiple_async_copy_into_vmem(self):
    @jax.jit
    def f(x):
      copy_start, copy_done = make_async_copy(pltpu.VMEM)
      x1, fut = copy_start(x)
      x2, fut2 = copy_start(x)
      y = copy_done(x1, fut)
      y2 = copy_done(x2, fut2)
      return y, y2

    x = jax.random.normal(jax.random.key(0), (8, 128), dtype=jnp.float32)
    y, y2 = f(x)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(y2, x)

  def test_copy_in_a_loop(self):

    @jax.jit
    def f(x):
      def body(_, carry):
        x = carry
        copy_start, copy_done = make_async_copy()
        x, fut = copy_start(x)
        y = copy_done(x, fut)
        return y
      x = jax.lax.fori_loop(0, x.shape[0], body, x)
      return x

    x = jax.random.normal(jax.random.key(0), (16, 8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_staggered_copy_in_a_loop(self):

    @jax.jit
    def f(x):
      copy_start, copy_done = make_async_copy()
      x, fut = copy_start(x)
      def body(_, carry):
        x, fut = carry
        y = copy_done(x, fut)
        y, fut = copy_start(y)
        return y, fut
      # We *must* use unroll > 2 here because of aliasing constraints. XLA will
      # introduce copies of the active buffer with unroll=1.
      y, fut = jax.lax.fori_loop(0, x.shape[0] - 1, body, (x, fut), unroll=2)
      x = copy_done(y, fut)
      return x

    x = jax.random.normal(jax.random.key(0), (16, 8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_full_copy_in_a_loop(self):

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      def body(i, carry):
        x, ys = carry
        copy_start, copy_done = make_async_dynamic_slice(i)
        x, fut = copy_start(x)
        y = copy_done(x, fut)
        ys = ys.at[i].set(y)
        return x, ys
      _, y = jax.lax.fori_loop(0, x.shape[0], body, (x, y))
      return y

    x = jax.random.normal(jax.random.key(0), (16, 8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_staggered_full_copy_in_a_loop(self):

    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      copy_start, _ = make_async_dynamic_slice(jnp.array(0))
      x, fut = copy_start(x)
      def body(i, carry):
        x, fut, ys = carry
        _, copy_done = make_async_dynamic_slice(i)
        y = copy_done(x, fut)
        copy_start, _ = make_async_dynamic_slice(i + 1)
        ys = ys.at[i].set(y)
        x, fut = copy_start(x)
        return x, fut, ys
      # We can use unroll=1 here because we have the ys.at[i].set(y) in the
      # middle
      x, fut, ys = jax.lax.fori_loop(0, x.shape[0] - 1, body, (x, fut, y),
                                     unroll=1)
      _, copy_done = make_async_dynamic_slice(x.shape[0] - 1)
      y = copy_done(x, fut)
      ys = ys.at[x.shape[0] - 1].set(y)
      return ys

    x = jax.random.normal(jax.random.key(0), (16, 8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)


def make_async_remote_copy(axis_name: str, direction: str = 'right',
                           target_memory_space=None):
  if target_memory_space is None:
    target_memory_space = pltpu.ANY
  @jax.named_call
  def copy_start(x: jax.Array) -> tuple[jax.Array, Future]:

    def copy_start_kernel(x_ref, aliased_x_ref, o_ref, send_sem, recv_sem):
      del aliased_x_ref
      axis_size = jax.lax.psum(1, axis_name)
      left_neighbor = jax.lax.rem(
          jax.lax.axis_index(axis_name) - 1 + axis_size, axis_size
      )
      right_neighbor = jax.lax.rem(
          jax.lax.axis_index(axis_name) + 1, axis_size
      )
      if direction == 'right':
        src_neighbor = left_neighbor
        dst_neighbor = right_neighbor
      else:
        src_neighbor = right_neighbor
        dst_neighbor = left_neighbor
      barrier_sem = pltpu.get_barrier_semaphore()
      pltpu.semaphore_signal(barrier_sem, device_id=src_neighbor, core_index=0)
      pltpu.semaphore_wait(barrier_sem, 1)
      pltpu.make_async_remote_copy(
          x_ref, o_ref, send_sem, recv_sem, device_id=dst_neighbor,
      ).start()

    x, out, send_sem, recv_sem = pl.pallas_call(
        copy_start_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),  # aliased x
            target_memory_space(x.shape, x.dtype),  # out
            pltpu.SemaphoreType.DMA(()),  # send_sem
            pltpu.SemaphoreType.DMA(()),  # recv_sem
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=target_memory_space),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.TPUCompilerParams(collective_id=0),
    )(x)
    return x, (out, send_sem, recv_sem)

  @jax.named_call
  def send_done(x: jax.Array, future: Future) -> jax.Array:
    _, send_sem, _ = future

    def send_done_kernel(x_ref, send_sem, aliased_o_ref):
      del aliased_o_ref
      pltpu.make_async_copy(x_ref, x_ref, send_sem).wait()

    x = pl.pallas_call(
        send_done_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),  # out
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
        input_output_aliases={0: 0},
    )(x, send_sem)
    return x

  @jax.named_call
  def recv_done(x: jax.Array, future: Future) -> jax.Array:
    out, _, recv_sem = future

    def send_done_kernel(x_ref, o_ref, send_sem, aliased_o_ref):
      del aliased_o_ref
      pltpu.make_async_copy(x_ref, o_ref, send_sem).wait()

    out = pl.pallas_call(
        send_done_kernel,
        out_shape=target_memory_space(x.shape, x.dtype),  # out
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=target_memory_space),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=target_memory_space),
        input_output_aliases={1: 0},
    )(x, out, recv_sem)
    return out

  return copy_start, send_done, recv_done


def make_bidi_collective_permute(axis_name: str):
  @jax.named_call
  def copy_start(x: jax.Array) -> tuple[jax.Array, Future]:

    def copy_start_kernel(x_ref, aliased_x_ref, o_ref, left_sems, right_sems):
      del aliased_x_ref
      axis_size = jax.lax.psum(1, axis_name)
      left_neighbor = jax.lax.rem(
          jax.lax.axis_index(axis_name) - 1 + axis_size, axis_size
      )
      right_neighbor = jax.lax.rem(
          jax.lax.axis_index(axis_name) + 1, axis_size
      )
      barrier_sem = pltpu.get_barrier_semaphore()
      pltpu.semaphore_signal(barrier_sem, device_id=left_neighbor, core_index=0)
      pltpu.semaphore_signal(
          barrier_sem, device_id=right_neighbor, core_index=0
      )
      pltpu.semaphore_wait(barrier_sem, 2)
      assert x.shape[0] % 2 == 0, x.shape
      pltpu.make_async_remote_copy(
          x_ref.at[pl.ds(0, x.shape[0] // 2)],
          o_ref.at[pl.ds(0, x.shape[0] // 2)],
          right_sems[0],
          right_sems[1],
          device_id=right_neighbor,
      ).start()
      pltpu.make_async_remote_copy(
          x_ref.at[pl.ds(x.shape[0] // 2, x.shape[0] // 2)],
          o_ref.at[pl.ds(x.shape[0] // 2, x.shape[0] // 2)],
          left_sems[0],
          left_sems[1],
          device_id=left_neighbor,
      ).start()

    x, out, left_sems, right_sems = pl.pallas_call(
        copy_start_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x.shape, x.dtype),  # aliased x
            pltpu.ANY(x.shape, x.dtype),  # out
            (pltpu.SemaphoreType.DMA(()),) * 2,  # left_sems
            (pltpu.SemaphoreType.DMA(()),) * 2,  # right_sems
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.ANY),
            (pl.BlockSpec(memory_space=pltpu.SEMAPHORE),) * 2,
            (pl.BlockSpec(memory_space=pltpu.SEMAPHORE),) * 2,
        ),
        input_output_aliases={0: 0},
        compiler_params=pltpu.TPUCompilerParams(collective_id=0),
    )(x)
    return x, (out, left_sems, right_sems)

  @jax.named_call
  def send_done(x: jax.Array, future: Future) -> jax.Array:
    _, (send_left_sem, _), (send_right_sem, _) = future

    def send_done_kernel(x_ref, send_left_sem, send_right_sem, aliased_o_ref):
      del aliased_o_ref
      pltpu.make_async_copy(
          x_ref.at[x_ref.shape[0] // 2 :],
          x_ref.at[x_ref.shape[0] // 2 :],
          send_left_sem,
      ).wait()
      pltpu.make_async_copy(
          x_ref.at[x_ref.shape[0] // 2 :],
          x_ref.at[x_ref.shape[0] // 2 :],
          send_right_sem,
      ).wait()

    x = pl.pallas_call(
        send_done_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),  # out
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
        input_output_aliases={0: 0},
    )(x, send_left_sem, send_right_sem)
    return x

  @jax.named_call
  def recv_done(x: jax.Array, future: Future) -> jax.Array:
    out, (_, recv_left_sem), (_, recv_right_sem) = future

    def recv_done_kernel(o_ref, x_ref, recv_left_sem, recv_right_sem,
                         aliased_o_ref):
      del aliased_o_ref
      pltpu.make_async_copy(
          x_ref.at[o_ref.shape[0] // 2 :],
          o_ref.at[o_ref.shape[0] // 2 :],
          recv_left_sem,
      ).wait()
      pltpu.make_async_copy(
          x_ref.at[o_ref.shape[0] // 2 :],
          o_ref.at[o_ref.shape[0] // 2 :],
          recv_right_sem,
      ).wait()

    out = pl.pallas_call(
        recv_done_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),  # out
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.ANY),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
        input_output_aliases={0: 0},
    )(out, x, recv_left_sem, recv_right_sem)
    return out
  return copy_start, send_done, recv_done


class PallasCallRemoteAsyncCopyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('DMAs only guaranteed to work ou TPU v4+')
    if jax.device_count() < 2:
      self.skipTest('Test only works with >2 devices')

  def test_basic_remote_copy(self):

    mesh = jax.make_mesh((jax.device_count(),), ('x',))

    @jax.jit
    @partial(
        shard_map.shard_map, mesh=mesh, in_specs=(P('x'),), out_specs=P('x'),
        check_rep=False,
    )
    def f(x):
      copy_start, send_done, recv_done = make_async_remote_copy('x')
      x, fut = copy_start(x)
      x = send_done(x, fut)
      y = recv_done(x, fut)
      return y

    x = jax.random.normal(
        jax.random.key(0), (jax.device_count(), 8, 128), dtype=jnp.float32
    )
    y = f(x)
    expected = jnp.roll(x, shift=1, axis=0)
    np.testing.assert_array_equal(y, expected)

  def test_multi_remote_copy(self):

    mesh = jax.make_mesh((jax.device_count(),), ('x',))

    @jax.jit
    @partial(
        shard_map.shard_map, mesh=mesh, in_specs=(P('x'),), out_specs=P('x'),
        check_rep=False,
    )
    def f(x):
      copy_start, send_done, recv_done = make_async_remote_copy(
          'x', direction='right'
      )
      copy_start2, send_done2, recv_done2 = make_async_remote_copy(
          'x', direction='left'
      )
      x, fut = copy_start(x)
      x, fut2 = copy_start2(x)
      x = send_done(x, fut)
      x = send_done2(x, fut2)
      y = recv_done(x, fut)
      y2 = recv_done2(x, fut2)
      return y, y2

    x = jax.random.normal(
        jax.random.key(0), (jax.device_count(), 8, 128), dtype=jnp.float32
    )
    y, y2 = f(x)
    y_expected = jnp.roll(x, shift=1, axis=0)
    y2_expected = jnp.roll(x, shift=-1, axis=0)
    np.testing.assert_array_equal(y, y_expected)
    np.testing.assert_array_equal(y2, y2_expected)

  def test_basic_collective_permute_loop(self):

    mesh = jax.make_mesh((jax.device_count(),), ('x',))

    @jax.jit
    @partial(
        shard_map.shard_map, mesh=mesh, in_specs=(P('x'),), out_specs=P('x'),
        check_rep=False,
    )
    def f(x):
      copy_start, send_done, recv_done = make_async_remote_copy('x')
      def body(_, x):
        x, fut = copy_start(x)
        x = send_done(x, fut)
        y = recv_done(x, fut)
        return y
      # Send all the way around except for one step
      return jax.lax.fori_loop(0, jax.device_count() - 1, body, x)
    x = jax.random.normal(
        jax.random.key(0), (jax.device_count(), 8, 128), dtype=jnp.float32
    )
    y = f(x)
    expected = jnp.roll(x, shift=-1, axis=0)
    np.testing.assert_array_equal(y, expected)

  def test_staggered_collective_permute_loop(self):

    mesh = jax.make_mesh((jax.device_count(),), ('x',))

    @jax.jit
    @partial(
        shard_map.shard_map, mesh=mesh, in_specs=(P('x'),), out_specs=P('x'),
        check_rep=False,
    )
    def f(x):
      assert x.shape[0] == 1
      copy_start, send_done, recv_done = make_async_remote_copy('x')
      x, fut = copy_start(x)
      def body(_, carry):
        x, fut = carry
        x = send_done(x, fut)
        y = recv_done(x, fut)
        y, fut = copy_start(y)
        return y, fut
      # Send all the way around except for one step
      x, fut = jax.lax.fori_loop(0, jax.device_count() - 2, body, (x, fut),
                                 unroll=2)
      x = send_done(x, fut)
      y = recv_done(x, fut)
      return y

    n_devices = jax.device_count()
    x = jax.random.normal(
        jax.random.key(0), (n_devices, 8, 128), dtype=jnp.float32
    )
    y = f(x)
    expected = jnp.roll(x, shift=-1, axis=0)
    np.testing.assert_array_equal(y, expected)

  def test_bidi_collective_permute_loop(self):
    mesh = jax.make_mesh((jax.device_count(),), ('x',))

    @jax.jit
    @partial(
        shard_map.shard_map, mesh=mesh, in_specs=(P('x'),), out_specs=P('x'),
        check_rep=False,
    )
    def f(x):
      assert x.shape[0] == 1
      x = x[0]
      copy_start, send_done, recv_done = make_bidi_collective_permute('x')
      def body(_, x):
        x, fut = copy_start(x)
        x = send_done(x, fut)
        y = recv_done(x, fut)
        return y
      # Send all the way around except for one step
      y = jax.lax.fori_loop(0, jax.device_count() - 1, body, x)
      return y[None]
    x = jax.random.normal(
        jax.random.key(0), (jax.device_count(), 16, 128), dtype=jnp.float32
    )
    y = f(x)
    expected = jnp.concatenate([
        jnp.roll(x[:, :8], axis=0, shift=-1),
        jnp.roll(x[:, 8:], axis=0, shift=1),
    ], axis=1)
    np.testing.assert_array_equal(y, expected)


def make_stateful_async_copy():
  @jax.named_call
  def copy_start(x_ref, o_ref) -> Future:

    def copy_start_kernel(sem):
      pltpu.make_async_copy(x_ref, o_ref, sem).start()
    sem = pl.pallas_call(
        copy_start_kernel,
        out_shape=pltpu.SemaphoreType.DMA(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
    )()
    return sem

  @jax.named_call
  def copy_done(x_ref, o_ref, future):
    sem = future

    def copy_done_kernel(sem):
      pltpu.make_async_copy(x_ref, o_ref, sem).wait()

    () = pl.pallas_call(
        copy_done_kernel,
        out_shape=(),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
    )(sem)

  return copy_start, copy_done


def make_stateful_async_slice(i: int):
  @jax.named_call
  def copy_start(x_ref, o_ref) -> Future:

    def copy_start_kernel(sem):
      pltpu.make_async_copy(x_ref.at[i], o_ref, sem).start()
    sem = pl.pallas_call(
        copy_start_kernel,
        out_shape=pltpu.SemaphoreType.DMA(()),
        out_specs=pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
    )()
    return sem

  @jax.named_call
  def copy_done(x_ref, o_ref, future):
    sem = future

    def copy_done_kernel(sem):
      pltpu.make_async_copy(x_ref.at[i], o_ref, sem).wait()

    () = pl.pallas_call(
        copy_done_kernel,
        out_shape=(),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
    )(sem)

  return copy_start, copy_done


class PallasCallStatefulAsyncTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('DMAs only guaranteed to work ou TPU v4+')

  def test_basic_stateful_async_copy(self):
    @jax.jit
    def f(x):
      y = jnp.zeros_like(x)
      def body(refs):
        copy_start, copy_done = make_stateful_async_copy()
        x_ref, y_ref = refs
        fut = copy_start(x_ref, y_ref)
        copy_done(x_ref, y_ref, fut)
      _, y = state_discharge.run_state(body)((x, y))
      return y
    x = jax.random.normal(jax.random.key(0), (8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_multiple_stateful_async_copy(self):
    @jax.jit
    def f(x):
      y = y2 = jnp.zeros_like(x)
      def body(refs):
        copy_start, copy_done = make_stateful_async_copy()
        x_ref, y_ref, y2_ref = refs
        fut = copy_start(x_ref, y_ref)
        fut2 = copy_start(x_ref, y2_ref)
        copy_done(x_ref, y_ref, fut)
        copy_done(x_ref, y2_ref, fut2)
      _, y, y2 = state_discharge.run_state(body)((x, y, y2))
      return y, y2
    x = jax.random.normal(jax.random.key(0), (8, 128), dtype=jnp.float32)
    y, y2 = f(x)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(y2, x)

  def test_basic_stateful_async_slice(self):
    @jax.jit
    def f(x):
      y = jnp.zeros(x.shape[1:], x.dtype)
      def body(refs):
        copy_start, copy_done = make_stateful_async_slice(2)
        x_ref, y_ref = refs
        fut = copy_start(x_ref, y_ref)
        copy_done(x_ref, y_ref, fut)
      _, y = state_discharge.run_state(body)((x, y))
      return y
    x = jax.random.normal(jax.random.key(0), (4, 8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x[2])

  def test_basic_stateful_async_slice_to_buffer_with_mutable_array(self):
    @jax.jit
    def f(x):
      x_ref = jax_core.mutable_array(x)
      y_ref = jax_core.mutable_array(jnp.ones(x.shape[1:], x.dtype))
      copy_start, copy_done = make_stateful_async_slice(2)
      fut = copy_start(x_ref, y_ref)
      copy_done(x_ref, y_ref, fut)
      return y_ref[...]
    x = jax.random.normal(jax.random.key(0), (4, 8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x[2])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
