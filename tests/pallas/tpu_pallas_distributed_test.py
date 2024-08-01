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

"""Tests for distributed pallas TPU operations."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

partial = functools.partial


class PallasCallRemoteDMATest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest('Only >=2 devices are supported.')
    if not jtu.is_device_tpu(5, 'e'):
      self.skipTest('Only works with TPU v5e.')

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

      pl.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

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

      pl.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

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

      pl.run_scoped(
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

      pl.run_scoped(
          body,
          pltpu.SemaphoreType.REGULAR,
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      )

    num_devices = jax.local_device_count()
    x = jnp.arange(num_devices * 8 * 128).reshape((num_devices * 8, 128))

    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
          out_shape=x,
          compiler_params=dict(mosaic=dict(collective_id=0)),
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


class PallasCallRemoteDMAInterpretTest(parameterized.TestCase):

  @parameterized.parameters(('left',), ('right',))
  def test_interpret_remote_dma_ppermute(self, permutation):
    if jax.device_count() <= 1:
      self.skipTest('Test requires multiple devices.')
    num_devices = jax.device_count()
    if permutation == 'left':
      permute_fn = lambda x: lax.rem(x + num_devices - 1, num_devices)
    else:
      permute_fn = lambda x: lax.rem(x + num_devices + 1, num_devices)

    # Construct a kernel which performs a ppermute based on permute_fn.
    def test_kernel(x_ref,
                    o_ref,
                    copy_send_sem,
                    copy_recv_sem,
                ):
      o_ref[...] = jnp.zeros_like(o_ref[...])
      my_id = lax.axis_index('x')
      dst_device = permute_fn(my_id)
      input_to_output_copy = pltpu.make_async_remote_copy(
          src_ref=x_ref,
          dst_ref=o_ref,
          send_sem=copy_send_sem,
          recv_sem=copy_recv_sem,
          device_id=dst_device,
          device_id_type=pltpu.DeviceIdType.LOGICAL,
      )
      input_to_output_copy.start()
      input_to_output_copy.wait()

    out_shape = (jax.ShapeDtypeStruct((8, 128), jnp.float32))
    grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
            ],
            scratch_shapes=(
                [pltpu.SemaphoreType.DMA] * 2
            )
        )

    devices = mesh_utils.create_device_mesh((1, num_devices))
    mesh = jax.sharding.Mesh(devices, P(None, 'x'))
    sharding = jax.sharding.NamedSharding(mesh, P(None, 'x'))
    unsharded_arr = jax.random.normal(
        jax.random.key(0), shape=(8, 128 * num_devices))
    sharded_arr = jax.device_put(unsharded_arr, sharding)

    kernel = pl.pallas_call(
        test_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        interpret=True,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_rep=False))
    result = compiled_func(sharded_arr)

    perm = tuple((src, permute_fn(src)) for src in range(num_devices))
    perm = jax.tree_util.tree_map(int, perm)
    def lax_permute(x):
      return lax.ppermute(x, 'x', perm)
    expected = jax.jit(shard_map.shard_map(lax_permute,
                                   mesh=mesh,
                                   in_specs=P(None, 'x'),
                                   out_specs=P(None, 'x')))(sharded_arr)
    np.testing.assert_array_equal(result, expected)

  def test_interpret_remote_dma_asymmetrical_indexer(self):
    # Test DMAs where destination slices are not the same.
    if jax.local_device_count() <= 1:
      self.skipTest('Test requires multiple devices.')
    if not jtu.is_device_tpu(5, 'e'):
      self.skipTest('Only works with TPU v5e.')
    num_devices = jax.local_device_count()

    def test_kernel(x_ref,
               output_ref,
               send_sem,
               recv_sem):
      output_ref[...] = jnp.zeros_like(output_ref[...])
      my_id = lax.axis_index('x')
      even_device = lax.rem(my_id, 2)
      odd_device = 1 - even_device
      neighbor = lax.rem(my_id + 1, num_devices)
      # If the device_id is even, we copy to output_ref[1].
      # If it's odd, we copy to output_ref[0].
      @pl.when(even_device)
      def _():
        remote_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=output_ref.at[1],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=neighbor,
        )
        remote_dma.start()
        remote_dma.wait()
      @pl.when(odd_device)
      def _():
        remote_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=output_ref.at[0],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=neighbor,
        )
        remote_dma.start()
        remote_dma.wait()

    out_shape = (jax.ShapeDtypeStruct((2, 8, 128), jnp.float32))
    grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
            scratch_shapes=(
                [pltpu.SemaphoreType.DMA] * 2
            )
        )

    devices = mesh_utils.create_device_mesh(( num_devices,))
    mesh = jax.sharding.Mesh(devices, P('x'))
    sharding = jax.sharding.NamedSharding(mesh, P(None, 'x'))
    unsharded_arr = jax.random.normal(
        jax.random.key(0), shape=(8, 128 * num_devices))
    sharded_arr = jax.device_put(unsharded_arr, sharding)

    # Compare interpret mode result to non-interpret mode result.
    kernel = pl.pallas_call(
        test_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        interpret=True,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_rep=False))
    result_interpret = compiled_func(sharded_arr)

    kernel = pl.pallas_call(
        test_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_rep=False))
    result_noninterpret = compiled_func(sharded_arr)
    np.testing.assert_allclose(result_interpret,
                               result_noninterpret,
                               atol=1e-5,
                               rtol=1e-3)

  def test_interpret_remote_dma_asymmetrical_refs(self):
    # Test DMAs where dst refs are not the same.
    self.skipTest('Known failure.')
    num_devices = jax.local_device_count()

    def test_kernel(x_ref,
               even_output,
               odd_output,
               send_sem,
               recv_sem):
      even_output[...] = jnp.zeros_like(even_output[...])
      odd_output[...] = jnp.zeros_like(odd_output[...])
      my_id = lax.axis_index('x')
      even_device = lax.rem(my_id, 2)
      odd_device = 1 - even_device
      neighbor = lax.rem(my_id + 1, num_devices)
      @pl.when(even_device)
      def _():
        remote_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=even_output,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=neighbor,
            device_id_type=pltpu.DeviceIdType.LOGICAL,
        )
        remote_dma.start()
        remote_dma.wait()
      @pl.when(odd_device)
      def _():
        remote_dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=odd_output,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=neighbor,
            device_id_type=pltpu.DeviceIdType.LOGICAL,
        )
        remote_dma.start()
        remote_dma.wait()

    out_shape = (jax.ShapeDtypeStruct((8, 128), jnp.float32))
    grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.VMEM),
            ],
            scratch_shapes=(
                [pltpu.SemaphoreType.DMA] * 2
            )
        )

    devices = mesh_utils.create_device_mesh((1, num_devices))
    mesh = jax.sharding.Mesh(devices, P(None, 'x'))
    sharding = jax.sharding.NamedSharding(mesh, P(None, 'x'))
    unsharded_arr = jax.random.normal(
        jax.random.key(0), shape=(8, 128 * num_devices))
    sharded_arr = jax.device_put(unsharded_arr, sharding)

    # Compare interpret mode result to non-interpret mode result.
    kernel = pl.pallas_call(
        test_kernel,
        out_shape=(out_shape, out_shape),
        grid_spec=grid_spec,
        interpret=True,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_rep=False))
    result_interpret = compiled_func(sharded_arr)

    kernel = pl.pallas_call(
        test_kernel,
        out_shape=(out_shape, out_shape),
        grid_spec=grid_spec,
    )
    compiled_func = jax.jit(shard_map.shard_map(
      kernel,
      mesh=mesh,
      in_specs=P(None, 'x'),
      out_specs=P(None, 'x'),
      check_rep=False))
    result_noninterpret = compiled_func(sharded_arr)
    np.testing.assert_allclose(result_interpret,
                               result_noninterpret,
                               atol=1e-5,
                               rtol=1e-3)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
