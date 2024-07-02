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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import blocked_sampler
from jax._src import config
from jax._src import test_util as jtu
import numpy as np


config.parse_flags_with_absl()


def call_kernel(
  kernel,
  grid: tuple[int, int],
  transpose_grid: bool,
  *args
  ):
  """Calls a kernel over a grid and concatenates results to a single array."""
  if transpose_grid:
    grid = (grid[1], grid[0])
  m, n = grid
  return jnp.concatenate([
      jnp.concatenate([
          kernel(i, j, *args) for j in range(n)], axis=1)
      for i in range(m)], axis=0)


def uniform_kernel(i: int, j: int, total_size, block_size, tile_size):
  """Uniform random sampling kernel function."""
  global_key = jax.random.key(0)
  keys = blocked_sampler.blocked_fold_in(global_key,
                                         total_size=total_size,
                                         block_size=block_size,
                                         tile_size=tile_size,
                                         block_index=(i, j))
  return blocked_sampler.sample_block(jax.random.uniform,
                                      keys,
                                      block_size=block_size,
                                      tile_size=tile_size,
                                      minval=0.0, maxval=1.0)


class BlockedSamplerTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    dict(testcase_name='8x128_vs_16x256', total_size=(32, 256),
         block_size_a=(8, 128), block_size_b=(16, 256),
         tile_size=(8, 128), transpose_grid=False),
    dict(testcase_name='transpose_8x128_vs_16x256', total_size=(32, 256),
         block_size_a=(8, 128), block_size_b=(16, 256),
         tile_size=(8, 128), transpose_grid=True),
    dict(testcase_name='8x128_vs_32x128', total_size=(32, 128),
         block_size_a=(8, 128), block_size_b=(32, 128),
         tile_size=(8, 128), transpose_grid=False),
    dict(testcase_name='16x256_vs_32x128', total_size=(32, 256),
         block_size_a=(16, 256), block_size_b=(32, 128),
         tile_size=(8, 128), transpose_grid=False),
  )
  def test_block_shape_invariance(self, total_size, block_size_a,
                          block_size_b, tile_size, transpose_grid):
    grid_a = tuple(_tot // _blk for _tot, _blk in zip(total_size, block_size_a))
    result_a = call_kernel(
        uniform_kernel, grid_a, transpose_grid,
        total_size, block_size_a, tile_size)

    grid_b = tuple(_tot // _blk for _tot, _blk in zip(total_size, block_size_b))
    result_b = call_kernel(
        uniform_kernel, grid_b, transpose_grid,
        total_size, block_size_b, tile_size)
    np.testing.assert_array_equal(result_a, result_b)


if __name__ == "__main__":
  absltest.main()
