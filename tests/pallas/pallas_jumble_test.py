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

import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
import jax
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.interpreters import batching
from jax._src.pallas.pallas_call import _trace_kernel_to_jaxpr
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


# TODO(mvoz): Update signatures of pallas_call to correct inputs/outputs.
# pylint: disable=no-value-for-parameter

config.parse_flags_with_absl()


intx = dtypes.canonicalize_dtype(jnp.int64)
floatx = dtypes.canonicalize_dtype(jnp.float64)


@jtu.with_config(jax_traceback_filtering="off")
class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jtu.test_device_matches(["cpu"]) and not self.INTERPRET:
      self.skipTest("On CPU the test works only in interpret mode")
    if jtu.test_device_matches(
        ["cuda"]
    ) and not jtu.is_cuda_compute_capability_at_least("8.0"):
      self.skipTest("Only works on GPU with capability >= sm80")
    if sys.platform == "win32" and not self.INTERPRET:
      self.skipTest("Only works on non-Windows platforms")

    super().setUp()
    _trace_kernel_to_jaxpr.cache_clear()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


@jtu.with_config(jax_dynamic_shapes=True, jax_numpy_dtype_promotion="standard")
class PallasCallRaggedVmapTest(PallasBaseTest):

  def test_vmap_jumble_over_sin_kernel(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Only tested on TPU")

    row_count = 8
    col_grid_size = 5
    ragged_shape = [3, 1, 4]
    sizes = lax.convert_element_type(
        jnp.array([128 * x for x in ragged_shape]),
        core.bint(col_grid_size * 128),
    )
    x = jax.vmap(
        lambda n: jnp.ones((row_count, n)), out_axes=batching.jumble_axis
    )(sizes)

    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.sin(x_ref[...])

    def invoke_kernel(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec((8, 128), lambda j, k: (j, k))],
          out_specs=pl.BlockSpec((8, 128), lambda j, k: (j, k)),
          out_shape=jax.ShapeDtypeStruct(
              (8, col_grid_size * 128), dtype=jnp.float32
          ),
          grid=(1, col_grid_size),
          interpret=self.INTERPRET,
          # See note - on zero filling counterfactuals
          debug=True,
      )(x)

    res = jax.vmap(
        invoke_kernel,
        out_axes=batching.jumble_axis,
        in_axes=batching.jumble_axis,
        axis_size=3,
    )(x)

    res = res.data
    total = len(ragged_shape) * row_count * col_grid_size * 128
    res_total = np.prod(res.shape)
    self.assertEqual(res_total, total)
    ragged_total = 0
    for dim in ragged_shape:
      ragged_total += row_count * dim * 128

    def correct(v):
      return np.count_nonzero(v == jnp.sin(1.0))

    for b, batch in enumerate(res):
      ragged_val = ragged_shape[b]
      for r, row in enumerate(batch):
        row_total = ragged_val * 128
        self.assertEqual(correct(row), row_total, msg=f"row {r}, : {row}")

    self.assertEqual(correct(res), ragged_total)

  def test_vmap_jumble_over_add_kernel(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Only tested on TPU")

    row_count = 8
    col_grid_size = 5
    ragged_shape = [3, 1, 4]
    sizes = lax.convert_element_type(
        jnp.array([128 * x for x in ragged_shape]),
        core.bint(col_grid_size * 128),
    )
    x = jax.vmap(
        lambda n: jnp.ones((row_count, n)), out_axes=batching.jumble_axis
    )(sizes)
    y = jax.vmap(
        lambda n: jnp.ones((row_count, n)), out_axes=batching.jumble_axis
    )(sizes)

    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[...]

    def invoke_kernel(x, y):
      return pl.pallas_call(
          kernel,
          in_specs=[
              pl.BlockSpec((8, 128), lambda j, k: (j, k)),
              pl.BlockSpec((8, 128), lambda j, k: (j, k)),
          ],
          out_specs=pl.BlockSpec((8, 128), lambda j, k: (j, k)),
          out_shape=jax.ShapeDtypeStruct(
              (8, col_grid_size * 128), dtype=jnp.float32
          ),
          grid=(1, col_grid_size),
          interpret=False,
      )(x, y)

    res = jax.vmap(
        invoke_kernel,
        out_axes=batching.jumble_axis,
        in_axes=batching.jumble_axis,
        axis_size=3,
    )(x, y)

    res = res.data
    total = len(ragged_shape) * row_count * col_grid_size * 128
    res_total = np.prod(res.shape)
    self.assertEqual(res_total, total)
    ragged_total = 0
    for dim in ragged_shape:
      ragged_total += row_count * dim * 128

    def correct(v):
      return np.count_nonzero(v == 2.0)

    for r, row in enumerate(res):
      ragged_val = ragged_shape[r]
      row_total = ragged_val * 128 * row_count
      self.assertEqual(correct(row), row_total)
      for col in row:
        col_total = ragged_val * 128
        self.assertEqual(correct(col), col_total)

    self.assertEqual(np.count_nonzero(res == 2.0), ragged_total)

  def test_vmap_jumble_over_sin_kernel_grid_remapping(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Only tested on TPU")

    row_count = 8
    col_grid_size = 5
    ragged_shape = [3, 1, 4]
    sizes = lax.convert_element_type(
        jnp.array([128 * x for x in ragged_shape]),
        core.bint(col_grid_size * 128),
    )
    x = jax.vmap(
        lambda n: jnp.ones((row_count, n)), out_axes=batching.jumble_axis
    )(sizes)

    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.sin(x_ref[...]) * pl.program_id(2)

    def invoke_kernel(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec((8, 128), lambda j, k: (j, k))],
          out_specs=pl.BlockSpec((8, 128), lambda j, k: (j, k)),
          out_shape=jax.ShapeDtypeStruct((8, 640), dtype=jnp.float32),
          grid=(1, 5),
          interpret=False,
      )(x)

    with self.assertRaisesRegex(ValueError, "Axis 2 is out of bounds for grid"):
      jax.vmap(
          invoke_kernel,
          out_axes=batching.jumble_axis,
          in_axes=batching.jumble_axis,
          axis_size=3,
      )(x)

  def test_vmap_jumble_over_matmul_kernel(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Only tested on TPU")

    m = 128
    k = 640
    n = 640

    def matmul_kernel(x_ref, y_ref, x_sentinel, z_ref):
      # weird little once-only reset
      @pl.when(x_sentinel[...][0][0] == 1.0)
      def _():
        z_ref[...] = jnp.zeros_like(z_ref)
        x_sentinel[...] = jnp.zeros_like(x_sentinel)

      z_ref[...] += x_ref[...] @ y_ref[...]

    def matmul(
        x: jax.Array,
        y: jax.Array,
        x_sentinel: jax.Array,
        *,
        bm: int = 128,
        bk: int = 128,
        bn: int = 640,
    ):
      # m, k = x.shape
      # _, n = y.shape
      # a (1, 5) grid
      # TODO(mvoz): parameterize this grid?
      grid = (n // bn, k // bk)
      return pl.pallas_call(
          matmul_kernel,
          out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
          in_specs=[
              pl.BlockSpec(
                  (bm, bk),
                  lambda j, k: (0, k),
              ),
              pl.BlockSpec(
                  (bk, bn),
                  lambda j, k: (k, j),
              ),
              pl.BlockSpec(
                  (bm, bn),
                  lambda j, k: (0, j),
              ),
          ],
          out_specs=pl.BlockSpec(
              (bm, bn),
              lambda j, k: (0, j),
          ),
          grid=grid,
          input_output_aliases={2: 0},
          interpret=False,
      )(x, y, x_sentinel)

    # TODO(mvoz): parameterize this shape?
    ragged_shape = [3, 1, 4]
    sizes = lax.convert_element_type(
        jnp.array([128 * x for x in ragged_shape]),
        core.bint(k),
    )
    x = jax.vmap(lambda k_: jnp.ones((m, k_)), out_axes=batching.jumble_axis)(
        sizes
    )
    x_sentinel = jax.vmap(
        lambda k_: jnp.ones((m, k_)), out_axes=batching.jumble_axis
    )(sizes)
    y = jax.vmap(lambda k_: jnp.ones((k_, n)), out_axes=batching.jumble_axis)(
        sizes
    )

    res = jax.vmap(
        matmul,
        out_axes=batching.jumble_axis,
        in_axes=batching.jumble_axis,
        axis_size=3,
    )(x, y, x_sentinel)

    ref = jax.vmap(
        jnp.dot,
        out_axes=batching.jumble_axis,
        in_axes=batching.jumble_axis,
        axis_size=3,
    )(x, y)

    ref = ref.data
    res = res.data
    np.testing.assert_allclose(ref, res)

  def test_vmap_jumble_ragged_boundary_unaligned_with_grid(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Only tested on TPU")

    self.skipTest("Checkify NYI")

    row_count = 8
    col_grid_size = 5
    ragged_shape = [3, 1, 4]
    sizes = lax.convert_element_type(
        jnp.array([(128 * x) - 1 for x in ragged_shape]),
        core.bint(col_grid_size * 128),
    )
    x = jax.vmap(
        lambda n: jnp.ones((row_count, n)), out_axes=batching.jumble_axis
    )(sizes)

    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.sin(x_ref[...])

    def invoke_kernel(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec((8, 128), lambda j, k: (j, k))],
          out_specs=pl.BlockSpec((8, 128), lambda j, k: (j, k)),
          out_shape=jax.ShapeDtypeStruct((8, 640), dtype=jnp.float32),
          grid=(1, 5),
          interpret=False,
      )(x)

    with self.assertRaisesRegex(
        ValueError,
        "Ragged input shape must be evenly divisble by the grid"  # noqa: W605
        " size at the ragged dimension 2",
    ):
      jax.vmap(
          invoke_kernel,
          out_axes=batching.jumble_axis,
          in_axes=batching.jumble_axis,
          axis_size=3,
      )(x)


class PallasCallNamedGridInterpretTest(PallasCallRaggedVmapTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main()
