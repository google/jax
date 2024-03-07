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

import math

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import dtypes
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental.sparse import nm

jax.config.parse_flags_with_absl()


class SpmmTest(jtu.JaxTestCase):
  def setUp(self):
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Only works on GPU")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPUs with capability >= sm80")
    super().setUp()

  # ----- Test different input shapes
  @parameterized.product(
      tile_m=(32, 128),
      tile_n=(32, 128),
      tile_k=(32, 128),
      batch=(None, 5),
      sparse_idx=(0, 1),
  )
  @jtu.run_on_devices("gpu")
  def test_shapes(self, tile_m, tile_n, tile_k, batch, sparse_idx):
    # Build keyword arguments
    kwargs = {
        "dimension_numbers": (((1,), (1,)), (tuple(), tuple())),
        "sparse_operand_idx": sparse_idx,
    }
    if batch:
      kwargs["dimension_numbers"] = (((2,), (2,)), ((0,), (0,)))

    # Build input data
    batch_dims = (batch,) if batch else tuple()
    lhs = (
        (np.arange((batch or 1) * tile_m * tile_k) % 11)
        .astype(dtypes.bfloat16)
        .reshape(batch_dims + (tile_m, tile_k))
    )
    rhs = (
        (np.arange((batch or 1) * tile_n * tile_k) % 13)
        .astype(dtypes.bfloat16)
        .reshape(batch_dims + (tile_n, tile_k))
    )

    # Build sparsity mask and metadata
    sp = [lhs, rhs][sparse_idx]
    mask = np.tile([True, False], math.prod(sp.shape) // 2).reshape(sp.shape)
    sparse = sp[mask].reshape(sp.shape[:-1] + (sp.shape[-1] // 2,))
    meta = nm.nm_pack(mask)

    # Calculate sparse and dense dots
    if sparse_idx == 0:
      dot_sparse = nm.nm_spmm(sparse, rhs, meta, **kwargs)
      dot_dense = jnp.einsum("...mk,...nk->...mn", (lhs * mask), rhs)
    else:
      dot_sparse = nm.nm_spmm(lhs, sparse, meta, **kwargs)
      dot_dense = jnp.einsum("...mk,...nk->...mn", lhs, (rhs * mask))

    # Verify the result
    jtu.check_eq(dot_sparse, dot_dense.astype(dtypes.bfloat16))

  # ----- Test different input types
  @parameterized.product(
      lhs_type=[jnp.int8, jnp.int16, jnp.float16, jnp.bfloat16],
      rhs_type=[jnp.bfloat16],
      output_type=[jnp.bfloat16, jnp.float32],
  )
  @jtu.run_on_devices("gpu")
  def test_types(self, lhs_type, rhs_type, output_type):
    tile_m, tile_n, tile_k = 64, 32, 128

    # Build input data
    lhs = (
        (np.arange(tile_m * tile_k) % 17)
        .astype(lhs_type)
        .reshape((tile_m, tile_k))
    )
    rhs = (
        (np.arange(tile_k * tile_n) % 19)
        .astype(rhs_type)
        .reshape((tile_k, tile_n))
    )

    # Build sparsity mask and metadata
    mask = np.tile([True, False], tile_m * tile_k // 2).reshape(lhs.shape)
    sparse = lhs[mask].reshape(tile_m, tile_k // 2)
    meta = nm.nm_pack(mask)

    # Calculate sparse and dense dots
    dot_sparse = nm.nm_spmm(sparse, rhs, meta, output_dtype=output_type)
    dot_dense = (lhs * mask) @ rhs

    # Verify the result
    jtu.check_close(dot_sparse, dot_dense.astype(output_type), rtol=0.01)

  # ----- Test validation
  @jtu.run_on_devices("gpu")
  def test_validate_nm_pack(self):
    with self.assertRaisesRegex(TypeError, "Mask should be bool"):
      nm.nm_pack(jnp.zeros(16, jnp.int8))
    with self.assertRaisesRegex(
        TypeError, "Inner dimension size should be divisible by 16"
    ):
      nm.nm_pack(jnp.array([False] * 8))

  @jtu.run_on_devices("gpu")
  def test_validate_nm_spmm(self):
    batch, tile_m, tile_n, tile_k = 2, 64, 32, 128
    lhs = jnp.zeros((batch, tile_m, tile_k // 2), dtype=jnp.bfloat16)
    rhs = jnp.zeros((batch, tile_k, tile_n), dtype=jnp.bfloat16)
    meta = jnp.zeros((batch, tile_m, tile_k // 16), dtype=jnp.uint16)

    if config.enable_x64.value:
      with self.assertRaisesRegex(TypeError, "Unsupported lhs input type"):
        nm.nm_spmm(jnp.zeros(lhs.shape, dtype=jnp.int64), rhs, meta)
      with self.assertRaisesRegex(TypeError, "Unsupported rhs input type"):
        nm.nm_spmm(lhs, jnp.zeros(rhs.shape, dtype=jnp.int64), meta)
      with self.assertRaisesRegex(TypeError, "Unsupported output type"):
        nm.nm_spmm(lhs, rhs, meta, output_dtype=jnp.int64)

    # Check dimension numbers
    nm_spmm_with_dnums = lambda c, b: nm.nm_spmm(
        lhs, rhs, meta, dimension_numbers=(c, b)
    )
    with self.assertRaisesRegex(
        TypeError, "Only single contracting dimension is supported"
    ):
      nm_spmm_with_dnums(((0, 2), (0, 1)), (tuple(), tuple()))
    with self.assertRaisesRegex(
        TypeError, "Incorrect dimension numbers for lhs"
    ):
      nm_spmm_with_dnums(((2,), (1,)), ((2,), (0,)))
    with self.assertRaisesRegex(
        TypeError, "Incorrect dimension numbers for rhs"
    ):
      nm_spmm_with_dnums(((2,), (1,)), ((0,), (1,)))
    with self.assertRaisesRegex(
        TypeError, "Only single non-contracting dimension is supported"
    ):
      nm_spmm_with_dnums(((2,), (1,)), (tuple(), tuple()))
    with self.assertRaisesRegex(
        TypeError, "Batch dimension sizes do not match"
    ):
      nm.nm_spmm(
          lhs,
          rhs.reshape(1, tile_k, tile_n * batch),
          meta,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      )

    # Check metadata
    nm_spmm_with_meta = lambda m: nm.nm_spmm(
        lhs, rhs, m, dimension_numbers=(((2,), (1,)), ((0,), (0,)))
    )
    with self.assertRaisesRegex(TypeError, "Metadata must be uint16"):
      nm_spmm_with_meta(jnp.zeros(meta.shape, dtype=jnp.uint8))
    with self.assertRaisesRegex(
        TypeError, "Metadata shape must match the operand shape"
    ):
      nm_spmm_with_meta(meta.reshape(1, batch * tile_m, tile_k // 16))
    with self.assertRaisesRegex(
        TypeError,
        "Metadata must be exactly 8 times less than the contracting dimension"
        " for 2:4 structured sparsity",
    ):
      nm_spmm_with_meta(jnp.repeat(meta, 2, axis=-1))
    with self.assertRaisesRegex(
        TypeError, "Contracting dimension must be the minor one"
    ):
      nm.nm_spmm(lhs, rhs, meta, dimension_numbers=(((1,), (1,)), ((0,), (0,))))
    with self.assertRaisesRegex(
        TypeError, "Contracting dimension sizes should have 2:4 ratio"
    ):
      nm.nm_spmm(
          lhs,
          jnp.repeat(rhs, 2, axis=1),
          meta,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
