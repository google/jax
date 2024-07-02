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

"""Tests for Pallas indexing logic and abstractions."""

from __future__ import annotations

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src import util
from jax._src.state import indexing
import numpy as np
import jax.numpy as jnp
from jax.experimental import pallas as pl

try:
  import hypothesis as hp
except (ModuleNotFoundError, ImportError):
  raise unittest.SkipTest("tests depend on hypothesis library")

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hps
hp.settings.register_profile(
    "deterministic", database=None, derandomize=True, deadline=None,
    max_examples=100, print_blob=True)
hp.settings.load_profile("deterministic")

jax.config.parse_flags_with_absl()

Slice = indexing.Slice
NDIndexer = indexing.NDIndexer
ds = indexing.ds


def int_indexer_strategy(dim) -> hps.SearchStrategy[int]:
  return hps.integers(min_value=np.iinfo(np.int32).min, max_value=dim - 1)


@hps.composite
def slice_indexer_strategy(draw, dim) -> Slice | slice:
  start = draw(int_indexer_strategy(dim))
  max_size = dim - start
  size = draw(hps.integers(min_value=0, max_value=max_size))
  return draw(
      hps.one_of(
          hps.just(Slice(start, size)), hps.just(slice(start, start + size))
      )
  )


@hps.composite
def array_indexer_strategy(draw, shape) -> jax.Array:
  unbcast = [draw(hps.booleans()) for _ in shape]
  shape = tuple(1 if unb else s for unb, s in zip(unbcast, shape))
  return draw(hnp.arrays(dtype=np.dtype("int32"), shape=shape))


@hps.composite
def indexer_strategy(draw, dim, int_indexer_shape
                     ) -> int | Slice | jax.Array:
  return draw(hps.one_of(
      int_indexer_strategy(dim),
      slice_indexer_strategy(dim),
      array_indexer_strategy(int_indexer_shape),
  ))


@hps.composite
def nd_indexer_strategy(draw, shape) -> NDIndexer:
  num_indices = draw(hps.integers(min_value=0, max_value=len(shape)))
  int_indexer_shape = draw(hnp.array_shapes())
  indices = tuple(draw(indexer_strategy(dim, int_indexer_shape))
                  for dim in shape[:num_indices])
  return NDIndexer.from_indices_shape(indices, shape)


class IndexerTest(jtu.JaxTestCase):

  def test_simple_ndindexer(self):
    indices = (0, 0)
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), ())

  def test_invalid_ndindexer(self):
    indices = (0, 0, 0)
    shape = (5, 5)
    with self.assertRaisesRegex(
        ValueError, "`indices` must not be longer than `shape`"
    ):
      _ = NDIndexer.from_indices_shape(indices, shape)

  @parameterized.parameters(
      ((4, 0), (3, 5)),
      ((slice(3, 2), 0), (3, 5)),
      ((Slice(2, 2), 0), (3, 5)),
  )
  def test_invalid_ndindexer_oob(self, indices, shape):
    with self.assertRaisesRegex(ValueError, "Out of bound"):
      _ = NDIndexer.from_indices_shape(indices, shape)

  def test_ndindexer_with_padding(self):
    indices = ()
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), shape)

  def test_ndindexer_with_ellipsis(self):
    indices = (..., 4)
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5,))

  def test_ndindexer_with_slices(self):
    indices = (slice(2, 3), slice(4, 7))
    shape = (5, 6)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (1, 2))

  def test_ndindexer_with_arrays(self):
    indices = (np.arange(10), np.arange(10))
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (10,))

    indices = (np.ones((10, 20)), np.ones((10, 20)))
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (10, 20))

  def test_ndindexer_with_arrays_and_broadcasting(self):
    indices = (np.arange(10)[None], np.arange(20)[:, None])
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (20, 10))

    indices = (np.arange(10)[:, None], np.arange(20)[None, :])
    shape = (5, 5)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (10, 20))

  def test_ndindexer_with_arrays_and_invalid_broadcasting(self):
    indices = (np.arange(10)[None], np.arange(20)[None, :])
    shape = (5, 5)
    with self.assertRaisesRegex(
        ValueError, "Cannot broadcast shapes for indexing"
    ):
      indexer = NDIndexer.from_indices_shape(indices, shape)

  def test_indexer_with_all_types(self):
    indices = (0, slice(10), np.arange(5))
    shape = (2, 3, 4)
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5, 3))

    indices = (0, slice(2, 10), np.arange(5))
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5, 1))

    indices = (0, 1, np.arange(5))
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5,))

    indices = (ds(0, 2), np.arange(5)[:, None], np.arange(4)[None])
    indexer = NDIndexer.from_indices_shape(indices, shape)
    self.assertTupleEqual(indexer.get_indexer_shape(), (5, 4, 2))

  @hp.given(hps.data())
  def test_ndindexer(self, data):
    shape = data.draw(hnp.array_shapes())
    indexer = data.draw(nd_indexer_strategy(shape))
    is_int_indexer = [not isinstance(idx, Slice) for idx in indexer.indices]
    rest_indexers, int_indexers = util.partition_list(
        is_int_indexer, indexer.indices
    )
    if int_indexers:
      expected_int_indexer_shape = int_indexers[0].shape
    else:
      expected_int_indexer_shape = ()
    self.assertTupleEqual(
        indexer.int_indexer_shape, expected_int_indexer_shape
    )
    for idx in rest_indexers:
      self.assertIsInstance(idx, (np.ndarray, Slice))
      if isinstance(idx, np.ndarray):
        self.assertTupleEqual(idx.shape, ())
        self.assertEqual(idx.dtype, np.dtype("int32"))
    rest_shape = tuple(
        r.size for r in rest_indexers if not isinstance(r, np.ndarray)
    )
    self.assertTupleEqual((*indexer.int_indexer_shape, *rest_shape),
                          indexer.get_indexer_shape())


  def test_multi_indexing_interpreter_only(self):
    # Interpreter only test! YMMV actually compiling this.
    def permute(left, right, left_out_ref, right_out_ref):
      left_out = jnp.zeros_like(left)
      left_out = left_out.at[:, 0].set(left[:, 0])
      left_out = left_out.at[:, 1].set(right[:, 0])
      left_out = left_out.at[:, 2:].set(left[:, 1:-1])

      right_out = jnp.zeros_like(right)
      right_out = right_out.at[:, :-1].set(right[:, 1:])
      right_out = right_out.at[:, -1].set(left[:, -1])

      left_out_ref[...] = left_out
      right_out_ref[...] = right_out

    def invoke_permutes(x_ref, y_ref, x_out_ref, y_out_ref):
      shape = x_ref.shape
      _, n = shape[-2], shape[-1]
      x_ref = x_ref.at[: n // 2, : n // 2]
      y_ref = y_ref.at[: n // 2, : n // 2]
      x_out_ref = x_out_ref.at[: n // 2, : n // 2]
      y_out_ref = y_out_ref.at[: n // 2, : n // 2]
      permute(x_ref, y_ref, x_out_ref, y_out_ref)

    n = 8
    x = jnp.ones([n, n])
    y = jnp.ones([n, n])
    jitted_permute = jax.jit(invoke_permutes)
    grid = (1,)
    pl.pallas_call(
        jitted_permute,
        grid=grid,
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(x.shape, y.dtype),
        ],
        in_specs=[
            pl.BlockSpec(x.shape, lambda i: (0, 0)),
            pl.BlockSpec(y.shape, lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec(x.shape, lambda i: (0, 0)),
            pl.BlockSpec(y.shape, lambda i: (0, 0)),
        ],
        interpret=True,
    )(x, y)

  def test_ellipsis_indexing_iterpret_only(self):
    # Interpreter only test! YMMV actually compiling this.
    def permute_columns_in_row_kernel(left, right, new_left, new_right):
      shape = left.shape
      k = shape[-1]
      ndim = len(shape)
      left_slices = [
          left[..., :1],
          right[..., :1],
          left[..., 1:k-1]
      ]
      right_slices = [
          right[..., 1:k],
          left[..., k-1:k]
      ]
      new_left[...] = np.concatenate(left_slices, axis=ndim - 1)
      new_right[...] = np.concatenate(right_slices, axis=ndim - 1)

    left = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    right = jnp.array([[7, 8, 9], [10, 11, 12]], dtype=jnp.float32)

    output_shape = left.shape

    # hack to reuse the same fn for np cat
    import jax.numpy as np  # noqa: F811
    left_out, right_out = pl.pallas_call(
        permute_columns_in_row_kernel,
        grid=(1,),
        out_shape=[
            jax.ShapeDtypeStruct(output_shape, jnp.float32),
            jax.ShapeDtypeStruct(output_shape, jnp.float32)
        ],
        in_specs=[
            pl.BlockSpec(left.shape, lambda i: (0, 0)),
            pl.BlockSpec(right.shape, lambda i: (0, 0))
        ],
        out_specs=[
            pl.BlockSpec(output_shape, lambda i: (0, 0)),
            pl.BlockSpec(output_shape, lambda i: (0, 0))
        ],
        interpret=True,
    )(left, right)


    import numpy as np  # noqa: F811
    left_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    right_np = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
    left_out_np = left_np.copy()
    right_out_np = right_np.copy()


    permute_columns_in_row_kernel(left_np, right_np, left_out_np, right_out_np)
    np.testing.assert_array_equal(left_out_np, left_out)
    np.testing.assert_array_equal(right_out_np, right_out)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
