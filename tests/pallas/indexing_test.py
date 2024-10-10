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
import sys
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax._src import test_util as jtu
from jax._src import util
from jax._src.state import indexing
import numpy as np
import jax.numpy as jnp
from jax.experimental import pallas as pl

if sys.platform != "win32":
  from jax.experimental.pallas import tpu as pltpu
else:
  pltpu = None

try:
  import hypothesis as hp
except (ModuleNotFoundError, ImportError):
  raise unittest.SkipTest("tests depend on hypothesis library")

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hps


jax.config.parse_flags_with_absl()
jtu.setup_hypothesis(max_examples=100)


Slice = indexing.Slice
NDIndexer = indexing.NDIndexer
ds = indexing.ds


_INDEXING_TEST_CASES = [
    ((4, 8, 128), (...,), (4, 8, 128)),
    ((4, 8, 128), (0,), (8, 128)),
    ((4, 8, 128), (pl.ds(1, 2),), (2, 8, 128)),
    ((4, 8, 128), (pl.ds(2, 2),), (2, 8, 128)),
    ((4, 8, 128), (pl.ds(2, 2), 0), (8, 128)),
    ((4, 8, 128), (pl.ds(2, 2), 1), (8, 128)),
    ((4, 8, 128), (slice(2, 4), 1), (8, 128)),
    ((4, 8, 128), (slice(2, 4), slice(0, 1), 0), (8, 128)),
    ((4, 8, 128), ((0, pl.ds(0, 8), pl.ds(0, 128)), ...), (8, 128)),
    ((4, 8, 128), (..., (0, pl.ds(0, 8), pl.ds(0, 128)), ...), (8, 128)),
]


def _maybe_ds_to_slice(x: int | slice | indexing.Slice) -> int | slice:
  if isinstance(x, indexing.Slice):
    return slice(x.start, x.start + x.size)
  return x


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


class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if not self.INTERPRET:
      if not jtu.test_device_matches(["tpu"]):
        self.skipTest("Only interpret mode supported on non-TPU")

    super().setUp()

  @classmethod
  def pallas_call(cls, *args, **kwargs):
    return pl.pallas_call(*args, interpret=cls.INTERPRET, **kwargs)


class IndexerTest(jtu.JaxTestCase):
  """These are unit tests for the indexer logic, not using pallas_call."""

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
    self.assertTupleEqual(indexer.get_indexer_shape(), (2, 5, 4))

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


class IndexerOpsTest(PallasBaseTest):

  def test_multi_indexing_interpreter_only(self):
    if not self.INTERPRET:
      self.skipTest("Only supported in interpret mode")
    # Interpret only test! YMMV actually compiling this.
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

  def test_multi_indexing_destination_ref(self):
    if not self.INTERPRET:
      self.skipTest("Only supported in interpret mode")
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.zeros_like(o_ref)
      new_o_ref = o_ref.at[pl.ds(0, 8)].at[0].at[pl.ds(0, 4), pl.ds(0, 4)]
      new_o_ref[...] = x_ref[...]

    x = jax.random.normal(jax.random.key(0), shape=(4, 4))
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 16, 16), x.dtype),
        interpret=True,
    )(x)
    expected = jnp.zeros((16, 16, 16)).at[0, 0:4, 0:4].set(x)
    np.testing.assert_array_equal(result, expected)

  def test_ellipsis_indexing_iterpret_only(self):
    if not self.INTERPRET:
      self.skipTest("Only supported in interpret mode")
    # Interpret only test! YMMV actually compiling this.
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

  @hp.given(hps.data())
  def test_vmap_nd_indexing(self, data):
    self.skipTest("TODO(necula): enable this test; was in jax_triton.")
    vmap_shape = data.draw(hnp.array_shapes(min_dims=1, max_dims=3, min_side=2),
                           label="vmap_shape")
    el_shape = data.draw(hnp.array_shapes(min_dims=2), label="el_shape")
    # TODO(sharadmv,apaszke): enable rank 0 and rank 1 Refs
    # hp.assume(len(el_shape) >= 2)
    nd_indexer = data.draw(nd_indexer_strategy(el_shape), label="nd_indexer")
    expected_shape = jax.eval_shape(lambda x: x[nd_indexer],
                                    jax.ShapeDtypeStruct(el_shape, jnp.float32))

    ref = lambda x: x[nd_indexer]
    def kernel(x_ref, y_ref):
      x = pl.load(x_ref, nd_indexer)
      pl.store(y_ref, (slice(None),) * len(y_ref.shape), x)
    func = pl.pallas_call(kernel, out_shape=expected_shape)

    shape = el_shape
    for vmap_dim in vmap_shape[::-1]:
      index = data.draw(hps.integers(min_value=0,
                                     max_value=max(0, len(shape) - 2)),
                        label="index")
      # hp.assume(index <= max(0, len(shape) - 2))
      # TODO(sharadmv,apaszke): enable vmapping over batch axes in 2 minormost
      #                         dimensions
      shape = (*shape[:index], vmap_dim, *shape[index:])
      ref = jax.vmap(ref, in_axes=index, out_axes=0)
      func = jax.vmap(func, in_axes=index, out_axes=0)
    key = random.PRNGKey(0)
    x = random.normal(key, shape, dtype=jnp.float32)
    expected = ref(x)
    y = func(x)
    np.testing.assert_array_equal(y, expected)

  @parameterized.product(
      indexer_type=["state", "pallas"],
      case=_INDEXING_TEST_CASES,
  )
  def test_can_load_with_ref_at(self, indexer_type, case):
    if self.INTERPRET:
      self.skipTest("TODO: fails in interpret mode.")
    in_shape, indexers, out_shape = case
    dtype = jnp.float32
    def body(x_ref, y_ref):
      for indexer in indexers[:-1]:
        x_ref = x_ref.at[indexer]
      if indexer_type == "state":
        x = x_ref[indexers[-1]]
        y_ref[...] = x
      elif indexer_type == "pallas":
        x = pl.load(x_ref, indexers[-1])
        pl.store(y_ref, ..., x)

    x = random.normal(random.key(0), in_shape, dtype=dtype)
    y = x
    for indexer in indexers:
      if not isinstance(indexer, tuple):
        indexer = (indexer,)
      indexer = tuple(map(_maybe_ds_to_slice, indexer))
      y = y[indexer]
    assert y.shape == out_shape
    out = self.pallas_call(body, out_shape=y)(x)
    self.assertAllClose(out, y)

  @parameterized.product(
      indexer_type=["state", "pallas"],
      case=_INDEXING_TEST_CASES,
  )
  def test_can_store_with_ref_at(self, indexer_type, case):
    if self.INTERPRET:
      self.skipTest("TODO: fails in interpret mode.")
    in_shape, indexers, val_shape = case
    dtype = jnp.float32
    def body(x_ref, y_ref):
      y_ref[...] = jnp.zeros_like(y_ref)
      for indexer in indexers[:-1]:
        y_ref = y_ref.at[indexer]
      if indexer_type == "state":
        x = x_ref[...]
        y_ref[indexers[-1]] = x
      elif indexer_type == "pallas":
        x = pl.load(x_ref, ...)
        pl.store(y_ref, indexers[-1], x)

    val = random.normal(random.key(0), val_shape, dtype=dtype)
    # Use NumPy arrays to do nested indexing and mutation. This is really
    # annoying to do in vanilla JAX.
    x = np.zeros(in_shape, dtype=dtype)
    y = x
    for indexer in indexers:
      if not isinstance(indexer, tuple):
        indexer = (indexer,)
      indexer = tuple(map(_maybe_ds_to_slice, indexer))
      y = y[indexer]
    assert y.shape == val_shape
    y[...] = val
    out = self.pallas_call(body, out_shape=x)(val)
    self.assertAllClose(out, x)

  @parameterized.product(
      indexer_type=["state", "pallas"],
      slice_type=["slice", "ds"],
  )
  @hp.given(
      ref_shape=hps.sampled_from(((8, 8, 32), (7, 7, 33))),
      indices=hps.tuples(
          hps.integers(0, 6), hps.integers(0, 6), hps.integers(0, 31)
      ),
      strides=hps.tuples(
          hps.integers(1, 10), hps.integers(1, 10), hps.integers(1, 10)
      ),
  )
  def test_strided_load_and_store(
      self, indexer_type, slice_type, ref_shape, indices, strides
  ):
    if self.INTERPRET:
      self.skipTest("TODO: fails in interpret mode.")
    ref_shape = (*ref_shape, 128)
    indices = (*indices, 0)
    strides = (*strides, 1)
    vec_shape = [
        (l - i + s - 1) // s for l, i, s in zip(ref_shape, indices, strides)
    ]
    dtype = jnp.float32

    def body(x_ref, y_ref1, y_ref2):
      if slice_type == "slice":
        slices = tuple(
            slice(i, rs, s) for i, rs, s in zip(indices, ref_shape, strides)
        )
      else:
        slices = tuple(
            pl.ds(i, vs, s) for i, vs, s in zip(indices, vec_shape, strides)
        )
      if indexer_type == "state":
        y_ref1[...] = x_ref[slices]
        y_ref2[slices] = y_ref1[...]
      elif indexer_type == "pallas":
        pl.store(y_ref1, ..., pl.load(x_ref, slices))
        pl.store(y_ref2, slices, pl.load(y_ref1, ...))

    x = random.normal(random.key(0), ref_shape, dtype=dtype)
    y1, y2 = self.pallas_call(
        body,
        out_shape=[
            jax.ShapeDtypeStruct(vec_shape, dtype),
            jax.ShapeDtypeStruct(ref_shape, dtype),
        ],
    )(x)
    slices = tuple(
        slice(i, l, s) for l, i, s in zip(ref_shape, indices, strides)
    )
    expected = x[slices]
    self.assertAllClose(y1, expected, err_msg="Strided Load Error")
    self.assertAllClose(
        y2[slices], expected, err_msg="Strided Store Error"
    )

  def test_load_with_dynamic_2nd_minor_index(self):
    if pltpu is None:
      self.skipTest("No TPU module available.")
    # We can take any dynamic index on the 2nd minor dimension as long as
    # the minormost dimsize is vreg lane count.
    m, n = 32, 128
    k = 10
    start = 2

    def kernel(x_ref, indices, y_ref):
      y_ref[...] = pl.load(x_ref, pl.ds(indices[0], k))

    x = jnp.arange(m * n, dtype=jnp.int32).reshape((m, n))
    indices = jnp.array([start])

    res = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((k, n), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
        ),
    )(x, indices)
    self.assertAllClose(res, x[start : start + k, :], atol=0., rtol=0.)

  def test_store_with_dynamic_2nd_minor_index(self):
    if pltpu is None:
      self.skipTest("No TPU module available.")
    # We can take any dynamic index on the 2nd minor dimension as long as
    # the minormost dimsize is vreg lane count.
    m, n = 10, 128
    k = 32
    start = 2

    def kernel(x_ref, indices, y_ref):
      pl.store(y_ref, pl.ds(indices[0], m), x_ref[...])

    x = jnp.arange(m * n, dtype=jnp.int32).reshape((m, n))
    indices = jnp.array([start])

    res = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((k, n), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
        ),
    )(x, indices)
    self.assertAllClose(res[start : start + m, :], x, atol=0., rtol=0.)

  def test_load_one_row_with_dynamic_2nd_minor_index(self):
    if pltpu is None:
      self.skipTest("No TPU module available.")
    # This test triggers strided load. We can take any dynamic index on the
    # 2nd minor dimension as long as we load one row on the 2nd minor dim.
    b, m, n = 4, 16, 256
    start = 3

    def kernel(x_ref, indices, y_ref):
      y_ref[...] = x_ref[:, pl.ds(indices[0], 1), :]

    x = jnp.arange(b * m * n, dtype=jnp.int32).reshape((b, m, n))
    indices = jnp.array([start])

    res = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((b, 1, n), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
        ),
    )(x, indices)
    self.assertAllClose(res, x[:, start : start + 1, :], atol=0., rtol=0.)

  def test_store_one_row_with_dynamic_2nd_minor_index(self):
    if pltpu is None:
      self.skipTest("No TPU module available.")
    # This test triggers strided store. We can take any dynamic index on the
    # 2nd minor dimension as long as we store one row on the 2nd minor dim.
    b, m, n = 4, 16, 256
    start = 3

    def kernel(x_ref, indices, y_ref):
      y_ref[:, pl.ds(indices[0], 1), :] = x_ref[...]

    x = jnp.arange(b * 1 * n, dtype=jnp.int32).reshape((b, 1, n))
    indices = jnp.array([start])

    res = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((b, m, n), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
        ),
    )(x, indices)
    self.assertAllClose(res[:, start : start + 1, :], x, atol=0., rtol=0.)


class IndexerOpsInterpretTest(IndexerOpsTest):
  INTERPRET = True


# TODO(ayx): Fix all test cases here
_ADVANCED_INDEXER_TEST_CASES = [
    # integer
    ((3, 2), lambda arr, a, b, c, d: arr[2]),
    # slice
    ((12, 12), lambda arr, a, b, c, d: arr[::4, ::4]),
    ((16, 16), lambda arr, a, b, c, d: arr[1:14:2, 2:13:4]),
    ((8, 2), lambda arr, a, b, c, d: arr[1::3, :]),
    # array
    ((4, 3), lambda arr, a, b, c, d: arr[a]),
    ((4, 3, 2), lambda arr, a, b, c, d: arr[c, c]),
    # integer + 1-D array
    ((4, 3), lambda arr, a, b, c, d: arr[2, a]),
    ((4, 3), lambda arr, a, b, c, d: arr[a, 2]),
    # slice + 1-D array
    ((4, 3), lambda arr, a, b, c, d: arr[a, :]),
    # ((4, 3), lambda arr, a, b, c, d: arr[:, a]),
    ((6, 8, 3), lambda arr, a, b, c, d: arr[c, ::3]),
    # ((8, 6, 3), lambda arr, a, b, c, d: arr[::3, c]),
    # ((8, 8, 3), lambda arr, a, b, c, d: arr[::4, ::2, a]),
    # ((8, 8, 3), lambda arr, a, b, c, d: arr[::4, a, ::2]),
    ((8, 8, 3, 7), lambda arr, a, b, c, d: arr[b, ::4, a, ::2]),
    ((3, 8, 8, 7), lambda arr, a, b, c, d: arr[b, a, ::4, ::2]),
    # ((8, 8, 3, 7), lambda arr, a, b, c, d: arr[::4, b, a, ::2]),
    ((16, 3, 6, 2), lambda arr, a, b, c, d: arr[::4, a, 1::2, b]),
    ((8, 8, 3, 6), lambda arr, a, b, c, d: arr[b, ::4, a, a]),
    # slice + array w/ broadcasting
    ((8, 8, 3, 6), lambda arr, a, b, c, d: \
        arr[b[:, None], ::4, a[None], a[:, None]]),
    # integer + slice + 1-D array
    ((5, 8, 8, 3), lambda arr, a, b, c, d: arr[2, ::4, ::2, a]),
    ((5, 8, 8, 3), lambda arr, a, b, c, d: arr[2, ::4, a, ::2]),
    # boolean
    # ((6, 2), lambda arr, a, b, c, d: arr[d]),
    # ((8, 6), lambda arr, a, b, c, d: arr[::4, d]),
]


class AdvancedIndexerOpsTest(PallasBaseTest):

  def setUp(self):
    if jtu.test_device_matches(["tpu"]):
      self.skipTest("Advanced indexers are not supported on TPU")

    # 4 arrays that are used in test cases of advanced indexing
    self.a = jnp.array([1, 1, 1, 1, 1], dtype=jnp.int32)
    self.b = jnp.array([1, 2, 2, 2, 2], dtype=jnp.int32)
    self.c = jnp.array([1, 0, 2, 2, -1, 1], dtype=jnp.int32)
    self.d = jnp.array([1, 0, 0, 0, 0, 1], dtype=jnp.bool_)

    super().setUp()

  @parameterized.parameters(_ADVANCED_INDEXER_TEST_CASES)
  def test_advanced_indexer(self, in_shape: tuple[int, ...], indexing_func):
    a, b, c, d = self.a, self.b, self.c, self.d

    x = jnp.arange(np.prod(in_shape), dtype=jnp.float32).reshape(in_shape)
    y = indexing_func(x, a, b, c, d)

    # `a_ref`, `b_ref`, `c_ref` and `d_ref` are for testing purposes.
    # We have them here because we need to have a unified function signature
    # for all test cases, even if the arrays are actually not used in any
    # computation.
    def kernel(x_ref, a_ref, b_ref, c_ref, d_ref, o_ref):
      a = a_ref[...]
      b = b_ref[...]
      c = c_ref[...]
      d = d_ref[...]
      o = indexing_func(x_ref, a, b, c, d)
      o_ref[...] = o

    y_ = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(y.shape, jnp.float32),
    )(x, a, b, c, d)

    np.testing.assert_array_equal(y_, y)


class AdvancedIndexerOpsInterpretTest(AdvancedIndexerOpsTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
