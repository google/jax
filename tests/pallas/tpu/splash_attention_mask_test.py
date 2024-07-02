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

"""Tests for splash_attention_masks."""
from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info as mask_info_lib
from jax._src import test_util as jtu
import numpy as np

jax.config.parse_flags_with_absl()


def _make_lazy_causal_mask(*args, **kwargs):
  mask = mask_lib.CausalMask(*args, **kwargs)
  return mask[:, :]


def _make_causal_mask(*args, **kwargs):
  return mask_lib.make_causal_mask(*args, **kwargs)


def _make_lazy_local_attention_mask(*args, **kwargs):
  mask = mask_lib.LocalMask(*args, **kwargs)
  return mask[:, :]


def _make_local_attention_mask(*args, **kwargs):
  return mask_lib.make_local_attention_mask(*args, **kwargs)


class SplashAttentionMaskTest(jtu.JaxTestCase):

  @parameterized.parameters([_make_lazy_causal_mask, _make_causal_mask])
  def test_causal_mask(self, make_causal_mask):
    expected = np.array([[1]], dtype=np.bool_)
    actual = make_causal_mask((1, 1))

    with self.subTest("unit"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_causal_mask((4, 4))

    with self.subTest("square"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_causal_mask((4, 6))

    with self.subTest("wide_rectangle"):
      self.assertArraysEqual(actual, expected)

    actual = make_causal_mask((6, 4))
    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )

    with self.subTest("tall_rectangle"):
      self.assertArraysEqual(actual, expected)

    actual = make_causal_mask((4, 4), -1)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=np.bool_,
    )

    with self.subTest("negative_offset"):
      self.assertArraysEqual(actual, expected)

    actual = make_causal_mask((4, 4), 1)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )

    with self.subTest("positive_offset"):
      self.assertArraysEqual(actual, expected)

  @parameterized.parameters(
      [_make_lazy_local_attention_mask, _make_local_attention_mask]
  )
  def test_local_attention_mask(self, make_local_attention_mask):
    expected = np.array([[1]], dtype=np.bool_)
    actual = make_local_attention_mask((1, 1), (0, None), offset=0)
    self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (1, None), offset=0)
    with self.subTest("left_1"):
      self.assertArraysEqual(actual, expected)
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (None, 2), offset=0)
    with self.subTest("right_2"):
      self.assertArraysEqual(actual, expected)
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (1, 1), offset=0)
    with self.subTest("left_1_right_1"):
      self.assertArraysEqual(actual, expected)
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (1, 0), offset=0)
    with self.subTest("left_1_right_0"):
      self.assertArraysEqual(actual, expected)
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 4), (0, 2), offset=0)
    with self.subTest("left_0_right_2"):
      self.assertArraysEqual(actual, expected)

  @parameterized.parameters(
      [_make_lazy_local_attention_mask, _make_local_attention_mask]
  )
  def test_local_attention_mask_wide_rectangle(self, make_local_attention_mask):
    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (1, None), offset=0)
    with self.subTest("left_1"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (None, 2), offset=0)
    with self.subTest("right_2"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (1, 1), offset=0)
    with self.subTest("left_1_right_1"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (1, 0), offset=0)
    with self.subTest("left_1_right_0"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((4, 6), (0, 2), offset=0)
    with self.subTest("left_0_right_2"):
      self.assertArraysEqual(actual, expected)

  @parameterized.parameters(
      [_make_lazy_local_attention_mask, _make_local_attention_mask]
  )
  def test_local_attention_mask_tall_rectangle(self, make_local_attention_mask):
    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (1, None), offset=0)
    with self.subTest("left_1"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (None, 2), offset=0)
    with self.subTest("right_2"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (1, 1), offset=0)
    with self.subTest("left_1_right_1"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (1, 0), offset=0)
    with self.subTest("left_1_right_0"):
      self.assertArraysEqual(actual, expected)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    actual = make_local_attention_mask((6, 4), (0, 2), offset=0)
    with self.subTest("left_0_right_2"):
      self.assertArraysEqual(actual, expected)

  @parameterized.product(
      block_size=[(256, 256), (256, 128), (128, 256)],
      shape=[(1024, 1024), (1024, 2048), (2048, 1024)],
  )
  def test_lazy_causal_mask_chunking(
      self, block_size: tuple[int, int], shape: tuple[int, int]
  ):
    dense_mask = mask_lib.make_causal_mask(shape=shape)
    self._compare_masks(
        dense_mask,
        mask_lib.CausalMask(shape),
        block_size,
    )

  @parameterized.parameters([
      ((256, 256), (1024, 1024), (128, None), 0),
      ((256, 128), (1024, 1024), (128, None), 16),
      ((128, 256), (1024, 1024), (128, None), 16),
      ((256, 256), (1024, 1024), (128, 256), 0),
      ((256, 128), (1024, 1024), (128, 256), 0),
      ((128, 256), (1024, 1024), (128, 256), 16),
      ((256, 256), (1024, 1024), (None, 256), 0),
      ((256, 128), (1024, 1024), (None, 256), 32),
      ((128, 256), (1024, 1024), (None, 256), 32),
      #
      ((256, 256), (1024, 2048), (128, None), 0),
      ((256, 128), (1024, 2048), (128, None), 16),
      ((128, 256), (1024, 2048), (128, None), 16),
      ((256, 256), (1024, 2048), (128, 256), 0),
      ((256, 128), (1024, 2048), (128, 256), 0),
      ((128, 256), (1024, 2048), (128, 256), 16),
      ((256, 256), (1024, 2048), (None, 256), 0),
      ((256, 128), (1024, 2048), (None, 256), 32),
      ((128, 256), (1024, 2048), (None, 256), 32),
      #
      ((256, 256), (2048, 1024), (128, None), 0),
      ((256, 128), (2048, 1024), (128, None), 16),
      ((128, 256), (2048, 1024), (128, None), 16),
      ((256, 256), (2048, 1024), (128, 256), 0),
      ((256, 128), (2048, 1024), (128, 256), 0),
      ((128, 256), (2048, 1024), (128, 256), 16),
      ((256, 256), (2048, 1024), (None, 256), 0),
      ((256, 128), (2048, 1024), (None, 256), 32),
      ((128, 256), (2048, 1024), (None, 256), 32),
  ])
  def test_lazy_local_mask_chunking(
      self,
      block_size: tuple[int, int],
      shape: tuple[int, int],
      window_size: tuple[int | None, int | None],
      offset: int,
  ):
    dense_mask = mask_lib.make_local_attention_mask(
        shape, window_size, offset=offset
    )
    self._compare_masks(
        dense_mask,
        mask_lib.LocalMask(shape, window_size, offset),
        block_size,
    )

  def test_using_logical_operators_raises_exception(self):
    mask_1 = mask_lib.NumpyMask(
        mask_lib.make_random_mask((256, 256), 0.5, seed=1)
    )
    mask_2 = mask_lib.NumpyMask(
        mask_lib.make_random_mask((256, 256), 0.5, seed=2)
    )

    with self.subTest("logical_or"):
      with self.assertRaises(NotImplementedError):
        res = mask_1 or mask_2
        del res

    with self.subTest("logical_and"):
      with self.assertRaises(NotImplementedError):
        res = mask_1 and mask_2
        del res

  @parameterized.parameters([((256, 256),), ((512, 256),), ((512, 256),)])
  def test_lazy_mask_or(self, shape: tuple[int, int]):
    mask_1 = mask_lib.make_random_mask(shape, 0.5, seed=1)
    mask_2 = mask_lib.make_random_mask(shape, 0.5, seed=2)

    lazy_or = mask_lib.NumpyMask(mask_1) | mask_lib.NumpyMask(mask_2)
    dense = np.logical_or(mask_1, mask_2)

    self._compare_masks(dense, lazy_or, (256, 256))

  @parameterized.parameters([((256, 256),), ((512, 256),), ((512, 256),)])
  def test_lazy_mask_and(self, shape: tuple[int, int]):
    mask_1 = mask_lib.make_random_mask(shape, 0.5, seed=1)
    mask_2 = mask_lib.make_random_mask(shape, 0.5, seed=2)

    lazy_and = mask_lib.NumpyMask(mask_1) & mask_lib.NumpyMask(mask_2)
    dense = np.logical_and(mask_1, mask_2)

    self._compare_masks(dense, lazy_and, (256, 256))

  @parameterized.parameters([((256, 256),), ((512, 256),), ((512, 256),)])
  def test_lazy_multi_head_mask(self, shape: tuple[int, int]):
    mask_1 = mask_lib.make_random_mask(shape, 0.5, seed=1)
    mask_2 = mask_lib.make_random_mask(shape, 0.5, seed=2)

    lazy_multi_head = mask_lib.MultiHeadMask(
        (mask_lib.NumpyMask(mask_1), mask_lib.NumpyMask(mask_2))
    )
    dense = np.stack((mask_1, mask_2), axis=0)

    self._compare_masks(dense, lazy_multi_head, (256, 256))

  @parameterized.parameters([((256, 256),), ((512, 256),), ((512, 256),)])
  def test_lazy_full_mask(self, shape: tuple[int, int]):
    lazy_full = mask_lib.FullMask(shape)
    dense = np.ones(shape, dtype=np.bool_)

    self._compare_masks(dense, lazy_full, (256, 256))

  def _compare_masks(
      self,
      dense_mask: np.ndarray,
      lazy_mask: mask_lib.Mask,
      block_size: tuple[int, int],
  ):
    self.assertEqual(dense_mask.shape, lazy_mask.shape)

    *prefix, width, height = dense_mask.shape

    assert width % block_size[0] == 0
    assert height % block_size[1] == 0

    full_lazy_mask = lazy_mask[
        (*[slice(p) for p in prefix], slice(None), slice(None))
    ]
    self.assertArraysEqual(dense_mask, full_lazy_mask)
    for i, j in np.ndindex(width // block_size[0], height // block_size[1]):
      indexer = (
          *[slice(p) for p in prefix],
          slice(i * block_size[0], (i + 1) * block_size[0]),
          slice(j * block_size[1], (j + 1) * block_size[1]),
      )
      dense_chunk = dense_mask[indexer]
      lazy_chunk = lazy_mask[indexer]
      self.assertArraysEqual(dense_chunk, lazy_chunk)


class SplashAttentionMaskInfoTest(jtu.JaxTestCase):
  """Check the construction of MaskInfo from Mask."""

  def _assert_mask_info_match(
      self,
      actual: mask_info_lib.MaskInfo,
      expected: mask_info_lib.MaskInfo,
  ):

    def assert_array_is_positive(array: np.ndarray | None):
      if array is None:
        return

      is_positive = np.all(array >= 0)
      self.assertTrue(is_positive)

    assert_array_is_positive(actual.mask_next)
    assert_array_is_positive(actual.partial_mask_blocks)
    assert_array_is_positive(actual.block_mask)
    assert_array_is_positive(actual.data_next)

    self.assertEqual(
        actual.data_next is not None, expected.data_next is not None
    )
    self.assertEqual(
        actual.block_mask is not None, expected.block_mask is not None
    )
    self.assertEqual(
        actual.mask_next is not None, expected.mask_next is not None
    )
    self.assertEqual(
        actual.partial_mask_blocks is not None,
        expected.partial_mask_blocks is not None,
    )

    self.assertEqual(
        actual.q_sequence is not None, expected.q_sequence is not None
    )

    if actual.partial_mask_blocks is not None:
      self.assertArraysEqual(
          actual.partial_mask_blocks,
          expected.partial_mask_blocks,
          err_msg="partial_mask_blocks",
          verbose=True,
      )

    if actual.q_sequence is not None:
      self.assertArraysEqual(
          actual.q_sequence,
          expected.q_sequence,
          err_msg="q_sequence",
          verbose=True,
      )

    self.assertArraysEqual(
        actual.block_mask,
        expected.block_mask,
        err_msg="block_mask",
        verbose=True,
    )

    if actual.data_next is not None and actual.block_mask is not None:
      self.assertEqual(actual.data_next.shape, actual.block_mask.shape)

    if actual.block_mask is not None:
      is_non_zero_block = np.where(expected.block_mask > 0, True, False)

      self.assertArraysEqual(
          np.where(is_non_zero_block, actual.data_next, -1),
          expected.data_next,
          err_msg="data_next",
          verbose=True,
      )

    if actual.mask_next is not None:
      is_partial_block = np.where(expected.block_mask == 1, True, False)
      self.assertArraysEqual(
          np.where(is_partial_block, actual.mask_next, -1),
          expected.mask_next,
          err_msg="mask_next",
          verbose=True,
      )

  def _process_mask(self, *args, **kwargs):
    mask_info, mask_function = mask_info_lib.process_mask(*args, **kwargs)
    mask_info_dkv, dkv_mask_function = mask_info_lib.process_mask_dkv(
        *args, **kwargs
    )
    self.assertEqual(mask_function, dkv_mask_function)
    return mask_info, mask_info_dkv, mask_function

  _expected_full_block_mask = np.array(
      [
          [2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2],
      ],
      dtype=np.int8,
  )

  _expected_full_block_mask_dkv = _expected_full_block_mask

  _expected_full_data_next = np.array(
      [
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ],
      dtype=np.int8,
  )

  _expected_full_data_next_dkv = np.array(
      [
          [0, 0, 0, 0],
          [1, 1, 1, 1],
          [2, 2, 2, 2],
          [3, 3, 3, 3],
      ],
      dtype=np.int8,
  )

  # The mask_next array for a full mask is typically empty. The exception to
  # this is when one head has a full mask and other heads have non-full masks.
  # In that case the mask_next array is full, but none of its elements are
  # actually relevant (they are never read).
  def _expected_full_mask_next(self):
    return np.array(
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int8,
    )

  _expected_full_mask_next_dkv = _expected_full_mask_next

  _expected_causal_block_mask = np.array(
      [
          [1, 0, 0, 0],
          [2, 1, 0, 0],
          [2, 2, 1, 0],
          [2, 2, 2, 1],
      ],
      dtype=np.int8,
  )

  _expected_causal_block_mask_dkv = _expected_causal_block_mask

  _expected_causal_data_next = np.array(
      [
          [0, -1, -1, -1],
          [0, 1, -1, -1],
          [0, 1, 2, -1],
          [0, 1, 2, 3],
      ],
      dtype=np.int8,
  )

  _expected_causal_data_next_dkv = np.array(
      [
          [0, -1, -1, -1],
          [1, 1, -1, -1],
          [2, 2, 2, -1],
          [3, 3, 3, 3],
      ],
      dtype=np.int8,
  )

  def _expected_causal_mask_next(self, mask_base_index: int):
    zero = mask_base_index
    return np.array(
        [
            [zero, -1, -1, -1],
            [-1, zero, -1, -1],
            [-1, -1, zero, -1],
            [-1, -1, -1, zero],
        ],
        dtype=np.int8,
    )

  _expected_causal_mask_next_dkv = _expected_causal_mask_next

  _expected_local_block_mask = np.array(
      [
          [1, 1, 0, 0],
          [1, 1, 1, 0],
          [0, 1, 1, 1],
          [0, 0, 1, 1],
      ],
      dtype=np.int8,
  )

  _expected_local_block_mask_dkv = _expected_local_block_mask

  _expected_local_data_next = np.array(
      [
          [0, 1, -1, -1],
          [0, 1, 2, -1],
          [-1, 1, 2, 3],
          [-1, -1, 2, 3],
      ],
      dtype=np.int8,
  )

  _expected_local_data_next_dkv = np.array(
      [
          [0, 0, -1, -1],
          [1, 1, 1, -1],
          [-1, 2, 2, 2],
          [-1, -1, 3, 3],
      ],
      dtype=np.int8,
  )

  def _expected_local_mask_next(self, mask_base_index: int):
    zero = mask_base_index
    one = mask_base_index + 1
    two = mask_base_index + 2
    return np.array(
        [
            [zero, one, -1, -1],
            [two, zero, one, -1],
            [-1, two, zero, one],
            [-1, -1, two, zero],
        ],
        dtype=np.int8,
    )

  _expected_local_mask_next_dkv = _expected_local_mask_next

  def _stack(self, arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack(arrays, axis=0)

  # For each test, check both the lazy and the dense versions of the mask.
  @parameterized.parameters((True,), (False,))
  def test_full_mask(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)

    if is_lazy_mask:
      full_mask = mask_lib.MultiHeadMask((mask_lib.FullMask(sequence_lengths),))
    else:
      full_mask = mask_lib.MultiHeadMask((
          mask_lib.NumpyMask(np.ones(sequence_lengths, dtype=np.bool_)),
      ))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        full_mask, block_shape
    )
    self.assertIsNone(mask_function)

    expected_mask_info = mask_info_lib.MaskInfo(
        self._expected_full_data_next[None],
        None,
        self._expected_full_block_mask[None],
        None,
        None,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        self._expected_full_data_next_dkv[None],
        None,
        self._expected_full_block_mask[None],
        None,
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_two_causal_masks(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
    else:
      causal_mask = mask_lib.NumpyMask(
          mask_lib.make_causal_mask(sequence_lengths)
      )

    multi_head = mask_lib.MultiHeadMask((causal_mask, causal_mask))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        multi_head, block_shape
    )
    if is_lazy_mask:
      self.assertIsNotNone(mask_function)
    else:
      self.assertIsNone(mask_function)

    expected_mask_info = mask_info_lib.MaskInfo(
        self._expected_causal_data_next[None],
        self._expected_causal_mask_next(0)[None] if not is_lazy_mask else None,
        self._expected_causal_block_mask[None],
        np.expand_dims(np.tril(np.ones(block_shape, dtype=np.int32)), 0)
        if not is_lazy_mask
        else None,
        np.arange(sequence_lengths[0], dtype=np.int32)
        if is_lazy_mask
        else None,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        self._expected_causal_data_next_dkv[None],
        self._expected_causal_mask_next_dkv(0)[None]
        if not is_lazy_mask
        else None,
        self._expected_causal_block_mask_dkv[None],
        np.expand_dims(
            np.tril(np.ones(block_shape, dtype=np.int32)), 0
        ).swapaxes(-1, -2)
        if not is_lazy_mask
        else None,
        np.arange(sequence_lengths[0], dtype=np.int32)
        if is_lazy_mask
        else None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_rectangular_wide_causal_mask(self, is_lazy_mask: bool):
    sequence_lengths = (64, 128)
    block_shape = (16, 16)

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
    else:
      causal_mask = mask_lib.NumpyMask(
          mask_lib.make_causal_mask(sequence_lengths)
      )

    multi_head = mask_lib.MultiHeadMask((causal_mask,))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        multi_head, block_shape
    )
    if is_lazy_mask:
      self.assertIsNotNone(mask_function)
    else:
      self.assertIsNone(mask_function)

    expected_mask_info = mask_info_lib.MaskInfo(
        self._expected_causal_data_next[None],
        self._expected_causal_mask_next(0)[None] if not is_lazy_mask else None,
        self._expected_causal_block_mask[None],
        np.expand_dims(np.tril(np.ones(block_shape, dtype=np.int32)), 0)
        if not is_lazy_mask
        else None,
        np.arange(sequence_lengths[0], dtype=np.int32)
        if is_lazy_mask
        else None,
    )

    expected_causal_data_next_dkv = np.array(
        [[
            [0, -1, -1, -1, -1, -1, -1, -1],
            [1, 1, -1, -1, -1, -1, -1, -1],
            [2, 2, 2, -1, -1, -1, -1, -1],
            [3, 3, 3, 3, -1, -1, -1, -1],
        ]],
        dtype=np.int8,
    )

    expected_causal_mask_next_dkv = np.array(
        [[
            [0, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, -1, -1, -1, -1, -1, -1],
            [-1, -1, 0, -1, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1, -1],
        ]],
        dtype=np.int8,
    )

    expected_causal_block_mask_dkv = np.array(
        [[
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0, 0],
            [2, 2, 1, 0, 0, 0, 0, 0],
            [2, 2, 2, 1, 0, 0, 0, 0],
        ]],
        dtype=np.int8,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_causal_data_next_dkv,
        expected_causal_mask_next_dkv if not is_lazy_mask else None,
        expected_causal_block_mask_dkv,
        np.expand_dims(
            np.tril(np.ones(block_shape, dtype=np.int32)), 0
        ).swapaxes(-1, -2)
        if not is_lazy_mask
        else None,
        np.arange(sequence_lengths[0], dtype=np.int32)
        if is_lazy_mask
        else None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_rectangular_tall_causal_mask(self, is_lazy_mask: bool):
    sequence_lengths = (128, 64)
    block_shape = (16, 16)

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
    else:
      causal_mask = mask_lib.NumpyMask(
          mask_lib.make_causal_mask(sequence_lengths)
      )

    multi_head = mask_lib.MultiHeadMask((causal_mask,))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        multi_head, block_shape
    )
    if is_lazy_mask:
      self.assertIsNotNone(mask_function)
    else:
      self.assertIsNone(mask_function)

    expected_causal_data_next = np.array(
        [[
            [0, -1, -1, -1],
            [0, 1, -1, -1],
            [0, 1, 2, -1],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]],
        dtype=np.int8,
    )

    expected_causal_mask_next = np.array(
        [[
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ]],
        dtype=np.int8,
    )

    expected_causal_block_mask = np.array(
        [[
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [2, 2, 1, 0],
            [2, 2, 2, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]],
        dtype=np.int8,
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_causal_data_next,
        expected_causal_mask_next if not is_lazy_mask else None,
        expected_causal_block_mask,
        np.expand_dims(np.tril(np.ones(block_shape, dtype=np.int32)), 0)
        if not is_lazy_mask
        else None,
        np.arange(sequence_lengths[0], dtype=np.int32)
        if is_lazy_mask
        else None,
    )

    expected_causal_data_next_dkv = np.array(
        [[
            [0, -1, -1, -1],
            [1, 1, -1, -1],
            [2, 2, 2, -1],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [7, 7, 7, 7],
        ]],
        dtype=np.int8,
    )

    expected_causal_mask_next_dkv = np.array(
        [[
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ]],
        dtype=np.int8,
    )

    expected_causal_block_mask_dkv = np.array(
        [[
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [2, 2, 1, 0],
            [2, 2, 2, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]],
        dtype=np.int8,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_causal_data_next_dkv,
        expected_causal_mask_next_dkv if not is_lazy_mask else None,
        expected_causal_block_mask_dkv,
        np.expand_dims(
            np.tril(np.ones(block_shape, dtype=np.int32)), 0
        ).swapaxes(-1, -2)
        if not is_lazy_mask
        else None,
        np.arange(sequence_lengths[0], dtype=np.int32)
        if is_lazy_mask
        else None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_local_mask(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8
    if is_lazy_mask:
      local_mask = mask_lib.LocalMask(
          sequence_lengths,
          window_size=(window_size, window_size),
          offset=0,
      )
    else:
      local_mask = mask_lib.NumpyMask(
          mask_lib.make_local_attention_mask(
              sequence_lengths, window_size=(window_size, window_size), offset=0
          )
      )

    multi_head = mask_lib.MultiHeadMask((local_mask,))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        multi_head, block_shape
    )
    self.assertIsNone(mask_function)

    expected_partial_mask_blocks = self._stack(
        [
            np.triu(
                np.tri(*block_shape, window_size, dtype=np.int32), -window_size
            ),
            np.tri(*block_shape, -window_size, dtype=np.int32),
            np.triu(np.ones(block_shape, dtype=np.int32), window_size),
        ],
    )

    expected_local_data_next = np.array(
        [[
            [0, 1, -1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, -1],
        ]],
        dtype=np.int8,
    )

    expected_local_mask_next = np.array(
        [[
            [0, 1, -1],
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, -1],
        ]],
        dtype=np.int8,
    )

    expected_local_block_mask = np.array(
        [[
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]],
        dtype=np.int8,
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_local_data_next,
        expected_local_mask_next,
        expected_local_block_mask,
        expected_partial_mask_blocks,
        None,
    )

    expected_local_data_next_dkv = np.array(
        [[
            [-1, 0, 1, -1],
            [0, 1, 2, 2],
            [1, 2, 3, 3],
        ]],
        dtype=np.int8,
    )

    expected_local_mask_next_dkv = np.array(
        [[
            [-1, 1, 1, -1],
            [0, 0, 0, 1],
            [2, 2, 2, 0],
        ]],
        dtype=np.int8,
    )

    expected_local_block_mask_dkv = np.array(
        [[
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]],
        dtype=np.int8,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_local_data_next_dkv,
        expected_local_mask_next_dkv,
        expected_local_block_mask_dkv,
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_local_mask_narrow(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8
    if is_lazy_mask:
      local_mask = mask_lib.LocalMask(
          sequence_lengths,
          window_size=(window_size, 0),
          offset=0,
      )
    else:
      local_mask = mask_lib.NumpyMask(
          mask_lib.make_local_attention_mask(
              sequence_lengths, window_size=(window_size, 0), offset=0
          )
      )

    multi_head = mask_lib.MultiHeadMask((local_mask,))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        multi_head, block_shape
    )
    self.assertIsNone(mask_function)

    expected_partial_mask_blocks = self._stack(
        [
            np.triu(np.tri(*block_shape, 0, dtype=np.int32), -window_size),
            np.triu(np.ones(block_shape, dtype=np.int32), window_size),
        ],
    )

    expected_local_data_next = np.array(
        [[
            [0, -1],
            [0, 1],
            [1, 2],
            [2, 3],
        ]],
        dtype=np.int8,
    )

    expected_local_mask_next = np.array(
        [[
            [0, -1],
            [1, 0],
            [1, 0],
            [1, 0],
        ]],
        dtype=np.int8,
    )

    expected_local_block_mask = np.array(
        [[
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 1],
        ]],
        dtype=np.int8,
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_local_data_next,
        expected_local_mask_next,
        expected_local_block_mask,
        expected_partial_mask_blocks,
        None,
    )

    expected_local_data_next_dkv = np.array(
        [[
            [0, 1, 2, -1],
            [1, 2, 3, 3],
        ]],
        dtype=np.int8,
    )

    expected_local_mask_next_dkv = np.array(
        [[
            [0, 0, 0, -1],
            [1, 1, 1, 0],
        ]],
        dtype=np.int8,
    )

    expected_local_block_mask_dkv = np.array(
        [[
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]],
        dtype=np.int8,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_local_data_next_dkv,
        expected_local_mask_next_dkv,
        expected_local_block_mask_dkv,
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_two_head_shards_one_causal_one_local(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
      local_mask = mask_lib.LocalMask(
          sequence_lengths,
          window_size=(window_size, window_size),
          offset=0,
      )
    else:
      causal_mask = mask_lib.NumpyMask(
          mask_lib.make_causal_mask(sequence_lengths)
      )
      local_mask = mask_lib.NumpyMask(
          mask_lib.make_local_attention_mask(
              sequence_lengths, window_size=(window_size, window_size), offset=0
          )
      )

    mask = mask_lib.MultiHeadMask((causal_mask, local_mask))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        mask, block_shape, head_shards=2
    )
    self.assertIsNone(mask_function)

    expected_block_mask = self._stack(
        [self._expected_causal_block_mask, self._expected_local_block_mask]
    )
    expected_data_next = self._stack(
        [self._expected_causal_data_next, self._expected_local_data_next]
    )
    expected_mask_next = self._stack(
        [self._expected_causal_mask_next(0), self._expected_local_mask_next(1)],
    )

    expected_partial_mask_blocks = self._stack([
        np.tril(np.ones(block_shape, dtype=np.int32)),
        np.triu(
            np.tri(*block_shape, window_size, dtype=np.int32),
            -window_size,
        ),
        np.tri(*block_shape, -window_size, dtype=np.int32),
        np.triu(np.ones(block_shape, dtype=np.int32), window_size),
    ])

    expected_block_mask_dkv = self._stack(
        [
            self._expected_causal_block_mask_dkv,
            self._expected_local_block_mask_dkv,
        ],
    )
    expected_data_next_dkv = self._stack(
        [
            self._expected_causal_data_next_dkv,
            self._expected_local_data_next_dkv,
        ],
    )
    expected_mask_next_dkv = self._stack(
        [
            self._expected_causal_mask_next_dkv(0),
            self._expected_local_mask_next_dkv(1),
        ],
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_data_next,
        expected_mask_next,
        expected_block_mask,
        expected_partial_mask_blocks,
        None,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_data_next_dkv,
        expected_mask_next_dkv,
        expected_block_mask_dkv,
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_two_head_shards_causal_full(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
      full_mask = mask_lib.FullMask(sequence_lengths)
    else:
      causal_mask = mask_lib.NumpyMask(
          mask_lib.make_causal_mask(sequence_lengths)
      )
      full_mask = mask_lib.NumpyMask(np.ones(sequence_lengths, dtype=np.bool_))

    mask = mask_lib.MultiHeadMask((causal_mask, full_mask))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        mask, block_shape, head_shards=2
    )
    self.assertIsNone(mask_function)

    expected_block_mask = self._stack(
        [
            self._expected_causal_block_mask,
            self._expected_full_block_mask,
        ],
    )

    expected_data_next = self._stack([
        self._expected_causal_data_next,
        self._expected_full_data_next,
    ])

    expected_mask_next = self._stack([
        self._expected_causal_mask_next(0),
        self._expected_full_mask_next(),
    ])

    expected_partial_mask_blocks = np.expand_dims(
        np.tril(np.ones(block_shape, dtype=np.int32)), 0
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_data_next,
        expected_mask_next,
        expected_block_mask,
        expected_partial_mask_blocks,
        None,
    )

    expected_block_mask_dkv = self._stack([
        self._expected_causal_block_mask_dkv,
        self._expected_full_block_mask_dkv,
    ])
    expected_data_next_dkv = self._stack(
        [self._expected_causal_data_next_dkv, self._expected_full_data_next_dkv]
    )

    expected_mask_next_dkv = self._stack([
        self._expected_causal_mask_next_dkv(0),
        self._expected_full_mask_next_dkv(),
    ])

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_data_next_dkv,
        expected_mask_next_dkv,
        expected_block_mask_dkv,
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_two_qseq_shards_causal_local(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
      local_mask = mask_lib.LocalMask(
          sequence_lengths,
          window_size=(window_size, window_size),
          offset=0,
      )
    else:
      causal_mask = mask_lib.NumpyMask(
          mask_lib.make_causal_mask(sequence_lengths)
      )
      local_mask = mask_lib.NumpyMask(
          mask_lib.make_local_attention_mask(
              sequence_lengths, window_size=(window_size, window_size), offset=0
          )
      )

    mask = mask_lib.MultiHeadMask((causal_mask, local_mask))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        mask, block_shape, q_seq_shards=2
    )
    self.assertIsNone(mask_function)

    expected_block_mask = self._stack(
        [self._expected_causal_block_mask, self._expected_local_block_mask]
    )
    expected_data_next = self._stack(
        [self._expected_causal_data_next, self._expected_local_data_next]
    )
    expected_mask_next = self._stack(
        [self._expected_causal_mask_next(0), self._expected_local_mask_next(1)]
    )

    expected_partial_mask_blocks = self._stack([
        np.tril(np.ones(block_shape, dtype=np.int32)),
        np.triu(
            np.tri(*block_shape, window_size, dtype=np.int32),
            -window_size,
        ),
        np.tri(*block_shape, -window_size, dtype=np.int32),
        np.triu(np.ones(block_shape, dtype=np.int32), window_size),
    ])

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_data_next,
        expected_mask_next,
        expected_block_mask,
        expected_partial_mask_blocks,
        None,
    )

    expected_block_mask_dkv = self._stack([
        self._expected_causal_block_mask_dkv,
        self._expected_local_block_mask_dkv,
    ])
    expected_data_next_dkv = np.array(
        [
            [
                [0, -1, -1, -1],
                [1, 1, -1, -1],
                [0, 0, 0, -1],
                [1, 1, 1, 1],
            ],
            [
                [0, 0, -1, -1],
                [1, 1, 1, -1],
                [-1, 0, 0, 0],
                [-1, -1, 1, 1],
            ],
        ],
        dtype=np.int8,
    )

    expected_mask_next_dkv = self._stack([
        self._expected_causal_mask_next_dkv(0),
        self._expected_local_mask_next_dkv(1),
    ])

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_data_next_dkv,
        expected_mask_next_dkv,
        expected_block_mask_dkv,
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  def test_two_qseq_shards_causal_local_stacked(self):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8

    causal_mask = mask_lib.make_causal_mask(sequence_lengths)
    local_mask = mask_lib.make_local_attention_mask(
        sequence_lengths, window_size=(window_size, window_size), offset=0
    )
    mask = np.concatenate((causal_mask, local_mask), axis=0)
    mask = mask_lib.NumpyMask(mask)
    mask = mask_lib.MultiHeadMask((mask,))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        mask, block_shape, q_seq_shards=2
    )
    self.assertIsNone(mask_function)

    expected_local_block_mask = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )

    expected_local_data_next = np.array(
        [
            [0, 1, -1, -1],
            [0, 1, 2, -1],
            [1, 2, 3, -1],
            [2, 3, -1, -1],
        ],
        dtype=np.int8,
    )

    expected_local_mask_next = np.array(
        [
            [1, 2, -1, -1],
            [3, 1, 2, -1],
            [3, 1, 2, -1],
            [3, 1, -1, -1],
        ],
        dtype=np.int8,
    )

    expected_block_mask = np.concatenate(
        [self._expected_causal_block_mask, expected_local_block_mask],
        axis=0,
    )
    expected_data_next = np.concatenate(
        [self._expected_causal_data_next, expected_local_data_next],
        axis=0,
    )
    expected_mask_next = np.concatenate(
        [self._expected_causal_mask_next(0), expected_local_mask_next],
        axis=0,
    )

    expected_partial_mask_blocks = self._stack([
        np.tril(np.ones(block_shape, dtype=np.int32)),
        np.triu(
            np.tri(*block_shape, window_size, dtype=np.int32),
            -window_size,
        ),
        np.tri(*block_shape, -window_size, dtype=np.int32),
        np.triu(np.ones(block_shape, dtype=np.int32), window_size),
    ])

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_data_next[None],
        expected_mask_next[None],
        expected_block_mask[None],
        expected_partial_mask_blocks,
        None,
    )

    # TODO(amagni): this mask can be improved by bringing all the padding on one
    # side.
    expected_local_block_mask_dkv = np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )

    expected_block_mask_dkv = np.concatenate([
        self._expected_causal_block_mask_dkv,
        expected_local_block_mask_dkv,
    ])

    expected_local_data_next_dkv = np.array(
        [
            [-1, 0, 1, -1],
            [0, 1, 2, 2],
            [1, 2, 3, 3],
            [-1, -1, -1, -1],
        ],
        dtype=np.int8,
    )

    expected_data_next_dkv = np.concatenate([
        self._expected_causal_data_next_dkv,
        expected_local_data_next_dkv,
    ])

    expected_local_mask_next_dkv = np.array(
        [
            [-1, 2, 2, -1],
            [1, 1, 1, 2],
            [3, 3, 3, 1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int8,
    )

    expected_mask_next_dkv = np.concatenate([
        self._expected_causal_mask_next_dkv(0),
        expected_local_mask_next_dkv,
    ])

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_data_next_dkv[None],
        expected_mask_next_dkv[None],
        expected_block_mask_dkv[None],
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  def test_two_qseq_shards_local_wide_local_narrow_stacked(self):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8

    local_mask_wide = mask_lib.make_local_attention_mask(
        sequence_lengths, window_size=(window_size, window_size), offset=0
    )
    local_mask_narrow = mask_lib.make_local_attention_mask(
        sequence_lengths, window_size=(window_size, 0), offset=0
    )

    mask = np.concatenate((local_mask_wide, local_mask_narrow), axis=0)
    mask = mask_lib.NumpyMask(mask)
    mask = mask_lib.MultiHeadMask((mask,))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        mask, block_shape, q_seq_shards=2
    )
    self.assertIsNone(mask_function)

    expected_local_wide_block_mask = np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )

    expected_local_wide_data_next = np.array(
        [
            [0, 1, -1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, -1],
        ],
        dtype=np.int8,
    )

    expected_local_wide_mask_next = np.array(
        [
            [0, 1, -1],
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, -1],
        ],
        dtype=np.int8,
    )

    expected_local_narrow_data_next = np.array(
        [
            [0, -1, -1],
            [0, 1, -1],
            [1, 2, -1],
            [2, 3, -1],
        ],
        dtype=np.int8,
    )

    expected_local_narrow_mask_next = np.array(
        [
            [3, -1, -1],
            [2, 3, -1],
            [2, 3, -1],
            [2, 3, -1],
        ],
        dtype=np.int8,
    )

    expected_local_narrow_block_mask = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )

    expected_block_mask = np.concatenate(
        [expected_local_wide_block_mask, expected_local_narrow_block_mask],
        axis=0,
    )
    expected_data_next = np.concatenate(
        [expected_local_wide_data_next, expected_local_narrow_data_next],
        axis=0,
    )
    expected_mask_next = np.concatenate(
        [expected_local_wide_mask_next, expected_local_narrow_mask_next],
        axis=0,
    )

    expected_partial_mask_blocks = self._stack([
        # Wide
        np.triu(
            np.tri(*block_shape, window_size, dtype=np.int32),
            -window_size,
        ),
        np.tri(*block_shape, -window_size, dtype=np.int32),
        np.triu(np.ones(block_shape, dtype=np.int32), window_size),
        # Narrow
        np.triu(np.tri(*block_shape, 0, dtype=np.int32), -window_size),
    ])

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_data_next[None],
        expected_mask_next[None],
        expected_block_mask[None],
        expected_partial_mask_blocks,
        None,
    )

    expected_local_wide_block_mask_dkv = np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int8,
    )

    expected_local_wide_data_next_dkv = np.array(
        [
            [-1, 0, 1, -1],
            [0, 1, 2, 2],
            [1, 2, 3, 3],
        ],
        dtype=np.int8,
    )

    expected_local_wide_mask_next_dkv = np.array(
        [
            [-1, 1, 1, -1],
            [0, 0, 0, 1],
            [2, 2, 2, 0],
        ],
        dtype=np.int8,
    )

    expected_local_narrow_data_next_dkv = np.array(
        [
            [0, 1, 2, -1],
            [1, 2, 3, 3],
            [-1, -1, -1, -1],
        ],
        dtype=np.int8,
    )

    expected_local_narrow_mask_next_dkv = np.array(
        [
            [3, 3, 3, -1],
            [2, 2, 2, 3],
            [-1, -1, -1, -1],
        ],
        dtype=np.int8,
    )

    expected_local_narrow_block_mask_dkv = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )

    expected_block_mask_dkv = np.concatenate(
        [
            expected_local_wide_block_mask_dkv,
            expected_local_narrow_block_mask_dkv,
        ],
        axis=0,
    )

    expected_data_next_dkv = np.concatenate(
        [
            expected_local_wide_data_next_dkv,
            expected_local_narrow_data_next_dkv,
        ],
        axis=0,
    )

    expected_mask_next_dkv = np.concatenate(
        [
            expected_local_wide_mask_next_dkv,
            expected_local_narrow_mask_next_dkv,
        ],
        axis=0,
    )

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_data_next_dkv[None],
        expected_mask_next_dkv[None],
        expected_block_mask_dkv[None],
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_two_head_shards_causal_mask(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
    else:
      causal_mask = mask_lib.NumpyMask(
          mask_lib.make_causal_mask(sequence_lengths)
      )

    mask = mask_lib.MultiHeadMask((causal_mask, causal_mask))

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        mask, block_shape, head_shards=2
    )
    if is_lazy_mask:
      self.assertIsNotNone(mask_function)
    else:
      self.assertIsNone(mask_function)

    expected_block_mask = self._stack(
        [self._expected_causal_block_mask, self._expected_causal_block_mask]
    )

    expected_data_next = self._stack(
        [self._expected_causal_data_next, self._expected_causal_data_next]
    )

    expected_mask_next = self._stack(
        [self._expected_causal_mask_next(0), self._expected_causal_mask_next(0)]
    )

    expected_partial_mask_blocks = np.expand_dims(
        np.tril(np.ones(block_shape, dtype=np.int32)), 0
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_data_next,
        expected_mask_next if not is_lazy_mask else None,
        expected_block_mask,
        expected_partial_mask_blocks if not is_lazy_mask else None,
        np.arange(sequence_lengths[0], dtype=np.int32)
        if is_lazy_mask
        else None,
    )

    expected_block_mask_dkv = self._stack([
        self._expected_causal_block_mask_dkv,
        self._expected_causal_block_mask_dkv,
    ])

    expected_data_next_dkv = self._stack([
        self._expected_causal_data_next_dkv,
        self._expected_causal_data_next_dkv,
    ])

    expected_mask_next_dkv = self._stack([
        self._expected_causal_mask_next_dkv(0),
        self._expected_causal_mask_next_dkv(0),
    ])

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_data_next_dkv,
        expected_mask_next_dkv if not is_lazy_mask else None,
        expected_block_mask_dkv,
        expected_partial_mask_blocks.swapaxes(-1, -2)
        if not is_lazy_mask
        else None,
        np.arange(sequence_lengths[0], dtype=np.int32)
        if is_lazy_mask
        else None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  @parameterized.parameters((True,), (False,))
  def test_two_head_shards_two_causal_two_local(self, is_lazy_mask: bool):
    sequence_lengths = (64, 64)
    block_shape = (16, 16)
    window_size = 8

    if is_lazy_mask:
      causal_mask = mask_lib.CausalMask(sequence_lengths)
      local_mask = mask_lib.LocalMask(
          sequence_lengths,
          window_size=(window_size, window_size),
          offset=0,
      )
    else:
      causal_mask = mask_lib.NumpyMask(
          mask_lib.make_causal_mask(sequence_lengths)
      )
      local_mask = mask_lib.NumpyMask(
          mask_lib.make_local_attention_mask(
              sequence_lengths, window_size=(window_size, window_size), offset=0
          )
      )

    mask = mask_lib.MultiHeadMask(
        (causal_mask, causal_mask, local_mask, local_mask)
    )

    mask_info, mask_info_dkv, mask_function = self._process_mask(
        mask, block_shape, head_shards=2
    )
    self.assertIsNone(mask_function)

    expected_block_mask = self._stack(
        [self._expected_causal_block_mask, self._expected_local_block_mask]
    )

    expected_data_next = self._stack(
        [self._expected_causal_data_next, self._expected_local_data_next]
    )

    expected_mask_next = self._stack(
        [self._expected_causal_mask_next(0), self._expected_local_mask_next(1)]
    )

    expected_partial_mask_blocks = self._stack(
        [
            np.tril(np.ones(block_shape, dtype=np.int32)),
            np.triu(
                np.tri(*block_shape, window_size, dtype=np.int32),
                -window_size,
            ),
            np.tri(*block_shape, -window_size, dtype=np.int32),
            np.triu(np.ones(block_shape, dtype=np.int32), window_size),
        ],
    )

    expected_mask_info = mask_info_lib.MaskInfo(
        expected_data_next,
        expected_mask_next,
        expected_block_mask,
        expected_partial_mask_blocks,
        None,
    )

    expected_block_mask_dkv = self._stack([
        self._expected_causal_block_mask_dkv,
        self._expected_local_block_mask_dkv,
    ])

    expected_data_next_dkv = self._stack([
        self._expected_causal_data_next_dkv,
        self._expected_local_data_next_dkv,
    ])

    expected_mask_next_dkv = self._stack([
        self._expected_causal_mask_next_dkv(0),
        self._expected_local_mask_next_dkv(1),
    ])

    expected_mask_info_dkv = mask_info_lib.MaskInfo(
        expected_data_next_dkv,
        expected_mask_next_dkv,
        expected_block_mask_dkv,
        expected_partial_mask_blocks.swapaxes(-1, -2),
        None,
    )

    self._assert_mask_info_match(mask_info, expected_mask_info)
    self._assert_mask_info_match(mask_info_dkv, expected_mask_info_dkv)

  def test_huge_mask(self):
    # Don't go too high with the mask size to avoid timeouts. Prefer covering
    # multiple cases rather one very large one. This configuration replicates
    # a realistic training shape. In particular, a large number of head shards
    # and interleaving contribute to increasing processing time.
    sequence_length = (32 * 1024, 32 * 1024)
    block_shape = (512, 1024)

    num_shards = 16
    causal_mask = mask_lib.CausalMask(
        sequence_length, 0, shard_count=num_shards
    )

    multi_head = mask_lib.MultiHeadMask((causal_mask,) * 64)

    mask_info, mask_function = mask_info_lib.process_mask(
        multi_head, block_shape, head_shards=8, q_seq_shards=16
    )

    self.assertIsNotNone(mask_function)
    self.assertIsNotNone(mask_info.block_mask)
    self.assertIsNotNone(mask_info.data_next)
    self.assertIsNone(mask_info.mask_next)
    self.assertIsNone(mask_info.partial_mask_blocks)
    self.assertIsNotNone(mask_info.q_sequence)

  def test_huge_mask2(self):
    sequence_lengths = (32 * 1024, 32 * 1024)
    block_shape = (1024, 1024)
    window_size = 8

    local_mask = mask_lib.LocalMask(
        sequence_lengths,
        window_size=(window_size, window_size),
        offset=0,
    )

    multi_head = mask_lib.MultiHeadMask((local_mask,) * 32)

    mask_info, mask_function = mask_info_lib.process_mask(
        multi_head, block_shape
    )

    self.assertIsNone(mask_function)
    self.assertIsNotNone(mask_info.block_mask)
    self.assertIsNotNone(mask_info.data_next)
    self.assertIsNotNone(mask_info.mask_next)
    self.assertIsNotNone(mask_info.partial_mask_blocks)

  def test_process_invalid_mask(self):
    """Masks with of an all-0 row causes undefined softmax, reject them."""
    sequence_length = 32

    invalid_mask = np.ones(
        (4, sequence_length, sequence_length), dtype=np.bool_
    )
    invalid_mask[2, 14, :] = False

    invalid_mask = mask_lib.MultiHeadMask(
        [mask_lib.NumpyMask(head_mask) for head_mask in invalid_mask]
    )

    with self.assertRaises(ValueError) as ctx:
      for mask in invalid_mask.masks:
        mask_info_lib._check_mask(mask)

    self.assertIn("softmax", str(ctx.exception))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
