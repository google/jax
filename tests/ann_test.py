# Copyright 2021 The JAX Authors.
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

from functools import partial
import math

from absl.testing import absltest

import numpy as np

import jax
from jax import lax
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()

ignore_jit_of_pmap_warning = partial(
    jtu.ignore_warning,message=".*jit-of-pmap.*")


def compute_recall(result_neighbors, ground_truth_neighbors) -> float:
  """Computes the recall of an approximate nearest neighbor search.

  Args:
    result_neighbors: int32 numpy array of the shape [num_queries,
      neighbors_per_query] where the values are the indices of the dataset.
    ground_truth_neighbors: int32 numpy array of with shape [num_queries,
      ground_truth_neighbors_per_query] where the values are the indices of the
      dataset.

  Returns:
    The recall.
  """
  assert len(
      result_neighbors.shape) == 2, "shape = [num_queries, neighbors_per_query]"
  assert len(ground_truth_neighbors.shape
            ) == 2, "shape = [num_queries, ground_truth_neighbors_per_query]"
  assert result_neighbors.shape[0] == ground_truth_neighbors.shape[0]
  gt_sets = [set(np.asarray(x)) for x in ground_truth_neighbors]
  hits = sum(len([x
                  for x in nn_per_q
                  if x.item() in gt_sets[q]])
             for q, nn_per_q in enumerate(result_neighbors))
  return hits / ground_truth_neighbors.size


class AnnTest(jtu.JaxTestCase):

  # TODO(b/258315194) Investigate probability property when input is around
  # few thousands.
  @jtu.sample_product(
    qy_shape=[(200, 128), (128, 128)],
    db_shape=[(128, 500), (128, 3000)],
    dtype=jtu.dtypes.all_floating,
    k=[1, 10],
    recall=[0.95],
  )
  def test_approx_max_k(self, qy_shape, db_shape, dtype, k, recall):
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, dtype)
    db = rng(db_shape, dtype)
    scores = lax.dot(qy, db)
    _, gt_args = lax.top_k(scores, k)
    _, ann_args = lax.approx_max_k(scores, k, recall_target=recall)
    self.assertEqual(k, len(ann_args[0]))
    ann_recall = compute_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreaterEqual(ann_recall, recall*0.9)

  @jtu.sample_product(
    qy_shape=[(200, 128), (128, 128)],
    db_shape=[(128, 500), (128, 3000)],
    dtype=jtu.dtypes.all_floating,
    k=[1, 10],
    recall=[0.95],
  )
  def test_approx_min_k(self, qy_shape, db_shape, dtype, k, recall):
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, dtype)
    db = rng(db_shape, dtype)
    scores = lax.dot(qy, db)
    _, gt_args = lax.top_k(-scores, k)
    _, ann_args = lax.approx_min_k(scores, k, recall_target=recall)
    ann_recall = compute_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreaterEqual(ann_recall, recall*0.9)

  @jtu.sample_product(
    dtype=[np.float32],
    shape=[(4,), (5, 5), (2, 1, 4)],
    k=[1, 3],
    is_max_k=[True, False],
  )
  def test_autodiff(self, shape, dtype, k, is_max_k):
    vals = np.arange(math.prod(shape), dtype=dtype)
    vals = self.rng().permutation(vals).reshape(shape)
    if is_max_k:
      fn = lambda vs: lax.approx_max_k(vs, k=k)[0]
    else:
      fn = lambda vs: lax.approx_min_k(vs, k=k)[0]
    jtu.check_grads(fn, (vals,), 2, ["fwd", "rev"], eps=1e-2)


  @jtu.sample_product(
    qy_shape=[(200, 128), (128, 128)],
    db_shape=[(2048, 128)],
    dtype=jtu.dtypes.all_floating,
    k=[1, 10],
    recall=[0.9, 0.95],
  )
  def test_pmap(self, qy_shape, db_shape, dtype, k, recall):
    num_devices = jax.device_count()
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, dtype)
    db = rng(db_shape, dtype)
    db_size = db.shape[0]
    gt_scores = lax.dot_general(qy, db, (([1], [1]), ([], [])))
    _, gt_args = lax.top_k(-gt_scores, k)  # negate the score to get min-k
    db_per_device = db_size//num_devices
    sharded_db = db.reshape(num_devices, db_per_device, 128)
    db_offsets = np.arange(num_devices, dtype=np.int32) * db_per_device
    def parallel_topk(qy, db, db_offset):
      scores = lax.dot_general(qy, db, (([1],[1]),([],[])))
      ann_vals, ann_args = lax.approx_min_k(
          scores,
          k=k,
          reduction_dimension=1,
          recall_target=recall,
          reduction_input_size_override=db_size,
          aggregate_to_topk=False)
      return (ann_vals, ann_args + db_offset)
    # shape = qy_size, num_devices, approx_dp
    ann_vals, ann_args = jax.pmap(
        parallel_topk,
        in_axes=(None, 0, 0),
        out_axes=(1, 1))(qy, sharded_db, db_offsets)
    # collapse num_devices and approx_dp
    ann_vals = lax.collapse(ann_vals, 1, 3)
    ann_args = lax.collapse(ann_args, 1, 3)
    ann_vals, ann_args = lax.sort_key_val(ann_vals, ann_args, dimension=1)
    ann_args = lax.slice_in_dim(ann_args, start_index=0, limit_index=k, axis=1)
    ann_recall = compute_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreater(ann_recall, recall)


  def test_vmap_before(self):
    batch = 4
    qy_size = 128
    db_size = 1024
    feature_dim = 32
    k = 10
    rng = jtu.rand_default(self.rng())
    qy = rng([batch, qy_size, feature_dim], np.float32)
    db = rng([batch, db_size, feature_dim], np.float32)
    recall = 0.95

    # Create ground truth
    gt_scores = lax.dot_general(qy, db, (([2], [2]), ([0], [0])))
    _, gt_args = lax.top_k(gt_scores, k)
    gt_args = lax.reshape(gt_args, [qy_size * batch, k])

    # test target
    def approx_max_k(qy, db):
      scores = qy @ db.transpose()
      return lax.approx_max_k(scores, k)
    _, ann_args = jax.vmap(approx_max_k, (0, 0))(qy, db)
    ann_args = lax.reshape(ann_args, [qy_size * batch, k])
    ann_recall = compute_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreater(ann_recall, recall)


  def test_vmap_after(self):
    batch = 4
    qy_size = 128
    db_size = 1024
    feature_dim = 32
    k = 10
    rng = jtu.rand_default(self.rng())
    qy = rng([qy_size, feature_dim, batch], np.float32)
    db = rng([db_size, feature_dim, batch], np.float32)
    recall = 0.95

    # Create ground truth
    gt_scores = lax.dot_general(qy, db, (([1], [1]), ([2], [2])))
    _, gt_args = lax.top_k(gt_scores, k)
    gt_args = lax.transpose(gt_args, [2, 0, 1])
    gt_args = lax.reshape(gt_args, [qy_size * batch, k])

    # test target
    def approx_max_k(qy, db):
      scores = qy @ db.transpose()
      return lax.approx_max_k(scores, k)
    _, ann_args = jax.vmap(approx_max_k, (2, 2))(qy, db)
    ann_args = lax.transpose(ann_args, [2, 0, 1])
    ann_args = lax.reshape(ann_args, [qy_size * batch, k])
    ann_recall = compute_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreater(ann_recall, recall)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
