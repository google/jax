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

import functools
import itertools
from typing import Any, Union

from absl.testing import absltest
from absl.testing import parameterized
import hypothesis as hp
import hypothesis.strategies as hps
import jax
from jax._src import test_util as jtu
import jax.experimental.pallas.ops.tpu.megablox as mblx
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

hp.settings.register_profile(
    "deterministic",
    database=None,
    derandomize=True,
    deadline=None,
    max_examples=30,
    print_blob=True,
)
hp.settings.load_profile("deterministic")

P = jax.sharding.PartitionSpec

partial = functools.partial


def seed_strategy() -> hps.SearchStrategy[int]:
  return hps.integers(min_value=0, max_value=4)


@hps.composite
def group_strategy(
    draw: hps.DrawFn,
    max_groups: int = 32,
    max_stride: int = 32,
    min_groups: int = 1,
) -> tuple[int, int]:
  assert max_stride <= max_groups

  # Sample the number of groups owned by each shard.
  group_stride = draw(hps.integers(min_value=1, max_value=max_stride))

  # Sample the number of groups as a multiple of the stride to ensure that we
  # have an equal number of groups per shard. Round down s.t. num_groups <=
  # max_groups.
  num_groups = group_stride * draw(
      hps.integers(min_value=min_groups, max_value=max_groups // group_stride)
  )
  return num_groups, group_stride


@hps.composite
def group_sizes_strategy(
    draw: hps.DrawFn, m: int, num_groups: int
) -> jnp.ndarray:
  # Randomly sample the ends of the groups in the m-dimension. Let the fuzzer
  # sample with replacement so that it's possible to get zero-sized groups. Get
  # 'num_groups - 1' run ends. The final group will end at 'm'.
  ends_no_final = np.sort(
      np.array(
          [
              draw(hps.integers(min_value=0, max_value=m))
              for _ in range(num_groups - 1)
          ],
          dtype=np.int32,
      ),
  )
  ends = np.concatenate([ends_no_final, np.array([m], dtype=np.int32)])

  # Calculate the run starts by shifting ends 1 to the right. The first run
  # starts at zero.
  starts = np.concatenate([np.zeros(1, dtype=np.int32), ends_no_final])
  return jnp.array(ends - starts, dtype=jnp.int32)


GROUPED_MATMUL_TESTS = (
    (128, 128, 128),
    (256, 128, 128),
    (128, 256, 128),
    (128, 128, 256),
    (256, 128, 512),
    (512, 128, 128),
    (512, 2048, 128),
    (128, 8, 16),  # Test partial tiles.
)


def random_dense(
    shape: tuple[int, ...],
    key: jax.Array,
    dtype: jnp.dtype,
    limit: Union[int, None] = None,
) -> jnp.ndarray:
  if limit is None:
    limit = 1 / np.prod(shape)
  x = jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)  # pylint: disable=invalid-unary-operand-type
  return x.astype(jnp.bfloat16).astype(dtype)


def dot(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    transpose_lhs: bool = False,
    transpose_rhs: bool = False,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
  lhs = jnp.transpose(lhs) if transpose_lhs else lhs
  rhs = jnp.transpose(rhs) if transpose_rhs else rhs
  return jax.lax.dot(lhs, rhs, preferred_element_type=preferred_element_type)


def reference_gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> jnp.ndarray:

  start = 0
  out = []
  for i, size in enumerate(group_sizes):
    result = dot(
        lhs[start : start + size, :],
        rhs[i, :, :],
        preferred_element_type=preferred_element_type,
    )

    out.append(result)
    start += group_sizes[i]
  return jnp.concatenate(out, axis=0)


def with_dtype_arguments(xs: tuple[Any, ...]) -> tuple[Any, ...]:
  dtypes = [jnp.float32, jnp.bfloat16]

  result = []
  for x in xs:
    for dtypes_tuple in itertools.product(dtypes, dtypes, dtypes):
      result.append(x + dtypes_tuple)
  return tuple(result)


def with_transpose_argument(xs: tuple[Any, ...]) -> tuple[Any, ...]:
  flags = [False, True]
  result = []
  for x in xs:
    for flag in flags:
      result.append(x + (flag,))
  return tuple(result)


def tolerances(
    lhs_dtype: jnp.dtype, rhs_dtype: jnp.dtype, out_dtype: jnp.dtype
) -> tuple[float, float]:
  if any(
      [dtype == jnp.bfloat16 for dtype in [lhs_dtype, rhs_dtype, out_dtype]]
  ):
    return 1e-3, 1e-2  # atol, rtol
  return 1e-3, 1e-5  # atol, rtol


# TODO(tgale): Fix errors with strict dtype promotion.
@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GroupedMatmulTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(1234)

    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Test requires TPU device.")

  def assert_allclose(
      self,
      out: jnp.ndarray,
      expected_out: jnp.ndarray,
      *,
      atol: float = 1e-5,
      rtol: float = 1e-5,
  ):
    self.assertEqual(out.dtype, expected_out.dtype)
    np.testing.assert_allclose(
        out.astype(jnp.float32),
        expected_out.astype(jnp.float32),
        atol=atol,
        rtol=rtol,
    )

  def gmm_test(
      self,
      m: int,
      k: int,
      n: int,
      lhs_dtype: jnp.dtype,
      rhs_dtype: jnp.dtype,
      out_dtype: jnp.dtype,
      transpose_rhs: bool,
      data: hps.SearchStrategy[hps.DataObject],
      interpret: bool = False,
  ):
    seed = data.draw(seed_strategy())
    num_groups, _ = data.draw(group_strategy(max_stride=1))

    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)
    lhs = random_dense((m, k), k1, lhs_dtype, limit=1)
    rhs = random_dense((num_groups, k, n), k2, rhs_dtype, limit=1)
    group_sizes = data.draw(group_sizes_strategy(m=m, num_groups=num_groups))

    out, vjpfun = jax.vjp(
        partial(
            mblx.gmm,
            preferred_element_type=out_dtype,
            transpose_rhs=transpose_rhs,
            interpret=interpret,
        ),
        lhs,
        rhs.swapaxes(1, 2) if transpose_rhs else rhs,
        group_sizes,
    )

    def reference_fn(lhs, rhs, group_sizes, preferred_element_type):
      rhs = rhs.swapaxes(1, 2) if transpose_rhs else rhs
      return reference_gmm(
          lhs, rhs, group_sizes, preferred_element_type=preferred_element_type
      )

    expected_out, reference_vjpfun = jax.vjp(
        partial(reference_fn, preferred_element_type=out_dtype),
        lhs,
        rhs.swapaxes(1, 2) if transpose_rhs else rhs,
        group_sizes,
    )
    self.assertEqual(out.dtype, out_dtype)
    self.assertEqual(expected_out.dtype, out_dtype)

    atol, rtol = tolerances(lhs_dtype, rhs_dtype, out_dtype)
    self.assert_allclose(out, expected_out, atol=atol, rtol=rtol)

    cotangent = random_dense((m, n), k1, out_dtype, limit=1)
    grad_lhs, grad_rhs, *_ = vjpfun(cotangent)
    expected_grad_lhs, expected_grad_rhs, *_ = reference_vjpfun(cotangent)
    self.assert_allclose(grad_lhs, expected_grad_lhs, atol=atol, rtol=rtol)
    self.assert_allclose(grad_rhs, expected_grad_rhs, atol=atol, rtol=rtol)

  @parameterized.parameters(
      *with_transpose_argument(with_dtype_arguments(GROUPED_MATMUL_TESTS))
  )
  @hp.given(hps.data())
  def test_gmm(
      self,
      m: int,
      k: int,
      n: int,
      lhs_dtype: jnp.dtype,
      rhs_dtype: jnp.dtype,
      out_dtype: jnp.dtype,
      transpose_rhs: bool,
      data: hps.SearchStrategy[hps.DataObject],
  ):
    self.gmm_test(m, k, n, lhs_dtype, rhs_dtype, out_dtype, transpose_rhs, data)

  # NOTE: Run fewer tests with interpret mode. We just want to sanity check that
  # changes do not break running these kernels with interpret=True.
  @parameterized.parameters(*with_dtype_arguments(GROUPED_MATMUL_TESTS[0:1]))
  @hp.given(hps.data())
  def test_gmm_interpret(
      self,
      m: int,
      k: int,
      n: int,
      lhs_dtype: jnp.dtype,
      rhs_dtype: jnp.dtype,
      out_dtype: jnp.dtype,
      data: hps.SearchStrategy[hps.DataObject],
  ):
    self.skipTest("interpret mode with dynamic grids is unsupported")
    self.gmm_test(
        m,
        k,
        n,
        lhs_dtype,
        rhs_dtype,
        out_dtype,
        transpose_rhs=False,
        data=data,
        interpret=True,
    )

  @parameterized.parameters(*with_dtype_arguments(GROUPED_MATMUL_TESTS))
  @hp.given(hps.data())
  def test_gmm_sharded_groups(
      self,
      m: int,
      k: int,
      n: int,
      lhs_dtype: jnp.dtype,
      rhs_dtype: jnp.dtype,
      out_dtype: jnp.dtype,
      data: hps.SearchStrategy[hps.DataObject],
  ):
    seed = data.draw(seed_strategy())
    num_groups, group_stride = data.draw(group_strategy())

    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)
    lhs = random_dense((m, k), k1, lhs_dtype, limit=1)
    rhs = random_dense((num_groups, k, n), k2, rhs_dtype, limit=1)
    group_sizes = data.draw(group_sizes_strategy(m=m, num_groups=num_groups))

    out, shard_vjpfun = jax.vjp(
        partial(mblx.gmm, preferred_element_type=out_dtype),
        lhs,
        rhs[0:group_stride],
        group_sizes,
    )
    vjpfuns = [shard_vjpfun]
    for group_offset in range(group_stride, num_groups, group_stride):
      out, shard_vjpfun = jax.vjp(
          lambda lhs, rhs, group_sizes, out: mblx.gmm(
              lhs,
              rhs,
              group_sizes,
              out_dtype,
              group_offset=jnp.array(group_offset, dtype=jnp.int32),  # pylint: disable=cell-var-from-loop
              existing_out=out,
          ),
          lhs,
          rhs[group_offset : group_offset + group_stride],
          group_sizes,
          out,
      )
      vjpfuns.append(shard_vjpfun)

    expected_out, reference_vjpfun = jax.vjp(
        partial(reference_gmm, preferred_element_type=out_dtype),
        lhs,
        rhs,
        group_sizes,
    )
    self.assertEqual(out.dtype, out_dtype)
    self.assertEqual(expected_out.dtype, out_dtype)
    atol, rtol = tolerances(lhs_dtype, rhs_dtype, out_dtype)
    self.assert_allclose(out, expected_out, atol=atol, rtol=rtol)

    cotangent = random_dense((m, n), k1, out_dtype, limit=1)
    shard_grad_lhs, shard_grad_rhs, *_ = vjpfuns[0](cotangent)
    grad_lhs = shard_grad_lhs
    grad_rhs = [shard_grad_rhs]
    for i, group_offset in enumerate(
        range(group_stride, num_groups, group_stride)
    ):
      shard_grad_lhs, shard_grad_rhs, *_ = vjpfuns[i + 1](cotangent)
      grad_lhs += shard_grad_lhs
      grad_rhs.append(shard_grad_rhs)
    grad_rhs = jnp.concatenate(grad_rhs, axis=0)
    expected_grad_lhs, expected_grad_rhs, *_ = reference_vjpfun(cotangent)
    self.assert_allclose(grad_lhs, expected_grad_lhs, atol=atol, rtol=rtol)
    self.assert_allclose(grad_rhs, expected_grad_rhs, atol=atol, rtol=rtol)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
