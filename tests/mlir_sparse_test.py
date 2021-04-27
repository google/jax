# Copyright 2021 Google LLC
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

import io
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
from jax.experimental import mlir_sparse
from jax import test_util as jtu

MAT_FROSTT = """
# extended FROSTT format
3 7
3 3 4
1 1 1  1.0
1 1 4  2.0
1 2 1  3.0
1 2 2  4.0
3 1 2  5.0
3 2 3  6.0
3 2 4  7.0
"""


class MLIRSparseTest(jtu.JaxTestCase):
  def test_known_input_DDS(self):
    # See https://llvm.discourse.group/t/mlir-support-for-sparse-tensors/2020/21
    M = mlir_sparse.MLIRSparse.fromfile(io.StringIO(MAT_FROSTT), format="DDS")

    self.assertIsNone(M.positions[0])
    self.assertIsNone(M.positions[1])
    self.assertArraysEqual(M.positions[2], jnp.array([0, 2, 4, 4, 4, 4, 4, 5, 7, 7]))

    self.assertIsNone(M.indices[0])
    self.assertIsNone(M.indices[1])
    self.assertArraysEqual(M.indices[2], jnp.array([0, 3, 0, 1, 1, 2, 3]))

    self.assertArraysEqual(M.values, jnp.array([1., 2., 3., 4., 5., 6., 7.]))

  def test_known_input_SSS(self):
    # See https://llvm.discourse.group/t/mlir-support-for-sparse-tensors/2020/21
    M = mlir_sparse.MLIRSparse.fromfile(io.StringIO(MAT_FROSTT), format="SSS")

    self.assertArraysEqual(M.positions[0], jnp.array([0, 2]))
    self.assertArraysEqual(M.positions[1], jnp.array([0, 2, 4]))
    self.assertArraysEqual(M.positions[2], jnp.array([0, 2, 4, 5, 7]))

    self.assertArraysEqual(M.indices[0], jnp.array([0, 2]))
    self.assertArraysEqual(M.indices[1], jnp.array([0, 1, 0, 1]))
    self.assertArraysEqual(M.indices[2], jnp.array([0, 3, 0, 1, 1, 2, 3]))

    self.assertArraysEqual(M.values, jnp.array([1., 2., 3., 4., 5., 6., 7.]))

  def test_known_input_SDS(self):
    # See https://llvm.discourse.group/t/rfc-introduce-a-sparse-tensor-type-to-core-mlir/2944/35
    M = mlir_sparse.MLIRSparse.fromfile(io.StringIO(MAT_FROSTT), format="SDS")

    self.assertArraysEqual(M.positions[0], jnp.array([0, 2]))
    self.assertIsNone(M.positions[1])
    self.assertArraysEqual(M.positions[2], jnp.array([0, 2, 4, 4, 5, 7, 7]))

    self.assertArraysEqual(M.indices[0], jnp.array([0, 2]))
    self.assertIsNone(M.indices[1])
    self.assertArraysEqual(M.indices[2], jnp.array([0, 3, 0, 1, 1, 2, 3]))

    self.assertArraysEqual(M.values, jnp.array([1., 2., 3., 4., 5., 6., 7.]))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_format={}".format(
          jtu.format_shape_dtype_string(shape, dtype), ''.join(format)),
       "shape": shape, "dtype": dtype, "format": format}
      for shape in [(5,), (4, 5), (4, 3, 5), (5, 3, 4)]
      for dtype in jtu.dtypes.all
      for format in itertools.product("SD", repeat=len(shape))))
  def test_round_trip(self, shape, dtype, format):
    rng = jtu.rand_some_zero(self.rng())
    mat = rng(shape, dtype)
    M = mlir_sparse.MLIRSparse.fromdense(mat, format=format)

    self.assertIsInstance(M.positions, list)
    self.assertIsInstance(M.indices, list)
    self.assertLen(M.positions, mat.ndim)
    self.assertLen(M.indices, mat.ndim)
    for i in range(mat.ndim):
      if format[i] == "D":
        self.assertIsNone(M.positions[i])
        self.assertIsNone(M.indices[i])
      else:
        self.assertIsInstance(M.positions[i], jnp.ndarray)
        self.assertIsInstance(M.indices[i], jnp.ndarray)
    self.assertIsInstance(M.values, jnp.ndarray)
    self.assertEqual(M.values.dtype, mat.dtype)

    self.assertArraysEqual(mat, M.todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_format={}".format(
          jtu.format_shape_dtype_string(shape, dtype), ''.join(format)),
       "shape": shape, "dtype": dtype, "format": format}
      for shape in [(5, 3), (3, 5), (3,), (5,)]
      for dtype in jtu.dtypes.floating
      for format in itertools.product("SD", repeat=len(shape))))
  def test_matvec(self, shape, dtype, format):
    rng = jtu.rand_some_zero(self.rng())
    mat = rng(shape, dtype)
    M = mlir_sparse.MLIRSparse.fromdense(mat, format=format)
    v = rng(shape[-1:], dtype)

    self.assertAllClose(mat @ v, M @ v)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
