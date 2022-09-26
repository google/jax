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

import contextlib
from functools import partial
import itertools
import operator
import random
import unittest
from typing import NamedTuple, Tuple
import warnings

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.random
from jax import config
from jax import dtypes
from jax.experimental import sparse
from jax.experimental.sparse import coo as sparse_coo
from jax.experimental.sparse import bcoo as sparse_bcoo
from jax.experimental.sparse import bcsr as sparse_bcsr
from jax.experimental.sparse.bcoo import BCOOInfo
from jax import lax
from jax._src.lib import xla_extension_version
from jax._src.lib import gpu_sparse
from jax._src.lib import xla_bridge
from jax._src.lib import version as jaxlib_version
from jax._src.util import unzip2
from jax import jit
from jax import tree_util
from jax import vmap
from jax._src import test_util as jtu
from jax._src.lax.lax import remaining, DotDimensionNumbers
from jax.interpreters import mlir
import jax.numpy as jnp
from jax.util import split_list
import numpy as np
import scipy.sparse

config.parse_flags_with_absl()
FLAGS = config.FLAGS

MATMUL_TOL = {
  np.float32: 1E-5,
  np.float64: 1E-10,
  np.complex64: 1e-5,
  np.complex128: 1E-10,
}

GPU_LOWERING_ENABLED = gpu_sparse and (gpu_sparse.cuda_is_supported or
                                       gpu_sparse.rocm_is_supported)

class BcooDotGeneralProperties(NamedTuple):
  lhs_shape: Tuple[int, ...]
  rhs_shape: Tuple[int, ...]
  dtype: np.dtype
  n_batch: int
  n_dense: int
  dimension_numbers: DotDimensionNumbers

  def testcase_name(self):
    return "_{}_{}_nbatch={}_ndense={}_dimension_numbers={}".format(
      jtu.format_shape_dtype_string(self.lhs_shape, self.dtype),
      jtu.format_shape_dtype_string(self.rhs_shape, self.dtype),
      self.n_batch, self.n_dense, self.dimension_numbers)

def _iter_subsets(s):
  return itertools.chain.from_iterable(itertools.combinations(s, n) for n in range(len(s) + 1))

def _generate_bcoo_dot_general_properties(shapes, dtypes) -> BcooDotGeneralProperties:
  """Generator of properties for bcoo_dot_general tests."""
  rng = random.Random(0)

  for shape in shapes:
    for n_batch in range(len(shape) + 1):
      for n_dense in range(len(shape) + 1 - n_batch):
        n_sparse = len(shape) - n_batch - n_dense
        subsets = split_list(range(len(shape)), [n_batch, n_sparse])
        for batch_dims in _iter_subsets(range(n_batch)):
          for contracting_dims in _iter_subsets(remaining(range(n_batch + n_sparse), batch_dims)):
            # We want coverage of permutations & dtypes without generating hundreds of thousands
            # of test cases; we do this by deterministic pseudo-random sampling instead of iterating.
            rhs_permute = rng.sample(range(len(shape)), len(shape))
            lhs_permute = list(itertools.chain.from_iterable(
              rng.sample(subset, len(subset)) for subset in subsets))
            yield BcooDotGeneralProperties(
              lhs_shape=tuple(shape[p] for p in lhs_permute),
              rhs_shape=tuple(shape[p] for p in rhs_permute),
              dtype=rng.choice(dtypes),
              n_batch=n_batch,
              n_dense=n_dense,
              dimension_numbers=(
                ([lhs_permute.index(d) for d in contracting_dims], [rhs_permute.index(d) for d in contracting_dims]),
                ([lhs_permute.index(d) for d in batch_dims], [rhs_permute.index(d) for d in batch_dims])
              ),
            )


all_dtypes = jtu.dtypes.integer + jtu.dtypes.floating + jtu.dtypes.complex


def rand_sparse(rng, nse=0.5, post=lambda x: x, rand_method=jtu.rand_default):
  def _rand_sparse(shape, dtype, nse=nse):
    rand = rand_method(rng)
    size = np.prod(shape).astype(int)
    if 0 <= nse < 1:
      nse = nse * size
    nse = min(size, int(nse))
    M = rand(shape, dtype)
    indices = rng.choice(size, size - nse, replace=False)
    M.flat[indices] = 0
    return post(M)
  return _rand_sparse

def _is_required_cuda_version_satisfied(cuda_version):
  version = xla_bridge.get_backend().platform_version
  if version == "<unknown>" or version.split()[0] == "rocm":
    return False
  else:
    return int(version.split()[-1]) >= cuda_version


class cuSparseTest(jtu.JaxTestCase):
  def gpu_dense_conversion_warning_context(self, dtype):
    if jtu.device_under_test() == "gpu" and np.issubdtype(dtype, np.integer):
      return self.assertWarns(sparse.CuSparseEfficiencyWarning)
    return contextlib.nullcontext()

  def gpu_matmul_warning_context(self, dtype):
    if jtu.device_under_test() == "gpu" and dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
      return self.assertWarns(sparse.CuSparseEfficiencyWarning)
    return contextlib.nullcontext()

  @contextlib.contextmanager
  def assertNoWarnings(self):
    with warnings.catch_warnings(record=True) as caught_warnings:
      yield
    self.assertEmpty(caught_warnings)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}",
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in all_dtypes))
  def test_csr_todense(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.indices, M.indptr)
    todense = lambda *args: sparse.csr_todense(*args, shape=M.shape)

    self.assertArraysEqual(M.toarray(), todense(*args))
    with self.gpu_dense_conversion_warning_context(dtype):
      self.assertArraysEqual(M.toarray(), jit(todense)(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}",
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_csr_todense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    data, indices, indptr = sparse.csr_fromdense(M, nse=(M != 0).sum())
    row, col = sparse.util._csr_to_coo(indices, indptr)
    f = lambda data: sparse.csr_todense(data, indices, indptr, shape=M.shape)

    # Forward-mode
    primals, tangents = jax.jvp(f, [data], [jnp.ones_like(data)])
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(tangents, jnp.zeros_like(M).at[row, col].set(1))

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, data)
    data_out, = vjp_fun(primals)
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(data_out, data)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}",
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_csr_fromdense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    nse = (M != 0).sum()
    f = lambda M: sparse.csr_fromdense(M, nse=nse)

    # Forward-mode
    primals, tangents = jax.jvp(f, [M], [jnp.ones_like(M)])
    self.assertArraysEqual(primals[0], f(M)[0])
    self.assertArraysEqual(primals[1], f(M)[1])
    self.assertArraysEqual(primals[2], f(M)[2])
    self.assertArraysEqual(tangents[0], jnp.ones(nse, dtype=dtype))
    self.assertEqual(tangents[1].dtype, dtypes.float0)
    self.assertEqual(tangents[2].dtype, dtypes.float0)

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, M)
    M_out, = vjp_fun(primals)
    self.assertArraysEqual(primals[0], f(M)[0])
    self.assertArraysEqual(primals[1], f(M)[1])
    self.assertArraysEqual(primals[2], f(M)[2])
    self.assertArraysEqual(M_out, M)

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
        jtu.format_shape_dtype_string(shape, dtype),
        jtu.format_shape_dtype_string(bshape, dtype)),
       "shape": shape, "dtype": dtype, "bshape": bshape}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [(), (1,), (3,)]]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_csr_matmul_ad(self, shape, dtype, bshape):
    csr_matmul = sparse.csr_matvec if len(bshape) == 1 else sparse.csr_matmat
    tol = {np.float32: 1E-5, np.float64: 1E-12, np.complex64: 1E-5, np.complex128: 1E-12}

    rng = rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())

    M = rng(shape, dtype)
    data, indices, indptr = sparse.csr_fromdense(M, nse=(M != 0).sum())
    x = rng_b(bshape, dtype)
    xdot = rng_b(bshape, dtype)

    # Forward-mode with respect to the vector
    f_dense = lambda x: M @ x
    f_sparse = lambda x: csr_matmul(data, indices, indptr, x, shape=M.shape)
    v_sparse, t_sparse = jax.jvp(f_sparse, [x], [xdot])
    v_dense, t_dense = jax.jvp(f_dense, [x], [xdot])
    self.assertAllClose(v_sparse, v_dense, atol=tol, rtol=tol)
    self.assertAllClose(t_sparse, t_dense, atol=tol, rtol=tol)

    # Reverse-mode with respect to the vector
    primals_dense, vjp_dense = jax.vjp(f_dense, x)
    primals_sparse, vjp_sparse = jax.vjp(f_sparse, x)
    out_dense, = vjp_dense(primals_dense)
    out_sparse, = vjp_sparse(primals_sparse)
    self.assertAllClose(primals_dense[0], primals_sparse[0], atol=tol, rtol=tol)
    self.assertAllClose(out_dense, out_sparse, atol=tol, rtol=tol)

    # Forward-mode with respect to nonzero elements of the matrix
    f_sparse = lambda data: csr_matmul(data, indices, indptr, x, shape=M.shape)
    f_dense = lambda data: sparse.csr_todense(data, indices, indptr, shape=M.shape) @ x
    data = rng((len(data),), data.dtype)
    data_dot = rng((len(data),), data.dtype)
    v_sparse, t_sparse = jax.jvp(f_sparse, [data], [data_dot])
    v_dense, t_dense = jax.jvp(f_dense, [data], [data_dot])

    self.assertAllClose(v_sparse, v_dense, atol=tol, rtol=tol)
    self.assertAllClose(t_sparse, t_dense, atol=tol, rtol=tol)

    # Reverse-mode with respect to nonzero elements of the matrix
    primals_dense, vjp_dense = jax.vjp(f_dense, data)
    primals_sparse, vjp_sparse = jax.vjp(f_sparse, data)
    out_dense, = vjp_dense(primals_dense)
    out_sparse, = vjp_sparse(primals_sparse)
    self.assertAllClose(primals_dense[0], primals_sparse[0], atol=tol, rtol=tol)
    self.assertAllClose(out_dense, out_sparse, atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}",
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in all_dtypes))
  def test_csr_fromdense(self, shape, dtype):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_csr = scipy.sparse.csr_matrix(M)

    nse = M_csr.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse.csr_fromdense(M, nse=nse, index_dtype=jnp.int32)

    data, indices, indptr = fromdense(M)
    self.assertArraysEqual(data, M_csr.data.astype(dtype))
    self.assertArraysEqual(indices, M_csr.indices.astype(index_dtype))
    self.assertArraysEqual(indptr, M_csr.indptr.astype(index_dtype))

    with self.gpu_dense_conversion_warning_context(dtype):
      data, indices, indptr = jit(fromdense)(M)
    self.assertArraysEqual(data, M_csr.data.astype(dtype))
    self.assertArraysEqual(indices, M_csr.indices.astype(index_dtype))
    self.assertArraysEqual(indptr, M_csr.indptr.astype(index_dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}_T={transpose}",
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in all_dtypes
      for transpose in [True, False]))
  @jtu.skip_on_devices("rocm")  # will be fixed in rocm-5.1
  def test_csr_matvec(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    v_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.indices, M.indptr, v)
    matvec = lambda *args: sparse.csr_matvec(*args, shape=M.shape, transpose=transpose)

    self.assertAllClose(op(M) @ v, matvec(*args), rtol=MATMUL_TOL)
    with self.gpu_matmul_warning_context(dtype):
      self.assertAllClose(op(M) @ v, jit(matvec)(*args), rtol=MATMUL_TOL)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}_T={transpose}",
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in all_dtypes
      for transpose in [True, False]))
  def test_csr_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.indices, M.indptr, B)
    matmat = lambda *args: sparse.csr_matmat(*args, shape=shape, transpose=transpose)

    self.assertAllClose(op(M) @ B, matmat(*args), rtol=MATMUL_TOL)
    with self.gpu_matmul_warning_context(dtype):
      self.assertAllClose(op(M) @ B, jit(matmat)(*args), rtol=MATMUL_TOL)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}",
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in all_dtypes))
  def test_coo_todense(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.row, M.col)
    todense = lambda *args: sparse_coo._coo_todense(*args, spinfo=sparse_coo.COOInfo(shape=M.shape, rows_sorted=True))

    self.assertArraysEqual(M.toarray(), todense(*args))
    with self.gpu_dense_conversion_warning_context(dtype):
      self.assertArraysEqual(M.toarray(), jit(todense)(*args))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}",
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in all_dtypes))
  def test_coo_fromdense(self, shape, dtype):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_coo = scipy.sparse.coo_matrix(M)

    nse = M_coo.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse_coo._coo_fromdense(M, nse=nse, index_dtype=jnp.int32)

    data, row, col = fromdense(M)
    self.assertArraysEqual(data, M_coo.data.astype(dtype))
    self.assertArraysEqual(row, M_coo.row.astype(index_dtype))
    self.assertArraysEqual(col, M_coo.col.astype(index_dtype))

    with self.gpu_dense_conversion_warning_context(dtype):
      data, indices, indptr = jit(fromdense)(M)
    self.assertArraysEqual(data, M_coo.data.astype(dtype))
    self.assertArraysEqual(row, M_coo.row.astype(index_dtype))
    self.assertArraysEqual(col, M_coo.col.astype(index_dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}_T={transpose}",
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in all_dtypes
      for transpose in [True, False]))
  def test_coo_matvec(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    v_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.row, M.col, v)
    matvec = lambda *args: sparse_coo._coo_matvec(*args, spinfo=sparse_coo.COOInfo(shape=M.shape, rows_sorted=True), transpose=transpose)

    self.assertAllClose(op(M) @ v, matvec(*args), rtol=MATMUL_TOL)
    with self.gpu_matmul_warning_context(dtype):
      self.assertAllClose(op(M) @ v, jit(matvec)(*args), rtol=MATMUL_TOL)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}_T={transpose}",
       "shape": shape, "dtype": dtype, "transpose": transpose}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in all_dtypes
      for transpose in [True, False]))
  @jtu.skip_on_devices("rocm")  # will be fixed in rocm-5.1
  def test_coo_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.row, M.col, B)
    matmat = lambda *args: sparse_coo._coo_matmat(*args, spinfo=sparse_coo.COOInfo(shape=shape, rows_sorted=True), transpose=transpose)

    self.assertAllClose(op(M) @ B, matmat(*args), rtol=MATMUL_TOL)
    with self.gpu_matmul_warning_context(dtype):
      self.assertAllClose(op(M) @ B, jit(matmat)(*args), rtol=MATMUL_TOL)

  def test_coo_matmat_layout(self):
    # Regression test for https://github.com/google/jax/issues/7533
    d = jnp.array([1.0, 2.0, 3.0, 4.0])
    i = jnp.array([0, 0, 1, 2])
    j = jnp.array([0, 2, 0, 0])
    shape = (3, 3)

    x = jnp.arange(9).reshape(3, 3).astype(d.dtype)

    def f(x):
      return sparse_coo._coo_matmat(d, i, j, x.T, spinfo=sparse_coo.COOInfo(shape=shape, rows_sorted=True))

    result = f(x)
    result_jit = jit(f)(x)

    self.assertAllClose(result, result_jit)

  def test_coo_sorted_indices(self):
    rng = self.rng()
    sprng = rand_sparse(rng)

    mat = sparse.COO.fromdense(sprng((5, 6), np.float32))
    perm = rng.permutation(mat.nse)
    mat_unsorted = sparse.COO((mat.data[perm], mat.row[perm], mat.col[perm]), shape=mat.shape)
    mat_resorted = mat_unsorted._sort_indices()
    self.assertArraysEqual(mat.todense(), mat_resorted.todense())

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @jtu.skip_on_devices("rocm")  # TODO(rocm): see SWDEV-328107
  def test_coo_sorted_indices_gpu_lowerings(self):
    dtype = jnp.float32

    mat = jnp.arange(12, dtype=dtype).reshape(4, 3)

    mat_rows_sorted = sparse.COO.fromdense(mat)
    self.assertTrue(mat_rows_sorted._rows_sorted)
    self.assertFalse(mat_rows_sorted._cols_sorted)

    mat_cols_sorted = sparse.COO.fromdense(mat.T).T
    self.assertFalse(mat_cols_sorted._rows_sorted)
    self.assertTrue(mat_cols_sorted._cols_sorted)

    mat_unsorted = sparse.COO(mat_rows_sorted._bufs, shape=mat_rows_sorted.shape)
    self.assertFalse(mat_unsorted._rows_sorted)
    self.assertFalse(mat_unsorted._cols_sorted)

    self.assertArraysEqual(mat, mat_rows_sorted._sort_indices().todense())
    self.assertArraysEqual(mat, mat_cols_sorted._sort_indices().todense())
    self.assertArraysEqual(mat, mat_unsorted._sort_indices().todense())

    todense = jit(sparse.coo_todense)
    with self.assertNoWarnings():
      dense_rows_sorted = todense(mat_rows_sorted)
      dense_cols_sorted = todense(mat_cols_sorted)
      dense_unsorted = todense(mat_unsorted._sort_indices())
    with self.assertWarnsRegex(sparse.CuSparseEfficiencyWarning, "coo_todense GPU lowering requires matrices with sorted rows.*"):
      dense_unsorted_fallback = todense(mat_unsorted)
    self.assertArraysEqual(mat, dense_rows_sorted)
    self.assertArraysEqual(mat, dense_cols_sorted)
    self.assertArraysEqual(mat, dense_unsorted)
    self.assertArraysEqual(mat, dense_unsorted_fallback)

    rhs_vec = jnp.arange(3, dtype=dtype)
    matvec = jit(sparse.coo_matvec)
    matvec_expected = mat @ rhs_vec
    with self.assertNoWarnings():
      matvec_rows_sorted = matvec(mat_rows_sorted, rhs_vec)
      matvec_cols_sorted = matvec(mat_cols_sorted, rhs_vec)
      matvec_unsorted = matvec(mat_unsorted._sort_indices(), rhs_vec)
    with self.assertWarnsRegex(sparse.CuSparseEfficiencyWarning, "coo_matvec GPU lowering requires matrices with sorted rows.*"):
      matvec_unsorted_fallback = matvec(mat_unsorted, rhs_vec)
    self.assertArraysEqual(matvec_expected, matvec_rows_sorted)
    self.assertArraysEqual(matvec_expected, matvec_cols_sorted)
    self.assertArraysEqual(matvec_expected, matvec_unsorted)
    self.assertArraysEqual(matvec_expected, matvec_unsorted_fallback)

    rhs_mat = jnp.arange(6, dtype=dtype).reshape(3, 2)
    matmat = jit(sparse.coo_matmat)
    matmat_expected = mat @ rhs_mat
    with self.assertNoWarnings():
      matmat_rows_sorted = matmat(mat_rows_sorted, rhs_mat)
      matmat_cols_sorted = matmat(mat_cols_sorted, rhs_mat)
      matmat_unsorted = matmat(mat_unsorted._sort_indices(), rhs_mat)
    with self.assertWarnsRegex(sparse.CuSparseEfficiencyWarning, "coo_matmat GPU lowering requires matrices with sorted rows.*"):
      matmat_unsorted_fallback = matmat(mat_unsorted, rhs_mat)
    self.assertArraysEqual(matmat_expected, matmat_rows_sorted)
    self.assertArraysEqual(matmat_expected, matmat_cols_sorted)
    self.assertArraysEqual(matmat_expected, matmat_unsorted)
    self.assertArraysEqual(matmat_expected, matmat_unsorted_fallback)

  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  def test_gpu_translation_rule(self):
    version = xla_bridge.get_backend().platform_version
    if version.split()[0] != "rocm":
      cuda_version = None if version == "<unknown>" else int(
          version.split()[-1])
      if cuda_version is None or cuda_version < 11000:
        self.assertFalse(gpu_sparse and gpu_sparse.cuda_is_supported)
        self.assertNotIn(sparse.csr_todense_p,
                         mlir._platform_specific_lowerings["cuda"])
      else:
        self.assertTrue(gpu_sparse and gpu_sparse.cuda_is_supported)
        self.assertIn(sparse.csr_todense_p,
                      mlir._platform_specific_lowerings["cuda"])
    else:
      self.assertTrue(gpu_sparse and gpu_sparse.rocm_is_supported)
      self.assertIn(sparse.csr_todense_p,
                    mlir._platform_specific_lowerings["rocm"])

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
         jtu.format_shape_dtype_string(shape, dtype), mat_type),
       "shape": shape, "dtype": dtype, "mat_type": mat_type}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for mat_type in ['csr', 'coo']))
  def test_extra_nse(self, shape, dtype, mat_type):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = (M != 0).sum() + 5
    fromdense = getattr(sparse, f"{mat_type}_fromdense")
    todense = getattr(sparse, f"{mat_type}_todense")
    args = fromdense(M, nse=nse, index_dtype=jnp.int32)
    if mat_type == 'coo':
      M_out = todense(args)
    else:
      M_out = todense(*args, shape=M.shape)
    self.assertArraysEqual(M, M_out)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}",
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_coo_todense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    data, row, col = sparse_coo._coo_fromdense(M, nse=(M != 0).sum())
    f = lambda data: sparse_coo._coo_todense(data, row, col, spinfo=sparse_coo.COOInfo(shape=M.shape, rows_sorted=True))

    # Forward-mode
    primals, tangents = jax.jvp(f, [data], [jnp.ones_like(data)])
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(tangents, jnp.zeros_like(M).at[row, col].set(1))

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, data)
    data_out, = vjp_fun(primals)
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(data_out, data)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_{jtu.format_shape_dtype_string(shape, dtype)}",
       "shape": shape, "dtype": dtype}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_coo_fromdense_ad(self, shape, dtype):
    rng = rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    nse = (M != 0).sum()
    f = lambda M: sparse_coo._coo_fromdense(M, nse=nse)

    # Forward-mode
    primals, tangents = jax.jvp(f, [M], [jnp.ones_like(M)])
    self.assertArraysEqual(primals[0], f(M)[0])
    self.assertArraysEqual(primals[1], f(M)[1])
    self.assertArraysEqual(primals[2], f(M)[2])
    self.assertArraysEqual(tangents[0], jnp.ones(nse, dtype=dtype))
    self.assertEqual(tangents[1].dtype, dtypes.float0)
    self.assertEqual(tangents[2].dtype, dtypes.float0)

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, M)
    M_out, = vjp_fun(primals)
    self.assertArraysEqual(primals[0], f(M)[0])
    self.assertArraysEqual(primals[1], f(M)[1])
    self.assertArraysEqual(primals[2], f(M)[2])
    self.assertArraysEqual(M_out, M)

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
        jtu.format_shape_dtype_string(shape, dtype),
        jtu.format_shape_dtype_string(bshape, dtype)),
       "shape": shape, "dtype": dtype, "bshape": bshape}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [(), (1,), (3,)]]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_coo_matmul_ad(self, shape, dtype, bshape):
    coo_matmul = sparse_coo._coo_matvec if len(bshape) == 1 else sparse_coo._coo_matmat
    tol = {np.float32: 1E-5, np.float64: 1E-12, np.complex64: 1E-5, np.complex128: 1E-12}

    rng = rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())

    M = rng(shape, dtype)
    data, row, col = sparse_coo._coo_fromdense(M, nse=(M != 0).sum())
    x = rng_b(bshape, dtype)
    xdot = rng_b(bshape, dtype)

    # Forward-mode with respect to the vector
    f_dense = lambda x: M @ x
    f_sparse = lambda x: coo_matmul(data, row, col, x, spinfo=sparse_coo.COOInfo(shape=M.shape))
    v_sparse, t_sparse = jax.jvp(f_sparse, [x], [xdot])
    v_dense, t_dense = jax.jvp(f_dense, [x], [xdot])
    self.assertAllClose(v_sparse, v_dense, atol=tol, rtol=tol)
    self.assertAllClose(t_sparse, t_dense, atol=tol, rtol=tol)

    # Reverse-mode with respect to the vector
    primals_dense, vjp_dense = jax.vjp(f_dense, x)
    primals_sparse, vjp_sparse = jax.vjp(f_sparse, x)
    out_dense, = vjp_dense(primals_dense)
    out_sparse, = vjp_sparse(primals_sparse)
    self.assertAllClose(primals_dense[0], primals_sparse[0], atol=tol, rtol=tol)
    self.assertAllClose(out_dense, out_sparse, atol=tol, rtol=tol)

    # Forward-mode with respect to nonzero elements of the matrix
    f_sparse = lambda data: coo_matmul(data, row, col, x, spinfo=sparse_coo.COOInfo(shape=M.shape))
    f_dense = lambda data: sparse_coo._coo_todense(data, row, col, spinfo=sparse_coo.COOInfo(shape=M.shape)) @ x
    data = rng((len(data),), data.dtype)
    data_dot = rng((len(data),), data.dtype)
    v_sparse, t_sparse = jax.jvp(f_sparse, [data], [data_dot])
    v_dense, t_dense = jax.jvp(f_dense, [data], [data_dot])

    self.assertAllClose(v_sparse, v_dense, atol=tol, rtol=tol)
    self.assertAllClose(t_sparse, t_dense, atol=tol, rtol=tol)

    # Reverse-mode with respect to nonzero elements of the matrix
    primals_dense, vjp_dense = jax.vjp(f_dense, data)
    primals_sparse, vjp_sparse = jax.vjp(f_sparse, data)
    out_dense, = vjp_dense(primals_dense)
    out_sparse, = vjp_sparse(primals_sparse)
    self.assertAllClose(primals_dense[0], primals_sparse[0], atol=tol, rtol=tol)
    self.assertAllClose(out_dense, out_sparse, atol=tol, rtol=tol)


class BCOOTest(jtu.JaxTestCase):

  def test_repr(self):
    x = sparse.BCOO.fromdense(jnp.arange(5, dtype='float32'))
    self.assertEqual(repr(x), "BCOO(float32[5], nse=4)")

    y = sparse.BCOO.fromdense(jnp.arange(6, dtype='float32').reshape(2, 3), n_batch=1)
    self.assertEqual(repr(y), "BCOO(float32[2, 3], nse=3, n_batch=1)")

    y = sparse.BCOO.fromdense(jnp.arange(6, dtype='float32').reshape(2, 3), n_batch=1, n_dense=1)
    self.assertEqual(repr(y), "BCOO(float32[2, 3], nse=1, n_batch=1, n_dense=1)")

    M_invalid = sparse.BCOO(([], []), shape=(100,))
    self.assertEqual(repr(M_invalid), "BCOO(<invalid>)")

    @jit
    def f(x):
      self.assertEqual(repr(x), "DynamicJaxprTracer[BCOO(float32[5], nse=4)]")
    f(x)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in all_dtypes
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_empty(self, shape, dtype, n_batch, n_dense):
    M = sparse.empty(shape, dtype=dtype, n_batch=n_batch, n_dense=n_dense)
    self.assertIsInstance(M, sparse.BCOO)
    self.assertEqual(M.nse, 0)
    self.assertEqual(M.n_batch, n_batch)
    self.assertEqual(M.n_dense, n_dense)
    self.assertEqual(M.dtype, dtype)
    self.assertArraysEqual(M.todense(), jnp.empty(shape, dtype))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_k={}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string((N, M), dtype), k, n_batch, n_dense),
       "N": N, "M": M, "k": k, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for N in [3, 5]
      for M in [None, 4]
      for k in [-3, -1, 0, 2, 4]
      for dtype in all_dtypes
      for n_batch in range(3)
      for n_dense in range(3 - n_batch)))
  def test_eye(self, N, M, k, dtype, n_batch, n_dense):
    mat = sparse.eye(N, M, k, dtype=dtype, n_batch=n_batch, n_dense=n_dense)
    expected = jnp.eye(N, M, k, dtype=dtype)
    expected_nse = sparse.BCOO.fromdense(expected, n_batch=n_batch, n_dense=n_dense).nse

    self.assertIsInstance(mat, sparse.BCOO)
    self.assertEqual(mat.n_batch, n_batch)
    self.assertEqual(mat.n_dense, n_dense)
    self.assertEqual(mat.dtype, dtype)
    self.assertEqual(mat.nse, expected_nse)
    self.assertArraysEqual(mat.todense(), expected)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in all_dtypes
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_dense_round_trip(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    n_sparse = M.ndim - n_batch - n_dense
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)
    data_jit, indices_jit = jit(partial(sparse_bcoo._bcoo_fromdense, nse=nse, n_batch=n_batch, n_dense=n_dense))(M)
    self.assertArraysEqual(data, data_jit)
    self.assertArraysEqual(indices, indices_jit)

    assert data.dtype == dtype
    assert data.shape == shape[:n_batch] + (nse,) + shape[n_batch + n_sparse:]
    assert indices.dtype == jnp.int32  # TODO: test passing this arg
    assert indices.shape == shape[:n_batch] + (nse, n_sparse)

    todense = partial(sparse_bcoo._bcoo_todense, spinfo=BCOOInfo(shape))
    self.assertArraysEqual(M, todense(data, indices))
    self.assertArraysEqual(M, jit(todense)(data, indices))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_todense_ad(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)

    todense = partial(sparse_bcoo._bcoo_todense, indices=indices, spinfo=BCOOInfo(shape))
    j1 = jax.jacfwd(todense)(data)
    j2 = jax.jacrev(todense)(data)
    hess = jax.hessian(todense)(data)
    self.assertArraysAllClose(j1, j2)
    self.assertEqual(j1.shape, M.shape + data.shape)
    self.assertEqual(hess.shape, M.shape + 2 * data.shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_fromdense_ad(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)

    def fromdense(M):
      return sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)[0]
    data = fromdense(M)

    j1 = jax.jacfwd(fromdense)(M)
    j2 = jax.jacrev(fromdense)(M)
    hess = jax.hessian(fromdense)(M)
    self.assertArraysAllClose(j1, j2)
    self.assertEqual(j1.shape, data.shape + M.shape)
    self.assertEqual(hess.shape, data.shape + 2 * M.shape)

  def test_bcoo_fromdense_sorted_and_unique_indices(self):
    rng = self.rng()
    rng_sparse = rand_sparse(rng)
    mat = sparse.BCOO.fromdense(rng_sparse((5, 6), np.float32))
    perm = rng.permutation(mat.nse)
    mat_unsorted = sparse.BCOO((mat.data[perm], mat.indices[perm]),
                               shape=mat.shape,
                               unique_indices=mat.unique_indices)
    mat_resorted = mat_unsorted.sort_indices()
    with self.subTest('sorted indices'):
      self.assertArraysEqual(mat.indices, mat_resorted.indices)
      self.assertArraysEqual(mat.data, mat_resorted.data)

    with self.subTest('unique indices'):
      self.assertTrue(mat.unique_indices)
      self.assertTrue(mat_unsorted.unique_indices)
      self.assertTrue(mat_resorted.unique_indices)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_dense_round_trip_batched(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    n_sparse = M.ndim - n_batch - n_dense
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)

    fromdense = partial(sparse_bcoo._bcoo_fromdense, nse=nse, n_dense=n_dense)
    todense = partial(sparse_bcoo._bcoo_todense, spinfo=BCOOInfo(shape[n_batch:]))
    for i in range(n_batch):
      fromdense = jax.vmap(fromdense)
      todense = jax.vmap(todense)

    data, indices = fromdense(M)

    assert data.dtype == dtype
    assert data.shape == shape[:n_batch] + (nse,) + shape[n_batch + n_sparse:]
    assert indices.dtype == jnp.int32  # TODO: test passing this arg
    assert indices.shape == shape[:n_batch] + (nse, n_sparse)

    self.assertArraysEqual(M, todense(data, indices))
    self.assertArraysEqual(M, jit(todense)(data, indices))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_extract(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse)
    data2 = sparse.bcoo_extract(indices, M)
    self.assertArraysEqual(data, data2)
    data3 = jit(sparse.bcoo_extract)(indices, M)
    self.assertArraysEqual(data, data3)

  def test_bcoo_extract_batching(self):
    # https://github.com/google/jax/issues/9431
    indices = jnp.zeros((4, 1, 1), dtype=int)
    mat = jnp.arange(4.).reshape((4, 1))

    # in_axes = (0, None)
    expected = jnp.vstack([sparse.bcoo_extract(i, mat[0]) for i in indices])
    actual = vmap(sparse.bcoo_extract, in_axes=(0, None))(indices, mat[0])
    self.assertArraysEqual(expected, actual)

    # in_axes = (None, 0)
    expected = jnp.vstack([sparse.bcoo_extract(indices[0], m) for m in mat])
    actual = vmap(sparse.bcoo_extract, in_axes=(None, 0))(indices[0], mat)
    self.assertArraysEqual(expected, actual)

    # in_axes = (0, 0)
    expected = jnp.vstack([sparse.bcoo_extract(i, m) for i, m in zip(indices, mat)])
    actual = vmap(sparse.bcoo_extract, in_axes=0)(indices, mat)
    self.assertArraysEqual(expected, actual)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_extract_ad(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)

    extract = partial(sparse.bcoo_extract, indices)
    j1 = jax.jacfwd(extract)(M)
    j2 = jax.jacrev(extract)(M)
    hess = jax.hessian(extract)(M)
    self.assertArraysAllClose(j1, j2)
    self.assertEqual(j1.shape, data.shape + M.shape)
    self.assertEqual(hess.shape, data.shape + 2 * M.shape)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_transpose(self, shape, dtype, n_batch, n_dense):
    n_sparse = len(shape) - n_batch - n_dense
    rng = self.rng()
    sprng = rand_sparse(rng)
    M = sprng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)

    permutation = np.concatenate([
      rng.permutation(range(n_batch)),
      rng.permutation(range(n_batch, n_batch + n_sparse)),
      rng.permutation(range(n_batch + n_sparse, len(shape)))]).astype(int)

    M_T = M.transpose(permutation)
    trans = partial(sparse_bcoo._bcoo_transpose, spinfo=BCOOInfo(shape), permutation=permutation)
    self.assertArraysEqual(M_T, sparse_bcoo._bcoo_todense(*trans(data, indices), spinfo=BCOOInfo(M_T.shape)))
    self.assertArraysEqual(M_T, sparse_bcoo._bcoo_todense(*jit(trans)(data, indices), spinfo=BCOOInfo(M_T.shape)))

    # test batched
    def trans(M):
      return M.transpose([p - n_batch for p in permutation[n_batch:]])
    for _ in range(n_batch):
      trans = jax.vmap(trans)
    Msp = sparse.BCOO.fromdense(M, n_batch=n_batch, n_dense=n_dense)
    self.assertArraysEqual(trans(M), trans(Msp).todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_slice(self, shape, dtype, n_batch, n_dense):
    rng = self.rng()
    sprng = rand_sparse(rng)
    M = sprng(shape, dtype)
    Msp = sparse.BCOO.fromdense(M, n_batch=n_batch, n_dense=n_dense)

    rng = self.rng()
    slices = rng.randint(0, M.shape, (2, M.ndim)).T
    slices.sort(1)
    start_indices, limit_indices = unzip2(slices)
    strides = None  # strides currently not implemented
    kwds = dict(start_indices=start_indices, limit_indices=limit_indices, strides=strides)

    dense_result = lax.slice(M, **kwds)
    sparse_result = sparse.bcoo_slice(Msp, **kwds)
    sparse_result_jit = jax.jit(partial(sparse.bcoo_slice, **kwds))(Msp)

    self.assertArraysEqual(dense_result, sparse_result.todense())
    self.assertArraysEqual(dense_result, sparse_result_jit.todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_dynamic_slice(self, shape, dtype, n_batch, n_dense):
    rng = self.rng()
    sprng = rand_sparse(rng)
    M = sprng(shape, dtype)
    Msp = sparse.BCOO.fromdense(M, n_batch=n_batch, n_dense=n_dense)

    rng = self.rng()
    # Note: test out-of-range start indices
    start_indices = rng.randint(-max(M.shape), max(M.shape), M.ndim)
    slice_sizes = rng.randint(0, M.shape, M.ndim)
    kwds = dict(start_indices=start_indices, slice_sizes=slice_sizes)

    dense_result = lax.dynamic_slice(M, **kwds)
    sparse_result = sparse.bcoo_dynamic_slice(Msp, **kwds)
    sparse_result_jit = partial(sparse.bcoo_dynamic_slice, slice_sizes=slice_sizes)(Msp, start_indices)

    self.assertArraysEqual(dense_result, sparse_result.todense())
    self.assertArraysEqual(dense_result, sparse_result_jit.todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}_idx={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense, idx),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense,
       "idx": idx}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)
      for idx in [1, slice(1, 3)]))
  def test_bcoo_getitem(self, shape, dtype, n_batch, n_dense, idx):
    # Note: __getitem__ is currently only supported for simple slices and indexing
    rng = self.rng()
    sprng = rand_sparse(rng)
    M = sprng(shape, dtype)
    Msp = sparse.BCOO.fromdense(M, n_batch=n_batch, n_dense=n_dense)
    self.assertArraysEqual(M[idx], Msp[idx].todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_transpose_ad(self, shape, dtype, n_batch, n_dense):
    n_sparse = len(shape) - n_batch - n_dense
    rng = self.rng()
    sprng = rand_sparse(self.rng())

    M = sprng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)

    permutation = np.concatenate([
      rng.permutation(range(n_batch)),
      rng.permutation(range(n_batch, n_batch + n_sparse)),
      rng.permutation(range(n_batch + n_sparse, len(shape)))]).astype(int)

    def f_sparse(data):
      return sparse_bcoo._bcoo_transpose(data, indices, spinfo=BCOOInfo(shape), permutation=permutation)[0]

    jf_sparse = jax.jacfwd(f_sparse)(data)
    jr_sparse = jax.jacrev(f_sparse)(data)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    # TODO(jakevdp) also test against dense version?
    self.assertAllClose(jf_sparse, jr_sparse, rtol=tol)

  def test_bcoo_transpose_indices_sorted(self):
    rng = self.rng()
    rng_sparse = rand_sparse(rng)
    n_batch, n_dense = 2, 2
    shape = (2, 3, 4, 5, 6, 7, 8)
    mat = sparse.BCOO.fromdense(rng_sparse(shape, np.float32),
                                n_dense=n_dense, n_batch=n_batch)

    permutations = (1, 0, 2, 3, 4, 6, 5)
    mat_T_indices_sorted = mat.transpose(axes=permutations)
    self.assertTrue(mat_T_indices_sorted.indices_sorted)

    permutations = (0, 1, 3, 2, 4, 5, 6)
    mat_T_indices_unsorted = mat.transpose(axes=permutations)
    self.assertFalse(mat_T_indices_unsorted.indices_sorted)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(1, len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_todense_partial_batch(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)

    M1 = sparse_bcoo._bcoo_todense(data, indices[:1], spinfo=BCOOInfo(M.shape))
    M2 = sparse_bcoo._bcoo_todense(data, jnp.stack(shape[0] * [indices[0]]), spinfo=BCOOInfo(M.shape))
    self.assertAllClose(M1, M2)

    M3 = sparse_bcoo._bcoo_todense(data[:1], indices, spinfo=BCOOInfo(M.shape))
    M4 = sparse_bcoo._bcoo_todense(jnp.stack(shape[0] * [data[0]]), indices, spinfo=BCOOInfo(M.shape))
    self.assertAllClose(M3, M4)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": props.testcase_name(), "props": props}
      for props in _generate_bcoo_dot_general_properties(
        shapes=[(5,), (2, 3), (2, 3, 4), (2, 3, 4, 4)],
        dtypes=jtu.dtypes.floating + jtu.dtypes.complex,
      )))
  def test_bcoo_dot_general(self, props: BcooDotGeneralProperties):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())

    def args_maker():
      lhs = rng_sparse(props.lhs_shape, props.dtype)
      rhs = rng(props.rhs_shape, props.dtype)
      nse = sparse.util._count_stored_elements(lhs, n_batch=props.n_batch,
                                               n_dense=props.n_dense)
      data, indices = sparse_bcoo._bcoo_fromdense(lhs, nse=nse, n_batch=props.n_batch, n_dense=props.n_dense)
      return data, indices, lhs, rhs

    def f_dense(data, indices, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=props.dimension_numbers)

    def f_sparse(data, indices, lhs, rhs):
      return sparse_bcoo._bcoo_dot_general(data, indices, rhs, lhs_spinfo=BCOOInfo(lhs.shape),
                                           dimension_numbers=props.dimension_numbers)

    tol = {'float32': 3E-2} if jtu.device_under_test() == 'tpu' else {}
    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker, tol=tol)
    self._CheckAgainstNumpy(f_dense, jit(f_sparse), args_maker, tol=tol)
    # TODO(jakevdp): In rare cases, this fails python_should_be_executing check. Why?
    # self._CompileAndCheck(f_sparse, args_maker)

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_lhs_contracting={}_rhs_contracting={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               lhs_contracting, rhs_contracting),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "lhs_contracting": lhs_contracting, "rhs_contracting": rhs_contracting}
      for lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [(5,), (5,), [0], [0]],
          [(5,), (5, 7), [0], [0]],
          [(5,), (7, 5), [0], [1]],
          [(5, 7), (5,), [0], [0]],
          [(7, 5), (5,), [1], [0]],
          [(3, 5), (2, 5), [1], [1]],
          [(3, 5), (5, 2), [1], [0]],
          [(5, 3), (2, 5), [0], [1]],
          [(5, 3), (5, 2), [0], [0]],
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  @jtu.skip_on_devices("rocm")
  def test_bcoo_dot_general_cusparse(
    self, lhs_shape, rhs_shape, dtype, lhs_contracting, rhs_contracting):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    def args_maker():
      lhs = rng_sparse(lhs_shape, dtype)
      rhs = rng(rhs_shape, dtype)
      nse = sparse.util._count_stored_elements(lhs, n_batch=0, n_dense=0)
      lhs_bcoo = sparse_bcoo.bcoo_fromdense(lhs, nse=nse, index_dtype=jnp.int32)
      return lhs_bcoo, lhs, rhs

    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))

    def f_dense(lhs_bcoo, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    def f_sparse(lhs_bcoo, lhs, rhs):
      return sparse_bcoo.bcoo_dot_general(lhs_bcoo, rhs,
                                          dimension_numbers=dimension_numbers)

    self._CompileAndCheck(f_sparse, args_maker)
    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker)

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_n_batch={}_lhs_shape={}_rhs_shape={}_lhs_contracting={}_rhs_contracting={}"
       .format(n_batch, jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               lhs_contracting, rhs_contracting),
       "n_batch": n_batch, "lhs_shape": lhs_shape, "rhs_shape": rhs_shape,
       "dtype": dtype, "lhs_contracting": lhs_contracting,
       "rhs_contracting": rhs_contracting}
      for n_batch, lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [1, (1, 2, 3), (3, 2), [2], [0]],
          [1, (1, 3, 2), (3, 2), [1], [0]],
          [1, (1, 3, 2), (4, 3), [1], [1]],
          [1, (4, 2, 3), (3, 5), [2], [0]],
          [1, (4, 2, 3), (2, 5), [1], [0]],
          [1, (4, 2, 3), (5, 3), [2], [1]],
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  @jtu.skip_on_devices("rocm")
  def test_bcoo_batched_matmat_cusparse(
    self, n_batch, lhs_shape, rhs_shape, dtype, lhs_contracting,
    rhs_contracting):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    def args_maker():
      lhs = rng_sparse(lhs_shape, dtype)
      rhs = rng(rhs_shape, dtype)
      nse = sparse.util._count_stored_elements(lhs, n_batch=n_batch,
                                               n_dense=0)
      lhs_bcoo = sparse_bcoo.bcoo_fromdense(lhs, n_batch=n_batch, nse=nse,
                                            index_dtype=jnp.int32)
      return lhs_bcoo, lhs, rhs

    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))

    def f_dense(lhs_bcoo, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    def f_sparse(lhs_bcoo, lhs, rhs):
      return sparse_bcoo.bcoo_dot_general(lhs_bcoo, rhs,
                                          dimension_numbers=dimension_numbers)

    cuda_version_11061_and_beyond = _is_required_cuda_version_satisfied(
        cuda_version=11061)
    jaxlib_version_0318_and_beyond = jaxlib_version >= (0, 3, 18)
    if cuda_version_11061_and_beyond and jaxlib_version_0318_and_beyond:
      # TODO(tianjianlu): In some cases, this fails python_should_be_executing.
      # self._CompileAndCheck(f_sparse, args_maker)
      self._CheckAgainstNumpy(f_dense, f_sparse, args_maker)
      self._CheckAgainstNumpy(f_dense, jit(f_sparse), args_maker)
    else:
      lhs_bcoo, lhs, rhs = args_maker()
      matmat_expected = f_dense(lhs_bcoo, lhs, rhs)
      with self.assertWarnsRegex(
          sparse.CuSparseEfficiencyWarning,
          "bcoo_dot_general GPU lowering currently does not support this "
          "batch-mode computation.*"):
        matmat_default_lowering_fallback = jit(f_sparse)(lhs_bcoo, lhs, rhs)
      self.assertAllClose(matmat_expected, matmat_default_lowering_fallback,
                          atol=1E-6, rtol=1E-6)

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_n_batch={}_lhs_shape={}_rhs_shape={}_lhs_contracting={}_rhs_contracting={}"
       .format(n_batch, jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               lhs_contracting, rhs_contracting),
       "n_batch": n_batch, "lhs_shape": lhs_shape, "rhs_shape": rhs_shape,
       "dtype": dtype, "lhs_contracting": lhs_contracting,
       "rhs_contracting": rhs_contracting}
      for n_batch, lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [1, (1, 2, 3), (3), [2], [0]],
          [1, (1, 2), (3, 2), [1], [1]],
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  @jtu.skip_on_devices("rocm")
  def test_bcoo_batched_matmat_default_lowering(
    self, n_batch, lhs_shape, rhs_shape, dtype, lhs_contracting,
    rhs_contracting):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())
    lhs = rng_sparse(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    nse = sparse.util._count_stored_elements(lhs, n_batch=n_batch,
                                             n_dense=0)
    lhs_bcoo = sparse_bcoo.bcoo_fromdense(lhs, n_batch=n_batch, nse=nse,
                                            index_dtype=jnp.int32)
    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))
    matmat_expected = lax.dot_general(lhs, rhs,
                                      dimension_numbers=dimension_numbers)
    sp_matmat = jit(partial(sparse_bcoo.bcoo_dot_general,
                            dimension_numbers=dimension_numbers))

    if config.jax_bcoo_cusparse_lowering:
      with self.assertWarnsRegex(
          sparse.CuSparseEfficiencyWarning,
          "bcoo_dot_general GPU lowering currently does not support this "
          "batch-mode computation.*"):
        matmat_default_lowering_fallback = sp_matmat(lhs_bcoo, rhs)

    self.assertArraysEqual(matmat_expected, matmat_default_lowering_fallback)

  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/hipsparse")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @jtu.skip_on_devices("rocm")  # TODO(rocm): see SWDEV-328107
  def test_bcoo_dot_general_oob_and_unsorted_indices_cusparse(self):
    """Tests bcoo dot general with out-of-bound and unsorted indices."""

    rhs = jnp.ones((5, 3), dtype=jnp.float32)

    # It creates out-of-bound indices when nse > nnz.
    lhs_mat_dense = jnp.array([[1, 0, 2, 3, 0], [0, 0, 0, 4, 0]],
                              dtype=jnp.float32)
    lhs_mat_bcoo = sparse.BCOO.fromdense(lhs_mat_dense, nse=7)
    rng = self.rng()
    perm = rng.permutation(lhs_mat_bcoo.nse)
    lhs_mat_bcoo_unsorted = sparse.BCOO(
        (lhs_mat_bcoo.data[perm], lhs_mat_bcoo.indices[perm]),
        shape=lhs_mat_dense.shape)

    dimension_numbers_2d = (([1], [0]), ([], []))
    sp_matmat = jit(partial(sparse_bcoo.bcoo_dot_general,
                            dimension_numbers=dimension_numbers_2d))

    matmat_expected = lax.dot_general(lhs_mat_dense, rhs,
                                      dimension_numbers=dimension_numbers_2d)
    if config.jax_bcoo_cusparse_lowering:
      with self.assertWarnsRegex(
          sparse.CuSparseEfficiencyWarning,
          "bcoo_dot_general GPU lowering requires matrices with sorted indices*"):
        matmat_unsorted_fallback = sp_matmat(lhs_mat_bcoo_unsorted, rhs)

      with self.subTest(msg="2D"):
        self.assertArraysEqual(matmat_expected, matmat_unsorted_fallback)

    lhs_vec_dense = jnp.array([0, 1, 0, 2, 0], dtype=jnp.float32)
    lhs_vec_bcoo = sparse.BCOO.fromdense(lhs_vec_dense, nse=5)
    rng = self.rng()
    perm = rng.permutation(lhs_vec_bcoo.nse)
    lhs_vec_bcoo_unsorted = sparse.BCOO(
        (lhs_vec_bcoo.data[perm], lhs_vec_bcoo.indices[perm]),
        shape=lhs_vec_dense.shape, indices_sorted=False)

    dimension_numbers_1d = (([0], [0]), ([], []))
    sp_vecmat = jit(partial(sparse_bcoo.bcoo_dot_general,
                            dimension_numbers=dimension_numbers_1d))

    vecmat_expected = lax.dot_general(lhs_vec_dense, rhs,
                                      dimension_numbers=dimension_numbers_1d)

    if config.jax_bcoo_cusparse_lowering:
      with self.assertWarnsRegex(
          sparse.CuSparseEfficiencyWarning,
          "bcoo_dot_general GPU lowering requires matrices with sorted indices*"):
        vecmat_unsorted_fallback = sp_vecmat(lhs_vec_bcoo_unsorted, rhs)

      with self.subTest(msg="1D"):
        self.assertArraysEqual(vecmat_expected, vecmat_unsorted_fallback)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": props.testcase_name(), "props": props}
      for props in _generate_bcoo_dot_general_properties(
        shapes=[(5,), (2, 3), (2, 3, 4), (2, 3, 4, 4)],
        dtypes=jtu.dtypes.floating + jtu.dtypes.complex,
      )))
  def test_bcoo_rdot_general(self, props: BcooDotGeneralProperties):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())

    lhs_shape, rhs_shape = props.rhs_shape, props.lhs_shape
    dimension_numbers = tuple(d[::-1] for d in props.dimension_numbers)

    def args_maker():
      lhs = rng_sparse(lhs_shape, props.dtype)
      rhs = rng(rhs_shape, props.dtype)
      nse = sparse.util._count_stored_elements(rhs, n_batch=props.n_batch,
                                               n_dense=props.n_dense)
      data, indices = sparse_bcoo._bcoo_fromdense(
          rhs, nse=nse, n_batch=props.n_batch, n_dense=props.n_dense)
      return data, indices, lhs, rhs

    def f_dense(data, indices, lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

    def f_sparse(data, indices, lhs, rhs):
      return sparse_bcoo._bcoo_rdot_general(lhs, data, indices,
                                            rhs_spinfo=BCOOInfo(rhs.shape),
                                            dimension_numbers=dimension_numbers)

    tol = {'float32': 3E-2} if jtu.device_under_test() == 'tpu' else {}
    self._CheckAgainstNumpy(f_dense, f_sparse, args_maker, tol=tol)
    self._CheckAgainstNumpy(f_dense, jit(f_sparse), args_maker, tol=tol)
    # TODO(jakevdp): In rare cases, this fails python_should_be_executing check. Why?
    # self._CompileAndCheck(f_sparse, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 2, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_dot_general_partial_batch(self, lhs_shape, rhs_shape, dtype,
                                          dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())

    X = rng_sparse(lhs_shape, dtype)
    nse = sparse.util._count_stored_elements(X, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(X, nse=nse, n_batch=n_batch, n_dense=n_dense)
    Y = rng(rhs_shape, dtype)

    def f_dense(X, Y):
      return lax.dot_general(X, Y, dimension_numbers=dimension_numbers)

    def f_sparse(data, indices, Y):
      return sparse_bcoo._bcoo_dot_general(data, indices, Y, lhs_spinfo=BCOOInfo(X.shape),
                                           dimension_numbers=dimension_numbers)

    for data, indices in itertools.product([data, data[:1]], [indices, indices[:1]]):
      X = sparse_bcoo._bcoo_todense(data, indices, spinfo=BCOOInfo(X.shape))
      self.assertAllClose(f_dense(X, Y), f_sparse(data, indices, Y))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((4, 5), (5, 3), (([1], [0]), ([], [])), 0, 0),
          ((2, 4, 5), (2, 5, 3), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 2, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 2, 0),
          # This requires contraction over dense dimensions, which is not yet implemented:
          # ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating))
  def test_bcoo_dot_general_ad(self, lhs_shape, rhs_shape, dtype,
                               dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_small(self.rng())
    rng_sparse = rand_sparse(self.rng())

    X = rng_sparse(lhs_shape, dtype)
    nse = sparse.util._count_stored_elements(X, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(X, nse=nse, n_batch=n_batch, n_dense=n_dense)
    Y = rng(rhs_shape, dtype)

    # gradient with respect to rhs
    def f_dense(Y):
      return lax.dot_general(X, Y, dimension_numbers=dimension_numbers)

    def f_sparse(Y):
      return sparse_bcoo._bcoo_dot_general(data, indices, Y, lhs_spinfo=BCOOInfo(X.shape),
                                           dimension_numbers=dimension_numbers)

    jf_dense = jax.jacfwd(f_dense)(Y)
    jr_dense = jax.jacrev(f_dense)(Y)
    jf_sparse = jax.jacfwd(f_sparse)(Y)
    jr_sparse = jax.jacrev(f_sparse)(Y)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    self.assertAllClose(jf_dense, jf_sparse, rtol=tol)
    self.assertAllClose(jr_dense, jr_sparse, rtol=tol)
    self.assertAllClose(jf_sparse, jr_sparse, rtol=tol)

    # gradient with respect to lhs
    def g_dense(X):
      return lax.dot_general(X, Y, dimension_numbers=dimension_numbers)

    def g_sparse(data):
      return sparse_bcoo._bcoo_dot_general(data, indices, Y, lhs_spinfo=BCOOInfo(X.shape),
                                           dimension_numbers=dimension_numbers)

    jf_dense = jax.jacfwd(g_dense)(X)
    jr_dense = jax.jacrev(g_dense)(X)
    jf_sparse = jax.jacfwd(g_sparse)(data)
    jr_sparse = jax.jacrev(g_sparse)(data)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    self.assertAllClose(jf_dense, jr_dense, rtol=tol)
    self.assertAllClose(jf_sparse, jr_sparse, rtol=tol)

    # Extract the sparse jacobian from the dense & compare.
    def extract(X):
      return sparse.bcoo_extract(indices, X)
    for i in range(g_dense(X).ndim):
      extract = jax.vmap(extract)
    self.assertAllClose(extract(jf_dense), jf_sparse, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 0, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 0, 1),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 0, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 1),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 0, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 1, 2),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_dot_general_sampled(self, lhs_shape, rhs_shape, dtype, dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_default(self.rng())
    sprng = rand_sparse(self.rng())
    out_shape = lax.dot_general(
      jnp.zeros(lhs_shape), jnp.zeros(rhs_shape),
      dimension_numbers=dimension_numbers).shape

    args_maker = lambda: [
      rng(lhs_shape, dtype), rng(rhs_shape, dtype),
      sparse.BCOO.fromdense(sprng(out_shape, dtype),
                            n_batch=n_batch, n_dense=n_dense).indices]

    def dense_fun(lhs, rhs, indices):
      AB = lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)
      return sparse.bcoo_extract(indices, AB)
    def sparse_fun(lhs, rhs, indices):
      return sparse.bcoo_dot_general_sampled(
                lhs, rhs, indices, dimension_numbers=dimension_numbers)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    self._CheckAgainstNumpy(dense_fun, sparse_fun, args_maker, tol=tol)
    # TODO: python_should_be_executing check occasionally fails... why?
    # self._CompileAndCheck(sparse_fun, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_n_batch={}_n_dense={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, n_batch, n_dense),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "n_batch": n_batch, "n_dense": n_dense}
      for lhs_shape, rhs_shape, dimension_numbers, n_batch, n_dense in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 0),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 1, 1),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0])), 2, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 0),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 1, 1),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 0),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1])), 2, 1),
      ]
      for dtype in jtu.dtypes.floating))
  def test_bcoo_dot_general_sampled_ad(self, lhs_shape, rhs_shape, dtype, dimension_numbers, n_batch, n_dense):
    rng = jtu.rand_default(self.rng())
    sprng = rand_sparse(self.rng())
    out_shape = lax.dot_general(
      jnp.zeros(lhs_shape), jnp.zeros(rhs_shape),
      dimension_numbers=dimension_numbers).shape

    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    indices = sparse.BCOO.fromdense(sprng(out_shape, dtype),
                                    n_batch=n_batch, n_dense=n_dense).indices

    def dense_fun(lhs, rhs, indices):
      AB = lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)
      return sparse.bcoo_extract(indices, AB)
    def sparse_fun(lhs, rhs, indices):
      return sparse.bcoo_dot_general_sampled(
                lhs, rhs, indices, dimension_numbers=dimension_numbers)

    jf_dense = jax.jacfwd(dense_fun)(lhs, rhs, indices)
    jf_sparse = jax.jacfwd(sparse_fun)(lhs, rhs, indices)
    jr_dense = jax.jacrev(dense_fun)(lhs, rhs, indices)
    jr_sparse = jax.jacrev(sparse_fun)(lhs, rhs, indices)

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-3}

    self.assertAllClose(jf_sparse, jf_dense, atol=tol)
    self.assertAllClose(jr_sparse, jr_dense, atol=tol)
    self.assertAllClose(jf_sparse, jr_sparse, atol=tol)

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}[n_batch={}]_{}[n_batch={}]_swap={}_dims={}".format(
        jtu.format_shape_dtype_string(lhs_shape, dtype), lhs_n_batch,
        jtu.format_shape_dtype_string(rhs_shape, dtype), rhs_n_batch,
        swap, dimension_numbers),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape,
       "lhs_n_batch": lhs_n_batch, "rhs_n_batch": rhs_n_batch,
       "dimension_numbers": dimension_numbers, "swap": swap, "dtype": dtype}
      for lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, dimension_numbers in [
          # (batched) outer products (no contraction)
          ((5,), 0, (6,), 0, (([], []), ([], []))),
          ((3, 5), 0, (2, 4), 0, (([], []), ([], []))),
          ((3, 5), 1, (3, 4), 1, (([], []), ([0], [0]))),
          # (batched) vector-vector products
          ((5,), 0, (5,), 0, (([0], [0]), ([], []))),
          ((7,), 0, (7,), 0, (([0], [0]), ([], []))),
          ((5, 7), 1, (7,), 0, (([1], [0]), ([], []))),
          ((2, 3, 4), 2, (2, 4), 1, (([2], [1]), ([0], [0]))),
          ((2, 3, 4), 2, (2, 4), 1, (([2], [1]), ([], []))),
          ((2, 3, 4), 2, (3, 4), 1, (([2], [1]), ([1], [0]))),
          ((2, 3, 4), 2, (3, 4), 1, (([2], [1]), ([], []))),
          # (batched) matrix-vector products
          ((5, 7), 0, (7,), 0, (([1], [0]), ([], []))),
          ((2, 3, 4), 1, (4,), 0, (([2], [0]), ([], []))),
          ((2, 3, 4), 1, (2, 4), 1, (([2], [1]), ([0], [0]))),
          ((3, 2, 4), 1, (3, 4), 1, (([2], [1]), ([0], [0]))),
          ((2, 3, 4), 0, (2,), 0, (([0], [0]), ([], []))),
          # (batched) matrix-matrix products
          ((5, 7), 0, (7, 3), 0, (([1], [0]), ([], []))),
          ((2, 3, 4), 1, (4, 3), 0, (([2], [0]), ([], []))),
          ((2, 3, 4), 1, (2, 4, 3), 1, (([2], [1]), ([0], [0]))),
          # more general operations
          ((2, 3, 4, 3), 1, (2, 4, 3, 4), 1, (([2, 3], [1, 2]), ([0], [0]))),
          ((2, 3, 4, 3, 1), 2, (3, 2, 3, 4), 2, (([2, 3], [3, 2]), ([0, 1], [1, 0]))),
      ]
      for swap in [True, False]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_spdot_general(self, lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, dtype, swap, dimension_numbers):
    if swap:
      dimension_numbers = tuple(d[::-1] for d in dimension_numbers)
      lhs_shape, rhs_shape = rhs_shape, lhs_shape
      lhs_n_batch, rhs_n_batch = rhs_n_batch, lhs_n_batch

    lhs_n_sparse = len(lhs_shape) - lhs_n_batch
    rhs_batch = dimension_numbers[1][1]
    lhs_contracting = dimension_numbers[0][0]
    should_error = (rhs_n_batch > len(rhs_batch) and lhs_n_sparse > len(lhs_contracting))

    sprng = rand_sparse(self.rng())
    def args_maker():
      x = sprng(lhs_shape, dtype)
      y = sprng(rhs_shape, dtype)
      xsp = sparse.BCOO.fromdense(x, n_batch=lhs_n_batch)
      ysp = sparse.BCOO.fromdense(y, n_batch=rhs_n_batch)
      return x, y, xsp, ysp

    def f_dense(x, y, xsp, ysp):
      return lax.dot_general(x, y, dimension_numbers=dimension_numbers)

    def f_sparse(x, y, xsp, ysp):
      shape = sparse.bcoo._dot_general_validated_shape(xsp.shape, ysp.shape, dimension_numbers)
      data, indices = sparse_bcoo._bcoo_spdot_general(
          xsp.data, xsp.indices, ysp.data, ysp.indices, lhs_spinfo=xsp._info,
          rhs_spinfo=ysp._info, dimension_numbers=dimension_numbers)
      return sparse_bcoo._bcoo_todense(data, indices, spinfo=BCOOInfo(shape))

    tol = {"complex128": 1E-14}
    if should_error:
      with self.assertRaisesRegex(ValueError, ".*cannot have unused batch dims on rhs with unused sparse dims on lhs."):
        f_sparse(*args_maker())
    else:
      self._CheckAgainstNumpy(f_dense, f_sparse, args_maker, tol=tol)
      self._CheckAgainstNumpy(jit(f_dense), jit(f_sparse), args_maker, tol=tol)
      # TODO(jakevdp): This occasionally fails python_should_be_executing check. Why?
      # self._CompileAndCheck(f_sparse, args_maker)

  def test_bcoo_spdot_general_nse(self):
    # vector-vector product -> nse=1
    x = sparse.BCOO.fromdense(jnp.arange(3))
    self.assertEqual((x @ x).nse, 1)

    # matrix-vector product -> nse matches matrix
    M = sparse.BCOO.fromdense(jnp.arange(6).reshape(2, 3))
    self.assertEqual((M @ x).nse, M.nse)

    # matrix-matrix product -> product of nse
    N = sparse.BCOO.fromdense(jnp.arange(12).reshape(3, 4))
    self.assertEqual((M @ N).nse, M.nse * N.nse)
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}[n_batch={}]_rhs_shape={}[n_batch={}]_dimension_numbers={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype), lhs_n_batch,
               jtu.format_shape_dtype_string(rhs_shape, dtype), rhs_n_batch,
               dimension_numbers),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers,
       "lhs_n_batch": lhs_n_batch, "rhs_n_batch": rhs_n_batch}
      for lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, dimension_numbers in [
          ((4, 5), 0, (5,), 0, (([1], [0]), ([], []))),
          ((2, 4, 5), 1, (5,), 0, (([2], [0]), ([], []))),
          ((4, 5), 0, (5, 3), 0, (([1], [0]), ([], []))),
          ((2, 4, 5), 1, (2, 5, 3), 1, (([2], [1]), ([0], [0]))),
      ]
      for dtype in jtu.dtypes.floating))
  def test_bcoo_spdot_general_ad(self, lhs_shape, rhs_shape, dtype,
                                 dimension_numbers, lhs_n_batch, rhs_n_batch):
    rng = rand_sparse(self.rng())

    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)

    lhs_sp = sparse.BCOO.fromdense(lhs, n_batch=lhs_n_batch)
    rhs_sp = sparse.BCOO.fromdense(rhs, n_batch=rhs_n_batch)

    def f_dense(lhs_data, rhs_data):
      lhs = sparse.BCOO((lhs_data, lhs_sp.indices), shape=lhs_sp.shape).todense()
      rhs = sparse.BCOO((rhs_data, rhs_sp.indices), shape=rhs_sp.shape).todense()
      return (lhs @ rhs).sum()

    def f_sparse(lhs_data, rhs_data):
      lhs = sparse.BCOO((lhs_data, lhs_sp.indices), shape=lhs_sp.shape)
      rhs = sparse.BCOO((rhs_data, rhs_sp.indices), shape=rhs_sp.shape)
      return (lhs @ rhs).sum()

    tol = {}
    if jtu.device_under_test() == "tpu":
      tol = {np.float32: 5E-2}

    jf_dense_0 = jax.jacfwd(f_dense, argnums=0)(lhs_sp.data, rhs_sp.data)
    jf_sparse_0 = jax.jacfwd(f_sparse, argnums=0)(lhs_sp.data, rhs_sp.data)
    self.assertAllClose(jf_dense_0, jf_sparse_0, rtol=tol)

    jf_dense_1 = jax.jacfwd(f_dense, argnums=1)(lhs_sp.data, rhs_sp.data)
    jf_sparse_1 = jax.jacfwd(f_sparse, argnums=1)(lhs_sp.data, rhs_sp.data)
    self.assertAllClose(jf_dense_1, jf_sparse_1, rtol=tol)

    jf_dense_0, jf_dense_1 = jax.jacfwd(f_dense, argnums=(0, 1))(lhs_sp.data, rhs_sp.data)
    jf_sparse_0, jf_sparse_1 = jax.jacfwd(f_sparse, argnums=(0, 1))(lhs_sp.data, rhs_sp.data)
    self.assertAllClose(jf_dense_0, jf_sparse_0, rtol=tol)
    self.assertAllClose(jf_dense_1, jf_sparse_1, rtol=tol)

  def test_bcoo_spdot_general_ad_bug(self):
    # Regression test for https://github.com/google/jax/issues/10163
    A_indices = jnp.array([[0, 1], [0, 2], [1, 1], [1, 2], [1, 0]])
    A_values = jnp.array([-2.0, 1.0, -1.0, 0.5, 2.0])
    A_shape = (2, 3)

    B_indices = jnp.array([[0, 2], [2, 1], [0, 3], [1, 3], [1, 0], [0, 0]])
    B_values = jnp.array([10.0, 100.0, 1000.0, -5.0, -50.0, -500.0])
    B_shape = (3, 4)

    def sp_sp_product(v1, v2):
        A = sparse.BCOO((v1, A_indices), shape=A_shape)
        B = sparse.BCOO((v2, B_indices), shape=B_shape)
        return (A @ B).todense()

    def sp_de_product(v1, v2):
        A = sparse.BCOO((v1, A_indices), shape=A_shape)
        B = sparse.BCOO((v2, B_indices), shape=B_shape).todense()
        return A @ B

    def de_de_product(v1, v2):
        sparse1 = sparse.BCOO((v1, A_indices), shape=A_shape).todense()
        dense2 = sparse.BCOO((v2, B_indices), shape=B_shape).todense()
        return sparse1 @ dense2

    sp_sp_jac = jax.jacfwd(sp_sp_product, argnums=1)(A_values, B_values)
    sp_de_jac = jax.jacfwd(sp_de_product, argnums=1)(A_values, B_values)
    de_de_jac = jax.jacfwd(de_de_product, argnums=1)(A_values, B_values)

    self.assertAllClose(sp_sp_jac, de_de_jac)
    self.assertAllClose(sp_de_jac, de_de_jac)

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}[n_batch={}]_{}[n_batch={}]_in_axes={}".format(
        jtu.format_shape_dtype_string(lhs_shape, dtype), lhs_n_batch,
        jtu.format_shape_dtype_string(rhs_shape, dtype), rhs_n_batch,
        in_axes),
       "lhs_shape": lhs_shape, "lhs_n_batch": lhs_n_batch,
       "rhs_shape": rhs_shape, "rhs_n_batch": rhs_n_batch,
       "dtype": dtype, "in_axes": in_axes}
      for lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, in_axes in [
        ((3, 5), 1, (3, 5), 1, 0),
        ((3, 4, 5), 1, (3, 5), 1, 0),
        ((3, 4, 5), 2, (3, 5), 1, 0),
        # TODO(jakevdp): test these once unequal batches are implemented
        # ((4, 5), 1, (5,), 0, (0, None)),
        # ((3, 4, 5), 1, (5,), 0, (0, None)),
        # ((4, 5), 0, (3, 5), 1, (None, 0)),
      ]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_spmm_batched(self, lhs_shape, lhs_n_batch, rhs_shape, rhs_n_batch, dtype, in_axes):
    sprng = rand_sparse(self.rng())
    def args_maker():
      x = sprng(lhs_shape, dtype)
      y = sprng(rhs_shape, dtype)
      xsp = sparse.BCOO.fromdense(x, n_batch=lhs_n_batch)
      ysp = sparse.BCOO.fromdense(y, n_batch=rhs_n_batch)
      return x, y, xsp, ysp

    def f_dense(x, y, _, __):
      return jax.vmap(operator.matmul, in_axes=in_axes)(x, y)
    def f_sparse(_, __, x, y):
      return jax.vmap(operator.matmul, in_axes=in_axes)(x, y)

    args = args_maker()
    result_dense = f_dense(*args)
    result_sparse = f_sparse(*args)
    self.assertAllClose(result_dense, result_sparse.todense())
    result_sparse_jit = jax.jit(f_sparse)(*args)
    self.assertAllClose(result_dense, result_sparse_jit.todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}_nse={}_remove_zeros={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense, nse, remove_zeros),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense,
       "nse": nse, "remove_zeros": remove_zeros}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)
      for nse in [None, np.prod(shape) - 1]
      for remove_zeros in [True, False]))
  def test_bcoo_sum_duplicates(self, shape, dtype, n_batch, n_dense, nse, remove_zeros):
    # Create a matrix with duplicate indices
    rng_sparse = rand_sparse(self.rng(), rand_method=jtu.rand_some_zero)
    M = sparse.BCOO.fromdense(rng_sparse(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    new_indices = jnp.concatenate([M.indices, M.indices], axis=n_batch)
    new_data = jnp.concatenate([M.data, M.data], axis=n_batch)
    M = sparse.BCOO((new_data, new_indices), shape=M.shape)

    dedupe = partial(M.sum_duplicates, nse=nse, remove_zeros=remove_zeros)
    jit_dedupe = jax.jit(dedupe)

    M_dedup = dedupe()
    self.assertAllClose(M.todense(), M_dedup.todense())
    if nse:
      self.assertEqual(M_dedup.nse, nse)

    if not nse:
      with self.assertRaisesRegex(ValueError, ".*nse must be specified.*"):
        jit_dedupe()
    else:
      M_dedup = jit_dedupe()
      self.assertAllClose(M.todense(), M_dedup.todense())
      self.assertEqual(M_dedup.nse, nse)

    self.assertTrue(M_dedup.unique_indices)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}_nse={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense, nse),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense, "nse": nse}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)
      for nse in [None, 5, np.prod(shape) - 1]
  ))
  def test_bcoo_sum_duplicates_ad(self, shape, dtype, n_batch, n_dense, nse):
    # Create a matrix with duplicate indices
    rng_sparse = rand_sparse(self.rng(), rand_method=jtu.rand_some_zero)
    M = sparse.BCOO.fromdense(rng_sparse(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    new_indices = jnp.concatenate([M.indices, M.indices], axis=n_batch)
    new_data = jnp.concatenate([M.data, M.data], axis=n_batch)
    M = sparse.BCOO((new_data, new_indices), shape=M.shape)

    # TODO(jakevdp) address this corner case.
    if M.nse == 0:
      self.skipTest("known failure for nse=0")

    if nse == 'all':
      nse = M.nse

    def dedupe(data, nse=nse):
      mat = sparse.BCOO((data, M.indices), shape=M.shape)
      mat_dedup = mat.sum_duplicates(nse=nse, remove_zeros=False)
      return mat_dedup.data

    data_dot_fwd = jax.jacfwd(dedupe)(M.data)
    data_dot_rev = jax.jacrev(dedupe)(M.data)

    self.assertAllClose(data_dot_fwd, data_dot_rev)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_sort_indices(self, shape, dtype, n_batch, n_dense):
    rng_sparse = rand_sparse(self.rng(), rand_method=jtu.rand_some_zero)
    M = sparse.BCOO.fromdense(rng_sparse(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    M.indices = M.indices[..., ::-1, :]

    M_sorted = M.sort_indices()
    self.assertArraysEqual(M.todense(), M_sorted.todense())
    self.assertEqual(M.unique_indices, M_sorted.unique_indices)

    indices = M_sorted.indices
    if indices.size > 0:
      flatind = indices.reshape(-1, *indices.shape[-2:]).transpose(0, 2, 1)
      sorted = jax.vmap(jnp.lexsort)(flatind[:, ::-1])
      self.assertArraysEqual(sorted, lax.broadcasted_iota(sorted.dtype, sorted.shape, sorted.ndim - 1))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_sort_indices_ad(self, shape, dtype, n_batch, n_dense):
    rng_sparse = rand_sparse(self.rng(), rand_method=jtu.rand_some_zero)
    M = sparse.BCOO.fromdense(rng_sparse(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    M.indices = M.indices[..., ::-1, :]

    def sort_indices(data):
      return sparse.BCOO((data, M.indices), shape=M.shape).sort_indices().data

    data_dot_fwd = jax.jacfwd(sort_indices)(M.data)
    data_dot_rev = jax.jacrev(sort_indices)(M.data)

    self.assertAllClose(data_dot_fwd, data_dot_rev)

  def test_bcoo_sort_indices_broadcasted(self):
    rng_index = jtu.rand_int(self.rng(), low=0, high=10)
    rng_data = jtu.rand_default(self.rng())

    # Construct matrix with three broadcasted batch dimensions.
    indices = rng_index((1, 3, 1, 10, 2), dtype='int32')
    data = rng_data((1, 1, 4, 10, 3), dtype='int32')
    shape = (2, 3, 4, 5, 4, 3)
    mat = sparse.BCOO((data, indices), shape=shape)

    indices_shape_out = indices.shape
    data_shape_out = (*map(max, indices.shape[:3], data.shape[:3]), *data.shape[3:])

    mat_sorted = sparse.bcoo_sort_indices(mat)
    assert mat_sorted.indices.shape == indices_shape_out
    assert mat_sorted.data.shape == data_shape_out
    self.assertArraysEqual(mat.todense(), mat_sorted.todense())

    mat_sorted_jit = jit(sparse.bcoo_sort_indices)(mat)
    assert mat_sorted_jit.indices.shape == indices_shape_out
    assert mat_sorted_jit.data.shape == data_shape_out
    self.assertArraysEqual(mat.todense(), mat_sorted_jit.todense())


  def test_bcoo_sum_duplicates_inferred_nse(self):
    x = sparse.BCOO.fromdense(jnp.diag(jnp.arange(4)))
    self.assertEqual(x.nse, 3)
    y = x + x.T
    self.assertEqual(y.nse, 6)
    y2 = y.sum_duplicates()
    self.assertEqual(y2.nse, 3)
    self.assertArraysEqual(y.todense(), y2.todense())

  def test_bcoo_sum_duplicates_remove_zeros(self):
    data = jnp.array([0, 1, 0, 0])
    indices = jnp.array([[0], [1], [2], [3]])
    x = sparse.BCOO((data, indices), shape=(4,))
    self.assertEqual(x.nse, 4)

    y1 = x.sum_duplicates(remove_zeros=True)
    self.assertArraysEqual(x.todense(), y1.todense())
    self.assertEqual(y1.nse, 1)

    y2 = x.sum_duplicates(remove_zeros=False)
    self.assertArraysEqual(x.todense(), y2.todense())
    self.assertEqual(y2.nse, x.nse)

  def test_bcoo_sum_duplicates_padding(self):
    # Regression test for https://github.com/google/jax/issues/8163
    size = 3
    data = jnp.array([1, 0, 0])
    indices = jnp.array([1, size, size])[:, None]
    x = sparse.BCOO((data, indices), shape=(3,))
    y = x.sum_duplicates(nse=x.nse)
    self.assertArraysEqual(x.todense(), y.todense())
    self.assertArraysEqual(x.indices, y.indices)
    self.assertArraysEqual(x.data, y.data)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}_axes={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense, axes),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense, "axes": axes}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)
      for naxes in range(len(shape))
      for axes in itertools.combinations(range(len(shape)), naxes)))
  def test_bcoo_reduce_sum(self, shape, dtype, n_batch, n_dense, axes):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense)
    data_out, indices_out, shape_out = sparse_bcoo._bcoo_reduce_sum(
        data, indices, spinfo=BCOOInfo(shape), axes=axes)
    result_dense = M.sum(axes)
    result_sparse = sparse_bcoo._bcoo_todense(data_out, indices_out, spinfo=BCOOInfo(shape_out))
    tol = {np.float32: 1E-6, np.float64: 1E-14}
    self.assertAllClose(result_dense, result_sparse, atol=tol, rtol=tol)

  def test_bcoo_reshape_error(self):
    x = sparse.BCOO.fromdense(jnp.ones((2, 2, 3)), n_batch=1)
    with self.assertRaisesRegex(ValueError, ".*cannot mix batch and sparse dimensions.*"):
      x.reshape(3, 2, 2)
    y = sparse.BCOO((x.data[:1], x.indices), shape=x.shape)
    with self.assertRaisesRegex(NotImplementedError, "reshape of arrays with broadacsted batch dimensions."):
      y.reshape(2, 3, 2)

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
        jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
        jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
      }
      for lhs_shape, rhs_shape in [[(3,), (3,)],
                                   [(3, 4), (4,)],
                                   [(4,), (4, 5)],
                                   [(3, 4), (4, 5)],
                                   [(3, 4), (2, 4, 5)],
                                   [(2, 3, 4), (4, 5)],
                                   [(2, 3, 4), (2, 4, 5)]]
      for lhs_dtype in all_dtypes
      for rhs_dtype in all_dtypes))
  def test_bcoo_matmul(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    lhs = jnp.array(rng(lhs_shape, lhs_dtype))
    rhs = jnp.array(rng(rhs_shape, rhs_dtype))

    # Note: currently, batch dimensions in matmul must correspond to batch
    # dimensions in the sparse representation.
    lhs_sp = sparse.BCOO.fromdense(lhs, n_batch=max(0, len(lhs_shape) - 2))
    rhs_sp = sparse.BCOO.fromdense(rhs, n_batch=max(0, len(rhs_shape) - 2))

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      out1 = lhs @ rhs
      out2 = lhs_sp @ rhs
      out3 = lhs @ rhs_sp

    tol = {np.float64: 1E-13, np.complex128: 1E-13,
           np.float32: 1E-6, np.complex64: 1E-6}
    self.assertAllClose(out1, out2, rtol=tol)
    self.assertAllClose(out1, out3, rtol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_n_batch={}_n_dense={}".format(
        jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
        jtu.format_shape_dtype_string(rhs_shape, rhs_dtype),
        n_batch, n_dense),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "n_batch": n_batch, "n_dense": n_dense,
      }
      for lhs_shape, rhs_shape in [[(3,), ()], [(3,), (1,)], [(3,), (3,)],
                                   [(3, 4), ()], [(3, 4), (4,)], [(3, 4), (3, 1)], [(3, 4), (3, 4)],
                                   [(3, 4, 5), (4, 5)], [(3, 4, 5), (3, 1, 1)], [(3, 4, 5), (1, 4, 1)]]
      for n_batch in range(len(lhs_shape) + 1)
      for n_dense in range(len(lhs_shape) + 1 - n_batch)
      for lhs_dtype in all_dtypes
      for rhs_dtype in all_dtypes))
  @jax.numpy_rank_promotion('allow')  # This test explicitly exercises implicit rank promotion.
  def test_bcoo_mul_dense(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, n_batch, n_dense):
    rng_lhs = rand_sparse(self.rng())
    rng_rhs = jtu.rand_default(self.rng())
    lhs = jnp.array(rng_lhs(lhs_shape, lhs_dtype))
    rhs = jnp.array(rng_rhs(rhs_shape, rhs_dtype))

    sp = lambda x: sparse.BCOO.fromdense(x, n_batch=n_batch, n_dense=n_dense)

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      out1 = lhs * rhs
      out2 = (sp(lhs) * rhs).todense()
      out3 = (rhs * sp(lhs)).todense()

    tol = {np.float64: 1E-13, np.complex128: 1E-13,
           np.float32: 1E-6, np.complex64: 1E-6}
    self.assertAllClose(out1, out2, rtol=tol)
    self.assertAllClose(out1, out3, rtol=tol)
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_n_batch={}_{}_n_batch={}_n_dense={}".format(
        jtu.format_shape_dtype_string(lhs_shape, lhs_dtype), lhs_n_batch,
        jtu.format_shape_dtype_string(rhs_shape, rhs_dtype), rhs_n_batch, n_dense),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "lhs_n_batch": lhs_n_batch, "rhs_n_batch": rhs_n_batch, "n_dense": n_dense,
      }
      # TODO(jakevdp): add broadcasted shapes (from bcoo_mul_dense) once sparse-sparse mul
      # supports inputs of differing rank.
      for lhs_shape, rhs_shape in [[(3,), (1,)], [(3,), (3,)],
                                   [(3, 4), (1, 1)], [(3, 4), (1, 4)], [(3, 4), (3, 1)], [(3, 4), (3, 4)],
                                   [(3, 4, 5), (1, 4, 5)], [(3, 4, 5), (3, 1, 1)], [(3, 4, 5), (1, 4, 1)]]
      # TODO(jakevdp): add tests for batch & dense dimensions.
      for lhs_n_batch in range(len(lhs_shape) + 1)
      for rhs_n_batch in range(len(lhs_shape) + 1)
      for n_dense in range(len(lhs_shape) + 1 - max(lhs_n_batch, rhs_n_batch))
      for lhs_dtype in all_dtypes
      for rhs_dtype in all_dtypes))
  def test_bcoo_mul_sparse(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, lhs_n_batch, rhs_n_batch, n_dense):
    rng = rand_sparse(self.rng())
    lhs = jnp.array(rng(lhs_shape, lhs_dtype))
    rhs = jnp.array(rng(rhs_shape, rhs_dtype))

    lhs_sp = sparse.BCOO.fromdense(lhs, n_batch=lhs_n_batch, n_dense=n_dense)
    rhs_sp = sparse.BCOO.fromdense(rhs, n_batch=rhs_n_batch, n_dense=n_dense)

    with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
      out1 = lhs * rhs
      out2 = (lhs_sp * rhs_sp).todense()

    tol = {np.float64: 1E-13, np.complex128: 1E-13,
           np.float32: 1E-6, np.complex64: 1E-6}
    self.assertAllClose(out1, out2, rtol=tol)

  def test_bcoo_mul_sparse_with_duplicates(self):
    # Regression test for https://github.com/google/jax/issues/8888
    indices = jnp.array([[0, 1, 0, 0, 1, 1],
                         [1, 0, 1, 2, 0, 2]]).T
    data = jnp.array([1, 2, 3, 4, 5, 6])
    mat = sparse.BCOO((data, indices), shape=(3, 3))
    self.assertArraysEqual((mat * mat).todense(), mat.todense() * mat.todense())

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_n_batch={}_n_dense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
       for shape in [(), (3,), (3, 5), (3, 5, 4)]
       for dtype in all_dtypes
       for n_batch in range(len(shape) + 1)
       for n_dense in range(len(shape) + 1 - n_batch)))
  def test_bcoo_broadcast_in_dim(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    x = jnp.array(rng(shape, dtype))
    xsp = sparse.BCOO.fromdense(x, n_batch=n_batch, n_dense=n_dense)

    self.assertEqual(xsp[None].n_batch, xsp.n_batch + 1)
    self.assertArraysEqual(xsp[None].todense(), x[None])

    if len(shape) >= 1:
      self.assertEqual(xsp[:, None].n_batch, xsp.n_batch if xsp.n_batch < 1 else xsp.n_batch + 1)
      self.assertArraysEqual(xsp[:, None].todense(), x[:, None])
      self.assertArraysEqual(xsp[:, None, None].todense(), x[:, None, None])
    if len(shape) >= 2:
      self.assertEqual(xsp[:, :, None].n_batch, xsp.n_batch if xsp.n_batch < 2 else xsp.n_batch + 1)
      self.assertArraysEqual(xsp[:, :, None].todense(), x[:, :, None])
      self.assertArraysEqual(xsp[:, None, :, None].todense(), x[:, None, :, None])

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_n_batch={}_n_dense={}_dimension={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense, dimension),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense, "dimension": dimension}
       for shape in [ (3,), (3, 5), (3, 5, 4)]
       for dtype in all_dtypes
       for n_batch in range(len(shape) + 1)
       for n_dense in range(len(shape) + 1 - n_batch)
       for dimension in range(len(shape) - n_dense)))  # Concatenation of dense dimensions not implemented.
  def test_bcoo_concatenate(self, shape, dtype, n_batch, n_dense, dimension):
    rng = rand_sparse(self.rng())
    operands_dense = [rng(shape, dtype) for i in range(3)]
    operands_sparse = [sparse.BCOO.fromdense(op, n_batch=n_batch, n_dense=n_dense)
                       for op in operands_dense]

    mat_dense = lax.concatenate(operands_dense, dimension=dimension)
    mat_sparse = sparse.bcoo_concatenate(operands_sparse, dimension=dimension)

    self.assertArraysEqual(mat_sparse.todense(), mat_dense)

  def test_bcoo_vmap_shape(self, shape=(2, 3, 4, 5), dtype=np.float32):
    # This test checks that BCOO shape metadata interacts correctly with vmap.
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)

    def make_bcoo(M):
      return sparse_bcoo._bcoo_fromdense(M, nse=np.prod(M.shape[:-1], dtype=int), n_dense=1)

    todense = partial(sparse_bcoo._bcoo_todense, spinfo=BCOOInfo(shape))

    for _ in range(3):
      make_bcoo = jax.vmap(make_bcoo)
      Msp_data, Msp_indices = make_bcoo(M)
      Msp_dense = todense(Msp_data, Msp_indices)
      self.assertEqual(Msp_dense.shape, M.shape)
      self.assertArraysEqual(Msp_dense, M)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}->{}_ndense={}->{}".format(
        jtu.format_shape_dtype_string(shape, dtype),
        n_batch, n_batch_out, n_dense, n_dense_out),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense,
       "n_batch_out": n_batch_out, "n_dense_out": n_dense_out}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.integer
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)
      for n_batch_out in range(len(shape) + 1)
      for n_dense_out in range(len(shape) + 1 - n_batch_out)))
  def test_bcoo_update_layout(self, shape, dtype, n_batch, n_batch_out, n_dense, n_dense_out):
    rng = rand_sparse(self.rng())
    mat = sparse.BCOO.fromdense(rng(shape, dtype), n_batch=n_batch, n_dense=n_dense)
    kwds = dict(n_batch=n_batch_out, n_dense=n_dense_out)
    # TODO(jakevdp): in case of length-0 or length-1 shapes errors/warnings will not be raised.
    if n_dense_out > n_dense or n_batch_out > n_batch:
      with self.assertRaises(sparse.SparseEfficiencyError):
        sparse.bcoo_update_layout(mat, **kwds)
      with self.assertRaises(sparse.SparseEfficiencyError):
        sparse.bcoo_update_layout(mat, **kwds, on_inefficient='error')
      with self.assertWarns(sparse.SparseEfficiencyWarning):
        sparse.bcoo_update_layout(mat, **kwds, on_inefficient='warn')
      kwds['on_inefficient'] = None
    mat_new = sparse.bcoo_update_layout(mat, **kwds)
    self.assertEqual(mat_new.n_batch, n_batch_out)
    self.assertEqual(mat_new.n_dense, n_dense_out)
    self.assertArraysEqual(mat.todense(), mat_new.todense())

  def test_bcoo_update_layout_method(self, shape=(2, 3, 4)):
    # simple test to make sure update_layout method properly forwards.
    rng = rand_sparse(self.rng())
    mat = sparse.BCOO.fromdense(rng((2, 3, 4), 'float32'), n_batch=1, n_dense=1)
    mat_new = mat.update_layout(n_batch=0, n_dense=0)
    self.assertEqual(mat_new.n_batch, 0)
    self.assertEqual(mat_new.n_dense, 0)
    self.assertArraysEqual(mat.todense(), mat_new.todense())

  def test_bcoo_bad_fillvals(self):
    # Extra values have 100 rather than zero. This lets us check that logic is
    # properly ignoring these indices.
    data = jnp.array([1, 2, 3, 100, 100])
    indices = jnp.array([1, 2, 3, 5, 5])[:, None]
    x_sp = sparse.BCOO((data, indices), shape=(5,))
    x_de = x_sp.todense()

    data = jnp.array([3, 2, 100, 100])
    indices = jnp.array([2, 3, 5, 5])[:, None]
    y_sp = sparse.BCOO((data, indices), shape=(5,))
    y_de = y_sp.todense()

    self.assertArraysEqual(x_de, jnp.array([0, 1, 2, 3, 0]))
    self.assertArraysEqual(y_de, jnp.array([0, 0, 3, 2, 0]))

    self.assertArraysEqual(x_sp.sum_duplicates().todense(), x_de)
    self.assertArraysEqual(y_sp.sum_duplicates().todense(), y_de)

    # reduce_sum:
    self.assertArraysEqual(x_sp.sum(), x_de.sum())

    # bcoo_dot_general
    self.assertArraysEqual(x_sp @ y_de, x_de @ y_de)

    # bcoo_rdot_general
    self.assertArraysEqual(x_de @ y_sp, x_de @ y_de)

    # bcoo_spdot_general
    self.assertArraysEqual((x_sp @ y_sp).todense(), x_de @ y_de)
    self.assertArraysEqual((y_sp @ x_sp).todense(), y_de @ x_de)


class SparseGradTest(jtu.JaxTestCase):
  def test_sparse_grad(self):
    rng_sparse = rand_sparse(self.rng())
    rng = jtu.rand_default(self.rng())

    y = rng(5, "float32")
    X = rng_sparse((10, 5), "float32")
    Xsp = sparse.BCOO.fromdense(X)

    def f(X, y):
      return jnp.sum(X @ y)

    grad_dense = jax.grad(f, argnums=0)(X, y)
    grad_sparse = sparse.grad(f, argnums=0)(Xsp, y)

    # extract sparse gradient from dense gradient
    indices = tuple(Xsp.indices.T)
    grad_sparse_from_dense = jnp.zeros_like(grad_dense).at[indices].set(grad_dense[indices])

    self.assertArraysEqual(grad_sparse.todense(), grad_sparse_from_dense)


# TODO(tianjianlu): Build a testing framework for BCOOTest and BCSRTest.
class BCSRTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), n_batch, n_dense),
       "shape": shape, "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex
      for n_batch in range(len(shape) - 1)
      for n_dense in [len(shape) - 2 - n_batch]))
  def test_bcsr_dense_round_trip(self, shape, dtype, n_batch, n_dense):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    n_sparse = M.ndim - n_batch - n_dense
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    data, indices, indptr = sparse_bcsr.bcsr_fromdense(M, n_batch=n_batch,
                                                       n_dense=n_dense)
    data_jit, indices_jit, indptr_jit = jit(partial(sparse_bcsr.bcsr_fromdense,
                                                    nse=nse, n_batch=n_batch,
                                                    n_dense=n_dense))(M)
    self.assertArraysEqual(data, data_jit)
    self.assertArraysEqual(indices, indices_jit)
    self.assertArraysEqual(indptr, indptr_jit)

    self.assertEqual(data.dtype, dtype)
    self.assertEqual(data.shape,
                     shape[:n_batch] + (nse,) + shape[n_batch + n_sparse:])
    self.assertEqual(indices.dtype, jnp.int32)  # TODO: test passing this arg
    self.assertEqual(indices.shape, shape[:n_batch] + (nse,))
    self.assertEqual(indptr.dtype, jnp.int32)  # TODO: test passing this arg
    self.assertEqual(indptr.shape, shape[:n_batch] + (shape[n_batch] + 1,))

    todense = partial(sparse_bcsr.bcsr_todense, shape=shape)
    self.assertArraysEqual(M, todense(data, indices, indptr))
    self.assertArraysEqual(M, jit(todense)(data, indices, indptr))


class SparseObjectTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}", "cls": cls}
    for cls in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_jit_lower(self, cls):
    sparse_format = cls.__name__.lower()
    M = sparse.empty((2, 4), sparse_format=sparse_format)
    self.assertIsInstance(M, cls)
    jax.jit(lambda x: x).lower(M)  # doesn't crash

  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}{shape}", "cls": cls, "shape": shape}
    for cls in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]
    for shape in ([2, 5], [5, 3]))
  def test_empty(self, cls, shape):
    sparse_format = cls.__name__.lower()
    M = sparse.empty(shape, sparse_format=sparse_format)
    self.assertIsInstance(M, cls)
    self.assertEqual(M.nse, 0)
    self.assertArraysEqual(M.todense(), jnp.empty(shape))

  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}{(N, M, k)}",
     "cls": cls, "N": N, "M": M, "k": k}
    for cls in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]
    for N in [2, 5]
    for M in [None, 3]
    for k in [-2, 0, 1])
  def test_eye(self, cls, N, M, k):
    sparse_format = cls.__name__.lower()
    mat = sparse.eye(N, M, k, sparse_format=sparse_format)
    expected = jnp.eye(N, M, k)
    expected_nse = jnp.count_nonzero(expected)

    self.assertIsInstance(mat, cls)
    self.assertArraysEqual(mat.todense(), expected)
    self.assertEqual(mat.nse, expected_nse)

  @parameterized.named_parameters(
    {"testcase_name": f"{nse}_BCOO{shape}", "shape": shape, "nse": nse}
    for shape in ([2, 5], [5, 3])
    for nse in [0, 2])
  def test_empty_nse(self, shape, nse=2):
    M = sparse.empty(shape, nse=nse)
    self.assertEqual(M.nse, nse)
    self.assertArraysEqual(M.todense(), jnp.empty(shape))

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_block_until_ready(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = rand_sparse(self.rng(), post=Obj.fromdense)
    M = rng(shape, dtype)
    self.assertEqual(M.shape, M.block_until_ready().shape)
    self.assertArraysEqual(M.data, M.block_until_ready().data)
    self.assertArraysEqual(M.todense(), M.block_until_ready().todense())

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [jnp.array, sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_todense(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = rand_sparse(self.rng())
    M_dense = rng(shape, dtype)
    M = jnp.array(M_dense) if Obj is jnp.array else Obj.fromdense(M_dense)
    self.assertArraysEqual(sparse.todense(M), M_dense)
    self.assertArraysEqual(jit(sparse.todense)(M), M_dense)

  def test_todense_scalar(self):
    self.assertEqual(sparse.todense(1.0), 1.0)
    self.assertEqual(jit(sparse.todense)(1.0), 1.0)

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [jnp.array, sparse.BCOO])
  def test_todense_batching(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = rand_sparse(self.rng())
    M_dense = rng(shape, dtype)
    if Obj is sparse.BCOO:
      M = sparse.BCOO.fromdense(M_dense, n_batch=1)
    else:
      M = jnp.asarray(M_dense)
    self.assertArraysEqual(vmap(sparse.todense)(M), M_dense)
    self.assertArraysEqual(jit(vmap(sparse.todense))(M), M_dense)

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [jnp.array, sparse.BCOO])
  def test_todense_ad(self, Obj, shape=(3,), dtype=np.float32):
    M_dense = jnp.array([1., 2., 3.])
    M = M_dense if Obj is jnp.array else Obj.fromdense(M_dense)
    bufs, tree = tree_util.tree_flatten(M)
    jac = jnp.eye(M.shape[0], dtype=M.dtype)
    jac1 = jax.jacfwd(lambda *bufs: sparse.todense_p.bind(*bufs, tree=tree))(*bufs)
    jac2 = jax.jacrev(lambda *bufs: sparse.todense_p.bind(*bufs, tree=tree))(*bufs)
    self.assertArraysEqual(jac1, jac2)
    self.assertArraysEqual(jac, jac2)

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_attrs(self, Obj, shape=(5, 8), dtype=np.float16):
    rng = rand_sparse(self.rng(), post=Obj.fromdense)
    M = rng(shape, dtype)

    assert isinstance(M, Obj)
    assert M.shape == shape
    assert M.size == np.prod(shape)
    assert M.ndim == len(shape)
    assert M.dtype == dtype
    assert M.nse == (M.todense() != 0).sum()
    assert M.data.dtype == dtype

    with self.assertRaises(TypeError):
      hash(M)

    if isinstance(M, sparse.CSR):
      assert len(M.data) == len(M.indices)
      assert len(M.indptr) == M.shape[0] + 1
    elif isinstance(M, sparse.CSC):
      assert len(M.data) == len(M.indices)
      assert len(M.indptr) == M.shape[1] + 1
    elif isinstance(M, sparse.COO):
      assert len(M.data) == len(M.row) == len(M.col)
    elif isinstance(M, sparse.BCOO):
      assert M.data.shape[M.n_batch] == M.indices.shape[-2]
      assert M.indices.shape[-1] == M.n_sparse
    else:
      raise ValueError("Obj={Obj} not expected.")

  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
      {"testcase_name": "_{}_Obj={}".format(
        jtu.format_shape_dtype_string(shape, dtype), Obj.__name__),
       "shape": shape, "dtype": dtype, "Obj": Obj}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex)
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_dense_round_trip(self, shape, dtype, Obj):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    Msparse = Obj.fromdense(M)
    self.assertArraysEqual(M, Msparse.todense())

  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
      {"testcase_name": "_{}_Obj={}".format(
        jtu.format_shape_dtype_string(shape, dtype), Obj.__name__),
       "shape": shape, "dtype": dtype, "Obj": Obj}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex)
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_transpose(self, shape, dtype, Obj):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    Msparse = Obj.fromdense(M)
    self.assertArraysEqual(M.T, Msparse.T.todense())

  @unittest.skipIf(jtu.device_under_test() == "tpu", "TPU has insufficient precision")
  @parameterized.named_parameters(itertools.chain.from_iterable(
    jtu.cases_from_list(
      {"testcase_name": "_{}_Obj={}_bshape={}".format(
        jtu.format_shape_dtype_string(shape, dtype), Obj.__name__, bshape),
       "shape": shape, "dtype": dtype, "Obj": Obj, "bshape": bshape}
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [(), (3,), (4,)]]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex)
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_matmul(self, shape, dtype, Obj, bshape):
    rng = rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())
    M = rng(shape, dtype)
    Msp = Obj.fromdense(M)

    # Test matching type
    x = rng_b(bshape, dtype)
    x = jnp.asarray(x)
    self.assertAllClose(M @ x, Msp @ x, rtol=MATMUL_TOL)

    # Test mismatched type
    x = rng_b(bshape, np.int32)
    x = jnp.asarray(x)
    with jax.numpy_dtype_promotion('standard'):
      self.assertAllClose(M @ x, Msp @ x, rtol=MATMUL_TOL)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}({})".format(
        input_type.__name__,
        jtu.format_shape_dtype_string(shape, dtype)),
       "input_type": input_type, "shape": shape, "dtype": dtype}
      for input_type in [scipy.sparse.coo_matrix, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  def test_bcoo_from_scipy_sparse(self, input_type, shape, dtype):
    rng = rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_sparse = input_type(M)
    M_bcoo = sparse.BCOO.from_scipy_sparse(M_sparse)
    self.assertArraysEqual(M, M_bcoo.todense())

  def test_bcoo_methods(self):
    M = jnp.arange(12).reshape(3, 4)
    Msp = sparse.BCOO.fromdense(M)

    self.assertArraysEqual(-M, (-Msp).todense())

    self.assertArraysEqual(2 * M, (2 * Msp).todense())
    self.assertArraysEqual(M * 2, (Msp * 2).todense())

    self.assertArraysEqual(M + M, (Msp + Msp).todense())

    self.assertArraysEqual(M.sum(0), Msp.sum(0).todense())
    self.assertArraysEqual(M.sum(1), Msp.sum(1).todense())
    self.assertArraysEqual(M.sum(), Msp.sum())


class SparseRandomTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_indices_dtype={}_nbatch={}_ndense={}".format(
        jtu.format_shape_dtype_string(shape, dtype), indices_dtype, n_batch, n_dense),
       "shape": shape, "dtype": dtype, "indices_dtype": indices_dtype,
       "n_batch": n_batch, "n_dense": n_dense}
      for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for dtype in jtu.dtypes.floating
      for indices_dtype in jtu.dtypes.integer
      for n_batch in range(len(shape) + 1)
      for n_dense in range(len(shape) + 1 - n_batch)))
  def test_random_bcoo(self, shape, dtype, indices_dtype, n_batch, n_dense):
    key = jax.random.PRNGKey(1701)
    mat = sparse.random_bcoo(
        key, shape=shape, dtype=dtype, indices_dtype=indices_dtype,
        n_batch=n_batch, n_dense=n_dense)

    mat_dense = mat.todense()
    self.assertEqual(mat_dense.shape, shape)
    self.assertEqual(mat_dense.dtype, dtype)
    self.assertEqual(mat.indices.dtype, indices_dtype)

    n_sparse = len(shape) - n_batch - n_dense
    batch_shape, sparse_shape, dense_shape = split_list(shape, [n_batch, n_sparse])

    approx_expected_num_nonzero = (
      np.ceil(0.2 * np.prod(sparse_shape))
      * np.prod(batch_shape) * np.prod(dense_shape))
    num_nonzero = (mat_dense != 0).sum()
    self.assertAlmostEqual(int(num_nonzero), approx_expected_num_nonzero, delta=2)


class SparseSolverTest(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_re{}_({})".format(reorder,
        jtu.format_shape_dtype_string((size, size), dtype)),
       "size": size, "reorder": reorder, "dtype": dtype}
      for size in [20, 50, 100]
      for reorder in [0, 1, 2, 3]
      for dtype in jtu.dtypes.floating + jtu.dtypes.complex))
  @unittest.skipIf(not GPU_LOWERING_ENABLED, "test requires cusparse/cusolver")
  @unittest.skipIf(jtu.device_under_test() != "gpu", "test requires GPU")
  @unittest.skipIf(xla_extension_version < 86, "test requires jaxlib version 86")
  @jtu.skip_on_devices("rocm")
  def test_sparse_qr_linear_solver(self, size, reorder, dtype):
    rng = rand_sparse(self.rng())
    a = rng((size, size), dtype)
    nse = (a != 0).sum()
    data, indices, indptr = sparse.csr_fromdense(a, nse=nse)

    rng_k = jtu.rand_default(self.rng())
    b = rng_k([size], dtype)

    def args_maker():
      return data, indices, indptr, b

    tol = 1e-8
    def sparse_solve(data, indices, indptr, b):
      return sparse.linalg.spsolve(data, indices, indptr, b, tol, reorder)
    x = sparse_solve(data, indices, indptr, b)

    self.assertAllClose(a @ x, b, rtol=1e-2)
    self._CompileAndCheck(sparse_solve, args_maker)


class SparseUtilTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "dtype_{}_nbatch={}_ndense={}_nse={}".format(
          dtype, n_batch, n_dense, expected_nse),
       "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense,
       "expected_nse": expected_nse}
      for n_batch, n_dense, expected_nse in
      [(0, 0, 4), (1, 0, 2), (0, 1, 2), (2, 0, 1), (1, 1, 1), (0, 2, 1)]
      for dtype in all_dtypes))
  def test_count_stored_elements(self, dtype, n_batch, n_dense, expected_nse):
    """Test counting nse."""
    mat = np.array([[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 4]], dtype=dtype)
    actual_nse = sparse.util._count_stored_elements(
        mat, n_batch=n_batch, n_dense=n_dense)
    self.assertEqual(expected_nse, actual_nse)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "dtype_{}_nbatch={}_ndense={}_nse={}".format(
          dtype, n_batch, n_dense, expected_nse),
       "dtype": dtype, "n_batch": n_batch, "n_dense": n_dense,
       "expected_nse": expected_nse}
      for n_batch, n_dense, expected_nse in
      [(0, 0, 14), (1, 0, np.array([6, 8])), (0, 1, 9),
       (2, 0, np.array([[3, 3], [4, 4]]))]
      for dtype in all_dtypes))
  def test_count_stored_elements_per_batch(self, dtype, n_batch, n_dense,
                                           expected_nse):
    """Test counting nse."""
    mat = np.array([[[[1, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 3]],
                     [[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 3]]],
                    [[[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 4]],
                     [[0, 0, 0, 1], [0, 0, 2, 0], [3, 0, 0, 4]]]], dtype=dtype)
    actual_nse = sparse.util._count_stored_elements_per_batch(
        mat, n_batch=n_batch, n_dense=n_dense)
    self.assertArraysEqual(expected_nse, actual_nse)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
