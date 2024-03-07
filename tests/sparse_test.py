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
import math

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.random
from jax import dtypes
from jax.experimental import sparse
from jax.experimental.sparse import coo as sparse_coo
from jax.experimental.sparse import csr as sparse_csr
from jax.experimental.sparse import bcoo as sparse_bcoo
from jax.experimental.sparse import bcsr as sparse_bcsr
from jax.experimental.sparse import util as sparse_util
from jax.experimental.sparse import test_util as sptu
from jax.experimental.sparse import _lowerings
from jax._src import xla_bridge
from jax._src.lib import gpu_sparse
from jax import jit
from jax import vmap
from jax._src import test_util as jtu
from jax.interpreters import mlir
import jax.numpy as jnp
from jax.util import split_list
import numpy as np
import scipy.sparse

jax.config.parse_flags_with_absl()

all_dtypes = jtu.dtypes.integer + jtu.dtypes.floating + jtu.dtypes.complex

class cuSparseTest(sptu.SparseTestCase):
  def gpu_dense_conversion_warning_context(self, dtype):
    if jtu.test_device_matches(["gpu"]) and np.issubdtype(dtype, np.integer):
      return self.assertWarns(sparse.CuSparseEfficiencyWarning)
    return contextlib.nullcontext()

  def gpu_matmul_dtype_warning_context(self, dtype):
    if jtu.test_device_matches(["gpu"]) and dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
      return self.assertWarns(sparse.CuSparseEfficiencyWarning)
    return contextlib.nullcontext()

  @jtu.sample_product(
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=all_dtypes,
  )
  def test_csr_todense(self, shape, dtype):
    rng = sptu.rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.indices, M.indptr)
    todense = lambda *args: sparse_csr._csr_todense(*args, shape=M.shape)

    with self.gpu_dense_conversion_warning_context(dtype):
      self.assertArraysEqual(M.toarray(), todense(*args))
      self.assertArraysEqual(M.toarray(), jit(todense)(*args))

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_csr_todense_ad(self, shape, dtype):
    rng = sptu.rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    data, indices, indptr = sparse_csr._csr_fromdense(M, nse=(M != 0).sum())
    row, col = sparse_util._csr_to_coo(indices, indptr)
    f = lambda data: sparse_csr._csr_todense(data, indices, indptr, shape=M.shape)

    # Forward-mode
    primals, tangents = jax.jvp(f, [data], [jnp.ones_like(data)])
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(tangents, jnp.zeros_like(M).at[row, col].set(1))

    # Reverse-mode
    primals, vjp_fun = jax.vjp(f, data)
    data_out, = vjp_fun(primals)
    self.assertArraysEqual(primals, f(data))
    self.assertArraysEqual(data_out, data)

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_csr_fromdense_ad(self, shape, dtype):
    rng = sptu.rand_sparse(self.rng(), post=jnp.array)
    M = rng(shape, dtype)
    nse = (M != 0).sum()
    f = lambda M: sparse_csr._csr_fromdense(M, nse=nse)

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

  @jtu.sample_product(
    [dict(shape=shape, bshape=bshape)
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [(), (1,), (3,)]]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_csr_matmul_ad(self, shape, dtype, bshape):
    csr_matmul = sparse_csr._csr_matvec if len(bshape) == 1 else sparse_csr._csr_matmat
    tol = {np.float32: 2E-5, np.float64: 1E-12, np.complex64: 1E-5,
           np.complex128: 1E-12}

    rng = sptu.rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())

    M = rng(shape, dtype)
    data, indices, indptr = sparse_csr._csr_fromdense(M, nse=(M != 0).sum())
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
    f_dense = lambda data: sparse_csr._csr_todense(data, indices, indptr, shape=M.shape) @ x
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

  @jtu.sample_product(
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=all_dtypes,
  )
  def test_csr_fromdense(self, shape, dtype):
    rng = sptu.rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_csr = scipy.sparse.csr_matrix(M)

    nse = M_csr.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse_csr._csr_fromdense(M, nse=nse, index_dtype=jnp.int32)

    with self.gpu_dense_conversion_warning_context(dtype):
      data, indices, indptr = fromdense(M)
    self.assertArraysEqual(data, M_csr.data.astype(dtype))
    self.assertArraysEqual(indices, M_csr.indices.astype(index_dtype))
    self.assertArraysEqual(indptr, M_csr.indptr.astype(index_dtype))

    with self.gpu_dense_conversion_warning_context(dtype):
      data, indices, indptr = jit(fromdense)(M)
    self.assertArraysEqual(data, M_csr.data.astype(dtype))
    self.assertArraysEqual(indices, M_csr.indices.astype(index_dtype))
    self.assertArraysEqual(indptr, M_csr.indptr.astype(index_dtype))

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=all_dtypes,
    transpose=[True, False],
  )
  def test_csr_matvec(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    v_rng = jtu.rand_default(self.rng())
    rng = sptu.rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.indices, M.indptr, v)
    matvec = lambda *args: sparse_csr._csr_matvec(*args, shape=M.shape, transpose=transpose)

    with self.gpu_matmul_dtype_warning_context(dtype):
      self.assertAllClose(op(M) @ v, matvec(*args), rtol=sptu.MATMUL_TOL)
      self.assertAllClose(op(M) @ v, jit(matvec)(*args), rtol=sptu.MATMUL_TOL)

  @jtu.sample_product(
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=all_dtypes,
      transpose=[True, False],
  )
  def test_csr_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = sptu.rand_sparse(self.rng(), post=scipy.sparse.csr_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.indices, M.indptr, B)
    matmat = lambda *args: sparse_csr._csr_matmat(*args, shape=shape, transpose=transpose)

    with self.gpu_matmul_dtype_warning_context(dtype):
      self.assertAllClose(op(M) @ B, matmat(*args), rtol=sptu.MATMUL_TOL)
      self.assertAllClose(op(M) @ B, jit(matmat)(*args), rtol=sptu.MATMUL_TOL)

  @jtu.sample_product(
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=all_dtypes,
  )
  def test_coo_todense(self, shape, dtype):
    rng = sptu.rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)

    args = (M.data, M.row, M.col)
    todense = lambda *args: sparse_coo._coo_todense(*args, spinfo=sparse_coo.COOInfo(shape=M.shape, rows_sorted=True))

    with self.gpu_dense_conversion_warning_context(dtype):
      self.assertArraysEqual(M.toarray(), todense(*args))
      self.assertArraysEqual(M.toarray(), jit(todense)(*args))

  @jtu.sample_product(
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=all_dtypes,
  )
  def test_coo_fromdense(self, shape, dtype):
    rng = sptu.rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_coo = scipy.sparse.coo_matrix(M)

    nse = M_coo.nnz
    index_dtype = jnp.int32
    fromdense = lambda M: sparse_coo._coo_fromdense(M, nse=nse, index_dtype=jnp.int32)

    with self.gpu_dense_conversion_warning_context(dtype):
      data, row, col = fromdense(M)
    self.assertArraysEqual(data, M_coo.data.astype(dtype))
    self.assertArraysEqual(row, M_coo.row.astype(index_dtype))
    self.assertArraysEqual(col, M_coo.col.astype(index_dtype))

    with self.gpu_dense_conversion_warning_context(dtype):
      data, row, col = jit(fromdense)(M)
    self.assertArraysEqual(data, M_coo.data.astype(dtype))
    self.assertArraysEqual(row, M_coo.row.astype(index_dtype))
    self.assertArraysEqual(col, M_coo.col.astype(index_dtype))

  @jtu.sample_product(
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=all_dtypes,
      transpose=[True, False],
  )
  def test_coo_matvec(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    v_rng = jtu.rand_default(self.rng())
    rng = sptu.rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)
    v = v_rng(op(M).shape[1], dtype)

    args = (M.data, M.row, M.col, v)
    matvec = lambda *args: sparse_coo._coo_matvec(*args, spinfo=sparse_coo.COOInfo(shape=M.shape, rows_sorted=True), transpose=transpose)

    with self.gpu_matmul_dtype_warning_context(dtype):
      self.assertAllClose(op(M) @ v, matvec(*args), rtol=sptu.MATMUL_TOL)
      self.assertAllClose(op(M) @ v, jit(matvec)(*args), rtol=sptu.MATMUL_TOL)

  @jtu.sample_product(
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=all_dtypes,
      transpose=[True, False],
  )
  def test_coo_matmat(self, shape, dtype, transpose):
    op = lambda M: M.T if transpose else M

    B_rng = jtu.rand_default(self.rng())
    rng = sptu.rand_sparse(self.rng(), post=scipy.sparse.coo_matrix)
    M = rng(shape, dtype)
    B = B_rng((op(M).shape[1], 4), dtype)

    args = (M.data, M.row, M.col, B)
    matmat = lambda *args: sparse_coo._coo_matmat(*args, spinfo=sparse_coo.COOInfo(shape=shape, rows_sorted=True), transpose=transpose)

    with self.gpu_matmul_dtype_warning_context(dtype):
      self.assertAllClose(op(M) @ B, matmat(*args), rtol=sptu.MATMUL_TOL)
      self.assertAllClose(op(M) @ B, jit(matmat)(*args), rtol=sptu.MATMUL_TOL)

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
    sprng = sptu.rand_sparse(rng)

    mat = sparse.COO.fromdense(sprng((5, 6), np.float32))
    perm = rng.permutation(mat.nse)
    mat_unsorted = sparse.COO((mat.data[perm], mat.row[perm], mat.col[perm]), shape=mat.shape)
    mat_resorted = mat_unsorted._sort_indices()
    self.assertArraysEqual(mat.todense(), mat_resorted.todense())

  @jtu.run_on_devices("gpu")
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

  @jtu.run_on_devices("gpu")
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

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_coo_todense_ad(self, shape, dtype):
    rng = sptu.rand_sparse(self.rng(), post=jnp.array)
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

  @jtu.sample_product(
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_coo_fromdense_ad(self, shape, dtype):
    rng = sptu.rand_sparse(self.rng(), post=jnp.array)
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

  @jtu.sample_product(
    [dict(shape=shape, bshape=bshape)
      for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
      for bshape in [shape[-1:] + s for s in [(), (1,), (3,)]]
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def test_coo_matmul_ad(self, shape, dtype, bshape):
    coo_matmul = sparse_coo._coo_matvec if len(bshape) == 1 else sparse_coo._coo_matmat
    tol = {np.float32: 1E-5, np.float64: 1E-12, np.complex64: 1E-5, np.complex128: 1E-12}

    rng = sptu.rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())

    M = rng(shape, dtype)
    data, row, col = sparse_coo._coo_fromdense(M, nse=(M != 0).sum())
    x = rng_b(bshape, dtype)
    xdot = rng_b(bshape, dtype)
    spinfo = sparse_coo.COOInfo(shape=M.shape, rows_sorted=True)

    # Forward-mode with respect to the vector
    f_dense = lambda x: M @ x
    f_sparse = lambda x: coo_matmul(data, row, col, x, spinfo=spinfo)
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
    f_sparse = lambda data: coo_matmul(data, row, col, x, spinfo=spinfo)
    f_dense = lambda data: sparse_coo._coo_todense(data, row, col, spinfo=spinfo) @ x
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

  @jtu.sample_product(
      shape=[(4, 5), (3, 4), (5, 4)],
      dtype=_lowerings.SUPPORTED_DATA_DTYPES,
      transpose=[True, False],
  )
  @jtu.run_on_devices("gpu")
  def test_coo_spmv(self, shape, dtype, transpose):
    rng_sparse = sptu.rand_sparse(self.rng())
    rng_dense = jtu.rand_default(self.rng())

    mat = rng_sparse(shape, dtype)
    vec = rng_dense(shape[0] if transpose else shape[1], dtype)

    row, col = jnp.where(mat != 0)
    data = mat[row, col]

    expected = (mat.T if transpose else mat) @ vec
    actual = _lowerings.coo_spmv_p.bind(
        data, row.astype('int32'), col.astype('int32'), vec,
        transpose=transpose,
        shape=mat.shape)
    self.assertArraysAllClose(actual, expected)

  @jtu.sample_product(
      shape=[(4, 5), (3, 4), (5, 4)],
      dtype=_lowerings.SUPPORTED_DATA_DTYPES,
      transpose=[True, False],
  )
  @jtu.run_on_devices("gpu")
  def test_coo_spmm(self, shape, dtype, transpose):
    rng_sparse = sptu.rand_sparse(self.rng())
    rng_dense = jtu.rand_default(self.rng())

    mat = rng_sparse(shape, dtype)
    vec = rng_dense((shape[0] if transpose else shape[1], 3), dtype)

    row, col = jnp.where(mat != 0)
    data = mat[row, col]

    expected = (mat.T if transpose else mat) @ vec
    actual = _lowerings.coo_spmm_p.bind(
        data, row.astype('int32'), col.astype('int32'), vec,
        transpose=transpose,
        shape=mat.shape)
    self.assertArraysAllClose(actual, expected)

  @jtu.sample_product(
      shape=[(4, 5), (3, 4), (5, 4)],
      dtype=_lowerings.SUPPORTED_DATA_DTYPES,
      transpose=[True, False],
  )
  @jtu.run_on_devices("gpu")
  def test_csr_spmv(self, shape, dtype, transpose):
    rng_sparse = sptu.rand_sparse(self.rng())
    rng_dense = jtu.rand_default(self.rng())

    mat = rng_sparse(shape, dtype)
    data, indices, indptr = sparse_csr._csr_fromdense(mat, nse=(mat != 0).sum())
    vec = rng_dense(shape[0] if transpose else shape[1], dtype)

    expected = (mat.T if transpose else mat) @ vec
    actual = _lowerings.csr_spmv_p.bind(
        data, indices.astype('int32'), indptr.astype('int32'), vec,
        transpose=transpose,
        shape=mat.shape)
    self.assertArraysAllClose(actual, expected)

  @jtu.sample_product(
      shape=[(4, 5), (3, 4), (5, 4)],
      dtype=_lowerings.SUPPORTED_DATA_DTYPES,
      transpose=[True, False],
  )
  @jtu.run_on_devices("gpu")
  def test_csr_spmm(self, shape, dtype, transpose):
    rng_sparse = sptu.rand_sparse(self.rng())
    rng_dense = jtu.rand_default(self.rng())

    mat = rng_sparse(shape, dtype)
    data, indices, indptr = sparse_csr._csr_fromdense(mat, nse=(mat != 0).sum())
    vec = rng_dense((shape[0] if transpose else shape[1], 3), dtype)

    expected = (mat.T if transpose else mat) @ vec
    actual = _lowerings.csr_spmm_p.bind(
        data, indices.astype('int32'), indptr.astype('int32'), vec,
        transpose=transpose,
        shape=mat.shape)
    self.assertArraysAllClose(actual, expected)

class SparseGradTest(sptu.SparseTestCase):
  @jtu.sample_product(has_aux=[True, False])
  def test_sparse_value_and_grad(self, has_aux):
    rng_sparse = sptu.rand_sparse(self.rng())
    rng = jtu.rand_default(self.rng())

    y = rng(5, "float32")
    X = rng_sparse((10, 5), "float32")
    Xsp = sparse.BCOO.fromdense(X)

    def f(X, y):
      if has_aux:
        return jnp.sum(X @ y), {'X': X.shape, 'y': y.shape}
      return jnp.sum(X @ y)

    with self.subTest("wrt sparse"):
      val_de, grad_de = jax.value_and_grad(f, argnums=0, has_aux=has_aux)(X, y)
      val_sp, grad_sp = sparse.value_and_grad(f, argnums=0, has_aux=has_aux)(Xsp, y)
      self.assertIsInstance(grad_sp, sparse.BCOO)
      self.assertAllClose(val_de, val_sp)
      self.assertAllClose(grad_sp.data, sparse_bcoo._bcoo_extract(grad_sp.indices, grad_de))

    with self.subTest("wrt dense"):
      self.assertAllClose(jax.value_and_grad(f, argnums=1, has_aux=has_aux)(X, y),
                          sparse.value_and_grad(f, argnums=1, has_aux=has_aux)(Xsp, y))

  @jtu.sample_product(has_aux=[True, False])
  def test_sparse_grad(self, has_aux):
    rng_sparse = sptu.rand_sparse(self.rng())
    rng = jtu.rand_default(self.rng())

    y = rng(5, "float32")
    X = rng_sparse((10, 5), "float32")
    Xsp = sparse.BCOO.fromdense(X)

    def f(X, y):
      if has_aux:
        return jnp.sum(X @ y), {'X': X.shape, 'y': y.shape}
      return jnp.sum(X @ y)

    with self.subTest("wrt sparse"):
      grad_de = jax.grad(f, argnums=0, has_aux=has_aux)(X, y)
      grad_sp = sparse.grad(f, argnums=0, has_aux=has_aux)(Xsp, y)
      if has_aux:
        grad_de, aux_de = grad_de
        grad_sp, aux_sp = grad_sp
        self.assertAllClose(aux_de, aux_sp)
      self.assertIsInstance(grad_sp, sparse.BCOO)
      self.assertAllClose(grad_sp.data, sparse_bcoo._bcoo_extract(grad_sp.indices, grad_de))

    with self.subTest("wrt dense"):
      self.assertAllClose(jax.grad(f, argnums=1, has_aux=has_aux)(X, y),
                          sparse.grad(f, argnums=1, has_aux=has_aux)(Xsp, y))

  @jtu.sample_product(
    has_aux=[True, False],
    transform=['jacrev', 'jacfwd', 'jacobian']
  )
  @jax.default_matmul_precision("float32")
  def test_sparse_jacobian(self, has_aux, transform):
    jac_dense = getattr(jax, transform)
    jac_sparse = getattr(sparse, transform)

    rng_sparse = sptu.rand_sparse(self.rng())
    rng = jtu.rand_default(self.rng())

    y = rng(5, "float32")
    X = rng_sparse((10, 5), "float32")
    Xsp = sparse.BCOO.fromdense(X)

    def f(X, y):
      if has_aux:
        return X @ y, {'X': X.shape, 'y': y.shape}
      return X @ y

    with self.subTest("wrt sparse"):
      grad_de = jac_dense(f, argnums=0, has_aux=has_aux)(X, y)
      grad_sp = jac_sparse(f, argnums=0, has_aux=has_aux)(Xsp, y)
      if has_aux:
        grad_de, aux_de = grad_de
        grad_sp, aux_sp = grad_sp
        self.assertAllClose(aux_de, aux_sp)
      self.assertIsInstance(grad_sp, sparse.BCOO)
      self.assertAllClose(grad_sp.data, sparse_bcoo._bcoo_extract(grad_sp.indices, grad_de))

    with self.subTest("wrt dense"):
      rtol = 0.01 if jtu.test_device_matches(['tpu']) else None
      self.assertAllClose(jac_dense(f, argnums=1, has_aux=has_aux)(X, y),
                          jac_sparse(f, argnums=1, has_aux=has_aux)(Xsp, y), rtol=rtol)


class SparseObjectTest(sptu.SparseTestCase):
  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}", "cls": cls}
    for cls in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO, sparse.BCSR])
  def test_pytree_flattening(self, cls):
    sparse_format = cls.__name__.lower()
    M = sparse.empty((2, 4), sparse_format=sparse_format)
    self.assertIsInstance(M, cls)
    buffers, tree = jax.tree.flatten(M)
    self.assertTrue(all(isinstance(buffer, jax.Array) for buffer in buffers))
    M_out = jax.tree.unflatten(tree, buffers)
    self.assertEqual(M.dtype, M_out.dtype)
    self.assertEqual(M.shape, M_out.shape)
    self.assertEqual(M.nse, M_out.nse)

  @parameterized.named_parameters(
    {"testcase_name": f"_{cls.__name__}", "cls": cls}
    for cls in [sparse.BCOO, sparse.BCSR])
  def test_vmappable(self, cls):
    # Note: test should avoid dependence on batching rules of BCOO/BCSR primitives
    M = jnp.arange(24).reshape((2, 3, 4))
    Msp = cls.fromdense(M, n_batch=1)

    def from_elt(x):
      assert x.ndim == 2
      return sparse.empty(x.shape, x.dtype, sparse_format=cls.__name__.lower())

    with self.subTest('from_elt'):
      M_out = vmap(from_elt)(M)
      self.assertIsInstance(M_out, cls)
      self.assertEqual(M_out.n_batch, 1)
      self.assertEqual(M.shape, M_out.shape)

    def to_elt(x):
      assert x.ndim == 2
      assert x.n_sparse == 2
      return jnp.empty(x.shape, x.dtype)

    with self.subTest('to_elt'):
      M_out = vmap(to_elt)(Msp)
      self.assertIsInstance(M_out, jax.Array)
      self.assertEqual(Msp.shape, M_out.shape)

    with self.subTest('axis_None'):
      x, y = vmap(lambda *args: args, in_axes=(0, None), out_axes=(0, None))(Msp, Msp)
      self.assertIsInstance(x, cls)
      self.assertEqual(x.n_batch, 1)
      self.assertEqual(x.shape, Msp.shape)
      self.assertEqual(x._info, Msp._info)

      self.assertIsInstance(y, cls)
      self.assertEqual(y.n_batch, 1)
      self.assertEqual(y.shape, Msp.shape)
      self.assertEqual(y._info, Msp._info)

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
    func = partial(sparse.eye, N, M, k, sparse_format=sparse_format)
    expected = jnp.eye(N, M, k)
    expected_nse = jnp.count_nonzero(expected)

    mat = func()
    self.assertIsInstance(mat, cls)
    self.assertArraysEqual(mat.todense(), expected)
    self.assertEqual(mat.nse, expected_nse)

    mat_jit = jit(func)()
    self.assertIsInstance(mat_jit, cls)
    self.assertArraysEqual(mat_jit.todense(), expected)
    self.assertEqual(mat_jit.nse, expected_nse)

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
    rng = sptu.rand_sparse(self.rng(), post=Obj.fromdense)
    M = rng(shape, dtype)
    self.assertEqual(M.shape, M.block_until_ready().shape)
    self.assertArraysEqual(M.data, M.block_until_ready().data)
    self.assertArraysEqual(M.todense(), M.block_until_ready().todense())

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [jnp.array, sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO])
  def test_todense(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = sptu.rand_sparse(self.rng())
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
    rng = sptu.rand_sparse(self.rng())
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
    bufs, tree = jax.tree.flatten(M)
    jac = jnp.eye(M.shape[0], dtype=M.dtype)
    jac1 = jax.jacfwd(lambda *bufs: sparse.todense_p.bind(*bufs, tree=tree))(*bufs)
    jac2 = jax.jacrev(lambda *bufs: sparse.todense_p.bind(*bufs, tree=tree))(*bufs)
    self.assertArraysEqual(jac1, jac2)
    self.assertArraysEqual(jac, jac2)

  @parameterized.named_parameters(
    {"testcase_name": f"_{Obj.__name__}", "Obj": Obj}
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO, sparse.BCSR])
  def test_attrs(self, Obj, shape=(5, 8), dtype=np.float32):
    rng = sptu.rand_sparse(self.rng(), post=Obj.fromdense)
    M = rng(shape, dtype)

    self.assertIsInstance(M, Obj)
    self.assertEqual(M.shape, shape)
    self.assertEqual(M.size, math.prod(shape))
    self.assertEqual(M.ndim, len(shape))
    self.assertEqual(M.dtype, dtype)
    self.assertEqual(M.nse, (M.todense() != 0).sum())
    self.assertEqual(M.data.dtype, dtype)
    self.assertEqual(len(M), M.shape[0])

    with self.assertRaises(TypeError):
      hash(M)

    if isinstance(M, sparse.CSR):
      self.assertEqual(len(M.data), len(M.indices))
      self.assertEqual(len(M.indptr), M.shape[0] + 1)
    elif isinstance(M, sparse.CSC):
      self.assertEqual(len(M.data), len(M.indices))
      self.assertEqual(len(M.indptr), M.shape[1] + 1)
    elif isinstance(M, sparse.COO):
      self.assertEqual(len(M.data), len(M.row))
      self.assertEqual(len(M.data), len(M.col))
    elif isinstance(M, sparse.BCOO):
      self.assertEqual(M.data.shape[M.n_batch], M.indices.shape[-2])
      self.assertEqual(M.indices.shape[-1], M.n_sparse)
    elif isinstance(M, sparse.BCSR):
      self.assertEqual(M.data.shape[M.n_batch], M.indices.shape[-1])
      self.assertEqual(M.indptr.shape[-1], M.shape[M.n_batch] + 1)
    else:
      raise ValueError(f"{Obj=} not expected.")

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      Obj=[Obj],
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    )
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_dense_round_trip(self, shape, dtype, Obj):
    rng = sptu.rand_sparse(self.rng())
    M = rng(shape, dtype)
    Msparse = Obj.fromdense(M)
    self.assertArraysEqual(M, Msparse.todense())

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      Obj=[Obj],
      shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    )
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  def test_transpose(self, shape, dtype, Obj):
    rng = sptu.rand_sparse(self.rng())
    M = rng(shape, dtype)
    Msparse = Obj.fromdense(M)
    self.assertArraysEqual(M.T, Msparse.T.todense())

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(shape=shape, bshape=bshape)
       for shape in [(5, 8), (8, 5), (5, 5), (8, 8)]
       for bshape in [shape[-1:] + s for s in [(), (3,), (4,)]]
      ],
      Obj=[Obj],
      dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    )
    for Obj in [sparse.CSR, sparse.CSC, sparse.COO, sparse.BCOO]))
  @jax.default_matmul_precision("float32")
  def test_matmul(self, shape, dtype, Obj, bshape):
    rng = sptu.rand_sparse(self.rng(), post=jnp.array)
    rng_b = jtu.rand_default(self.rng())
    M = rng(shape, dtype)
    Msp = Obj.fromdense(M)

    # Test matching type
    x = rng_b(bshape, dtype)
    x = jnp.asarray(x)
    self.assertAllClose(
        M @ x, Msp @ x, rtol=sptu.MATMUL_TOL, atol=sptu.MATMUL_TOL
    )

    # Test mismatched type
    x = rng_b(bshape, np.int32)
    x = jnp.asarray(x)
    with jax.numpy_dtype_promotion('standard'):
      self.assertAllClose(M @ x, Msp @ x, rtol=sptu.MATMUL_TOL)

  @jtu.sample_product(
    cls=[sparse.BCOO, sparse.BCSR],
    input_type=[scipy.sparse.coo_matrix, scipy.sparse.csr_matrix,
                scipy.sparse.csc_matrix],
    shape=[(5, 8), (8, 5), (5, 5), (8, 8)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_bcsr_from_scipy_sparse(self, cls, input_type, shape, dtype):
    """Test BCOO and BCSR from_scipy_sparse."""
    rng = sptu.rand_sparse(self.rng())
    M = rng(shape, dtype)
    M_scipy = input_type(M)
    M_jax = cls.from_scipy_sparse(M_scipy)
    self.assertArraysEqual(M, M_jax.todense())

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

    self.assertArraysEqual(M.astype(float), Msp.astype(float).todense())

  @jtu.sample_product(
    [dict(shape=shape, n_batch=n_batch)
      for shape in [(5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
      for n_batch in range(len(shape) - 1)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  def test_bcoo_to_bcsr_round_trip(self, shape, dtype, n_batch):
    rng = sptu.rand_sparse(self.rng())
    M = rng(shape, dtype)
    n_dense = len(shape) - 2 - n_batch
    nse = sparse.util._count_stored_elements(M, n_batch=n_batch,
                                             n_dense=n_dense)
    _, bcoo_indices = sparse_bcoo._bcoo_fromdense(M, nse=nse, n_batch=n_batch,
                                                  n_dense=n_dense)

    bcoo_to_bcsr = partial(sparse_bcsr._bcoo_to_bcsr, shape=shape)

    args_maker_bcoo_to_bcsr = lambda: [bcoo_indices]
    self._CompileAndCheck(bcoo_to_bcsr, args_maker_bcoo_to_bcsr)

    bcsr_indices, indptr = bcoo_to_bcsr(bcoo_indices)

    self.assertEqual(bcsr_indices.dtype, jnp.int32)
    self.assertEqual(bcsr_indices.shape, shape[:n_batch] + (nse,))
    self.assertEqual(indptr.dtype, jnp.int32)
    self.assertEqual(indptr.shape, shape[:n_batch] + (shape[n_batch] + 1,))

    bcsr_to_bcoo = partial(sparse_bcsr._bcsr_to_bcoo, shape=shape)
    self.assertArraysEqual(bcoo_indices, bcsr_to_bcoo(bcsr_indices, indptr))
    args_maker_bcsr_to_bcoo = lambda: [bcsr_indices, indptr]
    self._CompileAndCheck(bcsr_to_bcoo, args_maker_bcsr_to_bcoo)


class SparseRandomTest(sptu.SparseTestCase):

  @jtu.sample_product(
      [
          dict(shape=shape, n_batch=layout.n_batch, n_dense=layout.n_dense)
          for shape in [(5,), (5, 8), (8, 5), (3, 4, 5), (3, 4, 3, 2)]
          for layout in sptu.iter_sparse_layouts(shape)
      ],
      dtype=jtu.dtypes.floating,
      indices_dtype=jtu.dtypes.integer,
  )
  def test_random_bcoo(self, shape, dtype, indices_dtype, n_batch, n_dense):
    key = jax.random.PRNGKey(1701)
    with jax.legacy_prng_key('allow'):
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
      np.ceil(0.2 * math.prod(sparse_shape))
      * math.prod(batch_shape) * math.prod(dense_shape))
    num_nonzero = (mat_dense != 0).sum()
    self.assertAlmostEqual(int(num_nonzero), approx_expected_num_nonzero, delta=2)


class SparseSolverTest(sptu.SparseTestCase):
  @jtu.sample_product(
    size=[20, 50, 100],
    reorder=[0, 1, 2, 3],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jtu.run_on_devices("cpu", "cuda")
  def test_sparse_qr_linear_solver(self, size, reorder, dtype):
    rng = sptu.rand_sparse(self.rng())
    a = rng((size, size), dtype)
    nse = (a != 0).sum()
    data, indices, indptr = sparse_csr._csr_fromdense(a, nse=nse)

    rng_k = jtu.rand_default(self.rng())
    b = rng_k([size], dtype)

    def args_maker():
      return data, indices, indptr, b

    tol = 1e-8
    def sparse_solve(data, indices, indptr, b):
      return sparse.linalg.spsolve(data, indices, indptr, b, tol, reorder)
    x = sparse_solve(data, indices, indptr, b)

    self.assertAllClose(a @ x, b, rtol=1e-2, atol=1e-3)
    self._CompileAndCheck(sparse_solve, args_maker)

  @jtu.sample_product(
    size=[10, 20, 50],
    dtype=jtu.dtypes.floating,
  )
  @jtu.run_on_devices("cpu", "cuda")
  def test_sparse_qr_linear_solver_grads(self, size, dtype):
    rng = sptu.rand_sparse(self.rng())
    a = rng((size, size), dtype)
    nse = (a != 0).sum()
    data, indices, indptr = sparse_csr._csr_fromdense(a, nse=nse)

    rng_k = jtu.rand_default(self.rng())
    b = rng_k([size], dtype)

    def sparse_solve(data, b, tol=1e-8):
      return sparse.linalg.spsolve(data, indices, indptr, b, tol=tol)

    jtu.check_grads(sparse_solve, (data, b), order=1, rtol=0.05, atol=0.05)


class SparseUtilTest(sptu.SparseTestCase):

  @jtu.sample_product(
      [
          dict(n_batch=n_batch, n_dense=n_dense, expected_nse=expected_nse)
          for n_batch, n_dense, expected_nse in [
              (0, 0, 4),
              (1, 0, 2),
              (0, 1, 2),
              (2, 0, 1),
              (1, 1, 1),
              (0, 2, 1),
          ]
      ],
      dtype=all_dtypes,
  )
  def test_count_stored_elements(self, dtype, n_batch, n_dense, expected_nse):
    """Test counting nse."""
    mat = np.array([[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 4]], dtype=dtype)
    actual_nse = sparse.util._count_stored_elements(
        mat, n_batch=n_batch, n_dense=n_dense)
    self.assertEqual(expected_nse, actual_nse)

  @jtu.sample_product(
      [
          dict(n_batch=n_batch, n_dense=n_dense)
          for n_batch in range(3)
          for n_dense in range(3 - n_batch)
      ],
      dtype=all_dtypes,
  )
  def test_count_stored_elements_empty(self, dtype, n_batch, n_dense):
    mat = np.empty((0, 4), dtype=dtype)
    actual_nse = sparse.util._count_stored_elements(
        mat, n_batch=n_batch, n_dense=n_dense)
    self.assertEqual(0, actual_nse)

  @jtu.sample_product(
      [
          dict(n_batch=n_batch, n_dense=n_dense, expected_nse=expected_nse)
          for n_batch, n_dense, expected_nse in [
              (0, 0, 14),
              (1, 0, np.array([6, 8])),
              (0, 1, 9),
              (2, 0, np.array([[3, 3], [4, 4]])),
          ]
      ],
      dtype=all_dtypes,
  )
  def test_count_stored_elements_per_batch(self, dtype, n_batch, n_dense,
                                           expected_nse):
    """Test counting nse."""
    mat = np.array([[[[1, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 3]],
                     [[0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 3]]],
                    [[[1, 0, 2, 0], [0, 0, 0, 0], [0, 3, 0, 4]],
                     [[0, 0, 0, 1], [0, 0, 2, 0], [3, 0, 0, 4]]]], dtype=dtype)
    actual_nse = sparse.util._count_stored_elements_per_batch(
        mat, n_batch=n_batch, n_dense=n_dense)
    self.assertArraysEqual(expected_nse, actual_nse, check_dtypes=False)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
