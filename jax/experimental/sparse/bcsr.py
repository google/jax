# Copyright 2022 Google LLC
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

"""BCSR (Bached compressed row) matrix object and associated primitives."""

import operator
from typing import Tuple

import numpy as np

from jax import core
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.util import _count_stored_elements, _safe_asarray, _csr_to_coo
from jax.experimental.sparse import bcoo as sparse_bcoo
from jax import vmap
import jax.numpy as jnp
from jax.util import safe_zip
from jax.interpreters import mlir

Shape = Tuple[int, ...]

#----------------------------------------------------------------------
# BCSR primitives

def _validate_bcsr(data, indices, indptr, shape):
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  assert jnp.issubdtype(indptr.dtype, jnp.integer)
  shape = tuple(shape)

  nse = indices.shape[-1]
  n_batch = indices.ndim - 1
  n_dense = len(shape) - n_batch - 2
  assert n_dense >= 0

  def _compatible(shape1, shape2):
    return all(s1 in (1, s2) for s1, s2 in safe_zip(shape1, shape2))

  if data is not None:
    if not _compatible(data.shape[:n_batch], shape[:n_batch]):
      raise ValueError("data batch dimensions not compatible for "
                      f"data.shape={data.shape}, shape={shape}")
    if data.shape[-(n_dense + 1):] != (nse,) + shape[n_batch + 2:]:
      raise ValueError(f"Invalid data.shape={data.shape} for "
                      f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")
  if not _compatible(indices.shape[:n_batch], shape[:n_batch]):
    raise ValueError("indices batch dimensions not compatible for "
                     f"indices.shape={indices.shape}, shape={shape}")
  if not _compatible(indptr.shape[:n_batch], shape[:n_batch]):
    raise ValueError("indptr batch dimensions not compatible for "
                     f"indptr.shape={indptr.shape}, shape={shape}")
  if indptr.shape[n_batch:] != (shape[n_batch] + 1,):
    raise ValueError("indptr shape must match the matrix shape plus 1.")

  return n_batch, n_dense

#--------------------------------------------------------------------
# bcsr_fromdense

bcsr_fromdense_p = core.Primitive('bcsr_fromdense')
bcsr_fromdense_p.multiple_results = True

_TRACED_NSE_ERROR = """
The error arose for the nse argument of bcoo_fromdense. In order for
BCOO.fromdense() to be used in traced/compiled code, you must pass a concrete
value to the nse (number of stored elements) argument.
"""

def bcsr_fromdense(mat, *, nse=None, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create BCSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to BCOO.
    nse : number of stored elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of dense dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    mat_bcsr: BCSR representation of the matrix.
  """
  mat = jnp.asarray(mat)
  if nse is None:
    nse = _count_stored_elements(mat, n_batch, n_dense)
  nse = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  # return BCSR(_bcsr_fromdense(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
  #                             index_dtype=index_dtype),
  #             shape=mat.shape)
  # TODO(tianjianlu): Return a BCSR object when `transpose` and `tree_flatten`.
  # are implemented.
  return _bcsr_fromdense(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                         index_dtype=index_dtype)

def _bcsr_fromdense(mat, *, nse, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create BCSR-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to BCSR, with
      ``ndim = n_batch + n_sparse + n_dense``.
    nse : number of stored elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of dense dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    data : array of shape
    ``mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]``
      and dtype ``mat.dtype``
    indices : array of shape ``mat.shape[:n_batch] + (nse,)`` and dtype of
      ``index_type``.
    indptr: array of shape ``mat.shape[:n_batch] + (mat.shape[n_batch] + 1,)``
      and dtype of ``index_type``.
  """
  mat = jnp.asarray(mat)
  nse = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  return bcsr_fromdense_p.bind(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                               index_dtype=index_dtype)

@bcsr_fromdense_p.def_impl
def _bcsr_fromdense_impl(mat, *, nse, n_batch, n_dense, index_dtype):
  mat = jnp.asarray(mat)
  n_sparse = mat.ndim - n_dense - n_batch
  if n_sparse != 2:
    raise ValueError("bcsr_fromdense: must have 2 sparse dimensions.")

  data, bcoo_indices, true_nonzeros = sparse_bcoo._bcoo_fromdense_helper(mat,
      nse=nse, n_batch=n_batch, n_dense=n_dense, index_dtype=index_dtype)

  indices = bcoo_indices[..., 1]

  row_indices = bcoo_indices[..., 0]
  m = mat.shape[n_batch]
  get_ptr = lambda i: jnp.cumsum(jnp.bincount(i, length=m))
  for _ in range(n_batch):
    get_ptr = vmap(get_ptr)
  indptr = jnp.zeros((*mat.shape[:n_batch], m + 1), index_dtype)
  indptr_update = (get_ptr(row_indices) - nse +
                   true_nonzeros.sum(-1, keepdims=True)).astype(index_dtype)
  indptr = indptr.at[..., 1:].set(indptr_update)
  return data, indices, indptr

@bcsr_fromdense_p.def_abstract_eval
def _bcoo_fromdense_abstract_eval(mat, *, nse, n_batch, n_dense, index_dtype):
  n_sparse = mat.ndim - n_batch - n_dense
  if n_sparse != 2:
    raise ValueError("bcsr_fromdense: must have 2 sparse dimensions.")
  data_shape = mat.shape[:n_batch] + (nse,) + mat.shape[n_batch + n_sparse:]
  index_shape = mat.shape[:n_batch] + (nse,)
  indptr_shape = mat.shape[:n_batch] + (mat.shape[n_batch] + 1,)
  return (core.ShapedArray(data_shape, mat.dtype),
          core.ShapedArray(index_shape, index_dtype),
          core.ShapedArray(indptr_shape, index_dtype))

mlir.register_lowering(bcsr_fromdense_p, mlir.lower_fun(
    _bcsr_fromdense_impl, multiple_results=True))

#----------------------------------------------------------------------
# bcsr_todense

bcsr_todense_p = core.Primitive('bcsr_todense')

# TODO(tianjianlu): Take a BCSR matrix as sole input.
def bcsr_todense(data, indices, indptr, *, shape):
  """Convert batched sparse matrix to a dense matrix.
  Args:
    data : array of shape ``batch_dims + (nse,) + block_dims``.
    indices : array of shape ``batch_dims + (nse,)``
    indptr : array of shape ``batch_dims + (shape[len(batch_dims)] + 1,)
    shape : tuple; the shape of the (batched) matrix. Equal to
      ``batch_dims + 2(sparse_dims) + block_dims``
  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcsr_todense_p.bind(jnp.asarray(data), jnp.asarray(indices),
                             jnp.asarray(indptr), shape=tuple(shape))

@bcsr_todense_p.def_impl
def _bcsr_todense_impl(data, indices, indptr, *, shape):
  n_batch, _ = _validate_bcsr(data, indices, indptr, shape)

  if n_batch:
    num_batches = 1
    for i in range(n_batch):
      num_batches = num_batches * shape[i]
    single_batch_indices = jnp.reshape(
        indices, (num_batches, ) + indices.shape[n_batch:])
    single_batch_indptr = jnp.reshape(
        indptr, (num_batches, ) + indptr.shape[n_batch:])
    for _ in range(n_batch):
      csr_to_coo = vmap(_csr_to_coo)
  else:
    single_batch_indices, single_batch_indptr = indices, indptr
    csr_to_coo = _csr_to_coo

  row_indices, col_indices = csr_to_coo(single_batch_indices,
                                        single_batch_indptr)

  if n_batch:
    row_indices = jnp.reshape(row_indices, shape[:n_batch] + row_indices.shape[1:])
    col_indices = jnp.reshape(col_indices, shape[:n_batch] + col_indices.shape[1:])

  bcoo_indices = jnp.stack((row_indices, col_indices), axis=indices.ndim)
  return sparse_bcoo._bcoo_todense(data, bcoo_indices,
                                   spinfo=sparse_bcoo.BCOOInfo(shape))

@bcsr_todense_p.def_abstract_eval
def _bcsr_todense_abstract_eval(data, indices, indptr, *, shape):
  _validate_bcsr(data, indices, indptr, shape)
  return core.ShapedArray(shape, data.dtype)

mlir.register_lowering(bcsr_todense_p, mlir.lower_fun(
    _bcsr_todense_impl, multiple_results=False))


class BCSR(JAXSparse):
  """Experimental batched CSR matrix implemented in JAX."""

  data: jnp.ndarray
  indices: jnp.ndarray
  indptr: jnp.ndarray
  shape: Shape
  nse = property(lambda self: self.indices.shape[-1])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 1)
  n_sparse = property(lambda _: 2)
  n_dense = property(lambda self: self.data.ndim - self.indices.ndim)

  @property
  def _sparse_shape(self):
    return tuple(self.shape[self.n_batch:self.n_batch + 2])

  def __init__(self, args, *, shape):
    # JAX transforms will sometimes instantiate pytrees with null values, so we
    # must catch that in the initialization of inputs.
    self.data, self.indices, self.indptr = _safe_asarray(args)
    super().__init__(args, shape=shape)

  def __repr__(self):
    name = self.__class__.__name__
    try:
      nse = self.nse
      n_batch = self.n_batch
      n_dense = self.n_dense
      dtype = self.dtype
      shape = list(self.shape)
    except Exception:  # pylint: disable=broad-except
      repr_ = f"{name}(<invalid>)"
    else:
      extra = f", nse={nse}"
      if n_batch: extra += f", n_batch={n_batch}"
      if n_dense: extra += f", n_dense={n_dense}"
      repr_ = f"{name}({dtype}{shape}{extra})"
    if isinstance(self.data, core.Tracer):
      repr_ = f"{type(self.data).__name__}[{repr_}]"
    return repr_

  @classmethod
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32, n_dense=0,
                n_batch=0):
    """Create a BCSR array from a (dense) :class:`DeviceArray`."""
    return bcsr_fromdense(mat, nse=nse, index_dtype=index_dtype,
                          n_dense=n_dense, n_batch=n_batch)

  def todense(self):
    """Create a dense version of the array."""
    return bcsr_todense(self.data, self.indices, self.indptr, shape=self.shape)

  def tree_flatten(self):
    # return (self.data, self.indices, self.indptr), self._info._asdict()
    return (self.data, self.indices, self.indptr), {}
