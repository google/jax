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

"""BCOO (Bached coordinate format) matrix object and associated primitives."""

import functools
from functools import partial
import operator
from typing import Any, NamedTuple, Optional, Sequence, Tuple
import warnings

import numpy as np

import jax
from jax import core
from jax import lax
from jax import tree_util
from jax import vmap
from jax.config import config
from jax.experimental.sparse._base import JAXSparse
from jax.experimental.sparse.util import _count_stored_elements, _safe_asarray, CuSparseEfficiencyWarning, SparseEfficiencyError, SparseEfficiencyWarning
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
import jax.numpy as jnp
from jax.interpreters import ad
from jax.util import safe_zip, unzip2, split_list
from jax._src import api_util
from jax._src.api_util import flatten_axes
from jax._src.lax.lax import (
  _const, ranges_like, remaining, _dot_general_batch_dim_nums, _dot_general_shape_rule,
  DotDimensionNumbers)
from jax._src.lib.mlir import ir
from jax._src.lib import xla_bridge
from jax._src.lib import version as jaxlib_version
from jax._src.lib.mlir.dialects import mhlo
from jax._src.numpy.setops import _unique

from jax._src.lib import gpu_sparse

Dtype = Any
Shape = Tuple[int, ...]

#----------------------------------------------------------------------
# General utilities...
def broadcasting_vmap(fun, in_axes=0, out_axes=0):
  @functools.wraps(fun)
  def batched_fun(*args):
    args_flat, in_tree  = tree_util.tree_flatten(args)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, in_axes, kws=False)
    size = max(arg.shape[i] for arg, i in safe_zip(args_flat, in_axes_flat) if i is not None)
    if size > 1:
      if any(i is not None and arg.shape[i] not in (1, size)
             for arg, i in safe_zip(args_flat, in_axes_flat)):
        raise ValueError("broadcasting_vmap: mismatched input shapes")
      args_flat, in_axes_flat = zip(*(
          (arg, None) if i is None else (lax.squeeze(arg, (i,)), None) if arg.shape[i] == 1 else (arg, i)
          for arg, i in zip(args_flat, in_axes_flat)
      ))
    new_args = tree_util.tree_unflatten(in_tree, args_flat)
    new_in_axes = tree_util.tree_unflatten(in_tree, in_axes_flat)
    return vmap(fun, in_axes=new_in_axes, out_axes=out_axes)(*new_args)
  return batched_fun

#----------------------------------------------------------------------
# BCOO primitives: batched extension of COO.

def _bcoo_set_nse(mat, nse):
  """Return a copy of `mat` with the specified nse.
  Note that if nse < mat.nse, this will potentially discard data.
  """
  nse = operator.index(nse)
  assert nse >= 0
  if mat.nse == nse:
    return mat
  if nse <= mat.nse:
    data = mat.data[(*(slice(None) for i in range(mat.n_batch)), slice(nse))]
    indices = mat.indices[..., :nse, :]
  else:
    data = jnp.zeros_like(mat.data, shape=(*mat.data.shape[:mat.n_batch], nse, *mat.data.shape[mat.n_batch + 1:]))
    data = data.at[(*(slice(None) for i in range(mat.n_batch)), slice(mat.nse))].set(mat.data)
    indices = jnp.zeros_like(mat.indices, shape=(*mat.indices.shape[:-2], nse, mat.indices.shape[-1]))
    indices = indices.at[..., :mat.nse, :].set(mat.indices)
    indices = indices.at[..., mat.nse:, :].set(jnp.array(mat.shape[mat.n_batch:mat.n_batch + mat.n_sparse],
                                                         dtype=indices.dtype))
  return BCOO((data, indices), shape=mat.shape,
              indices_sorted=mat.indices_sorted,
              unique_indices=mat.unique_indices)

# TODO(jakevdp) this can be problematic when used with autodiff; see
# https://github.com/google/jax/issues/10163. Should this be a primitive?
# Alternatively, maybe roll this into bcoo_sum_duplicates as an optional argument.
def bcoo_eliminate_zeros(mat, nse=None):
  data, indices, shape = mat.data, mat.indices, mat.shape
  props = _validate_bcoo(data, indices, shape)
  mask = (data == 0).all(tuple(range(props.n_batch + 1, data.ndim)))
  dims_to_contract = tuple(i for i, s in enumerate(indices.shape[:props.n_batch]) if s == 1)
  mask = mask.all(dims_to_contract, keepdims=True)
  fill_value = jnp.array(shape[props.n_batch:props.n_batch + props.n_sparse], dtype=indices.dtype)
  f = lambda i, m: jnp.where(m[:, None], fill_value[None, :], i)
  for _ in range(props.n_batch):
    f = vmap(f)
  indices = f(indices, mask)
  return bcoo_sum_duplicates(BCOO((data, indices), shape=shape), nse=nse)

def _unbatch_bcoo(data, indices, shape):
  mat = bcoo_update_layout(BCOO((data, indices), shape=shape), n_batch=0)
  return mat.data, mat.indices


class BCOOProperties(NamedTuple):
  n_batch: int
  n_sparse: int
  n_dense: int
  nse: int

class BCOOInfo(NamedTuple):
  shape: Shape
  indices_sorted: bool = False
  unique_indices: bool = False

def _validate_bcoo(data: jnp.ndarray, indices: jnp.ndarray, shape: Sequence[int]) -> BCOOProperties:
  props = _validate_bcoo_indices(indices, shape)
  n_batch, n_sparse, n_dense, nse = props
  shape = tuple(shape)
  if any(s1 not in (1, s2) for s1, s2 in safe_zip(data.shape[:n_batch], shape[:n_batch])):
    raise ValueError("data batch dimensions not compatible for "
                     f"data.shape={data.shape}, shape={shape}")
  if data.shape[n_batch:] != (nse,) + shape[n_batch + n_sparse:]:
    raise ValueError(f"Invalid data.shape={data.shape} for "
                    f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")
  return props


def _validate_bcoo_indices(indices: jnp.ndarray, shape: Sequence[int]) -> BCOOProperties:
  assert jnp.issubdtype(indices.dtype, jnp.integer)
  shape = tuple(shape)
  nse, n_sparse = indices.shape[-2:]
  n_batch = indices.ndim - 2
  n_dense = len(shape) - n_batch - n_sparse
  assert n_dense >= 0
  if any(s1 not in (1, s2) for s1, s2 in safe_zip(indices.shape[:n_batch], shape[:n_batch])):
    raise ValueError("indices batch dimensions not compatible for "
                     f"indices.shape={indices.shape}, shape={shape}")
  if indices.shape[n_batch:] != (nse, n_sparse):
    raise ValueError(f"Invalid indices.shape={indices.shape} for "
                     f"nse={nse}, n_batch={n_batch}, n_dense={n_dense}")
  return BCOOProperties(n_batch=n_batch, n_sparse=n_sparse, n_dense=n_dense, nse=nse)


#----------------------------------------------------------------------
# bcoo_todense

bcoo_todense_p = core.Primitive('bcoo_todense')

def bcoo_todense(mat):
  """Convert batched sparse matrix to a dense matrix.

  Args:
    mat: BCOO matrix.

  Returns:
    mat_dense: dense version of ``mat``.
  """
  return _bcoo_todense(mat.data, mat.indices, spinfo=mat._info)

def _bcoo_todense(data, indices, *, spinfo):
  """Convert batched sparse matrix to a dense matrix.

  Args:
    data : array of shape ``batch_dims + (nse,) + block_dims``.
    indices : array of shape ``batch_dims + (n_sparse, nse)``
    spinfo : BCOOInfo. In particular, this includes the shape
      of the matrix, which is equal to ``batch_dims + sparse_dims + block_dims``
      where ``len(sparse_dims) == n_sparse``

  Returns:
    mat : array with specified shape and dtype matching ``data``
  """
  return bcoo_todense_p.bind(jnp.asarray(data), jnp.asarray(indices), spinfo=spinfo)

@bcoo_todense_p.def_impl
def _bcoo_todense_impl(data, indices, *, spinfo):
  shape = spinfo.shape
  n_batch, n_sparse, _, _ = _validate_bcoo(data, indices, shape)

  ind_slices = tuple(np.zeros(s, int) if i_s == 1 else np.arange(s)
                     for s, i_s in zip(shape[:n_batch], indices.shape[:n_batch]))
  grid = tuple(np.meshgrid(*ind_slices, indexing='ij', sparse=True))
  sparse_ind = tuple(indices[grid + (slice(None), i)] for i in range(n_sparse))

  batch_slices = tuple(np.arange(s) for s in shape[:n_batch])
  grid = np.meshgrid(*batch_slices, np.arange(1), indexing='ij', sparse=True)
  batch_ind = tuple(grid)[:-1]

  if not sparse_ind:
    data = data.sum(n_batch, keepdims=bool(batch_ind), dtype=data.dtype)
  return jnp.zeros(shape, data.dtype).at[batch_ind + sparse_ind].add(data)

@bcoo_todense_p.def_abstract_eval
def _bcoo_todense_abstract_eval(data, indices, *, spinfo):
  shape = spinfo.shape
  _validate_bcoo(data, indices, shape)
  return core.ShapedArray(shape, data.dtype)

def _bcoo_todense_jvp(data_dot, data, indices, *, spinfo):
  return _bcoo_todense(data_dot, indices, spinfo=spinfo)

def _bcoo_todense_transpose(ct, data, indices, *, spinfo):
  shape = spinfo.shape
  assert ad.is_undefined_primal(data)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.shape == shape
  assert ct.dtype == data.aval.dtype
  return bcoo_extract(indices, ct), indices

def _bcoo_todense_batching_rule(batched_args, batch_dims, *, spinfo):
  data, indices = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"batch_dims={batch_dims}. Only 0 and None are supported.")
  if batch_dims[0] is None:
    data = data[None, ...]
  if batch_dims[1] is None:
    indices = indices[None, ...]
  new_spinfo = BCOOInfo(
      shape=(max(data.shape[0], indices.shape[0]), *spinfo.shape),
      indices_sorted=spinfo.indices_sorted,
      unique_indices=spinfo.unique_indices)
  return _bcoo_todense(data, indices, spinfo=new_spinfo), 0

ad.defjvp(bcoo_todense_p, _bcoo_todense_jvp, None)
ad.primitive_transposes[bcoo_todense_p] = _bcoo_todense_transpose
batching.primitive_batchers[bcoo_todense_p] = _bcoo_todense_batching_rule
mlir.register_lowering(bcoo_todense_p, mlir.lower_fun(
    _bcoo_todense_impl, multiple_results=False))

#--------------------------------------------------------------------
# bcoo_fromdense

bcoo_fromdense_p = core.Primitive('bcoo_fromdense')
bcoo_fromdense_p.multiple_results = True

_TRACED_NSE_ERROR = """
The error arose for the nse argument of bcoo_fromdense. In order for
BCOO.fromdense() to be used in traced/compiled code, you must pass a concrete
value to the nse (number of stored elements) argument.
"""

def bcoo_fromdense(mat, *, nse=None, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create BCOO-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to BCOO.
    nse : number of stored elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of dense dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    mat_bcoo: BCOO representation of the matrix.
  """
  mat = jnp.asarray(mat)
  if nse is None:
    nse = _count_stored_elements(mat, n_batch, n_dense)
  nse = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  return BCOO(_bcoo_fromdense(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                              index_dtype=index_dtype),
              shape=mat.shape, indices_sorted=True, unique_indices=True)

def _bcoo_fromdense(mat, *, nse, n_batch=0, n_dense=0, index_dtype=jnp.int32):
  """Create BCOO-format sparse matrix from a dense matrix.

  Args:
    mat : array to be converted to BCOO, with ``ndim = n_batch + n_sparse + n_dense``.
    nse : number of stored elements in each batch
    n_batch : number of batch dimensions (default: 0)
    n_dense : number of dense dimensions (default: 0)
    index_dtype : dtype of sparse indices (default: int32)

  Returns:
    data : array of shape ``mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]``
      and dtype ``mat.dtype``
    indices : array of shape ``mat.shape[:n_batch] + (n_sparse, nse)``
  """
  mat = jnp.asarray(mat)
  nse = core.concrete_or_error(operator.index, nse, _TRACED_NSE_ERROR)
  return bcoo_fromdense_p.bind(mat, nse=nse, n_batch=n_batch, n_dense=n_dense,
                               index_dtype=index_dtype)

def _bcoo_fromdense_helper(mat, *, nse, n_batch, n_dense, index_dtype):
  mat = jnp.asarray(mat)
  mask = (mat != 0)
  if n_dense > 0:
    mask = mask.any([-(i + 1) for i in range(n_dense)])
  nonzero = lambda a: jnp.nonzero(a, size=nse) if a.ndim else ()
  for _ in range(n_batch):
    nonzero = vmap(nonzero, 0)
  indices = nonzero(mask)
  if not indices:
    indices = jnp.zeros(mask.shape[:n_batch] + (nse, 0), index_dtype)
  else:
    indices = jnp.moveaxis(jnp.array(indices, index_dtype), 0, n_batch + 1)
  data = bcoo_extract(indices, mat)

  # TODO(tianjianlu): Use expand_dims instead of broadcast_arrays.
  lhs_to_compare = jnp.arange(nse)
  rhs_to_compare = mask.sum(list(range(n_batch, mask.ndim)))[..., None]
  lhs_to_compare, rhs_to_compare = jnp.broadcast_arrays(lhs_to_compare,
                                                        rhs_to_compare)
  true_nonzeros = lhs_to_compare < rhs_to_compare
  data = jnp.where(true_nonzeros[(...,) + n_dense * (None,)], data, 0)

  return data, indices, true_nonzeros

@bcoo_fromdense_p.def_impl
def _bcoo_fromdense_impl(mat, *, nse, n_batch, n_dense, index_dtype):
  data, indices, _ = _bcoo_fromdense_helper(mat,
      nse=nse, n_batch=n_batch, n_dense=n_dense, index_dtype=index_dtype)
  return data, indices

@bcoo_fromdense_p.def_abstract_eval
def _bcoo_fromdense_abstract_eval(mat, *, nse, n_batch, n_dense, index_dtype):
  n_sparse = mat.ndim - n_batch - n_dense
  data_shape = mat.shape[:n_batch] + (nse,) + mat.shape[n_batch + n_sparse:]
  index_shape = mat.shape[:n_batch] + (nse, n_sparse)
  return core.ShapedArray(data_shape, mat.dtype), core.ShapedArray(index_shape, index_dtype)

def _bcoo_fromdense_jvp(primals, tangents, *, nse, n_batch, n_dense, index_dtype):
  M, = primals
  Mdot, = tangents

  primals_out = _bcoo_fromdense(M, nse=nse, n_batch=n_batch, n_dense=n_dense, index_dtype=index_dtype)
  data, indices = primals_out

  if type(Mdot) is ad.Zero:
    data_dot = ad.Zero.from_value(data)
  else:
    data_dot = bcoo_extract(indices, Mdot)

  tangents_out = (data_dot, ad.Zero.from_value(indices))

  return primals_out, tangents_out

def _bcoo_fromdense_transpose(ct, M, *, nse, n_batch, n_dense, index_dtype):
  data, indices = ct
  n_sparse = M.ndim = n_batch - n_dense
  assert data.shape == M.shape[:n_batch] + (nse,) + M.shape[n_batch + n_sparse:]
  assert indices.shape == M.shape[:n_batch] + (n_sparse, nse)
  assert indices.dtype == index_dtype
  if isinstance(indices, ad.Zero):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ad.is_undefined_primal(M)
  return _bcoo_todense(data, indices, spinfo=BCOOInfo(M.aval.shape))

def _bcoo_fromdense_batching_rule(batched_args, batch_dims, *, nse, n_batch, n_dense, index_dtype):
  M, = batched_args
  if batch_dims != (0,):
    raise NotImplementedError(f"batch_dims={batch_dims}")
  return _bcoo_fromdense(M, nse=nse, n_batch=n_batch + 1, n_dense=n_dense, index_dtype=index_dtype), (0, 0)

ad.primitive_jvps[bcoo_fromdense_p] = _bcoo_fromdense_jvp
ad.primitive_transposes[bcoo_fromdense_p] = _bcoo_fromdense_transpose
batching.primitive_batchers[bcoo_fromdense_p] = _bcoo_fromdense_batching_rule
mlir.register_lowering(bcoo_fromdense_p, mlir.lower_fun(
    _bcoo_fromdense_impl, multiple_results=True))

#----------------------------------------------------------------------
# bcoo_extract

bcoo_extract_p = core.Primitive('bcoo_extract')

def bcoo_extract(indices, mat):
  """Extract BCOO data values from a dense matrix at given BCOO indices.

  Args:
    indices: An ndarray; see BCOO indices.
    mat: A dense matrix.

  Returns:
    An ndarray; see BCOO data.
  """
  return bcoo_extract_p.bind(indices, mat)

@bcoo_extract_p.def_impl
def _bcoo_extract_impl(indices, mat):
  mat = jnp.asarray(mat)
  n_batch, n_sparse, _, _ = _validate_bcoo_indices(indices, mat.shape)

  ind_slices = tuple(np.zeros(s, int) if i_s == 1 else np.arange(s)
                     for s, i_s in zip(mat.shape[:n_batch], indices.shape[:n_batch]))
  grid = tuple(np.meshgrid(*ind_slices, indexing='ij', sparse=True))
  sparse_ind = tuple(indices[grid + (slice(None), i)] for i in range(n_sparse))

  batch_slices = tuple(np.arange(s) for s in mat.shape[:n_batch])
  grid = np.meshgrid(*batch_slices, np.arange(1), indexing='ij', sparse=True)
  batch_ind = tuple(grid)[:-1]

  if not sparse_ind + batch_ind:
    return mat[None]
  return mat.at[batch_ind + sparse_ind].get(mode='fill', fill_value=0)

@bcoo_extract_p.def_abstract_eval
def _bcoo_extract_abstract_eval(indices, mat):
  n_batch, _, n_dense, nse = _validate_bcoo_indices(indices, mat.shape)
  out_shape = mat.shape[:n_batch] + (nse,) + mat.shape[mat.ndim - n_dense:]
  return core.ShapedArray(out_shape, mat.dtype)

def _bcoo_extract_jvp(mat_dot, indices, mat):
  assert mat_dot.shape == mat.shape
  return bcoo_extract(indices, mat_dot)

def _bcoo_extract_transpose(ct, indices, mat):
  assert ad.is_undefined_primal(mat)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert ct.dtype == mat.aval.dtype
  return indices, _bcoo_todense(ct, indices, spinfo=BCOOInfo(mat.aval.shape))

def _bcoo_extract_batching_rule(batched_args, batch_dims):
  indices, mat = batched_args
  assert any(b is not None for b in batch_dims)
  if batch_dims[0] is None:
    bdim = batch_dims[1]
    indices = lax.expand_dims(indices, (bdim,))
  elif batch_dims[1] is None:
    # TODO(jakevdp) can we handle this case without explicit broadcasting?
    bdim = batch_dims[0]
    result_shape = list(mat.shape)
    result_shape.insert(bdim, indices.shape[bdim])
    mat = lax.broadcast_in_dim(mat, result_shape, (bdim,))
  else:
    if batch_dims[0] != batch_dims[1]:
      raise NotImplementedError("bcoo_extract with unequal batch dimensions.")
    bdim = batch_dims[0]
  n_batch = indices.ndim - 2
  if bdim >= n_batch:
    raise ValueError(f"batch_dims={batch_dims} out of range for indices with n_batch={n_batch}")
  return bcoo_extract(indices, mat), bdim

ad.defjvp(bcoo_extract_p, None, _bcoo_extract_jvp)
ad.primitive_transposes[bcoo_extract_p] = _bcoo_extract_transpose
batching.primitive_batchers[bcoo_extract_p] = _bcoo_extract_batching_rule
mlir.register_lowering(bcoo_extract_p, mlir.lower_fun(
    _bcoo_extract_impl, multiple_results=False))

#----------------------------------------------------------------------
# bcoo_transpose
# transpose of a BCOO array

bcoo_transpose_p = core.Primitive('bcoo_transpose')
bcoo_transpose_p.multiple_results = True

def bcoo_transpose(mat, *, permutation: Sequence[int]):
  """Transpose a BCOO-format array.

  Args:
    mat: A BCOO-format array.
    permutation:  A tuple or list or ndarray which contains a permutation of
      [0,1,..,N-1] where N is the number of axes of ``mat`` in the order of
      batch, sparse, and dense dimensions. The i’th axis of the returned array
      corresponds to the axis numbered permutation[i] of ``mat``. Transpose
      permutation currently does not support permuting batch axes with non-batch
      axes nor permutating dense axes with non-dense axes.

  Returns:
    A BCOO-format array.
  """
  return BCOO(_bcoo_transpose(mat.data, mat.indices, permutation=permutation, spinfo=mat._info),
              shape=mat._info.shape, unique_indices=mat.unique_indices)

def _bcoo_transpose(data, indices, *, permutation: Sequence[int], spinfo: BCOOInfo):
  permutation = tuple(permutation)
  if permutation == tuple(range(len(spinfo.shape))):
    return data, indices
  else:
    return bcoo_transpose_p.bind(data, indices, permutation=permutation,
                                 spinfo=spinfo)

def _validate_permutation(data, indices, permutation, shape):
  if not isinstance(permutation, (tuple, list, np.ndarray)):
    raise TypeError(f"transpose permutation must be a tuple/list/ndarray, got {type(permutation)}.")
  if tuple(sorted(permutation)) != tuple(range(len(shape))):
    raise TypeError("transpose permutation isn't a permutation of operand dimensions, "
                    f"got permutation {permutation} for shape {shape}.")
  n_batch, n_sparse, n_dense, _ = _validate_bcoo(data, indices, shape)
  batch_perm = permutation[:n_batch]
  sparse_perm = [p - n_batch for p in permutation[n_batch: n_batch + n_sparse]]
  dense_perm = [p - n_sparse - n_batch for p in permutation[n_batch + n_sparse:]]
  if n_batch and tuple(sorted(batch_perm)) != tuple(range(n_batch)):
    raise NotImplementedError("transpose permutation cannot permute batch axes with non-batch axes; "
                              f"got permutation {permutation}, with n_batch={n_batch}.")
  if n_dense and tuple(sorted(dense_perm)) != tuple(range(n_dense)):
    raise NotImplementedError("transpose permutation cannot permute dense axes with non-dense axes; "
                              f"got permutation {permutation}, with n_dense={n_dense}.")
  return batch_perm, sparse_perm, dense_perm

@bcoo_transpose_p.def_impl
def _bcoo_transpose_impl(data, indices, *, permutation: Sequence[int], spinfo: BCOOInfo):
  batch_perm, sparse_perm, dense_perm = _validate_permutation(data, indices, permutation, spinfo.shape)
  n_batch = len(batch_perm)
  indices = indices[..., sparse_perm].transpose(*batch_perm, n_batch, n_batch + 1)
  data = data.transpose(*batch_perm, n_batch, *(d + n_batch + 1 for d in dense_perm))
  return data, indices

@bcoo_transpose_p.def_abstract_eval
def _bcoo_transpose_abstract_eval(data, indices, *, permutation: Sequence[int], spinfo: BCOOInfo):
  batch_perm, _, dense_perm = _validate_permutation(data, indices, permutation, spinfo.shape)
  n_batch = len(batch_perm)
  indices_shape = np.array(indices.shape)[[*batch_perm, n_batch, n_batch + 1]]
  data_shape = np.array(data.shape)[[*batch_perm, n_batch, *(d + n_batch + 1 for d in dense_perm)]]
  return core.ShapedArray(data_shape, data.dtype), core.ShapedArray(indices_shape, indices.dtype)

def _bcoo_transpose_jvp(primals, tangents, *, permutation: Sequence[int], spinfo: BCOOInfo):
  data, indices = primals
  data_dot, _ = tangents
  primals_out = _bcoo_transpose(data, indices, permutation=permutation, spinfo=spinfo)
  data_dot_out, _ = _bcoo_transpose(data_dot, indices, permutation=permutation, spinfo=spinfo)
  return primals_out, (data_dot_out, ad.Zero.from_value(indices))

def _bcoo_transpose_transpose(ct, data, indices, *, permutation: Sequence[int], spinfo: BCOOInfo):
  data_ct, indices_ct = ct
  assert isinstance(indices_ct, ad.Zero)
  if ad.is_undefined_primal(indices):
    raise ValueError("Cannot transpose with respect to sparse indices")
  assert data_ct.dtype == data.aval.dtype
  ct_spinfo = BCOOInfo(tuple(spinfo.shape[p] for p in permutation))
  rev_permutation = list(np.argsort(permutation))
  # TODO(jakevdp) avoid dummy indices?
  dummy_indices = jnp.zeros([1 for i in range(indices.ndim - 2)] + list(indices.shape[-2:]), dtype=int)
  data_trans, _ = _bcoo_transpose(data_ct, dummy_indices, permutation=rev_permutation, spinfo=ct_spinfo)
  return data_trans, indices_ct

def _bcoo_transpose_batch_rule(batched_args, batch_dims, *, permutation: Sequence[int], spinfo: BCOOInfo):
  data, indices = batched_args
  batch_dims = list(batch_dims)
  batch_size = max(0 if dim is None else arg.shape[dim]
                   for arg, dim in zip(batched_args, batch_dims))
  if batch_dims[0] is None:
    data = data[None]
  else:
    assert batch_dims[0] == 0
  if batch_dims[1] is None:
    indices = indices[None]
  else:
    assert batch_dims[1] == 0
  batched_spinfo = BCOOInfo((batch_size, *spinfo.shape))
  batched_permutation = (0, *(p + 1 for p in permutation))
  data, indices = _bcoo_transpose(data, indices, permutation=batched_permutation, spinfo=batched_spinfo)
  if batch_dims[0] is None:
    data = data[0]
  if batch_dims[1] is None:
    indices = indices[0]
  return (data, indices), batch_dims

ad.primitive_jvps[bcoo_transpose_p] = _bcoo_transpose_jvp
ad.primitive_transposes[bcoo_transpose_p] = _bcoo_transpose_transpose
batching.primitive_batchers[bcoo_transpose_p] = _bcoo_transpose_batch_rule
mlir.register_lowering(bcoo_transpose_p, mlir.lower_fun(
    _bcoo_transpose_impl, multiple_results=True))

#----------------------------------------------------------------------
# bcoo_dot_general
# (batched) general dot product of a BCOO sparse ND array and a dense ND array,
# returning a dense ND array.

bcoo_dot_general_p = core.Primitive('bcoo_dot_general')

def _dot_general_validated_shape(lhs_shape: Shape, rhs_shape: Shape, dimension_numbers: DotDimensionNumbers) -> Shape:
  """Validate the inputs and return the output shape."""
  lhs = core.ShapedArray(lhs_shape, np.float32)
  rhs = core.ShapedArray(rhs_shape, np.float32)
  return _dot_general_shape_rule(
    lhs, rhs, dimension_numbers=dimension_numbers,
    precision=None, preferred_element_type=None)

def bcoo_dot_general(lhs, rhs, *, dimension_numbers):
  """A general contraction operation.

  Args:
    lhs: An ndarray or BCOO-format sparse array.
    rhs: An ndarray or BCOO-format sparse array..
    dimension_numbers: a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims),
      (lhs_batch_dims, rhs_batch_dims))`.

  Returns:
    An ndarray or BCOO-format sparse array containing the result. If both inputs
    are sparse, the result will be sparse, of type BCOO. If either input is dense,
    the result will be dense, of type ndarray.
  """
  if isinstance(lhs, BCOO) and isinstance(rhs, BCOO):
    shape = _dot_general_validated_shape(lhs.shape, rhs.shape, dimension_numbers)
    bufs = _bcoo_spdot_general(lhs.data, lhs.indices, rhs.data, rhs.indices,
                               lhs_spinfo=lhs._info, rhs_spinfo=rhs._info,
                               dimension_numbers=dimension_numbers)
    return BCOO(bufs, shape=shape)
  elif isinstance(lhs, BCOO):
    return _bcoo_dot_general(*lhs._bufs, rhs, dimension_numbers=dimension_numbers,
                             lhs_spinfo=lhs._info)
  elif isinstance(rhs, BCOO):
    return _bcoo_rdot_general(lhs, *rhs._bufs, dimension_numbers=dimension_numbers,
                              rhs_spinfo=rhs._info)
  else:
    return lax.dot_general(lhs, rhs, dimension_numbers=dimension_numbers)

def _bcoo_dot_general(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  return bcoo_dot_general_p.bind(jnp.asarray(lhs_data), jnp.asarray(lhs_indices), jnp.asarray(rhs),
                                 dimension_numbers=(cdims, bdims),
                                 lhs_spinfo=lhs_spinfo)

def _bcoo_rdot_general(lhs, rhs_data, rhs_indices, *, dimension_numbers: DotDimensionNumbers, rhs_spinfo: BCOOInfo):
  # TODO(jakevdp): perhaps this should be part of the bcoo_dot_general primitive?
  result = _bcoo_dot_general(rhs_data, rhs_indices, lhs, lhs_spinfo=rhs_spinfo,
                             dimension_numbers=[d[::-1] for d in dimension_numbers])
  n_contract, n_batch = (len(d[0]) for d in dimension_numbers)
  n_swap = len(rhs_spinfo.shape) - n_contract
  permutation = tuple([*range(n_batch), *range(n_swap, result.ndim), *range(n_batch, n_swap)])
  return lax.transpose(result, permutation)

@bcoo_dot_general_p.def_impl
def _bcoo_dot_general_impl(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  lhs_data = jnp.asarray(lhs_data)
  lhs_indices = jnp.asarray(lhs_indices)
  rhs = jnp.asarray(rhs)
  # Validate all inputs via abstract_eval
  out_aval = _bcoo_dot_general_abstract_eval(lhs_data.aval, lhs_indices.aval, rhs.aval,
                                             dimension_numbers=dimension_numbers,
                                             lhs_spinfo=lhs_spinfo)
  n_sparse = lhs_indices.shape[-1]
  n_batch = lhs_indices.ndim - 2

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_contracting_b, rhs_contracting_b = unzip2([
    (l, r) for l, r in safe_zip(lhs_contracting, rhs_contracting) if l < n_batch])
  lhs_contracting_s, rhs_contracting_s = unzip2([
    (l, r) for l, r in safe_zip(lhs_contracting, rhs_contracting) if l >= n_batch])

  # Reorder lhs batch dimensions
  if lhs_batch or lhs_contracting_b:
    batch_perm = [*lhs_batch, *remaining(range(n_batch), lhs_batch, lhs_contracting_b), *lhs_contracting_b]
    lhs_data = lhs_data.transpose([*batch_perm, *range(n_batch, lhs_data.ndim)])
    lhs_indices = lhs_indices.transpose([*batch_perm, *range(n_batch, lhs_indices.ndim)])

  # Reorder lhs sparse dimensions
  if lhs_contracting_s:
    lhs_contracting_s = [d - n_batch for d in lhs_contracting_s]
    sparse_perm = jnp.array([*lhs_contracting_s, *remaining(range(n_sparse), lhs_contracting_s)])
    lhs_indices = lhs_indices[..., sparse_perm]

  # Reorder rhs dimensions
  rhs_perm = [*rhs_batch, *rhs_contracting_b, *rhs_contracting_s,
              *remaining(range(rhs.ndim), rhs_batch, rhs_contracting)]
  rhs = rhs.transpose(rhs_perm)

  def result(out_array, lhs_data, lhs_indices, rhs):
    idx = tuple(lhs_indices[..., i] for i in range(n_sparse))
    idx_right = idx[:len(lhs_contracting_s)]
    idx_out = idx[len(lhs_contracting_s):]
    if idx_right and lhs_indices.ndim > 2:
      idx_batch = jnp.meshgrid(
          *(jnp.arange(n) for n in lhs_indices.shape[:-1]),
          indexing='ij')[:lhs_indices.ndim - 2]
      idx_right = (*idx_batch, *idx_right)
    batch_dims = list(range(len(lhs_contracting_b) + bool(lhs_contracting_s)))
    prod = lax.dot_general(lhs_data, rhs.at[idx_right].get(mode='fill', fill_value=0),
                           (([], []), (batch_dims, batch_dims)))
    if idx_out:
      return out_array.at[idx_out].add(prod)
    else:
      return prod.sum(tuple(range(prod.ndim - out_array.ndim)), dtype=out_array.dtype)
  for _ in range(n_batch - len(lhs_contracting_b)):
    result = broadcasting_vmap(result)
  rhs = lax.expand_dims(rhs, range(len(rhs_batch), n_batch - len(lhs_contracting_b)))
  out_array = jnp.zeros(out_aval.shape, out_aval.dtype)
  return result(out_array, lhs_data, lhs_indices, rhs)

@bcoo_dot_general_p.def_abstract_eval
def _bcoo_dot_general_abstract_eval(lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  if lhs_data.dtype != rhs.dtype:
    raise ValueError("bcoo_dot_general requires arguments to have matching dtypes; "
                     f"got lhs.dtype={lhs_data.dtype}, rhs.dtype={rhs.dtype}")

  (lhs_contracting, _), (lhs_batch, _) = dimension_numbers
  n_batch, n_sparse, _, _ = _validate_bcoo(lhs_data, lhs_indices, lhs_spinfo.shape)
  out_shape = _dot_general_validated_shape(lhs_spinfo.shape, rhs.shape, dimension_numbers)

  if lhs_batch and max(lhs_batch) >= n_batch:
    raise NotImplementedError(
      "bcoo_dot_general batch dimensions must be among the batch dimensions in the sparse representtaion.\n"
      f"got lhs_batch={lhs_batch}, n_batch={n_batch}")

  # TODO: support contraction of dense dimensions?
  if any(d >= n_batch + n_sparse for d in lhs_contracting):
    raise NotImplementedError("bcoo_dot_general: contracting over dense dimensions.")

  return core.ShapedArray(out_shape, lhs_data.dtype)

_bcoo_dot_general_default_lowering = mlir.lower_fun(
    _bcoo_dot_general_impl, multiple_results=False)

def _collapse_mhlo(x, start, end):
  x_type = ir.RankedTensorType(x.type)
  shape = x_type.shape
  shape = (shape[:start]
           + [functools.reduce(operator.mul, shape[start:end + 1])]
           + shape[end + 1:])
  return mhlo.ReshapeOp(
      ir.RankedTensorType.get(shape, x_type.element_type), x).result

def _bcoo_dot_general_cuda_lowering(
    coo_matvec_lowering, coo_matmat_lowering, ctx, lhs_data, lhs_indices, rhs,
    *, dimension_numbers, lhs_spinfo: BCOOInfo):
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_data_aval, lhs_indices_aval, rhs_aval, = ctx.avals_in
  props = _validate_bcoo_indices(lhs_indices_aval, lhs_spinfo.shape)
  rhs_ndim = len(ir.RankedTensorType(rhs.type).shape)

  # Checks the shapes of lhs and rhs.
  assert props.n_dense == 0
  assert (props.n_batch, props.n_sparse, rhs_ndim) in [
      (0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 2, 2), (1, 2, 2)]

  # Checks the operation dimensions.
  assert len(lhs_batch) == 0
  assert len(rhs_batch) == 0
  assert len(lhs_contract) == 1

  # Checks the dtype.
  assert lhs_data_aval.dtype in [np.float32, np.float64, np.complex64,
                                 np.complex128]
  assert lhs_data_aval.dtype == rhs_aval.dtype
  assert lhs_indices_aval.dtype == np.int32

  if rhs_ndim == 1:
    bcoo_dot_general_fn = coo_matvec_lowering
  elif rhs_ndim == 2:
    bcoo_dot_general_fn = coo_matmat_lowering
    if rhs_contract[0] == 1:
      rhs = mhlo.TransposeOp(
          rhs, permutation=mlir.dense_int_elements([1, 0])).result
  else:
    raise ValueError(f"rhs has to be 1d or 2d; get {rhs_ndim}d.")

  if props.n_batch == 0:
    # non-batch mode.
    lhs_transpose = False
    if props.n_sparse == 1:
      # Converts lhs to a row vector.
      col = _collapse_mhlo(lhs_indices, start=0, end=1)
      row = mlir.full_like_aval(
          0, core.ShapedArray(ir.RankedTensorType(col.type).shape,
                              np.dtype(np.int32)))
      lhs_shape = (1, lhs_spinfo.shape[0])
      dot_product = bcoo_dot_general_fn(
          lhs_data, row, col, rhs, shape=lhs_shape, transpose=lhs_transpose,
          data_dtype=lhs_data_aval.dtype, index_dtype=lhs_indices_aval.dtype,
          x_dtype=rhs_aval.dtype)

      if rhs_ndim == 1:
        # Transforms a single-element array to a scalar.
        return [mhlo.ReshapeOp(
            ir.RankedTensorType.get(
                [], ir.RankedTensorType(dot_product.type).element_type),
            dot_product).result]
      else:
        return [_collapse_mhlo(dot_product, start=0, end=1)]
    elif props.n_sparse == 2:
      lhs_indices_shape = ir.RankedTensorType(lhs_indices.type).shape
      row = _collapse_mhlo(
          mhlo.SliceOp(
              lhs_indices,
              start_indices=mlir.dense_int_elements([0, 0]),
              limit_indices=mlir.dense_int_elements([lhs_indices_shape[0], 1]),
              strides=mlir.dense_int_elements([1, 1])).result,
          start=0, end=1)
      col = _collapse_mhlo(
          mhlo.SliceOp(
              lhs_indices,
              start_indices=mlir.dense_int_elements([0, 1]),
              limit_indices=mlir.dense_int_elements([lhs_indices_shape[0], 2]),
              strides=mlir.dense_int_elements([1, 1])).result,
          start=0, end=1)

      if lhs_contract[0] == 0:
        lhs_transpose = True

      return [bcoo_dot_general_fn(
          lhs_data, row, col, rhs, shape=lhs_spinfo.shape,
          transpose=lhs_transpose, data_dtype=lhs_data_aval.dtype,
          index_dtype=lhs_indices_aval.dtype,
          x_dtype=rhs_aval.dtype)]
    else:
      raise ValueError(f"lhs has to be 1d or 2d; get {props.n_sparse}d.")
  elif props.n_batch == 1:
    # batch mode.
    lhs_indices_shape = ir.RankedTensorType(lhs_indices.type).shape
    lhs_data_shape = ir.RankedTensorType(lhs_data.type).shape
    batch_count, _, _ = lhs_indices_shape
    rhs_shape = ir.RankedTensorType(rhs.type).shape

    # Squeeze the batch dimension for both indices and data.
    lhs_indices_2d_shape = (np.prod(np.array(lhs_indices_shape)[:-1]),
                            lhs_indices_shape[-1])
    lhs_data_1d_shape = (np.prod(np.array(lhs_data_shape)), )

    lhs_indices_2d = mhlo.ReshapeOp(
        ir.RankedTensorType.get(
            lhs_indices_2d_shape,
            ir.RankedTensorType(lhs_indices.type).element_type),
        lhs_indices).result

    lhs_data_1d = mhlo.ReshapeOp(
        ir.RankedTensorType.get(
            lhs_data_1d_shape,
            ir.RankedTensorType(lhs_data.type).element_type),
        lhs_data).result

    row = _collapse_mhlo(
        mhlo.SliceOp(
            lhs_indices_2d,
            start_indices=mlir.dense_int_elements([0, 0]),
            limit_indices=mlir.dense_int_elements([lhs_indices_2d_shape[0], 1]),
            strides=mlir.dense_int_elements([1, 1])).result,
        start=0, end=1)

    col = _collapse_mhlo(
        mhlo.SliceOp(
            lhs_indices_2d,
            start_indices=mlir.dense_int_elements([0, 1]),
            limit_indices=mlir.dense_int_elements([lhs_indices_2d_shape[0], 2]),
            strides=mlir.dense_int_elements([1, 1])).result,
        start=0, end=1)

    # Broadcast rhs to have the same batch size as lhs.
    # TODO(tianjianlu): remove broadcasting.
    # Use batch_stride = 0 for non-batch.
    # The issue (https://github.com/NVIDIA/CUDALibrarySamples/issues/81#issuecomment-1205562643)
    # in cusparse library does not allow batch_stride = 0 for a non-batched rhs.
    batched_rhs_shape = (batch_count,) + tuple(rhs_shape)
    batched_rhs = mhlo.BroadcastInDimOp(
        ir.RankedTensorType.get(batched_rhs_shape,
                                ir.RankedTensorType(rhs.type).element_type),
        rhs,
        broadcast_dimensions=mlir.dense_int_elements([1, 2])).result
    batched_rhs_2d_shape = (np.prod(np.array(batched_rhs_shape)[:-1]), batched_rhs_shape[-1])
    batched_rhs_2d = mhlo.ReshapeOp(
        ir.RankedTensorType.get(
            batched_rhs_2d_shape,
            ir.RankedTensorType(batched_rhs.type).element_type),
        batched_rhs).result

    lhs_transpose = True if lhs_contract[0] == props.n_batch else False

    return [bcoo_dot_general_fn(
        lhs_data_1d, row, col, batched_rhs_2d, shape=lhs_spinfo.shape,
        transpose=lhs_transpose, data_dtype=lhs_data_aval.dtype,
        index_dtype=lhs_indices_aval.dtype,
        x_dtype=rhs_aval.dtype)]
  else:
    raise ValueError(f"n_batch has to be 0 or 1; get {props.n_batch}.")

def _bcoo_dot_general_gpu_lowering(
    coo_matvec_lowering, coo_matmat_lowering,
    ctx, lhs_data, lhs_indices, rhs, *, dimension_numbers,
    lhs_spinfo: BCOOInfo):

  if not config.jax_bcoo_cusparse_lowering:
    return _bcoo_dot_general_default_lowering(
      ctx, lhs_data, lhs_indices, rhs,
      dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_data_aval, lhs_indices_aval, rhs_aval, = ctx.avals_in
  n_batch, n_sparse, n_dense, _ = _validate_bcoo(
      lhs_data_aval, lhs_indices_aval, lhs_spinfo.shape)

  dtype = lhs_data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f'bcoo_dot_general cusparse/hipsparse lowering not available '
                  f'for dtype={dtype}. Falling back to default implementation.',
                  CuSparseEfficiencyWarning)
    return _bcoo_dot_general_default_lowering(
      ctx, lhs_data, lhs_indices, rhs,
      dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

  if (n_batch > 1 or n_dense or
      n_sparse not in [1, 2] or rhs_aval.ndim not in [1, 2] or
      lhs_batch or rhs_batch or len(lhs_contract) != 1):
    return _bcoo_dot_general_default_lowering(
      ctx, lhs_data, lhs_indices, rhs,
      dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)
  else:
    if not lhs_spinfo.indices_sorted:
      warnings.warn("bcoo_dot_general GPU lowering requires matrices with "
                    "sorted indices. To sort the rows in your matrix, use e.g. "
                    "mat = mat.sort_indices(). Falling back to the default "
                    "implementation.", CuSparseEfficiencyWarning)
      return _bcoo_dot_general_default_lowering(
        ctx, lhs_data, lhs_indices, rhs,
        dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

    if n_batch == 1:
      # The support for batched computation in cusparseSpMM COO was added in
      # 11.6.1: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cusparse-11.6.1
      cuda_version = int(xla_bridge.get_backend().platform_version.split()[-1])

      # TODO(tianjianlu): enable the batch mode of cusparseSpMv.
      cuda_supported_batch_mode = (
          n_sparse == 2 and rhs_aval.ndim == 2 and
          len(lhs_contract) == 1 and lhs_contract[0] in [1, 2] and
          len(rhs_contract) == 1 and rhs_contract[0] in [0, 1] and
          cuda_version >= 11061 and jaxlib_version >= (0, 3, 18))
      if not cuda_supported_batch_mode:
        warnings.warn("bcoo_dot_general GPU lowering currently does not "
                      "support this batch-mode computation. Falling back to "
                      "the default implementation.", CuSparseEfficiencyWarning)
        return _bcoo_dot_general_default_lowering(
          ctx, lhs_data, lhs_indices, rhs,
          dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

    return _bcoo_dot_general_cuda_lowering(
      coo_matvec_lowering, coo_matmat_lowering, ctx, lhs_data, lhs_indices, rhs,
      dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

def _bcoo_dot_general_jvp_lhs(lhs_data_dot, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  return _bcoo_dot_general(lhs_data_dot, lhs_indices, rhs, dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

def _bcoo_dot_general_jvp_rhs(rhs_dot, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  return _bcoo_dot_general(lhs_data, lhs_indices, rhs_dot, dimension_numbers=dimension_numbers, lhs_spinfo=lhs_spinfo)

def _bcoo_dot_general_transpose(ct, lhs_data, lhs_indices, rhs, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  assert not ad.is_undefined_primal(lhs_indices)
  if type(ct) is ad.Zero:
    return ad.Zero
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_ndim = len(lhs_spinfo.shape)
  rhs_ndim = rhs.aval.ndim if ad.is_undefined_primal(rhs) else rhs.ndim
  lhs_kept = remaining(range(lhs_ndim), lhs_contract, lhs_batch)
  rhs_kept = remaining(range(rhs_ndim), rhs_contract, rhs_batch)
  ans_batch, ans_lhs, ans_rhs = map(list, ranges_like(lhs_batch, lhs_kept, rhs_kept))
  if ad.is_undefined_primal(lhs_data):
    dims = ((ans_rhs, rhs_kept), (ans_batch, rhs_batch))
    lhs_contract_sorted_by_rhs = list(np.take(lhs_contract, np.argsort(rhs_contract)))
    permutation = list(lhs_batch) + lhs_kept + lhs_contract_sorted_by_rhs
    out_axes = list(np.argsort(permutation))

    # What follows is essentially this, but computed in terms of dot_general_sampled:
    # out_dense_T = lax.dot_general(ct, rhs, dimension_numbers=dims)
    # out_dense = lax.transpose(out_dense_T, out_axes)
    # result = bcoo_extract(lhs_indices, out_dense)

    # Instead we (1) un-transpose indices, (2) compute SDDMM, (3) re-transpose result
    dummy_data = jnp.ones([1 for i in range(lhs_indices.ndim - 2)] + [lhs_indices.shape[-2]])
    dummy_spinfo = BCOOInfo(tuple(lhs_indices.shape[:-2]) + tuple(1 for i in range(lhs_indices.shape[-1])))
    _, lhs_indices_T = _bcoo_transpose(dummy_data, lhs_indices, permutation=permutation, spinfo=dummy_spinfo)
    result_T = bcoo_dot_general_sampled(ct, rhs, lhs_indices_T, dimension_numbers=dims)
    result, _ = _bcoo_transpose(result_T, lhs_indices_T, permutation=out_axes, spinfo=dummy_spinfo)

    return result, lhs_indices, rhs
  else:
    dims = ((lhs_kept, ans_lhs), (lhs_batch, ans_batch))
    rhs_contract_sorted_by_lhs = list(np.take(rhs_contract, np.argsort(lhs_contract)))
    out_axes = list(np.argsort(list(rhs_batch) + rhs_contract_sorted_by_lhs + rhs_kept))
    result = _bcoo_dot_general(lhs_data, lhs_indices, ct, lhs_spinfo=lhs_spinfo, dimension_numbers=dims)
    return lhs_data, lhs_indices, lax.transpose(result, out_axes)

def _bcoo_dot_general_batch_rule(batched_args, batch_dims, *, dimension_numbers, lhs_spinfo: BCOOInfo):
  lhs_data, lhs_indices, rhs = batched_args
  batch_dims = list(batch_dims)
  batch_size = max(0 if dim is None else arg.shape[dim]
                   for arg, dim in zip(batched_args, batch_dims))
  if batch_dims[0] is None:
    lhs_data = lhs_data[None]
    batch_dims[0] = 0
  if batch_dims[1] is None:
    lhs_indices = lhs_indices[None]
    batch_dims[1] = 0
  # TODO: handle different batchings between lhs_data and lhs_indices?
  assert batch_dims[0] == batch_dims[1] == 0
  new_dimension_numbers, result_batch_dim = _dot_general_batch_dim_nums(
      (len(lhs_spinfo.shape), rhs.ndim), (batch_dims[0], batch_dims[2]), dimension_numbers)
  new_shape = (batch_size, *lhs_spinfo.shape)
  batched_out = _bcoo_dot_general(lhs_data, lhs_indices, rhs, lhs_spinfo=BCOOInfo(new_shape),
                                  dimension_numbers=new_dimension_numbers)
  return batched_out, result_batch_dim

ad.defjvp(bcoo_dot_general_p, _bcoo_dot_general_jvp_lhs, None, _bcoo_dot_general_jvp_rhs)
ad.primitive_transposes[bcoo_dot_general_p] = _bcoo_dot_general_transpose
batching.primitive_batchers[bcoo_dot_general_p] = _bcoo_dot_general_batch_rule

mlir.register_lowering(
    bcoo_dot_general_p, _bcoo_dot_general_default_lowering)

if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(bcoo_dot_general_p,
                          partial(_bcoo_dot_general_gpu_lowering,
                                  gpu_sparse.cuda_coo_matvec,
                                  gpu_sparse.cuda_coo_matmat),
                          platform='cuda')
if gpu_sparse.rocm_is_supported:
  mlir.register_lowering(bcoo_dot_general_p,
                          partial(_bcoo_dot_general_gpu_lowering,
                                  gpu_sparse.rocm_coo_matvec,
                                  gpu_sparse.rocm_coo_matmat),
                          platform='rocm')

#----------------------------------------------------------------------
# bcoo_dot_general_sampled
# (batched) general sampled dot product of two dense ND arrays, with
# output computed only at a given set of sparse indices.

bcoo_dot_general_sampled_p = core.Primitive("bcoo_dot_general_sampled")

def bcoo_dot_general_sampled(A, B, indices, *, dimension_numbers):
  """A contraction operation with output computed at given sparse indices.

  Args:
    lhs: An ndarray.
    rhs: An ndarray.
    indices: BCOO indices.
    dimension_numbers: a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims),
      (lhs_batch_dims, rhs_batch_dims))`.

  Returns:
    BCOO data, an ndarray containing the result.
  """
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  return bcoo_dot_general_sampled_p.bind(A, B, indices,
                                         dimension_numbers=(cdims, bdims))

@bcoo_dot_general_sampled_p.def_impl
def _bcoo_dot_general_sampled_impl(A, B, indices, *, dimension_numbers):
  # TODO(jakevdp): use a more efficient implementation that avoids the full dot product.
  dense_result = lax.dot_general(A, B, dimension_numbers=dimension_numbers)
  return bcoo_extract(indices, dense_result)

@bcoo_dot_general_sampled_p.def_abstract_eval
def _bcoo_dot_general_sampled_abstract_eval(A, B, indices, *, dimension_numbers):
  dense_result, = pe.abstract_eval_fun(lambda *args: [lax.dot_general(*args, dimension_numbers=dimension_numbers)], A, B)
  sparse_result, = pe.abstract_eval_fun(lambda *args: [bcoo_extract(*args)], indices, dense_result)
  return sparse_result

def _bcoo_dot_general_sampled_transpose(ct, A, B, indices, *, dimension_numbers):
  A_shape = A.aval.shape if hasattr(A, 'aval') else A.shape
  B_shape = B.aval.shape if hasattr(B, 'aval') else B.shape
  mat_shape = _dot_general_validated_shape(A_shape, B_shape, dimension_numbers)
  mat = ad.UndefinedPrimal(core.ShapedArray(mat_shape, ct.dtype))
  indices, ct = _bcoo_extract_transpose(ct, indices, mat)
  kwds = {'dimension_numbers': dimension_numbers,
          'precision': None,
          'preferred_element_type': None}
  A, B = ad.get_primitive_transpose(lax.dot_general_p)(ct, A, B, **kwds)
  return A, B, indices

def _bcoo_dot_general_sampled_jvp_A(A_dot, A, B, indices, *, dimension_numbers):
  return bcoo_dot_general_sampled(A_dot, B, indices, dimension_numbers=dimension_numbers)

def _bcoo_dot_general_sampled_jvp_B(B_dot, A, B, indices, *, dimension_numbers):
  return bcoo_dot_general_sampled(A, B_dot, indices, dimension_numbers=dimension_numbers)

def _bcoo_dot_general_sampled_batch_rule(batched_args, batch_dims, *, dimension_numbers):
  def impl(A, B, indices):
    return _bcoo_dot_general_sampled_impl(A, B, indices, dimension_numbers=dimension_numbers)
  return vmap(impl, in_axes=batch_dims, out_axes=0)(*batched_args), 0

ad.defjvp(bcoo_dot_general_sampled_p, _bcoo_dot_general_sampled_jvp_A,
          _bcoo_dot_general_sampled_jvp_B, None)
ad.primitive_transposes[bcoo_dot_general_sampled_p] = _bcoo_dot_general_sampled_transpose
batching.primitive_batchers[bcoo_dot_general_sampled_p] = _bcoo_dot_general_sampled_batch_rule
mlir.register_lowering(
    bcoo_dot_general_sampled_p,
    mlir.lower_fun(_bcoo_dot_general_sampled_impl, multiple_results=False))

#----------------------------------------------------------------------
# bcoo_spdot_general
# (batched) general dot product of two BCOO sparse arrays returning a
# Dense ND array.

bcoo_spdot_general_p = core.Primitive('bcoo_spdot_general')
bcoo_spdot_general_p.multiple_results = True

def _bcoo_spdot_general(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo: BCOOInfo, rhs_spinfo: BCOOInfo, dimension_numbers: DotDimensionNumbers):
  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
  cdims = (api_util._ensure_index_tuple(lhs_contract),
           api_util._ensure_index_tuple(rhs_contract))
  bdims = (api_util._ensure_index_tuple(lhs_batch),
           api_util._ensure_index_tuple(rhs_batch))
  return bcoo_spdot_general_p.bind(lhs_data, lhs_indices, rhs_data, rhs_indices,
                                   lhs_spinfo=lhs_spinfo, rhs_spinfo=rhs_spinfo,
                                   dimension_numbers=(cdims, bdims))

def _bcoo_spdot_general_unbatched(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo, rhs_spinfo, lhs_contracting, rhs_contracting):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)

  assert lhs.n_batch == rhs.n_batch == 0
  assert lhs.n_dense == rhs.n_dense == 0
  assert [lhs_shape[d] for d in lhs_contracting] == [rhs_shape[d] for d in rhs_contracting]
  assert max(lhs_contracting, default=-1) < lhs.n_sparse
  assert max(rhs_contracting, default=-1) < rhs.n_sparse

  out_shape = (
    *(s for i, s in enumerate(lhs_shape) if i not in lhs_contracting),
    *(s for i, s in enumerate(rhs_shape) if i not in rhs_contracting))

  lhs_i = lhs_indices[:, jnp.array(lhs_contracting, dtype=int)]
  rhs_i = rhs_indices[:, jnp.array(rhs_contracting, dtype=int)]
  lhs_j = lhs_indices[:, jnp.array(remaining(range(lhs.n_sparse), lhs_contracting), dtype=int)]
  rhs_j = rhs_indices[:, jnp.array(remaining(range(rhs.n_sparse), rhs_contracting), dtype=int)]

  # TODO(jakevdp): can we do this more efficiently than using an outer product? Note that
  #   jnp.isin() currently doesn't help much, because it also does all() over an outer
  #   comparison.
  overlap = (lhs_i[:, None] == rhs_i[None, :]).all(-1)
  lhs_fill_value = jnp.expand_dims(
    jnp.array([lhs_shape[d] for d in lhs_contracting], dtype=lhs_i.dtype),
    range(lhs_i.ndim - 1))
  rhs_fill_value = jnp.expand_dims(
    jnp.array([rhs_shape[d] for d in rhs_contracting], dtype=rhs_i.dtype),
    range(rhs_i.ndim - 1))
  lhs_valid = (lhs_i < lhs_fill_value).all(-1)
  rhs_valid = (rhs_i < rhs_fill_value).all(-1)
  out_data = jnp.where(overlap & lhs_valid[:, None] & rhs_valid[None, :],
                       lhs_data[:, None] * rhs_data[None, :], 0).ravel()

  out_indices = jnp.empty([lhs.nse, rhs.nse, lhs_j.shape[-1] + rhs_j.shape[-1]],
                          dtype=jnp.result_type(lhs_indices, rhs_indices))
  out_indices = out_indices.at[:, :, :lhs_j.shape[-1]].set(lhs_j[:, None])
  out_indices = out_indices.at[:, :, lhs_j.shape[-1]:].set(rhs_j[None, :])
  out_indices = out_indices.reshape(len(out_data), out_indices.shape[-1])
  out_nse = (lhs.nse if lhs_j.shape[1] else 1) * (rhs.nse if rhs_j.shape[1] else 1)
  # Note: we do not eliminate zeros here, because it can cause issues with autodiff.
  # See https://github.com/google/jax/issues/10163.
  return _bcoo_sum_duplicates(out_data, out_indices, spinfo=BCOOInfo(shape=out_shape), nse=out_nse)

@bcoo_spdot_general_p.def_impl
def _bcoo_spdot_general_impl(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo: BCOOInfo, rhs_spinfo: BCOOInfo, dimension_numbers):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  assert lhs.n_dense == rhs.n_dense == 0
  data_aval, indices_aval = _bcoo_spdot_general_abstract_eval(
    lhs_data.aval, lhs_indices.aval, rhs_data.aval, rhs_indices.aval,
    lhs_spinfo=lhs_spinfo, rhs_spinfo=rhs_spinfo, dimension_numbers=dimension_numbers)
  out_shape = _dot_general_validated_shape(lhs_shape, rhs_shape, dimension_numbers)
  _validate_bcoo(data_aval, indices_aval, out_shape)

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

  # Move batch dimensions to front of each array.
  lhs_batch_perm = [*lhs_batch, *remaining(range(lhs.n_batch), lhs_batch)]
  rhs_batch_perm = [*rhs_batch, *remaining(range(rhs.n_batch), rhs_batch)]
  lhs_data = lhs_data.transpose([*lhs_batch_perm, *range(lhs.n_batch, lhs_data.ndim)])
  rhs_data = rhs_data.transpose([*rhs_batch_perm, *range(rhs.n_batch, rhs_data.ndim)])
  lhs_indices = lhs_indices.transpose([*lhs_batch_perm, *range(lhs.n_batch, lhs_indices.ndim)])
  rhs_indices = rhs_indices.transpose([*rhs_batch_perm, *range(rhs.n_batch, rhs_indices.ndim)])

  # Implement batched dot product via vmap
  func = functools.partial(_bcoo_spdot_general_unbatched,
      lhs_spinfo=BCOOInfo(lhs_shape[lhs.n_batch:]),
      rhs_spinfo=BCOOInfo(rhs_shape[rhs.n_batch:]),
      lhs_contracting=[d - lhs.n_batch for d in lhs_contracting],
      rhs_contracting=[d - rhs.n_batch for d in rhs_contracting])

  for _ in reversed(range(len(rhs_batch), rhs.n_batch)):
    func = broadcasting_vmap(func, in_axes=(None, None, 0, 0))
  for _ in reversed(range(len(lhs_batch), lhs.n_batch)):
    func = broadcasting_vmap(func, in_axes=(0, 0, None, None))
  for _ in range(len(lhs_batch)):
    func = broadcasting_vmap(func, in_axes=0)
  return func(lhs_data, lhs_indices, rhs_data, rhs_indices)

@bcoo_spdot_general_p.def_abstract_eval
def _bcoo_spdot_general_abstract_eval(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo: BCOOInfo, rhs_spinfo: BCOOInfo, dimension_numbers):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  if lhs_data.dtype != rhs_data.dtype:
    raise ValueError("bcoo_spdot_general requires inputs to have matching dtypes; "
                     f"got lhs.dtype={lhs_data.dtype}, rhs.dtype={rhs_data.dtype}")
  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  _ = _dot_general_validated_shape(lhs_shape, rhs_shape, dimension_numbers)

  if lhs.n_dense or rhs.n_dense:
    # TODO(jakevdp): handle dense dimensions
    raise NotImplementedError("bcoo_spdot_general with dense dimensions.")

  if (lhs_batch and max(lhs_batch) >= lhs.n_batch) or (rhs_batch and max(rhs_batch) >= rhs.n_batch):
    raise NotImplementedError("bcoo_spdot_general: batch_dims must correspond to batch dimensions of the sparse representation.")

  if lhs_contracting and (min(lhs_contracting) < lhs.n_batch or max(lhs_contracting) >= lhs.n_batch + lhs.n_sparse):
    raise NotImplementedError("bcoo_spdot_general only supports contraction of sparse indices.")

  if rhs_contracting and (min(rhs_contracting) < rhs.n_batch or max(rhs_contracting) >= rhs.n_batch + rhs.n_sparse):
    raise NotImplementedError("bcoo_spdot_general only supports contraction of sparse indices.")

  if rhs.n_batch > len(rhs_batch) and lhs.n_sparse > len(lhs_contracting):
    raise ValueError("bcoo_spdot_general: cannot have unused batch dims on rhs with unused sparse dims on lhs.")

  out_nse = (
    (lhs.nse if lhs.n_sparse > len(lhs_contracting) else 1) *
    (rhs.nse if rhs.n_sparse > len(rhs_contracting) else 1)
  )

  data_shape = (
    *(lhs_shape[dim] for dim in lhs_batch),
    *(lhs_data.shape[dim] for dim in range(lhs.n_batch) if dim not in lhs_batch),
    *(rhs_data.shape[dim] for dim in range(rhs.n_batch) if dim not in rhs_batch),
    out_nse)
  indices_shape = (
    *(lhs_shape[dim] for dim in lhs_batch),
    *(lhs_indices.shape[dim] for dim in range(lhs.n_batch) if dim not in lhs_batch),
    *(rhs_indices.shape[dim] for dim in range(rhs.n_batch) if dim not in rhs_batch),
    out_nse, lhs.n_sparse + rhs.n_sparse - 2 * len(lhs_contracting))
  return core.ShapedArray(data_shape, lhs_data.dtype), core.ShapedArray(indices_shape, lhs_indices.dtype)

def _bcoo_spdot_general_batch_rule(batched_args, batch_dims, *, lhs_spinfo: BCOOInfo, rhs_spinfo: BCOOInfo, dimension_numbers):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  lhs_data, lhs_indices, rhs_data, rhs_indices = batched_args
  batch_dims = list(batch_dims)
  batch_size = max(0 if dim is None else arg.shape[dim]
                   for arg, dim in zip(batched_args, batch_dims))
  if batch_dims[0] is None:
    lhs_data = lhs_data[None]
    batch_dims[0] = 0
  if batch_dims[1] is None:
    lhs_indices = lhs_indices[None]
    batch_dims[1] = 0
  assert batch_dims[0] == batch_dims[1] == 0
  if batch_dims[2] is None:
    rhs_data = rhs_data[None]
    batch_dims[2] = 0
  if batch_dims[3] is None:
    rhs_indices = rhs_indices[None]
    batch_dims[3] = 0
  if any(dim != 0 for dim in batch_dims):
    raise NotImplementedError("batching along non-leading dimension.")
  assert all(dim == 0 for dim in batch_dims)
  new_dimension_numbers, result_batch_dim = _dot_general_batch_dim_nums(
      (len(lhs_shape), len(rhs_shape)), (batch_dims[0], batch_dims[2]), dimension_numbers)
  new_lhs_shape = (batch_size, *lhs_shape)
  new_rhs_shape = (batch_size, *rhs_shape)
  batched_out = _bcoo_spdot_general(lhs_data, lhs_indices, rhs_data, rhs_indices,
                                    dimension_numbers=new_dimension_numbers,
                                    lhs_spinfo=BCOOInfo(new_lhs_shape),
                                    rhs_spinfo=BCOOInfo(new_rhs_shape))
  return batched_out, (result_batch_dim, result_batch_dim)


def _bcoo_spdot_general_jvp(primals, tangents, **kwds):
  lhs_data, lhs_indices, rhs_data, rhs_indices = primals
  lhs_data_dot, lhs_indices_dot, rhs_data_dot, rhs_indices_dot = tangents
  primals_out = _bcoo_spdot_general(*primals, **kwds)
  assert type(lhs_indices_dot) is ad.Zero
  assert type(rhs_indices_dot) is ad.Zero
  data_dot_out = 0
  if type(lhs_data_dot) is not ad.Zero:
    data_dot_out += _bcoo_spdot_general(lhs_data_dot, lhs_indices, rhs_data, rhs_indices, **kwds)[0]
  if type(rhs_data_dot) is not ad.Zero:
    data_dot_out += _bcoo_spdot_general(lhs_data, lhs_indices, rhs_data_dot, rhs_indices, **kwds)[0]
  return primals_out, [data_dot_out, ad.Zero.from_value(primals_out[1])]

# TODO(JVP): transpose rule
batching.primitive_batchers[bcoo_spdot_general_p] = _bcoo_spdot_general_batch_rule
ad.primitive_jvps[bcoo_spdot_general_p] = _bcoo_spdot_general_jvp
mlir.register_lowering(bcoo_spdot_general_p, mlir.lower_fun(
    _bcoo_spdot_general_impl, multiple_results=True))


#----------------------------------------------------------------------
# bcoo_sort_indices
# Utility to sort the indices of a BCOO representation. This primitive
# does not support deduplication or removing of zeros; see bcoo_sum_duplicates.

bcoo_sort_indices_p = core.Primitive("bcoo_sort_indices")
bcoo_sort_indices_p.multiple_results = True

def bcoo_sort_indices(mat):
  """Sort indices of a BCOO array.

  Args:
    mat : BCOO array

  Returns:
    mat_out : BCOO array with sorted indices.
  """
  data, indices = bcoo_sort_indices_p.bind(*mat._bufs, spinfo=mat._info)
  return BCOO((data, indices), shape=mat.shape, indices_sorted=True,
              unique_indices=mat.unique_indices)

@bcoo_sort_indices_p.def_impl
def _bcoo_sort_indices_impl(data, indices, *, spinfo):
  props = _validate_bcoo(data, indices, spinfo.shape)
  if props.n_sparse == 0:
    return data, indices
  f = _bcoo_sort_indices_unbatched
  for _ in range(props.n_batch):
    f = vmap(f)
  indices, perm = f(indices)
  permute = lambda d, p: d[p]
  for _ in range(props.n_batch):
    permute = broadcasting_vmap(permute)
  data = permute(data, perm)
  return data, indices

def _bcoo_sort_indices_unbatched(indices):
  # sort indices without summing duplicates
  nse, N = indices.shape
  idx_cols = (indices[:, i] for i in range(N))
  *indices, perm = lax.sort((*idx_cols, lax.iota(indices.dtype, nse)), num_keys=N)
  return jnp.column_stack(indices), perm

@bcoo_sort_indices_p.def_abstract_eval
def _bcoo_sort_indices_abstract_eval(data, indices, *, spinfo):
  props = _validate_bcoo(data, indices, spinfo.shape)
  if props.n_sparse == 0:
    return data, indices
  data_out = core.ShapedArray(
    (*map(max, indices.shape[:props.n_batch], data.shape[:props.n_batch]),
     props.nse, *data.shape[props.n_batch + 1:]), data.dtype, weak_type=data.weak_type)
  return data_out, indices

def _bcoo_sort_indices_batching_rule(batched_args, batch_dims, *, spinfo):
  data, indices = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"batch_dims={batch_dims}. Only 0 and None are supported.")
  if batch_dims[0] is None:
    data = data[None, ...]
  if batch_dims[1] is None:
    indices = indices[None, ...]
  new_spinfo = BCOOInfo(shape=(max(data.shape[0], indices.shape[0]), *spinfo.shape))
  data_out, indices_out = bcoo_sort_indices_p.bind(data, indices, spinfo=new_spinfo)
  out_axes = (0, 0)
  # Note: if data is unbatched on input, it will be batched on output.
  # However, if indices are unbatched on input, they will be unbatched on output.
  if batch_dims[1] is None:
    indices_out = indices_out[0]
    out_axes = (0, None)
  return (data_out, indices_out), out_axes

def _bcoo_sort_indices_jvp(primals, tangents, *, spinfo):
  props = _validate_bcoo(*primals, spinfo.shape)
  if props.n_sparse == 0:
    return primals, tangents

  data, indices = primals
  data_dot, _ = tangents
  f = _bcoo_sort_indices_unbatched
  for _ in range(props.n_batch):
    f = broadcasting_vmap(f)
  indices_out, perm = f(indices)
  permute = lambda d, p: d[p]
  for _ in range(props.n_batch):
    permute = broadcasting_vmap(permute)
  data_out = permute(data, perm)

  indices_dot_out = ad.Zero.from_value(indices)
  data_dot_out = ad.Zero.from_value(data_out) if type(data_dot) is ad.Zero else permute(data_dot, perm)
  return (data_out, indices_out), (data_dot_out, indices_dot_out)

_bcoo_sort_indices_mhlo = mlir.lower_fun(
    _bcoo_sort_indices_impl, multiple_results=True)

ad.primitive_jvps[bcoo_sort_indices_p] = _bcoo_sort_indices_jvp
batching.primitive_batchers[bcoo_sort_indices_p] = _bcoo_sort_indices_batching_rule
mlir.register_lowering(bcoo_sort_indices_p, _bcoo_sort_indices_mhlo)


#----------------------------------------------------------------------
# bcoo_sum_duplicates
# Utility to sum duplicate indices in a BCOO array representation.

bcoo_sum_duplicates_p = core.Primitive("bcoo_sum_duplicates")
bcoo_sum_duplicates_p.multiple_results = True

def bcoo_sum_duplicates(mat, nse=None):
  """Sums duplicate indices within a BCOO array, returning an array with sorted indices.

  Args:
    mat : BCOO array
    nse : integer (optional). The number of specified elements in the output matrix. This must
      be specified for bcoo_sum_duplicates to be compatible with JIT and other JAX transformations.
      If not specified, the optimal nse will be computed based on the contents of the data and
      index arrays. If specified nse is larger than necessary, data and index arrays will be padded
      with standard fill values. If smaller than necessary, data elements will be dropped from the
      output matrix.

  Returns:
    mat_out : BCOO array with sorted indices and no duplicate indices.
  """
  data, indices = _bcoo_sum_duplicates(mat.data, mat.indices, spinfo=mat._info, nse=nse)
  return BCOO((data, indices), shape=mat.shape, indices_sorted=True,
              unique_indices=True)

def _bcoo_sum_duplicates(data, indices, *, spinfo, nse):
  if nse is not None:
    nse = core.concrete_or_error(operator.index, nse, "nse argument of bcoo_sum_duplicates.")
  return bcoo_sum_duplicates_p.bind(data, indices, spinfo=spinfo, nse=nse)

@bcoo_sum_duplicates_p.def_impl
def _bcoo_sum_duplicates_impl(data, indices, *, spinfo, nse):
  props = _validate_bcoo(data, indices, spinfo.shape)
  f = functools.partial(_bcoo_sum_duplicates_unbatched, shape=spinfo.shape[props.n_batch:])
  for _ in range(props.n_batch):
    f = vmap(f)
  indices_out, mapping, nse_batched = f(indices)
  if nse is None:
    nse = 1 if props.n_sparse == 0 else nse_batched.max()
  indices_out = _adjust_indices_nse(indices_out, nse=nse, shape=spinfo.shape)
  if props.n_sparse == 0:
    data = data.sum(props.n_batch, keepdims=True)
  data_out = jnp.empty((*map(max, indices.shape[:props.n_batch], data.shape[:props.n_batch]),
                        nse, *data.shape[props.n_batch + 1:]), dtype=data.dtype)
  permute = lambda d_out, m, d: d_out.at[m].add(d, mode='drop')
  for _ in range(props.n_batch):
    permute = broadcasting_vmap(permute)
  data_out = permute(data_out, mapping, data)
  return data_out, indices_out

def _adjust_indices_nse(indices, *, nse, shape):
  props = _validate_bcoo_indices(indices, shape)
  if nse <= props.nse:
    indices = indices[..., :nse, :]
  else:
    fill = lax.broadcast_in_dim(
      operand=jnp.array(shape[props.n_batch:props.n_batch + props.n_sparse], dtype=indices.dtype),
      shape=(*indices.shape[:-2], nse - props.nse, indices.shape[-1]),
      broadcast_dimensions=(indices.ndim - 1,)
    )
    indices = lax.concatenate([indices, fill], dimension=indices.ndim - 2)
  return indices

def _bcoo_sum_duplicates_unbatched(indices, *, shape):
  props = _validate_bcoo_indices(indices, shape)
  if props.n_sparse == 0:
    nse = 1
    mapping = jnp.zeros(nse, dtype='int32')
    indices_out = jnp.zeros_like(indices, shape=(nse, props.n_sparse))
    return indices_out, mapping, nse
  fill_value = jnp.expand_dims(jnp.array(shape[:props.n_sparse], dtype=indices.dtype), (0,))
  out_of_bounds = (indices >= fill_value).any(-1, keepdims=True)
  indices = jnp.where(out_of_bounds, fill_value, indices)
  # TODO: check if `indices_sorted` is True.
  indices_unique, inv_idx, nse = _unique(
    indices, axis=0, return_inverse=True, return_true_size=True,
    size=props.nse, fill_value=fill_value)
  nse = nse - (indices == fill_value).any().astype(nse.dtype)
  return indices_unique, inv_idx, nse

@bcoo_sum_duplicates_p.def_abstract_eval
def _bcoo_sum_duplicates_abstract_eval(data, indices, *, spinfo, nse):
  if nse is None:
    raise ValueError("bcoo_sum_duplicates: nse must be specified when using the function within "
                     "jit, vmap, and other transformations requiring abstract evaluation.")
  props = _validate_bcoo(data, indices, spinfo.shape)
  indices_out = core.ShapedArray((*indices.shape[:props.n_batch], nse, props.n_sparse),
                                  dtype=indices.dtype, weak_type=indices.weak_type)
  data_out = core.ShapedArray(
    (*map(max, indices.shape[:props.n_batch], data.shape[:props.n_batch]),
     nse, *data.shape[props.n_batch + 1:]), data.dtype, weak_type=data.weak_type)
  return data_out, indices_out

def _bcoo_sum_duplicates_batching_rule(batched_args, batch_dims, *, spinfo, nse):
  data, indices = batched_args
  if any(b not in [0, None] for b in batch_dims):
    raise NotImplementedError(f"batch_dims={batch_dims}. Only 0 and None are supported.")
  if batch_dims[0] is None:
    data = data[None, ...]
  if batch_dims[1] is None:
    indices = indices[None, ...]
  new_spinfo = BCOOInfo(shape=(max(data.shape[0], indices.shape[0]), *spinfo.shape))
  data_out, indices_out = bcoo_sum_duplicates_p.bind(data, indices, spinfo=new_spinfo, nse=nse)
  out_axes = (0, 0)
  # Note: if data is unbatched on input, it will be batched on output.
  # However, if indices are unbatched on input, they will be unbatched on output.
  if batch_dims[1] is None:
    indices_out = indices_out[0]
    out_axes = (0, None)
  return (data_out, indices_out), tuple(out_axes)

def _bcoo_sum_duplicates_jvp(primals, tangents, *, spinfo, nse):
  props = _validate_bcoo(*primals, spinfo.shape)

  data, indices = primals
  data_dot, _ = tangents
  f = functools.partial(_bcoo_sum_duplicates_unbatched, shape=spinfo.shape[props.n_batch:])
  for _ in range(props.n_batch):
    f = broadcasting_vmap(f)
  indices_out, mapping, nse_batched = f(indices)
  if nse is None:
    nse = jnp.sum(nse_batched)
  try:
    nse = core.concrete_or_error(operator.index, nse, "nse argument of bcoo_sum_duplicates.")
  except core.ConcretizationTypeError:
    raise ValueError("bcoo_sum_duplicates: nse must be specified when using the function within "
                     "jit, vmap, and other transformations requiring abstract evaluation.")
  indices_out = _adjust_indices_nse(indices_out, nse=nse, shape=spinfo.shape)
  if props.n_sparse == 0:
    data = data.sum(props.n_batch, keepdims=True)
    data_dot = data_dot.sum(props.n_batch, keepdims=True)
  data_out = jnp.empty((*map(max, indices.shape[:props.n_batch], data.shape[:props.n_batch]),
                        nse, *data.shape[props.n_batch + 1:]), dtype=data.dtype)
  data_dot_out = data_out
  permute = lambda d_out, m, d: d_out.at[m].add(d, mode='drop')
  for _ in range(props.n_batch):
    permute = broadcasting_vmap(permute)
  data_out = permute(data_out, mapping, data)
  indices_dot_out = ad.Zero.from_value(indices_out)
  data_dot_out = ad.Zero.from_value(data_out) if type(data_dot) is ad.Zero else permute(data_dot_out, mapping, data_dot)
  return (data_out, indices_out), (data_dot_out, indices_dot_out)

_bcoo_sum_duplicates_mhlo = mlir.lower_fun(
    _bcoo_sum_duplicates_impl, multiple_results=True)

ad.primitive_jvps[bcoo_sum_duplicates_p] = _bcoo_sum_duplicates_jvp
batching.primitive_batchers[bcoo_sum_duplicates_p] = _bcoo_sum_duplicates_batching_rule
mlir.register_lowering(bcoo_sum_duplicates_p, _bcoo_sum_duplicates_mhlo)

#----------------------------------------------------------------------
# BCOO functions that maybe should be primitives?

def bcoo_update_layout(mat, *, n_batch=None, n_dense=None, on_inefficient='error'):
  """Update the storage layout (i.e. n_batch & n_dense) of a BCOO matrix.

  In many cases this can be done without introducing undue storage overhead. However,
  increasing ``mat.n_batch`` or ``mat.n_dense`` will lead to very inefficient storage,
  with many explicitly-stored zeros, unless the new batch or dense dimensions have size
  0 or 1. In such cases, ``bcoo_update_layout`` will raise a :class:`SparseEfficiencyError`.
  This can be silenced by specifying the ``on_inefficient`` argument.

  Args:
    mat : BCOO array
    n_batch : optional(int) the number of batch dimensions in the output matrix. If None,
      then n_batch = mat.n_batch.
    n_dense : optional(int) the number of dense dimensions in the output matrix. If None,
      then n_dense = mat.n_dense.
    on_inefficient : optional(string), one of ``['error', 'warn', None]``. Specify the
      behavior in case of an inefficient reconfiguration. This is defined as a reconfiguration
      where the size of the resulting representation is much larger than the size of the
      input representation.

  Returns:
    mat_out : BCOO array
      A BCOO array representing the same sparse array as the input, with the specified
      layout. ``mat_out.todense()`` will match ``mat.todense()`` up to appropriate precision.
  """
  # TODO(jakevdp): allow specification of nse?
  # TODO(jakevdp): there is room for some improvements here:
  # - we could probably do better in the case of converting a dense dim to
  #   a batch dim or vice-versa. Worth adding that special case?
  # - we could work to preserve broadcasted batch dimensions when possible.
  # - if indices are known to be unique, we can convert them to batch/dense
  #   dimensions more efficiently.
  n_batch = mat.n_batch if n_batch is None else operator.index(n_batch)
  n_dense = mat.n_dense if n_dense is None else operator.index(n_dense)

  if (n_batch, n_dense) == (mat.n_batch, mat.n_dense):
    return mat

  n_sparse = mat.ndim - n_batch - n_dense
  if on_inefficient not in ['error', 'warn', None]:
    raise ValueError("on_inefficent={on_inefficient!r}; expected one of ['error', 'warn', None].")

  if n_batch < 0:
    raise ValueError(f"n_batch must be non-negative; got {n_batch}")
  if n_dense < 0:
    raise ValueError(f"n_dense must be non-negative; got {n_dense}")
  if n_sparse < 0:
    raise ValueError(f"sum of n_batch={n_batch} and n_dense={n_dense} "
                     f"cannot be larger than mat.ndim={mat.ndim}.")

  def _maybe_err_or_warn(msg):
    if on_inefficient == 'error':
      msg += (" To disable this error, set the on_inefficient argument "
              "of bcoo_update_layout to 'warn' or None.")
      raise SparseEfficiencyError(msg)
    elif on_inefficient == 'warn':
      msg += (" To disable this warning, set the on_inefficient argument "
              "of bcoo_update_layout to None.")
      warnings.warn(msg, category=SparseEfficiencyWarning)

  # TODO(jakevdp): are efficiency warnings necessary when nse is 0 or 1?
  if (n_dense > mat.n_dense and
      any(d > 1 for d in mat.shape[-n_dense:mat.ndim - mat.n_dense])):
    _maybe_err_or_warn(f"For matrix of shape {mat.shape}, increasing n_dense from "
                       f"{mat.n_dense} to {n_dense} results in inefficient storage.")
  if n_batch > mat.n_batch and any(d > 1 for d in mat.shape[mat.n_batch:n_batch]):
    _maybe_err_or_warn(f"For matrix of shape {mat.shape}, increasing n_batch from "
                       f"{mat.n_batch} to {n_batch} results in inefficient storage.")

  new_data, new_indices = mat.data, mat.indices
  shape = mat.shape
  current_n_batch = mat.n_batch
  current_n_dense = mat.n_dense

  if n_dense < current_n_dense:
    n = current_n_dense - n_dense
    def _update(d, i):
      new_d = d.reshape(np.prod(d.shape[:n]), *d.shape[n:])
      meshes = jnp.meshgrid(*(jnp.arange(s, dtype=i.dtype) for s in d.shape[:n]),
                            indexing='ij')
      new_i = jnp.column_stack([jnp.broadcast_to(i, (new_d.shape[0], i.size)),
                                *map(jnp.ravel, meshes)])
      return new_d, new_i
    for _ in range(current_n_batch + 1):
      _update = broadcasting_vmap(_update)
    new_data, new_indices = _update(new_data, new_indices)
    new_data = new_data.reshape(*new_data.shape[:current_n_batch],
                                np.prod(new_data.shape[current_n_batch:current_n_batch + 2]),
                                *new_data.shape[current_n_batch + 2:])
    new_indices = new_indices.reshape(*new_indices.shape[:current_n_batch],
                                      np.prod(new_indices.shape[current_n_batch: current_n_batch + 2]),
                                      *new_indices.shape[current_n_batch + 2:])
    current_n_dense = n_dense

  if n_batch < current_n_batch:
    n = current_n_batch - n_batch
    def _update(d, i):
      nse = i.shape[-2]
      new_d = d.reshape(np.prod(d.shape[:n + 1]), *d.shape[n + 1:])
      meshes = jnp.meshgrid(*(jnp.arange(d, dtype=i.dtype) for d in (*i.shape[:n], nse)),
                            indexing='ij')
      new_i = i.reshape(np.prod(i.shape[:n + 1]), *i.shape[n + 1:])
      new_i = jnp.column_stack((*(m.ravel() for m in meshes[:-1]), new_i))
      return new_d, new_i
    for _ in range(n_batch):
      _update = broadcasting_vmap(_update)
    new_data, new_indices = _update(new_data, new_indices)
    current_n_batch = n_batch

  if n_dense > current_n_dense:
    n = n_dense - current_n_dense
    def _update(d, i):
      new_d = jnp.zeros_like(d, shape=shape[-n_dense:]).at[tuple(i[-n:])].set(d)
      new_i = i[:-n]
      return new_d, new_i
    for _ in range(current_n_batch + 1):
      _update = broadcasting_vmap(_update)
    new_data, new_indices = _update(new_data, new_indices)
    current_n_dense = n_dense

  if n_batch > current_n_batch:
    n = n_batch - current_n_batch
    def _update(d, i):
      nse = i.shape[-2]
      idx = tuple(i[:, j] for j in range(n)) + (jnp.arange(nse),)
      new_i_shape = (*shape[current_n_batch:n_batch], nse, i.shape[-1] - n)
      new_i = jnp.broadcast_to(i[:, n:], new_i_shape)
      new_d_shape = (*shape[current_n_batch:n_batch], nse, *d.shape[d.ndim - n_dense:])
      new_d = jnp.zeros_like(d, shape=new_d_shape).at[idx].set(d)
      return new_d, new_i
    for _ in range(current_n_batch):
      _update = broadcasting_vmap(_update)
    new_data, new_indices = _update(new_data, new_indices)
    current_n_batch = n_batch

  return BCOO((new_data, new_indices), shape=shape)


def bcoo_broadcast_in_dim(mat, *, shape, broadcast_dimensions):
  """Expand the size and rank of a BCOO array by duplicating the data.

  A BCOO equivalence to jax.lax.broadcast_in_dim.

  Args:
    mat: A BCOO-format array.
    shape: The shape of the target array.
    broadcast_dimensions: The dimension in the shape of the target array which
      each dimension of the operand (``mat``) shape corresponds to.

  Returns:
    A BCOO-format array containing the target array.
  """
  return BCOO(_bcoo_broadcast_in_dim(mat.data, mat.indices, spinfo=mat._info,
                                     shape=shape,
                                     broadcast_dimensions=broadcast_dimensions),
              shape=shape)

def _bcoo_broadcast_in_dim(data, indices, *, spinfo, shape, broadcast_dimensions):
  """BCOO equivalent of lax.broadcast_in_dim"""
  if len(spinfo.shape) != len(broadcast_dimensions):
    raise ValueError(f"spinfo.shape={spinfo.shape} and broadcast_dimensions={broadcast_dimensions} must have the same length")
  props = _validate_bcoo(data, indices, spinfo.shape)
  batch_dims, sparse_dims, dense_dims = split_list(broadcast_dimensions, [props.n_batch, props.n_sparse])

  if max(batch_dims, default=0) > min(sparse_dims, default=len(shape)):
    raise ValueError("Cannot mix batch and sparse dimensions during broadcast_in_dim")
  if max(sparse_dims, default=0) > min(dense_dims, default=len(shape)):
    raise ValueError("Cannot mix sparse and dense dimensions during broadcast_in_dim")

  # All new dimensions preceding a sparse or dense dimension are batch dimensions:
  new_n_batch = min(broadcast_dimensions[props.n_batch:], default=len(shape))
  # TODO(jakevdp): Should new trailing dimensions be dense by default?
  new_n_dense = props.n_dense and len(shape) - min(broadcast_dimensions[-props.n_dense:])
  new_n_sparse = len(shape) - new_n_batch - new_n_dense

  if np.prod(spinfo.shape[props.n_batch: props.n_batch + props.n_sparse]) != np.prod(shape[new_n_batch:new_n_batch + new_n_sparse]):
    raise NotImplementedError("Adding sparse dimensions with lengths != 1")
  nse = props.nse
  # batch & dense dimensions
  new_data = lax.broadcast_in_dim(data,
      shape=(*shape[:new_n_batch], nse, *shape[new_n_batch + new_n_sparse:]),
      broadcast_dimensions=(*batch_dims, new_n_batch, *(b + 1 - new_n_sparse for b in dense_dims)))
  new_indices = lax.broadcast_in_dim(indices,
      shape=(*shape[:new_n_batch], nse, props.n_sparse),
      broadcast_dimensions=(*batch_dims, new_n_batch, new_n_batch + 1))

  # sparse dimensions
  new_indices = (jnp.zeros_like(new_indices, shape=(*shape[:new_n_batch], nse, new_n_sparse))
                   .at[..., jnp.array(sparse_dims, int) - new_n_batch].set(new_indices))

  return new_data, new_indices

def bcoo_concatenate(operands, *, dimension):
  """Sparse implementation of :func:`jax.lax.concatenate`

  Args:
    operands : Sequence of BCOO arrays to concatenate. The arrays must have equal
      shapes, except in the `dimension` axis. Additionally, the arrays must have
      have equivalent batch, sparse, and dense dimensions.
    dimension : Positive integer specifying the dimension along which to concatenate
      the arrays. The dimension must be among batch or sparse dimensions of the input;
      concatenation along dense dimensions is not supported.

  Returns:
    A BCOO array containing the concatenation of the inputs.
  """
  dimension = operator.index(dimension)
  if not all(isinstance(op, BCOO) for op in operands):
    raise ValueError("bcoo_concatenate: expected operands to be a sequence of BCOO arrays. "
                     f"Got {operands}")
  # Validate inputs using lax.concatenate abstract evaluation.
  out_aval = jax.eval_shape(
    functools.partial(lax.concatenate, dimension=dimension),
    [core.ShapedArray(op.shape, op.dtype) for op in operands])
  if len({op.n_dense for op in operands}) > 1:
    raise ValueError("bcoo_concatenate requires inputs to have matching nse dimensions.")

  n_batches = {op.n_batch for op in operands}
  # Correct for the common case, where op[None, :] adds a single batch dimension and we
  # need to align it in order to match the others & concatenate.
  if len(n_batches) != 1 and max(n_batches) == 1:
    if all(op.shape[0] == 1 for op in operands if op.n_batch == 0):
      operands = [bcoo_update_layout(op, n_batch=1) if op.n_batch == 0 else op for op in operands]
    elif all(op.shape[0] == 1 for op in operands if op.n_batch == 1):
      operands = [bcoo_update_layout(op, n_batch=0) if op.n_batch == 1 else op for op in operands]
    n_batches = {op.n_batch for op in operands}

  if len(n_batches) != 1:
    raise ValueError("bcoo_concatenate requires inputs to have matching batch dimensions.")

  n_batch, n_sparse = operands[0].n_batch, operands[0].n_sparse

  index_batches = [op.indices.shape[:n_batch] for op in operands]
  data_batches = [op.data.shape[:n_batch] for op in operands]
  if dimension < n_batch:
    index_batches = [s[:dimension] + s[dimension + 1:] for s in index_batches]
    data_batches = [s[:dimension] + s[dimension + 1:] for s in data_batches]
  if not (len(set(index_batches)) == len(set(data_batches)) == 1):
    raise NotImplementedError("concatenation of arrays with broadcasted batch indices")

  if dimension < n_batch:  # Concatenation along batch axes
    # Ensure nse of operands match.
    nses = {op.nse for op in operands}
    if len(nses) != 1:
      nse = max(nses)
      operands = [_bcoo_set_nse(op, nse) for op in operands]
    new_indices = lax.concatenate([op.indices for op in operands], dimension=dimension)
    new_data = lax.concatenate([op.data for op in operands], dimension=dimension)
  elif dimension < n_batch + n_sparse:  # Concatenation along sparse axes
    offsets = np.cumsum([0] + [op.shape[dimension] for op in operands[:-1]],
                        dtype=operands[0].indices.dtype)
    new_data = lax.concatenate([op.data for op in operands], dimension=n_batch)
    new_indices = lax.concatenate([op.indices.at[..., dimension - n_batch].add(offset)
                                   for op, offset in safe_zip(operands, offsets)],
                                  dimension=n_batch)
  else:  # Concatenation along dense axes
    # TODO(jakevdp) should we implement this? In general it results in a wasteful
    # representation because we cannot assume that the indices match.
    raise NotImplementedError("Concatenation along dense dimensions.")

  return BCOO((new_data, new_indices), shape=out_aval.shape)


def bcoo_reshape(mat, *, new_sizes, dimensions):
  """Sparse implementation of {func}`jax.lax.reshape`.

  Args:
    operand: BCOO array to be reshaped.
    new_sizes: sequence of integers specifying the resulting shape. The size
      of the final array must match the size of the input. This must be specified
      such that batch, sparse, and dense dimensions do not mix.
    dimensions: optional sequence of integers specifying the permutation order of
      the input shape. If specified, the length must match ``operand.shape``.
      Additionally, dimensions must only permute among like dimensions of mat:
      batch, sparse, and dense dimensions cannot be permuted.

  Returns:
    out: reshaped array.
  """
  if mat.n_dense:
    # TODO(jakevdp): implement reshape of dense dimensions.
    raise NotImplementedError("bcoo_reshape for matrices with dense dimensions.")

  if mat.n_batch:
    batch_size = np.prod(mat.shape[:mat.n_batch])
    cuml_shape = np.cumprod(new_sizes)
    if batch_size not in cuml_shape:
      raise ValueError("bcoo_reshape: new shape cannot mix batch and sparse dimensions; "
                      f"got shape={mat.shape} new_shape={new_sizes} with n_batch={mat.n_batch}")
    ind = cuml_shape.searchsorted(batch_size, side='right')
  else:
    ind = 0
  batch_sizes, sparse_sizes = new_sizes[:ind], new_sizes[ind:]
  batch_perm, sparse_perm, _ = _validate_permutation(mat.data, mat.indices, dimensions or tuple(range(mat.ndim)), mat.shape)

  if (mat.indices.shape[:mat.n_batch] != mat.data.shape[:mat.n_batch] != mat.shape[:mat.n_batch]):
    # TODO(jakevdp) implement this case via broadcast_in_dim
    raise NotImplementedError("reshape of arrays with broadacsted batch dimensions.")

  # Reshape batch dimensions: this is accomplished via a standard reshape.
  data = lax.reshape(
    mat.data, new_sizes=(*batch_sizes, *mat.data.shape[mat.n_batch:]),
    dimensions=(*batch_perm, *range(mat.n_batch, mat.data.ndim)))
  indices = lax.reshape(
    mat.indices, new_sizes=(*batch_sizes, *mat.indices.shape[mat.n_batch:]),
    dimensions=(*batch_perm, *range(mat.n_batch, mat.indices.ndim)))

  # Reshape the sparse dimensions: this is accomplished by re-indexing.
  index_cols = tuple(indices[..., i] for i in sparse_perm)
  sparse_shape = tuple(mat.shape[mat.n_batch + i] for i in sparse_perm)
  flat_indices = jnp.ravel_multi_index(index_cols, dims=sparse_shape, mode='clip')
  new_index_cols = jnp.unravel_index(flat_indices, sparse_sizes)
  new_indices = jnp.concatenate([col[..., None] for col in new_index_cols], axis=-1)
  with jax.numpy_rank_promotion('allow'):
    oob_indices = (indices >= jnp.array(mat.shape[mat.n_batch:], dtype=indices.dtype)).any(-1)
  new_indices = new_indices.at[oob_indices].set(jnp.array(sparse_sizes, dtype=new_indices.dtype))

  return BCOO((data, new_indices), shape=new_sizes)


def bcoo_slice(mat, *, start_indices: Sequence[int], limit_indices: Sequence[int],
               strides: Optional[Sequence[int]]=None):
  """Sparse implementation of {func}`jax.lax.slice`.

  Args:
    mat: BCOO array to be reshaped.
    start_indices: sequence of integers of length `mat.ndim` specifying the starting
      indices of each slice.
    limit_indices: sequence of integers of length `mat.ndim` specifying the ending
      indices of each slice
    strides: (not implemented) sequence of integers of length `mat.ndim` specifying
      the stride for each slice

  Returns:
    out: BCOO array containing the slice.
  """
  if not isinstance(mat, BCOO):
    raise ValueError(f"bcoo_slice: input should be BCOO array, got type(mat)={type(mat)}")
  start_indices = [operator.index(i) for i in start_indices]
  limit_indices = [operator.index(i) for i in limit_indices]
  if strides is not None:
    strides = [operator.index(i) for i in strides]
  else:
    strides = [1] * mat.ndim
  if len(start_indices) != len(limit_indices) != len(strides) != mat.ndim:
    raise ValueError(f"bcoo_slice: indices must have size mat.ndim={mat.ndim}")
  if strides != [1] * mat.ndim:
    raise NotImplementedError(f"non-unit strides; got {strides}")

  if not all(0 <= start <= end <= size
             for start, end, size in safe_zip(start_indices, limit_indices, mat.shape)):
    raise ValueError(f"bcoo_slice: invalid indices. Got start_indices={start_indices}, "
                     f"limit_indices={limit_indices} and shape={mat.shape}")

  start_batch, start_sparse, start_dense = split_list(start_indices, [mat.n_batch, mat.n_sparse])
  end_batch, end_sparse, end_dense = split_list(limit_indices, [mat.n_batch, mat.n_sparse])

  data_slices = []
  index_slices = []
  for i, (start, end) in enumerate(zip(start_batch, end_batch)):
    data_slices.append(slice(None) if mat.data.shape[i] != mat.shape[i] else slice(start, end))
    index_slices.append(slice(None) if mat.indices.shape[i] != mat.shape[i] else slice(start, end))
  data_slices.append(slice(None))
  index_slices.extend([slice(None), slice(None)])
  for i, (start, end) in enumerate(zip(start_dense, end_dense)):
    data_slices.append(slice(start, end))
  new_data = mat.data[tuple(data_slices)]
  new_indices = mat.indices[tuple(index_slices)]
  new_shape = [end - start for start, end in safe_zip(start_indices, limit_indices)]

  if mat.n_sparse:
    starts = jnp.expand_dims(jnp.array(start_sparse, dtype=new_indices.dtype), range(mat.n_batch + 1))
    ends = jnp.expand_dims(jnp.array(end_sparse, dtype=new_indices.dtype), range(mat.n_batch + 1))
    sparse_shape = jnp.array(mat.shape[mat.n_batch: mat.n_batch + mat.n_sparse], dtype=new_indices.dtype)

    keep = jnp.all((new_indices >= starts) & (new_indices < ends), -1, keepdims=True)
    new_indices = jnp.where(keep, new_indices - starts, sparse_shape)

    keep_data = lax.expand_dims(keep[..., 0], range(mat.n_batch + 1, mat.n_batch + 1 + mat.n_dense))
    new_data = jnp.where(keep_data, new_data, 0)

  return BCOO((new_data, new_indices), shape=new_shape)

def bcoo_dynamic_slice(mat, start_indices: Sequence[Any], slice_sizes: Sequence[int]):
  """Sparse implementation of {func}`jax.lax.dynamic_slice`.

  Args:
    mat: BCOO array to slice.
    start_indices: a list of scalar indices, one per dimension. These values
      may be dynamic.
    slice_sizes: the size of the slice. Must be a sequence of non-negative
      integers with length equal to `ndim(operand)`. Inside a JIT compiled
      function, only static values are supported (all JAX arrays inside JIT
      must have statically known size).

  Returns:
    out: BCOO array containing the slice.
  """
  if not isinstance(mat, BCOO):
    raise ValueError(f"bcoo_slice: input should be BCOO array, got type(mat)={type(mat)}")
  start_indices = tuple(jnp.asarray(i) for i in start_indices)
  assert all(jnp.issubdtype(i.dtype, np.integer) for i in start_indices)
  assert all(i.shape == () for i in start_indices)
  slice_sizes = tuple(operator.index(i) for i in slice_sizes)
  if len(start_indices) != len(slice_sizes) != mat.ndim:
    raise ValueError(f"bcoo_dynamic_slice: indices must have size mat.ndim={mat.ndim}")
  if not all(0 <= slice_size <= axis_size for slice_size, axis_size in zip(slice_sizes, mat.shape)):
    raise TypeError("slice_sizes must be less than or equal to operand shape, "
                    f"got slice_sizes {slice_sizes} for operand shape {mat.shape}")

  start_batch, start_sparse, start_dense = split_list(start_indices, [mat.n_batch, mat.n_sparse])
  size_batch, size_sparse, size_dense = split_list(slice_sizes, [mat.n_batch, mat.n_sparse])

  data_start = []
  data_sizes = []
  indices_start = []
  indices_sizes = []
  for i, (start, size) in enumerate(zip(start_batch, size_batch)):
    data_is_broadcast = mat.data.shape[i] != mat.shape[i]
    indices_is_broadcast = mat.indices.shape[i] != mat.shape[i]
    data_start.append(0 if data_is_broadcast else start)
    data_sizes.append(1 if data_is_broadcast else size)
    indices_start.append(0 if indices_is_broadcast else start)
    indices_sizes.append(1 if indices_is_broadcast else size)
  data_start.append(0)
  data_sizes.append(mat.nse)
  indices_start.extend([0, 0])
  indices_sizes.extend([mat.nse, mat.n_sparse])
  data_start.extend(start_dense)
  data_sizes.extend(size_dense)

  new_data = lax.dynamic_slice(mat.data, data_start, data_sizes)
  new_indices = lax.dynamic_slice(mat.indices, indices_start, indices_sizes)
  new_shape = slice_sizes

  if mat.n_sparse:
    starts = jnp.array(start_sparse, dtype=new_indices.dtype)
    sizes = jnp.array(size_sparse, dtype=new_indices.dtype)
    sparse_shape = jnp.array(mat.shape[mat.n_batch: mat.n_batch + mat.n_sparse], dtype=new_indices.dtype)
    starts = jnp.where(starts < 0, starts + sparse_shape, starts)
    starts = jnp.clip(starts, 0, sparse_shape - sizes)

    starts = jnp.expand_dims(starts, range(mat.n_batch + 1))
    sizes = jnp.expand_dims(sizes, range(mat.n_batch + 1))
    sparse_shape = jnp.expand_dims(sparse_shape, range(mat.n_batch + 1))

    keep = jnp.all((new_indices >= starts) & (new_indices < starts + sizes), -1, keepdims=True)
    new_indices = jnp.where(keep, new_indices - starts, sparse_shape)

    keep_data = lax.expand_dims(keep[..., 0], range(mat.n_batch + 1, mat.n_batch + 1 + mat.n_dense))
    new_data = jnp.where(keep_data, new_data, 0)

  return BCOO((new_data, new_indices), shape=new_shape)


def _tuple_replace(tup, ind, val):
  return tuple(val if i == ind else t for i, t in enumerate(tup))

def bcoo_reduce_sum(mat, *, axes):
  """Sum array element over given axes.

  Args:
    mat: A BCOO-format array.
    shape: The shape of the target array.
    axes:  A tuple or list or ndarray which contains axes of ``mat`` over which
      sum is performed.

  Returns:
    A BCOO-format array containing the result.
  """
  out_data, out_indices, out_shape = _bcoo_reduce_sum(
      mat.data, mat.indices, spinfo=mat._info, axes=axes)
  return BCOO((out_data, out_indices), shape=out_shape)

def _bcoo_reduce_sum(data, indices, *, spinfo, axes):
  shape = spinfo.shape
  assert all(0 <= a < len(shape) for a in axes)
  n_batch, n_sparse, _, nse = _validate_bcoo(data, indices, shape)
  axes = sorted(set(axes))

  # Sum over dense dimensions -> sum over data
  dense_axes = tuple(ax - n_sparse + 1 for ax in axes if ax >= n_batch + n_sparse)
  data = data.sum(dense_axes)
  if n_sparse:
    # zero-out data corresponding to invalid indices.
    fill_value = jnp.expand_dims(
      jnp.array(shape[n_batch: n_batch + n_sparse], dtype=indices.dtype),
      range(indices.ndim - 1))
    mask = jnp.all(indices < fill_value, -1)
    if data.ndim > mask.ndim:
      mask = lax.expand_dims(mask, tuple(range(mask.ndim, data.ndim)))
    data = jnp.where(mask, data, 0)

  # Sum over sparse dimensions -> drop index; sum is implicit
  sparse_idx = [i for i in range(n_sparse) if i + n_batch not in axes]
  if not sparse_idx:
    indices = jnp.zeros(_tuple_replace(indices.shape, n_batch + 1, 0), indices.dtype)
  else:
    indices = indices[..., np.array(sparse_idx)]

  # Sum over batch dimensions -> reshape into nse
  batch_axes = {ax for ax in axes if ax < n_batch}

  # First handle broadcasted batch dimensions
  for ax in batch_axes:
    if data.shape[ax] == 1:
      if indices.shape[ax] == 1:
        data = data * shape[ax]
      else:
        data = lax.broadcast_in_dim(data, _tuple_replace(data.shape, ax, shape[ax]), tuple(range(data.ndim)))
    else:
      if indices.shape[ax] == 1:
        data = data.sum(ax)
    assert data.shape[ax] == indices.shape[ax]

  new_batch_dims = tuple(sorted(set(range(n_batch)) - batch_axes))
  new_batch_shape = tuple(data.shape[i] for i in new_batch_dims)
  new_nse = int(nse * np.prod([data.shape[i] for i in batch_axes]))

  data = lax.reshape(data,
                     (*new_batch_shape, new_nse, *data.shape[n_batch + 1:]),
                     (*new_batch_dims, *batch_axes, *range(n_batch, data.ndim)))
  indices = lax.reshape(indices,
                        (*new_batch_shape, new_nse, *indices.shape[n_batch + 1:]),
                        (*new_batch_dims, *batch_axes, *range(n_batch, indices.ndim)))

  out_shape = tuple(shape[i] for i in range(len(shape)) if i not in axes)
  return data, indices, out_shape

def bcoo_multiply_sparse(lhs, rhs):
  """An element-wise multiplication of two sparse arrays.

  Args:
    lhs: A BCOO-format array.
    rhs: A BCOO-format array.

  Returns:
    An BCOO-format array containing the result.
  """
  out_data, out_indices, out_shape = _bcoo_multiply_sparse(
      lhs.data, lhs.indices, rhs.data, rhs.indices, lhs_spinfo=lhs._info,
      rhs_spinfo=rhs._info)
  return BCOO((out_data, out_indices), shape=out_shape)

def _bcoo_multiply_sparse(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_spinfo, rhs_spinfo):
  lhs_shape = lhs_spinfo.shape
  rhs_shape = rhs_spinfo.shape

  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  if len(lhs_shape) != len(rhs_shape):
    # Similar requirement as lax.mul:
    raise TypeError("bcoo_multiply_sparse: arrays must have same number of dimensions, "
                    f"got {lhs_shape}, {rhs_shape}")
  if lhs.n_dense != rhs.n_dense:
    raise NotImplementedError("bcoo_multiply_sparse: arrays with differing numbers of "
                              f"dense dimensions: {lhs}, {rhs}")
  n_batch = min(lhs.n_batch, rhs.n_batch)
  _mul = functools.partial(_bcoo_multiply_sparse_unbatched,
                           lhs_shape=lhs_shape[n_batch:],
                           rhs_shape=rhs_shape[n_batch:])
  for _ in range(n_batch):
    _mul = broadcasting_vmap(_mul)
  data, indices = _mul(lhs_data, lhs_indices, rhs_data, rhs_indices)
  return data, indices, jnp.broadcast_shapes(lhs_shape, rhs_shape)

def _bcoo_multiply_sparse_unbatched(lhs_data, lhs_indices, rhs_data, rhs_indices, *, lhs_shape, rhs_shape):
  lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  assert (lhs.n_batch == 0) or (rhs.n_batch == 0)  # Ensured at call site above

  # TODO(jakevdp): this can be made more efficient by utilizing batch structure.
  if lhs.n_batch:
    lhs_data, lhs_indices = _unbatch_bcoo(lhs_data, lhs_indices, lhs_shape)
    lhs = _validate_bcoo(lhs_data, lhs_indices, lhs_shape)
  elif rhs.n_batch:
    rhs_data, rhs_indices = _unbatch_bcoo(rhs_data, rhs_indices, rhs_shape)
    rhs = _validate_bcoo(rhs_data, rhs_indices, rhs_shape)
  dims = jnp.array([i for i, (s1, s2) in enumerate(safe_zip(lhs_shape[:lhs.n_sparse], rhs_shape[:rhs.n_sparse]))
                    if s1 != 1 and s2 != 1], dtype=int)

  # TODO(jakevdp): this nse can be tightened to min(lhs.nse, rhs.nse) if there
  # is no broadcasting and indices are unique.
  nse = lhs.nse * rhs.nse

  # TODO(jakevdp): this is pretty inefficient. Can we do this membership check
  # without constructing the full (lhs.nse, rhs.nse) masking matrix?
  mask = jnp.all(lhs_indices[:, None, dims] == rhs_indices[None, :, dims], -1)
  i_lhs, i_rhs = jnp.nonzero(mask, size=nse, fill_value=(lhs.nse, rhs.nse))
  data = (lhs_data.at[i_lhs].get(mode='fill', fill_value=0) *
          rhs_data.at[i_rhs].get(mode='fill', fill_value=0))
  indices = jnp.maximum(
      lhs_indices.at[i_lhs].get(mode='fill', fill_value=max(lhs_shape, default=0)),
      rhs_indices.at[i_rhs].get(mode='fill', fill_value=max(rhs_shape, default=0)))
  return data, indices

def bcoo_multiply_dense(sp_mat, v):
  """An element-wise multiplication between a sparse and a dense array.

  Args:
    lhs: A BCOO-format array.
    rhs: An ndarray.

  Returns:
    An ndarray containing the result.
  """
  return _bcoo_multiply_dense(*sp_mat._bufs, v, spinfo=sp_mat._info)

def _bcoo_multiply_dense(data, indices, v, *, spinfo):
  """Broadcasted elementwise multiplication between a BCOO array and a dense array."""
  # TODO(jakevdp): the logic here is similar to bcoo_extract... can we reuse that?
  shape = spinfo.shape
  if v.ndim == 0:
    return lax.mul(data, v)
  if shape == v.shape:
    # Note: due to distributive property, no deduplication necessary!
    return lax.mul(data, bcoo_extract(indices, v))

  if lax.broadcast_shapes(v.shape, shape) != shape:
    raise NotImplementedError(
      "multiplication between sparse and dense is only implemented for cases "
      "where the output shape matches the sparse matrix shape. Got "
      f"shape={shape}, v.shape={v.shape}")
  v = lax.expand_dims(v, range(len(shape) - v.ndim))

  props = _validate_bcoo(data, indices, shape)

  def _mul(data, indices, v):
    assert indices.shape[1] == v.ndim - props.n_dense
    ind = tuple(indices[:, i] for i in range(indices.shape[1]))
    ind = tuple(i if s != 1 else 0 for i, s in zip(ind, v.shape))
    return data * v[ind]
  for _ in range(props.n_batch):
    _mul = broadcasting_vmap(_mul)
  return _mul(data, indices, v)

@tree_util.register_pytree_node_class
class BCOO(JAXSparse):
  """Experimental batched COO matrix implemented in JAX

  Args:
    (data, indices) : data and indices in batched COO format.
    shape : shape of sparse array.

  Attributes:
    data : ndarray of shape ``[*batch_dims, nse, *dense_dims]`` containing the
      explicitly stored data within the sparse matrix.
    indices : ndarray of shape ``[*batch_dims, nse, n_sparse]`` containing the
      indices of the explicitly stored data. Duplicate entries will be summed.

  Examples:
    Create a sparse array from a dense array:

    >>> M = jnp.array([[0., 2., 0.], [1., 0., 4.]])
    >>> M_sp = BCOO.fromdense(M)
    >>> M_sp
    BCOO(float32[2, 3], nse=3)

    Examine the internal representation:

    >>> M_sp.data
    DeviceArray([2., 1., 4.], dtype=float32)
    >>> M_sp.indices
    DeviceArray([[0, 1],
                 [1, 0],
                 [1, 2]], dtype=int32)

    Create a dense array from a sparse array:

    >>> M_sp.todense()
    DeviceArray([[0., 2., 0.],
                 [1., 0., 4.]], dtype=float32)

    Create a sparse array from COO data & indices:

    >>> data = jnp.array([1., 3., 5.])
    >>> indices = jnp.array([[0, 0],
    ...                      [1, 1],
    ...                      [2, 2]])
    >>> mat = BCOO((data, indices), shape=(3, 3))
    >>> mat
    BCOO(float32[3, 3], nse=3)
    >>> mat.todense()
    DeviceArray([[1., 0., 0.],
                 [0., 3., 0.],
                 [0., 0., 5.]], dtype=float32)
  """
  # Note: additional BCOO methods are defined in transform.py

  data: jnp.ndarray
  indices: jnp.ndarray
  shape: Shape
  nse = property(lambda self: self.indices.shape[-2])
  dtype = property(lambda self: self.data.dtype)
  n_batch = property(lambda self: self.indices.ndim - 2)
  n_sparse = property(lambda self: self.indices.shape[-1])
  n_dense = property(lambda self: self.data.ndim - 1 - self.n_batch)
  indices_sorted: bool
  unique_indices: bool
  _info = property(lambda self: BCOOInfo(self.shape, self.indices_sorted,
                                         self.unique_indices))
  _bufs = property(lambda self: (self.data, self.indices))

  def __init__(self, args, *, shape, indices_sorted=False,
               unique_indices=False):
    # JAX transforms will sometimes instantiate pytrees with null values, so we
    # must catch that in the initialization of inputs.
    self.data, self.indices = _safe_asarray(args)
    self.indices_sorted = indices_sorted
    self.unique_indices = unique_indices
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
  def fromdense(cls, mat, *, nse=None, index_dtype=np.int32, n_dense=0, n_batch=0):
    """Create a BCOO array from a (dense) :class:`DeviceArray`."""
    return bcoo_fromdense(
      mat, nse=nse, index_dtype=index_dtype, n_dense=n_dense, n_batch=n_batch)

  @classmethod
  def from_scipy_sparse(cls, mat, *, index_dtype=None, n_dense=0, n_batch=0):
    """Create a BCOO array from a :mod:`scipy.sparse` array."""
    if n_dense != 0 or n_batch != 0:
      raise NotImplementedError("BCOO.fromscipy with nonzero n_dense/n_batch")

    mat = mat.tocoo()
    data = jnp.asarray(mat.data)
    indices = jnp.column_stack((mat.row, mat.col)).astype(
        index_dtype or jnp.int32)
    # TODO: determines sorted and unique indices for scipy conversion.
    return cls((data, indices), shape=mat.shape, indices_sorted=False,
               unique_indices=False)

  @classmethod
  def _empty(cls, shape, *, dtype=None, index_dtype='int32', n_dense=0, n_batch=0, nse=0):
    """Create an empty BCOO instance. Public method is sparse.empty()."""
    shape = tuple(shape)
    n_sparse = len(shape) - n_dense - n_batch
    if n_sparse < 0 or n_dense < 0 or n_batch < 0 or nse < 0:
      raise ValueError(f"Invalid inputs: shape={shape}, n_dense={n_dense}, n_batch={n_batch}, nse={nse}")
    batch_shape, sparse_shape, dense_shape = split_list(shape, [n_batch, n_sparse])
    data = jnp.zeros((*batch_shape, nse, *dense_shape), dtype)
    indices = jnp.full((*batch_shape, nse, n_sparse), jnp.array(sparse_shape), index_dtype)
    return cls((data, indices), shape=shape, indices_sorted=True,
               unique_indices=True)

  @classmethod
  def _eye(cls, N, M, k, *, dtype=None, index_dtype='int32', n_batch=0, n_dense=0):
    n_sparse = 2 - n_batch - n_dense
    if n_sparse < 0 or n_dense < 0 or n_batch < 0:
      raise ValueError(f"Invalid inputs: shape={(N, M)}, n_dense={n_dense}, n_batch={n_batch}")

    if k > 0:
      diag_size = min(N, M - k)
    else:
      diag_size = min(N + k, M)

    if diag_size <= 0:
      # if k is out of range, return an empty matrix.
      return cls._empty((N, M), dtype=dtype, index_dtype=index_dtype,
                        n_batch=n_batch, n_dense=n_dense)

    if n_dense > 0 or n_batch > 1:
      # These cases explicitly store all the zeros, so fall back to fromdense.
      return cls.fromdense(jnp.eye(N, M, k, dtype=dtype),
                           n_batch=n_batch, n_dense=n_dense,
                           index_dtype=index_dtype)
    k = jnp.asarray(k)
    if n_batch == 0:
      data = jnp.ones(diag_size, dtype=dtype)
      idx = jnp.arange(diag_size, dtype=index_dtype)
      zero = _const(idx, 0)
      k = _const(idx, k)
      indices = jnp.column_stack([
        lax.sub(idx, lax.cond(k >= 0, lambda: zero, lambda: k)),
        lax.add(idx, lax.cond(k <= 0, lambda: zero, lambda: k))])
    else:
      data = jnp.ones(N, dtype=dtype)
      indices = jnp.arange(N, dtype=index_dtype)
      indices = indices + _const(indices, k)
      if k < 0:
        data = data.at[:abs(k)].set(0)
        indices = indices.at[:abs(k)].set(M)
      elif k > 0:
        data = data.at[M - abs(k):].set(0)
        indices = indices.at[M - abs(k)].set(M)
      data = data[:, None]
      indices = indices[:, None, None]
    return cls((data, indices), shape=(N, M), indices_sorted=True,
               unique_indices=True)

  def update_layout(self, *, n_batch=None, n_dense=None, on_inefficient='error'):
    """Update the storage layout (i.e. n_batch & n_dense) of a BCOO matrix.

    In many cases this can be done without introducing undue storage overhead. However,
    increasing ``mat.n_batch`` or ``mat.n_dense`` will lead to very inefficient storage,
    with many explicitly-stored zeros, unless the new batch or dense dimensions have size
    0 or 1. In such cases, ``update_layout`` will raise a :class:`SparseEfficiencyError`.
    This can be silenced by specifying the ``on_inefficient`` argument.

    Args:
      n_batch : optional(int) the number of batch dimensions in the output matrix. If None,
        then n_batch = mat.n_batch.
      n_dense : optional(int) the number of dense dimensions in the output matrix. If None,
        then n_dense = mat.n_dense.
      on_inefficient : optional(string), one of ``['error', 'warn', None]``. Specify the
        behavior in case of an inefficient reconfiguration. This is defined as a reconfiguration
        where the size of the resulting representation is much larger than the size of the
        input representation.

    Returns:
      mat_out : BCOO array
        A BCOO array representing the same sparse array as the input, with the specified
        layout. ``mat_out.todense()`` will match ``mat.todense()`` up to appropriate precision.
    """
    return bcoo_update_layout(self, n_batch=n_batch, n_dense=n_dense, on_inefficient=on_inefficient)

  def sum_duplicates(self, nse=None, remove_zeros=True):
    """Return a copy of the array with duplicate indices summed.

    Additionally, this operation will result in explicit zero entries removed, and
    indices being sorted in lexicographic order.

    Because the size of the resulting representation depends on the values in the
    arrays, this operation is not compatible with JIT or other transforms. To use
    ``sum_duplicates`` in such cases, you may pass a value to `nse` to specify the
    desired size of the output representation.

    Args:
      nse : integer (optional), if specified, gives the number of specified elements in
        the output sparse representation; if it is larger than the number required, data
        will be padded with zeros and indices will be padded with out-of-bounds values.
        If it is smaller than the number required, data will be silently discarded.
      remove_zeros : bool (default=True). If True, remove explicit zeros from the data
        as part of summing duplicates. If False, then explicit zeros at unique indices
        will remain among the specified elements. Note: remove_zeros=True is incompatible
        with autodiff.
    """
    if remove_zeros:
      return bcoo_eliminate_zeros(self, nse=nse)
    else:
      return bcoo_sum_duplicates(self, nse=nse)

  def sort_indices(self):
    """Return a copy of the matrix with indices sorted."""
    return bcoo_sort_indices(self)

  def todense(self):
    """Create a dense version of the array."""
    return bcoo_todense(self)

  def transpose(self, axes=None):
    """Create a new array containing the transpose."""
    axes = np.arange(self.ndim)[::-1] if axes is None else axes
    mat_T = bcoo_transpose(self, permutation=axes)
    shape_T = tuple(self.shape[i] for i in axes)
    sparse_perm = [p - self.n_batch
                   for p in axes[self.n_batch: self.n_batch + self.n_sparse]]
    if tuple(sparse_perm) == tuple(range(self.n_sparse)):
      is_sorted = self.indices_sorted
    else:
      # TODO: address the corner cases that the transposed indices are sorted.
      # possibly use permutation?
      is_sorted = False
    return BCOO((mat_T.data, mat_T.indices), shape=shape_T,
                indices_sorted=is_sorted, unique_indices=self.unique_indices)

  def tree_flatten(self):
    return (self.data, self.indices), self._info._asdict()


# vmappable handlers
def _bcoo_to_elt(cont, _, val, axis):
  if axis is None:
    return val
  if axis >= val.n_batch:
    raise ValueError(f"Cannot map in_axis={axis} for BCOO array with n_batch={val.n_batch}. "
                     "in_axes for batched BCOO operations must correspond to a batch dimension.")
  return BCOO((cont(val.data, axis), cont(val.indices, axis)),
              shape= val.shape[:axis] + val.shape[axis + 1:])

def _bcoo_from_elt(cont, axis_size, elt, axis):
  if axis > elt.n_batch:
    raise ValueError(f"BCOO: cannot add out_axis={axis} for BCOO array with n_batch={elt.n_batch}. "
                     "BCOO batch axes must be a contiguous block of leading dimensions.")
  return BCOO((cont(axis_size, elt.data, axis), cont(axis_size, elt.indices, axis)),
              shape=elt.shape[:axis] + (axis_size,) + elt.shape[axis:])

batching.register_vmappable(BCOO, int, int, _bcoo_to_elt, _bcoo_from_elt, None)
