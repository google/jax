# Copyright 2018 The JAX Authors.
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

import inspect
import functools
from functools import partial
import warnings

import numpy as np

import jax
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy.vectorize import vectorize
from jax._src import ad_util
from jax._src import api
from jax import lax
from jax._src import dtypes
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters import ad
from jax.interpreters import batching
from jax._src.util import prod
from jax.core import Primitive, ShapedArray, raise_to_shaped
from jax._src.lax.lax import (
    standard_primitive, standard_unop, naryop_dtype_rule, _float, _complex,
    _input_dtype)
from jax._src.lax import control_flow
from jax._src.lax import eigh as lax_eigh
from jax._src.lax import lax as lax_internal
from jax._src.lax import svd as lax_svd
from jax._src.lib import lapack

from jax._src.lib import gpu_linalg
from jax._src.lib import gpu_solver
from jax._src.lib import gpu_sparse

from jax._src.lib import xla_client

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import mhlo

xops = xla_client.ops


# traceables

# TODO(phawkins): remove backward compatibility shim after 2022/08/11.
def _warn_on_positional_kwargs(f):
  """Decorator used for backward compatibility of keyword-only arguments.

  Some functions were changed to mark their keyword arguments as keyword-only.
  This decorator allows existing code to keep working temporarily, while issuing
  a warning if a now keyword-only parameter is passed positionally."""
  sig = inspect.signature(f)
  pos_names = [name for name, p in sig.parameters.items()
               if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
  kwarg_names = [name for name, p in sig.parameters.items()
                 if p.kind == inspect.Parameter.KEYWORD_ONLY]

  # This decorator assumes that all arguments to `f` are either
  # positional-or-keyword or keyword-only.
  assert len(pos_names) + len(kwarg_names) == len(sig.parameters)

  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    if len(args) < len(pos_names):
      a = pos_names[len(args)]
      raise TypeError(f"{f.__name__} missing required positional argument: {a}")

    pos_args = args[:len(pos_names)]
    extra_kwargs = args[len(pos_names):]

    if len(extra_kwargs) > len(kwarg_names):
      raise TypeError(f"{f.__name__} takes at most {len(sig.parameters)} "
                      f" arguments but {len(args)} were given.")

    for name, value in zip(kwarg_names, extra_kwargs):
      if name in kwargs:
        raise TypeError(f"{f.__name__} got multiple values for argument: "
                        f"{name}")

      warnings.warn(f"Argument {name} to {f.__name__} is now a keyword-only "
                    "argument. Support for passing it positionally will be "
                    "removed in an upcoming JAX release.",
                    DeprecationWarning)
      kwargs[name] = value
    return f(*pos_args, **kwargs)

  return wrapped

@_warn_on_positional_kwargs
def cholesky(x, *, symmetrize_input: bool = True):
  """Cholesky decomposition.

  Computes the Cholesky decomposition

  .. math::
    A = L . L^H

  of square matrices, :math:`A`, such that :math:`L`
  is lower triangular. The matrices of :math:`A` must be positive-definite and
  either Hermitian, if complex, or symmetric, if real.

  Args:
    x: A batch of square Hermitian (symmetric if real) positive-definite
      matrices with shape ``[..., n, n]``.
    symmetrize_input: If ``True``, the matrix is symmetrized before Cholesky
      decomposition by computing :math:`\\frac{1}{2}(x + x^H)`. If ``False``,
      only the lower triangle of ``x`` is used; the upper triangle is ignored
      and not accessed.

  Returns:
    The Cholesky decomposition as a matrix with the same dtype as ``x`` and
    shape ``[..., n, n]``. If Cholesky decomposition fails, returns a matrix
    full of NaNs. The behavior on failure may change in the future.
  """
  if symmetrize_input:
    x = symmetrize(x)
  return jnp.tril(cholesky_p.bind(x))

@_warn_on_positional_kwargs
def eig(x, *, compute_left_eigenvectors=True, compute_right_eigenvectors=True):
  """Eigendecomposition of a general matrix.

  Nonsymmetric eigendecomposition is at present only implemented on CPU.
  """
  return eig_p.bind(x, compute_left_eigenvectors=compute_left_eigenvectors,
                    compute_right_eigenvectors=compute_right_eigenvectors)

@_warn_on_positional_kwargs
def eigh(x, *, lower: bool = True, symmetrize_input: bool = True,
         sort_eigenvalues: bool = True, ):
  r"""Eigendecomposition of a Hermitian matrix.

  Computes the eigenvectors and eigenvalues of a complex Hermitian or real
  symmetric square matrix.

  Args:
    x: A batch of square complex Hermitian or real symmetric matrices with shape
      ``[..., n, n]``.
    lower: If ``symmetrize_input`` is ``False``, describes which triangle of the
      input matrix to use. If ``symmetrize_input`` is ``False``, only the
      triangle given by ``lower`` is accessed; the other triangle is ignored and
      not accessed.
    symmetrize_input: If ``True``, the matrix is symmetrized before the
      eigendecomposition by computing :math:`\frac{1}{2}(x + x^H)`.
    sort_eigenvalues: If ``True``, the eigenvalues will be sorted in ascending
      order. If ``False`` the eigenvalues are returned in an
      implementation-defined order.

  Returns:
    A tuple ``(v, w)``.

    ``v`` is an array with the same dtype as ``x`` such that ``v[..., :, i]`` is
    the normalized eigenvector corresponding to eigenvalue ``w[..., i]``.

    ``w`` is an array with the same dtype as ``x`` (or its real counterpart if
    complex) with shape ``[..., n]`` containing the eigenvalues of ``x`` in
    ascending order(each repeated according to its multiplicity).
  """
  if symmetrize_input:
    x = symmetrize(x)
  v, w = eigh_p.bind(x, lower=lower, sort_eigenvalues=sort_eigenvalues)
  return v, w


def lu_pivots_to_permutation(pivots, permutation_size: int):
  """Converts the pivots (row swaps) returned by LU to a permutation.

  We build a permutation rather than applying `pivots` directly to the rows
  of a matrix because lax loops aren't differentiable.

  Args:
    pivots: an int32 array of shape (..., k) of row swaps to perform
    permutation_size: the size of the output permutation. Has to be >= k.

  Returns:
    An int32 array of shape (..., permutation_size).
  """
  permutation = lu_pivots_to_permutation_p.bind(
      pivots, permutation_size=int(permutation_size))
  return permutation


def lu(x):
  """LU decomposition with partial pivoting.

  Computes the matrix decomposition:

  .. math::
    P.A = L.U

  where :math:`P` is a permutation of the rows of :math:`A`, :math:`L` is a
  lower-triangular matrix with unit-diagonal elements, and :math:`U` is an
  upper-triangular matrix.

  Args:
    x: A batch of matrices with shape ``[..., m, n]``.

  Returns:
    A tuple ``(lu, pivots, permutation)``.

    ``lu`` is a batch of matrices with the same shape and dtype as ``x``
    containing the :math:`L` matrix in its lower triangle and the :math:`U`
    matrix in its upper triangle. The (unit) diagonal elements of :math:`L` are
    not represented explicitly.

    ``pivots`` is an int32 array with shape ``[..., min(m, n)]`` representing a
    sequence of row swaps that should be performed on :math:`A`.

    ``permutation`` is an alternative representation of the sequence of row
    swaps as a permutation, represented as an int32 array with shape
    ``[..., m]``.
  """
  lu, pivots, permutation = lu_p.bind(x)
  return lu, pivots, permutation

@_warn_on_positional_kwargs
def qr(x, *, full_matrices: bool = True):
  """QR decomposition.

  Computes the QR decomposition

  .. math::
    A = Q . R

  of matrices :math:`A`, such that :math:`Q` is a unitary (orthogonal) matrix,
  and :math:`R` is an upper-triangular matrix.

  Args:
    x: A batch of matrices with shape ``[..., m, n]``.
    full_matrices: Determines if full or reduced matrices are returned; see
      below.

  Returns:
    A pair of arrays ``(q, r)``.

    Array ``q`` is a unitary (orthogonal) matrix,
    with shape ``[..., m, m]`` if ``full_matrices=True``, or
    ``[..., m, min(m, n)]`` if ``full_matrices=False``.

    Array ``r`` is an upper-triangular matrix with shape ``[..., m, n]`` if
    ``full_matrices=True``, or ``[..., min(m, n), n]`` if
    ``full_matrices=False``.
  """
  q, r = qr_p.bind(x, full_matrices=full_matrices)
  return q, r

# TODO: Add `max_qdwh_iterations` to the function signature for TPU SVD.
@_warn_on_positional_kwargs
def svd(x, *, full_matrices=True, compute_uv=True):
  """Singular value decomposition.

  Returns the singular values if compute_uv is False, otherwise returns a triple
  containing the left singular vectors, the singular values and the adjoint of
  the right singular vectors.
  """
  result = svd_p.bind(x, full_matrices=full_matrices, compute_uv=compute_uv)
  if compute_uv:
    s, u, v = result
    return u, s, v
  else:
    s, = result
    return s

@_warn_on_positional_kwargs
def triangular_solve(a, b, *, left_side: bool = False, lower: bool = False,
                     transpose_a: bool = False, conjugate_a: bool = False,
                     unit_diagonal: bool = False):
  r"""Triangular solve.

  Solves either the matrix equation

  .. math::
    \mathit{op}(A) . X = B

  if ``left_side`` is ``True`` or

  .. math::
    X . \mathit{op}(A) = B

  if ``left_side`` is ``False``.

  ``A`` must be a lower or upper triangular square matrix, and where
  :math:`\mathit{op}(A)` may either transpose :math:`A` if ``transpose_a``
  is ``True`` and/or take its complex conjugate if ``conjugate_a`` is ``True``.

  Args:
    a: A batch of matrices with shape ``[..., m, m]``.
    b: A batch of matrices with shape ``[..., m, n]`` if ``left_side`` is
      ``True`` or shape ``[..., n, m]`` otherwise.
    left_side: describes which of the two matrix equations to solve; see above.
    lower: describes which triangle of ``a`` should be used. The other triangle
      is ignored.
    transpose_a: if ``True``, the value of ``a`` is transposed.
    conjugate_a: if ``True``, the complex conjugate of ``a`` is used in the
      solve. Has no effect if ``a`` is real.
    unit_diagonal: if ``True``, the diagonal of ``a`` is assumed to be unit
      (all 1s) and not accessed.

  Returns:
    A batch of matrices the same shape and dtype as ``b``.
  """
  conjugate_a = conjugate_a and jnp.issubdtype(lax.dtype(a), jnp.complexfloating)
  singleton = jnp.ndim(b) == jnp.ndim(a) - 1
  if singleton:
    b = jnp.expand_dims(b, -1 if left_side else -2)
  out = triangular_solve_p.bind(
      a, b, left_side=left_side, lower=lower, transpose_a=transpose_a,
      conjugate_a=conjugate_a, unit_diagonal=unit_diagonal)
  if singleton:
    out = out[..., 0] if left_side else out[..., 0, :]
  return out


# utilities
@partial(vectorize, signature='(n,m),(m)->(n)')
def _matvec_multiply(a, b):
  return lax.dot(a, b, precision=lax.Precision.HIGHEST)

def _check_solve_shapes(a, b):
  if not (a.ndim >= 2 and b.ndim in [a.ndim, a.ndim - 1] and
          a.shape[-1] == a.shape[-2] == b.shape[a.ndim - 2]):
    raise ValueError(
        "The arguments to solve must have shapes a=[..., m, m] and "
        f"b=[..., m, k] or b=[..., m]; got a={a.shape} and b={b.shape}")

def _solve(a, b):
  _check_solve_shapes(a, b)

  # Broadcast leading dimensions of b to the shape of a, as is required by
  # custom_linear_solve.
  out_shape = tuple(d_a if d_b == 1 else d_b
                    for d_a, d_b in zip(a.shape[:-1] + (1,), b.shape))
  b = jnp.broadcast_to(b, out_shape)

  # With custom_linear_solve, we can reuse the same factorization when
  # computing sensitivities. This is considerably faster.
  lu_, _, permutation = lu(lax.stop_gradient(a))
  custom_solve = partial(
      lax.custom_linear_solve,
      lambda x: _matvec_multiply(a, x),
      solve=lambda _, x: lu_solve(lu_, permutation, x, trans=0),
      transpose_solve=lambda _, x: lu_solve(lu_, permutation, x, trans=1))
  if a.ndim == b.ndim + 1:
    # b.shape == [..., m]
    return custom_solve(b)
  else:
    # b.shape == [..., m, k]
    return api.vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)

def _T(x): return jnp.swapaxes(x, -1, -2)
def _H(x): return jnp.conj(_T(x))
def symmetrize(x): return (x + _H(x)) / 2

def _unpack_tuple(f, n):
  def g(c, *args, **kwargs):
    t = f(c, *args, **kwargs)
    return (xops.GetTupleElement(t, i) for i in range(n))
  return g

# primitives

_cpu_lapack_types = {np.dtype(np.float32), np.dtype(np.float64),
                     np.dtype(np.complex64), np.dtype(np.complex128)}

# Cholesky decomposition

def _cholesky_jvp_rule(primals, tangents):
  x, = primals
  sigma_dot, = tangents
  L = jnp.tril(cholesky_p.bind(x))

  # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
  def phi(X):
    l = jnp.tril(X)
    return l / lax.expand_dims(
        lax_internal._const(X, 1) + jnp.eye(X.shape[-1], dtype=X.dtype),
        range(l.ndim - 2))

  tmp = triangular_solve(L, sigma_dot, left_side=False, transpose_a=True,
                         conjugate_a=True, lower=True)
  L_dot = lax.batch_matmul(L, phi(triangular_solve(
      L, tmp, left_side=True, transpose_a=False, lower=True)),
      precision=lax.Precision.HIGHEST)
  return L, L_dot

def _cholesky_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return cholesky(x), 0

cholesky_p = standard_unop(_float | _complex, 'cholesky')
ad.primitive_jvps[cholesky_p] = _cholesky_jvp_rule
batching.primitive_batchers[cholesky_p] = _cholesky_batching_rule

def _cholesky_lowering(ctx, x):
  return mhlo.CholeskyOp(x, lower=ir.BoolAttr.get(True)).results

mlir.register_lowering(cholesky_p, _cholesky_lowering)

def _cholesky_cpu_gpu_lowering(potrf_impl, ctx, operand):
  operand_aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  batch_dims = operand_aval.shape[:-2]
  result, info = potrf_impl(operand_aval.dtype, operand, lower=True)
  ok = mlir.compare_mhlo(
      info, mlir.full_like_aval(0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")
  return [_broadcasting_select_mhlo(
      mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1, 1),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
      result, _nan_like_mhlo(out_aval))]

mlir.register_lowering(
    cholesky_p,
    partial(_cholesky_cpu_gpu_lowering, lapack.potrf_mhlo),
    platform='cpu')
mlir.register_lowering(
  cholesky_p,
  partial(_cholesky_cpu_gpu_lowering, gpu_solver.cuda_potrf),
  platform='cuda')
mlir.register_lowering(
  cholesky_p,
  partial(_cholesky_cpu_gpu_lowering, gpu_solver.rocm_potrf),
  platform='rocm')

# Asymmetric eigendecomposition

def eig_impl(operand, *, compute_left_eigenvectors, compute_right_eigenvectors):
  return (
    xla.apply_primitive(eig_p, operand,
                        compute_left_eigenvectors=compute_left_eigenvectors,
                        compute_right_eigenvectors=compute_right_eigenvectors))

def eig_lower(*args, **kw):
  raise NotImplementedError(
    "Nonsymmetric eigendecomposition is only implemented on the CPU backend. "
    "If your matrix is symmetric or Hermitian, you should use eigh instead.")

def eig_abstract_eval(operand, *, compute_left_eigenvectors,
                      compute_right_eigenvectors):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError("Argument to nonsymmetric eigendecomposition must have "
                       "shape [..., n, n], got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    dtype = np.complex64 if dtypes.finfo(operand.dtype).bits == 32 else np.complex128
    dtype = dtypes.canonicalize_dtype(dtype)
    vl = vr = operand.update(shape=batch_dims + (n, n), dtype=dtype)
    w = operand.update(shape=batch_dims + (n,), dtype=dtype)
  else:
    raise NotImplementedError

  output = [w]
  if compute_left_eigenvectors:
    output.append(vl)
  if compute_right_eigenvectors:
    output.append(vr)

  return tuple(output)

def _eig_cpu_lowering(ctx, operand, *, compute_left_eigenvectors,
                      compute_right_eigenvectors):
  operand_aval, = ctx.avals_in
  out_aval = ctx.avals_out[0]
  batch_dims = operand_aval.shape[:-2]

  w, vl, vr, info = lapack.geev_mhlo(operand_aval.dtype, operand,
                                     jobvl=compute_left_eigenvectors,
                                     jobvr=compute_right_eigenvectors)

  ok = mlir.compare_mhlo(
      info, mlir.full_like_aval(0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")
  w = _broadcasting_select_mhlo(
      mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1,),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
      w, _nan_like_mhlo(out_aval))
  output = [w]

  if compute_left_eigenvectors:
    aval = ctx.avals_out[len(output)]
    vl = _broadcasting_select_mhlo(
        mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1, 1),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
        vl, _nan_like_mhlo(aval))
    output.append(vl)

  if compute_right_eigenvectors:
    aval = ctx.avals_out[len(output)]
    vr = _broadcasting_select_mhlo(
        mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1, 1),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
        vr, _nan_like_mhlo(aval))
    output.append(vr)

  return output


def eig_batching_rule(batched_args, batch_dims, *, compute_left_eigenvectors,
                      compute_right_eigenvectors):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)

  return (eig_p.bind(x, compute_left_eigenvectors=compute_left_eigenvectors,
                     compute_right_eigenvectors=compute_right_eigenvectors),
          (0,) * (1 + compute_left_eigenvectors + compute_right_eigenvectors))

def eig_jvp_rule(primals, tangents, *, compute_left_eigenvectors,
                 compute_right_eigenvectors):
  if compute_left_eigenvectors or compute_right_eigenvectors:
    raise NotImplementedError(
        'The derivatives of eigenvectors are not implemented, only '
        'eigenvalues. See '
        'https://github.com/google/jax/issues/2748 for discussion.')
  # Formula for derivative of eigenvalues w.r.t. a is eqn 4.60 in
  # https://arxiv.org/abs/1701.00392
  a, = primals
  da, = tangents
  l, v = eig(a, compute_left_eigenvectors=False)
  return [l], [jnp.sum(_solve(v, da.astype(v.dtype)) * _T(v), -1)]

eig_p = Primitive('eig')
eig_p.multiple_results = True
eig_p.def_impl(eig_impl)
eig_p.def_abstract_eval(eig_abstract_eval)
mlir.register_lowering(eig_p, eig_lower)
mlir.register_lowering(eig_p, _eig_cpu_lowering, platform='cpu')
batching.primitive_batchers[eig_p] = eig_batching_rule
ad.primitive_jvps[eig_p] = eig_jvp_rule


# Symmetric/Hermitian eigendecomposition


def eigh_jacobi(x, *, lower: bool = True, sort_eigenvalues: bool = True):
  """Helper Jacobi eigendecomposition implemented by XLA.

  Used as a subroutine of QDWH-eig on TPU."""
  w, v = eigh_jacobi_p.bind(x, lower=lower, sort_eigenvalues=sort_eigenvalues)
  return w, v

def _eigh_jacobi_impl(operand, *, lower, sort_eigenvalues):
  w, v = xla.apply_primitive(eigh_jacobi_p, operand, lower=lower,
                             sort_eigenvalues=sort_eigenvalues)
  return w, v

def _eigh_jacobi_abstract_eval(operand, *, lower, sort_eigenvalues):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError(
        "Argument to symmetric eigendecomposition must have shape [..., n, n],"
        "got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    w = operand.update(shape=batch_dims + (n,),
                       dtype=lax_internal._complex_basetype(operand.dtype))
    v = operand.update(shape=batch_dims + (n, n))
  else:
    w, v = operand, operand
  return w, v

def _eigh_jacobi_translation_rule(ctx, avals_in, avals_out, operand, *, lower,
                                  sort_eigenvalues):
  operand_aval, = avals_in
  if operand_aval.shape[-1] == 0:
    return [xops.Real(xops.Reshape(operand, operand_aval.shape[:-1])), operand]
  v, w = xops.Eigh(operand, lower=lower, sort_eigenvalues=sort_eigenvalues)
  return w, v

eigh_jacobi_p = Primitive('eigh_jacobi')
eigh_jacobi_p.multiple_results = True
eigh_jacobi_p.def_impl(_eigh_jacobi_impl)
eigh_jacobi_p.def_abstract_eval(_eigh_jacobi_abstract_eval)
xla.register_translation(eigh_jacobi_p, _eigh_jacobi_translation_rule)


def _eigh_impl(operand, *, lower, sort_eigenvalues):
  v, w = xla.apply_primitive(eigh_p, operand, lower=lower,
                             sort_eigenvalues=sort_eigenvalues)
  return v, w

def _eigh_abstract_eval(operand, *, lower, sort_eigenvalues):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError(
        "Argument to symmetric eigendecomposition must have shape [..., n, n],"
        "got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    v = operand.update(shape=batch_dims + (n, n))
    w = operand.update(shape=batch_dims + (n,),
                       dtype=lax_internal._complex_basetype(operand.dtype))
  else:
    v, w = operand, operand
  return v, w

def _eigh_cpu_gpu_lowering(syevd_impl, ctx, operand, *, lower,
                           sort_eigenvalues):
  del sort_eigenvalues  # The CPU/GPU implementations always sort.
  operand_aval, = ctx.avals_in
  v_aval, w_aval = ctx.avals_out
  batch_dims = operand_aval.shape[:-2]
  v, w, info = syevd_impl(operand_aval.dtype, operand, lower=lower)
  zeros = mlir.full_like_aval(0, ShapedArray(batch_dims, np.dtype(np.int32)))
  ok = mlir.compare_mhlo(info, zeros, "EQ", "SIGNED")
  v = _broadcasting_select_mhlo(
      mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1, 1),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
      v, _nan_like_mhlo(v_aval))
  w = _broadcasting_select_mhlo(
      mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1,),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
      w, _nan_like_mhlo(w_aval))
  return [v, w]

def _eigh_tpu_impl(x, *, lower, sort_eigenvalues):
  *_, m, n = x.shape
  assert m == n, (m, n)

  termination_size = 256

  if m <= termination_size:
    eig_vals, eig_vecs = eigh_jacobi(x, lower=lower,
                                     sort_eigenvalues=sort_eigenvalues)
    return eig_vecs, eig_vals

  def eigh_qdwh(x):
    if len(x.shape) > 2:
      return control_flow.map(eigh_qdwh, x)

    # We should only look at elements from the lower/upper triangle. Reflects
    # that triangle into the other triangle to form a Hermitian matrix.
    if lower:
      mask = jnp.tri(n, k=0, dtype=bool)
    else:
      mask = jnp.logical_not(jnp.tri(n, k=-1, dtype=bool))
    if dtypes.issubdtype(x.dtype, jnp.complexfloating):
      re = lax.select(mask, lax.real(x), _T(lax.real(x)))
      if lower:
        im_mask = jnp.tri(n, k=-1, dtype=bool)
      else:
        im_mask = jnp.logical_not(jnp.tri(n, k=0, dtype=bool))
      im = lax.select(im_mask, lax.imag(x), jnp.zeros_like(lax.imag(x)))
      im = lax.select(mask, im, -_T(im))
      x = lax.complex(re, im)
    else:
      x = lax.select(mask, x, _T(x))

    return lax_eigh.eigh(x, sort_eigenvalues=sort_eigenvalues,
                         termination_size=termination_size)

  eig_vals, eig_vecs = eigh_qdwh(x)
  return eig_vecs, eig_vals

def _eigh_jvp_rule(primals, tangents, *, lower, sort_eigenvalues):
  # Derivative for eigh in the simplest case of distinct eigenvalues.
  # This is classic nondegenerate perurbation theory, but also see
  # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
  # The general solution treating the case of degenerate eigenvalues is
  # considerably more complicated. Ambitious readers may refer to the general
  # methods below or refer to degenerate perturbation theory in physics.
  # https://www.win.tue.nl/analysis/reports/rana06-33.pdf and
  # https://people.orie.cornell.edu/aslewis/publications/99-clarke.pdf
  a, = primals
  a_dot, = tangents

  v, w_real = eigh_p.bind(symmetrize(a), lower=lower,
                          sort_eigenvalues=sort_eigenvalues)

  # for complex numbers we need eigenvalues to be full dtype of v, a:
  w = w_real.astype(a.dtype)
  eye_n = jnp.eye(a.shape[-1], dtype=a.dtype)
  # carefully build reciprocal delta-eigenvalue matrix, avoiding NaNs.
  Fmat = jnp.reciprocal(eye_n + w[..., jnp.newaxis, :] - w[..., jnp.newaxis]) - eye_n
  # eigh impl doesn't support batch dims, but future-proof the grad.
  dot = partial(lax.dot if a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)
  vdag_adot_v = dot(dot(_H(v), a_dot), v)
  dv = dot(v, jnp.multiply(Fmat, vdag_adot_v))
  dw = jnp.real(jnp.diagonal(vdag_adot_v, axis1=-2, axis2=-1))
  return (v, w_real), (dv, dw)

def _eigh_batching_rule(batched_args, batch_dims, *, lower, sort_eigenvalues):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return eigh_p.bind(x, lower=lower, sort_eigenvalues=sort_eigenvalues), (0, 0)

eigh_p = Primitive('eigh')
eigh_p.multiple_results = True
eigh_p.def_impl(_eigh_impl)
eigh_p.def_abstract_eval(_eigh_abstract_eval)
ad.primitive_jvps[eigh_p] = _eigh_jvp_rule
batching.primitive_batchers[eigh_p] = _eigh_batching_rule

mlir.register_lowering(
    eigh_p, partial(_eigh_cpu_gpu_lowering, lapack.syevd_mhlo),
    platform='cpu')

if gpu_solver is not None:
  mlir.register_lowering(
    eigh_p, partial(_eigh_cpu_gpu_lowering, gpu_solver.cuda_syevd),
    platform='cuda')
  mlir.register_lowering(
    eigh_p, partial(_eigh_cpu_gpu_lowering, gpu_solver.rocm_syevd),
    platform='rocm')

mlir.register_lowering(
    eigh_p, mlir.lower_fun(_eigh_tpu_impl, multiple_results=True),
    platform='tpu')


triangular_solve_dtype_rule = partial(
    naryop_dtype_rule, _input_dtype, (_float | _complex, _float | _complex),
    'triangular_solve')

def triangular_solve_shape_rule(a, b, *, left_side=False, **unused_kwargs):
  if a.ndim < 2:
    msg = "triangular_solve requires a.ndim to be at least 2, got {}."
    raise TypeError(msg.format(a.ndim))
  if b.ndim < 2:
    msg = "triangular_solve requires b.ndim to be at least 2, got {}."
    raise TypeError(msg.format(b.ndim))
  if a.shape[-1] != a.shape[-2]:
    msg = ("triangular_solve requires the last two dimensions of a to be equal "
           "in size, got a.shape of {}.")
    raise TypeError(msg.format(a.shape))
  if a.shape[:-2] != b.shape[:-2]:
    msg = ("triangular_solve requires both arguments to have the same number "
           "of dimensions and equal batch dimensions, got {} and {}.")
    raise TypeError(msg.format(a.shape, b.shape))
  common_dim = -2 if left_side else -1
  if a.shape[-1] != b.shape[common_dim]:
    msg = "Incompatible shapes for arguments to triangular_solve: {} and {}."
    raise TypeError(msg.format(a.shape, b.shape))
  return b.shape

def triangular_solve_jvp_rule_a(
    g_a, ans, a, b, *, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  m, n = b.shape[-2:]
  k = 1 if unit_diagonal else 0
  g_a = jnp.tril(g_a, k=-k) if lower else jnp.triu(g_a, k=k)
  g_a = lax.neg(g_a)
  g_a = jnp.swapaxes(g_a, -1, -2) if transpose_a else g_a
  g_a = jnp.conj(g_a) if conjugate_a else g_a
  dot = partial(lax.dot if g_a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)

  def a_inverse(rhs):
    return triangular_solve(a, rhs, left_side=left_side, lower=lower,
                            transpose_a=transpose_a, conjugate_a=conjugate_a,
                            unit_diagonal=unit_diagonal)

  # triangular_solve is about the same cost as matrix multplication (~n^2 FLOPs
  # for matrix/vector inputs). Order these operations in whichever order is
  # cheaper.
  if left_side:
    assert g_a.shape[-2:] == a.shape[-2:] == (m, m) and ans.shape[-2:] == (m, n)
    if m > n:
      return a_inverse(dot(g_a, ans))  # A^{-1} (∂A X)
    else:
      return dot(a_inverse(g_a), ans)  # (A^{-1} ∂A) X
  else:
    assert g_a.shape[-2:] == a.shape[-2:] == (n, n) and ans.shape[-2:] == (m, n)
    if m < n:
      return a_inverse(dot(ans, g_a))  # (X ∂A) A^{-1}
    else:
      return dot(ans, a_inverse(g_a))  # X (∂A A^{-1})

def triangular_solve_transpose_rule(
    cotangent, a, b, *, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  # Triangular solve is nonlinear in its first argument and linear in its second
  # argument, analogous to `div` but swapped.
  assert not ad.is_undefined_primal(a) and ad.is_undefined_primal(b)
  if type(cotangent) is ad_util.Zero:
    cotangent_b = ad_util.Zero(b.aval)
  else:
    cotangent_b = triangular_solve(a, cotangent, left_side=left_side,
                                   lower=lower, transpose_a=not transpose_a,
                                   conjugate_a=conjugate_a,
                                   unit_diagonal=unit_diagonal)
  return [None, cotangent_b]


def triangular_solve_batching_rule(batched_args, batch_dims, *, left_side,
                                   lower, transpose_a, conjugate_a,
                                   unit_diagonal):
  x, y = batched_args
  bx, by = batch_dims
  if bx is batching.not_mapped:
    if left_side:
      y = batching.moveaxis(y, by, -1)
      y_flat = y.reshape(y.shape[:-2] + (y.shape[-2] * y.shape[-1],))
      bdim_out = y.ndim - 1
    else:
      y = batching.moveaxis(y, by, -2)
      y_flat = y.reshape(y.shape[:-3]  + (y.shape[-3] * y.shape[-2], y.shape[-1]))
      bdim_out = y.ndim - 2
    out_flat = triangular_solve(
        x, y_flat, left_side=left_side, lower=lower,
        transpose_a=transpose_a, conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal)
    return out_flat.reshape(y.shape), bdim_out
  else:
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    x = batching.bdim_at_front(x, bx, size)
    y = batching.bdim_at_front(y, by, size)
    return triangular_solve(x, y, left_side=left_side, lower=lower,
                            transpose_a=transpose_a, conjugate_a=conjugate_a,
                            unit_diagonal=unit_diagonal), 0

triangular_solve_p = standard_primitive(
    triangular_solve_shape_rule, triangular_solve_dtype_rule,
    'triangular_solve')
ad.defjvp2(triangular_solve_p,
           triangular_solve_jvp_rule_a,
           lambda g_b, _, a, b, **kws: triangular_solve(a, g_b, **kws))
ad.primitive_transposes[triangular_solve_p] = triangular_solve_transpose_rule
batching.primitive_batchers[triangular_solve_p] = triangular_solve_batching_rule


def _triangular_solve_lowering(
    ctx, a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  out_aval, = ctx.avals_out
  if conjugate_a and not transpose_a:
    a = chlo.ConjOp(a)
    conjugate_a = False
  if not transpose_a:
    transpose = "NO_TRANSPOSE"
  else:
    transpose = "ADJOINT" if conjugate_a else "TRANSPOSE"
  return mhlo.TriangularSolveOp(
      mlir.aval_to_ir_type(out_aval), a, b, ir.BoolAttr.get(left_side),
      ir.BoolAttr.get(lower), ir.BoolAttr.get(unit_diagonal),
      mhlo.TransposeAttr.get(transpose)).results

mlir.register_lowering(triangular_solve_p, _triangular_solve_lowering)

def _triangular_solve_cpu_lower(
    ctx, a, b, *, left_side, lower, transpose_a,
    conjugate_a, unit_diagonal):
  a_aval, _ = ctx.avals_in

  if conjugate_a and not transpose_a:
    a = chlo.ConjOp(a).result
    conjugate_a = False
  if len(a_aval.shape) == 2 and np.dtype(a_aval.dtype) in _cpu_lapack_types:
    alpha = mlir.ir_constant(np.array(1, dtype=a_aval.dtype))
    return [lapack.trsm_mhlo(
      a_aval.dtype, alpha,
      a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal)]
  else:
    # Fall back to the HLO implementation for unsupported types or batching.
    # TODO: Consider swapping XLA for LAPACK in batched case
    if transpose_a:
      transpose = "ADJOINT" if conjugate_a else "TRANSPOSE"
    else:
      transpose = "NO_TRANSPOSE"
    return mhlo.TriangularSolveOp(b.type, a, b, ir.BoolAttr.get(left_side),
                                  ir.BoolAttr.get(lower),
                                  ir.BoolAttr.get(unit_diagonal),
                                  mhlo.TransposeAttr.get(transpose)).results

mlir.register_lowering(triangular_solve_p, _triangular_solve_cpu_lower,
                       platform='cpu')


def _triangular_solve_gpu_lower(
    trsm_impl, ctx, a, b, *, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  a_aval, _ = ctx.avals_in
  m, n = a_aval.shape[-2:]
  batch = prod(a_aval.shape[:-2])
  if conjugate_a and not transpose_a:
    a = chlo.ConjOp(a).result
    conjugate_a = False
  if batch > 1 and m <= 256 and n <= 256:
    return [trsm_impl(a_aval.dtype, a, b, left_side, lower, transpose_a,
                      conjugate_a, unit_diagonal)]
  else:
    # Use the XLA implementation for unbatched triangular_solve.
    if transpose_a:
      transpose = "ADJOINT" if conjugate_a else "TRANSPOSE"
    else:
      transpose = "NO_TRANSPOSE"
    return mhlo.TriangularSolveOp(b.type, a, b, ir.BoolAttr.get(left_side),
                                  ir.BoolAttr.get(lower),
                                  ir.BoolAttr.get(unit_diagonal),
                                  mhlo.TransposeAttr.get(transpose)).results

mlir.register_lowering(
    triangular_solve_p,
    partial(_triangular_solve_gpu_lower, gpu_solver.cuda_trsm),
    platform='cuda')
mlir.register_lowering(
    triangular_solve_p,
    partial(_triangular_solve_gpu_lower, gpu_solver.rocm_trsm),
    platform='rocm')


# Support operation for LU decomposition: Transformation of the pivots returned
# by LU decomposition into permutations.


# Define this outside lu_pivots_to_permutation to ensure fori_loop cache hits
def _lu_pivots_body_fn(i, permutation_and_swaps):
  permutation, swaps = permutation_and_swaps
  batch_dims = swaps.shape[:-1]
  j = swaps[..., i]
  iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in batch_dims))
  x = permutation[..., i]
  y = permutation[iotas + (j,)]
  permutation = permutation.at[..., i].set(y)
  return permutation.at[iotas + (j,)].set(x), swaps


def _generic_lu_pivots_to_permutation(swaps, permutation_size):
  """Converts the pivots (row swaps) returned by LU to a permutation.

  We build a permutation rather than applying `swaps` directly to the rows
  of a matrix because lax loops aren't differentiable.

  Args:
    swaps: an array of shape (..., k) of row swaps to perform
    permutation_size: the size of the output permutation. Should be >= k.
  Returns:
    An int32 array of shape (..., m).
  """
  assert len(swaps.shape) >= 1
  batch_dims = swaps.shape[:-1]
  k = swaps.shape[-1]
  m = permutation_size

  permutation = lax.broadcasted_iota(jnp.int32, batch_dims + (m,),
                                     len(batch_dims))
  if m == 0:
    return permutation
  result, _ = lax.fori_loop(np.array(0, np.int32), np.array(k, np.int32),
                            _lu_pivots_body_fn, (permutation, swaps))
  return result


def _lu_pivots_to_permutation_abstract_eval(pivots, *, permutation_size):
  pivots = raise_to_shaped(pivots)
  if isinstance(pivots, ShapedArray):
    if pivots.ndim < 1 or pivots.dtype != np.dtype(np.int32):
      raise ValueError(
          'Argument to lu_pivots_to_permutation must have rank >= 1 and dtype '
          'int32. Got shape={} and dtype={}'.format(pivots.shape, pivots.dtype))

    if permutation_size < pivots.shape[-1]:
      raise ValueError(
          'Output permutation size {} has to exceed the trailing dimension of '
          'the pivots. Got shape {}'.format(permutation_size, pivots.shape))

    batch_dims = pivots.shape[:-1]
    permutations = pivots.update(shape=batch_dims + (permutation_size,))
  else:
    permutations = pivots

  return permutations


def _lu_pivots_to_permutation_batching_rule(batched_args, batch_dims, *,
                                            permutation_size):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return lu_pivots_to_permutation_p.bind(
      x, permutation_size=permutation_size), 0

def _lu_pivots_to_permutation_gpu_lowering(lowering, ctx, pivots, *,
                                           permutation_size):
  return [lowering(pivots, permutation_size=permutation_size)]


lu_pivots_to_permutation_p = Primitive('lu_pivots_to_permutation')
lu_pivots_to_permutation_p.multiple_results = False
lu_pivots_to_permutation_p.def_impl(
    partial(xla.apply_primitive, lu_pivots_to_permutation_p))
lu_pivots_to_permutation_p.def_abstract_eval(
    _lu_pivots_to_permutation_abstract_eval)
batching.primitive_batchers[lu_pivots_to_permutation_p] = (
    _lu_pivots_to_permutation_batching_rule)
mlir.register_lowering(
    lu_pivots_to_permutation_p,
    mlir.lower_fun(_generic_lu_pivots_to_permutation, multiple_results=False))
mlir.register_lowering(
    lu_pivots_to_permutation_p,
    partial(_lu_pivots_to_permutation_gpu_lowering,
            gpu_linalg.cuda_lu_pivots_to_permutation),
    platform='cuda')
mlir.register_lowering(
    lu_pivots_to_permutation_p,
    partial(_lu_pivots_to_permutation_gpu_lowering,
            gpu_linalg.hip_lu_pivots_to_permutation),
    platform='rocm')


# LU decomposition

# Computes a pivoted LU decomposition such that
# PA = LU
# In the style of LAPACK, LU are stored in the same matrix.

def _lu_unblocked(a):
  """Unblocked LU decomposition, as a rolled loop."""
  m, n = a.shape
  def body(k, state):
    pivot, perm, a = state
    m_idx = jnp.arange(m)
    n_idx = jnp.arange(n)

    if jnp.issubdtype(a.dtype, jnp.complexfloating):
      t = a[:, k]
      magnitude = jnp.abs(jnp.real(t)) + jnp.abs(jnp.imag(t))
    else:
      magnitude = jnp.abs(a[:, k])
    i = jnp.argmax(jnp.where(m_idx >= k, magnitude, -jnp.inf))
    pivot = pivot.at[k].set(i)
    a = a.at[[k, i],].set(a[[i, k],])
    perm = perm.at[[i, k],].set(perm[[k, i],])

    # a[k+1:, k] /= a[k, k], adapted for loop-invariant shapes
    x = a[k, k]
    a = a.at[:, k].set(jnp.where(m_idx > k, a[:, k] / x, a[:, k]))

    # a[k+1:, k+1:] -= jnp.outer(a[k+1:, k], a[k, k+1:])
    a = a - jnp.where((m_idx[:, None] > k) & (n_idx > k),
                     jnp.outer(a[:, k], a[k, :]), jnp.array(0, dtype=a.dtype))
    return pivot, perm, a

  pivot = jnp.zeros((min(m, n),), dtype=jnp.int32)
  perm = jnp.arange(m, dtype=jnp.int32)
  if m == 0 and n == 0:
    # If the array is empty, the loop body never executes but tracing it to a
    # jaxpr fails because the indexing cannot succeed.
    return (pivot, perm, a)
  return lax.fori_loop(0, min(m, n), body, (pivot, perm, a))


def _lu_blocked(a, block_size=128):
  """Blocked LU decomposition, as an unrolled loop."""
  m, n = a.shape
  r = min(m, n)
  pivot = jnp.zeros((r,), dtype=jnp.int32)
  perm = jnp.arange(m, dtype=jnp.int32)
  for k in range(0, r, block_size):
    b = min(r - k, block_size)
    block_pivot, block_perm, lu_block = _lu_unblocked(a[k:, k:k+b])

    pivot = pivot.at[k:k+b].set(block_pivot + k)
    perm = perm.at[k:].set(perm[block_perm + k])
    a = a.at[k:, :].set(a[block_perm + k, :])
    a = a.at[k:, k:k+b].set(lu_block)

    if k + b < n:
      a = a.at[k:k+b, k+b:].set(
        triangular_solve(a[k:k+b, k:k+b], a[k:k+b, k+b:], left_side=True,
                         lower=True, unit_diagonal=True))
      a = a.at[k+b:, k+b:].add(-lax.dot(a[k+b:, k:k+b], a[k:k+b, k+b:],
                                        precision=lax.Precision.HIGHEST))
  return a, pivot, perm

def _lu_python(x):
  """Default LU decomposition in Python, where no better version exists."""
  batch_dims = x.shape[:-2]
  fn = _lu_blocked
  for _ in range(len(batch_dims)):
    fn = api.vmap(fn)

  return fn(x)

def _lu_impl(operand):
  lu, pivot, perm = xla.apply_primitive(lu_p, operand)
  return lu, pivot, perm

def _lu_abstract_eval(operand):
  operand = raise_to_shaped(operand)
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to LU decomposition must have ndims >= 2")

    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    pivot = operand.update(shape=batch_dims + (min(m, n),), dtype=jnp.int32)
    perm = operand.update(shape=batch_dims + (m,), dtype=jnp.int32)
  else:
    pivot = operand
    perm = operand
  return operand, pivot, perm

def _lu_jvp_rule(primals, tangents):
  a, = primals
  a_dot, = tangents
  lu, pivots, permutation = lu_p.bind(a)

  a_shape = jnp.shape(a)
  m, n = a_shape[-2:]
  dtype = lax.dtype(a)
  k = min(m, n)

  batch_dims = a_shape[:-2]
  iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in batch_dims + (1,)))
  x = a_dot[iotas[:-1] + (permutation, slice(None))]

  # Differentiation of Matrix Functionals Using Triangular Factorization
  # F. R. De Hoog, R. S. Anderssen, and M. A. Lukas
  #
  #     LU = A
  # ==> L'U + LU' = A'
  # ==> inv(L) . L' + U' . inv(U) = inv(L) A' inv(U)
  # ==> L' = L . tril(inv(L) . A' . inv(U), -1)
  #     U' = triu(inv(L) . A' . inv(U)) . U

  ndims = len(a_shape)
  l_padding = [(0, 0, 0)] * ndims
  l_padding[-1] = (0, m - k, 0)
  zero = lax_internal._const(lu, 0)
  l = lax.pad(jnp.tril(lu[..., :, :k], -1), zero, l_padding)
  l = l + lax.expand_dims(jnp.eye(m, m, dtype=dtype), range(l.ndim - 2))

  u_eye = lax.pad(jnp.eye(n - k, n - k, dtype=dtype), zero,
                  ((k, 0, 0), (k, 0, 0)))
  u_padding = [(0, 0, 0)] * ndims
  u_padding[-2] = (0, n - k, 0)
  u = (lax.pad(jnp.triu(lu[..., :k, :]), zero, u_padding) +
       lax.expand_dims(u_eye, range(lu.ndim - 2)))

  la = triangular_solve(l, x, left_side=True, transpose_a=False, lower=True,
                        unit_diagonal=True)
  lau = triangular_solve(u, la, left_side=False, transpose_a=False,
                         lower=False)

  l_dot = jnp.matmul(l, jnp.tril(lau, -1))
  u_dot = jnp.matmul(jnp.triu(lau), u)
  lu_dot = l_dot + u_dot
  return (lu, pivots, permutation), (lu_dot, ad_util.Zero.from_value(pivots),
                                     ad_util.Zero.from_value(permutation))


def _lu_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return lu_p.bind(x), (0, 0, 0)

def _lu_cpu_gpu_lowering(getrf_impl, ctx, operand):
  operand_aval, = ctx.avals_in
  out_aval, pivot_aval, perm_aval = ctx.avals_out
  batch_dims = operand_aval.shape[:-2]
  m = operand_aval.shape[-2]
  lu, pivot, info = getrf_impl(operand_aval.dtype, operand)
  # Subtract 1 from the pivot to get 0-based indices.
  pivot = mhlo.SubtractOp(pivot, mlir.full_like_aval(1, pivot_aval)).result
  ok = mlir.compare_mhlo(
      info, mlir.full_like_aval(0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "GE", "SIGNED")
  lu = _broadcasting_select_mhlo(
      mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1, 1),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
      lu, _nan_like_mhlo(out_aval))
  sub_ctx = ctx.replace(primitive=None, avals_in=[pivot_aval], avals_out=[perm_aval])
  perm_fn = mlir.lower_fun(lambda x: lu_pivots_to_permutation(x, m),
                           multiple_results=False)
  perm, = perm_fn(sub_ctx, pivot)
  return [lu, pivot, perm]


def _lu_tpu_translation_rule(ctx, avals_in, avals_out, operand):
  return xops.LU(operand)


lu_p = Primitive('lu')
lu_p.multiple_results = True
lu_p.def_impl(_lu_impl)
lu_p.def_abstract_eval(_lu_abstract_eval)
mlir.register_lowering(lu_p, mlir.lower_fun(_lu_python, multiple_results=True))
ad.primitive_jvps[lu_p] = _lu_jvp_rule
batching.primitive_batchers[lu_p] = _lu_batching_rule

mlir.register_lowering(lu_p,
                       partial(_lu_cpu_gpu_lowering, lapack.getrf_mhlo),
                       platform='cpu')

mlir.register_lowering(
    lu_p, partial(_lu_cpu_gpu_lowering, gpu_solver.cuda_getrf),
    platform='cuda')
mlir.register_lowering(
    lu_p, partial(_lu_cpu_gpu_lowering, gpu_solver.rocm_getrf),
    platform='rocm')

xla.register_translation(lu_p, _lu_tpu_translation_rule, platform='tpu')


@partial(vectorize, excluded={3}, signature='(n,n),(n),(n,k)->(n,k)')
def _lu_solve_core(lu, permutation, b, trans):
  m = lu.shape[0]
  x = jnp.reshape(b, (m, np.prod(b.shape[1:])))
  if trans == 0:
    x = x[permutation, :]
    x = triangular_solve(lu, x, left_side=True, lower=True, unit_diagonal=True)
    x = triangular_solve(lu, x, left_side=True, lower=False)
  elif trans == 1 or trans == 2:
    conj = trans == 2
    x = triangular_solve(lu, x, left_side=True, lower=False, transpose_a=True,
                         conjugate_a=conj)
    x = triangular_solve(lu, x, left_side=True, lower=True, unit_diagonal=True,
                         transpose_a=True, conjugate_a=conj)
    x = x[jnp.argsort(permutation), :]
  else:
    raise ValueError(f"'trans' value must be 0, 1, or 2, got {trans}")
  return lax.reshape(x, b.shape)


@partial(api.jit, static_argnums=(3,))
def _lu_solve(lu, permutation, b, trans):
  if len(lu.shape) < 2 or lu.shape[-1] != lu.shape[-2]:
    raise ValueError("last two dimensions of LU decomposition must be equal, "
                     "got shape {}".format(lu.shape))
  if len(b.shape) < 1:
    raise ValueError("b matrix must have rank >= 1, got shape {}"
                     .format(b.shape))
  # Broadcasting follows NumPy's convention for linalg.solve: the RHS is
  # treated as a (batched) vector if the number of dimensions differ by 1.
  # Otherwise, broadcasting rules apply.
  rhs_vector = lu.ndim == b.ndim + 1
  if rhs_vector:
    if b.shape[-1] != lu.shape[-1]:
      raise ValueError("When LU decomposition matrix and b have the same "
                       "number of dimensions, last axis of LU decomposition "
                       "matrix (shape {}) and b array (shape {}) must match"
                       .format(lu.shape, b.shape))
    b = b[..., jnp.newaxis]
  else:
    if b.shape[-2] != lu.shape[-1]:
      raise ValueError("When LU decomposition matrix and b different "
                       "numbers of dimensions, last axis of LU decomposition "
                       "matrix (shape {}) and second to last axis of b array "
                       "(shape {}) must match"
                       .format(lu.shape, b.shape))
  x = _lu_solve_core(lu, permutation, b, trans)
  return x[..., 0] if rhs_vector else x


def lu_solve(lu, permutation, b, trans=0):
  """LU solve with broadcasting."""
  return _lu_solve(lu, permutation, b, trans)


# QR decomposition

# QR decomposition is implemented as a composition of two lower-level primitives
# geqrf and orgqr. The names, while cryptic Fortran alphabet soup, are LAPACK's
# names for the primitives, and we stick with them for consistency.

def geqrf(a):
  """Computes the QR decomposition of a matrix.

  Args:
    a: an ``[..., m, n]`` batch of matrices, with floating-point or complex type.
  Returns:
    An ``(a, taus)`` pair where ``r`` is in the upper triangle of ``a``,
    ``q`` is represented in the lower triangle of ``a`` and in ``taus`` as
    elementary Householder reflectors.
  """
  a_out, taus = geqrf_p.bind(a)
  return a_out, taus

def _geqrf_abstract_eval(operand):
  if not isinstance(operand, ShapedArray):
    raise NotImplementedError("Unsupported aval in geqrf_abstract_eval: "
                              f"{operand.aval}")
  if operand.ndim < 2:
    raise ValueError("Argument to QR decomposition must have ndims >= 2")
  *batch_dims, m, n = operand.shape
  taus = operand.update(shape=(*batch_dims, min(m, n)))
  return operand, taus

def _geqrf_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  return geqrf(batching.moveaxis(x, bd, 0)), (0, 0)

def _geqrf_translation_rule(ctx, avals_in, avals_out, operand):
  return xops.QrDecomposition(operand)

def _geqrf_cpu_gpu_lowering(geqrf_impl, batched_geqrf_impl, ctx, a):
  a_aval, taus_aval = ctx.avals_out
  *batch_dims, m, n = a_aval.shape
  batch = prod(batch_dims)

  if batch == 0 or m == 0 or n == 0:
    return mlir.full_like_aval(0, a_aval), mlir.full_like_aval(0, taus_aval)

  if (batched_geqrf_impl is not None and batch > 1 and m // batch <= 128 and
      n // batch <= 128):
    a_out, taus = batched_geqrf_impl(a_aval.dtype, a)
  else:
    a_out, taus, info_geqrf = geqrf_impl(a_aval.dtype, a)
    zeros = mlir.full_like_aval(0, ShapedArray(batch_dims, np.dtype(np.int32)))
    ok = mlir.compare_mhlo(info_geqrf, zeros, "EQ", "SIGNED")
    ok_a = mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get((*batch_dims, 1, 1),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result
    a_out = _broadcasting_select_mhlo(ok_a, a_out, _nan_like_mhlo(a_aval))
    ok_taus = mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get((*batch_dims, 1,),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result
    taus = _broadcasting_select_mhlo(ok_taus, taus, _nan_like_mhlo(taus_aval))
  return a_out, taus

geqrf_p = Primitive('geqrf')
geqrf_p.multiple_results = True
geqrf_p.def_impl(partial(xla.apply_primitive, geqrf_p))
geqrf_p.def_abstract_eval(_geqrf_abstract_eval)
batching.primitive_batchers[geqrf_p] = _geqrf_batching_rule
xla.register_translation(geqrf_p, _geqrf_translation_rule)

mlir.register_lowering(
    geqrf_p, partial(_geqrf_cpu_gpu_lowering, lapack.geqrf_mhlo, None),
    platform='cpu')
mlir.register_lowering(
    geqrf_p,
    partial(_geqrf_cpu_gpu_lowering, gpu_solver.cuda_geqrf,
            gpu_solver.cuda_geqrf_batched),
    platform='cuda')
mlir.register_lowering(
    geqrf_p,
    partial(_geqrf_cpu_gpu_lowering, gpu_solver.rocm_geqrf,
            gpu_solver.rocm_geqrf_batched),
    platform='rocm')


# orgqr: product of elementary Householder reflectors

def orgqr(a, taus):
  """Product of elementary Householder reflectors.

  Args:
    a: A matrix with shape ``[..., m, n]``, whose lower triangle contains
      elementary Householder reflectors.
    taus: A vector with shape ``[..., k]``, where ``k < min(m, n)``, containing
      the scalar factors of the elementary Householder reflectors.

  Returns:
    A batch of orthogonal (unitary) matrices with the same shape as ``a``,
    containing the products of the elementary Householder reflectors.
  """
  return orgqr_p.bind(a, taus)


def _orgqr_abstract_eval(a, taus):
  if not isinstance(a, ShapedArray) or not isinstance(taus, ShapedArray):
    raise NotImplementedError("Unsupported aval in orgqr_abstract_eval: "
                              f"{a.aval} {taus.aval}")
  if a.ndim < 2:
    raise ValueError("Argument to QR decomposition must have ndims >= 2")
  *batch_dims, m, n = a.shape
  *taus_batch_dims, k = taus.shape
  if a.dtype != taus.dtype or batch_dims != taus_batch_dims or k > min(m, n):
    raise ValueError(f"Type mismatch for orgqr: a={a} taus={taus}")
  return a

def _orgqr_batching_rule(batched_args, batch_dims):
  a, taus = batched_args
  b_a, b_taus, = batch_dims
  return orgqr(batching.moveaxis(a, b_a, 0),
               batching.moveaxis(taus, b_taus, 0)), (0,)

def _orgqr_translation_rule(ctx, avals_in, avals_out, a, taus):
  return [xops.ProductOfElementaryHouseholderReflectors(a, taus)]

def _orgqr_cpu_gpu_lowering(orgqr_impl, ctx, a, taus):
  a_aval, _ = ctx.avals_in
  *batch_dims, m, n = a_aval.shape

  if m == 0 or n == 0:
    return [mlir.full_like_aval(0, a_aval)]

  a, info_orgqr = orgqr_impl(a_aval.dtype, a, taus)
  zeros = mlir.full_like_aval(0, ShapedArray(batch_dims, np.dtype(np.int32)))
  ok = mlir.compare_mhlo(info_orgqr, zeros, "EQ", "SIGNED")
  ok = mhlo.BroadcastInDimOp(
        ir.RankedTensorType.get((*batch_dims, 1, 1),
                                ir.IntegerType.get_signless(1)),
        ok, mlir.dense_int_elements(range(len(batch_dims)))).result
  a = _broadcasting_select_mhlo(ok, a, _nan_like_mhlo(a_aval))
  return [a]


orgqr_p = Primitive('orgqr')
orgqr_p.def_impl(partial(xla.apply_primitive, orgqr_p))
orgqr_p.def_abstract_eval(_orgqr_abstract_eval)
batching.primitive_batchers[orgqr_p] = _orgqr_batching_rule
xla.register_translation(orgqr_p, _orgqr_translation_rule)

mlir.register_lowering(
    orgqr_p, partial(_orgqr_cpu_gpu_lowering, lapack.orgqr_mhlo),
    platform='cpu')
mlir.register_lowering(
    orgqr_p,
    partial(_orgqr_cpu_gpu_lowering, gpu_solver.cuda_orgqr),
    platform='cuda')
mlir.register_lowering(
    orgqr_p,
    partial(_orgqr_cpu_gpu_lowering, gpu_solver.rocm_orgqr),
    platform='rocm')


def _qr_impl(operand, *, full_matrices):
  q, r = xla.apply_primitive(qr_p, operand, full_matrices=full_matrices)
  return q, r

def _qr_translation_rule(ctx, avals_in, avals_out, operand, *, full_matrices):
  operand_aval, = avals_in
  shape = operand_aval.shape
  m, n = shape[-2:]
  if m == 0 or n == 0:
    return [_eye_like_xla(ctx.builder, avals_out[0]),
            _zeros_like_xla(ctx.builder, avals_out[1])]
  return xops.QR(operand, full_matrices)

def _qr_abstract_eval(operand, *, full_matrices):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to QR decomposition must have ndims >= 2")
    *batch_dims, m, n = operand.shape
    k = m if full_matrices else min(m, n)
    q = operand.update(shape=(*batch_dims, m, k))
    r = operand.update(shape=(*batch_dims, k, n))
  else:
    q = operand
    r = operand
  return q, r

def qr_jvp_rule(primals, tangents, *, full_matrices):
  # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
  x, = primals
  dx, = tangents
  q, r = qr_p.bind(x, full_matrices=False)
  *_, m, n = x.shape
  if m < n or (full_matrices and m != n):
    raise NotImplementedError(
      "Unimplemented case of QR decomposition derivative")
  dx_rinv = triangular_solve(r, dx)  # Right side solve by default
  qt_dx_rinv = jnp.matmul(_H(q), dx_rinv)
  qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
  do = qt_dx_rinv_lower - _H(qt_dx_rinv_lower)  # This is skew-symmetric
  # The following correction is necessary for complex inputs
  I = lax.expand_dims(jnp.eye(n, dtype=do.dtype), range(qt_dx_rinv.ndim - 2))
  do = do + I * (qt_dx_rinv - qt_dx_rinv.real.astype(qt_dx_rinv.dtype))
  dq = jnp.matmul(q, do - qt_dx_rinv) + dx_rinv
  dr = jnp.matmul(qt_dx_rinv - do, r)
  return (q, r), (dq, dr)

def _qr_batching_rule(batched_args, batch_dims, *, full_matrices):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return qr_p.bind(x, full_matrices=full_matrices), (0, 0)

def _qr_lowering(a, *, full_matrices):
  *batch_dims, m, n = a.shape
  if m == 0 or n == 0:
    k = m if full_matrices else min(m, n)
    q = jnp.broadcast_to(jnp.eye(m, k, dtype=a.dtype), (*batch_dims, m, k))
    r = jnp.empty((*batch_dims, k, n), dtype=a.dtype)
    return q, r

  r, taus = geqrf(a)
  if m < n:
    q = orgqr(r[..., :m, :m], taus)
  elif full_matrices:
    pads = [(0, 0, 0)] * (len(batch_dims) + 1) + [(0, m - n, 0)]
    q = lax.pad(r, lax_internal._zero(r), pads)
    q = orgqr(q, taus)
  else:
    q = orgqr(r, taus)
    r = r[..., :n, :n]
  r = jnp.triu(r)
  return q, r


qr_p = Primitive('qr')
qr_p.multiple_results = True
qr_p.def_impl(_qr_impl)
qr_p.def_abstract_eval(_qr_abstract_eval)

ad.primitive_jvps[qr_p] = qr_jvp_rule
batching.primitive_batchers[qr_p] = _qr_batching_rule

mlir.register_lowering(qr_p, mlir.lower_fun(_qr_lowering));


# Singular value decomposition

def _svd_impl(operand, *, full_matrices, compute_uv):
  return xla.apply_primitive(svd_p, operand, full_matrices=full_matrices,
                             compute_uv=compute_uv)


def _zeros_like_xla(c, aval):
  zero = xops.Constant(c, np.array(0, aval.dtype))
  return xops.Broadcast(zero, aval.shape)

def _eye_like_xla(c, aval):
  iota_shape = xla_client.Shape.array_shape(
      xla.dtype_to_primitive_type(np.dtype(np.int32)), aval.shape)
  x = xops.Eq(xops.Iota(c, iota_shape, len(aval.shape) - 1),
              xops.Iota(c, iota_shape, len(aval.shape) - 2))
  return xops.ConvertElementType(x, xla.dtype_to_primitive_type(aval.dtype))

def _svd_abstract_eval(operand, *, full_matrices, compute_uv):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to singular value decomposition must have ndims >= 2")

    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    s = operand.update(shape=batch_dims + (min(m, n),),
                       dtype=lax_internal._complex_basetype(operand.dtype))
    if compute_uv:
      u = operand.update(shape=batch_dims + (m, m if full_matrices else min(m, n)))
      vt = operand.update(shape=batch_dims + (n if full_matrices else min(m, n), n))
      return s, u, vt
    else:
      return s,
  else:
    raise NotImplementedError

def _svd_jvp_rule(primals, tangents, *, full_matrices, compute_uv):
  A, = primals
  dA, = tangents
  s, U, Vt = svd_p.bind(A, full_matrices=False, compute_uv=True)

  if compute_uv and full_matrices:
    # TODO: implement full matrices case, documented here: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    raise NotImplementedError(
      "Singular value decomposition JVP not implemented for full matrices")

  Ut, V = _H(U), _H(Vt)
  s_dim = s[..., None, :]
  dS = jnp.matmul(jnp.matmul(Ut, dA), V)
  ds = jnp.real(jnp.diagonal(dS, 0, -2, -1))

  if not compute_uv:
    return (s,), (ds,)

  s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
  s_diffs_zeros = jnp.eye(s.shape[-1], dtype=s.dtype)  # jnp.ones((), dtype=A.dtype) * (s_diffs == 0.)  # is 1. where s_diffs is 0. and is 0. everywhere else
  s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
  F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
  dSS = s_dim.astype(A.dtype) * dS  # dS.dot(jnp.diag(s))
  SdS = _T(s_dim.astype(A.dtype)) * dS  # jnp.diag(s).dot(dS)

  s_zeros = (s == 0).astype(s.dtype)
  s_inv = 1 / (s + s_zeros) - s_zeros
  s_inv_mat = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(s_inv)
  dUdV_diag = .5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
  dU = jnp.matmul(U, F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
  dV = jnp.matmul(V, F.astype(A.dtype) * (SdS + _H(SdS)))

  m, n = A.shape[-2:]
  if m > n:
    I = lax.expand_dims(jnp.eye(m, dtype=A.dtype), range(U.ndim - 2))
    dU = dU + jnp.matmul(I - jnp.matmul(U, Ut), jnp.matmul(dA, V)) / s_dim.astype(A.dtype)
  if n > m:
    I = lax.expand_dims(jnp.eye(n, dtype=A.dtype), range(V.ndim - 2))
    dV = dV + jnp.matmul(I - jnp.matmul(V, Vt), jnp.matmul(_H(dA), U)) / s_dim.astype(A.dtype)

  return (s, U, Vt), (ds, dU, _H(dV))

def _empty_svd(a, *, full_matrices, compute_uv):
  batch_shape = a.shape[:-2]
  m, n = a.shape[-2:]
  s = jnp.empty(batch_shape + (0,), dtype=lax_internal._complex_basetype(a.dtype))
  if not compute_uv:
    return (s,)
  if full_matrices:
    size = max(m, n)
    u = jnp.broadcast_to(jnp.eye(size, dtype=a.dtype), batch_shape + (size, size))
  else:
    u = jnp.empty(batch_shape + (m, n), dtype=a.dtype)
  v = jnp.empty(batch_shape + (0, 0), dtype=a.dtype)
  if m < n:
    u, v = v, u
  return s, u, v

def _svd_cpu_gpu_lowering(gesvd_impl, ctx, operand, *, full_matrices,
                          compute_uv):
  operand_aval, = ctx.avals_in
  s_aval = ctx.avals_out[0]
  m, n = operand_aval.shape[-2:]
  batch_dims = operand_aval.shape[:-2]

  if m == 0 or n == 0:
    return mlir.lower_fun(_empty_svd, multiple_results=True)(
      ctx, operand, full_matrices=full_matrices, compute_uv=compute_uv)

  s, u, vt, info = gesvd_impl(operand_aval.dtype, operand,
                              full_matrices=full_matrices,
                              compute_uv=compute_uv)
  zeros = mlir.full_like_aval(0, ShapedArray(batch_dims, np.dtype(np.int32)))
  ok = mlir.compare_mhlo(info, zeros, "EQ", "SIGNED")
  s = _broadcasting_select_mhlo(
      mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1,),
                                  ir.IntegerType.get_signless(1)),
          ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
      s, _nan_like_mhlo(s_aval))
  result = [s]

  if compute_uv:
    u_aval, vt_aval = ctx.avals_out[1:]
    u = _broadcasting_select_mhlo(
        mhlo.BroadcastInDimOp(
            ir.RankedTensorType.get(batch_dims + (1, 1),
                                    ir.IntegerType.get_signless(1)),
            ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
        u, _nan_like_mhlo(u_aval))
    vt = _broadcasting_select_mhlo(
        mhlo.BroadcastInDimOp(
            ir.RankedTensorType.get(batch_dims + (1, 1),
                                    ir.IntegerType.get_signless(1)),
            ok, mlir.dense_int_elements(range(len(batch_dims)))).result,
        vt, _nan_like_mhlo(vt_aval))
    result += [u, vt]

  return result

def _svd_tpu(a, *, full_matrices, compute_uv):
  batch_dims = a.shape[:-2]

  fn = partial(lax_svd.svd, full_matrices=full_matrices, compute_uv=compute_uv)
  for _ in range(len(batch_dims)):
    fn = api.vmap(fn)

  if compute_uv:
    u, s, vh = fn(a)
    return [s, u, vh]
  else:
    s = fn(a)
    return [s]

def _svd_tpu_lowering_rule(ctx, operand, *, full_matrices, compute_uv):
  operand_aval, = ctx.avals_in
  m, n = operand_aval.shape[-2:]

  if m == 0 or n == 0:
    return mlir.lower_fun(_empty_svd, multiple_results=True)(
      ctx, operand, full_matrices=full_matrices, compute_uv=compute_uv)

  return mlir.lower_fun(_svd_tpu, multiple_results=True)(
      ctx, operand, full_matrices=full_matrices, compute_uv=compute_uv)

def _svd_batching_rule(batched_args, batch_dims, *, full_matrices, compute_uv):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  outs = svd_p.bind(x, full_matrices=full_matrices, compute_uv=compute_uv)

  if compute_uv:
    return outs, (0, 0, 0)
  else:
    return outs, (0,)

svd_p = Primitive('svd')
svd_p.multiple_results = True
svd_p.def_impl(_svd_impl)
svd_p.def_abstract_eval(_svd_abstract_eval)
ad.primitive_jvps[svd_p] = _svd_jvp_rule
batching.primitive_batchers[svd_p] = _svd_batching_rule

mlir.register_lowering(
    svd_p, partial(_svd_cpu_gpu_lowering, lapack.gesdd_mhlo),
    platform='cpu')
mlir.register_lowering(
  svd_p, partial(_svd_cpu_gpu_lowering, gpu_solver.cuda_gesvd),
  platform='cuda')
mlir.register_lowering(
  svd_p, partial(_svd_cpu_gpu_lowering, gpu_solver.rocm_gesvd),
  platform='rocm')

mlir.register_lowering(svd_p, _svd_tpu_lowering_rule)

tridiagonal_solve_p = Primitive('tridiagonal_solve')
tridiagonal_solve_p.multiple_results = False
tridiagonal_solve_p.def_impl(
    functools.partial(xla.apply_primitive, tridiagonal_solve_p))
tridiagonal_solve_p.def_abstract_eval(
    lambda dl, d, du, b, *, m, n, ldb, t:
    ShapedArray(b.shape, b.dtype, b.weak_type))
# TODO(tomhennigan): Consider AD rules using lax.custom_linear_solve?

def _tridiagonal_solve_gpu_lowering(lowering, ctx, dl, d, du, b, *, m, n, ldb, t):
  return [lowering(dl, d, du, b, m=m, n=n, ldb=ldb,
                   t=dtypes.canonicalize_dtype(t))]
mlir.register_lowering(
    tridiagonal_solve_p,
    partial(_tridiagonal_solve_gpu_lowering, gpu_sparse.cuda_gtsv2),
    platform='cuda')
mlir.register_lowering(
    tridiagonal_solve_p,
    partial(_tridiagonal_solve_gpu_lowering, gpu_sparse.rocm_gtsv2),
    platform='rocm')


def _tridiagonal_solve_jax(dl, d, du, b, **kw):
  """Pure JAX implementation of `tridiagonal_solve`."""
  del kw
  prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1])
  fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_)
  fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2])
  bwd1 = lambda x_, x: x[0] - x[1] * x_
  double = lambda f, args: (f(*args), f(*args))

  # Forward pass.
  _, tu_ = lax.scan(lambda tu_, x: double(fwd1, (tu_, x)),
                    du[0] / d[0],
                    (d, du, dl),
                    unroll=32)

  _, b_ = lax.scan(lambda b_, x: double(fwd2, (b_, x)),
                   b[0] / d[0],
                   (b, d, prepend_zero(tu_), dl),
                   unroll=32)

  # Backsubstitution.
  _, x_ = lax.scan(lambda x_, x: double(bwd1, (x_, x)),
                   b_[-1],
                   (b_[::-1], tu_[::-1]),
                   unroll=32)

  return x_[::-1]

mlir.register_lowering(tridiagonal_solve_p, mlir.lower_fun(
    _tridiagonal_solve_jax, multiple_results=False))


def _tridiagonal_solve_vmap(batched_args, batch_dims, **_):
  return jax.vmap(_tridiagonal_solve_jax, in_axes=batch_dims)(*batched_args), 0
batching.primitive_batchers[tridiagonal_solve_p] = _tridiagonal_solve_vmap


def tridiagonal_solve(dl, d, du, b):
  r"""Computes the solution of a tridiagonal linear system.

  This function computes the solution of a tridiagonal linear system:

  .. math::
    A . X = B

  Args:
    dl: The lower diagonal of A: ``dl[i] := A[i, i-1]`` for i in ``[0,m)``.
      Note that ``dl[0] = 0``.
    d: The middle diagnoal of A: ``d[i]  := A[i, i]`` for i in ``[0,m)``.
    du: The upper diagonal of A: ``du[i] := A[i, i+1]`` for i in ``[0,m)``.
      Note that ``dl[m - 1] = 0``.
    b: Right hand side matrix.

  Returns:
    Solution ``X`` of tridiagonal system.
  """
  if dl.ndim != 1 or d.ndim != 1 or du.ndim != 1:
    raise ValueError('dl, d and du must be vectors')

  if dl.shape != d.shape or d.shape != du.shape:
    raise ValueError(
        f'dl={dl.shape}, d={d.shape} and du={du.shape} must all be `[m]`')

  if b.ndim != 2:
    raise ValueError(f'b={b.shape} must be a matrix')

  m, = dl.shape
  if m < 3:
    raise ValueError(f'm ({m}) must be >= 3')

  ldb, n = b.shape
  if ldb < max(1, m):
    raise ValueError(f'Leading dimension of b={ldb} must be ≥ max(1, {m})')

  if dl.dtype != d.dtype or d.dtype != du.dtype or du.dtype != b.dtype:
    raise ValueError(f'dl={dl.dtype}, d={d.dtype}, du={du.dtype} and '
                     f'b={b.dtype} must be the same dtype,')

  t = dl.dtype
  if t not in (np.float32, np.float64):
    raise ValueError(f'Only f32/f64 are supported, got {t}')

  return tridiagonal_solve_p.bind(dl, d, du, b, m=m, n=n, ldb=ldb, t=t)


# Schur Decomposition


@_warn_on_positional_kwargs
def schur(x, *,
          compute_schur_vectors=True,
          sort_eig_vals=False,
          select_callable=None):
  return schur_p.bind(
      x,
      compute_schur_vectors=compute_schur_vectors,
      sort_eig_vals=sort_eig_vals,
      select_callable=select_callable)


def _schur_impl(operand, *, compute_schur_vectors, sort_eig_vals,
                select_callable):
  return xla.apply_primitive(
      schur_p,
      operand,
      compute_schur_vectors=compute_schur_vectors,
      sort_eig_vals=sort_eig_vals,
      select_callable=select_callable)

def _schur_lowering(ctx, *args, **kwargs):
  raise NotImplementedError(
      "Schur decomposition is only implemented on the CPU backend.")

def _schur_abstract_eval(operand, *, compute_schur_vectors, sort_eig_vals,
                         select_callable):

  if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
    raise ValueError("Argument to Schur decomposition must have "
                     "shape [..., n, n], got shape {}".format(operand.shape))

  batch_dims = operand.shape[:-2]
  n = operand.shape[-1]
  dtype = operand.dtype
  dtype = dtypes.canonicalize_dtype(dtype)
  T = operand.update(shape=batch_dims + (n, n), dtype=dtype)
  vs = operand.update(shape=batch_dims + (n, n), dtype=dtype)

  return (T, vs) if compute_schur_vectors else (T,)

def _schur_cpu_lowering(ctx, operand, *, compute_schur_vectors, sort_eig_vals,
                        select_callable):
  operand_aval, = ctx.avals_in
  batch_dims = operand_aval.shape[:-2]

  gees_result = lapack.gees_mhlo(operand_aval.dtype, operand,
                                  jobvs=compute_schur_vectors,
                                  sort=sort_eig_vals,
                                  select=select_callable)
  # Number of return values depends on value of sort_eig_vals.
  T, vs, *_, info = gees_result

  ok = mlir.compare_mhlo(
      info, mlir.full_like_aval(0, ShapedArray(batch_dims, np.dtype(np.int32))),
      "EQ", "SIGNED")

  T = _broadcasting_select_mhlo(
      mhlo.BroadcastInDimOp(
          ir.RankedTensorType.get(batch_dims + (1, 1),
                                  ir.IntegerType.get_signless(1)),
          ok,
          mlir.dense_int_elements(range(len(batch_dims)))).result,
      T, _nan_like_mhlo(ctx.avals_out[0]))
  output = [T]
  if compute_schur_vectors:
    vs = _broadcasting_select_mhlo(
        mhlo.BroadcastInDimOp(
            ir.RankedTensorType.get(batch_dims + (1, 1),
                                    ir.IntegerType.get_signless(1)),
            ok,
            mlir.dense_int_elements(range(len(batch_dims)))).result,
        vs, _nan_like_mhlo(ctx.avals_out[1]))

    output.append(vs)

  return output


def _schur_batching_rule(batched_args, batch_dims, *, compute_schur_vectors,
                         sort_eig_vals, select_callable):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)

  return schur_p.bind(
      x,
      compute_schur_vectors=compute_schur_vectors,
      sort_eig_vals=sort_eig_vals,
      select_callable=select_callable), (0,) * (1 + compute_schur_vectors)


def _schur_jvp_rule(primals, tangents, *, compute_schur_vectors, sort_eig_vals):
  raise NotImplementedError(
      'The differentiation rules for the Schur factorization have not been implemented.'
  )


schur_p = Primitive('schur')
schur_p.multiple_results = True
schur_p.def_impl(_schur_impl)
schur_p.def_abstract_eval(_schur_abstract_eval)
mlir.register_lowering(schur_p, _schur_lowering)
mlir.register_lowering(schur_p, _schur_cpu_lowering, platform='cpu')
batching.primitive_batchers[schur_p] = _schur_batching_rule
ad.primitive_jvps[schur_p] = _schur_jvp_rule


# Utilities

def _nan_like_mhlo(aval):
  if jnp.issubdtype(aval.dtype, np.complexfloating):
    return mlir.full_like_aval(np.nan + np.nan * 1j, aval)
  else:
    return mlir.full_like_aval(np.nan, aval)

def _broadcasting_select_mhlo(which, x, y):
  """Wrapper around XLA `Select` that broadcasts its arguments."""
  which_type, x_type, y_type = (
    ir.RankedTensorType(v.type) for v in (which, x, y))
  out_shape = list(lax_internal.broadcast_shapes(
      tuple(which_type.shape), tuple(x_type.shape), tuple(y_type.shape)))
  bcast_dims = lambda shape: mlir.dense_int_elements(
      range(len(out_shape) - len(shape), len(out_shape)))
  if which_type.shape != out_shape:
    which = mhlo.BroadcastInDimOp(
        ir.RankedTensorType.get(out_shape, which_type.element_type), which,
        bcast_dims(which_type.shape))
  if x_type.shape != out_shape:
    x = mhlo.BroadcastInDimOp(
        ir.RankedTensorType.get(out_shape, x_type.element_type), x,
        bcast_dims(x_type.shape))
  if y_type.shape != out_shape:
    y = mhlo.BroadcastInDimOp(
        ir.RankedTensorType.get(out_shape, y_type.element_type), y,
        bcast_dims(y_type.shape))
  return mhlo.SelectOp(which, x, y).result
