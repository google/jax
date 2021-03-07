# coding=utf-8
# Copyright 2018 Google LLC
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

import numpy as np

from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy.vectorize import vectorize
from jax import ad_util
from jax import api
from jax import lax
from jax import ops
from jax import dtypes
from jax.interpreters import xla
from jax.interpreters import ad
from jax.interpreters import batching
from jax._src.util import partial, prod
from jax.core import Primitive, ShapedArray, raise_to_shaped
from jax._src.lax.lax import (
    standard_primitive, standard_unop, naryop_dtype_rule, _float, _complex,
    _input_dtype, _broadcasting_select)
from jax._src.lax import lax as lax_internal
from jax.lib import lapack

from jax.lib import cusolver
from jax.lib import rocsolver

from jax.lib import xla_client
from jax.lib import xla_bridge as xb

xops = xla_client.ops


# traceables

def cholesky(x, symmetrize_input: bool = True):
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

def eig(x, compute_left_eigenvectors=True, compute_right_eigenvectors=True):
  """Eigendecomposition of a general matrix.

  Nonsymmetric eigendecomposition is at present only implemented on CPU.
  """
  return eig_p.bind(x, compute_left_eigenvectors=compute_left_eigenvectors,
                    compute_right_eigenvectors=compute_right_eigenvectors)

def eigh(x, lower: bool = True, symmetrize_input: bool = True):
  """Eigendecomposition of a Hermitian matrix.

  Computes the eigenvalues and eigenvectors of a complex Hermitian or real
  symmetric square matrix.

  Args:
    x: A batch of square complex Hermitian or real symmetric matrices with shape
      ``[..., n, n]``.
    lower: If ``symmetrize_input`` is ``False``, describes which triangle of the
      input matrix to use. If ``symmetrize_input`` is ``False``, only the
      triangle given by ``lower`` is accessed; the other triangle is ignored and
      not accessed.
    symmetrize_input: If ``True``, the matrix is symmetrized before the
      eigendecomposition by computing :math:`\\frac{1}{2}(x + x^H)`.

  Returns:
    A tuple ``(w, v)``.

    ``w`` is an array with the same dtype as ``x`` (or its real counterpart if
    complex) with shape ``[..., n]`` containing the eigenvalues of ``x``.

    ``v`` is an array with the same dtype as ``x`` such that ``v[..., :, i]`` is
    the eigenvector corresponding to ``w[..., i]``.
  """
  if symmetrize_input:
    x = symmetrize(x)
  w, v = eigh_p.bind(x, lower=lower)
  return w, v

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

def qr(x, full_matrices: bool = True):
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

def svd(x, full_matrices=True, compute_uv=True):
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

def triangular_solve(a, b, left_side: bool = False, lower: bool = False,
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

def cholesky_jvp_rule(primals, tangents):
  x, = primals
  sigma_dot, = tangents
  L = jnp.tril(cholesky_p.bind(x))

  # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
  def phi(X):
    l = jnp.tril(X)
    return l / (jnp._constant_like(X, 1) + jnp.eye(X.shape[-1], dtype=X.dtype))

  tmp = triangular_solve(L, sigma_dot, left_side=False, transpose_a=True,
                         conjugate_a=True, lower=True)
  L_dot = lax.batch_matmul(L, phi(triangular_solve(
      L, tmp, left_side=True, transpose_a=False, lower=True)),
      precision=lax.Precision.HIGHEST)
  return L, L_dot

def cholesky_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return cholesky(x), 0

cholesky_p = standard_unop(_float | _complex, 'cholesky')
ad.primitive_jvps[cholesky_p] = cholesky_jvp_rule
batching.primitive_batchers[cholesky_p] = cholesky_batching_rule

def _nan_like(c, operand):
  shape = c.get_shape(operand)
  dtype = shape.element_type()
  if jnp.issubdtype(dtype, np.complexfloating):
    nan = xb.constant(c, np.array(np.nan * (1. + 1j), dtype=dtype))
  else:
    nan = xb.constant(c, np.array(np.nan, dtype=dtype))
  return xops.Broadcast(nan, shape.dimensions())

def _cholesky_cpu_gpu_translation_rule(potrf_impl, c, operand):
  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]
  result, info = potrf_impl(c, operand, lower=True)
  ok = xops.Eq(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  return _broadcasting_select(c,
                              xops.Reshape(ok, batch_dims + (1, 1)), result,
                              _nan_like(c, result))

xla.backend_specific_translations['cpu'][cholesky_p] = partial(
  _cholesky_cpu_gpu_translation_rule, lapack.potrf)

if cusolver is not None:
  xla.backend_specific_translations['gpu'][cholesky_p] = partial(
    _cholesky_cpu_gpu_translation_rule, cusolver.potrf)

if rocsolver is not None:
  xla.backend_specific_translations['gpu'][cholesky_p] = partial(
    _cholesky_cpu_gpu_translation_rule, rocsolver.potrf)

# Asymmetric eigendecomposition

def eig_impl(operand, *, compute_left_eigenvectors, compute_right_eigenvectors):
  return (
    xla.apply_primitive(eig_p, operand,
                        compute_left_eigenvectors=compute_left_eigenvectors,
                        compute_right_eigenvectors=compute_right_eigenvectors))

def eig_translation_rule(c, operand, *, compute_left_eigenvectors,
                         compute_right_eigenvectors):
  raise NotImplementedError(
    "Nonsymmetric eigendecomposition is only implemented on the CPU backend")

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

_cpu_geev = lapack.geev

def eig_cpu_translation_rule(c, operand, *, compute_left_eigenvectors,
                             compute_right_eigenvectors):
  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]

  w, vl, vr, info = _cpu_geev(c, operand, jobvl=compute_left_eigenvectors,
                              jobvr=compute_right_eigenvectors)

  ok = xops.Eq(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  w = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1,)), w,
                           _nan_like(c, w))
  output = [w]

  if compute_left_eigenvectors:
    vl = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), vl,
                              _nan_like(c, vl))
    output.append(vl)

  if compute_right_eigenvectors:
    vr = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), vr,
                              _nan_like(c, vr))
    output.append(vr)

  return xops.Tuple(c, output)

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
xla.translations[eig_p] = eig_translation_rule
xla.backend_specific_translations['cpu'][eig_p] = eig_cpu_translation_rule
batching.primitive_batchers[eig_p] = eig_batching_rule
ad.primitive_jvps[eig_p] = eig_jvp_rule


# Symmetric/Hermitian eigendecomposition

def eigh_impl(operand, lower):
  w, v = xla.apply_primitive(eigh_p, operand, lower=lower)
  return w, v

def eigh_translation_rule(c, operand, lower):
  shape = c.get_shape(operand)
  dims = shape.dimensions()
  if dims[-1] == 0:
    return xops.Tuple(c, [operand, xops.Reshape(operand, dims[:-1])])
  if not lower:
    n = len(dims)
    operand = xops.Transpose(operand, list(range(n - 2)) + [n - 1, n - 2])
  return xops.Tuple(c, xops.Eigh(operand))

def eigh_abstract_eval(operand, lower):
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
    w = v = operand
  return w, v

def _eigh_cpu_gpu_translation_rule(syevd_impl, c, operand, lower):
  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]
  v, w, info = syevd_impl(c, operand, lower=lower)
  ok = xops.Eq(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  v = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), v,
                           _nan_like(c, v))
  w = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1,)), w,
                           _nan_like(c, w))
  return xops.Tuple(c, [w, v])

def _eigh_tpu_translation_rule(c, operand, lower):
  # Fail gracefully for complex dtype (unsupported on TPU).
  shape = c.get_shape(operand)
  dtype = shape.element_type().type
  if np.issubdtype(dtype, np.complexfloating):
    raise NotImplementedError("eigh is not implemented on TPU for complex inputs.")
  return eigh_translation_rule(c, operand, lower)

def eigh_jvp_rule(primals, tangents, lower):
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

  w_real, v = eigh_p.bind(symmetrize(a), lower=lower)

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
  return (w_real, v), (dw, dv)

def eigh_batching_rule(batched_args, batch_dims, lower):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return eigh_p.bind(x, lower=lower), (0, 0)

eigh_p = Primitive('eigh')
eigh_p.multiple_results = True
eigh_p.def_impl(eigh_impl)
eigh_p.def_abstract_eval(eigh_abstract_eval)
xla.translations[eigh_p] = eigh_translation_rule
ad.primitive_jvps[eigh_p] = eigh_jvp_rule
batching.primitive_batchers[eigh_p] = eigh_batching_rule

_cpu_syevd = lapack.syevd

xla.backend_specific_translations['cpu'][eigh_p] = partial(
  _eigh_cpu_gpu_translation_rule, _cpu_syevd)

if cusolver is not None:
  xla.backend_specific_translations['gpu'][eigh_p] = partial(
    _eigh_cpu_gpu_translation_rule, cusolver.syevd)

if rocsolver is not None:
  xla.backend_specific_translations['gpu'][eigh_p] = partial(
    _eigh_cpu_gpu_translation_rule, rocsolver.syevd)

xla.backend_specific_translations['tpu'][eigh_p] = _eigh_tpu_translation_rule


triangular_solve_dtype_rule = partial(
    naryop_dtype_rule, _input_dtype, (_float | _complex, _float | _complex),
    'triangular_solve')

def triangular_solve_shape_rule(a, b, left_side=False, **unused_kwargs):
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
    g_a, ans, a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  m, n = b.shape[-2:]
  k = 1 if unit_diagonal else 0
  g_a = jnp.tril(g_a, k=-k) if lower else jnp.triu(g_a, k=k)
  g_a = lax.neg(g_a)
  g_a = jnp.swapaxes(g_a, -1, -2) if transpose_a else g_a
  g_a = jnp.conj(g_a) if conjugate_a else g_a
  dot = partial(lax.dot if g_a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)

  def a_inverse(rhs):
    return triangular_solve(a, rhs, left_side, lower, transpose_a, conjugate_a,
                            unit_diagonal)

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
    cotangent, a, b, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  # Triangular solve is nonlinear in its first argument and linear in its second
  # argument, analogous to `div` but swapped.
  assert not ad.is_undefined_primal(a) and ad.is_undefined_primal(b)
  if type(cotangent) is ad_util.Zero:
    cotangent_b = ad_util.Zero(b.aval)
  else:
    cotangent_b = triangular_solve(a, cotangent, left_side, lower,
                                   not transpose_a, conjugate_a, unit_diagonal)
  return [None, cotangent_b]


def triangular_solve_batching_rule(batched_args, batch_dims, left_side,
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

def _triangular_solve_translation_rule(
    c, a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  if conjugate_a and not transpose_a:
    a = xops.Conj(a)
    conjugate_a = False
  if not transpose_a:
    transpose = xops.TriangularSolveOptions_Transpose.NO_TRANSPOSE
  else:
    transpose = (xops.TriangularSolveOptions_Transpose.ADJOINT if conjugate_a
                 else xops.TriangularSolveOptions_Transpose.TRANSPOSE)
  return xops.TriangularSolve(a, b, left_side, lower, unit_diagonal, transpose)

triangular_solve_p = standard_primitive(
    triangular_solve_shape_rule, triangular_solve_dtype_rule,
    'triangular_solve', translation_rule=_triangular_solve_translation_rule)
ad.defjvp2(triangular_solve_p,
           triangular_solve_jvp_rule_a,
           lambda g_b, _, a, b, **kws: triangular_solve(a, g_b, **kws))
ad.primitive_transposes[triangular_solve_p] = triangular_solve_transpose_rule
batching.primitive_batchers[triangular_solve_p] = triangular_solve_batching_rule


def _triangular_solve_cpu_translation_rule(
    c, a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  shape = c.get_shape(a)
  dtype = shape.element_type().type

  if conjugate_a and not transpose_a:
    a = xops.Conj(a)
    conjugate_a = False
  if len(shape.dimensions()) == 2 and np.dtype(dtype) in _cpu_lapack_types:
    return lapack.jax_trsm(
      c, xb.constant(c, np.array(1, dtype=dtype)),
      a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal)
  else:
    # Fall back to the HLO implementation for unsupported types or batching.
    # TODO: Consider swapping XLA for LAPACK in batched case
    if not transpose_a:
      transpose = xops.TriangularSolveOptions_Transpose.NO_TRANSPOSE
    else:
      transpose = (xops.TriangularSolveOptions_Transpose.ADJOINT if conjugate_a
                   else xops.TriangularSolveOptions_Transpose.TRANSPOSE)
    return xops.TriangularSolve(a, b, left_side, lower, unit_diagonal, transpose)

xla.backend_specific_translations['cpu'][triangular_solve_p] = \
  _triangular_solve_cpu_translation_rule

def _triangular_solve_gpu_translation_rule(trsm_impl,
    c, a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  shape = c.get_shape(a)
  dims = shape.dimensions()
  m, n = dims[-2:]
  batch = prod(dims[:-2])
  if conjugate_a and not transpose_a:
    a = xops.Conj(a)
    conjugate_a = False
  if batch > 1 and m <= 32 and n <= 32:
    return trsm_impl(
      c, a, b, left_side, lower, transpose_a,
      conjugate_a, unit_diagonal)
  else:
    # Use the XLA implementation for unbatched triangular_solve.
    if not transpose_a:
      transpose = xops.TriangularSolveOptions_Transpose.NO_TRANSPOSE
    else:
      transpose = (xops.TriangularSolveOptions_Transpose.ADJOINT if conjugate_a
                   else xops.TriangularSolveOptions_Transpose.TRANSPOSE)
    return xops.TriangularSolve(a, b, left_side, lower, unit_diagonal,
                                transpose)

if cusolver is not None:
  xla.backend_specific_translations['gpu'][triangular_solve_p] = \
      partial(_triangular_solve_gpu_translation_rule, cusolver.trsm)

if rocsolver is not None:
  xla.backend_specific_translations['gpu'][triangular_solve_p] = \
      partial(_triangular_solve_gpu_translation_rule, rocsolver.trsm)

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
    pivot = ops.index_update(pivot, ops.index[k], i)

    a = ops.index_update(a, ops.index[[k, i],], a[[i, k],])

    perm = ops.index_update(perm, ops.index[[i, k],], perm[[k, i],])

    # a[k+1:, k] /= a[k, k], adapted for loop-invariant shapes
    x = a[k, k]
    a = ops.index_update(a, ops.index[:, k],
                         jnp.where(m_idx > k, a[:, k] / x, a[:, k]))

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

    pivot = ops.index_update(pivot, ops.index[k:k+b], block_pivot + k)
    perm = ops.index_update(perm, ops.index[k:], perm[block_perm + k])
    a = ops.index_update(a, ops.index[k:, :], a[block_perm + k, :])
    a = ops.index_update(a, ops.index[k:, k:k+b], lu_block)

    if k + b < n:
      a = ops.index_update(
        a, ops.index[k:k+b, k+b:],
        triangular_solve(a[k:k+b, k:k+b], a[k:k+b, k+b:],
                         left_side=True, lower=True, unit_diagonal=True))
      a = ops.index_add(
        a, ops.index[k+b:, k+b:],
        -lax.dot(a[k+b:, k:k+b], a[k:k+b, k+b:],
                 precision=lax.Precision.HIGHEST))
  return a, pivot, perm

def _lu_python(x):
  """Default LU decomposition in Python, where no better version exists."""
  m, n = x.shape[-2:]
  batch_dims = x.shape[:-2]
  if len(batch_dims) > 0:
    batch_size = np.prod(batch_dims, dtype=np.int64)
    lu, pivot, perm = api.vmap(_lu_blocked)(lax.reshape(x, (batch_size, m, n)))
    lu = lax.reshape(lu, batch_dims + (m, n))
    pivot = lax.reshape(pivot, batch_dims + (min(m, n),))
    perm = lax.reshape(perm, batch_dims + (m,))
  else:
    lu, pivot, perm = _lu_blocked(x)
  return lu, pivot, perm

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
  zero = jnp._constant_like(lu, 0)
  l = lax.pad(jnp.tril(lu[..., :, :k], -1), zero, l_padding)
  l = l + jnp.eye(m, m, dtype=dtype)

  u_eye = lax.pad(jnp.eye(n - k, n - k, dtype=dtype), zero,
                  ((k, 0, 0), (k, 0, 0)))
  u_padding = [(0, 0, 0)] * ndims
  u_padding[-2] = (0, n - k, 0)
  u = lax.pad(jnp.triu(lu[..., :k, :]), zero, u_padding) + u_eye

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

def _lu_cpu_gpu_translation_rule(getrf_impl, c, operand):
  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]
  m = shape.dimensions()[-2]
  lu, pivot, info = getrf_impl(c, operand)
  # Subtract 1 from the pivot to get 0-based indices.
  pivot = xops.Sub(pivot, xops.ConstantLiteral(c, np.array(1, np.int32)))
  ok = xops.Ge(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  lu = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), lu,
                            _nan_like(c, lu))
  perm = xla.lower_fun(lambda x: lu_pivots_to_permutation(x, m),
                       multiple_results=False)(c, pivot)
  return xops.Tuple(c, [lu, pivot, perm])


def _lu_tpu_translation_rule(c, operand):
  if hasattr(xops, "LU"):
    lu, pivot, perm = xops.LU(operand)
    return xops.Tuple(c, [lu, pivot, perm])
  else:
    return xla.lower_fun(_lu_python, multiple_results=True)(c, operand)


lu_p = Primitive('lu')
lu_p.multiple_results = True
lu_p.def_impl(_lu_impl)
lu_p.def_abstract_eval(_lu_abstract_eval)
xla.translations[lu_p] = xla.lower_fun(_lu_python, multiple_results=True)
ad.primitive_jvps[lu_p] = _lu_jvp_rule
batching.primitive_batchers[lu_p] = _lu_batching_rule

xla.backend_specific_translations['cpu'][lu_p] = partial(
  _lu_cpu_gpu_translation_rule, lapack.getrf)

if cusolver is not None:
  xla.backend_specific_translations['gpu'][lu_p] = partial(
    _lu_cpu_gpu_translation_rule, cusolver.getrf)

if rocsolver is not None:
  xla.backend_specific_translations['gpu'][lu_p] = partial(
    _lu_cpu_gpu_translation_rule, rocsolver.getrf)

xla.backend_specific_translations['tpu'][lu_p] = _lu_tpu_translation_rule


# Define this outside lu_pivots_to_permutation to ensure fori_loop cache hits
def _lu_pivots_body_fn(i, permutation_and_swaps):
  permutation, swaps = permutation_and_swaps
  batch_dims = swaps.shape[:-1]
  j = swaps[..., i]
  iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in batch_dims))
  x = permutation[..., i]
  y = permutation[iotas + (j,)]
  permutation = ops.index_update(permutation, ops.index[..., i], y)
  return ops.index_update(permutation, ops.index[iotas + (j,)], x), swaps


@partial(api.jit, static_argnums=(1,))
def lu_pivots_to_permutation(swaps, m):
  """Converts the pivots (row swaps) returned by LU to a permutation.

  We build a permutation rather than applying `swaps` directly to the rows
  of a matrix because lax loops aren't differentiable.

  Args:
    swaps: an array of shape (..., k) of row swaps to perform
    m: the size of the output permutation. m should be >= k.
  Returns:
    An int32 array of shape (..., m).
  """
  assert len(swaps.shape) >= 1
  batch_dims = swaps.shape[:-1]
  k = swaps.shape[-1]

  permutation = lax.broadcasted_iota(jnp.int32, batch_dims + (m,),
                                     len(batch_dims))
  if m == 0:
    return permutation
  result, _ = lax.fori_loop(np.array(0, np.int32), np.array(k, np.int32),
                            _lu_pivots_body_fn, (permutation, swaps))
  return result


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
    raise ValueError("'trans' value must be 0, 1, or 2, got {}".format(trans))
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

def qr_impl(operand, full_matrices):
  q, r = xla.apply_primitive(qr_p, operand, full_matrices=full_matrices)
  return q, r

def qr_translation_rule(c, operand, full_matrices):
  return xops.Tuple(c, xops.QR(operand, full_matrices))

def qr_abstract_eval(operand, full_matrices):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to QR decomposition must have ndims >= 2")
    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    k = m if full_matrices else min(m, n)
    q = operand.update(shape=batch_dims + (m, k))
    r = operand.update(shape=batch_dims + (k, n))
  else:
    q = operand
    r = operand
  return q, r

def qr_jvp_rule(primals, tangents, full_matrices):
  # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
  x, = primals
  dx, = tangents
  q, r = qr_p.bind(x, full_matrices=False)
  *_, m, n = x.shape
  if full_matrices or m < n:
    raise NotImplementedError(
      "Unimplemented case of QR decomposition derivative")
  dx_rinv = triangular_solve(r, dx)  # Right side solve by default
  qt_dx_rinv = jnp.matmul(_H(q), dx_rinv)
  qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
  do = qt_dx_rinv_lower - _H(qt_dx_rinv_lower)  # This is skew-symmetric
  # The following correction is necessary for complex inputs
  do = do + jnp.eye(n, dtype=do.dtype) * (qt_dx_rinv - jnp.real(qt_dx_rinv))
  dq = jnp.matmul(q, do - qt_dx_rinv) + dx_rinv
  dr = jnp.matmul(qt_dx_rinv - do, r)
  return (q, r), (dq, dr)

def qr_batching_rule(batched_args, batch_dims, full_matrices):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return qr_p.bind(x, full_matrices=full_matrices), (0, 0)

def _qr_cpu_gpu_translation_rule(geqrf_impl, orgqr_impl, c, operand,
                                 full_matrices):
  shape = c.get_shape(operand)
  dims = shape.dimensions()
  m, n = dims[-2:]
  batch_dims = dims[:-2]
  r, tau, info_geqrf = geqrf_impl(c, operand)
  if m < n:
    q = xops.Slice(r, [0] * len(dims), list(batch_dims) + [m, m],
                   [1] * len(dims))
    q, info_orgqr = orgqr_impl(c, q, tau)
  elif not full_matrices:
    q, info_orgqr = orgqr_impl(c, r, tau)
    r = xops.Slice(r, [0] * len(dims), list(batch_dims) + [n, n],
                   [1] * len(dims))
  else:
    padding_config = [(0, 0, 0)] * len(dims)
    padding_config[-1] = (0, m - n, 0)
    q = xops.Pad(r, xops.Constant(c, np.array(0, dtype=shape.element_type())),
                 xla_client.make_padding_config(padding_config))
    q, info_orgqr = orgqr_impl(c, q, tau)
  if info_geqrf is not None:
    ok = xops.And(
      xops.Eq(info_geqrf, xops.ConstantLiteral(c, np.array(0, np.int32))),
      xops.Eq(info_orgqr, xops.ConstantLiteral(c, np.array(0, np.int32))))
    q = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), q,
                             _nan_like(c, q))
    r = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), r,
                             _nan_like(c, r))
  else:
    pass # rocsolver does not return info

  r = xla.lower_fun(jnp.triu, multiple_results=False)(c, r)
  return xops.Tuple(c, [q, r])

qr_p = Primitive('qr')
qr_p.multiple_results = True
qr_p.def_impl(qr_impl)
qr_p.def_abstract_eval(qr_abstract_eval)
xla.translations[qr_p] = qr_translation_rule
ad.primitive_jvps[qr_p] = qr_jvp_rule
batching.primitive_batchers[qr_p] = qr_batching_rule

xla.backend_specific_translations['cpu'][qr_p] = partial(
  _qr_cpu_gpu_translation_rule, lapack.geqrf, lapack.orgqr)

if cusolver is not None:
  xla.backend_specific_translations['gpu'][qr_p] = partial(
    _qr_cpu_gpu_translation_rule, cusolver.geqrf, cusolver.orgqr)

if rocsolver is not None:
  xla.backend_specific_translations['gpu'][qr_p] = partial(
    _qr_cpu_gpu_translation_rule, rocsolver.geqrf, rocsolver.orgqr)


# Singular value decomposition

def svd_impl(operand, full_matrices, compute_uv):
  return xla.apply_primitive(svd_p, operand, full_matrices=full_matrices,
                             compute_uv=compute_uv)

def svd_translation_rule(c, operand, full_matrices, compute_uv):
  shape = c.get_shape(operand).dimensions()
  m, n = shape[-2:]
  if m == 0 or n == 0:
    return xla.lower_fun(_empty_svd, multiple_results=True)(
      c, operand, full_matrices=full_matrices, compute_uv=compute_uv)

  u, s, v = xops.SVD(operand)
  permutation = list(range(len(shape)))
  permutation[-1], permutation[-2] = permutation[-2], permutation[-1]
  vt = xops.Transpose(v, permutation)
  if not full_matrices and m != n:
    u = xops.SliceInDim(u, 0, min(m, n), stride=1, dimno=len(shape) - 1)
    vt = xops.SliceInDim(vt, 0, min(m, n), stride=1, dimno=len(shape) - 2)

  if not compute_uv:
    return xops.Tuple(c, [s])
  else:
    return xops.Tuple(c, [s, u, vt])


def svd_abstract_eval(operand, full_matrices, compute_uv):
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

def svd_jvp_rule(primals, tangents, full_matrices, compute_uv):
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

  s_diffs = jnp.square(s_dim) - jnp.square(_T(s_dim))
  s_diffs_zeros = jnp.eye(s.shape[-1], dtype=A.dtype)  # jnp.ones((), dtype=A.dtype) * (s_diffs == 0.)  # is 1. where s_diffs is 0. and is 0. everywhere else
  F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
  dSS = s_dim * dS  # dS.dot(jnp.diag(s))
  SdS = _T(s_dim) * dS  # jnp.diag(s).dot(dS)

  s_zeros = jnp.ones((), dtype=A.dtype) * (s == 0.)
  s_inv = 1 / (s + s_zeros) - s_zeros
  s_inv_mat = jnp.vectorize(jnp.diag, signature='(k)->(k,k)')(s_inv)
  dUdV_diag = .5 * (dS - _H(dS)) * s_inv_mat
  dU = jnp.matmul(U, F * (dSS + _H(dSS)) + dUdV_diag)
  dV = jnp.matmul(V, F * (SdS + _H(SdS)))

  m, n = A.shape[-2:]
  if m > n:
    dU = dU + jnp.matmul(jnp.eye(m, dtype=A.dtype) - jnp.matmul(U, Ut), jnp.matmul(dA, V)) / s_dim
  if n > m:
    dV = dV + jnp.matmul(jnp.eye(n, dtype=A.dtype) - jnp.matmul(V, Vt), jnp.matmul(_H(dA), U)) / s_dim

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

def _svd_cpu_gpu_translation_rule(gesvd_impl, c, operand, full_matrices, compute_uv):
  shape = c.get_shape(operand).dimensions()
  m, n = shape[-2:]
  batch_dims = shape[:-2]

  if m == 0 or n == 0:
    return xla.lower_fun(_empty_svd, multiple_results=True)(
      c, operand, full_matrices=full_matrices, compute_uv=compute_uv)

  s, u, vt, info = gesvd_impl(c, operand,
                              full_matrices=full_matrices,
                              compute_uv=compute_uv)
  ok = xops.Eq(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  s = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1,)), s,
                           _nan_like(c, s))

  result = [s]

  if compute_uv:
    u = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), u,
                             _nan_like(c, u))
    vt = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), vt,
                              _nan_like(c, vt))
    result += [u, vt]

  return xops.Tuple(c, result)

def svd_batching_rule(batched_args, batch_dims, full_matrices, compute_uv):
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
svd_p.def_impl(svd_impl)
svd_p.def_abstract_eval(svd_abstract_eval)
ad.primitive_jvps[svd_p] = svd_jvp_rule
batching.primitive_batchers[svd_p] = svd_batching_rule
xla.translations[svd_p] = svd_translation_rule

xla.backend_specific_translations['cpu'][svd_p] = partial(
  _svd_cpu_gpu_translation_rule, lapack.gesdd)

if cusolver is not None:
  xla.backend_specific_translations['gpu'][svd_p] = partial(
    _svd_cpu_gpu_translation_rule, cusolver.gesvd)

if rocsolver is not None:
  xla.backend_specific_translations['gpu'][svd_p] = partial(
    _svd_cpu_gpu_translation_rule, rocsolver.gesvd)
