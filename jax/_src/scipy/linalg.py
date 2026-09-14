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

from __future__ import annotations

from functools import partial
import math
import operator
import textwrap
from typing import overload, Any, Literal
import warnings

import numpy as np

from jax._src import config
from jax._src import dtypes
from jax._src import lax
from jax._src import numpy as jnp
from jax._src import random
from jax._src.api import jit, vmap, jvp
from jax._src.lax import linalg as lax_linalg
from jax._src.numpy import linalg as jnp_linalg
from jax._src.numpy import vectorize as jnp_vectorize
from jax._src.numpy.util import (
    check_arraylike, ensure_arraylike, promote_dtypes, promote_dtypes_inexact,
    promote_dtypes_complex, promote_args_inexact)
from jax._src.scipy.special import comb
from jax._src.tpu.linalg import qdwh
from jax._src.typing import Array, ArrayLike, DTypeLike


_no_chkfinite_doc = textwrap.dedent("""
Does not support the Scipy argument ``check_finite=True``,
because compiled JAX code cannot perform checks of array values at runtime.
""")
_no_overwrite_and_chkfinite_doc = _no_chkfinite_doc + "\nDoes not support the Scipy argument ``overwrite_*=True``."

@jit(static_argnames=('lower',))
def _cholesky(a: ArrayLike, lower: bool) -> Array:
  a, = promote_dtypes_inexact(jnp.asarray(a))
  l = lax_linalg.cholesky(a if lower else jnp.conj(a.mT), symmetrize_input=False)
  return l if lower else jnp.conj(l.mT)


def cholesky(a: ArrayLike, lower: bool = False, overwrite_a: bool = False,
             check_finite: bool = True) -> Array:
  """Compute the Cholesky decomposition of a matrix.

  JAX implementation of :func:`scipy.linalg.cholesky`.

  The Cholesky decomposition of a matrix `A` is:

  .. math::

     A = U^HU = LL^H

  where `U` is an upper-triangular matrix and `L` is a lower-triangular matrix.

  Args:
    a: input array, representing a (batched) positive-definite hermitian matrix.
      Must have shape ``(..., N, N)``.
    lower: if True, compute the lower Cholesky decomposition `L`. if False
      (default), compute the upper Cholesky decomposition `U`.
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    array of shape ``(..., N, N)`` representing the cholesky decomposition
    of the input.

  See Also:
   - :func:`jax.numpy.linalg.cholesky`: NumPy-stype Cholesky API
   - :func:`jax.lax.linalg.cholesky`: XLA-style Cholesky API
   - :func:`jax.scipy.linalg.cho_factor`
   - :func:`jax.scipy.linalg.cho_solve`

  Examples:
    A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Upper Cholesky factorization:

    >>> jax.scipy.linalg.cholesky(x)
    Array([[1.4142135 , 0.70710677],
           [0.        , 1.2247449 ]], dtype=float32)

    Lower Cholesky factorization:

    >>> jax.scipy.linalg.cholesky(x, lower=True)
    Array([[1.4142135 , 0.        ],
           [0.70710677, 1.2247449 ]], dtype=float32)

    Reconstructing ``x`` from its factorization:

    >>> L = jax.scipy.linalg.cholesky(x, lower=True)
    >>> jnp.allclose(x, L @ L.T)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # Unused
  return _cholesky(a, lower)


def cho_factor(a: ArrayLike, lower: bool = False, overwrite_a: bool = False,
               check_finite: bool = True) -> tuple[Array, bool]:
  """Factorization for Cholesky-based linear solves

  JAX implementation of :func:`scipy.linalg.cho_factor`. This function returns
  a result suitable for use with :func:`jax.scipy.linalg.cho_solve`. For direct
  Cholesky decompositions, prefer :func:`jax.scipy.linalg.cholesky`.

  Args:
    a: input array, representing a (batched) positive-definite hermitian matrix.
      Must have shape ``(..., N, N)``.
    lower: if True, compute the lower triangular Cholesky decomposition (default: False).
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    ``(c, lower)``: ``c`` is an array of shape ``(..., N, N)`` representing the lower or
    upper cholesky decomposition of the input; ``lower`` is a boolean specifying whether
    this is the lower or upper decomposition.

  See Also:
    - :func:`jax.scipy.linalg.cholesky`
    - :func:`jax.scipy.linalg.cho_solve`

  Examples:
    A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Compute the cholesky factorization via :func:`~jax.scipy.linalg.cho_factor`,
    and use it to solve a linear equation via :func:`~jax.scipy.linalg.cho_solve`.

    >>> b = jnp.array([3., 4.])
    >>> cfac = jax.scipy.linalg.cho_factor(x)
    >>> y = jax.scipy.linalg.cho_solve(cfac, b)
    >>> y
    Array([0.6666666, 1.6666666], dtype=float32)

    Check that the result is consistent:

    >>> jnp.allclose(x @ y, b)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # Unused
  return (cholesky(a, lower=lower), lower)

@jit(static_argnames=('lower',))
def _cho_solve(c: ArrayLike, b: ArrayLike, lower: bool) -> Array:
  c, b = promote_dtypes_inexact(jnp.asarray(c), jnp.asarray(b))
  if b.ndim == 1:
    signature = "(n,n),(n)->(n)"
  elif c.ndim == b.ndim + 1 and c.shape[-1] == b.shape[-1]:
    # Deprecation warning added 2026-03-23
    warnings.warn(
        "jax.scipy.linalg.cho_solve: batched 1D solves with b.ndim > 1 are "
        "deprecated, and in the future will be treated as a batched 2D solve. "
        "Use cho_solve(c_and_lower, b[..., None]).squeeze(-1) to avoid this warning.",
        category=FutureWarning)
    signature = "(n,n),(n)->(n)"
  else:
    signature = "(n,n),(n,k)->(n,k)"
  b = jnp_vectorize.vectorize(
      partial(lax_linalg.triangular_solve, left_side=True, lower=lower,
              transpose_a=not lower, conjugate_a=not lower),
      signature=signature)(c, b)
  return jnp_vectorize.vectorize(
      partial(lax_linalg.triangular_solve, left_side=True, lower=lower,
              transpose_a=lower, conjugate_a=lower),
      signature=signature)(c, b)


def cho_solve(c_and_lower: tuple[ArrayLike, bool], b: ArrayLike,
              overwrite_b: bool = False, check_finite: bool = True) -> Array:
  """Solve a linear system using a Cholesky factorization

  JAX implementation of :func:`scipy.linalg.cho_solve`. Uses the output
  of :func:`jax.scipy.linalg.cho_factor`.

  Args:
    c_and_lower: ``(c, lower)``, where ``c`` is an array of shape ``(..., N, N)``
      representing the lower or upper cholesky decomposition of the matrix, and
      ``lower`` is a boolean specifying whether this is the lower or upper decomposition.
    b: right-hand-side of linear system. Array of shape ``(N,)`` (for a
      1-dimensional right-hand-side) or ``(..., N, M)`` (for a batched
      2-dimensional right-hand-side).
    overwrite_b: unused by JAX
    check_finite: unused by JAX

  Returns:
    Array representing the solution of the linear system. The result has
    shape ``(..., N)`` if ``b`` is of shape ``(N,)``, and has shape
    ``(..., N, M)`` otherwise.

  See Also:
    - :func:`jax.scipy.linalg.cholesky`
    - :func:`jax.scipy.linalg.cho_factor`

  Examples:
    A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Compute the cholesky factorization via :func:`~jax.scipy.linalg.cho_factor`,
    and use it to solve a linear equation via :func:`~jax.scipy.linalg.cho_solve`.

    >>> b = jnp.array([3., 4.])
    >>> cfac = jax.scipy.linalg.cho_factor(x)
    >>> y = jax.scipy.linalg.cho_solve(cfac, b)
    >>> y
    Array([0.6666666, 1.6666666], dtype=float32)

    Check that the result is consistent:

    >>> jnp.allclose(x @ y, b)
    Array(True, dtype=bool)
  """
  del overwrite_b, check_finite  # Unused
  c, lower = c_and_lower
  return _cho_solve(c, b, lower)

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: Literal[True]) -> tuple[Array, Array, Array]: ...

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: Literal[False]) -> Array: ...

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: bool) -> Array | tuple[Array, Array, Array]: ...

@jit(static_argnames=('full_matrices', 'compute_uv'))
def _svd(a: ArrayLike, *, full_matrices: bool, compute_uv: bool) -> Array | tuple[Array, Array, Array]:
  a, = promote_dtypes_inexact(jnp.asarray(a))
  return lax_linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

@overload
def svd(a: ArrayLike, full_matrices: bool = True, compute_uv: Literal[True] = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> tuple[Array, Array, Array]: ...

@overload
def svd(a: ArrayLike, full_matrices: bool, compute_uv: Literal[False],
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array: ...

@overload
def svd(a: ArrayLike, full_matrices: bool = True, *, compute_uv: Literal[False],
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array: ...

@overload
def svd(a: ArrayLike, full_matrices: bool = True, compute_uv: bool = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array | tuple[Array, Array, Array]: ...


def svd(a: ArrayLike, full_matrices: bool = True, compute_uv: bool = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array | tuple[Array, Array, Array]:
  r"""Compute the singular value decomposition.

  JAX implementation of :func:`scipy.linalg.svd`.

  The SVD of a matrix `A` is given by

  .. math::

     A = U\Sigma V^H

  - :math:`U` contains the left singular vectors and satisfies :math:`U^HU=I`
  - :math:`V` contains the right singular vectors and satisfies :math:`V^HV=I`
  - :math:`\Sigma` is a diagonal matrix of singular values.

  Args:
    a: input array, of shape ``(..., N, M)``
    full_matrices: if True (default) compute the full matrices; i.e. ``u`` and ``vh`` have
      shape ``(..., N, N)`` and ``(..., M, M)``. If False, then the shapes are
      ``(..., N, K)`` and ``(..., K, M)`` with ``K = min(N, M)``.
    compute_uv: if True (default), return the full SVD ``(u, s, vh)``. If False then return
      only the singular values ``s``.
    overwrite_a: unused by JAX
    check_finite: unused by JAX
    lapack_driver: unused by JAX. If you want to select a non-default SVD driver, please
      check :func:`jax.lax.linalg.svd` which provides such functionality.

  Returns:
    A tuple of arrays ``(u, s, vh)`` if ``compute_uv`` is True, otherwise the array ``s``.

    - ``u``: left singular vectors of shape ``(..., N, N)`` if ``full_matrices`` is True
      or ``(..., N, K)`` otherwise.
    - ``s``: singular values of shape ``(..., K)``
    - ``vh``: conjugate-transposed right singular vectors of shape ``(..., M, M)``
      if ``full_matrices`` is True or ``(..., K, M)`` otherwise.

    where ``K = min(N, M)``.

  See also:
    - :func:`jax.numpy.linalg.svd`: NumPy-style SVD API
    - :func:`jax.lax.linalg.svd`: XLA-style SVD API

  Examples:
    Consider the SVD of a small real-valued array:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [6., 5., 4.]])
    >>> u, s, vt = jax.scipy.linalg.svd(x, full_matrices=False)
    >>> s  # doctest: +SKIP
    Array([9.361919 , 1.8315067], dtype=float32)

    The singular vectors are in the columns of ``u`` and ``v = vt.T``. These vectors are
    orthonormal, which can be demonstrated by comparing the matrix product with the
    identity matrix:

    >>> jnp.allclose(u.T @ u, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)
    >>> v = vt.T
    >>> jnp.allclose(v.T @ v, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)

    Given the SVD, ``x`` can be reconstructed via matrix multiplication:

    >>> x_reconstructed = u @ jnp.diag(s) @ vt
    >>> jnp.allclose(x_reconstructed, x)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite, lapack_driver  # unused
  return _svd(a, full_matrices=full_matrices, compute_uv=compute_uv)


def det(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> Array:
  """Compute the determinant of a matrix

  JAX implementation of :func:`scipy.linalg.det`.

  Args:
    a: input array, of shape ``(..., N, N)``
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns
    Determinant of shape ``a.shape[:-2]``

  See Also:
    :func:`jax.numpy.linalg.det`: NumPy-style determinant API

  Examples:
    Determinant of a small 2D array:

    >>> x = jnp.array([[1., 2.],
    ...                [2., 3.]])
    >>> jax.scipy.linalg.det(x)
    Array(-1., dtype=float32)

    Batch-wise determinant of multiple 2D arrays:

    >>> x = jnp.array([[[1., 2.],
    ...                 [2., 3.]],
    ...                [[8., 4.],
    ...                 [2., 5.]]])
    >>> jax.scipy.linalg.det(x)
    Array([-1., 32.], dtype=float32)
  """
  del overwrite_a, check_finite  # unused
  return jnp_linalg.det(a)


@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: Literal[True],
          eigvals: None, type: int) -> Array: ...

@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: Literal[False],
          eigvals: None, type: int) -> tuple[Array, Array]: ...

@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: bool,
          eigvals: None, type: int) -> Array | tuple[Array, Array]: ...

@jit(static_argnames=('lower', 'eigvals_only', 'eigvals', 'type'))
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: bool,
          eigvals: None, type: int) -> Array | tuple[Array, Array]:
  if b is not None:
    raise NotImplementedError("Only the b=None case of eigh is implemented")
  if type != 1:
    raise NotImplementedError("Only the type=1 case of eigh is implemented.")
  if eigvals is not None:
    raise NotImplementedError(
        "Only the eigvals=None case of eigh is implemented.")

  a, = promote_dtypes_inexact(jnp.asarray(a))
  v, w = lax_linalg.eigh(a, lower=lower)

  if eigvals_only:
    return w
  else:
    return w, v

@overload
def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True,
         eigvals_only: Literal[False] = False, overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> tuple[Array, Array]: ...

@overload
def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True, *,
         eigvals_only: Literal[True], overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array: ...

@overload
def eigh(a: ArrayLike, b: ArrayLike | None, lower: bool,
         eigvals_only: Literal[True], overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array: ...

@overload
def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True,
         eigvals_only: bool = False, overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array | tuple[Array, Array]: ...

def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True,
         eigvals_only: bool = False, overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array | tuple[Array, Array]:
  """Compute eigenvalues and eigenvectors for a Hermitian matrix

  JAX implementation of :func:`scipy.linalg.eigh`.

  Only the standard eigenvalue problem is supported: ``a @ v = lambda * v``.
    The parameter `b` must be None; the generalized problem (``a @ v = lambda * b @ v``)
    is not implemented.

  Args:
    a: Hermitian input array of shape ``(..., N, N)``
    b: Must be None. The generalized eigenvalue problem is not supported.
    lower: if True (default) access only the lower portion of the input matrix.
      Otherwise access only the upper portion.
    eigvals_only: If True, compute only the eigenvalues. If False (default) compute
      both eigenvalues and eigenvectors.
    type: Not used. Only type=1 is supported.

    eigvals: Not used. Only eigvals=None is supported.
    overwrite_a: unused by JAX.
    overwrite_b: unused by JAX.
    turbo: unused by JAX.
    check_finite: unused by JAX.

  Returns:
    A tuple of arrays ``(eigvals, eigvecs)`` if ``eigvals_only`` is False, otherwise
    an array ``eigvals``.

    - ``eigvals``: array of shape ``(..., N)`` containing the eigenvalues.
    - ``eigvecs``: array of shape ``(..., N, N)`` containing the eigenvectors.

  Raise:
    NotImplementedError: If `b` is not None.

  See also:
    - :func:`jax.numpy.linalg.eigh`: NumPy-style eigh API.
    - :func:`jax.lax.linalg.eigh`: XLA-style eigh API.
    - :func:`jax.numpy.linalg.eig`: non-hermitian eigenvalue problem.
    - :func:`jax.scipy.linalg.eigh_tridiagonal`: tri-diagonal eigenvalue problem.

  Examples:
    Compute the standard eigenvalue decomposition of a simple 2x2 matrix:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])
    >>> eigvals, eigvecs = jax.scipy.linalg.eigh(a)
    >>> eigvals
    Array([1., 3.], dtype=float32)
    >>> eigvecs
    Array([[-0.70710677,  0.70710677],
           [ 0.70710677,  0.70710677]], dtype=float32)

    Eigenvectors are orthonormal:

    >>> jnp.allclose(eigvecs.T @ eigvecs, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)

    Solution satisfies the eigenvalue problem:

    >>> jnp.allclose(a @ eigvecs, eigvecs @ jnp.diag(eigvals))
    Array(True, dtype=bool)
  """
  del overwrite_a, overwrite_b, turbo, check_finite  # unused
  return _eigh(a, b, lower, eigvals_only, eigvals, type)

@jit(static_argnames=('output',))
def _schur(a: Array, output: str) -> tuple[Array, Array]:
  if output == "complex":
    a = a.astype(dtypes.to_complex_dtype(a.dtype))
  return lax_linalg.schur(a)

def schur(a: ArrayLike, output: str = 'real') -> tuple[Array, Array]:
  """Compute the Schur decomposition

  Only implemented on CPU.

  JAX implementation of :func:`scipy.linalg.schur`.

  The Schur form `T` of a matrix `A` satisfies:

  .. math::

     A = Z T Z^H

  where `Z` is unitary, and `T` is upper-triangular for the complex-valued Schur
  decomposition (i.e. ``output="complex"``) and is quasi-upper-triangular for the
  real-valued Schur decomposition (i.e. ``output="real"``). In the quasi-triangular
  case, the diagonal may include 2x2 blocks associated with complex-valued
  eigenvalue pairs of `A`.

  Args:
    a: input array of shape ``(..., N, N)``
    output: Specify whether to compute the ``"real"`` (default) or ``"complex"``
      Schur decomposition.

  Returns:
    A tuple of arrays ``(T, Z)``

    - ``T`` is a shape ``(..., N, N)`` array containing the upper-triangular
      Schur form of the input.
    - ``Z`` is a shape ``(..., N, N)`` array containing the unitary Schur
      transformation matrix.

  See also:
    - :func:`jax.scipy.linalg.rsf2csf`: convert real Schur form to complex Schur form.
    - :func:`jax.lax.linalg.schur`: XLA-style API for Schur decomposition.

  Examples:
    A Schur decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [1., 4., 2.],
    ...                [3., 2., 1.]])
    >>> T, Z = jax.scipy.linalg.schur(a)

    The Schur form ``T`` is quasi-upper-triangular in general, but is truly
    upper-triangular in this case because the input matrix is symmetric:

    >>> T  # doctest: +SKIP
    Array([[-2.0000005 ,  0.5066295 , -0.43360388],
           [ 0.        ,  1.5505103 ,  0.74519426],
           [ 0.        ,  0.        ,  6.449491  ]], dtype=float32)

    The transformation matrix ``Z`` is unitary:

    >>> jnp.allclose(Z.T @ Z, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

    The input can be reconstructed from the outputs:

    >>> jnp.allclose(Z @ T @ Z.T, a)
    Array(True, dtype=bool)
  """
  if output not in ('real', 'complex'):
    raise ValueError(
      f"Expected 'output' to be either 'real' or 'complex', got {output=}.")
  return _schur(ensure_arraylike("scipy.schur", a), output)


def inv(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> Array:
  """Return the inverse of a square matrix

  JAX implementation of :func:`scipy.linalg.inv`.

  Args:
    a: array of shape ``(..., N, N)`` specifying square array(s) to be inverted.
    overwrite_a: unused in JAX
    check_finite: unused in JAX

  Returns:
    Array of shape ``(..., N, N)`` containing the inverse of the input.

  Notes:
    In most cases, explicitly computing the inverse of a matrix is ill-advised. For
    example, to compute ``x = inv(A) @ b``, it is more performant and numerically
    precise to use a direct solve, such as :func:`jax.scipy.linalg.solve`.

  See Also:
    - :func:`jax.numpy.linalg.inv`: NumPy-style API for matrix inverse
    - :func:`jax.scipy.linalg.solve`: direct linear solver

  Examples:
    Compute the inverse of a 3x3 matrix

    >>> a = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> a_inv = jax.scipy.linalg.inv(a)
    >>> a_inv  # doctest: +SKIP
    Array([[ 0.        , -0.25      ,  0.5       ],
           [-0.25      ,  0.5       , -0.25000003],
           [ 0.5       , -0.25      ,  0.        ]], dtype=float32)

    Check that multiplying with the inverse gives the identity:

    >>> jnp.allclose(a @ a_inv, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

    Multiply the inverse by a vector ``b``, to find a solution to ``a @ x = b``

    >>> b = jnp.array([1., 4., 2.])
    >>> a_inv @ b
    Array([ 0.  ,  1.25, -0.5 ], dtype=float32)

    Note, however, that explicitly computing the inverse in such a case can lead
    to poor performance and loss of precision as the size of the problem grows.
    Instead, you should use a direct solver like :func:`jax.scipy.linalg.solve`:

    >>> jax.scipy.linalg.solve(a, b)
     Array([ 0.  ,  1.25, -0.5 ], dtype=float32)
  """
  del overwrite_a, check_finite  # unused
  return jnp_linalg.inv(a)


@jit(static_argnames=('overwrite_a', 'check_finite'))
def lu_factor(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> tuple[Array, Array]:
  """Factorization for LU-based linear solves

  JAX implementation of :func:`scipy.linalg.lu_factor`.

  This function returns a result suitable for use with :func:`jax.scipy.linalg.lu_solve`.
  For direct LU decompositions, prefer :func:`jax.scipy.linalg.lu`.

  Args:
    a: input array of shape ``(..., M, N)``.
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    A tuple ``(lu, piv)``

    - ``lu`` is an array of shape ``(..., M, N)``, containing ``L`` in its
      lower triangle and ``U`` in its upper.
    - ``piv`` is an array of shape ``(..., K)`` with ``K = min(M, N)``,
      which encodes the pivots.

  See Also:
    - :func:`jax.scipy.linalg.lu`
    - :func:`jax.scipy.linalg.lu_solve`

  Examples:
    Solving a small linear system via LU factorization:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Compute the lu factorization via :func:`~jax.scipy.linalg.lu_factor`,
    and use it to solve a linear equation via :func:`~jax.scipy.linalg.lu_solve`.

    >>> b = jnp.array([3., 4.])
    >>> lufac = jax.scipy.linalg.lu_factor(a)
    >>> y = jax.scipy.linalg.lu_solve(lufac, b)
    >>> y
    Array([0.6666666, 1.6666667], dtype=float32)

    Check that the result is consistent:

    >>> jnp.allclose(a @ y, b)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # unused
  a, = promote_dtypes_inexact(jnp.asarray(a))
  lu, pivots, _ = lax_linalg.lu(a)
  return lu, pivots


@jit(static_argnames=('trans', 'overwrite_b', 'check_finite'))
def lu_solve(lu_and_piv: tuple[Array, ArrayLike], b: ArrayLike, trans: int = 0,
             overwrite_b: bool = False, check_finite: bool = True) -> Array:
  """Solve a linear system using an LU factorization

  JAX implementation of :func:`scipy.linalg.lu_solve`. Uses the output
  of :func:`jax.scipy.linalg.lu_factor`.

  Args:
    lu_and_piv: ``(lu, piv)``, output of :func:`~jax.scipy.linalg.lu_factor`.
      ``lu`` is an array of shape ``(..., M, N)``, containing ``L`` in its lower
      triangle and ``U`` in its upper. ``piv`` is an array of shape ``(..., K)``,
      with ``K = min(M, N)``, which encodes the pivots.
    b: right-hand-side of linear system. Array of shape ``(M,)`` (for a
      1-dimensional right-hand-side) or ``(..., M, K)`` (for a batched
      2-dimensional right-hand-side).
    trans: type of system to solve. Options are:

      - ``0``: :math:`A x = b`
      - ``1``: :math:`A^Tx = b`
      - ``2``: :math:`A^Hx = b`

    overwrite_b: unused by JAX
    check_finite: unused by JAX

  Returns:
    Array representing the solution of the linear system. The result has
    shape ``(..., N)`` if ``b`` is of shape ``(M,)``, and has shape
    ``(..., N, K)`` otherwise.

  See Also:
    - :func:`jax.scipy.linalg.lu`
    - :func:`jax.scipy.linalg.lu_factor`

  Examples:
    Solving a small linear system via LU factorization:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])

    Compute the lu factorization via :func:`~jax.scipy.linalg.lu_factor`,
    and use it to solve a linear equation via :func:`~jax.scipy.linalg.lu_solve`.

    >>> b = jnp.array([3., 4.])
    >>> lufac = jax.scipy.linalg.lu_factor(a)
    >>> y = jax.scipy.linalg.lu_solve(lufac, b)
    >>> y
    Array([0.6666666, 1.6666667], dtype=float32)

    Check that the result is consistent:

    >>> jnp.allclose(a @ y, b)
    Array(True, dtype=bool)
  """
  del overwrite_b, check_finite  # unused
  lu, pivots = lu_and_piv
  lu, b = promote_dtypes_inexact(jnp.asarray(lu), jnp.asarray(b))
  m, _ = lu.shape[-2:]
  perm = lax_linalg.lu_pivots_to_permutation(pivots, m)
  if b.ndim == 1:
    signature = "(n,n),(n),(n)->(n)"
  elif lu.ndim == b.ndim + 1 and lu.shape[-1] == b.shape[-1]:
    # Deprecation warning added 2026-03-23
    warnings.warn(
        "jax.scipy.linalg.lu_solve: batched 1D solves with b.ndim > 1 are "
        "deprecated, and in the future will be treated as a batched 2D solve. "
        "Use lu_solve(lu_and_piv, b[..., None]).squeeze(-1) to avoid this warning.",
        category=FutureWarning)
    signature = "(n,n),(n),(n)->(n)"
  else:
    signature = "(n,n),(n),(n,k)->(n,k)"
  return jnp_vectorize.vectorize(
      partial(lax_linalg.lu_solve, trans=trans),
      signature=signature)(lu, perm, b)

@overload
def _lu(a: ArrayLike, permute_l: Literal[True]) -> tuple[Array, Array]: ...

@overload
def _lu(a: ArrayLike, permute_l: Literal[False]) -> tuple[Array, Array, Array]: ...

@overload
def _lu(a: ArrayLike, permute_l: bool) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...

@jit(static_argnums=(1,))
def _lu(a: ArrayLike, permute_l: bool) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  a, = promote_dtypes_inexact(jnp.asarray(a))
  lu, _, permutation = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  m, n = np.shape(a)
  p = jnp.real(jnp.array(permutation[None, :] == jnp.arange(m, dtype=permutation.dtype)[:, None], dtype=dtype))
  k = min(m, n)
  l = jnp.tril(lu, -1)[:, :k] + jnp.eye(m, k, dtype=dtype)
  u = jnp.triu(lu)[:k, :]
  if permute_l:
    return jnp.matmul(p, l, precision=lax.Precision.HIGHEST), u
  else:
    return p, l, u

@overload
def lu(a: ArrayLike, permute_l: Literal[False] = False, overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array, Array]: ...

@overload
def lu(a: ArrayLike, permute_l: Literal[True], overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array]: ...

@overload
def lu(a: ArrayLike, permute_l: bool = False, overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...


@jit(static_argnames=('permute_l', 'overwrite_a', 'check_finite'))
def lu(a: ArrayLike, permute_l: bool = False, overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  """Compute the LU decomposition

  JAX implementation of :func:`scipy.linalg.lu`.

  The LU decomposition of a matrix `A` is:

  .. math::

     A = P L U

  where `P` is a permutation matrix, `L` is lower-triangular and `U` is upper-triangular.

  Args:
    a: array of shape ``(..., M, N)`` to decompose.
    permute_l: if True, then permute ``L`` and return ``(P @ L, U)`` (default: False)
    overwrite_a: not used by JAX
    check_finite: not used by JAX

  Returns:
    A tuple of arrays ``(P @ L, U)`` if ``permute_l`` is True, else ``(P, L, U)``:

    - ``P`` is a permutation matrix of shape ``(..., M, M)``
    - ``L`` is a lower-triangular matrix of shape ``(... M, K)``
    - ``U`` is an upper-triangular matrix of shape ``(..., K, N)``

    with ``K = min(M, N)``

  See also:
    - :func:`jax.numpy.linalg.lu`: NumPy-style API for LU decomposition.
    - :func:`jax.lax.linalg.lu`: XLA-style API for LU decomposition.
    - :func:`jax.scipy.linalg.lu_solve`: LU-based linear solver.

  Examples:
    An LU decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [5., 4., 2.],
    ...                [3., 2., 1.]])
    >>> P, L, U = jax.scipy.linalg.lu(a)

    ``P`` is a permutation matrix: i.e. each row and column has a single ``1``:

    >>> P
    Array([[0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.]], dtype=float32)

    ``L`` and ``U`` are lower-triangular and upper-triangular matrices:

    >>> with jnp.printoptions(precision=3):
    ...   print(L)
    ...   print(U)
    [[ 1.     0.     0.   ]
     [ 0.2    1.     0.   ]
     [ 0.6   -0.333  1.   ]]
    [[5.    4.    2.   ]
     [0.    1.2   2.6  ]
     [0.    0.    0.667]]

    The original matrix can be reconstructed by multiplying the three together:

    >>> a_reconstructed = P @ L @ U
    >>> jnp.allclose(a, a_reconstructed)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # unused
  return _lu(a, permute_l)


@overload
def _qr(a: ArrayLike, mode: Literal["r"], pivoting: Literal[False]
       ) -> tuple[Array]: ...

@overload
def _qr(a: ArrayLike, mode: Literal["r"], pivoting: Literal[True]
       ) -> tuple[Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: Literal["full", "economic"], pivoting: Literal[False]
       ) -> tuple[Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: Literal["full", "economic"], pivoting: Literal[True]
       ) -> tuple[Array, Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: str, pivoting: Literal[False]
       ) -> tuple[Array] | tuple[Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: str, pivoting: Literal[True]
       ) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: str, pivoting: bool
       ) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]: ...


@jit(static_argnames=('mode', 'pivoting'))
def _qr(a: ArrayLike, mode: str, pivoting: bool
       ) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]:
  if mode in ("full", "r"):
    full_matrices = True
  elif mode == "economic":
    full_matrices = False
  else:
    raise ValueError(f"Unsupported QR decomposition mode '{mode}'")
  a, = promote_dtypes_inexact(jnp.asarray(a))
  q, r, *p = lax_linalg.qr(a, pivoting=pivoting, full_matrices=full_matrices)
  if mode == "r":
    if pivoting:
      return r, p[0]
    return (r,)
  if pivoting:
    return q, r, p[0]
  return q, r


@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["full", "economic"], pivoting: Literal[False] = False,
       check_finite: bool = True) -> tuple[Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["full", "economic"], pivoting: Literal[True] = True,
       check_finite: bool = True) -> tuple[Array, Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["full", "economic"], pivoting: bool = False,
       check_finite: bool = True
      ) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["r"], pivoting: Literal[False] = False, check_finite: bool = True
      ) -> tuple[Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["r"], pivoting: Literal[True] = True, check_finite: bool = True
      ) -> tuple[Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *,
       mode: Literal["r"], pivoting: bool = False, check_finite: bool = True
      ) -> tuple[Array] | tuple[Array, Array]: ...

@overload
def qr(a: ArrayLike, overwrite_a: bool = False, lwork: Any = None, mode: str = "full",
       pivoting: bool = False, check_finite: bool = True
      ) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]: ...


def qr(a: ArrayLike, overwrite_a: bool = False, lwork: Any = None, mode: str = "full",
       pivoting: bool = False, check_finite: bool = True
      ) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]:
  """Compute the QR decomposition of an array

  JAX implementation of :func:`scipy.linalg.qr`.

  The QR decomposition of a matrix `A` is given by

  .. math::

     A = QR

  Where `Q` is a unitary matrix (i.e. :math:`Q^HQ=I`) and `R` is an upper-triangular
  matrix.

  Args:
    a: array of shape (..., M, N)
    mode: Computational mode. Supported values are:

      - ``"full"`` (default): return `Q` of shape ``(M, M)`` and `R` of shape ``(M, N)``.
      - ``"r"``: return only `R`
      - ``"economic"``: return `Q` of shape ``(M, K)`` and `R` of shape ``(K, N)``,
        where K = min(M, N).

    pivoting: Allows the QR decomposition to be rank-revealing. If ``True``, compute
      the column-pivoted decomposition ``A[:, P] = Q @ R``, where ``P`` is chosen such
      that the diagonal of ``R`` is non-increasing.
    overwrite_a: unused in JAX
    lwork: unused in JAX
    check_finite: unused in JAX

  Returns:
    A tuple ``(Q, R)`` or ``(Q, R, P)``, if ``mode`` is not ``"r"`` and ``pivoting`` is
    respectively ``False`` or ``True``, otherwise an array ``R`` or tuple ``(R, P)`` if
    mode is ``"r"``, and ``pivoting`` is respectively ``False`` or ``True``, where:

    - ``Q`` is an orthogonal matrix of shape ``(..., M, M)`` (if ``mode`` is ``"full"``)
      or ``(..., M, K)`` (if ``mode`` is ``"economic"``),
    - ``R`` is an upper-triangular matrix of shape ``(..., M, N)`` (if ``mode`` is
      ``"r"`` or ``"full"``) or ``(..., K, N)`` (if ``mode`` is ``"economic"``),
    - ``P`` is an index vector of shape ``(..., N)``.

    with ``K = min(M, N)``.

  Notes:
    - At present, pivoting is only implemented on the CPU and GPU backends. For further
      details about the GPU implementation, see the documentation for
      :func:`jax.lax.linalg.qr`.

  See also:
    - :func:`jax.numpy.linalg.qr`: NumPy-style QR decomposition API
    - :func:`jax.lax.linalg.qr`: XLA-style QR decomposition API

  Examples:
    Compute the QR decomposition of a matrix:

    >>> a = jnp.array([[1., 2., 3., 4.],
    ...                [5., 4., 2., 1.],
    ...                [6., 3., 1., 5.]])
    >>> Q, R = jax.scipy.linalg.qr(a)
    >>> Q  # doctest: +SKIP
    Array([[-0.12700021, -0.7581426 , -0.6396022 ],
           [-0.63500065, -0.43322435,  0.63960224],
           [-0.7620008 ,  0.48737738, -0.42640156]], dtype=float32)
    >>> R  # doctest: +SKIP
    Array([[-7.8740077, -5.080005 , -2.4130025, -4.953006 ],
           [ 0.       , -1.7870499, -2.6534991, -1.028908 ],
           [ 0.       ,  0.       , -1.0660033, -4.050814 ]], dtype=float32)

    Check that ``Q`` is orthonormal:

    >>> jnp.allclose(Q.T @ Q, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

    Reconstruct the input:

    >>> jnp.allclose(Q @ R, a)
    Array(True, dtype=bool)
  """
  del overwrite_a, lwork, check_finite  # unused
  return _qr(a, mode, pivoting)


@overload
def qr_multiply(a: ArrayLike, c: ArrayLike, mode: str = 'right',
                pivoting: Literal[False] = False, conjugate: bool = False,
                overwrite_a: bool = False, overwrite_c: bool = False
                ) -> tuple[Array, Array]: ...

@overload
def qr_multiply(a: ArrayLike, c: ArrayLike, mode: str = 'right',
                pivoting: Literal[True] = True, conjugate: bool = False,
                overwrite_a: bool = False, overwrite_c: bool = False
                ) -> tuple[Array, Array, Array]: ...

@overload
def qr_multiply(a: ArrayLike, c: ArrayLike, mode: str = 'right',
                pivoting: bool = False, conjugate: bool = False,
                overwrite_a: bool = False, overwrite_c: bool = False
                ) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...


def qr_multiply(a: ArrayLike, c: ArrayLike, mode: str = 'right',
                pivoting: bool = False, conjugate: bool = False,
                overwrite_a: bool = False, overwrite_c: bool = False
                ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  """Calculate the QR decomposition and multiply Q with a matrix.

  JAX implementation of :func:`scipy.linalg.qr_multiply`.

  Args:
    a: array of shape ``(..., M, N)``. Matrix to be decomposed.
    c: array to be multiplied by Q. For ``mode='left'``, ``c`` has shape
      ``(..., K, P)`` where ``K = min(M, N)``. For ``mode='right'``, ``c``
      has shape ``(..., P, M)``. 1-D arrays are supported: for
      ``mode='left'``, treated as a length-``K`` column vector; for
      ``mode='right'``, treated as a length-``M`` row vector. The result
      is raveled back to 1-D in either case.
    mode: ``'right'`` (default) or ``'left'``.

      - ``'left'``: compute ``Q @ c`` (or ``conj(Q) @ c`` if ``conjugate=True``)
        and return ``(Q @ c, R)`` with result shape ``(..., M, P)``
      - ``'right'``: compute ``c @ Q`` (or ``c @ conj(Q)`` if ``conjugate=True``)
        and return ``(c @ Q, R)`` with result shape ``(..., P, K)`` where
        ``K = min(M, N)``
    pivoting: Allows the QR decomposition to be rank-revealing. If ``True``, compute
      the column-pivoted QR decomposition and return permutation indices
      as a third element.
    conjugate: If ``True``, use ``conj(Q)`` (element-wise complex conjugate)
      instead of ``Q``. For real arrays this has no effect.
    overwrite_a: unused in JAX
    overwrite_c: unused in JAX

  Returns:
    If ``pivoting`` is ``False``: ``(result, R)``

    If ``pivoting`` is ``True``: ``(result, R, P)``

  See also:
    - :func:`jax.scipy.linalg.qr`: SciPy-style QR decomposition API
    - :func:`jax.lax.linalg.ormqr`: XLA-style Q-multiply primitive

  Examples:
    Use :func:`qr_multiply` to efficiently solve a least-squares problem.
    For an overdetermined system ``A @ x ≈ b``, pass ``b`` as a 1-D row
    via ``mode='right'`` to obtain ``Q^T @ b`` and ``R`` in one step:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
    >>> b = jnp.array([2., 4., 5., 4.])
    >>> Qtb, R = jax.scipy.linalg.qr_multiply(A, b, mode='right')
    >>> x = jax.scipy.linalg.solve_triangular(R, Qtb)
    >>> jnp.allclose(A.T @ A @ x, A.T @ b)
    Array(True, dtype=bool)
  """
  del overwrite_a, overwrite_c  # unused
  a, c = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(c))
  if mode not in ('right', 'left'):
    raise ValueError(f"mode must be 'right' or 'left', got {mode!r}")

  onedim = c.ndim == 1
  if onedim:
    c = c[:, None] if mode == 'left' else c[None, :]

  m, n = a.shape[-2:]
  k = min(m, n)

  if mode == 'left':
    if c.shape[-2] != k:
      raise ValueError(
          f"Array shapes are not compatible for Q @ c operation: "
          f"a has shape {tuple(a.shape)} so Q has {k} columns, "
          f"but c has {c.shape[-2]} rows (expected {k}).")
  else:
    if c.shape[-1] != m:
      raise ValueError(
          f"Array shapes are not compatible for c @ Q operation: "
          f"a has shape {tuple(a.shape)} so Q has {m} rows, "
          f"but c has {c.shape[-1]} columns (expected {m}).")

  batch = jnp.broadcast_shapes(a.shape[:-2], c.shape[:-2])
  a = jnp.broadcast_to(a, batch + a.shape[-2:])
  c = jnp.broadcast_to(c, batch + c.shape[-2:])

  p: Array | None = None
  if pivoting:
    jpvt = jnp.zeros(a.shape[:-2] + (n,), dtype=jnp.int32)
    r, p, taus = lax_linalg.geqp3(a, jpvt)
    p -= 1  # Convert geqp3's 1-based indices to 0-based indices by subtracting 1.
  else:
    r, taus = lax_linalg.geqrf(a)

  if m > n and mode == 'left':
    zeros = jnp.zeros(c.shape[:-2] + (m - k,) + c.shape[-1:], dtype=c.dtype)
    c = jnp.concatenate([c, zeros], axis=-2)

  if conjugate:
    # Avoid explicit jnp.conj call by instead transposing (for free under jit)
    # and telling LAPACK to compute Hermitian
    c = c.swapaxes(-1, -2)

  # conjugate swaps left/right because c and Q change sides when transposing.
  left = (mode == 'left') != conjugate
  cQ = lax_linalg.ormqr(r, taus, c, left=left, transpose=conjugate)

  if conjugate:
    cQ = cQ.swapaxes(-1, -2)

  if mode == 'right':
    cQ = cQ[..., :k]

  if onedim:
    cQ = cQ.ravel()

  r = jnp.triu(r[..., :k, :])

  if pivoting:
    assert p is not None
    return cQ, r, p
  return cQ, r


@jit(static_argnames=('assume_a', 'lower'))
def _solve(a: ArrayLike, b: ArrayLike, assume_a: str, lower: bool) -> Array:
  if assume_a != 'pos':
    return jnp_linalg.solve(a, b)

  a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))
  lax_linalg._check_solve_shapes(a, b)

  # With custom_linear_solve, we can reuse the same factorization when
  # computing sensitivities. This is considerably faster.
  factors = cho_factor(lax.stop_gradient(a), lower=lower)
  custom_solve = partial(
      lax.custom_linear_solve,
      lambda x: lax_linalg._broadcasted_matvec(a, x),
      solve=lambda _, x: cho_solve(factors, x[..., None]).squeeze(-1),
      symmetric=True)
  if a.ndim == b.ndim + 1:
    # b.shape == [..., m]
    return custom_solve(b)
  else:
    # b.shape == [..., m, k]
    return vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)


def solve(a: ArrayLike, b: ArrayLike, lower: bool = False,
          overwrite_a: bool = False, overwrite_b: bool = False, debug: bool = False,
          check_finite: bool = True, assume_a: str = 'gen') -> Array:
  """Solve a linear system of equations.

  JAX implementation of :func:`scipy.linalg.solve`.

  This solves a (batched) linear system of equations ``a @ x = b`` for ``x``
  given ``a`` and ``b``.

  If ``a`` is singular, this will return ``nan`` or ``inf`` values.

  Args:
    a: array of shape ``(..., N, N)``.
    b: array of shape ``(..., N)`` or ``(..., N, M)``
    lower: Referenced only if ``assume_a != 'gen'``. If True, only use the lower
      triangle of the input, If False (default), only use the upper triangle.
    assume_a: specify what properties of ``a`` can be assumed. Options are:

      - ``"gen"``: generic matrix (default)
      - ``"sym"``: symmetric matrix
      - ``"her"``: hermitian matrix
      - ``"pos"``: positive-definite matrix

    overwrite_a: unused by JAX
    overwrite_b: unused by JAX
    debug: unused by JAX
    check_finite: unused by JAX

  Returns:
    An array of the same shape as ``b`` containing the solution to the linear
    system if ``a`` is non-singular.
    If ``a`` is singular, the result contains ``nan`` or ``inf`` values.

  See also:
    - :func:`jax.scipy.linalg.lu_solve`: Solve via LU factorization.
    - :func:`jax.scipy.linalg.cho_solve`: Solve via Cholesky factorization.
    - :func:`jax.scipy.linalg.solve_triangular`: Solve a triangular system.
    - :func:`jax.numpy.linalg.solve`: NumPy-style API for solving linear systems.
    - :func:`jax.lax.custom_linear_solve`: matrix-free linear solver.

  Examples:
    A simple 3x3 linear system:

    >>> A = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> b = jnp.array([14., 16., 10.])
    >>> x = jax.scipy.linalg.solve(A, b)
    >>> x
    Array([1., 2., 3.], dtype=float32)

    Confirming that the result solves the system:

    >>> jnp.allclose(A @ x, b)
    Array(True, dtype=bool)
  """
  del overwrite_a, overwrite_b, debug, check_finite  #unused
  valid_assume_a = ['gen', 'sym', 'her', 'pos']
  if assume_a not in valid_assume_a:
    raise ValueError(f"Expected assume_a to be one of {valid_assume_a}; got {assume_a!r}")
  return _solve(a, b, assume_a, lower)

@jit(static_argnames=('trans', 'lower', 'unit_diagonal'))
def _solve_triangular(a: ArrayLike, b: ArrayLike, trans: int | str,
                      lower: bool, unit_diagonal: bool) -> Array:
  if trans == 0 or trans == "N":
    transpose_a, conjugate_a = False, False
  elif trans == 1 or trans == "T":
    transpose_a, conjugate_a = True, False
  elif trans == 2 or trans == "C":
    transpose_a, conjugate_a = True, True
  else:
    raise ValueError(f"Invalid 'trans' value {trans}")

  a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))

  if b.ndim == 1:
    signature = "(n,n),(n)->(n)"
  elif a.ndim == b.ndim + 1 and a.shape[-1] == b.shape[-1]:
    # Deprecation warning added 2026-03-23
    warnings.warn(
        "jax.scipy.linalg.solve_triangular: batched 1D solves with b.ndim > 1 "
        "are deprecated, and in the future will be treated as a batched 2D solve. "
        "Use solve_triangular(a, b[..., None]).squeeze(-1) to avoid this warning.",
        category=FutureWarning)
    signature = "(n,n),(n)->(n)"
  else:
    signature = "(n,n),(n,k)->(n,k)"

  return jnp_vectorize.vectorize(
      partial(lax_linalg.triangular_solve, left_side=True, lower=lower,
              transpose_a=transpose_a, conjugate_a=conjugate_a,
              unit_diagonal=unit_diagonal),
      signature=signature)(a, b)


def solve_triangular(a: ArrayLike, b: ArrayLike, trans: int | str = 0, lower: bool = False,
                     unit_diagonal: bool = False, overwrite_b: bool = False,
                     debug: Any = None, check_finite: bool = True) -> Array:
  """Solve a triangular linear system of equations

  JAX implementation of :func:`scipy.linalg.solve_triangular`.

  This solves a (batched) linear system of equations ``a @ x = b`` for ``x``
  given a triangular matrix ``a`` and a vector or matrix ``b``.

  Args:
    a: array of shape ``(..., N, N)``. Only part of the array will be accessed,
      depending on the ``lower`` and ``unit_diagonal`` arguments.
    b: array of shape ``(N,)`` (for a 1-dimensional right-hand-side) or
      ``(..., N, M)`` (for a batched 2-dimensional right-hand-side).
    lower: If True, only use the lower triangle of the input, If False (default),
      only use the upper triangle.
    unit_diagonal: If True, ignore diagonal elements of ``a`` and assume they are
      ``1`` (default: False).
    trans: specify what properties of ``a`` can be assumed. Options are:

      - ``0`` or ``'N'``: solve :math:`Ax=b`
      - ``1`` or ``'T'``: solve :math:`A^Tx=b`
      - ``2`` or ``'C'``: solve :math:`A^Hx=b`

    overwrite_b: unused by JAX
    debug: unused by JAX
    check_finite: unused by JAX

  Returns:
    An array containing the solution to the linear system. The result has
    shape ``(..., N)`` if ``b`` is of shape ``(N,)``, and has shape
    ``(..., N, M)`` otherwise.

  See also:
    :func:`jax.scipy.linalg.solve`: Solve a general linear system.

  Examples:
    A simple 3x3 triangular linear system:

    >>> A = jnp.array([[1., 2., 3.],
    ...                [0., 3., 2.],
    ...                [0., 0., 5.]])
    >>> b = jnp.array([10., 8., 5.])
    >>> x = jax.scipy.linalg.solve_triangular(A, b)
    >>> x
    Array([3., 2., 1.], dtype=float32)

    Confirming that the result solves the system:

    >>> jnp.allclose(A @ x, b)
    Array(True, dtype=bool)

    Computing the transposed problem:

    >>> x = jax.scipy.linalg.solve_triangular(A, b, trans='T')
    >>> x
    Array([10. , -4. , -3.4], dtype=float32)

    Confirming that the result solves the system:

    >>> jnp.allclose(A.T @ x, b)
    Array(True, dtype=bool)
  """
  del overwrite_b, debug, check_finite  # unused
  return _solve_triangular(a, b, trans, lower, unit_diagonal)


@jit(static_argnames=('upper_triangular', 'max_squarings'))
def expm(A: ArrayLike, *, upper_triangular: bool = False, max_squarings: int = 16) -> Array:
  """Compute the matrix exponential

  JAX implementation of :func:`scipy.linalg.expm`.

  Args:
    A: array of shape ``(..., N, N)``
    upper_triangular: if True, then assume that ``A`` is upper-triangular. Default=False.
    max_squarings: The number of squarings in the scaling-and-squaring approximation method
     (default: 16).

  Returns:
    An array of shape ``(..., N, N)`` containing the matrix exponent of ``A``.

  Notes:
    This uses the scaling-and-squaring approximation method, with computational complexity
    controlled by the optional ``max_squarings`` argument. Theoretically, the number of
    required squarings is ``max(0, ceil(log2(norm(A))) - c)`` where ``norm(A)`` is the L1
    norm and ``c=2.42`` for float64/complex128, or ``c=1.97`` for float32/complex64.

  See Also:
    :func:`jax.scipy.linalg.expm_frechet`

  Examples:

    ``expm`` is the matrix exponential, and has similar properties to the more
    familiar scalar exponential. For scalars ``a`` and ``b``, :math:`e^{a + b}
    = e^a e^b`. However, for matrices, this property only holds when ``A`` and
    ``B`` commute (``AB = BA``). In this case, ``expm(A+B) = expm(A) @ expm(B)``

    >>> A = jnp.array([[2, 0],
    ...                [0, 1]])
    >>> B = jnp.array([[3, 0],
    ...                [0, 4]])
    >>> jnp.allclose(jax.scipy.linalg.expm(A+B),
    ...              jax.scipy.linalg.expm(A) @ jax.scipy.linalg.expm(B),
    ...              rtol=0.0001)
    Array(True, dtype=bool)

    If a matrix ``X`` is invertible, then
    ``expm(X @ A @ inv(X)) = X @ expm(A) @ inv(X)``

    >>> X = jnp.array([[3, 1],
    ...                [2, 5]])
    >>> X_inv = jax.scipy.linalg.inv(X)
    >>> jnp.allclose(jax.scipy.linalg.expm(X @ A @ X_inv),
    ...              X @ jax.scipy.linalg.expm(A) @ X_inv)
    Array(True, dtype=bool)
  """
  A, = promote_dtypes_inexact(A)

  if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
    raise ValueError(f"Expected A to be a (batched) square matrix, got {A.shape=}.")

  if A.ndim > 2:
    return jnp_vectorize.vectorize(
      partial(expm, upper_triangular=upper_triangular, max_squarings=max_squarings),
      signature="(n,n)->(n,n)")(A)

  P, Q, n_squarings = _calc_P_Q(jnp.asarray(A))

  def _nan(args):
    A, *_ = args
    return jnp.full_like(A, np.nan)

  def _compute(args):
    A, P, Q = args
    R = _solve_P_Q(P, Q, upper_triangular)
    R = _squaring(R, n_squarings, max_squarings)
    return R

  R = lax.cond(n_squarings > max_squarings, _nan, _compute, (A, P, Q))
  return R

@jit
def _calc_P_Q(A: Array) -> tuple[Array, Array, Array]:
  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError('expected A to be a square matrix')
  A_L1 = jnp_linalg.norm(A,1)
  n_squarings: Array
  U: Array
  V: Array
  if A.dtype == 'float64' or A.dtype == 'complex128':
   maxnorm = 5.371920351148152
   n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
   A = A / 2 ** n_squarings.astype(A.dtype)
   conds = jnp.array([1.495585217958292e-002, 2.539398330063230e-001,
                      9.504178996162932e-001, 2.097847961257068e+000],
                      dtype=A_L1.dtype)
   idx = jnp.digitize(A_L1, conds)
   U, V = lax.switch(idx, [_pade3, _pade5, _pade7, _pade9, _pade13], A)
  elif A.dtype == 'float32' or A.dtype == 'complex64':
    maxnorm = 3.925724783138660
    n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
    A = A / 2 ** n_squarings.astype(A.dtype)
    conds = jnp.array([4.258730016922831e-001, 1.880152677804762e+000],
                      dtype=A_L1.dtype)
    idx = jnp.digitize(A_L1, conds)
    U, V = lax.switch(idx, [_pade3, _pade5, _pade7], A)
  else:
    raise TypeError(f"A.dtype={A.dtype} is not supported.")
  P = U + V  # p_m(A) : numerator
  Q = -U + V # q_m(A) : denominator
  return P, Q, n_squarings

def _solve_P_Q(P: ArrayLike, Q: ArrayLike, upper_triangular: bool = False) -> Array:
  if upper_triangular:
    return solve_triangular(Q, P)
  else:
    return jnp_linalg.solve(Q, P)

def _precise_dot(A: ArrayLike, B: ArrayLike) -> Array:
  return jnp.dot(A, B, precision=lax.Precision.HIGHEST)

@jit(static_argnums=2)
def _squaring(R: Array, n_squarings: Array, max_squarings: int) -> Array:
  # squaring step to undo scaling
  def _squaring_precise(x):
    return _precise_dot(x, x)

  def _identity(x):
    return x

  def _scan_f(c, i):
    return lax.cond(i < n_squarings, _squaring_precise, _identity, c), None
  res, _ = lax.scan(_scan_f, R, jnp.arange(max_squarings, dtype=n_squarings.dtype))

  return res

def _pade3(A: Array) -> tuple[Array, Array]:
  b = (120., 60., 12., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  U = _precise_dot(A, (b[3]*A2 + b[1]*ident))
  V: Array = b[2]*A2 + b[0]*ident
  return U, V

def _pade5(A: Array) -> tuple[Array, Array]:
  b = (30240., 15120., 3360., 420., 30., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  U = _precise_dot(A, b[5]*A4 + b[3]*A2 + b[1]*ident)
  V: Array = b[4]*A4 + b[2]*A2 + b[0]*ident
  return U, V

def _pade7(A: Array) -> tuple[Array, Array]:
  b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  A6 = _precise_dot(A4, A2)
  U = _precise_dot(A, b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
  return U,V

def _pade9(A: Array) -> tuple[Array, Array]:
  b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
       2162160., 110880., 3960., 90., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  A6 = _precise_dot(A4, A2)
  A8 = _precise_dot(A6, A2)
  U = _precise_dot(A, b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
  return U,V

def _pade13(A: Array) -> tuple[Array, Array]:
  b = (64764752532480000., 32382376266240000., 7771770303897600.,
       1187353796428800., 129060195264000., 10559470521600., 670442572800.,
       33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
  M, N = A.shape
  ident = jnp.eye(M, N, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  A6 = _precise_dot(A4, A2)
  U = _precise_dot(A, _precise_dot(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = _precise_dot(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
  return U,V


@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: Literal[True] = True) -> tuple[Array, Array]: ...

@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: Literal[False]) -> Array: ...

@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: bool = True) -> Array | tuple[Array, Array]: ...


@jit(static_argnames=('method', 'compute_expm'))
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: bool = True) -> Array | tuple[Array, Array]:
  """Compute the Frechet derivative of the matrix exponential.

  JAX implementation of :func:`scipy.linalg.expm_frechet`

  Args:
    A: array of shape ``(..., N, N)``
    E: array of shape ``(..., N, N)``; specifies the direction of the derivative.
    compute_expm: if True (default) then compute and return ``expm(A)``.
    method: ignored by JAX

  Returns:
    A tuple ``(expm_A, expm_frechet_AE)`` if ``compute_expm`` is True, else
    the array ``expm_frechet_AE``. Both returned arrays have shape ``(..., N, N)``.

  See also:
    :func:`jax.scipy.linalg.expm`

  Examples:
    We can use this API to compute the matrix exponential of ``A``, as well as its
    derivative in the direction ``E``:

    >>> key1, key2 = jax.random.split(jax.random.key(3372))
    >>> A = jax.random.normal(key1, (3, 3))
    >>> E = jax.random.normal(key2, (3, 3))
    >>> expmA, expm_frechet_AE = jax.scipy.linalg.expm_frechet(A, E)

    This can be equivalently computed using JAX's automatic differentiation methods;
    here we'll compute the derivative of :func:`~jax.scipy.linalg.expm` in the
    direction of ``E`` using :func:`jax.jvp`, and find the same results:

    >>> expmA2, expm_frechet_AE2 = jax.jvp(jax.scipy.linalg.expm, (A,), (E,))
    >>> jnp.allclose(expmA, expmA2)
    Array(True, dtype=bool)
    >>> jnp.allclose(expm_frechet_AE, expm_frechet_AE2)
    Array(True, dtype=bool)
  """
  del method  # unused
  A_arr = jnp.asarray(A)
  E_arr = jnp.asarray(E)
  if A_arr.ndim < 2 or A_arr.shape[-2] != A_arr.shape[1]:
    raise ValueError(f'expected A to be a (batched) square matrix, got A.shape={A_arr.shape}')
  if E_arr.ndim < 2 or E_arr.shape[-2] != E_arr.shape[-1]:
    raise ValueError(f'expected E to be a (batched) square matrix, got E.shape={E_arr.shape}')
  if A_arr.shape != E_arr.shape:
    raise ValueError('expected A and E to be the same shape, got '
                     f'A.shape={A_arr.shape} E.shape={E_arr.shape}')
  bound_fun = partial(expm, upper_triangular=False, max_squarings=16)
  expm_A, expm_frechet_AE = jvp(bound_fun, (A_arr,), (E_arr,))
  if compute_expm:
    return expm_A, expm_frechet_AE
  else:
    return expm_frechet_AE


@jit
def block_diag(*arrs: ArrayLike) -> Array:
  """Create a block diagonal matrix from input arrays.

  JAX implementation of :func:`scipy.linalg.block_diag`.

  Args:
    *arrs: arrays of at most two dimensions

  Returns:
    2D block-diagonal array constructed by placing the input arrays
    along the diagonal.

  Examples:
    >>> A = jnp.ones((1, 1))
    >>> B = jnp.ones((2, 2))
    >>> C = jnp.ones((3, 3))
    >>> jax.scipy.linalg.block_diag(A, B, C)
    Array([[1., 0., 0., 0., 0., 0.],
           [0., 1., 1., 0., 0., 0.],
           [0., 1., 1., 0., 0., 0.],
           [0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 1.]], dtype=float32)
  """
  if len(arrs) == 0:
    arrs = (jnp.zeros((1, 0)),)
  arrs = tuple(promote_dtypes(*arrs))
  bad_shapes = [i for i, a in enumerate(arrs) if np.ndim(a) > 2]
  if bad_shapes:
    raise ValueError("Arguments to jax.scipy.linalg.block_diag must have at "
                     "most 2 dimensions, got {} at argument {}."
                     .format(arrs[bad_shapes[0]], bad_shapes[0]))
  converted_arrs = [jnp.atleast_2d(a) for a in arrs]
  dtype = lax.dtype(converted_arrs[0])
  total_cols = sum(a.shape[1] for a in converted_arrs)

  padded_arrs = []
  current_col = 0
  for arr in converted_arrs:
    cols = arr.shape[1]
    padding_config = ((0, 0, 0), (current_col, total_cols - cols - current_col, 0))
    padded_arrs.append(lax.pad(arr, dtype.type(0), padding_config))
    current_col += cols
  return jnp.concatenate(padded_arrs, axis=0)


@jit(static_argnames=("eigvals_only", "select", "select_range"))
def eigh_tridiagonal(d: ArrayLike, e: ArrayLike, *, eigvals_only: bool = False,
                     select: str = 'a', select_range: tuple[float, float] | None = None,
                     tol: float | None = None,
                     key: Array | None = None) -> Array | tuple[Array, Array]:
  """Solve the eigenvalue problem for a symmetric real tridiagonal matrix

  JAX implementation of :func:`scipy.linalg.eigh_tridiagonal`.

  Args:
    d: real-valued array of shape ``(N,)`` specifying the diagonal elements.
    e: real-valued array of shape ``(N - 1,)`` specifying the off-diagonal elements.
    eigvals_only: If True, return only the eigenvalues (default: False). Computation
      of eigenvectors is not yet implemented, so ``eigvals_only`` must be set to True.
    select: specify which eigenvalues to calculate. Supported values are:

      - ``'a'``: all eigenvalues
      - ``'i'``: eigenvalues with indices ``select_range[0] <= i <= select_range[1]``

      JAX does not currently implement ``select = 'v'``.
    select_range: range of values used when ``select='i'``.
    tol: absolute tolerance to use when solving for the eigenvalues.
    key: a PRNG key, as returned by ``jax.random.key``, used to generate random
      initialization vectors for inverse iteration. If ``None``, defaults to a
      fixed PRNG key.

  Returns:
    An array of eigenvalues with shape ``(N,)``.

  See also:
    :func:`jax.scipy.linalg.eigh`: general Hermitian eigenvalue solver

  Examples:
    >>> d = jnp.array([1., 2., 3., 4.])
    >>> e = jnp.array([1., 1., 1.])
    >>> eigvals = jax.scipy.linalg.eigh_tridiagonal(d, e, eigvals_only=True)
    >>> eigvals
    Array([0.2547188, 1.8227171, 3.1772828, 4.745281 ], dtype=float32)

    For comparison, we can construct the full matrix and compute the same result
    using :func:`~jax.scipy.linalg.eigh`:

    >>> A = jnp.diag(d) + jnp.diag(e, 1) + jnp.diag(e, -1)
    >>> A
    Array([[1., 1., 0., 0.],
           [1., 2., 1., 0.],
           [0., 1., 3., 1.],
           [0., 0., 1., 4.]], dtype=float32)
    >>> eigvals_full = jax.scipy.linalg.eigh(A, eigvals_only=True)
    >>> jnp.allclose(eigvals, eigvals_full)
    Array(True, dtype=bool)
  """
  if not eigvals_only and key is None:
    key = random.key(42)

  def _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, x):
    """Implements the Sturm sequence recurrence."""
    n = alpha.shape[0]
    zeros = jnp.zeros(x.shape, dtype=np.int32)
    ones = jnp.ones(x.shape, dtype=np.int32)

    # The first step in the Sturm sequence recurrence
    # requires special care if x is equal to alpha[0].
    def sturm_step0():
      q = alpha[0] - x
      count = jnp.where(q < 0, ones, zeros)
      q = jnp.where(alpha[0] == x, alpha0_perturbation, q)
      return q, count

    # Subsequent steps all take this form:
    def sturm_step(i, q, count):
      q = alpha[i] - beta_sq[i - 1] / q - x
      count = jnp.where(q <= pivmin, count + 1, count)
      q = jnp.where(q <= pivmin, jnp.minimum(q, -pivmin), q)
      return q, count

    # The first step initializes q and count.
    q, count = sturm_step0()

    # Peel off ((n-1) % blocksize) steps from the main loop, so we can run
    # the bulk of the iterations unrolled by a factor of blocksize.
    blocksize = 16
    i = 1
    peel = (n - 1) % blocksize
    unroll_cnt = peel

    def unrolled_steps(args):
      start, q, count = args
      for j in range(unroll_cnt):
        q, count = sturm_step(start + j, q, count)
      return start + unroll_cnt, q, count

    i, q, count = unrolled_steps((i, q, count))

    # Run the remaining steps of the Sturm sequence using a partially
    # unrolled while loop.
    unroll_cnt = blocksize
    def cond(iqc):
      i, q, count = iqc
      return jnp.less(i, n)
    _, _, count = lax.while_loop(cond, unrolled_steps, (i, q, count))
    return count

  alpha = jnp.asarray(d)
  beta = jnp.asarray(e)
  supported_dtypes = (np.float32, np.float64, np.complex64, np.complex128)
  if alpha.dtype != beta.dtype:
    raise TypeError("diagonal and off-diagonal values must have same dtype, "
                    f"got {alpha.dtype} and {beta.dtype}")
  if alpha.dtype not in supported_dtypes or beta.dtype not in supported_dtypes:
    raise TypeError("Only float32 and float64 inputs are supported as inputs "
                    "to jax.scipy.linalg.eigh_tridiagonal, got "
                    f"{alpha.dtype} and {beta.dtype}")
  n = alpha.shape[0]
  if n <= 1:
    if eigvals_only:
      return jnp.real(alpha)
    else:
      return jnp.real(alpha), jnp.eye(n, dtype=alpha.dtype)

  if dtypes.issubdtype(alpha.dtype, np.complexfloating):
    alpha = jnp.real(alpha)
    beta_sq = jnp.real(beta * jnp.conj(beta))
    beta_abs = jnp.sqrt(beta_sq)
  else:
    beta_abs = jnp.abs(beta)
    beta_sq = jnp.square(beta)

  # Estimate the largest and smallest eigenvalues of T using the Gershgorin
  # circle theorem.
  off_diag_abs_row_sum = jnp.concatenate(
      [beta_abs[:1], beta_abs[:-1] + beta_abs[1:], beta_abs[-1:]], axis=0)
  lambda_est_max = jnp.amax(alpha + off_diag_abs_row_sum)
  lambda_est_min = jnp.amin(alpha - off_diag_abs_row_sum)
  # Upper bound on 2-norm of T.
  t_norm = jnp.maximum(jnp.abs(lambda_est_min), jnp.abs(lambda_est_max))

  # Compute the smallest allowed pivot in the Sturm sequence to avoid
  # overflow.
  finfo = np.finfo(alpha.dtype)
  one = np.ones([], dtype=alpha.dtype)
  safemin = np.maximum(one / finfo.max, (one + finfo.eps) * finfo.tiny)
  pivmin = safemin * jnp.maximum(1, jnp.amax(beta_sq))
  alpha0_perturbation = jnp.square(finfo.eps * beta_abs[0])
  abs_tol = finfo.eps * t_norm
  if tol is not None:
    abs_tol = jnp.maximum(tol, abs_tol)

  # In the worst case, when the absolute tolerance is eps*lambda_est_max and
  # lambda_est_max = -lambda_est_min, we have to take as many bisection steps
  # as there are bits in the mantissa plus 1.
  # The proof is left as an exercise to the reader.
  max_it = finfo.nmant + 1

  # Determine the indices of the desired eigenvalues, based on select and
  # select_range.
  if select == 'a':
    target_counts = jnp.arange(n, dtype=np.int32)
  elif select == 'i':
    if select_range is None:
      raise ValueError("for select='i', select_range must be specified.")
    if select_range[0] > select_range[1]:
      raise ValueError('Got empty index range in select_range.')
    target_counts = jnp.arange(select_range[0], select_range[1] + 1, dtype=np.int32)
  elif select == 'v':
    # TODO(phawkins): requires dynamic shape support.
    raise NotImplementedError("eigh_tridiagonal(..., select='v') is not "
                              "implemented")
  else:
    raise ValueError("'select must have a value in {'a', 'i', 'v'}.")

  # Run binary search for all desired eigenvalues in parallel, starting from
  # the interval lightly wider than the estimated
  # [lambda_est_min, lambda_est_max].
  fudge = 2.1  # We widen starting interval the Gershgorin interval a bit.
  norm_slack = jnp.array(n, alpha.dtype) * fudge * finfo.eps * t_norm
  lower = lambda_est_min - norm_slack - 2 * fudge * pivmin
  upper = lambda_est_max + norm_slack + fudge * pivmin

  # Pre-broadcast the scalars used in the Sturm sequence for improved
  # performance.
  target_shape = np.shape(target_counts)
  lower = jnp.broadcast_to(lower, shape=target_shape)
  upper = jnp.broadcast_to(upper, shape=target_shape)
  mid = 0.5 * (upper + lower)
  pivmin = jnp.broadcast_to(pivmin, target_shape)
  alpha0_perturbation = jnp.broadcast_to(alpha0_perturbation, target_shape)

  # Start parallel binary searches.
  def cond(args):
    i, lower, _, upper = args
    return jnp.logical_and(
        jnp.less(i, max_it),
        jnp.less(abs_tol, jnp.amax(upper - lower)))

  def body(args):
    i, lower, mid, upper = args
    counts = _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, mid)
    lower = jnp.where(counts <= target_counts, mid, lower)
    upper = jnp.where(counts > target_counts, mid, upper)
    mid = 0.5 * (lower + upper)
    return i + 1, lower, mid, upper

  def _compute_eigenvectors(alpha, beta, eigvals, key):
    """Implements inverse iteration to compute eigenvectors."""
    n = alpha.shape[0]
    k = eigvals.shape[0]

    # Pad beta to length n
    dl = jnp.pad(jnp.conj(beta), (1, 0))
    du = jnp.pad(beta, (0, 1))

    # Eigenvectors corresponding to cluster of close eigenvalues are
    # not unique and need to be explicitly orthogonalized. Here we
    # identify such clusters. Note: This function assumes that
    # eigenvalues are sorted in non-decreasing order.
    gap = eigvals[1:] - eigvals[:-1]
    eps = np.finfo(eigvals.dtype).eps
    t_norm = jnp.maximum(jnp.abs(eigvals[0]), jnp.abs(eigvals[-1]))
    gaptol = jnp.sqrt(eps) * t_norm

    # Find the beginning and end of runs of eigenvectors corresponding
    # to eigenvalues closer than "gaptol", which will need to be
    # orthogonalized against each other.
    close = gap < gaptol
    left_neighbor_close = jnp.pad(close, (1, 0), constant_values=False)
    right_neighbor_close = jnp.pad(close, (0, 1), constant_values=False)

    ortho_interval_start = jnp.logical_and(
        jnp.logical_not(left_neighbor_close), right_neighbor_close)
    ortho_interval_end = jnp.logical_and(
        left_neighbor_close, jnp.logical_not(right_neighbor_close))

    max_clusters = k // 2 + 1
    starts = jnp.nonzero(ortho_interval_start, size=max_clusters)[0]
    ends = jnp.nonzero(ortho_interval_end, size=max_clusters)[0] + 1
    num_clusters = jnp.sum(ortho_interval_start)

    arange_k = jnp.arange(k)

    solve_dtype = np.result_type(alpha.dtype, beta.dtype)
    base_dtype = np.finfo(solve_dtype).dtype
    # We perform inverse iteration for all eigenvectors in parallel,
    # starting from a random set of vectors, until all have converged.
    v = random.normal(key, (k, n), dtype=base_dtype).astype(solve_dtype)
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True).astype(solve_dtype)


    def orthogonalize_close_eigenvectors(ev):
      # Eigenvectors corresponding to a cluster of close eigenvalues are not
      # uniquely defined, but the subspace they span is. To avoid numerical
      # instability, we explicitly mutually orthogonalize such eigenvectors
      # after each step of inverse iteration. It is customary to use
      # modified Gram-Schmidt for this, but this is not very efficient
      # on some platforms, so here we defer to the QR decomposition in JAX.
      def orthogonalize_cluster(i, ev):
        start = starts[i]
        end = ends[i]
        c = end - start

        ev_padded = jnp.pad(ev, ((0, k), (0, 0)))
        v_block = lax.dynamic_slice(ev_padded, (start, 0), (k, n))

        mask = (arange_k < c).astype(ev.dtype)
        v_block_masked = v_block * mask[:, None]

        # We use the builtin QR factorization to orthonormalize the
        # vectors in the cluster.
        Q, _ = jnp_linalg.qr(v_block_masked.T)

        QT = Q.T
        QT_masked = QT * mask[:, None]

        big_zeros = jnp.zeros((2 * k, n), dtype=ev.dtype)
        big_update = lax.dynamic_update_slice(big_zeros, QT_masked, (start, 0))
        update_full = big_update[:k, :]

        is_in_cluster = (arange_k >= start) & (arange_k < end)
        ev = jnp.where(is_in_cluster[:, None], update_full, ev)

        return ev

      ev = lax.fori_loop(0, num_clusters, orthogonalize_cluster, ev)
      return ev

    # Replicate alpha-shifted and beta across the k eigenvectors so we
    # can solve the k systems
    #    [T - eigvals(i)*eye(n)] x_i = r_i
    # simultaneously using the batching mechanism.
    alpha_shifted = alpha[None, :] - eigvals[:, None]
    dl_batched = jnp.broadcast_to(dl[None, :], (k, n))
    du_batched = jnp.broadcast_to(du[None, :], (k, n))

    def inverse_iteration_step(state):
      i, v, nrm_v, nrm_v_old = state

      v_new = lax_linalg.tridiagonal_solve(
          dl_batched.astype(solve_dtype),
          alpha_shifted.astype(solve_dtype),
          du_batched.astype(solve_dtype),
          v[:, :, None].astype(solve_dtype),
          perturb_singular=True)
      v_new = jnp.squeeze(v_new, axis=-1)

      nrm_v_new = jnp.linalg.norm(v_new, axis=-1, keepdims=True)
      v_new = v_new / nrm_v_new.astype(v_new.dtype)
      nrm_v_new = jnp.squeeze(nrm_v_new, axis=-1)

      v_new = orthogonalize_close_eigenvectors(v_new)

      return i + 1, v_new, nrm_v_new, nrm_v

    def continue_iteration(state):
      i, _, nrm_v, nrm_v_old = state
      max_it = 5  # Taken from LAPACK xSTEIN.
      min_norm_growth = 0.1
      norm_growth_factor = 1 + min_norm_growth
      # We stop the inverse iteration when we reach the maximum number of
      # iterations or the norm growths is less than 10%.
      return jnp.logical_and(
          i < max_it,
          jnp.any(nrm_v >= norm_growth_factor * nrm_v_old)
      )

    _, v_final, _, _ = lax.while_loop(
        continue_iteration,
        inverse_iteration_step,
        (0, v, jnp.ones(k, dtype=eigvals.dtype), jnp.zeros(k, dtype=eigvals.dtype))
    )

    return v_final

  _, _, mid, _ = lax.while_loop(cond, body, (0, lower, mid, upper))

  if eigvals_only:
    return mid

  eigenvectors = _compute_eigenvectors(alpha, beta, mid, key)
  return mid, eigenvectors.T

@jit(static_argnames=('side', 'method'))
@config.default_matmul_precision("float32")
def _polar_2d(a: Array, side: str, method: str, eps: float | None,
              max_iterations: int | None) -> tuple[Array, Array]:
  m, n = a.shape
  if method == "qdwh":
    # TODO(phawkins): return info also if the user opts in?
    if m >= n and side == "right":
      unitary, posdef, _, _ = qdwh.qdwh(a, is_hermitian=False, eps=eps)
    elif m < n and side == "left":
      a = a.T.conj()
      unitary, posdef, _, _ = qdwh.qdwh(a, is_hermitian=False, eps=eps)
      posdef = posdef.T.conj()
      unitary = unitary.T.conj()
    else:
      raise NotImplementedError("method='qdwh' only supports mxn matrices "
                                "where m < n where side='right' and m >= n "
                                f"side='left', got {a.shape} with {side=}")
  elif method == "svd":
    u_svd, s_svd, vh_svd = lax_linalg.svd(a, full_matrices=False)
    s_svd = s_svd.astype(u_svd.dtype)
    unitary = u_svd @ vh_svd
    if side == "right":
      posdef = (vh_svd.T.conj() * s_svd[None, :]) @ vh_svd
    else:
      posdef = (u_svd * s_svd[None, :]) @ (u_svd.T.conj())
  else:
    raise ValueError(f"Unknown polar decomposition method {method}.")
  return unitary, posdef


@jit(static_argnames=('side', 'method'))
def polar(a: ArrayLike, side: str = 'right', *, method: str = 'qdwh', eps: float | None = None,
          max_iterations: int | None = None) -> tuple[Array, Array]:
  r"""Computes the polar decomposition.

  Given the :math:`m \times n` matrix :math:`a`, returns the factors of the polar
  decomposition :math:`u` (also :math:`m \times n`) and :math:`p` such that
  :math:`a = up` (if side is ``"right"``; :math:`p` is :math:`n \times n`) or
  :math:`a = pu` (if side is ``"left"``; :math:`p` is :math:`m \times m`),
  where :math:`p` is positive semidefinite.  If :math:`a` is nonsingular,
  :math:`p` is positive definite and the
  decomposition is unique. :math:`u` has orthonormal columns unless
  :math:`n > m`, in which case it has orthonormal rows.

  Writing the SVD of :math:`a` as
  :math:`a = u_\mathit{svd} \cdot s_\mathit{svd} \cdot v^h_\mathit{svd}`, we
  have :math:`u = u_\mathit{svd} \cdot v^h_\mathit{svd}`. Thus the unitary
  factor :math:`u` can be constructed as the application of the sign function to
  the singular values of :math:`a`; or, if :math:`a` is Hermitian, the
  eigenvalues.

  Several methods exist to compute the polar decomposition. Currently two
  are supported:

  * ``method="svd"``:

    Computes the SVD of :math:`a` and then forms
    :math:`u = u_\mathit{svd} \cdot v^h_\mathit{svd}`.

  * ``method="qdwh"``:

    Applies the `QDWH`_ (QR-based Dynamically Weighted Halley) algorithm.

  Args:
    a: array of shape ``(..., m, n)``.
    side: Determines whether a right or left polar decomposition is computed.
      If ``side`` is ``"right"`` then :math:`a = up`. If ``side`` is ``"left"``
      then :math:`a = pu`. The default is ``"right"``.
    method: Determines the algorithm used, as described above.
    eps: The final result will satisfy
      :math:`\left|x_k - x_{k-1}\right| < \left|x_k\right| (4\epsilon)^{\frac{1}{3}}`,
      where :math:`x_k` are the QDWH iterates. Ignored if ``method`` is not
      ``"qdwh"``.
    max_iterations: Iterations will terminate after this many steps even if the
      above is unsatisfied.  Ignored if ``method`` is not ``"qdwh"``.

  Returns:
    A ``(unitary, posdef)`` tuple, where ``unitary`` is the unitary factor
    of shape ``(..., m, n)``, and ``posdef`` is the positive-semidefinite
    factor. ``posdef`` has shape ``(..., n, n)`` or ``(..., m, m)`` depending
    on whether ``side`` is ``"right"`` or ``"left"``, respectively.

  Examples:

    Polar decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [5., 4., 2.],
    ...                [3., 2., 1.]])
    >>> U, P = jax.scipy.linalg.polar(a)

    U is a Unitary Matrix:

    >>> jnp.round(U.T @ U)  # doctest: +SKIP
    Array([[ 1., -0., -0.],
           [-0.,  1.,  0.],
           [-0.,  0.,  1.]], dtype=float32)

    P is positive-semidefinite Matrix:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(P)
    [[4.79 3.25 1.23]
     [3.25 3.06 2.01]
     [1.23 2.01 2.91]]

    The original matrix can be reconstructed by multiplying the U and P:

    >>> a_reconstructed = U @ P
    >>> jnp.allclose(a, a_reconstructed)
    Array(True, dtype=bool)

  .. _QDWH: https://epubs.siam.org/doi/abs/10.1137/090774999
  """
  arr = jnp.asarray(a)
  if arr.ndim < 2:
    raise ValueError("The input `a` must be at least a 2-D array.")

  if side not in ["right", "left"]:
    raise ValueError("The argument `side` must be either 'right' or 'left'.")

  sig = "(m,n)->(m,n),(n,n)" if side == "right" else "(m,n)->(m,n),(m,m)"
  return jnp_vectorize.vectorize(
      partial(_polar_2d, side=side, method=method, eps=eps,
              max_iterations=max_iterations),
      signature=sig)(arr)


@jit
def _sqrtm_triu(T: Array) -> Array:
  """
  Implements Björck, Å., & Hammarling, S. (1983).
      "A Schur method for the square root of a matrix". Linear algebra and
      its applications", 52, 127-140.
  """
  diag = jnp.sqrt(jnp.diag(T))
  n = diag.size
  U = jnp.diag(diag)

  def i_loop(l, data):
    j, U = data
    i = j - 1 - l
    s = lax.fori_loop(i + 1, j, lambda k, val: val + U[i, k] * U[k, j], 0.0)
    value = jnp.where(T[i, j] == s, 0.0,
                      (T[i, j] - s) / (diag[i] + diag[j]))
    return j, U.at[i, j].set(value)

  def j_loop(j, U):
    _, U = lax.fori_loop(0, j, i_loop, (j, U))
    return U

  U = lax.fori_loop(0, n, j_loop, U)
  return U

@jit
def _sqrtm(A: Array) -> Array:
  T, Z = schur(A, output='complex')
  sqrt_T = _sqrtm_triu(T)
  return jnp.matmul(jnp.matmul(Z, sqrt_T, precision=lax.Precision.HIGHEST),
                    jnp.conj(Z.T), precision=lax.Precision.HIGHEST)


def sqrtm(A: ArrayLike, blocksize: int = 1) -> Array:
  """Compute the matrix square root

  This function is implemented using :func:`scipy.linalg.schur`, which is only
  supported on CPU.

  JAX implementation of :func:`scipy.linalg.sqrtm`.

  Args:
    A: array of shape ``(N, N)``
    blocksize: Not supported in JAX; JAX always uses ``blocksize=1``.

  Returns:
    An array of shape ``(N, N)`` containing the matrix square root of ``A``

  See Also:
    :func:`jax.scipy.linalg.expm`

  Examples:
    >>> a = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> sqrt_a = jax.scipy.linalg.sqrtm(a)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(sqrt_a)
    [[0.92+0.71j 0.54+0.j   0.92-0.71j]
     [0.54+0.j   1.85+0.j   0.54-0.j  ]
     [0.92-0.71j 0.54-0.j   0.92+0.71j]]

    By definition, matrix multiplication of the matrix square root with itself should
    equal the input:

    >>> jnp.allclose(a, sqrt_a @ sqrt_a)
    Array(True, dtype=bool)

  Notes:
    This function implements the complex Schur method described in [1]_.  It does not use
    recursive blocking to speed up computations as a Sylvester Equation solver is not
    yet available in JAX.

  References:
    .. [1] Björck, Å., & Hammarling, S. (1983). "A Schur method for the square root of a matrix".
           Linear algebra and its applications, 52, 127-140.
  """
  if blocksize > 1:
      raise NotImplementedError("Blocked version is not implemented yet.")
  return _sqrtm(ensure_arraylike("scipy.sqrtm", A))


@jit
def _rsf2csf_2d(T: Array, Z: Array) -> tuple[Array, Array]:
  T, Z = promote_dtypes_complex(T, Z)
  eps = dtypes.finfo(T.dtype).eps
  N = T.shape[0]

  if N == 1:
    return T, Z

  def _update_T_Z(m, T, Z):
    mu = jnp_linalg.eigvals(lax.dynamic_slice(T, (m-1, m-1), (2, 2))) - T[m, m]
    r = jnp_linalg.norm(jnp.array([mu[0], T[m, m-1]])).astype(T.dtype)
    c = mu[0] / r
    s = T[m, m-1] / r
    G = jnp.array([[c.conj(), s], [-s, c]], dtype=T.dtype)

    T_rows = lax.dynamic_slice_in_dim(T, m-1, 2, axis=0)
    col_mask = jnp.arange(N) >= m-1
    G_dot_T_zeroed_cols = G @ jnp.where(col_mask, T_rows, 0)
    T_rows_new = jnp.where(~col_mask, T_rows, G_dot_T_zeroed_cols)
    T = lax.dynamic_update_slice_in_dim(T, T_rows_new, m-1, axis=0)

    T_cols = lax.dynamic_slice_in_dim(T, m-1, 2, axis=1)
    row_mask = jnp.arange(N)[:, np.newaxis] < m+1
    T_zeroed_rows_dot_GH = jnp.where(row_mask, T_cols, 0) @ G.conj().T
    T_cols_new = jnp.where(~row_mask, T_cols, T_zeroed_rows_dot_GH)
    T = lax.dynamic_update_slice_in_dim(T, T_cols_new, m-1, axis=1)

    Z_cols = lax.dynamic_slice_in_dim(Z, m-1, 2, axis=1)
    Z = lax.dynamic_update_slice_in_dim(Z, Z_cols @ G.conj().T, m-1, axis=1)
    return T, Z

  def _rsf2csf_iter(i, TZ):
    m = N-i
    T, Z = TZ
    T, Z = lax.cond(
      jnp.abs(T[m, m-1]) > eps*(jnp.abs(T[m-1, m-1]) + jnp.abs(T[m, m])),
      _update_T_Z,
      lambda m, T, Z: (T, Z),
      m, T, Z)
    T = T.at[m, m-1].set(0.0)
    return T, Z

  return lax.fori_loop(1, N, _rsf2csf_iter, (T, Z))


@jit(static_argnames=('check_finite',))
def rsf2csf(T: ArrayLike, Z: ArrayLike, check_finite: bool = True) -> tuple[Array, Array]:
  """Convert real Schur form to complex Schur form.

  JAX implementation of :func:`scipy.linalg.rsf2csf`.

  Args:
    T: array of shape ``(..., N, N)`` containing the real Schur form of the input.
    Z: array of shape ``(..., N, N)`` containing the corresponding Schur transformation
      matrix. Batch dimensions are broadcast with those of ``T``.
    check_finite: unused by JAX

  Returns:
    A tuple of arrays ``(T, Z)`` of the same shape as the inputs, containing the
    Complex Schur form and the associated Schur transformation matrix.

  See Also:
    :func:`jax.scipy.linalg.schur`: Schur decomposition

  Examples:
    >>> A = jnp.array([[0., 3., 3.],
    ...                [0., 1., 2.],
    ...                [2., 0., 1.]])
    >>> Tr, Zr = jax.scipy.linalg.schur(A)
    >>> Tc, Zc = jax.scipy.linalg.rsf2csf(Tr, Zr)

    Both the real and complex form can be used to reconstruct the input matrix
    to float32 precision:

    >>> jnp.allclose(Zr @ Tr @ Zr.T, A, atol=1E-5)
    Array(True, dtype=bool)
    >>> jnp.allclose(Zc @ Tc @ Zc.conj().T, A, atol=1E-5)
    Array(True, dtype=bool)

    The real-valued Schur form is only quasi-upper-triangular, as we can see in this case:

    >>> with jax.numpy.printoptions(precision=2, suppress=True):
    ...   print(Tr)
    [[ 3.76 -2.17  1.38]
     [ 0.   -0.88 -0.35]
     [ 0.    2.37 -0.88]]

    By contrast, the complex form is truly upper-triangular:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(Tc)
    [[ 3.76+0.j    1.29-0.78j  2.02-0.5j ]
     [ 0.  +0.j   -0.88+0.91j -2.02+0.j  ]
     [ 0.  +0.j    0.  +0.j   -0.88-0.91j]]
  """
  del check_finite  # unused

  T_arr = jnp.asarray(T)
  Z_arr = jnp.asarray(Z)

  if T_arr.ndim < 2 or T_arr.shape[-1] != T_arr.shape[-2]:
    raise ValueError("Input 'T' must be square.")
  if Z_arr.ndim < 2 or Z_arr.shape[-1] != Z_arr.shape[-2]:
    raise ValueError("Input 'Z' must be square.")
  if T_arr.shape[-1] != Z_arr.shape[-1]:
    raise ValueError(f"Input array shapes must match: Z: {Z_arr.shape} vs. T: {T_arr.shape}")

  return jnp_vectorize.vectorize(
      _rsf2csf_2d, signature="(n,n),(n,n)->(n,n),(n,n)")(T_arr, Z_arr)

@overload
def hessenberg(a: ArrayLike, *, calc_q: Literal[False], overwrite_a: bool = False,
               check_finite: bool = True) -> Array: ...

@overload
def hessenberg(a: ArrayLike, *, calc_q: Literal[True], overwrite_a: bool = False,
               check_finite: bool = True) -> tuple[Array, Array]: ...


@jit(static_argnames=('calc_q', 'check_finite', 'overwrite_a'))
def hessenberg(a: ArrayLike, *, calc_q: bool = False, overwrite_a: bool = False,
               check_finite: bool = True) -> Array | tuple[Array, Array]:
  """Compute the Hessenberg form of the matrix

  JAX implementation of :func:`scipy.linalg.hessenberg`.

  The Hessenberg form `H` of a matrix `A` satisfies:

  .. math::

     A = Q H Q^H

  where `Q` is unitary and `H` is zero below the first subdiagonal.

  Args:
    a : array of shape ``(..., N, N)``
    calc_q: if True, calculate the ``Q`` matrix (default: False)
    overwrite_a: unused by JAX
    check_finite: unused by JAX

  Returns:
    A tuple of arrays ``(H, Q)`` if ``calc_q`` is True, else an array ``H``

    - ``H`` has shape ``(..., N, N)`` and is the Hessenberg form of ``a``
    - ``Q`` has shape ``(..., N, N)`` and is the associated unitary matrix

  Examples:
    Computing the Hessenberg form of a 4x4 matrix

    >>> a = jnp.array([[1., 2., 3., 4.],
    ...                [1., 4., 2., 3.],
    ...                [3., 2., 1., 4.],
    ...                [2., 3., 2., 2.]])
    >>> H, Q = jax.scipy.linalg.hessenberg(a, calc_q=True)
    >>> with jnp.printoptions(suppress=True, precision=3):
    ...   print(H)
    [[ 1.    -5.078  1.167  1.361]
     [-3.742  5.786 -3.613 -1.825]
     [ 0.    -2.992  2.493 -0.577]
     [ 0.     0.    -0.043 -1.279]]

    Notice the zeros in the subdiagonal positions. The original matrix
    can be reconstructed using the ``Q`` vectors:

    >>> a_reconstructed = Q @ H @ Q.conj().T
    >>> jnp.allclose(a_reconstructed, a)
    Array(True, dtype=bool)
  """
  del overwrite_a, check_finite  # unused
  n = np.shape(a)[-1]
  if n == 0:
    if calc_q:
      return jnp.zeros_like(a), jnp.zeros_like(a)
    else:
      return jnp.zeros_like(a)
  a_out, taus = lax_linalg.hessenberg(a)
  h = jnp.triu(a_out, -1)
  if calc_q:
    q = lax_linalg.householder_product(a_out[..., 1:, :-1], taus)
    batch_dims = a_out.shape[:-2]
    q = jnp.block([[jnp.ones(batch_dims + (1, 1), dtype=a_out.dtype),
                    jnp.zeros(batch_dims + (1, n - 1), dtype=a_out.dtype)],
                   [jnp.zeros(batch_dims + (n - 1, 1), dtype=a_out.dtype), q]])
    return h, q
  else:
    return h


def toeplitz(c: ArrayLike, r: ArrayLike | None = None) -> Array:
  r"""Construct a Toeplitz matrix.

  JAX implementation of :func:`scipy.linalg.toeplitz`.

  A Toeplitz matrix has equal diagonals: :math:`A_{ij} = k_{i - j}`
  for :math:`0 \le i < n` and :math:`0 \le j < n`. This function
  specifies the diagonals via the first column ``c`` and the first row
  ``r``, such that for row `i` and column `j`:

  .. math::

     A_{ij} = \begin{cases}
      c_{i - j} & i \ge j \\
      r_{j - i} & i < j
     \end{cases}

  Notice this implies that :math:`r_0` is ignored.

  Args:
    c: array of shape ``(..., N)`` specifying the first column.
    r: (optional) array of shape ``(..., M)`` specifying the first row. Leading
      dimensions must be broadcast-compatible with those of ``c``. If not specified,
      ``r`` defaults to ``conj(c)``.

  Returns:
    A Toeplitz matrix of shape ``(... N, M)``.

  Examples:
    Specifying ``c`` only:

    >>> c = jnp.array([1, 2, 3])
    >>> jax.scipy.linalg.toeplitz(c)
    Array([[1, 2, 3],
           [2, 1, 2],
           [3, 2, 1]], dtype=int32)

    Specifying ``c`` and ``r``:

    >>> r = jnp.array([-1, -2, -3])
    >>> jax.scipy.linalg.toeplitz(c, r)  # Note r[0] is ignored
    Array([[ 1, -2, -3],
           [ 2,  1, -2],
           [ 3,  2,  1]], dtype=int32)

    If specifying only complex-valued ``c``, ``r`` defaults to ``c.conj()``,
    resulting in a Hermitian matrix if ``c[0].imag == 0``:

    >>> c = jnp.array([1, 2+1j, 1+2j])
    >>> M = jax.scipy.linalg.toeplitz(c)
    >>> M
    Array([[1.+0.j, 2.-1.j, 1.-2.j],
           [2.+1.j, 1.+0.j, 2.-1.j],
           [1.+2.j, 2.+1.j, 1.+0.j]], dtype=complex64)
    >>> print("M is Hermitian:", jnp.all(M == M.conj().T))
    M is Hermitian: True

    For N-dimensional ``c`` and/or ``r``, the result is a batch of Toeplitz matrices:

    >>> c = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> jax.scipy.linalg.toeplitz(c)
    Array([[[1, 2, 3],
            [2, 1, 2],
            [3, 2, 1]],
    <BLANKLINE>
           [[4, 5, 6],
            [5, 4, 5],
            [6, 5, 4]]], dtype=int32)
  """
  if r is None:
    check_arraylike("toeplitz", c)
    r = jnp.conjugate(jnp.asarray(c))
  else:
    check_arraylike("toeplitz", c, r)
  return _toeplitz(jnp.atleast_1d(jnp.asarray(c)), jnp.atleast_1d(jnp.asarray(r)))

@partial(jnp_vectorize.vectorize, signature="(m),(n)->(m,n)")
def _toeplitz(c: Array, r: Array) -> Array:
  ncols, = c.shape
  nrows, = r.shape
  if ncols == 0 or nrows == 0:
    return jnp.empty((ncols, nrows), dtype=jnp.promote_types(c.dtype, r.dtype))
  elems = jnp.concatenate((c[::-1], r[1:]))
  i = jnp.arange(ncols - 1, -1, -1)[:, None]
  j = jnp.arange(nrows)[None, :]
  return elems[i + j]

def hankel(c: ArrayLike, r: ArrayLike | None = None) -> Array:
  r"""Construct a Hankel matrix.

  JAX implementation of :func:`scipy.linalg.hankel`.

  A Hankel matrix has constant anti-diagonals: ``H[i, j] = v[i + j]``, where
  ``v = concatenate([c, r[1:]])``. Notice this implies that ``r[0]`` is ignored.

  Args:
    c: array of shape ``(..., N)`` specifying the first column.
    r: (optional) array of shape ``(..., M)`` specifying the last row. Leading
      dimensions must be broadcast-compatible with those of ``c``. If not
      specified, ``r`` defaults to ``zeros_like(c)``.

  Returns:
    A Hankel matrix of shape ``(..., N, M)``.

  Examples:
    >>> c = jnp.array([1, 2, 3])
    >>> jax.scipy.linalg.hankel(c)
    Array([[1, 2, 3],
           [2, 3, 0],
           [3, 0, 0]], dtype=int32)

    >>> r = jnp.array([999, 4, 5, 6]) # Note r[0] is ignored
    >>> jax.scipy.linalg.hankel(c, r)
    Array([[1, 2, 3, 4],
           [2, 3, 4, 5],
           [3, 4, 5, 6]], dtype=int32)

    For N-dimensional ``c`` and/or ``r``, the result is a batch of Hankel matrices.
  """
  if r is None:
    check_arraylike("hankel", c)
    c = jnp.asarray(c)
    r = jnp.zeros_like(c)
  else:
    check_arraylike("hankel", c, r)
    c = jnp.asarray(c)
    r = jnp.asarray(r)
  if c.ndim == 0:
    raise ValueError("hankel: c must be at least 1-dimensional, got a scalar.")
  if r.ndim == 0:
    raise ValueError("hankel: r must be at least 1-dimensional, got a scalar.")

  # Align batch ranks so jnp.vectorize doesn't need implicit rank promotion.
  if c.ndim < r.ndim:
    c = lax.expand_dims(c, range(r.ndim - c.ndim))
  elif r.ndim < c.ndim:
    r = lax.expand_dims(r, range(c.ndim - r.ndim))

  return _hankel(c, r)

@partial(jnp_vectorize.vectorize, signature="(m),(n)->(m,n)")
def _hankel(c: Array, r: Array) -> Array:
  ncols, = c.shape
  nrows, = r.shape
  if ncols == 0 or nrows == 0:
    return jnp.empty((ncols, nrows), dtype=jnp.result_type(c, r))
  v = jnp.concatenate((c, r[1:]))
  i = jnp.arange(ncols)[:, None]
  j = jnp.arange(nrows)[None, :]
  return v[i + j]


def circulant(c: ArrayLike) -> Array:
  r"""Construct a circulant matrix.

  JAX implementation of :func:`scipy.linalg.circulant`.

  A circulant matrix has cyclically shifted columns: :math:`A_{ij} = c_{(i - j) \bmod n}`
  for :math:`0 \le i, j < n`, where ``c`` specifies the first column.

  Args:
    c: array of shape ``(..., N)`` specifying the first column.

  Returns:
    A circulant matrix of shape ``(..., N, N)``.

  Examples:
    >>> c = jnp.array([1, 2, 3])
    >>> jax.scipy.linalg.circulant(c)
    Array([[1, 3, 2],
           [2, 1, 3],
           [3, 2, 1]], dtype=int32)

    For N-dimensional ``c``, the result is a batch of circulant matrices:

    >>> c = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> jax.scipy.linalg.circulant(c)
    Array([[[1, 3, 2],
            [2, 1, 3],
            [3, 2, 1]],
    <BLANKLINE>
           [[4, 6, 5],
            [5, 4, 6],
            [6, 5, 4]]], dtype=int32)
  """
  check_arraylike("circulant", c)
  return _circulant(jnp.atleast_1d(jnp.asarray(c)))

@partial(jnp_vectorize.vectorize, signature="(n)->(n,n)")
def _circulant(c: Array) -> Array:
  n, = c.shape
  idx = (jnp.arange(n)[:, None] - jnp.arange(n)[None, :]) % n
  return c[idx]


def leslie(f: ArrayLike, s: ArrayLike) -> Array:
  r"""Construct a Leslie matrix.

  JAX implementation of :func:`scipy.linalg.leslie`.

  Given fecundity coefficients ``f`` of shape ``(..., N)`` and survival
  coefficients ``s`` of shape ``(..., N - 1)``, the Leslie matrix has ``f`` as
  its first row, ``s`` along its first sub-diagonal, and zeros elsewhere.

  Args:
    f: array of shape ``(..., N)`` with ``N >= 2`` containing the fecundity
      coefficients.
    s: array of shape ``(..., N - 1)`` containing the survival coefficients.

  Returns:
    A Leslie matrix of shape ``(..., N, N)``.

  Examples:
    >>> jax.scipy.linalg.leslie(jnp.array([0.1, 2.0, 1.0, 0.1]),
    ...                         jnp.array([0.2, 0.8, 0.7]))
    Array([[0.1, 2. , 1. , 0.1],
           [0.2, 0. , 0. , 0. ],
           [0. , 0.8, 0. , 0. ],
           [0. , 0. , 0.7, 0. ]], dtype=float32)
  """
  check_arraylike("leslie", f, s)
  f_arr = jnp.atleast_1d(f)
  s_arr = jnp.atleast_1d(s)
  if f_arr.shape[-1] < 2:
    raise ValueError(
        "The length of f along the last axis must be at least 2; "
        f"got shape {f_arr.shape}.")
  if s_arr.shape[-1] != f_arr.shape[-1] - 1:
    raise ValueError(
        "Incorrect lengths for f and s. The length of s along the last axis "
        f"must be one less than the length of f; got f shape {f_arr.shape} "
        f"and s shape {s_arr.shape}.")
  return _leslie(f_arr, s_arr)

@partial(jnp_vectorize.vectorize, signature="(n),(m)->(n,n)")
def _leslie(f: Array, s: Array) -> Array:
  f, s = promote_dtypes(f, s)
  return jnp.diag(s, k=-1).at[0].set(f)


def companion(a: ArrayLike) -> Array:
  r"""Construct a companion matrix.

  JAX implementation of :func:`scipy.linalg.companion`.

  Given polynomial coefficients :math:`a = [a_0, a_1, \ldots, a_{n-1}]` with
  :math:`a_0 \neq 0`, the companion matrix is the :math:`(n-1) \times (n-1)`
  matrix whose first row is :math:`-[a_1, a_2, \ldots, a_{n-1}] / a_0` and
  whose first sub-diagonal is filled with ones.

  Args:
    a: array of shape ``(..., N)`` with ``N >= 2`` specifying the polynomial
      coefficients.

  Returns:
    A companion matrix of shape ``(..., N - 1, N - 1)``.

  Note:
    Unlike :func:`scipy.linalg.companion`, this function does not check at
    runtime that ``a[..., 0]`` is non-zero; if the leading coefficient is
    zero, the result will contain ``inf`` or ``nan`` entries.

  Examples:
    >>> jax.scipy.linalg.companion(jnp.array([1., -10., 31., -30.]))
    Array([[ 10., -31.,  30.],
           [  1.,   0.,   0.],
           [  0.,   1.,   0.]], dtype=float32)
  """
  a, = promote_args_inexact("companion", a)
  a = jnp.atleast_1d(a)
  if a.shape[-1] < 2:
    raise ValueError(
        "The length of `a` along the last axis must be at least 2; "
        f"got shape {a.shape}.")
  return _companion(a)

@partial(jnp_vectorize.vectorize, signature="(n)->(m,m)")
def _companion(a: Array) -> Array:
  first_row = -a[1:] / a[0]
  m = a.shape[0] - 1
  out = jnp.eye(m, m, k=-1, dtype=first_row.dtype)
  return out.at[0].set(first_row)


def fiedler(a: ArrayLike) -> Array:
  r"""Construct a symmetric Fiedler matrix.

  JAX implementation of :func:`scipy.linalg.fiedler`.

  The Fiedler matrix has entries :math:`F_{ij} = |a_i - a_j|` for
  :math:`0 \le i, j < n`, where ``a`` is the input vector. The result is
  symmetric with a zero diagonal.

  Args:
    a: array of shape ``(..., N)``.

  Returns:
    A Fiedler matrix of shape ``(..., N, N)``.

  Examples:
    >>> jax.scipy.linalg.fiedler(jnp.array([1, 4, 12, 45, 77]))
    Array([[ 0,  3, 11, 44, 76],
           [ 3,  0,  8, 41, 73],
           [11,  8,  0, 33, 65],
           [44, 41, 33,  0, 32],
           [76, 73, 65, 32,  0]], dtype=int32)
  """
  check_arraylike("fiedler", a)
  arr = jnp.atleast_1d(a)
  return jnp.abs(arr[..., None] - arr[..., None, :])


def fiedler_companion(a: ArrayLike) -> Array:
  r"""Construct a Fiedler companion matrix.

  JAX implementation of :func:`scipy.linalg.fiedler_companion`.

  Given polynomial coefficients :math:`a = [a_0, a_1, \ldots, a_{n}]` with
  :math:`a_0 \neq 0`, this constructs a pentadiagonal matrix whose
  eigenvalues coincide with the roots of the polynomial. The result is
  similar to :func:`companion` but with a sparser, banded structure.

  Args:
    a: array of shape ``(..., N)`` specifying the polynomial coefficients in
      descending order. The last axis must have nonzero length. For ``N == 1``
      an empty ``(0, 0)`` matrix is returned along that slice.

  Raises:
    ValueError: if the last axis of ``a`` has length zero.

  Returns:
    A Fiedler companion matrix of shape ``(..., N - 1, N - 1)``.

  Note:
    Unlike :func:`scipy.linalg.fiedler_companion`, this function does not
    check at runtime that ``a[..., 0]`` is non-zero; if the leading
    coefficient is zero, the result will contain ``inf`` or ``nan`` entries.

  Examples:
    >>> a = jnp.array([1., -16., 86., -176., 105.])
    >>> jax.scipy.linalg.fiedler_companion(a)
    Array([[ 16., -86.,   1.,   0.],
           [  1.,   0.,   0.,   0.],
           [  0., 176.,   0., -105.],
           [  0.,   1.,   0.,   0.]], dtype=float32)
  """
  a, = promote_args_inexact("fiedler_companion", a)
  a = jnp.atleast_1d(a)
  if a.shape[-1] == 0:
    raise ValueError(
        "fiedler_companion requires the last axis of 'a' to have nonzero "
        f"length, but got an array of shape {a.shape}.")
  return _fiedler_companion(a)

@partial(jnp_vectorize.vectorize, signature="(n)->(m,m)")
def _fiedler_companion(a: Array) -> Array:
  n = a.shape[0] - 1
  if n == 0:
    return jnp.empty_like(a, shape=(0, 0))
  a = a / a[0]
  if n == 1:
    return -a[1:].reshape(1, 1)
  # Build the matrix with full-grid masked assignments so static shapes are
  # preserved under jit and vectorize. The pentadiagonal layout is:
  #   c[0, 0]               = -a[1]            (first column top)
  #   c[1, 0]               = 1                (first column second row)
  #   c[i,   i+1]           = -a[i+2]          (super-diag, even i, i+1 < n)
  #   c[i,   i+2]           = 1                (second super, even i, i+2 < n)
  #   c[i,   i-1]           = -a[i+1]          (sub-diag, even i >= 2, < n)
  #   c[i,   i-2]           = 1                (second sub, odd i >= 3, < n)
  i = jnp.arange(n)[:, None]
  j = jnp.arange(n)[None, :]
  # Clip indices so out-of-range lookups are safe; masks below select valid
  # cells.
  super_val = -a[jnp.minimum(i + 2, n)]
  sub_val = -a[jnp.minimum(i + 1, n)]
  i_even = (i % 2 == 0)
  i_odd = (i % 2 == 1)
  one = jnp.array(1, dtype=a.dtype)
  zero = jnp.array(0, dtype=a.dtype)
  conditions = [
      (i == 0) & (j == 0),
      (i == 1) & (j == 0),
      (j == i + 1) & i_even,
      (j == i + 2) & i_even,
      (j == i - 1) & i_even & (i >= 2),
      (j == i - 2) & i_odd & (i >= 3),
  ]
  choices = [-a[1], one, super_val, one, sub_val, one]
  return jnp.select(conditions, choices, default=zero)


@jit(static_argnames=("n", "mode"))
def convolution_matrix(a: ArrayLike, n: int, mode: str = 'full') -> Array:
  r"""Construct a convolution matrix.

  JAX implementation of :func:`scipy.linalg.convolution_matrix`.

  Builds a Toeplitz matrix :math:`A` such that ``A @ v`` equals
  ``jnp.convolve(a, v, mode)``. The returned array has ``n`` columns;
  the row count :math:`k` depends on ``mode``:

  - ``'full'``: :math:`k = m + n - 1`
  - ``'same'``: :math:`k = \max(m, n)`
  - ``'valid'``: :math:`k = \max(m, n) - \min(m, n) + 1`

  where :math:`m` is the size of ``a`` along the last axis.

  Args:
    a: array of shape ``(..., m)`` to convolve. Must have ``m >= 1``.
    n: number of columns in the output. Must be a positive integer.
    mode: one of ``'full'``, ``'same'``, ``'valid'``. Defaults to ``'full'``.

  Returns:
    A convolution matrix of shape ``(..., k, n)``, where ``k`` depends on
    ``mode`` as described above.

  See also:
    :func:`jax.scipy.linalg.toeplitz`

  Examples:
    >>> jax.scipy.linalg.convolution_matrix(jnp.array([-1, 4, -2]), 5, mode='same')
    Array([[ 4, -1,  0,  0,  0],
           [-2,  4, -1,  0,  0],
           [ 0, -2,  4, -1,  0],
           [ 0,  0, -2,  4, -1],
           [ 0,  0,  0, -2,  4]], dtype=int32)
  """
  n = operator.index(n)
  if n <= 0:
    raise ValueError(f"n must be a positive integer; got {n}.")
  check_arraylike("convolution_matrix", a)
  a_arr = jnp.asarray(a)
  if a_arr.ndim == 0:
    raise ValueError(
        "convolution_matrix: a must be at least 1-dimensional, got a scalar.")
  m = a_arr.shape[-1]
  if m < 1:
    raise ValueError(f"len(a) must be at least 1; got shape {a_arr.shape}.")
  if mode not in ('full', 'valid', 'same'):
    raise ValueError(
        f"mode must be one of 'full', 'valid', 'same'; got {mode!r}.")
  pad_widths = [(0, 0)] * (a_arr.ndim - 1) + [(0, n - 1)]
  az = jnp.pad(a_arr, pad_widths)
  raz = jnp.pad(jnp.flip(a_arr, axis=-1), pad_widths)
  L = m + n - 1
  if mode == 'same':
    trim = min(n, m) - 1
    tb = trim // 2
    te = trim - tb
  elif mode == 'valid':
    tb = min(n, m) - 1
    te = tb
  else:  # 'full'
    tb = 0
    te = 0
  col0 = lax.slice_in_dim(az, tb, L - te, axis=-1)
  row0 = lax.slice_in_dim(raz, L - n - tb, L - tb, axis=-1)
  return toeplitz(col0, row0)


@jit(static_argnames=("n",))
def hilbert(n: int) -> Array:
  r"""Create a Hilbert matrix of order n.

  JAX implementation of :func:`scipy.linalg.hilbert`.

  The Hilbert matrix is defined by:

  .. math::

     H_{ij} = \frac{1}{i + j + 1}

  for :math:`1 \le i \le n` and :math:`1 \le j \le n`.

  Args:
    n: the size of the matrix to create.

  Returns:
    A Hilbert matrix of shape ``(n, n)``

  Examples:
    >>> jax.scipy.linalg.hilbert(2)
    Array([[1.        , 0.5       ],
           [0.5       , 0.33333334]], dtype=float32)
    >>> jax.scipy.linalg.hilbert(3)
    Array([[1.        , 0.5       , 0.33333334],
           [0.5       , 0.33333334, 0.25      ],
           [0.33333334, 0.25      , 0.2       ]], dtype=float32)
  """
  a = lax.broadcasted_iota(float, (n, 1), 0)
  return 1/(a + a.T + 1)


@jit(static_argnames=("n",))
def invhilbert(n: int) -> Array:
  r"""Compute the inverse of the Hilbert matrix of order n.

  JAX implementation of :func:`scipy.linalg.invhilbert`.

  The entries are given by a closed-form expression in terms of binomial
  coefficients:

  .. math::

     H^{-1}_{ij} = (-1)^{i + j} (i + j + 1)
                   \binom{n + i}{n - j - 1}
                   \binom{n + j}{n - i - 1}
                   \binom{i + j}{i}^2

  for :math:`0 \le i, j < n`.

  Args:
    n: the size of the matrix to create.

  Returns:
    The inverse of the Hilbert matrix of shape ``(n, n)``.

  Note:
    Unlike :func:`scipy.linalg.invhilbert`, this function does not support
    ``exact=True``. The result is computed in floating point using
    :func:`~jax.scipy.special.comb`; for large ``n`` the entries quickly
    exceed the dynamic range of finite-precision floats.

  See Also:
    :func:`jax.scipy.linalg.hilbert`

  Examples:
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jax.scipy.linalg.invhilbert(2))
    ...   print(jax.scipy.linalg.invhilbert(3))
    [[ 4. -6.]
     [-6. 12.]]
    [[   9.  -36.   30.]
     [ -36.  192. -180.]
     [  30. -180.  180.]]
  """
  a = jnp.arange(n, dtype=dtypes.default_float_dtype())
  i = a[:, None]
  j = a[None, :]
  s = i + j
  return ((-1.0) ** s * (s + 1.0)
          * comb(n + i, n - j - 1)
          * comb(n + j, n - i - 1)
          * comb(s, i) ** 2)


@jit(static_argnames=("n", "kind",))
def pascal(n: int, kind: str | None = None) -> Array:
  r"""Create a Pascal matrix approximation of order n.

  JAX implementation of :func:`scipy.linalg.pascal`.

  The elements of the Pascal matrix approximate the binomial coefficients. This
  implementation is not exact as JAX does not support exact factorials.

  Args:
    n: the size of the matrix to create.
    kind: (optional) must be one of ``lower``, ``upper``, or ``symmetric`` (default).

  Returns:
    A Pascal matrix of shape ``(n, n)``

  Examples:
    >>> with jnp.printoptions(precision=3):
    ...   print(jax.scipy.linalg.pascal(3, kind="lower"))
    ...   print(jax.scipy.linalg.pascal(4, kind="upper"))
    ...   print(jax.scipy.linalg.pascal(5))
    [[1. 0. 0.]
     [1. 1. 0.]
     [1. 2. 1.]]
    [[1. 1. 1. 1.]
     [0. 1. 2. 3.]
     [0. 0. 1. 3.]
     [0. 0. 0. 1.]]
    [[ 1.  1.  1.  1.  1.]
     [ 1.  2.  3.  4.  5.]
     [ 1.  3.  6. 10. 15.]
     [ 1.  4. 10. 20. 35.]
     [ 1.  5. 15. 35. 70.]]
  """
  if kind is None:
    kind = "symmetric"

  valid_kind = ["symmetric", "lower", "upper"]

  if kind not in valid_kind:
    raise ValueError(f"Expected kind to be on of: {valid_kind}; got {kind}")

  a = jnp.arange(n, dtype=np.float32)

  L_n = _binom(a[:, None], a[None, :])

  if kind == "lower":
    return L_n

  if kind == "upper":
    return L_n.T

  return jnp.dot(L_n, L_n.T)

@jit
def _binom(n, k):
  a = lax.lgamma(n + 1.0)
  b = lax.lgamma(n - k + 1.0)
  c = lax.lgamma(k + 1.0)
  return lax.exp(a - b - c)


@jit(static_argnames=("n", "kind",))
def invpascal(n: int, kind: str | None = None) -> Array:
  r"""Compute the inverse of the Pascal matrix of order n.

  JAX implementation of :func:`scipy.linalg.invpascal`.

  The inverses of the lower and upper Pascal matrices have a simple closed
  form,

  .. math::

     L^{-1}_{ij} = (-1)^{i - j} \binom{i}{j}, \qquad U^{-1} = (L^{-1})^T,

  and the symmetric inverse satisfies :math:`P_S^{-1} = U^{-1} L^{-1}`.

  Args:
    n: the size of the matrix to create.
    kind: (optional) must be one of ``lower``, ``upper``, or ``symmetric``
      (default).

  Returns:
    The inverse of the Pascal matrix of shape ``(n, n)``.

  Note:
    Unlike :func:`scipy.linalg.invpascal`, this function does not support
    ``exact=True``. The entries are computed in floating point through
    :func:`~jax.scipy.special.gammaln`-based binomial coefficients.

  See Also:
    :func:`jax.scipy.linalg.pascal`

  Examples:
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jax.scipy.linalg.invpascal(4, kind="lower"))
    ...   print(jax.scipy.linalg.invpascal(5))
    [[ 1. -0.  0. -0.]
     [-1.  1. -0.  0.]
     [ 1. -2.  1. -0.]
     [-1.  3. -3.  1.]]
    [[  5. -10.  10.  -5.   1.]
     [-10.  30. -35.  19.  -4.]
     [ 10. -35.  46. -27.   6.]
     [ -5.  19. -27.  17.  -4.]
     [  1.  -4.   6.  -4.   1.]]
  """
  if kind is None:
    kind = "symmetric"

  valid_kind = ["symmetric", "lower", "upper"]
  if kind not in valid_kind:
    raise ValueError(f"Expected kind to be one of: {valid_kind}; got {kind}")

  a = jnp.arange(n, dtype=dtypes.default_float_dtype())
  i = a[:, None]
  j = a[None, :]
  # Lower-triangular inverse: (-1)^(i-j) * binom(i, j).
  L_inv = ((-1.0) ** (i - j)) * _binom(i, j)

  if kind == "lower":
    return L_inv
  if kind == "upper":
    return L_inv.T
  return jnp.dot(L_inv.T, L_inv)


@jit(static_argnames=("n", "full"))
def helmert(n: int, full: bool = False) -> Array:
  r"""Construct a Helmert matrix.

  JAX implementation of :func:`scipy.linalg.helmert`.

  The Helmert matrix has rows of orthonormal contrasts. For ``k = 1, ...,
  n - 1``, row ``k - 1`` is :math:`(\underbrace{1, \ldots, 1}_{k}, -k, 0,
  \ldots, 0) / \sqrt{k(k + 1)}`. If ``full`` is ``True``, a row of
  :math:`1 / \sqrt{n}` is prepended.

  Args:
    n: size of the matrix. Must be a positive integer.
    full: if ``True``, return the full ``(n, n)`` matrix; otherwise return
      only the ``(n - 1, n)`` contrast block. Defaults to ``False``.

  Returns:
    A Helmert matrix of shape ``(n - 1, n)`` if ``full`` is ``False``, or
    ``(n, n)`` if ``full`` is ``True``.

  Examples:
    >>> jax.scipy.linalg.helmert(2, full=True).round(3)
    Array([[ 0.707,  0.707],
           [ 0.707, -0.707]], dtype=float32)
  """
  if n < 1:
    raise ValueError(f"n must be a positive integer; got {n}.")
  i = jnp.arange(n - 1)[:, None]
  j = jnp.arange(n)[None, :]
  k = i + 1
  k_f = k.astype(dtypes.default_float_dtype())
  denom = jnp.sqrt(k_f * (k_f + 1))
  H = jnp.where(j <= i, 1.0 / denom,
                jnp.where(j == k, -k_f / denom, jnp.zeros_like(denom)))
  if full:
    first_row = jnp.full((1, n), 1.0 / jnp.sqrt(n), dtype=H.dtype)
    H = jnp.concatenate([first_row, H], axis=0)
  return H


@jit(static_argnames=("n", "dtype"))
def hadamard(n: int, dtype: DTypeLike = int) -> Array:
  r"""Construct an n-by-n Hadamard matrix.

  JAX implementation of :func:`scipy.linalg.hadamard`.

  For ``n`` a positive power of 2, the Hadamard matrix :math:`H_n` satisfies
  :math:`H_n H_n^T = n I`. It is defined recursively by the Sylvester
  construction: :math:`H_1 = [[1]]`, and
  :math:`H_{2m} = \begin{bmatrix} H_m & H_m \\ H_m & -H_m \end{bmatrix}`.

  Args:
    n: size of the matrix. Must be a positive power of 2.
    dtype: output dtype. Defaults to ``int``.

  Returns:
    A Hadamard matrix of shape ``(n, n)``.

  Examples:
    >>> jax.scipy.linalg.hadamard(4)
    Array([[ 1,  1,  1,  1],
           [ 1, -1,  1, -1],
           [ 1,  1, -1, -1],
           [ 1, -1, -1,  1]], dtype=int32)
  """
  if n < 1 or not math.log2(n).is_integer():
    raise ValueError(
        f"n must be a positive power of 2; got {n}.")
  lg2 = int(math.log2(n))
  H = jnp.ones((1, 1), dtype=dtype)
  for _ in range(lg2):
    H = jnp.block([[H, H], [H, -H]])
  return H


@jit(static_argnames=("n", "scale", "dtype"))
def dft(n: int, scale: str | None = None, *,
        dtype: DTypeLike | None = None) -> Array:
  r"""Construct an n-by-n discrete Fourier transform matrix.

  JAX implementation of :func:`scipy.linalg.dft`.

  The DFT matrix :math:`W_n` has entries :math:`W_{ij} = \omega^{ij}`, where
  :math:`\omega = e^{-2\pi i / n}` is the primitive n-th root of unity, for
  :math:`0 \le i, j < n`.

  Args:
    n: size of the matrix.
    scale: (optional) ``None`` (default, unscaled), ``'sqrtn'`` (scale by
      :math:`1/\sqrt{n}`, making the matrix unitary), or ``'n'`` (scale by
      :math:`1/n`).
    dtype: (optional) complex floating-point dtype for the output. Defaults to
      JAX's default complex dtype.

  Returns:
    A DFT matrix of shape ``(n, n)``.

  Examples:
    >>> jax.scipy.linalg.dft(4).round(3)
    Array([[ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j],
           [ 1.+0.j, -0.-1.j, -1.+0.j,  0.+1.j],
           [ 1.+0.j, -1.+0.j,  1.-0.j, -1.+0.j],
           [ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j]], dtype=complex64)
  """
  if scale is not None and scale not in ('sqrtn', 'n'):
    raise ValueError(
        f"scale must be None, 'sqrtn', or 'n'; got {scale!r}.")
  if dtype is None:
    dtype = dtypes.default_complex_dtype()
  else:
    dtype = dtypes.check_and_canonicalize_user_dtype(dtype, "dft")
    if not dtypes.issubdtype(dtype, np.complexfloating):
      raise ValueError(
          f"dtype must be a complex floating-point type; got {dtype}.")
  a = jnp.arange(n, dtype=dtype)
  omegas = jnp.exp(-2j * np.pi * a[:, None] * a[None, :] / n)
  if scale == 'sqrtn':
    omegas = omegas / jnp.sqrt(n)
  elif scale == 'n':
    omegas = omegas / n
  return omegas


def _solve_sylvester_triangular_scan(R: Array, S: Array, F: Array) -> Array:
  """
  Solves the Sylvester equation using Bartels-Stewart algorithm
  .. math::

    RY + YS^T = F

  where R and S are upper triangular matrices following a Schur decomposition.

  Args:
    R: Matrix of shape m x m
    S: Matrix of shape n x n
    F: Matrix of shape m x n

  Returns:
    Y: Matrix of shape m x n
  """
  R, S, F = promote_args_inexact("_solve_sylvester_triangular_scan", R, S, F)

  m, n = F.shape
  total = m * n
  # scan the matrix from bottom-right to top-left
  flat_indices = jnp.arange(total - 1, -1, -1)
  Y0 = jnp.zeros((m * n,), dtype=F.dtype)

  def scan_fn(Y_flat, idx):
    i = idx // n
    j = idx % n
    Y = Y_flat.reshape((m, n))
    rhs = F[i, j]

    # Row term: gets contributions from R and already filled in Y. mask ensures that we only get non-zero elements from R because it is upper triangular
    k_row = jnp.arange(m)
    row_mask = k_row > i
    r_row = R[i, :]
    y_col = Y[:, j]
    row_term = jnp.sum(jnp.where(row_mask, r_row * y_col, 0.0))

    # Col term: same as Row term but now uses S instead of R.
    k_col = jnp.arange(n)
    col_mask = k_col > j
    y_row = Y[i, :]
    s_col = S[:, j]
    col_term = jnp.sum(jnp.where(col_mask, y_row * s_col, 0.0))

    # Here we are solving for the current Y[i, j]
    rhs -= row_term + col_term
    val = rhs / (R[i, i] + S[j, j])

    Y_flat = Y_flat.at[i * n + j].set(val)
    return Y_flat, None

  Y_flat_final, _ = lax.scan(scan_fn, Y0, flat_indices)
  return Y_flat_final.reshape((m, n))


@jit(static_argnames=["method"])
def _solve_sylvester_2d(A: Array, B: Array, C: Array, *, method: str, tol: float) -> Array:
  m, n = C.shape[-2:]
  if method == "schur":
    R, U = schur(A, output='complex')
    S, V = schur(B.conj().T, output='complex')
    F = U.conj().T @ C.astype(R.dtype) @ V
    Y = _solve_sylvester_triangular_scan(R, S.conj().T, F)
    X = U @ Y @ V.conj().T
  elif method == "eigen":
    RA, UA = jnp.linalg.eig(A)
    RB, UB = jnp.linalg.eig(B)
    F = solve(UA, C.astype(RA.dtype) @ UB)
    W = RA[:, None] + RB[None, :]
    Y = F / W
    X = UA[:m,:m] @ Y[:m,:n] @ inv(UB)[:n,:n]
  else:
    raise ValueError(f"Unrecognized method {method}. The two valid methods are either \"schur\" or \"eigen\".")
  if not dtypes.issubdtype(C.dtype, np.complexfloating):
    X = X.real
  return lax.cond(
    jnp.any(jnp.abs(jnp.linalg.eigvals(A)[:, None] + jnp.linalg.eigvals(B)[None, :]) < tol),
    lambda: jnp.zeros_like(X) * np.nan,
    lambda: X,
  )


@jit(static_argnames=["method"])
def solve_sylvester(A: ArrayLike, B: ArrayLike, C: ArrayLike, *, method: str = "schur", tol: float = 1e-8) -> Array:
  """
  Solves the Sylvester equation
  .. math::

    AX + XB = C

  Using one of two methods.

  (1) Bartell-Stewart (schur) algorithm (default) [CPU ONLY]:

  Where A and B are first decomposed using Schur decomposition to construct and alternate sylvester equation:
  .. math::

    RY + YS^T = F

  Where R and S are in quasitriangular form when A and B are real valued and triangular when A and B are complex.

  (2) The Eigen decomposition algorithm [CPU and GPU]

  Args:
    A: array of shape ``(..., m, m)``
    B: array of shape ``(..., n, n)``
    C: array of shape ``(..., m, n)``. Batch dimensions are broadcast across
      ``A``, ``B``, and ``C``.
    method: "schur" is the default and is accurate but slow, and "eigen" is an alternative that is faster but less accurate for ill-conditioned matrices.
    tol: How close the sum of the eigenvalues from A and B can be to zero before returning matrix of NaNs

  Returns:
    Array of shape ``(..., m, n)`` representing the solution ``X``.

  Examples:
    >>> A = jax.numpy.array([[1, 2], [3, 4]])
    >>> B = jax.numpy.array([[5, 6], [7, 8]])
    >>> C = jax.numpy.array([[6, 8], [10, 12]])
    >>> X = jax.scipy.linalg.solve_sylvester(A, B, C)
    >>> print(X) # doctest: +SKIP
    [[1. 0.]
     [0.  1.]]

  Notes:
    The Bartel-Stewart algorithm is robust because a Schur decomposition always exists even for defective matrices,
    and it handles complex and ill-conditioned problems better than the eigen decomposition method.
    However, there are a couple of drawbacks. First, It is computationally more expensive than
    the eigen decomposition method because you need to perform a Schur decomposition and then scan the entire solution matrix.
    Second, it requires more system memory compared to the eigen decomposition method.

    The eigen decomposition method is the fastest method to solve a sylvester equation. However, this speed brings with it a couple of drawbacks.
    First, A and B must be diagonalizable otherwise the eigenvectors will be linearly dependent and ill-conditioned leading to accuracy issues.
    Second, when the eigenvectors are not orthogonal roundoff errors are amplified.

    Additionally, for complex types as the size of the matrix increases the accuracy of the results degrades. Float64 types are most robust to degradation.

    The tol argument allows you to specify how ill-conditioned a matrix can be and still estimate a solution.
    For matrices that are ill-conditioned we recommend using float64 instead of the default float32 dtype. The solver
    can still return good estimates for ill-conditioned matrices depending on how close to zero the sums of the eigenvalues of A and B
    are.
  """
  A, B, C = promote_dtypes_inexact(jnp.asarray(A), jnp.asarray(B), jnp.asarray(C))

  m, n = C.shape[-2:]
  if A.shape[-2:] != (m, m) or B.shape[-2:] != (n, n):
    raise ValueError(f"Incompatible shapes for Sylvester equation:\nA: {A.shape}\nB: {B.shape}\nC: {C.shape}")

  return jnp_vectorize.vectorize(
      partial(_solve_sylvester_2d, method=method, tol=tol),
      signature="(m,m),(n,n),(m,n)->(m,n)")(A, B, C)
