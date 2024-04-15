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

import numpy as np
import scipy.linalg
import textwrap
from typing import overload, Any, Literal

import jax
import jax.numpy as jnp
from jax import jit, vmap, jvp
from jax import lax
from jax._src import dtypes
from jax._src.lax import linalg as lax_linalg
from jax._src.lax import qdwh
from jax._src.numpy.util import (
    check_arraylike, implements, promote_dtypes, promote_dtypes_inexact,
    promote_dtypes_complex)
from jax._src.typing import Array, ArrayLike


_no_chkfinite_doc = textwrap.dedent("""
Does not support the Scipy argument ``check_finite=True``,
because compiled JAX code cannot perform checks of array values at runtime.
""")
_no_overwrite_and_chkfinite_doc = _no_chkfinite_doc + "\nDoes not support the Scipy argument ``overwrite_*=True``."

@partial(jit, static_argnames=('lower',))
def _cholesky(a: ArrayLike, lower: bool) -> Array:
  a, = promote_dtypes_inexact(jnp.asarray(a))
  l = lax_linalg.cholesky(a if lower else jnp.conj(a.mT), symmetrize_input=False)
  return l if lower else jnp.conj(l.mT)

@implements(scipy.linalg.cholesky,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_a', 'check_finite'))
def cholesky(a: ArrayLike, lower: bool = False, overwrite_a: bool = False,
             check_finite: bool = True) -> Array:
  del overwrite_a, check_finite  # Unused
  return _cholesky(a, lower)

@implements(scipy.linalg.cho_factor,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_a', 'check_finite'))
def cho_factor(a: ArrayLike, lower: bool = False, overwrite_a: bool = False,
               check_finite: bool = True) -> tuple[Array, bool]:
  del overwrite_a, check_finite  # Unused
  return (cholesky(a, lower=lower), lower)

@partial(jit, static_argnames=('lower',))
def _cho_solve(c: ArrayLike, b: ArrayLike, lower: bool) -> Array:
  c, b = promote_dtypes_inexact(jnp.asarray(c), jnp.asarray(b))
  lax_linalg._check_solve_shapes(c, b)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=not lower, conjugate_a=not lower)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=lower, conjugate_a=lower)
  return b

@implements(scipy.linalg.cho_solve, update_doc=False,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_b', 'check_finite'))
def cho_solve(c_and_lower: tuple[ArrayLike, bool], b: ArrayLike,
              overwrite_b: bool = False, check_finite: bool = True) -> Array:
  del overwrite_b, check_finite  # Unused
  c, lower = c_and_lower
  return _cho_solve(c, b, lower)

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: Literal[True]) -> tuple[Array, Array, Array]: ...

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: Literal[False]) -> Array: ...

@overload
def _svd(x: ArrayLike, *, full_matrices: bool, compute_uv: bool) -> Array | tuple[Array, Array, Array]: ...

@partial(jit, static_argnames=('full_matrices', 'compute_uv'))
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

@implements(scipy.linalg.svd,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_a', 'check_finite', 'lapack_driver'))
def svd(a: ArrayLike, full_matrices: bool = True, compute_uv: bool = True,
        overwrite_a: bool = False, check_finite: bool = True,
        lapack_driver: str = 'gesdd') -> Array | tuple[Array, Array, Array]:
  del overwrite_a, check_finite, lapack_driver  # unused
  return _svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

@implements(scipy.linalg.det,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_a', 'check_finite'))
def det(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> Array:
  del overwrite_a, check_finite  # unused
  return jnp.linalg.det(a)


@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: Literal[True],
          eigvals: None, type: int) -> Array: ...

@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: Literal[False],
          eigvals: None, type: int) -> tuple[Array, Array]: ...

@overload
def _eigh(a: ArrayLike, b: ArrayLike | None, lower: bool, eigvals_only: bool,
          eigvals: None, type: int) -> Array | tuple[Array, Array]: ...

@partial(jit, static_argnames=('lower', 'eigvals_only', 'eigvals', 'type'))
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

@implements(scipy.linalg.eigh,
        lax_description=_no_overwrite_and_chkfinite_doc,
        skip_params=('overwrite_a', 'overwrite_b', 'turbo', 'check_finite'))
def eigh(a: ArrayLike, b: ArrayLike | None = None, lower: bool = True,
         eigvals_only: bool = False, overwrite_a: bool = False,
         overwrite_b: bool = False, turbo: bool = True, eigvals: None = None,
         type: int = 1, check_finite: bool = True) -> Array | tuple[Array, Array]:
  del overwrite_a, overwrite_b, turbo, check_finite  # unused
  return _eigh(a, b, lower, eigvals_only, eigvals, type)

@partial(jit, static_argnames=('output',))
def _schur(a: Array, output: str) -> tuple[Array, Array]:
  if output == "complex":
    a = a.astype(dtypes.to_complex_dtype(a.dtype))
  return lax_linalg.schur(a)

@implements(scipy.linalg.schur)
def schur(a: ArrayLike, output: str = 'real') -> tuple[Array, Array]:
  if output not in ('real', 'complex'):
    raise ValueError(
      f"Expected 'output' to be either 'real' or 'complex', got {output=}.")
  return _schur(a, output)

@implements(scipy.linalg.inv,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_a', 'check_finite'))
def inv(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> Array:
  del overwrite_a, check_finite  # unused
  return jnp.linalg.inv(a)


@implements(scipy.linalg.lu_factor,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_a', 'check_finite'))
@partial(jit, static_argnames=('overwrite_a', 'check_finite'))
def lu_factor(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> tuple[Array, Array]:
  del overwrite_a, check_finite  # unused
  a, = promote_dtypes_inexact(jnp.asarray(a))
  lu, pivots, _ = lax_linalg.lu(a)
  return lu, pivots


@implements(scipy.linalg.lu_solve,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_b', 'check_finite'))
@partial(jit, static_argnames=('trans', 'overwrite_b', 'check_finite'))
def lu_solve(lu_and_piv: tuple[Array, ArrayLike], b: ArrayLike, trans: int = 0,
             overwrite_b: bool = False, check_finite: bool = True) -> Array:
  del overwrite_b, check_finite  # unused
  lu, pivots = lu_and_piv
  m, _ = lu.shape[-2:]
  perm = lax_linalg.lu_pivots_to_permutation(pivots, m)
  return lax_linalg.lu_solve(lu, perm, b, trans)

@overload
def _lu(a: ArrayLike, permute_l: Literal[True]) -> tuple[Array, Array]: ...

@overload
def _lu(a: ArrayLike, permute_l: Literal[False]) -> tuple[Array, Array, Array]: ...

@overload
def _lu(a: ArrayLike, permute_l: bool) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...

@partial(jit, static_argnums=(1,))
def _lu(a: ArrayLike, permute_l: bool) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  a, = promote_dtypes_inexact(jnp.asarray(a))
  lu, _, permutation = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  m, n = jnp.shape(a)
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

@implements(scipy.linalg.lu, update_doc=False,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_a', 'check_finite'))
@partial(jit, static_argnames=('permute_l', 'overwrite_a', 'check_finite'))
def lu(a: ArrayLike, permute_l: bool = False, overwrite_a: bool = False,
       check_finite: bool = True) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  del overwrite_a, check_finite  # unused
  return _lu(a, permute_l)

@overload
def _qr(a: ArrayLike, mode: Literal["r"], pivoting: bool) -> tuple[Array] | tuple[Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: Literal["full", "economic"], pivoting: bool) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...

@overload
def _qr(a: ArrayLike, mode: str, pivoting: bool) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]: ...

@partial(jit, static_argnames=('mode', 'pivoting'))
def _qr(a: ArrayLike, mode: str, pivoting: bool) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]:
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
def qr(a: ArrayLike, overwrite_a: bool = False, lwork: Any = None, mode: Literal["full", "economic"] = "full",
       pivoting: bool = False, check_finite: bool = True) -> tuple[Array, Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool, lwork: Any, mode: Literal["r"],
       pivoting: bool = False, check_finite: bool = True) -> tuple[Array]: ...

@overload
def qr(a: ArrayLike,  overwrite_a: bool = False, lwork: Any = None, *, mode: Literal["r"],
       pivoting: bool = False, check_finite: bool = True) -> tuple[Array]: ...

@overload
def qr(a: ArrayLike, overwrite_a: bool = False, lwork: Any = None, mode: str = "full",
       pivoting: bool = False, check_finite: bool = True) -> tuple[Array] | tuple[Array, Array]: ...

@implements(scipy.linalg.qr,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_a', 'check_finite', 'lwork'))
def qr(a: ArrayLike, overwrite_a: bool = False, lwork: Any = None, mode: str = "full",
       pivoting: bool = False, check_finite: bool = True) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]:
  del overwrite_a, lwork, check_finite  # unused
  return _qr(a, mode, pivoting)


@partial(jit, static_argnames=('assume_a', 'lower'))
def _solve(a: ArrayLike, b: ArrayLike, assume_a: str, lower: bool) -> Array:
  if assume_a != 'pos':
    return jnp.linalg.solve(a, b)

  a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))
  lax_linalg._check_solve_shapes(a, b)

  # With custom_linear_solve, we can reuse the same factorization when
  # computing sensitivities. This is considerably faster.
  factors = cho_factor(lax.stop_gradient(a), lower=lower)
  custom_solve = partial(
      lax.custom_linear_solve,
      lambda x: lax_linalg._matvec_multiply(a, x),
      solve=lambda _, x: cho_solve(factors, x),
      symmetric=True)
  if a.ndim == b.ndim + 1:
    # b.shape == [..., m]
    return custom_solve(b)
  else:
    # b.shape == [..., m, k]
    return vmap(custom_solve, b.ndim - 1, max(a.ndim, b.ndim) - 1)(b)


@implements(scipy.linalg.solve,
        lax_description=_no_overwrite_and_chkfinite_doc,
        skip_params=('overwrite_a', 'overwrite_b', 'debug', 'check_finite'))
def solve(a: ArrayLike, b: ArrayLike, lower: bool = False,
          overwrite_a: bool = False, overwrite_b: bool = False, debug: bool = False,
          check_finite: bool = True, assume_a: str = 'gen') -> Array:
  del overwrite_a, overwrite_b, debug, check_finite  #unused
  valid_assume_a = ['gen', 'sym', 'her', 'pos']
  if assume_a not in valid_assume_a:
    raise ValueError(f"Expected assume_a to be one of {valid_assume_a}; got {assume_a!r}")
  return _solve(a, b, assume_a, lower)

@partial(jit, static_argnames=('trans', 'lower', 'unit_diagonal'))
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

  # lax_linalg.triangular_solve only supports matrix 'b's at the moment.
  b_is_vector = jnp.ndim(a) == jnp.ndim(b) + 1
  if b_is_vector:
    b = b[..., None]
  out = lax_linalg.triangular_solve(a, b, left_side=True, lower=lower,
                                    transpose_a=transpose_a,
                                    conjugate_a=conjugate_a,
                                    unit_diagonal=unit_diagonal)
  if b_is_vector:
    return out[..., 0]
  else:
    return out

@implements(scipy.linalg.solve_triangular,
        lax_description=_no_overwrite_and_chkfinite_doc, skip_params=('overwrite_b', 'debug', 'check_finite'))
def solve_triangular(a: ArrayLike, b: ArrayLike, trans: int | str = 0, lower: bool = False,
                     unit_diagonal: bool = False, overwrite_b: bool = False,
                     debug: Any = None, check_finite: bool = True) -> Array:
  del overwrite_b, debug, check_finite  # unused
  return _solve_triangular(a, b, trans, lower, unit_diagonal)

_expm_description = textwrap.dedent("""
In addition to the original NumPy argument(s) listed below,
also supports the optional boolean argument ``upper_triangular``
to specify whether the ``A`` matrix is upper triangular, and the optional
argument ``max_squarings`` to specify the max number of squarings allowed
in the scaling-and-squaring approximation method. Return nan if the actual
number of squarings required is more than ``max_squarings``.

The number of required squarings = max(0, ceil(log2(norm(A)) - c)
where norm() denotes the L1 norm, and

- c=2.42 for float64 or complex128,
- c=1.97 for float32 or complex64
""")

@implements(scipy.linalg.expm, lax_description=_expm_description)
@partial(jit, static_argnames=('upper_triangular', 'max_squarings'))
def expm(A: ArrayLike, *, upper_triangular: bool = False, max_squarings: int = 16) -> Array:
  A, = promote_dtypes_inexact(A)

  if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
    raise ValueError(f"Expected A to be a (batched) square matrix, got {A.shape=}.")

  if A.ndim > 2:
    return jnp.vectorize(
      partial(expm, upper_triangular=upper_triangular, max_squarings=max_squarings),
      signature="(n,n)->(n,n)")(A)

  P, Q, n_squarings = _calc_P_Q(jnp.asarray(A))

  def _nan(args):
    A, *_ = args
    return jnp.full_like(A, jnp.nan)

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
  A_L1 = jnp.linalg.norm(A,1)
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
    return jnp.linalg.solve(Q, P)

def _precise_dot(A: ArrayLike, B: ArrayLike) -> Array:
  return jnp.dot(A, B, precision=lax.Precision.HIGHEST)

@partial(jit, static_argnums=2)
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


_expm_frechet_description = textwrap.dedent("""
Does not currently support the Scipy argument ``jax.numpy.asarray_chkfinite``,
because `jax.numpy.asarray_chkfinite` does not exist at the moment. Does not
support the ``method='blockEnlarge'`` argument.
""")

@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: Literal[True] = True) -> tuple[Array, Array]: ...

@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: Literal[False]) -> Array: ...

@overload
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: bool = True) -> Array | tuple[Array, Array]: ...

@implements(scipy.linalg.expm_frechet, lax_description=_expm_frechet_description)
@partial(jit, static_argnames=('method', 'compute_expm'))
def expm_frechet(A: ArrayLike, E: ArrayLike, *, method: str | None = None,
                 compute_expm: bool = True) -> Array | tuple[Array, Array]:
  A_arr = jnp.asarray(A)
  E_arr = jnp.asarray(E)
  if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
    raise ValueError('expected A to be a square matrix')
  if E_arr.ndim != 2 or E_arr.shape[0] != E_arr.shape[1]:
    raise ValueError('expected E to be a square matrix')
  if A_arr.shape != E_arr.shape:
    raise ValueError('expected A and E to be the same shape')
  if method is None:
    method = 'SPS'
  if method == 'SPS':
    bound_fun = partial(expm, upper_triangular=False, max_squarings=16)
    expm_A, expm_frechet_AE = jvp(bound_fun, (A_arr,), (E_arr,))
  else:
    raise ValueError('only method=\'SPS\' is supported')
  if compute_expm:
    return expm_A, expm_frechet_AE
  else:
    return expm_frechet_AE


@implements(scipy.linalg.block_diag)
@jit
def block_diag(*arrs: ArrayLike) -> Array:
  if len(arrs) == 0:
    arrs =  (jnp.zeros((1, 0)),)
  arrs = tuple(promote_dtypes(*arrs))
  bad_shapes = [i for i, a in enumerate(arrs) if jnp.ndim(a) > 2]
  if bad_shapes:
    raise ValueError("Arguments to jax.scipy.linalg.block_diag must have at "
                     "most 2 dimensions, got {} at argument {}."
                     .format(arrs[bad_shapes[0]], bad_shapes[0]))
  converted_arrs = [jnp.atleast_2d(a) for a in arrs]
  acc = converted_arrs[0]
  dtype = lax.dtype(acc)
  for a in converted_arrs[1:]:
    _, c = a.shape
    a = lax.pad(a, dtype.type(0), ((0, 0, 0), (acc.shape[-1], 0, 0)))
    acc = lax.pad(acc, dtype.type(0), ((0, 0, 0), (0, c, 0)))
    acc = lax.concatenate([acc, a], dimension=0)
  return acc


@implements(scipy.linalg.eigh_tridiagonal)
@partial(jit, static_argnames=("eigvals_only", "select", "select_range"))
def eigh_tridiagonal(d: ArrayLike, e: ArrayLike, *, eigvals_only: bool = False,
                     select: str = 'a', select_range: tuple[float, float] | None = None,
                     tol: float | None = None) -> Array:
  if not eigvals_only:
    raise NotImplementedError("Calculation of eigenvectors is not implemented")

  def _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, x):
    """Implements the Sturm sequence recurrence."""
    n = alpha.shape[0]
    zeros = jnp.zeros(x.shape, dtype=jnp.int32)
    ones = jnp.ones(x.shape, dtype=jnp.int32)

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
  supported_dtypes = (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
  if alpha.dtype != beta.dtype:
    raise TypeError("diagonal and off-diagonal values must have same dtype, "
                    f"got {alpha.dtype} and {beta.dtype}")
  if alpha.dtype not in supported_dtypes or beta.dtype not in supported_dtypes:
    raise TypeError("Only float32 and float64 inputs are supported as inputs "
                    "to jax.scipy.linalg.eigh_tridiagonal, got "
                    f"{alpha.dtype} and {beta.dtype}")
  n = alpha.shape[0]
  if n <= 1:
    return jnp.real(alpha)

  if jnp.issubdtype(alpha.dtype, jnp.complexfloating):
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
    target_counts = jnp.arange(n, dtype=jnp.int32)
  elif select == 'i':
    if select_range is None:
      raise ValueError("for select='i', select_range must be specified.")
    if select_range[0] > select_range[1]:
      raise ValueError('Got empty index range in select_range.')
    target_counts = jnp.arange(select_range[0], select_range[1] + 1, dtype=jnp.int32)
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
  target_shape = jnp.shape(target_counts)
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

  _, _, mid, _ = lax.while_loop(cond, body, (0, lower, mid, upper))
  return mid

@partial(jit, static_argnames=('side', 'method'))
@jax.default_matmul_precision("float32")
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
    a: The :math:`m \times n` input matrix.
    side: Determines whether a right or left polar decomposition is computed.
      If ``side`` is ``"right"`` then :math:`a = up`. If ``side`` is ``"left"``
      then :math:`a = pu`. The default is ``"right"``.
    method: Determines the algorithm used, as described above.
    precision: :class:`~jax.lax.Precision` object specifying the matmul precision.
    eps: The final result will satisfy
      :math:`\left|x_k - x_{k-1}\right| < \left|x_k\right| (4\epsilon)^{\frac{1}{3}}`,
      where :math:`x_k` are the QDWH iterates. Ignored if ``method`` is not
      ``"qdwh"``.
    max_iterations: Iterations will terminate after this many steps even if the
      above is unsatisfied.  Ignored if ``method`` is not ``"qdwh"``.

  Returns:
    A ``(unitary, posdef)`` tuple, where ``unitary`` is the unitary factor
    (:math:`m \times n`), and ``posdef`` is the positive-semidefinite factor.
    ``posdef`` is either :math:`n \times n` or :math:`m \times m` depending on
    whether ``side`` is ``"right"`` or ``"left"``, respectively.

  .. _QDWH: https://epubs.siam.org/doi/abs/10.1137/090774999
  """
  arr = jnp.asarray(a)
  if arr.ndim != 2:
    raise ValueError("The input `a` must be a 2-D array.")

  if side not in ["right", "left"]:
    raise ValueError("The argument `side` must be either 'right' or 'left'.")

  m, n = arr.shape
  if method == "qdwh":
    # TODO(phawkins): return info also if the user opts in?
    if m >= n and side == "right":
      unitary, posdef, _, _ = qdwh.qdwh(arr, is_hermitian=False, eps=eps)
    elif m < n and side == "left":
      arr = arr.T.conj()
      unitary, posdef, _, _ = qdwh.qdwh(arr, is_hermitian=False, eps=eps)
      posdef = posdef.T.conj()
      unitary = unitary.T.conj()
    else:
      raise NotImplementedError("method='qdwh' only supports mxn matrices "
                                "where m < n where side='right' and m >= n "
                                f"side='left', got {arr.shape} with {side=}")
  elif method == "svd":
    u_svd, s_svd, vh_svd = lax_linalg.svd(arr, full_matrices=False)
    s_svd = s_svd.astype(u_svd.dtype)
    unitary = u_svd @ vh_svd
    if side == "right":
      # a = u * p
      posdef = (vh_svd.T.conj() * s_svd[None, :]) @ vh_svd
    else:
      # a = p * u
      posdef = (u_svd * s_svd[None, :]) @ (u_svd.T.conj())
  else:
    raise ValueError(f"Unknown polar decomposition method {method}.")

  return unitary, posdef


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
def _sqrtm(A: ArrayLike) -> Array:
  T, Z = schur(A, output='complex')
  sqrt_T = _sqrtm_triu(T)
  return jnp.matmul(jnp.matmul(Z, sqrt_T, precision=lax.Precision.HIGHEST),
                    jnp.conj(Z.T), precision=lax.Precision.HIGHEST)

@implements(scipy.linalg.sqrtm,
        lax_description="""
This differs from ``scipy.linalg.sqrtm`` in that the return type of
``jax.scipy.linalg.sqrtm`` is always ``complex64`` for 32-bit input,
and ``complex128`` for 64-bit input.

This function implements the complex Schur method described in [A]. It does not use recursive blocking
to speed up computations as a Sylvester Equation solver is not available yet in JAX.

[A] Björck, Å., & Hammarling, S. (1983).
    "A Schur method for the square root of a matrix". Linear algebra and its applications, 52, 127-140.
""")
def sqrtm(A: ArrayLike, blocksize: int = 1) -> Array:
  if blocksize > 1:
      raise NotImplementedError("Blocked version is not implemented yet.")
  return _sqrtm(A)

@implements(scipy.linalg.rsf2csf, lax_description=_no_chkfinite_doc)
@partial(jit, static_argnames=('check_finite',))
def rsf2csf(T: ArrayLike, Z: ArrayLike, check_finite: bool = True) -> tuple[Array, Array]:
  del check_finite  # unused

  T_arr = jnp.asarray(T)
  Z_arr = jnp.asarray(Z)

  if T_arr.ndim != 2 or T_arr.shape[0] != T_arr.shape[1]:
    raise ValueError("Input 'T' must be square.")
  if Z_arr.ndim != 2 or Z_arr.shape[0] != Z_arr.shape[1]:
    raise ValueError("Input 'Z' must be square.")
  if T_arr.shape[0] != Z_arr.shape[0]:
    raise ValueError(f"Input array shapes must match: Z: {Z_arr.shape} vs. T: {T_arr.shape}")

  T_arr, Z_arr = promote_dtypes_complex(T_arr, Z_arr)
  eps = jnp.finfo(T_arr.dtype).eps
  N = T_arr.shape[0]

  if N == 1:
    return T_arr, Z_arr

  def _update_T_Z(m, T, Z):
    mu = jnp.linalg.eigvals(lax.dynamic_slice(T, (m-1, m-1), (2, 2))) - T[m, m]
    r = jnp.linalg.norm(jnp.array([mu[0], T[m, m-1]])).astype(T.dtype)
    c = mu[0] / r
    s = T[m, m-1] / r
    G = jnp.array([[c.conj(), s], [-s, c]], dtype=T.dtype)

    # T[m-1:m+1, m-1:] = G @ T[m-1:m+1, m-1:]
    T_rows = lax.dynamic_slice_in_dim(T, m-1, 2, axis=0)
    col_mask = jnp.arange(N) >= m-1
    G_dot_T_zeroed_cols = G @ jnp.where(col_mask, T_rows, 0)
    T_rows_new = jnp.where(~col_mask, T_rows, G_dot_T_zeroed_cols)
    T = lax.dynamic_update_slice_in_dim(T, T_rows_new, m-1, axis=0)

    # T[:m+1, m-1:m+1] = T[:m+1, m-1:m+1] @ G.conj().T
    T_cols = lax.dynamic_slice_in_dim(T, m-1, 2, axis=1)
    row_mask = jnp.arange(N)[:, jnp.newaxis] < m+1
    T_zeroed_rows_dot_GH = jnp.where(row_mask, T_cols, 0) @ G.conj().T
    T_cols_new = jnp.where(~row_mask, T_cols, T_zeroed_rows_dot_GH)
    T = lax.dynamic_update_slice_in_dim(T, T_cols_new, m-1, axis=1)

    # Z[:, m-1:m+1] = Z[:, m-1:m+1] @ G.conj().T
    Z_cols = lax.dynamic_slice_in_dim(Z, m-1, 2, axis=1)
    Z = lax.dynamic_update_slice_in_dim(Z, Z_cols @ G.conj().T, m-1, axis=1)
    return T, Z

  def _rsf2scf_iter(i, TZ):
    m = N-i
    T, Z = TZ
    T, Z = lax.cond(
      jnp.abs(T[m, m-1]) > eps*(jnp.abs(T[m-1, m-1]) + jnp.abs(T[m, m])),
      _update_T_Z,
      lambda m, T, Z: (T, Z),
      m, T, Z)
    T = T.at[m, m-1].set(0.0)
    return T, Z

  return lax.fori_loop(1, N, _rsf2scf_iter, (T_arr, Z_arr))

@overload
def hessenberg(a: ArrayLike, *, calc_q: Literal[False], overwrite_a: bool = False,
               check_finite: bool = True) -> Array: ...

@overload
def hessenberg(a: ArrayLike, *, calc_q: Literal[True], overwrite_a: bool = False,
               check_finite: bool = True) -> tuple[Array, Array]: ...

@implements(scipy.linalg.hessenberg, lax_description=_no_overwrite_and_chkfinite_doc)
@partial(jit, static_argnames=('calc_q', 'check_finite', 'overwrite_a'))
def hessenberg(a: ArrayLike, *, calc_q: bool = False, overwrite_a: bool = False,
               check_finite: bool = True) -> Array | tuple[Array, Array]:
  del overwrite_a, check_finite
  n = jnp.shape(a)[-1]
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

@implements(scipy.linalg.toeplitz)
def toeplitz(c: ArrayLike, r: ArrayLike | None = None) -> Array:
  if r is None:
    check_arraylike("toeplitz", c)
    r = jnp.conjugate(jnp.asarray(c))
  else:
    check_arraylike("toeplitz", c, r)

  c_arr = jnp.asarray(c).flatten()
  r_arr = jnp.asarray(r).flatten()

  ncols, = c_arr.shape
  nrows, = r_arr.shape

  if ncols == 0 or nrows == 0:
    return jnp.empty((ncols, nrows),
                     dtype=jnp.promote_types(c_arr.dtype, r_arr.dtype))

  nelems = ncols + nrows - 1
  elems = jnp.concatenate((c_arr[::-1], r_arr[1:]))
  patches = lax.conv_general_dilated_patches(
      elems.reshape((1, nelems, 1)),
      (nrows,), (1,), 'VALID', dimension_numbers=('NTC', 'IOT', 'NTC'),
      precision=lax.Precision.HIGHEST)[0]
  return jnp.flip(patches, axis=0)

@implements(scipy.linalg.hilbert)
@partial(jit, static_argnames=("n",))
def hilbert(n: int) -> Array:
  a = lax.broadcasted_iota(jnp.float64, (n, 1), 0)
  return 1/(a + a.T + 1)
