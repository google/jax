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


from functools import partial

import scipy.linalg
import textwrap

from jax import jit, vmap
from jax import api
from jax import lax
from jax._src.lax import linalg as lax_linalg
from jax._src.numpy.util import _wraps
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import linalg as np_linalg

_T = lambda x: jnp.swapaxes(x, -1, -2)

@partial(jit, static_argnums=(1,))
def _cholesky(a, lower):
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  l = lax_linalg.cholesky(a if lower else jnp.conj(_T(a)), symmetrize_input=False)
  return l if lower else jnp.conj(_T(l))

@_wraps(scipy.linalg.cholesky)
def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return _cholesky(a, lower)

@_wraps(scipy.linalg.cho_factor)
def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
  return (cholesky(a, lower=lower), lower)

@partial(jit, static_argnums=(2,))
def _cho_solve(c, b, lower):
  c, b = np_linalg._promote_arg_dtypes(jnp.asarray(c), jnp.asarray(b))
  lax_linalg._check_solve_shapes(c, b)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=not lower, conjugate_a=not lower)
  b = lax_linalg.triangular_solve(c, b, left_side=True, lower=lower,
                                  transpose_a=lower, conjugate_a=lower)
  return b

@_wraps(scipy.linalg.cho_solve, update_doc=False)
def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
  del overwrite_b, check_finite
  c, lower = c_and_lower
  return _cho_solve(c, b, lower)

@_wraps(scipy.linalg.svd)
def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
        check_finite=True, lapack_driver='gesdd'):
  del overwrite_a, check_finite, lapack_driver
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  return lax_linalg.svd(a, full_matrices, compute_uv)


@_wraps(scipy.linalg.det)
def det(a, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return np_linalg.det(a)


@_wraps(scipy.linalg.eigh)
def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, turbo=True, eigvals=None, type=1,
         check_finite=True):
  del overwrite_a, overwrite_b, turbo, check_finite
  if b is not None:
    raise NotImplementedError("Only the b=None case of eigh is implemented")
  if type != 1:
    raise NotImplementedError("Only the type=1 case of eigh is implemented.")
  if eigvals is not None:
    raise NotImplementedError(
        "Only the eigvals=None case of eigh is implemented.")

  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  w, v = lax_linalg.eigh(a, lower=lower)

  if eigvals_only:
    return w
  else:
    return w, v


@_wraps(scipy.linalg.inv)
def inv(a, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return np_linalg.inv(a)


@_wraps(scipy.linalg.lu_factor)
def lu_factor(a, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  lu, pivots, _ = lax_linalg.lu(a)
  return lu, pivots


@_wraps(scipy.linalg.lu_solve)
def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
  del overwrite_b, check_finite
  lu, pivots = lu_and_piv
  m, n = lu.shape[-2:]
  perm = lax_linalg.lu_pivots_to_permutation(pivots, m)
  return lax_linalg.lu_solve(lu, perm, b, trans)


@partial(jit, static_argnums=(1,))
def _lu(a, permute_l):
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  lu, pivots, permutation = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  m, n = jnp.shape(a)
  p = jnp.real(jnp.array(permutation == jnp.arange(m)[:, None], dtype=dtype))
  k = min(m, n)
  l = jnp.tril(lu, -1)[:, :k] + jnp.eye(m, k, dtype=dtype)
  u = jnp.triu(lu)[:k, :]
  if permute_l:
    return jnp.matmul(p, l), u
  else:
    return p, l, u

@_wraps(scipy.linalg.lu, update_doc=False)
def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
  del overwrite_a, check_finite
  return _lu(a, permute_l)

@partial(jit, static_argnums=(1, 2))
def _qr(a, mode, pivoting):
  if pivoting:
    raise NotImplementedError(
        "The pivoting=True case of qr is not implemented.")
  if mode in ("full", "r"):
    full_matrices = True
  elif mode == "economic":
    full_matrices = False
  else:
    raise ValueError("Unsupported QR decomposition mode '{}'".format(mode))
  a = np_linalg._promote_arg_dtypes(jnp.asarray(a))
  q, r = lax_linalg.qr(a, full_matrices)
  if mode == "r":
    return r
  return q, r

@_wraps(scipy.linalg.qr)
def qr(a, overwrite_a=False, lwork=None, mode="full", pivoting=False,
       check_finite=True):
  del overwrite_a, lwork, check_finite
  return _qr(a, mode, pivoting)


@partial(jit, static_argnums=(2, 3))
def _solve(a, b, sym_pos, lower):
  if not sym_pos:
    return np_linalg.solve(a, b)

  a, b = np_linalg._promote_arg_dtypes(jnp.asarray(a), jnp.asarray(b))
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


@_wraps(scipy.linalg.solve)
def solve(a, b, sym_pos=False, lower=False, overwrite_a=False, overwrite_b=False,
          debug=False, check_finite=True):
  del overwrite_a, overwrite_b, debug, check_finite
  return _solve(a, b, sym_pos, lower)

@partial(jit, static_argnums=(2, 3, 4))
def _solve_triangular(a, b, trans, lower, unit_diagonal):
  if trans == 0 or trans == "N":
    transpose_a, conjugate_a = False, False
  elif trans == 1 or trans == "T":
    transpose_a, conjugate_a = True, False
  elif trans == 2 or trans == "C":
    transpose_a, conjugate_a = True, True
  else:
    raise ValueError("Invalid 'trans' value {}".format(trans))

  a, b = np_linalg._promote_arg_dtypes(jnp.asarray(a), jnp.asarray(b))

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

@_wraps(scipy.linalg.solve_triangular)
def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, debug=None, check_finite=True):
  del overwrite_b, debug, check_finite
  return _solve_triangular(a, b, trans, lower, unit_diagonal)



@_wraps(scipy.linalg.tril)
def tril(m, k=0):
  return jnp.tril(m, k)


@_wraps(scipy.linalg.triu)
def triu(m, k=0):
  return jnp.triu(m, k)

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

@_wraps(scipy.linalg.expm, lax_description=_expm_description)
def expm(A, *, upper_triangular=False, max_squarings=16):
  return _expm(A, upper_triangular, max_squarings)

@partial(jit, static_argnums=(1, 2))
def _expm(A, upper_triangular, max_squarings):
  P, Q, n_squarings = _calc_P_Q(A)

  def _nan(args):
    A, *_ = args
    return jnp.full_like(A, jnp.nan)

  def _compute(args):
    A, P, Q = args
    R = _solve_P_Q(P, Q, upper_triangular)
    R = _squaring(R, n_squarings)
    return R

  R = lax.cond(n_squarings > max_squarings, _nan, _compute, (A, P, Q))
  return R

@jit
def _calc_P_Q(A):
  A = jnp.asarray(A)
  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError('expected A to be a square matrix')
  A_L1 = np_linalg.norm(A,1)
  n_squarings = 0
  if A.dtype == 'float64' or A.dtype == 'complex128':
   U3, V3 = _pade3(A)
   U5, V5 = _pade5(A)
   U7, V7 = _pade7(A)
   U9, V9 = _pade9(A)
   maxnorm = 5.371920351148152
   n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
   A = A / 2**n_squarings
   U13, V13 = _pade13(A)
   conds=jnp.array([1.495585217958292e-002, 2.539398330063230e-001,
                    9.504178996162932e-001, 2.097847961257068e+000])
   U = jnp.select((A_L1<conds), (U3, U5, U7, U9), U13)
   V = jnp.select((A_L1<conds), (V3, V5, V7, V9), V13)
  elif A.dtype == 'float32' or A.dtype == 'complex64':
    U3,V3 = _pade3(A)
    U5,V5 = _pade5(A)
    maxnorm = 3.925724783138660
    n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))
    A = A / 2**n_squarings
    U7,V7 = _pade7(A)
    conds=jnp.array([4.258730016922831e-001, 1.880152677804762e+000])
    U = jnp.select((A_L1<conds), (U3, U5), U7)
    V = jnp.select((A_L1<conds), (V3, V5), V7)
  else:
    raise TypeError("A.dtype={} is not supported.".format(A.dtype))
  P = U + V  # p_m(A) : numerator
  Q = -U + V # q_m(A) : denominator
  return P, Q, n_squarings

def _solve_P_Q(P, Q, upper_triangular=False):
  if upper_triangular:
    return solve_triangular(Q, P)
  else:
    return np_linalg.solve(Q, P)

def _precise_dot(A, B):
  return jnp.dot(A, B, precision=lax.Precision.HIGHEST)

@jit
def _squaring(R, n_squarings):
  # squaring step to undo scaling
  def _squaring_precise(x):
    return _precise_dot(x, x)

  def _identity(x):
    return x

  def _scan_f(c, i):
    return lax.cond(i < n_squarings, _squaring_precise, _identity, c), None
  res, _ = lax.scan(_scan_f, R, jnp.arange(16))

  return res

def _pade3(A):
  b = (120., 60., 12., 1.)
  ident = jnp.eye(*A.shape, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  U = _precise_dot(A, (b[3]*A2 + b[1]*ident))
  V = b[2]*A2 + b[0]*ident
  return U, V

def _pade5(A):
  b = (30240., 15120., 3360., 420., 30., 1.)
  ident = jnp.eye(*A.shape, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  U = _precise_dot(A, b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = b[4]*A4 + b[2]*A2 + b[0]*ident
  return U, V

def _pade7(A):
  b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
  ident = jnp.eye(*A.shape, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  A6 = _precise_dot(A4, A2)
  U = _precise_dot(A, b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
  return U,V

def _pade9(A):
  b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
       2162160., 110880., 3960., 90., 1.)
  ident = jnp.eye(*A.shape, dtype=A.dtype)
  A2 = _precise_dot(A, A)
  A4 = _precise_dot(A2, A2)
  A6 = _precise_dot(A4, A2)
  A8 = _precise_dot(A6, A2)
  U = _precise_dot(A, b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
  V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
  return U,V

def _pade13(A):
  b = (64764752532480000., 32382376266240000., 7771770303897600.,
       1187353796428800., 129060195264000., 10559470521600., 670442572800.,
       33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
  ident = jnp.eye(*A.shape, dtype=A.dtype)
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

@_wraps(scipy.linalg.expm_frechet, lax_description=_expm_frechet_description)
def expm_frechet(A, E, *, method=None, compute_expm=True):
  return _expm_frechet(A, E, method, compute_expm)

def _expm_frechet(A, E, method=None, compute_expm=True):
  A = jnp.asarray(A)
  E = jnp.asarray(E)
  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError('expected A to be a square matrix')
  if E.ndim != 2 or E.shape[0] != E.shape[1]:
    raise ValueError('expected E to be a square matrix')
  if A.shape != E.shape:
    raise ValueError('expected A and E to be the same shape')
  if method is None:
    method = 'SPS'
  if method == 'SPS':
    bound_fun = partial(expm, upper_triangular=False, max_squarings=16)
    expm_A, expm_frechet_AE = api.jvp(bound_fun, (A,), (E,))
  else:
    raise ValueError('only method=\'SPS\' is supported')
  if compute_expm:
    return expm_A, expm_frechet_AE
  else:
    return expm_frechet_AE


@_wraps(scipy.linalg.block_diag)
@jit
def block_diag(*arrs):
  if len(arrs) == 0:
    arrs = [jnp.zeros((1, 0))]
  arrs = jnp._promote_dtypes(*arrs)
  bad_shapes = [i for i, a in enumerate(arrs) if jnp.ndim(a) > 2]
  if bad_shapes:
    raise ValueError("Arguments to jax.scipy.linalg.block_diag must have at "
                     "most 2 dimensions, got {} at argument {}."
                     .format(arrs[bad_shapes[0]], bad_shapes[0]))
  arrs = [jnp.atleast_2d(a) for a in arrs]
  acc = arrs[0]
  dtype = lax.dtype(acc)
  for a in arrs[1:]:
    _, c = a.shape
    a = lax.pad(a, dtype.type(0), ((0, 0, 0), (acc.shape[-1], 0, 0)))
    acc = lax.pad(acc, dtype.type(0), ((0, 0, 0), (0, c, 0)))
    acc = lax.concatenate([acc, a], dimension=0)
  return acc
