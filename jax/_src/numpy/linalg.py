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

from collections.abc import Sequence
from functools import partial
import warnings

import numpy as np
import textwrap
import operator
from typing import Literal, NamedTuple, cast, overload

import jax
from jax import jit, custom_jvp
from jax import lax

from jax._src.lax import lax as lax_internal
from jax._src.lax import linalg as lax_linalg
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import reductions, ufuncs
from jax._src.numpy.util import implements, promote_dtypes_inexact, check_arraylike
from jax._src.util import canonicalize_axis
from jax._src.typing import ArrayLike, Array


class EighResult(NamedTuple):
  eigenvalues: jax.Array
  eigenvectors: jax.Array


class QRResult(NamedTuple):
  Q: jax.Array
  R: jax.Array


class SlogdetResult(NamedTuple):
  sign: jax.Array
  logabsdet: jax.Array


class SVDResult(NamedTuple):
  U: jax.Array
  S: jax.Array
  Vh: jax.Array


def _H(x: ArrayLike) -> Array:
  return ufuncs.conjugate(jnp.matrix_transpose(x))


def _symmetrize(x: Array) -> Array: return (x + _H(x)) / 2


@implements(np.linalg.cholesky)
@partial(jit, static_argnames=['upper'])
def cholesky(a: ArrayLike, *, upper: bool = False) -> Array:
  check_arraylike("jnp.linalg.cholesky", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  L = lax_linalg.cholesky(a)
  return L.mT.conj() if upper else L


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[True],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> SVDResult:
  ...


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool,
    compute_uv: Literal[True],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> SVDResult:
  ...


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[False],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array:
  ...


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool,
    compute_uv: Literal[False],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array:
  ...


@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array | SVDResult:
  ...


@implements(np.linalg.svd)
@partial(
    jit,
    static_argnames=(
        "full_matrices",
        "compute_uv",
        "hermitian",
        "subset_by_index",
    ),
)
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array | SVDResult:
  check_arraylike("jnp.linalg.svd", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  if hermitian:
    w, v = lax_linalg.eigh(a, subset_by_index=subset_by_index)
    s = lax.abs(v)
    if compute_uv:
      sign = lax.sign(v)
      idxs = lax.broadcasted_iota(np.int64, s.shape, dimension=s.ndim - 1)
      s, idxs, sign = lax.sort((s, idxs, sign), dimension=-1, num_keys=1)
      s = lax.rev(s, dimensions=[s.ndim - 1])
      idxs = lax.rev(idxs, dimensions=[s.ndim - 1])
      sign = lax.rev(sign, dimensions=[s.ndim - 1])
      u = jnp.take_along_axis(w, idxs[..., None, :], axis=-1)
      vh = _H(u * sign[..., None, :].astype(u.dtype))
      return SVDResult(u, s, vh)
    else:
      return lax.rev(lax.sort(s, dimension=-1), dimensions=[s.ndim-1])

  if compute_uv:
    u, s, vh = lax_linalg.svd(
        a,
        full_matrices=full_matrices,
        compute_uv=True,
        subset_by_index=subset_by_index,
    )
    return SVDResult(u, s, vh)
  else:
    return lax_linalg.svd(
        a,
        full_matrices=full_matrices,
        compute_uv=False,
        subset_by_index=subset_by_index,
    )


@implements(np.linalg.matrix_power)
@partial(jit, static_argnames=('n',))
def matrix_power(a: ArrayLike, n: int) -> Array:
  check_arraylike("jnp.linalg.matrix_power", a)
  arr, = promote_dtypes_inexact(jnp.asarray(a))

  if arr.ndim < 2:
    raise TypeError("{}-dimensional array given. Array must be at least "
                    "two-dimensional".format(arr.ndim))
  if arr.shape[-2] != arr.shape[-1]:
    raise TypeError("Last 2 dimensions of the array must be square")
  try:
    n = operator.index(n)
  except TypeError as err:
    raise TypeError(f"exponent must be an integer, got {n}") from err

  if n == 0:
    return jnp.broadcast_to(jnp.eye(arr.shape[-2], dtype=arr.dtype), arr.shape)
  elif n < 0:
    arr = inv(arr)
    n = abs(n)

  if n == 1:
    return arr
  elif n == 2:
    return arr @ arr
  elif n == 3:
    return (arr @ arr) @ arr

  z = result = None
  while n > 0:
    z = arr if z is None else (z @ z)  # type: ignore[operator]
    n, bit = divmod(n, 2)
    if bit:
      result = z if result is None else (result @ z)
  assert result is not None
  return result


@implements(np.linalg.matrix_rank)
@jit
def matrix_rank(M: ArrayLike, tol: ArrayLike | None = None) -> Array:
  check_arraylike("jnp.linalg.matrix_rank", M)
  M, = promote_dtypes_inexact(jnp.asarray(M))
  if M.ndim < 2:
    return (M != 0).any().astype(jnp.int32)
  S = svd(M, full_matrices=False, compute_uv=False)
  if tol is None:
    tol = S.max(-1) * np.max(M.shape[-2:]).astype(S.dtype) * jnp.finfo(S.dtype).eps
  tol = jnp.expand_dims(tol, np.ndim(tol))
  return reductions.sum(S > tol, axis=-1)


@custom_jvp
def _slogdet_lu(a: Array) -> tuple[Array, Array]:
  dtype = lax.dtype(a)
  lu, pivot, _ = lax_linalg.lu(a)
  diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
  is_zero = reductions.any(diag == jnp.array(0, dtype=dtype), axis=-1)
  iota = lax.expand_dims(jnp.arange(a.shape[-1], dtype=pivot.dtype),
                         range(pivot.ndim - 1))
  parity = reductions.count_nonzero(pivot != iota, axis=-1)
  if jnp.iscomplexobj(a):
    sign = reductions.prod(diag / ufuncs.abs(diag).astype(diag.dtype), axis=-1)
  else:
    sign = jnp.array(1, dtype=dtype)
    parity = parity + reductions.count_nonzero(diag < 0, axis=-1)
  sign = jnp.where(is_zero,
                  jnp.array(0, dtype=dtype),
                  sign * jnp.array(-2 * (parity % 2) + 1, dtype=dtype))
  logdet = jnp.where(
      is_zero, jnp.array(-jnp.inf, dtype=dtype),
      reductions.sum(ufuncs.log(ufuncs.abs(diag)).astype(dtype), axis=-1))
  return sign, ufuncs.real(logdet)

@custom_jvp
def _slogdet_qr(a: Array) -> tuple[Array, Array]:
  # Implementation of slogdet using QR decomposition. One reason we might prefer
  # QR decomposition is that it is more amenable to a fast batched
  # implementation on TPU because of the lack of row pivoting.
  if jnp.issubdtype(lax.dtype(a), jnp.complexfloating):
    raise NotImplementedError("slogdet method='qr' not implemented for complex "
                              "inputs")
  n = a.shape[-1]
  a, taus = lax_linalg.geqrf(a)
  # The determinant of a triangular matrix is the product of its diagonal
  # elements. We are working in log space, so we compute the magnitude as the
  # the trace of the log-absolute values, and we compute the sign separately.
  a_diag = jnp.diagonal(a, axis1=-2, axis2=-1)
  log_abs_det = reductions.sum(ufuncs.log(ufuncs.abs(a_diag)), axis=-1)
  sign_diag = reductions.prod(ufuncs.sign(a_diag), axis=-1)
  # The determinant of a Householder reflector is -1. So whenever we actually
  # made a reflection (tau != 0), multiply the result by -1.
  sign_taus = reductions.prod(jnp.where(taus[..., :(n-1)] != 0, -1, 1), axis=-1).astype(sign_diag.dtype)
  return sign_diag * sign_taus, log_abs_det

@implements(
    np.linalg.slogdet,
    extra_params=textwrap.dedent("""
      method: string, optional
        One of ``lu`` or ``qr``, specifying whether the determinant should be
        computed using an LU decomposition or a QR decomposition. Defaults to
        LU decomposition if ``None``.
    """))
@partial(jit, static_argnames=('method',))
def slogdet(a: ArrayLike, *, method: str | None = None) -> SlogdetResult:
  check_arraylike("jnp.linalg.slogdet", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  a_shape = jnp.shape(a)
  if len(a_shape) < 2 or a_shape[-1] != a_shape[-2]:
    msg = "Argument to slogdet() must have shape [..., n, n], got {}"
    raise ValueError(msg.format(a_shape))

  if method is None or method == "lu":
    return SlogdetResult(*_slogdet_lu(a))
  elif method == "qr":
    return SlogdetResult(*_slogdet_qr(a))
  else:
    raise ValueError(f"Unknown slogdet method '{method}'. Supported methods "
                     "are 'lu' (`None`), and 'qr'.")

def _slogdet_jvp(primals, tangents):
  x, = primals
  g, = tangents
  sign, ans = slogdet(x)
  ans_dot = jnp.trace(solve(x, g), axis1=-1, axis2=-2)
  if jnp.issubdtype(jnp._dtype(x), jnp.complexfloating):
    sign_dot = (ans_dot - ufuncs.real(ans_dot).astype(ans_dot.dtype)) * sign
    ans_dot = ufuncs.real(ans_dot)
  else:
    sign_dot = jnp.zeros_like(sign)
  return (sign, ans), (sign_dot, ans_dot)

_slogdet_lu.defjvp(_slogdet_jvp)
_slogdet_qr.defjvp(_slogdet_jvp)

def _cofactor_solve(a: ArrayLike, b: ArrayLike) -> tuple[Array, Array]:
  """Equivalent to det(a)*solve(a, b) for nonsingular mat.

  Intermediate function used for jvp and vjp of det.
  This function borrows heavily from jax.numpy.linalg.solve and
  jax.numpy.linalg.slogdet to compute the gradient of the determinant
  in a way that is well defined even for low rank matrices.

  This function handles two different cases:
  * rank(a) == n or n-1
  * rank(a) < n-1

  For rank n-1 matrices, the gradient of the determinant is a rank 1 matrix.
  Rather than computing det(a)*solve(a, b), which would return NaN, we work
  directly with the LU decomposition. If a = p @ l @ u, then
  det(a)*solve(a, b) =
  prod(diag(u)) * u^-1 @ l^-1 @ p^-1 b =
  prod(diag(u)) * triangular_solve(u, solve(p @ l, b))
  If a is rank n-1, then the lower right corner of u will be zero and the
  triangular_solve will fail.
  Let x = solve(p @ l, b) and y = det(a)*solve(a, b).
  Then y_{n}
  x_{n} / u_{nn} * prod_{i=1...n}(u_{ii}) =
  x_{n} * prod_{i=1...n-1}(u_{ii})
  So by replacing the lower-right corner of u with prod_{i=1...n-1}(u_{ii})^-1
  we can avoid the triangular_solve failing.
  To correctly compute the rest of y_{i} for i != n, we simply multiply
  x_{i} by det(a) for all i != n, which will be zero if rank(a) = n-1.

  For the second case, a check is done on the matrix to see if `solve`
  returns NaN or Inf, and gives a matrix of zeros as a result, as the
  gradient of the determinant of a matrix with rank less than n-1 is 0.
  This will still return the correct value for rank n-1 matrices, as the check
  is applied *after* the lower right corner of u has been updated.

  Args:
    a: A square matrix or batch of matrices, possibly singular.
    b: A matrix, or batch of matrices of the same dimension as a.

  Returns:
    det(a) and cofactor(a)^T*b, aka adjugate(a)*b
  """
  a, = promote_dtypes_inexact(jnp.asarray(a))
  b, = promote_dtypes_inexact(jnp.asarray(b))
  a_shape = jnp.shape(a)
  b_shape = jnp.shape(b)
  a_ndims = len(a_shape)
  if not (a_ndims >= 2 and a_shape[-1] == a_shape[-2]
    and b_shape[-2:] == a_shape[-2:]):
    msg = ("The arguments to _cofactor_solve must have shapes "
           "a=[..., m, m] and b=[..., m, m]; got a={} and b={}")
    raise ValueError(msg.format(a_shape, b_shape))
  if a_shape[-1] == 1:
    return a[..., 0, 0], b
  # lu contains u in the upper triangular matrix and l in the strict lower
  # triangular matrix.
  # The diagonal of l is set to ones without loss of generality.
  lu, pivots, permutation = lax_linalg.lu(a)
  dtype = lax.dtype(a)
  batch_dims = lax.broadcast_shapes(lu.shape[:-2], b.shape[:-2])
  x = jnp.broadcast_to(b, batch_dims + b.shape[-2:])
  lu = jnp.broadcast_to(lu, batch_dims + lu.shape[-2:])
  # Compute (partial) determinant, ignoring last diagonal of LU
  diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
  iota = lax.expand_dims(jnp.arange(a_shape[-1], dtype=pivots.dtype),
                         range(pivots.ndim - 1))
  parity = reductions.count_nonzero(pivots != iota, axis=-1)
  sign = jnp.asarray(-2 * (parity % 2) + 1, dtype=dtype)
  # partial_det[:, -1] contains the full determinant and
  # partial_det[:, -2] contains det(u) / u_{nn}.
  partial_det = reductions.cumprod(diag, axis=-1) * sign[..., None]
  lu = lu.at[..., -1, -1].set(1.0 / partial_det[..., -2])
  permutation = jnp.broadcast_to(permutation, (*batch_dims, a_shape[-1]))
  iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in (*batch_dims, 1)))
  # filter out any matrices that are not full rank
  d = jnp.ones(x.shape[:-1], x.dtype)
  d = lax_linalg.triangular_solve(lu, d, left_side=True, lower=False)
  d = reductions.any(ufuncs.logical_or(ufuncs.isnan(d), ufuncs.isinf(d)), axis=-1)
  d = jnp.tile(d[..., None, None], d.ndim*(1,) + x.shape[-2:])
  x = jnp.where(d, jnp.zeros_like(x), x)  # first filter
  x = x[iotas[:-1] + (permutation, slice(None))]
  x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=True,
                                  unit_diagonal=True)
  x = jnp.concatenate((x[..., :-1, :] * partial_det[..., -1, None, None],
                      x[..., -1:, :]), axis=-2)
  x = lax_linalg.triangular_solve(lu, x, left_side=True, lower=False)
  x = jnp.where(d, jnp.zeros_like(x), x)  # second filter

  return partial_det[..., -1], x


def _det_2x2(a: Array) -> Array:
  return (a[..., 0, 0] * a[..., 1, 1] -
           a[..., 0, 1] * a[..., 1, 0])


def _det_3x3(a: Array) -> Array:
  return (a[..., 0, 0] * a[..., 1, 1] * a[..., 2, 2] +
          a[..., 0, 1] * a[..., 1, 2] * a[..., 2, 0] +
          a[..., 0, 2] * a[..., 1, 0] * a[..., 2, 1] -
          a[..., 0, 2] * a[..., 1, 1] * a[..., 2, 0] -
          a[..., 0, 0] * a[..., 1, 2] * a[..., 2, 1] -
          a[..., 0, 1] * a[..., 1, 0] * a[..., 2, 2])


@custom_jvp
@implements(np.linalg.det)
@jit
def det(a: ArrayLike) -> Array:
  check_arraylike("jnp.linalg.det", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  a_shape = jnp.shape(a)
  if len(a_shape) >= 2 and a_shape[-1] == 2 and a_shape[-2] == 2:
    return _det_2x2(a)
  elif len(a_shape) >= 2 and a_shape[-1] == 3 and a_shape[-2] == 3:
    return _det_3x3(a)
  elif len(a_shape) >= 2 and a_shape[-1] == a_shape[-2]:
    sign, logdet = slogdet(a)
    return sign * ufuncs.exp(logdet).astype(sign.dtype)
  else:
    msg = "Argument to _det() must have shape [..., n, n], got {}"
    raise ValueError(msg.format(a_shape))


@det.defjvp
def _det_jvp(primals, tangents):
  x, = primals
  g, = tangents
  y, z = _cofactor_solve(x, g)
  return y, jnp.trace(z, axis1=-1, axis2=-2)


@implements(np.linalg.eig, lax_description="""
This differs from :func:`numpy.linalg.eig` in that the return type of
:func:`jax.numpy.linalg.eig` is always ``complex64`` for 32-bit input,
and ``complex128`` for 64-bit input.

At present, non-symmetric eigendecomposition is only implemented on the CPU
backend. However eigendecomposition for symmetric/Hermitian matrices is
implemented more widely (see :func:`jax.numpy.linalg.eigh`).
""")
def eig(a: ArrayLike) -> tuple[Array, Array]:
  check_arraylike("jnp.linalg.eig", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  w, v = lax_linalg.eig(a, compute_left_eigenvectors=False)
  return w, v


@implements(np.linalg.eigvals)
@jit
def eigvals(a: ArrayLike) -> Array:
  check_arraylike("jnp.linalg.eigvals", a)
  return lax_linalg.eig(a, compute_left_eigenvectors=False,
                        compute_right_eigenvectors=False)[0]


@implements(np.linalg.eigh)
@partial(jit, static_argnames=('UPLO', 'symmetrize_input'))
def eigh(a: ArrayLike, UPLO: str | None = None,
         symmetrize_input: bool = True) -> EighResult:
  check_arraylike("jnp.linalg.eigh", a)
  if UPLO is None or UPLO == "L":
    lower = True
  elif UPLO == "U":
    lower = False
  else:
    msg = f"UPLO must be one of None, 'L', or 'U', got {UPLO}"
    raise ValueError(msg)

  a, = promote_dtypes_inexact(jnp.asarray(a))
  v, w = lax_linalg.eigh(a, lower=lower, symmetrize_input=symmetrize_input)
  return EighResult(w, v)


@implements(np.linalg.eigvalsh)
@partial(jit, static_argnames=('UPLO',))
def eigvalsh(a: ArrayLike, UPLO: str | None = 'L') -> Array:
  check_arraylike("jnp.linalg.eigvalsh", a)
  w, _ = eigh(a, UPLO)
  return w


@partial(custom_jvp, nondiff_argnums=(1, 2))
@implements(np.linalg.pinv, lax_description=textwrap.dedent("""\
    It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
    default `rcond` is `1e-15`. Here the default is
    `10. * max(num_rows, num_cols) * jnp.finfo(dtype).eps`.
    """))
@partial(jit, static_argnames=('hermitian',))
def pinv(a: ArrayLike, rcond: ArrayLike | None = None,
         hermitian: bool = False) -> Array:
  # Uses same algorithm as
  # https://github.com/numpy/numpy/blob/v1.17.0/numpy/linalg/linalg.py#L1890-L1979
  check_arraylike("jnp.linalg.pinv", a)
  arr = jnp.asarray(a)
  m, n = arr.shape[-2:]
  if m == 0 or n == 0:
    return jnp.empty(arr.shape[:-2] + (n, m), arr.dtype)
  arr = ufuncs.conj(arr)
  if rcond is None:
    max_rows_cols = max(arr.shape[-2:])
    rcond = 10. * max_rows_cols * jnp.array(jnp.finfo(arr.dtype).eps)
  rcond = jnp.asarray(rcond)
  u, s, vh = svd(arr, full_matrices=False, hermitian=hermitian)
  # Singular values less than or equal to ``rcond * largest_singular_value``
  # are set to zero.
  rcond = lax.expand_dims(rcond[..., jnp.newaxis], range(s.ndim - rcond.ndim - 1))
  cutoff = rcond * s[..., 0:1]
  s = jnp.where(s > cutoff, s, jnp.inf).astype(u.dtype)
  res = jnp.matmul(vh.mT, ufuncs.divide(u.mT, s[..., jnp.newaxis]),
                   precision=lax.Precision.HIGHEST)
  return lax.convert_element_type(res, arr.dtype)


@pinv.defjvp
@jax.default_matmul_precision("float32")
def _pinv_jvp(rcond, hermitian, primals, tangents):
  # The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems
  # Whose Variables Separate. Author(s): G. H. Golub and V. Pereyra. SIAM
  # Journal on Numerical Analysis, Vol. 10, No. 2 (Apr., 1973), pp. 413-432.
  # (via https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Derivative)
  a, = primals  # m x n
  a_dot, = tangents
  p = pinv(a, rcond=rcond, hermitian=hermitian)  # n x m
  if hermitian:
    # svd(..., hermitian=True) symmetrizes its input, and the JVP must match.
    a = _symmetrize(a)
    a_dot = _symmetrize(a_dot)

  # TODO(phawkins): this could be simplified in the Hermitian case if we
  # supported triangular matrix multiplication.
  m, n = a.shape[-2:]
  if m >= n:
    s = (p @ _H(p)) @ _H(a_dot)  # nxm
    t = (_H(a_dot) @ _H(p)) @ p  # nxm
    p_dot = -(p @ a_dot) @ p + s - (s @ a) @ p + t - (p @ a) @ t
  else:  # m < n
    s = p @ (_H(p) @ _H(a_dot))
    t = _H(a_dot) @ (_H(p) @ p)
    p_dot = -p @ (a_dot @ p) + s - s @ (a @ p) + t - p @ (a @ t)
  return p, p_dot


@implements(np.linalg.inv)
@jit
def inv(a: ArrayLike) -> Array:
  check_arraylike("jnp.linalg.inv", a)
  arr = jnp.asarray(a)
  if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
    raise ValueError(
      f"Argument to inv must have shape [..., n, n], got {arr.shape}.")
  return solve(
    arr, lax.broadcast(jnp.eye(arr.shape[-1], dtype=arr.dtype), arr.shape[:-2]))


@implements(np.linalg.norm)
@partial(jit, static_argnames=('ord', 'axis', 'keepdims'))
def norm(x: ArrayLike, ord: int | str | None = None,
         axis: None | tuple[int, ...] | int = None,
         keepdims: bool = False) -> Array:
  check_arraylike("jnp.linalg.norm", x)
  x, = promote_dtypes_inexact(jnp.asarray(x))
  x_shape = jnp.shape(x)
  ndim = len(x_shape)

  if axis is None:
    # NumPy has an undocumented behavior that admits arbitrary rank inputs if
    # `ord` is None: https://github.com/numpy/numpy/issues/14215
    if ord is None:
      return ufuncs.sqrt(reductions.sum(ufuncs.real(x * ufuncs.conj(x)), keepdims=keepdims))
    axis = tuple(range(ndim))
  elif isinstance(axis, tuple):
    axis = tuple(canonicalize_axis(x, ndim) for x in axis)
  else:
    axis = (canonicalize_axis(axis, ndim),)

  num_axes = len(axis)
  if num_axes == 1:
    if ord is None or ord == 2:
      return ufuncs.sqrt(reductions.sum(ufuncs.real(x * ufuncs.conj(x)), axis=axis,
                                        keepdims=keepdims))
    elif ord == jnp.inf:
      return reductions.amax(ufuncs.abs(x), axis=axis, keepdims=keepdims)
    elif ord == -jnp.inf:
      return reductions.amin(ufuncs.abs(x), axis=axis, keepdims=keepdims)
    elif ord == 0:
      return reductions.sum(x != 0, dtype=jnp.finfo(lax.dtype(x)).dtype,
                            axis=axis, keepdims=keepdims)
    elif ord == 1:
      # Numpy has a special case for ord == 1 as an optimization. We don't
      # really need the optimization (XLA could do it for us), but the Numpy
      # code has slightly different type promotion semantics, so we need a
      # special case too.
      return reductions.sum(ufuncs.abs(x), axis=axis, keepdims=keepdims)
    elif isinstance(ord, str):
      msg = f"Invalid order '{ord}' for vector norm."
      if ord == "inf":
        msg += "Use 'jax.numpy.inf' instead."
      if ord == "-inf":
        msg += "Use '-jax.numpy.inf' instead."
      raise ValueError(msg)
    else:
      abs_x = ufuncs.abs(x)
      ord_arr = lax_internal._const(abs_x, ord)
      ord_inv = lax_internal._const(abs_x, 1. / ord_arr)
      out = reductions.sum(abs_x ** ord_arr, axis=axis, keepdims=keepdims)
      return ufuncs.power(out, ord_inv)

  elif num_axes == 2:
    row_axis, col_axis = cast(tuple[int, ...], axis)
    if ord is None or ord in ('f', 'fro'):
      return ufuncs.sqrt(reductions.sum(ufuncs.real(x * ufuncs.conj(x)), axis=axis,
                                        keepdims=keepdims))
    elif ord == 1:
      if not keepdims and col_axis > row_axis:
        col_axis -= 1
      return reductions.amax(reductions.sum(ufuncs.abs(x), axis=row_axis, keepdims=keepdims),
                             axis=col_axis, keepdims=keepdims)
    elif ord == -1:
      if not keepdims and col_axis > row_axis:
        col_axis -= 1
      return reductions.amin(reductions.sum(ufuncs.abs(x), axis=row_axis, keepdims=keepdims),
                             axis=col_axis, keepdims=keepdims)
    elif ord == jnp.inf:
      if not keepdims and row_axis > col_axis:
        row_axis -= 1
      return reductions.amax(reductions.sum(ufuncs.abs(x), axis=col_axis, keepdims=keepdims),
                     axis=row_axis, keepdims=keepdims)
    elif ord == -jnp.inf:
      if not keepdims and row_axis > col_axis:
        row_axis -= 1
      return reductions.amin(reductions.sum(ufuncs.abs(x), axis=col_axis, keepdims=keepdims),
                     axis=row_axis, keepdims=keepdims)
    elif ord in ('nuc', 2, -2):
      x = jnp.moveaxis(x, axis, (-2, -1))
      if ord == 2:
        reducer = reductions.amax
      elif ord == -2:
        reducer = reductions.amin
      else:
        # `sum` takes an extra dtype= argument, unlike `amax` and `amin`.
        reducer = reductions.sum  # type: ignore[assignment]
      y = reducer(svd(x, compute_uv=False), axis=-1)
      if keepdims:
        y = jnp.expand_dims(y, axis)
      return y
    else:
      raise ValueError(f"Invalid order '{ord}' for matrix norm.")
  else:
    raise ValueError(
        f"Invalid axis values ({axis}) for jnp.linalg.norm.")

@overload
def qr(a: ArrayLike, mode: Literal["r"]) -> Array: ...
@overload
def qr(a: ArrayLike, mode: str = "reduced") -> Array | QRResult: ...

@implements(np.linalg.qr)
@partial(jit, static_argnames=('mode',))
def qr(a: ArrayLike, mode: str = "reduced") -> Array | QRResult:
  check_arraylike("jnp.linalg.qr", a)
  a, = promote_dtypes_inexact(jnp.asarray(a))
  if mode == "raw":
    a, taus = lax_linalg.geqrf(a)
    return QRResult(a.mT, taus)
  if mode in ("reduced", "r", "full"):
    full_matrices = False
  elif mode == "complete":
    full_matrices = True
  else:
    raise ValueError(f"Unsupported QR decomposition mode '{mode}'")
  q, r, *_ = lax_linalg.qr(a, full_matrices=full_matrices)
  if mode == "r":
    return r
  return QRResult(q, r)


@implements(np.linalg.solve)
@jit
def solve(a: ArrayLike, b: ArrayLike) -> Array:
  check_arraylike("jnp.linalg.solve", a, b)
  a, b = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(b))

  if b.ndim == 1:
    signature = "(m,m),(m)->(m)"
  elif a.ndim == b.ndim + 1:
    # Deprecation warning added 2024-02-06
    warnings.warn("jnp.linalg.solve: batched 1D solves with b.ndim > 1 are deprecated, "
                  "and in the future will be treated as a batched 2D solve. "
                  "Use solve(a, b[..., None])[..., 0] to avoid this warning.",
                  category=FutureWarning)
    signature = "(m,m),(m)->(m)"
  else:
    signature = "(m,m),(m,n)->(m,n)"
  return jnp.vectorize(lax_linalg._solve, signature=signature)(a, b)


def _lstsq(a: ArrayLike, b: ArrayLike, rcond: float | None, *,
           numpy_resid: bool = False) -> tuple[Array, Array, Array, Array]:
  # TODO: add lstsq to lax_linalg and implement this function via those wrappers.
  # TODO: add custom jvp rule for more robust lstsq differentiation
  a, b = promote_dtypes_inexact(a, b)
  if a.shape[0] != b.shape[0]:
    raise ValueError("Leading dimensions of input arrays must match")
  b_orig_ndim = b.ndim
  if b_orig_ndim == 1:
    b = b[:, None]
  if a.ndim != 2:
    raise TypeError(
      f"{a.ndim}-dimensional array given. Array must be two-dimensional")
  if b.ndim != 2:
    raise TypeError(
      f"{b.ndim}-dimensional array given. Array must be one or two-dimensional")
  m, n = a.shape
  dtype = a.dtype
  if a.size == 0:
    s = jnp.empty(0, dtype=a.dtype)
    rank = jnp.array(0, dtype=int)
    x = jnp.empty((n, *b.shape[1:]), dtype=a.dtype)
  else:
    if rcond is None:
      rcond = jnp.finfo(dtype).eps * max(n, m)
    else:
      rcond = jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)
    u, s, vt = svd(a, full_matrices=False)
    mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]
    rank = mask.sum()
    safe_s = jnp.where(mask, s, 1).astype(a.dtype)
    s_inv = jnp.where(mask, 1 / safe_s, 0)[:, jnp.newaxis]
    uTb = jnp.matmul(u.conj().T, b, precision=lax.Precision.HIGHEST)
    x = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)
  # Numpy returns empty residuals in some cases. To allow compilation, we
  # default to returning full residuals in all cases.
  if numpy_resid and (rank < n or m <= n):
    resid = jnp.asarray([])
  else:
    b_estimate = jnp.matmul(a, x, precision=lax.Precision.HIGHEST)
    resid = norm(b - b_estimate, axis=0) ** 2
  if b_orig_ndim == 1:
    x = x.ravel()
  return x, resid, rank, s

_jit_lstsq = jit(partial(_lstsq, numpy_resid=False))

@implements(np.linalg.lstsq, lax_description=textwrap.dedent("""\
    It has two important differences:

    1. In `numpy.linalg.lstsq`, the default `rcond` is `-1`, and warns that in the future
       the default will be `None`. Here, the default rcond is `None`.
    2. In `np.linalg.lstsq` the returned residuals are empty for low-rank or over-determined
       solutions. Here, the residuals are returned in all cases, to make the function
       compatible with jit. The non-jit compatible numpy behavior can be recovered by
       passing numpy_resid=True.

    The lstsq function does not currently have a custom JVP rule, so the gradient is
    poorly behaved for some inputs, particularly for low-rank `a`.
    """))
def lstsq(a: ArrayLike, b: ArrayLike, rcond: float | None = None, *,
          numpy_resid: bool = False) -> tuple[Array, Array, Array, Array]:
  check_arraylike("jnp.linalg.lstsq", a, b)
  if numpy_resid:
    return _lstsq(a, b, rcond, numpy_resid=True)
  return _jit_lstsq(a, b, rcond)


@implements(getattr(np.linalg, "cross", None))
def cross(x1: ArrayLike, x2: ArrayLike, /, *, axis=-1):
  check_arraylike("jnp.linalg.outer", x1, x2)
  x1, x2 = jnp.asarray(x1), jnp.asarray(x2)
  if x1.shape[axis] != 3 or x2.shape[axis] != 3:
    raise ValueError(
        "Both input arrays must be (arrays of) 3-dimensional vectors, "
        f"but they have {x1.shape[axis]=} and {x2.shape[axis]=}"
    )
  return jnp.cross(x1, x2, axis=axis)


@implements(getattr(np.linalg, "outer", None))
def outer(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike("jnp.linalg.outer", x1, x2)
  x1, x2 = jnp.asarray(x1), jnp.asarray(x2)
  if x1.ndim != 1 or x2.ndim != 1:
    raise ValueError(f"Input arrays must be one-dimensional, but they are {x1.ndim=} {x2.ndim=}")
  return x1[:, None] * x2[None, :]


@implements(getattr(np.linalg, "matrix_norm", None))
def matrix_norm(x: ArrayLike, /, *, keepdims: bool = False, ord: str = 'fro') -> Array:
  """
  Computes the matrix norm of a matrix (or a stack of matrices) x.
  """
  check_arraylike('jnp.linalg.matrix_norm', x)
  return norm(x, ord=ord, keepdims=keepdims, axis=(-2, -1))


@implements(getattr(np.linalg, "matrix_transpose", None))
def matrix_transpose(x: ArrayLike, /) -> Array:
  """Transposes a matrix (or a stack of matrices) x."""
  check_arraylike('jnp.linalg.matrix_transpose', x)
  x_arr = jnp.asarray(x)
  ndim = x_arr.ndim
  if ndim < 2:
    raise ValueError(f"matrix_transpose requres at least 2 dimensions; got {ndim=}")
  return jax.lax.transpose(x_arr, (*range(ndim - 2), ndim - 1, ndim - 2))


@implements(getattr(np.linalg, "vector_norm", None))
def vector_norm(x: ArrayLike, /, *, axis: int | None = None, keepdims: bool = False,
                ord: int | str = 2) -> Array:
  """Computes the vector norm of a vector (or batch of vectors) x."""
  check_arraylike('jnp.linalg.vector_norm', x)
  if axis is None:
    result = norm(jnp.ravel(x), ord=ord)
    if keepdims:
      result = lax.expand_dims(result, range(jnp.ndim(x)))
    return result
  return norm(x, axis=axis, keepdims=keepdims, ord=ord)


@implements(getattr(np.linalg, "vecdot", None))
def vecdot(x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1) -> Array:
  return jnp.vecdot(x1, x2, axis=axis)


@implements(getattr(np.linalg, "matmul", None))
def matmul(x1: ArrayLike, x2: ArrayLike, /) -> Array:
  check_arraylike('jnp.linalg.matmul', x1, x2)
  return jnp.matmul(x1, x2)


@implements(getattr(np.linalg, "tensordot", None))
def tensordot(x1: ArrayLike, x2: ArrayLike, /, *,
              axes: int | tuple[Sequence[int], Sequence[int]] = 2) -> Array:
  check_arraylike('jnp.linalg.tensordot', x1, x2)
  return jnp.tensordot(x1, x2, axes=axes)


@implements(getattr(np.linalg, "svdvals", None))
def svdvals(x: ArrayLike, /) -> Array:
  check_arraylike('jnp.linalg.svdvals', x)
  return svd(x, compute_uv=False, hermitian=False)


@implements(getattr(np.linalg, "diagonal", None))
def diagonal(x: ArrayLike, /, *, offset: int = 0) -> Array:
  check_arraylike('jnp.linalg.diagonal', x)
  return jnp.diagonal(x, offset=offset, axis1=-2, axis2=-1)
