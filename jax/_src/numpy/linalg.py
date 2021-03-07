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

import numpy as np
import textwrap
import operator
from typing import Tuple, Union, cast

from jax import jit, custom_jvp
from jax import lax
from jax import ops
from jax._src.lax import linalg as lax_linalg
from jax import dtypes
from .util import _wraps
from . import lax_numpy as jnp
from jax._src.util import canonicalize_axis

_T = lambda x: jnp.swapaxes(x, -1, -2)
_H = lambda x: jnp.conjugate(jnp.swapaxes(x, -1, -2))


def _promote_arg_dtypes(*args):
  """Promotes `args` to a common inexact type."""
  def _to_inexact_type(type):
    return type if jnp.issubdtype(type, jnp.inexact) else jnp.float_
  inexact_types = [_to_inexact_type(jnp._dtype(arg)) for arg in args]
  dtype = dtypes.canonicalize_dtype(jnp.result_type(*inexact_types))
  args = [lax.convert_element_type(arg, dtype) for arg in args]
  if len(args) == 1:
    return args[0]
  else:
    return args


@_wraps(np.linalg.cholesky)
def cholesky(a):
  a = _promote_arg_dtypes(jnp.asarray(a))
  return lax_linalg.cholesky(a)


@_wraps(np.linalg.svd)
def svd(a, full_matrices=True, compute_uv=True):
  a = _promote_arg_dtypes(jnp.asarray(a))
  return lax_linalg.svd(a, full_matrices, compute_uv)


@_wraps(np.linalg.matrix_power)
def matrix_power(a, n):
  a = _promote_arg_dtypes(jnp.asarray(a))

  if a.ndim < 2:
    raise TypeError("{}-dimensional array given. Array must be at least "
                    "two-dimensional".format(a.ndim))
  if a.shape[-2] != a.shape[-1]:
    raise TypeError("Last 2 dimensions of the array must be square")
  try:
    n = operator.index(n)
  except TypeError as err:
    raise TypeError("exponent must be an integer, got {}".format(n)) from err

  if n == 0:
    return jnp.broadcast_to(jnp.eye(a.shape[-2], dtype=a.dtype), a.shape)
  elif n < 0:
    a = inv(a)
    n = np.abs(n)

  if n == 1:
    return a
  elif n == 2:
    return a @ a
  elif n == 3:
    return (a @ a) @ a

  z = result = None
  while n > 0:
    z = a if z is None else (z @ z)
    n, bit = divmod(n, 2)
    if bit:
      result = z if result is None else (result @ z)

  return result


@_wraps(np.linalg.matrix_rank)
def matrix_rank(M, tol=None):
  M = _promote_arg_dtypes(jnp.asarray(M))
  if M.ndim > 2:
    raise TypeError("array should have 2 or fewer dimensions")
  if M.ndim < 2:
    return jnp.any(M != 0).astype(jnp.int32)
  S = svd(M, full_matrices=False, compute_uv=False)
  if tol is None:
    tol = S.max() * np.max(M.shape) * jnp.finfo(S.dtype).eps
  return jnp.sum(S > tol)


@custom_jvp
@_wraps(np.linalg.slogdet)
@jit
def slogdet(a):
  a = _promote_arg_dtypes(jnp.asarray(a))
  dtype = lax.dtype(a)
  a_shape = jnp.shape(a)
  if len(a_shape) < 2 or a_shape[-1] != a_shape[-2]:
    msg = "Argument to slogdet() must have shape [..., n, n], got {}"
    raise ValueError(msg.format(a_shape))
  lu, pivot, _ = lax_linalg.lu(a)
  diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
  is_zero = jnp.any(diag == jnp.array(0, dtype=dtype), axis=-1)
  parity = jnp.count_nonzero(pivot != jnp.arange(a_shape[-1]), axis=-1)
  if jnp.iscomplexobj(a):
    sign = jnp.prod(diag / jnp.abs(diag), axis=-1)
  else:
    sign = jnp.array(1, dtype=dtype)
    parity = parity + jnp.count_nonzero(diag < 0, axis=-1)
  sign = jnp.where(is_zero,
                  jnp.array(0, dtype=dtype),
                  sign * jnp.array(-2 * (parity % 2) + 1, dtype=dtype))
  logdet = jnp.where(
      is_zero, jnp.array(-jnp.inf, dtype=dtype),
      jnp.sum(jnp.log(jnp.abs(diag)), axis=-1))
  return sign, jnp.real(logdet)

@slogdet.defjvp
def _slogdet_jvp(primals, tangents):
  x, = primals
  g, = tangents
  sign, ans = slogdet(x)
  ans_dot = jnp.trace(solve(x, g), axis1=-1, axis2=-2)
  if jnp.issubdtype(jnp._dtype(x), jnp.complexfloating):
    sign_dot = (ans_dot - np.real(ans_dot)) * sign
    ans_dot = np.real(ans_dot)
  else:
    sign_dot = jnp.zeros_like(sign)
  return (sign, ans), (sign_dot, ans_dot)


def _cofactor_solve(a, b):
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
  a = _promote_arg_dtypes(jnp.asarray(a))
  b = _promote_arg_dtypes(jnp.asarray(b))
  a_shape = jnp.shape(a)
  b_shape = jnp.shape(b)
  a_ndims = len(a_shape)
  if not (a_ndims >= 2 and a_shape[-1] == a_shape[-2]
    and b_shape[-2:] == a_shape[-2:]):
    msg = ("The arguments to _cofactor_solve must have shapes "
           "a=[..., m, m] and b=[..., m, m]; got a={} and b={}")
    raise ValueError(msg.format(a_shape, b_shape))
  if a_shape[-1] == 1:
    return a[0, 0], b
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
  parity = jnp.count_nonzero(pivots != jnp.arange(a_shape[-1]), axis=-1)
  sign = jnp.asarray(-2 * (parity % 2) + 1, dtype=dtype)
  # partial_det[:, -1] contains the full determinant and
  # partial_det[:, -2] contains det(u) / u_{nn}.
  partial_det = jnp.cumprod(diag, axis=-1) * sign[..., None]
  lu = ops.index_update(lu, ops.index[..., -1, -1], 1.0 / partial_det[..., -2])
  permutation = jnp.broadcast_to(permutation, batch_dims + (a_shape[-1],))
  iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in batch_dims + (1,)))
  # filter out any matrices that are not full rank
  d = jnp.ones(x.shape[:-1], x.dtype)
  d = lax_linalg.triangular_solve(lu, d, left_side=True, lower=False)
  d = jnp.any(jnp.logical_or(jnp.isnan(d), jnp.isinf(d)), axis=-1)
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


@custom_jvp
@_wraps(np.linalg.det)
def det(a):
  sign, logdet = slogdet(a)
  return sign * jnp.exp(logdet)


@det.defjvp
def _det_jvp(primals, tangents):
  x, = primals
  g, = tangents
  y, z = _cofactor_solve(x, g)
  return y, jnp.trace(z, axis1=-1, axis2=-2)


@_wraps(np.linalg.eig)
def eig(a):
  a = _promote_arg_dtypes(jnp.asarray(a))
  return lax_linalg.eig(a, compute_left_eigenvectors=False)


@_wraps(np.linalg.eigvals)
def eigvals(a):
  return lax_linalg.eig(a, compute_left_eigenvectors=False,
                        compute_right_eigenvectors=False)[0]


@_wraps(np.linalg.eigh)
def eigh(a, UPLO=None, symmetrize_input=True):
  if UPLO is None or UPLO == "L":
    lower = True
  elif UPLO == "U":
    lower = False
  else:
    msg = "UPLO must be one of None, 'L', or 'U', got {}".format(UPLO)
    raise ValueError(msg)

  a = _promote_arg_dtypes(jnp.asarray(a))
  w, v = lax_linalg.eigh(a, lower=lower, symmetrize_input=symmetrize_input)
  return w, v


@_wraps(np.linalg.eigvalsh)
def eigvalsh(a, UPLO='L'):
  w, _ = eigh(a, UPLO)
  return w


@partial(custom_jvp, nondiff_argnums=(1,))
@_wraps(np.linalg.pinv, lax_description=textwrap.dedent("""\
    It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
    default `rcond` is `1e-15`. Here the default is
    `10. * max(num_rows, num_cols) * jnp.finfo(dtype).eps`.
    """))
def pinv(a, rcond=None):
  # Uses same algorithm as
  # https://github.com/numpy/numpy/blob/v1.17.0/numpy/linalg/linalg.py#L1890-L1979
  a = jnp.conj(a)
  if rcond is None:
    max_rows_cols = max(a.shape[-2:])
    rcond = 10. * max_rows_cols * jnp.finfo(a.dtype).eps
  rcond = jnp.asarray(rcond)
  u, s, v = svd(a, full_matrices=False)
  # Singular values less than or equal to ``rcond * largest_singular_value``
  # are set to zero.
  cutoff = rcond[..., jnp.newaxis] * jnp.amax(s, axis=-1, keepdims=True, initial=-jnp.inf)
  s = jnp.where(s > cutoff, s, jnp.inf)
  res = jnp.matmul(_T(v), jnp.divide(_T(u), s[..., jnp.newaxis]))
  return lax.convert_element_type(res, a.dtype)


@pinv.defjvp
def _pinv_jvp(rcond, primals, tangents):
  # The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems
  # Whose Variables Separate. Author(s): G. H. Golub and V. Pereyra. SIAM
  # Journal on Numerical Analysis, Vol. 10, No. 2 (Apr., 1973), pp. 413-432.
  # (via https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Derivative)
  a, = primals
  a_dot, = tangents
  p = pinv(a, rcond=rcond)
  m, n = a.shape[-2:]
  # TODO(phawkins): on TPU, we would need to opt into high precision here.
  # TODO(phawkins): consider if this can be simplified in the Hermitian case.
  p_dot = -p @ a_dot @ p
  p_dot = p_dot + p @ _H(p) @ _H(a_dot) @ (jnp.eye(m, dtype=a.dtype) - a @ p)
  p_dot = p_dot + (jnp.eye(n, dtype=a.dtype) - p @ a) @ _H(a_dot) @ _H(p) @ p
  return p, p_dot


@_wraps(np.linalg.inv)
def inv(a):
  if jnp.ndim(a) < 2 or a.shape[-1] != a.shape[-2]:
    raise ValueError(
      f"Argument to inv must have shape [..., n, n], got {a.shape}.")
  return solve(
    a, lax.broadcast(jnp.eye(a.shape[-1], dtype=lax.dtype(a)), a.shape[:-2]))


@partial(jit, static_argnums=(1, 2, 3))
def _norm(x, ord, axis: Union[None, Tuple[int, ...], int], keepdims):
  x = _promote_arg_dtypes(jnp.asarray(x))
  x_shape = jnp.shape(x)
  ndim = len(x_shape)

  if axis is None:
    # NumPy has an undocumented behavior that admits arbitrary rank inputs if
    # `ord` is None: https://github.com/numpy/numpy/issues/14215
    if ord is None:
      return jnp.sqrt(jnp.sum(jnp.real(x * jnp.conj(x)), keepdims=keepdims))
    axis = tuple(range(ndim))
  elif isinstance(axis, tuple):
    axis = tuple(canonicalize_axis(x, ndim) for x in axis)
  else:
    axis = (canonicalize_axis(axis, ndim),)

  num_axes = len(axis)
  if num_axes == 1:
    if ord is None or ord == 2:
      return jnp.sqrt(jnp.sum(jnp.real(x * jnp.conj(x)), axis=axis,
                            keepdims=keepdims))
    elif ord == jnp.inf:
      return jnp.amax(jnp.abs(x), axis=axis, keepdims=keepdims)
    elif ord == -jnp.inf:
      return jnp.amin(jnp.abs(x), axis=axis, keepdims=keepdims)
    elif ord == 0:
      return jnp.sum(x != 0, dtype=jnp.finfo(lax.dtype(x)).dtype,
                    axis=axis, keepdims=keepdims)
    elif ord == 1:
      # Numpy has a special case for ord == 1 as an optimization. We don't
      # really need the optimization (XLA could do it for us), but the Numpy
      # code has slightly different type promotion semantics, so we need a
      # special case too.
      return jnp.sum(jnp.abs(x), axis=axis, keepdims=keepdims)
    else:
      abs_x = jnp.abs(x)
      ord = lax._const(abs_x, ord)
      out = jnp.sum(abs_x ** ord, axis=axis, keepdims=keepdims)
      return jnp.power(out, 1. / ord)

  elif num_axes == 2:
    row_axis, col_axis = cast(Tuple[int, ...], axis)
    if ord is None or ord in ('f', 'fro'):
      return jnp.sqrt(jnp.sum(jnp.real(x * jnp.conj(x)), axis=axis,
                            keepdims=keepdims))
    elif ord == 1:
      if not keepdims and col_axis > row_axis:
        col_axis -= 1
      return jnp.amax(jnp.sum(jnp.abs(x), axis=row_axis, keepdims=keepdims),
                     axis=col_axis, keepdims=keepdims)
    elif ord == -1:
      if not keepdims and col_axis > row_axis:
        col_axis -= 1
      return jnp.amin(jnp.sum(jnp.abs(x), axis=row_axis, keepdims=keepdims),
                     axis=col_axis, keepdims=keepdims)
    elif ord == jnp.inf:
      if not keepdims and row_axis > col_axis:
        row_axis -= 1
      return jnp.amax(jnp.sum(jnp.abs(x), axis=col_axis, keepdims=keepdims),
                     axis=row_axis, keepdims=keepdims)
    elif ord == -jnp.inf:
      if not keepdims and row_axis > col_axis:
        row_axis -= 1
      return jnp.amin(jnp.sum(jnp.abs(x), axis=col_axis, keepdims=keepdims),
                     axis=row_axis, keepdims=keepdims)
    elif ord in ('nuc', 2, -2):
      x = jnp.moveaxis(x, axis, (-2, -1))
      if ord == 2:
        reducer = jnp.amax
      elif ord == -2:
        reducer = jnp.amin
      else:
        reducer = jnp.sum
      y = reducer(svd(x, compute_uv=False), axis=-1)
      if keepdims:
        result_shape = list(x_shape)
        result_shape[axis[0]] = 1
        result_shape[axis[1]] = 1
        y = jnp.reshape(y, result_shape)
      return y
    else:
      raise ValueError("Invalid order '{}' for matrix norm.".format(ord))
  else:
    raise ValueError(
        "Invalid axis values ({}) for jnp.linalg.norm.".format(axis))

@_wraps(np.linalg.norm)
def norm(x, ord=None, axis=None, keepdims=False):
  return _norm(x, ord, axis, keepdims)


@_wraps(np.linalg.qr)
def qr(a, mode="reduced"):
  if mode in ("reduced", "r", "full"):
    full_matrices = False
  elif mode == "complete":
    full_matrices = True
  else:
    raise ValueError("Unsupported QR decomposition mode '{}'".format(mode))
  a = _promote_arg_dtypes(jnp.asarray(a))
  q, r = lax_linalg.qr(a, full_matrices)
  if mode == "r":
    return r
  return q, r


@_wraps(np.linalg.solve)
@jit
def solve(a, b):
  a, b = _promote_arg_dtypes(jnp.asarray(a), jnp.asarray(b))
  return lax_linalg._solve(a, b)


@_wraps(np.linalg.lstsq, lax_description=textwrap.dedent("""\
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
def lstsq(a, b, rcond=None, *, numpy_resid=False):
  # TODO: add lstsq to lax_linalg and implement this function via those wrappers.
  # TODO: add custom jvp rule for more robust lstsq differentiation
  a, b = _promote_arg_dtypes(a, b)
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
  if rcond is None:
    rcond = jnp.finfo(dtype).eps * max(n, m)
  elif rcond < 0:
    rcond = jnp.finfo(dtype).eps
  u, s, vt = svd(a, full_matrices=False)
  mask = s >= rcond * s[0]
  rank = mask.sum()
  safe_s = jnp.where(mask, s, 1)
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
