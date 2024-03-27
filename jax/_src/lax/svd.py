# Copyright 2022 The JAX Authors.
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
# limitations under the License

"""A JIT-compatible library for QDWH-based singular value decomposition.

QDWH is short for QR-based dynamically weighted Halley iteration. The Halley
iteration implemented through QR decmopositions is numerically stable and does
not require solving a linear system involving the iteration matrix or
computing its inversion. This is desirable for multicore and heterogeneous
computing systems.

References:
Nakatsukasa, Yuji, and Nicholas J. Higham.
"Stable and efficient spectral divide and conquer algorithms for the symmetric
eigenvalue decomposition and the SVD." SIAM Journal on Scientific Computing 35,
no. 3 (2013): A1325-A1349.
https://epubs.siam.org/doi/abs/10.1137/120876605

Nakatsukasa, Yuji, Zhaojun Bai, and François Gygi.
"Optimizing Halley's iteration for computing the matrix polar decomposition."
SIAM Journal on Matrix Analysis and Applications 31, no. 5 (2010): 2700-2720.
https://epubs.siam.org/doi/abs/10.1137/090774999
"""

from __future__ import annotations

from collections.abc import Sequence
import functools
import operator
from typing import Any

import jax
from jax import lax
from jax._src import core
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _constant_svd(
    a: Any,
    return_nan: bool,
    full_matrices: bool,
    compute_uv: bool = True,
    subset_by_index: tuple[int, int] | None = None,
) -> Any | Sequence[Any]:
  """SVD on matrix of all zeros."""
  m, n = a.shape
  k = min(m, n)
  if subset_by_index is not None:
    k = min(k, subset_by_index[1] - subset_by_index[0])

  s = jnp.where(
      return_nan,
      jnp.full(shape=(k,), fill_value=jnp.nan, dtype=a.real.dtype),
      jnp.zeros(shape=(k,), dtype=a.real.dtype),
  )
  if compute_uv:
    fill_value = (
        jnp.nan + 1j * jnp.nan
        if jnp.issubdtype(a.dtype, jnp.complexfloating)
        else jnp.nan
    )
    if full_matrices:
      u = jnp.where(
          return_nan,
          jnp.full((m, m), fill_value, dtype=a.dtype),
          jnp.eye(m, m, dtype=a.dtype),
      )
      vh = jnp.where(
          return_nan,
          jnp.full((n, n), fill_value, dtype=a.dtype),
          jnp.eye(n, n, dtype=a.dtype),
      )
    else:
      u = jnp.where(
          return_nan,
          jnp.full((m, k), fill_value, dtype=a.dtype),
          jnp.eye(m, k, dtype=a.dtype),
      )
      vh = jnp.where(
          return_nan,
          jnp.full((k, n), fill_value, dtype=a.dtype),
          jnp.eye(k, n, dtype=a.dtype),
      )
    return (u, s, vh)
  else:
    return s


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _svd_tall_and_square_input(
    a: Any,
    hermitian: bool,
    compute_uv: bool,
    max_iterations: int,
    subset_by_index: tuple[int, int] | None = None,
) -> Any | Sequence[Any]:
  """Singular value decomposition for m x n matrix and m >= n.

  Args:
    a: A matrix of shape `m x n` with `m >= n`.
    hermitian: True if `a` is Hermitian.
    compute_uv: Whether to compute also `u` and `v` in addition to `s`.
    max_iterations: The predefined maximum number of iterations of QDWH.

  Returns:
    A 3-tuple (`u`, `s`, `v`), where `u` is a unitary matrix of shape `m x n`,
    `s` is vector of length `n` containing the singular values in the descending
    order, `v` is a unitary matrix of shape `n x n`, and
    `a = (u * s) @ v.T.conj()`. For `compute_uv=False`, only `s` is returned.
  """

  u, h, _, _ = lax.linalg.qdwh(a, is_hermitian=hermitian,
                               max_iterations=max_iterations)

  # TODO: Uses `eigvals_only=True` if `compute_uv=False`.
  v, s = lax.linalg.eigh(h, subset_by_index=subset_by_index)
  # Singular values are non-negative by definition. But eigh could return small
  # negative values, so we clamp them to zero.
  s = jnp.maximum(s, 0.0)

  # Flips the singular values in descending order.
  s_out = jnp.flip(s)

  if not compute_uv:
    return s_out

  # Reorders eigenvectors.
  v_out = jnp.fliplr(v)

  u_out = u @ v_out

  # Makes correction if computed `u` from qdwh is not unitary.
  # Section 5.5 of Nakatsukasa, Yuji, and Nicholas J. Higham. "Stable and
  # efficient spectral divide and conquer algorithms for the symmetric
  # eigenvalue decomposition and the SVD." SIAM Journal on Scientific Computing
  # 35, no. 3 (2013): A1325-A1349.
  def correct_rank_deficiency(u_out):
    u_out, r, *_ = lax.linalg.qr(u_out, full_matrices=False)
    u_out = u_out @ jnp.diag(lax.sign(jnp.diag(r)))
    return u_out

  eps = float(jnp.finfo(a.dtype).eps)
  u_out = lax.cond(s[0] < a.shape[1] * eps * s_out[0],
                   correct_rank_deficiency,
                   lambda u_out: u_out,
                   operand=(u_out))

  return (u_out, s_out, v_out)


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def _qdwh_svd(
    a: Any,
    full_matrices: bool,
    compute_uv: bool = True,
    hermitian: bool = False,
    max_iterations: int = 10,
    subset_by_index: tuple[int, int] | None = None,
) -> Any | Sequence[Any]:
  """Singular value decomposition.

  Args:
    a: A matrix of shape `m x n`.
    full_matrices: If True, `u` and `vh` have the shapes `m x m` and `n x n`,
      respectively. If False, the shapes are `m x k` and `k x n`, respectively,
      where `k = min(m, n)`.
    compute_uv: Whether to compute also `u` and `v` in addition to `s`.
    hermitian: True if `a` is Hermitian.
    max_iterations: The predefined maximum number of iterations of QDWH.

  Returns:
    A 3-tuple (`u`, `s`, `vh`), where `u` and `vh` are unitary matrices,
    `s` is vector of length `k` containing the singular values in the
    non-increasing order, and `k = min(m, n)`. The shapes of `u` and `vh`
    depend on the value of `full_matrices`. For `compute_uv=False`,
    only `s` is returned.
  """
  m, n = a.shape

  is_flip = False
  if m < n:
    a = a.T.conj()
    m, n = a.shape
    is_flip = True

  reduce_to_square = False
  if full_matrices:
    q_full, a_full, *_ = lax.linalg.qr(a, full_matrices=True)
    q = q_full[:, :n]
    u_out_null = q_full[:, n:]
    a = a_full[:n, :]
    reduce_to_square = True
  else:
    # The constant `1.15` comes from Yuji Nakatsukasa's implementation
    # https://www.mathworks.com/matlabcentral/fileexchange/36830-symmetric-eigenvalue-decomposition-and-the-svd?s_tid=FX_rc3_behav
    if m > 1.15 * n:
      q, a, *_ = lax.linalg.qr(a, full_matrices=False)
      reduce_to_square = True

  if not compute_uv:
    with jax.default_matmul_precision('float32'):
      return _svd_tall_and_square_input(
          a, hermitian, compute_uv, max_iterations, subset_by_index
      )

  with jax.default_matmul_precision('float32'):
    u_out, s_out, v_out = _svd_tall_and_square_input(
        a, hermitian, compute_uv, max_iterations, subset_by_index
    )
    if reduce_to_square:
      u_out = q @ u_out

  if full_matrices:
    u_out = jnp.hstack((u_out, u_out_null))

  if is_flip:
    return(v_out, s_out, u_out.T.conj())

  return (u_out, s_out, v_out.T.conj())


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def svd(
    a: Any,
    full_matrices: bool,
    compute_uv: bool = True,
    hermitian: bool = False,
    max_iterations: int = 10,
    subset_by_index: tuple[int, int] | None = None,
) -> Any | Sequence[Any]:
  """Singular value decomposition.

  Args:
    a: A matrix of shape `m x n`.
    full_matrices: If True, `u` and `vh` have the shapes `m x m` and `n x n`,
      respectively. If False, the shapes are `m x k` and `k x n`, respectively,
      where `k = min(m, n)`.
    compute_uv: Whether to compute also `u` and `v` in addition to `s`.
    hermitian: True if `a` is Hermitian.
    max_iterations: The predefined maximum number of iterations of QDWH.
    subset_by_index: Optional 2-tuple [start, end] indicating the range of
      indices of singular componenets to compute. For example, if
      ``subset_by_index`` = [0,2], then ``svd`` computes the two largest
      singular values (and their singular vectors if `compute_uv` is true.

  Returns:
    A 3-tuple (`u`, `s`, `vh`), where `u` and `vh` are unitary matrices,
    `s` is vector of length `k` containing the singular values in the
    non-increasing order, and `k = min(m, n)`. The shapes of `u` and `vh`
    depend on the value of `full_matrices`. For `compute_uv=False`,
    only `s` is returned.
  """
  full_matrices = core.concrete_or_error(
      bool, full_matrices, 'The `full_matrices` argument must be statically '
      'specified to use `svd` within JAX transformations.')

  compute_uv = core.concrete_or_error(
      bool, compute_uv, 'The `compute_uv` argument must be statically '
      'specified to use `svd` within JAX transformations.')

  hermitian = core.concrete_or_error(
      bool,
      hermitian,
      'The `hermitian` argument must be statically '
      'specified to use `svd` within JAX transformations.',
  )

  max_iterations = core.concrete_or_error(
      int,
      max_iterations,
      'The `max_iterations` argument must be statically '
      'specified to use `svd` within JAX transformations.',
  )

  if subset_by_index is not None:
    if len(subset_by_index) != 2:
      raise ValueError('subset_by_index must be a tuple of size 2.')
    # Make sure subset_by_index is a concrete tuple.
    subset_by_index = (
        operator.index(subset_by_index[0]),
        operator.index(subset_by_index[1]),
    )
    if subset_by_index[0] >= subset_by_index[1]:
      raise ValueError('Got empty index range in subset_by_index.')
    if subset_by_index[0] < 0:
      raise ValueError('Indices in subset_by_index must be non-negative.')
    m, n = a.shape
    rank = n if n < m else m
    if subset_by_index[1] > rank:
      raise ValueError('Index in subset_by_index[1] exceeds matrix size.')
    if full_matrices and subset_by_index != (0, rank):
      raise ValueError(
          'full_matrices and subset_by_index cannot be both be set.'
      )
    # By convention, eigenvalues are numbered in non-decreasing order, while
    # singular values are numbered non-increasing order, so change
    # subset_by_index accordingly.
    subset_by_index = (rank - subset_by_index[1], rank - subset_by_index[0])

  # QDWH algorithm fails at zero-matrix `A` and produces all NaNs, which can
  # be seen from a dynamically weighted Halley (DWH) iteration:
  # X_{k+1} = X_k(a_k I + b_k {X_k}^H X_k)(I + c_k {X_k}^H X_k)^{−1} and
  # X_0 = A/alpha, where alpha = ||A||_2, the triplet (a_k, b_k, c_k) are
  # weighting parameters, and X_k denotes the k^{th} iterate.
  all_zero = jnp.all(a == 0)
  non_finite = jnp.logical_not(jnp.all(jnp.isfinite(a)))
  return lax.cond(
      jnp.logical_or(all_zero, non_finite),
      functools.partial(
          _constant_svd,
          return_nan=non_finite,
          full_matrices=full_matrices,
          compute_uv=compute_uv,
          subset_by_index=subset_by_index,
      ),
      functools.partial(
          _qdwh_svd,
          full_matrices=full_matrices,
          compute_uv=compute_uv,
          hermitian=hermitian,
          max_iterations=max_iterations,
          subset_by_index=subset_by_index,
      ),
      operand=(a),
  )
