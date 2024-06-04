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

# Shims that allow the XLA CPU backend to call scipy-provided LAPACK kernels
# via CustomCallWithLayout.

from collections.abc import Sequence
from enum import Enum
from typing import Optional

import jaxlib.mlir.dialects.stablehlo as hlo
import jaxlib.mlir.ir as ir  # pylint: disable=consider-using-from-import
import numpy as np

from jaxlib import xla_client

from .cpu import _lapack
from .cpu._lapack import eig, schur, svd
from .hlo_helpers import (
    DimensionSize,
    ShapeTypePair,
    custom_call,
    hlo_add,
    hlo_min,
    mk_result_types_and_shapes,
)

for _name, _value in _lapack.registrations().items():
  xla_client.register_custom_call_target(
      _name,
      _value,
      platform="cpu",
      api_version=(1 if _name.endswith("_ffi") else 0),
  )


def char_attr(c):
  return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(8), ord(c))


def lapack_int_attr(value):
  return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)


def enum_to_char_attr(e: Enum):
  return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(8), e.value)


def matrix_side_attr(*, left_side: bool):
  return char_attr("L" if left_side else "R")


def matrix_uplo_attr(*, lower: bool):
  return char_attr("L" if lower else "U")


def matrix_transpose_attr(*, transpose: bool, conjugate: bool):
  return char_attr(("C" if conjugate else "T") if transpose else "N")


def matrix_diagonal_attr(*, unit_diag: bool):
  return char_attr("U" if unit_diag else "N")


def svd_computation_attr(
    *, compute_uv: bool, full_matrices: Optional[bool] = True
):
  mode = "A"
  if full_matrices is None:
    full_matrices = True
  if not compute_uv:
    # We should assert that `full_matrices` is never True here.
    # This should never happen because `full_matrices` can only be computed when
    # `compute_uv` is True. However, at this point there are too many tests that
    # rely on this behavior.
    mode = "N"
  elif not full_matrices:
    mode = "S"
  return char_attr(mode)


# TODO(phawkins): it would be nice to avoid duplicating code for each type.

# ?trsm(left_side, lower, trans_a, diag, m, n, alpha, a, b):
# triangular solve
def trsm_hlo(dtype, alpha, a, b,
             left_side=False, lower=False, trans_a=False,
             conj_a=False, diag=False, *,
             b_shape_vals: tuple[DimensionSize, ...]):
  _lapack.initialize()
  b_type = ir.RankedTensorType(b.type)

  batch_dims_vals = b_shape_vals[:-2]
  num_bd = len(batch_dims_vals)

  if dtype == np.float32:
    fn = "blas_strsm_ffi"
  elif dtype == np.float64:
    fn = "blas_dtrsm_ffi"
  elif dtype == np.complex64:
    fn = "blas_ctrsm_ffi"
  elif dtype == np.complex128:
    fn = "blas_ztrsm_ffi"
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  if conj_a and not trans_a:
    raise NotImplementedError("Conjugation without transposition not supported")
  scalar_layout = []
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  result_types, result_shapes = mk_result_types_and_shapes(
      [(b_shape_vals, b_type.element_type)])
  return custom_call(
      fn,
      result_types=result_types,
      operands=[a, b, alpha],
      operand_layouts=[layout] * 2 + [scalar_layout],
      result_layouts=[layout],
      operand_output_aliases={1: 0},
      result_shapes=result_shapes,
      backend_config={
          "side": matrix_side_attr(left_side=left_side),
          "uplo": matrix_uplo_attr(lower=lower),
          "trans_x": matrix_transpose_attr(transpose=trans_a, conjugate=conj_a),
          "diag": matrix_diagonal_attr(unit_diag=diag),
      },
      api_version=4,
  ).results


# # ?getrf: LU decomposition

def getrf_hlo(dtype, a: ir.Value, *,
              a_shape_vals: tuple[DimensionSize, ...]):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  assert len(a_shape_vals) >= 2
  batch_dims_vals = a_shape_vals[:-2]
  num_bd = len(a_shape_vals) - 2
  m, n = a_shape_vals[-2:]

  if dtype == np.float32:
    fn = "lapack_sgetrf_ffi"
  elif dtype == np.float64:
    fn = "lapack_dgetrf_ffi"
  elif dtype == np.complex64:
    fn = "lapack_cgetrf_ffi"
  elif dtype == np.complex128:
    fn = "lapack_zgetrf_ffi"
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  i32_type = ir.IntegerType.get_signless(32)
  shape_type_pairs: Sequence[ShapeTypePair] = [
      (a_shape_vals, a_type.element_type),
      (batch_dims_vals + (hlo_min(m, n),), i32_type),
      (batch_dims_vals, i32_type)
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)

  return custom_call(
      fn,
      result_types=result_types,
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[
        layout,
        tuple(range(num_bd, -1, -1)),
        tuple(range(num_bd - 1, -1, -1)),
      ],
      operand_output_aliases={0: 0},
      result_shapes=result_shapes,
      backend_config={},
      api_version=4,
  ).results

# # ?geqrf: QR decomposition

def geqrf_hlo(dtype, a: ir.Value, *,
              a_shape_vals: tuple[DimensionSize, ...]):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  assert len(a_shape_vals) >= 2
  m, n = a_shape_vals[-2:]
  assert type(m) is int
  assert type(n) is int

  batch_dims_vals = a_shape_vals[:-2]
  num_bd = len(batch_dims_vals)

  if dtype == np.float32:
    fn = "lapack_sgeqrf_ffi"
    lwork = _lapack.lapack_sgeqrf_workspace_ffi(m, n)
  elif dtype == np.float64:
    fn = "lapack_dgeqrf_ffi"
    lwork = _lapack.lapack_dgeqrf_workspace_ffi(m, n)
  elif dtype == np.complex64:
    fn = "lapack_cgeqrf_ffi"
    lwork = _lapack.lapack_cgeqrf_workspace_ffi(m, n)
  elif dtype == np.complex128:
    fn = "lapack_zgeqrf_ffi"
    lwork = _lapack.lapack_zgeqrf_workspace_ffi(m, n)
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)

  shape_type_pairs: Sequence[ShapeTypePair] = [
      (a_shape_vals, a_type.element_type),
      (batch_dims_vals + (min(m, n),), a_type.element_type),
      (batch_dims_vals, i32_type),
      ([lwork], a_type.element_type),
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      fn,
      result_types=result_types,
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[
          layout,
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
          [0],
      ],
      operand_output_aliases={0: 0},
      result_shapes=result_shapes,
      backend_config={},
      api_version=4,
  ).results
  return out[:3]


# # ?orgqr: product of elementary Householder reflectors:
def orgqr_hlo(dtype, a: ir.Value, tau, *,
              a_shape_vals: tuple[DimensionSize, ...],
              tau_shape_vals: tuple[DimensionSize, ...]):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  dims_vals = a_shape_vals
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m != ir.ShapedType.get_dynamic_size()
  assert n != ir.ShapedType.get_dynamic_size()
  batch_dims_vals = dims_vals[:-2]
  num_bd = len(batch_dims_vals)

  k = tau_shape_vals[-1]
  assert type(k) is int

  if dtype == np.float32:
    fn = "lapack_sorgqr_ffi"
    lwork = _lapack.lapack_sorgqr_workspace_ffi(m, n, k)
  elif dtype == np.float64:
    fn = "lapack_dorgqr_ffi"
    lwork = _lapack.lapack_dorgqr_workspace_ffi(m, n, k)
  elif dtype == np.complex64:
    fn = "lapack_cungqr_ffi"
    lwork = _lapack.lapack_cungqr_workspace_ffi(m, n, k)
  elif dtype == np.complex128:
    fn = "lapack_zungqr_ffi"
    lwork = _lapack.lapack_zungqr_workspace_ffi(m, n, k)
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)
  shape_type_pairs: Sequence[ShapeTypePair] = [
      (a_shape_vals, a_type.element_type),
      (batch_dims_vals, i32_type),
      ([lwork], a_type.element_type),
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      fn,
      result_types=result_types,
      operands=[
       a, tau
      ],
      operand_layouts=[
        layout,
        tuple(range(num_bd, -1, -1)),
      ],
      result_layouts=[
        layout,
        tuple(range(num_bd - 1, -1, -1)),
        [0],
      ],
      operand_output_aliases={0: 0},
      result_shapes=result_shapes,
      backend_config={},
      api_version=4,
  ).results
  return out[:2]


# ?potrf: Cholesky decomposition

def potrf_hlo(dtype, a: ir.Value, *, lower=False,
              a_shape_vals: tuple[DimensionSize, ...]):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  n = a_shape_vals[-1]
  if dtype == np.float32:
    fn = "lapack_spotrf_ffi"
  elif dtype == np.float64:
    fn = "lapack_dpotrf_ffi"
  elif dtype == np.complex64:
    fn = "lapack_cpotrf_ffi"
  elif dtype == np.complex128:
    fn = "lapack_zpotrf_ffi"
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")
  batch_dims_vals = a_shape_vals[:-2]
  num_bd = len(batch_dims_vals)

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  info_layout = tuple(range(num_bd - 1, -1, -1))

  shape_type_pairs: Sequence[ShapeTypePair] = [
      (a_shape_vals, a_type.element_type),
      (batch_dims_vals, ir.IntegerType.get_signless(32))
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      fn,
      result_types=result_types,
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[layout, info_layout],
      operand_output_aliases={0: 0},
      result_shapes=result_shapes,
      backend_config={
          "uplo": matrix_uplo_attr(lower=lower),
      },
      api_version=4,
  ).results
  return out[:2]


# # ?gesdd: Singular value decomposition

def gesdd_hlo(dtype, a: ir.Value, *, full_matrices=True, compute_uv=True,
              a_shape_vals: tuple[DimensionSize, ...]):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  assert len(a_shape_vals) >= 2
  m, n = a_shape_vals[-2:]
  assert type(m) is int
  assert type(n) is int
  batch_dims_vals = a_shape_vals[:-2]
  num_bd = len(batch_dims_vals)

  mode_attr = svd_computation_attr(
      compute_uv=compute_uv, full_matrices=full_matrices
  )
  mode_value = svd.ComputationMode(mode_attr.value)
  i32_type = ir.IntegerType.get_signless(32)
  workspace: list[ShapeTypePair]
  if dtype == np.float32:
    fn = "lapack_sgesdd_ffi"
    singular_vals_type = ir.F32Type.get()
    lwork = _lapack.sgesdd_work_size_ffi(m, n, mode_value)
    workspace = [
        ([_lapack.gesdd_iwork_size_ffi(m, n)], i32_type),
        ([lwork], a_type.element_type),
    ]
  elif dtype == np.float64:
    fn = "lapack_dgesdd_ffi"
    singular_vals_type = ir.F64Type.get()
    lwork = _lapack.dgesdd_work_size_ffi(m, n, mode_value)
    workspace = [
        ([_lapack.gesdd_iwork_size_ffi(m, n)], i32_type),
        ([lwork], a_type.element_type),
    ]
  elif dtype == np.complex64:
    fn = "lapack_cgesdd_ffi"
    singular_vals_type = ir.F32Type.get()
    lwork = _lapack.cgesdd_work_size_ffi(m, n, mode_value)
    workspace = [
        ([_lapack.gesdd_rwork_size_ffi(m, n, mode_value)], ir.F32Type.get()),
        ([_lapack.gesdd_iwork_size_ffi(m, n)], i32_type),
        ([lwork], a_type.element_type),
    ]
  elif dtype == np.complex128:
    fn = "lapack_zgesdd_ffi"
    singular_vals_type = ir.F64Type.get()
    lwork = _lapack.zgesdd_work_size_ffi(m, n, mode_value)
    workspace = [
        ([_lapack.gesdd_rwork_size_ffi(m, n, mode_value)], ir.F64Type.get()),
        ([_lapack.gesdd_iwork_size_ffi(m, n)], i32_type),
        ([lwork], a_type.element_type),
    ]
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  a_elem_type = a_type.element_type
  shape_type_pairs: Sequence[ShapeTypePair] = [
      (a_shape_vals, a_elem_type),
      (batch_dims_vals + (min(m, n),), singular_vals_type),
      (batch_dims_vals + (m, m if full_matrices else min(m, n)), a_elem_type),
      (batch_dims_vals + (n if full_matrices else min(m, n), n), a_elem_type),
      (batch_dims_vals, i32_type),
      *workspace,
  ]
  workspace_layout = [[0]] * len(workspace)
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      fn,
      result_types=result_types,
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[
          layout,
          (num_bd,) + tuple(range(num_bd - 1, -1, -1)),
          layout,
          layout,
          tuple(range(num_bd - 1, -1, -1)),
          *workspace_layout,
      ],
      operand_output_aliases={0: 0},
      result_shapes=result_shapes,
      backend_config={
          "mode": mode_attr,
      },
      api_version=4,
  ).results
  return out[1:5]


# # syevd: Symmetric eigendecomposition

def syevd_hlo(dtype, a: ir.Value,
              a_shape_vals: tuple[DimensionSize, ...],
              lower=False):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  assert len(a_shape_vals) >= 2
  m, n = a_shape_vals[-2:]
  # Non-batch dimensions must be static
  assert type(m) is int and type(n) is int and m == n, a_shape_vals

  batch_dims_vals = a_shape_vals[:-2]
  num_bd = len(a_shape_vals) - 2
  mode = enum_to_char_attr(eig.ComputationMode.kComputeEigenvectors)

  i32_type = ir.IntegerType.get_signless(32)
  workspace: list[ShapeTypePair]
  if dtype == np.float32:
    fn = "lapack_ssyevd_ffi"
    eigvals_type = ir.F32Type.get()
    workspace = [
        ([_lapack.syevd_work_size_ffi(n)], a_type.element_type),
        ([_lapack.syevd_iwork_size_ffi(n)], i32_type),
    ]
  elif dtype == np.float64:
    fn = "lapack_dsyevd_ffi"
    eigvals_type = ir.F64Type.get()
    workspace = [
        ([_lapack.syevd_work_size_ffi(n)], a_type.element_type),
        ([_lapack.syevd_iwork_size_ffi(n)], i32_type),
    ]
  elif dtype == np.complex64:
    fn = "lapack_cheevd_ffi"
    eigvals_type = ir.F32Type.get()
    workspace = [
        ([_lapack.heevd_work_size_ffi(n)], a_type.element_type),
        ([_lapack.heevd_rwork_size_ffi(n)], eigvals_type),
        ([_lapack.syevd_iwork_size_ffi(n)], i32_type),
    ]
  elif dtype == np.complex128:
    fn = "lapack_zheevd_ffi"
    eigvals_type = ir.F64Type.get()
    workspace = [
        ([_lapack.heevd_work_size_ffi(n)],  a_type.element_type),
        ([_lapack.heevd_rwork_size_ffi(n)], eigvals_type),
        ([_lapack.syevd_iwork_size_ffi(n)], i32_type),
    ]
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  workspace_layouts = [[0]] * len(workspace)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  result_types, result_shapes = mk_result_types_and_shapes([
      (a_shape_vals, a_type.element_type),
      (batch_dims_vals + (n,), eigvals_type),
      (batch_dims_vals, i32_type),
      *workspace,
  ])

  out = custom_call(
      fn,
      result_types=result_types,
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[
          layout,
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
          *workspace_layouts,
      ],
      operand_output_aliases={0: 0},
      result_shapes=result_shapes,
      backend_config={
          "uplo": matrix_uplo_attr(lower=lower),
          "mode": mode,
      },
      api_version=4,
  ).results
  return out[:3]


# # geev: Nonsymmetric eigendecomposition (eig)

def geev_hlo(dtype, input, *,
             input_shape_vals: tuple[DimensionSize, ...],  # input.shape as ir.Values
             jobvl=True, jobvr=True):
  # input_shape_vals are used for when input has dynamic shapes.
  _lapack.initialize()
  input_shape = ir.RankedTensorType(input.type).shape
  assert len(input_shape) >= 2
  n = input_shape_vals[-1]
  batch_dims_vals = input_shape_vals[:-2]
  num_bd = len(batch_dims_vals)

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  compute_left = (
      eig.ComputationMode.kComputeEigenvectors
      if jobvl
      else eig.ComputationMode.kNoEigenvectors
  )

  compute_right = (
      eig.ComputationMode.kComputeEigenvectors
      if jobvr
      else eig.ComputationMode.kNoEigenvectors
  )

  i32_type = ir.IntegerType.get_signless(32)
  f32_type = ir.F32Type.get()
  f64_type = ir.F64Type.get()
  c64_type = ir.ComplexType.get(ir.F32Type.get())
  c128_type = ir.ComplexType.get(ir.F64Type.get())

  workspace: list[ShapeTypePair]
  eigvals: list[ShapeTypePair]
  if dtype == np.float32:
    fn = "lapack_sgeev_ffi"
    real = True
    eigvecs_type = c64_type
    workspace = [([n, n], f32_type)] * 3
    workspace_layouts = [[0, 1]] * 3
    eigvals = [(batch_dims_vals + (n,), f32_type)] * 2
    eigvals_layouts = [tuple(range(num_bd, -1, -1))] * 2
  elif dtype == np.float64:
    fn = "lapack_dgeev_ffi"
    real = True
    eigvecs_type = c128_type
    workspace = [([n, n], f64_type)] * 3
    workspace_layouts = [[0, 1]] * 3
    eigvals = [(batch_dims_vals + (n,), f64_type)] * 2
    eigvals_layouts = [tuple(range(num_bd, -1, -1))] * 2
  elif dtype == np.complex64:
    fn = "lapack_cgeev_ffi"
    real = False
    eigvecs_type = c64_type
    workspace = [([n, n], c64_type), ([hlo_add(n, n)], f32_type)]
    workspace_layouts = [[0, 1], [0]]
    eigvals = [(batch_dims_vals + (n,), c64_type)]
    eigvals_layouts = [tuple(range(num_bd, -1, -1))]
  elif dtype == np.complex128:
    fn = "lapack_zgeev_ffi"
    real = False
    eigvecs_type = c128_type
    workspace = [([n, n], c128_type), ([hlo_add(n, n)], f64_type)]
    workspace_layouts = [[0, 1], [0]]
    eigvals = [(batch_dims_vals + (n,), c128_type)]
    eigvals_layouts = [tuple(range(num_bd, -1, -1))]
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  info_layout = tuple(range(num_bd - 1, -1, -1))
  shape_type_pairs: Sequence[ShapeTypePair] = [
      *eigvals,
      (input_shape_vals, eigvecs_type),
      (input_shape_vals, eigvecs_type),
      (batch_dims_vals, i32_type),
      *workspace,
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      fn,
      result_types=result_types,
      operands=[input],
      operand_layouts=[layout],
      result_layouts=(
          *eigvals_layouts,
          layout,
          layout,
          info_layout,
          *workspace_layouts,
      ),
      result_shapes=result_shapes,
      backend_config={
          "compute_left": enum_to_char_attr(compute_left),
          "compute_right": enum_to_char_attr(compute_right),
      },
      api_version=4,
  ).results
  if real:
    return (hlo.complex(out[0], out[1]), out[2], out[3], out[4])
  else:
    return out[0:4]

# # gees : Schur factorization

def gees_hlo(dtype, a, *, jobvs=True, sort=False, select=None,
             a_shape_vals: tuple[DimensionSize, ...]):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  etype = a_type.element_type
  assert len(a_shape_vals) >= 2
  n = a_shape_vals[-1]
  batch_dims_vals = a_shape_vals[:-2]
  num_bd = len(batch_dims_vals)
  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

  if sort:
    raise NotImplementedError(
        "The sort feature of LAPACK's gees routine is not implemented.")

  mode = (
      schur.ComputationMode.kComputeSchurVectors
      if jobvs
      else schur.ComputationMode.kNoComputeSchurVectors
  )
  sort = schur.Sort.kSortEigenvalues if sort else schur.Sort.kNoSortEigenvalues

  if dtype == np.float32:
    fn = "lapack_sgees_ffi"
  elif dtype == np.float64:
    fn = "lapack_dgees_ffi"
  elif dtype == np.complex64:
    fn = "lapack_cgees_ffi"
  elif dtype == np.complex128:
    fn = "lapack_zgees_ffi"
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  workspace: list[ShapeTypePair]
  eigvals: list[ShapeTypePair]
  is_complex = np.issubdtype(dtype, np.complexfloating)
  if not is_complex:
    workspace = []
    workspace_layouts = []
    eigvals = [(batch_dims_vals + (n,), etype)] * 2
    eigvals_layouts = [tuple(range(num_bd, -1, -1))] * 2
  else:
    workspace = [([n], ir.ComplexType(etype).element_type)]
    workspace_layouts = [[0]]
    eigvals = [(batch_dims_vals + (n,), etype)]
    eigvals_layouts = [tuple(range(num_bd, -1, -1))]

  i32_type = ir.IntegerType.get_signless(32)
  shape_type_pairs = [
      (a_shape_vals, etype),
      *eigvals,
      (a_shape_vals, etype),
      (batch_dims_vals, i32_type),
      (batch_dims_vals, i32_type),
      *workspace,
  ]
  result_types, result_shapes = mk_result_types_and_shapes(shape_type_pairs)
  out = custom_call(
      fn,
      result_types=result_types,
      operands=[a],
      # TODO: figure out how to put the callable select function here
      # TODO(paruzelp): answer: FFI supports execution context => put `select`
      operand_layouts=[layout],
      result_layouts=[
          layout,
          *eigvals_layouts,
          layout,
          tuple(range(num_bd - 1, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
          *workspace_layouts,
      ],
      operand_output_aliases={0: 0},
      result_shapes=result_shapes,
      backend_config={
          "mode": enum_to_char_attr(mode),
          "sort": enum_to_char_attr(sort),
      },
      api_version=4,
  ).results
  # out: Schur Form, Eigenvalues, Schur Vectors, Selected Eigenvalues, Info
  if is_complex:
    return out[0], out[1], out[2], out[3], out[4]
  else:
    return out[0], (out[1], out[2]), out[3], out[4], out[5]


# gehrd: Reduction of a non-symmetric square matrix to upper Hessenberg form.
def gehrd_hlo(dtype, a):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n, (m, n)
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)

  if dtype == np.float32:
    fn = "lapack_sgehrd_ffi"
    lwork = _lapack.lapack_sgehrd_workspace_ffi(n, n, 1, n)
  elif dtype == np.float64:
    fn = "lapack_dgehrd_ffi"
    lwork = _lapack.lapack_dgehrd_workspace_ffi(n, n, 1, n)
  elif dtype == np.complex64:
    fn = "lapack_cgehrd_ffi"
    lwork = _lapack.lapack_cgehrd_workspace_ffi(n, n, 1, n)
  elif dtype == np.complex128:
    fn = "lapack_zgehrd_ffi"
    lwork = _lapack.lapack_zgehrd_workspace_ffi(n, n, 1, n)
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)
  out = custom_call(
      fn,
      result_types=[
          a.type,
          ir.RankedTensorType.get(batch_dims + (n - 1,), a_type.element_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
      ],
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[
          layout,
          (num_bd,) + tuple(range(num_bd - 1, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
          [0],
      ],
      operand_output_aliases={0: 0},
      backend_config={
          "low": lapack_int_attr(1),
          "high": lapack_int_attr(n),
      },
      api_version=4,
  ).results
  return out[:3]


# sytrd: Reduction of a symmetric (Hermitian) matrix to tridiagonal form.
def sytrd_hlo(dtype, a, *, lower):
  _lapack.initialize()
  a_type = ir.RankedTensorType(a.type)
  dims = a_type.shape
  assert len(dims) >= 2
  m, n = dims[-2:]
  assert m == n, (m, n)
  batch_dims = tuple(dims[:-2])
  num_bd = len(batch_dims)

  if dtype == np.float32:
    fn = "lapack_ssytrd_ffi"
    lwork = _lapack.lapack_ssytrd_workspace_ffi(n, n)
    diag_type = a_type.element_type
  elif dtype == np.float64:
    fn = "lapack_dsytrd_ffi"
    lwork = _lapack.lapack_dsytrd_workspace_ffi(n, n)
    diag_type = a_type.element_type
  elif dtype == np.complex64:
    fn = "lapack_chetrd_ffi"
    lwork = _lapack.lapack_chetrd_workspace_ffi(n, n)
    diag_type = ir.F32Type.get()
  elif dtype == np.complex128:
    fn = "lapack_zhetrd_ffi"
    lwork = _lapack.lapack_zhetrd_workspace_ffi(n, n)
    diag_type = ir.F64Type.get()
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))
  i32_type = ir.IntegerType.get_signless(32)
  out = custom_call(
      fn,
      result_types=[
          a.type,
          ir.RankedTensorType.get(batch_dims + (n,), a_type.element_type),
          ir.RankedTensorType.get(batch_dims + (n,), diag_type),
          ir.RankedTensorType.get(batch_dims + (n - 1,), diag_type),
          ir.RankedTensorType.get(batch_dims, i32_type),
          ir.RankedTensorType.get([lwork], a_type.element_type),
      ],
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[
          layout,
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd, -1, -1)),
          tuple(range(num_bd - 1, -1, -1)),
          [0],
      ],
      operand_output_aliases={0: 0},
      backend_config={
          "uplo": matrix_uplo_attr(lower=lower),
      },
      api_version=4,
  ).results
  x_out, tau, on_diag, off_diag, info, _work = out
  return x_out, tau, on_diag, off_diag, info
