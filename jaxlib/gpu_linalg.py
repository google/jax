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

import functools
from functools import partial
import importlib
import numpy as np
import operator

import jaxlib.mlir.ir as ir

from .hlo_helpers import custom_call
from .gpu_common_utils import GpuLibNotLinkedError

from jaxlib import xla_client

for cuda_module_name in [".cuda", "jax_cuda12_plugin"]:
  try:
    _cuda_linalg = importlib.import_module(
        f"{cuda_module_name}._linalg", package="jaxlib"
    )
  except ImportError:
    _cuda_linalg = None
  else:
    break

if _cuda_linalg:
  for _name, _value in _cuda_linalg.registrations().items():
    api_version = 0 if _name == "cu_cholesky_update" else 1
    xla_client.register_custom_call_target(
        _name, _value, platform="CUDA", api_version=api_version
    )

for rocm_module_name in [".rocm", "jax_rocm60_plugin"]:
  try:
    _hip_linalg = importlib.import_module(
        f"{rocm_module_name}._linalg", package="jaxlib"
    )
  except ImportError:
    _hip_linalg = None
  else:
    break

if _hip_linalg:
  for _name, _value in _hip_linalg.registrations().items():
    xla_client.register_custom_call_target(
        _name, _value, platform="ROCM", api_version=1
    )

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def _lu_pivots_to_permutation_hlo(platform, gpu_linalg, pivots, *, permutation_size):
  """Kernel for the transformation of pivots to permutations on GPU."""
  typ = ir.RankedTensorType(pivots.type)
  dims = typ.shape
  i32_type = ir.IntegerType.get_signless(32)

  assert typ.element_type == i32_type, typ

  if not gpu_linalg:
    raise GpuLibNotLinkedError()

  pivots_layout = tuple(range(len(dims) - 1, -1, -1))
  permutations_layout = pivots_layout
  permutations_dims = list(dims)
  permutations_dims[-1] = permutation_size
  permutations_type = ir.RankedTensorType.get(permutations_dims, i32_type)
  return custom_call(
      f"{platform}_lu_pivots_to_permutation",
      api_version=4,
      result_types=[permutations_type],
      operands=[pivots],
      backend_config=dict(
          permutation_size=ir.IntegerAttr.get(i32_type, permutation_size),
      ),
      operand_layouts=[pivots_layout],
      result_layouts=[permutations_layout],
  ).results

cuda_lu_pivots_to_permutation = partial(_lu_pivots_to_permutation_hlo, "cu",
                                        _cuda_linalg)
hip_lu_pivots_to_permutation = partial(
    _lu_pivots_to_permutation_hlo, "hip", _hip_linalg)



def _cholesky_update_hlo(platform, gpu_linalg, r_matrix, w_vector, dtype):
  """Cholesky update."""
  del platform
  r_type = ir.RankedTensorType(r_matrix.type)
  dims = r_type.shape
  assert dims[0] == dims[1]
  n = dims[0]

  if not gpu_linalg:
    raise GpuLibNotLinkedError()

  np_type = np.dtype(dtype)
  opaque = gpu_linalg.build_cholesky_update_descriptor(np_type, n)

  return custom_call(
      "cu_cholesky_update",
      operands = [r_matrix, w_vector],
      result_types=[
          ir.RankedTensorType.get((n, n), r_type.element_type),
          ir.RankedTensorType.get((n,), r_type.element_type),
      ],
      operand_output_aliases={0: 0, 1: 1},
      backend_config=opaque,
  ).results[:1]


cuda_cholesky_update = partial(_cholesky_update_hlo, "cu", _cuda_linalg)
