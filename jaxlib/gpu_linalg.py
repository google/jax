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
import operator

import jaxlib.mlir.ir as ir

from .hlo_helpers import custom_call
from .gpu_common_utils import GpuLibNotLinkedError

from jaxlib import xla_client

for cuda_module_name in [".cuda", "jax_cuda12_plugin", "jax_cuda11_plugin"]:
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
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")

try:
  from .rocm import _linalg as _hip_linalg  # pytype: disable=import-error
  for _name, _value in _hip_linalg.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  _hip_linalg = None

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def _lu_pivots_to_permutation_hlo(platform, gpu_linalg, pivots, *, permutation_size):
  """Kernel for the transformation of pivots to permutations on GPU."""
  typ = ir.RankedTensorType(pivots.type)
  dims = typ.shape
  i32_type = ir.IntegerType.get_signless(32)

  assert typ.element_type == i32_type, typ

  batch_size = _prod(dims[:-1])
  pivot_size = dims[-1]

  if not gpu_linalg:
    raise GpuLibNotLinkedError()

  opaque = gpu_linalg.lu_pivots_to_permutation_descriptor(
      batch_size, pivot_size, permutation_size)
  pivots_layout = tuple(range(len(dims) - 1, -1, -1))
  permutations_layout = pivots_layout
  permutations_dims = list(dims)
  permutations_dims[-1] = permutation_size
  permutations_type = ir.RankedTensorType.get(permutations_dims, i32_type)
  return custom_call(
      f"{platform}_lu_pivots_to_permutation",
      result_types=[permutations_type],
      operands=[pivots],
      backend_config=opaque,
      operand_layouts=[pivots_layout],
      result_layouts=[permutations_layout]).results

cuda_lu_pivots_to_permutation = partial(_lu_pivots_to_permutation_hlo, "cu",
                                        _cuda_linalg)
hip_lu_pivots_to_permutation = partial(
    _lu_pivots_to_permutation_hlo, "hip", _hip_linalg)
