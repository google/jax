# Copyright 2019 The JAX Authors.
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

import functools
from functools import partial
import importlib
import itertools
import operator

import jaxlib.mlir.ir as ir

from jaxlib import xla_client

from .hlo_helpers import custom_call
from .gpu_common_utils import GpuLibNotLinkedError

for cuda_module_name in [".cuda", "jax_cuda12_plugin", "jax_cuda11_plugin"]:
  try:
    _cuda_prng = importlib.import_module(
        f"{cuda_module_name}._prng", package="jaxlib"
    )
  except ImportError:
    _cuda_prng = None
  else:
    break

if _cuda_prng:
  for _name, _value in _cuda_prng.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")

try:
  from .rocm import _prng as _hip_prng  # pytype: disable=import-error
  for _name, _value in _hip_prng.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="ROCM")
except ImportError:
  _hip_prng = None

_prod = lambda xs: functools.reduce(operator.mul, xs, 1)

def _threefry2x32_lowering(prng, platform, keys, data,
                           length: int | ir.Value | None = None,
                           output_shape: ir.Value | None = None):
  """ThreeFry2x32 kernel for GPU.

  In presence of dynamic shapes, `length` is an `ir.Value` and `output_shape`
  is a 1D tensor describing the shape of the two outputs.
  """
  if not prng:
    raise GpuLibNotLinkedError()
  assert len(keys) == 2, keys
  assert len(data) == 2, data
  assert (ir.RankedTensorType(keys[0].type).element_type ==
          ir.IntegerType.get_unsigned(32)), keys[0].type

  typ = keys[0].type
  dims = ir.RankedTensorType(typ).shape

  for x in itertools.chain(keys, data):
    assert x.type == typ, (x.type, typ)
  ndims = len(dims)
  layout = tuple(range(ndims - 1, -1, -1))
  operand_layouts = [layout] * 4
  operands = [keys[0], keys[1], data[0], data[1]]

  if length is None:
    length = _prod(dims)

  if isinstance(length, int):
    opaque = prng.threefry2x32_descriptor(length)
    result_shapes = None
  else:
    assert output_shape is not None
    opaque = prng.threefry2x32_descriptor(-1)
    assert (ir.RankedTensorType(length.type).element_type ==
            ir.IntegerType.get_signless(64)), length
    assert (ir.RankedTensorType(length.type).shape ==
            [1]), (length, ir.RankedTensorType(length.type).shape)
    # Pass the length, which will be used by the custom call target since the
    # static length in the descriptor is -1.
    operands.append(length)
    operand_layouts.append((0,))
    # We also need to pass separately the shapes of the outputs.
    result_shapes = [output_shape, output_shape]

  return custom_call(
      f"{platform}_threefry2x32",
      result_types=[typ, typ],
      operands=operands,
      backend_config=opaque,
      operand_layouts=operand_layouts,
      result_layouts=[layout] * 2,
      result_shapes=result_shapes).results


cuda_threefry2x32 = partial(_threefry2x32_lowering, _cuda_prng, "cu")
rocm_threefry2x32 = partial(_threefry2x32_lowering, _hip_prng, "hip")
