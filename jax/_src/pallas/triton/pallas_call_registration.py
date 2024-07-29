# Copyright 2024 The JAX Authors.
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

"""Module registering a lowering rule for pallas_call on GPU."""

from __future__ import annotations

import io
from typing import Any

from jax import core as jax_core
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.triton import lowering


def normalize_grid(grid: pallas_core.StaticGrid) -> tuple[int, int, int]:
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))  # type: ignore


def avals_to_layouts(avals):
  return [list(reversed(range(aval.ndim))) for aval in avals]


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name: str,
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    compiler_params: dict[str, Any],
):
  del interpret
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "dynamic grid bounds not supported in the Triton backend"
    )
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "scalar prefetch not implemented in the Triton backend"
    )
  triton_params = compiler_params.get("triton", compiler_params)
  num_warps = triton_params.pop("num_warps", 4)
  [lowering_platform] = ctx.platforms or ctx.module_context.platforms
  if lowering_platform == "rocm":
    num_stages = triton_params.pop("num_stages", 1)
  else:
    num_stages = triton_params.pop("num_stages", 3)

  if debug:
    print(jaxpr)
    print(grid_mapping)

  lowering_result = lowering.lower_jaxpr_to_triton_module(
      jaxpr, grid_mapping, name, lowering_platform
  )
  module_op = lowering_result.module.operation
  if debug:
    print(module_op.get_asm(enable_debug_info=True, pretty_debug_info=True))

  grid_x, grid_y, grid_z = normalize_grid(lowering_result.grid)
  out_types = [
      ir.RankedTensorType.get(bm.array_shape_dtype.shape,
                              mlir.dtype_to_ir_type(bm.array_shape_dtype.dtype))
      for bm in grid_mapping.block_mappings_output
  ]
  buf = io.BytesIO()
  module_op.write_bytecode(buf)
  backend_config = dict(
      name=ir.StringAttr.get(name),
      ir=ir.StringAttr.get(buf.getvalue()),
      num_stages=mlir.i32_attr(num_stages),
      num_warps=mlir.i32_attr(num_warps),
      grid_x=mlir.i32_attr(grid_x),
      grid_y=mlir.i32_attr(grid_y),
      grid_z=mlir.i32_attr(grid_z),
      debug=ir.BoolAttr.get(debug),
  )
  if "serialized_metadata" in (triton_params or {}):
    # This field is unstable and may be removed in the future.
    backend_config["serialized_metadata"] = ir.StringAttr.get(
        triton_params["serialized_metadata"]
    )
  return mlir.custom_call(
      call_target_name="__gpu$xla.gpu.triton",
      result_types=out_types,
      operands=in_nodes,
      backend_config=backend_config,
      api_version=4,
      operand_layouts=avals_to_layouts(ctx.avals_in),
      result_layouts=avals_to_layouts(ctx.avals_out),
      operand_output_aliases=dict(input_output_aliases),
  ).results
