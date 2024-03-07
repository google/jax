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

from typing import Any

import jax
from jax import core as jax_core
from jax._src.interpreters import mlir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.pallas.pallas_call import pallas_call_p
from jax.experimental.mosaic import gpu as mosaic_gpu


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr: jax_core.Jaxpr,
    name: str,
    in_shapes: tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    compiler_params: dict[str, Any],
):
  if interpret:
    return mlir.lower_fun(pallas_call_p.impl, multiple_results=True)(
        ctx,
        *args,
        jaxpr=jaxpr,
        name=name,
        out_shapes=out_shapes,
        in_shapes=in_shapes,
        interpret=interpret,
        debug=debug,
        input_output_aliases=input_output_aliases,
        grid_mapping=grid_mapping,
        compiler_params=compiler_params,
    )

  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "dynamic grid bounds not supported in the Mosaic GPU backend"
    )
  if input_output_aliases:
    raise NotImplementedError(
        "input_output_aliases not supported in the Mosaic GPU backend"
    )

  if debug:
    print(jaxpr)
    print(grid_mapping)

  lowering_result = lowering.lower_jaxpr_to_module(
      grid_mapping,
      in_shapes,
      out_shapes,
      jaxpr,
      name,
      compiler_params,
  )
  if debug:
    print(lowering_result.module.operation)

  return mosaic_gpu._mosaic_gpu_lowering_rule(
      ctx,
      *args,
      module=lowering_result.module,
      gmem_scratch_bytes=lowering_result.gmem_scratch_bytes,
      out_types=lowering_result.out_structs,
  )
