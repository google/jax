# Copyright 2023 The JAX Authors.
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

"""Contains registrations for pallas_call on TPU."""

from __future__ import annotations

from typing import Any
import warnings

import jax
from jax import core as jax_core
from jax._src import core as jax_src_core
from jax._src import sharding_impls
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.pallas import core
from jax._src.pallas.mosaic import lowering
from jax._src.pallas.pallas_call import pallas_call_p
from jax.experimental import mosaic
from jax.experimental.mosaic.dialects import tpu


def pallas_call_tpu_lowering_rule(
    ctx: mlir.LoweringRuleContext, *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name: str,
    grid_mapping: core.GridMapping,
    input_output_aliases: tuple[tuple[int, int], ...],
    in_shapes: tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
    debug: bool,
    interpret: bool,
    compiler_params: dict[str, Any]):
  """Lowers a pallas_call to a Mosaic TPU custom call."""
  if interpret:
    return mlir.lower_fun(pallas_call_p.impl, multiple_results=True)(
        ctx, *in_nodes, jaxpr=jaxpr, name=name, out_shapes=out_shapes,
        in_shapes=in_shapes,
        interpret=interpret, debug=debug,
        input_output_aliases=input_output_aliases,
        grid_mapping=grid_mapping,
        compiler_params=compiler_params)
  if debug:
    print(jaxpr)
  if "mosaic_params" in compiler_params:
    # TODO(slebedev): Remove this branch after July 12th 2024.
    warnings.warn(
        "Passing Mosaic parameters via compiler_params=dict(mosaic_params=...)"
        " is deprecated. Use compiler_params=dict(mosaic=...) instead.",
        DeprecationWarning,
    )
    assert "mosaic" not in compiler_params
    mosaic_params = compiler_params["mosaic_params"]
  elif "mosaic" in compiler_params:
    mosaic_params = compiler_params["mosaic"]
  else:
    mosaic_params = {}
  mesh = None
  axis_context = ctx.module_context.axis_context
  if axis_context is not None:
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      mesh = axis_context.mesh
  with ir.Context() as mlir_ctx, ir.Location.unknown(mlir_ctx):
    mlir_ctx.append_dialect_registry(mlir.upstream_dialects)
    mlir_ctx.load_all_available_dialects()
    tpu.register_dialect(mlir_ctx)
    dimension_semantics = mosaic_params.get("dimension_semantics", None)
    mosaic_module, extra_args = lowering.lower_jaxpr_to_module(
        mlir_ctx, grid_mapping, in_shapes, out_shapes, jaxpr,
        dimension_semantics=dimension_semantics, mesh=mesh)
    if debug:
      print(mosaic_module)
  num_extra_args = len(extra_args)
  num_dyn_bounds = grid_mapping.num_dynamic_grid_bounds
  input_output_aliases = tuple(
      (a[0] + num_dyn_bounds + num_extra_args, a[1])
      for a in input_output_aliases
  )
  out_avals = [jax_core.ShapedArray(s.shape, s.dtype) for s in out_shapes]

  # Replace in_avals to physical avals.
  # This step is required for mapping logical types to physical types.
  # (e.g. PRNG key -> uint32[2])
  physical_avals = [jax_src_core.physical_aval(aval) for aval in ctx.avals_in]
  ctx = ctx.replace(avals_in=physical_avals)

  def _lower_fun(*args):
    # Dynamic grid bounds have to go at the front.
    dynamic_grid_args, args = args[:num_dyn_bounds], args[num_dyn_bounds:],
    return mosaic.as_tpu_kernel(
        mosaic_module,
        out_avals,
        backend="tpu",
        kernel_name=name,
        cost_estimate=mosaic_params.get("cost_estimate"),
        vmem_limit_bytes=mosaic_params.get("vmem_limit_bytes"),
        flags=mosaic_params.get("flags"),
        allow_input_fusion=mosaic_params.get("allow_input_fusion"),
        input_output_aliases=input_output_aliases,
        internal_scratch_in_bytes=mosaic_params.get(
            "internal_scratch_in_bytes"
        ),
    )(
        *dynamic_grid_args,
        *extra_args,
        *args,
        collective_id=mosaic_params.get("collective_id", None),
    )
  return mlir.lower_fun(_lower_fun, multiple_results=True)(
      ctx, *in_nodes)
