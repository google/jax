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

"""JAX bindings for Mosaic."""

# mypy: ignore-errors
from __future__ import annotations

import base64
import collections.abc
from collections.abc import Callable, Sequence
import dataclasses
import functools
import io
import os
import time
from typing import Any

import jax
from jax import core
from jax._src import config
from jax._src import sharding_impls
from jax._src.interpreters import mlir
from jax._src.lib import tpu
from jax._src.lib import xla_client
from jax.interpreters import xla
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import mhlo
from jaxlib.mlir.passmanager import PassManager

try:
  from absl import flags
  FLAGS = flags.FLAGS
except ImportError:
  FLAGS = {}

_MOSAIC_USE_PYTHON_PIPELINE = config.bool_state(
    name="mosaic_use_python_pipeline",
    default=False,
    help=(
        "Run the initial Mosaic MLIR passes from Python, when as_tpu_kernel"
        " is called (for Pallas, this happens at JAX lowering time), instead of"
        " later within XLA."
    ),
)

_MOSAIC_ALLOW_HLO = config.bool_state(
    name="jax_mosaic_allow_hlo",
    default=False,
    help="Allow hlo dialects in Mosaic",
)

tpu_custom_call_p = core.Primitive("tpu_custom_call")
tpu_custom_call_p.def_impl(
    functools.partial(xla.apply_primitive, tpu_custom_call_p))
tpu_custom_call_p.multiple_results = True


@dataclasses.dataclass(frozen=True)
class CostEstimate:
  flops: int
  transcendentals: int
  bytes_accessed: int

  def to_json(self) -> bytes:
    return (
        f'{{"flops": {self.flops}, "transcendentals": {self.transcendentals},'
        f' "bytes_accessed": {self.bytes_accessed}}}'
    ).encode('ascii')


@dataclasses.dataclass(frozen=True)
class CustomCallBackendConfig:
  """Represents an unserialized backend config for custom calls."""
  lowered_module_asm: bytes
  has_communication: bool
  collective_id: int | None
  device_type: str | None
  cost_estimate: CostEstimate | None
  needs_hlo_passes: bool
  needs_layout_passes: bool
  vmem_limit_bytes: int | None
  flags: dict[str, bool | int | float] | None
  allow_input_fusion: list[bool] | None
  serialization_format: int | None
  internal_scratch_in_bytes: int | None

  # We omit the body while printing, because primitive params get embedded
  # in HLO metadata, and the body blows up its size.
  def __repr__(self):
    return "CustomCallBackendConfig(<omitted>)"

  def to_json(self) -> bytes:
    """Serializes the backend config into JSON."""
    # We format the JSON ourselves, because json.dumps seems to be overly slow.
    config = io.BytesIO()
    config.write(b'{"custom_call_config": {"body": "')
    config.write(base64.b64encode(self.lowered_module_asm))
    config.write(b'"')
    if self.has_communication:
      config.write(b', "has_communication": ')
      config.write(str(self.has_communication).lower().encode("ascii"))
    if self.collective_id is not None:
      config.write(b', "collective_id": ')
      config.write(str(self.collective_id).encode("ascii"))
    if self.cost_estimate is not None:
      config.write(b', "cost_estimate": ')
      config.write(self.cost_estimate.to_json())
    if self.needs_hlo_passes:
      config.write(b', "needs_hlo_passes": ')
      config.write(str(self.needs_hlo_passes).lower().encode("ascii"))
    if self.serialization_format is not None:
      config.write(b', "serialization_format": ')
      config.write(str(self.serialization_format).lower().encode("ascii"))
    if self.needs_layout_passes:
      config.write(b', "needs_layout_passes": ')
      config.write(str(self.needs_layout_passes).lower().encode("ascii"))
    if self.allow_input_fusion is not None:
      config.write(b', "allow_input_fusion": [')
      for i, value in enumerate(self.allow_input_fusion):
        config.write(b"true" if value else b"false")
        # config.write(str(value).lower().encode("ascii"))
        if i + 1 != len(self.allow_input_fusion):
          config.write(b",")
      config.write(b"]")
    if self.internal_scratch_in_bytes is not None:
      config.write(b', "internal_scratch_in_bytes": ')
      config.write(str(self.internal_scratch_in_bytes).encode("ascii"))
    config.write(b"}")  # End of custom_call_config.
    if self.device_type is not None:
      config.write(b', "device_type": ')
      config.write(
          ('"DEVICE_TYPE_' + self.device_type.upper() + '"').encode("ascii")
      )
    if self.vmem_limit_bytes is not None:
      config.write(
          b', "scoped_memory_configs": [{"memory_space":1, "offset": 0,'
          b' "size": '
      )
      config.write(str(self.vmem_limit_bytes).encode("ascii"))
      config.write(b'}]')
    if self.flags is not None:
      config.write(b', "flag_configs": [')
      for i, (flag, value) in enumerate(self.flags.items()):
        config.write(b'{"flag_type": "')
        config.write(flag.encode("ascii"))
        config.write(b'", value: {')
        if isinstance(value, bool):
          config.write(b'"boolean_value": ')
          config.write(b"true" if value else b"false")
        elif isinstance(value, int):
          config.write(b'"integer_value": ')
          config.write(str(value).encode("ascii"))
        elif isinstance(value, float):
          config.write(b'"double_value": ')
          config.write(str(value).encode("ascii"))
        else:
          raise ValueError("invalid flag value: " + str(value))
        config.write(b"}}")
        if i + 1 != len(self.flags):
          config.write(b",")
      config.write(b"]")
    # Prevent the compiler from sharding the custom call beyond what Mosaic does
    # based on user annotations
    config.write(b', "implicit_sharding": {"type": "MANUAL"}')
    config.write(b"}")
    return config.getvalue()


@tpu_custom_call_p.def_abstract_eval
def _tpu_custom_call_abstract_eval(*_, out_avals, **__):
  return out_avals


def _avals_to_layouts(avals) -> Sequence[Sequence[int]]:
  return [tuple(range(a.ndim - 1, -1, -1)) for a in avals]


def _tpu_custom_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,  # pylint: disable=missing-function-docstring
    config: CustomCallBackendConfig,
    kernel_name: str | None,
    out_avals: Any,
    input_output_aliases: tuple[tuple[int, int], ...],
) -> ...:
  result_types = [mlir.aval_to_ir_type(aval) for aval in out_avals]
  axis_context = ctx.module_context.axis_context
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "Mosaic kernels cannot be automatically partitioned. Please wrap the"
          " call in a shard_map."
      )
  elif isinstance(axis_context, sharding_impls.ShardingContext):
    if axis_context.num_devices != 1:
      raise NotImplementedError(
          "Mosaic kernels cannot be automatically partitioned. Please wrap the"
          " call in a shard_map."
      )
  elif config.has_communication:
    raise NotImplementedError(
        "Replica lowering for Mosaic kernels not implemented."
    )
  if all(core.is_constant_shape(aval_out.shape) for aval_out in ctx.avals_out):
    result_shapes = None
  else:
    result_shapes = [
        mlir.shape_tensor(mlir.eval_dynamic_shape(ctx, aval_out.shape))
        for aval_out in ctx.avals_out]
  extra_attributes = None
  # Add kernel_name and kernel_metadata as attributes to the custom call op.
  # This is because we do not want to pollute the backend_config with this
  # information.
  if kernel_name is not None:
    extra_attributes = dict(kernel_name=ir.StringAttr.get(kernel_name))
  call = mlir.custom_call(
      "tpu_custom_call",
      result_types=result_types,
      operands=in_nodes,
      backend_config=config.to_json(),
      api_version=1,
      operand_output_aliases=dict(input_output_aliases),
      operand_layouts=_avals_to_layouts(ctx.avals_in),
      result_layouts=_avals_to_layouts(ctx.avals_out),
      result_shapes=result_shapes,
      extra_attributes=extra_attributes)

  return call.results


mlir.register_lowering(tpu_custom_call_p, _tpu_custom_call_lowering,
                       platform="tpu")


def _lower_tpu_kernel(
    module: ir.Module,
    hardware_generation: int,
) -> ir.Module:
  """Runs MLIR passes lowering the given module to an MLIR module.

  Uses Python versions of canonicalize-mosaic,infer-memref-layout and
    apply-vector-layout.

  Args:
    module: The MLIR module to lower.
    hardware_generation: The TPU hardware generation to target.

  Returns:
    An MLIR module implementing the kernel.
  """
  try:
    module.operation.verify()
  except ir.MLIRError as e:
    raise ValueError("The compiled module fails MLIR verification") from e

  with module.context as ctx, module.operation.location as _:
    ctx.append_dialect_registry(mlir.upstream_dialects)
    ctx.load_all_available_dialects()
    tpu.register_dialect(ctx)
    mhlo.register_mhlo_dialect(ctx)
    mhlo.register_mhlo_passes()

    dump_mlir(module, "original")

    if _MOSAIC_ALLOW_HLO.value:
      # Run hlo dialect conversion: hlo -> linalg -> vector.
      pipeline = [
          "hlo-legalize-to-arithmetic",
          "func.func(hlo-legalize-to-linalg)",
          "func.func(linalg-vectorization)",
      ]
      pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
      pipeline.run(module.operation)
      dump_mlir(module, "post-hlo-conversion")

    # Note: we don't pass the TpuTilingFlags here, since we don't know the
    # tiling decisions made by the compiler / what flags are enabled at this
    # point, so we assume everything can be tiled up to default tiling.
    pipeline = [
        f"func.func(tpu-infer-memref-layout{{hardware-generation={hardware_generation}}})"
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-infer-memref-layout")

    pipeline = [
        "canonicalize",
        "cse",
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-infer-memref-layout-simplify")

    try:
      on_device_checks = FLAGS["xla_mosaic_on_device_checks"].value
    except KeyError:
      on_device_checks = False

    if checks := on_device_checks:
      checks = set(checks.split(","))
      if checks == {"bounds"}:  # We only support one kind of checks now.
        pipeline = PassManager.parse(
            "builtin.module(func.func(debug-assert-insertion))"
        )
        pipeline.run(module.operation)
        dump_mlir(module, "post-assert-insertion")
      elif checks:
        checks.discard("bounds")
        raise ValueError(
            f"Unrecognized on-device check categories: {', '.join(checks)}"
        )

    pipeline = [
        f"func.func(tpu-canonicalize-mosaic{{hardware-generation={hardware_generation}}})",
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-canonicalize-mosaic")

    pipeline = [
        "func.func(tpu-infer-vector-layout{sublane-count=8 lane-count=128})",
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-infer-vector-layout")

    sl_cnt = 8
    l_cnt = 128
    mxu_size = 128 if hardware_generation < 6 else 256
    pipeline = [
        "func.func(tpu-apply-vector-layout{"
        f" sublane-count={sl_cnt} lane-count={l_cnt}"
        f" hardware-generation={hardware_generation}"
        f" mxu-contracting-size={mxu_size} mxu-noncontracting-size={mxu_size}"
        f" max-sublanes-in-scratch={sl_cnt * (sl_cnt + 1)}"
        "})"
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-apply-vector-layout")

    pipeline = [
        "canonicalize",
        "cse",
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-apply-vector-layout-simplify")

    return module


def _lower_mosaic_module_to_asm(
    module: ir.Module,
    *,
    backend: str,
    device_type: str | None,
) -> tuple[ir.Module, tuple[bool, bool, bool, bool]]:
  has_communication, has_custom_barrier = tpu.private_has_communication(
      module.operation
  )
  needs_hlo_passes = _MOSAIC_ALLOW_HLO.value
  needs_layout_passes = not device_type
  # We'll mutate the module, so clone it
  with module.context as ctx, module.operation.location as _:
    module = ir.Module.parse(
        module.operation.get_asm(binary=True, enable_debug_info=True)
    )
    if needs_layout_passes and _MOSAIC_USE_PYTHON_PIPELINE.value:
      some_tpu = jax.devices(backend)[0]
      device_kind = some_tpu.device_kind
      if not device_kind.startswith("TPU v"):
        raise ValueError(
            f"Unrecognized TPU device kind: {device_kind}. "
            "tpu_custom_call cannot be lowered on a machine without TPUs "
            "when mosaic_use_python_pipeline=True.")
      hardware_generation = int(device_kind[len("TPU v")])
      module = _lower_tpu_kernel(module, hardware_generation)
      needs_hlo_passes = False
      needs_layout_passes = False
    prev_allow_unregistered_dialects = ctx.allow_unregistered_dialects
    ctx.allow_unregistered_dialects = True
    try:
      pipeline = PassManager.parse("builtin.module(mosaic-serde{serialize=true})")
      pipeline.run(module.operation)
    finally:
      ctx.allow_unregistered_dialects = prev_allow_unregistered_dialects
    bytecode_buffer = io.BytesIO()
    module.operation.write_bytecode(bytecode_buffer, desired_version=0)
    asm = bytecode_buffer.getvalue()
    return asm, (
        has_communication,
        has_custom_barrier,
        needs_hlo_passes,
        needs_layout_passes,
    )


def _lower_to_custom_call_config(
    module: ir.Module,
    *,
    backend: str,
    device_type: str | None,
    vmem_limit_bytes: int | None,
    cost_estimate: CostEstimate | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: list[bool] | None,
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    serialization_format: int | None,
) -> CustomCallBackendConfig:
  lowered_module_asm, (
      has_communication,
      has_custom_barrier,
      needs_hlo_passes,
      needs_layout_passes,
  ) = _lower_mosaic_module_to_asm(
      module,
      backend=backend,
      device_type=device_type,
  )
  return _lowered_to_custom_call_config(
      lowered_module_asm,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      device_type=device_type,
      serialization_format=serialization_format,
      has_custom_barrier=has_custom_barrier,
      has_communication=has_communication,
      needs_hlo_passes=needs_hlo_passes,
      needs_layout_passes=needs_layout_passes,
  )


def _lowered_to_custom_call_config(
    lowered_module_asm: bytes,
    *,
    vmem_limit_bytes: int | None,
    cost_estimate: CostEstimate | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: list[bool] | None,
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    serialization_format: int | None,
    has_custom_barrier: bool,
    has_communication: bool,
    needs_hlo_passes: bool,
    needs_layout_passes: bool,
    device_type: str | None,
):
  if has_custom_barrier:
    if collective_id is None:
      raise ValueError(
          "collective_id has to be specified when using a custom barrier"
      )
  elif collective_id is not None:
    raise ValueError(
        "collective_id has to be unspecified or None when not using a custom"
        " barrier"
    )
  if vmem_limit_bytes is not None and not isinstance(vmem_limit_bytes, int):
    raise ValueError(
        "vmem_limit_bytes must be an int: provided with a"
        f" {type(vmem_limit_bytes)}."
    )
  config = CustomCallBackendConfig(
      lowered_module_asm,
      has_communication,
      collective_id,
      device_type,
      cost_estimate,
      needs_hlo_passes,
      needs_layout_passes,
      vmem_limit_bytes,
      flags,
      allow_input_fusion,
      serialization_format,
      internal_scratch_in_bytes,
  )
  return config


def lower_module_to_custom_call(
    ctx: mlir.LoweringRuleContext,
    *in_nodes: ir.Value,
    module: ir.Module,
    out_type: Any,
    backend: str,
    kernel_name: str,
    cost_estimate: CostEstimate | None,
    vmem_limit_bytes: int | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: list[bool] | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    serialization_format: int | None,
    device_type: str | None,
) -> Sequence[ir.Value]:
  config = _lower_to_custom_call_config(
      module,
      backend=backend,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      device_type=device_type,
      serialization_format=serialization_format,
  )
  return _tpu_custom_call_lowering(
      ctx,
      *in_nodes,
      config=config,
      kernel_name=kernel_name,
      out_avals=out_type,
      input_output_aliases=input_output_aliases,
  )


def as_tpu_kernel(
    module: ir.Module,
    out_type: Any,
    *,
    cost_estimate: CostEstimate | None = None,
    backend: str | xla_client.Client = "tpu",
    device_type: str | None = None,
    kernel_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    allow_input_fusion: list[bool] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    internal_scratch_in_bytes: int | None = None,
    collective_id: int | None = None,
    serialization_format: int | None = 1,
) -> Callable[..., Any]:
  """Turns an MLIR Mosaic kernel into a JAX-compatible function."""
  config = _lower_to_custom_call_config(
      module,
      backend=backend,
      device_type=device_type,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      serialization_format=serialization_format,
  )
  return _as_jax_callable(
      config,
      out_type,
      kernel_name=kernel_name,
      input_output_aliases=input_output_aliases,
  )


def lowered_as_tpu_kernel(
    lowered_module: ir.Module,
    out_type: Any,
    *,
    collective_id: int | None = None,
    cost_estimate: CostEstimate | None = None,
    needs_hlo_passes: bool = False,
    needs_layout_passes: bool = False,
    device_type: str | None = None,
    has_communication: bool = False,
    has_custom_barrier: bool = False,
    kernel_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    allow_input_fusion: list[bool] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    serialization_format: int | None = None,
    internal_scratch_in_bytes: int | None = None,
) -> Callable[..., Any]:
  lowered_module_asm = lowered_module.operation.get_asm(
      binary=True, enable_debug_info=True
  )
  config = _lowered_to_custom_call_config(
      lowered_module_asm,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      device_type=device_type,
      serialization_format=serialization_format,
      has_custom_barrier=has_custom_barrier,
      has_communication=has_communication,
      needs_hlo_passes=needs_hlo_passes,
      needs_layout_passes=needs_layout_passes,
  )
  return _as_jax_callable(
      config,
      out_type,
      kernel_name=kernel_name,
      input_output_aliases=input_output_aliases,
  )


def _as_jax_callable(
    config: CustomCallBackendConfig,
    out_type: Any,
    *,
    kernel_name: str | None,
    input_output_aliases: tuple[tuple[int, int], ...],
) -> Callable[..., Any]:
  unpack = False
  if not isinstance(out_type, collections.abc.Iterable):
    out_type = (out_type,)
    unpack = True
  out_avals = tuple(core.ShapedArray(ty.shape, ty.dtype) for ty in out_type)

  # We use jax.jit to make sure we hit the fast compilation cache.
  def apply_kernel(*args):
    result = tpu_custom_call_p.bind(
        *args,
        config=config,
        kernel_name=kernel_name,
        out_avals=out_avals,
        input_output_aliases=input_output_aliases,
    )
    return result[0] if unpack else result

  return jax.jit(apply_kernel)


def dump_mlir(module: ir.Module, name: str):
  """A helper function to dump mosaic mlir module"""
  try:
    should_dump = FLAGS["xla_mosaic_dump_to"].value
  except KeyError:
    return
  if should_dump == "sponge":
    outdir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", None)
    if outdir:
      path = os.path.join(outdir, f"{time.time_ns()}-mosaic-dump-{name}-py.txt")
      with open(path, "w") as f:
        f.write(str(module))
