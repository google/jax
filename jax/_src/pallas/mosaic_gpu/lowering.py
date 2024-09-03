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

"""Module for lowering JAX primitives to Mosaic GPU."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import itertools
import math
from typing import Any, cast

import jax
from jax._src import core as jax_core
from jax._src import pjit
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import nvvm as nvvm_dialect
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
from jax._src.lib.mlir.dialects import memref as memref_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.pallas.mosaic_gpu import core as mgpu_core
from jax._src.pallas.mosaic_gpu import primitives as mgpu_primitives
from jax._src.state import primitives as sp
from jax.experimental.mosaic import gpu as mosaic_gpu
from jax.experimental.mosaic.gpu import dsl as mgpu
import jax.numpy as jnp
import numpy as np


# TODO(slebedev): Enable type checking.
# mypy: ignore-errors
# pytype: skip-file

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

partial = functools.partial


@dataclasses.dataclass
class ModuleContext:
  name: str
  grid_mapping: pallas_core.GridMapping
  runtime_smem: ir.Value  # ir.MemRefType
  smem_used_bytes: int

  # TODO(cperivol): Only return the shapes and figure out the sizes when freeing.
  def scratch_view(
      self, structs: Sequence[jax.ShapeDtypeStruct]
  ) -> tuple[int, Sequence[ir.Value]]:
    """Creates a view into the runtime scratch buffer for each struct.

    This is a low-level API. Use it only if you know what you are doing.

    The function allocates bytes at the top of a stack, which need to be
    deallocated in a FIFO fashion with :meth:`ModuleContext.stack_free_smem`.
    After deallocation, the view is invalid and cannot be used.

    Args:
      structus: The shapes and dtypes of the views to create.

    Returns:
      A tuple, where the first element is the number of bytes allocated,
      and the second element is a sequence of memref views into the
      runtime scratch buffer.
    """
    smem_scratch_bytes = math.prod(ir.MemRefType(self.runtime_smem.type).shape)
    required_scratch_bytes = sum(
        math.prod(sh.shape) * jnp.dtype(sh.dtype).itemsize for sh in structs
    )
    if smem_scratch_bytes < required_scratch_bytes:
      raise ValueError(
          f"Too few {smem_scratch_bytes=} provided (pass via compiler_params),"
          f" we need {required_scratch_bytes} ({structs=})"
      )

    views = []
    off = self.smem_used_bytes
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    for s in structs:
      scratch_ty = ir.MemRefType.get(
          s.shape,
          mlir.dtype_to_ir_type(s.dtype),
          memory_space=smem,
      )
      views.append(
          memref_dialect.view(scratch_ty, self.runtime_smem, _as_index(off), [])
      )
      off += math.prod(s.shape) * jnp.dtype(s.dtype).itemsize

    total_bytes = off - self.smem_used_bytes
    self.smem_used_bytes = off
    return total_bytes, views

  def stack_free_smem(self, bytes: int):
    """Frees the ``bytes`` last allocated."""
    if bytes > self.smem_used_bytes:
      raise ValueError("Tried to free more bytes than was allocated")
    self.smem_used_bytes -= bytes


@dataclasses.dataclass(frozen=True)
class LoweringRuleContext:
  module_context: ModuleContext
  avals_in: Sequence[jax_core.ShapedArray]
  avals_out: Sequence[jax_core.ShapedArray]

  replace = dataclasses.replace


@dataclasses.dataclass(frozen=True)
class LoweringResult:
  module: ir.Module
  grid: tuple[int, ...]
  gmem_scratch_bytes: int
  out_structs: tuple[jax.ShapeDtypeStruct, ...]


class LoweringError(Exception):  # pylint: disable=g-bad-exception-name
  pass


def _eval_index_map(
    ctx: ModuleContext, idx, block_mapping: pallas_core.BlockMapping
) -> Sequence[ir.Value]:
  block_indices = lower_jaxpr_to_mosaic_gpu(
      ctx, block_mapping.index_map_jaxpr.jaxpr, idx
  )
  result = []
  for i, b in zip(block_indices, block_mapping.block_shape):
    if b is pallas_core.mapped:
      result.append(i)
    else:
      # TODO(slebedev): Use a type-agnostic multiplication wrapper.
      result.append(arith_dialect.muli(_as_index(i), _as_index(b)))
  return tuple(result)


@dataclasses.dataclass(frozen=True)
class SmemBlockInfo:
  tiled_shape: tuple[int, ...]
  untiled_shape: tuple[int, ...]
  dtype: jnp.dtype
  swizzle: int | None = None
  tma_transpose: bool = False

  @property
  def ndim(self) -> int:
    return len(self.untiled_shape)

  @property
  def byte_size(self) -> int:
    return math.prod(self.untiled_shape) * self.dtype.itemsize

  @property
  def gmem_transform(self) -> mgpu.MemRefTransform | tuple[mgpu.MemRefTransform, ...]:
    if len(self.tiled_shape) == len(self.untiled_shape):
      return ()

    transform = (mosaic_gpu.TileTransform(self.tiled_shape[2:]),)
    if self.tma_transpose:
      transform += (mosaic_gpu.TransposeTransform((1, 0, 2, 3)),)
    return transform

  def gmem_slice(self, *dim_start: ir.Value) -> tuple[mgpu.DynamicSlice, ...]:
    gslice = map(mgpu.ds, dim_start, self.untiled_shape)
    if self.tma_transpose:
      return tuple(gslice[::-1])
    return tuple(gslice)

  @classmethod
  def from_block_mapping(cls, bm):
    memspace = bm.block_aval.memory_space or mgpu_core.SMemSpace(None)
    if not isinstance(memspace, mgpu_core.SMemSpace):
      raise ValueError(f"Can only pre-load shared memory blocks. Not {bm.block_aval.memory_space}")

    if memspace.wgmma_operand_config:
      cfg = bm.block_aval.memory_space.wgmma_operand_config
      return cls(
          tiled_shape=cfg.tiled_shape,
          untiled_shape=cfg.shape,
          swizzle=cfg.swizzle,
          tma_transpose=cfg.tma_transpose,
          dtype=bm.array_shape_dtype.dtype
      )


    return cls(tiled_shape=bm.block_shape, untiled_shape=bm.block_shape, dtype=bm.array_shape_dtype.dtype)

def lower_jaxpr_to_module(
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    name_and_src_info: pallas_core.NameAndSrcInfo,
    compiler_params: dict[str, Any],
    cost_estimate: pallas_core.CostEstimate | None,
) -> LoweringResult:
  del cost_estimate  # Unused.

  block_infos = [SmemBlockInfo.from_block_mapping(bm) for bm in grid_mapping.block_mappings]
  in_block_infos = block_infos[:grid_mapping.num_inputs]
  out_block_infos = block_infos[grid_mapping.num_inputs:]
  assert len(jaxpr.outvars) == 0
  assert not grid_mapping.vmapped_dims
  if len(grid_mapping.grid) > 3:
    raise NotImplementedError(
        "Only <=3D grids are supported in Mosaic GPU lowering."
    )
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "Dynamic grid bounds not supported in the Mosaic GPU lowering."
    )
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "Scalar prefetch not supported in Mosaic GPU lowering."
    )
  if not all(
      isinstance(bm.indexing_mode, pallas_core.Blocked)
      for bm in grid_mapping.block_mappings
  ):
    raise NotImplementedError(
        "Only Blocked indexing mode is supported in Mosaic GPU lowering."
    )

  with grid_mapping.trace_env():
    jaxpr, _ = pe.dce_jaxpr(
        jaxpr, [True] * len(jaxpr.outvars), instantiate=True
    )

  grid = grid_mapping.grid
  if len(grid) < 3:
    grid += (1,) * (3 - len(grid))
  block = (128,) + (1,) * (len(grid) - 1)
  in_structs_gmem = [*grid_mapping.in_shapes]
  in_structs_smem = [
      jax.ShapeDtypeStruct(bi.tiled_shape, s.dtype)
      for bi, s in zip(in_block_infos, grid_mapping.in_shapes,)
  ]
  out_structs_gmem = [*grid_mapping.out_shapes]
  out_structs_smem = [
      jax.ShapeDtypeStruct(bi.tiled_shape, s.dtype)
      for bi, s in zip(
          out_block_infos,
          grid_mapping.out_shapes,
      )
  ]

  def body(launch_ctx: mosaic_gpu.LaunchContext, *buffers):
    *buffers_gmem, (*buffers_smem, barrier, runtime_smem) = buffers
    assert len(buffers_gmem) == len(buffers_smem)
    in_buffers_gmem, out_buffers_gmem = util.split_list(
        buffers_gmem, [grid_mapping.num_inputs]
    )
    in_buffers_smem, out_buffers_smem = util.split_list(
        buffers_smem, [grid_mapping.num_inputs]
    )

    barrier = cast(mgpu.BarrierRef, barrier)

    module_ctx = ModuleContext(
        name_and_src_info.name, grid_mapping, runtime_smem, smem_used_bytes=0
    )
    program_ids = map(_program_id, range(len(grid_mapping.grid)))
    start_indices = map(
        functools.partial(_eval_index_map, module_ctx, program_ids),
        grid_mapping.block_mappings,
    )
    in_start_indices, out_start_indices = util.split_list(
        start_indices, [grid_mapping.num_inputs]
    )

    txcount = sum(bi.byte_size for bi in in_block_infos)

    with mgpu.single_thread():
      barrier.arrive_expect_tx(txcount)
      for start_indices, b_gmem, b_smem, bi in unsafe_zip(
          in_start_indices, in_buffers_gmem, in_buffers_smem, in_block_infos
      ):
        launch_ctx.async_copy(
            src_ref=b_gmem,
            dst_ref=b_smem,
            gmem_slice=bi.gmem_slice(*start_indices),
            gmem_transform=bi.gmem_transform,
            barrier=barrier,
            swizzle=bi.swizzle,
            arrive=False,
            uniform=False,
        )

    if grid_mapping.num_inputs:
      # Only wait if async copies were issued.
      barrier.wait()

    _ = lower_jaxpr_to_mosaic_gpu(module_ctx, jaxpr, buffers_smem)
    mgpu.commit_shared()


    for start_indices, b_gmem, b_smem, bi in zip(
        out_start_indices, out_buffers_gmem, out_buffers_smem, out_block_infos
    ):
      # TODO(slebedev): Support 128-byte swizzling, once we can lower matmuls.
      launch_ctx.async_copy(
          src_ref=b_smem,
          dst_ref=b_gmem,
          gmem_slice=tuple(map(mgpu.ds, start_indices, bi.untiled_shape)),
          gmem_transform=bi.gmem_transform,
          swizzle=bi.swizzle,
      )

    launch_ctx.await_async_copy(0)

  # TODO(b/354568888): Add a jaxpr traversal to calculate the precise
  # amount of memory required.
  extra_smem_scratch = [
      jax.ShapeDtypeStruct(
          shape=[compiler_params.get("smem_scratch_bytes", 100000)],
          dtype=np.int8,
      )
  ]
  module, out_structs_smem, gmem_scratch_bytes, _ = (
      mosaic_gpu._lower_as_gpu_kernel(
          body,
          grid=grid,
          cluster=(),
          block=block,
          in_shapes=in_structs_gmem,
          out_shape=out_structs_gmem,
          smem_scratch_shape=(
              *in_structs_smem,
              *out_structs_smem,
              mgpu.TMABarrier(),
              *extra_smem_scratch,
          ),
          module_name=name_and_src_info.name,
      )
  )

  return LoweringResult(module, grid, gmem_scratch_bytes, out_structs_smem)


mosaic_lowering_rules = {}


def register_lowering_rule(primitive: jax_core.Primitive):
  def deco(fn):
    mosaic_lowering_rules[primitive] = fn
    return fn

  return deco


def lower_jaxpr_to_mosaic_gpu(
    ctx: ModuleContext,
    jaxpr: jax_core.Jaxpr,
    args: Sequence[ir.Value],
    consts=(),
) -> Sequence[ir.Value]:
  env = {}

  def read_env(atom: jax_core.Atom):
    return atom.val if isinstance(atom, jax_core.Literal) else env[atom]

  def write_env(var: jax_core.Var, val):
    env[var] = val

  map(write_env, jaxpr.constvars, consts)
  map(write_env, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    if eqn.primitive not in mosaic_lowering_rules:
      raise NotImplementedError(
          "Unimplemented primitive in Pallas Mosaic GPU lowering: "
          f"{eqn.primitive.name}. "
          "Please file an issue on https://github.com/google/jax/issues."
      )
    rule = mosaic_lowering_rules[eqn.primitive]
    rule_ctx = LoweringRuleContext(
        ctx,
        avals_in=[cast(jax_core.ShapedArray, v.aval) for v in eqn.invars],
        avals_out=[cast(jax_core.ShapedArray, v.aval) for v in eqn.outvars],
    )
    try:
      outvals = rule(rule_ctx, *invals, **eqn.params)
    except LoweringError:
      raise  # We only add the extra info to the innermost exception.
    except Exception as e:
      inval_types = map(lambda t: getattr(t, "type", None), invals)
      raise LoweringError(
          f"Exception while lowering eqn:\n  {eqn}\nWith context:\n "
          f" {rule_ctx}\nWith inval types={inval_types}\nIn jaxpr:\n{jaxpr}"
      ) from e
    if eqn.primitive.multiple_results:
      map(write_env, eqn.outvars, outvals)
    else:
      write_env(eqn.outvars[0], outvals)
  return map(read_env, jaxpr.outvars)


@register_lowering_rule(primitives.program_id_p)
def _program_id_lowering_rule(ctx: LoweringRuleContext, axis):
  del ctx  # Unused.
  return _program_id(axis)


def _program_id(axis: int) -> ir.Value:
  return arith_dialect.index_cast(
      ir.IntegerType.get_signless(32),
      gpu_dialect.block_id(gpu_dialect.Dimension(axis)),
  )


@register_lowering_rule(primitives.num_programs_p)
def _num_programs_lowering_rule(ctx: LoweringRuleContext, axis):
  del ctx  # Unused.
  return arith_dialect.index_cast(
      ir.IntegerType.get_signless(32),
      gpu_dialect.block_dim(gpu_dialect.Dimension(axis)),
  )


@register_lowering_rule(sp.get_p)
def _get_lowering_rule(ctx: LoweringRuleContext, x_smem, *indexers, tree):
  del tree  # Unused.
  if indexers:
    raise NotImplementedError("No support for indexers yet")

  [out_aval] = ctx.avals_out
  if isinstance(x_smem, mgpu.WGMMAAccumulator):
    return x_smem

  return mgpu.FragmentedArray.load_strided(x_smem)


@register_lowering_rule(sp.swap_p)
def _swap_lowering_rule(
    ctx: LoweringRuleContext, x_smem, value, *indexers, tree
):
  del tree  # Unused.
  if indexers:
    raise NotImplementedError("No support for indexers yet")

  if isinstance(value, mgpu.WGMMAAccumulator):
    value = value.value

  if value.layout == mgpu.WGMMA_LAYOUT:
    assert len(ctx.avals_in) == 2, value
    aref = cast(pallas_core.AbstractMemoryRef, ctx.avals_in[0])
    assert isinstance(aref.memory_space, mgpu_core.SMemSpace) and aref.memory_space.wgmma_operand_config, ctx.avals_in
    value.store_tiled(x_smem, swizzle=aref.memory_space.wgmma_operand_config.swizzle)
    return

  old_value = mgpu.FragmentedArray.load_strided(x_smem)
  value.store_untiled(x_smem)
  return old_value


@register_lowering_rule(pjit.pjit_p)
def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_mosaic_gpu(ctx.module_context, jaxpr.jaxpr, args)


@register_lowering_rule(lax.broadcast_in_dim_p)
def _broadcast_in_dim_lowering_rule(
    ctx: LoweringRuleContext,
    x: mgpu.FragmentedArray,
    *,
    broadcast_dimensions,
    shape,
):
  if broadcast_dimensions:
    raise NotImplementedError
  return _ensure_fa(x, ctx.avals_in[0]).broadcast(shape)


@register_lowering_rule(lax.convert_element_type_p)
def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  del weak_type, sharding
  return _ensure_fa(x, *ctx.avals_in).astype(mlir.dtype_to_ir_type(new_dtype))


def _binary_op_lowering_rule(ctx: LoweringRuleContext, x, y, *, impl):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return impl(x, y)


mosaic_lowering_rules.update({
    lax.add_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x + y),
    lax.sub_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x - y),
    lax.mul_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x * y),
    lax.div_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x / y),
})


@register_lowering_rule(lax.integer_pow_p)
def _integer_pow_lowering_rule(ctx: LoweringRuleContext, x, y):
  x = _ensure_fa(x, *ctx.avals_in)
  if y == 2:
    return x * x
  return NotImplementedError


@register_lowering_rule(lax.rsqrt_p)
def _rsqrt_lowering_rule(ctx: LoweringRuleContext, x):
  return _ensure_fa(x, *ctx.avals_in).rsqrt()


@register_lowering_rule(lax.reduce_sum_p)
def _reduce_sum_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  if axes != (0,):
    raise NotImplementedError("No support for axes other than 0 yet")
  [x_aval] = ctx.avals_in
  _, [scratch] = ctx.module_context.scratch_view(
      [jax.ShapeDtypeStruct(shape=(4,), dtype=x_aval.dtype)]
  )
  return mgpu.FragmentedArray.splat(x.reduce_sum(scratch), ())


@register_lowering_rule(primitives.debug_print_p)
def _debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    fmt,
    has_placeholders: bool,
):
  del ctx
  del has_placeholders
  primitives.check_debug_print_format(fmt, *args)
  mgpu.debug_print(fmt, *args)
  return ()


def _is_accum_aval(aval):
  return (
      aval.memory_space
      and isinstance(aval.memory_space, mgpu_core.RegsSpace)
      and aval.memory_space.wgmma_config
  )


def _is_smem_aval(aval):
  return aval.memory_space is None or isinstance(
      aval.memory_space, mgpu_core.SMemSpace
  )


def _partition(lst, pred) -> tuple[list[Any], list[Any]]:
  return [x for x in lst if pred(x)], [x for x in lst if not pred(x)]


def _idx_partition(idx_avals: list[tuple[int, Any]], pred) -> tuple[list[int], list[Any], list[tuple[int, Any]]]:
  avals, rest = _partition(idx_avals, lambda x: pred(x[1]))
  if not avals:
    return [], [], list(rest)

  return *zip(*avals), rest

@register_lowering_rule(primitives.run_scoped_p)
def _run_scoped_lowering_rule(
    ctx: LoweringRuleContext, *consts, jaxpr: jax_core.Jaxpr
):
  if len(jaxpr.invars):
    idx_inavals = [(i, v.aval) for i, v in enumerate(jaxpr.invars)]
    in_smem_idx, in_smem_avals, idx_inavals = _idx_partition(idx_inavals, _is_smem_aval)
    in_accum_idx, in_accum_avals, idx_inavals = _idx_partition(idx_inavals, _is_accum_aval)
    if len(idx_inavals) != 0:
      raise ValueError("Unexpacted avals: ", idx_inavals)

    accum_refs = [
        mgpu.WGMMAAccumulator.zero(*aval.shape, mlir.dtype_to_ir_type(aval.dtype))
        for aval in in_accum_avals
    ]

    bytes_allocated, in_smem_refs = ctx.module_context.scratch_view(
        [jax.ShapeDtypeStruct(shape=aval.shape, dtype=aval.dtype) for aval in in_smem_avals]
    )

    # zip the accum refs and in_smem_refs back in argument order.
    idx_refs = zip(in_accum_idx, accum_refs) + zip(in_smem_idx, in_smem_refs)

    assert len(idx_refs) == len(jaxpr.invars), (idx_refs, jaxpr.invars)
    _, input_refs = zip(*sorted(idx_refs, key=lambda x: x[0]))
  else:
    bytes_allocated = 0
    input_refs = []

  outs = lower_jaxpr_to_mosaic_gpu(
      ctx.module_context, jaxpr, input_refs, consts
  )
  ctx.module_context.stack_free_smem(bytes_allocated)
  return outs


@register_lowering_rule(mgpu_primitives.wgmma_zero_accumulator_p)
def _wgmma_zero_accumulator_lowering_rule(
    ctx: LoweringRuleContext, m: int, n: int, config
):
  del ctx, config
  return mgpu.wgmma.WGMMAAccumulator.zero(m, n)

@register_lowering_rule(mgpu_primitives.wgmma_p)
def _wgmma_lowering_rule(
    ctx: LoweringRuleContext, acc, a, b, swizzle, b_row_major,
):
  del ctx
  acc.value = mgpu.wgmma(
      acc,
      a,
      b,
      swizzle=swizzle,
      b_order=mgpu.WGMMALayout.ROW_MAJOR
      if b_row_major
      else mgpu.WGMMALayout.COL_MAJOR,
  ).value
  nvvm_dialect.wgmma_commit_group_sync_aligned()
  return []

@register_lowering_rule(mgpu_primitives.wgmma_wait_p)
def _wgmma_wait_lowering_rule(ctx: LoweringRuleContext, wait_until):
  del ctx
  nvvm_dialect.wgmma_wait_group_sync_aligned(wait_until)
  return ()


def _bcast(
    x,
    y,
    x_aval: jax_core.ShapedArray,
    y_aval: jax_core.ShapedArray,
    out_aval: jax_core.ShapedArray,
) -> ir.Value:
  if isinstance(x, mgpu.WGMMAAccumulator):
    x = x.value

  if isinstance(y, mgpu.WGMMAAccumulator):
    x = y.value

  if isinstance(x, (np.ndarray, np.number, int, float)):
    x_dtype = x_aval.dtype
    if x_aval.weak_type:
      x_dtype = y_aval.dtype
    x = mgpu.FragmentedArray.splat(
        _ir_constant(x, mlir.dtype_to_ir_type(x_dtype)), ()
    )
  if isinstance(y, (np.ndarray, np.number, int, float)):
    y_dtype = y_aval.dtype
    if y_aval.weak_type:
      y_dtype = x_aval.dtype
    y = mgpu.FragmentedArray.splat(
        _ir_constant(y, mlir.dtype_to_ir_type(y_dtype)), ()
    )
  assert isinstance(x, mgpu.FragmentedArray), x
  assert isinstance(y, mgpu.FragmentedArray), y
  if x_aval.shape != out_aval.shape:
    x = x.broadcast(out_aval.shape)
  if y_aval.shape != out_aval.shape:
    y = y.broadcast(out_aval.shape)
  return x, y


def _ensure_fa(x: object, aval: jax_core.ShapedArray) -> mgpu.FragmentedArray:
  if isinstance(x, mgpu.FragmentedArray):
    return x
  elif isinstance(x, (np.number, np.ndarray, int, float)):
    return mgpu.FragmentedArray.splat(
        _ir_constant(x, mlir.dtype_to_ir_type(aval.dtype)), ()
    )
  elif isinstance(x, ir.Value):
    if isinstance(x.type, (ir.IntegerType, ir.FloatType)):
      return mgpu.FragmentedArray.splat(x, ())
  raise NotImplementedError


def _ir_constant(v: object, t: ir.Type) -> ir.Value:
  if isinstance(v, (np.number, np.ndarray, int, float)):
    if isinstance(t, (ir.IntegerType, ir.IndexType)):
      v = int(v)
    else:
      assert isinstance(t, ir.FloatType)
      v = float(v)
    return arith_dialect.constant(t, v)
  raise NotImplementedError(f"Unsupported constant: {v!r}")


def _as_index(v: int | ir.Value) -> ir.Value:
  if isinstance(v, int):
    return arith_dialect.constant(ir.IndexType.get(), v)
  if ir.IndexType.isinstance(v.type):
    return v
  return arith_dialect.index_cast(ir.IndexType.get(), v)
