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
import itertools as it
import math
from typing import Any, cast

import jax
from jax import lax
from jax._src import core as jax_core
from jax._src import pjit
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
from jax._src.lib.mlir.dialects import memref as memref_dialect
from jax._src.lib.mlir.dialects import scf as scf_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives as sp
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.experimental.mosaic.gpu import utils as mgpu_utils
import jax.numpy as jnp
import numpy as np


# TODO(slebedev): Enable type checking.
# mypy: ignore-errors
# pytype: skip-file

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

partial = functools.partial
SMEM = gpu_core.SMEM

_smem_estimators = {}


def _regiter_smem_estimator(primitive: jax_core.Primitive):
  def deco(fn):
    _smem_estimators[primitive] = fn
    return fn

  return deco


def _estimate_smem_scratch_bytes(jaxpr: jax_core.Jaxpr) -> int:
  """Estimates the amount of SMEM scratch bytes required by the kernel."""
  max_used = 0
  for eqn in jaxpr.eqns:
    # TODO(slebedev): Add support for other primitives, notably control flow.
    rule = _smem_estimators.get(eqn.primitive)
    if rule is None:
      # Assume that unsupported primitives are neutral wrt SMEM usage.
      continue
    max_used = max(
        max_used, rule(*(invar.aval for invar in eqn.invars), **eqn.params)
    )
  return max_used


@_regiter_smem_estimator(primitives.run_scoped_p)
def _run_scoped_smem_estimator(*consts, jaxpr: jax_core.Jaxpr) -> int:
  del consts  # Unused.
  in_avals = (v.aval.inner_aval for v in jaxpr.invars)
  return sum(math.prod(aval.shape) * aval.dtype.itemsize for aval in in_avals)


@_regiter_smem_estimator(lax.reduce_sum_p)
def _reduce_sum_smem_estimator(x_aval: jax_core.ShapedArray, *, axes) -> int:
  if axes != (0,):
    raise NotImplementedError("No support for axes other than 0 yet")
  return 4 * x_aval.dtype.itemsize


@dataclasses.dataclass
class ModuleContext:
  name: str
  grid_mapping: pallas_core.GridMapping
  program_ids: Sequence[ir.Value] | None
  approx_math: bool
  runtime_smem: ir.Value  # ir.MemRefType
  smem_used_bytes: int = 0

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
          mgpu_utils.dtype_to_ir_type(s.dtype),
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
  module_ctx: ModuleContext
  launch_ctx: mgpu.LaunchContext
  avals_in: Sequence[jax_core.ShapedArray]
  avals_out: Sequence[jax_core.ShapedArray]

  replace = dataclasses.replace


@dataclasses.dataclass(frozen=True)
class LoweringResult:
  module: ir.Module
  grid: tuple[int, ...]
  out_structs: tuple[jax.ShapeDtypeStruct, ...]


class LoweringError(Exception):  # pylint: disable=g-bad-exception-name
  pass


def _eval_index_map(
    module_ctx: ModuleContext,
    launch_ctx: mgpu.LaunchContext,
    idx: Sequence[ir.Value],
    block_mapping: pallas_core.BlockMapping,
) -> Sequence[ir.Value]:
  block_indices = lower_jaxpr_to_mosaic_gpu(
      module_ctx, launch_ctx, block_mapping.index_map_jaxpr.jaxpr, idx
  )
  result = []
  for i, b in zip(block_indices, block_mapping.block_shape):
    if b is pallas_core.mapped:
      result.append(i)
    else:
      # TODO(slebedev): Use a type-agnostic multiplication wrapper.
      result.append(arith_dialect.muli(_as_index(i), _as_index(b)))
  return tuple(result)


def _uses_arguments(cjaxpr: jax_core.ClosedJaxpr) -> list[bool]:
  jaxpr = cjaxpr.jaxpr
  return pe.dce_jaxpr(jaxpr, used_outputs=[True] * len(jaxpr.outvars))[1]


def lower_jaxpr_to_module(
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    name_and_src_info: pallas_core.NameAndSrcInfo,
    compiler_params: dict[str, Any],
    cost_estimate: pallas_core.CostEstimate | None,
) -> LoweringResult:
  del cost_estimate  # Unused.

  block_mappings = grid_mapping.block_mappings

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
      isinstance(bm.indexing_mode, pallas_core.Blocked) for bm in block_mappings
  ):
    raise NotImplementedError(
        "Only Blocked indexing mode is supported in Mosaic GPU lowering."
    )

  block = (128, 1, 1)
  params = compiler_params.get("mosaic_gpu", {})
  approx_math = params.get("approx_math", False)
  max_concurrent_steps = params.get("max_concurrent_steps", 1)
  dimension_semantics = params.get("dimension_semantics")
  if dimension_semantics is None:
    dimension_semantics = ["parallel"] * len(grid_mapping.grid)
  elif len(dimension_semantics) != len(grid_mapping.grid):
    raise ValueError(
        "dimension_semantics must have an entry for each grid dimension:"
        f" {len(dimension_semantics)=}, but len(grid) is {grid_mapping.grid})."
    )
  sequential_axes = tuple(
      i for i, s in enumerate(dimension_semantics) if s == "sequential"
  )

  grid = [d for i, d in enumerate(grid_mapping.grid) if i not in sequential_axes]
  if len(grid) < 3:
    grid += (1,) * (3 - len(grid))
  else:
    raise NotImplementedError(
        "Only <=3D grids are supported in Mosaic GPU lowering."
    )
  if sequential_axes:
    # TODO(slebedev): Support multiple sequential axes.
    if len(sequential_axes) > 1:
      raise NotImplementedError(
          "Multiple sequential axes are not supported in Mosaic GPU lowering."
      )
    [sequential_axis] = sequential_axes
    num_steps = grid_mapping.grid[sequential_axis]
    out_sequential_invariant = [
        not _uses_arguments(bm.index_map_jaxpr)[sequential_axis]
        for bm in grid_mapping.block_mappings_output
    ]
  else:
    num_steps = 1
    out_sequential_invariant = [True] * len(grid_mapping.out_shapes)

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the allocated buffers below.
  max_concurrent_steps = min(max_concurrent_steps, num_steps)

  in_in_smem, out_in_smem = util.split_list(
      [
          bm.transformed_block_aval.memory_space in (None, gpu_core.SMEM)
          for bm in block_mappings
      ],
      [grid_mapping.num_inputs],
  )

  in_block_mappings, out_block_mappings = util.split_list(
      block_mappings, [grid_mapping.num_inputs]
  )
  in_structs_gmem = [*grid_mapping.in_shapes]
  # We allocate the fully transformed shapes here. All primitives have seen the
  # inverse transformation stack and will understand how to handle it.
  in_structs_smem = [
      jax.ShapeDtypeStruct(
          [max_concurrent_steps, *bm.transformed_block_aval.shape],
          bm.transformed_block_aval.dtype,
      )
      if in_smem
      else None
      for bm, in_smem in zip(
          block_mappings[: grid_mapping.num_inputs], in_in_smem
      )
  ]
  in_gmem_transforms = [
      cast(gpu_core.MemoryRefTransform, bm.transforms)
      for bm in in_block_mappings
  ]
  out_structs_gmem = [*grid_mapping.out_shapes]
  out_structs_smem = [
      jax.ShapeDtypeStruct(
          [max_concurrent_steps, *bm.transformed_block_aval.shape], s.dtype
      )
      if in_smem
      else None
      for bm, in_smem, s in zip(
          block_mappings[grid_mapping.num_inputs :],
          out_in_smem,
          grid_mapping.out_shapes,
      )
  ]
  out_gmem_transforms = [
      cast(gpu_core.MemoryRefTransform, bm.transforms)
      for bm in out_block_mappings
  ]

  def body(launch_ctx: mgpu.LaunchContext, *buffers: ir.Value):
    *buffers_gmem, (
        buffers_smem,
        *scratch_buffers_smem,
        runtime_smem,
        barriers,
    ) = buffers
    assert len(buffers_gmem) == len(buffers_smem)
    in_buffers_gmem, out_buffers_gmem = util.split_list(
        buffers_gmem, [grid_mapping.num_inputs]
    )
    in_buffers_smem, out_buffers_smem = util.split_list(
        buffers_smem, [grid_mapping.num_inputs]
    )
    barriers, *extra_barriers = barriers

    parallel_count = it.count()
    program_ids_template = [
        _program_id(next(parallel_count))
        if axis not in sequential_axes
        else None
        for axis in range(len(grid_mapping.grid))
    ]

    def make_program_ids(step: ir.Value):
      assert ir.IndexType.isinstance(step.type)
      step = arith_dialect.index_cast(ir.IntegerType.get_signless(32), step)
      return [step if pid is None else pid for pid in program_ids_template]

    module_ctx = ModuleContext(
        name_and_src_info.name, grid_mapping, None, approx_math, runtime_smem
    )

    smem_scratch_it = iter(scratch_buffers_smem)
    scratch_buffers_template = []
    should_discharge = []
    accs = []
    for aval in scratch_avals:
      match aval:
        case gpu_core.WGMMAAbstractAccumulatorRef():
          scratch_buffers_template.append(None)
          should_discharge.append(True)
          accs.append(
              mgpu.WGMMAAccumulator.zero(
                  *aval.shape, dtype=mgpu_utils.dtype_to_ir_type(aval.dtype)
              )
          )
        case gpu_core.AbstractMemoryRef() if isinstance(
            aval.dtype, gpu_core.BarrierType
        ):
          pass
        case gpu_core.AbstractMemoryRef() if aval.memory_space == SMEM:
          scratch_buffers_template.append(next(smem_scratch_it))
          should_discharge.append(False)
        case _:
          raise NotImplementedError(
              f"Unsupported scratch operand type: {aval}"
          )
    assert not jaxpr.outvars
    if any(should_discharge):
      # User-visible WGMMA APIs use the effectful accumulator references, but we
      # can't lower that directly to Mosaic GPU that uses pure dataflow for
      # accumulators. So we have to discharge the effects first.
      assert not jaxpr.constvars
      should_discharge = (
          [False] * len(grid_mapping.block_mappings)
          + should_discharge
          + [False] * len(extra_barriers)
      )
      with grid_mapping.trace_env():
        lowered_jaxpr, _ = discharge.discharge_state(
            jaxpr, (), should_discharge=should_discharge
        )
    else:
      lowered_jaxpr = jaxpr

    # Precompute the total number of bytes transferred from GMEM to SMEM,
    # so that we can do a single arrive instruction for all of the inputs.
    in_transfer_bytes = 0
    for in_smem, b_smem in zip(in_in_smem, in_buffers_smem):
      if not in_smem:
        continue
      b_smem_type = ir.MemRefType(b_smem.type)
      in_transfer_bytes += math.prod(b_smem_type.shape[1:]) * mgpu.bytewidth(
          b_smem_type.element_type
      )

    def gmem_slice(
        step: ir.Value,
        block_mapping: pallas_core.BlockMapping,
    ) -> Sequence[mgpu.DynamicSlice]:
      assert len(sequential_axes) <= 1
      program_ids = make_program_ids(step)
      idxs = _eval_index_map(module_ctx, launch_ctx, program_ids, block_mapping)
      return tuple(
          mgpu.ds(idx, dim) for idx, dim in zip(idxs, block_mapping.block_shape)
      )

    is_memory_thread = mgpu.single_thread_predicate(per_block=True)

    def fetch(idx: int, step: ir.Value, slot: ir.Value) -> None:
      if not in_in_smem[idx]:
        return

      swizzle = None
      pl_transforms = in_gmem_transforms[idx]
      if pl_transforms and isinstance(
          pl_transforms[-1], gpu_core.SwizzleTransform
      ):
        swizzle = pl_transforms[-1].swizzle
        pl_transforms = pl_transforms[:-1]
      gmem_transforms = tuple(x.to_gpu_transform() for x in pl_transforms)
      launch_ctx.async_copy(
          src_ref=in_buffers_gmem[idx],
          dst_ref=mgpu.memref_slice(in_buffers_smem[idx], slot),
          gmem_slice=gmem_slice(step, in_block_mappings[idx]),
          barrier=barriers[slot],
          gmem_transform=gmem_transforms,
          swizzle=swizzle,
          arrive=False,  # The caller must do ``arrive_expect_tx`` manually!
          uniform=False,
          predicate=is_memory_thread,
      )

    def store(
        idx: int, step: ir.Value, slot: ir.Value, prev_base_offset: ir.Value | None
    ) -> ir.Value | None:
      if not out_in_smem[idx]:
        return _as_index(-1)

      store_slice = gmem_slice(step, out_block_mappings[idx])
      if out_sequential_invariant[idx]:
        assert prev_base_offset is None
        do_store = None  # Lack of predicate defaults to True.
        base_offset = None
      else:
        assert prev_base_offset is not None
        # We have to do some work to make sure that consecutive stores are not
        # going to be writing to the same location, or else we'll end up with
        # multiple concurrent writes and a racy program.
        # TODO(apaszke,slebedev): This still diverges significantly from the TPU
        # semantics in that it will move on to the next SMEM output slice even if
        # it's not storing the previous one.
        strides, _ = ir.MemRefType(out_buffers_gmem[idx].type).get_strides_and_offset()
        base_offset = _as_index(0)
        for stride, slc in zip(strides, store_slice):
          base_offset = arith_dialect.addi(
              base_offset, arith_dialect.muli(slc.base, _as_index(stride))
          )
        base_offset_changed = arith_dialect.cmpi(
            arith_dialect.CmpIPredicate.ne, base_offset, prev_base_offset
        )
        is_last_step = arith_dialect.cmpi(
            arith_dialect.CmpIPredicate.eq, step, _as_index(num_steps - 1)
        )
        do_store = arith_dialect.andi(
            is_memory_thread, arith_dialect.ori(base_offset_changed, is_last_step)
        )

      swizzle = None
      pl_transforms = out_gmem_transforms[idx]
      if pl_transforms and isinstance(
          pl_transforms[-1], gpu_core.SwizzleTransform
      ):
        swizzle = pl_transforms[-1].swizzle
        pl_transforms = pl_transforms[:-1]
      gmem_transforms = tuple(x.to_gpu_transform() for x in pl_transforms)
      launch_ctx.async_copy(
          src_ref=mgpu.memref_slice(out_buffers_smem[idx], slot),
          dst_ref=out_buffers_gmem[idx],
          gmem_slice=store_slice,
          gmem_transform=gmem_transforms,
          swizzle=swizzle,
          uniform=False,
          predicate=do_store,
      )
      return base_offset

    for slot in range(min(max_concurrent_steps, num_steps)):
      barriers[slot].arrive_expect_tx(in_transfer_bytes, predicate=is_memory_thread)
      for idx in range(grid_mapping.num_inputs):
        fetch(idx, _as_index(slot), _as_index(slot))

    last_store_offsets = [None if inv else _as_index(-1) for inv in out_sequential_invariant]

    @mgpu.fori(_as_index(num_steps), (accs, last_store_offsets))
    def _(step, carry):
      accs, last_store_offsets = carry
      slot = arith_dialect.remui(step, _as_index(max_concurrent_steps))
      if grid_mapping.num_inputs:
        # Only wait if async copies were issued.
        barriers[slot].wait()
      # We need to make sure the output copy is complete before the kernel starts
      # writing to the output window.
      launch_ctx.await_async_copy(max_concurrent_steps - 1, await_read_only=True)

      args = [
          mgpu.memref_slice(buffers_smem[idx], slot)
          if in_smem
          else buffers_gmem[idx]
          for idx, in_smem in enumerate(it.chain(in_in_smem, out_in_smem))
      ]
      accs_it = iter(accs)
      scratch_buffers = [
          b if b is not None else next(accs_it)
          for b in scratch_buffers_template
      ]
      args.extend(scratch_buffers)
      # TODO(apaszke): This assumes barriers come after buffers in scratch args,
      # but that's not necessarily true.
      args.extend(extra_barriers)
      new_accs = lower_jaxpr_to_mosaic_gpu(
          dataclasses.replace(module_ctx, program_ids=make_program_ids(step)),
          launch_ctx,
          lowered_jaxpr,
          args,
      )

      if not all(out_sequential_invariant):
        mgpu.commit_shared()
      new_store_offsets = []
      for idx in range(grid_mapping.num_outputs):
        last_offset = last_store_offsets[idx]
        new_store_offsets.append(
            store(idx, step, slot, last_offset)
            if not out_sequential_invariant[idx]
            else last_offset  # Only store if the output can depend on the step.
        )

      next_step = arith_dialect.addi(step, _as_index(max_concurrent_steps))
      next_step_in_bounds = arith_dialect.cmpi(
          arith_dialect.CmpIPredicate.ult, next_step, _as_index(num_steps)
      )
      next_slot = slot  # (x + y) % y == x % y
      with mgpu.when(next_step_in_bounds):
        barriers[slot].arrive_expect_tx(in_transfer_bytes, predicate=is_memory_thread)
        for idx in range(grid_mapping.num_inputs):
          fetch(idx, next_step, next_slot)

      return list(new_accs), new_store_offsets

    # Outputs invariant to the sequential axis are never written from inside the
    # loop. This is the only place where we store them.
    if all(out_sequential_invariant):
      mgpu.commit_shared()
    last_slot = _as_index((num_steps - 1) % max_concurrent_steps)
    for idx in range(grid_mapping.num_outputs):
      if out_sequential_invariant[idx]:
        store(idx, _as_index(0), last_slot, None)

    launch_ctx.await_async_copy(0)

  scratch_avals = [
      var.aval for var in jaxpr.invars[grid_mapping.slice_scratch_ops]
  ]
  local_spaces = (gpu_core.SMEM, gpu_core.REGS)
  if not all(
      isinstance(aval, pallas_core.AbstractMemoryRef)
      and aval.memory_space in local_spaces
      for aval in scratch_avals
  ):
    raise TypeError(
        "All scratch operands must be SMEM references or accumulators (ACC),"
        f" but got: {scratch_avals}"
    )
  extra_barriers = [
      mgpu.Barrier(aval.dtype.num_arrivals, *aval.shape)
      for aval in scratch_avals
      if isinstance(aval.dtype, gpu_core.BarrierType)
  ]
  extra_smem_scratch = [
      jax.ShapeDtypeStruct(aval.shape, aval.dtype)
      for aval in scratch_avals
      if not isinstance(aval.dtype, gpu_core.BarrierType)
      and aval.memory_space == gpu_core.SMEM
  ]
  smem_scratch_bytes = compiler_params.get("smem_scratch_bytes")
  if smem_scratch_bytes is None:
    smem_scratch_bytes = _estimate_smem_scratch_bytes(jaxpr)
  extra_smem_scratch.append(
      jax.ShapeDtypeStruct(shape=[smem_scratch_bytes], dtype=np.int8)
  )

  module, out_structs_smem, _ = mgpu_core._lower_as_gpu_kernel(
      body,
      grid=grid,
      cluster=(),
      block=block,
      in_shapes=in_structs_gmem,
      out_shape=out_structs_gmem,
      smem_scratch_shape=(
          (*in_structs_smem, *out_structs_smem),
          *extra_smem_scratch,
          (
              mgpu.Barrier(arrival_count=1, num_barriers=max_concurrent_steps),
              *extra_barriers,
          ),
      ),
      module_name=name_and_src_info.name,
  )

  return LoweringResult(module, grid, out_structs_smem)


mosaic_lowering_rules = {}


def register_lowering_rule(primitive: jax_core.Primitive):
  def deco(fn):
    mosaic_lowering_rules[primitive] = fn
    return fn

  return deco


def lower_jaxpr_to_mosaic_gpu(
    module_ctx: ModuleContext,
    launch_ctx: mgpu.LaunchContext,
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
          "Please file an issue on https://github.com/jax-ml/jax/issues."
      )
    rule = mosaic_lowering_rules[eqn.primitive]
    rule_ctx = LoweringRuleContext(
        module_ctx,
        launch_ctx,
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
  if ctx.module_ctx.program_ids is None:
    raise NotImplementedError("pl.program_id() is not supported in this context")
  return ctx.module_ctx.program_ids[axis]


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


def _handle_indexing(
    ref: ir.Value, transforms: Sequence[gpu_core.Transform]
) -> ir.Value:
  if not transforms:
    pass
  if not any(isinstance(t, indexing.NDIndexer) for t in transforms):
    return ref
  if any(
      isinstance(t, indexing.NDIndexer) for t in transforms[:-1]
  ) or not isinstance(transforms[-1], indexing.NDIndexer):
    raise NotImplementedError("Only one level of indexing supported.")

  indexer = cast(indexing.NDIndexer, transforms[-1])
  if indexer.int_indexer_shape:
    raise NotImplementedError("int_indexer_shape non-empty")
  indices = _ndindexer_indices(indexer)
  for t in reversed(transforms[:-1]):
    indices = t.untransform_index(indices)

  if any(isinstance(t, gpu_core.ReshapeRef) for t in transforms[:-1]):
    if not all(isinstance(t, gpu_core.ReshapeRef) for t in transforms[:-1]):
      raise NotImplementedError(f"Reshapes do not compose with other transforms for now ({transforms[:-1]})")
    # The latest reshape overrides all the others. If we reached this
    # point all transforms are reshapes so the last non-indexer must
    # be.
    ref = mgpu.memref_reshape(ref, transforms[-2].shape)

  return mgpu.memref_slice(ref, indices)


def _ndindexer_indices(indexer: indexing.NDIndexer) -> tuple[gpu_core.Index, ...]:
  indices = []
  for idx in indexer.indices:
    if isinstance(idx, indexing.Slice):
      if idx.is_dynamic_start or idx.is_dynamic_size:
        raise NotImplementedError(f"Unsupported slice: {idx}")
      indices.append(slice(idx.start, idx.start + idx.size, idx.stride))
    else:
      indices.append(_as_index(idx))
  return tuple(indices)


def _is_swizzled(transforms: tuple[gpu_core.Transform, ...]) -> int | None:
  if not transforms:
    return None
  if any(isinstance(t, gpu_core.UnswizzleRef) for t in transforms[1:]):
    raise NotImplementedError(
        "Swizzling must be the last transform applied to a ref"
    )
  if isinstance(t := transforms[0], gpu_core.UnswizzleRef):
    return t.swizzle
  return None


@register_lowering_rule(sp.get_p)
def _get_lowering_rule(ctx: LoweringRuleContext, x_smem, *leaves, tree):
  if not isinstance(x_smem, ir.Value) and ir.MemRefType.isinstance(x_smem):
    raise TypeError(f"Can only load from references (got {x_smem}).")
  x_aval = ctx.avals_in[0]
  transform = jax.tree.unflatten(tree, leaves)
  swizzle = _is_swizzled(transform)
  x_smem = _handle_indexing(x_smem, transform)
  if swizzle is None:
    return mgpu.FragmentedArray.load_strided(
        x_smem, is_signed=mgpu_utils.is_signed(x_aval.dtype)
    )
  else:
    return mgpu.FragmentedArray.load_tiled(
        x_smem, is_signed=mgpu_utils.is_signed(x_aval.dtype), swizzle=swizzle
    )


@register_lowering_rule(sp.swap_p)
def _swap_lowering_rule(
    ctx: LoweringRuleContext, x_smem, value, *leaves, tree
):
  if not isinstance(value, mgpu.FragmentedArray):
    raise TypeError(f"Can only store arrays (got {value}).")
  if not isinstance(x_smem, ir.Value) and ir.MemRefType.isinstance(x_smem):
    raise TypeError(f"Can only store to references (got {x_smem}).")
  transforms = jax.tree.unflatten(tree, leaves)
  swizzle = _is_swizzled(transforms)
  x_smem = _handle_indexing(x_smem, transforms)
  x_aval = ctx.avals_in[0]
  if swizzle is None:
    old_value = mgpu.FragmentedArray.load_strided(
        x_smem, is_signed=mgpu_utils.is_signed(x_aval.dtype)
    )
    value.store_untiled(x_smem)
    return old_value
  else:
    old_value = mgpu.FragmentedArray.load_tiled(
        x_smem, is_signed=mgpu_utils.is_signed(x_aval.dtype), swizzle=swizzle
    )
    value.store_tiled(x_smem, swizzle=swizzle)
    return old_value


@register_lowering_rule(pjit.pjit_p)
def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_mosaic_gpu(
      ctx.module_ctx, ctx.launch_ctx, jaxpr.jaxpr, args
  )


@register_lowering_rule(lax.slice_p)
def _slice_lowering_rule(
    ctx: LoweringRuleContext, x, limit_indices, start_indices, strides
):
  if strides is not None:
    raise NotImplementedError("Strides are not supported.")

  return x[tuple(slice(b, e) for b, e in zip(start_indices, limit_indices))]


@register_lowering_rule(lax.select_n_p)
def _select_n_lowering_rule(ctx: LoweringRuleContext, pred, *cases):
  if len(cases) != 2:
    raise NotImplementedError(
        "Mosaic GPU lowering only supports select_n with 2 cases, got"
        f" {len(cases)}"
    )
  pred_aval, *cases_avals = ctx.avals_in
  [out_aval] = ctx.avals_out
  pred = _ensure_fa(pred, pred_aval.dtype)
  cases = _bcast(*cases, *cases_avals, out_aval)
  # ``select`` expects the first case to be the true branch, but ``select_n``
  # orders the cases in reverse.
  return pred.select(*reversed(cases))


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
  [x_aval] = ctx.avals_in
  return _ensure_fa(x, x_aval.dtype).broadcast(shape)


@register_lowering_rule(lax.convert_element_type_p)
def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  del weak_type, sharding
  [x_aval] = ctx.avals_in
  return _ensure_fa(x, x_aval.dtype).astype(
      mgpu_utils.dtype_to_ir_type(new_dtype), is_signed=mgpu_utils.is_signed(new_dtype)
  )


def _binary_op_lowering_rule(ctx: LoweringRuleContext, x, y, *, impl):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return impl(x, y)


mosaic_lowering_rules.update({
    lax.add_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x + y),
    lax.sub_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x - y),
    lax.mul_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x * y),
    lax.div_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x / y),
    lax.rem_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x % y),
    lax.and_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x & y),
    lax.or_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x | y),
    lax.xor_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x ^ y),
    lax.gt_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x > y),
    lax.lt_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x < y),
    lax.ge_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x >= y),
    lax.le_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x <= y),
    lax.eq_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x == y),
    lax.ne_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x != y),
})


@register_lowering_rule(lax.integer_pow_p)
def _integer_pow_lowering_rule(ctx: LoweringRuleContext, x, y):
  [x_aval] = ctx.avals_in
  x = _ensure_fa(x, x_aval.dtype)
  if y == 2:
    return x * x
  return NotImplementedError


@register_lowering_rule(lax.rsqrt_p)
def _rsqrt_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  return _ensure_fa(x, x_aval.dtype).rsqrt(approx=ctx.module_ctx.approx_math)

@register_lowering_rule(lax.logistic_p)
def _logistic_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  a = _ensure_fa(x, x_aval.dtype)
  return 1. / (1. + (-a).exp(approx=ctx.module_ctx.approx_math))


@register_lowering_rule(lax.reduce_sum_p)
def _reduce_sum_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  if axes != (0,):
    raise NotImplementedError("No support for axes other than 0 yet")
  [x_aval] = ctx.avals_in
  _, [scratch] = ctx.module_ctx.scratch_view(
      [jax.ShapeDtypeStruct(shape=(4,), dtype=x_aval.dtype)]
  )
  return mgpu.FragmentedArray.splat(
      x.reduce_sum(scratch), (), is_signed=mgpu_utils.is_signed(x_aval.dtype)
  )


@register_lowering_rule(primitives.debug_print_p)
def _debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    fmt,
    has_placeholders: bool,
):
  del has_placeholders  # Unused.
  primitives.check_debug_print_format(fmt, *args)
  if not any(aval.shape for aval in ctx.avals_in):
    mgpu.debug_print(
        fmt,
        *(
            _ensure_ir_value(arg, aval.dtype)
            for arg, aval in zip(args, ctx.avals_in)
        ),
    )
  elif len(ctx.avals_in) == 1:
    [arg] = args
    @arg.foreach
    def _(val, idx):
      idx_fmt = ", ".join(["{}"] * len(idx))
      fmt_str = fmt.format(f"[{idx_fmt}]/{list(arg.shape)}: {{}}")
      mgpu.debug_print(fmt_str, *idx, val, uniform=False)
  else:
    raise NotImplementedError(
        "debug_print only supports printing of scalar values, or a single array"
        " value when using the Mosaic GPU backend."
    )

  return ()


@register_lowering_rule(primitives.run_scoped_p)
def _run_scoped_lowering_rule(
    ctx: LoweringRuleContext, *consts, jaxpr: jax_core.Jaxpr
):
  input_refs = []
  bytes_allocated = 0
  should_discharge = []
  for a in jaxpr.invars:
    a = a.aval
    if isinstance(a, gpu_core.WGMMAAbstractAccumulatorRef):
      mlir_dtype = mlir.dtype_to_ir_type(a.dtype)
      input_refs.append(mgpu.WGMMAAccumulator.zero(*a.shape, mlir_dtype))
      should_discharge.append(True)
    elif a.memory_space == gpu_core.SMEM:
      ref_bytes, [input_ref] = ctx.module_ctx.scratch_view(
          [jax.ShapeDtypeStruct(shape=a.shape, dtype=a.dtype)]
      )
      bytes_allocated += ref_bytes
      input_refs.append(input_ref)
      should_discharge.append(False)
    else:
      raise ValueError(f"Can't convert to ref: {a}")

  if any(should_discharge):
    # We convert consts to args, because we only have ir.Values and
    # not JAX values during lowering. discharge_state() produces JAX
    # valiues for the aguments but expects them to be provided for the
    # consts. We also don't want to wrap the values in refs.
    no_const_jaxpr = pe.convert_constvars_jaxpr(jaxpr)
    should_discharge = [False] * len(consts) + should_discharge
    discharged_jaxpr, _ = discharge.discharge_state(no_const_jaxpr, (), should_discharge=should_discharge)
    new_input_vals = consts + tuple(input_refs)
    outs = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, discharged_jaxpr, new_input_vals, ()
    )
    # Discharge appends to the output the refs that got discharged.
    outs = outs[:-sum(should_discharge)]
  else:
    outs = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, jaxpr, input_refs, consts
    )

  for o in outs:
    # This is definitely one of the accumulators we produced. Each
    # run_scoped call is responsible for dereferencing its own
    # accumulators.
    if isinstance(o, mgpu.WGMMAAccumulator) or (
        isinstance(o, ir.Value) and ir.MemRefType.isinstance(o.type)
    ):
      raise ValueError(f"No references are allowed to escape a scope. (got {o})")

  assert len(outs) == len(jaxpr.outvars), (jaxpr, outs)
  if bytes_allocated:
    ctx.module_ctx.stack_free_smem(bytes_allocated)

  return outs


def _lower_jaxpr_to_for_loop(
    ctx: LoweringRuleContext,
    jaxpr: jax_core.Jaxpr,
    start: ir.Value,
    length: ir.Value,
    consts,
    *args,
    has_loop_index: bool,
):

  @mgpu.fori(length, [*args])
  def loop(loop_index, body_args):
    if has_loop_index:
      loop_index = arith_dialect.addi(loop_index, start)
      jaxpr_args = [*consts, loop_index, *body_args]
    else:
      jaxpr_args = [*consts, *body_args]
    return lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, jaxpr, jaxpr_args
    )

  return loop.results


@register_lowering_rule(lax.scan_p)
def _scan_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr: jax_core.ClosedJaxpr,
    linear: tuple[bool, ...],
    length: int,
    reverse: bool,
    unroll: bool | int,
    num_consts: int,
    num_carry: int,
    _split_transpose: bool,
):
  # Can only handle fori_loop-like scans.
  if (
      (num_extensive := len(args) - num_consts - num_carry)
      or reverse
      or unroll != 1
  ):
    raise NotImplementedError
  del linear, num_extensive, reverse, unroll

  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts:
    raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = pallas_utils.pattern_match_scan_to_fori_loop(
      jaxpr, num_consts, num_carry
  )
  consts, args = util.split_list(args, [num_consts])
  _consts_avals, arg_avals = util.split_list(ctx.avals_in, [num_consts])
  if has_loop_index:
    start, *args = args
    index_aval, *arg_avals = arg_avals
    start: ir.Value = _ensure_ir_value(start, index_aval.dtype)
    length = _ir_constant(length, start.type)
  else:
    start = _i32_constant(0)
    length = _i32_constant(length)
  args = map(lambda arg, aval: _ensure_fa(arg, aval.dtype), args, arg_avals)
  for_out = _lower_jaxpr_to_for_loop(
      ctx, jaxpr, start, length, consts, *args, has_loop_index=has_loop_index
  )
  if has_loop_index:
    # Need to return the final loop index value if the outer scan expects
    # it as an output.
    return [length, *for_out]
  return for_out


@register_lowering_rule(lax.cond_p)
def _cond_lowering_rule(ctx: LoweringRuleContext, index, *args, branches):
  index_aval, *_arg_avals = ctx.avals_in
  switch_op = scf_dialect.IndexSwitchOp(
      map(mgpu_utils.dtype_to_ir_type, ctx.avals_out),
      _as_index(_ensure_ir_value(index, index_aval.dtype)),
      ir.DenseI64ArrayAttr.get(range(len(branches) - 1)),
      num_caseRegions=len(branches) - 1,
  )

  # ``RegionSequence`` in MLIR does not support slicing, so the
  # auto-generated Python bindings for ``caseRegions`` fail at runtime!
  # We convert it to a list to work around that.
  regions = list(switch_op.regions)
  # Move the default region to the back.
  regions = regions[1:] + regions[:1]
  for branch, region in zip(branches, regions):
    with ir.InsertionPoint(region.blocks.append()):
      outs = lower_jaxpr_to_mosaic_gpu(
          ctx.module_ctx, ctx.launch_ctx, branch.jaxpr, args
      )
      scf_dialect.yield_([
          _ensure_ir_value(out, aval.dtype)
          for out, aval in zip(outs, ctx.avals_out)
      ])
  return list(switch_op.results)


def _bcast(
    x: ir.Value,
    y: ir.Value,
    x_aval: jax_core.ShapedArray,
    y_aval: jax_core.ShapedArray,
    out_aval: jax_core.ShapedArray,
) -> tuple[mgpu.FragmentedArray, mgpu.FragmentedArray]:
  if not isinstance(x, mgpu.FragmentedArray):
    x_dtype = x_aval.dtype
    if x_aval.weak_type:
      x_dtype = y_aval.dtype
    x = _ensure_fa(x, x_dtype)
  if not isinstance(y, mgpu.FragmentedArray):
    y_dtype = y_aval.dtype
    if y_aval.weak_type:
      y_dtype = x_aval.dtype
    y = _ensure_fa(y, y_dtype)
  if x_aval.shape != out_aval.shape:
    x = x.broadcast(out_aval.shape)
  if y_aval.shape != out_aval.shape:
    y = y.broadcast(out_aval.shape)
  return x, y


def _ensure_fa(x: object, dtype: jnp.dtype) -> mgpu.FragmentedArray:
  if isinstance(x, mgpu.FragmentedArray):
    assert x.mlir_dtype == mgpu_utils.dtype_to_ir_type(dtype)
    return x
  elif isinstance(x, (np.number, np.ndarray, int, float)):
    return mgpu.FragmentedArray.splat(
        _ir_constant(x, mgpu_utils.dtype_to_ir_type(dtype)),
        (),
        is_signed=mgpu_utils.is_signed(dtype),
    )
  elif isinstance(x, ir.Value):
    if isinstance(x.type, (ir.IntegerType, ir.FloatType, ir.IndexType)):
      assert x.type == mgpu_utils.dtype_to_ir_type(dtype)
      return mgpu.FragmentedArray.splat(x, (), is_signed=mgpu_utils.is_signed(dtype))
  raise NotImplementedError(f"Unsupported type: {type(x)}")


def _ensure_ir_value(x: object, dtype: jnp.dtype) -> ir.Value:
  if isinstance(x, ir.Value):
    assert x.type == mgpu_utils.dtype_to_ir_type(dtype)
    return x
  elif isinstance(x, (np.number, np.ndarray, int, float)):
    return _ir_constant(x, mgpu_utils.dtype_to_ir_type(dtype))
  elif isinstance(x, mgpu.FragmentedArray):
    if isinstance(x.layout, mgpu.WGSplatFragLayout):
      return x.registers.item()
  raise NotImplementedError(f"Unsupported type: {type(x)}")


def _ir_constant(v: object, t: ir.Type) -> ir.Value:
  if isinstance(v, (np.number, np.ndarray, int, float)):
    if isinstance(t, (ir.IntegerType, ir.IndexType)):
      v = int(v)
    else:
      assert isinstance(t, ir.FloatType)
      v = float(v)
    return arith_dialect.constant(t, v)
  raise NotImplementedError(f"Unsupported constant: {v!r}")


def _i32_constant(v: int) -> ir.Value:
  return arith_dialect.constant(ir.IntegerType.get_signless(32), v)


def _i64_constant(v: int) -> ir.Value:
  return arith_dialect.constant(ir.IntegerType.get_signless(64), v)


def _as_index(v: object) -> ir.Value:
  match v:
    case int():
      return arith_dialect.constant(ir.IndexType.get(), v)
    case ir.Value() if ir.IndexType.isinstance(v.type):
      return v
    case ir.Value() if ir.IntegerType.isinstance(v.type):
      return arith_dialect.index_cast(ir.IndexType.get(), v)
    case mgpu.FragmentedArray(layout=mgpu.WGSplatFragLayout()):
      return _as_index(v.registers.item())
    case _:
      raise ValueError(f"Unsupported index: {v}")
