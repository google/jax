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
import math
from typing import Any, cast

import jax
from jax._src import core as jax_core
from jax._src import pjit
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import memref as memref_dialect
from jax._src.lib.mlir.dialects import nvgpu as nvgpu_dialect
from jax._src.pallas import core as pl_core
from jax._src.pallas import primitives
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
  grid_mapping: pl_core.GridMapping
  runtime_smem: ir.Value  # ir.MemRefType

  def scratch_view(self, shapes: list[jax.ShapeDtypeStruct]) -> list[ir.Value]:
    """Return memref views into the runtime scrath based on the shapes."""

    smem_scratch_bytes = math.prod(ir.MemRefType(self.runtime_smem.type).shape)
    required_scratch_bytes = sum(
        math.prod(sh.shape) * jnp.dtype(sh.dtype).itemsize for sh in shapes
    )
    if smem_scratch_bytes < required_scratch_bytes:
      raise ValueError(
          f"Too few {smem_scratch_bytes=} provided (pass via compiler_params), we"
          f" need {required_scratch_bytes} ({shapes=})"
      )

    views = []
    off = 0
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    for sh in shapes:
      sh_bytes = math.prod(sh.shape) * jnp.dtype(sh.dtype).itemsize
      strides = (*np.cumprod(sh.shape)[:-1:-1], 1)

      # We need scratch to be able to store 128 items of x.
      scratch = memref_dialect.subview(
          self.runtime_smem,
          offsets=[_index(off)],
          sizes=[_index(sh_bytes)],
          strides=[_index(i) for i in strides],
      )
      scratch_ty = ir.MemRefType.get(
          [np.prod(sh.shape)], mlir.dtype_to_ir_type(sh.dtype), memory_space=smem
      )
      off += sh_bytes
      views.append(memref_dialect.view(scratch_ty, scratch, _index(off), []))

    return views


@dataclasses.dataclass
class LoweringRuleContext:
  module_context: ModuleContext
  avals_in: Sequence[jax_core.ShapedArray]
  avals_out: Sequence[jax_core.ShapedArray]
  block_shapes: list[tuple[int | pl_core.Mapped, ...]] | None

  replace = dataclasses.replace


@dataclasses.dataclass
class LoweringResult:
  module: ir.Module
  grid: tuple[int, ...]
  gmem_scratch_bytes: int
  out_structs: tuple[jax.ShapeDtypeStruct, ...]


@dataclasses.dataclass
class BlockInfo:
  full_shape_dtype: jax.ShapeDtypeStruct
  start_indices: Sequence[Any]
  block_shape: tuple[int, ...]


class LoweringError(Exception):
  pass


def lower_jaxpr_to_module(
    grid_mapping: pl_core.GridMapping,
    in_structs: tuple[jax.ShapeDtypeStruct, ...],
    out_structs: tuple[jax.ShapeDtypeStruct, ...],
    jaxpr: jax_core.Jaxpr,
    name: str,
    compiler_params: dict[str, Any],
) -> LoweringResult:
  assert len(jaxpr.outvars) == 0
  assert not grid_mapping.mapped_dims
  grid = grid_mapping.grid
  if len(grid) < 3:
    grid += (1,) * (3 - len(grid))
  block = (128,) + (1,) * (len(grid) - 1)

  def body(launch_ctx: mosaic_gpu.LaunchContext, *buffers):
    *buffers_gmem, (*buffers_smem, runtime_smem) = buffers
    assert len(buffers_gmem) == len(buffers_smem)
    in_buffers_gmem = buffers_gmem[: len(in_structs)]
    in_buffers_smem = buffers_smem[: len(in_structs)]
    out_buffers_gmem = buffers_gmem[len(in_structs) :]
    out_buffers_smem = buffers_smem[len(in_structs) :]

    # arrival_count= determines the expected number of arrivals for each
    # barrier in the array. It is not accidental that we do just a single
    # mbarrier_arrive_expect_tx below.
    # TODO(slebedev): Consider enforcing this in the mgpu.BarrierArray.
    [barrier] = mgpu.BarrierArray(1, arrival_count=1)

    with mgpu.single_thread():
      nvgpu_dialect.mbarrier_arrive_expect_tx(
          barrier.barrier_array.value,
          _index(
              sum(math.prod(s.shape) * s.dtype.itemsize for s in in_structs)
          ),
          barrier.offset,
      )

      for b_gmem, b_smem in zip(in_buffers_gmem, in_buffers_smem):
        # TODO(slebedev): Support 128-byte swizzling, once we can lower matmuls.
        launch_ctx.async_copy(
            src_ref=b_gmem,
            dst_ref=b_smem,
            barrier=barrier,
            swizzle=None,
            arrive=False,
            uniform=False,
        )

    barrier.wait()

    module_ctx = ModuleContext(name, grid_mapping, runtime_smem)
    _ = lower_jaxpr_to_mosaic_gpu(module_ctx, jaxpr, None, *buffers_smem)

    for b_gmem, b_smem in zip(out_buffers_gmem, out_buffers_smem):
      # TODO(slebedev): Support 128-byte swizzling, once we can lower matmuls.
      launch_ctx.async_copy(src_ref=b_smem, dst_ref=b_gmem, swizzle=None)

    launch_ctx.await_async_copy(0)

  extra_smem_scratch = [
      jax.ShapeDtypeStruct(
          shape=[compiler_params.get("smem_scratch_bytes", 0)], dtype=np.int8
      )
  ]
  module, out_structs, gmem_scratch_bytes, _ = mosaic_gpu._lower_as_gpu_kernel(
      body,
      grid,
      block,
      in_shapes=in_structs,
      out_shape=out_structs,
      smem_scratch_shape=(*in_structs, *out_structs, *extra_smem_scratch),
  )

  return LoweringResult(module, grid, gmem_scratch_bytes, out_structs)


mosaic_lowering_rules = {}


def register_lowering_rule(primitive: jax_core.Primitive):
  def deco(fn):
    mosaic_lowering_rules[primitive] = fn
    return fn

  return deco


def lower_jaxpr_to_mosaic_gpu(
    ctx: ModuleContext,
    jaxpr: jax_core.Jaxpr,
    block_infos: Sequence[BlockInfo | None] | None,
    *args,
) -> Sequence[ir.Value]:
  env = {}
  block_info_env = {}

  def read_env(atom: jax_core.Atom):
    return atom.val if isinstance(atom, jax_core.Literal) else env[atom]

  def read_block_info_env(atom: jax_core.Atom):
    if isinstance(atom, jax_core.Literal):
      return None
    return block_info_env.get(atom, None)

  def write_env(var: jax_core.Var, val):
    env[var] = val

  if block_infos is None:
    block_infos = [None] * len(jaxpr.invars)
  for invar, block_info in zip(jaxpr.invars, block_infos):
    block_info_env[invar] = block_info
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
        block_shapes=map(read_block_info_env, eqn.invars),
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


@register_lowering_rule(sp.get_p)
def _get_lowering_rule(ctx: LoweringRuleContext, x_smem, *indexers, tree):
  del tree  # Unused.
  if indexers:
    raise NotImplementedError("No support for indexers yet")
  return mgpu.FragmentedArray.load_strided(x_smem)


@register_lowering_rule(sp.swap_p)
def _swap_lowering_rule(
    ctx: LoweringRuleContext, x_smem, value, *indexers, tree
):
  del tree  # Unused.
  if indexers:
    raise NotImplementedError("No support for indexers yet")
  old_value = mgpu.FragmentedArray.load_strided(x_smem)
  value.store_untiled(x_smem)
  return old_value


@register_lowering_rule(pjit.pjit_p)
def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_mosaic_gpu(ctx.module_context, jaxpr.jaxpr, None, *args)


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
  return x.broadcast(shape)


@register_lowering_rule(lax.convert_element_type_p)
def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type
):
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
  [scratch] = ctx.module_context.scratch_view(
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
  del has_placeholders
  primitives.check_debug_print_format(fmt, *args)
  mgpu.debug_print(fmt, *args)
  return ()


def _bcast(
    x: ir.Value,
    y: ir.Value,
    x_aval: jax_core.ShapedArray,
    y_aval: jax_core.ShapedArray,
    out_aval: jax_core.ShapedArray,
) -> ir.Value:
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
  assert isinstance(x, mgpu.FragmentedArray)
  assert isinstance(y, mgpu.FragmentedArray)
  if x_aval.shape != out_aval.shape:
    x = x.broadcast(out_aval.shape)
  if y_aval.shape != out_aval.shape:
    y = y.broadcast(out_aval.shape)
  return x, y


def _ensure_fa(x: object, aval: jax_core.ShapedArray) -> mgpu.FragmentedArray:
  if isinstance(x, mgpu.FragmentedArray):
    return x
  elif isinstance(x, (np.number, np.ndarray,int, float)):
    return mgpu.FragmentedArray.splat(
        _ir_constant(x, mlir.dtype_to_ir_type(aval.dtype)), ()
    )
  raise NotImplementedError


def _ir_constant(v: object, t: ir.Type) -> ir.Value:
  if isinstance(v, (np.number, np.ndarray, int, float)):
    if isinstance(t, ir.IntegerType):
      v = int(v)
    else:
      assert isinstance(t, ir.FloatType)
      v = float(v)
    return arith_dialect.constant(t, v)
  raise NotImplementedError(f"Unsupported constant: {v!r}")


def _index(i: int) -> ir.Value:
  return arith_dialect.constant(ir.IndexType.get(), int(i))
