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

"""Module for lowering JAX to Mosaic-compatible MLIR dialects."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import functools
import string
from typing import Any, Hashable

import jax
from jax import lax
from jax import tree_util
from jax._src import ad_util
from jax._src import core as jax_core
from jax._src import custom_derivatives
from jax._src import debugging
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import pjit
from jax._src import prng
from jax._src import source_info_util
from jax._src import state
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.lax.control_flow import for_loop
from jax._src.lib import version as jaxlib_version
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import math
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
from jax._src.pallas import pallas_call
from jax._src.pallas import core as pl_core
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.util import safe_map
from jax._src.util import safe_zip
from jax._src.util import split_list
from jax._src.util import unzip2
from jax.experimental.mosaic.dialects import tpu
import jax.numpy as jnp
from jaxlib.mlir.ir import Module
import numpy as np

# TODO(sharadmv): enable type checking
# mypy: ignore-errors

NDIndexer = indexing.NDIndexer
TPUMemorySpace = tpu_core.TPUMemorySpace
MemorySpace = pl_core.MemorySpace | TPUMemorySpace
VMEM = tpu_core.TPUMemorySpace.VMEM
SMEM = tpu_core.TPUMemorySpace.SMEM
# Booleans are stored as the following type in memrefs.
BOOL_MEMREF_TYPE = np.dtype('int32')

# The value interpreted as a dynamic dimension by MLIR.
MLIR_DYNAMIC = -9223372036854775808

partial = functools.partial
map, unsafe_map = safe_map, map  # pylint: disable=redefined-builtin
zip, unsafe_zip = safe_zip, zip  # pylint: disable=redefined-builtin

UNSIGNED_TO_SIGNED = {
    np.dtype('uint8'): np.dtype('int8'),
    np.dtype('uint16'): np.dtype('int16'),
    np.dtype('uint32'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('int64'),
}

@dataclasses.dataclass
class MeshContext:
  mesh_shape: tuple[int, ...]
  axis_names: tuple[str, ...]
  mesh_strides: tuple[int, ...]


@dataclasses.dataclass
class LoweringContext:
  ir_context: ir.Context
  grid_sizes: tuple[int, ...]  # Includes both user and vmap axes.
  grid_names: tuple[Hashable, ...] | None
  mapped_dims: tuple[int, ...]  # Indices of vmapped grid dimensions.
  user_grid_indices: Sequence[ir.Value] | None
  block_shapes: list[tuple[int | pl_core.Mapped, ...]]
  name_stack: source_info_util.NameStack
  mesh_context: MeshContext | None
  replace = dataclasses.replace
  traceback_caches: mlir.TracebackCaches
  for_verification: bool

  @property
  def grid_rank(self):
    return len(self.grid_sizes)

  @contextlib.contextmanager
  def grid_name_context(self):
    # TODO(b/355036977): generalize this across other platforms
    if not self.grid_names:
      yield
      return
    grid_names = self.grid_names
    valid_grid_sizes = tuple(
        d for i, d in enumerate(self.grid_sizes) if i not in self.mapped_dims
    )
    grid_env = zip(grid_names, valid_grid_sizes)
    with jax_core.extend_axis_env_nd(grid_env):
      yield


@dataclasses.dataclass
class LoweringRuleContext:
  lowering_context: LoweringContext
  avals_in: Sequence[jax_core.AbstractValue]
  avals_out: Sequence[jax_core.AbstractValue]
  block_shapes: Sequence[tuple[int | pl_core.Mapped, ...] | None]

  replace = dataclasses.replace


def _memory_space_to_tpu_memspace(memory_space: MemorySpace | None
                                  ) -> ir.Attribute:
  if memory_space is None:
    memory_space = VMEM
  elif memory_space == pl_core.MemorySpace.ERROR:
    memory_space = SMEM
  elif memory_space == pl_core.MemorySpace.INDEX:
    memory_space = SMEM
  return ir.Attribute.parse(f"#tpu.memory_space<{memory_space}>")

def _dtype_to_ir_type(dtype: jnp.dtype,
                      is_kernel_boundary: bool = False) -> ir.Type:
  if jnp.issubdtype(dtype, tpu_core.semaphore_dtype):
    if jnp.issubdtype(dtype, tpu_core.dma_semaphore):
      return ir.Type.parse("!tpu.dma_semaphore")
    elif jnp.issubdtype(dtype, tpu_core.semaphore):
      return ir.Type.parse("!tpu.semaphore")
    elif jnp.issubdtype(dtype, tpu_core.barrier_semaphore):
      return ir.Type.parse("!tpu.semaphore")
    else:
      raise NotImplementedError
  if is_kernel_boundary and jnp.issubdtype(dtype, jnp.dtype('bool')):
    dtype = BOOL_MEMREF_TYPE
  # TODO(justinfu): Remove after mosaic supports unsigned types.
  # This conversion makes mosaic interpret all unsigned types as signed types.
  type =  mlir.dtype_to_ir_type(dtype)
  if isinstance(type, ir.IntegerType):
    return ir.IntegerType.get_signless(type.width)
  else:
    return type

def aval_to_ir_type(aval,
                    shape=None,
                    memory_space: MemorySpace | None = None,
                    is_kernel_boundary: bool = False):
  if isinstance(aval, tpu_core.AbstractSemaphore):
    if aval.sem_type is tpu_core.SemaphoreType.DMA:
      sem_type = ir.Type.parse("!tpu.dma_semaphore")
    elif aval.sem_type is tpu_core.SemaphoreType.REGULAR:
      sem_type = ir.Type.parse("!tpu.semaphore")
    elif aval.sem_type is tpu_core.SemaphoreType.BARRIER:
      sem_type = ir.Type.parse("!tpu.semaphore")
    else:
      raise ValueError(f"Cannot allocate {aval.sem_type}.")
    memspace = _memory_space_to_tpu_memspace(TPUMemorySpace.SEMAPHORE)
    return ir.MemRefType.get((), sem_type, memory_space=memspace)
  if dtypes.issubdtype(aval.dtype, dtypes.prng_key):
    shape = aval.dtype._impl.key_shape
    if memory_space is None:
      memory_space = TPUMemorySpace.SMEM
    if memory_space != TPUMemorySpace.SMEM:
      raise ValueError(f"PRNG keys must be stored in SMEM. Got {memory_space}")
    memspace = _memory_space_to_tpu_memspace(memory_space)
    return ir.MemRefType.get(shape, _dtype_to_ir_type(np.dtype(np.uint32)),
                             memory_space=memspace)
  if isinstance(aval, state.AbstractRef):
    if shape is None:
      shape = aval.shape
    memspace = _memory_space_to_tpu_memspace(memory_space)
    return ir.MemRefType.get(shape,
      _dtype_to_ir_type(aval.dtype, is_kernel_boundary=True),
      memory_space=memspace)
  if isinstance(aval, jax_core.ShapedArray):
    if shape is None:
      shape = aval.shape
    if not shape:
      return _dtype_to_ir_type(
          aval.dtype, is_kernel_boundary=is_kernel_boundary)
    return ir.VectorType.get(
        shape,
        _dtype_to_ir_type(aval.dtype, is_kernel_boundary=is_kernel_boundary))
  raise NotImplementedError(aval)


def ir_constant(x, mlir_type=None):
  if not hasattr(x, "dtype"):
    if isinstance(x, int):
      x = np.array(x, np.int32)
    elif isinstance(x, float):
      x = np.array(x, np.float32)
  if not mlir_type:
    mlir_type = _dtype_to_ir_type(x.dtype)
  if isinstance(x, int) or x.dtype in (np.int32, np.uint32, np.int8):
    return arith.ConstantOp(mlir_type, ir.IntegerAttr.get(mlir_type, int(x))
                            ).result
  elif isinstance(x, float) or x.dtype == np.float32:
    return arith.ConstantOp(
        mlir_type, ir.FloatAttr.get(mlir_type, float(x))
    ).result
  elif x.dtype == jnp.bfloat16:
    return arith.ConstantOp(
        mlir_type, ir.FloatAttr.get(mlir_type, float(x))
    ).result
  elif x.dtype == jnp.bool_:
    return arith.ConstantOp(
        mlir_type, ir.BoolAttr.get(bool(x))
    ).result
  raise NotImplementedError(x.dtype)


lowering_rules = {}
skip_mlir_conversions = set()

def _get_aval_physical_dtype_shape(aval):
  dtype_physical_shape = jax_core.physical_aval(aval).shape[
      len(aval.shape) :
  ]
  return dtype_physical_shape

def _get_arg_type(
    aval,
    block_mapping: pl_core.BlockMapping | None,
):
  memory_space = None
  if isinstance(aval, pl_core.AbstractMemoryRef):
    memory_space = aval.memory_space
    # We assume unannotated memory refs are in VMEM
    if memory_space is None:
      memory_space = TPUMemorySpace.VMEM
  if isinstance(aval, tpu_core.AbstractSemaphore):
    return aval_to_ir_type(aval), None
  # TODO(necula): clean this None block_mapping
  if block_mapping is None:
    return aval_to_ir_type(aval, memory_space=memory_space), aval.shape
  shape = tuple(1 if b is pl_core.mapped else b for b in block_mapping.block_shape)
  return (
      aval_to_ir_type(aval, shape=shape, memory_space=memory_space),
      block_mapping.block_shape,
  )


@dataclasses.dataclass(init=False)
class MosaicGridMapping:
  grid: tuple[int, ...] | None
  grid_names: tuple[Hashable, ...] | None
  jaxpr: jax_core.Jaxpr
  block_mappings: tuple[pl_core.BlockMapping | None, ...]
  mapped_dims: tuple[int, ...]
  scalar_prefetch_types: tuple[ir.Type, ...]
  operand_types: tuple[ir.Type, ...]
  scratch_types: tuple[ir.Type, ...]
  grid_types: tuple[ir.Type, ...]
  scalar_prefetch_block_shapes: tuple[tuple[int, ...], ...]
  operand_block_shapes: tuple[tuple[int, ...], ...]
  scratch_block_shapes: tuple[tuple[int, ...], ...]
  mesh_info: MeshInfo | None
  get_grid_indices: Callable | None

  def __init__(self, jaxpr: jax_core.Jaxpr, grid_mapping: pl_core.GridMapping,
               dimension_semantics: tuple[str, ...] | None,
               mesh: mesh_lib.Mesh | None):
    self.grid = grid_mapping.grid
    self.grid_names = grid_mapping.grid_names
    self.jaxpr = jaxpr
    self.block_mappings = grid_mapping.block_mappings
    self.mapped_dims = grid_mapping.vmapped_dims
    # TODO(necula): clean this using new grid_mapping helpers
    num_scalar_prefetch = grid_mapping.num_index_operands
    num_scratch = grid_mapping.num_scratch_operands
    # jaxpr has signature [*scalar_prefetch, *consts, *in_ops, *out_ops, *scratch]
    num_operands = (
        len(self.jaxpr.invars)
        - num_scalar_prefetch
        - num_scratch
    )
    user_grid = tuple(
        g for i, g in enumerate(self.grid) if i not in self.mapped_dims
    )
    if dimension_semantics is None:
      dimension_semantics = ("arbitrary",) * len(user_grid)
    if len(user_grid) != len(dimension_semantics):
      raise ValueError(
          "Must have dimension semantics for each dimension of the grid."
      )
    if num_operands != len(self.block_mappings):
      raise ValueError("Must have block mappings for each operand.")
    assert len(self.mapped_dims) + len(dimension_semantics) == len(
        self.grid
    ), (
        f"Misconfigured grid: {self.mapped_dims=}, {dimension_semantics=},"
        f" {self.grid=}"
    )
    # dimension_semantics is user provided and won't take into account vmap
    # dimensions. Here we add in parallel dimensions for the vmaps.
    semantics_iter = iter(dimension_semantics)
    self._dimension_semantics = tuple(
        next(semantics_iter) if i not in self.mapped_dims else "parallel"
        for i in range(len(self.grid))
    )

    in_avals = [invar.aval for invar in self.jaxpr.invars]
    scalar_prefetch_avals, operand_avals, scratch_avals = split_list(
        in_avals, [num_scalar_prefetch, num_operands]
    )
    self.scalar_prefetch_types, _ = unzip2([
        _get_arg_type(aval, None)
        for aval in scalar_prefetch_avals])
    self.scalar_prefetch_block_shapes = tuple(
        aval.shape for aval in scalar_prefetch_avals)
    self.operand_types, self.operand_block_shapes = unzip2([
        _get_arg_type(aval, block_mapping)
        for aval, block_mapping in zip(operand_avals, self.block_mappings)])
    self.scratch_types, _ = unzip2([
        _get_arg_type(aval, None) for aval in scratch_avals])
    self.scratch_block_shapes = tuple(
        aval.shape if not isinstance(aval, tpu_core.AbstractSemaphore) else None
        for aval in scratch_avals
    )
    self.grid_types, _ = unzip2([
        _get_arg_type(pl_core.index_map_grid_aval, None)
        for _ in range(len(self.grid))
    ])
    self._prepare_mesh_info(mesh)
    def _get_grid_indices(indices):
      return indices
    self.get_grid_indices = _get_grid_indices

  def _prepare_mesh_info(self, mesh: mesh_lib.Mesh | None):
    if not self.has_communication:
      self.mesh_info = None
      return
    if mesh is None:
      raise ValueError(
          "Cannot use communication in pallas_call without shard_map."
      )
    axis_names = mesh.axis_names
    if self.grid_names is not None:
      if any(a in self.grid_names for a in axis_names):
        raise ValueError(
            "Cannot shadow axis mesh axis names with grid names. mesh axis"
            f" names: {mesh.axis_names}, grid names: {self.grid_names}"
        )
    # We need mesh <-> logical translation tables. Since the logical IDs are
    # just linearized versions of the mesh IDs, we create those tables.
    mesh_strides = pallas_utils.strides_from_shape(tuple(
        mesh.shape[a] for a in axis_names
    ))
    self.mesh_info = MeshInfo(mesh.device_ids.shape, axis_names, mesh_strides)

  def maybe_compress_grid(self):
    # If we have many leading parallel dimensions, we should "compress" them
    # into one so we can load balance across cores as best as we can.
    # TODO(sharadmv): implement this optimization
    pass

  @functools.cached_property
  def has_communication(self) -> bool:
    nonlocal_axis_names = set()
    def _get_nonlocal_axis_names(jaxpr: jax_core.Jaxpr):
      return {
          e.name
          for e in jaxpr.effects
          if isinstance(e, jax_core.NamedAxisEffect)
          and (not self.grid_names or e.name not in self.grid_names)
      }
    nonlocal_axis_names.update(_get_nonlocal_axis_names(self.jaxpr))
    for bm in self.block_mappings:
      if bm is not None:
        nonlocal_axis_names.update(_get_nonlocal_axis_names(bm.index_map_jaxpr))
    return bool(nonlocal_axis_names)

  def get_extra_args(self) -> tuple[Any, ...]:
    return ()

  def get_dimension_semantics(self) -> ir.ArrayAttr:

    def _get_semantics(s: str | None) -> str:
      if s is None:
        return "#tpu.dimension_semantics<arbitrary>"
      return f"#tpu.dimension_semantics<{s}>"

    return ir.ArrayAttr.get(
        map(
            ir.Attribute.parse,
            map(_get_semantics, self._dimension_semantics),
        )
    )

@dataclasses.dataclass
class MeshInfo:
  mesh_shape: tuple[int, ...]
  axis_names: list[str]
  mesh_strides: tuple[int, ...]

def lower_jaxpr_to_module(
    lowering_context: mlir.LoweringRuleContext,
    ctx: ir.Context,
    grid_mapping: pl_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    dimension_semantics: tuple[str | None, ...] | None,
    mesh: mesh_lib.Mesh | None = None,
    for_verification: bool = False,
) -> tuple[Module, tuple[Any, ...]]:
  for bm in grid_mapping.block_mappings:
    def err_details():
      return (f"Block spec for {bm.origin} has block shape "
              f"{bm.block_shape}, array shape {bm.array_shape_dtype.shape}, "
              # TODO(necula): add index_map source location info
              f"and index_map returning {bm.index_map_jaxpr.jaxpr.outvars}, in "
              f"memory space {bm.block_aval.memory_space}.")

    if len(bm.block_shape) < 2:
      if not (bm.block_aval.memory_space == tpu_core.TPUMemorySpace.SMEM and
              bm.has_trivial_window()):
        # TODO(necula): enable this check, seems to regress
        if False:
          raise ValueError(
              "The Pallas TPU lowering currently supports only blocks of "
              "rank >= 2 for blocks, except those in the SMEM memory space "
              "having the same block shape as the array shape and a "
              "trivial index_map (returning all 0s)." + err_details())
      continue

    if (bm.block_aval.memory_space == tpu_core.TPUMemorySpace.ANY and
        not bm.has_trivial_window()):
      raise ValueError(
          "The Pallas TPU lowering currently supports in memory space ANY "
          "only blocks having the same block shape as the array shape "
          "and a trivial index_map (returning all 0s)." + err_details())

    bs0, bs1 = (1 if bs is pl_core.mapped else bs
                for bs in bm.block_shape[-2:])
    as0, as1 = bm.array_shape_dtype.shape[-2:]
    if lowering_context.is_forward_compat() or jaxlib_version < (0, 4, 31):
      # The window divisibility condition in older versions of jaxlib

      # With shape polymorphism block_shape is static, but the array shape may
      # be symbolic. Write the divisibility comparisons to defer inequality
      # comparisons on dimensions as much as possible.
      evenly_divisible = (
          (bs1 % 128 == 0 or (bs1 == as1 and as1 < 128)) and
          (bs0 % 8 == 0 or (bs0 == as0 and as0 < 8))
      )
      if not evenly_divisible:
        raise ValueError(
            "The Pallas TPU lowering currently requires that the last two "
            "dimensions of your block shape are divisible by 8 and 128 "
            "respectively, if the respective dimensions of the overall array "
            "are larger than the respective factors. If array dimensions are "
            "smaller, the block should span the full array dimension. "
            + err_details())
    else:
      evenly_divisible = (
          (bs1 == as1 or bs1 % 128 == 0) and
          (bs0 == as0 or bs0 % 8 == 0)
      )
      if not evenly_divisible:
        raise ValueError(
              "The Pallas TPU lowering currently requires that the last two "
              "dimensions of your block shape are divisible by 8 and 128 "
              "respectively, or be equal to the respective dimensions of the "
              "overall array. "
              + err_details())

  mosaic_grid_mapping = MosaicGridMapping(
      jaxpr, grid_mapping, dimension_semantics, mesh)
  mosaic_grid_mapping.maybe_compress_grid()
  m = ir.Module.create()
  sym_tab = ir.SymbolTable(m.operation)
  func_op = lower_jaxpr_to_func(
      ctx, jaxpr, mosaic_grid_mapping=mosaic_grid_mapping,
      name="main", for_verification=for_verification,
  )
  m.body.append(func_op)
  sym_tab.insert(func_op)
  window_params = []
  grid = mosaic_grid_mapping.grid
  if grid:
    for i, bm in enumerate(grid_mapping.block_mappings):
      func_name = f"transform_{i}"
      # ANY operands don't support windowing and require empty window_params.
      if bm.block_aval.memory_space == tpu_core.TPUMemorySpace.ANY:
        # We checked above that the block does not require windowing.
        window_params.append(ir.DictAttr.get())
        continue
      mlir_func = lower_jaxpr_to_transform_func(
          ctx,
          bm.index_map_jaxpr.jaxpr,
          bm.block_aval,
          name=func_name,
          mosaic_grid_mapping=mosaic_grid_mapping,
          for_verification=for_verification,
      )
      assert mlir_func.verify(), mlir_func
      block_shape = [
          1 if b is pl_core.mapped else b for b in bm.block_shape
      ]
      # If we have an extended dtype, we need to add the block shape for the
      # remaining physical dtype.
      block_shape += list(_get_aval_physical_dtype_shape(bm.block_aval.inner_aval))
      window_shape = ir.DenseI64ArrayAttr.get(block_shape)
      block_params = dict(
          window_bounds=window_shape,
          transform_indices=ir.FlatSymbolRefAttr.get(func_name),
      )
      if isinstance(bm.indexing_mode, pl_core.Unblocked):
        if bm.indexing_mode.padding is None:
          pad_low = pad_high = [0] * len(bm.block_shape)
        else:
          pad_low, pad_high = map(list, zip(*bm.indexing_mode.padding))
        block_params["window_kind"] = ir.Attribute.parse(
            f"#tpu.element_window<{pad_low},{pad_high}>"
        )
      window_params.append(ir.DictAttr.get(block_params))
      m.body.append(mlir_func)
      sym_tab.insert(mlir_func)
    func_op.attributes["window_params"] = ir.ArrayAttr.get(window_params)
    static_grid = [
        MLIR_DYNAMIC if b is pl_core.dynamic_grid_dim else b for b in grid
    ]
    func_op.attributes["iteration_bounds"] = ir.DenseI64ArrayAttr.get(static_grid)

  func_op.attributes["scalar_prefetch"] = ir.IntegerAttr.get(
      ir.IntegerType.get_signless(64), len(mosaic_grid_mapping.scalar_prefetch_types))
  func_op.attributes["scratch_operands"] = ir.IntegerAttr.get(
      ir.IntegerType.get_signless(64), len(mosaic_grid_mapping.scratch_types))
  func_op.attributes["dimension_semantics"] = (
      mosaic_grid_mapping.get_dimension_semantics()
  )
  return m, mosaic_grid_mapping.get_extra_args()


def lower_jaxpr_to_transform_func(
    ctx: ir.Context,
    jaxpr: jax_core.Jaxpr,
    aval: jax_core.AbstractValue,
    *,
    name: str,
    mosaic_grid_mapping: MosaicGridMapping,
    for_verification: bool,
) -> func.FuncOp:
  num_grid = len(mosaic_grid_mapping.grid_types)
  arg_types = [
      *mosaic_grid_mapping.grid_types,
      *mosaic_grid_mapping.scalar_prefetch_types,
  ]
  def body_func(*args):
    grid_indices, scalar_prefetch = split_list(args, [num_grid])
    jaxpr_indices = mosaic_grid_mapping.get_grid_indices(grid_indices)
    arg_block_shapes = [
        *[()] * len(jaxpr_indices),
        *mosaic_grid_mapping.scalar_prefetch_block_shapes,
    ]

    mesh_info = mosaic_grid_mapping.mesh_info
    if mesh_info is not None:
      mesh_context = MeshContext(
          mesh_info.mesh_shape, mesh_info.axis_names, mesh_info.mesh_strides
      )
    else:
      mesh_context = None
    lowering_context = LoweringContext(
        ctx,
        mosaic_grid_mapping.grid,
        mosaic_grid_mapping.grid_names,
        mosaic_grid_mapping.mapped_dims,
        None,
        arg_block_shapes,
        source_info_util.NameStack(),
        mesh_context=mesh_context,
        traceback_caches=mlir.TracebackCaches(),
        for_verification=for_verification,
    )
    out = jaxpr_subcomp(lowering_context, jaxpr, *jaxpr_indices,
                        *scalar_prefetch)
    assert isinstance(aval, state.AbstractRef), aval
    # If we have an extended dtype, we need to add 0s for the block indices
    # for the remaining physical dtype.
    out += [
        ir_constant(0, mlir_type=_dtype_to_ir_type(jnp.dtype("int32")))
    ] * len(_get_aval_physical_dtype_shape(aval.inner_aval))
    return out

  body_func.__name__ = name
  body = func.FuncOp.from_py_func(*arg_types, name=name)(body_func)
  try:
    body.func_op.verify()
  except Exception as e:
    raise LoweringException(
        f"Body failed to verify: {body.func_op}.\nThis is an internal error."
        " Please report a bug at:"
        " https://github.com/google/jax/issues/new?assignees=sharadmv."
    ) from e
  return body.func_op


def lower_jaxpr_to_func(
    ctx: ir.Context,
    jaxpr: jax_core.Jaxpr,
    *,
    mosaic_grid_mapping: MosaicGridMapping,
    name: str,
    for_verification: bool,
) -> func.FuncOp:
  num_grid = len(mosaic_grid_mapping.grid_types)
  num_scalar_prefetch = len(mosaic_grid_mapping.scalar_prefetch_types)
  arg_types = [
      *mosaic_grid_mapping.grid_types,
      *mosaic_grid_mapping.scalar_prefetch_types,
      *mosaic_grid_mapping.operand_types,
      *mosaic_grid_mapping.scratch_types,
  ]
  arg_block_shapes = [
      *mosaic_grid_mapping.scalar_prefetch_block_shapes,
      *mosaic_grid_mapping.operand_block_shapes,
      *mosaic_grid_mapping.scratch_block_shapes,
  ]
  def body_func(*args):
    grid_indices, scalar_prefetch, operands_and_scratch = split_list(
        args, [num_grid, num_scalar_prefetch])
    grid_indices = mosaic_grid_mapping.get_grid_indices(grid_indices)
    jaxpr_indices = tuple(idx for i, idx in enumerate(grid_indices)
                          if i not in mosaic_grid_mapping.mapped_dims)
    mesh_info = mosaic_grid_mapping.mesh_info
    if mesh_info is not None:
      mesh_context = MeshContext(
          mesh_info.mesh_shape, mesh_info.axis_names, mesh_info.mesh_strides
      )
    else:
      mesh_context = None
    lowering_context = LoweringContext(
        ctx,
        mosaic_grid_mapping.grid,
        mosaic_grid_mapping.grid_names,
        mosaic_grid_mapping.mapped_dims,
        jaxpr_indices,
        arg_block_shapes,
        source_info_util.NameStack(),
        mesh_context=mesh_context,
        traceback_caches=mlir.TracebackCaches(),
        for_verification=for_verification,
    )
    return jaxpr_subcomp(
        lowering_context, jaxpr, *scalar_prefetch, *operands_and_scratch
    )
  body_func.__name__ = name
  body = func.FuncOp.from_py_func(*arg_types, name=name)(body_func)
  try:
    body.func_op.verify()
  except Exception as e:
    raise LoweringException(
        f"Body failed to verify: {body.func_op}.\nThis is an internal error."
        " Please report a bug at:"
        " https://github.com/google/jax/issues/new?assignees=sharadmv."
    ) from e
  return body.func_op


def lower_fun(fun: Callable, *, multiple_results: bool) -> Callable:
  def f_lowered(ctx: LoweringRuleContext, *args, **params):
    f = fun if multiple_results else lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params)
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
    if consts:
      raise NotImplementedError
    jaxpr = pe.convert_constvars_jaxpr(jaxpr)
    lowering_context = ctx.lowering_context.replace(
        block_shapes=ctx.block_shapes)
    out = jaxpr_subcomp(lowering_context, jaxpr, *consts, *args)
    if not multiple_results:
      return out[0]
    return out

  return f_lowered


class LoweringException(Exception):
  pass


def _compute_name_stack_updates(
    old_name_stack: list[str],
    new_name_stack: list[str]
) -> tuple[list[str], list[str]]:
  """Computes the popped/pushed items to the name stack after an update.

  Args:
    old_name_stack: The name stack prior to the update.
    new_name_stack: The name stack after the update.

  Returns:
    popped: A list of names popped from the name stack as part of the update.
    pushed: A list of names pushed to the name stack as part of the update.
  """
  common_prefix_idx = 0
  for i, (old, new) in enumerate(unsafe_zip(old_name_stack, new_name_stack)):
    if old == new:
      common_prefix_idx = i+1
    else:
      break
  return old_name_stack[common_prefix_idx:], new_name_stack[common_prefix_idx:]


def jaxpr_subcomp(
    ctx: LoweringContext, jaxpr: jax_core.Jaxpr, *args: ir.Value
) -> Sequence[ir.Value]:
  assert not jaxpr.constvars
  env = {}
  block_shape_env = {}

  def read_block_shape(atom: jax_core.Atom):
    if isinstance(atom, jax_core.Literal):
      return None
    return block_shape_env.get(atom, None)

  def read_env(atom: jax_core.Atom):
    return atom.val if isinstance(atom, jax_core.Literal) else env[atom]

  def write_env(var: jax_core.Var, val):
    is_valid_type = isinstance(val, (ir.Value, KeyScalarBundle))
    assert is_valid_type, type(val)
    env[var] = val

  for invar, bs in zip(jaxpr.invars, ctx.block_shapes):
    block_shape_env[invar] = bs
  map(write_env, jaxpr.invars, args)

  initial_name_stack = [scope.name for scope in ctx.name_stack.stack]
  current_name_stack: list[str] = []
  # TODO(justinfu): Handle transform scopes.
  current_name_stack.extend(initial_name_stack)
  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    source_info = eqn.source_info.replace(
        name_stack=ctx.name_stack + eqn.source_info.name_stack
    )
    loc = mlir._source_info_to_location(
        ctx, eqn.primitive, eqn.params, source_info
    )
    with source_info_util.user_context(eqn.source_info.traceback), loc:
      if eqn.primitive in lowering_rules:
        if eqn.primitive not in skip_mlir_conversions:
          invals = [_ensure_mlir_value(x, v.aval)
                    for x, v in zip(invals, eqn.invars)]
        block_shapes = map(read_block_shape, eqn.invars)
        rule_context = LoweringRuleContext(
            ctx,
            [v.aval for v in eqn.invars],
            [v.aval for v in eqn.outvars],
            block_shapes,
        )

        # Insert trace_start and trace_stop ops on named_scope boundaries.
        name_stack = [scope.name for scope in source_info.name_stack.stack]
        popped, pushed = _compute_name_stack_updates(
            current_name_stack, name_stack)
        current_name_stack = name_stack
        for _ in popped:
          tpu.TraceStopOp()
        for name in pushed:
          tpu.TraceStartOp(message=name, level=10)

        try:
          ans = lowering_rules[eqn.primitive](
              rule_context, *invals, **eqn.params
          )
        except LoweringException:
          raise  # We only add the extra info to the innermost exception.
        except Exception as e:
          raise LoweringException(
              f"Exception while lowering eqn:\n  {eqn}\nWith context:\n "
              f" {rule_context}\nWith inval"
              f" shapes={map(lambda t: getattr(t, 'shape', None), invals)}\nWith"
              " inval"
              f" types={map(lambda t: getattr(t, 'type', None), invals)}\nIn"
              f" jaxpr:\n{jaxpr}"
              f"\nException: {e}"
          ) from e
      else:
        raise NotImplementedError(
            "Unimplemented primitive in Pallas TPU lowering: "
            f"{eqn.primitive.name}. "
            "Please file an issue on https://github.com/google/jax/issues.")
      if eqn.primitive.multiple_results:
        map(write_env, eqn.outvars, ans)
      else:
        write_env(eqn.outvars[0], ans)

  # Drain the name stack at the end of a jaxpr and insert trace_stop ops.
  popped, pushed = _compute_name_stack_updates(
      current_name_stack, initial_name_stack)
  for _ in popped:
    tpu.TraceStopOp()
  assert len(pushed) == 0

  outvals = map(read_env, jaxpr.outvars)
  outvals = [
      ir_constant(x) if isinstance(var, jax_core.Literal) else x
      for x, var in zip(outvals, jaxpr.outvars)
  ]
  return outvals


def _ensure_mlir_value(val, aval):
  if isinstance(val, ir.Value):
    return val
  if isinstance(val, KeyScalarBundle):
    return val
  elif isinstance(val, (np.generic, np.ndarray, int, float)):
    return ir_constant(val, _dtype_to_ir_type(aval.dtype))
  else:
    raise RuntimeError(
        f"Unsupported argument to a JAX primitive of type: {type(val)}"
    )


def _convert_flat_indexing_to_indexer(ref_aval, non_slice_idx,
                                      non_slice_idx_avals, indexed_dims):
  non_slice_idx_iter = iter(zip(non_slice_idx, non_slice_idx_avals))
  splatted_idx_idx_avals = tuple(
      next(non_slice_idx_iter)
      if indexed
      else (primitives.Slice(0, s), primitives.Slice(0, s))
      for s, indexed in zip(ref_aval.shape,indexed_dims)
  )
  splatted_idx, splatted_idx_avals = unzip2(splatted_idx_idx_avals)
  if non_slice_idx:
    (int_indexer_shape,) = {idx_aval.shape for idx_aval in splatted_idx_avals
                            if not isinstance(idx_aval, primitives.Slice)}
  else:
    int_indexer_shape = ()
  nd_indexer = NDIndexer(splatted_idx, ref_aval.shape, int_indexer_shape)
  nd_indexer_avals = NDIndexer(splatted_idx_avals, ref_aval.shape,
                               int_indexer_shape)
  return nd_indexer, nd_indexer_avals


def _get_lowering_rule(
    ctx: LoweringRuleContext, ref, *idx, tree,
):
  indexers = tree_util.tree_unflatten(tree, idx)
  indexers_avals = tree_util.tree_unflatten(tree, ctx.avals_in[1:])
  # Call _load_lowering_rule (since it's more general)
  ref_aval, *_ = ctx.avals_in
  args_flat, args_tree = tree_util.tree_flatten((ref, indexers, None, None))
  avals_flat = tree_util.tree_leaves((ref_aval, indexers_avals, None, None))
  ctx = ctx.replace(
      avals_in=avals_flat,
      block_shapes=[ctx.block_shapes[0], *[None] * (len(avals_flat) - 1)],
  )
  return _load_lowering_rule(ctx, *args_flat, args_tree=args_tree)


lowering_rules[state_primitives.get_p] = _get_lowering_rule
skip_mlir_conversions.add(state_primitives.get_p)


def _swap_lowering_rule(
    ctx: LoweringRuleContext,
    ref,
    val,
    *idx,
    tree
):
  indexers = tree_util.tree_unflatten(tree, idx)
  indexers_avals = tree_util.tree_unflatten(tree, ctx.avals_in[2:])
  # Call _masked_swap_lowering_rule (since it's more general)
  ref_aval, val_aval, *_ = ctx.avals_in
  args_flat, args_tree = tree_util.tree_flatten((ref, indexers, val, None))
  avals_flat = tree_util.tree_leaves(
      (ref_aval, indexers_avals, val_aval, None)
  )
  ctx = ctx.replace(
      avals_in=avals_flat,
      block_shapes=[ctx.block_shapes[0], *[None] * (len(avals_flat) - 1)],
  )
  return _masked_swap_lowering_rule(ctx, *args_flat, args_tree=args_tree)

lowering_rules[state_primitives.swap_p] = _swap_lowering_rule
skip_mlir_conversions.add(state_primitives.swap_p)


def _make_index(s):
  if isinstance(s, (int, np.ndarray)):
    return ir_constant(s, ir.IndexType.get())
  if s.type == ir.IndexType.get():
    return s
  return arith.IndexCastOp(ir.IndexType.get(), s).result


def _maybe_cast_to_index(cast_to_index, x):
  if cast_to_index:
    return _make_index(x)
  return _ensure_mlir_value(x, aval=pl_core.index_map_grid_aval)


def _index_to_start_size_stride(
    idx: tuple[indexing.Slice | int | ir.Value, ...], cast_to_index: bool
) -> tuple[ir.Value, int | ir.Value, int, bool]:
  assert not isinstance(idx, slice)
  if isinstance(idx, indexing.Slice):
    start = _maybe_cast_to_index(cast_to_index, idx.start)
    size = idx.size
    stride = idx.stride
    squeeze = False
  elif isinstance(idx, int):
    start = _maybe_cast_to_index(cast_to_index, idx)
    size = 1
    stride = 1
    squeeze = True
  else:
    if np.shape(idx):
      raise ValueError(f"Can only use ()-shaped and slice indexing: {idx}")
    start = _maybe_cast_to_index(cast_to_index, idx)
    size = 1
    stride = 1
    squeeze = True
  return start, size, stride, squeeze


def _indexer_to_start_size_stride(
    indexer: NDIndexer,
    ref_block_shape: tuple[int | pl_core.Mapped, ...],
    *,
    cast_to_index: bool,
) -> tuple[
    tuple[ir.Value, ...],
    tuple[int | ir.Value, ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[int | pl_core.Mapped, ...],
]:
  indices_iter = iter(indexer.indices)
  starts, sizes, strides, squeeze_dims = [], [], [], []
  for s in ref_block_shape:
    start, size, stride, squeeze_dim = (
        (
            _maybe_cast_to_index(cast_to_index, 0),
            1,
            1,
            True,
        )
        if s is pl_core.mapped
        else _index_to_start_size_stride(next(indices_iter), cast_to_index)
    )
    starts.append(start)
    sizes.append(size)
    strides.append(stride)
    squeeze_dims.append(squeeze_dim)
  next_index = next(indices_iter, None)
  assert next_index is None, (indexer.indices, ref_block_shape)
  new_ref_block_shape = tuple(s for s, squeeze in zip(sizes, squeeze_dims)
                              if not squeeze)
  return (
      tuple(starts),
      tuple(sizes),
      tuple(strides),
      tuple(squeeze_dims),
      new_ref_block_shape,
  )


def _slice_memref(ref: ir.Value, ref_aval: state.AbstractRef,
                  indexer: NDIndexer,
                  ref_block_shape: tuple[int | pl_core.Mapped, ...]
                  ) -> tuple[ir.Value, tuple[int | pl_core.Mapped, ...],
                             tuple[int | pl_core.Mapped, ...]]:
  assert ref_block_shape is not None
  target_shape = indexer.get_indexer_shape()
  starts, sizes, strides, squeeze_dims, ref_block_shape = (
      _indexer_to_start_size_stride(
          indexer,
          ref_block_shape,
          cast_to_index=False,
      )
  )
  if not all((s is None or s == 1) for s in strides):
    raise NotImplementedError("Strided slices of references are unsupported.")
  dynamic_sizes = tuple(s for s in sizes if isinstance(s, ir.Value))
  ir_dynamic_size = ir.ShapedType.get_dynamic_size()
  static_sizes = tuple(s if not isinstance(s, ir.Value)
                       else ir_dynamic_size for s in sizes)
  target_ref_ty = ir.MemRefType.get(
      static_sizes, _dtype_to_ir_type(ref_aval.dtype),
      memory_space=ref.type.memory_space)
  out = tpu.MemRefSliceOp(target_ref_ty, ref, starts, dynamic_sizes).result
  if any(squeeze_dims):
    # We need to squeeze out some dimensions
    static_sizes = tuple(s if not isinstance(s, ir.Value)
                         else ir_dynamic_size for s in target_shape)
    squeezed_ref_ty = ir.MemRefType.get(
        static_sizes, _dtype_to_ir_type(ref_aval.dtype),
        memory_space=ref.type.memory_space)
    out = tpu.MemRefSqueezeOp(squeezed_ref_ty, out).result
  return out, ref_block_shape


def _index_ref(ref, ref_aval, ref_block_shape, indexers):
  for indexer in indexers:
    ref, ref_block_shape = _slice_memref(ref, ref_aval, indexer,
                                         ref_block_shape)
  return ref, ref_block_shape

@dataclasses.dataclass(frozen=True)
class KeyScalarBundle:
  """A container class for PRNG key data.

  We pass around keys as a KeyScalarBundle in the lowering pass rather than
  as a vector, since we want the key data to live in scalar registers rather
  than vector registers. This special dataclass exists so we can return
  multiple scalar values from load_op, because the load_op primitive does
  not allow multiple results.

  Attributes:
    scalars: A list of OpResults representing scalar key data during the
      lowering pass.
  """
  key_shape: tuple[int, ...]
  scalars: list[ir.OpResult]

def _load_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree, **_):
  ref, indexers, mask, _ = args_tree.unflatten(args_flat)
  ref_aval, indexers_avals, _, _ = args_tree.unflatten(ctx.avals_in)
  (*slice_indexers, idx) = indexers
  # Select last aval, which is the one that will be used for the load.
  (*_, idx_aval) = indexers_avals

  if mask is not None:
    raise NotImplementedError

  ref_block_shape, *_ = ctx.block_shapes
  ref, ref_block_shape = _index_ref(
      ref, ref_aval, ref_block_shape, slice_indexers)
  ref_type = ir.MemRefType(ref.type)
  is_smem_load = str(ref_type.memory_space) == "#tpu.memory_space<smem>"
  ref_aval, *_ = ctx.avals_in
  (aval_out,) = ctx.avals_out
  if isinstance(aval_out.dtype, prng.KeyTy):
    if not is_smem_load:
      raise ValueError("PRNG keys must be loaded from SMEM. Did you set "
                       "the memory space to TPUMemorySpace.SMEM in the "
                       "BlockSpec for the PRNG key input?")
    return _prng_key_load_lowering_rule(ctx, *args_flat, args_tree=args_tree)
  if not is_smem_load and not ref_block_shape:
    raise NotImplementedError(
        "Indexing into a ()-shaped Ref not yet supported on TPU.")
  if any(
      (not isinstance(a, primitives.Slice) and a.shape)
      for a in idx_aval.indices
  ):
    raise ValueError("Cannot do int indexing on TPU")
  starts, sizes, strides, _, _ = _indexer_to_start_size_stride(
      idx,
      ref_block_shape,
      cast_to_index=True,
  )
  need_stride = not all((s is None or s == 1) for s in strides)
  if is_smem_load:
    if ctx.avals_out[0].shape:
      raise ValueError("Can only load scalars from SMEM")
    return _maybe_cast_load_to_bool(
        aval_out, memref.LoadOp(ref, starts).result)
  load_aval = jax_core.ShapedArray(sizes, dtype=ref_aval.dtype)
  if need_stride:
    load_val = tpu.StridedLoadOp(
      aval_to_ir_type(load_aval, is_kernel_boundary=True), ref, starts, strides
    ).result
  else:
    load_val = vector.LoadOp(
        aval_to_ir_type(load_aval, is_kernel_boundary=True), ref, starts).result
  load_val = _maybe_cast_load_to_bool(aval_out, load_val)
  if load_aval == aval_out:
    return load_val
  vec_type = ir.VectorType.get(aval_out.shape,
                               _dtype_to_ir_type(aval_out.dtype))
  return vector.ShapeCastOp(vec_type, load_val).result

def _prng_key_load_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree) -> KeyScalarBundle:
  """Lowering rule for loading PRNG keys from SMEM.

  PRNG key loads are currently lowered as a list of scalar loads from SMEM,
  rather than a single vector load.
  We store these scalars in a bundle type called KeyScalarBundle, which has
  special case handling for functions that consume the key such as set_seed.
  """
  ref, _, _, _ = args_tree.unflatten(args_flat)
  (aval_out,) = ctx.avals_out
  assert isinstance(aval_out.dtype, prng.KeyTy)
  ref_block_shape = aval_out.dtype._impl.key_shape

  if len(ref_block_shape) != 2:
    raise NotImplementedError("Seed key_data must be 2D.")
  if tuple(ref_block_shape) != (1, 1):
    raise NotImplementedError(
      f"Seed key_data of shape != (1, 1) not supported. Got: {ref_block_shape}")

  load_ops = []
  for i in range(ref_block_shape[0]):
    idx = NDIndexer(indices=(0, i), shape=ref_block_shape,
                    int_indexer_shape=tuple())
    starts, _, _, _, _ = _indexer_to_start_size_stride(
        idx,
        ref_block_shape,
        cast_to_index=True,
    )
    load_ops.append(memref.LoadOp(ref, starts).result)
  return KeyScalarBundle(scalars=load_ops, key_shape=tuple(ref_block_shape))


lowering_rules[primitives.load_p] = _load_lowering_rule
skip_mlir_conversions.add(primitives.load_p)

def _maybe_cast_load_to_bool(
    out_aval, val: ir.Value) -> tuple[ir.Value, jnp.dtype]:
  """Casts a memref load value to bool if the requested value is a bool.

  Mosaic does not support boolean-type memrefs, since booleans
  typically live in mask registers. We instead load booleans as integers from
  memrefs and move them to mask registers on load using this function.

  Args:
    out_aval: The output aval of the load.
    val: The input value.

  Returns:
    The loaded value, and the JAX dtype of the input value.
  """
  if out_aval.dtype != jnp.bool_:
    return val
  load_scalar_type = _dtype_to_ir_type(BOOL_MEMREF_TYPE)
  if not out_aval.shape:
    # For scalars, truncate the value to a bool.
    pred = _cmpi_lowering_types[lax.ne_p]
    predicate = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), pred)
    const_zero = ir.IntegerAttr.get(load_scalar_type, 0)
    const_zero = arith.ConstantOp(load_scalar_type, const_zero)
    return arith.CmpIOp(predicate, val, const_zero).result
  else:
    raise NotImplementedError("Boolean vector loads are not supported.")

def _maybe_cast_store_to_memref_type(
    expected_aval, val: ir.Value) -> ir.Value:
  """Casts a boolean value back to an integer for storing in a memref."""
  if expected_aval.dtype != jnp.bool_:
    return val
  int_out_type = aval_to_ir_type(expected_aval, is_kernel_boundary=True)
  return arith.ExtUIOp(int_out_type, val).result


def _masked_swap_lowering_rule(
    ctx: LoweringRuleContext, *args_flat, args_tree, **_
):
  ref, indexers, val, mask = args_tree.unflatten(args_flat)
  ref_aval, indexers_avals, val_aval, _ = args_tree.unflatten(ctx.avals_in)
  (*slice_indexers, idx) = indexers
  (*_, idx_aval) = indexers_avals

  if mask is not None:
    raise NotImplementedError

  ref_block_shape, *_ = ctx.block_shapes
  ref, ref_block_shape = _index_ref(
      ref, ref_aval, ref_block_shape, slice_indexers)

  ref_type = ir.MemRefType(ref.type)
  is_smem_store = str(ref_type.memory_space) == "#tpu.memory_space<smem>"
  (aval_out,) = ctx.avals_out
  if not isinstance(val, ir.Value):
    val = ir_constant(val, mlir_type=_dtype_to_ir_type(val_aval.dtype))
  if any(
      (not isinstance(a, primitives.Slice) and a.shape)
      for a in idx_aval.indices
  ):
    raise ValueError("Cannot do int indexing on TPU")
  if not is_smem_store and not ref_block_shape:
    raise NotImplementedError(
        "Indexing into a ()-shaped Ref not yet supported on TPU.")

  starts, _, strides, _, _ = _indexer_to_start_size_stride(
      idx,
      ref_block_shape,
      cast_to_index=True,
  )
  need_stride = not all((s is None or s == 1) for s in strides)
  if is_smem_store:
    if val_aval.shape:
      raise ValueError("Can only store scalars to SMEM")
    result = memref.LoadOp(ref, starts).result
    result = _maybe_cast_load_to_bool(val_aval, result)
    val = _maybe_cast_store_to_memref_type(val_aval, val)
    memref.StoreOp(val, ref, starts)
    return result
  mem_slice_shape = list(aval_out.shape)
  for i, a in enumerate(idx_aval.indices):
    if not isinstance(a, primitives.Slice):
      mem_slice_shape.insert(i, 1)
  mem_slice_shape_iter = iter(mem_slice_shape)
  mem_slice_shape = [
      1 if b is pl_core.mapped else next(mem_slice_shape_iter)
      for b in ref_block_shape
  ]
  mem_aval = aval_out.update(shape=tuple(mem_slice_shape))
  mem_aval_vec_type = ir.VectorType.get(mem_aval.shape,
    _dtype_to_ir_type(mem_aval.dtype, is_kernel_boundary=True))
  if need_stride:
    result = tpu.StridedLoadOp(mem_aval_vec_type, ref, starts, strides).result
  else:
    result = vector.LoadOp(mem_aval_vec_type, ref, starts).result
  val = _maybe_cast_store_to_memref_type(val_aval, val)
  if mem_aval != aval_out:
    # We are slicing a scalar so provided dummy 1 indices
    result_vec_type = ir.VectorType.get(aval_out.shape,
      _dtype_to_ir_type(aval_out.dtype, is_kernel_boundary=True))
    result = vector.ShapeCastOp(result_vec_type, result).result
    val_vec_type = ir.VectorType.get(mem_aval.shape,
      _dtype_to_ir_type(mem_aval.dtype, is_kernel_boundary=True))
    val = vector.ShapeCastOp(val_vec_type, val).result
  result = _maybe_cast_load_to_bool(val_aval, result)

  if need_stride:
    tpu.StridedStoreOp(val, ref, starts, strides)
  else:
    vector.StoreOp(val, ref, starts)
  return result


lowering_rules[primitives.swap_p] = _masked_swap_lowering_rule
skip_mlir_conversions.add(primitives.swap_p)


def _multiple_of_lowering_rule(ctx: LoweringRuleContext, val, *, values):
  del ctx
  for multiple in values:
    val = tpu.assume_multiple(val, multiple)
  return val


lowering_rules[primitives.multiple_of_p] = _multiple_of_lowering_rule


def reduce_lowering_rule(reduce_fn, type_to_kind, type_to_identity):
  def _lowering_rule(ctx: LoweringRuleContext, x, *, axes):
    (x_aval,) = ctx.avals_in
    if not ctx.avals_out[0].shape:
      # If reducing to a scalar, we reduce by adding a leading singleton
      # dimension and reducing over all other dimensions. This avoids
      # the materialization of a scalar tensor by the reduction op which
      # is not supported.
      def _proxy_fun(val, *, axes):
        val = val[jnp.newaxis, ...]
        axes = [axis + 1 for axis in axes]
        val = reduce_fn(val, axis=axes, keepdims=True)
        # Squeeze lowers to vector.ExtractOp which will place the final
        # value in a scalar register.
        return jnp.squeeze(val)
      proxy_lowering = lower_fun(
          _proxy_fun, multiple_results=False)
      return proxy_lowering(ctx, x, axes=axes)

    if jnp.issubdtype(x_aval.dtype, jnp.floating):
      kind = type_to_kind[jnp.floating]
      val = type_to_identity[jnp.floating]
      val = ir.FloatAttr.get(ir.F32Type.get(), val)
    elif jnp.issubdtype(x_aval.dtype, jnp.signedinteger):
      raise NotImplementedError("Reductions over integers not implemented.")
    elif jnp.issubdtype(x_aval.dtype, jnp.unsignedinteger):
      raise NotImplementedError("Reductions over integers not implemented.")
    else:
      raise NotImplementedError(
          f"Reductions over {x_aval.dtype} not implemented.")
    out_type = aval_to_ir_type(ctx.avals_out[0])
    identity = ir.DenseElementsAttr.get_splat(out_type, val)
    acc = arith.ConstantOp(out_type, identity)
    op = vector.MultiDimReductionOp(
        kind,
        x,
        acc,
        ir.ArrayAttr.get(
            [ir.IntegerAttr.get(ir.IntegerType.get_signless(64), a) for a in axes]
        ),
    )
    return op.result
  return _lowering_rule


REDUCE_MAX_KINDS = {
    jnp.floating: vector.CombiningKind.MAXIMUMF,
    jnp.signedinteger: vector.CombiningKind.MAXSI,
    jnp.unsignedinteger: vector.CombiningKind.MAXUI,
}
REDUCE_MAX_IDENTITY = {
    jnp.floating: float("-inf"),
    jnp.signedinteger: np.iinfo(np.int32).min,
}
_reduce_max_lowering_rule = reduce_lowering_rule(
    jnp.max, REDUCE_MAX_KINDS, REDUCE_MAX_IDENTITY)
lowering_rules[lax.reduce_max_p] = _reduce_max_lowering_rule


REDUCE_MIN_KINDS = {
    jnp.floating: vector.CombiningKind.MINIMUMF,
    jnp.signedinteger: vector.CombiningKind.MINSI,
    jnp.unsignedinteger: vector.CombiningKind.MINUI,
}
REDUCE_MIN_IDENTITY = {
    jnp.floating: float("inf"),
    jnp.signedinteger: np.iinfo(np.int32).max,
}
_reduce_min_lowering_rule = reduce_lowering_rule(
    jnp.min, REDUCE_MIN_KINDS, REDUCE_MIN_IDENTITY)
lowering_rules[lax.reduce_min_p] = _reduce_min_lowering_rule


REDUCE_SUM_KINDS = {
    jnp.floating: vector.CombiningKind.ADD,
    jnp.signedinteger: vector.CombiningKind.ADD,
    jnp.unsignedinteger: vector.CombiningKind.ADD,
}
REDUCE_SUM_IDENTITY = {
    jnp.floating: 0.0,
    jnp.signedinteger: 0,
}
_reduce_sum_lowering_rule = reduce_lowering_rule(
    jnp.sum, REDUCE_SUM_KINDS, REDUCE_SUM_IDENTITY)
lowering_rules[lax.reduce_sum_p] = _reduce_sum_lowering_rule


def _reduce_and_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  def _proxy_reduce(arg, *, axes):
    # Mosaic currently only supports float reductions, so we cast the boolean
    # arg to a float and use reduce_min to implement reduce_and.
    # TODO(b/351017807): Implement this logic in Mosaic MultiDimReductionOp
    # instead.
    float_arg = jnp.where(arg, 1.0, 0.0)
    return jnp.min(float_arg, axis=axes) > 0.0
  proxy_lowering = lower_fun(
      _proxy_reduce, multiple_results=False)
  return proxy_lowering(ctx, x, axes=axes)

lowering_rules[lax.reduce_and_p] = _reduce_and_lowering_rule


def _reduce_or_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  def _proxy_reduce(arg, *, axes):
    # Mosaic currently only supports float reductions, so we cast the boolean
    # arg to a float and use reduce_max to implement reduce_or.
    # TODO(b/351017807): Implement this logic in Mosaic MultiDimReductionOp
    # instead.
    float_arg = jnp.where(arg, 1.0, 0.0)
    return jnp.max(float_arg, axis=axes) > 0.0
  proxy_lowering = lower_fun(
      _proxy_reduce, multiple_results=False)
  return proxy_lowering(ctx, x, axes=axes)

lowering_rules[lax.reduce_or_p] = _reduce_or_lowering_rule


def _broadcast_in_dim_lowering_rule(
    ctx: LoweringRuleContext, val, *, shape, broadcast_dimensions
):
  (aval_in,) = ctx.avals_in
  (aval_out,) = ctx.avals_out

  if jnp.issubdtype(aval_in.dtype, jnp.bool_):
    # Direct broadcasts for bools are not supported in Mosaic due to booleans
    # living in mask registers and broadcast operating on vregs. Broadcast as an
    # integer instead and cast back to a bool.
    # TODO(b/351019164): Implement this logic in Mosaic BroadcastOp instead.
    def _proxy_fun(val, *, shape, broadcast_dimensions):
      int_val = jnp.where(val, 1, 0)
      bcast_val = jax.lax.broadcast_in_dim(int_val, shape, broadcast_dimensions)
      return bcast_val == 1
    proxy_lowering = lower_fun(
        _proxy_fun, multiple_results=False)
    return proxy_lowering(
        ctx, val, shape=shape, broadcast_dimensions=broadcast_dimensions)

  if broadcast_dimensions:
    out_shape_list = [1] * len(shape)
    for i, s in zip(broadcast_dimensions, aval_in.shape):
      out_shape_list[i] = s
    out_shape = tuple(out_shape_list)
    out_type = ir.VectorType.get(
        out_shape, _dtype_to_ir_type(aval_out.dtype)
    )
    val = vector.ShapeCastOp(out_type, val).result
    if out_shape == aval_out.shape:
      return val
  out_type = ir.VectorType.get(
      aval_out.shape, _dtype_to_ir_type(aval_out.dtype)
  )
  return vector.BroadcastOp(out_type, val).result


lowering_rules[lax.broadcast_in_dim_p] = _broadcast_in_dim_lowering_rule


def _dot_general_lowering_rule(
    ctx: LoweringRuleContext, x, y, dimension_numbers, precision, **_
):
  (lhs_dims, rhs_dims), _ = dimension_numbers
  (aval_out,) = ctx.avals_out
  out_type = aval_to_ir_type(aval_out)
  val_type = out_type.element_type
  if any(
      cls.isinstance(val_type)
      for cls in [
          ir.BF16Type,
          ir.F32Type,
          ir.Float8E5M2Type,
          ir.Float8E4M3FNType,
      ]
  ):
    val = ir.FloatAttr.get(val_type, 0.0)
  elif ir.IntegerType.isinstance(val_type):
    val = ir.IntegerAttr.get(val_type, 0)
  else:
    raise NotImplementedError(ctx.avals_out[0].dtype)
  if any(len(a.shape) != 2 for a in ctx.avals_in):
    raise NotImplementedError(
        f"Only 2D tensors supported in dot; received: {ctx.avals_in}"
    )
  lhs_aval, _ = ctx.avals_in
  # This is really a matrix-vector product. It only looks like matrix-matrix.
  if lhs_dims == (1,) and rhs_dims == (1,) and ctx.avals_in[1].shape[0] == 1:
    if ctx.avals_in[0].shape != ctx.avals_in[1].shape:
      bcast_shape = jnp.broadcast_shapes(
          ctx.avals_in[0].shape, ctx.avals_out[0].shape
      )
      bcast_shape = ir.VectorType.get(
          list(bcast_shape), _dtype_to_ir_type(ctx.avals_out[0].dtype)
      )
      if ctx.avals_in[0].shape != bcast_shape:
        x = vector.BroadcastOp(bcast_shape, x)
      if ctx.avals_in[1].shape != bcast_shape:
        y = vector.BroadcastOp(bcast_shape, y)
    red_type = aval_to_ir_type(lhs_aval.update(shape=(lhs_aval.shape[0],)))
    acc = arith.ConstantOp(
        red_type, ir.DenseElementsAttr.get_splat(red_type, val)
    )
    red = vector.MultiDimReductionOp(
        ir.Attribute.parse("#vector.kind<add>"),
        arith.MulFOp(x, y),
        acc,
        ir.ArrayAttr.get(
            [ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)]
        ),
    )
    return vector.ShapeCastOp(out_type, red).result

  if lhs_dims == (1,):
    transpose_lhs = False
  elif lhs_dims == (0,):
    transpose_lhs = True
  else:
    raise NotImplementedError
  if rhs_dims == (0,):
    transpose_rhs = False
  elif rhs_dims == (1,):
    transpose_rhs = True
  else:
    raise NotImplementedError
  if precision is not None:
    if precision[0] != precision[1]:
      raise NotImplementedError("Per-operand dot precision unsupported")
    precision = precision[0]
  if precision is None or precision == lax.Precision.DEFAULT:
    precision_attr = None  # That's the default in Mosaic.
  elif precision == lax.Precision.HIGHEST:
    precision_attr = ir.Attribute.parse(
        "#tpu.contract_precision<fp32>"
    )
  else:
    raise NotImplementedError(f"Unsupported dot precision: {precision}")
  out_tile = arith.ConstantOp(
      out_type, ir.DenseElementsAttr.get_splat(out_type, val)
  )
  op = tpu.MatmulOp(
      out_type, x, y, out_tile,
      transpose_lhs=transpose_lhs, transpose_rhs=transpose_rhs,
      precision=precision_attr
  )
  return op.result


lowering_rules[lax.dot_general_p] = _dot_general_lowering_rule

def _convert_helper(x, *, to_dtype):
  # Helper function for dtype conversion
  from_dtype = x.dtype
  if jnp.issubdtype(from_dtype, jnp.dtype("bool")):
    x = x.astype(jnp.int32)
    return _convert_helper(x, to_dtype=to_dtype)
  if jnp.issubdtype(from_dtype, jnp.signedinteger):
    if from_dtype.itemsize < 4:
      x = x.astype(jnp.int32)
    if jnp.issubdtype(to_dtype, jnp.floating) and to_dtype.itemsize < 4:
      x = x.astype(jnp.float32)
    return x.astype(to_dtype)
  if jnp.issubdtype(from_dtype, jnp.floating):
    if jnp.issubdtype(to_dtype, jnp.signedinteger):
      if from_dtype.itemsize < 4:
        x = x.astype(jnp.float32)
      if to_dtype.itemsize < 4:
        # Need to clip values to match XLA
        minval, maxval = jnp.iinfo(to_dtype).min, jnp.iinfo(to_dtype).max
        x = jnp.clip(x, minval, maxval)
        return x.astype(jnp.int32).astype(to_dtype)
      return x.astype(to_dtype)
    elif jnp.issubdtype(to_dtype, np.dtype("bool")):
      x = x.astype(jnp.int32)
    return x.astype(jnp.float32)
  raise NotImplementedError(f"Unsupported cast: {from_dtype} -> {to_dtype}")

def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  del weak_type
  del sharding
  out_aval = ctx.avals_out[0]
  old_dtype = ctx.avals_in[0].dtype
  out_type = aval_to_ir_type(out_aval)

  # TODO(justinfu): Remove after mosaic supports unsigned types.
  # This conversion makes mosaic interpret all unsigned types as signed types.
  if np.issubdtype(new_dtype, jnp.unsignedinteger):
    new_dtype = UNSIGNED_TO_SIGNED[new_dtype]
  if old_dtype == new_dtype:
    return x
  if jnp.issubdtype(old_dtype, jnp.floating) and jnp.issubdtype(
      new_dtype, jnp.floating
  ):
    if old_dtype.itemsize < new_dtype.itemsize and new_dtype.itemsize == 4:
      return arith.ExtFOp(out_type, x).result
    elif old_dtype.itemsize > new_dtype.itemsize and old_dtype.itemsize == 4:
      return arith.TruncFOp(out_type, x).result
  elif jnp.issubdtype(old_dtype, jnp.signedinteger) and jnp.issubdtype(
      new_dtype, jnp.signedinteger
  ):
    if old_dtype.itemsize < new_dtype.itemsize and new_dtype.itemsize == 4:
      return arith.ExtSIOp(out_type, x).result
    elif old_dtype.itemsize > new_dtype.itemsize and old_dtype.itemsize == 4:
      return arith.TruncIOp(out_type, x).result
  elif jnp.issubdtype(old_dtype, jnp.floating) and jnp.issubdtype(
      new_dtype, jnp.signedinteger
  ) and old_dtype.itemsize == new_dtype.itemsize == 4:
    return arith.FPToSIOp(out_type, x).result
  elif jnp.issubdtype(old_dtype, jnp.signedinteger) and jnp.issubdtype(
      new_dtype, jnp.floating
  ) and old_dtype.itemsize == new_dtype.itemsize == 4:
    return arith.SIToFPOp(out_type, x).result
  elif (
      old_dtype == jnp.bool_
      and jnp.issubdtype(new_dtype, jnp.integer)
      and new_dtype.itemsize == 4
  ):
    return arith.extui(out_type, x)
  elif (
      jnp.issubdtype(old_dtype, jnp.integer)
      and new_dtype == jnp.bool_
      and old_dtype.itemsize == 4
  ):
    pred = _cmpi_lowering_types[lax.ne_p]
    predicate = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), pred)
    const_type = _dtype_to_ir_type(old_dtype)
    const_zero = ir.IntegerAttr.get(const_type, 0)
    const_zero = arith.ConstantOp(const_type, const_zero)
    return arith.CmpIOp(predicate, x, const_zero).result
  return lower_fun(functools.partial(_convert_helper, to_dtype=new_dtype),
                   multiple_results=False)(ctx, x)


lowering_rules[lax.convert_element_type_p] = _convert_element_type_lowering_rule


def _reshape_lowering_rule(ctx: LoweringRuleContext, x, new_sizes, dimensions):
  if dimensions is not None:
    raise NotImplementedError
  if any(d is None for d in new_sizes):
    raise NotImplementedError
  if not ctx.avals_in[0].shape:
    return vector.BroadcastOp(aval_to_ir_type(ctx.avals_out[0]), x).result
  return vector.ShapeCastOp(aval_to_ir_type(ctx.avals_out[0]), x).result


lowering_rules[lax.reshape_p] = _reshape_lowering_rule


def _squeeze_lowering_rule(ctx: LoweringRuleContext, x, dimensions):
  del dimensions  # Unused.
  (aval_in,) = ctx.avals_in
  (aval_out,) = ctx.avals_out
  if not aval_out.shape:
    return vector.ExtractOp(x, [], [0] * len(aval_in.shape)).result
  return vector.ShapeCastOp(aval_to_ir_type(ctx.avals_out[0]), x).result


lowering_rules[lax.squeeze_p] = _squeeze_lowering_rule


def _concatenate_lowering_rule(ctx: LoweringRuleContext, *xs, dimension):
  return tpu.ConcatenateOp(
      aval_to_ir_type(ctx.avals_out[0]), xs, dimension=dimension
  ).result


lowering_rules[lax.concatenate_p] = _concatenate_lowering_rule


def _iota_lowering_rule(ctx: LoweringRuleContext, dtype, shape, dimension):
  out_type = aval_to_ir_type(ctx.avals_out[0])
  return tpu.IotaOp(out_type, dimension=dimension).result


lowering_rules[lax.iota_p] = _iota_lowering_rule


def _transpose_lowering_rule(ctx: LoweringRuleContext, x, *, permutation):
  if permutation != (1, 0):
    raise NotImplementedError
  out_type = aval_to_ir_type(ctx.avals_out[0])
  return vector.TransposeOp(out_type, x, permutation).result


lowering_rules[lax.transpose_p] = _transpose_lowering_rule


def _bcast(x, y, x_aval, y_aval, out_aval):
  x_dtype = x_aval.dtype
  y_dtype = y_aval.dtype
  if y_aval.weak_type:
    y_dtype = x_aval.dtype
  elif x_aval.weak_type:
    x_dtype = y_aval.dtype
  if isinstance(x, (np.ndarray, np.number, int, float)):
    if getattr(y, "type", None) == ir.IndexType.get():
      mlir_type = y.type
    else:
      mlir_type = _dtype_to_ir_type(x_dtype)
    x = ir_constant(x, mlir_type)
  if isinstance(y, (np.ndarray, np.number, int, float)):
    if getattr(x, "type", None) == ir.IndexType.get():
      mlir_type = x.type
    else:
      mlir_type = _dtype_to_ir_type(y_dtype)
    y = ir_constant(y, mlir_type)
  out_shape = list(out_aval.shape)
  if x_aval.shape != out_aval.shape:
    x_ty = ir.VectorType.get(out_shape, _dtype_to_ir_type(x_dtype))
    x = vector.BroadcastOp(x_ty, x)
  if y_aval.shape != out_aval.shape:
    y_ty = ir.VectorType.get(out_shape, _dtype_to_ir_type(y_dtype))
    y = vector.BroadcastOp(y_ty, y)
  return x, y


def _add_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return arith.AddIOp(x, y).result
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.AddFOp(x, y).result
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.add_p] = _add_lowering_rule
skip_mlir_conversions.add(lax.add_p)
lowering_rules[ad_util.add_any_p] = _add_lowering_rule
skip_mlir_conversions.add(ad_util.add_any_p)


def _max_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.signedinteger):
    return arith.MaxSIOp(x, y).result
  elif jnp.issubdtype(aval_out.dtype, jnp.unsignedinteger):
    return arith.MaxUIOp(x, y).result
  elif jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.MaximumFOp(x, y).result
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.max_p] = _max_lowering_rule
skip_mlir_conversions.add(lax.max_p)


def _min_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.signedinteger):
    return arith.MinSIOp(x, y).result
  elif jnp.issubdtype(aval_out.dtype, jnp.unsignedinteger):
    return arith.MinUIOp(x, y).result
  elif jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.MinimumFOp(x, y).result
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.min_p] = _min_lowering_rule
skip_mlir_conversions.add(lax.min_p)


def _sub_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return arith.SubIOp(x, y).result
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.SubFOp(x, y).result
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.sub_p] = _sub_lowering_rule
skip_mlir_conversions.add(lax.max_p)


def _mul_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return arith.MulIOp(x, y).result
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.MulFOp(x, y).result
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.mul_p] = _mul_lowering_rule
skip_mlir_conversions.add(lax.mul_p)


def _div_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return arith.DivSIOp(x, y).result
  if jnp.issubdtype(aval_out.dtype, jnp.unsignedinteger):
    return arith.DivUIOp(x, y).result
  elif jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.DivFOp(x, y).result
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.div_p] = _div_lowering_rule
skip_mlir_conversions.add(lax.div_p)


def _rem_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return arith.RemSIOp(x, y).result
  if jnp.issubdtype(aval_out.dtype, jnp.unsignedinteger):
    return arith.RemUIOp(x, y).result
  elif jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.RemFOp(x, y).result
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.rem_p] = _rem_lowering_rule
skip_mlir_conversions.add(lax.rem_p)


def _abs_lowering_rule(ctx: LoweringRuleContext, x):
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return math.AbsIOp(x).result
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return math.AbsFOp(x).result
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.abs_p] = _abs_lowering_rule


def _neg_lowering_rule(ctx: LoweringRuleContext, x):
  (x_aval,) = ctx.avals_in
  new_ctx = ctx.replace(
      avals_in=(jax_core.ShapedArray((), x_aval.dtype), x_aval),
      block_shapes=((), *ctx.block_shapes)
  )
  return _sub_lowering_rule(new_ctx, np.array(0, dtype=x_aval.dtype), x)


lowering_rules[lax.neg_p] = _neg_lowering_rule
skip_mlir_conversions.add(lax.neg_p)


def _sign_lowering_helper(x):
  if jnp.issubdtype(x.dtype, jnp.unsignedinteger):
    return (x != 0).astype(x.dtype)

  if jnp.issubdtype(x.dtype, jnp.integer):
    return (x > 0).astype(x.dtype) - (x < 0).astype(x.dtype)

  if jnp.issubdtype(x.dtype, jnp.floating):
    out = (x > 0.).astype(x.dtype) - (x < 0.).astype(x.dtype)
    return jnp.where(jnp.isnan(x), jnp.nan, out)

  raise NotImplementedError


def _sign_lowering_rule(ctx: LoweringRuleContext, x):
  return lower_fun(_sign_lowering_helper, multiple_results=False)(ctx, x)


lowering_rules[lax.sign_p] = _sign_lowering_rule


def _rsqrt_lowering_rule(ctx: LoweringRuleContext, x):
  return math.RsqrtOp(x).result


lowering_rules[lax.rsqrt_p] = _rsqrt_lowering_rule


def _sqrt_lowering_rule(ctx: LoweringRuleContext, x):
  return math.SqrtOp(x).result


lowering_rules[lax.sqrt_p] = _sqrt_lowering_rule


def _exp_lowering_rule(ctx: LoweringRuleContext, x):
  return math.ExpOp(x).result


lowering_rules[lax.exp_p] = _exp_lowering_rule


def _pow_lowering_rule(ctx: LoweringRuleContext, x, y):
  if not isinstance(x, ir.Value) and x == 2.:
    return math.Exp2Op(y).result
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  return math.PowFOp(x, y).result


lowering_rules[lax.pow_p] = _pow_lowering_rule
skip_mlir_conversions.add(lax.pow_p)


def _integer_pow_lowering_rule(ctx: LoweringRuleContext, x, *, y):
  return lower_fun(lax_internal._integer_pow, multiple_results=False)(
      ctx, x, y=y)


lowering_rules[lax.integer_pow_p] = _integer_pow_lowering_rule


def _exp2_lowering_rule(ctx: LoweringRuleContext, x):
  # exp2 in JAX lowers to exp(ln2 * x), not to pow2. We match that behavior
  # here.
  return lower_fun(lambda x: jnp.exp(np.log(2) * x), multiple_results=False)(
      ctx, x)


lowering_rules[lax.exp2_p] = _exp2_lowering_rule
skip_mlir_conversions.add(lax.exp2_p)


def _logistic_lowering_rule(ctx: LoweringRuleContext, x):
  neg_x = arith.NegFOp(x).result
  exp_neg_x = math.ExpOp(neg_x).result
  aval_out = ctx.avals_out[0]
  out_type = aval_to_ir_type(aval_out)
  if aval_out.shape == ():
    one = ir_constant(1.0, mlir_type=out_type)
  else:
    one = vector.BroadcastOp(out_type, ir_constant(1.0))
  denom = arith.AddFOp(one, exp_neg_x).result
  return arith.DivFOp(one, denom).result


lowering_rules[lax.logistic_p] = _logistic_lowering_rule


def _sin_lowering_rule(ctx: LoweringRuleContext, x):
  return math.SinOp(x).result


lowering_rules[lax.sin_p] = _sin_lowering_rule


def _tanh_lowering_rule(ctx: LoweringRuleContext, x):
  return math.TanhOp(x).result


lowering_rules[lax.tanh_p] = _tanh_lowering_rule


def _log_lowering_rule(ctx: LoweringRuleContext, x):
  return math.LogOp(x).result


lowering_rules[lax.log_p] = _log_lowering_rule


def _log1p_lowering_rule(ctx: LoweringRuleContext, x):
  return math.Log1pOp(x).result


lowering_rules[lax.log1p_p] = _log1p_lowering_rule


def _round_lowering_rule(ctx: LoweringRuleContext, x, *, rounding_method):
  if rounding_method == 0:
    return math.RoundOp(x).result
  elif rounding_method == 1:
    return math.RoundEvenOp(x).result
  else:
    raise NotImplementedError(f"Unsupported rounding method: {rounding_method}")


lowering_rules[lax.round_p] = _round_lowering_rule


# See https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-arithcmpiop for
# the mapping from comparison type to integer predicates for int comparisons.
_cmpi_lowering_types = {
    lax.eq_p: 0,
    lax.ne_p: 1,
    lax.lt_p: 2,
    lax.le_p: 3,
    lax.gt_p: 4,
    lax.ge_p: 5,
}

# See https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpf-arithcmpfop for
# the mapping from comparison type to integer predicate for float comparisons.
_cmpf_lowering_types = {
    lax.eq_p: 1,
    lax.ne_p: 6,
    lax.lt_p: 4,
    lax.le_p: 5,
    lax.gt_p: 2,
    lax.ge_p: 3,
}


def _cmp_lowering_rule(prim, ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  x_aval, y_aval = ctx.avals_in
  dtypes = x_aval.dtype, y_aval.dtype
  if all(jnp.issubdtype(dtype, jnp.integer) for dtype in dtypes):
    pred = _cmpi_lowering_types[prim]
    predicate = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), pred)
    return arith.CmpIOp(predicate, x, y).result
  elif all(jnp.issubdtype(dtype, jnp.floating) for dtype in dtypes):
    pred = _cmpf_lowering_types[prim]
    predicate = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), pred)
    return arith.CmpFOp(predicate, x, y).result
  raise NotImplementedError("Mixed dtype operands in cmp")


lowering_rules[lax.eq_p] = functools.partial(_cmp_lowering_rule, lax.eq_p)
lowering_rules[lax.ne_p] = functools.partial(_cmp_lowering_rule, lax.ne_p)
lowering_rules[lax.lt_p] = functools.partial(_cmp_lowering_rule, lax.lt_p)
lowering_rules[lax.le_p] = functools.partial(_cmp_lowering_rule, lax.le_p)
lowering_rules[lax.gt_p] = functools.partial(_cmp_lowering_rule, lax.gt_p)
lowering_rules[lax.ge_p] = functools.partial(_cmp_lowering_rule, lax.ge_p)


def _and_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return arith.AndIOp(x, y).result


lowering_rules[lax.and_p] = _and_lowering_rule
skip_mlir_conversions.add(lax.and_p)


def _or_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return arith.OrIOp(x, y).result


lowering_rules[lax.or_p] = _or_lowering_rule
skip_mlir_conversions.add(lax.or_p)


def _not_lowering_rule(ctx: LoweringRuleContext, x):
  # The primitive not_p is lowered to
  # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not
  # which is arithmetic for integers and logical for booleans.
  # Lowering to:
  # xor x, -1
  # covers both cases.
  out_aval = ctx.avals_out[0]
  out_scalar_type = _dtype_to_ir_type(out_aval.dtype)
  if not out_aval.shape:
    # Create a scalar constant.
    minus_one = ir_constant(-1, out_scalar_type)
  else:
    # Create a vector constant.
    out_type = aval_to_ir_type(out_aval)
    scalar_minus_one = ir.IntegerAttr.get(out_scalar_type, -1)
    minus_one = arith.ConstantOp(
        out_type, ir.DenseElementsAttr.get_splat(out_type, scalar_minus_one)
    )
  return arith.XOrIOp(x, minus_one).result


lowering_rules[lax.not_p] = _not_lowering_rule

def _select_n_lowering_rule(ctx: LoweringRuleContext, pred, x, *args):
  if len(args) > 1:
    raise NotImplementedError("select_n only supported with <= 2 arguments")
  pred_aval, x_aval = ctx.avals_in[:2]
  if pred_aval.dtype != np.dtype(np.bool_):
    lower_ctx = LoweringRuleContext(
        ctx.lowering_context,
        avals_in=[pred_aval],
        avals_out=[pred_aval.update(dtype=np.bool_)],
        block_shapes=[None],
    )
    pred = lower_fun(lambda x: x != 0, multiple_results=False)(lower_ctx, pred)
  if not args:
    return x
  # Assume x and y, which we check above.
  y, = args
  return arith.SelectOp(pred, y, x).result


lowering_rules[lax.select_n_p] = _select_n_lowering_rule


def _clamp(min, operand, max):
  res = jnp.maximum(operand, min)
  return jnp.minimum(res, max)


def _clamp_lowering_rule(ctx: LoweringRuleContext, min, operand, max):
  """Compute minimum_p(maximum_p(min, operand), max)."""
  return lower_fun(_clamp, multiple_results=False)(ctx, min, operand, max)


lowering_rules[lax.clamp_p] = _clamp_lowering_rule


def _for_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr,
    nsteps,
    reverse,
    unroll,
    which_linear,
):
  should_discharge = [
      not isinstance(aval, state.AbstractRef) for aval in ctx.avals_in
  ]
  jaxpr, () = state_discharge.discharge_state(
      jaxpr, (), should_discharge=[False, *should_discharge]
  )
  for i in range(nsteps):
    if reverse:
      i = nsteps - i - 1
    i = ir_constant(i)
    lowering_context = ctx.lowering_context.replace(
        block_shapes=[(), *ctx.block_shapes],
    )
    non_ref_args = jaxpr_subcomp(lowering_context, jaxpr, i, *args)
    non_ref_args_iter = iter(non_ref_args)
    args = [
        next(non_ref_args_iter) if s else a
        for a, s in zip(args, should_discharge)
    ]
  return args


lowering_rules[for_loop.for_p] = _for_lowering_rule


def _lower_jaxpr_to_for_loop(ctx: LoweringRuleContext,
                             jaxpr: jax_core.Jaxpr, start: int | ir.Value,
                             num_steps: int | ir.Value, consts, *args,
                             has_loop_index: bool,
                             unroll: int):
  def _run_body(i, args):
    if has_loop_index:
      lowering_context = ctx.lowering_context.replace(
          block_shapes=ctx.block_shapes)
      args = jaxpr_subcomp(lowering_context, jaxpr, *consts, i, *args)
    else:
      del i
      lowering_context = ctx.lowering_context.replace(
          block_shapes=ctx.block_shapes[:len(consts)]
          + ctx.block_shapes[len(consts) + 1:],
      )
      args = jaxpr_subcomp(lowering_context, jaxpr, *consts, *args)
    return args

  if (
      not isinstance(start, ir.Value)
      and not isinstance(num_steps, ir.Value)
      and num_steps == unroll
  ):
    # No need for an scf.For. We can just unroll completely
    for i in range(start, start + num_steps):
      args = _run_body(
          ir_constant(i, mlir_type=_dtype_to_ir_type(jnp.dtype("int32"))),
          args,
      )
    return args
  if unroll != 1:
    raise NotImplementedError(
        f"Only unroll={num_steps=} and unroll=1 supported. Got {unroll=}.")
  lbd = _ensure_mlir_value(start, pl_core.index_map_grid_aval)
  ubd = arith.addi(lbd, _ensure_mlir_value(num_steps, pl_core.index_map_grid_aval))
  step = ir_constant(1, mlir_type=_dtype_to_ir_type(jnp.dtype("int32")))
  for_op = scf.ForOp(lbd, ubd, step, args)
  with ir.InsertionPoint(for_op.body):
    iv = for_op.induction_variable
    inner_args = for_op.inner_iter_args
    inner_out = _run_body(iv, inner_args)
    scf.YieldOp(inner_out)
  return for_op.results


def _lower_jaxpr_to_unrolled_for_loop(ctx: LoweringRuleContext,
                                      jaxpr: jax_core.Jaxpr, start: int,
                                      num_steps: int, consts, *args,
                                      has_loop_index: bool):
  for i in range(start, start + num_steps):
    if has_loop_index:
      lowering_context = ctx.lowering_context.replace(
          block_shapes=ctx.block_shapes)
      args = jaxpr_subcomp(
          lowering_context, jaxpr, *consts,
          ir_constant(i, mlir_type=_dtype_to_ir_type(jnp.dtype('int32'))),
          *args)
    else:
      lowering_context = ctx.lowering_context.replace(
          block_shapes=ctx.block_shapes[:len(consts)]
          + ctx.block_shapes[len(consts) + 1:],
      )
      args = jaxpr_subcomp(lowering_context, jaxpr, *consts, *args)
  return args


def _scan_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr: jax_core.Jaxpr,
    linear: tuple[bool, ...],
    length: int,
    reverse: bool,
    unroll: bool | int,
    num_consts: int,
    num_carry: int,
    _split_transpose: bool,
):
  del _split_transpose
  # Can only handle fori_loop-like scans
  num_extensive = len(args) - num_consts - num_carry
  if num_extensive: raise NotImplementedError
  if reverse: raise NotImplementedError
  del linear, num_extensive, reverse

  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts: raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = pallas_utils.pattern_match_scan_to_fori_loop(
      jaxpr, num_consts, num_carry
  )
  consts, args = split_list(args, [num_consts])
  consts_avals, args_avals = split_list(ctx.avals_in, [num_consts])
  if has_loop_index:
    loop_index_start, *args = args
    args_avals = args_avals[1:]
  else:
    loop_index_start = 0
  consts = map(_ensure_mlir_value, consts, consts_avals)
  args = map(_ensure_mlir_value, args, args_avals)
  out = _lower_jaxpr_to_for_loop(
      ctx, jaxpr, loop_index_start, length,
      consts, *args, has_loop_index=has_loop_index,
      unroll=unroll)
  if has_loop_index:
    out = [ir_constant(length,
                       mlir_type=_dtype_to_ir_type(jnp.dtype('int32'))),
           *out]
  return out
lowering_rules[lax.scan_p] = _scan_lowering_rule
skip_mlir_conversions.add(lax.scan_p)


def _lower_while_via_fori(
    ctx: LoweringRuleContext,
    *args,
    fori_jaxpr,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr,
):
  _, body_consts, carry = split_list(args, [cond_nconsts, body_nconsts])
  (lb, ub), args = carry[:2], carry[2:]
  for_out = _lower_jaxpr_to_for_loop(
      ctx.replace(
          block_shapes=ctx.block_shapes[: body_nconsts + 1]
          + ctx.block_shapes[body_nconsts + 2 :],
      ),
      fori_jaxpr,
      lb,
      arith.subi(ub, lb),
      body_consts,
      *args,
      has_loop_index=True,
      unroll=1,
  )
  return [ub, ub, *for_out]


def _while_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr,
):
  # First try to lower via a simpler fori loop, which may optimize better.
  fori_jaxpr, err = pallas_utils.pattern_match_while_to_fori_loop(
      cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts
  )
  if fori_jaxpr is not None:
    return _lower_while_via_fori(
        ctx,
        *args,
        fori_jaxpr=fori_jaxpr,
        cond_nconsts=cond_nconsts,
        cond_jaxpr=cond_jaxpr,
        body_nconsts=body_nconsts,
        body_jaxpr=body_jaxpr,
    )

  # If we fail conversion to fori, fallback to an ordinary while loop.
  cond_consts, body_consts, carry = split_list(
      args, [cond_nconsts, body_nconsts]
  )
  cond_const_block_shapes, body_const_block_shapes, carry_block_shapes = (
      split_list(ctx.block_shapes, [cond_nconsts, body_nconsts])
  )
  cond_const_types = [a.type for a in cond_consts]
  body_const_types = [a.type for a in body_consts]
  carry_types = [a.type for a in carry]
  all_types = [*cond_const_types, *body_const_types, *carry_types]
  while_op = scf.WhileOp(all_types, args)

  before_block = while_op.before.blocks.append(*all_types)
  cond_consts_, _, carry_ = split_list(
      before_block.arguments,
      [cond_nconsts, body_nconsts],
  )
  cond_args = [*cond_consts_, *carry_]
  with ir.InsertionPoint.at_block_begin(before_block):
    [cond] = jaxpr_subcomp(
        ctx.lowering_context.replace(
            block_shapes=[*cond_const_block_shapes, *carry_block_shapes]
        ),
        cond_jaxpr.jaxpr,
        *cond_args,
    )
    scf.condition(cond, before_block.arguments)

  after_block = while_op.after.blocks.append(*all_types)
  cond_consts_, body_consts_, carry_ = split_list(
      after_block.arguments,
      [cond_nconsts, body_nconsts],
  )
  all_args = [*cond_consts_, *body_consts_, *carry_]
  cond_const_args, body_const_args, carry_args = split_list(
      all_args, [cond_nconsts, body_nconsts]
  )
  with ir.InsertionPoint.at_block_begin(after_block):
    loop_out = jaxpr_subcomp(
        ctx.lowering_context.replace(
            block_shapes=[*body_const_block_shapes, *carry_block_shapes],
        ),
        body_jaxpr.jaxpr,
        *body_const_args,
        *carry_args,
    )
    all_handles = [*cond_const_args, *body_const_args, *loop_out]
    if all_handles:
      scf.yield_(all_handles)

  all_out = list(while_op.results_)
  return all_out[cond_nconsts + body_nconsts :]


lowering_rules[lax.while_p] = _while_lowering_rule

def _cond_lowering_rule(ctx: LoweringRuleContext, *args, branches):
  index, *args = args
  out_types = map(aval_to_ir_type, ctx.avals_out)
  pred = arith.CmpIOp(
      arith.CmpIPredicate.ne, index, ir_constant(0, index.type)
  ).result
  if_op = scf.IfOp(pred, out_types, hasElse=True)
  lowering_context = ctx.lowering_context.replace(
      block_shapes=ctx.block_shapes[1:],
  )
  with ir.InsertionPoint(if_op.then_block):
    # TODO(b/300272065): Use `scf.IndexSwitchOp` instead of a cascade of
    # if/else.
    if len(branches) > 2:
      out = _cond_lowering_rule(
          ctx,
          arith.SubIOp(index, ir_constant(1, index.type)).result,
          *args,
          branches=branches[1:],
      )
    else:
      out = jaxpr_subcomp(lowering_context, branches[1].jaxpr, *args)
    scf.YieldOp(out)
  with ir.InsertionPoint(if_op.else_block):
    out = jaxpr_subcomp(lowering_context, branches[0].jaxpr, *args)
    scf.YieldOp(out)
  return if_op.results


lowering_rules[lax.cond_p] = _cond_lowering_rule


def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  lowering_context = ctx.lowering_context.replace(block_shapes=ctx.block_shapes)
  return jaxpr_subcomp(lowering_context, jaxpr.jaxpr, *args)


lowering_rules[pjit.pjit_p] = _pjit_lowering_rule


def _custom_jvp_call_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    call_jaxpr: jax_core.Jaxpr,
    jvp_jaxpr_thunk: Callable,
    num_consts: int,
    symbolic_zeros: bool,
):
  del jvp_jaxpr_thunk
  if symbolic_zeros: raise NotImplementedError
  if num_consts: raise NotImplementedError
  if call_jaxpr.consts: raise NotImplementedError
  lowering_context = ctx.lowering_context.replace(block_shapes=ctx.block_shapes)
  return jaxpr_subcomp(lowering_context, call_jaxpr.jaxpr, *args)


lowering_rules[custom_derivatives.custom_jvp_call_p] = (
    _custom_jvp_call_lowering_rule)


def _debug_callback_lowering_rule(ctx: LoweringRuleContext, *args, **kwargs):
  del ctx, args, kwargs
  # No-op debug callbacks in Mosaic for now
  return []


lowering_rules[debugging.debug_callback_p] = _debug_callback_lowering_rule


def _program_id_lowering_rule(ctx: LoweringRuleContext, *, axis: int):
  if ctx.lowering_context.user_grid_indices is None:
    raise ValueError(
        f"program id: {axis} was passed, but user did not provide a grid."
    )
  length = len(ctx.lowering_context.user_grid_indices)
  if not (0 <= axis < length):
    raise ValueError(
        f"user passed in program id with axis: {axis}, but grid only has"
        f" length: {length}"
    )
  return ctx.lowering_context.user_grid_indices[axis]
lowering_rules[primitives.program_id_p] = _program_id_lowering_rule

def _num_programs_lowering_rule(ctx: LoweringRuleContext, *, axis: int):
  mapped_axes = set(ctx.lowering_context.mapped_dims)
  seen_user_axes = 0
  for i in range(ctx.lowering_context.grid_rank):
    seen_user_axes += int(i not in mapped_axes)
    if seen_user_axes == axis + 1:
      break
  else:
    raise ValueError(
        f"user passed in program id with axis: {axis}, but grid only has"
        f" length: {len(ctx.lowering_context.grid_rank)}"
    )
  return tpu.iteration_bound(i)
lowering_rules[primitives.num_programs_p] = _num_programs_lowering_rule


def _repeat_lowering_rule(ctx: LoweringRuleContext, x, *, repeats, axis):
  (out_aval,) = ctx.avals_out
  return tpu.RepeatOp(aval_to_ir_type(out_aval), x, axis, repeats).result


lowering_rules[tpu_primitives.repeat_p] = _repeat_lowering_rule


def _roll_lowering_rule(
    ctx: LoweringRuleContext, x, shift, *, axis, stride, stride_axis
):
  (out_aval,) = ctx.avals_out
  return tpu.DynamicRotateOp(
      aval_to_ir_type(out_aval),
      x,
      shift,
      axis,
      stride=stride,
      stride_dimension=stride_axis,
  ).result


lowering_rules[tpu_primitives.roll_p] = _roll_lowering_rule


def _slice_lowering_rule(
    ctx: LoweringRuleContext, x, limit_indices, start_indices, strides
):
  """Lowers a slice to vector dialect."""
  (aval_out,) = ctx.avals_out
  if strides is None:
    strides = [1] * len(start_indices)
  sizes = np.array(limit_indices) - np.array(start_indices)
  op = vector.ExtractStridedSliceOp(
      aval_to_ir_type(aval_out), x, start_indices, sizes, strides
  )
  return op.result


lowering_rules[lax.slice_p] = _slice_lowering_rule


def _xor_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return arith.XOrIOp(x, y).result


lowering_rules[lax.xor_p] = _xor_lowering_rule
skip_mlir_conversions.add(lax.xor_p)


def _shift_left_lowering_rule(ctx: LoweringRuleContext, x, d):
  x, d = _bcast(x, d, *ctx.avals_in, *ctx.avals_out)
  return arith.ShLIOp(x, d).result


lowering_rules[lax.shift_left_p] = _shift_left_lowering_rule
skip_mlir_conversions.add(lax.shift_left_p)


def _shift_right_arithmetic_lowering_rule(ctx: LoweringRuleContext, x, d):
  x, d = _bcast(x, d, *ctx.avals_in, *ctx.avals_out)
  return arith.ShRSIOp(x, d).result


lowering_rules[lax.shift_right_arithmetic_p] = _shift_right_arithmetic_lowering_rule
skip_mlir_conversions.add(lax.shift_right_arithmetic_p)


def _shift_right_logical_lowering_rules(ctx: LoweringRuleContext, x, d):
  x, d = _bcast(x, d, *ctx.avals_in, *ctx.avals_out)
  return arith.ShRUIOp(x, d).result


lowering_rules[lax.shift_right_logical_p] = _shift_right_logical_lowering_rules
skip_mlir_conversions.add(lax.shift_right_logical_p)


# based on https://github.com/openxla/xla/blob/a7a09d56c3599123f8148bbf3e44c9ebc04624b9/xla/mlir_hlo/mhlo/transforms/chlo_legalize_to_hlo/chlo_legalize_to_hlo.cc#L644-L802
def _erf_inv_32_helper(x):
  k_degree = 9
  w_lt_5_constants = [
    2.81022636e-08,  3.43273939e-07, -3.5233877e-06,
    -4.39150654e-06, 0.00021858087,  -0.00125372503,
    -0.00417768164,  0.246640727,    1.50140941,
  ]
  w_gt_5_constants = [
    -0.000200214257, 0.000100950558, 0.00134934322,
    -0.00367342844,  0.00573950773,  -0.0076224613,
    0.00943887047,   1.00167406,     2.83297682,
  ]

  w = -jnp.log1p(x * -x)
  w_lt_5 = w < 5.0

  w = jnp.where(w_lt_5, w - 2.5, jnp.sqrt(w) - 3.0)

  p = jnp.where(w_lt_5, w_lt_5_constants[0], w_gt_5_constants[0])
  for i in range(1, k_degree):
    c = jnp.where(w_lt_5, w_lt_5_constants[i], w_gt_5_constants[i])
    p = c + p * w

  return jnp.where(jnp.abs(x) == 1.0, jnp.inf * x, p * x)


def _erf_inv_lowering_rule(ctx: LoweringRuleContext, x):
  (x_aval,) = ctx.avals_in
  if x_aval.dtype == jnp.float32:
    return lower_fun(_erf_inv_32_helper, multiple_results=False)(ctx, x)
  else:
    raise NotImplementedError


lowering_rules[lax.erf_inv_p] = _erf_inv_lowering_rule


def _bitcast_lowering_rule(ctx: LoweringRuleContext, x, *, ty):
  del ty
  (out_aval,) = ctx.avals_out
  return tpu.BitcastOp(aval_to_ir_type(out_aval), x).result

lowering_rules[tpu_primitives.bitcast_p] = _bitcast_lowering_rule

def _bitcast_convert_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype):
  (in_aval, ) = ctx.avals_in
  (out_aval,) = ctx.avals_out
  if in_aval.dtype.itemsize != new_dtype.itemsize:
    raise NotImplementedError("Changing bitwidths not supported.")
  return tpu.BitcastOp(aval_to_ir_type(out_aval), x).result
lowering_rules[lax.bitcast_convert_type_p] = _bitcast_convert_type_lowering_rule

def _alloc_value(aval: jax_core.AbstractValue) -> ir.Value:
  if isinstance(aval, pl_core.AbstractMemoryRef):
    memspace = ir.Attribute.parse(f"#tpu.memory_space<{aval.memory_space}>")
    if jnp.issubdtype(aval.dtype, tpu_core.semaphore_dtype):
      assert aval.memory_space == TPUMemorySpace.SEMAPHORE
      memref_type = aval_to_ir_type(aval, memory_space=TPUMemorySpace.SEMAPHORE)
      return tpu.AllocaSemaphoreOp(memref_type).result
    else:
      out_type = ir.MemRefType.get(
          aval.shape,
          _dtype_to_ir_type(aval.dtype, is_kernel_boundary=True),
          memory_space=memspace)
      return memref.AllocaOp(out_type, [], []).result
  elif isinstance(aval, tpu_core.AbstractSemaphore):
    memref_type = aval_to_ir_type(aval, memory_space=TPUMemorySpace.SEMAPHORE)
    return tpu.AllocaSemaphoreOp(memref_type).result
  raise NotImplementedError(f"Cannot allocate {type(aval)}.")


def _run_scoped_lowering_rule(ctx: LoweringRuleContext, *consts, jaxpr):
  out_type = [aval_to_ir_type(aval) for aval in ctx.avals_out]
  region = tpu.RegionOp(out_type)
  in_avals = [v.aval for v in jaxpr.invars]
  with ctx.lowering_context.grid_name_context():
    jaxpr = pe.convert_constvars_jaxpr(jaxpr)
  with ir.InsertionPoint(region.body):
    args = map(_alloc_value, in_avals)
    block_shapes = tuple(a.shape if isinstance(a, state.AbstractRef) else None
                         for a in in_avals)
    ctx = ctx.lowering_context.replace(
        block_shapes=(*ctx.block_shapes, *block_shapes)
    )
    out = jaxpr_subcomp(ctx, jaxpr, *consts, *args)
    tpu.YieldOp(out)
  return region.results


lowering_rules[primitives.run_scoped_p] = _run_scoped_lowering_rule

def _device_id_to_logical(
    ctx: LoweringRuleContext, device_id,
    device_id_type: tpu_primitives.DeviceIdType):
  if device_id_type is tpu_primitives.DeviceIdType.MESH:
    # Mesh means we are passed the mesh coordinates for the device
    device_ids = tree_util.tree_leaves(device_id)
    mesh_strides = ctx.lowering_context.mesh_context.mesh_strides
    def _linearize_mesh_indices(*indices):
      return sum(a * b for a, b in zip(indices, mesh_strides))
    lower_ctx = LoweringRuleContext(
        lowering_context=ctx.lowering_context,
        avals_in=[pl_core.index_map_grid_aval] * len(device_ids),
        avals_out=[pl_core.index_map_grid_aval],
        block_shapes=(None,) * len(device_ids),
    )
    return lower_fun(_linearize_mesh_indices, multiple_results=False)(
        lower_ctx, *device_ids)
  elif device_id_type is tpu_primitives.DeviceIdType.LOGICAL:
    return device_id
  raise NotImplementedError(f"Unsupported device id type: {device_id_type}")


def _semaphore_read_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    args_tree,
):
  sem_aval, _ = tree_util.tree_unflatten(args_tree, ctx.avals_in)
  sem, indexers = tree_util.tree_unflatten(args_tree, args)
  sem, _ = _index_ref(sem, sem_aval, sem_aval.shape, indexers)
  return tpu.SemaphoreReadOp(sem).result


lowering_rules[tpu_primitives.semaphore_read_p] = _semaphore_read_lowering_rule

def _semaphore_signal_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    args_tree,
    device_id_type: tpu_primitives.DeviceIdType,
):
  sem_aval, _, _, _, _ = tree_util.tree_unflatten(args_tree, ctx.avals_in)
  sem, indexers, value, device_id, core_index = tree_util.tree_unflatten(args_tree, args)
  sem, _ = _index_ref(sem, sem_aval, sem_aval.shape, indexers)
  if device_id is not None:
    device_id = _device_id_to_logical(ctx, device_id, device_id_type)
  return tpu.SemaphoreSignalOp(
      sem, value, device_id=device_id, core_id=core_index
  ).results


lowering_rules[tpu_primitives.semaphore_signal_p] = (
    _semaphore_signal_lowering_rule)


def _semaphore_wait_lowering_rule(ctx: LoweringRuleContext, *args, args_tree):
  sem_aval, _, _ = tree_util.tree_unflatten(args_tree, ctx.avals_in)
  sem, indexers, value = tree_util.tree_unflatten(args_tree, args)
  sem, _ = _index_ref(sem, sem_aval, sem_aval.shape, indexers)
  return tpu.SemaphoreWaitOp(sem, value).results
lowering_rules[tpu_primitives.semaphore_wait_p] = _semaphore_wait_lowering_rule

def _dma_start_lowering_rule(ctx: LoweringRuleContext, *args, tree,
                             device_id_type: tpu_primitives.DeviceIdType):
  (
      src_ref,
      src_indexers,
      dst_ref,
      dst_indexers,
      sem,
      sem_indexers,
      src_sem,
      src_sem_indexers,
      device_id,
  ) = tree_util.tree_unflatten(tree, args)
  (src_ref_aval, _, dst_ref_aval, _, sem_aval, _, src_sem_aval, _, _) = (
      tree_util.tree_unflatten(tree, ctx.avals_in)
  )
  if src_ref_aval.dtype == jnp.bool_:
    raise NotImplementedError("DMAs with bool dtypes are not supported.")
  block_shapes = tree_util.tree_unflatten(tree, ctx.block_shapes)
  src_ref_block_shape, dst_ref_block_shape = block_shapes[0], block_shapes[2]
  src_ref, _ = _index_ref(
      src_ref, src_ref_aval, src_ref_block_shape, src_indexers
  )
  if src_sem is not None:
    src_sem, _ = _index_ref(
        src_sem, src_sem_aval, src_sem_aval.shape, src_sem_indexers)
  dst_ref, _ = _index_ref(
      dst_ref, dst_ref_aval, dst_ref_block_shape, dst_indexers
  )
  sem, _ = _index_ref(sem, sem_aval, sem_aval.shape, sem_indexers)
  if device_id is not None:
    device_id = _device_id_to_logical(ctx, device_id, device_id_type)
  return tpu.EnqueueDMAOp(src_ref, dst_ref, sem, source_semaphore=src_sem,
                          device_id=device_id).results
lowering_rules[tpu_primitives.dma_start_p] = _dma_start_lowering_rule


def _dma_wait_lowering_rule(ctx: LoweringRuleContext, *args, tree,
                            device_id_type: tpu_primitives.DeviceIdType):
  del device_id_type
  sem, sem_indexers, ref, indexers = tree_util.tree_unflatten(tree, args)
  sem_aval, _, ref_aval, _ = tree_util.tree_unflatten(tree, ctx.avals_in)
  block_shapes = tree_util.tree_unflatten(tree, ctx.block_shapes)
  ref_block_shape = block_shapes[2]
  ref, _ = _index_ref(
      ref, ref_aval, ref_block_shape, indexers
  )
  sem, _ = _index_ref(sem, sem_aval, sem_aval.shape, sem_indexers)
  return tpu.WaitDMAOp(sem, ref).results
lowering_rules[tpu_primitives.dma_wait_p] = _dma_wait_lowering_rule

def _device_id_lowering_rule(ctx: LoweringRuleContext):
  return tpu.DeviceIdOp().result
lowering_rules[tpu_primitives.device_id_p] = _device_id_lowering_rule

def _axis_index_rule(ctx: LoweringRuleContext, *, axis_name: Hashable):
  grid_names = ctx.lowering_context.grid_names
  if grid_names and axis_name in grid_names:
    # We are querying a named axis corresponding to a grid dimension.
    return _program_id_lowering_rule(ctx, axis=grid_names.index(axis_name))
  # We are querying a named axis corresponding to a mesh dimension.
  device_id = tpu.DeviceIdOp().result
  mesh_context = ctx.lowering_context.mesh_context
  if mesh_context is None:
    raise ValueError("Mesh context is not set.")
  mesh_shape = mesh_context.mesh_shape
  axis_names = mesh_context.axis_names
  axis_index = axis_names.index(axis_name)
  axis_size = ir_constant(mesh_shape[axis_index])
  minor_divisor = ir_constant(
      np.prod(mesh_shape[axis_index + 1 :], dtype=np.int32)
  )
  return arith.remsi(arith.divsi(device_id, minor_divisor), axis_size)
lowering_rules[lax.axis_index_p] = _axis_index_rule

def _get_barrier_semaphore_rule(ctx: LoweringRuleContext):
  memref_type = aval_to_ir_type(ctx.avals_out[0])
  return tpu.GetBarrierSemaphoreOp(memref_type).result
lowering_rules[tpu_primitives.get_barrier_semaphore_p] = _get_barrier_semaphore_rule


def _delay_rule(ctx: LoweringRuleContext, nanos: int):
  return tpu.DelayOp(nanos).results


lowering_rules[tpu_primitives.delay_p] = _delay_rule


def _debug_print_rule(
    ctx: LoweringRuleContext, *args, fmt: str, has_placeholders: bool
):
  primitives.check_debug_print_format(fmt, *args)
  if has_placeholders:
    if not all(
        isinstance(arg.type, ir.IntegerType) and arg.type.width == 32
        for arg in args
    ):
      raise TypeError(
          "All arguments must be 32-bit integers when using"
          " placeholders (`{...}`). If you need to print values of other types,"
          " remove placeholders from the format string."
      )

    # TPU expects $0, $1 etc as placeholders.
    tpu_fmt = "".join(
        f"{text}${idx}"
        for idx, (text, _, _, _) in enumerate(string.Formatter().parse(fmt))
    )
  else:
    tpu_fmt = fmt
  tpu.log(args, tpu_fmt, formatted=has_placeholders)
  return ()


lowering_rules[primitives.debug_print_p] = _debug_print_rule


def _prng_seed_lowering_rule(ctx: LoweringRuleContext, *seeds):
  del ctx
  # In the KeyScalarBundle case we unpack the bundle and set the seed with
  # the list of scalars.
  if len(seeds) == 1 and isinstance(seeds[0], KeyScalarBundle):
    return tpu.PRNGSeed32Op(seeds[0].scalars).results
  # For integer seeds, we can set the seed directly as PRNGSeed32Op natively
  # takes in a list of integers as input.
  all_integers = all(isinstance(seed.type, ir.IntegerType) for seed in seeds)
  if not all_integers:
    seed_types = [seed.type for seed in seeds]
    raise ValueError(f"All seed data must be scalar integers. Got {seed_types}")
  return tpu.PRNGSeed32Op(seeds).results
lowering_rules[tpu_primitives.prng_seed_p] = _prng_seed_lowering_rule


def _prng_random_bits_lowering_rule(ctx: LoweringRuleContext, *, shape):
  if len(shape) <= 1:
    # TODO(b/342054464): Support implicit dims for PRNGRandomBitsOp.
    raise NotImplementedError("random_bits only supports rank>=2 outputs.")
  out_aval = ctx.avals_out[0]
  out_type = aval_to_ir_type(out_aval)
  return tpu.PRNGRandomBitsOp(out_type).result
lowering_rules[tpu_primitives.prng_random_bits_p] = _prng_random_bits_lowering_rule


def random_seed_lowering(ctx, seeds, *, impl):
  seed_lowering = lower_fun(
      impl.seed, multiple_results=False)
  return seed_lowering(ctx, seeds)
lowering_rules[prng.random_seed_p] = random_seed_lowering


def random_bits_lowering(ctx, keys, *, bit_width, shape):
  assert bit_width == 32, "Only 32-bit PRNG supported."
  aval, = ctx.avals_in
  impl = aval.dtype._impl
  bits_lowering = lower_fun(
      impl.random_bits, multiple_results=False)
  return bits_lowering(ctx, keys, bit_width=bit_width, shape=shape)
lowering_rules[prng.random_bits_p] = random_bits_lowering


def random_fold_in_lowering(ctx, keys, msgs):
  keys_aval, _ = ctx.avals_in
  impl = keys_aval.dtype._impl
  fold_in_lowering = lower_fun(
      impl.fold_in, multiple_results=False)
  return fold_in_lowering(ctx, keys, msgs)
lowering_rules[prng.random_fold_in_p] = random_fold_in_lowering


def random_unwrap_lowering(ctx, key):
  del ctx
  assert isinstance(key, KeyScalarBundle)
  # Convert to a vector.
  if tuple(key.key_shape) != (1, 1):
    raise NotImplementedError(
      "Seed key_data of shape != (1, 1) not supported. "
      f"Got: {key.key_shape}")
  scalar = key.scalars[0]
  out_type = ir.VectorType.get(
      key.key_shape, _dtype_to_ir_type(jnp.dtype('int32'))
  )
  val = vector.BroadcastOp(out_type, scalar).result
  return val
lowering_rules[prng.random_unwrap_p] = random_unwrap_lowering


def random_wrap_lowering(ctx, key_data, *, impl):
  del ctx, impl
  if isinstance(key_data.type, ir.VectorType):
    # If the key data lives in vregs, need to unpack it to sregs.
    key_data_list = []
    key_data_shape = key_data.type.shape
    if len(key_data_shape) != 2:
      raise NotImplementedError("Seed key_data must be 2D.")
    if tuple(key_data_shape) != (1, 1):
      raise NotImplementedError(
        "Seed key_data of shape != (1, 1) not supported. "
        f"Got: {key_data_shape}")
    for i in range(key_data_shape[1]):
      key_data_list.append(vector.ExtractOp(key_data, [], [0, i]))
    return KeyScalarBundle(
        scalars=key_data_list, key_shape=tuple(key_data_shape))
  if isinstance(key_data, KeyScalarBundle):
    return key_data
  else:
    raise NotImplementedError(f"key_data wrap {type(key_data)}")

lowering_rules[prng.random_wrap_p] = random_wrap_lowering


# Lowering for shard_map

# Technically this is not a lowering rule, but a discharge rule. When we use
# a special pallas mesh for a shard_map inside of a run_state, we turn it into
# a pallas call. The pallas_call has named grid axes corresponding to the names
# in the pallas mesh. It also sets up input/output aliasing automatically.

def _shard_map_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    auto,
    in_names,
    out_names,
    jaxpr,
    check_rep,
    rewrite,
):
  del out_avals, auto, in_names, out_names, check_rep, rewrite
  if not isinstance(mesh, pl_core.PallasMesh):
    raise NotImplementedError("Mesh must be a PallasMesh")
  if len(mesh.shape) > 1:
    raise NotImplementedError("Mesh must be 1D")
  core_axis_name, num_cores = list(mesh.shape.items())[0]
  def body(*args):
    in_refs = args[:len(in_avals)]
    jax_core.eval_jaxpr(jaxpr, (), *in_refs)
  assert len(jaxpr.outvars) == 0
  out = pallas_call.pallas_call(
      body,
      out_shape=in_avals,
      in_specs=[pl_core.BlockSpec(memory_space=tpu_core.TPUMemorySpace.ANY)]
      * len(in_avals),
      out_specs=[pl_core.BlockSpec(memory_space=tpu_core.TPUMemorySpace.ANY)]
      * len(in_avals),
      input_output_aliases={i: i for i in range(len(in_avals))},
      grid=((core_axis_name, num_cores),),
      compiler_params=dict(
          mosaic=dict(dimension_semantics=("parallel",)),
      ),
  )(*args)
  return out, ()


from jax.experimental import shard_map
state_discharge.register_discharge_rule(shard_map.shard_map_p)(
    _shard_map_discharge_rule
)
