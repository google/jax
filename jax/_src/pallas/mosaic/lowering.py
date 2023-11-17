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

import dataclasses
import functools
from typing import Any, Callable, Sequence

from jax import core as jax_core
from jax import lax
from jax import tree_util
from jax._src import custom_derivatives
from jax._src import debugging
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import pjit
from jax._src import source_info_util
from jax._src import state
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.lax.control_flow import for_loop
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import math
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
from jax._src.pallas import core
from jax._src.pallas import indexing
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax._src.state import discharge as state_discharge
from jax._src.state import primitives as state_primitives
from jax._src.util import safe_map
from jax._src.util import safe_zip
from jax._src.util import split_list
from jax._src.util import unzip2
from jax.experimental.mosaic.dialects import tpu
import jax.numpy as jnp
import numpy as np

# TODO(sharadmv): enable type checking
# mypy: ignore-errors

NDIndexer = indexing.NDIndexer
TPUMemorySpace = tpu_core.TPUMemorySpace
VMEM = tpu_core.TPUMemorySpace.VMEM
SMEM = tpu_core.TPUMemorySpace.SMEM

partial = functools.partial
map, unsafe_map = safe_map, map  # pylint: disable=redefined-builtin
zip, unsafe_zip = safe_zip, zip  # pylint: disable=redefined-builtin


@dataclasses.dataclass
class MeshContext:
  logical_to_mesh: ir.Value
  axis_names: tuple[str, ...]
  mesh_strides: tuple[int, ...]


@dataclasses.dataclass
class LoweringContext:
  ir_context: ir.Context
  grid_indices: Sequence[ir.Value] | None
  block_shapes: list[tuple[int | core.Mapped, ...]]
  name_stack: source_info_util.NameStack
  mesh_context: MeshContext | None
  replace = dataclasses.replace


@dataclasses.dataclass
class LoweringRuleContext:
  lowering_context: LoweringContext
  avals_in: Sequence[jax_core.AbstractValue]
  avals_out: Sequence[jax_core.AbstractValue]
  block_shapes: list[tuple[int | core.Mapped, ...]] | None

  replace = dataclasses.replace


def _memory_space_to_tpu_memspace(memory_space: TPUMemorySpace | None
                                  ) -> ir.Attribute:
  if memory_space is None:
    memory_space = VMEM
  return ir.Attribute.parse(f"#tpu.memory_space<{memory_space}>")


def aval_to_ir_type(aval, shape=None, memory_space: TPUMemorySpace | None = None):
  if isinstance(aval, tpu_core.AbstractSemaphore):
    if aval.sem_type == tpu_core.SemaphoreType.DMA:
      return ir.Type.parse("!tpu.dma_semaphore")
    elif aval.sem_type == tpu_core.SemaphoreType.REGULAR:
      return ir.Type.parse("!tpu.semaphore")
    raise NotImplementedError(aval.sem_type)
  if isinstance(aval, state.AbstractRef):
    if shape is None:
      shape = aval.shape
    memspace = _memory_space_to_tpu_memspace(memory_space)
    return ir.MemRefType.get(shape, mlir.dtype_to_ir_type(aval.dtype),
                             memory_space=memspace)
  if isinstance(aval, jax_core.ShapedArray):
    if shape is None:
      shape = aval.shape
    if not shape:
      return mlir.dtype_to_ir_type(aval.dtype)
    return ir.VectorType.get(shape, mlir.dtype_to_ir_type(aval.dtype))
  raise NotImplementedError(aval)


def ir_constant(x, mlir_type=None):
  if not hasattr(x, "dtype"):
    if isinstance(x, int):
      x = np.array(x, np.int32)
    elif isinstance(x, float):
      x = np.array(x, np.float32)
  if not mlir_type:
    mlir_type = mlir.dtype_to_ir_type(x.dtype)
  if isinstance(x, int) or x.dtype == np.int32 or x.dtype == np.uint32:
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


def _get_arg_type(
    aval,
    block_mapping: core.BlockMapping | None,
    memory_space: tpu_core.TPUMemorySpace | None,
):
  if isinstance(aval, tpu_core.AbstractMemoryRef):
    assert memory_space is None
    memory_space = aval.memory_space
  if isinstance(aval, tpu_core.AbstractSemaphore):
    return aval_to_ir_type(aval), None
  if block_mapping is None:
    return aval_to_ir_type(aval, memory_space=memory_space), aval.shape
  assert memory_space is None
  shape = tuple(1 if b is core.mapped else b for b in block_mapping.block_shape)
  return (
      aval_to_ir_type(aval, shape=shape, memory_space=block_mapping.memory_space),
      block_mapping.block_shape,
  )


@dataclasses.dataclass(init=False)
class MosaicGridMapping:
  grid: tuple[int, ...] | None
  jaxpr: jax_core.Jaxpr
  block_mappings: tuple[core.BlockMapping | None, ...]
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

  def __init__(self, jaxpr: jax_core.Jaxpr, grid_mapping: core.GridMapping,
               dimension_semantics: tuple[str, ...] | None,
               mesh: mesh_lib.Mesh | None):
    self.grid = grid_mapping.grid
    self.jaxpr = jaxpr
    self.block_mappings = grid_mapping.block_mappings
    self.mapped_dims = grid_mapping.mapped_dims
    num_scalar_prefetch = grid_mapping.num_index_operands
    num_scratch = grid_mapping.num_scratch_operands
    # jaxpr has signature [*scalar_prefetch, *in_ops *out_ops, *scratch]
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
    assert num_operands == len(
        self.block_mappings
    ), f"Misconfigured operands: {num_operands=}, {self.block_mappings=}"
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
        _get_arg_type(aval, None, memory_space=SMEM)
        for aval in scalar_prefetch_avals])
    self.scalar_prefetch_block_shapes = tuple(
        aval.shape for aval in scalar_prefetch_avals)
    self.operand_types, self.operand_block_shapes = unzip2([
        _get_arg_type(aval, block_mapping, memory_space=None)
        for aval, block_mapping in zip(operand_avals, self.block_mappings)])
    self.scratch_types, _ = unzip2([
        _get_arg_type(aval, None, memory_space=None) for aval in scratch_avals])
    self.scratch_block_shapes = tuple(
        aval.shape if not isinstance(aval, tpu_core.AbstractSemaphore) else None
        for aval in scratch_avals
    )
    self.grid_types, _ = unzip2([
        _get_arg_type(jax_core.ShapedArray((), jnp.int32), None,
                      memory_space=None)
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
    # We need mesh <-> logical translation tables. Since the logical IDs are
    # just linearized versions of the mesh IDs, we create those tables.
    mesh_strides = pallas_utils.strides_from_shape(tuple(
        mesh.shape[a] for a in axis_names
    ))
    logical_to_mesh = np.empty((mesh.size, len(axis_names)), dtype=np.int32)
    for i, idx in enumerate(np.ndindex(*mesh.device_ids.shape)):
      logical_to_mesh[i] = np.array(idx)
    self.mesh_info = MeshInfo(logical_to_mesh, axis_names, mesh_strides)
    l_to_m_aval = state.AbstractRef(
        jax_core.raise_to_shaped(jax_core.get_aval(logical_to_mesh))
    )
    # We are now passing in the logical -> mesh index mapping
    # TODO(sharadmv,apaszke): avoid stalling pipeline by marking the index
    # mapping as scalar prefetch and instead just mark it as an SMEM operand.
    self.scalar_prefetch_types = (
        _get_arg_type(l_to_m_aval, None, memory_space=SMEM)[0],
        *self.scalar_prefetch_types)

  def maybe_compress_grid(self):
    # TODO(sharadmv,apaszke,blakehechtman): remove this if this optimization
    #                                       gets implemented in mosaic
    # If we have many leading parallel dimensions, we should "compress" them
    # into one so we can load balance across cores as best as we can.
    # In general, we only have two cores, so we will compress only leading
    # odd-sized grid dimensions until we combine them with an even dimension.
    num_leading_parallel = 0
    done = False
    for dim_size, sem in zip(self.grid, self._dimension_semantics):
      if sem != "parallel":
        break
      if dim_size % 2 == 0:
        done = True
      num_leading_parallel += 1
      if done:
        # We hit an even dimension, so we stop compressing
        break

    if not num_leading_parallel:
      # If there are no dimensions to compress, don't do anything
      return

    parallel_dims, grid = split_list(self.grid, [num_leading_parallel])
    new_parallel_dim_size = np.prod(parallel_dims)
    self.grid = (new_parallel_dim_size, *grid)
    self._dimension_semantics = ("parallel", *self._dimension_semantics[num_leading_parallel:])
    self.grid_types, _ = unzip2([
        _get_arg_type(jax_core.ShapedArray((), jnp.int32), None,
                      memory_space=None)
        for _ in range(len(self.grid))
    ])
    def _get_grid_indices(indices):
      idx, *rest = indices
      out_indices = [None] * len(parallel_dims)
      for i, s in reversed(list(enumerate(parallel_dims))):
        s = ir_constant(s, mlir_type=mlir.dtype_to_ir_type(np.dtype("int32")))
        out_indices[i] = arith.RemSIOp(idx, s).result
        idx = arith.DivSIOp(idx, s).result
      assert None not in out_indices
      return (*out_indices, *rest)
    self.get_grid_indices = _get_grid_indices

  @functools.cached_property
  def has_communication(self) -> bool:
    return bool(jax_core.used_axis_names_jaxpr(self.jaxpr))

  def get_extra_args(self) -> tuple[Any, ...]:
    if self.mesh_info is None:
      return ()
    return (self.mesh_info.logical_to_mesh,)

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
  logical_to_mesh: np.ndarray
  axis_names: list[str]
  mesh_strides: tuple[int, ...]

def lower_jaxpr_to_module(
    ctx: ir.Context,
    grid_mapping: core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    dimension_semantics: tuple[str | None, ...] | None,
    mesh: mesh_lib.Mesh | None = None
) -> ir.Module:
  mosaic_grid_mapping = MosaicGridMapping(
      jaxpr, grid_mapping, dimension_semantics, mesh)
  mosaic_grid_mapping.maybe_compress_grid()
  m = ir.Module.create()
  sym_tab = ir.SymbolTable(m.operation)
  func_op = lower_jaxpr_to_func(ctx, jaxpr, mosaic_grid_mapping=mosaic_grid_mapping,
                                name="main")
  m.body.append(func_op)
  sym_tab.insert(func_op)
  window_params = []
  grid = mosaic_grid_mapping.grid
  if grid:
    for i, bm in enumerate(grid_mapping.block_mappings):
      # TODO(sharadmv): generate default block mapping if left as no_block_spec
      if bm is None:
        raise NotImplementedError("Please specify block mappings if "
                                  "grid is specified.")
      if bm.index_map_jaxpr  is None:
        raise NotImplementedError("Please specify index_maps if "
                                  "grid is specified.")
      func_name = f"transform_{i}"
      if bm.index_map_jaxpr.consts:
        raise NotImplementedError("Index map jaxpr with consts not supported.")
      mlir_func = lower_jaxpr_to_transform_func(
          ctx,
          bm.index_map_jaxpr.jaxpr,
          name=func_name,
          mosaic_grid_mapping=mosaic_grid_mapping,
      )
      assert mlir_func.verify(), mlir_func
      block_shape = [
          1 if b is core.mapped else b for b in bm.block_shape
      ]
      window_shape = ir.DenseI64ArrayAttr.get(block_shape)
      window_params.append(
          ir.DictAttr.get(
              dict(
                  window_bounds=window_shape,
                  transform_indices=ir.FlatSymbolRefAttr.get(func_name),
              )
          )
      )
      m.body.append(mlir_func)
      sym_tab.insert(mlir_func)
    func_op.attributes["window_params"] = ir.ArrayAttr.get(window_params)
    func_op.attributes["iteration_bounds"] = ir.DenseI64ArrayAttr.get(grid)

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
    *,
    name: str,
    mosaic_grid_mapping: MosaicGridMapping,
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
      (l_to_m,), scalar_prefetch = split_list(scalar_prefetch, [1])
      mesh_context = MeshContext(l_to_m, mesh_info.axis_names,
                                 mesh_info.mesh_strides)
    else:
      mesh_context = None
    lowering_context = LoweringContext(
        ctx,
        None,
        arg_block_shapes,
        source_info_util.NameStack(),
        mesh_context=mesh_context,
    )
    return jaxpr_subcomp(lowering_context, jaxpr, *jaxpr_indices,
                         *scalar_prefetch)
  body_func.__name__ = name
  body = func.FuncOp.from_py_func(*arg_types, name=name)(body_func)
  body.func_op.verify()
  return body.func_op


def lower_jaxpr_to_func(
    ctx: ir.Context,
    jaxpr: jax_core.Jaxpr,
    *,
    mosaic_grid_mapping: MosaicGridMapping,
    name: str,
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
      (l_to_m,), scalar_prefetch = split_list(scalar_prefetch, [1])
      mesh_context = MeshContext(l_to_m, mesh_info.axis_names,
                                 mesh_info.mesh_strides)
    else:
      mesh_context = None
    lowering_context = LoweringContext(
        ctx,
        jaxpr_indices,
        arg_block_shapes,
        source_info_util.NameStack(),
        mesh_context=mesh_context,
    )
    return jaxpr_subcomp(
        lowering_context, jaxpr, *scalar_prefetch, *operands_and_scratch
    )
  body_func.__name__ = name
  body = func.FuncOp.from_py_func(*arg_types, name=name)(body_func)
  body.func_op.verify()
  return body.func_op


def lower_fun(fun: Callable, *, multiple_results: bool) -> Callable:
  def f_lowered(ctx: LoweringRuleContext, *args, **params):
    f = fun if multiple_results else lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params)
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
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
    assert isinstance(val, ir.Value), type(val)
    env[var] = val

  for invar, bs in zip(jaxpr.invars, ctx.block_shapes):
    block_shape_env[invar] = bs
  map(write_env, jaxpr.invars, args)

  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    source_info = eqn.source_info.replace(
        name_stack=ctx.name_stack + eqn.source_info.name_stack
    )
    loc = mlir._source_info_to_location(
        eqn.primitive, eqn.params, source_info, ctx.name_stack
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
  outvals = map(read_env, jaxpr.outvars)
  outvals = [
      ir_constant(x) if isinstance(var, jax_core.Literal) else x
      for x, var in zip(outvals, jaxpr.outvars)
  ]
  return outvals


def _ensure_mlir_value(val, aval):
  if isinstance(val, ir.Value):
    return val
  elif isinstance(val, (np.generic, np.ndarray, int, float)):
    return ir_constant(val, mlir.dtype_to_ir_type(aval.dtype))
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
    ctx: LoweringRuleContext, ref, *non_slice_idx, indexed_dims: Sequence[bool]
):
  # Call _load_lowering_rule (since it's more general)
  ref_aval, *non_slice_idx_avals = ctx.avals_in
  nd_indexer, nd_indexer_avals = _convert_flat_indexing_to_indexer(
      ref_aval, non_slice_idx, non_slice_idx_avals, indexed_dims)
  args_flat, args_tree = tree_util.tree_flatten((ref, nd_indexer, None, None))
  avals_flat = tree_util.tree_leaves((ref_aval, nd_indexer_avals, None, None))
  ctx = ctx.replace(avals_in=avals_flat)
  return _load_lowering_rule(ctx, *args_flat, args_tree=args_tree)


lowering_rules[state_primitives.get_p] = _get_lowering_rule
skip_mlir_conversions.add(state_primitives.get_p)


def _swap_lowering_rule(
    ctx: LoweringRuleContext,
    ref,
    val,
    *non_slice_idx,
    indexed_dims: Sequence[bool],
):
  # Call _masked_swap_lowering_rule (since it's more general)
  ref_aval, val_aval, *non_slice_idx_avals = ctx.avals_in
  nd_indexer, nd_indexer_avals = _convert_flat_indexing_to_indexer(
      ref_aval, non_slice_idx, non_slice_idx_avals, indexed_dims)
  args_flat, args_tree = tree_util.tree_flatten((ref, nd_indexer, val, None))
  avals_flat = tree_util.tree_leaves(
      (ref_aval, nd_indexer_avals, val_aval, None)
  )
  ctx = ctx.replace(avals_in=avals_flat)
  return _masked_swap_lowering_rule(ctx, *args_flat, args_tree=args_tree)

lowering_rules[state_primitives.swap_p] = _swap_lowering_rule
skip_mlir_conversions.add(state_primitives.swap_p)


def _make_index(s):
  if isinstance(s, (int, np.ndarray)):
    return ir_constant(s, ir.IndexType.get())
  if s.type == ir.IndexType.get():
    return s
  return arith.IndexCastOp(ir.IndexType.get(), s).result


def _load_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree, **_):
  ref, idx, mask, _ = args_tree.unflatten(args_flat)
  _, idx_aval, _, _ = args_tree.unflatten(ctx.avals_in)

  if mask is not None:
    raise NotImplementedError

  ref_type = ir.MemRefType(ref.type)
  is_smem_load = str(ref_type.memory_space) == "#tpu.memory_space<smem>"
  ref_aval, *_ = ctx.avals_in
  (aval_out,) = ctx.avals_out
  ref_block_shape, *_ = ctx.block_shapes
  if not is_smem_load and not ref_block_shape:
    raise NotImplementedError(
        "Indexing into a ()-shaped Ref not yet supported on TPU.")
  if any(
      (not isinstance(a, primitives.Slice) and a.shape)
      for a in idx_aval.indices
  ):
    raise ValueError("Cannot do int indexing on TPU")
  starts = tuple(
      i.start if isinstance(i, primitives.Slice) else i for i in idx.indices
  )
  mlir_indices = [
      s if isinstance(s, primitives.Slice) else _make_index(s) for s in starts
  ]
  # Need to now insert indexing the 0-th element for mapped dimensions
  idx_iter = iter(mlir_indices)
  mlir_indices = [
      _make_index(0) if b is core.mapped else next(idx_iter)
      for b in ref_block_shape
  ]
  assert len(mlir_indices) == len(ref_block_shape)
  load_shape = list(aval_out.shape)
  for i, a in enumerate(idx_aval.indices):
    if not isinstance(a, primitives.Slice):
      load_shape.insert(i, 1)
  assert len(load_shape) == len(ref_aval.shape)
  load_shape_iter = iter(load_shape)
  load_shape = [
      1 if b is core.mapped else next(load_shape_iter) for b in ref_block_shape
  ]
  load_aval = aval_out.update(shape=tuple(load_shape))
  if is_smem_load:
    if ctx.avals_out[0].shape:
      raise ValueError("Can only load scalars from SMEM")
    return memref.LoadOp(ref, mlir_indices).result
  else:
    load_val = vector.LoadOp(aval_to_ir_type(load_aval), ref, mlir_indices).result
  if load_aval == aval_out:
    return load_val
  vec_type = ir.VectorType.get(aval_out.shape,
                               mlir.dtype_to_ir_type(aval_out.dtype))
  return vector.ShapeCastOp(vec_type, load_val).result


lowering_rules[primitives.load_p] = _load_lowering_rule
skip_mlir_conversions.add(primitives.load_p)


def _masked_swap_lowering_rule(
    ctx: LoweringRuleContext, *args_flat, args_tree, **_
):
  ref, idx, val, mask = args_tree.unflatten(args_flat)
  _, idx_aval, _, _ = args_tree.unflatten(ctx.avals_in)

  if mask is not None:
    raise NotImplementedError

  ref_type = ir.MemRefType(ref.type)
  is_smem_store = str(ref_type.memory_space) == "#tpu.memory_space<smem>"
  ref_block_shape, *_ = ctx.block_shapes
  _, val_aval, *_ = ctx.avals_in
  (aval_out,) = ctx.avals_out
  if not isinstance(val, ir.Value):
    val = ir_constant(val, mlir_type=mlir.dtype_to_ir_type(val_aval.dtype))
  if any(
      (not isinstance(a, primitives.Slice) and a.shape)
      for a in idx_aval.indices
  ):
    raise ValueError("Cannot do int indexing on TPU")
  if not is_smem_store and not ref_block_shape:
    raise NotImplementedError(
        "Indexing into a ()-shaped Ref not yet supported on TPU.")
  starts = tuple(
      i.start if isinstance(i, primitives.Slice) else i for i in idx.indices
  )
  mlir_indices = [
      s if isinstance(s, primitives.Slice) else _make_index(s) for s in starts
  ]
  # Need to now insert indexing the 0-th element for mapped dimensions
  idx_iter = iter(mlir_indices)
  mlir_indices = [
      _make_index(0) if b is core.mapped else next(idx_iter)
      for b in ref_block_shape
  ]
  assert len(mlir_indices) == len(ref_block_shape)
  if is_smem_store:
    if val_aval.shape:
      raise ValueError("Can only store scalars to SMEM")
    result = memref.LoadOp(ref, mlir_indices).result
    memref.StoreOp(val, ref, mlir_indices)
    return result
  mem_slice_shape = list(aval_out.shape)
  for i, a in enumerate(idx_aval.indices):
    if not isinstance(a, primitives.Slice):
      mem_slice_shape.insert(i, 1)
  mem_slice_shape_iter = iter(mem_slice_shape)
  mem_slice_shape = [
      1 if b is core.mapped else next(mem_slice_shape_iter)
      for b in ref_block_shape
  ]
  mem_aval = aval_out.update(shape=tuple(mem_slice_shape))
  mem_aval_vec_type = ir.VectorType.get(mem_aval.shape,
                                        mlir.dtype_to_ir_type(mem_aval.dtype))
  result = vector.LoadOp(mem_aval_vec_type, ref, mlir_indices).result
  if mem_aval != aval_out:
    # We are slicing a scalar so provided dummy 1 indices
    result_vec_type = ir.VectorType.get(aval_out.shape,
                                        mlir.dtype_to_ir_type(aval_out.dtype))
    result = vector.ShapeCastOp(result_vec_type, result).result
    val_vec_type = ir.VectorType.get(mem_aval.shape,
                                     mlir.dtype_to_ir_type(mem_aval.dtype))
    val = vector.ShapeCastOp(val_vec_type, val).result
  vector.StoreOp(val, ref, mlir_indices)
  return result


lowering_rules[primitives.swap_p] = _masked_swap_lowering_rule
skip_mlir_conversions.add(primitives.swap_p)


def _multiple_of_lowering_rule(ctx: LoweringRuleContext, val, *, values):
  del ctx, values
  return val


lowering_rules[primitives.multiple_of_p] = _multiple_of_lowering_rule


def _reduce_max_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  (x_aval,) = ctx.avals_in
  out_type = aval_to_ir_type(ctx.avals_out[0])
  if jnp.issubdtype(x_aval.dtype, jnp.floating):
    kind = ir.Attribute.parse("#vector.kind<maxf>")
    val = ir.FloatAttr.get(ir.F32Type.get(), float("-inf"))
    identity = ir.DenseElementsAttr.get_splat(out_type, val)
  elif jnp.issubdtype(x_aval.dtype, jnp.signedinteger):
    kind = ir.Attribute.parse("#vector.kind<maxsi>")
    raise NotImplementedError
  elif jnp.issubdtype(x_aval.dtype, jnp.unsignedinteger):
    kind = ir.Attribute.parse("#vector.kind<maxui>")
    raise NotImplementedError
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


lowering_rules[lax.reduce_max_p] = _reduce_max_lowering_rule


def _reduce_sum_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  (x_aval,) = ctx.avals_in
  out_type = aval_to_ir_type(ctx.avals_out[0])
  if jnp.issubdtype(x_aval.dtype, jnp.floating):
    kind = ir.Attribute.parse("#vector.kind<add>")
    val = ir.FloatAttr.get(ir.F32Type.get(), 0.0)
    identity = ir.DenseElementsAttr.get_splat(out_type, val)
  elif jnp.issubdtype(x_aval.dtype, jnp.signedinteger):
    kind = ir.Attribute.parse("#vector.kind<add>")
    raise NotImplementedError
  elif jnp.issubdtype(x_aval.dtype, jnp.unsignedinteger):
    kind = ir.Attribute.parse("#vector.kind<add>")
    raise NotImplementedError
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


lowering_rules[lax.reduce_sum_p] = _reduce_sum_lowering_rule


def _broadcast_in_dim_lowering_rule(
    ctx: LoweringRuleContext, val, *, shape, broadcast_dimensions
):
  (aval_in,) = ctx.avals_in
  (aval_out,) = ctx.avals_out
  if broadcast_dimensions:
    out_shape_list = [1] * len(shape)
    for i, s in zip(broadcast_dimensions, aval_in.shape):
      out_shape_list[i] = s
    out_shape = tuple(out_shape_list)
    out_type = ir.VectorType.get(
        out_shape, mlir.dtype_to_ir_type(aval_out.dtype)
    )
    val = vector.ShapeCastOp(out_type, val).result
    if out_shape == aval_out.shape:
      return val
  out_type = ir.VectorType.get(
      aval_out.shape, mlir.dtype_to_ir_type(aval_out.dtype)
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
  if any(cls.isinstance(val_type) for cls in [ir.BF16Type, ir.F32Type]):
    val = ir.FloatAttr.get(val_type, 0.0)
  elif ir.IntegerType.isinstance(val_type):
    val = ir.IntegerAttr.get(val_type, 0)
  else:
    raise NotImplementedError(ctx.avals_out[0].dtype)
  if any(len(a.shape) != 2 for a in ctx.avals_in):
    raise NotImplementedError(ctx.avals_in)
  lhs_aval, _ = ctx.avals_in
  # This is really a matrix-vector product. It only looks like matrix-matrix.
  if lhs_dims == (1,) and rhs_dims == (1,) and ctx.avals_in[1].shape[0] == 1:
    if ctx.avals_in[0].shape != ctx.avals_in[1].shape:
      bcast_shape = jnp.broadcast_shapes(
          ctx.avals_in[0].shape, ctx.avals_out[0].shape
      )
      bcast_shape = ir.VectorType.get(
          list(bcast_shape), mlir.dtype_to_ir_type(ctx.avals_out[0].dtype)
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

_INT_DTYPES = {
    8: np.dtype(np.int8),
    16: np.dtype(np.int16),
    32: np.dtype(np.int32),
}


def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type
):
  del weak_type
  out_aval = ctx.avals_out[0]
  old_dtype = ctx.avals_in[0].dtype
  out_type = aval_to_ir_type(out_aval)
  if old_dtype == new_dtype:
    return x
  if jnp.issubdtype(old_dtype, jnp.floating) and jnp.issubdtype(
      new_dtype, jnp.floating
  ):
    if old_dtype.itemsize < new_dtype.itemsize:
      return arith.ExtFOp(out_type, x).result
    else:
      return arith.TruncFOp(out_type, x).result
  elif old_dtype == jnp.bool_ and jnp.issubdtype(new_dtype, jnp.integer):
    return arith.ExtUIOp(out_type, x).result
  elif jnp.issubdtype(old_dtype, jnp.signedinteger) and jnp.issubdtype(
      new_dtype, jnp.floating
  ):
    # TODO(sharadmv,apaszke): remove this when Mosaic handles SIToFP with
    #                         differing element bitwidths
    if old_dtype.itemsize < new_dtype.itemsize:
      ext_dtype = _INT_DTYPES[new_dtype.itemsize * 8]
      ext_type = aval_to_ir_type(out_aval.update(dtype=ext_dtype))
      x = arith.ExtSIOp(ext_type, x).result
    elif old_dtype.itemsize > new_dtype.itemsize:
      ext_dtype = _INT_DTYPES[new_dtype.itemsize * 8]
      ext_type = aval_to_ir_type(out_aval.update(dtype=ext_dtype))
      x = arith.TruncIOp(ext_type, x).result
    return arith.SIToFPOp(out_type, x).result
  elif jnp.issubdtype(old_dtype, jnp.signedinteger) and jnp.issubdtype(
      new_dtype, jnp.signedinteger
  ):
    if old_dtype.itemsize < new_dtype.itemsize:
      return arith.ExtSIOp(out_type, x).result
    else:
      return arith.TruncIOp(out_type, x).result
  elif jnp.issubdtype(old_dtype, jnp.floating) and jnp.issubdtype(
      new_dtype, jnp.signedinteger
  ):
    return arith.FPToSIOp(out_type, x).result
  raise NotImplementedError(f"Unsupported cast: {old_dtype} -> {new_dtype}")


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
  i64_type = ir.IntegerType.get_signless(64)
  transp = ir.ArrayAttr.get(
      [ir.IntegerAttr.get(i64_type, i) for i in permutation]
  )
  return vector.TransposeOp(out_type, x, transp).result


lowering_rules[lax.transpose_p] = _transpose_lowering_rule


def _bcast(x, y, x_aval, y_aval, out_aval):
  if isinstance(x, (np.ndarray, np.number, int, float)):
    if hasattr(y, "type") and y.type == ir.IndexType.get():
      mlir_type = y.type
    else:
      mlir_type = mlir.dtype_to_ir_type(x_aval.dtype)
    x = ir_constant(x, mlir_type)
  if isinstance(y, (np.ndarray, np.number, int, float)):
    if hasattr(x, "type") and x.type == ir.IndexType.get():
      mlir_type = x.type
    else:
      mlir_type = mlir.dtype_to_ir_type(y_aval.dtype)
    y = ir_constant(y, mlir_type)
  out_shape = list(out_aval.shape)
  if x_aval.shape != out_aval.shape:
    x_ty = ir.VectorType.get(out_shape, mlir.dtype_to_ir_type(x_aval.dtype))
    x = vector.BroadcastOp(x_ty, x)
  if y_aval.shape != out_aval.shape:
    y_ty = ir.VectorType.get(out_shape, mlir.dtype_to_ir_type(y_aval.dtype))
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


_cmpi_lowering_types = {
    lax.eq_p: 0,
    lax.ne_p: 1,
    lax.lt_p: 2,
    lax.le_p: 3,
    lax.gt_p: 4,
    lax.ge_p: 5,
}

_cmpf_lowering_types = {
    lax.eq_p: 1,
    lax.ne_p: 6,
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


def _and_lowering_rule(ctx: LoweringRuleContext, lhs, rhs):
  return arith.AndIOp(lhs, rhs).result


lowering_rules[lax.and_p] = _and_lowering_rule


def _or_lowering_rule(ctx: LoweringRuleContext, lhs, rhs):
  return arith.OrIOp(lhs, rhs).result


lowering_rules[lax.or_p] = _or_lowering_rule

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
          ir_constant(i, mlir_type=mlir.dtype_to_ir_type(jnp.dtype('int32'))),
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
    unroll: bool,
    num_consts: int,
    num_carry: int,
):
  # Can only handle fori_loop-like scans
  num_extensive = len(args) - num_consts - num_carry
  if num_extensive: raise NotImplementedError
  if reverse: raise NotImplementedError
  del linear, num_extensive, unroll, reverse

  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts: raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = (
      pallas_utils.pattern_match_scan_to_fori_loop(jaxpr, num_consts, num_carry)
      )
  consts, args = split_list(args, [num_consts])
  if has_loop_index:
    loop_index_start, *args = args
  else:
    loop_index_start = 0
  out = _lower_jaxpr_to_unrolled_for_loop(ctx, jaxpr, loop_index_start, length,
                                          consts, *args,
                                          has_loop_index=has_loop_index)
  if has_loop_index:
    out = [ir_constant(length,
                       mlir_type=mlir.dtype_to_ir_type(jnp.dtype('int32'))),
           *out]
  return out
lowering_rules[lax.scan_p] = _scan_lowering_rule
skip_mlir_conversions.add(lax.scan_p)


def _cond_lowering_rule(ctx: LoweringRuleContext, *args, branches, linear):
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
          linear=linear,
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
  if ctx.lowering_context.grid_indices is None:
    raise ValueError(
        f"program id: {axis} was passed, but user did not provide a grid."
    )
  length = len(ctx.lowering_context.grid_indices)
  if not (0 <= axis < length):
    raise ValueError(
        f"user passed in program id with axis: {axis}, but grid only has"
        f" length: {length}"
    )
  return ctx.lowering_context.grid_indices[axis]
lowering_rules[primitives.program_id_p] = _program_id_lowering_rule


def _repeat_lowering_rule(ctx: LoweringRuleContext, x, *, repeats, axis):
  (out_aval,) = ctx.avals_out
  return tpu.RepeatOp(aval_to_ir_type(out_aval), x, axis, repeats).result


lowering_rules[tpu_primitives.repeat_p] = _repeat_lowering_rule


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
  return arith.XOrIOp(x, y).result


lowering_rules[lax.xor_p] = _xor_lowering_rule


def _shift_left_lowering_rule(ctx: LoweringRuleContext, x, d):
  return arith.ShLIOp(x, d).result


lowering_rules[lax.shift_left_p] = _shift_left_lowering_rule


def _shift_right_logical_lowering_rules(ctx: LoweringRuleContext, x, d):
  return arith.ShRUIOp(x, d).result


lowering_rules[lax.shift_right_logical_p] = _shift_right_logical_lowering_rules


def _trace_start_lowering_rule(
    ctx: LoweringRuleContext, *, message: str, level: int
):
  return tpu.TraceStartOp(message=message, level=level).results


lowering_rules[tpu_primitives.trace_start_p] = _trace_start_lowering_rule


def _trace_stop_lowering_rule(ctx: LoweringRuleContext):
  return tpu.TraceStopOp().results


lowering_rules[tpu_primitives.trace_stop_p] = _trace_stop_lowering_rule


def _alloc_value(aval: jax_core.AbstractValue) -> ir.Value:
  if isinstance(aval, tpu_core.AbstractMemoryRef):
    memspace = ir.Attribute.parse(f"#tpu.memory_space<{aval.memory_space}>")
    out_type = ir.MemRefType.get(
        aval.shape, mlir.dtype_to_ir_type(aval.dtype), memory_space=memspace)
    return memref.AllocaOp(out_type, [], []).result
  elif isinstance(aval, tpu_core.AbstractSemaphore):
    if aval.sem_type is tpu_core.SemaphoreType.DMA:
      sem_type = ir.Type.parse("!tpu.dma_semaphore")
      return tpu.AllocaSemaphoreOp(sem_type).result
    elif aval.sem_type is tpu_core.SemaphoreType.REGULAR:
      sem_type = ir.Type.parse("!tpu.semaphore")
      return tpu.AllocaSemaphoreOp(sem_type).result
    raise ValueError(f"Cannot allocate {aval.sem_type}.")
  raise NotImplementedError(f"Cannot allocate {type(aval)}.")


def _run_scoped_lowering_rule(ctx: LoweringRuleContext, *consts, jaxpr):
  region = tpu.RegionOp()
  in_avals = [v.aval for v in jaxpr.invars]
  jaxpr = pe.convert_constvars_jaxpr(jaxpr)
  with ir.InsertionPoint(region.body):
    args = map(_alloc_value, in_avals)
    block_shapes = tuple(a.shape if isinstance(a, state.AbstractRef) else None
                         for a in in_avals)
    ctx = ctx.lowering_context.replace(
        block_shapes=(*ctx.block_shapes, *block_shapes)
    )
    jaxpr_subcomp(ctx, jaxpr, *consts, *args)
    tpu.YieldOp([])
  return []


lowering_rules[tpu_primitives.run_scoped_p] = _run_scoped_lowering_rule

def _device_id_to_logical(
    ctx: LoweringRuleContext, device_id,
    device_id_type: tpu_primitives.DeviceIdType):
  if device_id_type is tpu_primitives.DeviceIdType.MESH:
    # Mesh means we are passed the mesh coordinates for the device
    device_ids = tree_util.tree_leaves(device_id)
    mesh_strides = ctx.lowering_context.mesh_context.mesh_strides
    def _linearize_mesh_indices(*indices):
      return sum([a * b for a, b in zip(indices, mesh_strides)])
    lower_ctx = LoweringRuleContext(
        lowering_context=ctx.lowering_context,
        avals_in=[jax_core.ShapedArray((), jnp.int32)] * len(device_ids),
        avals_out=[jax_core.ShapedArray((), jnp.int32)],
        block_shapes=(None,) * len(device_ids),
    )
    return lower_fun(_linearize_mesh_indices, multiple_results=False)(
        lower_ctx, *device_ids)
  elif device_id_type is tpu_primitives.DeviceIdType.LOGICAL:
    return device_id
  raise NotImplementedError(f"Unsupported device id type: {device_id_type}")


def _semaphore_signal_lowering_rule(
    ctx: LoweringRuleContext,
    semaphore,
    value,
    *args,
    has_device_id: bool,
    device_id_type: tpu_primitives.DeviceIdType,
    device_id_tree,
):
  device_id = None
  assert semaphore.type == ir.Type.parse("!tpu.semaphore")
  if has_device_id:
    if len(args) == 1:
      device_id = args[0]
    else:
      device_id = tree_util.tree_unflatten(device_id_tree, args)
    device_id = _device_id_to_logical(ctx, device_id, device_id_type)
  return tpu.SemaphoreSignalOp(semaphore, value, device_id=device_id).results


lowering_rules[tpu_primitives.semaphore_signal_p] = (
    _semaphore_signal_lowering_rule)


def _semaphore_wait_lowering_rule(ctx: LoweringRuleContext, semaphore,
                                  value):
  sem_aval = ctx.avals_in[0]
  assert isinstance(sem_aval, tpu_core.AbstractSemaphore)
  assert sem_aval.sem_type is tpu_core.SemaphoreType.REGULAR
  assert ctx.avals_in[1].dtype == jnp.dtype('int32')
  return tpu.SemaphoreWaitOp(semaphore, value).results
lowering_rules[tpu_primitives.semaphore_wait_p] = _semaphore_wait_lowering_rule

def _indexer_to_start_size(indexer: NDIndexer):
  starts = [
      i.start if isinstance(i, primitives.Slice) else i
      for i in indexer.indices
  ]
  if any(jnp.shape(i) for i in starts):
    raise ValueError(f"Can only use integer and slice indexing: {starts}")
  starts = map(
      partial(_ensure_mlir_value, aval=jax_core.ShapedArray((), jnp.int32)),
      starts,
  )
  sizes = [
      s.size if isinstance(s, primitives.Slice) else 1 for s in indexer.indices
  ]
  return tuple(starts), tuple(sizes)

def _slice_memref(ref: ir.Value, ref_aval: state.AbstractRef, indexer: NDIndexer
                  ) -> ir.Value:
  target_shape = indexer.get_indexer_shape()
  starts, sizes = _indexer_to_start_size(indexer)
  target_ref_ty = ir.MemRefType.get(
      tuple(sizes), mlir.dtype_to_ir_type(ref_aval.dtype),
      memory_space=ref.type.memory_space)
  out = tpu.MemRefSliceOp(target_ref_ty, ref, starts).result
  if sizes != target_shape:
    # We need to squeeze out some dimensions
    squeezed_ref_ty = ir.MemRefType.get(
        tuple(target_shape), mlir.dtype_to_ir_type(ref_aval.dtype),
        memory_space=ref.type.memory_space)
    out = tpu.MemRefSqueezeOp(squeezed_ref_ty, out).result
  return out

def _dma_start_lowering_rule(ctx: LoweringRuleContext, *args, tree,
                             device_id_type: tpu_primitives.DeviceIdType):
  (src_ref, src_idx, dst_ref, dst_idx, sem, src_sem, device_id) = (
      tree_util.tree_unflatten(tree, args)
  )
  (src_ref_aval, src_idx_aval, dst_ref_aval, *_) = (
      tree_util.tree_unflatten(tree, ctx.avals_in)
  )
  del src_idx_aval
  src = _slice_memref(src_ref, src_ref_aval, src_idx)
  dst = _slice_memref(dst_ref, dst_ref_aval, dst_idx)
  if device_id is not None:
    device_id = _device_id_to_logical(ctx, device_id, device_id_type)
  return tpu.EnqueueDMAOp(src, dst, sem, source_semaphore=src_sem,
                          device_id=device_id).results
lowering_rules[tpu_primitives.dma_start_p] = _dma_start_lowering_rule


def _dma_wait_lowering_rule(ctx: LoweringRuleContext, *args, tree,
                            device_id_type: tpu_primitives.DeviceIdType):
  del device_id_type
  sem, ref, idx = tree_util.tree_unflatten(tree, args)
  sem_aval, ref_aval, idx_aval = tree_util.tree_unflatten(tree, ctx.avals_in)
  del sem_aval, idx_aval
  ref_slc = _slice_memref(ref, ref_aval, idx)
  return tpu.WaitDMAOp(sem, ref_slc).results
lowering_rules[tpu_primitives.dma_wait_p] = _dma_wait_lowering_rule

def _device_id_lowering_rule(ctx: LoweringRuleContext):
  return tpu.DeviceIdOp().result
lowering_rules[tpu_primitives.device_id_p] = _device_id_lowering_rule

def _axis_index_rule(ctx: LoweringRuleContext, *, axis_name: str):
  device_id = _make_index(tpu.DeviceIdOp().result)
  l_to_m = ctx.lowering_context.mesh_context.logical_to_mesh
  axis_names = ctx.lowering_context.mesh_context.axis_names
  col = _make_index(axis_names.index(axis_name))
  return memref.LoadOp(l_to_m, [device_id, col]).result
lowering_rules[lax.axis_index_p] = _axis_index_rule
