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

"""Module for lowering JAX primitives to Triton IR."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import math
import operator
from typing import Any, Hashable, TypeVar

import jax
from jax import lax
from jax import tree_util
from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import api_util
from jax._src import config
from jax._src import core as jax_core
from jax._src import custom_derivatives
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import source_info_util
from jax._src import state
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow import for_loop
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import math as math_dialect
from jax._src.lib.mlir.dialects import scf as scf_dialect
from jax._src.lib.triton import dialect as tt_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives as sp
from jax._src.util import merge_lists
from jax._src.util import partition_list
from jax._src.util import split_list
import jax.numpy as jnp
import numpy as np

# TODO(sharadmv): Enable type checking.
# mypy: ignore-errors
# pytype: skip-file

_T = TypeVar("_T")

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

NDIndexer = indexing.NDIndexer
GridMapping = pallas_core.GridMapping
BlockMapping = pallas_core.BlockMapping
Blocked = pallas_core.Blocked


# # General lowering logic
@dataclasses.dataclass
class ModuleContext:
  name: str
  grid_mapping: GridMapping
  program_ids: Sequence[ir.Value]
  traceback_caches: mlir.TracebackCaches = dataclasses.field(repr=False)
  platform: str


@dataclasses.dataclass
class BlockInfo:
  full_shape_dtype: jax.ShapeDtypeStruct
  start_indices: Sequence[Any]
  block_shape: tuple[int, ...]


@dataclasses.dataclass
class LoweringRuleContext:
  context: ModuleContext
  avals_in: Sequence[jax_core.ShapedArray]
  avals_out: Sequence[jax_core.ShapedArray]
  block_infos: Sequence[BlockInfo | None]

  replace = dataclasses.replace


@dataclasses.dataclass
class LoweringResult:
  """Keeps pybind11 objects alive."""

  module: ir.Module
  grid: tuple[int, ...]


class LoweringError(Exception):
  pass


def _eval_index_map(
    ctx: ModuleContext, idx, block_mapping: BlockMapping
):
  block_indices = lower_jaxpr_to_triton_ir(
      ctx, block_mapping.index_map_jaxpr.jaxpr, None, *idx
  )
  block_indices = (
      _ensure_ir_value(i, jax_core.ShapedArray((), jnp.int32))
      for i in block_indices
  )
  return tuple(
      i if b is pallas_core.mapped else _mul(i, _ir_constant(b, i.type))
      for i, b in zip(block_indices, block_mapping.block_shape)
  )


def _bcast_to(a: ir.Value, shape: tuple[int, ...]) -> ir.Value:
  if not ir.RankedTensorType.isinstance(a.type):
    if not shape:
      return a
    return tt_dialect.splat(ir.RankedTensorType.get(shape, a.type), a)
  else:
    a_type = ir.RankedTensorType(a.type)
    if a_type.shape == [*shape]:
      return a
    if a_type.rank != len(shape) or not all(
        a_type.shape[i] in (dim, 1) for i, dim in enumerate(shape)
    ):
      raise ValueError(f"Cannot broadcast from {a_type.shape} to {[*shape]}")
    return tt_dialect.broadcast(
        ir.RankedTensorType.get(shape, a_type.element_type, a_type.encoding), a
    )


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
    x = _ir_constant(x, _dtype_to_ir_type(x_dtype))
  if isinstance(y, (np.ndarray, np.number, int, float)):
    y_dtype = y_aval.dtype
    if y_aval.weak_type:
      y_dtype = x_aval.dtype
    y = _ir_constant(y, _dtype_to_ir_type(y_dtype))
  if x_aval.shape != out_aval.shape:
    x = _bcast_to(x, out_aval.shape)
  if y_aval.shape != out_aval.shape:
    y = _bcast_to(y, out_aval.shape)
  return x, y


triton_lowering_rules = {}


def register_lowering(primitive: jax_core.Primitive) -> Callable[[_T], _T]:
  def wrapper(fn):
    triton_lowering_rules[primitive] = fn
    return fn
  return wrapper


def _process_grid_to_3d_grid(grid_mapping: GridMapping):
  launch_grid = []
  launch_grid_to_pallas_grid = []

  # Preserve grid order provided to pallas_call
  for i, s in enumerate(grid_mapping.grid):
    if i not in grid_mapping.vmapped_dims:
      launch_grid.append(s)
      launch_grid_to_pallas_grid.append(i)

  # For mapped dims, iterate from inner to outer. This follows the pallas_call
  # batching rule that prepends the vmapped dimension.
  for dim in reversed(grid_mapping.vmapped_dims):
    s = grid_mapping.grid[dim]
    launch_grid.append(s)
    launch_grid_to_pallas_grid.append(dim)

  num_collapse = len(launch_grid[:-2])

  cuda_yz_limit = 2**16 - 1

  # Check Z and then Y launch dims to make sure they're within CUDA bounds
  if (num_collapse + 1 < len(launch_grid) and
      launch_grid[num_collapse + 1] > cuda_yz_limit):
    num_collapse += 2
  elif (num_collapse < len(launch_grid) and
        launch_grid[num_collapse] > cuda_yz_limit):
    num_collapse += 1

  collapse_dims = launch_grid[:num_collapse]
  prog_id_dims = launch_grid[num_collapse:]

  if len(collapse_dims) == 0:
    prog_ids = [None] * len(prog_id_dims)
    for i in range(len(prog_id_dims)):
      out_idx = launch_grid_to_pallas_grid[i]
      prog_ids[out_idx] = _program_id(i)

    return prog_id_dims, prog_ids
  else:
    new_grid = [math.prod(collapse_dims), *prog_id_dims]

  assert new_grid[0] < 2**31 - 1, \
          "Cannot fix pallas kernel launch grid within CUDA limits"

  out_indices = [None] * len(grid_mapping.grid)

  grid0 = _program_id(0)
  for i, s in enumerate(collapse_dims):
    out_idx = launch_grid_to_pallas_grid[i]
    s = _i32_constant(s)
    out_indices[out_idx] = _mod(grid0, s, signed=False)
    grid0 = _floordiv(grid0, s, signed=False)

  for i in range(len(prog_id_dims)):
    out_idx = launch_grid_to_pallas_grid[num_collapse + i]
    out_indices[out_idx] = _program_id(i + 1)

  assert len(out_indices) == len(grid_mapping.grid)
  return new_grid, out_indices


def _new_ir_context() -> ir.Context:
  ctx = ir.Context()
  tt_dialect.register_dialect(ctx)
  ctx.load_all_available_dialects()
  return ctx


def lower_jaxpr_to_triton_module(
    jaxpr: jax_core.Jaxpr,
    grid_mapping: GridMapping,
    name: str,
    platform: str
) -> LoweringResult:
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "dynamic grid bounds not supported in the Triton backend"
    )
  # TODO(necula): this fails many tests, figure it out
  # for bm in grid_mapping.block_mappings:
  #   block_size = math.prod(d for d in bm.block_shape  # type: ignore[misc]
  #                          if d is not pallas_core.mapped)
  #   power_of_2 = (block_size & (block_size - 1)) == 0

  #   if not power_of_2:
  #     raise ValueError(
  #         "The Pallas GPU lowering currently requires that the block sizes be "
  #         f"a power of 2. Found block spec for {bm.origin} with "
  #         f"shape {bm.block_shape}")
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "scalar prefetch not implemented in the Triton backend"
    )
  with grid_mapping.trace_env():
    jaxpr, _ = pe.dce_jaxpr(
        jaxpr, [True] * len(jaxpr.outvars), instantiate=True
    )
  with _new_ir_context(), ir.Location.unknown():
    module = ir.Module.create()
    param_types = [
        tt_dialect.PointerType.get(_dtype_to_ir_type(var.aval.dtype), 1)
        for var in jaxpr.invars
    ]
    assert len(jaxpr.outvars) == 0
    fn_type = ir.FunctionType.get(param_types, [])
    fn = tt_dialect.FuncOp(
        name,
        ir.TypeAttr.get(fn_type),
        sym_visibility="public",
        res_attrs=ir.DictAttr.get(dict(noinline=ir.BoolAttr.get(False))),
        ip=ir.InsertionPoint.at_block_begin(module.body),
    )
    fn.arg_attrs = ir.ArrayAttr.get(
        [ir.DictAttr.get({"tt.divisibility": mlir.i32_attr(32)})]
        * len(param_types)
    )
    fn.body.blocks.append(*fn_type.inputs)
    [entry] = fn.body.blocks
    with ir.InsertionPoint(entry):
      new_grid, program_ids = _process_grid_to_3d_grid(grid_mapping)
      local_program_ids = [
          pid
          for i, pid in enumerate(program_ids)
          if i not in grid_mapping.vmapped_dims
      ]
      ctx = ModuleContext(
          name, grid_mapping, local_program_ids, mlir.TracebackCaches(), platform
      )
      if grid_mapping.num_index_operands:
        raise NotImplementedError(
            "Scalar prefetch not supported in Triton lowering."
        )
      if not all(isinstance(bm.indexing_mode, Blocked)
                 for bm in grid_mapping.block_mappings):
        raise NotImplementedError(
            "Only Blocked indexing mode is supported in Triton lowering."
        )
      start_indices = map(
          functools.partial(_eval_index_map, ctx, program_ids),
          grid_mapping.block_mappings,
      )
      block_infos = [
          BlockInfo(
              block_mapping.array_shape_dtype,
              start_idx,
              block_mapping.block_shape,
          )
          for block_mapping, start_idx in zip(
              grid_mapping.block_mappings,
              start_indices,
          )
      ]
      () = lower_jaxpr_to_triton_ir(ctx, jaxpr, block_infos, *entry.arguments)
      tt_dialect.return_([])
    return LoweringResult(module, new_grid)


def lower_jaxpr_to_triton_ir(
    ctx: ModuleContext,
    jaxpr: jax_core.Jaxpr,
    block_infos: Sequence[BlockInfo | None] | None,
    *args,
) -> Sequence[Any]:
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

  if block_infos is not None:
    for invar, block_info in zip(jaxpr.invars, block_infos):
      block_info_env[invar] = block_info

  map(write_env, jaxpr.invars, args)

  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    if eqn.primitive not in triton_lowering_rules:
      raise NotImplementedError(
          "Unimplemented primitive in Pallas GPU lowering: "
          f"{eqn.primitive.name}. "
          "Please file an issue on https://github.com/google/jax/issues.")
    rule = triton_lowering_rules[eqn.primitive]
    avals_in = [v.aval for v in eqn.invars]
    avals_out = [v.aval for v in eqn.outvars]
    eqn_block_infos = map(read_block_info_env, eqn.invars)
    loc = mlir._source_info_to_location(
        ctx, eqn.primitive, eqn.params, eqn.source_info
    )
    rule_ctx = LoweringRuleContext(ctx, avals_in, avals_out, eqn_block_infos)
    try:
      with source_info_util.user_context(eqn.source_info.traceback), loc:
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


# # Primitive lowering rules
# ## Programming model primitives


def _program_id(axis: int) -> ir.Value:
  if axis not in range(3):
    raise ValueError(f"axis must be in [0, 3), but got: {axis}")
  return tt_dialect.get_program_id(axis)


@register_lowering(primitives.program_id_p)
def _program_id_lowering_rule(ctx: LoweringRuleContext, *, axis):
  return ctx.context.program_ids[axis]


@register_lowering(primitives.num_programs_p)
def _num_programs_lowering_rule(ctx: LoweringRuleContext, *, axis):
  if axis not in range(3):
    raise ValueError(f"axis must be in [0, 3), but got: {axis}")
  return tt_dialect.get_num_programs(axis)

def _atomic_rmw(
    op: tt_dialect.RMWOp,
    ptr: ir.Value,
    val: ir.Value,
    mask: ir.Value | None = None,
    semantic: tt_dialect.MemSemantic = tt_dialect.MemSemantic.ACQUIRE_RELEASE,
    sync_scope: tt_dialect.MemSyncScope = tt_dialect.MemSyncScope.GPU,
) -> ir.Value:
  if ir.RankedTensorType.isinstance(ptr.type):
    ptr_type = ir.RankedTensorType(ptr.type)
    element_type = tt_dialect.PointerType(ptr_type.element_type)
    result_type = ir.RankedTensorType.get(
        ptr_type.shape, element_type.pointee_type, ptr_type.encoding
    )
  else:
    result_type = tt_dialect.PointerType(ptr.type).pointee_type
  return tt_dialect.atomic_rmw(
      result_type, op, ptr, val, mask=mask, sem=semantic, scope=sync_scope
  )


@register_lowering(primitives.atomic_rmw_p)
def _atomic_lowering_rule(
    ctx: LoweringRuleContext,
    *args_flat,
    args_tree,
    atomic_type: primitives.AtomicOpType,
):
  ptr, indexers, val, mask = args_tree.unflatten(args_flat)
  *_, value_aval, mask_aval = args_tree.unflatten(ctx.avals_in)
  if len(indexers) != 1:
    raise NotImplementedError("Only single indexer is supported.")
  idx = indexers[0]
  ptr = _compute_pointers_from_indices(
      ptr, ctx.block_infos[0], idx, ctx.avals_in[0].shape
  )
  val = _ensure_ir_value(val, value_aval)
  if mask is not None:
    mask = _ensure_ir_value(mask, mask_aval)
  if atomic_type == primitives.AtomicOpType.XCHG:
    op = tt_dialect.RMWOp.XCHG
  elif atomic_type == primitives.AtomicOpType.ADD:
    if isinstance(val.type, ir.IntegerType):
      op = tt_dialect.RMWOp.ADD
    else:
      op = tt_dialect.RMWOp.FADD
  elif atomic_type == primitives.AtomicOpType.MIN:
    op = tt_dialect.RMWOp.MIN
  elif atomic_type == primitives.AtomicOpType.MAX:
    op = tt_dialect.RMWOp.MAX
  elif atomic_type == primitives.AtomicOpType.AND:
    op = tt_dialect.RMWOp.AND
  elif atomic_type == primitives.AtomicOpType.OR:
    op = tt_dialect.RMWOp.OR
  elif atomic_type == primitives.AtomicOpType.XOR:
    op = tt_dialect.RMWOp.XOR
  else:
    raise NotImplementedError(f"unsupported atomic operation: {atomic_type}")
  return _atomic_rmw(op, ptr, val, mask=mask)


@register_lowering(primitives.atomic_cas_p)
def _atomic_cas_lowering_rule(ctx: LoweringRuleContext, ptr, cmp, val):
  _, cmp_aval, val_aval = ctx.avals_in
  if ir.RankedTensorType.isinstance(ptr.type):
    ptr_type = ir.RankedTensorType(ptr.type)
    element_type = tt_dialect.PointerType(ptr_type.element_type)
    result_type = ir.RankedTensorType.get(
        ptr_type.shape, element_type.pointee_type, ptr_type.encoding
    )
  else:
    result_type = tt_dialect.PointerType(ptr.type).pointee_type
  return tt_dialect.atomic_cas(
      result_type,
      ptr,
      _ensure_ir_value(cmp, cmp_aval),
      _ensure_ir_value(val, val_aval),
      sem=tt_dialect.MemSemantic.ACQUIRE_RELEASE,
      scope=tt_dialect.MemSyncScope.GPU,
  )


def _associative_scan_lowering(body, ctx: LoweringRuleContext, args, axes):
  flat_args = tree_util.tree_leaves(args)
  (axis,) = axes
  dtype = ctx.avals_in[0].dtype
  in_avals = [
      jax_core.ShapedArray((), dtype=dtype),
      jax_core.ShapedArray((), dtype=dtype),
  ]
  in_tree = tree_util.tree_structure((args, args))
  flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(body), in_tree
  )
  combine_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      flat_fun, in_avals
  )
  out_tree = out_tree_thunk()
  del out_tree  # Not needed
  if consts:
    raise NotImplementedError("Associative scan with constants not supported.")
  element_types = [_element_type(arg.type) for arg in flat_args]
  scan_op = tt_dialect.ScanOp(flat_args, axis)
  param_types = element_types * 2
  entry = scan_op.regions[0].blocks.append(*param_types)
  with ir.InsertionPoint.at_block_begin(entry):
    results = lower_jaxpr_to_triton_ir(
        ctx.context, combine_jaxpr, None, *entry.arguments
    )
    tt_dialect.scan_return(results)
  scan_op.verify()
  return list(scan_op.result)


@register_lowering(lax.cumsum_p)
def _cumsum_lowering_rule(
    ctx: LoweringRuleContext, x, *, axis: int, reverse: bool
):
  if reverse:
    raise NotImplementedError("Reverse cumsum is not supported.")
  return _associative_scan_lowering(jnp.add, ctx, x, (axis,))[0]


@register_lowering(lax.not_p)
def _not_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  return arith_dialect.xori(x, _full(x.type, ~x_aval.dtype.type(0)))


@dataclasses.dataclass(frozen=True)
class _Extern:
  arg_types: Sequence[str]
  symbol: str
  result_type: str

  def matches(self, avals: Sequence[jax_core.ShapedArray]) -> bool:
    if len(avals) != len(self.arg_types):
      return False
    return all(
        aval.weak_type or aval.dtype.name == arg_type
        for aval, arg_type in zip(avals, self.arg_types)
    )

  def lower(self, ctx: LoweringRuleContext, *args: Sequence[ir.Value]):
    [out_aval] = ctx.avals_out
    result_type = _dtype_to_ir_type(jnp.dtype(self.result_type))
    if out_aval.shape:
      result_type = ir.RankedTensorType.get(out_aval.shape, result_type)
    return tt_dialect.extern_elementwise(
        result_type,
        args,
        libname="",
        libpath="",
        symbol=self.symbol,
        pure=True,
    )


@dataclasses.dataclass(frozen=True)
class _Fallback:
  arg_types: Sequence[str]
  lower: Callable[..., ir.Value]

  matches = _Extern.matches


def _make_dispatch_table(
    name: str, **tables: Sequence[_Extern | _Fallback]
) -> Callable[..., ir.Value]:

  def inner(ctx: LoweringRuleContext, *args: ir.Value) -> ir.Value:
    table = tables[ctx.context.platform]
    h = next((e for e in table if e.matches(ctx.avals_in)), None)
    if h is None:
      arg_aval_dtypes = tuple(aval.dtype.name for aval in ctx.avals_in)
      raise NotImplementedError(
          f"unsupported types for {name}: {arg_aval_dtypes}"
      )

    [out_aval] = ctx.avals_out
    bcast_args = []
    for aval, arg, arg_type in zip(ctx.avals_in, args, h.arg_types):
      bcast_arg = _bcast_to(_ensure_ir_value(arg, aval), out_aval.shape)
      if aval.weak_type and aval.dtype.name != arg_type:
        bcast_arg = _cast(bcast_arg, aval.dtype, jnp.dtype(arg_type))
      bcast_args.append(bcast_arg)
    return h.lower(ctx, *bcast_args)

  return inner


_abs_dispatch_table = _make_dispatch_table(
    "abs",
    cuda=[
        _Extern(["int32"], "__nv_abs", "int32"),
        _Extern(["int64"], "__nv_llabs", "int64"),
        _Extern(["float32"], "__nv_fabsf", "float32"),
        _Extern(["float64"], "__nv_fabs", "float64"),
    ],
    rocm=[
        _Fallback(["int32"], lambda ctx, x: math_dialect.absi(x)),
        _Fallback(["int64"], lambda ctx, x: math_dialect.absi(x)),
        _Fallback(["float32"], lambda ctx, x: math_dialect.absf(x)),
        _Fallback(["float64"], lambda ctx, x: math_dialect.absf(x)),
    ],
)


@register_lowering(lax.abs_p)
def _abs_lowering_rule(ctx: LoweringRuleContext, x):
  try:
    return _abs_dispatch_table(ctx, x)
  except NotImplementedError as e:
    [x_aval] = ctx.avals_in
    if jnp.issubdtype(x_aval, jnp.integer):
      return math_dialect.absi(x)
    elif jnp.issubdtype(x_aval, jnp.floating):
      return math_dialect.absf(x)
    else:
      raise e from None


triton_lowering_rules.update({
    lax.neg_p: lambda ctx, x: _minus(x),
    lax.ceil_p: _make_dispatch_table(
        "ceil",
        cuda=[
            _Extern(["float32"], "__nv_ceilf", "float32"),
            _Extern(["float64"], "__nv_ceil", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_ceil_f32", "float32"),
            _Extern(["float64"], "__ocml_ceil_f64", "float64"),
        ],
    ),
    lax.floor_p: _make_dispatch_table(
        "floor",
        cuda=[
            _Extern(["float32"], "__nv_floorf", "float32"),
            _Extern(["float64"], "__nv_floor", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.floor(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.floor(x)),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_floor_f32", "float32"),
            _Extern(["float64"], "__ocml_floor_f64", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.floor(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.floor(x)),
        ],
    ),
    lax.exp_p: _make_dispatch_table(
        "exp",
        cuda=[
            _Extern(["float32"], "__nv_expf", "float32"),
            _Extern(["float64"], "__nv_exp", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.exp(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.exp(x)),
        ],
        rocm=[
            _Fallback(["float32"], lambda ctx, x: math_dialect.exp(x)),
            _Fallback(["float64"], lambda ctx, x: math_dialect.exp(x)),
            _Fallback(["float16"], lambda ctx, x: math_dialect.exp(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.exp(x)),
        ],
    ),
    lax.exp2_p: _make_dispatch_table(
        "exp2",
        cuda=[
            _Extern(["float32"], "__nv_exp2f", "float32"),
            _Extern(["float64"], "__nv_exp2", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.exp2(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.exp2(x)),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_exp2_f32", "float32"),
            _Extern(["float64"], "__ocml_exp2_f64", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.exp2(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.exp2(x)),
        ],
    ),
    lax.expm1_p: _make_dispatch_table(
        "expm1",
        cuda=[
            _Extern(["float32"], "__nv_expm1f", "float32"),
            _Extern(["float64"], "__nv_expm1", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_expm1_f32", "float32"),
            _Extern(["float64"], "__ocml_expm1_f64", "float64"),
        ],
    ),
    lax.log_p: _make_dispatch_table(
        "log",
        cuda=[
            _Extern(["float32"], "__nv_logf", "float32"),
            _Extern(["float64"], "__nv_log", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.log(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.log(x)),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_log_f32", "float32"),
            _Extern(["float64"], "__ocml_log_f64", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.log(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.log(x)),
        ],
    ),
    lax.log1p_p: _make_dispatch_table(
        "log1p",
        cuda=[
            _Extern(["float32"], "__nv_log1pf", "float32"),
            _Extern(["float64"], "__nv_log1p", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_log1p_f32", "float32"),
            _Extern(["float64"], "__ocml_log1p_f64", "float64"),
        ],
    ),
    lax.sqrt_p: _make_dispatch_table(
        "sqrt",
        cuda=[
            _Extern(["float32"], "__nv_sqrtf", "float32"),
            _Extern(["float64"], "__nv_sqrt", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.sqrt(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.sqrt(x)),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_sqrt_f32", "float32"),
            _Extern(["float64"], "__ocml_sqrt_f64", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.sqrt(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.sqrt(x)),
        ],
    ),
    lax.pow_p: _make_dispatch_table(
        "pow",
        cuda=[
            _Extern(["float32", "int32"], "__nv_powif", "float32"),
            _Extern(["float64", "int32"], "__nv_powi", "float64"),
            _Extern(["float32", "float32"], "__nv_powf", "float32"),
            _Extern(["float64", "float64"], "__nv_pow", "float64"),
        ],
        rocm=[
            _Extern(["float32", "int32"], "__ocml_pown_f32", "float32"),
            _Extern(["float64", "int32"], "__ocml_pown_f64", "float64"),
            _Extern(["float32", "float32"], "__ocml_pow_f32", "float32"),
            _Extern(["float64", "float64"], "__ocml_pow_f64", "float64"),
        ],
    ),
    lax.cbrt_p: _make_dispatch_table(
        "cbrt",
        cuda=[
            _Extern(["float32"], "__nv_cbrtf", "float32"),
            _Extern(["float64"], "__nv_cbrt", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_cbrt_f32", "float32"),
            _Extern(["float64"], "__ocml_cbrt_f64", "float64"),
        ],
    ),
    lax.rsqrt_p: _make_dispatch_table(
        "rsqrt",
        cuda=[
            _Extern(["float32"], "__nv_rsqrtf", "float32"),
            _Extern(["float64"], "__nv_rsqrt", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_rsqrt_f32", "float32"),
            _Extern(["float64"], "__ocml_rsqrt_f64", "float64"),
        ],
    ),
    lax.sin_p: _make_dispatch_table(
        "sin",
        cuda=[
            _Extern(["float32"], "__nv_sinf", "float32"),
            _Extern(["float64"], "__nv_sin", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.sin(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.sin(x)),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_sin_f32", "float32"),
            _Extern(["float64"], "__ocml_sin_f64", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.sin(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.sin(x)),
        ],
    ),
    lax.cos_p: _make_dispatch_table(
        "cos",
        cuda=[
            _Extern(["float32"], "__nv_cosf", "float32"),
            _Extern(["float64"], "__nv_cos", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.cos(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.cos(x)),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_cos_f32", "float32"),
            _Extern(["float64"], "__ocml_cos_f64", "float64"),
            _Fallback(["float16"], lambda ctx, x: math_dialect.cos(x)),
            _Fallback(["bfloat16"], lambda ctx, x: math_dialect.cos(x)),
        ],
    ),
    lax.tan_p: _make_dispatch_table(
        "tan",
        cuda=[
            _Extern(["float32"], "__nv_tanf", "float32"),
            _Extern(["float64"], "__nv_tan", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_tan_f32", "float32"),
            _Extern(["float64"], "__ocml_tan_f64", "float64"),
        ],
    ),
    lax.asin_p: _make_dispatch_table(
        "asin",
        cuda=[
            _Extern(["float32"], "__nv_asinf", "float32"),
            _Extern(["float64"], "__nv_asin", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_asin_f32", "float32"),
            _Extern(["float64"], "__ocml_asin_f64", "float64"),
        ],
    ),
    lax.acos_p: _make_dispatch_table(
        "acos",
        cuda=[
            _Extern(["float32"], "__nv_acosf", "float32"),
            _Extern(["float64"], "__nv_acos", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_acos_f32", "float32"),
            _Extern(["float64"], "__ocml_acos_f64", "float64"),
        ],
    ),
    lax.atan_p: _make_dispatch_table(
        "atan",
        cuda=[
            _Extern(["float32"], "__nv_atanf", "float32"),
            _Extern(["float64"], "__nv_atan", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_atan_f32", "float32"),
            _Extern(["float64"], "__ocml_atan_f64", "float64"),
        ],
    ),
    lax.atan2_p: _make_dispatch_table(
        "atan2",
        cuda=[
            _Extern(["float32", "float32"], "__nv_atan2f", "float32"),
            _Extern(["float64", "float64"], "__nv_atan2", "float64"),
        ],
        rocm=[
            _Extern(["float32", "float32"], "__ocml_atan2_f32", "float32"),
            _Extern(["float64", "float64"], "__ocml_atan2_f64", "float64"),
        ],
    ),
    lax.sinh_p: _make_dispatch_table(
        "sinh",
        cuda=[
            _Extern(["float32"], "__nv_sinhf", "float32"),
            _Extern(["float64"], "__nv_sinh", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_sinh_f32", "float32"),
            _Extern(["float64"], "__ocml_sinh_f64", "float64"),
        ],
    ),
    lax.cosh_p: _make_dispatch_table(
        "cosh",
        cuda=[
            _Extern(["float32"], "__nv_coshf", "float32"),
            _Extern(["float64"], "__nv_cosh", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_cosh_f32", "float32"),
            _Extern(["float64"], "__ocml_cosh_f64", "float64"),
        ],
    ),
    lax.tanh_p: _make_dispatch_table(
        "tanh",
        cuda=[
            _Extern(["float32"], "__nv_tanhf", "float32"),
            _Extern(["float64"], "__nv_tanh", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_tanh_f32", "float32"),
            _Extern(["float64"], "__ocml_tanh_f64", "float64"),
        ],
    ),
    lax.asinh_p: _make_dispatch_table(
        "asinh",
        cuda=[
            _Extern(["float32"], "__nv_asinhf", "float32"),
            _Extern(["float64"], "__nv_asinh", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_asinh_f32", "float32"),
            _Extern(["float64"], "__ocml_asinh_f64", "float64"),
        ],
    ),
    lax.acosh_p: _make_dispatch_table(
        "acosh",
        cuda=[
            _Extern(["float32"], "__nv_acoshf", "float32"),
            _Extern(["float64"], "__nv_acosh", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_acosh_f32", "float32"),
            _Extern(["float64"], "__ocml_acosh_f64", "float64"),
        ],
    ),
    lax.atanh_p: _make_dispatch_table(
        "atanh",
        cuda=[
            _Extern(["float32"], "__nv_atanhf", "float32"),
            _Extern(["float64"], "__nv_atanh", "float64"),
        ],
        rocm=[
            _Extern(["float32"], "__ocml_atanh_f32", "float32"),
            _Extern(["float64"], "__ocml_atanh_f64", "float64"),
        ],
    ),
    lax.population_count_p: _make_dispatch_table(
        "population_count",
        cuda=[
            _Extern(["int32"], "__nv_popc", "int32"),
            _Extern(["int64"], "__nv_popcll", "int32"),
        ],
        rocm=[
            _Fallback(["int32"], lambda ctx, x: math_dialect.ctpop(x)),
            _Fallback(["int64"], lambda ctx, x: math_dialect.ctpop(x)),
        ],
    ),
    lax.clz_p: _make_dispatch_table(
        "clz",
        cuda=[
            _Extern(["int32"], "__nv_clz", "int32"),
            _Extern(["int64"], "__nv_clzll", "int32"),
        ],
        rocm=[
            _Fallback(["int32"], lambda ctx, x: math_dialect.ctlz(x)),
            _Fallback(["int64"], lambda ctx, x: math_dialect.ctlz(x)),
        ],
    ),
    lax.nextafter_p: _make_dispatch_table(
        "nextafter",
        cuda=[
            _Extern(["float32", "float32"], "__nv_nextafterf", "float32"),
            _Extern(["float64", "float64"], "__nv_nextafter", "float64"),
        ],
        rocm=[
            _Extern(["float32", "float32"], "__ocml_nextafter_f32", "float32"),
            _Extern(["float64", "float64"], "__ocml_nextafter_f64", "float64"),
        ],
    ),
})


def _minus(x: ir.Value) -> ir.Value:
  if tt_dialect.PointerType.isinstance(_element_type(x.type)):
    raise NotImplementedError(f"unsupported type: {x.type}")
  return _sub(_full(x.type, 0), x)


def _add(x: ir.Value, y: ir.Value):
  x_element_type = _element_type(x.type)
  y_element_type = _element_type(y.type)
  if tt_dialect.PointerType.isinstance(y_element_type):
    assert not tt_dialect.PointerType.isinstance(x_element_type)
    x, y = y, x
    x_element_type, y_element_type = y_element_type, x_element_type

  if tt_dialect.PointerType.isinstance(x_element_type):
    return tt_dialect.addptr(x.type, x, y)

  assert x.type == y.type, (str(x.type), str(y.type))
  if isinstance(x_element_type, ir.IntegerType):
    return arith_dialect.addi(x, y)
  elif isinstance(x_element_type, ir.FloatType):
    return arith_dialect.addf(x, y)
  else:
    raise NotImplementedError(f"unsupported dtypes: {x.type} and {y.type}")


def _sub(x: ir.Value, y: ir.Value) -> ir.Value:
  x_element_type = _element_type(x.type)
  y_element_type = _element_type(y.type)
  if tt_dialect.PointerType.isinstance(x_element_type):
    return tt_dialect.addptr(x.type, x, _minus(y))
  elif not tt_dialect.PointerType.isinstance(y_element_type):
    assert x.type == y.type, (str(x.type), str(y.type))
    if isinstance(x_element_type, ir.IntegerType):
      return arith_dialect.subi(x, y)
    elif isinstance(x_element_type, ir.FloatType):
      return arith_dialect.subf(x, y)
  raise NotImplementedError(f"unsupported dtype: {y.type}")


def _mul(x: ir.Value, y: ir.Value) -> ir.Value:
  assert x.type == y.type, (str(x.type), str(y.type))
  x_element_type = _element_type(x.type)
  if isinstance(x_element_type, ir.IntegerType):
    return arith_dialect.muli(x, y)
  elif isinstance(x_element_type, ir.FloatType):
    return arith_dialect.mulf(x, y)
  raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")


def _floordiv(x: ir.Value, y: ir.Value, *, signed: bool) -> ir.Value:
  assert x.type == y.type, (str(x.type), str(y.type))
  x_element_type = _element_type(x.type)
  if isinstance(x_element_type, (ir.F32Type, ir.F64Type)):
    return arith_dialect.divf(x, y)
  if not isinstance(x_element_type, ir.IntegerType):
    raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")
  if signed:
    return arith_dialect.divsi(x, y)
  else:
    return arith_dialect.divui(x, y)


def _truediv(x: ir.Value, y: ir.Value, *, signed: bool) -> ir.Value:
  assert x.type == y.type, (str(x.type), str(y.type))
  x_element_type = _element_type(x.type)
  if isinstance(x_element_type, ir.IntegerType):
    x_element_type = ir.F32Type.get()
    x = _int_float_cast(x, x_element_type, signed=signed)
    y = _int_float_cast(y, x_element_type, signed=signed)
  if isinstance(x_element_type, (ir.F32Type, ir.F64Type)):
    return arith_dialect.divf(x, y)
  raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")


def _mod(x: ir.Value, y: ir.Value, *, signed: bool) -> ir.Value:
  assert x.type == y.type, (str(x.type), str(y.type))
  x_element_type = _element_type(x.type)
  if isinstance(x_element_type, ir.FloatType):
    return arith_dialect.remf(x, y)
  if not isinstance(x_element_type, ir.IntegerType):
    raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")
  if signed:
    return arith_dialect.remsi(x, y)
  else:
    return arith_dialect.remui(x, y)


def _cmp(
    x: ir.Value,
    y: ir.Value,
    si_pred: arith_dialect.CmpIPredicate,
    ui_pred: arith_dialect.CmpIPredicate,
    f_pred: arith_dialect.CmpFPredicate,
    *,
    signed: bool,
) -> ir.Value:
  assert x.type == y.type, (str(x.type), str(y.type))
  x_element_type = _element_type(x.type)
  if isinstance(x_element_type, ir.IntegerType):
    return arith_dialect.cmpi(si_pred if signed else ui_pred, x, y)
  elif isinstance(x_element_type, ir.FloatType):
    return arith_dialect.cmpf(f_pred, x, y)
  else:
    raise NotImplementedError(f"unsupported types: {x.type} and {y.type}")


_equal = functools.partial(
    _cmp,
    si_pred=arith_dialect.CmpIPredicate.eq,
    ui_pred=arith_dialect.CmpIPredicate.eq,
    f_pred=arith_dialect.CmpFPredicate.OEQ,
)
_not_equal = functools.partial(
    _cmp,
    si_pred=arith_dialect.CmpIPredicate.ne,
    ui_pred=arith_dialect.CmpIPredicate.ne,
    f_pred=arith_dialect.CmpFPredicate.UNE,
)
_less_than = functools.partial(
    _cmp,
    si_pred=arith_dialect.CmpIPredicate.slt,
    ui_pred=arith_dialect.CmpIPredicate.ult,
    f_pred=arith_dialect.CmpFPredicate.OLT,
)
_less_equal = functools.partial(
    _cmp,
    si_pred=arith_dialect.CmpIPredicate.sle,
    ui_pred=arith_dialect.CmpIPredicate.ule,
    f_pred=arith_dialect.CmpFPredicate.OLE,
)
_greater_than = functools.partial(
    _cmp,
    si_pred=arith_dialect.CmpIPredicate.sgt,
    ui_pred=arith_dialect.CmpIPredicate.ugt,
    f_pred=arith_dialect.CmpFPredicate.OGT,
)
_greater_equal = functools.partial(
    _cmp,
    si_pred=arith_dialect.CmpIPredicate.sge,
    ui_pred=arith_dialect.CmpIPredicate.uge,
    f_pred=arith_dialect.CmpFPredicate.OGE,
)


_JAX_TO_TRITON_BINARY = {
    lax.add_p: _add,
    lax.sub_p: _sub,
    lax.mul_p: _mul,
    lax.and_p: arith_dialect.andi,
    lax.or_p: arith_dialect.ori,
    lax.xor_p: arith_dialect.xori,
    lax.shift_left_p: arith_dialect.shli,
    lax.shift_right_arithmetic_p: arith_dialect.shrsi,
    lax.shift_right_logical_p: arith_dialect.shrui,
    ad_util.add_any_p: _add,
}

for prim, fn in _JAX_TO_TRITON_BINARY.items():

  def signless_rule(ctx: LoweringRuleContext, x, y, fn=fn):
    x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
    return fn(x, y)

  triton_lowering_rules[prim] = signless_rule


_JAX_TO_TRITON_SIGNED_BINARY = {
    lax.rem_p: _mod,
    lax.eq_p: _equal,
    lax.ne_p: _not_equal,
    lax.gt_p: _greater_than,
    lax.ge_p: _greater_equal,
    lax.lt_p: _less_than,
    lax.le_p: _less_equal,
}

for prim, fn in _JAX_TO_TRITON_SIGNED_BINARY.items():

  def signed_rule(ctx: LoweringRuleContext, x, y, fn=fn):
    x_aval, _ = ctx.avals_in
    x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
    return fn(x, y, signed=jnp.issubdtype(x_aval.dtype, jnp.signedinteger))

  triton_lowering_rules[prim] = signed_rule


@register_lowering(primitives.debug_print_p)
def debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args: ir.Value,
    fmt: str,
    has_placeholders: bool,
):
  if has_placeholders:
    raise ValueError(
        "pl.debug_print() does not support placeholders when lowering to Triton"
    )

  tt_dialect.print_(f" {fmt} ", hex=False, args=args)
  return ()


def _set_attr(v: ir.Value, name: str, attr: ir.Attribute) -> None:
  if not ir.BlockArgument.isinstance(v):
    v.owner.attributes[name] = attr
    return

  arg = ir.BlockArgument(v)
  name += f"_arg{arg.arg_number}"
  owner = arg.owner
  is_entry = owner.region.blocks[0] == owner
  if not is_entry:
    return
  if (op := owner.owner.operation) and not isinstance(op, tt_dialect.FuncOp):
    op.attributes[name] = attr


@register_lowering(primitives.multiple_of_p)
def _multiple_of_rule(ctx: LoweringRuleContext, x, values: Sequence[int]):
  [x_aval] = ctx.avals_in
  assert max(1, len(x_aval.shape)) == len(values)
  _set_attr(
      x,
      "tt.divisibility",
      ir.DenseIntElementsAttr.get(np.asarray(values, dtype=np.int32)),
  )
  return x


@register_lowering(primitives.max_contiguous_p)
def _max_contiguous_rule(ctx: LoweringRuleContext, x, values: Sequence[int]):
  [x_aval] = ctx.avals_in
  assert len(x_aval.shape) == len(values)
  _set_attr(
      x,
      "tt.contiguity",
      ir.DenseIntElementsAttr.get(np.asarray(values, dtype=np.int32)),
  )
  return x


@register_lowering(sp.broadcast_to_p)
def _broadcast_to_rule(ctx: LoweringRuleContext, x, shape: Sequence[int]):
  (x_aval,) = ctx.avals_in
  return _bcast_to(_ensure_ir_value(x, x_aval), shape)


@register_lowering(lax.integer_pow_p)
def _integer_pow_rule(ctx: LoweringRuleContext, x, *, y: int):
  if y == 0:
    return _full(x.type, 1)

  is_reciprocal = y < 0
  if is_reciprocal:
    y = -y

  acc = None
  while y > 0:
    y, mod = divmod(y, 2)
    if mod:
      acc = x if acc is None else _mul(acc, x)
    if y > 0:
      x = _mul(x, x)
  assert acc is not None

  [x_aval] = ctx.avals_in
  [out_aval] = ctx.avals_out
  acc = _cast(acc, x_aval.dtype, out_aval.dtype)
  if is_reciprocal:
    signed = jnp.issubdtype(out_aval.dtype, jnp.signedinteger)
    return  _truediv(_full(acc.type, 1), acc, signed=signed)
  else:
    return acc


def lower_fun(
    fun: Callable[..., Any], *, multiple_results: bool
) -> Callable[..., Any]:
  fn = fun if multiple_results else lambda *args, **kw: (fun(*args, **kw),)

  def f_lowered(ctx: LoweringRuleContext, *args, **params):
    wrapped_fun = lu.wrap_init(fn, params)
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
    jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
    out = _closed_call_lowering_rule(ctx, *args, call_jaxpr=jaxpr)
    return out if multiple_results else out[0]

  return f_lowered


_JAX_FN_MAPPING = {
    lax.clamp_p: lambda min, a, max: jnp.minimum(jnp.maximum(min, a), max),
    lax.logistic_p: lambda a: 1 / (1 + jnp.exp(-a)),
}

for prim, fn in _JAX_FN_MAPPING.items():
  triton_lowering_rules[prim] = lower_fun(fn, multiple_results=False)


@register_lowering(lax.min_p)
def _min_lowering_rule(ctx: LoweringRuleContext, x, y):
  # TODO(slebedev): Consider allowing customizing nan behavior.
  x_aval, y_aval = ctx.avals_in
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  if jnp.issubdtype(x_aval.dtype, jnp.floating):
    # TODO(slebedev): Triton promotes bfloat16 to float32 and back here.
    return arith_dialect.minnumf(x, y)
  if not jnp.issubdtype(x_aval.dtype, jnp.integer):
    raise NotImplementedError(
        f"unsupported dtypes: {x_aval.dtype} and {y_aval.dtype}"
    )
  if jnp.issubdtype(x_aval.dtype, jnp.signedinteger):
    return arith_dialect.minsi(x, y)
  else:
    return arith_dialect.minui(x, y)


@register_lowering(lax.max_p)
def _max_lowering_rule(ctx: LoweringRuleContext, x, y):
  # TODO(slebedev): Consider allowing customizing nan behavior.
  x_aval, y_aval = ctx.avals_in
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  if jnp.issubdtype(x_aval.dtype, jnp.floating):
    # TODO(slebedev): Triton promotes bfloat16 to float32 and back here.
    return arith_dialect.maxnumf(x, y)
  if not jnp.issubdtype(x_aval.dtype, jnp.integer):
    raise NotImplementedError(
        f"unsupported dtypes: {x_aval.dtype} and {y_aval.dtype}"
    )
  if jnp.issubdtype(x_aval.dtype, jnp.signedinteger):
    return arith_dialect.maxsi(x, y)
  else:
    return arith_dialect.maxui(x, y)


@register_lowering(lax.div_p)
def _div_lowering_rule(ctx: LoweringRuleContext, x, y):
  x_aval, y_aval = ctx.avals_in
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  signed = jnp.issubdtype(x_aval.dtype, jnp.signedinteger) or jnp.issubdtype(
      y_aval.dtype, jnp.signedinteger
  )
  if jnp.issubdtype(x_aval.dtype, np.floating) or jnp.issubdtype(
      y_aval.dtype, np.floating
  ):
    return _truediv(x, y, signed=signed)
  return _floordiv(x, y, signed=signed)


@register_lowering(lax.sign_p)
def _sign_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  signed = jnp.issubdtype(x_aval.dtype, jnp.signedinteger)
  zero = _full(x.type, 0)
  return _sub(
      _cast(_greater_than(x, zero, signed=signed), jnp.bool_, x_aval.dtype),
      _cast(_less_than(x, zero, signed=signed), jnp.bool_, x_aval.dtype),
  )


@register_lowering(lax.iota_p)
def _iota_lowering_rule(ctx: LoweringRuleContext, *, dtype, shape, dimension):
  iota = _make_range(0, shape[dimension])
  iota = _cast(iota, jnp.int32, dtype)
  for i in range(len(shape)):
    if i != dimension:
      iota = _expand_dims(iota, i)
  return _bcast_to(iota, shape)


def _element_type(t: ir.Type) -> ir.Type:
  if ir.RankedTensorType.isinstance(t):
    return ir.RankedTensorType(t).element_type
  else:
    return t


def _make_range(start: int, end: int) -> ir.Value:
  if end <= start:
    raise ValueError(
        f"end must be greater than start, but got: {end} <= {start}"
    )
  if max(start, end) >= 2**32:
    raise ValueError("start and end must fit in int32")
  return tt_dialect.make_range(
      ir.RankedTensorType.get([end - start], ir.IntegerType.get_signless(32)),
      start,
      end,
  )


def _full(t: ir.Type, v: object) -> ir.Type:
  element_type = _element_type(t)
  if isinstance(element_type, ir.IntegerType):
    result = arith_dialect.constant(element_type, int(v))
  elif isinstance(element_type, ir.FloatType):
    result = arith_dialect.constant(element_type, float(v))
  else:
    raise NotImplementedError

  if ir.RankedTensorType.isinstance(t):
    return tt_dialect.splat(t, result)
  else:
    return result


def _splat(x: ir.value, shape: Sequence[int]) -> ir.Value:
  if ir.RankedTensorType.isinstance(x.type):
    raise TypeError("cannot splat a tensor")
  if not shape:
    return x
  return tt_dialect.splat(ir.RankedTensorType.get(shape, x.type), x)


def _expand_dims(x: ir.Value, axis: int) -> ir.Value:
  if not ir.RankedTensorType.isinstance(x.type):
    shape = list(ir.RankedTensorType(x.type).shape)
    shape.insert(axis, 1)
    return _splat(x, shape)
  return tt_dialect.expand_dims(x, axis)


def _float_float_cast(src: ir.Value, dst_type: ir.Type) -> ir.Value:
  src_element_type = ir.FloatType(_element_type(src.type))
  dst_element_type = ir.FloatType(_element_type(dst_type))
  if src_element_type.width == 8 or dst_element_type.width == 8:
    return tt_dialect.fp_to_fp(
        dst_type,
        src,
        rounding=tt_dialect.RoundingMode.RTNE,
    )
  if src_element_type.width > dst_element_type.width:
    return arith_dialect.truncf(dst_type, src)
  elif src_element_type.width < dst_element_type.width:
    return arith_dialect.extf(dst_type, src)
  else:
    raise NotImplementedError


def _int_int_cast(src: ir.Value, dst_type: ir.Type, signed: bool) -> ir.Value:
  src_element_type = ir.IntegerType(_element_type(src.type))
  dst_element_type = ir.IntegerType(_element_type(dst_type))
  assert src_element_type != dst_element_type
  if dst_element_type.width == 1:
    return _not_equal(src, _full(src.type, 0), signed=signed)

  if src_element_type.width == dst_element_type.width:
    return arith_dialect.bitcast(dst_type, src)
  elif src_element_type.width > dst_element_type.width:
    return arith_dialect.trunci(dst_type, src)
  elif signed and src_element_type.width != 1:
    return arith_dialect.extsi(dst_type, src)
  else:
    return arith_dialect.extui(dst_type, src)


def _float_int_cast(
    src: ir.Value, dst_type: ir.Type, *, signed: bool
) -> ir.Value:
  src_element_type = _element_type(src.type)
  if not isinstance(src_element_type, (ir.BF16Type, ir.F16Type, ir.F32Type, ir.F64Type)):
    raise NotImplementedError(f"cannot cast {src} tp {dst_type}")
  dst_element_type = ir.IntegerType(_element_type(dst_type))
  if dst_element_type.width == 1:
    return _not_equal(src, _full(src.type, 0), signed=signed)
  elif signed:
    return arith_dialect.fptosi(dst_type, src)
  else:
    return arith_dialect.fptoui(dst_type, src)


def _int_float_cast(
    src: ir.Value, dst_type: ir.Type, *, signed: bool
) -> ir.Value:
  src_element_type = ir.IntegerType(_element_type(src.type))
  dst_element_type = _element_type(dst_type)
  if not isinstance(
      dst_element_type, (ir.BF16Type, ir.F16Type, ir.F32Type, ir.F64Type)
  ):
    raise NotImplementedError(f"cannot cast {src} tp {dst_type}")
  if src_element_type.width == 1 or not signed:
    return arith_dialect.uitofp(dst_type, src)
  else:
    return arith_dialect.sitofp(dst_type, src)


def _cast(
    src: ir.Value,
    src_type: jax.typing.DTypeLike,
    dst_type: jax.typing.DTypeLike,
) -> ir.Value:
  return _ir_cast(
      src,
      _dtype_to_ir_type(dst_type),
      signed=jnp.issubdtype(src_type, jnp.signedinteger),
  )


def _ir_cast(src: ir.Value, dst_type: ir.Type, *, signed: bool) -> ir.Value:
  if ir.RankedTensorType.isinstance(
      src.type
  ) and not ir.RankedTensorType.isinstance(dst_type):
    src_type = ir.RankedTensorType(src.type)
    dst_type = ir.RankedTensorType.get(
        src_type.shape,
        dst_type,
        src_type.encoding,
    )
  if src.type == dst_type:
    return src

  src_element_type = _element_type(src.type)
  dst_element_type = _element_type(dst_type)
  if isinstance(src_element_type, ir.Float8E4M3FNUZType) or isinstance(
      dst_element_type, ir.Float8E4M3FNUZType
  ):
    # TODO(slebedev): Check the CUDA version and raise conditionally.
    raise NotImplementedError("cannot cast from or to float8_e4m3fnuz")

  if isinstance(src_element_type, (ir.F16Type, ir.BF16Type)) and not isinstance(
      dst_element_type, ir.F32Type
  ):
    return _ir_cast(
        _ir_cast(src, ir.F32Type.get(), signed=False), dst_type, signed=False
    )

  if isinstance(src_element_type, ir.FloatType) and isinstance(
      dst_element_type, ir.FloatType
  ):
    return _float_float_cast(src, dst_type)

  if isinstance(src_element_type, ir.IntegerType) and isinstance(
      dst_element_type, ir.IntegerType
  ):
    return _int_int_cast(src, dst_type, signed=signed)

  if isinstance(src_element_type, ir.FloatType) and isinstance(
      dst_element_type, ir.IntegerType
  ):
    return _float_int_cast(src, dst_type, signed=signed)
  if isinstance(src_element_type, ir.IntegerType) and isinstance(
      dst_element_type, ir.FloatType
  ):
    return _int_float_cast(src, dst_type, signed=signed)

  if tt_dialect.PointerType.isinstance(src_element_type) and isinstance(
      dst_element_type, ir.IntegerType
  ):
    if dst_element_type.width == 64:
      return tt_dialect.ptr_to_int(dst_type, src)
    elif dst_element_type.width == 1:
      x = _ir_cast(src, ir.IntegerType.get_signless(64), signed=signed)
      zero = _full(x.type, 0)
      return _ir_cast(_not_equal(x, zero, signed=signed), dst_type, signed=signed)
  if isinstance(
      src_element_type, ir.IntegerType
  ) and tt_dialect.PointerType.isinstance(dst_element_type):
    return tt_dialect.int_to_ptr(dst_type, src)
  if tt_dialect.PointerType.isinstance(
      src_element_type
  ) and tt_dialect.PointerType.isinstance(dst_element_type):
    return tt_dialect.bitcast(dst_type, src)

  raise NotImplementedError(f"cannot cast {src} to {dst_type}")


@register_lowering(lax.convert_element_type_p)
def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  [x_aval] = ctx.avals_in
  x = _ensure_ir_value(x, x_aval)
  if new_dtype == x_aval.dtype:
    return x
  return _cast(x, x_aval.dtype, new_dtype)


@register_lowering(lax.select_n_p)
def select_n_lowering_rule(ctx: LoweringRuleContext, pred, x, y):
  pred_aval, a_aval, b_aval = ctx.avals_in
  [out_aval] = ctx.avals_out
  pred, x = _bcast(pred, x, pred_aval, a_aval, out_aval)
  pred, y = _bcast(pred, y, pred_aval, b_aval, out_aval)
  return arith_dialect.select(pred, y, x)


@register_lowering(lax.broadcast_in_dim_p)
def _broadcast_in_dim_lowering_rule(
    ctx: LoweringRuleContext, x, *, broadcast_dimensions, shape
):
  x = _ensure_ir_value(x, *ctx.avals_in)
  if not ir.RankedTensorType.isinstance(x.type):
    return _bcast_to(x, shape)
  expand_dims = [i for i in range(len(shape)) if i not in broadcast_dimensions]
  for dim in expand_dims:
    x = _expand_dims(x, dim)
  return _bcast_to(x, shape)


@register_lowering(lax.squeeze_p)
def _squeeze_lowering_rule(ctx: LoweringRuleContext, a, *, dimensions):
  del dimensions
  return _reshape_lowering_rule(ctx, a, new_sizes=None, dimensions=None)


def _reshape(x: ir.Value, shape: Sequence[int]) -> ir.Value:
  if not shape:
    raise ValueError("cannot reshape to an empty shape")
  ty = ir.RankedTensorType(x.type)
  return tt_dialect.reshape(
      ir.RankedTensorType.get(shape, ty.element_type, ty.encoding),
      x,
      allow_reorder=False,
  )


@register_lowering(lax.reshape_p)
def _reshape_lowering_rule(
    ctx: LoweringRuleContext, a, *, new_sizes, dimensions
):
  del new_sizes  # Unused.
  if dimensions is not None:
    return ValueError("`dimensions` is not supported.")

  a = _ensure_ir_value(a, *ctx.avals_in)
  [out_aval] = ctx.avals_out
  if not ir.RankedTensorType.isinstance(a.type):
    assert all(dim_size == 1 for dim_size in out_aval.shape)
    return _splat(a, out_aval.shape)

  # TODO(slebedev): Check that the following comment still applies.
  # Expand-dims or reduce-sum to handle singleton dims as `tl.reshape` is not
  # currently implemented.
  dst_shape = [*out_aval.shape]
  i = 0
  while (
      ir.RankedTensorType.isinstance(a.type)
      and (a_shape := ir.RankedTensorType(a.type).shape) != dst_shape
  ):
    dim_size = a_shape[i] if i < len(a_shape) else None
    dst_dim_size = dst_shape[i] if i < len(dst_shape) else None
    if dim_size == dst_dim_size:
      i += 1
    elif dst_dim_size == 1:
      a = _expand_dims(a, axis=i)
      i += 1
    elif dim_size == 1:
      in_shape = a_shape
      out_shape = tuple(d for di, d in enumerate(a_shape) if di != i)
      reduce_ctx = ctx.replace(
          avals_in=[ctx.avals_in[0].update(shape=in_shape)],
          avals_out=[ctx.avals_in[0].update(shape=out_shape)],
      )
      a = _reduce_lowering(jnp.add, reduce_ctx, a, axes=(i,))
    else:  # We expect this to fail.
      return _reshape(a, dst_shape)

  return a


def _compute_pointers_from_indices(
    root_ptr: ir.Value,
    block_info: BlockInfo | None,
    nd_indexer: NDIndexer,
    array_shape: tuple[int, ...],
) -> ir.Value:
  if block_info is None:
    full_shape = array_shape
    num_mapped_dims = 0
    block_shape = array_shape
  else:
    full_shape = block_info.full_shape_dtype.shape
    num_mapped_dims = sum(
        b is pallas_core.mapped for b in block_info.block_shape
    )
    block_shape = block_info.block_shape
  strides = pallas_utils.strides_from_shape(full_shape)
  indexer_shape = nd_indexer.get_indexer_shape()
  int_indexer_shape = nd_indexer.int_indexer_shape
  indices = nd_indexer.indices
  other_shape = indexer_shape[len(int_indexer_shape) :]
  bcast_indices = []
  other_shape_idx = 0
  if block_info is None:
    start_index_offsets = [None] * len(indices)
  else:
    start_index_offsets = block_info.start_indices
  assert len(indices) + num_mapped_dims == len(full_shape)
  assert len(start_index_offsets) == len(full_shape)
  indexer_iter = iter(indices)
  for dim_stride, dim_block_size, start_offset in zip(
      strides, block_shape, start_index_offsets
  ):
    if dim_block_size is pallas_core.mapped:
      index = _i32_constant(0)
    else:
      index = next(indexer_iter)
    if isinstance(index, primitives.Slice):
      if index.is_dynamic_start:
        # Compute the offset as start + range(0, size).
        ptr_dim_offset = _add(
            _bcast_to(index.start, [index.size]),
            _ir_cast(_make_range(0, index.size), index.start.type, signed=False),
        )
      elif index.stride > 1:
        # Compute the offset as start + range(0, size) * stride.
        iota = _make_range(0, index.size)
        ptr_dim_offset = _add(
            _bcast_to(_i32_constant(index.start), [index.size]),
            _mul(iota, _full(iota.type, index.stride)),
        )
      else:
        ptr_dim_offset = _make_range(index.start, index.start + index.size)

      # We need to add broadcastable dimensions for the advanced int indexing
      # and for previous slices
      num_left_expand_dims = len(int_indexer_shape) + other_shape_idx
      num_right_expand_dims = len(other_shape) - other_shape_idx - 1
      other_shape_idx += 1
    elif isinstance(index, slice):
      if index != slice(None):
        raise NotImplementedError("Only `slice(None)` allowed.")
      ptr_dim_offset = _make_range(0, dim_block_size)
      num_left_expand_dims = len(int_indexer_shape) + other_shape_idx
      num_right_expand_dims = len(other_shape) - other_shape_idx - 1
      other_shape_idx += 1
    else:
      # indexer is either a *scalar* or an array of size `int_indexer_shape`
      ptr_dim_offset = _ensure_ir_value(
          index, jax_core.ShapedArray((), jnp.int32)
      )
      num_left_expand_dims = 0
      num_right_expand_dims = len(other_shape)
      if not ir.RankedTensorType.isinstance(ptr_dim_offset.type):
        num_left_expand_dims = max(len(indexer_shape) - 1, 0)
      else:
        num_right_expand_dims = len(other_shape)

    if indexer_shape and not ir.RankedTensorType.isinstance(ptr_dim_offset.type):
      ptr_dim_offset = _splat(ptr_dim_offset, [1] * len(indexer_shape))
    else:
      for _ in range(num_left_expand_dims):
        ptr_dim_offset = _expand_dims(ptr_dim_offset, 0)
      for _ in range(num_right_expand_dims):
        ndim = len(getattr(ptr_dim_offset.type, "shape", []))
        ptr_dim_offset = _expand_dims(ptr_dim_offset, ndim)

    ptr_dim_offset = _bcast_to(ptr_dim_offset, indexer_shape)
    index_type = ir.IntegerType(_element_type(ptr_dim_offset.type))
    if start_offset is not None:
      start_offset = _ir_cast(start_offset, index_type, signed=False)
      ptr_dim_offset = _add(
          ptr_dim_offset, _bcast_to(start_offset, indexer_shape)
      )

    if index_type.width == 32:
      stride_size = _i32_constant(dim_stride)
    else:
      stride_size = _i64_constant(dim_stride)
    stride_size = _splat(stride_size, indexer_shape)
    bcast_indices.append(_mul(ptr_dim_offset, stride_size))

  return functools.reduce(
      _add, bcast_indices, _bcast_to(root_ptr, indexer_shape)
  )


@register_lowering(sp.get_p)
def _get_lowering_rule(ctx: LoweringRuleContext, ptr, *idx, tree):
  indexers = tree_util.tree_unflatten(tree, idx)
  if not tt_dialect.PointerType.isinstance(ptr.type):
    assert len(indexers) == 0
    return ptr
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  indexer = indexers[0]
  args_flat, args_tree = tree_util.tree_flatten((ptr, (indexer,), None, None))
  return _masked_load_lowering_rule(
      ctx,
      *args_flat,
      args_tree=args_tree,
      eviction_policy=None,
      cache_modifier=None,
      is_volatile=False,
  )


_STR_TO_EVICTION_POLICY = {str(e): e for e in tt_dialect.EvictionPolicy}
_STR_TO_CACHE_MODIFIER = {str(c): c for c in tt_dialect.CacheModifier}


def _load(
    ptr: ir.Value,
    mask: ir.Value | None = None,
    other: ir.Value | None = None,
    *,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
    is_volatile: bool = False,
) -> ir.Value:
  if cache_modifier is None:
    cache_modifier = tt_dialect.CacheModifier.NONE
  elif cache_modifier == ".ca" or cache_modifier == ".cg":
    cache_modifier = _STR_TO_CACHE_MODIFIER[cache_modifier]
  else:
    raise ValueError(f"unsupported cache modifier: {cache_modifier}")
  if eviction_policy is None:
    eviction_policy = tt_dialect.EvictionPolicy.NORMAL
  else:
    try:
      eviction_policy = _STR_TO_EVICTION_POLICY[eviction_policy]
    except KeyError:
      raise ValueError(
          f"unsupported eviction policy: {eviction_policy}"
      ) from None

  if tt_dialect.PointerType.isinstance(ptr.type):
    ptr_type = tt_dialect.PointerType(ptr.type)
    if ir.RankedTensorType.isinstance(ptr_type.pointee_type):
      raise NotImplementedError("loading from a block pointer is not supported")

  ptr_type = _element_type(ptr.type)
  if not tt_dialect.PointerType.isinstance(ptr_type):
    raise ValueError(f"unsupported pointer type: {ptr_type}")
  ptr_type = tt_dialect.PointerType(ptr_type)
  if other is not None and mask is None:
    raise ValueError("other requires mask to be provided")
  if not ir.RankedTensorType.isinstance(ptr.type):
    if other is not None and ir.RankedTensorType.isinstance(other.type):
      raise ValueError("other cannot be a block if pointer is not a block")
    if mask is not None and ir.RankedTensorType.isinstance(mask.type):
      raise ValueError("mask cannot be a block if pointer is not a block")

  pointee_type = ptr_type.pointee_type
  is_int1 = isinstance(pointee_type, ir.IntegerType) and pointee_type.width == 1
  if is_int1:
    pointee_type = ir.IntegerType.get_signless(8)
    ptr = _ir_cast(
        ptr,
        tt_dialect.PointerType.get(pointee_type, ptr_type.address_space),
        signed=False,
    )

  if other is not None:
    other = _ir_cast(other, pointee_type, signed=False)

  result = tt_dialect.load(
      ptr,
      mask=mask,
      other=other,
      cache=cache_modifier,
      evict=eviction_policy,
      is_volatile=is_volatile,
  )
  return (
      result
      if not is_int1
      else _ir_cast(result, ir.IntegerType.get_signless(1), signed=False)
  )


@register_lowering(primitives.load_p)
def _masked_load_lowering_rule(
    ctx: LoweringRuleContext,
    *args_flat,
    args_tree,
    eviction_policy,
    cache_modifier,
    is_volatile,
):
  ptr, indexers, mask, other = args_tree.unflatten(args_flat)
  *_, mask_aval, other_aval = args_tree.unflatten(ctx.avals_in)
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  idx = indexers[0]
  if not tt_dialect.PointerType.isinstance(ptr.type):
    assert len(ctx.avals_in) == 1
    return ptr
  ptr = _compute_pointers_from_indices(
      ptr, ctx.block_infos[0], idx, ctx.avals_in[0].shape
  )
  if mask is not None:
    mask = _bcast_to(_ensure_ir_value(mask, mask_aval), idx.get_indexer_shape())
  if other is not None:
    other = _bcast_to(
        _ensure_ir_value(other, other_aval), idx.get_indexer_shape()
    )
  return _load(
      ptr,
      mask=mask,
      other=other,
      cache_modifier=cache_modifier,
      is_volatile=is_volatile,
      eviction_policy=eviction_policy,
  )


@register_lowering(sp.swap_p)
def _swap_lowering_rule(ctx: LoweringRuleContext, ptr, value, *idx, tree):
  indexers = tree_util.tree_unflatten(tree, idx)
  if not tt_dialect.PointerType.isinstance(ptr.type):
    assert len(indexers) == 0
    return ptr
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  indexer = indexers[0]
  args_flat, args_tree = tree_util.tree_flatten((ptr, (indexer,), value, None))
  return _masked_swap_lowering_rule(
      ctx, *args_flat, args_tree=args_tree, eviction_policy=None
  )


def _store(
    ptr: ir.Value,
    value: ir.Value,
    mask: ir.Value | None = None,
    *,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
) -> ir.Value:
  if cache_modifier is None:
    cache_modifier = tt_dialect.CacheModifier.NONE
  elif cache_modifier != ".ca":
    cache_modifier = _STR_TO_CACHE_MODIFIER[cache_modifier]
  else:
    raise ValueError(f"unsupported cache modifier: {cache_modifier}")
  if eviction_policy is None:
    eviction_policy = tt_dialect.EvictionPolicy.NORMAL
  else:
    try:
      eviction_policy = _STR_TO_EVICTION_POLICY[eviction_policy]
    except KeyError:
      raise ValueError(
          f"unsupported eviction policy: {eviction_policy}"
      ) from None

  if tt_dialect.PointerType.isinstance(ptr.type):
    ptr_type = tt_dialect.PointerType(ptr.type)
    if ir.RankedTensorType.isinstance(ptr_type.pointee_type):
      raise NotImplementedError("loading from a block pointer is not supported")

  ptr_type = _element_type(ptr.type)
  if not tt_dialect.PointerType.isinstance(ptr_type):
    raise ValueError(f"unsupported pointer type: {ptr_type}")
  ptr_type = tt_dialect.PointerType(ptr_type)
  if not ir.RankedTensorType.isinstance(ptr.type):
    if ir.RankedTensorType.isinstance(value.type):
      raise ValueError("value cannot be a block if pointer is not a block")
    if mask is not None and ir.RankedTensorType.isinstance(mask.type):
      raise ValueError("mask cannot be a block if pointer is not a block")

  pointee_type = ptr_type.pointee_type
  if isinstance(pointee_type, ir.IntegerType) and pointee_type.width == 1:
    pointee_type = ir.IntegerType.get_signless(8)
    ptr = _ir_cast(
        ptr,
        tt_dialect.PointerType.get(pointee_type, ptr_type.address_space),
        signed=False,
    )

  value = _ir_cast(value, pointee_type, signed=False)
  return tt_dialect.store(
      ptr, value, mask=mask, cache=cache_modifier, evict=eviction_policy
  )


@register_lowering(primitives.swap_p)
def _masked_swap_lowering_rule(
    ctx: LoweringRuleContext, *args_flat, args_tree, eviction_policy
):
  ptr, indexers, value, mask = args_tree.unflatten(args_flat)
  *_, value_aval, mask_aval = args_tree.unflatten(ctx.avals_in)
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  idx = indexers[0]
  ptr = _compute_pointers_from_indices(
      ptr, ctx.block_infos[0], idx, ctx.avals_in[0].shape
  )
  other = None
  if value is not None:
    value = _ensure_ir_value(value, value_aval)
  if mask is not None:
    mask = _bcast_to(_ensure_ir_value(mask, mask_aval), idx.get_indexer_shape())
    if value is not None:
      other = _bcast_to(value, idx.get_indexer_shape())

  old_value = _load(ptr, mask=mask, other=other)
  _store(ptr, value, mask=mask, eviction_policy=eviction_policy)
  return old_value


@register_lowering(sp.addupdate_p)
def _addupdate_lowering_rule(ctx: LoweringRuleContext, ptr, value, *idx, tree):
  indexers = tree_util.tree_unflatten(tree, idx)
  if not tt_dialect.PointerType.isinstance(ptr.type):
    assert len(indexers) == 0
    return ptr
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  indexer = indexers[0]
  ptr = _compute_pointers_from_indices(
      ptr,
      ctx.block_infos[0],
      indexer,
      ctx.avals_in[0].shape,
  )
  op = tt_dialect.RMWOp.FADD
  if isinstance(_element_type(value.type), ir.IntegerType):
    op = tt_dialect.RMWOp.ADD
  _atomic_rmw(op, ptr, value)
  return []


@register_lowering(lax.transpose_p)
def _transpose_lowering(ctx: LoweringRuleContext, x, *, permutation):
  return tt_dialect.trans(x, permutation)


def _check_dot_operands(
    x_type: ir.RankedTensorType, y_type: ir.RankedTensorType, options: Any
):
  # TODO(slebedev): Ensure that the dtypes are supported by CUDA.
  return


def _dot(
    x: ir.Value,
    y: ir.Value,
    acc: ir.Value | None = None,
    *,
    allow_tf32: bool = True,
    max_num_imprecise_acc: int | None = None,
    out_type: ir.Type | None = None,
) -> ir.Value:
  if out_type is None:
    out_type = ir.F32Type.get()
  elif isinstance(out_type, ir.BF16Type):
    raise NotImplementedError(f"unsupported output type: {out_type}")

  x_type = ir.RankedTensorType(x.type)
  y_type = ir.RankedTensorType(y.type)
  if min(*x_type.shape, *y_type.shape) < 16:
    raise ValueError("all dimensions of x and y must be >= 16 ")
  if x_type.element_type != y_type.element_type:
    raise ValueError(
        "x and y must have the same element type, but got:"
        f" {x_type.element_type} and {y_type.element_type}"
    )

  _check_dot_operands(x_type, y_type, object())

  element_type = x_type.element_type
  if isinstance(element_type, ir.IntegerType):
    if element_type.width != 8:
      raise TypeError(f"unsupported element type: {element_type}")
    element_type = ir.IntegerType.get_signless(32)
  elif isinstance(element_type, (ir.F32Type, ir.BF16Type)):
    element_type = ir.F32Type.get()
  else:
    element_type = out_type

  if element_type != out_type:
    raise TypeError(
        f"output type {out_type} does not match element type {element_type}"
    )

  m, _ = x_type.shape
  _, n = y_type.shape

  if acc is None:
    acc = _full(ir.RankedTensorType.get([m, n], element_type), 0)

  if max_num_imprecise_acc is None:
    if isinstance(element_type, ir.FloatType) and element_type.width == 8:
      # TODO(slebedev): Fill in from options.
      raise NotImplementedError
    else:
      max_num_imprecise_acc = 0

  # Ideally, replace all allow_tf32 usages with InputPrecision directly.
  input_precision = tt_dialect.InputPrecision.IEEE
  if allow_tf32:
    input_precision = tt_dialect.InputPrecision.TF32

  return tt_dialect.dot(
      x,
      y,
      acc,
      max_num_imprecise_acc=max_num_imprecise_acc,
      input_precision=input_precision
  )


_TF32_PRECISIONS = (lax.Precision.HIGH, lax.Precision.DEFAULT)


@register_lowering(lax.dot_general_p)
def _dot_general_lowering(
    ctx: LoweringRuleContext,
    a,
    b,
    *,
    dimension_numbers,
    precision,
    preferred_element_type,
):
  del preferred_element_type  # Unused.
  ((a_contract_dim,), (b_contract_dim,)), batch_dims = dimension_numbers
  assert batch_dims == ((), ())

  if a_contract_dim == 0:
    a = tt_dialect.trans(a, (1, 0))
  if b_contract_dim == 1:
    b = tt_dialect.trans(b, (1, 0))

  if precision is None:
    allow_tf32 = True
  else:
    prec_a, prec_b = precision
    allow_tf32 = prec_a in _TF32_PRECISIONS or prec_b in _TF32_PRECISIONS

  [out_aval] = ctx.avals_out
  out_dtype = acc_dtype = out_aval.dtype
  if acc_dtype != jnp.int32 and acc_dtype != jnp.float16:
    acc_dtype = jnp.dtype(jnp.float32)

  return _cast(
      _dot(
          a,
          b,
          allow_tf32=allow_tf32,
          out_type=_dtype_to_ir_type(acc_dtype),
      ),
      acc_dtype,
      out_dtype,
  )


def _reduction_lowering(body, ctx: LoweringRuleContext, a, axes):
  flat_args = tree_util.tree_leaves(a)
  (axis,) = axes
  mapped_avals = [jax_core.ShapedArray((), aval.dtype) for aval in ctx.avals_in]
  in_tree = tree_util.tree_structure((a, a))
  flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(body), in_tree
  )
  combine_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      flat_fun, [*mapped_avals, *mapped_avals]
  )
  out_tree = out_tree_thunk()
  del out_tree  # Not needed
  if consts:
    raise NotImplementedError("Reductions with constants not supported.")
  element_types = [_element_type(arg.type) for arg in flat_args]
  reduce_op = tt_dialect.ReduceOp(flat_args, axis)
  param_types = element_types * 2
  entry = reduce_op.regions[0].blocks.append(*param_types)
  with ir.InsertionPoint.at_block_begin(entry):
    results = lower_jaxpr_to_triton_ir(
        ctx.context, combine_jaxpr, None, *entry.arguments
    )
    tt_dialect.reduce_return(results)
  reduce_op.verify()
  return list(reduce_op.result)


def _reduce_lowering(body, ctx: LoweringRuleContext, a, *, axes):
  assert isinstance(axes, tuple)
  if not axes:
    return a
  while len(axes) > 1:
    axis = max(axes)
    dst_avals = tuple(v.update(shape=v.shape[:axis] + v.shape[axis + 1:])
                      for v in ctx.avals_in)
    a = _reduce_lowering(
        body, ctx.replace(avals_out=dst_avals), a, axes=(axis,))
    # Adding an intervening -(-reduce(.)) introduces a convert_layout between
    # reduces, which seems necessary for correctness.
    # TODO(bjp): Get rid of the double negation.
    #     https://github.com/openai/triton/issues/1776
    a = _minus(_minus(a))
    ctx = ctx.replace(avals_in=dst_avals)
    axes = tuple(ax for ax in axes if ax != axis)
  return _reduction_lowering(body, ctx, a, axes=axes)[0]


triton_lowering_rules[lax.reduce_max_p] = functools.partial(
    _reduce_lowering, jnp.maximum
)
triton_lowering_rules[lax.reduce_min_p] = functools.partial(
    _reduce_lowering, jnp.minimum
)
triton_lowering_rules[lax.reduce_sum_p] = functools.partial(
    _reduce_lowering, jnp.add
)


def _argreduce_lowering(
    body, ctx: LoweringRuleContext, a, *, axes, index_dtype
):
  if index_dtype != jnp.int32:
    raise ValueError("`index_type` must be i32.")
  if len(axes) != 1:
    raise ValueError("`pallas` reduce operations only support one reduce axis.")
  [axis] = axes
  [a_aval] = ctx.avals_in
  index = _make_range(0, a_aval.shape[axis])
  if len(a_aval.shape) > 1:
    # Broadcast index across the non-reduced axes
    for i in range(len(a_aval.shape)):
      if i != axis:
        index = _expand_dims(index, i)
    index = _bcast_to(index, a_aval.shape)
  ctx = ctx.replace(avals_in=[a_aval, a_aval.update(dtype=jnp.dtype("int32"))])
  _, indices = _reduction_lowering(body, ctx, (a, index), axes=axes)
  return indices


def _reduce_argmax_combine(left, right):
  value1, index1 = left
  value2, index2 = right
  gt = value1 > value2
  lt = value1 < value2
  index_min = jnp.minimum(index1, index2)
  index_ret = jnp.where(gt, index1, jnp.where(lt, index2, index_min))
  value_ret = jnp.maximum(value1, value2)
  return value_ret, index_ret


triton_lowering_rules[lax.argmax_p] = functools.partial(
    _argreduce_lowering, _reduce_argmax_combine
)


def _reduce_argmin_combine(left, right):
  value1, index1 = left
  value2, index2 = right
  gt = value1 > value2
  lt = value1 < value2
  index_min = jnp.minimum(index1, index2)
  index_ret = jnp.where(lt, index1, jnp.where(gt, index2, index_min))
  value_ret = jnp.minimum(value1, value2)
  return value_ret, index_ret


triton_lowering_rules[lax.argmin_p] = functools.partial(
    _argreduce_lowering, _reduce_argmin_combine
)


@register_lowering(pjit.pjit_p)
def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_triton_ir(
      ctx.context, jaxpr.jaxpr, ctx.block_infos, *args
  )


@register_lowering(jax_core.closed_call_p)
@register_lowering(custom_derivatives.custom_jvp_call_p)
def _closed_call_lowering_rule(
    ctx: LoweringRuleContext, *args, call_jaxpr, **_
):
  jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  if consts:
    raise NotImplementedError
  return lower_jaxpr_to_triton_ir(ctx.context, jaxpr, ctx.block_infos, *args)


@register_lowering(ad_checkpoint.remat_p)
def _remat_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  return lower_jaxpr_to_triton_ir(ctx.context, jaxpr, ctx.block_infos, *args)


triton_lowering_rules[ad_util.stop_gradient_p] = lambda _, x: x


@register_lowering(lax.axis_index_p)
def _axis_index_rule(ctx: LoweringRuleContext, *, axis_name: Hashable):
  grid_names = ctx.context.grid_mapping.grid_names
  if axis_name in grid_names:
    # We are querying a named axis corresponding to a grid dimension.
    return _program_id_lowering_rule(ctx, axis=grid_names.index(axis_name))
  raise LookupError(f"Axis name {axis_name} not found in grid.")

def _is_read_only(ref_effects) -> bool:
  if len(ref_effects) == 0:
    return True
  if len(ref_effects) > 1:
    # Means we must have a write or accum effect so not read-only
    return False
  (eff,) = ref_effects
  return isinstance(eff, state.ReadEffect)


@register_lowering(for_loop.for_p)
def _for_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr,
    which_linear,
    nsteps,
    reverse,
    unroll,
):
  del which_linear
  if reverse or unroll != 1:
    raise NotImplementedError
  _i_constant = _i64_constant if config.enable_x64.value else _i32_constant
  lower_bound = _i_constant(0)
  upper_bound = _i_constant(nsteps)
  step = _i_constant(1)
  init_args = map(_ensure_ir_value, args, ctx.avals_in)
  # Partially discharge state from jaxpr for non-pointers
  should_discharge = [
      not isinstance(a, state.AbstractRef) for a in ctx.avals_in
  ]
  discharged_jaxpr, () = discharge.discharge_state(
      jaxpr, (), should_discharge=[True, *should_discharge]
  )
  in_avals = [v.aval for v in jaxpr.invars]
  state_effects = state.get_ref_state_effects(in_avals, jaxpr.effects)[1:]
  # Read-only `Ref`s don't need to be passed in explicitly as loop arguments so
  # we can filter them out.
  read_only = map(_is_read_only, state_effects)
  is_loop_arg = map(
      operator.and_, map(operator.not_, read_only), should_discharge
  )
  ptrs, _ = partition_list(should_discharge, init_args)
  non_loop_args, loop_args = partition_list(is_loop_arg, init_args)
  for_op = scf_dialect.ForOp(lower_bound, upper_bound, step, loop_args)
  with ir.InsertionPoint(for_op.body):
    loop_index = for_op.induction_variable
    for_body_args = [
        for_op.body.arguments[i + 1] for i, _ in enumerate(loop_args)
    ]
    loop_body_args = merge_lists(is_loop_arg, non_loop_args, for_body_args)
    out_discharged = lower_jaxpr_to_triton_ir(
        ctx.context,
        discharged_jaxpr,
        [None, *ctx.block_infos],
        loop_index,
        *loop_body_args,
    )
    all_out = merge_lists(should_discharge, ptrs, out_discharged)
    _, loop_out = partition_list(is_loop_arg, all_out)
    scf_dialect.yield_(loop_out)
  return merge_lists(is_loop_arg, non_loop_args, list(for_op.results_))


def _lower_jaxpr_to_for_loop(
    ctx: LoweringRuleContext,
    jaxpr: jax_core.Jaxpr,
    lower_bound,
    upper_bound,
    consts,
    *args,
    has_loop_index: bool,
    step: int = 1,
    bound_type: ir.IntegerType | None = None,
):
  if step != 1:
    raise NotImplementedError
  if bound_type is None or bound_type.width == 32:
    step = _i32_constant(step)
  else:
    step = _i64_constant(step)

  for_op = scf_dialect.ForOp(lower_bound, upper_bound, step, args)
  with ir.InsertionPoint.at_block_begin(for_op.body):
    loop_index = for_op.induction_variable
    for_body_args = [for_op.body.arguments[i + 1] for i, _ in enumerate(args)]
    if has_loop_index:
      jaxpr_args = [*consts, loop_index, *for_body_args]
    else:
      jaxpr_args = [*consts, *for_body_args]
    all_out = lower_jaxpr_to_triton_ir(
        ctx.context,
        jaxpr,
        ctx.block_infos,
        *jaxpr_args)
    scf_dialect.yield_(all_out)

  return list(for_op.results_)


@register_lowering(lax.scan_p)
def _scan_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr,
    linear,
    length,
    reverse,
    unroll,
    num_consts,
    num_carry,
    _split_transpose,
):
  del _split_transpose
  # Only implements fori_loop-like scans
  num_extensive = len(args) - num_consts - num_carry
  if num_extensive: raise NotImplementedError
  if reverse: raise NotImplementedError
  if unroll != 1: raise NotImplementedError
  del linear, num_extensive, unroll, reverse

  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts: raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = (
      pallas_utils.pattern_match_scan_to_fori_loop(jaxpr, num_consts, num_carry)
  )
  args = map(_ensure_ir_value, args, ctx.avals_in)
  consts, args = util.split_list(args, [num_consts])
  if has_loop_index:
    lb, *args = args
    lower_bound = lb
    ub = _add(lb, _ir_constant(length, lb.type))
    upper_bound = ub
    bound_type = ub.type
  else:
    lower_bound = _i32_constant(0)
    upper_bound = _i32_constant(length)
    bound_type = ir.IntegerType.get_signless(32)
  for_out = _lower_jaxpr_to_for_loop(
      ctx, jaxpr, lower_bound, upper_bound, consts, *args,
      has_loop_index=has_loop_index, step=1, bound_type=bound_type)
  if has_loop_index:
    # Need to return the final loop index value if the outer scan expects
    # it as an output
    return [upper_bound, *for_out]
  return for_out


def _maybe_pattern_match_fori_loop(
    ctx: LoweringRuleContext,
    *args,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr,
):
  if cond_nconsts:
    return None
  _, cond_invars = split_list(cond_jaxpr.jaxpr.invars, [cond_nconsts])
  cond_in_avals = [v.aval for v in cond_invars]
  if len(cond_in_avals) < 2:
    return None
  # Check that the first two carry values are scalar ints
  a1, a2 = cond_in_avals[:2]
  if a1.shape != () or a1.dtype not in (jnp.int32, jnp.int64):
    return None
  if a2.shape != () or a2.dtype not in (jnp.int32, jnp.int64):
    return None
  # Check that the only eqn in the cond checks the loop index condition
  v1, v2 = cond_invars[:2]
  outvar = cond_jaxpr.jaxpr.outvars[0]
  assert outvar.aval.dtype == jnp.bool_
  if len(cond_jaxpr.jaxpr.eqns) != 1:
    return None
  eqn = cond_jaxpr.jaxpr.eqns[0]
  if eqn.primitive != lax.lt_p:
    return None
  if eqn.outvars != [outvar]:
    return None
  if eqn.invars != [v1, v2]:
    return None
  # Check that the carry is updated in the body appropriately
  _, body_invars = split_list(body_jaxpr.jaxpr.invars, [body_nconsts])
  v1, v2 = body_invars[:2]
  vo1, vo2 = body_jaxpr.jaxpr.outvars[:2]
  # Upper bound should be constant
  if v2 is not vo2:
    return None
  # Check that we increment the loop index in the body
  for i, eqn in enumerate(body_jaxpr.jaxpr.eqns):
    if eqn.primitive is lax.add_p:
      if eqn.invars[0] is v1:
        if isinstance(eqn.invars[1], jax_core.Literal):
          if eqn.invars[1].val == 1:
            if eqn.outvars[0] == vo1:
              eqn_index = i
              break
  else:
    return None
  jaxpr = body_jaxpr.jaxpr
  new_invars = (*jaxpr.invars[:body_nconsts],
                jaxpr.invars[body_nconsts],
                *jaxpr.invars[body_nconsts + 2:])
  new_outvars = tuple(jaxpr.outvars[2:])
  jaxpr = jaxpr.replace(
      eqns=jaxpr.eqns[:eqn_index] + jaxpr.eqns[eqn_index + 1:],
      invars=new_invars,
      outvars=new_outvars)
  _, body_consts, carry = split_list(args, [cond_nconsts, body_nconsts])
  (lb, ub), args = carry[:2], carry[2:]
  const_block_infos, args_block_infos = split_list(ctx.block_infos,
                                                   [body_nconsts])
  ctx = ctx.replace(block_infos=[*const_block_infos, None,
                                 *args_block_infos[2:]])
  for_out = _lower_jaxpr_to_for_loop(
      ctx,
      jaxpr,
      lb,
      ub,
      body_consts,
      *args,
      has_loop_index=True,
      step=1,
      bound_type=lb.type,
  )
  return [ub, ub, *for_out]


@register_lowering(lax.while_p)
def _while_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr,
):
  args = map(_ensure_ir_value, args, ctx.avals_in)

  # First, try to pattern match to fori_loop and lower to scf.for if possible
  result = _maybe_pattern_match_fori_loop(ctx, *args, cond_nconsts=cond_nconsts,
                                          body_nconsts=body_nconsts, cond_jaxpr=cond_jaxpr,
                                          body_jaxpr=body_jaxpr)
  if result is not None:
    return result
  # Fall back to default while lowering
  cond_consts, body_consts, carry = util.split_list(
      args, [cond_nconsts, body_nconsts]
  )
  cond_const_block_infos, body_const_block_infos, carry_block_infos = (
      util.split_list(ctx.block_infos, [cond_nconsts, body_nconsts])
  )
  cond_const_types = [a.type for a in cond_consts]
  body_const_types = [a.type for a in body_consts]
  carry_types = [a.type for a in carry]
  all_types = [*cond_const_types, *body_const_types, *carry_types]
  while_op = scf_dialect.WhileOp(all_types, args)

  before_block = while_op.before.blocks.append(*all_types)
  cond_consts_, _, carry_ = util.split_list(
      before_block.arguments,
      [cond_nconsts, body_nconsts],
  )
  cond_args = [*cond_consts_, *carry_]
  with ir.InsertionPoint.at_block_begin(before_block):
    [cond] = lower_jaxpr_to_triton_ir(
        ctx.context,
        cond_jaxpr.jaxpr,
        [*cond_const_block_infos, *carry_block_infos],
        *cond_args,
    )
    scf_dialect.condition(cond, before_block.arguments)

  after_block = while_op.after.blocks.append(*all_types)
  cond_consts_, body_consts_, carry_ = util.split_list(
      after_block.arguments,
      [cond_nconsts, body_nconsts],
  )
  all_args = [*cond_consts_, *body_consts_, *carry_]
  cond_const_args, body_const_args, carry_args = util.split_list(
      all_args, [cond_nconsts, body_nconsts]
  )
  with ir.InsertionPoint.at_block_begin(after_block):
    loop_out = lower_jaxpr_to_triton_ir(
        ctx.context,
        body_jaxpr.jaxpr,
        [*body_const_block_infos, *carry_block_infos],
        *body_const_args,
        *carry_args
    )
    all_handles = [*cond_const_args, *body_const_args, *loop_out]
    if all_handles:
      scf_dialect.yield_(all_handles)

  all_out = list(while_op.results_)
  return all_out[cond_nconsts + body_nconsts :]


@register_lowering(lax.cond_p)
def _cond_lowering_rule(
    ctx: LoweringRuleContext,
    index,
    *args,  # *consts, *ops
    branches,  # tuple(jaxprs)
):
  block_infos = ctx.block_infos

  def to_type(out_aval):
    element_type = _dtype_to_ir_type(out_aval.dtype)
    if not out_aval.shape:
      return element_type
    return ir.RankedTensorType.get(out_aval.shape, element_type)

  out_types = [to_type(out) for out in ctx.avals_out]

  use_branch0 = _equal(index, _ir_constant(0, index.type), signed=False)
  # TODO(bjp): Switch to scf.index_switch once exposed in triton.cc
  if_op = scf_dialect.IfOp(use_branch0, out_types, hasElse=True)
  with ir.InsertionPoint.at_block_begin(if_op.then_block):
    outs0 = lower_jaxpr_to_triton_ir(
        ctx.context,
        branches[0].jaxpr,
        block_infos[1:],
        *args)
    scf_dialect.yield_(outs0)
  with ir.InsertionPoint.at_block_begin(if_op.else_block):
    # TODO(bjp): Instead of linear nest of 'if's, partition into halves.
    if len(branches) > 2:
      outs1 = _cond_lowering_rule(
          ctx,
          _sub(index, _ir_constant(1, index.type)),
          *args,
          branches=branches[1:],
      )
    else:
      outs1 = lower_jaxpr_to_triton_ir(
          ctx.context,
          branches[1].jaxpr,
          block_infos[1:],
          *args)
    scf_dialect.yield_(outs1)

  return list(if_op.results_)


def _ensure_ir_value(x: object, aval: jax_core.ShapedArray) -> ir.Value:
  if isinstance(x, ir.Value):
    return x
  elif isinstance(x, (np.number, np.ndarray, int, float)):
    return _ir_constant(x, _dtype_to_ir_type(aval.dtype))
  raise NotImplementedError


def _ir_constant(v: object, t: ir.Type) -> ir.Value:
  if isinstance(v, (np.number, np.ndarray, int, float)):
    if isinstance(t, ir.IntegerType):
      v = int(v)
    else:
      assert isinstance(t, ir.FloatType)
      v = float(v)
    return arith_dialect.constant(t, v)
  raise NotImplementedError


def _i32_constant(v: int) -> ir.Value:
  return arith_dialect.constant(ir.IntegerType.get_signless(32), v)


def _i64_constant(v: int) -> ir.Value:
  return arith_dialect.constant(ir.IntegerType.get_signless(64), v)


def _dtype_to_ir_type(dtype: jnp.dtype) -> ir.Type:
  if jnp.issubdtype(dtype, np.integer):
    # All integer types in Triton are signless.
    return ir.IntegerType.get_signless(dtype.itemsize * 8)
  return mlir.dtype_to_ir_type(dtype)
