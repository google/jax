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

import dataclasses
import functools
import operator
from typing import Any, Dict, Sequence, Tuple
import zlib

import jax
from jax import lax
from jax import tree_util
from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import state
from jax._src import util
from jax._src.lax.control_flow import for_loop
from jax._src.lib import gpu_triton as triton_kernel_call_lib
from jax._src.lib import hlo_helpers
from jax._src.lib import version
from jax._src.lib.mlir import ir
from jax._src.pallas import core as pallas_core
from jax._src.pallas import indexing
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.pallas_call import pallas_call_p
from jax._src.state import AbstractRef
from jax._src.state import discharge
from jax._src.state import primitives as sp
from jax._src.util import merge_lists
from jax._src.util import partition_list
from jax._src.util import split_list
from jax._src.util import weakref_lru_cache
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
from jax_triton import triton_lib
from jax_triton.triton_lib import compile_ttir_to_ptx_inplace
from jax_triton.triton_lib import get_triton_type
import numpy as np
from triton._C.libtriton.triton import ir as tl_ir
from triton.compiler import code_generator as code_gen
import triton.language as tl

# TODO(sharadmv): enable type checking
# mypy: ignore-errors

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip
partial = functools.partial
Grid = Tuple[int, ...]
NDIndexer = indexing.NDIndexer
GridMapping = pallas_core.GridMapping
BlockMapping = pallas_core.BlockMapping


# # General lowering logic
@dataclasses.dataclass
class TritonModuleContext:
  name: str
  ir_context: tl_ir.context
  builder: tl_ir.builder
  module: tl_ir.module
  grid_mapping: GridMapping
  program_ids: Sequence[tl.tensor]


@dataclasses.dataclass
class BlockInfo:
  full_shape_dtype: jax.ShapeDtypeStruct
  start_indices: Sequence[Any]
  block_shape: Tuple[int, ...]


@dataclasses.dataclass
class TritonLoweringRuleContext:
  context: TritonModuleContext
  avals_in: Any
  avals_out: Any
  block_infos: Sequence[BlockInfo | None]

  def __post_init__(self):
    self.builder = self.context.builder

  replace = dataclasses.replace


@dataclasses.dataclass
class TritonLoweringResult:
  """Keeps pybind11 objects alive."""

  ir_context: tl_ir.context
  module: tl_ir.module
  builder: tl_ir.builder
  grid: Tuple[int, ...]


@dataclasses.dataclass
class TritonCompilationResult:
  kernel_name: str
  ttir: str
  ptx: str
  shared_mem_bytes: int
  compute_capability: int
  lowering_result: TritonLoweringResult


class TritonLoweringException(Exception):
  pass


def _eval_index_map(
    ctx: TritonModuleContext, idx, block_mapping: BlockMapping | None
):
  if block_mapping is None:
    return None
  block_indices = tuple(
      lower_jaxpr_to_triton_ir(
          ctx, block_mapping.index_map_jaxpr.jaxpr, None, *idx
      )
  )
  return tuple(
      i if b is pallas_core.mapped else i.__mul__(b, _builder=ctx.builder)
      for i, b in zip(block_indices, block_mapping.block_shape)
  )


triton_lowering_rules = {}


def _process_grid_to_3d_grid(builder, grid_mapping: GridMapping):
  if len(grid_mapping.grid) <= 3:
    program_ids = [
        tl.program_id(axis=i, _builder=builder)
        for i in range(len(grid_mapping.grid))
    ]
    return grid_mapping.grid, program_ids
  grid_prefix = grid_mapping.grid[:-2]
  grid_suffix = grid_mapping.grid[-2:]
  total_axis_size = np.prod(grid_prefix)
  new_grid = (total_axis_size, *grid_suffix)
  out_indices = [0] * len(grid_prefix)
  grid0 = tl.program_id(0, _builder=builder)
  for i, s in reversed(list(enumerate(grid_prefix))):
    grid0, out_indices[i] = (
        grid0.__floordiv__(s, _builder=builder),
        grid0.__mod__(s, _builder=builder),
    )
  out_indices = [
      *out_indices,
      tl.program_id(1, _builder=builder),
      tl.program_id(2, _builder=builder),
  ]
  assert len(out_indices) == len(grid_mapping.grid)
  return new_grid, out_indices


def lower_jaxpr_to_triton_module(
    jaxpr: jax_core.Jaxpr, in_shapes, grid_mapping: GridMapping, name: str
) -> tl_ir.module:
  jaxpr, _ = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars), instantiate=True)
  ir_context = tl_ir.context()
  ir_context.load_triton()
  builder = tl_ir.builder(ir_context)
  # TODO(sharadmv): handle multiple devices, right now we assume device 0
  # which is fine when we have multiple of the same GPU but this won't work in
  # general.
  device = 0
  builder.arch = triton_kernel_call_lib.get_compute_capability(device)
  module = builder.create_module()
  in_avals = [var.aval for var in jaxpr.invars]
  triton_types = [get_triton_type(x) for x in in_avals]
  arg_types = [code_gen.str_to_ty(arg) for arg in triton_types]
  assert len(jaxpr.outvars) == 0
  prototype = tl.function_type([], arg_types)
  out = prototype.to_ir(builder)
  fn = builder.get_or_insert_function(module, name, out, "public", False)
  module.push_back(fn)
  entry = fn.add_entry_block()
  args = []
  for i in range(len(in_avals)):
    fn.set_arg_attr(i, "tt.divisibility", 16)
    ptr = tl.tensor(fn.args(i), prototype.param_types[i])
    args.append(ptr)
  builder.set_insertion_point_to_start(entry)
  new_grid, program_ids = _process_grid_to_3d_grid(builder, grid_mapping)
  local_program_ids = [
      pid for i, pid in enumerate(program_ids) if i not in grid_mapping.mapped_dims
  ]
  ctx = TritonModuleContext(
      name, ir_context, builder, module, grid_mapping, local_program_ids
  )
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "Scalar prefetch not supported in Triton lowering.")
  start_indices = map(
      partial(_eval_index_map, ctx, program_ids), grid_mapping.block_mappings
  )
  block_infos = [
      BlockInfo(
          jax.ShapeDtypeStruct(shape_dtype.shape, shape_dtype.dtype),
          start_idx,
          block_mapping.block_shape,
      )
      if block_mapping is not None
      else None
      for shape_dtype, block_mapping, start_idx in zip(
          in_shapes, grid_mapping.block_mappings, start_indices
      )
  ]
  () = lower_jaxpr_to_triton_ir(ctx, jaxpr, block_infos, *args)
  module.context = ir_context
  ctx.builder.ret([])
  return TritonLoweringResult(ir_context, module, builder, new_grid)


def lower_jaxpr_to_triton_ir(
    ctx: TritonModuleContext,
    jaxpr: jax_core.Jaxpr,
    block_infos: Sequence[BlockInfo | None] | None,
    *args
) -> Sequence[Any]:
  env = {}
  block_info_env = {}

  def read_env(var: jax_core.Atom):
    if type(var) is jax_core.Literal:
      t = tl.core._to_tensor(np.array(var.val).tolist(), builder=ctx.builder)
      dst_ty = code_gen.str_to_ty(get_triton_type(var.aval)).element_ty
      if t.type.scalar != dst_ty:
        # _to_tensor(np.array(var.val).tolist()) can be lossy e.g. np.float64
        # comes out of .tolist() as list[float], which then comes out of
        # _to_tensor as a block of f32.
        t = tl.semantic.cast(t, dst_ty, ctx.builder)
      return t
    return env[var]

  def read_block_info_env(var: jax_core.Atom):
    if type(var) is jax_core.Literal:
      return None
    return block_info_env.get(var, None)

  def write_env(var: jax_core.Var, val):
    env[var] = val

  if block_infos is None:
    block_infos = [None] * len(jaxpr.invars)
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
    rule_ctx = TritonLoweringRuleContext(
        ctx, avals_in, avals_out, eqn_block_infos
    )
    try:
      outvals = rule(rule_ctx, *invals, **eqn.params)
    except TritonLoweringException:
      raise  # We only add the extra info to the innermost exception.
    except Exception as e:
      raise TritonLoweringException(
          f"Exception while lowering eqn:\n  {eqn}\n"
          f"With context:\n  {rule_ctx}\n"
          f"With inval shapes={map(lambda t: t.shape, invals)}\n"
          f"With inval types={map(lambda t: t.type, invals)}\n"
          f"In jaxpr:\n{jaxpr}") from e
    if eqn.primitive.multiple_results:
      map(write_env, eqn.outvars, outvals)
    else:
      write_env(eqn.outvars[0], outvals)
  return map(read_env, jaxpr.outvars)


# # Primitive lowering rules
# ## Programming model primitives
def _program_id_lowering_rule(ctx: TritonLoweringRuleContext, *, axis):
  return ctx.context.program_ids[axis]


triton_lowering_rules[primitives.program_id_p] = _program_id_lowering_rule
# ## Atomic op primitives
_ATOMIC_OP_MAPPING = {
    primitives.AtomicOpType.XCHG: tl.core.atomic_xchg,
    primitives.AtomicOpType.ADD: tl.core.atomic_add,
    primitives.AtomicOpType.MAX: tl.core.atomic_max,
    primitives.AtomicOpType.MIN: tl.core.atomic_min,
    primitives.AtomicOpType.AND: tl.core.atomic_and,
    primitives.AtomicOpType.OR: tl.core.atomic_or,
    primitives.AtomicOpType.XOR: tl.core.atomic_xor,
}


def _atomic_lowering_rule(
    ctx: TritonLoweringRuleContext,
    ptr,
    value,
    *args,
    args_tree,
    masked: bool,
    atomic_type: primitives.AtomicOpType
):
  ref_block_info, *_ = ctx.block_infos
  idx, *mask_rest = tree_util.tree_unflatten(args_tree, args)
  avals_in = ctx.avals_in
  ptr = _compute_pointers_from_indices(
      ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder
  )
  mask = None
  if masked:
    assert len(mask_rest) == 1
    (mask,) = mask_rest
  if atomic_type not in _ATOMIC_OP_MAPPING:
    raise NotImplementedError(atomic_type)
  op = _ATOMIC_OP_MAPPING[atomic_type]
  return op(ptr, value, mask=mask, _builder=ctx.builder)


triton_lowering_rules[primitives.atomic_rmw_p] = _atomic_lowering_rule


def _atomic_cas_lowering_rule(ctx: TritonLoweringRuleContext, ptr, cmp, val):
  return tl.atomic_cas(ptr, cmp, val, _builder=ctx.builder)


triton_lowering_rules[primitives.atomic_cas_p] = _atomic_cas_lowering_rule


def _max_contiguous_lowering_rule(ctx: TritonLoweringRuleContext, x, *, values):
  values = [tl.constexpr(v) for v in values]
  return tl.max_contiguous(x, values, _builder=ctx.builder)


triton_lowering_rules[primitives.max_contiguous_p] = (
    _max_contiguous_lowering_rule
)


def _multiple_of_lowering_rule(ctx: TritonLoweringRuleContext, x, *, values):
  values = [tl.constexpr(v) for v in values]
  return tl.multiple_of(x, values, _builder=ctx.builder)


triton_lowering_rules[primitives.multiple_of_p] = _multiple_of_lowering_rule


def _abs_lowering_rule(ctx: TritonLoweringRuleContext, x):
  return tl.abs(x, _builder=ctx.builder)


triton_lowering_rules[lax.abs_p] = _abs_lowering_rule


def _clamp_lowering_rule(ctx: TritonLoweringRuleContext, min, operand, max):
  return _min_lowering_rule(ctx, max_lowering_rule(ctx, min, operand), max)


triton_lowering_rules[lax.clamp_p] = _clamp_lowering_rule


def _exp_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.exp(a, _builder=ctx.builder)


triton_lowering_rules[lax.exp_p] = _exp_lowering_rule


def _exp2_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.math.exp2(a, _builder=ctx.builder)


triton_lowering_rules[lax.exp2_p] = _exp2_lowering_rule


def _log_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.log(a, _builder=ctx.builder)


triton_lowering_rules[lax.log_p] = _log_lowering_rule


def _log1p_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.math.log1p(a, _builder=ctx.builder)


triton_lowering_rules[lax.log1p_p] = _log1p_lowering_rule


def _logistic_lowering_rule(ctx: TritonLoweringRuleContext, a):
  one_ = tl.core._to_tensor(1.0, ctx.builder)
  x = tl.exp(a.__neg__(_builder=ctx.builder), _builder=ctx.builder)
  x = x.__add__(one_, _builder=ctx.builder)
  x = one_.__truediv__(x, _builder=ctx.builder)
  return x


triton_lowering_rules[lax.logistic_p] = _logistic_lowering_rule


def _sin_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.sin(a, _builder=ctx.builder)


triton_lowering_rules[lax.sin_p] = _sin_lowering_rule


def _cos_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.cos(a, _builder=ctx.builder)


triton_lowering_rules[lax.cos_p] = _cos_lowering_rule


def _mul_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__mul__(b, _builder=ctx.builder)


triton_lowering_rules[lax.mul_p] = _mul_lowering_rule


def _div_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  floating_dtypes = {tl.float16, tl.float32, tl.float64, tl.bfloat16}
  if a.dtype in floating_dtypes and b.dtype in floating_dtypes:
    return a.__truediv__(b, _builder=ctx.builder)
  return a.__floordiv__(b, _builder=ctx.builder)


triton_lowering_rules[lax.div_p] = _div_lowering_rule


def _iota_lowering_rule(
    ctx: TritonLoweringRuleContext, *, dtype, shape, dimension
):
  if dimension != 0:
    raise NotImplementedError()
  return tl.arange(0, shape[0], _builder=ctx.builder)


triton_lowering_rules[lax.iota_p] = _iota_lowering_rule


def _add_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__add__(b, _builder=ctx.builder)


triton_lowering_rules[lax.add_p] = _add_lowering_rule
triton_lowering_rules[ad_util.add_any_p] = _add_lowering_rule


def _integer_pow_lowering_rule(ctx: TritonLoweringRuleContext, a, *, y):
  if y == 2:
    return a.__mul__(a, _builder=ctx.builder)
  if y == 3:
    return a.__mul__(a.__mul__(a, _builder=ctx.builder), _builder=ctx.builder)
  if y == -2:
    one_ = tl.core._to_tensor(1.0, ctx.builder)
    a_sq = a.__mul__(a, _builder=ctx.builder)
    return one_.__truediv__(a_sq, _builder=ctx.builder)
  return tl.math.pow(a, y, _builder=ctx.builder)


triton_lowering_rules[lax.integer_pow_p] = _integer_pow_lowering_rule


def _pow_lowering_rule(ctx: TritonLoweringRuleContext, a, y):
  return tl.math.pow(a, y, _builder=ctx.builder)


triton_lowering_rules[lax.pow_p] = _pow_lowering_rule


def _tanh_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.math.tanh(a, _builder=ctx.builder)


triton_lowering_rules[lax.tanh_p] = _tanh_lowering_rule


def _min_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  pred = a.__lt__(b, _builder=ctx.builder)
  return tl.semantic.where(pred, a, b, ctx.builder)


triton_lowering_rules[lax.min_p] = _min_lowering_rule


def _convert_element_type_lowering_rule(
    ctx: TritonLoweringRuleContext, a, *, new_dtype, weak_type
):
  if new_dtype == ctx.avals_in[0].dtype:
    return a
  if new_dtype == jnp.float32:
    new_dtype = tl.float32
  elif new_dtype == jnp.float64:
    new_dtype = tl.float64
  elif new_dtype == jnp.float16:
    new_dtype = tl.float16
  elif new_dtype == jnp.bfloat16:
    new_dtype = tl.bfloat16
  elif new_dtype == jnp.int32:
    new_dtype = tl.int32
  elif new_dtype == jnp.int64:
    new_dtype = tl.int64
  else:
    raise ValueError(f"Unhandled dtype: {new_dtype}")
  return tl.semantic.cast(a, new_dtype, ctx.builder)


triton_lowering_rules[lax.convert_element_type_p] = (
    _convert_element_type_lowering_rule
)


def max_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  pred = a.__gt__(b, _builder=ctx.builder)
  return tl.semantic.where(pred, a, b, ctx.builder)


triton_lowering_rules[lax.max_p] = max_lowering_rule


def ge_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__ge__(b, _builder=ctx.builder)


triton_lowering_rules[lax.ge_p] = ge_lowering_rule


def eq_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__eq__(b, _builder=ctx.builder)


triton_lowering_rules[jax.lax.eq_p] = eq_lowering_rule


def ne_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__ne__(b, _builder=ctx.builder)


triton_lowering_rules[jax.lax.ne_p] = ne_lowering_rule

def bitwise_and_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__and__(b, _builder=ctx.builder)


triton_lowering_rules[jax.lax.and_p] = bitwise_and_lowering_rule


def bitwise_or_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__or__(b, _builder=ctx.builder)


triton_lowering_rules[jax.lax.or_p] = bitwise_or_lowering_rule


def select_n_lowering_rule(ctx: TritonLoweringRuleContext, pred, a, b):
  return tl.semantic.where(pred, b, a, ctx.builder)


triton_lowering_rules[lax.select_n_p] = select_n_lowering_rule


def _rem_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__mod__(b, _builder=ctx.builder)


triton_lowering_rules[lax.rem_p] = _rem_lowering_rule


def _sub_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__sub__(b, _builder=ctx.builder)


triton_lowering_rules[lax.sub_p] = _sub_lowering_rule


def _lt_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__lt__(b, _builder=ctx.builder)


triton_lowering_rules[lax.lt_p] = _lt_lowering_rule


def _gt_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__gt__(b, _builder=ctx.builder)


triton_lowering_rules[lax.gt_p] = _gt_lowering_rule


def _sqrt_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.sqrt(a, _builder=ctx.builder)


triton_lowering_rules[lax.sqrt_p] = _sqrt_lowering_rule


def _rsqrt_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return tl.math.rsqrt(a, _builder=ctx.builder)


triton_lowering_rules[lax.rsqrt_p] = _rsqrt_lowering_rule


def _neg_lowering_rule(ctx: TritonLoweringRuleContext, a):
  return a.__neg__(_builder=ctx.builder)


triton_lowering_rules[lax.neg_p] = _neg_lowering_rule


def _shift_left_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return a.__lshift__(b, _builder=ctx.builder)


triton_lowering_rules[lax.shift_left_p] = _shift_left_lowering_rule


def _shift_right_arithmetic_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return tl.semantic.ashr(a, b, builder=ctx.builder)


triton_lowering_rules[lax.shift_right_arithmetic_p] = (
    _shift_right_arithmetic_lowering_rule
)


def _shift_right_logical_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return tl.semantic.lshr(a, b, builder=ctx.builder)


triton_lowering_rules[lax.shift_right_logical_p] = (
    _shift_right_logical_lowering_rule
)


def _broadcast_in_dim_lowering_rule(
    ctx: TritonLoweringRuleContext, a, *, broadcast_dimensions, shape
):
  shape = map(tl.constexpr, shape)
  if not a.type.is_block():
    return tl.broadcast_to(a, shape, _builder=ctx.builder)
  expand_dims = [i for i in range(len(shape)) if i not in broadcast_dimensions]
  for dim in expand_dims:
    a = tl.semantic.expand_dims(a, dim, ctx.builder)
  return tl.broadcast_to(a, shape, _builder=ctx.builder)


triton_lowering_rules[jax.lax.broadcast_in_dim_p] = (
    _broadcast_in_dim_lowering_rule
)


def _broadcast_to_lowering_rule(
    ctx: TritonLoweringRuleContext, a, *, shape
):
  shape = map(tl.constexpr, shape)
  return tl.broadcast_to(a, shape, _builder=ctx.builder)


triton_lowering_rules[indexing.broadcast_to_p] = (
    _broadcast_to_lowering_rule
)


def _squeeze_lowering_rule(ctx: TritonLoweringRuleContext, a, *, dimensions):
  del dimensions
  return _reshape_lowering_rule(ctx, a, new_sizes=None, dimensions=None)


triton_lowering_rules[lax.squeeze_p] = _squeeze_lowering_rule


def _reshape_lowering_rule(
    ctx: TritonLoweringRuleContext, a, *, new_sizes, dimensions
):
  del new_sizes, dimensions
  # Short-circuit to avoid unneeded reshape.
  dst_shp = ctx.avals_out[0].shape
  if tuple(s.value for s in a.shape) == dst_shp:
    return a
  if not a.type.is_block():
    if dst_shp:
      return tl.broadcast_to(a, [tl.constexpr(s) for s in dst_shp],
                             _builder=ctx.builder)
    return a
  # Expand-dims or reduce-sum to handle singleton dims.
  if ([s.value for s in a.shape if s.value != 1] ==
      [s for s in dst_shp if s != 1]):
    # Eliminate one difference and recurse.
    for i in range(max(len(a.shape), len(dst_shp))):
      if (i < len(a.shape) and i < len(dst_shp) and
          a.shape[i].value == dst_shp[i]):
        continue
      # Use expand_dims to add a singleton dim.
      if i < len(dst_shp) and dst_shp[i] == 1:
        return _reshape_lowering_rule(
            ctx, tl.semantic.expand_dims(a, i, builder=ctx.builder),
            new_sizes=None, dimensions=None)
      # Use a reduction to eliminate singleton dim.
      if a.shape[i].value == 1:
        reduce_ctx = ctx.replace(
            avals_in=[ctx.avals_in[0].update(
                shape=tuple(d.value for d in a.shape))],
            avals_out=[ctx.avals_in[0].update(
                shape=tuple(d.value for di, d in enumerate(a.shape)
                            if di != i))])
        return _reshape_lowering_rule(
            ctx,
            _reduce_lowering(jnp.add, reduce_ctx, a, axes=(i,)),
            new_sizes=None, dimensions=None)

  shape = [tl.constexpr(s) for s in dst_shp]
  return tl.reshape(a, shape, _builder=ctx.builder)


triton_lowering_rules[jax.lax.reshape_p] = _reshape_lowering_rule


def _compute_pointers_from_indices(
    root_ptr: tl.core.tensor,
    block_info: BlockInfo | None,
    nd_indexer: NDIndexer,
    array_shape: Tuple[int, ...],
    builder: tl_ir.builder,
) -> tl.core.tensor:
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
      index = tl.core._to_tensor(0, builder)
    else:
      index = next(indexer_iter)
    if isinstance(index, primitives.Slice):
      # Handle slices with static and dynamic indices and static sizes
      if isinstance(index.start, int):
        ptr_dim_offset = tl.arange(
            index.start, index.start + index.size, _builder=builder
        )
      else:
        ptr_dim_offset = index.start.__add__(
            tl.arange(0, index.size, _builder=builder), _builder=builder
        )
      # We need to add broadcastable dimensions for the advanced int indexing
      # and for previous slices
      num_left_expand_dims = len(int_indexer_shape) + other_shape_idx
      num_right_expand_dims = len(other_shape) - other_shape_idx - 1
      other_shape_idx += 1
    elif isinstance(index, slice):
      if index != slice(None):
        raise NotImplementedError("Only `slice(None)` allowed.")
      ptr_dim_offset = tl.arange(0, dim_block_size, _builder=builder)
      num_left_expand_dims = len(int_indexer_shape) + other_shape_idx
      num_right_expand_dims = len(other_shape) - other_shape_idx - 1
      other_shape_idx += 1
    else:
      # indexer is either a *scalar* or an array of size `int_indexer_shape`
      ptr_dim_offset = index
      num_left_expand_dims = 0
      num_right_expand_dims = len(other_shape)
      if not ptr_dim_offset.type.is_block():
        num_left_expand_dims = max(len(indexer_shape) - 1, 0)
      else:
        num_right_expand_dims = len(other_shape)
    if not ptr_dim_offset.type.is_block() and indexer_shape:
      ptr_dim_offset = tl.broadcast_to(
          ptr_dim_offset,
          [tl.constexpr(1)] * len(indexer_shape),
          _builder=builder,
      )
    else:
      for _ in range(num_left_expand_dims):
        ptr_dim_offset = tl.semantic.expand_dims(ptr_dim_offset, 0, builder)
      for _ in range(num_right_expand_dims):
        ndim = len(ptr_dim_offset.shape)
        ptr_dim_offset = tl.semantic.expand_dims(ptr_dim_offset, ndim, builder)
    if start_offset is not None:
      ptr_dim_offset = ptr_dim_offset.__add__(start_offset, _builder=builder)
    stride_size = tl.core._to_tensor(int(dim_stride), builder)
    bcast_indices.append(ptr_dim_offset.__mul__(stride_size, _builder=builder))
  block_shapes = [
      () if not index.type.is_block() else tuple(index.type.get_block_shapes())
      for index in bcast_indices
  ]
  bcast_indices = [
      tl.core.broadcast_to(
          index, map(tl.constexpr, indexer_shape), _builder=builder
      )
      if indexer_shape != block_shape
      else index
      for index, block_shape in zip(bcast_indices, block_shapes)
  ]
  ptr = root_ptr
  for bcast_idx in bcast_indices:
    ptr = ptr.__add__(bcast_idx, _builder=builder)
  return ptr


def _pack_indices(non_slice_idx, indexed_dims):
  non_slice_idx_iter = iter(non_slice_idx)
  return tuple(
      next(non_slice_idx_iter) if indexed else slice(None)
      for indexed in indexed_dims
  )


def _get_lowering_rule(
    ctx: TritonLoweringRuleContext, ptr, *non_slice_idx, indexed_dims
):
  ref_block_info, *_ = ctx.block_infos
  idx = _pack_indices(non_slice_idx, indexed_dims)
  avals_in = ctx.avals_in
  idx_avals = _pack_indices(avals_in[1:], indexed_dims)
  if not isinstance(ptr.type, tl.pointer_type):
    assert len(avals_in) == 1
    return ptr
  if non_slice_idx:
    (int_indexer_shape,) = {
        i.shape for i in idx_avals if not isinstance(i, slice)
    }
  else:
    int_indexer_shape = ()
  idx = tuple(
      primitives.Slice.from_slice(slc, s) if isinstance(slc, slice) else slc
      for s, slc in zip(avals_in[0].shape, idx)
  )
  idx = primitives.NDIndexer(idx, avals_in[0].shape, int_indexer_shape)
  ptr = _compute_pointers_from_indices(
      ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder
  )
  return tl.load(ptr, _builder=ctx.builder)


triton_lowering_rules[sp.get_p] = _get_lowering_rule


def _masked_load_lowering_rule(
    ctx: TritonLoweringRuleContext,
    ptr,
    *args,
    args_tree,
    masked,
    eviction_policy,
    cache_modifier,
    is_volatile
):
  ref_block_info, *_ = ctx.block_infos
  idx, *mask_other = tree_util.tree_unflatten(args_tree, args)
  avals_in = ctx.avals_in
  if not isinstance(ptr.type, tl.pointer_type):
    assert len(avals_in) == 1
    return ptr
  ptr = _compute_pointers_from_indices(
      ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder
  )
  mask, other = None, None
  if masked:
    assert 0 < len(mask_other) <= 2
    if len(mask_other) == 2:
      mask, other = mask_other
    elif len(mask_other) == 1:
      (mask,) = mask_other
  return tl.load(
      ptr,
      mask=mask,
      other=other,
      cache_modifier=cache_modifier,
      volatile=is_volatile,
      eviction_policy=eviction_policy,
      _builder=ctx.builder,
  )


triton_lowering_rules[primitives.load_p] = _masked_load_lowering_rule


def _swap_lowering_rule(
    ctx: TritonLoweringRuleContext, ptr, value, *non_slice_idx, indexed_dims
):
  ref_block_info, *_ = ctx.block_infos
  avals_in = ctx.avals_in
  idx = _pack_indices(non_slice_idx, indexed_dims)
  idx_avals = _pack_indices(avals_in[2:], indexed_dims)
  if non_slice_idx:
    (int_indexer_shape,) = {
        i.shape for i in idx_avals if not isinstance(i, slice)
    }
  else:
    int_indexer_shape = ()
  idx = tuple(
      primitives.Slice.from_slice(slc, s) if isinstance(slc, slice) else slc
      for s, slc in zip(avals_in[0].shape, idx)
  )
  idx = primitives.NDIndexer(idx, avals_in[0].shape, int_indexer_shape)
  ptr = _compute_pointers_from_indices(
      ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder
  )
  mask = None
  old_value = tl.load(ptr, mask=mask, _builder=ctx.builder)
  tl.store(ptr, value, mask=mask, _builder=ctx.builder)
  return old_value


triton_lowering_rules[sp.swap_p] = _swap_lowering_rule


def _masked_swap_lowering_rule(
    ctx: TritonLoweringRuleContext,
    ptr,
    value,
    *args,
    args_tree,
    masked,
    eviction_policy
):
  ptr_type = (
      ptr.type.element_ty.element_ty
      if ptr.type.is_block()
      else ptr.type.element_ty
  )
  value_type = value.type.element_ty if value.type.is_block() else value.type
  assert ptr_type == value_type, (ptr_type, value_type)
  ref_block_info, *_ = ctx.block_infos
  idx, *mask_other = tree_util.tree_unflatten(args_tree, args)
  avals_in = ctx.avals_in
  idx_avals, *_ = tree_util.tree_unflatten(args_tree, avals_in[2:])
  ptr = _compute_pointers_from_indices(
      ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder
  )
  mask = None
  if masked:
    assert len(mask_other) == 1
    (mask,) = mask_other
  return tl.store(ptr, value, mask=mask, _builder=ctx.builder)


triton_lowering_rules[primitives.swap_p] = _masked_swap_lowering_rule


def _addupdate_lowering_rule(
    ctx: TritonLoweringRuleContext, ptr, value, *non_slice_idx, indexed_dims
):
  ref_block_info, *_ = ctx.block_infos
  avals_in = ctx.avals_in
  idx = _pack_indices(non_slice_idx, indexed_dims)
  if non_slice_idx:
    (int_indexer_shape,) = {
        tuple(map(lambda x: x.value, i.shape)) for i in non_slice_idx
    }
  else:
    int_indexer_shape = ()
  idx = tuple(
      primitives.Slice.from_slice(slc, s) if isinstance(slc, slice) else slc
      for s, slc in zip(avals_in[0].shape, idx)
  )
  idx = primitives.NDIndexer(idx, avals_in[0].shape, int_indexer_shape)
  ptr = _compute_pointers_from_indices(
      ptr, ref_block_info, idx, avals_in[0].shape, ctx.builder
  )
  tl.atomic_add(ptr, value, _builder=ctx.builder)
  return []


triton_lowering_rules[sp.addupdate_p] = _addupdate_lowering_rule


def _transpose_lowering(ctx: TritonLoweringRuleContext, a, *, permutation):
  if permutation != (1, 0):
    raise NotImplementedError(permutation)
  return tl.trans(a, _builder=ctx.builder)


triton_lowering_rules[lax.transpose_p] = _transpose_lowering


def _dot_general_lowering(
    ctx: TritonLoweringRuleContext,
    a,
    b,
    *,
    dimension_numbers,
    precision,
    preferred_element_type
):
  contract_dims, batch_dims = dimension_numbers
  assert batch_dims == ((), ())
  (a_contract_dim,) = contract_dims[0]
  (b_contract_dim,) = contract_dims[1]
  trans_a = a_contract_dim == 0
  trans_b = b_contract_dim == 1
  if trans_a:
    a = tl.trans(a, _builder=ctx.builder)
  if trans_b:
    b = tl.trans(b, _builder=ctx.builder)
  allow_tf32 = (
      precision == lax.Precision.HIGH or precision == lax.Precision.DEFAULT
  )
  return tl.dot(a, b, _builder=ctx.builder, allow_tf32=allow_tf32)


triton_lowering_rules[lax.dot_general_p] = _dot_general_lowering


def _reduction_lowering(body, ctx: TritonLoweringRuleContext, args, axes):
  flat_args = tree_util.tree_leaves(args)
  (axis,) = axes
  mapped_avals = [
      jax_core.mapped_aval(aval.shape[axis], axis, aval)
      for aval in ctx.avals_in
  ]
  in_tree = tree_util.tree_structure((args, args))
  flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(body), in_tree
  )
  combine_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      flat_fun, [*mapped_avals, *mapped_avals]
  )
  out_tree = out_tree_thunk()
  del out_tree  # Not needed
  if consts:
    raise NotImplementedError("Reductions with constants not supported.")
  element_types = [arg.type.scalar for arg in flat_args]
  builder = ctx.builder
  reduce_op = builder.create_reduce([t.handle for t in flat_args], axis)
  region = reduce_op.get_region(0)
  old_ip = builder.get_insertion_point()
  param_types = element_types * 2
  ir_param_types = [ty.to_ir(builder) for ty in param_types]
  block = builder.create_block_with_parent(region, ir_param_types)
  combine_args = [
      tl.core.tensor(block.arg(i), ty) for i, ty in enumerate(param_types)
  ]
  results = lower_jaxpr_to_triton_ir(ctx, combine_jaxpr, None, *combine_args)
  handles = [r.handle for r in results]
  builder.create_reduce_ret(*handles)
  builder.restore_insertion_point(old_ip)
  reduce_op.verify()

  def wrap_tensor(x, scalar_ty):
    if ctx.avals_out[0].shape:
      res_ty = tl.block_type(scalar_ty, ctx.avals_out[0].shape)
    else:
      # 0d-tensor -> scalar
      res_ty = scalar_ty
    return tl.core.tensor(x, res_ty)

  results = [
      wrap_tensor(reduce_op.get_result(i), ty)
      for i, ty in enumerate(element_types)
  ]
  return results


def _reduce_lowering(body, ctx: TritonLoweringRuleContext, a, *, axes):
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
    a = a.__neg__(_builder=ctx.builder).__neg__(_builder=ctx.builder)
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
    body, ctx: TritonLoweringRuleContext, a, *, axes, index_dtype
):
  if index_dtype != jnp.int32:
    raise ValueError("`index_type` must be f32.")
  if len(axes) != 1:
    raise ValueError("`pallas` reduce operations only support one reduce axis.")
  (axis,) = axes
  n = ctx.avals_in[0].shape[axis]
  index = tl.arange(0, n, _builder=ctx.builder)
  if len(a.shape) > 1:
    # Broadcast index across the non-reduced axes
    expand_dims_index = [tl.constexpr(None)] * len(a.shape)
    expand_dims_index[axis] = slice(None)
    index = index.__getitem__(expand_dims_index, _builder=ctx.builder)
    index = tl.core.broadcast_to(index, a.shape, _builder=ctx.builder)
  ctx = ctx.replace(
      avals_in=[
          ctx.avals_in[0],
          ctx.avals_in[0].update(dtype=jnp.dtype("int32")),
      ]
  )
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


def _pjit_lowering_rule(ctx: TritonLoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_triton_ir(
      ctx.context, jaxpr.jaxpr, ctx.block_infos, *args
  )


triton_lowering_rules[pjit.pjit_p] = _pjit_lowering_rule


def _closed_call_lowering_rule(
    ctx: TritonLoweringRuleContext, *args, call_jaxpr, **_
):
  jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  if consts:
    raise NotImplementedError
  return lower_jaxpr_to_triton_ir(ctx.context, jaxpr, ctx.block_infos, *args)


triton_lowering_rules[jax_core.closed_call_p] = _closed_call_lowering_rule


def _remat_lowering_rule(ctx: TritonLoweringRuleContext, *args, jaxpr, **_):
  return lower_jaxpr_to_triton_ir(ctx.context, jaxpr, ctx.block_infos, *args)


triton_lowering_rules[ad_checkpoint.remat_p] = _remat_lowering_rule


def _is_read_only(ref_effects) -> bool:
  if len(ref_effects) == 0:
    return True
  if len(ref_effects) > 1:
    # Means we must have a write or accum effect so not read-only
    return False
  (eff,) = ref_effects
  return isinstance(eff, state.ReadEffect)


def _for_lowering_rule(
    ctx: TritonLoweringRuleContext,
    *args,
    jaxpr,
    which_linear,
    nsteps,
    reverse,
    unroll
):
  del which_linear
  if reverse or unroll != 1:
    raise NotImplementedError
  builder = ctx.builder
  lower_bound = builder.get_int32(0)
  upper_bound = builder.get_int32(nsteps)
  step = builder.get_int32(1)
  current_block = builder.get_insertion_block()
  init_args = args
  # Partially discharge state from jaxpr for non-pointers
  should_discharge = [not isinstance(a, AbstractRef) for a in ctx.avals_in]
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
  for_op = builder.create_for_op(
      lower_bound, upper_bound, step, [arg.handle for arg in loop_args]
  )
  loop_block = builder.create_block()
  builder.set_insertion_point_to_start(loop_block)
  loop_index = tl.core.tensor(for_op.get_induction_var(), tl.core.int32)
  # Emit loop body
  for_body_args = [
      tl.core.tensor(for_op.get_body(0).arg(i + 1), arg.type)
      for i, arg in enumerate(loop_args)
  ]
  loop_body_args = merge_lists(is_loop_arg, non_loop_args, for_body_args)
  out_discharged = lower_jaxpr_to_triton_ir(
      ctx.context,
      discharged_jaxpr,
      [None, *ctx.block_infos],
      loop_index,
      *loop_body_args
  )
  all_out = merge_lists(should_discharge, ptrs, out_discharged)
  _, loop_out = partition_list(is_loop_arg, all_out)
  if loop_out:
    builder.create_yield_op([arg.handle for arg in loop_out])
  loop_block.merge_block_before(for_op.get_body(0))
  for_results = [for_op.get_result(i) for i in range(len(loop_args))]
  builder.set_insertion_point_to_end(current_block)
  for_out = [tl.core.tensor(r, a.type) for r, a in zip(for_results, loop_args)]
  return merge_lists(is_loop_arg, non_loop_args, for_out)


triton_lowering_rules[for_loop.for_p] = _for_lowering_rule

def _lower_jaxpr_to_for_loop(ctx: TritonLoweringRuleContext, jaxpr: jax_core.Jaxpr,
                             lower_bound, upper_bound, consts, *args,
                             has_loop_index: bool,
                             step: int = 1,
                             bound_type: tl.dtype = tl.int32):
  if step != 1:
    raise NotImplementedError
  builder = ctx.builder
  if bound_type == tl.int64:
    step = builder.get_int64(step)
  else:
    step = builder.get_int32(step)
  current_block = builder.get_insertion_block()
  for_op = builder.create_for_op(
      lower_bound, upper_bound, step, [arg.handle for arg in args]
  )
  loop_block = builder.create_block()
  builder.set_insertion_point_to_start(loop_block)
  loop_index = tl.core.tensor(for_op.get_induction_var(), tl.core.int32)
  # Emit loop body
  for_body_args = [
      tl.core.tensor(for_op.get_body(0).arg(i + 1), arg.type)
      for i, arg in enumerate(args)
  ]
  if has_loop_index:
    jaxpr_args = [*consts, loop_index, *for_body_args]
  else:
    jaxpr_args = [*consts, *for_body_args]
  all_out = lower_jaxpr_to_triton_ir(
      ctx.context,
      jaxpr,
      ctx.block_infos,
      *jaxpr_args)
  if all_out:
    builder.create_yield_op([arg.handle for arg in all_out])
  loop_block.merge_block_before(for_op.get_body(0))
  for_results = [for_op.get_result(i) for i in range(len(args))]
  builder.set_insertion_point_to_end(current_block)
  return [tl.core.tensor(r, a.type) for r, a in zip(for_results, args)]


def _scan_lowering_rule(
    ctx: TritonLoweringRuleContext,
    *args,
    jaxpr,
    linear,
    length,
    reverse,
    unroll,
    num_consts,
    num_carry,
):
  # Only implements fori_loop-like scans
  num_extensive = len(args) - num_consts - num_carry
  if num_extensive: raise NotImplementedError
  if reverse: raise NotImplementedError
  if unroll != 1: raise NotImplementedError
  del linear, num_extensive, unroll, reverse

  builder = ctx.builder
  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts: raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = (
      pallas_utils.pattern_match_scan_to_fori_loop(jaxpr, num_consts, num_carry)
      )
  consts, args = util.split_list(args, [num_consts])
  if has_loop_index:
    lb, *args = args
    lower_bound = lb.handle
    ub = lb.__add__(tl.constexpr(length), _builder=builder)
    upper_bound = ub.handle
    bound_type = ub.type
  else:
    lower_bound = builder.get_int32(0)
    upper_bound = builder.get_int32(length)
    bound_type = tl.int32
  for_out = _lower_jaxpr_to_for_loop(
      ctx, jaxpr, lower_bound, upper_bound, consts, *args,
      has_loop_index=has_loop_index, step=1, bound_type=bound_type)
  if has_loop_index:
    # Need to return the final loop index value if the outer scan expects
    # it as an output
    return [tl.core.tensor(upper_bound, bound_type), *for_out]
  return for_out

triton_lowering_rules[lax.scan_p] = _scan_lowering_rule

def _maybe_pattern_match_fori_loop(ctx: TritonLoweringRuleContext, *args,
                                   cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr
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
  new_invars = tuple((*jaxpr.invars[:body_nconsts],
                      jaxpr.invars[body_nconsts],
                      *jaxpr.invars[body_nconsts + 2:]))
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
  for_out = _lower_jaxpr_to_for_loop(ctx, jaxpr, lb.handle, ub.handle,
                                     body_consts, *args, has_loop_index=True,
                                     step=1, bound_type=lb.type)
  return [ub, ub, *for_out]

def _while_lowering_rule(
    ctx: TritonLoweringRuleContext,
    *args,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr
):
  # First, try to pattern match to fori_loop and lower to scf.for if possible
  result = _maybe_pattern_match_fori_loop(ctx, *args, cond_nconsts=cond_nconsts,
                                          body_nconsts=body_nconsts, cond_jaxpr=cond_jaxpr,
                                          body_jaxpr=body_jaxpr)
  if result is not None:
    return result
  # Fall back to default while lowering
  num_args = len(args)
  cond_consts, body_consts, carry = util.split_list(
      args, [cond_nconsts, body_nconsts]
  )
  cond_const_block_infos, body_const_block_infos, carry_block_infos = (
      util.split_list(ctx.block_infos, [cond_nconsts, body_nconsts])
  )
  current_bb = ctx.builder.get_insertion_block()
  cond_const_types = [a.type.to_ir(ctx.builder) for a in cond_consts]
  body_const_types = [a.type.to_ir(ctx.builder) for a in body_consts]
  carry_types = [a.type.to_ir(ctx.builder) for a in carry]
  all_types = [*cond_const_types, *body_const_types, *carry_types]
  while_op = ctx.builder.create_while_op(
      [*cond_const_types, *body_const_types, *carry_types],
      [arg.handle for arg in args],
  )
  before_block = ctx.builder.create_block_with_parent(
      while_op.get_before(), all_types
  )
  ctx.builder.set_insertion_point_to_start(before_block)
  cond_consts_, _, carry_ = util.split_list(
      [before_block.arg(i) for i in range(num_args)],
      [cond_nconsts, body_nconsts],
  )
  cond_args = [
      tl.core.tensor(a, b.type)
      for a, b in zip([*cond_consts_, *carry_], [*cond_consts, *carry])
  ]
  (cond,) = lower_jaxpr_to_triton_ir(
      ctx.context,
      cond_jaxpr.jaxpr,
      [*cond_const_block_infos, *carry_block_infos],
      *cond_args
  )
  ctx.builder.create_condition_op(
      cond.handle, [before_block.arg(i) for i in range(num_args)]
  )
  after_block = ctx.builder.create_block_with_parent(
      while_op.get_after(), all_types
  )
  ctx.builder.set_insertion_point_to_start(after_block)
  cond_consts_, body_consts_, carry_ = util.split_list(
      [after_block.arg(i) for i in range(num_args)],
      [cond_nconsts, body_nconsts],
  )
  all_args = [
      tl.core.tensor(a, b.type)
      for a, b in zip(
          [*cond_consts_, *body_consts_, *carry_],
          [*cond_consts, *body_consts, *carry],
      )
  ]
  cond_const_args, body_const_args, carry_args = util.split_list(
      all_args, [cond_nconsts, body_nconsts]
  )
  loop_out = lower_jaxpr_to_triton_ir(
      ctx.context,
      body_jaxpr.jaxpr,
      [*body_const_block_infos, *carry_block_infos],
      *body_const_args,
      *carry_args
  )
  cond_consts_handles = [a.handle for a in cond_const_args]
  body_consts_handles = [a.handle for a in body_const_args]
  loop_out_handles = [a.handle for a in loop_out]
  all_handles = [*cond_consts_handles, *body_consts_handles, *loop_out_handles]
  if all_handles:
    ctx.builder.create_yield_op(all_handles)
  ctx.builder.set_insertion_point_to_end(current_bb)
  all_out = [
      tl.core.tensor(while_op.get_result(i), a.type) for i, a in enumerate(args)
  ]
  return all_out[cond_nconsts + body_nconsts :]


triton_lowering_rules[lax.while_p] = _while_lowering_rule


def _cond_lowering_rule(
    ctx: TritonLoweringRuleContext,
    index,
    *args,  # *consts, *ops
    branches,  # tuple(jaxprs)
    linear,
):
  block_infos = ctx.block_infos
  current_bb = ctx.builder.get_insertion_block()

  def to_type(out_aval):
    elt_type = code_gen.str_to_ty(get_triton_type(out_aval)).element_ty
    if not out_aval.shape:
      # Scalar types get special handling.
      return elt_type
    return tl.block_type(elt_type, out_aval.shape)

  out_types = [to_type(out) for out in ctx.avals_out]
  out_ir_types = [t.to_ir(ctx.builder) for t in out_types]

  use_branch0 = index.__eq__(0, _builder=ctx.builder)
  # TODO(bjp): Switch to scf.index_switch once exposed in triton.cc
  if_op = ctx.builder.create_if_op(
      out_ir_types,  # retTypes
      use_branch0.handle,  # condition
      True)  # withElse
  # Lower then block.
  ctx.builder.set_insertion_point_to_start(if_op.get_then_block())
  outs0 = lower_jaxpr_to_triton_ir(
      ctx.context,
      branches[0].jaxpr,
      block_infos[1:],
      *args)
  if outs0:
    ctx.builder.create_yield_op([out.handle for out in outs0])
  # Lower else block.
  ctx.builder.set_insertion_point_to_start(if_op.get_else_block())
  # TODO(bjp): Instead of linear nest of 'if's, partition into halves.
  if len(branches) > 2:
    outs1 = _cond_lowering_rule(
        ctx,
        index.__sub__(1, _builder=ctx.builder),
        *args,
        branches=branches[1:],
        linear=linear)
  else:
    outs1 = lower_jaxpr_to_triton_ir(
        ctx.context,
        branches[1].jaxpr,
        block_infos[1:],
        *args)
  if outs1:
    ctx.builder.create_yield_op([out.handle for out in outs1])
  ctx.builder.set_insertion_point_to_end(current_bb)
  all_out = [
      tl.core.tensor(if_op.get_result(i), ty)
      for i, ty in enumerate(out_types)
  ]
  return all_out

triton_lowering_rules[lax.cond_p] = _cond_lowering_rule


@weakref_lru_cache
def compile_jaxpr(
    jaxpr: jax_core.Jaxpr,
    in_shapes,
    grid_mapping: GridMapping,
    name: str,
    num_warps: int,
    num_stages: int,
    debug: bool,
) -> TritonCompilationResult:
  lowering_result = lower_jaxpr_to_triton_module(
      jaxpr, in_shapes, grid_mapping, name
  )
  device = 0
  ttir = str(lowering_result.module)
  ptx, name, shared_mem_bytes, compute_capability = compile_ttir_to_ptx_inplace(
      lowering_result.module,
      device=device,
      num_warps=num_warps,
      num_stages=num_stages,
      dump=debug,
  )
  return TritonCompilationResult(
      name, ttir, ptx, shared_mem_bytes, compute_capability, lowering_result
  )


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name: str,
    in_shapes: Tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: Tuple[jax.ShapeDtypeStruct, ...],
    which_linear: Tuple[bool, ...],
    interpret: bool,
    debug: bool,
    input_output_aliases: Tuple[Tuple[int, int], ...],
    grid_mapping: GridMapping,
    triton_params: Dict[str, Any] | None = None,
    **compiler_params: Any,
):
  if interpret:
    return mlir.lower_fun(pallas_call_p.impl, multiple_results=True)(
        ctx,
        *in_nodes,
        jaxpr=jaxpr,
        name=name,
        out_shapes=out_shapes,
        in_shapes=in_shapes,
        which_linear=which_linear,
        interpret=interpret,
        debug=debug,
        input_output_aliases=input_output_aliases,
        grid_mapping=grid_mapping,
        **compiler_params
    )
  num_warps = compiler_params.get("num_warps", 4)
  num_stages = compiler_params.get("num_stages", 3)
  if debug:
    print(jaxpr)
    print(grid_mapping)
  compilation_result = compile_jaxpr(
      jaxpr,
      tuple((*in_shapes, *out_shapes)),
      grid_mapping,
      name,
      num_warps,
      num_stages,
      debug=debug,
  )

  if debug:
    compilation_result.lowering_result.module.dump()

  kernel = triton_kernel_call_lib.TritonKernel(
      compilation_result.kernel_name,
      num_warps,
      compilation_result.shared_mem_bytes,
      compilation_result.ptx,
      compilation_result.ttir,
      compilation_result.compute_capability,
  )

  grid = triton_lib.normalize_grid(
      compilation_result.lowering_result.grid, metaparams={}
  )

  kernel_params = []
  for _ in range(len(in_shapes) + len(out_shapes)):
    kernel_params.append(
        triton_kernel_call_lib.create_array_parameter(
            0,  # bytes to zero  # TODO(cjfj): Expose through user API.
            16,  # divisible by 16
        )
    )

  kernel_call = triton_kernel_call_lib.TritonKernelCall(
      kernel, grid[0], grid[1], grid[2], kernel_params
  )

  out_types = [
      ir.RankedTensorType.get(shape.shape, mlir.dtype_to_ir_type(shape.dtype))
      for shape in out_shapes
  ]

  if triton_params is None:
    triton_params = {}
  serialized_metadata = triton_params.get("serialized_metadata", b"")
  if version >= (0, 4, 15):
    kernel_call_proto = kernel_call.to_proto(name, serialized_metadata)
  else:
    kernel_call_proto = kernel_call.to_proto(serialized_metadata)
  return hlo_helpers.custom_call(
      call_target_name="triton_kernel_call",
      result_types=out_types,
      operands=in_nodes,
      backend_config=zlib.compress(kernel_call_proto),
      operand_layouts=triton_lib.avals_to_layouts(ctx.avals_in),
      result_layouts=triton_lib.avals_to_layouts(ctx.avals_out),
      operand_output_aliases=dict(input_output_aliases),
  ).results


mlir.register_lowering(pallas_call_p, pallas_call_lowering, platform="cuda")
