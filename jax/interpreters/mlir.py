# Copyright 2021 Google LLC
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

# Lowering and execution path that converts jaxprs into the MLIR MHLO/CHLO
# dialects.

import collections
import dataclasses
import functools
from functools import partial
import io
import itertools
import re
import typing
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Tuple,
                    Type, Union, FrozenSet)
from typing_extensions import Protocol
import warnings

from jax import core
from jax import linear_util as lu
from jax._src import ad_util
from jax._src import device_array
from jax._src import dtypes
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import builtin
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib import xla_client as xc
from jax._src import source_info_util
import jax._src.util as util
import jax.interpreters.ad as ad
import jax.interpreters.partial_eval as pe
import jax.interpreters.xla as xla
import numpy as np


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

T = typing.TypeVar("T")

# mypy implicitly sets this variable to true when type checking.
MYPY = False


# IR Helpers

def dense_int_elements(xs) -> ir.DenseIntElementsAttr:
  return ir.DenseIntElementsAttr.get(np.asarray(xs, np.int64))

def dense_bool_elements(xs: Sequence[bool]) -> ir.DenseElementsAttr:
  a = np.packbits(np.array(xs, np.bool_), bitorder='little')
  # TODO(b/209005197): Work around for MLIR crash for non-splat single element
  # buffers.
  if len(xs) == 1:
    a = np.array(0 if a.item() == 0 else 0xff, np.uint8)
  return ir.DenseElementsAttr.get(
      a, type=ir.IntegerType.get_signless(1), shape=[len(xs)])

def i32_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)
def i64_attr(i): return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)


# IR Types

# Non-canonicalized dtype to IR type mapping.
_dtype_to_ir_type : Dict[np.dtype, Callable[[], ir.Type]] = {
  np.dtype(dtypes.float0): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.bool_): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.int8): partial(ir.IntegerType.get_signless, 8),
  np.dtype(np.int16): partial(ir.IntegerType.get_signless, 16),
  np.dtype(np.int32): partial(ir.IntegerType.get_signless, 32),
  np.dtype(np.int64): partial(ir.IntegerType.get_signless, 64),
  np.dtype(np.uint8): partial(ir.IntegerType.get_unsigned, 8),
  np.dtype(np.uint16): partial(ir.IntegerType.get_unsigned, 16),
  np.dtype(np.uint32): partial(ir.IntegerType.get_unsigned, 32),
  np.dtype(np.uint64): partial(ir.IntegerType.get_unsigned, 64),
  np.dtype(dtypes.bfloat16): ir.BF16Type.get,
  np.dtype(np.float16): ir.F16Type.get,
  np.dtype(np.float32): ir.F32Type.get,
  np.dtype(np.float64): ir.F64Type.get,
  np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
  np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
}

def dtype_to_ir_type(dtype: Union[np.dtype, np.generic]) -> ir.Type:
  assert isinstance(dtype, (np.dtype, np.generic)), type(dtype)
  dtype = np.dtype(dtype)
  try:
    ir_type_factory = _dtype_to_ir_type[dtype]
  except KeyError as err:
    raise TypeError(
        f"No dtype_to_ir_type handler for dtype: {dtype}") from err
  return ir_type_factory()

def _array_ir_types(aval: core.ShapedArray) -> ir.Type:
  return (ir.RankedTensorType.get(aval.shape, dtype_to_ir_type(aval.dtype)),)

ir_type_handlers: Dict[Type[core.AbstractValue],
                        Callable[[Any], Sequence[ir.Type]]] = {}

def aval_to_ir_types(aval: core.AbstractValue) -> Sequence[ir.Type]:
  """Converts a JAX aval to zero or more MLIR IR types.

  In general, a JAX value may be represented by multiple IR values, so this
  function returns multiple types."""
  try:
    return ir_type_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No ir_type_handler for aval type: {type(aval)}") from err

ir_type_handlers[core.AbstractUnit] = lambda _: ()
ir_type_handlers[core.ShapedArray] = _array_ir_types
ir_type_handlers[core.ConcreteArray] = _array_ir_types
ir_type_handlers[core.AbstractToken] = lambda _: [mhlo.TokenType.get()]

def aval_to_ir_type(aval: core.AbstractValue) -> ir.Type:
  """Convenience wrapper around aval_to_ir_types for single types.

  For some common cases, e.g. dense arrays, we know JAX values are represented
  by a single IR value."""
  types = aval_to_ir_types(aval)
  if len(types) != 1:
    raise TypeError(f"aval_to_ir_type called on {aval} which corresponds to "
                    f"multiple IR types {types}")
  return types[0]


# Constants

class ConstantHandler(Protocol):
  def __call__(self, val: Any, canonicalize_types: bool) -> Sequence[ir.Value]:
    """Builds an IR representation for a constant `val`.

    A JAX value is represented by zero or more IR values."""

_constant_handlers : Dict[type, ConstantHandler] = {}

def register_constant_handler(type_: type, handler_fun: ConstantHandler):
  _constant_handlers[type_] = handler_fun

def ir_constants(val: Any,
                 canonicalize_types: bool = True) -> Sequence[ir.Value]:
  """Translate a Python `val` to an IR constant, canonicalizing its dtype.

  Args:
    val: a Python value to be translated to a constant.

  Returns:
    A representation of the constant as a list of IR values.
  """
  for t in type(val).__mro__:
    handler = _constant_handlers.get(t)
    if handler:
      out = handler(val, canonicalize_types)
      assert all(isinstance(v, ir.Value) for v in out), (type(val), out)
      return out
  if hasattr(val, '__jax_array__'):
    return ir_constants(val.__jax_array__(), canonicalize_types)
  raise TypeError("No constant handler for type: {}".format(type(val)))

def ir_constant(val: Any, canonicalize_types: bool = True) -> ir.Value:
  """Convenience wrapper around ir_constants for singleton values."""
  values = ir_constants(val)
  if len(values) != 1:
    raise TypeError(f"ir_constant called on {val} which corresponds to "
                    f"multiple IR values {values}")
  return values[0]

register_constant_handler(core.Unit, lambda val, canonicalize_types: ())

def _numpy_array_constant(x: np.ndarray, canonicalize_types
                         ) -> Sequence[ir.Value]:
  if canonicalize_types:
    x = np.asarray(x, dtypes.canonicalize_dtype(x.dtype))
  ir_type = ir.RankedTensorType.get(x.shape, dtype_to_ir_type(x.dtype))
  shape = x.shape
  if x.dtype == np.bool_:
    nelems = x.size
    x = np.packbits(x, bitorder='little')
    # TODO(b/209005197): Work around for MLIR crash for non-splat single element
    # buffers.
    if nelems == 1:
      x = np.array(0 if x.item() == 0 else 0xff, np.uint8)
  elif x.dtype == dtypes.bfloat16:
    x = x.view(np.uint16)
  x = np.ascontiguousarray(x)
  attr = ir.DenseElementsAttr.get(x, type=ir_type.element_type, shape=shape)
  return (mhlo.ConstOp(ir_type, attr).result,)


def _ndarray_constant_handler(val: np.ndarray, canonicalize_types
                             ) -> Sequence[ir.Value]:
  """Constant handler for ndarray literals, handling zero-size strides.

  In most cases this function calls _numpy_array_constant(val) except it has
  special handling of arrays with any strides of size zero: for those, it
  generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
  to avoid staging in large literals that might arise from np.zeros or np.ones
  or the output of lax.broadcast (which uses np.broadcast_to which in turn
  uses size-zero strides).

  Args:
    val: an ndarray.

  Returns:
    An XLA ComputationDataHandle / XlaOp representing the constant ndarray
    staged into the XLA Computation.
  """
  if dtypes.result_type(val) == dtypes.float0:
    return _numpy_array_constant(np.zeros(val.shape, dtype=np.bool_),
                                 canonicalize_types=False)
  elif np.any(np.equal(0, val.strides)) and val.size > 0:
    zero_stride_axes, = np.where(np.equal(0, val.strides))
    other_axes, = np.where(np.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)
                              for ax in range(val.ndim))]  # type: ignore
    if canonicalize_types:
      collapsed_val = np.asarray(
          collapsed_val, dtypes.canonicalize_dtype(collapsed_val.dtype))
    out = mhlo.BroadcastInDimOp(
        ir.RankedTensorType.get(
            val.shape, dtype_to_ir_type(collapsed_val.dtype)),
        _numpy_array_constant(collapsed_val, canonicalize_types=False)[0],
        dense_int_elements(other_axes)).result
    return (out,)
  else:
    return _numpy_array_constant(val, canonicalize_types)

register_constant_handler(np.ndarray, _ndarray_constant_handler)

for _scalar_type in [np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64,
                     np.float16, np.float32, np.float64,
                     np.complex64, np.complex128,
                     np.bool_, np.longlong, dtypes.bfloat16]:
  register_constant_handler(_scalar_type, _ndarray_constant_handler)

def _python_scalar_handler(dtype, val, canonicalize_dtypes):
  return _numpy_array_constant(np.array(val, dtype), canonicalize_dtypes)

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_constant_handler(ptype, partial(_python_scalar_handler, dtype))

def _device_array_constant_handler(val, canonicalize_types):
  return _ndarray_constant_handler(val.device_buffer.to_py(),
                                   canonicalize_types)
for t in device_array.device_array_types:
  register_constant_handler(t, _device_array_constant_handler)

register_constant_handler(
  core.Token, lambda _, __: [mhlo.CreateTokenOp(mhlo.TokenType.get()).result])

# Source locations

def _source_info_to_location(
    primitive: core.Primitive, params: Dict,
    source_info: source_info_util.SourceInfo,
    name_stack: Union[str, source_info_util.NameStack] = "") -> ir.Location:
  eqn_str = str(name_stack) + core.str_eqn_compact(primitive.name, params)
  frame = source_info_util.user_frame(source_info)
  if frame is None:
    loc = ir.Location.unknown()
  else:
    loc = ir.Location.file(xla._get_canonical_source_file(frame),
                           frame.line_num, 1)
  loc = ir.Location.name(eqn_str, childLoc=loc)
  # TODO(phawkins): also include primitive.name as the operator type.
  return loc


# Translation rules
NameStack = Union[str, source_info_util.NameStack]

def make_ir_context() -> ir.Context:
  """Creates an MLIR context suitable for JAX IR."""
  context = ir.Context()
  mhlo.register_mhlo_dialect(context)
  chlo.register_chlo_dialect(context)
  return context


Mesh = Any
MeshAxisName = Any

@dataclasses.dataclass(frozen=True)
class SPMDAxisContext:
  """A hardware axis context for parallel computations that use the GSPMD partitioner.

  This includes the mesh that will later by used to execute this computation,
  as well as a set of mesh axes that are currently (e.g. because the current lowering
  is invoked inside an xmap) lowered in the MANUAL sharding mode.
  """
  mesh: Mesh
  manual_axes: FrozenSet[MeshAxisName] = frozenset()

  @property
  def axis_env(self):
    # TODO: Should we restrict the mesh to manual axes? This depends on the
    # semantics of ReplicaId in partially manual computations, but I don't
    # know what they are...
    return xla.AxisEnv(nreps=1, names=(), sizes=())

  def extend_manual(self, axes: FrozenSet[MeshAxisName]) -> 'SPMDAxisContext':
    return SPMDAxisContext(self.mesh, self.manual_axes | axes)


@dataclasses.dataclass(frozen=True)
class ReplicaAxisContext:
  """A hardware axis context for parallel computations that are partitioned by JAX.

  Unlike in the SPMDAxisContext, this means that JAX might need to emit calls to
  explicit collectives.
  """
  axis_env: xla.AxisEnv

AxisContext = Union[SPMDAxisContext, ReplicaAxisContext]

@dataclasses.dataclass
class ModuleContext:
  """Module-wide context information for MLIR lowering."""
  context: ir.Context
  module: ir.Module
  ip: ir.InsertionPoint
  symbol_table: ir.SymbolTable
  platform: str
  axis_context: AxisContext
  name_stack: NameStack

  # Cached primitive lowerings.
  cached_primitive_lowerings: Dict[Any, builtin.FuncOp]

  @property
  def axis_env(self) -> xla.AxisEnv:
    return self.axis_context.axis_env

  def __init__(
      self, platform: str, axis_context: AxisContext, name_stack: NameStack,
      context: Optional[ir.Context] = None,
      module: Optional[ir.Module] = None,
      ip: Optional[ir.InsertionPoint] = None,
      symbol_table: Optional[ir.SymbolTable] = None,
      cached_primitive_lowerings: Optional[Dict[Any, builtin.FuncOp]] = None):
    assert platform is not None
    self.context = context or make_ir_context()
    self.module = module or ir.Module.create(loc=ir.Location.unknown(self.context))
    self.ip = ip or ir.InsertionPoint(self.module.operation.opview.body)
    self.symbol_table = symbol_table or ir.SymbolTable(self.module.operation)
    self.platform = platform
    self.axis_context = axis_context
    self.name_stack = name_stack
    self.cached_primitive_lowerings = ({} if cached_primitive_lowerings is None
                                       else cached_primitive_lowerings)

  def replace(self, **kw): return dataclasses.replace(self, **kw)


@dataclasses.dataclass
class LoweringRuleContext:
  """Per-rule context information for MLIR lowering."""
  module_context: ModuleContext
  primitive: Optional[core.Primitive]
  avals_in: Sequence[core.AbstractValue]
  avals_out: Any  # Usually Sequence[core.AbstractValue], but sometimes None.

  def replace(self, **kw): return dataclasses.replace(self, **kw)


if not MYPY:
  class LoweringRule(Protocol):
    def __call__(self, ctx: LoweringRuleContext,
                 *args: Union[ir.Value, Sequence[ir.Value]],
                 **kw) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
      """Converts a JAX primitive invocation into MLIR."""
else:
  LoweringRule = Any

_lowerings: Dict[core.Primitive, LoweringRule] = {}
_platform_specific_lowerings: Dict[str, Dict[core.Primitive, LoweringRule]]
_platform_specific_lowerings = collections.defaultdict(dict)

def register_lowering(prim: core.Primitive, rule: LoweringRule,
                      platform: Optional[str] = None):
  ts = (_lowerings if platform is None
        else _platform_specific_lowerings[platform])
  ts[prim] = rule


def _unwrap_singleton_ir_values(x): return x[0] if len(x) == 1 else x
def wrap_singleton_ir_values(x: Union[ir.Value, Sequence[ir.Value]]
                             ) -> Sequence[ir.Value]:
  """Adds a consistent tuples to a mixture of tupled and untuple values."""
  return (x,) if isinstance(x, ir.Value) else tuple(x)

def flatten_lowering_ir_args(
    xs: Sequence[Union[ir.Value, Sequence[ir.Value]]]
) -> Sequence[Sequence[ir.Value]]:
  return util.flatten(map(wrap_singleton_ir_values, xs))

_module_unique_id = itertools.count()
_module_name_regex = re.compile(r"[^\w.-]")

def sharded_aval(aval: core.ShapedArray,
                 sharding: Optional[xc.OpSharding]) -> core.ShapedArray:
  """Returns the new aval sharded based on sharding proto."""
  if sharding is None:
    return aval

  if (sharding.type == xc.OpSharding.Type.REPLICATED or
      sharding.type == xc.OpSharding.Type.MANUAL):
    return aval

  sharded_shape = []
  tile_rank = len(sharding.tile_assignment_dimensions)
  if sharding.replicate_on_last_tile_dim:
    tile_rank -= 1
  if sharding.last_tile_dims:
    tile_rank -= len(sharding.last_tile_dims)
  if tile_rank == 0:
    return aval

  for i in range(tile_rank):
    partitions = sharding.tile_assignment_dimensions[i]
    assert partitions > 0
    sharded_shape.append((aval.shape[i] + partitions - 1) // partitions)
  return aval.update(tuple(sharded_shape))

def lower_jaxpr_to_module(
    module_name: str, jaxpr: core.ClosedJaxpr, platform: str,
    axis_context: AxisContext,
    name_stack: NameStack, donated_args: Sequence[bool],
    replicated_args: Optional[Sequence[bool]] = None,
    arg_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    result_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None
    ) -> ir.Module:
  """Lowers a top-level jaxpr to an MHLO module.

  Handles the quirks of the argument/return value passing conventions of the
  runtime."""
  input_output_aliases = None
  in_avals = jaxpr.in_avals
  if arg_shardings is not None:
    in_avals = [
        sharded_aval(in_aval, in_sharding)
        for in_aval, in_sharding in zip(in_avals, arg_shardings)
    ]
  out_avals = jaxpr.out_avals
  if result_shardings is not None:
    out_avals = [
        sharded_aval(out_aval, out_sharding)
        for out_aval, out_sharding in zip(out_avals, result_shardings)
    ]
  platforms_with_donation = ("gpu", "tpu")
  if platform in platforms_with_donation:
    input_output_aliases, donated_args = _set_up_aliases(
        in_avals, out_avals, donated_args)
  if any(donated_args):
    # TODO(tomhennigan): At call time we should mark these buffers as deleted.
    unused_donations = [str(a) for a, d in zip(in_avals, donated_args)
                        if d]
    msg = "See an explanation at https://jax.readthedocs.io/en/latest/notebooks/faq.html#buffer-donation."
    if platform not in platforms_with_donation:
      msg = f"Donation is not implemented for {platform}.\n{msg}"
    warnings.warn(f"Some donated buffers were not usable: {', '.join(unused_donations)}.\n{msg}")

  ctx = ModuleContext(platform, axis_context, name_stack)
  with ctx.context, ir.Location.unknown(ctx.context):
    # Remove module name characters that XLA would alter. This ensures that
    # XLA computation preserves the module name.
    module_name = _module_name_regex.sub("_", module_name)
    # Some clients expect modules to have unique names, e.g., in trace data.
    # This may or may not be a reasonable assumption.
    ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get(
        f"{module_name}.{next(_module_unique_id)}")
    # TODO(phawkins): represent units with zero buffers at the runtime level.
    lower_jaxpr_to_fun(
        ctx, "main", jaxpr, public=True, replace_units_with_dummy=True,
        replace_tokens_with_dummy=True, replicated_args=replicated_args,
        arg_shardings=arg_shardings, result_shardings=result_shardings,
        input_output_aliases=input_output_aliases)

  ctx.module.operation.verify()
  return ctx.module

def module_to_string(module: ir.Module) -> str:
  output = io.StringIO()
  module.operation.print(file=output, enable_debug_info=True,
                         print_generic_op_form=False)
  return output.getvalue()

def _set_up_aliases(avals_in, avals_out, donated_args):
  input_output_aliases = [None] * len(avals_in)
  # To match-up in-avals to out-avals we only care about the number of
  # bytes, so we strip off unrelated aval metadata (eg. the named shape)
  strip_metadata = lambda a: a.strip_named_shape().strip_weak_type()
  avals_in = map(strip_metadata, avals_in)
  avals_out = map(strip_metadata, avals_out)

  donations = collections.defaultdict(collections.deque)
  for i, (aval, donated) in enumerate(zip(avals_in, donated_args)):
    if donated:
      donations[aval].append(i)

  out_donated_args = list(donated_args)
  for i, aval in enumerate(avals_out):
    if donations.get(aval, ()):
      input_id = donations[aval].popleft()
      input_output_aliases[input_id] = i
      out_donated_args[input_id] = False

  return input_output_aliases, out_donated_args

def lower_jaxpr_to_fun(
    ctx: ModuleContext, name: str, jaxpr: core.ClosedJaxpr, *,
    public: bool = False, replace_units_with_dummy: bool = False,
    replace_tokens_with_dummy: bool = False,
    replicated_args: Optional[Sequence[bool]] = None,
    arg_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    result_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    input_output_aliases: Optional[Sequence[Optional[int]]] = None
  ) -> builtin.FuncOp:
  """Lowers jaxpr and its callees to an IR function.

  Assumes that an MLIR context, location, and insertion point are set.

  Args:
    ctx: the lowering context.
    name: the function name. The name will be uniquified by the symbol table,
      so it is ok to use the same name multiple times.
    jaxpr: the jaxpr to lower.
    public: if true, the function's visibility is set to "public".
    replace_units_with_dummy: if true, unit arguments/return values are
      replaced with bool arrays of size [0].
    replace_tokens_with_dummy: if true, token arguments/return values are
      replaced with bool arrays of size [0].
    replicated_args: if present, annotates arguments as replicated.
    arg_shardings: sharding annotations for each argument (optional).
    result_shardings: sharding annotations for each argument (optional).
    input_output_aliases: optional sequence that maps argument numbers to the
      corresponding output that should alias them.
  Returns the name of the function.
  """
  def aval_to_types(aval):
    if replace_units_with_dummy and aval is core.abstract_unit:
      aval = core.ShapedArray((), np.dtype(np.bool_))
    elif replace_tokens_with_dummy and aval is core.abstract_token:
      aval = core.ShapedArray((), np.dtype(np.bool_))
    return aval_to_ir_types(aval)

  input_types = map(aval_to_types, jaxpr.in_avals)
  output_types = map(aval_to_types, jaxpr.out_avals)
  flat_input_types = util.flatten(input_types)
  flat_output_types = util.flatten(output_types)
  ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
  func_op = builtin.FuncOp(name, ftype, ip=ctx.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get(
      "public" if public else "private")
  ctx.symbol_table.insert(func_op)
  if (replicated_args is not None or arg_shardings is not None
      or input_output_aliases is not None):
    arg_attrs: List[Dict[str, ir.Attribute]] = [
        {} for _ in range(len(flat_input_types))]

    if replicated_args is not None:
      replicated_ir_args = [[replicated] * len(types) for replicated, types
                            in zip(replicated_args, input_types)]
      for attrs, replicated in zip(arg_attrs, util.flatten(replicated_ir_args)):
        if replicated:
          attrs["mhlo.is_same_data_across_replicas"] = ir.UnitAttr.get()

    if arg_shardings is not None:
      ir_arg_shardings = [[sharding] * len(types) for sharding, types
                          in zip(arg_shardings, input_types)]
      for attrs, sharding in zip(arg_attrs, util.flatten(ir_arg_shardings)):
        if sharding is not None:
          attrs["mhlo.sharding"] = ir.StringAttr.get(
              sharding.SerializeToString())

    if input_output_aliases is not None:
      output_ids = util.unflatten(list(range(len(flat_output_types))),
                                  map(len, output_types))
      aliases: List[Optional[int]] = []
      for types, alias in zip(input_types, input_output_aliases):
        if alias is None:
          aliases.extend([None] * len(types))
        else:
          aliases.extend(output_ids[alias])

      for attrs, alias in zip(arg_attrs, aliases):
        if alias is not None:
          attrs["tf.aliasing_output"] = i32_attr(alias)

    func_op.arg_attrs = ir.ArrayAttr.get(
        [ir.DictAttr.get(attrs) for attrs in arg_attrs])

  if result_shardings is not None:
    ir_result_shardings = util.flatten(
        [[sharding] * len(types) for sharding, types
         in zip(result_shardings, output_types)])
    func_op.result_attrs = ir.ArrayAttr.get([
        ir.DictAttr.get(
            {} if sharding is None else
            {"mhlo.sharding": ir.StringAttr.get(sharding.SerializeToString())}
        ) for sharding in ir_result_shardings
    ])

  entry_block = func_op.add_entry_block()
  with ir.InsertionPoint(entry_block):
    unflattened_args = util.unflatten(entry_block.arguments,
                                      map(len, input_types))
    args: List[List[ir.Value]] = []
    for aval, arg in zip(jaxpr.in_avals, unflattened_args):
      if replace_units_with_dummy and aval is core.abstract_unit:
        args.append([])
      elif replace_tokens_with_dummy and aval is core.abstract_token:
        args.append(mhlo.CreateTokenOp(mhlo.TokenType.get()).results)
      else:
        args.append(arg)
    callee_name_stack = xla.extend_name_stack(ctx.name_stack,
                                              xla.wrap_name(name, 'jit'))
    out_vals = jaxpr_subcomp(ctx.replace(name_stack=callee_name_stack),
                             jaxpr.jaxpr, map(ir_constants, jaxpr.consts),
                             *args)
    outs = []
    for aval, out in zip(jaxpr.out_avals, out_vals):
      if replace_units_with_dummy and aval is core.abstract_unit:
        outs.append(ir_constants(np.zeros((), np.bool_)))
      elif replace_tokens_with_dummy and aval is core.abstract_token:
        outs.append(ir_constants(np.zeros((), np.bool_)))
      else:
        outs.append(out)
    func_dialect.ReturnOp(util.flatten(outs))

  return func_op

def _emit_lowering_rule_as_fun(lowering_rule,
                               ctx: LoweringRuleContext) -> builtin.FuncOp:
  """Emits the contents of a lowering rule as a private function."""
  input_types = map(aval_to_ir_types, ctx.avals_in)
  output_types = map(aval_to_ir_types, ctx.avals_out)
  flat_input_types = util.flatten(input_types)
  flat_output_types = util.flatten(output_types)
  ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
  assert ctx.primitive is not None
  func_op = builtin.FuncOp(ctx.primitive.name, ftype, ip=ctx.module_context.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get("private")
  ctx.module_context.symbol_table.insert(func_op)
  entry_block = func_op.add_entry_block()
  with ir.InsertionPoint(entry_block):
    unflattened_args = util.unflatten(entry_block.arguments,
                                      map(len, input_types))
    outs = lowering_rule(ctx, *_unwrap_singleton_ir_values(unflattened_args))
    func_dialect.ReturnOp(util.flatten(map(wrap_singleton_ir_values, outs)))
  return func_op

def jaxpr_subcomp(ctx: ModuleContext, jaxpr: core.Jaxpr,
                  consts: Sequence[Sequence[ir.Value]],
                  *args: Sequence[ir.Value]) -> Sequence[Sequence[ir.Value]]:
  """Lowers a jaxpr into mHLO, inlined into an existing function.

  Assumes that an MLIR context, location, and insertion point are set.
  """
  def read(v: core.Var) -> Sequence[ir.Value]:
    if type(v) is core.Literal:
      return ir_constants(v.val, canonicalize_types=True)
    else:
      return env[v]

  def aval(v: core.Var) -> core.AbstractValue:
    if type(v) is core.Literal:
      return xla.abstractify(v.val)
    else:
      return v.aval

  def write(v: core.Var, node: Sequence[ir.Value]):
    assert node is not None
    env[v] = tuple(node)


  env: Dict[core.Var, Tuple[ir.Value, ...]] = {}

  assert len(args) == len(jaxpr.invars), (jaxpr, args)
  assert len(consts) == len(jaxpr.constvars), (jaxpr, consts)
  assert all(isinstance(v, ir.Value) for vs in consts for v in vs), consts
  write(core.unitvar, ())
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_nodes = map(read, eqn.invars)
    loc = _source_info_to_location(eqn.primitive, eqn.params, eqn.source_info,
                                   name_stack=ctx.name_stack)
    with source_info_util.user_context(eqn.source_info.traceback), loc:
      if eqn.primitive in _platform_specific_lowerings[ctx.platform]:
        rule = _platform_specific_lowerings[ctx.platform][eqn.primitive]
      elif eqn.primitive in _lowerings:
        rule = _lowerings[eqn.primitive]
      elif (eqn.primitive in xla._translations or
            eqn.primitive in xla._backend_specific_translations[ctx.platform]):
        rule = xla_fallback_lowering(eqn.primitive)
      else:
        raise NotImplementedError(
            f"MLIR translation rule for primitive '{eqn.primitive.name}' not "
            f"found for platform {ctx.platform}")

      rule_ctx = LoweringRuleContext(
          module_context=ctx, primitive=eqn.primitive,
          avals_in=map(aval, eqn.invars), avals_out=map(aval, eqn.outvars))
      ans = rule(rule_ctx, *map(_unwrap_singleton_ir_values, in_nodes),
                 **eqn.params)

    try:
      out_nodes = tuple(map(wrap_singleton_ir_values, ans))
    except TypeError as e:
      raise ValueError("Output of translation rule must be iterable: "
                       f"{eqn}") from e

    assert all(isinstance(v, tuple) for v in out_nodes), (ans, eqn)
    assert all(isinstance(v, ir.Value) for w in out_nodes for v in w), (ans, eqn)
    assert len(ans) == len(eqn.outvars), (ans, eqn)
    map(write, eqn.outvars, out_nodes)
  return map(read, jaxpr.outvars)

def _ir_consts(consts):
  unique_consts = {id(const): const for const in consts}
  ir_consts = {
      id_: ir_constants(const) for id_, const in unique_consts.items()}
  return [ir_consts[id(const)] for const in consts]

def lower_fun(fun: Callable, multiple_results: bool = True) -> Callable:
  """Converts a traceable JAX function `fun` into a lowering rule.

  The returned function does not use `avals_out`, so callers may pass any value
  as `avals_out`."""
  def f_lowered(ctx, *args, **params):
    if multiple_results:
      f = fun
    else:
      f = lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params)
    axis_env = ctx.module_context.axis_env
    with core.extend_axis_env_nd(zip(axis_env.names, axis_env.sizes)):
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
    return jaxpr_subcomp(ctx.module_context, jaxpr, _ir_consts(consts),
                         *map(wrap_singleton_ir_values, args))

  return f_lowered



def _call_lowering(fn_name, stack_name, call_jaxpr, backend, ctx, avals_in,
                   avals_out, *args):
  xla.check_backend_matches(backend, ctx.platform)
  output_types = map(aval_to_ir_types, avals_out)
  flat_output_types = util.flatten(output_types)
  sub_ctx = ctx.replace(
      name_stack=xla.extend_name_stack(ctx.name_stack, stack_name))
  symbol_name = lower_jaxpr_to_fun(sub_ctx, fn_name,
                                   core.ClosedJaxpr(call_jaxpr, ())).name.value
  call = func_dialect.CallOp(flat_output_types,
                             ir.FlatSymbolRefAttr.get(symbol_name),
                             flatten_lowering_ir_args(args))
  return util.unflatten(call.results, map(len, output_types))

def _xla_call_lower(ctx, *args,
                    backend=None, name, call_jaxpr, donated_invars, inline=None,
                    device=None):
  del device, donated_invars, inline  # Ignored.
  return _call_lowering(f"jit_{name}", xla.wrap_name(name, "jit"), call_jaxpr,
                        backend, ctx.module_context, ctx.avals_in, ctx.avals_out,
                        *args)

register_lowering(xla.xla_call_p, _xla_call_lower)

def _named_call_lowering(ctx, *args, name, backend=None,
                         call_jaxpr):
  return _call_lowering(name, name, call_jaxpr, backend, ctx.module_context,
                        ctx.avals_in, ctx.avals_out, *args)

register_lowering(core.named_call_p, _named_call_lowering)
register_lowering(core.call_p, partial(_named_call_lowering, name="core_call"))


def full_like_aval(value, aval: core.ShapedArray) -> ir.Value:
  """Returns an IR constant shaped full of `value` shaped like `aval`."""
  zero = ir_constant(np.array(value, aval.dtype))
  return mhlo.BroadcastOp(aval_to_ir_type(aval), zero,
                          dense_int_elements(aval.shape)).result

def zeros_like_lowering(ctx, x):
  aval, = ctx.avals_in
  assert isinstance(aval, core.ShapedArray), aval
  return [full_like_aval(0, aval)]
register_lowering(ad_util.zeros_like_p, zeros_like_lowering)

def add_jaxvals_lowering(ctx, x, y):
  return mhlo.AddOp(x, y).results
register_lowering(ad_util.add_jaxvals_p, add_jaxvals_lowering)

register_lowering(ad_util.stop_gradient_p, lambda ctx, x: [x])


def _minmax_mhlo(op, cmp, x, y):
  """Min/max that compares complex values lexicographically as pairs."""
  tensor_type = ir.RankedTensorType(x.type)
  if ir.ComplexType.isinstance(tensor_type.element_type):
    rx = mhlo.RealOp(x).result
    ry = mhlo.RealOp(y).result
    dims = [tensor_type.get_dim_size(i) for i in range(tensor_type.rank)]
    bool_shape = ir.RankedTensorType.get(dims, ir.IntegerType.get_signless(1))
    real_eq = mhlo.CompareOp(bool_shape, rx, ry, ir.StringAttr.get("EQ"),
                             ir.StringAttr.get("FLOAT"))
    real_cmp = mhlo.CompareOp(bool_shape, rx, ry,
                              ir.StringAttr.get(cmp),
                              ir.StringAttr.get("FLOAT"))
    imag_cmp = mhlo.CompareOp(bool_shape, mhlo.ImagOp(x).result,
                              mhlo.ImagOp(y).result,
                              ir.StringAttr.get(cmp),
                              ir.StringAttr.get("FLOAT"))
    which = mhlo.SelectOp(real_eq, imag_cmp, real_cmp).result
    return mhlo.SelectOp(which, x, y)
  else:
    return op(x, y)

min_mhlo = partial(_minmax_mhlo, mhlo.MinOp, "LT")
max_mhlo = partial(_minmax_mhlo, mhlo.MaxOp, "GT")


def convert_mhlo(x, aval_in, aval_out, rounding_mode: str = "ROUND_DEFAULT"):
  """Variant of convert that has XLA HLO semantics.

  In particular, treat casts to boolean as x != 0, rather than truncating
  integer values (b/209440332).
  """
  if aval_out.dtype == np.dtype(np.bool_):
    if dtypes.issubdtype(aval_in.dtype, np.inexact):
      compare_type = "FLOAT"
    elif dtypes.issubdtype(aval_in.dtype, np.signedinteger):
      compare_type = "SIGNED"
    else:
      compare_type = "UNSIGNED"
    return mhlo.CompareOp(
        aval_to_ir_type(aval_out), x, full_like_aval(0, aval_in),
        ir.StringAttr.get("NE"), ir.StringAttr.get(compare_type)).result
  if xc._version < 49:
    return mhlo.ConvertOp(aval_to_ir_type(aval_out), x).result
  else:
    if rounding_mode == "ROUND_DEFAULT":
      if dtypes.issubdtype(aval_out.dtype, np.integer):
        rounding_mode = "ROUND_TOWARD_ZERO"
      else:
        rounding_mode = "ROUND_TO_NEAREST_TIES_TO_EVEN"
    return mhlo.ConvertOp(
        aval_to_ir_type(aval_out), x, ir.StringAttr.get(rounding_mode)).result

def _wrap_with_spmd_op(name: str,
                       result_type: ir.Type,
                       x: ir.Value,
                       sharding_proto: xc.OpSharding,
                       unspecified_dims: Optional[Set[int]] = None):
  # unspecified_dims indicate dimensions whose shardings are not specified and
  # XLA sharding propagation can change them.
  if unspecified_dims:
    backend_config = "unspecified_dims=[" + ",".join(
        [str(i) for i in sorted(unspecified_dims)]) + "]"
  else:
    backend_config = ""
  op = mhlo.CustomCallOp([result_type], [x],
                         call_target_name=ir.StringAttr.get(name),
                         has_side_effect=ir.BoolAttr.get(False),
                         backend_config=ir.StringAttr.get(backend_config),
                         api_version=i32_attr(1),
                         called_computations=ir.ArrayAttr.get([]),
                         operand_layouts=None,
                         result_layouts=None)
  op.attributes["mhlo.sharding"] = ir.StringAttr.get(
      sharding_proto.SerializeToString())
  return op.result

def wrap_with_sharding_op(x: ir.Value,
                          sharding_proto: xc.OpSharding,
                          unspecified_dims: Optional[Set[int]] = None):
  return _wrap_with_spmd_op("Sharding", x.type, x, sharding_proto, unspecified_dims)

wrap_with_full_to_shard_op = partial(_wrap_with_spmd_op, "SPMDFullToShardShape")
wrap_with_shard_to_full_op = partial(_wrap_with_spmd_op, "SPMDShardToFullShape")

def set_sharding(op, sharding_proto: xc.OpSharding):
  op.attributes["mhlo.sharding"] = ir.StringAttr.get(
      sharding_proto.SerializeToString())

# MLIR lowerings for lax primitives

def cache_lowering(f):
  """Decorator that causes the contents of a lowering rule to be reused.

  The lowering will be emitted out-of-line in a separate function, together with
  a call to that function. If the same primitive is called with the same shapes
  and parameters, a new call to the original function will be added, without
  emitting a new function.
  """
  @functools.wraps(f)
  def cached_lowering(ctx, *args, **params):
    assert ctx.primitive is not None
    key = (ctx.primitive, tuple(ctx.avals_in), tuple(ctx.avals_out),
           tuple(params.items()))
    try:
      func = ctx.module_context.cached_primitive_lowerings.get(key)
    except TypeError:
      # If the parameters aren't hashable, give up on caching.
      # TODO(phawkins): switch to requiring hashability, when XLA fallback
      # computations have been ported to MHLO.
      return f(ctx, *args, **params)
    if func is None:
      func = _emit_lowering_rule_as_fun(partial(f, **params), ctx)
      ctx.module_context.cached_primitive_lowerings[key] = func

    output_types = map(aval_to_ir_types, ctx.avals_out)
    flat_output_types = util.flatten(output_types)
    call = func_dialect.CallOp(flat_output_types,
                               ir.FlatSymbolRefAttr.get(func.name.value),
                               flatten_lowering_ir_args(args))
    return util.unflatten(call.results, map(len, output_types))
  return cached_lowering


def xla_fallback_lowering(prim: core.Primitive):
  @cache_lowering
  def fallback(ctx: LoweringRuleContext, *args, **params):
    module_ctx = ctx.module_context
    xla_computation = xla.primitive_subcomputation(
        module_ctx.platform, module_ctx.axis_env, prim, *ctx.avals_in, **params)
    submodule_str = xc._xla.mlir.xla_computation_to_mlir_module(xla_computation)
    submodule = ir.Module.parse(submodule_str)
    callee_name = None
    for op in submodule.body.operations:
      module_ctx.module.body.append(op)
      if op.name.value == "main":
        op.attributes["sym_name"] = ir.StringAttr.get(f"xla_fallback_{prim.name}")
        callee_name = ir.StringAttr(module_ctx.symbol_table.insert(op)).value
        op.attributes["sym_visibility"] = ir.StringAttr.get("private")
      else:
        module_ctx.symbol_table.insert(op)

    output_types = map(aval_to_ir_types, ctx.avals_out)
    flat_output_types = util.flatten(output_types)
    output_type = (ir.TupleType.get_tuple(flat_output_types)
                   if prim.multiple_results else flat_output_types[0])

    call = func_dialect.CallOp([output_type],
                               ir.FlatSymbolRefAttr.get(callee_name),
                               flatten_lowering_ir_args(args)).result
    if not prim.multiple_results:
      return [call]
    flat_results = [mhlo.GetTupleElementOp(typ, call, i32_attr(i)).result
                    for i, typ in enumerate(flat_output_types)]
    return util.unflatten(flat_results, map(len, output_types))
  return fallback

register_lowering(ad.custom_lin_p, ad._raise_custom_vjp_error_on_jvp)

# Lax ops missing MLIR lowerings.
# # TODO(b/203775215): these are missing from the cHLO dialect. Either add
# # them or port them to Python.
# lax.igamma_p,
# lax.igammac_p,
# lax.igamma_grad_a,
# lax.random_gamma_grad_p,
# lax.bessel_i0e_p,
# lax.bessel_i1e_p,
# lax.erf_inv_p,
# lax.regularized_incomplete_beta_p,

# # CHLO doesn't have complex lowerings of these primitives (b/203718937)
# lax.acos_p,
# lax.acosh_p,
# lax.asin_p,
# lax.asinh_p,
# lax.atan_p,
# lax.atanh_p,
# lax.cosh_p,
# lax.tan_p,

# # CHLO doesn't have a legalization for bf16 (b/203774470)
# lax.erf_p,
# lax.erfc_p,

# # Not present in cHLO or mHLO (b/203798239), although we could just emit the
# # lowered pattern ourselves.
# lax.top_k_p,

# # TODO(phawkins): implement these lax ops:
# lax.rng_bit_generator_p,
