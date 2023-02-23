# Copyright 2021 The JAX Authors.
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

# Lowering and execution path that converts jaxprs into MLIR.
from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
from functools import partial
import io
import itertools
import operator
import re
import threading
import typing
from typing import (Any, Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Protocol, Sequence, Set, Tuple, Type, Union, FrozenSet)
import warnings

import numpy as np

from jax._src import linear_util as lu
from jax.config import config
from jax._src.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src import ad_util
from jax._src import core
from jax._src import device_array
from jax._src import dtypes
from jax._src import effects as effects_lib
from jax._src import source_info_util
from jax._src import util
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.lib.mlir.dialects import func as func_dialect


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

T = typing.TypeVar("T")

Value = Any  # = ir.Value

# mypy implicitly sets this variable to true when type checking.
MYPY = False

lowerable_effects: effects_lib.EffectTypeSet = effects_lib.lowerable_effects

class _MlirThreadLocalState(threading.local):
  # We have this to allow AOT APIs to override the lowering platforms.
  # We should explore a more direct API.
  def __init__(self):
    # Used for cross-platform lowering
    # Includes some of 'cpu', 'tpu', 'cuda', 'rocm'
    self.lowering_platforms_override = None

  @contextlib.contextmanager
  def lowering_platforms(self, platforms: Sequence[str]):
    prev = self.lowering_platforms_override
    self.lowering_platforms_override = tuple(platforms)
    try:
      yield
    finally:
      self.lowering_platforms_override = prev

thread_local_state = _MlirThreadLocalState()

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

def shape_tensor(sizes: Sequence[Union[int, ir.RankedTensorType]]
                 ) -> ir.RankedTensorType:
  int1d = aval_to_ir_type(core.ShapedArray((1,), np.int32))
  def lower_dim(d):
    if type(d) is int:
      return ir_constant(np.array([d], np.int32))
    else:
      return hlo.ReshapeOp(int1d, hlo.ConvertOp(aval_to_ir_type(core.ShapedArray((), np.int32)), d))
  d, *ds = map(lower_dim, sizes)
  if not ds:
    return d
  else:
    return hlo.ConcatenateOp([d, *ds], i64_attr(0)).result


def delegate_lowering(ctx, lowering_fun, *args, **ctx_override_kwargs):
  """Side-effects on `ctx`"""
  ctx_new = ctx.replace(**ctx_override_kwargs)
  out = lowering_fun(ctx_new, *args)
  ctx.set_tokens_out(ctx_new.tokens_out)
  return out


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

_dtype_to_ir_type.update({
    np.dtype(dtypes.float8_e4m3fn): ir.Float8E4M3FNType.get,
    np.dtype(dtypes.float8_e5m2): ir.Float8E5M2Type.get,
})

def dtype_to_ir_type(dtype: Union[np.dtype, np.generic]) -> ir.Type:
  assert isinstance(dtype, (np.dtype, np.generic)), type(dtype)
  dtype = np.dtype(dtype)
  try:
    ir_type_factory = _dtype_to_ir_type[dtype]
  except KeyError as err:
    raise TypeError(
        f"No dtype_to_ir_type handler for dtype: {dtype}") from err
  return ir_type_factory()

def _array_ir_types(aval: Union[core.ShapedArray, core.DShapedArray]
                    ) -> Sequence[ir.Type]:
  if core.is_opaque_dtype(aval.dtype):
    phys_avals = aval.dtype._rules.physical_avals(aval)
    return tuple(itertools.chain(*map(_array_ir_types, phys_avals)))
  if not core.is_constant_shape(aval.shape):
    return _dynamic_array_ir_types(aval)  # type: ignore
  return (ir.RankedTensorType.get(aval.shape, dtype_to_ir_type(aval.dtype)),)

def _dynamic_array_ir_types(aval: core.ShapedArray) -> Sequence[ir.Type]:
  dyn_size = ir.ShapedType.get_dynamic_size()
  shape = [d if type(d) is int else dyn_size for d in aval.shape]
  return (ir.RankedTensorType.get(shape, dtype_to_ir_type(aval.dtype)),)

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

ir_type_handlers[core.ShapedArray] = _array_ir_types
ir_type_handlers[core.ConcreteArray] = _array_ir_types
ir_type_handlers[core.AbstractToken] = lambda _: [hlo.TokenType.get()]
ir_type_handlers[core.DShapedArray] = _dynamic_array_ir_types

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

def get_constant_handler(type_: type) -> ConstantHandler:
  return _constant_handlers[type_]

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
  raise TypeError(f"No constant handler for type: {type(val)}")

def ir_constant(val: Any, canonicalize_types: bool = True) -> ir.Value:
  """Convenience wrapper around ir_constants for singleton values."""
  values = ir_constants(val, canonicalize_types=canonicalize_types)
  if len(values) != 1:
    raise TypeError(f"ir_constant called on {val} which corresponds to "
                    f"multiple IR values {values}")
  return values[0]


def _numpy_array_constant(x: np.ndarray, canonicalize_types
                         ) -> Sequence[ir.Value]:
  if canonicalize_types:
    x = np.asarray(x, dtypes.canonicalize_dtype(x.dtype))
  element_type = dtype_to_ir_type(x.dtype)
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
  attr = ir.DenseElementsAttr.get(x, type=element_type, shape=shape)
  return (hlo.ConstantOp(attr).result,)


def _masked_array_constant_handler(*args, **kwargs):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")

register_constant_handler(np.ma.MaskedArray, _masked_array_constant_handler)

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
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None) # type: ignore
                              for ax in range(val.ndim))]  # type: ignore
    if canonicalize_types:
      collapsed_val = np.asarray(
          collapsed_val, dtypes.canonicalize_dtype(collapsed_val.dtype))
    out = hlo.BroadcastInDimOp(
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
  return _ndarray_constant_handler(np.asarray(val.device_buffer),
                                   canonicalize_types)
for t in device_array.device_array_types:
  register_constant_handler(t, _device_array_constant_handler)

def _token_constant_handler(val, canonicalize_types):
  return [hlo.CreateTokenOp().result]
register_constant_handler(core.Token, _token_constant_handler)

# Source locations

def _source_info_to_location(
    primitive: core.Primitive, params: Dict,
    source_info: source_info_util.SourceInfo,
    name_stack: source_info_util.NameStack) -> ir.Location:
  eqn_str = (f'{str(source_info.name_stack)}/'
             f'{core.str_eqn_compact(primitive.name, params)}')
  frame = source_info_util.user_frame(source_info)
  if frame is None:
    loc = ir.Location.unknown()
  else:
    loc = ir.Location.file(xla.get_canonical_source_file(frame),
                           frame.start_line, frame.start_column)
  loc = ir.Location.name(eqn_str, childLoc=loc)
  # TODO(phawkins): also include primitive.name as the operator type.
  return loc


# Translation rules
def make_ir_context() -> ir.Context:
  """Creates an MLIR context suitable for JAX IR."""
  from jax._src.lib.mlir import dialects
  context = ir.Context()
  dialects.mhlo.register_mhlo_dialect(context)
  dialects.chlo.register_dialect(context)
  dialects.stablehlo.register_dialect(context)
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
    # All collectives that touch axis_env should remember to set use_global_device_ids
    # when this context is enabled!
    if self.manual_axes != frozenset(self.mesh.axis_names):
      raise NotImplementedError(
          "Collectives in manually partitioned computations are only supported "
          "when all mesh axes are partitioned manually (no partial automatic sharding). "
          "Make sure that you mention all mesh axes in axis_resources!")
    return self.unsafe_axis_env

  @property
  def unsafe_axis_env(self):
    return xla.AxisEnv(
        nreps=self.mesh.size,
        names=self.mesh.axis_names,
        sizes=tuple(self.mesh.shape.values()))

  def extend_manual(self, axes: FrozenSet[MeshAxisName]) -> SPMDAxisContext:
    return SPMDAxisContext(self.mesh, self.manual_axes | axes)


@dataclasses.dataclass(frozen=True)
class ReplicaAxisContext:
  """A hardware axis context for parallel computations that are partitioned by JAX.

  Unlike in the SPMDAxisContext, this means that JAX might need to emit calls to
  explicit collectives.
  """
  axis_env: xla.AxisEnv


@dataclasses.dataclass(frozen=True)
class ShardingContext:
  """A hardware axis context for parallel computations that use the sharding
  interface.

  This context also uses the GSPMD partitioner.
  """
  device_assignment: Sequence[xc.Device]

  # Similar to SPMDContext as ShardingContext also uses the GSPMD partitioner.
  @property
  def axis_env(self):
    return xla.AxisEnv(nreps=1, names=(), sizes=())


AxisContext = Union[SPMDAxisContext, ReplicaAxisContext, ShardingContext]

@dataclasses.dataclass
class ModuleContext:
  """Module-wide context information for MLIR lowering."""
  context: ir.Context
  module: ir.Module
  ip: ir.InsertionPoint
  symbol_table: ir.SymbolTable
  backend_or_name: Optional[Union[str, xb.XlaBackend]]
  platforms: Sequence[str]
  axis_context: AxisContext
  name_stack: source_info_util.NameStack
  keepalives: List[Any]
  channel_iterator: Iterator[int]
  host_callbacks: List[Any]
  # The names of the dimension variables, sorted by name. This is the order in
  # which they are passed to the IR functions that need them. This is only
  # used for native serialization with polymorphic shapes when
  # --jax_dynamic_shapes is off.
  dim_vars: Sequence[str]

  # Cached primitive lowerings.
  cached_primitive_lowerings: Dict[Any, func_dialect.FuncOp]
  cached_call_jaxpr_lowerings: Dict[Any, func_dialect.FuncOp]


  @property
  def axis_env(self) -> xla.AxisEnv:
    return self.axis_context.axis_env

  def __init__(
      self,
      backend_or_name: Optional[Union[str, xb.XlaBackend]],
      platforms: Sequence[str],
      axis_context: AxisContext,
      name_stack: source_info_util.NameStack,
      keepalives: List[Any],
      channel_iterator: Iterator[int],
      host_callbacks: List[Any],
      context: Optional[ir.Context] = None,
      module: Optional[ir.Module] = None,
      ip: Optional[ir.InsertionPoint] = None,
      symbol_table: Optional[ir.SymbolTable] = None,
      cached_primitive_lowerings: Optional[Dict[Any,
                                                func_dialect.FuncOp]] = None,
      cached_call_jaxpr_lowerings: Optional[Dict[Any,
                                                 func_dialect.FuncOp]] = None,
      dim_vars: Sequence[str] = (),):
    assert platforms
    self.context = context or make_ir_context()
    self.module = module or ir.Module.create(loc=ir.Location.unknown(self.context))
    self.ip = ip or ir.InsertionPoint(self.module.body)
    self.symbol_table = symbol_table or ir.SymbolTable(self.module.operation)
    self.backend_or_name = backend_or_name
    self.platforms = platforms
    if thread_local_state.lowering_platforms_override is not None:
      assert platforms == thread_local_state.lowering_platforms_override, (platforms, thread_local_state.lowering_platforms_override)

    self.axis_context = axis_context
    self.name_stack = name_stack
    self.cached_primitive_lowerings = ({} if cached_primitive_lowerings is None
                                       else cached_primitive_lowerings)
    self.channel_iterator = channel_iterator
    self.keepalives = keepalives
    self.host_callbacks = host_callbacks
    self.cached_call_jaxpr_lowerings = ({}
                                        if cached_call_jaxpr_lowerings is None
                                        else cached_call_jaxpr_lowerings)
    self.dim_vars = dim_vars

  @property
  def backend(self) -> xb.XlaBackend:
    # It seems that only emit_python_callback uses this!
    if thread_local_state.lowering_platforms_override is not None:
      if len(self.platforms) > 1:
        raise NotImplementedError("multi-platform lowering (backend not supported)")
      backend_or_name = self.platforms[0]
    else:
      backend_or_name = self.backend_or_name  # type: ignore

    if backend_or_name is None or isinstance(backend_or_name, str):
      return xb.get_backend(backend_or_name)
    return backend_or_name

  def new_channel(self) -> int:
    return next(self.channel_iterator)

  def add_host_callback(self, host_callback: Any) -> None:
    self.host_callbacks.append(host_callback)

  def add_keepalive(self, keepalive: Any) -> None:
    self.keepalives.append(keepalive)

  def replace(self, **kw): return dataclasses.replace(self, **kw)


@dataclasses.dataclass
class LoweringRuleContext:
  """Per-rule context information for MLIR lowering."""
  module_context: ModuleContext
  platforms: Sequence[str]  # When we do multi-platform lowering, these are
                            # the plaforms for which the current rule applies
  primitive: Optional[core.Primitive]
  avals_in: Sequence[core.AbstractValue]
  avals_out: Any  # Usually Sequence[core.AbstractValue], but sometimes None.
  tokens_in: TokenSet
  tokens_out: Optional[TokenSet]  # Mutable store for output containers
  axis_size_env: Optional[Dict[core.Var, ir.Value]] = None  # Dynamic axis sizes
  dim_var_values: Sequence[ir.Value] = ()  # The values for the dimension variables
                                           # in same order as module_context.dim_vars

  def set_tokens_out(self, tokens_out: TokenSet):
    assert self.tokens_out is None, 'Should only set `tokens_out` once.'
    self.tokens_out = tokens_out

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
  if platform is None:
    _lowerings[prim] = rule
  else:
    # For backward compatibility reasons, we allow rules to be registered
    # under "gpu" even though the platforms are now called "cuda" and "rocm".
    # TODO(phawkins): fix up users to specify either "cuda" or "rocm" and remove
    # this expansion.
    for p in xb.expand_platform_alias(platform):
      _platform_specific_lowerings[p][prim] = rule
  return rule


def _unwrap_singleton_ir_values(x): return x[0] if len(x) == 1 else x
def wrap_singleton_ir_values(x: Union[ir.Value, Sequence[ir.Value]]
                             ) -> Sequence[ir.Value]:
  """Adds a consistent tuples to a mixture of tupled and untuple values."""
  return (x,) if isinstance(x, ir.Value) else tuple(x)

def flatten_lowering_ir_args(
    xs: Sequence[Union[ir.Value, Sequence[ir.Value]]]
) -> Sequence[Sequence[ir.Value]]:
  return util.flatten(map(wrap_singleton_ir_values, xs))

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


class DimExprEvaluator:
  # A wrapper for an ir.Value that overloads + and * to be used for evaluating
  # symbolic dimensions.
  __array_priority__ = 1000  # Same as tracer, for __radd__ and others on ndarray
  def __init__(self, value: ir.Value):
    self.value = value

  def __add__(self, other: Union[np.int32, np.int64, DimExprEvaluator]):
    if not isinstance(other, DimExprEvaluator):
      other = DimExprEvaluator(ir_constant(other))
    return DimExprEvaluator(hlo.AddOp(self.value, other.value).result)

  def __radd__(self, other: Union[np.int32, np.int64]):
    return DimExprEvaluator(ir_constant(other)).__add__(self)

  def __mul__(self, other: Union[np.int32, np.int64, DimExprEvaluator]):
    if not isinstance(other, DimExprEvaluator):
      other = DimExprEvaluator(ir_constant(other))
    return DimExprEvaluator(hlo.MulOp(self.value, other.value).result)

  def __rmul__(self, other: Union[np.int32, np.int64]):
    return DimExprEvaluator(ir_constant(other)).__mul__(self)

  def __divmod__(self, divisor: Union[np.int32, np.int64, DimExprEvaluator]):
    if not isinstance(divisor, DimExprEvaluator):
      divisor = DimExprEvaluator(ir_constant(divisor))
    # Quotient
    raw_quotient = hlo.DivOp(self.value, divisor.value)
    raw_remainder = hlo.RemOp(self.value, divisor.value)
    ops_different_sign = compare_hlo(hlo.SignOp(self.value),
                                     hlo.SignOp(divisor.value),
                                     "NE", "SIGNED")
    rem_ne_zero = compare_hlo(raw_remainder, ir_constant(np.int64(0)),
                              "NE", "SIGNED")
    must_adjust = hlo.AndOp(ops_different_sign, rem_ne_zero)
    quotient = hlo.SelectOp(must_adjust,
                            hlo.SubtractOp(raw_quotient, ir_constant(np.int64(1))),
                            raw_quotient)
    # Remainder
    remainder = hlo.SubtractOp(self.value, hlo.MulOp(divisor.value, quotient))
    return (DimExprEvaluator(quotient.result),
            DimExprEvaluator(remainder.result))

  def __rdivmod__(self, dividend: Union[np.int32, np.int64]):
    return DimExprEvaluator(ir_constant(dividend)).__divmod__(self)

def eval_dynamic_shape(ctx: LoweringRuleContext,
                       shape: core.Shape) -> Tuple[Union[int, Value], ...]:
  # assert not core.is_constant_shape(shape)
  if config.jax_dynamic_shapes:
    return tuple(ctx.axis_size_env.get(d, d) for d in shape)  # type: ignore
  else:
    dim_var_env = {dv_name : DimExprEvaluator(dv_val[0])
                   for dv_name, dv_val in zip(ctx.module_context.dim_vars, ctx.dim_var_values)}
    def eval_dim(d: core.DimSize) -> Union[int, ir.Value]:
      try:
        return operator.index(d)
      except:
        if isinstance(d, ir.Value):
          return d
        else:
          # Is a dimension polynomial
          return d.evaluate(dim_var_env).value  # type: ignore
    return tuple(eval_dim(d) for d in shape)

class LoweringResult(NamedTuple):
  module: ir.Module
  keepalive: Optional[Any]
  host_callbacks: List[Any]


_platforms_with_donation = ["cpu", "cuda", "rocm", "tpu"]


def lower_jaxpr_to_module(
    module_name: str,
    jaxpr: core.ClosedJaxpr,
    unordered_effects: List[core.Effect],
    ordered_effects: List[core.Effect],
    backend_or_name: Optional[Union[str, xb.XlaBackend]],
    platforms: Sequence[str],
    axis_context: AxisContext,
    name_stack: source_info_util.NameStack,
    donated_args: Sequence[bool],
    replicated_args: Optional[Sequence[bool]] = None,
    arg_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    result_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None
) -> LoweringResult:
  """Lowers a top-level jaxpr to an MLIR module.

  Handles the quirks of the argument/return value passing conventions of the
  runtime.
  """
  assert platforms
  for platform in platforms:
    platform = xb.canonicalize_platform(platform)
    if not xb.is_known_platform(platform):
      raise ValueError(f"Unknown platform {platform}")
  input_output_aliases = None
  in_avals = jaxpr.in_avals
  if arg_shardings is not None:
    in_avals = [
        sharded_aval(in_aval, in_sharding)
        for in_aval, in_sharding in zip(in_avals, arg_shardings)
    ]
  out_avals = jaxpr.out_avals
  if result_shardings is not None:
    out_avals = []
    for out_aval, out_sharding in zip(jaxpr.out_avals, result_shardings):
      if (out_aval is not core.abstract_token and
          core.is_opaque_dtype(out_aval.dtype)):
        # TODO(frostig,mattjj,necula): asserts a single physical aval
        out_aval, = out_aval.dtype._rules.physical_avals(out_aval)
      out_avals.append(sharded_aval(out_aval, out_sharding))

  if all(p in _platforms_with_donation for p in platforms):
    input_output_aliases, donated_args = _set_up_aliases(
        in_avals, out_avals, donated_args)
  unlowerable_effects = lowerable_effects.filter_not_in(jaxpr.effects)
  if unlowerable_effects:
    raise ValueError(f'Cannot lower jaxpr with effects: {jaxpr.effects}')
  if any(donated_args):
    # TODO(tomhennigan): At call time we should mark these buffers as deleted.
    unused_donations = [str(a) for a, d in zip(in_avals, donated_args)
                        if d]
    msg = "See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation."
    if not all(p in _platforms_with_donation for p in platforms):
      msg = f"Donation is not implemented for some of {platforms}.\n{msg}"
    warnings.warn(f"Some donated buffers were not usable: {', '.join(unused_donations)}.\n{msg}")

  # HLO channels need to start at 1
  channel_iter = itertools.count(1)
  # Create a keepalives list that will be mutated during the lowering.
  keepalives: List[Any] = []
  host_callbacks: List[Any] = []

  dim_vars: Sequence[str]
  if not config.jax_dynamic_shapes:
    # Find the dimension variables
    all_dim_poly = [d
                    for aval in jaxpr.in_avals if hasattr(aval, "shape")
                    for d in aval.shape if not core.is_constant_dim(d)]
    dim_vars = tuple(sorted(functools.reduce(lambda acc, new: acc.union(new.get_vars()),
                                             all_dim_poly, set())))
  else:
    dim_vars = ()

  if len(platforms) > 1:
    dim_vars = ("_platform_index",) + dim_vars  # We will have an extra platform_index argument
  ctx = ModuleContext(backend_or_name, platforms, axis_context, name_stack,
                      keepalives, channel_iter, host_callbacks,
                      dim_vars=dim_vars)
  with ctx.context, ir.Location.unknown(ctx.context):
    # Remove module name characters that XLA would alter. This ensures that
    # XLA computation preserves the module name.
    module_name = _module_name_regex.sub("_", module_name)
    ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get(
        module_name)
    unlowerable_effects = lowerable_effects.filter_not_in(jaxpr.effects)
    if unlowerable_effects:
      raise ValueError(
          f'Cannot lower jaxpr with unlowerable effects: {unlowerable_effects}')
    lower_jaxpr_to_fun(
        ctx, "main", jaxpr, ordered_effects, public=True, create_tokens=True,
        replace_tokens_with_dummy=True,
        num_output_tokens=0,
        replicated_args=replicated_args,
        arg_shardings=arg_shardings, result_shardings=result_shardings,
        input_output_aliases=input_output_aliases)

  if not ctx.module.operation.verify():
    module_string = module_to_string(ctx.module)
    raise ValueError(
        f"Cannot lower jaxpr with verifier errors: {module_string}")

  return LoweringResult(ctx.module, ctx.keepalives, ctx.host_callbacks)

def module_to_string(module: ir.Module) -> str:
  output = io.StringIO()
  module.operation.print(file=output, enable_debug_info=True,
                         print_generic_op_form=False)
  return output.getvalue()

def module_to_bytecode(module: ir.Module) -> bytes:
  output = io.BytesIO()
  module.operation.write_bytecode(file=output)
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

Token = Sequence[ir.Value]

def token_type() -> Sequence[ir.Type]:
  return [hlo.TokenType.get()]

def create_token() -> Token:
  return wrap_singleton_ir_values(hlo.CreateTokenOp().result)

class TokenSet:
  """An immutable container of tokens to be used to lower effectful jaxprs. When lowering
  effectful jaxprs, we need to thread HLO tokens to sequence them. Each effect
  will need its own token that will be threaded in and out of the effectful
  primitives. A `TokenSet` encapsulates a set of HLO tokens that will be
  used by the lowering rules.
  """
  _tokens: typing.OrderedDict[core.Effect, Token]

  def __init__(self, *args, **kwargs):
    self._tokens = collections.OrderedDict(*args, **kwargs)

  def __len__(self):
    return len(self._tokens)

  def get(self, effect: core.Effect) -> Token:
    return self._tokens[effect]

  @classmethod
  def create(cls, effects: Sequence[core.Effect]) -> TokenSet:
    """Creates a `TokenSet` corresponding to a list of `core.Effect`s."""
    tokens = [create_token() for _ in effects]
    return TokenSet(zip(effects, tokens))

  def items(self) -> Sequence[Tuple[core.Effect, Token]]:
    return tuple(self._tokens.items())

  def effects(self) -> Sequence[core.Effect]:
    return tuple(self._tokens.keys())

  def tokens(self) -> Sequence[Token]:
    return tuple(self._tokens.values())

  def subset(self, effects: Sequence[core.Effect]) -> TokenSet:
    """Return a subset of the `TokenSet` restricted to a set of `core.Effect`s."""
    return TokenSet((eff, self._tokens[eff]) for eff in effects)

  def update_tokens(self, tokens: TokenSet) -> TokenSet:
    """Returns a new `TokenSet` with tokens replaced with ones from the input `TokenSet`."""
    new_tokens = []
    for eff in self.effects():
      if eff in tokens._tokens:
        new_tokens.append(tokens._tokens[eff])
      else:
        new_tokens.append(self._tokens[eff])
    return TokenSet(zip(self.effects(), new_tokens))

def dummy_token_type() -> Sequence[ir.Type]:
  return aval_to_ir_types(core.ShapedArray((0,), np.bool_))

def dummy_token() -> Sequence[ir.Value]:
  return ir_constants(np.zeros(0, np.bool_))

def lower_jaxpr_to_fun(
    ctx: ModuleContext,
    name: str,
    jaxpr: core.ClosedJaxpr,
    effects: Sequence[core.Effect],
    *,
    create_tokens: bool = False,
    public: bool = False,
    replace_tokens_with_dummy: bool = False,
    replicated_args: Optional[Sequence[bool]] = None,
    arg_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    result_shardings: Optional[Sequence[Optional[xc.OpSharding]]] = None,
    use_sharding_annotations: bool = True,
    input_output_aliases: Optional[Sequence[Optional[int]]] = None,
    num_output_tokens: int = 0,
    api_name: str = 'jit',
) -> func_dialect.FuncOp:
  """Lowers jaxpr and its callees to an IR function.

  Assumes that an MLIR context, location, and insertion point are set.

  Args:
    ctx: the lowering context.
    name: the function name. The name will be uniquified by the symbol table,
      so it is ok to use the same name multiple times.
    jaxpr: the jaxpr to lower.
    effects: a sequence of `core.Effect`s corresponding to an ordering of tokens
      that will be created in or used by the lowered function.
    create_tokens: if true, the HLO will create tokens and ignore dummy input tokens.
    public: if true, the function's visibility is set to "public".
    replace_tokens_with_dummy: if true, token arguments/return values are
      replaced with bool arrays of size [0].
    replicated_args: if present, annotates arguments as replicated.
    arg_shardings: sharding annotations for each argument (optional).
    result_shardings: sharding annotations for each argument (optional).
    use_sharding_annotations: if True, use "mhlo.sharding" annotations on
      parameters and return values to express sharding. If False, use
      hlo.custom_call operators with sharding annotations.
      TODO(b/228598865): remove this option when "mhlo.sharding" annotations are
      propagated on non-entry functions during MLIR->HLO conversion.
    input_output_aliases: optional sequence that maps argument numbers to the
      corresponding output that should alias them.
    api_name: The name of the higher level primitive which should show up in the
      name stack.
  Returns the name of the function.
  """
  def aval_to_types(aval):
    if replace_tokens_with_dummy and aval is core.abstract_token:
      aval = core.ShapedArray((), np.dtype(np.bool_))
    return aval_to_ir_types(aval)

  num_dim_vars = len(ctx.dim_vars)
  dim_var_types = map(aval_to_types, [core.ShapedArray((), dtypes.canonicalize_dtype(np.int64))] * num_dim_vars)

  # Function inputs: *dim_var_values, *tokens, *actual_inputs
  input_types = map(aval_to_types, jaxpr.in_avals)
  output_types = map(aval_to_types, jaxpr.out_avals)
  num_tokens = len(effects)

  if create_tokens:
    # If we create the tokens they won't be inputs to the MLIR function.
    token_types = [dummy_token_type() for _ in effects]
    output_token_types = [dummy_token_type() for _ in range(num_output_tokens)]
  else:
    # If we aren't creating tokens they will be the initial inputs to the
    # MLIR function.
    output_token_types = []
    num_tokens = len(effects)
    token_types = [token_type() for _ in effects]
  input_types = [*dim_var_types, *token_types, *input_types]
  output_types = [*output_token_types, *token_types, *output_types]
  if input_output_aliases is not None:
    token_input_output_aliases = [None] * (num_dim_vars + num_tokens)
    input_output_aliases = [*token_input_output_aliases, *input_output_aliases]
    # Update the existing aliases to account for the new output values
    input_output_aliases = [None if a is None
                            else a + num_output_tokens + num_tokens
                            for a in input_output_aliases]
  if arg_shardings is not None:
    token_shardings = [None] * (num_dim_vars + num_tokens)
    arg_shardings = [*token_shardings, *arg_shardings]
  if result_shardings is not None:
    token_shardings = [None] * (num_tokens + num_output_tokens)
    result_shardings = [*token_shardings, *result_shardings]
  if replicated_args is not None:
    token_replicated_args = [False] * (num_dim_vars + num_tokens)
    replicated_args = [*token_replicated_args, *replicated_args]
  flat_input_types = util.flatten(input_types)
  flat_output_types = util.flatten(output_types)
  ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
  func_op = func_dialect.FuncOp(name, ftype, ip=ctx.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get(
      "public" if public else "private")
  ctx.symbol_table.insert(func_op)
  ir_arg_shardings = None
  if arg_shardings is not None:
    ir_arg_shardings = util.flatten(
        [[sharding] * len(types) for sharding, types
         in zip(arg_shardings, input_types)])
  ir_result_shardings = None
  if result_shardings is not None:
    ir_result_shardings = util.flatten(
        [[sharding] * len(types)
         for sharding, types in zip(result_shardings, output_types)])

  if (replicated_args is not None or ir_arg_shardings is not None
      or input_output_aliases is not None):
    arg_attrs: List[Dict[str, ir.Attribute]] = [
        {} for _ in range(len(flat_input_types))]

    if replicated_args is not None:
      replicated_ir_args = [[replicated] * len(types) for replicated, types
                            in zip(replicated_args, input_types)]
      for attrs, replicated in zip(arg_attrs, util.flatten(replicated_ir_args)):
        if replicated:
          attrs["mhlo.is_same_data_across_replicas"] = ir.UnitAttr.get()

    if use_sharding_annotations and ir_arg_shardings is not None:
      for attrs, sharding in zip(arg_attrs, ir_arg_shardings):
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

  if use_sharding_annotations and ir_result_shardings is not None:
    func_op.result_attrs = ir.ArrayAttr.get([
        ir.DictAttr.get(
            {} if sharding is None else
            {"mhlo.sharding": ir.StringAttr.get(sharding.SerializeToString())}
        ) for sharding in ir_result_shardings
    ])

  entry_block = func_op.add_entry_block()
  with ir.InsertionPoint(entry_block):
    flat_args = entry_block.arguments
    if not use_sharding_annotations and ir_arg_shardings is not None:
      flat_args = [a if s is None else wrap_with_sharding_op(a, s)
                   for a, s in zip(flat_args, ir_arg_shardings)]

    unflattened_args = util.unflatten(flat_args, map(len, input_types))
    # We separate out the dimension variable inputs, the token inputs and
    # the usual inputs. The dimension variables and token inputs
    # will be passed to `jaxpr_subcomp` separately from the `args`.
    dim_var_values, token_args, unflattened_args = util.split_list(unflattened_args, [num_dim_vars, num_tokens])
    if create_tokens:
      tokens_in = TokenSet.create(effects)
    else:
      tokens_in = TokenSet(zip(effects, token_args))
    args: List[List[ir.Value]] = []
    for aval, arg in zip(jaxpr.in_avals, unflattened_args):
      if replace_tokens_with_dummy and aval is core.abstract_token:
        args.append(hlo.CreateTokenOp().results)
      else:
        args.append(arg)
    callee_name_stack = util.extend_name_stack(
        ctx.name_stack, util.wrap_name(name, api_name))
    out_vals, tokens_out = jaxpr_subcomp(ctx.replace(name_stack=callee_name_stack),
                                         jaxpr.jaxpr, tokens_in, map(ir_constants, jaxpr.consts),
                                         *args, dim_var_values=dim_var_values)
    outs = []
    if create_tokens:
      for _ in range(num_output_tokens):
        outs.append(dummy_token())
      for _ in effects:
        outs.append(dummy_token())
    else:
      for token in tokens_out.tokens():
        outs.append(token)
    for aval, out in zip(jaxpr.out_avals, out_vals):
      if replace_tokens_with_dummy and aval is core.abstract_token:
        outs.append(ir_constants(np.zeros((), np.bool_)))
      else:
        outs.append(out)
    flat_outputs = util.flatten(outs)
    if not use_sharding_annotations and ir_result_shardings is not None:
      flat_outputs = [o if s is None else wrap_with_sharding_op(o, s)
                      for o, s in zip(flat_outputs, ir_result_shardings)]

    func_dialect.ReturnOp(flat_outputs)

  return func_op

def _emit_lowering_rule_as_fun(lowering_rule,
                               ctx: LoweringRuleContext) -> func_dialect.FuncOp:
  """Emits the contents of a lowering rule as a private function."""
  num_dim_vars = len(ctx.module_context.dim_vars)
  # TODO(necula) maybe only pass the dim_vars if they are needed?
  dim_var_types = map(aval_to_ir_types, [core.ShapedArray((), dtypes.canonicalize_dtype(np.int64))] * num_dim_vars)

  input_types = map(aval_to_ir_types, ctx.avals_in)
  output_types = map(aval_to_ir_types, ctx.avals_out)
  token_types = [token_type() for _ in ctx.tokens_in.items()]
  input_types = [*dim_var_types, *token_types, *input_types]
  output_types = [*token_types, *output_types]

  flat_input_types = util.flatten(input_types)
  flat_output_types = util.flatten(output_types)
  ftype = ir.FunctionType.get(flat_input_types, flat_output_types)
  assert ctx.primitive is not None
  func_op = func_dialect.FuncOp(ctx.primitive.name, ftype,
                                ip=ctx.module_context.ip)
  func_op.attributes["sym_visibility"] = ir.StringAttr.get("private")
  ctx.module_context.symbol_table.insert(func_op)
  entry_block = func_op.add_entry_block()
  with ir.InsertionPoint(entry_block):
    unflattened_args = util.unflatten(entry_block.arguments,
                                      map(len, input_types))
    dim_var_values, token_args, unflattened_args = util.split_list(unflattened_args, [num_dim_vars, len(ctx.tokens_in)])
    sub_ctx = ctx.replace(tokens_in=TokenSet(zip(ctx.tokens_in.effects(), token_args)),
                          dim_var_values=dim_var_values)
    outs = lowering_rule(sub_ctx, *_unwrap_singleton_ir_values(unflattened_args))
    if sub_ctx.tokens_out:
      outs = [*sub_ctx.tokens_out.tokens(), outs]
    func_dialect.ReturnOp(util.flatten(map(wrap_singleton_ir_values, outs)))
  return func_op

def jaxpr_subcomp(ctx: ModuleContext, jaxpr: core.Jaxpr,
                  tokens: TokenSet,
                  consts: Sequence[Sequence[ir.Value]],
                  *args: Sequence[ir.Value],
                  dim_var_values: Sequence[ir.Value]
                  ) -> Tuple[Sequence[Sequence[ir.Value]], TokenSet]:
  """Lowers a jaxpr into MLIR, inlined into an existing function.

  Assumes that an MLIR context, location, and insertion point are set.

  dim_var_values: the list of dimension variables values in the current
    IR function, in the order of ctx.dim_vars.
  """
  assert "gpu" not in ctx.platforms
  def read(v: core.Atom) -> Sequence[ir.Value]:
    if type(v) is core.Literal:
      return ir_constants(v.val, canonicalize_types=True)
    else:
      assert isinstance(v, core.Var)
      return env[v]

  def aval(v: core.Atom) -> core.AbstractValue:
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
  assert len(ctx.dim_vars) == len(dim_var_values), (ctx.dim_vars, dim_var_values)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_nodes = map(read, eqn.invars)
    assert isinstance(ctx.name_stack, source_info_util.NameStack), type(ctx.name_stack)
    source_info = eqn.source_info.replace(
        name_stack=ctx.name_stack + eqn.source_info.name_stack)
    loc = _source_info_to_location(eqn.primitive, eqn.params, source_info,
                                   ctx.name_stack)
    with source_info_util.user_context(eqn.source_info.traceback), loc:
      eqn_ctx = ctx.replace(name_stack=source_info.name_stack)
      effects = list(effects_lib.ordered_effects.filter_in(eqn.effects))
      tokens_in = tokens.subset(effects)
      avals_in = map(aval, eqn.invars)
      rule_ctx = LoweringRuleContext(
          module_context=eqn_ctx, platforms=ctx.platforms,
          primitive=eqn.primitive, avals_in=avals_in,
          avals_out=map(aval, eqn.outvars), tokens_in=tokens_in,
          tokens_out=None, dim_var_values=dim_var_values)
      if config.jax_dynamic_shapes:
        axis_size_env = {d: read(d)[0]
                         for a in avals_in if type(a) is core.DShapedArray
                         for d in a.shape if type(d) is core.Var}
        rule_ctx = rule_ctx.replace(axis_size_env=axis_size_env)
      rule_inputs = map(_unwrap_singleton_ir_values, in_nodes)

      rules = []  # See lower_multi_platform
      for p in ctx.platforms:
        if eqn.primitive in _platform_specific_lowerings[p]:
          rules.append(([p], _platform_specific_lowerings[p][eqn.primitive]))
        elif eqn.primitive in xla._backend_specific_translations[p]:
          rules.append(([p], xla_fallback_lowering(eqn.primitive)))
      if eqn.primitive in _lowerings:
        rules.append((None, _lowerings[eqn.primitive]))  # type: ignore
      elif eqn.primitive in xla._translations:
        rules.append((None, xla_fallback_lowering(eqn.primitive)))  # type: ignore

      output = lower_multi_platform(rule_ctx, str(eqn), rules, *rule_inputs, **eqn.params)
      if effects:
        # If there were ordered effects in the primitive, there should be output
        # tokens we need for subsequent ordered effects.
        tokens_out = rule_ctx.tokens_out
        if tokens_out is None:
          raise ValueError(
              f'Lowering rule for `{eqn.primitive}` needs to set `tokens_out` '
              f'because it has effects: {eqn.effects}.')
        if tokens_out.effects() != tokens_in.effects():
          raise ValueError(
              f'Lowering rule for `{eqn.primitive}` '
              'returns incorrect set of output tokens. '
              f'Expected: {tuple(tokens_in.effects())} vs. Actual: {tuple(tokens_out.effects())}')
        tokens = tokens.update_tokens(tokens_out)

    try:
      out_nodes = tuple(map(wrap_singleton_ir_values, output))
    except TypeError as e:
      raise ValueError("Output of translation rule must be iterable: "
                       f"{eqn}, got output {output}") from e

    assert all(isinstance(v, tuple) for v in out_nodes), (output, eqn)
    assert all(isinstance(v, ir.Value) for w in out_nodes for v in w), (output, eqn)
    assert len(output) == len(eqn.outvars), (output, eqn)
    map(write, eqn.outvars, out_nodes)
  return map(read, jaxpr.outvars), tokens

def lower_multi_platform(ctx: LoweringRuleContext,
                         description: str,
                         rules: Sequence[Tuple[Optional[Sequence[str]], Callable]],
                         *rule_args: ir.Value,
                         **rule_kwargs) -> ir.Value:
  """Emit multi-platform code.

  Given ctx.module_context.multi_platform = ["cpu", "gpu", "tpu"] and
        rules = [(["tpu", "cpu"], rule1), (None, rule2)
  emits:
     rule_idx = case current_platform_idx:
                    0: return 0  # cpu
                    1: return 1  # gpu
                    2: return 0  # tpu
     output = case rule_idx
                0: return rule1(*rule_inputs, **eqn_params)
                1: return rule2(*rule_inputs, **eqn_params)
  Args:
    ctx:
    description: a string to include in error messages
    rules: a sequence of per-platform rules. Each entry is a tuple, with the
      first element specifying the platforms, either a sequence of platforms
      names (may be empty), or None to denote a default entry to use when no
      other entry applies. The second element of the tuple is a function to
      invoke with a LoweringRuleContext (a sub-context of `ctx`),
      and `*rule_args` and `**rule_kwargs`.
    rule_args: the args of the lowering rules
    rule_kwargs: the kwars of the lowering rules
  """
  platforms_with_specific_rules = util.flatten([ps for ps, _ in rules if ps is not None])
  platforms_with_default_rule = [p for p in ctx.module_context.platforms
                                 if p not in platforms_with_specific_rules]
  filtered_rules: Any = []  # Only the rules we care about
  map_rule_idx: Dict[str, int] = {}  # Maps platform name to index into rule index
  for ps, r in rules:
    rule_index = len(filtered_rules)
    if ps is not None:
      # Keep only rules that mention our platforms
      interesting_ps = [p for p in ctx.module_context.platforms if p in ps]
      if interesting_ps:
        for p in interesting_ps:
          map_rule_idx[p] = rule_index
        filtered_rules.append((ps, r))
    else:
      if platforms_with_default_rule:
        for p in platforms_with_default_rule:
          map_rule_idx[p] = rule_index
        filtered_rules.append((platforms_with_default_rule, r))
  if not filtered_rules:
    raise NotImplementedError(
        f"MLIR translation rule for primitive '{description}' not "
        f"found for platforms {ctx.module_context.platforms}")

  # Maybe there is a single rule left
  if len(filtered_rules) == 1:
    return filtered_rules[0][1](ctx, *rule_args, **rule_kwargs)

  assert len(ctx.module_context.platforms) > 1 and len(filtered_rules) >= 2
  assert len(ctx.dim_var_values) >= 1, "Must have a platform_index variable"

  current_platform_idx = ctx.dim_var_values[0][0]  # The first dim_var_values is the platform index
  rule_idx_op = hlo.CaseOp(aval_to_ir_types(core.ShapedArray((), dtype=np.int32)),
                           index=current_platform_idx,
                           num_branches=len(ctx.module_context.platforms))
  for i, p in enumerate(ctx.module_context.platforms):
    branch = rule_idx_op.regions[i].blocks.append()
    with ir.InsertionPoint(branch):
      hlo.ReturnOp(ir_constants(np.int32(map_rule_idx[p])))

  case_op = hlo.CaseOp(util.flatten(map(aval_to_ir_types, ctx.avals_out)),
                       index=rule_idx_op,
                       num_branches=len(rules))
  for i, (ps, rule) in enumerate(filtered_rules):
    inner_ctx = ctx.replace(tokens_out=None, platforms=ps)
    branch = case_op.regions[i].blocks.append()
    with ir.InsertionPoint(branch):
      output = rule(inner_ctx, *rule_args, **rule_kwargs)
      try:
        out_nodes = tuple(map(wrap_singleton_ir_values, output))
      except TypeError as e:
        raise ValueError("Output of translation rule must be iterable: "
                         f"{description}, got output {output}") from e
      hlo.ReturnOp(util.flatten(out_nodes))

  return case_op.results

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
    f = fun if multiple_results else lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params)

    if config.jax_dynamic_shapes:
      # We might be applying this function to arguments with dynamic shapes,
      # i.e. there might be Vars in the shape tuples of ctx.avals_in. In that
      # case, we need to form a jaxpr with leading binders for those axis size
      # arguments (by computing an InputType and using trace_to_jaxpr_dynamic2),
      # and we need to call jaxpr_subcomp with these arguments made explicit.
      args = (*ctx.axis_size_env.values(), *args)
      idx = {d: core.DBIdx(i) for i, d in enumerate(ctx.axis_size_env)}
      i32_aval = core.ShapedArray((), np.dtype('int32'))
      implicit_args = [(i32_aval, False)] * len(ctx.axis_size_env)
      explicit_args = [(a.update(shape=tuple(idx.get(d, d) for d in a.shape))
                        if type(a) is core.DShapedArray else a, True)
                       for a in ctx.avals_in]
      wrapped_fun = lu.annotate(wrapped_fun, (*implicit_args, *explicit_args))
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic2(wrapped_fun)
    else:
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
      # TODO(frostig,mattjj): check ctx.avals_out against jaxpr avals out?

    out, tokens = jaxpr_subcomp(
        ctx.module_context, jaxpr, ctx.tokens_in, _ir_consts(consts),
        *map(wrap_singleton_ir_values, args), dim_var_values=ctx.dim_var_values)
    ctx.set_tokens_out(tokens)
    return out

  return f_lowered


def _lower_jaxpr_to_fun_cached(ctx, fn_name, call_jaxpr, effects):
  if not call_jaxpr.consts:
    # Cacheable.
    key = (fn_name, call_jaxpr.jaxpr, tuple(effects))
    try:
      func_op = ctx.cached_call_jaxpr_lowerings[key]
    except KeyError:
      func_op = lower_jaxpr_to_fun(ctx, fn_name, call_jaxpr, effects)
      ctx.cached_call_jaxpr_lowerings[key] = func_op
  else:
    func_op = lower_jaxpr_to_fun(ctx, fn_name, call_jaxpr, effects)
  return func_op


def _call_lowering(fn_name, stack_name, call_jaxpr, backend, ctx, avals_in,
                   avals_out, tokens_in, *args,
                   dim_var_values: Sequence[ir.Value]):
  if isinstance(call_jaxpr, core.Jaxpr):
    call_jaxpr = core.ClosedJaxpr(call_jaxpr, ())
  # TODO(necula): why did we need the check below? backend is not used
  # xla.check_backend_matches(backend, ctx.platform)
  effects = tokens_in.effects()
  output_types = map(aval_to_ir_types, avals_out)
  output_types = [token_type()] * len(effects) + output_types
  flat_output_types = util.flatten(output_types)
  symbol_name = _lower_jaxpr_to_fun_cached(ctx, fn_name, call_jaxpr, effects).name.value
  args = tuple([*dim_var_values, *tokens_in.tokens(), *args])
  call = func_dialect.CallOp(flat_output_types,
                             ir.FlatSymbolRefAttr.get(symbol_name),
                             flatten_lowering_ir_args(args))
  out_nodes = util.unflatten(call.results, map(len, output_types))
  tokens, out_nodes = util.split_list(out_nodes, [len(effects)])
  tokens_out = tokens_in.update_tokens(TokenSet(zip(effects, tokens)))
  return out_nodes, tokens_out

def _xla_call_lower(ctx, *args,
                    backend=None, name, call_jaxpr, donated_invars, inline=None,
                    device=None, keep_unused=None):
  del device, donated_invars, inline, keep_unused  # Ignored.
  out_nodes, tokens = _call_lowering(
      name, util.wrap_name(name, "jit"), call_jaxpr, backend,
      ctx.module_context, ctx.avals_in, ctx.avals_out, ctx.tokens_in,
      *args, dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens)
  return out_nodes

register_lowering(xla.xla_call_p, _xla_call_lower)

def _core_call_lowering(ctx, *args, name, backend=None, call_jaxpr):
  out_nodes, tokens = _call_lowering(
      name, name, call_jaxpr, backend, ctx.module_context,
      ctx.avals_in, ctx.avals_out, ctx.tokens_in, *args,
      dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens)
  return out_nodes

register_lowering(core.call_p, partial(_core_call_lowering, name="core_call"))
register_lowering(core.closed_call_p,
                  partial(_core_call_lowering, name="core_closed_call"))

def broadcast_in_dim(ctx: LoweringRuleContext, op, aval_out: core.AbstractValue, *,
                     broadcast_dimensions) -> ir.Value:
  # Lower a possibly-dynamic broadcast_in_dim
  if core.is_opaque_dtype(aval_out.dtype):  # type: ignore
    return aval_out.dtype._rules.broadcast_in_dim_mlir(  # type: ignore
        ctx, aval_out, op,
        broadcast_dimensions=broadcast_dimensions)
  if not core.is_constant_shape(aval_out.shape):  # type: ignore
    shape = eval_dynamic_shape(ctx, aval_out.shape)  # type: ignore
    return hlo.DynamicBroadcastInDimOp(
        aval_to_ir_type(aval_out), op,
        shape_tensor(shape),
        dense_int_elements(broadcast_dimensions),
    ).result
  else:
    return hlo.BroadcastInDimOp(
        aval_to_ir_type(aval_out), op,
        dense_int_elements(broadcast_dimensions)).result

def multi_broadcast_in_dim(ctx: LoweringRuleContext,
                           ops: Sequence[ir.Value],
                           ops_avals: Sequence[core.AbstractValue],
                           out_shape: core.Shape) -> Sequence[ir.Value]:
  """Broadcasts multiple ops to the out_shape."""
  out = []
  for op, op_aval in zip(ops, ops_avals):
    op_aval_shape = op_aval.shape  # type: ignore
    if core.symbolic_equal_shape(op_aval_shape, out_shape):  # type: ignore
      out.append(op)
    else:
      assert len(op_aval_shape) <= len(out_shape), (op_aval_shape, out_shape)
      broadcast_dimensions = list(range(len(out_shape) - len(op_aval_shape), len(out_shape)))
      out.append(broadcast_in_dim(ctx, op,
                                  core.ShapedArray(out_shape, op_aval.dtype),  # type: ignore
                                  broadcast_dimensions=broadcast_dimensions))
  return out

def reshape(ctx: LoweringRuleContext, op, aval_out: core.AbstractValue) -> ir.Value:
  if core.is_opaque_dtype(aval_out.dtype):  # type: ignore
    # TODO(frostig,mattjj,necula): asserts a single physical aval, and a
    # particular reshape rule (reshape to the output physical aval's shape)
    aval_out, = aval_out.dtype._rules.physical_avals(aval_out)  # type: ignore
  if not core.is_constant_shape(aval_out.shape):  # type: ignore
    shape = eval_dynamic_shape(ctx, aval_out.shape)  # type: ignore
    return hlo.DynamicReshapeOp(
        aval_to_ir_type(aval_out), op,
        shape_tensor(shape),
    ).result
  else:
    return hlo.ReshapeOp(aval_to_ir_type(aval_out), op).result

def slice_op(ctx: LoweringRuleContext, x, aval_out, *,
             start_indices, limit_indices, strides) -> ir.Value:
  if core.is_opaque_dtype(aval_out.dtype):
    return [aval_out.dtype._rules.slice_mlir(
        ctx, aval_out, x, start_indices, limit_indices, strides)]

  if any(not core.is_constant_shape(s) for s in (start_indices, limit_indices, strides)):
    start_indices = eval_dynamic_shape(ctx, start_indices)
    limit_indices = eval_dynamic_shape(ctx, limit_indices)
    strides = eval_dynamic_shape(ctx, strides)
    return hlo.RealDynamicSliceOp(aval_to_ir_type(aval_out),
                                  x,
                                  shape_tensor(start_indices),
                                  shape_tensor(limit_indices),
                                  shape_tensor(strides)).result
  else:
    return hlo.SliceOp(x,
                       dense_int_elements(start_indices),
                       dense_int_elements(limit_indices),
                       dense_int_elements(strides)).result

def dynamic_slice(ctx: LoweringRuleContext, aval_out, x, *,
                  start_indices) -> ir.Value:
  if core.is_opaque_dtype(aval_out.dtype):
    return aval_out.dtype._rules.dynamic_slice_mlir(ctx, aval_out, x,
                                                    start_indices)
  slice_sizes = aval_out.shape
  if not core.is_constant_shape(slice_sizes):
    slice_sizes = eval_dynamic_shape(ctx, slice_sizes)
    return hlo.RealDynamicSliceOp(
        aval_to_ir_type(aval_out), x,
        shape_tensor(start_indices),
        hlo.AddOp(shape_tensor(start_indices),
                  shape_tensor(slice_sizes)).result,
        shape_tensor([1] * len(slice_sizes))
    ).result
  else:
    return hlo.DynamicSliceOp(x, start_indices,
                              dense_int_elements(slice_sizes)).result

def dynamic_update_slice(ctx: LoweringRuleContext, aval_out, x, update, *,
                         start_indices) -> ir.Value:
  if core.is_opaque_dtype(aval_out.dtype):
    return aval_out.dtype._rules.dynamic_update_slice_mlir(
        ctx, aval_out, x, update, *start_indices)

  # TODO(necula): handle dynamic shapes
  return hlo.DynamicUpdateSliceOp(x, update, start_indices).result

def pad(ctx: LoweringRuleContext, aval_out,
        x, padding_value,
        padding_low, padding_high, padding_interior) -> ir.Value:
  if all(core.is_constant_shape(s) for s in (padding_low,
                                             padding_high, padding_interior)):
    return hlo.PadOp(x, padding_value,
                     dense_int_elements(padding_low),
                     dense_int_elements(padding_high),
                     dense_int_elements(padding_interior)).result
  else:
    padding_low = shape_tensor(eval_dynamic_shape(ctx, padding_low))
    padding_high = shape_tensor(eval_dynamic_shape(ctx, padding_high))
    padding_interior = shape_tensor(eval_dynamic_shape(ctx, padding_interior))
    return hlo.DynamicPadOp(
        aval_to_ir_type(aval_out),
        x, padding_value, padding_low, padding_high, padding_interior).result

def full_like_aval(ctx: LoweringRuleContext, value, aval: core.ShapedArray) -> ir.Value:
  """Returns an IR constant shaped full of `value` shaped like `aval`."""
  zero = ir_constant(np.array(value, aval.dtype))
  return broadcast_in_dim(ctx, zero, aval, broadcast_dimensions=())

def zeros_like_lowering(ctx, x):
  aval, = ctx.avals_in
  assert isinstance(aval, core.ShapedArray), aval
  return [full_like_aval(ctx, 0, aval)]
register_lowering(ad_util.zeros_like_p, zeros_like_lowering)

def add_jaxvals_lowering(ctx, x, y):
  return hlo.AddOp(x, y).results
register_lowering(ad_util.add_jaxvals_p, add_jaxvals_lowering)

register_lowering(ad_util.stop_gradient_p, lambda ctx, x: [x])


def compare_hlo(x, y, direction: str, comparison_type: Optional[str] = None):
  """Creates CompareOp."""
  if comparison_type is None:
    elem_type = ir.RankedTensorType(x.type).element_type
    if ir.IntegerType.isinstance(elem_type):
      comparison_type = ("UNSIGNED" if ir.IntegerType.is_unsigned(elem_type)
                         else "SIGNED")
    else:
      comparison_type = "FLOAT"

  return hlo.CompareOp(
      x,
      y,
      hlo.ComparisonDirectionAttr.get(direction),
      compare_type=hlo.ComparisonTypeAttr.get(comparison_type))

def _minmax_hlo(op, cmp, x, y):
  """Min/max that compares complex values lexicographically as pairs."""
  tensor_type = ir.RankedTensorType(x.type)
  if ir.ComplexType.isinstance(tensor_type.element_type):
    rx = hlo.RealOp(x).result
    ry = hlo.RealOp(y).result
    real_eq = compare_hlo(rx, ry, "EQ", "FLOAT")
    real_cmp = compare_hlo(rx, ry, cmp, "FLOAT")
    imag_cmp = compare_hlo(
        hlo.ImagOp(x).result,
        hlo.ImagOp(y).result, cmp, "FLOAT")
    which = hlo.SelectOp(real_eq, imag_cmp, real_cmp).result
    return hlo.SelectOp(which, x, y)
  else:
    return op(x, y)

min_hlo = partial(_minmax_hlo, hlo.MinOp, "LT")
max_hlo = partial(_minmax_hlo, hlo.MaxOp, "GT")


def convert_hlo(ctx: LoweringRuleContext, x, aval_in, aval_out):
  """Variant of convert that has HLO semantics.

  In particular, treat casts to boolean as x != 0, rather than truncating
  integer values (b/209440332)."""
  if (not core.is_opaque_dtype(aval_out.dtype) and
      aval_out.dtype == np.dtype(np.bool_)):
    if dtypes.issubdtype(aval_in.dtype, np.inexact):
      compare_type = "FLOAT"
    elif dtypes.issubdtype(aval_in.dtype, np.signedinteger):
      compare_type = "SIGNED"
    else:
      compare_type = "UNSIGNED"
    return compare_hlo(x, full_like_aval(ctx, 0, aval_in), "NE",
                       compare_type).result
  return hlo.ConvertOp(aval_to_ir_type(aval_out), x).result

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
  op = hlo.CustomCallOp([result_type], [x],
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
  return _wrap_with_spmd_op("Sharding", x.type, x, sharding_proto,
                            unspecified_dims)

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
    key = (ctx.primitive, tuple(ctx.platforms), tuple(ctx.avals_in), tuple(ctx.avals_out),
           tuple(params.items()))
    try:
      func = ctx.module_context.cached_primitive_lowerings.get(key)
    except TypeError:
      # If the parameters aren't hashable, give up on caching.
      # TODO(phawkins): switch to requiring hashability, when XLA fallback
      # computations have been ported to MLIR.
      return f(ctx, *args, **params)
    if func is None:
      func = _emit_lowering_rule_as_fun(partial(f, **params), ctx)
      ctx.module_context.cached_primitive_lowerings[key] = func

    output_types = map(aval_to_ir_types, ctx.avals_out)
    args = tuple(ctx.dim_var_values) + args
    flat_output_types = util.flatten(output_types)
    call = func_dialect.CallOp(flat_output_types,
                               ir.FlatSymbolRefAttr.get(func.name.value),
                               flatten_lowering_ir_args(args))
    return util.unflatten(call.results, map(len, output_types))
  return cached_lowering



def xla_computation_to_mlir_module(xla_computation: xc.XlaComputation
                                  ) -> ir.Module:
  module_str = xc._xla.mlir.xla_computation_to_mlir_module(xla_computation)
  return ir.Module.parse(module_str)

def merge_mlir_modules(dst_module: ir.Module,
                       sym_name: str,
                       src_module: ir.Module) -> str:
  """Returns the name of src_module's main() function, after renaming."""
  callee_name = None
  assert dst_module.context == src_module.context
  dst_symtab = ir.SymbolTable(dst_module.operation)

  n = len(dst_module.body.operations)
  for op in src_module.body.operations:
    dst_module.body.append(op)
  ops = list(dst_module.body.operations)[n:]

  for op in ops:
    op = typing.cast(func_dialect.FuncOp, op)
    old_name = op.name.value
    if op.name.value == "main":
      dst_symtab.set_symbol_name(op, sym_name)
      op.attributes["sym_visibility"] = ir.StringAttr.get("private")
      callee_name = ir.StringAttr(dst_symtab.insert(op)).value
      new_name = callee_name
    else:
      new_name = ir.StringAttr(dst_symtab.insert(op)).value

    # Replace references to the symbol with the new name
    for other_op in ops:
      dst_symtab.replace_all_symbol_uses(
          old_name, new_name, other_op.operation)


  assert callee_name is not None
  return callee_name


def xla_fallback_lowering(prim: core.Primitive):
  @cache_lowering
  def fallback(ctx: LoweringRuleContext, *args, **params):
    module_ctx = ctx.module_context
    axis_ctx = module_ctx.axis_context
    if isinstance(axis_ctx, SPMDAxisContext):
      axis_env = axis_ctx.unsafe_axis_env
    else:
      axis_env = module_ctx.axis_env

    if any(hasattr(a, "shape") and
           not core.is_constant_shape(a.shape) for a in (ctx.avals_in + ctx.avals_out)):
      raise NotImplementedError(
          f"Shape polymorphism for xla_fallback_lowering is not implemented ({ctx.primitive}); b/261682623")

    xla_computation = xla.primitive_subcomputation(
        module_ctx.platforms, axis_env, prim, ctx.avals_in,
        ctx.avals_out, **params)
    xla_module = xla_computation_to_mlir_module(xla_computation)
    callee_name = merge_mlir_modules(
        module_ctx.module, f"xla_fallback_{prim.name}", xla_module)
    output_types = map(aval_to_ir_types, ctx.avals_out)
    flat_output_types = util.flatten(output_types)
    output_type = (ir.TupleType.get_tuple(flat_output_types)
                   if prim.multiple_results else flat_output_types[0])

    call = func_dialect.CallOp([output_type],
                               ir.FlatSymbolRefAttr.get(callee_name),
                               flatten_lowering_ir_args(args)).result
    if not prim.multiple_results:
      return [call]
    flat_results = [hlo.GetTupleElementOp(call, i32_attr(i)).result
                    for i in range(len(flat_output_types))]

    return util.unflatten(flat_results, map(len, output_types))
  return fallback

register_lowering(ad.custom_lin_p, ad.raise_custom_vjp_error_on_jvp)

DEVICE_TO_DEVICE_TYPE = 1
SEND_TO_HOST_TYPE = 2
RECV_FROM_HOST_TYPE = 3

_dtype_to_xla_type_string_map = {
    np.dtype("bool"): "pred",
    np.dtype("float16"): "f16",
    np.dtype("float32"): "f32",
    np.dtype("float64"): "f64",
    np.dtype("int8"): "s8",
    np.dtype("uint8"): "u8",
    np.dtype("int16"): "s16",
    np.dtype("uint16"): "u16",
    np.dtype("int32"): "s32",
    np.dtype("uint32"): "u32",
    np.dtype("int64"): "s64",
    np.dtype("uint64"): "u64",
    dtypes._bfloat16_dtype: "bf16",
    np.dtype("complex64"): "c64",
    np.dtype("complex128"): "c128",
}

def _dtype_to_xla_type_string(dtype: np.dtype) -> str:
  if dtype not in _dtype_to_xla_type_string_map:
    raise NotImplementedError(dtype)
  return _dtype_to_xla_type_string_map[dtype]

def send_to_host(channel: int, token: hlo.TokenType, operand: Any,
                 aval: core.ShapedArray, name: str, *,
                 sharding: Optional[xc.OpSharding] = None) -> ir.Value:
  channel_handle = hlo.ChannelHandle.get(channel, SEND_TO_HOST_TYPE)
  send_op = hlo.SendOp([operand], token, channel_handle,
                        is_host_transfer=ir.BoolAttr.get(True))
  dtype_str = _dtype_to_xla_type_string(aval.dtype)
  if dtype_str in {"f64", "s64", "u64", "c64", "c128"}:
    raise NotImplementedError("64-bit types not supported.")
  send_op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      dict(
          _xla_host_transfer_handler_name=ir.StringAttr.get(str(name)),
          _xla_host_transfer_original_type=ir.StringAttr.get(dtype_str),
          _xla_host_transfer_rendezvous=ir.StringAttr.get(str(name))))
  if sharding is not None:
    set_sharding(send_op, sharding)
  return send_op.result


def receive_from_host(channel: int, token: hlo.TokenType,
                      out_aval: core.ShapedArray, name: str, *,
                      sharding: Optional[xc.OpSharding] = None) -> ir.Value:
  channel_handle = hlo.ChannelHandle.get(channel, RECV_FROM_HOST_TYPE)
  recv_op = hlo.RecvOp([aval_to_ir_type(out_aval),
                        hlo.TokenType.get()], token, channel_handle,
                        is_host_transfer=ir.BoolAttr.get(True))
  dtype_str = _dtype_to_xla_type_string(out_aval.dtype)
  if dtype_str in {"f64", "s64", "u64", "c64", "c128"}:
    raise NotImplementedError("64-bit types not supported.")
  recv_op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      dict(
          _xla_host_transfer_handler_name=ir.StringAttr.get(str(name)),
          _xla_host_transfer_original_type=ir.StringAttr.get(dtype_str),
          _xla_host_transfer_rendezvous=ir.StringAttr.get(str(name))))
  if sharding is not None:
    set_sharding(recv_op, sharding)
  # Token should be at the end of the results
  result, token = recv_op.results
  return token, result


def _emit_tpu_python_callback(
    backend: xb.XlaBackend,
    ctx: LoweringRuleContext,
    callback,
    token: Optional[Any],
    operands: List[ir.Value],
    operand_avals: List[core.ShapedArray],
    operand_shapes: List[xc.Shape],
    result_avals: List[core.ShapedArray],
    result_shapes: List[xc.Shape],
    *,
    sharding: Optional[xc.OpSharding] = None
) -> Tuple[List[ir.Value], Any, Any]:
  token = token or hlo.CreateTokenOp().result
  _wrapped_callback = callback

  send_channels = []
  if not operand_avals:
    # If there are no operands to the callback, we need to insert a dummy send
    # op or the callback will never be triggered!
    # TODO(sharadmv,chky): Enable this fix in the runtime as opposed to in
    # MLIR builder.
    callback_without_args = _wrapped_callback
    def _wrapped_callback(*args):  # pylint: disable=function-redefined
      del args
      return callback_without_args()
    send_channel = ctx.module_context.new_channel()
    dummy_send_aval = core.ShapedArray((1,), np.float32)
    dummy_send_val = ir_constant(np.zeros(1, np.float32))
    operand_shapes = [*operand_shapes,
                      xla.aval_to_xla_shapes(dummy_send_aval)[0]]
    token = send_to_host(send_channel, token, dummy_send_val, dummy_send_aval,
                         callback.__name__, sharding=sharding)
    send_channels.append(send_channel)
  else:
    for operand, operand_aval in zip(operands, operand_avals):
      if any(s == 0 for s in operand_aval.shape):
        raise NotImplementedError(
            "Callbacks with zero-dimensional values not supported on TPU.")
      channel = ctx.module_context.new_channel()
      token = send_to_host(channel, token, operand, operand_aval,
                           callback.__name__, sharding=sharding)
      send_channels.append(channel)

  recv_channels = []
  outputs = []
  for result_aval in result_avals:
    if any(s == 0 for s in result_aval.shape):
      raise NotImplementedError(
          "Callbacks with zero-dimensional values not supported on TPU.")
    channel = ctx.module_context.new_channel()
    assert isinstance(result_aval, core.ShapedArray)
    token, out = receive_from_host(channel, token, result_aval,
                                   callback.__name__, sharding=sharding)
    outputs.append(out)
    recv_channels.append(channel)
  opaque = backend.make_python_callback_from_host_send_and_recv(
      _wrapped_callback, operand_shapes, result_shapes, send_channels,
      recv_channels)
  ctx.module_context.add_host_callback(opaque)
  return outputs, token, opaque

def _layout_to_mlir_layout(minor_to_major: Optional[Sequence[int]]):
  if minor_to_major is None:
    # Needed for token layouts
    layout = np.zeros((0,), dtype="int64")
  else:
    layout = np.array(minor_to_major, dtype="int64")
  return ir.DenseIntElementsAttr.get(layout, type=ir.IndexType.get())

def _aval_to_default_layout(aval):
  # Row major order is default for `NumPy`.
  return list(range(aval.ndim - 1, -1, -1))

def emit_python_callback(
    ctx: LoweringRuleContext, callback, token: Optional[Any],
    operands: List[ir.Value], operand_avals: List[core.ShapedArray],
    result_avals: List[core.ShapedArray],
    has_side_effect: bool, *, sharding: Optional[xc.OpSharding] = None,
    operand_layouts: Optional[Sequence[Optional[Sequence[int]]]] = None,
    result_layouts: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> Tuple[List[ir.Value], Any, Any]:
  """Emits MLIR that calls back to a provided Python function."""
  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering in emit_python_callback")
  platform = ctx.module_context.platforms[0]
  if platform not in {"cpu", "cuda", "rocm", "tpu"}:
    raise ValueError(
        f"`EmitPythonCallback` not supported on {platform} backend.")
  backend = ctx.module_context.backend
  result_shapes = util.flatten(
      [xla.aval_to_xla_shapes(result_aval) for result_aval in result_avals])
  operand_shapes = util.flatten(
      [xla.aval_to_xla_shapes(op_aval) for op_aval in operand_avals])
  # Handling layouts
  if operand_layouts is None:
    operand_layouts = map(_aval_to_default_layout, operand_avals)
  operand_mlir_layouts = [
      _layout_to_mlir_layout(_aval_to_default_layout(layout)) if layout is None
      else _layout_to_mlir_layout(layout) for layout, aval
      in zip(operand_layouts, operand_avals)]
  if result_layouts is None:
    result_layouts = map(_aval_to_default_layout, result_avals)
  result_mlir_layouts = [
      _layout_to_mlir_layout(_aval_to_default_layout(aval)) if layout is None
      else _layout_to_mlir_layout(layout) for layout, aval
      in zip(result_layouts, result_avals)]

  # First we apply checks to ensure output shapes and dtypes match the expected
  # ones.
  def _wrapped_callback(*args):
    out_vals = callback(*args)
    if len(out_vals) != len(result_avals):
      raise RuntimeError(
          "Mismatched number of outputs from callback. "
          "Expected: {}, Actual: {}".format(len(result_avals), len(out_vals)))
    for i, (out_val, out_aval) in enumerate(zip(out_vals, result_avals)):
      if out_val.shape != out_aval.shape:
        raise RuntimeError(
            f"Incorrect output shape for return value {i}: "
            "Expected: {}, Actual: {}".format(out_aval.shape, out_val.shape))
      if out_val.dtype != out_aval.dtype:
        raise RuntimeError(
            f"Incorrect output dtype for return value {i}: "
            "Expected: {}, Actual: {}".format(out_aval.dtype, out_val.dtype))
    return out_vals

  if platform == "tpu":
    return _emit_tpu_python_callback(backend, ctx, _wrapped_callback,  token,
        operands, operand_avals, operand_shapes, result_avals, result_shapes,
        sharding=sharding)
  result_types = util.flatten([aval_to_ir_types(aval) for aval in result_avals])
  if token:

    callback_without_token = _wrapped_callback
    def _wrapped_callback(token, *args):  # type: ignore  # pylint: disable=function-redefined
      return (token, *callback_without_token(*args))

    operand_shapes = [
        xla.aval_to_xla_shapes(core.abstract_token)[0], *operand_shapes
    ]
    result_shapes = [
        xla.aval_to_xla_shapes(core.abstract_token)[0], *result_shapes
    ]
    operands = [token, *operands]
    result_types = [token_type()[0], *result_types]
    operand_mlir_layouts = [_layout_to_mlir_layout(None), *operand_mlir_layouts]
    result_mlir_layouts = [_layout_to_mlir_layout(None), *result_mlir_layouts]
  callback_descriptor, keepalive = (
      backend.get_emit_python_callback_descriptor(_wrapped_callback,
                                                  operand_shapes,
                                                  result_shapes))
  descriptor_operand = ir_constant(
      callback_descriptor, canonicalize_types=False)
  callback_operands = [descriptor_operand, *operands]
  if operand_mlir_layouts is not None:
    operand_mlir_layouts = [_layout_to_mlir_layout([]), *operand_mlir_layouts]
  result_type = ir.TupleType.get_tuple(result_types)
  call_target_name = ("xla_python_gpu_callback"
                     if platform in {"cuda", "rocm"} else "xla_python_cpu_callback")
  result = hlo.CustomCallOp(
      [result_type],
      callback_operands,
      call_target_name=ir.StringAttr.get(call_target_name),
      has_side_effect=ir.BoolAttr.get(has_side_effect),
      api_version=i32_attr(2),
      called_computations=ir.ArrayAttr.get([]),
      backend_config=ir.StringAttr.get(str(callback_descriptor)),
      operand_layouts=(
        None if operand_mlir_layouts is None
        else ir.ArrayAttr.get(operand_mlir_layouts)),
      result_layouts=(
        None if result_mlir_layouts is None
        else ir.ArrayAttr.get(result_mlir_layouts)))
  if sharding is not None:
    set_sharding(result, sharding)
  results = [
      hlo.GetTupleElementOp(result, i32_attr(i)).result
      for i in range(len(result_types))
  ]
  if token:
    token, *results = results
  return results, token, keepalive

def build_xla_computation_helper(
    closed_jaxpr: core.ClosedJaxpr, *, name: str, platforms: Sequence[str],
    backend_or_name: str, axis_context: AxisContext) -> xc.XlaComputation:
  """Helper to generate pmap-style XLA computations for custom partitioners."""
  if closed_jaxpr.effects:
    raise NotImplementedError()
  lowering_result = lower_jaxpr_to_module(name, closed_jaxpr,
      backend_or_name=backend_or_name, unordered_effects=[], ordered_effects=[],
      name_stack=source_info_util.NameStack(),
      donated_args=[False] * len(closed_jaxpr.jaxpr.invars),
      axis_context=axis_context, platforms=platforms)
  return xc._xla.mlir.mlir_module_to_xla_computation(
      module_to_string(lowering_result.module), use_tuple_args=False,
      return_tuple=False)

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

# # CHLO doesn't have a legalization for bf16 (b/203774470)
# lax.erf_p,
# lax.erfc_p,
