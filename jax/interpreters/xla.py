# Copyright 2018 Google LLC
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

# Lowering of jaxprs into XLA (HLO) computations.

from collections import defaultdict, deque
import collections.abc
import dataclasses
import functools
from functools import partial
import itertools as it
import operator
import re
from typing import (Any, Callable, Deque, Dict, List, NamedTuple, Optional,
                    Sequence, Set, Type, Tuple, Union)
from typing_extensions import Protocol
import warnings

import numpy as np

from jax.config import config
from jax import core
from jax._src import ad_util
from jax._src import device_array
from jax._src import dtypes
from jax._src import profiler
from jax import linear_util as lu
from jax._src import source_info_util
from jax._src.abstract_arrays import (make_shaped_array, array_types)
from jax.core import (ConcreteArray, ShapedArray,
                      Literal, pp_eqn_compact, JaxprPpContext,
                      abstract_token)
import jax._src.pretty_printer as pp
from jax._src import util
from jax._src.util import (prod, extend_name_stack, wrap_name,
                           safe_zip, safe_map, partition_list)
from jax._src.lib import xla_client as xc
from jax.interpreters import partial_eval as pe
from jax.interpreters import ad

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

xe = xc._xla
xops = xc._xla.ops

# Types
Backend = xe.Client
Device = xc.Device
Buffer = xe.Buffer

XlaOp = xc.XlaOp
XlaShape = xc.Shape
XlaBuilder = xc.XlaBuilder
XlaExecutable = xc.Executable

# apply_primitive is defined in jax._src.dispatch.
apply_primitive: Callable
backend_compile: Callable
device_put: Callable

# TODO(phawkins): update code to point to new locations.
DeviceArray = device_array.DeviceArray
_DeviceArray = device_array._DeviceArray
_CppDeviceArray = xe.Buffer
make_device_array = device_array.make_device_array


def identity(x): return x

_scalar_types = dtypes.python_scalar_dtypes.keys()

# unit representation
def _make_unit_constant(c): return [
    xops.Constant(c, np.zeros((), dtype=np.dtype('bool')))]
def _make_unit_shape(_): return (xc.Shape.array_shape(np.dtype('bool'), ()),)
def _make_array_shape(a: ShapedArray) -> Sequence[XlaShape]:
  if a.dtype is dtypes.float0:
    return (xc.Shape.array_shape(np.dtype('bool'), a.shape),)
  else:
    return (xc.Shape.array_shape(a.dtype, a.shape),)

def _get_canonical_source_file(frame: source_info_util.Frame):
  source_file = frame.file_name
  if config.jax_hlo_source_file_canonicalization_regex:
    source_file = re.sub(config.jax_hlo_source_file_canonicalization_regex,
                         '', source_file)
  return source_file

tracebacks = {}
def make_op_metadata(primitive: core.Primitive,
                     params: Dict, *,
                     source_info: source_info_util.SourceInfo,
                     name_stack: str = "",
                     ) -> xc.OpMetadata:
  eqn_str = str(pp.text(name_stack) +
                pp_eqn_compact(primitive.name, params, JaxprPpContext()))
  tracebacks[eqn_str] = source_info.traceback
  frame = source_info_util.user_frame(source_info) if source_info else None
  return xc.OpMetadata(
        op_type=primitive.name,
        op_name=eqn_str,
        source_file=_get_canonical_source_file(frame) if frame else None,
        source_line=frame.line_num if frame else None)

# Utilities

def parameter(builder, num, shape, name=None, replicated=None):
  if name is None:
    name = ''
  if replicated is None:
    replicated = []
  elif isinstance(replicated, bool):
    replicated = [replicated] * shape.leaf_count()

  return xops.Parameter(builder, num,
                        shape.with_major_to_minor_layout_if_absent(), name,
                        replicated)

# HLO instructions optionally can be annotated to say how the output should be
# spatially partitioned (represented in XLA as OpSharding protos, see
# sharding_to_proto). For array outputs, the annotation is either an int per
# dimension specifying the number of ways that dimension divided (i.e. the total
# number of shards is the product), or None to indicate the array should be
# replicated. Tuple outputs are represented as tuples thereof. XLA supports
# arbitrary tuple nesting, but JAX only uses one level of tupling (and our type
# checkers don't support recursive types), so we only represent one level of
# nesting in this type definition.
SpatialSharding = Union[Tuple[int, ...],
                        None,
                        Tuple[Optional[Tuple[int, ...]], ...]]

def sharding_to_proto(sharding: SpatialSharding):
  """Converts a SpatialSharding to an OpSharding.

  See
  https://github.com/tensorflow/tensorflow/blob/main/tensorflow/compiler/xla/xla_data.proto#L601
  for details on the OpSharding proto.
  """
  proto = xc.OpSharding()
  if isinstance(sharding, tuple) and not isinstance(sharding[0], int):
    assert all(s is None or isinstance(s, tuple) for s in sharding)
    return tuple_sharding_proto(list(map(sharding_to_proto, sharding)))  # type: ignore

  if sharding is None:
    proto.type = xc.OpSharding.Type.REPLICATED
  else:
    proto.type = xc.OpSharding.Type.OTHER
    proto.tile_assignment_dimensions = list(sharding)
    proto.tile_assignment_devices = list(range(np.product(sharding)))
  return proto

def tuple_sharding_proto(elems):
  proto = xc.OpSharding()
  assert all(isinstance(e, type(proto)) for e in elems)
  proto.type = xc.OpSharding.Type.TUPLE
  proto.tuple_shardings = elems
  return proto

def set_sharding_proto(builder, op, sharding_proto):
  """Uses CustomCall to annotate a value as sharded."""
  # "Sharding" is a built-in custom call target that acts like an identity
  # function, and is used to attach an OpSharding to.
  return with_sharding_proto(builder, sharding_proto, xops.CustomCall,
                             builder, b"Sharding", [op], builder.get_shape(op))

def with_sharding_proto(builder, sharding_proto, op_fn, *args, **kwargs):
  """Builds op_fn(*args, **kwargs) with sharding annotation."""
  builder.set_sharding(sharding_proto)
  try:
    return op_fn(*args, **kwargs)
  finally:
    builder.clear_sharding()

def set_sharding(builder, op, sharding: SpatialSharding):
  """Uses CustomCall to annotate a value as sharded."""
  return set_sharding_proto(builder, op, sharding_to_proto(sharding))

def with_sharding(builder, sharding: SpatialSharding, op_fn, *args, **kwargs):
  """Builds op_fn(*args, **kwargs) with sharding annotation."""
  return with_sharding_proto(builder, sharding_to_proto(sharding), op_fn, *args,
                             **kwargs)


### handlers

# Numpy dtypes -> XLA primitive types

_dtype_to_primitive_type: Dict[np.dtype, xc.PrimitiveType] = {
  np.dtype('bool'): xc.PrimitiveType.PRED,
  np.dtype('int8'): xc.PrimitiveType.S8,
  np.dtype('int16'): xc.PrimitiveType.S16,
  np.dtype('int32'): xc.PrimitiveType.S32,
  np.dtype('int64'): xc.PrimitiveType.S64,
  np.dtype('uint8'): xc.PrimitiveType.U8,
  np.dtype('uint16'): xc.PrimitiveType.U16,
  np.dtype('uint32'): xc.PrimitiveType.U32,
  np.dtype('uint64'): xc.PrimitiveType.U64,
  np.dtype(dtypes.bfloat16): xc.PrimitiveType.BF16,
  np.dtype('float16'): xc.PrimitiveType.F16,
  np.dtype('float32'): xc.PrimitiveType.F32,
  np.dtype('float64'): xc.PrimitiveType.F64,
  np.dtype('complex64'): xc.PrimitiveType.C64,
  np.dtype('complex128'): xc.PrimitiveType.C128,
}

def dtype_to_primitive_type(dtype: np.dtype) -> xc.PrimitiveType:
  """Converts a NumPy dtype into an XLA PrimitiveType."""
  # Many things (e.g., strings, scalar types) can be compared with NumPy dtypes,
  # but may not hash correctly. Make sure we have a true np.dtype.
  assert isinstance(dtype, np.dtype), type(dtype)
  try:
    return _dtype_to_primitive_type[dtype]
  except KeyError as err:
    raise TypeError(f"No XLA lowering for NumPy dtype: {dtype}") from err


# JAX abstract values -> XLA shapes

def aval_to_xla_shapes(aval: core.AbstractValue) -> Sequence[XlaShape]:
  try:
    return xla_shape_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No xla_shape_handler for type: {type(aval)}") from err

xla_shape_handlers: Dict[Type[core.AbstractValue],
                         Callable[[Any], Sequence[XlaShape]]] = {
    core.AbstractUnit: _make_unit_shape,
    ShapedArray: _make_array_shape,
    ConcreteArray: _make_array_shape,
}
xla_shape_handlers[core.AbstractToken] = lambda _: (xc.Shape.token_shape(),)



# IR constants

_constant_handlers: Dict[type, Callable] = {}

def pyval_to_ir_constants(builder, py_val, canonicalize_types=True):
  """Translate a general constant `py_val` to a constant, canonicalizing its dtype.

  Args:
    py_val: a Python value to be translated to a constant.

  Returns:
    A representation of the constant as a list of xla ops.
  """
  for t in type(py_val).__mro__:
    handler = _constant_handlers.get(t)
    if handler: return handler(builder, py_val, canonicalize_types)
  if hasattr(py_val, '__jax_array__'):
    return pyval_to_ir_constants(builder, py_val.__jax_array__(),
                                 canonicalize_types)
  raise TypeError("No constant handler for type: {}".format(type(py_val)))

def pyval_to_ir_constant(builder, py_val, canonicalize_types=True):
  """Translate constant `py_val` to a constant, canonicalizing its dtype.

  Args:
    py_val: a Python value to be translated to a constant.

  Returns:
    A representation of the constant, either a ComputationDataHandle or None
  """
  const = pyval_to_ir_constants(builder, py_val, canonicalize_types=canonicalize_types)
  assert len(const) == 1, f"Internal error: cannot create constant from object of type {type(py_val)}"
  return const[0]


def register_constant_handler(type_, handler_fun):
  _constant_handlers[type_] = handler_fun

register_constant_handler(core.Unit, lambda c, *_: _make_unit_constant(c))


# TODO(mattjj,frostig): try to remove this function
def _normalize_to_xla_dtypes(val):
  """Normalize dtypes in a value."""
  if hasattr(val, '__array__') or np.isscalar(val):
    return np.asarray(val, dtype=dtypes.canonicalize_dtype(dtypes.result_type(val)))
  elif isinstance(val, (tuple, list)):
    return tuple(_normalize_to_xla_dtypes(x) for x in val)
  raise TypeError('Can\'t convert to XLA: {}'.format(val))

def _numpy_array_constant(builder, value, canonicalize_types=True):
  if canonicalize_types:
    value = _normalize_to_xla_dtypes(value)
  return [xops.Constant(builder, value)]


def _ndarray_constant_handler(c, val, canonicalize_types=True):
  """Constant handler for ndarray literals, handling zero-size strides.

  This function essentially calls _numpy_array_constant(val) except it has
  special handling of arrays with any strides of size zero: for those, it
  generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
  to avoid staging in large literals that might arise from np.zeros or np.ones
  or the output of lax.broadcast (which uses np.broadcast_to which in turn
  uses size-zero strides).

  Args:
    c: an XlaBuilder
    val: an ndarray.

  Returns:
    An XLA ComputationDataHandle / XlaOp representing the constant ndarray
    staged into the XLA Computation.
  """
  # TODO(mattjj): revise this to use xops.BroadcastInDim rather than Transpose
  if dtypes.result_type(val) == dtypes.float0:
    return _numpy_array_constant(c, np.zeros(val.shape, dtype=np.bool_))
  elif np.any(np.equal(0, val.strides)) and val.size > 0:
    zero_stride_axes, = np.where(np.equal(0, val.strides))
    other_axes, = np.where(np.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)
                              for ax in range(val.ndim))]
    xla_val = xops.Broadcast(
        _numpy_array_constant(c, collapsed_val, canonicalize_types)[0],
        np.take(val.shape, zero_stride_axes))
    permutation = np.argsort(tuple(zero_stride_axes) + tuple(other_axes))
    return [xops.Transpose(xla_val, permutation)]
  else:
    return _numpy_array_constant(c, val, canonicalize_types)
register_constant_handler(np.ndarray, _ndarray_constant_handler)


def _scalar_constant_handler(c, val, canonicalize_types=True):
  return _numpy_array_constant(c, val, canonicalize_types)

for scalar_type in [np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float16, np.float32, np.float64,
                    np.bool_, np.longlong,
                    dtypes.bfloat16]:
  register_constant_handler(scalar_type, _scalar_constant_handler)

# https://github.com/winpython/winpython/issues/613#issuecomment-380121523
if hasattr(np, "float128"):
  register_constant_handler(np.float128, _scalar_constant_handler)

def _python_scalar_handler(dtype, c, val, canonicalize_dtypes=True):
  return _numpy_array_constant(c, dtype.type(val))

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_constant_handler(ptype, partial(_python_scalar_handler, dtype))

def _device_array_constant_handler(c, val, canonicalize_types=True):
  return pyval_to_ir_constants(c, val.device_buffer.to_py())
for t in device_array.device_array_types:
  register_constant_handler(t, _device_array_constant_handler)


# TODO(mattjj): try to remove this canonicalize_dtype stuff
def canonicalize_dtype(x):
  typ = type(x)
  handler = canonicalize_dtype_handlers.get(typ)
  if handler: return handler(x)
  for typ in typ.__mro__:
    handler = canonicalize_dtype_handlers.get(typ)
    if handler: return handler(x)
  if hasattr(x, '__jax_array__'):
    return canonicalize_dtype(x.__jax_array__())
  raise TypeError(f"No canonicalize_dtype handler for type: {type(x)}")

def _canonicalize_ndarray_dtype(x):
  return np.asarray(x, dtypes.canonicalize_dtype(dtypes.result_type(x)))

def _canonicalize_python_scalar_dtype(typ, x):
  return np.asarray(
      x, dtypes.canonicalize_dtype(dtypes._scalar_type_to_dtype(typ, x)))

canonicalize_dtype_handlers: Dict[Any, Callable] = {core.Unit: identity}
for t in device_array.device_array_types:
  canonicalize_dtype_handlers[t] = lambda x: x
canonicalize_dtype_handlers.update(
    (t, _canonicalize_ndarray_dtype) for t in array_types)
canonicalize_dtype_handlers.update(
    (t, partial(_canonicalize_python_scalar_dtype, t)) for t in _scalar_types)
canonicalize_dtype_handlers[core.Token] = lambda x: x

def abstractify(x) -> core.AbstractValue:
  typ = type(x)
  aval_fn = pytype_aval_mappings.get(typ)
  if aval_fn: return aval_fn(x)
  for typ in typ.__mro__:
    aval_fn = pytype_aval_mappings.get(typ)
    if aval_fn: return aval_fn(x)
  if hasattr(x, '__jax_array__'):
    return abstractify(x.__jax_array__())
  raise TypeError(f"Argument '{x}' of type '{type(x)}' is not a valid JAX type")

def _make_abstract_python_scalar(typ, val):
  return ShapedArray((), dtypes._scalar_type_to_dtype(typ, val), weak_type=True)

pytype_aval_mappings: Dict[Any, Callable[[Any], core.AbstractValue]] = {
    core.Unit: lambda _: core.abstract_unit,
}
for t in device_array.device_array_types:
  pytype_aval_mappings[t] = operator.attrgetter('aval')
pytype_aval_mappings[core.Token] = lambda _: core.abstract_token
pytype_aval_mappings.update((t, make_shaped_array) for t in array_types)
pytype_aval_mappings.update(
    (t, partial(_make_abstract_python_scalar, t)) for t in _scalar_types)


def primitive_subcomputation(platform: str, axis_env: 'AxisEnv',
                             prim: core.Primitive,
                             *avals: core.AbstractValue, **params):
  c = xc.XlaBuilder(f"primitive_computation_{prim.name}")
  f = lower_fun(prim.bind, multiple_results=prim.multiple_results,
                new_style=True)
  xla_args, _ = _xla_callable_args(c, avals, tuple_args=False,
                                   filter_tokens=False)
  ctx = TranslationContext(builder=c, platform=platform, axis_env=axis_env,
                           name_stack="")
  ans = f(ctx.replace(builder=c), avals, None, *xla_args, **params)
  if prim.multiple_results:
    ans = xops.Tuple(c, ans)
  else:
    ans, = ans
  return c.build(ans)


# Used within _xla_callable_args and _xla_param to distinguish between None (no
# sharding annotation set) and replicated.
_replicated_param = object()

def _token_param_shape():
  """Shape used in place of tokens as top-level computation arguments."""
  return xc.Shape.array_shape(np.dtype(np.bool_), [])

def _make_token_return_value(c):
  """Value used in place of tokens as a top-level computation return value."""
  return xops.Constant(c, np.zeros((), dtype=np.dtype(np.bool_)))

def _xla_callable_args(
    c, avals, tuple_args, *,
    replicated=None,
    partitions=None,
    partitions_proto: bool = False,
    donated_invars=None,
    filter_tokens=True):
  assert partitions is None or len(partitions) == len(avals)
  if not tuple_args:
    if replicated is None:
      replicated = [None] * len(avals)
    if partitions is None:
      parts: List[object] = [None] * len(avals)
    elif partitions_proto:
      parts = partitions
    else:
      parts = [_replicated_param if part is None else part
               for part in partitions]
    counts = it.count()
    xla_args = [_xla_param(c, next(counts), xla_shape, r, p, partitions_proto,
                           filter_tokens)
                for (a, r, p) in safe_zip(avals, replicated, parts)
                for xla_shape in aval_to_xla_shapes(a)]
    if donated_invars is not None:
      donated_invars = [
          d for (a, _, _, d) in zip(avals, replicated, parts, donated_invars)
          for xla_shape in aval_to_xla_shapes(a)]
    return xla_args, donated_invars
  else:
    if replicated is not None:
      replicated = [r for a, r in zip(avals, replicated)
                    if a is not abstract_token]
    if partitions is None:
      tuple_parts = None
    elif partitions_proto:
      tuple_parts = tuple_sharding_proto(partitions)
    else:
      tuple_parts = tuple(partitions)
    tuple_shape = xc.Shape.tuple_shape(
        [shape if not (filter_tokens and a is abstract_token)
         else _token_param_shape()
         for a in avals for shape in aval_to_xla_shapes(a)])
    tuple_param = _xla_param(c, 0, tuple_shape, replicated, tuple_parts,
                             partitions_proto, filter_tokens)
    xla_args = [v if not (filter_tokens and a is abstract_token)
                else xops.CreateToken(c)
                for a, v in zip(avals, xla_destructure(c, tuple_param))]
    return xla_args, donated_invars

def _xla_param(builder, param_num, xla_shape, replicated, partitions,
               parts_proto, filter_tokens):
  is_token = xla_shape.is_token()
  if filter_tokens and is_token:
    xla_shape = _token_param_shape()
  make_param = partial(parameter, builder, param_num, xla_shape,
                       replicated=replicated)
  with_sharding_fn = with_sharding_proto if parts_proto else with_sharding
  if partitions is None:
    out = make_param()
  elif partitions is _replicated_param:
    out = with_sharding_fn(builder, None, make_param)
  else:
    out = with_sharding_fn(builder, partitions, make_param)
  if filter_tokens and is_token:
    out = xops.CreateToken(builder)
  return out


### compiling jaxprs


def _flatmap(func: Callable, vars: Sequence):
  return list(it.chain.from_iterable(map(func, vars)))

def _partitionmap(func: Callable, vars: Sequence, nodes: Sequence):
  return map(func, vars,
             util.unflatten(nodes,
                            [len(aval_to_xla_shapes(v.aval)) for v in vars]))

class AxisEnv(NamedTuple):
  """Represents a pmap mesh (only along the replica axes)."""
  nreps: int
  names: Tuple[Any, ...]
  sizes: Tuple[int, ...]

@dataclasses.dataclass
class TranslationContext:
  builder: xc.XlaBuilder
  # TODO(phawkins): make platform non-optional. We should always be translating
  # with a specific platform in mind.
  platform: Optional[str]
  axis_env: AxisEnv
  name_stack: str

  def replace(self, **kw): return dataclasses.replace(self, **kw)


def jaxpr_subcomp(ctx: TranslationContext, jaxpr: core.Jaxpr,
                  consts: Sequence[XlaOp], *args: XlaOp) -> Sequence[XlaOp]:
  assert ctx.platform is not None
  def read(v):
    if type(v) is Literal:
      return pyval_to_ir_constants(ctx.builder, canonicalize_dtype(v.val))
    else:
      return env[v]

  def aval(v):
    if type(v) is Literal:
      return abstractify(v.val)
    else:
      return v.aval

  def write(v, node):
    assert node is not None
    env[v] = node

  env: Dict[core.Var, Sequence[XlaOp]] = {}
  _partitionmap(write, [core.unitvar],
                pyval_to_ir_constants(ctx.builder, core.unit))
  _partitionmap(write, jaxpr.constvars, consts)
  _partitionmap(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    op_metadata = make_op_metadata(
        eqn.primitive, eqn.params, name_stack=ctx.name_stack,
        source_info=eqn.source_info)
    ctx.builder.set_op_metadata(op_metadata)
    in_nodes = _flatmap(read, eqn.invars)
    if (ctx.platform is not None and
        eqn.primitive in _backend_specific_translations[ctx.platform]):
      rule = _backend_specific_translations[ctx.platform][eqn.primitive]
    elif eqn.primitive in _translations:
      rule = _translations[eqn.primitive]
    else:
      raise NotImplementedError(
          f"XLA translation rule for primitive '{eqn.primitive.name}' not found")

    with source_info_util.user_context(eqn.source_info.traceback):
      ans = rule(ctx, map(aval, eqn.invars), map(aval, eqn.outvars),
                 *in_nodes, **eqn.params)

    assert isinstance(ans, collections.abc.Sequence), (ans, eqn)
    assert all(isinstance(x, xe.XlaOp) for x in ans), (ans, eqn)
    map(ctx.builder.get_shape, ans)  # force xla to do shape error checking
    ctx.builder.clear_op_metadata()
    _partitionmap(write, eqn.outvars, ans)
  return _flatmap(read, jaxpr.outvars)


def xla_destructure(c, ans):
  num_elements = len(c.get_shape(ans).tuple_shapes())
  return [xops.GetTupleElement(ans, i) for i in range(num_elements)]

def check_backend_matches(inner_backend, outer_backend):
  # For nested calls, the outermost call sets the backend for all inner calls;
  # it's an error if the inner call has a conflicting explicit backend spec.
  if inner_backend and inner_backend != outer_backend:
    raise ValueError(
        f"Outer-jit backend specification {outer_backend} must match explicit "
        f"inner-jit backend specification {inner_backend}.")


def extend_axis_env(env: AxisEnv, name, size: int):
  return AxisEnv(env.nreps, env.names + (name,), env.sizes + (size,))

def axis_read(axis_env, axis_name):
  try:
    return max(i for i, name in enumerate(axis_env.names) if name == axis_name)
  except ValueError:
    raise NameError("unbound axis name: {}".format(axis_name)) from None

def axis_groups(axis_env: AxisEnv, name) -> Tuple[Tuple[int, ...]]:
  if not isinstance(name, (list, tuple)):
    name = (name,)
  mesh_axes = tuple(unsafe_map(partial(axis_read, axis_env), name))
  trailing_size, ragged = divmod(axis_env.nreps, prod(axis_env.sizes))
  assert not ragged
  mesh_spec = axis_env.sizes + (trailing_size,)
  return _axis_groups(mesh_spec, mesh_axes)

def _axis_groups(mesh_spec, mesh_axes):
  """Computes replica group ids for a collective performed over a subset of the mesh.

  Args:
    mesh_spec: A sequence of integers representing the mesh shape.
    mesh_axes: A sequence of integers between 0 and `len(mesh_spec)` (exclusive)
      indicating over which axes the collective is performed.
  Returns:
    A tuple of replica groups (i.e. tuples containing replica ids).
  """
  iota = np.arange(prod(mesh_spec)).reshape(mesh_spec)
  groups = np.reshape(
      np.moveaxis(iota, mesh_axes, np.arange(len(mesh_axes))),
      (prod(np.take(mesh_spec, mesh_axes)), -1))
  return tuple(unsafe_map(tuple, groups.T))


# TODO(mattjj,skyewm): the functions here are utilities for checking if
# not-yet-supported features are used with multi-host programming


def jaxpr_collectives(jaxpr):
  """Generates all the collective primitives anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if eqn.primitive in _collective_primitives:
      yield eqn.primitive
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from jaxpr_collectives(subjaxpr)


### xla_call underlying jit


def flatten_shape(s: XlaShape) -> Sequence[Tuple[Sequence[int], XlaShape]]:
  """Expands a given shape tree into a flat list of indices to arrays.

  Given the following computation:

  >>> c = xc.XlaBuilder("example")
  >>> p0 = parameter(c, 1, xc.shape_from_pyval(jnp.ones([1])))
  >>> p1 = parameter(c, 2, xc.shape_from_pyval(jnp.ones([2])))
  >>> p2 = parameter(c, 3, xc.shape_from_pyval(jnp.ones([3])))
  >>> o = xops.Tuple(c, [p0, p1, p2])

  We can query the arrays in the output tuple:

  >>> flatten_shape(c.GetShape(o))
  [((0,), f32[1]{0}), ((1,), f32[2]{0}), ((2,), f32[3]{0})]

  Or the arrays in one of the parameters (which is itself an array):

  >>> flatten_shape(c.GetShape(p0))
  [((), f32[1]{0})]

  Args
    s: The input shape.

  Returns:
    An iterable of pairs of indices and shapes for each array within the shape
    tree.
  """
  results: List[Tuple[Tuple[int, ...], XlaShape]] = []
  _flatten_shape(s, (), results)
  return results

def _flatten_shape(s: XlaShape, index: Tuple[int, ...],
                   results: List[Tuple[Tuple[int, ...], XlaShape]]) -> None:
  if s.is_array() or s.is_token():
    results.append((index, s))
  else:
    assert s.is_tuple()
    for i, sub in enumerate(s.tuple_shapes()):
      _flatten_shape(sub, index + (i,), results)


def _xla_consts(c, consts):
  unique_consts = {id(const): const for const in consts}
  xla_consts = {
      id_: pyval_to_ir_constants(c, const) for id_, const in unique_consts.items()}
  return [c for const in consts for c in xla_consts[id(const)]]




def set_up_aliases(c, xla_args, out_shape: XlaShape, donated_args, tuple_args):
  """Configures input/output "must" aliasing based on `donated_args`."""
  # First for every input array add it to `donations` iff it is a member of
  # `donated_args`.
  donations: Dict[Tuple[Tuple[int, ...], Any], Deque]
  donations = defaultdict(deque)
  for arg_index, arg in enumerate(xla_args):
    if donated_args[arg_index]:
      for param_index, element in flatten_shape(c.GetShape(arg)):
        key = (element.dimensions(), element.xla_element_type())
        if tuple_args:
          param_number = 0
          param_index = (arg_index,) + tuple(param_index)
          donations[key].append((param_number, param_index, arg_index))
        else:
          param_number = arg_index
          donations[key].append((param_number, param_index, arg_index))

  # Consume donations for outputs.
  out_donated_args = list(donated_args)
  for output_index, element in flatten_shape(out_shape):
    key = (element.dimensions(), element.xla_element_type())
    if donations.get(key, ()):
      param_number, param_index, arg_index = donations[key].popleft()
      out_donated_args[arg_index] = False
      c.setup_alias(output_index, param_number, param_index)

  return tuple(out_donated_args)


@profiler.annotate_function
def lower_jaxpr_to_xla_module(
    fn_name: str, jaxpr: core.ClosedJaxpr, platform: str, axis_env: AxisEnv,
    name_stack: str, tuple_args: bool, donated_invars: Sequence[bool],
    replicated_args: Optional[Sequence[bool]],
    arg_partitions: Optional[Any],
    out_partitions: Optional[Any],
    partitions_are_protos: bool = False
    ) -> xc.XlaComputation:
  """Lowers a closed jaxpr to a top-level XLA module."""
  c = xc.XlaBuilder(fn_name)
  xla_consts = _xla_consts(c, jaxpr.consts)
  xla_args, donated_invars = _xla_callable_args(
      c, jaxpr.in_avals, tuple_args, donated_invars=donated_invars,
      replicated=replicated_args, partitions=arg_partitions,
      partitions_proto=partitions_are_protos)
  ctx = TranslationContext(c, platform, axis_env, name_stack)
  out_nodes = jaxpr_subcomp(ctx, jaxpr.jaxpr, xla_consts, *xla_args)
  # Replace tokens with a dummy array value, because the runtime cannot
  # handle token arguments.
  out_aval_lens = [len(aval_to_xla_shapes(a)) for a in jaxpr.out_avals]
  out_nodes = util.flatten(
      [[_make_token_return_value(c)] if a is core.abstract_token
       else v for a, v in zip(jaxpr.out_avals,
                              util.unflatten(out_nodes, out_aval_lens))])

  # There is a non-zero cost to building an output tuple, particularly on TPU.
  # Avoid it if the output arity is 1.
  if out_partitions is None:
    output = out_nodes[0] if len(out_nodes) == 1 else xc.ops.Tuple(c, out_nodes)
  else:
    build_out_tuple = partial(xops.Tuple, c, out_nodes)
    if partitions_are_protos:
      output = with_sharding_proto(c, out_partitions, build_out_tuple)
    else:
      output = with_sharding(c, out_partitions, build_out_tuple)

  if platform in ("gpu", "tpu"):
    donated_invars = set_up_aliases(
        c, xla_args, c.GetShape(output), donated_invars, tuple_args)
  if any(donated_invars):
    # TODO(tomhennigan): At call time we should mark these buffers as deleted.
    unused_donations = [str(c.GetShape(a))
                        for a, d in zip(xla_args, donated_invars) if d]
    warnings.warn("Some donated buffers were not usable: {}".format(
        ", ".join(unused_donations)))
  return c.build(output)



xla_call_p: core.CallPrimitive = core.CallPrimitive('xla_call')
xla_call = xla_call_p.bind

def _xla_call_partial_eval_update_params(params, in_unknowns):
  call_jaxpr = params['call_jaxpr']
  donated_invars = params['donated_invars']
  if not in_unknowns and donated_invars:
    # JaxprTrace.post_process_call creates a call with no input tracers
    new_donated_invars = (False,) * len(call_jaxpr.invars)
  else:
    # JaxprTrace.process_call drops known input tracers
    donated_invars = [d for d, uk in zip(donated_invars, in_unknowns) if uk]
    new_donated_invars = ((False,) * (len(call_jaxpr.invars) - len(donated_invars))
                          + tuple(donated_invars))
  return dict(params, donated_invars=new_donated_invars)
pe.call_param_updaters[xla_call_p] = _xla_call_partial_eval_update_params

def _xla_call_jvp_update_params(params, nz_tangents, nz_tangents_out_thunk):
  donated_invars = params['donated_invars']
  donated_tangents = [d for d, nz in zip(donated_invars, nz_tangents) if nz]
  new_donated_invars = (*donated_invars, *donated_tangents)
  return dict(params, donated_invars=new_donated_invars)
ad.call_param_updaters[xla_call_p] = _xla_call_jvp_update_params

def _xla_call_transpose_update_params(params, undef_primals, nonzero_cts):
  donated_invars = params['donated_invars']
  donated_primals = [d for d, u in zip(donated_invars, undef_primals) if not u]
  donated_cotangents = [False for nz in nonzero_cts if nz]
  return dict(params, donated_invars=(*donated_primals, *donated_cotangents))
ad.call_transpose_param_updaters[xla_call_p] = _xla_call_transpose_update_params


def _xla_call_translation_rule(ctx, avals_in, avals_out, *in_nodes, name,
                               backend=None, call_jaxpr, donated_invars,
                               inline=None, device=None):
  del device, donated_invars, inline  # Ignored.
  c = ctx.builder
  check_backend_matches(backend, ctx.platform)
  subc = xc.XlaBuilder(f"jit_{name}")
  args = [parameter(subc, i, c.get_shape(n)) for i, n in enumerate(in_nodes)]
  sub_ctx = ctx.replace(
      builder=subc,
      name_stack=extend_name_stack(ctx.name_stack, wrap_name(name, 'jit')))
  out_nodes = jaxpr_subcomp(sub_ctx, call_jaxpr, (), *args)
  subc = subc.build(xops.Tuple(subc, out_nodes))
  return xla_destructure(c, xops.Call(c, subc, list(in_nodes)))
ad.primitive_transposes[xla_call_p] = partial(ad.call_transpose, xla_call_p)


def _xla_call_partial_eval_custom_params_updater(
    unks_in: List[bool], num_res: int, params_known: dict, params_staged: dict
  ) -> Tuple[dict, dict]:
  # pruned inputs to jaxpr_known according to unks_in, so prune donated_invars
  donated_invars_known, _ = partition_list(unks_in, params_known['donated_invars'])
  new_params_known = dict(params_known, donated_invars=tuple(donated_invars_known))
  # added num_res new inputs to jaxpr_staged, so extend donated_invars
  donated_invars_staged = [*([False] * num_res), *params_staged['donated_invars']]
  new_params_staged = dict(params_staged, donated_invars=tuple(donated_invars_staged))
  return new_params_known, new_params_staged
pe.partial_eval_jaxpr_custom_rules[xla_call_p] = \
    partial(pe.call_partial_eval_custom_rule, 'call_jaxpr',
            _xla_call_partial_eval_custom_params_updater)
pe.dce_rules[xla_call_p] = pe.dce_jaxpr_call_rule


def _pp_xla_call(eqn: core.JaxprEqn, context: core.JaxprPpContext
                 ) -> List[pp.Doc]:
  printed_params = {k:v for k, v in eqn.params.items() if
                    k == 'call_jaxpr' or k == 'name' or
                    k == 'backend' and v is not None or
                    k == 'device' and v is not None or
                    k == 'donated_invars' and any(v)}
  return [pp.text(eqn.primitive.name),
          core.pp_kv_pairs(sorted(printed_params.items()), context),
          pp.text(" ") + core.pp_vars(eqn.invars, context)]
core.pp_eqn_rules[xla_call_p] = _pp_xla_call


### translation tables

MYPY = False
if not MYPY:
  class TranslationRule(Protocol):
    def __call__(self, ctx: TranslationContext,
                 avals_in: Sequence[core.AbstractValue],
                 avals_out: Sequence[core.AbstractValue],
                 *args: XlaOp, **kw
                ) -> Sequence[XlaOp]:
      """A translation rule lowers a primitive invocation into an XLA HLO."""
else:
  TranslationRule = Any

_translations: Dict[core.Primitive, TranslationRule] = {}
_backend_specific_translations: Dict[str, Dict[core.Primitive, TranslationRule]]
_backend_specific_translations = defaultdict(dict)

_collective_primitives: Set[core.Primitive] = set()
_initial_style_primitives: Set[core.Primitive] = set()

def register_translation(prim: core.Primitive, rule: TranslationRule, *,
                         platform: Optional[str] = None,
                         is_collective: bool = False,
                         initial_style: bool = False) -> None:
  ts = (_translations if platform is None
        else _backend_specific_translations[platform])
  ts[prim] = rule
  if is_collective:
    _collective_primitives.add(prim)
  if initial_style:
    _initial_style_primitives.add(prim)

# As a temporary backward compatibility measure, we use an adapter class to
# convert from the old styles of translation rules to the newer ones.
# TODO(phawkins): update users of the older translation rule styles and remove
# the adapters.
class _TranslationRuleAdapter:
  def __init__(self, translations,
               wrap_fn: Callable[[core.Primitive, Callable], TranslationRule]):
    self._translations = translations
    self._wrap_fn = wrap_fn

  def __setitem__(self, key: core.Primitive, value: Callable):
    self._translations[key] = self._wrap_fn(key, value)


def _wrap_old_translation(prim: core.Primitive, f: Callable) -> TranslationRule:
  @functools.wraps(f)
  def wrapped(ctx: TranslationContext, avals_in: Sequence[core.AbstractValue],
              avals_out: Sequence[core.AbstractValue],
               *args: XlaOp, **kw) -> Sequence[XlaOp]:
    ans = f(ctx.builder, *args, **kw)
    if (prim.multiple_results or
        any(len(aval_to_xla_shapes(aval)) > 1 for aval in avals_out)):
      return xla_destructure(ctx.builder, ans)
    else:
      return [ans]
  return wrapped


def _wrap_old_call_translation(prim: core.Primitive,
                               f: Callable) -> TranslationRule:
  @functools.wraps(f)
  def wrapped(ctx: TranslationContext, avals_in: Sequence[core.AbstractValue],
              avals_out: Sequence[core.AbstractValue],
               *args: XlaOp, **kw) -> Sequence[XlaOp]:
    platform = kw.pop("backend", None)
    check_backend_matches(platform, ctx.platform)
    ans = f(ctx.builder, ctx.axis_env, args, ctx.name_stack,
            backend=ctx.platform, **kw)
    if (prim.multiple_results or
        any(len(aval_to_xla_shapes(aval)) > 1 for aval in avals_out)):
      return xla_destructure(ctx.builder, ans)
    else:
      return [ans]
  return wrapped

translations : _TranslationRuleAdapter
translations = _TranslationRuleAdapter(_translations, _wrap_old_translation)

class _BackendSpecificTranslationsAdapter(defaultdict):
  def __missing__(self, key):
    ret = self[key] = _TranslationRuleAdapter(
        _backend_specific_translations[key], _wrap_old_translation)
    return ret

backend_specific_translations: Dict[str, _TranslationRuleAdapter]
backend_specific_translations = _BackendSpecificTranslationsAdapter()
call_translations : _TranslationRuleAdapter
call_translations = _TranslationRuleAdapter(
    _translations, _wrap_old_call_translation)



register_translation(xla_call_p, _xla_call_translation_rule)

def zeros_like_translation_rule(c, x):
  shape = c.get_shape(x)
  assert not shape.is_tuple()
  zero = xops.Constant(c, np.array(0, shape.element_type()))
  return xops.Broadcast(zero, shape.dimensions())
translations[ad_util.zeros_like_p] = zeros_like_translation_rule

def add_jaxvals_translation_rule(c, x, y):
  shape = c.get_shape(x)
  assert not shape.is_tuple()
  return xops.Add(x, y)
translations[ad_util.add_jaxvals_p] = add_jaxvals_translation_rule

translations[ad_util.stop_gradient_p] = lambda c, x: x


@lu.transformation
def _tuple_output(*args, **kwargs):
  ans = yield args, kwargs
  yield (ans,)

def lower_fun(fun: Callable, *, multiple_results: bool, backend=None,
              new_style: bool = False) -> Callable:
  if new_style:
    def f_new(ctx: TranslationContext, avals_in: Sequence[core.AbstractValue],
              avals_out: Optional[Sequence[core.AbstractValue]],
              *xla_args: xc.XlaOp,
              **params) -> Sequence[xc.XlaOp]:
      wrapped_fun = lu.wrap_init(fun, params)
      if not multiple_results:
        wrapped_fun = _tuple_output(wrapped_fun)
      with core.extend_axis_env_nd(zip(ctx.axis_env.names, ctx.axis_env.sizes)):
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, avals_in)
      return jaxpr_subcomp(ctx, jaxpr, _xla_consts(ctx.builder, consts),
                           *xla_args)
    return f_new

  # TODO(phawkins): migrate dependent code & always use new_style=True.

  if backend is None:
    # The user didn't specify a backend. This isn't possible with the new style
    # API.
    backend = "backend_not_specified"

  def f(c, *xla_args, **params):
    avals = [_array_aval_from_xla_shape(c.get_shape(x)) for x in xla_args]
    return f_with_avals(c, avals, xla_args, params)

  def f_with_avals(c, avals, xla_args, params):
    # parallelism is only supported via the new-style API.
    axis_env = AxisEnv(1, (), ())
    wrapped_fun = lu.wrap_init(fun, params)
    if not multiple_results:
      wrapped_fun = _tuple_output(wrapped_fun)
    with core.extend_axis_env_nd(zip(axis_env.names, axis_env.sizes)):
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, avals)
    # TODO(issue #8853): not quite right, does not pass the tokens properly
    jaxpr = core.apply_outfeed_rewriter(jaxpr)
    ctx = TranslationContext(c, backend, axis_env, '')
    outs = jaxpr_subcomp(ctx, jaxpr, _xla_consts(c, consts), *xla_args)
    if (multiple_results or
        any(len(aval_to_xla_shapes(v.aval)) > 1 for v in jaxpr.outvars)):
      return xops.Tuple(c, outs)
    else:
      assert len(outs) == 1, outs
      return outs[0]

  return f

def _array_aval_from_xla_shape(xla_shape):
  # This function instantiates the assumption that we can map fro XLA array
  # types to JAX array types.
  # TODO(mattjj): remove assumption can map XLA array types to JAX array types
  assert not xla_shape.is_tuple()
  return ShapedArray(xla_shape.dimensions(), xla_shape.numpy_dtype())

# Backwards compatibility flag. We keep this in for a bit longer to
# allow us to turn off the refactoring of the remat lowering if we
# run into test failures. A previous attempt at this was rolledback
# due to bugs.
# TODO(necula): remove this
_USE_LAX_REMAT_LOWERING = True

def _zeros(c, xla_shape):
  if xla_shape.is_array():
    shape, dtype = xla_shape.dimensions(), xla_shape.numpy_dtype()
    zero = xops.Constant(c, np.array(0, dtype=dtype))
    return xops.Broadcast(zero, shape)
  else:
    # It is a token
    return xops.CreateToken(c)


def _remat_using_cond(ctx, in_nodes, name, call_jaxpr):
  """Lower remat to a Conditional which always returns true. This:
    1. Circumvents common subexpression elimination.
    2. In common case of `jax.grad(jax.remat(f))`, ensures the remat blocks
       occur after the primal blocks, because cotangent is an input to the
       Conditional."""
  # Fake condition which always selects True branch.
  c = ctx.builder
  rng = xops.RngUniform(xops.Constant(c, np.array(0, dtype=np.float32)),
                        xops.Constant(c, np.array(1, dtype=np.float32)),
                        xc.Shape.array_shape(xc.PrimitiveType.F32, []))
  pred = xops.Lt(rng, xops.Constant(c, np.array(2, dtype=np.float32)))

  true_op = xops.Tuple(c, in_nodes)
  remat_subc = xc.XlaBuilder("remat_call_subcomputation")
  input_op = parameter(remat_subc, 0, c.get_shape(true_op), replicated=[])
  args = xla_destructure(remat_subc, input_op)
  sub_ctx = ctx.replace(
      builder=remat_subc,
      name_stack=extend_name_stack(ctx.name_stack, wrap_name(name, 'remat')))
  out_nodes = jaxpr_subcomp(sub_ctx, call_jaxpr, (), *args)
  out_node_shapes = [remat_subc.get_shape(o) for o in out_nodes]
  remat_subc = remat_subc.build(xops.Tuple(remat_subc, out_nodes))

  false_op = true_op
  dummy_subc = xc.XlaBuilder("remat_call_dummy_subcomputation")
  parameter(dummy_subc, 0, c.get_shape(false_op), replicated=[])
  out_nodes = [_zeros(dummy_subc, s) for s in out_node_shapes]
  dummy_subc = dummy_subc.build(xops.Tuple(dummy_subc, out_nodes))

  return xla_destructure(
      c, xops.Conditional(pred, true_op, remat_subc, false_op, dummy_subc))


def _remat_using_while(ctx, in_nodes, name, call_jaxpr):
  """Lower remat to a single iteration while loop."""
  c = ctx.builder
  # Dummy subc for getting subcomp shapes.
  dummy_inputs = xops.Tuple(c, in_nodes)
  dummy_subc = xc.XlaBuilder("remat_dummy_subcomputation")
  dummy_input_op = parameter(dummy_subc, 0, c.get_shape(dummy_inputs), replicated=[])
  dummy_args = xla_destructure(dummy_subc, dummy_input_op)
  dummy_ctx = ctx.replace(
      builder=dummy_subc,
      name_stack=extend_name_stack(ctx.name_stack, wrap_name(name, 'remat')))
  dummy_subcomp_outs = jaxpr_subcomp(dummy_ctx, call_jaxpr, (), *dummy_args)
  out_node_shapes = [dummy_subc.get_shape(o) for o in dummy_subcomp_outs]

  i_init = xops.Constant(c, np.array(0, dtype=np.int32))
  zeros_like_outs = [_zeros(c, s) for s in out_node_shapes]
  inputs = xops.Tuple(c, [i_init] + list(in_nodes) + zeros_like_outs)

  cond_subc = xc.XlaBuilder("remat_cond_subcomputation")
  input_op = parameter(cond_subc, 0, c.get_shape(inputs), replicated=[])
  i = xops.GetTupleElement(input_op, 0)
  rng = xops.RngUniform(xops.Constant(cond_subc, np.array(1, dtype=np.int32)),
                        xops.Constant(cond_subc, np.array(2, dtype=np.int32)),
                        xc.Shape.array_shape(xc.PrimitiveType.S32, []))
  cond_subc = cond_subc.build(xops.Lt(i, rng))

  body_subc = xc.XlaBuilder("remat_body_subcomputation")
  input_op = parameter(body_subc, 0, c.get_shape(inputs), replicated=[])
  i, *args = xla_destructure(body_subc, input_op)[:len(in_nodes)+1]
  i_next = xops.Add(i, xops.Constant(body_subc, np.array(1, dtype=np.int32)))
  body_ctx = ctx.replace(
      builder=body_subc,
      name_stack=extend_name_stack(ctx.name_stack, wrap_name(name, 'remat')))
  subcomp_outs = jaxpr_subcomp(body_ctx, call_jaxpr, (), *args)
  out_nodes = [i_next] + args + list(subcomp_outs)
  body_subc = body_subc.build(xops.Tuple(body_subc, out_nodes))
  outs = xops.While(cond_subc, body_subc, inputs)
  return xla_destructure(c, outs)[len(in_nodes)+1:]



def _remat_translation_rule(ctx, avals_in, avals_out, *in_nodes,
                            name, call_jaxpr,
                            prevent_cse, differentiated, concrete,
                            policy, device=None):
  del device, concrete, policy  # Unused.
  if differentiated and prevent_cse:
    if ctx.platform == "gpu":
      return _remat_using_while(ctx, in_nodes, name, call_jaxpr)
    else:
      return _remat_using_cond(ctx, in_nodes, name, call_jaxpr)
  else:
    return jaxpr_subcomp(ctx, call_jaxpr, (), *in_nodes)

if not _USE_LAX_REMAT_LOWERING:
  register_translation(pe.remat_call_p, _remat_translation_rule)


ad.primitive_transposes[core.named_call_p] = partial(ad.call_transpose,
                                                     core.named_call_p)


def _named_call_translation_rule(ctx, avals_in, avals_out, *in_nodes,
                                 name="core_call", backend=None, call_jaxpr):
  check_backend_matches(backend, ctx.platform)
  c = ctx.builder
  subc = xc.XlaBuilder(name)
  args = [parameter(subc, i, c.GetShape(n)) for i, n in enumerate(in_nodes)]
  sub_ctx = ctx.replace(builder=subc,
                        name_stack=extend_name_stack(ctx.name_stack, name))
  out_nodes = jaxpr_subcomp(sub_ctx, call_jaxpr, (), *args)
  subc = subc.Build(xops.Tuple(subc, out_nodes))
  return xla_destructure(c, xops.Call(c, subc, list(in_nodes)))
register_translation(core.named_call_p, _named_call_translation_rule)


def _call_translation_rule(ctx, avals_in, avals_out, *in_nodes, backend=None,
                           call_jaxpr):
  return _named_call_translation_rule(
      ctx, avals_in, avals_out, *in_nodes, name="core_call", backend=backend,
      call_jaxpr=call_jaxpr)
register_translation(core.call_p, _call_translation_rule)
