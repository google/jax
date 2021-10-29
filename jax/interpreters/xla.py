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


from collections import defaultdict, deque
import collections.abc
import dataclasses
import functools
from functools import partial, partialmethod
import itertools as it
import operator as op
import re
from typing import (Any, Callable, Deque, Dict, List, Optional, Sequence, Set,
                    Type, Tuple, Union, NamedTuple)
from typing_extensions import Protocol
from warnings import warn
import weakref

from absl import logging
import numpy as np

from ..config import config
from .. import core
from jax._src import ad_util
from jax._src import dtypes
from .. import linear_util as lu
from jax._src import source_info_util
from jax._src.abstract_arrays import (make_shaped_array, array_types)
from ..core import (ConcreteArray, ShapedArray, AbstractToken,
                    Literal, pp_eqn_compact, JaxprPpContext, raise_to_shaped,
                    abstract_token)
from ..errors import UnexpectedTracerError
import jax._src.pretty_printer as pp
from .._src.util import (cache, prod, unzip2, extend_name_stack, wrap_name,
                         safe_zip, safe_map, partition_list)
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from . import partial_eval as pe
from . import ad
from . import masking

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

# This flag is set on exit; no logging should be attempted
_on_exit = False


def identity(x): return x

_scalar_types = dtypes.python_scalar_dtypes.keys()

# unit representation
def _make_unit_constant(c): return [
    xops.Constant(c, np.zeros((), dtype=np.dtype('bool')))]
def _make_unit_shape(_): return (xc.Shape.array_shape(np.dtype('bool'), ()),)
def _device_put_unit(_, device):
  backend = xb.get_device_backend(device)
  return (backend.buffer_from_pyval(np.zeros((), dtype=np.dtype('bool')),
                                    device),)
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
                     name_stack: str = "",
                     source_info: Optional[source_info_util.Traceback] = None
                     ) -> xc.OpMetadata:
  eqn_str = str(pp.text(name_stack) +
                pp_eqn_compact(primitive.name, params, JaxprPpContext()))
  tracebacks[eqn_str] = source_info
  frame = source_info_util.user_frame(source_info) if source_info else None
  return xc.OpMetadata(
        op_type=primitive.name,
        op_name=eqn_str,
        source_file=_get_canonical_source_file(frame) if frame else None,
        source_line=frame.line_num if frame else None)

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



# IR constants

_constant_handlers: Dict[type, Callable] = {}

def pyval_to_ir_constants(builder, py_val, canonicalize_types=True):
  """Translate a general constant `py_val` to a constant, canonicalizing its dtype.

  Args:
    py_val: a Python value to be translated to a constant.

  Returns:
    A representation of the constant as a list of xla ops.
  """
  for t in type(py_val).mro():
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


# Result handlers

def aval_to_result_handler(device: Optional[Device],
                           aval: core.AbstractValue) -> Callable:
  try:
    return xla_result_handlers[type(aval)](device, aval)
  except KeyError as err:
    raise TypeError(f"No xla_result_handler for type: {type(aval)}") from err

def array_result_handler(device: Optional[Device], aval: core.ShapedArray):
  if aval.dtype is dtypes.float0:
    return lambda _: np.zeros(aval.shape, dtypes.float0)
  return partial(make_device_array, raise_to_shaped(aval), device)


xla_result_handlers: Dict[Type[core.AbstractValue], Callable[..., Callable]] = {
    core.AbstractUnit: lambda _, __: lambda _: core.unit,
    ShapedArray: array_result_handler,
    ConcreteArray: array_result_handler,
}

def device_put(x, device: Optional[Device] = None) -> Tuple[Any]:
  x = canonicalize_dtype(x)
  try:
    return device_put_handlers[type(x)](x, device)
  except KeyError as err:
    raise TypeError(f"No device_put handler for type: {type(x)}") from err

def _device_put_array(x, device: Optional[Device]):
  backend = xb.get_device_backend(device)
  if x.dtype is dtypes.float0:
    x = np.zeros(x.shape, dtype=np.dtype(bool))
  return (backend.buffer_from_pyval(x, device),)

def _device_put_scalar(x, device):
  return _device_put_array(dtypes.coerce_to_array(x), device)

device_put_handlers: Dict[Any, Callable[[Any, Optional[Device]], Tuple[Any]]] = {
  core.Unit: _device_put_unit
}
device_put_handlers.update((t, _device_put_array) for t in array_types)
device_put_handlers.update((t, _device_put_scalar) for t in _scalar_types)

# TODO(mattjj): try to remove this canonicalize_dtype stuff
def canonicalize_dtype(x):
  typ = type(x)
  handler = canonicalize_dtype_handlers.get(typ)
  if handler: return handler(x)
  for typ in typ.mro():
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
canonicalize_dtype_handlers.update(
    (t, _canonicalize_ndarray_dtype) for t in array_types)
canonicalize_dtype_handlers.update(
    (t, partial(_canonicalize_python_scalar_dtype, t)) for t in _scalar_types)

def abstractify(x) -> core.AbstractValue:
  typ = type(x)
  aval_fn = pytype_aval_mappings.get(typ)
  if aval_fn: return aval_fn(x)
  for typ in typ.mro():
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
pytype_aval_mappings.update((t, make_shaped_array) for t in array_types)
pytype_aval_mappings.update(
    (t, partial(_make_abstract_python_scalar, t)) for t in _scalar_types)

# We can optionally set a Jaxpr rewriter that can be applied just before
# compilation. This mechanism is used for compiling id_tap, we can
# remove it once we bring the id_tap implementation into the core.
outfeed_rewriter: Optional[Callable[[core.Jaxpr], core.Jaxpr]] = None
def apply_outfeed_rewriter(jaxpr: core.Jaxpr) -> core.Jaxpr:
  if outfeed_rewriter is not None:
    return outfeed_rewriter(jaxpr)
  else:
    return jaxpr

outfeed_primitives: Set[core.Primitive] = set()
def jaxpr_uses_outfeed(jaxpr: core.Jaxpr) -> bool:
  """Finds if there are outfeed primitives anywhere inside a Jaxpr."""
  return any(primitive_uses_outfeed(eqn.primitive, eqn.params)
             for eqn in jaxpr.eqns)

def _param_uses_outfeed(param):
  if type(param) is core.Jaxpr:
    if jaxpr_uses_outfeed(param):
      return True
  elif type(param) is core.ClosedJaxpr:
    if jaxpr_uses_outfeed(param.jaxpr):
      return True
  return False

def primitive_uses_outfeed(prim: core.Primitive, params: Dict) -> bool:
  if prim in outfeed_primitives:
    return True
  for param in params.values():
    if isinstance(param, tuple):
      if any(unsafe_map(_param_uses_outfeed, param)):
        return True
    elif _param_uses_outfeed(param):
      return True
  return False

### op-by-op execution


ArgSpec = Tuple[core.AbstractValue, Optional[Device]]

def arg_spec(x: Any) -> ArgSpec:
  aval = abstractify(x)
  try:
    return aval, x._device
  except:
    return aval, None

def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  compiled_fun = xla_primitive_callable(prim, *unsafe_map(arg_spec, args),
                                        **params)
  return compiled_fun(*args)


def _partition_outputs(nouts: Sequence[int], outs):
  if len(nouts) == 1:
    return [outs]
  outs = iter(outs)
  return [[next(outs) for _ in range(nout)] for nout in nouts]


@cache()
def xla_primitive_callable(prim, *arg_specs: ArgSpec, **params):
  avals, arg_devices = unzip2(arg_specs)
  donated_invars = (False,) * len(arg_specs)
  device = _device_from_arg_devices(arg_devices)
  def prim_fun(*args):
    out = prim.bind(*args, **params)
    if prim.multiple_results:
      return out
    else:
      return out,
  compiled = _xla_callable_uncached(lu.wrap_init(prim_fun), device, None,
                                    prim.name, donated_invars, *arg_specs)
  if not prim.multiple_results:
    return lambda *args, **kw: compiled(*args, **kw)[0]
  else:
    return compiled

def _device_from_arg_devices(devices: Sequence[Optional[Device]]) -> Optional[Device]:
  """Given devices of inputs, determine where to perform a computation.

  Args:
    devices: list where each element is a either a `Device` instance or `None`.
  Returns:
    A `Device` instance or None.
  Raises:
    ValueError if input devices are inconsistent.
  """
  try:
    device, = {d for d in devices if d is not None} or (None,)
    return device
  except ValueError as err:
    msg = "primitive arguments must be colocated on the same device, got {}"
    raise ValueError(msg.format(", ".join(map(str, devices)))) from err

def primitive_subcomputation(prim: core.Primitive, *avals: core.AbstractValue,
                             **params):
  c = xc.XlaBuilder(f"primitive_computation_{prim.name}")
  f = lower_fun(prim.bind, multiple_results=prim.multiple_results)
  xla_args, _ = _xla_callable_args(c, avals, tuple_args=False)
  ans = f(c, *xla_args, **params)
  return c.build(ans)

def backend_compile(backend, built_c, options):
  # we use a separate function call to ensure that XLA compilation appears
  # separately in Python profiling results
  return backend.compile(built_c, compile_options=options)


def needs_check_special():
  return config.jax_debug_infs or config.jax_debug_nans

def check_special(name, bufs):
  if needs_check_special():
    for buf in bufs:
      _check_special(name, buf.xla_shape(), buf)

def _check_special(name, xla_shape, buf):
  assert not xla_shape.is_tuple()
  if dtypes.issubdtype(xla_shape.element_type(), np.inexact):
    if config.jax_debug_nans and np.any(np.isnan(buf.to_py())):
      raise FloatingPointError(f"invalid value (nan) encountered in {name}")
    if config.jax_debug_infs and np.any(np.isinf(buf.to_py())):
      raise FloatingPointError(f"invalid value (inf) encountered in {name}")

### compiling jaxprs

def prefetch(x):
  if isinstance(x, DeviceArray):
    x.copy_to_host_async()
  return x

def jaxpr_literals(jaxpr):
  """Generates all the literals inside a jaxpr, including nested subjaxprs."""
  for eqn in jaxpr.eqns:
    for v in eqn.invars:
      if type(v) is core.Literal:
        yield v.val
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from jaxpr_literals(subjaxpr)


def _flatmap(func: Callable, vars: Sequence):
  return list(it.chain.from_iterable(map(func, vars)))

def _partitionmap(func: Callable, vars: Sequence, nodes: Sequence):
  return map(func, vars,
             _partition_outputs([len(aval_to_xla_shapes(v.aval)) for v in vars],
                                nodes))

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
  # TODO(phawkins): make platform non-optional.
  # assert ctx.platform is not None
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
    if (ctx.platform is not None and
        eqn.primitive in _backend_specific_translations[ctx.platform]):
      rule = _backend_specific_translations[ctx.platform][eqn.primitive]
    elif eqn.primitive in _translations:
      rule = _translations[eqn.primitive]
    else:
      raise NotImplementedError(
          f"XLA translation rule for primitive '{eqn.primitive.name}' not found")

    in_nodes = _flatmap(read, eqn.invars)
    avals_out = map(aval, eqn.outvars)
    op_metadata = make_op_metadata(
        eqn.primitive, eqn.params, name_stack=ctx.name_stack,
        source_info=eqn.source_info)
    ctx.builder.set_op_metadata(op_metadata)
    with source_info_util.user_context(eqn.source_info):
      ans = rule(ctx, map(aval, eqn.invars), avals_out, *in_nodes, **eqn.params)
    ctx.builder.clear_op_metadata()

    assert isinstance(ans, collections.abc.Sequence), (ans, eqn)
    assert all(isinstance(x, xe.XlaOp) for x in ans), (ans, eqn)
    expected_shapes = [aval_to_xla_shapes(aval) for aval in avals_out]
    partitioned_ans = _partition_outputs([len(xs) for xs in expected_shapes],
                                         ans)
    # Check the XLA shapes match what we would expect from the output avals.
    for xs, shapes in zip(partitioned_ans, expected_shapes):
      for x, expected_shape in zip(xs, shapes):
        # force xla to do its own shape checking
        actual_shape = ctx.builder.get_shape(x)
        # TODO(phawkins): the dtype canonicalization here is incorrect and
        # is a work around for x64-mode problems. Remove it.
        if (dtypes.canonicalize_dtype(actual_shape.numpy_dtype()) !=
            dtypes.canonicalize_dtype(expected_shape.numpy_dtype()) or
            actual_shape.dimensions() != expected_shape.dimensions()):
          raise TypeError(
              "Type mismatch in outputs of XLA translation rule for "
              f"primitive '{eqn.primitive.name}'. Expected shape "
              f"{expected_shape}, got shape {actual_shape}. This is a bug in "
              "JAX or in a user-provided XLA translation rule; "
              "please report it.")
    map(write, eqn.outvars, partitioned_ans)
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

def axis_groups(axis_env: AxisEnv, name):
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

def jaxpr_replicas(jaxpr) -> int:
  """The number of replicas needed for a jaxpr.

  For a eqn, multiply the `axis_size` with the `jaxpr_replicas` of the
  subjaxprs. For a list of eqns, take the maximum number of replicas.
  """
  if isinstance(jaxpr, core.ClosedJaxpr):
    jaxpr = jaxpr.jaxpr
  return max(unsafe_map(eqn_replicas, jaxpr.eqns), default=1)

# TODO(mattjj): this function assumes that only pmap has a parameter named
# axis_size, and that it corresponds to cross-replica mapping
def eqn_replicas(eqn):
  call_jaxpr = eqn.params.get("call_jaxpr")
  if call_jaxpr:
    return eqn.params.get('axis_size', 1) * jaxpr_replicas(call_jaxpr)
  elif eqn.primitive in _initial_style_primitives:
    return initial_style_primitive_replicas(eqn.params)
  else:
    return 1

def initial_style_primitive_replicas(params):
  return max(core.traverse_jaxpr_params(jaxpr_replicas, params).values(), default=1)

# TODO(mattjj,skyewm): the functions here are utilities for checking if
# not-yet-supported features are used with multi-host programming

def jaxpr_has_pmap(jaxpr):
  """Whether there is an xla_pmap primitive anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if 'xla_pmap' in eqn.primitive.name:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_pmap(subjaxpr):
      return True
  return False


def jaxpr_collectives(jaxpr):
  """Generates all the collective primitives anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if eqn.primitive in _collective_primitives:
      yield eqn.primitive
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from jaxpr_collectives(subjaxpr)


### xla_call underlying jit

def _xla_call_impl(fun: lu.WrappedFun, *args, device, backend, name,
                   donated_invars, inline):
  del inline  # Only used at tracing time
  compiled_fun = _xla_callable(fun, device, backend, name, donated_invars,
                               *unsafe_map(arg_spec, args))
  try:
    out = compiled_fun(*args)
  except FloatingPointError:
    assert config.jax_debug_nans or config.jax_debug_infs  # compiled_fun can only raise in this case
    print("Invalid value encountered in the output of a jit/pmap-ed function. "
          "Calling the de-optimized version.")
    # We want to run the wrapped function again (after _xla_callable already ran
    # it), but linear_util.WrappedFun instances are meant to be run only once.
    # In addition to re-executing the Python code, which is usually undesirable
    # but which config.jax_debug_nans is meant to opt into, we'll be re-executing
    # any linear_util.py-style side effects, i.e. re-populating Stores created
    # by any transformation_with_aux's applied to fun. Since this is
    # intentional here, to avoid "Store occupied" errors we clone the WrappedFun
    # with empty stores.
    stores = [lu.Store() for _ in fun.stores]
    clone = lu.WrappedFun(fun.f, fun.transforms, stores, fun.params)
    with core.new_sublevel():
      _ = clone.call_wrapped(*args)  # probably won't return
  return out

def flatten_shape(s: XlaShape) -> Sequence[Tuple[Sequence[int], XlaShape]]:
  """Expands a given shape tree into a flat list of indices to arrays.

  Given the following computation:

  >>> c = xc.XlaBuilder("example")
  >>> p0 = xb.parameter(c, 1, xc.shape_from_pyval(jnp.ones([1])))
  >>> p1 = xb.parameter(c, 2, xc.shape_from_pyval(jnp.ones([2])))
  >>> p2 = xb.parameter(c, 3, xc.shape_from_pyval(jnp.ones([3])))
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

def _xla_callable_uncached(fun: lu.WrappedFun, device, backend, name,
                           donated_invars, *arg_specs):
  return lower_xla_callable(fun, device, backend, name, donated_invars,
                            *arg_specs).compile().unsafe_call

_xla_callable = lu.cache(_xla_callable_uncached)

def lower_xla_callable(fun: lu.WrappedFun, device, backend, name,
                       donated_invars, *arg_specs):
  if device is not None and backend is not None:
    raise ValueError("can't specify both a device and a backend for jit, "
                     "got device={} and backend={}".format(device, backend))

  abstract_args, arg_devices = unzip2(arg_specs)
  jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(
      fun, abstract_args, pe.debug_info_final(fun, "jit"))
  if any(isinstance(c, core.Tracer) for c in consts):
    raise UnexpectedTracerError("Encountered an unexpected tracer.")
  jaxpr, kept_const_idx, kept_var_idx = _prune_unused_inputs(jaxpr)
  consts = [c for i, c in enumerate(consts) if i in kept_const_idx]
  pruned_arg_specs = (a for i, a in enumerate(arg_specs) if i in kept_var_idx)
  abstract_args, arg_devices = unzip2(pruned_arg_specs)
  donated_invars = [
      x for i, x in enumerate(donated_invars) if i in kept_var_idx
  ]
  map(prefetch, it.chain(consts, jaxpr_literals(jaxpr)))
  jaxpr = apply_outfeed_rewriter(jaxpr)

  nreps = jaxpr_replicas(jaxpr)
  device = _xla_callable_device(nreps, backend, device, arg_devices)
  backend = xb.get_device_backend(device) if device else xb.get_backend(backend)

  # Computations that only produce constants and/or only rearrange their inputs,
  # which are often produced from partial evaluation, don't need compilation,
  # and don't need to evaluate their arguments.
  if not jaxpr.eqns:
    return XlaComputation(
        name, None, True, jaxpr, consts, device, abstract_args, out_avals,
        kept_var_idx)

  if not _on_exit:
    log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
    logging.log(log_priority, "Compiling %s (%s) for args %s.",
                fun.__name__, id(fun), abstract_args)

  if nreps > 1:
    warn(f"The jitted function {name} includes a pmap. Using "
         "jit-of-pmap can lead to inefficient data movement, as the outer jit "
         "does not preserve sharded data representations and instead collects "
         "input and output arrays onto a single device. "
         "Consider removing the outer jit unless you know what you're doing. "
         "See https://github.com/google/jax/issues/2926.")

  if nreps > xb.device_count(backend):
    raise ValueError(
        f"compiling computation `{name}` that requires {nreps} replicas, but "
        f"only {xb.device_count(backend)} XLA devices are available.")

  if xb.process_count() > 1 and (nreps > 1 or jaxpr_has_pmap(jaxpr)):
    raise NotImplementedError(
        "jit of multi-host pmap not implemented (and jit-of-pmap can cause "
        "extra data movement anyway, so maybe you don't want it after all).")

  tuple_args = len(abstract_args) > 100  # pass long arg lists as tuple for TPU

  c = xc.XlaBuilder(f"jit_{fun.__name__}")
  xla_consts = _xla_consts(c, consts)
  xla_args, donated_invars = _xla_callable_args(c, abstract_args, tuple_args,
                                                donated_invars=donated_invars)
  platform = backend.platform
  ctx = TranslationContext(c, platform, AxisEnv(nreps, (), ()),
                           extend_name_stack(wrap_name(name, 'jit')))
  out_nodes = jaxpr_subcomp(ctx, jaxpr, xla_consts, *xla_args)
  backend = xb.get_backend(backend)
  # There is a non-zero cost to building an output tuple, particularly on TPU.
  # Avoid it if the output arity is 1.
  output = out_nodes[0] if len(out_nodes) == 1 else xops.Tuple(c, out_nodes)
  if platform in ("gpu", "tpu"):
    donated_invars = set_up_aliases(
        c, xla_args, c.GetShape(output), donated_invars, tuple_args)
  if any(donated_invars):
    # TODO(tomhennigan): At call time we should mark these buffers as deleted.
    unused_donations = [str(c.GetShape(a))
                        for a, d in zip(xla_args, donated_invars) if d]
    warn("Some donated buffers were not usable: {}".format(
         ", ".join(unused_donations)))
  built = c.build(output)
  return XlaComputation(
      name, built, False, nreps, device, backend, tuple_args, abstract_args,
      out_avals, kept_var_idx)


def compile_or_get_cached(backend, computation, compile_options):
    # Avoid import cycle between jax and jax.experimental
    from jax.experimental.compilation_cache import compilation_cache as cc
    # Persistent compilation cache only implemented on TPU.
    # TODO(skye): add warning when initializing cache on unsupported default platform
    if cc.is_initialized() and backend.platform == 'tpu':
        cached_executable = cc.get_executable(computation, compile_options, backend)
        if cached_executable is not None:
            logging.info('Persistent compilation cache hit')
            return cached_executable
        else:
            compiled = backend_compile(backend, computation, compile_options)
            cc.put_executable(computation, compile_options, compiled, backend)
            return compiled
    return backend_compile(backend, computation, compile_options)


class XlaComputation:
  name: str
  _is_trivial: bool
  _executable: Optional['XlaCompiledComputation']

  def __init__(self, name: str, hlo, is_trivial: bool, *compile_args):
    self.name = name
    self._hlo = hlo
    self._is_trivial = is_trivial
    self._executable = None
    self.compile_args = compile_args

  def is_trivial(self):
    return self._is_trivial

  def hlo(self):
    if self.is_trivial():
      raise ValueError("A trivial computation has no HLO")
    return self._hlo

  def compile(self) -> 'XlaCompiledComputation':
    if self._executable is None:
      if self.is_trivial():
        self._executable = XlaCompiledComputation.from_trivial_jaxpr(
            *self.compile_args)
      else:
        self._executable = XlaCompiledComputation.from_xla_computation(
            self.name, self.hlo(), *self.compile_args)
    return self._executable


class XlaCompiledComputation:
  def __init__(self, xla_executable, in_avals, kept_var_idx, unsafe_call):
    self._xla_executable = xla_executable
    self.in_avals = in_avals
    self._kept_var_idx = kept_var_idx
    self.unsafe_call = unsafe_call

  @staticmethod
  def from_xla_computation(
      name: str,
      xla_computation,
      nreps: int,
      device,
      backend,
      tuple_args: bool,
      in_avals,
      out_avals,
      kept_var_idx) -> 'XlaCompiledComputation':
    result_handlers = map(partial(aval_to_result_handler, device), out_avals)
    options = xb.get_compile_options(
        num_replicas=nreps,
        num_partitions=1,
        device_assignment=(device.id,) if device else None)
    options.parameter_is_tupled_arguments = tuple_args
    compiled = compile_or_get_cached(backend, xla_computation, options)
    buffer_counts = (None if len(out_avals) == 1 else
                     [len(aval_to_xla_shapes(aval)) for aval in out_avals])
    execute = _execute_compiled if nreps == 1 else _execute_replicated
    unsafe_call = partial(execute, name, compiled, buffer_counts,
                          result_handlers, kept_var_idx)
    return XlaCompiledComputation(compiled, in_avals, kept_var_idx, unsafe_call)

  def is_trivial(self):
    return self._xla_executable == None

  def xla_executable(self):
    if self.is_trivial():
      raise ValueError("A trivial compiled computation has no XLA executable")
    return self._xla_executable

  @staticmethod
  def from_trivial_jaxpr(jaxpr, consts, device, in_avals, out_avals,
                         kept_var_idx) -> 'XlaCompiledComputation':
    result_handlers = map(partial(aval_to_result_handler, device), out_avals)
    unsafe_call = partial(_execute_trivial, jaxpr, device, consts,
                          out_avals, result_handlers, kept_var_idx)
    return XlaCompiledComputation(None, in_avals, kept_var_idx, unsafe_call)

  def call(self, *args):
    arg_specs = unsafe_map(arg_spec, args)
    arg_avals = [spec[0] for i, spec in enumerate(arg_specs)
                 if i in self._kept_var_idx]
    check_arg_avals_for_call(self.in_avals, arg_avals)
    return self.unsafe_call(*args)


def check_arg_avals_for_call(ref_avals, arg_avals):
  if len(ref_avals) != len(arg_avals):
    raise TypeError(
        f"Computation compiled for {len(ref_avals)} inputs "
        f"but called with {len(arg_avals)}")
  for ref_aval, arg_aval in zip(ref_avals, arg_avals):
    if not core.typematch(ref_aval, arg_aval):
      ref_avals_fmt = ', '.join(str(a) for a in ref_avals)
      arg_avals_fmt = ', '.join(str(a) for a in arg_avals)
      raise TypeError(
        f"Computation compiled for input types:\n  {ref_avals_fmt}\n"
        f"called with:\n  {arg_avals_fmt}")


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


def _prune_unused_inputs(
    jaxpr: core.Jaxpr) -> Tuple[core.Jaxpr, Set[int], Set[int]]:
  used = {v for v in jaxpr.outvars if isinstance(v, core.Var)}
  # TODO(zhangqiaorjc): Improve the DCE algorithm by also pruning primitive
  # applications that do not produce used outputs. Must handle side-effecting
  # primitives and nested jaxpr.
  used.update(
      v for eqn in jaxpr.eqns for v in eqn.invars if isinstance(v, core.Var))
  kept_const_idx, new_constvars = unzip2(
      (i, v) for i, v in enumerate(jaxpr.constvars) if v in used)
  kept_var_idx, new_invars = unzip2(
      (i, v) for i, v in enumerate(jaxpr.invars) if v in used)
  new_jaxpr = core.Jaxpr(new_constvars, new_invars, jaxpr.outvars, jaxpr.eqns)
  return new_jaxpr, set(kept_const_idx), set(kept_var_idx)


def _xla_callable_device(nreps, backend, device, arg_devices):
  if nreps > 1:
    if device is not None or backend is not None:
      raise ValueError(f"can't specify device or backend for jit-of-pmap, "
                       f"got device={device} and backend={backend}")
    return None
  else:
    if device is None and backend is None:
      return _device_from_arg_devices(arg_devices)
    elif device is not None and backend is None:
      return device
    elif device is None and backend is not None:
      return xb.get_backend(backend).get_default_device_assignment(1)[0]
    else:
      assert False  # Unreachable given the error check in _xla_callable

# Used within _xla_callable_args and _xla_param to distinguish between None (no
# sharding annotation set) and replicated.
_replicated_param = object()

def _xla_callable_args(
    c, avals, tuple_args, *,
    replicated=None,
    partitions=None,
    partitions_proto: bool = False,
    donated_invars=None):
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
    xla_args = [_xla_param(c, next(counts), xla_shape, r, p, partitions_proto)
                if a is not abstract_token else xops.CreateToken(c)
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
      tuple_parts = xb.tuple_sharding_proto(partitions)
    else:
      tuple_parts = tuple(partitions)
    tuple_shape = xc.Shape.tuple_shape(
        [shape for a in avals for shape in aval_to_xla_shapes(a) if a is not abstract_token])
    tuple_param = _xla_param(c, 0, tuple_shape, replicated, tuple_parts, partitions_proto)
    xla_inputs = iter(xla_destructure(c, tuple_param))
    xla_args = [next(xla_inputs) if a is not abstract_token else
                xops.CreateToken(c) for a in avals]
    assert next(xla_inputs, None) is None
    return xla_args, donated_invars

def _xla_param(builder, param_num, xla_shape, replicated, partitions, parts_proto):
  make_param = partial(xb.parameter, builder, param_num, xla_shape,
                       replicated=replicated)
  with_sharding = xb.with_sharding_proto if parts_proto else xb.with_sharding
  if partitions is None:
    return make_param()
  elif partitions is _replicated_param:
    return with_sharding(builder, None, make_param)
  else:
    return with_sharding(builder, partitions, make_param)


def _execute_compiled(name: str, compiled: XlaExecutable,
                      output_buffer_counts: Optional[Sequence[int]], handlers,
                      kept_var_idx, *args):
  device, = compiled.local_devices()
  input_bufs = list(
      it.chain.from_iterable(
          device_put(x, device)
          for i, x in enumerate(args)
          if x is not token and i in kept_var_idx))
  out_bufs = compiled.execute(input_bufs)
  check_special(name, out_bufs)
  if output_buffer_counts is None:
    return (handlers[0](*out_bufs),)
  return tuple(
      handler(*bs) for handler, bs in
      unsafe_zip(handlers, _partition_outputs(output_buffer_counts, out_bufs)))


def _execute_replicated(name: str, compiled: XlaExecutable,
                        output_buffer_counts: Optional[Sequence[int]], handlers,
                        kept_var_idx, *args):
  input_bufs = [
      list(
          it.chain.from_iterable(
              device_put(x, device)
              for i, x in enumerate(args)
              if x is not token and i in kept_var_idx))
      for device in compiled.local_devices()
  ]
  out_bufs = [
      buf[0] for buf in compiled.execute_sharded_on_local_devices(
          list(zip(*input_bufs)))
  ]
  check_special(name, out_bufs)
  if output_buffer_counts is None:
    return (handlers[0](*out_bufs),)
  return tuple(
      handler(*bs) for handler, bs in
      unsafe_zip(handlers, _partition_outputs(output_buffer_counts, out_bufs)))


def _execute_trivial(jaxpr, device: Optional[Device], consts, avals, handlers,
                     kept_var_idx, *args):
  env = {core.unitvar: core.unit}
  pruned_args = (x for i, x in enumerate(args) if i in kept_var_idx)
  map(env.setdefault, jaxpr.invars, pruned_args)
  map(env.setdefault, jaxpr.constvars, consts)
  outs = [canonicalize_dtype(v.val) if type(v) is Literal else env[v]
          for v in jaxpr.outvars]
  return [_copy_device_array_to_device(x, device) if type_is_device_array(x)
          else h(*device_put(x, device)) for h, x in zip(handlers, outs)]

xla_call_p: core.CallPrimitive = core.CallPrimitive('xla_call')
xla_call = xla_call_p.bind
xla_call_p.def_impl(_xla_call_impl)

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
  args = [xb.parameter(subc, i, c.get_shape(n)) for i, n in enumerate(in_nodes)]
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

def lower_fun(fun: Callable, *, multiple_results: bool, parallel: bool = False,
              backend=None, new_style: bool = False) -> Callable:
  if new_style:
    def f_new(ctx: TranslationContext, avals_in: Sequence[core.AbstractValue],
              avals_out: Sequence[core.AbstractValue], *xla_args: xc.XlaOp,
              **params) -> Sequence[xc.XlaOp]:
      wrapped_fun = lu.wrap_init(fun, params)
      if not multiple_results:
        wrapped_fun = _tuple_output(wrapped_fun)
      if parallel:
        axis_env = ctx.axis_env
      else:
        axis_env = AxisEnv(1, (), ())
      with core.extend_axis_env_nd(zip(axis_env.names, axis_env.sizes)):
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, avals_in)
      return jaxpr_subcomp(ctx, jaxpr, _xla_consts(ctx.builder, consts),
                           *xla_args)
    return f_new

  # TODO(phawkins): migrate dependent code & always use new_style=True.
  def f(c, *xla_args, **params):
    avals = [_array_aval_from_xla_shape(c.get_shape(x)) for x in xla_args]
    return f_with_avals(c, avals, xla_args, params)

  def f_with_avals(c, avals, xla_args, params):
    if parallel:
      axis_env = params.pop('axis_env')
      del params['platform']
    else:
      axis_env = AxisEnv(1, (), ())
    wrapped_fun = lu.wrap_init(fun, params)
    if not multiple_results:
      wrapped_fun = _tuple_output(wrapped_fun)
    with core.extend_axis_env_nd(zip(axis_env.names, axis_env.sizes)):
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, avals)
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

### device-persistent data

class Token(object): pass
token = Token()

pytype_aval_mappings[Token] = lambda _: abstract_token
core.pytype_aval_mappings[Token] = lambda _: abstract_token
xla_shape_handlers[AbstractToken] = lambda _: (xc.Shape.token_shape(),)
xla_result_handlers[AbstractToken] = lambda _, __: lambda _: token
canonicalize_dtype_handlers[Token] = identity
device_put_handlers[Token] = lambda x, _: (x,)


def _forward_method(attrname, self, fun, *args):
  return fun(getattr(self, attrname), *args)
_forward_to_value = partial(_forward_method, "_value")


# The following is used for the type _CppDeviceArray or _DeviceArray.
DeviceArrayProtocol = Any
DeviceArray = xc.DeviceArrayBase

_CppDeviceArray: DeviceArrayProtocol = xc.Buffer

def make_device_array(
    aval: core.ShapedArray,
    device: Optional[Device],
    device_buffer: Buffer,
) -> Union[Buffer, "_DeviceArray"]:
  """Returns a DeviceArray implementation based on arguments.

  This is to be used only within JAX. It will return either a PythonDeviceArray
  or a C++ equivalent implementation.
  """
  if isinstance(device_buffer, _CppDeviceArray):

    if device_buffer.aval == aval and device_buffer._device == device:
      return device_buffer
    device_buffer = device_buffer.clone()
    device_buffer._device = device
    device_buffer.aval = aval
    device_buffer.weak_type = aval.weak_type
    return device_buffer

  return _DeviceArray(aval, device, device_buffer)


def type_is_device_array(x):
  """Returns `True` if `x` is a non-sharded DeviceArray.

  Use this function instead of `type(x) is Devicearray`.
  """
  type_x = type(x)
  return type_x is _DeviceArray or type_x is _CppDeviceArray


def device_array_supports_weakrefs():
  try:
    weakref.ref(DeviceArray())
    return True
  except TypeError:
    return False


class _DeviceArray(DeviceArray):  # type: ignore
  """A DeviceArray is an ndarray backed by a single device memory buffer."""
  # We don't subclass ndarray because that would open up a host of issues,
  # but lax_numpy.py overrides isinstance behavior and attaches ndarray methods.
  __slots__ = [
      "aval", "device_buffer", "_npy_value", "_device", "__weakref__"
  ]
  __array_priority__ = 100

  # DeviceArray has methods that are dynamically populated in lax_numpy.py,
  # and this annotation is needed to make pytype happy.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, aval: core.ShapedArray, device: Optional[Device],
               device_buffer: Buffer):
    """Initializer.

    Args:
      aval: The abstract value associated to this array (shape+dtype+weak_type).
      device:  The optional sticky device. See
        https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
      device_buffer: The underlying buffer owning the on-device data.
    """
    DeviceArray.__init__(self)
    self.aval = aval
    self.device_buffer = device_buffer
    self._device = device

    self._npy_value = None
    if config.jax_enable_checks:
      assert type(aval) is ShapedArray
      npy_value = self._value
      assert npy_value.dtype == aval.dtype and npy_value.shape == aval.shape
      assert (device is None) or device is device_buffer.device()

  def _check_if_deleted(self):
    if self.device_buffer is deleted_buffer:
      raise RuntimeError("DeviceArray has been deleted.")

  def block_until_ready(self):
    """Blocks the caller until the buffer's value has been computed on device.

    This method is mostly useful for timing microbenchmarks that wish to
    time how long a computation takes, without transferring the result back
    to the host.

    Returns the buffer object (`self`).
    """
    self._check_if_deleted()
    self.device_buffer.block_host_until_ready()  # pytype: disable=attribute-error
    return self

  @property
  def _value(self):
    self._check_if_deleted()
    if self._npy_value is None:
      self._npy_value = self.device_buffer.to_py()  # pytype: disable=attribute-error  # bind-properties
      self._npy_value.flags.writeable = False
    return self._npy_value

  @property
  def shape(self):
    return self.aval.shape

  @property
  def dtype(self):
    return self.aval.dtype

  @property
  def size(self):
    return prod(self.aval.shape)

  @property
  def ndim(self):
    return len(self.aval.shape)

  def copy_to_host_async(self):
    """Requests a copy of the buffer to the host."""
    self._check_if_deleted()
    if self._npy_value is None:
      self.device_buffer.copy_to_host_async()  # pytype: disable=attribute-error

  def delete(self):
    """Deletes the device array and any cached copy on the host.

    It is an error to access the contents of a `DeviceArray` after it has
    been deleted.

    Use of this method is optional; device buffers will be reclaimed
    automatically by Python when a DeviceArray object is garbage collected.
    However, it is sometimes useful to have more explicit control over the
    time of deletion.
    """
    self.device_buffer.delete()  # pytype: disable=attribute-error
    self.device_buffer = deleted_buffer
    self._npy_value = None

  @property
  def __cuda_array_interface__(self):
    return self.device_buffer.__cuda_array_interface__  # pytype: disable=attribute-error  # bind-properties


# Adding methods dynamically to both _DeviceArray and _CppDeviceArray
# pylint: disable=protected-access
for device_array in [DeviceArray]:


  def copy(self):
    """Returns an ndarray (backed by host memory, not device memory)."""
    return np.asarray(self)
  setattr(device_array, "copy", copy)

  def __repr__(self):
    line_width = np.get_printoptions()["linewidth"]
    prefix = '{}('.format(self.__class__.__name__.lstrip('_'))
    s = np.array2string(self._value, prefix=prefix, suffix=',',
                        separator=', ', max_line_width=line_width)
    if self.aval is not None and self.aval.weak_type:
      dtype_str = f'dtype={self.dtype.name}, weak_type=True)'
    else:
      dtype_str = f'dtype={self.dtype.name})'
    last_line_len = len(s) - s.rfind('\n') + 1
    sep = ' '
    if last_line_len + len(dtype_str) + 1 > line_width:
      sep = ' ' * len(prefix)
    return "{}{},{}{}".format(prefix, s, sep, dtype_str)

  setattr(device_array, "__repr__", __repr__)

  def item(self):
    if dtypes.issubdtype(self.dtype, np.complexfloating):
      return complex(self)
    elif dtypes.issubdtype(self.dtype, np.floating):
      return float(self)
    elif dtypes.issubdtype(self.dtype, np.integer):
      return int(self)
    elif dtypes.issubdtype(self.dtype, np.bool_):
      return bool(self)
    else:
      raise TypeError(self.dtype)

  setattr(device_array, "item", item)

  def __len__(self):
    try:
      return self.aval.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error

  setattr(device_array, "__len__", __len__)

  def __iter__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")  # same as numpy error
    else:
      return self._value.__iter__()

  setattr(device_array, "__iter__", __iter__)

  def __reversed__(self):
    if self.ndim == 0:
      raise TypeError("iteration over a 0-d array")
    else:
      return reversed(self._value)

  setattr(device_array, "__reversed__", __reversed__)

  def __format__(self, format_spec):
    # Simulates behavior of https://github.com/numpy/numpy/pull/9883
    if self.ndim == 0:
      return format(self._value[()], format_spec)
    else:
      return format(self._value, format_spec)

  setattr(device_array, "__format__", __format__)

  def __array__(self, dtype=None, context=None):
    return np.asarray(self._value, dtype=dtype)

  setattr(device_array, "__array__", __array__)

  setattr(device_array, "__str__", partialmethod(_forward_to_value, str))
  setattr(device_array, "__bool__", partialmethod(_forward_to_value, bool))
  setattr(device_array, "__nonzero__", partialmethod(_forward_to_value, bool))
  setattr(device_array, "__float__", lambda self: self._value.__float__())
  setattr(device_array, "__int__", lambda self: self._value.__int__())
  setattr(device_array, "__complex__", lambda self: self._value.__complex__())
  setattr(device_array, "__hex__", partialmethod(_forward_to_value, hex))
  setattr(device_array, "__oct__", partialmethod(_forward_to_value, oct))
  setattr(device_array, "__index__", partialmethod(_forward_to_value, op.index))
  to_bytes = lambda self, order="C": self._value.tobytes(order)
  setattr(device_array, "tobytes", to_bytes)
  del to_bytes
  setattr(device_array, "tolist", lambda self: self._value.tolist())

  # pickle saves and loads just like an ndarray
  setattr(device_array, "__reduce__",
          partialmethod(_forward_to_value, op.methodcaller("__reduce__")))

  # explicitly set to be unhashable.
  setattr(device_array, "__hash__", None)

  # clobbered when jax.numpy is imported, but useful in tests
  setattr(device_array, "__eq__", lambda self, other: self._value == other)

  # The following methods are dynamically overridden in lax_numpy.py.
  def raise_not_implemented():
    raise NotImplementedError

  setattr(device_array, "__getitem__", lambda self, i: raise_not_implemented())
# pylint: enable=protected-access


class DeletedBuffer(object): pass
deleted_buffer = DeletedBuffer()

for device_array in [_CppDeviceArray, _DeviceArray]:
  core.literalable_types.add(device_array)
  core.pytype_aval_mappings[device_array] = ConcreteArray
  pytype_aval_mappings[device_array] = op.attrgetter('aval')
  canonicalize_dtype_handlers[device_array] = identity

def _device_array_constant_handler(c, val, canonicalize_types=True):
  return pyval_to_ir_constants(c, val.device_buffer.to_py())
register_constant_handler(_DeviceArray, _device_array_constant_handler)
register_constant_handler(_CppDeviceArray, _device_array_constant_handler)

def _device_put_device_array(x: Union[DeviceArrayProtocol, _DeviceArray], device: Optional[Device]):
  x = _copy_device_array_to_device(x, device)
  return (x.device_buffer,)
device_put_handlers[_CppDeviceArray] = _device_put_device_array
device_put_handlers[_DeviceArray] = _device_put_device_array

def _copy_device_array_to_device(x: Union[DeviceArrayProtocol, _DeviceArray], device: Optional[xc.Device]) -> Union[DeviceArrayProtocol, _DeviceArray]:
  if device is None:
    # no copying to be done because there's no target specified
    return x
  elif xb.get_device_backend(device).platform == x.device_buffer.platform():
    # source and target platforms are the same
    if x.device_buffer.device() == device:
      # no copying to be done because source equals target
      if x._device == device:
        return x
      else:
        moved_buf = x.device_buffer  # We need to change stickyness
    else:
      # move the buffer with a device-to-device copy
      moved_buf = x.device_buffer.copy_to_device(device)
  else:
    # buffers from different XLA backends are passed through the host.
    backend = xb.get_device_backend(device)
    moved_buf = backend.buffer_from_pyval(x.device_buffer.to_py(), device)
  return make_device_array(x.aval, device, moved_buf)


def _device_put_impl(x, device: Optional[Device] = None):
  if type_is_device_array(x):
    return _copy_device_array_to_device(x, device)

  try:
    a = abstractify(x)
  except TypeError as err:
    raise TypeError(
        f"Argument '{x}' of type {type(x)} is not a valid JAX type") from err
  return aval_to_result_handler(device, a)(*device_put(x, device))

device_put_p = core.Primitive('device_put')
device_put_p.def_impl(_device_put_impl)
device_put_p.def_abstract_eval(lambda x, device=None: x)
translations[device_put_p] = lambda c, x, device=None: x
ad.deflinear2(device_put_p, lambda cotangent, _, **kwargs: [cotangent])
masking.defvectorized(device_put_p)


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
  input_op = xb.parameter(remat_subc, 0, c.get_shape(true_op), replicated=[])
  args = xla_destructure(remat_subc, input_op)
  sub_ctx = ctx.replace(
      builder=remat_subc,
      name_stack=extend_name_stack(ctx.name_stack, wrap_name(name, 'remat')))
  out_nodes = jaxpr_subcomp(sub_ctx, call_jaxpr, (), *args)
  out_node_shapes = [remat_subc.get_shape(o) for o in out_nodes]
  remat_subc = remat_subc.build(xops.Tuple(remat_subc, out_nodes))

  false_op = true_op
  dummy_subc = xc.XlaBuilder("remat_call_dummy_subcomputation")
  xb.parameter(dummy_subc, 0, c.get_shape(false_op), replicated=[])
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
  dummy_input_op = xb.parameter(dummy_subc, 0, c.get_shape(dummy_inputs), replicated=[])
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
  input_op = xb.parameter(cond_subc, 0, c.get_shape(inputs), replicated=[])
  i = xops.GetTupleElement(input_op, 0)
  rng = xops.RngUniform(xops.Constant(cond_subc, np.array(1, dtype=np.int32)),
                        xops.Constant(cond_subc, np.array(2, dtype=np.int32)),
                        xc.Shape.array_shape(xc.PrimitiveType.S32, []))
  cond_subc = cond_subc.build(xops.Lt(i, rng))

  body_subc = xc.XlaBuilder("remat_body_subcomputation")
  input_op = xb.parameter(body_subc, 0, c.get_shape(inputs), replicated=[])
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

register_translation(pe.remat_call_p, _remat_translation_rule)


ad.primitive_transposes[core.named_call_p] = partial(ad.call_transpose,
                                                     core.named_call_p)


def _named_call_translation_rule(ctx, avals_in, avals_out, *in_nodes,
                                 name="core_call", backend=None, call_jaxpr):
  check_backend_matches(backend, ctx.platform)
  c = ctx.builder
  subc = xc.XlaBuilder(name)
  args = [xb.parameter(subc, i, c.GetShape(n)) for i, n in enumerate(in_nodes)]
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
