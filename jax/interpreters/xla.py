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
import itertools as it
import operator as op
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Type,
                    Tuple, Union, NamedTuple)
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
                    Literal, pp_eqn_compact, raise_to_shaped, abstract_token)
from jax._src.pprint_util import pp
from .._src.util import (partial, partialmethod, cache, prod, unzip2,
                    extend_name_stack, wrap_name, safe_zip, safe_map)
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from . import partial_eval as pe
from . import ad
from . import masking

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

xe = xc._xla
xops = xc._xla.ops

# Types
Backend = Any  # xc.LocalBackend (why does mypy not like this?)
Device = Any  # xc.Device
PyLocalBuffer = Any

XlaOp = Any  # xla_extension.XlaOp
XlaShape = Any # xla_client.Shape
XlaComputationBuilder = Any  # xla_bridge._JaxComputationBuilder
XlaExecutable = Any # xla_extension.LocalExecutable

# This flag is set on exit; no logging should be attempted
_on_exit = False

def identity(x): return x

_scalar_types = dtypes.python_scalar_dtypes.keys()

# unit representation
def _make_unit_constant(c): return xb.constant_general(c, np.zeros((), dtype=np.dtype('bool')))
def _make_unit_shape(_): return (xc.Shape.array_shape(np.dtype('bool'), ()),)
def _device_put_unit(_, device):
  backend = xb.get_device_backend(device)
  return (backend.buffer_from_pyval(np.zeros((), dtype=np.dtype('bool')),
                                    device),)
def _make_array_shape(a):
  if a.dtype is dtypes.float0:
    return (xc.Shape.array_shape(np.dtype('bool'), a.shape),)
  else:
    return (xc.Shape.array_shape(a.dtype, a.shape),)

tracebacks = {}
def make_op_metadata(primitive: core.Primitive,
                     params: Dict, *,
                     name_stack: str = "",
                     source_info: Optional[source_info_util.Traceback] = None
                     ) -> xc.OpMetadata:
  tracebacks[str(pp(name_stack) >> pp_eqn_compact(primitive.name, params))] = source_info
  frame = source_info_util.user_frame(source_info) if source_info else None
  return xc.OpMetadata(
        op_type=primitive.name,
        op_name=str(pp(name_stack) >> pp_eqn_compact(primitive.name, params)),
        source_file=frame.file_name if frame else None,
        source_line=frame.line_num if frame else None)

### handlers

xb.register_constant_handler(core.Unit, lambda c, *_: _make_unit_constant(c))

def aval_to_xla_shapes(aval):
  try:
    return xla_shape_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No xla_shape_handler for type: {type(aval)}") from err

xla_shape_handlers: Dict[Type[core.AbstractValue], Callable] = {
    core.AbstractUnit: _make_unit_shape,
    ShapedArray: _make_array_shape,
    ConcreteArray: _make_array_shape,
}

def aval_to_result_handler(device: Optional[Device], aval: core.AbstractValue) -> Callable:
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
  compiled_fun = xla_primitive_callable(prim, *unsafe_map(arg_spec, args), **params)
  return compiled_fun(*args)


def _partition_outputs(avals, outs):
  nouts = [aval._num_buffers for aval in avals]
  if config.jax_enable_checks:
    assert sum(nouts) == len(outs), f"Internal error: sum(nouts)={sum(nouts)} should equal len(outs)={len(outs)}."
  outs = iter(outs)
  return [[next(outs) for _ in range(nout)] for nout in nouts]


@cache()
def xla_primitive_callable(prim, *arg_specs: ArgSpec, **params):
  avals, arg_devices = unzip2(arg_specs)
  donated_invars = (False,) * len(arg_specs)
  device = _device_from_arg_devices(arg_devices)
  backend = xb.get_device_backend(device)
  if primitive_uses_outfeed(prim, params):
    # We use the _xla_callable path, where we pre-process the primitives
    def prim_fun(*args):
      return prim.bind(*args, **params)
    return _xla_callable(lu.wrap_init(prim_fun), device, None, "prim", donated_invars,
                         *arg_specs)
  aval_out = prim.abstract_eval(*avals, **params)
  if not prim.multiple_results:
    handle_result = aval_to_result_handler(device, aval_out)
  else:
    handlers = map(partial(aval_to_result_handler, device), aval_out)
    handle_result = lambda *bufs:\
      tuple(handler(*bs) for handler, bs in zip(handlers, _partition_outputs(aval_out, bufs)))
  tuple_args = len(avals) > 100
  if prim in initial_style_translations:
    nreps = initial_style_primitive_replicas(params)
  else:
    nreps = 1

  if nreps > xb.device_count(backend):
    raise ValueError(
        f"compiling a primitive computation `{prim}` that requires {nreps} "
        f"replicas, but only {xb.device_count(backend)} XLA devices are "
        f"available on backend {backend.platform}.")
  built_c = primitive_computation(prim, AxisEnv(nreps, (), ()), backend,
                                  tuple_args, *avals, **params)
  options = xb.get_compile_options(
      num_replicas=nreps,
      num_partitions=1,
      device_assignment=device and (device.id,))
  options.parameter_is_tupled_arguments = tuple_args
  compiled = backend_compile(backend, built_c, options)
  if nreps == 1:
    return partial(_execute_compiled_primitive, prim, compiled, handle_result)
  else:
    return partial(_execute_replicated_primitive, prim, compiled, handle_result)

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

@cache()
def primitive_computation(prim, axis_env, backend, tuple_args, *avals, **params):
  c = xb.make_computation_builder(f"primitive_computation_{prim.name}")
  op_metadata = make_op_metadata(prim, params)
  c.set_op_metadata(op_metadata)
  platform = xb.get_backend(backend).platform
  xla_args, _ = _xla_callable_args(c, avals, tuple_args)
  # return val always set as a side-effect on c
  if prim in backend_specific_translations[platform]:
    rule = backend_specific_translations[platform][prim]
    ans = rule(c, *xla_args, **params)
  elif prim in translations:
    rule = translations[prim]
    ans = rule(c, *xla_args, **params)
  elif prim in translations_with_avals:
    rule = translations_with_avals[prim]
    ans = rule(c, avals, xla_args, params)
  elif prim in initial_style_translations:
    rule = initial_style_translations[prim]
    ans = rule(c, axis_env, extend_name_stack(prim.name), avals, backend,
         *xla_args, **params)
  else:
    raise NotImplementedError(f"XLA translation rule for {prim!r} on platform {platform!r} not found")
  assert isinstance(ans, xe.XlaOp)
  c.clear_op_metadata()
  try:
    return c.build(ans)
  except RuntimeError as e:
    msg = (" ".join(map(str, e.args)) + "\n"
           "This is a bug in JAX's shape-checking rules; please report it!\n"
           "https://github.com/google/jax/issues\n")
    raise RuntimeError(msg) from e

def primitive_subcomputation(prim, *avals, **params):
  axis_env = AxisEnv(1, (), ())
  return primitive_computation(prim, axis_env, None, False, *avals, **params)

def backend_compile(backend, built_c, options):
  # we use a separate function call to ensure that XLA compilation appears
  # separately in Python profiling results
  return backend.compile(built_c, compile_options=options)

def _execute_compiled_primitive(prim, compiled, result_handler, *args):
  device, = compiled.local_devices()
  input_bufs = list(it.chain.from_iterable(device_put(x, device) for x in args if x is not token))
  out_bufs = compiled.execute(input_bufs)
  check_special(prim.name, out_bufs)
  return result_handler(*out_bufs)

def _execute_replicated_primitive(prim, compiled, result_handler, *args):
  input_bufs = [
      list(it.chain.from_iterable(device_put(x, device) for x in args if x is not token))
      for device in compiled.local_devices()]
  out_bufs = [
      buf[0] for buf in compiled.execute_sharded_on_local_devices(
          list(zip(*input_bufs)))
  ]
  return result_handler(*out_bufs)

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
  return map(func, vars, _partition_outputs([v.aval for v in vars], nodes))

def jaxpr_subcomp(c, jaxpr, backend, axis_env, consts, name_stack, *args):
  if backend not in ('cpu', 'gpu', 'tpu'):
    platform = xb.get_backend(backend).platform  # canonicalize
  else:
    platform = backend

  def read(v):
    if type(v) is Literal:
      return xb.constant_general(c, canonicalize_dtype(v.val))
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

  env = {}
  _partitionmap(write, [core.unitvar], _make_unit_constant(c))
  _partitionmap(write, jaxpr.constvars, consts)
  _partitionmap(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    op_metadata = make_op_metadata(
        eqn.primitive, eqn.params, name_stack=name_stack,
        source_info=eqn.source_info)
    c.set_op_metadata(op_metadata)
    in_nodes = _flatmap(read, eqn.invars)
    # TODO(jakevdp): migrate `translations` table to `translations_with_avals`
    if eqn.primitive in backend_specific_translations[platform]:
      rule = backend_specific_translations[platform][eqn.primitive]
      ans = rule(c, *in_nodes, **eqn.params)
    elif eqn.primitive in translations:
      ans = translations[eqn.primitive](c, *in_nodes, **eqn.params)
    elif eqn.primitive in translations_with_avals:
      rule = translations_with_avals[eqn.primitive]
      ans = rule(c, map(aval, eqn.invars), in_nodes, eqn.params)
    elif eqn.primitive in initial_style_translations:
      new_params = check_backend_params(eqn.params, backend)
      rule = initial_style_translations[eqn.primitive]
      ans = rule(c, axis_env, extend_name_stack(name_stack, eqn.primitive.name),
                 map(aval, eqn.invars), backend, *in_nodes, **new_params)
    elif eqn.primitive in parallel_translations:
      rule = parallel_translations[eqn.primitive]
      ans = rule(c, *in_nodes, axis_env=axis_env, platform=platform, **eqn.params)
    elif eqn.primitive in call_translations:
      new_params = check_backend_params(eqn.params, backend)
      rule = call_translations[eqn.primitive]
      ans = rule(c, axis_env, in_nodes,
                 name_stack, backend=backend, **new_params)
    else:
      raise NotImplementedError(
          f"XLA translation rule for primitive '{eqn.primitive.name}' not found")

    assert isinstance(ans, xe.XlaOp)
    c.get_shape(ans)  # force xla to do shape error checking
    if eqn.primitive.multiple_results or any(v.aval._num_buffers > 1 for v in eqn.outvars):
      out_nodes = xla_destructure(c, ans)
    else:
      out_nodes = [ans]
    c.clear_op_metadata()
    _partitionmap(write, eqn.outvars, out_nodes)
  return _flatmap(read, jaxpr.outvars)


def xla_destructure(c, ans):
  num_elements = len(c.get_shape(ans).tuple_shapes())
  return [xops.GetTupleElement(ans, i) for i in range(num_elements)]

def check_backend_params(params, outer_backend):
  # For nested calls, the outermost call sets the backend for all inner calls;
  # it's an error if the inner call has a conflicting explicit backend spec.
  inner_backend = params.get('backend', None)
  if inner_backend and inner_backend != outer_backend:
    raise ValueError(
        f"Outer-jit backend specification {outer_backend} must match explicit "
        f"inner-jit backend specification {inner_backend}.")
  return {k: params[k] for k in params if k != 'backend'}


class AxisEnv(NamedTuple):
  """Represents a pmap mesh (only along the replica axes)."""
  nreps: int
  names: Tuple[Any, ...]
  sizes: Tuple[int, ...]

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

def jaxpr_replicas(jaxpr: core.Jaxpr) -> int:
  """The number of replicas needed for a jaxpr.

  For a eqn, multiply the `axis_size` with the `jaxpr_replicas` of the
  subjaxprs. For a list of eqns, take the maximum number of replicas.
  """
  return max(unsafe_map(eqn_replicas, jaxpr.eqns), default=1)

# TODO(mattjj): this function assumes that only pmap has a parameter named
# axis_size, and that it corresponds to cross-replica mapping
def eqn_replicas(eqn):
  call_jaxpr = eqn.params.get("call_jaxpr")
  if call_jaxpr:
    return eqn.params.get('axis_size', 1) * jaxpr_replicas(call_jaxpr)
  elif eqn.primitive in initial_style_translations:
    return initial_style_primitive_replicas(eqn.params)
  else:
    return 1

def initial_style_primitive_replicas(params):
  return max(core.traverse_jaxpr_params(jaxpr_replicas, params), default=1)

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
    if eqn.primitive in parallel_translations:
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
    return compiled_fun(*args)
  except FloatingPointError:
    assert config.jax_debug_nans or config.jax_debug_infs  # compiled_fun can only raise in this case
    print("Invalid value encountered in the output of a jit function. "
          "Calling the de-optimized version.")
    # We want to run the wrapped function again (after _xla_callable already ran
    # it), but linear_util.WrappedFun instances are meant to be run only once.
    # In addition to re-executing the Python code, which is usually undesirable
    # but which config.jax_debug_nans is meant to opt into, we'll be re-executing
    # any linear_util.py-style side effects, i.e. re-populating Stores created
    # by any transformation_with_aux's applied to fun. Since this is
    # intentional here, to avoid "Store occupied" errors we reset the stores to
    # be empty.
    for store in fun.stores: store and store.reset()
    with core.new_sublevel():
      return fun.call_wrapped(*args)  # probably won't return

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
      id_: xb.constant_general(c, const) for id_, const in unique_consts.items()}
  return [c for const in consts for c in xla_consts[id(const)]]

@lu.cache
def _xla_callable(fun: lu.WrappedFun, device, backend, name, donated_invars, *arg_specs):
  if device is not None and backend is not None:
    raise ValueError("can't specify both a device and a backend for jit, "
                     "got device={} and backend={}".format(device, backend))

  abstract_args, arg_devices = unzip2(arg_specs)
  jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(
      fun, abstract_args, pe.debug_info_final(fun, "jit"))
  if any(isinstance(c, core.Tracer) for c in consts):
    raise core.UnexpectedTracerError("Encountered an unexpected tracer.")
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
  backend = xb.get_device_backend(device) if device else (
      xb.get_backend(backend) if backend is not None else None)
  result_handlers = map(partial(aval_to_result_handler, device), out_avals)

  # Computations that only produce constants and/or only rearrange their inputs,
  # which are often produced from partial evaluation, don't need compilation,
  # and don't need to evaluate their arguments.
  if not jaxpr.eqns:
    return partial(_execute_trivial, jaxpr, device, consts, out_avals,
                   result_handlers, kept_var_idx)

  if not _on_exit:
    log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
    logging.log(log_priority, "Compiling %s (%s) for args %s.",
                fun.__name__, id(fun), abstract_args)

  if nreps > 1:
    warn(f"The jitted function {fun.__name__} includes a pmap. Using "
         "jit-of-pmap can lead to inefficient data movement, as the outer jit "
         "does not preserve sharded data representations and instead collects "
         "input and output arrays onto a single device. "
         "Consider removing the outer jit unless you know what you're doing. "
         "See https://github.com/google/jax/issues/2926.")

  if nreps > xb.device_count(backend):
    raise ValueError(
        f"compiling computation that requires {nreps} replicas, but only "
        f"{xb.device_count(backend)} XLA devices are available")

  if xb.process_count() > 1 and (nreps > 1 or jaxpr_has_pmap(jaxpr)):
    raise NotImplementedError(
        "jit of multi-host pmap not implemented (and jit-of-pmap can cause "
        "extra data movement anyway, so maybe you don't want it after all).")

  tuple_args = len(abstract_args) > 100  # pass long arg lists as tuple for TPU

  c = xb.make_computation_builder("jit_{}".format(fun.__name__))
  xla_consts = _xla_consts(c, consts)
  xla_args, donated_invars = _xla_callable_args(c, abstract_args, tuple_args,
                                                donated_invars=donated_invars)
  out_nodes = jaxpr_subcomp(
      c, jaxpr, backend.platform if backend is not None else None,
      AxisEnv(nreps, (), ()), xla_consts,
      extend_name_stack(wrap_name(name, 'jit')), *xla_args)
  backend = xb.get_backend(backend)
  out_tuple = xops.Tuple(c, out_nodes)
  if backend.platform in ("gpu", "tpu"):
    donated_invars = set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
  if any(donated_invars):
    # TODO(tomhennigan): At call time we should mark these buffers as deleted.
    unused_donations = [str(c.GetShape(a))
                        for a, d in zip(xla_args, donated_invars) if d]
    warn("Some donated buffers were not usable: {}".format(", ".join(unused_donations)))
  built = c.build(out_tuple)

  options = xb.get_compile_options(
      num_replicas=nreps,
      num_partitions=1,
      device_assignment=(device.id,) if device else None)
  options.parameter_is_tupled_arguments = tuple_args
  compiled = backend_compile(backend, built, options)
  if nreps == 1:
    return partial(_execute_compiled, compiled, out_avals, result_handlers,
                   kept_var_idx)
  else:
    return partial(_execute_replicated, compiled, out_avals, result_handlers,
                   kept_var_idx)


def set_up_aliases(c, xla_args, out_tuple, donated_args, tuple_args):
  """Configures input/output "must" aliasing based on `donated_args`."""
  # First for every input array add it to `donations` iff it is a member of
  # `donated_args`.
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
  for output_index, element in flatten_shape(c.GetShape(out_tuple)):
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


def _execute_compiled(compiled: XlaExecutable, avals, handlers, kept_var_idx,
                      *args):
  device, = compiled.local_devices()
  input_bufs = list(
      it.chain.from_iterable(
          device_put(x, device)
          for i, x in enumerate(args)
          if x is not token and i in kept_var_idx))
  out_bufs = compiled.execute(input_bufs)
  check_special(xla_call_p.name, out_bufs)
  return [handler(*bs) for handler, bs in zip(handlers, _partition_outputs(avals, out_bufs))]


def _execute_replicated(compiled: XlaExecutable, avals, handlers, kept_var_idx,
                        *args):
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
  check_special(xla_call_p.name, out_bufs)
  return [handler(*bs) for handler, bs in zip(handlers, _partition_outputs(avals, out_bufs))]


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

def _xla_call_partial_eval_update_params(params, _, __, in_unknowns, ___):
  call_jaxpr = params['call_jaxpr']
  donated_invars = params['donated_invars']
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


def _xla_call_translation_rule(c, axis_env, in_nodes, name_stack, backend, name,
                               call_jaxpr, donated_invars, inline=None, device=None):
  del device, donated_invars, inline  # Ignored.
  subc = xb.make_computation_builder(f"jit_{name}")
  args = [xb.parameter(subc, i, c.get_shape(n)) for i, n in enumerate(in_nodes)]
  out_nodes = jaxpr_subcomp(subc, call_jaxpr, backend, axis_env, (),
                            extend_name_stack(name_stack, wrap_name(name, 'jit')), *args)
  subc = subc.build(xops.Tuple(subc, out_nodes))
  return xops.Call(c, subc, list(in_nodes))
ad.primitive_transposes[xla_call_p] = partial(ad.call_transpose, xla_call_p)


### translation tables

translations: Dict[core.Primitive, Callable] = {}
translations_with_avals: Dict[core.Primitive, Callable] = {}
parallel_translations: Dict[core.Primitive, Callable] = {}
initial_style_translations: Dict[core.Primitive, Callable] = {}
call_translations: Dict[core.Primitive, Callable] = {}
backend_specific_translations: Dict[str, Dict[core.Primitive, Callable]] = defaultdict(dict)

call_translations[xla_call_p] = _xla_call_translation_rule

def zeros_like_translation_rule(c, x):
  shape = c.get_shape(x)
  assert not shape.is_tuple()
  zero = xb.constant(c, np.array(0, shape.element_type()))
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

def lower_fun(fun, multiple_results, parallel=False, with_avals=False, backend=None):
  # TODO(jakevdp): migrate dependent code & always use the with_avals=True.
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
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, avals)
    outs = jaxpr_subcomp(c, jaxpr, backend, axis_env, _xla_consts(c, consts), '',
                         *xla_args)
    if multiple_results or any(v.aval._num_buffers > 1 for v in jaxpr.outvars):
      return xops.Tuple(c, outs)
    else:
      assert len(outs) == 1, outs
      return outs[0]

  return f_with_avals if with_avals else f

def _array_aval_from_xla_shape(xla_shape):
  # This function instantiates the assumption that we can map fro XLA array
  # types to JAX array types.
  # TODO(mattjj): remove assumption can map XLA array types to JAX array types
  assert not xla_shape.is_tuple()
  return ShapedArray(xla_shape.dimensions(), xla_shape.numpy_dtype())

def lower_fun_initial_style(fun):
  def f(c, axis_env, name_stack, avals, backend, *xla_args, **params):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(lu.wrap_init(fun, params), avals)
    outs = jaxpr_subcomp(c, jaxpr, backend, axis_env, _xla_consts(c, consts),
                         name_stack, *xla_args)
    return xops.Tuple(c, outs)
  return f


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
    device_buffer: PyLocalBuffer,
) -> Union[PyLocalBuffer, "_DeviceArray"]:
  """Returns a DeviceArray implementation based on arguments.

  This is to be used only within JAX. It will return either a PythonDeviceArray
  or a C++ equivalent implementation.
  """
  if (isinstance(device_buffer, _CppDeviceArray)):

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
               device_buffer: PyLocalBuffer):
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
      self._npy_value = self.device_buffer.to_py()
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
    return self.device_buffer.__cuda_array_interface__


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
    dtype_str = 'dtype={})'.format(self.dtype.name)
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

  # clobbered when jax.numpy is imported, but useful in tests
  setattr(device_array, "__eq__", lambda self, other: self._value == other)

  def __hash__(self):
    raise TypeError("JAX DeviceArray, like numpy.ndarray, is not hashable.")

  setattr(device_array, "__hash__", __hash__)

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
  return xb.constant_general(c, val.device_buffer.to_py())
xb.register_constant_handler(_DeviceArray, _device_array_constant_handler)
xb.register_constant_handler(_CppDeviceArray, _device_array_constant_handler)

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
    zero = xb.constant(c, np.array(0, dtype=dtype))
    return xops.Broadcast(zero, shape)
  else:
    # It is a token
    return xops.CreateToken(c)


def _remat_using_cond(
    c, axis_env, in_nodes, name_stack, backend, name, call_jaxpr):
  """Lower remat to a Conditional which always returns true. This:
    1. Circumvents common subexpression elimination.
    2. In common case of `jax.grad(jax.remat(f))`, ensures the remat blocks
       occur after the primal blocks, because cotangent is an input to the
       Conditional."""
  # Fake condition which always selects True branch.
  rng = xops.RngUniform(xb.constant(c, np.array(0, dtype=np.float32)),
                        xb.constant(c, np.array(1, dtype=np.float32)),
                        xc.Shape.array_shape(xc.PrimitiveType.F32, []))
  pred = xops.Lt(rng, xb.constant(c, np.array(2, dtype=np.float32)))

  true_op = xops.Tuple(c, in_nodes)
  remat_subc = xb.make_computation_builder("remat_call_subcomputation")
  input_op = xb.parameter(remat_subc, 0, c.get_shape(true_op), replicated=[])
  args = xla_destructure(remat_subc, input_op)
  out_nodes = jaxpr_subcomp(remat_subc, call_jaxpr, backend, axis_env, (),
                            extend_name_stack(name_stack, wrap_name(name, 'remat')),
                            *args)
  out_node_shapes = [remat_subc.get_shape(o) for o in out_nodes]
  remat_subc = remat_subc.build(xops.Tuple(remat_subc, out_nodes))

  false_op = true_op
  dummy_subc = xb.make_computation_builder("remat_call_dummy_subcomputation")
  xb.parameter(dummy_subc, 0, c.get_shape(false_op), replicated=[])
  out_nodes = [_zeros(dummy_subc, s) for s in out_node_shapes]
  dummy_subc = dummy_subc.build(xops.Tuple(dummy_subc, out_nodes))

  return xops.Conditional(pred, true_op, remat_subc, false_op, dummy_subc)


def _remat_using_while(
    c, axis_env, in_nodes, name_stack, backend, name, call_jaxpr):
  """Lower remat to a single iteration while loop."""
  # Dummy subc for getting subcomp shapes.
  dummy_inputs = xops.Tuple(c, in_nodes)
  dummy_subc = xb.make_computation_builder("remat_dummy_subcomputation")
  dummy_input_op = xb.parameter(dummy_subc, 0, c.get_shape(dummy_inputs), replicated=[])
  dummy_args = xla_destructure(dummy_subc, dummy_input_op)
  dummy_subcomp_outs = jaxpr_subcomp(
      dummy_subc, call_jaxpr, backend, axis_env, (),
      extend_name_stack(name_stack, wrap_name(name, "remat")), *dummy_args)
  out_node_shapes = [dummy_subc.get_shape(o) for o in dummy_subcomp_outs]

  i_init = xb.constant(c, np.array(0, dtype=np.int32))
  zeros_like_outs = [_zeros(c, s) for s in out_node_shapes]
  inputs = xops.Tuple(c, [i_init] + in_nodes + zeros_like_outs)

  cond_subc = xb.make_computation_builder("remat_cond_subcomputation")
  input_op = xb.parameter(cond_subc, 0, c.get_shape(inputs), replicated=[])
  i = xops.GetTupleElement(input_op, 0)
  rng = xops.RngUniform(xb.constant(cond_subc, np.array(1, dtype=np.int32)),
                        xb.constant(cond_subc, np.array(2, dtype=np.int32)),
                        xc.Shape.array_shape(xc.PrimitiveType.S32, []))
  cond_subc = cond_subc.build(xops.Lt(i, rng))

  body_subc = xb.make_computation_builder("remat_body_subcomputation")
  input_op = xb.parameter(body_subc, 0, c.get_shape(inputs), replicated=[])
  i, *args = xla_destructure(body_subc, input_op)[:len(in_nodes)+1]
  i_next = xops.Add(i, xb.constant(body_subc, np.array(1, dtype=np.int32)))
  subcomp_outs = jaxpr_subcomp(
      body_subc, call_jaxpr, backend, axis_env, (),
      extend_name_stack(name_stack, wrap_name(name, "remat")), *args)
  out_nodes = [i_next] + args + subcomp_outs
  body_subc = body_subc.build(xops.Tuple(body_subc, out_nodes))
  outs = xops.While(cond_subc, body_subc, inputs)
  return xops.Tuple(c, xla_destructure(c, outs)[len(in_nodes)+1:])


def _remat_translation_rule(c, axis_env, in_nodes,
                            name_stack, backend, name, call_jaxpr,
                            prevent_cse, differentiated, concrete, device=None):
  del device, concrete  # Unused.
  if differentiated and prevent_cse:
    if backend == "gpu":
      return _remat_using_while(
          c, axis_env, in_nodes, name_stack, backend, name, call_jaxpr)
    else:
      return _remat_using_cond(
          c, axis_env, in_nodes, name_stack, backend, name, call_jaxpr)
  else:
    outs = jaxpr_subcomp(c, call_jaxpr, backend, axis_env, (), "", *in_nodes)
    return xops.Tuple(c, outs)

call_translations[pe.remat_call_p] = _remat_translation_rule  # type: ignore


ad.primitive_transposes[core.named_call_p] = partial(ad.call_transpose,
                                                     core.named_call_p)


def _named_call_translation_rule(c, axis_env, in_nodes, name_stack, *,
                                 name="core_call", backend, call_jaxpr):
  subc = xb.make_computation_builder(name)
  args = [xb.parameter(subc, i, c.GetShape(n)) for i, n in enumerate(in_nodes)]
  out_nodes = jaxpr_subcomp(subc, call_jaxpr, backend, axis_env, (),
                            extend_name_stack(name_stack, name), *args)
  subc = subc.Build(xops.Tuple(subc, out_nodes))
  return xops.Call(c, subc, list(in_nodes))
call_translations[core.named_call_p] = _named_call_translation_rule


def _call_translation_rule(c, axis_env, in_nodes, name_stack, *, backend,
                           call_jaxpr):
  return _named_call_translation_rule(
      c, axis_env, in_nodes, name_stack, name="core_call",
      backend=backend, call_jaxpr=call_jaxpr)
call_translations[core.call_p] = _call_translation_rule
