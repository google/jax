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

# Primitive dispatch and jit dispatch.

import contextlib
from functools import partial
import itertools
import time
from typing import (
    Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union)
from typing_extensions import Protocol
import os
import re
import warnings

from absl import logging
import numpy as np

from jax import core
from jax import linear_util as lu
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.masking as masking
import jax.interpreters.mlir as mlir
import jax.interpreters.xla as xla
import jax.interpreters.partial_eval as pe
from jax.errors import UnexpectedTracerError
from jax._src.abstract_arrays import array_types
from jax._src.config import config, flags
from jax._src import device_array
from jax._src import dtypes
from jax._src import profiler
from jax._src.lib.mlir import ir
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
import jax._src.util as util
from jax._src import traceback_util

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'jax_dump_ir_to', os.getenv('JAX_DUMP_IR_TO', ''),
    help="Path to which HLO/MHLO IR that is emitted by JAX as input to the "
         "compiler should be dumped as text files. Optional. If omitted, JAX "
         "will not dump IR.")


traceback_util.register_exclusion(__file__)

MYPY = False  # Are we currently type checking with mypy?

xe = xc._xla

Backend = xe.Client
Device = xc.Device
Buffer = xe.Buffer

XlaExecutable = xc.Executable

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

# This flag is set on exit; no logging should be attempted
_on_exit = False

### op-by-op execution

ArgSpec = Tuple[core.AbstractValue, Optional[Device]]

def arg_spec(x: Any) -> ArgSpec:
  aval = xla.abstractify(x)
  try:
    return aval, x._device
  except:
    return aval, None


def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  compiled_fun = xla_primitive_callable(prim, *unsafe_map(arg_spec, args),
                                        **params)
  return compiled_fun(*args)

# TODO(phawkins): update code referring to xla.apply_primitive to point here.
xla.apply_primitive = apply_primitive


@util.cache()
def xla_primitive_callable(prim, *arg_specs: ArgSpec, **params):
  avals, arg_devices = util.unzip2(arg_specs)
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


# JIT execution

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

xla.xla_call_p.def_impl(_xla_call_impl)


def _xla_callable_uncached(fun: lu.WrappedFun, device, backend, name,
                           donated_invars, *arg_specs):
  return lower_xla_callable(fun, device, backend, name, donated_invars,
                            *arg_specs).compile().unsafe_call

_xla_callable = lu.cache(_xla_callable_uncached)


@contextlib.contextmanager
def log_elapsed_time(fmt: str):
  if _on_exit:
    yield
  else:
    log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logging.log(log_priority, fmt.format(elapsed_time=elapsed_time))


@profiler.annotate_function
def lower_xla_callable(fun: lu.WrappedFun, device, backend, name,
                       donated_invars, *arg_specs):
  if device is not None and backend is not None:
    raise ValueError("can't specify both a device and a backend for jit, "
                     "got device={} and backend={}".format(device, backend))

  abstract_args, arg_devices = util.unzip2(arg_specs)
  with log_elapsed_time(f"Finished tracing + transforming {fun.__name__} "
                        "for jit in {elapsed_time} sec"):
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(
        fun, abstract_args, pe.debug_info_final(fun, "jit"))
  if any(isinstance(c, core.Tracer) for c in consts):
    raise UnexpectedTracerError("Encountered an unexpected tracer.")
  jaxpr, kept_const_idx, kept_var_idx = _prune_unused_inputs(jaxpr)
  consts = [c for i, c in enumerate(consts) if i in kept_const_idx]
  pruned_arg_specs = (a for i, a in enumerate(arg_specs) if i in kept_var_idx)
  abstract_args, arg_devices = util.unzip2(pruned_arg_specs)
  donated_invars = [
      x for i, x in enumerate(donated_invars) if i in kept_var_idx
  ]
  map(prefetch, itertools.chain(consts, jaxpr_literals(jaxpr)))
  jaxpr = core.apply_outfeed_rewriter(jaxpr)

  nreps = jaxpr_replicas(jaxpr)
  device = _xla_callable_device(nreps, backend, device, arg_devices)
  backend = xb.get_device_backend(device) if device else xb.get_backend(backend)

  # Computations that only produce constants and/or only rearrange their inputs,
  # which are often produced from partial evaluation, don't need compilation,
  # and don't need to evaluate their arguments.
  if not jaxpr.eqns:
    return XlaComputation(
        name, None, True, None, jaxpr=jaxpr, consts=consts, device=device,
        in_avals=abstract_args, out_avals=out_avals, kept_var_idx=kept_var_idx)

  if not _on_exit:
    log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
    if len(abstract_args) > 10:
      msg = f"Compiling {fun.__name__} ({id(fun)}) for {len(abstract_args)} args."
    else:
      msg = f"Compiling {fun.__name__} ({id(fun)} for args {abstract_args}."
    logging.log(log_priority, msg)

  if nreps > 1:
    warnings.warn(
        f"The jitted function {name} includes a pmap. Using "
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

  # pass long arg lists as tuple for TPU
  tuple_args = len(abstract_args) > 100
  axis_env = xla.AxisEnv(nreps, (), ())
  name_stack = xla.extend_name_stack(xla.wrap_name(name, 'jit'))
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  module: Union[str, xc.XlaComputation]
  module_name = f"jit_{fun.__name__}"
  if config.jax_enable_mlir:
    module = mlir.lower_jaxpr_to_module(
        module_name, closed_jaxpr, backend.platform, axis_env, name_stack,
        donated_invars)
  else:
    module = xla.lower_jaxpr_to_xla_module(
        module_name, closed_jaxpr, backend.platform, axis_env,
        name_stack, tuple_args, donated_invars, replicated_args=None,
        arg_partitions=None, out_partitions=None)
  return XlaComputation(
      name, module, False, donated_invars, nreps=nreps, device=device,
      backend=backend, tuple_args=tuple_args, in_avals=abstract_args,
      out_avals=out_avals, kept_var_idx=kept_var_idx)


def prefetch(x):
  if isinstance(x, device_array.DeviceArray):
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


def jaxpr_has_pmap(jaxpr):
  """Whether there is an xla_pmap primitive anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if 'xla_pmap' in eqn.primitive.name:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_pmap(subjaxpr):
      return True
  return False

def _prune_unused_inputs(
    jaxpr: core.Jaxpr) -> Tuple[core.Jaxpr, Set[int], Set[int]]:
  used = {v for v in jaxpr.outvars if isinstance(v, core.Var)}
  # TODO(zhangqiaorjc): Improve the DCE algorithm by also pruning primitive
  # applications that do not produce used outputs. Must handle side-effecting
  # primitives and nested jaxpr.
  used.update(
      v for eqn in jaxpr.eqns for v in eqn.invars if isinstance(v, core.Var))
  kept_const_idx, new_constvars = util.unzip2(
      (i, v) for i, v in enumerate(jaxpr.constvars) if v in used)
  kept_var_idx, new_invars = util.unzip2(
      (i, v) for i, v in enumerate(jaxpr.invars) if v in used)
  new_jaxpr = core.Jaxpr(new_constvars, new_invars, jaxpr.outvars, jaxpr.eqns)
  return new_jaxpr, set(kept_const_idx), set(kept_var_idx)


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
  elif eqn.primitive in xla._initial_style_primitives:
    return initial_style_primitive_replicas(eqn.params)
  else:
    return 1

def initial_style_primitive_replicas(params):
  return max(core.traverse_jaxpr_params(jaxpr_replicas, params).values(), default=1)


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


# Argument and result handlers

num_buffers_handlers: Dict[Type[core.AbstractValue],
                           Callable[[core.AbstractValue], int]] = {}

def aval_to_num_buffers(aval: core.AbstractValue) -> int:
  """Returns the number of buffers in the runtime representation of `aval`.

  In general this may differ from the number of buffers in the compiler-IR
  representation of the same value.
  """
  try:
    return num_buffers_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No num_buffers handler for type: {type(aval)}") from err

# TODO(phawkins): use zero buffers to represent a unit.
num_buffers_handlers[core.AbstractUnit] = lambda _: 1
num_buffers_handlers[core.AbstractToken] = lambda _: 1
num_buffers_handlers[core.ShapedArray] = lambda _: 1
num_buffers_handlers[core.ConcreteArray] = lambda _: 1


if MYPY:
  ResultHandler = Any
else:
  class ResultHandler(Protocol):
    def __call__(self, *args: xla.Buffer) -> Any:
      """Boxes raw buffers into their user-facing representation."""

def aval_to_result_handler(sticky_device: Optional[Device],
                           aval: core.AbstractValue) -> ResultHandler:
  try:
    return result_handlers[type(aval)](sticky_device, aval)
  except KeyError as err:
    raise TypeError(f"No result handler for type: {type(aval)}") from err

def array_result_handler(sticky_device: Optional[Device],
                          aval: core.ShapedArray):
  if aval.dtype is dtypes.float0:
    return lambda _: np.zeros(aval.shape, dtypes.float0)
  return partial(device_array.make_device_array, core.raise_to_shaped(aval),
                 sticky_device)


result_handlers: Dict[
    Type[core.AbstractValue],
    Callable[[Optional[Device], Any], ResultHandler]] = {}
result_handlers[core.AbstractUnit] = lambda _, __: lambda _: core.unit
result_handlers[core.AbstractToken] = lambda _, __: lambda _: core.token
result_handlers[core.ShapedArray] = array_result_handler
result_handlers[core.ConcreteArray] = array_result_handler


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


def _execute_compiled(name: str, compiled: XlaExecutable,
                      output_buffer_counts: Optional[Sequence[int]],
                      result_handlers, kept_var_idx, *args):
  device, = compiled.local_devices()
  input_bufs = util.flatten(
      device_put(x, device) for i, x in enumerate(args) if i in kept_var_idx)
  out_bufs = compiled.execute(input_bufs)
  check_special(name, out_bufs)
  if output_buffer_counts is None:
    return (result_handlers[0](*out_bufs),)
  return tuple(
      handler(*bs) for handler, bs in
      unsafe_zip(result_handlers, util.unflatten(out_bufs, output_buffer_counts)))


def _execute_replicated(name: str, compiled: XlaExecutable,
                        output_buffer_counts: Optional[Sequence[int]],
                        result_handlers, kept_var_idx, *args):
  input_bufs = [
      util.flatten(
        device_put(x, device) for i, x in enumerate(args) if i in kept_var_idx)
      for device in compiled.local_devices()
  ]
  out_bufs = [
      buf[0] for buf in compiled.execute_sharded_on_local_devices(
          list(zip(*input_bufs)))
  ]
  check_special(name, out_bufs)
  if output_buffer_counts is None:
    return (result_handlers[0](*out_bufs),)
  return tuple(
      handler(*bs) for handler, bs in
      unsafe_zip(result_handlers, util.unflatten(out_bufs, output_buffer_counts)))


def _execute_trivial(jaxpr, device: Optional[Device], consts, avals, handlers,
                     kept_var_idx, *args):
  env = {core.unitvar: core.unit}
  pruned_args = (x for i, x in enumerate(args) if i in kept_var_idx)
  map(env.setdefault, jaxpr.invars, pruned_args)
  map(env.setdefault, jaxpr.constvars, consts)
  outs = [xla.canonicalize_dtype(v.val) if type(v) is core.Literal else env[v]
          for v in jaxpr.outvars]
  return [_copy_device_array_to_device(x, device) if device_array.type_is_device_array(x)
          else h(*device_put(x, device)) for h, x in zip(handlers, outs)]


class XlaComputation:
  name: str
  _is_trivial: bool
  _executable: Optional['XlaCompiledComputation']
  _donated_invars: Optional[Sequence[bool]]

  def __init__(self, name: str, hlo, is_trivial: bool,
               donated_invars: Optional[Sequence[bool]],
               **compile_args):
    self.name = name
    self._hlo = hlo
    self._is_trivial = is_trivial
    self._donated_invars = donated_invars
    self._executable = None
    self.compile_args = compile_args

  def is_trivial(self):
    return self._is_trivial

  def hlo(self) -> xc.XlaComputation:
    if self.is_trivial():
      raise ValueError("A trivial computation has no HLO")
    if isinstance(self._hlo, xc.XlaComputation):
      return self._hlo
    return xe.mlir.mlir_module_to_xla_computation(
        mlir.module_to_string(self._hlo),
        use_tuple_args=self.compile_args["tuple_args"])

  def mhlo(self) -> str:
    if self.is_trivial():
      raise ValueError("A trivial computation has no MHLO")
    if isinstance(self._hlo, xc.XlaComputation):
      return xe.mlir.xla_computation_to_mlir_module(self._hlo)
    return mlir.module_to_string(self._hlo)

  def compile(self) -> 'XlaCompiledComputation':
    if self._executable is None:
      if self.is_trivial():
        self._executable = XlaCompiledComputation.from_trivial_jaxpr(
            **self.compile_args)
      else:
        self._executable = XlaCompiledComputation.from_xla_computation(
            self.name, self._hlo, **self.compile_args)

    return self._executable

@profiler.annotate_function
def backend_compile(backend, built_c, options):
  # we use a separate function call to ensure that XLA compilation appears
  # separately in Python profiling results
  return backend.compile(built_c, compile_options=options)

# TODO(phawkins): update users.
xla.backend_compile = backend_compile

_ir_dump_counter = itertools.count()

def _make_string_safe_for_filename(s: str) -> str:
  return re.sub(r'[^\w.)( -]', '', s)

def _dump_ir_to_file(name: str, ir: str):
  id = next(_ir_dump_counter)
  name = f"jax_ir{id}_{_make_string_safe_for_filename(name)}.mlir"
  name = os.path.join(FLAGS.jax_dump_ir_to, name)
  with open(name, "w") as f:
    f.write(ir)


def compile_or_get_cached(backend, computation, compile_options):
  # Avoid import cycle between jax and jax.experimental
  from jax.experimental.compilation_cache import compilation_cache as cc

  if isinstance(computation, ir.Module):
    module_name = computation.operation.name
    computation = mlir.module_to_string(computation)
  else:
    module_name = computation.name()

  # Persistent compilation cache only implemented on TPU.
  # TODO(skye): add warning when initializing cache on unsupported default platform
  if cc.is_initialized() and backend.platform == 'tpu':
    cached_executable = cc.get_executable(computation, compile_options, backend)
    if cached_executable is not None:
      logging.info('Persistent compilation cache hit for %s.', module_name)
      return cached_executable
    else:
      compiled = backend_compile(backend, computation, compile_options)
      cc.put_executable(module_name, computation, compile_options, compiled,
                        backend)
      return compiled

  if FLAGS.jax_dump_ir_to:
    ir_str = (computation if isinstance(computation, str)
              else computation.as_hlo_text())
    _dump_ir_to_file(module_name, ir_str)
  return backend_compile(backend, computation, compile_options)


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
      device: Optional[Device],
      backend,
      tuple_args: bool,
      in_avals,
      out_avals,
      kept_var_idx) -> 'XlaCompiledComputation':
    sticky_device = device
    result_handlers = map(partial(aval_to_result_handler, sticky_device),
                          out_avals)
    options = xb.get_compile_options(
        num_replicas=nreps,
        num_partitions=1,
        device_assignment=(sticky_device.id,) if sticky_device else None)
    options.parameter_is_tupled_arguments = tuple_args
    with log_elapsed_time(f"Finished XLA compilation of {name} "
                          "in {elapsed_time} sec"):
      compiled = compile_or_get_cached(backend, xla_computation, options)
    buffer_counts = (None if len(out_avals) == 1 else
                     [aval_to_num_buffers(aval) for aval in out_avals])
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


def device_put(x, device: Optional[Device] = None) -> Tuple[Any]:
  x = xla.canonicalize_dtype(x)
  try:
    return device_put_handlers[type(x)](x, device)
  except KeyError as err:
    raise TypeError(f"No device_put handler for type: {type(x)}") from err

# TODO(phawkins): update users.
xla.device_put = device_put

def _device_put_array(x, device: Optional[Device]):
  backend = xb.get_device_backend(device)
  if x.dtype is dtypes.float0:
    x = np.zeros(x.shape, dtype=np.dtype(bool))
  return (backend.buffer_from_pyval(x, device),)

def _device_put_scalar(x, device):
  return _device_put_array(dtypes.coerce_to_array(x), device)

def _device_put_unit(_, device):
  backend = xb.get_device_backend(device)
  return (backend.buffer_from_pyval(np.zeros((), dtype=np.dtype(np.bool_)),
                                    device),)

def _device_put_token(_, device):
  backend = xb.get_device_backend(device)
  return (backend.buffer_from_pyval(np.zeros((), dtype=np.dtype(np.bool_)),
                                    device),)

_scalar_types = dtypes.python_scalar_dtypes.keys()

device_put_handlers: Dict[Any, Callable[[Any, Optional[Device]], Tuple[Any]]] = {}
device_put_handlers.update((t, _device_put_array) for t in array_types)
device_put_handlers.update((t, _device_put_scalar) for t in _scalar_types)
device_put_handlers[core.Unit] = _device_put_unit
device_put_handlers[core.Token] = _device_put_token


def _device_put_device_array(x: Union[device_array.DeviceArrayProtocol, device_array._DeviceArray], device: Optional[Device]):
  x = _copy_device_array_to_device(x, device)
  return (x.device_buffer,)
for t in device_array.device_array_types:
  device_put_handlers[t] = _device_put_device_array

def _copy_device_array_to_device(x: Union[device_array.DeviceArrayProtocol, device_array._DeviceArray], device: Optional[xc.Device]) -> Union[device_array.DeviceArrayProtocol, device_array._DeviceArray]:
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
  return device_array.make_device_array(x.aval, device, moved_buf)


def _device_put_impl(x, device: Optional[Device] = None):
  if device_array.type_is_device_array(x):
    return _copy_device_array_to_device(x, device)

  try:
    a = xla.abstractify(x)
  except TypeError as err:
    raise TypeError(
        f"Argument '{x}' of type {type(x)} is not a valid JAX type") from err
  return aval_to_result_handler(device, a)(*device_put(x, device))

device_put_p = core.Primitive('device_put')
device_put_p.def_impl(_device_put_impl)
device_put_p.def_abstract_eval(lambda x, device=None: x)
xla.translations[device_put_p] = lambda c, x, device=None: x
ad.deflinear2(device_put_p, lambda cotangent, _, **kwargs: [cotangent])
masking.defvectorized(device_put_p)
batching.defvectorized(device_put_p)

def _device_put_lowering(ctx, avals_in, avals_out, x, *, device):
  return [x]


mlir.register_lowering(device_put_p, _device_put_lowering)
