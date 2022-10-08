# Copyright 2018 The JAX Authors.
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
from __future__ import annotations

import atexit
import collections
import contextlib
from functools import partial
import itertools
import time
from typing import (
    Any, Callable, Dict, Optional, Sequence, Set, Tuple, List, Type, Union,
    TYPE_CHECKING)
from typing_extensions import Protocol
import os
import re
import threading
import warnings

from absl import logging
import numpy as np

import jax
from jax import core
from jax import linear_util as lu
from jax.errors import UnexpectedTracerError
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.interpreters.xla as xla
import jax.interpreters.partial_eval as pe
from jax._src import device_array
from jax._src import dtypes
from jax._src import profiler
from jax._src import stages
from jax._src import traceback_util
from jax._src.abstract_arrays import array_types
from jax._src.config import config, flags
from jax._src.lib.mlir import ir
from jax._src.lib import can_execute_with_token
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
import jax._src.util as util
from jax._src.util import flatten, unflatten
from etils import epath


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

XlaLoadedExecutable = xla.XlaLoadedExecutable
CompileOptions = xc.CompileOptions

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

# This flag is set on exit; no logging should be attempted
_on_exit = False

### op-by-op execution

ArgSpec = Tuple[core.AbstractValue, Optional[Device]]

def arg_spec(x: Any) -> ArgSpec:
  from jax._src.sharding import PmapSharding
  from jax.experimental import pjit

  aval = xla.abstractify(x)
  try:
    if config.jax_array:
      if isinstance(x.sharding, PmapSharding):
        return aval, None
      return aval, (pjit.to_op_sharding_sharding(x.sharding, x.ndim)  # type: ignore
                    if x._committed else None)
    else:
      return aval, x._device
  except:
    return aval, None


def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  compiled_fun = xla_primitive_callable(prim, *unsafe_map(arg_spec, args),
                                        **params)
  return compiled_fun(*args)

# TODO(phawkins,frostig,mattjj): update code referring to
# xla.apply_primitive to point here, or use simple_impl if that's why
# it is using apply_primitive to begin with
xla.apply_primitive = apply_primitive

def simple_impl(prim):
  prim.def_impl(partial(apply_primitive, prim))

RuntimeToken = Any

class RuntimeTokenSet(threading.local):
  tokens: Dict[core.Effect, Tuple[RuntimeToken, Device]]
  output_tokens: Dict[Device, RuntimeToken]
  output_runtime_tokens: Dict[Device, RuntimeToken]

  def __init__(self):
    self.tokens = {}
    # TODO(sharadmv): remove redundant output token dictionary when minimum
    # jaxlib version is bumped to 0.3.16.
    self.output_tokens = {}
    self.output_runtime_tokens = {}

  def get_token(self, eff: core.Effect, device: Device) -> RuntimeToken:
    if eff not in self.tokens:
      self.tokens[eff] = device_put(np.zeros(0, np.bool_), device), device
    elif self.tokens[eff][1] != device:
      (old_token,), _ = self.tokens[eff]
      old_token.aval = core.ShapedArray((0,), np.bool_)
      self.tokens[eff] = device_put(old_token, device), device
    return self.tokens[eff][0]

  def update_token(self, eff: core.Effect, token: RuntimeToken):
    self.tokens[eff] = token, self.tokens[eff][1]

  def set_output_token(self, device: Device, token: RuntimeToken):
    # We're free to clobber the previous output token because on each
    # device we have a total ordering of computations. Only the token
    # from the latest computation matters. If this weren't the case
    # we'd need to store a set of output tokens.
    self.output_tokens[device] = token

  def set_output_runtime_token(self, device: Device, token: RuntimeToken):
    # TODO(sharadmv): remove this method when minimum jaxlib version is bumped
    self.output_runtime_tokens[device] = token

  def clear(self):
    self.tokens = {}
    self.output_tokens = {}
    self.output_runtime_tokens = {}

  def block_until_ready(self):
    for token, _ in self.tokens.values():
      token[0].block_until_ready()
    for token in self.output_tokens.values():
      token[0].block_until_ready()
    for token in self.output_runtime_tokens.values():
      token.block_until_ready()
    self.clear()

runtime_tokens: RuntimeTokenSet = RuntimeTokenSet()

@atexit.register
def wait_for_tokens():
  runtime_tokens.block_until_ready()

@util.cache()
def xla_primitive_callable(prim, *arg_specs: ArgSpec, **params):
  _, arg_devices = util.unzip2(arg_specs)
  donated_invars = (False,) * len(arg_specs)
  if config.jax_array:
    # This will be resolved in sharded_lowering.
    device = None
  else:
    device = _device_from_arg_devices(arg_devices)
  def prim_fun(*args):
    out = prim.bind(*args, **params)
    if prim.multiple_results:
      return out
    else:
      return out,
  compiled = _xla_callable_uncached(lu.wrap_init(prim_fun), device, None,
                                    prim.name, donated_invars, False, *arg_specs)
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
                   donated_invars, inline, keep_unused: bool):
  del inline  # Only used at tracing time
  if fun.in_type is None:
    arg_specs = unsafe_map(arg_spec, args)
  else:
    # fun.in_type is used for dynamic shapes.
    if config.jax_array:
      raise NotImplementedError('Dynamic shapes do not work with Array.')
    arg_specs = [(None, getattr(x, '_device', None)) for x in args]
  compiled_fun = xla_callable(fun, device, backend, name, donated_invars,
                              keep_unused, *arg_specs)
  try:
    return compiled_fun(*args)
  except FloatingPointError:
    assert config.jax_debug_nans or config.jax_debug_infs  # compiled_fun can only raise in this case
    print("Invalid value encountered in the output of a jit-decorated function. "
          "Calling the de-optimized version.")
    # We want to run the wrapped function again (after xla_callable already ran
    # it), but linear_util.WrappedFun instances are meant to be run only once.
    # In addition to re-executing the Python code, which is usually undesirable
    # but which config.jax_debug_nans is meant to opt into, we'll be
    # re-executing any linear_util.py-style side effects, i.e. re-populating
    # Stores created by any transformation_with_aux's applied to fun. Since this
    # is intentional here, to avoid "Store occupied" errors we clone the
    # WrappedFun with empty stores.
    stores = [lu.Store() for _ in fun.stores]
    clone = lu.WrappedFun(fun.f, fun.transforms, stores, fun.params,
                          fun.in_type)

    with core.new_sublevel():
      _ = clone.call_wrapped(*args)  # may raise, not return

    # If control reaches this line, we got a NaN on the output of `compiled_fun`
    # but not `clone.call_wrapped` on the same arguments. Let's tell the user.
    fun_info = pe.fun_sourceinfo(fun.f)
    msg = ("An invalid value was encountered in the output of the "
           f"`jit`-decorated function {fun_info}. Because "
           "config.jax_debug_nans and/or config.jax_debug_infs is set, the "
           "de-optimized function (i.e., the function as if the `jit` "
           "decorator were removed) was called in an attempt to get a more "
           "precise error message. However, the de-optimized function did not "
           "produce invalid values during its execution. This behavior can "
           "result from `jit` optimizations causing the invalud value to be "
           "produced. It may also arise from having nan/inf constants as "
           "outputs, like `jax.jit(lambda ...: jax.numpy.nan)(...)`. "
           "\n\n"
           "It may be possible to avoid the invalid value by removing the "
           "`jit` decorator, at the cost of losing optimizations. "
           "\n\n"
           "If you see this error, consider opening a bug report at "
           "https://github.com/google/jax.")
    raise FloatingPointError(msg)

xla.xla_call_p.def_impl(_xla_call_impl)


# TODO(yashkatariya,mattjj): Try to handle this in api.py via a device_put and
# don't pass the device and backend argument to `_xla_callable_uncached`.
def not_none_device_or_backend_on_jit(backend, device, num_ins):
  """This is to support the backend and device argument on jit. It's a feature
  that's deprecated but needs to be supported for feature parity and so that we
  can delete the non-Array paths when Array is switched on.
  """
  # TODO(yashkatariya): Remove this entire function when backend and device are
  # removed as arguments on jit.

  from jax._src import sharding

  if device is not None and backend is not None:
    raise ValueError("can't specify both a device and a backend for jit, "
                     "got device={} and backend={}".format(device, backend))

  if backend is not None:
    da = [xb.get_backend(backend).get_default_device_assignment(1)[0]]
  else:
    assert device is not None
    da = [device]

  assert len(da) == 1
  # Set committed to True for this path because it simulates a device_put on
  # behalf of a user.
  committed = True
  # in_shardings will be marked as replicated regardless of whatever the input
  # had. Given that only a single device is allowed above, this is correct.
  in_shardings = [sharding.OpShardingSharding.get_replicated(da)] * num_ins
  return committed, da, in_shardings


def sharded_lowering(fun, device, backend, name, donated_invars, always_lower,
                     keep_unused, *arg_specs):
  # TODO(yashkatariya): Remove the local imports from here when the functions
  # in pxla.py move to dispatch.py or a utils file.
  from jax._src import sharding
  from jax.interpreters import pxla
  from jax.experimental import pjit

  in_avals, in_shardings = util.unzip2(arg_specs)

  if backend is not None or device is not None:
    committed, da, in_shardings = not_none_device_or_backend_on_jit(
        backend, device, len(in_shardings))
  else:
    committed = any(i is not None for i in in_shardings)
    da = pjit._get_and_check_device_assignment(
        (i for i in in_shardings if i is not None), pxla.EMPTY_ENV.physical_mesh)
    in_shardings = [sharding.OpShardingSharding.get_replicated(da) if i is None else i
                    for i in in_shardings]

  process_index = xb.process_index()
  local_da = [d for d in da if d.process_index == process_index]
  if len(local_da) != len(da):
    warnings.warn(
        "Running operations on `Array`s that are not fully addressable by this "
        "process (i.e. `Array`s with data sharded across multiple devices and "
        "processes.) is dangerous. It’s very important that all processes run "
        "the same cross-process computations in the same order otherwise it "
        "can lead to hangs.\n"
        "If you’re not already familiar with JAX’s multi-process "
        "programming model, please read "
        "https://jax.readthedocs.io/en/latest/multi_process.html.")

  if not in_shardings:
    inp_device_assignment = da
  else:
    inp_device_assignment = None

  # Pass in a singleton `_UNSPECIFIED` for out_shardings because we don't know
  # the number of output avals at this stage. lower_sharding_computation will
  # apply it to all out_avals.
  return pxla.lower_sharding_computation(
      fun, 'jit', name, in_shardings, pjit._UNSPECIFIED,
      donated_invars, in_avals,
      in_is_global=(True,) * len(arg_specs), keep_unused=keep_unused,
      committed=committed, always_lower=always_lower,
      inp_device_assignment=inp_device_assignment)


def _xla_callable_uncached(fun: lu.WrappedFun, device, backend, name,
                           donated_invars, keep_unused, *arg_specs):
  if config.jax_array:
    computation = sharded_lowering(fun, device, backend, name, donated_invars,
                                   False, keep_unused, *arg_specs)
    return computation.compile(_allow_propagation_to_outputs=True).unsafe_call
  else:
    return lower_xla_callable(fun, device, backend, name, donated_invars, False,
                              keep_unused, *arg_specs).compile().unsafe_call

xla_callable = lu.cache(_xla_callable_uncached)


def is_single_device_sharding(sharding) -> bool:
  from jax._src.sharding import PmapSharding
  # Special case PmapSharding here because PmapSharding maps away an axis
  # and needs to be handled separately.
  return len(sharding.device_set) == 1 and not isinstance(sharding, PmapSharding)


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


def should_tuple_args(num_args: int, platform: str):
  # CPU does not need a tuple as it uses a buffer table
  # TPU only needs a tuple for very long lists
  if platform == "cpu":
    return False
  elif platform == "tpu":
    return num_args > 2000
  else:
    return num_args > 100


def raise_warnings_or_errors_for_jit_of_pmap(nreps, backend, name, jaxpr):
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

  if xb.process_count() > 1 and (nreps > 1 or
                                 jaxpr_has_primitive(jaxpr, "xla_pmap")):
    raise NotImplementedError(
        "jit of multi-host pmap not implemented (and jit-of-pmap can cause "
        "extra data movement anyway, so maybe you don't want it after all).")


@profiler.annotate_function
def lower_xla_callable(
    fun: lu.WrappedFun, device, backend, name, donated_invars,
    always_lower: bool, keep_unused: bool, *arg_specs):
  """Lower into XLA.

  Args:
    always_lower: If `True`, even trivial programs (not doing any computation
      such as lambda x: x) will be lowered into an XLA program.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.
  """
  if device is not None and backend is not None:
    raise ValueError("can't specify both a device and a backend for jit, "
                     "got device={} and backend={}".format(device, backend))
  abstract_args, arg_devices = util.unzip2(arg_specs)
  if fun.in_type is None:
    # Add an annotation inferred from the arguments; no dynamic axes here.
    in_type = tuple(unsafe_zip(abstract_args, itertools.repeat(True)))
    fun = lu.annotate(fun, in_type)
  else:
    assert abstract_args == (None,) * len(abstract_args)
    abstract_args = [aval for aval, _ in fun.in_type]

  with log_elapsed_time(f"Finished tracing + transforming {fun.__name__} "
                        "for jit in {elapsed_time} sec"):
    jaxpr, out_type, consts = pe.trace_to_jaxpr_final2(
        fun, pe.debug_info_final(fun, "jit"))
  out_avals, kept_outputs = util.unzip2(out_type)

  if any(isinstance(c, core.Tracer) for c in consts):
    raise UnexpectedTracerError("Encountered an unexpected tracer.")

  if config.jax_dynamic_shapes:
    keep_unused = True
    has_outfeed = False
    donated_invars = [False] * len(fun.in_type)
  else:
    has_outfeed = core.jaxpr_uses_outfeed(jaxpr)
    jaxpr = apply_outfeed_rewriter(jaxpr)

  jaxpr, kept_const_idx, kept_var_idx = _dce_helper(jaxpr, keep_unused)
  consts = [c for i, c in enumerate(consts) if i in kept_const_idx]
  abstract_args, arg_devices = util.unzip2(
      [a for i, a in enumerate(arg_specs) if i in kept_var_idx])
  donated_invars = [x for i, x in enumerate(donated_invars)
                    if i in kept_var_idx]
  del kept_const_idx

  nreps = jaxpr_replicas(jaxpr)
  device = _xla_callable_device(nreps, backend, device, arg_devices)
  backend = xb.get_device_backend(device) if device else xb.get_backend(backend)

  if config.jax_dynamic_shapes and jaxpr_has_bints(jaxpr):
    jaxpr, consts = pe.pad_jaxpr(jaxpr, consts)

  map(prefetch, itertools.chain(consts, jaxpr_literals(jaxpr)))

  # Computations that only produce constants and/or only rearrange their inputs,
  # which are often produced from partial evaluation, don't need compilation,
  # and don't need to evaluate their arguments.
  if (not always_lower and not (jaxpr.effects or has_outfeed) and
      (not jaxpr.eqns and all(kept_outputs) or not jaxpr.outvars)):
    return XlaComputation(
        name, None, True, None, None, None, jaxpr=jaxpr, consts=consts,
        device=device, in_avals=abstract_args, out_avals=out_avals,
        has_unordered_effects=False, ordered_effects=[],
        kept_var_idx=kept_var_idx, keepalive=None, host_callbacks=[])

  if not _on_exit:
    log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
    if len(abstract_args) > 10:
      msg = f"Compiling {fun.__name__} ({id(fun)}) for {len(abstract_args)} args."
    else:
      msg = f"Compiling {fun.__name__} ({id(fun)} for args {abstract_args}."
    logging.log(log_priority, msg)

  raise_warnings_or_errors_for_jit_of_pmap(nreps, backend, name, jaxpr)

  # pass long arg lists as tuple for TPU
  tuple_args = should_tuple_args(len(abstract_args), backend.platform)
  axis_env = xla.AxisEnv(nreps, (), ())
  name_stack = util.new_name_stack(util.wrap_name(name, 'jit'))
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  module_name = f"jit_{fun.__name__}"
  unordered_effects = [eff for eff in closed_jaxpr.effects
                       if eff not in core.ordered_effects]
  ordered_effects = [eff for eff in closed_jaxpr.effects
                     if eff in core.ordered_effects]
  lowering_result = mlir.lower_jaxpr_to_module(
      module_name, closed_jaxpr, unordered_effects,
      ordered_effects, backend, backend.platform,
      mlir.ReplicaAxisContext(axis_env), name_stack, donated_invars)
  module, keepalive, host_callbacks = (
      lowering_result.module, lowering_result.keepalive,
      lowering_result.host_callbacks)
  return XlaComputation(
      name, module, False, donated_invars, fun.in_type, out_type, nreps=nreps,
      device=device, backend=backend, tuple_args=tuple_args,
      in_avals=abstract_args, out_avals=out_avals,
      has_unordered_effects=bool(unordered_effects),
      ordered_effects=ordered_effects, kept_var_idx=kept_var_idx,
      keepalive=keepalive, host_callbacks=host_callbacks)


def _backend_supports_unbounded_dynamic_shapes(backend: Backend) -> bool:
  return backend.platform == 'iree'


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


def jaxpr_has_primitive(jaxpr, prim_name: str):
  """Whether there is a primitive given by user anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if prim_name in eqn.primitive.name:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_primitive(subjaxpr, prim_name):
      return True
  return False

def jaxpr_has_bints(jaxpr: core.Jaxpr) -> bool:
  return (any(type(v.aval) is core.AbstractBInt for v in jaxpr.invars) or
          any(type(v.aval) is core.AbstractBInt
              for j in itertools.chain([jaxpr], core.subjaxprs(jaxpr))
              for e in j.eqns for v in e.outvars))

def _dce_helper(jaxpr: core.Jaxpr, keep_unused: bool,
                ) -> Tuple[core.Jaxpr, Set[int], Set[int]]:
  used_outputs = [True] * len(jaxpr.outvars)
  new_jaxpr, used_consts, used_inputs = pe.dce_jaxpr_consts(
      jaxpr, used_outputs, instantiate=keep_unused)
  kept_const_idx = {i for i, b in enumerate(used_consts) if b}
  kept_var_idx = {i for i, b in enumerate(used_inputs) if b}
  return new_jaxpr, kept_const_idx, kept_var_idx


# We can optionally set a Jaxpr rewriter that can be applied just before
# compilation. This mechanism is used for compiling id_tap, we can
# remove it once we bring the id_tap implementation into the core.
outfeed_rewriter: Optional[Callable[[core.Jaxpr], core.Jaxpr]] = None
def apply_outfeed_rewriter(jaxpr: core.Jaxpr) -> core.Jaxpr:
  if outfeed_rewriter is not None:
    return outfeed_rewriter(jaxpr)
  else:
    return jaxpr


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


def _xla_callable_device(nreps, backend, device,
                         arg_devices) -> Optional[Device]:
  if nreps > 1:
    if device is not None or backend is not None:
      raise ValueError(f"can't specify device or backend for jit-of-pmap, "
                       f"got device={device} and backend={backend}")
    return None
  else:
    # TODO(skye): dedup with C++ jit logic for determining jit device?
    if device is not None:
      assert backend is None
      return device

    if backend is not None:
      return xb.get_backend(backend).get_default_device_assignment(1)[0]

    arg_device = _device_from_arg_devices(arg_devices)
    if arg_device is not None:
      return arg_device

    return config.jax_default_device


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

num_buffers_handlers[core.AbstractToken] = lambda _: 1
num_buffers_handlers[core.ShapedArray] = lambda _: 1
num_buffers_handlers[core.DShapedArray] = lambda _: 1
num_buffers_handlers[core.ConcreteArray] = lambda _: 1
num_buffers_handlers[core.AbstractBInt] = lambda _: 1


def _input_handler(backend: Backend,
                   in_type: Optional[pe.InputType],
                   out_type: Optional[pe.OutputType],
                   ) -> Optional[Callable]:
  if in_type is None:
    assert out_type is None
    return None
  in_avals, which_explicit = util.unzip2(in_type)
  # Check whether we actually need an input_handler.
  needs_implicit = which_explicit and not all(which_explicit)
  needs_out_handling = any(type(d) is core.InDBIdx for a, _ in out_type or []
                           if type(a) is core.DShapedArray for d in a.shape)

  if not needs_implicit and not needs_out_handling:
    return None
  assert config.jax_dynamic_shapes

  # Precompute how to grab implicit inputs from explicit inputs' axis sizes.
  which_explicit = which_explicit or [True] * len(in_avals)
  implicit_idxs = {i for i, ex in enumerate(which_explicit) if not ex}
  implicit_args_from_axes: List[Tuple[int, int, int]] = []
  for arg_idx, aval in enumerate(in_avals):
    if isinstance(aval, core.DShapedArray):
      for axis_idx, d in enumerate(aval.shape):
        if isinstance(d, core.DBIdx) and d.val in implicit_idxs:
          implicit_args_from_axes.append((d.val, arg_idx, axis_idx))
  assert {i for i, _, _ in implicit_args_from_axes} == implicit_idxs

  # Precompute which input values are needed for output types.
  inputs_needed_for_out_types = out_type and [
      d.val for aval, _ in out_type if type(aval) is core.DShapedArray  # type: ignore
      for d in aval.shape if type(d) is core.InDBIdx]

  def elaborate(explicit_args: Sequence[Any]) -> Tuple[Tuple, Optional[Tuple]]:
    if needs_implicit:
      # Build full argument list, leaving Nones for implicit arguments.
      explicit_args_ = iter(explicit_args)
      args = [next(explicit_args_) if ex else None for ex in which_explicit]
      assert next(explicit_args_, None) is None
      # Populate implicit arguments.
      for i, j, k in implicit_args_from_axes:
        if args[i] is None:
          args[i] = args[j].shape[k]  # type: ignore
        else:
          if args[i] != args[j].shape[k]:
            raise Exception("inconsistent argument axis sizes for type")
    else:
      args = list(explicit_args)

    if needs_out_handling:
      # Make a list of inputs needed by output types, leaving unneeded as None.
      out_type_env = [None] * len(args)
      for i in inputs_needed_for_out_types or []:
        out_type_env[i] = args[i]
    else:
      out_type_env = None  # type: ignore

    return tuple(args), out_type_env and tuple(out_type_env)  # type: ignore
  return elaborate

def _result_handler(backend: Backend,
                    sticky_device: Optional[Device],
                    out_type: Optional[pe.OutputType]
                    ) -> Callable:
  out_avals, kept_outputs = util.unzip2(out_type)
  handlers = map(partial(aval_to_result_handler, sticky_device), out_avals)
  dyn_outs = any(type(aval) is core.DShapedArray and
                 any(type(d) in (core.InDBIdx, core.OutDBIdx) for d in aval.shape)
                 for aval in out_avals)
  if not dyn_outs:
    return SimpleResultHandler(handlers)
  assert config.jax_dynamic_shapes

  def result_handler(input_env, lists_of_bufs):
    results = []
    for handler, bufs in unsafe_zip(handlers, lists_of_bufs):
      results.append(handler((input_env, results), *bufs))
    return [r for r, keep in unsafe_zip(results, kept_outputs) if keep]
  return result_handler

class SimpleResultHandler:
  handlers: Sequence[ResultHandler]
  def __init__(self, handlers): self.handlers = handlers
  def __iter__(self): return iter(self.handlers)
  def __len__(self): return len(self.handlers)
  def __call__(self, env, lists_of_bufs):
    return tuple(h(env, *bs) for h, bs in zip(self.handlers, lists_of_bufs))


def maybe_create_array_from_da(buf, aval, device):
  if config.jax_array:
    from jax._src.array import ArrayImpl
    from jax._src.sharding import SingleDeviceSharding
    return ArrayImpl(aval, SingleDeviceSharding(buf.device()), [buf],
                     committed=(device is not None), _skip_checks=True)
  else:
    return device_array.make_device_array(aval, device, buf)


if MYPY:
  ResultHandler = Any
else:
  class ResultHandler(Protocol):
    def __call__(self, env: Optional[Sequence[Any]], *args: xla.Buffer) -> Any:
      """Boxes raw buffers into their user-facing representation."""

def aval_to_result_handler(sticky_device: Optional[Device],
                           aval: core.AbstractValue) -> ResultHandler:
  try:
    return result_handlers[type(aval)](sticky_device, aval)
  except KeyError as err:
    raise TypeError(f"No result handler for type: {type(aval)}") from err

def array_result_handler(sticky_device: Optional[Device],
                         aval: core.ShapedArray):
  if aval.dtype == dtypes.float0:
    return lambda _, __: np.zeros(aval.shape, dtypes.float0)
  aval = core.raise_to_shaped(aval)
  if core.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.result_handler(sticky_device, aval)
  handler = lambda _, b: maybe_create_array_from_da(b, aval, sticky_device)
  handler.args = aval, sticky_device  # for C++ dispatch path in api.py
  return handler

def dynamic_array_result_handler(sticky_device: Optional[Device],
                                 aval: core.DShapedArray):
  if aval.dtype == dtypes.float0:
    return lambda _: np.zeros(aval.shape, dtypes.float0)  # type: ignore
  else:
    return partial(_dynamic_array_result_handler, sticky_device, aval)

def _dynamic_array_result_handler(sticky_device, aval, env, buf):
  in_env, out_env = env or (None, None)
  shape = [in_env[d.val] if type(d) is core.InDBIdx else
           out_env[d.val] if type(d) is core.OutDBIdx else d
           for d in aval.shape]
  if all(type(d) is int for d in shape):
    aval = core.ShapedArray(tuple(shape), aval.dtype)
    return maybe_create_array_from_da(buf, aval, sticky_device)
  elif any(type(d) is core.BInt for d in shape):
    padded_shape = [d.bound if type(d) is core.BInt else d for d in shape]
    buf_aval = core.ShapedArray(tuple(padded_shape), aval.dtype, aval.weak_type)
    data = maybe_create_array_from_da(buf, buf_aval, sticky_device)
    return core.PaddedArray(aval.update(shape=tuple(shape)), data)
  else:
    aval = core.ShapedArray(tuple(shape), aval.dtype)
    return maybe_create_array_from_da(buf, aval, sticky_device)



result_handlers: Dict[
    Type[core.AbstractValue],
    Callable[[Optional[Device], Any], ResultHandler]] = {}
result_handlers[core.AbstractToken] = lambda _, __: lambda _, __: core.token
result_handlers[core.ShapedArray] = array_result_handler
result_handlers[core.DShapedArray] = dynamic_array_result_handler
result_handlers[core.ConcreteArray] = array_result_handler
result_handlers[core.AbstractBInt] = \
    lambda _, a: lambda _, b: core.BInt(int(b), a.bound)


def needs_check_special():
  return config.jax_debug_infs or config.jax_debug_nans

def check_special(name, bufs):
  if needs_check_special():
    for buf in bufs:
      _check_special(name, buf.dtype, buf)

def _check_special(name, dtype, buf):
  if dtypes.issubdtype(dtype, np.inexact):
    if config.jax_debug_nans and np.any(np.isnan(np.asarray(buf))):
      raise FloatingPointError(f"invalid value (nan) encountered in {name}")
    if config.jax_debug_infs and np.any(np.isinf(np.asarray(buf))):
      raise FloatingPointError(f"invalid value (inf) encountered in {name}")

def _add_tokens(has_unordered_effects: bool, ordered_effects: List[core.Effect],
                has_host_callbacks: bool, device: Device, input_bufs):
  tokens = [runtime_tokens.get_token(eff, device) for eff in ordered_effects]
  tokens_flat = flatten(tokens)
  input_bufs = [*tokens_flat, *input_bufs]
  def _remove_tokens(output_bufs, runtime_token):
    # TODO(sharadmv): simplify when minimum jaxlib version is bumped
    num_output_tokens = len(ordered_effects) + (not can_execute_with_token and
        has_unordered_effects)
    token_bufs, output_bufs = util.split_list(output_bufs, [num_output_tokens])
    if has_unordered_effects or has_host_callbacks:
      if can_execute_with_token:
        runtime_tokens.set_output_runtime_token(device, runtime_token)
      else:
        output_token_buf, *token_bufs = token_bufs
        runtime_tokens.set_output_token(device, output_token_buf)
    for eff, token_buf in zip(ordered_effects, token_bufs):
      runtime_tokens.update_token(eff, token_buf)
    return output_bufs
  return input_bufs, _remove_tokens


def _execute_compiled(name: str, compiled: XlaLoadedExecutable,
                      input_handler: Optional[Callable],
                      output_buffer_counts: Sequence[int],
                      result_handler: Callable, has_unordered_effects: bool,
                      ordered_effects: List[core.Effect], kept_var_idx,
                      has_host_callbacks: bool, *args):
  device, = compiled.local_devices()
  args, env = input_handler(args) if input_handler else (args, None)
  in_flat = flatten(device_put(x, device) for i, x in enumerate(args)
                    if i in kept_var_idx)
  if has_unordered_effects or ordered_effects or has_host_callbacks:
    in_flat, token_handler = _add_tokens(
        has_unordered_effects, ordered_effects, has_host_callbacks, device,
        in_flat)
    if can_execute_with_token:
      out_flat, runtime_token = compiled.execute_with_token(in_flat)
    else:
      out_flat = compiled.execute(in_flat)
      runtime_token = None
  else:
    out_flat = compiled.execute(in_flat)
  check_special(name, out_flat)
  out_bufs = unflatten(out_flat, output_buffer_counts)
  if ordered_effects or has_unordered_effects or has_host_callbacks:
    out_bufs = token_handler(out_bufs, runtime_token)
  return result_handler(env, out_bufs)


def _execute_replicated(name: str,
                        compiled: XlaLoadedExecutable,
                        input_handler: Optional[Callable],
                        output_buffer_counts: Sequence[int],
                        result_handler: Callable,
                        has_unordered_effects: bool,
                        ordered_effects: List[core.Effect],
                        kept_var_idx,
                        has_host_callbacks: bool,
                        *args,
                        from_lower_sharding_computation: bool = False):
  if has_unordered_effects or ordered_effects:
    # TODO(sharadmv): support jit-of-pmap with effects
    raise NotImplementedError(
        "Cannot execute replicated computation with effects.")
  if input_handler: raise NotImplementedError  # TODO(mattjj, dougalm)
  input_bufs = [flatten(device_put(x, device) for i, x in enumerate(args)
                        if i in kept_var_idx)
                for device in compiled.local_devices()]
  input_bufs_flip = list(unsafe_zip(*input_bufs))
  out_bufs_flat_rep = compiled.execute_sharded_on_local_devices(input_bufs_flip)
  out_flat = [bufs[0] for bufs in out_bufs_flat_rep]
  check_special(name, out_flat)
  out_bufs = unflatten(out_flat, output_buffer_counts)
  if from_lower_sharding_computation:
    return result_handler(out_bufs)
  return result_handler(None, out_bufs)


def _execute_trivial(jaxpr, device: Optional[Device], consts, avals, handlers,
                     has_unordered_effects: bool,
                     ordered_effects: List[core.Effect], kept_var_idx,
                     host_callbacks, *args):
  env: Dict[core.Var, Any]  = {}
  pruned_args = (x for i, x in enumerate(args) if i in kept_var_idx)
  map(env.setdefault, jaxpr.invars, pruned_args)
  map(env.setdefault, jaxpr.constvars, consts)
  outs = [xla.canonicalize_dtype(v.val) if type(v) is core.Literal else env[v]
          for v in jaxpr.outvars]
  return [_copy_device_array_to_device(x, device) if device_array.type_is_device_array(x)
          else h(None, *device_put(x, device)) for h, x in zip(handlers, outs)]


class XlaComputation(stages.XlaLowering):
  name: str
  _is_trivial: bool
  _executable: Optional[XlaCompiledComputation]
  _donated_invars: Optional[Sequence[bool]]

  def __init__(self, name: str, hlo, is_trivial: bool,
               donated_invars: Optional[Sequence[bool]],
               in_type: Optional[pe.InputType],
               out_type: Optional[pe.OutputType],
               **compile_args):
    self.name = name
    self._hlo = hlo
    self._is_trivial = is_trivial
    self._donated_invars = donated_invars
    self._in_type = in_type
    self._out_type = out_type
    self._executable = None
    self.compile_args = compile_args

  def is_trivial(self):
    return self._is_trivial

  # -- stages.XlaLowering overrides

  def hlo(self) -> xc.XlaComputation:
    if self.is_trivial():
      raise ValueError("A trivial computation has no HLO")
    if isinstance(self._hlo, xc.XlaComputation):
      return self._hlo
    return xe.mlir.mlir_module_to_xla_computation(
        mlir.module_to_string(self._hlo),
        use_tuple_args=self.compile_args["tuple_args"])

  def mhlo(self) -> ir.Module:
    if self.is_trivial():
      raise ValueError("A trivial computation has no MHLO")
    if isinstance(self._hlo, xc.XlaComputation):
      module_str = xe.mlir.xla_computation_to_mlir_module(self._hlo)
      with mlir.make_ir_context():
        return ir.Module.parse(module_str)
    return self._hlo

  def compile(self) -> XlaCompiledComputation:
    if self._executable is None:
      if self.is_trivial():
        self._executable = XlaCompiledComputation.from_trivial_jaxpr(
            **self.compile_args)
      else:
        self._executable = XlaCompiledComputation.from_xla_computation(
            self.name, self._hlo, self._in_type, self._out_type,
            **self.compile_args)

    return self._executable

@profiler.annotate_function
def backend_compile(backend, built_c, options, host_callbacks):
  # we use a separate function call to ensure that XLA compilation appears
  # separately in Python profiling results
  if host_callbacks:
    return backend.compile(built_c, compile_options=options,
                           host_callbacks=host_callbacks)
  # Some backends don't have `host_callbacks` option yet
  # TODO(sharadmv): remove this fallback when all backends allow `compile`
  # to take in `host_callbacks`
  return backend.compile(built_c, compile_options=options)

# TODO(phawkins): update users.
xla.backend_compile = backend_compile

_ir_dump_counter = itertools.count()

def _make_string_safe_for_filename(s: str) -> str:
  return re.sub(r'[^\w.)( -]', '', s)

def _dump_ir_to_file(name: str, ir: str):
  id = next(_ir_dump_counter)
  name = f"jax_ir{id}_{_make_string_safe_for_filename(name)}.mlir"
  name = epath.Path(FLAGS.jax_dump_ir_to) / name
  name.write_text(ir)


def compile_or_get_cached(backend, computation: ir.Module, compile_options,
                          host_callbacks):
  # Avoid import cycle between jax and jax.experimental
  from jax.experimental.compilation_cache import compilation_cache as cc

  sym_name = computation.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value

  if FLAGS.jax_dump_ir_to:
    _dump_ir_to_file(module_name, mlir.module_to_string(computation))

  # Convert ir.Module to a string representation, unless the
  # back-end expliclity flags the ability to handle a module directly
  # (avoiding the overhead of back and forth conversions)
  serialized_computation: Union[str, bytes, ir.Module]
  if getattr(backend, "needs_str_ir", True):
    if xc.mlir_api_version >= 34:
      serialized_computation = mlir.module_to_bytecode(computation)
    else:
      serialized_computation = mlir.module_to_string(computation)
  else:
    serialized_computation = computation

  # Persistent compilation cache only implemented on TPU.
  # TODO(skye): add warning when initializing cache on unsupported default platform
  supported_platforms = ["tpu"]
  # GPU caching can be enabled if JitRt is enabled.
  # TODO(b/232263664): Remove check when JitRt is enabled by default.
  if "--xla_gpu_enable_xla_runtime_executable=true" in os.environ.get("XLA_FLAGS", ""):
    supported_platforms.append("gpu")
  if cc.is_initialized() and backend.platform in supported_platforms:
    cached_executable = _cache_read(serialized_computation, module_name,
                                    compile_options, backend)
    if cached_executable is not None:
      logging.info("Persistent compilation cache hit for '%s'", module_name)
      return cached_executable
    else:
      compiled = backend_compile(backend, serialized_computation,
                                 compile_options, host_callbacks)
      _cache_write(serialized_computation, module_name,  compile_options,
                   backend, compiled)
      return compiled

  return backend_compile(backend, serialized_computation, compile_options,
                         host_callbacks)


def _cache_read(computation: Union[str, bytes, ir.Module], module_name: str,
                compile_options: CompileOptions,
                backend: Backend) -> Optional[XlaLoadedExecutable]:
  """Looks up `computation` in the persisent compilation cache."""
  # Avoid import cycle between jax and jax.experimental
  from jax.experimental.compilation_cache import compilation_cache as cc

  try:
    return cc.get_executable(computation, compile_options, backend)
  except Exception as ex:
    if config.jax_raise_persistent_cache_errors:
      raise
    warnings.warn(
        f"Error reading persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")
    return None


def _cache_write(computation: Union[str, bytes, ir.Module], module_name: str,
                 compile_options: CompileOptions, backend: Backend,
                 compiled: XlaLoadedExecutable):
  """Writes `computation` to the persistent compilation cache."""
  # Avoid import cycle between jax and jax.experimental
  from jax.experimental.compilation_cache import compilation_cache as cc

  try:
    cc.put_executable(module_name, computation, compile_options, compiled,
                      backend)
  except Exception as ex:
    if config.jax_raise_persistent_cache_errors:
      raise
    warnings.warn(
        f"Error writing persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")

def get_buffer_counts(out_avals, ordered_effects, has_unordered_effects):
  buffer_counts = [aval_to_num_buffers(aval) for aval in out_avals]
  if ordered_effects or has_unordered_effects:
    num_output_tokens = len(ordered_effects)
    # TODO(sharadmv): remove check when minimum jaxlib version is bumped
    if not can_execute_with_token:
      num_output_tokens += has_unordered_effects
    buffer_counts = ([1] * num_output_tokens) + buffer_counts
  return buffer_counts


class XlaCompiledComputation(stages.XlaExecutable):
  def __init__(self, xla_executable, in_avals, kept_var_idx, unsafe_call,
               keepalive: Any):
    self._xla_executable = xla_executable
    self.in_avals = in_avals
    self._kept_var_idx = kept_var_idx
    self.unsafe_call = unsafe_call
    # Only the `unsafe_call` function is cached, so to avoid the `keepalive`
    # being garbage collected we attach it to `unsafe_call`.
    self.unsafe_call.keepalive = keepalive

  @staticmethod
  def from_xla_computation(name: str, xla_computation: Optional[ir.Module],
                           in_type: Optional[pe.InputType],
                           out_type: Optional[pe.OutputType], nreps: int,
                           device: Optional[Device], backend: Backend,
                           tuple_args: bool,
                           in_avals: Sequence[core.AbstractValue],
                           out_avals: Sequence[core.AbstractValue],
                           has_unordered_effects: bool,
                           ordered_effects: List[core.Effect],
                           kept_var_idx: Set[int], keepalive: Optional[Any],
                           host_callbacks: List[Any]) -> XlaCompiledComputation:
    sticky_device = device
    input_handler = _input_handler(backend, in_type, out_type)
    result_handler = _result_handler(backend, sticky_device, out_type)
    options = xb.get_compile_options(
        num_replicas=nreps, num_partitions=1,
        device_assignment=(sticky_device,) if sticky_device else None)
    options.parameter_is_tupled_arguments = tuple_args
    with log_elapsed_time(f"Finished XLA compilation of {name} "
                          "in {elapsed_time} sec"):
      compiled = compile_or_get_cached(backend, xla_computation, options,
                                       host_callbacks)
    buffer_counts = get_buffer_counts(out_avals, ordered_effects,
                                      has_unordered_effects)
    execute = _execute_compiled if nreps == 1 else _execute_replicated
    unsafe_call = partial(execute, name, compiled, input_handler, buffer_counts,  # type: ignore  # noqa: F811
                          result_handler, has_unordered_effects,
                          ordered_effects, kept_var_idx, bool(host_callbacks))
    return XlaCompiledComputation(compiled, in_avals, kept_var_idx, unsafe_call,
                                  keepalive)

  def is_trivial(self):
    return self._xla_executable == None

  @property
  def xla_executable(self):
    # TODO(frostig): remove in favor of runtime_executable?
    if self.is_trivial():
      raise ValueError("A trivial compiled computation has no XLA executable")
    return self._xla_executable

  @staticmethod
  def from_trivial_jaxpr(jaxpr, consts, device, in_avals, out_avals,
                         has_unordered_effects, ordered_effects, kept_var_idx,
                         keepalive: Optional[Any],
                         host_callbacks: List[Any]) -> XlaCompiledComputation:
    assert keepalive is None
    result_handlers = map(partial(aval_to_result_handler, device), out_avals)
    unsafe_call = partial(_execute_trivial, jaxpr, device, consts, out_avals,
                          result_handlers, has_unordered_effects,
                          ordered_effects, kept_var_idx, bool(host_callbacks))
    return XlaCompiledComputation(None, in_avals, kept_var_idx, unsafe_call,
                                  keepalive)

  # -- stages.XlaExecutable overrides

  def xla_extension_executable(self):
    return self.xla_executable

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


def device_put(x, device: Optional[Device] = None) -> Tuple[Any, ...]:
  x = xla.canonicalize_dtype(x)
  try:
    return device_put_handlers[type(x)](x, device)
  except KeyError as err:
    raise TypeError(f"No device_put handler for type: {type(x)}") from err

# TODO(phawkins): update users.
xla.device_put = device_put

def _device_put_array(x, device: Optional[Device]):
  backend = xb.get_device_backend(device)
  if x.dtype == dtypes.float0:
    x = np.zeros(x.shape, dtype=np.dtype(bool))
  return (backend.buffer_from_pyval(x, device),)

def _device_put_scalar(x, device):
  return _device_put_array(dtypes.coerce_to_array(x), device)

def _device_put_token(_, device):
  backend = xb.get_device_backend(device)
  return (backend.buffer_from_pyval(np.zeros((), dtype=np.dtype(np.bool_)),
                                    device),)

_scalar_types = dtypes.python_scalar_dtypes.keys()

device_put_handlers: Dict[Any, Callable[[Any, Optional[Device]],
                                        Tuple[Any, ...]]] = {}
device_put_handlers.update((t, _device_put_array) for t in array_types)
device_put_handlers.update((t, _device_put_scalar) for t in _scalar_types)
device_put_handlers[core.Token] = _device_put_token
device_put_handlers[core.BInt] = lambda x, d: _device_put_scalar(x.val, d)


def _device_put_device_array(x: Union[device_array.DeviceArrayProtocol, device_array._DeviceArray], device: Optional[Device]):
  x = _copy_device_array_to_device(x, device)
  return (x.device_buffer,)
for t in device_array.device_array_types:
  device_put_handlers[t] = _device_put_device_array
device_put_handlers[core.PaddedArray] = lambda x, d: device_put(x._data, d)

def _copy_device_array_to_device(
    x: Union[device_array.DeviceArrayProtocol, device_array._DeviceArray],
    device: Optional[xc.Device]
  ) -> Union[device_array.DeviceArrayProtocol, device_array._DeviceArray]:
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
    moved_buf = backend.buffer_from_pyval(np.asarray(x.device_buffer), device)
  return device_array.make_device_array(x.aval, device, moved_buf)


def _copy_array_to_device(x: jax.Array, device: Optional[xc.Device]) -> jax.Array:
  """Copies `Array`s with SingleDeviceSharding to a different device."""
  from jax._src import array, sharding

  if device is None:
    # no copying to be done because there's no target specified
    return x

  buf = x._arrays[0]
  if xb.get_device_backend(device).platform == buf.platform():
    # source and target platforms are the same
    if x.device() == device:
      # no copying to be done because source equals target
      if x._committed:
        return x
      else:
        moved_buf = buf  # We need to change stickyness
    else:
      # move the buffer with a device-to-device copy
      moved_buf = buf.copy_to_device(device)
  else:
    # buffers from different XLA backends are passed through the host.
    backend = xb.get_device_backend(device)
    moved_buf = backend.buffer_from_pyval(np.asarray(buf), device)
  return array.ArrayImpl(
      x.aval, sharding.SingleDeviceSharding(moved_buf.device()), [moved_buf],
      committed=(device is not None))


def _device_put_impl(
    x, device: Optional[Union[Device, jax.sharding.Sharding]] = None):
  from jax._src import array, sharding

  if isinstance(device, sharding.Sharding):
    if not device.is_fully_addressable():  # type: ignore
      raise ValueError(
          "device_put's second argument must be a Device or a Sharding which "
          f"represents addressable devices, but got {sharding}")
    if getattr(x, 'sharding', None) == device:
      return x
    # TODO(mattjj,yashkatariya,phawkins): runtime fast resharding here?
    return array.make_array_from_callback(x.shape, device, lambda idx: x[idx])

  # Only `Device` exists below. `Sharding` instance is handled above.
  if isinstance(x, array.ArrayImpl):
    if not x.is_fully_addressable():
      raise ValueError(
          "device_put's first argument must be a fully addressable array, but "
          f"got value with devices {x.devices()}")
    if device is None:
      return x
    elif is_single_device_sharding(x.sharding):
      return _copy_array_to_device(x, device)

  if device_array.type_is_device_array(x):
    return _copy_device_array_to_device(x, device)

  try:
    a = xla.abstractify(x)
  except TypeError as err:
    raise TypeError(
        f"Argument '{x}' of type {type(x)} is not a valid JAX type") from err
  return aval_to_result_handler(device, a)(None, *device_put(x, device))


device_put_p = core.Primitive('device_put')
device_put_p.def_impl(_device_put_impl)
device_put_p.def_abstract_eval(lambda x, device=None: x)
ad.deflinear2(device_put_p, lambda cotangent, _, **kwargs: [cotangent])
batching.defvectorized(device_put_p)

def _device_put_lowering(ctx, x, *, device):
  return [x]


mlir.register_lowering(device_put_p, _device_put_lowering)
