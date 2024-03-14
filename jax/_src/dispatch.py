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
from collections.abc import Iterator, Sequence
import contextlib
from functools import partial
import itertools
import time
from typing import Any, Callable, NamedTuple
import logging
import threading

import numpy as np

import jax
from jax._src import basearray
from jax._src import config
from jax._src import core
from jax._src import api
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import xla
from jax._src.interpreters import pxla
from jax._src import lib
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax._src.monitoring import record_event_duration_secs
from jax._src.partition_spec import PartitionSpec
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (
    PmapSharding, SingleDeviceSharding, NamedSharding, XLACompatibleSharding,
    GSPMDSharding, TransferToMemoryKind)


JAXPR_TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
JAXPR_TO_MLIR_MODULE_EVENT = "/jax/core/compile/jaxpr_to_mlir_module_duration"
BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"

traceback_util.register_exclusion(__file__)

xe = xc._xla

Backend = xe.Client
Device = xc.Device

CompileOptions = xc.CompileOptions

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

logger = logging.getLogger(__name__)

# This flag is set on exit; no logging should be attempted
_on_exit = False

### op-by-op execution

def apply_primitive(prim, *args, **params):
  """Impl rule that compiles and runs a single primitive 'prim' using XLA."""
  fun = xla_primitive_callable(prim, **params)
  # TODO(yashkatariya): Investigate adding is_primitive to jit and never
  # triggering the disable jit path instead of messing around with it here.
  if xla_extension_version >= 218:
    prev = lib.jax_jit.swap_thread_local_state_disable_jit(False)
    try:
      outs = fun(*args)
    finally:
      lib.jax_jit.swap_thread_local_state_disable_jit(prev)
  else:
    with config.disable_jit(False):
      outs = fun(*args)
  return outs

@util.cache()
def xla_primitive_callable(prim: core.Primitive, **params):
  def prim_fun(*args):
    return prim.bind(*args, **params)
  prim_fun.__name__ = prim.name
  prim_fun.__qualname__ = prim.name
  return api.jit(prim_fun)


def simple_impl(prim):
  prim.def_impl(partial(apply_primitive, prim))

RuntimeToken = Any

class RuntimeTokenSet(threading.local):
  """See docstring for effect.py module for the calling convention for tokens."""

  # For each ordered effect, the token returned by the last dispatched
  # computation, sharded over the devices in that computation.
  current_tokens: dict[core.Effect, jax.Array]

  # For each device, the runtime token returned by the last dispatched
  # computation on that device.
  output_runtime_tokens: dict[Device, RuntimeToken]

  def __init__(self):
    self.current_tokens = {}
    self.output_runtime_tokens = {}

  def get_token_input(self, eff: core.Effect,
                      devices: list[Device]) -> jax.Array:
    tok = self.current_tokens.get(eff, np.zeros(0, np.bool_))
    s = jax.sharding.GSPMDSharding.get_replicated(devices)
    sharded_tok = pxla.shard_args([s], [tok])[0]
    self.current_tokens[eff] = sharded_tok
    return sharded_tok

  def set_token_result(self, eff: core.Effect, token: jax.Array):
    self.current_tokens[eff] = token

  def set_output_runtime_token(self, device: Device, token: RuntimeToken):
    # We're free to clobber the previous output token because on each
    # device we have a total ordering of computations. Only the token
    # from the latest computation matters.
    self.output_runtime_tokens[device] = token

  def clear(self):
    self.current_tokens = {}
    self.output_runtime_tokens = {}

  def block_until_ready(self):
    for token in self.current_tokens.values():
      token.block_until_ready()
    for token in self.output_runtime_tokens.values():
      token.block_until_ready()
    self.clear()

runtime_tokens: RuntimeTokenSet = RuntimeTokenSet()

@atexit.register
def wait_for_tokens():
  runtime_tokens.block_until_ready()


def is_single_device_sharding(sharding: Sharding) -> bool:
  # Special case PmapSharding here because PmapSharding maps away an axis
  # and needs to be handled separately.test_pjit_single_device_sharding_add
  return len(sharding.device_set) == 1 and not isinstance(sharding, PmapSharding)


@contextlib.contextmanager
def log_elapsed_time(fmt: str, fun_name: str, event: str | None = None):
  if _on_exit:
    yield
  else:
    log_priority = logging.WARNING if config.log_compiles.value else logging.DEBUG
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    if logger.isEnabledFor(log_priority):
      logger.log(log_priority, fmt.format(
          fun_name=fun_name, elapsed_time=elapsed_time))
    if event is not None:
      record_event_duration_secs(event, elapsed_time)


def should_tuple_args(num_args: int, platform: str) -> bool:
  # CPU and GPU do not need tuples as they use host-side data structures that
  # do not have small bounds.
  # TPU only needs a tuple for very long lists
  if platform == "tpu":
    return num_args > 2000
  else:
    return False

def jaxpr_has_primitive(jaxpr: core.Jaxpr, prim_name: str) -> bool:
  """Whether there is a primitive given by user anywhere inside a Jaxpr."""
  for eqn in jaxpr.eqns:
    if prim_name in eqn.primitive.name:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_primitive(subjaxpr, prim_name):
      return True
  return False


# Use this registry with caution. It will void the guarantee that lowering to
# stablehlo is oblivious of physical devices.
prim_requires_devices_during_lowering: set[core.Primitive] = set()

def jaxpr_has_prim_requiring_devices(jaxpr: core.Jaxpr):
  for eqn in jaxpr.eqns:
    if eqn.primitive in prim_requires_devices_during_lowering:
      return True
  for subjaxpr in core.subjaxprs(jaxpr):
    if jaxpr_has_prim_requiring_devices(subjaxpr):
      return True
  return False


class SourceInfo(NamedTuple):
  source_info: source_info_util.SourceInfo
  eqn_name: str


def jaxpr_shardings(
    jaxpr: core.Jaxpr,
) -> Iterator[tuple[XLACompatibleSharding, SourceInfo]]:
  from jax._src import pjit, shard_map
  for eqn in jaxpr.eqns:
    if eqn.primitive is pjit.sharding_constraint_p:
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      yield (eqn.params['sharding'], source_info)
    elif eqn.primitive is pjit.pjit_p:
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      yield from ((i, source_info) for i in eqn.params['in_shardings'])
      yield from ((o, source_info) for o in eqn.params['out_shardings'])
    elif eqn.primitive is shard_map.shard_map_p:
      source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
      def _names_to_pspec(names):
        ndmin = max(names) + 1 if names else 0
        return PartitionSpec(*(names.get(i) for i in range(ndmin)))
      yield from ((NamedSharding(eqn.params['mesh'], _names_to_pspec(names)), source_info)
                  for names in [*eqn.params['in_names'], *eqn.params['out_names']])
    elif eqn.primitive is device_put_p:
      s = eqn.params['device']
      if isinstance(s, XLACompatibleSharding) and s.memory_kind is not None:
        source_info = SourceInfo(eqn.source_info, eqn.primitive.name)
        yield (s, source_info)
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from jaxpr_shardings(subjaxpr)


def jaxpr_has_bints(jaxpr: core.Jaxpr) -> bool:
  return (any(type(v.aval.dtype) is core.bint for v in jaxpr.invars
              if isinstance(v.aval, core.UnshapedArray)) or
          any(_is_bint_axis_size(d)
              for j in itertools.chain([jaxpr], core.subjaxprs(jaxpr))
              for e in j.eqns for v in e.outvars
              if isinstance(v.aval, core.DShapedArray) for d in v.aval.shape))

def _is_bint_axis_size(d: core.AxisSize) -> bool:
  if isinstance(d, core.DArray):
    assert not d.shape
    return type(d.dtype) is core.bint
  elif isinstance(d, core.Var):
    return (isinstance(d.aval, core.DShapedArray) and
            type(d.aval.dtype) is core.bint)
  return False


# We can optionally set a Jaxpr rewriter that can be applied just before
# compilation. This mechanism is used for compiling id_tap, we can
# remove it once we bring the id_tap implementation into the core.
outfeed_rewriter: Callable[[core.Jaxpr], core.Jaxpr] | None = None
def apply_outfeed_rewriter(jaxpr: core.Jaxpr) -> core.Jaxpr:
  if outfeed_rewriter is not None:
    return outfeed_rewriter(jaxpr)
  else:
    return jaxpr


def check_arg(arg: Any):
  if not (isinstance(arg, core.Tracer) or core.valid_jaxtype(arg)):
    raise TypeError(f"Argument '{arg}' of type {type(arg)} is not a valid "
                    "JAX type.")


def jaxpr_replicas(jaxpr: core.Jaxpr) -> int:
  """The number of replicas needed for a jaxpr.

  For a eqn, multiply the `axis_size` with the `jaxpr_replicas` of the
  subjaxprs. For a list of eqns, take the maximum number of replicas.
  """
  return max(unsafe_map(_eqn_replicas, jaxpr.eqns), default=1)

# TODO(mattjj): this function assumes that only pmap has a parameter named
# axis_size, and that it corresponds to cross-replica mapping
def _eqn_replicas(eqn: core.JaxprEqn) -> int:
  call_jaxpr = eqn.params.get("call_jaxpr")
  if call_jaxpr:
    return eqn.params.get('axis_size', 1) * jaxpr_replicas(call_jaxpr)
  elif eqn.primitive in xla.initial_style_primitives:
    return _initial_style_primitive_replicas(eqn.params)
  else:
    return 1

def _initial_style_primitive_replicas(params: dict[str, Any]) -> int:
  return max(core.traverse_jaxpr_params(jaxpr_replicas, params).values(),
             default=1)

def needs_check_special() -> bool:
  return config.debug_infs.value or config.debug_nans.value

def check_special(name: str, bufs: Sequence[basearray.Array]) -> None:
  if needs_check_special():
    for buf in bufs:
      _check_special(name, buf.dtype, buf)

def _check_special(name: str, dtype: np.dtype, buf: basearray.Array) -> None:
  if dtypes.issubdtype(dtype, np.inexact):
    if config.debug_nans.value and np.any(np.isnan(np.asarray(buf))):
      raise FloatingPointError(f"invalid value (nan) encountered in {name}")
    if config.debug_infs.value and np.any(np.isinf(np.asarray(buf))):
      raise FloatingPointError(f"invalid value (inf) encountered in {name}")


def _put_x(x, s: Sharding, aval: core.AbstractValue, committed: bool):
  result_handler = pxla.global_aval_to_result_handler(aval, s, committed)
  return result_handler(pxla.shard_arg(x, s))

def _override_get_device_assignment(sharding, *args, **kwargs):
  da = sharding._device_assignment
  return xb.get_device_backend(da[0]), da

def _identity_fn(x):
  return x

def _mcjax_reshard(x, target_sharding):
  from jax._src import api, array

  inp_sharding = x.sharding

  if inp_sharding._device_assignment == target_sharding._device_assignment:
    return api.jit(_identity_fn, out_shardings=target_sharding)(x)

  if inp_sharding.device_set != target_sharding.device_set:
    inp_ids = [d.id for d in inp_sharding._device_assignment]
    inp_plat = inp_sharding._device_assignment[0].platform.upper()
    target_ids = [d.id for d in target_sharding._device_assignment]
    target_plat = target_sharding._device_assignment[0].platform.upper()
    raise ValueError("Input and target sharding should have the same set of "
                     f"devices. Got input's device set ids: {inp_ids} on "
                     f"platform {inp_plat} and target sharding's device set "
                     f"ids: {target_ids} on platform {target_plat}")

  old_hlo_sharding = inp_sharding._to_xla_hlo_sharding(x.ndim)
  if old_hlo_sharding.is_replicated():
    new_hlo_sharding = old_hlo_sharding
  else:
    permute_order = np.vectorize(target_sharding._device_assignment.index,
                                 otypes=[int])(inp_sharding._device_assignment)
    # Unfortunately need to fallback to V1 sharding here.
    new_op_sharding = old_hlo_sharding.to_proto()
    new_op_sharding.iota_reshape_dims = []
    new_op_sharding.iota_transpose_perm = []
    new_op_sharding.tile_assignment_devices = np.take(
        old_hlo_sharding.tile_assignment_devices(), permute_order
    )
    new_hlo_sharding = xc.HloSharding.from_proto(new_op_sharding)

  new_x = array.make_array_from_single_device_arrays(
      x.shape,
      GSPMDSharding(target_sharding._device_assignment, new_hlo_sharding),
      x._arrays,
  )

  _orig_get_and_check_device_assignment = pxla._get_and_check_device_assignment.fn
  pxla._get_and_check_device_assignment.fn = partial(
      _override_get_device_assignment, target_sharding)
  try:
    return api.jit(_identity_fn, out_shardings=target_sharding)(new_x)
  finally:
    pxla._get_and_check_device_assignment.fn = _orig_get_and_check_device_assignment


def _device_put_impl(
    x,
    device: Device | Sharding | None = None,
    src: Device | Sharding | None = None):
  from jax._src import array

  if (isinstance(device, TransferToMemoryKind) or
      isinstance(src, TransferToMemoryKind)):
    raise ValueError(
        "TransferToMemoryKind argument to jax.device_put can only be used"
        " inside jax.jit. If you are using device_put outside jax.jit, then"
        " please provide a concrete Sharding with memory_kind.")

  try:
    aval = xla.abstractify(x)
  except TypeError as err:
    raise TypeError(
        f"Argument '{x}' of type {type(x)} is not a valid JAX type") from err

  if isinstance(device, Sharding):
    s = device
    if getattr(x, 'sharding', None) == s and getattr(x, '_committed', False):
      return x
    if (not s.is_fully_addressable and  # type: ignore
        isinstance(x, array.ArrayImpl) and not x.is_fully_addressable):
      # This has to be XLACompatible because _mcjax_reshard will run a
      # XLA computation.
      assert isinstance(s, XLACompatibleSharding)
      return _mcjax_reshard(x, s)
    if not s.is_fully_addressable:  # type: ignore
      # TODO(yashkatariya,mattjj): Link to a doc about McJAX and jax.Array.
      raise ValueError(
          "device_put's second argument must be a Device or a Sharding which"
          f" represents addressable devices, but got {s}. You are probably"
          " trying to use device_put in multi-controller JAX which is not"
          " supported. Please use jax.make_array_from_single_device_arrays API"
          " or pass device or Sharding which represents addressable devices.")
    return _put_x(x, s, aval, True)

  # Only `Device` exists below. `Sharding` instance is handled above.
  if isinstance(x, array.ArrayImpl):
    if not x.is_fully_addressable:
      raise ValueError(
          "device_put's first argument must be a fully addressable array, but "
          f"got value with devices {x.devices()}")
    if device is None:
      return x
    elif is_single_device_sharding(x.sharding):
      return pxla.batched_device_put(aval, SingleDeviceSharding(device), [x],
                                     [device])

  sh = SingleDeviceSharding(pxla._get_default_device()
                            if device is None else device)
  return _put_x(x, sh, aval, device is not None)


device_put_p = core.Primitive('device_put')
device_put_p.def_impl(_device_put_impl)
device_put_p.def_abstract_eval(lambda x, device=None, src=None: x)

def device_put_transpose_rule(ct, _, device, src):
  return [device_put_p.bind(ct, device=src, src=device)]
ad.deflinear2(device_put_p, device_put_transpose_rule)
batching.defvectorized(device_put_p)

def _tpu_device_put_lowering(ctx, x, *, device, src):
  if (isinstance(device, (XLACompatibleSharding, TransferToMemoryKind)) and
      device.memory_kind is not None):
    aval, = ctx.avals_in
    out_aval, = ctx.avals_out
    x = mlir.wrap_with_memory_kind(x, device.memory_kind, out_aval)
    if isinstance(device, XLACompatibleSharding):
      x = mlir.wrap_with_sharding_op(
          ctx, x, out_aval, device._to_xla_hlo_sharding(aval.ndim).to_proto())
    return [x]
  return [x]
mlir.register_lowering(device_put_p, _tpu_device_put_lowering, platform='tpu')


def _common_device_put_lowering(ctx, x, *, device, src):
  if (isinstance(device, (XLACompatibleSharding, TransferToMemoryKind)) and
      device.memory_kind is not None):
    raise NotImplementedError(
        "Passing memory_kind to device_put via Shardings is not supported on"
        f" platforms {ctx.module_context.platforms}")
  return [x]
mlir.register_lowering(device_put_p, _common_device_put_lowering)
