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
"""Implementation of pmap and related functionality."""

from __future__ import annotations

import enum
from contextlib import contextmanager
from collections import namedtuple
from collections.abc import Sequence, Iterable
import dataclasses
from functools import partial, lru_cache, cached_property
import itertools as it
import logging
import math
import threading
from typing import Any, Callable, NamedTuple, TypeVar, Union, cast
from collections.abc import Iterator
import warnings

import numpy as np

import jax
from jax.errors import JAXTypeError

from jax._src import api_util
from jax._src import compiler
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import op_shardings
from jax._src import sharding_specs
from jax._src import profiler
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import stages
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.abstract_arrays import array_types
from jax._src.core import DShapedArray
from jax._src.core import ShapedArray
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import mlir
from jax._src.interpreters import xla
from jax._src.layout import XLACompatibleLayout, SpecifiedLayout, LayoutRequest
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.partition_spec import PartitionSpec
from jax._src.sharding_impls import (
    ArrayMapping, ArrayMappingOrAutoOrUnspecified,
    AUTO, UnspecifiedValue, get_array_mapping as _get_array_mapping, is_auto,
    is_unspecified, is_unspecified_or_auto, array_mapping_to_axis_resources,
    SingleDeviceSharding, GSPMDSharding
)
from jax._src.util import (safe_map, safe_zip, partition_list,
                           wrap_name, tuple_update, tuple_delete,
                           distributed_debug_log,
                           unzip2, HashableFunction, weakref_lru_cache)


# Built in Python lists don't support weak refs but subclasses of lists do.
class WeakRefList(list):
  pass


xe = xc._xla

unsafe_map, map = map, safe_map  # type: ignore

logger = logging.getLogger(__name__)

Index = Union[int, slice, tuple[Union[int, slice], ...]]

NoSharding = sharding_specs.NoSharding
Chunked = sharding_specs.Chunked
Unstacked = sharding_specs.Unstacked

ShardedAxis = sharding_specs.ShardedAxis
Replicated = sharding_specs.Replicated

AvalDimSharding = Union[Unstacked, Chunked, NoSharding]
Mesh = mesh_lib.Mesh
MeshAxisName = sharding_impls.MeshAxisName
MeshDimAssignment = Union[ShardedAxis, Replicated]
ShardingSpec = sharding_specs.ShardingSpec

### util

def identity(x): return x

def shard_arg(arg, sharding, canonicalize=True):
  if canonicalize:
    arg = xla.canonicalize_dtype(arg)
  return shard_arg_handlers[type(arg)](arg, sharding)


@profiler.annotate_function
def shard_args(
    shardings: Sequence[sharding_impls.XLACompatibleSharding], args,
) -> Sequence[jax.Array]:
  return [shard_arg(arg, shardings[i]) for i, arg in enumerate(args)]

shard_arg_handlers: dict[Any, Callable[[Any, Any], Any]] = {}


@lru_cache(maxsize=1024)
def get_addressable_devices_for_shard_arg(
    s: sharding_impls.XLACompatibleSharding) -> tuple[xc.Device, ...]:
  return s._addressable_device_assignment

@lru_cache(maxsize=1024)
def _get_replicated_slices(num_addressable_devices: int):
  return ((slice(None),),) * num_addressable_devices

def _shard_token(x, sharding):
  devices = get_addressable_devices_for_shard_arg(sharding)
  indices = _get_replicated_slices(len(devices))
  zeros = np.zeros((), dtype=np.dtype(np.bool_))
  aval = api_util.shaped_abstractify(zeros)
  return batched_device_put(aval, sharding, [zeros for _ in indices], devices)
shard_arg_handlers[core.Token] = _shard_token

def _masked_array_error(x, sharding):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")
shard_arg_handlers[np.ma.MaskedArray] = _masked_array_error

def _shard_array(x, sharding):
  indices = tuple(sharding.addressable_devices_indices_map(x.shape).values())
  devices = get_addressable_devices_for_shard_arg(sharding)
  if x.dtype == dtypes.float0:
    x = np.zeros(x.shape, dtype=np.dtype(bool))
  aval = api_util.shaped_abstractify(x)
  return batched_device_put(aval, sharding, [x[i] for i in indices], devices)
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array

def _shard_darray(x, sharding):
  return shard_arg(x._data, sharding)
shard_arg_handlers[core.DArray] = _shard_darray

def batched_device_put(aval: core.ShapedArray,
                       sharding: jax.sharding.Sharding, xs: Sequence[Any],
                       devices: Sequence[jax.Device], committed: bool = True):
  from jax._src import array

  bufs = [x for x, d in safe_zip(xs, devices)
          if (isinstance(x, array.ArrayImpl) and
              dispatch.is_single_device_sharding(x.sharding) and
              x.devices() == {d})]
  if len(bufs) == len(xs):
    return array.ArrayImpl(
        aval, sharding, bufs, committed=committed, _skip_checks=True)
  return xc.batched_device_put(aval, sharding, xs, list(devices), committed)  # type: ignore

def _shard_aval(size, axis: int, aval):
  try:
    return _shard_aval_handlers[type(aval)](size, axis, aval)
  except KeyError as err:
    raise TypeError(f"No _shard_aval handler for type: {type(aval)}") from err
_shard_aval_handlers: dict[type[core.AbstractValue], Callable[[int, int, Any], Any]] = {}

def _shard_abstract_array(size, axis: int, x):
  try:
    if x.shape[axis] != size:
      raise ValueError(f"Axis size {size} does not match dimension {axis} of "
                       f"shape {x.shape}")
  except IndexError:
    raise ValueError("Cannot split a {x.dim}D value along axis {axis}") from None
  if config.pmap_no_rank_reduction.value:
    return x.update(shape=tuple_update(x.shape, axis, 1))
  else:
    return x.update(shape=tuple_delete(x.shape, axis))
_shard_aval_handlers[ShapedArray] = _shard_abstract_array


def local_aval_to_result_handler(
    aval: core.AbstractValue,
    sharding: sharding_impls.XLACompatibleSharding,
    indices: tuple[Index, ...] | None,
) -> Callable[[list[xc.ArrayImpl]], Any]:
  """Returns a function for handling the raw buffers of a single output aval.

  Args:
    aval: The local output AbstractValue.
    sharding_spec: Indicates how the output is sharded across devices, or None
      for non-array avals.
    indices: The pre-computed result of spec_to_indices, or None for non-array
      avals.

  Returns:
    A function for handling the Buffers that will eventually be produced
    for this output. The function will return an object suitable for returning
    to the user, e.g. an Array.
  """
  try:
    return local_result_handlers[(type(aval))](aval, sharding, indices)
  except KeyError as err:
    raise TypeError(
        f"No pxla_result_handler for type: {type(aval)}") from err

PxlaResultHandler = Callable[..., Callable[[Any], Any]]
local_result_handlers: dict[type[core.AbstractValue], PxlaResultHandler] = {}


def global_aval_to_result_handler(
    aval: core.AbstractValue, out_sharding, committed: bool,
    is_out_sharding_from_xla: bool
) -> Callable[[Sequence[xc.ArrayImpl]], Any]:
  """Returns a function for handling the raw buffers of a single output aval.

  Args:
    aval: The global output AbstractValue.
    out_axis_resources: A PartitionSpec specifying the sharding of outputs.
      Used for creating GSDAs.
    global_mesh: The global device mesh that generated this output. Used
      for creating GSDAs.
    is_out_sharding_from_xla: True, if the out_sharding comes from XLA i.e.
      the sharding is extracted from the HLO.

  Returns:
    A function for handling the Buffers that will eventually be produced
    for this output. The function will return an object suitable for returning
    to the user, e.g. an Array.
  """
  try:
    return global_result_handlers[type(aval)](
        aval, out_sharding, committed, is_out_sharding_from_xla)
  except KeyError as err:
    raise TypeError(
        f"No pxla_result_handler for type: {type(aval)}") from err

global_result_handlers: dict[type[core.AbstractValue], PxlaResultHandler] = {}

### lazy device-memory persistence and result handling

### the xla_pmap primitive and its rules are comparable to xla_call in xla.py


def xla_pmap_impl_lazy(
    fun: lu.WrappedFun,
    *args,
    backend: str | None,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[Any] | None,
    name: str,
    in_axes: Sequence[int | None],
    out_axes_thunk: Callable[[], Sequence[int | None]],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
) -> Callable:
  if (config.disable_jit.value and config.eager_pmap.value and
      not is_explicit_global_axis_size and not any(d for d in donated_invars)):
    def _emap_apply_fn(*args):
      return _emap_impl(fun, *args, backend=backend, axis_name=axis_name,
                        axis_size=axis_size, global_axis_size=global_axis_size,
                        devices=devices, name=name, in_axes=in_axes,
                        out_axes_thunk=out_axes_thunk,
                        donated_invars=donated_invars,
                        is_explicit_global_axis_size=is_explicit_global_axis_size)
    return _emap_apply_fn
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun, fingerprint = parallel_callable(
      fun, backend, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, out_axes_thunk, donated_invars,
      is_explicit_global_axis_size, *abstract_args)

  # Don't re-abstractify args unless logging is enabled for performance.
  if config.distributed_debug.value:
    distributed_debug_log(("Running pmapped function", name),
                          ("python function", fun.f),
                          ("devices", devices),
                          ("abstract args", map(xla.abstractify, args)),
                          ("fingerprint", fingerprint))
  return compiled_fun

def xla_pmap_impl(fun: lu.WrappedFun, *args, **params):
  compiled_fun = xla_pmap_impl_lazy(fun, *args, **params)
  return compiled_fun(*args)

class EmapInfo(NamedTuple):
  backend: str | None
  devices: Sequence[Any] | None

def _emap_impl(fun: lu.WrappedFun, *args,
               backend: str | None,
               axis_name: core.AxisName,
               axis_size: int,
               global_axis_size: int,
               devices: Sequence[Any] | None,
               name: str,
               in_axes: Sequence[int | None],
               out_axes_thunk: Callable[[], Sequence[int | None]],
               donated_invars: Sequence[bool],
               is_explicit_global_axis_size: bool,
               ):
  from jax._src import array
  # TODO(sharadmv,mattjj): implement these cases
  if any(d for d in donated_invars):
    raise NotImplementedError("Buffer donation not supported in eager pmap.")
  if is_explicit_global_axis_size:
    raise NotImplementedError("Non-default global_axis_size not supported in "
                              "eager pmap.")

  emap_info = EmapInfo(backend, devices)
  shard_axes = [{} if in_axis is None else {axis_name: in_axis} for in_axis in in_axes]
  with core.new_base_main(MapTrace, emap_info=emap_info) as main:
    with core.new_sublevel(), core.extend_axis_env(axis_name, axis_size, main):
      t = main.with_cur_sublevel()
      tracers = [
          MapTracer(t, arg, s) for arg, s in zip(args, shard_axes)]
      ans = fun.call_wrapped(*tracers)
      out_tracers = map(t.full_raise, ans)
      outvals, out_axes_src = unzip2((t.val, t.shard_axes) for t in out_tracers)
    del main
  out_axes = out_axes_thunk()

  platform = xb.get_backend(backend).platform
  donate_argnums = (1,) if platform in {"cuda", "rocm", "tpu"} else ()
  new_outvals = []
  for out_axis_src, out_axis, outval in zip(out_axes_src, out_axes, outvals):
    with jax.disable_jit(False):
      donate_argnums_ = donate_argnums
      if isinstance(outval, array.ArrayImpl):
        # We don't want to donate if it's already sharded.
        donate_argnums_ = ()
      out = jax.pmap(
          lambda _, x: x,
          in_axes=(0, out_axis_src.get(axis_name)),
          out_axes=out_axis,
          devices=(None if devices is None else list(devices)),
          backend=backend,
          donate_argnums=donate_argnums_)(np.arange(axis_size), outval)
      new_outvals.append(out)
  return new_outvals

def _map_schedule(idx: tuple[int | None, ...]) -> tuple[int | None, ...]:
  # In order to do a multi-map (a simultaneous map over several axes), we will
  # nest several maps. Each time we do a map, we "remove" an input axis so we
  # need to update the remaining map axes. For example, if we are to map over
  # the axes 0, 3, and 4, we make three calls to pmap with in_axes as 0, 2, 2.
  return tuple(None if i is None else
               i - sum(j is not None and j < i for j in idx[:l])
               for l, i in enumerate(idx))


# We're often creating `f`s on the fly and we try to carefully make them have
# the right __hash__ and __eq__. However, despite our attempts pmap's caching
# still ends up not working, because it has a separate cache per
# _function object_. Adding this annotation here lets us reuse the same pmap
# callable for all equivalent primitive pmaps.
@lru_cache
def _multi_pmap(f: Callable, info: EmapInfo, names: list[core.AxisName],
                all_axes: list[tuple[int | None, ...]]
                ) -> tuple[Callable, dict[core.AxisName, int]]:
  used_names = []
  for i, name in reversed(list(enumerate(names))):
    in_axes = tuple(arg_axis[i] for arg_axis in all_axes)
    if any(in_axis is not None for in_axis in in_axes):
      f = jax.pmap(
          f,
          in_axes=in_axes,
          axis_name=name,
          out_axes=0,
          backend=info.backend,
          devices=(None if info.devices is None else list(info.devices)))
      used_names.append(name)
  out_shard_axes = {name: i for i, name in enumerate(reversed(used_names))}
  return f, out_shard_axes

FakePrimitive = namedtuple("FakePrimitive", ["multiple_results", "bind"])

class MapTrace(core.Trace):

  def __init__(self, *args, emap_info):
    super().__init__(*args)
    self.emap_info = emap_info

  def pure(self, val):
    return MapTracer(self, val, {})

  def sublift(self, tracer):
    return MapTracer(self, tracer.val, tracer.shard_axes)

  def process_primitive(self, primitive, tracers, params):
    info = self.main.payload["emap_info"]
    vals, shard_axes = unzip2([(t.val, t.shard_axes) for t in tracers])
    names = tuple(f.name for f in core.thread_local_state.trace_state.axis_env
                  if f.main_trace is self.main)
    all_axes = tuple(_map_schedule(map(s.get, names)) for s in shard_axes)  # pytype: disable=wrong-arg-types  # always-use-return-annotations
    f = HashableFunction(lambda *args: primitive.bind(*args, **params),
                         (primitive, tuple(params.items())))
    f_mapped, out_shard_axes = _multi_pmap(f, info, names, all_axes)
    with core.eval_context(), jax.disable_jit(False):
      outvals = f_mapped(*vals)
    if primitive.multiple_results:
      return [MapTracer(self, val, out_shard_axes) for val in outvals]
    return MapTracer(self, outvals, out_shard_axes)

  def process_call(self, call_primitive, fun, tracers, params):
    raise NotImplementedError

  def process_map(self, map_primitive, fun, tracers, params):
    if params['devices'] is not None:
      raise ValueError("Nested pmap with explicit devices argument.")
    if not config.disable_jit.value:
      bind = HashableFunction(
          lambda *args, **kwargs: map_primitive.bind(fun, *args, **kwargs),
          (map_primitive, fun))
      fake_primitive = FakePrimitive(multiple_results=True, bind=bind)
      return self.process_primitive(fake_primitive, tracers, params)
    axis_name, in_axes, out_axes_thunk, axis_size = (params["axis_name"],
        params["in_axes"], params["out_axes_thunk"], params["axis_size"])
    vals, shard_axes = unzip2((t.val, t.shard_axes) for t in tracers)
    shard_axes = [{axis_name: _annot_to_flat(np.ndim(v), s.values(), ax), **s}
                  if ax is not None else s
                  for v, ax, s in zip(vals, in_axes, shard_axes)]
    # TODO(mattjj): use _emap_subtrace here?
    with core.new_sublevel(), core.extend_axis_env(axis_name, axis_size, self.main):
      t = self.main.with_cur_sublevel()
      in_tracers = map(partial(MapTracer, t), vals, shard_axes)
      ans = fun.call_wrapped(*in_tracers)
      out_tracers = map(t.full_raise, ans)
      out, outaxes = unzip2((t.val, t.shard_axes) for t in out_tracers)
      del t, in_tracers, ans, out_tracers
    out, outaxes = unzip2(_match_annot(axis_name, axis_size, v, s, dst)
                           for v, s, dst in zip(out, outaxes, out_axes_thunk()))
    return map(partial(MapTracer, self), out, outaxes)

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros):
    if symbolic_zeros:
      msg = ("custom_jvp with symbolic_zeros=True not supported with eager pmap. "
             "Please open an issue at https://github.com/google/jax/issues !")
      raise NotImplementedError(msg)
    del prim, jvp, symbolic_zeros  # always base main, can drop jvp
    in_vals, in_axes = unzip2((t.val, t.shard_axes) for t in tracers)
    fun, out_axes = _emap_subtrace(fun, self.main, in_axes)
    with core.new_sublevel():
      out_vals = fun.call_wrapped(*in_vals)
    return map(partial(MapTracer, self), out_vals, out_axes())

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,
                              out_trees, symbolic_zeros):
    if symbolic_zeros:
      msg = ("custom_vjp with symbolic_zeros=True not supported with eager pmap. "
             "Please open an issue at https://github.com/google/jax/issues !")
      raise NotImplementedError(msg)
    del primitive, fwd, bwd, out_trees, symbolic_zeros  # always base main, drop vjp
    in_vals, in_axes = unzip2((t.val, t.shard_axes) for t in tracers)
    fun, out_axes = _emap_subtrace(fun, self.main, in_axes)
    with core.new_sublevel():
      out_vals = fun.call_wrapped(*in_vals)
    return map(partial(MapTracer, self), out_vals, out_axes())

  def process_axis_index(self, frame):
    bind = HashableFunction(
        lambda _: jax.lax.axis_index(frame.name),
        (jax.lax.axis_index, frame.name))
    fake_primitive = FakePrimitive(multiple_results=False, bind=bind)
    with core.eval_context():
      range = jax.lax.iota(np.int32, frame.size)
    dummy_tracer = MapTracer(self, range, {frame.name: 0})
    return self.process_primitive(fake_primitive, (dummy_tracer,), {})

@lu.transformation_with_aux
def _emap_subtrace(main, in_axes, *in_vals):
  t = main.with_cur_sublevel()
  in_tracers = map(partial(MapTracer, t), in_vals, in_axes)
  ans = yield in_tracers, {}
  out_tracers = map(t.full_raise, ans)
  out_vals, out_axes = unzip2((t.val, t.shard_axes) for t in out_tracers)
  del t, in_tracers, ans, out_tracers
  yield out_vals, out_axes

def _annot_to_flat(ndim: int, mapped_axes: Iterable[int],
                 annotation: int | None) -> int | None:
  if annotation is None: return None
  mapped_axes_ = set(mapped_axes)
  return [i for i in range(ndim) if i not in mapped_axes_][annotation]

def _match_annot(axis_name: core.AxisName, axis_size: int, val: Any,
                 shard_axis_src: dict[core.AxisName, int],
                 dst_annotation: int | None
                 ) -> tuple[Any, dict[core.AxisName, int]]:
  shard_axis_out = dict(shard_axis_src)
  src = shard_axis_out.pop(axis_name, None)
  dst = _annot_to_flat(np.ndim(val) + (src is None), shard_axis_out.values(),
                       dst_annotation)
  with core.eval_context():
    if src == dst:
      outval = val
    elif type(src) == type(dst) == int:
      outval = batching.moveaxis(val, src, dst)
      shard_axis_out = _moveaxis(np.ndim(val), shard_axis_src, src, dst)
    elif src is None and dst is not None:
      outval = batching.broadcast(val, axis_size, dst)
      shard_axis_out = {n: d + (dst <= d) for n, d in shard_axis_out.items()}
    else:
      raise NotImplementedError
  return outval, shard_axis_out

def _moveaxis(ndim: int, shard_axes: dict[core.AxisName, int],
              src: int, dst: int) -> dict[core.AxisName, int]:
  lst: list[core.AxisName | None] = [None] * ndim
  for k, v in shard_axes.items():
    lst[v] = k
  name = lst.pop(src)
  lst.insert(dst - (src < dst), name)
  return {name: i for i, name in enumerate(lst) if name is not None}

class MapTracer(core.Tracer):
  __slots__ = ["val", "shard_axes"]

  def __init__(self, trace: MapTrace, val, shard_axes: dict[core.AxisName, int]):
    self._trace = trace
    self.val = val
    self.shard_axes = shard_axes
    assert all(val < self.val.ndim for val in self.shard_axes.values())

  @property
  def aval(self):
    aval = xla.abstractify(self.val)
    shard_axes = dict(self.shard_axes)
    for axis_idx in sorted(shard_axes.values())[::-1]:
      aval = core.mapped_aval(aval.shape[axis_idx], axis_idx, aval)
    return aval

  def full_lower(self):
    return self

  def __str__(self):
    named_axes = [f"{k}={v}" for k, v in self.shard_axes.items()]
    return f"{self.val}{{{','.join(named_axes)}}}"

@lu.cache
def parallel_callable(fun: lu.WrappedFun,
                      backend_name: str | None,
                      axis_name: core.AxisName,
                      axis_size: int,
                      global_axis_size: int,
                      devices: Sequence[Any] | None,
                      name: str,
                      in_axes: Sequence[int | None],
                      out_axes_thunk: Callable[[], Sequence[int | None]],
                      donated_invars: Sequence[bool],
                      is_explicit_global_axis_size: bool,
                      *avals):
  pmap_computation = lower_parallel_callable(
      fun, backend_name, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, out_axes_thunk, donated_invars,
      is_explicit_global_axis_size, avals,
      lowering_parameters=mlir.LoweringParameters())
  pmap_executable = pmap_computation.compile()
  return WeakRefList([pmap_executable.unsafe_call, pmap_executable.fingerprint])


@dataclasses.dataclass(frozen=True)
class ParallelCallableInfo:
  name: str
  backend: xc.Client
  axis_name: core.AxisName
  axis_size: int
  global_axis_size: int
  devices: Sequence[xc.Device] | None
  in_axes: Iterable[int | None]
  out_axes_thunk: Callable[[], Sequence[int | None]]
  avals: Sequence[core.AbstractValue]

  @cached_property
  def local_devices(self):
    if self.devices:
      out = [d for d in self.devices
             if d.process_index == xb.process_index(self.backend)]
      assert len(out) > 0
    else:
      out = None  # type: ignore
    return out

  @cached_property
  def out_axes(self):
    return self.out_axes_thunk()


class ShardInfo(NamedTuple):
  sharded_avals: Sequence[core.AbstractValue]
  out_sharded_avals: Sequence[core.ShapedArray]
  global_sharded_avals: Sequence[core.AbstractValue]
  num_local_shards: int
  num_global_shards: int


class ReplicaInfo(NamedTuple):
  jaxpr_replicas: int
  num_local_replicas: int
  num_global_replicas: int


def find_replicas(
    jaxpr: core.Jaxpr, axis_size: int, global_axis_size: int
) -> ReplicaInfo:
  # TODO(skyewm): replace this with a chain of pmaps and/or sharded_jits
  jaxpr_replicas = dispatch.jaxpr_replicas(jaxpr)
  num_local_replicas = axis_size * jaxpr_replicas
  num_global_replicas = global_axis_size * jaxpr_replicas
  return ReplicaInfo(jaxpr_replicas, num_local_replicas, num_global_replicas)

@lu.transformation
def _change_argument_ranks(in_axes, out_axes_thunk, *args):
  args = tuple(
      arg if in_axis is None else jax.lax.squeeze(arg, dimensions=(in_axis,))
      for in_axis, arg in zip(in_axes, args)
  )
  results = yield (args, {})
  out_axes = out_axes_thunk()
  yield tuple(
      x if axis is None else jax.lax.expand_dims(x, dimensions=(axis,))
      for x, axis in zip(results, out_axes)
  )


def stage_parallel_callable(
    pci: ParallelCallableInfo, fun: lu.WrappedFun
) -> tuple[core.Jaxpr, list[Any], ReplicaInfo, ShardInfo]:
  sharded_avals = tuple(
      _shard_aval(pci.axis_size, axis, aval) if axis is not None else aval
      for axis, aval in safe_zip(pci.in_axes, pci.avals))

  orig_fun = fun
  if config.pmap_no_rank_reduction.value:
    fun = _change_argument_ranks(fun, pci.in_axes, pci.out_axes_thunk)
  else:
    fun = orig_fun
  with core.extend_axis_env(pci.axis_name, pci.global_axis_size, None):  # type: ignore
    with dispatch.log_elapsed_time(
        "Finished tracing + transforming {fun_name} for pmap in {elapsed_time} sec",
        fun_name=fun.__name__, event=dispatch.JAXPR_TRACE_EVENT):
      jaxpr, out_sharded_avals, consts = pe.trace_to_jaxpr_final(
          fun, sharded_avals, pe.debug_info_final(fun, "pmap"))
  jaxpr = api_util.jaxpr_debug_info(jaxpr, orig_fun.debug_info)
  jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)

  assert len(out_sharded_avals) == len(pci.out_axes), (
      len(out_sharded_avals), len(pci.out_axes))

  replicas = find_replicas(jaxpr, pci.axis_size, pci.global_axis_size)
  num_local_shards = replicas.num_local_replicas
  num_global_shards = replicas.num_global_replicas

  shards = ShardInfo(
      sharded_avals, out_sharded_avals, sharded_avals,
      num_local_shards, num_global_shards)

  return jaxpr, consts, replicas, shards


@profiler.annotate_function
def lower_parallel_callable(
    fun: lu.WrappedFun,
    backend_name: str | None,
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Sequence[xc.Device] | None,
    name: str,
    in_axes: Iterable[int | None],
    out_axes_thunk: Callable[[], Sequence[int | None]],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
    avals: Sequence[core.AbstractValue],
    *,
    lowering_parameters: mlir.LoweringParameters) -> PmapComputation:
  # Determine global_axis_size for use in AxisEnv.
  # TODO(mattjj,skyewm): revive this check (inner_pmap always False now)
  # if xb.process_count() > 1 and global_axis_size is None and inner_pmap:
  #   raise ValueError("'axis_size' must be specified for nested multi-host pmaps")
  if (xb.process_count() == 1 and is_explicit_global_axis_size
      and global_axis_size != axis_size):
    raise ValueError(
        f"Specified axis_size {global_axis_size} doesn't match received "
        f"axis_size {axis_size}.")

  if devices is not None and backend_name is None:
    backend = xb.get_device_backend(devices[0])
  else:
    backend = xb.get_backend(backend_name)

  no_nested_sharding = False
  must_run_on_all_devices = False
  if not is_explicit_global_axis_size:
    if xb.process_count(backend) > 1:
      if devices:
        # This allows each host in a multi-host pmap to run on a different number
        # of devices, but precludes nested sharding (i.e. inner pmaps).
        no_nested_sharding = True
      else:
        # This assumes all hosts run on the same number of devices. We make sure
        # this assumption is true by requiring that the pmap is run on all devices
        # (and making the further assumption that each host has the same number of
        # devices). Nested sharding is ok in this case.
        must_run_on_all_devices = True

  pci = ParallelCallableInfo(
      name, backend, axis_name, axis_size, global_axis_size, devices,
      in_axes, out_axes_thunk, avals)
  jaxpr, consts, replicas, shards = stage_parallel_callable(pci, fun)
  if logger.isEnabledFor(logging.DEBUG):
    logger.debug("sharded_avals: %s", shards.sharded_avals)
    logger.debug("global_sharded_avals: %s", shards.global_sharded_avals)
    logger.debug("num_replicas: %d  num_local_replicas: %d",
                 replicas.num_global_replicas, replicas.num_local_replicas)
    logger.debug("devices: %s", devices)
    logger.debug("local_devices: %s", pci.local_devices)

  if (xb.process_count(backend) > 1 and must_run_on_all_devices and
      shards.num_local_shards != xb.local_device_count(backend)):
    if shards.num_local_shards == axis_size:
      raise ValueError(
         f"On multi-host platforms, the input to pmapped functions must have "
         f"leading axis size equal to the number of local devices if no "
         f"`devices` argument is specified. Got {axis_size=}, "
         f"num_local_devices={xb.local_device_count(backend)}")
    else:
      raise ValueError(
        f"On multi-host platforms, pmapped functions must run across all "
        f"devices, i.e. num_replicas * num_partitions should equal the "
        f"number of local devices. Got "
        f"num_replicas={replicas.num_local_replicas}, and "
        f"num_local_devices={xb.local_device_count(backend)}")

  if no_nested_sharding and replicas.jaxpr_replicas > 1:
    raise ValueError(
      f"On multi-host platforms, pmapped functions that both have `devices` "
      f"specified and contain an inner_pmap must specify an "
      f"`axis_size` (or remove the `devices` argument). Got nested_replicas="
      f"{replicas.jaxpr_replicas}")

  log_priority = logging.WARNING if config.log_compiles.value else logging.DEBUG
  if logger.isEnabledFor(log_priority):
    logger.log(log_priority,
               "Compiling %s (%d) for %d devices with args %s. (num_replicas=%d)",
               fun.__name__, id(fun),
               shards.num_global_shards, avals, replicas.num_global_replicas)

  axis_env = sharding_impls.AxisEnv(
      replicas.num_global_replicas, (axis_name,), (global_axis_size,))
  name_stack = source_info_util.new_name_stack(wrap_name(name, 'pmap'))
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  replicated_args = [axis is None for axis in in_axes]
  tuple_args = dispatch.should_tuple_args(len(shards.global_sharded_avals),
                                          backend.platform)
  module_name = f"pmap_{fun.__name__}"
  with maybe_extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
    ordered_effects = list(
        effects.ordered_effects.filter_in(closed_jaxpr.effects))
    if ordered_effects:
      raise ValueError("Ordered effects not supported in `pmap`.")
    unordered_effects = list(
        effects.ordered_effects.filter_not_in(closed_jaxpr.effects))
    with dispatch.log_elapsed_time(
        "Finished jaxpr to MLIR module conversion {fun_name} in {elapsed_time} sec",
        fun_name=str(name_stack), event=dispatch.JAXPR_TO_MLIR_MODULE_EVENT):
      lowering_result = mlir.lower_jaxpr_to_module(
          module_name,
          closed_jaxpr,
          ordered_effects=ordered_effects,
          backend_or_name=backend,
          platforms=lowering_parameters.platforms or (backend.platform,),
          axis_context=sharding_impls.ReplicaAxisContext(axis_env),
          name_stack=name_stack,
          donated_args=donated_invars,
          replicated_args=replicated_args,
          arg_shardings=None,
          result_shardings=None,
          arg_names=jaxpr.debug_info and jaxpr.debug_info.arg_names,
          result_names=jaxpr.debug_info and jaxpr.debug_info.result_paths,
          num_replicas=replicas.num_global_replicas,
          lowering_parameters=lowering_parameters)
  return PmapComputation(lowering_result.module, pci=pci, replicas=replicas,
                         shards=shards, tuple_args=tuple_args,
                         unordered_effects=unordered_effects,
                         ordered_effects=ordered_effects,
                         keepalive=lowering_result.keepalive,
                         host_callbacks=lowering_result.host_callbacks,
                         jaxpr_debug_info=closed_jaxpr.jaxpr.debug_info)


def _pmap_unmap_shaped_array(
    size: int, axis_name: core.AxisName, axis: int | None, aval: ShapedArray
  ) -> ShapedArray:
  named_shape = dict(aval.named_shape)
  named_shape.pop(axis_name, None)  # TODO: make this mandatory
  if axis is None: return aval.update(named_shape=named_shape)
  elif type(axis) is int:
    return ShapedArray(tuple_update(aval.shape, axis, size), aval.dtype,
                       named_shape=named_shape, weak_type=aval.weak_type)
  else: raise TypeError(axis)


AvalMapHandlerPair = tuple[Any, Callable]
_pmap_aval_mapping_handlers: dict[type, AvalMapHandlerPair] = {
    ShapedArray:   (Any, _pmap_unmap_shaped_array),
}

def _pmap_unmapped_aval(size: core.AxisSize, axis_name, axis: int | None,
                       aval: core.AbstractValue) -> core.AbstractValue:
  if not config.pmap_no_rank_reduction.value:
    return core.unmapped_aval(size, axis_name, axis, aval)

  _, handler = _pmap_aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis_name, axis, aval)
  else:
    raise TypeError(f"no unmapping handler for {aval} of type {type(aval)}")


class PmapComputation(stages.XlaLowering):
  _hlo: ir.Module
  _executable: PmapExecutable | None

  def __init__(self, hlo: ir.Module, **compile_args):
    self._executable = None
    self._hlo = hlo
    self.compile_args = compile_args

  # -- stages.XlaLowering overrides

  def stablehlo(self) -> ir.Module:
    return self._hlo

  @profiler.annotate_function
  def compile(self, compiler_options=None) -> PmapExecutable:
    if self._executable is None or compiler_options is not None:
      executable = UnloadedPmapExecutable.from_hlo(
          self._hlo, **self.compile_args,
          compiler_options=compiler_options)
      if compiler_options is None:
        self._executable = executable
      return executable
    return self._executable

def _cast_to_shaped_array(aval: core.AbstractValue) -> ShapedArray:
  assert isinstance(aval, ShapedArray), aval
  return cast(ShapedArray, aval)

@dataclasses.dataclass
class UnloadedPmapExecutable:
  compiled: Any
  backend: xb.XlaBackend
  local_input_avals: Sequence[core.AbstractValue]
  input_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  local_output_avals: Sequence[ShapedArray]
  output_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  unordered_effects: list[core.Effect]
  ordered_effects: list[core.Effect]
  keepalive: Sequence[Any]
  host_callbacks: Sequence[Any]
  jaxpr_debug_info: core.JaxprDebugInfo

  def build_execute_fun(self):
    input_indices = []
    for aval, spec in safe_zip(self.local_input_avals, self.input_shardings):
      assert isinstance(spec, sharding_impls.PmapSharding), spec
      assert isinstance(aval, core.ShapedArray), aval
      input_indices.append(
          sharding_specs.spec_to_indices(aval.shape, spec.sharding_spec)
          if spec.sharding_spec is not None else None)
    handle_outs = local_avals_to_results_handler(self.local_output_avals,
                                                 self.output_shardings)
    handle_args = InputsHandler(self.input_shardings,
                                self.compiled.local_devices(), input_indices)
    execute_fun = ExecuteReplicated(self.compiled, "parallel computation",
                                    self.backend, handle_args, handle_outs,
                                    self.unordered_effects,
                                    self.ordered_effects, self.keepalive,
                                    bool(self.host_callbacks),
                                    set(range(len(input_indices))))
    return execute_fun

  def load(self) -> PmapExecutable:
    fingerprint = getattr(self.compiled, "fingerprint", None)

    return PmapExecutable(
        self.compiled, self.build_execute_fun, fingerprint,
        self.local_input_avals, self.jaxpr_debug_info, self)

  @staticmethod
  def from_hlo(hlo: ir.Module,
               pci: ParallelCallableInfo,
               replicas: ReplicaInfo,
               shards: ShardInfo,
               tuple_args: bool,
               unordered_effects: list[core.Effect],
               ordered_effects: list[core.Effect],
               host_callbacks: list[Any],
               keepalive: Any,
               jaxpr_debug_info: core.JaxprDebugInfo,
               compiler_options=None):
    devices = pci.devices
    if devices is None:
      if shards.num_global_shards > xb.device_count(pci.backend):
        msg = ("compiling computation that requires {} logical devices, but only {} XLA "
               "devices are available (num_replicas={})")
        raise ValueError(msg.format(shards.num_global_shards,
                                    xb.device_count(pci.backend),
                                    replicas.num_global_replicas))
      # On a single host, we simply grab the first N devices from jax.devices().
      # In the single host case, we want the default device order of pmap to
      # match jax.devices().
      # On multiple hosts, we create a default device assignment that ensures
      # each host is responsible for a contiguous set of replicas.
      if shards.num_global_shards > shards.num_local_shards:
        # TODO(skye): use a locality-aware assignment that satisfies the above
        # constraint.
        devices = [d for process_index in range(xb.process_count(pci.backend))
                  for d in xb.local_devices(process_index, pci.backend)]
      else:
        devices = xb.local_devices(backend=pci.backend)[:shards.num_local_shards]
    else:
      if shards.num_local_shards != len(pci.local_devices):
        local_devices_str = ", ".join(map(str, pci.local_devices))
        if shards.num_local_shards == pci.axis_size:
          raise ValueError(
              f"Leading axis size of input to pmapped function must equal the "
              f"number of local devices passed to pmap. Got axis_size="
              f"{pci.axis_size}, num_local_devices={len(pci.local_devices)}.\n"
              f"(Local devices available to pmap: {local_devices_str})")
        else:
          raise ValueError(
              f"pmapped function requires {shards.num_local_shards} local "
              f"devices to run due to nested pmapped or other parallel "
              f"functions, but only {len(pci.local_devices)} are available.\n"
              f"(outer axis size: {pci.axis_size}, local devices available to "
              f"pmap: {local_devices_str})")
      if shards.num_global_shards != len(devices):
        raise ValueError("compiling computation that creates %s shards, "
                        "but %s devices were specified" %
                        (shards.num_global_shards, len(devices)))

    # 'devices' may be 1D or 2D at this point (e.g.
    # get_default_device_assignment() returns 2D assignment, caller may have
    # provided 1D list of devices).
    # Convert to 2D in case it's 1D and we have > 1 partitions.
    num_partitions = 1
    device_assignment: np.ndarray = np.array(devices).reshape(
        (replicas.num_global_replicas, num_partitions))
    compile_options = compiler.get_compile_options(
        num_replicas=replicas.num_global_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=False,
        env_options_overrides=compiler_options,
        detailed_logging=compiler.use_detailed_logging(hlo),
        backend=pci.backend,
    )
    compile_options.parameter_is_tupled_arguments = tuple_args

    process_index = xb.process_index(pci.backend)
    local_device_assignment = np.array([
        d for d in device_assignment.flat if d.process_index == process_index
    ])

    input_sharding_specs = [
        sharding_specs.pmap_sharding_spec(
            replicas.num_local_replicas, pci.axis_size,
            cast(ShapedArray, aval).shape, in_axis)
        for aval, in_axis in safe_zip(shards.sharded_avals, pci.in_axes)]
    in_shardings = _get_pmap_sharding(local_device_assignment,
                                      input_sharding_specs)

    local_unmapped_avals = [
        _cast_to_shaped_array(
            _pmap_unmapped_aval(pci.axis_size, pci.axis_name, out_axis, aval))
        if out_axis is not None else aval
        for aval, out_axis in safe_zip(shards.out_sharded_avals, pci.out_axes)]
    out_specs = [
        sharding_specs.pmap_sharding_spec(
            replicas.num_local_replicas, pci.axis_size, aval.shape, out_axis)
        for aval, out_axis in safe_zip(
            shards.out_sharded_avals, pci.out_axes)]
    out_shardings = _get_pmap_sharding(local_device_assignment, out_specs)

    if hasattr(pci.backend, "compile_replicated"):
      input_indices = [
          sharding_specs.spec_to_indices(aval.shape, spec)
          if spec is not None else None
          for aval, spec in safe_zip(pci.avals, input_sharding_specs)
      ]
      handle_outs = local_avals_to_results_handler(local_unmapped_avals,
                                                   out_shardings)
      return _compile_replicated_pmap_executable_from_hlo(
          hlo, pci, input_indices, in_shardings, handle_outs,
          compile_options, host_callbacks, bool(unordered_effects),
          ordered_effects, jaxpr_debug_info)

    with dispatch.log_elapsed_time(
        "Finished XLA compilation of {fun_name} in {elapsed_time} sec",
        fun_name=pci.name, event=dispatch.BACKEND_COMPILE_EVENT):
      compiled = compiler.compile_or_get_cached(
          pci.backend, hlo, device_assignment, compile_options,
          host_callbacks)

    return UnloadedPmapExecutable(
        compiled=compiled,
        backend=pci.backend,
        local_input_avals=pci.avals,
        input_shardings=in_shardings,
        local_output_avals=local_unmapped_avals,
        output_shardings=out_shardings,
        unordered_effects=unordered_effects,
        ordered_effects=ordered_effects,
        keepalive=keepalive,
        host_callbacks=host_callbacks,
        jaxpr_debug_info=jaxpr_debug_info).load()


def _compile_replicated_pmap_executable_from_hlo(
    hlo: ir.Module, pci, input_indices, in_shardings, handle_outs,
    compile_options, host_callbacks, has_unordered_effects, ordered_effects,
    jaxpr_debug_info):
  # Use the standard out_handler.
  execute_fun = pci.backend.compile_replicated(
      is_trivial=False, name=pci.name, computation=hlo,
      compile_options=compile_options, host_callbacks=host_callbacks,
      has_unordered_effects=has_unordered_effects,
      ordered_effects=ordered_effects, in_avals=pci.avals,
      in_indices=input_indices, in_shardings=in_shardings,
      kept_var_idx=set(range(len(pci.avals))), out_handler=handle_outs)
  # TODO(frostig): need `compile_replicated` to give us the XLA executable
  return PmapExecutable(None, lambda: execute_fun, None, pci.avals,
                        jaxpr_debug_info, None)


class PmapExecutable(stages.XlaExecutable):
  __slots__ = ["xla_executable", "_unsafe_call", "build_unsafe_call",
               "fingerprint", "in_avals", "_jaxpr_debug_info",
               "_unloaded_executable"]

  def __init__(self, xla_executable, build_unsafe_call, fingerprint,
               in_avals, jaxpr_debug_info, unloaded_executable):
    self.xla_executable = xla_executable
    self._unsafe_call = None
    self.build_unsafe_call = build_unsafe_call
    self.fingerprint = fingerprint
    self.in_avals = in_avals
    self._jaxpr_debug_info = jaxpr_debug_info
    self._unloaded_executable = unloaded_executable

  @property
  def unsafe_call(self) -> Callable[..., Any]:
    if self._unsafe_call is None:
      self._unsafe_call = self.build_unsafe_call()
    return self._unsafe_call

  # -- stages.XlaExecutable overrides

  def xla_extension_executable(self):
    return self.xla_executable

  @profiler.annotate_function
  def call(self, *args):
    # TODO(frostig): do we need to check sharding and sharded avals?
    arg_avals = map(xla.abstractify, args)
    check_arg_avals_for_call(self.in_avals, arg_avals, self._jaxpr_debug_info)
    return self.unsafe_call(*args)  # pylint: disable=not-callable


def _get_pmap_sharding(devices, specs):
  return [sharding_impls.PmapSharding(devices, spec) for spec in specs]


class InputsHandler:
  __slots__ = ("handler", "local_devices", "in_shardings", "input_indices")

  def __init__(self, in_shardings, local_devices=None, input_indices=None):
    self.handler = partial(shard_args, in_shardings)
    self.local_devices = local_devices
    self.in_shardings = in_shardings
    self.input_indices = input_indices

  def __call__(self, input_buffers):
    return self.handler(input_buffers)

  def __str__(self):
    return ("InputsHandler(\n"
            f"local_devices={self.local_devices},\n"
            f"in_shardings={self.in_shardings},\n"
            f"input_indices={self.input_indices})")


class ResultsHandler:
  # `out_avals` is the `Array` global avals when using pjit or xmap
  # with `config.parallel_functions_output_gda=True`. It is the local one
  # otherwise, and also when using `pmap`.
  __slots__ = ("handlers", "out_shardings", "out_avals")

  def __init__(self, handlers, out_shardings, out_avals):
    self.handlers = handlers
    self.out_shardings = out_shardings
    self.out_avals = out_avals

  def __call__(self, out_bufs):
    return [h(bufs) for h, bufs in safe_zip(self.handlers, out_bufs)]


def local_avals_to_results_handler(
    unmapped_local_out_avals: Sequence[ShapedArray],
    local_shardings: Sequence[sharding_impls.XLACompatibleSharding]) -> ResultsHandler:
  out_indices = [tuple(s.devices_indices_map(aval.shape).values())
                 for s, aval in safe_zip(local_shardings, unmapped_local_out_avals)]
  handlers = [
      local_aval_to_result_handler(aval, s, idcs)
      for aval, s, idcs in safe_zip(unmapped_local_out_avals, local_shardings, out_indices)
  ]
  return ResultsHandler(handlers, local_shardings, unmapped_local_out_avals)


def global_avals_to_results_handler(
    global_out_avals: Sequence[ShapedArray],
    shardings: Sequence[sharding_impls.XLACompatibleSharding],
    committed: bool,
    are_out_shardings_from_xla: Sequence[bool]) -> ResultsHandler:
  handlers = [
      global_aval_to_result_handler(global_aval, s, committed, x)
      for global_aval, s, x in safe_zip(global_out_avals, shardings,
                                        are_out_shardings_from_xla)
  ]
  return ResultsHandler(handlers, shardings, global_out_avals)


class ExecuteReplicated:
  """The logic to shard inputs, execute a replicated model, returning outputs."""
  __slots__ = ['xla_executable', 'name', 'backend', 'in_handler', 'out_handler',
               'has_unordered_effects', 'ordered_effects', 'keepalive',
               'has_host_callbacks', '_local_devices', 'kept_var_idx',
               '__weakref__']

  def __init__(self, xla_executable, name, backend, in_handler: InputsHandler,
               out_handler: ResultsHandler,
               unordered_effects: list[core.Effect],
               ordered_effects: list[core.Effect], keepalive: Any,
               has_host_callbacks: bool, kept_var_idx: set[int]):
    self.xla_executable = xla_executable
    self.name = name
    self.backend = backend
    self.in_handler = in_handler
    self.out_handler = out_handler
    self.has_unordered_effects = bool(unordered_effects)
    self.ordered_effects = ordered_effects
    self._local_devices = self.xla_executable.local_devices()
    self.keepalive = keepalive
    self.has_host_callbacks = has_host_callbacks
    self.kept_var_idx = kept_var_idx

  def _add_tokens_to_inputs(self, input_bufs):
    if self.ordered_effects:
      tokens = [
        dispatch.runtime_tokens.get_token_input(eff, self._local_devices)
        for eff in self.ordered_effects]
      input_bufs = [*tokens, *input_bufs]
    return input_bufs

  def _handle_token_bufs(self, token_bufs, sharded_token):
    # token_bufs: Sequence[Sequence[tokenArray]], for each effect the returned
    # token buffer (as a singleton list).
    # sharded_token: ShardedToken, containing the RuntimeTokens for each device
    for i, device in enumerate(self._local_devices):
      dispatch.runtime_tokens.set_output_runtime_token(
          device, sharded_token.get_token(i))
    for eff, token_buf in zip(self.ordered_effects, token_bufs):
      dispatch.runtime_tokens.set_token_result(eff, token_buf[0])

  @profiler.annotate_function
  def __call__(self, *args):
    args = [x for i, x in enumerate(args) if i in self.kept_var_idx]
    input_bufs = self.in_handler(args)
    if (self.ordered_effects or self.has_unordered_effects
        or self.has_host_callbacks):
      input_bufs = self._add_tokens_to_inputs(input_bufs)
      results = self.xla_executable.execute_sharded(
          input_bufs, with_tokens=True
      )
      result_token_bufs = results.disassemble_prefix_into_single_device_arrays(
          len(self.ordered_effects))
      sharded_runtime_token = results.consume_token()
      self._handle_token_bufs(result_token_bufs, sharded_runtime_token)
    else:
      results = self.xla_executable.execute_sharded(input_bufs)
    if dispatch.needs_check_special():
      out_arrays = results.disassemble_into_single_device_arrays()
      for arrays in out_arrays:
        dispatch.check_special(self.name, arrays)
      return self.out_handler(out_arrays)
    return results.consume_with_handlers(self.out_handler.handlers)


xla_pmap_p = core.MapPrimitive('xla_pmap')
xla_pmap = xla_pmap_p.bind
xla_pmap_p.def_impl(xla_pmap_impl)

def _pmap_partial_eval_custom_params_updater(
    unks_in, inst_in, kept_outs_known, kept_outs_staged, num_res, params_known,
    params_staged):
  # prune inputs to jaxpr_known according to unks_in
  donated_invars_known, _ = partition_list(unks_in, params_known['donated_invars'])
  in_axes_known, _ = partition_list(unks_in, params_known['in_axes'])
  _, out_axes_known = partition_list(kept_outs_known, params_known['out_axes'])
  out_axes_known = out_axes_known + [0] * num_res
  new_params_known = dict(params_known, in_axes=tuple(in_axes_known),
                          out_axes=tuple(out_axes_known),
                          donated_invars=tuple(donated_invars_known))

  # added num_res new inputs to jaxpr_staged, pruning according to inst_in
  _, donated_invars_staged = partition_list(inst_in, params_staged['donated_invars'])
  donated_invars_staged = [False] * num_res + donated_invars_staged
  _, in_axes_staged = partition_list(inst_in, params_staged['in_axes'])
  in_axes_staged = [0] * num_res + in_axes_staged
  _, out_axes_staged = partition_list(kept_outs_staged, params_staged['out_axes'])
  new_params_staged = dict(params_staged, in_axes=tuple(in_axes_staged),
                           out_axes=tuple(out_axes_staged),
                           donated_invars=tuple(donated_invars_staged))
  return new_params_known, new_params_staged

def _pmap_partial_eval_custom_res_maker(params_known, aval):
  return core.unmapped_aval(params_known['axis_size'], core.no_axis_name, 0, aval)

def _pmap_dce_rule(used_outputs, eqn):
  # just like pe.dce_jaxpr_call_rule, except handles in_axes / out_axes
  with maybe_extend_axis_env(eqn.params['axis_name'],
                             eqn.params['global_axis_size'], None):
    new_jaxpr, used_inputs = pe.dce_jaxpr(eqn.params['call_jaxpr'], used_outputs)
  _, donated_invars = partition_list(used_inputs, eqn.params['donated_invars'])
  _, in_axes = partition_list(used_inputs, eqn.params['in_axes'])
  _, out_axes = partition_list(used_outputs, eqn.params['out_axes'])
  new_params = dict(eqn.params, call_jaxpr=new_jaxpr,
                    donated_invars=tuple(donated_invars),
                    in_axes=tuple(in_axes), out_axes=tuple(out_axes))
  if not any(used_inputs) and not any(used_outputs) and not new_jaxpr.effects:
    return used_inputs, None
  else:
    new_eqn = pe.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, new_jaxpr.effects, eqn.source_info)
    return used_inputs, new_eqn


def _xla_call_partial_eval_update_params(
    params: core.ParamDict, kept_inputs: Sequence[bool], num_new_inputs: int
  ) -> core.ParamDict:
  donated_invars = params['donated_invars']
  if not kept_inputs and donated_invars:
    # JaxprTrace.post_process_call creates a call with no input tracers
    donated_invars = (False,) * num_new_inputs
  else:
    assert len(kept_inputs) == len(donated_invars)
    # JaxprTrace.process_call drops known input tracers
    donated_invars = [d for d, kept in zip(donated_invars, kept_inputs) if kept]
    # Any new inputs are prepended to the left, so mark those as not donated.
    donated_invars = [False] * num_new_inputs + donated_invars
  return dict(params, donated_invars=tuple(donated_invars))

def xla_call_jvp_update_params(params, nz_tangents):
  donated_invars = params['donated_invars']
  donated_tangents = [d for d, nz in zip(donated_invars, nz_tangents) if nz]
  new_donated_invars = (*donated_invars, *donated_tangents)
  return dict(params, donated_invars=new_donated_invars)

def _xla_call_transpose_update_params(params, undef_primals, nonzero_cts):
  donated_invars = params['donated_invars']
  donated_primals = [d for d, u in zip(donated_invars, undef_primals) if not u]
  donated_cotangents = [False for nz in nonzero_cts if nz]
  return dict(params, donated_invars=(*donated_primals, *donated_cotangents))


# Set param update handlers to update `donated_invars` just like xla_call_p
pe.call_param_updaters[xla_pmap_p] = _xla_call_partial_eval_update_params
pe.partial_eval_jaxpr_custom_rules[xla_pmap_p] = \
    partial(pe.call_partial_eval_custom_rule,
            'call_jaxpr', _pmap_partial_eval_custom_params_updater,
            res_aval=_pmap_partial_eval_custom_res_maker)
pe.dce_rules[xla_pmap_p] = _pmap_dce_rule
ad.call_param_updaters[xla_pmap_p] = xla_call_jvp_update_params
ad.call_transpose_param_updaters[xla_pmap_p] = _xla_call_transpose_update_params

ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)

def _pmap_axis_subst(params, subst, traverse):
  if 'call_jaxpr' not in params:
    return params
  if not traverse:
    return params
  def shadowed_subst(name):
    return (name,) if name in params['axis_name'] else subst(name)
  with maybe_extend_axis_env(params['axis_name'],
                             params['global_axis_size'], None):
    new_jaxpr = core.subst_axis_names_jaxpr(params['call_jaxpr'],
                                            shadowed_subst)
  return dict(params, call_jaxpr=new_jaxpr)
core.axis_substitution_rules[xla_pmap_p] = _pmap_axis_subst


def _unravel_index_hlo(axis_env):
  div = mlir.ir_constant(
      np.array(axis_env.nreps // math.prod(axis_env.sizes), np.uint32))
  mod = mlir.ir_constant(np.array(axis_env.sizes[-1], np.uint32))
  return hlo.remainder(hlo.divide(hlo.replica_id(), div), mod)

def _hlo_shard(aval, axis_env, xs, in_axis):
  if aval is core.abstract_token:
    return xs
  elif isinstance(aval, core.ShapedArray):
    x, = xs
    dims = list(aval.shape)
    zero = mlir.ir_constant(np.zeros((), dtype=np.uint32))
    idxs = [zero] * len(dims)
    idxs.insert(in_axis, _unravel_index_hlo(axis_env))
    dims_unsqueezed = dims.copy()
    dims_unsqueezed.insert(in_axis, 1)
    dynamic_slice_result = hlo.dynamic_slice(
        x, idxs, mlir.dense_int_array(dims_unsqueezed))
    return [
      hlo.reshape(mlir.aval_to_ir_type(aval), dynamic_slice_result)
    ]
  else:
    raise TypeError(aval)


def _axis_read(axis_env, axis_name):
  try:
    return max(i for i, name in enumerate(axis_env.names) if name == axis_name)
  except ValueError:
    raise NameError(f"unbound axis name: {axis_name}") from None

def axis_groups(axis_env: sharding_impls.AxisEnv, name) -> tuple[tuple[int, ...]]:
  if not isinstance(name, (list, tuple)):
    name = (name,)
  mesh_axes = tuple(unsafe_map(partial(_axis_read, axis_env), name))
  trailing_size, ragged = divmod(axis_env.nreps, math.prod(axis_env.sizes))
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
  iota = np.arange(math.prod(mesh_spec)).reshape(mesh_spec)
  groups = np.reshape(
      np.moveaxis(iota, mesh_axes, np.arange(len(mesh_axes))),
      (math.prod(np.take(mesh_spec, mesh_axes)), -1))
  return tuple(unsafe_map(tuple, groups.T))


# TODO(b/110096942): more efficient gather
def _hlo_unshard(ctx: mlir.LoweringRuleContext, aval, axis_env, out_axis, xs):
  if aval is core.abstract_token:
    return xs
  elif isinstance(aval, core.ShapedArray):
    x, = xs
    dims = list(aval.shape)
    padded_aval = aval.update(shape=[axis_env.sizes[-1]] + dims)
    padded = mlir.full_like_aval(ctx, 0, padded_aval)
    zero = mlir.ir_constant(np.zeros((), dtype=np.uint32))
    idxs = [_unravel_index_hlo(axis_env)] + [zero] * len(dims)
    broadcast_result = hlo.broadcast(x, mlir.dense_int_array([1]))
    padded = hlo.dynamic_update_slice(padded, broadcast_result, idxs)
    replica_groups = mlir.dense_int_elements(
      axis_groups(axis_env, axis_env.names[-1]))
    out = hlo.cross_replica_sum(padded, replica_groups)
    if out_axis != 0:
      # TODO(apaszke,mattjj): Change the indices to DynamicUpdateSlice instead
      perm = list(range(1, len(dims)))
      perm.insert(out_axis, 0)
      transposed_dims = list(dims)
      transposed_dims.insert(out_axis, axis_env.sizes[-1])
      out = hlo.transpose(out, mlir.dense_int_array(perm))

    return out
  else:
    raise TypeError(aval)

def _extend_axis_env(env: sharding_impls.AxisEnv, name, size: int):
  return sharding_impls.AxisEnv(env.nreps, env.names + (name,),
                                env.sizes + (size,))


def _pmap_lowering(ctx, *in_nodes, axis_name,
                   axis_size, global_axis_size, devices, name,
                   call_jaxpr, backend=None, in_axes, out_axes,
                   donated_invars, is_explicit_global_axis_size):
  del donated_invars  # Unused.
  mlir.check_backend_matches(backend, ctx.module_context.platforms)
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if ctx.module_context.axis_env.names and devices is not None:
    raise ValueError("Nested pmap with explicit devices argument.")
  new_env = _extend_axis_env(ctx.module_context.axis_env, axis_name,
                             global_axis_size)
  # Shard the in_nodes that are mapped
  in_avals = [v.aval for v in call_jaxpr.invars]
  in_nodes_sharded = (
    _hlo_shard(aval, new_env, mlir.wrap_singleton_ir_values(in_node), in_axis)
    if in_axis is not None else mlir.wrap_singleton_ir_values(in_node)
    for aval, in_node, in_axis in zip(in_avals, in_nodes, in_axes))

  with maybe_extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
    sub_ctx = ctx.module_context.replace(
        axis_context=sharding_impls.ReplicaAxisContext(new_env))
    sharded_outs, _ = mlir.jaxpr_subcomp(
        sub_ctx, call_jaxpr,
        ctx.name_stack.extend(util.wrap_name(name, 'pmap')),
        mlir.TokenSet(), (), *in_nodes_sharded,
        dim_var_values=ctx.dim_var_values)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_hlo_unshard(ctx, aval, new_env, out_axis, shard)
          for aval, out_axis, shard in zip(out_avals, out_axes, sharded_outs)]
  return outs

mlir.register_lowering(xla_pmap_p, _pmap_lowering)


# ------------------- xmap -------------------

def tile_aval_nd(axis_sizes, in_axes: ArrayMapping, aval):
  assert isinstance(aval, ShapedArray)
  shape = list(aval.shape)
  named_shape = dict(aval.named_shape)
  for name, axis in in_axes.items():
    assert shape[axis] % axis_sizes[name] == 0
    assert name not in named_shape
    named_shape[name] = axis_sizes[name]
    shape[axis] //= axis_sizes[name]
  return aval.update(shape=tuple(shape), named_shape=named_shape)

def untile_aval_nd(axis_sizes, out_axes: ArrayMapping, aval):
  assert isinstance(aval, ShapedArray)
  shape = list(aval.shape)
  named_shape = dict(aval.named_shape)
  for name, axis in out_axes.items():
    shape[axis] *= axis_sizes[name]
    named_shape.pop(name, None)  # The name might be missing --- it's a broadcast.
  return aval.update(shape=tuple(shape), named_shape=named_shape)


def mesh_local_to_global(mesh, axes: ArrayMapping, aval):
  return untile_aval_nd(mesh.shape, axes,
                        tile_aval_nd(mesh.local_mesh.shape, axes, aval))

def mesh_global_to_local(mesh, axes: ArrayMapping, aval):
  return untile_aval_nd(mesh.local_mesh.shape, axes,
                        tile_aval_nd(mesh.shape, axes, aval))


class SPMDBatchTrace(batching.BatchTrace):
  def get_axis_primitive_batcher(self, primitive, frame):
    if primitive in spmd_primitive_batchers:
      return partial(spmd_primitive_batchers[primitive],
          frame.size, frame.name, frame.main_trace.trace_type)
    return super().get_axis_primitive_batcher(primitive, frame)


spmd_primitive_batchers: dict[core.Primitive, Callable] = {}


def vtile_by_mesh(fun: lu.WrappedFun,
                  mesh: Mesh,
                  in_axes: Sequence[ArrayMapping],
                  out_axes: Sequence[ArrayMapping]):
  # We vectorize in reversed order, because vmap is often biased towards
  # moving the batch axis to the front, and this way of stacking transforms
  # will order the batch axes according to the mesh axis order.
  # Not strictly necessary, but seems nicer than reversing it?
  for name, size in reversed(mesh.shape.items()):
    fun = batching.vtile(fun,
                         tuple(a.get(name, None) for a in in_axes),
                         tuple(a.get(name, None) for a in out_axes),
                         tile_size=size,
                         axis_name=name,
                         main_type=SPMDBatchTrace)
  return fun

full_to_shard_p = core.Primitive('full_to_shard')

@full_to_shard_p.def_abstract_eval
def _full_to_shard_abstract_eval(x, axes, mesh, **_):
  # TODO: Assert x is a global aval! Or ideally check that it's global in dims from axes!
  return tile_aval_nd(mesh.shape, axes, x)

def manual_proto(
    aval: core.ShapedArray,
    manual_axes_set: frozenset[sharding_impls.MeshAxisName], mesh: Mesh):
  """Create an OpSharding proto that declares all mesh axes from `axes` as manual
  and all others as replicated.
  """
  named_mesh_shape = mesh.shape
  mesh_shape = list(named_mesh_shape.values())
  axis_order = {axis: i for i, axis in enumerate(mesh.axis_names)}

  manual_axes = sorted(manual_axes_set, key=str)
  replicated_axes = [axis for axis in mesh.axis_names
                     if axis not in manual_axes_set]

  tad_perm = ([axis_order[a] for a in replicated_axes] +
              [axis_order[a] for a in manual_axes])
  tad_shape = [1] * aval.ndim
  tad_shape.append(math.prod([named_mesh_shape[a] for a in replicated_axes]))
  tad_shape.append(math.prod([named_mesh_shape[a] for a in manual_axes]))

  raw_mesh = np.arange(math.prod(mesh_shape)).reshape(mesh_shape)
  proto = xc.OpSharding()
  proto.type = xc.OpSharding.Type.OTHER
  proto.tile_assignment_dimensions = tad_shape
  proto.tile_assignment_devices = list(raw_mesh.transpose(tad_perm).reshape(tad_shape).flat)
  proto.last_tile_dims = [xc.OpSharding.Type.REPLICATED, xc.OpSharding.Type.MANUAL]
  return proto

@partial(mlir.register_lowering, full_to_shard_p)
def _full_to_shard_lowering(ctx, x, *, axes: ArrayMapping, mesh: Mesh,
                            manual_axes: frozenset[sharding_impls.MeshAxisName]):
  # TODO: Can we short-circuit for replicated values? Probably not.
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  sharding_proto = (
      sharding_impls.NamedSharding(mesh, array_mapping_to_axis_resources(axes))
      ._to_xla_hlo_sharding(aval_in.ndim).to_proto())
  unspecified_dims = set(range(aval_in.ndim)) - set(axes.values())
  sx = mlir.wrap_with_sharding_op(ctx, x, aval_in, sharding_proto,
                                  unspecified_dims=unspecified_dims)
  proto = manual_proto(aval_in, manual_axes, mesh)
  return (mlir.wrap_with_full_to_shard_op(ctx, sx, aval_out, proto,
                                          unspecified_dims=unspecified_dims),)

shard_to_full_p = core.Primitive('shard_to_full')

@shard_to_full_p.def_abstract_eval
def _shard_to_full_abstract_eval(x, axes, mesh, **_):
  # TODO: Assert x is a global aval! Or ideally check that it's global in dims from axes!
  return untile_aval_nd(mesh.shape, axes, x)

@partial(mlir.register_lowering, shard_to_full_p)
def _shard_to_full_lowering(ctx: mlir.LoweringRuleContext, x, *, axes: ArrayMapping, mesh: Mesh,
                            manual_axes: frozenset[sharding_impls.MeshAxisName]):
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  proto = manual_proto(aval_in, manual_axes, mesh)  # type: ignore
  unspecified_dims = set(range(aval_in.ndim)) - set(axes.values())  # type: ignore
  sx = mlir.wrap_with_sharding_op(ctx, x, aval_in, proto,
                                  unspecified_dims=unspecified_dims)
  sharding_proto = (
      sharding_impls.NamedSharding(mesh, array_mapping_to_axis_resources(axes))
      ._to_xla_hlo_sharding(aval_out.ndim).to_proto())
  return (mlir.wrap_with_shard_to_full_op(ctx, sx, aval_out, sharding_proto,
                                          unspecified_dims),)

@lu.transformation
def vtile_manual(manual_axes: frozenset[sharding_impls.MeshAxisName],
                 mesh: Mesh,
                 in_axes: Sequence[ArrayMapping],
                 out_axes: Sequence[ArrayMapping],
                 *args):
  tiled_args = [full_to_shard_p.bind(arg, axes=axes, mesh=mesh, manual_axes=manual_axes)
                for arg, axes in zip(args, in_axes)]
  tiled_outs = yield tiled_args, {}
  outs = [shard_to_full_p.bind(out, axes=axes, mesh=mesh, manual_axes=manual_axes)
          for out, axes in zip(tiled_outs, out_axes)]
  yield outs


@dataclasses.dataclass(frozen=True)
class TileVectorize:
  pass

@dataclasses.dataclass(frozen=True)
class TileManual:
  manual_axes: frozenset[sharding_impls.MeshAxisName]

TilingMethod = Union[TileVectorize, TileManual]


def check_if_any_auto(
    shardings: Iterable[(sharding_impls.XLACompatibleSharding |
                              AUTO | UnspecifiedValue)]) -> bool:
  for s in shardings:
    if is_auto(s):
      return True
  return False

class MismatchType(enum.Enum):
  ARG_SHARDING = 0
  OUT_SHARDING = 1
  SHARDING_INSIDE_COMPUTATION = 2
  CONTEXT_DEVICES = 3
  IN_SHARDING = 4

  def __str__(self):
    if self.name == 'IN_SHARDING':
      return 'explicit input sharding'
    elif self.name == 'OUT_SHARDING':
      return 'explicit output sharding'
    elif self.name == 'CONTEXT_DEVICES':
      return 'devices'
    return f'{self.name}'


@dataclasses.dataclass
class DeviceAssignmentMismatch:
  da: Sequence[xc.Device]
  m_type: MismatchType
  source_info: dispatch.SourceInfo | None

  @property
  def device_ids(self) -> Sequence[int]:
    return [d.id for d in self.da]

  @property
  def platform(self) -> str:
    return self.da[0].platform.upper()

  def _maybe_api_name(self, api_name) -> str:
    return f" {api_name}'s" if self.m_type == MismatchType.CONTEXT_DEVICES else ""

  @property
  def source_info_str(self):
    return (
        "" if self.source_info is None
        else f" at {source_info_util.summarize(self.source_info.source_info)}"
    )

  @property
  def _dev_ids_plat_str(self):
    return f"device ids {self.device_ids} on platform {self.platform}"

  def m_type_str(self, api_name):
    return (f'{self.source_info and self.source_info.eqn_name} inside {api_name}'
            if self.m_type == MismatchType.SHARDING_INSIDE_COMPUTATION else self.m_type)

  def _str(self, api_name):
    return (f"{self._maybe_api_name(api_name)} {self.m_type_str(api_name)} with "
            f"{self._dev_ids_plat_str}{self.source_info_str}")


class DeviceAssignmentMismatchError(Exception):
  pass


ShardingInfo = tuple[
    Union[sharding_impls.XLACompatibleSharding, UnspecifiedValue, AUTO],
    MismatchType,
    Union[Any, None],  # Any is dispatch.SourceInfo to avoid circular imports
]


def _get_default_device() -> xc.Device:
  return config.default_device.value or xb.local_devices()[0]


class _thread_local_decorator(threading.local):

  def __init__(self, fn):
    self.fn = fn

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)


@_thread_local_decorator
def _get_and_check_device_assignment(
    shardings: Iterable[ShardingInfo],
    devices: Sequence[xc.Device] | None,
) -> tuple[xc.Client, tuple[xc.Device, ...]]:
  first_sharding_info = None
  if devices is None:
    devices = ()
  else:
    devices = tuple(devices)

  for i, s_type, source_info in shardings:
    if is_unspecified(i):
      continue

    if first_sharding_info is None:
      first_sharding_info = (
          (i.mesh._flat_devices_tuple, s_type, source_info) if is_auto(i)  # type: ignore
          else (i._device_assignment, s_type, source_info))  # type: ignore
    arr_device_assignment = i.mesh._flat_devices_tuple if is_auto(i) else i._device_assignment  # type: ignore
    if not devices:
      if first_sharding_info[0] != arr_device_assignment:
        raise DeviceAssignmentMismatchError([
            DeviceAssignmentMismatch(*first_sharding_info),
            DeviceAssignmentMismatch(arr_device_assignment, s_type, source_info)])
    else:
      if devices != arr_device_assignment:
        raise DeviceAssignmentMismatchError([
            DeviceAssignmentMismatch(devices, MismatchType.CONTEXT_DEVICES, None),
            DeviceAssignmentMismatch(arr_device_assignment, s_type, source_info)])
  if first_sharding_info is None and devices:
    final_device_assignment = devices
  elif first_sharding_info is None:
    final_device_assignment = (_get_default_device(),)
  else:
    final_device_assignment = first_sharding_info[0]
  return xb.get_device_backend(final_device_assignment[0]), final_device_assignment

MaybeSharding = Union[sharding_impls.XLACompatibleSharding, UnspecifiedValue]


def prune_unused_inputs(
    jaxpr: core.Jaxpr,
) -> tuple[core.Jaxpr, set[int], set[int]]:
  used_outputs = [True] * len(jaxpr.outvars)
  new_jaxpr, used_consts, used_inputs = pe.dce_jaxpr_consts(jaxpr, used_outputs)
  kept_const_idx = {i for i, b in enumerate(used_consts) if b}
  kept_var_idx = {i for i, b in enumerate(used_inputs) if b}
  return new_jaxpr, kept_const_idx, kept_var_idx


@weakref_lru_cache
def _dce_jaxpr(closed_jaxpr, global_in_avals, api_name, fun_name,
               keep_unused, donated_invars, auto_spmd_lowering):
  name_stack = source_info_util.new_name_stack(wrap_name(fun_name, api_name))

  assert isinstance(closed_jaxpr, core.ClosedJaxpr)
  jaxpr = closed_jaxpr.jaxpr
  global_out_avals = closed_jaxpr.out_avals
  consts = closed_jaxpr.consts

  if (keep_unused or auto_spmd_lowering or
      any(hasattr(a, "shape") and not core.is_constant_shape(a.shape)
          for a in global_in_avals)):
    kept_var_idx = set(range(len(global_in_avals)))
  else:
    jaxpr, kept_const_idx, kept_var_idx = prune_unused_inputs(jaxpr)
    consts = [c for i, c in enumerate(consts) if i in kept_const_idx]
    global_in_avals = tuple(a for i, a in enumerate(global_in_avals) if i in kept_var_idx)
    donated_invars = tuple(x for i, x in enumerate(donated_invars) if i in kept_var_idx)
    del kept_const_idx

  jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  return (closed_jaxpr, global_in_avals, tuple(global_out_avals), donated_invars,
          kept_var_idx, name_stack)


@dataclasses.dataclass(frozen=True)
class SemanticallyEqualShardings:
  shardings: tuple[sharding_impls.GSPMDSharding | UnspecifiedValue, ...]

  def __hash__(self):
    return hash(tuple(
        s._hlo_sharding_hash if isinstance(s, sharding_impls.GSPMDSharding) else s  # type: ignore
        for s in self.shardings))

  def __eq__(self, other):
    if not isinstance(other, SemanticallyEqualShardings):
      return False
    return all(
        (op_shardings.are_op_shardings_equal(s._hlo_sharding, o._hlo_sharding)
         and s.memory_kind == o.memory_kind)
        if (isinstance(s, sharding_impls.GSPMDSharding) and
            isinstance(o, sharding_impls.GSPMDSharding))
        else s == o
        for s, o in zip(self.shardings, other.shardings)
    )


def _raise_warnings_or_errors_for_jit_of_pmap(
    nreps: int, backend: xc.Client, name: str, jaxpr: core.Jaxpr) -> None:
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

  if xb.process_count() > 1 and (
      nreps > 1 or dispatch.jaxpr_has_primitive(jaxpr, "xla_pmap")
  ):
    raise NotImplementedError(
        "jit of multi-host pmap not implemented (and jit-of-pmap can cause "
        "extra data movement anyway, so maybe you don't want it after all).")


@weakref_lru_cache
def _cached_lowering_to_hlo(closed_jaxpr, api_name, fun_name, backend,
                            semantic_in_shardings, semantic_out_shardings,
                            in_layouts, out_layouts, num_devices, device_assignment,
                            donated_invars, name_stack, all_default_mem_kind,
                            lowering_parameters: mlir.LoweringParameters):
  jaxpr = closed_jaxpr.jaxpr
  in_shardings = semantic_in_shardings.shardings
  out_shardings = semantic_out_shardings.shardings
  global_in_avals = closed_jaxpr.in_avals
  global_out_avals = closed_jaxpr.out_avals

  log_priority = logging.WARNING if config.log_compiles.value else logging.DEBUG
  if logger.isEnabledFor(log_priority):
    logger.log(log_priority,
               "Compiling %s for with global shapes and types %s. "
               "Argument mapping: %s.",
               fun_name, global_in_avals, in_shardings)

  # Look at the number of replcas present in the jaxpr. In
  # lower_sharding_computation, nreps > 1 during `jit(pmap)` cases. This is
  # handled here so as to deprecate the lower_xla_callable codepath when
  # `jax.Array` is turned on by default.
  # TODO(yashkatariya): Remove this when `jit(pmap)` is removed.
  nreps = dispatch.jaxpr_replicas(jaxpr)
  _raise_warnings_or_errors_for_jit_of_pmap(nreps, backend, fun_name, jaxpr)

  in_mlir_shardings: list[sharding_impls.XLACompatibleSharding | None] | None
  out_mlir_shardings: list[sharding_impls.XLACompatibleSharding | None] | None
  axis_ctx: mlir.AxisContext

  if nreps == 1:
    in_mlir_shardings = map(_to_logical_sharding, global_in_avals, in_shardings)
    out_mlir_shardings = map(_to_logical_sharding, global_out_avals, out_shardings)
    replicated_args = [False] * len(global_in_avals)
    axis_ctx = sharding_impls.ShardingContext(num_devices, device_assignment)
    num_partitions = num_devices
  else:
    # This path is triggered for `jit(pmap)` cases.
    replicated_args = None
    in_mlir_shardings = None
    out_mlir_shardings = None
    axis_env = sharding_impls.AxisEnv(nreps, (), ())
    axis_ctx = sharding_impls.ReplicaAxisContext(axis_env)
    num_partitions = 1

  module_name = f"{api_name}_{fun_name}"

  if num_devices > 1:
    unsupported_effects = effects.ordered_effects.filter_in(closed_jaxpr.effects)
    unsupported_effects = effects.shardable_ordered_effects.filter_not_in(
        unsupported_effects)
    if len(unsupported_effects) > 0:
      raise ValueError(
        "The following ordered effects are not supported for "
        f"more than 1 device: {unsupported_effects}")
  ordered_effects = list(effects.ordered_effects.filter_in(closed_jaxpr.effects))

  with dispatch.log_elapsed_time(
        "Finished jaxpr to MLIR module conversion {fun_name} in {elapsed_time} sec",
        fun_name=str(name_stack), event=dispatch.JAXPR_TO_MLIR_MODULE_EVENT):
    lowering_result = mlir.lower_jaxpr_to_module(
        module_name,
        closed_jaxpr,
        ordered_effects=ordered_effects,
        backend_or_name=backend,
        # Optionally, override the lowering platform
        platforms=lowering_parameters.platforms or (backend.platform,),
        axis_context=axis_ctx,
        name_stack=name_stack,
        donated_args=donated_invars,
        replicated_args=replicated_args,
        arg_shardings=in_mlir_shardings,
        result_shardings=out_mlir_shardings,
        in_layouts=in_layouts,
        out_layouts=out_layouts,
        arg_names=jaxpr.debug_info and jaxpr.debug_info.arg_names,
        result_names=jaxpr.debug_info and jaxpr.debug_info.result_paths,
        num_replicas=nreps,
        num_partitions=num_partitions,
        all_default_mem_kind=all_default_mem_kind,
        lowering_parameters=lowering_parameters)
  tuple_args = dispatch.should_tuple_args(len(global_in_avals), backend.platform)
  unordered_effects = list(
      effects.ordered_effects.filter_not_in(closed_jaxpr.effects))
  return (lowering_result.module, lowering_result.keepalive,
          lowering_result.host_callbacks, unordered_effects, ordered_effects,
          nreps, tuple_args, lowering_result.shape_poly_state)


@lru_cache(maxsize=2048)
def _create_da_object(  # pytype: disable=invalid-annotation
    device_assignment: tuple[xc.Device, ...]) -> xc.DeviceList:  # type: ignore
  return xc.DeviceList(device_assignment)


def jaxpr_transfer_mem_kinds(
    jaxpr: core.Jaxpr) -> Iterator[sharding_impls.TransferToMemoryKind]:
  for eqn in jaxpr.eqns:
    if (eqn.primitive is dispatch.device_put_p and
        isinstance(eqn.params['device'], sharding_impls.TransferToMemoryKind)):
      yield eqn.params['device']
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from jaxpr_transfer_mem_kinds(subjaxpr)


def are_all_shardings_default_mem_kind(da_object, shardings):
  try:
    default_mem_kind = da_object.default_memory_kind
  except:
    return True
  for i in shardings:
    if is_unspecified_or_auto(i):
      continue
    if i.memory_kind != default_mem_kind:
      return False
  return True

MaybeLayout = Sequence[Union[XLACompatibleLayout, LayoutRequest, None]]


class AllArgsInfo(NamedTuple):
  """Avals, shardings, layouts and debug_info for all arguments prior to DCE."""
  in_avals: Sequence[core.ShapedArray]
  in_shardings: Any
  debug_info: core.JaxprDebugInfo | None


@profiler.annotate_function
def lower_sharding_computation(
    closed_jaxpr: core.ClosedJaxpr,
    api_name: str,
    fun_name: str,
    in_shardings: Sequence[MaybeSharding],
    out_shardings: Sequence[MaybeSharding],
    donated_invars: Sequence[bool],
    global_in_avals: Sequence[core.ShapedArray],
    *,
    keep_unused: bool,
    inline: bool,
    devices_from_context: Sequence[xc.Device] | None = None,
    lowering_parameters: mlir.LoweringParameters,
    in_layouts: MaybeLayout,
    out_layouts: MaybeLayout,
) -> MeshComputation:
  """Lowers a computation to XLA. It can take arbitrary shardings as input.

  The caller of this code can pass in a singleton UNSPECIFIED because the
  number of out_avals might not be known at that time and
  lower_sharding_computation calculates the number of out_avals so it can apply
  the singleton UNSPECIFIED to all out_avals.
  """
  # 1. Trace to jaxpr and preprocess/verify it
  auto_spmd_lowering = check_if_any_auto(
      it.chain.from_iterable([in_shardings, out_shardings]))  # type: ignore

  all_args_info = AllArgsInfo(global_in_avals, in_shardings,
                              closed_jaxpr.jaxpr.debug_info)

  (closed_jaxpr, global_in_avals, global_out_avals, donated_invars,
   kept_var_idx, name_stack) = _dce_jaxpr(
      closed_jaxpr, global_in_avals, api_name, fun_name, keep_unused,
      donated_invars, auto_spmd_lowering)
  jaxpr = closed_jaxpr.jaxpr
  in_shardings = tuple(s for i, s in enumerate(in_shardings) if i in kept_var_idx)
  in_layouts = tuple(l for i, l in enumerate(in_layouts) if i in kept_var_idx)

  assert len(out_shardings) == len(out_layouts) == len(global_out_avals), (
      len(out_shardings), len(out_layouts), len(global_out_avals))

  # Device assignment across all inputs, outputs and shardings inside jaxpr
  # should be the same.
  jaxpr_sharding = list(dispatch.jaxpr_shardings(jaxpr))
  backend, device_assignment = _get_and_check_device_assignment(
      it.chain(
          ((i, MismatchType.ARG_SHARDING, None) for i in util.stable_unique(in_shardings)),
          ((o, MismatchType.OUT_SHARDING, None) for o in util.stable_unique(out_shardings)),
          ((js, MismatchType.SHARDING_INSIDE_COMPUTATION, source_info)
           for js, source_info in util.stable_unique(jaxpr_sharding))),
      devices_from_context)

  # TODO(yashkatariya): Enable this when offload APIs are stable.
  # transfer_mem_kind_in_jaxpr = list(jaxpr_transfer_mem_kinds(jaxpr))

  committed = bool(
      devices_from_context or
      len(device_assignment) > 1 or
      any(not is_unspecified(i) for i in in_shardings) or
      any(not is_unspecified(js) for js, _ in jaxpr_sharding) or
      any(not is_unspecified(o) for o in out_shardings))

  gs = sharding_impls.GSPMDSharding.get_replicated(device_assignment)
  in_shardings = tuple(gs if is_unspecified(i) else i for i in in_shardings)

  da_object = _create_da_object(tuple(device_assignment))

  all_default_mem_kind = are_all_shardings_default_mem_kind(
      da_object,
      it.chain(in_shardings, out_shardings, [js for js, _ in jaxpr_sharding]))  # type: ignore

  if not da_object.is_fully_addressable:  # type: ignore
    if inline and config.spmd_mode.value != 'allow_all':
      raise RuntimeError(
          "Running operations on `Array`s that are not fully addressable by this "
          "process (i.e. `Array`s with data sharded across multiple devices and "
          "processes.) is dangerous. It’s very important that all processes run "
          "the same cross-process computations in the same order otherwise it "
          "can lead to hangs. "
          "If you’re not already familiar with JAX’s multi-process "
          "programming model, please read "
          "https://jax.readthedocs.io/en/latest/multi_process.html. "
          "To fix this error, run your `jitted` computation inside "
          "`with jax.spmd_mode('allow_all'):` context manager.")

  # 2. Build up the HLO
  semantic_in_shardings = SemanticallyEqualShardings(in_shardings)  # type: ignore
  semantic_out_shardings = SemanticallyEqualShardings(out_shardings)  # type: ignore
  prim_requires_devices = dispatch.jaxpr_has_prim_requiring_devices(jaxpr)

  (module, keepalive, host_callbacks, unordered_effects, ordered_effects,
   nreps, tuple_args, shape_poly_state) = _cached_lowering_to_hlo(
       closed_jaxpr, api_name, fun_name, backend, semantic_in_shardings,
       semantic_out_shardings, in_layouts, out_layouts, len(da_object),
       tuple(da_object) if prim_requires_devices else None, donated_invars,
       name_stack, all_default_mem_kind, lowering_parameters=lowering_parameters)

  # backend and device_assignment is passed through to MeshExecutable because
  # if keep_unused=False and all in_shardings are pruned, then there is no way
  # to get the device_assignment and backend. So pass it to MeshExecutable
  # because we calculate the device_assignment and backend before in_shardings,
  # etc are pruned.
  return MeshComputation(
      str(name_stack),
      module,
      donated_invars,
      global_in_avals=global_in_avals,
      global_out_avals=global_out_avals,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      spmd_lowering=True,
      tuple_args=tuple_args,
      auto_spmd_lowering=auto_spmd_lowering,
      unordered_effects=unordered_effects,
      ordered_effects=ordered_effects,
      host_callbacks=host_callbacks,
      keepalive=keepalive,
      kept_var_idx=kept_var_idx,
      backend=backend,
      device_assignment=da_object,
      committed=committed,
      in_layouts=in_layouts,
      out_layouts=out_layouts,
      pmap_nreps=nreps,
      shape_poly_state=shape_poly_state,
      all_default_mem_kind=all_default_mem_kind,
      all_args_info=all_args_info)


def _to_logical_sharding(
    aval: core.AbstractValue, sharding: MaybeSharding | AUTO
) -> sharding_impls.XLACompatibleSharding | None:
  if is_unspecified(sharding) or is_auto(sharding):
    return None
  elif isinstance(aval, (ShapedArray, DShapedArray)):
    assert isinstance(sharding, sharding_impls.XLACompatibleSharding)
    return sharding
  elif isinstance(aval, core.AbstractToken):
    return None
  else:
    raise TypeError(aval)


@profiler.annotate_function
def lower_mesh_computation(
    fun_or_jaxpr: lu.WrappedFun | core.ClosedJaxpr,
    api_name: str,
    fun_name: str,
    mesh: Mesh,
    in_shardings: Sequence[sharding_impls.NamedSharding | AUTO],
    out_shardings: Sequence[(sharding_impls.NamedSharding | AUTO |
                                  UnspecifiedValue)],
    donated_invars: Sequence[bool],
    spmd_lowering: bool,
    global_in_avals: Sequence[core.ShapedArray],
    tiling_method: TilingMethod | None,
    lowering_parameters: mlir.LoweringParameters) -> MeshComputation:
  assert not mesh.empty
  backend = xb.get_device_backend(mesh.devices.flat[0])
  name_stack = source_info_util.new_name_stack(wrap_name(fun_name, api_name))

  global_axis_sizes = mesh.shape

  log_priority = logging.WARNING if config.log_compiles.value else logging.DEBUG
  if logger.isEnabledFor(log_priority):
    logger.log(log_priority,
               "Compiling %s for %s mesh with global shapes and types %s. "
               "Argument mapping: %s.",
               fun_name, tuple(global_axis_sizes.items()), global_in_avals,
               in_shardings)

  # 1. Trace to jaxpr and preprocess/verify it
  if spmd_lowering:
    manual_axes: frozenset[MeshAxisName] = frozenset()
    # TODO: Consider handling xmap's 'vectorize' in here. We can vmap once instead of vtile twice!
    if tiling_method is not None:
      if isinstance(tiling_method, TileVectorize):
        tiling_transform = vtile_by_mesh
      elif isinstance(tiling_method, TileManual):
        tiling_transform = lambda f, *args: vtile_manual(f, tiling_method.manual_axes, *args)  # type: ignore
        manual_axes = tiling_method.manual_axes
      else:
        raise NotImplementedError(f"Unrecognized tiling method: {tiling_method}")
      assert not callable(out_shardings)
      assert isinstance(fun_or_jaxpr, lu.WrappedFun)
      # This is the xmap path where there is no `AUTO` or `UNSPECIFIED`, which
      # is why `.spec` can be accessed.
      fun_or_jaxpr = tiling_transform(
          fun_or_jaxpr, mesh, [get_array_mapping(i.spec) for i in in_shardings],  # type: ignore
          [get_array_mapping(o.spec) for o in out_shardings])  # type: ignore
    in_jaxpr_avals = global_in_avals
  else:
    assert isinstance(tiling_method, TileVectorize)
    # In non-spmd lowering path, there is no `AUTO` or `UNSPECIFIED`, which is
    # why `.spec` can be accessed.
    in_tiled_avals = [tile_aval_nd(global_axis_sizes, get_array_mapping(i.spec), aval)  # type: ignore
                      for aval, i in safe_zip(global_in_avals, in_shardings)]
    in_jaxpr_avals = in_tiled_avals

  with core.extend_axis_env_nd(mesh.shape.items()):
    if isinstance(fun_or_jaxpr, lu.WrappedFun):
      with dispatch.log_elapsed_time(
          "Finished tracing + transforming {fun_name} in {elapsed_time} sec",
          fun_name=str(name_stack), event=dispatch.JAXPR_TRACE_EVENT):
        jaxpr, out_jaxpr_avals, consts = pe.trace_to_jaxpr_final(
            fun_or_jaxpr, in_jaxpr_avals)
    else:
      assert isinstance(fun_or_jaxpr, core.ClosedJaxpr)
      jaxpr = fun_or_jaxpr.jaxpr
      out_jaxpr_avals = fun_or_jaxpr.out_avals
      consts = fun_or_jaxpr.consts

  all_args_info = AllArgsInfo(global_in_avals, in_shardings, jaxpr.debug_info)

  assert len(out_shardings) == len(out_jaxpr_avals)
  if spmd_lowering:
    global_out_avals = out_jaxpr_avals
  else:
    # In non-spmd lowering path, there is no `AUTO` or `UNSPECIFIED`, which is
    # why `.spec` can be accessed.
    global_out_avals = [untile_aval_nd(global_axis_sizes, get_array_mapping(o.spec), aval)  # type: ignore
                        for aval, o in safe_zip(out_jaxpr_avals, out_shardings)]

  _sanitize_mesh_jaxpr(jaxpr)
  jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)

  # 2. Build up the HLO
  tuple_args = dispatch.should_tuple_args(len(in_jaxpr_avals), backend.platform)

  in_partitions: list[sharding_impls.XLACompatibleSharding | None] | None
  out_partitions: list[sharding_impls.XLACompatibleSharding | None] | None
  axis_ctx: mlir.AxisContext
  if spmd_lowering:
    in_partitions = map(_to_logical_sharding, global_in_avals, in_shardings)
    out_partitions = map(_to_logical_sharding, global_out_avals, out_shardings)
    replicated_args = [False] * len(in_jaxpr_avals)
    axis_ctx = sharding_impls.SPMDAxisContext(mesh, manual_axes)
    num_replicas = 1
    num_partitions = mesh.devices.size
  else:
    replicated_args = [not get_array_mapping(i.spec) for i in in_shardings]  # type: ignore
    in_partitions = None
    out_partitions = None
    axis_env = sharding_impls.AxisEnv(
        nreps=mesh.size,
        names=tuple(global_axis_sizes.keys()),
        sizes=tuple(global_axis_sizes.values()))
    axis_ctx = sharding_impls.ReplicaAxisContext(axis_env)
    num_replicas = mesh.devices.size
    num_partitions = 1
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  module_name = f"{api_name}_{fun_name}"
  with core.extend_axis_env_nd(mesh.shape.items()):
    if any(effects.ordered_effects.contains(eff) for eff
           in closed_jaxpr.effects):
      raise ValueError("Ordered effects not supported in mesh computations.")
    unordered_effects = list(effects.ordered_effects.filter_not_in(
      closed_jaxpr.effects))
    ordered_effects = list(effects.ordered_effects.filter_in(
      closed_jaxpr.effects))
    with dispatch.log_elapsed_time(
        "Finished jaxpr to MLIR module conversion {fun_name} in {elapsed_time} sec",
        fun_name=str(name_stack), event=dispatch.JAXPR_TO_MLIR_MODULE_EVENT):
      lowering_result = mlir.lower_jaxpr_to_module(
          module_name,
          closed_jaxpr,
          ordered_effects=ordered_effects,
          backend_or_name=backend,
          platforms=lowering_parameters.platforms or (backend.platform,),
          axis_context=axis_ctx,
          name_stack=name_stack,
          donated_args=donated_invars,
          replicated_args=replicated_args,
          arg_shardings=in_partitions,
          result_shardings=out_partitions,
          arg_names=jaxpr.debug_info and jaxpr.debug_info.arg_names,
          result_names=jaxpr.debug_info and jaxpr.debug_info.result_paths,
          num_replicas=num_replicas,
          num_partitions=num_partitions,
          lowering_parameters=lowering_parameters)

  return MeshComputation(
      str(name_stack),
      lowering_result.module,
      donated_invars,
      global_in_avals=global_in_avals,
      global_out_avals=global_out_avals,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      spmd_lowering=spmd_lowering,
      tuple_args=tuple_args,
      auto_spmd_lowering=False,
      unordered_effects=unordered_effects,
      ordered_effects=ordered_effects,
      host_callbacks=lowering_result.host_callbacks,
      keepalive=lowering_result.keepalive,
      kept_var_idx=set(range(len(global_in_avals))),
      backend=backend,
      device_assignment=_create_da_object(tuple(mesh.devices.flat)),
      committed=True,
      in_layouts=(None,) * len(global_in_avals),
      out_layouts=(None,) * len(global_out_avals),
      shape_poly_state=lowering_result.shape_poly_state,
      all_args_info=all_args_info)

class MeshComputation(stages.XlaLowering):
  _hlo: ir.Module | None
  _executable: MeshExecutable | None

  def __init__(self, name: str, hlo: ir.Module | None,
               donated_invars: Sequence[bool], **compile_args):
    self._name = name
    self._hlo = hlo
    self._donated_invars = donated_invars
    self.compile_args = compile_args
    self._executable = None

  # -- stages.XlaLowering overrides

  def stablehlo(self) -> ir.Module:
    return self._hlo

  def compile(self, compiler_options=None) -> MeshExecutable:
    if self._executable is None or compiler_options is not None:
      executable = UnloadedMeshExecutable.from_hlo(
          self._name, self._hlo, **self.compile_args,
          compiler_options=compiler_options)
      if compiler_options is None:
        self._executable = executable
      return executable
    return self._executable

  def cost_analysis(self) -> dict[str, float]:
    backend = self.compile_args["backend"]
    if xb.using_pjrt_c_api(backend):
      raise NotImplementedError(
          "Lowered.cost_analysis not implemented on platform "
          f"'{backend.platform}'. Use compile().cost_analysis() for "
          "post-compilation cost estimates.")
    return xe.hlo_module_cost_analysis(backend, self.hlo().as_hlo_module())


if xla_extension_version < 229:
  def _get_input_indices(
      avals: Sequence[ShapedArray],
      shardings: Sequence[sharding_impls.XLACompatibleSharding],
      da_object: xc.DeviceList | Sequence[xc.Device],  # type: ignore
  ) -> Sequence[tuple[Index | None, ...]]:

    input_indices = []
    if not isinstance(da_object, xc.DeviceList):
      da_object = _create_da_object(tuple(da_object))
    num_addressable_devices = len(da_object.addressable_device_list)

    def _get_replicated_slices(num_addressable_devices: int, ndim: int | None):
      if ndim is None:
        return ((slice(None),),) * num_addressable_devices
      else:
        return ((slice(None),) * ndim,) * num_addressable_devices

    for aval, sharding in zip(avals, shardings):
      if aval is core.abstract_token:
        index = _get_replicated_slices(num_addressable_devices, None)
      else:
        if sharding.is_fully_replicated:
          index = _get_replicated_slices(num_addressable_devices, aval.ndim)
        else:
          index = tuple(
              sharding.addressable_devices_indices_map(aval.shape).values())  # type: ignore
      input_indices.append(index)

    return input_indices


def get_gspmd_shardings_from_executable(
    xla_executable,
    device_assignment: Sequence[xc.Device],
    num_out_avals: int,
    num_ordered_effects: int,
    all_default_mem_kind: bool,
) -> Sequence[sharding_impls.XLACompatibleSharding] | None:
  from jax._src import pjit

  if config.enable_memories.value:
    if all_default_mem_kind:
      omk = [None] * num_out_avals
    else:
      try:
        omk = xla_executable.get_output_memory_kinds()[0]
        if num_ordered_effects > 0:
          omk = omk[num_ordered_effects:]
      except:
        omk = [None] * num_out_avals
  else:
    omk = [None] * num_out_avals

  assert len(omk) == num_out_avals, (len(omk), num_out_avals)

  # When the device assignment only has 1 device, SPMD partitioner will not run.
  # Hence the op shardings will not be set on the `hlo_module`.
  if len(device_assignment) == 1:
    return [sharding_impls.GSPMDSharding.get_replicated(device_assignment, memory_kind=mk)
            for mk in omk]

  _, out_op_shardings = pjit.get_op_sharding_from_executable(xla_executable)
  if not out_op_shardings:
    return None

  if num_ordered_effects > 0:
    out_op_shardings = out_op_shardings[num_ordered_effects:]

  # This means that there are no outputs for JAX but for XLA there is an empty
  # tuple output which gets a replicated sharding.
  if num_out_avals == 0 and len(out_op_shardings) == 1:
    return None

  # This condition happens when all the elements in the output tuple have the
  # same sharding, so XLA decides to run the `FusionTupleDeduplicator` to
  # put the sharding on ROOT instead of the tuple.
  # TODO(b/245667823): Remove this when XLA fixes this.
  if len(out_op_shardings) == 1 and len(out_op_shardings) < num_out_avals:
    out_op_shardings = out_op_shardings * num_out_avals  # type: ignore

  assert len(out_op_shardings) == num_out_avals == len(omk), (
      len(out_op_shardings), num_out_avals, len(omk))

  return [sharding_impls.GSPMDSharding(device_assignment, os, memory_kind=mk)
          for os, mk in safe_zip(out_op_shardings, omk)]


# TODO(yashkatariya): Remove this function after `AUTO` can return shardings
# without mesh.
def _get_mesh_pspec_shardings_from_executable(
    xla_executable, mesh: Mesh
) -> tuple[Sequence[sharding_impls.NamedSharding],
           Sequence[sharding_impls.NamedSharding]]:
  from jax._src import pjit

  in_pspec, out_pspec = pjit.get_pspec_from_executable(xla_executable, mesh)
  return ([sharding_impls.NamedSharding(mesh, i) for i in in_pspec],
          [sharding_impls.NamedSharding(mesh, o) for o in out_pspec])


_orig_out_sharding_handlers = {}

_ShardingT = TypeVar("_ShardingT", bound=sharding_impls.XLACompatibleSharding)


def _register_out_sharding_handler(
    sharding_cls: type[_ShardingT],
    handler: Callable[[sharding_impls.GSPMDSharding, _ShardingT], _ShardingT],
) -> None:
  _orig_out_sharding_handlers[sharding_cls] = handler


def _gspmd_to_named_sharding(
    out_s: sharding_impls.GSPMDSharding,
    orig_in_s: sharding_impls.NamedSharding) -> sharding_impls.NamedSharding:
  parsed_pspec = sharding_impls.parse_flatten_op_sharding(
      out_s._hlo_sharding, orig_in_s.mesh)[0]
  return create_mesh_pspec_sharding(
      orig_in_s.mesh, parsed_pspec.get_partition_spec(), parsed_pspec,
      out_s.memory_kind)

_register_out_sharding_handler(
    sharding_impls.NamedSharding, _gspmd_to_named_sharding)


def _gspmd_to_positional_sharding(
    out_s: sharding_impls.GSPMDSharding,
    orig_in_s: sharding_impls.PositionalSharding
    ) -> sharding_impls.PositionalSharding:
  return sharding_impls._op_sharding_to_pos_sharding(
      out_s._hlo_sharding, orig_in_s._device_assignment, out_s.memory_kind)

_register_out_sharding_handler(
    sharding_impls.PositionalSharding, _gspmd_to_positional_sharding)

def _gspmd_to_single_device_sharding(
    out_s: GSPMDSharding, orig_in_s: SingleDeviceSharding) -> SingleDeviceSharding:
  assert isinstance(orig_in_s, SingleDeviceSharding)
  return SingleDeviceSharding(
      out_s._device_assignment[0], memory_kind=out_s.memory_kind)

_register_out_sharding_handler(
    SingleDeviceSharding, _gspmd_to_single_device_sharding)


def _get_out_sharding_from_orig_sharding(
    out_shardings, out_avals, orig_in_s, orig_aval, are_out_sharding_from_xla):
  out = []
  orig_handler = _orig_out_sharding_handlers[type(orig_in_s)]
  for o, out_aval, from_xla in safe_zip(out_shardings, out_avals,
                                        are_out_sharding_from_xla):
    if isinstance(o, sharding_impls.GSPMDSharding):
      try:
        # Only return the same input sharding object if the OpShardings and
        # in_aval.ndim and out_aval.ndim match. This is because if OpSharding is
        # replicated then, it doesn't encode the ndim in it. The devices
        # will be the same at this point because those checks happen before.
        if (orig_aval is not None and out_aval is not None and
            out_aval.ndim == orig_aval.ndim
            and sharding_impls.are_op_shardings_equal(
                o._hlo_sharding, orig_in_s._to_xla_hlo_sharding(orig_aval.ndim))
            and o.memory_kind == orig_in_s.memory_kind):
          out.append((orig_in_s, False))
        else:
          out.append((orig_handler(o, orig_in_s), False))
      except:
        out.append((o, from_xla))
    else:
      out.append((o, from_xla))
  return out

def maybe_get_orig_out_sharding(
    in_shardings, out_shardings, are_out_shardings_from_xla, in_avals,
    out_avals):
  if all(hasattr(o, '_original_sharding') for o in out_shardings):
    return ([o._original_sharding for o in out_shardings],
            (False,) * len(out_shardings))

  orig_in_s = None
  orig_aval = None
  for i, aval in safe_zip(in_shardings, in_avals):
    oi = getattr(i, '_original_sharding', None)
    if type(oi) in _orig_out_sharding_handlers:
      orig_in_s = oi
      orig_aval = aval
      break
  if orig_in_s is not None:
    return zip(*_get_out_sharding_from_orig_sharding(
        out_shardings, out_avals, orig_in_s, orig_aval, are_out_shardings_from_xla))

  return out_shardings, are_out_shardings_from_xla


def _get_layouts_from_executable(
    xla_executable, in_layouts, out_layouts, num_ordered_effects
) -> tuple[Sequence[SpecifiedLayout | None], Sequence[SpecifiedLayout | None]]:
  try:
    in_layouts_xla = xla_executable.get_parameter_layouts()
    out_layouts_xla = xla_executable.get_output_layouts()
  except:
    return (None,) * len(in_layouts), (None,) * len(out_layouts)

  if num_ordered_effects > 0:
    in_layouts_xla = in_layouts_xla[num_ordered_effects:]
    out_layouts_xla = out_layouts_xla[num_ordered_effects:]

  new_in_layouts = []
  for x, i in safe_zip(in_layouts_xla, in_layouts):
    x = SpecifiedLayout(x)
    if isinstance(i, SpecifiedLayout):
      if i != x:
        raise AssertionError(
            f"Unexpected XLA layout override: (XLA) {x} != {i} (User sharding)")
      new_in_layouts.append(i)
    else:
      new_in_layouts.append(x)

  new_out_layouts = []
  for x, o in safe_zip(out_layouts_xla, out_layouts):
    x = SpecifiedLayout(x)
    if isinstance(o, SpecifiedLayout):
      if o != x:
        raise AssertionError(
            f"Unexpected XLA layout override: (XLA) {x} != {o} (User sharding)")
      new_out_layouts.append(o)
    else:
      new_out_layouts.append(x)

  assert all(isinstance(i, SpecifiedLayout) for i in new_in_layouts)
  assert all(isinstance(o, SpecifiedLayout) for o in new_out_layouts)
  return new_in_layouts, new_out_layouts  # type: ignore


def get_logical_mesh_ids(mesh_shape):
  return np.arange(math.prod(mesh_shape)).reshape(mesh_shape)


@weakref_lru_cache
def _cached_compilation(
    computation,
    name,
    mesh,
    spmd_lowering,
    tuple_args,
    auto_spmd_lowering,
    _allow_propagation_to_outputs,
    _allow_propagation_to_parameters,
    host_callbacks,
    backend,
    da,
    pmap_nreps,
    compiler_options_keys,
    compiler_options_values,
):
  # TODO(phawkins): One would normally just write:
  # dev = np.array(device_assignment)
  # The formulation below is substantially faster if there are many devices.
  # If we were to optimize __getattr__ on xc.Device we might not need this
  # workaround.
  dev = np.vectorize(lambda i: da[i], otypes=[object])(
    np.arange(len(da))
  )
  if pmap_nreps > 1:
    num_replicas, num_partitions = pmap_nreps, 1
  elif spmd_lowering:
    num_replicas, num_partitions = 1, dev.size
  else:
    num_replicas, num_partitions = dev.size, 1

  if pmap_nreps > 1:
    # In `jit` device_assignment is set to None when num_replicas > 1. Do
    # the same thing here too.
    xla_device_assignment = None
  else:
    xla_device_assignment = dev.reshape((num_replicas, num_partitions))

  if compiler_options_keys is None:
    compiler_options = None
  else:
    compiler_options = dict(safe_zip(compiler_options_keys, compiler_options_values))

  fdo_profile = (None if compiler_options is None else
                 compiler_options.pop("fdo_profile", None))

  compile_options = compiler.get_compile_options(
      num_replicas=num_replicas,
      num_partitions=num_partitions,
      device_assignment=xla_device_assignment,
      use_spmd_partitioning=spmd_lowering,
      use_auto_spmd_partitioning=auto_spmd_lowering,
      env_options_overrides=compiler_options,
      fdo_profile=fdo_profile,
      detailed_logging=compiler.use_detailed_logging(computation),
      backend=backend,
  )

  opts = compile_options.executable_build_options
  if auto_spmd_lowering:
    assert mesh is not None
    opts.auto_spmd_partitioning_mesh_shape = list(mesh.shape.values())
    opts.auto_spmd_partitioning_mesh_ids = (
        get_logical_mesh_ids(list(mesh.shape.values()))
        .reshape(-1))
  compile_options.parameter_is_tupled_arguments = tuple_args
  opts.allow_spmd_sharding_propagation_to_output = list(_allow_propagation_to_outputs)
  opts.allow_spmd_sharding_propagation_to_parameters = list(
      _allow_propagation_to_parameters
  )

  if hasattr(backend, "compile_replicated"):
    return None, compile_options

  with dispatch.log_elapsed_time(
      "Finished XLA compilation of {fun_name} in {elapsed_time} sec",
      fun_name=name, event=dispatch.BACKEND_COMPILE_EVENT):
    xla_executable = compiler.compile_or_get_cached(
        backend, computation, dev, compile_options, host_callbacks)
  return xla_executable, compile_options


def _get_shardings_from_executable(
    xla_executable, out_shardings, device_assignment, global_out_avals,
    num_ordered_effects, all_default_mem_kind
  ):
  out_shardings_xla = get_gspmd_shardings_from_executable(  # type: ignore
      xla_executable, device_assignment, len(global_out_avals),
      num_ordered_effects, all_default_mem_kind)  # type: ignore
  if out_shardings_xla is None:
    return out_shardings, (False,) * len(global_out_avals)

  orig_out_shardings = out_shardings
  out_shardings, are_out_shardings_from_xla = [], []  # type: ignore
  for xla_s, orig, aval in safe_zip(out_shardings_xla, orig_out_shardings,
                                    global_out_avals):
    if is_unspecified(orig):
      out_shardings.append(xla_s)
      are_out_shardings_from_xla.append(True)
    else:
      xla_hlo_s = xla_s._to_xla_hlo_sharding(aval.ndim)  # type: ignore
      orig_hlo_s = orig._to_xla_hlo_sharding(aval.ndim)  # type: ignore
      # MANUAL HloSharding comes from other partitioning frameworks.
      if (not dtypes.issubdtype(aval.dtype, dtypes.extended) and
          not xla_hlo_s.is_manual() and
          (not op_shardings.are_op_shardings_equal(xla_hlo_s, orig_hlo_s) or
           xla_s.memory_kind != orig.memory_kind)):  # type: ignore
        raise AssertionError(
            f"Unexpected XLA sharding override: (XLA) {xla_s} != {orig} "
            "(User sharding)")
      out_shardings.append(orig)
      are_out_shardings_from_xla.append(False)
  return out_shardings, are_out_shardings_from_xla


def finalize_out_shardings(out_shardings, are_out_shardings_from_xla,
                           device_assignment):
  if len(device_assignment) == 1:
    return ([SingleDeviceSharding(device_assignment[0], memory_kind=o.memory_kind)
             if isinstance(o, GSPMDSharding) else o for o in out_shardings],
            are_out_shardings_from_xla)
  return out_shardings, are_out_shardings_from_xla


@dataclasses.dataclass
class UnloadedMeshExecutable:
  xla_executable: Any
  device_assignment: xc.DeviceList | Sequence[xc.Device]  # type: ignore
  backend: xb.XlaBackend
  input_avals: Sequence[ShapedArray]
  input_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  output_avals: Sequence[ShapedArray]
  output_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  committed: bool
  are_out_shardings_from_xla: Sequence[bool]
  name: str
  unordered_effects: list[core.Effect]
  ordered_effects: list[core.Effect]
  keepalive: Sequence[Any]
  host_callbacks: Sequence[Any]
  kept_var_idx: set[int]
  auto_spmd_lowering: bool
  in_layouts: Sequence[SpecifiedLayout | None]
  out_layouts: Sequence[SpecifiedLayout | None]
  all_args_info: AllArgsInfo | None

  def build_unsafe_call(self):
    if xla_extension_version >= 229:
      handle_args = InputsHandler(self.input_shardings)
    else:
      input_indices = _get_input_indices(self.input_avals, self.input_shardings,
                                         self.device_assignment)
      handle_args = InputsHandler(
          self.input_shardings, self.xla_executable.local_devices(), input_indices)
    handle_outs = global_avals_to_results_handler(
        self.output_avals, self.output_shardings, self.committed,
        self.are_out_shardings_from_xla)  # type: ignore  # arg-type

    unsafe_call = ExecuteReplicated(  # type: ignore  # assignment
        self.xla_executable, self.name, self.backend, handle_args,
        handle_outs, self.unordered_effects, self.ordered_effects, self.keepalive,
        bool(self.host_callbacks), self.kept_var_idx)
    return unsafe_call

  def load(self) -> MeshExecutable:
    return MeshExecutable(self.xla_executable, self.build_unsafe_call,
                          self.input_avals,
                          self.input_shardings, self.output_shardings,
                          self.auto_spmd_lowering, self.kept_var_idx,
                          self.in_layouts, self.out_layouts,
                          self.all_args_info, self)

  # May return a MeshExecutable in the compile_replicated case.
  @staticmethod
  def from_hlo(
      name: str,
      hlo: ir.Module,
      global_in_avals: Sequence[ShapedArray],
      global_out_avals: Sequence[ShapedArray],
      in_shardings: Sequence[
          (sharding_impls.XLACompatibleSharding | AUTO | UnspecifiedValue)
      ],
      out_shardings: Sequence[
          (sharding_impls.XLACompatibleSharding | AUTO | UnspecifiedValue)
      ],
      spmd_lowering: bool,
      tuple_args: bool,
      auto_spmd_lowering: bool,
      unordered_effects: list[core.Effect],
      ordered_effects: list[core.Effect],
      host_callbacks: list[Any],
      keepalive: Any,
      kept_var_idx: set[int],
      backend: xb.XlaBackend,
      device_assignment: xc.DeviceList | Sequence[xc.Device],  # type: ignore
      committed: bool,
      in_layouts: MaybeLayout,
      out_layouts: MaybeLayout,
      pmap_nreps: int = 1,
      shape_poly_state: mlir.ShapePolyLoweringState | None = None,
      all_default_mem_kind: bool = True,
      all_args_info: AllArgsInfo | None = None,
      compiler_options=None,
  ) -> MeshExecutable:
    if shape_poly_state is not None and shape_poly_state.uses_dim_vars:
      hlo = mlir.refine_polymorphic_shapes(hlo)
    compiler_options_keys = tuple(
        compiler_options.keys()) if compiler_options is not None else None
    compiler_options_values = tuple(
        compiler_options.values()) if compiler_options is not None else None
    if isinstance(device_assignment, xc.DeviceList):
      da = device_assignment
    else:
      da = _create_da_object(tuple(device_assignment))
    del device_assignment
    allow_prop_to_parameters = tuple(is_unspecified(i) for i in in_shardings)
    allow_prop_to_outputs = tuple(is_unspecified(o) for o in out_shardings)

    mesh = None
    if auto_spmd_lowering:
      for i in it.chain.from_iterable([in_shardings, out_shardings]):
        if is_auto(i):
          mesh = i.mesh  # type: ignore
          break

    xla_executable, compile_options = _cached_compilation(
        hlo,
        name,
        mesh,
        spmd_lowering,
        tuple_args,
        auto_spmd_lowering,
        allow_prop_to_outputs,
        allow_prop_to_parameters,
        tuple(host_callbacks),
        backend,
        da,
        pmap_nreps,
        compiler_options_keys,
        compiler_options_values,
    )

    if hasattr(backend, "compile_replicated"):
      semantics_in_shardings = SemanticallyEqualShardings(in_shardings)  # type: ignore
      semantics_out_shardings = SemanticallyEqualShardings(out_shardings)  # type: ignore
      return _compile_replicated_mesh_executable_from_hlo(
          hlo, name, tuple(global_in_avals), tuple(global_out_avals),
          semantics_in_shardings, semantics_out_shardings, auto_spmd_lowering,
          compile_options, tuple(host_callbacks), bool(unordered_effects),
          tuple(ordered_effects), tuple(kept_var_idx), backend, da, committed,
          pmap_nreps)

    if auto_spmd_lowering:
      assert mesh is not None
      in_shardings_xla, out_shardings_xla = _get_mesh_pspec_shardings_from_executable(
          xla_executable, mesh)
      in_shardings = [x if is_auto(i) else getattr(i, '_original_sharding', i)  # type: ignore
                      for x, i in safe_zip(in_shardings_xla, in_shardings)]
      out_shardings_tuple = [
          (x, True) if is_auto(o) else (o, False)
          for x, o in safe_zip(out_shardings_xla, out_shardings)
      ]
      out_shardings, are_out_shardings_from_xla = unzip2(out_shardings_tuple)
    else:
      if pmap_nreps == 1:
        assert mesh is None
        # TODO(yashkatariya): Make da directly usable in the downstream code
        # without tuple conversion.
        out_shardings, are_out_shardings_from_xla = _get_shardings_from_executable(
            xla_executable, out_shardings, tuple(da), global_out_avals,
            len(ordered_effects), all_default_mem_kind)
      else:
        in_shardings, out_shardings, committed, da = _get_metadata_jit_pmap(
            xla_executable.local_devices(), len(in_shardings), len(out_shardings))
        are_out_shardings_from_xla = (False,) * len(global_out_avals)

    if xla_extension_version >= 217:
      in_layouts, out_layouts = _get_layouts_from_executable(
          xla_executable, in_layouts, out_layouts, len(ordered_effects))
    else:
      assert all(i is None for i in in_layouts)
      assert all(o is None for o in out_layouts)

    out_shardings, are_out_shardings_from_xla = maybe_get_orig_out_sharding(
        in_shardings, out_shardings, are_out_shardings_from_xla,
        global_in_avals, global_out_avals)

    out_shardings, are_out_shardings_from_xla = finalize_out_shardings(
        out_shardings, are_out_shardings_from_xla, da)

    return UnloadedMeshExecutable(
        xla_executable=xla_executable,
        device_assignment=da,  # type: ignore
        backend=backend,
        input_avals=global_in_avals,
        input_shardings=in_shardings,  # type: ignore
        output_avals=global_out_avals,
        output_shardings=out_shardings,  # type: ignore # arg-type
        committed=committed,
        are_out_shardings_from_xla=are_out_shardings_from_xla,
        name=name,
        unordered_effects=unordered_effects,
        ordered_effects=ordered_effects,
        keepalive=keepalive,
        host_callbacks=host_callbacks,
        kept_var_idx=kept_var_idx,
        auto_spmd_lowering=auto_spmd_lowering,
        in_layouts=in_layouts,  # type: ignore
        out_layouts=out_layouts,  # type: ignore
        all_args_info=all_args_info).load()  # type: ignore


class MeshExecutableFastpathData(NamedTuple):
  xla_executable: xc.LoadedExecutable
  out_pytree_def: Any
  in_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  out_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  out_avals: Sequence[ShapedArray]
  out_committed: Sequence[bool]
  kept_var_bitvec: Iterable[bool]
  # TODO(yashkatariya): Remove once minimum jaxlib version is 0.4.24
  arg_handler_devices: Sequence[xc.Device]
  arg_handler_indices: Sequence[tuple[Index | None, ...]]


def reflatten_outputs_for_dispatch(out_tree, out_flat):
  # We arrive at dispatch having flattened according to the default
  # pytree registry, but we want to re-flatten according to our
  # dispatch-specific registry.
  out_unflat = tree_util.tree_unflatten(out_tree, out_flat)
  return tree_util.dispatch_registry.flatten(out_unflat, None)


class MeshExecutable(stages.XlaExecutable):
  __slots__ = [
      "xla_executable", "_unsafe_call", "build_unsafe_call", "in_avals",
      "_in_shardings", "_out_shardings", "_auto_spmd_lowering", "_kept_var_idx",
      "_in_layouts", "_out_layouts", "_all_args_info", "_unloaded_executable",
  ]

  def __init__(self, xla_executable, build_unsafe_call, in_avals, in_shardings,
               out_shardings, auto_spmd_lowering, kept_var_idx,
               in_layouts, out_layouts,
               all_args_info: AllArgsInfo | None = None,
               unloaded_executable=None):
    self.xla_executable = xla_executable
    self.build_unsafe_call = build_unsafe_call
    # in_avals is a list of global and local avals. Aval is global if input
    # is a GDA or jax.Array else local.
    self.in_avals = in_avals
    self._unsafe_call = None
    self._in_shardings = in_shardings
    self._out_shardings = out_shardings
    self._auto_spmd_lowering = auto_spmd_lowering
    self._kept_var_idx = kept_var_idx
    self._in_layouts = in_layouts
    self._out_layouts = out_layouts
    self._all_args_info = all_args_info
    self._unloaded_executable = unloaded_executable

  @property
  def unsafe_call(self) -> Callable[..., Any]:
    if self._unsafe_call is None:
      self._unsafe_call = self.build_unsafe_call()
    return self._unsafe_call  # type: ignore

  # -- stages.XlaExecutable overrides

  def xla_extension_executable(self):
    return self.xla_executable

  def call(self, *args):
    if self._all_args_info is None:
      kept_args = [a for i, a in enumerate(args) if i in self._kept_var_idx]
      ref_avals = self.in_avals
      in_shardings = self._in_shardings
      debug_info = None
    else:
      kept_args = args
      ref_avals = self._all_args_info.in_avals
      iter_in_shardings = iter(self._in_shardings)
      in_shardings = [next(iter_in_shardings) if i in self._kept_var_idx else s
                      for i, s in enumerate(self._all_args_info.in_shardings)]
      debug_info = self._all_args_info.debug_info

    arg_avals = map(xla.abstractify, kept_args)
    check_arg_avals_for_call(ref_avals, arg_avals, debug_info)
    # Check the GDA sharding and the input sharding.
    check_gda_or_array_xla_sharding_match(kept_args, in_shardings, debug_info)
    return self.unsafe_call(*args)  # pylint: disable=not-callable

  def input_shardings(self) -> Sequence[sharding_impls.XLACompatibleSharding]:
    return self._in_shardings

  def output_shardings(self) -> Sequence[sharding_impls.XLACompatibleSharding]:
    return self._out_shardings

  def input_layouts(self):
    return self._in_layouts

  def output_layouts(self):
    return self._out_layouts

  def create_cpp_call(self, no_kwargs, in_tree, out_tree):
    if not (isinstance(self.unsafe_call, ExecuteReplicated) and
            not self.unsafe_call.has_unordered_effects and
            not self.unsafe_call.has_host_callbacks):
      return None

    def aot_cache_miss(*args, **kwargs):
      params = stages.CompiledCallParams(self, no_kwargs, in_tree, out_tree)
      outs, out_flat, args_flat = stages.Compiled.call(params, *args, **kwargs)
      out_flat, out_tree_dispatch = reflatten_outputs_for_dispatch(
          out_tree, out_flat)
      use_fastpath = (all(isinstance(x, xc.ArrayImpl) for x in out_flat))

      if use_fastpath:
        out_avals = [o.aval for o in out_flat]
        out_committed = [o._committed for o in out_flat]
        kept_var_bitvec = [i in self._kept_var_idx
                           for i in range(len(args_flat))]
        fastpath_data = MeshExecutableFastpathData(
            self.xla_executable, out_tree_dispatch, self._in_shardings,
            self._out_shardings, out_avals, out_committed, kept_var_bitvec,
            self.unsafe_call.in_handler.local_devices,
            self.unsafe_call.in_handler.input_indices)
      else:
        fastpath_data = None
      return outs, fastpath_data

    if xla_extension_version >= 226:
      return xc._xla.pjit(
          self.unsafe_call.name, None, aot_cache_miss, [], [], [],
          tree_util.dispatch_registry,
          shard_arg if xla_extension_version >= 229 else temp_shard_arg)  # type: ignore
    else:
      return xc._xla.pjit(self.unsafe_call.name, None, aot_cache_miss, [], [], [],  # type: ignore
                          tree_util.dispatch_registry)


# TODO(yashkatariya): Remove once minimum jaxlib version is 0.4.24
def temp_shard_arg(arg, devices, arg_indices, sharding, canonicalize=True):
  return shard_arg(arg, sharding)


def check_arg_avals_for_call(ref_avals, arg_avals,
                             jaxpr_debug_info: core.JaxprDebugInfo | None = None):
  if len(ref_avals) != len(arg_avals):
    raise TypeError(
        f"Computation compiled for {len(ref_avals)} inputs "
        f"but called with {len(arg_avals)}")

  if jaxpr_debug_info is not None:
    arg_names = [f"'{name}'" for name in jaxpr_debug_info.arg_names]
  else:
    num_args = len(ref_avals)
    arg_names = [f"{i + 1}/{num_args}" for i in range(num_args)]

  errors = []
  for ref_aval, arg_aval, name in safe_zip(ref_avals, arg_avals, arg_names):
    if not core.typematch(ref_aval, arg_aval):
      errors.append(
          f"Argument {name} compiled with {ref_aval.str_short()} and called "
          f"with {arg_aval.str_short()}")
  if errors:
    max_num_errors = 5
    str_errors = "\n".join(errors[:max_num_errors])
    if len(errors) >= max_num_errors:
      num_mismatch_str = f"The first {max_num_errors} of {len(errors)}"
    else:
      num_mismatch_str = "The"
    raise TypeError(
        "Argument types differ from the types for which this computation was "
        f"compiled. {num_mismatch_str} mismatches are:\n{str_errors}")


def _get_metadata_jit_pmap(local_devices, num_in_shardings, num_out_shardings):
  # Create replicated shardings for jit(pmap) path with local devices
  # because multihost jit(pmap) is not allowed.
  gs = sharding_impls.GSPMDSharding.get_replicated(local_devices)
  in_shardings = [gs] * num_in_shardings
  out_shardings = [gs] * num_out_shardings
  # jit(pmap) will generate Arrays with multi-device sharding.
  # It is unsupported for these shardings to be uncommitted, so force
  # the outputs to be committed.
  committed = True
  return in_shardings, out_shardings, committed, tuple(local_devices)


@weakref_lru_cache
def _compile_replicated_mesh_executable_from_hlo(
    computation, name, global_in_avals, global_out_avals, semantics_in_shardings,
    semantics_out_shardings, auto_spmd_lowering, compile_options,
    host_callbacks, has_unordered_effects, ordered_effects, kept_var_idx,
    backend, da, committed, pmap_nreps):
  assert not auto_spmd_lowering
  in_shardings = semantics_in_shardings.shardings
  out_shardings = semantics_out_shardings.shardings

  kept_var_idx = set(kept_var_idx)
  # Will compute out_handler with executable information.
  unsafe_call = backend.compile_replicated(
      is_trivial=False, name=name, computation=computation,
      compile_options=compile_options, host_callbacks=host_callbacks,
      has_unordered_effects=has_unordered_effects,
      device_assignment=da, ordered_effects=ordered_effects,
      in_avals=global_in_avals,
      in_shardings=in_shardings, kept_var_idx=kept_var_idx,
      out_avals=global_out_avals, out_shardings=out_shardings,
      committed=committed, pmap_nreps=pmap_nreps)
  xla_executable = None
  return MeshExecutable(xla_executable, lambda: unsafe_call, global_in_avals,
                        in_shardings, out_shardings, auto_spmd_lowering,
                        kept_var_idx, (None,) * len(global_in_avals),
                        (None,) * len(global_out_avals))


@lru_cache
def create_mesh_pspec_sharding(
    mesh: Mesh, pspec: PartitionSpec | None, parsed_pspec=None,
    memory_kind: str | None = None) -> sharding_impls.NamedSharding:
  if pspec is None:
    pspec, parsed_pspec = PartitionSpec(), None
  return sharding_impls.NamedSharding(mesh, pspec, _parsed_pspec=parsed_pspec,
                                      memory_kind=memory_kind)


def check_device_backend_on_shardings(shardings) -> bool:
  for i in shardings:
    if is_unspecified(i) or is_auto(i):
      continue
    if hasattr(i, '_original_sharding') and getattr(
        i._original_sharding, '_device_backend', False):
      return True
  return False


def check_gda_or_array_xla_sharding_match(
    args, in_xla_shardings: Sequence[sharding_impls.XLACompatibleSharding],
    jaxpr_debug_info: core.JaxprDebugInfo | None) -> None:
  from jax._src.array import ArrayImpl
  arg_names = ([''] * len(args) if jaxpr_debug_info is None else
               jaxpr_debug_info.arg_names)
  errors = []
  num_errors = 5
  for arg, xs, name in safe_zip(args, in_xla_shardings, arg_names):
    if not isinstance(arg, ArrayImpl):
      continue
    if is_unspecified_or_auto(xs):
      continue

    db_xs = check_device_backend_on_shardings([xs])
    if not db_xs:
      xs = getattr(xs, '_original_sharding', xs)

    # Raise memory kind mismatch error even if the arg is uncommitted.
    if arg.sharding.memory_kind != xs.memory_kind:
      errors.append(
          "Got input sharding(s) that compiled object was called with: "
          f"{arg.sharding} and sharding(s) the computation was compiled "
          f"with: {xs} for arg {name} with shape: {arg.aval.str_short()}")

    if (not db_xs and arg._committed and
        not op_shardings.are_op_shardings_equal(
            arg.sharding._to_xla_hlo_sharding(arg.ndim),
            xs._to_xla_hlo_sharding(arg.ndim))):
      errors.append(
          "Got input sharding(s) that compiled object was called with: "
          f"{arg.sharding} and sharding(s) the computation was compiled "
          f"with: {xs} for arg {name} with shape: {arg.aval.str_short()}")

  if errors:
    str_errors = '\n'.join(errors[:num_errors])
    num_mismatch_str = (
        f'the {len(errors)} mismatches' if len(errors) < num_errors else
        f"{num_errors} mismatches out of {len(errors)}")
    raise ValueError(
          "Compiled object called with input sharding(s) does not match the "
          "sharding(s) the computation was compiled with. "
          f"Here are {num_mismatch_str}:\n{str_errors}")


def get_array_mapping(pspec: PartitionSpec) -> ArrayMappingOrAutoOrUnspecified:
  parsed_pspec, _, _ = sharding_impls.prepare_axis_resources(
      pspec, "pspec to array_mapping")
  return _get_array_mapping(parsed_pspec)


_forbidden_primitives = {
  'xla_pmap': 'pmap',
}
def _sanitize_mesh_jaxpr(jaxpr):
  if isinstance(jaxpr, core.ClosedJaxpr):
    jaxpr = jaxpr.jaxpr
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in _forbidden_primitives:
      raise RuntimeError(f"Nesting {_forbidden_primitives[eqn.primitive.name]} "
                         f"inside xmaps not supported!")
    core.traverse_jaxpr_params(_sanitize_mesh_jaxpr, eqn.params)


custom_resource_typing_rules: dict[core.Primitive, Callable] = {}

def resource_typecheck(jaxpr, resource_env, axis_resources, what_jaxpr_thunk):
  if isinstance(jaxpr, core.ClosedJaxpr):
    jaxpr = jaxpr.jaxpr
  def _check_aval(aval, what_thunk):
    if not hasattr(aval, 'named_shape'):
      return
    resource_to_axis = {}
    for axis in aval.named_shape:
      if axis_resources:
        for resource in axis_resources[axis]:
          if resource in resource_to_axis:
            other_axis = resource_to_axis[resource]
            axis, other_axis = sorted([str(axis), str(other_axis)])
            raise JAXTypeError(
                f"Axes `{axis}` and `{other_axis}` are both mapped to the "
                f"resource `{resource}`, but they coincide in the named_shape "
                f"of {what_thunk()}")
          resource_to_axis[resource] = axis

  what_thunk = lambda: (f"an input to {what_jaxpr_thunk()}")
  for v in jaxpr.constvars:
    _check_aval(v.aval, what_thunk)
  for v in jaxpr.invars:
    _check_aval(v.aval, what_thunk)
  what_thunk = lambda: (f"a value returned from a primitive {eqn.primitive} created "
                        f"at {source_info_util.summarize(eqn.source_info)}")
  rec_what_jaxpr_thunk = lambda: (f"a primitive {eqn.primitive} created at"
                                  f"{source_info_util.summarize(eqn.source_info)}")
  for eqn in jaxpr.eqns:
    typing_rule = custom_resource_typing_rules.get(eqn.primitive, None)
    if typing_rule:
      typing_rule([v.aval for v in eqn.invars], eqn.params, eqn.source_info,
                  resource_env, axis_resources)
    else:
      core.traverse_jaxpr_params(partial(resource_typecheck,
                                         resource_env=resource_env,
                                         axis_resources=axis_resources,
                                         what_jaxpr_thunk=rec_what_jaxpr_thunk),
                                 eqn.params)
    for v in eqn.outvars:
      _check_aval(v.aval, what_thunk)


@contextmanager
def maybe_extend_axis_env(*args, **kwargs):
  with core.extend_axis_env(*args, **kwargs):
    yield


def device_put(x, devices: Sequence[xc.ArrayImpl],
               replicate: bool=False) -> list[xc.ArrayImpl]:
  """Call device_put on a sequence of devices and return a flat sequence of buffers."""
  if replicate:
    return [jax.device_put(x, device) for device in devices]
  else:
    return [jax.device_put(val, device) for val, device in safe_zip(x, devices)]
