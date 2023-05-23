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
from collections import defaultdict, namedtuple
import dataclasses
from functools import partial, lru_cache, cached_property
import itertools as it
import logging
import math
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, FrozenSet,
                    Sequence, Set, Tuple, Type, Union, Iterable,
                    TYPE_CHECKING, cast, TypeVar)

import numpy as np

import jax
from jax.errors import JAXTypeError
from jax.tree_util import tree_map

from jax._src import api_util
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
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.abstract_arrays import array_types
from jax._src.config import config
from jax._src.core import ShapedArray
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import mlir
from jax._src.interpreters import xla
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.partition_spec import PartitionSpec
from jax._src.sharding_impls import (
    ArrayMapping, ArrayMappingOrAutoOrUnspecified,
    AUTO, UnspecifiedValue, UNSPECIFIED,
    get_array_mapping as _get_array_mapping, is_auto, is_unspecified
)
from jax._src.util import (unzip3, safe_map, safe_zip, partition_list,
                           wrap_name, tuple_delete, distributed_debug_log,
                           unzip2, HashableFunction, weakref_lru_cache)


# Built in Python lists don't support weak refs but subclasses of lists do.
class WeakRefList(list):
  pass


xe = xc._xla

unsafe_map, map = map, safe_map  # type: ignore

logger = logging.getLogger(__name__)

Index = Union[int, slice, Tuple[Union[int, slice], ...]]

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

def shard_arg(arg, devices, arg_indices, sharding):
  """Returns a list of size len(devices) containing per-device buffers.

  For the C++ pmap path, we fallback to Python (this function) to shard
  arguments that are not supported by the C++ `ShardArg`.

  Args:
    arg: The Python argument.
    devices: The list of devices to shard over.
    arg_indices: A list of `len(devices)` indices to use to shard the argument.
  """
  arg = xla.canonicalize_dtype(arg)
  return shard_arg_handlers[type(arg)](arg, devices, arg_indices, sharding)


@profiler.annotate_function
def shard_args(
    devices: Sequence[xb.xla_client.Device],
    indices: Sequence[Sequence[Index]],
    shardings: Sequence[sharding_impls.XLACompatibleSharding],
    args,
) -> Sequence[jax.Array]:
  """Shard each argument data array along its leading axis.

  Args:
    devices: sequence of Devices mapping replica index to a physical device.
    indices: sequence of the same length as `args` describing how each arg
      should be sharded/replicated across `devices`. Each element in `indices`
      is the same length as `devices`.
    args: a sequence of JaxTypes representing arguments to be sharded according
      to `indices` and placed on `devices`.

  Returns:
    A list of length matching args, containing lists of per-device buffers
    for each argument.
  """
  return [shard_arg(arg, devices, indices[i], shardings[i])
          for i, arg in enumerate(args)]

shard_arg_handlers: Dict[Any, Callable[[Any, Any, Any, Any], Any]] = {}

def _shard_token(x, devices, indices, sharding):
  zeros = np.zeros((), dtype=np.dtype(np.bool_))
  aval = api_util.shaped_abstractify(zeros)
  out = batched_device_put(aval, sharding, [zeros for i in indices], devices)
  return out
shard_arg_handlers[core.Token] = _shard_token

def _masked_array_error(x, devices, indices, sharding):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")
shard_arg_handlers[np.ma.MaskedArray] = _masked_array_error

def _shard_array(x, devices, indices, sharding):
  if x.dtype == dtypes.float0:
    x = np.zeros(x.shape, dtype=np.dtype(bool))
  aval = api_util.shaped_abstractify(x)
  out = batched_device_put(aval, sharding, [x[i] for i in indices], devices)
  return out
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array

def shard_device_array(x, devices, indices, sharding):
  start_indices, limit_indices, removed_dims = unzip3(
      as_slice_indices(x, idx) for idx in indices)
  shards = x._multi_slice(start_indices, limit_indices, removed_dims)
  aval = api_util.shaped_abstractify(x)
  out = batched_device_put(aval, sharding, shards, devices)
  return out

def batched_device_put(aval: core.ShapedArray,
                       sharding: jax.sharding.Sharding, xs: Sequence[Any],
                       devices: Sequence[jax.Device], committed: bool = True):
  from jax._src import array

  bufs = [x for x, d in safe_zip(xs, devices)
          if (isinstance(x, array.ArrayImpl) and
              dispatch.is_single_device_sharding(x.sharding) and
              x.device() == d)]
  if len(bufs) == len(xs):
    return array.ArrayImpl(
        aval, sharding, bufs, committed=committed, _skip_checks=True)
  return xc.batched_device_put(aval, sharding, xs, devices, committed)  # type: ignore


# NOTE(skye): we could refactor to generate _multi_slice parameters directly
# from the input ShardingSpec, rather than the indices. However, this would
# require duplicating the ordering logic of spec_to_indices, which is more
# subtle and more likely to change than the index logic we have to support here.
def as_slice_indices(arr: Any, idx: Index) -> Tuple[
    Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
  """Returns start_indices, limit_indices, removed_dims"""
  start_indices = [0] * arr.ndim
  limit_indices = list(arr.shape)
  removed_dims = []

  tuple_idx = idx if isinstance(idx, tuple) else (idx,)
  for dim, sub_idx in enumerate(tuple_idx):
    if isinstance(sub_idx, int):
      start_indices[dim] = sub_idx
      limit_indices[dim] = sub_idx + 1
      removed_dims.append(dim)
    elif sub_idx == slice(None):
      continue
    else:
      assert isinstance(sub_idx, slice), sub_idx
      assert isinstance(sub_idx.start, int), sub_idx
      assert isinstance(sub_idx.stop, int), sub_idx
      start_indices[dim] = sub_idx.start
      limit_indices[dim] = sub_idx.stop

  return tuple(start_indices), tuple(limit_indices), tuple(removed_dims) # type: ignore


def shard_aval(size, axis: int, aval):
  try:
    return shard_aval_handlers[type(aval)](size, axis, aval)
  except KeyError as err:
    raise TypeError(f"No shard_aval handler for type: {type(aval)}") from err
shard_aval_handlers: Dict[Type[core.AbstractValue], Callable[[int, int, Any], Any]] = {}
def _shard_abstract_array(size, axis: int, x):
  try:
    if x.shape[axis] != size:
      raise ValueError(f"Axis size {size} does not match dimension {axis} of "
                       f"shape {x.shape}")
  except IndexError:
    raise ValueError("Cannot split a {x.dim}D value along axis {axis}") from None
  return x.update(shape=tuple_delete(x.shape, axis))
shard_aval_handlers[ShapedArray] = _shard_abstract_array


def local_aval_to_result_handler(
    aval: core.AbstractValue,
    sharding: sharding_impls.XLACompatibleSharding,
    indices: Optional[Tuple[Index, ...]],
) -> Callable[[List[xc.ArrayImpl]], Any]:
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
    to the user, e.g. a ShardedDeviceArray.
  """
  try:
    return local_result_handlers[(type(aval))](aval, sharding, indices)
  except KeyError as err:
    raise TypeError(
        f"No pxla_result_handler for type: {type(aval)}") from err

PxlaResultHandler = Callable[..., Callable[[Any], Any]]
local_result_handlers: Dict[Type[core.AbstractValue], PxlaResultHandler] = {}


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
    to the user, e.g. a ShardedDeviceArray.
  """
  try:
    return global_result_handlers[type(aval)](
        aval, out_sharding, committed, is_out_sharding_from_xla)
  except KeyError as err:
    raise TypeError(
        f"No pxla_result_handler for type: {type(aval)}") from err

global_result_handlers: Dict[Type[core.AbstractValue], PxlaResultHandler] = {}

### lazy device-memory persistence and result handling

# TODO(yashkatariya, phawkins): Remove this function after March 15, 2023.
def make_sharded_device_array(
    aval: ShapedArray,
    sharding_spec: Optional[ShardingSpec],
    # Any is for JAX extensions implementing their own buffer.
    device_buffers: List[Any],
    indices: Optional[Tuple[Index, ...]] = None,
):
  """Returns a ShardedDeviceArray implementation based on arguments.

  Returns either a C++ SDA or a Python DeviceArray when the buffers are not
  JAX buffers.

  Args:
    aval: The `ShapedArray` for this array.
    sharding_spec: If `None`, assumes a pmap-style ShardedDeviceArrays over the
      first dimension.
    device_buffers: If a list of Jax `Buffer` objects, a C++ SDA will be
      returned (if the version is high enough). Otherwise, a Python object will
      be returned, for JAX extensions not implementing the C++ API.
    indices: For caching purposes, will be computed if `None`.
  """
  if sharding_spec is None:
    sharding_spec = sharding_specs.create_pmap_sharding_spec(aval.shape)

  mesh = mesh_lib.thread_resources.env.physical_mesh
  sharding: sharding_impls.XLACompatibleSharding
  if mesh.empty:
    sharding = sharding_impls.PmapSharding(
        np.asarray([d.device() for d in device_buffers]), sharding_spec)
  else:
    op_sharding = sharding_specs.sharding_spec_sharding_proto(sharding_spec)
    pspec = sharding_impls.parse_flatten_op_sharding(
        op_sharding, mesh)[0].get_partition_spec()
    sharding = sharding_impls.NamedSharding(mesh, pspec)

  return jax.make_array_from_single_device_arrays(
      aval.shape, sharding, device_buffers)  # type: ignore


if TYPE_CHECKING:
  ShardedDeviceArray = Any
else:
  class ShardedDeviceArray(object):
    def __init__(self):
      raise RuntimeError("ShardedDeviceArray is a backward compatibility shim "
                         "and cannot be instantiated.")


def _hashable_index(idx):
  return tree_map(lambda x: (x.start, x.stop) if type(x) == slice else x, idx)

# The fast path is handled directly in shard_args().
# TODO(yashkatariya): Move this to array.py when SDA is deleted. The local
# import of Array should go away at that time.
def shard_sharded_device_array_slow_path(x, devices, indices, sharding):
  from jax._src.array import ArrayImpl

  candidates = defaultdict(list)
  if isinstance(x, ArrayImpl):
    bufs = [buf.data for buf in x.addressable_shards]
    arr_indices = tuple(x.sharding.devices_indices_map(x.shape).values())
  else:
    bufs = x.device_buffers
    arr_indices = x.indices
  for buf, idx in safe_zip(bufs, arr_indices):
    candidates[_hashable_index(idx)].append(buf)

  bufs = []
  for idx, device in safe_zip(indices, devices):
    # Look up all buffers that contain the correct slice of the logical array.
    candidates_list = candidates[_hashable_index(idx)]
    if not candidates_list:
      # This array isn't sharded correctly. Reshard it via host roundtrip.
      # TODO(skye): more efficient reshard?
      return shard_arg_handlers[type(x._value)](
          x._value, devices, indices, sharding)
    # Try to find a candidate buffer already on the correct device,
    # otherwise copy one of them.
    for buf in candidates_list:
      if buf.device() == device:
        bufs.append(buf)
        break
    else:
      bufs.append(buf)

  return batched_device_put(x.aval, sharding, bufs, devices)

### the xla_pmap primitive and its rules are comparable to xla_call in xla.py


def xla_pmap_impl_lazy(
    fun: lu.WrappedFun,
    *args,
    backend: Optional[str],
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Optional[Sequence[Any]],
    name: str,
    in_axes: Sequence[Optional[int]],
    out_axes_thunk: Callable[[], Sequence[Optional[int]]],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
) -> Callable:
  if (config.jax_disable_jit and config.jax_eager_pmap and
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
  if config.jax_distributed_debug:
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
  backend: Optional[str]
  devices: Optional[Sequence[Any]]

def _emap_impl(fun: lu.WrappedFun, *args,
               backend: Optional[str],
               axis_name: core.AxisName,
               axis_size: int,
               global_axis_size: int,
               devices: Optional[Sequence[Any]],
               name: str,
               in_axes: Sequence[Optional[int]],
               out_axes_thunk: Callable[[], Sequence[Optional[int]]],
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

def _map_schedule(idx: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
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
@lru_cache()
def _multi_pmap(f: Callable, info: EmapInfo, names: List[core.AxisName],
                all_axes: List[Tuple[Optional[int], ...]]
                ) -> Tuple[Callable, Dict[core.AxisName, int]]:
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
    if not config.jax_disable_jit:
      bind = HashableFunction(
          lambda *args, **kwargs: map_primitive.bind(fun, *args, **kwargs),
          (map_primitive, fun))
      fake_primitive = FakePrimitive(multiple_results=True, bind=bind)
      return self.process_primitive(fake_primitive, tracers, params)
    axis_name, in_axes, out_axes_thunk, axis_size = (params["axis_name"],
        params["in_axes"], params["out_axes_thunk"], params["axis_size"])
    vals, shard_axes = unzip2([(t.val, t.shard_axes) for t in tracers])
    shard_axes = [{axis_name: _annot_to_flat(np.ndim(v), s.values(), ax), **s}
                  if ax is not None else s
                  for v, ax, s in zip(vals, in_axes, shard_axes)]
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
    bind = HashableFunction(
        lambda *args, **kwargs: prim.bind(
            fun, jvp, *args, symbolic_zeros=symbolic_zeros, **kwargs),
        (prim, fun, jvp, symbolic_zeros))
    fake_primitive = FakePrimitive(multiple_results=True, bind=bind)
    return self.process_primitive(fake_primitive, tracers, {})

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,
                              out_trees, symbolic_zeros):
    bind = HashableFunction(
        lambda *args, **kwargs: primitive.bind(
            fun, fwd, bwd, *args, out_trees=out_trees,
            symbolic_zeros=symbolic_zeros, **kwargs),
        (primitive, fun, fwd, bwd))
    fake_primitive = FakePrimitive(multiple_results=True, bind=bind)
    return self.process_primitive(fake_primitive, tracers, {})

  def process_axis_index(self, frame):
    bind = HashableFunction(
        lambda _: jax.lax.axis_index(frame.name),
        (jax.lax.axis_index, frame.name))
    fake_primitive = FakePrimitive(multiple_results=False, bind=bind)
    with core.eval_context():
      range = jax.lax.iota(np.int32, frame.size)
    dummy_tracer = MapTracer(self, range, {frame.name: 0})
    return self.process_primitive(fake_primitive, (dummy_tracer,), {})

def _annot_to_flat(ndim: int, mapped_axes: Iterable[int],
                 annotation: Optional[int]) -> Optional[int]:
  if annotation is None: return None
  mapped_axes_ = set(mapped_axes)
  return [i for i in range(ndim) if i not in mapped_axes_][annotation]

def _match_annot(axis_name: core.AxisName, axis_size: int, val: Any,
                 shard_axis_src: Dict[core.AxisName, int],
                 dst_annotation: Optional[int]
                 ) -> Tuple[Any, Dict[core.AxisName, int]]:
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

def _moveaxis(ndim: int, shard_axes: Dict[core.AxisName, int],
              src: int, dst: int) -> Dict[core.AxisName, int]:
  lst: List[Optional[core.AxisName]] = [None] * ndim
  for k, v in shard_axes.items():
    lst[v] = k
  name = lst.pop(src)
  lst.insert(dst - (src < dst), name)
  return {name: i for i, name in enumerate(lst) if name is not None}

class MapTracer(core.Tracer):
  __slots__ = ["val", "shard_axes"]

  def __init__(self, trace: MapTrace, val, shard_axes: Dict[core.AxisName, int]):
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
                      backend_name: Optional[str],
                      axis_name: core.AxisName,
                      axis_size: int,
                      global_axis_size: int,
                      devices: Optional[Sequence[Any]],
                      name: str,
                      in_axes: Sequence[Optional[int]],
                      out_axes_thunk: Callable[[], Sequence[Optional[int]]],
                      donated_invars: Sequence[bool],
                      is_explicit_global_axis_size: bool,
                      *avals):
  pmap_computation = lower_parallel_callable(
      fun, backend_name, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, out_axes_thunk, donated_invars,
      is_explicit_global_axis_size, avals, lowering_platform=None)
  pmap_executable = pmap_computation.compile()
  return WeakRefList([pmap_executable.unsafe_call, pmap_executable.fingerprint])


@dataclasses.dataclass(frozen=True)
class ParallelCallableInfo:
  name: str
  backend: xc.Client
  axis_name: core.AxisName
  axis_size: int
  global_axis_size: int
  devices: Optional[Sequence[xc.Device]]
  in_axes: Iterable[Optional[int]]
  out_axes_thunk: Callable[[], Sequence[Optional[int]]]
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


def stage_parallel_callable(
    pci: ParallelCallableInfo, fun: lu.WrappedFun
) -> Tuple[core.Jaxpr, List[Any], ReplicaInfo, ShardInfo]:
  sharded_avals = tuple(
      shard_aval(pci.axis_size, axis, aval) if axis is not None else aval
      for axis, aval in safe_zip(pci.in_axes, pci.avals))

  with core.extend_axis_env(pci.axis_name, pci.global_axis_size, None):  # type: ignore
    with dispatch.log_elapsed_time(
        "Finished tracing + transforming {fun_name} for pmap in {elapsed_time} sec",
        fun_name=fun.__name__, event=dispatch.JAXPR_TRACE_EVENT):
      jaxpr, out_sharded_avals, consts = pe.trace_to_jaxpr_final(
          fun, sharded_avals, pe.debug_info_final(fun, "pmap"))
  jaxpr = api_util.jaxpr_debug_info(jaxpr, fun.debug_info)
  jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)

  assert len(out_sharded_avals) == len(pci.out_axes), (
      len(out_sharded_avals), len(pci.out_axes))

  # TODO(skye,mattjj): allow more collectives on multi-host as we test them, but
  # for now raise an error
  if pci.devices is not None:
    is_multi_host_pmap = len(pci.local_devices) != len(pci.devices)
  else:
    is_multi_host_pmap = xb.process_count(pci.backend) > 1
  if is_multi_host_pmap:
    check_multihost_collective_allowlist(jaxpr)

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
    backend_name: Optional[str],
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: int,
    devices: Optional[Sequence[xc.Device]],
    name: str,
    in_axes: Iterable[Optional[int]],
    out_axes_thunk: Callable[[], Sequence[Optional[int]]],
    donated_invars: Sequence[bool],
    is_explicit_global_axis_size: bool,
    avals: Sequence[core.AbstractValue],
    *,
    lowering_platform: Optional[str]):
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

  log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
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
          ordered_effects,
          backend,
          lowering_platform or backend.platform,
          sharding_impls.ReplicaAxisContext(axis_env),
          name_stack,
          donated_invars,
          replicated_args=replicated_args,
          arg_shardings=None,
          result_shardings=None,
          arg_names=jaxpr.debug_info and jaxpr.debug_info.arg_names,
          result_names=jaxpr.debug_info and jaxpr.debug_info.result_paths,
          num_replicas=replicas.num_global_replicas)
  return PmapComputation(lowering_result.module, pci=pci, replicas=replicas,
                         shards=shards, tuple_args=tuple_args,
                         unordered_effects=unordered_effects,
                         ordered_effects=ordered_effects,
                         keepalive=lowering_result.keepalive,
                         host_callbacks=lowering_result.host_callbacks,
                         jaxpr_debug_info=closed_jaxpr.jaxpr.debug_info)


class PmapComputation(stages.XlaLowering):
  _hlo: ir.Module
  _executable: Optional[PmapExecutable]

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
  unordered_effects: List[core.Effect]
  ordered_effects: List[core.Effect]
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
    handle_args = InputsHandler(self.compiled.local_devices(),
                                self.input_shardings, input_indices)
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
               unordered_effects: List[core.Effect],
               ordered_effects: List[core.Effect],
               host_callbacks: List[Any],
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
    compile_options = xb.get_compile_options(
        num_replicas=replicas.num_global_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=False,
        env_options_overrides=compiler_options,
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
            core.unmapped_aval(pci.axis_size, pci.axis_name, out_axis, aval))
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
      compiled = dispatch.compile_or_get_cached(
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


multi_host_supported_collectives: Set[core.Primitive] = set()


def check_multihost_collective_allowlist(jaxpr):
  used_collectives = set(xla.jaxpr_collectives(jaxpr))
  if not used_collectives.issubset(multi_host_supported_collectives):
    bad_collectives = used_collectives - multi_host_supported_collectives
    msg = "using collectives that aren't supported for multi-host: {}"
    raise TypeError(msg.format(", ".join(map(str, bad_collectives))))


class InputsHandler:
  __slots__ = ("handler", "local_devices", "in_shardings", "input_indices")

  def __init__(self, local_devices, in_shardings, input_indices):
    self.handler = partial(
        shard_args, local_devices, input_indices, in_shardings)
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
  # `out_avals` is the `GlobalDeviceArray` global avals when using pjit or xmap
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


@profiler.annotate_function
def replicate(val, axis_size, nrep, devices=None, backend=None, in_axis=0):
  """Replicates ``val`` across multiple devices.

  Args:
    val: the value to be replicated.
    axis_size: the length of the output, i.e. the logical number of replicas to
    create. Usually equal to `nrep`, but in the case of nested pmaps, `nrep` may
    be a multiple of `axis_size`.
    nrep: the number of replicas to create. If ``devices`` is set, must be equal
      to ``len(devices)``.
    devices: the devices to replicate across. If None, ``nrep`` will be used to
      generate a default device assignment.
    backend: string specifying which backend to use.
    in_axis: axis along which the value is to be replciated.

  Returns:
    A ShardedDeviceArray of length `axis_size` where each shard is equal to
    ``val``.
  """
  device_count = (len(devices) if devices else xb.local_device_count(backend))
  if nrep > device_count:
    msg = ("Cannot replicate across %d replicas because only %d local devices "
           "are available." % (nrep, device_count))
    if devices:
      msg += (" (local devices = %s)"
              % ", ".join(map(str, devices)) if devices else str(None))
    raise ValueError(msg)

  if devices is None:
    assert nrep is not None
    # TODO(skye): use different device assignment on multihost
    devices = xb.get_backend(backend).get_default_device_assignment(nrep)
  assert nrep == len(devices)

  aval = xla.abstractify(val)
  if in_axis is not None:
    replicated_aval = aval.update(shape=(axis_size,) + aval.shape)
  else:
    replicated_aval = aval
  # TODO(skye): figure out how partitioning should work here
  sharding_spec = sharding_specs.pmap_sharding_spec(
      nrep, axis_size, aval.shape, in_axis)

  buf = jax.device_put(val, devices[0])
  sharding = sharding_impls.PmapSharding(
      np.asarray([d for d in devices]), sharding_spec)
  return batched_device_put(replicated_aval, sharding, [buf] * len(devices),
                            devices)


class ExecuteReplicated:
  """The logic to shard inputs, execute a replicated model, returning outputs."""
  __slots__ = ['xla_executable', 'name', 'backend', 'in_handler', 'out_handler',
               'has_unordered_effects', 'ordered_effects', 'keepalive',
               'has_host_callbacks', '_local_devices', 'kept_var_idx',
               '__weakref__']

  def __init__(self, xla_executable, name, backend, in_handler: InputsHandler,
               out_handler: ResultsHandler,
               unordered_effects: List[core.Effect],
               ordered_effects: List[core.Effect], keepalive: Any,
               has_host_callbacks: bool, kept_var_idx: Set[int]):
    self.xla_executable = xla_executable
    self.name = name
    self.backend = backend
    self.in_handler = in_handler
    self.out_handler = out_handler
    self.has_unordered_effects = bool(unordered_effects)
    self.ordered_effects = ordered_effects
    self._local_devices = self.xla_executable.local_devices()
    if ordered_effects:
      assert len(self._local_devices) == 1
    self.keepalive = keepalive
    self.has_host_callbacks = has_host_callbacks
    self.kept_var_idx = kept_var_idx

  def _add_tokens_to_inputs(self, input_bufs):
    if self.ordered_effects:
      device, = self._local_devices
      tokens = [list(dispatch.runtime_tokens.get_token(eff, device))
                for eff in self.ordered_effects]
      input_bufs = [*tokens, *input_bufs]
    return input_bufs

  def _handle_token_bufs(self, token_bufs, sharded_token):
    for i, device in enumerate(self._local_devices):
      dispatch.runtime_tokens.set_output_runtime_token(
          device, sharded_token.get_token(i))
    for eff, token_buf in zip(self.ordered_effects, token_bufs):
      dispatch.runtime_tokens.update_token(eff, token_buf)

  def _call_with_tokens(self, input_bufs):
    input_bufs = self._add_tokens_to_inputs(input_bufs)
    out_bufs, sharded_token = (
        self.xla_executable.execute_sharded_on_local_devices_with_tokens(
            input_bufs
        )
    )
    num_output_tokens = len(self.ordered_effects)
    token_bufs, out_bufs = util.split_list(out_bufs, [num_output_tokens])
    self._handle_token_bufs(token_bufs, sharded_token)
    return out_bufs

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
      self._handle_token_bufs(
          results.disassemble_prefix_into_single_device_arrays(
              len(self.ordered_effects)),
          results.consume_token())
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


# Set param update handlers to update `donated_invars` just like xla_call_p
pe.call_param_updaters[xla_pmap_p] = xla.xla_call_partial_eval_update_params
pe.partial_eval_jaxpr_custom_rules[xla_pmap_p] = \
    partial(pe.call_partial_eval_custom_rule,
            'call_jaxpr', _pmap_partial_eval_custom_params_updater,
            res_aval=_pmap_partial_eval_custom_res_maker)
pe.dce_rules[xla_pmap_p] = _pmap_dce_rule
ad.call_param_updaters[xla_pmap_p] = xla.xla_call_jvp_update_params
ad.call_transpose_param_updaters[xla_pmap_p] = xla.xla_call_transpose_update_params

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
  return hlo.RemOp(
      hlo.DivOp(hlo.ReplicaIdOp().result, div).result, mod).result

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
    dynamic_slice_result = hlo.DynamicSliceOp(
        x, idxs, mlir.dense_int_elements(dims_unsqueezed)).result
    return [
      hlo.ReshapeOp(mlir.aval_to_ir_type(aval), dynamic_slice_result).result
    ]
  else:
    raise TypeError(aval)


# TODO(b/110096942): more efficient gather
def _hlo_unshard(ctx: mlir.LoweringRuleContext, aval, axis_env, out_axis, xs, platform):
  if aval is core.abstract_token:
    return xs
  elif isinstance(aval, core.ShapedArray):
    x, = xs
    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    convert_bool = (np.issubdtype(aval.dtype, np.bool_)
                    and platform in ('cpu', 'gpu'))
    if convert_bool:
      aval = aval.update(dtype=np.dtype(np.float32))
      x = hlo.ConvertOp(mlir.aval_to_ir_type(aval), x).result

    dims = list(aval.shape)
    padded_aval = aval.update(shape=[axis_env.sizes[-1]] + dims)
    padded = mlir.full_like_aval(ctx, 0, padded_aval)
    zero = mlir.ir_constant(np.zeros((), dtype=np.uint32))
    idxs = [_unravel_index_hlo(axis_env)] + [zero] * len(dims)
    broadcast_result = hlo.BroadcastOp(
        x, mlir.dense_int_elements([1])).result
    padded = hlo.DynamicUpdateSliceOp(padded, broadcast_result, idxs).result
    replica_groups = mlir.dense_int_elements(
      xla.axis_groups(axis_env, axis_env.names[-1]))
    out = hlo.CrossReplicaSumOp(padded, replica_groups).result
    if out_axis != 0:
      # TODO(apaszke,mattjj): Change the indices to DynamicUpdateSlice instead
      perm = list(range(1, len(dims)))
      perm.insert(out_axis, 0)
      transposed_dims = list(dims)
      transposed_dims.insert(out_axis, axis_env.sizes[-1])
      aval = aval.update(shape=transposed_dims)
      out = hlo.TransposeOp(out, mlir.dense_int_elements(perm)).result

    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    if convert_bool:
      float_zero = mlir.full_like_aval(ctx, 0, padded_aval)
      out = hlo.CompareOp(
          out,
          float_zero,
          hlo.ComparisonDirectionAttr.get("NE"),
          compare_type=hlo.ComparisonTypeAttr.get("FLOAT")).result
    return out
  else:
    raise TypeError(aval)


def _pmap_lowering(ctx, *in_nodes, axis_name,
                   axis_size, global_axis_size, devices, name,
                   call_jaxpr, backend=None, in_axes, out_axes,
                   donated_invars, is_explicit_global_axis_size):
  del donated_invars  # Unused.
  xla.check_backend_matches(backend, ctx.module_context.platform)
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if ctx.module_context.axis_env.names and devices is not None:
    raise ValueError("Nested pmap with explicit devices argument.")
  new_env = xla.extend_axis_env(ctx.module_context.axis_env, axis_name,
                                global_axis_size)
  # Shard the in_nodes that are mapped
  in_avals = [v.aval for v in call_jaxpr.invars]
  in_nodes_sharded = (
    _hlo_shard(aval, new_env, mlir.wrap_singleton_ir_values(in_node), in_axis)
    if in_axis is not None else mlir.wrap_singleton_ir_values(in_node)
    for aval, in_node, in_axis in zip(in_avals, in_nodes, in_axes))

  with maybe_extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
    sub_ctx = ctx.module_context.replace(
        axis_context=sharding_impls.ReplicaAxisContext(new_env),
        name_stack=ctx.module_context.name_stack.extend(
            util.wrap_name(name, 'pmap')))
    sharded_outs, _ = mlir.jaxpr_subcomp(sub_ctx, call_jaxpr, mlir.TokenSet(), (),
                                         *in_nodes_sharded,
                                         dim_var_values=ctx.dim_var_values)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_hlo_unshard(ctx, aval, new_env, out_axis, shard,
                        platform=ctx.module_context.platform)
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


spmd_primitive_batchers: Dict[core.Primitive, Callable] = {}


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
    manual_axes_set: FrozenSet[sharding_impls.MeshAxisName], mesh: Mesh):
  """Create an OpSharding proto that declares all mesh axes from `axes` as manual
  and all others as replicated.
  """
  named_mesh_shape = mesh.shape
  mesh_shape = list(named_mesh_shape.values())
  axis_order = {axis: i for i, axis in enumerate(mesh.axis_names)}

  manual_axes = list(sorted(manual_axes_set, key=str))
  replicated_axes = list(axis for axis in mesh.axis_names if axis not in manual_axes_set)

  tad_perm = ([axis_order[a] for a in replicated_axes] +
              [axis_order[a] for a in manual_axes])
  tad_shape = [1] * aval.ndim
  tad_shape.append(math.prod([named_mesh_shape[a] for a in replicated_axes]))
  tad_shape.append(math.prod([named_mesh_shape[a] for a in manual_axes]))

  proto = xc.OpSharding()
  proto.type = xc.OpSharding.Type.OTHER
  proto.tile_assignment_dimensions = tad_shape
  proto.iota_reshape_dims = mesh_shape
  proto.iota_transpose_perm = tad_perm
  proto.last_tile_dims = [xc.OpSharding.Type.REPLICATED, xc.OpSharding.Type.MANUAL]
  return proto

@partial(mlir.register_lowering, full_to_shard_p)
def _full_to_shard_lowering(ctx, x, *, axes: ArrayMapping, mesh: Mesh,
                            manual_axes: FrozenSet[sharding_impls.MeshAxisName]):
  # TODO: Can we short-circuit for replicated values? Probably not.
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  sharding_proto = mesh_sharding_specs(mesh.shape, mesh.axis_names)(aval_in, axes).sharding_proto()
  unspecified_dims = set(range(aval_in.ndim)) - set(axes.values())
  sx = mlir.wrap_with_sharding_op(ctx, x, aval_in, sharding_proto, unspecified_dims=unspecified_dims)
  proto = manual_proto(aval_in, manual_axes, mesh)
  return mlir.wrap_with_full_to_shard_op(ctx, sx, aval_out, proto, unspecified_dims=unspecified_dims),

shard_to_full_p = core.Primitive('shard_to_full')

@shard_to_full_p.def_abstract_eval
def _shard_to_full_abstract_eval(x, axes, mesh, **_):
  # TODO: Assert x is a global aval! Or ideally check that it's global in dims from axes!
  return untile_aval_nd(mesh.shape, axes, x)

@partial(mlir.register_lowering, shard_to_full_p)
def _shard_to_full_lowering(ctx: mlir.LoweringRuleContext, x, *, axes: ArrayMapping, mesh: Mesh,
                            manual_axes: FrozenSet[sharding_impls.MeshAxisName]):
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  proto = manual_proto(aval_in, manual_axes, mesh)  # type: ignore
  unspecified_dims = set(range(aval_in.ndim)) - set(axes.values())  # type: ignore
  sx = mlir.wrap_with_sharding_op(ctx, x, aval_in, proto, unspecified_dims=unspecified_dims)
  sharding_proto = mesh_sharding_specs(mesh.shape, mesh.axis_names)(aval_out, axes).sharding_proto()
  return mlir.wrap_with_shard_to_full_op(ctx, sx, aval_out, sharding_proto, unspecified_dims),

@lu.transformation
def vtile_manual(manual_axes: FrozenSet[sharding_impls.MeshAxisName],
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
  manual_axes: FrozenSet[sharding_impls.MeshAxisName]

TilingMethod = Union[TileVectorize, TileManual]


def check_if_any_auto(
    shardings: Iterable[Union[sharding_impls.XLACompatibleSharding,
                              AUTO, UnspecifiedValue]]) -> bool:
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
  source_info: Optional[dispatch.SourceInfo]

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
    return "" if self.source_info is None else f" at {self.source_info.source_info}"

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


ShardingInfo = Tuple[
    Union[sharding_impls.XLACompatibleSharding, UnspecifiedValue, AUTO],
    MismatchType, Optional[Any]]  # Any is dispatch.SourceInfo to avoid circular imports


def _get_default_device() -> xc.Device:
  return config.jax_default_device or xb.local_devices()[0]


def _get_and_check_device_assignment(
    shardings: Iterable[ShardingInfo],
    devices: Optional[Sequence[xc.Device]],
) -> Tuple[xc.Client, Tuple[xc.Device, ...]]:
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

def cache_wrap(fn):
  _wrapped_with_lu_cache = lu.cache(fn)
  _wrapped_with_weakref_lru_cache = weakref_lru_cache(fn)
  def wrapped(f, *args, **kwargs):
    if isinstance(f, lu.WrappedFun):
      return _wrapped_with_lu_cache(f, *args, **kwargs)
    else:
      return _wrapped_with_weakref_lru_cache(f, *args, **kwargs)
  return wrapped


@cache_wrap
def _trace_to_jaxpr_and_dce(fun_or_jaxpr, global_in_avals, api_name, fun_name,
                            keep_unused, donated_invars, auto_spmd_lowering):
  name_stack = source_info_util.new_name_stack(wrap_name(fun_name, api_name))

  if isinstance(fun_or_jaxpr, lu.WrappedFun):
    with dispatch.log_elapsed_time(
        "Finished tracing + transforming {fun_name} in {elapsed_time} sec",
        fun_name=str(name_stack), event=dispatch.JAXPR_TRACE_EVENT):
      jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_final(
          fun_or_jaxpr, global_in_avals)
  else:
    assert isinstance(fun_or_jaxpr, core.ClosedJaxpr)
    jaxpr = fun_or_jaxpr.jaxpr
    global_out_avals = fun_or_jaxpr.out_avals
    consts = fun_or_jaxpr.consts

  if (keep_unused or auto_spmd_lowering or
      any(hasattr(a, "shape") and not core.is_constant_shape(a.shape)
          for a in global_in_avals)):
    kept_var_idx = set(range(len(global_in_avals)))
  else:
    jaxpr, kept_const_idx, kept_var_idx = dispatch._prune_unused_inputs(jaxpr)
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
  shardings: Tuple[Union[sharding_impls.GSPMDSharding, UnspecifiedValue], ...]

  def __hash__(self):
    return hash(tuple(
        s._op_sharding_hash if isinstance(s, sharding_impls.GSPMDSharding) else s  # type: ignore
        for s in self.shardings))

  def __eq__(self, other):
    if not isinstance(other, SemanticallyEqualShardings):
      return False
    return all(op_shardings.are_op_shardings_equal(s._op_sharding, o._op_sharding)
               if (isinstance(s, sharding_impls.GSPMDSharding) and
                   isinstance(o, sharding_impls.GSPMDSharding))
               else s == o for s, o in zip(self.shardings, other.shardings))


@weakref_lru_cache
def _cached_lowering_to_hlo(closed_jaxpr, api_name, fun_name, backend,
                            semantic_in_shardings, semantic_out_shardings,
                            da_object, lowering_platform,
                            donated_invars, name_stack):
  jaxpr = closed_jaxpr.jaxpr
  in_shardings = semantic_in_shardings.shardings
  out_shardings = semantic_out_shardings.shardings
  global_in_avals = closed_jaxpr.in_avals
  global_out_avals = closed_jaxpr.out_avals
  device_assignment = da_object.device_assignment

  log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
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
  dispatch.raise_warnings_or_errors_for_jit_of_pmap(
      nreps, backend, fun_name, jaxpr)

  in_mlir_shardings: Optional[List[Optional[sharding_impls.XLACompatibleSharding]]]
  out_mlir_shardings: Optional[List[Optional[sharding_impls.XLACompatibleSharding]]]
  axis_ctx: mlir.AxisContext

  if nreps == 1:
    in_mlir_shardings = map(_to_logical_sharding, global_in_avals, in_shardings)
    out_mlir_shardings = map(_to_logical_sharding, global_out_avals, out_shardings)
    replicated_args = [False] * len(global_in_avals)
    axis_ctx = sharding_impls.ShardingContext(device_assignment)
    num_partitions = len(device_assignment)
  else:
    # This path is triggered for `jit(pmap)` cases.
    replicated_args = None
    in_mlir_shardings = None
    out_mlir_shardings = None
    axis_env = sharding_impls.AxisEnv(nreps, (), ())
    axis_ctx = sharding_impls.ReplicaAxisContext(axis_env)
    num_partitions = 1

  module_name = f"{api_name}_{fun_name}"

  if len(device_assignment) > 1:
    if any(effects.ordered_effects.contains(eff) for eff
           in closed_jaxpr.effects):
      raise ValueError("Ordered effects are not supported for more than 1 device.")
  ordered_effects = list(effects.ordered_effects.filter_in(closed_jaxpr.effects))

  with dispatch.log_elapsed_time(
        "Finished jaxpr to MLIR module conversion {fun_name} in {elapsed_time} sec",
        fun_name=str(name_stack), event=dispatch.JAXPR_TO_MLIR_MODULE_EVENT):
    lowering_result = mlir.lower_jaxpr_to_module(
        module_name,
        closed_jaxpr,
        ordered_effects,
        backend,
        # Optionally, override the lowering platform
        lowering_platform or backend.platform,
        axis_ctx,
        name_stack,
        donated_invars,
        replicated_args=replicated_args,
        arg_shardings=in_mlir_shardings,
        result_shardings=out_mlir_shardings,
        arg_names=jaxpr.debug_info and jaxpr.debug_info.arg_names,
        result_names=jaxpr.debug_info and jaxpr.debug_info.result_paths,
        num_replicas=nreps,
        num_partitions=num_partitions)
  tuple_args = dispatch.should_tuple_args(len(global_in_avals), backend.platform)
  unordered_effects = list(
      effects.ordered_effects.filter_not_in(closed_jaxpr.effects))
  return (lowering_result.module, lowering_result.keepalive,
          lowering_result.host_callbacks, unordered_effects, ordered_effects,
          nreps, tuple_args)


@dataclasses.dataclass(frozen=True)
class _DeviceAssignment:
  device_assignment: Tuple[xc.Device, ...]

  @cached_property
  def _hash(self):
    return hash(self.device_assignment)

  def __hash__(self):
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, _DeviceAssignment):
      return False
    if id(self) == id(other):
      return True
    return (self.device_assignment == other.device_assignment)

  @cached_property
  def is_fully_addressable(self):
    return len(self.device_assignment) == len(self.addressable_device_assignment)

  @cached_property
  def addressable_device_assignment(self):
    return [d for d in self.device_assignment
            if d.process_index == d.client.process_index()]


@lru_cache(maxsize=2048)
def _create_da_object(
    device_assignment: Tuple[xc.Device, ...]) -> _DeviceAssignment:
  return _DeviceAssignment(device_assignment)


@profiler.annotate_function
def lower_sharding_computation(
    fun_or_jaxpr: Union[lu.WrappedFun, core.ClosedJaxpr],
    api_name: str,
    fun_name: str,
    in_shardings: Sequence[MaybeSharding],
    out_shardings: Union[Sequence[MaybeSharding], UnspecifiedValue],
    donated_invars: Sequence[bool],
    global_in_avals: Sequence[core.ShapedArray],
    *,
    keep_unused: bool,
    inline: bool,
    always_lower: bool,
    devices_from_context: Optional[Sequence[xc.Device]] = None,
    lowering_platform: Optional[str],
) -> MeshComputation:
  """Lowers a computation to XLA. It can take arbitrary shardings as input.

  The caller of this code can pass in a singleton UNSPECIFIED because the
  number of out_avals might not be known at that time and
  lower_sharding_computation calculates the number of out_avals so it can apply
  the singleton UNSPECIFIED to all out_avals.
  """
  # 1. Trace to jaxpr and preprocess/verify it
  auto_spmd_lowering = (
      check_if_any_auto(in_shardings) if is_unspecified(out_shardings) else
      check_if_any_auto(it.chain.from_iterable([in_shardings, out_shardings])))  # type: ignore

  (closed_jaxpr, global_in_avals, global_out_avals, donated_invars,
   kept_var_idx, name_stack) = _trace_to_jaxpr_and_dce(
      fun_or_jaxpr, global_in_avals, api_name, fun_name, keep_unused,
      donated_invars, auto_spmd_lowering)
  jaxpr = closed_jaxpr.jaxpr
  in_shardings = tuple(s for i, s in enumerate(in_shardings) if i in kept_var_idx)

  if is_unspecified(out_shardings):
    out_shardings = (UNSPECIFIED,) * len(global_out_avals)
  assert isinstance(out_shardings, tuple)
  assert len(out_shardings) == len(global_out_avals), (
      len(out_shardings), len(global_out_avals))

  # Device assignment across all inputs, outputs and shardings inside jaxpr
  # should be the same.
  jaxpr_sharding = list(dispatch.jaxpr_shardings(jaxpr))
  backend, device_assignment = _get_and_check_device_assignment(
      it.chain([(i, MismatchType.ARG_SHARDING, None) for i in in_shardings],
               [(o, MismatchType.OUT_SHARDING, None) for o in out_shardings],
               [(js, MismatchType.SHARDING_INSIDE_COMPUTATION, source_info)
                for js, source_info in jaxpr_sharding]),
      devices_from_context)

  committed = bool(
      devices_from_context or
      len(device_assignment) > 1 or
      any(not is_unspecified(i) for i in in_shardings) or
      any(not is_unspecified(js) for js, _ in jaxpr_sharding) or
      any(not is_unspecified(o) for o in out_shardings))

  gs = sharding_impls.GSPMDSharding.get_replicated(device_assignment)
  in_shardings = tuple(gs if is_unspecified(i) else i for i in in_shardings)

  da_object = _create_da_object(tuple(device_assignment))

  if not da_object.is_fully_addressable:
    check_multihost_collective_allowlist(jaxpr)
    if inline and config.jax_spmd_mode != 'allow_all':
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

  has_outfeed = core.jaxpr_uses_outfeed(jaxpr)
  kept_outputs = [True] * len(global_out_avals)

  # Computations that only produce constants and/or only rearrange their inputs,
  # which are often produced from partial evaluation, don't need compilation,
  # and don't need to evaluate their arguments.
  if (not always_lower and not (jaxpr.effects or has_outfeed) and
      (not jaxpr.eqns and all(kept_outputs) or not jaxpr.outvars) and
      all(is_unspecified(o) for o in out_shardings)):
    return MeshComputation(
        str(name_stack), None, True, donated_invars, jaxpr=jaxpr,
        consts=closed_jaxpr.consts, global_in_avals=global_in_avals,
        global_out_avals=global_out_avals, in_shardings=in_shardings,
        backend=backend, da_object=da_object,
        committed=committed, kept_var_idx=kept_var_idx, keepalive=None)

  # 2. Build up the HLO
  semantic_in_shardings = SemanticallyEqualShardings(in_shardings)  # type: ignore
  semantic_out_shardings = SemanticallyEqualShardings(out_shardings)
  (module, keepalive, host_callbacks, unordered_effects, ordered_effects,
   nreps, tuple_args) = _cached_lowering_to_hlo(
       closed_jaxpr, api_name, fun_name, backend, semantic_in_shardings,
       semantic_out_shardings, da_object, lowering_platform,
       donated_invars, name_stack)

  # backend and device_assignment is passed through to MeshExecutable because
  # if keep_unused=False and all in_shardings are pruned, then there is no way
  # to get the device_assignment and backend. So pass it to MeshExecutable
  # because we calculate the device_assignment and backend before in_shardings,
  # etc are pruned.
  return MeshComputation(
      str(name_stack),
      module,
      False,
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
      pmap_nreps=nreps,
      jaxpr_debug_info=closed_jaxpr.jaxpr.debug_info)


def _to_logical_sharding(
    aval: core.AbstractValue, sharding: Union[MaybeSharding, AUTO]
) -> Optional[sharding_impls.XLACompatibleSharding]:
  if is_unspecified(sharding) or is_auto(sharding):
    return None
  elif isinstance(aval, ShapedArray):
    assert isinstance(sharding, sharding_impls.XLACompatibleSharding)
    return sharding
  elif isinstance(aval, core.AbstractToken):
    return None
  else:
    raise TypeError(aval)


@profiler.annotate_function
def lower_mesh_computation(
    fun_or_jaxpr: Union[lu.WrappedFun, core.ClosedJaxpr],
    api_name: str,
    fun_name: str,
    mesh: Mesh,
    in_shardings: Sequence[Union[sharding_impls.NamedSharding, AUTO]],
    out_shardings: Sequence[Union[sharding_impls.NamedSharding, AUTO,
                                  UnspecifiedValue]],
    donated_invars: Sequence[bool],
    spmd_lowering: bool,
    global_in_avals: Sequence[core.ShapedArray],
    tiling_method: Optional[TilingMethod],
    lowering_platform: Optional[str]) -> MeshComputation:
  assert not mesh.empty
  backend = xb.get_device_backend(mesh.devices.flat[0])
  name_stack = source_info_util.new_name_stack(wrap_name(fun_name, api_name))

  global_axis_sizes = mesh.shape

  log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
  if logger.isEnabledFor(log_priority):
    logger.log(log_priority,
               "Compiling %s for %s mesh with global shapes and types %s. "
               "Argument mapping: %s.",
               fun_name, tuple(global_axis_sizes.items()), global_in_avals,
               in_shardings)

  # 1. Trace to jaxpr and preprocess/verify it
  if spmd_lowering:
    manual_axes: FrozenSet[MeshAxisName] = frozenset()
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

  assert len(out_shardings) == len(out_jaxpr_avals)
  if spmd_lowering:
    global_out_avals = out_jaxpr_avals
  else:
    # In non-spmd lowering path, there is no `AUTO` or `UNSPECIFIED`, which is
    # why `.spec` can be accessed.
    global_out_avals = [untile_aval_nd(global_axis_sizes, get_array_mapping(o.spec), aval)  # type: ignore
                        for aval, o in safe_zip(out_jaxpr_avals, out_shardings)]

  _sanitize_mesh_jaxpr(jaxpr)
  if mesh.is_multi_process:
    check_multihost_collective_allowlist(jaxpr)
  jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)

  # 2. Build up the HLO
  tuple_args = dispatch.should_tuple_args(len(in_jaxpr_avals), backend.platform)

  in_partitions: Optional[List[Optional[sharding_impls.XLACompatibleSharding]]]
  out_partitions: Optional[List[Optional[sharding_impls.XLACompatibleSharding]]]
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
          ordered_effects,
          backend,
          lowering_platform or backend.platform,
          axis_ctx,
          name_stack,
          donated_invars,
          replicated_args=replicated_args,
          arg_shardings=in_partitions,
          result_shardings=out_partitions,
          arg_names=jaxpr.debug_info and jaxpr.debug_info.arg_names,
          result_names=jaxpr.debug_info and jaxpr.debug_info.result_paths,
          num_replicas=num_replicas,
          num_partitions=num_partitions)

  return MeshComputation(
      str(name_stack),
      lowering_result.module,
      False,
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
      jaxpr_debug_info=closed_jaxpr.jaxpr.debug_info)

class MeshComputation(stages.XlaLowering):
  _hlo: Optional[ir.Module]
  _executable: Optional[MeshExecutable]

  def __init__(self, name: str, hlo: Optional[ir.Module],
               is_trivial: bool, donated_invars: Sequence[bool], **compile_args):
    self._name = name
    self._hlo = hlo
    self.is_trivial = is_trivial
    self._donated_invars = donated_invars
    self.compile_args = compile_args
    self._executable = None

  # -- stages.XlaLowering overrides

  def stablehlo(self) -> ir.Module:
    if self.is_trivial:
      raise ValueError("A trivial computation has no HLO")
    return self._hlo

  def compile(
      self,
      compiler_options=None,
  ) -> MeshExecutable:
    if self._executable is None or compiler_options is not None:
      if self.is_trivial:
        executable = MeshExecutable.from_trivial_jaxpr(
            **self.compile_args)
      else:
        executable = UnloadedMeshExecutable.from_hlo(
            self._name,
            self._hlo,
            **self.compile_args,
            compiler_options=compiler_options)
      if compiler_options is None:
        self._executable = executable
      return executable
    return self._executable

  def cost_analysis(self) -> Dict[str, float]:
    backend = self.compile_args["backend"]
    if xb.using_pjrt_c_api(backend):
      raise NotImplementedError(
          "Lowered.cost_analysis not implemented on platform "
          f"'{backend.platform}'. Use compile().cost_analysis() for "
          "post-compilation cost estimates.")
    return xe.hlo_module_cost_analysis(backend, self.hlo().as_hlo_module())


@lru_cache(maxsize=1024)
def _get_replicated_slices(num_addressable_devices: int, ndim: Optional[int]):
  if ndim is None:
    return ((slice(None),),) * num_addressable_devices
  else:
    return ((slice(None),) * ndim,) * num_addressable_devices


def _get_input_indices(
    avals: Sequence[ShapedArray],
    shardings: Sequence[sharding_impls.XLACompatibleSharding],
    da_object: Union[_DeviceAssignment, Sequence[xc.Device]],
) -> Sequence[Tuple[Optional[Index], ...]]:

  input_indices = []
  if isinstance(da_object, _DeviceAssignment):
    num_addressable_devices = len(da_object.addressable_device_assignment)
  else:
    num_addressable_devices = len(
        [d for d in da_object if d.process_index == d.client.process_index()])

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
    xla_executable, device_assignment: Sequence[xc.Device],
    num_in_avals: int, num_out_avals: int
) -> Tuple[Sequence[sharding_impls.XLACompatibleSharding],
           Sequence[sharding_impls.XLACompatibleSharding]]:
  from jax.experimental import pjit

  # When the device assignment only has 1 device, SPMD partitioner will not run.
  # Hence the op shardings will not be set on the `hlo_module`. In that case,
  # just return SingleDeviceShardings since we know the computation is running
  # only on 1 device.
  if len(device_assignment) == 1:
    ss = sharding_impls.SingleDeviceSharding(device_assignment[0])
    return [ss] * num_in_avals, [ss] * num_out_avals

  in_op_shardings, out_op_shardings = pjit._get_op_sharding_from_executable(xla_executable)

  in_shardings_xla = [sharding_impls.GSPMDSharding(device_assignment, i)
                      for i in in_op_shardings]
  out_shardings_xla = [sharding_impls.GSPMDSharding(device_assignment, o)
                       for o in out_op_shardings]
  # This condition happens when all the elements in the output tuple have the
  # same sharding, so XLA decides to run the `FusionTupleDeduplicator` to
  # put the sharding on ROOT instead of the tuple.
  # TODO(b/245667823): Remove this when XLA fixes this.
  if len(out_shardings_xla) == 1 and len(out_shardings_xla) < num_out_avals:
    out_shardings_xla = out_shardings_xla * num_out_avals
  assert len(out_shardings_xla) == num_out_avals, (
      len(out_shardings_xla), num_out_avals)
  return in_shardings_xla, out_shardings_xla


# TODO(yashkatariya): Remove this function after `AUTO` can return shardings
# without mesh.
def _get_mesh_pspec_shardings_from_executable(
    xla_executable, mesh: Mesh
) -> Tuple[Sequence[sharding_impls.NamedSharding],
           Sequence[sharding_impls.NamedSharding]]:
  from jax.experimental import pjit

  in_pspec, out_pspec = pjit._get_pspec_from_executable(xla_executable, mesh)
  return ([sharding_impls.NamedSharding(mesh, i) for i in in_pspec],
          [sharding_impls.NamedSharding(mesh, o) for o in out_pspec])


SubClassT = TypeVar("SubClassT", bound=sharding_impls.XLACompatibleSharding)
OrigHandlerType = Dict[Type[SubClassT],
                       Callable[[xc.OpSharding, SubClassT], SubClassT]]

orig_out_sharding_handlers: OrigHandlerType = {}

def _gspmd_to_named_sharding(
    op_sharding: xc.OpSharding,
    self: sharding_impls.NamedSharding) -> sharding_impls.NamedSharding:
  parsed_pspec = sharding_impls.parse_flatten_op_sharding(
      op_sharding, self.mesh)[0]
  return create_mesh_pspec_sharding(
      self.mesh, parsed_pspec.get_partition_spec(), parsed_pspec)
orig_out_sharding_handlers[sharding_impls.NamedSharding] = _gspmd_to_named_sharding


def _gspmd_to_positional_sharding(
    op_sharding: xc.OpSharding,
    self: sharding_impls.PositionalSharding) -> sharding_impls.PositionalSharding:
  return sharding_impls._from_op_sharding_to_pos_sharding(
      op_sharding, self._device_assignment)
orig_out_sharding_handlers[sharding_impls.PositionalSharding] = _gspmd_to_positional_sharding


def _get_out_sharding_from_orig_sharding(
    out_shardings, out_avals, orig_s, orig_aval, are_out_sharding_from_xla):
  out = []
  orig_handler = orig_out_sharding_handlers[type(orig_s)]
  for o, out_aval, from_xla in safe_zip(out_shardings, out_avals,
                                        are_out_sharding_from_xla):
    if isinstance(o, sharding_impls.GSPMDSharding):
      try:
        # Only return the same input sharding object if the OpShardings and
        # in_aval.ndim and out_aval.ndim match. This is because if OpSharding is
        # replicated then, it doesn't encode the ndim in it. The devices
        # will be the same at this point because those checks happen before.
        if (orig_aval is not None and out_aval is not None and
            out_aval.ndim == orig_aval.ndim and
            sharding_impls.are_op_shardings_equal(
                o._op_sharding, orig_s._to_xla_op_sharding(orig_aval.ndim))):
          out.append((orig_s, False))
        else:
          out.append((orig_handler(o._op_sharding, orig_s), False))
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

  orig_s = None
  orig_aval = None
  for i, aval in safe_zip(in_shardings, in_avals):
    oi = getattr(i, '_original_sharding', None)
    if type(oi) in orig_out_sharding_handlers:
      orig_s = oi
      orig_aval = aval
      break
  if orig_s is not None:
    return zip(*_get_out_sharding_from_orig_sharding(
        out_shardings, out_avals, orig_s, orig_aval, are_out_shardings_from_xla))

  return out_shardings, are_out_shardings_from_xla


@weakref_lru_cache
def _cached_compilation(computation, name, mesh, spmd_lowering,
                        tuple_args, auto_spmd_lowering,
                        _allow_propagation_to_outputs, host_callbacks, backend,
                        da, pmap_nreps, compiler_options_keys,
                        compiler_options_values):
  device_assignment = da.device_assignment if isinstance(
      da, _DeviceAssignment) else da

  # TODO(phawkins): One would normally just write:
  # dev = np.array(device_assignment)
  # The formulation below is substantially faster if there are many devices.
  # If we were to optimize __getattr__ on xc.Device we might not need this
  # workaround.
  dev = np.vectorize(lambda i: device_assignment[i], otypes=[object])(
    np.arange(len(device_assignment))
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

  compile_options = xb.get_compile_options(
      num_replicas=num_replicas,
      num_partitions=num_partitions,
      device_assignment=xla_device_assignment,
      use_spmd_partitioning=spmd_lowering,
      use_auto_spmd_partitioning=auto_spmd_lowering,
      env_options_overrides=compiler_options,
  )

  opts = compile_options.executable_build_options
  if auto_spmd_lowering:
    assert mesh is not None
    opts.auto_spmd_partitioning_mesh_shape = list(mesh.shape.values())
    opts.auto_spmd_partitioning_mesh_ids = (
        sharding_specs.get_logical_mesh_ids(list(mesh.shape.values()))
        .reshape(-1))
  compile_options.parameter_is_tupled_arguments = tuple_args
  opts.allow_spmd_sharding_propagation_to_output = list(_allow_propagation_to_outputs)

  if hasattr(backend, "compile_replicated"):
    return None, compile_options

  with dispatch.log_elapsed_time(
      "Finished XLA compilation of {fun_name} in {elapsed_time} sec",
      fun_name=name, event=dispatch.BACKEND_COMPILE_EVENT):
    xla_executable = dispatch.compile_or_get_cached(
        backend, computation, dev, compile_options, host_callbacks)
  return xla_executable, compile_options


@dataclasses.dataclass
class UnloadedMeshExecutable:
  xla_executable: Any
  device_assignment: Union[_DeviceAssignment, Sequence[xc.Device]]
  backend: xb.XlaBackend
  input_avals: Sequence[ShapedArray]
  input_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  output_avals: Sequence[ShapedArray]
  output_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  committed: bool
  are_out_shardings_from_xla: Sequence[bool]
  name: str
  unordered_effects: List[core.Effect]
  ordered_effects: List[core.Effect]
  keepalive: Sequence[Any]
  host_callbacks: Sequence[Any]
  kept_var_idx: Set[int]
  auto_spmd_lowering: bool
  jaxpr_debug_info: Optional[core.JaxprDebugInfo]

  def build_unsafe_call(self):
    input_indices = _get_input_indices(self.input_avals, self.input_shardings,
                                       self.device_assignment)
    handle_args = InputsHandler(self.xla_executable.local_devices(),
                                self.input_shardings, input_indices)
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
                          self.jaxpr_debug_info, self)

  # May return a MeshExecutable in the compile_replicated case.
  @staticmethod
  def from_hlo(name: str,
               hlo: ir.Module,
               global_in_avals: Sequence[ShapedArray],
               global_out_avals: Sequence[ShapedArray],
               in_shardings: Sequence[Union[sharding_impls.XLACompatibleSharding, AUTO]],
               out_shardings: Sequence[Union[sharding_impls.XLACompatibleSharding, AUTO,
                                             UnspecifiedValue]],
               spmd_lowering: bool,
               tuple_args: bool,
               auto_spmd_lowering: bool,
               unordered_effects: List[core.Effect],
               ordered_effects: List[core.Effect],
               host_callbacks: List[Any],
               keepalive: Any,
               kept_var_idx: Set[int],
               backend: xb.XlaBackend,
               device_assignment: Union[_DeviceAssignment, Sequence[xc.Device]],
               committed: bool,
               pmap_nreps: int = 1,
               jaxpr_debug_info: Optional[core.JaxprDebugInfo] = None,
               compiler_options=None
  ) -> MeshExecutable:
    compiler_options_keys = tuple(
        compiler_options.keys()) if compiler_options is not None else None
    compiler_options_values = tuple(
        compiler_options.values()) if compiler_options is not None else None
    da = device_assignment if isinstance(
        device_assignment, _DeviceAssignment) else tuple(device_assignment)
    del device_assignment
    allow_prop_to_outputs = tuple(is_unspecified(o) for o in out_shardings)

    mesh = None
    if auto_spmd_lowering:
      for i in it.chain.from_iterable([in_shardings, out_shardings]):
        if is_auto(i):
          mesh = i.mesh  # type: ignore
          break

    xla_executable, compile_options = _cached_compilation(
        hlo, name, mesh, spmd_lowering,
        tuple_args, auto_spmd_lowering, allow_prop_to_outputs,
        tuple(host_callbacks), backend, da, pmap_nreps,
        compiler_options_keys, compiler_options_values)

    if hasattr(backend, "compile_replicated"):
      semantics_in_shardings = SemanticallyEqualShardings(in_shardings)  # type: ignore
      semantics_out_shardings = SemanticallyEqualShardings(out_shardings)  # type: ignore
      return _compile_replicated_mesh_executable_from_hlo(
          hlo, name, tuple(global_in_avals), tuple(global_out_avals),
          semantics_in_shardings, semantics_out_shardings, auto_spmd_lowering,
          compile_options, tuple(host_callbacks), bool(unordered_effects),
          tuple(ordered_effects), tuple(kept_var_idx), backend, da, committed,
          pmap_nreps, jaxpr_debug_info)

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
    elif (out_shardings and any(is_unspecified(o) for o in out_shardings)
          and pmap_nreps == 1):
      assert mesh is None
      device_assignment = da.device_assignment if isinstance(  # type: ignore
          da, _DeviceAssignment) else da
      _, out_shardings_xla = get_gspmd_shardings_from_executable(  # type: ignore
          xla_executable, device_assignment,  # type: ignore
          len(global_in_avals), len(global_out_avals))
      orig_out_shardings = out_shardings
      out_shardings, are_out_shardings_from_xla = [], []  # type: ignore
      for xla_s, orig, aval in safe_zip(out_shardings_xla, orig_out_shardings,
                                        global_out_avals):
        if is_unspecified(orig):
          out_shardings.append(xla_s)
          are_out_shardings_from_xla.append(True)
        else:
          if not op_shardings.are_op_shardings_equal(
              xla_s._to_xla_op_sharding(aval.ndim),  # type: ignore
              orig._to_xla_op_sharding(aval.ndim)):  # type: ignore
            raise AssertionError(
                f"Unexpected XLA sharding override: (XLA) {xla_s} != {orig} "
                "(User sharding)")
          out_shardings.append(orig)
          are_out_shardings_from_xla.append(False)
    else:
      are_out_shardings_from_xla = (False,) * len(global_out_avals)

    if pmap_nreps > 1:
      in_shardings, out_shardings, committed, da = _get_metadata_jit_pmap(
          xla_executable.local_devices(), len(in_shardings), len(out_shardings))

    out_shardings, are_out_shardings_from_xla = maybe_get_orig_out_sharding(
        in_shardings, out_shardings, are_out_shardings_from_xla,
        global_in_avals, global_out_avals)

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
        jaxpr_debug_info=jaxpr_debug_info).load()


class MeshExecutableFastpathData(NamedTuple):
  xla_executable: xc.LoadedExecutable
  out_pytree_def: Any
  in_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  out_shardings: Sequence[sharding_impls.XLACompatibleSharding]
  out_avals: Sequence[ShapedArray]
  out_committed: Sequence[bool]
  kept_var_bitvec: Iterable[bool]


class MeshExecutable(stages.XlaExecutable):
  __slots__ = [
      "xla_executable", "_unsafe_call", "build_unsafe_call", "in_avals",
      "_in_shardings", "_out_shardings", "_auto_spmd_lowering", "_kept_var_idx",
      "_jaxpr_debug_info", "_unloaded_executable",
  ]

  def __init__(self, xla_executable, build_unsafe_call, in_avals, in_shardings,
               out_shardings, auto_spmd_lowering, kept_var_idx,
               jaxpr_debug_info=None, unloaded_executable=None):
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
    self._jaxpr_debug_info = jaxpr_debug_info
    self._unloaded_executable = unloaded_executable

  @property
  def unsafe_call(self) -> Callable[..., Any]:
    if self._unsafe_call is None:
      self._unsafe_call = self.build_unsafe_call()
    return self._unsafe_call

  @staticmethod
  def from_trivial_jaxpr(jaxpr, consts, global_in_avals, global_out_avals,
                         in_shardings, backend, da_object,
                         committed, kept_var_idx, keepalive) -> MeshExecutable:
    assert keepalive is None
    if hasattr(backend, "compile_replicated"):
      return _compile_replicated_mesh_executable_from_trivial_jaxpr(
          jaxpr, consts, global_in_avals, global_out_avals, in_shardings,
          backend, da_object, committed, kept_var_idx, 1)

    out_shardings = _out_shardings_for_trivial(
        jaxpr, consts, in_shardings, da_object.device_assignment)
    indices = _get_input_indices(global_out_avals, out_shardings, da_object)
    local_device_assignment = da_object.addressable_device_assignment
    handle_ins = InputsHandler(local_device_assignment, out_shardings, indices)
    handle_outs = global_avals_to_results_handler(
          global_out_avals, out_shardings, committed,
          [False] * len(global_out_avals))
    unsafe_call = partial(_execute_trivial, jaxpr, consts, handle_ins,
                          handle_outs, kept_var_idx)
    return MeshExecutable(None, lambda: unsafe_call, global_in_avals,
                          in_shardings, out_shardings, False, kept_var_idx,
                          None)

  # -- stages.XlaExecutable overrides

  def xla_extension_executable(self):
    return self.xla_executable

  def call(self, *args):
    kept_args = [a for i, a in enumerate(args) if i in self._kept_var_idx]
    arg_avals = map(xla.abstractify, kept_args)
    ref_avals = self.in_avals
    check_arg_avals_for_call(ref_avals, arg_avals, self._jaxpr_debug_info)
    # Check the GDA sharding and the input sharding.
    check_gda_or_array_xla_sharding_match(kept_args, self._in_shardings,
                                          self._jaxpr_debug_info)
    return self.unsafe_call(*args)  # pylint: disable=not-callable

  def input_shardings(self) -> Sequence[sharding_impls.XLACompatibleSharding]:
    return self._in_shardings

  def output_shardings(self) -> Sequence[sharding_impls.XLACompatibleSharding]:
    return self._out_shardings

  def create_cpp_call(self, no_kwargs, in_tree, out_tree):
    if not (isinstance(self.unsafe_call, ExecuteReplicated) and
            not self.unsafe_call.has_unordered_effects and
            not self.unsafe_call.has_host_callbacks):
      return None

    def aot_cache_miss(*args, **kwargs):
      params = stages.CompiledCallParams(self, no_kwargs, in_tree, out_tree)
      outs, out_flat, args_flat = stages.Compiled.call(params, *args, **kwargs)
      use_fastpath = (all(isinstance(x, xc.ArrayImpl) for x in out_flat))

      if use_fastpath:
        out_avals = [o.aval for o in out_flat]
        out_committed = [o._committed for o in out_flat]
        kept_var_bitvec = [i in self._kept_var_idx
                           for i in range(len(args_flat))]
        fastpath_data = MeshExecutableFastpathData(
            self.xla_executable, out_tree, self._in_shardings,
            self._out_shardings, out_avals, out_committed, kept_var_bitvec)
      else:
        fastpath_data = None
      return outs, fastpath_data

    return xc._xla.pjit(self.unsafe_call.name, None, aot_cache_miss, [], [], [])  # type: ignore


def check_arg_avals_for_call(ref_avals, arg_avals,
                             jaxpr_debug_info: Optional[core.JaxprDebugInfo] = None):
  if len(ref_avals) != len(arg_avals):
    raise TypeError(
        f"Computation compiled for {len(ref_avals)} inputs "
        f"but called with {len(arg_avals)}")
  arg_names = ([''] * len(ref_avals) if jaxpr_debug_info is None else
               jaxpr_debug_info.arg_names)
  errors = []
  num_errors = 5
  for ref_aval, arg_aval, name in safe_zip(ref_avals, arg_avals, arg_names):
    if not core.typematch(ref_aval, arg_aval):
      errors.append(f"Compiled with {ref_aval} and called with {arg_aval} for "
                    f"arg {name}")
  if errors:
    str_errors = '\n'.join(errors[:num_errors])
    num_mismatch_str = (
        f'the {len(errors)} mismatches' if len(errors) < num_errors else
        f"{num_errors} mismatches out of {len(errors)}")
    raise TypeError(
        "Computation was compiled for different input types and called with "
        f"different types. Here are {num_mismatch_str}:\n{str_errors}")


def _get_metadata_jit_pmap(local_devices, num_in_shardings, num_out_shardings):
  # Create replicated shardings for jit(pmap) path with local devices
  # because multihost jit(pmap) is not allowed.
  gs = sharding_impls.GSPMDSharding.get_replicated(local_devices)
  in_shardings = [gs] * num_in_shardings
  out_shardings = [gs] * num_out_shardings
  # jit(pmap) will generate Arrays with multi-device sharding.
  # It is unsupported for these shardings to be uncommited, so force
  # the outputs to be committed.
  committed = True
  return in_shardings, out_shardings, committed, tuple(local_devices)


def _out_shardings_for_trivial(
    jaxpr: core.Jaxpr, consts: Sequence[Any],
    in_shardings: Sequence[sharding_impls.XLACompatibleSharding],
    device_assignment: Sequence[xc.Device],
  ) -> List[sharding_impls.XLACompatibleSharding]:
  # For each jaxpr output, compute a Sharding by:
  #   * if the output is a forwarded input, get the corresponding in_sharding;
  #   * if the output is a constant Array, get its .sharding attribute;
  #   * otherwise, the output is a literal or numpy.ndarray constant, so give it
  #     a replicated sharding
  from jax._src import array

  if len(device_assignment) > 1:
    rep = sharding_impls.GSPMDSharding.get_replicated(device_assignment)
    in_shardings = tuple(
        i._original_sharding if hasattr(i, '_original_sharding') else i
        for i in in_shardings)
  else:
    dev, = device_assignment
    rep = sharding_impls.SingleDeviceSharding(dev)
    in_shardings = (sharding_impls.SingleDeviceSharding(dev),) * len(in_shardings)

  shardings: Dict[core.Var, sharding_impls.XLACompatibleSharding] = {}
  for constvar, constval in zip(jaxpr.constvars, consts):
    if isinstance(constval, array.ArrayImpl):
      shardings[constvar] = constval.sharding
  map(shardings.setdefault, jaxpr.invars, in_shardings)
  return [rep if isinstance(x, core.Literal) else shardings.get(x, rep)
          for x in jaxpr.outvars]


def _execute_trivial(jaxpr, consts, in_handler, out_handler, kept_var_idx, *args):
  env: Dict[core.Var, Any]  = {}
  pruned_args = (x for i, x in enumerate(args) if i in kept_var_idx)
  map(env.setdefault, jaxpr.invars, pruned_args)
  map(env.setdefault, jaxpr.constvars, consts)
  outs = [xla.canonicalize_dtype(v.val) if type(v) is core.Literal else env[v]
          for v in jaxpr.outvars]
  return out_handler(in_handler(outs))


@weakref_lru_cache
def _compile_replicated_mesh_executable_from_hlo(
    computation, name, global_in_avals, global_out_avals, semantics_in_shardings,
    semantics_out_shardings, auto_spmd_lowering, compile_options,
    host_callbacks, has_unordered_effects, ordered_effects, kept_var_idx,
    backend, da, committed, pmap_nreps, jaxpr_debug_info):
  assert not auto_spmd_lowering
  in_shardings = semantics_in_shardings.shardings
  out_shardings = semantics_out_shardings.shardings

  input_indices = _get_input_indices(global_in_avals, in_shardings, da)  # type: ignore
  if pmap_nreps > 1:
    # For a jit wrapping a pmap, replicate each input index to match the
    # devices of the replicated jit computation.
    input_indices = [index * pmap_nreps for index in input_indices]
  kept_var_idx = set(kept_var_idx)
  # Will compute out_handler with executable information.
  unsafe_call = backend.compile_replicated(
      is_trivial=False, name=name, computation=computation,
      compile_options=compile_options, host_callbacks=host_callbacks,
      has_unordered_effects=has_unordered_effects,
      ordered_effects=ordered_effects, in_avals=global_in_avals,
      in_indices=input_indices, in_shardings=in_shardings,
      kept_var_idx=kept_var_idx,
      out_avals=global_out_avals, out_shardings=out_shardings,
      committed=committed, pmap_nreps=pmap_nreps)
  xla_executable = None
  return MeshExecutable(xla_executable, lambda: unsafe_call, global_in_avals,
                        in_shardings, out_shardings, auto_spmd_lowering,
                        kept_var_idx, jaxpr_debug_info, None)


def _compile_replicated_mesh_executable_from_trivial_jaxpr(
    jaxpr, consts, global_in_avals, global_out_avals, in_shardings, backend,
    da_object, committed, kept_var_idx, pmap_nreps):
  out_shardings = _out_shardings_for_trivial(
      jaxpr, consts, in_shardings, da_object.device_assignment)

  input_indices = _get_input_indices(global_in_avals, in_shardings, da_object)  # type: ignore
  handle_outs = global_avals_to_results_handler(
      global_out_avals, out_shardings, committed,
      [False] * len(global_out_avals))
  # Use the standard out_handler.
  unsafe_call = backend.compile_replicated(
      is_trivial=True, jaxpr=jaxpr, consts=consts,
      device_assignment=da_object.device_assignment, in_avals=global_in_avals,
      in_indices=input_indices, in_shardings=in_shardings,
      kept_var_idx=kept_var_idx, out_handler=handle_outs,
      out_shardings=out_shardings, pmap_nreps=pmap_nreps)
  return MeshExecutable(None, lambda: unsafe_call, global_in_avals,
                        in_shardings, out_shardings, False, kept_var_idx,
                        None)


@lru_cache()
def create_mesh_pspec_sharding(
    mesh: Mesh, pspec: Optional[PartitionSpec], parsed_pspec=None
) -> sharding_impls.NamedSharding:
  if pspec is None:
    pspec, parsed_pspec = PartitionSpec(), None
  return sharding_impls.NamedSharding(mesh, pspec, parsed_pspec)


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
    jaxpr_debug_info: Optional[core.JaxprDebugInfo]) -> None:
  from jax._src.array import ArrayImpl
  arg_names = ([''] * len(args) if jaxpr_debug_info is None else
               jaxpr_debug_info.arg_names)
  errors = []
  num_errors = 5
  for arg, xs, name in safe_zip(args, in_xla_shardings, arg_names):
    if not isinstance(arg, ArrayImpl):
      continue

    # No need to cache this check since MeshExecutable has a C++ fast path
    # for AOT compiled call.
    if (not check_device_backend_on_shardings([xs]) and
        arg._committed and
        not op_shardings.are_op_shardings_equal(
            arg.sharding._to_xla_op_sharding(arg.ndim),
            xs._to_xla_op_sharding(arg.ndim))):
      errors.append(
          f"Got Array sharding: {arg.sharding} and input sharding: {xs} for "
          f"arg {name} with shape: {arg.aval.str_short()}")

  if errors:
    str_errors = '\n'.join(errors[:num_errors])
    num_mismatch_str = (
        f'the {len(errors)} mismatches' if len(errors) < num_errors else
        f"{num_errors} mismatches out of {len(errors)}")
    raise ValueError(
          "Array(s) sharding does not match the input(s) sharding. "
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


custom_resource_typing_rules: Dict[core.Primitive, Callable] = {}

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


def mesh_sharding_specs(axis_sizes, axis_names, allow_uneven_axes=False):
  mesh_axis_pos = {name: i for i, name in enumerate(axis_names)}
  # NOTE: This takes in the non-sharded avals!
  def mk_sharding_spec(aval, aval_axes):
    if aval is core.abstract_token:
      assert not aval_axes
      return ShardingSpec([], [Replicated(axis_size) for axis_size in axis_sizes.values()])
    aval_shape = list(aval.shape)
    # NOTE: sorted is stable, which is important when multiple resources
    #       map to the same axis.
    for name, axis in sorted(aval_axes.items(), key=lambda x: x[1]):
      if not allow_uneven_axes:
        if aval_shape[axis] % axis_sizes[name] != 0:
          raise ValueError(
            f'The aval shape on dimension {axis} is {aval_shape[axis]} and '
            f'the size of axis {name} is {axis_sizes[name]}. The aval shape % '
            'axis size should be zero but got '
            f'{aval_shape[axis] % axis_sizes[name]}')
      aval_shape[axis] //= axis_sizes[name]
    return sharding_specs.make_sharding_spec(
        axis_sizes, mesh_axis_pos, len(aval.shape), aval_axes)
  return mk_sharding_spec


@contextmanager
def maybe_extend_axis_env(*args, **kwargs):
  with core.extend_axis_env(*args, **kwargs):
    yield


def device_put(x, devices: Sequence[xc.ArrayImpl],
               replicate: bool=False) -> List[xc.ArrayImpl]:
  """Call device_put on a sequence of devices and return a flat sequence of buffers."""
  if replicate:
    return [jax.device_put(x, device) for device in devices]
  else:
    return [jax.device_put(val, device) for val, device in safe_zip(x, devices)]

# TODO(phawkins): fix external users not to use these functions.
def _create_pmap_sharding_spec(aval, sharded_dim=0, sharded_dim_size=None):
  return sharding_specs.create_pmap_sharding_spec(
      aval.shape, sharded_dim, sharded_dim_size)

def _pmap_sharding_spec(nrep, axis_size, npart, parts,
                        sharded_aval, map_axis: Optional[int]) -> ShardingSpec:
  assert npart == 1, npart
  assert parts is None, parts
  return sharding_specs.pmap_sharding_spec(
      nrep, axis_size, sharded_aval.shape, map_axis)
