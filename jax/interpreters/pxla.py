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
"""Implementation of pmap and related functionality."""

# A ShardingSpec describes at a high level how a logical array is sharded across
# devices (each ShardedDeviceArray has a ShardingSpec, and ShardingSpecs also
# describe how to shard inputs to a parallel computation). spec_to_indices()
# encodes exactly how a given ShardingSpec is translated to device buffers, i.e.
# how the sharded array is "laid out" across devices. Given a sequence of
# devices, we shard the data across the devices in row-major order, with
# replication treated as an extra inner dimension.
#
# For example, given the logical data array [1, 2, 3, 4], if we were to
# partition this array 4 ways with a replication factor of 2, for a total of 8
# devices, the data on each device would be: [1, 1], [2, 2], [3, 3], [4, 4].
#
# This encoding is assumed by various parts of the system, e.g. generating
# replica groups for collective operations.

from contextlib import contextmanager
from collections import defaultdict, OrderedDict
import dataclasses
from functools import partial
import itertools as it
import operator as op
import threading
from typing import (Any, Callable, Dict, List, NamedTuple, Optional,
                    Sequence, Set, Tuple, Type, Union, Iterable)
import sys

from absl import logging
import numpy as np

from jax._src.config import config
from jax import core
from jax import linear_util as lu
from jax._src import abstract_arrays
from jax._src.abstract_arrays import array_types
from jax.core import ConcreteArray, ShapedArray
from jax._src import device_array
from jax._src import source_info_util
from jax._src import util
from jax._src.util import (unzip3, prod, safe_map, safe_zip,
                           extend_name_stack, wrap_name, assert_unreachable,
                           tuple_insert, tuple_delete, distributed_debug_log)
from jax.errors import JAXTypeError
from jax._src import dispatch
from jax._src import profiler
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.lib import pmap_lib
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax.tree_util import tree_flatten, tree_map
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import ad

# Built in Python lists don't support weak refs but subclasses of lists do.
class WeakRefList(list):
  pass

if sys.version_info >= (3, 8):
  from functools import cached_property as maybe_cached_property
else:
  maybe_cached_property = property

if sys.version_info >= (3, 9):
  OrderedDictType = OrderedDict
else:
  OrderedDictType = Dict

xops = xc.ops
xe = xc._xla

unsafe_map, map = map, safe_map  # type: ignore

Index = Union[int, slice, Tuple[Union[int, slice], ...]]

NoSharding = pmap_lib.NoSharding
Chunked = pmap_lib.Chunked
Unstacked = pmap_lib.Unstacked

ShardedAxis = pmap_lib.ShardedAxis
Replicated = pmap_lib.Replicated

_UNSHARDED_INSTANCE = NoSharding()
AvalDimSharding = Union[Unstacked, Chunked, NoSharding]
MeshDimAssignment = Union[ShardedAxis, Replicated]
ShardingSpec = pmap_lib.ShardingSpec


def sharding_spec_mesh_shape(self):
  sharded_axis_sizes = []
  for sharding in self.sharding:
    if isinstance(sharding, NoSharding):
      continue
    elif isinstance(sharding, Unstacked):
      sharded_axis_sizes.append(sharding.size)
    elif isinstance(sharding, Chunked):
      sharded_axis_sizes.extend(sharding.chunks)
    else:
      assert_unreachable(sharding)
  return tuple(sharded_axis_sizes[a.axis] if isinstance(a, ShardedAxis) else a.replicas
               for a in self.mesh_mapping)

def sharding_spec_sharding_proto(self):
  """Converts a ShardingSpec to an OpSharding proto.

  See
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto#L601
  for details on the OpSharding proto.
  Unfortunately the semantics are not very well described in the proto spec, but the code here might help:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/experimental/xla_sharding/xla_sharding.py
  """
  mesh_shape = self.mesh_shape
  mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)

  sharded_axes = {}  # maps sharded axis identifiers to mesh axis indices to which they're mapped
  replicated_maxes = []  # lists mesh axis identifiers to replicate over
  for maxis, assignment in enumerate(self.mesh_mapping):
    if isinstance(assignment, Replicated):
      replicated_maxes.append(maxis)
    elif isinstance(assignment, ShardedAxis):
      sharded_axes[assignment.axis] = maxis
    else:
      assert_unreachable(assignment)

  proto = xc.OpSharding()
  if len(replicated_maxes) == len(self.mesh_mapping):
    proto.type = xc.OpSharding.Type.REPLICATED
    return proto
  else:
    proto.type = xc.OpSharding.Type.OTHER

  mesh_permutation = []
  new_mesh_shape = []
  next_sharded_axis = 0
  for axis, sharding in enumerate(self.sharding):
    if isinstance(sharding, NoSharding):
      new_mesh_shape.append(1)  # Add a dummy mesh axis we won't be sharding over
    elif isinstance(sharding, Chunked):
      for nchunks in sharding.chunks:
        maxis = sharded_axes[next_sharded_axis]
        assert mesh_shape[maxis] == nchunks
        mesh_permutation.append(maxis)
        next_sharded_axis += 1
      new_mesh_shape.append(int(np.prod(sharding.chunks)))
    elif isinstance(sharding, Unstacked):
      raise RuntimeError("Cannot convert unstacked sharding specs to XLA OpSharding")
    else:
      assert_unreachable(sharding)

  # Create the partial sharding proto if tensor is replicated over some mesh axes
  if replicated_maxes:
    new_mesh_shape.append(-1)
    mesh_permutation.extend(replicated_maxes)
    proto.replicate_on_last_tile_dim = True

  proto_mesh = mesh.transpose(mesh_permutation).reshape(new_mesh_shape)
  proto.tile_assignment_dimensions = list(proto_mesh.shape)
  proto.tile_assignment_devices = list(proto_mesh.flat)
  return proto

def sharding_spec_indices(self, shape: Tuple[int, ...]) -> np.ndarray:
  """Returns NumPy-style indices corresponding to a sharding spec.

  Args:
    shape: The shape of the logical array being sharded.

  Returns:
    An ndarray with the same shape as the logical mesh (as derived form
    `mesh_mapping`). Each entry is a NumPy-style index selecting the subset of
    the data array to be placed on a corresponding device. The indices can be
    ints, slice objects with step=1, or tuples of those.
  """
  assert len(shape) == len(self.sharding), (shape, self.sharding)

  axis_indices: List[Sequence[Index]] = []
  shard_indices_shape = []
  for dim, sharding in enumerate(self.sharding):
    axis_size = shape[dim]
    if isinstance(sharding, NoSharding):
      axis_indices.append([slice(None)])
      # NOTE: We don't append unsharded dimensions to shard_indices_shape here,
      #       because they do not appear in the mesh mapping.
    elif isinstance(sharding, Unstacked):
      assert axis_size == sharding.size, f'{axis_size} != {sharding.size}'
      axis_indices.append(range(axis_size))
      shard_indices_shape.append(axis_size)
    elif isinstance(sharding, Chunked):
      total_chunks = int(np.prod(sharding.chunks))
      shard_size, ragged = divmod(axis_size, total_chunks)
      assert not ragged, (axis_size, total_chunks, dim)
      axis_indices.append([slice(i * shard_size, (i + 1) * shard_size)
                           for i in range(total_chunks)])
      shard_indices_shape.extend(sharding.chunks)
    else:
      assert_unreachable(sharding)

  # shard_indices is an ndarray representing the sharded axes of the logical array,
  # with each dimension having size equal to the number of shards across the corresponding
  # logical array dimension, and each element containing the multi-dimensional index that
  # is used to extract the corresponding shard of the logical array.
  shard_indices = np.empty([prod(shard_indices_shape)], dtype=np.object_)
  for i, idxs in enumerate(it.product(*axis_indices)):
    shard_indices[i] = idxs
  shard_indices = shard_indices.reshape(shard_indices_shape)

  # Ensure that each sharded axis is used exactly once in the mesh mapping
  num_sharded_dim = len(shard_indices_shape)
  sharded_dim_perm = [a.axis for a in self.mesh_mapping if isinstance(a, ShardedAxis)]
  assert (set(sharded_dim_perm) == set(range(num_sharded_dim)) and
          len(sharded_dim_perm) == num_sharded_dim)
  # Replicate/reorder the indices according to the mesh mapping
  replica_sizes = tuple(a.replicas for a in self.mesh_mapping if isinstance(a, Replicated))
  replica_dim, sharded_dim = it.count(0), iter(sharded_dim_perm)
  perm = [next(replica_dim) if isinstance(a, Replicated) else
          len(replica_sizes) + next(sharded_dim)
          for a in self.mesh_mapping]
  return (np.broadcast_to(shard_indices, replica_sizes + shard_indices.shape)
            .transpose(perm))

def sharding_spec_repr(self):
  return f'ShardingSpec({self.sharding}, {self.mesh_mapping})'


ShardingSpec.mesh_shape = property(sharding_spec_mesh_shape)
ShardingSpec.sharding_proto = sharding_spec_sharding_proto
ShardingSpec.indices = sharding_spec_indices
# mypy raises: error: Cannot assign to a method  [assignment]
ShardingSpec.__repr__ = sharding_spec_repr  # type: ignore
# Do not pollute the namespace
del sharding_spec_mesh_shape, sharding_spec_indices, sharding_spec_repr

def spec_to_indices(shape: Tuple[int, ...],
                    spec: ShardingSpec) -> Tuple[Index, ...]:
  """Returns numpy-style indices corresponding to a sharding spec.

  Each index describes a shard of the array. The order of the indices is the
  same as the device_buffers of a ShardedDeviceArray (i.e. the data is laid out
  row-major).

  Args:
    shape: The shape of the logical array being sharded.
    spec: Describes how the array is sharded and how the shards are assigned to
      the logical mesh.

  Returns:
    A tuple of length equal to the size of the mesh (inferred as the product of
    sharded dimension sizes and all replication factors).  Each element is an
    int, a slice object with step=1, or a tuple thereof, to be treated as an
    index into the full logical array.
  """
  return tuple(spec.indices(shape).flat)  # type: ignore


### util

def identity(x): return x

def _shard_arg(arg, devices, arg_indices):
  """Returns a list of size len(devices) containing per-device buffers.

  For the C++ pmap path, we fallback to Python (this function) to shard
  arguments that are not supported by the C++ `ShardArg`.

  Arrgs:
    arg: The Python argument.
    devices: The list of devices to shard over.
    arg_indices: A list of `len(devices)` indices to use to shard the argument.
  """
  if isinstance(arg, ShardedDeviceArray) and arg_indices == arg.indices:
    # The shard_arg_handlers allow an extensible set of types to be sharded, but
    # inline handling for ShardedDeviceArray as a special case for performance
    # NOTE: we compare indices instead of sharding_spec because
    # pmap_benchmark.pmap_shard_args_benchmark indicates this is faster.
    return [
        buf if buf.device() == d else buf.copy_to_device(d)
        for d, buf in zip(devices, arg.device_buffers)
    ]
  else:
    arg = xla.canonicalize_dtype(arg)
    return shard_arg_handlers[type(arg)](arg, devices, arg_indices)


@profiler.annotate_function
def shard_args(devices: Sequence[xb.xla_client.Device],
               indices: Sequence[Sequence[Index]],
               args) -> Sequence[Sequence[xb.xla_client.Buffer]]:
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
  return [_shard_arg(arg, devices, indices[a]) for a, arg in enumerate(args)]


shard_arg_handlers: Dict[Any, Callable[[Any, Any, Any], Sequence[Any]]] = {}
shard_arg_handlers[core.Unit] = \
    lambda x, devices, _: device_put(core.unit, devices, replicate=True)  # type: ignore[has-type]
def _shard_array(x, devices, indices):
  return device_put([x[i] for i in indices], devices)
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array

def _shard_device_array(x, devices, indices):
  start_indices, limit_indices, removed_dims = unzip3(
      _as_slice_indices(x, idx) for idx in indices)
  shards = x._multi_slice(start_indices, limit_indices, removed_dims)
  return device_put(shards, devices)
for t in device_array.device_array_types:
  shard_arg_handlers[t] = _shard_device_array


# NOTE(skye): we could refactor to generate _multi_slice parameters directly
# from the input ShardingSpec, rather than the indices. However, this would
# require duplicating the ordering logic of spec_to_indices, which is more
# subtle and more likely to change than the index logic we have to support here.
def _as_slice_indices(arr: device_array.DeviceArrayProtocol, idx: Index) -> Tuple[
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
shard_aval_handlers[core.AbstractUnit] = lambda size, axis, x: x
def _shard_abstract_array(size, axis: int, x):
  try:
    if x.shape[axis] != size:
      raise ValueError(f"Axis size {size} does not match dimension {axis} of "
                       f"shape {x.shape}")
  except IndexError:
    raise ValueError("Cannot split a {x.dim}D value along axis {axis}") from None
  return x.update(shape=tuple_delete(x.shape, axis))
shard_aval_handlers[ShapedArray] = _shard_abstract_array

MeshAxisName = Any
"""
ArrayMapping specifies how an ndarray should map to mesh axes.

Note that the ordering is crucial for the cases when this mapping is non-injective
(i.e. when multiple mesh axes map to the same positional axis). Then, the
order of entries of the mapping determines a major-to-minor order on mesh axes,
according to which chunks of the value along the repeated dimension will be assigned.

For example, consider a mapping {'x': 1, 'y': 1} and a mesh with shape {'x': 2, 'y': 3}.
The second dimension of the value would get chunked into 6 pieces, and assigned to the
mesh in a way that treats 'y' as the fastest changing (minor) dimension. In this case,
that would mean that a flat list of chunks would get assigned to a flattened list of
mesh devices without any modifications. If the mapping was {'y': 1, 'x': 1}, then the
mesh devices ndarray would have to be transposed before flattening and assignment.
"""
ArrayMapping = OrderedDictType[MeshAxisName, int]

AxisResource = Tuple[Optional[Tuple[Any, ...]], ...]

def array_mapping_to_axis_resources(array_mapping: ArrayMapping) -> AxisResource:
  if not array_mapping:
    return tuple()
  max_index = -1
  reverse_map = defaultdict(list)
  for axis, index in array_mapping.items():
    reverse_map[index].append(axis)
    if index > max_index:
      max_index = index
  return tuple(
      tuple(reverse_map[i]) if reverse_map[i] else None for i in range(max_index + 1)
  )

def local_aval_to_result_handler(
    aval: core.AbstractValue,
    sharding_spec: Optional[ShardingSpec],
    indices: Optional[Tuple[Index]],
) -> Callable[[List[xb.xla_client.Buffer]], Any]:
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
    return local_result_handlers[type(aval)](aval, sharding_spec, indices)
  except KeyError as err:
    raise TypeError(
        "No pxla_result_handler for type: {}".format(type(aval))) from err

PxlaResultHandler = Callable[..., Callable[[List[xb.xla_client.Buffer]], Any]]
local_result_handlers: Dict[Type[core.AbstractValue], PxlaResultHandler] = {}
local_result_handlers[core.AbstractUnit] = lambda *_: lambda _: core.unit
def sda_array_result_handler(aval: ShapedArray, sharding_spec, indices):
  return lambda bufs: make_sharded_device_array(aval, sharding_spec, bufs,
                                                indices)
local_result_handlers[ShapedArray] = sda_array_result_handler
local_result_handlers[ConcreteArray] = sda_array_result_handler


def global_aval_to_result_handler(
    aval: core.AbstractValue,
    out_axis_resources: Optional[AxisResource], global_mesh,
) -> Callable[[List[xb.xla_client.Buffer]], Any]:
  """Returns a function for handling the raw buffers of a single output aval.

  Args:
    aval: The global output AbstractValue.
    out_axis_resources: A tuple specifying the sharding of outputs.
      Used for creating GSDAs.
    global_mesh: The global device mesh that generated this output. Used
      for creating GSDAs.

  Returns:
    A function for handling the Buffers that will eventually be produced
    for this output. The function will return an object suitable for returning
    to the user, e.g. a ShardedDeviceArray.
  """
  try:
    return global_result_handlers[type(aval)](aval, out_axis_resources,
                                              global_mesh)
  except KeyError as err:
    raise TypeError(
        "No pxla_result_handler for type: {}".format(type(aval))) from err

global_result_handlers: Dict[Type[core.AbstractValue], PxlaResultHandler] = {}
global_result_handlers[core.AbstractUnit] = lambda *_: lambda _: core.unit

### lazy device-memory persistence and result handling

# TODO(jblespiau): Consider removing this option.
_USE_CPP_SDA = True


def make_sharded_device_array(
    aval: ShapedArray,
    sharding_spec: Optional[ShardingSpec],
    # Any is for JAX extensions implementing their own buffer.
    device_buffers: List[Union[Any, xb.xla_client.Buffer]],
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
    sharded_aval = aval.update(shape=aval.shape[1:])
    sharding_spec = _pmap_sharding_spec(aval.shape[0], aval.shape[0], 1, None,
                                        sharded_aval, 0)

  if indices is None:
    indices = spec_to_indices(aval.shape, sharding_spec)

  if (_USE_CPP_SDA and
      (not device_buffers or
       isinstance(device_buffers[0], xb.xla_client.Buffer))):
    return pmap_lib.ShardedDeviceArray.make(
        aval, sharding_spec, device_buffers,
        indices, aval.weak_type)

  return _ShardedDeviceArray(aval, sharding_spec, device_buffers, indices)


if _USE_CPP_SDA:
  ShardedDeviceArrayBase = pmap_lib.ShardedDeviceArrayBase  # type: ignore
  # We want the C++ SDA to extend the DeviceArrayBase. We want this both to
  # benefit from its methods, and to have isinstance(x, DeviceArray) return true
  ShardedDeviceArrayBase.__bases__ = ((device_array.DeviceArray,) +  # type: ignore
                                      ShardedDeviceArrayBase.__bases__)
  _SDA_BASE_CLASS = pmap_lib.ShardedDeviceArrayBase  # type: ignore
else:
  _SDA_BASE_CLASS: Type[device_array.DeviceArray] = device_array.DeviceArray  # type: ignore


class _ShardedDeviceArray(_SDA_BASE_CLASS):  # type: ignore
  """A ShardedDeviceArray is an ndarray sharded across devices.

  The purpose of a ShardedDeviceArray is to reduce the number of transfers when
  executing replicated computations, by allowing results to persist on the
  devices that produced them. That way dispatching a similarly replicated
  computation that consumes the same sharded memory layout does not incur any
  transfers.

  A ShardedDeviceArray represents one logical ndarray value, and simulates the
  behavior of an ndarray so that it can be treated by user code as an ndarray;
  that is, it is only an optimization to reduce transfers.

  Attributes:
    aval: A ShapedArray indicating the shape and dtype of this array.
    sharding_spec: describes how this array is sharded across `device_buffers`.
    device_buffers: the buffers containing the data for this array. Each buffer
      is the same shape and on a different device. Buffers are in row-major
      order, with replication treated as an extra innermost dimension.
    indices: the result of spec_to_indices(sharding_spec). Can optionally be
      precomputed for efficiency. A list the same length as
      `device_buffers`. Each index indicates what portion of the full array is
      stored in the corresponding device buffer, i.e. `array[indices[i]] ==
      device_buffers[i].to_py()`.
  """
  __slots__ = [
      "aval", "device_buffers", "sharding_spec", "indices",
      "_one_replica_buffer_indices", "_npy_value"
  ]

  def __init__(self,
               aval: ShapedArray,
               sharding_spec: ShardingSpec,
               device_buffers: List[xb.xla_client.Buffer],
               indices: Optional[Tuple[Index, ...]] = None):
    super().__init__()

    # TODO(skye): assert invariants. Keep performance in mind though.
    if indices is None:
      indices = spec_to_indices(aval.shape, sharding_spec)

    self.aval = aval
    self.device_buffers = device_buffers
    self.sharding_spec = sharding_spec
    self.indices = indices
    self._npy_value = None
    self._one_replica_buffer_indices = None
    if config.jax_enable_checks:
      assert type(aval) is ShapedArray

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

  def delete(self):
    if self.device_buffers is None:
      return
    for buf in self.device_buffers:
      buf.delete()
    self.device_buffers = None
    self._npy_value = None


def _sda_one_replica_buffer_indices(self):
  """Indices of buffers containing one complete copy of the array data."""
  if self._one_replica_buffer_indices is None:
    one_replica_indices = []
    seen_index_hashes = set()
    for i, index in enumerate(self.indices):
      hashed_index = _hashable_index(index)
      if hashed_index not in seen_index_hashes:
        one_replica_indices.append(i)
        seen_index_hashes.add(hashed_index)
    self._one_replica_buffer_indices = one_replica_indices
  return self._one_replica_buffer_indices


def _sda_copy_to_host_async(self):
  for buffer_index in self.one_replica_buffer_indices:
    self.device_buffers[buffer_index].copy_to_host_async()


def _sda_check_if_deleted(self):
  if self.device_buffers is None:
    raise ValueError("ShardedDeviceArray has been deleted.")


def _sda_block_until_ready(self):
  self._check_if_deleted()
  for buf in self.device_buffers:
    buf.block_host_until_ready()
  return self


def _sda_value(self):
  if self._npy_value is None:
    self.copy_to_host_async()
    npy_value = np.empty(self.aval.shape, self.aval.dtype)
    for i in self.one_replica_buffer_indices:
      npy_value[self.indices[i]] = self.device_buffers[i].to_py()
    self._npy_value = npy_value
  return self._npy_value


def _sda__getitem__(self, idx):
  self._check_if_deleted()
  if not isinstance(idx, tuple):
    cidx = (idx,) + (slice(None),) * (len(self.aval.shape) - 1)
  else:
    cidx = idx + (slice(None),) * (len(self.aval.shape) - len(idx))
  if self._npy_value is None:
    try:
      buf_idx = self.indices.index(cidx)
    except ValueError:
      buf_idx = None
    if buf_idx is not None:
      buf = self.device_buffers[buf_idx]
      aval = ShapedArray(buf.xla_shape().dimensions(), self.aval.dtype)
      return device_array.make_device_array(aval, None, buf)
  return super(self.__class__, self).__getitem__(idx)


def _sda__iter__(self):
  if self.ndim == 0:
    raise TypeError("iteration over a 0-d array")  # same as numpy error
  else:
    return (self[i] for i in range(self.shape[0]))

def _sda__reversed__(self):
  if self.ndim == 0:
    raise TypeError("iteration over a 0-d array")  # same as numpy error
  else:
    return (self[i] for i in range(self.shape[0] - 1, -1, -1))


for sda in [_ShardedDeviceArray, pmap_lib.ShardedDeviceArray]:
  setattr(sda, "one_replica_buffer_indices",
          property(_sda_one_replica_buffer_indices))
  setattr(sda, "copy_to_host_async", _sda_copy_to_host_async)
  setattr(sda, "_check_if_deleted", _sda_check_if_deleted)
  setattr(sda, "block_until_ready", _sda_block_until_ready)
  setattr(sda, "_value", property(_sda_value))
  setattr(sda, "__getitem__", _sda__getitem__)
  setattr(sda, "__iter__", _sda__iter__)
  setattr(sda, "__reversed__", _sda__reversed__)

del (_sda_one_replica_buffer_indices, _sda_copy_to_host_async,
     _sda_check_if_deleted, _sda_block_until_ready, _sda_value, _sda__getitem__)


ShardedDeviceArray: Type[object]
if _USE_CPP_SDA:
  ShardedDeviceArray = pmap_lib.ShardedDeviceArrayBase
else:
  ShardedDeviceArray = _ShardedDeviceArray



def _hashable_index(idx):
  return tree_map(lambda x: (x.start, x.stop) if type(x) == slice else x,
                  idx)

# The fast path is handled directly in shard_args().
# TODO(skye): is there a simpler way to rewrite this using sharding_spec?
def _shard_sharded_device_array_slow_path(x, devices, indices):
  candidates = defaultdict(list)
  for buf, idx in safe_zip(x.device_buffers, x.indices):
    candidates[_hashable_index(idx)].append(buf)

  bufs = []
  for idx, device in safe_zip(indices, devices):
    # Look up all buffers that contain the correct slice of the logical array.
    candidates_list = candidates[_hashable_index(idx)]
    if not candidates_list:
      # This array isn't sharded correctly. Reshard it via host roundtrip.
      # TODO(skye): more efficient reshard?
      return shard_arg_handlers[type(x._value)](x._value, devices, indices)
    # Try to find a candidate buffer already on the correct device,
    # otherwise copy one of them.
    for buf in candidates_list:
      if buf.device() == device:
        bufs.append(buf)
        break
    else:
      bufs.append(buf.copy_to_device(device))
  return bufs


def _sharded_device_array_constant_handler(c, val, canonicalize_types=True):
  return xla.pyval_to_ir_constants(c, np.asarray(val),
                                   canonicalize_types=canonicalize_types)


def _register_handlers_for_sharded_device_array(sda):
  shard_arg_handlers[sda] = _shard_sharded_device_array_slow_path
  xla.register_constant_handler(sda, _sharded_device_array_constant_handler)

  core.pytype_aval_mappings[sda] = abstract_arrays.canonical_concrete_aval
  dispatch.device_put_handlers[sda] = dispatch._device_put_array
  xla.pytype_aval_mappings[sda] = op.attrgetter("aval")
  xla.canonicalize_dtype_handlers[sda] = identity

_register_handlers_for_sharded_device_array(_ShardedDeviceArray)
_register_handlers_for_sharded_device_array(pmap_lib.ShardedDeviceArray)

### the xla_pmap primitive and its rules are comparable to xla_call in xla.py

def xla_pmap_impl(fun: lu.WrappedFun, *args,
                  backend: Optional[str],
                  axis_name: core.AxisName,
                  axis_size: int,
                  global_axis_size: Optional[int],
                  devices: Optional[Sequence[Any]],
                  name: str,
                  in_axes: Sequence[Optional[int]],
                  out_axes_thunk: Callable[[], Sequence[Optional[int]]],
                  donated_invars: Sequence[bool],
                  global_arg_shapes: Sequence[Optional[Tuple[int, ...]]]):
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun, fingerprint = parallel_callable(
      fun, backend, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, out_axes_thunk, donated_invars, global_arg_shapes,
      *abstract_args)

  # Don't re-abstractify args unless logging is enabled for performance.
  if config.jax_distributed_debug:
    distributed_debug_log(("Running pmapped function", name),
                          ("python function", fun.f),
                          ("devices", devices),
                          ("abstract args", map(xla.abstractify, args)),
                          ("fingerprint", fingerprint))
  return compiled_fun(*args)


@lu.cache
def parallel_callable(fun: lu.WrappedFun,
                      backend_name: Optional[str],
                      axis_name: core.AxisName,
                      axis_size: int,
                      global_axis_size: Optional[int],
                      devices: Optional[Sequence[Any]],
                      name: str,
                      in_axes: Sequence[Optional[int]],
                      out_axes_thunk: Callable[[], Sequence[Optional[int]]],
                      donated_invars: Sequence[bool],
                      global_arg_shapes: Sequence[Optional[Tuple[int, ...]]],
                      *avals):
  pmap_computation = lower_parallel_callable(
      fun, backend_name, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, out_axes_thunk, donated_invars, global_arg_shapes, avals)
  pmap_executable = pmap_computation.compile()
  return WeakRefList([pmap_executable.unsafe_call, pmap_executable.fingerprint])


@dataclasses.dataclass(frozen=True)
class ParallelCallableInfo:
  name: str
  backend: xla.Backend
  axis_name: core.AxisName
  axis_size: int
  global_axis_size: Optional[int]
  devices: Optional[Sequence[xla.Device]]
  in_axes: Iterable[Optional[int]]
  out_axes_thunk: Callable[[], Sequence[Optional[int]]]
  avals: Sequence[core.AbstractValue]

  @maybe_cached_property
  def local_devices(self):
    if self.devices:
      out = [d for d in self.devices
             if d.process_index == xb.process_index(self.backend)]
      assert len(out) > 0
    else:
      out = None  # type: ignore
    return out

  @maybe_cached_property
  def out_axes(self):
    return self.out_axes_thunk()


class ShardInfo(NamedTuple):
  sharded_avals: Sequence[core.AbstractValue]
  out_sharded_avals: Sequence[core.AbstractValue]
  global_sharded_avals: Sequence[core.AbstractValue]
  num_local_shards: int
  num_global_shards: int


class ReplicaInfo(NamedTuple):
  jaxpr_replicas: int
  num_local_replicas: int
  num_global_replicas: int


def find_replicas(jaxpr, axis_size, global_axis_size):
  # TODO(skyewm): replace this with a chain of pmaps and/or sharded_jits
  jaxpr_replicas = dispatch.jaxpr_replicas(jaxpr)
  num_local_replicas = axis_size * jaxpr_replicas
  num_global_replicas = global_axis_size * jaxpr_replicas
  return ReplicaInfo(jaxpr_replicas, num_local_replicas, num_global_replicas)


def should_tuple_args(shards: ShardInfo):
  # tuplify long arg lists for TPU
  return len(shards.global_sharded_avals) > 100


def stage_parallel_callable(
    pci: ParallelCallableInfo,
    fun: lu.WrappedFun,
    global_arg_shapes: Sequence[Optional[Tuple[int, ...]]]):
  sharded_avals = tuple(
      shard_aval(pci.axis_size, axis, aval) if axis is not None else aval
      for axis, aval in safe_zip(pci.in_axes, pci.avals))
  if any(s is not None for s in global_arg_shapes):
    # TODO(skye): we could take this branch unconditionally if we handled
    # grad of global_arg_shapes correctly.
    global_sharded_avals = [
        aval.update(shape=shape) if shape is not None else aval
        for shape, aval in safe_zip(global_arg_shapes, sharded_avals)]
  else:
    global_sharded_avals = sharded_avals  # type: ignore

  with core.extend_axis_env(pci.axis_name, pci.global_axis_size, None):  # type: ignore
    with dispatch.log_elapsed_time(f"Finished tracing + transforming {fun.__name__} "
                                   "for pmap in {elapsed_time} sec"):
      jaxpr, out_sharded_avals, consts = pe.trace_to_jaxpr_final(
          fun, global_sharded_avals, pe.debug_info_final(fun, "pmap"))
  jaxpr = core.apply_outfeed_rewriter(jaxpr)

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
  parts = find_partitions(jaxpr)

  num_local_shards = replicas.num_local_replicas * parts.local_num_partitions
  num_global_shards = replicas.num_global_replicas * parts.num_partitions

  shards = ShardInfo(
      sharded_avals, out_sharded_avals, global_sharded_avals,
      num_local_shards, num_global_shards)

  return jaxpr, consts, replicas, parts, shards


def _shardings_to_mlir_shardings(
    shardings: Optional[Sequence['PartitionsOrReplicated']]
    ) -> Optional[Sequence[Optional[xc.OpSharding]]]:
  if shardings is None:
    return None
  return [xla.sharding_to_proto(s) for s in shardings]

@profiler.annotate_function
def lower_parallel_callable(
    fun: lu.WrappedFun,
    backend_name: Optional[str],
    axis_name: core.AxisName,
    axis_size: int,
    global_axis_size: Optional[int],
    devices: Optional[Sequence[xla.Device]],
    name: str,
    in_axes: Iterable[Optional[int]],
    out_axes_thunk: Callable[[], Sequence[Optional[int]]],
    donated_invars: Sequence[bool],
    global_arg_shapes: Sequence[Optional[Tuple[int, ...]]],
    avals: Sequence[core.AbstractValue]):
  if devices is not None and len(devices) == 0:
    raise ValueError("'devices' argument to pmap must be non-empty, or None.")

  # Determine global_axis_size for use in AxisEnv.
  # TODO(mattjj,skyewm): revive this check (inner_pmap always False now)
  # if xb.process_count() > 1 and global_axis_size is None and inner_pmap:
  #   raise ValueError("'axis_size' must be specified for nested multi-host pmaps")
  if (xb.process_count() == 1 and global_axis_size is not None and
      global_axis_size != axis_size):
    raise ValueError(
        f"Specified axis_size {global_axis_size} doesn't match received "
        f"axis_size {axis_size}.")

  if devices is not None and backend_name is None:
    backend = xb.get_device_backend(devices[0])
  else:
    backend = xb.get_backend(backend_name)

  must_run_on_all_devices = False
  no_nested_sharding = False
  if global_axis_size is None:
    if xb.process_count(backend) == 1:
      global_axis_size = axis_size
    elif devices:
      # This allows each host in a multi-host pmap to run on a different number
      # of devices, but precludes nested sharding (i.e. inner pmaps or
      # sharded_jits).
      global_axis_size = len(devices)
      no_nested_sharding = True
    else:
      # This assumes all hosts run on the same number of devices. We make sure
      # this assumption is true by requiring that the pmap is run on all devices
      # (and making the further assumption that each host has the same number of
      # devices). Nested sharding is ok in this case.
      global_axis_size = axis_size * xb.process_count(backend)
      assert all(
          len(xb.local_devices(process_index, backend)) == xb.local_device_count(backend)
          for process_index in range(xb.process_count(backend)))
      must_run_on_all_devices = True

  pci = ParallelCallableInfo(
      name, backend, axis_name, axis_size, global_axis_size, devices,
      in_axes, out_axes_thunk, avals)
  jaxpr, consts, replicas, parts, shards = stage_parallel_callable(
      pci, fun, global_arg_shapes)

  if logging.vlog_is_on(2):
    logging.vlog(2, "sharded_avals: %s", shards.sharded_avals)
    logging.vlog(2, "global_sharded_avals: %s", shards.global_sharded_avals)
    logging.vlog(2, "num_replicas: %d  num_local_replicas: %d",
                 replicas.num_global_replicas, replicas.num_local_replicas)
    logging.vlog(2, "num_partitions: %d  local_num_partitions: %d",
                 parts.num_partitions, parts.local_num_partitions)
    logging.vlog(2, "arg_parts: %s", parts.arg_parts)
    logging.vlog(2, "local_arg_parts: %s", parts.local_arg_parts)
    logging.vlog(2, "out_parts: %s", parts.out_parts)
    logging.vlog(2, "local_out_parts: %s", parts.local_out_parts)
    logging.vlog(2, "devices: %s", devices)
    logging.vlog(2, "local_devices: %s", pci.local_devices)

  if (xb.process_count(backend) > 1 and must_run_on_all_devices and
      shards.num_local_shards != xb.local_device_count(backend)):
    if shards.num_local_shards == axis_size:
      raise ValueError(
         f"On multi-host platforms, the input to pmapped functions must have "
         f"leading axis size equal to the number of local devices if no "
         f"`devices` argument is specified. Got axis_size={axis_size}, "
         f"num_local_devices={xb.local_device_count(backend)}")
    else:
      raise ValueError(
        f"On multi-host platforms, pmapped functions must run across all "
        f"devices, i.e. num_replicas * num_partitions should equal the "
        f"number of local devices. Got "
        f"num_replicas={replicas.num_local_replicas}, "
        f"num_partitions={parts.num_partitions}, and "
        f"num_local_devices={xb.local_device_count(backend)}")

  if no_nested_sharding and (
      replicas.jaxpr_replicas > 1 or parts.num_partitions > 1):
    raise ValueError(
      f"On multi-host platforms, pmapped functions that both have `devices` "
      f"specified and contain an inner_pmap or sharded_jit must specify an "
      f"`axis_size` (or remove the `devices` argument). Got nested_replicas="
      f"{replicas.jaxpr_replicas} and nested_partitions={parts.num_partitions}")

  log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
  logging.log(log_priority,
              "Compiling %s (%d) for %d devices with args %s. (num_replicas=%d"
              " num_partitions=%d)", fun.__name__, id(fun),
              shards.num_global_shards, avals, replicas.num_global_replicas,
              parts.num_partitions)

  axis_env = xla.AxisEnv(
      replicas.num_global_replicas, (axis_name,), (global_axis_size,))
  name_stack = extend_name_stack(wrap_name(name, 'pmap'))
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  replicated_args = [axis is None for axis in in_axes]
  module: Union[str, xc.XlaComputation]
  tuple_args = should_tuple_args(shards)
  module_name = f"pmap_{fun.__name__}"
  with maybe_extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
    if config.jax_enable_mlir:
      module = mlir.lower_jaxpr_to_module(
          module_name, closed_jaxpr, backend.platform, axis_env,
          name_stack, donated_invars, replicated_args=replicated_args,
          arg_shardings=_shardings_to_mlir_shardings(parts.arg_parts),
          result_shardings=_shardings_to_mlir_shardings(parts.out_parts))
    else:
      module = xla.lower_jaxpr_to_xla_module(
          module_name, closed_jaxpr, backend.platform, axis_env,
          name_stack, tuple_args, donated_invars, replicated_args,
          parts.arg_parts, parts.out_parts)
  return PmapComputation(module, pci=pci, replicas=replicas, parts=parts,
                         shards=shards, tuple_args=tuple_args)


class PmapComputation:
  def __init__(self, hlo, **compile_args):
    self._executable = None
    self._hlo = hlo
    self.compile_args = compile_args

  def hlo(self):
    # this is a method for api consistency with dispatch.XlaComputation
    if isinstance(self._hlo, xc.XlaComputation):
      return self._hlo
    else:
      return xe.mlir.mlir_module_to_xla_computation(
          mlir.module_to_string(self._hlo),
          use_tuple_args=self.compile_args["tuple_args"])

  def mhlo(self) -> str:
    if isinstance(self._hlo, xc.XlaComputation):
      return xe.mlir.xla_computation_to_mlir_module(self._hlo)
    return mlir.module_to_string(self._hlo)

  @profiler.annotate_function
  def compile(self):
    if self._executable is None:
      self._executable = PmapExecutable.from_hlo(self._hlo, **self.compile_args)
    return self._executable


class PmapExecutable:
  __slots__ = ['xla_executable', 'unsafe_call', 'fingerprint', 'in_avals']

  def __init__(self, xla_executable, unsafe_call, fingerprint, in_avals):
    self.xla_executable = xla_executable
    self.unsafe_call = unsafe_call
    self.fingerprint = fingerprint
    self.in_avals = in_avals

  @staticmethod
  def from_hlo(xla_computation,
               pci: ParallelCallableInfo,
               replicas: ReplicaInfo,
               parts: 'PartitionInfo',
               shards: ShardInfo,
               tuple_args: bool):
    devices = pci.devices
    if devices is None:
      if shards.num_global_shards > xb.device_count(pci.backend):
        msg = ("compiling computation that requires {} logical devices, but only {} XLA "
               "devices are available (num_replicas={}, num_partitions={})")
        raise ValueError(msg.format(shards.num_global_shards,
                                    xb.device_count(pci.backend),
                                    replicas.num_global_replicas,
                                    parts.num_partitions))
      # On a single host, we use the platform's default device assignment to
      # potentially take advantage of device locality. On multiple hosts, the
      # default device assignment may interleave different hosts' replicas,
      # violating pmap's semantics where data is sharded across replicas in
      # row-major order. Instead, manually create a device assignment that ensures
      # each host is responsible for a continguous set of replicas.
      if shards.num_global_shards > shards.num_local_shards:
        # TODO(skye): use a locality-aware assignment that satisfies the above
        # constraint.
        devices = [d for process_index in range(xb.process_count(pci.backend))
                  for d in xb.local_devices(process_index, pci.backend)]
      else:
        devices = xb.get_backend(pci.backend).get_default_device_assignment(
            replicas.num_global_replicas, parts.num_partitions)
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
    device_assignment = tree_map(lambda d: d.id, devices)
    # Convert to 2D in case it's 1D and we have > 1 partitions.
    device_assignment = np.array(device_assignment).reshape(
        (replicas.num_global_replicas, parts.num_partitions))
    # TODO(b/162356737): Enabling SPMD partitioning causes issues with some
    # non-partitioned workloads, so disable unless needed.
    use_spmd_partitioning = parts.num_partitions > 1
    compile_options = xb.get_compile_options(
        num_replicas=replicas.num_global_replicas,
        num_partitions=parts.num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=use_spmd_partitioning,
    )
    compile_options.parameter_is_tupled_arguments = tuple_args

    local_arg_parts_ = parts.local_arg_parts or [None] * len(pci.avals)
    input_sharding_specs = [
        _pmap_sharding_spec(replicas.num_local_replicas, pci.axis_size,
                            parts.local_num_partitions, arg_parts, aval, in_axis)
        if aval is not core.abstract_unit else None
        for aval, arg_parts, in_axis in safe_zip(
            shards.sharded_avals, local_arg_parts_, pci.in_axes)]
    input_indices = [spec_to_indices(aval.shape, spec)
                    if spec is not None else None
                    for aval, spec in safe_zip(pci.avals, input_sharding_specs)]
    nouts = len(shards.out_sharded_avals)

    out_parts, local_out_parts = parts.out_parts, parts.local_out_parts
    if parts.out_parts is None:
      out_parts = (None,) * nouts
    if parts.local_out_parts is None:
      local_out_parts = (None,) * nouts

    local_out_avals = [
        get_local_aval(aval, parts, lparts)
        for aval, parts, lparts
        in safe_zip(shards.out_sharded_avals, out_parts, local_out_parts)]
    local_unmapped_avals = [
        core.unmapped_aval(pci.axis_size, pci.axis_name, out_axis, aval)
        if out_axis is not None else aval
        for aval, out_axis in safe_zip(local_out_avals, pci.out_axes)]

    out_specs = [
        _pmap_sharding_spec(replicas.num_local_replicas, pci.axis_size,
                            parts.local_num_partitions, out_parts, aval, out_axis)
        if aval is not core.abstract_unit else None
        for out_parts, aval, out_axis in safe_zip(
            local_out_parts, local_out_avals, pci.out_axes)]
    handle_outs = local_avals_to_results_handler(out_specs, local_unmapped_avals)

    if hasattr(pci.backend, "compile_replicated"):
      execute_fun = pci.backend.compile_replicated(
          xla_computation, compile_options, input_indices, input_sharding_specs,
          handle_outs)
      # TODO(frostig): need `compile_replicated` to give us the XLA executable
      return PmapExecutable(None, execute_fun, None, pci.avals)

    with dispatch.log_elapsed_time(
        f"Finished XLA compilation of {pci.name} in {{elapsed_time}} sec"):
      compiled = dispatch.compile_or_get_cached(
          pci.backend, xla_computation, compile_options)
    handle_args = InputsHandler(
        compiled.local_devices(), input_sharding_specs, input_indices)
    execute_fun = partial(
        execute_replicated, compiled, pci.backend, handle_args, handle_outs)
    fingerprint = getattr(compiled, "fingerprint", None)

    return PmapExecutable(compiled, execute_fun, fingerprint, pci.avals)

  @profiler.annotate_function
  def call(self, *args):
    # TODO(frostig): do we need to check sharding and sharded avals?
    arg_avals = map(xla.abstractify, args)
    dispatch.check_arg_avals_for_call(self.in_avals, arg_avals)
    return self.unsafe_call(*args)


multi_host_supported_collectives: Set[core.Primitive] = set()


def check_multihost_collective_allowlist(jaxpr):
  used_collectives = set(xla.jaxpr_collectives(jaxpr))
  if not used_collectives.issubset(multi_host_supported_collectives):
    bad_collectives = used_collectives - multi_host_supported_collectives
    msg = "using collectives that aren't supported for multi-host: {}"
    raise TypeError(msg.format(", ".join(map(str, bad_collectives))))


PartitionsOrReplicated = Optional[Tuple[int, ...]]

class PartitionInfo(NamedTuple):
  arg_parts: Optional[Tuple[PartitionsOrReplicated, ...]]
  out_parts: Optional[Tuple[PartitionsOrReplicated, ...]]
  num_partitions: int
  local_arg_parts: Optional[Tuple[PartitionsOrReplicated, ...]]
  local_out_parts: Optional[Tuple[PartitionsOrReplicated, ...]]
  local_num_partitions: Optional[int]

def _find_partitions(jaxpr):
  """Returns (in_partitions, out_partitions, num_partitions, local_in_parts,
              local_out_parts, local_num_partitions).
  """
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "sharded_call":
      if len(jaxpr.eqns) > 1:
        raise NotImplementedError(
            "pmap of sharded_jit + non-sharded operations not yet implemented.")
      num_partitions = reconcile_num_partitions(eqn.params["call_jaxpr"],
                                                eqn.params["nparts"])
      return (eqn.params["in_parts"],
              eqn.params["out_parts_thunk"](),
              num_partitions,
              eqn.params["local_in_parts"],
              eqn.params["local_out_parts_thunk"](),
              eqn.params["local_nparts"])
  return None, None, 1, None, None, None

def find_partitions(jaxpr) -> PartitionInfo:
  (arg_parts, out_parts, num_partitions, local_arg_parts, local_out_parts,
   local_num_partitions) = _find_partitions(jaxpr)

  if local_num_partitions is None:
    local_num_partitions = num_partitions
  if local_arg_parts is None:
    local_arg_parts = arg_parts
  if local_out_parts is None:
    local_out_parts = out_parts

  return PartitionInfo(arg_parts, out_parts, num_partitions,
                       local_arg_parts, local_out_parts, local_num_partitions)


def reconcile_num_partitions(jaxpr, outer_num_parts: Optional[int]):
  """Returns the total number of partitions to use.

  Validates that any inner partitioning matches outer_num_parts if provided, and
  returns the number of partitions to use based on outer_num_parts and any inner
  partitioning.
  """
  inner_num_parts = _inner_partitions(jaxpr, outer_num_parts)
  if outer_num_parts is None and inner_num_parts is None:
    # No partitions specified anywhere, everything is replicated.
    return 1
  if outer_num_parts is None:
    return inner_num_parts
  return outer_num_parts


def _inner_partitions(jaxpr, expected_num_parts: Optional[int]):
  """Returns the total number of partitions from PartitionSpecs inside `jaxpr`.

  Also validates that this number matches `expected_num_parts` if provided.
  """
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in ["sharding_constraint", "infeed"]:
      parts = eqn.params["partitions"]
      nparts = get_num_partitions(parts)
      if expected_num_parts is None:
        expected_num_parts = nparts
      elif nparts is not None and nparts != expected_num_parts:
        # TODO(skye): raise this error as we trace the jaxpr
        raise ValueError(
            f"with_sharding_constraint with partitions={parts} "
            f"(total partitions: {nparts}) doesn't match expected number of "
            f"partitions: {expected_num_parts}. If these partitions look "
            f"right, check outer sharded_jit and/or other "
            f"with_sharding_constraint calls.")
    else:
      for subjaxpr in core.jaxprs_in_params(eqn.params):
        expected_num_parts = _inner_partitions(subjaxpr, expected_num_parts)
  return expected_num_parts


def get_num_partitions(*partitions):
  partition_specs = tree_flatten(partitions)[0]
  if len(partition_specs) == 0:
    # Everything is specified as replicated (all Nones).
    return None
  num_partitions_set = {np.prod(spec) for spec in partition_specs}
  if len(num_partitions_set) > 1:
    raise ValueError(
        f"All partition specs must use the same number of total partitions, "
        f"got {partitions}, with distinct number of partitions "
        f"{num_partitions_set} (the total number of partitions is the product "
        f"of a partition spec)")
  assert len(num_partitions_set) == 1
  return num_partitions_set.pop()


def get_global_aval(local_aval, global_parts: PartitionsOrReplicated,
                    local_parts: PartitionsOrReplicated):
  if local_aval is core.abstract_unit:
    return local_aval
  if global_parts is None:
    return local_aval
  assert local_parts is not None
  global_shape = [dim * _safe_div(ngparts, nlparts)
                  for dim, ngparts, nlparts
                  in safe_zip(local_aval.shape, global_parts, local_parts)]
  return local_aval.update(shape=global_shape)


def get_local_aval(global_aval, global_parts: PartitionsOrReplicated,
                   local_parts: PartitionsOrReplicated):
  if global_aval is core.abstract_unit:
    return global_aval
  if global_parts is None:
    return global_aval
  assert local_parts is not None
  local_shape = [_safe_div(dim, _safe_div(ngparts, nlparts))
                 for dim, ngparts, nlparts
                 in safe_zip(global_aval.shape, global_parts, local_parts)]
  return global_aval.update(shape=local_shape)


def _safe_div(x, y):
  result, ragged = divmod(x, y)
  assert not ragged, f"{x} % {y} != 0"
  return result


class InputsHandler:
  __slots__ = ("handler", "local_devices", "sharding_specs", "input_indices")

  def __init__(self, local_devices, sharding_specs, input_indices):
    self.handler = partial(shard_args, local_devices, input_indices)
    self.local_devices = local_devices
    self.sharding_specs = sharding_specs
    self.input_indices = input_indices

  def __call__(self, input_buffers):
    return self.handler(input_buffers)


class ResultsHandler:
  __slots__ = ("handlers", "out_specs", "out_indices", "unmapped_local_out_avals")

  def __init__(self, handlers, out_specs, out_indices, unmapped_local_out_avals):
    self.out_specs = out_specs
    self.out_indices = out_indices
    self.handlers = handlers
    self.unmapped_local_out_avals = unmapped_local_out_avals

  def __call__(self, out_bufs):
    return [h(bufs) for h, bufs in safe_zip(self.handlers, out_bufs)]


def local_avals_to_results_handler(
    local_out_specs: Sequence[Optional[ShardingSpec]],
    unmapped_local_out_avals: Sequence[Optional[ShapedArray]]):
  out_indices = [spec_to_indices(aval.shape, spec)
                 if aval is not core.abstract_unit else None
                 for aval, spec in safe_zip(unmapped_local_out_avals, local_out_specs)]  # pytype: disable=attribute-error
  handlers = [
      local_aval_to_result_handler(aval, spec, idcs)
      for aval, spec, idcs in safe_zip(unmapped_local_out_avals, local_out_specs, out_indices)
  ]
  return ResultsHandler(handlers, local_out_specs, out_indices, unmapped_local_out_avals)


def global_avals_to_results_handler(global_out_avals: Sequence[ShapedArray],
                                    out_axes: Sequence[ArrayMapping],
                                    global_mesh):
  if config.jax_parallel_functions_output_gda:
    global_sharding_spec = mesh_sharding_specs(global_mesh.shape, global_mesh.axis_names)
    global_out_specs = [global_sharding_spec(aval, oa)
                        for aval, oa in safe_zip(global_out_avals, out_axes)]
    out_indices = [spec_to_indices(aval.shape, spec)
                   if aval is not core.abstract_unit else None
                   for aval, spec in safe_zip(global_out_avals, global_out_specs)]
    out_axis_resources = [array_mapping_to_axis_resources(o) for o in out_axes]
    handlers = [
        global_aval_to_result_handler(global_aval, out_axis, global_mesh)
        for global_aval, out_axis in safe_zip(global_out_avals, out_axis_resources)
    ]
    return ResultsHandler(handlers, global_out_specs, out_indices, global_out_avals)
  else:
    local_sharding_spec = mesh_sharding_specs(global_mesh.local_mesh.shape, global_mesh.axis_names)
    local_out_untiled_avals = [global_mesh.global_to_local(axis, aval)
                               for axis, aval in safe_zip(out_axes, global_out_avals)]
    local_out_specs = [local_sharding_spec(aval, oa)
                       for aval, oa in safe_zip(local_out_untiled_avals, out_axes)]
    return local_avals_to_results_handler(
        local_out_specs, local_out_untiled_avals)


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

  aval = xla.abstractify(val)  # type: ShapedArray
  if in_axis is not None:
    replicated_aval = aval.update(shape=(axis_size,) + aval.shape)
  else:
    replicated_aval = aval
  # TODO(skye): figure out how partitioning should work here
  sharding_spec = _pmap_sharding_spec(nrep, axis_size, 1, None, aval, in_axis)
  device_buffers = device_put(val, devices, replicate=True)
  return make_sharded_device_array(replicated_aval, sharding_spec,
                                   device_buffers)


def _pmap_sharding_spec(nrep, axis_size, npart, parts, sharded_aval,
                        map_axis: Optional[int]) -> ShardingSpec:
  """Sharding spec for arguments or results of a pmap.
  Args:
    nrep: number of local XLA replicas (product of local axis sizes)
    axis_size: local axis size for outer pmap
    npart: total number of XLA partitions (required by sharded_jit calls)
    parts: the partitioning of the value or None
    sharded_aval: the aval of the value inside the outer pmap, an instance of
      a ShapedArray.
    map_axis: the axis along which the value is mapped in the outer pmap
  Returns:
    A ShardingSpec.
  """
  assert isinstance(sharded_aval, ShapedArray), sharded_aval
  replication_factor, ragged = divmod(nrep, axis_size)
  assert not ragged
  # get the sharding spec from inner sharded_jits as if we weren't in a pmap
  pspec = partitioned_sharding_spec(npart, parts, sharded_aval)
  maybe_replicate = () if replication_factor == 1 else (Replicated(replication_factor),)
  if map_axis is not None:
    sharded_in_axis = sum(not isinstance(s, NoSharding) for s in pspec.sharding[:map_axis])
    def shift_sharded_axis(a: MeshDimAssignment):
      if isinstance(a, ShardedAxis) and a.axis >= sharded_in_axis:
        return ShardedAxis(a.axis + 1)
      return a
    # replication_factor represents the product of inner pmaps, so it goes
    # after the outer pmapped axis at index 0
    return ShardingSpec(
      sharding=tuple_insert(pspec.sharding, map_axis, Unstacked(axis_size)),
      mesh_mapping=it.chain([ShardedAxis(sharded_in_axis)],
                            maybe_replicate,
                            map(shift_sharded_axis, pspec.mesh_mapping)))
  else:
    return ShardingSpec(
      sharding=pspec.sharding,
      mesh_mapping=(Replicated(axis_size),) + maybe_replicate + pspec.mesh_mapping)

def partitioned_sharding_spec(num_partitions: int,
                              partitions: Optional[Sequence[int]],
                              aval) -> ShardingSpec:
  if partitions is None:
    maybe_replicate = () if num_partitions == 1 else (Replicated(num_partitions),)
    return ShardingSpec(
        sharding=[_UNSHARDED_INSTANCE] * len(aval.shape),
        mesh_mapping=maybe_replicate)
  else:
    assert len(partitions) == len(aval.shape)
    return ShardingSpec(
        # Chunked expects a list of integers
        sharding=map(Chunked, [[x] for x in partitions]),
        mesh_mapping=map(ShardedAxis, range(len(partitions))))


@profiler.annotate_function
def execute_replicated(compiled, backend, in_handler, out_handler, *args):
  input_bufs = in_handler(args)
  out_bufs = compiled.execute_sharded_on_local_devices(input_bufs)
  if dispatch.needs_check_special():
    for bufs in out_bufs:
      dispatch.check_special("parallel computation", bufs)
  return out_handler(out_bufs)


xla_pmap_p = core.MapPrimitive('xla_pmap')
xla_pmap = xla_pmap_p.bind
xla_pmap_p.def_impl(xla_pmap_impl)

# Set param update handlers to update `donated_invars` just like xla_call_p
pe.call_param_updaters[xla_pmap_p] = pe.call_param_updaters[xla.xla_call_p]
ad.call_param_updaters[xla_pmap_p] = ad.call_param_updaters[xla.xla_call_p]
ad.call_transpose_param_updaters[xla_pmap_p] = \
    ad.call_transpose_param_updaters[xla.xla_call_p]

def _pmap_translation_rule(ctx, avals_in, avals_out, *in_nodes,
                           axis_name, axis_size,
                           global_axis_size, devices, name,
                           call_jaxpr, backend=None, in_axes, out_axes,
                           donated_invars, global_arg_shapes):
  del donated_invars  # Unused.
  xla.check_backend_matches(backend, ctx.platform)
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if ctx.axis_env.names and devices is not None:
    raise ValueError("Nested pmap with explicit devices argument.")
  if global_axis_size is None:
    global_axis_size = axis_size
  new_env = xla.extend_axis_env(ctx.axis_env, axis_name, global_axis_size)
  # Shard the in_nodes that are mapped
  in_avals = [v.aval for v in call_jaxpr.invars]
  in_nodes_sharded = (
      _xla_shard(ctx.builder, aval, new_env, in_node, in_axis)
      if in_axis is not None else in_node
      for aval, in_node, in_axis in safe_zip(in_avals, in_nodes, in_axes))

  with maybe_extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
    sub_ctx = ctx.replace(
        axis_env=new_env,
        name_stack=extend_name_stack(ctx.name_stack, wrap_name(name, 'pmap')))
    sharded_outs = xla.jaxpr_subcomp(sub_ctx, call_jaxpr, (), *in_nodes_sharded)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_xla_unshard(ctx.builder, aval, new_env, out_axis, shard,
                       backend=backend)
          for aval, out_axis, shard in safe_zip(out_avals, out_axes, sharded_outs)]
  return outs

xla.register_translation(xla_pmap_p, _pmap_translation_rule)
ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)

def _xla_shard(c, aval, axis_env, x, in_axis):
  if aval is core.abstract_unit:
    return x
  elif aval is core.abstract_token:
    return x
  elif isinstance(aval, ShapedArray):
    dims = list(c.get_shape(x).dimensions())
    zero = xops.Constant(c, np.zeros((), dtype=np.uint32))
    idxs = [zero] * (len(dims) - 1)
    idxs.insert(in_axis, _unravel_index(c, axis_env))
    dims_unsqueezed = dims.copy()
    dims_unsqueezed[in_axis] = 1
    dims_squeezed = dims.copy()
    dims_squeezed.pop(in_axis)
    return xops.Reshape(xops.DynamicSlice(x, idxs, dims_unsqueezed), dims_squeezed)
  else:
    raise TypeError((aval, c.get_shape(x)))

# TODO(b/110096942): more efficient gather
def _xla_unshard(c, aval, axis_env, out_axis, x, backend):
  if aval is core.abstract_unit:
    return x
  elif aval is core.abstract_token:
    return x
  elif isinstance(aval, ShapedArray):
    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    convert_bool = (np.issubdtype(aval.dtype, np.bool_)
                    and xb.get_backend(backend).platform in ('cpu', 'gpu'))
    if convert_bool:
      x = xops.ConvertElementType(
          x, xla.dtype_to_primitive_type(np.dtype(np.float32)))

    xla_shape = c.get_shape(x)
    dims = list(xla_shape.dimensions())
    padded = xops.Broadcast(
        xops.Constant(c, np.array(0, xla_shape.numpy_dtype())),
        [axis_env.sizes[-1]] + dims)
    zero = xops.Constant(c, np.zeros((), dtype=np.uint32))
    idxs = [_unravel_index(c, axis_env)] + [zero] * len(dims)
    padded = xops.DynamicUpdateSlice(padded, xops.Reshape(x, [1] + dims), idxs)
    replica_groups_protos = xc.make_replica_groups(
      xla.axis_groups(axis_env, axis_env.names[-1]))
    out = xops.CrossReplicaSum(padded, replica_groups_protos)
    if out_axis != 0:
      # TODO(apaszke,mattjj): Change the indices to DynamicUpdateSlice instead
      perm = list(range(1, len(dims)))
      perm.insert(out_axis, 0)
      out = xops.Transpose(out, perm)

    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    if convert_bool:
      nonzero = xops.Ne(out, xops.Constant(c, np.array(0, dtype=np.float32)))
      out = xops.ConvertElementType(
          nonzero, xla.dtype_to_primitive_type(np.dtype(np.bool_)))
    return out
  else:
    raise TypeError((aval, c.get_shape(x)))

def _unravel_index(c, axis_env):
  div = xops.Constant(c, np.array(axis_env.nreps // prod(axis_env.sizes),
                                  np.uint32))
  mod = xops.Constant(c, np.array(axis_env.sizes[-1], np.uint32))
  return xops.Rem(xops.Div(xops.ReplicaId(c), div), mod)



def _unravel_index_mhlo(axis_env):
  div = mlir.ir_constant(
      np.array(axis_env.nreps // util.prod(axis_env.sizes), np.uint32))
  mod = mlir.ir_constant(np.array(axis_env.sizes[-1], np.uint32))
  return mhlo.RemOp(
      mhlo.DivOp(mhlo.ReplicaIdOp().result, div).result, mod).result

def _mhlo_shard(aval, axis_env, xs, in_axis):
  if aval is core.abstract_unit:
    return xs
  elif aval is core.abstract_token:
    return xs
  elif isinstance(aval, core.ShapedArray):
    x, = xs
    dims = list(aval.shape)
    zero = mlir.ir_constant(np.zeros((), dtype=np.uint32))
    idxs = [zero] * len(dims)
    idxs.insert(in_axis, _unravel_index_mhlo(axis_env))
    dims_unsqueezed = dims.copy()
    dims_unsqueezed.insert(in_axis, 1)
    return [
      mhlo.ReshapeOp(
        mlir.aval_to_ir_type(aval),
        mhlo.DynamicSliceOp(
            mlir.aval_to_ir_type(aval.update(shape=dims_unsqueezed)),
            x, idxs, mlir.dense_int_elements(dims_unsqueezed)).result
      ).result
    ]
  else:
    raise TypeError(aval)

# TODO(b/110096942): more efficient gather
def _mhlo_unshard(aval, axis_env, out_axis, xs, platform):
  if aval is core.abstract_unit:
    return xs
  elif aval is core.abstract_token:
    return xs
  elif isinstance(aval, core.ShapedArray):
    x, = xs
    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    convert_bool = (np.issubdtype(aval.dtype, np.bool_)
                    and platform in ('cpu', 'gpu'))
    if convert_bool:
      aval = aval.update(dtype=np.dtype(np.float32))
      x = mhlo.ConvertOp(mlir.aval_to_ir_type(aval), x).result

    dims = list(aval.shape)
    padded_aval = aval.update(shape=[axis_env.sizes[-1]] + dims)
    padded = mlir.full_like_aval(0, padded_aval)
    zero = mlir.ir_constant(np.zeros((), dtype=np.uint32))
    idxs = [_unravel_index_mhlo(axis_env)] + [zero] * len(dims)
    padded = mhlo.DynamicUpdateSliceOp(
        padded.type,
        padded,
        mhlo.BroadcastOp(mlir.aval_to_ir_type(aval.update(shape=[1] + dims)), x,
                         mlir.dense_int_elements([1])).result,
        idxs).result
    replica_groups = mlir.dense_int_elements(
      xla.axis_groups(axis_env, axis_env.names[-1]))
    out = mhlo.CrossReplicaSumOp(padded, replica_groups).result
    if out_axis != 0:
      # TODO(apaszke,mattjj): Change the indices to DynamicUpdateSlice instead
      perm = list(range(1, len(dims)))
      perm.insert(out_axis, 0)
      transposed_dims = list(dims)
      transposed_dims.insert(out_axis, axis_env.sizes[-1])
      aval = aval.update(shape=transposed_dims)
      out = mhlo.TransposeOp(mlir.aval_to_ir_type(aval), out,
                             mlir.dense_int_elements(perm)).result

    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    if convert_bool:
      float_zero = mlir.full_like_aval(0, padded_aval)
      out = mhlo.CompareOp(
          mlir.aval_to_ir_type(padded_aval.update(dtype=np.dtype(np.bool_))),
          out, float_zero, ir.StringAttr.get("NE"),
          ir.StringAttr.get("FLOAT")).result
    return out
  else:
    raise TypeError(aval)


def _pmap_lowering(ctx, avals_in, avals_out, *in_nodes, axis_name,
                   axis_size, global_axis_size, devices, name,
                   call_jaxpr, backend=None, in_axes, out_axes,
                   donated_invars, global_arg_shapes):
  del donated_invars  # Unused.
  xla.check_backend_matches(backend, ctx.platform)
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if ctx.axis_env.names and devices is not None:
    raise ValueError("Nested pmap with explicit devices argument.")
  if global_axis_size is None:
    global_axis_size = axis_size
  new_env = xla.extend_axis_env(ctx.axis_env, axis_name, global_axis_size)
  # Shard the in_nodes that are mapped
  in_avals = [v.aval for v in call_jaxpr.invars]
  in_nodes_sharded = (
    _mhlo_shard(aval, new_env, mlir.wrap_singleton_ir_values(in_node), in_axis)
    if in_axis is not None else mlir.wrap_singleton_ir_values(in_node)
    for aval, in_node, in_axis in zip(in_avals, in_nodes, in_axes))

  with maybe_extend_axis_env(axis_name, global_axis_size, None):  # type: ignore
    sub_ctx = ctx.replace(
        axis_env=new_env,
        name_stack=xla.extend_name_stack(ctx.name_stack,
                                         util.wrap_name(name, 'pmap')))
    sharded_outs = mlir.jaxpr_subcomp(sub_ctx, call_jaxpr, (),
                                      *in_nodes_sharded)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_mhlo_unshard(aval, new_env, out_axis, shard, platform=ctx.platform)
          for aval, out_axis, shard in zip(out_avals, out_axes, sharded_outs)]
  return outs

mlir.register_lowering(xla_pmap_p, _pmap_lowering)


# ------------------- xmap -------------------

class Mesh:

  def __init__(self, devices: np.ndarray, axis_names: Sequence[MeshAxisName]):
    assert devices.ndim == len(axis_names)
    # TODO: Make sure that devices are unique? At least with the quick and
    #       dirty check that the array size is not larger than the number of
    #       available devices?
    self.devices = devices.copy()
    self.devices.flags.writeable = False
    self.axis_names = tuple(axis_names)

  def __eq__(self, other):
    if not isinstance(other, Mesh):
      return False
    return (self.axis_names == other.axis_names and
            np.array_equal(self.devices, other.devices))

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self.axis_names, tuple(self.devices.flat)))
    return self._hash

  def __setattr__(self, name, value):
    if hasattr(self, name):
      raise RuntimeError("Cannot reassign attributes of immutable mesh objects")
    super().__setattr__(name, value)

  @property
  def shape(self):
    return OrderedDict((name, size) for name, size in safe_zip(self.axis_names, self.devices.shape))

  @property
  def size(self):
    return np.prod(list(self.shape.values()))

  @property
  def empty(self):
    return self.devices.ndim == 0

  @property
  def is_multi_process(self):
    return self.devices.size != len(self.local_devices)

  @maybe_cached_property
  def local_mesh(self):
    return self._local_mesh(xb.process_index())

  def _local_mesh(self, process_index):
    if self.empty:
      return self
    is_local_device = np.vectorize(
        lambda d: d.process_index == process_index, otypes=[bool])(self.devices)
    subcube_indices = []
    # We take the smallest slice of each dimension that doesn't skip any local device.
    for axis in range(self.devices.ndim):
      other_axes = tuple_delete(tuple(range(self.devices.ndim)), axis)
      # NOTE: This re-reduces over many axes multiple times, so we could definitely
      #       optimize it, but I hope it won't be a bottleneck anytime soon.
      local_slices = is_local_device.any(other_axes, keepdims=False)
      nonzero_indices = np.flatnonzero(local_slices)
      start, end = int(np.min(nonzero_indices)), int(np.max(nonzero_indices))
      subcube_indices.append(slice(start, end + 1))
    subcube_indices = tuple(subcube_indices)
    # We only end up with all conditions being true if the local devices formed a
    # subcube of the full array. This is because we were biased towards taking a
    # "hull" spanned by the devices, and in case the local devices don't form a
    # subcube that hull will contain non-local devices.
    if not is_local_device[subcube_indices].all():
      raise ValueError(
          "When passing non-GlobalDeviceArray inputs to pjit or xmap, devices "
          "connected to a single host must form a contiguous subcube of the "
          "global device mesh")
    return Mesh(self.devices[subcube_indices], self.axis_names)

  @property
  def device_ids(self):
    assert not self.empty
    return np.vectorize(lambda d: d.id, otypes=[int])(self.devices)

  def __repr__(self):
    if self.empty:
      return "Mesh([], ())"
    return f"Mesh({self.device_ids!r}, {self.axis_names!r})"

  @maybe_cached_property
  def local_devices(self):
    process_index = xb.process_index()
    return [d for d in self.devices.flat if d.process_index == process_index]

  def local_to_global(self, axes: ArrayMapping, aval):
    return untile_aval_nd(self.shape, axes,
                          tile_aval_nd(self.local_mesh.shape, axes, aval))

  def global_to_local(self, axes: ArrayMapping, aval):
    return untile_aval_nd(self.local_mesh.shape, axes,
                          tile_aval_nd(self.shape, axes, aval))


def tile_aval_nd(axis_sizes, in_axes: ArrayMapping, aval):
  if aval is core.abstract_unit:
    return aval
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
  if aval is core.abstract_unit:
    return aval
  assert isinstance(aval, ShapedArray)
  shape = list(aval.shape)
  named_shape = dict(aval.named_shape)
  for name, axis in out_axes.items():
    shape[axis] *= axis_sizes[name]
    named_shape.pop(name, None)  # The name might be missing --- it's a broadcast.
  return aval.update(shape=tuple(shape), named_shape=named_shape)


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

@profiler.annotate_function
def lower_mesh_computation(
    fun: lu.WrappedFun,
    transformed_name: str,
    mesh: Mesh,
    in_axes: Sequence[ArrayMapping],
    out_axes: Union[Sequence[ArrayMapping], Callable[[], Sequence[ArrayMapping]]],
    donated_invars: Sequence[bool],
    spmd_lowering: bool,
    global_in_avals: Sequence[core.ShapedArray],
    tile_by_mesh_axes: bool,
    in_is_gda: Sequence[bool]):
  assert not mesh.empty
  backend = xb.get_device_backend(mesh.devices.flat[0])
  name_stack = extend_name_stack(wrap_name(transformed_name, 'xmap'))

  global_axis_sizes = mesh.shape

  log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
  logging.log(log_priority,
              "Compiling %s (%d) for %s mesh with global shapes and types %s. "
              "Argument mapping: %s.",
              getattr(fun, '__name__', '<unnamed function>'), id(fun),
              tuple(global_axis_sizes.items()), global_in_avals,
              in_axes)

  # 1. Trace to jaxpr and preprocess/verify it
  in_tiled_avals = [tile_aval_nd(global_axis_sizes, aval_in_axes, aval)
                    for aval, aval_in_axes in safe_zip(global_in_avals, in_axes)]
  if spmd_lowering:
    # TODO: Consider handling xmap's 'vectorize' in here. We can vmap once instead of vtile twice!
    if tile_by_mesh_axes:
      assert not callable(out_axes)
      fun = vtile_by_mesh(fun, mesh, in_axes, out_axes)
    in_jaxpr_avals = global_in_avals
  else:
    assert tile_by_mesh_axes
    in_jaxpr_avals = in_tiled_avals
  with core.extend_axis_env_nd(mesh.shape.items()):
    with dispatch.log_elapsed_time(f"Finished tracing + transforming {name_stack} "
                                   "in {elapsed_time} sec"):
      jaxpr, out_jaxpr_avals, consts = pe.trace_to_jaxpr_final(fun, in_jaxpr_avals)
  if callable(out_axes):
    out_axes = out_axes()
  assert len(out_axes) == len(out_jaxpr_avals)
  if spmd_lowering:
    global_out_avals = out_jaxpr_avals
  else:
    global_out_avals = [untile_aval_nd(global_axis_sizes, aval_out_axes, aval)
                        for aval, aval_out_axes in safe_zip(out_jaxpr_avals, out_axes)]
  _sanitize_mesh_jaxpr(jaxpr)
  if mesh.is_multi_process:
    check_multihost_collective_allowlist(jaxpr)
  jaxpr = core.apply_outfeed_rewriter(jaxpr)

  # 3. Build up the HLO
  tuple_args = len(in_jaxpr_avals) > 100  # pass long arg lists as tuple for TPU
  in_partitions: Optional[List[Optional[xc.OpSharding]]]
  out_partitions: Optional[List[Optional[xc.OpSharding]]]
  if spmd_lowering:
    replicated_args = [False] * len(in_jaxpr_avals)
    global_sharding_spec = mesh_sharding_specs(global_axis_sizes, mesh.axis_names)
    in_partitions = [global_sharding_spec(aval, aval_in_axes).sharding_proto()
                     if aval is not core.abstract_unit else None
                     for aval, aval_in_axes in safe_zip(global_in_avals, in_axes)]
    out_partitions = [global_sharding_spec(aval, aval_out_axes).sharding_proto()
                      for aval, aval_out_axes in safe_zip(global_out_avals, out_axes)]
    out_partitions_t = xla.tuple_sharding_proto(out_partitions)
    partitions_proto = True
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
  else:
    replicated_args = [not axis for axis in in_axes]
    in_partitions = None
    out_partitions = None
    out_partitions_t = None
    partitions_proto = False
    axis_env = xla.AxisEnv(nreps=mesh.size,
                           names=tuple(global_axis_sizes.keys()),
                           sizes=tuple(global_axis_sizes.values()))
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  module: Union[str, xc.XlaComputation]
  module_name = f"xmap_{fun.__name__}"
  with core.extend_axis_env_nd(mesh.shape.items()):
    if config.jax_enable_mlir:
      module = mlir.lower_jaxpr_to_module(
          module_name, closed_jaxpr, backend.platform, axis_env, name_stack,
          donated_invars, replicated_args=replicated_args,
          arg_shardings=in_partitions, result_shardings=out_partitions)
    else:
      module = xla.lower_jaxpr_to_xla_module(
          module_name, closed_jaxpr, backend.platform, axis_env,
          name_stack, tuple_args, donated_invars, replicated_args,
          in_partitions, out_partitions_t,
          partitions_are_protos=partitions_proto)

  return MeshComputation(
      name_stack, module, donated_invars, mesh=mesh, global_in_avals=global_in_avals,
      global_out_avals=global_out_avals, in_axes=in_axes, out_axes=out_axes,
      spmd_lowering=spmd_lowering, tuple_args=tuple_args, in_is_gda=in_is_gda)


class MeshComputation:
  _executable: Optional['MeshExecutable']

  def __init__(self, name: str, hlo: Union[str, xc.XlaComputation],
               donated_invars: Sequence[bool], **compile_args):
    self._name = name
    self._hlo = hlo
    self._donated_invars = donated_invars
    self.compile_args = compile_args
    self._executable = None

  def hlo(self):
    # this is a method for api consistency with dispatch.XlaComputation
    if isinstance(self._hlo, xc.XlaComputation):
      return self._hlo
    return xe.mlir.mlir_module_to_xla_computation(
        mlir.module_to_string(self._hlo),
        use_tuple_args=self.compile_args["tuple_args"])

  def mhlo(self) -> str:
    if isinstance(self._hlo, xc.XlaComputation):
      return xe.mlir.xla_computation_to_mlir_module(self._hlo)
    return mlir.module_to_string(self._hlo)

  def compile(self,
              _allow_propagation_to_outputs : bool = False,
              _allow_compile_replicated : bool = True) -> 'MeshExecutable':
    if self._executable is None:
      self._executable = MeshExecutable.from_hlo(
          self._name, self._hlo, **self.compile_args,
          _allow_propagation_to_outputs=_allow_propagation_to_outputs,
          _allow_compile_replicated=_allow_compile_replicated)  # type: ignore
    return self._executable


def _get_input_metadata(global_in_avals, global_mesh, in_axes, in_is_gda):
  input_specs, input_indices, input_avals = [], [], []
  for gaval, axis, is_gda in safe_zip(global_in_avals, in_axes, in_is_gda):
    if is_gda:
      aval = gaval
      mesh = global_mesh
    else:
      aval = global_mesh.global_to_local(axis, gaval)
      mesh = global_mesh.local_mesh

    spec = (mesh_sharding_specs(mesh.shape, mesh.axis_names)(aval, axis)
            if aval is not core.abstract_unit else None)
    index = spec_to_indices(aval.shape, spec) if spec is not None else None
    input_specs.append(spec)
    input_indices.append(index)
    input_avals.append(aval)
  return input_specs, input_indices, input_avals


class MeshExecutable:
  __slots__ = ['xla_executable', 'unsafe_call', '_input_avals']

  def __init__(self, xla_executable, unsafe_call, input_avals):
    self.xla_executable = xla_executable
    self.unsafe_call = unsafe_call
    # input_avals is a list of global and local avals. Aval is global if input
    # is a GDA else local.
    self._input_avals = input_avals

  @staticmethod
  def from_hlo(name: str,
               computation: Union[str, xc.XlaComputation],
               mesh: Mesh,
               global_in_avals: Sequence[ShapedArray],
               global_out_avals: Sequence[ShapedArray],
               in_axes: Sequence[ArrayMapping],
               out_axes: Sequence[ArrayMapping],
               spmd_lowering: bool, tuple_args: bool,
               in_is_gda: Sequence[bool],
               _allow_propagation_to_outputs: bool,
               _allow_compile_replicated: bool) -> 'MeshExecutable':
    assert not mesh.empty
    backend = xb.get_device_backend(mesh.devices.flat[0])

    if spmd_lowering:
      num_replicas, num_partitions = 1, mesh.size
    else:
      num_replicas, num_partitions = mesh.size, 1
    device_assignment = mesh.device_ids.reshape((num_replicas, num_partitions))
    compile_options = xb.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=spmd_lowering,
    )
    compile_options.parameter_is_tupled_arguments = tuple_args
    compile_options.executable_build_options.allow_spmd_sharding_propagation_to_output = \
        _allow_propagation_to_outputs

    input_specs, input_indices, input_avals = _get_input_metadata(
        global_in_avals, mesh, in_axes, in_is_gda)
    # Calculate local information here instead of calculating it in
    # `avals_to_results_handler` because pmap also uses this function.
    handle_outs = global_avals_to_results_handler(global_out_avals, out_axes, mesh)

    if _allow_compile_replicated and hasattr(backend, "compile_replicated"):
      unsafe_call = backend.compile_replicated(
          computation, compile_options,
          input_indices, input_specs,
          handle_outs)
      xla_executable = None
    else:
      with dispatch.log_elapsed_time(f"Finished XLA compilation of {name} "
                                     "in {elapsed_time} sec"):
        compiled = dispatch.compile_or_get_cached(backend, computation, compile_options)
      handle_args = InputsHandler(compiled.local_devices(), input_specs, input_indices)
      unsafe_call = partial(execute_replicated, compiled, backend, handle_args, handle_outs)
      xla_executable = compiled

    return MeshExecutable(xla_executable, unsafe_call, input_avals)

  def call(self, *args):
    # TODO(yashkatariya): Add a AOT lowering test where GDA is an input.
    arg_avals = map(xla.abstractify, args)
    ref_avals = self._input_avals
    dispatch.check_arg_avals_for_call(ref_avals, arg_avals)
    return self.unsafe_call(*args)


_forbidden_primitives = {
  'xla_pmap': 'pmap',
  'sharded_call': 'sharded_jit',
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


def mesh_sharding_specs(axis_sizes, axis_names):
  mesh_axis_pos = {name: i for i, name in enumerate(axis_names)}
  # NOTE: This takes in the non-sharded avals!
  def mk_sharding_spec(aval, aval_axes):
    mesh_mapping = [Replicated(axis_size) for axis_size in axis_sizes.values()]
    if aval is core.abstract_token:
      assert not aval_axes
      return ShardingSpec([], mesh_mapping)
    sharding = [_UNSHARDED_INSTANCE] * len(aval.shape)
    next_sharded_axis = 0
    aval_shape = list(aval.shape)
    # NOTE: sorted is stable, which is important when multiple resources
    #       map to the same axis.
    for name, axis in sorted(aval_axes.items(), key=lambda x: x[1]):
      assert aval_shape[axis] % axis_sizes[name] == 0, (axis_sizes[name], aval.shape[axis])
      aval_shape[axis] //= axis_sizes[name]
      if isinstance(sharding[axis], NoSharding):
        sharding[axis] = Chunked([])
      sharding[axis] = Chunked(sharding[axis].chunks + [axis_sizes[name]])
      assert isinstance(mesh_mapping[mesh_axis_pos[name]], Replicated), \
          "Value mapped to the same mesh axis twice"
      mesh_mapping[mesh_axis_pos[name]] = ShardedAxis(next_sharded_axis)
      next_sharded_axis += 1
    return ShardingSpec(sharding, mesh_mapping)
  return mk_sharding_spec


@contextmanager
def maybe_extend_axis_env(*args, **kwargs):
  with core.extend_axis_env(*args, **kwargs):
    yield

class DynamicAxisEnvFrame(object):
  __slots__ = ["name", "pmap_trace", "hard_size"]
  def __init__(self, name, pmap_trace, hard_size):
    self.name = name
    self.pmap_trace = pmap_trace
    self.hard_size = hard_size

class DynamicAxisEnv(list):
  def __contains__(self, axis_name):
    return axis_name in (frame.name for frame in self)

  def __getitem__(self, axis_name):
    if axis_name not in self:
      raise NameError("unbound axis name: {}".format(axis_name))
    for frame in reversed(self):
      if frame.name == axis_name:
        return frame

    raise AssertionError

  @property
  def sizes(self):
    return tuple(frame.hard_size for frame in self)

  @property
  def nreps(self):
    return prod(frame.hard_size for frame in self)

class _ThreadLocalState(threading.local):
  def __init__(self):
    self.dynamic_axis_env = DynamicAxisEnv()

_thread_local_state = _ThreadLocalState()

def device_put(x, devices: Sequence[xb.xla_client.Device], replicate: bool=False) -> List[xb.xla_client.Buffer]:
  """Call device_put on a sequence of devices and return a flat sequence of buffers."""
  if replicate:
    return list(it.chain.from_iterable(dispatch.device_put(x, device) for device in devices))
  else:
    return list(it.chain.from_iterable(dispatch.device_put(val, device) for val, device in safe_zip(x, devices)))
