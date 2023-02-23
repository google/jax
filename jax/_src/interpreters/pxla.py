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

from __future__ import annotations

import enum
from contextlib import contextmanager, ContextDecorator
from collections import defaultdict, OrderedDict, namedtuple
import dataclasses
from functools import partial, lru_cache, cached_property
import itertools as it
import logging
import operator as op
import sys
import threading
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, FrozenSet,
                    Sequence, Set, Tuple, Type, Union, Iterable, Mapping, cast)

import numpy as np

import jax
from jax.errors import JAXTypeError
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_map

from jax._src import abstract_arrays
from jax._src import api_util
from jax._src import basearray
from jax._src import core
from jax._src import device_array
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import profiler
from jax._src import sharding as sharding_internal
from jax._src import source_info_util
from jax._src import stages
from jax._src import util
from jax._src.abstract_arrays import array_types
from jax._src.config import config
from jax._src.config import flags
from jax._src.core import ConcreteArray, ShapedArray
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import xla
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax._src.lib import pmap_lib
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.util import (unzip3, prod, safe_map, safe_zip, partition_list,
                           new_name_stack, wrap_name, assert_unreachable,
                           tuple_insert, tuple_delete, distributed_debug_log,
                           unzip2, HashableFunction)


# Built in Python lists don't support weak refs but subclasses of lists do.
class WeakRefList(list):
  pass


if sys.version_info >= (3, 9):
  OrderedDictType = OrderedDict
else:
  OrderedDictType = Dict

xe = xc._xla

unsafe_map, map = map, safe_map  # type: ignore

logger = logging.getLogger(__name__)

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

MeshAxisName = Any
OpShardingType = Any

PartitionSpec = sharding_internal.PartitionSpec

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


def _get_logical_mesh_ids(mesh_shape):
  return np.arange(np.prod(mesh_shape)).reshape(mesh_shape)


def sharding_spec_sharding_proto(self, special_axes: Mapping[int, OpShardingType] = {}):
  """Converts a ShardingSpec to an OpSharding proto.

  See
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto#L601
  for details on the OpSharding proto.
  Unfortunately the semantics are not very well described in the proto spec, but the code here might help:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/compiler/xla/experimental/xla_sharding/xla_sharding.py
  """
  mesh_shape = cast(Tuple[int, ...], self.mesh_shape)
  mesh = _get_logical_mesh_ids(self.mesh_shape)

  sharded_axes = {}  # maps sharded axis identifiers to mesh axis indices to which they're mapped
  replicated_maxes = []  # lists mesh axis identifiers to replicate over
  for maxis, assignment in enumerate(self.mesh_mapping):
    if isinstance(assignment, Replicated):
      replicated_maxes.append((maxis, assignment.replicas))
    elif isinstance(assignment, ShardedAxis):
      sharded_axes[assignment.axis] = maxis
    else:
      assert_unreachable(assignment)

  proto = xc.OpSharding()
  if len(replicated_maxes) == len(self.mesh_mapping) and not special_axes:
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

  # Create a partial sharding proto if tensor is replicated or partitioned
  # specially over some mesh axes.
  if replicated_maxes:
    last_tile_dims = []
    axes_by_type: Dict[OpShardingType, List[MeshAxisName]] = {}
    size_by_type: Dict[OpShardingType, int] = defaultdict(lambda: 1)
    assert {x[0] for x in replicated_maxes}.issuperset(set(special_axes.keys()))
    for axis, size in replicated_maxes:
      ty = special_axes.get(axis, xc.OpSharding.Type.REPLICATED)
      axes_by_type.setdefault(ty, []).append(axis)
      size_by_type[ty] *= size
    for ty, axes in sorted(axes_by_type.items(), key=lambda x: x[0].value):
      last_tile_dims.append(ty)
      new_mesh_shape.append(size_by_type[ty])
      mesh_permutation.extend(axes)
    proto.last_tile_dims = last_tile_dims

  proto_mesh = mesh.transpose(mesh_permutation).reshape(new_mesh_shape)
  proto.tile_assignment_dimensions = list(proto_mesh.shape)
  proto.tile_assignment_devices = list(proto_mesh.flat)
  return proto


def get_num_ways_dim_sharded(op_sharding: xc.OpSharding) -> Tuple[Sequence[int], int]:
  partitions = op_sharding.tile_assignment_dimensions
  if op_sharding.last_tile_dims == [xc.OpSharding.Type.REPLICATED]:
    replicate_on_last_tile_dim = True
  else:
    replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim
    if op_sharding.last_tile_dims:
      raise NotImplementedError("Unhandled OpSharding type. Please open a bug report!")
  num_replicas = 1
  if replicate_on_last_tile_dim:
    num_replicas = partitions[-1]
    partitions = partitions[:-1]
  return partitions, num_replicas


def _op_sharding_to_numpy_indices(
    op_sharding: xc.OpSharding, shape: Sequence[int],
    num_devices: int) -> np.ndarray:
  indices = np.empty(num_devices, dtype=np.object_)

  # num_devices is required as an argument when op_sharding is
  # REPLICATED. `jax.device_count()` cannot be used because you can create
  # an opsharding with less number of devices than `jax.device_count()`.
  if is_op_sharding_replicated(op_sharding):
    indices.fill((slice(None),) * len(shape))
    return indices

  assert num_devices == len(op_sharding.tile_assignment_devices)

  partitions, num_replicas = get_num_ways_dim_sharded(op_sharding)
  assert len(partitions) == len(shape), (len(partitions), len(shape))

  axis_indices: List[Sequence[Index]] = []
  for dim, n_shards in zip(shape, partitions):
    if n_shards == 1:
      axis_indices.append([slice(None)])
    elif n_shards > 1:
      shard_size, ragged = divmod(dim, n_shards)
      assert not ragged, (dim, n_shards)
      axis_indices.append([slice(i * shard_size, (i + 1) * shard_size)
                           for i in range(n_shards)])
    else:
      raise AssertionError('Unrecognized number of shards. Please file a bug!')

  device_it = iter(op_sharding.tile_assignment_devices)
  for i, idxs in enumerate(it.product(*axis_indices)):
    for _ in range(num_replicas):
      indices[next(device_it)] = idxs
  return indices


def op_sharding_to_indices(op_sharding: xc.OpSharding, shape: Sequence[int],
                           num_devices: int) -> Tuple[Tuple[slice, ...], ...]:
  indices = _op_sharding_to_numpy_indices(op_sharding, shape, num_devices)
  return tuple(indices.flat)


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

  has_unstacked = any(isinstance(s, Unstacked) for s in self.sharding)
  # Take the op sharding indices generation route for pjit/xmap cases.
  if not has_unstacked:
    op_sharding_proto = sharding_spec_sharding_proto(self)
    return _op_sharding_to_numpy_indices(
        op_sharding_proto, shape, prod(self.mesh_shape)).reshape(self.mesh_shape)

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

def spec_to_indices(shape: Sequence[int],
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

def shard_arg(arg, devices, arg_indices):
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
    # pmap_benchmark.pmapshard_args_benchmark indicates this is faster.
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
  return [shard_arg(arg, devices, indices[i]) for i, arg in enumerate(args)]


shard_arg_handlers: Dict[Any, Callable[[Any, Any, Any], Sequence[Any]]] = {}

def _shard_token(x, devices, indices):
  return device_put(np.zeros((), dtype=np.dtype(np.bool_)), devices, replicate=True)
shard_arg_handlers[core.Token] = _shard_token

def _masked_array_error(x, devices, indices):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")
shard_arg_handlers[np.ma.MaskedArray] = _masked_array_error

def _shard_array(x, devices, indices):
  if x.dtype == dtypes.float0:
    x = np.zeros(x.shape, dtype=np.dtype(bool))
  return device_put([x[i] for i in indices], devices)
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array

def shard_device_array(x, devices, indices):
  start_indices, limit_indices, removed_dims = unzip3(
      as_slice_indices(x, idx) for idx in indices)
  shards = x._multi_slice(start_indices, limit_indices, removed_dims)
  return device_put(shards, devices)
for t in device_array.device_array_types:
  shard_arg_handlers[t] = shard_device_array


# NOTE(skye): we could refactor to generate _multi_slice parameters directly
# from the input ShardingSpec, rather than the indices. However, this would
# require duplicating the ordering logic of spec_to_indices, which is more
# subtle and more likely to change than the index logic we have to support here.
def as_slice_indices(arr: device_array.DeviceArrayProtocol, idx: Index) -> Tuple[
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


class AUTOAxisResource:
  pass
AUTO = AUTOAxisResource()

def is_auto(x):
  return isinstance(x, AUTOAxisResource)


class UnspecifiedValue:
  pass
_UNSPECIFIED = UnspecifiedValue()

def _is_unspecified(x):
  return isinstance(x, UnspecifiedValue)

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
ArrayMappingOrAutoOrUnspecified = Union[ArrayMapping, AUTOAxisResource,
                                        UnspecifiedValue]


def array_mapping_to_axis_resources(array_mapping: ArrayMapping):
  if not array_mapping:
    return PartitionSpec()
  max_index = -1
  reverse_map = defaultdict(list)
  for axis, index in array_mapping.items():
    reverse_map[index].append(axis)
    if index > max_index:
      max_index = index
  partitions = tuple(tuple(reverse_map[i]) if reverse_map[i] else None
                     for i in range(max_index + 1))
  return PartitionSpec(*partitions)


class OutputType(enum.Enum):
  Array = 0
  GlobalDeviceArray = 1
  ShardedDeviceArray = 2


def local_aval_to_result_handler(
    aval: core.AbstractValue,
    sharding: sharding_internal.XLACompatibleSharding,
    indices: Optional[Tuple[Index, ...]],
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
  if config.jax_array:
    output_type = OutputType.Array
  else:
    output_type = OutputType.ShardedDeviceArray
  try:
    return local_result_handlers[(type(aval), output_type)](aval, sharding, indices)
  except KeyError as err:
    raise TypeError(
        f"No pxla_result_handler for type: {type(aval)}") from err

PxlaResultHandler = Callable[..., Callable[[Sequence[xb.xla_client.Buffer]], Any]]
local_result_handlers: Dict[Tuple[Type[core.AbstractValue], OutputType], PxlaResultHandler] = {}

def sda_array_result_handler(aval: ShapedArray, sharding, indices):
  sharding_spec = _get_sharding_specs([sharding], [aval])[0]
  if core.is_opaque_dtype(aval.dtype):
    return aval.dtype._rules.local_sharded_result_handler(
        aval, sharding, indices)
  else:
    return lambda bufs: make_sharded_device_array(aval, sharding_spec, bufs,
                                                  indices)
local_result_handlers[(ShapedArray, OutputType.ShardedDeviceArray)] = sda_array_result_handler
local_result_handlers[(ConcreteArray, OutputType.ShardedDeviceArray)] = sda_array_result_handler


def global_aval_to_result_handler(
    aval: core.AbstractValue, out_sharding, committed: bool,
    is_out_sharding_from_xla: bool
) -> Callable[[Sequence[xb.xla_client.Buffer]], Any]:
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
  if config.jax_array:
    output_type = OutputType.Array
  elif config.jax_parallel_functions_output_gda:
    output_type = OutputType.GlobalDeviceArray
  try:
    return global_result_handlers[(type(aval), output_type)](
        aval, out_sharding, committed, is_out_sharding_from_xla)
  except KeyError as err:
    raise TypeError(
        f"No pxla_result_handler for type: {type(aval)}") from err

global_result_handlers: Dict[Tuple[Type[core.AbstractValue], OutputType], PxlaResultHandler] = {}

### lazy device-memory persistence and result handling

# TODO(jblespiau): Consider removing this option.
_USE_CPP_SDA = True


def _create_pmap_sharding_spec(aval, sharded_dim=0, sharded_dim_size=None):
  if sharded_dim is not None:
    sharded_aval = aval.update(
        shape=aval.shape[:sharded_dim] + aval.shape[sharded_dim+1:])
    if sharded_dim_size is None:
      sharded_dim_size = aval.shape[sharded_dim]
  else:
    assert sharded_dim_size is not None
    sharded_aval = aval

  return _pmap_sharding_spec(sharded_dim_size, sharded_dim_size, 1, None,
                             sharded_aval, sharded_dim)


# TODO(yashkatariya, phawkins): Remove this function after March 15, 2023.
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
  from jax._src import pjit

  if sharding_spec is None:
    sharding_spec = _create_pmap_sharding_spec(aval)

  if jax.config.jax_array:
    mesh = thread_resources.env.physical_mesh
    if mesh.empty:
      sharding = sharding_internal.PmapSharding(
          np.asarray([d.device() for d in device_buffers]), sharding_spec)
    else:
      op_sharding = sharding_spec_sharding_proto(sharding_spec)
      pspec = pjit.parse_flatten_op_sharding(
          op_sharding, mesh)[0].get_partition_spec()
      sharding = sharding_internal.NamedSharding(mesh, pspec)

    return jax.make_array_from_single_device_arrays(
        aval.shape, sharding, device_buffers)  # type: ignore
  else:
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
basearray.Array.register(_SDA_BASE_CLASS)


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
      np.asarray(device_buffers[i])`.
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


def _one_replica_buffer_indices(indices: Tuple[Index, ...]):
  """Returns a set of buffer-indices containing one complete copy of the array."""
  one_replica_indices = []
  seen_index_hashes = set()
  for i, index in enumerate(indices):
    hashed_index = _hashable_index(index)
    if hashed_index not in seen_index_hashes:
      one_replica_indices.append(i)
      seen_index_hashes.add(hashed_index)
  return one_replica_indices


def _sda_one_replica_buffer_indices(self):
  """Indices of buffers containing one complete copy of the array data."""
  if self._one_replica_buffer_indices is None:
    self._one_replica_buffer_indices = _one_replica_buffer_indices(self.indices)
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
    buf.block_until_ready()
  return self


def _sda_value(self):
  if self._npy_value is None:
    self.copy_to_host_async()
    npy_value = np.empty(self.aval.shape, self.aval.dtype)
    for i in self.one_replica_buffer_indices:
      npy_value[self.indices[i]] = np.asarray(self.device_buffers[i])
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
      aval = ShapedArray(buf.shape, self.aval.dtype)
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


def _sda_sharding(self):
  has_unstacked = any(isinstance(s, Unstacked) for s in self.sharding_spec.sharding)
  if has_unstacked:
    devices = np.array([d.device() for d in self.device_buffers])
    return sharding_internal.PmapSharding(devices, self.sharding_spec)
  raise NotImplementedError(
      'SDAs that are the output of pjit/xmap do not have the sharding attribute '
      'implemented. If you are trying to pass the SDA to pjit/xmap, please '
      'use multihost_utils.host_local_array_to_global_array(...) to convert '
      'SDAs to global `jax.Array` and then pass them to pjit/xmap with '
      '`jax_array` enabled.')

# TODO(yashkatariya): Remove this when SDA is deleted. The local import of Array
# will also go away.
def _sda_addressable_shards(self):
  from jax._src import array
  out = []
  for db in self.device_buffers:
    db = dispatch._set_aval(db)
    out.append(array.Shard(db.device(), self.sharding, self.shape, db))
  return out


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
  setattr(sda, "sharding", property(_sda_sharding))
  setattr(sda, "addressable_shards", property(_sda_addressable_shards))

del (_sda_one_replica_buffer_indices, _sda_copy_to_host_async,
     _sda_check_if_deleted, _sda_block_until_ready, _sda_value, _sda__getitem__,
     _sda_sharding, _sda_addressable_shards)


ShardedDeviceArray: Type[object]
if _USE_CPP_SDA:
  ShardedDeviceArray = pmap_lib.ShardedDeviceArrayBase
else:
  ShardedDeviceArray = _ShardedDeviceArray


def _hashable_index(idx):
  return tree_map(lambda x: (x.start, x.stop) if type(x) == slice else x, idx)

# The fast path is handled directly in shard_args().
# TODO(yashkatariya): Move this to array.py when SDA is deleted. The local
# import of Array should go away at that time.
def shard_sharded_device_array_slow_path(x, devices, indices):
  from jax._src.array import ArrayImpl

  candidates = defaultdict(list)
  if isinstance(x, ArrayImpl):
    bufs = x._arrays
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


def _sharded_device_array_mlir_constant_handler(val, canonicalize_types=True):
  return mlir.ir_constants(np.asarray(val),
                           canonicalize_types=canonicalize_types)

def _register_handlers_for_sharded_device_array(sda):
  shard_arg_handlers[sda] = shard_sharded_device_array_slow_path
  mlir.register_constant_handler(sda,
                                 _sharded_device_array_mlir_constant_handler)

  core.pytype_aval_mappings[sda] = abstract_arrays.canonical_concrete_aval
  xla.pytype_aval_mappings[sda] = op.attrgetter("aval")
  xla.canonicalize_dtype_handlers[sda] = identity
  api_util._shaped_abstractify_handlers[sda] = op.attrgetter("aval")

_register_handlers_for_sharded_device_array(_ShardedDeviceArray)
_register_handlers_for_sharded_device_array(pmap_lib.ShardedDeviceArray)

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
    global_arg_shapes: Sequence[Optional[Tuple[int, ...]]],
    is_explicit_global_axis_size: bool,
) -> Callable:
  if (config.jax_disable_jit and config.jax_eager_pmap and
      not is_explicit_global_axis_size and not any(d for d in donated_invars)
      and not all(g is not None for g in global_arg_shapes)):
    def _emap_apply_fn(*args):
      return _emap_impl(fun, *args, backend=backend, axis_name=axis_name,
                        axis_size=axis_size, global_axis_size=global_axis_size,
                        devices=devices, name=name, in_axes=in_axes,
                        out_axes_thunk=out_axes_thunk,
                        donated_invars=donated_invars,
                        global_arg_shapes=global_arg_shapes,
                        is_explicit_global_axis_size=is_explicit_global_axis_size)
    return _emap_apply_fn
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun, fingerprint = parallel_callable(
      fun, backend, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, out_axes_thunk, donated_invars, global_arg_shapes,
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
               global_arg_shapes: Sequence[Optional[Tuple[int, ...]]],
               is_explicit_global_axis_size: bool,
               ):
  from jax._src import array
  # TODO(sharadmv,mattjj): implement these cases
  if any(d for d in donated_invars):
    raise NotImplementedError("Buffer donation not supported in eager pmap.")
  if any(g is not None for g in global_arg_shapes):
    raise NotImplementedError("Global arg shapes not supported in eager pmap.")
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
      if isinstance(outval, (ShardedDeviceArray, array.ArrayImpl)):
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
    if call_primitive is not xla.xla_call_p: raise NotImplementedError
    bind = HashableFunction(
        lambda *args, **kwargs: call_primitive.bind(fun, *args, **kwargs),
        (call_primitive, fun))
    fake_primitive = FakePrimitive(multiple_results=True, bind=bind)
    return self.process_primitive(fake_primitive, tracers, params)

  def process_map(self, call_primitive, fun, tracers, params):
    if params['devices'] is not None:
      raise ValueError("Nested pmap with explicit devices argument.")
    if not config.jax_disable_jit:
      bind = HashableFunction(
          lambda *args, **kwargs: call_primitive.bind(fun, *args, **kwargs),
          (call_primitive, fun))
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

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    bind = HashableFunction(
        lambda *args, **kwargs: primitive.bind(fun, jvp, *args, **kwargs),
        (primitive, fun, jvp))
    fake_primitive = FakePrimitive(multiple_results=True, bind=bind)
    return self.process_primitive(fake_primitive, tracers, {})

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,
                              out_trees):
    bind = HashableFunction(
        lambda *args, **kwargs: primitive.bind(fun, fwd, bwd, *args,
                                               out_trees=out_trees, **kwargs),
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
                      global_arg_shapes: Sequence[Optional[Tuple[int, ...]]],
                      is_explicit_global_axis_size: bool,
                      *avals):
  pmap_computation = lower_parallel_callable(
      fun, backend_name, axis_name, axis_size, global_axis_size, devices, name,
      in_axes, out_axes_thunk, donated_invars, global_arg_shapes,
      is_explicit_global_axis_size, avals)
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
                                   "for pmap in {elapsed_time} sec",
                                   event=dispatch.JAXPR_TRACE_EVENT):
      jaxpr, out_sharded_avals, consts = pe.trace_to_jaxpr_final(
          fun, global_sharded_avals, pe.debug_info_final(fun, "pmap"))
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
  parts = find_partitions(jaxpr)

  num_local_shards = replicas.num_local_replicas * parts.local_num_partitions
  num_global_shards = replicas.num_global_replicas * parts.num_partitions

  shards = ShardInfo(
      sharded_avals, out_sharded_avals, global_sharded_avals,
      num_local_shards, num_global_shards)

  return jaxpr, consts, replicas, parts, shards


def _shardings_to_mlir_shardings(
    shardings: Optional[Sequence[PartitionsOrReplicated]]
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
    global_axis_size: int,
    devices: Optional[Sequence[xc.Device]],
    name: str,
    in_axes: Iterable[Optional[int]],
    out_axes_thunk: Callable[[], Sequence[Optional[int]]],
    donated_invars: Sequence[bool],
    global_arg_shapes: Sequence[Optional[Tuple[int, ...]]],
    is_explicit_global_axis_size: bool,
    avals: Sequence[core.AbstractValue]):
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
        # of devices, but precludes nested sharding (i.e. inner pmaps or
        # sharded_jits).
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
  jaxpr, consts, replicas, parts, shards = stage_parallel_callable(
      pci, fun, global_arg_shapes)

  if logger.isEnabledFor(logging.DEBUG):
    logger.debug("sharded_avals: %s", shards.sharded_avals)
    logger.debug("global_sharded_avals: %s", shards.global_sharded_avals)
    logger.debug("num_replicas: %d  num_local_replicas: %d",
                 replicas.num_global_replicas, replicas.num_local_replicas)
    logger.debug("num_partitions: %d  local_num_partitions: %d",
                 parts.num_partitions, parts.local_num_partitions)
    logger.debug("arg_parts: %s", parts.arg_parts)
    logger.debug("local_arg_parts: %s", parts.local_arg_parts)
    logger.debug("out_parts: %s", parts.out_parts)
    logger.debug("local_out_parts: %s", parts.local_out_parts)
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
  logger.log(log_priority,
              "Compiling %s (%d) for %d devices with args %s. (num_replicas=%d"
              " num_partitions=%d)", fun.__name__, id(fun),
              shards.num_global_shards, avals, replicas.num_global_replicas,
              parts.num_partitions)

  axis_env = xla.AxisEnv(
      replicas.num_global_replicas, (axis_name,), (global_axis_size,))
  name_stack = new_name_stack(wrap_name(name, 'pmap'))
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
    lowering_result = mlir.lower_jaxpr_to_module(
        module_name,
        closed_jaxpr,
        unordered_effects,
        ordered_effects,
        backend,
        (backend.platform,),
        mlir.ReplicaAxisContext(axis_env),
        name_stack,
        donated_invars,
        replicated_args=replicated_args,
        arg_shardings=_shardings_to_mlir_shardings(parts.arg_parts),
        result_shardings=_shardings_to_mlir_shardings(parts.out_parts))
    module, keepalive, host_callbacks = (
        lowering_result.module, lowering_result.keepalive,
        lowering_result.host_callbacks)
  return PmapComputation(module, pci=pci, replicas=replicas, parts=parts,
                         shards=shards, tuple_args=tuple_args,
                         unordered_effects=unordered_effects,
                         ordered_effects=ordered_effects,
                         keepalive=keepalive, host_callbacks=host_callbacks)


class PmapComputation(stages.XlaLowering):
  _hlo: ir.Module
  _executable: Optional[PmapExecutable]

  def __init__(self, hlo: ir.Module, **compile_args):
    self._executable = None
    self._hlo = hlo
    self.compile_args = compile_args

  def _compile_unloaded(self) -> Union[UnloadedPmapExecutable, PmapExecutable]:
    return UnloadedPmapExecutable.from_hlo(self._hlo, **self.compile_args)

  # -- stages.XlaLowering overrides

  def hlo(self) -> xc.XlaComputation:
    # this is a method for api consistency with dispatch.XlaComputation
    return xe.mlir.mlir_module_to_xla_computation(
        mlir.module_to_string(self._hlo),
        use_tuple_args=self.compile_args["tuple_args"])

  def mhlo(self) -> ir.Module:
    return super().mhlo()

  def stablehlo(self) -> ir.Module:
    return self._hlo

  @profiler.annotate_function
  def compile(self) -> PmapExecutable:
    if self._executable is None:
      executable = self._compile_unloaded()
      if isinstance(executable, UnloadedPmapExecutable):
        executable = executable.load()
      self._executable = executable
    return self._executable


@dataclasses.dataclass
class UnloadedPmapExecutable:
  compiled: Any
  backend: xb.XlaBackend
  local_input_avals: Sequence[core.AbstractValue]
  input_shardings: Sequence[sharding_internal.XLACompatibleSharding]
  local_output_avals: Sequence[ShapedArray]
  output_shardings: Sequence[sharding_internal.XLACompatibleSharding]
  unordered_effects: List[core.Effect]
  ordered_effects: List[core.Effect]
  keepalive: Sequence[Any]
  host_callbacks: Sequence[Any]

  @staticmethod
  def from_hlo(xla_computation,
               pci: ParallelCallableInfo,
               replicas: ReplicaInfo,
               parts: PartitionInfo,
               shards: ShardInfo,
               tuple_args: bool,
               unordered_effects: List[core.Effect],
               ordered_effects: List[core.Effect],
               host_callbacks: List[Any],
               keepalive: Any):
    devices = pci.devices
    if devices is None:
      if shards.num_global_shards > xb.device_count(pci.backend):
        msg = ("compiling computation that requires {} logical devices, but only {} XLA "
               "devices are available (num_replicas={}, num_partitions={})")
        raise ValueError(msg.format(shards.num_global_shards,
                                    xb.device_count(pci.backend),
                                    replicas.num_global_replicas,
                                    parts.num_partitions))
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
    device_assignment: np.ndarray = np.array(devices).reshape(
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

    process_index = xb.process_index(pci.backend)
    local_device_assignment = np.array([
        d for d in device_assignment.flat if d.process_index == process_index
    ])

    local_arg_parts_ = parts.local_arg_parts or [None] * len(pci.avals)
    input_sharding_specs = [
        _pmap_sharding_spec(replicas.num_local_replicas, pci.axis_size,
                            parts.local_num_partitions, arg_parts, aval, in_axis)
        for aval, arg_parts, in_axis in safe_zip(
            shards.sharded_avals, local_arg_parts_, pci.in_axes)]
    in_shardings = _get_pmap_sharding(local_device_assignment, input_sharding_specs)
    nouts = len(shards.out_sharded_avals)

    out_parts = (None,) * nouts if parts.out_parts is None else parts.out_parts
    local_out_parts = (None,) * nouts if parts.local_out_parts is None else parts.local_out_parts

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
        for out_parts, aval, out_axis in safe_zip(
            local_out_parts, local_out_avals, pci.out_axes)]
    out_shardings = _get_pmap_sharding(local_device_assignment, out_specs)

    if hasattr(pci.backend, "compile_replicated"):
      input_indices = [
          spec_to_indices(aval.shape, spec)  # pytype: disable=attribute-error
          if spec is not None else None
          for aval, spec in safe_zip(pci.avals, input_sharding_specs)
      ]
      handle_outs = local_avals_to_results_handler(local_unmapped_avals,
                                                   out_shardings)
      return _compile_replicated_pmap_executable_from_hlo(
          xla_computation, pci, input_indices, in_shardings, handle_outs,
          compile_options, host_callbacks, bool(unordered_effects),
          ordered_effects)

    with dispatch.log_elapsed_time(
        f"Finished XLA compilation of {pci.name} in {{elapsed_time}} sec",
         event=dispatch.BACKEND_COMPILE_EVENT):
      compiled = dispatch.compile_or_get_cached(
          pci.backend, xla_computation, compile_options, host_callbacks)

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
    )

  def load(self) -> PmapExecutable:
    input_indices = [
        spec_to_indices(aval.shape, spec.sharding_spec)  # pytype: disable=attribute-error
        if spec.sharding_spec is not None else None
        for aval, spec in safe_zip(self.local_input_avals, self.input_shardings)
    ]
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
    fingerprint = getattr(self.compiled, "fingerprint", None)

    return PmapExecutable(self.compiled, execute_fun, fingerprint,
                          self.local_input_avals)


class PmapExecutable(stages.XlaExecutable):
  __slots__ = ["xla_executable", "unsafe_call", "fingerprint", "in_avals"]

  def __init__(self, xla_executable, unsafe_call, fingerprint, in_avals):
    self.xla_executable = xla_executable
    self.unsafe_call = unsafe_call
    self.fingerprint = fingerprint
    self.in_avals = in_avals

  # -- stages.XlaExecutable overrides

  def xla_extension_executable(self):
    return self.xla_executable

  @profiler.annotate_function
  def call(self, *args):
    # TODO(frostig): do we need to check sharding and sharded avals?
    arg_avals = map(xla.abstractify, args)
    dispatch.check_arg_avals_for_call(self.in_avals, arg_avals)
    return self.unsafe_call(*args)


def _get_pmap_sharding(devices, specs):
  return [sharding_internal.PmapSharding(devices, spec) for spec in specs]


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
  if global_parts is None:
    return local_aval
  assert local_parts is not None
  global_shape = [dim * _safe_div(ngparts, nlparts)
                  for dim, ngparts, nlparts
                  in safe_zip(local_aval.shape, global_parts, local_parts)]
  return local_aval.update(shape=global_shape)


def get_local_aval(global_aval, global_parts: PartitionsOrReplicated,
                   local_parts: PartitionsOrReplicated):
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
  __slots__ = ("handler", "local_devices", "in_shardings", "input_indices")

  def __init__(self, local_devices, in_shardings, input_indices):
    self.handler = partial(shard_args, local_devices, input_indices)
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


def _get_sharding_specs(
    shardings: Sequence[sharding_internal.XLACompatibleSharding], avals: Sequence[ShapedArray]
) -> Sequence[ShardingSpec]:
  if all(isinstance(s, sharding_internal.PmapSharding) for s in shardings):
    return [s.sharding_spec for s in shardings]  # type: ignore
  elif all(isinstance(s, sharding_internal.NamedSharding) for s in shardings):
    return [new_mesh_sharding_specs(s.mesh.shape, s.mesh.axis_names)(
              aval.ndim, get_array_mapping(s.spec))
            for aval, s in safe_zip(avals, shardings)]
  else:
    raise ValueError('Getting sharding spec is only supported for '
                     "PmapSharding and NamedSharding, "
                     f"but got {shardings}.")

def local_avals_to_results_handler(
    unmapped_local_out_avals: Sequence[ShapedArray],
    local_shardings: Sequence[sharding_internal.XLACompatibleSharding]) -> ResultsHandler:
  out_indices = [tuple(s.devices_indices_map(aval.shape).values())
                 for s, aval in safe_zip(local_shardings, unmapped_local_out_avals)]
  handlers = [
      local_aval_to_result_handler(aval, s, idcs)
      for aval, s, idcs in safe_zip(unmapped_local_out_avals, local_shardings, out_indices)
  ]
  return ResultsHandler(handlers, local_shardings, unmapped_local_out_avals)


def global_avals_to_results_handler(
    global_out_avals: Sequence[ShapedArray],
    shardings: Sequence[sharding_internal.XLACompatibleSharding],
    committed: bool,
    are_out_shardings_from_xla: Sequence[bool]) -> ResultsHandler:
  if config.jax_parallel_functions_output_gda or config.jax_array:
    handlers = [
        global_aval_to_result_handler(global_aval, s, committed, x)
        for global_aval, s, x in safe_zip(global_out_avals, shardings,
                                          are_out_shardings_from_xla)
    ]
    return ResultsHandler(handlers, shardings, global_out_avals)
  else:
    # This path is taken when the outputs are SDAs.
    assert all(isinstance(s, sharding_internal.NamedSharding) for s in shardings)
    local_out_avals = [s.mesh._global_to_local(get_array_mapping(s.spec), aval)
                       for aval, s in safe_zip(global_out_avals, shardings)]
    local_shardings = [sharding_internal.NamedSharding(s.mesh.local_mesh, s.spec)  # type: ignore
                       for s in shardings]
    return local_avals_to_results_handler(local_out_avals, local_shardings)


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

  if jax.config.jax_array:
    sharding = sharding_internal.PmapSharding(
        np.asarray([d.device() for d in device_buffers]), sharding_spec)
    return jax.make_array_from_single_device_arrays(
        replicated_aval.shape, sharding, device_buffers)
  else:
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

  def _call_with_tokens(self, input_bufs):
    if self.ordered_effects:
      device, = self._local_devices
      tokens = [list(dispatch.runtime_tokens.get_token(eff, device))
                for eff in self.ordered_effects]
      input_bufs = [*tokens, *input_bufs]
    num_output_tokens = len(self.ordered_effects)
    out_bufs, sharded_token = (
        self.xla_executable.execute_sharded_on_local_devices_with_tokens(
          input_bufs))
    token_bufs, out_bufs = util.split_list(out_bufs, [num_output_tokens])
    for i, device in enumerate(self._local_devices):
      dispatch.runtime_tokens.set_output_runtime_token(
          device, sharded_token.get_token(i))
    for eff, token_buf in zip(self.ordered_effects, token_bufs):
      dispatch.runtime_tokens.update_token(eff, token_buf)
    return out_bufs

  @profiler.annotate_function
  def __call__(self, *args):
    args = [x for i, x in enumerate(args) if i in self.kept_var_idx]
    input_bufs = self.in_handler(args)
    if (self.ordered_effects or self.has_unordered_effects or
        self.has_host_callbacks):
      out_bufs = self._call_with_tokens(input_bufs)
    else:
      out_bufs = self.xla_executable.execute_sharded_on_local_devices(
          input_bufs)
    if dispatch.needs_check_special():
      for bufs in out_bufs:
        dispatch.check_special(self.name, bufs)
    return self.out_handler(out_bufs)


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
  # TODO(yashkatariya,mattjj): Handle global_arg_shapes here too.
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
pe.call_param_updaters[xla_pmap_p] = pe.call_param_updaters[xla.xla_call_p]
pe.partial_eval_jaxpr_custom_rules[xla_pmap_p] = \
    partial(pe.call_partial_eval_custom_rule,
            'call_jaxpr', _pmap_partial_eval_custom_params_updater,
            res_aval=_pmap_partial_eval_custom_res_maker)
pe.dce_rules[xla_pmap_p] = _pmap_dce_rule
ad.call_param_updaters[xla_pmap_p] = ad.call_param_updaters[xla.xla_call_p]
ad.call_transpose_param_updaters[xla_pmap_p] = \
    ad.call_transpose_param_updaters[xla.xla_call_p]

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
      np.array(axis_env.nreps // util.prod(axis_env.sizes), np.uint32))
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
def _hlo_unshard(ctx: mlir.LoweringRuleContext, aval, axis_env, out_axis, xs, platforms):
  if aval is core.abstract_token:
    return xs
  elif not isinstance(aval, core.ShapedArray):
    raise TypeError(aval)

  x, = xs

  def do_lower(ctx, x, is_cpu_or_gpu=False):
    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    nonlocal aval
    convert_bool = (np.issubdtype(aval.dtype, np.bool_) and is_cpu_or_gpu)
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

  return mlir.lower_multi_platform(ctx, "hlo_unshard",
      [(["cpu", "rocm", "cuda"], partial(do_lower, is_cpu_or_gpu=True)),
       (["tpu"], partial(do_lower, is_cpu_or_gpu=False))], x)


def _pmap_lowering(ctx, *in_nodes, axis_name,
                   axis_size, global_axis_size, devices, name,
                   call_jaxpr, backend=None, in_axes, out_axes,
                   donated_invars, global_arg_shapes,
                   is_explicit_global_axis_size):
  del donated_invars  # Unused.
  # TODO(necula): why do we need this check?
  # xla.check_backend_matches(backend, ctx.module_context.platform)
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
        axis_context=mlir.ReplicaAxisContext(new_env),
        name_stack=util.extend_name_stack(ctx.module_context.name_stack,
                                          util.wrap_name(name, 'pmap')))
    sharded_outs, _ = mlir.jaxpr_subcomp(sub_ctx, call_jaxpr, mlir.TokenSet(), (),
                                         *in_nodes_sharded,
                                         dim_var_values=ctx.dim_var_values)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_hlo_unshard(ctx, aval, new_env, out_axis, shard,
                        platforms=ctx.module_context.platforms)
          for aval, out_axis, shard in zip(out_avals, out_axes, sharded_outs)]
  return outs

mlir.register_lowering(xla_pmap_p, _pmap_lowering)


# ------------------- xmap -------------------

class Mesh(ContextDecorator):
  """Declare the hardware resources available in the scope of this manager.

  In particular, all ``axis_names`` become valid resource names inside the
  managed block and can be used e.g. in the ``in_axis_resources`` argument of
  :py:func:`jax.experimental.pjit.pjit`. Also see JAX's multi-process programming
  model (https://jax.readthedocs.io/en/latest/multi_process.html)
  and the Distributed arrays and automatic parallelization tutorial
  (https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)

  If you are compiling in multiple threads, make sure that the
  ``with Mesh`` context manager is inside the function that the threads will
  execute.

  Args:
    devices: A NumPy ndarray object containing JAX device objects (as
      obtained e.g. from :py:func:`jax.devices`).
    axis_names: A sequence of resource axis names to be assigned to the
      dimensions of the ``devices`` argument. Its length should match the
      rank of ``devices``.

  Example:

    >>> from jax.experimental.pjit import pjit
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> inp = np.arange(16).reshape((8, 2))
    >>> devices = np.array(jax.devices()).reshape(4, 2)
    ...
    >>> # Declare a 2D mesh with axes `x` and `y`.
    >>> global_mesh = Mesh(devices, ('x', 'y'))
    >>> # Use the mesh object directly as a context manager.
    >>> with global_mesh:
    ...   out = pjit(lambda x: x, in_axis_resources=None, out_axis_resources=None)(inp)

    >>> # Initialize the Mesh and use the mesh as the context manager.
    >>> with Mesh(devices, ('x', 'y')) as global_mesh:
    ...   out = pjit(lambda x: x, in_axis_resources=None, out_axis_resources=None)(inp)

    >>> # Also you can use it as `with ... as ...`.
    >>> global_mesh = Mesh(devices, ('x', 'y'))
    >>> with global_mesh as m:
    ...   out = pjit(lambda x: x, in_axis_resources=None, out_axis_resources=None)(inp)

    >>> # You can also use it as `with Mesh(...)`.
    >>> with Mesh(devices, ('x', 'y')):
    ...   out = pjit(lambda x: x, in_axis_resources=None, out_axis_resources=None)(inp)
  """

  devices: np.ndarray
  axis_names: Tuple[MeshAxisName, ...]

  def __init__(self, devices: Union[np.ndarray, Sequence[xc.Device]],
               axis_names: Union[str, Sequence[MeshAxisName]]):
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices)
    if isinstance(axis_names, str):
      axis_names = (axis_names,)
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
    # This is a performance optimization. Comparing thousands of devices
    # can be expensive.
    if id(self) == id(other):
      return True
    return (self.axis_names == other.axis_names and
            np.array_equal(self.devices, other.devices))

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(
          (self.axis_names, tuple(self.devices.flat), self.devices.shape))
    return self._hash

  def __setattr__(self, name, value):
    if hasattr(self, name):
      raise RuntimeError("Cannot reassign attributes of immutable mesh objects")
    super().__setattr__(name, value)

  def __enter__(self):
    new_env = thread_resources.stack[-1].with_mesh(self)
    thread_resources.stack.append(new_env)
    thread_resources.env = new_env
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    thread_resources.stack.pop()
    thread_resources.env = thread_resources.stack[-1]
    return False

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

  @cached_property
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
          "When passing host local inputs to pjit or xmap, devices "
          "connected to a single host must form a contiguous subcube of the "
          "global device mesh")
    return Mesh(self.devices[subcube_indices], self.axis_names)

  @property
  def device_ids(self):
    assert not self.empty
    return np.vectorize(lambda d: d.id, otypes=[int])(self.devices)

  def __repr__(self):
    if self.empty:
      return "Mesh(device_ids=[], axis_names=())"
    return f"Mesh(device_ids={self.device_ids!r}, axis_names={self.axis_names!r})"

  @cached_property
  def local_devices(self):
    return [d for d in self.devices.flat
            if d.process_index == d.client.process_index()]

  def _local_to_global(self, axes: ArrayMapping, aval):
    return untile_aval_nd(self.shape, axes,
                          tile_aval_nd(self.local_mesh.shape, axes, aval))

  def _global_to_local(self, axes: ArrayMapping, aval):
    return untile_aval_nd(self.local_mesh.shape, axes,
                          tile_aval_nd(self.shape, axes, aval))


ResourceAxisName = core.AxisName

class Loop(NamedTuple):
  name: ResourceAxisName
  length: int


def show_axes(axes):
  return ", ".join(sorted(f"`{a}`" for a in axes))


class ResourceEnv(NamedTuple):
  physical_mesh: Mesh
  loops: Tuple[Loop, ...]

  def with_mesh(self, mesh: Mesh):
    overlap = set(mesh.axis_names) & (self.resource_axes - set(self.physical_mesh.axis_names))
    if overlap:
      raise ValueError(f"Cannot update the mesh of the current resource "
                       f"environment. The new mesh shadows already defined axes "
                       f"{show_axes(overlap)}")
    return self._replace(physical_mesh=mesh)

  def with_extra_loop(self, loop: Loop):
    if loop.name in self.resource_axes:
      raise ValueError(f"Cannot extend the resource environment with loop named "
                       f"`{loop.name}`. An axis of this name is already defined!")
    return self._replace(loops=self.loops + (loop,))

  @property
  def physical_resource_axes(self) -> Set[ResourceAxisName]:
    return set(self.physical_mesh.axis_names)

  @property
  def loop_resource_axes(self) -> Set[ResourceAxisName]:
    return {loop.name for loop in self.loops}

  @property
  def resource_axes(self) -> Set[ResourceAxisName]:
    return self.physical_resource_axes | self.loop_resource_axes

  @property
  def shape(self):
    shape = self.physical_mesh.shape
    shape.update(self.loops)
    return shape

  @property
  def local_shape(self):
    shape = self.physical_mesh.local_mesh.shape
    shape.update(self.loops)
    return shape

  def __repr__(self):
    return f"ResourceEnv({self.physical_mesh!r}, {self.loops!r})"

EMPTY_ENV = ResourceEnv(Mesh(np.empty((), dtype=object), ()), ())

class _ThreadResourcesLocalState(threading.local):

  def __init__(self):
    self.stack = [EMPTY_ENV]
    self.env = self.stack[-1]

thread_resources = _ThreadResourcesLocalState()


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

def manual_proto(aval: core.ShapedArray, manual_axes_set: FrozenSet[MeshAxisName], mesh: Mesh):
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
  tad_shape.append(int(np.prod([named_mesh_shape[a] for a in replicated_axes], dtype=int)))
  tad_shape.append(int(np.prod([named_mesh_shape[a] for a in manual_axes], dtype=int)))

  raw_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
  proto = xc.OpSharding()
  proto.type = xc.OpSharding.Type.OTHER
  proto.tile_assignment_dimensions = tad_shape
  proto.tile_assignment_devices = list(raw_mesh.transpose(tad_perm).reshape(tad_shape).flat)
  proto.last_tile_dims = [xc.OpSharding.Type.REPLICATED, xc.OpSharding.Type.MANUAL]
  return proto

@partial(mlir.register_lowering, full_to_shard_p)
def _full_to_shard_lowering(ctx, x, *, axes: ArrayMapping, mesh: Mesh, manual_axes: FrozenSet[MeshAxisName]):
  # TODO: Can we short-circuit for replicated values? Probably not.
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  sharding_proto = mesh_sharding_specs(mesh.shape, mesh.axis_names)(aval_in, axes).sharding_proto()
  unspecified_dims = set(range(aval_in.ndim)) - set(axes.values())
  sx = mlir.wrap_with_sharding_op(x, sharding_proto, unspecified_dims=unspecified_dims)
  proto = manual_proto(aval_in, manual_axes, mesh)
  result_type, = mlir.aval_to_ir_types(aval_out)
  return mlir.wrap_with_full_to_shard_op(result_type, sx, proto, unspecified_dims=unspecified_dims),

shard_to_full_p = core.Primitive('shard_to_full')

@shard_to_full_p.def_abstract_eval
def _shard_to_full_abstract_eval(x, axes, mesh, **_):
  # TODO: Assert x is a global aval! Or ideally check that it's global in dims from axes!
  return untile_aval_nd(mesh.shape, axes, x)

@partial(mlir.register_lowering, shard_to_full_p)
def _shard_to_full_lowering(ctx, x, *, axes: ArrayMapping, mesh: Mesh, manual_axes: FrozenSet[MeshAxisName]):
  aval_in, = ctx.avals_in
  aval_out, = ctx.avals_out
  proto = manual_proto(aval_in, manual_axes, mesh)
  result_type, = mlir.aval_to_ir_types(aval_out)
  unspecified_dims = set(range(aval_in.ndim)) - set(axes.values())
  sx = mlir.wrap_with_sharding_op(x, proto, unspecified_dims=unspecified_dims)
  sharding_proto = mesh_sharding_specs(mesh.shape, mesh.axis_names)(aval_out, axes).sharding_proto()
  return mlir.wrap_with_shard_to_full_op(result_type, sx, sharding_proto, unspecified_dims),

@lu.transformation
def vtile_manual(manual_axes: FrozenSet[MeshAxisName],
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
  manual_axes: FrozenSet[MeshAxisName]

TilingMethod = Union[TileVectorize, TileManual]


class _PositionalSemantics(enum.Enum):
  """Indicates whether the positional shapes of inputs should be interpreted as
  global or local with respect to the multi-host mesh.

  While named axes are always associated with global sizes, the outermost pjit
  is the boundary between the local shapes in the outer scope and global
  positional shapes in its inner scope. pjits nested inside that one should not
  attempt to increase the sizes of avals again, and xmap has to take this into
  account when inferring the global size of a named axis.
  """
  LOCAL = 0
  GLOBAL = 1


class _PSThreadLocalState(threading.local):

  def __init__(self):
    self.val = _PositionalSemantics.LOCAL

positional_semantics = _PSThreadLocalState()


def check_if_any_auto(
    shardings: Iterable[Union[sharding_internal.XLACompatibleSharding,
                              AUTOAxisResource, UnspecifiedValue]]) -> bool:
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
    return (f'{self.source_info.eqn_name} inside {api_name}'
            if self.m_type == MismatchType.SHARDING_INSIDE_COMPUTATION else self.m_type)

  def _str(self, api_name):
    return (f"{self._maybe_api_name(api_name)} {self.m_type_str(api_name)} with "
            f"{self._dev_ids_plat_str}{self.source_info_str}")


class DeviceAssignmentMismatchError(Exception):
  pass


ShardingInfo = Tuple[
    Union[sharding_internal.XLACompatibleSharding, UnspecifiedValue,
          AUTOAxisResource],
    MismatchType, Optional[Any]]  # Any is dispatch.SourceInfo to avoid circular imports

def _get_and_check_device_assignment(
    shardings: Iterable[ShardingInfo],
    devices: Optional[Sequence[xc.Device]],
) -> Tuple[xc.Client, Sequence[xc.Device]]:
  from jax._src.api import local_devices

  first_sharding_info = None
  if devices is None:
    devices = []
  else:
    devices = list(devices)

  for i, s_type, source_info in shardings:
    if is_auto(i) or _is_unspecified(i):
      continue
    # Assign `first_sharding_info` after `AUTO` and `UNSPECIFIED` have been
    # skipped.
    if first_sharding_info is None:
      first_sharding_info = (list(i._device_assignment), s_type, source_info)  # type: ignore
    arr_device_assignment = list(i._device_assignment)  # type: ignore
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
    final_device_assignment = [config.jax_default_device or local_devices()[0]]
  else:
    final_device_assignment = first_sharding_info[0]
  return xb.get_device_backend(final_device_assignment[0]), final_device_assignment


@profiler.annotate_function
def lower_sharding_computation(
    fun: lu.WrappedFun,
    api_name: str,
    fun_name: str,
    in_shardings: Sequence[Union[sharding_internal.XLACompatibleSharding, UnspecifiedValue]],
    out_shardings: Union[Sequence[Union[sharding_internal.XLACompatibleSharding, UnspecifiedValue]], UnspecifiedValue],
    donated_invars: Sequence[bool],
    global_in_avals: Sequence[core.ShapedArray],
    in_is_global: Sequence[bool],
    keep_unused: bool,
    always_lower: bool,
    devices_from_context: Optional[Sequence[xc.Device]] = None,
    lowering_platforms_override: Optional[str] = None,
) -> MeshComputation:
  """Lowers a computation to XLA. It can take arbitrary shardings as input.

  The caller of this code can pass in a singleton _UNSPECIFIED because the
  number of out_avals might not be known at that time and
  lower_sharding_computation calculates the number of out_avals so it can apply
  the singleton _UNSPECIFIED to all out_avals.
  """
  # 1. Trace to jaxpr and preprocess/verify it
  name_stack = new_name_stack(wrap_name(fun_name, api_name))

  with dispatch.log_elapsed_time(f"Finished tracing + transforming {name_stack} "
                                 "in {elapsed_time} sec",
                                 event=dispatch.JAXPR_TRACE_EVENT):
    jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_final(
        fun, global_in_avals, debug_info=pe.debug_info_final(fun, api_name))
  kept_outputs = [True] * len(global_out_avals)

  if _is_unspecified(out_shardings):
    out_shardings = (_UNSPECIFIED,) * len(global_out_avals)
  # mypy doesn't understand that out_sharding here is always a sequence.
  assert len(out_shardings) == len(global_out_avals), (  # type: ignore
      len(out_shardings), len(global_out_avals))  # type: ignore

  # Device assignment across all inputs, outputs and shardings inside jaxpr
  # should be the same.
  jaxpr_sharding = list(dispatch.jaxpr_shardings(jaxpr))
  backend, device_assignment = _get_and_check_device_assignment(
      it.chain([(i, MismatchType.ARG_SHARDING, None) for i in in_shardings],
               [(o, MismatchType.OUT_SHARDING, None) for o in out_shardings],  # type: ignore
               [(js, MismatchType.SHARDING_INSIDE_COMPUTATION, source_info)  # type: ignore
                for js, source_info in jaxpr_sharding]),
      devices_from_context)

  # TODO(yashkatariya): Make this logic work after DCE because there can be
  # equations inside the jaxpr that don't affect the output so whether the
  # output(s) are committed or not should not depend on it.
  committed = bool(
      devices_from_context or
      len(device_assignment) > 1 or
      any(not _is_unspecified(i) for i in in_shardings) or
      any(not _is_unspecified(js) for js, _ in jaxpr_sharding) or  # type: ignore
      any(not _is_unspecified(o) for o in out_shardings))  # type: ignore

  in_shardings = tuple(sharding_internal.GSPMDSharding.get_replicated(device_assignment)
                       if _is_unspecified(i) else i for i in in_shardings)

  log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
  logger.log(log_priority,
              "Compiling %s (%d) for with global shapes and types %s. "
              "Argument mapping: %s.",
              getattr(fun, '__name__', '<unnamed function>'), id(fun),
              global_in_avals, in_shardings)

  if keep_unused:
    kept_var_idx = set(range(len(global_in_avals)))
  else:
    jaxpr, kept_const_idx, kept_var_idx = dispatch._prune_unused_inputs(jaxpr)
    consts = [c for i, c in enumerate(consts) if i in kept_const_idx]
    global_in_avals = tuple(a for i, a in enumerate(global_in_avals) if i in kept_var_idx)
    in_shardings = tuple(s for i, s in enumerate(in_shardings) if i in kept_var_idx)
    in_is_global = tuple(g for i, g in enumerate(in_is_global) if i in kept_var_idx)
    donated_invars = tuple(x for i, x in enumerate(donated_invars) if i in kept_var_idx)
    del kept_const_idx

  local_device_assignment = [d for d in device_assignment
                             if d.process_index == d.client.process_index()]
  if len(device_assignment) != len(local_device_assignment):
    check_multihost_collective_allowlist(jaxpr)
    # TODO(yashkatariya): Once jit and pjit's frontend is merged, use the
    # argument on jit `_allow_multiprocess` (which will be added later) instead
    # of the `api_name` check here.
    # Furthermore, `allow_jit` is not allowed yet because `allow_jit` only
    # allows explicit `jax.jit` to work but not implicitly jitted `jnp`.
    # operations. This restriction will be relaxed in the future when the
    # default value of `spmd_mode` config changes to `allow_jit`.
    if (config.jax_array and api_name == 'jit' and
        config.jax_spmd_mode != 'allow_all'):
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
  jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)

  # Computations that only produce constants and/or only rearrange their inputs,
  # which are often produced from partial evaluation, don't need compilation,
  # and don't need to evaluate their arguments.
  if (not always_lower and not (jaxpr.effects or has_outfeed) and
      (not jaxpr.eqns and all(kept_outputs) or not jaxpr.outvars) and
      all(_is_unspecified(o) for o in out_shardings)):  # type: ignore
    return MeshComputation(
        str(name_stack), None, True, donated_invars, jaxpr=jaxpr, consts=consts,
        global_in_avals=global_in_avals, global_out_avals=global_out_avals,
        in_shardings=in_shardings, backend=backend,
        device_assignment=device_assignment, committed=committed,
        kept_var_idx=kept_var_idx, keepalive=None)

  # Look at the number of replcas present in the jaxpr. In
  # lower_sharding_computation, nreps > 1 during `jit(pmap)` cases. This is
  # handled here so as to deprecate the lower_xla_callable codepath when
  # `jax.Array` is turned on by default.
  # TODO(yashkatariya): Remove this when `jit(pmap)` is removed.
  nreps = dispatch.jaxpr_replicas(jaxpr)
  dispatch.raise_warnings_or_errors_for_jit_of_pmap(nreps, backend, fun_name, jaxpr)

  # 2. Build up the HLO
  tuple_args = dispatch.should_tuple_args(len(global_in_avals), backend.platform)

  in_op_shardings: Optional[List[Optional[xc.OpSharding]]]
  out_op_shardings: Optional[List[Optional[xc.OpSharding]]]
  axis_ctx: mlir.AxisContext

  if nreps == 1:
    in_op_shardings = []
    for aval, i in safe_zip(global_in_avals, in_shardings):
      if aval is core.abstract_token:
        in_op_shardings.append(None)
      elif core.is_opaque_dtype(aval.dtype):
        in_op_shardings.append(aval.dtype._rules.physical_op_sharding(aval, i))
      else:
        in_op_shardings.append(i._to_xla_op_sharding(aval.ndim))  # type: ignore[union-attr]

    # TODO(yashkatariya): Fix the HLO produced if out_partitions is
    # [None, OpShardingProto] has the sharding annotations.
    out_op_shardings = []
    for aval, o in safe_zip(global_out_avals, out_shardings):  # type: ignore[arg-type]
      if _is_unspecified(o) or aval is core.abstract_token:
        out_op_shardings.append(None)
      elif core.is_opaque_dtype(aval.dtype):
        out_op_shardings.append(aval.dtype._rules.physical_op_sharding(aval, o))
      else:
        out_op_shardings.append(o._to_xla_op_sharding(aval.ndim))  # type: ignore[union-attr]
    replicated_args = [False] * len(global_in_avals)
    axis_ctx = mlir.ShardingContext(device_assignment)
  else:
    # This path is triggered for `jit(pmap)` cases.
    replicated_args = None
    in_op_shardings = None
    out_op_shardings = None
    axis_env = xla.AxisEnv(nreps, (), ())
    axis_ctx = mlir.ReplicaAxisContext(axis_env)

  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  module_name = f"{api_name}_{fun_name}"

  if len(device_assignment) > 1:
    if any(effects.ordered_effects.contains(eff) for eff
           in closed_jaxpr.effects):
      raise ValueError("Ordered effects are not supported for more than 1 device.")
  unordered_effects = list(
      effects.ordered_effects.filter_not_in(closed_jaxpr.effects))
  ordered_effects = list(effects.ordered_effects.filter_in(closed_jaxpr.effects))
  lowering_result = mlir.lower_jaxpr_to_module(
      module_name,
      closed_jaxpr,
      unordered_effects,
      ordered_effects,
      backend,
      lowering_platforms_override or (backend.platform,),
      axis_ctx,
      name_stack,
      donated_invars,
      replicated_args=replicated_args,
      arg_shardings=in_op_shardings,
      result_shardings=out_op_shardings)

  module, keepalive, host_callbacks = (
      lowering_result.module, lowering_result.keepalive,
      lowering_result.host_callbacks)

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
      mesh=None,
      global_in_avals=global_in_avals,
      global_out_avals=global_out_avals,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      spmd_lowering=True,
      tuple_args=tuple_args,
      in_is_global=in_is_global,
      auto_spmd_lowering=False,
      unordered_effects=unordered_effects,
      ordered_effects=ordered_effects,
      host_callbacks=host_callbacks,
      keepalive=keepalive,
      kept_var_idx=kept_var_idx,
      backend=backend,
      device_assignment=device_assignment,
      committed=committed,
      pmap_nreps=nreps)


@profiler.annotate_function
def lower_mesh_computation(
    fun: lu.WrappedFun,
    api_name: str,
    fun_name: str,
    mesh: Mesh,
    in_shardings: Sequence[Union[sharding_internal.NamedSharding, AUTOAxisResource]],
    out_shardings: Sequence[Union[sharding_internal.NamedSharding, AUTOAxisResource,
                            UnspecifiedValue]],
    donated_invars: Sequence[bool],
    spmd_lowering: bool,
    global_in_avals: Sequence[core.ShapedArray],
    tiling_method: Optional[TilingMethod],
    in_is_global: Sequence[bool],
    lowering_platforms_override: Sequence[str] = ()) -> MeshComputation:
  assert not mesh.empty
  backend = xb.get_device_backend(mesh.devices.flat[0])
  name_stack = new_name_stack(wrap_name(fun_name, api_name))

  auto_spmd_lowering = check_if_any_auto(in_shardings + out_shardings)  # type: ignore

  if auto_spmd_lowering and not spmd_lowering:
    raise ValueError('Enable spmd_lowering to use auto spmd lowering.')

  global_axis_sizes = mesh.shape

  log_priority = logging.WARNING if config.jax_log_compiles else logging.DEBUG
  logger.log(log_priority,
              "Compiling %s (%d) for %s mesh with global shapes and types %s. "
              "Argument mapping: %s.",
              getattr(fun, '__name__', '<unnamed function>'), id(fun),
              tuple(global_axis_sizes.items()), global_in_avals,
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
      assert not auto_spmd_lowering
      # This is the xmap path where there is no `AUTO` or `UNSPECIFIED`, which
      # is why `.spec` can be accessed.
      fun = tiling_transform(
          fun, mesh, [get_array_mapping(i.spec) for i in in_shardings],  # type: ignore
          [get_array_mapping(o.spec) for o in out_shardings])  # type: ignore
    in_jaxpr_avals = global_in_avals
  else:
    assert isinstance(tiling_method, TileVectorize)
    assert not auto_spmd_lowering
    # In non-spmd lowering path, there is no `AUTO` or `UNSPECIFIED`, which is
    # why `.spec` can be accessed.
    in_tiled_avals = [tile_aval_nd(global_axis_sizes, get_array_mapping(i.spec), aval)  # type: ignore
                      for aval, i in safe_zip(global_in_avals, in_shardings)]
    in_jaxpr_avals = in_tiled_avals
  with core.extend_axis_env_nd(mesh.shape.items()):
    with dispatch.log_elapsed_time(f"Finished tracing + transforming {name_stack} "
                                   "in {elapsed_time} sec",
                                   event=dispatch.JAXPR_TRACE_EVENT):
      jaxpr, out_jaxpr_avals, consts = pe.trace_to_jaxpr_final(fun, in_jaxpr_avals)
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

  in_partitions: Optional[List[Optional[xc.OpSharding]]]
  out_partitions: Optional[List[Optional[xc.OpSharding]]]
  axis_ctx: mlir.AxisContext
  if spmd_lowering:
    in_partitions = []
    for aval, i in safe_zip(global_in_avals, in_shardings):
      if is_auto(i):
        in_partitions.append(None)
      elif core.is_opaque_dtype(aval.dtype):
        in_partitions.append(aval.dtype._rules.physical_op_sharding(aval, i))
      else:
        in_partitions.append(i._to_xla_op_sharding(aval.ndim))  # type: ignore[union-attr]

    # TODO(yashkatariya): Fix the HLO produced if out_partitions is
    # [None, OpShardingProto] has the sharding annotations.
    out_partitions = []
    for aval, o in safe_zip(global_out_avals, out_shardings):
      if is_auto(o) or _is_unspecified(o):
        out_partitions.append(None)
      elif core.is_opaque_dtype(aval.dtype):
        out_partitions.append(aval.dtype._rules.physical_op_sharding(aval, o))
      else:
        out_partitions.append(o._to_xla_op_sharding(aval.ndim))  # type: ignore[union-attr]
    replicated_args = [False] * len(in_jaxpr_avals)
    axis_ctx = mlir.SPMDAxisContext(mesh, manual_axes)
  else:
    replicated_args = [not get_array_mapping(i.spec) for i in in_shardings]  # type: ignore
    in_partitions = None
    out_partitions = None
    axis_env = xla.AxisEnv(nreps=mesh.size,
                           names=tuple(global_axis_sizes.keys()),
                           sizes=tuple(global_axis_sizes.values()))
    axis_ctx = mlir.ReplicaAxisContext(axis_env)
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  module: Union[str, xc.XlaComputation]
  module_name = f"{api_name}_{fun_name}"
  with core.extend_axis_env_nd(mesh.shape.items()):
    if any(effects.ordered_effects.contains(eff) for eff
           in closed_jaxpr.effects):
      raise ValueError("Ordered effects not supported in mesh computations.")
    unordered_effects = list(effects.ordered_effects.filter_not_in(
      closed_jaxpr.effects))
    ordered_effects = list(effects.ordered_effects.filter_in(
      closed_jaxpr.effects))
    lowering_result = mlir.lower_jaxpr_to_module(
        module_name,
        closed_jaxpr,
        unordered_effects,
        ordered_effects,
        backend,
        lowering_platforms_override or (backend.platform,),
        axis_ctx,
        name_stack,
        donated_invars,
        replicated_args=replicated_args,
        arg_shardings=in_partitions,
        result_shardings=out_partitions)
    module, keepalive, host_callbacks = (
        lowering_result.module, lowering_result.keepalive,
        lowering_result.host_callbacks)
  return MeshComputation(
      str(name_stack),
      module,
      False,
      donated_invars,
      mesh=mesh,
      global_in_avals=global_in_avals,
      global_out_avals=global_out_avals,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      spmd_lowering=spmd_lowering,
      tuple_args=tuple_args,
      in_is_global=in_is_global,
      auto_spmd_lowering=auto_spmd_lowering,
      unordered_effects=unordered_effects,
      ordered_effects=ordered_effects,
      host_callbacks=host_callbacks,
      keepalive=keepalive,
      kept_var_idx=set(range(len(global_in_avals))),
      backend=backend,
      device_assignment=list(mesh.devices.flat),
      committed=True)


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

  def _compile_unloaded(
      self,
      _allow_propagation_to_outputs: Optional[Sequence[bool]] = None,
      _allow_compile_replicated: bool = True
  ) -> Union[UnloadedMeshExecutable, MeshExecutable]:
    if self.is_trivial:
      return MeshExecutable.from_trivial_jaxpr(**self.compile_args)
    else:
      return UnloadedMeshExecutable.from_hlo(
          self._name,
          self._hlo,
          **self.compile_args,
          _allow_propagation_to_outputs=_allow_propagation_to_outputs,
          _allow_compile_replicated=_allow_compile_replicated)  # type: ignore

  # -- stages.XlaLowering overrides

  def hlo(self) -> xc.XlaComputation:
    if self.is_trivial:
      raise ValueError("A trivial computation has no HLO")
    # this is a method for api consistency with dispatch.XlaComputation
    return xe.mlir.mlir_module_to_xla_computation(
        mlir.module_to_string(self._hlo),
        use_tuple_args=self.compile_args["tuple_args"])

  def mhlo(self) -> ir.Module:
    return super().mhlo()

  def stablehlo(self) -> ir.Module:
    if self.is_trivial:
      raise ValueError("A trivial computation has no StableHLO")
    return self._hlo

  def compile(self,
              _allow_propagation_to_outputs: Optional[Sequence[bool]] = None,
              _allow_compile_replicated: bool = True) -> MeshExecutable:
    if self._executable is None:
      executable = self._compile_unloaded(
          _allow_propagation_to_outputs, _allow_compile_replicated)
      if isinstance(executable, UnloadedMeshExecutable):
        executable = executable.load()
      self._executable = executable
    return self._executable

  def cost_analysis(self) -> Dict[str, float]:
    backend = self.compile_args["backend"]
    if xb.using_pjrt_c_api(backend):
      raise NotImplementedError(
          "Lowered.cost_analysis not implemented on platform "
          f"'{backend.platform}'. Use compile().cost_analysis() for "
          "post-compilation cost estimates.")
    return xe.hlo_module_cost_analysis(backend, self.hlo().as_hlo_module())

def get_input_metadata(
    global_in_avals: Sequence[ShapedArray],
    in_shardings: Sequence[sharding_internal.XLACompatibleSharding], in_is_global: Sequence[bool]
) -> Tuple[Sequence[sharding_internal.XLACompatibleSharding], Sequence[Tuple[Optional[Index], ...]],
           Sequence[ShapedArray]]:
  avals, shardings = _get_normalized_avals_and_shardings(
      global_in_avals, in_shardings, in_is_global)
  return shardings, _get_input_indices(avals, shardings), avals


def _get_normalized_avals_and_shardings(
    global_in_avals: Sequence[ShapedArray],
    in_shardings: Sequence[sharding_internal.XLACompatibleSharding], in_is_global: Sequence[bool]
) -> Tuple[Sequence[ShapedArray], Sequence[sharding_internal.XLACompatibleSharding]]:
  avals = []
  shardings = []

  for gaval, i, is_global in safe_zip(global_in_avals, in_shardings,
                                      in_is_global):
    if is_global:
      aval = gaval
      in_sharding = i
    else:
      assert isinstance(i, sharding_internal.NamedSharding)
      aval = i.mesh._global_to_local(
          cast(ArrayMapping, get_array_mapping(i.spec)), gaval)  # pylint: disable=g-bare-generic
      in_sharding = sharding_internal.NamedSharding(i.mesh.local_mesh, i.spec)
    avals.append(aval)
    shardings.append(in_sharding)

  return avals, shardings


def _get_input_indices(
    avals: Sequence[ShapedArray], shardings: Sequence[sharding_internal.XLACompatibleSharding]
) -> Sequence[Tuple[Optional[Index], ...]]:

  input_indices = []
  for aval, sharding in zip(avals, shardings):
    if aval is core.abstract_token:
      index = tuple(
          (slice(None),) for _ in range(len(sharding.addressable_devices)))
    else:
      # We special case this logic to support fully replicated values because
      # the mesh is global mesh and the indices returned by `spec_to_indices` will
      # represent index for each device in the global mesh. But here we want
      # indices for the local devices of the global mesh.
      proto = sharding._to_xla_op_sharding(aval.ndim)
      if is_op_sharding_replicated(proto):
        index = tuple(
            (slice(None),) * aval.ndim
            for _ in range(len(sharding.addressable_devices)))  # type: ignore
      else:
        index = tuple(
            sharding.addressable_devices_indices_map(
                aval.shape).values())  # type: ignore
    input_indices.append(index)

  return input_indices


def get_gspmd_shardings_from_executable(
    xla_executable, device_assignment: Sequence[xc.Device],
    num_in_avals: int, num_out_avals: int
) -> Tuple[Sequence[sharding_internal.XLACompatibleSharding],
           Sequence[sharding_internal.XLACompatibleSharding]]:
  from jax.experimental import pjit

  # When the device assignment only has 1 device, SPMD partitioner will not run.
  # Hence the op shardings will not be set on the `hlo_module`. In that case,
  # just return SingleDeviceShardings since we know the computation is running
  # only on 1 device.
  if len(device_assignment) == 1:
    return ([sharding_internal.SingleDeviceSharding(device_assignment[0])
             for _ in range(num_in_avals)],
            [sharding_internal.SingleDeviceSharding(device_assignment[0])
             for _ in range(num_out_avals)])

  in_op_shardings, out_op_shardings = pjit._get_op_sharding_from_executable(xla_executable)

  in_shardings_xla = [sharding_internal.GSPMDSharding(device_assignment, i)
                      for i in in_op_shardings]
  out_shardings_xla = [sharding_internal.GSPMDSharding(device_assignment, o)
                       for o in out_op_shardings]
  # This condition happens when all the elements in the output tuple have the
  # same sharding, so XLA decides to run the `FusionTupleDeduplicator` to
  # put the sharding on ROOT instead of the tuple.
  # TODO(b/245667823): Remove this when XLA fixes this.
  if len(out_shardings_xla) == 1 and len(out_shardings_xla) < num_out_avals:
    out_shardings_xla = out_shardings_xla * num_out_avals
  assert len(out_shardings_xla) == num_out_avals
  return in_shardings_xla, out_shardings_xla


# TODO(yashkatariya): Remove this function after `AUTO` can return shardings
# without mesh.
def _get_mesh_pspec_shardings_from_executable(
    xla_executable, mesh: Mesh
) -> Tuple[Sequence[sharding_internal.NamedSharding],
           Sequence[sharding_internal.NamedSharding]]:
  from jax.experimental import pjit

  in_pspec, out_pspec = pjit._get_pspec_from_executable(xla_executable, mesh)
  return ([sharding_internal.NamedSharding(mesh, i) for i in in_pspec],
          [sharding_internal.NamedSharding(mesh, o) for o in out_pspec])


@dataclasses.dataclass
class UnloadedMeshExecutable:
  xla_executable: Any
  device_assignment: Sequence[xc.Device]
  backend: xb.XlaBackend
  input_avals: Sequence[ShapedArray]
  input_shardings: Sequence[sharding_internal.XLACompatibleSharding]
  output_avals: Sequence[ShapedArray]
  output_shardings: Sequence[sharding_internal.XLACompatibleSharding]
  committed: bool
  are_out_shardings_from_xla: Sequence[bool]
  pmap_nreps: int
  name: str
  unordered_effects: List[core.Effect]
  ordered_effects: List[core.Effect]
  keepalive: Sequence[Any]
  host_callbacks: Sequence[Any]
  kept_var_idx: Set[int]
  auto_spmd_lowering: bool

  def load(self) -> MeshExecutable:
    input_indices = _get_input_indices(self.input_avals, self.input_shardings)
    handle_args = InputsHandler(self.xla_executable.local_devices(),
                                self.input_shardings, input_indices)
    handle_outs = global_avals_to_results_handler(
        self.output_avals, self.output_shardings, self.committed,
        self.are_out_shardings_from_xla)  # type: ignore  # arg-type

    # This path is taken for `jit(pmap)` cases. Nothing else should flow
    # through this path. This is exactly same to what happens in `jit`.
    if self.pmap_nreps > 1:
      has_unordered_effects = bool(self.unordered_effects)
      buffer_counts = dispatch.get_buffer_counts(
          self.output_avals, self.ordered_effects, has_unordered_effects)
      unsafe_call = partial(
          dispatch._execute_replicated, self.name, self.xla_executable, None,
          buffer_counts, handle_outs, has_unordered_effects, self.ordered_effects,
          self.kept_var_idx, bool(self.host_callbacks),
          from_lower_sharding_computation=True)
    else:
      unsafe_call = ExecuteReplicated(  # type: ignore  # assignment
          self.xla_executable, self.name, self.backend, handle_args,
          handle_outs, self.unordered_effects, self.ordered_effects, self.keepalive,
          bool(self.host_callbacks), self.kept_var_idx)

    return MeshExecutable(self.xla_executable, unsafe_call, self.input_avals,
                          self.input_shardings, self.output_shardings,
                          self.auto_spmd_lowering, self.kept_var_idx,
                          self.device_assignment)

  # May return a MeshExecutable in the compile_replicated case.
  @staticmethod
  def from_hlo(name: str,
               computation: Union[ir.Module, xc.XlaComputation],
               # TODO(yashkatariya): Remove `mesh` from here once AUTO can work
               # without mesh.
               mesh: Optional[Mesh],
               global_in_avals: Sequence[ShapedArray],
               global_out_avals: Sequence[ShapedArray],
               in_shardings: Sequence[Union[sharding_internal.XLACompatibleSharding, AUTOAxisResource]],
               out_shardings: Sequence[Union[sharding_internal.XLACompatibleSharding, AUTOAxisResource,
                                       UnspecifiedValue]],
               spmd_lowering: bool,
               tuple_args: bool,
               in_is_global: Sequence[bool],
               auto_spmd_lowering: bool,
               _allow_propagation_to_outputs: Optional[Sequence[bool]],
               _allow_compile_replicated: bool,
               unordered_effects: List[core.Effect],
               ordered_effects: List[core.Effect],
               host_callbacks: List[Any],
               keepalive: Any,
               kept_var_idx: Set[int],
               backend: xb.XlaBackend,
               device_assignment: Sequence[xc.Device],
               committed: bool,
               pmap_nreps: int = 1
  ) -> Union[MeshExecutable, UnloadedMeshExecutable]:

    dev: np.ndarray
    if auto_spmd_lowering:
      assert mesh is not None and spmd_lowering
      dev = mesh.devices
      num_replicas, num_partitions = 1, mesh.size
    else:
      dev = np.array(device_assignment)
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

    compile_options = xb.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=xla_device_assignment,
        use_spmd_partitioning=spmd_lowering,
        use_auto_spmd_partitioning=auto_spmd_lowering,
    )
    if auto_spmd_lowering:
      assert mesh is not None
      compile_options.executable_build_options.auto_spmd_partitioning_mesh_shape = \
          list(mesh.shape.values())
      compile_options.executable_build_options.auto_spmd_partitioning_mesh_ids = \
          _get_logical_mesh_ids(list(mesh.shape.values())).reshape(-1)
    compile_options.parameter_is_tupled_arguments = tuple_args

    if _allow_propagation_to_outputs is None:
      _allow_propagation_to_outputs = [False] * len(out_shardings)
    if xla_extension_version >= 123:
      # Convert bools to int because that is what the compilation option expects.
      compile_options.executable_build_options.allow_spmd_sharding_propagation_to_output = \
          _allow_propagation_to_outputs
    else:
      compile_options.executable_build_options.allow_spmd_sharding_propagation_to_output = \
          all(_allow_propagation_to_outputs)

    if _allow_compile_replicated and hasattr(backend, "compile_replicated"):
      return _compile_replicated_mesh_executable_from_hlo(
          name, computation, global_in_avals, global_out_avals, in_shardings,
          out_shardings, in_is_global, auto_spmd_lowering, compile_options,
          host_callbacks, bool(unordered_effects), ordered_effects,
          kept_var_idx, backend, device_assignment, committed, pmap_nreps)
    else:
      with dispatch.log_elapsed_time(f"Finished XLA compilation of {name} "
                                     "in {elapsed_time} sec",
                                     event=dispatch.BACKEND_COMPILE_EVENT):
        xla_executable = dispatch.compile_or_get_cached(
            backend, computation, compile_options, host_callbacks)

      if auto_spmd_lowering:
        assert mesh is not None
        in_shardings_xla, out_shardings_xla = _get_mesh_pspec_shardings_from_executable(
            xla_executable, mesh)
        in_shardings = [x if is_auto(i) else i
                        for x, i in safe_zip(in_shardings_xla, in_shardings)]
        out_shardings_tuple = [
            (x, True) if is_auto(o) else (o, False)
            for x, o in safe_zip(out_shardings_xla, out_shardings)
        ]
        out_shardings, are_out_shardings_from_xla = unzip2(out_shardings_tuple)
      elif out_shardings and any(_is_unspecified(o) for o in out_shardings):
        assert mesh is None
        _, out_shardings_xla = get_gspmd_shardings_from_executable(  # type: ignore
            xla_executable, device_assignment,
            len(global_in_avals), len(global_out_avals))
        orig_out_shardings = out_shardings
        out_shardings, are_out_shardings_from_xla = [], []  # type: ignore
        for xla_s, orig, aval in safe_zip(out_shardings_xla, orig_out_shardings,
                                          global_out_avals):
          if _is_unspecified(orig):
            out_shardings.append(xla_s)
            are_out_shardings_from_xla.append(True)
          else:
            if xla_extension_version >= 123 and not are_op_shardings_equal(
                xla_s._to_xla_op_sharding(aval.ndim),  # type: ignore
                orig._to_xla_op_sharding(aval.ndim)):  # type: ignore
              raise AssertionError(
                  f"Unexpected XLA sharding override: (XLA) {xla_s} != {orig} "
                  "(User sharding)")
            out_shardings.append(orig)
            are_out_shardings_from_xla.append(False)
      else:
        are_out_shardings_from_xla = (False,) * len(global_out_avals)

      input_avals, input_shardings = (
          _get_normalized_avals_and_shardings(
              global_in_avals, in_shardings, in_is_global))  # type: ignore # arg-type

      return UnloadedMeshExecutable(
          xla_executable=xla_executable,
          device_assignment=device_assignment,
          backend=backend,
          input_avals=input_avals,
          input_shardings=input_shardings,
          output_avals=global_out_avals,
          output_shardings=out_shardings,  # type: ignore # arg-type
          committed=committed,
          are_out_shardings_from_xla=are_out_shardings_from_xla,
          pmap_nreps=pmap_nreps,
          name=name,
          unordered_effects=unordered_effects,
          ordered_effects=ordered_effects,
          keepalive=keepalive,
          host_callbacks=host_callbacks,
          kept_var_idx=kept_var_idx,
          auto_spmd_lowering=auto_spmd_lowering)


class MeshExecutableFastpathData(NamedTuple):
  xla_executable: xc.LoadedExecutable
  out_pytree_def: Any
  in_shardings: Sequence[sharding_internal.XLACompatibleSharding]
  out_shardings: Sequence[sharding_internal.XLACompatibleSharding]
  out_avals: Sequence[ShapedArray]
  out_committed: Sequence[bool]
  kept_var_bitvec: Iterable[bool]


class MeshExecutable(stages.XlaExecutable):
  __slots__ = [
      "xla_executable", "unsafe_call", "in_avals", "_in_shardings",
      "_out_shardings", "_auto_spmd_lowering", "_kept_var_idx",
      "_device_assignment"
  ]

  def __init__(self, xla_executable, unsafe_call, in_avals, in_shardings,
               out_shardings, auto_spmd_lowering, kept_var_idx,
               device_assignment):
    self.xla_executable = xla_executable
    self.unsafe_call = unsafe_call
    # in_avals is a list of global and local avals. Aval is global if input
    # is a GDA or jax.Array else local.
    self.in_avals = in_avals
    self._in_shardings = in_shardings
    self._out_shardings = out_shardings
    self._auto_spmd_lowering = auto_spmd_lowering
    self._kept_var_idx = kept_var_idx
    self._device_assignment = device_assignment

  @staticmethod
  def from_trivial_jaxpr(jaxpr, consts, global_in_avals, global_out_avals,
                         in_shardings, backend, device_assignment,
                         committed, kept_var_idx, keepalive) -> MeshExecutable:
    assert keepalive is None
    if hasattr(backend, "compile_replicated"):
      return _compile_replicated_mesh_executable_from_trivial_jaxpr(
          jaxpr, consts, global_in_avals, global_out_avals, in_shardings,
          backend, device_assignment, committed, kept_var_idx)

    out_shardings = _out_shardings_for_trivial(
        jaxpr, consts, in_shardings, device_assignment)
    if config.jax_array or config.jax_parallel_functions_output_gda:
      are_global = [True] * len(global_out_avals)
    else:
      are_global = [False] * len(global_out_avals)
    _, indices, _ = get_input_metadata(global_out_avals, out_shardings,
                                        are_global)
    local_device_assignment = [d for d in device_assignment
                               if d.process_index == d.client.process_index()]
    handle_ins = InputsHandler(local_device_assignment, out_shardings, indices)
    handle_outs = global_avals_to_results_handler(
          global_out_avals, out_shardings, committed,
          [False] * len(global_out_avals))
    unsafe_call = partial(_execute_trivial, jaxpr, consts, handle_ins,
                          handle_outs, kept_var_idx)
    return MeshExecutable(None, unsafe_call, global_in_avals, in_shardings,
                          out_shardings, False, kept_var_idx, device_assignment)

  # -- stages.XlaExecutable overrides

  def xla_extension_executable(self):
    return self.xla_executable

  def call(self, *args):
    kept_args = [a for i, a in enumerate(args) if i in self._kept_var_idx]
    arg_avals = map(xla.abstractify, kept_args)
    ref_avals = self.in_avals
    dispatch.check_arg_avals_for_call(ref_avals, arg_avals)
    # Check the GDA sharding and the input sharding.
    check_gda_or_array_xla_sharding_match(kept_args, self._in_shardings)
    return self.unsafe_call(*args)

  def input_shardings(self) -> Sequence[sharding_internal.XLACompatibleSharding]:
    return self._in_shardings

  def output_shardings(self) -> Sequence[sharding_internal.XLACompatibleSharding]:
    return self._out_shardings

  def create_cpp_call(self, no_kwargs, in_tree, out_tree):
    if not (isinstance(self.unsafe_call, ExecuteReplicated) and
            not self.unsafe_call.has_unordered_effects and
            not self.unsafe_call.has_host_callbacks):
      return None

    if not flags.FLAGS.experimental_cpp_pjit:
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

    if xla_extension_version < 124:
      return xc._xla.pjit(self.unsafe_call.name, None, aot_cache_miss, [], [])  # type: ignore
    else:
      return xc._xla.pjit(self.unsafe_call.name, None, aot_cache_miss, [], [], [])  # type: ignore


def _out_shardings_for_trivial(
    jaxpr: core.Jaxpr, consts: Sequence[Any],
    in_shardings: Sequence[sharding_internal.XLACompatibleSharding],
    device_assignment: Sequence[xc.Device],
  ) -> List[sharding_internal.XLACompatibleSharding]:
  # For each jaxpr output, compute a Sharding by:
  #   * if the output is a forwarded input, get the corresponding in_sharding;
  #   * if the output is a constant Array, get its .sharding attribute;
  #   * otherwise, the output is a literal or numpy.ndarray constant, so give it
  #     a replicated sharding
  from jax._src import array

  rep = sharding_internal.GSPMDSharding(
      device_assignment, sharding_internal._get_replicated_op_sharding())
  shardings: Dict[core.Var, sharding_internal.XLACompatibleSharding] = {}
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


def _compile_replicated_pmap_executable_from_hlo(
    xla_computation, pci, input_indices, in_shardings, handle_outs,
    compile_options, host_callbacks, has_unordered_effects, ordered_effects):
  # Use the standard out_handler.
  execute_fun = pci.backend.compile_replicated(
      is_trivial=False, name=pci.name, computation=xla_computation,
      compile_options=compile_options, host_callbacks=host_callbacks,
      has_unordered_effects=has_unordered_effects,
      ordered_effects=ordered_effects, in_avals=pci.avals,
      in_indices=input_indices, in_shardings=in_shardings,
      kept_var_idx=set(range(len(pci.avals))), out_handler=handle_outs)
  # TODO(frostig): need `compile_replicated` to give us the XLA executable
  return PmapExecutable(None, execute_fun, None, pci.avals)


def _compile_replicated_mesh_executable_from_hlo(
    name, computation, global_in_avals, global_out_avals, in_shardings,
    out_shardings, in_is_global, auto_spmd_lowering, compile_options,
    host_callbacks, has_unordered_effects, ordered_effects, kept_var_idx,
    backend, device_assignment, committed, pmap_nreps):
  assert not auto_spmd_lowering
  in_shardings, input_indices, input_avals = get_input_metadata(
      global_in_avals, in_shardings, in_is_global)  # type: ignore
  if pmap_nreps > 1:
    # For a jit wrapping a pmap, replicate each input index to match the
    # devices of the replicated jit computation.
    input_indices = [index * pmap_nreps for index in input_indices]

  # Will compute out_handler with executable information.
  unsafe_call = backend.compile_replicated(
      is_trivial=False, name=name, computation=computation,
      compile_options=compile_options, host_callbacks=host_callbacks,
      has_unordered_effects=has_unordered_effects,
      ordered_effects=ordered_effects, in_avals=input_avals,
      in_indices=input_indices, in_shardings=in_shardings,
      kept_var_idx=kept_var_idx,
      out_avals=global_out_avals, out_shardings=out_shardings,
      committed=committed)
  xla_executable = None
  return MeshExecutable(xla_executable, unsafe_call, input_avals,
                        in_shardings, out_shardings, auto_spmd_lowering,
                        kept_var_idx, device_assignment)


def _compile_replicated_mesh_executable_from_trivial_jaxpr(
    jaxpr, consts, global_in_avals, global_out_avals, in_shardings, backend,
    device_assignment, committed, kept_var_idx):
  out_shardings = _out_shardings_for_trivial(
      jaxpr, consts, in_shardings, device_assignment)

  if config.jax_array or config.jax_parallel_functions_output_gda:
    in_is_global = [True] * len(global_in_avals)
  else:
    in_is_global = [False] * len(global_in_avals)
  in_shardings, input_indices, input_avals = get_input_metadata(
      global_in_avals, in_shardings, in_is_global)  # type: ignore
  handle_outs = global_avals_to_results_handler(
      global_out_avals, out_shardings, committed,
      [False] * len(global_out_avals))
  # Use the standard out_handler.
  unsafe_call = backend.compile_replicated(
      is_trivial=True, jaxpr=jaxpr, consts=consts,
      device_assignment=device_assignment, in_avals=input_avals,
      in_indices=input_indices, in_shardings=in_shardings,
      kept_var_idx=kept_var_idx, out_handler=handle_outs,
      out_shardings=out_shardings)
  return MeshExecutable(None, unsafe_call, global_in_avals, in_shardings,
                        out_shardings, False, kept_var_idx,
                        device_assignment)


@lru_cache()
def create_mesh_pspec_sharding(
    mesh: Mesh, pspec: PartitionSpec, parsed_pspec=None
) -> sharding_internal.NamedSharding:
  return sharding_internal.NamedSharding(mesh, pspec, parsed_pspec)


def check_device_backend_on_shardings(shardings) -> bool:
  for i in shardings:
    if _is_unspecified(i) or is_auto(i):
      continue
    if hasattr(i, '_original_sharding') and getattr(
        i._original_sharding, '_device_backend', False):
      return True
  return False


def check_gda_or_array_xla_sharding_match(
    args, in_xla_shardings: Sequence[sharding_internal.XLACompatibleSharding]) -> None:
  from jax._src.global_device_array import GlobalDeviceArray
  from jax._src.array import ArrayImpl

  for arg, xs in safe_zip(args, in_xla_shardings):
    if not isinstance(arg, (GlobalDeviceArray, ArrayImpl)):
      continue
    if isinstance(arg, GlobalDeviceArray):
      arg_sharding = create_mesh_pspec_sharding(arg.mesh, arg.mesh_axes)
      arg_type = 'GDA'
      committed = True
    else:
      arg_sharding = arg.sharding
      arg_type = 'Array'
      committed = arg._committed

    # No need to cache this check since MeshExecutable has a C++ fast path
    # for AOT compiled call.
    if (not check_device_backend_on_shardings([xs]) and
        committed and
        not are_op_shardings_equal(arg_sharding._to_xla_op_sharding(arg.ndim),
                                   xs._to_xla_op_sharding(arg.ndim))):
      raise ValueError(
          f"{arg_type} sharding does not match the input sharding. "
          f"Got {arg_type} sharding: {arg_sharding} and xla sharding: {xs} for "
          f"arg shape: {arg.shape}, arg value: {arg}")


def get_array_mapping(pspec: PartitionSpec) -> ArrayMappingOrAutoOrUnspecified:
  # Import here to avoid cyclic import error when importing gda in pjit.py.
  from jax.experimental.pjit import get_array_mapping as _get_array_mapping, _prepare_axis_resources

  parsed_pspec, _, _ = _prepare_axis_resources(pspec, "pspec to array_mapping")
  return _get_array_mapping(parsed_pspec)


def are_op_shardings_equal(op1: xc.OpSharding, op2: xc.OpSharding) -> bool:
  if id(op1) == id(op2):
    return True
  if is_op_sharding_replicated(op1) and is_op_sharding_replicated(op2):
    return True
  return xc.HloSharding.from_proto(op1) == xc.HloSharding.from_proto(op2)


def is_op_sharding_replicated(op: xc.OpSharding) -> bool:
  if len(op.tile_assignment_devices) == 1:
    return True
  return xc.HloSharding.from_proto(op).is_replicated()  # type: ignore


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


def _make_sharding_spec(axis_sizes, mesh_axis_pos, num_dimensions, aval_axes):
  mesh_mapping = [Replicated(axis_size) for axis_size in axis_sizes.values()]
  sharding = [_UNSHARDED_INSTANCE] * num_dimensions
  next_sharded_axis = 0
  # NOTE: sorted is stable, which is important when multiple resources
  #       map to the same axis.
  for name, axis in sorted(aval_axes.items(), key=lambda x: x[1]):
    chunked = sharding[axis]
    if isinstance(chunked, NoSharding):
      chunked = Chunked([])
    sharding[axis] = Chunked(list(chunked.chunks) + [axis_sizes[name]])
    assert isinstance(mesh_mapping[mesh_axis_pos[name]], Replicated), \
        "Value mapped to the same mesh axis twice"
    mesh_mapping[mesh_axis_pos[name]] = ShardedAxis(next_sharded_axis)
    next_sharded_axis += 1
  return ShardingSpec(sharding, mesh_mapping)


def new_mesh_sharding_specs(axis_sizes, axis_names):
  mesh_axis_pos = {name: i for i, name in enumerate(axis_names)}
  return partial(_make_sharding_spec, axis_sizes, mesh_axis_pos)


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
    return _make_sharding_spec(axis_sizes, mesh_axis_pos, len(aval.shape), aval_axes)
  return mk_sharding_spec


@contextmanager
def maybe_extend_axis_env(*args, **kwargs):
  with core.extend_axis_env(*args, **kwargs):
    yield

class DynamicAxisEnvFrame:
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
      raise NameError(f"unbound axis name: {axis_name}")
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
