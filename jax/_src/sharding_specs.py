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

import collections
import functools
import itertools
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

from jax._src import op_shardings
from jax._src import util
from jax._src.lib import pmap_lib
from jax._src.lib import xla_client as xc

unsafe_map, map = map, util.safe_map

NoSharding = pmap_lib.NoSharding
Chunked = pmap_lib.Chunked
Unstacked = pmap_lib.Unstacked

_UNSHARDED_INSTANCE = NoSharding()

ShardedAxis = pmap_lib.ShardedAxis
Replicated = pmap_lib.Replicated
MeshDimAssignment = Union[ShardedAxis, Replicated]

ShardingSpec = pmap_lib.ShardingSpec

OpShardingType = Any


def _sharding_spec_mesh_shape(self):
  sharded_axis_sizes = []
  for sharding in self.sharding:
    if isinstance(sharding, NoSharding):
      continue
    elif isinstance(sharding, Unstacked):
      sharded_axis_sizes.append(sharding.size)
    elif isinstance(sharding, Chunked):
      sharded_axis_sizes.extend(sharding.chunks)
    else:
      util.assert_unreachable(sharding)
  return tuple(sharded_axis_sizes[a.axis] if isinstance(a, ShardedAxis)
               else a.replicas
               for a in self.mesh_mapping)


def get_logical_mesh_ids(mesh_shape):
  return np.arange(math.prod(mesh_shape)).reshape(mesh_shape)


_MeshAxisName = Any

def sharding_spec_sharding_proto(
    self, special_axes: Mapping[int, OpShardingType] = {}):
  """Converts a ShardingSpec to an OpSharding proto.

  See
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto#L601
  for details on the OpSharding proto.
  Unfortunately the semantics are not very well described in the proto spec, but
  the code here might help:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/compiler/xla/experimental/xla_sharding/xla_sharding.py
  """
  mesh_shape = cast(Tuple[int, ...], self.mesh_shape)
  mesh = get_logical_mesh_ids(self.mesh_shape)

  sharded_axes = {}  # maps sharded axis identifiers to mesh axis indices to which they're mapped
  replicated_maxes = []  # lists mesh axis identifiers to replicate over
  for maxis, assignment in enumerate(self.mesh_mapping):
    if isinstance(assignment, Replicated):
      replicated_maxes.append((maxis, assignment.replicas))
    elif isinstance(assignment, ShardedAxis):
      sharded_axes[assignment.axis] = maxis
    else:
      util.assert_unreachable(assignment)

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
      new_mesh_shape.append(math.prod(sharding.chunks))
    elif isinstance(sharding, Unstacked):
      raise RuntimeError("Cannot convert unstacked sharding specs to XLA OpSharding")
    else:
      util.assert_unreachable(sharding)

  # Create a partial sharding proto if tensor is replicated or partitioned
  # specially over some mesh axes.
  if replicated_maxes:
    last_tile_dims = []
    axes_by_type: Dict[OpShardingType, List[_MeshAxisName]] = {}
    size_by_type: Dict[OpShardingType, int] = collections.defaultdict(lambda: 1)
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

  proto.tile_assignment_dimensions = new_mesh_shape
  num_devices = math.prod(new_mesh_shape)
  if num_devices == 1:
    proto.tile_assignment_devices = [0]
  else:
    proto.iota_reshape_dims = list(mesh_shape)
    proto.iota_transpose_perm = mesh_permutation
    assert num_devices == math.prod(proto.iota_reshape_dims)
  return proto


def _sharding_spec_indices(self, shape: Tuple[int, ...]) -> np.ndarray:
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
    return op_shardings.op_sharding_to_numpy_indices(
        op_sharding_proto, shape, math.prod(self.mesh_shape)
    ).reshape(self.mesh_shape)

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
      total_chunks = math.prod(sharding.chunks)
      shard_size, ragged = divmod(axis_size, total_chunks)
      assert not ragged, (axis_size, total_chunks, dim)
      axis_indices.append([slice(i * shard_size, (i + 1) * shard_size)
                           for i in range(total_chunks)])
      shard_indices_shape.extend(sharding.chunks)
    else:
      util.assert_unreachable(sharding)

  # shard_indices is an ndarray representing the sharded axes of the logical array,
  # with each dimension having size equal to the number of shards across the corresponding
  # logical array dimension, and each element containing the multi-dimensional index that
  # is used to extract the corresponding shard of the logical array.
  shard_indices = np.empty([math.prod(shard_indices_shape)], dtype=np.object_)
  for i, idxs in enumerate(itertools.product(*axis_indices)):
    shard_indices[i] = idxs
  shard_indices = shard_indices.reshape(shard_indices_shape)

  # Ensure that each sharded axis is used exactly once in the mesh mapping
  num_sharded_dim = len(shard_indices_shape)
  sharded_dim_perm = [a.axis for a in self.mesh_mapping if isinstance(a, ShardedAxis)]
  assert (set(sharded_dim_perm) == set(range(num_sharded_dim)) and
          len(sharded_dim_perm) == num_sharded_dim)
  # Replicate/reorder the indices according to the mesh mapping
  replica_sizes = tuple(a.replicas for a in self.mesh_mapping if isinstance(a, Replicated))
  replica_dim, sharded_dim = itertools.count(0), iter(sharded_dim_perm)
  perm = [next(replica_dim) if isinstance(a, Replicated) else
          len(replica_sizes) + next(sharded_dim)
          for a in self.mesh_mapping]
  return (np.broadcast_to(shard_indices, replica_sizes + shard_indices.shape)
            .transpose(perm))

def _sharding_spec_repr(self):
  return f'ShardingSpec({self.sharding}, {self.mesh_mapping})'


ShardingSpec.mesh_shape = property(_sharding_spec_mesh_shape)
ShardingSpec.sharding_proto = sharding_spec_sharding_proto
ShardingSpec.indices = _sharding_spec_indices
# mypy raises: error: Cannot assign to a method  [assignment]
ShardingSpec.__repr__ = _sharding_spec_repr  # type: ignore


Index = Union[int, slice, Tuple[Union[int, slice], ...]]

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


def make_sharding_spec(axis_sizes, mesh_axis_pos, num_dimensions, aval_axes):
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
  return functools.partial(make_sharding_spec, axis_sizes, mesh_axis_pos)

def pmap_sharding_spec(nrep, axis_size, sharded_shape: Sequence[int],
                       map_axis: Optional[int]) -> ShardingSpec:
  """Sharding spec for arguments or results of a pmap.
  Args:
    nrep: number of local XLA replicas (product of local axis sizes)
    axis_size: local axis size for outer pmap
    sharded_aval: the aval of the value inside the outer pmap, an instance of
      a ShapedArray.
    map_axis: the axis along which the value is mapped in the outer pmap
  Returns:
    A ShardingSpec.
  """
  replication_factor, ragged = divmod(nrep, axis_size)
  assert not ragged
  pspec = ShardingSpec(sharding=[_UNSHARDED_INSTANCE] * len(sharded_shape),
                       mesh_mapping=())
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
      sharding=util.tuple_insert(
          pspec.sharding, map_axis, Unstacked(axis_size)),
      mesh_mapping=itertools.chain(
          [ShardedAxis(sharded_in_axis)], maybe_replicate,
          map(shift_sharded_axis, pspec.mesh_mapping)))
  else:
    return ShardingSpec(
      sharding=pspec.sharding,
      mesh_mapping=(Replicated(axis_size),) + maybe_replicate + pspec.mesh_mapping)


def create_pmap_sharding_spec(shape: Tuple[int, ...], sharded_dim: int = 0,
                              sharded_dim_size: Optional[int] = None):
  if sharded_dim is not None:
    sharded_shape = shape[:sharded_dim] + shape[sharded_dim+1:]
    if sharded_dim_size is None:
      sharded_dim_size = shape[sharded_dim]
  else:
    assert sharded_dim_size is not None
    sharded_shape = shape

  return pmap_sharding_spec(sharded_dim_size, sharded_dim_size, sharded_shape,
                            sharded_dim)
