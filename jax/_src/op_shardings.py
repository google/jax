# Copyright 2023 The JAX Authors.
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
"""Sharding utilities"""

import itertools
from typing import List, Sequence, Tuple, Union

import math
import numpy as np

from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version


def get_num_ways_dim_sharded(
    op_sharding: xc.OpSharding) -> Tuple[Sequence[int], int]:
  partitions = op_sharding.tile_assignment_dimensions
  if op_sharding.last_tile_dims == [xc.OpSharding.Type.REPLICATED]:
    replicate_on_last_tile_dim = True
  else:
    replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim
    if op_sharding.last_tile_dims:
      raise NotImplementedError(
          "Unhandled OpSharding type. Please open a bug report!")
  num_replicas = 1
  if replicate_on_last_tile_dim:
    num_replicas = partitions[-1]
    partitions = partitions[:-1]
  return partitions, num_replicas


if xla_extension_version >= 147:
  is_op_sharding_replicated = xc._xla.is_op_sharding_fully_replicated
else:
  def is_op_sharding_replicated(op: xc.OpSharding) -> bool:
    if len(op.tile_assignment_devices) == 1:
      return True
    return xc.HloSharding.from_proto(op).is_replicated()  # type: ignore

def are_op_shardings_equal(op1: xc.OpSharding, op2: xc.OpSharding) -> bool:
  if id(op1) == id(op2):
    return True
  if is_op_sharding_replicated(op1) and is_op_sharding_replicated(op2):
    return True
  return xc.HloSharding.from_proto(op1) == xc.HloSharding.from_proto(op2)

_Index = Union[int, slice, Tuple[Union[int, slice], ...]]


def op_sharding_to_numpy_indices(
    op_sharding: xc.OpSharding, shape: Sequence[int],
    num_devices: int) -> np.ndarray:
  indices = np.empty(num_devices, dtype=np.object_)

  # num_devices is required as an argument when op_sharding is
  # REPLICATED. `jax.device_count()` cannot be used because you can create
  # an opsharding with less number of devices than `jax.device_count()`.
  if is_op_sharding_replicated(op_sharding):
    indices.fill((slice(None),) * len(shape))
    return indices

  if op_sharding.tile_assignment_devices:
    assert num_devices == len(op_sharding.tile_assignment_devices)
  else:
    assert num_devices == math.prod(op_sharding.iota_reshape_dims)

  partitions, num_replicas = get_num_ways_dim_sharded(op_sharding)
  assert len(partitions) == len(shape), (len(partitions), len(shape))

  axis_indices: List[Sequence[_Index]] = []
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

  if op_sharding.tile_assignment_devices:
    device_it = iter(op_sharding.tile_assignment_devices)
  else:
    tad = (
        np.arange(num_devices)
        .reshape(op_sharding.iota_reshape_dims)
        .transpose(op_sharding.iota_transpose_perm)
        .flat
    )
    device_it = iter(tad)
  for i, idxs in enumerate(itertools.product(*axis_indices)):
    for _ in range(num_replicas):
      indices[next(device_it)] = idxs
  return indices


def op_sharding_to_indices(op_sharding: xc.OpSharding, shape: Sequence[int],
                           num_devices: int) -> Tuple[Tuple[slice, ...], ...]:
  indices = op_sharding_to_numpy_indices(op_sharding, shape, num_devices)
  return tuple(indices.flat)
