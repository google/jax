# Copyright 2021 Google LLC
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

import abc
import functools
from collections import Counter
from typing import Sequence, Tuple, Optional, Mapping, Dict, Set, Union

from jax._src.util import safe_zip
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax.interpreters import pxla, mlir

import numpy as np

Shape = Tuple[int, ...]
Device = xc.Device
Index = Tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]


class Sharding(metaclass=abc.ABCMeta):

  @abc.abstractproperty
  def device_set(self) -> Set[Device]:
    """A unique set of devices that this sharding represents.

    Devices can be non-addressable too.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @pxla.maybe_cached_property
  def addressable_devices(self) -> Set[Device]:
    """A set of addressable devices by the current process"""
    process_index = xb.process_index()
    return {d for d in self.device_set if d.process_index == process_index}

  def is_fully_addressable(self) -> bool:
    # The pytype disable is because pytype can't recognize a cached property.
    return len(self.device_set) == len(self.addressable_devices)  # type: ignore

  def device_indices(self, device: Device,
                     global_shape: Shape) -> Optional[Index]:
    return self.devices_indices_map(global_shape)[device]

  @abc.abstractmethod
  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    raise NotImplementedError('Subclasses should implement this method.')


class XLACompatibleSharding(Sharding):

  @abc.abstractproperty
  def _device_assignment(self) -> XLADeviceAssignment:
    raise NotImplementedError('Subclasses should implement this method.')

  @abc.abstractmethod
  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    raise NotImplementedError('Subclasses should implement this method.')

  @pxla.maybe_cached_property
  def _addressable_device_assignment(self) -> XLADeviceAssignment:
    process_index = xb.process_index()
    return [d for d in self._device_assignment if d.process_index == process_index]

  @abc.abstractmethod
  def _to_xla_op_sharding(self, num_dimensions: int) -> Optional[xc.OpSharding]:
    raise NotImplementedError('Subclasses should implement this method.')


@functools.lru_cache()
def _check_mesh_resource_axis(mesh, parsed_pspec):
  try:
    [mesh.shape[r] for p in parsed_pspec if p is not None
     for r in p]
  except KeyError as e:
    raise ValueError(f"Resource axis: {e.args[0]} of {parsed_pspec.user_spec} is "
                     "undefined.") from None


def _hashed_index(x) -> int:
  # This works for both `pjit`/`xmap` indices and `pmap` indices (which might
  # have an integer instead of a slice).
  assert all(v.step is None for v in x if isinstance(v, slice))
  return hash(tuple((v.start, v.stop) if isinstance(v, slice) else v for v in x))


@functools.lru_cache(maxsize=4096)
def _device_replica_id_map(sharding, global_shape: Shape) -> Mapping[Device, int]:
  index_to_replica: Dict[int, int] = Counter()
  out = {}
  for device, index in sharding.devices_indices_map(global_shape).items():
    h_index = _hashed_index(index)
    replica_id = index_to_replica[h_index]
    index_to_replica[h_index] += 1
    out[device] = replica_id
  return out


class MeshPspecSharding(XLACompatibleSharding):

  def __init__(
      self, mesh: pxla.Mesh, spec: pxla.PartitionSpec, _parsed_pspec = None):

    self.mesh = mesh
    self.spec = spec

    # This split exists because you can pass `_parsed_pspec` that has been
    # modified from the original. For example: Adding extra dimension to
    # axis_resources for vmap handlers. In such cases you need to preserve the
    # `sync` attribute of parsed pspecs.
    # PartitionSpec is inferred from the parsed pspec in this case.
    # TODO(yaskatariya): Remove this and replace this with a normalized
    # representation of Parsed Pspec
    if _parsed_pspec is None:
      from jax.experimental import pjit
      self._parsed_pspec, _, _, _ = pjit._prepare_axis_resources(
          self.spec, "MeshPspecSharding spec")
    else:
      self._parsed_pspec = _parsed_pspec

    _check_mesh_resource_axis(self.mesh, self._parsed_pspec)

  def __repr__(self):
    return f'MeshPspecSharding(mesh={dict(self.mesh.shape)}, partition_spec={self.spec})'

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self.mesh, self._parsed_pspec))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, MeshPspecSharding):
      return False

    if id(self.mesh) == id(other.mesh) and self._parsed_pspec == other._parsed_pspec:
      return True
    return self.mesh == other.mesh and self._parsed_pspec == other._parsed_pspec

  def is_compatible_aval(self, aval_shape: Shape):
    if len(aval_shape) < len(self._parsed_pspec):
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(self._parsed_pspec)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @classmethod
  def _from_parsed_pspec(cls, mesh, parsed_pspec):
    from jax.experimental import pjit
    return cls(mesh, pjit._get_single_pspec(parsed_pspec), parsed_pspec)

  @pxla.maybe_cached_property
  def device_set(self) -> Set[Device]:
    return set(self.mesh.devices.flat)

  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Index]:
    # TODO(yashkatariya): Remove this when utilities are moved to pxla.py.
    from jax.experimental import global_device_array

    # `get_shard_indices` is cached.
    return global_device_array.get_shard_indices(global_shape, self.mesh, self.spec)

  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    return _device_replica_id_map(self, global_shape)

  @pxla.maybe_cached_property
  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self.mesh.devices.flat)

  @functools.lru_cache(maxsize=4096)
  def _to_xla_op_sharding(
      self,
      num_dimensions: int,
      axis_ctx: Optional[Union[mlir.SPMDAxisContext, mlir.ShardingContext]] = None
  ) -> xc.OpSharding:
    from jax.experimental.pjit import get_array_mapping

    array_mapping = get_array_mapping(self._parsed_pspec)
    # TODO(yashkatariya): Move away from sharding spec in MeshPspecSharding
    # since we don't really need sharding spec.
    sharding_spec = pxla.new_mesh_sharding_specs(
        self.mesh.shape, self.mesh.axis_names)(num_dimensions, array_mapping)
    # Used in `with_sharding_constraint`.
    special_axes = {}
    # Manual axes is only used with xmap.
    if axis_ctx is not None and isinstance(axis_ctx, mlir.SPMDAxisContext):
      axis_names = self.mesh.axis_names
      # Ignore type because mypy doesn't recognize the `hasattr` check above.
      for manual_axis in axis_ctx.manual_axes:  # type: ignore
        special_axes[axis_names.index(manual_axis)] = xc.OpSharding.Type.MANUAL
    return sharding_spec.sharding_proto(special_axes=special_axes)


@functools.lru_cache()
def _get_replicated_op_sharding():
  proto = xc.OpSharding()
  proto.type = xc.OpSharding.Type.REPLICATED
  return proto


class SingleDeviceSharding(XLACompatibleSharding):

  def __init__(self, device: Device):
    self._device = device

  def __repr__(self):
    return f"SingleDeviceSharding(device={repr(self._device)})"

  def __hash__(self):
    return hash(self._device)

  def __eq__(self, other):
    if not isinstance(other, SingleDeviceSharding):
      return False
    return self._device == other._device

  @property
  def device_set(self) -> Set[Device]:
    return {self._device}

  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Index]:
    return {self._device: (slice(None),) * len(global_shape)}

  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    return {self._device: 0}

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return [self._device]

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    return _get_replicated_op_sharding()


class PmapSharding(XLACompatibleSharding):

  def __init__(self, devices: np.ndarray, sharding_spec: pxla.ShardingSpec):
    self.devices = devices
    # The sharding spec should be pmap's sharding spec.
    self.sharding_spec = sharding_spec

  @pxla.maybe_cached_property
  def device_set(self) -> Set[Device]:
    return set(self.devices.flat)

  @functools.lru_cache(maxsize=4096)
  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    indices = pxla.spec_to_indices(global_shape, self.sharding_spec)
    return {d: i for d, i in safe_zip(self.devices.flat, indices)}

  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    return _device_replica_id_map(self, global_shape)

  @pxla.maybe_cached_property
  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self.devices.flat)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError("pmap doesn't use OpSharding.")


# TODO(yashkatariya): Remove this when minimum_jaxlib version is 0.3.17
def _hash_op_sharding(op: xc.OpSharding):
  if op.type == xc.OpSharding.Type.TUPLE:
    return hash(tuple(_hash_op_sharding(o) for o in  op.tuple_shardings))
  return hash((tuple(op.tile_assignment_devices), tuple(op.tile_assignment_dimensions),
               op.type, op.replicate_on_last_tile_dim, tuple(op.last_tile_dims)))


class OpShardingSharding(XLACompatibleSharding):

  def __init__(self, devices: Sequence[Device], op_sharding: xc.OpSharding):
    self._devices = tuple(devices)
    self._op_sharding = op_sharding

  @pxla.maybe_cached_property
  def _op_sharding_hash(self):
    if xla_extension_version >= 81:
      return hash(xc.HloSharding.from_proto(self._op_sharding))
    else:
      return _hash_op_sharding(self._op_sharding)

  def __eq__(self, other):
    if not isinstance(other, OpShardingSharding):
      return False
    return (pxla.are_op_shardings_equal(self._op_sharding, other._op_sharding) and
            self._devices == other._devices)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._devices, self._op_sharding_hash))
    return self._hash

  def __repr__(self):
    return repr(self._op_sharding)

  def is_compatible_aval(self, aval_shape: Shape):
    num_ways_dim_sharded, _ = pxla._get_num_ways_dim_sharded(self._op_sharding)
    if len(aval_shape) < len(num_ways_dim_sharded):
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(num_ways_dim_sharded)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @pxla.maybe_cached_property
  def device_set(self) -> Set[Device]:
    return set(self._devices)

  @functools.lru_cache(maxsize=4096)
  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Index]:
    indices = pxla.op_sharding_to_indices(self._op_sharding, global_shape,
                                          len(self._devices))
    return dict(safe_zip(self._devices, indices))

  def device_replica_id_map(self, global_shape: Shape) -> Mapping[Device, int]:
    return _device_replica_id_map(self, global_shape)

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self._devices)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    return self._op_sharding

  @classmethod
  def get_replicated(cls, device_assignment):
    proto = _get_replicated_op_sharding()
    return cls(device_assignment, proto)
