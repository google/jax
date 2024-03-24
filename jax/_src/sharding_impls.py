# Copyright 2021 The JAX Authors.
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

from __future__ import annotations

import collections
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import functools
import itertools
import math
from typing import Any, NamedTuple, Union, cast

from jax._src import mesh as mesh_lib
from jax._src.op_shardings import (
    is_op_sharding_replicated, are_op_shardings_equal, get_num_ways_dim_sharded,
    op_sharding_to_indices)
from jax._src import sharding
from jax._src import sharding_specs
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge
from jax._src.util import safe_map, safe_zip, use_cpp_class, use_cpp_method
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax._src.partition_spec import PartitionSpec

import numpy as np


Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
XLADeviceAssignment = tuple[Device, ...]


@dataclasses.dataclass(frozen=True)
class TransferToMemoryKind:
  memory_kind: str


@functools.lru_cache(maxsize=4096)
def common_devices_indices_map(s, global_shape: Shape) -> Mapping[Device, Index]:
  hlo_sharding = s._to_xla_hlo_sharding(len(global_shape))
  gspmd_sharding = GSPMDSharding(s._device_assignment, hlo_sharding)
  return gspmd_sharding.devices_indices_map(global_shape)


@functools.lru_cache(maxsize=4096)
def _common_shard_shape(self, global_shape: Shape) -> Shape:
  hlo_sharding = self._to_xla_hlo_sharding(len(global_shape))
  if is_op_sharding_replicated(hlo_sharding):
    return global_shape
  partitions, _ = get_num_ways_dim_sharded(hlo_sharding)
  assert len(partitions) == len(global_shape), (len(partitions), len(global_shape))
  out = []
  for dim, (s, p) in enumerate(safe_zip(global_shape, partitions)):
    try:
      quotient, remainder = divmod(s, p)
    except TypeError:
      # TODO Figure out how to partition dynamic shapes
      raise NotImplementedError
    if remainder != 0:
      raise ValueError(
          f"Sharding {self} implies that array axis {dim} is partitioned "
          f"{p} times, but the dimension size is {s} "
          f"(full shape: {global_shape}, "
          f"per-dimension tiling factors: {partitions} should evenly divide "
          "the shape)")
    out.append(quotient)
  return tuple(out)


# Shardings that inherit from XLACompatibleSharding should implement the
# `_device_assignment` property and `_to_xla_hlo_sharding` method.
@use_cpp_class(xc.XLACompatibleSharding)
class XLACompatibleSharding(sharding.Sharding):
  """A :class:`Sharding` that describes shardings expressible to XLA.

  Subclasses of :class:`XLACompatibleSharding` work with
  all JAX APIs and transformations that use XLA.
  """

  # Abstract methods below that subclasses should implement.

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    raise NotImplementedError('Subclasses should implement this method.')

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    raise NotImplementedError('Subclasses should implement this method.')

  #############################################################################
  # Default implementations below that all subclasses will inherit.

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    return common_devices_indices_map(self, global_shape)

  @functools.cached_property
  def _addressable_device_assignment(self) -> XLADeviceAssignment:
    if self.is_fully_addressable:
      return self._device_assignment
    if hasattr(self, '_internal_device_list'):
      return tuple(self._internal_device_list.addressable_device_list)
    return tuple(d for d in self._device_assignment
                 if d.process_index == d.client.process_index())

  def shard_shape(self, global_shape: Shape) -> Shape:
    return _common_shard_shape(self, global_shape)

  def is_equivalent_to(self: XLACompatibleSharding,  # type: ignore
                       other: XLACompatibleSharding, ndim: int) -> bool:
    try:
      return (are_op_shardings_equal(self._to_xla_hlo_sharding(ndim),
                                      other._to_xla_hlo_sharding(ndim))
              and self._internal_device_list == other._internal_device_list and  # type: ignore
              self.memory_kind == other.memory_kind)
    # NotImplementedError is raised by PmapSharding because it can't lower
    # to OpSharding. So if `other` is a PmapSharding, default to a strict
    # equality check.
    except NotImplementedError:
      return self == other


@functools.lru_cache
def _check_mesh_resource_axis(mesh, parsed_pspec):
  try:
    [mesh.shape[r] for p in parsed_pspec if p is not None
     for r in p]
  except KeyError as e:
    raise ValueError(f"Resource axis: {e.args[0]} of {parsed_pspec.user_spec} is "
                     "undefined.") from None


def hashed_index(x) -> int:
  # This works for both `pjit`/`xmap` indices and `pmap` indices (which might
  # have an integer instead of a slice).
  assert all(v.step is None for v in x if isinstance(v, slice))
  return hash(tuple((v.start, v.stop) if isinstance(v, slice) else v for v in x))


@functools.lru_cache(maxsize=4096)
def device_replica_id_map(sharding, global_shape: Shape) -> Mapping[Device, int]:
  try:
    device_indices_map_fn = sharding.devices_indices_map
  except AttributeError:
    raise ValueError(
        f'Cannot calculate replica ids from sharding: {sharding}. Please '
        'create a device to index mapping for your sharding from which replica '
        'ids will be calculated.') from None

  index_to_replica: dict[int, int] = collections.Counter()
  out = {}
  for device, index in device_indices_map_fn(global_shape).items():
    h_index = hashed_index(index)
    replica_id = index_to_replica[h_index]
    index_to_replica[h_index] += 1
    out[device] = replica_id
  return out


@functools.lru_cache(maxsize=4096)
def named_sharding_to_xla_hlo_sharding(
    self, num_dimensions: int) -> xc.HloSharding:
  mesh_shape = self.mesh.shape
  array_mapping = get_array_mapping(self._parsed_pspec)
  mesh_axis_pos = {name: i for i, name in enumerate(self.mesh.axis_names)}

  special_axes = {}
  if self._manual_axes:
    axis_names = self.mesh.axis_names
    for manual_axis in self._manual_axes:
      special_axes[axis_names.index(manual_axis)] = xc.OpSharding.Type.MANUAL

  replicated_mesh_axes = []
  for i, (axis_name, axis_val) in enumerate(mesh_shape.items()):
    if axis_name not in array_mapping:  # type: ignore
      replicated_mesh_axes.append((i, axis_val))

  if len(replicated_mesh_axes) == len(mesh_shape) and not special_axes:
    return xc.HloSharding.replicate()

  mesh_permutation = []
  new_mesh_shape = [1] * num_dimensions
  for name, pos in sorted(array_mapping.items(), key=lambda x: x[1]):  # type: ignore
    new_mesh_shape[pos] *= mesh_shape[name]
    mesh_permutation.append(mesh_axis_pos[name])

  last_tile_dims = []
  if replicated_mesh_axes:
    axes_by_type = collections.defaultdict(list)
    size_by_type = collections.defaultdict(lambda: 1)  # type: ignore
    assert {x[0] for x in replicated_mesh_axes}.issuperset(set(special_axes.keys()))
    for i, size in replicated_mesh_axes:
      ty = special_axes.get(i, xc.OpSharding.Type.REPLICATED)
      axes_by_type[ty].append(i)
      size_by_type[ty] *= size
    for ty, axes in sorted(axes_by_type.items(), key=lambda x: x[0].value):
      last_tile_dims.append(ty)
      new_mesh_shape.append(size_by_type[ty])
      mesh_permutation.extend(axes)

  # Explanation of the parameters of `HloSharding.iota_tile`.
  # This is the HloShardingV2 format:
  #   * dims: How many ways each dimension is sharded.
  #       Replicated/Manual dims are added added at the end
  #   * reshape_dims: This is the just the shape of the mesh.
  #   * transpose_perm: This is the order in which mesh axes in PartitionSpec
  #       appear relative to mesh.axis_names order.
  #   * subgroup_types: List of type of OpSharding. Type can be REPLICATED and MANUAL.
  # Let's see an example:
  #   Consider input_shape=(8, 4, 2, 2), mesh={'a': 2, 'b': 2, 'c': 2, 'd': 2}
  #   and partition_spec=P(None, ('d', 'b'), 'c').
  #   Arguments to iota_tile will be:
  #     dims = [1, 4, 2, 1, 2]  # 'a' is replicated hence `2` is at the end.
  #     reshape_dims = [2, 2, 2, 2]
  #     transpose_perm = [3, 1, 2, 0]  # 'a' is replicated hence 0 is at the end
  #     subgroup_types = [xc.OpSharding.Type.REPLICATED]
  return xc.HloSharding.iota_tile(
      dims=new_mesh_shape, reshape_dims=tuple(self.mesh.shape.values()),
      transpose_perm=mesh_permutation, subgroup_types=last_tile_dims)


@use_cpp_class(xc.NamedSharding)
class NamedSharding(XLACompatibleSharding):
  r"""A :class:`NamedSharding` expresses sharding using named axes.

  A :class:`NamedSharding` is a pair of a :class:`Mesh` of devices and
  :class:`PartitionSpec` which describes how to shard an array across that
  mesh.

  A :class:`Mesh` is a multidimensional NumPy array of JAX devices,
  where each axis of the mesh has a name, e.g. ``'x'`` or ``'y'``.

  A :class:`PartitionSpec` is a tuple, whose elements can be a ``None``,
  a mesh axis, or a tuple of mesh axes. Each element describes how an input
  dimension is partitioned across zero or more mesh dimensions. For example,
  ``PartitionSpec('x', 'y')`` says that the first dimension of data
  is sharded across ``x`` axis of the mesh, and the second dimension is sharded
  across ``y`` axis of the mesh.

  The Distributed arrays and automatic parallelization
  (https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#namedsharding-gives-a-way-to-express-shardings-with-names)
  tutorial has more details and diagrams that explain how
  :class:`Mesh` and :class:`PartitionSpec` are used.

  Args:
    mesh: A :class:`jax.sharding.Mesh` object.
    spec: A :class:`jax.sharding.PartitionSpec` object.

  Example:

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> spec = P('x', 'y')
    >>> named_sharding = jax.sharding.NamedSharding(mesh, spec)
  """

  mesh: mesh_lib.Mesh
  spec: PartitionSpec
  _memory_kind: str | None
  _parsed_pspec: ParsedPartitionSpec
  _manual_axes: frozenset[MeshAxisName]

  @use_cpp_method()
  def __init__(
      self, mesh: mesh_lib.Mesh, spec: PartitionSpec, *,
      memory_kind: str | None = None, _parsed_pspec=None,
      _manual_axes=frozenset()):
    self.mesh = mesh
    self.spec = spec
    self._memory_kind = memory_kind
    self._manual_axes = _manual_axes
    self._parsed_pspec = preprocess(self.mesh, self.spec, _parsed_pspec)

  # TODO(phawkins): remove this method when jaxlib 0.4.26 or newer is the
  # minimum. This method is called by the C++ sharding implementation in earlier
  # versions.
  if xla_extension_version < 243:
    def _preprocess(self):
      self._parsed_pspec = preprocess(self.mesh, self.spec, self._parsed_pspec)

  def __repr__(self):
    mesh_repr = ", ".join(f"'{k}': {v}" for k, v in self.mesh.shape.items())
    mem = '' if self.memory_kind is None else f', memory_kind={self.memory_kind}'
    return f'NamedSharding(mesh=Mesh({mesh_repr}), spec={self.spec}{mem})'

  def __reduce__(self):
    return (type(self), (self.mesh, self.spec),
            {'memory_kind': self.memory_kind,
             '_manual_axes': self._manual_axes})

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(
          (self.mesh, self.memory_kind, self._parsed_pspec, self._manual_axes))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, NamedSharding):
      return False
    if id(self) == id(other):
      return True
    if (self._parsed_pspec != other._parsed_pspec
        or self.memory_kind != other.memory_kind
        or self._manual_axes != other._manual_axes):
      return False
    return id(self.mesh) == id(other.mesh) or self.mesh == other.mesh

  def is_compatible_aval(self, aval_shape: Shape):
    assert self._parsed_pspec is not None
    if len(aval_shape) < len(self._parsed_pspec):
      extra_msg = (' For scalars the PartitionSpec should be P()'
                   if len(aval_shape) == 0 else '')
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(self._parsed_pspec)}, but was applied to a value of rank "
          f"{len(aval_shape)}.{extra_msg}")

  @classmethod
  def _from_parsed_pspec(
      cls, mesh, parsed_pspec, *, memory_kind=None, _manual_axes=frozenset()
  ):
    return cls(mesh, parsed_pspec.get_partition_spec(),
                memory_kind=memory_kind, _parsed_pspec=parsed_pspec,
                _manual_axes=_manual_axes)

  @property
  def device_set(self) -> set[Device]:
    return self.mesh._flat_devices_set

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return self.mesh._flat_devices_tuple

  @property
  def is_fully_addressable(self) -> bool:
    # Speed up `is_fully_addressable` since there is a high chance that the
    # mesh across multiple NamedSharding objects will be the same.
    return not self.mesh.is_multi_process

  @property
  def addressable_devices(self) -> set[Device]:
    # Override addressable devices because there is a high chance that the mesh
    # across multiple NamedSharding objects will be the same.
    return self.mesh._local_devices_set

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    if self.mesh.size == 1:
      return True
    array_mapping = cast(ParsedPartitionSpec, get_array_mapping(self._parsed_pspec))
    mesh_shape = self.mesh.shape
    num_partitions = 1
    for name in array_mapping:
      num_partitions *= mesh_shape[name]
    return num_partitions == 1

  def with_memory_kind(self, kind: str) -> NamedSharding:
    return NamedSharding(self.mesh, self.spec, memory_kind=kind)

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return named_sharding_to_xla_hlo_sharding(self, num_dimensions)


@functools.lru_cache
def get_replicated_hlo_sharding():
  return xc.HloSharding.replicate()


@use_cpp_class(xc.SingleDeviceSharding)
class SingleDeviceSharding(XLACompatibleSharding):
  """A :class:`Sharding` that places its data on a single device.

  Args:
    device: A single :py:class:`Device`.

  Example:

    >>> single_device_sharding = jax.sharding.SingleDeviceSharding(
    ...     jax.devices()[0])
  """

  _device: Device
  _memory_kind: str | None

  @use_cpp_method()
  def __init__(self, device: Device, *, memory_kind: str | None = None):
    self._device = device
    self._memory_kind = memory_kind

  def __reduce__(self):
    return type(self), (self._device,), {'memory_kind': self._memory_kind}

  def __repr__(self):
    mem = '' if self._memory_kind is None else f', memory_kind={self._memory_kind}'
    return f"SingleDeviceSharding(device={self._device!r}{mem})"

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._device, self.memory_kind))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, SingleDeviceSharding):
      return False
    if id(self) == id(other):
      return True
    return (self._device == other._device and
            self.memory_kind == other.memory_kind)

  @property
  def device_set(self) -> set[Device]:
    return {self._device}

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def with_memory_kind(self, kind: str) -> SingleDeviceSharding:
    return SingleDeviceSharding(self._device, memory_kind=kind)

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:  # type: ignore
    return {self._device: (slice(None),) * len(global_shape)}

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return (self._device,)

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return get_replicated_hlo_sharding()

  @property
  def is_fully_replicated(self) -> bool:
    return True

  @property
  def is_fully_addressable(self) -> bool:
    return True


@functools.lru_cache(maxsize=4096)
def pmap_sharding_devices_indices_map(
    self, global_shape: Shape) -> Mapping[Device, Index]:
  self.shard_shape(global_shape)  # raises a good error message
  indices = sharding_specs.spec_to_indices(global_shape, self.sharding_spec)
  return dict(safe_zip(self.devices.flat, indices))  # type: ignore[arg-type]


@use_cpp_class(xc.PmapSharding)
class PmapSharding(XLACompatibleSharding):
  """Describes a sharding used by :func:`jax.pmap`."""
  devices: np.ndarray
  sharding_spec: sharding_specs.ShardingSpec

  @use_cpp_method()
  def __init__(self, devices: Sequence[Device] | np.ndarray,
               sharding_spec: sharding_specs.ShardingSpec):
    self.devices = np.asarray(devices)
    # The sharding spec should be pmap's sharding spec.
    self.sharding_spec = sharding_spec

  def __reduce__(self):
    return (type(self), (self.devices, self.sharding_spec),
            {'memory_kind': self.memory_kind})

  def __eq__(self, other):
    if not isinstance(other, PmapSharding):
      return False
    if id(self) == id(other):
      return True
    return (self.sharding_spec == other.sharding_spec and
            self.devices.shape == other.devices.shape and
            self._internal_device_list == other._internal_device_list)  # type: ignore

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._internal_device_list, self.sharding_spec))  # type: ignore
    return self._hash

  def __str__(self):
    device_ids = [d.id for d in self.devices.flat]
    return (f'PmapSharding(sharding_spec={self.sharding_spec}, '
            f'{device_ids=}, '
            f'device_platform={self.devices.flat[0].platform.upper()}, '
            f'device_shape={self.devices.shape})')

  def __repr__(self):
    return (f'PmapSharding(sharding_spec={self.sharding_spec}, '
            f'devices={self.devices})')

  def is_equivalent_to(self: PmapSharding, other: PmapSharding,  # type: ignore
                       ndim: int) -> bool:
    return self == other

  # TODO(yashkatariya): Expose `sharded_dim_size` in the API if required.
  @classmethod
  def default(cls, shape: Shape, sharded_dim: int = 0,
              devices: Sequence[xc.Device] | None = None) -> PmapSharding:
    """Creates a :class:`PmapSharding` which matches the default placement
    used by :func:`jax.pmap`.

    Args:
      shape: The shape of the input array.
      sharded_dim: Dimension the input array is sharded on. Defaults to 0.
      devices: Optional sequence of devices to use. If omitted, the implicit
      device order used by pmap is used, which is the order of
        :func:`jax.local_devices`.
    """
    # The dtype doesn't matter here. Its only used for creating the
    # sharding_spec.
    sharding_spec = sharding_specs.create_pmap_sharding_spec(
        tuple(shape), sharded_dim)

    num_ways_sharded = None
    for s in sharding_spec.sharding:
      if isinstance(s, sharding_specs.Unstacked):
        assert num_ways_sharded is None
        num_ways_sharded = s.size
      elif isinstance(s, sharding_specs.Chunked):
        assert num_ways_sharded is None
        if len(s.chunks) == 1:
          num_ways_sharded = s.chunks[0]
        else:
          raise NotImplementedError(
              'Multiple chunks in Chunked dimension not supported.')

    if num_ways_sharded is None:
      raise NotImplementedError(
          '`None` to sharded_dim is not supported. Please file a jax '
          'issue if you need this feature.')

    if devices is None:
      pmap_devices: np.ndarray = np.array(
          xla_bridge.local_devices()[:num_ways_sharded])
    else:
      pmap_devices = np.array(devices)
    return cls(pmap_devices, sharding_spec)

  @functools.cached_property
  def device_set(self) -> set[Device]:
    return set(self.devices.flat)

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    return pmap_sharding_devices_indices_map(self, global_shape)

  @functools.cached_property
  def _device_assignment(self) -> XLADeviceAssignment:
    return tuple(self.devices.flat)

  @property
  def memory_kind(self) -> str | None:
    try:
      return self._internal_device_list.default_memory_kind  # type: ignore
    except:
      return None

  def with_memory_kind(self, kind: str):
    raise NotImplementedError("pmap does not support memories.")

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    raise NotImplementedError("pmap doesn't use OpSharding.")

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    for s in self.sharding_spec.sharding:
      if isinstance(s, (sharding_specs.Unstacked, sharding_specs.Chunked)):
        return False
    return True

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    return self._internal_device_list.is_fully_addressable  # type: ignore

  def shard_shape(self, global_shape: Shape) -> Shape:
    sharded_dim = None
    sharded_dim_size = None
    for i, s in enumerate(self.sharding_spec.sharding):
      if isinstance(s, sharding_specs.Unstacked):
        sharded_dim = i
        sharded_dim_size = s.size
        sharded_shape = util.tuple_delete(global_shape, sharded_dim)
        break
      elif isinstance(s, sharding_specs.Chunked):
        sharded_dim = i
        assert len(s.chunks) == 1, s.chunks
        sharded_dim_size = s.chunks[0]
        sharded_shape = util.tuple_update(global_shape, sharded_dim, 1)
        break
    if sharded_dim is None:
      return global_shape
    if global_shape[sharded_dim] != sharded_dim_size:
      raise ValueError(
          f'The sharded dimension must be equal to the number of '
          f'devices passed to PmapSharding. Got sharded dimension {sharded_dim} '
          f'with value {global_shape[sharded_dim]} in shape {global_shape} and '
          f'the number of devices={len(self._device_assignment)}')
    return sharded_shape


def _op_sharding_to_pos_sharding(
    op_sharding: xc.OpSharding | xc.HloSharding,
    device_assignment: Sequence[xc.Device],
    memory_kind: str | None = None) -> PositionalSharding:
  if isinstance(op_sharding, xc.OpSharding):
    op_sharding = xc.HloSharding.from_proto(op_sharding)  # type: ignore

  if op_sharding.is_replicated():
    return PositionalSharding(
        device_assignment, memory_kind=memory_kind).replicate()

  if len(op_sharding.subgroup_types()) > 1:
    raise NotImplementedError(
        'Unhandled HloSharding type. Please open a bug report!'
    )

  name = device_assignment[0].platform.upper()
  ids = np.array(
      [DeviceIdSet(name, i) for i in op_sharding.tile_assignment_devices()]
  )
  p = PositionalSharding._remake(tuple(device_assignment), ids,
                                 memory_kind=memory_kind)
  p = p.reshape(op_sharding.tile_assignment_dimensions())
  if op_sharding.replicate_on_last_tile_dim():
    p = p.replicate(-1, keepdims=False)
  return p


@functools.lru_cache(maxsize=4096)
def _positional_sharding_to_xla_hlo_sharding(
    self, num_dimensions: int) -> xc.HloSharding:
  if self.shape == (1,) * self.ndim:
    return get_replicated_hlo_sharding()

  pbuf = xc.OpSharding()
  shape = self.shape[self.ndim - num_dimensions:]  # 'rank promotion' of val
  set_size, = {len(device_set) for device_set in self._ids.flat}
  pbuf.type = xc.OpSharding.Type.OTHER
  if set_size > 1:
    pbuf.last_tile_dims = [xc.OpSharding.Type.REPLICATED]
    pbuf.tile_assignment_dimensions = (*shape, set_size)
  else:
    pbuf.tile_assignment_dimensions = shape
  pbuf.tile_assignment_devices = [i for ids in self._ids.flat for i in ids]
  product_of_dims = math.prod(pbuf.tile_assignment_dimensions)
  num_devices = len(pbuf.tile_assignment_devices)
  assert product_of_dims == num_devices, (product_of_dims, num_devices)
  return xc.HloSharding.from_proto(pbuf)


class PositionalSharding(XLACompatibleSharding):
  """
  A sharding strategy that arranges data based on device positions.

  This strategy enables efficient data distribution across devices,
  making it suitable for parallel processing on different hardware platforms.

  Example:
    >>> devices = [xc.Device("GPU", 0), xc.Device("GPU", 1)] # doctest: +SKIP
    >>> sharding = PositionalSharding(devices) # doctest: +SKIP
    >>> print(sharding.shape)  # Output: (2,) # doctest: +SKIP
    >>> print(sharding.ndim)   # Output: 1 # doctest: +SKIP

  Initialize a PositionalSharding instance with two GPU devices:
    >>> devices = [xc.Device("GPU", 0), xc.Device("GPU", 1)] # doctest: +SKIP
    >>> sharding = PositionalSharding(devices) # doctest: +SKIP
  """
  _devices: tuple[xc.Device, ...]
  _memory_kind: str | None
  _ids: np.ndarray  # dtype DeviceIdSet

  def __init__(self, devices: Sequence[xc.Device] | np.ndarray,
               *, memory_kind: str | None = None):
    if xla_extension_version >= 235:
      super().__init__()
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices, dtype='object')
    if not devices.size:
      raise ValueError(f"{self.__class__.__name__}.__init__ requires at least "
                       f"one device, got {devices}")
    self._devices = tuple(devices.flat)
    self._memory_kind = memory_kind
    name = self._devices[0].platform.upper()
    self._ids = np.array([DeviceIdSet(name, i) for i in range(devices.size)],
                         dtype='object').reshape(devices.shape)
    self._internal_device_list = xc.DeviceList(self._devices)
    self._memory_kind = xc.check_and_canonicalize_memory_kind(
        self._memory_kind, self._internal_device_list)

  @property
  def shape(self):
    return self._ids.shape

  @property
  def ndim(self):
    return self._ids.ndim

  def __repr__(self) -> str:
    cls_name = self.__class__.__name__
    ids = self._ids.copy()
    platform_name = self._devices[0].platform.upper()
    for idx, x in np.ndenumerate(ids):
      ids[idx] = DeviceIdSet(platform_name, *(self._devices[i].id for i in x))
    body = np.array2string(ids, prefix=cls_name + '(', suffix=')',
                           max_line_width=100)
    mem = '' if self._memory_kind is None else f', memory_kind={self._memory_kind}'
    return f'{cls_name}({body}{mem}, shape={self.shape})'

  def reshape(self, *shape) -> PositionalSharding:
    """
    Returns a new PositionalSharding instance with a reshaped data layout.

    Args:
      *shape: New shape dimensions.

    Returns:
      PositionalSharding: A new PositionalSharding instance.

    Example:
      Reshape a PositionalSharding instance:
        >>> new_sharding = sharding.reshape(2, 1) # doctest: +SKIP
    """
    return self._remake(self._devices, self._ids.reshape(*shape))

  def transpose(self, *axes) -> PositionalSharding:
    """
    Create a new PositionalSharding instance with a transposed data layout.

    Transposing a sharding instance involves rearranging the dimensions of the
    data layout. This method allows you to change the order of dimensions for
    efficient data processing or compatibility with other operations.

    Args:
      *axes: Axes permutation for the transpose operation as individual arguments.

    Example:
      Transpose a PositionalSharding instance, swapping the first and second dimensions:
        >>> transposed_sharding = sharding.transpose(1, 0) # doctest: +SKIP

    Returns:
      PositionalSharding: A new PositionalSharding instance with the transposed data layout.
    """
    return self._remake(self._devices, self._ids.transpose(*axes))
  T = property(transpose)

  def replicate(self, axis=None, keepdims=True) -> PositionalSharding:
    """
    Create a new PositionalSharding instance with replicated data along a specified axis.

    Replicating a sharding instance involves creating additional copies of the
    data to distribute it along a specific axis. This method allows you to replicate
    the data within the sharding strategy, which can be useful for parallel
    processing or data redundancy.

    Args:
      axis: The axis along which data replication is performed. If not specified, replication
        is applied across all dimensions.
      keepdims: Whether to keep the dimensions of the sharding consistent or not. When set to
        True, the replicated dimensions are retained; otherwise, they are collapsed.

    Example:
      Replicate a PositionalSharding instance along the first axis while keeping dimensions:
        >>> replicated_sharding = sharding.replicate(axis=0, keepdims=True) # doctest: +SKIP

    Returns:
      PositionalSharding: A new PositionalSharding instance with the data replicated along the specified axis and dimensionality adjustments based on the 'keepdims' parameter.
    """
    new_ids = self._ids.sum(axis=axis, keepdims=keepdims)  # union
    return self._remake(self._devices, new_ids)

  @classmethod
  def _remake(
      cls, devices: tuple[xc.Device, ...], ids: np.ndarray,
      *, memory_kind: str | None = None) -> PositionalSharding:
    self = cls.__new__(cls)
    self._devices = devices
    self._ids = ids
    self._internal_device_list = xc.DeviceList(self._devices)
    self._memory_kind = xc.check_and_canonicalize_memory_kind(
        memory_kind, self._internal_device_list)
    return self

  # Hashable

  def __hash__(self) -> int:
    if not hasattr(self, '_hash'):
      self._hash = hash((self._internal_device_list, self.memory_kind))
    return self._hash

  def __eq__(self, other) -> bool:
    if not isinstance(other, PositionalSharding):
      return False
    if id(self) == id(other):
      return True
    all_ids_equal = np.array_equal(self._ids,other._ids)
    mem_kind_equal = self.memory_kind == other.memory_kind
    if (id(self._devices) == id(other._devices) and mem_kind_equal and
        all_ids_equal):
      return True
    return (mem_kind_equal and all_ids_equal and
            self._internal_device_list == other._internal_device_list)

  # Sharding interface

  @functools.cached_property
  def device_set(self) -> set[xc.Device]:
    return set(self._devices)

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def with_memory_kind(self, kind: str) -> PositionalSharding:
    """
    Create a new PositionalSharding instance with a specified memory kind.

    Memory kind refers to the type of memory or storage associated with the sharding strategy.
    This method allows you to customize the memory kind used by the sharding for efficient
    memory management and allocation.

    Args:
      kind (str): The memory kind to associate with the sharding.
        Common memory kinds include "HBM" (High-Bandwidth Memory) and "DDR" (Double Data Rate memory).

    Example:
      Create a new PositionalSharding instance with a specific memory kind,
      such as "HBM" (High-Bandwidth Memory):
      >>> sharding_with_mem_kind = sharding.with_memory_kind("HBM") # doctest: +SKIP

    Returns:
      PositionalSharding: A new PositionalSharding instance with the specified memory kind.
    """
    return PositionalSharding(self._devices, memory_kind=kind)

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    return self.shape == (1,) * self.ndim

  # XLACompatibleSharding interface

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return self._devices

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return _positional_sharding_to_xla_hlo_sharding(self, num_dimensions)

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    return self._internal_device_list.is_fully_addressable


class DeviceIdSet:
  _name: str
  _ids: frozenset[int]
  def __init__(self, name, *ids):
    self._name = name
    self._ids = frozenset(ids)

  def __iter__(self):
    return iter(sorted(self._ids))

  def __add__(self, other) -> DeviceIdSet:
    assert isinstance(other, DeviceIdSet)
    return DeviceIdSet(self._name, *(self._ids | other._ids))

  def __len__(self) -> int:
    return len(self._ids)

  def __repr__(self) -> str:
    ids = ', '.join(safe_map(str, sorted(self._ids)))
    return f'{{{self._name} {ids}}}'

  def __hash__(self) -> int:
    return hash((self._name, self._ids))

  def __eq__(self, other) -> bool:
    return (isinstance(other, DeviceIdSet) and self._name == other._name and
            self._ids == other._ids)


@functools.lru_cache(maxsize=4096)
def gspmd_sharding_devices_indices_map(
    self, global_shape: Shape) -> Mapping[Device, Index]:
  self.shard_shape(global_shape)  # raises a good error message
  indices = op_sharding_to_indices(self._hlo_sharding, global_shape,
                                    len(self._devices))
  return dict(safe_zip(self._devices, indices))


@use_cpp_class(xc.GSPMDSharding)
class GSPMDSharding(XLACompatibleSharding):
  _devices: tuple[Device, ...]
  _hlo_sharding: xc.HloSharding
  _memory_kind: str | None
  _device_list: xc.DeviceList | None

  @use_cpp_method()
  def __init__(self, devices: Sequence[Device],
               op_sharding: xc.OpSharding | xc.HloSharding,
               *, memory_kind: str | None = None,
               _device_list: xc.DeviceList | None = None):
    self._devices = tuple(devices)
    if isinstance(op_sharding, xc.OpSharding):
      self._hlo_sharding = xc.HloSharding.from_proto(op_sharding)
    else:
      self._hlo_sharding = op_sharding
    self._memory_kind = memory_kind

  def __reduce__(self):
    return (type(self), (self._devices, self._hlo_sharding.to_proto()),
            {'memory_kind': self._memory_kind})

  @functools.cached_property
  def _hlo_sharding_hash(self):
    if self.is_fully_replicated:
      return hash(get_replicated_hlo_sharding())
    return hash(self._hlo_sharding)

  def __eq__(self, other):
    if not isinstance(other, GSPMDSharding):
      return False
    if id(self) == id(other):
      return True
    return (are_op_shardings_equal(self._hlo_sharding, other._hlo_sharding)
            and self.memory_kind == other.memory_kind
            and self._internal_device_list == other._internal_device_list)  # type: ignore

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._internal_device_list, self._hlo_sharding_hash,  # type: ignore
                        self.memory_kind))
    return self._hash

  def __repr__(self):
    mem = '' if self._memory_kind is None else f', memory_kind={self._memory_kind}'
    return f'GSPMDSharding({self._hlo_sharding!r}{mem})'

  def is_compatible_aval(self, aval_shape: Shape):
    num_ways_dim_sharded, _ = get_num_ways_dim_sharded(self._hlo_sharding)
    if len(aval_shape) < len(num_ways_dim_sharded):
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(num_ways_dim_sharded)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @functools.cached_property
  def device_set(self) -> set[Device]:
    return set(self._devices)

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def with_memory_kind(self, kind: str) -> GSPMDSharding:
    return GSPMDSharding(self._devices, self._hlo_sharding, memory_kind=kind)

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    return gspmd_sharding_devices_indices_map(self, global_shape)

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return self._devices

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return self._hlo_sharding

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    return is_op_sharding_replicated(self._hlo_sharding)

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    return self._internal_device_list.is_fully_addressable  # type: ignore

  @classmethod
  def get_replicated(cls, device_assignment, *, memory_kind: str | None = None):
    return cls(tuple(device_assignment), get_replicated_hlo_sharding(),
               memory_kind=memory_kind)


class AUTO:

  def __init__(self, mesh: mesh_lib.Mesh):
    self.mesh = mesh


def is_auto(x):
  return isinstance(x, AUTO)


class UnspecifiedValue:
  def __repr__(self):
    return "UnspecifiedValue"
UNSPECIFIED = UnspecifiedValue()

def is_unspecified(x):
  return isinstance(x, UnspecifiedValue)

def is_unspecified_or_auto(x):
  return is_auto(x) or is_unspecified(x)


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
ArrayMapping = OrderedDict[MeshAxisName, int]
ArrayMappingOrAutoOrUnspecified = Union[ArrayMapping, AUTO, UnspecifiedValue]

def array_mapping_to_axis_resources(array_mapping: ArrayMapping):
  if not array_mapping:
    return PartitionSpec()
  max_index = -1
  reverse_map = collections.defaultdict(list)
  for axis, index in array_mapping.items():
    reverse_map[index].append(axis)
    if index > max_index:
      max_index = index
  partitions = []
  for i in range(max_index + 1):
    axis = reverse_map[i]
    if axis:
      partitions.append(axis[0] if len(axis) == 1 else tuple(axis))
    else:
      partitions.append(None)
  return PartitionSpec(*partitions)

def get_array_mapping(
    axis_resources: ParsedPartitionSpec | AUTO | UnspecifiedValue
) -> ArrayMappingOrAutoOrUnspecified:
  # TODO(yashkatariya): Use `TypeGuard` on `is_auto` when it is supported.
  # Don't use `is_auto` here to satisfy pytype and mypy.
  if isinstance(axis_resources, (AUTO, UnspecifiedValue)):
    return axis_resources
  return OrderedDict((axis, i)
                     for i, axes in enumerate(axis_resources)
                     if axes is not None for axis in axes)


get_single_pspec = lambda p: array_mapping_to_axis_resources(
    cast(ArrayMapping, get_array_mapping(p)))


class SpecSync(enum.IntEnum):
  """Encodes how much out of sync the real value of partitions is compared to the user specified one.

  We use this to make sure we don't show garbage modified values while claiming
  that the users have specified them like that.
  """
  OUT_OF_SYNC = 0  # Arbitrary changes, including new axes inserted
  DIM_PERMUTE = 1  # Dimensions permuted, but no new sharding axes
  IN_SYNC = 2  # Entirely in sync

class ParsedPartitionSpec:
  __slots__ = ('unsafe_user_spec', 'partitions', 'sync')

  def __init__(self, user_spec, partitions, sync=SpecSync.IN_SYNC):
    self.unsafe_user_spec = user_spec
    # None in partitions represents unconstrained dim.
    # TODO(yashkatariya): May use a sentinel value.
    self.partitions = tuple(partitions)
    self.sync = sync

  @property
  def user_spec(self):
    return self.unsynced_user_spec(SpecSync.IN_SYNC)

  def get_partition_spec(self) -> PartitionSpec:
    if self.sync < SpecSync.IN_SYNC:
      return get_single_pspec(self)
    else:
      if isinstance(self.unsafe_user_spec, PartitionSpec):
        return self.unsafe_user_spec
      else:
        return get_single_pspec(self)

  def unsynced_user_spec(self, min_sync):
    if self.sync < min_sync:
      raise AssertionError(f"Please open a bug report! ({self.sync} >= {min_sync})")
    return self.unsafe_user_spec

  def insert_axis_partitions(self, dim, val):
    parts = self.partitions
    too_short = dim - len(parts)
    if too_short > 0:
      parts += ((),) * too_short
    new_partitions = util.tuple_insert(parts, dim, val)
    new_sync = SpecSync.DIM_PERMUTE if (val == () or val is None) else SpecSync.OUT_OF_SYNC
    return ParsedPartitionSpec(self.unsafe_user_spec, new_partitions, sync=new_sync)

  @classmethod
  def from_user_input(cls, entry, arg_name, allow_unconstrained_dims=False):
    if entry is None:
      return cls(entry, ())
    if not isinstance(entry, PartitionSpec):
      raise TypeError(f"{arg_name} are expected to be "
                      f"PartitionSpec instances or None, but got {entry}")
    axis_specs = []
    for axis_spec in entry:
      if axis_spec is None:
        axis_spec = ()
      elif isinstance(axis_spec, (list, tuple)):
        axis_spec = tuple(axis_spec)
      elif axis_spec == PartitionSpec.UNCONSTRAINED:
        if not allow_unconstrained_dims:
          raise ValueError(f"Unconstrained dims are not allowed: {entry}")
        axis_spec = None
      else:
        axis_spec = (axis_spec,)
      axis_specs.append(axis_spec)
    return cls(entry, axis_specs)

  def __hash__(self):
    return hash((self.partitions, self.sync))

  def __eq__(self, other):
    return (self.partitions == other.partitions and
            self.sync == other.sync)

  def __len__(self):
    return len(self.partitions)

  def __getitem__(self, i):
    return self.partitions[i]

  def __iter__(self):
    return iter(self.partitions)

  def __repr__(self):
    return (f"ParsedPartitionSpec(partitions={self.partitions}, "
            f"unsafe_user_spec={self.unsafe_user_spec}, "
            f"sync={self.sync})")

class CanonicalizedParsedPartitionSpec(ParsedPartitionSpec):
  """ParsedPartitionSpecs that are canonicalized.

  ParsedPartitionSpecs may contain trailing empty tuples, that make them
  semantically different in general, and yet in some situations we prefer
  to regard them as equivalent. For example, partitions of () and ((),)
  cannot be always considered equivalent, since the first one is a valid
  spec for a scalar value, while the second is not! However, when either of
  those are applied to a 2D array, they both mean that the array is fully
  replicated.

  So CanonicalizedParsedPartitionSpecs removes the trailing empty tuples from
  partitions.
  """

  def __init__(self, parsed_pspec: ParsedPartitionSpec):
    partitions = list(parsed_pspec.partitions)
    while partitions and partitions[-1] == ():
      partitions.pop()

    super().__init__(parsed_pspec.unsafe_user_spec, partitions,
                     parsed_pspec.sync)

  def __repr__(self):
    return (f"CanonicalizedParsedPartitionSpec(partitions={self.partitions}, "
            f"unsafe_user_spec={self.unsafe_user_spec}, "
            f"sync={self.sync})")


def preprocess(mesh, spec, parsed_pspec):
  # This split exists because you can pass `_parsed_pspec` that has been
  # modified from the original. For example: Adding extra dimension to
  # axis_resources for vmap handlers. In such cases you need to preserve the
  # `sync` attribute of parsed pspecs.
  # PartitionSpec is inferred from the parsed pspec in this case.
  # TODO(yaskatariya): Remove this and replace this with a normalized
  # representation of Parsed Pspec
  if parsed_pspec is None:
    parsed_pspec, _, _ = prepare_axis_resources(
        PartitionSpec() if spec is None else spec,
        "NamedSharding spec", allow_unconstrained_dims=True)

  _check_mesh_resource_axis(mesh, parsed_pspec)
  return parsed_pspec


def prepare_axis_resources(axis_resources,
                           arg_name,
                           allow_unconstrained_dims=False):
  # PyTrees don't treat None values as leaves, so we use an is_leaf function.
  entries, treedef = tree_util.tree_flatten(
      axis_resources, is_leaf=lambda x: x is None)
  what = f"{arg_name} leaf specifications"

  new_entries = []
  for entry in entries:
    if is_unspecified_or_auto(entry) or entry is None:
      new_entries.append(entry)
    elif isinstance(entry, sharding.Sharding):
      if isinstance(entry, PmapSharding):
        raise ValueError(f'One of {what} got sharding {entry} which is not '
                         'allowed.')
      if not isinstance(entry, XLACompatibleSharding):
        raise ValueError(f'One of {what} got sharding {entry} which is not a '
                         'subclass of XLACompatibleSharding.')
      new_entries.append(entry)
    else:
      new_entries.append(ParsedPartitionSpec.from_user_input(
          entry, what, allow_unconstrained_dims=allow_unconstrained_dims))

  _check_unique_resources(new_entries, arg_name)
  return tree_util.tree_unflatten(treedef, new_entries), new_entries, treedef


def _check_unique_resources(axis_resources, arg_name):
  for arg_axis_resources in axis_resources:
    if not arg_axis_resources: continue
    if (is_unspecified_or_auto(arg_axis_resources) or
        isinstance(arg_axis_resources, XLACompatibleSharding)):
      continue
    constrained_dims = [d for d in arg_axis_resources if d is not None]
    resource_counts = collections.Counter(
        itertools.chain.from_iterable(constrained_dims))
    if not resource_counts: continue
    if resource_counts.most_common(1)[0][1] > 1:
      multiple_uses = [r for r, c in resource_counts.items() if c > 1]
      if multiple_uses:
        raise ValueError(f"A single {arg_name} specification can map every mesh axis "
                         f"to at most one positional dimension, but {arg_axis_resources.user_spec} "
                         f"has duplicate entries for {mesh_lib.show_axes(multiple_uses)}")

# Axis environments

class AxisEnv(NamedTuple):
  """Represents a pmap mesh (only along the replica axes)."""
  nreps: int
  names: tuple[Any, ...]
  sizes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SPMDAxisContext:
  """A hardware axis context for parallel computations that use the GSPMD partitioner.

  This includes the mesh that will later by used to execute this computation,
  as well as a set of mesh axes that are currently (e.g. because the current lowering
  is invoked inside an xmap) lowered in the MANUAL sharding mode.
  """
  mesh: mesh_lib.Mesh
  manual_axes: frozenset[MeshAxisName] = frozenset()

  @property
  def axis_env(self):
    # All collectives that touch axis_env should remember to set use_global_device_ids
    # when this context is enabled!
    if self.manual_axes != frozenset(self.mesh.axis_names):
      raise NotImplementedError(
          "Collectives in manually partitioned computations are only supported "
          "when all mesh axes are partitioned manually (no partial automatic sharding). "
          "Make sure that you mention all mesh axes in axis_resources!")
    return self.unsafe_axis_env

  @property
  def unsafe_axis_env(self):
    return AxisEnv(
        nreps=self.mesh.size,
        names=self.mesh.axis_names,
        sizes=tuple(self.mesh.shape.values()))

  def extend_manual(self, axes: frozenset[MeshAxisName]) -> SPMDAxisContext:
    return SPMDAxisContext(self.mesh, self.manual_axes | axes)


@dataclasses.dataclass(frozen=True)
class ReplicaAxisContext:
  """A hardware axis context for parallel computations that are partitioned by JAX.

  Unlike in the SPMDAxisContext, this means that JAX might need to emit calls to
  explicit collectives.
  """
  axis_env: AxisEnv


@dataclasses.dataclass(frozen=True)
class ShardingContext:
  """A hardware axis context for parallel computations that use the sharding
  interface.

  This context also uses the GSPMD partitioner.
  """
  num_devices: int
  device_assignment: tuple[xc.Device, ...] | None = None

  def __post_init__(self):
    if self.device_assignment is not None:
      assert isinstance(self.device_assignment, tuple)
      assert self.num_devices == len(self.device_assignment)

  # Similar to SPMDContext as ShardingContext also uses the GSPMD partitioner.
  @property
  def axis_env(self):
    return AxisEnv(nreps=1, names=(), sizes=())


# -------------------- XLA OpSharding to PartitionSpec --------------------
# Note that OpSharding is more expressive than PartitionSpecs, so it's not
# always possible to convert them, but the code below should at least
# support handle all cases when this is possible.

def strides_for_sizes(sizes):
  """Returns an array of strides for major-to-minor sizes."""
  return np.cumprod(sizes[::-1])[::-1] // np.asarray(sizes)

def unflatten_array(named_sizes, assignment):
  """Recovers the ordering of axis names based on a device assignment.

  The device assignments that this function can convert into axis orders
  are of the form::

    np.arange(np.prod(named_sizes.values())).transpose(...).flatten()

  for some transposition ``...``. This is satisfied by all OpSharding assignments
  generated from partition specs.

  Arguments:
    named_sizes: A dictionary mapping axis names to their sizes.
    assignment: A permutation of integers between 0 and the product of all
      named sizes.

  Returns:
    A major-to-minor list of axis names that corresponds to the given assignment.
  """
  named_sizes = {name: size for name, size in named_sizes.items() if size != 1}
  sizes = np.fromiter(named_sizes.values(), dtype=np.int64)
  strides = strides_for_sizes(sizes)
  dims = explode_superdims(sizes, unflatten_superdims(assignment))
  dim_to_name = {(size, stride): name for size, stride, name in zip(sizes, strides, named_sizes)}
  return [dim_to_name[d] for d in dims]

def unflatten_superdims(assignment):
  """Unflatten a list of dimension sizes and their strides that generates assignment.

  If this function succeeds for a given ``assignment``, then the following property
  should be satisfied::

    dims_with_strides = unflatten_superdims(assignment)
    base_array = np.arange(map(fst, sorted(dims_with_strides, key=snd, reverse=True)))
    assignment == base_array.transpose(argsort(dims_with_strides, key=snd, reverse=True)).flatten()

  That is, the returned dimensions list all sizes of the base array (with strides
  indicating their initial order). The order of dimensions in the list corresponds
  to the permutation that applied to the base array generates the assignment.
  """
  def check(cond):
    if cond: return
    raise NotImplementedError("Failed to convert OpSharding into a ShardingSpec. "
                              "Please open a bug report!")
  flat_assignment = np.asarray(assignment, dtype=np.int64)
  check(flat_assignment[0] == 0)
  dims = []
  while flat_assignment.size > 1:
    stride = flat_assignment[1]
    for i in range(len(flat_assignment)):
      if flat_assignment[i] != i * stride: break
    else:
      # After this loop i should point to an "element after the sequence", so
      # we have to increment it if the whole array is a strided sequence.
      i += 1
    size = i
    dims.append((size, stride))
    assert size > 1  # Ensure progress
    flat_assignment = flat_assignment[::size]
  return dims

def explode_superdims(sizes, dims):
  """Explode superdims to fit a known shape.

  The unflattening process might mistakenly generate too few too large dimensions.
  For example, ``unflatten_superdims(np.arange(n))`` always returns ``[(n, 1)]``.
  This function takes a list of such contiguous super-dimensions and splits them
  into smaller dimensions such that::

    set(map(fst, explode_superdims(sizes, dims))) == set(sizes)
  """
  strides_to_sizes = {stride: size for size, stride in zip(sizes, strides_for_sizes(sizes))}
  dims = list(reversed(dims))
  final_dims = []
  for size, stride in dims:
    target_size = strides_to_sizes[stride]
    new_dims = []
    while size > target_size:
      assert target_size > 1  # Ensure progress
      assert size % target_size == 0
      new_dims.append((target_size, stride))
      size //= target_size
      stride *= target_size
      target_size = strides_to_sizes[stride]
    assert size == target_size
    new_dims.append((size, stride))
    final_dims += reversed(new_dims)
  return final_dims

def parse_flatten_op_sharding(hlo_sharding: xc.OpSharding | xc.HloSharding,
                              mesh: mesh_lib.Mesh) -> Sequence[ParsedPartitionSpec]:
  if isinstance(hlo_sharding, xc.OpSharding):
    hlo_sharding = xc.HloSharding.from_proto(hlo_sharding)  # type: ignore
  if hlo_sharding.tuple_elements():
    out: list[ParsedPartitionSpec] = []
    for s in hlo_sharding.tuple_elements():
      out.extend(parse_flatten_op_sharding(s, mesh))
    return out
  elif hlo_sharding.is_replicated():
    return [CanonicalizedParsedPartitionSpec(
        ParsedPartitionSpec(PartitionSpec(), ()))]
  elif hlo_sharding.is_tiled():
    mesh_shape = mesh.shape
    mesh_axis_order = unflatten_array(
        mesh.shape, hlo_sharding.tile_assignment_devices()
    )
    mesh_axis = iter(mesh_axis_order)
    shape = hlo_sharding.tile_assignment_dimensions()
    partitions = []
    for dim_size in shape:
      dim_partitions = []
      while dim_size > 1:
        axis = next(mesh_axis)
        axis_size = mesh_shape[axis]
        assert dim_size % axis_size == 0
        dim_size //= axis_size
        dim_partitions.append(axis)
      partitions.append(tuple(dim_partitions))
    if len(hlo_sharding.subgroup_types()) > 1:
      raise NotImplementedError(
          'Unhandled HloSharding type. Please open a bug report!'
      )
    if hlo_sharding.replicate_on_last_tile_dim():
      partitions = partitions[:-1]
    return [CanonicalizedParsedPartitionSpec(
        ParsedPartitionSpec('<internally generated spec>', partitions))]
  else:
    raise AssertionError("Unhandled OpSharding type. Please open a bug report!")
