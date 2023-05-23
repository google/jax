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
import dataclasses
import enum
import functools
import itertools
import math
import operator as op
import sys
from typing import (Any, Dict, FrozenSet, List, Mapping, Optional, OrderedDict,
                    NamedTuple, Sequence, Set, Tuple, Union, cast)

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
from jax._src.partition_spec import PartitionSpec

import numpy as np

if sys.version_info >= (3, 9):
  OrderedDictType = OrderedDict
else:
  OrderedDictType = Dict


Shape = Tuple[int, ...]
Device = xc.Device
Index = Tuple[slice, ...]
XLADeviceAssignment = Tuple[Device, ...]


# Shardings that inherit from XLACompatibleSharding should implement the
# `_device_assignment` property and `_to_xla_op_sharding` method.
@use_cpp_class(xc.XLACompatibleSharding)
class XLACompatibleSharding(sharding.Sharding):
  """A `Sharding` that describes shardings expressible to XLA.

  Any ``Sharding`` that is a subclass of ``XLACompatibleSharding`` will work
  with all JAX APIs and transformations that use XLA.
  """

  # Abstract methods below that subclasses should implement.

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    raise NotImplementedError('Subclasses should implement this method.')

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError('Subclasses should implement this method.')

  #############################################################################
  # Default implementations below that all subclasses will inherit.

  @functools.lru_cache(maxsize=4096)
  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    op_sharding = self._to_xla_op_sharding(len(global_shape))
    gspmd_sharding = GSPMDSharding(self._device_assignment, op_sharding)
    return gspmd_sharding.devices_indices_map(global_shape)

  @functools.cached_property
  def _addressable_device_assignment(self) -> XLADeviceAssignment:
    return tuple(d for d in self._device_assignment
                 if d.process_index == d.client.process_index())

  @functools.lru_cache(maxsize=4096)
  def shard_shape(self, global_shape: Shape) -> Shape:
    op_sharding = cast(xc.OpSharding, self._to_xla_op_sharding(len(global_shape)))
    if is_op_sharding_replicated(op_sharding):
      return global_shape
    partitions, _ = get_num_ways_dim_sharded(op_sharding)
    assert len(partitions) == len(global_shape), (len(partitions), len(global_shape))
    out = []
    for dim, (s, p) in enumerate(safe_zip(global_shape, partitions)):
      quotient, remainder = divmod(s, p)
      if remainder != 0:
        raise ValueError(
            f"Sharding {self} implies that array axis {dim} is partitioned "
            f"{p} times, but the dimension size is {s} "
            f"(full shape: {global_shape}, "
            f"per-dimension tiling factors: {partitions} should evenly divide "
            "the shape)")
      out.append(quotient)
    return tuple(out)

  def is_equivalent_to(self: XLACompatibleSharding,  # type: ignore
                       other: XLACompatibleSharding, ndim: int) -> bool:
    try:
      return (are_op_shardings_equal(self._to_xla_op_sharding(ndim),
                                     other._to_xla_op_sharding(ndim))
              and self._device_assignment == other._device_assignment)
    # NotImplementedError is raised by PmapSharding because it can't lower
    # to OpSharding. So if `other` is a PmapSharding, default to a strict
    # equality check.
    except NotImplementedError:
      return self == other


@functools.lru_cache()
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

  index_to_replica: Dict[int, int] = collections.Counter()
  out = {}
  for device, index in device_indices_map_fn(global_shape).items():
    h_index = hashed_index(index)
    replica_id = index_to_replica[h_index]
    index_to_replica[h_index] += 1
    out[device] = replica_id
  return out


@use_cpp_class(xc.NamedSharding)
class NamedSharding(XLACompatibleSharding):
  r"""NamedSharding is a way to express ``Sharding``\s using named axes.

  ``Mesh`` and ``PartitionSpec`` can be used to express a ``Sharding`` with a name.

  ``Mesh`` is a NumPy array of JAX devices in a multi-dimensional grid,
  where each axis of the mesh has a name, e.g. 'x' or 'y'. Another name for
  ``Mesh`` is "logical mesh".

  ``PartitionSpec`` is a tuple, whose elements can be a ``None``,
  a mesh axis or a tuple of mesh axes. Each element describes how an input
  dimension is partitioned across zero or more mesh dimensions. For example,
  PartitionSpec('x', 'y') is a PartitionSpec where the first dimension of data
  is sharded across ``x`` axis of the mesh, and the second dimension is sharded
  across ``y`` axis of the mesh.

  The Distributed arrays and automatic parallelization
  (https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#namedsharding-gives-a-way-to-express-shardings-with-names)
  goes into more details and has diagrams to help explain the concept about
  ``Mesh`` and ``PartitionSpec``.

  Args:
    mesh: A ``jax.sharding.Mesh`` object.
    spec: A ``jax.sharding.PartitionSpec`` object.

  Example:

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> spec = P('x', 'y')
    >>> named_sharding = jax.sharding.NamedSharding(mesh, spec)
  """

  mesh: mesh_lib.Mesh
  spec: PartitionSpec
  _parsed_pspec: ParsedPartitionSpec

  @use_cpp_method()
  def __init__(
      self, mesh: mesh_lib.Mesh, spec: PartitionSpec, _parsed_pspec = None):

    self.mesh = mesh
    self.spec = spec
    self._parsed_pspec = _parsed_pspec
    self._preprocess()

  def __reduce__(self):
    return type(self), (self.mesh, self.spec)

  def _preprocess(self):
    # This split exists because you can pass `_parsed_pspec` that has been
    # modified from the original. For example: Adding extra dimension to
    # axis_resources for vmap handlers. In such cases you need to preserve the
    # `sync` attribute of parsed pspecs.
    # PartitionSpec is inferred from the parsed pspec in this case.
    # TODO(yaskatariya): Remove this and replace this with a normalized
    # representation of Parsed Pspec
    if self._parsed_pspec is None:
      self._parsed_pspec, _, _ = prepare_axis_resources(
          self.spec, "NamedSharding spec", allow_unconstrained_dims=True)

    _check_mesh_resource_axis(self.mesh, self._parsed_pspec)

  def __repr__(self):
    return f'NamedSharding(mesh={dict(self.mesh.shape)}, spec={self.spec})'

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self.mesh, self._parsed_pspec))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, NamedSharding):
      return False
    if id(self) == id(other):
      return True
    parsed_pspec_equal = self._parsed_pspec == other._parsed_pspec
    if id(self.mesh) == id(other.mesh) and parsed_pspec_equal:
      return True
    return self.mesh == other.mesh and parsed_pspec_equal

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
  def _from_parsed_pspec(cls, mesh, parsed_pspec):
    return cls(mesh, parsed_pspec.get_partition_spec(), parsed_pspec)

  @property
  def device_set(self) -> Set[Device]:
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
  def addressable_devices(self) -> Set[Device]:
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

  @functools.lru_cache(maxsize=4096)
  def _to_xla_op_sharding(
      self,
      num_dimensions: int,
      axis_ctx: Optional[Union[SPMDAxisContext, ShardingContext]] = None
  ) -> xc.OpSharding:
    assert self._parsed_pspec is not None
    array_mapping = get_array_mapping(self._parsed_pspec)
    # TODO(yashkatariya): Move away from sharding spec in NamedSharding
    # since we don't really need sharding spec.
    sharding_spec = sharding_specs.new_mesh_sharding_specs(
        self.mesh.shape, self.mesh.axis_names)(num_dimensions, array_mapping)
    # Used in `with_sharding_constraint`.
    special_axes = {}
    # Manual axes is only used with xmap.
    if axis_ctx is not None and isinstance(axis_ctx, SPMDAxisContext):
      axis_names = self.mesh.axis_names
      # Ignore type because mypy doesn't recognize the `hasattr` check above.
      for manual_axis in axis_ctx.manual_axes:  # type: ignore
        special_axes[axis_names.index(manual_axis)] = xc.OpSharding.Type.MANUAL
    return sharding_spec.sharding_proto(special_axes=special_axes)


@functools.lru_cache()
def get_replicated_op_sharding():
  proto = xc.OpSharding()
  proto.type = xc.OpSharding.Type.REPLICATED
  return proto


@use_cpp_class(xc.SingleDeviceSharding)
class SingleDeviceSharding(XLACompatibleSharding):
  """A subclass of ``XLACompatibleSharding`` that places its data on a single device.

  Args:
    device: A single :py:class:`Device`.

  Example:

    >>> single_device_sharding = jax.sharding.SingleDeviceSharding(
    ...     jax.devices()[0])
  """

  _device: Device

  @use_cpp_method()
  def __init__(self, device: Device):
    self._device = device

  def __reduce__(self):
    return type(self), (self._device,)

  def __repr__(self):
    return f"SingleDeviceSharding(device={repr(self._device)})"

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(self._device)
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, SingleDeviceSharding):
      return False
    if id(self) == id(other):
      return True
    return self._device == other._device

  @property
  def device_set(self) -> Set[Device]:
    return {self._device}

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:  # type: ignore
    return {self._device: (slice(None),) * len(global_shape)}

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return (self._device,)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    return get_replicated_op_sharding()

  @property
  def is_fully_replicated(self) -> bool:
    return True


@use_cpp_class(xc.PmapSharding)
class PmapSharding(XLACompatibleSharding):
  devices: np.ndarray
  sharding_spec: sharding_specs.ShardingSpec

  @use_cpp_method()
  def __init__(self, devices: Union[Sequence[Device], np.ndarray],
               sharding_spec: sharding_specs.ShardingSpec):
    self.devices = np.asarray(devices)
    # The sharding spec should be pmap's sharding spec.
    self.sharding_spec = sharding_spec

  def __reduce__(self):
    return type(self), (self.devices, self.sharding_spec)

  def __eq__(self, other):
    if not isinstance(other, PmapSharding):
      return False
    if id(self) == id(other):
      return True
    return (self.sharding_spec == other.sharding_spec and
            np.array_equal(self.devices, other.devices))

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((tuple(self.devices.flat), self.sharding_spec))
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
              devices: Optional[Sequence[xc.Device]] = None) -> PmapSharding:
    """Creates a `PmapSharding` which matches the implicit device order used by
    `pmap` if devices is None. If devices is specified, it will use those
    devices.

    Args:
      shape: The shape of the input array.
      sharded_dim: Dimension the input array is sharded on. Defaults to 0.
      devices: Optional sequence of devices used to create PmapSharding. If not
        specified, it will use the implicit device order used by pmap which is
        the order of jax.local_devices()
    """
    # The dtype doesn't matter here. Its only used for creating the
    # sharding_spec.
    sharding_spec = sharding_specs.create_pmap_sharding_spec(
        tuple(shape), sharded_dim)

    num_ways_sharded = None
    for s in sharding_spec.sharding:
      if isinstance(s, sharding_specs.Unstacked):
        num_ways_sharded = s.size
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
  def device_set(self) -> Set[Device]:
    return set(self.devices.flat)

  @functools.lru_cache(maxsize=4096)
  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    self.shard_shape(global_shape)  # raises a good error message
    indices = sharding_specs.spec_to_indices(global_shape, self.sharding_spec)
    return dict(safe_zip(self.devices.flat, indices))  # type: ignore[arg-type]

  @functools.cached_property
  def _device_assignment(self) -> XLADeviceAssignment:
    return tuple(self.devices.flat)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError("pmap doesn't use OpSharding.")

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    for s in self.sharding_spec.sharding:
      if isinstance(s, sharding_specs.Unstacked):
        return False
    return True

  @functools.lru_cache(maxsize=4096)
  def shard_shape(self, global_shape: Shape) -> Shape:
    sharded_dim = None
    sharded_dim_size = None
    for i, s in enumerate(self.sharding_spec.sharding):
      if isinstance(s, sharding_specs.Unstacked):
        sharded_dim = i
        sharded_dim_size = s.size
        break
    if sharded_dim is None:
      return global_shape
    if global_shape[sharded_dim] != sharded_dim_size:
      raise ValueError(
          f'The sharded dimension must be equal to the number of '
          f'devices passed to PmapSharding. Got sharded dimension {sharded_dim} '
          f'with value {global_shape[sharded_dim]} in shape {global_shape} and '
          f'the number of devices={len(self._device_assignment)}')
    return global_shape[:sharded_dim] + global_shape[sharded_dim+1:]


def _from_op_sharding_to_pos_sharding(
    op_sharding: xc.OpSharding,
    device_assignment: Sequence[xc.Device]) -> PositionalSharding:
  if op_sharding.type == xc.OpSharding.Type.REPLICATED:
    return PositionalSharding(device_assignment).replicate()

  if op_sharding.last_tile_dims == [xc.OpSharding.Type.REPLICATED]:
    replicate_on_last_tile_dim = True
  else:
    replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim
    if op_sharding.last_tile_dims:
      raise NotImplementedError(
          "Unhandled OpSharding type. Please open a bug report!")

  name = device_assignment[0].platform.upper()
  if op_sharding.tile_assignment_devices:
    tad = op_sharding.tile_assignment_devices
  else:
    tad = (
        np.arange(math.prod(op_sharding.tile_assignment_dimensions))
        .reshape(op_sharding.iota_reshape_dims)
        .transpose(op_sharding.iota_transpose_perm)
        .flat
    )
  ids = np.array([DeviceIdSet(name, i) for i in tad])
  p = PositionalSharding.remake(tuple(device_assignment), ids)
  p = p.reshape(op_sharding.tile_assignment_dimensions)
  if replicate_on_last_tile_dim:
    p = p.replicate(-1, keepdims=False)
  return p


class PositionalSharding(XLACompatibleSharding):
  _devices: Tuple[xc.Device, ...]
  _ids: np.ndarray  # dtype DeviceIdSet

  def __init__(self, devices: Union[Sequence[xc.Device], np.ndarray]):
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices, dtype='object')
    if not devices.size:
      raise ValueError(f"{self.__class__.__name__}.__init__ requires at least "
                       f"one device, got {devices}")
    self._devices = tuple(devices.flat)
    name = self._devices[0].platform.upper()
    self._ids = np.array([DeviceIdSet(name, i) for i in range(devices.size)],
                         dtype='object').reshape(devices.shape)

  shape = property(op.attrgetter('_ids.shape'))
  ndim = property(op.attrgetter('_ids.ndim'))

  def __repr__(self) -> str:
    cls_name = self.__class__.__name__
    ids = self._ids.copy()
    platform_name = self._devices[0].platform.upper()
    for idx, x in np.ndenumerate(ids):
      ids[idx] = DeviceIdSet(platform_name, *(self._devices[i].id for i in x))
    body = np.array2string(ids, prefix=cls_name + '(', suffix=')',
                           max_line_width=100)
    return f'{cls_name}({body})'

  def reshape(self, *shape):
    return self.remake(self._devices, self._ids.reshape(*shape))

  def transpose(self, *axes):
    return self.remake(self._devices, self._ids.transpose(*axes))
  T = property(transpose)

  def replicate(self, axis=None, keepdims=True):
    new_ids = self._ids.sum(axis=axis, keepdims=keepdims)  # union
    return self.remake(self._devices, new_ids)

  @classmethod
  def remake(
      cls, devices: Tuple[xc.Device, ...], ids: np.ndarray) -> PositionalSharding:
    self = cls.__new__(cls)
    self._devices = devices
    self._ids = ids
    return self

  # Hashable

  def __hash__(self) -> int:
    if not hasattr(self, '_hash'):
      self._hash = hash(self._devices)
    return self._hash

  def __eq__(self, other) -> bool:
    if not isinstance(other, PositionalSharding):
      return False
    if id(self) == id(other):
      return True
    all_ids_equal = np.array_equal(self._ids,other._ids)
    if id(self._devices) == id(other._devices) and all_ids_equal:
      return True
    return self._devices == other._devices and all_ids_equal

  # Sharding interface

  @functools.cached_property
  def device_set(self) -> set[xc.Device]:
    return set(self._devices)

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    return self.shape == (1,) * self.ndim

  # XLACompatibleSharding interface

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return self._devices

  @functools.lru_cache(maxsize=4096)
  def _to_xla_op_sharding(self, num_dimensions: int, axis_ctx=None):
    assert axis_ctx is None

    pbuf = xc.OpSharding()
    if self.shape == (1,) * self.ndim:
      pbuf.type = xc.OpSharding.Type.REPLICATED
      return pbuf

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
    return pbuf


class DeviceIdSet:
  _name: str
  _ids: FrozenSet[int]
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


@use_cpp_class(xc.GSPMDSharding)
class GSPMDSharding(XLACompatibleSharding):
  _devices: Tuple[Device, ...]
  _op_sharding: xc.OpSharding

  @use_cpp_method()
  def __init__(self, devices: Sequence[Device], op_sharding: xc.OpSharding):
    self._devices = tuple(devices)
    self._op_sharding = op_sharding

  def __reduce__(self):
    return type(self), (self._devices, self._op_sharding)

  @functools.cached_property
  def _op_sharding_hash(self):
    return hash(xc.HloSharding.from_proto(self._op_sharding))

  def __eq__(self, other):
    if not isinstance(other, GSPMDSharding):
      return False
    if id(self) == id(other):
      return True
    return (are_op_shardings_equal(self._op_sharding, other._op_sharding)
            and self._devices == other._devices)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._devices, self._op_sharding_hash))
    return self._hash

  def __repr__(self):
    return f'GSPMDSharding({repr(xc.HloSharding.from_proto(self._op_sharding))})'

  def is_compatible_aval(self, aval_shape: Shape):
    num_ways_dim_sharded, _ = get_num_ways_dim_sharded(
        self._op_sharding)
    if len(aval_shape) < len(num_ways_dim_sharded):
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(num_ways_dim_sharded)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @functools.cached_property
  def device_set(self) -> Set[Device]:
    return set(self._devices)

  @functools.lru_cache(maxsize=4096)
  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    self.shard_shape(global_shape)  # raises a good error message
    indices = op_sharding_to_indices(self._op_sharding, global_shape,
                                     len(self._devices))
    return dict(safe_zip(self._devices, indices))

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return self._devices

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    return self._op_sharding

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    return is_op_sharding_replicated(self._op_sharding)

  @classmethod
  def get_replicated(cls, device_assignment):
    proto = get_replicated_op_sharding()
    return cls(tuple(device_assignment), proto)


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
ArrayMapping = OrderedDictType[MeshAxisName, int]
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
    axis_resources: Union[ParsedPartitionSpec, AUTO, UnspecifiedValue]
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


def check_all_or_none_unspecified(axis_resources, name):
  if not axis_resources:
    return False
  unspecified_count = 0
  unspecified = is_unspecified(axis_resources[0])
  for resource in axis_resources:
    current_is_unspecified = is_unspecified(resource)
    if current_is_unspecified:
      unspecified_count += 1
      assert unspecified_count == 1
    if current_is_unspecified != unspecified:
      raise ValueError(f'`pjit.UNSPECIFIED` exists in {name}. '
                       f'Make sure that every entry in {name} is '
                       '`pjit.UNSPECIFIED`.')
  return unspecified


def prepare_axis_resources(axis_resources,
                           arg_name,
                           allow_unconstrained_dims=False):
  # PyTrees don't treat None values as leaves, so we use an is_leaf function.
  entries, treedef = tree_util.tree_flatten(
      axis_resources, is_leaf=lambda x: x is None)
  what = f"{arg_name} leaf specifications"
  # All entries should be specified or if unspecified then there should only
  # be 1 entry for that since UNSPECIFIED is a private API.
  check_all_or_none_unspecified(entries, arg_name)

  new_entries = []
  for entry in entries:
    if is_unspecified_or_auto(entry):
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
  names: Tuple[Any, ...]
  sizes: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SPMDAxisContext:
  """A hardware axis context for parallel computations that use the GSPMD partitioner.

  This includes the mesh that will later by used to execute this computation,
  as well as a set of mesh axes that are currently (e.g. because the current lowering
  is invoked inside an xmap) lowered in the MANUAL sharding mode.
  """
  mesh: mesh_lib.Mesh
  manual_axes: FrozenSet[MeshAxisName] = frozenset()

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

  def extend_manual(self, axes: FrozenSet[MeshAxisName]) -> SPMDAxisContext:
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
  device_assignment: Sequence[xc.Device]

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

def parse_flatten_op_sharding(op_sharding: xc.OpSharding,
                              mesh: mesh_lib.Mesh) -> Sequence[ParsedPartitionSpec]:
  if op_sharding.type == xc.OpSharding.Type.TUPLE:
    out: List[ParsedPartitionSpec] = []
    for s in op_sharding.tuple_shardings:
      out.extend(parse_flatten_op_sharding(s, mesh))
    return out
  elif op_sharding.type == xc.OpSharding.Type.REPLICATED:
    return [CanonicalizedParsedPartitionSpec(
        ParsedPartitionSpec(PartitionSpec(), ()))]
  elif op_sharding.type == xc.OpSharding.Type.OTHER:
    mesh_shape = mesh.shape
    if op_sharding.tile_assignment_devices:
      tad = op_sharding.tile_assignment_devices
    else:
      tad = (
          np.arange(math.prod(op_sharding.tile_assignment_dimensions))
          .reshape(op_sharding.iota_reshape_dims)
          .transpose(op_sharding.iota_transpose_perm)
          .flat
      )
    mesh_axis_order = unflatten_array(mesh.shape, tad)
    mesh_axis = iter(mesh_axis_order)
    shape = op_sharding.tile_assignment_dimensions
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
    if op_sharding.last_tile_dims == [xc.OpSharding.Type.REPLICATED]:
      replicate_on_last_tile_dim = True
    else:
      replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim
      if op_sharding.last_tile_dims:
        raise NotImplementedError("Unhandled OpSharding type. Please open a bug report!")
    if replicate_on_last_tile_dim:
      partitions = partitions[:-1]
    return [CanonicalizedParsedPartitionSpec(
        ParsedPartitionSpec('<internally generated spec>', partitions))]
  else:
    raise AssertionError("Unhandled OpSharding type. Please open a bug report!")
