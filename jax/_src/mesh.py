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
"""Definitions of Mesh and ResourceEnv."""

from __future__ import annotations

import collections
from collections.abc import Hashable, Sequence
import contextlib
import functools
import math
import threading
from typing import Any, NamedTuple

import numpy as np

from jax._src import config as jax_config
from jax._src import xla_bridge as xb
from jax._src import util
from jax._src.lib import xla_client as xc


MeshAxisName = Any
ResourceAxisName = Hashable

class Loop(NamedTuple):
  name: ResourceAxisName
  length: int


def show_axes(axes):
  return ", ".join(sorted(f"`{a}`" for a in axes))


class ResourceEnv(NamedTuple):
  physical_mesh: Mesh
  loops: tuple[Loop, ...]

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
  def physical_resource_axes(self) -> set[ResourceAxisName]:
    return set(self.physical_mesh.axis_names)

  @property
  def loop_resource_axes(self) -> set[ResourceAxisName]:
    return {loop.name for loop in self.loops}

  @property
  def resource_axes(self) -> set[ResourceAxisName]:
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
    mesh_repr = ", ".join(
        f"'{k}': {v}" for k, v in self.physical_mesh.shape.items())
    return f"ResourceEnv(mesh=Mesh({mesh_repr}), {self.loops!r})"


@util.cache(max_size=128, trace_context_in_key=False)
def _get_local_mesh(global_mesh: Mesh, process_index: int) -> Mesh:
  if global_mesh.empty:
      return global_mesh
  is_local_device = np.vectorize(
      lambda d: d.process_index == process_index, otypes=[bool])(global_mesh.devices)
  subcube_indices = []
  # We take the smallest slice of each dimension that doesn't skip any local device.
  for axis in range(global_mesh.devices.ndim):
    other_axes = util.tuple_delete(tuple(range(global_mesh.devices.ndim)), axis)
    # NOTE: This re-reduces over many axes multiple times, so we could definitely
    #       optimize it, but I hope it won't be a bottleneck anytime soon.
    local_slices = is_local_device.any(other_axes, keepdims=False)
    nonzero_indices = np.flatnonzero(local_slices)
    start, end = int(np.min(nonzero_indices)), int(np.max(nonzero_indices))
    subcube_indices.append(slice(start, end + 1))
  subcube_indices_tuple = tuple(subcube_indices)
  # We only end up with all conditions being true if the local devices formed a
  # subcube of the full array. This is because we were biased towards taking a
  # "hull" spanned by the devices, and in case the local devices don't form a
  # subcube that hull will contain non-local devices.
  if not is_local_device[subcube_indices_tuple].all():
    raise ValueError(
        "When passing host local inputs to pjit or xmap, devices "
        "connected to a single host must form a contiguous subcube of the "
        "global device mesh")
  return Mesh(global_mesh.devices[subcube_indices_tuple], global_mesh.axis_names)


_mesh_object_dict = {}  # type: ignore


class Mesh(contextlib.ContextDecorator):
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

  Examples:

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
    ...   out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(inp)

    >>> # Initialize the Mesh and use the mesh as the context manager.
    >>> with Mesh(devices, ('x', 'y')) as global_mesh:
    ...   out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(inp)

    >>> # Also you can use it as `with ... as ...`.
    >>> global_mesh = Mesh(devices, ('x', 'y'))
    >>> with global_mesh as m:
    ...   out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(inp)

    >>> # You can also use it as `with Mesh(...)`.
    >>> with Mesh(devices, ('x', 'y')):
    ...   out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(inp)
  """

  devices: np.ndarray
  axis_names: tuple[MeshAxisName, ...]

  def __new__(cls, devices: np.ndarray | Sequence[xc.Device],
              axis_names: str | Sequence[MeshAxisName]):
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices)
    if isinstance(axis_names, str):
      axis_names = (axis_names,)
    axis_names = tuple(axis_names)

    if devices.ndim != len(axis_names):
      raise ValueError(
          "Mesh requires the ndim of its first argument (`devices`) to equal "
          "the length of its second argument (`axis_names`), but got "
          f"devices.ndim == {devices.ndim} and "
          f"len(axis_names) == {len(axis_names)}.")

    key = (axis_names, devices.shape, tuple(devices.flat))
    val = _mesh_object_dict.get(key, None)
    if val is not None:
      return val

    self = super().__new__(cls)
    self.devices = devices.copy()
    self.devices.flags.writeable = False
    self.axis_names = axis_names
    _mesh_object_dict[key] = self
    return self

  def __reduce__(self):
    return (type(self), (self.devices, self.axis_names))

  def __eq__(self, other):
    if not isinstance(other, Mesh):
      return False
    # This is a performance optimization. Comparing thousands of devices
    # can be expensive.
    if id(self) == id(other):
      return True
    return (self.axis_names == other.axis_names and
            self.devices.shape == other.devices.shape and
            self._internal_device_list == other._internal_device_list)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(
          (self.axis_names, self._internal_device_list, self.devices.shape))
    return self._hash

  def __setattr__(self, name, value):
    if hasattr(self, name):
      if getattr(self, name) == value:
        # This can to happen if two threads race, for example if two threads
        # are trying to hash the same Mesh instance.
        return
      raise RuntimeError(
          f"Cannot reassign attributes ({name}) of immutable mesh objects"
      )
    super().__setattr__(name, value)

  def __enter__(self):
    new_env = thread_resources.stack[-1].with_mesh(self)
    thread_resources.stack.append(new_env)
    thread_resources.env = new_env
    jax_config.update_thread_local_jit_state(
        mesh_context_manager=tuple(t.physical_mesh for t in thread_resources.stack
                                   if not t.physical_mesh.empty))
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    thread_resources.stack.pop()
    thread_resources.env = thread_resources.stack[-1]
    jax_config.update_thread_local_jit_state(
        mesh_context_manager=tuple(t.physical_mesh for t in thread_resources.stack
                                   if not t.physical_mesh.empty))
    return False

  @property
  def shape(self):
    return collections.OrderedDict(
        (name, size)
        for name, size in util.safe_zip(self.axis_names, self.devices.shape))

  @property
  def size(self):
    return math.prod(self.shape.values())

  @property
  def empty(self):
    return self.devices.ndim == 0

  @functools.cached_property
  def is_multi_process(self):
    return self.devices.size != len(self.local_devices)

  @property
  def local_mesh(self):
    return self._local_mesh(xb.process_index())

  def _local_mesh(self, process_index):
    return _get_local_mesh(self, process_index)

  @functools.cached_property
  def device_ids(self):
    assert not self.empty
    return np.vectorize(lambda d: d.id, otypes=[int])(self.devices)

  @functools.cached_property
  def _local_devices_set(self):
    return set(self.local_devices)

  @functools.cached_property
  def _flat_devices_tuple(self):
    return tuple(self.devices.flat)

  @functools.cached_property
  def _internal_device_list(self):
    return xc.DeviceList(self._flat_devices_tuple)

  @functools.cached_property
  def _flat_devices_set(self):
    return set(self.devices.flat)

  def __str__(self):
    mesh_str = ", ".join(f"'{k}': {v}" for k, v in self.shape.items())
    return f"Mesh({mesh_str})"

  @functools.cached_property
  def _repr(self):
    if self.empty:
      return "Mesh(device_ids=[], axis_names=())"
    return f"Mesh(device_ids={self.device_ids!r}, axis_names={self.axis_names!r})"

  def __repr__(self):
    return self._repr

  @functools.cached_property
  def local_devices(self):
    return [d for d in self.devices.flat
            if d.process_index == d.client.process_index()]


EMPTY_ENV = ResourceEnv(Mesh(np.empty((), dtype=object), ()), ())

class _ThreadResourcesLocalState(threading.local):

  def __init__(self):
    self.stack = [EMPTY_ENV]
    self.env = self.stack[-1]

thread_resources = _ThreadResourcesLocalState()
