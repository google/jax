# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for building a device mesh."""

import collections
import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import numpy as np

logger = logging.getLogger(__name__)

_TPU_V2 = 'TPU v2'
_TPU_V3 = 'TPU v3'
_TPU_V4 = 'TPU v4'

# Maps physical topology -> mesh shape -> transpose to use for jekbradbury's
# famous contiguous mesh trick.
#
# The trick only works for certain topologies and mesh shapes. Trivial dims of
# size 1 can be added to the shapes listed, and they are also supported.
_TRANSPOSE_TRICKS: Dict[Tuple[int, ...],
                        Dict[Tuple[int, ...], Tuple[int, ...]]] = {
    (2, 2, 1): {
        (2, 2): (0, 1, 2),
    },
    (2, 2, 4): {
        (4, 4): (0, 1, 2),
    },
    (4, 4, 4): {
        (16, 4): (0, 2, 1),
    },
    (4, 8, 8): {
        (64, 4): (0, 2, 1),
        (4, 64): (0, 2, 1),
    },
    (8, 8, 8): {
        (64, 8): (0, 2, 1),
    },
    (8, 16, 16): {
        (256, 8): (0, 2, 1),
        (8, 256): (0, 2, 1),
    },
}

# Physical ordering of core IDs in a tray that creates a ring
_TRAY_RING_ORDER = (0, 1, 2, 3, 6, 7, 4, 5)


def _create_device_mesh_for_nd_torus(
    physical_mesh: np.ndarray, mesh_shape: Sequence[int],
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
  """Assigns logical parallelism axes to physical axes of an N-D torus network.

  Given logical parallelism axes with sizes in `mesh_shape` and devices in an
  N-dimensional torus network represented by `physical_mesh`, maps each logical
  axis to one or more physical axes. Prefer to map more-performance-sensitive
  logical axes to larger numbers of physical axes to maximize the bandwidth
  available to them. Also prefer to assign logical axes to multiple physical
  axes of the same size (e.g., a 2D square) rather than multiple physical axes
  of different sizes when possible.

  Note that this routine will never split a physical axis over more than one
  logical axis (which would reduce total usable bandwidth but may sometimes be
  desired anyway). As a result, it will error out in cases where this is
  necessary to produce a valid mapping.

  Let's use a concrete example to explain the concepts and considerations.

  As an example, suppose the logical mesh is [data, model], for data and model
  parallelism respectively. Also suppose that data parallelism is less
  performance sensitive than model parallelism. Consider a 3D TPU pod slice of
  shape 4x4x16, represented by a physical mesh of shape (4, 4, 16).

  A TPU pod slice has equal bandwidth along all axes with wraparound links, but
  a 2D plane of size 4x4 may have faster XLA collective implementations than a
  non-square plane or a 1D subgroup. If the mesh_shape is [16, 16], we may want
  the more performance sensitive `model` axis to be mapped to the 4x4 XY plane.

  Args:
    physical_mesh: a np.ndarray of devices in the shape of the N-D torus
      physical topology.
    mesh_shape: shape of the logical mesh (size of the various logical
      parallelism axes), with axes ordered by increasing network intensity.
    prefer_symmetric: whether to prefer to assign a logical axis to multiple
      physical axes of the same size rather than axes of different sizes.

  Returns:
    An np.ndarray of devices in the shape of the logical mesh (mesh_shape), with
      each logical parallelism axis mapped to one or more physical mesh axes.
    The axis assignment (a list of length num_logical_axes, whose elements
      are tuples representing physical axis indices).
  """
  # Remaining physical axes to be assigned to logical axes.
  assignable_physical_mesh = list(physical_mesh.shape)
  # Map each logical axis to a subset of physical axes.
  assignment: List[Tuple[int, ...]] = [() for _ in mesh_shape]

  # Assign logical axes from highest network intensity to lowest.
  # `mesh_shape` is assumed to ordered by lowest network intensity first, so
  # reverse it first.
  for logical_axis_index, logical_axis_size in reversed(
      list(enumerate(mesh_shape))):
    # Preferentially map to more physical axes first for higher bandwidth.
    for num_axes in range(3, 0, -1):
      # Try assign to any subset of size num_axes. Generate all candidates.
      axes = itertools.combinations(assignable_physical_mesh, num_axes)
      indices = itertools.combinations(
          range(len(assignable_physical_mesh)), num_axes)
      for c_axes, c_indices in zip(axes, indices):
        # TODO(zhangqiaorjc): Due to limitations in XLA, 2D collectives only
        # implemented for square 2D plane. Mapping a physical axis to two
        # logical axes might be slower for non-square 2D plane, e.g., map 32 to
        # 4x8 or a single axis. If XLA 2D collectives support non-square plane
        # soon, we can continue to preferentially map to 2D plane in general,
        # otherwise, we should treat non-square 2D plane and 1D submesh equally.
        if np.product(c_axes) == logical_axis_size:
          assignment[logical_axis_index] = c_indices
          # Zero the assigned physical axes.
          assignable_physical_mesh = [
              0 if i in c_indices else v
              for i, v in enumerate(assignable_physical_mesh)
          ]
          break
      if assignment[logical_axis_index]:
        # We already found an assignment from one candidate above.
        break
    else:
      # If the num_axes for loop did not break, i.e. none of the candidates work
      # goto here with this while-else construct.
      if logical_axis_size > 1:
        raise NotImplementedError(
            'Failed to find assignment for logical_axis_index'
            f' {logical_axis_index} of size {logical_axis_size} with remaining'
            f' assignable mesh {assignable_physical_mesh}. The size of each'
            ' axis in your logical mesh must be equal to the product of'
            ' some subset of the physical mesh axis sizes. E.g logical mesh (4,'
            ' 16) is compatible with physical mesh 4x4x4 since 4=4 and 16=4x4.'
        )
  # Flatten the assignment, e.g., [(), (2,), (0, 1)] -> (2, 0, 1).
  transpose: List[int] = []
  for x in assignment:
    for y in x:
      transpose.append(int(y))
  return physical_mesh.transpose(transpose).reshape(mesh_shape), assignment


def _bounds_from_last_device(last_device) -> Sequence[int]:
  """Gets the bound from the given last device."""
  # Must be passed the device at the highest-coordinate corner of the
  # relevant mesh, which is a requirement we know is satisfied by the last
  # device in jax.devices().
  assert hasattr(last_device, 'coords'), 'Only TPU supported'
  x, y, z = last_device.coords
  return x + 1, y + 1, z + 1, last_device.core_on_chip + 1


def _get_physical_tpu_mesh(jax_devices: Sequence[Any]) -> np.ndarray:
  r"""Rearrange TPU devices in a slice into a physical mesh.

  Args:
    jax_devices: A list of JAX devices in a TPU slice in process-tiled z, y, x,
      core order, e.g. from jax.devices().

  Returns:
    A np.ndarray of JAX devices with shape [global_x, global_y, global_z]. On
      v2 and v3, global_z is instead cores_per_chip (i.e., 2).
  """
  device_kind = jax_devices[0].device_kind
  device_coords = [d.coords for d in jax_devices]
  dims = tuple(d + 1 for d in max(device_coords))
  assert len(dims) == 3, dims
  if device_kind in (_TPU_V2, _TPU_V3):
    cores_per_chip = max(d.core_on_chip for d in jax_devices) + 1
    out = np.empty(dims[:2] + (cores_per_chip,), dtype=object)
    for coords, d in zip(device_coords, jax_devices):
      assert coords[2] == 0, d
      out[coords[0], coords[1], d.core_on_chip] = d
  else:
    out = np.empty(dims, dtype=object)
    for coords, d in zip(device_coords, jax_devices):
      if d.core_on_chip != 0:
        raise AssertionError(
            'Creating meshes for TPU >v3 requires one device per chip.'
            f'Got device id {d.core_on_chip} for a device of kind {device_kind}'
            f': {d}'
        )
      out[coords[0], coords[1], coords[2]] = d
  return out


# jekbradbury's famous trick for creating contiguous submeshes (where available)
def _transpose_trick(physical_mesh: np.ndarray,
                     mesh_shape: Sequence[int]) -> np.ndarray:
  mesh_shape = tuple(mesh_shape)
  topology = physical_mesh.shape
  if topology not in _TRANSPOSE_TRICKS:
    raise ValueError(
        f"create_device_mesh cannot create contiguous submeshes for "
        f"physical mesh topology {topology}")

  mesh_shape_no_trivial_dims: Tuple[int, ...] = ()
  for dim_size in mesh_shape:
    if dim_size != 1:
      mesh_shape_no_trivial_dims += (dim_size,)

  if mesh_shape_no_trivial_dims not in _TRANSPOSE_TRICKS[topology]:
    raise ValueError(
        f"create_device_mesh cannot create contiguous submeshes for "
        f"mesh_shape {mesh_shape} and physical mesh topology {topology}. "
        f"Available mesh_shapes: {list(_TRANSPOSE_TRICKS[topology].keys())}")

  return physical_mesh.transpose(
      *_TRANSPOSE_TRICKS[topology][mesh_shape_no_trivial_dims])


def create_device_mesh(
    mesh_shape: Sequence[int],
    devices: Optional[Sequence[Any]] = None, *,
    contiguous_submeshes: bool = False) -> np.ndarray:
  """Creates a performant device mesh for jax.sharding.Mesh.

  Args:
    mesh_shape: shape of logical mesh, ordered by increasing network-intensity
      e.g. [replica, data, mdl] where mdl has the most network communication
      requirements.
    devices: optionally, the devices to construct a mesh for. Defaults to
      jax.devices().
    contiguous_submeshes: if True, this function will attempt to create a mesh
      where each process's local devices form a contiguous submesh. This is
      required when passing host local inputs to `pjit`. A ValueError will be
      raised if this function can't produce a suitable mesh.

  Returns:
    A np.ndarray of JAX devices with mesh_shape as its shape that can be fed
    into jax.sharding.Mesh with good collective performance.
  """
  if devices is None:
    devices = jax.devices()
  if np.prod(mesh_shape) != len(devices):
    raise ValueError(f'Number of devices {len(devices)} must equal the product '
                     f'of mesh_shape {mesh_shape}')
  last_device = devices[-1]
  if last_device.device_kind in (_TPU_V2, _TPU_V3):
    if len(devices) == 8:
      logger.info('Reordering mesh to physical ring order on single-tray TPU v2/v3.')
      device_mesh = np.asarray(devices)
      device_mesh = device_mesh[np.array(_TRAY_RING_ORDER)]
      device_mesh = device_mesh.reshape(mesh_shape)
      return device_mesh
    elif mesh_shape[-1] == 8:
      device_mesh = np.asarray(devices).reshape(mesh_shape)
      logger.info('Reordering mesh to physical ring order on each TPU v2/v3 tray.')
      perm = np.array(_TRAY_RING_ORDER)
      device_mesh = device_mesh[..., perm]
      return device_mesh
    else:
      # TODO(skye): implement 2D mesh_shape logic here:
      # https://github.com/tensorflow/lingvo/blob/0df40cf604dfcd14e28f7087d73687a0bd2fe5c6/lingvo/core/gshard_utils.py#L187
      # (possibly replaces above mesh_shape[-1] == 8 case)
      return np.asarray(devices).reshape(mesh_shape)
  elif last_device.platform == 'tpu':
    physical_mesh = _get_physical_tpu_mesh(devices)
    if contiguous_submeshes:
      physical_mesh = _transpose_trick(physical_mesh, mesh_shape)
    device_mesh, assignment = _create_device_mesh_for_nd_torus(
        physical_mesh, mesh_shape)
    logger.info('_create_device_mesh_for_nd_torus assignment: %s', assignment)
    return device_mesh
  else:
    device_mesh = np.asarray(devices).reshape(mesh_shape)
    return device_mesh


def create_hybrid_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Optional[Sequence[Any]] = None,
    *,
    process_is_granule: bool = False,
    contiguous_submeshes: bool = False,
) -> np.ndarray:
  """Creates a device mesh for hybrid (e.g., ICI and DCN) parallelism.

  Args:
    mesh_shape: shape of the logical mesh for the faster/inner network, ordered
      by increasing network intensity, e.g. [replica, data, mdl] where mdl has
      the most network communication requirements.
    dcn_mesh_shape: shape of the logical mesh for the slower/outer network, in
      the same order as mesh_shape.
    devices: optionally, the devices to construct a mesh for. Defaults to
      jax.devices().
    process_is_granule: if True, this function will treat processes as the units
      of the slower/outer network. Otherwise it will look for slice_index
      attributes on devices and use slices as the units. Enabling this is meant
      as a fallback for platforms (e.g., GPU) that don't set slice_index.
    contiguous_submeshes: if True, this function will attempt to create a mesh
      where each process's local devices form a contiguous submesh. This is
      required when passing host local inputs to `pjit`. A ValueError will be
      raised if this function can't produce a suitable mesh.

  Returns:
    A np.ndarray of JAX devices with mesh_shape * dcn_mesh_shape as its shape
    that can be fed into jax.sharding.Mesh for hybrid parallelism.
  """
  if devices is None:
    devices = jax.devices()
  attr = 'process_index' if process_is_granule else 'slice_index'
  assert hasattr(devices[0], attr)
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = list(granule_dict[key] for key in sorted(granule_dict.keys()))
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(
        'Number of slices must equal the product of dcn_mesh_shape')
  per_granule_meshes = [
      create_device_mesh(
          mesh_shape, granule, contiguous_submeshes=contiguous_submeshes
      )
      for granule in granules
  ]
  # TODO(jekbradbury): handle non-uniform DCN topologies
  granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
  blocks = np.vectorize(
    lambda i: per_granule_meshes[i], otypes=[object])(granule_mesh)
  device_mesh = np.block(blocks.tolist())
  return device_mesh
