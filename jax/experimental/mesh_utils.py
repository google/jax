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
from collections.abc import Sequence
import itertools
import logging
from typing import Any, Callable, Optional

import numpy as np
from jax._src import xla_bridge as xb

logger = logging.getLogger(__name__)

_TPU_V2 = 'TPU v2'
_TPU_V3 = 'TPU v3'
_TPU_V4 = 'TPU v4'

# Maps physical topology -> mesh shape -> transpose to use for jekbradbury's
# famous contiguous mesh trick.
#
# The trick only works for certain topologies and mesh shapes. Trivial dims of
# size 1 can be added to the shapes listed, and they are also supported.
_TRANSPOSE_TRICKS: dict[tuple[int, ...],
                        dict[tuple[int, ...], tuple[int, ...]]] = {
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


def _tpu_v2_v3_create_device_mesh(
    mesh_shape: Sequence[int],
    devices: Sequence[Any],
    **unused_kwargs,
) -> np.ndarray:
  if len(devices) == 8:
    logger.info(
        'Reordering mesh to physical ring order on single-tray TPU v2/v3.'
    )
    device_mesh = np.asarray(devices)
    device_mesh = device_mesh[np.array(_TRAY_RING_ORDER)]
    device_mesh = device_mesh.reshape(mesh_shape)
    return device_mesh
  elif mesh_shape[-1] == 8:
    device_mesh = np.asarray(devices).reshape(mesh_shape)
    logger.info(
        'Reordering mesh to physical ring order on each TPU v2/v3 tray.'
    )
    perm = np.array(_TRAY_RING_ORDER)
    device_mesh = device_mesh[..., perm]
    return device_mesh
  else:
    # TODO(skye): implement 2D mesh_shape logic here:
    # https://github.com/tensorflow/lingvo/blob/0df40cf604dfcd14e28f7087d73687a0bd2fe5c6/lingvo/core/gshard_utils.py#L187
    # (possibly replaces above mesh_shape[-1] == 8 case)
    return np.asarray(devices).reshape(mesh_shape)


# Registers functions to create device mesh for specific device kinds. Takes
# precedence over the more general logic in create_device_mesh(). Handler may
# return None; in that case, it will fall back to using the default logic.
device_kind_handler_dict: dict[
    str,
    Callable[..., Optional[np.ndarray]],
] = {
    _TPU_V2: _tpu_v2_v3_create_device_mesh,
    _TPU_V3: _tpu_v2_v3_create_device_mesh,
}


def _create_device_mesh_for_nd_torus(
    physical_mesh: np.ndarray, mesh_shape: Sequence[int],
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
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

  Returns:
    An np.ndarray of devices in the shape of the logical mesh (mesh_shape), with
      each logical parallelism axis mapped to one or more physical mesh axes.
    The axis assignment (a list of length num_logical_axes, whose elements
      are tuples representing physical axis indices).
  """
  # Remaining physical axes to be assigned to logical axes.
  assignable_physical_mesh = list(physical_mesh.shape)
  # Map each logical axis to a subset of physical axes.
  assignment: list[tuple[int, ...]] = [() for _ in mesh_shape]

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
        if np.prod(c_axes) == logical_axis_size:
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
  transpose: list[int] = []
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
    jax_devices: A list of JAX devices in a TPU slice, e.g. from jax.devices().

  Returns:
    A np.ndarray of JAX devices with shape (X, Y, Z, cores). On some platforms
      z and/or cores may be 1.
  """
  device_coords = [d.coords + (d.core_per_chip,) for d in jax_devices]
  dims = tuple(d + 1 for d in max(device_coords))
  out = np.empty(dims, dtype=object)
  for coords, d in zip(device_coords, jax_devices):
    out[coords] = d
  return out


# jekbradbury's famous trick for creating contiguous submeshes (where available)
def _transpose_trick(physical_mesh: np.ndarray,
                     mesh_shape: Sequence[int]) -> np.ndarray:
  nontrivial_mesh_shape = tuple(d for d in mesh_shape if d != 1)
  topology = tuple(d for d in physical_mesh.shape if d != 1)
  if topology not in _TRANSPOSE_TRICKS:
    raise ValueError(
        f"create_device_mesh cannot create contiguous submeshes for "
        f"physical mesh topology {topology}")

  if nontrivial_mesh_shape not in _TRANSPOSE_TRICKS[topology]:
    raise ValueError(
        f"create_device_mesh cannot create contiguous submeshes for "
        f"mesh_shape {mesh_shape} and physical mesh topology {topology}. "
        f"Available mesh_shapes: {list(_TRANSPOSE_TRICKS[topology].keys())}")

  return physical_mesh.reshape(topology).transpose(
      *_TRANSPOSE_TRICKS[topology][nontrivial_mesh_shape])


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
      where each process's local devices form a contiguous submesh. A ValueError
      will be raised if this function can't produce a suitable mesh. This
      setting was sometimes necessary before the introduction of jax.Array to
      ensure non-ragged local arrays; if using jax.Arrays, it's better to keep
      this set to False.

  Raises:
    ValueError: if the number of devices doesn't equal the product of
      `mesh_shape`.

  Returns:
    A np.ndarray of JAX devices with mesh_shape as its shape that can be fed
    into jax.sharding.Mesh with good collective performance.
  """
  if devices is None:
    devices = xb.devices()
  if np.prod(mesh_shape) != len(devices):
    raise ValueError(f'Number of devices {len(devices)} must equal the product '
                     f'of mesh_shape {mesh_shape}')
  nontrivial_mesh_shape = tuple(d for d in mesh_shape if d != 1)
  a_device = devices[0]
  if a_device.platform == 'tpu':
    physical_mesh = _get_physical_tpu_mesh(devices)
    X, Y, Z, cores = physical_mesh.shape
    if Z > 1:
      subcube = X < 4 or Y < 4 or Z < 4
      wrap = tuple(size >= 4 and not subcube for size in (X, Y, Z))
    else:
      if a_device.device_kind == _TPU_V3:
        wrap = tuple(size == 32 for size in (X, Y)) + (False,)
      else:
        wrap = tuple(size == 16 for size in (X, Y)) + (False,)
    # assign includes stacking the results of a recursive call
    # any assign can also have cores on both sides
    # if cores == 2 and there's a size 2 laxis, assign
    # if there's a laxis of the same size as multiple paxes with wrap, assign
    #  (preferring zx or zy for host contig)
    # if there's a laxis of the same size as a paxis with wrap, assign
    # if there's a laxis of the same size as two paxes without wrap, assign as two vrings if >32 or one ring if <=32
    # if there's a laxis of the same size as a paxis without wrap, assign as a vring
    # all remaining cases should be three paxes without wrap; prefer snake
    def _create_device_mesh(physical_mesh, wrap, mesh_shape):
      # here mesh_shape is ordered with decreasing network intensity
      X, Y, Z, cores = physical_mesh.shape
      Xw, Yw, Zw = wrap
      # first, try assigning an axis to cores
      if cores > 1 and cores in mesh_shape:
        axis_index = mesh_shape.index(2)
        mesh_shape = (mesh_shape[:axis_index] + mesh_shape[axis_index + 1:])
        return np.stack(
            [_create_device_mesh(physical_mesh[:,:,:,core], wrap, mesh_shape)
             for core in range(cores)], axis_index)
      # assign axis by axis, preferring physical submeshes in this order:
      # 1. cores alone
      # 2. one or more torus axes plus cores
      # 3. one or more torus axes
      # 4. preference to pairs of axes
      #    where the assignment can be made host-contiguous
      # - assigning an axis to one or more torus axes plus cores
      # - assigning an axis to that can be made host-contiguous
      
      # should we prefer three axes without cores or two with
      # and two axes without cores or one with
      # oh absolutely N+1 without is better than N with
      def num_links(submesh):
        return sum(1 + wrap[i] for i in submesh)
      submeshes = [()] + sorted(
          [(0,), (1,), (2,), (2, 1), (2, 0), (0, 1), (0, 1, 2)],
          key=num_links, reverse=True)
      for physical_axis_indices in submeshes:
        if any(physical_mesh.shape[i] == 1 for i in physical_axis_indices):
          continue
        size = np.prod((physical_mesh.shape[i] for i in physical_axis_indices),
                       dtype=np.int32)
        def iterate_through_submesh():
          if any(wrap[i] for i in physical_axis_indices) or size > 32:
            for submesh_indices in np.ndindex(tuple(
                physical_mesh.shape[i] for i in physical_axis_indices)):
              physical_indices = [slice() for _ in physical_mesh.shape]
              for i, ind, w in zip(
                  physical_axis_indices, submesh_indices, wrap):
                physical_indices[i] = (ind if w else _permute_for_virtual_wrap(
                    physical_mesh.shape[i], ind))
              yield tuple(physical_indices)
          else:
            # spaaaaace filling curve
            # the whole thing should return indices and not devices!
        if cores > 1 and size * cores in mesh_shape:
          axis_index = mesh_shape.index(size * cores)
          mesh_shape = (mesh_shape[:axis_index] + mesh_shape[axis_index + 1:])
          
          return np.stack([_create_device_mesh(
              physical_mesh[index + (core,)], wrap, mesh_shape)
                           for index in iterate_through_submesh() for core in range(cores)],
                          axis_index)
        if size in mesh_shape:
          axis_index = mesh_shape.index(size)
          mesh_shape = mesh_shape[:axis_index] + mesh_shape[axis_index + 1:]
          return np.stack(
              [_create_device_mesh(physical_mesh[index], wrap, mesh_shape)
               for index in iterate_through_submesh()], axis_index)
        
    result needs to be transposed!
            
    if not all(wrap):
      # so far, seems like:
      # cores should be taken by 2 if available, otherwise can attach to any axis
      # take any rings
      # take 
      # possible decompositions
      # one wrap axis to ring (2/2)
      # two nowrap axes to two virtual rings (2,2) <-- better for two axes or one >32 one
      # two nowrap axes to ring (2,2) <-- better for one <=32 one
      # one nowrap axis to virtual ring (1/1)
      # three nowrap axes to ring (2/3)
      # 2,2,1,1 and 4,: ring
      # 2,2,1,2 and 8,: ring
      # 2,2,1,2 and 2,4 or 4,2: ring and cores
      # 2,2,1,1 and 2,2: 2 and 2
      # 2,2,1,2 and 2,2,2: 2 and cores
      # 2,2,2,1 and 8: EITHER snake (2 links) OR 2,2,2 (3 links) <- prefer snake since allreduce will reorder
      # 2,2,2,1 and 2,4 or 4,2: ring and 2
      # 2,2,2,2 and 16: snake
      # 2,2,2,2 and 8,2: snake and cores
      # 2,2,2,2 and 4,4: ring and 2
      # 2,2,2,2 and 4,2,2: ring, 2, cores
      # 2,2,2,2 and 2,2,2,2: 2,2,2
      # 2,2,4,1 and 
      # 2,4,4,1
      # 4,2,1,1
      # 4,2,1,2
      # 4,4,1,1
      # 4,4,1,2
      # 4,8,1,1
      # 4,8,1,2
      # 8,8,1,1
      # 8,8,1,2
      # 8,16,1,1
      # 8,16,1,2
      if len(nontrivial_mesh_shape) == 1:
        # space filling curve
      else:
        # create virtual wrap
    if contiguous_submeshes:
      physical_mesh = _transpose_trick(physical_mesh, mesh_shape)
    device_mesh, assignment = _create_device_mesh_for_nd_torus(
      physical_mesh, mesh_shape)
    logger.info('_create_device_mesh_for_nd_torus assignment: %s', assignment)
    return device_mesh
        
    if not all(wrap):
      if len(nontrivial_mesh_shape) == 1:
        # use a ring over wrapped axes and a space-filling curve over others
        # no
        # try to assign (groups of) wrapped axes to mesh axes
        # 
    if X == Y == 2 and Z == 1 and len(nontrivial_mesh_shape) == 1:
      # use a physical ring order on a single tray
      device_mesh = [physical_mesh[0, 0], physical_mesh[0, 1],
                     physical_mesh[1, 1], physical_mesh[1, 0]]
    
  if a_device.device_kind in (_TPU_V2, _TPU_V3):
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

def create_hybrid_device_mesh(mesh_shape: Sequence[int],
                              dcn_mesh_shape: Sequence[int],
                              devices: Optional[Sequence[Any]] = None, *,
                              process_is_granule: bool = False) -> np.ndarray:
  """Creates a device mesh for hybrid (e.g., ICI and DCN) parallelism.

  Args:
    mesh_shape: shape of the logical mesh for the faster network, ordered by
      increasing network intensity, e.g. [replica, data, mdl] where mdl has the
      most network communication requirements.
    dcn_mesh_shape: shape of the logical mesh for the slower network, in the
      same order as mesh_shape.
    devices: optionally, the devices to construct a mesh for. Defaults to
      jax.devices().
    process_is_granule: if True, this function will treat processes as the units
      of the slower/outer network. Otherwise it will look for slice_index
      attributes on devices and use slices as the units. Enabling this is meant
      as a fallback for platforms (e.g., GPU) that don't set slice_index.

  Raises:
    ValueError: if the number of slices to which the `devices` belong doesn't
      equal the product of `dcn_mesh_shape`, or if the number of devices
      belonging to any single slice does not equal the product of `mesh_shape`.

  Returns:
    A np.ndarray of JAX devices with mesh_shape * dcn_mesh_shape as its shape
    that can be fed into jax.sharding.Mesh for hybrid parallelism. Any axis that
    is placed on both the faster and slower network will have the faster network
    "major" and the slower one "minor" in order to maximize data contiguity for
    hierarchical collective algorithms that do slow-network communication first. 
  """
  if devices is None:
    devices = xb.devices()
  attr = 'process_index' if process_is_granule else 'slice_index'
  assert hasattr(devices[0], attr)
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = [granule_dict[key] for key in sorted(granule_dict.keys())]
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(
        f'Number of slices {len(granules)} must equal the product of '
        f'dcn_mesh_shape {dcn_mesh_shape}')
  per_granule_meshes = [create_device_mesh(mesh_shape, granule)
                        for granule in granules]
  # TODO(jekbradbury): handle non-uniform DCN topologies
  granule_indices = np.arange(len(granules)).reshape(dcn_mesh_shape)
  indices_in_granule = np.fromiter(np.ndindex(mesh_shape), dtype=object,
                                   count=len(granules[0])).reshape(mesh_shape)
  blocks = np.vectorize(
      lambda index_in_granule: np.vectorize(
          lambda granule_index: per_granule_meshes[granule_index][
              index_in_granule], otypes=[object])(
                  granule_indices), otypes=[object])(indices_in_granule)
  device_mesh = np.block(blocks.tolist())
  return device_mesh
