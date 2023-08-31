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
"""Tests for mesh utils."""

import collections
from collections.abc import Sequence
import dataclasses

import numpy as np

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax._src import test_util


@dataclasses.dataclass(frozen=True)
class MockTpuDevice:
  """Mock TPU device for testing."""
  id: int
  platform: str
  device_kind: str
  process_index: int
  coords: Sequence[int]
  core_on_chip: int
  slice_index: int = 0


def mock_tpu_devices(x, y, z, dev_kind, one_device_per_chip, num_slices=1,
                     reorder=False):
  """Produce fake jax.devices() output for a TPU slice."""
  cores_per_chip = 1 if one_device_per_chip else 2
  nxd, nyd, nzd = (2, 2, 1)
  nxp, nyp, nzp = x // nxd, y // nyd, z // nzd
  def mock_tpu_device(core_on_chip, xd, yd, zd, xp, yp, zp, slice_index):
    process_index = xp + nxp * (yp + nyp * (zp + nzp * slice_index))
    coords =  (xd + nxd * xp, yd + nyd * yp, zd + nzd * zp)
    device_id = core_on_chip + cores_per_chip * (xd + nxd * (xp + nxp * (
        yd + nyd * (yp + nyp * (zd + nzd * (zp + nzp * slice_index))))))
    return MockTpuDevice(device_id, 'tpu', dev_kind, process_index, coords,
                         core_on_chip, slice_index)
  devices = [mock_tpu_device(core_on_chip, xd, yd, zd, xp, yp, zp, slice_index)
             for slice_index in range(num_slices)
             for zp in range(nzp) for yp in range(nyp) for xp in range(nxp)
             for zd in range(nzd) for yd in range(nyd) for xd in range(nxd)
             for core_on_chip in range(cores_per_chip)]
  if reorder:
    devices = devices[::-1]
  _validate_mocked_process_indices(devices, one_device_per_chip)
  return devices

# If this function raises, it's a bug in the test code!
def _validate_mocked_process_indices(devices, one_device_per_chip):
  process_to_devices = collections.defaultdict(lambda: [])
  for d in devices:
    process_to_devices[d.process_index].append(d)

  for local_devices in process_to_devices.values():
    if one_device_per_chip:
      # 4 devices per process
      assert len(local_devices) == 4, local_devices
    else:
      # 8 devices per process
      assert len(local_devices) == 8, local_devices
    # All devices have same z coord
    assert len({d.coords[2] for d in local_devices}) == 1, local_devices
    # All devices in a 2x2 subgrid
    min_coords = min(d.coords for d in local_devices)
    expected = set()
    for x, y in [(0,0), (0,1), (1,0), (1,1)]:
      expected.add((min_coords[0] + x, min_coords[1] + y, min_coords[2]))
    assert {d.coords for d in local_devices} == expected, local_devices


def mock_2x2_devices():
  """Hard-coded reproduction of jax.devices() output on v3-2x2."""
  return mock_tpu_devices(2, 2, 1, 'TPU v3', False)


def mock_4x4_devices():
  """Hard-coded reproduction of jax.devices() output on v3-4x4."""
  return mock_tpu_devices(4, 4, 1, 'TPU v3', False)


def mock_8x8_devices(one_device_per_chip=False):
  """Hard-coded reproduction of jax.devices() output on v3-8x8."""
  return mock_tpu_devices(8, 8, 1, 'TPU v3', one_device_per_chip)


def mock_2x2x1_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 2x2x1."""
  return mock_tpu_devices(2, 2, 1, 'TPU v4', one_device_per_chip)


def mock_2x2x4_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 2x2x4."""
  return mock_tpu_devices(2, 2, 4, 'TPU v4', one_device_per_chip)


def mock_4x4x4_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 4x4x4."""
  return mock_tpu_devices(4, 4, 4, 'TPU v4', one_device_per_chip)


def mock_4x4x8_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 4x4x8."""
  return mock_tpu_devices(4, 4, 8, 'TPU v4', one_device_per_chip)


def mock_8x8x8_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 8x8x8."""
  return mock_tpu_devices(8, 8, 8, 'TPU v4', one_device_per_chip)


def mock_4x8x8_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 4x8x8."""
  return mock_tpu_devices(4, 8, 8, 'TPU v4', one_device_per_chip)


def mock_4x8x16_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 4x8x16."""
  return mock_tpu_devices(4, 8, 16, 'TPU v4', one_device_per_chip)


def mock_8x8x16_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 8x8x16."""
  return mock_tpu_devices(8, 8, 16, 'TPU v4', one_device_per_chip)


class MeshUtilsTest(test_util.JaxTestCase):

  @parameterized.named_parameters(
      ('2x2x1_t', mock_2x2x1_devices, True, (2, 2, 1, 1)),
      ('2x2x1_f', mock_2x2x1_devices, False, (2, 2, 1, 2)),
      ('8x8x16_t', mock_8x8x16_devices, True, (8, 8, 16, 1)),
      ('8x8x16_f', mock_8x8x16_devices, False, (8, 8, 16, 2)),
  )
  def test_bounds_from_last_device(self, devices, one_device_per_chip,
                                   expected_bounds):
    self.assertEqual(
        mesh_utils._bounds_from_last_device(devices(one_device_per_chip)[-1]),
        expected_bounds)

  @parameterized.named_parameters(
      ('4x4x4_t', (4, 4, 4), True),
      ('4x4x4_f', (4, 4, 4), False),
      ('8x8x16_t', (8, 8, 16), True),
      ('8x8x16_f', (8, 8, 16), False),
  )
  def test_get_physical_tpu_mesh(self, xyz, reorder):
    x, y, z = xyz
    jax_devices = mock_tpu_devices(x, y, z, 'TPU v4', True, reorder=reorder)
    normalized = mesh_utils._get_physical_tpu_mesh(jax_devices)
    self.assertEqual(normalized.shape, xyz)
    # major_to_minor: x, y, z
    for i in range(x):
      for j in range(y):
        for k in range(z):
          self.assertEqual(normalized[i, j, k].coords, (i, j, k))

  @parameterized.named_parameters(
      ('2x2x1', mock_2x2x1_devices, [1, 1, 4], [(), (), (0, 1, 2)]),
      ('2x2x4', mock_2x2x4_devices, [1, 4, 4], [(), (2,), (0, 1)]),
      ('4x4x4', mock_4x4x4_devices, [1, 16, 4], [(), (1, 2), (0,)]),
      ('4x4x8a', mock_4x4x8_devices, [1, 16, 8], [(), (0, 1), (2,)]),
      ('4x4x8b', mock_4x4x8_devices, [1, 8, 16], [(), (2,), (0, 1)]),
      ('4x4x8c', mock_4x4x8_devices, [16, 8, 1], [(0, 1), (2,), ()]),
      ('4x8x8', mock_4x8x8_devices, [1, 32, 8], [(), (0, 2), (1,)]),
      ('8x8x8', mock_8x8x8_devices, [1, 64, 8], [(), (1, 2), (0,)]),
      ('8x8x16', mock_8x8x16_devices, [1, 64, 16], [(), (0, 1), (2,)]),
      ('8x8', mock_8x8_devices, [8, 8], [(1,), (0, 2)])
  )
  def test_create_device_mesh_for_nd_torus(self, devices, mesh_shape,
                                           expected_assignment):
    jax_devices = devices(True)
    physical_mesh = mesh_utils._get_physical_tpu_mesh(jax_devices)
    _, assignment = mesh_utils._create_device_mesh_for_nd_torus(
        physical_mesh, mesh_shape)
    self.assertEqual(assignment, expected_assignment)

  @parameterized.named_parameters(
      ('2X4x4x4a', (1, 16, 4), (2, 1, 1)),
      ('2X4x4x4b', (1, 4, 16), (1, 2, 1)),
  )
  def test_create_hybrid_device_mesh(self, mesh_shape, dcn_mesh_shape):
    devices = mock_tpu_devices(4, 4, 4, 'TPU v4', True, 2)
    mesh = mesh_utils.create_hybrid_device_mesh(
        mesh_shape, dcn_mesh_shape, devices)
    total_mesh_shape = tuple(
        m1 * m2 for m1, m2 in zip(mesh_shape, dcn_mesh_shape))
    assert mesh.shape == total_mesh_shape
    axis_with_dcn = dcn_mesh_shape.index(2)
    i1, i2 = [0, 0, 0], [0, 0, 0]
    i2[axis_with_dcn] = 1
    d1, d2 = mesh[tuple(i1)], mesh[tuple(i2)]
    # mesh should have DCN minor to slice
    assert d1.coords == d2.coords
    assert d1.slice_index != d2.slice_index

  @parameterized.named_parameters(
      # Physical ring order over tray
      ('2x2_1d', mock_2x2_devices, [8], [0, 1, 2, 3, 6, 7, 4, 5]),
      # Reshaped physical ring order over tray
      ('2x2_2d', mock_2x2_devices, [2, 4], [[0, 1, 2, 3],
                                            [6, 7, 4, 5]]),
      # 4 per-tray rings
      ('4x4_2d', mock_4x4_devices, [4, 8], [[ 0,  1,  2,  3, 10, 11,  8,  9],
                                            [ 4,  5,  6,  7, 14, 15, 12, 13],
                                            [16, 17, 18, 19, 26, 27, 24, 25],
                                            [20, 21, 22, 23, 30, 31, 28, 29]]),
  )
  def test_v3_create_device_mesh(self, devices, mesh_shape,
                                 expected_device_id_mesh):
    global_devices = devices()
    mesh = mesh_utils.create_device_mesh(
        mesh_shape, devices=global_devices, contiguous_submeshes=False)
    device_id_mesh = np.vectorize(lambda d: d.id)(mesh)
    self.assertAllClose(np.array(expected_device_id_mesh), device_id_mesh)

  def _assert_contiguous_submeshes(self, global_device_mesh):
    global_mesh = Mesh(global_device_mesh, list(range(global_device_mesh.ndim)))
    max_process_index = max(d.process_index
                            for d in global_device_mesh.flatten())
    for p_idx in range(max_process_index + 1):
      # Raises an error if non-contiguous
      global_mesh._local_mesh(p_idx)

  def test_create_contiguous_submeshes_for_tpu_v4(self):
    v4 = mesh_utils._TPU_V4
    for topology, mesh_shapes in mesh_utils._TRANSPOSE_TRICKS.items():
      logging.vlog(1, "topology: %s", topology)
      devices = mock_tpu_devices(topology[0], topology[1], topology[2], v4,
                             one_device_per_chip=True)
      for mesh_shape in mesh_shapes:
        logging.vlog(1, "  mesh_shape: %s", mesh_shape)
        mesh = mesh_utils.create_device_mesh(
            mesh_shape, devices=devices, contiguous_submeshes=True)
        self._assert_contiguous_submeshes(mesh)

  def test_create_contiguous_submeshes_for_tpu_v4_leading_1_dims(self):
    v4 = mesh_utils._TPU_V4
    for topology, mesh_shapes in mesh_utils._TRANSPOSE_TRICKS.items():
      logging.vlog(1, "topology: %s", topology)
      devices = mock_tpu_devices(topology[0], topology[1], topology[2], v4,
                             one_device_per_chip=True)
      for mesh_shape in mesh_shapes:
        logging.vlog(1, '  mesh_shape: %s', (1, 1) + mesh_shape + (1, 1))
        mesh = mesh_utils.create_device_mesh(
            (1, 1) + mesh_shape + (1, 1),
            devices=devices,
            contiguous_submeshes=True)
        self._assert_contiguous_submeshes(mesh)

  def test_create_contiguous_submeshes_errors(self):
    v4 = mesh_utils._TPU_V4

    topology = (4, 4, 8)
    mesh_shape = (1, 16, 8)
    devices = mock_tpu_devices(topology[0], topology[1], topology[2], v4,
                           one_device_per_chip=True)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "create_device_mesh cannot create contiguous submeshes for "
        "physical mesh topology (4, 4, 8)"):
      mesh_utils.create_device_mesh(
          mesh_shape, devices=devices, contiguous_submeshes=True)

    topology = (4, 8, 8)
    mesh_shape = (1, 128, 2)
    devices = mock_tpu_devices(topology[0], topology[1], topology[2], v4,
                           one_device_per_chip=True)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "create_device_mesh cannot create contiguous submeshes for mesh_shape "
        "(1, 128, 2) and physical mesh topology (4, 8, 8). "
        'Available mesh_shapes: [(64, 4), (4, 64)]'):
      mesh_utils.create_device_mesh(
          mesh_shape, devices=devices, contiguous_submeshes=True)


if __name__ == '__main__':
  absltest.main()
