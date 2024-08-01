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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from jax._src import mesh as mesh_lib
from jax._src import test_util
from jax._src.sharding_impls import NamedSharding, PartitionSpec, local_to_global_shape
from jax.experimental import mesh_utils
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import numpy as np

# pyformat: disable


@dataclasses.dataclass(frozen=True)
class MockClient:
  """Mock client for testing, everything is done as process index 0."""
  def process_index(self) -> int:
    return 0


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
  client: MockClient = dataclasses.field(default_factory=MockClient)


def mock_tpu_devices(x, y, z, dev_kind, one_device_per_chip, num_slices=1,
                     reorder=False):
  """Produce fake jax.devices() output for a TPU slice."""
  assert x > 0 and y > 0 and z > 0

  cores_per_chip = 1 if one_device_per_chip else 2

  # 3D shape of the mesh of devices on each host (= process).
  nxd, nyd, nzd = (min(x, 2), min(y, 2), 1)
  # 3D shape of the mesh of hosts (= processes):
  nxp, nyp, nzp = x // nxd, y // nyd, z // nzd
  assert nxp * nxd == x
  assert nyp * nyd == y
  assert nzp * nzd == z

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

  # Validate the generated mock devices:
  num_local_chips = nxd * nyd  # Number of mock devices / process.
  if num_local_chips < 4:
    # Sub-host slice = fewer than the 4 chips available on a host:
    # e.g., 1x1 TPU v2.  All devices should be on one host.
    num_all_chips = x * y * z
    assert num_all_chips == num_local_chips, f'Bad shape: {x=}, {y=}, {z=}'
    # Implied by the previous assertion, but let's be explicit:
    assert z == 1
    _validate_mocked_devices_for_subhost_slice(devices, x, y, cores_per_chip)
  else:
    _validate_mocked_devices(devices, num_local_chips * cores_per_chip)

  return devices


# If this function raises, it's a bug in the test code!
def _validate_mocked_devices_for_subhost_slice(devices, x, y, cores_per_chip):
  first_device = devices[0]
  distinct_coords = set()
  for d in devices:
    assert d.process_index == first_device.process_index
    assert d.coords[0] >= 0 and d.coords[0] < x
    assert d.coords[1] >= 0 and d.coords[1] < y
    assert d.coords[2] == 0
    assert d.core_on_chip >= 0 and d.core_on_chip < cores_per_chip
    distinct_coords.add((d.coords[0], d.coords[1], 0, d.core_on_chip))
  assert len(distinct_coords) == x * y * cores_per_chip


# If this function raises, it's a bug in the test code!
def _validate_mocked_devices(devices, num_local_devices):
  # NOTE: this function is not called for sub-host slices.
  process_to_devices = collections.defaultdict(list)
  for d in devices:
    process_to_devices[d.process_index].append(d)

  for local_devices in process_to_devices.values():
    assert len(local_devices) == num_local_devices, local_devices
    # All devices have same z coord
    assert len({d.coords[2] for d in local_devices}) == 1, local_devices
    # All devices in a 2x2 subgrid
    min_coords = min(d.coords for d in local_devices)
    expected = set()
    for x, y in [(0,0), (0,1), (1,0), (1,1)]:
      expected.add((min_coords[0] + x, min_coords[1] + y, min_coords[2]))
    assert {d.coords for d in local_devices} == expected, local_devices


def mock_1x1_devices():
  """Hard-coded reproduction of jax.devices() output on v3-1x1."""
  return mock_tpu_devices(1, 1, 1, 'TPU v3', False)


def mock_2x2_devices():
  """Hard-coded reproduction of jax.devices() output on v3-2x2."""
  return mock_tpu_devices(2, 2, 1, 'TPU v3', False)


def mock_4x4_devices():
  """Hard-coded reproduction of jax.devices() output on v3-4x4."""
  return mock_tpu_devices(4, 4, 1, 'TPU v3', False)


def mock_8x8_devices(one_device_per_chip=False):
  """Hard-coded reproduction of jax.devices() output on v3-8x8."""
  return mock_tpu_devices(8, 8, 1, 'TPU v3', one_device_per_chip)


def mock_1x2x1_devices(one_device_per_chip):
  """Hard-coded reproduction of jax.devices() output on 2x2x1."""
  return mock_tpu_devices(1, 2, 1, 'TPU v4', one_device_per_chip)


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
      ('1x1', mock_1x1_devices, (1, 1, 1, 2)),
      ('2x2', mock_2x2_devices, (2, 2, 1, 2)),
      ('4x4', mock_4x4_devices, (4, 4, 1, 2)),
      ('8x8', mock_8x8_devices, (8, 8, 1, 2)),
  )
  def test_bounds_from_last_device_2d(self, devices, expected_bounds):
    self.assertEqual(
        mesh_utils._bounds_from_last_device(devices()[-1]),
        expected_bounds)

  @parameterized.named_parameters(
      ('1x2x1_t', mock_1x2x1_devices, True, (1, 2, 1, 1)),
      ('1x2x1_f', mock_1x2x1_devices, False, (1, 2, 1, 2)),
      ('2x2x1_t', mock_2x2x1_devices, True, (2, 2, 1, 1)),
      ('2x2x1_f', mock_2x2x1_devices, False, (2, 2, 1, 2)),
      ('8x8x16_t', mock_8x8x16_devices, True, (8, 8, 16, 1)),
      ('8x8x16_f', mock_8x8x16_devices, False, (8, 8, 16, 2)),
  )
  def test_bounds_from_last_device_3d(self, devices, one_device_per_chip,
                                      expected_bounds):
    self.assertEqual(
        mesh_utils._bounds_from_last_device(devices(one_device_per_chip)[-1]),
        expected_bounds)

  @parameterized.named_parameters(
      ('1x2x1_t', (1, 2, 1), True),
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

  def test_get_physical_tpu_mesh_with_subslice_TPU_v2_1x1(self):
    one_device_per_chip = False  # Each TPU v2 chip has 2 devices.
    device_list = mock_tpu_devices(1, 1, 1, 'TPU v2', one_device_per_chip)
    device_array = mesh_utils._get_physical_tpu_mesh(device_list)
    self.assertEqual(device_array.shape, (1, 1, 2))

    # A subslice that includes the device at (0, 0, 0): core #0 of the
    # device at (x, y, z) == (0, 0, 0).
    subslice0 = mesh_utils._get_physical_tpu_mesh([device_array[0, 0, 0]])
    self.assertEqual(subslice0.shape, (1, 1, 1))
    self.assertEqual(subslice0[0, 0, 0], device_array[0, 0, 0])
    self.assertEqual(subslice0[0, 0, 0].coords, (0, 0, 0))
    self.assertEqual(subslice0[0, 0, 0].core_on_chip, 0)

    # Another subsublice, without the device at (0, 0, 0): core #1 of
    # the device at (x, y, z) == (0, 0, 0).
    subslice1 = mesh_utils._get_physical_tpu_mesh([device_array[0, 0, 1]])
    self.assertEqual(subslice1.shape, (1, 1, 1))
    self.assertEqual(subslice1[0, 0, 0], device_array[0, 0, 1])
    self.assertEqual(subslice1[0, 0, 0].coords, (0, 0, 0))
    self.assertEqual(subslice1[0, 0, 0].core_on_chip, 1)

  def test_get_physical_tpu_mesh_with_subslice_TPU_v4_1x2x1(self):
    one_device_per_chip = True  # For TPU v4, chip == device.
    device_list = mock_tpu_devices(1, 2, 1, 'TPU v4', one_device_per_chip)
    device_array = mesh_utils._get_physical_tpu_mesh(device_list)
    self.assertEqual(device_array.shape, (1, 2, 1))

    # A subslice that includes the device at (0, 0, 0).
    subslice0 = mesh_utils._get_physical_tpu_mesh([device_array[0, 0, 0]])
    self.assertEqual(subslice0.shape, (1, 1, 1))
    self.assertEqual(subslice0[0, 0, 0], device_array[0, 0, 0])

    # Another subsublice, without the device at (0, 0, 0).
    subslice1 = mesh_utils._get_physical_tpu_mesh([device_array[0, 1, 0]])
    self.assertEqual(subslice1.shape, (1, 1, 1))
    self.assertEqual(subslice1[0, 0, 0], device_array[0, 1, 0])

  def test_get_physical_tpu_mesh_with_subslice_TPU_v5e_4x4(self):
    one_device_per_chip = True  # For TPU v5e, chip == device.
    device_list = mock_tpu_devices(4, 4, 1, 'TPU v5e', one_device_per_chip)
    device_array = mesh_utils._get_physical_tpu_mesh(device_list)

    # `device_array` is isomorphic with a 4x4 grid (z coord == 0).
    self.assertEqual(device_array.shape, (4, 4, 1))

    # Two subslices: each subslice has shape (4, 2); first one starts
    # at (x=0, y=0), the other at (x=0, y=2); visually, the left
    # and right halves of the (4, 4) grid.
    for start_y in (0, 2):
      subslice_devices = []
      for x in range(4):
        for delta_y in range(2):
          subslice_devices.append(device_array[x, start_y + delta_y, 0])
      logging.info(
          'start_y=%s subslice_devices=%s', start_y, subslice_devices
      )
      subslice = mesh_utils._get_physical_tpu_mesh(subslice_devices)
      self.assertEqual(subslice.shape, (4, 2, 1))
      for x in range(4):
        for delta_y in range(2):
          self.assertEqual(
              subslice[x, delta_y],
              device_array[x, start_y + delta_y, 0],
          )

  def test_get_physical_tpu_mesh_with_bad_subslice(self):
    one_device_per_chip = True  # For TPU v5e, chip == device.
    device_list = mock_tpu_devices(4, 4, 1, 'TPU v5e', one_device_per_chip)
    device_array = mesh_utils._get_physical_tpu_mesh(device_list)

    self.assertEqual(device_array.shape, (4, 4, 1))

    # Second subslice from
    # test_get_physical_tpu_mesh_with_subslice_TPU_v5e_4x4, without
    # the device from its top-left corner (device from (0, 2, 0)).
    start_y = 2
    subslice_devices = []
    for x in range(4):
      for delta_y in range(2):
        if (x == 0) and (delta_y == 0):
          # Skip device from (0, 2, 0).
          continue
        subslice_devices.append(device_array[x, start_y + delta_y, 0])

    # subslice_devices are obviously not a cuboid: only 7 devices.
    with self.assertRaises(AssertionError):
      mesh_utils._get_physical_tpu_mesh(subslice_devices)

    # Make it a bit harder, such that just a simple test on
    # len(subslice_devices) is not enough: 8 devices, but two of them
    # are identical (device from (2, 2, 0) is duplicated.
    subslice_devices.append(device_array[2, 2, 0])
    with self.assertRaisesRegex(AssertionError, 'not a contiguous cuboid'):
      mesh_utils._get_physical_tpu_mesh(subslice_devices)

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
      ('8x8', mock_8x8_devices, [8, 8], [(1,), (0, 2)]),
  )
  def test_create_device_mesh_for_nd_torus(
      self, devices, mesh_shape, expected_assignment
  ):
    jax_devices = devices(True)
    physical_mesh = mesh_utils._get_physical_tpu_mesh(jax_devices)
    _, assignment = mesh_utils._create_device_mesh_for_nd_torus(
        physical_mesh, mesh_shape
    )

    # The expected assignment is specified as a list, where each element is a
    # sequence of physical axis assigned. We convert this into assignment
    # matrix.
    expected_assignment_matrix = np.ones(
        [physical_mesh.ndim, len(mesh_shape)], dtype=np.int64
    )
    for logical_axis, axis_assignment in enumerate(expected_assignment):
      for physical_axis in axis_assignment:
        expected_assignment_matrix[physical_axis, logical_axis] = (
            physical_mesh.shape[physical_axis]
        )
    self.assertArraysEqual(assignment, expected_assignment_matrix)

  @parameterized.named_parameters(
      ('2x2x1', mock_2x2x1_devices,),
      ('2x2x4', mock_2x2x4_devices, ),
      ('4x4x4', mock_4x4x4_devices,),
      ('4x4x8', mock_4x4x8_devices,),
      ('4x8x8', mock_4x8x8_devices, ),
      ('8x8', mock_8x8_devices),
  )
  def test_create_device_mesh_has_computable_global_shape(self, devices):
    def factorize(n, max_factors=3):
      if max_factors == 1 or n == 1:
        yield (n, ) * max_factors
        return
      for i in range(2, n+1):
        if n % i == 0:
          for remaining in factorize(n // i, max_factors=max_factors - 1):
            yield (i, *remaining)
    jax_devices = devices(True)
    for mesh_shape in factorize(len(jax_devices), max_factors=3):
      mesh = mesh_utils.create_device_mesh(mesh_shape, devices=jax_devices,
                                           allow_split_physical_axes=True)
      mesh = mesh_lib.Mesh(mesh, ('a', 'b', 'c'))
      sharding = NamedSharding(mesh, PartitionSpec('a', 'b', 'c'))
      computed_global_shape = local_to_global_shape(sharding, (1, 1, 1))
      self.assertFalse(
          np.any([x is None for x in computed_global_shape]),
          f'{mesh_shape=}, {computed_global_shape=} is not uniform')

      sharding = NamedSharding(mesh, PartitionSpec(('a', 'c',), 'b'))
      computed_global_shape = local_to_global_shape(sharding, (1, 1, 1))
      self.assertFalse(
          np.any([x is None for x in computed_global_shape]),
          f'{mesh_shape=}, {computed_global_shape=} is not uniform')


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
      ('8x8', mock_8x8_devices, [8, 8], [(1,), (0, 2)]),
  )
  def test_create_device_mesh_for_nd_torus_split_axes_backward_compatible(
      self, devices, mesh_shape, expected_assignment
  ):
    jax_devices = devices(True)
    physical_mesh = mesh_utils._get_physical_tpu_mesh(jax_devices)
    _, assignment = mesh_utils._create_device_mesh_for_nd_torus_splitting_axes(
        physical_mesh, mesh_shape
    )

    # The expected assignment is specified as a list, where each element is a
    # sequence of physical axis assigned. We convert this into assignment
    # matrix.
    expected_assignment_matrix = np.ones(
        [physical_mesh.ndim, len(mesh_shape)], dtype=np.int64
    )
    for logical_axis, axis_assignment in enumerate(expected_assignment):
      for physical_axis in axis_assignment:
        expected_assignment_matrix[physical_axis, logical_axis] = (
            physical_mesh.shape[physical_axis]
        )
    self.assertArraysEqual(assignment, expected_assignment_matrix)

  @parameterized.named_parameters(
      (
          '4x4x4a',
          mock_4x4x4_devices,
          [2, 1, 32],
          [
              [1, 1, 4],
              [1, 1, 4],
              [2, 1, 2],
          ],
      ),
      (
          '4x4x4b',
          mock_4x4x4_devices,
          [8, 8, 1],
          [
              [1, 4, 1],
              [2, 2, 1],
              [4, 1, 1],
          ],
      ),
      (
          '4x4x8a',
          mock_4x4x8_devices,
          [2, 2, 8, 4],
          [
              [1, 1, 1, 4],
              [2, 2, 1, 1],
              [1, 1, 8, 1],
          ],
      ),
      (
          '4x4x8b',
          mock_4x4x8_devices,
          [2, 4, 1, 16],
          [
              [1, 1, 1, 4],
              [1, 1, 1, 4],
              [2, 4, 1, 1],
          ],
      ),
      (
          '4x8x8',
          mock_4x8x8_devices,
          [1, 128, 2],
          [
              [1, 2, 2],
              [1, 8, 1],
              [1, 8, 1],
          ],
      ),
      (
          '8x8',
          mock_8x8_devices,
          [2, 1, 32, 1],
          [
              [1, 1, 8, 1],
              [2, 1, 4, 1],
              [1, 1, 1, 1],
          ],
      ),
  )
  def test_create_device_mesh_for_nd_torus_split_axes_can_handle_axes_split(
      self, devices, mesh_shape, assignment_matrix
  ):
    jax_devices = devices(True)
    physical_mesh = mesh_utils._get_physical_tpu_mesh(jax_devices)
    logical_mesh, assignment = mesh_utils._create_device_mesh_for_nd_torus(
        physical_mesh, mesh_shape, allow_split_physical_axes=True
    )
    self.assertEqual(logical_mesh.shape, tuple(mesh_shape))
    self.assertArraysEqual(
        assignment, np.array(assignment_matrix, dtype=np.int64)
    )

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
    self.assertEqual(mesh.shape, total_mesh_shape)

  @parameterized.named_parameters(
      ('2X4x4x4a', (1, 16, 4), (2, 1, 1)),
      ('2X4x4x4b', (1, 4, 16), (1, 2, 1)),
  )
  def test_create_hybrid_device_mesh_device_sorting(
      self,
      mesh_shape: tuple[int, ...],
      dcn_mesh_shape: tuple[int, ...],
  ):
    devices = mock_tpu_devices(4, 4, 4, 'TPU v4', True, 2)
    reversed_slices_devices = list(
        np.flip(np.array(devices).reshape(2, -1), axis=0).flat)
    mesh = mesh_utils.create_hybrid_device_mesh(
        mesh_shape,
        dcn_mesh_shape,
        devices,
        should_sort_granules_by_key=False,
    )
    sorted_slices_mesh = mesh_utils.create_hybrid_device_mesh(
        mesh_shape,
        dcn_mesh_shape,
        reversed_slices_devices,
        should_sort_granules_by_key=True,
    )
    np.testing.assert_array_equal(mesh, sorted_slices_mesh)
    self.assertSetEqual(
        {0, 1},
        {d.slice_index for d in sorted_slices_mesh.flat},
    )

    reversed_slices_mesh = mesh_utils.create_hybrid_device_mesh(
        mesh_shape,
        dcn_mesh_shape,
        reversed_slices_devices,
        should_sort_granules_by_key=False,
    )
    self.assertSetEqual(
        {1, 0},
        {d.slice_index for d in reversed_slices_mesh.flat},
    )

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
          mesh_shape, devices=devices, contiguous_submeshes=True
      )


def int64_array(x) -> np.ndarray:
  return np.array(x, dtype=np.int64)


def get_int_mesh(shape: Sequence[int]) -> np.ndarray:
  return np.arange(np.prod(shape), dtype=np.int64).reshape(shape)


class SplitAxesDeviceMeshCreationTest(test_util.JaxTestCase):

  def test_get_prime_factors(self):
    self.assertEqual(mesh_utils._get_prime_factors(1), [])  # 1 has no factor.
    self.assertEqual(mesh_utils._get_prime_factors(2), [2])
    self.assertEqual(mesh_utils._get_prime_factors(4), [2, 2])
    self.assertEqual(mesh_utils._get_prime_factors(8), [2, 2, 2])
    self.assertEqual(mesh_utils._get_prime_factors(6), [2, 3])
    self.assertEqual(mesh_utils._get_prime_factors(16), [2, 2, 2, 2])
    self.assertEqual(mesh_utils._get_prime_factors(12), [2, 2, 3])
    self.assertEqual(mesh_utils._get_prime_factors(121), [11, 11])  # square
    self.assertEqual(mesh_utils._get_prime_factors(43), [43])  # prime

  @parameterized.named_parameters(
      (
          '2x2x1',
          [2, 2, 1],
          [1, 2, 1],
          4,
          [],  # infeasible
      ),
      (
          '12x4x4',
          [12, 4, 4],
          [2, 2, 1],
          6,
          [[6, 1, 1], [3, 2, 1], [3, 1, 2]],
      ),
      (
          '4x4x8',
          [4, 4, 8],
          [2, 2, 2],
          4,
          [[2, 2, 1], [2, 1, 2], [1, 2, 2], [1, 1, 4]],
      ),
  )
  def test_enumerate_feasible_axis_assignments(
      self,
      physical_mesh_shape,
      assigned_physical_mesh_shape,
      logical_axis_size,
      expected_assignments,
  ):
    assignment = int64_array([list(assigned_physical_mesh_shape)]).T
    self.assertArraysEqual(
        list(
            mesh_utils._enumerate_feasible_logical_axis_assignments(
                physical_mesh_shape,
                assignment,
                logical_axis_size=logical_axis_size,
            )
        ),
        [int64_array(a) for a in expected_assignments],
    )

  @parameterized.named_parameters(
      (
          '2x2x1',
          [2, 2, 1],
          [1, 2, 2, 1],
          [
              [1, 2, 1, 1],
              [1, 1, 2, 1],
              [1, 1, 1, 1],
          ],
      ),
      (
          '4x4x4',
          [4, 4, 4],
          [2, 1, 32],
          [
              [1, 1, 4],
              [2, 1, 2],
              [1, 1, 4],
          ],
      ),
      (
          '12x4x8',
          [12, 4, 8],
          [2, 8, 24],
          [
              [2, 2, 3],
              [1, 2, 4],
              [1, 2, 2],
          ],
      ),
  )
  def test_generate_logical_mesh(
      self,
      physical_mesh_shape,
      logical_mesh_shape,
      assignment,
  ):
    assignment = np.array(assignment, dtype=np.int64)
    physical_mesh = get_int_mesh(physical_mesh_shape)
    logical_mesh = mesh_utils._generate_logical_mesh(
        physical_mesh, logical_mesh_shape, assignment
    )
    self.assertEqual(logical_mesh.shape, tuple(logical_mesh_shape))
    # We check that the logical mesh is assigned correctly using the following
    # consistency check, which transforms the logical mesh back to physical
    # mesh.
    transpose = (
        np.arange(assignment.size).reshape(assignment.shape).T.reshape([-1])
    )
    self.assertArraysEqual(
        physical_mesh.reshape([-1]),
        logical_mesh.reshape(np.reshape(assignment.T, [-1]))
        .transpose(transpose)
        .reshape([-1]),
    )

  def test_prefer_assignment_whole_axis_size(self):
    self.assertTrue(
        mesh_utils._prefer_first_logical_axis_assignment(
            int64_array([1, 2, 1]),
            int64_array([1, 1, 2]),
            physical_mesh_shape=[2, 2, 4],
            assignment=int64_array([[1, 1, 1]]).T,
        )
    )

  def test_prefer_assignment_more_whole_axes(self):
    # This entails the original implementation already.
    self.assertTrue(
        mesh_utils._prefer_first_logical_axis_assignment(
            int64_array([4, 4, 1]),
            int64_array([1, 1, 16]),
            physical_mesh_shape=[4, 4, 16],
            assignment=int64_array([[1, 1, 1]]).T,
        )
    )

  def test_prefer_assignment_avoid_already_assigned(self):
    self.assertTrue(
        mesh_utils._prefer_first_logical_axis_assignment(
            int64_array([2, 1]),
            int64_array([1, 2]),
            physical_mesh_shape=[2, 4],
            assignment=int64_array([[1, 2]]).T,
        )
    )


if __name__ == '__main__':
  absltest.main()
