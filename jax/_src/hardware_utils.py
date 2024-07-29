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

import os
import pathlib
import glob

_GOOGLE_PCI_VENDOR_ID = '0x1ae0'
_TPU_PCI_DEVICE_IDS = [
    # TPU v2, v3
    '0x0027',
    # TPU v4
    '0x005e',
    # TPU v5e
    '0x0063',
    # Testing only
    '0x0056',
    '0x0062',
]

_TPU_ENHANCED_BARRIER_SUPPORTED = [
    # TPU v2, v3
    '0x0027',
    # TPU v4
    '0x005e',
]

_NVIDIA_GPU_DEVICES = [
    '/dev/nvidia0',
    '/dev/dxg',  # WSL2
]

def num_available_tpu_chips_and_device_id():
  """Returns the device id and number of TPU chips attached through PCI."""
  num_chips = 0
  device_id = ''
  for vendor_path in glob.glob('/sys/bus/pci/devices/*/vendor'):
    vendor_id = pathlib.Path(vendor_path).read_text().strip()
    if vendor_id != _GOOGLE_PCI_VENDOR_ID:
      continue

    device_path = os.path.join(os.path.dirname(vendor_path), 'device')
    device_id = pathlib.Path(device_path).read_text().strip()
    if device_id in _TPU_PCI_DEVICE_IDS:
      num_chips += 1

  return num_chips, device_id


def tpu_enhanced_barrier_supported() -> bool:
  """Returns if tpu_enhanced_barrier flag is supported on this TPU version."""
  _, device_id = num_available_tpu_chips_and_device_id()
  return device_id in _TPU_ENHANCED_BARRIER_SUPPORTED


def has_visible_nvidia_gpu() -> bool:
  """True if there's a visible nvidia gpu available on device, False otherwise."""

  return any(os.path.exists(d) for d in _NVIDIA_GPU_DEVICES)
