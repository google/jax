# Copyright 2022 The JAX Authors.
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

import os
import re
import socket
import time
from jax._src import clusters
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm

# We use an arbitrarily chosen port for the coordinator since we cannot
# rely on communication to choose one in real time.
coordinator_port = '8476'

def get_metadata(key):
  import requests  # pytype: disable=import-error
  import time  # pytype: disable=import-error
  # Based on https://github.com/tensorflow/tensorflow/pull/40317
  gce_metadata_endpoint = 'http://' + os.environ.get(
      'GCE_METADATA_IP', 'metadata.google.internal')

  retry_count = 0
  retrySeconds = 0.500
  api_resp = None

  while retry_count < 6:
    api_resp = requests.get(
        f'{gce_metadata_endpoint}/computeMetadata/v1/instance/attributes/{key}',
        headers={'Metadata-Flavor': 'Google'})
    if api_resp.status_code == 200:
      break
    retry_count += 1
    time.sleep(retrySeconds)

  if api_resp is None:
    raise RuntimeError(f"Getting metadata['{key}'] failed for 6 tries")
  return api_resp.text

def get_tpu_env_value(key):
  def get_tpu_env_value_from_metadata(key):
    tpu_env_data = get_metadata('tpu-env')
    key_value_pairs = tpu_env_data.split('\n')
    for key_value_pair in key_value_pairs:
      # Typical line is MEGASCALE_NUM_SLICES: '2'
      if ':' in key_value_pair:
        row_key, value = re.split(':', key_value_pair, 1)
        row_key = row_key.strip()
        if row_key == key:
          return value.strip().strip("'")
    return None

  value = os.environ.get(key, None)
  return value if value is not None else get_tpu_env_value_from_metadata(key)

def is_gce_env():
  worker_number_string = get_metadata('agent-worker-number')
  try:
    worker_number = int(worker_number_string)
    return True
  except:
    return False

def has_megascale_address():
  return get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS') is not None

def is_multislice_gce_env():
  return is_gce_env() and has_megascale_address()

def is_gke_env():
  return os.environ.get("TPU_WORKER_HOSTNAMES", None) is not None

def get_gce_worker_endpoints() -> str:
  return get_metadata('worker-network-endpoints').split(',')

class SingleSliceGceTpuCluster(clusters.ClusterEnv):
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm and is_gce_env() and not is_multislice_gce_env()

  @classmethod
  def get_coordinator_address(cls) -> str:
    return f"{get_gce_worker_endpoints()[0].split(':')[2]}:{coordinator_port}"

  @classmethod
  def get_process_count(cls) -> int:
    return len(get_gce_worker_endpoints())

  @classmethod
  def get_process_id(cls) -> int:
    return int(get_metadata('agent-worker-number'))

  @classmethod
  def get_local_process_id(cls) -> int | None:
    return None

class MultisliceGceTpuCluster(clusters.ClusterEnv):
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm and is_multislice_gce_env()

  @classmethod
  def get_coordinator_address(cls) -> str:
    if has_megascale_address():
      coordinator_address = get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS')
    else:
      coordinator_address = str(os.environ.get('TPU_WORKER_HOSTNAMES', None)).split(',')[0]
    coordinator_address = coordinator_address.split(':')[0]

    # The coordinator may not be up before the other hosts try to
    # communicate with it. We check for its existence with retries.
    coordinator_found = False
    lookup_attempt = 1
    max_coordinator_lookups = 50
    while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
      try:
        ip_address = socket.gethostbyname(coordinator_address)
        coordinator_found = True
      except socket.gaierror:
        print(f"Failed to recognize coordinator address {coordinator_address} on attempt {lookup_attempt}, retrying...")
        lookup_attempt += 1
        time.sleep(5)

    if not coordinator_found:
      raise RuntimeError(f"Failed to recognize coordinator address {coordinator_address}")

    # Use a different port for the jax coordinator than the MXLA coordinator,
    # which is set to 8080 in multislice GCE.
    return f'{coordinator_address}:{coordinator_port}'

  @classmethod
  def get_process_count(cls) -> int:
    return cls._get_process_count_per_slice() * cls._get_num_slices()

  @classmethod
  def get_process_id(cls) -> int:
    process_id_in_slice = cls._get_process_id_in_slice()
    slice_id = cls._get_slice_id()
    processes_per_slice = cls._get_process_count_per_slice()
    return process_id_in_slice + slice_id * processes_per_slice

  @classmethod
  def get_local_process_id(cls) -> int | None:
    return None

  @staticmethod
  def _get_process_count_per_slice() -> int:
    return len(get_gce_worker_endpoints())

  @staticmethod
  def _get_slice_id() -> int:
    if has_megascale_address():
      return int(get_tpu_env_value('MEGASCALE_SLICE_ID'))
    else:
      return 0

  @staticmethod
  def _get_process_id_in_slice() -> int:
    return int(get_metadata('agent-worker-number'))

  @staticmethod
  def _get_num_slices() -> int:
    if has_megascale_address():
      return int(get_tpu_env_value('MEGASCALE_NUM_SLICES'))
    else:
      return 1

class GkeTpuCluster(MultisliceGceTpuCluster):
  # This class handles both single and multislice GKE as the environment
  # variables are set similarly in both cases.
  @classmethod
  def is_env_present(cls) -> bool:
    return running_in_cloud_tpu_vm and is_gke_env()

  @staticmethod
  def _get_process_count_per_slice() -> int:
    tpu_worker_hostnames = str(os.environ.get('TPU_WORKER_HOSTNAMES', None))
    return len(tpu_worker_hostnames.split(','))

  @staticmethod
  def _get_process_id_in_slice() -> int:
    return int(str(os.environ.get('TPU_WORKER_ID')))
