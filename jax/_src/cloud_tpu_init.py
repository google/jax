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

import os
from jax._src import hardware_utils

running_in_cloud_tpu_vm: bool = False


def maybe_import_libtpu():
  try:
    # pylint: disable=import-outside-toplevel
    # pytype: disable=import-error
    import libtpu

    # pytype: enable=import-error
    # pylint: enable=import-outside-toplevel
  except ImportError:
    return None
  else:
    return libtpu


def jax_force_tpu_init() -> bool:
  return 'JAX_FORCE_TPU_INIT' in os.environ


def cloud_tpu_init() -> None:
  """Automatically sets Cloud TPU topology and other env vars.

  **This must be called before the TPU runtime is loaded, which happens as soon
  as JAX's C++ backend is loaded! I.e. call this before xla_bridge or xla_client
  is imported.**

  Safe to call in non-Cloud TPU environments.

  Some of these environment variables are used to tell the TPU runtime what kind
  of mesh topology to use. It assumes a single-host topology by default, so we
  manually set them here to default to the full pod slice if applicable.

  This will not set any env vars if a single topology-related env var is already
  set.
  """
  global running_in_cloud_tpu_vm

  # Exit early if we're not running on Cloud TPU.
  libtpu_module = maybe_import_libtpu()
  num_tpu_chips = hardware_utils.num_available_tpu_chips_and_device_id()[0]
  if (libtpu_module is None or num_tpu_chips == 0) and not jax_force_tpu_init():
    return

  running_in_cloud_tpu_vm = True

  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
  os.environ.setdefault('JAX_PLATFORMS', 'tpu,cpu')
  os.environ['TPU_ML_PLATFORM'] = 'JAX'
  if hardware_utils.tpu_enhanced_barrier_supported():
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + " --xla_tpu_use_enhanced_launch_barrier=true"
