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

import functools
import importlib
import logging
import os
import pathlib
import platform
import sys

from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

# cuda_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
# preinstalled jax cuda plugin packages.
for pkg_name in ['jax_cuda12_plugin', 'jax_cuda11_plugin', 'jaxlib']:
  try:
    cuda_plugin_extension = importlib.import_module(
        f'{pkg_name}.cuda_plugin_extension'
    )
  except ImportError:
    cuda_plugin_extension = None
  else:
    break

logger = logging.getLogger(__name__)


def _get_library_path():
  installed_path = (
      pathlib.Path(__file__).resolve().parent / 'xla_cuda_plugin.so'
  )
  if installed_path.exists():
    return installed_path

  local_path = os.path.join(
      os.path.dirname(__file__), 'pjrt_c_api_gpu_plugin.so'
  )
  if os.path.exists(local_path):
    logger.debug(
        'Native library %s does not exist. This most likely indicates an issue'
        ' with how %s was built or installed. Fallback to local test'
        ' library %s',
        installed_path,
        __package__,
        local_path,
    )
    return local_path

  logger.debug(
      'WARNING: Native library %s and local test library path %s do not'
      ' exist. This most likely indicates an issue with how %s was built or'
      ' installed or missing src files.',
      installed_path,
      local_path,
      __package__,
  )
  return None


def initialize():
  path = _get_library_path()
  if path is None:
    return

  options = xla_client.generate_pjrt_gpu_plugin_options()
  c_api = xb.register_plugin(
      'cuda', priority=500, library_path=str(path), options=options
  )
  if cuda_plugin_extension:
    xla_client.register_custom_call_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_call_target, c_api
        ),
    )
    for _name, _value in cuda_plugin_extension.registrations().items():
      xla_client.register_custom_call_target(_name, _value, platform="CUDA")
  else:
    logger.warning('cuda_plugin_extension is not found.')
