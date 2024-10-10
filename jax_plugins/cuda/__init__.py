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
import traceback
from typing import Any

from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

# cuda_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
# preinstalled jax cuda plugin packages.
for pkg_name in ['jax_cuda12_plugin', 'jaxlib']:
  try:
    cuda_plugin_extension = importlib.import_module(
        f'{pkg_name}.cuda_plugin_extension'
    )
    cuda_versions = importlib.import_module(
        f'{pkg_name}.cuda_versions'
    )
  except ImportError:
    cuda_plugin_extension = None
    cuda_versions = None
  else:
    break

logger = logging.getLogger(__name__)


def _check_cuda_versions(raise_on_first_error: bool = False,
                         debug: bool = False):
  assert cuda_versions is not None
  results: list[dict[str, Any]] = []

  def _make_msg(name: str,
                runtime_version: int,
                build_version: int,
                min_supported: int,
                debug_msg: bool = False):
    if debug_msg:
      return (f"Package: {name}\n"
              f"Version JAX was built against: {build_version}\n"
              f"Minimum supported: {min_supported}\n"
              f"Installed version: {runtime_version}")
    if min_supported:
      req_str = (f"The local installation version must be no lower than "
                 f"{min_supported}.")
    else:
      req_str = ("The local installation must be the same version as "
                 "the version against which JAX was built.")
    msg = (f"Outdated {name} installation found.\n"
           f"Version JAX was built against: {build_version}\n"
           f"Minimum supported: {min_supported}\n"
           f"Installed version: {runtime_version}\n"
           f"{req_str}")
    return msg

  def _version_check(name: str,
                     get_version,
                     get_build_version,
                     scale_for_comparison: int = 1,
                     min_supported_version: int = 0):
    """Checks the runtime CUDA component version against the JAX one.

    Args:
      name: Of the CUDA component.
      get_version: A function to get the local runtime version of the component.
      get_build_version: A function to get the build version of the component.
      scale_for_comparison: For rounding down a version to ignore patch/minor.
      min_supported_version: An absolute minimum version required. Must be
        passed without rounding down.

    Raises:
      RuntimeError: If the component is not found, or is of unsupported version,
        and if raising the error is not deferred till later.
    """

    build_version = get_build_version()
    try:
      version = get_version()
    except Exception as e:
      err_msg = f"Unable to load {name}. Is it installed?"
      if raise_on_first_error:
        raise RuntimeError(err_msg) from e
      err_msg += f"\n{traceback.format_exc()}"
      results.append({"name": name, "installed": False, "msg": err_msg})
      return

    if not min_supported_version:
      min_supported_version = build_version // scale_for_comparison
    passed = min_supported_version <= version

    if not passed or debug:
      msg = _make_msg(name=name,
                      runtime_version=version,
                      build_version=build_version,
                      min_supported=min_supported_version,
                      debug_msg=passed)
      if not passed and raise_on_first_error:
        raise RuntimeError(msg)
      else:
        record = {"name": name,
                  "installed": True,
                  "msg": msg,
                  "passed": passed,
                  "build_version": build_version,
                  "version": version,
                  "minimum_supported": min_supported_version}
        results.append(record)

  _version_check("CUDA", cuda_versions.cuda_runtime_get_version,
                 cuda_versions.cuda_runtime_build_version,
                 scale_for_comparison=10,
                 min_supported_version=12010)
  _version_check(
      "cuDNN",
      cuda_versions.cudnn_get_version,
      cuda_versions.cudnn_build_version,
      # NVIDIA promise both backwards and forwards compatibility for cuDNN patch
      # versions:
      # https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#api-compat
      scale_for_comparison=100,
      min_supported_version=9100
  )
  _version_check("cuFFT", cuda_versions.cufft_get_version,
                 cuda_versions.cufft_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100)
  _version_check("cuSOLVER", cuda_versions.cusolver_get_version,
                 cuda_versions.cusolver_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100,
                 min_supported_version=11400)
  _version_check("cuPTI", cuda_versions.cupti_get_version,
                 cuda_versions.cupti_build_version,
                 min_supported_version=18)
  _version_check("cuBLAS", cuda_versions.cublas_get_version,
                 cuda_versions.cublas_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100,
                 min_supported_version=120100)
  _version_check("cuSPARSE", cuda_versions.cusparse_get_version,
                 cuda_versions.cusparse_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100,
                 min_supported_version=12100)

  errors = []
  debug_results = []
  for result in results:
    message: str = result['msg']
    if not result['installed'] or not result['passed']:
      errors.append(message)
    else:
      debug_results.append(message)

  join_str = f'\n{"-" * 50}\n'
  if debug_results:
    print(f'CUDA components status (debug):\n'
          f'{join_str.join(debug_results)}')
  if errors:
    raise RuntimeError(f'Unable to use CUDA because of the '
                       f'following issues with CUDA components:\n'
                       f'{join_str.join(errors)}')


def _get_library_path():
  installed_path = (
      pathlib.Path(__file__).resolve().parent / 'xla_cuda_plugin.so'
  )
  if installed_path.exists():
    return installed_path

  local_path = os.path.join(
      os.path.dirname(__file__), 'pjrt_c_api_gpu_plugin.so'
  )
  if not os.path.exists(local_path):
    runfiles_dir = os.getenv('RUNFILES_DIR', None)
    if runfiles_dir:
      local_path = os.path.join(
          runfiles_dir, 'xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so'
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

  if not os.getenv("JAX_SKIP_CUDA_CONSTRAINTS_CHECK"):
    _check_cuda_versions()
  else:
    print('Skipped CUDA versions constraints check due to the '
          'JAX_SKIP_CUDA_CONSTRAINTS_CHECK env var being set.')

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
