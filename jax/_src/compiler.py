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

# Interface to the compiler

from __future__ import annotations

from collections.abc import Sequence
import logging
import os
import tempfile
import time
from typing import Any
import warnings

from jax._src import compilation_cache
from jax._src import config as config
from jax._src import distributed
from jax._src import lib
from jax._src import monitoring
from jax._src import profiler
from jax._src import traceback_util
from jax._src.interpreters import mlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
import numpy as np


_DISABLE_MOST_OPTIMIZATIONS = config.bool_flag(
    'jax_disable_most_optimizations',
    config.bool_env('JAX_DISABLE_MOST_OPTIMIZATIONS', False),
    'Try not to do much optimization work. This can be useful if the cost of '
    'optimization is greater than that of running a less-optimized program.')

_COMPILER_DETAILED_LOGGING_MIN_OPS = config.int_flag(
    "jax_compiler_detailed_logging_min_ops",
    config.int_env("JAX_COMPILER_DETAILED_LOGGING_MIN_OPS", 10),
    help=(
        'How big should a module be in MLIR operations before JAX enables '
        'detailed compiler logging? The intent of this flag is to suppress '
        'detailed logging for small/uninteresting computations.'
    ),
)

# The special XLA-AutoFDO profile version that indicates that a profile is not
# available and retrieval should not be attempted.
_NO_PROFILE_DONT_RETRIEVE = -1

traceback_util.register_exclusion(__file__)

CompileOptions = xc.CompileOptions

logger = logging.getLogger(__name__)


# Will be monkeypatched with the function that gets the XLA-AutoFDO profile
# version. The default (-1) takes care of errors.
# TODO(b/289098047): consider refactoring this interface.
def get_latest_profile_version(backend: xc.Client) -> int:
  del backend
  return -1


def _walk_operations(op, k):
  k -= 1
  if k < 0:
    return k
  for region in op.regions:
    for block in region:
      for child_op in block:
        k = _walk_operations(child_op, k)
        if k < 0:
          return k
  return k


def use_detailed_logging(module: ir.Module) -> bool:
  """Returns 'true' if detailed logging should be enabled for 'module'."""
  bound = _COMPILER_DETAILED_LOGGING_MIN_OPS.value
  return _walk_operations(module.operation, bound) < 0


def get_compile_options(
    num_replicas: int,
    num_partitions: int,
    device_assignment=None,
    use_spmd_partitioning: bool = True,
    use_auto_spmd_partitioning: bool = False,
    auto_spmd_partitioning_mesh_shape: list[int] | None = None,
    auto_spmd_partitioning_mesh_ids: list[int] | None = None,
    env_options_overrides: dict[str, str] | None = None,
    fdo_profile: bytes | None = None,
    detailed_logging: bool = True,
    backend: xc.Client | None = None,
) -> xc.CompileOptions:
  """Returns the compile options to use, as derived from flag values.

  Args:
    num_replicas: Number of replicas for which to compile.
    num_partitions: Number of partitions for which to compile.
    device_assignment: Optional ndarray of jax devices indicating the assignment
      of logical replicas to physical devices (default inherited from
      xla_client.CompileOptions). Must be consistent with `num_replicas` and
      `num_partitions`.
    use_spmd_partitioning: boolean indicating whether to enable SPMD or MPMD
      partitioning in XLA.
    use_auto_spmd_partitioning: boolean indicating whether to automatically
      generate XLA shardings for SPMD partitioner.
    auto_spmd_partitioning_mesh_shape: device mesh shape used to create
      auto_spmd_partitioning search space.
    auto_spmd_partitioning_mesh_ids: device ids used to create
      auto_spmd_partitioning search space.
    env_options_overrides: dict of additional options parsed by the compiler
    fdo_profile: Optional profile for feedback-directed optimization passed to
      XLA.
    detailed_logging: Is this an "interesting" computation about which XLA
      would be wise to log compilation information?
    backend: the client, if available.
  """
  compile_options = xc.CompileOptions()
  compile_options.num_replicas = num_replicas
  compile_options.num_partitions = num_partitions
  build_options = compile_options.executable_build_options
  build_options.use_spmd_partitioning = use_spmd_partitioning
  build_options.use_auto_spmd_partitioning = use_auto_spmd_partitioning
  if fdo_profile is not None:
    build_options.fdo_profile = fdo_profile
  if use_auto_spmd_partitioning:
    build_options.auto_spmd_partitioning_mesh_shape = auto_spmd_partitioning_mesh_shape or []
    build_options.auto_spmd_partitioning_mesh_ids = auto_spmd_partitioning_mesh_ids or []
  if device_assignment is not None:
    logger.debug(
        'get_compile_options: num_replicas=%s num_partitions=%s device_assignment=%s',
        num_replicas, num_partitions, device_assignment)
    device_assignment = np.array(device_assignment)

    # Allow 1D device assignment if num_partitions is 1.
    if (device_assignment.ndim == 1) and (num_partitions == 1):
      device_assignment = device_assignment[:, None]

    if num_replicas != device_assignment.shape[0]:
      msg = 'device_assignment does not match num_replicas: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_replicas))

    if num_partitions != device_assignment.shape[1]:
      msg = 'device_assignment does not match num_partitions: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_partitions))

    if device_assignment.dtype == object:
      device_assignment = np.vectorize(lambda d: d.id, otypes=[int])(
          device_assignment)
    device_assignment = xc.DeviceAssignment.create(device_assignment)
    assert device_assignment.replica_count() == num_replicas
    assert device_assignment.computation_count() == num_partitions
    compile_options.device_assignment = device_assignment

  if env_options_overrides is not None:
    compile_options.env_option_overrides = list(env_options_overrides.items())

  debug_options = compile_options.executable_build_options.debug_options
  if lib.cuda_path is not None:
    debug_options.xla_gpu_cuda_data_dir = lib.cuda_path

  if _DISABLE_MOST_OPTIMIZATIONS.value:
    debug_options.xla_backend_optimization_level = 0
    debug_options.xla_llvm_disable_expensive_passes = True
    debug_options.xla_test_all_input_layouts = False

  # XLA-AutoFDO profile version: precedence order is:
  # 1. Whatever --jax_xla_profile_version is set to.
  # 2. If --jax_xla_profile_version is not set (i.e., 0), call the function
  #    set in get_latest_profile_version and use the return value if non-zero.
  #    If the function returns 0, set -1; this is an error.
  # -1 indicates that no attempt should be made to retrieve the latest profile
  # later on.
  jax_xla_profile_version = config.jax_xla_profile_version.value
  if jax_xla_profile_version > 0:
    compile_options.profile_version = jax_xla_profile_version
    logger.debug("get_compile_options XLA-AutoFDO profile: " +
                 "using JAX XLA profile version %d from flag",
                 jax_xla_profile_version)
  else:
    compile_options.profile_version = _NO_PROFILE_DONT_RETRIEVE
    if backend is None:
      logging.info("get_compile_options: no backend supplied; "
                   "disabling XLA-AutoFDO profile")
    else:
      fdo_profile_version = get_latest_profile_version(backend)
      if fdo_profile_version != 0:
        compile_options.profile_version = fdo_profile_version
        logger.debug("get_compile_options XLA-AutoFDO profile: " +
                     "using XLA-AutoFDO profile version %d",
                     fdo_profile_version)
      else:
        logger.error("get_compile_options XLA-AutoFDO profile: " +
                     "XLA-AutoFDO profile version is 0; this should not happen")

  debug_options.xla_detailed_logging = detailed_logging

  return compile_options

@profiler.annotate_function
def backend_compile(
    backend: xc.Client,
    module: ir.Module,
    options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
) -> xc.LoadedExecutable:
  # Convert ir.Module to a string representation, unless the backend
  # explicitly flags the ability to handle a module directly (avoiding the
  # overhead of back and forth conversions).
  # TODO(slebedev): Change the backend.compile() to accept ir.Module.
  built_c: Any
  if getattr(backend, "needs_str_ir", True):
    built_c = mlir.module_to_bytecode(module)
  else:
    built_c = module

  # we use a separate function call to ensure that XLA compilation appears
  # separately in Python profiling results
  if host_callbacks:
    return backend.compile(built_c, compile_options=options,
                           host_callbacks=host_callbacks)
  # Some backends don't have `host_callbacks` option yet
  # TODO(sharadmv): remove this fallback when all backends allow `compile`
  # to take in `host_callbacks`
  return backend.compile(built_c, compile_options=options)

def compile_or_get_cached(
    backend: xc.Client,
    computation: ir.Module,
    devices: np.ndarray,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
    pgle_profiler: profiler.PGLEProfiler | None = None,
) -> xc.LoadedExecutable:
  sym_name = computation.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value

  if dumped_to := mlir.dump_module_to_file(computation, "compile"):
    logging.info("Dumped the module to %s.", dumped_to)

  # Persistent compilation cache only implemented on TPU and GPU and the backend
  # that supports serialization of executables.
  # TODO(skye): add warning when initializing cache on unsupported default platform
  supported_platforms = ["tpu", "gpu", "cpu"]
  use_compilation_cache = (
      config.enable_compilation_cache.value
      and getattr(backend, "supports_executable_serialization", True)
      and backend.platform in supported_platforms
  )

  if not use_compilation_cache:
    return backend_compile(backend, computation, compile_options,
                           host_callbacks)

  compilation_cache.set_once_cache_used(
      lambda: monitoring.record_event(
          "/jax/compilation_cache/tasks_using_cache"))
  monitoring.record_event('/jax/compilation_cache/compile_requests_use_cache')

  try:
    cache_key = compilation_cache.get_cache_key(
        computation, devices, compile_options, backend)
  except xc._xla.XlaRuntimeError as ex:
    logger.error("compile_or_get_cached: unable to generate cache key, "
                 "skipping the cache: %s", ex)
    return backend_compile(backend, computation, compile_options,
                           host_callbacks)

  is_multi_process = (
      len({device.process_index for device in devices.flatten()}) > 1)
  min_device_process_id = (
      min(devices.flatten(), key=lambda device: device.id).process_index)

  # When PGLE is enabled there might be 3 types of situations:
  # 1. PGLE profiled module (the one which was recompiled with FDO profile) is
  # in the persistent cache. In this case the module should be returned from
  # cache and PGLE should be disabled for this module. Is module is stored in
  # the persistent cache under the "pgle_profiled_module_key" which calculated
  # with replacing FDO profile with flag which identify that module were PGLE
  # profiled.
  # 2. PGLE profiled module is not in the persistent cache and the module is
  # getting built with an FDO profile. In this case we need to share FDO profile
  # with other processes and store the result under the
  # "pgle_profiled_module_key" so later in case 1 we will be able to find the
  # module.
  # 3. PGLE profiled module is not in the persistent cache and the module is
  # getting compiled to be PGLEd (FDO profile is empty). In this case we need to
  # simply return the non-PGLE profiled module from the persistent cache.
  if (config.enable_pgle.value
      and config.pgle_profiling_runs.value > 0):
    fdo_profile = compile_options.executable_build_options.fdo_profile
    compile_options.executable_build_options.fdo_profile = b"pgle profiled"

    pgle_profiled_module_key = compilation_cache.get_cache_key(
        computation, devices, compile_options, backend)
    compile_options.executable_build_options.fdo_profile = fdo_profile

    if _is_executable_in_cache(backend, pgle_profiled_module_key):
      # Load PGLE profiled module from the persistent cache.
      cache_key = pgle_profiled_module_key
      if pgle_profiler is not None:
        pgle_profiler.disable()
    elif fdo_profile is not None and len(fdo_profile) > 0:
      # Store module under PGLE profiled module cache key.
      cache_key = pgle_profiled_module_key
      if is_multi_process and distributed.global_state.client is not None:
        compile_options.executable_build_options.fdo_profile = _share_fdo_profiles(
          computation, devices, compile_options, backend,
          distributed.global_state.client,
          min_device_process_id
        )

  cache_retrieval_start = time.monotonic()
  retrieved_executable, retrieved_compile_time = _cache_read(
      module_name, cache_key, compile_options, backend)
  cache_retrieval_time = time.monotonic() - cache_retrieval_start

  if retrieved_executable is not None:
    assert retrieved_compile_time is not None
    logger.debug("Persistent compilation cache hit for '%s'", module_name)

    monitoring.record_event('/jax/compilation_cache/cache_hits')
    monitoring.record_event_duration_secs(
        '/jax/compilation_cache/compile_time_saved_sec',
        retrieved_compile_time - cache_retrieval_time)

    monitoring.record_event_duration_secs(
        "/jax/compilation_cache/cache_retrieval_time_sec", cache_retrieval_time)

    return retrieved_executable
  elif (
      config.share_binary_between_hosts.value
      and is_multi_process
      and distributed.global_state.client is not None
      # Host callbacks are currently baked into the HLO module so we cant share
      # them.
      and len(host_callbacks) == 0
  ):
    return _compile_and_share_module(
        backend,
        computation,
        compile_options,
        host_callbacks,
        distributed.global_state.client,
        module_name,
        cache_key,
        min_device_process_id
    )
  elif (
      config.share_autotune_config_between_hosts.value
      and is_multi_process
      and distributed.global_state.client is not None
  ):
    return _compile_and_write_autotune_config(
        backend,
        computation,
        compile_options,
        host_callbacks,
        distributed.global_state.client,
        module_name,
        cache_key,
        min_device_process_id
    )
  else:
    return _compile_and_write_cache(
        backend,
        computation,
        compile_options,
        host_callbacks,
        module_name,
        cache_key,
    )

# The process that has the lowest device ID should share FDO profile before
# compilation with other processes.
def _share_fdo_profiles(
    computation: ir.Module,
    devices: np.ndarray,
    compile_options: xc.CompileOptions,
    backend: xc.Client,
    global_client: lib.xla_extension.DistributedRuntimeClient,
    min_process_id
) -> bytes | None:
  sym_name = computation.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value
  fdo_profile = compile_options.executable_build_options.fdo_profile
  if fdo_profile is None or len(fdo_profile) == 0:
    return fdo_profile

  compile_options.executable_build_options.fdo_profile = b""
  profile_key = (
      compilation_cache.get_cache_key(
          computation, devices, compile_options, backend
      )
      + "_fdo_sync"
  )
  if profile_key in _share_fdo_profiles.modules_profiles:
    return _share_fdo_profiles.modules_profiles[profile_key]

  share_timeout = config.share_binary_between_hosts_timeout_ms.value
  if distributed.global_state.process_id == min_process_id:
    logger.debug(
        "Sharing FDO profile: %s. For module %s. Process %d.",
        fdo_profile,
        module_name,
        min_process_id,
    )
    global_client.key_value_set_bytes(profile_key, fdo_profile)
  else:
    logger.debug(
        "Waiting for FDO profile: %s. For module %s. Should be set by process %d.",
        fdo_profile,
        module_name,
        min_process_id,
    )
    fdo_profile = global_client.blocking_key_value_get_bytes(
        profile_key, share_timeout
    )

  _share_fdo_profiles.modules_profiles[profile_key] = fdo_profile
  return fdo_profile


_share_fdo_profiles.modules_profiles = {}


# The process with the first_process_id should compile the module and write an
# autotune config to the K-V storage.
def _compile_and_write_autotune_config(
    backend: xc.Client,
    computation: ir.Module,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
    global_client: lib.xla_extension.DistributedRuntimeClient,
    module_name: str,
    cache_key: str,
    first_process_id: int
) -> xc.LoadedExecutable:
  share_timeout = config.share_binary_between_hosts_timeout_ms.value
  debug_options = compile_options.executable_build_options.debug_options

  if _compile_and_write_autotune_config.autotune_configs_dir is None:
    _compile_and_write_autotune_config.autotune_configs_dir = tempfile.mkdtemp()

  autotune_tmp_file = os.path.join(
      _compile_and_write_autotune_config.autotune_configs_dir, cache_key
  )

  if os.path.exists(autotune_tmp_file):
    logger.debug(
        "Compiling module: %s. Use existing autotune config file: %s",
        module_name,
        autotune_tmp_file,
    )
    debug_options.xla_gpu_load_autotune_results_from = autotune_tmp_file
    return _compile_and_write_cache(
        backend,
        computation,
        compile_options,
        host_callbacks,
        module_name,
        cache_key,
    )

  if distributed.global_state.process_id == first_process_id:
    debug_options.xla_gpu_dump_autotune_results_to = autotune_tmp_file
    logger.debug("Process %d compiling and dumping autotune for module: %s",
                 first_process_id, module_name)
    executable = _compile_and_write_cache(
        backend,
        computation,
        compile_options,
        host_callbacks,
        module_name,
        cache_key,
    )

    logger.debug(
        "Writing autotune config for module %s to %s",
        module_name,
        autotune_tmp_file,
    )
    with open(autotune_tmp_file, "rb") as f:
      autotune_config = f.read()

    autotune_config = compilation_cache.compress_executable(autotune_config)
    global_client.key_value_set_bytes(cache_key, autotune_config)
    logger.debug(
        "Autotune config for module %s with size %d shared by cache_key %s",
        module_name,
        len(autotune_config),
        cache_key,
    )
  else:
    logger.debug(
        "Compiling module %s, waiting for config to be shared by cache_key %s"
        "from process %d",
        module_name,
        cache_key,
        first_process_id
    )
    autotune_config = global_client.blocking_key_value_get_bytes(
        cache_key, share_timeout
    )

    logger.debug(
        "Received autotune config for module %s of size %d",
        module_name,
        len(autotune_config),
    )
    autotune_config = compilation_cache.decompress_executable(autotune_config)
    with open(autotune_tmp_file, "wb") as f:
      f.write(autotune_config)

    logger.debug(
        "Compiling module %s, using autotune config from %s",
        module_name,
        autotune_tmp_file,
    )
    debug_options.xla_gpu_load_autotune_results_from = autotune_tmp_file
    executable = _compile_and_write_cache(
        backend,
        computation,
        compile_options,
        host_callbacks,
        module_name,
        cache_key,
    )
  return executable

_compile_and_write_autotune_config.autotune_configs_dir = None

# The process with the first_process_id should compile the module and write it
# to the K-V storage.
def _compile_and_share_module(
    backend: xc.Client,
    computation: ir.Module,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
    global_client: lib.xla_extension.DistributedRuntimeClient,
    module_name: str,
    cache_key: str,
    first_process_id: int
) -> xc.LoadedExecutable:
  share_timeout = config.share_binary_between_hosts_timeout_ms.value

  # TODO: We need a proper eviction protocol here, otherwise all compiled
  # modules will pile in memory.
  if cache_key in _compile_and_share_module.modules_cache:
    return _compile_and_share_module.modules_cache[cache_key]

  if distributed.global_state.process_id == first_process_id:
    logger.debug("Process %d compiling and sharing module: %s",
                 first_process_id, module_name)
    executable = _compile_and_write_cache(
        backend,
        computation,
        compile_options,
        host_callbacks,
        module_name,
        cache_key,
    )
    serialized_executable = backend.serialize_executable(executable)
    serialized_executable = compilation_cache.compress_executable(
        serialized_executable
    )
    global_client.key_value_set_bytes(cache_key, serialized_executable)
  else:
    logger.debug("Waiting for module: %s from process %d", module_name,
                 first_process_id)
    serialized_executable = global_client.blocking_key_value_get_bytes(
        cache_key, share_timeout
    )
    serialized_executable = compilation_cache.decompress_executable(
        serialized_executable
    )
    executable = backend.deserialize_executable(
        serialized_executable, compile_options
    )

  _compile_and_share_module.modules_cache[cache_key] = executable
  return executable

_compile_and_share_module.modules_cache = {}

def _compile_and_write_cache(
    backend: xc.Client,
    computation: ir.Module,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
    module_name: str,
    cache_key: str,
) -> xc.LoadedExecutable:
  start_time = time.monotonic()
  executable = backend_compile(
      backend, computation, compile_options, host_callbacks
  )
  compile_time = time.monotonic() - start_time
  _cache_write(
      cache_key, compile_time, module_name, backend, executable, host_callbacks
  )
  return executable

def _is_executable_in_cache(backend, cache_key) -> bool:
  """Checks if executable is presented in cache on a given key
  """
  try:
    return compilation_cache.is_executable_in_cache(backend, cache_key)
  except Exception as ex:
    if config.raise_persistent_cache_errors.value:
      raise
    warnings.warn(
        f"Error reading persistent compilation cache entry for "
        f"'{cache_key}': {type(ex).__name__}: {ex}")
    return False


def _cache_read(
    module_name: str, cache_key: str, compile_options: xc.CompileOptions,
    backend: xc.Client
) -> tuple[xc.LoadedExecutable | None, int | None]:
  """Looks up the `computation` and it's compilation time in the persistent
  compilation cache repository.
  """
  try:
    return compilation_cache.get_executable_and_time(
        cache_key, compile_options, backend)
  except Exception as ex:
    if config.raise_persistent_cache_errors.value:
      raise
    warnings.warn(
        f"Error reading persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")
    return None, None


def _cache_write(cache_key: str,
                 compile_time_secs: float,
                 module_name: str,
                 backend: xc.Client, executable: xc.LoadedExecutable,
                 host_callbacks: Sequence[Any]) -> None:
  """Writes the `serialized_computation` and its compilation time to the
  persistent compilation cache repository.
  """
  # Only write cache entries from the first process. Otherwise we create
  # problems with contention for writes on some filesystems, e.g., GCS.
  if distributed.global_state.process_id != 0:
    logger.debug("Not writing persistent cache entry since process_id != 0")
    return

  if host_callbacks:
    logger.debug(
        "Not writing persistent cache entry for '%s' because it uses host "
        "callbacks (e.g. from jax.debug.print or breakpoint)", module_name)
    return

  min_compile_time = config.persistent_cache_min_compile_time_secs.value
  if compile_time_secs < min_compile_time:
    logger.debug(
        "Not writing persistent cache entry for '%s' because it took < %.2f "
        "seconds to compile (%.2fs)", module_name, min_compile_time,
        compile_time_secs)
    return
  else:
    logger.debug(
        "'%s' took at least %.2f seconds to compile (%.2fs)",
        module_name, min_compile_time, compile_time_secs)

  try:
    compilation_cache.put_executable_and_time(
        cache_key, module_name, executable, backend, int(compile_time_secs))
  except Exception as ex:
    if config.raise_persistent_cache_errors.value:
      raise
    warnings.warn(
        f"Error writing persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")
