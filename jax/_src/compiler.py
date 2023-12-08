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
import io
import itertools
import time
from typing import Any
import logging
import os
import re
import warnings

import numpy as np

from jax._src import lib
from jax._src import compilation_cache
from jax._src import config as config
from jax._src import monitoring
from jax._src import path
from jax._src import profiler
from jax._src import traceback_util
from jax._src.lib.mlir import ir
from jax._src.lib import xla_client as xc


_DISABLE_MOST_OPTIMIZATIONS = config.DEFINE_bool(
    'jax_disable_most_optimizations',
    config.bool_env('JAX_DISABLE_MOST_OPTIMIZATIONS', False),
    'Try not to do much optimization work. This can be useful if the cost of '
    'optimization is greater than that of running a less-optimized program.')

_DUMP_IR_TO = config.DEFINE_string(
    'jax_dump_ir_to', os.getenv('JAX_DUMP_IR_TO', ''),
    help="Path to which the IR that is emitted by JAX as input to the "
         "compiler should be dumped as text files. Optional. If omitted, JAX "
         "will not dump IR.")

_COMPILER_DETAILED_LOGGING_MIN_OPS = config.DEFINE_integer(
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

# This variable captures whether a process has ever used cache. It will be set
# to true only once, regardless of how many times compile_or_get_cached() or
# compilation_cache.reset_cache() is called.
_cache_used: bool = False


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


def _module_to_string(module: ir.Module) -> str:
  output = io.StringIO()
  module.operation.print(file=output, enable_debug_info=True)
  return output.getvalue()

def _module_to_bytecode(module: ir.Module) -> bytes:
  output = io.BytesIO()
  module.operation.write_bytecode(file=output)
  return output.getvalue()


@profiler.annotate_function
def backend_compile(
    backend: xc.Client,
    module: ir.Module,
    options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
) -> xc.LoadedExecutable:
  # Convert ir.Module to a string representation, unless the
  # back-end expliclity flags the ability to handle a module directly
  # (avoiding the overhead of back and forth conversions)
  if getattr(backend, "needs_str_ir", True):
    built_c = _module_to_bytecode(module)
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


_ir_dump_counter = itertools.count()

def _make_string_safe_for_filename(s: str) -> str:
  return re.sub(r'[^\w.)( -]', '', s)

def _dump_ir_to_file(name: str, ir: str):
  id = next(_ir_dump_counter)
  name = f"jax_ir{id}_{_make_string_safe_for_filename(name)}.mlir"
  out_dir = path.Path(_DUMP_IR_TO.value)
  out_dir.mkdir(parents=True, exist_ok=True)
  name = out_dir / name
  name.write_text(ir)


def compile_or_get_cached(
    backend: xc.Client,
    computation: ir.Module,
    devices: np.ndarray,
    compile_options: xc.CompileOptions,
    host_callbacks: Sequence[Any],
) -> xc.LoadedExecutable:
  sym_name = computation.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value

  if _DUMP_IR_TO.value:
    _dump_ir_to_file(module_name, _module_to_string(computation))

  # Persistent compilation cache only implemented on TPU and GPU.
  # TODO(skye): add warning when initializing cache on unsupported default platform
  supported_platforms = ["tpu", "gpu"]
  # (b/233850967) CPU caching can be enabled if XLA Runtime is enabled.
  if "--xla_cpu_use_xla_runtime=true" in os.environ.get("XLA_FLAGS", ""):
    supported_platforms.append("cpu")
  use_compilation_cache = (compilation_cache.is_initialized() and
                           backend.platform in supported_platforms)

  if not use_compilation_cache:
    return backend_compile(backend, computation, compile_options,
                           host_callbacks)

  global _cache_used
  if not _cache_used:
    _cache_used = True
    monitoring.record_event('/jax/compilation_cache/tasks_using_cache')

  monitoring.record_event('/jax/compilation_cache/compile_requests_use_cache')

  try:
    cache_key = compilation_cache.get_cache_key(
        computation, devices, compile_options, backend,
        config.use_original_compilation_cache_key_generation.value,
    )
  except xc._xla.XlaRuntimeError as ex:
    logger.warning("compile_or_get_cached: unable to generate cache key, "
                   "skipping the cache. Please report this to the JAX team. "
                   "Error: %s", ex)
    return backend_compile(backend, computation, compile_options,
                           host_callbacks)

  cache_retrieval_start = time.monotonic()
  retrieved_executable, retrieved_compile_time = _cache_read(
      module_name, cache_key, compile_options, backend)
  cache_retrieval_time = time.monotonic() - cache_retrieval_start

  if retrieved_executable is not None:
    assert retrieved_compile_time is not None
    logger.info("Persistent compilation cache hit for '%s'", module_name)

    if config.use_original_compilation_cache_key_generation.value:
      # TODO(b/293308239) Remove metrics for the original cache after the new
      # compilation cache key implementation is fully rolled out.
      monitoring.record_event('/jax/compilation_cache/cache_hits_original')
      monitoring.record_event_duration_secs(
          "/jax/compilation_cache/original_compile_time_saved_sec",
          retrieved_compile_time - cache_retrieval_time)
    else:
      monitoring.record_event('/jax/compilation_cache/cache_hits')
      monitoring.record_event_duration_secs(
          '/jax/compilation_cache/compile_time_saved_sec',
          retrieved_compile_time - cache_retrieval_time)

    monitoring.record_event_duration_secs(
        "/jax/compilation_cache/cache_retrieval_time_sec", cache_retrieval_time)

    return retrieved_executable
  else:
    start_time = time.monotonic()
    executable = backend_compile(backend, computation,
                                compile_options, host_callbacks)
    compile_time = time.monotonic() - start_time
    _cache_write(cache_key, compile_time, module_name, backend, executable,
                 host_callbacks)
    return executable


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
  if backend.process_index() != 0:
    logger.debug("Not writing persistent cache entry since process_index != 0")
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
        "'%s' took at least %.2f seconds to compile (%.2fs), writing persistent"
        " cache entry", module_name, min_compile_time, compile_time_secs)
    monitoring.record_event('/jax/compilation_cache/cache_misses')

  try:
    compilation_cache.put_executable_and_time(
        cache_key, module_name, executable, backend, int(compile_time_secs))
  except Exception as ex:
    if config.raise_persistent_cache_errors.value:
      raise
    warnings.warn(
        f"Error writing persistent compilation cache entry for "
        f"'{module_name}': {type(ex).__name__}: {ex}")
