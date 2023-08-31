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

import hashlib
import io
import logging
import os
import sys

import numpy as np

from jax._src.config import config
from jax._src.lib import xla_client
from jax._src.lib import version_str as jaxlib_version_str
from jax._src.lib import version as jaxlib_version
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager as pm

if jaxlib_version < (0, 4, 14):
  import jaxlib.mlir.jax as mlir_jax  # pytype: disable=import-error
  assert mlir_jax is not None  # Imported for side effects only.


logger = logging.getLogger(__name__)

_extra_flag_prefixes: list[str] = []

def add_flag_prefixes(flag_prefixes: list[str]) -> None:
  """Add flag prefixes to include in the cache key. Call prior to get().
  """
  global _extra_flag_prefixes
  _extra_flag_prefixes += flag_prefixes


def clear_flag_prefixes() -> None:
  """Clear flag prefixes added by add_flag_prefixes().
  """
  global _extra_flag_prefixes
  _extra_flag_prefixes = []


def get_flag_prefixes() -> list[str]:
  """Return flag prefixes added by add_flag_prefixes().
  """
  return _extra_flag_prefixes


def get(module: ir.Module,
        devices: np.ndarray,
        compile_options: xla_client.CompileOptions,
        backend: xla_client.Client,
        compression_algorithm: str = "zstandard",
        produce_original_cache_key: bool = True) -> str:
  """Creates a hashed string to use as a key to the compilation cache.

  Creates a cache key that is a hex-encoded string of a unique hash based on
  the arguments. The hex-encoded string is 256 characters long.

  Args:
    module: the input program
    devices: an array of accelerator devices that the program will run on
    compile_options: options passed to the XLA compiler
    backend: description of the platform (e.g., TPU version)
    compression_algorithm: a string representing the compression algorithm used
      for the executable before persisting in the cache
    produce_original_cache_key: if True, the original cache-key generation
      algorithm is run, else the new one. This is transient; once the migration
      is complete, this parameter and the original algorithm will be removed.
      (New one not implemented as yet.)

  Typical return value example:
   '14ac577cdb2ef6d986078b4054cc9893a9a14a16dbb0d8f37b89167c1f1aacdf'
  """
  if produce_original_cache_key:
    entries = [
        ("computation", lambda hash_obj: _hash_computation(hash_obj, module)),
        ("devices", lambda hash_obj: _hash_devices(hash_obj, devices)),
        ("compile_options",
         lambda hash_obj: _hash_compile_options(hash_obj, compile_options)),
        ("jax_lib version",
         lambda hash_obj: hash_obj.update(
             bytes(jaxlib_version_str.encode("utf-8")))),
        ("the backend", lambda hash_obj: _hash_platform(hash_obj, backend)),
        ("XLA flags",
         lambda hash_obj: _hash_xla_flags(hash_obj, get_flag_prefixes())),
        ("compression",
         lambda hash_obj: _hash_string(hash_obj, compression_algorithm)),
    ]
  else:
    entries = [
        ("computation", lambda hash_obj: _hash_computation(hash_obj, module)),
        ("devices", lambda hash_obj: _hash_devices(hash_obj, devices)),
        ("compile_options",
         lambda hash_obj: hash_obj.update(compile_options.SerializeAsString())),
        ("jax_lib version",
         lambda hash_obj: hash_obj.update(
             bytes(jaxlib_version_str.encode("utf-8")))),
        ("the backend", lambda hash_obj: _hash_platform(hash_obj, backend)),
        ("XLA flags",
         lambda hash_obj: _hash_xla_flags(hash_obj, get_flag_prefixes())),
        ("compression",
         lambda hash_obj: _hash_string(hash_obj, compression_algorithm)),
    ]

  hash_obj = hashlib.sha256()
  for name, hashfn in entries:
    hashfn(hash_obj)
    _log_cache_key_hash(hash_obj, name, hashfn)
  return hash_obj.digest().hex()


def _log_cache_key_hash(hash_obj, last_serialized: str, hashfn):
  if logger.isEnabledFor(logging.DEBUG):
    # Log the hash of just this entry
    fresh_hash_obj = hashlib.sha256()
    hashfn(fresh_hash_obj)
    logger.debug(
        "get_cache_key hash of serialized %s: %s",
        last_serialized,
        fresh_hash_obj.digest().hex(),
    )
    # Log the cumulative hash
    logger.debug(
        "get_cache_key hash after serializing %s: %s",
        last_serialized,
        hash_obj.digest().hex(),
    )


def _serialize_ir(m: ir.Module) -> bytes:
  output = io.BytesIO()
  m.operation.write_bytecode(file=output)
  return output.getvalue()


def _canonicalize_ir(m_original: ir.Module) -> bytes:
  with m_original.context:
    m = m_original.operation.clone()
    if jaxlib_version < (0, 4, 14):
      passes = pm.PassManager.parse(
          "builtin.module(func.func(jax-strip-locations))"
      )
    else:
      passes = pm.PassManager.parse(
          "builtin.module(strip-debuginfo)"
      )
    passes.run(m.operation)
    return _serialize_ir(m)


def _hash_computation(hash_obj, module):
  if config.jax_compilation_cache_include_metadata_in_key:
    canonical_ir = _serialize_ir(module)
  else:
    canonical_ir = _canonicalize_ir(module)
  hash_obj.update(canonical_ir)


def _hash_devices(hash_obj, devices: np.ndarray) -> None:
  for device in devices.flat:
    _hash_string(hash_obj, device.device_kind)


def _hash_compile_options(hash_obj, compile_options_obj):
  expected_num_compile_options = 12
  # Ignore private and built-in methods. These can unexpectedly change and lead
  # to false positives, e.g. when different Python versions include different
  # built-ins.
  num_actual_options = len(
      [x for x in dir(compile_options_obj) if not x.startswith("_")]
  )
  assert num_actual_options == expected_num_compile_options, (
      "Unexpected number of CompileOption fields: "
      f"{num_actual_options}. This likely: means that an extra "
      "field was added, and this function needs to be updated."
  )

  if compile_options_obj.argument_layouts is not None:
    map(
        lambda shape: hash_obj.update(shape.to_serialized_proto()),
        compile_options_obj.argument_layouts,
    )
  _hash_int(hash_obj, compile_options_obj.parameter_is_tupled_arguments)
  _hash_executable_build_options(
      hash_obj, compile_options_obj.executable_build_options
  )
  _hash_bool(hash_obj, compile_options_obj.tuple_arguments)
  _hash_int(hash_obj, compile_options_obj.num_replicas)
  _hash_int(hash_obj, compile_options_obj.num_partitions)
  _hash_signed_int(hash_obj, compile_options_obj.profile_version)
  if compile_options_obj.device_assignment is not None:
    hash_obj.update(compile_options_obj.device_assignment.serialize())
  _hash_bool(hash_obj, compile_options_obj.compile_portable_executable)
  _hash_int(hash_obj, len(compile_options_obj.env_option_overrides))
  for kv in compile_options_obj.env_option_overrides:
    _hash_string(hash_obj, kv[0])
    if isinstance(kv[1], str):
      _hash_string(hash_obj, kv[1])
    elif isinstance(kv[1], bool):
      _hash_bool(hash_obj, kv[1])
    elif isinstance(kv[1], int):
      _hash_int(hash_obj, kv[1])
    else:
      raise RuntimeError("Invalid type: %s" % repr(type(kv[1])))


def _hash_executable_build_options(hash_obj, executable_obj):
  expected_options = 11
  # Ignore private and built-in methods. These can unexpectedly change and lead
  # to false positives, e.g. when different Python versions include different
  # built-ins.
  actual_options = len(
      [x for x in dir(executable_obj) if not x.startswith("_")]
  )
  assert actual_options == expected_options, (
      "Unexpected number of executable_build_options fields: "
      f"{actual_options}, expected: {expected_options}. This likely means "
      "that an extra field was added, and this function needs to be updated."
  )
  if executable_obj.result_layout is not None:
    hash_obj.update(executable_obj.result_layout.to_serialized_proto())
  _hash_int(hash_obj, executable_obj.num_replicas)
  _hash_int(hash_obj, executable_obj.num_partitions)
  _hash_debug_options(hash_obj, executable_obj.debug_options)
  if executable_obj.device_assignment is not None:
    hash_obj.update(executable_obj.device_assignment.serialize())
  _hash_bool(hash_obj, executable_obj.use_spmd_partitioning)
  _hash_bool(hash_obj, executable_obj.use_auto_spmd_partitioning)
  if executable_obj.use_auto_spmd_partitioning:
    if executable_obj.auto_spmd_partitioning_mesh_shape is not None:
      _hash_int_list(hash_obj, executable_obj.auto_spmd_partitioning_mesh_shape)
    if executable_obj.auto_spmd_partitioning_mesh_ids is not None:
      _hash_int_list(hash_obj, executable_obj.auto_spmd_partitioning_mesh_ids)
  _hash_bool_list(
      hash_obj, executable_obj.allow_spmd_sharding_propagation_to_output
  )
  if executable_obj.fdo_profile is not None:
    _hash_string(hash_obj, executable_obj.fdo_profile)


def _hash_debug_options(hash_obj, debug_obj):
  _hash_bool(hash_obj, debug_obj.xla_cpu_enable_fast_math)
  _hash_bool(hash_obj, debug_obj.xla_cpu_fast_math_honor_infs)
  _hash_bool(hash_obj, debug_obj.xla_cpu_fast_math_honor_nans)
  _hash_bool(hash_obj, debug_obj.xla_cpu_fast_math_honor_division)
  _hash_bool(hash_obj, debug_obj.xla_cpu_fast_math_honor_functions)
  _hash_bool(hash_obj, debug_obj.xla_gpu_enable_fast_min_max)
  _hash_int(hash_obj, debug_obj.xla_backend_optimization_level)
  _hash_bool(hash_obj, debug_obj.xla_cpu_enable_xprof_traceme)
  _hash_bool(hash_obj, debug_obj.xla_llvm_disable_expensive_passes)
  _hash_bool(hash_obj, debug_obj.xla_test_all_input_layouts)


def _hash_platform(hash_obj, backend):
  _hash_string(hash_obj, backend.platform)
  _hash_string(hash_obj, backend.platform_version)
  _hash_string(hash_obj, backend.runtime_type)


def _hash_xla_flags(hash_obj, extra_flag_prefixes: list[str]):
  xla_flags_to_exclude_from_cache_key = [
      "--xla_dump_compress_protos",
      "--xla_dump_module_metadata",
      "--xla_dump_max_hlo_modules",
      "--xla_dump_include_timestamp",
      "--xla_dump_hlo_pass_re",
      "--xla_dump_hlo_module_re",
      "--xla_dump_hlo_snapshots",
      "--xla_dump_fusion_visualization",
      "--xla_dump_hlo_as_url",
      "--xla_dump_hlo_as_proto",
      "--xla_dump_hlo_as_text",
      "--xla_dump_to",
      "--xla_force_host_platform_device_count",
      "--xla_dump_disable_metadata",
      "--xla_dump_hlo_pipeline_re",
      "--xla_tpu_sdc_checker_streamz_metric",
      "--xla_tpu_sdc_checker_enable_sdc_event_callbacks",
  ]

  xla_flags = []

  xla_flags_env_var = os.getenv("XLA_FLAGS")
  if xla_flags_env_var:
    xla_flags.extend(xla_flags_env_var.split())

  for arg in sys.argv:
    if arg.startswith("--xla") or any(
        arg.startswith(p) for p in extra_flag_prefixes
    ):
      xla_flags.append(arg)

  # N.B. all XLA flags that take an argument must use '=' and not a space
  # (e.g. --xla_force_host_platform_device_count=8) (I think).
  for flag in xla_flags:
    if flag.split("=")[0] in xla_flags_to_exclude_from_cache_key:
      logger.debug("Not including XLA flag in cache key: %s", flag)
      continue
    logger.debug("Including XLA flag in cache key: %s", flag)
    _hash_string(hash_obj, flag)


def _hash_int(hash_obj, int_var):
  hash_obj.update(int_var.to_bytes(8, byteorder="big"))


def _hash_signed_int(hash_obj, int_var):
  hash_obj.update(int_var.to_bytes(8, byteorder="big", signed=True))


def _hash_bool(hash_obj, bool_var):
  hash_obj.update(bool_var.to_bytes(1, byteorder="big"))


def _hash_string(hash_obj, str_var):
  hash_obj.update(str_var.encode("utf-8").strip())


def _hash_bool_list(hash_obj, bool_list):
  for b in bool_list:
    _hash_bool(hash_obj, b)
  _hash_int(hash_obj, len(bool_list))


def _hash_int_list(hash_obj, int_list):
  for i in int_list:
    _hash_int(hash_obj, i)
  _hash_int(hash_obj, len(int_list))
