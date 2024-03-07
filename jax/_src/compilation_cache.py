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

from __future__ import annotations

import logging
import threading
import warnings
import zlib

import numpy as np

# If zstandard is installed, we use zstd compression, otherwise we use zlib.
try:
  import zstandard
except ImportError:
  zstandard = None

from jax._src import cache_key
from jax._src.compilation_cache_interface import CacheInterface
from jax._src import config
from jax._src import monitoring
from jax._src.gfile_cache import GFileCache
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.lru_cache import LRUCache


logger = logging.getLogger(__name__)

_cache: CacheInterface | None = None

_cache_initialized: bool = False

_cache_used: bool = False

# Mutex to protect _cache_initialized and _cache_used.
_cache_initialized_mutex = threading.Lock()

_UNSUPPORTED_RUNTIMES: set[str] = set()

def set_once_cache_used(f) -> None:
  """One-time setting of _cache_used.

  If _cache_used is False, set it to True and execute the provided function
  f. No action if _cache_used is True. This provides a mechanism to execute f
  once per task. Note that reset_cache() will reset _cache_used also.
  """
  global _cache_used
  with _cache_initialized_mutex:
    if not _cache_used:
      _cache_used = True
      if f is not None:
        f()


def get_file_cache(path: str) -> tuple[CacheInterface, str] | None:
  """Returns the file cache and the path to the cache."""

  def is_local_filesystem(path: str) -> bool:
    return path.startswith("file://") or "://" not in path

  # `LRUCache` currently only supports local filesystem. Therefore, if `path`
  # is not on a local filesystem, instead of using `LRUCache`, we
  # fallback to the old `GFileCache`, which does not support LRU eviction.
  # TODO(ayx): Add cloud storage support for `LRUCache`, so that all these code
  # can be removed.
  if not is_local_filesystem(path):
    return GFileCache(path), path

  max_size = config.compilation_cache_max_size.value
  return LRUCache(path, max_size=max_size), path


def set_cache_dir(path) -> None:
  """
  Sets the persistent compilation cache directory.

  After calling this, jit-compiled functions are saved to `path`, so they
  do not need be recompiled if the process is restarted or otherwise run again.
  This also tells Jax where to look for compiled functions before compiling.
  """
  config.config.update("jax_compilation_cache_dir", path)


def initialize_cache(path) -> None:
  """
  This API is deprecated; use set_cache_dir instead.

  Set the path. To take effect, should be called prior to any calls to
  get_executable_and_time() and put_executable_and_time().
  """
  warnings.warn("initialize_cache is deprecated; use set_cache_dir instead",
                DeprecationWarning, stacklevel=2)
  config.config.update("jax_compilation_cache_dir", path)


def default_min_cache_entry_size() -> int:
  """Returns the minimum size below which the entry should not be cached."""
  return 0


def _is_cache_enabled() -> bool:
  return config.enable_compilation_cache.value


def _initialize_cache() -> None:
  # Attempt to initialize the cache at most once.
  global _cache_initialized
  with _cache_initialized_mutex:
    if _cache_initialized:
      return
    _cache_initialized = True

    # Nothing to do if the cache is disabled.
    if not _is_cache_enabled():
      logger.debug("_initialize_cache: cache is disabled!")
      return

    # Set the minimum cache size entry only if the flag
    # --jax_persistent_cache_min_entry_size_bytes has not been set.
    if config.persistent_cache_min_entry_size_bytes.value == 0:
      config.config.update("jax_persistent_cache_min_entry_size_bytes",
                           default_min_cache_entry_size())

    global _cache
    assert _cache is None, "The cache has already been initialized!"
    path: str | None = config.compilation_cache_dir.value
    # If the path is not set, the cache will not be enabled.
    if not path:
      return

    cache_and_path = get_file_cache(path)
    if cache_and_path is None:
      logger.debug("_initialize_cache: cache initialization failed!")
    else:
      _cache, path = cache_and_path
      logger.debug("Initialized persistent compilation cache at %s", path)


def _get_cache(backend) -> CacheInterface | None:
  # TODO(b/289098047): consider making this an API and changing the callers of
  # get_executable_and_time() and put_executable_and_time() to call get_cache()
  # and passing the result to them.
  if backend.runtime_type in _UNSUPPORTED_RUNTIMES:
    logger.debug("_get_cache: Unsupported runtime: %s", backend.runtime_type)
    return None
  if _cache is None:
    _initialize_cache()  # initialization is done at most once; see above
  return _cache


def compress_executable(executable):
  if zstandard:
    compressor = zstandard.ZstdCompressor()
    return compressor.compress(executable)
  else:
    return zlib.compress(executable)

def decompress_executable(executable):
  if zstandard:
    decompressor = zstandard.ZstdDecompressor()
    return decompressor.decompress(executable)
  else:
    return zlib.decompress(executable)


def is_executable_in_cache(backend, cache_key: str) -> bool:
  """Checks if the executable is in the cache."""
  cache = _get_cache(backend)
  if cache is None:
    return False

  # TODO(patrios): add check cache key method to cache interface.
  executable_and_time = cache.get(cache_key)
  return executable_and_time is not None


def get_executable_and_time(
    cache_key: str, compile_options, backend
) -> tuple[xla_client.LoadedExecutable | None, int | None]:
  """Returns the cached executable and its compilation time if present, or None
  otherwise.
  """
  cache = _get_cache(backend)
  if cache is None:
    logger.debug("get_executable_and_time: cache is disabled/not initialized")
    return None, None
  executable_and_time = cache.get(cache_key)
  if not executable_and_time:
    return None, None

  executable_and_time = decompress_executable(executable_and_time)
  serialized_executable, compile_time = extract_executable_and_time(
      executable_and_time)
  xla_executable_deserialized = backend.deserialize_executable(
      serialized_executable, compile_options)
  return xla_executable_deserialized, compile_time


def put_executable_and_time(
    cache_key: str,
    module_name: str,
    executable: xla_client.LoadedExecutable,
    backend,
    compile_time: int
) -> None:
  """Adds the 'executable' and its compilation time to the cache, possibly
  evicting older entries.
  """
  cache = _get_cache(backend)
  if cache is None:
    logger.debug("put_executable_and_time: cache is disabled/not initialized")
    return

  serialized_executable = backend.serialize_executable(executable)
  executable_and_time = combine_executable_and_time(
      serialized_executable, compile_time)
  executable_and_time = compress_executable(executable_and_time)

  min_entry_size = config.persistent_cache_min_entry_size_bytes.value
  entry_size = len(executable_and_time)
  if entry_size < min_entry_size:
    logger.info(
        "Not writing cache entry with key %s since its size (%d bytes) "
        "is less than threshold (%d bytes)",
        cache_key,
        entry_size,
        min_entry_size,
    )
  else:
    logger.info(
        "Writing %s to persistent compilation cache with key %s.",
        module_name,
        cache_key
    )
    monitoring.record_event('/jax/compilation_cache/cache_misses')
    cache.put(cache_key, executable_and_time)


def get_cache_key(module: ir.Module, devices: np.ndarray, compile_options,
                  backend) -> str:
  return cache_key.get(module, devices, compile_options, backend,
                       "zstandard" if zstandard is not None else "zlib")


def is_initialized() -> bool:
  """
  Deprecated.

  Return whether the cache is enabled. Initialization can be deferred, so
  initialized status is not checked. The name is retained for backwards
  compatibility.
  """
  warnings.warn("is_initialized is deprecated; do not use",
                DeprecationWarning, stacklevel=2)
  return _is_cache_enabled()


def reset_cache() -> None:
  """Get back to pristine, uninitialized state."""
  global _cache
  global _cache_initialized
  global _cache_used
  logger.info("Resetting cache at %s.",
               _cache._path if _cache is not None else "<empty>")
  _cache = None
  with _cache_initialized_mutex:
    _cache_initialized = False
    _cache_used = False


def combine_executable_and_time(
    serialized_executable: bytes, compile_time: int
) -> bytes:
  """Given the serialized executable and the compilation time, produce a cache
  entry in the format shown below.

  The cache entry is of the form:
  Byte:     0    1    2    3    4 ...
  Content:  compilation time    serialized executable
            (big-endian int)
  """
  return int(compile_time).to_bytes(4, byteorder='big') + serialized_executable


def extract_executable_and_time(
    exectuable_and_time: bytes
) -> tuple[bytes, int]:
  """Given the cache entry in the format shown below, extract the serialized
  executable and the compilation time.

  The cache entry 'executable_and_time' is of the form:
  Byte:     0    1    2    3    4 ...
  Content:  compilation time    serialized executable
            (big-endian int)
  """
  return exectuable_and_time[4:], int.from_bytes(
      exectuable_and_time[:4], byteorder='big')
