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

import logging
import threading
from typing import Optional
import zlib

import numpy as np

# If zstandard is installed, we use zstd compression, otherwise we use zlib.
try:
  import zstandard
except ImportError:
  zstandard = None

from jax._src import cache_key
from jax._src.compilation_cache_interface import CacheInterface
from jax._src.config import config
from jax._src.gfile_cache import GFileCache
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir


logger = logging.getLogger(__name__)

_cache: Optional[CacheInterface] = None

_cache_initialized: bool = False

_cache_initialized_mutex = threading.Lock()


def get_file_cache(path: str) -> CacheInterface:
  return GFileCache(path)


def initialize_cache(path) -> None:
  """
  Set the path. To take effect, should be called prior to any calls to
  get_executable_and_time() and put_executable_and_time().
  """
  config.update("jax_compilation_cache_dir", path)


def _is_cache_enabled() -> bool:
  return config.jax_enable_compilation_cache


def _initialize_cache() -> None:
  # Attempt to initialize the cache at most once.
  global _cache_initialized
  with _cache_initialized_mutex:
    if _cache_initialized:
      logger.debug("_initialize_cache: cache has already been initialized!")
      return
    _cache_initialized = True

    # Nothing to do if the cache is disabled.
    if not _is_cache_enabled():
      logger.debug("_initialize_cache: cache is disabled!")
      return

    global _cache
    assert _cache is None, "The cache has already been initialized!"
    path: str = config.jax_compilation_cache_dir
    # If the path is not set, the cache will not be enabled.
    if not path:
      return

    _cache = get_file_cache(path)
    logger.info("Initialized persistent compilation cache at %s", path)


def _get_cache() -> Optional[CacheInterface]:
  # TODO(b/289098047): consider making this an API and changing the callers of
  # get_executable_and_time() and put_executable_and_time() to call get_cache()
  # and passing the result to them.
  if _cache is None:
    _initialize_cache()  # initialization is done at most once; see above
  return _cache


def get_executable_and_time(
    cache_key: str, compile_options, backend
) -> tuple[Optional[xla_client.LoadedExecutable], Optional[int]]:
  """Returns the cached executable and its compilation time if present, or None
  otherwise.
  """
  cache = _get_cache()
  if cache is None:
    logger.debug("get_executable_and_time: cache is disabled/not initialized")
    return None, None
  executable_and_time = cache.get(cache_key)
  if not executable_and_time:
    return None, None
  if zstandard:
    decompressor = zstandard.ZstdDecompressor()
    executable_and_time = decompressor.decompress(executable_and_time)
  else:
    executable_and_time = zlib.decompress(executable_and_time)
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
  cache = _get_cache()
  if cache is None:
    logger.debug("put_executable_and_time: cache is disabled/not initialized")
    return
  logger.info(
      "Writing %s to persistent compilation cache with key %s.",
      module_name,
      cache_key,
  )
  serialized_executable = backend.serialize_executable(executable)
  executable_and_time = combine_executable_and_time(
      serialized_executable, compile_time)
  if zstandard:
    compressor = zstandard.ZstdCompressor()
    executable_and_time = compressor.compress(executable_and_time)
  else:
    executable_and_time = zlib.compress(executable_and_time)
  cache.put(cache_key, executable_and_time)


def get_cache_key(module: ir.Module, devices: np.ndarray, compile_options,
                  backend, produce_original_cache_key: bool = True) -> str:
  return cache_key.get(module, devices, compile_options, backend,
                       "zstandard" if zstandard is not None else "zlib",
                       produce_original_cache_key)


def is_initialized() -> bool:
  """
  Return whether the cache is enabled. Initialization can be deferred, so
  initialized status is not checked. The name is retained for backwards
  compatibility.
  """
  return _is_cache_enabled()


def reset_cache() -> None:
  """Get back to pristine, uninitialized state."""
  global _cache
  global _cache_initialized
  logger.info("Resetting cache at %s.",
               _cache._path if _cache is not None else "<empty>")
  _cache = None
  _cache_initialized = False


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
