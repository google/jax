# Copyright 2024 The JAX Authors.
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

import heapq
import logging
import pathlib
import warnings

from jax._src.compilation_cache_interface import CacheInterface


try:
  import filelock
except ImportError:
  filelock = None


logger = logging.getLogger(__name__)


class LRUCache(CacheInterface):
  """Bounded cache with least-recently-used (LRU) eviction policy.

  This implementation includes cache reading, writing and eviction
  based on the LRU policy.

  Notably, when ``max_size`` is set to -1, the cache eviction
  is disabled, and the LRU cache functions as a normal cache
  without any size limitations.
  """

  def __init__(self, path: str, *, max_size: int, lock_timeout_secs: float | None = 10):
    """Args:

      path: The path to the cache directory.
      max_size: The maximum size of the cache in bytes. Caching will be
        disabled if this value is set to ``0``. A special value of ``-1``
        indicates no limit, allowing the cache size to grow indefinitely.
      lock_timeout_secs: (optional) The timeout for acquiring a file lock.
    """
    # TODO(ayx): add support for cloud other filesystems such as GCS
    if not self._is_local_filesystem(path):
      raise NotImplementedError("LRUCache only supports local filesystem at this time.")

    self.path = pathlib.Path(path)
    self.path.mkdir(parents=True, exist_ok=True)

    # TODO(ayx): having a `self._path` is required by the base class
    # `CacheInterface`, but the base class can be removed after `LRUCache`
    # and the original `GFileCache` are unified
    self._path = self.path

    self.eviction_enabled = max_size != -1  # no eviction if `max_size` is set to -1

    if self.eviction_enabled:
      if filelock is None:
        raise RuntimeError("Please install filelock package to set `jax_compilation_cache_max_size`")

      self.max_size = max_size
      self.lock_timeout_secs = lock_timeout_secs

      self.lock_path = self.path / ".lockfile"
      self.lock = filelock.FileLock(self.lock_path)

  def get(self, key: str) -> bytes | None:
    """Retrieves the cached value for the given key.

    Args:
      key: The key for which the cache value is retrieved.

    Returns:
      The cached data as bytes if available; ``None`` otherwise.
    """
    if not key:
      raise ValueError("key cannot be empty")

    file = self.path / key

    if self.eviction_enabled:
      self.lock.acquire(timeout=self.lock_timeout_secs)

    try:
      if not file.exists():
        logger.debug(f"Cache miss for key: {key!r}")
        return None

      logger.debug(f"Cache hit for key: {key!r}")
      file.touch()  # update mtime
      return file.read_bytes()

    finally:
      if self.eviction_enabled:
        self.lock.release()

  def put(self, key: str, val: bytes) -> None:
    """Adds a new entry to the cache.

    If a cache item with the same key already exists, no action
    will be taken, even if the value is different.

    Args:
      key: The key under which the data will be stored.
      val: The data to be stored.
    """
    if not key:
      raise ValueError("key cannot be empty")

    # prevent adding entries that exceed the maximum size limit of the cache
    if self.eviction_enabled and len(val) > self.max_size:
      msg = (f"Cache value for key {key!r} of size {len(val)} bytes exceeds "
             f"the maximum cache size of {self.max_size} bytes")
      warnings.warn(msg)
      return

    file = self.path / key

    if self.eviction_enabled:
      self.lock.acquire(timeout=self.lock_timeout_secs)

    try:
      if file.exists():
        return

      self._evict_if_needed(additional_size=len(val))
      file.write_bytes(val)

    finally:
      if self.eviction_enabled:
        self.lock.release()

  def _evict_if_needed(self, *, additional_size: int = 0) -> None:
    """Evicts the least recently used items from the cache if necessary
    to ensure the cache does not exceed its maximum size.

    Args:
      additional_size: The size of the new entry being added to the cache.
        This is included to account for the new entry when checking if
        eviction is needed.
    """
    if not self.eviction_enabled:
      return

    # a priority queue, each element is a tuple `(file_mtime, file, file_size)`
    h: list[tuple[int, pathlib.Path, int]] = []
    dir_size = 0
    for file in self.path.iterdir():
      if file.is_file() and file != self.lock_path:
        file_size = file.stat().st_size
        file_mtime = file.stat().st_mtime_ns

        dir_size += file_size
        heapq.heappush(h, (file_mtime, file, file_size))

    target_size = self.max_size - additional_size
    # evict files until the directory size is less than or equal
    # to `target_size`
    while dir_size > target_size:
      file_mtime, file, file_size = heapq.heappop(h)
      msg = (f"Evicting cache file {file.name}: file size {file_size} bytes, "
             f"target cache size {target_size} bytes")
      logger.debug(msg)
      file.unlink()
      dir_size -= file_size

  # See comments in `jax.src.compilation_cache.get_file_cache()` for details.
  # TODO(ayx): This function has a duplicate in that place, and there is
  # redundancy here. However, this code is temporary, and once the issue
  # is fixed, this code can be removed.
  @staticmethod
  def _is_local_filesystem(path: str) -> bool:
    return path.startswith("file://") or "://" not in path
