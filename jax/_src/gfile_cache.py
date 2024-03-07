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

import os

from jax._src import path as pathlib
from jax._src.compilation_cache_interface import CacheInterface


# TODO (ayx): This class will be ultimately removed after `lru_cache.py` is
# finished. It exists because the current `lru_cache.py` does not support
# `gs://`.
class GFileCache(CacheInterface):

  def __init__(self, path: str):
    """Sets up a cache at 'path'. Cached values may already be present."""
    self._path = pathlib.Path(path)
    self._path.mkdir(parents=True, exist_ok=True)

  def get(self, key: str):
    """Returns None if 'key' isn't present."""
    if not key:
      raise ValueError("key cannot be empty")
    path_to_key = self._path / key
    if path_to_key.exists():
      return path_to_key.read_bytes()
    else:
      return None

  def put(self, key: str, value: bytes):
    """Adds new cache entry."""
    if not key:
      raise ValueError("key cannot be empty")
    path_to_new_file = self._path / key
    if str(path_to_new_file).startswith('gs://'):
      # Writes to gcs are atomic.
      path_to_new_file.write_bytes(value)
    elif str(path_to_new_file).startswith('file://') or '://' not in str(path_to_new_file):
      tmp_path = self._path / f"_temp_{key}"
      with open(str(tmp_path), "wb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
      os.replace(tmp_path, path_to_new_file)
    else:
      tmp_path = self._path / f"_temp_{key}"
      tmp_path.write_bytes(value)
      tmp_path.replace(str(path_to_new_file))
