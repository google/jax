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

import logging
import pathlib

logger = logging.getLogger(__name__)

try:
  import etils.epath as epath
except:
  epath = None

# If etils.epath (aka etils[epath] to pip) is present, we prefer it because it
# can read and write to, e.g., GCS buckets. Otherwise we use the builtin
# pathlib and can only read/write to the local filesystem.
if not epath:
  logger.debug("etils.epath was not found. Using pathlib for file I/O.")
  Path = pathlib.Path

else:
  logger.debug("etils.epath found. Using etils.epath for file I/O.")

  class Path:
    """A wrapper class of that can be either a `pathlib.Path` or
    `etils.epath.Path`. If etils is installed, `etils.epath.Path` will be used.
    If etils is not installed, the built-in `pathlib.Path` will be used.
    """

    def __init__(self, *args, **kwargs):
      self._path = epath.Path(*args, **kwargs)

    def __str__(self):
      return str(self._path)
    
    def __repr__(self):
      return repr(self._path)

    def __fspath__(self):
      return self._path.__fspath__()

    def __eq__(self, other):
      return self._path == other._path

    def __truediv__(self, other):
      return self._path / other

    def __rtruediv__(self, other):
      return other / self._path

    def __iter__(self):
      return iter(self._path)

    def __len__(self):
      return len(self._path)

    def __getattr__(self, name):
      if name == "stat":
          return self._overridden_stat
      return getattr(self._path, name)

    def _overridden_stat(self, *args, **kwargs):
      raise NotImplementedError
      # return self._path.stat(*args, **kwargs)
