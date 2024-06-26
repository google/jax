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

import importlib.util
import tempfile
import time

from absl.testing import absltest

from jax._src import path as pathlib
from jax._src.lru_cache import _CACHE_SUFFIX, LRUCache
import jax._src.test_util as jtu


class LRUCacheTestCase(jtu.JaxTestCase):
  name: str | None
  path: pathlib.Path | None

  def setUp(self):
    if importlib.util.find_spec("filelock") is None:
      self.skipTest("filelock is not installed")

    super().setUp()
    tmpdir = tempfile.TemporaryDirectory()
    self.enter_context(tmpdir)
    self.name = tmpdir.name
    self.path = pathlib.Path(self.name)

  def tearDown(self):
    self.path = None
    self.name = None
    super().tearDown()


class LRUCacheTest(LRUCacheTestCase):

  def test_get_nonexistent_key(self):
    cache = LRUCache(self.name, max_size=-1)
    self.assertIsNone(cache.get("a"))

  def test_put_and_get_key(self):
    cache = LRUCache(self.name, max_size=-1)

    cache.put("a", b"a")
    self.assertEqual(cache.get("a"), b"a")
    self.assertEqual(set(self.path.glob(f"*{_CACHE_SUFFIX}")), {self.path / f"a{_CACHE_SUFFIX}"})

    cache.put("b", b"b")
    self.assertEqual(cache.get("a"), b"a")
    self.assertEqual(cache.get("b"), b"b")
    self.assertEqual(set(self.path.glob(f"*{_CACHE_SUFFIX}")), {self.path / f"a{_CACHE_SUFFIX}", self.path / f"b{_CACHE_SUFFIX}"})

  def test_put_empty_value(self):
    cache = LRUCache(self.name, max_size=-1)

    cache.put("a", b"")
    self.assertEqual(cache.get("a"), b"")

  def test_put_empty_key(self):
    cache = LRUCache(self.name, max_size=-1)

    with self.assertRaisesRegex(ValueError, r"key cannot be empty"):
      cache.put("", b"a")

  def test_eviction(self):
    cache = LRUCache(self.name, max_size=2)

    cache.put("a", b"a")
    cache.put("b", b"b")

    # `sleep()` is necessary to guarantee that `b`'s timestamp is strictly greater than `a`'s
    time.sleep(1)
    cache.get("b")

    # write `c`. `a` should be evicted
    cache.put("c", b"c")
    self.assertEqual(set(self.path.glob(f"*{_CACHE_SUFFIX}")), {self.path / f"b{_CACHE_SUFFIX}", self.path / f"c{_CACHE_SUFFIX}"})

    # calling `get()` on `b` makes `c` least recently used
    time.sleep(1)
    cache.get("b")

    # write `d`. `c` should be evicted
    cache.put("d", b"d")
    self.assertEqual(set(self.path.glob(f"*{_CACHE_SUFFIX}")), {self.path / f"b{_CACHE_SUFFIX}", self.path / f"d{_CACHE_SUFFIX}"})

  def test_eviction_with_empty_value(self):
    cache = LRUCache(self.name, max_size=1)

    cache.put("a", b"a")

    # write `b` with length 0
    # eviction should not happen even though the cache is full
    cache.put("b", b"")
    self.assertEqual(set(self.path.glob(f"*{_CACHE_SUFFIX}")), {self.path / f"a{_CACHE_SUFFIX}", self.path / f"b{_CACHE_SUFFIX}"})

    # calling `get()` on `a` makes `b` least recently used
    time.sleep(1)
    cache.get("a")

    # writing `c` should result in evicting the
    # least recent used file (`b`) first,
    # but this is not sufficient to make room for `c`,
    # so `a` should be evicted as well
    cache.put("c", b"c")
    self.assertEqual(set(self.path.glob(f"*{_CACHE_SUFFIX}")), {self.path / f"c{_CACHE_SUFFIX}"})

  def test_existing_cache_dir(self):
    cache = LRUCache(self.name, max_size=2)

    cache.put("a", b"a")

    # simulates reinitializing the cache in another process
    del cache
    cache = LRUCache(self.name, max_size=2)

    self.assertEqual(cache.get("a"), b"a")

    # ensure that the LRU policy survives cache reinitialization
    cache.put("b", b"b")

    # calling `get()` on `a` makes `b` least recently used
    time.sleep(1)
    cache.get("a")

    # write `c`. `b` should be evicted
    cache.put("c", b"c")
    self.assertEqual(set(self.path.glob(f"*{_CACHE_SUFFIX}")), {self.path / f"a{_CACHE_SUFFIX}", self.path / f"c{_CACHE_SUFFIX}"})

  def test_max_size(self):
    cache = LRUCache(self.name, max_size=1)

    msg = (r"Cache value for key .+? of size \d+ bytes exceeds the maximum "
           r"cache size of \d+ bytes")
    with self.assertWarnsRegex(UserWarning, msg):
      cache.put("a", b"aaaa")
    self.assertIsNone(cache.get("a"))
    self.assertEqual(set(self.path.glob(f"*{_CACHE_SUFFIX}")), set())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
