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
from jax._src.lru_cache import LRUCache
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
    self.assertIsNone(cache.get("cache-a"))

  def test_put_and_get_key(self):
    cache = LRUCache(self.name, max_size=-1)

    cache.put("cache-a", b"a")
    self.assertEqual(cache.get("cache-a"), b"a")
    self.assertEqual(set(self.path.glob("cache-*")), {self.path / "cache-a"})

    cache.put("cache-b", b"b")
    self.assertEqual(cache.get("cache-a"), b"a")
    self.assertEqual(cache.get("cache-b"), b"b")
    self.assertEqual(set(self.path.glob("cache-*")), {self.path / "cache-a", self.path / "cache-b"})

  def test_put_empty_value(self):
    cache = LRUCache(self.name, max_size=-1)

    cache.put("cache-a", b"")
    self.assertEqual(cache.get("cache-a"), b"")

  def test_put_empty_key(self):
    cache = LRUCache(self.name, max_size=-1)

    with self.assertRaisesRegex(ValueError, r"key cannot be empty"):
      cache.put("", b"a")

  def test_eviction(self):
    cache = LRUCache(self.name, max_size=2)

    cache.put("cache-a", b"a")
    cache.put("cache-b", b"b")

    # `sleep()` is necessary to guarantee that `cache-b`"s timestamp is strictly greater than `cache-a`"s
    time.sleep(1)
    cache.get("cache-b")

    # write `cache-c`, evict `cache-a`
    cache.put("cache-c", b"c")
    self.assertEqual(set(self.path.glob("cache-*")), {self.path / "cache-b", self.path / "cache-c"})

    # calling `get()` on `cache-b` makes `cache-c` least recently used
    time.sleep(1)
    cache.get("cache-b")

    # write `cache-d`, evict `cache-c`
    cache.put("cache-d", b"d")
    self.assertEqual(set(self.path.glob("cache-*")), {self.path / "cache-b", self.path / "cache-d"})

  def test_eviction_with_empty_value(self):
    cache = LRUCache(self.name, max_size=1)

    cache.put("cache-a", b"a")

    # write `cache-b` with length 0
    # eviction should not happen even though the cache is full
    cache.put("cache-b", b"")
    self.assertEqual(set(self.path.glob("cache-*")), {self.path / "cache-a", self.path / "cache-b"})

    # calling `get()` on `cache-a` makes `cache-b` least recently used
    time.sleep(1)
    cache.get("cache-a")

    # writing `cache-c` should result in evicting the
    # least recent used file (`cache-b`) first,
    # but this is not sufficient to make room for `cache-c`,
    # so `cache-a` should be evicted as well
    cache.put("cache-c", b"c")
    self.assertEqual(set(self.path.glob("cache-*")), {self.path / "cache-c"})

  def test_existing_cache_dir(self):
    cache = LRUCache(self.name, max_size=2)

    cache.put("cache-a", b"a")

    # simulates reinitializing the cache in another process
    del cache
    cache = LRUCache(self.name, max_size=2)

    self.assertEqual(cache.get("cache-a"), b"a")

    # ensure that the LRU policy survives cache reinitialization
    cache.put("cache-b", b"b")

    # calling `get()` on `cache-a` makes `cache-b` least recently used
    time.sleep(1)
    cache.get("cache-a")

    # write `cache-c`, evict `cache-b`
    cache.put("cache-c", b"c")
    self.assertEqual(set(self.path.glob("cache-*")), {self.path / "cache-a", self.path / "cache-c"})

  def test_max_size(self):
    cache = LRUCache(self.name, max_size=1)

    msg = (r"Cache value for key .+? of size \d+ bytes exceeds the maximum "
           r"cache size of \d+ bytes")
    with self.assertWarnsRegex(UserWarning, msg):
      cache.put("cache-a", b"aaaa")
    self.assertIsNone(cache.get("cache-a"))
    self.assertEqual(set(self.path.glob("cache-*")), set())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
