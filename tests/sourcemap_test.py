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

import json

from absl.testing import absltest
from absl.testing import parameterized
from jax._src import sourcemap
from jax._src import test_util as jtu


class SourceMapTest(jtu.JaxTestCase):

  @parameterized.parameters(
      (0,),
      (1,),
      (2,),
      (3,),
      (4,),
      (5,),
      (-1,),
      (-2,),
      (-3,),
      (-4,),
      (123,),
      (456,),
      (1024,),
      (1025,),
      (2**16,),
      (2**31 - 1,),
  )
  def test_roundtrip_vlq(self, value):
    actual = sourcemap.decode_vlq(sourcemap.encode_vlq(value))
    self.assertEqual(actual, value)

  @parameterized.parameters(
      (b"A",),
      (b"C",),
      (b"AAAA",),
      (b"ACDE",),
      (b"AACAA",),
  )
  def test_roundtrip_segment(self, enc):
    actual = sourcemap.encode_segment(sourcemap.decode_segment(enc))
    self.assertEqual(actual, enc)

  def test_roundtrip_sourcemap_json(self):
    data = {
        "version": 3,
        # "file": "out.js",
        # "sourceRoot": "",
        "sources": ["foo.js", "bar.js"],
        "sourcesContent": [None, None],
        "names": ["src", "maps", "are", "fun"],
        "mappings": "A,AAAC;;AACDE",
    }
    json_data = json.dumps(data)
    json_data_roundtripped = sourcemap.SourceMap.from_json(json_data).to_json()
    self.assertEqual(json.loads(json_data_roundtripped), data)

  def test_generate_mappings(self):
    expected = "A,AAAC;;AACDE"
    gen = sourcemap.MappingsGenerator()
    # A
    gen.new_group()
    gen.new_segment(0)
    # ,AAAC
    gen.new_segment(0, 0, 0, 1)
    # ;
    gen.new_group()
    # ;AACDE
    gen.new_group()
    gen.new_segment(0, 0, 1, 0, 2)
    self.assertEqual(sourcemap.serialize_mappings(gen.mappings()), expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
