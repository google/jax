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

from absl.testing import absltest

from jax._src import test_util as jtu
from jax._src import pretty_printer as pp


class PrettyPrinterTest(jtu.JaxTestCase):

  def testSourceMap(self):
    doc = pp.concat([
      pp.text("abc"), pp.source_map(pp.text("def"), 101),
      pp.source_map(pp.concat([pp.text("gh"), pp.brk(""), pp.text("ijkl")]), 77),
      pp.text("mn"),
    ])
    source_map = []
    out = doc.format(width=8, source_map=source_map)
    self.assertEqual(out, "abcdefgh\nijklmn")
    self.assertEqual(source_map, [[(3, 6, 101), (6, 8, 77)], [(0, 4, 77)]])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
