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

"""Tests whether the frontend attributes added by the context manager are

correctly propagated to the jaxpr and mlir.
"""

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental.xla_metadata import set_xla_metadata


config.parse_flags_with_absl()

class XlaMetadataTest(jtu.JaxTestCase):

  def test_f_jitted_jaxpr(self):
    @jax.jit
    def f(a, b):
      with set_xla_metadata(a = True, b = 'x'):
        return a + b

    jaxpr = f.trace(1, 2).jaxpr
    for e in jaxpr.eqns:
      print(e.ctx)
    print(f.lower(1, 2).as_text())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
