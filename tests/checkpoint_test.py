# Copyright 2020 Google LLC
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

import numpy as np
from absl.testing import absltest
from unittest import skipIf

import jax
from jax import core
from jax import numpy as jnp
from jax import test_util as jtu
import jax.experimental.checkpoint as C

from jax.config import config, flags
config.parse_flags_with_absl()

FLAGS = flags.FLAGS

class CheckpointTest(jtu.JaxTestCase):
  @skipIf(not jax.config.omnistaging_enabled, "need omnistaging")
  def testSimpleChain(self):
    def f(x):
      for i in range(8):
        x = jnp.exp(x)
      return x
    x = np.ones((4,))
    jaxpr = jax.make_jaxpr(C.checkpoint_recursive(f))(x).jaxpr
    def verify(jaxpr, lvl):
      if lvl < 0: return
      remats, exp = jaxpr.eqns[:-1], jaxpr.eqns[-1]
      self.assertEqual([eqn.primitive.name for eqn in remats],
                       ["remat_call"] * lvl)
      self.assertEqual(exp.primitive.name, "exp")
      for i, remat in enumerate(remats):
        verify(core.extract_call_jaxpr(remat.primitive, remat.params)[0], lvl - 1 - i)
    verify(jaxpr, 3)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
