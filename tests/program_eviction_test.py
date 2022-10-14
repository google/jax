# Copyright 2022 Google LLC
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

import jax

from absl.testing import absltest
from jax._src import test_util as jtu
if jax._src.lib.xla_extension_version >= 85:
  from jax.experimental import program_eviction as pe

import numpy as np
from jax.config import config

config.parse_flags_with_absl()

if jax._src.lib.xla_extension_version < 85:

  class TransferGuardTest(jtu.JaxTestCase):
    pass

else:

  class ProgramEvictionTest(jtu.JaxTestCase):

    def test_program_eviction_scope(self):
      with pe.loaded_program_eviction_scope():

        def f(x):
          return x + 1

        f = jax.jit(f)
        f(np.array(1)).block_until_ready()


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
