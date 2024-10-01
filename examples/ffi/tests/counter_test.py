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

import jax
from jax._src import test_util as jtu

from jax_ffi_example import counter

jax.config.parse_flags_with_absl()


class CounterTests(jtu.JaxTestCase):
  def test_basic(self):
    self.assertEqual(counter.counter(0), 0)
    self.assertEqual(counter.counter(0), 1)
    self.assertEqual(counter.counter(0), 2)
    self.assertEqual(counter.counter(1), 0)
    self.assertEqual(counter.counter(0), 3)

  def test_jit(self):
    @jax.jit
    def counter_fun(x):
      return x, counter.counter(2)

    self.assertEqual(counter_fun(0)[1], 0)
    self.assertEqual(counter_fun(0)[1], 1)

    # Persists across different cache hits
    self.assertEqual(counter_fun(1)[1], 2)

    # Persists after the cache is cleared
    counter_fun.clear_cache()
    self.assertEqual(counter_fun(0)[1], 3)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
