# Copyright 2020 The JAX Authors.
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
import gzip
import json

from absl.testing import absltest

import jax
from jax import jit, make_jaxpr, numpy as jnp
from jax._src import config
from jax._src import jaxpr_util
from jax._src import test_util as jtu
from jax._src.lib import xla_client


config.parse_flags_with_absl()


class JaxprStatsTest(jtu.JaxTestCase):

  def test_primitives(self):
    def f(x, y):
      s = jit(jnp.sin)(x)
      return jnp.sin(s) + jnp.cos(y)

    hist = jaxpr_util.primitives(make_jaxpr(f)(1., 1.).jaxpr)

    primitives = ['add', 'sin', 'cos']
    primitives.append('pjit')
    for k in primitives:
      assert k in hist, k
    self.assertEqual(hist['sin'], 2)
    self.assertTrue(all(count == 1 for k, count in hist.items() if k != 'sin'))

  def test_primitives_by_source(self):
    def f(x, y):
      s = jnp.sin(x)
      return jnp.sin(s) + jnp.cos(y)

    hist = jaxpr_util.primitives_by_source(make_jaxpr(f)(1., 1.).jaxpr)

    sin_keys = [k for k in hist.keys() if k.startswith('sin @ ')]
    rem_keys = [k for k in hist.keys() if not k.startswith('sin @ ')]

    self.assertEqual(sum(hist[k] for k in sin_keys), 2)
    self.assertTrue(all(hist[k] == 1 for k in rem_keys))

  def test_primitives_by_shape(self):
    def f(x, y):
      def sub(x, y):
        return jnp.sum(jnp.array([x, y]))
      s = jit(sub)(x, y)
      return jnp.sin(s) + jnp.cos(y)

    hist = jaxpr_util.primitives_by_shape(make_jaxpr(f)(1., 1.).jaxpr)

    t = '64' if config.enable_x64.value else '32'
    shapes = [
        f'add :: float{t}[]',
        f'sin :: float{t}[]',
        f'cos :: float{t}[]',
        f'reduce_sum :: float{t}[]',
        f'concatenate :: float{t}[2]',
        f'pjit :: float{t}[]',
    ]
    for k in shapes:
      self.assertEqual(hist[k], 1)

  def test_source_locations(self):
    def f(x, y):
      s = jnp.sin(x)                  # sin
      return jnp.sin(s) + jnp.cos(y)  # sin, cos, add

    hist = jaxpr_util.source_locations(make_jaxpr(f)(1., 1.).jaxpr)
    self.assertEqual(sum(hist.values()), 4)

  def test_source_locations_exclude_contextlib(self):

    def f(x):
      # This generates a stack where the most recent non-jax frame
      # comes from contextlib.
      return jax.named_call(jnp.cos, name='test')(x)

    hist = jaxpr_util.source_locations(make_jaxpr(f)(jnp.arange(8.)).jaxpr)
    for filename in hist.keys():
      self.assertIn(os.path.basename(__file__), filename)

  def test_print_histogram(self):
    def f(x, y):
      s = jit(jnp.sin)(x)
      return jnp.sin(s) + jnp.cos(y)
    hist = jaxpr_util.primitives_by_source(make_jaxpr(f)(1., 1.).jaxpr)
    jaxpr_util.print_histogram(hist)

  def test_pprof_equation_profile(self):
    def f(x, y):
      s = jit(jnp.sin)(x)
      return jnp.sin(s) + jnp.cos(y)
    profile_gz = jaxpr_util.pprof_equation_profile(make_jaxpr(f)(1., 1.).jaxpr)
    profile_proto = gzip.decompress(profile_gz)
    json_str = xla_client._xla.pprof_profile_to_json(profile_proto)
    profile = json.loads(json_str)
    self.assertSetEqual(
        {"sampleType", "sample", "stringTable", "location", "function"},
        set(profile.keys()))


if __name__ == "__main__":
  absltest.main()
