# Copyright 2018 The JAX Authors.
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

import unittest

import jax
from jax import lax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax._src.lax import eigh as lax_eigh

from absl.testing import absltest

jax.config.parse_flags_with_absl()


linear_sizes = [16, 97, 128]


class LaxScipySpectralDacTest(jtu.JaxTestCase):

  @jtu.sample_product(
    linear_size=linear_sizes,
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
    termination_size=[1, 19],
  )
  def test_spectral_dac_eigh(self, linear_size, dtype, termination_size):
    if not jtu.test_device_matches(["tpu"]) and termination_size != 1:
      raise unittest.SkipTest(
          "Termination sizes greater than 1 only work on TPU")

    rng = self.rng()
    H = rng.randn(linear_size, linear_size)
    H = jnp.array(0.5 * (H + H.conj().T)).astype(dtype)
    if jnp.dtype(dtype).name in ("bfloat16", "float16"):
      self.assertRaises(
        NotImplementedError, lax_eigh.eigh, H)
      return
    evs, V = lax_eigh.eigh(H, termination_size=termination_size)
    ev_exp, _ = jnp.linalg.eigh(H)
    HV = jnp.dot(H, V, precision=lax.Precision.HIGHEST)
    vV = evs.astype(V.dtype)[None, :] * V
    eps = jnp.finfo(H.dtype).eps
    atol = jnp.linalg.norm(H) * eps
    self.assertAllClose(ev_exp, jnp.sort(evs), atol=20 * atol)
    self.assertAllClose(
        HV, vV, atol=atol * (140 if jnp.issubdtype(dtype, jnp.complexfloating)
                             else 40))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
