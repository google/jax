# Copyright 2023 The JAX Authors.
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

"""Test exporting Pallas kernels."""

import sys

from absl.testing import absltest
import jax
from jax import export
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
import numpy as np


jax.config.parse_flags_with_absl()


class ExportTest(jtu.JaxTestCase):

  def setUp(self):
    if sys.platform == "win32":
      self.skipTest("Only works on non-Windows platforms")

    super().setUp()

  def test_cross_platform(self):
    def add_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x + y

    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(add_vectors_kernel,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
                            )(x, y)

    a = np.arange(8 * 16, dtype=np.int32).reshape((8, 16))
    exp = export.export(
        add_vectors,
        lowering_platforms=["tpu", "cuda"],
    )(a, a)

    if (jtu.device_under_test() == "tpu" or
        (jtu.device_under_test() == "gpu" and
         jtu.is_cuda_compute_capability_at_least("8.0"))):
      res = exp.call(a, a)
      self.assertAllClose(res, a + a)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
