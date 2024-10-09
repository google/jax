# Copyright 2022 The JAX Authors.
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

import operator
from functools import reduce
import numpy as np

import jax
from jax._src import test_util as jtu
import scipy.interpolate as sp_interp
import jax.scipy.interpolate as jsp_interp


jax.config.parse_flags_with_absl()


class LaxBackedScipyInterpolateTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.interpolate implementations"""

  @jtu.sample_product(
    spaces=(((0., 10., 10),), ((-15., 20., 12), (3., 4., 24))),
    method=("linear", "nearest"),
    fill_value=(np.nan, None),
  )
  def testRegularGridInterpolator(self, spaces, method, fill_value):
    rng = jtu.rand_default(self.rng())
    scipy_fun = lambda init_args, call_args: sp_interp.RegularGridInterpolator(
        *init_args[:2], method, False, *init_args[2:])(*call_args)
    lax_fun = lambda init_args, call_args: jsp_interp.RegularGridInterpolator(
        *init_args[:2], method, False, *init_args[2:])(*call_args)

    def args_maker():
      points = tuple(map(lambda x: np.linspace(*x), spaces))
      values = rng(reduce(operator.add, tuple(map(np.shape, points))), float)

      init_args = (points, values, fill_value)
      n_validation_points = 50
      valid_points = tuple(
          map(
              lambda x: np.linspace(x[0] - 0.2 * (x[1] - x[0]), x[1] + 0.2 *
                                    (x[1] - x[0]), n_validation_points),
              spaces))
      valid_points = np.squeeze(np.stack(valid_points, axis=1))
      call_args = (valid_points,)
      return init_args, call_args

    self._CheckAgainstNumpy(
        scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-4)
    self._CompileAndCheck(lax_fun, args_maker, rtol={np.float64: 1e-14})


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
