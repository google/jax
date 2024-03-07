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

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax
from jax import lax

from jax._src import test_util as jtu
from jax._src.internal_test_util import lax_test_util
from jax._src import util

jax.config.parse_flags_with_absl()

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


class LaxVmapOpTest(jtu.JaxTestCase):

  def _CheckBatching(self, op, bdim_size, bdims, shapes, dtypes, rng,
                     rtol=None, atol=None, multiple_results=False):
    batched_shapes = map(functools.partial(lax_test_util.add_bdim, bdim_size),
                         bdims, shapes)
    args = [rng(shape, dtype) for shape, dtype in zip(batched_shapes, dtypes)]
    args_slice = lax_test_util.args_slicer(args, bdims)
    ans = jax.vmap(op, bdims)(*args)
    if bdim_size == 0:
      args = [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]
      out = op(*args)
      if not multiple_results:
        expected = np.zeros((0,) + out.shape, out.dtype)
      else:
        expected = [np.zeros((0,) + o.shape, o.dtype) for o in out]
    else:
      outs = [op(*args_slice(i)) for i in range(bdim_size)]
      if not multiple_results:
        expected = np.stack(outs)
      else:
        expected = [np.stack(xs) for xs in zip(*outs)]
    self.assertAllClose(ans, expected, rtol=rtol, atol=atol)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(op_name=rec.op, rng_factory=rec.rng_factory, tol=rec.tol)],
      [dict(shapes=shapes, bdims=bdims)
        for shape_group in lax_test_util.compatible_shapes
        for shapes in itertools.combinations_with_replacement(shape_group, rec.nargs)
        for bdims in lax_test_util.all_bdims(*shapes)],
      dtype=rec.dtypes,
    ) for rec in lax_test_util.lax_ops()))
  def testOp(self, op_name, rng_factory, shapes, dtype, bdims, tol):
    if dtype == np.float64 or any(len(shape) > 2 for shape in shapes):
      self.skipTest('Skipping big tests under sanitizers due to slowdown.')

    rng = rng_factory(self.rng())
    op = getattr(lax, op_name)
    self._CheckBatching(op, 10, bdims, shapes, [dtype] * len(shapes), rng,
                        atol=tol, rtol=tol)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
