# Copyright 2019 The JAX Authors.
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

from functools import partial

import numpy as np
from scipy.sparse import csgraph, csr_matrix

from absl.testing import absltest

import jax
from jax._src import dtypes
from jax import numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


all_dtypes = jtu.dtypes.floating + jtu.dtypes.integer + jtu.dtypes.complex


# TODO: these tests fail without fixed PRNG seeds.


class TestPolynomial(jtu.JaxTestCase):

  def assertSetsAllClose(self, x, y, rtol=None, atol=None, check_dtypes=True):
    """Assert that x and y contain permutations of the same approximate set of values.

    For non-complex inputs, this is accomplished by comparing the sorted inputs.
    For complex, such an approach can be confounded by numerical errors. In this case,
    we compute the structural rank of the pairwise comparison matrix: if the structural
    rank is full, it implies that the matrix can be permuted so that the diagonal is
    non-zero, which implies a one-to-one approximate match between the permuted sets.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    atol = max(jtu.tolerance(x.dtype, atol), jtu.tolerance(y.dtype, atol))
    rtol = max(jtu.tolerance(x.dtype, rtol), jtu.tolerance(y.dtype, rtol))

    if not (np.issubdtype(x.dtype, np.complexfloating) or
            np.issubdtype(y.dtype, np.complexfloating)):
      return self.assertAllClose(np.sort(x), np.sort(y), atol=atol, rtol=rtol,
                                 check_dtypes=check_dtypes)

    if check_dtypes:
      self.assertEqual(x.dtype, y.dtype)
    self.assertEqual(x.size, y.size)

    pairwise = np.isclose(x[:, None], x[None, :],
                          atol=atol, rtol=rtol, equal_nan=True)
    rank = csgraph.structural_rank(csr_matrix(pairwise))
    self.assertEqual(rank, x.size)


  @jtu.sample_product(
    dtype=all_dtypes,
    length=[0, 3, 5],
    leading=[0, 2],
    trailing=[0, 2],
  )
  # TODO(phawkins): no nonsymmetric eigendecomposition implementation on GPU.
  @jtu.run_on_devices("cpu")
  def testRoots(self, dtype, length, leading, trailing):
    rng = jtu.rand_some_zero(self.rng())

    def args_maker():
      p = rng((length,), dtype)
      return [jnp.concatenate(
        [jnp.zeros(leading, p.dtype), p, jnp.zeros(trailing, p.dtype)])]

    jnp_fun = jnp.roots
    def np_fun(arg):
      return np.roots(arg).astype(dtypes.to_complex_dtype(arg.dtype))

    # Note: outputs have no defined order, so we need to use a special comparator.
    args = args_maker()
    np_roots = np_fun(*args)
    jnp_roots = jnp_fun(*args)
    self.assertSetsAllClose(np_roots, jnp_roots)

  @jtu.sample_product(
    dtype=all_dtypes,
    length=[0, 3, 5],
    leading=[0, 2],
    trailing=[0, 2],
  )
  # TODO(phawkins): no nonsymmetric eigendecomposition implementation on GPU.
  @jtu.run_on_devices("cpu")
  def testRootsNoStrip(self, dtype, length, leading, trailing):
    rng = jtu.rand_some_zero(self.rng())

    def args_maker():
      p = rng((length,), dtype)
      return [jnp.concatenate(
        [jnp.zeros(leading, p.dtype), p, jnp.zeros(trailing, p.dtype)])]

    jnp_fun = partial(jnp.roots, strip_zeros=False)
    def np_fun(arg):
      roots = np.roots(arg).astype(dtypes.to_complex_dtype(arg.dtype))
      if len(roots) < len(arg) - 1:
        roots = np.pad(roots, (0, len(arg) - len(roots) - 1),
                       constant_values=complex(np.nan, np.nan))
      return roots

    # Note: outputs have no defined order, so we need to use a special comparator.
    args = args_maker()
    np_roots = np_fun(*args)
    jnp_roots = jnp_fun(*args)
    self.assertSetsAllClose(np_roots, jnp_roots)
    self._CompileAndCheck(jnp_fun, args_maker)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
