# Copyright 2021 The JAX Authors.
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
import itertools

from absl.testing import absltest

import jax
from jax._src import test_util as jtu
import jax.scipy.fft as jsp_fft
import scipy.fft as osp_fft

jax.config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.floating
real_dtypes = float_dtypes + jtu.dtypes.integer + jtu.dtypes.boolean

def _get_dctn_test_axes(shape):
  axes = [[]]
  ndims = len(shape)
  axes.append(None)
  for naxes in range(1, min(ndims, 3) + 1):
    axes.extend(itertools.combinations(range(ndims), naxes))
  for index in range(1, ndims + 1):
    axes.append((-index,))
  return axes

def _get_dctn_test_s(shape, axes):
  s_list = [None]
  if axes is not None:
    s_list.extend(itertools.product(*[[shape[ax]+i for i in range(-shape[ax]+1, shape[ax]+1)] for ax in axes]))
  return s_list


class LaxBackedScipyFftTests(jtu.JaxTestCase):
  """Tests for LAX-backed scipy.fft implementations"""

  @jtu.sample_product(
    dtype=real_dtypes,
    shape=[(10,), (2, 5)],
    n=[None, 1, 7, 13, 20],
    axis=[-1, 0],
    norm=[None, 'ortho'],
  )
  def testDct(self, shape, dtype, n, axis, norm):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda a: jsp_fft.dct(a, n=n, axis=axis, norm=norm)
    np_fn = lambda a: osp_fft.dct(a, n=n, axis=axis, norm=norm)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    [dict(shape=shape, axes=axes, s=s)
     for shape in [(10,), (10, 10), (9,), (2, 3, 4), (2, 3, 4, 5)]
     for axes in _get_dctn_test_axes(shape)
     for s in _get_dctn_test_s(shape, axes)],
    dtype=real_dtypes,
    norm=[None, 'ortho'],
  )
  def testDctn(self, shape, dtype, s, axes, norm):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda a: jsp_fft.dctn(a, s=s, axes=axes, norm=norm)
    np_fn = lambda a: osp_fft.dctn(a, s=s, axes=axes, norm=norm)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    dtype=real_dtypes,
    shape=[(10,), (2, 5)],
    n=[None, 1, 7, 13, 20],
    axis=[-1, 0],
    norm=[None, 'ortho'],
  )
  # TODO(phawkins): these tests are failing on T4 GPUs in CI with a
  # CUDA_ERROR_ILLEGAL_ADDRESS.
  @jtu.skip_on_devices("cuda")
  def testiDct(self, shape, dtype, n, axis, norm):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda a: jsp_fft.idct(a, n=n, axis=axis, norm=norm)
    np_fn = lambda a: osp_fft.idct(a, n=n, axis=axis, norm=norm)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

  @jtu.sample_product(
    [dict(shape=shape, axes=axes, s=s)
     for shape in [(10,), (10, 10), (9,), (2, 3, 4), (2, 3, 4, 5)]
     for axes in _get_dctn_test_axes(shape)
     for s in _get_dctn_test_s(shape, axes)],
    dtype=real_dtypes,
    norm=[None, 'ortho'],
  )
  # TODO(phawkins): these tests are failing on T4 GPUs in CI with a
  # CUDA_ERROR_ILLEGAL_ADDRESS.
  @jtu.skip_on_devices("cuda")
  def testiDctn(self, shape, dtype, s, axes, norm):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(shape, dtype),)
    jnp_fn = lambda a: jsp_fft.idctn(a, s=s, axes=axes, norm=norm)
    np_fn = lambda a: osp_fft.idctn(a, s=s, axes=axes, norm=norm)
    self._CheckAgainstNumpy(np_fn, jnp_fn, args_maker, check_dtypes=False,
                            tol=1e-4)
    self._CompileAndCheck(jnp_fn, args_maker, atol=1e-4)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
