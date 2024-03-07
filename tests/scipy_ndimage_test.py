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
import math

import numpy as np

from absl.testing import absltest
import scipy.ndimage as osp_ndimage

import jax
from jax import grad
from jax._src import test_util as jtu
from jax import dtypes
from jax.scipy import ndimage as lsp_ndimage

jax.config.parse_flags_with_absl()


float_dtypes = jtu.dtypes.floating
int_dtypes = jtu.dtypes.integer


def _fixed_ref_map_coordinates(input, coordinates, order, mode, cval=0.0):
  # SciPy's implementation of map_coordinates handles boundaries incorrectly,
  # unless mode='reflect'. For order=1, this only affects interpolation outside
  # the bounds of the original array.
  # https://github.com/scipy/scipy/issues/2640
  assert order <= 1
  padding = [(max(-np.floor(c.min()).astype(int) + 1, 0),
              max(np.ceil(c.max()).astype(int) + 1 - size, 0))
             for c, size in zip(coordinates, input.shape)]
  shifted_coords = [c + p[0] for p, c in zip(padding, coordinates)]
  pad_mode = {
      'nearest': 'edge', 'mirror': 'reflect', 'reflect': 'symmetric'
  }.get(mode, mode)
  if mode == 'constant':
    padded = np.pad(input, padding, mode=pad_mode, constant_values=cval)
  else:
    padded = np.pad(input, padding, mode=pad_mode)
  result = osp_ndimage.map_coordinates(
      padded, shifted_coords, order=order, mode=mode, cval=cval)
  return result


class NdimageTest(jtu.JaxTestCase):

  @jtu.sample_product(
    [dict(mode=mode, cval=cval)
     for mode in ['wrap', 'constant', 'nearest', 'mirror', 'reflect']
     for cval in ([0, -1] if mode == 'constant' else [0])
    ],
    [dict(impl=impl, rng_factory=rng_factory)
     for impl, rng_factory in [
       ("original", partial(jtu.rand_uniform, low=0, high=1)),
       ("fixed", partial(jtu.rand_uniform, low=-0.75, high=1.75)),
     ]
    ],
    shape=[(5,), (3, 4), (3, 4, 5)],
    coords_shape=[(7,), (2, 3, 4)],
    dtype=float_dtypes + int_dtypes,
    coords_dtype=float_dtypes,
    order=[0, 1],
    round_=[True, False],
  )
  def testMapCoordinates(self, shape, dtype, coords_shape, coords_dtype, order,
                         mode, cval, impl, round_, rng_factory):

    def args_maker():
      x = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
      coords = [(size - 1) * rng(coords_shape, coords_dtype) for size in shape]
      if round_:
        coords = [c.round().astype(int) for c in coords]
      return x, coords

    rng = rng_factory(self.rng())
    lsp_op = lambda x, c: lsp_ndimage.map_coordinates(
        x, c, order=order, mode=mode, cval=cval)
    impl_fun = (osp_ndimage.map_coordinates if impl == "original"
                else _fixed_ref_map_coordinates)
    osp_op = lambda x, c: impl_fun(x, c, order=order, mode=mode, cval=cval)

    with jtu.strict_promotion_if_dtypes_match([dtype, int if round else coords_dtype]):
      if dtype in float_dtypes:
        epsilon = max(dtypes.finfo(dtypes.canonicalize_dtype(d)).eps
                      for d in [dtype, coords_dtype])
        self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=100*epsilon)
      else:
        self._CheckAgainstNumpy(osp_op, lsp_op, args_maker, tol=0)

  def testMapCoordinatesErrors(self):
    x = np.arange(5.0)
    c = [np.linspace(0, 5, num=3)]
    with self.assertRaisesRegex(NotImplementedError, 'requires order<=1'):
      lsp_ndimage.map_coordinates(x, c, order=2)
    with self.assertRaisesRegex(
        NotImplementedError, 'does not yet support mode'):
      lsp_ndimage.map_coordinates(x, c, order=1, mode='grid-wrap')
    with self.assertRaisesRegex(ValueError, 'sequence of length'):
      lsp_ndimage.map_coordinates(x, [c, c], order=1)

  @jtu.sample_product(
    dtype=float_dtypes + int_dtypes,
    order=[0, 1],
  )
  def testMapCoordinatesRoundHalf(self, dtype, order):
    x = np.arange(-3, 3, dtype=dtype)
    c = np.array([[.5, 1.5, 2.5, 3.5]])
    def args_maker():
      return x, c

    lsp_op = lambda x, c: lsp_ndimage.map_coordinates(x, c, order=order)
    osp_op = lambda x, c: osp_ndimage.map_coordinates(x, c, order=order)

    with jtu.strict_promotion_if_dtypes_match([dtype, c.dtype]):
      self._CheckAgainstNumpy(osp_op, lsp_op, args_maker)

  def testContinuousGradients(self):
    # regression test for https://github.com/google/jax/issues/3024

    def loss(delta):
      x = np.arange(100.0)
      border = 10
      indices = np.arange(x.size, dtype=x.dtype) + delta
      # linear interpolation of the linear function y=x should be exact
      shifted = lsp_ndimage.map_coordinates(x, [indices], order=1)
      return ((x - shifted) ** 2)[border:-border].mean()

    # analytical gradient of (x - (x - delta)) ** 2 is 2 * delta
    self.assertAllClose(grad(loss)(0.5), 1.0, check_dtypes=False)
    self.assertAllClose(grad(loss)(1.0), 2.0, check_dtypes=False)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
