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

from functools import partial
import unittest

import numpy as np

from absl.testing import absltest

import jax
from jax import image
from jax import numpy as jnp
from jax._src import test_util as jtu

# We use TensorFlow and PIL as reference implementations.
try:
  import tensorflow as tf
except ImportError:
  tf = None

try:
  from PIL import Image as PIL_Image
except ImportError:
  PIL_Image = None

jax.config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.all_floating
inexact_dtypes = jtu.dtypes.inexact

class ImageTest(jtu.JaxTestCase):

  _TF_SHAPES = [[2, 3, 2, 4], [2, 6, 4, 4], [2, 33, 17, 4], [2, 50, 38, 4]]

  @jtu.sample_product(
    dtype=float_dtypes,
    target_shape=_TF_SHAPES,
    image_shape=_TF_SHAPES,
    method=["nearest", "bilinear", "lanczos3", "lanczos5", "bicubic"],
    antialias=[False, True],
  )
  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testResizeAgainstTensorFlow(self, dtype, image_shape, target_shape, method,
                                  antialias):
    # TODO(phawkins): debug this. There is a small mismatch between TF and JAX
    # for some cases of non-antialiased bicubic downscaling; we would expect
    # exact equality.
    if method == "bicubic" and not antialias:
      raise unittest.SkipTest("non-antialiased bicubic downscaling mismatch")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(image_shape, dtype),)
    def tf_fn(x):
      out = tf.image.resize(
        x.astype(np.float64), tf.constant(target_shape[1:-1]),
        method=method, antialias=antialias).numpy().astype(dtype)
      return out
    jax_fn = partial(image.resize, shape=target_shape, method=method,
                     antialias=antialias)
    self._CheckAgainstNumpy(tf_fn, jax_fn, args_maker, check_dtypes=True,
                            tol={np.float16: 2e-2, jnp.bfloat16: 1e-1,
                                 np.float32: 1e-4, np.float64: 1e-4})


  _PIL_SHAPES = [[3, 2], [6, 4], [33, 17], [50, 39]]

  @jtu.sample_product(
    dtype=[np.float32],
    target_shape=_PIL_SHAPES,
    image_shape=_PIL_SHAPES,
    method=["nearest", "bilinear", "lanczos3", "bicubic"],
  )
  @unittest.skipIf(not PIL_Image, "Test requires PIL")
  def testResizeAgainstPIL(self, dtype, image_shape, target_shape, method):
    rng = jtu.rand_uniform(self.rng())
    args_maker = lambda: (rng(image_shape, dtype),)
    def pil_fn(x):
      pil_methods = {
        "nearest": PIL_Image.Resampling.NEAREST,
        "bilinear": PIL_Image.Resampling.BILINEAR,
        "bicubic": PIL_Image.Resampling.BICUBIC,
        "lanczos3": PIL_Image.Resampling.LANCZOS,
      }
      img = PIL_Image.fromarray(x.astype(np.float32))
      out = np.asarray(img.resize(target_shape[::-1], pil_methods[method]),
                       dtype=dtype)
      return out
    if (image_shape == [6, 4] and target_shape == [33, 17]
        and method == "nearest"):
      # TODO(phawkins): I suspect we're simply handling ties differently for
      # this test case.
      raise unittest.SkipTest("Test fails")
    jax_fn = partial(image.resize, shape=target_shape, method=method,
                     antialias=True)
    self._CheckAgainstNumpy(pil_fn, jax_fn, args_maker, check_dtypes=True,
                            atol=3e-5)

  @jtu.sample_product(
    [dict(image_shape=image_shape, target_shape=target_shape)
     for image_shape, target_shape in [
       ([3, 1, 2], [6, 1, 4]),
       ([1, 3, 2, 1], [1, 6, 4, 1]),
     ]],
    dtype=inexact_dtypes,
    method=["nearest", "linear", "lanczos3", "lanczos5", "cubic"],
  )
  def testResizeUp(self, dtype, image_shape, target_shape, method):
    data = [64, 32, 32, 64, 50, 100]
    expected_data = {}
    expected_data["nearest"] = [
        64.0, 64.0, 32.0, 32.0, 64.0, 64.0, 32.0, 32.0, 32.0, 32.0, 64.0, 64.0,
        32.0, 32.0, 64.0, 64.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 100.0,
        100.0
    ]
    expected_data["linear"] = [
        64.0, 56.0, 40.0, 32.0, 56.0, 52.0, 44.0, 40.0, 40.0, 44.0, 52.0, 56.0,
        36.5, 45.625, 63.875, 73.0, 45.5, 56.875, 79.625, 91.0, 50.0, 62.5,
        87.5, 100.0
    ]
    expected_data["lanczos3"] = [
        75.8294, 59.6281, 38.4313, 22.23, 60.6851, 52.0037, 40.6454, 31.964,
        35.8344, 41.0779, 47.9383, 53.1818, 24.6968, 43.0769, 67.1244, 85.5045,
        35.7939, 56.4713, 83.5243, 104.2017, 44.8138, 65.1949, 91.8603, 112.2413
    ]
    expected_data["lanczos5"] = [
        77.5699, 60.0223, 40.6694, 23.1219, 61.8253, 51.2369, 39.5593, 28.9709,
        35.7438, 40.8875, 46.5604, 51.7041, 21.5942, 43.5299, 67.7223, 89.658,
        32.1213, 56.784, 83.984, 108.6467, 44.5802, 66.183, 90.0082, 111.6109
    ]
    expected_data["cubic"] = [
        70.1453, 59.0252, 36.9748, 25.8547, 59.3195, 53.3386, 41.4789, 35.4981,
        36.383, 41.285, 51.0051, 55.9071, 30.2232, 42.151, 65.8032, 77.731,
        41.6492, 55.823, 83.9288, 98.1026, 47.0363, 62.2744, 92.4903, 107.7284
    ]
    x = np.array(data, dtype=dtype).reshape(image_shape)
    output = image.resize(x, target_shape, method)
    expected = np.array(expected_data[method], dtype=dtype).reshape(target_shape)
    self.assertAllClose(output, expected, atol=1e-04)

  _RESIZE_GRADIENTS_SHAPES = [[2, 3, 2, 4], [2, 6, 4, 4], [2, 33, 17, 4],
                              [2, 50, 38, 4]]

  @jtu.sample_product(
    dtype=[np.float32],
    target_shape=_RESIZE_GRADIENTS_SHAPES,
    image_shape=_RESIZE_GRADIENTS_SHAPES,
    method=["bilinear", "lanczos3", "lanczos5", "bicubic"],
    antialias=[False, True],
  )
  def testResizeGradients(self, dtype, image_shape, target_shape, method,
                           antialias):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(image_shape, dtype),)
    jax_fn = partial(image.resize, shape=target_shape, method=method,
                     antialias=antialias)
    jtu.check_grads(jax_fn, args_maker(), order=2, rtol=1e-2, eps=1.)

  @jtu.sample_product(
    [dict(image_shape=image_shape, target_shape=target_shape)
     for image_shape, target_shape in [
       ([1], [0]),
       ([5, 5], [5, 0]),
       ([5, 5], [0, 1]),
       ([5, 5], [0, 0])
     ]
    ],
    dtype=[np.float32],
    method=["nearest", "linear", "lanczos3", "lanczos5", "cubic"],
    antialias=[False, True],
  )
  def testResizeEmpty(self, dtype, image_shape, target_shape, method, antialias):
    # Regression test for https://github.com/jax-ml/jax/issues/7586
    image = np.ones(image_shape, dtype)
    out = jax.image.resize(image, shape=target_shape, method=method, antialias=antialias)
    self.assertArraysEqual(out, jnp.zeros(target_shape, dtype))

  @jtu.sample_product(
    [dict(image_shape=image_shape, target_shape=target_shape, scale=scale,
          translation=translation)
     for image_shape, target_shape, scale, translation in [
       ([3, 1, 2], [6, 1, 4], [2.0, 1.0, 2.0], [1.0, 0.0, -1.0]),
       ([1, 3, 2, 1], [1, 6, 4, 1], [1.0, 2.0, 2.0, 1.0], [0.0, 1.0, -1.0, 0.0])
     ]
    ],
    dtype=inexact_dtypes,
    method=["linear", "lanczos3", "lanczos5", "cubic"],
  )
  def testScaleAndTranslateUp(self, dtype, image_shape, target_shape, scale,
                              translation, method):
    data = [64, 32, 32, 64, 50, 100]
    # Note zeros occur in the output because the sampling location is outside
    # the boundaries of the input image.
    expected_data = {}
    expected_data["linear"] = [
        0.0, 0.0, 0.0, 0.0, 56.0, 40.0, 32.0, 0.0, 52.0, 44.0, 40.0, 0.0, 44.0,
        52.0, 56.0, 0.0, 45.625, 63.875, 73.0, 0.0, 56.875, 79.625, 91.0, 0.0
    ]
    expected_data["lanczos3"] = [
        0.0, 0.0, 0.0, 0.0, 59.6281, 38.4313, 22.23, 0.0, 52.0037, 40.6454,
        31.964, 0.0, 41.0779, 47.9383, 53.1818, 0.0, 43.0769, 67.1244, 85.5045,
        0.0, 56.4713, 83.5243, 104.2017, 0.0
    ]
    expected_data["lanczos5"] = [
        0.0, 0.0, 0.0, 0.0, 60.0223, 40.6694, 23.1219, 0.0, 51.2369, 39.5593,
        28.9709, 0.0, 40.8875, 46.5604, 51.7041, 0.0, 43.5299, 67.7223, 89.658,
        0.0, 56.784, 83.984, 108.6467, 0.0
    ]
    expected_data["cubic"] = [
        0.0, 0.0, 0.0, 0.0, 59.0252, 36.9748, 25.8547, 0.0, 53.3386, 41.4789,
        35.4981, 0.0, 41.285, 51.0051, 55.9071, 0.0, 42.151, 65.8032, 77.731,
        0.0, 55.823, 83.9288, 98.1026, 0.0
    ]
    x = np.array(data, dtype=dtype).reshape(image_shape)
    # Should we test different float types here?
    scale_a = jnp.array(scale, dtype=jnp.float32)
    translation_a = jnp.array(translation, dtype=jnp.float32)
    output = image.scale_and_translate(x, target_shape, range(len(image_shape)),
                                       scale_a, translation_a,
                                       method)

    expected = np.array(
        expected_data[method], dtype=dtype).reshape(target_shape)
    self.assertAllClose(output, expected, atol=2e-03)

  @jtu.sample_product(
    dtype=inexact_dtypes,
    method=["linear", "lanczos3", "lanczos5", "cubic"],
    antialias=[True, False],
  )
  def testScaleAndTranslateDown(self, dtype, method, antialias):
    image_shape = [1, 6, 7, 1]
    target_shape = [1, 3, 3, 1]

    data = [
        51, 38, 32, 89, 41, 21, 97, 51, 33, 87, 89, 34, 21, 97, 43, 25, 25, 92,
        41, 11, 84, 11, 55, 111, 23, 99, 50, 83, 13, 92, 52, 43, 90, 43, 14, 89,
        71, 32, 23, 23, 35, 93
    ]
    if antialias:
      expected_data = {}
      expected_data["linear"] = [
          43.5372, 59.3694, 53.6907, 49.3221, 56.8168, 55.4849, 0, 0, 0
      ]
      expected_data["lanczos3"] = [
          43.2884, 57.9091, 54.6439, 48.5856, 58.2427, 53.7551, 0, 0, 0
      ]
      expected_data["lanczos5"] = [
          43.9209, 57.6360, 54.9575, 48.9272, 58.1865, 53.1948, 0, 0, 0
      ]
      expected_data["cubic"] = [
          42.9935, 59.1687, 54.2138, 48.2640, 58.2678, 54.4088, 0, 0, 0
      ]
    else:
      expected_data = {}
      expected_data["linear"] = [
          43.6071, 89, 59, 37.1785, 27.2857, 58.3571, 0, 0, 0
      ]
      expected_data["lanczos3"] = [
          44.1390, 87.8786, 63.3111, 25.1161, 20.8795, 53.6165, 0, 0, 0
      ]
      expected_data["lanczos5"] = [
          44.8835, 85.5896, 66.7231, 16.9983, 19.8891, 47.1446, 0, 0, 0
      ]
      expected_data["cubic"] = [
          43.6426, 88.8854, 60.6638, 31.4685, 22.1204, 58.3457, 0, 0, 0
      ]
    x = np.array(data, dtype=dtype).reshape(image_shape)

    expected = np.array(
        expected_data[method], dtype=dtype).reshape(target_shape)
    scale_a = jnp.array([1.0, 0.35, 0.4, 1.0], dtype=jnp.float32)
    translation_a = jnp.array([0.0, 0.2, 0.1, 0.0], dtype=jnp.float32)

    output = image.scale_and_translate(
        x, target_shape, (0,1,2,3),
        scale_a, translation_a, method, antialias=antialias)
    self.assertAllClose(output, expected, atol=2e-03)

    # Tests that running with just a subset of dimensions that have non-trivial
    # scale and translation.
    output = image.scale_and_translate(
        x, target_shape, (1,2),
        scale_a[1:3], translation_a[1:3], method, antialias=antialias)
    self.assertAllClose(output, expected, atol=2e-03)


  @jtu.sample_product(antialias=[True, False])
  def testScaleAndTranslateJITs(self, antialias):
    image_shape = [1, 6, 7, 1]
    target_shape = [1, 3, 3, 1]

    data = [
        51, 38, 32, 89, 41, 21, 97, 51, 33, 87, 89, 34, 21, 97, 43, 25, 25, 92,
        41, 11, 84, 11, 55, 111, 23, 99, 50, 83, 13, 92, 52, 43, 90, 43, 14, 89,
        71, 32, 23, 23, 35, 93
    ]
    if antialias:
      expected_data = [
          43.5372, 59.3694, 53.6907, 49.3221, 56.8168, 55.4849, 0, 0, 0
      ]
    else:
      expected_data = [43.6071, 89, 59, 37.1785, 27.2857, 58.3571, 0, 0, 0]
    x = jnp.array(data, dtype=jnp.float32).reshape(image_shape)

    expected = jnp.array(expected_data, dtype=jnp.float32).reshape(target_shape)
    scale_a = jnp.array([1.0, 0.35, 0.4, 1.0], dtype=jnp.float32)
    translation_a = jnp.array([0.0, 0.2, 0.1, 0.0], dtype=jnp.float32)

    def jit_fn(in_array, s, t):
      return jax.image.scale_and_translate(
          in_array, target_shape, (0, 1, 2, 3), s, t,
          "linear", antialias, precision=jax.lax.Precision.HIGHEST)

    output = jax.jit(jit_fn)(x, scale_a, translation_a)
    self.assertAllClose(output, expected, atol=2e-03)

  @jtu.sample_product(antialias=[True, False])
  def testScaleAndTranslateGradFinite(self, antialias):
    image_shape = [1, 6, 7, 1]
    target_shape = [1, 3, 3, 1]

    data = [
        51, 38, 32, 89, 41, 21, 97, 51, 33, 87, 89, 34, 21, 97, 43, 25, 25, 92,
        41, 11, 84, 11, 55, 111, 23, 99, 50, 83, 13, 92, 52, 43, 90, 43, 14, 89,
        71, 32, 23, 23, 35, 93
    ]

    x = jnp.array(data, dtype=jnp.float32).reshape(image_shape)
    scale_a = jnp.array([1.0, 0.35, 0.4, 1.0], dtype=jnp.float32)
    translation_a = jnp.array([0.0, 0.2, 0.1, 0.0], dtype=jnp.float32)

    def scale_fn(s):
      return jnp.sum(jax.image.scale_and_translate(
        x, target_shape, (0, 1, 2, 3), s, translation_a, "linear", antialias,
        precision=jax.lax.Precision.HIGHEST))

    scale_out = jax.grad(scale_fn)(scale_a)
    self.assertTrue(jnp.all(jnp.isfinite(scale_out)))

    def translate_fn(t):
      return jnp.sum(jax.image.scale_and_translate(
        x, target_shape, (0, 1, 2, 3), scale_a, t, "linear", antialias,
        precision=jax.lax.Precision.HIGHEST))

    translate_out = jax.grad(translate_fn)(translation_a)
    self.assertTrue(jnp.all(jnp.isfinite(translate_out)))

  def testScaleAndTranslateNegativeDims(self):
    data = jnp.full((3, 3), 0.5)
    actual = jax.image.scale_and_translate(
      data, (5, 5), (-2, -1), jnp.ones(2), jnp.zeros(2), "linear")
    expected = jax.image.scale_and_translate(
      data, (5, 5), (0, 1), jnp.ones(2), jnp.zeros(2), "linear")
    self.assertAllClose(actual, expected)

  def testResizeWithUnusualShapes(self):
    x = jnp.ones((3, 4))
    # Array shapes are accepted
    self.assertEqual((10, 17),
                     jax.image.resize(x, jnp.array((10, 17)), "nearest").shape)
    with self.assertRaises(TypeError):
      # Fractional shapes are disallowed
      jax.image.resize(x, [10.5, 17], "bicubic")

  # References for tests below were computed using torch.nn.Unfold v1.13.

  def testUnfold0D(self):
    # OD unfold only adds an extra singleton dimension.
    x = jnp.arange(10).reshape((2, 5))
    x = x.astype(jax.dtypes.canonicalize_dtype(jnp.float64))
    ref = jnp.expand_dims(x, -1)
    unfold = image.unfold(x, ())
    self.assertAllClose(ref, unfold)

  def testUnfold1D(self):
    x = jnp.arange(20).reshape((2, 2, 5))
    x = x.astype(jax.dtypes.canonicalize_dtype(jnp.float64))
    ref = jnp.array(
        [[[ 0.,  2.],
         [ 1.,  3.],
         [ 2.,  4.],
         [ 5.,  7.],
         [ 6.,  8.],
         [ 7.,  9.]],

        [[10., 12.],
         [11., 13.],
         [12., 14.],
         [15., 17.],
         [16., 18.],
         [17., 19.]]]
    )
    unfold = image.unfold(x, 3, stride=2)
    self.assertAllClose(ref, unfold)

  def testUnfold2D(self):
    x = jnp.arange(60).reshape((2, 2, 5, 3))
    x = x.astype(jax.dtypes.canonicalize_dtype(jnp.float64))
    ref = jnp.array(
        [[[ 0.,  1.,  3.,  4.,  6.,  7.],
          [ 1.,  2.,  4.,  5.,  7.,  8.],
          [ 3.,  4.,  6.,  7.,  9., 10.],
          [ 4.,  5.,  7.,  8., 10., 11.],
          [ 6.,  7.,  9., 10., 12., 13.],
          [ 7.,  8., 10., 11., 13., 14.],
          [15., 16., 18., 19., 21., 22.],
          [16., 17., 19., 20., 22., 23.],
          [18., 19., 21., 22., 24., 25.],
          [19., 20., 22., 23., 25., 26.],
          [21., 22., 24., 25., 27., 28.],
          [22., 23., 25., 26., 28., 29.]],

         [[30., 31., 33., 34., 36., 37.],
          [31., 32., 34., 35., 37., 38.],
          [33., 34., 36., 37., 39., 40.],
          [34., 35., 37., 38., 40., 41.],
          [36., 37., 39., 40., 42., 43.],
          [37., 38., 40., 41., 43., 44.],
          [45., 46., 48., 49., 51., 52.],
          [46., 47., 49., 50., 52., 53.],
          [48., 49., 51., 52., 54., 55.],
          [49., 50., 52., 53., 55., 56.],
          [51., 52., 54., 55., 57., 58.],
          [52., 53., 55., 56., 58., 59.]]]
    )
    unfold = image.unfold(x, (3, 2))
    self.assertAllClose(ref, unfold)

  def testUnfold2DDilated1(self):
    x = jnp.arange(60).reshape((2, 2, 5, 3))
    x = x.astype(jax.dtypes.canonicalize_dtype(jnp.float64))
    ref = jnp.array(
      [[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  0.,
           0.,  0.,  0.,  6.,  7.,  8.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  0.,  0.,  0.,
           0.,  6.,  7.,  8.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  1.,  2.,  0.,  0.,  0.,  0.,  6.,  7.,  8.,  0.,
           0.,  0.,  0., 12., 13., 14.,  0.],
         [ 0.,  0.,  1.,  2.,  0.,  0.,  0.,  0.,  6.,  7.,  8.,  0.,  0.,  0.,
           0., 12., 13., 14.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  6.,  7.,  8.,  0.,  0.,  0.,  0., 12., 13., 14.,  0.,
           0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  6.,  7.,  8.,  0.,  0.,  0.,  0., 12., 13., 14.,  0.,  0.,  0.,
           0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15., 16., 17.,  0.,
           0.,  0.,  0., 21., 22., 23.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15., 16., 17.,  0.,  0.,  0.,
           0., 21., 22., 23.,  0.,  0.,  0.],
         [ 0.,  0.,  0., 15., 16., 17.,  0.,  0.,  0.,  0., 21., 22., 23.,  0.,
           0.,  0.,  0., 27., 28., 29.,  0.],
         [ 0., 15., 16., 17.,  0.,  0.,  0.,  0., 21., 22., 23.,  0.,  0.,  0.,
           0., 27., 28., 29.,  0.,  0.,  0.],
         [ 0.,  0.,  0., 21., 22., 23.,  0.,  0.,  0.,  0., 27., 28., 29.,  0.,
           0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0., 21., 22., 23.,  0.,  0.,  0.,  0., 27., 28., 29.,  0.,  0.,  0.,
           0.,  0.,  0.,  0.,  0.,  0.,  0.]],

        [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 30., 31., 32.,  0.,
           0.,  0.,  0., 36., 37., 38.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 30., 31., 32.,  0.,  0.,  0.,
           0., 36., 37., 38.,  0.,  0.,  0.],
         [ 0.,  0.,  0., 30., 31., 32.,  0.,  0.,  0.,  0., 36., 37., 38.,  0.,
           0.,  0.,  0., 42., 43., 44.,  0.],
         [ 0., 30., 31., 32.,  0.,  0.,  0.,  0., 36., 37., 38.,  0.,  0.,  0.,
           0., 42., 43., 44.,  0.,  0.,  0.],
         [ 0.,  0.,  0., 36., 37., 38.,  0.,  0.,  0.,  0., 42., 43., 44.,  0.,
           0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0., 36., 37., 38.,  0.,  0.,  0.,  0., 42., 43., 44.,  0.,  0.,  0.,
           0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 45., 46., 47.,  0.,
           0.,  0.,  0., 51., 52., 53.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 45., 46., 47.,  0.,  0.,  0.,
           0., 51., 52., 53.,  0.,  0.,  0.],
         [ 0.,  0.,  0., 45., 46., 47.,  0.,  0.,  0.,  0., 51., 52., 53.,  0.,
           0.,  0.,  0., 57., 58., 59.,  0.],
         [ 0., 45., 46., 47.,  0.,  0.,  0.,  0., 51., 52., 53.,  0.,  0.,  0.,
           0., 57., 58., 59.,  0.,  0.,  0.],
         [ 0.,  0.,  0., 51., 52., 53.,  0.,  0.,  0.,  0., 57., 58., 59.,  0.,
           0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0., 51., 52., 53.,  0.,  0.,  0.,  0., 57., 58., 59.,  0.,  0.,  0.,
           0.,  0.,  0.,  0.,  0.,  0.,  0.]]]
    )
    unfold = image.unfold(x, (3, 2), (2, 2), (2, 3), (2, 1))
    self.assertAllClose(ref, unfold)

  def testUnfold2DDilated2(self):
    x = jnp.arange(60).reshape((1, 3, 4, 5))
    x = x.astype(jax.dtypes.canonicalize_dtype(jnp.float64))
    ref = jnp.array(
      [[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,
           0.,  0.,  0., 10., 11., 12., 13.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,  4.,  0.,
           0., 10., 11., 12., 13., 14.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,  4.,  0.,  0.,  0.,
          11., 12., 13., 14.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5.,  6.,  7.,  8.,
           0.,  0.,  0., 15., 16., 17., 18.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5.,  6.,  7.,  8.,  9.,  0.,
           0., 15., 16., 17., 18., 19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  6.,  7.,  8.,  9.,  0.,  0.,  0.,
          16., 17., 18., 19.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 20., 21., 22., 23.,
           0.,  0.,  0., 30., 31., 32., 33.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 20., 21., 22., 23., 24.,  0.,
           0., 30., 31., 32., 33., 34.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0., 21., 22., 23., 24.,  0.,  0.,  0.,
          31., 32., 33., 34.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 25., 26., 27., 28.,
           0.,  0.,  0., 35., 36., 37., 38.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 25., 26., 27., 28., 29.,  0.,
           0., 35., 36., 37., 38., 39.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0., 26., 27., 28., 29.,  0.,  0.,  0.,
          36., 37., 38., 39.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 40., 41., 42., 43.,
           0.,  0.,  0., 50., 51., 52., 53.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 40., 41., 42., 43., 44.,  0.,
           0., 50., 51., 52., 53., 54.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0., 41., 42., 43., 44.,  0.,  0.,  0.,
          51., 52., 53., 54.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 45., 46., 47., 48.,
           0.,  0.,  0., 55., 56., 57., 58.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 45., 46., 47., 48., 49.,  0.,
           0., 55., 56., 57., 58., 59.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0., 46., 47., 48., 49.,  0.,  0.,  0.,
          56., 57., 58., 59.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]
       ]
    )
    unfold = image.unfold(x, (2, 3), (1, 2), (2, 3), (2, 1))
    self.assertAllClose(ref, unfold)

  def testUnfold3D(self):
    # No reference implementation for 3D unfold - only check shape.
    x = jnp.arange(240).reshape((2, 3, 4, 5, 2))
    x = x.astype(jax.dtypes.canonicalize_dtype(jnp.float64))
    kernel_size = (2, 3, 1)
    padding = 1
    unfold = image.unfold(x, kernel_size, padding=padding)
    patch_size = np.prod(kernel_size) * x.shape[1]
    n_patches = np.prod(tuple(
      x.shape[i + 2] - kernel_size[i] + 1 + 2 * padding
      for i in range(len(kernel_size))))
    self.assertEqual((x.shape[0], patch_size, n_patches), unfold.shape)
    self.assertEqual(x.dtype, unfold.dtype)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
