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

"""Tests for Stax library."""

from absl.testing import absltest

import numpy as np

import jax
from jax._src import test_util as jtu
from jax import random
from jax.example_libraries import stax
from jax import dtypes

jax.config.parse_flags_with_absl()


def random_inputs(rng, input_shape):
  if type(input_shape) is tuple:
    return rng.randn(*input_shape).astype(dtypes.canonicalize_dtype(float))
  elif type(input_shape) is list:
    return [random_inputs(rng, shape) for shape in input_shape]
  else:
    raise TypeError(type(input_shape))


def _CheckShapeAgreement(test_case, init_fun, apply_fun, input_shape):
  rng_key = random.PRNGKey(0)
  rng_key, init_key = random.split(rng_key)
  result_shape, params = init_fun(init_key, input_shape)
  inputs = random_inputs(test_case.rng(), input_shape)
  if params:
    inputs = inputs.astype(np.dtype(params[0]))
  result = apply_fun(params, inputs, rng=rng_key)
  test_case.assertEqual(result.shape, result_shape)


# stax makes use of implicit rank promotion, so we allow it in the tests.
@jtu.with_config(jax_numpy_rank_promotion="allow",
                 jax_legacy_prng_key="allow")
class StaxTest(jtu.JaxTestCase):

  @jtu.sample_product(shape=[(2, 3), (5,)])
  def testRandnInitShape(self, shape):
    key = random.PRNGKey(0)
    out = stax.randn()(key, shape)
    self.assertEqual(out.shape, shape)

  @jtu.sample_product(shape=[(2, 3), (2, 3, 4)])
  def testGlorotInitShape(self, shape):
    key = random.PRNGKey(0)
    out = stax.glorot()(key, shape)
    self.assertEqual(out.shape, shape)

  @jtu.sample_product(
    channels=[2, 3],
    filter_shape=[(1, 1), (2, 3)],
    padding=["SAME", "VALID"],
    strides=[None, (2, 1)],
    input_shape=[(2, 10, 11, 1)],
  )
  def testConvShape(self, channels, filter_shape, padding, strides,
                    input_shape):
    init_fun, apply_fun = stax.Conv(channels, filter_shape, strides=strides,
                                    padding=padding)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(
    channels=[2, 3],
    filter_shape=[(1, 1), (2, 3), (3, 3)],
    padding=["SAME", "VALID"],
    strides=[None, (2, 1), (2, 2)],
    input_shape=[(2, 10, 11, 1)],
  )
  def testConvTransposeShape(self, channels, filter_shape, padding, strides,
                               input_shape):
    init_fun, apply_fun = stax.ConvTranspose(channels, filter_shape,  # 2D
                                               strides=strides, padding=padding)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(
    channels=[2, 3],
    filter_shape=[(1,), (2,), (3,)],
    padding=["SAME", "VALID"],
    strides=[None, (1,), (2,)],
    input_shape=[(2, 10, 1)],
  )
  def testConv1DTransposeShape(self, channels, filter_shape, padding, strides,
                               input_shape):
    init_fun, apply_fun = stax.Conv1DTranspose(channels, filter_shape,
                                               strides=strides, padding=padding)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(
    out_dim=[3, 4],
    input_shape=[(2, 3), (3, 4)],
  )
  def testDenseShape(self, out_dim, input_shape):
    init_fun, apply_fun = stax.Dense(out_dim)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(
    input_shape=[(2, 3), (2, 3, 4)],
    nonlinear=["Relu", "Sigmoid", "Elu", "LeakyRelu"],
  )
  def testNonlinearShape(self, input_shape, nonlinear):
    init_fun, apply_fun = getattr(stax, nonlinear)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(
    window_shape=[(1, 1), (2, 3)],
    padding=["VALID"],
    strides=[None, (2, 1)],
    input_shape=[(2, 5, 6, 4)],
    max_pool=[False, True],
    spec=["NHWC", "NCHW", "WHNC", "WHCN"],
  )
  def testPoolingShape(self, window_shape, padding, strides, input_shape,
                       max_pool, spec):
    layer = stax.MaxPool if max_pool else stax.AvgPool
    init_fun, apply_fun = layer(window_shape, padding=padding, strides=strides,
                                spec=spec)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(input_shape=[(2, 3), (2, 3, 4)])
  def testFlattenShape(self, input_shape):
    init_fun, apply_fun = stax.Flatten
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(
    input_shape=[(2, 5, 6, 1)],
    spec=[
      [stax.Conv(3, (2, 2))],
      [stax.Conv(3, (2, 2)), stax.Flatten, stax.Dense(4)],
    ],
  )
  def testSerialComposeLayersShape(self, input_shape, spec):
    init_fun, apply_fun = stax.serial(*spec)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(input_shape=[(3, 4), (2, 5, 6, 1)])
  def testDropoutShape(self, input_shape):
    init_fun, apply_fun = stax.Dropout(0.9)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shape)

  @jtu.sample_product(input_shape=[(3, 4), (2, 5, 6, 1)])
  def testFanInSum(self, input_shape):
    init_fun, apply_fun = stax.FanInSum
    _CheckShapeAgreement(self, init_fun, apply_fun, [input_shape, input_shape])

  @jtu.sample_product(
    [dict(input_shapes=input_shapes, axis=axis)
      for input_shapes, axis in [
          ([(2, 3), (2, 1)], 1),
          ([(2, 3), (2, 1)], -1),
          ([(1, 2, 4), (1, 1, 4)], 1),
      ]
    ],
  )
  def testFanInConcat(self, input_shapes, axis):
    init_fun, apply_fun = stax.FanInConcat(axis)
    _CheckShapeAgreement(self, init_fun, apply_fun, input_shapes)

  def testIssue182(self):
    key = random.PRNGKey(0)
    init_fun, apply_fun = stax.Softmax
    input_shape = (10, 3)
    inputs = np.arange(30.).astype("float32").reshape(input_shape)

    out_shape, params = init_fun(key, input_shape)
    out = apply_fun(params, inputs)

    assert out_shape == out.shape
    assert np.allclose(np.sum(np.asarray(out), -1), 1.)

  def testBatchNormNoScaleOrCenter(self):
    key = random.PRNGKey(0)
    axes = (0, 1, 2)
    init_fun, apply_fun = stax.BatchNorm(axis=axes, center=False, scale=False)
    input_shape = (4, 5, 6, 7)
    inputs = random_inputs(self.rng(), input_shape)

    out_shape, params = init_fun(key, input_shape)
    out = apply_fun(params, inputs)
    means = np.mean(out, axis=(0, 1, 2))
    std_devs = np.std(out, axis=(0, 1, 2))
    assert np.allclose(means, np.zeros_like(means), atol=1e-4)
    assert np.allclose(std_devs, np.ones_like(std_devs), atol=1e-4)

  def testBatchNormShapeNHWC(self):
    key = random.PRNGKey(0)
    init_fun, apply_fun = stax.BatchNorm(axis=(0, 1, 2))
    input_shape = (4, 5, 6, 7)

    out_shape, params = init_fun(key, input_shape)
    inputs = random_inputs(self.rng(), input_shape).astype(params[0].dtype)
    out = apply_fun(params, inputs)

    self.assertEqual(out_shape, input_shape)
    beta, gamma = params
    self.assertEqual(beta.shape, (7,))
    self.assertEqual(gamma.shape, (7,))
    self.assertEqual(out_shape, out.shape)

  def testBatchNormShapeNCHW(self):
    key = random.PRNGKey(0)
    # Regression test for https://github.com/google/jax/issues/461
    init_fun, apply_fun = stax.BatchNorm(axis=(0, 2, 3))
    input_shape = (4, 5, 6, 7)

    out_shape, params = init_fun(key, input_shape)
    inputs = random_inputs(self.rng(), input_shape).astype(params[0].dtype)
    out = apply_fun(params, inputs)

    self.assertEqual(out_shape, input_shape)
    beta, gamma = params
    self.assertEqual(beta.shape, (5,))
    self.assertEqual(gamma.shape, (5,))
    self.assertEqual(out_shape, out.shape)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
