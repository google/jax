# Copyright 2024 The JAX Authors.
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
"""Tests for TPU specific operations within pallas_call."""

import sys
import unittest

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax.experimental import pallas as pl

if sys.platform != "win32":
  from jax.experimental.pallas import tpu as pltpu
else:
  pltpu = None

try:
  import hypothesis as hp
except (ModuleNotFoundError, ImportError):
  raise unittest.SkipTest("tests depend on hypothesis library")

import hypothesis.strategies as hps

jax.config.parse_flags_with_absl()
jtu.setup_hypothesis(max_examples=100)

_JAX_DTYPES = (
    jnp.float32,
    jnp.bfloat16,
    jnp.int32,
    jnp.int16,
    jnp.int8,
    jnp.bool_,
)


class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if not self.INTERPRET:
      if not jtu.test_device_matches(["tpu"]):
        self.skipTest("Only interpret mode supported on non-TPU")

    super().setUp()

  @classmethod
  def pallas_call(cls, *args, **kwargs):
    return pl.pallas_call(*args, interpret=cls.INTERPRET, **kwargs)


class OpsTest(PallasBaseTest):

  @parameterized.product(from_dtype=_JAX_DTYPES, to_dtype=_JAX_DTYPES)
  def test_bitcast(self, from_dtype, to_dtype):
    # TODO(jevinjiang): remove this after 2nd minor large tiling is enabled.
    if (not jtu.is_device_tpu_at_least(version=5)) and (
        from_dtype in (jnp.int8, jnp.int16) or to_dtype in (jnp.int8, jnp.int16)
    ):
      self.skipTest(
          "Not implemented: packing and unpacking int8, int16 are not supported"
          " on < TPUv5"
      )
    if from_dtype == to_dtype:
      self.skipTest("No bitcast needed")
    if from_dtype == jnp.bool_ or to_dtype == jnp.bool_:
      self.skipTest("Bitcasting with bool is not supported")

    def kernel(x_ref, y_ref):
      y_ref[...] = pltpu.bitcast(x_ref[...], to_dtype)

    m, n = 32, 256
    shape = (m, n)
    out_shape = (m * from_dtype.dtype.itemsize // to_dtype.dtype.itemsize, n)
    inp = np.arange(np.prod(shape), dtype=from_dtype).reshape(shape)
    out = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, to_dtype),
    )(inp)
    if not self.INTERPRET:
      out_interpret = pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(out_shape, to_dtype),
          interpret=True,
      )(inp)
      self.assertAllClose(out, out_interpret)

  @parameterized.product(is_dynamic=(False, True))
  @hp.given(
      axis=hps.integers(0, 3),
      shift=hps.integers(0, 3),
      stride=hps.one_of(hps.just(None), hps.integers(0, 2)),
      # Stride dimension on the minor most is not supported.
      stride_axis=hps.one_of(hps.just(None), hps.integers(0, 2)),
  )
  @hp.example(3, 9, 1, 2)
  @hp.example(3, 9, 2, 2)
  @hp.example(0, 9, 0, 1)
  @hp.example(0, 9, 1, 1)
  def test_roll(self, is_dynamic, axis, shift, stride, stride_axis):
    if (stride is None) != (stride_axis is None):
      self.skipTest(
          "Roll op requires both stride and stride_axis to be either specified"
          " or not specified."
      )
    if (not jtu.is_device_tpu(version=5)) and stride_axis == 2:
      self.skipTest(
          "Roll op with stride axis on 2nd minor requires at least TPU v5"
      )
    shape = (4, 4, 32, 512)

    def kernel(s_ref, x_ref, y_ref):
      amt = s_ref[0] if is_dynamic else shift
      y_ref[...] = pltpu.roll(
          x_ref[...], amt, axis, stride=stride, stride_axis=stride_axis
      )

    def roll(x, shift, axis, stride=None, stride_axis=None):
      assert (stride is None) == (stride_axis is None)
      if stride is None:
        return np.roll(x, shift, axis)
      outputs = [
          np.roll(xs, shift + i * stride, axis)
          for i, xs in enumerate(np.split(x, x.shape[stride_axis], stride_axis))
      ]
      return np.concatenate(outputs, stride_axis)

    inp = np.arange(np.prod(shape), dtype=jnp.int32).reshape(shape)
    ref = roll(inp, shift, axis, stride, stride_axis)
    dynamic_shift = jnp.array([abs(shift)], jnp.int32)
    for interpret in [False, True]:
      out = pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(shape, jnp.int32),
          grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1),
          interpret=interpret,
      )(dynamic_shift, inp)
      np.testing.assert_array_equal(out, ref, err_msg=f"{interpret=}")

  def test_interleave_vectors(self):
    if not jtu.is_device_tpu_at_least(version=4):
      self.skipTest("Expect TPUv4+")

    def kernel(x_ref, y_ref, out_ref):
      x = pltpu.bitcast(x_ref[...].astype(jnp.float32), jnp.int32)
      y = pltpu.bitcast(y_ref[...].astype(jnp.float32), jnp.int32)
      shift = jax.lax.broadcast(16, x.shape)
      out_ref[...] = pltpu.bitcast(
          y | jax.lax.shift_right_logical(x, shift), jnp.bfloat16
      )

    m, n = 16, 128
    inp = np.arange(m * n * 2, dtype=jnp.bfloat16).reshape(m, n * 2)
    x, y = np.split(inp, 2, axis=1)
    out = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m * 2, n), jnp.bfloat16),
    )(x, y)
    np.testing.assert_array_equal(out, inp.reshape(m * 2, n))


class OpsInterpreterTest(OpsTest):
  INTERPRET = True


if __name__ == "__main__":
  absltest.main()
