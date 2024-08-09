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

"""Tests for common JAX operations within pallas_call."""

from collections.abc import Sequence
import contextlib
import functools
import itertools
import sys
from typing import Any
import unittest

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
from jax._src import config
from jax._src import linear_util as lu
from jax._src import state
from jax._src import test_util as jtu
from jax.interpreters import partial_eval as pe
from jax.experimental import pallas as pl

if sys.platform != "win32":
  from jax.experimental.pallas import gpu as plgpu
  from jax.experimental.pallas import tpu as pltpu
else:
  plgpu = None
  pltpu = None

try:
  import hypothesis as hp
except (ModuleNotFoundError, ImportError):
  raise unittest.SkipTest("tests depend on hypothesis library")

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as hps

# There are many inherited redefinitions of _
# ruff: noqa: F811

jax.config.parse_flags_with_absl()
jtu.setup_hypothesis(max_examples=100)


def smem_on_tpu():
  if jtu.test_device_matches(["tpu"]):
    return pltpu.SMEM
  else:
    return None


def _random_value(key: jax.Array, shape_dtype: jax.ShapeDtypeStruct
                  ) -> jax.Array:
  if jnp.issubdtype(shape_dtype.dtype, jnp.floating):
    return random.normal(key, shape_dtype.shape, dtype=shape_dtype.dtype)
  elif jnp.issubdtype(shape_dtype.dtype, jnp.integer):
    return random.randint(
        key, shape_dtype.shape, minval=-4, maxval=4, dtype=shape_dtype.dtype
    )
  raise NotImplementedError(shape_dtype)


_DTYPES = (
    "float32",
    "bfloat16",
    "int32",
    "int16",
    "int8",
    "bool",
)


@hps.composite
def make_shape_dtype_strategy(
    draw, *,
    min_rank: int,
    max_rank: int,
    min_size_exp: int,
    max_size_exp: int,
    valid_dtypes: Sequence[jnp.dtype],
    max_bytes: int = 2**16,
) -> jax.ShapeDtypeStruct:
  dtype = draw(hps.sampled_from(valid_dtypes))
  # To generate shapes with power-of-two sizes, we draw the exponents of the
  # sizes, and then generate the sizes from the exponents.
  shape_exponents = tuple(
      draw(hps.lists(
          hps.integers(min_value=min_size_exp, max_value=max_size_exp),
          min_size=min_rank, max_size=max_rank))
  )
  shape = tuple(2**exp for exp in shape_exponents)
  size = np.prod(shape) * dtype.itemsize
  hp.assume(size <= max_bytes)  # Make sure we don't take more than 4K VMEM
  return jax.ShapeDtypeStruct(shape, dtype)


@hps.composite
def arrays(
    draw, shape: tuple[int, ...], dtype: np.dtype,
    *, elements: hps.SearchStrategy[Any] | None = None,
) -> np.ndarray:
  cast_to_bf16 = False
  if dtype == np.dtype(jnp.bfloat16):
    dtype = np.dtype('float32')
    cast_to_bf16 = True
  arr = draw(hnp.arrays(shape=shape, dtype=dtype, elements=elements))
  if cast_to_bf16:
    arr = arr.astype(np.dtype(jnp.bfloat16))
  return arr


@hps.composite
def select_n_strategy(
    draw, *, max_cases: int = 4,
    min_rank: int = 0, max_rank: int = 2,
    min_size_exp: int = 0, max_size_exp: int = 8,
) -> tuple[np.ndarray, ...]:
  n_cases = draw(hps.integers(min_value=1, max_value=max_cases))
  case_shape_dtype = draw(
      make_shape_dtype_strategy(
          min_rank=min_rank, max_rank=max_rank,
          min_size_exp=min_size_exp, max_size_exp=max_size_exp,
          valid_dtypes=[
              np.dtype("int32"),
              np.dtype("float32"),
              # TODO(sharadmv,apaszke): enable bf16
              # np.dtype(jnp.bfloat16),
          ],
      )
  )
  allowed_elements = hps.integers(min_value=0, max_value=n_cases - 1)
  pred_shape = draw(hps.sampled_from([(), case_shape_dtype.shape]))
  # TODO(sharadmv,apaszke): enable passing bool arrays into Pallas kernels
  if n_cases == 2 and not pred_shape:
    pred_dtype = draw(hps.sampled_from([np.dtype(np.bool_),
                                        np.dtype(np.int32)]))
    allowed_elements = hps.booleans()
  else:
    pred_dtype = np.int32
  pred = draw(arrays(shape=pred_shape, dtype=pred_dtype,
                     elements=allowed_elements))
  cases = (
      draw(
          arrays(shape=case_shape_dtype.shape, dtype=case_shape_dtype.dtype)
      )
      for _ in range(n_cases)
  )
  return pred, *cases


UNARY_PRIMITIVES = [
    # TODO(sharadmv,apaszke): enable zero rank
    # TODO(sharadmv,apaszke): enable one rank
    # TODO(sharadmv,apaszke): enable zero dim sizes
    # TODO(sharadmv,apaszke): enable one dim sizes
    (
        lax.neg_p,
        make_shape_dtype_strategy(
            min_rank=2,
            max_rank=3,
            min_size_exp=1,
            max_size_exp=6,
            valid_dtypes=[jnp.dtype("float32"), jnp.dtype("int32")],
        ),
    ),
    (
        lax.not_p,
        make_shape_dtype_strategy(
            min_rank=2,
            max_rank=3,
            min_size_exp=1,
            max_size_exp=6,
            valid_dtypes=[jnp.dtype("int32")],
        ),
    ),
    *[
        (
            prim,
            make_shape_dtype_strategy(
                min_rank=2,
                max_rank=3,
                min_size_exp=1,
                max_size_exp=6,
                valid_dtypes=[jnp.dtype("float32")],
            ),
        )
        for prim in [
            lax.exp_p,
            lax.tanh_p,
            lax.logistic_p,
            lax.rsqrt_p,
            lax.log_p,
            lax.exp2_p,
            lax.abs_p,
            lax.log1p_p,
            lax.sin_p,
            lax.sqrt_p,
        ]
    ],
]

UNARY_FUNCTIONS = [
    (prim.name, prim.bind, strategy) for prim, strategy in UNARY_PRIMITIVES
] + [
    (
        name,
        func,
        make_shape_dtype_strategy(
            min_rank=2,
            max_rank=3,
            min_size_exp=1,
            max_size_exp=6,
            valid_dtypes=[jnp.dtype("float32")],
        ),
    )
    for name, func in [
        ("relu", jax.nn.relu),
        ("pow2", lambda x: jnp.power(2, x)),
        ("square", jnp.square),
        ("reciprocal", jnp.reciprocal),
        ("round", jnp.round),
        ("rint", jnp.rint),
    ]
]


class PallasBaseTest(jtu.JaxTestCase):
  INTERPRET = False

  def setUp(self):
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")
    if not self.INTERPRET:
      if jtu.device_under_test() == "cpu":
        self.skipTest("Only interpreter mode supported on CPU")
      if (jtu.test_device_matches(["cuda"]) and
          not jtu.is_cuda_compute_capability_at_least("8.0")):
        self.skipTest("Only works on GPUs with capability >= sm80")

    super().setUp()

  @classmethod
  def pallas_call(cls, *args, **kwargs):
    return pl.pallas_call(*args, interpret=cls.INTERPRET, **kwargs)


class OpsTest(PallasBaseTest):

  def setUp(self):
    super().setUp()
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")

  @parameterized.named_parameters(
      (fn.__name__, fn, dtype) for fn, dtype in [
          (lax.pow, jnp.float32),
          (lax.bitwise_and, jnp.int32),
          (lax.bitwise_or, jnp.int32),
          (lax.bitwise_xor, jnp.int32),
          (lax.shift_left, jnp.int32),
          (lax.shift_right_arithmetic, jnp.int32),
          (lax.shift_right_logical, jnp.int32),
      ]
  )
  def test_weak_dtype(self, fn, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8, 128), dtype),
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = fn(x_ref[...], y_ref[...])

    x = jnp.full((8, 128), 4, dtype=dtype)
    y = jnp.full((8, 128), 2 if jnp.issubdtype(dtype, jnp.integer) else 2.0,
                 dtype=dtype)
    np.testing.assert_allclose(kernel(x, y), fn(x, y))

  @parameterized.named_parameters(
      ('integer_1_1', (1, 1)),
      ('integer_1_16', (1, 16)),
      ('integer_16_1', (16, 1)),
      ('integer_-1_1', (-1, 1)),
      ('integer_1_-1', (1, -1)),
      ('float_1_1', (1.0, 1.0)),
      ('float_1_16', (1.0, 16.0)),
      ('float_16_1', (16.0, 1.0)),
      ('float_-1_1', (-1.0, 1.0)),
      ('float_1_-1', (1.0, -1.0)),
      ('float_1_inf', (1.0, float('inf'))),
      ('float_inf_1', (float('inf'), 1.0)),
      ('float_inf_inf', (float('inf'), float('inf'))),
      ('float_1_nan', (1.0, float('nan'))),
      ('float_nan_1', (float('nan'), 1.0)),
      ('float_nan_nan', (float('nan'), float('nan'))),
      ('float_inf_nan', (float('inf'), float('nan'))),
      ('float_nan_inf', (float('inf'), float('inf'))),
  )
  def test_scalar_compare(self, params):
    """Test some scalar compares.

    We don't really expect that the results would be wrong, but rather we want
    to exercise the lowering rules.
    """

    def kernel(x_ref, y_ref, o_ref):
      x = x_ref[0, 0]
      y = y_ref[0, 0]
      o_ref[0, 0] = jax.lax.select(x == y, 1, 0)
      o_ref[0, 1] = jax.lax.select(x != y, 1, 0)
      o_ref[0, 2] = jax.lax.select(x < y, 1, 0)
      o_ref[0, 3] = jax.lax.select(x <= y, 1, 0)
      o_ref[0, 4] = jax.lax.select(x > y, 1, 0)
      o_ref[0, 5] = jax.lax.select(x >= y, 1, 0)

    x, y = params
    r = jnp.array(
        [
            [x == y, x != y, x < y, x <= y, x > y, x >= y],
        ],
        jnp.int32,
    )
    x = jnp.array([[x]])
    y = jnp.array([[y]])

    result = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([1, 128], jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=smem_on_tpu()),
            pl.BlockSpec(memory_space=smem_on_tpu()),
        ],
        out_specs=pl.BlockSpec(
            (1, 128), lambda i: (0, 0), memory_space=smem_on_tpu()
        ),
        grid=(1,),
    )(x, y)
    np.testing.assert_array_equal(r, result[..., 0:6])

  @parameterized.named_parameters(
      ('integer_1_1', (1, 1)),
      ('integer_1_16', (1, 16)),
      ('integer_16_1', (16, 1)),
      ('integer_-1_1', (-1, 1)),
      ('integer_1_-1', (1, -1)),
      ('float_1_1', (1.0, 1.0)),
      ('float_1_16', (1.0, 16.0)),
      ('float_16_1', (16.0, 1.0)),
      ('float_-1_1', (-1.0, 1.0)),
      ('float_1_-1', (1.0, -1.0)),
      ('float_1_inf', (1.0, float('inf'))),
      ('float_inf_1', (float('inf'), 1.0)),
      ('float_inf_inf', (float('inf'), float('inf'))),
      ('float_1_nan', (1.0, float('nan'))),
      ('float_nan_1', (float('nan'), 1.0)),
      ('float_nan_nan', (float('nan'), float('nan'))),
      ('float_inf_nan', (float('inf'), float('nan'))),
      ('float_nan_inf', (float('inf'), float('inf'))),
  )
  def test_vector_compare(self, params):
    """Test some vector compares.

    We don't really expect that the results would be wrong, but rather we want
    to exercise the lowering rules.
    """

    def kernel(x_ref, y_ref, o_ref):
      x = x_ref[:]
      y = y_ref[:]
      one = jnp.ones([8, 128], dtype=jnp.int32)
      zero = jnp.zeros([8, 128], dtype=jnp.int32)
      o_ref[0] = jax.lax.select(x == y, one, zero)
      o_ref[1] = jax.lax.select(x != y, one, zero)
      o_ref[2] = jax.lax.select(x < y, one, zero)
      o_ref[3] = jax.lax.select(x <= y, one, zero)
      o_ref[4] = jax.lax.select(x > y, one, zero)
      o_ref[5] = jax.lax.select(x >= y, one, zero)

    # Widen out our params to (8, 128) vectors.
    x, y = params
    x = jnp.full([8, 128], x)
    y = jnp.full([8, 128], y)

    r = [x == y, x != y, x < y, x <= y, x > y, x >= y]

    result = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([6, 8, 128], jnp.int32),
        in_specs=[
            pl.BlockSpec((8, 128), lambda *_: (0, 0)),
            pl.BlockSpec((8, 128), lambda *_: (0, 0)),
        ],
        out_specs=pl.BlockSpec((6, 8, 128), lambda *_: (0, 0, 0)),
        grid=(1,),
    )(x, y)
    np.testing.assert_array_equal(r[0], result[0])
    np.testing.assert_array_equal(r[1], result[1])
    np.testing.assert_array_equal(r[2], result[2])
    np.testing.assert_array_equal(r[3], result[3])
    np.testing.assert_array_equal(r[4], result[4])
    np.testing.assert_array_equal(r[5], result[5])

  @parameterized.named_parameters(
      ("reduce_all_true", "all_true", jnp.all, True),
      ("reduce_all_false", "all_false", jnp.all, False),
      ("reduce_all_mixed", "one_false", jnp.all, False),
      ("reduce_any_true", "all_true", jnp.any, True),
      ("reduce_any_false", "all_false", jnp.any, False),
      ("reduce_any_mixed", "one_false", jnp.any, True),
  )
  def test_reduce_boolean(self, input_type, reduction_op, expected_result):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("TODO: error on GPU")

    def kernel(x_ref, ones_ref, o_ref):
      # Convert float to bool with a comparison.
      bool_x = x_ref[...] == ones_ref[...]
      reduced_as_bool = reduction_op(bool_x, keepdims=True)
      # Convert bool to float with a select.
      float_value = jnp.where(reduced_as_bool, 1.0, 0.0)
      o_ref[0, 0] = float_value[0, 0]

    if input_type == 'all_true':
      x = jnp.ones((8, 128), dtype=jnp.float32)
    elif input_type == 'all_false':
      x = jnp.zeros((8, 128), dtype=jnp.float32)
    elif input_type == 'one_false':
      x = jnp.ones((8, 128), dtype=jnp.float32)
      x = x.at[0, 0].set(0.0)
    ones = jnp.ones_like(x)

    result = self.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec((8, 128), lambda *_: (0, 0)),
            pl.BlockSpec((8, 128), lambda *_: (0, 0)),
        ],
        out_specs=pl.BlockSpec(block_shape=(1, 1), memory_space=smem_on_tpu()),
        out_shape=jax.ShapeDtypeStruct([1, 1], jnp.float32),
        grid=(1,),
    )(x, ones)
    np.testing.assert_array_equal(result[0, 0], float(expected_result))

  @parameterized.named_parameters(
      ("sum", jnp.sum,), ("max", jnp.max,), ("min", jnp.min,)
  )
  def test_reduce_float(self, reduction_op):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("TODO: error on GPU")

    def kernel(x_ref, o_ref):
      o_ref[0, 0] = reduction_op(x_ref[...])

    x = jax.random.normal(jax.random.key(0), (8, 128))
    result = self.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec((8, 128), lambda *_: (0, 0)),
        ],
        out_specs=pl.BlockSpec((1, 1), memory_space=smem_on_tpu()),
        out_shape=jax.ShapeDtypeStruct([1, 1], jnp.float32),
        grid=(1,),
    )(x)

    np.testing.assert_allclose(result[0, 0], reduction_op(x), atol=1e-5)

  # TODO(sharadmv): test rank < 2, size < 2
  @hp.given(select_n_strategy(max_cases=2, min_rank=2, max_rank=4,
                              min_size_exp=1))
  def test_select_n(self, args):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("TODO: error on GPU, lowering bug for select_n")
    pred, *cases = args
    scalar_pred = not pred.shape

    def kernel(*refs):
      if scalar_pred:
        *case_refs, o_ref = refs
        pred_ = pred
      else:
        pred_ref, *case_refs, o_ref = refs
        pred_ = pred_ref[...]
      vals = [case_ref[...] for case_ref in case_refs]
      o_ref[...] = lax.select_n(pred_, *vals)
    out_ref = lax.select_n(pred, *cases)
    if scalar_pred:
      args = cases
    else:
      args = [pred, *cases]
    out = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(out_ref.shape, out_ref.dtype),
    )(*args)
    if out.dtype == jnp.bfloat16:
      out, out_ref = out.astype(jnp.float32), out_ref.astype(jnp.float32)
    np.testing.assert_allclose(out, out_ref)

  @parameterized.named_parameters(
      (name, name, func, strategy)
      for name, func, strategy in UNARY_FUNCTIONS
  )
  @hp.given(hps.data())
  def test_unary_primitives(self, name, func, shape_dtype_strategy, data):
    if self.INTERPRET:
      self.skipTest("This hypothesis test is slow, even more so in interpret mode.")
    # We want exact equality here to match how JAX lowers to XLA
    tol = 0.
    if jtu.test_device_matches(["gpu"]):
      if func == jnp.round or func == jnp.rint:
        self.skipTest("TODO: not implemented on GPU")
      if name == "tanh":
        tol = 1e-6
      elif name == "exp2":
        tol = 1e-6
    elif jtu.test_device_matches(["tpu"]):
      if not jtu.is_device_tpu_at_least(version=5) and False:
        self.skipTest("TODO: not implemented on TPU v{3,4}")

    def kernel(x_ref, y_ref):
      y_ref[...] = func(x_ref[...])
    x_shape_dtype = data.draw(shape_dtype_strategy)
    key = random.key(0)
    x = _random_value(key, x_shape_dtype)
    out = self.pallas_call(kernel, out_shape=x_shape_dtype)(x)
    self.assertAllClose(out, func(x), atol=tol, rtol=tol)

  @parameterized.product(from_dtype=_DTYPES, to_dtype=_DTYPES)
  @hp.given(hps.data())
  def test_cast(self, from_dtype, to_dtype, data):
    if from_dtype == to_dtype:
      self.skipTest("Unnecessary test")
    if jtu.is_device_tpu(version=4):
      if from_dtype in {"int16", "int8"} or to_dtype in {"int16", "int8"}:
        self.skipTest(
            "Not supported: TPU generation doesn't support this cast."
        )
    if jtu.test_device_matches(["tpu"]) and jtu.get_tpu_version() < 4:
      if from_dtype in {"int32", "float32", "bfloat16"} and to_dtype in {"int16", "int8"}:
        self.skipTest(
            "Not supported: TPU generation doesn't support this cast."
        )

    # TODO(sharadmv,apaszke): add support for the following casts
    if from_dtype == "int16" and to_dtype == "int8":
      self.skipTest("Not supported: bad canonicalization")
    if from_dtype == "int8" and to_dtype == "int16":
      self.skipTest("Not supported: bad canonicalization")
    if from_dtype == "bool" and to_dtype in {"int16", "int8"}:
      self.skipTest("Not supported: cannot extend to sub-32 bit types")
    if from_dtype in {"int32", "bfloat16", "float32"} and to_dtype == "bool":
      self.skipTest("Not supported: unsupported relayout")
    if from_dtype == "bool" and to_dtype in {"int32", "bfloat16", "float32"}:
      self.skipTest("Not supported: unsupported relayout")
    if from_dtype in {"int16", "int8"} and to_dtype == "bool":
      self.skipTest("Not supported: cannot truncate from sub-32 bit types")
    if from_dtype in {"int16", "int8"} and to_dtype == "bool":
      self.skipTest("Not supported: cannot truncate from sub-32 bit types")
    if jtu.test_device_matches(["gpu"]):
      if (from_dtype in {"bfloat16", "float32"} and
          to_dtype in {"int8", "int16", "int32"}):
        self.skipTest("TODO: wrong result on GPU")

    if from_dtype == "bfloat16":
      from_dtype = jnp.bfloat16
    if to_dtype == "bfloat16":
      to_dtype = jnp.bfloat16

    if from_dtype == jnp.bfloat16:
      x = jnp.asarray(data.draw(hnp.arrays(jnp.float32, (8, 128))))
      x = x.astype(jnp.bfloat16)
    else:
      x = data.draw(hnp.arrays(from_dtype, (8, 128)))
    x = jnp.asarray(x)
    if from_dtype == jnp.dtype("bool"):
      x = x.astype(jnp.int32)
    def kernel(x_ref, y_ref):
      x = x_ref[...]
      if from_dtype == jnp.dtype("bool"):
        x = x.astype(jnp.dtype("bool"))
      y = x.astype(to_dtype)
      if to_dtype == jnp.dtype("bool"):
        y = y.astype(jnp.int32)
      y_ref[...] = y
    if (y_dtype := to_dtype) == jnp.dtype("bool"):
      y_dtype = jnp.int32
    y = self.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(x.shape, y_dtype))(x)
    if to_dtype == jnp.dtype("bool"):
      y = y.astype(jnp.dtype("bool"))
    y_ref = x.astype(to_dtype)
    if to_dtype == jnp.bfloat16:
      y, y_ref = y.astype(np.float32), y_ref.astype(np.float32)
    np.testing.assert_allclose(y, y_ref, atol=0., rtol=0.)

  @parameterized.product(
      shape=((64,), (8, 8)),
      dtype=(jnp.int32, jnp.int16, jnp.int8),
  )
  def test_scalar_map(self, shape, dtype):
    if dtype != jnp.int32 and len(shape) < 2:
      # TODO(b/299280718): Implement this.
      self.skipTest(
          "Loads and stores not implemented for 1D arrays of non-32bit types"
      )
    def kernel(x_ref, y_ref):
      for idx in np.ndindex(shape):
        x = x_ref[idx].astype(jnp.int32)
        y_ref[idx] = (x * x).astype(y_ref.dtype)
    f = self.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
    )
    x = jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)
    self.assertAllClose(f(x), x * x)

  @jtu.skip_on_devices("gpu")  # TODO: not implemented
  def test_extract_scalar(self):
    def kernel(x_ref, y_ref):
      y_ref[0, 0] = x_ref[:][0, 0]
    f = self.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, 1), jnp.float32),
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
    )
    x = np.arange(1024, dtype=jnp.float32).reshape(8, 128) + 10
    self.assertAllClose(f(x).item(), 10.0)

  @jtu.skip_on_devices("gpu")  # TODO: not implemented
  def test_concat_constant(self):
    def kernel(out):
      result = []
      for i in range(16):
        result.append(jnp.full((1, 128), i, jnp.float32))
      out[:] = jnp.stack(result).reshape(16, 128)

    def run(interpret=False):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
          out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
          interpret=interpret,
      )()
    expected = run(True)
    if not self.INTERPRET:
      actual = run(False)
      self.assertAllClose(actual, expected)


class OpsInterpreterTest(OpsTest):
  INTERPRET = True

  def test_debug_print(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        grid=1,
    )
    def kernel(x_ref, o_ref):
      jax.debug.print("x = {}", x_ref)

    x = jnp.array([4.2, 2.4]).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
      jax.effects_barrier()

    self.assertIn("x = [4.2 2.4]", output())


class OpsExtraTest(PallasBaseTest):
  """These are additional ops tests that have not been ported to TPU yet."""
  # TODO: fix these for TPU and merge with OpsTest.

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["tpu"]) and not self.INTERPRET:
      # TODO: most tests fail on TPU in non-interpreter mode
      self.skipTest("On TPU the test works only in interpret mode")

  ELEMENTWISE_OPS = [
      (
          [jnp.abs, jnp.negative],
          ["int16", "int32", "int64", "float16", "float32", "float64"],
      ),
      ([jnp.ceil, jnp.floor], ["float32", "float64", "int32"]),
      (
          [jnp.exp, jnp.exp2, jnp.sin, jnp.cos, jnp.log, jnp.sqrt],
          ["float16", "float32", "float64"],
      ),
      (
          # fmt: off
          [jnp.expm1, jnp.log1p, jnp.cbrt, lax.rsqrt, jnp.tan, jnp.asin,
           jnp.acos, jnp.atan, jnp.sinh, jnp.cosh, jnp.asinh, jnp.acosh,
           jnp.atanh],
          # fmt: on
          ["float32", "float64"],
      ),
      ([lax.population_count, lax.clz, jnp.invert], ["int32", "int64"]),
  ]

  @parameterized.named_parameters(
      (f"{fn.__name__}_{dtype}", fn, dtype)
      for args in ELEMENTWISE_OPS
      for fn, dtype in itertools.product(*args)
  )
  def test_elementwise(self, fn, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((2,), dtype), grid=1
    )
    def kernel(x_ref, o_ref):
      o_ref[:] = fn(x_ref[...])

    with contextlib.ExitStack() as stack:
      if jnp.dtype(dtype).itemsize == 8:
        stack.enter_context(config.enable_x64(True))
      x = jnp.array([0.42, 2.4]).astype(dtype)
      np.testing.assert_allclose(kernel(x), fn(x), rtol=1e-6)

  @parameterized.parameters(
      ("float32", "int32"),
      ("float64", "int32"),
      ("float32", "float32"),
      ("float64", "float64"),
  )
  def test_pow(self, x_dtype, y_dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), x_dtype), grid=1
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[:] = lax.pow(x_ref[...], y_ref[...])

    with contextlib.ExitStack() as stack:
      if jnp.dtype(x_dtype).itemsize == 8:
        stack.enter_context(config.enable_x64(True))
      x = jnp.array([1, 2, 3, 4]).astype(x_dtype)
      y = jnp.array([1, 2, 3, 4]).astype(y_dtype)
      np.testing.assert_allclose(kernel(x, y), lax.pow(x, y))

  @parameterized.parameters(0, 1, 2, 3, 4, 5, -1, -2, -3)
  def test_integer_pow(self, y):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[:] = lax.integer_pow(x_ref[...], y)

    x = jnp.array([1, 2, 3, 4]).astype(jnp.float32) / 10
    np.testing.assert_allclose(kernel(x), lax.integer_pow(x, y))

  @parameterized.parameters("float32", "float64")
  def test_nextafter(self, dtype):
    if jtu.test_device_matches(["tpu"]) and dtype == "float64":
      self.skipTest("float64 disabled on TPU.")
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), dtype), grid=1
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[:] = jnp.nextafter(x_ref[...], y_ref[...])

    with contextlib.ExitStack() as stack:
      if jnp.dtype(dtype).itemsize == 8:
        stack.enter_context(config.enable_x64(True))
      x = jnp.array([1, 2, 3, 4]).astype(dtype)
      y = jnp.array([1, 2, 3, 4]).astype(dtype)
      np.testing.assert_allclose(kernel(x, y), jnp.nextafter(x, y))

  COMPARISON_OPS = [
      jnp.equal,
      jnp.not_equal,
      jnp.less,
      jnp.less_equal,
      jnp.greater,
      jnp.greater_equal,
  ]

  @parameterized.named_parameters(
      (f"{fn.__name__}_{dtype}", fn, dtype)
      for fn, dtype in itertools.product(
          COMPARISON_OPS, ["int32", "uint32", "float16", "float32", "bool"]
      )
  )
  def test_comparison(self, fn, dtype):
    if jtu.test_device_matches(["gpu"]) and dtype == "bool":
      self.skipTest("Not implemented on GPU.")

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), jnp.bool_),
        grid=1)
    def kernel(x_ref, y_ref, o_ref):
      o_ref[:] = fn(x_ref[...], y_ref[...])

    x = jnp.array([0, 3, -4, -6, 0, 5, 4, -7]).astype(dtype)
    y = jnp.array([3, 1, -4, -5, 0, -2, 2, 4]).astype(dtype)
    np.testing.assert_allclose(kernel(x, y), fn(x, y))

  def test_isnan(self):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), jnp.bool_),
        grid=1)
    def isnan(x_ref, o_ref):
      o_ref[:] = jnp.isnan(x_ref[...])

    x = jnp.arange(8.)
    x = x.at[3].set(jnp.nan)
    np.testing.assert_allclose(isnan(x), jnp.isnan(x))

  @parameterized.parameters(
      ("int32", "float32"),
      ("float32", "float32"),
  )
  def test_true_divide(self, dtype, out_dtype):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8,), out_dtype),
        grid=1,
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = jnp.true_divide(x_ref[...], y_ref[...])

    x = jnp.array([1, 3, -4, -6, 2, 5, 4, -7]).astype(dtype)
    y = jnp.array([3, 1, -4, -5, 2, -2, 2, 4]).astype(dtype)
    np.testing.assert_allclose(jnp.true_divide(x, y), kernel(x, y))

  @parameterized.parameters("float16", "bfloat16")
  def test_true_divide_unsupported(self, dtype):
    if self.INTERPRET:
      self.skipTest("No lowering in interpreter mode")

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), dtype),
        grid=1,
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = jnp.true_divide(x_ref[...], y_ref[...])

    x = jnp.array([2.4, 4.2]).astype(dtype)
    y = jnp.array([4.2, 2.4]).astype(dtype)
    with self.assertRaises(Exception):
      kernel(x, y)

  BINARY_OPS = [
      ([jnp.floor_divide], ["int32", "uint32"]),
      (
          [jnp.add, jnp.subtract, jnp.multiply],
          ["int16", "int32", "uint32", "float16", "float32"],
      ),
      ([jnp.remainder], ["int32", "uint32", "float32"]),
      (
          # fmt: off
          [jnp.bitwise_and, jnp.bitwise_or, jnp.bitwise_xor,
           jnp.bitwise_left_shift, jnp.bitwise_right_shift],
          # fmt: on
          ["int32", "uint32"],
      ),
  ]

  @parameterized.named_parameters(
      (f"{fn.__name__}_{dtype}", fn, dtype)
      for args in BINARY_OPS
      for fn, dtype in itertools.product(*args)
  )
  def test_binary(self, f, dtype):
    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((8,), dtype), grid=1
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = f(x_ref[...], y_ref[...])

    x = jnp.array([1, 3, -4, -6, 2, 5, 4, -7]).astype(dtype)
    if (f == jnp.bitwise_left_shift):
      y = jnp.array([3, 1, 4, 5, 2, 2, 2, 4]).astype(dtype)
    else:
      y = jnp.array([3, 1, -4, -5, 2, -2, 2, 4]).astype(dtype)

    np.testing.assert_allclose(f(x, y), kernel(x, y))

  @parameterized.parameters(
      ((8, 4), jnp.int32, 0),
      ((8, 16), jnp.float32, 1),
      ((8, 16, 2), jnp.int8, 1),
  )
  def test_broadcasted_iota(self, shape, dtype, dimension):
    f = lambda: jax.lax.broadcasted_iota(dtype, shape, dimension)

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct(shape, dtype), grid=1
    )
    def kernel(o_ref):
      o_ref[...] = f()

    np.testing.assert_allclose(f(), kernel())

  @parameterized.parameters("float16", "bfloat16", "float32")
  def test_approx_tanh(self, dtype):
    if self.INTERPRET:
      self.skipTest("approx_tanh is not supported in interpreter mode")
    if (dtype == "bfloat16" and
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("tanh.approx.bf16 requires a GPU with capability >= sm90")

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((4,), dtype), grid=1
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = plgpu.approx_tanh(x_ref[...])

    x = jnp.asarray([-1, 0.42, 0.24, 1]).astype(dtype)
    # We upcast to float32 because NumPy <2.0 does not handle custom dtypes
    # properly. See https://github.com/google/jax/issues/11014.
    np.testing.assert_allclose(
        kernel(x).astype(jnp.float32),
        jnp.tanh(x).astype(jnp.float32),
        atol=5e-3,
        rtol=5e-3,
    )

  def test_elementwise_inline_asm(self):
    if self.INTERPRET:
      self.skipTest(
          "elementwise_inline_asm is not supported in interpreter mode"
      )

    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float16),
        grid=1,
    )
    def kernel(x_ref, o_ref):
      [o_ref[...]] = plgpu.elementwise_inline_asm(
          "tanh.approx.f16x2 $0, $1;",
          args=[x_ref[...]],
          constraints="=r,r",
          pack=2,
          result_shape_dtypes=[jax.ShapeDtypeStruct(x_ref.shape, x_ref.dtype)],
      )

    x = jnp.arange(256).astype(jnp.float16)
    np.testing.assert_allclose(kernel(x), jnp.tanh(x), atol=5e-3, rtol=5e-3)

  def test_debug_print(self):
    # TODO: this test flakes on gpu
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test flakes on gpu")
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        grid=1,
        compiler_params=dict(triton=dict(num_warps=1, num_stages=1))
    )
    def kernel(x_ref, o_ref):
      pl.debug_print("It works!")

    x = jnp.array([4.2, 2.4]).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
      jax.effects_barrier()

    self.assertIn("It works!", output())

  def test_debug_print_with_values(self):
    # TODO: this test flakes on gpu
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("This test flakes on gpu")
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((2,), jnp.float32),
        grid=1,
        compiler_params=dict(triton=dict(num_warps=1, num_stages=1))
    )
    def kernel(x_ref, o_ref):
      pl.debug_print("x[0] =", x_ref[0])

    x = jnp.array([4.2, 2.4]).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
      jax.effects_barrier()

    self.assertIn("x[0] = 4.2", output())

  @parameterized.parameters(
      ((2, 4), (8,)),
      ((2, 4), (8, 1)),
      ((2, 4), (1, 8)),
      ((64,), (32, 2)),
  )
  def test_reshape(self, in_shape, out_shape):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1,
    )
    def f(x_ref, o_ref):
      o_ref[...] = x_ref[...].reshape(out_shape)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = x.reshape(out_shape)
    np.testing.assert_allclose(f(x), expected)

  @parameterized.parameters(
      # fmt: off
      ((), (1,)),
      ((), (1, 1)),
      ((2, 4), (2, 4)),
      ((2, 4), (2, 4, 1)),
      ((2, 4, 1), (2, 4)),
      ((2, 4), (1, 2, 4)),
      ((1, 2, 4), (2, 4)),
      ((2, 4), (2, 1, 4)),
      ((1, 2, 1, 4, 1), (2, 4)),
      ((2, 4,), (1, 2, 1, 4)),
      ((2, 4,), (1, 2, 4, 1)),
      ((1, 2, 4, 1), (1, 2, 1, 4, 1)),
      # fmt: on
  )
  def test_reshape_noop_or_singleton_dims(self, in_shape, out_shape):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1,
    )
    def f(x_ref, o_ref):
      o_ref[...] = x_ref[...].reshape(out_shape)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = x.reshape(out_shape)
    np.testing.assert_allclose(f(x), expected)

  def test_num_programs(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.int32),
        grid=4,
    )
    def kernel(o_ref):
      o_ref[pl.program_id(0)] = pl.num_programs(0)

    np.testing.assert_array_equal(
        kernel(), np.asarray([4, 4, 4, 4], dtype=np.int32)
    )

  def test_where_broadcasting(self):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4, 2, 2), jnp.float32),
        grid=1,
    )
    def copyitem(x_ref, in_idx_ref, out_idx_ref, o_ref):
      mask = (jnp.arange(o_ref.shape[0]) == out_idx_ref[()])[:, None, None]
      o_ref[...] = jnp.where(mask, x_ref[in_idx_ref[()]], 0)

    x = jnp.arange(7 * 2 * 2.0).reshape(7, 2, 2)
    for ii in range(7):
      for oi in range(4):
        out = copyitem(x, ii, oi)
        self.assertEqual((4, 2, 2), out.shape)
        np.testing.assert_allclose(out[:oi], jnp.zeros_like(out[:oi]))
        np.testing.assert_allclose(out[oi], x[ii])
        np.testing.assert_allclose(out[oi + 1 :], jnp.zeros_like(out[oi + 1 :]))

  @parameterized.parameters(
      ((), (2,), ()),
      ((1,), (2,), (0,)),
      ((1, 1), (2, 2), (0, 1)),
      ((), (2, 2), ()),
  )
  def test_broadcast_in_dim(self, in_shape, out_shape, dims):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=1,
    )
    def f(x_ref, o_ref):
      x = x_ref[...]
      o_ref[...] = jax.lax.broadcast_in_dim(x, out_shape, dims)

    x = jnp.arange(int(np.prod(in_shape)), dtype=jnp.float32).reshape(in_shape)
    expected = jax.lax.broadcast_in_dim(x, out_shape, dims)
    np.testing.assert_allclose(f(x), expected)

  @parameterized.product(
      size=[16, 32, 64],
      dtype=["float32", "float16"],
      trans_x=[False, True],
      trans_y=[False, True],
  )
  def test_dot(self, size, dtype, trans_x, trans_y):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((size, size), dtype),
        grid=1,
    )
    def dot(x_ref, y_ref, o_ref):
      x = x_ref[:, :]
      y = y_ref[:, :]
      o_ref[:, :] = pl.dot(x, y, trans_x, trans_y).astype(o_ref.dtype)

    k1, k2 = random.split(random.key(0))
    x = random.normal(k1, (size, size), dtype=dtype)
    y = random.normal(k2, (size, size), dtype=dtype)
    out = dot(x, y)
    expected = jnp.dot(x.T if trans_x else x, y.T if trans_y else y)
    np.testing.assert_allclose(out, expected, atol=0.05, rtol=0.05)

  @parameterized.product(
      size=[1, 2, 64, 129, 1021],
      block_size=[1, 2, 32, 64, 128],
  )
  def test_masked_load_store(self, size, block_size):
    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((size,), jnp.float32)),
        grid=pl.cdiv(size, block_size),
    )
    def kernel(x_ref, o_ref):
      idx = pl.program_id(0) * block_size + jnp.arange(block_size)
      mask = idx < x_ref.shape[0]
      x = pl.load(x_ref, (idx,), mask=mask)
      pl.store(o_ref, (idx,), x + 1.0, mask=mask)

    key = random.key(0)
    x = random.normal(key, (size,))
    np.testing.assert_allclose(kernel(x), x + 1.0, atol=1e-5, rtol=1e-5)

  def test_masked_oob_load_store_slice(self):
    n = 16

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((n,), jnp.float32)),
        grid=1,
    )
    def masked_oob_load_store_slice(x_ref, mask_ref, start_idx_ref, o_ref):
      x = pl.load(x_ref, (pl.dslice(start_idx_ref[()], n)),
                  mask=mask_ref[:], other=-1.)
      pl.store(o_ref, (pl.dslice(None),), x)

    x = random.normal(random.key(0), (n,))
    slice_start = random.randint(random.key(2), (), 1, n)
    indices = jnp.arange(n) + slice_start
    mask = indices < n
    out = masked_oob_load_store_slice(x, mask, slice_start)
    o_new = jnp.where(mask, x[indices], jnp.full_like(x, -1.))
    np.testing.assert_array_equal(out, o_new)

  def test_strided_load(self):
    if self.INTERPRET:
      # TODO(b/329733289): Remove this once the bug is fixed.
      self.skipTest("Strided load not yet supported in interpreter mode")

    # Reproducer from https://github.com/google/jax/issues/20895.
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[::4]

    x = jnp.arange(16, dtype=jnp.float32)
    np.testing.assert_array_equal(kernel(x), x[::4])

  def test_broadcasted_load_store(self):
    m, n = 16, 32

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((m, n), jnp.float32)),
        grid=1,
    )
    def load(x_ref, o_ref):
      x = pl.load(x_ref, (jnp.arange(m)[:, None], jnp.arange(n)[None, :]))
      pl.store(o_ref, (jnp.arange(m)[:, None], jnp.arange(n)[None, :]), x + 1.0)

    key = random.key(0)
    x = random.normal(key, (m, n))
    np.testing.assert_allclose(load(x), x + 1.0, atol=1e-5, rtol=1e-5)

  @parameterized.parameters(
      ((16, 32), (16,)),
      ((16, 32), (32,)),
      ((16, 32), (16, 16)),
  )
  def test_invalid_broadcasted_load(self, x_shape, mask_shape):
    if self.INTERPRET:
      self.skipTest("No broadcasting checks in pl.load in interpreter mode")

    @functools.partial(
        self.pallas_call, out_shape=jax.ShapeDtypeStruct((), jnp.float32)
    )
    def kernel(x_ref, mask_ref, o_ref):
      del o_ref  # Unused.
      pl.load(x_ref, slice(None), mask=mask_ref[:])

    x = jnp.ones(x_shape, dtype=jnp.float32)
    mask = jnp.ones(mask_shape, dtype=jnp.bool_)
    # assertRaises* methods do not support inspecting the __cause__, so
    # we have to check it manually.
    try:
      kernel(x, mask)
    except Exception as e:
      self.assertIn("Cannot broadcast", str(e.__cause__))
    else:
      self.fail("Expected exception due to invalid broadcasting")

  def test_swap(self):
    m, n = 16, 32

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((m, n), jnp.float32),) * 2,
        grid=1,
        input_output_aliases={0: 0, 1: 1},
    )
    def swap(_, _2, x_ref, y_ref):
      x = x_ref[:]
      y = pl.swap(y_ref, (slice(None),), x)
      x_ref[:] = y

    x = random.normal(random.key(0), (m, n))
    y = random.normal(random.key(1), (m, n))
    out = swap(x, y)
    np.testing.assert_array_equal(out[0], y)
    np.testing.assert_array_equal(out[1], x)

  def test_masked_swap(self):
    m, n = 16, 32

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((m, n), jnp.float32),) * 2,
        grid=1,
        input_output_aliases={0: 0, 1: 1},
    )
    def masked_swap(_, _2, mask_ref, x_ref, y_ref):
      x = x_ref[:]
      y = pl.swap(y_ref, (slice(None),), x, mask=mask_ref[:])
      x_ref[:] = y

    x = random.normal(random.key(0), (m, n))
    y = random.normal(random.key(1), (m, n))
    mask = random.bernoulli(random.key(2), shape=(m, n))
    out = masked_swap(x, y, mask)
    np.testing.assert_array_equal(out[0], jnp.where(mask, y, x))
    np.testing.assert_array_equal(out[1], jnp.where(mask, x, y))

  def test_masked_oob_swap_slice(self):
    m, n = 32, 16

    @functools.partial(
        self.pallas_call,
        out_shape=(jax.ShapeDtypeStruct((n,), jnp.float32),
                   jax.ShapeDtypeStruct((m,), jnp.float32)),
        grid=1,
        input_output_aliases={0: 0, 1: 1},
    )
    def masked_oob_swap_slice(_, _2, mask_ref, start_idx_ref, x_ref, y_ref):
      x, mask = x_ref[:], mask_ref[:]
      y = pl.swap(y_ref, (pl.dslice(start_idx_ref[()], n)), x, mask=mask)
      x_ref[:] = y

    x = random.normal(random.key(0), (n,))
    y = random.normal(random.key(1), (m,))
    slice_start = random.randint(random.key(2), (), m-n+1, m)
    indices = jnp.arange(n) + slice_start
    mask = indices < m
    out = masked_oob_swap_slice(x, y, mask, slice_start)

    # the unjittable masked indexing equivalent
    unmasked_idx = indices[mask]
    x_new = x.at[mask].set(y[unmasked_idx])
    y_new = y.at[unmasked_idx].set(x[mask])
    np.testing.assert_array_equal(out[0], x_new)
    np.testing.assert_array_equal(out[1], y_new)

  @parameterized.named_parameters(
      ("add_i32", pl.atomic_add, np.array([1, 2, 3, 4], np.int32), np.sum),
      ("max_i", pl.atomic_max, np.array([1, 2, 3, 4], np.int32), np.max),
      ("min_i32", pl.atomic_min, np.array([1, 2, 3, 4], np.int32), np.min),
      ("add_f16", pl.atomic_add, np.array([1, 2, 3, 4], np.float16), np.sum),
      ("add_f32", pl.atomic_add, np.array([1, 2, 3, 4], np.float32), np.sum),
      ("max_f32", pl.atomic_max, np.array([1, 2, 3, 4], np.float32), np.max),
      ("min_f32", pl.atomic_min, np.array([1, 2, 3, 4], np.float32), np.min),
  )
  def test_scalar_atomic(self, op, value, numpy_op):
    @functools.partial(
        self.pallas_call,
        out_shape=jax.ShapeDtypeStruct((), value.dtype),
        grid=value.shape[0],
        input_output_aliases={1: 0},
    )
    def atomic_kernel(x_ref, _, o_ref):
      pid = pl.program_id(axis=0)
      op(o_ref, (), x_ref[pid])

    if op == pl.atomic_add:
      neutral = np.array(0, dtype=value.dtype)
    elif op == pl.atomic_max:
      if np.issubdtype(value.dtype, np.integer):
        neutral = np.array(np.iinfo(value.dtype).min, value.dtype)
      else:
        neutral = np.array(-float("inf"), value.dtype)
    elif op == pl.atomic_min:
      if np.issubdtype(value.dtype, np.integer):
        neutral = np.array(np.iinfo(value.dtype).max, value.dtype)
      else:
        neutral = np.array(float("inf"), value.dtype)
    elif op == pl.atomic_or:
      neutral = np.array(False, value.dtype)
    else:
      raise NotImplementedError()
    out = atomic_kernel(value, neutral)
    np.testing.assert_allclose(out, numpy_op(value))

  @parameterized.parameters((0,), (1,))
  def test_array_atomic_add(self, axis):
    m, n = 32, 8
    if axis == 0:
      grid = m
    else:
      grid = n
    out_shape = jax.ShapeDtypeStruct((n if axis == 0 else m,), jnp.float32)

    @functools.partial(
        self.pallas_call,
        out_shape=out_shape,
        grid=grid,
        input_output_aliases={1: 0},
    )
    def reduce(x_ref, _, y_ref):
      i = pl.program_id(axis=0)
      if axis == 0:
        idx = (i, jnp.arange(n))
      else:
        idx = (jnp.arange(m), i)
      x = pl.load(x_ref, idx)
      pl.atomic_add(y_ref, (jnp.arange(y.shape[0]),), x)

    x = random.normal(random.key(0), (m, n))
    y = jnp.zeros(out_shape.shape, out_shape.dtype)
    y = reduce(x, y)
    y_ref = np.sum(x, axis=axis)
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)

  @parameterized.parameters(
      (0, 0, 1),
      (0, 1, 1),
      (1, 0, 1),
      (1, 1, 1),
      (2, 1, 1),
      (2, 1, 1),
  )
  def test_atomic_cas(self, init_value, cmp, new_value):
    @functools.partial(
        self.pallas_call, out_shape=(
          jax.ShapeDtypeStruct((), jnp.int32),
          jax.ShapeDtypeStruct((), jnp.int32)),
        input_output_aliases={0: 0})
    def swap(_, lock_ref, out_ref):
      out_ref[()] = pl.atomic_cas(lock_ref, cmp, new_value)

    lock, out = swap(init_value)
    np.testing.assert_allclose(lock, new_value if cmp == init_value else
                               init_value)
    np.testing.assert_allclose(out, init_value)

  @parameterized.parameters(1, 2, 3, 4, 8)
  def test_atomic_counter(self, num_threads):
    if self.INTERPRET:
      self.skipTest("While loop not supported in interpreter mode.")

    @functools.partial(
        self.pallas_call, out_shape=(
          jax.ShapeDtypeStruct((), jnp.int32),
          jax.ShapeDtypeStruct((), jnp.int32)),
        input_output_aliases={0: 0, 1: 1},
        grid=(num_threads,))
    def increment(_, __, lock_ref, counter_ref):
      def _cond(_):
        return pl.atomic_cas(lock_ref, 0, 1) == 1
      lax.while_loop(_cond, lambda a: a, 0)
      counter_ref[...] += 1
      pl.atomic_xchg(lock_ref, (), 0)

    lock, count = increment(0, 0)
    np.testing.assert_allclose(lock, 0)
    np.testing.assert_allclose(count, num_threads)

  @parameterized.parameters(False, True)
  def test_reduce_only_dim(self, use_store):
    m = 32
    x = random.normal(random.key(0), (m,), dtype=jnp.float32)
    out_shape = jax.ShapeDtypeStruct((), x.dtype)

    @functools.partial(
        self.pallas_call, out_shape=out_shape, grid=1
    )
    def reduce(x_ref, y_ref):
      x = pl.load(x_ref, (jnp.arange(m),))
      y = jnp.sum(x, axis=-1)
      if use_store:
        pl.store(y_ref, (), y)
      else:
        y_ref[...] = y

    y = reduce(x)
    y_ref = jnp.sum(x, axis=-1)
    np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(*[
      (f"{op_name}_{dtype}_{axis}", op, dtype, axis)
      for op_name, op in [
          ("add", jnp.sum),
          ("max", jnp.max),
          ("min", jnp.min),
          ("argmax", jnp.argmax),
          ("argmin", jnp.argmin),
      ]
      for axis in [0, 1, (1,), (0, 1)]
      for dtype in ["float16", "float32", "int32", "uint32"]
      if isinstance(axis, int) or "arg" not in op_name
  ])
  def test_array_reduce(self, op, dtype, axis):
    m, n = 32, 8
    out_dtype = dtype
    if op in {jnp.argmin, jnp.argmax}:
      out_dtype = jnp.int32

    def make_x(key):
      if jnp.issubdtype(dtype, jnp.integer):
        return random.permutation(
            key, jnp.arange(m * n, dtype=dtype), independent=True
        ).reshape(m, n)
      else:
        return random.normal(key, (m, n), dtype=dtype)

    out_shape = jax.ShapeDtypeStruct(
        op(make_x(random.key(0)), axis=axis).shape, out_dtype
    )
    if isinstance(axis, int):
      grid = tuple(a for i, a in enumerate((m, n)) if i != axis)
    else:
      grid = tuple(a for i, a in enumerate((m, n)) if i not in axis)

    @functools.partial(self.pallas_call, out_shape=out_shape, grid=grid)
    def reduce(x_ref, y_ref):
      x = pl.load(x_ref, (jnp.arange(m)[:, None], jnp.arange(n)[None]))
      y = op(x, axis=axis)
      pl.store(y_ref, tuple(jnp.arange(d) for d in y.shape), y)

    for i, key in enumerate(random.split(random.key(0), 20)):
      x = make_x(key)
      y = reduce(x)
      y_ref = op(x, axis=axis)
      np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2, err_msg=i)

  @parameterized.product(
      axis=[0, 1],
      dtype=["float16", "float32", "int32", "uint32"],
  )
  def test_cumsum(self, dtype, axis):
    m, n = 32, 8
    out_dtype = dtype

    def make_x(key):
      if jnp.issubdtype(dtype, jnp.integer):
        return random.permutation(
            key, jnp.arange(m * n, dtype=dtype), independent=True
        ).reshape(m, n)
      else:
        return random.normal(key, (m, n), dtype=dtype)

    out_shape = jax.ShapeDtypeStruct((m, n), out_dtype)
    grid = ()

    @functools.partial(self.pallas_call, out_shape=out_shape, grid=grid)
    def reduce(x_ref, y_ref):
      x = x_ref[...]
      y_ref[...] = jnp.cumsum(x, axis=axis)

    for i, key in enumerate(random.split(random.key(0), 20)):
      x = make_x(key)
      y = reduce(x)
      y_ref = jnp.cumsum(x, axis=axis)
      np.testing.assert_allclose(y, y_ref, atol=1e-2, rtol=1e-2, err_msg=i)


class OpsExtraInterpreterTest(OpsExtraTest):
  INTERPRET = True


class PallasPrimitivesTest(PallasBaseTest):

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)), "<- a[:,:,:]"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)), "<- a[:3,:,:]"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)), "<- a[1:,:,:4]"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)), "<- a[b,:,:4]"),
    (lambda: (jnp.arange(5)[:, None], jnp.arange(3)[None], pl.ds(4)), "<- a[f,g,:4]"),
  ])
  def test_load_pretty_print(self, expr, expected):
    def body(x_ref):
      x = pl.load(x_ref, expr())
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)), "a[:,:,:] <-"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)), "a[:3,:,:] <-"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)), "a[1:,:,:4] <-"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)), "a[b,:,:4] <-"),
    (lambda: (jnp.arange(5)[:, None], jnp.arange(3)[None], pl.dslice(4)), "a[m,n,:4] <-"),
  ])
  def test_store_pretty_print(self, expr, expected):
    def body(x_ref):
      pl.store(x_ref, expr(), pl.load(x_ref, expr()))
      return []
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))

  @parameterized.parameters(*[
    (lambda: (pl.dslice(0, 4), slice(None), slice(None)),
     "c:i32[4,3,2], a[:,:,:] <-"),
    (lambda: (pl.dslice(0, 3), slice(None), slice(None)),
     "c:i32[3,3,2], a[:3,:,:] <-"),
    (lambda: (pl.dslice(1, 3), slice(None), pl.dslice(0, 4)),
     "c:i32[3,3,4], a[1:,:,:4] <-"),
    (lambda: (jnp.arange(5), slice(None), pl.dslice(0, 4)),
     "e:i32[5,3,4], a[b,:,:4] <-"),
    (lambda: (jnp.arange(5)[:, None], jnp.arange(3)[None], pl.dslice(4)),
     "o:i32[5,3,4], a[m,n,:4] <-"),
  ])
  def test_swap_pretty_print(self, expr, expected):
    def body(x_ref):
      x = pl.swap(x_ref, expr(), pl.load(x_ref, expr()))
      return [x]
    jaxpr, _ , _ = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(body), [state.shaped_array_ref((4, 3, 2), jnp.int32)])
    self.assertIn(expected, jaxpr.pretty_print(use_color=False))


class PallasPrimitivesInterpreterTest(PallasPrimitivesTest):
  INTERPRET = True


class TpuOpsTest(PallasBaseTest):

  def setUp(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Test requires TPU device.")

    super().setUp()

  @parameterized.parameters([-3.2, -1.0, -0.4, 0., 0.72, 1.0, 2.4])
  def test_erf_inv(self, x):
    @jax.jit
    @functools.partial(
        pl.pallas_call,
        # TODO(ayx): add float64 support for `erf_inv`
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = lax.erf_inv(x_ref[...])

    x = jnp.full((8, 128), x)
    out = kernel(x)
    expected = lax.erf_inv(x)
    np.testing.assert_array_equal(out, expected)

  SIGN_PARAMS = [
    (jnp.int32, (-3, 0, 5)),
    (jnp.uint32, (0, 5)),
    (jnp.float32, (-3.2, -0., 0., 5.1, jnp.nan, jnp.inf, -jnp.inf)),
  ]

  @parameterized.named_parameters(
      (f"{dtype.__name__}_{value}", dtype, value)
      for dtype, values in SIGN_PARAMS
      for value in values
  )
  def test_sign(self, dtype, value):
    @jax.jit
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((8, 128), dtype),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jnp.sign(x_ref[...])

    x = jnp.full((8, 128,), value, dtype=dtype)
    out = kernel(x)
    expected = jnp.sign(x)
    np.testing.assert_array_equal(out, expected)


if __name__ == "__main__":
  absltest.main()
