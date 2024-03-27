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
from __future__ import annotations

from functools import partial
import itertools
import math
import operator
import platform
import types
import unittest
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax._src import core
from jax import lax
import jax.numpy as jnp
from jax.test_util import check_grads
import jax.util

from jax.interpreters import batching
from jax.interpreters import xla
from jax._src import array
from jax._src import config
from jax._src import dtypes
from jax._src import lax_reference
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.internal_test_util import lax_test_util
from jax._src.lax import lax as lax_internal
from jax._src.lib import xla_extension_version
from jax._src.util import NumpyComplexWarning

config.parse_flags_with_absl()


### lax tests

# We check cases where the preferred type is at least as wide as the input
# type and where both are either both floating-point or both integral,
# which are the only supported configurations.
preferred_type_combinations = [
  (np.float16, np.float16), (np.float16, np.float32), (np.float16, np.float64),
  (dtypes.bfloat16, dtypes.bfloat16), (dtypes.bfloat16, np.float32),
  (dtypes.bfloat16, np.float64), (np.float32, np.float32), (np.float32, np.float64),
  (np.float64, np.float64), (np.int8, np.int8), (np.int8, np.int16), (np.int8, np.int32),
  (np.int8, np.int64), (np.int16, np.int16), (np.int16, np.int32), (np.int16, np.int64),
  (np.int32, np.int32), (np.int32, np.int64), (np.int64, np.int64),
  (np.complex64, np.complex64), (np.complex64, np.complex128), (np.complex128, np.complex128),
  (np.int8, np.float16), (np.int8, dtypes.bfloat16), (np.int8, np.float32), (np.int8, np.float64),
  (np.int16, np.float16), (np.int16, dtypes.bfloat16), (np.int16, np.float32), (np.int16, np.float64),
  (np.int32, np.float32), (np.int32, np.float64), (np.int64, np.float64)]


class LaxTest(jtu.JaxTestCase):
  """Numerical tests for LAX operations."""

  @parameterized.parameters(itertools.chain.from_iterable(
      jtu.sample_product_testcases(
          [dict(op_name=rec.op, rng_factory=rec.rng_factory)],
          shapes=itertools.chain.from_iterable(
              itertools.combinations_with_replacement(shape_group, rec.nargs)
              for shape_group in lax_test_util.compatible_shapes),
          dtype=rec.dtypes)
      for rec in lax_test_util.lax_ops()))
  def testOp(self, op_name, rng_factory, shapes, dtype):
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = getattr(lax, op_name)
    self._CompileAndCheck(op, args_maker)

  @parameterized.parameters(itertools.chain.from_iterable(
      jtu.sample_product_testcases(
        [dict(op_name=rec.op, rng_factory=rec.rng_factory, tol=rec.tol)],
        shapes=itertools.chain.from_iterable(
          itertools.combinations_with_replacement(shape_group, rec.nargs)
          for shape_group in lax_test_util.compatible_shapes),
        dtype=rec.dtypes)
      for rec in lax_test_util.lax_ops()))
  def testOpAgainstNumpy(self, op_name, rng_factory, shapes, dtype, tol):
    if (not config.enable_x64.value and op_name == "nextafter"
        and dtype == np.float64):
      raise SkipTest("64-bit mode disabled")
    rng = rng_factory(self.rng())
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = getattr(lax, op_name)
    numpy_op = getattr(lax_reference, op_name)
    tol = tol or jtu.default_tolerance()
    if jtu.test_device_matches(["tpu"]):
      if dtype in (np.float32, np.complex64) and op_name in (
        "acosh", "asinh", "betainc", "cos", "cosh", "digamma", "exp", "exp2", "igamma",
        "igammac", "log", "log1p", "logistic", "pow", "sin", "sinh", "tan"):
        tol = jtu.join_tolerance(tol, 2e-4)
      elif op_name == "asinh" and dtype == np.float16:
        tol = jtu.join_tolerance(tol, 1e-3)
      elif op_name == "lgamma" and dtype == np.float32:
        tol = jtu.join_tolerance(tol, 1e-3)
    self._CheckAgainstNumpy(numpy_op, op, args_maker, tol=tol)

  # TODO test shift_left, shift_right_arithmetic, shift_right_logical

  @jtu.sample_product(
    [dict(from_dtype=from_dtype, to_dtype=to_dtype)
     for from_dtype, to_dtype in itertools.product(
      [None, np.float32, np.int32, "float32", "int32"], repeat=2)],
    weak_type=[False, True],
  )
  def testConvertElementType(self, from_dtype, to_dtype, weak_type):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax_internal._convert_element_type(x, to_dtype, weak_type)
    self._CompileAndCheck(op, args_maker)

    x = rng((1,), from_dtype)
    out = op(x)
    self.assertEqual(out.dtype, dtypes.canonicalize_dtype(to_dtype or x.dtype))
    self.assertEqual(out.aval.weak_type, weak_type)

  def testConvertElementTypeOOB(self):
    out = lax.convert_element_type(2 ** 32, 'int32')
    self.assertEqual(out, 0)

  @jtu.sample_product(
    [dict(from_dtype=from_dtype, to_dtype=to_dtype)
     for from_dtype, to_dtype in itertools.product(
      [np.float32, np.int32, "float32", "int32"], repeat=2)],
  )
  def testConvertElementTypeAgainstNumpy(self, from_dtype, to_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.convert_element_type(x, to_dtype)
    numpy_op = lambda x: lax_reference.convert_element_type(x, to_dtype)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
    from_dtype=jtu.dtypes.all_floating + jtu.dtypes.all_integer + jtu.dtypes.all_unsigned,
    to_dtype=jtu.dtypes.all_floating + jtu.dtypes.all_integer + jtu.dtypes.all_unsigned,
    shape = [(), (2,), (2, 3)]
  )
  @jtu.skip_on_devices("gpu")  # TODO(b/313567948): Test fails on GPU jaxlib build
  def testBitcastConvertType(self, from_dtype, to_dtype, shape):
    rng = jtu.rand_default(self.rng())
    itemsize_in = np.dtype(from_dtype).itemsize
    itemsize_out = np.dtype(to_dtype).itemsize
    if itemsize_in < itemsize_out:
      shape = (*shape, itemsize_out // itemsize_in)
    args_maker = lambda: [rng(shape, from_dtype)]
    op = lambda x: lax.bitcast_convert_type(x, to_dtype)
    self._CompileAndCheck(op, args_maker)

    # Test the shape and dtype of the output. We avoid testing the values here
    # because the bitwise representation may vary from platform to platform.
    out = op(*args_maker())
    if itemsize_in == itemsize_out:
      expected_shape = shape
    elif itemsize_in < itemsize_out:
      expected_shape = shape[:-1]
    else:
      expected_shape = (*shape, itemsize_in // itemsize_out)
    self.assertEqual(out.dtype, to_dtype)
    self.assertEqual(out.shape, expected_shape)

  @jtu.sample_product(
    [dict(from_dtype=from_dtype, to_dtype=to_dtype)
     for from_dtype, to_dtype in itertools.product(
       [np.float32, np.int32, "float32", "int32"], repeat=2)],
  )
  def testBitcastConvertTypeAgainstNumpy(self, from_dtype, to_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((2, 3), from_dtype)]
    op = lambda x: lax.bitcast_convert_type(x, to_dtype)
    numpy_op = lambda x: lax_reference.bitcast_convert_type(x, to_dtype)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
    [dict(from_dtype=from_dtype, to_dtype=to_dtype)
     for from_dtype, to_dtype in itertools.product(
       [np.float32, np.int32, "float32", "int32"], repeat=2)],
    weak_type=[False, True],
  )
  def testBitcastConvertWeakType(self, from_dtype, to_dtype, weak_type):
    rng = jtu.rand_default(self.rng())
    x_in = lax_internal._convert_element_type(rng((2, 3), from_dtype),
                                              weak_type=weak_type)
    op = lambda x: lax.bitcast_convert_type(x, to_dtype)
    self.assertEqual(dtypes.is_weakly_typed(x_in), weak_type)
    x_out = op(x_in)
    self.assertEqual(dtypes.is_weakly_typed(x_out), False)
    x_out_jit = jax.jit(op)(x_in)
    self.assertEqual(dtypes.is_weakly_typed(x_out_jit), False)

  @jtu.sample_product(
    [dict(min_shape=min_shape, operand_shape=operand_shape, max_shape=max_shape)
     for min_shape, operand_shape, max_shape in [
          [(), (2, 3), ()],
          [(2, 3), (2, 3), ()],
          [(), (2, 3), (2, 3)],
          [(2, 3), (2, 3), (2, 3)],
     ]],
    dtype=lax_test_util.default_dtypes,
  )
  def testClamp(self, min_shape, operand_shape, max_shape, dtype):
    rng = jtu.rand_default(self.rng())
    shapes = [min_shape, operand_shape, max_shape]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    self._CompileAndCheck(lax.clamp, args_maker)

  @jtu.sample_product(
    [dict(min_shape=min_shape, operand_shape=operand_shape, max_shape=max_shape)
     for min_shape, operand_shape, max_shape in [
          [(), (2, 3), ()],
          [(2, 3), (2, 3), ()],
          [(), (2, 3), (2, 3)],
          [(2, 3), (2, 3), (2, 3)],
     ]],
    dtype=lax_test_util.default_dtypes,
  )
  def testClampAgainstNumpy(self, min_shape, operand_shape, max_shape, dtype):
    rng = jtu.rand_default(self.rng())
    shapes = [min_shape, operand_shape, max_shape]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    self._CheckAgainstNumpy(lax_reference.clamp, lax.clamp, args_maker)

  @jtu.sample_product(
    [dict(base_shape=shape, dim=dim) for shape in [(4,), (3, 4), (2, 3, 4)]
     for dim in range(len(shape))],
    num_arrs=[3],
    dtype=lax_test_util.default_dtypes,
  )
  def testConcatenate(self, dim, base_shape, dtype, num_arrs):
    rng = jtu.rand_default(self.rng())
    shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = lambda *args: lax.concatenate(args, dim)
    self._CompileAndCheck(op, args_maker)

  @jtu.sample_product(
    [dict(base_shape=shape, dim=dim) for shape in [(4,), (3, 4), (2, 3, 4)]
     for dim in range(len(shape))],
    num_arrs=[3],
    dtype=lax_test_util.default_dtypes,
  )
  def testConcatenateAgainstNumpy(self, dim, base_shape, dtype, num_arrs):
    rng = jtu.rand_default(self.rng())
    shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    op = lambda *args: lax.concatenate(args, dim)
    numpy_op = lambda *args: lax_reference.concatenate(args, dim)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
      [
          dict(lhs_shape=(b, i, 9, 10), rhs_shape=(j, i, 4, 5))
          for b, i, j in itertools.product([2, 3], repeat=3)
      ],
      dtype=lax_test_util.float_dtypes,
      strides=[(1, 1), (1, 2), (2, 1)],
      padding=["VALID", "SAME", "SAME_LOWER"],
  )
  def testConv(self, lhs_shape, rhs_shape, dtype, strides, padding):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv(lhs, rhs, strides, padding)

    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=(b, i, 9, 10), rhs_shape=(j, i, 4, 5))
     for b, i, j in itertools.product([2, 3], repeat=3)],
    [dict(dtype=dtype, preferred_element_type=preferred)
     for dtype, preferred in preferred_type_combinations]
  )
  @jax.default_matmul_precision("float32")
  def testConvPreferredElement(self, lhs_shape, rhs_shape, dtype, preferred_element_type):
    if (not config.enable_x64.value and
       (dtype == np.float64 or preferred_element_type == np.float64
        or dtype == np.int64 or preferred_element_type == np.int64
        or dtype == np.complex128 or preferred_element_type == np.complex128)):
      raise SkipTest("64-bit mode disabled")
    if (jtu.test_device_matches(["tpu"]) and
       (dtype == np.complex128 or preferred_element_type == np.complex128)):
      raise SkipTest("np.complex128 is not yet supported on TPU")
    if jtu.test_device_matches(["gpu"]) and np.issubdtype(dtype, np.integer):
      # TODO(b/183565702): Support integer convolutions on CPU/GPU.
      raise SkipTest("Integer convolution not yet supported on GPU")
    # x64 implementation is only accurate to ~float32 precision for this case.
    if dtype == np.complex64 and preferred_element_type == np.complex128:
      tol = 1e-5
    else:
      tol = {np.float64: 1e-14}
    if (jtu.test_device_matches(["tpu"]) and dtype == np.float16 and
        preferred_element_type == np.float32):
      tol = 2e-3
    if (jtu.test_device_matches(["tpu"]) and dtype == jnp.bfloat16 and
        preferred_element_type == np.float32):
      tol = 1e-5

    rng = jtu.rand_default(self.rng())
    x = rng(lhs_shape, dtype)
    y = rng(rhs_shape, dtype)
    # We first compute the conv when both inputs are a lower-precision type and
    # preferred_element_type is a higher-precision type. We then compute results
    # where the inputs are first upcast to the higher-precision type and no
    # `preferred_element_type` is given. We expect the result to be extremely
    # similar given the semantics of `preferred_element_type`.
    result_with_preferred_type = lax.conv(
      x, y, (1, 1), "VALID",
      preferred_element_type=preferred_element_type)
    result_with_upcast_inputs = lax.conv(
      x.astype(preferred_element_type),
      y.astype(preferred_element_type),
      (1, 1), "VALID")
    self.assertArraysAllClose(
      result_with_preferred_type, result_with_upcast_inputs, rtol=tol, atol=tol)

  @jtu.sample_product(
    [dict(lhs_shape=(b, i, 9, 10), rhs_shape=(j, i, 4, 5))
     for b, i, j in itertools.product([2, 3], repeat=3)],
    dtype=lax_test_util.float_dtypes,
    strides=[(1, 1), (1, 2), (2, 1)],
    padding=["VALID", "SAME"],
  )
  def testConvAgainstNumpy(self, lhs_shape, rhs_shape, dtype, strides, padding):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    op = lambda lhs, rhs: lax.conv(lhs, rhs, strides, padding)
    numpy_op = lambda lhs, rhs: lax_reference.conv(lhs, rhs, strides, padding)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=(b, i, 9, 10), rhs_shape=(j, i, 4, 5))
     for b, i, j in itertools.product([1, 2, 3], repeat=3)],
    dtype=lax_test_util.float_dtypes,
    strides=[(1, 1), (1, 2), (2, 1)],
    padding=[((0, 0), (0, 0)), ((1, 2), (2, 0))],
    lhs_dilation=[(1, 1), (1, 2), (2, 2)],
    rhs_dilation=[(1, 1), (1, 2), (2, 2)],
  )
  def testConvWithGeneralPadding(self, lhs_shape, rhs_shape, dtype, strides,
                                 padding, lhs_dilation, rhs_dilation):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation)

    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=(b, i, 9, 10), rhs_shape=(j, i, 4, 5))
     for b, i, j in itertools.product([1, 2, 3], repeat=3)],
    dtype=[np.float32],
    strides=[(1, 1), (1, 2), (2, 1)],
    padding=[((0, 0), (0, 0)), ((1, 2), (2, 0))],
    lhs_dilation=[(1, 1), (1, 2), (2, 2)],
    rhs_dilation=[(1, 1), (1, 2), (2, 2)],
  )
  def testConvWithGeneralPaddingAgainstNumpy(
      self, lhs_shape, rhs_shape, dtype, strides, padding, lhs_dilation,
      rhs_dilation):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation,
          precision=lax.Precision.HIGHEST)

    def numpy_fun(lhs, rhs):
      return lax_reference.conv_with_general_padding(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation)

    self._CheckAgainstNumpy(numpy_fun, fun, args_maker)

  @jtu.sample_product(
      [
          dict(
              lhs_shape=(b * batch_group_count, i * feature_group_count),
              rhs_shape=(j * feature_group_count * batch_group_count, i),
              batch_group_count=batch_group_count,
              feature_group_count=feature_group_count,
          )
          for batch_group_count, feature_group_count in [(1, 1), (2, 1), (1, 2)]
          for b, i, j in itertools.product([2, 3], repeat=3)
      ],
      [dict(dimension_numbers=("NC", "OI", "NC"), perms=([0, 1], [0, 1]))],
      dtype=lax_test_util.all_dtypes,
  )
  def testConvGeneralDilated0D(self, lhs_shape, rhs_shape, dtype,
                               feature_group_count, batch_group_count,
                               dimension_numbers, perms):
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
      # TODO(b/183565702): Support integer convolutions on CPU/GPU.
      if jtu.test_device_matches(["gpu"]):
        raise SkipTest("Integer convolution not yet supported on GPU")
    rng = jtu.rand_small(self.rng())
    lhs_perm, rhs_perm = perms  # permute to compatible shapes

    def args_maker():
      return [lax.transpose(rng(lhs_shape, dtype), lhs_perm),
              lax.transpose(rng(rhs_shape, dtype), rhs_perm)]

    def fun(lhs, rhs):
      return lax.conv_general_dilated(
          lhs, rhs, window_strides=(), padding=(),
          dimension_numbers=dimension_numbers,
          feature_group_count=feature_group_count,
          batch_group_count=batch_group_count)

    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
      [
          dict(
              lhs_shape=(b * batch_group_count, i * feature_group_count, 9, w),
              rhs_shape=(j * feature_group_count * batch_group_count, i, 4, 5),
              batch_group_count=batch_group_count,
              feature_group_count=feature_group_count,
          )
          for batch_group_count, feature_group_count in [(1, 1), (2, 1), (1, 2)]
          for w in [0, 10]
          for b, i, j in itertools.product([2, 3], repeat=3)
      ],
      [
          dict(
              dimension_numbers=("NCHW", "OIHW", "NCHW"),
              perms=([0, 1, 2, 3], [0, 1, 2, 3]),
          ),
          dict(
              dimension_numbers=("NHWC", "HWIO", "NHWC"),
              perms=([0, 2, 3, 1], [2, 3, 1, 0]),
          ),
          dict(
              dimension_numbers=("NCHW", "HWIO", "NHWC"),
              perms=([0, 1, 2, 3], [2, 3, 1, 0]),
          ),
      ],
      dtype=lax_test_util.all_dtypes,
      strides=[(1, 1), (2, 1)],
      padding=[((1, 2), (2, 0)), ((10, 8), (7, 13))],
      lhs_dilation=[(1, 1), (1, 2), (1, 4)],
      rhs_dilation=[(1, 1), (1, 2), (1, 4)],
  )
  def testConvGeneralDilated(self, lhs_shape, rhs_shape, dtype, strides,
                             padding, lhs_dilation, rhs_dilation,
                             feature_group_count, batch_group_count,
                             dimension_numbers, perms):
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
      # TODO(b/183565702): Support integer convolutions on CPU/GPU.
      if jtu.test_device_matches(["gpu"]):
        raise SkipTest("Integer convolution not yet supported on GPU")
    rng = jtu.rand_small(self.rng())
    lhs_perm, rhs_perm = perms  # permute to compatible shapes

    def args_maker():
      return [lax.transpose(rng(lhs_shape, dtype), lhs_perm),
              lax.transpose(rng(rhs_shape, dtype), rhs_perm)]

    def fun(lhs, rhs):
      return lax.conv_general_dilated(
          lhs, rhs, strides, padding, lhs_dilation, rhs_dilation,
          dimension_numbers, feature_group_count=feature_group_count,
          batch_group_count=batch_group_count)

    self._CompileAndCheck(fun, args_maker)

  @jax.default_matmul_precision("float32")
  def testConvGeneralDilatedPatchesOverlapping1D(self):
    lhs = np.array([[1]], np.float32).reshape((1, 1))
    patches = lax.conv_general_dilated_patches(
      lhs=lhs,
      filter_shape=(),
      window_strides=(),
      padding='SAME'
    )
    self.assertAllClose(lhs, patches)

    dn = ('NHC', 'OIH', 'NHC')
    lhs = np.array([1, 2, 3, 4, 5], np.float32).reshape((1, -1, 1))

    patches = lax.conv_general_dilated_patches(
        lhs=lhs,
        filter_shape=(2,),
        window_strides=(2,),
        padding='VALID',
        dimension_numbers=dn
    )
    self.assertAllClose(
        np.array([[1, 2],
                  [3, 4]], np.float32).reshape((1, 2, 2)), patches)

    patches = lax.conv_general_dilated_patches(
        lhs=lhs,
        filter_shape=(3,),
        window_strides=(1,),
        padding='SAME',
        dimension_numbers=dn
    )
    self.assertAllClose(
        np.array([[0, 1, 2],
                  [1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5],
                  [4, 5, 0]], np.float32).reshape((1, 5, 3)), patches)

    patches = lax.conv_general_dilated_patches(
        lhs=lhs,
        filter_shape=(3,),
        window_strides=(1,),
        padding='SAME',
        rhs_dilation=(2,),
        dimension_numbers=dn
    )
    self.assertAllClose(
        np.array([[0, 1, 3],
                  [0, 2, 4],
                  [1, 3, 5],
                  [2, 4, 0],
                  [3, 5, 0]], np.float32).reshape((1, 5, 3)), patches)

  def testConvGeneralDilatedPatchesOverlapping2D(self):
    lhs = np.array([[1, 2, 3],
                    [4, 5, 6]], np.float32).reshape((1, 2, 3, 1))
    patches = lax.conv_general_dilated_patches(
        lhs=lhs,
        filter_shape=(2, 2),
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC')
    )
    self.assertAllClose(np.array([[1, 2, 4, 5],
                                  [2, 3, 5, 6],
                                  [3, 0, 6, 0],
                                  [4, 5, 0, 0],
                                  [5, 6, 0, 0],
                                  [6, 0, 0, 0]],
                                 np.float32).reshape((1, 2, 3, 4)), patches)

  @jtu.sample_product(
      [
          dict(
              lhs_shape=lhs_shape,
              filter_shape=filter_shape,
              strides=strides,
              padding=padding,
              dimension_numbers=dim_nums,
          )
          for lhs_shape, filter_shape, strides, padding, dim_nums in [
              ((2, 5), (), (), [], ("NC", "OI", "CN")),
              ((2, 3, 4), (2,), (2,), [(0, 2)], ("CNH", "OHI", "HNC")),
              (
                  (3, 1, 4, 5),
                  (1, 3),
                  (1, 3),
                  [(3, 1), (2, 2)],
                  ("NCHW", "OIHW", "NCHW"),
              ),
              ((3, 2, 5, 6), (4, 3), (4, 3), [(5, 2), (2, 4)], None),
              (
                  (1, 2, 3, 4),
                  (1, 1),
                  (1, 1),
                  [(0, 0), (0, 0)],
                  ("NCWH", "OHWI", "CNHW"),
              ),
              (
                  (1, 2, 3, 4),
                  (3, 2),
                  (1, 1),
                  [(0, 0), (0, 0)],
                  ("CWHN", "HOWI", "NCHW"),
              ),
              (
                  (2, 3, 4, 5, 6),
                  (2, 1, 3),
                  (2, 1, 3),
                  [(1, 2), (5, 3), (3, 5)],
                  ("NHWDC", "HDIWO", "DCWNH"),
              ),
          ]
      ],
      dtype=lax_test_util.all_dtypes,
      precision=[
          None,
          lax.Precision.DEFAULT,
          lax.Precision.HIGH,
          lax.Precision.HIGHEST,
      ],
  )
  def testConvGeneralDilatedPatchesNonOverlapping(self,
                                                  lhs_shape,
                                                  filter_shape,
                                                  dtype,
                                                  strides,
                                                  padding,
                                                  dimension_numbers,
                                                  precision):
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
      # TODO(b/183565702): Support integer convolutions on CPU/GPU.
      if jtu.test_device_matches(["gpu"]):
        raise SkipTest("Integer convolution not yet supported on GPU")
    rng = jtu.rand_small(self.rng())
    lhs = rng(lhs_shape, dtype)

    if dimension_numbers is None:
      lhs_spec, rhs_spec, out_spec = "NCHW", "OIHW", "NCHW"
    else:
      lhs_spec, rhs_spec, out_spec = dimension_numbers

    filter_spec = ''.join(c for c in rhs_spec if c not in ('I', 'O'))
    patches_spec = out_spec.replace('C', 'C' + filter_spec.lower())

    full_padding = []
    for c in lhs_spec:
      if c in ('N', 'C'):
        full_padding += [(0, 0)]
      else:
        full_padding += [padding[filter_spec.index(c)]]

    lhs_padded = np.pad(lhs, full_padding, 'constant')
    out = lax.transpose(lhs_padded, [lhs_spec.index(c) for c in out_spec])

    patches = lax.conv_general_dilated_patches(
        lhs=lhs,
        filter_shape=filter_shape,
        window_strides=strides,
        padding=padding,
        dimension_numbers=dimension_numbers,
        precision=precision
    )

    source = []

    # Test that output spatial shape is factored into `#patches x patch_size`.
    for c in out_spec:
      out_c = out.shape[out_spec.index(c)]
      patch_c = patches.shape[out_spec.index(c)]

      if c == 'N':
        self.assertEqual(out_c, patch_c)
      elif c == 'C':
        self.assertEqual(out_c * math.prod(filter_shape), patch_c)
      else:
        self.assertEqual(out_c, patch_c * filter_shape[filter_spec.index(c)])

        source += [patches_spec.index(c), patches_spec.index(c.lower())]

    # Test that stacking patches together gives the source image, padded.
    c = out_spec.index('C')
    patches = patches.reshape(patches.shape[:c] +
                              (lhs_shape[lhs_spec.index('C')],) +
                              filter_shape +
                              patches.shape[c + 1:]
                              )
    patches = np.moveaxis(patches, source, range(len(source)))
    for i in range(len(filter_shape)):
      patches = patches.reshape(patches.shape[:i] + (-1,) +
                                patches.shape[2 + i:])
    patches = np.moveaxis(
        patches,
        range(len(filter_shape)),
        [out_spec.index(c) for c in out_spec if c not in ('N', 'C')])
    tol = None
    if (jtu.test_device_matches(["tpu"]) and
        precision in (None, lax.Precision.DEFAULT)):
      tol = 1e-3
    self.assertAllClose(out, patches, atol=tol, rtol=tol)

  @jtu.sample_product(
      [
          dict(n=n, lhs_spec=lhs_spec, rhs_spec=rhs_spec, out_spec=out_spec)
          for n in [1, 2]
          for lhs_spec in [
              "".join(s) for s in itertools.permutations("NCHWD"[: n + 2])
          ]
          for rhs_spec in [
              "".join(s) for s in itertools.permutations("OIHWDX"[: n + 2])
          ]
          for out_spec in [
              "".join(s) for s in itertools.permutations("NCHWDX"[: n + 2])
          ]
      ],
      dtype=lax_test_util.inexact_dtypes,
      precision=[
          None,
          lax.Precision.DEFAULT,
          lax.Precision.HIGH,
          lax.Precision.HIGHEST,
          (lax.Precision.DEFAULT, lax.Precision.HIGHEST),
      ],
      padding=["SAME", "VALID"],
  )
  def testConvGeneralDilatedLocal(self, dtype, precision, n, padding, lhs_spec,
                                  rhs_spec, out_spec):
    """Make sure LCN with tiled CNN kernel matches CNN."""
    lhs_spec_default = 'NCHWDX'[:n + 2]
    rhs_spec_default = 'OIHWDX'[:n + 2]

    rng = jtu.rand_small(self.rng())

    lhs_default = rng((2, 4, 7, 6, 5, 8)[:n + 2], dtype)
    rhs_default = rng((5, 4, 2, 3, 1, 2)[:n + 2], dtype)

    window_strides = (1, 2, 3, 4)[:n]
    rhs_dilation = (2, 1, 3, 2)[:n]

    lhs_perm = [lhs_spec_default.index(c) for c in lhs_spec]
    lhs = np.transpose(lhs_default, lhs_perm)

    rhs_perm = [rhs_spec_default.index(c) for c in rhs_spec]
    rhs = np.transpose(rhs_default, rhs_perm)

    kwargs = dict(
        lhs=lhs,
        window_strides=window_strides,
        padding=padding,
        rhs_dilation=rhs_dilation,
        dimension_numbers=(lhs_spec, rhs_spec, out_spec),
        precision=precision
    )

    out_conv = lax.conv_general_dilated(rhs=rhs, **kwargs)

    rhs_local = np.moveaxis(rhs, (rhs_spec.index('O'), rhs_spec.index('I')),
                            (0, 1))
    rhs_local = rhs_local.reshape((rhs_local.shape[0], -1) + (1,) * n)

    rhs_shape = (rhs_local.shape[:2] +
                 tuple(out_conv.shape[out_spec.index(c)]
                       for c in rhs_spec_default[2:]))

    rhs_local = np.broadcast_to(rhs_local, rhs_shape)
    rhs_local = np.transpose(rhs_local, rhs_perm)

    filter_shape = [rhs.shape[i]
                    for i in range(n + 2) if rhs_spec[i] not in ('O', 'I')]
    out_local = lax.conv_general_dilated_local(rhs=rhs_local,
                                               filter_shape=filter_shape,
                                               **kwargs)

    self.assertAllClose(out_conv, out_local)

  # TODO(mattjj): test conv_general_dilated against numpy

  def testConv0DIsDot(self):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      return [rng((10, 5), np.float32), rng((5, 7), np.float32)]
    jnp_fun = partial(lax.conv_general_dilated, window_strides=(),
                      padding='VALID', dimension_numbers=('NC', 'IO', 'NC'))
    self._CompileAndCheck(jnp_fun, args_maker)
    self._CheckAgainstNumpy(np.dot, jnp_fun, args_maker, tol=.1)

  def testGradConv0D(self):
    # Reproduces a failure in neural_tangents not caught in our presubmit tests
    # See cl/367416742.
    lhs = np.ones((2, 5), dtype=np.float32)
    rhs = np.ones((5, 10), dtype=np.float32)

    def f_jax(lhs, rhs):
      return lax.conv_general_dilated(
          lhs, rhs, window_strides=(),
          padding=(), lhs_dilation=(), rhs_dilation=(),
          dimension_numbers=lax.ConvDimensionNumbers((0, 1), (1, 0), (0, 1)),
          batch_group_count=1, feature_group_count=1, precision=None,
          preferred_element_type=None)
    res, pullback = jax.vjp(f_jax, lhs, rhs)
    grad = pullback(np.ones_like(res))
    self.assertAllClose((lhs * 10., rhs * 2.), grad)

  @staticmethod
  def _conv_transpose_via_grad(data, kernel, strides, padding,
                               rhs_dilation=None, dimension_numbers=None):
    """Helper method: calculates conv transpose via grad for testing."""
    assert len(data.shape) == len(kernel.shape)
    nspatial = len(data.shape) - 2
    one = (1,) * nspatial
    rhs_dilation = rhs_dilation or one
    dn = lax.conv_dimension_numbers(data.shape, kernel.shape,
                                    dimension_numbers)
    in_shape = np.take(data.shape, dn.lhs_spec)
    in_sdims = in_shape[2:]
    k_shape = np.take(kernel.shape, dn.rhs_spec)
    k_sdims = k_shape[2:]
    e_k_sdims = [(k-1) * r + 1 for k, r in zip(k_sdims, rhs_dilation)]
    if padding == 'VALID':
      o_sdims = [in_sdims[i]*strides[i] + max(e_k_sdims[i]-strides[i],0)
                 for i in range(nspatial)]
    elif padding == 'SAME':
      o_sdims = [in_sdims[i]*strides[i] for i in range(nspatial)]
    o_shape =  [in_shape[0], k_shape[1]] + o_sdims
    out_spec_inv = [x[0] for x in
                    sorted(enumerate(dn.out_spec), key=lambda x: x[1])]
    o_layout = np.take(np.array(o_shape), out_spec_inv)
    placeholder = np.ones(o_layout, data.dtype)
    conv = lambda x: lax.conv_general_dilated(x, kernel, strides, padding,
                                              one, rhs_dilation, dn)
    _, g = jax.vjp(conv, placeholder)
    return g(data)[0]

  @staticmethod
  def _transpose_conv_kernel(data, kernel, dimension_numbers):
    dn = lax.conv_dimension_numbers(data.shape, kernel.shape,
                                    dimension_numbers)
    spatial_axes = np.array(dn.rhs_spec)[2:]
    for axis in spatial_axes:
      kernel = np.flip(kernel, axis)
    kernel = np.swapaxes(kernel, dn.rhs_spec[0], dn.rhs_spec[1])
    return kernel

  @jtu.sample_product(
      [
          dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
          for lhs_shape, rhs_shape in [
              (
                  (b, 9, 10, i),
                  (k, k, j, i),
              )  # NB: i,j flipped in RHS for transpose
              for b, i, j, k in itertools.product(
                  [2, 3], [2, 3], [2, 3], [3, 4, 5]
              )
          ]
      ],
      dtype=lax_test_util.float_dtypes,
      strides=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)],
      padding=["VALID", "SAME"],
      dspec=[
          ("NHWC", "HWIO", "NHWC"),
      ],
      rhs_dilation=[None, (2, 2)],
  )
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testConvTranspose2DT(self, lhs_shape, rhs_shape, dtype, strides,
                          padding, dspec, rhs_dilation):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    # NB: this test calculates conv_transpose performing identically to the
    # lhs-grad of conv.
    def fun(lhs, rhs):
      return lax.conv_transpose(lhs, rhs, strides, padding,
                                rhs_dilation=rhs_dilation,
                                dimension_numbers=dspec,
                                transpose_kernel=True)

    def fun_via_grad(lhs, rhs):
      return self._conv_transpose_via_grad(lhs, rhs, strides, padding,
                                           rhs_dilation=rhs_dilation,
                                           dimension_numbers=dspec)

    # NB: below just checks for agreement, we're not calling numpy.
    self._CheckAgainstNumpy(fun_via_grad, fun, args_maker)

  @jtu.sample_product(
      [
          dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
          for lhs_shape, rhs_shape in [
              ((b, 9, 10, i), (k, k, i, j))
              for b, i, j, k in itertools.product(
                  [2, 3], [2, 3], [2, 3], [3, 4, 5]
              )
          ]
      ],
      dtype=lax_test_util.float_dtypes,
      strides=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)],
      padding=["VALID", "SAME"],
      dspec=[
          ("NHWC", "HWIO", "NHWC"),
      ],
      rhs_dilation=[None, (2, 2)],
  )
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def testConvTranspose2D(self, lhs_shape, rhs_shape, dtype, strides,
                          padding, dspec, rhs_dilation):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_transpose(lhs, rhs, strides, padding,
                                rhs_dilation=rhs_dilation,
                                dimension_numbers=dspec,
                                transpose_kernel=False)

    def fun_via_grad(lhs, rhs):
      rhs_t = self._transpose_conv_kernel(lhs, rhs, dimension_numbers=dspec)
      return self._conv_transpose_via_grad(lhs, rhs_t, strides, padding,
                                           rhs_dilation=rhs_dilation,
                                           dimension_numbers=dspec)

    # NB: below just checks for agreement, we're not calling numpy.
    self._CheckAgainstNumpy(fun_via_grad, fun, args_maker)

  @jtu.sample_product(
      [
          dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
          for lhs_shape, rhs_shape in [
              ((b, 10, i), (k, i, j))
              for b, i, j, k in itertools.product(
                  [2, 3], [2, 3], [2, 3], [3, 4, 5]
              )
          ]
      ],
      dtype=lax_test_util.float_dtypes,
      strides=[(1,), (2,), (3,)],
      padding=["VALID", "SAME"],
      dspec=[
          ("NHC", "HIO", "NHC"),
      ],
      rhs_dilation=[None, (2,)],
  )
  def testConvTranspose1D(self, lhs_shape, rhs_shape, dtype, strides,
                          padding, dspec, rhs_dilation):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_transpose(lhs, rhs, strides, padding,
                                dimension_numbers=dspec,
                                rhs_dilation=rhs_dilation,
                                transpose_kernel=False)

    def fun_via_grad(lhs, rhs):
      rhs_t = self._transpose_conv_kernel(lhs, rhs, dimension_numbers=dspec)
      return self._conv_transpose_via_grad(lhs, rhs_t, strides, padding,
                                           rhs_dilation=rhs_dilation,
                                           dimension_numbers=dspec)

    # NB: below just checks for agreement, we're not calling numpy.
    self._CheckAgainstNumpy(fun_via_grad, fun, args_maker)

  @jtu.sample_product(
      [
          dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
          for lhs_shape, rhs_shape in [
              ((b, i), (i, j))
              for b, i, j in itertools.product([2, 3], [2, 3], [2, 3])
          ]
      ],
      dtype=lax_test_util.float_dtypes,
      strides=[()],
      padding=["VALID", "SAME"],
      dspec=[
          ("NC", "IO", "NC"),
      ],
      rhs_dilation=[None, ()],
  )
  def testConvTranspose0D(self, lhs_shape, rhs_shape, dtype, strides,
                          padding, dspec, rhs_dilation):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.conv_transpose(lhs, rhs, strides, padding,
                                dimension_numbers=dspec,
                                rhs_dilation=rhs_dilation,
                                transpose_kernel=False)

    def fun_via_grad(lhs, rhs):
      rhs_t = self._transpose_conv_kernel(lhs, rhs, dimension_numbers=dspec)
      return self._conv_transpose_via_grad(lhs, rhs_t, strides, padding,
                                           rhs_dilation=rhs_dilation,
                                           dimension_numbers=dspec)

    # NB: below just checks for agreement, we're not calling numpy.
    self._CheckAgainstNumpy(fun_via_grad, fun, args_maker)

  def testConvTransposePaddingList(self):
    # Regression test for https://github.com/google/jax/discussions/8695
    a = jnp.ones((28,28))
    b = jnp.ones((3,3))
    c = lax.conv_general_dilated(a[None, None], b[None, None], (1,1), [(0,0),(0,0)], (1,1))
    self.assertAllClose(c, 9 * jnp.ones((1, 1, 26, 26)))

  def testConvInvalidPadding(self):
    x = jnp.ones((1, 10, 10, 5), dtype=jnp.bfloat16)
    with self.assertRaisesRegex(ValueError,
                                r"padding argument.*, got \(3, 3\)"):
      jax.lax.conv_general_dilated_patches(x, (5, 5), window_strides=(1, 1),
                                           padding=(3, 3))

  @jtu.sample_product(
      [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
       for lhs_shape in [(3,), (4, 3)] for rhs_shape in [(3,), (3, 6)]],
      [dict(lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype)
       for lhs_dtype, rhs_dtype in
       itertools.chain(
           itertools.product(lax_test_util.int_dtypes +
                             lax_test_util.float_dtypes +
                             lax_test_util.complex_dtypes +
                             lax_test_util.uint_dtypes,
                             repeat=2),
           zip(lax_test_util.bool_dtypes, lax_test_util.bool_dtypes))],
      precision=[
          None,
          lax.Precision.DEFAULT,
          lax.Precision.HIGH,
          lax.Precision.HIGHEST,
          (lax.Precision.DEFAULT, lax.Precision.HIGHEST),
      ],
  )
  def testDot(self, lhs_shape, rhs_shape, lhs_dtype, rhs_dtype, precision):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    self._CompileAndCheck(partial(lax.dot, precision=precision), args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
     for lhs_shape in [(3,), (4, 3)] for rhs_shape in [(3,), (3, 6)]],
    [dict(dtype=d, preferred_element_type=p)
     for d, p in preferred_type_combinations],
  )
  def testDotPreferredElement(self, lhs_shape, rhs_shape, dtype,
                              preferred_element_type):
    if (not config.enable_x64.value and
       (dtype == np.float64 or preferred_element_type == np.float64
        or dtype == np.int64 or preferred_element_type == np.int64)):
      raise SkipTest("64-bit mode disabled")
    if (jtu.test_device_matches(["tpu"]) and
       (dtype == np.complex128 or preferred_element_type == np.complex128)):
      raise SkipTest("np.complex128 is not yet supported on TPU")
    if jtu.test_device_matches(["gpu"]):
      # TODO(b/189287598)
      raise SkipTest("dot_general with preferred_element_type returns NaN "
                     "non-deterministically on GPU")
    rng = jtu.rand_default(self.rng())
    x = rng(lhs_shape, dtype)
    y = rng(rhs_shape, dtype)
    # We first compute the dot when both inputs are a lower-precision type and
    # preferred_element_type is a higher-precision type. We then compute results
    # where the inputs are first upcast to the higher-precision type and no
    # `preferred_element_type` is given. We expect the result to be extremely
    # similar given the semantics of `preferred_element_type`.
    result_with_preferred_type = lax.dot(x, y, preferred_element_type=preferred_element_type)
    result_with_upcast_inputs = lax.dot(
      x.astype(preferred_element_type),
      y.astype(preferred_element_type))
    self.assertArraysAllClose(result_with_preferred_type, result_with_upcast_inputs)

  @jtu.sample_product(
      [
          dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
          for lhs_shape in [(3,), (4, 3)]
          for rhs_shape in [(3,), (3, 6)]
      ],
      dtype=lax_test_util.all_dtypes,
  )
  def testDotAgainstNumpy(self, lhs_shape, rhs_shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    tol = {
      np.float16: 1e-2,
      np.float64: max(jtu.default_tolerance()[np.dtype(np.float64)], 1e-14),
      np.complex128: max(jtu.default_tolerance()[np.dtype(np.complex128)],
                          1e-14),
      jnp.bfloat16: 1e-1
    }
    lax_op = partial(lax.dot, precision=lax.Precision.HIGHEST)
    self._CheckAgainstNumpy(lax_reference.dot, lax_op, args_maker, tol=tol)

  @jtu.sample_product(
      [
          dict(
              lhs_shape=lhs_shape,
              rhs_shape=rhs_shape,
              lhs_contracting=lhs_contracting,
              rhs_contracting=rhs_contracting,
          )
          for lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
              [(5,), (5,), [0], [0]],
              [(5, 7), (5,), [0], [0]],
              [(7, 5), (5,), [1], [0]],
              [(3, 5), (2, 5), [1], [1]],
              [(5, 3), (5, 2), [0], [0]],
              [(5, 3, 2), (5, 2, 4), [0], [0]],
              [(5, 3, 2), (5, 2, 4), [0, 2], [0, 1]],
              [(5, 3, 2), (3, 5, 2, 4), [0, 2], [1, 2]],
              [(1, 2, 2, 3), (1, 2, 3, 1), [1], [1]],
              [(3, 2), (2, 4), [1], [0]],
          ]
      ],
      dtype=lax_test_util.all_dtypes,
  )
  def testDotGeneralContractOnly(self, lhs_shape, rhs_shape, dtype,
                                 lhs_contracting, rhs_contracting):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))

    def fun(lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers)

    self._CompileAndCheck(fun, args_maker, check_dtypes=False)

  @jtu.sample_product(
      [
          dict(
              lhs_shape=lhs_shape,
              rhs_shape=rhs_shape,
              dimension_numbers=dimension_numbers,
          )
          for lhs_shape, rhs_shape, dimension_numbers in [
              ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
              ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1]))),
              ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1]))),
          ]
      ],
      dtype=lax_test_util.all_dtypes,
  )
  def testDotGeneralContractAndBatch(self, lhs_shape, rhs_shape, dtype,
                                     dimension_numbers):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]

    def fun(lhs, rhs):
      return lax.dot_general(lhs, rhs, dimension_numbers)

    self._CompileAndCheck(fun, args_maker, check_dtypes=False)

  @jtu.sample_product(
      [
          dict(
              lhs_shape=lhs_shape,
              rhs_shape=rhs_shape,
              dimension_numbers=dimension_numbers,
          )
          for lhs_shape, rhs_shape, dimension_numbers in [
              ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
              ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1]))),
              ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1]))),
          ]
      ],
      dtype=lax_test_util.all_dtypes,
  )
  def testDotGeneralAgainstNumpy(self, lhs_shape, rhs_shape, dtype,
                                 dimension_numbers):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    op = lambda x, y: lax.dot_general(x, y, dimension_numbers)
    numpy_op = lambda x, y: lax_reference.dot_general(x, y, dimension_numbers)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
      shape=[(), (2, 3)],
      dtype=lax_test_util.default_dtypes,
      broadcast_sizes=[(), (2,), (1, 2)],
  )
  def testBroadcast(self, shape, dtype, broadcast_sizes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.broadcast(x, broadcast_sizes)
    self._CompileAndCheck(op, args_maker)

  @jtu.sample_product(
      shape=[(), (2, 3)],
      dtype=lax_test_util.default_dtypes,
      broadcast_sizes=[(), (2,), (1, 2)],
  )
  def testBroadcastAgainstNumpy(self, shape, dtype, broadcast_sizes):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.broadcast(x, broadcast_sizes)
    numpy_op = lambda x: lax_reference.broadcast(x, broadcast_sizes)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
      [
          dict(inshape=inshape, outshape=outshape, dimensions=dimensions)
          for inshape, outshape, dimensions in [
              ([2], [2, 2], [0]),
              ([2], [2, 2], [1]),
              ([2], [2, 3], [0]),
              ([], [2, 3], []),
              ([1], [2, 3], [1]),
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testBroadcastInDim(self, inshape, dtype, outshape, dimensions):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(inshape, dtype)]
    op = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    self._CompileAndCheck(op, args_maker)

  def testBroadcastInDimOperandShapeTranspose(self):
    # Regression test for https://github.com/google/jax/issues/5276
    def f(x):
      return lax.broadcast_in_dim(x, (2, 3, 4), broadcast_dimensions=(0, 1, 2)).sum()
    def g(x):
      return lax.broadcast_in_dim(x.reshape((3,)), (2, 3, 4), broadcast_dimensions=(1,)).sum()
    x = np.ones((1, 3, 1))
    self.assertArraysEqual(jax.grad(f)(x), jax.grad(g)(x))

  @parameterized.parameters(
    {"inshape": inshape, "outshape": outshape,
      "broadcast_dimensions": broadcast_dimensions, "err_msg": err_msg}
    for inshape, outshape, broadcast_dimensions, err_msg in [
      ([2], [2, 2], [0, 1], ('broadcast_dimensions must have length equal to '
                              'operand ndim')),
      ([2, 2], [2], [0, 1], ('target broadcast shape must have equal or higher rank '
                             'to the operand shape')),
      ([2], [2, 3], [2], ('broadcast_in_dim broadcast_dimensions must be a subset of output '
                          'dimensions')),
      ([2], [3], [0], ('operand dimension sizes must either be 1, or be '
                       'equal to their corresponding dimensions in the target broadcast shape')),
      ([2, 2], [2, 2], [1, 0], ('broadcast_dimensions must be strictly increasing')),
    ])
  def testBroadcastInDimShapeCheck(self, inshape, outshape, broadcast_dimensions, err_msg):
    rng = jtu.rand_default(self.rng())
    x = rng(inshape, np.float32)
    with self.assertRaisesRegex(TypeError, err_msg):
      lax.broadcast_in_dim(x, shape=outshape, broadcast_dimensions=broadcast_dimensions)

  @jtu.sample_product(
      [
          dict(inshape=inshape, outshape=outshape, dimensions=dimensions)
          for inshape, outshape, dimensions in [
              ([2], [2, 2], [0]),
              ([2], [2, 2], [1]),
              ([2], [2, 3], [0]),
              ([], [2, 3], []),
              ([1], [2, 3], [1]),
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testBroadcastInDimAgainstNumpy(self, inshape, dtype, outshape, dimensions):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(inshape, dtype)]
    op = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    numpy_op = lambda x: lax_reference.broadcast_in_dim(x, outshape, dimensions)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @parameterized.parameters(
    {"inshape": inshape, "dimensions": dimensions, "error_type": error_type,
     "err_msg": err_msg}
    for inshape, dimensions, error_type, err_msg in [
      ((1, 2, 3), (0, 0), ValueError, 'dimensions are not unique'),
      ((1, 2, 3), (3,), ValueError, 'axis 3 is out of bounds'),
      ((1, 2, 3), (-4,), ValueError, 'axis -4 is out of bounds'),
      ((1, 2, 3), (1,), ValueError, 'cannot select an axis to squeeze out'),
      ((1, 2, 3), (None,), TypeError, 'cannot be interpreted as an integer'),
    ])
  def testSqueezeShapeCheck(self, inshape, dimensions, error_type, err_msg):
    rng = jtu.rand_default(self.rng())
    x = rng(inshape, np.float32)
    with self.assertRaisesRegex(error_type, err_msg):
      lax.squeeze(x, dimensions=dimensions)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, dimensions=dimensions)
      for arg_shape, dimensions in [
        [(1,), (0,)],
        [(1,), (-1,)],
        [(2, 1, 4), (1,)],
        [(2, 1, 3, 1), (1,)],
        [(2, 1, 3, 1), (1, 3)],
        [(2, 1, 3, 1), (3,)],
      ]],
  )
  def testSqueeze(self, arg_shape, dimensions):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(arg_shape, np.float32)]
    op = lambda x: lax.squeeze(x, dimensions)
    numpy_op = lambda x: lax_reference.squeeze(x, dimensions)
    self._CompileAndCheck(op, args_maker)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)
    check_grads(op, args_maker(), 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    input_type=["np.array", "jnp.array", "float", "np.float32"],
    jit=[True, False],
  )
  def testEmptySqueezeReturnType(self, input_type, jit):
    if input_type == "np.array":
      operand = np.arange(5)
    elif input_type == "jnp.array":
      operand = jnp.arange(5)
    elif input_type == "float":
      operand = 2.0
    elif input_type == "np.float32":
      operand = np.float32(2.0)
    else:
      raise ValueError(f"Unrecognized {input_type=}")

    op = lambda x: lax.squeeze(x, dimensions=())
    if jit:
      op = jax.jit(op)
    result = op(operand)
    self.assertIsInstance(result, jax.Array)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, out_shape=out_shape)
     for arg_shape, out_shape in [
        [(3, 4), (12,)], [(2, 1, 4), (8,)], [(2, 2, 4), (2, 8)]
     ]],
    dtype=lax_test_util.default_dtypes,
  )
  def testReshape(self, arg_shape, out_shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(arg_shape, dtype)]
    op = lambda x: lax.reshape(x, out_shape)
    self._CompileAndCheck(op, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, out_shape=out_shape)
     for arg_shape, out_shape in [
        [(3, 4), (12,)], [(2, 1, 4), (8,)], [(2, 2, 4), (2, 8)]
     ]],
    dtype=lax_test_util.default_dtypes,
  )
  def testReshapeAgainstNumpy(self, arg_shape, out_shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(arg_shape, dtype)]
    op = lambda x: lax.reshape(x, out_shape)
    numpy_op = lambda x: lax_reference.reshape(x, out_shape)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  def testRoundRoundingMethods(self):
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    self.assertAllClose(lax.round(x, lax.RoundingMethod.AWAY_FROM_ZERO),
                        np.array([-3, -2, -1, 1, 2, 3], dtype=np.float32))
    self.assertAllClose(lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN),
                        np.array([-2, -2, 0, 0, 2, 2], dtype=np.float32))

  @jtu.sample_product(
    [dict(shape=shape, pads=pads) for shape, pads in [
        ((0, 2), [(1, 2, 1), (0, 1, 0)]),
        ((2, 3), [(1, 2, 1), (0, 1, 0)]),
        ((2,), [(1, 2, 0)]),
        ((1, 2), [(1, 2, 0), (3, 4, 0)]),
        ((1, 2), [(0, 0, 0), (0, 0, 0)]),
        ((2,), [(1, 2, 3),]),
        ((3, 2), [(1, 2, 1), (3, 4, 2)]),
        ((2,), [(-1, 2, 0),]),
        ((4, 2), [(-1, -2, 0), (1, 2, 0)]),
        ((4, 2), [(-1, 2, 0), (1, 2, 2)]),
        ((5,), [(-1, -2, 2),]),
        ((4, 2), [(-1, -2, 1), (1, 2, 2)])
      ]
    ],
    dtype=lax_test_util.default_dtypes,
  )
  def testPad(self, shape, dtype, pads):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    fun = lambda operand: lax.pad(operand, np.array(0, dtype), pads)
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    shape=[(2, 3)],
    dtype=lax_test_util.default_dtypes,
    pads=[
        [(0, 0, 0), (0, 0, 0)],  # no padding
        [(1, 1, 0), (2, 2, 0)],  # only positive edge padding
        [(1, 2, 1), (0, 1, 0)],  # edge padding and interior padding
        [(0, 0, 0), (-1, -1, 0)],  # negative padding
        [(0, 0, 0), (-2, -2, 4)],  # add big dilation then remove from edges
        [(0, 0, 0), (-2, -3, 1)],  # remove everything in one dimension
    ]
  )
  def testPadAgainstNumpy(self, shape, dtype, pads):
    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.pad(x, np.array(0, dtype), pads)
    numpy_op = lambda x: lax_reference.pad(x, np.array(0, dtype), pads)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  def testPadErrors(self):
    with self.assertRaisesRegex(ValueError, "padding_config"):
      lax.pad(np.zeros(2), 0., [(0, 1, 0), (0, 1, 0)])
    with self.assertRaisesRegex(ValueError, "interior padding in padding_config must be nonnegative"):
      lax.pad(np.zeros(2), 0., [(0, 1, -1)])
    with self.assertRaisesRegex(ValueError, "Dimension size after padding is not at least 0"):
      lax.pad(np.zeros(2), 0., [(-3, 0, 0)])
    with self.assertRaisesRegex(ValueError, "Dimension size after padding is not at least 0"):
      lax.pad(np.zeros(2), 0., [(-4, 0, 1)])

  def testReverse(self):
    rev = jax.jit(lambda operand: lax.rev(operand, dimensions))

    dimensions = []
    self.assertAllClose(np.array([0, 1, 2, 3]), rev(np.array([0, 1, 2, 3])),
                        check_dtypes=False)

    dimensions = [0]
    self.assertAllClose(np.array([3, 2, 1]), rev(np.array([1, 2, 3])),
                        check_dtypes=False)

    dimensions = [0, 1]
    self.assertAllClose(np.array([[6, 5, 4], [3, 2, 1]]),
                        rev(np.array([[1, 2, 3], [4, 5, 6]])),
                        check_dtypes=False)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, pred_shape=pred_shape)
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
    ],
    arg_dtype=lax_test_util.default_dtypes,
  )
  def testSelect(self, pred_shape, arg_shape, arg_dtype):
    rng = jtu.rand_default(self.rng())
    def args_maker():
      return [rng(pred_shape, np.bool_), rng(arg_shape, arg_dtype),
              rng(arg_shape, arg_dtype)]
    return self._CheckAgainstNumpy(lax_reference.select, lax.select, args_maker)
    return self._CompileAndCheck(lax.select, args_maker)

  @jtu.sample_product(
      [
          dict(arg_shape=arg_shape, pred_shape=pred_shape)
          for arg_shape in [(), (3,), (2, 3)]
          for pred_shape in ([(), arg_shape] if arg_shape else [()])
      ],
      [
          dict(pred_dtype=pred_dtype, num_args=num_args)
          for (pred_dtype, num_args) in (
              list(
                  itertools.product(
                      [np.dtype(np.bool_), np.dtype(np.int32)], [1, 2]
                  )
              )
              + [(np.dtype(np.int32), 6)]
          )
      ],
      arg_dtype=lax_test_util.default_dtypes,
  )
  def testSelectN(self, pred_dtype, pred_shape, arg_shape, arg_dtype, num_args):
    if pred_dtype == np.bool_:
      pred_rng = jtu.rand_default(self.rng())
    else:
      pred_rng = jtu.rand_int(self.rng(), low=-1, high=num_args + 1)
    rng = jtu.rand_default(self.rng())
    def args_maker():
      return [pred_rng(pred_shape, pred_dtype)] + (
          [rng(arg_shape, arg_dtype) for _ in range(num_args)])
    return self._CheckAgainstNumpy(lambda c, *xs: np.choose(c, xs, mode='clip'),
                                   lax.select_n, args_maker)
    return self._CompileAndCheck(lax.select_n, args_maker)

  @jtu.sample_product(
      [
          dict(
              shape=shape, starts=indices, limits=limit_indices, strides=strides
          )
          for shape, indices, limit_indices, strides in [
              [(3,), (1,), (2,), None],
              [(7,), (4,), (7,), None],
              [(5,), (1,), (5,), (2,)],
              [(8,), (1,), (6,), (2,)],
              [(5, 3), (1, 1), (3, 2), None],
              [(5, 3), (1, 1), (3, 1), None],
              [(7, 5, 3), (4, 0, 1), (7, 1, 3), None],
              [(5, 3), (1, 1), (2, 1), (1, 1)],
              [(5, 3), (1, 1), (5, 3), (2, 1)],
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testSlice(self, shape, dtype, starts, limits, strides):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.slice(x, starts, limits, strides)
    self._CompileAndCheck(op, args_maker)

  @jtu.sample_product(
      [
          dict(
              shape=shape, starts=indices, limits=limit_indices, strides=strides
          )
          for shape, indices, limit_indices, strides in [
              [(3,), (1,), (2,), None],
              [(7,), (4,), (7,), None],
              [(5,), (1,), (5,), (2,)],
              [(8,), (1,), (6,), (2,)],
              [(5, 3), (1, 1), (3, 2), None],
              [(5, 3), (1, 1), (3, 1), None],
              [(7, 5, 3), (4, 0, 1), (7, 1, 3), None],
              [(5, 3), (1, 1), (2, 1), (1, 1)],
              [(5, 3), (1, 1), (5, 3), (2, 1)],
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testSliceAgainstNumpy(self, shape, dtype, starts, limits, strides):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.slice(x, starts, limits, strides)
    numpy_op = lambda x: lax_reference.slice(x, starts, limits, strides)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
      [
          dict(shape=shape, indices=indices, size_indices=size_indices)
          for shape, indices, size_indices in [
              [(3,), np.array((1,)), (1,)],
              [(5, 3), (1, 1), (3, 1)],
              [(5, 3), np.array((1, 1)), (3, 1)],
              [(7, 5, 3), np.array((4, 1, 0)), (2, 0, 1)],
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testDynamicSlice(self, shape, dtype, indices, size_indices):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), np.array(indices)]
    op = lambda x, starts: lax.dynamic_slice(x, starts, size_indices)
    self._CompileAndCheck(op, args_maker)

  @jtu.sample_product(
      [
          dict(shape=shape, indices=indices, size_indices=size_indices)
          for shape, indices, size_indices in [
              [(3,), (1,), (1,)],
              [(5, 3), (1, 1), (3, 1)],
              [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testDynamicSliceAgainstNumpy(self, shape, dtype, indices, size_indices):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype), np.array(indices)]
    op = lambda x, s: lax.dynamic_slice(x, s, size_indices)
    numpy_op = lambda x, s: lax_reference.dynamic_slice(x, s, size_indices)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  def testDynamicSliceInDim(self):
    # Regression test for mixed type problem in dynamic_slice_in_dim.
    rng = jtu.rand_default(self.rng())
    x = rng((6, 7), np.int32)
    np.testing.assert_equal(lax.dynamic_slice_in_dim(x, 2, 3), x[2:5])

  def testDynamicSliceArraySliceSizes(self):
    rng = jtu.rand_default(self.rng())
    x = rng((6, 7), np.int32)
    np.testing.assert_equal(lax.dynamic_slice(x, [2, 3], jnp.array([2, 2])),
                            x[2:4, 3:5])

  def testDynamicSliceWithNonScalarIndex(self):
    x = jnp.ones((6, 7), np.int32)
    with self.assertRaises(TypeError):
      lax.dynamic_slice_in_dim(x, jnp.array([2, 2]), 3)

  @jtu.sample_product(
      [
          dict(shape=shape, indices=indices, update_shape=update_shape)
          for shape, indices, update_shape in [
              [(3,), (1,), (1,)],
              [(5, 3), (1, 1), (3, 1)],
              [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testDynamicUpdateSlice(self, shape, dtype, indices, update_shape):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype), np.array(indices)]

    self._CompileAndCheck(lax.dynamic_update_slice, args_maker)

  @jtu.sample_product(
      [
          dict(shape=shape, indices=indices, update_shape=update_shape)
          for shape, indices, update_shape in [
              [(3,), (1,), (1,)],
              [(5, 3), (1, 1), (3, 1)],
              [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testDynamicUpdateSliceAgainstNumpy(self, shape, dtype, indices,
                                         update_shape):
    rng = jtu.rand_default(self.rng())

    def args_maker():
      return [rng(shape, dtype), rng(update_shape, dtype), np.array(indices)]

    self._CheckAgainstNumpy(lax_reference.dynamic_update_slice,
                            lax.dynamic_update_slice, args_maker)

  def testDynamicUpdateSliceBatched(self):
    # Regression test for https://github.com/google/jax/issues/9083
    x = jnp.arange(5)
    y = jnp.arange(6, 9)
    ind = jnp.arange(6)
    expected = jnp.vstack([lax.dynamic_update_slice(x, y, (i,)) for i in ind])
    actual = jax.vmap(lax.dynamic_update_slice, (None, None, 0))(x, y, (ind,))
    self.assertAllClose(expected, actual)

  def testDynamicUpdateSliceWithNonScalarIndex(self):
    x = jnp.ones((6, 7), np.int32)
    with self.assertRaises(TypeError):
      lax.dynamic_update_slice_in_dim(x, jnp.ones((2, 7), np.int32),
                                      jnp.array([2, 2]), axis=0)

  @jtu.sample_product(
      [
          dict(shape=shape, perm=perm)
          for shape, perm in [
              [(3, 4), (1, 0)],
              [(3, 4), (0, 1)],
              [(3, 4, 5), (2, 1, 0)],
              [(3, 4, 5), (1, 0, 2)],
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testTranspose(self, shape, dtype, perm):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.transpose(x, perm)
    self._CompileAndCheck(op, args_maker)

  def testTransposeWithArrayPermutation(self):
    x = lax.transpose(np.ones((2, 3)), jnp.array([1, 0]))
    self.assertEqual((3, 2), x.shape)

  @jtu.sample_product(
      [
          dict(shape=shape, perm=perm)
          for shape, perm in [
              [(3, 4), (1, 0)],
              [(3, 4), (0, 1)],
              [(3, 4, 5), (2, 1, 0)],
              [(3, 4, 5), (1, 0, 2)],
          ]
      ],
      dtype=lax_test_util.default_dtypes,
  )
  def testTransposeAgainstNumpy(self, shape, dtype, perm):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.transpose(x, perm)
    numpy_op = lambda x: lax_reference.transpose(x, perm)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
      [
          dict(
              op=rec.op,
              reference_op=rec.reference_op,
              init_val=rec.init_val,
              primitive=rec.primitive,
              dtype=dtype,
          )
          for rec in lax_test_util.lax_reduce_ops()
          for dtype in rec.dtypes
      ],
      [
          dict(shape=shape, dims=dims)
          for shape, dims in [
              [(3, 4, 5), (0,)],
              [(3, 4, 5), (1, 2)],
              [(3, 4, 5), (0, 2)],
              [(3, 4, 5), (0, 1, 2)],
          ]
      ],
  )
  def testReduce(self, op, reference_op, init_val, shape, dtype, dims, primitive):
    if not config.enable_x64.value and dtype in (np.float64, np.int64, np.uint64):
      raise SkipTest("x64 mode is disabled.")
    def reference_fun(operand):
      if hasattr(reference_op, "reduce"):
        initial = np.array(init_val, dtype=dtype)
        result = reference_op.reduce(operand, axis=dims, initial=initial)
      else:
        result = reference_op(operand, axis=dims)

      return result.astype(dtype)

    rng_factory = (jtu.rand_default if dtypes.issubdtype(dtype, np.integer)
                   else jtu.rand_small)
    rng = rng_factory(self.rng())
    init_val = np.asarray(init_val).astype(dtype)
    fun = lambda operand, init_val: lax.reduce(operand, init_val, op, dims)
    args_maker = lambda: [rng(shape, dtype), init_val]
    self._CompileAndCheck(fun, args_maker)

    # we separately test the version that uses a concrete init_val because it
    # can hit different code paths
    fun = lambda operand: lax.reduce(operand, init_val, op, dims)
    args_maker = lambda: [rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker)
    self._CheckAgainstNumpy(reference_fun, fun, args_maker)

    # check that the correct monoid reducer primitive is used inside the jaxpr.
    # This requires the init_val (monoid identity element) to be static
    jaxpr = jax.make_jaxpr(fun)(rng(shape, dtype))
    self.assertEqual(jaxpr.eqns[0].primitive, primitive)

  @jtu.sample_product(
    op=["add", "mul"],
    op_namespace=[lax, operator],
    arr_weak_type=[False, True],
    init_weak_type=[False, True],
  )
  def testReduceWeakType(self, op_namespace, op, arr_weak_type, init_weak_type):
    op = getattr(op_namespace, op)
    arr = lax_internal._convert_element_type(np.arange(10), int,
                                             weak_type=arr_weak_type)
    init = lax_internal._convert_element_type(1, int, weak_type=init_weak_type)
    fun = lambda arr, init: lax.reduce(arr, init, op, (0,))
    out = fun(arr, init)
    self.assertEqual(dtypes.is_weakly_typed(out), arr_weak_type and init_weak_type)
    out_jit = jax.jit(fun)(arr, init)
    self.assertEqual(dtypes.is_weakly_typed(out_jit), arr_weak_type and init_weak_type)

  def testReduceWindowScalar(self):
    rng = jtu.rand_small(self.rng())
    dtype = jnp.float32
    init_val = np.asarray(0, dtype=dtype)
    op = lax.add

    def fun(operand, init_val):
      return lax.reduce_window(
          operand, init_val, op, window_dimensions=(), window_strides=(),
          padding=(), base_dilation=(), window_dilation=())

    def reference_fun(operand, init_val):
      return lax_reference.reduce_window(
          operand, init_val, op, window_dimensions=(), window_strides=(),
          padding=(), base_dilation=())

    args_maker = lambda: [rng((), dtype), init_val]
    self._CompileAndCheck(fun, args_maker)
    self._CheckAgainstNumpy(reference_fun, fun, args_maker)

  @jtu.sample_product(
    [dict(init_val=init_val, op=op, dtype=dtype)
      for init_val, op, dtypes in [
          (0, lax.add, [np.float32]),
          (-np.inf, lax.max, [np.float32]),
          (np.inf, lax.min, [np.float32]),
      ]
      for dtype in dtypes
    ],
    [dict(shape=shape, dims=dims, strides=strides, padding=padding,
          base_dilation=base_dilation, window_dilation=window_dilation)
      for shape, dims, strides, padding, base_dilation, window_dilation in (
        itertools.chain(
          itertools.product(
            [(4, 6)],
            [(2, 1), (1, 2)],
            [(1, 1), (2, 1), (1, 2)],
            ["VALID", "SAME", [(0, 3), (1, 2)]],
            [(1, 1), (2, 3)],
            [(1, 1), (1, 2)]),
          itertools.product(
            [(3, 2, 4, 6)], [(1, 1, 2, 1), (2, 1, 2, 1)],
            [(1, 2, 2, 1), (1, 1, 1, 1)],
            ["VALID", "SAME", [(0, 1), (1, 0), (2, 3), (0, 2)]],
            [(1, 1, 1, 1), (2, 1, 3, 2)],
            [(1, 1, 1, 1), (1, 2, 2, 1)])))
    ],
  )
  def testReduceWindow(self, op, init_val, dtype, shape, dims, strides, padding,
                       base_dilation, window_dilation):
    rng = jtu.rand_small(self.rng())
    init_val = np.asarray(init_val, dtype=dtype)

    def fun(operand, init_val):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding,
                               base_dilation, window_dilation)

    def reference_fun(operand, init_val):
      return lax_reference.reduce_window(operand, init_val, op, dims, strides,
                                         padding, base_dilation)

    args_maker = lambda: [rng(shape, dtype), init_val]
    self._CompileAndCheck(fun, args_maker)
    if all(d == 1 for d in window_dilation):
      self._CheckAgainstNumpy(reference_fun, fun, args_maker)

    # we separately test the version that uses a concrete init_val because it
    # can hit different code paths
    def fun(operand):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding,
                               base_dilation, window_dilation)

    args_maker = lambda: [rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, dims=dims, strides=strides, padding=padding,
          base_dilation=base_dilation, window_dilation=window_dilation)
      for shape, dims, strides, padding, base_dilation, window_dilation in (
        itertools.chain(
          itertools.product(
            [(4, 6)],
            [(2, 1), (1, 2)],
            [(1, 1), (2, 1), (1, 2)],
            ["VALID", "SAME", [(0, 3), (1, 2)]],
            [(1, 1), (2, 3)],
            [(1, 1), (1, 2)]),
          itertools.product(
            [(3, 2, 4, 6)], [(1, 1, 2, 1), (2, 1, 2, 1)],
            [(1, 2, 2, 1), (1, 1, 1, 1)],
            ["VALID", "SAME", [(0, 1), (1, 0), (2, 3), (0, 2)]],
            [(1, 1, 1, 1), (2, 1, 3, 2)],
            [(1, 1, 1, 1), (1, 2, 2, 1)])))
    ],
    dtype=[np.float32],
  )
  # TODO(b/183233858): variadic reduce-window is not implemented on XLA:GPU
  @jtu.skip_on_devices("gpu")
  def testReduceWindowVariadic(self, dtype, shape, dims, strides, padding,
                               base_dilation, window_dilation):
    if (jtu.test_device_matches(["tpu"]) and
        any(d != 1 for d in window_dilation)):
      raise SkipTest("TPU support missing for arbitrary window dilation.")
    rng = jtu.rand_small(self.rng())
    init_values = (np.asarray(0, dtype=dtype), np.array(-np.inf, dtype=dtype))

    def reducer(xs, ys):
      x1, x2 = xs
      y1, y2 = ys
      return (x1 + y1, lax.max(x2, y2))

    def fun(*operands):
      return lax.reduce_window(operands, init_values, reducer, dims, strides,
                               padding, base_dilation, window_dilation)

    def reference_fun(*operands):
      return [
          lax_reference.reduce_window(operand, init_val, op, dims, strides,
                                      padding, base_dilation)
          for operand, init_val, op in zip(operands, init_values,
                                           [np.add, np.maximum])]

    args_maker = lambda: [rng(shape, dtype), rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker)
    if all(d == 1 for d in window_dilation):
      self._CheckAgainstNumpy(reference_fun, fun, args_maker)

  def testReduceWindowFailures(self):
    def empty_window_test():
      return lax.reduce_window(np.ones((1,)), 0., lax.add, padding='VALID',
                               window_dimensions=(0,), window_strides=(1,))

    def zero_stride_test():
      return lax.reduce_window(np.ones((1,)), 0., lax.add, padding='VALID',
                               window_dimensions=(1,), window_strides=(0,))

    for failure_fun in [empty_window_test, zero_stride_test]:
      with self.assertRaisesRegex(TypeError, "must have every element be"):
        failure_fun()

    with self.assertRaisesRegex(
        ValueError,
        "reduce_window output must have the same tree structure as the "
        "operands.*"):
      return lax.reduce_window(
          np.ones((1,)), 0., lambda x, y: [x + y],
          padding='VALID', window_dimensions=(1,), window_strides=(1,))

  @jtu.sample_product(
    [dict(shape=shape, window_dimensions=window_dimensions,
          base_dilation=base_dilation, window_dilation=window_dilation)
      for shape, window_dimensions, base_dilation, window_dilation in (
        itertools.chain(
          itertools.product(
            [(4, 6)],
            [(1, 1), (3, 4)],
            [(1, 1), (1, 2), (2, 13), (40, 60)],
            [(1, 1), (1, 2), (2, 13), (40, 60)]),
          itertools.product(
            [(3, 2, 4, 6)],
            [(1, 1, 1, 1), (2, 1, 2, 1)],
            [(1, 1, 1, 1), (1, 2, 2, 1), (30, 40, 3, 2)],
            [(1, 1, 1, 1), (1, 2, 2, 1), (30, 40, 3, 2)])))
     ],
  )
  def testReduceWindowShapeDilation(self, shape, window_dimensions,
                                    base_dilation, window_dilation):
    operand, padding, strides = np.ones(shape), 'SAME', (1,) * len(shape)
    result = lax.reduce_window(operand, 0., lax.add, padding=padding,
                               window_strides=strides,
                               window_dimensions=window_dimensions)
    # With a stride of 1 in each direction and a padding of 'SAME', the
    # shape of the input should be equal to the shape of the result according
    # to https://www.tensorflow.org/xla/operation_semantics#reducewindow.
    self.assertEqual(shape, result.shape)

  def testReduceWindowWithEmptyOutput(self):
    # https://github.com/google/jax/issues/10315
    shape = (5, 3, 2)
    operand, padding, strides = np.ones(shape), 'VALID', (1,) * len(shape)
    out = jax.eval_shape(lambda x: lax.reduce_window(x, 0., lax.add, padding=padding,
                         window_strides=strides,
                         window_dimensions=(3, 1, 1),
                         window_dilation=(3, 1, 1)), operand)
    self.assertEqual((0, 3, 2), out.shape)

  @jtu.sample_product(
    [dict(op=op, np_op=np_op) for op, np_op in [
      (lax.cumsum, np.cumsum),
      (lax.cumprod, np.cumprod),
      (lax.cummax, np.maximum.accumulate),
      (lax.cummin, np.minimum.accumulate),
    ]],
    [dict(shape=shape, axis=axis)
     for shape in [[10], [3, 4, 5]] for axis in range(len(shape))],
    dtype=lax_test_util.default_dtypes,
    reverse=[False, True],
  )
  def testCumulativeReduce(self, op, np_op, shape, dtype, axis, reverse):
    rng_factory = (jtu.rand_default if dtypes.issubdtype(dtype, np.integer)
                   else jtu.rand_small)
    rng = rng_factory(self.rng())
    fun = partial(op, axis=axis, reverse=reverse)
    def np_fun(x):
      if reverse:
        return np.flip(np_op(np.flip(x, axis), axis=axis, dtype=dtype), axis)
      else:
        return np_op(x, axis=axis, dtype=dtype)
    args_maker = lambda: [rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker)
    self._CheckAgainstNumpy(np_fun, fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
     for shape in [[10], [3, 4, 5]] for axis in range(len(shape))],
    dtype=lax_test_util.float_dtypes,
    reverse=[False, True],
  )
  def testCumulativeLogSumExp(self, shape, dtype, axis, reverse):
    # This op only works on floating-point types, so we've separated out the
    # test.
    rng = jtu.rand_small(self.rng())
    fun = partial(lax.cumlogsumexp, axis=axis, reverse=reverse)
    def np_fun(x):
      if reverse:
        return np.flip(np.logaddexp.accumulate(
            np.flip(x, axis), axis=axis, dtype=dtype), axis)
      else:
        return np.logaddexp.accumulate(x, axis=axis, dtype=dtype)
    args_maker = lambda: [rng(shape, dtype)]
    self._CompileAndCheck(fun, args_maker)
    tol = None
    if jtu.test_device_matches(["tpu"]) and dtype == np.float32:
      tol = 1e-4
    self._CheckAgainstNumpy(np_fun, fun, args_maker, atol=tol, rtol=tol)

  @jtu.sample_product(
    shape=[(), (3,), (3, 4)],
    dtype=lax_test_util.float_dtypes,
    out_dtype=lax_test_util.float_dtypes,
  )
  def testReducePrecision(self, shape, dtype, out_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    info = dtypes.finfo(out_dtype)
    fun = lambda x: lax.reduce_precision(x, info.nexp, info.nmant)
    np_fun = lambda x: np.asarray(x).astype(out_dtype).astype(dtype)
    self._CheckAgainstNumpy(np_fun, fun, args_maker)
    self._CompileAndCheck(fun, args_maker)

  def testReducePrecisionGrad(self):
    info = dtypes.finfo(jnp.dtype('bfloat16'))
    y, f_vjp = jax.vjp(lambda x: lax.reduce_precision(x, info.nexp, info.nmant), jnp.pi)
    y2 = f_vjp(jnp.pi)
    y3 = lax.reduce_precision(jnp.pi, info.nexp, info.nmant)
    self.assertArraysEqual(y, y2)
    self.assertArraysEqual(y, y3)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
     for shape in [(5,), (5, 7)] for axis in [-1, len(shape) - 1]],
    dtype=lax_test_util.all_dtypes,
    is_stable=[False, True],
  )
  def testSort(self, shape, dtype, axis, is_stable):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    fun = lambda x: lax.sort(x, dimension=axis, is_stable=is_stable)
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(dtype=lax_test_util.float_dtypes)
  def testSortFloatSpecialValues(self, dtype):
    # Test confirms that
    # - NaNs are sorted to the end, regardless of representation
    # - sign bit of 0.0 is ignored
    x = jnp.array([-np.inf, 0.0, -0.0, np.inf, np.nan, -np.nan], dtype=dtype)
    index = lax.iota(dtypes.int_, x.size)
    argsort = lambda x: lax.sort_key_val(x, lax.iota(dtypes.int_, x.size), is_stable=True)[1]
    self.assertArraysEqual(argsort(x), index)
    self.assertArraysEqual(jax.jit(argsort)(x), index)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
     for shape in [(5,), (5, 7)] for axis in [-1, len(shape) - 1]],
    dtype=lax_test_util.all_dtypes,
    is_stable=[False, True],
  )
  def testSortAgainstNumpy(self, shape, dtype, axis, is_stable):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    op = lambda x: lax.sort(x, dimension=axis, is_stable=is_stable)
    def numpy_op(x):
      if is_stable:
        return lax_reference.sort(x, axis, kind='stable')
      else:
        return lax_reference.sort(x, axis)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
     for shape in [(3,), (5, 3)] for axis in [-1, len(shape) - 1]],
    key_dtype=lax_test_util.float_dtypes + lax_test_util.complex_dtypes +
              lax_test_util.int_dtypes + lax_test_util.uint_dtypes,
    val_dtype=[np.float32, np.int32, np.uint32],
    is_stable=[False, True],
  )
  def testSortKeyVal(self, shape, key_dtype, val_dtype, axis, is_stable):
    if (np.issubdtype(key_dtype, np.complexfloating) and
        jtu.test_device_matches(["cpu"])):
      raise SkipTest("Complex-valued sort not implemented")
    rng = jtu.rand_default(self.rng())
    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    def args_maker():
      flat_keys = np.arange(math.prod(shape), dtype=key_dtype)
      keys = self.rng().permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values

    fun = lambda keys, values: lax.sort_key_val(keys, values, axis, is_stable)
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, num_keys=num_keys)
     for shape in [(3, 5), (4, 3)] for num_keys in range(1, shape[0] + 1)],
    dtype=lax_test_util.all_dtypes,
  )
  def testSortNumKeys(self, shape, dtype, num_keys):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype)]
    lax_fun = lambda x: lax.sort(tuple(x), num_keys=num_keys)
    numpy_fun = lambda x: tuple(x[:, np.lexsort(x[:num_keys][::-1])])
    # self._CompileAndCheck(lax_fun, args_maker)
    self._CheckAgainstNumpy(numpy_fun, lax_fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
     for shape in [(3,), (5, 3)] for axis in [-1, len(shape) - 1]],
    key_dtype=lax_test_util.float_dtypes + lax_test_util.complex_dtypes +
              lax_test_util.int_dtypes + lax_test_util.uint_dtypes,
    val_dtype=[np.float32, np.int32, np.uint32],
  )
  def testSortKeyValAgainstNumpy(self, shape, key_dtype, val_dtype, axis):
    if (np.issubdtype(key_dtype, np.complexfloating) and
        jtu.test_device_matches(["cpu"])):
      raise SkipTest("Complex-valued sort not implemented")
    rng = jtu.rand_default(self.rng())
    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    def args_maker():
      flat_keys = np.arange(math.prod(shape), dtype=key_dtype)
      keys = self.rng().permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values

    op = lambda ks, vs: lax.sort_key_val(ks, vs, axis)
    numpy_op = lambda ks, vs: lax_reference.sort_key_val(ks, vs, axis)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  @jtu.sample_product(
    dtype=[np.float32, np.int32, np.uint32],
    shape=[(20,), (5, 20), (2000,)],
    k=[1, 3, 12],
    negative=[False, True]
  )
  def testTopK(self, shape, dtype, k, negative):
    def args_maker():
      flat_values = np.arange(math.prod(shape), dtype=dtype)
      values = self.rng().permutation(flat_values).reshape(shape)
      if negative:
        values = -values
      return [values]
    def reference_top_k(x):
      bcast_idxs = np.broadcast_to(np.arange(shape[-1], dtype=np.int32), shape)
      sorted_vals, sorted_idxs = lax_reference.sort_key_val(x, bcast_idxs)
      return sorted_vals[..., :-k-1:-1], sorted_idxs[..., :-k-1:-1]
    op = lambda vs: lax.top_k(vs, k=k)
    self._CheckAgainstNumpy(op, reference_top_k, args_maker)
    self._CompileAndCheck(op, args_maker)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
      for lhs_shape, rhs_shape in [((3, 2), (2, 4)),
                                   ((5, 3, 2), (5, 2, 4)),
                                   ((1, 2, 2, 3), (1, 2, 3, 1))]],
    dtype=lax_test_util.float_dtypes,
  )
  def testBatchMatMul(self, lhs_shape, rhs_shape, dtype):
    rng = jtu.rand_small(self.rng())
    arg_maker = lambda: [rng(lhs_shape, dtype), rng(rhs_shape, dtype)]
    self._CompileAndCheck(lax.batch_matmul, arg_maker)

  def testCollapse(self):

    @jax.jit
    def collapse_first_two(x):
      return lax.collapse(x, 0, 2)

    self.assertEqual((6,), collapse_first_two(np.zeros((2, 3))).shape)
    self.assertEqual((6, 4), collapse_first_two(np.zeros((2, 3, 4))).shape)
    self.assertEqual((2, 3, 4),
                     collapse_first_two(np.zeros((1, 2, 3, 4))).shape)

  def testCollapseLastTwo(self):

    @jax.jit
    def collapse_last_two_none_end(x):
      return lax.collapse(x, -2)

    @jax.jit
    def collapse_last_two_pos_end(x):
      return lax.collapse(x, -2)

    self.assertEqual((4, 3, 10),
                     collapse_last_two_none_end(np.zeros((4, 3, 2, 5))).shape)
    self.assertEqual((4, 3, 10),
                     collapse_last_two_pos_end(np.zeros((4, 3, 2, 5))).shape)

  @jtu.sample_product(
    [dict(shape=shape, idxs=idxs, axes=axes)
      for shape, idxs, axes in [
          [(3, 4, 5), (np.array([0, 2, 1]),), (0,)],
          [(3, 4, 5), (np.array([-1, -2]),), (0,)],
          [(3, 4, 5), (np.array([0, 2]), np.array([1, 3])), (0, 1)],
          [(3, 4, 5), (np.array([0, 2]), np.array([1, 3])), [0, 2]],
    ]],
    dtype=lax_test_util.all_dtypes,
  )
  def testIndexTake(self, shape, dtype, idxs, axes):
    rng = jtu.rand_default(self.rng())
    rand_idxs = lambda: tuple(rng(e.shape, e.dtype) for e in idxs)
    args_maker = lambda: [rng(shape, dtype), rand_idxs()]
    fun = lambda src, idxs: lax.index_take(src, idxs, axes)
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(shape=shape, idxs=idxs, dnums=dnums, slice_sizes=slice_sizes)
      for shape, idxs, dnums, slice_sizes in [
          ((5,), np.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,)),
          ((10,), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,)),
          ((10, 5,), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3)),
          ((10, 5), np.array([[0, 2], [1, 0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
            (1, 3)),
    ]],
    dtype=lax_test_util.all_dtypes,
  )
  def testGather(self, shape, dtype, idxs, dnums, slice_sizes):
    rng = jtu.rand_default(self.rng())
    rng_idx = jtu.rand_int(self.rng(), high=max(shape))
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(shape, dtype), rand_idxs()]
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    self._CompileAndCheck(fun, args_maker)

  # These tests are adapted from the corresponding tests in
  # tensorflow/compiler/xla/service/shape_inference_test.cc with slight
  # variations to account for the implicit setting of index_vector_dim in JAX.
  @parameterized.named_parameters(
      {"testcase_name": f"_{testcase_name}", "operand_shape": operand_shape,
       "indices_shape": indices_shape,
       "dimension_numbers": lax.GatherDimensionNumbers(
          offset_dims=offset_dims,
          collapsed_slice_dims=collapsed_slice_dims,
          start_index_map=start_index_map),
       "slice_sizes": slice_sizes, "msg": msg}
      for (testcase_name, operand_shape, indices_shape, offset_dims,
           collapsed_slice_dims, start_index_map, slice_sizes, msg) in [
        ("NonAscendingWindowIndices", (10, 9, 8, 7, 6), (5, 4, 3, 2, 1),
         (4, 5, 6, 8, 7), (), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         "offset_dims in gather op must be sorted"),
        ("RepeatedWindowIndices", (10, 9, 8, 7, 6), (5, 4, 3, 2, 1),
         (4, 5, 6, 7, 7), (), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         "offset_dims in gather op must not repeat"),
        ("WindowIndexOutOfBounds", (10, 9, 8, 7, 6), (5, 4, 3, 2, 1),
         (4, 5, 100, 101, 102), (), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         "Offset dimension 2 in gather op is out of bounds"),
        ("WindowIndexBarelyOutOfBounds", (10, 9, 8, 7, 6), (5, 4, 3, 2, 1),
         (4, 5, 6, 7, 9), (), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         "Offset dimension 4 in gather op is out of bounds"),
        ("MismatchingElidedWindowDims", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7, 8), (4,), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         ("All components of the offset index in a gather op must either be a "
          "offset dimension or explicitly collapsed")),
        ("OutOfBoundsWindowToInputMapping", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7, 8), (0, 1, 2, 3, 19), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         "Invalid collapsed_slice_dims set in gather op; valid range is"),
        ("RepeatedWindowToInputMapping", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7, 8), (0, 1, 2, 3, 3), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         "collapsed_slice_dims in gather op must not repeat"),
        ("MismatchingGatherToInputMapping", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7, 8), (), (0, 1, 2, 3), (10, 9, 8, 7, 6),
         ("Gather op has 4 elements in start_index_map and the bound of "
          "dimension index_vector_dim=4 of indices is 5. These two "
          "numbers must be equal.")),
        ("OutOfBoundsGatherToInputMapping", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7, 8), (), (0, 1, 2, 3, 7), (10, 9, 8, 7, 6),
         "Invalid start_index_map"),
        ("RepeatedGatherToInputMapping", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7, 8), (), (0, 1, 2, 3, 3), (10, 9, 8, 7, 6),
         "start_index_map in gather op must not repeat"),
        ("NonAscendingElidedWindowDims", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7, 8), (2, 1), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         "collapsed_slice_dims in gather op must be sorted"),
        ("WindowBoundsTooLarge", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7), (2,), (0, 1, 2, 3, 4), (10, 9, 8, 100, 6),
         "Slice size at index 3 in gather op is out of range"),
        ("MismatchingNumberOfWindowBounds", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7), (), (0, 1, 2, 3, 4), (10, 9, 8, 7),
         "Gather op must have one slice size for every input dimension"),
        ("WindowBoundsNot1ForElidedDim", (10, 9, 8, 7, 6), (5, 4, 3, 2, 5),
         (4, 5, 6, 7), (1,), (0, 1, 2, 3, 4), (10, 9, 8, 7, 6),
         ("Gather op can only collapse slice dims with bound 1, but bound "
          "is 9 for index 1 at position 0."))
      ]
  )
  def testGatherShapeCheckingRule(self, operand_shape, indices_shape,
                                  dimension_numbers, slice_sizes, msg):
    operand = np.ones(operand_shape, dtype=np.int32)
    indices = np.ones(indices_shape, dtype=np.int32)

    with self.assertRaisesRegex(TypeError, msg):
      lax.gather(operand, indices, dimension_numbers, slice_sizes)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
          dnums=dnums)
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
    ]],
    dtype=lax_test_util.inexact_dtypes,
    mode=["clip", "fill", None],
  )
  def testScatterAdd(self, arg_shape, dtype, idxs, update_shape, dnums, mode):
    rng = jtu.rand_default(self.rng())
    rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs(),
                          rng(update_shape, dtype)]
    fun = partial(lax.scatter_add, dimension_numbers=dnums, mode=mode)
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
          dnums=dnums)
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), np.array([[0], [2], [1]], dtype=np.uint64), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
    ]],
    dtype=lax_test_util.float_dtypes,
  )
  def testScatterMin(self, arg_shape, dtype, idxs, update_shape, dnums):
    rng = jtu.rand_default(self.rng())
    rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs(),
                          rng(update_shape, dtype)]
    fun = partial(lax.scatter_min, dimension_numbers=dnums)
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
          dnums=dnums)
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
    ]],
    dtype=lax_test_util.float_dtypes,
  )
  def testScatterMax(self, arg_shape, dtype, idxs, update_shape, dnums):
    rng = jtu.rand_default(self.rng())
    rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs(),
                          rng(update_shape, dtype)]
    fun = partial(lax.scatter_max, dimension_numbers=dnums)
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
    ]],
    dtype=lax_test_util.float_dtypes,
  )
  def testScatterApply(self, arg_shape, dtype, idxs, update_shape, dnums):
    rng = jtu.rand_default(self.rng())
    rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs()]
    fun = partial(lax.scatter_apply, func=jnp.sin, update_shape=update_shape, dimension_numbers=dnums)
    self._CompileAndCheck(fun, args_maker)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
          dnums=dnums)
      for arg_shape, idxs, update_shape, dnums in [
          ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
          ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,))),
          ((10, 5,), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,))),
    ]],
    dtype=lax_test_util.float_dtypes,
  )
  def testScatter(self, arg_shape, dtype, idxs, update_shape, dnums):
    rng = jtu.rand_default(self.rng())
    rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
    rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
    args_maker = lambda: [rng(arg_shape, dtype), rand_idxs(),
                          rng(update_shape, dtype)]
    fun = partial(lax.scatter, dimension_numbers=dnums)
    self._CompileAndCheck(fun, args_maker)

  # These tests are adapted from the corresponding tests in
  # tensorflow/compiler/xla/service/shape_inference_test.cc with slight
  # variations to account for the implicit setting of index_vector_dim in JAX.
  @parameterized.named_parameters(
      {"testcase_name": f"_{testcase_name}", "operand_shape": operand_shape,
       "indices": indices, "update_shape": update_shape,
       "dimension_numbers": lax.ScatterDimensionNumbers(
          update_window_dims=update_window_dims,
          inserted_window_dims=inserted_window_dims,
          scatter_dims_to_operand_dims=scatter_dims_to_operand_dims),
       "msg": msg}
      for (testcase_name, operand_shape, indices, update_shape,
           update_window_dims, inserted_window_dims,
           scatter_dims_to_operand_dims, msg) in [
              ("ScatterWithUpdatesBiggerThanInput", (64, 48), np.zeros((32, 1)),
               (65, 32), (0,), (1,), (1,), "Bounds of the window dimensions"),
              ("ScatterWithUpdatesBiggerThanInputV2", (64, 48),
               np.zeros((32, 1)), (32, 49), (1,), (0,), (1,),
               "Bounds of the window dimensions"),
              ("ScatterWithUpdatesNotMatchingIndices", (64, 48),
               np.zeros((32, 1)), (64, 31), (0,), (1,), (1,),
               "Bounds of the scatter dimensions"),
              ("ScatterWithUpdatesNotMatchingIndicesV2", (64, 48),
               np.zeros((32, 1)), (31, 48), (1,), (0,), (1,),
               "Bounds of the scatter dimensions"),
              ("ScatterNdWithUpdatesBiggerThanInput", (64, 48),
               np.zeros((10, 9, 8, 7, 1)), (10, 9, 8, 7, 65), (4,), (1,),
               (0,), "Bounds of the window dimensions"),
              ("ScatterNdWithUpdatesNotMatchingIndices", (64, 48),
               np.zeros((10, 9, 8, 7, 1)), (9, 9, 8, 7, 64), (4,), (1,), (0,),
               "Bounds of the scatter dimensions"),
              ("InvalidUpdates", (50, 49, 48, 47, 46),
               np.zeros((10, 9, 8, 7, 5)), (10, 9, 8, 7, 3, 2, 4, 1),
               (4, 5, 6), (1, 2), (0, 1, 2, 3, 4),
               "Updates tensor must be of rank 7; got 8."),
              ("NonAscendingUpdateWindowDims", (6, 5, 4, 3, 2),
               np.zeros((5, 4, 3, 2, 1)), (10, 9, 8, 7, 6, 5, 4, 3, 2),
               (4, 5, 6, 8, 7), (), (0, 1, 2, 3, 4),
               "update_window_dims in scatter op must be sorted"),
              ("RepeatedUpdateWindowDims", (6, 5, 4, 3, 2),
               np.zeros((5, 4, 3, 2, 1)), (10, 9, 8, 7, 6, 5, 4, 3, 2),
               (4, 5, 6, 7, 7), (), (0, 1, 2, 3, 4),
               "update_window_dims in scatter op must not repeat"),
              ("OutOfBoundsUpdateWindowDims", (6, 5, 4, 3, 2),
               np.zeros((5, 4, 3, 2, 1)), (10, 9, 8, 7, 6, 5, 4, 3, 2),
               (4, 5, 6, 7, 9), (), (0, 1, 2, 3, 4),
               "Invalid update_window_dims set in scatter op"),
              ("NonAscendingInsertedWindowDims", (50, 49, 48, 47, 46),
               np.zeros((10, 9, 8, 7, 5)), (10, 9, 8, 7, 3, 2, 4),
               (4, 5, 6), (2, 1), (0, 1, 2, 3, 4),
               "inserted_window_dims in scatter op must be sorted"),
              ("RepeatedInsertedWindowDims", (50, 49, 48, 47, 46),
               np.zeros((10, 9, 8, 7, 5)), (10, 9, 8, 7, 3, 2, 4),
               (4, 5, 6), (1, 1), (0, 1, 2, 3, 4),
               "inserted_window_dims in scatter op must not repeat"),
              ("OutOfBoundsInsertedWindowDims", (50, 49, 48, 47, 46),
               np.zeros((10, 9, 8, 7, 5)), (10, 9, 8, 7, 3, 2, 4),
               (4, 5, 6), (1, 5), (0, 1, 2, 3, 4),
               "Invalid inserted_window_dims set in scatter op"),
              ("MismatchingScatterDimsToOperandDims", (50, 49, 48, 47, 46),
               np.zeros((10, 9, 8, 7, 5)), (10, 9, 8, 7, 3, 2, 4),
               (4, 5, 6), (1, 2), (0, 1, 2, 3),
               ("Scatter op has 4 elements in scatter_dims_to_operand_dims and "
                "the bound of dimension index_vector_dim=4 of indices "
                "is 5. These two numbers must be equal")),
              ("OutOfBoundsScatterDimsToOperandDims", (50, 49, 48, 47, 46),
               np.zeros((10, 9, 8, 7, 5)), (10, 9, 8, 7, 3, 2, 4),
               (4, 5, 6), (1, 2), (0, 1, 2, 3, 10),
               "Invalid scatter_dims_to_operand_dims mapping"),
              ("RepeatedValuesInScatterDimsToOperandDims", (50, 49, 48, 47, 46),
               np.zeros((10, 9, 8, 7, 5)), (10, 9, 8, 7, 3, 2, 4),
               (4, 5, 6), (1, 2), (0, 1, 2, 2, 3),
               "scatter_dims_to_operand_dims in scatter op must not repeat"),
              ("InsufficientWindowDims", (50, 49, 48, 47, 46),
               np.zeros((10, 9, 8, 7, 5)), (10, 9, 8, 7, 3, 2, 4),
               (4, 5, 6), (1,), (0, 1, 2, 3),
               ("Scatter op has window of size 4; doesn't match operand of "
                "rank 5."))
           ]
      )
  def testScatterShapeCheckingRule(self, operand_shape, indices,
                                   update_shape, dimension_numbers, msg):

    def f(x, y):
      operand = lax.broadcast(x, operand_shape)
      updates = lax.broadcast(y, update_shape)
      return lax.scatter(operand, indices, updates, dimension_numbers)
    with self.assertRaisesRegex(TypeError, msg):
      jax.eval_shape(f, np.int32(1), np.int32(1))

  def testIssue831(self):
    # Tests the DeviceTuple constant handler
    def f(x):
      g = lambda *args: args[1]
      return jax.jit(lax.fori_loop, static_argnums=(2,))( 0, 10, g, x)

    jax.jit(f)(1.)  # doesn't crash

  def testReshapeWithUnusualShapes(self):
    ans = lax.reshape(np.ones((3,), np.float32), (lax.add(1, 2), 1))
    self.assertAllClose(ans, np.ones((3, 1), np.float32))

    self.assertRaisesRegex(
      TypeError,
      "Shapes must be 1D sequences of concrete values of integer type.*",
      lambda: lax.reshape(np.ones(3,), (np.array([3, 1]),)))

    self.assertRaisesRegex(
      TypeError,
      "Shapes must be 1D sequences of concrete values of integer type.*",
      lambda: lax.reshape(np.ones(3,), (1.5, 2.0)))

  def testDynamicSliceTypeErrors(self):
    self.assertRaisesRegex(
      TypeError,
      "index arguments to dynamic_slice must be integers of the same type",
      lambda: lax.dynamic_slice(np.ones((3, 4), dtype=np.float32),
                                (np.int32(1), np.int16(2)), (2, 2)))

  def testDynamicUpdateSliceTypeErrors(self):
    self.assertRaisesRegex(
      TypeError,
      "index arguments to dynamic_update_slice must be integers of the same "
      "type",
      lambda: lax.dynamic_update_slice(np.ones((3, 4), dtype=np.float32),
                                       np.zeros((2, 2), dtype=np.float32),
                                       (np.int32(1), np.int16(2))))

  def test_primitive_jaxtype_error(self):
    with jax.enable_checks(False):
      with self.assertRaisesRegex(
          TypeError, "Argument .* of type .* is not a valid JAX type"):
        lax.add(1, 'hi')

  def test_reduction_with_repeated_axes_error(self):
    with self.assertRaisesRegex(ValueError, "duplicate value in 'axes' .*"):
      lax.reduce(np.arange(3), 0, lax.add, (0, 0))

  @parameterized.parameters([lax.rem, lax.lt, lax.gt, lax.ge, lax.le])
  def test_ops_do_not_accept_complex_dtypes(self, op):
    with self.assertRaisesRegex(TypeError, ".*does not accept dtype complex.*"):
      op(2+3j, 4+5j)

  def test_population_count_booleans_not_supported(self):
    # https://github.com/google/jax/issues/3886
    msg = "population_count does not accept dtype bool"
    with self.assertRaisesRegex(TypeError, msg):
      lax.population_count(True)

  def test_conv_general_dilated_different_input_ranks_error(self):
    # https://github.com/google/jax/issues/4316
    msg = ("conv_general_dilated lhs and rhs must have the same number of "
           "dimensions")
    dimension_numbers = lax.ConvDimensionNumbers(lhs_spec=(0, 1, 2),
                                                 rhs_spec=(0, 1, 2),
                                                 out_spec=(0, 1, 2))
    kwargs = { 'window_strides': (1,)
             , 'padding': ((0, 0),)
             , 'lhs_dilation': (1,)
             , 'rhs_dilation': (1,)
             , 'dimension_numbers': dimension_numbers
             , 'feature_group_count': 1
             , 'batch_group_count': 1
             , 'precision': None
             }
    lhs, rhs = np.ones((1, 1, 1)), np.ones((1, 1, 1, 1))
    with self.assertRaisesRegex(ValueError, msg):
      lax.conv_general_dilated(lhs, rhs, **kwargs)

  def test_window_strides_dimension_shape_rule(self):
    # https://github.com/google/jax/issues/5087
    msg = ("conv_general_dilated window and window_strides must have "
           "the same number of dimensions")
    lhs = jax.numpy.zeros((1, 1, 3, 3))
    rhs = np.zeros((1, 1, 1, 1))
    with self.assertRaisesRegex(ValueError, msg):
      jax.lax.conv(lhs, rhs, [1], 'SAME')

  def test_reduce_window_scalar_init_value_shape_rule(self):
    # https://github.com/google/jax/issues/4574
    args = { "operand": np.ones((4, 4), dtype=np.int32)
           , "init_value": np.zeros((1,), dtype=np.int32)
           , "computation": lax.max
           , "window_dimensions": (2, 2)
           , "window_strides": (2, 2)
           , "padding": "VALID"
           , "base_dilation": (1, 1)
           , "window_dilation": (1, 1)
           }

    msg = (r"reduce_window expected init_values to be scalars but init_values "
           r"have shapes \[\(1,\)\].")
    with self.assertRaisesRegex(TypeError, msg):
      lax.reduce_window(**args)

  def test_reduce_correctly_works_with_pytrees(self):
    operands = {'x': [np.ones(5), np.arange(5)]}
    init_values = {'x': [0., 0]}
    result = lax.reduce(operands, init_values,
                        lambda x, y: jax.tree.map(lax.add, x, y),
                        [0])
    self.assertDictEqual(result, {'x': [5., 10]})

  def test_reduce_with_mismatched_pytrees_errors(self):
    operands = {'x': np.ones(5)}
    bad_init_values = {'y': 0.}

    with self.assertRaisesRegex(ValueError, 'Operands must have the same '
                                'tree structure as init_values'):
      lax.reduce(operands, bad_init_values,
                 lambda x, y: dict(x=x['x'] + y['x']), [0])

  def test_reduce_with_nonscalar_inits_errors(self):
    operands = {'x': np.ones(5)}
    bad_init_values = {'x': np.ones(5)}

    with self.assertRaisesRegex(ValueError,
                                'reduce found non-scalar initial value'):
      lax.reduce(operands, bad_init_values,
                 lambda x, y: dict(x=x['x'] + y['x']), [0])

  def test_select_jvp_complexity(self):
    jaxpr = jax.make_jaxpr(lambda x: jax.jvp(lambda x: lax.select(True, x, x),
                                             (x,), (1.,)))(1.)
    self.assertLen(jaxpr.jaxpr.eqns, 2)

  def testRngBitGenerator(self):
    # This test covers the original behavior of lax.rng_bit_generator, which
    # required x64=True, and only checks shapes and jit invariance.
    if not config.enable_x64.value:
      raise SkipTest("RngBitGenerator requires 64bit key")

    key = np.array((1, 2)).astype(np.uint64)
    def fn(k):
      return lax.rng_bit_generator(
          k, shape=(5, 7), algorithm=lax.RandomAlgorithm.RNG_THREE_FRY)

    out = fn(key)
    out_jit = jax.jit(fn)(key)
    self.assertEqual(out[0].shape, (2,))
    self.assertEqual(out[1].shape, (5, 7))
    self.assertArraysEqual(out[0], out_jit[0])
    self.assertArraysEqual(out[1], out_jit[1])

  def testRngBitGenerator2(self):
    def f(key):
      return lax.rng_bit_generator(key, shape=(5, 7))

    key = np.array((1, 2, 3, 4)).astype(np.uint32)
    out1 = f(key)
    out2 = jax.jit(f)(key)
    self.assertEqual(out1[0].shape, (4,))
    self.assertEqual(out1[1].shape, (5, 7))
    self.assertArraysEqual(out1[0], out2[0])
    self.assertArraysEqual(out1[1], out2[1])

  @jtu.skip_on_devices("tpu")
  def testRngBitGeneratorReturnedKey(self):
    # This test ensures that the key bit-packing/unpacking operations used in
    # the translation rule for rng_bit_generator, on older jaxlibs and at time
    # of writing on GPU, are inverses of one another.
    key = np.array([3, 1, 4, 2], dtype=np.dtype('uint32'))
    new_key, _ = lax.rng_bit_generator(key, (0,))
    self.assertAllClose(key, new_key)

  def test_rng_bit_generator_vmap(self):
    def f(key):
      return lax.rng_bit_generator(key, shape=(5, 7))

    keys = np.arange(3 * 4).reshape((3, 4)).astype(np.uint32)
    out_keys, bits = jax.vmap(f)(keys)
    self.assertEqual(out_keys.shape, (3, 4))
    self.assertEqual(bits.shape, (3, 5, 7))

  def test_rng_bit_generator_vmap_vmap(self):
    def f(key):
      return lax.rng_bit_generator(key, shape=(5, 7))

    keys = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.uint32)
    out_keys, bits = jax.vmap(jax.vmap(f))(keys)
    self.assertEqual(out_keys.shape, (2, 3, 4))
    self.assertEqual(bits.shape, (2, 3, 5, 7))

  @jtu.sample_product(
      dtype=lax_test_util.all_dtypes + lax_test_util.python_scalar_types,
      weak_type=[True, False],
  )
  def test_const(self, dtype, weak_type):
    if dtype in set(lax_test_util.python_scalar_types):
      val = dtype(0)
    else:
      val = lax_internal._convert_element_type(0, dtype, weak_type=weak_type)

    const = lax_internal._const(val, 0)
    self.assertEqual(dtypes.dtype(val, canonicalize=True),
                     dtypes.dtype(const, canonicalize=True))

  def testIgammaSpecial(self):
    self.assertEqual(lax.igamma(1., np.inf), 1.)
    self.assertEqual(lax.igammac(1., np.inf), 0.)

  def testRegressionIssue5728(self):
    # The computation in this test gave garbage data on CPU due to an LLVM bug.
    @jax.jit
    def f(inputs):
      out_action_2 = lax.slice_in_dim(inputs, 0, 15, axis=-1)
      mask = lax.slice_in_dim(inputs, 7, 22, axis=-1)
      out_action_2 = lax.select(lax.eq(mask, np.float32(0)),
                                lax.broadcast(np.float32(42), (1, 15)),
                                out_action_2)
      return lax.pad(out_action_2, np.float32(42), [(0, 0, 0), (0, 15, 0)])
    self.assertArraysEqual(np.full((1, 30), np.float32(42)),
                           f(np.zeros((1, 24), dtype=np.float32)))

  def testDynamicSliceUnsignedNoNormalization(self):
    # Test that no negative index correction is done for unsigned indices.
    f = lambda x, i: lax.dynamic_slice(x, [i], [1])
    x = np.arange(200)
    i = np.uint32(128)
    jaxpr = jax.make_jaxpr(f)(x, i)
    self.assertLen(jaxpr.eqns, 1)
    self.assertEqual(jaxpr.eqns[0].primitive, lax.dynamic_slice_p)

  def testDynamicSliceU8Index(self):
    # Regression test for u8 index in dynamic-slice (#6122)
    x = np.arange(200)
    np.testing.assert_equal(
        np.array(lax.dynamic_slice(x, np.uint8([128]), (1,))), [128])

  def test_dot_general_batching_python_builtin_arg(self):
    # https://github.com/google/jax/issues/16805
    @jax.remat
    def f(x):
      return jax.lax.dot_general(x, x, (([], []), ([], [])))

    jax.hessian(f)(1.0)  # don't crash

  def test_constant_folding_complex_to_real_scan_regression(self):
    # regression test for github.com/google/jax/issues/19059
    def g(hiddens):
        hiddens_aug = jnp.vstack((hiddens[0], hiddens))
        new_hiddens = hiddens_aug.copy()
        diff = new_hiddens[:-1] - hiddens
        diff = new_hiddens[:-1] - hiddens
        out = jnp.trace(jnp.conj(diff).T @ diff).real
        return jnp.array(out, dtype=jnp.complex64)


    def _step(carry, arg):
        primals, f_vjp = jax.vjp(
            g,
            jax.random.normal(jax.random.key(0), (9, 8), dtype=jnp.complex64),
        )
        out = f_vjp(np.array(1.0 + 0j, 'complex64'))[0]
        return carry, carry

    a, b = jax.lax.scan(_step, 0, jnp.arange(4, dtype=jnp.complex64))

  @parameterized.parameters([float, np.array, np.float32, jnp.float32])
  def testAsarray(self, typ):
    x = typ(1.0)
    x_arr = lax_internal.asarray(x)
    self.assertArraysEqual(x, x_arr)
    self.assertIsInstance(x_arr, jax.Array)

    # jaxpr should not bind any primitives, whether called directly or
    # as a closure:
    jaxpr = jax.make_jaxpr(lax_internal.asarray)(x)
    self.assertLen(jaxpr.eqns, 0)

    asarray_closure = lambda: lax_internal.asarray(x)
    jaxpr = jax.make_jaxpr(asarray_closure)()
    self.assertLen(jaxpr.eqns, 0)

    # Regression test for https://github.com/google/jax/issues/19334
    # lax.asarray as a closure should not trigger transfer guard.
    with jax.transfer_guard('disallow'):
      jax.jit(asarray_closure)()



class LazyConstantTest(jtu.JaxTestCase):
  def _Check(self, make_const, expected):
    # check casting to ndarray works
    asarray_result = np.asarray(make_const())

    # check passing as an argument works (should hit constant handler)
    zero = np.array(0, expected.dtype)
    argument_result = lax.add(zero, make_const())

    # check looping into a compiled computation works
    jit_result = jax.jit(lambda x: lax.add(x, make_const()))(zero)

    # ensure they're all the same
    self.assertAllClose(asarray_result, expected)
    self.assertAllClose(argument_result, expected)
    self.assertAllClose(jit_result, expected)

    # ensure repr doesn't crash
    repr(make_const())

  @jtu.sample_product(
    dtype=lax_test_util.default_dtypes + [None],
    shape=[(), (3,), (2, 3), (2, 3, 4), (1001, 1001)],
    fill_value=[0, 1, np.pi],
  )
  def testFilledConstant(self, shape, fill_value, dtype):
    make_const = lambda: lax.full(shape, fill_value, dtype)
    expected = np.full(shape, fill_value,
                        dtype or dtypes.dtype(fill_value))
    self._Check(make_const, expected)

  @jtu.sample_product(
    [dict(shape=shape, dimension=dimension)
     for shape in [(), (3,), (2, 3), (2, 3, 4)]
     # TODO(mattjj): re-enable (1001, 1001), (101, 101, 101),
     for dimension in range(len(shape))],
    dtype=lax_test_util.default_dtypes,
  )
  def testIotaConstant(self, dtype, shape, dimension):
    make_const = lambda: lax.broadcasted_iota(dtype, shape, dimension)

    arr = np.arange(shape[dimension], dtype=dtypes.canonicalize_dtype(dtype))
    singleton_shape = [1] * len(shape)
    singleton_shape[dimension] = shape[dimension]
    expected = np.broadcast_to(arr.reshape(singleton_shape), shape)

    self._Check(make_const, expected)

  @jtu.sample_product(
    [dict(shape=shape, axes=axes)
      for shape, axes in [
          [(2, 3), (0, 1)],
          [(2, 3, 4), (0, 1)],
          [(2, 3, 4), (0, 2)],
          [(2, 3, 4), (1, 2)],
          [(2, 3, 4), (0, 1, 2)],
          [(2, 3, 4, 2), (0, 1, 2)],
          [(2, 3, 4, 2), (0, 2, 3)],
          [(1001, 1001), (0, 1)],
      ]],
    dtype=lax_test_util.default_dtypes,
  )
  def testDeltaConstant(self, dtype, shape, axes):
    make_const = lambda: lax_internal._delta(dtype, shape, axes)
    # don't check the asarray case, just assume it's right
    expected = np.asarray(make_const())
    self._Check(make_const, expected)

  def testBroadcastInDim(self):
    arr = lax.full((2, 1), 1.) + 1.
    arr_np = np.full((2, 1), 1.) + 1.
    expected = lax_reference.broadcast_in_dim(arr_np, (2, 1, 3), (0, 2))
    make_const = lambda: lax.broadcast_in_dim(arr, (2, 1, 3), (0, 2))
    self._Check(make_const, expected)

  @jtu.sample_product(
    input_type=[int, float, np.int32, np.float32, np.array],
    dtype=[np.int32, np.float32],
    jit=[True, False],
    value=[0, 1],
  )
  def testConvertElementReturnType(self, input_type, dtype, value, jit):
    op = lambda x: lax.convert_element_type(x, dtype)
    if jit:
      op = jax.jit(op)
    result = op(input_type(value))
    assert isinstance(result, jax.Array)

  @jtu.sample_product(
      dtype_in=lax_test_util.all_dtypes, dtype_out=lax_test_util.all_dtypes)
  @jtu.ignore_warning(category=NumpyComplexWarning)
  def testConvertElementTypeAvoidsCopies(self, dtype_in, dtype_out):
    x = jax.device_put(np.zeros(5, dtype_in))
    self.assertEqual(x.dtype, dtype_in)
    y = lax.convert_element_type(x, dtype_out)
    self.assertEqual(y.dtype, dtype_out)
    x_buf = x
    y_buf = y
    if np.dtype(dtype_in) == np.dtype(dtype_out):
      self.assertEqual(x_buf.unsafe_buffer_pointer(),
                       y_buf.unsafe_buffer_pointer())
    else:
      self.assertNotEqual(x_buf.unsafe_buffer_pointer(),
                          y_buf.unsafe_buffer_pointer())

  @jtu.sample_product(
    index_dtype=jtu.dtypes.all_inexact + jtu.dtypes.boolean,
    jax_fn=[lax.argmin, lax.argmax],
  )
  def testArgMinMaxIndexDtypeError(self, jax_fn, index_dtype):
    with self.assertRaisesRegex(TypeError,
                                "index_dtype must be an integer type"):
      jax_fn(np.ones((2, 2)), axis=0, index_dtype=index_dtype)

  @parameterized.parameters([lax.argmin, lax.argmax])
  def testArgMinMaxEmptyError(self, jax_fn):
    with self.assertRaisesRegex(ValueError,
                                "require non-empty reduced dimension"):
      jax_fn(np.ones((0, 2)), axis=0, index_dtype=np.int32)

  @parameterized.parameters([lax.argmin, lax.argmax])
  def testArgMinMaxInvalidAxisError(self, jax_fn):
    with self.assertRaisesRegex(ValueError,
                                "Invalid axis -1 for operand"):
      jax_fn(np.ones((2, 3)), axis=-1, index_dtype=np.int32)

  @jtu.sample_product(
    jax_fn=[lax.argmin, lax.argmax],
    weak_type=[False, True],
  )
  def testArgMinMaxWeakType(self, jax_fn, weak_type):
    op = lambda x: jax_fn(x, axis=0, index_dtype=np.int32)
    x_in = lax_internal._convert_element_type(np.ones((2, 2)),
                                              weak_type=weak_type)
    self.assertEqual(dtypes.is_weakly_typed(x_in), weak_type)
    x_out = op(x_in)
    self.assertEqual(dtypes.is_weakly_typed(x_out), False)
    x_out_jit = jax.jit(op)(x_in)
    self.assertEqual(dtypes.is_weakly_typed(x_out_jit), False)

  def testArgMaxOfNanChoosesNaN(self):
    self.assertEqual(lax.argmax(np.array([0., np.nan]), axis=0,
                                index_dtype=np.int32), 1)

  unary_op_types = {}
  for r in lax_test_util.lax_ops():
    if r.nargs == 1:
      unary_op_types[r.op] = (unary_op_types.get(r.op, set()) |
                              {np.dtype(t) for t in r.dtypes})

  @parameterized.named_parameters(
      {"testcase_name": f"_{op}", "op_name": op, "rec_dtypes": dtypes}
      for op, dtypes in unary_op_types.items())
  def testUnaryWeakTypes(self, op_name, rec_dtypes):
    """Test that all lax unary ops propagate weak_type information appropriately."""
    if op_name == "bitwise_not":
      raise unittest.SkipTest("https://github.com/google/jax/issues/12066")
    # Find a valid dtype for the function.
    for dtype in [float, int, complex, bool]:
      dtype = dtypes.canonicalize_dtype(dtype)
      if dtype in rec_dtypes:
        py_val = dtype.type(1).item()
        lax_val = lax.full((), py_val, dtype)
        break
    else:
      raise ValueError(f"no available dtypes in {rec_dtypes}")

    op = getattr(lax, op_name)
    py_op = op(py_val)
    lax_op = op(lax_val)

    self.assertAllClose(py_op, lax_op, check_dtypes=True)
    self.assertFalse(lax_op.aval.weak_type)
    if type(py_val) == bool:
      # Booleans should have weak types stripped.
      self.assertFalse(py_op.aval.weak_type)
    else:
      self.assertTrue(py_op.aval.weak_type)

  def testCumsumLengthOne(self):
    # regression test for issue 4672
    x = lax.full((1,), 1)
    out = lax.cumsum(x)
    self.assertArraysEqual(out, x)

  def testLog1pNearOne(self):
    expected = np.log1p(np.float32(1e-5))
    np.testing.assert_array_almost_equal_nulp(
        expected.astype(np.float32), lax.log1p(np.float32(1e-5)))
    np.testing.assert_array_almost_equal_nulp(
        expected.astype(np.complex64), lax.log1p(np.complex64(1e-5)))


class LaxNamedShapeTest(jtu.JaxTestCase):

  def test_abstract_eval(self):
    aval1 = core.ShapedArray((2, 3), np.float32, False, {'i': 10})
    out, _ = lax.sin_p.abstract_eval(aval1)
    self.assertEqual(out, aval1)

    aval1 = core.ShapedArray((2, 3), np.float32, False, {'i': 10})
    aval2 = core.ShapedArray((2, 3), np.float32, False, {'j': 5})
    expected = core.ShapedArray((2, 3), np.float32, False, {'i': 10, 'j': 5})
    out, _ = lax.add_p.abstract_eval(aval1, aval2)
    self.assertEqual(out, expected)

  def test_abstract_eval_collective(self):
    with core.extend_axis_env('i', 10, None):
      aval1 = core.ShapedArray((2, 3), np.float32, False, {'i': 10, 'j': 5})
      expected = core.ShapedArray((2, 3), np.float32, False, {'j': 5})
      (out,), _ = lax.psum_p.abstract_eval(aval1, axes=('i',), axis_index_groups=None)
      self.assertEqual(out, expected)

class FooTyRules:
  # handlers

  @staticmethod
  def physical_element_aval(dtype) -> core.ShapedArray:
    return core.ShapedArray((2,), jnp.dtype('uint32'))

  @staticmethod
  def logical_sharding(aval, phys_sharding):
    return phys_sharding

  @staticmethod
  def physical_sharding(aval, sharding):
    return sharding

  @staticmethod
  def result_handler(sticky_device, aval):
    def handler(_, buf):
      buf.aval = core.ShapedArray(buf.shape, buf.dtype)
      return FooArray(aval.shape, buf)
    return handler

  @staticmethod
  def global_sharded_result_handler(aval, out_sharding, committed):
    def handler(arr):
      from jax._src.array import ArrayImpl
      if isinstance(arr, ArrayImpl):
        buf, = arr._arrays
      else:
        buf, = arr
      return FooArray(aval.shape, buf)
    return handler

  @staticmethod
  def replicate_trailing_dims(ctx, val, aval):
    return val

  @staticmethod
  def check_replicated_trailing_dims(sharding: jax.sharding.GSPMDSharding, aval):
    pass


class FooTy(dtypes.ExtendedDType):
  type = dtypes.extended
  name = 'foo'
  _rules = FooTyRules

  def __hash__(self) -> int:
    return hash(FooTy)
  def __eq__(self, other) -> bool:
    return type(other) is FooTy
  def __repr__(self) -> str:
    return self.name
  __str__ = __repr__

# primitives

make_p = core.Primitive('make')
bake_p = core.Primitive('bake')
take_p = core.Primitive('take')
jake_p = core.Primitive('jake')

def make(shape): return make_p.bind(shape=tuple(shape))
def bake(k):     return bake_p.bind(k)
def take(k):     return take_p.bind(k)
def jake(k):     return jake_p.bind(k)

@make_p.def_abstract_eval
def make_abstract_eval(*, shape):
  return core.ShapedArray(shape, FooTy())

@bake_p.def_abstract_eval
def bake_abstract_eval(x):
  if type(x.dtype) != FooTy: raise TypeError
  return core.ShapedArray(tuple(reversed(x.shape)), FooTy())

@take_p.def_abstract_eval
def take_abstract_eval(x):
  return core.ShapedArray(x.shape, jnp.dtype('float32'))

@jake_p.def_abstract_eval
def jake_abstract_eval(x):
  return x

# runtime ('outside jit') data types

class FooArray:
  shape: tuple[int, ...]
  data: jax.Array

  def __init__(self, shape, data):
    assert data.shape == (*shape, 2)
    self.shape = shape
    self.data = data

  def __repr__(self) -> str:
    shape = ','.join(map(str, self.shape))
    return f'foo[{shape}] with value\n{self.data}'

  size = property(lambda self: self.data.size // 2)
  ndim = property(lambda self: self.data.ndim - 1)

def shard_foo_array_handler(x, sharding):
  device, = sharding._addressable_device_assignment
  aval = core.raise_to_shaped(core.get_aval(x.data))
  return pxla.batched_device_put(
      aval, jax.sharding.SingleDeviceSharding(device), [x.data], [device])

def foo_array_constant_handler(x):
  return array._array_mlir_constant_handler(x.data)

def make_lowering(*, shape):
  return jnp.zeros((*shape, 2), 'uint32')

def bake_lowering(k):
  return k.T

def take_lowering(k):
  return jnp.broadcast_to(jnp.float32(k.size), k.shape)

def jake_lowering(k):
  return jnp.ones((*k.shape, 2), 'uint32')

def bake_vmap(batched_args, batch_dims):
  xs, = batched_args
  bdim_in, = batch_dims
  ys = bake(xs)
  perm = list(reversed(range(xs.ndim)))
  bdim_out = perm[bdim_in]
  return ys, bdim_out


class CustomElementTypesTest(jtu.JaxTestCase):

  def setUp(self):
    core.pytype_aval_mappings[FooArray] = \
        lambda x: core.ShapedArray(x.shape, FooTy())
    xla.canonicalize_dtype_handlers[FooArray] = lambda x: x
    xla.pytype_aval_mappings[FooArray] = \
        lambda x: core.ShapedArray(x.shape, FooTy())
    pxla.shard_arg_handlers[FooArray] = shard_foo_array_handler
    mlir._constant_handlers[FooArray] = foo_array_constant_handler
    mlir.register_lowering(make_p, mlir.lower_fun(make_lowering, False))
    mlir.register_lowering(bake_p, mlir.lower_fun(bake_lowering, False))
    mlir.register_lowering(take_p, mlir.lower_fun(take_lowering, False))
    mlir.register_lowering(jake_p, mlir.lower_fun(jake_lowering, False))
    batching.defvectorized(take_p)
    batching.primitive_batchers[bake_p] = bake_vmap

  def tearDown(self):
    del core.pytype_aval_mappings[FooArray]
    del xla.canonicalize_dtype_handlers[FooArray]
    del xla.pytype_aval_mappings[FooArray]
    del mlir._constant_handlers[FooArray]
    del mlir._lowerings[make_p]
    del mlir._lowerings[bake_p]
    del mlir._lowerings[take_p]
    del batching.primitive_batchers[take_p]
    del batching.primitive_batchers[bake_p]

  def test_shaped_array_construction(self):
    aval = core.ShapedArray((), FooTy())
    self.assertEqual(aval.str_short(), 'foo[]')
    aval = core.ShapedArray((3, 4), FooTy())
    self.assertEqual(aval.str_short(), 'foo[3,4]')

  def test_make_jaxpr_identity(self):
    x = types.SimpleNamespace(shape=(3,), dtype=FooTy())
    jaxpr = jax.make_jaxpr(lambda x: x)(x).jaxpr
    # { lambda ; a:foo[3]. let  in (a,) }
    self.assertLen(jaxpr.invars, 1)
    a, = jaxpr.invars
    self.assertEqual(a.aval, core.ShapedArray((3,), FooTy()))
    self.assertLen(jaxpr.outvars, 1)
    a, = jaxpr.outvars
    self.assertEqual(a.aval, core.ShapedArray((3,), FooTy()))

  # tests after here need the primitives

  def test_make_jaxpr_with_primitives(self):
    def f():
      k1 = make((3, 4))
      k2 = bake(k1)
      x  = take(k2)
      return x

    jaxpr = jax.make_jaxpr(f)().jaxpr
    # { lambda ; . let
    #     a:foo[3,4] = make[shape=(3, 4)]
    #     b:foo[4,3] = bake a
    #     c:f32[4,3] = take b
    #   in (c,) }
    self.assertLen(jaxpr.invars, 0)
    self.assertLen(jaxpr.eqns, 3)
    e1, e2, e3 = jaxpr.eqns

    self.assertIs(e1.primitive, make_p)
    self.assertLen(e1.outvars, 1)
    a, = e1.outvars
    self.assertEqual(a.aval, core.ShapedArray((3, 4), FooTy()))

    self.assertIs(e2.primitive, bake_p)
    self.assertLen(e2.outvars, 1)
    b, = e2.outvars
    self.assertEqual(b.aval, core.ShapedArray((4, 3), FooTy()))

    self.assertIs(e3.primitive, take_p)
    self.assertLen(e3.outvars, 1)
    c, = e3.outvars
    self.assertEqual(c.aval, core.ShapedArray((4, 3), np.dtype('float32')))

  # tests after here need FooArray and lowerings

  def test_jit_closure(self):
    k = FooArray((), jnp.arange(2, dtype='uint32'))

    @jax.jit
    def f():
      jnp.add(1, 1)  # make jit not hit trivial dispatch path
      return k

    y = f()  # doesn't crash
    self.assertIsInstance(y, FooArray)
    self.assertEqual(y.shape, ())

  def test_jit_identity(self):
    k = FooArray((), jnp.arange(2, dtype='uint32'))

    @jax.jit
    def f(k):
      jnp.add(1, 1)  # make jit not hit trivial dispatch path
      return k

    y = f(k)  # doesn't crash
    self.assertIsInstance(y, FooArray)
    self.assertEqual(y.shape, ())

  def test_jit_multiple_primitives(self):
    @jax.jit
    def f():
      k1 = make((3,))
      k2 = bake(k1)
      y  = take(k2)
      return y

    y = f()
    self.assertArraysAllClose(y, jnp.array([3., 3., 3.]), check_dtypes=False)

  def test_scan_jaxpr(self):
    ks = jax.jit(lambda: make((3, 4)))()
    f = lambda ks: jax.lax.scan(lambda _, k: (None, bake(k)), None, ks)
    jaxpr = jax.make_jaxpr(f)(ks).jaxpr
    # { lambda ; a:foo[3,4]. let
    #     b:foo[3,4] = scan[
    #       jaxpr={ lambda ; c:foo[4]. let d:foo[4] = bake c in (d,) }
    #     ] a
    #   in (b,) }
    self.assertLen(jaxpr.invars, 1)
    a, = jaxpr.invars
    self.assertEqual(a.aval, core.ShapedArray((3, 4), FooTy()))
    self.assertLen(jaxpr.eqns, 1)
    e, = jaxpr.eqns
    self.assertLen(e.outvars, 1)
    b, = e.outvars
    self.assertEqual(b.aval, core.ShapedArray((3, 4), FooTy()))

  def test_scan_lowering(self):
    ks = jax.jit(lambda: make((3, 4)))()
    f = lambda ks: jax.lax.scan(lambda _, k: (None, bake(k)), None, ks)
    _, out = jax.jit(f)(ks)  # doesn't crash
    self.assertIsInstance(out, FooArray)
    self.assertEqual(out.shape, (3, 4))

  def test_vmap(self):
    ks = jax.jit(lambda: make((3, 4, 5)))()
    ys = jax.vmap(jax.jit(lambda k: take(bake(k))))(ks)
    expected = jnp.broadcast_to(3 * 4 * 5, (3, 5, 4)).astype('float32')
    self.assertAllClose(ys, expected)

  def test_slice(self):
    ks = jax.jit(lambda: make((3, 4)))()
    ys = jax.jit(lambda x: lax.slice_in_dim(x, 1, 3))(ks)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (2, 4))

  def test_dynamic_slice(self):
    ks = jax.jit(lambda: make((3, 4)))()
    ys = jax.jit(lambda x, i: lax.dynamic_slice_in_dim(x, i, 2))(ks, 1)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (2, 4))

  def test_transpose(self):
    ks = jax.jit(lambda: make((3, 4)))()
    ys = jax.jit(lambda x: x.T)(ks)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (4, 3))

  def test_gather(self):
    ks = jax.jit(lambda: make((3, 4)))()
    ys = jax.jit(lambda x: x[1])(ks)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (4,))

    ks = jax.jit(lambda: make((3, 4, 5)))()

    ys = jax.jit(lambda x: x[1])(ks)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (4, 5))

    ys = jax.jit(lambda x: x[1, 2:4])(ks)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (2, 5))

    ys = jax.jit(lambda x: x[1, 2:4, 3])(ks)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (2,))

    ys = jax.jit(lambda x: x[:, 2:4, 3:4])(ks)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (3, 2, 1))

  def test_gather_batched_index_dtype(self):
    # Regression test for https://github.com/google/jax/issues/16557
    dtype = jnp.int8
    size = jnp.iinfo(dtype).max + 10
    indices = jnp.zeros(size, dtype=dtype)
    values = jnp.zeros((size, 1))
    results = jax.vmap(lambda x, i: jnp.take(x, i, axis=0))(values, indices)
    self.assertArraysEqual(results, jnp.zeros(size))

  @parameterized.parameters([
    (0,),
    (slice(1),),
    (np.array([0, 2]),),
    (np.array([False, True, True]),)
  ])
  def test_scatter(self, idx):
    k  = jax.jit(lambda: make(()))()
    ks = jax.jit(lambda: make((3,)))()
    ys = jax.jit(lambda x, y: x.at[idx].set(y))(ks, k)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (3,))

  def test_equality(self):
    eq = jax.jit(lambda k1, k2: k1 == k2)
    ne = jax.jit(lambda k1, k2: k1 != k2)

    k1 = jax.jit(lambda: make(()))()
    k2 = jax.jit(lambda: jake(make(())))()

    self.assertTrue(eq(k1, k1))
    self.assertFalse(eq(k1, k2))
    self.assertTrue(ne(k1, k2))
    self.assertFalse(ne(k1, k1))

    size = 5
    idx = slice(2, 4)
    ks = jax.jit(lambda k: jake(make((size,))).at[idx].set(k))(k1)
    expected = jnp.zeros(size, dtype=bool).at[idx].set(True)
    self.assertArraysEqual(eq(k1, ks), expected)
    self.assertArraysEqual(ne(k1, ks), ~expected)

  def test_select(self):
    ks = jax.jit(lambda: make((3,)))()
    cs = jnp.array([True, False, False])
    ys = jax.jit(lax.select)(cs, ks, ks)
    self.assertIsInstance(ys, FooArray)
    self.assertEqual(ys.shape, (3,))

  def test_xla_reverse_bug(self):
    # Regression test for b/248295786
    # This was an XLA bug related to an incorrect optimization of reverse
    def f(x):
      y = jnp.array([2, 5])
      return lax.rev(x * y, (0,))
    x = jnp.array([1, 2])
    self.assertArraysEqual(f(x), jax.jit(f)(x))

  # TODO(frostig,mattjj): more polymorphic primitives tests


class FunctionAccuracyTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    dict(testcase_name=f"_{dtype.__name__}", dtype=dtype)
    for dtype in jtu.dtypes.supported([np.float32, np.float64, np.complex64, np.complex128]))
  def testMPMathUtils(self, dtype):
    try:
      import mpmath
    except ImportError as msg:
      self.skipTest(f'could not import mpmath: {msg}')

    prec = {np.float32: 24, np.float64: 53, np.complex64: 24, np.complex128: 53}[dtype]
    is_complex = dtype().dtype.kind == 'c'

    def func(x):
      assert isinstance(x, mpmath.ctx_mp.mpnumeric)
      assert x.context.prec == prec
      assert isinstance(x, x.context.mpc if is_complex else x.context.mpf)
      return x

    ufunc = jtu.vectorize_with_mpmath(func, mpmath=mpmath)

    with jtu.ignore_warning(category=RuntimeWarning, message="(overflow|invalid value|divide by zero) encountered in.*"):
      if is_complex:
        arr = jtu.complex_plane_sample(dtype=dtype, size_re=11)
      else:
        cdtype = getattr(np, ufunc.map_float_to_complex[dtype.__name__])
        arr = jtu.complex_plane_sample(dtype=cdtype, size_re=11, size_im=0)[1:2].real

    arr2 = ufunc.mptonp(ufunc.nptomp(arr))
    with jtu.ignore_warning(category=RuntimeWarning, message="(overflow|invalid value|divide by zero) encountered in.*"):
      self.assertAllClose(arr, arr2, atol=0, rtol=0)

    arr3 = ufunc(arr)
    with jtu.ignore_warning(category=RuntimeWarning, message="(overflow|invalid value|divide by zero) encountered in.*"):
      self.assertAllClose(arr, arr3, atol=0, rtol=0)

    if is_complex:
      # tests scale in normalize
      v = dtype(1.1071487177940644+1.1102230246251565e-16j)
      r = dtype(1.1071487177940644+0j)
      mnp = jtu.numpy_with_mpmath(mpmath, extra_prec=1)
      nr, nv = mnp.normalize(r, r, v)
      self.assertAllClose(nr, nv)

  _functions_on_complex_plane = [
    'arccos', 'arccosh', 'arcsin', 'arcsinh',
    'arctan', 'arctanh', 'conjugate', 'cos',
    'cosh', 'exp', 'exp2', 'expm1', 'log',
    'log10', 'log1p', 'sin', 'sinh', 'sqrt',
    'square', 'tan', 'tanh', 'sinc', 'positive',
    'negative', 'absolute', 'sign'
  ]

  @parameterized.named_parameters(
    dict(testcase_name=f"_{name}_{dtype.__name__}", name=name, dtype=dtype)
    for name, dtype in itertools.product(
        _functions_on_complex_plane,
        jtu.dtypes.supported([np.complex64, np.complex128]),
    ))
  @jtu.skip_on_devices("tpu")
  def testSuccessOnComplexPlane(self, name, dtype):
    self._testOnComplexPlaneWorker(name, dtype, 'success')

  @parameterized.named_parameters(
    dict(testcase_name=f"_{name}_{dtype.__name__}", name=name, dtype=dtype)
    for name, dtype in itertools.product(
        _functions_on_complex_plane,
        jtu.dtypes.supported([np.complex64, np.complex128]),
    ))
  @jtu.skip_on_devices("tpu")
  def testFailureOnComplexPlane(self, name, dtype):
    self._testOnComplexPlaneWorker(name, dtype, 'failure')

  def _testOnComplexPlaneWorker(self, name, dtype, kind):
    try:
      import mpmath
    except ImportError as msg:
      self.skipTest(f'could not import mpmath: {msg}')

    is_cpu = jtu.test_device_matches(["cpu"])
    machine = platform.machine()
    # TODO: remove is_arm_cpu as previously arm cpu related failures
    # were due to numpy issues. Confirm?
    is_arm_cpu = machine.startswith('aarch') or machine.startswith('arm')
    is_cuda = jtu.test_device_matches(["cuda"])

    size_re = 11
    size_im = 11
    atol = None

    mnp = jtu.numpy_with_mpmath(mpmath, extra_prec=1)
    mnp2 = jtu.numpy_with_mpmath(mpmath, extra_prec_multiplier=1)

    ref_op = getattr(mnp, name)
    ref2_op = getattr(mnp2, name)
    jnp_op = getattr(jnp, name)

    with jtu.ignore_warning(category=RuntimeWarning, message="(overflow|invalid value|divide by zero) encountered in.*"):
      args = (jtu.complex_plane_sample(dtype=dtype, size_re=size_re, size_im=size_im),)
      result = np.asarray(jnp_op(*args))
      expected = ref_op(*args)
      expected2 = ref2_op(*args)

    normalized_expected, normalized_result = mnp2.normalize(expected2, expected, result)

    # When comparing the results with expected, we'll divide the
    # complex plane grid into smaller regions and perform the
    # closeness tests on each region separately. The reason for this
    # is that the inaccuracy or incorrectness issues with a particular
    # function exists typically in specific regions while in other
    # regions the function is accurate. So, such a division of the
    # complex plane helps to identify the problematic regions as well
    # as to fix the inaccuracy or incorrectness issues.
    #
    # Regions in complex plane:
    #
    #       (                pinfj                 )
    #              (  q2   ) (posj) (  q1   )
    #       (ninf) (  neg  ) (zero) (  pos  ) (pinf)
    #              (  q3   ) (negj) (  q4   )
    #       (                 ninfj                )
    #
    # In addition, the 1/3 middle parts of regions q1, q2, q3, q4,
    # neg, pos are tested separately as these don't contain extremely
    # small or extremelly large values and functions on these regions
    # ought not to possess any incorrectness issues.

    s0, s1 = size_re, size_im
    s03, s13 = s0 // 3, s1 // 3
    s_dict = dict(
      q1=(slice(s0 + 2, -1), slice(s1 + 2, -1)),
      q2=(slice(s0 + 2, -1), slice(1, s1 + 1)),
      q3=(slice(1, s0 + 1), slice(1, s1 + 1)),
      q4=(slice(1, s0 + 1), slice(s1 + 2, -1)),
      neg=(s0 + 1, slice(1, s1 + 1)),
      pos=(s0 + 1, slice(s1 + 2, -1)),
      negj=(slice(1, s0 + 1), s1 + 1),
      posj=(slice(s0 + 2, -1), s1 + 1),
      ninf=(slice(None), 0),
      pinf=(slice(None), -1),
      ninfj=(0, slice(None)),
      pinfj=(-1, slice(None)),
      zero=(slice(s0 + 1, s0 + 2), slice(s1 + 1, s1 + 2)),
    )

    if s03 and s13:
      s_dict.update(
        mq1 = (slice(s0 + 3 + s03, s0 + 3 + 2 * s03), slice(s1 + 3 + s13, s1 + 3 + 2 * s13)),
        mq2 = (slice(s0 + 3 + s03, s0 + 3 + 2 * s03), slice(2 + s13, 2 + 2 * s13)),
        mq3 = (slice(2 + s03, 2 + 2 * s03), slice(2 + s13, 2 + 2 * s13)),
        mq4 = (slice(2 + s03, 2 + 2 * s03), slice(s1 + 3 + s13, s1 + 3 + 2 * s13)),
        mneg=(s0 + 1, slice(2 + s13, 2 + 2 * s13)),
        mpos=(s0 + 1, slice(s1 + 3 + s13, s1 + 3 + 2 * s13)),
        mnegj=(slice(2 + s03, 2 + 2 * s03), s1 + 1),
        mposj=(slice(s0 + 3 + s03, s0 + 3 + 2 * s03), s1 + 1),
      )

    # Start with an assumption that all regions are problematic for a
    # particular function:
    regions_with_inaccuracies = list(s_dict)

    # Next, we'll remove non-problematic regions from the
    # regions_with_inaccuracies list by explicitly keeping problematic
    # regions:
    def regions_with_inaccuracies_keep(*to_keep):
      for item in regions_with_inaccuracies[:]:
        if item not in to_keep:
          regions_with_inaccuracies.remove(item)

    if name == 'absolute':
      if is_cuda and dtype == np.complex128:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4')
      else:
        regions_with_inaccuracies.clear()

    elif name == 'sign':
      regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4')

    elif name == 'square':
      if is_cuda:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'ninf', 'pinf', 'ninfj', 'pinfj')
      if is_cpu:
        regions_with_inaccuracies_keep('ninf', 'pinf')

    elif name == 'log':
      regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'ninf', 'pinf', 'ninfj', 'pinfj')

    elif name == 'log10':
      regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'ninf', 'pinf', 'ninfj', 'pinfj', 'zero')

    elif name == 'log1p':
      regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj')
      # TODO(pearu): after landing openxla/xla#10503, switch to
      #   regions_with_inaccuracies_keep('ninf', 'pinf', 'ninfj', 'pinfj')

    elif name == 'exp':
      regions_with_inaccuracies_keep('pos', 'pinf', 'mpos')

    elif name == 'exp2':
      if dtype == np.complex64:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'mq1', 'mq2', 'mq3', 'mq4', 'mpos', 'mnegj', 'mposj')
      if dtype == np.complex128:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'mpos')

    elif name == 'expm1' and xla_extension_version < 250:
      regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'pinf', 'mq1', 'mq2', 'mq3', 'mq4', 'mneg', 'mpos')

    elif name == 'sinc':
      regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'mq1', 'mq2', 'mq3', 'mq4', 'mneg', 'mpos', 'mnegj', 'mposj')

    elif name == 'tan':
      regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'negj', 'posj', 'ninfj', 'pinfj', 'mq1', 'mq2', 'mq3', 'mq4', 'mnegj', 'mposj')

    elif name == 'sinh':
      if is_cuda:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'ninf', 'pinf', 'mq1', 'mq2', 'mq3', 'mq4', 'mneg', 'mpos')
      if is_cpu:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'mq1', 'mq2', 'mq3', 'mq4', 'mneg', 'mpos')

    elif name == 'cosh':
      regions_with_inaccuracies_keep('neg', 'pos', 'ninf', 'pinf', 'mneg', 'mpos')

    elif name == 'tanh':
      regions_with_inaccuracies_keep('ninf', 'pinf', 'ninfj', 'pinfj')

    elif name == 'arccos':
      if dtype == np.complex64:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq2', 'mq3', 'mneg', 'mpos', 'mnegj')
      if dtype == np.complex128:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq2', 'mq3', 'mq4', 'mneg', 'mpos', 'mnegj')

    elif name == 'arccosh':
      if dtype == np.complex64:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq2', 'mq3', 'mneg', 'mpos', 'mnegj')
      if dtype == np.complex128:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq2', 'mq3', 'mq4', 'mneg', 'mnegj')

    elif name == 'arcsin':
      if dtype == np.complex64:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq1', 'mq2', 'mq3', 'mq4', 'mneg', 'mpos', 'mnegj', 'mposj')
      if dtype == np.complex128:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq1', 'mq2', 'mq3', 'mq4', 'mneg', 'mpos', 'mnegj', 'mposj')

    elif name == 'arcsinh':
      if dtype == np.complex64:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq1', 'mq2', 'mq3', 'mq4', 'mneg', 'mpos', 'mnegj')
      if dtype == np.complex128:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq2', 'mq3', 'mneg', 'mnegj')

    elif name == 'arctan':
      if dtype == np.complex64:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mq1', 'mq2', 'mq3', 'mq4', 'mnegj', 'mposj')
      if dtype == np.complex128:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj')

    elif name == 'arctanh':
      if xla_extension_version < 251:
        regions_with_inaccuracies_keep('q1', 'q2', 'q3', 'q4', 'neg', 'pos', 'negj', 'posj', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mpos')
      else:
        regions_with_inaccuracies_keep('pos', 'ninf', 'pinf', 'ninfj', 'pinfj', 'mpos')

    elif name in {'positive', 'negative', 'conjugate', 'sin', 'cos', 'sqrt', 'expm1'}:
      regions_with_inaccuracies.clear()
    else:
      assert 0  # unreachable

    # Finally, perform the closeness tests per region:
    unexpected_success_regions = []
    for region_name, region_slice in s_dict.items():
      region = args[0][region_slice]
      inexact_indices = np.where(normalized_result[region_slice] != normalized_expected[region_slice])

      if inexact_indices[0].size == 0:
        inexact_samples = ''
      else:
        inexact_samples = []
        for ind in zip(*inexact_indices):
          x = region[ind]
          y1, y2 = result[region_slice][ind],  expected[region_slice][ind]
          ny1, ny2 = normalized_result[region_slice][ind],  normalized_expected[region_slice][ind]
          if str(y1) == str(y2):  # skip equal nan-s
            continue
          max_abs_diff = abs(ny1 - ny2).max() if np.isfinite(y1) and np.isfinite(y1) else np.inf
          inexact_samples.append((max_abs_diff, f'jax.numpy.{name}({x}) -> {y1} [{ny1}], expected {y2} [{ny2}]'))
        inexact_samples = "\n".join([msg for _, msg in sorted(inexact_samples)])

      if kind == 'success' and region_name not in regions_with_inaccuracies:
        with jtu.ignore_warning(category=RuntimeWarning, message="overflow encountered in.*"):
          self.assertAllClose(
            normalized_result[region_slice], normalized_expected[region_slice], atol=atol,
            err_msg=f"{name} in {region_name}, {is_cpu=} {is_cuda=}, {xla_extension_version=}\n{inexact_samples}")

      if kind == 'failure' and region_name in regions_with_inaccuracies:
        try:
          with self.assertRaises(AssertionError, msg=f"{name} in {region_name}, {is_cpu=} {is_cuda=}, {xla_extension_version=}"):
            with jtu.ignore_warning(category=RuntimeWarning, message="overflow encountered in.*"):
              self.assertAllClose(normalized_result[region_slice], normalized_expected[region_slice])
        except AssertionError as msg:
          if str(msg).startswith('AssertionError not raised'):
            unexpected_success_regions.append(region_name)
          else:
            raise  # something else is wrong..

    if kind == 'success' and regions_with_inaccuracies:
      reason = "xfail: problematic regions: " + ", ".join(regions_with_inaccuracies)
      raise unittest.SkipTest(reason)

    if kind == 'failure':
      if not regions_with_inaccuracies:
        raise unittest.SkipTest("no problematic regions")
      elif unexpected_success_regions:
        # This skip ought to be effective only when fixing functions
        # on problematic regions in XLA that should follow up a JAX PR
        # that enables testing the functions on these regions for
        # success.
        raise unittest.SkipTest(
          f"detected success in regions {', '.join(unexpected_success_regions)}, please update regions_with_inaccuracies!"
        )

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
