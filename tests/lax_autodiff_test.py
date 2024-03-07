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


import collections
from functools import partial
import itertools
import math
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import dtypes
from jax import lax
from jax._src import test_util as jtu
from jax._src.util import NumpyComplexWarning
from jax.test_util import check_grads

jax.config.parse_flags_with_absl()


compatible_shapes = [[(3,)],
                     [(), (3, 4), (3, 1), (1, 4)],
                     [(2, 3, 4), (2, 1, 4)]]


GradTestSpec = collections.namedtuple(
    "GradTestSpec",
    ["op", "nargs", "order", "rng_factory", "dtypes", "name", "tol"])
def grad_test_spec(op, nargs, order, rng_factory, dtypes, name=None, tol=None):
  return GradTestSpec(
      op, nargs, order, rng_factory, dtypes, name or op.__name__, tol)

float_dtypes = jtu.dtypes.all_floating
inexact_dtypes = jtu.dtypes.all_inexact
grad_float_dtypes = jtu.dtypes.floating
grad_complex_dtypes = jtu.dtypes.complex
grad_inexact_dtypes = jtu.dtypes.inexact

LAX_GRAD_OPS = [
    grad_test_spec(lax.neg, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.floor, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=0.1, high=0.4),
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.ceil, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=0.1, high=0.4),
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.round, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=0.1, high=0.4),
                   dtypes=grad_float_dtypes),

    grad_test_spec(lax.exp, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.exp2, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.expm1, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.log, nargs=1, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.log1p, nargs=1, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.sinh, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_float_dtypes + [np.complex64], tol=1e-5),
    grad_test_spec(lax.cosh, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes, tol=1e-5),
    grad_test_spec(lax.tanh, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes, tol=2e-4),
    grad_test_spec(lax.sin, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes, tol={np.float32: 5e-1}),
    grad_test_spec(lax.cos, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.tan, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-1.3, high=1.3),
                   dtypes=grad_inexact_dtypes, tol=1e-3),
    grad_test_spec(lax.asin, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-1.3, high=1.3),
                   dtypes=grad_float_dtypes, tol=1e-3),
    grad_test_spec(lax.acos, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, low=-1.3, high=1.3),
                   dtypes=grad_float_dtypes, tol=2e-2),
    # TODO(proteneer): atan2 input is already a representation of a
    # complex number. Need to think harder about what this even means
    # if each input itself is a complex number.
    grad_test_spec(lax.atan2, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_float_dtypes),

    grad_test_spec(lax.erf, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.erfc, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.erf_inv, nargs=1, order=2, rng_factory=jtu.rand_small,
                   dtypes=grad_float_dtypes),
    # grad_test_spec(lax.lgamma, nargs=1, order=2, rng_factory=jtu.rand_small,
    #                dtypes=grad_float_dtypes),  # TODO(mattjj): enable
    grad_test_spec(lax.bessel_i0e, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.bessel_i1e, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_float_dtypes),

    grad_test_spec(lax.real, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_complex_dtypes),
    grad_test_spec(lax.imag, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_complex_dtypes),
    grad_test_spec(lax.complex, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.conj, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.abs, nargs=1, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.pow, nargs=2, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_inexact_dtypes, tol={np.float32: 3e-1}),
    grad_test_spec(lax.sqrt, nargs=1, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.sqrt, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_complex_dtypes),
    grad_test_spec(lax.rsqrt, nargs=1, order=2, rng_factory=jtu.rand_positive,
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.rsqrt, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_complex_dtypes, tol={np.float64: 2e-3}),
    grad_test_spec(lax.cbrt, nargs=1, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_float_dtypes, tol={np.float64: 5e-3}),
    grad_test_spec(lax.logistic, nargs=1, order=2,
                   rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),

    grad_test_spec(lax.add, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.sub, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.mul, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_inexact_dtypes),
    grad_test_spec(lax.div, nargs=2, order=1, rng_factory=jtu.rand_not_small,
                   dtypes=grad_inexact_dtypes),

    grad_test_spec(lax.max, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_float_dtypes),
    grad_test_spec(lax.min, nargs=2, order=2, rng_factory=jtu.rand_default,
                   dtypes=grad_float_dtypes),
    # TODO(mattjj): make some-equal checks more robust, enable second-order
    # grad_test_spec(lax.max, nargs=2, order=1, rng_factory=jtu.rand_some_equal,
    #                dtypes=grad_float_dtypes, name="MaxSomeEqual"),
    # grad_test_spec(lax.min, nargs=2, order=1, rng_factory=jtu.rand_some_equal,
    #                dtypes=grad_float_dtypes, name="MinSomeEqual"),
]

GradSpecialValuesTestSpec = collections.namedtuple(
    "GradSpecialValuesTestSpec", ["op", "values", "tol"])
def grad_special_values_test_spec(op, values, tol=None):
  return GradSpecialValuesTestSpec(op, values, tol)

LAX_GRAD_SPECIAL_VALUE_TESTS = [
    grad_special_values_test_spec(lax.sinh, [0.]),
    grad_special_values_test_spec(lax.cosh, [0.]),
    grad_special_values_test_spec(lax.tanh, [0., 1000.], tol=5e-3),
    grad_special_values_test_spec(lax.sin, [0., np.pi, np.pi/2., np.pi/4.]),
    grad_special_values_test_spec(lax.cos, [0., np.pi, np.pi/2., np.pi/4.]),
    grad_special_values_test_spec(lax.tan, [0.]),
    grad_special_values_test_spec(lax.asin, [0.]),
    grad_special_values_test_spec(lax.acos, [0.]),
    grad_special_values_test_spec(lax.atan, [0., 1000.]),
    grad_special_values_test_spec(lax.erf, [0., 10.]),
    grad_special_values_test_spec(lax.erfc, [0., 10.]),
]


def check_grads_bilinear(f, args, order,
                         modes=("fwd", "rev"), atol=None, rtol=None):
  # Can use large eps to make up for numerical inaccuracies since the op is
  # bilinear (relying on the fact that we only check one arg at a time)
  lhs, rhs = args
  check_grads(lambda lhs: f(lhs, rhs), (lhs,), order,
              modes=modes, atol=atol, rtol=rtol, eps=1.)
  check_grads(lambda rhs: f(lhs, rhs), (rhs,), order,
              modes=modes, atol=atol, rtol=rtol, eps=1.)


class LaxAutodiffTest(jtu.JaxTestCase):

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(op=rec.op, rng_factory=rec.rng_factory, order=rec.order, tol=rec.tol)],
      shapes=[
        shapes for shape_group in compatible_shapes
        for shapes in itertools.combinations_with_replacement(shape_group, rec.nargs)
      ],
      dtype=rec.dtypes,
    )
    for rec in LAX_GRAD_OPS
  ))
  def testOpGrad(self, op, rng_factory, shapes, dtype, order, tol):
    rng = rng_factory(self.rng())
    if jtu.test_device_matches(["tpu"]):
      if op is lax.pow:
        raise SkipTest("pow grad imprecise on tpu")
      if op is lax.cos:
        order = 1  # 2nd-order gradient is imprecise on TPU.
      if op is lax.log:
        order = 1  # 2nd-order gradient is imprecise on TPU.

    tol = jtu.join_tolerance(1.5e-1, tol) if jtu.num_float_bits(dtype) == 32 else tol
    args = tuple(rng(shape, dtype) for shape in shapes)
    check_grads(op, args, order, ["fwd", "rev"], tol, tol)

  @parameterized.parameters(itertools.chain.from_iterable(
    jtu.sample_product_testcases(
      [dict(op=rec.op, tol=rec.tol)],
      special_value=rec.values,
    )
    for rec in LAX_GRAD_SPECIAL_VALUE_TESTS
  ))
  def testOpGradSpecialValue(self, op, special_value, tol):
    if op in (lax.sinh, lax.cosh) and jtu.test_device_matches(["tpu"]):
      tol = {np.float32: 1e-2}
    check_grads(op, (special_value,), 2, ["fwd", "rev"], rtol=tol, atol=tol)

  @jtu.sample_product(
    from_dtype=inexact_dtypes,
    to_dtype=inexact_dtypes,
  )
  def testConvertElementTypeGrad(self, from_dtype, to_dtype):
    rng = jtu.rand_default(self.rng())
    tol = max(jtu.tolerance(to_dtype, jtu.default_gradient_tolerance),
              jtu.tolerance(from_dtype, jtu.default_gradient_tolerance))
    args = (rng((2, 3), from_dtype),)
    convert_element_type = lambda x: lax.convert_element_type(x, to_dtype)
    convert_element_type = jtu.ignore_warning(category=NumpyComplexWarning)(
      convert_element_type)
    check_grads(convert_element_type, args, 2, ["fwd", "rev"], tol, tol, eps=1.)

  @jtu.sample_product(
    shape=[(), (2, 3)],
    dtype=grad_float_dtypes,
  )
  def testClampGrad(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    operand = rng(shape, dtype)
    low = operand - dtype(10)
    high = operand + dtype(10)
    # Avoids points near the boundary where the gradient may be inaccurate.
    check_grads(lax.clamp, (operand, low, high), 2, ["fwd", "rev"], eps=1e-2)
    check_grads(lax.clamp, (low, operand, high), 2, ["fwd", "rev"], eps=1e-2)
    check_grads(lax.clamp, (low, high, operand), 2, ["fwd", "rev"], eps=1e-2)

  @jtu.sample_product(
    [dict(base_shape=base_shape, dim=dim)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for dim in range(len(base_shape))
    ],
    num_arrs=[3],
    dtype=float_dtypes,
  )
  def testConcatenateGrad(self, dim, base_shape, dtype, num_arrs):
    rng = jtu.rand_default(self.rng())
    shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
    operands = tuple(rng(shape, dtype) for shape in shapes)
    concatenate = lambda *args: lax.concatenate(args, dim)
    check_grads(concatenate, operands, 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, strides=strides)
       for lhs_shape, rhs_shape, all_strides in itertools.chain(
           [((b, i, 3, 4), (j, i, 1, 2), [(1, 1), (1, 2), (2, 1)])
            for b, i, j in itertools.product([2, 3], repeat=3)],
           [((4, 2, 1), (3, 2, 1), [(1,)])])
       for strides in all_strides
    ],
    dtype=float_dtypes,
    padding=["VALID", "SAME"],
  )
  def testConvGrad(self, lhs_shape, rhs_shape, dtype, strides, padding):
    rng = jtu.rand_small(self.rng())
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    conv = partial(lax.conv, window_strides=strides, padding=padding,
                   precision=lax.Precision.HIGHEST)
    check_grads_bilinear(conv, (lhs, rhs), order=2, modes=["fwd", "rev"],
                         atol=1e-2, rtol=1e-2)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, strides=strides,
          padding=padding, lhs_dil=lhs_dil, rhs_dil=rhs_dil)
       for lhs_shape, rhs_shape, all_strides, all_pads, lhs_dils, rhs_dils in
        itertools.chain(
           [((b, i, 3, 4), (j, i, 1, 2), [(1, 1), (1, 2), (2, 1)],
             [((0, 0), (0, 0)), ((-1, 0), (0, -1)), ((1, 0), (0, 1))],
             [(1, 1), (2, 1)], [(1, 1)])
            for b, i, j in itertools.product([2, 3], repeat=3)],
           [((4, 2, 1), (3, 2, 1), [(1,)], [((1, 1),), ((0, 0),)],
             [(1,), (2,)], [(1,), (2,)])])
       for strides in all_strides
       for rhs_dil in rhs_dils
       for lhs_dil in lhs_dils
       for padding in all_pads
    ],
    dtype=float_dtypes,
  )
  def testConvWithGeneralPaddingGrad(self, lhs_shape, rhs_shape, dtype, strides,
                                     padding, lhs_dil, rhs_dil):
    rng = jtu.rand_small(self.rng())
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    conv = partial(lax.conv_with_general_padding, window_strides=strides,
                   padding=padding, lhs_dilation=lhs_dil, rhs_dilation=rhs_dil,
                   precision=lax.Precision.HIGHEST)
    check_grads_bilinear(conv, (lhs, rhs), order=2, modes=["fwd", "rev"],
                         atol=1e-2, rtol=1e-2)

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, strides=strides,
          padding=padding, lhs_dil=lhs_dil, rhs_dil=rhs_dil,
          feature_group_count=feature_group_count,
          batch_group_count=batch_group_count)
      for batch_group_count, feature_group_count in ([(1, 1), (2, 1), (1, 2)])
      for lhs_shapes, rhs_shape, all_strides, lhs_dils, rhs_dils in [
          ([(b * batch_group_count, i * feature_group_count, 6, 7),
            (b * batch_group_count, i * feature_group_count, 0, 4)],  # lhs_shape
           (j * batch_group_count * feature_group_count, i, 1, 2),  # rhs_shape
           [(1, 1), (1, 2), (2, 1)],  # strides
           [(1, 1), (2, 1)],  # lhs_dils
           [(1, 1), (2, 2)])  # rhs_dils
          for b, i, j in itertools.product([1, 2], repeat=3)]
      for lhs_shape in lhs_shapes
      for strides in all_strides
      for rhs_dil in rhs_dils
      for lhs_dil in lhs_dils
      for padding in ([((0, 0), (0, 0)), ((1, 0), (0, 1))] +
                     ([((0, -1), (0, 0))] if lhs_shape[2] != 0 else []))
    ],
    [dict(dimension_numbers=dim_nums, perms=perms)
      for dim_nums, perms in [
        (("NCHW", "OIHW", "NCHW"), ([0, 1, 2, 3], [0, 1, 2, 3])),
        (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0])),
        (("NHWC", "OIHW", "NCHW"), ([0, 2, 3, 1], [0, 1, 2, 3]))]
    ],
    dtype=grad_inexact_dtypes,
  )
  def testConvGeneralDilatedGrad(self, lhs_shape, rhs_shape, dtype, strides,
                                 padding, lhs_dil, rhs_dil, dimension_numbers,
                                 perms, feature_group_count, batch_group_count):
    if dtype == np.float16:
      raise SkipTest("float16 numerical issues")  # TODO(mattjj): resolve

    rng = jtu.rand_default(self.rng())
    tol = {dtypes.bfloat16: 1e-0, np.float16: 5e-1, np.float32: 1e-3}

    # permute shapes to match dim_spec, scale by feature_group_count
    lhs_perm, rhs_perm = perms
    lhs_shape = list(np.take(lhs_shape, lhs_perm))
    rhs_shape = list(np.take(rhs_shape, rhs_perm))

    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    conv = partial(lax.conv_general_dilated, window_strides=strides,
                   padding=padding, lhs_dilation=lhs_dil, rhs_dilation=rhs_dil,
                   dimension_numbers=dimension_numbers,
                   feature_group_count=feature_group_count,
                   batch_group_count=batch_group_count,
                   precision=lax.Precision.HIGHEST)
    check_grads_bilinear(conv, (lhs, rhs), order=2, modes=["fwd", "rev"],
                         atol=tol, rtol=tol)

  @jtu.sample_product(
    lhs_shape=[(2,), (3, 2)],
    rhs_shape=[(2,), (2, 4)],
    dtype=float_dtypes,
  )
  def testDotGrad(self, lhs_shape, rhs_shape, dtype):
    rng = jtu.rand_default(self.rng())
    tol = {np.float16: 1e-1, np.float32: 1e-4}
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    dot = partial(lax.dot, precision=lax.Precision.HIGHEST)
    check_grads_bilinear(dot, (lhs, rhs), order=2, modes=["fwd", "rev"],
                         atol=tol, rtol=tol)
    # check that precision config is preserved
    result, pullback = jax.vjp(dot, lhs, rhs)
    s = str(jax.make_jaxpr(pullback)(result))
    assert "Precision.HIGHEST" in s

  @jtu.sample_product(
    [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape,
          dimension_numbers=dimension_numbers)
      for lhs_shape, rhs_shape, dimension_numbers in [
          ((3, 2), (2, 4), (([1], [0]), ([], []))),
          ((3, 5), (2, 5), (([1], [1]), ([], []))),
          ((5, 3), (5, 2), (([0], [0]), ([], []))),
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
          ((3, 5, 2), (2, 4, 5), (([2], [0]), ([1], [2]))),
          ((7, 3, 5, 2), (2, 2, 4, 5), (([3], [0]), ([2], [3]))),
      ]
    ],
    dtype=float_dtypes,
  )
  def testDotGeneralContractAndBatchGrads(self, lhs_shape, rhs_shape, dtype,
                                          dimension_numbers):
    rng = jtu.rand_small(self.rng())
    lhs = rng(lhs_shape, dtype)
    rhs = rng(rhs_shape, dtype)
    dot_general = partial(lax.dot_general, dimension_numbers=dimension_numbers,
                          precision=lax.Precision.HIGHEST)
    atol = {np.float16: 5E-2} if jtu.test_device_matches(['tpu']) else None
    check_grads_bilinear(dot_general, (lhs, rhs), order=2,
                         modes=["fwd", "rev"], atol=atol)
    # check that precision config is preserved
    result, pullback = jax.vjp(dot_general, lhs, rhs)
    s = str(jax.make_jaxpr(pullback)(result))
    assert "Precision.HIGHEST" in s

  def testDotPreferredElementType(self):
    # https://github.com/google/jax/issues/10818
    x = jax.numpy.ones((), jax.numpy.float16)
    def f(x):
      return jax.lax.dot_general(x, x, (((), ()), ((), ())),
                                 preferred_element_type=jax.numpy.float32)
    jax.jacrev(f)(x)  # don't crash!

  @jtu.sample_product(
    shape=[(), (2, 3)],
    dtype=float_dtypes,
    broadcast_sizes=[(), (2,), (1, 2)],
  )
  def testBroadcastGrad(self, shape, dtype, broadcast_sizes):
    rng = jtu.rand_default(self.rng())
    args = (rng(shape, dtype),)
    broadcast = lambda x: lax.broadcast(x, broadcast_sizes)
    check_grads(broadcast, args, 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(inshape=inshape, outshape=outshape, dimensions=broadcast_dimensions)
      for inshape, outshape, broadcast_dimensions in [
          ([2], [2, 2], [0]),
          ([2], [2, 2], [1]),
          ([2], [2, 3], [0]),
          ([], [2, 3], []),
      ]
    ],
    dtype=float_dtypes,
  )
  def testBroadcastInDimGrad(self, inshape, dtype, outshape, dimensions):
    rng = jtu.rand_default(self.rng())
    operand = rng(inshape, dtype)
    broadcast_in_dim = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    check_grads(broadcast_in_dim, (operand,), 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, out_shape=out_shape, permutation=permutation)
      for arg_shape, out_shape, permutation in [
          [(3, 4), (12,), None],
          [(2, 1, 4), (8,), None],
          [(2, 2, 4), (2, 8), None],
          [(3, 4), (12,), (0, 1)],
          [(3, 4), (12,), (1, 0)],
          [(2, 1, 4), (8,), (0, 2, 1)],
          [(2, 1, 4), (8,), (2, 0, 1)],
          [(2, 2, 4), (2, 8), (0, 2, 1)],
          [(2, 2, 4), (2, 8), (2, 0, 1)],
      ]
    ],
    dtype=float_dtypes,
  )
  def testReshapeGrad(self, arg_shape, out_shape, permutation, dtype):
    rng = jtu.rand_default(self.rng())
    operand = rng(arg_shape, dtype)
    reshape = lambda x: lax.reshape(x, out_shape, permutation)
    check_grads(reshape, (operand,), 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(shape=shape, pads=pads)
      for shape, paddings in [
        [(), [()]],
        ((2, 3), [[(1, 2, 1), (0, 1, 0)], [(-1, 0, 0), (-1, 0, 2)]]),
      ]
      for pads in paddings
    ],
    dtype=float_dtypes,
  )
  def testPadGrad(self, shape, dtype, pads):
    rng = jtu.rand_small(self.rng())
    operand = rng(shape, dtype)
    pad = lambda operand: lax.pad(operand, np.array(0, dtype), pads)
    check_grads(pad, (operand,), 2, ["fwd", "rev"], eps=1.)

    operand = rng(shape, dtype)
    padding_value = np.array(0., dtype)
    pad = lambda operand, padding_value: lax.pad(operand, padding_value, pads)
    check_grads(pad, (operand, padding_value), 2, ["fwd", "rev"], eps=1.)

  def testReverseGrad(self):
    rev = lambda operand: lax.rev(operand, dimensions)

    dimensions = [0]
    check_grads(rev, (np.array([3., 2., 1.]),), 2)

    dimensions = [0, 1]
    check_grads(rev, (np.array([[6., 5., 4.], [3., 2., 1.]]),), 2,
                rtol={np.float32: 3e-3})

  def testPowSecondDerivative(self):
    # https://github.com/google/jax/issues/12033
    x, y = 4.0, 0.0
    expected = ((0.0, 1/x), (1/x, np.log(x) ** 2))

    with self.subTest("jacfwd"):
      result_fwd = jax.jacfwd(jax.jacfwd(lax.pow, (0, 1)), (0, 1))(x, y)
      self.assertAllClose(result_fwd, expected)

    with self.subTest("jacrev"):
      result_rev = jax.jacrev(jax.jacrev(lax.pow, (0, 1)), (0, 1))(x, y)
      self.assertAllClose(result_rev, expected)

    with self.subTest("zero to the zero"):
      result = jax.grad(lax.pow)(0.0, 0.0)
      # TODO(jakevdp) special-case zero in a way that doesn't break other cases
      # See https://github.com/google/jax/pull/12041#issuecomment-1222766191
      # self.assertEqual(result, 0.0)
      self.assertAllClose(result, np.nan)

  def testPowIntPowerAtZero(self):
    # https://github.com/google/jax/issues/14397
    ans = jax.grad(jax.jit(lambda x, n: x ** n))(0., 0)
    self.assertAllClose(ans, 0., check_dtypes=False)

  @jax.numpy_dtype_promotion('standard')  # This test explicitly exercises mixed type promotion
  def testPowIntPowerAtZero2(self):
    # https://github.com/google/jax/issues/17995
    a = lambda z: jax.numpy.sum(z**jax.numpy.arange(0, 2, dtype=int))
    b = lambda z: jax.numpy.sum(z**jax.numpy.arange(0, 2, dtype=float))
    c = lambda z: 1 + z
    d = lambda z: z ** 0 + z
    e = lambda z: z ** 0. + z

    self.assertAllClose(jax.grad(a)(3.14), 1., check_dtypes=False)
    self.assertAllClose(jax.grad(b)(3.14), 1., check_dtypes=False)
    self.assertAllClose(jax.grad(c)(3.14), 1., check_dtypes=False)
    self.assertAllClose(jax.grad(d)(3.14), 1., check_dtypes=False)
    self.assertAllClose(jax.grad(e)(3.14), 1., check_dtypes=False)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, pred_shape=pred_shape)
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
    ],
    dtype=float_dtypes,
  )
  def testSelectGrad(self, pred_shape, arg_shape, dtype):
    rng = jtu.rand_default(self.rng())
    pred = rng(pred_shape, np.bool_)
    on_true = rng(arg_shape, dtype)
    on_false = rng(arg_shape, dtype)
    select = lambda on_true, on_false: lax.select(pred, on_true, on_false)
    check_grads(select, (on_true, on_false), 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(shape=shape, starts=start_indices, limits=limit_indices,
          strides=strides)
      for shape, start_indices, limit_indices, strides in [
        [(3,), (1,), (2,), None],
        [(7,), (4,), (7,), None],
        [(5,), (1,), (5,), (2,)],
        [(8,), (1,), (6,), (2,)],
        [(5, 3), (1, 1), (3, 2), None],
        [(5, 3), (1, 1), (3, 1), None],
        [(7, 5, 3), (4, 0, 1), (7, 1, 3), None],
        [(5, 3), (1, 1), (2, 1), (1, 1)],
        [(5, 3), (1, 1), (5, 3), (2, 1)],
        [(3, 3, 5), (0, 2, 0), (3, 2, 5), (1, 2, 1)]
      ]
    ],
    dtype=float_dtypes,
  )
  def testSliceGrad(self, shape, dtype, starts, limits, strides):
    rng = jtu.rand_default(self.rng())
    operand = rng(shape, dtype)
    slice = lambda x: lax.slice(x, starts, limits, strides)
    check_grads(slice, (operand,), 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(shape=shape, start_indices=start_indices, size_indices=size_indices)
      for shape, start_indices, size_indices in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
    ],
    dtype=float_dtypes,
  )
  def testDynamicSliceGrad(self, shape, dtype, start_indices, size_indices):
    rng = jtu.rand_default(self.rng())
    operand = rng(shape, dtype)
    dynamic_slice = lambda x: lax.dynamic_slice(x, start_indices, size_indices)
    check_grads(dynamic_slice, (operand,), 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(shape=shape, start_indices=start_indices, update_shape=update_shape)
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
      ]
    ],
    dtype=float_dtypes,
  )
  def testDynamicUpdateSliceGrad(self, shape, dtype, start_indices,
                                 update_shape):
    rng = jtu.rand_default(self.rng())
    operand = rng(shape, dtype)
    update = rng(update_shape, dtype)
    start_indices = np.array(start_indices)

    dus = lambda x, y: lax.dynamic_update_slice(x, y, start_indices)
    check_grads(dus, (operand, update), 2, ["fwd", "rev"], eps=1.)

    dus = lambda x: lax.dynamic_update_slice(x, update, start_indices)
    check_grads(dus, (operand,), 2, ["fwd", "rev"], eps=1.)

    dus = lambda y: lax.dynamic_update_slice(operand, y, start_indices)
    check_grads(dus, (update,), 2, ["fwd", "rev"], eps=1.)

  def testDynamicSliceValueAndGrad(self):
    # Regression test for https://github.com/google/jax/issues/10984
    # Issue arose due to an out-of-range negative index.
    rng = jtu.rand_default(self.rng())
    shape = (5, 5)
    axis = 0
    index = -(shape[axis] + 3)
    def f(x):
      return lax.dynamic_index_in_dim(x, index, axis).sum()
    x = rng(shape, np.float32)

    result1 = f(x)
    result2, _ = jax.value_and_grad(f, 0)(x)
    self.assertAllClose(result1, result2)

  def testDynamicUpdateSliceValueAndGrad(self):
    # Regression test for https://github.com/google/jax/issues/10984
    # Issue arose due to an out-of-range negative index.
    rng = jtu.rand_default(self.rng())
    shape = (5, 5)
    axis = 0
    index = -(shape[axis] + 3)
    def f(x, y):
      return lax.dynamic_update_index_in_dim(x, y, index, axis).sum()
    x = rng(shape, np.float32)
    y = rng([1 for s in shape], np.float32)

    result1 = f(x, y)
    result2, _ = jax.value_and_grad(f, 0)(x, y)
    self.assertAllClose(result1, result2)

  @jtu.sample_product(
    [dict(shape=shape, perm=perm)
      for shape, perm in [
        [(3, 4), (1, 0)],
        [(3, 4), (0, 1)],
        [(3, 4, 5), (2, 1, 0)],
        [(3, 4, 5), (1, 0, 2)],
      ]
    ],
    dtype=float_dtypes,
  )
  def testTransposeGrad(self, shape, dtype, perm):
    rng = jtu.rand_default(self.rng())
    operand = rng(shape, dtype)
    transpose = lambda x: lax.transpose(x, perm)
    check_grads(transpose, (operand,), 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(init_val=init_val, op=op, dtype=dtype, rng_factory=rng_factory)
      for init_val, op, dtypes, rng_factory in [
          (0, lax.add, float_dtypes + jtu.dtypes.complex, jtu.rand_default),
          (-np.inf, lax.max, grad_inexact_dtypes, jtu.rand_unique_int),
          (np.inf, lax.min, grad_inexact_dtypes, jtu.rand_unique_int),
          (1, lax.mul, grad_float_dtypes, partial(jtu.rand_default, scale=1)),
      ]
      for dtype in dtypes
    ],
    [dict(shape=shape, dims=dims)
      for shape, dims in [
          [(), ()],
          [(3, 4, 5), ()],
          [(3, 4, 5), (0,)],
          [(3, 4, 5), (1, 2)],
          [(3, 4, 5), (0, 2)],
          [(3, 4, 5), (0, 1, 2)],
          [(3, 1), (1,)],
          [(3, 0, 5), (1,)],
      ]
    ],
  )
  def testReduceGrad(self, op, init_val, shape, dtype, dims, rng_factory):
    rng = rng_factory(self.rng())
    if jtu.test_device_matches(["tpu"]) and op is lax.mul:
      raise SkipTest("unimplemented case")
    tol = {dtypes.bfloat16: 2e-1, np.float16: 1e-1, np.float32: 1e-1,
           np.float64: 1e-3, np.complex64: 1e-1}
    operand = rng(shape, dtype)
    init_val = np.asarray(init_val, dtype=dtype)
    reduce = lambda operand: lax.reduce(operand, init_val, op, dims)
    eps = (1.0 if dtypes.finfo(dtype).bits == 16 and op is lax.add else
           1e-1 if dtype == dtypes.bfloat16 else
           1e-2 if dtypes.finfo(dtype).bits == 32 else None)
    if op not in (lax.max, lax.min) or all(d > 0 for d in shape):
      check_grads(reduce, (operand,), 2, ["fwd", "rev"], tol, tol, eps)

  @jtu.sample_product(
    [dict(shape=shape, dims=dims)
      for shape, dims in [
          [(3, 4, 5), ()],
          [(3, 4, 5), (0,)],
          [(3, 4, 5), (1, 2)],
          [(3, 4, 5), (0, 2)],
          [(3, 4, 5), (0, 1, 2)],
          [(3, 1), (1,)],
          [(3, 0, 5), (1,)],
      ]
    ],
    dtype=grad_float_dtypes,
  )
  def testReducePairGrad(self, shape, dtype, dims):
    rng = jtu.rand_default(self.rng(), scale=1)
    tol = {np.float32: 1e-2, np.float64: 1e-4}
    operands = (rng(shape, dtype), rng(shape, dtype))
    init_vals = (np.array(0, dtype), np.array(1, dtype))
    def op(xs, ys):
      return (xs[0] + ys[0], xs[1] * ys[1])
    reduce = lambda xs, ys: lax.reduce((xs, ys), init_vals, op, dims)
    check_grads(reduce, operands, 2, ["fwd", "rev"], tol, tol)

  @jtu.sample_product(
    [dict(init_val=init_val, op=op, dtype=dtype, rng_factory=rng_factory,
          shape=shape, dims=dims, strides=strides, padding=padding,
          base_dilation=base_dilation, window_dilation=window_dilation)
      for init_val, op, dtypes, rng_factory in [
          (0, lax.add, grad_float_dtypes, jtu.rand_small),
          (-np.inf, lax.max, grad_float_dtypes, jtu.rand_unique_int),
          (np.inf, lax.min, grad_float_dtypes, jtu.rand_unique_int),
      ]
      for dtype in dtypes
      for shape, dims, strides, padding, base_dilation, window_dilation in (
        itertools.chain(
          itertools.product(
            [(4, 6)],
            [(2, 1), (1, 2)],
            [(1, 1), (2, 1), (1, 2)],
            ["VALID", "SAME", [(0, 3), (1, 2)]],
            [(1, 1)] + ([(2, 3)]),
            [(1, 1)] + ([(1, 2)] if op is lax.add else [])),
          itertools.product(
            [(3, 2, 4, 6)],
            [(1, 1, 2, 1), (2, 1, 2, 1)],
            [(1, 2, 2, 1), (1, 1, 1, 1)],
            ["VALID", "SAME", [(0, 1), (1, 0), (2, 3), (0, 2)]],
            [(1, 1, 1, 1)] + ([(2, 1, 3, 2)]),
            [(1, 1, 1, 1)] + ([(1, 2, 2, 1)] if op is lax.add else []))))
    ],
  )
  @jtu.ignore_warning(category=UserWarning,
                      message="Using reduced precision for gradient.*")
  def testReduceWindowGrad(
      self, op, init_val, dtype, shape, dims, strides,
      padding, base_dilation, window_dilation, rng_factory):
    rng = rng_factory(self.rng())
    init_val = np.asarray(init_val, dtype=dtype)

    gradient_order = 3
    # We need this conditional and the corresponding loop logic to be in the
    # test method, rather than at the parameterized test level, because it
    # depends on FLAGS for the device under test.
    # TODO(b/31565929): enable when fixed.
    if jtu.test_device_matches(["tpu"]) and op is not lax.add:
      if (len(shape) != 4 or dims != (1, 1, 2, 1)
          or not isinstance(padding, str)):
        raise SkipTest("Only R4 SelectAndScatter implemented on TPU")

    def fun(operand):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding,
                               base_dilation, window_dilation)

    operand = rng(shape, dtype)
    if op is lax.add:
      eps = 1.
      tol = None
    else:
      # this test can fail if there are duplicates in operand
      self.assertEqual(np.unique(operand).size, operand.size,
                       msg="test requires operand elements to be unique.")
      eps = 1e-2
      tol = {np.float16: 1e-1, np.float32: 6e-2, np.float64: 6e-2}
    check_grads(fun, (operand,), gradient_order, ["fwd", "rev"], tol, tol,
                eps)

  @jtu.sample_product(
    [dict(op=op, dtype=dtype)
      for op, types in [
          (lax.cumsum, [np.float32, np.float64]),
          (lax.cumprod, [np.float32, np.float64]),
      ]
      for dtype in types
    ],
    [dict(shape=shape, axis=axis)
      for shape in [[10], [3, 4, 5]]
      for axis in range(len(shape))
    ],
    reverse=[False, True],
  )
  def testCumulativeReduceGrad(self, op, shape, dtype, axis, reverse):
    rng_factory = (jtu.rand_default if dtypes.issubdtype(dtype, np.integer)
                   else jtu.rand_small)
    rng = rng_factory(self.rng())
    check_grads(partial(op, axis=axis, reverse=reverse), (rng(shape, dtype),),
                order=2)


  # TODO(b/205052657): enable more tests when supported
  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in [(5,), (5, 7)]
      for axis in [len(shape) - 1]
    ],
    dtype=[np.float32],
    is_stable=[False, True],
  )
  def testSortGrad(self, shape, dtype, axis, is_stable):
    rng = jtu.rand_unique_int(self.rng())
    operand = rng(shape, dtype)
    sort = lambda x: lax.sort(x, dimension=axis, is_stable=is_stable)
    check_grads(sort, (operand,), 2, ["fwd", "rev"], eps=1e-2)

  # TODO(b/205052657): enable more tests when supported
  @jtu.sample_product(
    [dict(shape=shape, axis=axis)
      for shape in [(3,), (5, 3)]
      for axis in [len(shape) - 1]
    ],
    key_dtype=[np.float32],
    val_dtype=[np.float32],
    is_stable=[False, True],
  )
  def testSortKeyValGrad(self, shape, key_dtype, val_dtype, axis, is_stable):
    rng = jtu.rand_default(self.rng())
    # This test relies on the property that wherever keys are tied, values are
    # too, since we don't guarantee the same ordering of values with equal keys.
    # To avoid that case, we generate unique keys (globally in the key array).
    def args_maker():
      flat_keys = np.arange(math.prod(shape), dtype=key_dtype)
      keys = self.rng().permutation(flat_keys).reshape(shape)
      values = rng(shape, val_dtype)
      return keys, values
    keys, values = args_maker()

    fun = lambda keys, values: lax.sort_key_val(keys, values, axis, is_stable)
    check_grads(fun, (keys, values), 2, ["fwd", "rev"], 1e-2, 1e-2, 1e-2)

  @jtu.sample_product(
    dtype=[np.float32,],
    shape=[(4,), (5, 5), (2, 1, 4)],
    k=[1, 3],
  )
  def testTopKGrad(self, shape, dtype, k):
    flat_values = np.arange(math.prod(shape), dtype=dtype)
    values = self.rng().permutation(flat_values).reshape(shape)
    fun = lambda vs: lax.top_k(vs, k=k)[0]
    check_grads(fun, (values,), 2, ["fwd", "rev"], eps=1e-2)

  @jtu.sample_product(
    [dict(shape=shape, idxs=idxs, axes=axes)
      for shape, idxs, axes in [
          [(3, 4, 5), (np.array([0, 2, 1]),), (0,)],
          [(3, 4, 5), (np.array([-1, -2]),), (0,)],
          [(3, 4, 5), (np.array([0, 2]), np.array([1, 3])), (0, 1)],
          [(3, 4, 5), (np.array([0, 2]), np.array([1, 3])), (0, 2)],
      ]
    ],
    dtype=float_dtypes,
  )
  @jax.numpy_rank_promotion('allow')  # Test explicitly exercises implicit rank promotion.
  def testIndexTakeGrad(self, shape, dtype, idxs, axes):
    rng = jtu.rand_default(self.rng())
    src = rng(shape, dtype)
    index_take = lambda src: lax.index_take(src, idxs, axes)
    check_grads(index_take, (src,), 2, ["fwd", "rev"], eps=1.)

  @jtu.sample_product(
    [dict(shape=shape, idxs_shape=idxs.shape, idxs_dtype=idxs.dtype,
          dnums=dnums, slice_sizes=slice_sizes, max_idx=max_idx)
      for shape, idxs, dnums, slice_sizes, max_idx in [
          ((5,), np.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,), 5),
          ((10,), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,), 9),
          ((10, 5,), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3), 3),
      ]
    ],
    dtype=grad_float_dtypes,
    mode=["clip", "fill", "promise_in_bounds"],
    iteration=range(5),
  )
  def testGatherGrad(self, shape, dtype, idxs_shape, idxs_dtype, dnums,
                     slice_sizes, mode, max_idx, iteration):
    rng = jtu.rand_default(self.rng())
    if mode == "promise_in_bounds":
      rng_idx = jtu.rand_int(self.rng(), high=max_idx)
    else:
      # Only test out-of-bounds indices if using a mode that guarantees correct
      # gradients for out-of-bounds indices.
      rng_idx = jtu.rand_int(self.rng(), low=-max_idx, high=2 * max_idx)
    idxs = rng_idx(idxs_shape, idxs_dtype)
    # Use an arbitrary finite fill_value, since NaNs won't work in a numerical
    # gradient test.
    gather = lambda x: lax.gather(x, idxs, dimension_numbers=dnums,
                                  slice_sizes=slice_sizes, mode=mode,
                                  fill_value=-1)
    x = rng(shape, dtype)
    check_grads(gather, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2, 1.)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs_shape=idxs.shape, idxs_dtype=idxs.dtype,
          dnums=dnums, update_shape=update_shape, max_idx=max_idx)
      for arg_shape, idxs, update_shape, dnums, max_idx in [
          ((5,), np.array([[0], [2]]), (2,),
           lax.ScatterDimensionNumbers(update_window_dims=(),
                                       inserted_window_dims=(0,),
                                       scatter_dims_to_operand_dims=(0,)), 4),
          ((10,), np.array([[0], [0], [0]]), (3, 2),
           lax.ScatterDimensionNumbers(update_window_dims=(1,),
                                       inserted_window_dims=(),
                                       scatter_dims_to_operand_dims=(0,)), 9),
          ((10, 5,), np.array([[0], [2], [1]]), (3, 3),
           lax.ScatterDimensionNumbers(update_window_dims=(1,),
                                       inserted_window_dims=(0,),
                                       scatter_dims_to_operand_dims=(0,)), 3),
      ]
    ],
    dtype=grad_float_dtypes,
    mode=["clip", "fill", "promise_in_bounds"],
    iteration=range(5),
  )
  def testScatterAddGrad(self, arg_shape, dtype, idxs_shape, idxs_dtype,
                         update_shape, dnums, max_idx, mode, iteration):
    rng = jtu.rand_default(self.rng())
    if mode == "promise_in_bounds":
      rng_idx = jtu.rand_int(self.rng(), high=max_idx)
    else:
      # Only test out-of-bounds indices if using a mode that guarantees correct
      # gradients for out-of-bounds indices.
      rng_idx = jtu.rand_int(self.rng(), low=-max_idx, high=2 * max_idx)
    idxs = rng_idx(idxs_shape, idxs_dtype)
    scatter_add = lambda x, y: lax.scatter_add(
      x, idxs, y, dimension_numbers=dnums, mode=mode)
    x = rng(arg_shape, dtype)
    y = rng(update_shape, dtype)
    check_grads(scatter_add, (x, y), 2, ["fwd", "rev"], 1e-2, 1e-2, 1.)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs=idxs, dnums=dnums,
          update_shape=update_shape, max_idx=max_idx, multiplier=multiplier)
      for arg_shape, idxs, update_shape, dnums, max_idx, multiplier in [
          ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
            update_window_dims=(), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)), 4, 1),
          ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(),
            scatter_dims_to_operand_dims=(0,)), 4, 2),
          ((10, 5,), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)), 9, 1),
      ]
    ],
    dtype=grad_float_dtypes,
  )
  def testScatterGrad(self, arg_shape, dtype, idxs, update_shape, dnums,
                      max_idx, multiplier):
    # Scatters with conflicting indices are not deterministic on GPU, so we
    # use indices that do not collide.
    rng_idx = jtu.rand_unique_int(self.rng(), high=max_idx)
    rng = jtu.rand_default(self.rng())
    # The multiplier ensures we don't pick overlapping windows if the update
    # window is not of size 1.
    idxs = rng_idx(idxs.shape, idxs.dtype) * multiplier
    scatter = lambda x, y: lax.scatter(x, idxs, y, dimension_numbers=dnums)
    x = rng(arg_shape, dtype)
    y = rng(update_shape, dtype)
    check_grads(scatter, (x, y), 2, ["fwd", "rev"], 1e-2, 1e-2, 1.)

  def testScatterGradSymbolicZeroUpdate(self):
    # https://github.com/google/jax/issues/1901
    def f(x):
      n = x.shape[0]
      y = np.arange(n, dtype=x.dtype)
      return jax.device_put(x).at[np.diag_indices(n)].set(y)
    rng = jtu.rand_default(self.rng())
    check_grads(f, (rng((5, 5), np.float32),), 2, ["fwd", "rev"], 1e-2, 1e-2,
                1.)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs=idxs, dnums=dnums,
          update_shape=update_shape)
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
      ]
    ],
    dtype=grad_float_dtypes,
  )
  def testScatterMax(self, arg_shape, dtype, idxs, update_shape, dnums):
    rng = jtu.rand_default(self.rng())
    rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
    idxs = rng_idx(idxs.shape, idxs.dtype)
    scatter_max = lambda x, y: lax.scatter_max(x, idxs, y, dnums)
    x = rng(arg_shape, dtype)
    y = rng(update_shape, dtype)
    check_grads(scatter_max, (x, y), 2, ["fwd", "rev"], 1e-2, 1e-2)

  @jtu.sample_product(
    [dict(arg_shape=arg_shape, idxs=idxs, dnums=dnums,
          update_shape=update_shape)
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
      ]
    ],
    dtype=grad_float_dtypes,
  )
  def testScatterMin(self, arg_shape, dtype, idxs, update_shape, dnums):
    rng = jtu.rand_default(self.rng())
    rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
    idxs = rng_idx(idxs.shape, idxs.dtype)
    scatter_min = lambda x, y: lax.scatter_min(x, idxs, y, dnums)
    x = rng(arg_shape, dtype)
    y = rng(update_shape, dtype)
    check_grads(scatter_min, (x, y), 2, ["fwd", "rev"], 1e-2, 1e-2)

  def testStopGradient(self):
    def f(x):
      return lax.sin(x) * lax.cos(lax.stop_gradient(x))

    def f2(x, y):
      return lax.sin(x) * lax.cos(y)

    x = 3.14
    ans = jax.grad(f)(x)
    expected = jax.grad(f2)(x, x)
    self.assertAllClose(ans, expected)

    ans = jax.grad(jax.grad(f))(x)
    expected = jax.grad(jax.grad(f2))(x, x)
    self.assertAllClose(ans, expected)

    ans = jax.grad(lambda x: lax.stop_gradient({'foo':x})['foo'])(3.)
    expected = np.array(0.0)
    self.assertAllClose(ans, expected, check_dtypes=False)

    with jax.enable_checks(False):
      with self.assertRaises(TypeError):
        lax.stop_gradient(lambda x: x)

  # TODO(mattjj): make this a more systematic test
  def testRemainder(self):
    def gen_x(rng, size):
      return rng.uniform(-9, 9, size=size)

    def gen_y(rng, size):
      # avoid values near zero because gradients diverge
      return rng.uniform(0.1, 5, size=size) * rng.choice([-1, 1], size=size)

    rng = self.rng()
    x = gen_x(rng, (5, 8))
    y = gen_y(rng, (1, 8))
    assert not set(np.unique(x)) & set(np.unique(y))
    check_grads(lax.rem, (x, y), 2, ["fwd", "rev"])

    rng = self.rng()
    x = gen_x(rng, (1, 8))
    y = gen_y(rng, (5, 8))
    assert not set(np.unique(x)) & set(np.unique(y))
    check_grads(lax.rem, (x, y), 2, ["fwd", "rev"])

  def testHigherOrderGradientOfReciprocal(self):
    # Regression test for https://github.com/google/jax/issues/3136
    def inv(x):
      # N.B.: intentionally written as 1/x, not x ** -1 or reciprocal(x)
      return 1 / x
    grad_fn = jax.grad(jax.grad(jax.grad(jax.grad(jax.grad(jax.grad(inv))))))
    self.assertAllClose(np.float32(0.0439453125), grad_fn(np.float32(4.)))

  def test_linear_transpose_real(self):
    f = lambda x: x.real
    transpose = jax.linear_transpose(f, 1.j)
    actual, = transpose(1.)
    expected = 1.
    self.assertEqual(actual, expected)

  def test_linear_transpose_imag(self):
    f = lambda x: x.imag
    transpose = jax.linear_transpose(f, 1.j)
    actual, = transpose(1.)
    expected = -1.j
    self.assertEqual(actual, expected)

  def test_scatter_apply_jvp(self):
    def f(x):
      return x.at[1].apply(jax.numpy.sin)

    x = jax.numpy.array([1.0, 2.0])
    with self.assertRaises(NotImplementedError):
      jax.jacfwd(f)(x)

  def test_scatter_apply_vjp(self):
    def f(x):
      return x.at[1].apply(jax.numpy.sin)

    x = jax.numpy.array([1.0, 2.0])

    with self.assertRaises(NotImplementedError):
      jax.jacrev(f)(x)

  def testPowShapeMismatch(self):
    # Regression test for https://github.com/google/jax/issues/17294
    x = lax.iota('float32', 4)
    y = 2
    actual = jax.jacrev(jax.jit(jax.lax.pow))(x, y)  # no error
    expected = jax.numpy.diag(y * x ** (y - 1))
    self.assertArraysEqual(actual, expected)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
