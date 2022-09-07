# Copyright 2020 Google LLC
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
import itertools
from typing import Optional, cast
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
import jax.numpy as jnp
from jax import dtypes
from jax import lax

from jax._src import test_util as jtu
from jax._src.lax import windowed_reductions as lax_windowed_reductions
from jax._src.lib import xla_client
from jax._src.util import safe_map, safe_zip

from lax_test import LAX_OPS

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

float_dtypes = jtu.dtypes.all_floating
default_dtypes = jtu.dtypes.all_floating + jtu.dtypes.integer
all_dtypes = jtu.dtypes.all

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

compatible_shapes = [[(3,)], [(3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]

def all_bdims(*shapes):
  bdims = (itertools.chain([cast(Optional[int], None)],
                           range(len(shape) + 1)) for shape in shapes)
  return (t for t in itertools.product(*bdims) if not all(e is None for e in t))

def add_bdim(bdim_size, bdim, shape):
  shape = list(shape)
  if bdim is not None:
    shape.insert(bdim, bdim_size)
  return tuple(shape)

def slicer(x, bdim):
  if bdim is None:
    return lambda _: x
  else:
    return lambda i: lax.index_in_dim(x, i, bdim, keepdims=False)

def args_slicer(args, bdims):
  slicers = map(slicer, args, bdims)
  return lambda i: [sl(i) for sl in slicers]

class LaxVmapTest(jtu.JaxTestCase):

  def _CheckBatching(self, op, bdim_size, bdims, shapes, dtypes, rng,
                     rtol=None, atol=None, multiple_results=False):
    batched_shapes = map(partial(add_bdim, bdim_size), bdims, shapes)
    args = [rng(shape, dtype) for shape, dtype in zip(batched_shapes, dtypes)]
    args_slice = args_slicer(args, bdims)
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

  @parameterized.named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_bdims={}".format(
            jtu.format_test_name_suffix(rec.op, shapes,
                                        itertools.repeat(dtype)), bdims),
         "op_name": rec.op, "rng_factory": rec.rng_factory, "shapes": shapes,
         "dtype": dtype, "bdims": bdims, "tol": rec.tol}
        for shape_group in compatible_shapes
        for shapes in itertools.combinations_with_replacement(shape_group, rec.nargs)
        for bdims in all_bdims(*shapes)
        for dtype in rec.dtypes)
      for rec in LAX_OPS))
  def testOp(self, op_name, rng_factory, shapes, dtype, bdims, tol):
    rng = rng_factory(self.rng())
    op = getattr(lax, op_name)
    self._CheckBatching(op, 10, bdims, shapes, [dtype] * len(shapes), rng,
                        atol=tol, rtol=tol)

  @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
       "testcase_name":
       "_lhs_shape={}_rhs_shape={}_strides={}_padding={}_lhs_dilation={}_"
       "rhs_dilation={}_dims={}_feature_group_count={}_batch_group_count={}"
       "_lhs_bdim={}_rhs_bdim={}_bdim_size={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               strides, padding, lhs_dil, rhs_dil, ",".join(dim_nums),
               feature_group_count, batch_group_count, lhs_bdim, rhs_bdim,
               bdim_size),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "strides": strides, "padding": padding, "lhs_dil": lhs_dil,
       "rhs_dil": rhs_dil, "dimension_numbers": dim_nums,
       "perms": perms, "lhs_bdim": lhs_bdim, "rhs_bdim": rhs_bdim,
       "feature_group_count": feature_group_count,
       "batch_group_count": batch_group_count, "bdim_size": bdim_size,
     } for batch_group_count, feature_group_count in s([(1, 1), (2, 1), (1, 2)])
       for lhs_shape, rhs_shape, all_strides, all_pads, lhs_dils, rhs_dils in s([
           ((b * batch_group_count, i * feature_group_count, 6, 7),  # lhs_shape
            (j * batch_group_count * feature_group_count, i, 1, 2),  # rhs_shape
            [(1, 1), (1, 2), (2, 1)],  # strides
            [((0, 0), (0, 0)), ((1, 0), (0, 1)), ((0, -1), (0, 0))],  # pads
            [(1, 1), (2, 1)],  # lhs_dils
            [(1, 1), (2, 2)])  # rhs_dils
           for b, i, j in itertools.product([1, 2], repeat=3)])
       for strides in s(all_strides)
       for rhs_dil in s(rhs_dils)
       for lhs_dil in s(lhs_dils)
       for dtype in s([np.float32])
       for padding in s(all_pads)
       for dim_nums, perms in s([
           (("NCHW", "OIHW", "NCHW"), ([0, 1, 2, 3], [0, 1, 2, 3])),
           (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0])),
           (("NHWC", "OIHW", "NCHW"), ([0, 2, 3, 1], [0, 1, 2, 3])),
           (("HWCN", "HWIO", "HWCN"), ([2, 3, 1, 0], [2, 3, 1, 0])),
       ])
       for bdim_size in s([0, 5])
       for lhs_bdim in s(itertools.chain([cast(Optional[int], None)],
                                         range(len(lhs_shape) + 1)))
       for rhs_bdim in s(itertools.chain([cast(Optional[int], None)],
                                         range(len(rhs_shape) + 1)))
       if (lhs_bdim, rhs_bdim) != (None, None)
       )))
  def testConvGeneralDilatedBatching(
      self, lhs_shape, rhs_shape, dtype, strides, padding, lhs_dil, rhs_dil,
      dimension_numbers, perms, feature_group_count, batch_group_count,
      lhs_bdim, rhs_bdim, bdim_size):
    rng = jtu.rand_default(self.rng())
    tol = 1e-1 if dtypes.finfo(dtype).bits <= 32 else 1e-3

    # permute shapes to match dim_spec, scale by feature_group_count
    lhs_perm, rhs_perm = perms
    lhs_shape = list(np.take(lhs_shape, lhs_perm))
    rhs_shape = list(np.take(rhs_shape, rhs_perm))

    conv = partial(lax.conv_general_dilated, window_strides=strides,
                   padding=padding, lhs_dilation=lhs_dil, rhs_dilation=rhs_dil,
                   dimension_numbers=dimension_numbers,
                   feature_group_count=feature_group_count,
                   batch_group_count=batch_group_count,
                   precision=lax.Precision.HIGHEST)
    self._CheckBatching(conv, bdim_size, (lhs_bdim, rhs_bdim),
                        (lhs_shape, rhs_shape), (dtype, dtype), rng, rtol=tol,
                        atol=tol)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_from_dtype={}_to_dtype={}_bdims={}".format(
          shape, from_dtype, to_dtype, bdims),
       "shape": shape, "from_dtype": from_dtype, "to_dtype": to_dtype,
       "bdims": bdims}
      for from_dtype, to_dtype in itertools.product(
          [np.float32, np.int32, "float32", "int32"], repeat=2)
      for shape in [(2, 3)]
      for bdims in all_bdims(shape)))
  def testConvertElementType(self, shape, from_dtype, to_dtype, bdims):
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.convert_element_type(x, to_dtype)
    self._CheckBatching(op, 10, bdims, (shape,), (from_dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_nmant={}_nexp={}_bdims={}".format(
          jtu.format_shape_dtype_string(shape, dtype), nmant, nexp, bdims),
       "shape": shape, "dtype": dtype, "nmant": nmant, "nexp": nexp, "bdims": bdims}
      for dtype in float_dtypes
      for shape in [(2, 4)]
      for nexp in [1, 3, 5]
      for nmant in [0, 2, 4]
      for bdims in all_bdims(shape)))
  def testReducePrecision(self, shape, dtype, nmant, nexp, bdims):
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.reduce_precision(x, exponent_bits=nexp, mantissa_bits=nmant)
    self._CheckBatching(op, 10, bdims, (shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_from_dtype={}_to_dtype={}_bdims={}".format(
          shape, from_dtype, to_dtype, bdims),
       "shape": shape, "from_dtype": from_dtype, "to_dtype": to_dtype,
       "bdims": bdims}
      for from_dtype, to_dtype in itertools.product(
          [np.float32, np.int32, "float32", "int32"], repeat=2)
      for shape in [(2, 3)]
      for bdims in all_bdims(shape)))
  def testBitcastElementType(self, shape, from_dtype, to_dtype, bdims,):
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.bitcast_convert_type(x, to_dtype)
    self._CheckBatching(op, 10, bdims, (shape,), (from_dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_min_shape={}_operand_shape={}_max_shape={}_bdims={}"
       .format(jtu.format_shape_dtype_string(min_shape, dtype),
               jtu.format_shape_dtype_string(operand_shape, dtype),
               jtu.format_shape_dtype_string(max_shape, dtype),
               bdims),
       "min_shape": min_shape, "operand_shape": operand_shape,
       "max_shape": max_shape, "dtype": dtype, "bdims": bdims}
      for min_shape, operand_shape, max_shape in [
          [(), (2, 3), ()],
          [(2, 3), (2, 3), ()],
          [(), (2, 3), (2, 3)],
          [(2, 3), (2, 3), (2, 3)],
      ]
      for dtype in default_dtypes
      for bdims in all_bdims(min_shape, operand_shape, max_shape)))
  def testClamp(self, min_shape, operand_shape, max_shape, dtype, bdims):
    rng = jtu.rand_default(self.rng())
    shapes = [min_shape, operand_shape, max_shape]
    self._CheckBatching(lax.clamp, 10, bdims, shapes, [dtype] * 3, rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_lhs_shape={}_rhs_shape={}_bdims={}".format(
          jtu.format_shape_dtype_string(lhs_shape, dtype),
          jtu.format_shape_dtype_string(rhs_shape, dtype),
          bdims),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "bdims": bdims}
      for lhs_shape in [(3,), (4, 3)] for rhs_shape in [(3,), (3, 6)]
      for bdims in all_bdims(lhs_shape, rhs_shape)
      for dtype in default_dtypes))
  def testDot(self, lhs_shape, rhs_shape, dtype, bdims):
    rng = jtu.rand_default(self.rng())
    op = partial(lax.dot, precision=lax.Precision.HIGHEST)
    self._CheckBatching(op, 5, bdims, (lhs_shape, rhs_shape), (dtype, dtype),
                        rng, rtol={np.float16: 5e-2, np.float64: 5e-14})

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_lhs_contracting={}_rhs_contracting={}_bdims={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               lhs_contracting, rhs_contracting, bdims),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "lhs_contracting": lhs_contracting, "rhs_contracting": rhs_contracting,
       "bdims": bdims}
      for lhs_shape, rhs_shape, lhs_contracting, rhs_contracting in [
          [(5,), (5,), [0], [0]],
          [(5, 7), (5,), [0], [0]],
          [(7, 5), (5,), [1], [0]],
          [(3, 5), (2, 5), [1], [1]],
          [(5, 3), (5, 2), [0], [0]],
          [(5, 3, 2), (5, 2, 4), [0], [0]],
          [(5, 3, 2), (5, 2, 4), [0,2], [0,1]],
          [(5, 3, 2), (3, 5, 2, 4), [0,2], [1,2]],
          [(1, 2, 2, 3), (1, 2, 3, 1), [1], [1]],
          [(3, 2), (2, 4), [1], [0]],
      ]
      for bdims in all_bdims(lhs_shape, rhs_shape)
      for dtype in default_dtypes))
  def testDotGeneralContractOnly(self, lhs_shape, rhs_shape, dtype,
                                 lhs_contracting, rhs_contracting, bdims):
    rng = jtu.rand_small(self.rng())
    dimension_numbers = ((lhs_contracting, rhs_contracting), ([], []))
    dot = partial(lax.dot_general, dimension_numbers=dimension_numbers)
    self._CheckBatching(dot, 5, bdims, (lhs_shape, rhs_shape), (dtype, dtype),
                        rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_lhs_shape={}_rhs_shape={}_dimension_numbers={}_bdims={}"
       .format(jtu.format_shape_dtype_string(lhs_shape, dtype),
               jtu.format_shape_dtype_string(rhs_shape, dtype),
               dimension_numbers, bdims),
       "lhs_shape": lhs_shape, "rhs_shape": rhs_shape, "dtype": dtype,
       "dimension_numbers": dimension_numbers, "bdims": bdims}
      for lhs_shape, rhs_shape, dimension_numbers in [
          ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
          ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1]))),
          ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1]))),
      ]
      for bdims in all_bdims(lhs_shape, rhs_shape)
      for dtype in default_dtypes))
  def testDotGeneralContractAndBatch(self, lhs_shape, rhs_shape, dtype,
                                     dimension_numbers, bdims):
    rng = jtu.rand_small(self.rng())
    dot = partial(lax.dot_general, dimension_numbers=dimension_numbers)
    self._CheckBatching(dot, 5, bdims, (lhs_shape, rhs_shape), (dtype, dtype),
                        rng)

    # Checks that batching didn't introduce any transposes or broadcasts.
    jaxpr = jax.make_jaxpr(dot)(np.zeros(lhs_shape, dtype),
                                np.zeros(rhs_shape, dtype))
    for eqn in jtu.iter_eqns(jaxpr.jaxpr):
      self.assertFalse(eqn.primitive in ["transpose", "broadcast"])

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype={}_broadcast_sizes={}_bdims={}".format(
          shape, np.dtype(dtype).name, broadcast_sizes, bdims),
       "shape": shape, "dtype": dtype, "broadcast_sizes": broadcast_sizes,
       "bdims": bdims}
      for shape in [(), (2, 3)]
      for dtype in default_dtypes
      for broadcast_sizes in [(), (2,), (1, 2)]
      for bdims in all_bdims(shape)))
  def testBroadcast(self, shape, dtype, broadcast_sizes, bdims):
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.broadcast(x, broadcast_sizes)
    self._CheckBatching(op, 5, bdims, (shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_bcdims={}_bdims={}".format(
          jtu.format_shape_dtype_string(inshape, dtype),
          outshape, broadcast_dimensions, bdims),
       "inshape": inshape, "dtype": dtype, "outshape": outshape,
       "dimensions": broadcast_dimensions, "bdims": bdims}
      for inshape, outshape, broadcast_dimensions in [
          ([2], [2, 2], [0]),
          ([2], [2, 2], [1]),
          ([2], [2, 3], [0]),
          ([], [2, 3], []),
      ]
      for dtype in default_dtypes
      for bdims in all_bdims(inshape)))
  @unittest.skip("this test has failures in some cases")  # TODO(mattjj)
  def testBroadcastInDim(self, inshape, dtype, outshape, dimensions, bdims):
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.broadcast_in_dim(x, outshape, dimensions)
    self._CheckBatching(op, 5, bdims, (inshape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_dimensions={}_bdims={}".format(
          jtu.format_shape_dtype_string(arg_shape, np.float32),
          dimensions, bdims),
       "arg_shape": arg_shape, "dimensions": dimensions, "bdims": bdims}
      for arg_shape, dimensions in [
          [(1,), (0,)],
          [(1,), (-1,)],
          [(2, 1, 4), (1,)],
          [(2, 1, 4), (-2,)],
          [(2, 1, 3, 1), (1,)],
          [(2, 1, 3, 1), (1, 3)],
          [(2, 1, 3, 1), (3,)],
          [(2, 1, 3, 1), (1, -1)],
      ]
      for bdims in all_bdims(arg_shape)))
  def testSqueeze(self, arg_shape, dimensions, bdims):
    dtype = np.float32
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.squeeze(x, dimensions)
    self._CheckBatching(op, 10, bdims, (arg_shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_dims={}_bdims={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype),
          dimensions, bdims),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "dimensions": dimensions, "bdims": bdims}
      for dtype in default_dtypes
      for arg_shape, dimensions, out_shape in [
          [(3, 4), None, (12,)],
          [(2, 1, 4), None, (8,)],
          [(2, 2, 4), None, (2, 8)],
          [(2, 2, 4), (0, 1, 2), (2, 8)],
          [(2, 2, 4), (1, 0, 2), (8, 2)],
          [(2, 2, 4), (2, 1, 0), (4, 2, 2)]
      ]
      for bdims in all_bdims(arg_shape)))
  def testReshape(self, arg_shape, out_shape, dtype, dimensions, bdims):
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.reshape(x, out_shape, dimensions=dimensions)
    self._CheckBatching(op, 10, bdims, (arg_shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_pads={}_bdims={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), pads, bdims),
       "shape": shape, "dtype": dtype, "pads": pads, "bdims": bdims}
      for shape in [(2, 3)]
      for bdims in all_bdims(shape, ())
      for dtype in default_dtypes
      for pads in [[(1, 2, 1), (0, 1, 0)]]))
  def testPad(self, shape, dtype, pads, bdims):
    rng = jtu.rand_small(self.rng())
    fun = lambda operand, padding: lax.pad(operand, padding, pads)
    self._CheckBatching(fun, 5, bdims, (shape, ()), (dtype, dtype), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_predshape={}_argshapes={}_bdims={}".format(
          jtu.format_shape_dtype_string(pred_shape, np.bool_),
          jtu.format_shape_dtype_string(arg_shape, arg_dtype),
          bdims),
       "pred_shape": pred_shape, "arg_shape": arg_shape, "arg_dtype": arg_dtype,
       "bdims": bdims}
      for arg_shape in [(), (3,), (2, 3)]
      for pred_shape in ([(), arg_shape] if arg_shape else [()])
      for bdims in all_bdims(pred_shape, arg_shape, arg_shape)
      for arg_dtype in default_dtypes))
  def testSelect(self, pred_shape, arg_shape, arg_dtype, bdims):
    rng = jtu.rand_default(self.rng())
    op = lambda c, x, y: lax.select(c < 0, x, y)
    self._CheckBatching(op, 5, bdims, (pred_shape, arg_shape, arg_shape,),
                        (arg_dtype, arg_dtype, arg_dtype), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_shape={}_start_indices={}_limit_indices={}_strides={}_bdims={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, limit_indices, strides, bdims),
       "shape": shape, "dtype": dtype, "starts": start_indices,
       "limits": limit_indices, "strides": strides, "bdims": bdims}
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
      ]
      for bdims in all_bdims(shape)
      for dtype in default_dtypes))
  def testSlice(self, shape, dtype, starts, limits, strides, bdims):
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.slice(x, starts, limits, strides)
    self._CheckBatching(op, 5, bdims, (shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_perm={}_bdims={}".format(
          jtu.format_shape_dtype_string(shape, dtype), perm, bdims),
       "shape": shape, "dtype": dtype, "perm": perm, "bdims": bdims}
      for shape, perm in [
        [(3, 4), (1, 0)],
        [(3, 4), (0, 1)],
        [(3, 4, 5), (2, 1, 0)],
        [(3, 4, 5), (1, 0, 2)],
      ]
      for bdims in all_bdims(shape)
      for dtype in default_dtypes))
  def testTranspose(self, shape, dtype, perm, bdims):
    rng = jtu.rand_default(self.rng())
    op = lambda x: lax.transpose(x, perm)
    self._CheckBatching(op, 5, bdims, (shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_inshape={}_reducedims={}_initval={}_bdims={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), dims,
               init_val, bdims),
       "op": op, "init_val": init_val, "shape": shape, "dtype": dtype,
       "dims": dims, "bdims": bdims}
      for init_val, op, dtypes in [
          (0, lax.add, default_dtypes),
          (1, lax.mul, default_dtypes),
          # non-monoidal for everything except unsigned integers
          (0, lax.max, all_dtypes),
          (-np.inf, lax.max, float_dtypes),
          (dtypes.iinfo(np.int32).min, lax.max, [np.int32]),
          (dtypes.iinfo(np.int64).min, lax.max, [np.int64]),
          (np.inf, lax.min, float_dtypes),
          (dtypes.iinfo(np.int32).max, lax.min, [np.int32]),
          (dtypes.iinfo(np.int64).max, lax.min, [np.int64]),
          (dtypes.iinfo(np.uint32).max, lax.min, [np.uint32]),
          (dtypes.iinfo(np.uint64).max, lax.min, [np.uint64]),
      ]
      for dtype in dtypes
      for shape, dims in [
          [(3, 4, 5), (0,)], [(3, 4, 5), (1, 2)],
          [(3, 4, 5), (0, 2)], [(3, 4, 5), (0, 1, 2)]
      ]
      for bdims in all_bdims(shape)))
  def testReduce(self, op, init_val, shape, dtype, dims, bdims):
    rng = jtu.rand_small(self.rng())
    init_val = np.asarray(init_val, dtype=dtype)
    fun = lambda operand: lax.reduce(operand, init_val, op, dims)
    self._CheckBatching(fun, 5, bdims, (shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_reducedims={}_bdims={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), dims, bdims),
       "shape": shape, "dtype": dtype, "dims": dims, "bdims": bdims}
      for dtype in default_dtypes
      for shape, dims in [
          [(3, 4, 5), (0,)], [(3, 4, 5), (1, 2)],
          [(3, 4, 5), (0, 2)], [(3, 4, 5), (0, 1, 2)]
      ]
      for bdims in all_bdims(shape, shape)))
  def testVariadicReduce(self, shape, dtype, dims, bdims):
    def op(a, b):
      x1, y1 = a
      x2, y2 = b
      return x1 + x2, y1 * y2
    rng = jtu.rand_small(self.rng())
    init_val = tuple(np.asarray([0, 1], dtype=dtype))
    fun = lambda x, y: lax.reduce((x, y), init_val, op, dims)
    self._CheckBatching(fun, 5, bdims, (shape, shape), (dtype, dtype), rng,
                        multiple_results=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_inshape={}_reducedims={}_bdims={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), dim,
               bdims),
       "op": op, "shape": shape, "dtype": dtype,
       "dim": dim, "bdims": bdims}
      for op in [lax.argmin, lax.argmax]
      for dtype in default_dtypes
      for shape in [(3, 4, 5)]
      for dim in range(len(shape))
      for bdims in all_bdims(shape)))
  def testArgminmax(self, op, shape, dtype, dim, bdims):
    rng = jtu.rand_default(self.rng())
    fun = lambda operand: op(operand, dim, np.int32)
    self._CheckBatching(fun, 5, bdims, (shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": ("_op={}_shape={}_dims={}_strides={}_padding={}"
                         "_basedilation={}_windowdilation={}")
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype),
               dims, strides, padding, base_dilation, window_dilation),
       "op": op, "init_val": init_val, "dtype": dtype, "shape": shape,
       "dims": dims, "strides": strides, "padding": padding,
       "base_dilation": base_dilation, "window_dilation": window_dilation}
      for init_val, op, dtypes in [
          (0, lax.add, [np.float32]),
          (-np.inf, lax.max, [np.float32]),
          (np.inf, lax.min, [np.float32]),
      ]
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
      for dtype in dtypes))
  def testReduceWindow(self, op, init_val, dtype, shape, dims, strides, padding,
                       base_dilation, window_dilation):
    rng = jtu.rand_small(self.rng())
    init_val = np.asarray(init_val, dtype=dtype)

    def fun(operand):
      return lax.reduce_window(operand, init_val, op, dims, strides, padding,
                               base_dilation, window_dilation)

    for bdims in all_bdims(shape):
      self._CheckBatching(fun, 3, bdims, (shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_shape={}_axis={}_bdims={}_reverse={}"
       .format(op.__name__, jtu.format_shape_dtype_string(shape, dtype), axis,
               bdims, reverse),
       "op": op, "shape": shape, "dtype": dtype, "bdims": bdims,
       "axis": axis, "reverse": reverse}
      for op, types in [
          (lax.cumsum, [np.float32, np.float64]),
          (lax.cumprod, [np.float32, np.float64]),
      ]
      for dtype in types
      for shape in [[10], [3, 4, 5]]
      for axis in range(len(shape))
      for bdims in all_bdims(shape)
      for reverse in [False, True]))
  def testCumulativeReduce(self, op, shape, dtype, axis, bdims, reverse):
    rng_factory = (jtu.rand_default if dtypes.issubdtype(dtype, np.integer)
                   else jtu.rand_small)
    rng = rng_factory(self.rng())
    self._CheckBatching(partial(op, axis=axis, reverse=reverse), 7, bdims,
                        (shape,), (dtype,), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}_padding={}".format(np.dtype(dtype).name,
                                                      padding),
       "dtype": dtype, "padding": padding}
      for dtype in float_dtypes
      for padding in ["VALID", "SAME"]))
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  @jtu.ignore_warning(message="Using reduced precision for gradient.*")
  def testSelectAndGatherAdd(self, dtype, padding):
    rng = jtu.rand_small(self.rng())
    all_configs = itertools.chain(
        itertools.product(
            [(4, 6)],
            [(2, 1), (1, 2)],
            [(1, 1), (2, 1), (1, 2)]),
        itertools.product(
            [(3, 2, 4, 6)], [(1, 1, 2, 1), (2, 1, 2, 1)],
            [(1, 2, 2, 1), (1, 1, 1, 1)]))

    def fun(operand, tangents):
      pads = lax.padtype_to_pads(operand.shape, dims, strides, padding)
      ones = (1,) * len(operand.shape)
      return lax_windowed_reductions._select_and_gather_add(
          operand, tangents, lax.ge_p, dims, strides, pads, ones, ones)

    for shape, dims, strides in all_configs:
      for bdims in all_bdims(shape, shape):
        self._CheckBatching(fun, 3, bdims, (shape, shape), (dtype, dtype), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_dtype={jtu.format_shape_dtype_string(shape, dtype)}"
      f"_padding={padding}_dims={dims}_strides={strides}",
       "dtype": dtype, "padding": padding, "shape": shape,
       "dims": dims, "strides": strides}
      for dtype in float_dtypes
      for padding in ["VALID", "SAME"]
      for shape in [(3, 2, 4, 6)]
      for dims in [(1, 1, 2, 1)]
      for strides in [(1, 2, 2, 1), (1, 1, 1, 1)]))
  def testSelectAndScatterAdd(self, dtype, padding, shape, dims, strides):
    rng = jtu.rand_small(self.rng())

    pads = lax.padtype_to_pads(shape, dims, strides, padding)

    def fun(operand, cotangents):
      return lax_windowed_reductions._select_and_scatter_add(
          operand, cotangents, lax.ge_p, dims, strides, pads)
    ones = (1,) * len(shape)
    cotangent_shape = jax.eval_shape(
      lambda x: lax_windowed_reductions._select_and_gather_add(
          x, x, lax.ge_p, dims, strides, pads, ones, ones),
      np.ones(shape, dtype)).shape

    for bdims in all_bdims(cotangent_shape, shape):
      self._CheckBatching(fun, 3, bdims, (cotangent_shape, shape),
                          (dtype, dtype), rng)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_bdims={}_fft_ndims={}"
       .format(shape, bdims, fft_ndims),
       "shape": shape, "bdims": bdims, "fft_ndims": fft_ndims}
      for shape in [(5,), (3, 4, 5), (2, 3, 4, 5)]
      for bdims in all_bdims(shape)
      for fft_ndims in range(0, min(3, len(shape)) + 1)))
  def testFft(self, fft_ndims, shape, bdims):
    rng = jtu.rand_default(self.rng())
    ndims = len(shape)
    axes = range(ndims - fft_ndims, ndims)
    fft_lengths = tuple(shape[axis] for axis in axes)
    op = lambda x: lax.fft(x, xla_client.FftType.FFT, fft_lengths)
    self._CheckBatching(op, 5, bdims, [shape], [np.complex64], rng,
                        rtol=1e-5)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_dnums={}_slice_sizes={}_bdims={}"
       .format(jtu.format_shape_dtype_string(shape, dtype), idxs, dnums,
               slice_sizes, bdims),
       "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes, "bdims": bdims}
      for dtype in all_dtypes
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
      ]
      for bdims in all_bdims(shape, idxs.shape)))
  def testGather(self, shape, dtype, idxs, dnums, slice_sizes, bdims):
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    self._CheckBatching(fun, 0, bdims, [shape, idxs.shape], [dtype, idxs.dtype],
                        jtu.rand_default(self.rng()))
    self._CheckBatching(fun, 5, bdims, [shape, idxs.shape], [dtype, idxs.dtype],
                        jtu.rand_default(self.rng()))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_idxs={}_update={}_dnums={}_bdims={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          idxs, update_shape, dnums, bdims),
       "arg_shape": arg_shape, "dtype": dtype, "idxs": idxs,
       "update_shape": update_shape, "dnums": dnums, "bdims": bdims}
      for dtype in float_dtypes
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
      for bdims in all_bdims(arg_shape, idxs.shape, update_shape)))
  def testScatterAdd(self, arg_shape, dtype, idxs, update_shape, dnums, bdims):
    fun = partial(lax.scatter_add, dimension_numbers=dnums)
    self._CheckBatching(fun, 5, bdims, [arg_shape, idxs.shape, update_shape],
                        [dtype, idxs.dtype, dtype], jtu.rand_default(self.rng()),
                        rtol={np.float16: 5e-3, dtypes.bfloat16: 3e-2})

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name":
              "_shape={}_idxs={}_update={}_dnums={}_bdims={}".format(
                  jtu.format_shape_dtype_string(arg_shape, dtype), idxs,
                  update_shape, dnums, bdims),
          "arg_shape":
              arg_shape,
          "dtype":
              dtype,
          "idxs":
              idxs,
          "update_shape":
              update_shape,
          "dnums":
              dnums,
          "bdims":
              bdims
      } for dtype in float_dtypes for arg_shape, idxs, update_shape, dnums in [
          ((5,), np.array([[0], [2]]), (2,),
           lax.ScatterDimensionNumbers(
               update_window_dims=(),
               inserted_window_dims=(0,),
               scatter_dims_to_operand_dims=(0,))),
          ((10,), np.array([[0], [0], [0]]), (3, 2),
           lax.ScatterDimensionNumbers(
               update_window_dims=(1,),
               inserted_window_dims=(),
               scatter_dims_to_operand_dims=(0,))),
          ((
              10,
              5,
          ), np.array([[0], [2], [1]]), (3, 3),
           lax.ScatterDimensionNumbers(
               update_window_dims=(1,),
               inserted_window_dims=(0,),
               scatter_dims_to_operand_dims=(0,))),
      ] for bdims in all_bdims(arg_shape, idxs.shape, update_shape)))
  def testScatterReduce(self, arg_shape, dtype, idxs, update_shape, dnums,
                        bdims):
    fun = partial(
        lax.scatter_reduce, computation=jnp.logaddexp, dimension_numbers=dnums)
    self._CheckBatching(
        fun,
        5,
        bdims, [arg_shape, idxs.shape, update_shape],
        [dtype, idxs.dtype, dtype],
        jtu.rand_default(self.rng()),
        rtol={
            np.float16: 5e-3,
            dtypes.bfloat16: 3e-2
        })

  def testShapeUsesBuiltinInt(self):
    x = lax.iota(np.int32, 3) + 1
    self.assertIsInstance(x.shape[0], int)  # not np.int64

  def testBroadcastShapesReturnsPythonInts(self):
    shape1, shape2 = (1, 2, 3), (2, 3)
    out_shape = lax.broadcast_shapes(shape1, shape2)
    self.assertTrue(all(type(s) is int for s in out_shape))

  def testBroadcastShapesFaultyInputs(self):
    err_shape1, err_shape2 = (-1,), "hello"
    # negative inputs should fail while informing about illegal negative indices...
    with self.assertRaisesRegex(TypeError, "Only non-negative indices are allowed.*"):
      lax.broadcast_shapes(err_shape1)
    # ... while non-integers should error earlier, in the canonicalize_shape machinery.
    with self.assertRaisesRegex(TypeError, "Shapes must be 1D sequences.*"):
      lax.broadcast_shapes(err_shape2)  # pytype: disable=wrong-arg-types

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}_bdims={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k, bdims),
       "shape": shape, "dtype": dtype, "k": k, "bdims": bdims, "rng_factory": rng_factory}
      for shape in [(4,), (3, 5, 3)]
      for k in [1, 3]
      for bdims in all_bdims(shape)
      # TODO(b/155170120): test with repeats once the XLA:CPU stable top_k bug is fixed:
      # The top_k indices for integer arrays with identical entries won't match between
      # vmap'd version and manual reference, so only test unique integer arrays for int_dtypes.
      # Note also that we chose 3 * 5 * 3 * 5 such that it fits in the range of
      # values a bfloat16 can represent exactly to avoid ties.
      for dtype, rng_factory in itertools.chain(
        unsafe_zip(default_dtypes, itertools.repeat(jtu.rand_unique_int)))))
  def testTopK(self, shape, dtype, k, bdims, rng_factory):
    rng = rng_factory(self.rng())
    # _CheckBatching doesn't work with tuple outputs, so test outputs separately.
    op1 = lambda x: lax.top_k(x, k=k)[0]
    self._CheckBatching(op1, 5, bdims, (shape,), (dtype,), rng)
    op2 = lambda x: lax.top_k(x, k=k)[1]
    self._CheckBatching(op2, 5, bdims, (shape,), (dtype,), rng)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dimension={}_arity={}_bdims={}_isstable={}"
       .format(jtu.format_shape_dtype_string(shape, np.float32), dimension,
               arity, bdims, is_stable),
       "shape": shape, "dimension": dimension, "arity": arity, "bdims": bdims,
       "is_stable": is_stable}
      for shape in [(2, 3)]
      for dimension in [0, 1]
      for arity in range(3)
      for bdims in all_bdims(*((shape,) * arity))
      for is_stable in [False, True]))
  def testSort(self, shape, dimension, arity, bdims, is_stable):
    rng = jtu.rand_default(self.rng())
    if arity == 1:
      fun = partial(lax.sort, dimension=dimension)
      self._CheckBatching(fun, 5, bdims, (shape,) * arity, (np.float32,) * arity,
                          rng)
    else:
      for i in range(arity):
        fun = lambda *args, i=i: lax.sort(args,
                                          dimension=dimension,
                                          is_stable=is_stable)[i]
        self._CheckBatching(fun, 5, bdims, (shape,) * arity,
                            (np.float32,) * arity, rng)


  # TODO Concatenate
  # TODO Reverse
  # TODO DynamicSlice
  # TODO DynamicUpdateSlice
  # TODO Collapse
  # TODO Scatter

  # TODO(b/183233858): variadic reduce-window is not implemented on XLA:GPU
  @jtu.skip_on_devices("gpu")
  def test_variadic_reduce_window(self):
    # https://github.com/google/jax/discussions/9818 and
    # https://github.com/google/jax/issues/9837
    def normpool(x):
      norms = jnp.linalg.norm(x, axis=-1)
      idxs = jnp.arange(x.shape[0])

      def g(a, b):
        an, ai = a
        bn, bi = b
        which = an >= bn
        return (jnp.where(which, an, bn), jnp.where(which, ai, bi))

      _, idxs = lax.reduce_window((norms, idxs), (-np.inf, -1), g,
                        window_dimensions=(2,), window_strides=(2,),
                        padding=((0, 0),))
      return x[idxs]


    inpt = jnp.array([
      [1.0, 0.0, 1.0],
      [2.0, 2.0, 0.0],
      [3.0, 0.0, 1.0],
      [0.0, 1.0, 1.0],
    ])
    output = jax.vmap(normpool)(inpt[None, ...])  # doesn't crash
    expected = jnp.array([[[2.0, 2.0, 0.0], [3.0, 0.0, 1.0]]])
    self.assertAllClose(output, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
