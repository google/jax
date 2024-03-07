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


import dataclasses
import enum
import functools
from functools import partial
import itertools
import operator

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import numpy as jnp
from jax._src import earray
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.lax import lax as lax_internal

config.parse_flags_with_absl()

bool_dtypes = [np.dtype('bool')]

np_signed_dtypes = [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
                    np.dtype('int64')]
signed_dtypes = list(np_signed_dtypes)

np_unsigned_dtypes = [np.dtype('uint8'), np.dtype('uint16'), np.dtype('uint32'),
                     np.dtype('uint64')]
unsigned_dtypes = list(np_unsigned_dtypes)

int4_dtypes = [np.dtype('int4'), np.dtype('uint4')]
signed_dtypes += [np.dtype('int4')]
unsigned_dtypes += [np.dtype('uint4')]

np_float_dtypes = [np.dtype('float16'), np.dtype('float32'),
                   np.dtype('float64')]

float_dtypes = [np.dtype(dtypes.bfloat16)] + np_float_dtypes
custom_float_dtypes = [np.dtype(dtypes.bfloat16)]

fp8_dtypes = [np.dtype(dtypes.float8_e4m3b11fnuz), np.dtype(dtypes.float8_e4m3fn),
              np.dtype(dtypes.float8_e4m3fnuz), np.dtype(dtypes.float8_e5m2),
              np.dtype(dtypes.float8_e5m2fnuz)]
float_dtypes += fp8_dtypes
custom_float_dtypes += fp8_dtypes

complex_dtypes = [np.dtype('complex64'), np.dtype('complex128')]


all_dtypes = (bool_dtypes + signed_dtypes + unsigned_dtypes + float_dtypes +
              complex_dtypes)

scalar_types = [jnp.bool_, jnp.int4, jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                jnp.uint4, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64,
                jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64,
                jnp.complex64, jnp.complex128]

dtype_kinds = {
  'bool': bool_dtypes,
  'signed integer': signed_dtypes,
  'unsigned integer': unsigned_dtypes,
  'integral': signed_dtypes + unsigned_dtypes,
  'real floating': float_dtypes,
  'complex floating': complex_dtypes,
  'numeric': signed_dtypes + unsigned_dtypes + float_dtypes + complex_dtypes,
}

python_scalar_types = [bool, int, float, complex]

_EXPECTED_CANONICALIZE_X64 = {value: value for value in scalar_types}

_EXPECTED_CANONICALIZE_X32 = {value: value for value in scalar_types}
_EXPECTED_CANONICALIZE_X32[np.int64] = np.int32
_EXPECTED_CANONICALIZE_X32[np.uint64] = np.uint32
_EXPECTED_CANONICALIZE_X32[np.float64] = np.float32
_EXPECTED_CANONICALIZE_X32[np.complex128] = np.complex64
_EXPECTED_CANONICALIZE_X32[np.longlong] = np.int32

UINT_DTYPES = {
  4: jnp.uint4,
  8: np.uint8,
  16: np.uint16,
  32: np.uint32,
  64: np.uint64,
}

def identity(x):
  """A named identity function for use in tests"""
  return x


class DtypesTest(jtu.JaxTestCase):

  def test_canonicalize_type(self):
    expected = {
        True: _EXPECTED_CANONICALIZE_X64,
        False: _EXPECTED_CANONICALIZE_X32,
    }
    for in_dtype, expected_dtype in expected[config.enable_x64.value].items():
      self.assertEqual(dtypes.canonicalize_dtype(in_dtype), expected_dtype)

  @parameterized.named_parameters(
    {"testcase_name": f"_type={type_.__name__}", "type_": type_}
    for type_ in python_scalar_types)
  def testDefaultTypes(self, type_):
    expected_dtype = dtypes.canonicalize_dtype(dtypes.python_scalar_dtypes[type_])
    for f in [jnp.array, jax.jit(jnp.array), jax.jit(lambda x: x)]:
      y = f(type_(0))
      self.assertTrue(isinstance(y, jax.Array), msg=(f, y))
      self.assertEqual(y.dtype, expected_dtype, msg=(f, y))

  def testUnsupportedType(self):
    with self.assertRaisesRegex(TypeError, "nonsense.* not understood"):
      dtypes.canonicalize_dtype("nonsense")

  @parameterized.named_parameters(
    {"testcase_name": f"_{swap=}_{jit=}",
     "swap": swap, "jit": jit}
    for swap in [False, True] for jit in [False, True])
  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype.*")
  @jax.numpy_dtype_promotion('standard')
  def testBinaryPromotion(self, swap, jit):
    testcases = [
      (jnp.array(1.), 0., jnp.float64),
      (jnp.array(1.), jnp.array(0.), jnp.float64),
      (jnp.array(1.), jnp.array(0., dtype=jnp.float16), jnp.float16),
      (jnp.array(1.), jnp.array(0., dtype=jnp.float32), jnp.float32),
      (jnp.array(1.), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array(1., dtype=jnp.float16), 0., jnp.float16),
      (jnp.array(1., dtype=jnp.float32), 0., jnp.float32),
      (jnp.array(1., dtype=jnp.float64), 0., jnp.float64),
      (jnp.array(1., dtype=jnp.float16), jnp.array(0., dtype=jnp.float16), jnp.float16),
      (jnp.array(1., dtype=jnp.float16), jnp.array(0., dtype=jnp.float32), jnp.float32),
      (jnp.array(1., dtype=jnp.float16), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array(1., dtype=jnp.float32), jnp.array(0., dtype=jnp.float32), jnp.float32),
      (jnp.array(1., dtype=jnp.float32), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array(1., dtype=jnp.float64), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array([1.]), 0., jnp.float64),
      (jnp.array([1.]), jnp.array(0.), jnp.float64),
      (jnp.array([1.]), jnp.array(0., dtype=jnp.float16), jnp.float64),
      (jnp.array([1.]), jnp.array(0., dtype=jnp.float32), jnp.float64),
      (jnp.array([1.]), jnp.array(0., dtype=jnp.float64), jnp.float64),
      (jnp.array([1.], dtype=jnp.float32), jnp.array(0., dtype=jnp.float16), jnp.float32),
      (jnp.array([1.], dtype=jnp.float16), jnp.array(0., dtype=jnp.float32), jnp.float32),
      (jnp.array([1.], dtype=jnp.float16), 0., jnp.float16),
    ]
    op = jax.jit(operator.add) if jit else operator.add
    for x, y, dtype in testcases:
      x, y = (y, x) if swap else (x, y)
      z = op(x, y)
      self.assertTrue(isinstance(z, jax.Array), msg=(x, y, z))
      self.assertEqual(z.dtype, dtypes.canonicalize_dtype(dtype), msg=(x, y, z))

  @jax.numpy_dtype_promotion('strict')
  def testPromoteDtypesStrict(self):
    msg = ("Input dtypes .* have no available implicit dtype promotion "
           "path when jax_numpy_dtype_promotion=strict")

    assertTypePromotionError = functools.partial(
      self.assertRaisesRegex, dtypes.TypePromotionError, msg,
      dtypes.promote_types)

    # Check that strong types have diagonal promotion table:
    for t1 in all_dtypes:
      for t2 in all_dtypes:
        if t1 == t2:
          self.assertEqual(t1, dtypes.promote_types(t1, t2))
        else:
          assertTypePromotionError(t1, t2)

    # Promotion between weak types matches numpy promotion
    for t1 in [int, float, complex]:
      for t2 in [int, float, complex]:
        py_result = type(t1(0) + t2(0))
        # np.dtype(int) is int32 on Windows and int64 on Linux/Mac.
        py_result_dtype = (np.dtype(np.int64) if py_result is int
                           else np.dtype(py_result))
        lattice_dtype, lattice_weak_type = dtypes._lattice_result_type(t1, t2)
        self.assertTrue(lattice_weak_type)
        self.assertEqual(lattice_dtype, py_result_dtype)

    # Check that weak promotion only works if strong value is not cast:
    for t1 in bool_dtypes:
      assertTypePromotionError(t1, int)
      assertTypePromotionError(t1, float)
      assertTypePromotionError(t1, complex)
    for t1 in signed_dtypes + unsigned_dtypes:
      self.assertEqual(dtypes.promote_types(t1, int), t1)
      assertTypePromotionError(t1, float)
      assertTypePromotionError(t1, complex)
    for t1 in float_dtypes:
      self.assertEqual(dtypes.promote_types(t1, int), t1)
      self.assertEqual(dtypes.promote_types(t1, float), t1)
      assertTypePromotionError(t1, complex)
    for t1 in complex_dtypes:
      self.assertEqual(dtypes.promote_types(t1, int), t1)
      self.assertEqual(dtypes.promote_types(t1, float), t1)
      self.assertEqual(dtypes.promote_types(t1, complex), t1)

  @jax.numpy_dtype_promotion('standard')
  def testPromoteDtypesStandard(self):
    for t1 in all_dtypes:
      self.assertEqual(t1, dtypes.promote_types(t1, t1))

      self.assertEqual(t1, dtypes.promote_types(t1, np.bool_))
      # TODO(zhangqiaorjc): Consider more dtype promotion rules for fp8.
      if t1 in fp8_dtypes:
        continue
      if t1 in int4_dtypes:
        continue
      self.assertEqual(np.dtype(np.complex128),
                       dtypes.promote_types(t1, np.complex128))

      for t2 in all_dtypes:
        # TODO(zhangqiaorjc): Consider more dtype promotion rules for fp8.
        if t2 in fp8_dtypes:
          continue
        if t2 in int4_dtypes:
          continue
        # Symmetry
        self.assertEqual(dtypes.promote_types(t1, t2),
                         dtypes.promote_types(t2, t1))

    self.assertEqual(np.dtype(np.float32),
                     dtypes.promote_types(np.float16, dtypes.bfloat16))

    # Promotions of non-inexact types against inexact types always prefer
    # the inexact types.
    for t in float_dtypes + complex_dtypes:
      for i in bool_dtypes + signed_dtypes + unsigned_dtypes:
        # TODO(zhangqiaorjc): Consider more dtype promotion rules for fp8.
        if t in fp8_dtypes:
          continue
        if t in int4_dtypes or i in int4_dtypes:
          continue
        self.assertEqual(t, dtypes.promote_types(t, i))

    # Promotions between exact types, or between inexact types, match NumPy.
    for groups in [bool_dtypes + np_signed_dtypes + np_unsigned_dtypes,
                   np_float_dtypes + complex_dtypes]:
      for t1, t2 in itertools.combinations(groups, 2):
        self.assertEqual(np.promote_types(t1, t2),
                         dtypes.promote_types(t1, t2))

    # Promotion between weak types matches numpy promotion
    for t1 in [int, float, complex]:
      for t2 in [int, float, complex]:
        py_result = type(t1(0) + t2(0))
        # np.dtype(int) is int32 on Windows and int64 on Linux/Mac.
        py_result_dtype = (np.dtype(np.int64) if py_result is int
                           else np.dtype(py_result))
        lattice_dtype, lattice_weak_type = dtypes._lattice_result_type(t1, t2)
        self.assertTrue(lattice_weak_type)
        self.assertEqual(lattice_dtype, py_result_dtype)

  @parameterized.parameters([jnp.bool_, jnp.int32, jnp.bfloat16, jnp.float32, jnp.complex64])
  def testScalarInstantiation(self, scalar_type):
    a = scalar_type(1)
    self.assertEqual(a.dtype, jnp.dtype(scalar_type))
    self.assertIsInstance(a, jax.Array)
    self.assertEqual(0, jnp.ndim(a))
    self.assertIsInstance(np.dtype(scalar_type).type(1), scalar_type)

  def testIsSubdtype(self):
    for t in scalar_types:
      self.assertTrue(dtypes.issubdtype(t, t))
      self.assertTrue(dtypes.issubdtype(np.dtype(t).type, t))
      self.assertTrue(dtypes.issubdtype(t, np.dtype(t).type))
      self.assertTrue(dtypes.issubdtype(t, np.dtype(t)))
      if t in [jnp.int4, jnp.uint4, jnp.bfloat16]:
        # These dtype have no equivalent in NumPy.
        continue
      for category in [np.generic, jnp.inexact, jnp.integer, jnp.signedinteger,
                       jnp.unsignedinteger, jnp.floating, jnp.complexfloating]:
        self.assertEqual(dtypes.issubdtype(t, category),
                         np.issubdtype(np.dtype(t).type, category))
        self.assertEqual(dtypes.issubdtype(t, category),
                         np.issubdtype(np.dtype(t).type, category))

  def testIsSubdtypeExtended(self):
    self.assertTrue(dtypes.issubdtype(dtypes.extended, dtypes.extended))
    self.assertTrue(dtypes.issubdtype(dtypes.extended, np.generic))
    self.assertFalse(dtypes.issubdtype(dtypes.extended, np.number))

    self.assertTrue(jnp.issubdtype(dtypes.prng_key, dtypes.prng_key))
    self.assertTrue(jnp.issubdtype(dtypes.prng_key, dtypes.extended))
    self.assertTrue(jnp.issubdtype(dtypes.prng_key, np.generic))
    self.assertFalse(dtypes.issubdtype(dtypes.prng_key, np.number))

  @parameterized.product(dtype=custom_float_dtypes)
  def testIsSubdtypeCustomFloats(self, dtype):
    for dt in [dtype, np.dtype(dtype), str(np.dtype(dtype))]:
      self.assertTrue(dtypes.issubdtype(dt, dt))
      self.assertTrue(dtypes.issubdtype(dt, np.dtype(dtype)))
      self.assertTrue(dtypes.issubdtype(dt, str(np.dtype(dtype))))
      self.assertTrue(dtypes.issubdtype(dt, np.floating))
      self.assertTrue(dtypes.issubdtype(dt, np.inexact))
      self.assertTrue(dtypes.issubdtype(dt, np.number))
      self.assertTrue(dtypes.issubdtype(dt, np.generic))
      self.assertFalse(dtypes.issubdtype(dt, object))
      self.assertFalse(dtypes.issubdtype(dt, np.float64))
      self.assertFalse(dtypes.issubdtype(np.generic, dt))

  @parameterized.product(dtype=int4_dtypes)
  def testIsSubdtypeInt4(self, dtype):
    if dtype == 'int4':
      int_category = np.signedinteger
    elif dtype == 'uint4':
      int_category = np.unsignedinteger
    else:
      raise ValueError("Unexpected dtype: {dtype}")
    for dt in [dtype, np.dtype(dtype), str(np.dtype(dtype))]:
      self.assertTrue(dtypes.issubdtype(dt, dt))
      self.assertTrue(dtypes.issubdtype(dt, np.dtype(dtype)))
      self.assertTrue(dtypes.issubdtype(dt, str(np.dtype(dtype))))
      self.assertTrue(dtypes.issubdtype(dt, int_category))
      self.assertTrue(dtypes.issubdtype(dt, np.integer))
      self.assertTrue(dtypes.issubdtype(dt, np.number))
      self.assertTrue(dtypes.issubdtype(dt, np.generic))
      self.assertFalse(dtypes.issubdtype(dt, object))
      self.assertFalse(dtypes.issubdtype(dt, np.int64))
      self.assertFalse(dtypes.issubdtype(np.generic, dt))

  @jtu.sample_product(
      dtype=all_dtypes,
      kind=(*dtype_kinds, *all_dtypes)
  )
  def testIsDtype(self, dtype, kind):
    if isinstance(kind, np.dtype):
      expected = (dtype == kind)
    else:
      expected = (dtype in dtype_kinds[kind])
    self.assertEqual(expected, dtypes.isdtype(dtype, kind))

  def testArrayCasts(self):
    for t in [jnp.bool_, jnp.int32, jnp.bfloat16, jnp.float32, jnp.complex64]:
      a = np.array([1, 2.5, -3.7])
      self.assertEqual(a.astype(t).dtype, jnp.dtype(t))
      self.assertEqual(jnp.array(a).astype(t).dtype, jnp.dtype(t))

  def testEnumPromotion(self):
    class AnEnum(enum.IntEnum):
      A = 42
      B = 101
    np.testing.assert_equal(np.array(42), np.array(AnEnum.A))
    np.testing.assert_equal(jnp.array(42), jnp.array(AnEnum.A))
    np.testing.assert_equal(np.int32(101), np.int32(AnEnum.B))
    np.testing.assert_equal(jnp.int32(101), jnp.int32(AnEnum.B))

  def testScalarCastInsideJitWorks(self):
    # jnp.int32(tracer) should work.
    self.assertEqual(jnp.int32(101),
                     jax.jit(lambda x: jnp.int32(x))(jnp.float32(101.4)))

  @parameterized.parameters(python_scalar_types)
  def testDtypeFromScalarType(self, typ):
    self.assertEqual(dtypes.dtype(typ), dtypes.python_scalar_dtypes[typ])

  @parameterized.parameters(python_scalar_types)
  def testDtypeFromScalarValue(self, typ):
    self.assertEqual(dtypes.dtype(typ(0)), dtypes.python_scalar_dtypes[typ])

  @parameterized.parameters(all_dtypes)
  def testDtypeFromValue(self, dtype):
    self.assertEqual(dtypes.dtype(dtype.type(0)), dtype)

  @parameterized.parameters(all_dtypes)
  def testDtypeFromDtype(self, dtype):
    self.assertEqual(dtypes.dtype(dtype), dtype)

  @parameterized.parameters(all_dtypes)
  def testDtypeFromString(self, dtype):
    self.assertEqual(dtypes.dtype(str(dtype)), dtype)

  def testDtypeFromNone(self):
    with self.assertRaisesRegex(ValueError, "Invalid argument to dtype"):
      dtypes.dtype(None)

  def testDefaultDtypes(self):
    precision = config.default_dtype_bits.value
    assert precision in ['32', '64']
    self.assertEqual(dtypes.bool_, np.bool_)
    self.assertEqual(dtypes.int_, np.int32 if precision == '32' else np.int64)
    self.assertEqual(dtypes.uint, np.uint32 if precision == '32' else np.uint64)
    self.assertEqual(dtypes.float_, np.float32 if precision == '32' else np.float64)
    self.assertEqual(dtypes.complex_, np.complex64 if precision == '32' else np.complex128)

  def test_custom_tangent_dtype(self):
    from jax._src import core

    class scale(dtypes.extended):
      pass

    class ScalesTyRules:
      @staticmethod
      def physical_element_aval(dtype) -> core.ShapedArray:
        return core.ShapedArray((), dtype.float_dtype)

      @staticmethod
      def global_sharded_result_handler(aval, sharding, committed, is_from_xla):
        raise NotImplementedError("convert back under the jit")

      @staticmethod
      def add(dt, x, y):
        fromscale = partial(jax.lax.convert_element_type, new_dtype=dt.float_dtype)
        toscale = partial(jax.lax.convert_element_type, new_dtype=dt)
        return toscale(jax.lax.max(fromscale(x), fromscale(y)))

      @staticmethod
      def zero(dt):
        neginf = np.array(-np.inf if dtypes.supports_inf(dt.float_dtype)
                          else dtypes.finfo(dt.float_dtype).min, dt.float_dtype)
        return jax.lax.convert_element_type(neginf, dt)

      @staticmethod
      def convert_from(dtype, other_dtype) -> bool:
        return dtype.float_dtype == other_dtype

      @staticmethod
      def convert_to(other_dtype, dtype) -> bool:
        return dtype.float_dtype == other_dtype

    @dataclasses.dataclass(frozen=True)
    class ScaleTy(dtypes.ExtendedDType):
      float_dtype: dtypes.DType
      name: str = 'scale'
      _rules: type = ScalesTyRules
      type: type = scale

    @jax.custom_vjp
    def g(x):
      return x
    def g_fwd(x):
      return x, None
    def g_bwd(_, ct):
      ct = jax.lax.convert_element_type(ct, ScaleTy(dtypes.float8_e5m2))
      return ct,
    g.defvjp(g_fwd, g_bwd)

    @jax.custom_vjp
    def convert(x):
      return x
    def convert_fwd(x):
      return x, None
    def convert_bwd(_, ct):
      ct = jax.lax.convert_element_type(ct, ct.dtype.float_dtype)
      return ct,
    convert.defvjp(convert_fwd, convert_bwd)

    @jax.jit
    def f(x):
      x = convert(x)
      x = g(x) + g(x)
      return x

    x = jnp.array(3., dtypes.float8_e5m2)
    out = jax.grad(f)(x)
    self.assertAllClose(out, 1., check_dtypes=False)
    self.assertTrue(dtypes.issubdtype(ScaleTy(dtypes.float8_e5m2), scale))

  def test_custom_tangent_dtype_with_scan(self):
    from jax._src import core

    class ScalesTyRules:
      # tell JAX how to lower this dtype to an HLO dtype
      @staticmethod
      def physical_element_aval(dtype) -> core.ShapedArray:
        return core.ShapedArray((), dtype.float_dtype)

      # allow conversions to and from the corresponding float type
      @staticmethod
      def convert_from(scale_dtype, other_dtype) -> bool:
        return scale_dtype.float_dtype == other_dtype

      @staticmethod
      def convert_to(other_dtype, scale_dtype) -> bool:
        return scale_dtype.float_dtype == other_dtype

      # define how autodiff should accumulate these values
      @staticmethod
      def add(dt, x, y):
        fromscale = partial(jax.lax.convert_element_type, new_dtype=dt.float_dtype)
        toscale = partial(jax.lax.convert_element_type, new_dtype=dt)
        return toscale(jax.lax.max(fromscale(x), fromscale(y)))

      @staticmethod
      def zero(dt):
        neginf = np.array(-np.inf if dtypes.supports_inf(dt.float_dtype)
                          else dtypes.finfo(dt.float_dtype).min, dt.float_dtype)
        return jax.lax.convert_element_type(neginf, dt)

      @staticmethod
      def tangent_dtype(dtype):
        return dtype

      # NOTE: by skipping some rules, this dtype can only be used underneath jit

    # class to use as second argument to jax.dtypes.issubdtype
    class scale_dtype(dtypes.extended): pass

    # parameterized datatype for use in e.g. jax.lax.convert_element_type
    @dataclasses.dataclass(frozen=True)
    class scale_dtype(dtypes.ExtendedDType):
      float_dtype: dtypes.DType
      _rules: type = ScalesTyRules
      type: type = scale_dtype

      def __repr__(self) -> str:
        nbits = dtypes.finfo(self.float_dtype).bits
        return f'scale{nbits}'
      name = property(__repr__)

    f32 = jnp.dtype('float32')
    sc32 = scale_dtype(f32)

    def outer(x, scale_f32):
      scale = jax.lax.convert_element_type(scale_f32, sc32)
      def body_fun(carry, _):
        carry = inner(carry, scale)
        return carry, None
      x, _ = jax.lax.scan(body_fun, x, None, length=3)
      return x

    @jax.custom_vjp
    def inner(carry, scale):
      del scale  # only used under autodiff
      return carry
    def inner_fwd(carry, scale):
      return inner(carry, scale), scale
    def inner_bwd(prev_scale, grads):
      # convert sc32->f32 so we can do math
      prev_scale_f32 = jax.lax.convert_element_type(prev_scale, f32)
      new_scale_f32 = 1./2 * (prev_scale_f32 + jnp.max(grads))
      # convert f32->sc32 so the autodiff system accumulates scales correctly
      new_scale = jax.lax.convert_element_type(new_scale_f32, sc32)
      return grads, new_scale
    inner.defvjp(inner_fwd, inner_bwd)

    scale = jnp.float32(1.)
    _, new_scale = jax.jit(jax.grad(outer, (0, 1)))(jnp.float32(3.14), scale)
    self.assertAllClose(new_scale, jnp.float32(1.0))

  def test_check_dtype_non_hashable(self):
    # regression test for issue with checking non-hashable custom dtype
    class MyDtype:
      __hash__ = None
      dtype = np.dtype('float32')
    dtypes.check_user_dtype_supported(MyDtype())

  def test_check_dtype_array(self):
    x = jnp.arange(4)
    msg = "Passing an array as a dtype argument is deprecated"
    with self.assertWarnsRegex(DeprecationWarning, msg):
      dtypes.check_user_dtype_supported(x)
    with self.assertWarnsRegex(DeprecationWarning, msg):
      jax.jit(dtypes.check_user_dtype_supported)(x)


class EArrayTest(jtu.JaxTestCase):

  @parameterized.parameters([True, False])
  def test_extended_dtypes_at_rest(self, jit):
    # Test a trivial isomorphic-to-float32 extended dtype working with EArray
    from jax._src import core
    from jax._src.interpreters import pxla

    class foo(dtypes.extended): pass

    class FooTyRules:

      @staticmethod
      def convert_to(foo_dtype, target_dtype):
        return True

      @staticmethod
      def physical_element_aval(foo_dtype):
        return core.ShapedArray((), dtypes.dtype('float32'))

      @staticmethod
      def global_sharded_result_handler(aval, out_sharding, committed):
        phys_sharding = out_sharding  # unlike KeyTyRules, assume same shape
        phys_aval = core.physical_aval(aval)
        phys_handler_maker = pxla.global_result_handlers[core.ShapedArray]
        phys_handler = phys_handler_maker(phys_aval, phys_sharding, committed)
        return lambda bufs: earray.EArray(aval, phys_handler(bufs))

    @dataclasses.dataclass(frozen=True)
    class FooTy(dtypes.ExtendedDType):
      name: str = 'foo'
      _rules: type = FooTyRules
      type: type = foo

    # Can we make one?
    def f(x):
      return jax.lax.convert_element_type(x, FooTy())
    if jit:
      f = jax.jit(f)
    x = f(jnp.arange(3, dtype='float32'))  # don't crash
    self.assertIsInstance(x.dtype, FooTy)

    # Can we consume one?
    def g(x):
      self.assertIsInstance(x.dtype, FooTy)
      return x
    if jit:
      g = jax.jit(g)
    y = g(x)
    self.assertIsInstance(y.dtype, FooTy)


class TestPromotionTables(jtu.JaxTestCase):

  @parameterized.named_parameters(
      {"testcase_name": f"_{jaxtype=}", "jaxtype": jaxtype}
      for jaxtype in dtypes._jax_types + dtypes._weak_types)
  def testJaxTypeFromType(self, jaxtype):
    self.assertIs(dtypes._jax_type(*dtypes._dtype_and_weaktype(jaxtype)), jaxtype)

  @parameterized.named_parameters(
      {"testcase_name": f"_{jaxtype=}", "jaxtype": jaxtype}
      for jaxtype in dtypes._jax_types + dtypes._weak_types)
  def testJaxTypeFromVal(self, jaxtype):
    try:
      val = jaxtype(0)
    except TypeError:
      val = jaxtype.type(0)
    self.assertIs(dtypes._jax_type(*dtypes._dtype_and_weaktype(val)), jaxtype)

  @parameterized.named_parameters(
      {"testcase_name": f"_{dtype=}", "dtype": dtype}
      for dtype in dtypes._jax_types)
  def testJaxTypeWeak(self, dtype):
    jax_type = dtypes._jax_type(dtype, weak_type=True)
    if dtypes.issubdtype(jax_type, np.complexfloating):
      self.assertIs(jax_type, complex)
    elif dtypes.issubdtype(jax_type, np.floating):
      self.assertIs(jax_type, float)
    elif dtypes.issubdtype(jax_type, np.integer):
      self.assertIs(jax_type, int)
    else:
      self.assertIs(jax_type, np.dtype(bool))

  @parameterized.named_parameters(
      {"testcase_name": f"_{typ}", "typ": typ}
       for typ in [bool, int, float, complex])
  def testScalarWeakTypes(self, typ):
    # Regression test for https://github.com/google/jax/issues/11377
    val = typ(0)

    result1 = jnp.array(val)
    result2 = jax.jit(jnp.array)(val)
    self.assertEqual(result1.aval, result2.aval)

    with jax.numpy_dtype_promotion('standard'):
      f = lambda x: x / 2
      result1 = jnp.array(f(val))
      result2 = jax.jit(f)(val)
    self.assertEqual(result1.aval, result2.aval)

  def testResultTypeNone(self):
    # This matches the behavior of np.result_type(None) => np.float64
    self.assertEqual(dtypes.result_type(None), dtypes.canonicalize_dtype(dtypes.float_))

  def testResultTypeWeakFlag(self):
    float_ = dtypes.canonicalize_dtype(dtypes.float_)
    x_weak = jnp.array(1.)
    x_strong = x_weak.astype(float_)
    self.assertEqual(dtypes.result_type(x_weak), float_)
    self.assertEqual(dtypes.result_type(x_weak, return_weak_type_flag=True), (float_, True))
    self.assertEqual(dtypes.result_type(x_strong), float_)
    self.assertEqual(dtypes.result_type(x_strong, return_weak_type_flag=True), (float_, False))

  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype.*")
  @jax.numpy_dtype_promotion('standard')
  def testObservedPromotionTable(self):
    """Test that the weak & strong dtype promotion table does not change over time."""
    # Note: * here refers to weakly-typed values
    typecodes = \
        ['b1','u1','u2','u4','u8','i1','i2','i4','i8','bf','f2','f4','f8','c4','c8','i*','f*','c*']
    if config.enable_x64.value:
      expected = [
        ['b1','u1','u2','u4','u8','i1','i2','i4','i8','bf','f2','f4','f8','c4','c8','i*','f*','c*'],
        ['u1','u1','u2','u4','u8','i2','i2','i4','i8','bf','f2','f4','f8','c4','c8','u1','f*','c*'],
        ['u2','u2','u2','u4','u8','i4','i4','i4','i8','bf','f2','f4','f8','c4','c8','u2','f*','c*'],
        ['u4','u4','u4','u4','u8','i8','i8','i8','i8','bf','f2','f4','f8','c4','c8','u4','f*','c*'],
        ['u8','u8','u8','u8','u8','f*','f*','f*','f*','bf','f2','f4','f8','c4','c8','u8','f*','c*'],
        ['i1','i2','i4','i8','f*','i1','i2','i4','i8','bf','f2','f4','f8','c4','c8','i1','f*','c*'],
        ['i2','i2','i4','i8','f*','i2','i2','i4','i8','bf','f2','f4','f8','c4','c8','i2','f*','c*'],
        ['i4','i4','i4','i8','f*','i4','i4','i4','i8','bf','f2','f4','f8','c4','c8','i4','f*','c*'],
        ['i8','i8','i8','i8','f*','i8','i8','i8','i8','bf','f2','f4','f8','c4','c8','i8','f*','c*'],
        ['bf','bf','bf','bf','bf','bf','bf','bf','bf','bf','f4','f4','f8','c4','c8','bf','bf','c4'],
        ['f2','f2','f2','f2','f2','f2','f2','f2','f2','f4','f2','f4','f8','c4','c8','f2','f2','c4'],
        ['f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f8','c4','c8','f4','f4','c4'],
        ['f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','c8','c8','f8','f8','c8'],
        ['c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c8','c4','c8','c4','c4','c4'],
        ['c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8','c8'],
        ['i*','u1','u2','u4','u8','i1','i2','i4','i8','bf','f2','f4','f8','c4','c8','i*','f*','c*'],
        ['f*','f*','f*','f*','f*','f*','f*','f*','f*','bf','f2','f4','f8','c4','c8','f*','f*','c*'],
        ['c*','c*','c*','c*','c*','c*','c*','c*','c*','c4','c4','c4','c8','c4','c8','c*','c*','c*'],
      ]
    else:
      expected = [
        ['b1','u1','u2','u4','u4','i1','i2','i4','i4','bf','f2','f4','f4','c4','c4','i*','f*','c*'],
        ['u1','u1','u2','u4','u4','i2','i2','i4','i4','bf','f2','f4','f4','c4','c4','u1','f*','c*'],
        ['u2','u2','u2','u4','u4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','u2','f*','c*'],
        ['u4','u4','u4','u4','u4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','u4','f*','c*'],
        ['u4','u4','u4','u4','u4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','u4','f*','c*'],
        ['i1','i2','i4','i4','i4','i1','i2','i4','i4','bf','f2','f4','f4','c4','c4','i1','f*','c*'],
        ['i2','i2','i4','i4','i4','i2','i2','i4','i4','bf','f2','f4','f4','c4','c4','i2','f*','c*'],
        ['i4','i4','i4','i4','i4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','i4','f*','c*'],
        ['i4','i4','i4','i4','i4','i4','i4','i4','i4','bf','f2','f4','f4','c4','c4','i4','f*','c*'],
        ['bf','bf','bf','bf','bf','bf','bf','bf','bf','bf','f4','f4','f4','c4','c4','bf','bf','c4'],
        ['f2','f2','f2','f2','f2','f2','f2','f2','f2','f4','f2','f4','f4','c4','c4','f2','f2','c4'],
        ['f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','c4','c4','f4','f4','c4'],
        ['f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','c4','c4','f4','f4','c4'],
        ['c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4'],
        ['c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4','c4'],
        ['i*','u1','u2','u4','u4','i1','i2','i4','i4','bf','f2','f4','f4','c4','c4','i*','f*','c*'],
        ['f*','f*','f*','f*','f*','f*','f*','f*','f*','bf','f2','f4','f4','c4','c4','f*','f*','c*'],
        ['c*','c*','c*','c*','c*','c*','c*','c*','c*','c4','c4','c4','c4','c4','c4','c*','c*','c*'],
      ]
    typecode_to_dtype = {
      'b1': jnp.bool_,
      'u1': jnp.uint8, 'u2': jnp.uint16, 'u4': jnp.uint32, 'u8': jnp.uint64,
      'i1': jnp.int8, 'i2': jnp.int16, 'i4': jnp.int32, 'i8': jnp.int64,
      'bf': jnp.bfloat16, 'f2': jnp.float16, 'f4': jnp.float32, 'f8': jnp.float64,
      'c4': jnp.complex64, 'c8': jnp.complex128,
      'i*': jnp.int64, 'f*': jnp.float64, 'c*': jnp.complex128,
    }
    dtype_to_typecode = {jnp.dtype(v): k for k, v in typecode_to_dtype.items()
                        if not k.endswith('*')}

    def typecode_to_val(typecode):
      weak_type = typecode.endswith('*')
      dtype = typecode_to_dtype[typecode]
      val = dtype(0)
      if weak_type:
        val = val.item()
      return val

    def val_to_typecode(val):
      dtype = dtypes.result_type(val)
      weak_type = dtypes.is_weakly_typed(val)
      typecode = dtype_to_typecode[dtype]
      if weak_type:
        typecode = typecode[:-1] + '*'
      return typecode

    vals = [typecode_to_val(t) for t in typecodes]
    table = [[val_to_typecode(v1 + v2) for v1 in vals] for v2 in vals]

    def show_differences(expected, actual):
      diffs = ""
      for i, t1 in enumerate(typecodes):
        for j, t2 in enumerate(typecodes):
          if expected[i][j] != actual[i][j]:
            diffs += f"\n{t1}, {t2} -> want {expected[i][j]}, got {actual[i][j]}"
      return diffs

    self.assertEqual(table, expected, show_differences(expected, table))

  @parameterized.named_parameters(
    {"testcase_name": "_xtype={}_ytype={}_xfun={}_yfun={}".format(
      xtype.__name__, ytype.__name__, xfun.__name__, yfun.__name__),
     "xtype": xtype, "ytype": ytype, "xfun": xfun, "yfun": yfun}
    for xtype, ytype in itertools.product(
      [int, float, jnp.int16, jnp.int32, jnp.float16, jnp.float32], repeat=2)
    for xfun, yfun in itertools.product(
      [identity, abs, jnp.array], repeat=2)
    )
  @jax.numpy_dtype_promotion('standard')
  def testBinaryPromotionJitInvariance(self, xtype, ytype, xfun, yfun):
    """Test jit invariance of simple binary promotion rules with and without weak types."""
    f = lambda x, y: xfun(x) + yfun(y)
    args_maker = lambda: [xtype(1), ytype(1)]
    self._CompileAndCheck(f, args_maker, check_dtypes=True)

  @parameterized.named_parameters(
    {"testcase_name": f"_{dtype=}_{weak_type=}",
     "dtype": dtype, "weak_type": weak_type}
    for dtype in all_dtypes
    for weak_type in [True, False]
  )
  def testUnaryPromotion(self, dtype, weak_type):
    # Regression test for https://github.com/google/jax/issues/6051
    if dtype in int4_dtypes:
      self.skipTest("XLA support for int4 is incomplete.")
    x = lax_internal._convert_element_type(0, dtype, weak_type=weak_type)
    if weak_type:
      expected = dtypes.canonicalize_dtype(
        dtypes._default_types['f' if x.dtype in ["bfloat16", *fp8_dtypes] else x.dtype.kind])
    else:
      expected = x.dtype
    self.assertEqual(dtypes.result_type(x), expected)

  @jax.numpy_dtype_promotion('standard')
  def testFloat8PromotionError(self):
    for dtype in fp8_dtypes:
      x = jnp.array(1, dtype=dtype)
      y = jnp.array(1, dtype='float32')
      with self.assertRaisesRegex(dtypes.TypePromotionError,
                                  ".*8-bit floats do not support implicit promotion"):
        x + y

  @jax.numpy_dtype_promotion('standard')
  @jtu.run_on_devices("tpu")
  def testInt4PromotionError(self):
    for dtype in int4_dtypes:
      x = jnp.array(1, dtype=dtype)
      y = jnp.array(1, dtype='int32')
      with self.assertRaisesRegex(dtypes.TypePromotionError,
                                  ".*4-bit integers do not support implicit promotion"):
        x + y

  @jtu.sample_product(
    dtype=all_dtypes,
    weak_type=[True, False],
    promotion=['standard', 'strict'],
  )
  def testBinaryNonPromotion(self, dtype, weak_type, promotion):
    if dtype in fp8_dtypes:
      self.skipTest("XLA support for float8 is incomplete.")
    if dtype in int4_dtypes:
      self.skipTest("XLA support for int4 is incomplete.")
    # Regression test for https://github.com/google/jax/issues/6051
    x = lax_internal._convert_element_type(0, dtype, weak_type=weak_type)
    with jax.numpy_dtype_promotion(promotion):
      y = (x + x)

    if promotion == 'standard' or not weak_type or dtype == dtypes.bool_:
      expected_dtype = dtype
    elif dtypes.issubdtype(dtype, np.complexfloating):
      expected_dtype = np.complex128
    elif dtypes.issubdtype(dtype, np.floating):
      expected_dtype = np.float64
    else:
      expected_dtype = np.int64

    # No boolean weak types.
    expected_weak_type = weak_type and dtype != bool
    expected_dtype = dtypes.canonicalize_dtype(expected_dtype)

    self.assertEqual(y.dtype, expected_dtype)
    self.assertEqual(dtypes.is_weakly_typed(y), expected_weak_type)

  @parameterized.named_parameters(
    {"testcase_name": f"_{dtype=}_{weak_type=}",
     "dtype": dtype, "weak_type": weak_type}
    for dtype in all_dtypes
    for weak_type in [True, False]
  )
  def testArrayRepr(self, dtype, weak_type):
    if dtype in int4_dtypes and not jtu.test_device_matches(["tpu"]):
      self.skipTest("XLA support for int4 is incomplete.")
    val = lax_internal._convert_element_type(0, dtype, weak_type=weak_type)
    rep = repr(val)
    self.assertStartsWith(rep, 'Array(')
    if weak_type:
      self.assertEndsWith(rep, f"dtype={val.dtype.name}, weak_type=True)")
    else:
      self.assertEndsWith(rep, f"dtype={val.dtype.name})")

  @jtu.sample_product(
      input_dtype=jtu.dtypes.all + [bool, int, float, complex],
      output_dtype=jtu.dtypes.all + [bool, int, float, complex],
      numpy_dtype_promotion=['strict', 'standard']
  )
  def testSafeToCast(self, input_dtype, output_dtype, numpy_dtype_promotion):
    with jax.numpy_dtype_promotion(numpy_dtype_promotion):
      # First the special cases which are always safe:
      always_safe = (
        (input_dtype == output_dtype) or
        (dtypes.issubdtype(output_dtype, np.integer) and input_dtype in {int}) or
        (dtypes.issubdtype(output_dtype, np.floating) and input_dtype in {int, float}) or
        (dtypes.issubdtype(output_dtype, np.complexfloating) and input_dtype in {int, float, complex})
      )
      if always_safe:
        self.assertTrue(dtypes.safe_to_cast(input_dtype, output_dtype))

      try:
        result_dtype = dtypes.result_type(input_dtype, dtypes.canonicalize_dtype(output_dtype))
      except dtypes.TypePromotionError:
        result_dtype = None

      if result_dtype is None and input_dtype != output_dtype:
        with self.assertRaises(dtypes.TypePromotionError):
          dtypes.safe_to_cast(input_dtype, output_dtype)
      else:
        self.assertEqual(dtypes.result_type(output_dtype) == result_dtype,
                         dtypes.safe_to_cast(input_dtype, output_dtype))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
