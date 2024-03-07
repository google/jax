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

from absl.testing import absltest
import numpy as np

import jax
from jax import numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class VectorizeTest(jtu.JaxTestCase):

  @jtu.sample_product(
    [dict(left_shape=left_shape, right_shape=right_shape,
          result_shape=result_shape)
      for left_shape, right_shape, result_shape in [
          ((2, 3), (3, 4), (2, 4)),
          ((2, 3), (1, 3, 4), (1, 2, 4)),
          ((1, 2, 3), (1, 3, 4), (1, 2, 4)),
          ((5, 2, 3), (1, 3, 4), (5, 2, 4)),
          ((6, 5, 2, 3), (3, 4), (6, 5, 2, 4)),
      ]
    ],
  )
  def test_matmat(self, left_shape, right_shape, result_shape):
    matmat = jnp.vectorize(jnp.dot, signature='(n,m),(m,k)->(n,k)')
    self.assertEqual(matmat(jnp.zeros(left_shape),
                            jnp.zeros(right_shape)).shape, result_shape)

  @jtu.sample_product(
    [dict(left_shape=left_shape, right_shape=right_shape,
          result_shape=result_shape)
      for left_shape, right_shape, result_shape in [
          ((2, 3), (3,), (2,)),
          ((2, 3), (1, 3), (1, 2)),
          ((4, 2, 3), (1, 3), (4, 2)),
          ((5, 4, 2, 3), (1, 3), (5, 4, 2)),
      ]
    ],
  )
  def test_matvec(self, left_shape, right_shape, result_shape):
    matvec = jnp.vectorize(jnp.dot, signature='(n,m),(m)->(n)')
    self.assertEqual(matvec(jnp.zeros(left_shape),
                            jnp.zeros(right_shape)).shape, result_shape)

  @jtu.sample_product(
    [dict(left_shape=left_shape, right_shape=right_shape,
          result_shape=result_shape)
      for left_shape, right_shape, result_shape in [
          ((3,), (3,), ()),
          ((2, 3), (3,), (2,)),
          ((4, 2, 3), (3,), (4, 2)),
      ]
    ],
  )
  def test_vecmat(self, left_shape, right_shape, result_shape):
    vecvec = jnp.vectorize(jnp.dot, signature='(m),(m)->()')
    self.assertEqual(vecvec(jnp.zeros(left_shape),
                            jnp.zeros(right_shape)).shape, result_shape)

  @jtu.sample_product(
    [dict(shape=shape, result_shape=result_shape)
      for shape, result_shape in [
          ((3,), ()),
          ((2, 3,), (2,)),
          ((1, 2, 3,), (1, 2)),
      ]
    ],
  )
  def test_magnitude(self, shape, result_shape):
    size = 1
    for x in shape:
        size *= x
    inputs = jnp.arange(size).reshape(shape)

    @partial(jnp.vectorize, signature='(n)->()')
    def magnitude(x):
      return jnp.dot(x, x)

    self.assertEqual(magnitude(inputs).shape, result_shape)

  @jtu.sample_product(
    [dict(shape=shape, result_shape=result_shape)
      for shape, result_shape in [
          ((3,), ()),
          ((2, 3), (2,)),
          ((1, 2, 3, 4), (1, 2, 3)),
      ]
    ],
  )
  def test_mean(self, shape, result_shape):
    mean = jnp.vectorize(jnp.mean, signature='(n)->()')
    self.assertEqual(mean(jnp.zeros(shape)).shape, result_shape)

  @jtu.sample_product(
    [dict(shape=shape, result_shape=result_shape)
      for shape, result_shape in [
          ((), (2,)),
          ((3,), (3,2,)),
      ]
    ],
  )
  def test_stack_plus_minus(self, shape, result_shape):

    @partial(jnp.vectorize, signature='()->(n)')
    def stack_plus_minus(x):
      return jnp.stack([x, -x])

    self.assertEqual(stack_plus_minus(jnp.zeros(shape)).shape, result_shape)

  def test_center(self):

    @partial(jnp.vectorize, signature='(n)->(),(n)')
    def center(array):
      bias = jnp.mean(array)
      debiased = array - bias
      return bias, debiased

    b, a = center(jnp.arange(3.0))
    self.assertEqual(a.shape, (3,))
    self.assertEqual(b.shape, ())
    self.assertAllClose(1.0, b, check_dtypes=False)

    b, a = center(jnp.arange(6.0).reshape(2, 3))
    self.assertEqual(a.shape, (2, 3))
    self.assertEqual(b.shape, (2,))
    self.assertAllClose(jnp.array([1.0, 4.0]), b, check_dtypes=False)

  def test_exclude_first(self):

    @partial(jnp.vectorize, excluded={0})
    def f(x, y):
      assert x == 'foo'
      assert y.ndim == 0
      return y

    x = jnp.arange(3)
    self.assertAllClose(x, f('foo', x))
    self.assertAllClose(x, jax.jit(f, static_argnums=0)('foo', x))

  def test_exclude_second(self):

    @partial(jnp.vectorize, excluded={1})
    def f(x, y):
      assert x.ndim == 0
      assert y == 'foo'
      return x

    x = jnp.arange(3)
    self.assertAllClose(x, f(x, 'foo'))
    self.assertAllClose(x, jax.jit(f, static_argnums=1)(x, 'foo'))

  def test_exclude_kwargs(self):
    @partial(np.vectorize, excluded=(2, 'func'))
    def f_np(x, y, func=np.add):
      assert np.ndim(x) == np.ndim(y) == 0
      return func(x, y)

    @partial(jnp.vectorize, excluded=(2, 'func'))
    def f_jnp(x, y, func=jnp.add):
      assert x.ndim == y.ndim == 0
      return func(x, y)

    x = np.arange(4, dtype='int32')
    y = np.int32(2)

    self.assertArraysEqual(f_np(x, y), f_jnp(x, y))
    self.assertArraysEqual(f_np(x, y, np.power), f_jnp(x, y, jnp.power))
    self.assertArraysEqual(f_np(x, y, func=np.power), f_jnp(x, y, func=jnp.power))

  def test_exclude_errors(self):
    with self.assertRaisesRegex(
        TypeError, "jax.numpy.vectorize can only exclude"):
      jnp.vectorize(lambda x: x, excluded={1.5})

    with self.assertRaisesRegex(
        ValueError, r"excluded=\{-1\} contains negative numbers"):
      jnp.vectorize(lambda x: x, excluded={-1})

  def test_bad_inputs(self):
    matmat = jnp.vectorize(jnp.dot, signature='(n,m),(m,k)->(n,k)')
    with self.assertRaisesRegex(
        TypeError, "wrong number of positional arguments"):
      matmat(jnp.zeros((3, 2)))
    with self.assertRaisesRegex(
        ValueError,
        r"input with shape \(2,\) does not have enough dimensions"):
      matmat(jnp.zeros((2,)), jnp.zeros((2, 2)))
    with self.assertRaisesRegex(
        ValueError, r"inconsistent size for core dimension 'm'"):
      matmat(jnp.zeros((2, 3)), jnp.zeros((4, 5)))

  def test_wrong_output_type(self):
    f = jnp.vectorize(jnp.dot, signature='(n,m),(m,k)->(n,k),()')
    with self.assertRaisesRegex(
        TypeError, "output must be a tuple"):
      f(jnp.zeros((2, 2)), jnp.zeros((2, 2)))

  def test_wrong_num_outputs(self):
    f = jnp.vectorize(lambda *args: args, signature='(),()->(),(),()')
    with self.assertRaisesRegex(
        TypeError, "wrong number of output arguments"):
      f(1, 2)

  def test_wrong_output_shape(self):
    f = jnp.vectorize(jnp.dot, signature='(n,m),(m,k)->(n)')
    with self.assertRaisesRegex(
        ValueError, r"output shape \(2, 2\) does not match"):
      f(jnp.zeros((2, 2)), jnp.zeros((2, 2)))

  def test_inconsistent_output_size(self):
    f = jnp.vectorize(jnp.dot, signature='(n,m),(m,k)->(n,n)')
    with self.assertRaisesRegex(
        ValueError, r"inconsistent size for core dimension 'n'"):
      f(jnp.zeros((2, 3)), jnp.zeros((3, 4)))

  def test_expand_dims_multiple_outputs_no_signature(self):
    f = jnp.vectorize(lambda x: (x, x))
    x = jnp.arange(1)
    xx = f(x)
    self.assertAllClose(xx[0], x)
    self.assertAllClose(xx[1], x)
    self.assertIsInstance(xx, tuple)

  def test_none_arg(self):
    f = jnp.vectorize(lambda x, y: x if y is None else x + y)
    x = jnp.arange(10)
    self.assertAllClose(f(x, None), x)

    y = jnp.arange(10, 20)
    self.assertAllClose(f(x, y), x + y)

  def test_none_arg_bad_signature(self):
    f = jnp.vectorize(lambda x, y: x if y is None else x + y,
                      signature='(k),(k)->(k)')
    args = jnp.arange(10), None
    msg = r"Cannot pass None at locations \{1\} with signature='\(k\),\(k\)->\(k\)'"
    with self.assertRaisesRegex(ValueError, msg):
      f(*args)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
