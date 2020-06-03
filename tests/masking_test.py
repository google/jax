# Copyright 2018 Google LLC
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
import itertools as it
from unittest import SkipTest

import numpy as np
from absl.testing import absltest, parameterized

import jax
from jax.interpreters.masking import (shape_as_value, parse_spec, ShapeError,
  Poly, Mon, eval_polymorphic_shape, remap_ids, UniqueIds, finalize_spec)
from jax import (numpy as jnp, test_util as jtu, mask, vmap, jit, grad, lax,
  core as jc, shapecheck, tree_flatten, tree_unflatten, tree_map)
from jax.util import safe_map, safe_zip, unzip2
from jax.config import config
from jax.lax.lax import _identity
from jax.numpy.lax_numpy import _polymorphic_slice_indices
from jax.random import uniform, PRNGKey
from jax.scipy.special import expit
from jax.test_util import rand_default
from operator import add, sub
import scipy.stats

config.parse_flags_with_absl()

map = safe_map
zip = safe_zip


# TODO:
# These should be only the 'manual' tests for masking.
# Move the more exhaustive, systematic tests into lax_test.py.

def constant_poly(c):
  return Poly({Mon(): c})

class MaskingTest(jtu.JaxTestCase):

  @parameterized.parameters([
      ['(m, n)', 'ShapeSpec(m, n)'],
      ['(m * n)', 'ShapeSpec(m n)'],
      ['m * n', 'ShapeSpec(m n)'],
      ['(m * n,)', 'ShapeSpec(m n)'],
      ['(3, m)', 'ShapeSpec(3, m)'],
      ['(10, m)', 'ShapeSpec(10, m)'],
      ['(-10, m)', 'ShapeSpec(-10, m)'],
      ['(3 * m)', 'ShapeSpec(3 m)'],
      ['m', 'ShapeSpec(m)'],
      ['', 'ShapeSpec()'],
      ['n + -1*n', 'ShapeSpec(0)'],
      ['m + n', 'ShapeSpec(m + n)'],
      ['m + n * k', 'ShapeSpec(k n + m)'],
      ['m + 3 * k', 'ShapeSpec(3 k + m)'],
      ['-3 + k + k * k', 'ShapeSpec(k**2 + k + -3)'],
      ['', 'ShapeSpec()'],
      ['_', 'ShapeSpec(_)'],
  ])
  def test_parse_spec(self, spec, ans):
    self.assertEqual(str(parse_spec(spec)), ans)
    self.assertEqual(str(remap_ids(UniqueIds(), parse_spec(spec))), ans)

  def test_Poly_equal(self):
    assert constant_poly(3) == 3
    assert np.array(3, np.int64) == constant_poly(3)
    assert np.array(3, np.int64)[()] == constant_poly(3)
    assert not np.array(3, np.int64) != constant_poly(3)
    assert constant_poly(4) != 3
    assert 3 == constant_poly(3)
    assert 4 != constant_poly(3)
    assert constant_poly(4) == constant_poly(4)
    assert constant_poly(3) != constant_poly(4)
    assert Poly({Mon(): 3, Mon({'n': 1}): 4}) == Poly({Mon({'n': 1}): 4, Mon(): 3})
    assert Poly({Mon(): 3, Mon({'n': 1}): 4}) != Poly({Mon(): 3, Mon({'n': 2}): 4})
    assert Poly({Mon(): 3, Mon({'m': 1}): 4}) != Poly({Mon(): 3, Mon({'n': 1}): 4})

  def test_Poly_hash(self):
    assert not len(set(hash(Poly({Mon(): i})) for i in range(10))) == 1
    assert (hash(Poly({Mon(): 3, Mon({'n': 1}): 4}))
            == hash(Poly({Mon({'n': 1}): 4, Mon(): 3})))

  def test_Mon_hash(self):
    assert not len(set(hash(Mon({'a': i})) for i in range(10))) == 1
    assert hash(Mon({'a': 1, 'b': 1})) == hash(Mon({'b': 1, 'a': 1}))

  def test_Poly_compare(self):
    poly = Poly({Mon(): 3, Mon({'n': 1}): 4})
    # Assume poly > 0 to make various shape rules work with polymorphic shapes:
    assert poly >= 0
    assert poly >= 1
    assert poly > 0

    assert 0 <= poly
    assert 0 < poly
    assert constant_poly(3) >= 1
    assert constant_poly(3) > 1
    self.assertRaisesRegex(ValueError, "", lambda: poly >= 2)
    self.assertRaisesRegex(ValueError, "", lambda: poly > 1)

  def test_Poly_divmod(self):
    n = Poly({Mon({'n': 1}): 1})
    assert (n, 1) == divmod(2*n+1, 2)
    assert (2*n, 0) == divmod(10*n, 5)
    assert (2*n+4, 3) == divmod(10*n+23, 5)

  def test_Poly_rsub(self):
    n = Poly({Mon({'n': 1}): 1})
    assert -1 - n == -n - 1

class MaskingTest(jtu.JaxTestCase):
  def test_sum(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return jnp.sum(x)

    ans = padded_sum([jnp.array([3, 1, 4, 1, 5])], dict(n=3))
    expected = 8
    self.assertAllClose(ans, expected, check_dtypes=False)

    ans = padded_sum([jnp.array([3, 1, 4, 1, 5])], dict(n=4))
    expected = 9
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_sum_vmap(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return jnp.sum(x)

    ans = vmap(padded_sum)([jnp.ones((5, 10))], dict(n=jnp.arange(5)))
    expected = np.array([0, 1, 2, 3, 4])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def check(self, fun, in_shapes, out_shape, logical_env, padded_in_shapes,
            dtypes, rng, rtol=None, atol=None):
    shapecheck(in_shapes, out_shape)(fun)
    masked_fun = mask(fun, in_shapes, out_shape)
    padded_args = [rng(shape, dtype)
                   for shape, dtype in zip(padded_in_shapes, dtypes)]
    padded_outs, outs_tree = tree_flatten(masked_fun(padded_args, logical_env))

    out_specs, _ = tree_flatten(out_shape)
    out_specs = map(parse_spec, out_specs)
    out_specs = map(finalize_spec, out_specs, map(np.shape, padded_outs))
    logical_out_shapes = [eval_polymorphic_shape(s, logical_env)
                          for s in out_specs]
    logical_out_slices = [tuple(map(slice, s)) for s in logical_out_shapes]
    logical_outs = [o[s] for o, s in zip(padded_outs, logical_out_slices)]

    in_specs = map(parse_spec, in_shapes)
    in_specs = map(finalize_spec, in_specs, padded_in_shapes)
    logical_in_shapes = [eval_polymorphic_shape(s, logical_env)
                         for s in in_specs]
    logical_in_slices = [tuple(map(slice, s)) for s in logical_in_shapes]
    logical_args = [a[s] for a, s in zip(padded_args, logical_in_slices)]
    logical_outs_expected, _ = tree_flatten(fun(*logical_args))
    self.assertAllClose(logical_outs, logical_outs_expected, check_dtypes=True,
                        atol=atol, rtol=rtol)

    padded_outs_jit, _ = tree_flatten(jit(masked_fun)(padded_args, logical_env))
    self.assertAllClose(padded_outs_jit, padded_outs, check_dtypes=True,
                        atol=atol, rtol=rtol)


  def test_add(self):
    self.check(add, ['n', ''], 'n', {'n': 3}, [(4,), ()], ['float_', 'float_'],
               rand_default(self.rng()))
    self.check(add, ['n', ''], dict(n=jnp.array([2, 3])), 'n')
    self.check(add, ['n', 'n'], dict(n=jnp.array([2, 3])), 'n')

    addvecs = mask(add, in_shapes=['n', 'n'], out_shape='n')

    x = jnp.array([3, 1, 4, 1, 5, 9])
    y = jnp.array([2, 6, 5, 3, 5, 8])
    ans = addvecs([x, y], dict(n=3))
    expected = np.array([5, 7, 9])
    self.assertAllClose(ans[:3], expected, check_dtypes=False)

    thunk = lambda: addvecs([jnp.arange(5), jnp.arange(6)], dict(n=3))
    self.assertRaisesRegex(ShapeError, "", thunk)

    raise SkipTest
    self.check(add, ['(m, n)', 'n'], dict(m=np.array([2, 3]), n=np.array([4, 4])), '(m, n)', unpadded_vars=['n'])

  def test_scan(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    ans = cumsum([jnp.array([5, 2, 9, 1, 4])], dict(n=3))
    expected = 16
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_scan_vmap(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    ans = vmap(cumsum)([jnp.arange(6).reshape(2, 3)], dict(n=jnp.array([1, 2])))
    expected = np.array([0, 7])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_scan_jit(self):
    @partial(mask, in_shapes=['n'], out_shape='')
    def cumsum(arr):
      out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
      return out

    @jit
    def jit_cumsum(args, shape_env):
      assert python_should_be_executing
      return cumsum(args, shape_env)

    python_should_be_executing = True
    ans = jit_cumsum([jnp.array([5, 2, 9, 1, 4])], dict(n=3))
    expected = 16
    self.assertAllClose(ans, expected, check_dtypes=False)

    python_should_be_executing = False
    ans = jit_cumsum([jnp.array([5, 2, 9, 1, 4])], dict(n=4))
    expected = 17
    self.assertAllClose(ans, expected, check_dtypes=False)

    python_should_be_executing = False
    ans = jit_cumsum([jnp.array([5, 2, 9, 1, 4])], dict(n=1))
    expected = 5
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_mean(self):
    self.check(lambda x: jnp.sum(x) / shape_as_value(x.shape)[0], ['n'],
               dict(n=jnp.array([2, 3])), '')

  def test_arithmetic(self):
    @partial(mask, in_shapes=['(n, m)', 'm'], out_shape='(n, m)')
    def times(x, y):
      return x * y

    # TODO(shoyer): enable this check when broadcast_in_dim supports masking
    with self.assertRaisesRegex(
        NotImplementedError,
        'Masking rule for broadcast_in_dim not implemented yet.'):
      ans = times([jnp.array([[1, 2], [3, 4], [5, 6]]), jnp.array([1, 2])],
                  dict(n=4, m=5))
      # expected = np.array([[1, 2, 3], [8, 10, 12]])
      # self.assertAllClose(ans, expected, check_dtypes=False)

  def test_stack(self):
    @partial(mask, in_shapes=['n','n'], out_shape='(2, n)')
    def stack(x, y):
      return jnp.stack([x, y], 0)

    # TODO(shoyer): enable this check when broadcast_in_dim supports masking
    with self.assertRaisesRegex(
        NotImplementedError,
        'Masking rule for broadcast_in_dim not implemented yet.'):
      ans = stack([jnp.array([1, 2, 3]), jnp.array([4, 5, 6])], dict(n=10))
      # expected = np.array([[1, 2, 3], [4, 5, 6]])
      # self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='')
    def padded_sum(x):
      return jnp.sum(x)

    ans = padded_sum([jnp.array([[3, 4], [5, 6]])], dict(n=1))
    expected = 8
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic2(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='n')
    def padded_sum(x):
      return jnp.sum(x, axis=0)

    ans = padded_sum([jnp.array([[3, 4], [5, 6]])], dict(n=2))
    expected = jnp.array([8, 10])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_monomorphic3(self):
    @partial(mask, in_shapes=['(_, n)'], out_shape='_')
    def padded_sum(x):
      return jnp.sum(x, axis=1)

    ans = padded_sum([jnp.array([[3, 4], [5, 6]])], dict(n=1))
    expected = jnp.array([3, 5])
    self.assertAllClose(ans, expected, check_dtypes=False)

    @shapecheck(['(2*n, n)'], '_, n')
    def identity(x):
      return x

  def test_rnn(self):
    n = 3

    @partial(mask, in_shapes=['(_, _)', '(t, _)'], out_shape='_')
    def rnn(W, xs):
      def step(h, x):
        new_h = jnp.dot(W, h) + jnp.dot(W, x)
        return new_h, ()
      predicted, _ = lax.scan(step, jnp.zeros(n), xs)
      return predicted

    rng = np.random.RandomState(0)
    W = jnp.eye(n)
    xs = rng.randn(10, n).astype(jnp.float_)
    ans = rnn([W, xs], dict(t=4))
    expected = xs[:4].sum(0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_rnn_grad(self):
    n = 3

    @partial(mask, in_shapes=['(_, _)', '(t, _)', '_'], out_shape='')
    def rnn(W, xs, target):
      def step(h, x):
        new_h = jnp.tanh(jnp.dot(W, h) + jnp.dot(W, x))
        return new_h, ()
      predicted, _ = lax.scan(step, jnp.zeros(n), xs)
      return jnp.sum((predicted - target)**2)

    rng = np.random.RandomState(0)
    W = rng.randn(n, n).astype(jnp.float_)
    xs = rng.randn(10, n).astype(jnp.float_)
    y = rng.randn(n).astype(jnp.float_)

    ans = grad(lambda W: rnn([W, xs, y], dict(t=4)))(W)

    def rnn_reference(W, xs, target):
      h = jnp.zeros(n)
      for x in xs:
        h = jnp.tanh(jnp.dot(W, h) + jnp.dot(W, x))
      predicted = h
      return jnp.sum((predicted - target)**2)

    expected = grad(lambda W: rnn_reference(W, xs[:4], y))(W)

    self.assertAllClose(ans, expected, check_dtypes=False)

  def test_ragged_batched_rnn(self):
    n = 3

    @partial(mask, in_shapes=('(_, _)', '(t, _)', '_'), out_shape='')
    def rnn(W, xs, target):
      def step(h, x):
        new_h = jnp.tanh(jnp.dot(W, h) + jnp.dot(W, x))
        return new_h, ()
      predicted, _ = lax.scan(step, jnp.zeros(n), xs)
      return jnp.sum((predicted - target)**2)

    rng = np.random.RandomState(0)
    W = rng.randn(n, n).astype(jnp.float_)
    seqs = rng.randn(3, 10, n).astype(jnp.float_)
    ts = jnp.array([2, 5, 4])
    ys = rng.randn(3, n)

    ans = grad(lambda W: vmap(rnn, ((None, 0, 0), 0))((W, seqs, ys), dict(t=ts)).sum())(W)

    def rnn_reference(W, seqs, targets):
      total_loss = jnp.array(0, jnp.float_)
      for xs, target in zip(seqs, targets):
        h = jnp.zeros(n)
        for x in xs:
          h = jnp.tanh(jnp.dot(W, h) + jnp.dot(W, x))
        predicted = h
        total_loss = total_loss + jnp.sum((predicted - target)**2)
      return total_loss

    seqs_ = [xs[:t] for xs, t in zip(seqs, ts)]
    expected = grad(lambda W: rnn_reference(W, seqs_, ys).sum())(W)

    self.assertAllClose(
      ans, expected, check_dtypes=False,
      rtol=2e-2 if jtu.device_under_test() == "tpu" else 1e-5)

  def test_concatenate(self):
    self.check(lambda x, y, z: lax.concatenate([x, y, z], 0),
               ['n', 'm', 'n'], dict(n=jnp.array((1, 2)), m=jnp.array((2, 3))), 'm + 2 * n')

  def test_dot(self):
    self.check(lambda x, y: lax.dot(x, y), ['(m, k)', '(k, n)'],
               dict(m=jnp.array([2, 3]), k=jnp.array([2, 3]), n=jnp.array([2, 3])), '(m, n)')
    self.check(lambda A, b: jnp.dot(A, b), ['(m, n)', 'n'],
               dict(m=jnp.array([2, 3]), n=jnp.array([2, 3])), 'm')

    def thunk():
      self.check(lambda A, b: lax.dot_general(A, b, [((0,), (0,)), ((), ())]),
                 ['(m, n)', 'n'], dict(m=2, n=2), 'm')
    self.assertRaisesRegex(TypeError, "", thunk)

  def test_jit(self):
    raise SkipTest
    @partial(mask, in_shapes=['n'], out_shape='2*n')
    @jit
    def duplicate(x):
      assert python_should_be_executing
      return lax.concatenate([x, x], 0)

    python_should_be_executing = True
    out = duplicate([jnp.arange(3)], dict(n=2))
    assert np.all(np.array([0, 1, 0, 1]) == out[:4])

    python_should_be_executing = False
    out = duplicate([jnp.arange(3)], dict(n=2))
    assert np.all(np.array([0, 1, 0, 1]) == out[:4])

  def test_device_put(self):
    raise SkipTest
    self.check(lambda x: jax.device_put(x), ['n'], dict(n=jnp.array([2, 3])), 'n')

  @parameterized.named_parameters({
                                    'testcase_name': "padding_config={}_shapes={}".format(
                                      padding_config, shape),
                                    'padding_config': padding_config,
                                    'shape': shape}
                                  for padding_config, shape in (
                                          (((1, 2, 0),), (2,)),
                                          (((1, 2, 0), (3, 4, 0)), (1, 2)),
                                          (((0, 0, 0), (0, 0, 0)), (1, 2)),
                                          (((1, 2, 3),), (2,)),
                                          (((1, 2, 1), (3, 4, 2)), (3, 2)),
                                          (((-1, 2, 0),), (2,)),
                                          (((-1, -2, 0), (1, 2, 0)), (4, 2)),
                                          (((-1, 2, 0), (1, 2, 2)), (4, 2)),
                                          (((-1, -2, 2),), (5,)),
                                          (((-1, -2, 1), (1, 2, 2)), (4, 2))))
  def test_pad(self, padding_config, shape):
    def pad(x):
      return lax.pad(x, jnp.array(1., x.dtype), padding_config)

    flat = len(shape) == 1
    value_dict = dict(
      [('h', jnp.array([shape[0], shape[0] + 1]))] +
      ([] if flat else [('w', jnp.array([shape[1], shape[1] + 1]))]))
    self.check(pad, ['h' if flat else '(h,w)'], value_dict)

  def test_pad_check_out_shape(self):
    self.check(lambda x: lax.pad(x, jnp.array(0., x.dtype), [(1, 1, 1)]),
               ['n'], dict(n=jnp.array([2, 3])), '2*n+1')

  def test_numpy_pad(self):
    raise SkipTest
    def numpy_pad(x):
      return jnp.pad(x, (0, 1), constant_values=5.)

    self.check(numpy_pad, ['n'], dict(n=jnp.array([2, 3])), 'n+1')

  @parameterized.named_parameters(jtu.cases_from_list(
    {
      'testcase_name': "strides={}_padding={}_lhs_dilation={}_dimension_numbers"
                       "={}_lhs_perm={}_rhs_perm={}_out_perm={}".format(
        strides, padding, lhs_dilation, dimension_numbers, lhs_perm, rhs_perm, out_perm),
      'strides': strides, 'padding': padding, 'lhs_dilation': lhs_dilation,
      'dimension_numbers': dimension_numbers, 'lhs_perm': lhs_perm,
      'rhs_perm': rhs_perm, 'out_perm': out_perm}
    for strides in [(1, 1), (2, 1)]
    for padding in ['SAME', 'VALID', ((0, 1), (2, 0))]
    for lhs_dilation in (None, (1, 2))
    for dimension_numbers, (lhs_perm, rhs_perm, out_perm) in (
            (("NCHW", "OIHW", "NCHW"), ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3))),
            (("NHWC", "HWIO", "NHWC"), ((0, 2, 3, 1), (2, 3, 1, 0), (0, 2, 3, 1))),
            (("NCHW", "HWIO", "NHWC"), ((0, 1, 2, 3), (2, 3, 1, 0), (0, 2, 3, 1)))
    )
    # String padding is not implemented for transposed convolution, see conv_general_dilated implementation:
    if (lhs_dilation is None or not isinstance(padding, str)) and
    # only test strides with same padding:
    (strides[0] == 1 or padding == 'SAME')))
  def test_conv(self, strides, padding, lhs_dilation,
                dimension_numbers, lhs_perm, rhs_perm, out_perm):
    valid = padding == 'VALID'
    is_strided = strides[0] != 1
    lhs_shape = '({}, {}, {}, {})'.format(*np.take(['n', 'i', '2*h' if is_strided else 'h', 'w'], lhs_perm))
    rhs_shape = '({}, {}, {}, {})'.format(*np.take(['o', 'i', '2', '3'], rhs_perm))
    out_shape = '({}, {}, {}, {})'.format(*np.take([
      'n', 'o', 'h+-1' if valid and not is_strided else 'h',
      ('w+-2' if valid else 'w') if lhs_dilation is None else '2*w+-1'], out_perm))

    def conv(lhs, rhs):
      return lax.conv_general_dilated(
        lhs, rhs, strides, padding,
        lhs_dilation=lhs_dilation, dimension_numbers=dimension_numbers)

    self.check(conv, [lhs_shape, rhs_shape],
               dict(n=jnp.array([1, 1]), i=jnp.array([3, 3]), o=jnp.array([2, 2]),
                    h=jnp.array([1, 2]), w=jnp.array([2, 3])),
               out_shape, unpadded_vars=['n', 'i', 'o'])

  def test_indexing(self):
    raise SkipTest
    self.check(lambda x: x[0], ['n'], dict(n=jnp.array([2, 3])), '')
    self.check(lambda x: x[-1], ['n'], dict(n=jnp.array([2, 3])), '')

  def test_slicing(self):
    raise SkipTest
    self.check(lambda x: x[1:], ['n'], dict(n=jnp.array([2, 3])), 'n+-1')
    self.check(lambda x: x[:-1], ['n'], dict(n=jnp.array([2, 3])), 'n+-1')
    self.check(lambda x: x[..., -1], ['(n,3)'], dict(n=jnp.array([2, 3])), 'n')
    self.check(lambda x: x[:x.shape[0] - 1], ['n'], dict(n=jnp.array([2, 3])), 'n+-1')
    # TODO: self.check(lambda x: x[x.shape[0] - 1:], ['n'], dict(n=jnp.array([2, 3])), '1')

  def test_rev(self):
    @shapecheck(['n'], 'n+-1')
    def rev(x):
      return x[:0:-1]

    @shapecheck(['n'], 'n+-1')
    def rev(x):
      return x[-2::-1]

    # TODO implement masking for rev_p:
    # self.check(lambda x: x[:0:-1], ['n'], dict(n=jnp.array([2, 3])), 'n+-1')
    # self.check(lambda x: x[-2::-1], ['n'], dict(n=jnp.array([2, 3])), 'n+-1')

  def test_lax_slice(self):
    self.check(lambda x: lax.slice(x, (1,), (x.shape[0],)), ['n'],
               dict(n=jnp.array([2, 3])), 'n+-1')
    # TODO: self.check(lambda x: lax.slice(x, (x.shape[0] // 2,), (x.shape[0],)), ['2*n'], dict(n=jnp.array([2, 3])), 'n')

  def test_reshape(self):
    raise SkipTest

  def test_transpose(self):
    self.check(lambda x: jnp.transpose(x, (1, 0, 2)),
               ['(a, b, c)'], dict(a=jnp.array([2, 3]), b=jnp.array([2, 3]), c=jnp.array([3, 2])), 'b, a, c')

  def test_arange(self):
    raise SkipTest
    self.check(lambda x: -jnp.arange(x.shape[0]), ['n'],
               dict(n=jnp.array([2, 3])), 'n')

  def test_eye(self):
    raise SkipTest
    self.check(lambda x: -jnp.eye(x.shape[0], 2 * x.shape[0]), ['n'],
               dict(n=jnp.array([2, 3])), 'n, 2*n')

  def test_tri(self):
    raise SkipTest
    self.check(lambda x: -jnp.tri(x.shape[0], 2 * x.shape[0]), ['n'],
               dict(n=jnp.array([2, 3])), 'n, 2*n')

  def test_delta(self):
    raise SkipTest
    self.check(lambda x: -lax._delta(jnp.float32, (x.shape[0], 2 * x.shape[0], 3 * x.shape[0]), axes=(0, 1)), ['n'],
               dict(n=jnp.array([2, 3])), 'n, 2*n, 3*n')

  def test_sum_2d(self):
    self.check(lambda x: jnp.sum(x), ['(m, n)'],
               dict(m=jnp.array([2, 3]), n=jnp.array([2, 3])), '')

  def test_expit(self):
    raise SkipTest("custom_jvp doesn't work with masking yet")

    self.check(lambda x: expit(x), ['n'], dict(n=jnp.array([2, 3])), 'n')

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}".format(dtype), "dtype": np.dtype(dtype).name}
    for dtype in [np.float32, np.float64]))
  def test_uniform(self, dtype):
    raise SkipTest("not yet implemented")
    # TODO needs fix for https://github.com/google/jax/issues/2155

  def test_zeros(self):
    raise SkipTest
    self.check(lambda x: -jnp.zeros(x.shape), ['n'],
               dict(n=jnp.array([2, 3])), 'n')

  def test_ones(self):
    raise SkipTest
    self.check(lambda x: -jnp.ones(x.shape), ['n'],
               dict(n=jnp.array([2, 3])), 'n')

  def test_broadcast_to(self):
    raise SkipTest
    self.check(lambda x: -jnp.broadcast_to(0, x.shape), ['n'],
               dict(n=jnp.array([2, 3])), 'n')

  def test_broadcast_in_dim(self):
    raise SkipTest
    self.check(lambda x: -lax.broadcast_in_dim(jnp.zeros((1, 1)), shape=(3, x.shape[0], 4), broadcast_dimensions=(1, 2)),
               ['(n, 1)'], dict(n=jnp.array([2, 3])), '(3, n, 4)')

  def test_destructure(self):
    def d(key):
      key1, key2 = key
      return key1

    self.check(d, ['2'], dict(), '')

  def test_where(self):
    raise SkipTest
    self.check(lambda x: jnp.where(x < 0, x, jnp.zeros_like(x)), ['n'],
               dict(n=jnp.array([2, 3])), 'n')

    message = (
      "mask(jit(broadcast_in_dim))) is not supported yet. "
      "Consider using jit(mask(broadcast_in_dim)) instead."
      "If you are using jnp.where, consider disabling jit on jax.lax._where or "
      "manually broadcasting arguments to the same shape.")

    self.assertRaisesWithLiteralMatch(NotImplementedError, message,
                                      lambda: self.check(lambda x: jnp.where(x < 0, x, 0.), ['n'], dict(n=jnp.array([2, 3])), 'n'))
    self.assertRaisesWithLiteralMatch(NotImplementedError, message,
                                      lambda: self.check(lambda x: jnp.where(x < 0, 0., x), ['n'], dict(n=jnp.array([2, 3])), 'n'))
    self.assertRaisesWithLiteralMatch(NotImplementedError, message,
                                      lambda: self.check(lambda x: jnp.where(x < 0, 0., 0.), ['n'], dict(n=jnp.array([2, 3])), 'n'))

  def test_split(self):
    raise SkipTest
    self.check(lambda x: jnp.split(x, 2), ['2*n'],
               dict(n=jnp.array([4, 4])), ['n', 'n'], unpadded_vars=['n'])
    self.check(lambda x: jnp.split(x, [10]), ['n'],
               dict(n=jnp.array([12, 12])), ['10', 'n+-10'], unpadded_vars=['n'])

  @parameterized.named_parameters(jtu.cases_from_list([{
    'testcase_name': "operator={}".format(operator.__name__), 'operator': operator}
    for operator in [jnp.sum, jnp.prod, jnp.max, jnp.min]]))
  def test_reduce(self, operator):
    self.check(operator, ['(m, n)'],
               dict(m=jnp.array([3, 3]), n=jnp.array([3, 3])), '', unpadded_vars=['m', 'n'])

  def test_output_shape_error(self):
    def thunk():
      self.check(lambda x: x, ['n'], dict(n=jnp.array([3, 3])), 'n+-1')

    message = "Output shapes should be (n + -1,) but are (n,)."
    self.assertRaisesWithLiteralMatch(ShapeError, message, thunk)
    self.assertRaisesWithLiteralMatch(ShapeError, message, partial(thunk))

    def thunk():
      self.check(lambda x: (x, x),
                 ['n'], dict(n=jnp.array([2, 2])), ['7*n', 'n'], unpadded_vars=['n'])

    message = "Output shapes should be [(7 n,), (n,)] but are ((n,), (n,))."
    self.assertRaisesWithLiteralMatch(ShapeError, message, thunk)
    self.assertRaisesWithLiteralMatch(ShapeError, message, partial(thunk))

  def test_output_tree_error(self):
    def thunk():
      self.check(lambda x: [x, x], ['n'], dict(n=jnp.array([3, 3])), ('n', 'n'), unpadded_vars=['n'])
    message = "Output shapes should be ((n,), (n,)) but are [(n,), (n,)]."
    self.assertRaisesWithLiteralMatch(ShapeError, message, thunk)
    self.assertRaisesWithLiteralMatch(ShapeError, message, partial(thunk))

  def test_unsupported_op(self):
    p = jc.Primitive('unsupported_op')
    p.def_abstract_eval(_identity)
    p.def_impl(lambda x: x)

    def thunk():
      self.check(lambda x: p.bind(x), ['n'], dict(n=jnp.array([2, 3])), 'n')

    message = "Masking rule for unsupported_op not implemented yet."
    self.assertRaisesWithLiteralMatch(NotImplementedError, message, thunk)

  def test_nesting(self):
    raise SkipTest("not yet implemented")

    @partial(mask, in_shapes=['n'], out_shape='')
    def padded_sum(x):
      return jnp.sum(x)

    batched_sum = vmap(padded_sum)

    @partial(mask, in_shapes=['(m, _)', 'm'], out_shape='')
    def fun(x, ns):
      return batched_sum([x], dict(n=ns)).sum()

    x = jnp.array([[3, 1, 4, 1],
                  [5, 9, 2, 6],
                  [5, 3, 5, 8]])
    ns = jnp.array([2, 3, 2])
    ans = fun([x, ns], dict(m=2))
    expected = 3+1 + 5+9+2
    self.assertAllClose(ans, expected, check_dtypes=False)


  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_start={}_stop={}_step={}_length={}"
       .format(start, stop, step, length),
       "start": start, "stop": stop, "step": step, "length": length}
      for length in range(1, 5)
      for start, stop, step
      in it.product(it.chain([None], range(-10, 10)), repeat=3)
      if step != 0))
  def test_slice_indices(self, start, stop, step, length):
    s = slice(start, stop, step)
    assert _polymorphic_slice_indices(s, length) == s.indices(length)

  def test_slice_index_poly_start(self):
    n = Poly({Mon({'n': 1}): 1})
    s = slice(n, None, None)
    assert (n, 2 * n, 1) == _polymorphic_slice_indices(s, 2 * n)


  def test_slice_oob_indexing(self):
    # https://github.com/google/jax/issues/2245
    self.assertAllClose(jnp.ones(5), jnp.ones(5)[:10])
    self.assertAllClose(jnp.ones(5), jnp.ones(5)[-10:])

if __name__ == '__main__':
  absltest.main()
