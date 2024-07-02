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

from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
import itertools as it
from typing import Any, TypeVar, Union

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax._src import core
from jax._src import dtypes
from jax._src import test_util as jtu
from jax import lax
from jax._src.lax import parallel
from jax import random
from jax import jit, grad, jvp, vjp, make_jaxpr, jacfwd, jacrev, hessian
from jax import vmap
from jax.interpreters import batching
from jax.tree_util import register_pytree_node

jax.config.parse_flags_with_absl()


# These are 'manual' tests for batching (vmap). The more exhaustive, more
# systematic tests are in lax_test.py's LaxVmapTest class.

class BatchingTest(jtu.JaxTestCase):

  def testConstantFunction(self):
    ans = vmap(lambda x: 3)(np.ones(4))
    expected = 3 * np.ones(4)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jax.default_matmul_precision("float32")
  def testNestedBatchingMatMat(self):
    matvec = vmap(jnp.vdot, in_axes=(0, None))
    matmat = vmap(matvec, in_axes=(None, 1), out_axes=1)

    R = self.rng().randn
    A = R(4, 3)
    B = R(3, 2)

    ans = matmat(A, B)
    expected = np.dot(A, B)
    self.assertAllClose(ans, expected, check_dtypes=False)

    jaxpr = make_jaxpr(matmat)(A, B)
    self.assertLen(jaxpr.jaxpr.eqns, 1)

  def testPerExampleGradients(self):
    def predict(params, inputs):
      for W, b in params:
        outputs = jnp.dot(W, inputs) + b
        inputs = jnp.tanh(outputs)
      return outputs

    def loss(params, data):
      inputs, targets = data
      predictions = predict(params, inputs)
      return jnp.sum((predictions - targets)**2)

    batch_size = 5
    layer_sizes = [3, 2, 4]

    R = self.rng().randn
    params = [(R(m, n), R(m))
              for m, n in zip(layer_sizes[1:], layer_sizes[:-1])]

    input_batch = R(5, 3)
    target_batch = R(5, 4)
    batch = (input_batch, target_batch)

    ans = vmap(partial(grad(loss), params))(batch)

    for ans_pair, param_pair in zip(ans, params):
      dW, db = ans_pair
      W, b = param_pair

      self.assertEqual(dW.shape, (batch_size,) + W.shape)
      self.assertEqual(db.shape, (batch_size,) + b.shape)

  @jax.default_matmul_precision("float32")
  def testJacobians(self):
    def jacbwd(f, x):
      y, pullback = vjp(f, x)
      std_basis = np.eye(np.size(y)).reshape((-1,) + np.shape(y))
      jac_flat, = vmap(pullback, out_axes=np.ndim(y))(std_basis)
      return jac_flat.reshape(np.shape(y) + np.shape(x))

    def jacfwd(f, x):
      pushfwd = lambda v: jvp(f, (x,), (v,))
      std_basis = np.eye(np.size(x)).reshape((-1,) + np.shape(x))
      y, jac_flat = vmap(pushfwd, out_axes=(None, 0))(std_basis)
      return jac_flat.reshape(np.shape(y) + np.shape(x))

    R = self.rng().randn

    A = R(4, 3)
    b = R(4)
    f = lambda x: jnp.tanh(jnp.dot(A, x) + b)

    x = R(3)
    self.assertAllClose(jacfwd(f, x), jacbwd(f, x), check_dtypes=False)

  def testBatchOfCompile(self):
    side = []

    @jit
    def f(x):
      side.append(None)
      return x + x

    g = jit(vmap(f))
    self.assertAllClose(g(np.ones(2)), 2 * np.ones(2), check_dtypes=False)
    self.assertEqual(len(side), 1)
    self.assertAllClose(g(2 * np.ones(2)), 4 * np.ones(2),
                        check_dtypes=False)
    self.assertEqual(len(side), 1)

  def testSliceLax(self):
    fun = lambda x: lax.slice(x, (2,), (4,))
    R = self.rng().randn
    x = R(5, 10)

    ans = vmap(fun)(x)
    expected_ans = x[:, 2:4]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testSliceNumpy(self):
    fun = lambda x: x[:, 2]
    R = self.rng().randn
    x = R(10, 5, 3, 7)

    ans = vmap(fun)(x)
    expected_ans = x[:, :, 2]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testRevLax(self):
    fun = lambda x: lax.rev(x, [0])
    R = self.rng().randn
    x = R(2, 3)

    ans = vmap(fun)(x)
    expected_ans = x[:, ::-1]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    ans = vmap(fun, (1,), 1)(x)
    expected_ans = x[::-1, :]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testRevNumpy(self):
    fun = lambda x: x[:, ::-1]
    R = self.rng().randn
    x = R(3, 2, 4)

    ans = vmap(fun)(x)
    expected_ans = x[:, :, ::-1]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    ans = vmap(fun, (1,), 1)(x)
    expected_ans = x[:, :, ::-1]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    ans = vmap(fun, (2,), 2)(x)
    expected_ans = x[:, ::-1, :]
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testNpMaximum(self):
    fun = lambda x: jnp.maximum(x, 0.0)
    R = self.rng().randn
    x = R(10, 5, 3, 7)

    ans = vmap(fun)(x)
    expected_ans = np.maximum(x, 0.0)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testNpGtrThan(self):
    R = self.rng().randn
    x = R(10, 5, 3, 7)

    ans = vmap(lambda x: x > 1.0)(x)
    expected_ans = x > 1.0
    self.assertAllClose(ans, expected_ans)

  @jax.default_matmul_precision("float32")
  def testNpMaximumPerExampleGrad(self):
    R = self.rng().randn
    x = R(10, 5)
    W = R(5, 5)

    fun = lambda W, x: jnp.sum(jnp.maximum(jnp.dot(x, W), 0.0) ** 2)

    ans = vmap(partial(grad(fun), W))(x)

    W_t = jnp.transpose(W)
    for i in range(10):
      x_ex = x[i:i + 1]

      expected_ans = 2.0 * jnp.dot(
          jnp.maximum(jnp.dot(W_t, jnp.transpose(x_ex)), 0.0), x_ex)
      expected_ans = jnp.transpose(expected_ans)

      self.assertAllClose(ans[i], expected_ans, check_dtypes=False)

  # Replace the default TF32 with float32 in order to make it pass on A100
  @jax.default_matmul_precision("float32")
  def testDotGeneral(self):
    R = self.rng().randn

    x = R(10, 3, 4, 5)
    y = R(10, 3, 5, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun)(x, y)
    expected = lax.dot_general(x, y, [((3,), (2,)), ((0, 1), (0, 1))])
    self.assertAllClose(ans, expected)

    x = R(3, 4, 10, 5)
    y = R(3, 10, 5, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun, in_axes=(2, 1))(x, y)
    expected = np.stack([fun(x[..., i, :], y[:, i, ...]) for i in range(10)])
    self.assertAllClose(ans, expected)

    x = R(3, 4, 5, 10)
    y = R(3, 5, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun, in_axes=(3, None))(x, y)
    expected = np.stack([fun(x[..., i], y) for i in range(10)])
    self.assertAllClose(ans, expected)

    x = R(3, 4, 5)
    y = R(3, 5, 10, 6)
    fun = lambda x, y: lax.dot_general(x, y, [((2,), (1,)), ((0,), (0,))])
    ans = vmap(fun, in_axes=(None, 2))(x, y)
    expected = np.stack([fun(x, y[..., i, :]) for i in range(10)])
    self.assertAllClose(ans, expected)

    x = R(4)
    y = R(4, 10)
    fun = lambda x, y: lax.dot_general(x, y, [((0,), (0,)), ((), ())])
    ans = vmap(fun, in_axes=(None, 1))(x, y)
    expected = np.stack([fun(x, y[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected)

  def testDot(self):
    # these tests are based on @shoyer's notebook studying gufuncs

    def vecvec(a, b):
      dot = jnp.dot
      for ndim in range(1, max(a.ndim, b.ndim)):
        a_ax = 0 if a.ndim > ndim else None
        b_ax = 0 if b.ndim > ndim else None
        dot = vmap(dot, in_axes=(a_ax, b_ax))
      return dot(a, b)

    assert vecvec(jnp.zeros((3,)), jnp.zeros((3,))).shape == ()
    assert vecvec(jnp.zeros((2, 3)), jnp.zeros((3,))).shape == (2,)
    assert vecvec(jnp.zeros((4, 2, 3)), jnp.zeros((3,))).shape == (4, 2)

  def testDot2(self):
    R = self.rng().randn
    xs = R(10, 3)
    ys = R(10, 3)
    ans = vmap(jnp.dot)(xs, ys)
    expected = np.einsum('ni,ni->n', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot3(self):
    R = self.rng().randn
    xs = R(5, 8, 10)
    ys = R(10, 1)
    ans = vmap(jnp.dot, in_axes=(1, None))(xs, ys)
    expected = np.einsum('inj,jk->nik', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot4(self):
    R = self.rng().randn
    xs = R(3, 2)
    ys = R(3)
    ans = vmap(jnp.dot, in_axes=(1, None))(xs, ys)
    expected = np.einsum('ij,i->j', xs, ys)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testPad(self):
    R = self.rng().randn

    fun = lambda x: lax.pad(x, np.float32(0), [(1, 2, 1)])
    x = R(5, 10).astype(np.float32)
    ans = vmap(fun)(x)
    expected_ans = jnp.stack(list(map(fun, x)))
    self.assertAllClose(ans, expected_ans, check_dtypes=False)


    fun = lambda x: lax.pad(x, np.float32(0), [(1, 2, 1), (0, 1, 0)])
    x = R(5, 10, 3).astype(np.float32)
    ans = vmap(fun)(x)
    expected_ans = jnp.stack(list(map(fun, x)))
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testConcatenate(self):
    R = lambda *shape: self.rng().randn(*shape).astype(np.float32)

    fun = lambda *args: lax.concatenate(args, dimension=0)
    x, y, z = R(10, 2, 3), R(1, 10, 3), R(4, 3)
    ans = vmap(fun, in_axes=(0, 1, None))(x, y, z)
    expected_ans = np.concatenate([x, np.swapaxes(y, 0, 1),
                                    np.broadcast_to(z, (10, 4, 3))], 1)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

    fun = lambda *args: lax.concatenate(args, dimension=1)
    x, y, z = R(10, 2, 1), R(2, 3), R(2, 4, 10)
    ans = vmap(fun, in_axes=(0, None, 2))(x, y, z)
    expected_ans = np.concatenate([x, np.broadcast_to(y, (10, 2, 3)),
                                    np.moveaxis(z, 2, 0)], 2)
    self.assertAllClose(ans, expected_ans, check_dtypes=False)

  def testJacobianIssue54(self):
    # test modeling the code in https://github.com/google/jax/issues/54

    def func(xs):
      return jnp.array(list(xs))

    xs = jnp.ones((5, 1))
    jacrev(func)(xs)  # don't crash
    jacfwd(func)(xs)  # don't crash

  def testAny(self):
    # test modeling the code in https://github.com/google/jax/issues/108

    ans = vmap(jnp.any)(jnp.array([[True, False], [False, False]]))
    expected = jnp.array([True, False])
    self.assertAllClose(ans, expected)

  def testHessian(self):
    # test based on code from sindhwani@google
    def fun(x, t):
      return jnp.sum(jnp.power(jnp.maximum(x, 0.0), 2)) + t

    x = np.array([-1., -0.5, 0., 0.5, 1.0])

    ans = hessian(lambda x: fun(x, 0.0))(x)
    expected = np.array([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0.,0.5, 0., 0.],
                          [0., 0., 0., 2., 0.],
                          [0., 0., 0., 0., 2.]])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDynamicSlice(self):
    # test dynamic_slice via numpy indexing syntax
    # see https://github.com/google/jax/issues/1613 for an explanation of why we
    # need to use np rather than np to create x and idx
    x = jnp.arange(30).reshape((10, 3))

    ans = vmap(lambda x, i: x[i], in_axes=(0, None))(x, 1)
    expected = x[:, 1]
    self.assertAllClose(ans, expected, check_dtypes=False)


    idx = jnp.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, i: x[i], in_axes=(0, 0))(x, idx)
    expected = x[np.arange(10), idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = jnp.arange(3)
    idx = jnp.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, i: x[i], in_axes=(None, 0))(x, idx)
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDynamicUpdateSlice(self):
    x = self.rng().randn(10, 3)
    y = self.rng().randn(10)
    ans = vmap(lambda x, y, i: lax.dynamic_update_index_in_dim(x, y, i, axis=0),
               in_axes=(0, 0, None))(x, y, 1)
    expected = x.copy()
    expected[:, 1] = y
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = self.rng().randn(3)
    idx = np.array([0, 1, 2, 1, 0] * 2)
    ans = vmap(lambda x, y, i: lax.dynamic_update_index_in_dim(x, y, i, axis=0),
               in_axes=(None, 0, 0))(x, y, idx)
    expected = np.broadcast_to(x, (10, 3)).copy()
    expected[np.arange(10), idx] = y
    self.assertAllClose(ans, expected, check_dtypes=False)

  @jax.legacy_prng_key('allow')
  def testRandom(self):
    seeds = vmap(random.PRNGKey)(np.arange(10))
    ans = vmap(partial(random.normal, shape=(3, 2)))(seeds)
    expected = np.stack([random.normal(random.PRNGKey(seed), (3, 2))
                          for seed in np.arange(10)])
    self.assertAllClose(ans, expected, check_dtypes=False)
    assert len(np.unique(ans)) == 10 * 3 * 2

  def testSort(self):
    v = np.arange(12)[::-1].reshape(3, 4)

    sv = vmap(partial(lax.sort, dimension=0), (0,))(v)
    self.assertAllClose(sv, v[:, ::-1])

    sv = vmap(partial(lax.sort, dimension=-1), (0,))(v)
    self.assertAllClose(sv, v[:, ::-1])

    sv = vmap(partial(lax.sort, dimension=0), (1,))(v)
    self.assertAllClose(sv, v[::-1, :].T)

    sv = vmap(partial(lax.sort, dimension=0), (1,), 1)(v)
    self.assertAllClose(sv, v[::-1, :])

  def testSortKeyVal(self):
    k = np.arange(12)[::-1].reshape(3, 4)
    v = self.rng().permutation(12).reshape(3, 4)

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (0, 0))(k, v)
    self.assertAllClose(sk, k[:, ::-1])
    self.assertAllClose(sv, v[:, ::-1])

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (1, 1), 1)(k, v)
    self.assertAllClose(sk, k[::-1, :])
    self.assertAllClose(sv, v[::-1, :])

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (0, 1))(k, v.T)
    self.assertAllClose(sk, k[:, ::-1])
    self.assertAllClose(sv, v[:, ::-1])

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (1, 0))(k.T, v)
    self.assertAllClose(sk, k[:, ::-1])
    self.assertAllClose(sv, v[:, ::-1])

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (None, 0))(k[0], v)
    self.assertAllClose(sk, np.broadcast_to(k[0, ::-1], (3, 4)))
    self.assertAllClose(sv, v[:, ::-1])

    sk, sv = vmap(partial(lax.sort_key_val, dimension=0), (1, None))(k.T, v[0])
    self.assertAllClose(sk, k[:, ::-1])
    self.assertAllClose(sv, np.broadcast_to(v[0, ::-1], (3, 4)))

  def testConvGeneralDilated(self):
    W = jnp.array(self.rng().randn(3, 3, 1, 5), dtype=np.float32)
    X = jnp.array(self.rng().randn(10, 5, 5, 1), dtype=np.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      return y
    grad_loss = grad(lambda params, x: jnp.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example = jnp.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, jnp.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          jnp.reshape(g, (1,) + g.shape)]
    per_example_direct = jnp.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct,
                        rtol=2e-2, atol=2e-3)

  def testConvGeneralDilatedBatchNotMajor(self):
    W = jnp.array(self.rng().randn(3, 3, 1, 4), dtype=np.float32)
    x = jnp.array(self.rng().randn(3, 5, 7, 5, 1), dtype=np.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('HNWC', 'HWIO', 'HWNC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      return y

    per_example = vmap(partial(f, W))(x)
    per_example = jnp.reshape(jnp.transpose(per_example, (1, 2, 0, 3, 4)),
                             (5, 5, 21, 4))
    per_example_direct = f(W, jnp.reshape(jnp.transpose(x, (1, 0, 2, 3, 4)),
                                         (5, 21, 5, 1)))
    self.assertAllClose(per_example, per_example_direct)

  @parameterized.named_parameters(
    {"testcase_name": f"_op={name}", "op": op, "unit": unit}
    for name, op, unit in [("max", lax.max, -jnp.inf), ("min", lax.min, jnp.inf)])
  def testMinMaxPool(self, op, unit):
    W = jnp.array(self.rng().randn(3, 3, 1, 5), dtype=np.float32)
    X = jnp.array(self.rng().randn(10, 5, 5, 1), dtype=np.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      y = lax.reduce_window(
          y, unit, op, (1, 2, 2, 1), (1, 1, 1, 1), 'SAME')
      return y
    grad_loss = grad(lambda params, x: jnp.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example = jnp.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, jnp.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          jnp.reshape(g, (1,) + g.shape)]
    per_example_direct = jnp.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct, rtol=5e-2, atol=1e-3)

  def testSumPool(self):
    W = jnp.array(self.rng().randn(3, 3, 1, 5), dtype=np.float32)
    X = jnp.array(self.rng().randn(10, 5, 5, 1), dtype=np.float32)

    def f(params, x):
      one = (1, 1)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
      y = lax.conv_general_dilated(
          x, params, one, 'SAME', one, one, dimension_numbers)
      y = lax.reduce_window(
          y, 0.0, lax.add, (1, 2, 2, 1), (1, 1, 1, 1), 'SAME')
      return y
    grad_loss = grad(lambda params, x: jnp.mean(f(params, x) ** 2))

    # Test forward prop.
    per_example = vmap(partial(f, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example = jnp.reshape(per_example, (10, 5, 5, 5))
    per_example_direct = f(W, X)
    self.assertAllClose(per_example, per_example_direct)

    # Test gradients.
    per_example = vmap(partial(grad_loss, W))(jnp.reshape(X, (10, 1, 5, 5, 1)))
    per_example_direct = []
    for i in range(10):
      g = grad_loss(W, jnp.reshape(X[i], (1, 5, 5, 1)))
      per_example_direct += [
          jnp.reshape(g, (1,) + g.shape)]
    per_example_direct = jnp.concatenate(per_example_direct, axis=0)
    self.assertAllClose(per_example, per_example_direct,
                        rtol=3e-2, atol=1e-3)

  def testCumProd(self):
   x = jnp.arange(9).reshape(3, 3) + 1
   y = vmap(lambda x: jnp.cumprod(x, axis=-1))(x)
   self.assertAllClose(jnp.cumprod(x, axis=1), y)

  def testSelect(self):
    pred = np.array([True, False])
    on_true = np.array([0, 1])
    on_false = np.array([2, 3])
    ans = vmap(lax.select)(pred, on_true, on_false)
    expected = np.array([0, 3])
    self.assertAllClose(ans, expected)

    pred = np.array([False, True])
    on_true = np.array([0, 1])
    on_false = np.array([2, 3])
    ans = vmap(lax.select, (0, None, None))(pred, on_true, on_false)
    expected = np.array([[2, 3],
                          [0, 1]])
    self.assertAllClose(ans, expected)

    pred = True
    on_true = np.array([0, 1], np.float32)
    on_false = np.array(3, np.float32)
    ans = vmap(lax.select, (None, 0, None))(pred, on_true, on_false)
    expected = np.array([0, 1], np.float32)
    self.assertAllClose(ans, expected)

    pred = np.array([False, True])
    on_true = np.array([0, 1], np.float32)
    on_false = np.array(3, np.float32)
    ans = vmap(lax.select, (0, 0, None))(pred, on_true, on_false)
    expected = np.array([3, 1], np.float32)
    self.assertAllClose(ans, expected)

    pred = np.array([False, True])
    on_true = np.array([2], np.float32)
    on_false = np.array([[3, 4]], np.float32)
    ans = vmap(lax.select, (0, None, 1), 1)(pred, on_true, on_false)
    expected = np.array([[3, 2]], np.float32)
    self.assertAllClose(ans, expected)

  def testLaxLinalgCholesky(self):
    a = self.rng().randn(10, 5, 5).astype(np.float32)
    a = np.matmul(a, np.conj(np.swapaxes(a, -1, -2)))

    ans = vmap(lax.linalg.cholesky)(a)
    expected = np.linalg.cholesky(a)
    self.assertAllClose(ans, expected, check_dtypes=False, atol=1E-3)

    b = self.rng().randn(10, 5, 5).astype(np.float32)
    b = np.matmul(b, np.conj(np.swapaxes(b, -1, -2)))
    b_trans = np.swapaxes(b, 0, 1)  # shape is (5, 10, 5)

    ans = vmap(lax.linalg.cholesky, in_axes=1, out_axes=0)(b_trans)
    expected = np.linalg.cholesky(b)
    self.assertAllClose(ans, expected, check_dtypes=False, rtol=1e-4)

  def testLaxLinalgTriangularSolve(self):
    a = self.rng().randn(4, 10, 4).astype(np.float32)
    a += np.eye(4, dtype=jnp.float32)[:, None, :]
    b = self.rng().randn(5, 4, 10).astype(np.float32)

    ans = vmap(lax.linalg.triangular_solve, in_axes=(1, 2))(a, b)
    expected = np.stack(
      [lax.linalg.triangular_solve(a[:, i], b[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected, atol=1e-5, rtol=1e-5)

    ans = vmap(lax.linalg.triangular_solve, in_axes=(None, 2))(a[:, 0], b)
    expected = np.stack(
      [lax.linalg.triangular_solve(a[:, 0], b[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected)

    ans = vmap(lax.linalg.triangular_solve, in_axes=(1, None))(a, b[..., 0])
    expected = np.stack(
      [lax.linalg.triangular_solve(a[:, i], b[..., 0]) for i in range(10)])
    self.assertAllClose(ans, expected, atol=1e-5, rtol=1e-5)

  def testLaxLinalgTridiagonalSolve(self):
    dl = self.rng().randn(4, 10).astype(np.float32)
    d = self.rng().randn(4, 10).astype(np.float32) + 1.
    du = self.rng().randn(4, 10).astype(np.float32)
    b = self.rng().randn(4, 5, 10).astype(np.float32)

    ans = vmap(lax.linalg.tridiagonal_solve, in_axes=(1, 1, 1, 2))(dl, d, du, b)
    expected = np.stack(
        [lax.linalg.tridiagonal_solve(
            dl[:, i], d[:, i], du[:, i], b[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected, atol=1e-5, rtol=1e-5)

    ans = vmap(lax.linalg.tridiagonal_solve, in_axes=(None, None, None, 2))(
        dl[:, 0], d[:, 0], du[:, 0], b)
    expected = np.stack(
        [lax.linalg.tridiagonal_solve(
            dl[:, 0], d[:, 0], du[:, 0], b[..., i]) for i in range(10)])
    self.assertAllClose(ans, expected)

    ans = vmap(lax.linalg.tridiagonal_solve, in_axes=(1, 1, 1, None))(
        dl, d, du, b[..., 0])
    expected = np.stack(
        [lax.linalg.tridiagonal_solve(
            dl[:, i], d[:, i], du[:, i], b[..., 0]) for i in range(10)])
    self.assertAllClose(ans, expected, atol=1e-5, rtol=1e-5)


  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes}
      for dtype in [np.float32, np.int32]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (3, 5), np.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,)),
          (1, (10, 3), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,)),
          (1, (10, 3, 5), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3)),
          (2, (10, 5, 3), np.array([[0, 2], [1, 0]]),
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,),
             start_index_map=(0, 1)),
            (1, 3)),
      ])
  def testGatherBatchedOperand(self, axis, shape, dtype, idxs, dnums, slice_sizes):
    rng = jtu.rand_default(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    ans = vmap(fun, (axis, None))(operand, idxs)
    expected = np.stack([fun(operand[(slice(None),) * axis + (i,)], idxs)
                          for i in range(operand.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes}
      for dtype in [np.float32, np.float64]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (3, 5), np.array([[0], [2]]), lax.GatherDimensionNumbers(
            offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1,)),
          (1, (10, 3), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
            (2,)),
          (1, (10, 3, 5), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
            (1, 3)),
          (2, (10, 5, 3), np.array([[0, 2], [1, 0]]),
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,),
             start_index_map=(0, 1)),
            (1, 3))
      ])
  def testGatherGradBatchedOperand(self, axis, shape, dtype, idxs, dnums, slice_sizes):
    rng = jtu.rand_default(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: jnp.sum(jnp.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    ans = vmap(gfun, (axis, None))(operand, idxs)
    expected = np.stack([gfun(operand[(slice(None),) * axis + (i,)], idxs)
                          for i in range(operand.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes}
      for dtype in [np.float32, np.int32]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (5,), np.array([[[0], [2]], [[1], [3]]]), lax.GatherDimensionNumbers(
              offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)), (1,)),
          (1, (10,), np.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)), (2,)),
          (1, (10, 5), np.array([[0, 2, 1], [0, 3, 3]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)), (1, 3)),
          (0, (10, 5), np.array([[[0, 1], [2, 0]],
                                  [[1, 0], [2, 3]]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)), (1, 3)),
      ])
  def testGatherBatchedIndices(self, axis, shape, dtype, idxs, dnums, slice_sizes):
    rng = jtu.rand_default(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    ans = vmap(fun, (None, axis))(operand, idxs)
    expected = np.stack([fun(operand, idxs[(slice(None),) * axis + (i,)])
                          for i in range(idxs.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, idxs, dnums,
          slice_sizes),
       "axis": axis, "shape": shape, "dtype": dtype, "idxs": idxs, "dnums": dnums,
       "slice_sizes": slice_sizes}
      for dtype in [np.float32, np.float64]
      for axis, shape, idxs, dnums, slice_sizes in [
          (0, (5,), np.array([[[0], [2]], [[1], [3]]]), lax.GatherDimensionNumbers(
              offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)), (1,)),
          (1, (10,), np.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)), (2,)),
          (1, (10, 5), np.array([[0, 2, 1], [0, 3, 3]]).T[..., None],
           lax.GatherDimensionNumbers(
               offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)), (1, 3)),
          (0, (10, 5), np.array([[[0, 1], [2, 0]],
                                  [[1, 0], [2, 3]]]), lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)), (1, 3)),
      ])
  def testGatherGradBatchedIndices(self, axis, shape, dtype, idxs, dnums, slice_sizes):
    rng = jtu.rand_default(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: jnp.sum(jnp.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    ans = vmap(gfun, (None, axis))(operand, idxs)
    expected = np.stack([gfun(operand, idxs[(slice(None),) * axis + (i,)])
                          for i in range(idxs.shape[axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_op_axis={}_idxs_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), op_axis, idxs_axis, idxs,
          dnums, slice_sizes),
       "op_axis": op_axis, "idxs_axis": idxs_axis, "shape": shape, "dtype":
       dtype, "idxs": idxs, "dnums": dnums, "slice_sizes": slice_sizes}
      for dtype in [np.float32, np.int32]
      for op_axis, idxs_axis, shape, idxs, dnums, slice_sizes in [
          (0, 0, (2, 5), np.array([[[0], [2]], [[1], [3]]]),
           lax.GatherDimensionNumbers(
             offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1,)),
          (1, 1, (10, 2), np.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
           (2,)),
          (0, 1, (2, 10, 5,), np.array([[[0, 2, 1], [0, 3, 3]]]).T,
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1, 3)),
          (2, 0, (10, 5, 2), np.array([[[0, 2], [1, 0]],
                                        [[1, 0], [2, 0]]]),
          lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
           (1, 3)),
      ])
  def testGatherBatchedBoth(self, op_axis, idxs_axis, shape, dtype, idxs, dnums, slice_sizes):
    rng = jtu.rand_default(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    operand = rng(shape, dtype)
    assert operand.shape[op_axis] == idxs.shape[idxs_axis]
    ans = vmap(fun, (op_axis, idxs_axis))(operand, idxs)
    expected = np.stack([fun(operand[(slice(None),) * op_axis + (i,)],
                              idxs[(slice(None),) * idxs_axis + (i,)])
                          for i in range(idxs.shape[idxs_axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_op_axis={}_idxs_axis={}_idxs={}_dnums={}_slice_sizes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), op_axis, idxs_axis, idxs,
          dnums, slice_sizes),
       "op_axis": op_axis, "idxs_axis": idxs_axis, "shape": shape, "dtype":
       dtype, "idxs": idxs, "dnums": dnums, "slice_sizes": slice_sizes}
      for dtype in [np.float32]
      for op_axis, idxs_axis, shape, idxs, dnums, slice_sizes in [
          (0, 0, (2, 5), np.array([[[0], [2]], [[1], [3]]]),
           lax.GatherDimensionNumbers(
             offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1,)),
          (1, 1, (10, 2), np.array([[0, 0, 0], [0, 2, 1]]).T[..., None],
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
           (2,)),
          (0, 1, (2, 10, 5,), np.array([[[0, 2, 1], [0, 3, 3]]]).T,
           lax.GatherDimensionNumbers(
             offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
           (1, 3)),
          (2, 0, (10, 5, 2), np.array([[[0, 2], [1, 0]],
                                        [[1, 0], [2, 0]]]),
          lax.GatherDimensionNumbers(
            offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
           (1, 3)),
      ])
  def testGatherGradBatchedBoth(self, op_axis, idxs_axis, shape, dtype, idxs, dnums,
                                slice_sizes):
    rng = jtu.rand_default(self.rng())
    fun = partial(lax.gather, dimension_numbers=dnums, slice_sizes=slice_sizes)
    gfun = grad(lambda x, idx: jnp.sum(jnp.sin(fun(x, idx))))
    operand = rng(shape, dtype)
    assert operand.shape[op_axis] == idxs.shape[idxs_axis]
    ans = vmap(gfun, (op_axis, idxs_axis))(operand, idxs)
    expected = np.stack([gfun(operand[(slice(None),) * op_axis + (i,)],
                              idxs[(slice(None),) * idxs_axis + (i,)])
                          for i in range(idxs.shape[idxs_axis])])
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testNumpyIndexing1(self):
    a = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    ind = np.array([[0, 1],
                    [2, 0]])
    def f(a, ind):
      return a[:, ind]
    expected = np.stack([f(a, ind[i, :]) for i in range(ind.shape[0])])
    ans = vmap(f, (None, 0))(a, ind)
    assert np.all(ans == expected)

  def testNumpyIndexing2(self):
    a = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    def f(a):
      inds = jnp.array([0, 2])
      return a[:, inds]
    ans = vmap(f)(a)
    expected = np.stack([f(a[:, i, :]) for i in range(a.shape[1])], axis=1)
    assert np.all(ans == expected)

  def testTranspose(self):
    x = np.arange(4 * 3 * 3).reshape((4, 3, 3))
    ans = vmap(lambda x: x + x.T)(x)
    expected = x + np.swapaxes(x, -1, -2)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposePermutation(self):
    x = np.arange(6 * 3 * 4 * 5).reshape((6, 3, 4, 5))
    ans = vmap(lambda x: jnp.transpose(x, (1, 0, 2)))(x)
    expected = np.transpose(x, (0, 2, 1, 3))
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = np.arange(6 * 3 * 4 * 5).reshape((6, 3, 4, 5))
    ans = vmap(lambda x: jnp.transpose(x, (1, 2, 0)))(x)
    expected = np.transpose(x, (0, 2, 3, 1))
    self.assertAllClose(ans, expected, check_dtypes=False)

    x = np.arange(6 * 3 * 4 * 5).reshape((3, 4, 6, 5))
    ans = vmap(lambda x: jnp.transpose(x, (1, 2, 0)), in_axes=2)(x)
    expected = np.transpose(x, (2, 1, 3, 0))
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testIssue354(self):
    psd_mat = self.rng().randn(20, 10)
    psd_mat = psd_mat.T.dot(psd_mat)
    vec = self.rng().randn(10)

    def f(scale):
      scaled_mat = scale[jnp.newaxis] * psd_mat
      chol = jnp.linalg.cholesky(scaled_mat)
      return -0.5 * jnp.sum((jnp.einsum('ij,j->i', chol, vec))**2)
    vmapped_f = vmap(f)
    vmapped_f_grad = grad(lambda x: jnp.sum(vmapped_f(x)))

    scales = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    ans = vmapped_f_grad(scales)  # don't crash!
    expected = np.stack([grad(f)(scale) for scale in scales])
    self.assertAllClose(ans, expected, check_dtypes=False,
                        rtol=jtu.default_gradient_tolerance)

  def testIssue387(self):
    # https://github.com/google/jax/issues/387
    R = self.rng().rand(100, 2)

    def dist_sq(R):
      dR = R[:, jnp.newaxis, :] - R[jnp.newaxis, :, :]
      zero = jnp.zeros_like(dR)
      dR = dR - jnp.where(jnp.abs(dR) < 0.5, zero, 0.5 * jnp.sign(dR))
      return jnp.sum(dR ** 2, axis=2)

    @jit
    def f(R):
      _ = dist_sq(R)
      return jnp.sum(R ** 2)

    _ = hessian(f)(R)  # don't crash on UnshapedArray

  @jax.legacy_prng_key('allow')
  def testIssue489(self):
    # https://github.com/google/jax/issues/489
    def f(key):
      def body_fn(uk):
        key = uk[1]
        u = random.uniform(key, ())
        key, _ = random.split(key)
        return u, key

      u, _ = lax.while_loop(lambda uk: uk[0] > 0.5, body_fn, (1., key))
      return u

    with jax.debug_key_reuse(False):
      print(vmap(f)(random.split(random.PRNGKey(0), 2)))  # no crash

  def testEmptyTuples(self):
    # Ensure there is no crash when a vectorized input contains empty tuples.
    result = vmap(lambda x, _: x + 1)(np.array([0, 1]), ())
    self.assertAllClose(result, np.array([1, 2]), check_dtypes=False)
    # Ensure there is no crash when a vectorized output contains empty tuples.
    result, empty_tuple = vmap(lambda x: (x + 1, ()))(np.array([0, 1]))
    self.assertAllClose(result, np.array([1, 2]), check_dtypes=False)
    self.assertEqual((), empty_tuple)

  def testIndexAddBatchedIndexesOnly(self):
    f = lambda x, idx, y: jnp.asarray(x).at[idx].add(y)
    result = vmap(f, (None, 0, None))(np.zeros((10,)), np.arange(10,), 1.)
    self.assertAllClose(result, np.eye(10), check_dtypes=False)

  def testIssue1170(self):
    def f(index1, index2):
      return jnp.arange(36).reshape(6, 6)[index1, index2]
    g = jax.jit(jax.pmap(f))
    ans = g(index1=np.asarray([1]), index2=np.asarray([2]))
    expected = g(np.asarray([1]), np.asarray([2]))
    self.assertAllClose(ans, expected)

  def testIssue3883(self):
    def scalar_f(x):
      return lax.dynamic_slice(x, [], [])

    xs = jnp.array([1, 2, 3, 4])
    ans = vmap(scalar_f)(xs)
    expected = jnp.array([scalar_f(x) for x in xs])
    self.assertAllClose(ans, expected)

    def scalar_f2(x):
      return lax.dynamic_update_slice(x, 7, [])

    xs = jnp.array([1, 2, 3, 4])
    ans = vmap(scalar_f2)(xs)
    expected = jnp.array([scalar_f2(x) for x in xs])
    self.assertAllClose(ans, expected)

  @parameterized.named_parameters(
      {"testcase_name": "_{}_vmap_names={}_collective_names={}".format(
          collective.__name__.replace(" ", ""),
          "".join(vmap_names), "".join(collective_names)),
       "collective": collective, "bulk_op": bulk_op, "vmap_names": vmap_names,
       "collective_names": collective_names}
      for collective, bulk_op in [(lax.psum, jnp.sum),
                                  (lax.pmax, jnp.max),
                                  (lax.pmin, jnp.min)]
      for vmap_names in [('i',), ('i', 'j'), ('i', 'j', 'k')]
      for subset_size in range(1, len(vmap_names) + 1)
      for collective_subset in it.combinations(vmap_names, subset_size)
      for collective_names in it.permutations(collective_subset))
  def testCommAssocCollective(self, collective, bulk_op, vmap_names, collective_names):
    shape = (2, 2, 2)
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)

    # To test relative permutations of the order in which the axis names appear
    # in the primitive call versus the order the vmaps are applied, we always
    # apply vmaps in the order of the `vmap_names` argument, and apply the
    # collective with names according to the `collective_names` argument.
    f = lambda x: x - collective(x, collective_names)
    # Use non-zero in and out axes to improve the coverage
    for i, axis_name in enumerate(vmap_names):
      f = vmap(f, axis_name=axis_name, in_axes=i, out_axes=i)
    pos_axis = [i for i, name in enumerate(vmap_names) if name in collective_names]
    self.assertAllClose(f(x), x - bulk_op(x, axis=pos_axis, keepdims=True))

    if collective is lax.psum:
      jtu.check_grads(f, (x,), 2, eps=1)

  def testPPermute(self):
    nelem = 10
    ntests = 10
    x = np.arange(nelem)
    rng = self.rng()
    for i in range(ntests):
      perm = np.arange(nelem)
      rng.shuffle(perm)
      perm_pairs = np.stack([np.arange(nelem), perm], axis=-1)
      rng.shuffle(perm_pairs)
      self.assertAllClose(
        vmap(lambda x: x - lax.ppermute(x, 'i', perm_pairs), axis_name='i')(x),
        x - x[np.argsort(perm)])

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}_vmap={vmap_axis}",
       "split_axis": split_axis, "concat_axis": concat_axis, "vmap_axis": vmap_axis}
      for split_axis, concat_axis, vmap_axis in it.product(range(3), range(3), range(4)))
  def testAllToAll(self, vmap_axis, split_axis, concat_axis):
    shape = (4, 4, 4, 4)
    x = np.arange(np.prod(shape)).reshape(shape)
    f = vmap(lambda x: lax.all_to_all(x, 'i', split_axis, concat_axis),
             in_axes=vmap_axis, axis_name='i')
    y = f(x)
    ref = jnp.moveaxis(x, (vmap_axis, split_axis + (vmap_axis <= split_axis)),
                          (concat_axis + 1, 0))
    self.assertAllClose(y, ref)

  @parameterized.named_parameters(
      {"testcase_name": f"_split={split_axis}_concat={concat_axis}_vmap={vmap_axis}",
       "split_axis": split_axis, "concat_axis": concat_axis, "vmap_axis": vmap_axis}
      for split_axis, concat_axis, vmap_axis in it.product(range(2), range(2), range(3)))
  def testAllToAllSplitAxis(self, vmap_axis, split_axis, concat_axis):
    shape = (4, 4, 4)
    x = np.arange(np.prod(shape)).reshape(shape)

    @partial(vmap, in_axes=vmap_axis, axis_name='i')
    @partial(vmap, in_axes=vmap_axis, axis_name='j')
    def f(x):
      return lax.all_to_all(x, ('i', 'j'), split_axis, concat_axis)

    unroll_shape = (2, 2, *shape[1:])
    unroll_shape = list(shape)
    unroll_shape[vmap_axis:vmap_axis+1] = (2, 2)
    x_unroll = x.reshape(unroll_shape)
    y_unrolled = f(x_unroll)
    y = y_unrolled.reshape(shape)

    if vmap_axis <= split_axis:
      split_axis += 1
    ref = jnp.moveaxis(x, (vmap_axis, split_axis),
                          (concat_axis + 1, 0))
    self.assertAllClose(y, ref)

  def testNegativeAxes(self):
    x = np.arange(3*4*5).reshape(3, 4, 5)
    self.assertAllClose(jax.vmap(jnp.sum, in_axes=-3)(x),
                        jnp.sum(x, axis=(1, 2)))
    self.assertAllClose(jax.vmap(jnp.sum, in_axes=-2)(x),
                        jnp.sum(x, axis=(0, 2)))
    self.assertAllClose(jax.vmap(jnp.sum, in_axes=-1)(x),
                        jnp.sum(x, axis=(0, 1)))


    error = (r"vmap was requested to map its argument along axis -4, which "
             r"implies that its rank should be at least 4, but is only 3 "
             r"\(its shape is \(3, 4, 5\)\)")
    with self.assertRaisesRegex(ValueError, error):
      jax.vmap(jnp.sum, in_axes=-4)(x)

    id = lambda y: y
    self.assertAllClose(x, jax.vmap(id, in_axes=0, out_axes=-3)(x))
    self.assertAllClose(x.transpose(1, 0, 2),
                        jax.vmap(id, in_axes=0, out_axes=-2)(x))
    self.assertAllClose(x.transpose(1, 2, 0),
                        jax.vmap(id, in_axes=0, out_axes=-1)(x))

    with self.assertRaisesRegex(ValueError, "axis -4 is out of bounds.*"):
      jax.vmap(id, in_axes=0, out_axes=-4)(x)

    self.assertAllClose(
      np.full((5,), 7),
      jax.vmap(lambda *xs: xs, in_axes=(0, None), out_axes=(0, -1))(
        np.arange(5), 7)[1])

    with self.assertRaisesRegex(ValueError, "axis -2 is out of bounds.*"):
      jax.vmap(lambda *xs: xs, in_axes=(0, None), out_axes=(0, -2))(
        np.arange(5), 7)

  def testAxisIndex(self):
    x = np.arange(10, dtype='int32')
    self.assertAllClose(
      vmap(lambda x: x - lax.axis_index('i'), axis_name='i')(x),
      x - np.arange(x.shape[0], dtype='int32'))

  def testCollectivePdot(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = self.rng()

    x = rng.randn(3, 4)
    y = rng.randn(4, 5)
    z = vmap(f, axis_name='i', in_axes=(1, 0), out_axes=None)(x, y)
    self.assertAllClose(z, jnp.dot(x, y))

    x = rng.randn(4, 3)
    y = rng.randn(4, 5)
    z = vmap(f, axis_name='i', in_axes=(0, 0), out_axes=None)(x, y)
    self.assertAllClose(z, jnp.dot(x.T, y))

  def testCollectivePdotBatching(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = self.rng()
    xs = rng.randn(2, 8, 3)
    ys = rng.randn(2, 3, 5)
    zs = vmap(vmap(f, axis_name='i', in_axes=(1, 0), out_axes=None))(xs, ys)
    self.assertAllClose(zs, jnp.einsum('nij,njk->nik', xs, ys))

  def testPdotPrecision(self):
    def f(x, y):
      return lax.pdot(x, y, 'i', precision=lax.Precision.HIGHEST)

    f_jaxpr = make_jaxpr(f, axis_env=(('i', 4),))(jnp.ones(4), jnp.ones(4))
    self.assertIn('HIGHEST', str(f_jaxpr))

    vmap_jaxpr = make_jaxpr(jax.vmap(f, axis_name='i'))(jnp.ones((3, 4)),
        jnp.ones((3, 4)))
    self.assertIn('HIGHEST', str(vmap_jaxpr))

  def testPdotJvp(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = self.rng()
    x = rng.randn(3, 4)
    x_dot = rng.randn(*x.shape)
    y = rng.randn(4, 5)
    y_dot = rng.randn(*y.shape)

    z, z_dot = vmap(lambda x, y, x_dot, y_dot: jvp(f, (x, y), (x_dot, y_dot)),
                    axis_name='i', in_axes=(1, 0, 1, 0), out_axes=None)(x, y, x_dot, y_dot)
    self.assertAllClose(z, jnp.dot(x, y))
    self.assertAllClose(z_dot, jnp.dot(x_dot, y) + jnp.dot(x, y_dot))

  def testPdotVjp(self):
    def f(x, y):
      return lax.pdot(x, y, 'i')

    rng = self.rng()
    x = rng.randn(3, 4)
    y = rng.randn(4, 5)
    z_bar = rng.randn(3, 5)

    x_bar, y_bar = vmap(lambda x, y, z_bar: vjp(f, x, y)[1](z_bar),
                        axis_name='i', in_axes=(1, 0, None), out_axes=(1, 0))(x, y, z_bar)
    self.assertAllClose(x_bar, jnp.dot(z_bar, y.T))
    self.assertAllClose(y_bar, jnp.dot(x.T, z_bar))

  def testVmapKwargs(self):
    # https://github.com/google/jax/issues/912

    def f(a, b):
      return (2*a, 3*b)

    x = vmap(f)(jnp.array([1]), jnp.array([2]))  # works
    y = vmap(f)(a=jnp.array([1]), b=jnp.array([2]))  # doesn't work
    self.assertAllClose(x, y)

  def testGradOfPsum(self):
    a = jnp.ones(5)
    f = vmap(jax.grad(lambda x: -lax.psum(x, 'i')), out_axes=None, axis_name='i')
    self.assertEqual(
        f(a),
        core.jaxpr_as_fun(jax.make_jaxpr(f)(a))(a)[0])

  def testAllGatherToUnmapped(self):
    def f(x):
      return lax.all_gather(x, axis_name='i')

    x = jnp.arange(15).reshape((3, 5))
    # Original mapped axis becomes first axis of unmapped return value.
    self.assertAllClose(vmap(f, axis_name='i', in_axes=1, out_axes=None)(x), x.T)

  def testBatchedAllGather(self):
    def f(x):
      return lax.all_gather(x, axis_name='i')

    x = jnp.arange(15).reshape((3, 5))
    res = vmap(vmap(f, axis_name='i', out_axes=None), axis_name='j')(x)
    self.assertAllClose(res, x)

    res = vmap(vmap(f, axis_name='j'), axis_name='i', out_axes=None)(x)
    self.assertAllClose(res, x.T)

  def testAllGatherTiled(self):
    def f(x):
      return lax.all_gather(x, axis_name='i', tiled=True)

    x = jnp.arange(60).reshape((4, 3, 5))
    res = vmap(f, axis_name='i', in_axes=(1,), out_axes=None)(x)
    self.assertAllClose(res, x.transpose((1, 0, 2)).reshape(-1, 5))

  def testBatchedAllGatherTiled(self):
    def f(x):
      return lax.all_gather(x, axis_name='i', tiled=True)

    x = jnp.arange(60).reshape((4, 3, 5))
    res = vmap(vmap(f, in_axes=1, out_axes=1), axis_name='i', in_axes=1, out_axes=None)(x)
    self.assertAllClose(res, x.transpose((1, 0, 2)).reshape(-1, 5))

  def testAllGatherVjp(self):
    def f(x):
      return lax.all_gather(x, axis_name='i')

    rng = self.rng()
    x = rng.randn(3, 4)
    y_bar = rng.randn(3, 3, 4)

    x_bar, = vmap(lambda x, y_bar: vjp(f, x)[1](y_bar), axis_name='i')(x, y_bar)
    self.assertAllClose(x_bar, np.sum(y_bar, axis=0))

  def testAllGatherOfConst(self):
    def f(x):
      a = lax.all_gather(jnp.ones_like(x), axis_name='i')
      b = lax.all_gather(1, axis_name='i')
      return a, b

    x = jnp.arange(15).reshape((3, 5))
    a, b = vmap(f, axis_name='i', in_axes=1, out_axes=None)(x)
    self.assertAllClose(a, jnp.ones(shape=(5, 3), dtype=x.dtype))
    self.assertAllClose(b, jnp.ones(shape=(5,), dtype=b.dtype))

  @parameterized.named_parameters(
      {"testcase_name": "_shape={}_axis={}_collective={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis, collective.__name__.replace(" ", "")),
       "shape": shape, "dtype": dtype, "axis": axis,
       "collective": collective, "bulk_op": bulk_op}
      for collective, bulk_op in [(parallel.pargmax, jnp.argmax),
                                  (parallel.pargmin, jnp.argmin)]
      for dtype in [np.float32, np.int32]
      for shape in [(7,), (5, 8)]
      for axis in range(len(shape))
  )
  def testArgAllReduce(self, shape, dtype, axis, collective, bulk_op):
    rng = jtu.rand_default(self.rng())
    x = rng(shape, dtype)
    ans = vmap(lambda x: collective(x, 'i'), in_axes=axis, out_axes=None,
               axis_name='i')(x)
    expected = bulk_op(x, axis=axis)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testReduceScatterAutodiff(self):
    f = vmap(partial(lax.psum_scatter, axis_name='i'), axis_name='i')
    x = self.rng().randn(3, 3, 4)
    jtu.check_grads(f, (x,), 2, ["fwd", "rev"], 1e-2, 1e-2, eps=1.)

  def testNonJaxTypedOutput(self):
    with self.assertRaisesRegex(
      TypeError, "Output from batched function.*is not a valid JAX type"):
      vmap(lambda x: "hello")(np.arange(5))

  def testIssue6096(self):
    def f(x):
      return jsp.special.betainc(jnp.ones(3), 1., x)

    self.assertEqual(f(jnp.ones(3)).shape, (3,))
    self.assertEqual(jax.vmap(f)(jnp.ones((2, 3))).shape, (2, 3))

  def testPpermuteBatcherTrivial(self):
    # https://github.com/google/jax/issues/8688
    def ppermute(input):
      return jax.lax.ppermute(input, axis_name="i", perm=[[0, 1], [1, 0]])

    grad_fn = jax.grad(ppermute)

    vmapped_gradients_fn = jax.vmap(grad_fn, axis_name="i")

    vector = jax.numpy.array([1., 2.])
    ans = vmapped_gradients_fn(vector)  # doesn't crash
    self.assertAllClose(ans, jnp.ones(2), check_dtypes=False)

  def testBatchingPreservesWeakType(self):
    # Regression test for https://github.com/google/jax/issues/10025
    x = jnp.ravel(1)
    self.assertTrue(dtypes.is_weakly_typed(x))
    @vmap
    def f(x):
      self.assertTrue(dtypes.is_weakly_typed(x), f"{x} is not weakly-typed")
      return x
    y = f(x)
    self.assertTrue(dtypes.is_weakly_typed(y))


Array = Any
ArrayElt = Any
Int = Union[int, core.Tracer]

# Can't used NamedTuple here b/c those are pytrees
class NamedArray:
  names: list[str]
  data: Array

  def __init__(self, names, data):
    assert len(names) == data.ndim
    self.names = names
    self.data = data

  def __repr__(self) -> str:
    return f'NamedArray(names={self.names}, data={self.data})'

class NamedMapSpec:
  name: str | None
  axis: int | None

  def __init__(self, name: str, axis: int | None):
    assert (name is None) == (axis is None)
    self.name = name
    self.axis = axis

def named_mul(x: NamedArray, y: NamedArray) -> NamedArray:
  if x.names != y.names: raise Exception
  return NamedArray(x.names, lax.mul(x.data, y.data))

# TODO(mattjj): don't make this a pytree
register_pytree_node(NamedArray,
                     lambda x: ((x.data,), x.names),
                     lambda names, xs: NamedArray(names, xs[0]))


def named_to_elt(cont: Callable[[Array, int | None], ArrayElt],
                 _: Int, val: NamedArray, spec: NamedMapSpec) -> NamedArray:
  if spec.name is None:
    return val
  else:
    elt_names, mapped_name = list_pop(val.names, spec.axis)
    if mapped_name != spec.name: raise Exception
    elt = cont(val.data, spec.axis)
    return NamedArray(elt_names, elt)

def named_from_elt(cont: Callable[[int, ArrayElt, int | None], Array],
                   axis_size: int, elt: NamedArray, annotation: NamedMapSpec
                   ) -> NamedArray:
  data = cont(axis_size, elt.data, annotation.axis)
  if annotation.axis is None:
    return NamedArray(elt.names, data)
  else:
    names = list_insert(elt.names, annotation.axis, annotation.name)
    return NamedArray(names, data)

@contextmanager
def temporarily_register_named_array_vmappable():
  batching.register_vmappable(NamedArray, NamedMapSpec, int,
                              named_to_elt, named_from_elt, None)
  try:
    yield
  finally:
    batching.unregister_vmappable(NamedArray)

a = TypeVar('a')

def list_pop(lst: list[a], idx: int) -> a:
  lst = list(lst)
  return lst, lst.pop(idx)

def list_insert(lst: list[a], idx: int, val: a) -> list[a]:
  lst = list(lst)
  lst.insert(idx, val)
  return lst


class VmappableTest(jtu.JaxTestCase):
  def test_basic(self):
    with temporarily_register_named_array_vmappable():
      def f(x):
        return named_mul(x, x)

      x = NamedArray(['i', 'j'], jnp.arange(12.).reshape(3, 4))
      g = jax.vmap(f,
                  in_axes=NamedMapSpec('i', 0),
                  out_axes=NamedMapSpec('i', 1),
                  axis_size=3)
      ans = g(x)
      expected = NamedArray(['j', 'i'], jnp.arange(12.).reshape(3, 4).T ** 2)

      self.assertEqual(ans.names, expected.names)
      self.assertAllClose(ans.data, expected.data)

  def test_basic_jit(self):
    with temporarily_register_named_array_vmappable():
      def f(x):
        return named_mul(x, x)

      x = NamedArray(['i', 'j'], jnp.arange(12.).reshape(3, 4))
      ans = jax.jit(f)(x)
      expected = NamedArray(['i', 'j'], jnp.arange(12.).reshape(3, 4) ** 2)

      self.assertEqual(ans.names, expected.names)
      self.assertAllClose(ans.data, expected.data)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
