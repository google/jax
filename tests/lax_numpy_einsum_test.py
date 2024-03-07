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


from collections import defaultdict
from functools import partial
import itertools

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import dtypes
from jax import lax
import jax.numpy as jnp
import jax._src.test_util as jtu

jax.config.parse_flags_with_absl()


class EinsumTest(jtu.JaxTestCase):

  def _check(self, s, *ops):
    a = np.einsum(s, *ops)
    b = jnp.einsum(s, *ops, precision=lax.Precision.HIGHEST)
    self.assertAllClose(a, b, atol=1e-4, rtol=1e-4)

  def test_three_operands_1(self):
    r = self.rng()
    x = r.randn(3)
    y = r.randn(4)
    z = r.randn(5)
    s = 'i,j,k->ijk'
    self._check(s, x, y, z)

  def test_three_operands_2(self):
    r = self.rng()
    x = r.randn(3)
    y = r.randn(4)
    z = r.randn(5)
    s = 'i,j,k->ijk'
    self._check(s, x, y, z)

  def test_two_operands_1(self):
    r = self.rng()
    x = r.randn(3, 4)
    y = r.randn(4)
    s = 'ij,j->i'
    self._check(s, x, y)

  def test_two_operands_2(self):
    r = self.rng()
    x = r.randn(3, 4, 5)
    y = r.randn(4)
    s = 'ijk,j->i'
    self._check(s, x, y)

  def test_two_operands_3(self):
    r = self.rng()
    x = r.randn(3, 4, 3)
    y = r.randn(3)
    s = 'iji,i->j'
    self._check(s, x, y)

  def test_two_operands_4(self):
    r = self.rng()
    x = r.randn(3, 4)
    y = r.randn(3, 4)
    s = 'ij,ij->'
    self._check(s, x, y)

  def test_two_operands_5(self):
    r = self.rng()
    x = r.randn(10, 2, 3)
    y = r.randn(3, 4)
    s = 'nij,jk->nik'
    self._check(s, x, y)

  def test_two_operands_6(self):
    # based on https://github.com/google/jax/issues/37#issuecomment-448572187
    r = self.rng()
    x = r.randn(2, 1)
    y = r.randn(2, 3, 4)
    s = 'sa,shb->shab'
    self._check(s, x, y)

  def test_one_operand_1(self):
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->j'
    self._check(s, x)

  def test_one_operand_2(self):
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->kij'
    self._check(s, x)

  def test_one_operand_3(self):
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->ki'
    self._check(s, x)

  def test_one_operand_4(self):
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->ki'
    self._check(s, x)

  def test_one_operand_5(self):
    r = self.rng()
    x = r.randn(2, 3, 4, 5)
    s = '...ijk->...ki'
    self._check(s, x)

  def test_one_operand_6(self):
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = '...ijk->ki'
    self._check(s, x)

  def test_one_operand_7(self):
    r = self.rng()
    x = r.randn(3, 3)
    s = 'ii->'
    self._check(s, x)

  def test_one_operand_8(self):
    r = self.rng()
    x = r.randn(3, 3)
    s = 'ij->'
    self._check(s, x)

  def test_one_operand_9(self):
    r = self.rng()
    x = r.randn(3, 3, 3)
    s = 'iii->'
    self._check(s, x)

  def test_one_operand_10(self):
    r = self.rng()
    x = r.randn(3, 3)
    s = 'ii->i'
    self._check(s, x)

  def test_one_operand_11(self):
    r = self.rng()
    x = r.randn(3, 3, 4)
    s = 'iij->i'
    self._check(s, x)

  def test_one_operand_12(self):
    r = self.rng()
    x = r.randn(3, 3, 3)
    s = 'iii->i'
    self._check(s, x)

  def test_one_operand_13(self):
    r = self.rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkk->i'
    self._check(s, x)

  def test_one_operand_14(self):
    r = self.rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkk->ik'
    self._check(s, x)

  def test_one_operand_15(self):
    r = self.rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkl->il'
    self._check(s, x)

  def test_one_operand_16(self):
    r = self.rng()
    x = r.randn(3, 3)
    s = 'ij->ij'
    self._check(s, x)

  def test_tf_unsupported_1(self):
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = self.rng()
    x = r.randn(2, 3, 5, 1)
    y = r.randn(3, 4, 5, 1)
    s = 'ij...,jk...->ik...'
    self._check(s, x, y)

  def test_tf_unsupported_2(self):
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = self.rng()
    x = r.randn(2, 3, 3)
    y = r.randn(4)
    s = 'ijj,k->ik'
    self._check(s, x, y)

  def test_tf_unsupported_3(self):
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = self.rng()
    x = r.randn(2, 3)
    y = r.randn(2, 3)
    z = r.randn(3, 4)
    s = 'ij,ij,jk->ik'
    self._check(s, x, y, z)

  # these tests are based on https://github.com/dask/dask/pull/3412/files
  @parameterized.named_parameters(
      {"testcase_name": f"_{einstr}_dtype={dtype.__name__}",
      "einstr": einstr, "dtype": dtype}
      for einstr in [
          'abc,bad->abcd',
          'abcdef,bcdfg->abcdeg',
          'ea,fb,abcd,gc,hd->efgh',
          'ab,b',
          'aa',
          'a,a->',
          'a,a->a',
          'a,a',
          'a,b',
          'a,b,c',
          'a',
          'ba,b',
          'ba,b->',
          'defab,fedbc->defac',
          'ab...,bc...->ac...',
          'a...a',
          'abc...->cba...',
          '...ab->...a',
          'a...a->a...',
          # Following 2 from # https://stackoverflow.com/a/19203475/1611416
          '...abc,...abcd->...d',
          'ab...,b->ab...',
          # https://github.com/dask/dask/pull/3412#discussion_r182413444
          'aa->a',
          'ab,ab,c->c',
          'aab,bc->ac',
          'aab,bcc->ac',
          'fdf,cdd,ccd,afe->ae',
          'fff,fae,bef,def->abd',
      ]
      for dtype in [jnp.float32, jnp.int32, jnp.complex64, jnp.bool_])
  def test_from_dask(self, einstr, dtype):
    r = jtu.rand_default(self.rng())
    if '->' in einstr:
      input_str, result_names = einstr.split('->')
    else:
      input_str = einstr
    input_names = input_str.split(',')

    dims = itertools.cycle([2, 3, 4])
    shapes = defaultdict(lambda: next(dims))
    input_shapes = [tuple(shapes[c] for c in names.replace('...', '01'))
                    for names in input_names]
    operands = [r(shape, dtype) for shape in input_shapes]

    self._check(einstr, *operands)

  def test_ordered_front_batch_dim_case(self):
    x = np.ones((1,8,20,4))
    y = np.ones((1,8,20,4))
    s = 'ijkl,ijml->ijkm'
    self._check(s, x, y)

  def test_einsum_path(self):
    # just check examples from np.einsum_path docstring
    a = self.rng().rand(2, 2)
    b = self.rng().rand(2, 5)
    c = self.rng().rand(5, 2)

    path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
    self.assertEqual(str(path_info[0]), "['einsum_path', (1, 2), (0, 1)]")
    self.assertEqual(path_info[1].split('\n')[0],
                     '  Complete contraction:  ij,jk,kl->il')

    # check this doesn't crash
    I = self.rng().rand(10, 10, 10, 10)
    C = self.rng().rand(10, 10)
    np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C, optimize='greedy')

  @jax.default_matmul_precision("float32")
  def test_einsum_kpmurphy_example(self):
    # code from an email with @murphyk
    N, C, D, K, T = 2, 3, 4, 5, 6
    r = self.rng()
    S = r.randn(N, T, K)
    W = r.randn(K, D)
    V = r.randn(D, C)
    L = np.zeros((N, C))
    for n in range(N):
      for c in range(C):
        s = 0
        for d in range(D):
          for k in range(K):
            for t in range(T):
                s += S[n,t,k] * W[k,d] * V[d,c]
        L[n,c] = s

    path = jnp.einsum_path('ntk,kd,dc->nc', S, W, V, optimize='optimal')[0]
    self.assertAllClose(L, jnp.einsum('ntk,kd,dc->nc', S, W, V, optimize=path),
                        check_dtypes=False)

  def test_contraction_broadcasting(self):
    r = self.rng()
    x = r.randn(3, 4, 5)
    y = r.randn(3, 1, 6)
    s = 'cij,cjk->cik'
    self._check(s, x, y)

  def test_batch_broadcasting(self):
    r = self.rng()
    x = r.randn(1, 4, 5)
    y = r.randn(3, 5, 6)
    s = 'cij,cjk->cik'
    self._check(s, x, y)

  def test_batch_and_contraction_broadcasting(self):
    r = self.rng()
    x = r.randn(1, 4, 5)
    y = r.randn(3, 1, 6)
    s = 'cij,cjk->cik'
    self._check(s, x, y)

  def test_broadcasting_issue_2189(self):
    r = self.rng()
    x = r.randn(2, 1, 3, 3)
    y = r.randn(2, 4, 3)
    s = '...ij,...j'
    self._check(s, x, y)

  def test_no_unnecessary_transpose(self):
    r = self.rng()
    x = r.randn(2, 2, 2)
    y = r.randn(2, 2)
    jaxpr = jax.make_jaxpr(partial(jnp.einsum, "ijk,kl->ijl"))(x, y)
    self.assertNotIn('transpose', str(jaxpr))

  def test_preferred_element_type(self):
    r = self.rng()
    x = r.randn(2, 2).astype('bfloat16')
    y = r.randn(2).astype('bfloat16')
    pattern = "ij,j->i"
    f1 = partial(jnp.einsum, pattern)
    jaxpr = jax.make_jaxpr(f1)(x, y)
    self.assertLen(jaxpr.eqns, 1)
    self.assertEqual(jaxpr.eqns[0].params['preferred_element_type'],
                     dtypes.result_type(x, y))

    f2 = partial(jnp.einsum, pattern, preferred_element_type='float32')
    jaxpr = jax.make_jaxpr(f2)(x, y)
    self.assertLen(jaxpr.eqns, 1)
    self.assertEqual(jaxpr.eqns[0].params['preferred_element_type'], 'float32')

  def test_inf_nan(self):
    x = np.array([[[np.inf, np.inf],
                   [   1.0,    1.0]]])
    out = jnp.einsum('baa->ba', x)
    expected = np.einsum('baa->ba', x)
    self.assertAllClose(out, expected, check_dtypes=False)

  @jtu.sample_product(
      lhs_dtype=jtu.dtypes.numeric,
      rhs_dtype=jtu.dtypes.numeric,
  )
  @jax.numpy_dtype_promotion('standard')
  def test_einsum_mixed_precision(self, lhs_dtype, rhs_dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng((10,), lhs_dtype), rng((10,), rhs_dtype)]
    f_jax = partial(jnp.einsum, 'a,a->a')
    jaxpr = jax.make_jaxpr(f_jax)(*args_maker())
    self.assertIn(
      [eqn.primitive for eqn in jaxpr.eqns],
      [
        [lax.dot_general_p],
        [lax.dot_general_p, lax.convert_element_type_p],
      ])

    # Check result and expected dtype for all combinations
    f_np = jtu.promote_like_jnp(partial(np.einsum, 'a,a->a'))
    self._CheckAgainstNumpy(f_np, f_jax, args_maker, check_dtypes=True)

  @jtu.sample_product(
    [
      {'signature': 'i->', 'shapes': [(3,)]},
      {'signature': 'ii->i', 'shapes': [(4, 4)]},
      {'signature': 'ij,jk', 'shapes': [(3, 4), (4, 3)]},
      {'signature': 'ij,jkl,klm', 'shapes': [(2, 2), (2, 3, 4), (3, 4, 2)]},
    ],
    optimize=[True, False, 'optimal', 'auto', 'greedy', 'eager'],
    dtype=[np.dtype('float32')],
  )
  @jtu.skip_on_devices('tpu')
  def test_einsum_optimization_modes(self, signature, shapes, optimize, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    jnp_fun = partial(jnp.einsum, signature, optimize=optimize)
    np_fun = partial(np.einsum, signature)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=1E-4)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=1E-4)

  @jtu.sample_product(
    [
      {'signature': 'i->', 'shapes': [(3,)]},
      {'signature': 'ii->i', 'shapes': [(4, 4)]},
      {'signature': 'ij,jk', 'shapes': [(3, 4), (4, 3)]},
      {'signature': 'ij,jkl,klm', 'shapes': [(2, 2), (2, 3, 4), (3, 4, 2)]},
    ],
    optimize=[True, False, 'optimal', 'auto', 'greedy', 'eager'],
    dtype=[np.dtype('float32')],
  )
  @jtu.skip_on_devices('tpu')
  def test_einsum_path_optimization_modes(self, signature, shapes, optimize, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [rng(shape, dtype) for shape in shapes]
    def jnp_fun(*args, signature=signature, optimize=optimize):
      path, _ = jnp.einsum_path(signature, *args, optimize=optimize)
      return jnp.einsum(signature, *args, optimize=path)
    np_fun = partial(np.einsum, signature)
    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=1E-4)
    self._CompileAndCheck(jnp_fun, args_maker, rtol=1E-4)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
