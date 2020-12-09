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

import inspect
import unittest
import pickle

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lib as jaxlib
from jax import numpy as jnp
from jax import test_util as jtu
from jax.config import flags
from jax.lib import version
from jax.lib import xla_bridge
import numpy as np


FLAGS = flags.FLAGS



class JaxJitTest(parameterized.TestCase):

  def test_is_float_0(self):
    if version <= (0, 1, 56):
      raise unittest.SkipTest("old jaxlib version")

    self.assertTrue(
        jaxlib.jax_jit._is_float0(np.zeros((5, 5), dtype=jax.float0)))
    self.assertFalse(jaxlib.jax_jit._is_float0(np.zeros((5, 5))))

  def test_DtypeTo32BitDtype(self):
    if version <= (0, 1, 56):
      raise unittest.SkipTest("old jaxlib version")
    self.assertEqual(np.float32, jaxlib.jax_jit._DtypeTo32BitDtype(np.float64))

  def test_convert_scalars(self):
    jax_jit = jaxlib.jax_jit

    jax_enable_x64 = FLAGS.jax_enable_x64

    if jax_enable_x64:
      int_type = np.int64
      float_type = np.float64
      complex_type = np.complex128
    else:
      int_type = np.int32
      float_type = np.float32
      complex_type = np.complex64

    # int
    res = jax_jit._ScalarToBuffer(1, jax_enable_x64,
                                  xla_bridge.get_backend()).to_py()
    self.assertEqual(res, 1)
    self.assertEqual(res.dtype, int_type)
    # We also compare to the Python Jax API, to make sure we have the exact
    # same behavior. When Jax removes the flag and removes this feature, this
    # test will fail.
    self.assertEqual(jnp.asarray(1).dtype, res.dtype)

    # float
    res = jax_jit._ScalarToBuffer(1.0, jax_enable_x64,
                                  xla_bridge.get_backend()).to_py()
    self.assertEqual(res, 1.0)
    self.assertEqual(res.dtype, float_type)
    self.assertEqual(jnp.asarray(1.0).dtype, res.dtype)

    # bool
    for bool_value in [True, False]:
      res = jax_jit._ScalarToBuffer(bool_value, jax_enable_x64,
                                    xla_bridge.get_backend()).to_py()
      self.assertEqual(res, np.asarray(bool_value))
      self.assertEqual(res.dtype, np.bool)
      self.assertEqual(jnp.asarray(bool_value).dtype, res.dtype)

    # Complex
    res = jax_jit._ScalarToBuffer(1 + 1j, jax_enable_x64,
                                  xla_bridge.get_backend()).to_py()
    self.assertEqual(res, 1 + 1j)
    self.assertEqual(res.dtype, complex_type)
    self.assertEqual(jnp.asarray(1 + 1j).dtype, res.dtype)

  def test_signature_support(self):
    # TODO(jblespiau): remove after version release
    if version < (0, 1, 56):
      raise unittest.SkipTest("old jaxlib version")

    def f(a, b, c):
      return a + b + c

    jitted_f = jax.api._cpp_jit(f)
    self.assertEqual(inspect.signature(f), inspect.signature(jitted_f))

  def test_jit_args(self):
    def f(a, b, c, d):
      return a + b + c + d

    with self.subTest('_python_jit'):
      jitted_f = jax.api._python_jit(f, static_argnums=(1, 2), donate_argnums=3)
      jit_args = jitted_f._jit_args
      self.assertEqual(id(f), id(jit_args.fun))
      self.assertEqual(jit_args.static_argnums, (1, 2))
      self.assertEqual(jit_args.donate_argnums, (3,))

    # TODO(jblespiau): remove after version release
    if version < (0, 1, 56):
      raise unittest.SkipTest("old jaxlib version")

    with self.subTest('_cpp_jit'):
      jitted_f = jax.api._cpp_jit(f, static_argnums=(1, 2), donate_argnums=3)
      jit_args = jitted_f._jit_args
      self.assertEqual(id(f), id(jit_args.fun))
      self.assertEqual(jit_args.static_argnums, (1, 2))
      self.assertEqual(jit_args.donate_argnums, (3,))

  def test_pickle(self):
    # TODO(jblespiau): remove after version release
    if version < (0, 1, 56):
      raise unittest.SkipTest("old jaxlib version")

    with self.subTest('broken'):
      old = _WithJittedFuncBroken()
      with self.assertRaisesRegex(Exception, r"Can't pickle"):
        pickle.loads(pickle.dumps(old))

    with self.subTest('working'):
      old = _WithJittedFuncWorking()
      new = pickle.loads(pickle.dumps(old))
      self.assertEqual(old.func(1, 2, 3, 4), new.func(1, 2, 3, 4))


class _WithJittedFuncBroken:
  def __init__(self):
    self.func = jax.api._cpp_jit(_func, static_argnums=(1, 2), donate_argnums=3)


class _WithJittedFuncWorking:
  def __init__(self):
    self.func = jax.api._cpp_jit(_func, static_argnums=(1, 2), donate_argnums=3)

  def __getstate__(self):
    jit_args = self.func._jit_args
    return (jit_args.fun, jit_args.static_argnums, jit_args.donate_argnums)

  def __setstate__(self, state):
    f, s, d = state
    self.func = jax.api._cpp_jit(f, static_argnums=s, donate_argnums=d)


def _func(a, b, c, d):
  return a + b + c + d


if __name__ == '__main__':
  jax.config.config_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
