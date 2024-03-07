# Copyright 2022 The JAX Authors.
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


"""Tests for stack."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax._src.lax.stack import Stack
from jax._src import test_util as jtu


jax.config.parse_flags_with_absl()


class StackTest(jtu.JaxTestCase):

  def test_empty(self):
    stack = Stack.create(7, jnp.zeros((), jnp.int32))
    self.assertTrue(stack.empty())

  def test_pushes_and_pops(self):
    stack = Stack.create(7, jnp.zeros((), jnp.int32))
    stack = stack.push(jnp.int32(7))
    self.assertFalse(stack.empty())
    stack = stack.push(jnp.int32(8))
    self.assertFalse(stack.empty())
    x, stack = stack.pop()
    self.assertFalse(stack.empty())
    self.assertEqual(8, x)
    stack = stack.push(jnp.int32(9))
    x, stack = stack.pop()
    self.assertFalse(stack.empty())
    self.assertEqual(9, x)
    x, stack = stack.pop()
    self.assertTrue(stack.empty())
    self.assertEqual(7, x)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
