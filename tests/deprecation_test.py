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

import warnings

from absl.testing import absltest
from jax._src import deprecations
from jax._src import test_util as jtu
from jax._src.internal_test_util import deprecation_module as m

class DeprecationTest(absltest.TestCase):

  def testModuleDeprecation(self):
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      self.assertEqual(m.x, 42)

    with self.assertWarns(DeprecationWarning, msg="Please use x"):
      self.assertEqual(m.y, 101)

    with self.assertRaisesRegex(AttributeError, "Please do not use z"):
      _ = m.z

    with self.assertRaisesRegex(AttributeError,
                                "module .* has no attribute 'w'"):
      _ = m.w

  def testNamedDeprecation(self):
    some_unique_id = "some-unique-id"
    try:
      deprecations.register(some_unique_id)
      self.assertFalse(deprecations.is_accelerated(some_unique_id))
      deprecations.accelerate(some_unique_id)
      self.assertTrue(deprecations.is_accelerated(some_unique_id))
    finally:
      deprecations.unregister(some_unique_id)

    msg = f"deprecation_id={some_unique_id!r} not registered"
    with self.assertRaisesRegex(ValueError, msg):
      deprecations.accelerate(some_unique_id)
    with self.assertRaisesRegex(ValueError, msg):
      deprecations.is_accelerated(some_unique_id)
    with self.assertRaisesRegex(ValueError, msg):
      deprecations.unregister(some_unique_id)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
