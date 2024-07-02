# Copyright 2024 The JAX Authors.
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

from absl.testing import absltest

import jax
from jax._src import test_util as jtu
from jax._src import config

jax.config.parse_flags_with_absl()


jax_test_enum_config = config.enum_state(
    name='jax_test_enum_config',
    enum_values=['default', 'xxx', 'yyy'],
    default='default',
    help='Configuration only used for tests.')


class ConfigTest(jtu.JaxTestCase):
  def test_config_setting_via_update(self):
    self.assertEqual(jax_test_enum_config.value, 'default')

    jax.config.update('jax_test_enum_config', 'xxx')
    self.assertEqual(jax_test_enum_config.value, 'xxx')

    jax.config.update('jax_test_enum_config', 'yyy')
    self.assertEqual(jax_test_enum_config.value, 'yyy')

    jax.config.update('jax_test_enum_config', 'default')
    self.assertEqual(jax_test_enum_config.value, 'default')

  def test_config_setting_via_context(self):
    self.assertEqual(jax_test_enum_config.value, 'default')

    with jax_test_enum_config('xxx'):
      self.assertEqual(jax_test_enum_config.value, 'xxx')

      with jax_test_enum_config('yyy'):
        self.assertEqual(jax_test_enum_config.value, 'yyy')

      self.assertEqual(jax_test_enum_config.value, 'xxx')

    self.assertEqual(jax_test_enum_config.value, 'default')

  def test_config_update_validation(self):
    self.assertEqual(jax_test_enum_config.value, 'default')
    with self.assertRaisesRegex(ValueError, 'new enum value must be in.*'):
      jax.config.update('jax_test_enum_config', 'invalid')
    # Error should raise before changing the value
    self.assertEqual(jax_test_enum_config.value, 'default')

  def test_config_context_validation(self):
    self.assertEqual(jax_test_enum_config.value, 'default')
    with self.assertRaisesRegex(ValueError, 'new enum value must be in.*'):
      with jax_test_enum_config('invalid'):
        pass
    self.assertEqual(jax_test_enum_config.value, 'default')


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
