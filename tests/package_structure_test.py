# Copyright 2023 The JAX Authors.
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

"""Tests of the JAX public package structure"""

from collections.abc import Sequence
import importlib
import types

from absl.testing import absltest, parameterized

from jax._src import test_util as jtu


def _mod(module_name: str, *, include: Sequence[str] = (), exclude: Sequence[str] = ()):
  return {"module_name": module_name, "include": include, "exclude": exclude}


class PackageStructureTest(jtu.JaxTestCase):
  @parameterized.parameters([
    # TODO(jakevdp): expand test to other public modules.
    _mod("jax.errors"),
    _mod("jax.nn.initializers"),
    _mod("jax.tree_util", exclude=['PyTreeDef', 'default_registry']),
  ])
  def test_exported_names_match_module(self, module_name, include, exclude):
    """Test that all public exports have __module__ set correctly."""
    module = importlib.import_module(module_name)
    self.assertEqual(module.__name__, module_name)
    for name in dir(module):
      if name not in include and (name.startswith('_') or name in exclude):
        continue
      obj = getattr(module, name)
      if isinstance(obj, types.ModuleType):
        continue
      self.assertEqual(obj.__module__, module_name,
                       f"{obj} has {obj.__module__=}, expected {module_name}")


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
