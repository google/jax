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

from types import ModuleType
import warnings

# Module __getattr__ factory that warns if deprecated names are used.
#
# Example usage:
# from jax._src.interpreters.pxla import (
#   Mesh as _deprecated_Mesh,
# )
#
# _deprecations = {
#   # Added Feb 8, 2023:
#   "Mesh": (
#     "jax.interpreters.pxla.Mesh is deprecated. Use jax.sharding.Mesh.",
#     _deprecated_Mesh,
#   ),
# }
#
# from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
# __getattr__ = _deprecation_getattr(__name__, _deprecations)
# del _deprecation_getattr

# Note that type checkers such as Pytype will not know about the deprecated
# names. If it is desirable that a deprecated name is known to the type checker,
# add:
# import typing
# if typing.TYPE_CHECKING:
#   from jax._src.interpreters.pxla import (
#     Mesh as Mesh,
#   )
# del typing
def deprecation_getattr(module, deprecations):
  def getattr(name):
    if name in deprecations:
      message, fn = deprecations[name]
      if fn is None:  # Is the deprecation accelerated?
        raise AttributeError(message)
      warnings.warn(message, DeprecationWarning, stacklevel=2)
      return fn
    raise AttributeError(f"module {module!r} has no attribute {name!r}")

  return getattr


def accelerate_module_deprecation(module: ModuleType, name: str) -> None:
  """Accelerate the deprecation of a module-level attribute.

  Raises an AttributeError instead of a DeprecationWarning upon attribute access.
  Used in Google-internal code to implement faster deprecation.
  """
  message, _ = module._deprecations[name]
  module._deprecations[name] = (message, None)


# Maps (module, attr) to False if the attribute is deprecated and should emit
# a DeprecationWarning and to True if it is deprecated and should raise an
# AttributeError.
_registered_deprecations: dict[tuple[str, str], bool] = {}


def register(module: str, key: str) -> None:
  _registered_deprecations[module, key] = False


def unregister(module: str, key: str) -> None:
  _registered_deprecations.pop((module, key))


def accelerate(module: str, key: str) -> None:
  assert (module, key) in _registered_deprecations
  _registered_deprecations[module, key] = True


def is_accelerated(module: str, key: str) -> bool:
  return _registered_deprecations[module, key]
