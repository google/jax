# Copyright 2023 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

_deprecation_message = (
    "The jax.experimental.export module is deprecated. "
    "Use jax.export instead. "
    "See the migration guide at https://jax.readthedocs.io/en/latest/export/export.html#migration-guide-from-jax-experimental-export."
)

from jax._src.export import _export as _src_export
from jax._src.export import shape_poly as _src_shape_poly
from jax._src.export import serialization as _src_serialization
# Import only to set the shape poly decision procedure
from jax._src.export import shape_poly_decision
del shape_poly_decision

# All deprecations added Jun 14, 2024
_deprecations = {
    # Added Jun 13, 2024
    "Exported": (_deprecation_message, _src_export.Exported),
    "DisabledSafetyCheck": (_deprecation_message, _src_export.DisabledSafetyCheck),
    "export": (_deprecation_message, _src_export.export_back_compat),
    "call": (_deprecation_message, _src_export.call),
    "call_exported": (_deprecation_message, _src_export.call_exported),
    "default_lowering_platform":  (_deprecation_message, _src_export.default_lowering_platform),
    "minimum_supported_serialization_version" : (_deprecation_message, _src_export.minimum_supported_calling_convention_version),
    "maximum_supported_serialization_version" : (_deprecation_message, _src_export.maximum_supported_calling_convention_version),

    "serialize": (_deprecation_message, _src_serialization.serialize),
    "deserialize": (_deprecation_message, _src_serialization.deserialize),

    "SymbolicScope": (_deprecation_message, _src_shape_poly.SymbolicScope),
    "is_symbolic_dim": (_deprecation_message, _src_shape_poly.is_symbolic_dim),
    "symbolic_shape": (_deprecation_message, _src_shape_poly.symbolic_shape),
    "symbolic_args_specs": (_deprecation_message, _src_shape_poly.symbolic_args_specs),
}

import typing
if typing.TYPE_CHECKING:
  Exported = _src_export.Exported
  DisabledSafetyCheck = _src_export.DisabledSafetyCheck
  export = _src_export.export_back_compat
  call = _src_export.call
  call_exported = _src_export.call_exported
  default_lowering_platform = _src_export.default_lowering_platform

  serialize = _src_serialization.serialize
  deserialize = _src_serialization.deserialize

  SymbolicScope = _src_shape_poly.SymbolicScope
  is_symbolic_dim = _src_shape_poly.is_symbolic_dim
  symbolic_shape = _src_shape_poly.symbolic_shape
  symbolic_args_specs = _src_shape_poly.symbolic_args_specs
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _src_export
del _src_serialization
del _src_shape_poly
