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
__all__ = ["DisabledSafetyCheck", "Exported", "export", "deserialize",
           "create",
           "maximum_supported_calling_convention_version",
           "minimum_supported_calling_convention_version",
           "default_export_platform",
           "SymbolicScope", "is_symbolic_dim",
           "symbolic_shape", "symbolic_args_specs"]

from jax._src.export._export import (
  DisabledSafetyCheck,
  Exported,
  create,
  deserialize,
  maximum_supported_calling_convention_version,
  minimum_supported_calling_convention_version,
  default_export_platform)

from jax._src.export import shape_poly_decision  # Import only to set the decision procedure
del shape_poly_decision
from jax._src.export.shape_poly import (
  SymbolicScope,
  is_symbolic_dim,
  symbolic_shape,
  symbolic_args_specs)

# For backwards compatibility only
def export(
    fun_jit,
    *,
    platforms = None,
    lowering_platforms = None,
    disabled_checks = (),
    ):
  import warnings
  warnings.warn("The function jax.export.export is deprecated. Use jax.export.create instead",
                DeprecationWarning, stacklevel=2)
  if platforms is not None and lowering_platforms is not None:
    raise ValueError("Cannot use both `platforms` and `lowering_platforms`")
  if platforms is None and lowering_platforms is not None:
    platforms = lowering_platforms
  def create_exported(*fun_args, **fun_kwargs):
    return create(fun_jit, *fun_args, **fun_kwargs,
                  export_platforms=platforms,
                  export_disabled_checks=disabled_checks)
  return create_exported
