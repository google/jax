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

from jax._src.lib import xla_client as _xc

dtype_to_etype = _xc.dtype_to_etype
execute_with_python_values = _xc.execute_with_python_values
get_topology_for_devices = _xc.get_topology_for_devices
heap_profile = _xc.heap_profile
mlir_api_version = _xc.mlir_api_version
ops = _xc.ops
register_custom_call_target = _xc.register_custom_call_target
shape_from_pyval = _xc.shape_from_pyval
ArrayImpl = _xc.ArrayImpl
Client = _xc.Client
CompileOptions = _xc.CompileOptions
Device = _xc.Device
DeviceAssignment = _xc.DeviceAssignment
FftType = _xc.FftType
Frame = _xc.Frame
HloSharding = _xc.HloSharding
OpSharding = _xc.OpSharding
PaddingType = _xc.PaddingType
PrimitiveType = _xc.PrimitiveType
Shape = _xc.Shape
Traceback = _xc.Traceback
XlaBuilder = _xc.XlaBuilder
XlaComputation = _xc.XlaComputation
XlaRuntimeError = _xc.XlaRuntimeError

_deprecations = {
  # Added Aug 5 2024
  "_xla" : (
    "jax.lib.xla_client._xla is deprecated; use jax.lib.xla_extension.",
    _xc._xla
  ),
  "bfloat16" : (
    "jax.lib.xla_client.bfloat16 is deprecated; use ml_dtypes.bfloat16.",
    _xc.bfloat16
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  _xla = _xc._xla
  bfloat16 = _xc.bfloat16
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _xc
