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

"""Contains Triton-specific pallas modules."""

from jax._src.pallas.triton import lowering
from jax._src.lib import gpu_triton as triton_kernel_call_lib

if triton_kernel_call_lib is None:
  raise ImportError("Jaxlib with Triton not available.")

get_compute_capability = triton_kernel_call_lib.get_compute_capability

del lowering
