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

"""An example demonstrating how an FFI call can maintain "state" between calls

In this case, the ``counter`` call simply accumulates the number of times it
was executed, but this pattern can also be used for more advanced use cases.
For example, this pattern is used in jaxlib for:

1. The GPU solver linear algebra kernels which require an expensive "handler"
   initialization, and
2. The ``triton_call`` function which caches the compiled triton modules after
   their first use.
"""

import jax
import jax.extend as jex

from jax_ffi_example import _counter

for name, target in _counter.registrations().items():
  jex.ffi.register_ffi_target(name, target)


def counter(index):
  return jex.ffi.ffi_call(
    "counter", jax.ShapeDtypeStruct((), jax.numpy.int32), index=int(index)
  )
