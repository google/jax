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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570

import sys as _sys
import warnings as _warnings

import jax.numpy as _array_api

# Added 2024-08-01
_warnings.warn(
    "jax.experimental.array_api import is no longer required as of JAX v0.4.32; "
    "jax.numpy supports the array API by default.",
    DeprecationWarning, stacklevel=2
)

_sys.modules['jax.experimental.array_api'] = _array_api

del _array_api, _sys, _warnings
