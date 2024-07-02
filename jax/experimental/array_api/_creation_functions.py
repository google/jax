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

from __future__ import annotations

import jax
import jax.numpy as jnp


def asarray(obj, /, *, dtype=None, device=None, copy=None):
  return jax.device_put(jnp.array(obj, dtype=dtype, copy=copy), device=device)

def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
  return jax.device_put(jnp.linspace(start, stop, num=num, dtype=dtype, endpoint=endpoint), device=device)
