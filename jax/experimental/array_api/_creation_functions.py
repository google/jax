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
from jax._src.lib import xla_client as xc
from jax._src.sharding import Sharding

def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
  return jax.device_put(jnp.arange(start, stop, step, dtype=dtype), device=device)

def asarray(obj, /, *, dtype=None, device=None, copy=None):
  return jax.device_put(jnp.array(obj, dtype=dtype, copy=copy), device=device)

def empty(shape, *, dtype=None, device=None):
  return jax.device_put(jnp.empty(shape, dtype=dtype), device=device)

def empty_like(x, /, *, dtype=None, device=None):
  return jax.device_put(jnp.empty_like(x, dtype=dtype), device=device)

def eye(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None):
  return jax.device_put(jnp.eye(n_rows, n_cols, k=k, dtype=dtype), device=device)

def from_dlpack(x, /, *, device: xc.Device | Sharding | None = None, copy: bool | None = None):
  return jnp.from_dlpack(x, device=device, copy=copy)

def full(shape, fill_value, *, dtype=None, device=None):
  return jax.device_put(jnp.full(shape, fill_value, dtype=dtype), device=device)

def full_like(x, /, fill_value, *, dtype=None, device=None):
  return jax.device_put(jnp.full_like(x, fill_value=fill_value, dtype=dtype), device=device)

def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
  return jax.device_put(jnp.linspace(start, stop, num=num, dtype=dtype, endpoint=endpoint), device=device)

def meshgrid(*arrays, indexing='xy'):
  return jnp.meshgrid(*arrays, indexing=indexing)

def ones(shape, *, dtype=None, device=None):
  return jax.device_put(jnp.ones(shape, dtype=dtype), device=device)

def ones_like(x, /, *, dtype=None, device=None):
  return jax.device_put(jnp.ones_like(x, dtype=dtype), device=device)

def tril(x, /, *, k=0):
  return jnp.tril(x, k=k)

def triu(x, /, *, k=0):
  return jnp.triu(x, k=k)

def zeros(shape, *, dtype=None, device=None):
  return jax.device_put(jnp.zeros(shape, dtype=dtype), device=device)

def zeros_like(x, /, *, dtype=None, device=None):
  return jax.device_put(jnp.zeros_like(x, dtype=dtype), device=device)
