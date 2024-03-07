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

from typing import Any

import jax
from jax._src.array import ArrayImpl
from jax.experimental.array_api._version import __array_api_version__
from jax.sharding import Sharding

from jax._src.lib import xla_extension as xe


def _array_namespace(self, /, *, api_version: None | str = None):
  if api_version is not None and api_version != __array_api_version__:
    raise ValueError(f"{api_version=!r} is not available; "
                     f"available versions are: {[__array_api_version__]}")
  return jax.experimental.array_api


def _to_device(self, device: xe.Device | Sharding | None, *,
               stream: int | Any | None = None):
  if stream is not None:
    raise NotImplementedError("stream argument of array.to_device()")
  return jax.device_put(self, device)


def add_array_object_methods():
  # TODO(jakevdp): set on tracers as well?
  setattr(ArrayImpl, "__array_namespace__", _array_namespace)
  setattr(ArrayImpl, "to_device", _to_device)
  setattr(ArrayImpl, "device", property(lambda self: self.sharding))
