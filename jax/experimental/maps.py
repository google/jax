# Copyright 2020 The JAX Authors.
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

import warnings

from jax._src import deprecations
from jax._src.maps import (
  AxisName as AxisName,
  ResourceSet as ResourceSet,
  SerialLoop as SerialLoop,
  _prepare_axes as _prepare_axes,
  make_xmap_callable as make_xmap_callable,
  serial_loop as serial_loop,
  xmap_p as xmap_p,
  xmap as xmap,
)
from jax._src.mesh import (
  EMPTY_ENV as EMPTY_ENV,
  ResourceEnv as ResourceEnv,
  thread_resources as thread_resources,
)

# Added March 7, 2024.
_msg = (
    "jax.experimental.maps and jax.experimental.maps.xmap are deprecated and"
    " will be removed in a future release. Use jax.experimental.shard_map or"
    " jax.vmap with the spmd_axis_name argument for expressing SPMD"
    " device-parallel computations. Please file an issue on"
    " https://github.com/google/jax/issues if neither"
    " jax.experimental.shard_map nor jax.vmap are suitable for your use case."
)

if deprecations.is_accelerated("jax-experimental-maps-module"):
  raise ImportError(_msg)
else:
  warnings.warn(_msg, DeprecationWarning, stacklevel=2)

del deprecations, warnings, _msg
