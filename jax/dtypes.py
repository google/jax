# Copyright 2019 Google LLC
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

# flake8: noqa: F401
from jax._src.dtypes import (
    _jax_types,  # TODO(phawkins): fix users and remove?
    bfloat16,
    canonicalize_dtype,
    make_array_dtype,
    finfo,  # TODO(phawkins): switch callers to jnp.finfo?
    float0,
    iinfo,  # TODO(phawkins): switch callers to jnp.iinfo?
    issubdtype,  # TODO(phawkins): switch callers to jnp.issubdtype?
    result_type,
    scalar_type_of,
)
