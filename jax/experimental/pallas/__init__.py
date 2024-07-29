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

"""Module for Pallas, a JAX extension for custom kernels.

See the Pallas documentation at
https://jax.readthedocs.io/en/latest/pallas.html.
"""

from jax._src.deprecations import register as _register_deprecation
from jax._src.pallas.core import Blocked
from jax._src.pallas.core import BlockSpec
from jax._src.pallas.core import IndexingMode
from jax._src.pallas.core import no_block_spec
from jax._src.pallas.core import Unblocked
from jax._src.pallas.core import unblocked
from jax._src.pallas.core import GridSpec
from jax._src.pallas.pallas_call import pallas_call
from jax._src.pallas.pallas_call import pallas_call_p
from jax._src.pallas.primitives import atomic_add
from jax._src.pallas.primitives import atomic_and
from jax._src.pallas.primitives import atomic_cas
from jax._src.pallas.primitives import atomic_max
from jax._src.pallas.primitives import atomic_min
from jax._src.pallas.primitives import atomic_or
from jax._src.pallas.primitives import atomic_xchg
from jax._src.pallas.primitives import atomic_xor
from jax._src.pallas.primitives import debug_print
from jax._src.pallas.primitives import dot
from jax._src.pallas.primitives import load
from jax._src.pallas.primitives import max_contiguous
from jax._src.pallas.primitives import multiple_of
from jax._src.pallas.primitives import num_programs
from jax._src.pallas.primitives import program_id
from jax._src.pallas.primitives import run_scoped
from jax._src.pallas.primitives import store
from jax._src.pallas.primitives import swap
from jax._src.pallas.utils import cdiv
from jax._src.pallas.utils import next_power_of_2
from jax._src.pallas.utils import strides_from_shape
from jax._src.pallas.utils import when
from jax._src.state.indexing import ds
from jax._src.state.indexing import dslice
from jax._src.state.indexing import Slice
from jax._src.state.primitives import broadcast_to

_register_deprecation("pallas-block-spec-order")
del _register_deprecation
