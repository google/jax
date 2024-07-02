# Copyright 2018 The JAX Authors.
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

from jax._src.interpreters.xla import (
  abstractify as abstractify,
  canonicalize_dtype as canonicalize_dtype,
  canonicalize_dtype_handlers as canonicalize_dtype_handlers,
  pytype_aval_mappings as pytype_aval_mappings,
)

from jax._src.dispatch import (
  apply_primitive as apply_primitive,
)

from jax._src import xla_bridge as _xb
from jax._src.lib import xla_client as _xc

_xe = _xc._xla
Backend = _xe.Client

# Deprecations
_deprecations = {
    # Added 2024-06-28
    "xb": (
        "jax.interpreters.xla.xb is deprecated. Use jax.lib.xla_bridge instead.",
        _xb
    ),
    "xc": (
        "jax.interpreters.xla.xc is deprecated. Use jax.lib.xla_client instead.",
        _xc,
    ),
    "xe": (
        "jax.interpreters.xla.xe is deprecated. Use jax.lib.xla_extension instead.",
        _xe,
    ),
    # Finalized 2024-05-13; remove after 2024-08-13
    "backend_specific_translations": (
        "jax.interpreters.xla.backend_specific_translations is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        None,
    ),
    "translations": (
        "jax.interpreters.xla.translations is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        None,
    ),
    "register_translation": (
        "jax.interpreters.xla.register_translation is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        None,
    ),
    "xla_destructure": (
        "jax.interpreters.xla.xla_destructure is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        None,
    ),
    "TranslationRule": (
        "jax.interpreters.xla.TranslationRule is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        None,
    ),
    "TranslationContext": (
        "jax.interpreters.xla.TranslationContext is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        None,
    ),
    "XlaOp": (
        "jax.interpreters.xla.XlaOp is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        None,
    ),
}

import typing
from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
if typing.TYPE_CHECKING:
  xb = _xb
  xc = _xc
  xe = _xe
else:
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
del typing
