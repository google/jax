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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/google/jax/issues/7570

from __future__ import annotations

from jax._src.interpreters.ad import (
  CustomJVPException as CustomJVPException,
  CustomVJPException as CustomVJPException,
  JVPTrace as JVPTrace,
  JVPTracer as JVPTracer,
  UndefinedPrimal as UndefinedPrimal,
  Zero as Zero,
  add_jaxvals as add_jaxvals,
  add_jaxvals_p as add_jaxvals_p,
  add_tangents as add_tangents,
  backward_pass as backward_pass,
  bilinear_transpose as bilinear_transpose,
  call_param_updaters as call_param_updaters,
  call_transpose as call_transpose,
  call_transpose_param_updaters as call_transpose_param_updaters,
  closed_backward_pass as closed_backward_pass,
  custom_lin_p as custom_lin_p,
  defbilinear as defbilinear,
  defjvp as defjvp,
  defjvp2 as defjvp2,
  defjvp_zero as defjvp_zero,
  deflinear as deflinear,
  deflinear2 as deflinear2,
  f_jvp_traceable as f_jvp_traceable,
  get_primitive_transpose as get_primitive_transpose,
  instantiate_zeros as instantiate_zeros,
  is_undefined_primal as is_undefined_primal,
  jvp as jvp,
  jvp_jaxpr as jvp_jaxpr,
  jvp_subtrace as jvp_subtrace,
  jvp_subtrace_aux as jvp_subtrace_aux,
  jvpfun as jvpfun,
  linear_jvp as linear_jvp,
  linear_transpose as linear_transpose,
  linear_transpose2 as linear_transpose2,
  linearize as linearize,
  map_transpose as map_transpose,
  nonzero_outputs as nonzero_outputs,
  nonzero_tangent_outputs as nonzero_tangent_outputs,
  primitive_jvps as primitive_jvps,
  primitive_transposes as primitive_transposes,
  rearrange_binders as rearrange_binders,
  recast_to_float0 as recast_to_float0,
  reducing_transposes as reducing_transposes,
  replace_float0s as replace_float0s,
  standard_jvp as standard_jvp,
  standard_jvp2 as standard_jvp2,
  traceable as traceable,
  unpair_pval as unpair_pval,
  vjp as vjp,
  zero_jvp as zero_jvp,
  zeros_like_aval as zeros_like_aval,
  zeros_like_jaxval as zeros_like_jaxval,
  zeros_like_p as zeros_like_p,
)

from jax import config as _deprecated_config
from jax._src import source_info_util as _deprecated_source_info_util
_deprecations = {
    # Added Oct 13, 2023:
    "config": (
        "jax.interpreters.ad.config is deprecated. Use jax.config directly.",
        _deprecated_config,
    ),
    "source_info_util": (
        "jax.interpreters.ad.source_info_util is deprecated. Use jax.extend.source_info_util.",
        _deprecated_source_info_util,
    ),
}

import typing
if typing.TYPE_CHECKING:
  config = _deprecated_config
  source_info_util = _deprecated_source_info_util
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _deprecated_config
del _deprecated_source_info_util
