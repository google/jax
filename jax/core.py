# Copyright 2022 The JAX Authors.
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

from jax._src.core import (
  AbstractToken as AbstractToken,
  AbstractValue as AbstractValue,
  Atom as Atom,
  AxisSize as AxisSize,
  CallPrimitive as CallPrimitive,
  ClosedJaxpr as ClosedJaxpr,
  ConcreteArray as ConcreteArray,
  ConcretizationTypeError as ConcretizationTypeError,
  DShapedArray as DShapedArray,
  DropVar as DropVar,
  Effect as Effect,
  Effects as Effects,
  EvalTrace as EvalTrace,
  InDBIdx as InDBIdx,
  InconclusiveDimensionOperation as InconclusiveDimensionOperation,
  InputType as InputType,
  Jaxpr as Jaxpr,
  JaxprDebugInfo as JaxprDebugInfo,
  JaxprEqn as JaxprEqn,
  JaxprPpContext as JaxprPpContext,
  JaxprPpSettings as JaxprPpSettings,
  JaxprTypeError as JaxprTypeError,
  Literal as Literal,
  MainTrace as MainTrace,
  MapPrimitive as MapPrimitive,
  NameGatheringSubst as NameGatheringSubst,
  NamedShape as NamedShape,
  OutDBIdx as OutDBIdx,
  OutputType as OutputType,
  ParamDict as ParamDict,
  Primitive as Primitive,
  ShapedArray as ShapedArray,
  Sublevel as Sublevel,
  TRACER_LEAK_DEBUGGER_WARNING as TRACER_LEAK_DEBUGGER_WARNING,
  ThreadLocalState as ThreadLocalState,
  Token as Token,
  Trace as Trace,
  TraceStack as TraceStack,
  TraceState as TraceState,
  Tracer as Tracer,
  UnshapedArray as UnshapedArray,
  Value as Value,
  Var as Var,
  abstract_token as abstract_token,
  apply_todos as apply_todos,
  as_named_shape as as_named_shape,
  aval_mapping_handlers as aval_mapping_handlers,
  axis_frame as axis_frame,
  call as call,
  call_bind_with_continuation as call_bind_with_continuation,
  call_impl as call_impl,
  call_p as call_p,
  check_eqn as check_eqn,
  check_jaxpr as check_jaxpr,
  check_type as check_type,
  check_valid_jaxtype as check_valid_jaxtype,
  closed_call_p as closed_call_p,
  concrete_aval as concrete_aval,
  concrete_or_error as concrete_or_error,
  concretization_function_error as concretization_function_error,
  cur_sublevel as cur_sublevel,
  custom_typechecks as custom_typechecks,
  dedup_referents as dedup_referents,
  do_subst_axis_names_jaxpr as do_subst_axis_names_jaxpr,
  ensure_compile_time_eval as ensure_compile_time_eval,
  escaped_tracer_error as escaped_tracer_error,
  eval_context as eval_context,
  eval_jaxpr as eval_jaxpr,
  extend_axis_env as extend_axis_env,
  extend_axis_env_nd as extend_axis_env_nd,
  find_top_trace as find_top_trace,
  full_lower as full_lower,
  gensym as gensym,
  get_aval as get_aval,
  get_referent as get_referent,
  is_constant_dim as is_constant_dim,
  is_constant_shape as is_constant_shape,
  jaxpr_as_fun as jaxpr_as_fun,
  jaxpr_uses_outfeed as jaxpr_uses_outfeed,
  jaxprs_in_params as jaxprs_in_params,
  join_effects as join_effects,
  join_named_shapes as join_named_shapes,
  lattice_join as lattice_join,
  leaked_tracer_error as leaked_tracer_error,
  literalable_types as literalable_types,
  map_bind as map_bind,
  map_bind_with_continuation as map_bind_with_continuation,
  mapped_aval as mapped_aval,
  maybe_find_leaked_tracers as maybe_find_leaked_tracers,
  max_dim as max_dim,
  min_dim as min_dim,
  new_base_main as new_base_main,
  new_jaxpr_eqn as new_jaxpr_eqn,
  new_main as new_main,
  new_sublevel as new_sublevel,
  no_axis_name as no_axis_name,
  no_effects as no_effects,
  non_negative_dim as _deprecated_non_negative_dim,
  outfeed_primitives as outfeed_primitives,
  primal_dtype_to_tangent_dtype as primal_dtype_to_tangent_dtype,
  primitive_uses_outfeed as primitive_uses_outfeed,
  process_env_traces_call as process_env_traces_call,
  process_env_traces_map as process_env_traces_map,
  pytype_aval_mappings as pytype_aval_mappings,
  raise_as_much_as_possible as raise_as_much_as_possible,
  raise_to_shaped as raise_to_shaped,
  raise_to_shaped_mappings as raise_to_shaped_mappings,
  reset_trace_state as reset_trace_state,
  stash_axis_env as stash_axis_env,
  str_eqn_compact as str_eqn_compact,
  subjaxprs as subjaxprs,
  subst_axis_names as subst_axis_names,
  subst_axis_names_eqn as subst_axis_names_eqn,
  subst_axis_names_jaxpr as subst_axis_names_jaxpr,
  subst_axis_names_var as subst_axis_names_var,
  substitute_vars_in_output_ty as substitute_vars_in_output_ty,
  thread_local_state as thread_local_state,
  trace_state_clean as trace_state_clean,
  traverse_jaxpr_params as traverse_jaxpr_params,
  typecheck as typecheck,
  typecompat as typecompat,
  typematch as typematch,
  unmapped_aval as unmapped_aval,
  used_axis_names as used_axis_names,
  used_axis_names_jaxpr as used_axis_names_jaxpr,
  valid_jaxtype as valid_jaxtype,
)


from jax._src import core as _src_core
_deprecations = {
    # Added 2024-06-12
    "pp_aval": ("jax.core.pp_aval is deprecated.", _src_core.pp_aval),
    "pp_eqn": ("jax.core.pp_eqn is deprecated.", _src_core.pp_eqn),
    "pp_eqn_rules": ("jax.core.pp_eqn_rules is deprecated.", _src_core.pp_eqn_rules),
    "pp_eqns": ("jax.core.pp_eqns is deprecated.", _src_core.pp_eqns),
    "pp_jaxpr": ("jax.core.pp_jaxpr is deprecated.", _src_core.pp_jaxpr),
    "pp_jaxpr_eqn_range": ("jax.core.pp_jaxpr_eqn_range is deprecated.", _src_core.pp_jaxpr_eqn_range),
    "pp_jaxpr_skeleton": ("jax.core.pp_jaxpr_skeleton is deprecated.", _src_core.pp_jaxpr_skeleton),
    "pp_jaxprs": ("jax.core.pp_jaxprs is deprecated.", _src_core.pp_jaxprs),
    "pp_kv_pair": ("jax.core.pp_kv_pair is deprecated.", _src_core.pp_kv_pair),
    "pp_kv_pairs": ("jax.core.pp_kv_pairs is deprecated.", _src_core.pp_kv_pairs),
    "pp_var": ("jax.core.pp_var is deprecated.", _src_core.pp_var),
    "pp_vars": ("jax.core.pp_vars is deprecated.", _src_core.pp_vars),
    # Finalized 2024-05-13; remove after 2024-08-13
    "DimSize": (
        "jax.core.DimSize is deprecated. Use DimSize = int | Any.",
        None,
    ),
    "Shape": (
        "jax.core.Shape is deprecated. Use Shape = Sequence[int | Any].",
        None,
    ),
    # Finalized 2024-06-24; remove after 2024-09-24
    "canonicalize_shape": (
      "jax.core.canonicalize_shape is deprecated.", None,
    ),
    "dimension_as_value": (
      "jax.core.dimension_as_value is deprecated. Use jnp.array.", None,
    ),
    "definitely_equal": (
      "jax.core.definitely_equal is deprecated. Use ==.", None,
    ),
    "symbolic_equal_dim": (
      "jax.core.symbolic_equal_dim is deprecated. Use ==.", None,
    ),
    # Added Jan 8, 2024
    "non_negative_dim": (
      "jax.core.non_negative_dim is deprecated. Use max_dim(..., 0).", _deprecated_non_negative_dim,
    ),
}

import typing
if typing.TYPE_CHECKING:
  non_negative_dim = _deprecated_non_negative_dim
  pp_aval = _src_core.pp_aval
  pp_eqn = _src_core.pp_eqn
  pp_eqn_rules = _src_core.pp_eqn_rules
  pp_eqns = _src_core.pp_eqns
  pp_jaxpr = _src_core.pp_jaxpr
  pp_jaxpr_eqn_range = _src_core.pp_jaxpr_eqn_range
  pp_jaxpr_skeleton = _src_core.pp_jaxpr_skeleton
  pp_jaxprs = _src_core.pp_jaxprs
  pp_kv_pair = _src_core.pp_kv_pair
  pp_kv_pairs = _src_core.pp_kv_pairs
  pp_var = _src_core.pp_var
  pp_vars = _src_core.pp_vars
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _src_core
