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

from .lax import (
  ConvDimensionNumbers,
  DotDimensionNumbers,
  GatherDimensionNumbers,
  Precision,
  ScatterDimensionNumbers,
  abs,
  abs_p,
  acos,
  acosh,
  acosh_p,
  abs,
  abs_p,
  acos,
  acosh,
  acosh_p,
  add,
  add_p,
  after_all,
  after_all_p,
  and_p,
  asin,
  asinh,
  asinh_p,
  atan,
  atan2,
  atan2_p,
  atanh,
  atanh_p,
  batch_matmul,
  bessel_i0e,
  bessel_i0e_p,
  bessel_i1e,
  bessel_i1e_p,
  betainc,
  bitcast_convert_type,
  bitcast_convert_type_p,
  bitwise_and,
  bitwise_not,
  bitwise_or,
  bitwise_xor,
  broadcast,
  broadcast_in_dim,
  broadcast_in_dim_p,
  broadcast_p,
  broadcast_shapes,
  broadcast_to_rank,
  broadcasted_iota,
  ceil,
  ceil_p,
  clamp,
  clamp_p,
  collapse,
  complex,
  complex_p,
  concatenate,
  concatenate_p,
  conj,
  conj_p,
  conv,
  conv_dimension_numbers,
  conv_general_dilated,
  conv_general_dilated_p,
  conv_general_permutations,
  conv_general_shape_tuple,
  conv_shape_tuple,
  conv_transpose,
  conv_transpose_shape_tuple,
  conv_with_general_padding,
  convert_element_type,
  convert_element_type_p,
  cos,
  cos_p,
  cosh,
  cosh_p,
  create_token,
  create_token_p,
  cumprod,
  cumprod_p,
  cumsum,
  cumsum_p,
  digamma,
  digamma_p,
  div,
  div_p,
  dot,
  dot_general,
  dot_general_p,
  dtype,
  dtypes,
  dynamic_index_in_dim,
  dynamic_slice,
  dynamic_slice_in_dim,
  dynamic_slice_p,
  dynamic_update_index_in_dim,
  dynamic_update_slice,
  dynamic_update_slice_in_dim,
  dynamic_update_slice_p,
  eq,
  eq_p,
  erf,
  erf_inv,
  erf_inv_p,
  erf_p,
  erfc,
  erfc_p,
  exp,
  exp_p,
  expm1,
  expm1_p,
  floor,
  floor_p,
  full,
  full_like,
  gather,
  gather_p,
  ge,
  ge_p,
  gt,
  gt_p,
  igamma,
  igamma_grad_a,
  igamma_grad_a_p,
  igamma_p,
  igammac,
  igammac_p,
  imag,
  imag_p,
  index_in_dim,
  index_take,
  infeed,
  infeed_p,
  iota,
  is_finite,
  is_finite_p,
  itertools,
  le,
  le_p,
  lgamma,
  lgamma_p,
  log,
  log1p,
  log1p_p,
  log_p,
  lt,
  lt_p,
  lu,
  max,
  max_p,
  min,
  min_p,
  mul,
  mul_p,
  naryop,
  naryop_dtype_rule,
  ne,
  ne_p,
  neg,
  neg_p,
  nextafter,
  nextafter_p,
  not_p,
  or_p,
  outfeed,
  outfeed_p,
  pad,
  pad_p,
  padtype_to_pads,
  partial,
  population_count,
  population_count_p,
  pow,
  pow_p,
  prod,
  real,
  real_p,
  reciprocal,
  reduce,
  reduce_and_p,
  reduce_max_p,
  reduce_min_p,
  reduce_or_p,
  reduce_p,
  reduce_prod_p,
  reduce_sum_p,
  reduce_window,
  reduce_window_max_p,
  reduce_window_min_p,
  reduce_window_p,
  reduce_window_shape_tuple,
  reduce_window_sum_p,
  regularized_incomplete_beta_p,
  rem,
  rem_p,
  reshape,
  reshape_p,
  rev,
  rev_p,
  rng_uniform,
  rng_uniform_p,
  round,
  round_p,
  rsqrt,
  rsqrt_p,
  scatter,
  scatter_add,
  scatter_add_p,
  scatter_max,
  scatter_max_p,
  scatter_min,
  scatter_min_p,
  scatter_mul,
  scatter_mul_p,
  scatter_p,
  select,
  select_and_gather_add_p,
  select_and_scatter_add_p,
  select_and_scatter_p,
  select_p,
  shift_left,
  shift_left_p,
  shift_right_arithmetic,
  shift_right_arithmetic_p,
  shift_right_logical,
  shift_right_logical_p,
  sign,
  sign_p,
  sin,
  sin_p,
  sinh,
  sinh_p,
  slice,
  slice_in_dim,
  slice_p,
  sort,
  sort_key_val,
  sort_key_val_p,
  sort_p,
  sqrt,
  sqrt_p,
  square,
  standard_abstract_eval,
  standard_naryop,
  standard_primitive,
  standard_translate,
  standard_unop,
  stop_gradient,
  sub,
  sub_p,
  tan,
  tanh,
  tanh_p,
  tie_in,
  tie_in_p,
  top_k,
  top_k_p,
  transpose,
  transpose_p,
  unop,
  unop_dtype_rule,
  xor_p,
  zeros_like_array,
)
from .lax import (_reduce_sum, _reduce_max, _reduce_min, _reduce_or,
                  _reduce_and, _reduce_window_sum, _reduce_window_max,
                  _reduce_window_min, _reduce_window_prod,
                  _select_and_gather_add, _float, _complex, _input_dtype,
                  _const, _eq_meet, _broadcasting_select,
                  _check_user_dtype_supported, _one, _const,
                  _upcast_fp16_for_computation, _broadcasting_shape_rule,
                  _eye, _tri, _delta, _ones, _zeros)
from .lax_control_flow import (
  cond,
  cond_p,
  custom_linear_solve,
  custom_root,
  fori_loop,
  linear_solve_p,
  map,
  scan,
  scan_bind,
  scan_p,
  while_loop,
  while_p,
)
from .lax_fft import (
  fft,
  fft_p,
)
from .lax_parallel import (
  all_gather,
  all_to_all,
  all_to_all_p,
  axis_index,
  pmax,
  pmax_p,
  pmean,
  pmin,
  pmin_p,
  ppermute,
  ppermute_p,
  pshuffle,
  psum,
  psum_p,
  pswapaxes,
  standard_pmap_primitive,
)
