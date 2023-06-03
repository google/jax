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
"""JAX APIs for exporting code for interoperation.

This module is used with jax2tf, but has no TensorFlow dependencies.
"""
import dataclasses
import functools
import itertools
import re
from typing import Any, Callable, List, Optional, Sequence, Union

from absl import logging

import jax
from jax import sharding

from jax._src import core
from jax._src import pjit
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib import xla_client
from jax._src.lib.mlir.dialects import stablehlo
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.lib.mlir.dialects import func as func_dialect

from jax.experimental.jax2tf import shape_poly

map = util.safe_map
zip = util.safe_zip

@dataclasses.dataclass
class Exported:
  """A lowered and serialized JAX function.

  Currently this works only for functions that take a flat argument list
  and return a tuple of results. (No pytree support yet.)

  Attributes:
    in_avals: the input abstract values. May contain dimension expressions in
        the shapes.
    out_avals: the output abstract values. May contain dimension expressions in
        the shapes, with dimension variables among those in `in_avals`.
    in_shardings: the input shardings. Only for the `module_kept_var_idx`.
    out_shardings: the output shardings.
    lowering_platform: one of 'tpu', 'cpu', 'cuda', 'rocm'

    mlir_module_serialized: the serialized lowered VHLO module.
    mlir_module_version: a version number for the serialized module.
        The following version numbers are valid:
           4 - mlir_module_serialized is a portable artifact.
    module_kept_var_idx: the sorted indices of the arguments among `in_avals` that
        must be passed to the module. The other arguments have been dropped
        because they are not used. Same length as `in_shardings`.
    strict_checks: whether the module was serialized with the following safety
        checking: (A) the lowered computation can only be executed on a platform
        for which it was lowered; (B) the serialized computation contains only
        custom calls with targets that are guaranteed to be stable, (more to come).
    _get_vjp: an optional function that takes the current exported function and
        returns the exported VJP function.
        The VJP function takes a flat list of arguments,
        starting with the primal arguments and followed by a cotangent argument
        for each primal output. It returns a tuple with the cotangents
        corresponding to the primal inputs.
  """
  in_avals: Sequence[core.ShapedArray]
  out_avals: Sequence[core.ShapedArray]
  in_shardings: Sequence[Union[sharding.XLACompatibleSharding, pxla.UnspecifiedValue]]
  out_shardings: Sequence[Union[sharding.XLACompatibleSharding, pxla.UnspecifiedValue]]
  lowering_platform: str
  strict_checks: bool

  mlir_module_serialized: bytes
  xla_call_module_version: int
  module_kept_var_idx: Sequence[int]

  _get_vjp: Optional[Callable[["Exported"], "Exported"]]

  @property
  def mlir_module(self) -> mlir.ir.Module:
    return xla_client._xla.mlir.deserialize_portable_artifact(self.mlir_module_serialized)

  def vjp(self) -> "Exported":
    """Gets the exported VJP.

    Returns None if not available, which can happen if the Exported has been
    loaded from an external format, without a VJP."""
    if self._get_vjp is None:
      raise ValueError("No VJP is available")
    return self._get_vjp(self)


def _default_jax_backend() -> str:
  # Canonicalize to turn into CUDA or ROCM
  return xb.canonicalize_platform(jax.default_backend())


def export_native(fun_jax: Callable,
                  args_avals: Sequence[core.ShapedArray], *,
                  lowering_platform: Optional[str],
                  strict_checks: bool) -> Exported:
  """Exports native serialization for a JAX function.

  At the moment works only for JAX functions that take a flat list of arguments
  and return a flat list of results.

  Args:
    fun_jax: the function to lower and serialize
    args_avals: the abstract values at which to lower.
    lowering_platform: one of 'tpu', 'cpu', 'cuda', 'rocm'
    strict_checks: whether to do strict safety checks. See Exported.strict_checks
        for more details.
  """
  arg_specs_jax = [
    jax.ShapeDtypeStruct(aval.shape, aval.dtype, named_shape=aval.named_shape)
    for aval in args_avals
  ]

  if not hasattr(fun_jax, "lower"):
    # We support convert(pjit(f_jax)) and convert(jit(f_jax)) but also
    # convert(f_jax), in which case a "jit" is implied. In that case we raise
    # an error if the lowered function contains non-replicated sharding annotations.
    wrapped_fun_jax = jax.jit(fun_jax)
    allow_non_replicated_sharding = False
  else:
    # If we have a pjit or pmap already we do not wrap with another, and we
    # allow shardings.
    wrapped_fun_jax = fun_jax  # type: ignore
    allow_non_replicated_sharding = True

  lowered = wrapped_fun_jax.lower(
      *arg_specs_jax,
      _experimental_lowering_platform=lowering_platform)._lowering  # type: ignore

  _check_lowered(lowered)

  mlir_module = lowered.stablehlo()
  if "kept_var_idx" in lowered.compile_args:
    module_kept_var_idx = tuple(sorted(lowered.compile_args["kept_var_idx"]))
  else:
    # For pmap
    module_kept_var_idx = tuple(range(len(args_avals)))

  if not all(core.is_constant_shape(a.shape) for a in args_avals):
    # All arguments are kept if we have dimension variables.
    assert len(module_kept_var_idx) == len(args_avals)
    mlir_module = _add_dim_arg_computation(mlir_module, args_avals)

  xla_call_module_version = 4
  mlir_str = mlir.module_to_bytecode(mlir_module)
  target_version = stablehlo.get_earliest_forward_compatible_version()
  mlir_module_serialized = xla_client._xla.mlir.serialize_portable_artifact(
      mlir_str, target_version)

  # Figure out the result types and shapes
  if "global_out_avals" in lowered.compile_args:
    # This is currently the case for pjit
    out_avals = lowered.compile_args["global_out_avals"]
  elif "shards" in lowered.compile_args:  # for PmapComputation
    out_avals = lowered.compile_args["shards"].out_sharded_avals
  else:
    out_avals = lowered.compile_args["out_avals"]

  # Log and then check the module.
  if logging.vlog_is_on(3):
    mlir_module_text = mlir.module_to_string(mlir_module)
    logmsg = f"version={xla_call_module_version} lowering_platform={lowering_platform}"
    logging.vlog(3, "Lowered JAX module: %s\n%s", logmsg, mlir_module_text)

  _check_module(mlir_module,
                allow_non_replicated_sharding=allow_non_replicated_sharding,
                allow_all_custom_calls=not strict_checks)

  return Exported(
      in_avals=args_avals,
      out_avals=out_avals,
      in_shardings=lowered.compile_args["in_shardings"],
      out_shardings=lowered.compile_args["out_shardings"],
      lowering_platform=lowering_platform or _default_jax_backend(),
      strict_checks=strict_checks,
      mlir_module_serialized=mlir_module_serialized,
      module_kept_var_idx=module_kept_var_idx,
      xla_call_module_version=xla_call_module_version,
      _get_vjp=lambda exported: _export_native_vjp(wrapped_fun_jax, exported))


def _add_dim_arg_computation(module: mlir.ir.Module,
                             args_avals: Sequence[core.ShapedArray]) -> mlir.ir.Module:
  """Wraps the lowered module with a new "main" that computes the dim args.

  JAX lowering in presence of shape polymorphism produces a `module` that
  takes one or more dimension arguments, specified using 0-dimensional tensors
  of type i32 or i64, followed by the regular array arguments.
  The dimension arguments correspond to the dimension variables appearing in
  the `args_avals`, in sorted order.

  Consider the lowering of a function with one array argument of type "f32[w, h]",
  where "w" and "h" are two dimension variables. The `module` will also
  contain two dimension arguments, corresponding to "h" and "w" respectively:

      func public main(arg_h: i32, arg_w: i32, arg: f32[?, ?]) {
        ...
      }

      we rename "main" to "_wrapped_jax_export_main" and add a new "main":

      func public main(arg: f32[?, ?]) {
         arg_h = hlo.get_dimension_size(arg, 1)
         arg_w = hlo.get_dimension_size(arg, 0)
         res = call _wrapped_jax_export_main(arg_h, arg_w, arg)
         return res
      }

  Args:
    module: the HLO module as obtained from lowering. May have a number of
      dimension arguments, followed by the kept array arguments.
    args_avals: the avals for all the arguments of the lowered function, which
      correspond to the array arguments of the `module`.

  Returns the wrapped module.
  """
  dim_vars = shape_poly.all_dim_vars(args_avals)

  # Make a new module, do not mutate the "module" because it may be cached
  context = mlir.make_ir_context()
  with context, ir.Location.unknown(context):
    new_module = ir.Module.parse(mlir.module_to_bytecode(module))
    symbol_table = ir.SymbolTable(new_module.operation)
    orig_main = symbol_table["main"]
    orig_main.attributes["sym_visibility"] = ir.StringAttr.get("private")
    orig_main_name = "_wrapped_jax_export_main"
    symbol_table.set_symbol_name(orig_main, orig_main_name)

    orig_input_types = orig_main.type.inputs
    nr_array_args = len(orig_input_types) - len(dim_vars)
    assert nr_array_args >= 0

    new_main_input_types = orig_input_types[- nr_array_args:]
    orig_output_types = orig_main.type.results

    ftype = ir.FunctionType.get(new_main_input_types, orig_output_types)
    new_main_op = func_dialect.FuncOp(
        "main", ftype, ip=ir.InsertionPoint.at_block_begin(new_module.body))
    new_main_op.attributes["sym_visibility"] = ir.StringAttr.get("public")
    try:
      new_main_op.arg_attrs = list(orig_main.arg_attrs)[- nr_array_args:]
    except KeyError:
      pass  # TODO: better detection if orig_main.arg_attrs does not exist
    try:
      new_main_op.result_attrs = orig_main.result_attrs
    except KeyError:
      pass
    symbol_table.insert(new_main_op)
    entry_block = new_main_op.add_entry_block()
    with ir.InsertionPoint(entry_block):
      orig_main_args: List[mlir.ir.Value] = []
      module_context = mlir.ModuleContext(
          "cpu", "cpu", sharding_impls.ShardingContext([]),
          source_info_util.new_name_stack(),
          [], itertools.count(1), [], module=new_module, context=context)
      ctx = mlir.LoweringRuleContext(module_context=module_context,
          primitive=None, avals_in=args_avals, avals_out=None,
          tokens_in=mlir.TokenSet(), tokens_out=None)
      dim_args = _compute_dim_args(ctx, args_avals, tuple(new_main_op.arguments),
                                  orig_input_types[:len(dim_vars)])
      # The first arguments are the dimension variable
      orig_main_args.extend(dim_args)
      # Then the array arguments
      orig_main_args.extend(new_main_op.arguments)
      call = func_dialect.CallOp(orig_output_types,
                                 ir.FlatSymbolRefAttr.get(orig_main_name),
                                 orig_main_args)
      func_dialect.ReturnOp(call.results)
    symbol_table.set_symbol_name(new_main_op, "main")
    return new_module


def _compute_dim_args(
    ctx: mlir.LoweringRuleContext,
    args_avals: Sequence[core.ShapedArray],
    array_args: Sequence[mlir.ir.Value],
    dim_arg_types: Sequence[mlir.ir.Type]) -> Sequence[mlir.ir.Value]:
  """Compute the values of the dimension arguments.

  Args:
    args_avals: the abstract values of the array arguments.
    array_args: the values of the array arguments.
    dim_arg_types: the desired types for the dimension arguments.

  Returns:
    the values of the dimension variables, in the sorted order of the
    dimension variables.
  """
  dim_vars = shape_poly.all_dim_vars(args_avals)
  dim_values = mlir.lower_fun(
      functools.partial(shape_poly.compute_dim_values, args_avals, dim_vars),
      multiple_results=True)(ctx, *array_args)
  res = []
  for dim_arg, dim_arg_type in zip(util.flatten(dim_values), dim_arg_types):
    if dim_arg.type != dim_arg_type:
      res.append(hlo.ConvertOp(dim_arg_type, dim_arg).result)
    else:
      res.append(dim_arg)
  return tuple(res)


def _check_lowered(lowered) -> None:
  if not isinstance(lowered, pxla.MeshComputation):
    raise NotImplementedError(f"serialization is supported only for pjit. {lowered}")

  if lowered.compile_args["host_callbacks"] or lowered.compile_args["keepalive"]:
    raise NotImplementedError("serialization of host_callbacks is not yet implemented")
  # Check that we do not see new compile_args. When we add a compile_args it is
  # safe to add it to the allowed_compile_args if it does not change the semantics
  # or the calling convention of the lowered module.
  allowed_compile_args = ["backend", "mesh", "global_in_avals",
      "global_out_avals", "in_shardings", "out_shardings", "kept_var_idx",
      "spmd_lowering", "auto_spmd_lowering",
      "tuple_args", "ordered_effects", "unordered_effects",
      "keepalive", "host_callbacks", "pmap_nreps", "committed", "device_assignment"]
  for compile_arg in lowered.compile_args.keys():
    if compile_arg not in allowed_compile_args:
      raise NotImplementedError(f"Unrecognized lowered.compile_args[{compile_arg}]")

  # We have not implemented support for some of the compile_args.
  not_implemented_msgs = []
  for compile_arg, check_value, err_msg in (
      ("spmd_lowering", lambda v: v, "True"),
      ("auto_spmd_lowering", lambda v: not v, "False"),
      # tuple_args is a compilation flag, does not affect lowering.
      ("tuple_args", lambda v: True, "N/A"),
      # Used for debug(ordered=True), changes the calling convention, but will
      # also set keepalive to non-empty.
      ("ordered_effects", lambda v: not v, "empty"),
      # unordered_effects do not change the calling convention. Those from
      # jax.debug will also result in keepalive being non-empty and unsupported
      # custom calls. The CallTfEffect is an exception, but we want to allow
      # that one.
      ("unordered_effects", lambda v: True, "N/A"),
      # used for TPU jax.debug, send/recv. Not supported yet.
      ("host_callbacks", lambda v: not v, "empty"),
      # used on all platforms for callbacks. Not supported yet.
      ("keepalive", lambda v: not v, "empty"),
      ("pmap_nreps", lambda v: v == 1, "1"),
  ):
    if compile_arg in lowered.compile_args:
      if not check_value(lowered.compile_args[compile_arg]):
        not_implemented_msgs.append(
            f"{compile_arg} must be {err_msg} and it is {lowered.compile_args[compile_arg]}")
  if not_implemented_msgs:
    raise NotImplementedError(
        "serialization error, unimplemented lowered.compile_args:\n" +
        "\n".join(not_implemented_msgs))

# These are the JAX custom call target names that are guaranteed to be stable.
# Their backwards compatibility is tested by back_compat_test.py.
_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE = [
    "Sharding", "SPMDFullToShardShape", "SPMDShardToFullShape",
    "ducc_fft", "cu_threefry2x32",
    # eigh on CPU
    "lapack_ssyevd", "lapack_dsyevd", "lapack_cheevd", "lapack_zheevd",
    # eigh on GPU
    "cusolver_syevj", "cusolver_syevd",
    # eigh on TPU
    "Eigh",
    # qr on CPU
    "lapack_sgeqrf", "lapack_dgeqrf", "lapack_cgeqrf", "lapack_zgeqrf",
    "lapack_sorgqr", "lapack_dorgqr", "lapack_cungqr", "lapack_zungqr",
    # qr on GPU
    "cusolver_geqrf", "cublas_geqrf_batched",
    "cusolver_geqrf", "cusolver_orgqr",
    # qr and svd on TPU
    "Qr", "ProductOfElementaryHouseholderReflectors",
    # TODO(atondwal, necula): add back_compat tests for lu on CPU/GPU
    # # lu on CPU
    # "lapack_sgetrf" , "lapack_dgetrf" , "lapack_cgetrf" , "lapack_zgetrf",
    # # lu on GPU
    # "cublas_getrf_batched", "cusolver_getrf",
    # "hipblas_getrf_batched", "hipsolver_getrf",
    # lu on TPU
    "LuDecomposition",
]

def _check_module(mod: mlir.ir.Module, *,
                  allow_non_replicated_sharding: bool,
                  allow_all_custom_calls: bool):
  """Run a number of checks on the module.

  Args:
    allow_non_replicated_sharding: whether the module is allowed to contain
      non_replicated sharding annotations.
    allow_all_custom_calls: whether we should allow all custom calls, or
      only those who we have explicitly marked as stable.
  """
  sharding_attr = mlir.ir.StringAttr.get("Sharding", mod.context)
  allowed_custom_call_targets_attrs = [
      mlir.ir.StringAttr.get(target, mod.context)
      for target in _CUSTOM_CALL_TARGETS_GUARANTEED_STABLE]
  disallowed_custom_call_ops: List[str] = []
  def check_sharding(op: mlir.ir.Operation, loc: mlir.ir.Location):
    if not allow_non_replicated_sharding:
      try:
        sharding = op.attributes["mhlo.sharding"]
      except KeyError:
        pass
      else:
        if mlir.ir.StringAttr(sharding).value not in ["{replicated}", ""]:
          raise ValueError(
              "Lowered function does not have a top-level pjit but it has"
              f" non-replicated sharding annotations, e.g., {op} at {loc}.\nSee"
              " https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#support-for-partitioning"
              " for a discussion."
          )

  def check_op(op: mlir.ir.Operation):
    op_name = op.operation.name
    if op_name == "func.func":
      check_sharding(op.operation, op.location)

    elif op_name == "stablehlo.custom_call":
      call_target_name_attr = op.operation.attributes["call_target_name"]
      if (not allow_all_custom_calls and
          call_target_name_attr not in allowed_custom_call_targets_attrs):
        disallowed_custom_call_ops.append(str(op))
      if call_target_name_attr == sharding_attr:
        check_sharding(op, op.location)

  def walk_operations(op):
    check_op(op)
    for region in op.operation.regions:
      for block in region:
        for op in block:
          walk_operations(op)

  walk_operations(mod)
  if disallowed_custom_call_ops:
    disallowed_custom_call_ops_str = "\n".join(disallowed_custom_call_ops)
    msg = ("Cannot serialize code with custom calls whose targets have no "
           "compatibility guarantees. Examples are:\n"
           f"{disallowed_custom_call_ops_str}.\n"
           "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-lowering-supports-only-select-custom-calls")
    raise ValueError(msg)

def _export_native_vjp(primal_fun_jax, primal: Exported) -> Exported:
  # Export the VJP of `fun_flat_jax`

  def fun_vjp_jax(*args_and_out_cts_flat_jax):
    # Takes a flat list of primals and output cotangents
    args_flat_jax, out_cts_flat_jax = util.split_list(args_and_out_cts_flat_jax, [len(primal.in_avals)])
    _, pullback_jax = jax.vjp(primal_fun_jax, *args_flat_jax)
    return pullback_jax(out_cts_flat_jax)

  vjp_in_avals = list(
      itertools.chain(primal.in_avals,
                      map(lambda a: a.at_least_vspace(), primal.out_avals)))

  # Expand in_shardings to all in_avals even not kept ones.
  all_in_shardings = [sharding_impls.UNSPECIFIED] * len(primal.in_avals)
  for idx, in_s in zip(sorted(primal.module_kept_var_idx),
                       primal.in_shardings):
    all_in_shardings[idx] = in_s  # type: ignore
  all_shardings = all_in_shardings + list(primal.out_shardings)
  # Cannot mix unspecified and specified shardings. Make the unspecified
  # ones replicated.
  specified_shardings = [
      s for s in all_shardings if not sharding_impls.is_unspecified(s)]

  vjp_in_shardings: Any  # The primal inputs followed by output cotangents
  vjp_out_shardings: Any  # The primal output cotangents
  if 0 == len(specified_shardings):
    vjp_in_shardings = sharding_impls.UNSPECIFIED
    vjp_out_shardings = sharding_impls.UNSPECIFIED
  else:
    if len(specified_shardings) < len(all_shardings):
      # There are some specified, but not all; pjit front-end does not liwk
      in_s = specified_shardings[0]  # pjit will enforce that all have same devices
      assert isinstance(in_s, sharding.XLACompatibleSharding)
      replicated_s = sharding.GSPMDSharding.get_replicated(in_s._device_assignment)
      all_shardings = [
          s if not sharding_impls.is_unspecified(s) else replicated_s
          for s in all_shardings]

    vjp_in_shardings = tuple(all_shardings)
    vjp_out_shardings = tuple(all_shardings[:len(primal.in_avals)])
    if all(sharding_impls.is_unspecified(s) for s in vjp_out_shardings):
      vjp_out_shardings = sharding_impls.UNSPECIFIED

  fun_vjp_jax = pjit.pjit(fun_vjp_jax,
                          in_shardings=vjp_in_shardings,
                          out_shardings=vjp_out_shardings)

  return export_native(fun_vjp_jax, vjp_in_avals,
                       lowering_platform=primal.lowering_platform,
                       strict_checks=primal.strict_checks)
