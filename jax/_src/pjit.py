# Copyright 2021 The JAX Authors.
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

from collections.abc import Sequence, Iterable
import dataclasses
from functools import partial, lru_cache
import inspect
import itertools as it
import logging
import operator as op
import weakref
from typing import (cast, Callable, NamedTuple, Any, Generic, Optional,
                    TypeVar, TYPE_CHECKING, Union)
import threading
import warnings

import numpy as np
if TYPE_CHECKING:
  from typing_extensions import ParamSpec

from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import op_shardings
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import stages
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.api_util import (
    argnums_partial_except, flatten_axes, flatten_fun, flatten_fun_nokwargs,
    donation_vector, shaped_abstractify, check_callable, resolve_argnums,
    argnames_partial_except, debug_info, result_paths, jaxpr_debug_info,
    hoist_obj_attrs)
from jax._src.errors import JAXTypeError
from jax._src.interpreters import partial_eval as pe
from jax._src.partition_spec import PartitionSpec
from jax._src.interpreters import xla

from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax._src.sharding_impls import (
    NamedSharding, XLACompatibleSharding, GSPMDSharding,
    XLADeviceAssignment, SingleDeviceSharding, PmapSharding,
    AUTO, UNSPECIFIED, UnspecifiedValue,
    ParsedPartitionSpec, SpecSync, get_single_pspec, is_auto, is_unspecified,
    is_unspecified_or_auto, prepare_axis_resources, parse_flatten_op_sharding)
from jax._src.state import discharge as state_discharge, RefEffect
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import (
    tree_flatten, tree_unflatten, treedef_is_leaf, tree_structure,
    treedef_children, broadcast_prefix, all_leaves, prefix_errors, keystr,
    PyTreeDef)
from jax._src.util import (
    HashableFunction, safe_map, safe_zip, wraps,
    distributed_debug_log, split_list, weakref_lru_cache,
    merge_lists, flatten, unflatten, subs_list)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

traceback_util.register_exclusion(__file__)

PjitSharding = Union[GSPMDSharding, UnspecifiedValue, AUTO]
PjitShardingMinusUnspecified = Union[GSPMDSharding, AUTO]
MeshSharding = Union[NamedSharding, UnspecifiedValue, AUTO]
MeshShardingMinusUnspecified = Union[NamedSharding, AUTO]

logger = logging.getLogger(__name__)


def _find_arg_mismatch(arg_list, fails, fun_name):
  mismatched_args_msg = []
  def mismatch(err):
    for name, inp_da, aval in arg_list:
      if err.m_type == pxla.MismatchType.ARG_SHARDING and err.da == inp_da:
        mismatched_args_msg.append(
            f"argument {name} of {fun_name} with shape {aval.str_short()} and "
            f"{err._dev_ids_plat_str}")
        break
  first_err, second_err = fails
  mismatch(first_err)
  mismatch(second_err)
  return mismatched_args_msg


def _device_assignment_mismatch_error(fun_name, fails, args_flat, api_name,
                                      arg_names):
  arg_list = []
  if arg_names is None:
    arg_names = [''] * len(args_flat)
  for a, n in zip(args_flat, arg_names):
    da = a.sharding._device_assignment if hasattr(a, 'sharding') else None
    arg_list.append((n, da, shaped_abstractify(a)))

  mismatched_args_msg = _find_arg_mismatch(arg_list, fails, fun_name)

  if len(mismatched_args_msg) == 2:
    first, second = mismatched_args_msg  # type: ignore
    extra_msg = f" Got {first} and {second}"
  elif len(mismatched_args_msg) == 1:
    first, second  = fails
    # Choose the failure left which is not already covered by ARG_SHARDING.
    left = second if first.m_type == pxla.MismatchType.ARG_SHARDING else first
    extra_msg = f" Got {mismatched_args_msg[0]} and{left._str(api_name)}"
  else:
    first, second = fails
    extra_msg = f" Got{first._str(api_name)} and{second._str(api_name)}"
  msg = (f"Received incompatible devices for {api_name}ted computation.{extra_msg}")
  return msg


class PjitInfo(NamedTuple):
  """Things that we know about a jit instance before it is called.

  In other words, this structure contains arguments to jit()/pjit(),
  preprocessed and validated.
  """
  fun: Callable
  fun_sourceinfo: str | None
  fun_signature: inspect.Signature | None
  # Shardings, as specified by the user. These can either be UNSPECIFIED or they
  # can be a tree (prefix) of shardings or None.
  user_specified_in_shardings: bool
  in_shardings_treedef: PyTreeDef
  in_shardings_leaves: tuple[Any, ...]
  out_shardings_treedef: PyTreeDef
  out_shardings_leaves: tuple[Any, ...]
  static_argnums: tuple[int, ...]
  static_argnames: tuple[str, ...]
  donate_argnums: tuple[int, ...]
  donate_argnames: tuple[str, ...]
  device: xc.Device | None
  backend: str | None
  keep_unused: bool
  inline: bool
  abstracted_axes: Any | None
  has_explicit_sharding: bool
  use_resource_env: bool  # False for jit, True for pjit


def _python_pjit_helper(jit_info, *args, **kwargs):
  args_flat, _, params, _, out_tree, _, _, _, arg_names, attrs_tracked = \
      _infer_params(jit_info, args, kwargs)
  for arg in args_flat:
    dispatch.check_arg(arg)

  if attrs_tracked:
    init_states = _get_states(attrs_tracked)
    args_flat = [*init_states, *args_flat]

  try:
    out_flat = pjit_p.bind(*args_flat, **params)
  except pxla.DeviceAssignmentMismatchError as e:
    fails, = e.args
    api_name = 'jit' if params['resource_env'] is None else 'pjit'
    fun = jit_info.fun
    fun_name = getattr(fun, '__qualname__', getattr(fun, '__name__', str(fun)))
    msg = _device_assignment_mismatch_error(
        fun_name, fails, args_flat, api_name, arg_names)
    raise ValueError(msg) from None
  except xla.InvalidInputException as e:
    arg_names = [''] * len(args_flat) if arg_names is None else arg_names
    # Run canonicalization again to figure out which arg failed.
    if params['jaxpr'].consts:
      raise TypeError(e.args[0]) from e
    else:
      for arg, name, aval in zip(args_flat, arg_names, params['jaxpr'].in_avals):
        try:
          xla.canonicalize_dtype(arg)
        except xla.InvalidInputException as _:
          # Reraise as TypeError with the new message.
          raise TypeError(
              f"Argument '{name}' of shape {aval.str_short()} of type"
              f' {type(arg)} is not a valid JAX type.') from e
      raise AssertionError("Unreachable") from e

  if attrs_tracked:
    final_states, out_flat = split_list(out_flat, [len(attrs_tracked)])
    _set_states(attrs_tracked, final_states)
  outs = tree_unflatten(out_tree, out_flat)
  return outs, out_flat, out_tree, args_flat, params['jaxpr'], attrs_tracked


def _set_states(attrs_tracked, vals):
  from jax.experimental.attrs import jax_setattr  # type: ignore
  for ((obj, attr), val) in zip(attrs_tracked, vals):
    jax_setattr(obj, attr, val)

def _get_states(attrs_tracked):
  from jax.experimental.attrs import jax_getattr  # type: ignore
  return [jax_getattr(obj, attr) for (obj, attr) in attrs_tracked]


def _get_fastpath_data(
    executable, out_tree, args_flat, out_flat, attrs_tracked, effects,
    abstracted_axes
) -> Optional[pxla.MeshExecutableFastpathData]:
  out_reflattened, out_tree = pxla.reflatten_outputs_for_dispatch(out_tree, out_flat)

  use_fastpath = (
      executable is not None
      and isinstance(executable, pxla.MeshExecutable)
      and isinstance(executable.unsafe_call, pxla.ExecuteReplicated)
      # No effects in computation
      and not executable.unsafe_call.ordered_effects
      and not executable.unsafe_call.has_unordered_effects
      and not executable.unsafe_call.has_host_callbacks
      and all(isinstance(x, xc.ArrayImpl) for x in out_reflattened)
      and abstracted_axes is None
      # no attr state effects
      and not attrs_tracked
      # no ref state effects
      and not any(isinstance(e, RefEffect) for e in effects)
      # no prng reuse checking
      and not (config.debug_key_reuse.value and any(
        hasattr(arg, 'dtype') and dtypes.issubdtype(arg.dtype, dtypes.prng_key)
        for arg in (*args_flat, *out_flat)))
      )

  if use_fastpath:
    out_avals = [o.aval for o in out_reflattened]
    out_committed = [o._committed for o in out_reflattened]
    kept_var_bitvec = [i in executable._kept_var_idx
                       for i in range(len(args_flat))]
    in_shardings = [
        a.dtype._rules.physical_sharding(a, s)
        if a is not core.abstract_token and dtypes.issubdtype(a.dtype, dtypes.extended)
        else s
        for s, a in zip(executable._in_shardings, executable.in_avals)
    ]
    fastpath_data = pxla.MeshExecutableFastpathData(
        executable.xla_executable, out_tree, in_shardings,
        executable._out_shardings, out_avals, out_committed, kept_var_bitvec,
        executable.unsafe_call.in_handler.local_devices,
        executable.unsafe_call.in_handler.input_indices)
  else:
    fastpath_data = None
  return fastpath_data


class _MostRecentPjitCallExecutable(threading.local):
  def __init__(self):
    self.weak_key_dict = weakref.WeakKeyDictionary()

_most_recent_pjit_call_executable = _MostRecentPjitCallExecutable()


def _read_most_recent_pjit_call_executable(jaxpr):
  return _most_recent_pjit_call_executable.weak_key_dict.get(jaxpr, None)


def _cpp_pjit_evict_fn(self):
  self._clear_cache()
  _create_pjit_jaxpr.evict_function(self._fun)  # type: ignore


# The entries are doubled here from the default 4096 because _pjit_call_impl
# also has a cpp dispatch path and that would double the number of entries in
# the global shared cache.
_cpp_pjit_cache = xc._xla.PjitFunctionCache(capacity=8192)


def _get_cpp_global_cache(pjit_has_explicit_sharding):
  if pjit_has_explicit_sharding:
    return xc._xla.PjitFunctionCache()
  else:
    return _cpp_pjit_cache


def _cpp_pjit(jit_info: PjitInfo):

  @api_boundary
  def cache_miss(*args, **kwargs):
    outs, out_flat, out_tree, args_flat, jaxpr, attrs_tracked = _python_pjit_helper(
        jit_info, *args, **kwargs)
    executable = _read_most_recent_pjit_call_executable(jaxpr)
    maybe_fastpath_data = _get_fastpath_data(
        executable, out_tree, args_flat, out_flat, attrs_tracked, jaxpr.effects,
        jit_info.abstracted_axes)
    return outs, maybe_fastpath_data

  fun = jit_info.fun
  if xla_extension_version >= 226:
    cpp_pjit_f = xc._xla.pjit(  # type: ignore
      getattr(fun, "__name__", "<unnamed function>"),
      fun, cache_miss, jit_info.static_argnums, jit_info.static_argnames,
      jit_info.donate_argnums, tree_util.dispatch_registry,
      pxla.shard_arg if xla_extension_version >= 229 else pxla.temp_shard_arg,  # type: ignore
      _get_cpp_global_cache(jit_info.has_explicit_sharding))  # type: ignore
  else:
    cpp_pjit_f = xc._xla.pjit(  # type: ignore
      getattr(fun, "__name__", "<unnamed function>"),
      fun, cache_miss, jit_info.static_argnums, jit_info.static_argnames,
      jit_info.donate_argnums, tree_util.dispatch_registry,
      _get_cpp_global_cache(jit_info.has_explicit_sharding))

  cpp_pjitted_f = wraps(fun)(cpp_pjit_f)
  cpp_pjitted_f._fun = fun
  type(cpp_pjitted_f).clear_cache = _cpp_pjit_evict_fn
  return cpp_pjitted_f


def _pjit_explicit_sharding(in_shardings, out_shardings, device,
                            backend) -> bool:
  in_shardings_flat, _ = tree_flatten(in_shardings)
  out_shardings_flat, _ = tree_flatten(out_shardings)
  return (device is not None or
          backend is not None or
          any(not is_unspecified(i) for i in in_shardings_flat) or
          any(not is_unspecified(i) for i in out_shardings_flat))


def _parse_jit_arguments(fun: Callable, in_shardings: Any, out_shardings: Any,
                         donate_argnums: int | Sequence[int] | None,
                         donate_argnames: str | Iterable[str] | None,
                         static_argnums: int | Sequence[int] | None,
                         static_argnames: str | Iterable[str] | None,
                         device: xc.Device | None, backend: str | None,
                         abstracted_axes: Any | None, keep_unused: bool,
                         inline: bool, use_resource_env: bool) -> PjitInfo:
  """Parses the arguments to jit/pjit.

  Performs any preprocessing and validation of the arguments that we can do
  ahead of time before the jit()-ed function is invoked.
  """
  if abstracted_axes and not config.dynamic_shapes.value:
    raise ValueError("abstracted_axes must be used with --jax_dynamic_shapes")

  check_callable(fun)

  if backend is not None or device is not None:
    warnings.warn(
        'backend and device argument on jit is deprecated. You can use a '
        '`jax.sharding.Mesh` context manager or device_put the arguments '
        'before passing them to `jit`. Please see '
        'https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html '
        'for more information.', DeprecationWarning)
    if device is not None and backend is not None:
      raise ValueError("can't specify both a device and a backend for jit, "
                       f"got {device=} and {backend=}")
    if in_shardings is not None and not is_unspecified(in_shardings):
      raise ValueError('If backend or device is specified on jit, then '
                       'in_shardings should not be specified.')
    if out_shardings is not None and not is_unspecified(out_shardings):
      raise ValueError('If backend or device is specified on jit, then '
                       'out_shardings should not be specified.')

  if isinstance(in_shardings, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/google/jax/issues/2367
    in_shardings = tuple(in_shardings)

  in_shardings, _, _ = prepare_axis_resources(in_shardings, 'in_shardings')
  out_shardings, _, _ = prepare_axis_resources(out_shardings, 'out_shardings')

  user_specified_in_shardings = (in_shardings is not None and
                                 not is_unspecified(in_shardings))
  none_leaf_registry = tree_util.none_leaf_registry
  in_shardings_leaves, in_shardings_treedef = none_leaf_registry.flatten(
      in_shardings)
  out_shardings_leaves, out_shardings_treedef = none_leaf_registry.flatten(
      out_shardings)

  fun_sourceinfo = api_util.fun_sourceinfo(fun)
  fun_signature = api_util.fun_signature(fun)

  donate_argnums, donate_argnames, static_argnums, static_argnames = resolve_argnums(
      fun, fun_signature, donate_argnums, donate_argnames, static_argnums,
      static_argnames)

  has_explicit_sharding = _pjit_explicit_sharding(
      in_shardings, out_shardings, device, backend)

  return PjitInfo(
        fun=fun,
        fun_sourceinfo=fun_sourceinfo,
        fun_signature=fun_signature,
        user_specified_in_shardings=user_specified_in_shardings,
        in_shardings_treedef=in_shardings_treedef,
        in_shardings_leaves=tuple(in_shardings_leaves),
        out_shardings_treedef=out_shardings_treedef,
        out_shardings_leaves=tuple(out_shardings_leaves),
        static_argnums=static_argnums,
        static_argnames=static_argnames, donate_argnums=donate_argnums,
        donate_argnames=donate_argnames, device=device, backend=backend,
        keep_unused=keep_unused, inline=inline,
        abstracted_axes=abstracted_axes,
        has_explicit_sharding=has_explicit_sharding,
        use_resource_env=use_resource_env)


def _make_jit_wrapper(jit_info: PjitInfo):
  wrapped = _cpp_pjit(jit_info)

  @api_boundary
  def lower(*args, **kwargs):
    lowering_parameters = kwargs.pop(
        '_experimental_lowering_parameters', mlir.LoweringParameters())
    # TODO(yashkatariya): Remove this when it's added on jit.
    in_layouts = kwargs.pop('_in_layouts', None)
    out_layouts = kwargs.pop('_out_layouts', None)
    (args_flat, flat_global_in_avals, params, in_tree, out_tree,
     donated_invars, in_layouts_flat, out_layouts_flat,
     arg_names, ()) = _infer_params(
         jit_info, args, kwargs, in_layouts=in_layouts, out_layouts=out_layouts)
    resource_env = params['resource_env']
    mesh = None if resource_env is None else resource_env.physical_mesh
    try:
      in_shardings = _resolve_in_shardings(
          args_flat, params['in_shardings'], params['out_shardings'], mesh)
      lowering = _pjit_lower(
          params['jaxpr'], in_shardings, params['out_shardings'],
          params['resource_env'], params['donated_invars'], params['name'],
          params['keep_unused'], params['inline'], in_layouts=in_layouts_flat,
          out_layouts=out_layouts_flat, lowering_parameters=lowering_parameters)
    except pxla.DeviceAssignmentMismatchError as e:
      fails, = e.args
      api_name = 'jit' if params['resource_env'] is None else 'pjit'
      fun = jit_info.fun
      fun_name = getattr(fun, '__qualname__',
                         getattr(fun, '__name__', str(fun)))
      msg = _device_assignment_mismatch_error(
          fun_name, fails, args_flat, api_name, arg_names)
      raise ValueError(msg) from None

    donate_argnums = tuple(i for i, d in enumerate(donated_invars) if d)
    return stages.Lowered.from_flat_info(
        lowering, in_tree, flat_global_in_avals, donate_argnums,
        out_tree)

  @api_boundary
  def eval_shape(*args, **kwargs):
    _, _, params, _, out_tree, _, _, _, _, _ = _infer_params(
        jit_info, args, kwargs, in_layouts=None, out_layouts=None
    )
    out_s = [None if is_unspecified(s) else getattr(s, '_original_sharding', s)
             for s in params['out_shardings']]
    out = [api.ShapeDtypeStruct(x.shape, x.dtype, x.named_shape, sharding=s)
           for x, s in zip(params['jaxpr'].out_avals, out_s)]
    return tree_unflatten(out_tree, out)

  wrapped.lower = lower
  wrapped.eval_shape = eval_shape
  return wrapped

def make_jit(fun: Callable, in_shardings: Any, out_shardings: Any,
             donate_argnums: int | Sequence[int] | None,
             donate_argnames: str | Iterable[str] | None,
             static_argnums: int | Sequence[int] | None,
             static_argnames: str | Iterable[str] | None,
             device: xc.Device | None, backend: str | None,
             abstracted_axes: Any | None, keep_unused: bool,
             inline: bool, use_resource_env: bool) -> Any:
  """jit() and pjit() are thin wrappers around this function."""
  jit_info = _parse_jit_arguments(
        fun, in_shardings, out_shardings, donate_argnums, donate_argnames,
        static_argnums, static_argnames, device, backend, abstracted_axes,
        keep_unused, inline, use_resource_env)
  return _make_jit_wrapper(jit_info)


def _infer_params(jit_info, args, kwargs, *, in_layouts=None, out_layouts=None):
  (fun, fun_sourceinfo, fun_signature, user_specified_in_shardings,
   in_shardings_treedef, in_shardings_leaves, out_shardings_treedef,
   out_shardings_leaves, static_argnums, static_argnames,
   donate_argnums, donate_argnames, device, backend, keep_unused, inline,
   abstracted_axes, _, use_resource_env) = jit_info

  have_kwargs = bool(kwargs)
  if have_kwargs and user_specified_in_shardings:
    raise ValueError(
        "pjit does not support kwargs when in_shardings is specified.")

  if use_resource_env:
    # We need to fetch the mesh from inside the wrapped function, because
    # meshes are dynamically scoped (i.e., with a context manager).
    resource_env = mesh_lib.thread_resources.env
    pjit_mesh = resource_env.physical_mesh
    jit_name = 'pjit'
    if (backend or device) and not pjit_mesh.empty:
      raise ValueError(
          "Mesh context manager should not be used with jit when backend or "
          "device is also specified as an argument to jit.")
  else:
    resource_env = None
    pjit_mesh = None
    jit_name = 'jit'


  axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)

  dbg = debug_info(jit_name, fun_sourceinfo, fun_signature, args, kwargs,
                   static_argnums, static_argnames)
  f = lu.wrap_init(fun)
  f, res_paths = result_paths(f)
  f, dyn_args = argnums_partial_except(f, static_argnums, args, allow_invalid=True)
  del args

  f, dyn_kwargs = argnames_partial_except(f, static_argnames, kwargs)
  explicit_args, in_tree = tree_flatten((dyn_args, dyn_kwargs))
  flat_fun, out_tree = flatten_fun(f, in_tree)
  flat_fun, explicit_args = hoist_obj_attrs(flat_fun, explicit_args)

  if (donate_argnums or donate_argnames) and not config.debug_nans.value:
    donated_invars = donation_vector(
        donate_argnums, donate_argnames, dyn_args, dyn_kwargs)
  else:
    donated_invars = (False,) * len(explicit_args)
  del donate_argnums, donate_argnames

  # If backend or device is set as an arg on jit, then resolve them to
  # in_shardings and out_shardings as if user passed in in_shardings
  # and out_shardings.
  device_or_backend_set = bool(backend or device)
  if device_or_backend_set:
    sharding = _create_sharding_with_device_backend(device, backend)
    leaves, treedef = tree_flatten(sharding)
    in_shardings_leaves = out_shardings_leaves = tuple(leaves)
    in_shardings_treedef = out_shardings_treedef = treedef
  else:
    in_shardings_leaves = tuple(
        _create_sharding_for_array(pjit_mesh, x, 'in_shardings', jit_name)
        for x in in_shardings_leaves)
    out_shardings_leaves = tuple(
        _create_sharding_for_array(pjit_mesh, x, 'out_shardings', jit_name)
        for x in out_shardings_leaves)

  assert None not in in_shardings_leaves
  assert None not in out_shardings_leaves

  if config.dynamic_shapes.value:
    in_type = pe.infer_lambda_input_type(axes_specs, explicit_args)
    in_avals = tuple(a for a, e in in_type if e)
  else:
    avals = []
    for i, a in enumerate(explicit_args):
      try:
        avals.append(shaped_abstractify(a))
      except OverflowError as e:
        arg_path = (f"argument path is {dbg.arg_names[i]}" if dbg
                    else f"flattened argument number is {i}")
        raise OverflowError(
          "An overflow was encountered while parsing an argument to a jitted "
          f"computation, whose {arg_path}."
        ) from e
    in_type = in_avals = tuple(avals)

  canonicalized_in_shardings_flat, in_layouts_flat = _process_in_axis_resources(
      in_shardings_treedef, in_shardings_leaves, hashable_pytree(in_layouts),
      in_avals, in_tree, dbg, device_or_backend_set, have_kwargs)

  jaxpr, consts, out_shardings, out_layouts_flat, attrs_tracked = _pjit_jaxpr(
      flat_fun, out_shardings_treedef, out_shardings_leaves,
      hashable_pytree(out_layouts), in_type, dbg, device_or_backend_set,
      HashableFunction(out_tree, closure=()),
      HashableFunction(res_paths, closure=()), inline)

  assert len(explicit_args) == len(canonicalized_in_shardings_flat) == len(in_layouts_flat)

  if config.dynamic_shapes.value:
    implicit_args = _extract_implicit_args(in_type, explicit_args)
  else:
    implicit_args = []
  args_flat = [*implicit_args, *explicit_args]

  num_extra_args = len(implicit_args) + len(attrs_tracked) + len(consts)
  canonicalized_in_shardings_flat = \
      (UNSPECIFIED,) * num_extra_args + canonicalized_in_shardings_flat
  in_layouts_flat = (None,) * num_extra_args + in_layouts_flat
  donated_invars = (False,) * num_extra_args + donated_invars
  assert (len(canonicalized_in_shardings_flat) == len(in_layouts_flat) ==
          len(donated_invars) == len(attrs_tracked) + len(consts) + len(args_flat))

  # in_shardings and out_shardings here are all GSPMDSharding.
  params = dict(
      jaxpr=jaxpr,
      in_shardings=canonicalized_in_shardings_flat,
      out_shardings=out_shardings,
      resource_env=resource_env,
      donated_invars=donated_invars,
      name=getattr(flat_fun, '__name__', '<unknown>'),
      keep_unused=keep_unused,
      inline=inline,
  )
  return (consts + args_flat, in_type, params, in_tree, out_tree(),
          donated_invars, in_layouts_flat, out_layouts_flat,
          dbg.arg_names if dbg else None, attrs_tracked)

def _extract_implicit_args(
  in_type: Sequence[tuple[core.AbstractValue, bool]],
  explicit_args: Sequence[Any]
) -> Sequence[core.Tracer]:
  """
  Given an input type and explicitly-passed arguments (per the user-facing API
  calling convention), extract implicit axis size arguments from shapes of
  explicit arguments (for the trace-time / jaxpr-level calling convention).
  """
  # First, using `in_type` construct a list to represent the full argument list,
  # leaving the implicit arguments as None placeholders for now.
  explicit_args_ = iter(explicit_args)
  args = [next(explicit_args_) if expl else None for _, expl in in_type]
  assert next(explicit_args_, None) is None
  del explicit_args, explicit_args_

  # Next, populate the implicit arguments using the DBIdxs in `in_type`.
  for i, (aval, explicit) in enumerate(in_type):
    if not explicit or not isinstance(aval, core.DShapedArray):
      continue  # can't populate an implicit argument
    arg = args[i]
    assert arg is not None
    for d1, d2 in zip(aval.shape, arg.aval.shape):
      if isinstance(d1, core.DBIdx):
        if args[d1.val] is None:
          args[d1.val] = d2
        assert core.same_referent(args[d1.val], d2)
  assert all(x is not None for x in args)
  return [x for x, (_, e) in zip(args, in_type) if not e]  # type: ignore

def _flat_axes_specs(abstracted_axes, *args, **kwargs
                     ) -> list[pe.AbstractedAxesSpec] | None:
  if abstracted_axes is None: return None
  if kwargs: raise NotImplementedError
  def ax_leaf(l):
    return (isinstance(l, dict) and all_leaves(l.values()) or
            isinstance(l, tuple) and all_leaves(l, lambda x: x is None))
  return broadcast_prefix(abstracted_axes, args, ax_leaf)



V_co = TypeVar("V_co", covariant=True)
if TYPE_CHECKING:
  P = ParamSpec("P")
else:
  P = TypeVar("P")


class JitWrapped(stages.Wrapped[P, V_co], Generic[P, V_co]):

  def eval_shape(self, *args, **kwargs):
    """See ``jax.eval_shape``."""
    raise NotImplementedError


# in_shardings and out_shardings can't be None as the default value
# because `None` means that the input is fully replicated.
def pjit(
    fun: Callable,
    in_shardings=UNSPECIFIED,
    out_shardings=UNSPECIFIED,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: xc.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any | None = None,
) -> JitWrapped:
  """Makes ``fun`` compiled and automatically partitioned across multiple devices.

  NOTE: This function is now equivalent to jax.jit please use that instead.
  The returned function has semantics equivalent to those of ``fun``, but is
  compiled to an XLA computation that runs across multiple devices
  (e.g. multiple GPUs or multiple TPU cores). This can be useful if the jitted
  version of ``fun`` would not fit in a single device's memory, or to speed up
  ``fun`` by running each operation in parallel across multiple devices.

  The partitioning over devices happens automatically based on the
  propagation of the input partitioning specified in ``in_shardings`` and
  the output partitioning specified in ``out_shardings``. The resources
  specified in those two arguments must refer to mesh axes, as defined by
  the :py:func:`jax.sharding.Mesh` context manager. Note that the mesh
  definition at :func:`~pjit` application time is ignored, and the returned function
  will use the mesh definition available at each call site.

  Inputs to a :func:`~pjit`'d function will be automatically partitioned across devices
  if they're not already correctly partitioned based on ``in_shardings``.
  In some scenarios, ensuring that the inputs are already correctly pre-partitioned
  can increase performance. For example, if passing the output of one
  :func:`~pjit`'d function to another :func:`~pjit`’d function (or the same
  :func:`~pjit`’d function in a loop), make sure the relevant
  ``out_shardings`` match the corresponding ``in_shardings``.

  .. note::
    **Multi-process platforms:** On multi-process platforms such as TPU pods,
    :func:`~pjit` can be used to run computations across all available devices across
    processes. To achieve this, :func:`~pjit` is designed to be used in SPMD Python
    programs, where every process is running the same Python code such that all
    processes run the same :func:`~pjit`'d function in the same order.

    When running in this configuration, the mesh should contain devices across
    all processes. However, any input argument dimensions partitioned over
    multi-process mesh axes should be of size equal to the corresponding *local*
    mesh axis size, and outputs will be similarly sized according to the local
    mesh. ``fun`` will still be executed across *all* devices in the mesh,
    including those from other processes, and will be given a global view of the
    data spread across multiple processes as a single array. However, outside
    of :func:`~pjit` every process only "sees" its local piece of the input and output,
    corresponding to its local sub-mesh.

    This means that each process's participating local devices must form a
    _contiguous_ local sub-mesh within the full global mesh. A contiguous
    sub-mesh is one where all of its devices are adjacent within the global
    mesh, and form a rectangular prism.

    The SPMD model also requires that the same multi-process :func:`~pjit`'d
    functions must be run in the same order on all processes, but they can be
    interspersed with arbitrary operations running in a single process.

  Args:
    fun: Function to be compiled. Should be a pure function, as side-effects may
      only be executed once. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.
    in_shardings: Pytree of structure matching that of arguments to ``fun``,
      with all actual arguments replaced by resource assignment specifications.
      It is also valid to specify a pytree prefix (e.g. one value in place of a
      whole subtree), in which case the leaves get broadcast to all values in
      that subtree.

      The ``in_shardings`` argument is optional. JAX will infer the shardings
      from the input :py:class:`jax.Array`'s, and defaults to replicating the input
      if the sharding cannot be inferred.

      The valid resource assignment specifications are:

      - :py:class:`XLACompatibleSharding`, which will decide how the value
        will be partitioned. With this, using a mesh context manager is not
        required.
      - :py:obj:`None` is a special case whose semantics are:
          - if the mesh context manager is *not* provided, JAX has the freedom to
            choose whatever sharding it wants.
            For in_shardings, JAX will mark is as replicated but this behavior
            can change in the future.
            For out_shardings, we will rely on the XLA GSPMD partitioner to
            determine the output shardings.
          - If the mesh context manager is provided, None will imply that the
            value will be replicated on all devices of the mesh.
      - For backwards compatibility, in_shardings still supports ingesting
        :py:class:`PartitionSpec`. This option can *only* be used with the
        mesh context manager.

        - :py:class:`PartitionSpec`, a tuple of length at most equal to the rank
          of the partitioned value. Each element can be a :py:obj:`None`, a mesh
          axis or a tuple of mesh axes, and specifies the set of resources assigned
          to partition the value's dimension matching its position in the spec.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it.
    out_shardings: Like ``in_shardings``, but specifies resource
      assignment for function outputs.
      The ``out_shardings`` argument is optional. If not specified, :py:func:`jax.jit`
      will use GSPMD's sharding propagation to determine how to shard the outputs.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If ``static_argnums`` is not provided, no arguments are treated as static.
    static_argnames: An optional string or collection of strings specifying
      which named arguments to treat as static (compile-time constant). See the
      comment on ``static_argnums`` for details. If not
      provided but ``static_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer
      need them once the computation has finished. In some cases XLA can make
      use of donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.

      If neither ``donate_argnums`` nor ``donate_argnames`` is provided, no
      arguments are donated. If ``donate_argnums`` is not provided but
      ``donate_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``donate_argnames``
      (or vice versa). If both ``donate_argnums`` and ``donate_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``donate_argnums`` or ``donate_argnames`` will
      be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
    donate_argnames: An optional string or collection of strings specifying
      which named arguments are donated to the computation. See the
      comment on ``donate_argnums`` for details. If not
      provided but ``donate_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.
    device: This argument is deprecated. Please put your arguments on the
      device you want before passing them to jit.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This argument is deprecated. Please put your arguments on the
      backend you want before passing them to jit.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation and
    automatically partitioned by the mesh available at each call site.

  For example, a convolution operator can be automatically partitioned over
  an arbitrary set of devices by a single :func:`~pjit` application:

  >>> import jax
  >>> import jax.numpy as jnp
  >>> import numpy as np
  >>> from jax.sharding import Mesh, PartitionSpec
  >>> from jax.experimental.pjit import pjit
  >>>
  >>> x = jnp.arange(8, dtype=jnp.float32)
  >>> f = pjit(lambda x: jax.numpy.convolve(x, jnp.asarray([0.5, 1.0, 0.5]), 'same'),
  ...         in_shardings=None, out_shardings=PartitionSpec('devices'))
  >>> with Mesh(np.array(jax.devices()), ('devices',)):
  ...   print(f(x))  # doctest: +SKIP
  [ 0.5  2.   4.   6.   8.  10.  12.  10. ]
  """
  return make_jit(
       fun, in_shardings, out_shardings, donate_argnums, donate_argnames,
       static_argnums, static_argnames, device, backend, abstracted_axes,
       keep_unused, inline, use_resource_env=True)


def hashable_pytree(pytree):
  vals, treedef = tree_flatten(pytree)
  vals = tuple(vals)
  return HashableFunction(lambda: tree_unflatten(treedef, vals),
                          closure=(treedef, vals))


def _create_sharding_for_array(mesh, x, name, api_name):
  if x is None and (mesh is None or mesh.empty):
    return UNSPECIFIED
  if isinstance(x, XLACompatibleSharding) or is_unspecified_or_auto(x):
    return x
  if mesh is None:
    msg = ('jax.jit only supports `XLACompatibleSharding`s being passed to'
           f' {name}. Looks like you are passing either `PartitionSpec` or `None`'
           f' which is not allowed in jax.jit.\n')
    if name == 'in_shardings':
      msg += (f'Note that {name} argument is optional. JAX will infer the shardings'
              " from the input jax.Array's and will default to replicating the"
              ' input if the sharding cannot be inferred.')
    elif name == 'out_shardings':
      msg += (f'Note that {name} is optional. If not specified, jax.jit will'
              " use GSPMD's sharding propagation to figure out what the sharding"
              ' of the output(s) should be.')
    raise RuntimeError(msg)
  if mesh.empty:
    raise RuntimeError(
        f'{api_name} requires a non-empty mesh if you are passing'
        f' `PartitionSpec`s or `None` to {name}! Is a mesh defined at the call'
        f' site? Alternatively, provide `XLACompatibleSharding`s to {name} and'
        ' then the mesh context manager is not required.')
  # A nice user error is raised in prepare_axis_resources.
  assert x is None or isinstance(x, ParsedPartitionSpec), x
  return (pxla.create_mesh_pspec_sharding(mesh, x)
          if x is None else pxla.create_mesh_pspec_sharding(mesh, x.user_spec, x))


def _create_sharding_with_device_backend(device, backend):
  if device is not None:
    assert backend is None
    out = SingleDeviceSharding(device)
  elif backend is not None:
    assert device is None
    out = SingleDeviceSharding(xb.get_backend(backend).local_devices()[0])
  return out


def flatten_axis_resources(what, tree, shardings, tupled_args):
  try:
    return tuple(flatten_axes(what, tree, shardings, tupled_args=tupled_args))
  except ValueError:
    pass  # Raise a tree prefix error below

  # Tree leaves are always valid prefixes, so if there was a prefix error as
  # assumed here, axis_resources must not be a leaf.
  assert not treedef_is_leaf(tree_structure(shardings))

  # Check the type directly rather than using isinstance because of namedtuples.
  if tupled_args and (type(shardings) is not tuple or
                      len(shardings) != len(tree.children())):
    # We know axis_resources is meant to be a tuple corresponding to the args
    # tuple, but while it is a non-leaf pytree, either it wasn't a tuple or it
    # wasn't the right length.
    msg = (f"{what} specification must be a tree prefix of the positional "
           f"arguments tuple passed to the `pjit`-decorated function. In "
           f"particular, {what} must either be a None, a PartitionSpec, or "
           f"a tuple of length equal to the number of positional arguments.")
    # If `tree` represents an args tuple, then `axis_resources` must be a tuple.
    # TODO(mattjj,apaszke): disable implicit list casts, remove 'or list' below
    if type(shardings) is not tuple:
      msg += f" But {what} is not a tuple: got {type(shardings)} instead."
    elif len(shardings) != len(tree.children()):
      msg += (f" But {what} is the wrong length: got a tuple or list of length "
              f"{len(shardings)} for an args tuple of length "
              f"{len(tree.children())}.")

    # As an extra hint, let's check if the user just forgot to wrap
    # shardings in a singleton tuple.
    if len(tree.children()) == 1:
      try: flatten_axes(what, tree, (shardings,))
      except ValueError: pass  # That's not the issue.
      else:
        msg += (f" Given the corresponding argument being "
                f"passed, it looks like {what} might need to be wrapped in "
                f"a singleton tuple.")

    raise ValueError(msg)

  axis_tree = shardings

  # Because we only have the `tree` treedef and not the full pytree here,
  # we construct a dummy tree to compare against. Revise this in callers?
  dummy_tree = tree_unflatten(tree, [PytreeLeaf()] * tree.num_leaves)
  errors = prefix_errors(axis_tree, dummy_tree)
  if errors:
    e = errors[0]  # Only show information about the first disagreement found.
    raise e(what)

  # At this point we've failed to find a tree prefix error.
  assert False, "Please open a bug report!"  # This should be unreachable.

class PytreeLeaf:
  def __repr__(self): return "pytree leaf"


@lru_cache(maxsize=4096)
def _process_in_axis_resources(in_shardings_treedef, in_shardings_leaves,
                               in_layouts_thunk, in_avals,
                               in_tree, debug_info,
                               device_or_backend_set, kws):
  if not kws:
    in_tree, _ = treedef_children(in_tree)

  orig_in_shardings = tree_unflatten(in_shardings_treedef, in_shardings_leaves)
  # Only do this if original in_shardings are unspecified. If it is AUTO, go
  # via flatten_axis_resources.
  if is_unspecified(orig_in_shardings):
    in_shardings_flat = (orig_in_shardings,) * len(in_avals)
  else:
    in_shardings_flat = flatten_axis_resources(
        "pjit in_shardings", in_tree, orig_in_shardings, tupled_args=True)

  in_layouts = in_layouts_thunk()
  if in_layouts is None:
    in_layouts_flat = (in_layouts,) * len(in_avals)
  else:
    in_layouts_flat = flatten_axis_resources(
        "pjit in_layouts", in_tree, in_layouts, tupled_args=True)

  # TODO(dougalm,mattjj): enable debug info with attrs_tracked
  attrs_tracked = debug_info and len(debug_info.arg_names) != len(in_avals)
  if not config.dynamic_shapes.value and not attrs_tracked:
    pjit_check_aval_sharding(in_shardings_flat, in_avals,
                             None if debug_info is None else debug_info.arg_names,
                             "pjit arguments", allow_uneven_sharding=False)
  canonicalized_shardings = tuple(
      i if is_unspecified_or_auto(i) else
      to_gspmd_sharding(i, aval.ndim, device_or_backend_set)
      for i, aval in zip(in_shardings_flat, in_avals))
  return canonicalized_shardings, tuple(in_layouts_flat)

callsites: set[str] = set()

def explain_tracing_cache_miss(
    f: Callable, unseen_f: bool, cache: dict, key: tuple, result: tuple):
  if config.check_tracer_leaks.value: return

  def unpack(key):
    transforms, (), _, (in_type, debug_info, _, inline), *_, ctx = key
    # TODO(dougalm,mattjj): enable cache miss explanation with attrs
    _, (_, (in_tree,)), (_, ()) = transforms
    return in_tree, in_type, debug_info, inline.val, ctx
  in_tree, in_type, debug_info, inline, ctx = unpack(key)
  if inline: return

  msg: list[str] = []
  p = msg.append
  done = lambda: logger.log(logging.WARNING, '\n'.join(msg))

  callsite = source_info_util.summarize(source_info_util.current())
  p(f"TRACING CACHE MISS at {callsite} because:")

  # have we seen this function before at all?
  fun_name = getattr(f, '__qualname__', f)
  if debug_info.func_src_info:
    _, _, *rest = debug_info.func_src_info.split(' ')
    src_info = " defined at "  + ' '.join(rest)
  else:
    src_info = ''
  if unseen_f:
    p(f"  never seen function:\n    {fun_name} id={id(f)}{src_info}")
    if callsite in callsites:
      p("  but seen another function defined on the same line; maybe the function is\n"
        "  being re-defined repeatedly, preventing caching?")
    callsites.add(callsite)
    return done()
  else:
    p(f"  for {fun_name}{src_info}")

  seen_keys = map(unpack, cache.keys())

  # have we maybe switched some args to be kwargs or visa-versa?
  args_tree, kwargs_tree = treedef_children(in_tree)
  args_kwargs_trees = [treedef_children(k) for k, *_ in seen_keys]
  args_kwargs_match = [t for t in args_kwargs_trees
                       if t == [args_tree, kwargs_tree]]
  if not args_kwargs_match:
    num_args = len(treedef_children(args_tree))
    _, kwarg_keys = kwargs_tree.node_data()  # type: ignore
    p(f"  never seen passing {num_args} positional args and {len(kwarg_keys)} "
      "keyword args with keys:\n"
      f"    {', '.join(map(repr, kwarg_keys))}")
    dont_match = [set(t[1].node_data()[1]) for t in args_kwargs_trees  # type: ignore
                  if t != [args_tree, kwargs_tree]]
    close_kwargs = min(dont_match, key=set(kwarg_keys).symmetric_difference)
    if not close_kwargs:
      p("  closest seen is passing no keyword args")
    else:
      p(f"  closest seen passes {len(close_kwargs)} keyword args with keys:\n"
        f"    {', '.join(map(repr, close_kwargs))}")
    return done()

  # have we never seen this tracing context before?
  ctxs_match = [c for *_, c in seen_keys if c == ctx]
  if not ctxs_match:
    p("  tracing context doesn't match, e.g. due to config or context manager")
    dont_match = [c for *_, c in seen_keys if c != ctx]
    closest_ctx = min(dont_match, key=lambda c: sum(map(op.ne, c, ctx)))
    idxs = [i for i, (c1, c2) in enumerate(zip(ctx, closest_ctx)) if c1 != c2]
    p("  closest seen context tuple differs at positions:\n"
      f"    {', '.join(map(str, idxs))}\n"
      "  compare to tuple returned by config._trace_context() in jax/_src/config.py.")
    return done()

  # have we never seen this input pytree before?
  trees_match = [k for k in seen_keys if k[0] == in_tree]
  if not trees_match:
    in_tree_str = f':\n    {in_tree}' if len(str(in_tree)) < 76 else ''
    p(f"  never seen input pytree{in_tree_str}")
    dont_match = [t for t, *_ in seen_keys if t != in_tree]
    closest_tree = min(dont_match, key=lambda t: abs(t.num_leaves - in_tree.num_leaves))
    # TODO(mattjj): make equality_errors not print type name, avoid metaclass
    leaf = type('LeafMeta', (type,), dict(__repr__=lambda _: 'leaf'))('Leaf', (), {})()
    this_dummy = tree_unflatten(in_tree, [leaf] * in_tree.num_leaves)
    close_dummy = tree_unflatten(closest_tree, [leaf] * closest_tree.num_leaves)  # type: ignore
    errs = list(tree_util.equality_errors(this_dummy, close_dummy))
    p(f"  closest seen input pytree has {len(errs)} mismatches, including:")
    for path, thing1, thing2, explanation in errs:
      fst, *path = path  # type: ignore
      base = ['args', 'kwargs'][fst.idx]
      p(f"    * at {base}{keystr(path)}, seen {thing2} but now given {thing1},"  # type: ignore
        f"      so {explanation}")
    return done()

  # have we never seen these input types (eg shapes, dtypes) before?
  types_match = [k for k in trees_match if k[1] == in_type]
  if not types_match:
    if len(in_type) < 5:
      in_type_str = ':\n    {}'.format(',  '.join(
          f'{n}: {ty.str_short(short_dtypes=True)}'
          for n, ty in zip(debug_info.arg_names, in_type)))
    else:
      in_type_str = ''
    p(f"  never seen input type signature{in_type_str}")
    dont_match = [t for _, t, *_ in trees_match if t != in_type]
    closest_ty = min(dont_match, key=lambda t: sum(map(op.ne, t, in_type)))
    num_mismatch = sum(map(op.ne, closest_ty, in_type))
    p(f"  closest seen input type signature has {num_mismatch} mismatches, including:")
    add_weak_type_hint = False
    for name, ty1, ty2 in zip(debug_info.arg_names, closest_ty, in_type):
      if ty1 != ty2:
        if type(ty1) == type(ty2) == core.ShapedArray:
          s1, s2 = ty1.str_short(True), ty2.str_short(True)
          if s1 == s2:  # weak types don't show up in str_short()
            assert ty1.weak_type ^ ty2.weak_type
            s1 += f'{{weak_type={ty1.weak_type}}}'
            s2 += f'{{weak_type={ty2.weak_type}}}'
            add_weak_type_hint = True
        else:
          s1, s2 = str(ty1), str(ty2)
        p(f"    * at {name}, seen {s1}, but now given {s2}")
    if add_weak_type_hint:
      p('where weak_type=True often means a Python builtin numeric value, and ')
      p('weak_type=False means a jax.Array.')
      p('See https://jax.readthedocs.io/en/latest/type_promotion.html#weak-types')
    return done()

  # we think this is unreachable...
  p("explanation unavailable! please open an issue at https://github.com/google/jax")
  return done()


@partial(lu.cache, explain=explain_tracing_cache_miss)
def _create_pjit_jaxpr(fun, in_type, debug_info, out_paths, ignored_inline):
  del ignored_inline  # just for explain_cache_miss
  with dispatch.log_elapsed_time(
      "Finished tracing + transforming {fun_name} for pjit in {elapsed_time} sec",
      fun_name=fun.__name__, event=dispatch.JAXPR_TRACE_EVENT):
    pe_debug = debug_info and pe.debug_info_final(fun, debug_info.traced_for)
    if config.dynamic_shapes.value:
      jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_dynamic2(
          lu.annotate(fun, in_type), debug_info=pe_debug)
      attrs_tracked = []
    else:
      jaxpr, global_out_avals, consts, attrs_tracked = pe.trace_to_jaxpr_dynamic(
          fun, in_type, debug_info=pe_debug)

  # TODO(dougalm,mattjj): enable debug info with attrs_tracked
  if not config.dynamic_shapes.value and not attrs_tracked:
    jaxpr = jaxpr_debug_info(jaxpr, debug_info, out_paths())

  if config.debug_key_reuse.value:
    # Import here to avoid circular imports
    from jax.experimental.key_reuse._core import check_key_reuse_jaxpr
    check_key_reuse_jaxpr(jaxpr)

  if any(isinstance(c, core.Tracer) for c in consts):
    closed_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
    final_consts = consts
  else:
    closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    final_consts = []
  return closed_jaxpr, final_consts, global_out_avals, attrs_tracked


@lru_cache(maxsize=4096)
def _check_and_canonicalize_out_shardings(
    out_shardings_treedef, out_shardings_leaves, out_layouts_thunk, out_tree,
    out_type, debug_info, device_or_backend_set):
  orig_out_shardings = tree_unflatten(out_shardings_treedef, out_shardings_leaves)
  # TODO(yashkatariya): Remove the if branch and fix flatten_axis_resources
  # instead. This condition exists because flatten_axis_resources passes in an
  # `object()` while unflattening which breaks assertion is user defined
  # pytrees (which shouldn't exist but they do).
  if (is_unspecified(orig_out_shardings) or
      isinstance(orig_out_shardings, XLACompatibleSharding)):
    out_shardings_flat = (orig_out_shardings,) * len(out_type)
  else:
    out_shardings_flat = flatten_axis_resources(
        "pjit out_shardings", out_tree(), orig_out_shardings,
        tupled_args=False)

  out_layouts = out_layouts_thunk()
  if out_layouts is None:
    out_layouts_flat = (out_layouts,) * len(out_type)
  else:
    out_layouts_flat = flatten_axis_resources(
        "pjit out_layouts", out_tree(), out_layouts, tupled_args=False)

  if not config.dynamic_shapes.value:
    pjit_check_aval_sharding(
        out_shardings_flat, out_type,
        None if debug_info is None else debug_info.result_paths,
        "pjit outputs", allow_uneven_sharding=False)

  canonicalized_out_shardings_flat = tuple(
      o if is_unspecified(o) or is_auto(o) else
      to_gspmd_sharding(o, aval.ndim, device_or_backend_set)
      for o, aval in zip(out_shardings_flat, out_type)
  )
  return canonicalized_out_shardings_flat, tuple(out_layouts_flat)


def _pjit_jaxpr(fun, out_shardings_treedef, out_shardings_leaves,
                out_layouts_thunk, in_type, debug_info, device_or_backend_set,
                out_tree, result_paths, inline):
  jaxpr, final_consts, out_type, attrs_tracked = _create_pjit_jaxpr(
      fun, in_type, debug_info, result_paths, IgnoreKey(inline))
  canonicalized_out_shardings_flat, out_layouts_flat = _check_and_canonicalize_out_shardings(
      out_shardings_treedef, out_shardings_leaves, out_layouts_thunk, out_tree, tuple(out_type),
      jaxpr.jaxpr.debug_info, device_or_backend_set)
  return jaxpr, final_consts, canonicalized_out_shardings_flat, out_layouts_flat, attrs_tracked


@dataclasses.dataclass(frozen=True)
class IgnoreKey:
  val: Any
  def __hash__(self):
    return hash(self.__class__)
  def __eq__(self, other):
    return isinstance(other, IgnoreKey)  # ignore self.val!


def pjit_check_aval_sharding(
    shardings, flat_avals, names: tuple[str, ...] | None,
    what_aval: str, allow_uneven_sharding: bool):
  new_names = [''] * len(shardings) if names is None else names
  for aval, s, name in zip(flat_avals, shardings, new_names):
    if is_unspecified_or_auto(s):
      continue
    s = getattr(s, '_original_sharding', s)
    name_str = f' with pytree key path {name}' if name else ''
    shape = aval.shape
    try:
      # Sharding interfaces can implement `is_compatible_aval` as an optional
      # method to raise a more meaningful error.
      if hasattr(s, 'is_compatible_aval'):
        s.is_compatible_aval(shape)
      else:
        s._to_xla_hlo_sharding(len(shape))
    except ValueError as e:
      raise ValueError(
          f'One of {what_aval}{name_str} is incompatible with its sharding '
          f'annotation {s}: {e}')
    # Use the `OpSharding` proto to find out how many ways each dimension of
    # the aval is sharded. This approach will work across all
    # XLACompatibleSharding.
    hlo_sharding = s._to_xla_hlo_sharding(len(shape))
    assert hlo_sharding is not None
    num_ways_dim_sharded, _ = op_shardings.get_num_ways_dim_sharded(hlo_sharding)
    for i, size in enumerate(num_ways_dim_sharded):
      if not allow_uneven_sharding and shape[i] % size != 0:
        raise ValueError(f"One of {what_aval}{name_str} was given the sharding "
                         f"of {s}, which implies that "
                         f"the global size of its dimension {i} should be "
                         f"divisible by {size}, but it is equal to {shape[i]} "
                         f"(full shape: {shape})")


# -------------------- pjit rules --------------------

pjit_p = core.AxisPrimitive("pjit")
pjit_p.multiple_results = True


def _resolve_in_shardings(
    args, pjit_in_shardings: Sequence[PjitSharding],
    out_shardings: Sequence[PjitSharding],
    pjit_mesh: pxla.Mesh | None,
    check_device_assignment: bool = True) -> Sequence[PjitSharding]:
  # If True, means that device or backend is set by the user on pjit and it
  # has the same semantics as device_put i.e. doesn't matter which device the
  # arg is on, reshard it to the device mentioned. So don't do any of the
  # checks and just return the pjit_in_shardings directly. `shard_args` will
  # handle the resharding.
  if pxla.check_device_backend_on_shardings(pjit_in_shardings):
    return pjit_in_shardings

  committed_arg_shardings = []
  for a in args:
    arg_s = getattr(a, 'sharding', None)
    # arg sharding can be None in case of ShapeDtypeStruct. jax.Array does
    # not allow None as the sharding.
    if arg_s is None:
      continue
    if not isinstance(arg_s, XLACompatibleSharding):
      raise ValueError(f'One of the argument to pjit got sharding {arg_s} '
                       'which is not a subclass of XLACompatibleSharding.')
    # Don't consider PmapSharding inputs as committed. They will get resharded
    # unconditionally.
    if isinstance(arg_s, PmapSharding):
      continue
    if getattr(a, '_committed', True):
      committed_arg_shardings.append((arg_s, pxla.MismatchType.ARG_SHARDING, None))

  # Check if the device_assignment across inputs, outputs and arguments is the
  # same.
  if check_device_assignment:
    pxla._get_and_check_device_assignment(
        it.chain(
            util.stable_unique(committed_arg_shardings),
            ((i, pxla.MismatchType.IN_SHARDING, None) for i in util.stable_unique(pjit_in_shardings)),
            ((o, pxla.MismatchType.OUT_SHARDING, None) for o in util.stable_unique(out_shardings))),
        (None if pjit_mesh is None or pjit_mesh.empty else list(pjit_mesh.devices.flat)))

  resolved_in_shardings = []
  for arg, pjit_in_s in zip(args, pjit_in_shardings):
    # arg sharding can be None in case of ShapeDtypeStruct. jax.Array does
    # not allow None as the sharding.
    arg_s, committed = ((arg.sharding, getattr(arg, '_committed', True))
                        if hasattr(arg, 'sharding') and arg.sharding is not None
                        else (UNSPECIFIED, False))
    if is_unspecified(pjit_in_s):
      if is_unspecified(arg_s):
        resolved_in_shardings.append(arg_s)
      else:
        if committed:
          # If the arg has a PmapSharding, then reshard it unconditionally.
          if isinstance(arg_s, PmapSharding):
            resolved_in_shardings.append(UNSPECIFIED)
          else:
            resolved_in_shardings.append(to_gspmd_sharding(
                cast(XLACompatibleSharding, arg_s), arg.ndim))
        else:
          if dispatch.is_single_device_sharding(arg_s):
            resolved_in_shardings.append(UNSPECIFIED)
          else:
            raise NotImplementedError('Having uncommitted Array sharded on '
                                      'multiple devices is not supported.')
    else:
      if (isinstance(arg, np.ndarray) and
          not pjit_in_s.is_fully_replicated and  # type: ignore
          xb.process_count() > 1):
        raise ValueError(
            'Passing non-trivial shardings for numpy '
            'inputs is not allowed. To fix this error, either specify a '
            'replicated sharding explicitly or use '
            '`jax.experimental.multihost_utils.host_local_array_to_global_array(...)` '
            'to convert your host local numpy inputs to a jax.Array which you '
            'can pass to pjit. '
            'If the numpy input is the same on each process, then you can use '
            '`jax.make_array_from_callback(...) to create a `jax.Array` which '
            'you can pass to pjit. '
            'Please see the jax.Array migration guide for more information '
            'https://jax.readthedocs.io/en/latest/jax_array_migration.html#handling-of-host-local-inputs-to-pjit-like-batch-etc. '
            f'Got arg shape: {arg.shape}, arg value: {arg}')
      if not is_unspecified(arg_s):
        # jax.jit does not allow resharding across different memory kinds even
        # if the argument is uncommitted. Use jax.device_put for those cases,
        # either outside or inside jax.jit.
        if pjit_in_s.memory_kind != arg_s.memory_kind:  # type: ignore
          raise ValueError(
              'Memory kinds passed to jax.jit does not match memory kind on the'
              f' respective arg. Got pjit memory kind: {pjit_in_s.memory_kind}, '  # type: ignore
              f'arg memory kind: {arg_s.memory_kind} for '  # type: ignore
              f'arg shape: {shaped_abstractify(arg).str_short()}')
        if (committed and
            not isinstance(arg_s, PmapSharding) and
            not op_shardings.are_op_shardings_equal(
                pjit_in_s._to_xla_hlo_sharding(arg.ndim),  # type: ignore
                arg_s._to_xla_hlo_sharding(arg.ndim))):
          op =  getattr(pjit_in_s, '_original_sharding', pjit_in_s)
          raise ValueError('Sharding passed to pjit does not match the sharding '
                           'on the respective arg. '
                           f'Got pjit sharding: {op},\n'
                           f'arg sharding: {arg_s} for '
                           f'arg shape: {shaped_abstractify(arg).str_short()}')
      resolved_in_shardings.append(pjit_in_s)

  return tuple(resolved_in_shardings)


def _pjit_call_impl_python(
    *args, jaxpr, in_shardings, out_shardings, resource_env, donated_invars,
    name, keep_unused, inline):
  global _most_recent_pjit_call_executable

  in_shardings = _resolve_in_shardings(
      args, in_shardings, out_shardings,
      resource_env.physical_mesh if resource_env is not None else None)

  compiled = _pjit_lower(
      jaxpr, in_shardings, out_shardings, resource_env,
      donated_invars, name, keep_unused, inline,
      lowering_parameters=mlir.LoweringParameters()).compile()
  _most_recent_pjit_call_executable.weak_key_dict[jaxpr] = compiled
  # This check is expensive so only do it if enable_checks is on.
  if compiled._auto_spmd_lowering and config.enable_checks.value:
    pxla.check_gda_or_array_xla_sharding_match(args, compiled._in_shardings,
                                               jaxpr.jaxpr.debug_info)
  if config.distributed_debug.value:
    # Defensively only perform fingerprint logic if debug logging is enabled
    # NOTE(skyewm): I didn't benchmark this
    fingerprint = None
    if hasattr(compiled.runtime_executable(), "fingerprint"):
      fingerprint = compiled.runtime_executable().fingerprint
    if fingerprint is not None:
      fingerprint = fingerprint.hex()
    distributed_debug_log(("Running pjit'd function", name),
                          ("in_shardings", in_shardings),
                          ("out_shardings", out_shardings),
                          ("abstract args", map(xla.abstractify, args)),
                          ("fingerprint", fingerprint))
  try:
    return compiled.unsafe_call(*args), compiled
  except FloatingPointError as e:
    assert config.debug_nans.value or config.debug_infs.value  # compiled_fun can only raise in this case

    if len(jaxpr.eqns) > 1:
      _ = core.jaxpr_as_fun(jaxpr)(*args)  # may raise, not return

    # If control reaches this line, we got a NaN on the output of `compiled`
    # but not `fun.call_wrapped` on the same arguments. Let's tell the user.
    msg = (f"{str(e)}. Because "
           "jax_config.debug_nans.value and/or config.jax_debug_infs is set, the "
           "de-optimized function (i.e., the function as if the `jit` "
           "decorator were removed) was called in an attempt to get a more "
           "precise error message. However, the de-optimized function did not "
           "produce invalid values during its execution. This behavior can "
           "result from `jit` optimizations causing the invalid value to be "
           "produced. It may also arise from having nan/inf constants as "
           "outputs, like `jax.jit(lambda ...: jax.numpy.nan)(...)`. "
           "\n\n"
           "It may be possible to avoid the invalid value by removing the "
           "`jit` decorator, at the cost of losing optimizations. "
           "\n\n"
           "If you see this error, consider opening a bug report at "
           "https://github.com/google/jax.")
    raise FloatingPointError(msg)


@weakref_lru_cache
def _get_jaxpr_as_fun(jaxpr, in_shardings, out_shardings, resource_env,
                      donated_invars, name, keep_unused, inline):
  # The input jaxpr to `_get_jaxpr_as_fun` is under a weakref_lru_cache so
  # returning `core.jaxpr_as_fun(jaxpr)` directly creates a strong reference to
  # the jaxpr defeating the purpose of weakref_lru_cache. So return a function
  # that closes over a weakrefed jaxpr and gets called inside that function.
  # This way there won't be a strong reference to the jaxpr from the output
  # function.
  jaxpr = weakref.ref(jaxpr)
  return lambda *args: core.jaxpr_as_fun(jaxpr())(*args)  # pylint: disable=unnecessary-lambda


def _pjit_call_impl(*args, jaxpr,
                    in_shardings, out_shardings, resource_env,
                    donated_invars, name, keep_unused, inline):
  def call_impl_cache_miss(*args_, **kwargs_):
    out_flat, compiled = _pjit_call_impl_python(
        *args, jaxpr=jaxpr, in_shardings=in_shardings,
        out_shardings=out_shardings, resource_env=resource_env,
        donated_invars=donated_invars, name=name, keep_unused=keep_unused,
        inline=inline)
    fastpath_data = _get_fastpath_data(
        compiled, tree_structure(out_flat), args, out_flat, [], jaxpr.effects,
        None)
    return out_flat, fastpath_data

  f = _get_jaxpr_as_fun(
      jaxpr, tuple(getattr(i, '_original_sharding', i) for i in in_shardings),
      tuple(getattr(o, '_original_sharding', o) for o in out_shardings),
      resource_env, donated_invars, name, keep_unused, inline)
  donated_argnums = [i for i, d in enumerate(donated_invars) if d]
  has_explicit_sharding = _pjit_explicit_sharding(
      in_shardings, out_shardings, None, None)
  if xla_extension_version >= 226:
    return xc._xla.pjit(
        name, f, call_impl_cache_miss, [], [], donated_argnums,
        tree_util.dispatch_registry,
        pxla.shard_arg if xla_extension_version >= 229 else pxla.temp_shard_arg,  # type: ignore
        _get_cpp_global_cache(has_explicit_sharding))(*args)
  else:
    return xc._xla.pjit(name, f, call_impl_cache_miss, [], [], donated_argnums,  # type: ignore
                        tree_util.dispatch_registry,
                        _get_cpp_global_cache(has_explicit_sharding))(*args)

pjit_p.def_impl(_pjit_call_impl)


@dataclasses.dataclass(frozen=True)
class SameDeviceAssignmentTuple:
  shardings: tuple[PjitSharding, ...]
  # device_assignment is Optional because shardings can contain `AUTO` and in
  # that case `mesh` is compulsory to be used. So in that case
  # `_pjit_lower_cached` cache, resource_env will check against the devices.
  device_assignment: XLADeviceAssignment | None

  def __hash__(self):
    shardings_hash = tuple(
        (s._hlo_sharding_hash, s.memory_kind)  # type: ignore
        if isinstance(s, GSPMDSharding) else s
        for s in self.shardings)
    if self.device_assignment is None:
      return hash(shardings_hash)
    else:
      return hash((shardings_hash, *self.device_assignment))

  def __eq__(self, other):
    if not isinstance(other, SameDeviceAssignmentTuple):
      return False
    eq = []
    for s, o in zip(self.shardings, other.shardings):
      s = getattr(s, "_original_sharding", s)
      o = getattr(o, "_original_sharding", o)
      if isinstance(s, GSPMDSharding) and isinstance(o, GSPMDSharding):
        eq.append(
            op_shardings.are_op_shardings_equal(s._hlo_sharding, o._hlo_sharding)
            and s.memory_kind == o.memory_kind)
      else:
        eq.append(s == o)
    return all(eq) and self.device_assignment == other.device_assignment


def _pjit_lower(
    jaxpr: core.ClosedJaxpr,
    in_shardings,
    out_shardings,
    *args, **kwargs):
  da = _fast_path_get_device_assignment(it.chain(in_shardings, out_shardings))
  in_shardings = SameDeviceAssignmentTuple(tuple(in_shardings), da)
  out_shardings = SameDeviceAssignmentTuple(tuple(out_shardings), da)
  return _pjit_lower_cached(jaxpr, in_shardings, out_shardings, *args, **kwargs)


@weakref_lru_cache
def _pjit_lower_cached(
    jaxpr: core.ClosedJaxpr,
    sdat_in_shardings: SameDeviceAssignmentTuple,
    sdat_out_shardings: SameDeviceAssignmentTuple,
    resource_env,
    donated_invars,
    name: str,
    keep_unused: bool,
    inline: bool,
    *,
    lowering_parameters: mlir.LoweringParameters,
    in_layouts: pxla.MaybeLayout | None = None,
    out_layouts: pxla.MaybeLayout | None = None):
  in_shardings: tuple[PjitShardingMinusUnspecified, ...] = cast(
      tuple[PjitShardingMinusUnspecified, ...], sdat_in_shardings.shardings)
  out_shardings: tuple[PjitSharding, ...] = sdat_out_shardings.shardings

  # TODO(yashkatariya): Remove this when layouts are supported on jit and
  # passed to params.
  if in_layouts is None:
    in_layouts = (None,) * len(in_shardings)
  if out_layouts is None:
    out_layouts = (None,) * len(out_shardings)

  if resource_env is not None:
    pxla.resource_typecheck(jaxpr, resource_env, {}, lambda: "pjit")

  if resource_env is not None:
    mesh = resource_env.physical_mesh
    api_name = 'pjit'
  else:
    # resource_env is `None` in the jit wrapper around pjit.
    mesh = None
    api_name = 'jit'

  # For `pjit(xmap)` cases, it needs to take the `lower_mesh_computation` path
  # because `xmap` only supports SPMDAxisContext right now.
  if dispatch.jaxpr_has_primitive(jaxpr.jaxpr, 'xmap'):
    return pxla.lower_mesh_computation(
      jaxpr, api_name, name, mesh,
      in_shardings, out_shardings, donated_invars,
      True, jaxpr.in_avals, tiling_method=None,
      lowering_parameters=lowering_parameters)
  else:
    return pxla.lower_sharding_computation(
        jaxpr, api_name, name, in_shardings, out_shardings,
        tuple(donated_invars), tuple(jaxpr.in_avals),
        keep_unused=keep_unused, inline=inline,
        devices_from_context=(
            None if mesh is None or mesh.empty else list(mesh.devices.flat)),
        lowering_parameters=lowering_parameters, in_layouts=in_layouts,
        out_layouts=out_layouts)


def pjit_staging_rule(trace, *args, **params):
  if (params["inline"] and
      all(is_unspecified(i) for i in params["in_shardings"]) and
      all(is_unspecified(o) for o in params["out_shardings"])):
    jaxpr = params['jaxpr']
    if config.dynamic_shapes.value:
      # Inline jaxpr doesn't handle dynamic shapes when inlining. If dynamic
      # shapes are enabled, use eval_jaxpr, which uses the tracing machinery,
      # but redundantly performs abstract evaluation again.
      return core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args,
                            propagate_source_info=False)
    else:
      return pe.inline_jaxpr_into_trace(trace, jaxpr.jaxpr, jaxpr.consts, *args)
  elif config.dynamic_shapes.value:
    source_info = source_info_util.current()
    out_tracers = []
    for aval in _out_type(params['jaxpr']):
      if type(aval) is core.DShapedArray:
        shape = [args[d.val] if type(d) is core.InDBIdx else
                 out_tracers[d.val] if type(d) is core.OutDBIdx else
                 d for d in aval.shape]
        aval = aval.update(shape=tuple(core.get_referent(d) for d in shape))
      out_tracers.append(pe.DynamicJaxprTracer(trace, aval, source_info))
    eqn = core.new_jaxpr_eqn(
      map(trace.getvar, args), map(trace.makevar, out_tracers), pjit_p, params,
      params['jaxpr'].effects, source_info)
    trace.frame.add_eqn(eqn)
    return out_tracers
  elif any(isinstance(c, core.MutableArray) for c in params['jaxpr'].consts):
    jaxpr, consts = pxla._move_mutable_consts(params['jaxpr'])
    consts = map(trace.instantiate_const, consts)
    in_shardings = (*params['in_shardings'],) + (UNSPECIFIED,) * len(consts)
    donated_invars = (*params['donated_invars'],) + (False,) * len(consts)
    new_params = dict(params, jaxpr=jaxpr, in_shardings=in_shardings,
                      donated_invars=donated_invars)
    return trace.default_process_primitive(pjit_p, (*args, *consts), new_params)
  else:
    return trace.default_process_primitive(pjit_p, args, params)
pe.custom_staging_rules[pjit_p] = pjit_staging_rule

# TODO(mattjj): remove/trivialize this when jaxprs have type annotation on them,
# since it's actually not possible in general to infer the type from the term
def _out_type(jaxpr: core.ClosedJaxpr) -> list[core.AbstractValue]:
  out = []
  in_idx = {v: i for i, v in enumerate(jaxpr.jaxpr.invars)}
  out_idx = {x: i for i, x in enumerate(jaxpr.jaxpr.invars)
             if type(x) is core.Var}
  for x in jaxpr.jaxpr.outvars:
    aval = x.aval
    if type(aval) is core.DShapedArray:
      shape = [core.InDBIdx(in_idx[d]) if d in in_idx else
               core.OutDBIdx(out_idx[d]) if d in out_idx else
               d for d in x.aval.shape]
      aval = aval.update(shape=tuple(shape))
    out.append(aval)
  return out


def _pjit_typecheck(ctx_factory, *in_atoms, jaxpr, **params):
  return core._check_call(ctx_factory, pjit_p, in_atoms,
                          dict(params, call_jaxpr=jaxpr.jaxpr))
core.custom_typechecks[pjit_p] = _pjit_typecheck


def _pjit_abstract_eval(*args, jaxpr, out_shardings, resource_env, **_):
  return jaxpr.out_avals, jaxpr.effects
pjit_p.def_effectful_abstract_eval(_pjit_abstract_eval)


def _pjit_cached_lower_jaxpr_to_fun(ctx, name, jaxpr, effects, in_shardings,
                                    out_shardings, api_name):
  mod_ctx = ctx.module_context
  axis_ctx = ctx.module_context.axis_context
  num_devices = None
  if isinstance(axis_ctx, sharding_impls.ShardingContext):
    num_devices = axis_ctx.num_devices
  elif isinstance(axis_ctx, sharding_impls.SPMDAxisContext):
    num_devices = axis_ctx.mesh.size
  key = (pjit_p, name, jaxpr, effects, num_devices,
         pxla.SemanticallyEqualShardings(in_shardings),
         pxla.SemanticallyEqualShardings(out_shardings), api_name)

  func = mod_ctx.cached_primitive_lowerings.get(key, None)
  if func is None:
    arg_shardings = [None if is_unspecified(i) else i for i in in_shardings]
    result_shardings = [None if is_unspecified(o) else o for o in out_shardings]
    # TODO(b/228598865): inlined calls cannot have shardings set directly on the
    # inputs or outputs because they are lost during MLIR->HLO conversion.
    # using_sharding_annotation=False means we add an identity operation instead.
    func = mlir.lower_jaxpr_to_fun(
        mod_ctx, name, jaxpr, effects, ctx.name_stack,
        arg_shardings=arg_shardings, result_shardings=result_shardings,
        use_sharding_annotations=False, api_name=api_name)
    mod_ctx.cached_primitive_lowerings[key] = func
  return func


def _pjit_lowering(ctx, *args, name, jaxpr, in_shardings,
                   out_shardings, resource_env, donated_invars,
                   keep_unused, inline):
  effects = list(ctx.tokens_in.effects())
  output_types = map(mlir.aval_to_ir_types, ctx.avals_out)
  output_types = [mlir.token_type()] * len(effects) + output_types
  flat_output_types = flatten(output_types)

  func = _pjit_cached_lower_jaxpr_to_fun(
      ctx, name, jaxpr, tuple(effects), in_shardings,
      out_shardings, api_name=('jit' if resource_env is None else 'pjit'))

  tokens_in = [ctx.tokens_in.get(eff) for eff in effects]
  args = (*ctx.dim_var_values, *tokens_in, *args)
  call = func_dialect.CallOp(flat_output_types,
                             ir.FlatSymbolRefAttr.get(func.name.value),
                             mlir.flatten_lowering_ir_args(args))
  out_nodes = unflatten(call.results, map(len, output_types))
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(zip(effects, tokens)))
  ctx.set_tokens_out(tokens_out)
  return out_nodes

mlir.register_lowering(pjit_p, _pjit_lowering)


def _pjit_batcher(insert_axis, spmd_axis_name,
                  axis_size, axis_name, main_type,
                  vals_in, dims_in,
                  jaxpr, in_shardings, out_shardings,
                  resource_env, donated_invars, name, keep_unused, inline):
  segment_lens, dims_in = batching.indirectify_ragged_axes(dims_in)
  new_jaxpr, axes_out = batching.batch_jaxpr2(
      jaxpr, axis_size, dims_in, axis_name=axis_name,
      spmd_axis_name=spmd_axis_name, main_type=main_type)

  # `insert_axis` is set to True only for some `xmap` uses.
  new_parts = (axis_name,) if insert_axis else (
      () if spmd_axis_name is None else spmd_axis_name)

  if resource_env is not None:
    mesh = resource_env.physical_mesh
  else:
    mesh = None

  # TODO(axch): prepend with Nones (?) to account for new segment_lens inputs
  in_shardings = tuple(
      _pjit_batcher_for_sharding(i, axis_in, new_parts, mesh, aval.ndim)
      if axis_in is not None else i
      for axis_in, i, aval in zip(dims_in, in_shardings, new_jaxpr.in_avals))
  out_shardings = tuple(
      _pjit_batcher_for_sharding(o, axis_out, new_parts, mesh, aval.ndim)
      if axis_out is not None else o
      for axis_out, o, aval in zip(axes_out, out_shardings, new_jaxpr.out_avals))
  vals_out = pjit_p.bind(
    *vals_in,
    jaxpr=new_jaxpr,
    in_shardings=in_shardings,
    out_shardings=out_shardings,
    resource_env=resource_env,
    donated_invars=donated_invars,
    name=name,
    keep_unused=keep_unused,
    inline=inline)
  resolved_axes_out = batching.resolve_ragged_axes_against_inputs_outputs(
      vals_in, vals_out, axes_out)
  return vals_out, resolved_axes_out

batching.spmd_axis_primitive_batchers[pjit_p] = partial(_pjit_batcher, False)
batching.axis_primitive_batchers[pjit_p] = partial(_pjit_batcher, False, None)
pxla.spmd_primitive_batchers[pjit_p] = partial(_pjit_batcher, True, None)

def _pjit_batcher_for_sharding(
    s: GSPMDSharding | UnspecifiedValue,
    dim: int, val: tuple[str, ...], mesh, ndim: int):
  if is_unspecified(s):
    return s
  if not val:
    if sharding_impls.is_op_sharding_replicated(s._hlo_sharding):  # type: ignore
      return s
    old_op = s._hlo_sharding.to_proto()  # type: ignore
    new_op = old_op.clone()  # type: ignore
    tad = list(new_op.tile_assignment_dimensions)
    tad.insert(dim, 1)
    new_op.tile_assignment_dimensions = tad
    if xla_extension_version >= 234:
      new_gs = GSPMDSharding(
          s._device_assignment, new_op,  # type: ignore
          _device_list=getattr(s, '_internal_device_list', None))
    else:
      new_gs = GSPMDSharding(s._device_assignment, new_op)  # type: ignore
    if hasattr(s, '_original_sharding'):
      vmapped_s = pxla._get_out_sharding_from_orig_sharding(
          [new_gs], [None], s._original_sharding, None)[0]  # type: ignore
      new_gs = to_gspmd_sharding(vmapped_s, ndim)
    return new_gs
  else:
    assert isinstance(s, GSPMDSharding)
    if isinstance(getattr(s, '_original_sharding', None), NamedSharding):
      mesh = s._original_sharding.mesh  # type: ignore
    if mesh is None or mesh.empty:
      s_type = (f', got: {s._original_sharding!r}'
                if hasattr(s, '_original_sharding') else '')
      raise ValueError(
          'If you are using xmap or spmd_axis_name parameter of jax.vmap,'
          ' please make sure to run your jitted function inside the mesh'
          ' context manager. Only `jax.lax.with_sharding_constraint` with'
          ' `jax.sharding.NamedSharding` as an input can be transformed with'
          ' spmd_axis_name batching rules outside of an explicit mesh context'
          f' manager scope{s_type}')
    parsed_pspec = parse_flatten_op_sharding(s._hlo_sharding, mesh)[0]  # type: ignore
    parsed_pspec = parsed_pspec.insert_axis_partitions(dim, val)
    mps = NamedSharding._from_parsed_pspec(mesh, parsed_pspec)
    if xla_extension_version >= 234:
      return GSPMDSharding(mps._device_assignment, mps._to_xla_hlo_sharding(ndim),
                           _device_list=getattr(mps, '_internal_device_list', None))
    else:
      return GSPMDSharding(mps._device_assignment, mps._to_xla_hlo_sharding(ndim))


def _pjit_jvp(primals_in, tangents_in,
              jaxpr, in_shardings, out_shardings,
              resource_env, donated_invars, name, keep_unused, inline):
  is_nz_tangents_in = [type(t) is not ad.Zero for t in tangents_in]
  jaxpr_jvp, is_nz_tangents_out = ad.jvp_jaxpr(
      jaxpr, is_nz_tangents_in, instantiate=False)

  def _filter_zeros(is_nz_l, l):
    return (x for nz, x in zip(is_nz_l, l) if nz)
  _filter_zeros_in = partial(_filter_zeros, is_nz_tangents_in)
  _filter_zeros_out = partial(_filter_zeros, is_nz_tangents_out)
  outputs = pjit_p.bind(
      *primals_in, *_filter_zeros_in(tangents_in),
      jaxpr=jaxpr_jvp,
      in_shardings=(*in_shardings, *_filter_zeros_in(in_shardings)),
      out_shardings=(*out_shardings, *_filter_zeros_out(out_shardings)),
      resource_env=resource_env,
      donated_invars=(*donated_invars, *_filter_zeros_in(donated_invars)),
      name=name,
      keep_unused=keep_unused,
      inline=inline)

  primals_out, tangents_out = split_list(outputs, [len(jaxpr.jaxpr.outvars)])
  assert len(primals_out) == len(jaxpr.jaxpr.outvars)
  tangents_out_it = iter(tangents_out)
  return primals_out, [next(tangents_out_it) if nz else ad.Zero(aval)
                       for nz, aval in zip(is_nz_tangents_out, jaxpr.out_avals)]
ad.primitive_jvps[pjit_p] = _pjit_jvp


@weakref_lru_cache
def _known_jaxpr_fwd(known_jaxpr: core.ClosedJaxpr,
                     in_fwd: tuple[int | None, ...]) -> core.ClosedJaxpr:
  updated_jaxpr = known_jaxpr.jaxpr.replace(
      outvars=[x for x, i in zip(known_jaxpr.jaxpr.outvars, in_fwd)
               if i is None])
  return known_jaxpr.replace(jaxpr=updated_jaxpr)


def _pjit_partial_eval(trace, *in_tracers,
                       jaxpr, in_shardings, out_shardings,
                       resource_env, donated_invars, name, keep_unused, inline):
  in_pvals = [t.pval for t in in_tracers]

  known_ins = tuple(pv.is_known() for pv in in_pvals)
  unknown_ins = tuple(not k for k in known_ins)
  known_jaxpr, unknown_jaxpr, unknown_outs, res_avals = \
      pe.partial_eval_jaxpr_nounits(jaxpr, unknown_ins, instantiate=False)
  unknown_outs = tuple(unknown_outs)
  known_outs = tuple(not uk for uk in unknown_outs)
  num_residuals = len(res_avals)
  res_shardings = (UNSPECIFIED,) * num_residuals

  def keep_where(l, should_keep):
    return tuple(x for x, keep in zip(l, should_keep) if keep)

  known_out_shardings = keep_where(out_shardings, known_outs) + res_shardings

  # Input-to-output forwarding: compute which outputs are just forwarded inputs.
  num_out_primals = len(known_jaxpr.out_avals) - num_residuals
  in_fwd: list[int | None] = pe._jaxpr_forwarding(known_jaxpr.jaxpr)
  # Only forward primal outputs when corresponding out_sharding is UNSPECIFIED.
  in_fwd_primal, in_fwd_res = split_list(in_fwd, [num_out_primals])
  in_fwd = [fwd if is_unspecified(os) else None for os, fwd in
            zip(keep_where(out_shardings, known_outs), in_fwd_primal)
            ] + in_fwd_res
  del in_fwd_primal, in_fwd_res
  # Prune jaxpr outputs and out_shardings by removing the input-forwards.
  keep = [f is None for f in in_fwd]
  known_jaxpr = pe.prune_closed_jaxpr_outputs(known_jaxpr, keep)
  known_out_shardings = keep_where(known_out_shardings, keep)
  # Update num_out_primals to reflect pruning.
  kept_primals, kept_res = split_list(keep, [num_out_primals])
  num_out_primals = sum(kept_primals)
  del keep, kept_primals, kept_res

  # Output-to-output forwarding: compute which residuals are just primal outputs
  out_vars, res_vars = split_list(known_jaxpr.jaxpr.outvars, [num_out_primals])
  idx_map = {id(v): i for i, v in enumerate(out_vars)}
  out_fwd = [None] * num_out_primals + [idx_map.get(id(v)) for v in res_vars]
  # Prune jaxpr outputs and out_shardings by removing forwarded residuals.
  keep = [f is None for f in out_fwd]
  known_jaxpr = pe.prune_closed_jaxpr_outputs(known_jaxpr, keep)
  known_out_shardings = keep_where(known_out_shardings, keep)
  del keep

  known_params = dict(
      jaxpr=known_jaxpr, in_shardings=keep_where(in_shardings, known_ins),
      out_shardings=known_out_shardings, resource_env=resource_env,
      donated_invars=keep_where(donated_invars, known_ins),
      name=name, keep_unused=keep_unused, inline=inline)
  assert len(known_params['out_shardings']) == len(known_params['jaxpr'].out_avals)

  # Bind known things to pjit_p.
  known_inputs = [pv.get_known() for pv in in_pvals if pv.is_known()]
  all_known_outs = pjit_p.bind(*known_inputs, **known_params)
  # Add back in the output fwds.
  all_known_outs = subs_list(out_fwd, all_known_outs, all_known_outs)
  # Add back in the input fwds.
  all_known_outs = subs_list(in_fwd, known_inputs, all_known_outs)

  known_out_vals, residual_vals = \
      split_list(all_known_outs, [len(all_known_outs) - num_residuals])
  residual_tracers = map(trace.new_instantiated_const, residual_vals)

  # The convention of partial_eval_jaxpr_nounits is to place residual binders at
  # the front of the jaxpr produced, so we move them to the back since both the
  # jaxpr equation built below and the pjit transpose rule assume a
  # residual-inputs-last convention.
  unknown_jaxpr = pe.move_binders_to_back(
      unknown_jaxpr, [True] * num_residuals + [False] * sum(unknown_ins))
  # Prepare unknown tracers
  unknown_params = dict(
      jaxpr=unknown_jaxpr,
      in_shardings=(keep_where(in_shardings, unknown_ins) + res_shardings),
      out_shardings=keep_where(out_shardings, unknown_outs),
      resource_env=resource_env,
      donated_invars=(keep_where(donated_invars, unknown_ins) +
                      (False,) * num_residuals),
      name=name,
      keep_unused=keep_unused,
      inline=inline)
  unknown_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  unknown_out_avals = unknown_jaxpr.out_avals
  unknown_tracers_out = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
      for aval in unknown_out_avals
  ]
  eqn = pe.new_eqn_recipe((*unknown_tracers_in, *residual_tracers),
                          unknown_tracers_out,
                          pjit_p,
                          unknown_params,
                          unknown_jaxpr.effects,
                          source_info_util.current())
  for t in unknown_tracers_out: t.recipe = eqn
  return merge_lists(unknown_outs, known_out_vals, unknown_tracers_out)

pe.custom_partial_eval_rules[pjit_p] = _pjit_partial_eval


def _pjit_partial_eval_custom_params_updater(
    unks_in: Sequence[bool], inst_in: Sequence[bool],
    kept_outs_known: Sequence[bool], kept_outs_staged: Sequence[bool],
    num_res_out: int, num_res_in: int, params_known: dict, params_staged: dict
  ) -> tuple[dict, dict]:
  # prune inputs to jaxpr_known according to unks_in
  donated_invars_known, _ = pe.partition_list(unks_in, params_known['donated_invars'])
  in_shardings_known, _ = pe.partition_list(unks_in, params_known['in_shardings'])
  _, out_shardings_known = pe.partition_list(kept_outs_known, params_known['out_shardings'])
  new_params_known = dict(params_known,
                          in_shardings=tuple(in_shardings_known),
                          out_shardings=(*out_shardings_known,
                                         *[UNSPECIFIED] * num_res_out),
                          donated_invars=tuple(donated_invars_known))
  assert len(new_params_known['in_shardings']) == len(params_known['jaxpr'].in_avals)
  assert len(new_params_known['out_shardings']) == len(params_known['jaxpr'].out_avals)

  # added num_res new inputs to jaxpr_staged, and pruning according to inst_in
  _, donated_invars_staged = pe.partition_list(inst_in, params_staged['donated_invars'])
  donated_invars_staged = [False] * num_res_in + donated_invars_staged
  _, in_shardings_staged = pe.partition_list(inst_in, params_staged['in_shardings'])
  in_shardings_staged = [*[UNSPECIFIED] * num_res_in, *in_shardings_staged]

  _, out_shardings_staged = pe.partition_list(kept_outs_staged, params_staged['out_shardings'])

  new_params_staged = dict(params_staged,
                           in_shardings=tuple(in_shardings_staged),
                           out_shardings=tuple(out_shardings_staged),
                           donated_invars=tuple(donated_invars_staged))
  assert len(new_params_staged['in_shardings']) == len(params_staged['jaxpr'].in_avals)
  assert len(new_params_staged['out_shardings']) == len(params_staged['jaxpr'].out_avals)
  return new_params_known, new_params_staged

pe.partial_eval_jaxpr_custom_rules[pjit_p] = \
    partial(pe.closed_call_partial_eval_custom_rule, 'jaxpr',
            _pjit_partial_eval_custom_params_updater)


@lu.cache
def _pjit_transpose_trace(fun, in_avals):
  transpose_jaxpr, _, consts, attrs_tracked = pe.trace_to_jaxpr_dynamic(
      fun, in_avals)
  transpose_jaxpr = core.ClosedJaxpr(transpose_jaxpr, consts)
  return transpose_jaxpr, attrs_tracked


def _pjit_transpose(cts_in, *primals_in,
                    jaxpr, in_shardings, out_shardings,
                    resource_env, donated_invars, name, keep_unused, inline):
  def prune_type(ty, xs, maybe_zeros):
    return tuple(x for x, mz in zip(xs, maybe_zeros) if type(mz) is not ty)

  body = lu.wrap_init(ad.closed_backward_pass)
  body = lu.hashable_partial(body, jaxpr, False)
  primals_and_nz_cts_in, in_treedef = tree_flatten((primals_in, cts_in))
  body, cts_out_treedef_thunk = flatten_fun_nokwargs(body, in_treedef)

  transpose_in_shardings = (
    *prune_type(ad.UndefinedPrimal, in_shardings, primals_in),
    *prune_type(ad.Zero, out_shardings, cts_in)
  )
  global_cts_in_avals = tuple(core.raise_to_shaped(core.get_aval(ct))
                              for ct in primals_and_nz_cts_in)

  transpose_jaxpr, attrs_tracked = _pjit_transpose_trace(
      body, global_cts_in_avals)
  cts_out_treedef = cts_out_treedef_thunk()
  transpose_out_shardings = prune_type(
      ad.Zero,
      in_shardings,
      tree_unflatten(cts_out_treedef, [object()] * cts_out_treedef.num_leaves))

  if attrs_tracked:
    init_states =  _get_states(attrs_tracked)
    primals_and_nz_cts_in = [*init_states, *primals_and_nz_cts_in]
    transpose_in_shardings = (UNSPECIFIED,) * len(attrs_tracked) + transpose_in_shardings
    transpose_out_shardings = (UNSPECIFIED,) * len(attrs_tracked) + transpose_out_shardings

  nz_cts_out = pjit_p.bind(
      *primals_and_nz_cts_in,
      jaxpr=transpose_jaxpr,
      in_shardings=transpose_in_shardings,
      out_shardings=transpose_out_shardings,
      resource_env=resource_env,
      donated_invars=(False,) * len(primals_and_nz_cts_in),
      name=name,
      keep_unused=keep_unused,
      inline=inline)
  if attrs_tracked:
    final_states, nz_cts_out = split_list(nz_cts_out, [len(init_states)])
    _set_states(attrs_tracked, final_states)
  return tree_unflatten(cts_out_treedef, nz_cts_out)
ad.reducing_transposes[pjit_p] = _pjit_transpose


@weakref_lru_cache
def _dce_jaxpr_pjit(
    jaxpr: core.ClosedJaxpr, used_outputs: tuple[bool, ...]
) -> tuple[core.ClosedJaxpr, list[bool]]:
  new_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr.jaxpr, used_outputs)
  return core.ClosedJaxpr(new_jaxpr, jaxpr.consts), used_inputs


def dce_jaxpr_pjit_rule(used_outputs: list[bool], eqn: core.JaxprEqn
                        ) -> tuple[list[bool], core.JaxprEqn | None]:
  dced_jaxpr, used_inputs = _dce_jaxpr_pjit(
      eqn.params['jaxpr'], tuple(used_outputs))

  def keep_where(xs, keeps):
    return tuple(x for x, keep in zip(xs, keeps) if keep)

  eqn_params = eqn.params
  new_params = dict(
      eqn_params,
      jaxpr=dced_jaxpr,
      in_shardings=keep_where(eqn_params["in_shardings"], used_inputs),
      out_shardings=keep_where(eqn_params["out_shardings"], used_outputs),
      donated_invars=keep_where(eqn_params["donated_invars"], used_inputs),
  )
  if not any(used_inputs) and not any(used_outputs) and not dced_jaxpr.effects:
    return used_inputs, None
  else:
    new_eqn = core.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, dced_jaxpr.effects, eqn.source_info)
    return used_inputs, new_eqn

pe.dce_rules[pjit_p] = dce_jaxpr_pjit_rule


def _check_resources_against_named_axes(what, aval, pos_axis_resources, named_axis_resources):
  pjit_resources = set(
      it.chain.from_iterable([d for d in pos_axis_resources if d is not None]))
  aval_resources = set(it.chain.from_iterable(
    named_axis_resources[a] for a in aval.named_shape))
  overlap = pjit_resources & aval_resources
  if overlap:
    raise JAXTypeError(
        f"{what} has an axis resources specification of "
        f"{pos_axis_resources.unsynced_user_spec(SpecSync.DIM_PERMUTE)} "
        f"that uses one or more mesh axes already used by xmap to partition "
        f"a named axis appearing in its named_shape (both use mesh axes "
        f"{mesh_lib.show_axes(overlap)})")

def _resource_typing_pjit(avals, params, source_info, resource_env, named_axis_resources):
  jaxpr = params["jaxpr"]
  what = "pjit input"
  if (resource_env is not None and params['resource_env'] is not None and
      resource_env.physical_mesh != params['resource_env'].physical_mesh):
      raise RuntimeError("Changing the physical mesh is not allowed inside pjit.")

  for aval, s in zip(jaxpr.in_avals, params['in_shardings']):
    if is_unspecified(s) or is_auto(s):
      continue
    elif hasattr(s, '_original_sharding') and hasattr(
        s._original_sharding, '_parsed_pspec'):
      parsed_pspec = s._original_sharding._parsed_pspec
    else:
      if resource_env is not None and not resource_env.physical_mesh.empty:
        parsed_pspec = parse_flatten_op_sharding(
            s._hlo_sharding, resource_env.physical_mesh)[0]
      else:
        parsed_pspec = None
    if parsed_pspec is not None:
      _check_resources_against_named_axes(what, aval, parsed_pspec,
                                          named_axis_resources)

  pxla.resource_typecheck(
      jaxpr.jaxpr, resource_env, named_axis_resources,
      lambda: (f"a pjit'ed function {params['name']} "
               f"(pjit called at {source_info_util.summarize(source_info)})"))

  what = "pjit output"
  for aval, s in zip(jaxpr.out_avals, params['out_shardings']):
    if is_unspecified(s) or is_auto(s):
      continue
    elif hasattr(s, '_original_sharding') and hasattr(
        s._original_sharding, '_parsed_pspec'):
      parsed_pspec = s._original_sharding._parsed_pspec
    else:
      if resource_env is not None and not resource_env.physical_mesh.empty:
        parsed_pspec = parse_flatten_op_sharding(
            s._hlo_sharding, resource_env.physical_mesh)[0]
      else:
        parsed_pspec = None
    if parsed_pspec is not None:
      _check_resources_against_named_axes(what, aval, parsed_pspec,
                                          named_axis_resources)

pxla.custom_resource_typing_rules[pjit_p] = _resource_typing_pjit


def _pjit_pp_rule(eqn, context, settings):
  params = dict(eqn.params)
  del params['inline']
  if not any(params['donated_invars']):
    del params['donated_invars']
  if all(is_unspecified(s) for s in params['in_shardings']):
    del params['in_shardings']
  if all(is_unspecified(s) for s in params['out_shardings']):
    del params['out_shardings']
  if not params['keep_unused']:
    del params['keep_unused']
  if (params['resource_env'] is None or
      params['resource_env'].physical_mesh.empty):
    del params['resource_env']

  # Move name= to the front to make the resulting equation easier to scan.
  del params["name"]
  return core._pp_eqn(eqn, context, settings, params=["name"] + sorted(params))

core.pp_eqn_rules[pjit_p] = _pjit_pp_rule


def _pjit_state_discharge_rule(
    in_avals, out_avals, *args, jaxpr, in_shardings, out_shardings, **params):
  if not (all(map(is_unspecified, in_shardings)) and
          all(map(is_unspecified, out_shardings))): raise NotImplementedError
  jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
  num_outs = len(jaxpr.outvars)
  discharged_jaxpr, discharged_consts = state_discharge.discharge_state(jaxpr, consts)
  discharged_closed_jaxpr = core.ClosedJaxpr(discharged_jaxpr, discharged_consts)
  new_in_shardings = (UnspecifiedValue(),) * len(discharged_jaxpr.invars)
  new_out_shardings = (UnspecifiedValue(),) * len(discharged_jaxpr.outvars)
  out_and_ref_vals = pjit_p.bind(
      *args, jaxpr=discharged_closed_jaxpr, in_shardings=new_in_shardings,
      out_shardings=new_out_shardings, **params)
  out_vals, ref_vals = split_list(out_and_ref_vals, [num_outs])
  ref_vals_iter = iter(ref_vals)
  new_invals = tuple(next(ref_vals_iter) if isinstance(aval, state_discharge.AbstractRef)
                     else None for aval in in_avals)
  sentinel = object()
  assert next(ref_vals_iter, sentinel) is sentinel
  return new_invals, out_vals
state_discharge.register_discharge_rule(pjit_p)(_pjit_state_discharge_rule)


# -------------------- with_sharding_constraint --------------------

def with_sharding_constraint(x, shardings):
  """Mechanism to constrain the sharding of an Array inside a jitted computation

  This is a strict constraint for the GSPMD partitioner and not a hint. For examples
  of how to use this function, see `Distributed arrays and automatic parallelization`_.

  Args:
    x: PyTree of jax.Arrays which will have their shardings constrained
    shardings: PyTree of sharding specifications. Valid values are the same as for
      the ``in_shardings`` argument of :func:`jax.experimental.pjit`.
  Returns:
    x_with_shardings: PyTree of jax.Arrays with specified sharding constraints.

  .. _Distributed arrays and automatic parallelization: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
  """
  x_flat, tree = tree_flatten(x)
  user_shardings, _, _ = prepare_axis_resources(
      shardings, "shardings", allow_unconstrained_dims=True)
  del shardings

  user_shardings_flat = tuple(
      flatten_axes("with_sharding_constraint shardings", tree, user_shardings))
  del user_shardings

  resource_env = mesh_lib.thread_resources.env
  mesh = resource_env.physical_mesh

  shardings_flat = [_create_sharding_for_array(mesh, a, 'shardings',
                                               'with_sharding_constraint')
                    for a in user_shardings_flat]
  unconstrained_dims = [get_unconstrained_dims(s)
                        if isinstance(s, NamedSharding) else {}
                        for s in shardings_flat]
  del user_shardings_flat

  pjit_check_aval_sharding(
      shardings_flat, x_flat, None, "with_sharding_constraint arguments",
      allow_uneven_sharding=True)

  outs = [sharding_constraint_p.bind(xf, sharding=to_gspmd_sharding(i, xf.ndim),
                                     resource_env=resource_env,
                                     unconstrained_dims=ud)
          for xf, i, ud in zip(x_flat, shardings_flat, unconstrained_dims)]
  return tree_unflatten(tree, outs)

def _identity_fn(x): return x

def _sharding_constraint_impl(x, sharding, resource_env, unconstrained_dims):
  if hasattr(x, 'sharding') and x.sharding.is_equivalent_to(sharding, x.ndim):
    return x
  # Run a jit here to raise good errors when device assignment don't match.
  return api.jit(_identity_fn, out_shardings=sharding)(x)


sharding_constraint_p = core.Primitive("sharding_constraint")
sharding_constraint_p.def_impl(_sharding_constraint_impl)
sharding_constraint_p.def_abstract_eval(lambda x, **_: x)
ad.deflinear2(sharding_constraint_p,
              lambda ct, _, **params: (sharding_constraint_p.bind(ct, **params),))

def _sharding_constraint_hlo_lowering(ctx, x_node, *, sharding,
                                      resource_env, unconstrained_dims):
  aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  axis_ctx = ctx.module_context.axis_context
  # axis_ctx and manual_axes is *only used with xmap* and xmap only works with
  # NamedSharding. So update the NamedSharding to have the manual axes.
  if isinstance(axis_ctx, sharding_impls.SPMDAxisContext):
    mesh = resource_env.physical_mesh
    parsed_pspec = parse_flatten_op_sharding(sharding._hlo_sharding, mesh)[0]
    sharding = NamedSharding._from_parsed_pspec(
        mesh, parsed_pspec, _manual_axes=axis_ctx.manual_axes)
  return [
      mlir.wrap_with_sharding_op(ctx,
          x_node, out_aval,
          sharding._to_xla_hlo_sharding(aval.ndim).to_proto(),
          unspecified_dims=unconstrained_dims)
  ]
mlir.register_lowering(sharding_constraint_p,
                       _sharding_constraint_hlo_lowering)


def _sharding_constraint_batcher(insert_axis, spmd_axis_name, axis_size,
                                 axis_name, main_type, vals_in, dims_in,
                                 sharding, resource_env, unconstrained_dims):
  x, = vals_in
  d, = dims_in
  # None means unconstrained in ParsedPartitionSpec
  new_parts = (axis_name,) if insert_axis else (
      None if spmd_axis_name is None else spmd_axis_name)
  unconstrained_dims = {ud + (d <= ud) for ud in unconstrained_dims}
  if new_parts is None:
    unconstrained_dims.add(d)
  y = sharding_constraint_p.bind(
      x,
      sharding=_pjit_batcher_for_sharding(
          sharding, d, new_parts, resource_env.physical_mesh, x.ndim),
      resource_env=resource_env,
      unconstrained_dims=unconstrained_dims)
  return y, d
batching.spmd_axis_primitive_batchers[sharding_constraint_p] = partial(
    _sharding_constraint_batcher, False)
batching.axis_primitive_batchers[sharding_constraint_p] = partial(
    _sharding_constraint_batcher, False, None)
pxla.spmd_primitive_batchers[sharding_constraint_p] = partial(
    _sharding_constraint_batcher, True, None)


def _resource_typing_sharding_constraint(avals, params, source_info,
                                         resource_env, named_axis_resources):
  aval, = avals
  if hasattr(params['sharding'], '_original_sharding'):
    parsed_pspec = params['sharding']._original_sharding._parsed_pspec
  else:
    parsed_pspec = parse_flatten_op_sharding(
        params['sharding']._hlo_sharding, resource_env.physical_mesh)[0]
  _check_resources_against_named_axes(
    "with_sharding_constraint input", aval, parsed_pspec, named_axis_resources)

pxla.custom_resource_typing_rules[sharding_constraint_p] = \
    _resource_typing_sharding_constraint

# -------------------- helpers --------------------

@lru_cache(maxsize=2048)
def to_gspmd_sharding(s: XLACompatibleSharding, ndim: int,
                      device_or_backend_set: bool = False) -> GSPMDSharding:
  if isinstance(s, GSPMDSharding):
    return s
  if xla_extension_version >= 234:
    gs = GSPMDSharding(s._device_assignment, s._to_xla_hlo_sharding(ndim),
                       memory_kind=s.memory_kind,
                       _device_list=getattr(s, '_internal_device_list', None))
  else:
    gs = GSPMDSharding(s._device_assignment, s._to_xla_hlo_sharding(ndim),
                       memory_kind=s.memory_kind)
  gs._original_sharding = s
  if device_or_backend_set:
    gs._original_sharding._device_backend = device_or_backend_set
  return gs


def get_unconstrained_dims(sharding: NamedSharding):
  assert sharding._parsed_pspec is not None
  return {i for i, axes in enumerate(sharding._parsed_pspec)
          if axes is None}


def _fast_path_get_device_assignment(
    shardings: Iterable[PjitSharding]) -> XLADeviceAssignment | None:
  da = None
  for i in shardings:
    if is_unspecified(i):
      continue
    if is_auto(i):
      return i.mesh._flat_devices_tuple  # type: ignore
    return i._device_assignment  # type: ignore
  return da


def _get_partition_spec(
    ppspec: Sequence[ParsedPartitionSpec]) -> Sequence[PartitionSpec]:
  return [get_single_pspec(p) for p in ppspec]


def get_op_sharding_from_executable(
    executable) -> tuple[Sequence[xc.OpSharding], Sequence[xc.OpSharding]]:
  in_op_shardings: list[xc.OpSharding] = []
  parameter_shardings_from_xla = executable.get_parameter_shardings()
  if parameter_shardings_from_xla is not None:
    in_op_shardings = parameter_shardings_from_xla

  out_op_shardings: list[xc.OpSharding] = []
  output_shardings_from_xla = executable.get_output_shardings()
  if output_shardings_from_xla is not None:
    out_op_shardings = output_shardings_from_xla

  return in_op_shardings, out_op_shardings


def _get_ppspec_from_executable(
    executable, mesh
  ) -> tuple[Sequence[ParsedPartitionSpec], Sequence[ParsedPartitionSpec]]:
  input_op_shardings, output_op_sharding = get_op_sharding_from_executable(
      executable
  )
  in_ppspec: list[ParsedPartitionSpec] = []
  for s in input_op_shardings:
    in_ppspec.extend(parse_flatten_op_sharding(s, mesh))

  out_ppspec: list[ParsedPartitionSpec] = []
  for s in output_op_sharding:
    out_ppspec.extend(parse_flatten_op_sharding(s, mesh))
  return in_ppspec, out_ppspec


def get_pspec_from_executable(
    executable, mesh: pxla.Mesh
) -> tuple[tuple[PartitionSpec, ...], tuple[PartitionSpec, ...]]:
  in_ppspec, out_ppspec = _get_ppspec_from_executable(executable, mesh)
  out_partition_spec = _get_partition_spec(out_ppspec)
  in_partition_spec = _get_partition_spec(in_ppspec)
  return tuple(in_partition_spec), tuple(out_partition_spec)
