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

import dataclasses
from enum import IntEnum
import inspect
import numpy as np
from collections import OrderedDict, Counter
from typing import (Callable, Sequence, Tuple, Union, cast, List, Optional,
                    Iterable, NamedTuple, Any)
import itertools as it
from functools import partial, lru_cache
import threading
import warnings

import jax
from jax._src import core
from jax import stages
from jax.errors import JAXTypeError
from jax._src.global_device_array import GlobalDeviceArray as GDA
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax._src.interpreters.pxla import PartitionSpec
from jax.tree_util import (
    tree_map, tree_flatten, tree_unflatten, treedef_is_leaf, tree_structure,
    treedef_tuple)

from jax._src.sharding import (
    NamedSharding, Sharding, XLACompatibleSharding, GSPMDSharding,
    XLADeviceAssignment, SingleDeviceSharding, PmapSharding)
from jax._src import array
from jax._src import dispatch
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src.api_util import (
    argnums_partial_except, flatten_axes, flatten_fun, flatten_fun_nokwargs,
    donation_vector, shaped_abstractify, check_callable,
    argnames_partial_except, resolve_argnums, FLAGS)
from jax._src.config import config
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import (prefix_errors, _generate_key_paths)
from jax._src.util import (
    HashableFunction, safe_map, safe_zip, wraps,
    distributed_debug_log, split_list, tuple_insert, weakref_lru_cache,
    merge_lists)

traceback_util.register_exclusion(__file__)

class _FromGdaSingleton:
  pass
FROM_GDA = _FromGdaSingleton()

def _is_from_gda(x):
  # It's occasionally possible to end up with two FROM_GDA singletons (e.g. if
  # pickling in_axis_resources and sending to other processes). Make sure this
  # doesn't cause an error to avoid user confusion.
  return isinstance(x, type(FROM_GDA))

_AUTOAxisResource = pxla.AUTOAxisResource
AUTO = pxla.AUTO
is_auto = pxla.is_auto

_UnspecifiedValue = pxla.UnspecifiedValue
_UNSPECIFIED = pxla._UNSPECIFIED
_is_unspecified = pxla._is_unspecified

def _is_unspecified_or_from_gda_or_auto(x):
  return _is_from_gda(x) or is_auto(x) or _is_unspecified(x)


PjitSharding = Union[GSPMDSharding, _UnspecifiedValue, _AUTOAxisResource]
PjitShardingMinusUnspecified = Union[GSPMDSharding, _AUTOAxisResource]
MeshSharding = Union[NamedSharding, _UnspecifiedValue, _AUTOAxisResource]
MeshShardingMinusUnspecified = Union[NamedSharding, _AUTOAxisResource]


def _check_all_or_none_unspecified(axis_resources, name):
  if not axis_resources:
    return False
  unspecified_count = 0
  unspecified = _is_unspecified(axis_resources[0])
  for resource in axis_resources:
    current_is_unspecified = _is_unspecified(resource)
    if current_is_unspecified:
      unspecified_count += 1
      assert unspecified_count == 1
    if current_is_unspecified != unspecified:
      raise ValueError(f'`pjit._UNSPECIFIED` exists in {name}. '
                       f'Make sure that every entry in {name} is '
                       '`pjit._UNSPECIFIED`.')
  return unspecified


def _try_infer_args(f, tree):
  dummy_args = tree_unflatten(tree, [False] * tree.num_leaves)
  try:
    return inspect.signature(f).bind(*dummy_args)
  except (TypeError, ValueError):
    return None


def _find_arg_mismatch(arg_list, fails, fun_name):
  first_err, second_err = fails
  mismatched_args_msg = []
  for name, inp_da, aval in arg_list:
    if first_err.m_type == pxla.MismatchType.ARG_SHARDING:
      if first_err.da == inp_da:
        mismatched_args_msg.append(
            (f"argument {name} of {fun_name} with shape {aval.str_short()} and "
             f"{first_err._dev_ids_plat_str}"))
        break

  for name, inp_da, aval in arg_list:
    if second_err.m_type == pxla.MismatchType.ARG_SHARDING:
      if second_err.da == inp_da:
        mismatched_args_msg.append(
            (f"argument {name} of {fun_name} with shape {aval.str_short()} and "
             f"{second_err._dev_ids_plat_str}"))
        break
  return mismatched_args_msg


def _device_assignment_mismatch_error(fun, fails, in_tree, args_flat, api_name):
  sig = _try_infer_args(fun, in_tree)
  args = tree_unflatten(in_tree, args_flat)
  args_aug = _generate_key_paths(args)

  arg_list = []
  for arg_key, val in args_aug:
    ak, *rem_keys = arg_key.keys
    if sig is not None:
      loc = ''.join(k.pprint() for k in rem_keys)
      arg_name = f'{list(sig.arguments.keys())[ak.key]}{loc}'
    else:
      arg_name = ''
    da = val.sharding._device_assignment if hasattr(val, 'sharding') else None
    arg_list.append((arg_name, da, shaped_abstractify(val)))

  fun_name = getattr(fun, '__qualname__', getattr(fun, '__name__', str(fun)))
  mismatched_args_msg = _find_arg_mismatch(arg_list, fails, fun_name)

  if len(mismatched_args_msg) == 2:
    first, second = mismatched_args_msg  # pylint: disable=unbalanced-tuple-unpacking
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


def _python_pjit_helper(fun, infer_params_fn, *args, **kwargs):
  args_flat, _, params, in_tree, out_tree, _ = infer_params_fn(
      *args, **kwargs)
  for arg in args_flat:
    dispatch.check_arg(arg)
  try:
    out_flat = pjit_p.bind(*args_flat, **params)
  except pxla.DeviceAssignmentMismatchError as e:
    fails, = e.args
    api_name = 'jit' if params['resource_env'] is None else 'pjit'
    msg = _device_assignment_mismatch_error(
        fun, fails, in_tree, args_flat, api_name)
    raise ValueError(msg) from None
  outs = tree_unflatten(out_tree, out_flat)
  return outs, out_flat, out_tree, args_flat


def _python_pjit(fun: Callable, infer_params_fn):

  @wraps(fun)
  @api_boundary
  def wrapped(*args, **kwargs):
    if config.jax_disable_jit:
      return fun(*args, **kwargs)
    return _python_pjit_helper(fun, infer_params_fn, *args, **kwargs)[0]

  def _python_pjit_evict_fn():
    _create_pjit_jaxpr.evict_function(fun)  # type: ignore
  wrapped.clear_cache = _python_pjit_evict_fn
  return wrapped

class _MostRecentPjitCallExecutable(threading.local):
  def __init__(self):
    self.value = None

_most_recent_pjit_call_executable = _MostRecentPjitCallExecutable()


def _read_most_recent_pjit_call_executable():
  executable = _most_recent_pjit_call_executable.value
  _most_recent_pjit_call_executable.value = None
  return executable


def _cpp_pjit_evict_fn(self):
  self._clear_cache()
  _create_pjit_jaxpr.evict_function(self._fun)  # type: ignore


if xla_extension_version >= 124:
  _cpp_pjit_cache = xc._xla.PjitFunctionCache()


def _cpp_pjit(fun: Callable, infer_params_fn, static_argnums, static_argnames,
              donate_argnums, pjit_has_explicit_sharding):

  @api_boundary
  def cache_miss(*args, **kwargs):
    outs, out_flat, out_tree, args_flat = _python_pjit_helper(
        fun, infer_params_fn, *args, **kwargs)

    executable = _read_most_recent_pjit_call_executable()

    use_fastpath = (
        executable is not None and
        isinstance(executable, pxla.MeshExecutable) and
        isinstance(executable.unsafe_call, pxla.ExecuteReplicated) and
        # No effects in computation
        not executable.unsafe_call.ordered_effects and
        not executable.unsafe_call.has_unordered_effects and
        not executable.unsafe_call.has_host_callbacks and
        all(isinstance(x, xc.ArrayImpl) for x in out_flat)
    )

    if use_fastpath:
      out_avals = [o.aval for o in out_flat]
      out_committed = [o._committed for o in out_flat]
      kept_var_bitvec = [i in executable._kept_var_idx
                         for i in range(len(args_flat))]
      fastpath_data = pxla.MeshExecutableFastpathData(
          executable.xla_executable, out_tree, executable._in_shardings,
          executable._out_shardings, out_avals, out_committed, kept_var_bitvec)
    else:
      fastpath_data = None

    return outs, fastpath_data

  if xla_extension_version < 124:
    cpp_pjit_f = xc._xla.pjit(  # type: ignore
        getattr(fun, "__name__", "<unnamed function>"),  # type: ignore
        fun, cache_miss, static_argnums, static_argnames)  # type: ignore
  else:
    if pjit_has_explicit_sharding:
      global_cache = xc._xla.PjitFunctionCache()
    else:
      global_cache = _cpp_pjit_cache
    cpp_pjit_f = xc._xla.pjit(  # type: ignore
        getattr(fun, "__name__", "<unnamed function>"),  # type: ignore
        fun, cache_miss, static_argnums, static_argnames,  # type: ignore
        donate_argnums, global_cache)  # type: ignore

  cpp_pjitted_f = wraps(fun)(cpp_pjit_f)
  cpp_pjitted_f._fun = fun
  type(cpp_pjitted_f).clear_cache = _cpp_pjit_evict_fn
  return cpp_pjitted_f


def _resolve_axis_resources_and_shardings_arg(
    in_shardings, out_shardings, in_axis_resources, out_axis_resources):
  if not _is_unspecified(in_shardings) and not _is_unspecified(in_axis_resources):
    raise ValueError(
        'Setting both in_shardings and in_axis_resources is not '
        'allowed. in_axis_resources is deprecated. Please use in_shardings.')
  if not _is_unspecified(out_shardings) and not _is_unspecified(out_axis_resources):
    raise ValueError(
        'Setting both out_shardings and out_axis_resources is not '
        'allowed. out_axis_resources is deprecated. Please use out_shardings.')

  if not _is_unspecified(in_axis_resources):
    final_in_shardings = in_axis_resources
  else:
    final_in_shardings = in_shardings

  if not _is_unspecified(out_axis_resources):
    final_out_shardings = out_axis_resources
  else:
    final_out_shardings = out_shardings
  return final_in_shardings, final_out_shardings


def pre_infer_params(fun, in_shardings, out_shardings,
                     donate_argnums, static_argnums, static_argnames, device,
                     backend, abstracted_axes):
  # TODO(yashkatariya, mattjj): Remove when pjit supports dynamic shapes.
  if jax.config.jax_dynamic_shapes:
    raise ValueError("Dynamic shapes is not supported with pjit yet.")
  if abstracted_axes and not jax.config.jax_dynamic_shapes:
    raise ValueError("abstracted_axes must be used with --jax_dynamic_shapes")

  check_callable(fun)

  if (not jax.config.jax_array and
      (_is_unspecified(in_shardings) or _is_unspecified(out_shardings))):
    raise ValueError(
        "in_shardings and out_shardings should not "
        "be the unspecified singleton value. Please enable `jax.Array` to use "
        "this feature. You can use jax.config.update('jax_array', True) or "
        "set the environment variable  JAX_ARRAY=1 , or set the `jax_array` "
        "boolean flag to something true-like.")

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
    if not _is_unspecified(in_shardings):
      raise ValueError('If backend or device is specified on jit, then '
                       'in_shardings should not be specified.')
    if not _is_unspecified(out_shardings):
      raise ValueError('If backend or device is specified on jit, then '
                       'out_shardings should not be specified.')

  if isinstance(in_shardings, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/google/jax/issues/2367
    in_shardings = tuple(in_shardings)

  in_shardings, _, _ = _prepare_axis_resources(in_shardings, 'in_shardings')
  out_shardings, _, _ = _prepare_axis_resources(out_shardings, 'out_shardings')

  donate_argnums, static_argnums, static_argnames = resolve_argnums(
      fun, donate_argnums, static_argnums, static_argnames)

  return (in_shardings, out_shardings, donate_argnums, static_argnums,
          static_argnames)


def post_infer_params(fun, infer_params_fn, static_argnums, static_argnames,
                      donate_argnums, abstracted_axes,
                      pjit_has_explicit_sharding):
  if FLAGS.experimental_cpp_pjit and abstracted_axes is None:
    wrapped = _cpp_pjit(fun, infer_params_fn, static_argnums, static_argnames,
                        donate_argnums, pjit_has_explicit_sharding)
  else:
    wrapped = _python_pjit(fun, infer_params_fn)

  @api_boundary
  def lower(*args, **kwargs):
    (args_flat, flat_local_in_avals, params, in_tree, out_tree,
     donate_argnums) = infer_params_fn(*args, **kwargs)
    if jax.config.jax_array:
      resource_env = params['resource_env']
      mesh = None if resource_env is None else resource_env.physical_mesh
      in_shardings = _resolve_in_shardings(
          args_flat, params['in_shardings'], params['out_shardings'], mesh)
    else:
      in_shardings = params['in_shardings']
    in_is_global = _calc_is_global_sequence(
        params['in_positional_semantics'], in_shardings)
    lowering = _pjit_lower(
        params['jaxpr'], in_shardings, params['out_shardings'],
        params['resource_env'], params['donated_invars'], params['name'],
        in_is_global, params['keep_unused'], always_lower=True)

    if kwargs:
      args_kwargs_in_tree = in_tree
    else:
      args_kwargs_in_tree = treedef_tuple([in_tree, tree_flatten({})[1]])

    return stages.Lowered.from_flat_info(
        lowering, args_kwargs_in_tree, flat_local_in_avals, donate_argnums,
        out_tree)

  wrapped.lower = lower
  return wrapped


def _pjit_explicit_sharding(in_shardings, out_shardings, device,
                            backend) -> bool:
  in_shardings_flat, _ = tree_flatten(in_shardings)
  out_shardings_flat, _ = tree_flatten(out_shardings)
  return (device is not None or
          backend is not None or
          any(not _is_unspecified(i) for i in in_shardings_flat) or
          any(not _is_unspecified(i) for i in out_shardings_flat))


class PjitInfo(NamedTuple):
  fun: Callable
  in_shardings: Any
  out_shardings: Any
  static_argnums: Tuple[int, ...]
  static_argnames: Tuple[str, ...]
  donate_argnums: Tuple[int, ...]
  device: Optional[xc.Device]
  backend: Optional[str]
  keep_unused: bool
  inline: bool
  resource_env: Any


def common_infer_params(pjit_info_args, *args, **kwargs):
  (fun, user_in_shardings, user_out_shardings, static_argnums, static_argnames,
   donate_argnums, device, backend, keep_unused, inline,
   resource_env) = pjit_info_args

  if kwargs and not _is_unspecified(user_in_shardings):
    raise ValueError(
        "pjit does not support kwargs when in_shardings is specified.")

  if resource_env is not None:
    pjit_mesh = resource_env.physical_mesh
    if pjit_mesh.empty:
      if jax.config.jax_array:
        # Don't enforce requiring a mesh when `jax_array` flag is enabled. But
        # if mesh is not empty then pjit will respect it.
        pass
      else:
        raise RuntimeError("pjit requires a non-empty mesh! Are you sure that "
                            "it's defined at the call site?")
  else:
    pjit_mesh = None

  if (backend or device) and pjit_mesh is not None and not pjit_mesh.empty:
    raise ValueError(
        "Mesh context manager should not be used with jit when backend or "
        "device is also specified as an argument to jit.")

  f = lu.wrap_init(fun)
  f, dyn_args = argnums_partial_except(f, static_argnums, args,
                                        allow_invalid=True)
  del args

  # TODO(yashkatariya): Merge the nokwargs and kwargs path. One blocker is
  # flatten_axes which if kwargs are present in the treedef (even empty {}),
  # leads to wrong expansion.
  if kwargs:
    f, dyn_kwargs = argnames_partial_except(f, static_argnames, kwargs)
    args_flat, in_tree = tree_flatten((dyn_args, dyn_kwargs))
    flat_fun, out_tree = flatten_fun(f, in_tree)
  else:
    args_flat, in_tree = tree_flatten(dyn_args)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    dyn_kwargs = ()
  del kwargs

  if donate_argnums and not jax.config.jax_debug_nans:
    donated_invars = donation_vector(donate_argnums, dyn_args, dyn_kwargs)
  else:
    donated_invars = (False,) * len(args_flat)

  if jax.config.jax_array:
    # If backend or device is set as an arg on jit, then resolve them to
    # in_shardings and out_shardings as if user passed in in_shardings
    # and out_shardings.
    if backend or device:
      in_shardings = out_shardings = _create_sharding_with_device_backend(
          device, backend)
    else:
      in_shardings = tree_map(
          lambda x: _create_sharding_for_array(pjit_mesh, x), user_in_shardings)
      out_shardings = tree_map(
          lambda x: _create_sharding_for_array(pjit_mesh, x), user_out_shardings)
  else:
    in_shardings = tree_map(
        lambda x: _create_mesh_pspec_sharding_from_parsed_pspec(pjit_mesh, x),
        user_in_shardings)
    out_shardings = tree_map(
        lambda x: x if _is_unspecified(x) else
        _create_mesh_pspec_sharding_from_parsed_pspec(pjit_mesh, x), user_out_shardings)
    # This check fails extremely rarely and has a huge cost in the dispatch
    # path. So hide it behind the jax_enable_checks flag.
    if jax.config.jax_enable_checks:
      _maybe_check_pjit_gda_mesh(args_flat, pjit_mesh)

  del user_in_shardings, user_out_shardings

  local_in_avals = tuple(shaped_abstractify(a) for a in args_flat)
  # TODO(yashkatariya): This is a hack. This should go away when avals have
  # is_global attribute.
  if jax.config.jax_array:
    in_positional_semantics = (pxla._PositionalSemantics.GLOBAL,) * len(args_flat)
  else:
    in_positional_semantics = tuple(tree_map(_get_in_positional_semantics, args_flat))
  out_positional_semantics = (
      pxla._PositionalSemantics.GLOBAL
      if jax.config.jax_parallel_functions_output_gda or jax.config.jax_array else
      pxla.positional_semantics.val)

  global_in_avals, canonicalized_in_shardings_flat = _process_in_axis_resources(
      hashable_pytree(in_shardings), local_in_avals, in_tree, in_positional_semantics,
      tuple(isinstance(a, GDA) for a in args_flat), resource_env)

  jaxpr, consts, canonicalized_out_shardings_flat = _pjit_jaxpr(
      flat_fun, hashable_pytree(out_shardings), global_in_avals,
      HashableFunction(out_tree, closure=()),
      ('jit' if resource_env is None else 'pjit'))

  if (any(_is_from_gda(i) for i in canonicalized_in_shardings_flat) or
      not jax.config.jax_array):
    canonicalized_in_shardings_flat = _maybe_replace_from_gda_with_pspec(
        canonicalized_in_shardings_flat, args_flat)

  assert len(args_flat) == len(canonicalized_in_shardings_flat)

  canonicalized_in_shardings_flat = (
      _UNSPECIFIED,) * len(consts) + canonicalized_in_shardings_flat
  donated_invars = (False,) * len(consts) + donated_invars
  in_positional_semantics = (
      pxla._PositionalSemantics.GLOBAL,) * len(consts) + in_positional_semantics

  # in_shardings and out_shardings here are all GSPMDSharding.
  params = dict(
      jaxpr=jaxpr,
      in_shardings=canonicalized_in_shardings_flat,
      out_shardings=canonicalized_out_shardings_flat,
      resource_env=resource_env,
      donated_invars=donated_invars,
      name=getattr(flat_fun, '__name__', '<unnamed function>'),
      in_positional_semantics=in_positional_semantics,
      out_positional_semantics=out_positional_semantics,
      keep_unused=keep_unused,
      inline=inline,
  )
  return (consts + args_flat, local_in_avals, params, in_tree, out_tree(),
          donate_argnums)


# in_shardings and out_shardings can't be None as the default value
# because `None` means that the input is fully replicated.
def pjit(
    fun: Callable,
    in_shardings=_UNSPECIFIED,
    out_shardings=_UNSPECIFIED,
    in_axis_resources=_UNSPECIFIED,
    out_axis_resources=_UNSPECIFIED,
    static_argnums: Union[int, Sequence[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    donate_argnums: Union[int, Sequence[int]] = (),
    keep_unused: bool = False,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    inline: bool = False,
    abstracted_axes: Optional[Any] = None,
) -> stages.Wrapped:
  """Makes ``fun`` compiled and automatically partitioned across multiple devices.

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

      The valid resource assignment specifications are:
        - :py:obj:`None`, in which case the value will be replicated on all devices
        - :py:class:`XLACompatibleSharding`, which will decide how the value
          will be partitioned. With this, using a mesh context manager is not
          required.
        - :py:class:`PartitionSpec`, a tuple of length at most equal to the rank
          of the partitioned value. Each element can be a :py:obj:`None`, a mesh
          axis or a tuple of mesh axes, and specifies the set of resources assigned
          to partition the value's dimension matching its position in the spec.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it.
    out_shardings: Like ``in_shardings``, but specifies resource
      assignment for function outputs.
    in_axis_resources: (Deprecated) Please use in_shardings.
    out_axis_resources: (Deprecated) Please use out_shardings.
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
    donate_argnums: Specify which argument buffers are "donated" to the computation.
      It is safe to donate argument buffers if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to.
      For more details on buffer donation see the `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
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
  in_shardings, out_shardings = _resolve_axis_resources_and_shardings_arg(
      in_shardings, out_shardings, in_axis_resources, out_axis_resources)

  (in_shardings, out_shardings, donate_argnums, static_argnums,
   static_argnames) = pre_infer_params(
       fun, in_shardings, out_shardings, donate_argnums,
       static_argnums, static_argnames, device, backend, abstracted_axes)

  def infer_params(*args, **kwargs):
    # Putting this outside of wrapped would make resources lexically scoped
    resource_env = pxla.thread_resources.env
    pjit_info_args = PjitInfo(
          fun=fun, in_shardings=in_shardings,
          out_shardings=out_shardings, static_argnums=static_argnums,
          static_argnames=static_argnames, donate_argnums=donate_argnums,
          device=device, backend=backend, keep_unused=keep_unused,
          inline=inline, resource_env=resource_env)
    return common_infer_params(pjit_info_args, *args, **kwargs)

  has_explicit_sharding = _pjit_explicit_sharding(
      in_shardings, out_shardings, device, backend)
  return post_infer_params(fun, infer_params, static_argnums, static_argnames,
                           donate_argnums, abstracted_axes,
                           has_explicit_sharding)


def hashable_pytree(pytree):
  vals, treedef = tree_flatten(pytree)
  vals = tuple(vals)
  return HashableFunction(lambda: tree_unflatten(treedef, vals),
                          closure=(treedef, vals))


@lru_cache(maxsize=4096)
def _create_mesh_pspec_sharding_from_parsed_pspec(mesh, x):
  if _is_unspecified_or_from_gda_or_auto(x):
    return x
  return pxla.create_mesh_pspec_sharding(mesh, x.user_spec, x)


def _create_sharding_for_array(mesh, x):
  # TODO(yashkatariya): Only check for auto and unspecified here after
  # FROM_GDA is removed.
  if isinstance(x, XLACompatibleSharding) or _is_unspecified_or_from_gda_or_auto(x):
    return x
  if mesh is None:
    raise RuntimeError(
        "jit does not support using the mesh context manager. Please pass in "
        "the sharding explicitly via in_shardings or out_shardings.")
  if mesh.empty:
    raise RuntimeError("pjit requires a non-empty mesh! Is a mesh defined at "
                       "the call site? Alternatively, provide a "
                       "XLACompatibleSharding to pjit and then the "
                       "mesh context manager is not required.")
  # A nice user error is raised in _prepare_axis_resources.
  assert isinstance(x, ParsedPartitionSpec), x
  return _create_mesh_pspec_sharding_from_parsed_pspec(mesh, x)


def _create_sharding_with_device_backend(device, backend):
  if device is not None:
    assert backend is None
    out = SingleDeviceSharding(device)
  elif backend is not None:
    assert device is None
    out = SingleDeviceSharding(
        xb.get_backend(backend).get_default_device_assignment(1)[0])
  out._device_backend = True
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

  if config.jax_array:
    axis_tree = shardings
  else:
    # Replace axis_resources with unparsed versions to avoid revealing internal details
    axis_tree = tree_map(lambda parsed: parsed.spec, shardings)

  # Because ecause we only have the `tree` treedef and not the full pytree here,
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
def _process_in_axis_resources(in_shardings_thunk, local_in_avals,
                               in_tree, in_positional_semantics, is_gda,
                               resource_env):
  orig_in_shardings = in_shardings_thunk()
  # Only do this if original in_shardings are unspecified. If they are
  # FROM_GDA or AUTO, go via flatten_axis_resources.
  if _is_unspecified(orig_in_shardings):
    in_shardings_flat = (orig_in_shardings,) * len(local_in_avals)
  else:
    in_shardings_flat = flatten_axis_resources(
          "pjit in_shardings", in_tree, orig_in_shardings,
          tupled_args=True)

  # Fork here because the `Array` path is very simple and doesn't need all the
  # complexity below.
  if config.jax_array:
    pjit_check_aval_sharding(in_shardings_flat, local_in_avals, "pjit arguments",
                             allow_uneven_sharding=False)
    global_in_avals = local_in_avals
    # TODO(yashkatariya): Only check for is_auto or _is_unspecified when
    # FROM_GDA is removed.
    canonicalized_shardings = tuple(
        i if _is_unspecified_or_from_gda_or_auto(i) else to_gspmd_sharding(i, aval.ndim)
        for i, aval in safe_zip(in_shardings_flat, global_in_avals))
    return tuple(global_in_avals), canonicalized_shardings

  if not local_in_avals:
    assert not in_shardings_flat
    return (), ()

  in_axis_resources_flat = tuple(
      i if _is_from_gda(i) or is_auto(i) else i._parsed_pspec
      for i in in_shardings_flat)

  # This check should be above local_to_global call below otherwise if
  # `FROM_GDA` is passed to any input other than GDA, a ugly error message
  # will be raised because get_array_mapping (in local_to_global) of a
  # FROM_GDA cannot happen.
  tree_map(_check_resources_mismatch, in_axis_resources_flat, is_gda)
  # If all inputs have global semantics or fully replicated, then the avals are
  # global and the mesh should also be global. This split is because
  # non-contiguous mesh can only be used if all inputs have global semantics or
  # fully replicated.
  # Use canonicalized in_axis_resources here because we want to treat P(None)
  # and None (for example) as equivalent.
  if all(
      (not _is_from_gda(p) and not is_auto(p) and
       CanonicalizedParsedPartitionSpec(p).partitions == ()) or
      ips == pxla._PositionalSemantics.GLOBAL
      for p, ips in safe_zip(in_axis_resources_flat, in_positional_semantics)):
    # Shapes should be checked against non canonicalized in_axis_resources.
    # For example, partitions of () and ((),) are not equivalent, since the
    # first one is a valid spec for a scalar value, while the second is not!
    pjit_check_aval_sharding(in_shardings_flat, local_in_avals, "pjit arguments",
                             allow_uneven_sharding=False)
  else:
    pjit_check_aval_sharding(
        [i if _is_from_gda(i) or is_auto(i) else
         NamedSharding(i.mesh.local_mesh, i.spec)
         for i in in_shardings_flat],
        local_in_avals, "pjit arguments", allow_uneven_sharding=False)

  # Local or global avals doesn't matter for converting to op sharding because
  # the `ndim` does not change.
  canonicalized_in_shardings_flat = tuple(
      i if _is_from_gda(i) or is_auto(i) else to_gspmd_sharding(i, aval.ndim)
      for i, aval in safe_zip(in_shardings_flat, local_in_avals))

  global_in_avals = local_to_global(
      in_positional_semantics, local_in_avals, canonicalized_in_shardings_flat,
      resource_env.physical_mesh)

  return tuple(global_in_avals), canonicalized_in_shardings_flat


@lu.cache
def _create_pjit_jaxpr(fun, global_in_avals, api_name):
  prev_positional_val = pxla.positional_semantics.val
  try:
    pxla.positional_semantics.val = pxla._PositionalSemantics.GLOBAL
    with dispatch.log_elapsed_time(f"Finished tracing + transforming {fun.__name__} "
                                   "for pjit in {elapsed_time} sec",
                                    event=dispatch.JAXPR_TRACE_EVENT):
      jaxpr, global_out_avals, consts = pe.trace_to_jaxpr_dynamic(
          fun, global_in_avals, debug_info=pe.debug_info_final(fun, api_name))
  finally:
    pxla.positional_semantics.val = prev_positional_val

  if any(isinstance(c, core.Tracer) for c in consts):
    jaxpr = pe.convert_constvars_jaxpr(jaxpr)
    jaxpr = pe.close_jaxpr(jaxpr)
    final_consts = consts
  else:
    jaxpr = core.ClosedJaxpr(jaxpr, consts)
    final_consts = []
  return jaxpr, final_consts, global_out_avals


@lru_cache(maxsize=4096)
def _check_and_canonicalize_out_shardings(
    out_shardings_thunk, out_tree, global_out_avals):
  orig_out_shardings = out_shardings_thunk()
  # TODO(yashkatariya): Remove the if branch and fix flatten_axis_resources
  # instead. This condition exists because flatten_axis_resources passes in an
  # `object()` while unflattening which breaks assertion is user defined
  # pytrees (which shouldn't exist but they do).
  if (_is_unspecified(orig_out_shardings) or
      isinstance(orig_out_shardings, XLACompatibleSharding)):
    out_shardings_flat = (orig_out_shardings,) * len(global_out_avals)
  else:
    out_shardings_flat = flatten_axis_resources(
        "pjit out_shardings", out_tree(), orig_out_shardings,
        tupled_args=False)

  pjit_check_aval_sharding(out_shardings_flat, global_out_avals, "pjit outputs",
                           allow_uneven_sharding=False)

  canonicalized_out_shardings_flat = tuple(
      o if _is_unspecified(o) or is_auto(o) else to_gspmd_sharding(o, aval.ndim)
      for o, aval in safe_zip(out_shardings_flat, global_out_avals)
  )
  return canonicalized_out_shardings_flat


def _pjit_jaxpr(fun, out_shardings_thunk, global_in_avals, out_tree, api_name):
  jaxpr, final_consts, global_out_avals = _create_pjit_jaxpr(
      fun, global_in_avals, api_name)
  canonicalized_out_shardings_flat = _check_and_canonicalize_out_shardings(
      out_shardings_thunk, out_tree, tuple(global_out_avals))
  # lu.cache needs to be able to create weakrefs to outputs, so we can't return a plain tuple
  return jaxpr, final_consts, canonicalized_out_shardings_flat


def pjit_check_aval_sharding(
    shardings, flat_avals, what_aval: str, allow_uneven_sharding: bool):
  for aval, s in zip(flat_avals, shardings):
    if _is_unspecified_or_from_gda_or_auto(s):
      continue
    global_str = "" if s.is_fully_addressable else " global"
    shape = aval.shape
    try:
      # Sharding interfaces can implement `is_compatible_aval` as an optional
      # method to raise a more meaningful error.
      if hasattr(s, 'is_compatible_aval'):
        s.is_compatible_aval(shape)
      else:
        s._to_xla_op_sharding(len(shape))
    except ValueError as e:
      raise ValueError(f'One of {what_aval} is incompatible with its sharding '
                       f'annotation {s}: {str(e)}')
    # Use the `OpSharding` proto to find out how many ways each dimension of
    # the aval is sharded. This approach will work across all
    # XLACompatibleSharding.
    op_sharding = s._to_xla_op_sharding(len(shape))
    assert op_sharding is not None
    num_ways_dim_sharded, _ = pxla.get_num_ways_dim_sharded(
        cast(xc.OpSharding, op_sharding))
    for i, size in enumerate(num_ways_dim_sharded):
      if not allow_uneven_sharding and shape[i] % size != 0:
        raise ValueError(f"One of {what_aval} was given the sharding "
                         f"of {s}, which implies that "
                         f"the{global_str} size of its dimension {i} should be "
                         f"divisible by {size}, but it is equal to {shape[i]} "
                         f"(full shape: {shape}) ")


class SpecSync(IntEnum):
  """Encodes how much out of sync the real value of partitions is compared to the user specified one.

  We use this to make sure we don't show garbage modified values while claiming
  that the users have specified them like that.
  """
  OUT_OF_SYNC = 0  # Arbitrary changes, including new axes inserted
  DIM_PERMUTE = 1  # Dimensions permuted, but no new sharding axes
  IN_SYNC = 2  # Entirely in sync

class ParsedPartitionSpec:
  __slots__ = ('unsafe_user_spec', 'partitions', 'sync')

  def __init__(self, user_spec, partitions, sync=SpecSync.IN_SYNC):
    self.unsafe_user_spec = user_spec
    # None in partitions represents unconstrained dim.
    # TODO(yashkatariya): May use a sentinel value.
    self.partitions = tuple(partitions)
    self.sync = sync

  @property
  def user_spec(self):
    return self.unsynced_user_spec(SpecSync.IN_SYNC)

  def get_partition_spec(self) -> PartitionSpec:
    if self.sync < SpecSync.IN_SYNC:
      return _get_single_pspec(self)
    else:
      if isinstance(self.unsafe_user_spec, PartitionSpec):
        return self.unsafe_user_spec
      else:
        return _get_single_pspec(self)

  def unsynced_user_spec(self, min_sync):
    if self.sync < min_sync:
      raise AssertionError(f"Please open a bug report! ({self.sync} >= {min_sync})")
    return self.unsafe_user_spec

  def insert_axis_partitions(self, dim, val):
    parts = self.partitions
    too_short = dim - len(parts)
    if too_short > 0:
      parts += ((),) * too_short
    new_partitions = tuple_insert(parts, dim, val)
    new_sync = SpecSync.DIM_PERMUTE if (val == () or val is None) else SpecSync.OUT_OF_SYNC
    return ParsedPartitionSpec(self.unsafe_user_spec, new_partitions, sync=new_sync)

  @classmethod
  def from_user_input(cls, entry, arg_name, allow_unconstrained_dims=False):
    if entry is None:
      return cls(entry, ())
    if not isinstance(entry, PartitionSpec):
      raise TypeError(f"{arg_name} are expected to be "
                      f"PartitionSpec instances or None, but got {entry}")
    axis_specs = []
    for axis_spec in entry:
      if axis_spec is None:
        axis_spec = ()
      elif isinstance(axis_spec, (list, tuple)):
        axis_spec = tuple(axis_spec)
      elif axis_spec == PartitionSpec.UNCONSTRAINED:
        if not allow_unconstrained_dims:
          raise ValueError(f"Unconstrained dims are not allowed: {entry}")
        axis_spec = None
      else:
        axis_spec = (axis_spec,)
      axis_specs.append(axis_spec)
    return cls(entry, axis_specs)

  def __hash__(self):
    return hash((self.partitions, self.sync))

  def __eq__(self, other):
    return (self.partitions == other.partitions and
            self.sync == other.sync)

  def __len__(self):
    return len(self.partitions)

  def __getitem__(self, i):
    return self.partitions[i]

  def __iter__(self):
    return iter(self.partitions)

  def __repr__(self):
    return (f"ParsedPartitionSpec(partitions={self.partitions}, "
            f"unsafe_user_spec={self.unsafe_user_spec}, "
            f"sync={self.sync})")

class CanonicalizedParsedPartitionSpec(ParsedPartitionSpec):
  """ParsedPartitionSpecs that are canonicalized.

  ParsedPartitionSpecs may contain trailing empty tuples, that make them
  semantically different in general, and yet in some situations we prefer
  to regard them as equivalent. For example, partitions of () and ((),)
  cannot be always considered equivalent, since the first one is a valid
  spec for a scalar value, while the second is not! However, when either of
  those are applied to a 2D array, they both mean that the array is fully
  replicated.

  So CanonicalizedParsedPartitionSpecs removes the trailing empty tuples from
  partitions.
  """

  def __init__(self, parsed_pspec: ParsedPartitionSpec):
    partitions = list(parsed_pspec.partitions)
    while partitions and partitions[-1] == ():
      partitions.pop()

    super().__init__(parsed_pspec.unsafe_user_spec, partitions,
                     parsed_pspec.sync)

  def __repr__(self):
    return (f"CanonicalizedParsedPartitionSpec(partitions={self.partitions}, "
            f"unsafe_user_spec={self.unsafe_user_spec}, "
            f"sync={self.sync})")


def _prepare_axis_resources(axis_resources,
                            arg_name,
                            allow_unconstrained_dims=False):
  # PyTrees don't treat None values as leaves, so we use an is_leaf function.
  entries, treedef = tree_flatten(axis_resources, is_leaf=lambda x: x is None)
  what = f"{arg_name} leaf specifications"
  # All entries should be specified or if unspecified then there should only
  # be 1 entry for that since _UNSPECIFIED is a private API.
  _check_all_or_none_unspecified(entries, arg_name)

  new_entries = []
  for entry in entries:
    if _is_unspecified_or_from_gda_or_auto(entry):
      new_entries.append(entry)
    elif isinstance(entry, Sharding):
      if isinstance(entry, PmapSharding):
        raise ValueError(f'One of {what} got sharding {entry} which is not '
                         'allowed.')
      if not isinstance(entry, XLACompatibleSharding):
        raise ValueError(f'One of {what} got sharding {entry} which is not a '
                         'subclass of XLACompatibleSharding.')
      new_entries.append(entry)
    else:
      new_entries.append(ParsedPartitionSpec.from_user_input(
          entry, what, allow_unconstrained_dims=allow_unconstrained_dims))

  _check_unique_resources(new_entries, arg_name)
  return tree_unflatten(treedef, new_entries), new_entries, treedef


def _check_resources_mismatch(in_axis_resources_flat, is_gda):
  if not is_gda and _is_from_gda(in_axis_resources_flat):
    raise ValueError('For a non-GDA input, the corresponding resource in '
                     'in_axis_resources cannot be `pjit.FROM_GDA`.')

def _check_unique_resources(axis_resources, arg_name):
  for arg_axis_resources in axis_resources:
    if not arg_axis_resources: continue
    if (_is_unspecified_or_from_gda_or_auto(arg_axis_resources) or
        isinstance(arg_axis_resources, XLACompatibleSharding)):
      continue
    constrained_dims = [d for d in arg_axis_resources if d is not None]
    resource_counts = Counter(it.chain.from_iterable(constrained_dims))
    if not resource_counts: continue
    if resource_counts.most_common(1)[0][1] > 1:
      multiple_uses = [r for r, c in resource_counts.items() if c > 1]
      if multiple_uses:
        raise ValueError(f"A single {arg_name} specification can map every mesh axis "
                         f"to at most one positional dimension, but {arg_axis_resources.user_spec} "
                         f"has duplicate entries for {pxla.show_axes(multiple_uses)}")

# -------------------- pjit rules --------------------

pjit_p = core.AxisPrimitive("pjit")
pjit_p.multiple_results = True


def _resolve_in_shardings(
    args, pjit_in_shardings: Sequence[PjitSharding],
    out_shardings: Sequence[PjitSharding],
    pjit_mesh: Optional[pxla.Mesh]) -> Sequence[PjitSharding]:
  # If True, means that device or backend is set by the user on pjit and it
  # has the same semantics as device_put i.e. doesn't matter which device the
  # arg is on, reshard it to the device mentioned. So don't do any of the
  # checks and just return the pjit_in_shardings directly. `shard_args` will
  # handle the resharding.
  if pxla.check_device_backend_on_shardings(pjit_in_shardings):
    return pjit_in_shardings

  committed_arg_shardings = []
  for a in args:
    if hasattr(a, 'sharding'):
      arg_s = a.sharding
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
  pxla._get_and_check_device_assignment(
      it.chain(
          committed_arg_shardings,
          [(i, pxla.MismatchType.IN_SHARDING, None) for i in pjit_in_shardings],
          [(o, pxla.MismatchType.OUT_SHARDING, None) for o in out_shardings]),
      (None if pjit_mesh is None or pjit_mesh.empty else list(pjit_mesh.devices.flat)))

  resolved_in_shardings = []
  for arg, pjit_in_s in safe_zip(args, pjit_in_shardings):
    arg_s, committed = ((arg.sharding, getattr(arg, '_committed', True))
                        if hasattr(arg, 'sharding') else (_UNSPECIFIED, False))
    if _is_unspecified(pjit_in_s):
      if _is_unspecified(arg_s):
        resolved_in_shardings.append(arg_s)
      else:
        if committed:
          # If the arg has a PmapSharding, then reshard it unconditionally.
          if isinstance(arg_s, PmapSharding):
            resolved_in_shardings.append(_UNSPECIFIED)
          else:
            resolved_in_shardings.append(to_gspmd_sharding(
                cast(XLACompatibleSharding, arg_s), arg.ndim))
        else:
          if dispatch.is_single_device_sharding(arg_s):
            resolved_in_shardings.append(_UNSPECIFIED)
          else:
            raise NotImplementedError('Having uncommitted Array sharded on '
                                      'multiple devices is not supported.')
    else:
      if isinstance(arg, np.ndarray) and not pxla.is_op_sharding_replicated(
          pjit_in_s._to_xla_op_sharding(arg.ndim)) and xb.process_count() > 1:  # type: ignore
        raise ValueError(
            'When jax.Array is enabled, passing non-trivial shardings for numpy '
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
      if not _is_unspecified(arg_s):
        if (committed and
            not isinstance(arg_s, PmapSharding) and
            not pxla.are_op_shardings_equal(
                pjit_in_s._to_xla_op_sharding(arg.ndim),  # type: ignore
                arg_s._to_xla_op_sharding(arg.ndim))):
          op =  getattr(pjit_in_s, '_original_sharding', pjit_in_s)
          raise ValueError('Sharding passed to pjit does not match the sharding '
                           'on the respective arg. '
                           f'Got pjit sharding: {op},\n'
                           f'arg sharding: {arg_s} for arg shape: {arg.shape}, '
                           f'arg value: {arg}')
      resolved_in_shardings.append(pjit_in_s)

  return tuple(resolved_in_shardings)


def _pjit_call_impl(*args, jaxpr,
                    in_shardings, out_shardings, resource_env,
                    donated_invars, name,
                    in_positional_semantics, out_positional_semantics,
                    keep_unused, inline):

  global _most_recent_pjit_call_executable

  if config.jax_array:
    in_shardings = _resolve_in_shardings(
        args, in_shardings, out_shardings,
        resource_env.physical_mesh if resource_env is not None else None)

  in_is_global = _calc_is_global_sequence(in_positional_semantics, in_shardings)
  if config.jax_array:
    _allow_propagation_to_outputs = [_is_unspecified(o) for o in out_shardings]
  else:
    _allow_propagation_to_outputs = [False] * len(out_shardings)
  compiled = _pjit_lower(
      jaxpr, in_shardings, out_shardings, resource_env,
      donated_invars, name, in_is_global, keep_unused,
      always_lower=False).compile(
          _allow_propagation_to_outputs=_allow_propagation_to_outputs)
  _most_recent_pjit_call_executable.value = compiled
  # This check is expensive so only do it if enable_checks is on.
  if compiled._auto_spmd_lowering and config.jax_enable_checks:
    pxla.check_gda_or_array_xla_sharding_match(args, compiled._in_shardings)
  if config.jax_distributed_debug:
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
                          ("abstract args", list(map(xla.abstractify, args))),
                          ("fingerprint", fingerprint))
  try:
    return compiled.unsafe_call(*args)
  except FloatingPointError:
    assert config.jax_debug_nans or config.jax_debug_infs  # compiled_fun can only raise in this case

    _ = core.jaxpr_as_fun(jaxpr)(*args)  # may raise, not return

    # If control reaches this line, we got a NaN on the output of `compiled`
    # but not `fun.call_wrapped` on the same arguments. Let's tell the user.
    msg = ("An invalid value was encountered in the output of the "
           f"`jit`-decorated function {name}. Because "
           "config.jax_debug_nans and/or config.jax_debug_infs is set, the "
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

pjit_p.def_impl(_pjit_call_impl)


@dataclasses.dataclass(frozen=True)
class SameDeviceAssignmentTuple:
  shardings: Tuple[PjitSharding, ...]
  # device_assignment is Optional because shardings can contain `AUTO` and in
  # that case `mesh` is compulsory to be used. So in that case
  # `_pjit_lower_cached` cache, resource_env will check against the devices.
  device_assignment: Optional[XLADeviceAssignment]

  def __hash__(self):
    shardings_hash = tuple(s._op_sharding_hash if isinstance(s, GSPMDSharding) else s # type: ignore
                           for s in self.shardings)
    if self.device_assignment is None:
      return hash(shardings_hash)
    else:
      return hash((shardings_hash, *self.device_assignment))

  def __eq__(self, other):
    if not isinstance(other, SameDeviceAssignmentTuple):
      return False
    return (all(pxla.are_op_shardings_equal(s._op_sharding, o._op_sharding)  # pytype: disable=attribute-error
                if isinstance(s, GSPMDSharding) and isinstance(o, GSPMDSharding)
                else s == o
                for s, o in safe_zip(self.shardings, other.shardings)) and
            self.device_assignment == other.device_assignment)


def _pjit_lower(
    jaxpr: core.ClosedJaxpr,
    in_shardings,
    out_shardings,
    *args, **kwargs):
  da = _fast_path_get_device_assignment(it.chain(in_shardings, out_shardings))
  in_shardings = SameDeviceAssignmentTuple(in_shardings, da)
  out_shardings = SameDeviceAssignmentTuple(out_shardings, da)
  return _pjit_lower_cached(jaxpr, in_shardings, out_shardings, *args, **kwargs,
                            lowering_platforms_override=mlir.thread_local_state.lowering_platforms_override)


@weakref_lru_cache
def _pjit_lower_cached(
    jaxpr: core.ClosedJaxpr,
    sdat_in_shardings: SameDeviceAssignmentTuple,
    sdat_out_shardings: SameDeviceAssignmentTuple,
    resource_env,
    donated_invars,
    name: str,
    in_is_global: Sequence[bool],
    keep_unused: bool,
    always_lower: bool,
    lowering_platforms_override: Sequence[str] = ()):
  in_shardings: Tuple[PjitShardingMinusUnspecified, ...] = cast(
      Tuple[PjitShardingMinusUnspecified, ...], sdat_in_shardings.shardings)
  out_shardings: Tuple[PjitSharding, ...] = sdat_out_shardings.shardings

  if resource_env is not None:
    pxla.resource_typecheck(jaxpr, resource_env, {}, lambda: "pjit")

  f = core.jaxpr_as_fun(jaxpr)
  f.__name__ = name
  fun = lu.wrap_init(f)

  if resource_env is not None:
    mesh = resource_env.physical_mesh
    api_name = 'pjit'
  else:
    # resource_env is `None` in the jit wrapper around pjit.
    mesh = None
    api_name = 'jit'

  # Convert to `NamedSharding` when `jax_array` is not enabled. This is
  # because GDA/SDA/DA are dependent on mesh for generating outputs.
  # NamedSharding is required for host-local inputs too.
  any_auto = pxla.check_if_any_auto(it.chain(in_shardings,  out_shardings))
  if not config.jax_array or any_auto:
    in_shardings: Tuple[MeshShardingMinusUnspecified, ...] = cast(  # type:ignore[no-redef]
        Tuple[MeshShardingMinusUnspecified, ...], tuple(
            NamedSharding._from_parsed_pspec(
                mesh, parse_flatten_op_sharding(i._op_sharding, mesh)[0]) # type: ignore
            if isinstance(i, GSPMDSharding) else i
            for i in in_shardings
    ))
    out_shardings: Tuple[MeshSharding, ...] = cast(  # type: ignore[no-redef]
        Tuple[MeshSharding, ...], tuple(
            NamedSharding._from_parsed_pspec(
                mesh, parse_flatten_op_sharding(o._op_sharding, mesh)[0]) # type: ignore
            if isinstance(o, GSPMDSharding) else o
            for o in out_shardings
    ))

  # For `pjit(xmap)` cases, it needs to take the `lower_mesh_computation` path
  # because `xmap` only supports SPMDAxisContext right now.
  if (any_auto or dispatch.jaxpr_has_primitive(jaxpr.jaxpr, 'xmap')):
    return pxla.lower_mesh_computation(
      fun, api_name, name, mesh,
      in_shardings, out_shardings, donated_invars,
      True, jaxpr.in_avals, tiling_method=None, in_is_global=in_is_global,
      lowering_platforms_override=lowering_platforms_override)
  else:
    # Pass `in_is_global` here because this path is taken by both host local
    # avals and global avals.
    # TODO(yashkatariya): Don't set committed to True always. Infer that from
    # the arguments just like dispatch.py in `sharded_lowering`.
    return pxla.lower_sharding_computation(
        fun, api_name, name, in_shardings, out_shardings, donated_invars,
        jaxpr.in_avals, in_is_global=in_is_global, keep_unused=keep_unused,
        always_lower=always_lower,
        devices_from_context=(
            None if mesh is None or mesh.empty else list(mesh.devices.flat)),
        lowering_platforms_override=lowering_platforms_override)


def pjit_staging_rule(trace, *args, **params):
  if (params["inline"] and
      all(_is_unspecified(i) for i in params["in_shardings"]) and
      all(_is_unspecified(o) for o in params["out_shardings"])):
    jaxpr = params['jaxpr']
    return core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
  else:
    return trace.default_process_primitive(pjit_p, args, params)

pe.custom_staging_rules[pjit_p] = pjit_staging_rule


def _pjit_abstract_eval(*args, jaxpr, out_shardings, resource_env,
                        out_positional_semantics, **_):
  if config.jax_array:
    return jaxpr.out_avals, jaxpr.effects
  return global_to_local(out_positional_semantics, jaxpr.out_avals,
                         out_shardings, resource_env.physical_mesh), jaxpr.effects
pjit_p.def_effectful_abstract_eval(_pjit_abstract_eval)


def _pjit_lowering(ctx, *args, name, jaxpr, in_shardings,
                   out_shardings, resource_env, donated_invars,
                   in_positional_semantics, out_positional_semantics,
                   keep_unused, inline):
  if not config.jax_jit_pjit_api_merge:
    if not isinstance(ctx.module_context.axis_context,
                      (mlir.SPMDAxisContext, mlir.ShardingContext)):
      raise RuntimeError("Nesting pjit() inside jit() is not allowed.")

  effects = ctx.tokens_in.effects()
  output_types = safe_map(mlir.aval_to_ir_types, ctx.avals_out)
  output_types = [mlir.token_type()] * len(effects) + output_types
  flat_output_types = util.flatten(output_types)

  arg_shardings = [None if _is_unspecified(i) else i._to_xla_op_sharding(aval.ndim)
                   for aval, i in safe_zip(ctx.avals_in, in_shardings)]
  result_shardings = [None if _is_unspecified(o) else o._to_xla_op_sharding(aval.ndim)
                      for aval, o in safe_zip(ctx.avals_out, out_shardings)]

  # TODO(b/228598865): inlined calls cannot have shardings set directly on the
  # inputs or outputs because they are lost during MLIR->HLO conversion.
  # using_sharding_annotation=False means we add an identity operation instead.
  func = mlir.lower_jaxpr_to_fun(
      ctx.module_context, name, jaxpr, effects, arg_shardings=arg_shardings,
      result_shardings=result_shardings, use_sharding_annotations=False,
      api_name=('jit' if resource_env is None else 'pjit'))
  args = (*ctx.dim_var_values, *ctx.tokens_in.tokens(), *args)
  call = func_dialect.CallOp(flat_output_types,
                             ir.FlatSymbolRefAttr.get(func.name.value),
                             mlir.flatten_lowering_ir_args(args))
  out_nodes = util.unflatten(call.results, safe_map(len, output_types))
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(zip(effects, tokens)))
  ctx.set_tokens_out(tokens_out)
  return out_nodes

mlir.register_lowering(pjit_p, _pjit_lowering)


def _pjit_batcher(insert_axis, spmd_axis_name,
                  axis_size, axis_name, main_type,
                  vals_in, dims_in,
                  jaxpr, in_shardings, out_shardings,
                  resource_env, donated_invars, name, in_positional_semantics,
                  out_positional_semantics, keep_unused, inline):
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
    in_positional_semantics=in_positional_semantics,
    out_positional_semantics=out_positional_semantics,
    keep_unused=keep_unused,
    inline=inline)
  return vals_out, axes_out

batching.spmd_axis_primitive_batchers[pjit_p] = partial(_pjit_batcher, False)
batching.axis_primitive_batchers[pjit_p] = partial(_pjit_batcher, False, None)
pxla.spmd_primitive_batchers[pjit_p] = partial(_pjit_batcher, True, None)

def _pjit_batcher_for_sharding(
    s: Union[GSPMDSharding, _UnspecifiedValue],
    dim: int, val: Tuple[str, ...], mesh, ndim: int):
  if _is_unspecified(s):
    return s
  if not val:
    new_op = s._op_sharding.clone()  # type: ignore
    tad = list(new_op.tile_assignment_dimensions)
    tad.insert(dim, 1)
    new_op.tile_assignment_dimensions = tad
    return GSPMDSharding(s._device_assignment, new_op)  # type: ignore
  else:
    assert isinstance(s, GSPMDSharding)
    assert mesh is not None and not mesh.empty
    parsed_pspec = parse_flatten_op_sharding(s._op_sharding, mesh)[0]  # type: ignore
    parsed_pspec = parsed_pspec.insert_axis_partitions(dim, val)
    mps = NamedSharding._from_parsed_pspec(mesh, parsed_pspec)
    return GSPMDSharding(mps._device_assignment, mps._to_xla_op_sharding(ndim))


def _pjit_jvp(primals_in, tangents_in,
              jaxpr, in_shardings, out_shardings,
              resource_env, donated_invars, name, in_positional_semantics,
              out_positional_semantics, keep_unused, inline):
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
      in_positional_semantics=(*in_positional_semantics, *_filter_zeros_in(in_positional_semantics)),
      out_positional_semantics=out_positional_semantics,
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
                     fwds_known: Tuple[Optional[int]]) -> core.ClosedJaxpr:
  updated_jaxpr = known_jaxpr.jaxpr.replace(
      outvars=[x for x, i in safe_zip(known_jaxpr.jaxpr.outvars, fwds_known)
               if i is None])
  return known_jaxpr.replace(jaxpr=updated_jaxpr)


def _pjit_partial_eval(trace, *in_tracers,
                       jaxpr, in_shardings, out_shardings,
                       resource_env, donated_invars, name, in_positional_semantics,
                       out_positional_semantics, keep_unused, inline):
  in_pvals = [t.pval for t in in_tracers]

  known_ins = tuple(pv.is_known() for pv in in_pvals)
  unknown_ins = tuple(not k for k in known_ins)
  known_jaxpr, unknown_jaxpr, unknown_outs, res_avals = pe.partial_eval_jaxpr_nounits(
      jaxpr, unknown_ins, instantiate=False)
  unknown_outs = tuple(unknown_outs)
  known_outs = tuple(not uk for uk in unknown_outs)
  num_residuals = len(res_avals)

  def keep_where(l, should_keep):
    return tuple(x for x, keep in zip(l, should_keep) if keep)

  if config.jax_array:
    residual_shardings = (_UNSPECIFIED,) * num_residuals
  else:
    da = list(resource_env.physical_mesh.devices.flat)
    residual_shardings = (GSPMDSharding.get_replicated(da),) * num_residuals
  # Compute the known outputs
  known_params = dict(
      jaxpr=known_jaxpr,
      in_shardings=keep_where(in_shardings, known_ins),
      out_shardings=(
          keep_where(out_shardings, known_outs) + residual_shardings),
      resource_env=resource_env,
      donated_invars=keep_where(donated_invars, known_ins),
      name=name,
      in_positional_semantics=keep_where(in_positional_semantics, known_ins),
      out_positional_semantics=out_positional_semantics,
      keep_unused=keep_unused,
      inline=inline)

  # TODO(yashkatariya): After xla_extension_version is bumped to >= 123, make
  # this condition: `if not config.jax_array`.
  if (resource_env is not None and xla_extension_version < 123) or not config.jax_array:
    if num_residuals:
      in_is_global = _calc_is_global_sequence(
          known_params['in_positional_semantics'], known_params['in_shardings'])
      compiled = _pjit_lower(
          known_params["jaxpr"], known_params["in_shardings"],
          known_params["out_shardings"], known_params["resource_env"],
          known_params["donated_invars"], known_params["name"],
          in_is_global, known_params['keep_unused'], always_lower=False).compile(
              _allow_propagation_to_outputs=[True] * len(known_params['out_shardings']),
              _allow_compile_replicated=False)
      da = compiled._device_assignment
      _, out_gspmd_shardings = pxla.get_gspmd_shardings_from_executable(
          compiled.xla_executable, da, len(known_jaxpr.in_avals),
          len(known_jaxpr.out_avals))
      assert len(out_gspmd_shardings) == len(known_jaxpr.out_avals), (
          len(out_gspmd_shardings), len(known_jaxpr.out_avals))
      out_op_shardings = [o._to_xla_op_sharding(a.ndim) for o, a in
                          safe_zip(out_gspmd_shardings, known_jaxpr.out_avals)]
      residual_op_shardings = tuple(out_op_shardings[-num_residuals:])
    else:
      residual_op_shardings = ()
    assert len(residual_shardings) == len(residual_op_shardings), (
        len(residual_shardings), len(residual_op_shardings))
    residual_shardings = tuple(GSPMDSharding(da, op) for op in residual_op_shardings)
    known_params['out_shardings'] = (
        keep_where(out_shardings, known_outs) + residual_shardings)

  fwds_known = pe._jaxpr_forwarding(known_params['jaxpr'].jaxpr)

  # Only forward the outvars where the out_sharding is UNSPECIFIED.
  known_user_out_shardings = keep_where(known_params['out_shardings'], known_outs)
  fwds_known_user = [
      fwd if _is_unspecified(os) else None
      for os, fwd in safe_zip(known_user_out_shardings,
                              fwds_known[:len(known_user_out_shardings)])]
  fwds_known = fwds_known_user + fwds_known[len(known_user_out_shardings):]
  del fwds_known_user

  # Remove forwarded outvars and out_shardings
  known_params['jaxpr'] = _known_jaxpr_fwd(known_params['jaxpr'], tuple(fwds_known))
  known_out_shardings = tuple(
      s for s, i in safe_zip(known_params['out_shardings'], fwds_known) if i is None)
  known_params['out_shardings'] = known_out_shardings
  del known_out_shardings

  assert len(known_params['out_shardings']) == len(known_params['jaxpr'].out_avals)

  # Bind known things to pjit_p.
  known_inputs = [pv.get_known() for pv in in_pvals if pv.is_known()]
  all_known_outs = pjit_p.bind(*known_inputs, **known_params)

  known_outs_iter = iter(all_known_outs)
  all_known_outs = [next(known_outs_iter)
                    if fwd_idx is None else known_inputs[fwd_idx]
                    for fwd_idx in fwds_known]
  assert next(known_outs_iter, None) is None
  del known_outs_iter, known_inputs

  if num_residuals:
    known_out_vals, residual_vals = \
        split_list(all_known_outs, [len(all_known_outs) - num_residuals])
  else:
    known_out_vals, residual_vals = all_known_outs, ()
  residual_tracers = [trace.new_instantiated_const(residual) for residual in residual_vals]

  # The convention of partial_eval_jaxpr_nounits is to place residual binders
  # at the front of the jaxpr produced, so we move them to the back since both
  # the jaxpr equation built below and the pjit transpose rule assume a
  # residual-inputs-last convention.
  unknown_jaxpr = pe.move_binders_to_back(
      unknown_jaxpr, [True] * num_residuals + [False] * sum(unknown_ins))
  # Prepare unknown tracers
  unknown_params = dict(
      jaxpr=unknown_jaxpr,
      in_shardings=(keep_where(in_shardings, unknown_ins) + residual_shardings),
      out_shardings=keep_where(out_shardings, unknown_outs),
      resource_env=resource_env,
      donated_invars=(keep_where(donated_invars, unknown_ins) +
                      (False,) * num_residuals),
      name=name,
      in_positional_semantics=(keep_where(
          in_positional_semantics, unknown_ins) + (out_positional_semantics,) * num_residuals),
      out_positional_semantics=out_positional_semantics,
      keep_unused=keep_unused,
      inline=inline)
  unknown_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
  if config.jax_array:
    unknown_out_avals = unknown_jaxpr.out_avals
  else:
    unknown_out_avals = global_to_local(
        unknown_params["out_positional_semantics"], unknown_jaxpr.out_avals,
        unknown_params["out_shardings"],
        unknown_params["resource_env"].physical_mesh)
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
    num_res: int, params_known: dict, params_staged: dict
  ) -> Tuple[dict, dict]:
  # prune inputs to jaxpr_known according to unks_in
  donated_invars_known, _ = pe.partition_list(unks_in, params_known['donated_invars'])
  in_shardings_known, _ = pe.partition_list(unks_in, params_known['in_shardings'])
  in_positional_semantics_known, _ = pe.partition_list(
      unks_in, params_known['in_positional_semantics'])
  if num_res == 0:
    residual_shardings = []
  else:
    residual_shardings = [_UNSPECIFIED] * num_res
  _, out_shardings_known = pe.partition_list(kept_outs_known, params_known['out_shardings'])
  new_params_known = dict(params_known,
                          in_shardings=tuple(in_shardings_known),
                          out_shardings=(*out_shardings_known, *residual_shardings),
                          donated_invars=tuple(donated_invars_known),
                          in_positional_semantics=tuple(in_positional_semantics_known))
  assert len(new_params_known['in_shardings']) == len(params_known['jaxpr'].in_avals)
  assert len(new_params_known['out_shardings']) == len(params_known['jaxpr'].out_avals)

  # added num_res new inputs to jaxpr_staged, and pruning according to inst_in
  _, donated_invars_staged = pe.partition_list(inst_in, params_staged['donated_invars'])
  donated_invars_staged = [False] * num_res + donated_invars_staged
  _, in_shardings_staged = pe.partition_list(inst_in, params_staged['in_shardings'])
  in_shardings_staged = [*residual_shardings, *in_shardings_staged]
  _, in_positional_semantics_staged = pe.partition_list(
      inst_in, params_staged['in_positional_semantics'])
  in_positional_semantics_staged = [
      pxla._PositionalSemantics.GLOBAL] * num_res + in_positional_semantics_staged

  _, out_shardings_staged = pe.partition_list(kept_outs_staged, params_staged['out_shardings'])

  new_params_staged = dict(params_staged,
                           in_shardings=tuple(in_shardings_staged),
                           out_shardings=tuple(out_shardings_staged),
                           donated_invars=tuple(donated_invars_staged),
                           in_positional_semantics=tuple(in_positional_semantics_staged))
  assert len(new_params_staged['in_shardings']) == len(params_staged['jaxpr'].in_avals)
  assert len(new_params_staged['out_shardings']) == len(params_staged['jaxpr'].out_avals)
  return new_params_known, new_params_staged

pe.partial_eval_jaxpr_custom_rules[pjit_p] = \
    partial(pe.closed_call_partial_eval_custom_rule, 'jaxpr',
            _pjit_partial_eval_custom_params_updater)


@lu.cache
def _pjit_transpose_trace(fun, in_avals, api_name):
  transpose_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      fun, in_avals, debug_info=pe.debug_info_final(fun, api_name))
  transpose_jaxpr = core.ClosedJaxpr(transpose_jaxpr, consts)
  return transpose_jaxpr


def _pjit_transpose(reduce_axes, cts_in, *primals_in,
                    jaxpr, in_shardings, out_shardings,
                    resource_env, donated_invars, name, in_positional_semantics,
                    out_positional_semantics, keep_unused, inline):
  def prune_type(ty, xs, maybe_zeros):
    return tuple(x for x, mz in zip(xs, maybe_zeros) if type(mz) is not ty)

  body = lu.wrap_init(ad.closed_backward_pass)
  body = lu.hashable_partial(body, jaxpr, reduce_axes, False)
  primals_and_nz_cts_in, in_treedef = tree_flatten((primals_in, cts_in))
  body, cts_out_treedef_thunk = flatten_fun_nokwargs(body, in_treedef)

  transpose_in_shardings = (
    *prune_type(ad.UndefinedPrimal, in_shardings, primals_in),
    *prune_type(ad.Zero, out_shardings, cts_in)
  )
  transpose_in_positional_semantics = (
    *prune_type(ad.UndefinedPrimal, in_positional_semantics, primals_in),
    *prune_type(ad.Zero, (out_positional_semantics,) * len(cts_in), cts_in)
  )
  global_cts_in_avals = tuple(core.raise_to_shaped(core.get_aval(ct))
                              for ct in primals_and_nz_cts_in)
  if not config.jax_array:
    global_cts_in_avals = tuple(local_to_global(
        transpose_in_positional_semantics, global_cts_in_avals,
        transpose_in_shardings, resource_env.physical_mesh))

  api_name = 'jit' if resource_env is None else 'pjit'
  transpose_jaxpr = _pjit_transpose_trace(body, global_cts_in_avals, api_name)
  cts_out_treedef = cts_out_treedef_thunk()
  transpose_out_shardings = prune_type(
      ad.Zero,
      in_shardings,
      tree_unflatten(cts_out_treedef, [object()] * cts_out_treedef.num_leaves))

  nz_cts_out = pjit_p.bind(
      *primals_and_nz_cts_in,
      jaxpr=transpose_jaxpr,
      in_shardings=transpose_in_shardings,
      out_shardings=transpose_out_shardings,
      resource_env=resource_env,
      donated_invars=(False,) * len(primals_and_nz_cts_in),
      name=name,
      in_positional_semantics=transpose_in_positional_semantics,
      out_positional_semantics=out_positional_semantics,
      keep_unused=keep_unused,
      inline=inline)
  return tree_unflatten(cts_out_treedef, nz_cts_out)
ad.reducing_transposes[pjit_p] = _pjit_transpose


@weakref_lru_cache
def _dce_jaxpr_pjit(
    jaxpr: core.ClosedJaxpr, used_outputs: Tuple[bool]
) -> Tuple[core.ClosedJaxpr, List[bool]]:
  new_jaxpr, used_inputs = pe.dce_jaxpr(jaxpr.jaxpr, used_outputs)
  return core.ClosedJaxpr(new_jaxpr, jaxpr.consts), used_inputs


def dce_jaxpr_pjit_rule(used_outputs: List[bool], eqn: core.JaxprEqn
                        ) -> Tuple[List[bool], Optional[core.JaxprEqn]]:
  dced_jaxpr, used_inputs = _dce_jaxpr_pjit(
      eqn.params['jaxpr'], tuple(used_outputs))

  def keep_where(xs, keeps):
    return tuple(x for x, keep in safe_zip(xs, keeps) if keep)

  eqn_params = eqn.params
  new_params = dict(
      eqn_params,
      jaxpr=dced_jaxpr,
      in_shardings=keep_where(eqn_params["in_shardings"], used_inputs),
      out_shardings=keep_where(eqn_params["out_shardings"], used_outputs),
      in_positional_semantics=keep_where(eqn_params["in_positional_semantics"],
                                         used_inputs),
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
        f"{pxla.show_axes(overlap)})")

def _resource_typing_pjit(avals, params, source_info, resource_env, named_axis_resources):
  jaxpr = params["jaxpr"]
  what = "pjit input"
  if (resource_env is not None and params['resource_env'] is not None and
      resource_env.physical_mesh != params['resource_env'].physical_mesh):
      raise RuntimeError("Changing the physical mesh is not allowed inside pjit.")

  for aval, s in zip(jaxpr.in_avals, params['in_shardings']):
    if _is_unspecified(s) or is_auto(s):
      continue
    elif hasattr(s, '_original_sharding') and hasattr(
        s._original_sharding, '_parsed_pspec'):
      parsed_pspec = s._original_sharding._parsed_pspec
    else:
      if resource_env is not None:
        parsed_pspec = parse_flatten_op_sharding(
            s._op_sharding, resource_env.physical_mesh)[0]
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
    if _is_unspecified(s) or is_auto(s):
      continue
    elif hasattr(s, '_original_sharding') and hasattr(
        s._original_sharding, '_parsed_pspec'):
      parsed_pspec = s._original_sharding._parsed_pspec
    else:
      if resource_env is not None:
        parsed_pspec = parse_flatten_op_sharding(
            s._op_sharding, resource_env.physical_mesh)[0]
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
  if all(p == pxla._PositionalSemantics.GLOBAL
         for p in params['in_positional_semantics']):
    del params['in_positional_semantics']
  if params['out_positional_semantics'] == pxla._PositionalSemantics.GLOBAL:
    del params['out_positional_semantics']
  if all(pxla._is_unspecified(s) for s in params['in_shardings']):
    del params['in_shardings']
  if all(pxla._is_unspecified(s) for s in params['out_shardings']):
    del params['out_shardings']
  if not params['keep_unused']:
    del params['keep_unused']
  if (params['resource_env'] is None or
      params['resource_env'].physical_mesh.empty):
    del params['resource_env']
  return core._pp_eqn(eqn.replace(params=params), context, settings)
core.pp_eqn_rules[pjit_p] = _pjit_pp_rule


# -------------------- with_sharding_constraint --------------------

def _resolve_wsc_args(axis_resources, shardings):
  if not _is_unspecified(axis_resources) and not _is_unspecified(shardings):
    raise ValueError(
        'Setting both axis_resources and shardings is not '
        'allowed. axis_resources is deprecated. Please use shardings.')
  if _is_unspecified(axis_resources) and _is_unspecified(shardings):
    raise ValueError(
        'Not specifying shardings to `with_sharding_constraint` is not allowed. '
        'Please specify the shardings argument with a concrete sharding. Note '
        'that axis_resources is deprecated, so use the shardings argument.')

  if not _is_unspecified(axis_resources):
    final_shardings = axis_resources
  else:
    final_shardings = shardings
  return final_shardings


# TODO(yashkatariya): Remove the axis_resources argument and make the signature
# `with_sharding_constraint(x, shardings)` with no defaults after deprecation
# period is finished. The deprecation period expires 3 months from Feb 13, 2023.
def with_sharding_constraint(x, axis_resources=_UNSPECIFIED,
                             shardings=_UNSPECIFIED):
  final_shardings = _resolve_wsc_args(axis_resources, shardings)
  x_flat, tree = tree_flatten(x)
  user_shardings, _, _ = _prepare_axis_resources(
      final_shardings, "shardings", allow_unconstrained_dims=True)
  del final_shardings

  user_shardings_flat = tuple(
      flatten_axes("with_sharding_constraint shardings", tree, user_shardings))
  del user_shardings

  resource_env = pxla.thread_resources.env
  mesh = resource_env.physical_mesh

  if config.jax_array:
    shardings_flat = [_create_sharding_for_array(mesh, a)
                      for a in user_shardings_flat]
    unconstrained_dims = [get_unconstrained_dims(s)
                          if isinstance(s, NamedSharding) else {}
                          for s in shardings_flat]
  else:
    shardings_flat = [pxla.create_mesh_pspec_sharding(mesh, a.user_spec, a)
                      for a in user_shardings_flat]
    # Calculate unconstrained_dims from NamedSharding because that information
    # is lost when converted to OpSharding. Bind unconstrained_dims to
    # with_sharding_constraint primitive.
    unconstrained_dims = [get_unconstrained_dims(s) for s in shardings_flat]

  del user_shardings_flat

  pjit_check_aval_sharding(shardings_flat, x_flat, "with_sharding_constraint arguments",
                           allow_uneven_sharding=True)

  outs = [sharding_constraint_p.bind(xf, sharding=to_gspmd_sharding(i, xf.ndim),
                                     resource_env=resource_env,
                                     unconstrained_dims=ud)
          for xf, i, ud in safe_zip(x_flat, shardings_flat, unconstrained_dims)]
  return tree_unflatten(tree, outs)

def _sharding_constraint_impl(x, sharding, resource_env, unconstrained_dims):
  # TODO(skye): can we also prevent this from being called in other
  # non-pjit contexts? (e.g. pmap, control flow)
  raise NotImplementedError(
      "with_sharding_constraint() should only be called inside pjit()")

sharding_constraint_p = core.Primitive("sharding_constraint")
sharding_constraint_p.def_impl(_sharding_constraint_impl)
sharding_constraint_p.def_abstract_eval(lambda x, **_: x)
ad.deflinear2(sharding_constraint_p,
              lambda ct, _, **params: (sharding_constraint_p.bind(ct, **params),))

def _sharding_constraint_hlo_lowering(ctx, x_node, *, sharding,
                                      resource_env, unconstrained_dims):
  aval, = ctx.avals_in
  axis_ctx = ctx.module_context.axis_context
  # axis_ctx and manual_axes is *only used with xmap* and xmap only works with
  # NamedSharding. So convert the GSPMDSharding to NamedSharding
  # and then convert it back with the added special axes.
  if isinstance(axis_ctx, mlir.SPMDAxisContext):
    mesh = resource_env.physical_mesh
    parsed_pspec = parse_flatten_op_sharding(sharding._op_sharding, mesh)[0]
    mps = NamedSharding._from_parsed_pspec(mesh, parsed_pspec)
    sharding = GSPMDSharding(
        mps._device_assignment, mps._to_xla_op_sharding(aval.ndim, axis_ctx=axis_ctx))
  return [
      mlir.wrap_with_sharding_op(
          x_node,
          sharding._to_xla_op_sharding(aval.ndim),
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
        params['sharding']._op_sharding, resource_env.physical_mesh)[0]
  _check_resources_against_named_axes(
    "with_sharding_constraint input", aval, parsed_pspec, named_axis_resources)

pxla.custom_resource_typing_rules[sharding_constraint_p] = \
    _resource_typing_sharding_constraint

# -------------------- helpers --------------------

def get_array_mapping(
    axis_resources: Union[ParsedPartitionSpec, _AUTOAxisResource, _UnspecifiedValue]
) -> pxla.ArrayMappingOrAutoOrUnspecified:
  # TODO(yashkatariya): Use `TypeGuard` on `is_auto` when it is supported.
  # Don't use `is_auto` here to satisfy pytype and mypy.
  if isinstance(axis_resources, (_AUTOAxisResource, _UnspecifiedValue)):
    return axis_resources
  return OrderedDict((axis, i)
                     for i, axes in enumerate(axis_resources)
                     if axes is not None for axis in axes)


def to_gspmd_sharding(s: XLACompatibleSharding, ndim: int) -> GSPMDSharding:
  if isinstance(s, GSPMDSharding):
    return s
  gspmd_sharding = GSPMDSharding(
      s._device_assignment, s._to_xla_op_sharding(ndim))
  gspmd_sharding._original_sharding = s
  return gspmd_sharding


def get_unconstrained_dims(sharding: NamedSharding):
  return {i for i, axes in enumerate(sharding._parsed_pspec)
          if axes is None}


def global_to_local(positional_semantics, avals, shardings, mesh):
  if config.jax_array:
    return avals
  if isinstance(positional_semantics, pxla._PositionalSemantics):
    positional_semantics = [positional_semantics] * len(shardings)

  out = []
  for aval, s, ps in safe_zip(avals, shardings, positional_semantics):
    if (ps == pxla._PositionalSemantics.GLOBAL or
        pxla.is_op_sharding_replicated(s._op_sharding)):
      out.append(aval)
    else:
      # This path is only taken by host-local values. GDA, Array and fully
      # replicated avals don't go through this code path. To convert global
      # avals to host local avals, round trip it via NamedSharding.
      parsed_pspec = parse_flatten_op_sharding(s._op_sharding, mesh)[0]
      out.append(mesh._global_to_local(get_array_mapping(parsed_pspec), aval))
  return out


def local_to_global(positional_semantics, avals, shardings, mesh):
  if config.jax_array:
    return avals
  out = []
  for aval, s, ps in safe_zip(avals, shardings, positional_semantics):
    if (ps == pxla._PositionalSemantics.GLOBAL or
        pxla.is_op_sharding_replicated(s._op_sharding)):
      out.append(aval)
    else:
      # This path is only taken by host-local values. GDA, Array and fully
      # replicated avals don't go through this code path. To convert host local
      # avals to global avals, round trip it via NamedSharding.
      parsed_pspec = parse_flatten_op_sharding(s._op_sharding, mesh)[0]
      out.append(mesh._local_to_global(get_array_mapping(parsed_pspec), aval))
  return out


def _calc_is_global_sequence(in_positional_semantics, in_shardings):
  if config.jax_array:
    return (True,) * len(in_positional_semantics)
  return tuple((ips == pxla._PositionalSemantics.GLOBAL or
                pxla.is_op_sharding_replicated(i._op_sharding))
               for ips, i in safe_zip(in_positional_semantics, in_shardings))

def _get_in_positional_semantics(arg) -> pxla._PositionalSemantics:
  if isinstance(arg, GDA):
    return pxla._PositionalSemantics.GLOBAL
  return pxla.positional_semantics.val


def _fast_path_get_device_assignment(
    shardings: Iterable[PjitSharding]) -> Optional[XLADeviceAssignment]:
  da = None
  for i in shardings:
    if is_auto(i) or _is_unspecified(i):
      continue
    da = i._device_assignment  # type: ignore
    break
  return da


def _maybe_replace_from_gda_with_pspec(
    in_shardings_flat, args_flat) -> Sequence[XLACompatibleSharding]:

  @lru_cache()
  def _gda_check_and_get_sharding(
      gda_sharding: NamedSharding, in_sharding: GSPMDSharding, ndim: int):
    if not _is_from_gda(in_sharding) and not pxla.are_op_shardings_equal(
        gda_sharding._to_xla_op_sharding(ndim),
        in_sharding._to_xla_op_sharding(ndim)):
      raise ValueError(
          f"Got an input GDA to pjit with different partitioning than specified in "
          "the in_axis_resources argument to pjit. The partitioning must match, or "
          "use `jax.experimental.pjit.FROM_GDA` in `in_axis_resources` for GDA. "
          f"Got GDA sharding: {gda_sharding} and "
          f"pjit sharding: {in_sharding._original_sharding}")  # type: ignore
    return to_gspmd_sharding(gda_sharding, ndim)

  out = []
  for in_sharding_flat, arg in safe_zip(in_shardings_flat, args_flat):
    if is_auto(in_sharding_flat):
      out.append(in_sharding_flat)
    elif isinstance(arg, array.ArrayImpl):
      out.append(to_gspmd_sharding(arg.sharding, arg.ndim))
    elif isinstance(arg, GDA):
      gda_sharding = pxla.create_mesh_pspec_sharding(arg.mesh, arg.mesh_axes)
      out.append(_gda_check_and_get_sharding(gda_sharding, in_sharding_flat, arg.ndim))
    else:
      out.append(in_sharding_flat)
  return tuple(out)


def _maybe_check_pjit_gda_mesh(args, mesh):
  for x in args:
    if isinstance(x, GDA) and x.mesh != mesh:
      raise ValueError("Pjit's mesh and GDA's mesh should be equal. Got Pjit "
                       f"mesh: {mesh},\n GDA mesh: {x.mesh}")

# -------------------- XLA OpSharding to PartitionSpec --------------------
# Note that OpSharding is more expressive than PartitionSpecs, so it's not
# always possible to convert them, but the code below should at least
# support handle all cases when this is possible.

def strides_for_sizes(sizes):
  """Returns an array of strides for major-to-minor sizes."""
  return np.cumprod(sizes[::-1])[::-1] // np.asarray(sizes)

def unflatten_array(named_sizes, assignment):
  """Recovers the ordering of axis names based on a device assignment.

  The device assignments that this function can convert into axis orders
  are of the form::

    np.arange(np.prod(named_sizes.values())).transpose(...).flatten()

  for some transposition ``...``. This is satisfied by all OpSharding assignments
  generated from partition specs.

  Arguments:
    named_sizes: A dictionary mapping axis names to their sizes.
    assignment: A permutation of integers between 0 and the product of all
      named sizes.

  Returns:
    A major-to-minor list of axis names that corresponds to the given assignment.
  """
  named_sizes = {name: size for name, size in named_sizes.items() if size != 1}
  sizes = np.fromiter(named_sizes.values(), dtype=np.int64)
  strides = strides_for_sizes(sizes)
  dims = explode_superdims(sizes, unflatten_superdims(assignment))
  dim_to_name = {(size, stride): name for size, stride, name in zip(sizes, strides, named_sizes)}
  return [dim_to_name[d] for d in dims]

def unflatten_superdims(assignment):
  """Unflatten a list of dimension sizes and their strides that generates assignment.

  If this function succeeds for a given ``assignment``, then the following property
  should be satisfied::

    dims_with_strides = unflatten_superdims(assignment)
    base_array = np.arange(map(fst, sorted(dims_with_strides, key=snd, reverse=True)))
    assignment == base_array.transpose(argsort(dims_with_strides, key=snd, reverse=True)).flatten()

  That is, the returned dimensions list all sizes of the base array (with strides
  indicating their initial order). The order of dimensions in the list corresponds
  to the permutation that applied to the base array generates the assignment.
  """
  def check(cond):
    if cond: return
    raise NotImplementedError("Failed to convert OpSharding into a ShardingSpec. "
                              "Please open a bug report!")
  flat_assignment = np.asarray(assignment, dtype=np.int64)
  check(flat_assignment[0] == 0)
  dims = []
  while flat_assignment.size > 1:
    stride = flat_assignment[1]
    for i in range(len(flat_assignment)):
      if flat_assignment[i] != i * stride: break
    else:
      # After this loop i should point to an "element after the sequence", so
      # we have to increment it if the whole array is a strided sequence.
      i += 1
    size = i
    dims.append((size, stride))
    assert size > 1  # Ensure progress
    flat_assignment = flat_assignment[::size]
  return dims

def explode_superdims(sizes, dims):
  """Explode superdims to fit a known shape.

  The unflattening process might mistakenly generate too few too large dimensions.
  For example, ``unflatten_superdims(np.arange(n))`` always returns ``[(n, 1)]``.
  This function takes a list of such contiguous super-dimensions and splits them
  into smaller dimensions such that::

    set(map(fst, explode_superdims(sizes, dims))) == set(sizes)
  """
  strides_to_sizes = {stride: size for size, stride in zip(sizes, strides_for_sizes(sizes))}
  dims = list(reversed(dims))
  final_dims = []
  for size, stride in dims:
    target_size = strides_to_sizes[stride]
    new_dims = []
    while size > target_size:
      assert target_size > 1  # Ensure progress
      assert size % target_size == 0
      new_dims.append((target_size, stride))
      size //= target_size
      stride *= target_size
      target_size = strides_to_sizes[stride]
    assert size == target_size
    new_dims.append((size, stride))
    final_dims += reversed(new_dims)
  return final_dims

def parse_flatten_op_sharding(op_sharding: xc.OpSharding,
                              mesh: pxla.Mesh) -> Sequence[ParsedPartitionSpec]:
  if op_sharding.type == xc.OpSharding.Type.TUPLE:
    out: List[ParsedPartitionSpec] = []
    for s in op_sharding.tuple_shardings:
      out.extend(parse_flatten_op_sharding(s, mesh))
    return out
  elif op_sharding.type == xc.OpSharding.Type.REPLICATED:
    return [CanonicalizedParsedPartitionSpec(ParsedPartitionSpec(None, ()))]
  elif op_sharding.type == xc.OpSharding.Type.OTHER:
    mesh_shape = mesh.shape
    mesh_axis_order = unflatten_array(mesh.shape, op_sharding.tile_assignment_devices)
    mesh_axis = iter(mesh_axis_order)
    shape = op_sharding.tile_assignment_dimensions
    partitions = []
    for dim_size in shape:
      dim_partitions = []
      while dim_size > 1:
        axis = next(mesh_axis)
        axis_size = mesh_shape[axis]
        assert dim_size % axis_size == 0
        dim_size //= axis_size
        dim_partitions.append(axis)
      partitions.append(tuple(dim_partitions))
    if op_sharding.last_tile_dims == [xc.OpSharding.Type.REPLICATED]:
      replicate_on_last_tile_dim = True
    else:
      replicate_on_last_tile_dim = op_sharding.replicate_on_last_tile_dim
      if op_sharding.last_tile_dims:
        raise NotImplementedError("Unhandled OpSharding type. Please open a bug report!")
    if replicate_on_last_tile_dim:
      partitions = partitions[:-1]
    return [ParsedPartitionSpec('<internally generated spec>', partitions)]
  else:
    raise AssertionError("Unhandled OpSharding type. Please open a bug report!")


def _get_op_sharding(op_sharding) -> Sequence[xc.OpSharding]:
  if op_sharding.type == xc.OpSharding.Type.TUPLE:
    out: List[xc.OpSharding] = []
    for s in op_sharding.tuple_shardings:
      out.extend(_get_op_sharding(s))
    return out
  else:
    return [op_sharding]


_get_single_pspec = lambda p: pxla.array_mapping_to_axis_resources(
    cast(pxla.ArrayMapping, get_array_mapping(p)))

def _get_partition_spec(ppspec: Sequence[ParsedPartitionSpec]) -> Sequence[PartitionSpec]:
  return [_get_single_pspec(p) for p in ppspec]


def _get_op_sharding_from_executable(
    executable) -> Tuple[Sequence[xc.OpSharding], Sequence[xc.OpSharding]]:
  in_op_shardings: List[xc.OpSharding] = []
  parameter_shardings_from_xla = executable.get_parameter_shardings()
  if parameter_shardings_from_xla is not None:
    in_op_shardings = parameter_shardings_from_xla

  out_op_shardings: List[xc.OpSharding] = []
  output_shardings_from_xla = executable.get_output_shardings()
  if output_shardings_from_xla is not None:
    out_op_shardings = output_shardings_from_xla

  return in_op_shardings, out_op_shardings


def _get_ppspec_from_executable(executable, mesh) -> Tuple[Sequence[ParsedPartitionSpec], Sequence[ParsedPartitionSpec]]:
  input_op_shardings: Sequence[xc.OpSharding] = executable.hlo_modules()[0].spmd_parameters_shardings
  output_op_sharding: xc.OpSharding = executable.hlo_modules()[0].spmd_output_sharding
  in_ppspec: List[ParsedPartitionSpec] = []
  for s in input_op_shardings:
    in_ppspec.extend(parse_flatten_op_sharding(s, mesh))
  out_ppspec = parse_flatten_op_sharding(output_op_sharding, mesh)
  return in_ppspec, out_ppspec


def _get_pspec_from_executable(
    executable, mesh: pxla.Mesh
) -> Tuple[Tuple[PartitionSpec, ...], Tuple[PartitionSpec, ...]]:
  in_ppspec, out_ppspec = _get_ppspec_from_executable(executable, mesh)
  out_partition_spec = _get_partition_spec(out_ppspec)
  in_partition_spec = _get_partition_spec(in_ppspec)
  return tuple(in_partition_spec), tuple(out_partition_spec)
