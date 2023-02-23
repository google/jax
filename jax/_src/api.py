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

"""JAX user-facing transformations and utilities.

The transformations here mostly wrap internal transformations, providing
convenience flags to control behavior and handling Python containers of
arguments and outputs. The Python containers handled are pytrees (see
tree_util.py), which include nested tuples/lists/dicts, where the leaves are
arrays.
"""
from __future__ import annotations

import collections
import functools
from functools import partial
import inspect
import typing
from typing import (Any, Callable, Generator, Hashable, Iterable, List, Literal,
                    NamedTuple, Optional, Sequence, Tuple, TypeVar, Union, overload)

import numpy as np
from contextlib import contextmanager, ExitStack

import jax
from jax._src import linear_util as lu
from jax import stages
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           tree_structure, tree_transpose, tree_leaves,
                           Partial, PyTreeDef, all_leaves)
from jax._src import callback as jcb
from jax._src import core
from jax._src import device_array
from jax._src import dispatch
from jax._src import effects
from jax._src import array
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import pjit
from jax._src.core import eval_jaxpr
from jax._src.api_util import (
    flatten_fun, apply_flat_fun, flatten_fun_nokwargs, flatten_fun_nokwargs2,
    argnums_partial, argnums_partial_except, flatten_axes, donation_vector,
    rebase_donate_argnums, _ensure_index, _ensure_index_tuple,
    shaped_abstractify, _ensure_str_tuple, argnames_partial_except,
    validate_argnames, validate_argnums, check_callable, resolve_argnums,
    FLAGS)
from jax._src.lax import lax as lax_internal
from jax._src.lib import jax_jit
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax._src.lib import pmap_lib
from jax._src.sharding import PmapSharding
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import broadcast_prefix, _generate_key_paths
from jax._src.util import (unzip2, curry, safe_map, safe_zip, prod, split_list,
                           extend_name_stack, new_name_stack, wrap_name, cache,
                           wraps, HashableFunction, weakref_lru_cache)


# Unused imports to be exported
from jax.ad_checkpoint import checkpoint_policies, checkpoint as new_checkpoint
from jax.custom_batching import custom_vmap
from jax.custom_derivatives import (closure_convert, custom_gradient, custom_jvp,
                                    custom_vjp, linear_call)
from jax.custom_transpose import custom_transpose
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax._src.interpreters import xla

from jax._src.config import (
    config,
    disable_jit as _disable_jit,
    debug_nans as config_debug_nans,
    debug_infs as config_debug_infs,
    _thread_local_state as config_thread_local_state,
    explicit_device_put_scope as config_explicit_device_put_scope,
    explicit_device_get_scope as config_explicit_device_get_scope)
from jax._src.core import ShapedArray, raise_to_shaped
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import pxla
from jax._src.lib.xla_bridge import (device_count, local_device_count, devices,
                                     local_devices, process_index,
                                     process_count, host_id, host_ids,
                                     host_count, default_backend)


traceback_util.register_exclusion(__file__)

_dtype = partial(dtypes.dtype, canonicalize=True)

AxisName = Hashable

Device = xc.Device

# These TypeVars are used below to express the fact that function types
# (i.e. call signatures) are invariant under the vmap transformation.
F = TypeVar("F", bound=Callable)
T = TypeVar("T")
U = TypeVar("U")

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


def _nan_check_posthook(fun, args, kwargs, output):
  """Hook function called by the C++ jit/pmap to perform NaN checking."""
  leaves = tree_leaves(output)

  buffers = []
  for da_or_sda in leaves:
    if hasattr(da_or_sda, "device_buffers"):
      buffers.extend(da_or_sda.device_buffers)
    # TODO(yashkatariya): Remove the `elif` branch once `jax.Array` is enabled
    # by default.
    elif hasattr(da_or_sda, "device_buffer"):
      buffers.append(da_or_sda.device_buffer)

  try:
    if jax.config.jax_jit_pjit_api_merge:
      dispatch.check_special(pjit.pjit_p, buffers)
    else:
      dispatch.check_special(xla.xla_call_p, buffers)
  except FloatingPointError:
    # compiled_fun can only raise in this case
    assert config.jax_debug_nans or config.jax_debug_infs
    print("Invalid nan value encountered in the output of a C++-jit/pmap "
          "function. Calling the de-optimized version.")
    fun._cache_miss(*args, **kwargs)[0]  # probably won't return

def _update_debug_special_global(_):
  if config._read("jax_debug_nans") or config._read("jax_debug_infs"):
    jax_jit.global_state().post_hook = _nan_check_posthook
  else:
    jax_jit.global_state().post_hook = None

def _update_debug_special_thread_local(_):
  if (getattr(config_thread_local_state, "jax_debug_nans", False) or
      getattr(config_thread_local_state, "jax_debug_infs", False)):
    jax_jit.thread_local_state().post_hook = _nan_check_posthook
  else:
    jax_jit.thread_local_state().post_hook = None

config_debug_nans._add_hooks(_update_debug_special_global,
                             _update_debug_special_thread_local)
config_debug_infs._add_hooks(_update_debug_special_global,
                             _update_debug_special_thread_local)


float0 = dtypes.float0


def jit(
    fun: Callable,
    *,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
  ) -> stages.Wrapped:
  """Sets up ``fun`` for just-in-time compilation with XLA.

  Args:
    fun: Function to be jitted. ``fun`` should be a pure function, as
      side-effects may only be executed once.

      The arguments and return value of ``fun`` should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.

      JAX keeps a weak reference to ``fun`` for use as a compilation cache key,
      so the object ``fun`` must be weakly-referenceable. Most :class:`Callable`
      objects will already satisfy this requirement.
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

      If neither ``static_argnums`` nor ``static_argnames`` is provided, no
      arguments are treated as static. If ``static_argnums`` is not provided but
      ``static_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``static_argnames``
      (or vice versa). If both ``static_argnums`` and ``static_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``static_argnums`` or ``static_argnames`` will
      be treated as static.
    static_argnames: An optional string or collection of strings specifying
      which named arguments to treat as static (compile-time constant). See the
      comment on ``static_argnums`` for details. If not
      provided but ``static_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    device: This is an experimental feature and the API is likely to change.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer
      need them once the computation has finished. In some cases XLA can make
      use of donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.
      Note that donate_argnums only work for positional arguments, and keyword
      arguments will not be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.

    inline: Specify whether this function should be inlined into enclosing
      jaxprs (rather than being represented as an application of the xla_call
      primitive with its own subjaxpr). Default False.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation.

  Examples:
    In the following example, ``selu`` can be compiled into a single fused kernel
    by XLA:

    >>> import jax
    >>>
    >>> @jax.jit
    ... def selu(x, alpha=1.67, lmbda=1.05):
    ...   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
    >>>
    >>> key = jax.random.PRNGKey(0)
    >>> x = jax.random.normal(key, (10,))
    >>> print(selu(x))  # doctest: +SKIP
    [-0.54485  0.27744 -0.29255 -0.91421 -0.62452 -0.24748
    -0.85743 -0.78232  0.76827  0.59566 ]

    To pass arguments such as ``static_argnames`` when decorating a function, a common
    pattern is to use :func:`functools.partial`:

    >>> from functools import partial
    >>>
    >>> @partial(jax.jit, static_argnames=['n'])
    ... def g(x, n):
    ...   for i in range(n):
    ...     x = x ** 2
    ...   return x
    >>>
    >>> g(jnp.arange(4), 3)
    Array([   0,    1,  256, 6561], dtype=int32)
  """
  if abstracted_axes and not config.jax_dynamic_shapes:
    raise ValueError("abstracted_axes must be used with --jax_dynamic_shapes")
  if FLAGS.experimental_cpp_jit and not config.jax_dynamic_shapes:
    return _jit(True, fun, static_argnums, static_argnames, device, backend,
                    donate_argnums, inline, keep_unused)
  return _jit(False, fun, static_argnums, static_argnames, device, backend,
                      donate_argnums, inline, keep_unused, abstracted_axes)


if jax.config.jax_jit_pjit_api_merge:
  def jit(  # type: ignore  # noqa: F811  # pylint: disable=function-redefined
    fun: Callable,
    in_shardings=pxla._UNSPECIFIED,
    out_shardings=pxla._UNSPECIFIED,
    static_argnums: Union[int, Sequence[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    donate_argnums: Union[int, Sequence[int]] = (),
    keep_unused: bool = False,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    inline: bool = False,
    abstracted_axes: Optional[Any] = None,
  ) -> stages.Wrapped:
    """Sets up ``fun`` for just-in-time compilation with XLA.

    Args:
      fun: Function to be jitted. ``fun`` should be a pure function, as
        side-effects may only be executed once.

        The arguments and return value of ``fun`` should be arrays,
        scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
        Positional arguments indicated by ``static_argnums`` can be anything at
        all, provided they are hashable and have an equality operation defined.
        Static arguments are included as part of a compilation cache key, which is
        why hash and equality operators must be defined.

        JAX keeps a weak reference to ``fun`` for use as a compilation cache key,
        so the object ``fun`` must be weakly-referenceable. Most :class:`Callable`
        objects will already satisfy this requirement.
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
        resources assigned to it. This is similar to pjit's in_shardings.
      out_shardings: Like ``in_shardings``, but specifies resource
        assignment for function outputs. This is similar to pjit's
        out_shardings.
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

        If neither ``static_argnums`` nor ``static_argnames`` is provided, no
        arguments are treated as static. If ``static_argnums`` is not provided but
        ``static_argnames`` is, or vice versa, JAX uses
        :code:`inspect.signature(fun)` to find any positional arguments that
        correspond to ``static_argnames``
        (or vice versa). If both ``static_argnums`` and ``static_argnames`` are
        provided, ``inspect.signature`` is not used, and only actual
        parameters listed in either ``static_argnums`` or ``static_argnames`` will
        be treated as static.
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
        Note that donate_argnums only work for positional arguments, and keyword
        arguments will not be donated.

        For more details on buffer donation see the
        `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
      keep_unused: If `False` (the default), arguments that JAX determines to be
        unused by `fun` *may* be dropped from resulting compiled XLA executables.
        Such arguments will not be transferred to the device nor provided to the
        underlying executable. If `True`, unused arguments will not be pruned.
      device: This is an experimental feature and the API is likely to change.
        Optional, the Device the jitted function will run on. (Available devices
        can be retrieved via :py:func:`jax.devices`.) The default is inherited
        from XLA's DeviceAssignment logic and is usually to use
        ``jax.devices()[0]``.
      backend: This is an experimental feature and the API is likely to change.
        Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
        ``'tpu'``.
      inline: Specify whether this function should be inlined into enclosing
        jaxprs (rather than being represented as an application of the xla_call
        primitive with its own subjaxpr). Default False.

    Returns:
      A wrapped version of ``fun``, set up for just-in-time compilation.

    Examples:
      In the following example, ``selu`` can be compiled into a single fused kernel
      by XLA:

      >>> import jax
      >>>
      >>> @jax.jit
      ... def selu(x, alpha=1.67, lmbda=1.05):
      ...   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
      >>>
      >>> key = jax.random.PRNGKey(0)
      >>> x = jax.random.normal(key, (10,))
      >>> print(selu(x))  # doctest: +SKIP
      [-0.54485  0.27744 -0.29255 -0.91421 -0.62452 -0.24748
      -0.85743 -0.78232  0.76827  0.59566 ]

      To pass arguments such as ``static_argnames`` when decorating a function, a common
      pattern is to use :func:`functools.partial`:

      >>> from functools import partial
      >>>
      >>> @partial(jax.jit, static_argnames=['n'])
      ... def g(x, n):
      ...   for i in range(n):
      ...     x = x ** 2
      ...   return x
      >>>
      >>> g(jnp.arange(4), 3)
      Array([   0,    1,  256, 6561], dtype=int32)
    """
    (in_shardings, out_shardings, donate_argnums, static_argnums,
     static_argnames) = pjit.pre_infer_params(
         fun, in_shardings, out_shardings, donate_argnums,
         static_argnums, static_argnames, device, backend, abstracted_axes)

    def infer_params(*args, **kwargs):
      pjit_info_args = pjit.PjitInfo(
          fun=fun, in_shardings=in_shardings,
          out_shardings=out_shardings, static_argnums=static_argnums,
          static_argnames=static_argnames, donate_argnums=donate_argnums,
          device=device, backend=backend, keep_unused=keep_unused,
          inline=inline, resource_env=None)
      return pjit.common_infer_params(pjit_info_args, *args, **kwargs)

    has_explicit_sharding = pjit._pjit_explicit_sharding(
        in_shardings, out_shardings, device, backend)
    return pjit.post_infer_params(fun, infer_params, static_argnums,
                                  static_argnames, donate_argnums,
                                  abstracted_axes, has_explicit_sharding)


def _jit(
    use_cpp_jit: bool,
    fun: Callable,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
  ) -> stages.Wrapped:
  # Implemements common logic between CPP and Python backends
  check_callable(fun)

  donate_argnums, static_argnums, static_argnames = resolve_argnums(
      fun, donate_argnums, static_argnums, static_argnames)

  if use_cpp_jit:
    return _cpp_jit(
        fun, static_argnums=static_argnums, static_argnames=static_argnames,
        device=device, backend=backend, donate_argnums=donate_argnums,
        inline=inline, keep_unused=keep_unused)

  return _python_jit(
      fun, static_argnums=static_argnums, static_argnames=static_argnames,
      device=device, backend=backend, donate_argnums=donate_argnums,
      inline=inline, keep_unused=keep_unused, abstracted_axes=abstracted_axes)

def _prepare_jit(fun, static_argnums, static_argnames, donate_argnums,
                 args, kwargs):
  # Validate donate_argnums
  if max(donate_argnums, default=-1) >= len(args):
    raise ValueError(
        f"jitted function has {donate_argnums=} but "
        f"was called with only {len(args)} positional arguments.")

  f = lu.wrap_init(fun)
  f, args = argnums_partial_except(f, static_argnums, args, allow_invalid=True)
  f, kwargs = argnames_partial_except(f, static_argnames, kwargs)
  args_flat, in_tree = tree_flatten((args, kwargs))
  # Argument donation is incompatible with jax_debug_nans because it re-uses
  # donated buffers when rerunning the user's function.
  if donate_argnums and not config.jax_debug_nans:
    donated_invars = donation_vector(donate_argnums, args, kwargs)
  else:
    donated_invars = (False,) * len(args_flat)

  return f, in_tree, args_flat, donated_invars


PytreeOfAbstractedAxesSpec = Any

def _python_jit(
    fun: Callable,
    *,
    static_argnums: Tuple[int, ...],
    static_argnames: Tuple[str, ...],
    device: Optional[xc.Device],
    backend: Optional[str],
    donate_argnums: Tuple[int, ...],
    inline: bool,
    keep_unused: bool,
    abstracted_axes: Optional[PytreeOfAbstractedAxesSpec],
  ) -> stages.Wrapped:
  @wraps(fun)
  @api_boundary
  def f_jitted(*args, **kwargs):
    if config.jax_disable_jit:
      return fun(*args, **kwargs)
    closed_fun, in_tree, args_flat, donated_invars = _prepare_jit(
        fun, static_argnums, static_argnames, donate_argnums, args, kwargs)
    flat_fun, out_tree = flatten_fun(closed_fun, in_tree)
    for arg in args_flat:
      dispatch.check_arg(arg)
    if jax.config.jax_dynamic_shapes:
      axes_specs = (None if abstracted_axes is None else
                    _flat_axes_specs(abstracted_axes, *args, **kwargs))
      in_type = pe.infer_lambda_input_type(axes_specs, args_flat)
      flat_fun = lu.annotate(flat_fun, in_type)
    out_flat = xla.xla_call(
        flat_fun, *args_flat,
        device=device, backend=backend, name=flat_fun.__name__,
        donated_invars=donated_invars, inline=inline,
        keep_unused=keep_unused)
    return tree_unflatten(out_tree(), out_flat)

  f_jitted.lower = _jit_lower(fun, static_argnums, static_argnames, device,
                              backend, donate_argnums, inline, keep_unused,
                              abstracted_axes)

  def clear_cache():
    dispatch.xla_callable.evict_function(fun)
  f_jitted.clear_cache = clear_cache

  return f_jitted

def _flat_axes_specs(abstracted_axes, *args, **kwargs
                     ) -> List[pe.AbstractedAxesSpec]:
  if kwargs: raise NotImplementedError
  def ax_leaf(l):
    return (isinstance(l, dict) and all_leaves(l.values()) or
            isinstance(l, tuple) and all_leaves(l, lambda x: x is None))
  return broadcast_prefix(abstracted_axes, args, ax_leaf)


class _BackendAndDeviceInfo(NamedTuple):
  default_device: xc.Device
  committed_to_device: bool

class _FastpathData(NamedTuple):
  xla_executable: xc.LoadedExecutable
  out_pytree_def: Any
  sticky_device: Optional[xc.Device]
  avals: Iterable[Any]
  lazy_exprs: Iterable[Any]
  kept_var_bitvec: Iterable[bool]
  shardings: Iterable[Any]
  committed: Iterable[bool]

_cpp_jit_cache = jax_jit.CompiledFunctionCache()


def _cpp_jit_clear_cache(self):
  self._clear_cache()
  dispatch.xla_callable.evict_function(self._fun)

def _jax_array_use_fast_path(execute, out_pytree_def, args_flat, out_flat):
  use_fastpath = (
      # This is if we have already executed this code-path (most-recent entry
      # has been reset to None). Thus, we do not support the fast-path.
      execute is not None and
      type(execute) is pxla.ExecuteReplicated and
      len(execute._local_devices) == 1 and
      # No effects in computation
      not execute.ordered_effects and
      not execute.has_unordered_effects and
      not execute.has_host_callbacks and
      all(isinstance(x, xc.ArrayImpl) for x in out_flat) and
      # Not supported: dynamic shapes
      not jax.config.jax_dynamic_shapes
      # TODO(chky): Check sharding is SingleDeviceSharding
  )

  if use_fastpath:
    sticky_device = None
    lazy_exprs = [None] * len(out_flat)
    kept_var_bitvec = [i in execute.kept_var_idx for i in range(len(args_flat))]
    avals = [out.aval for out in out_flat]
    shardings = [out.sharding for out in out_flat]
    committed = [out._committed for out in out_flat]

    return _FastpathData(execute.xla_executable, out_pytree_def, sticky_device,
                         avals, lazy_exprs, kept_var_bitvec, shardings,
                         committed)

  return None

def _device_array_use_fast_path(execute, out_pytree_def, args_flat, out_flat):
  # TODO(sharadmv): Clean up usage of `execute.args`
  use_fastpath = (
      # This is if we have already executed this code-path (most-recent entry
      # has been reset to None). Thus, we do not support the fast-path.
      execute is not None and
      execute.func is dispatch._execute_compiled and  # not trivial, not pmap
      # No effects in computation
      not execute.args[5] and not execute.args[6] and
      # Has no host callbacks
      not execute.args[8] and
      # impl rule must have been called, i.e. top trace is an EvalTrace
      isinstance(core.find_top_trace(args_flat), core.EvalTrace) and
      # Not supported: ShardedDeviceArray
      all(device_array.type_is_device_array(x) for x in out_flat) and
      # Not supported: dynamic shapes
      not jax.config.jax_dynamic_shapes
      and type(execute.args[4]) is dispatch.SimpleResultHandler)

  ### If we can use the fastpath, we return required info to the caller.
  if use_fastpath:
    (_, xla_executable, _, _, result_handlers, _, _, kept_var_idx,
     _) = execute.args  # pytype: disable=attribute-error
    sticky_device = None
    avals = []
    lazy_exprs = [None] * len(result_handlers)
    for result_handler in result_handlers:
      aval, sticky_device = result_handler.args
      avals.append(aval)
    assert len(avals) == len(out_flat)
    kept_var_bitvec = [i in kept_var_idx for i in range(len(args_flat))]
    shardings = []
    committed = []

    return _FastpathData(xla_executable, out_pytree_def, sticky_device, avals,
                         lazy_exprs, kept_var_bitvec, shardings, committed)

  return None

def _cpp_jit(
    fun: Callable,
    *,
    static_argnums: Tuple[int, ...],
    static_argnames: Tuple[str, ...],
    device: Optional[xc.Device],
    backend: Optional[str],
    donate_argnums: Tuple[int, ...],
    inline: bool,
    keep_unused: bool,
  ) -> stages.Wrapped:
  # An implementation of `jit` that tries to do as much as possible in C++.
  # The goal of this function is to speed up the time it takes to process the
  # arguments, find the correct C++ executable, start the transfer of arguments
  # and schedule the computation.
  # As long as it does not support all features of the Python implementation
  # the C++ code will fallback to `_python_jit` when it faces some unsupported
  # feature.
  if device is not None and backend is not None:
    raise ValueError("can't specify both a device and a backend for jit, "
                     f"got {device=} and {backend=}.")

  @api_boundary
  def cache_miss(*args, **kwargs):
    ### This first part is basically the same code as in _python_jit.
    # An alternative would be for cache_miss to accept from C++ the arguments
    # (dyn_args, donated_invars, args_flat, in_tree), since otherwise we have
    # work/code that is redundant between C++ and Python. We can try that later.
    closed_fun, in_tree, args_flat, donated_invars = _prepare_jit(
        fun, static_argnums, static_argnames, donate_argnums, args, kwargs)
    for arg in args_flat:
      dispatch.check_arg(arg)
    flat_fun, out_tree = flatten_fun(closed_fun, in_tree)
    if jax.config.jax_dynamic_shapes:
      in_type = pe.infer_lambda_input_type(None, args_flat)
      flat_fun = lu.annotate(flat_fun, in_type)

    primitive = xla.xla_call_p
    call_bind_continuation, top_trace, fun_, tracers, params = (
        core.call_bind_with_continuation(primitive, flat_fun, *args_flat,
        device=device,
        backend=backend,
        name=flat_fun.__name__,
        donated_invars=donated_invars,
        inline=inline,
        keep_unused=keep_unused))
    execute = None
    try:
      if isinstance(top_trace, core.EvalTrace) and not (
          jax.config.jax_debug_nans or jax.config.jax_debug_infs):
        execute = dispatch._xla_call_impl_lazy(fun_, *tracers, **params)
        out_flat = call_bind_continuation(execute(*args_flat))
      else:
        out_flat = call_bind_continuation(
            top_trace.process_call(primitive, fun_, tracers, params))
    except pxla.DeviceAssignmentMismatchError as e:
      fails, = e.args
      msg = pjit._device_assignment_mismatch_error(
          fun, fails, in_tree, args_flat, 'jit')
      raise ValueError(msg) from None
    out_pytree_def = out_tree()
    out = tree_unflatten(out_pytree_def, out_flat)

    ### Decide whether we can support the C++ fast path
    # High level note: The Python tracing mechanism is complex; in particular
    # to know whether `jax.jit(f)(x)` will execute or trace, it's not enough to
    # inspect the argument x, we actually do need to execute it and look at the
    # outputs that could be tracers (if f is capturing `Tracer` by closure).

    fastpath_data = None

    # TODO(sharadmv): Enable fast path for effectful jaxprs
    if jax.config.jax_array:
      fastpath_data = _jax_array_use_fast_path(execute, out_pytree_def, args_flat, out_flat)
    else:
      fastpath_data = _device_array_use_fast_path(execute, out_pytree_def, args_flat, out_flat)

    return out, fastpath_data

  def get_device_info():
    """Backends do not exist before __main__ is being executed."""
    committed_to_device = device is not None or backend is not None

    if device is not None:
      default_device = device
    else:
      backend_ = xb.get_backend(backend)
      default_device = backend_.get_default_device_assignment(1)[0]

    return _BackendAndDeviceInfo(default_device, committed_to_device)

  jitted_f_kwargs = {}
  jitted_f_kwargs["has_explicit_device"] = (
      device is not None or backend is not None)
  cpp_jitted_f = jax_jit.jit(
      fun,
      cache_miss,
      get_device_info,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      donate_argnums=donate_argnums,
      cache=_cpp_jit_cache,
      **jitted_f_kwargs)  # type: ignore
  f_jitted = wraps(fun)(cpp_jitted_f)

  f_jitted.lower = _jit_lower(fun, static_argnums, static_argnames, device,
                              backend, donate_argnums, inline, keep_unused,
                              None)
  f_jitted._fun = fun
  type(f_jitted).clear_cache = _cpp_jit_clear_cache

  return f_jitted


def _jit_lower(fun, static_argnums, static_argnames, device, backend,
               donate_argnums, inline,  keep_unused: bool,
               abstracted_axes: Optional[PytreeOfAbstractedAxesSpec]):
  """Make a ``lower`` method for jitted functions."""
  # If the function we returned from ``jit`` were a class instance,
  # this might naturally be a method, with ``fun`` as a ``self`` and
  # all the other arguments stored as attributes.

  def arg_spec(x):
    # like xla.arg_spec but duck-types on x.shape and x.dtype
    aval = None if jax.config.jax_dynamic_shapes else shaped_abstractify(x)
    if jax.config.jax_array:
      if hasattr(x, 'sharding'):
        if isinstance(x.sharding, PmapSharding):
          return aval, None
        # If `x` has a sharding attribute but not `_committed` attribute,
        # assume that `x` is committed. This might happen when the input is
        # a `ShapedDtypeStruct` or `types.SimpleNamespace`, etc that might
        # only have a `sharding` attribute on them.
        return aval, (pjit.to_gspmd_sharding(x.sharding, x.ndim)
                      if getattr(x, '_committed', True) else None)
      else:
        return aval, None
    else:
      device = getattr(x, '_device', None)
      return aval, device

  @api_boundary
  def lower(*args, **kwargs) -> stages.Lowered:
    """Lower this function for the given arguments.

    A lowered function is staged out of Python and translated to a
    compiler's input language, possibly in a backend-dependent
    manner. It is ready for compilation but not yet compiled.

    Returns:
      A ``Lowered`` instance representing the lowering.
    """
    closed_fun, in_tree, args_flat, donated_invars = _prepare_jit(
        fun, static_argnums, static_argnames, donate_argnums, args, kwargs)
    flat_fun, out_tree = flatten_fun(closed_fun, in_tree)
    arg_specs_and_devices = map(arg_spec, args_flat)
    in_avals: Sequence[core.AbstractValue]
    if jax.config.jax_dynamic_shapes:
      axes_specs = (None if abstracted_axes is None else
                    _flat_axes_specs(abstracted_axes, *args, **kwargs))
      in_type = pe.infer_lambda_input_type(axes_specs, args_flat)
      flat_fun = lu.annotate(flat_fun, in_type)
      in_avals = [aval for aval, explicit in in_type if explicit]
    else:
      if abstracted_axes:
        raise ValueError("abstracted_axes must be used with --jax_dynamic_shapes")
      in_avals, _ = unzip2(arg_specs_and_devices)
      if any(not core.is_constant_shape(a.shape) for a in in_avals):
        # TODO(b/262808613): Do not drop unused inputs when we have
        # shape polymorphism, to ensure that we can always derive
        # the dimension variables from the kept inputs.
        nonlocal keep_unused
        keep_unused = True
    if jax.config.jax_array:
      computation = dispatch.sharded_lowering(
          flat_fun, device, backend, flat_fun.__name__, donated_invars, True,
          keep_unused, *arg_specs_and_devices)
      return stages.Lowered.from_flat_info(
          computation, in_tree, in_avals, donate_argnums, out_tree())
    else:
      computation = dispatch.lower_xla_callable(
          flat_fun, device, backend, flat_fun.__name__, donated_invars, True,
          keep_unused, *arg_specs_and_devices)
      return stages.Lowered.from_flat_info(
          computation, in_tree, in_avals, donate_argnums, out_tree())

  return lower


@contextmanager
def disable_jit(disable: bool = True):
  """Context manager that disables :py:func:`jit` behavior under its dynamic context.

  For debugging it is useful to have a mechanism that disables :py:func:`jit`
  everywhere in a dynamic context. Note that this not only disables explicit
  uses of :func:`jit` by the user, but will also remove any implicit JIT compilation
  used by the JAX library: this includes implicit JIT computation of `body` and
  `cond` functions passed to higher-level primitives like :func:`~jax.lax.scan` and
  :func:`~jax.lax.while_loop`, JIT used in implementations of :mod:`jax.numpy` functions,
  and any other case where :func:`jit` is used within an API's implementation.

  Values that have a data dependence on the arguments to a jitted function are
  traced and abstracted. For example, an abstract value may be a
  :py:class:`ShapedArray` instance, representing the set of all possible arrays
  with a given shape and dtype, but not representing one concrete array with
  specific values. You might notice those if you use a benign side-effecting
  operation in a jitted function, like a print:

  >>> import jax
  >>>
  >>> @jax.jit
  ... def f(x):
  ...   y = x * 2
  ...   print("Value of y is", y)
  ...   return y + 3
  ...
  >>> print(f(jax.numpy.array([1, 2, 3])))  # doctest:+ELLIPSIS
  Value of y is Traced<ShapedArray(int32[3])>with<DynamicJaxprTrace...>
  [5 7 9]

  Here ``y`` has been abstracted by :py:func:`jit` to a :py:class:`ShapedArray`,
  which represents an array with a fixed shape and type but an arbitrary value.
  The value of ``y`` is also traced. If we want to see a concrete value while
  debugging, and avoid the tracer too, we can use the :py:func:`disable_jit`
  context manager:

  >>> import jax
  >>>
  >>> with jax.disable_jit():
  ...   print(f(jax.numpy.array([1, 2, 3])))
  ...
  Value of y is [2 4 6]
  [5 7 9]
  """
  with _disable_jit(disable):
    yield


def xla_computation(fun: Callable,
                    static_argnums: Union[int, Iterable[int]] = (),
                    axis_env: Optional[Sequence[Tuple[AxisName, int]]] = None,
                    in_parts=None, out_parts=None,
                    backend: Optional[str] = None,
                    tuple_args: bool = False,
                    instantiate_const_outputs: Optional[bool] = None,
                    return_shape: bool = False,
                    donate_argnums: Union[int, Iterable[int]] = ()) -> Callable:
  """Creates a function that produces its XLA computation given example args.

  Args:
    fun: Function from which to form XLA computations.
    static_argnums: See the :py:func:`jax.jit` docstring.
    axis_env: Optional, a sequence of pairs where the first element is an axis
      name and the second element is a positive integer representing the size of
      the mapped axis with that name. This parameter is useful when lowering
      functions that involve parallel communication collectives, and it
      specifies the axis name/size environment that would be set up by
      applications of :py:func:`jax.pmap`. See the examples below.
    in_parts: Optional, how each argument to ``fun`` should be partitioned or
      replicated. This is used to specify partitioned XLA computations, see
      ``sharded_jit`` for more info.
    out_parts: Optional, how each output of ``fun`` should be partitioned or
      replicated. This is used to specify partitioned XLA computations, see
      ``sharded_jit`` for more info.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    tuple_args: Optional bool, defaults to ``False``. If ``True``, the resulting
      XLA computation will have a single tuple argument that is unpacked into
      the specified function arguments. If `None`, tupling will be enabled when
      there are more than 100 arguments, since some platforms have limits on
      argument arity.
    instantiate_const_outputs: Deprecated argument, does nothing.
    return_shape: Optional boolean, defaults to ``False``. If ``True``, the
      wrapped function returns a pair where the first element is the XLA
      computation and the second element is a pytree with the same structure as
      the output of ``fun`` and where the leaves are objects with ``shape``,
      ``dtype``, and ``named_shape`` attributes representing the corresponding
      types of the output leaves.
    donate_argnums: Specify which arguments are "donated" to the computation.
      It is safe to donate arguments if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to.

  Returns:
    A wrapped version of ``fun`` that when applied to example arguments returns
    a built XLA Computation (see xla_client.py), from which representations of
    the unoptimized XLA HLO computation can be extracted using methods like
    ``as_hlo_text``, ``as_serialized_hlo_module_proto``, and
    ``as_hlo_dot_graph``. If the argument ``return_shape`` is ``True``, then the
    wrapped function returns a pair where the first element is the XLA
    Computation and the second element is a pytree representing the structure,
    shapes, dtypes, and named shapes of the output of ``fun``.

    Concrete example arguments are not always necessary. For those arguments not
    indicated by ``static_argnums``, any object with ``shape`` and ``dtype``
    attributes is acceptable (excepting namedtuples, which are treated as Python
    containers).

  For example:

  >>> import jax
  >>>
  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> c = jax.xla_computation(f)(3.)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule xla_computation_f.6
  <BLANKLINE>
  ENTRY xla_computation_f.6 {
    constant.2 = pred[] constant(false)
    parameter.1 = f32[] parameter(0)
    cosine.3 = f32[] cosine(parameter.1)
    sine.4 = f32[] sine(cosine.3)
    ROOT tuple.5 = (f32[]) tuple(sine.4)
  }
  <BLANKLINE>
  <BLANKLINE>


  Alternatively, the assignment to ``c`` above could be written:

  >>> import types
  >>> scalar = types.SimpleNamespace(shape=(), dtype=np.dtype(np.float32))
  >>> c = jax.xla_computation(f)(scalar)


  Here's an example that involves a parallel collective and axis name:

  >>> def f(x): return x - jax.lax.psum(x, 'i')
  >>> c = jax.xla_computation(f, axis_env=[('i', 4)])(2)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule jaxpr_computation.9
  primitive_computation.3 {
    parameter.4 = s32[] parameter(0)
    parameter.5 = s32[] parameter(1)
    ROOT add.6 = s32[] add(parameter.4, parameter.5)
  }
  ENTRY jaxpr_computation.9 {
    tuple.1 = () tuple()
    parameter.2 = s32[] parameter(0)
    all-reduce.7 = s32[] all-reduce(parameter.2), replica_groups={{0,1,2,3}}, to_apply=primitive_computation.3
    ROOT subtract.8 = s32[] subtract(parameter.2, all-reduce.7)
  }
  <BLANKLINE>
  <BLANKLINE>

  Notice the ``replica_groups`` that were generated. Here's an example that
  generates more interesting ``replica_groups``:

  >>> from jax import lax
  >>> def g(x):
  ...   rowsum = lax.psum(x, 'i')
  ...   colsum = lax.psum(x, 'j')
  ...   allsum = lax.psum(x, ('i', 'j'))
  ...   return rowsum, colsum, allsum
  ...
  >>> axis_env = [('i', 4), ('j', 2)]
  >>> c = xla_computation(g, axis_env=axis_env)(5.)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule jaxpr_computation__1.19
  [removed uninteresting text here]
  ENTRY jaxpr_computation__1.19 {
    tuple.1 = () tuple()
    parameter.2 = f32[] parameter(0)
    all-reduce.7 = f32[] all-reduce(parameter.2), replica_groups={{0,2,4,6},{1,3,5,7}}, to_apply=primitive_computation__1.3
    all-reduce.12 = f32[] all-reduce(parameter.2), replica_groups={{0,1},{2,3},{4,5},{6,7}}, to_apply=primitive_computation__1.8
    all-reduce.17 = f32[] all-reduce(parameter.2), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=primitive_computation__1.13
    ROOT tuple.18 = (f32[], f32[], f32[]) tuple(all-reduce.7, all-reduce.12, all-reduce.17)
  }
  """
  del instantiate_const_outputs  # Unused

  check_callable(fun)
  static_argnums = _ensure_index_tuple(static_argnums)
  donate_argnums = _ensure_index_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  fun_name = getattr(fun, "__name__", "unknown")

  platform = backend if backend is not None else xb.get_backend().platform

  def make_axis_env(nreps):
    if axis_env is None:
      return xla.AxisEnv(nreps, (), ())
    else:
      nreps = nreps * prod(size for name, size in axis_env)
      names, sizes = unzip2(axis_env)
      return xla.AxisEnv(nreps, names, sizes)

  @wraps(fun)
  @api_boundary
  def computation_maker(*args, **kwargs):
    if max(static_argnums + donate_argnums, default=-1) >= len(args):
      raise ValueError(f"jitted function has {static_argnums=}, {donate_argnums=} but "
                       f"was called with only {len(args)} positional arguments.")

    f = lu.wrap_init(fun)
    f, dyn_args = argnums_partial_except(f, static_argnums, args, allow_invalid=False)
    args_flat, in_tree = tree_flatten((dyn_args, kwargs))
    if donate_argnums:
      donated_invars = donation_vector(donate_argnums, dyn_args, kwargs)
    else:
      donated_invars = (False,) * len(args_flat)

    if in_parts is None:
      in_parts_flat = None
    else:
      in_parts_flat = tuple(flatten_axes(
          "xla_computation in_parts", in_tree.children()[0], in_parts))
    jaxtree_fun, out_tree = flatten_fun(f, in_tree)
    avals = map(shaped_abstractify, args_flat)
    with ExitStack() as stack:
      for axis_name, size in axis_env or []:
        stack.enter_context(core.extend_axis_env(axis_name, size, None))
      jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(jaxtree_fun, avals)
      jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)
      axis_env_ = make_axis_env(dispatch.jaxpr_replicas(jaxpr))
      if out_parts is None:
        out_parts_flat = None
      else:
        out_parts_flat = tuple(flatten_axes(
            "xla_computation out_parts", out_tree(), out_parts))
      unordered_effects = list(
          effects.ordered_effects.filter_not_in(jaxpr.effects))
      ordered_effects = list(
          effects.ordered_effects.filter_in(jaxpr.effects))
      lowering_result = mlir.lower_jaxpr_to_module(
          f"xla_computation_{fun_name}",
          core.ClosedJaxpr(jaxpr, consts),
          unordered_effects=unordered_effects,
          ordered_effects=ordered_effects,
          backend_or_name=backend,
          platforms=(platform,),
          axis_context=mlir.ReplicaAxisContext(axis_env_),
          name_stack=new_name_stack(wrap_name(fun_name, "xla_computation")),
          donated_args=donated_invars,
          arg_shardings=(None if in_parts_flat is None else map(
              xla.sharding_to_proto, in_parts_flat)),
          result_shardings=(None if out_parts_flat is None else map(
              xla.sharding_to_proto, out_parts_flat)))
      if tuple_args is not None:
        should_tuple = tuple_args
      else:
        dispatch.should_tuple_args(len(avals), backend.platform)
      built = xc._xla.mlir.mlir_module_to_xla_computation(
          mlir.module_to_string(lowering_result.module),
          use_tuple_args=should_tuple,
          return_tuple=True)
    out_shapes_flat = [
        ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
    out_shape = tree_unflatten(out_tree(), out_shapes_flat)
    for out_aval in out_avals:
      if not isinstance(out_aval, ShapedArray):
        raise RuntimeError("As we want to propagate the weak_type, we need "
                           "to get a ShapedArray, otherwise this "
                           "information is lost")

    if return_shape:
      return built, out_shape
    else:
      return built

  return computation_maker

def grad(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
         has_aux: bool = False, holomorphic: bool = False,
         allow_int: bool = False,
         reduce_axes: Sequence[AxisName] = ()) -> Callable:
  """Creates a function that evaluates the gradient of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers.
      Argument arrays in the positions specified by ``argnums`` must be of
      inexact (i.e., floating-point or complex) type. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding gradient. Otherwise, the
      gradient will be per-example over named axes. For example, if ``'batch'``
      is a named batch axis, ``grad(f, reduce_axes=('batch',))`` will create a
      function that computes the total gradient while ``grad(f)`` will create
      one that computes the per-example gradient.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the gradient
    of ``fun``. If ``argnums`` is an integer then the gradient has the same
    shape and type as the positional argument indicated by that integer. If
    argnums is a tuple of integers, the gradient is a tuple of values with the
    same shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a pair of (gradient, auxiliary_data) is returned.

  For example:

  >>> import jax
  >>>
  >>> grad_tanh = jax.grad(jax.numpy.tanh)
  >>> print(grad_tanh(0.2))
  0.961043
  """
  value_and_grad_f = value_and_grad(fun, argnums, has_aux=has_aux,
                                    holomorphic=holomorphic,
                                    allow_int=allow_int,
                                    reduce_axes=reduce_axes)

  docstr = ("Gradient of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "gradient, which has the same shape as the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def grad_f(*args, **kwargs):
    _, g = value_and_grad_f(*args, **kwargs)
    return g

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def grad_f_aux(*args, **kwargs):
    (_, aux), g = value_and_grad_f(*args, **kwargs)
    return g, aux

  return grad_f_aux if has_aux else grad_f

def value_and_grad(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                   has_aux: bool = False, holomorphic: bool = False,
                   allow_int: bool = False, reduce_axes: Sequence[AxisName] = ()
  ) -> Callable[..., Tuple[Any, Any]]:
  """Create a function that evaluates both ``fun`` and the gradient of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding gradient. Otherwise, the
      gradient will be per-example over named axes. For example, if ``'batch'``
      is a named batch axis, ``value_and_grad(f, reduce_axes=('batch',))`` will
      create a function that computes the total gradient while
      ``value_and_grad(f)`` will create one that computes the per-example
      gradient.

  Returns:
    A function with the same arguments as ``fun`` that evaluates both ``fun``
    and the gradient of ``fun`` and returns them as a pair (a two-element
    tuple). If ``argnums`` is an integer then the gradient has the same shape
    and type as the positional argument indicated by that integer. If argnums is
    a sequence of integers, the gradient is a tuple of values with the same
    shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a tuple of ((value, auxiliary_data), gradient) is returned.
  """

  docstr = ("Value and gradient of {fun} with respect to positional "
            "argument(s) {argnums}. Takes the same arguments as {fun} but "
            "returns a two-element tuple where the first element is the value "
            "of {fun} and the second element is the gradient, which has the "
            "same shape as the arguments at positions {argnums}.")

  check_callable(fun)
  argnums = core.concrete_or_error(_ensure_index, argnums)
  reduce_axes = _ensure_str_tuple(reduce_axes)  # type: ignore

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def value_and_grad_f(*args, **kwargs):
    max_argnum = argnums if isinstance(argnums, int) else max(argnums)
    if max_argnum >= len(args):
      raise TypeError(f"differentiating with respect to {argnums=} requires at least "
                      f"{max_argnum + 1} positional arguments to be passed by the caller, "
                      f"but got only {len(args)} positional arguments.")

    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    for leaf in tree_leaves(dyn_args):
      _check_input_dtype_grad(holomorphic, allow_int, leaf)
    if not has_aux:
      ans, vjp_py = _vjp(f_partial, *dyn_args, reduce_axes=reduce_axes)
    else:
      ans, vjp_py, aux = _vjp(
          f_partial, *dyn_args, has_aux=True, reduce_axes=reduce_axes)
    _check_scalar(ans)
    tree_map(partial(_check_output_dtype_grad, holomorphic), ans)
    g = vjp_py(lax_internal._one(ans))
    g = g[0] if isinstance(argnums, int) else g
    if not has_aux:
      return ans, g
    else:
      return (ans, aux), g

  return value_and_grad_f

def _check_scalar(x):
  msg = "Gradient only defined for scalar-output functions. Output {}.".format
  try:
    aval = core.get_aval(x)
  except TypeError as e:
    raise TypeError(msg(f"was {x}")) from e
  else:
    if isinstance(aval, ShapedArray):
      if aval.shape != ():
        raise TypeError(msg(f"had shape: {aval.shape}"))
    else:
      raise TypeError(msg(f"had abstract value {aval}"))

def _check_input_dtype_revderiv(name, holomorphic, allow_int, x):
  dispatch.check_arg(x)
  aval = core.get_aval(x)
  if core.is_opaque_dtype(aval.dtype):
    raise TypeError(
        f"{name} with input element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError(f"{name} with holomorphic=True requires inputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  if (dtypes.issubdtype(aval.dtype, np.integer) or
      dtypes.issubdtype(aval.dtype, np.bool_)):
    if not allow_int:
      raise TypeError(f"{name} requires real- or complex-valued inputs (input dtype "
                      f"that is a sub-dtype of np.inexact), but got {aval.dtype.name}. "
                      "If you want to use Boolean- or integer-valued inputs, use vjp "
                      "or set allow_int to True.")
  elif not dtypes.issubdtype(aval.dtype, np.inexact):
    raise TypeError(f"{name} requires numerical-valued inputs (input dtype that is a "
                    f"sub-dtype of np.bool_ or np.number), but got {aval.dtype.name}.")
_check_input_dtype_grad = partial(_check_input_dtype_revderiv, "grad")

def _check_output_dtype_revderiv(name, holomorphic, x):
  aval = core.get_aval(x)
  if core.is_opaque_dtype(aval.dtype):
    raise TypeError(
        f"{name} with output element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError(f"{name} with holomorphic=True requires outputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  elif dtypes.issubdtype(aval.dtype, np.complexfloating):
    raise TypeError(f"{name} requires real-valued outputs (output dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For holomorphic differentiation, pass holomorphic=True. "
                    "For differentiation of non-holomorphic functions involving complex "
                    "outputs, use jax.vjp directly.")
  elif not dtypes.issubdtype(aval.dtype, np.floating):
    raise TypeError(f"{name} requires real-valued outputs (output dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For differentiation of functions with integer outputs, use "
                    "jax.vjp directly.")
_check_output_dtype_grad = partial(_check_output_dtype_revderiv, "grad")


def jacfwd(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           has_aux: bool = False, holomorphic: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using forward-mode automatic differentiation. If ``has_aux`` is True
    then a pair of (jacobian, auxiliary_data) is returned.

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  check_callable(fun)
  argnums = _ensure_index(argnums)

  docstr = ("Jacobian of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "jacobian of the output with respect to the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    if not has_aux:
      pushfwd = partial(_jvp, f_partial, dyn_args)
      y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    else:
      pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
      y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
    if not has_aux:
      return jac_tree
    else:
      return jac_tree, aux

  return jacfun

def _check_input_dtype_jacfwd(holomorphic: bool, x: Any) -> None:
  dispatch.check_arg(x)
  aval = core.get_aval(x)
  if core.is_opaque_dtype(aval.dtype):
    raise TypeError(
        f"jacfwd with input element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError("jacfwd with holomorphic=True requires inputs with complex "
                      f"dtype, but got {aval.dtype.name}.")
  elif not dtypes.issubdtype(aval.dtype, np.floating):
    raise TypeError("jacfwd requires real-valued inputs (input dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For holomorphic differentiation, pass holomorphic=True. "
                    "For differentiation of non-holomorphic functions involving "
                    "complex inputs or integer inputs, use jax.jvp directly.")

def _check_output_dtype_jacfwd(holomorphic, x):
  aval = core.get_aval(x)
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError("jacfwd with holomorphic=True requires outputs with complex dtype, "
                      f"but got {aval.dtype.name}.")

def jacrev(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
    then a pair of (jacobian, auxiliary_data) is returned.

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacrev(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  check_callable(fun)

  docstr = ("Jacobian of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "jacobian of the output with respect to the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    if not has_aux:
      y, pullback = _vjp(f_partial, *dyn_args)
    else:
      y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
    jac_tree = tree_transpose(tree_structure(example_args), tree_structure(y), jac_tree)
    if not has_aux:
      return jac_tree
    else:
      return jac_tree, aux

  return jacfun
jacobian = jacrev

_check_input_dtype_jacrev = partial(_check_input_dtype_revderiv, "jacrev")
_check_output_dtype_jacrev = partial(_check_output_dtype_revderiv, "jacrev")


def hessian(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
            has_aux: bool = False, holomorphic: bool = False) -> Callable:
  """Hessian of ``fun`` as a dense array.

  Args:
    fun: Function whose Hessian is to be computed.  Its arguments at positions
      specified by ``argnums`` should be arrays, scalars, or standard Python
      containers thereof. It should return arrays, scalars, or standard Python
      containers thereof.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Hessian of
    ``fun``.

  >>> import jax
  >>>
  >>> g = lambda x: x[0]**3 - 2*x[0]*x[1] - x[1]**6
  >>> print(jax.hessian(g)(jax.numpy.array([1., 2.])))
  [[   6.   -2.]
   [  -2. -480.]]

  :py:func:`hessian` is a generalization of the usual definition of the Hessian
  that supports nested Python containers (i.e. pytrees) as inputs and outputs.
  The tree structure of ``jax.hessian(fun)(x)`` is given by forming a tree
  product of the structure of ``fun(x)`` with a tree product of two copies of
  the structure of ``x``. A tree product of two tree structures is formed by
  replacing each leaf of the first tree with a copy of the second. For example:

  >>> import jax.numpy as jnp
  >>> f = lambda dct: {"c": jnp.power(dct["a"], dct["b"])}
  >>> print(jax.hessian(f)({"a": jnp.arange(2.) + 1., "b": jnp.arange(2.) + 2.}))
  {'c': {'a': {'a': Array([[[ 2.,  0.], [ 0.,  0.]],
                           [[ 0.,  0.], [ 0., 12.]]], dtype=float32),
               'b': Array([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                           [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32)},
         'b': {'a': Array([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                           [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32),
               'b': Array([[[0.      , 0.      ], [0.      , 0.      ]],
                           [[0.      , 0.      ], [0.      , 3.843624]]], dtype=float32)}}}

  Thus each leaf in the tree structure of ``jax.hessian(fun)(x)`` corresponds to
  a leaf of ``fun(x)`` and a pair of leaves of ``x``. For each leaf in
  ``jax.hessian(fun)(x)``, if the corresponding array leaf of ``fun(x)`` has
  shape ``(out_1, out_2, ...)`` and the corresponding array leaves of ``x`` have
  shape ``(in_1_1, in_1_2, ...)`` and ``(in_2_1, in_2_2, ...)`` respectively,
  then the Hessian leaf has shape ``(out_1, out_2, ..., in_1_1, in_1_2, ...,
  in_2_1, in_2_2, ...)``. In other words, the Python tree structure represents
  the block structure of the Hessian, with blocks determined by the input and
  output pytrees.

  In particular, an array is produced (with no pytrees involved) when the
  function input ``x`` and output ``fun(x)`` are each a single array, as in the
  ``g`` example above. If ``fun(x)`` has shape ``(out1, out2, ...)`` and ``x``
  has shape ``(in1, in2, ...)`` then ``jax.hessian(fun)(x)`` has shape
  ``(out1, out2, ..., in1, in2, ..., in1, in2, ...)``. To flatten pytrees into
  1D vectors, consider using :py:func:`jax.flatten_util.flatten_pytree`.
  """
  return jacfwd(jacrev(fun, argnums, has_aux=has_aux, holomorphic=holomorphic),
                argnums, has_aux=has_aux, holomorphic=holomorphic)

def _std_basis(pytree):
  leaves, _ = tree_flatten(pytree)
  ndim = sum(map(np.size, leaves))
  dtype = dtypes.result_type(*leaves)
  flat_basis = jax.numpy.eye(ndim, dtype=dtype)
  return _unravel_array_into_pytree(pytree, 1, None, flat_basis)

def _jacfwd_unravel(input_pytree, output_pytree_leaf, arr):
  return _unravel_array_into_pytree(
    input_pytree, -1, output_pytree_leaf, arr)

def _jacrev_unravel(output_pytree, input_pytree_leaf, arr):
  return _unravel_array_into_pytree(
    output_pytree, 0, input_pytree_leaf, arr)

def _possible_downcast(x, example):
  if (dtypes.issubdtype(x.dtype, np.complexfloating) and
      not dtypes.issubdtype(_dtype(example), np.complexfloating)):
    x = x.real
  dtype = None if example is None else _dtype(example)
  weak_type = None if example is None else dtypes.is_weakly_typed(example)
  return lax_internal._convert_element_type(x, dtype, weak_type)

def _unravel_array_into_pytree(pytree, axis, example, arr):
  """Unravel an array into a PyTree with a given structure.
  Args:
      pytree: The pytree that provides the structure.
      axis: The parameter axis is either -1, 0, or 1.  It controls the
        resulting shapes.
      example: If specified, cast the components to the matching dtype/weak_type,
        or else use the pytree leaf type if example is None.
      arr: The array to be unraveled.
  """
  leaves, treedef = tree_flatten(pytree)
  axis = axis % arr.ndim
  shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis+1:] for l in leaves]
  parts = _split(arr, np.cumsum(map(np.size, leaves[:-1])), axis)
  reshaped_parts = [
      _possible_downcast(np.reshape(x, shape), leaf if example is None else example)
      for x, shape, leaf in zip(parts, shapes, leaves)]
  return tree_unflatten(treedef, reshaped_parts)

def _split(x, indices, axis):
  if isinstance(x, np.ndarray):
    return np.split(x, indices, axis)
  else:
    return x.split(indices, axis)


def vmap(fun: F,
         in_axes: Union[int, Sequence[Any]] = 0,
         out_axes: Any = 0,
         axis_name: Optional[AxisName] = None,
         axis_size: Optional[int] = None,
         spmd_axis_name: Optional[Union[AxisName, Tuple[AxisName, ...]]] = None
         ) -> F:
  """Vectorizing map. Creates a function which maps ``fun`` over argument axes.

  Args:
    fun: Function to be mapped over additional axes.
    in_axes: An integer, None, or (nested) standard Python container
      (tuple/list/dict) thereof specifying which input array axes to map over.

      If each positional argument to ``fun`` is an array, then ``in_axes`` can
      be an integer, a None, or a tuple of integers and Nones with length equal
      to the number of positional arguments to ``fun``. An integer or ``None``
      indicates which array axis to map over for all arguments (with ``None``
      indicating not to map any axis), and a tuple indicates which axis to map
      for each corresponding positional argument. Axis integers must be in the
      range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
      dimensions (axes) of the corresponding input array.

      If the positional arguments to ``fun`` are container (pytree) types, the
      corresponding element of ``in_axes`` can itself be a matching container,
      so that distinct array axes can be mapped for different container
      elements. ``in_axes`` must be a container tree prefix of the positional
      argument tuple passed to ``fun``. See this link for more detail:
      https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees

      Either ``axis_size`` must be provided explicitly, or at least one
      positional argument must have ``in_axes`` not None. The sizes of the
      mapped input axes for all mapped positional arguments must all be equal.

      Arguments passed as keywords are always mapped over their leading axis
      (i.e. axis index 0).

      See below for examples.

    out_axes: An integer, None, or (nested) standard Python container
      (tuple/list/dict) thereof indicating where the mapped axis should appear
      in the output. All outputs with a mapped axis must have a non-None
      ``out_axes`` specification. Axis integers must be in the range ``[-ndim,
      ndim)`` for each output array, where ``ndim`` is the number of dimensions
      (axes) of the array returned by the :func:`vmap`-ed function, which is one
      more than the number of dimensions (axes) of the corresponding array
      returned by ``fun``.
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    axis_size: Optional, an integer indicating the size of the axis to be
      mapped. If not provided, the mapped axis size is inferred from arguments.

  Returns:
    Batched/vectorized version of ``fun`` with arguments that correspond to
    those of ``fun``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``fun``, but
    with extra array axes at positions indicated by ``out_axes``.

  For example, we can implement a matrix-matrix product using a vector dot
  product:

  >>> import jax.numpy as jnp
  >>>
  >>> vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
  >>> mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
  >>> mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

  Here we use ``[a,b]`` to indicate an array with shape (a,b). Here are some
  variants:

  >>> mv1 = vmap(vv, (0, 0), 0)   #  ([b,a], [b,a]) -> [b]        (b is the mapped axis)
  >>> mv2 = vmap(vv, (0, 1), 0)   #  ([b,a], [a,b]) -> [b]        (b is the mapped axis)
  >>> mm2 = vmap(mv2, (1, 1), 0)  #  ([b,c,a], [a,c,b]) -> [c,b]  (c is the mapped axis)

  Here's an example of using container types in ``in_axes`` to specify which
  axes of the container elements to map over:

  >>> A, B, C, D = 2, 3, 4, 5
  >>> x = jnp.ones((A, B))
  >>> y = jnp.ones((B, C))
  >>> z = jnp.ones((C, D))
  >>> def foo(tree_arg):
  ...   x, (y, z) = tree_arg
  ...   return jnp.dot(x, jnp.dot(y, z))
  >>> tree = (x, (y, z))
  >>> print(foo(tree))
  [[12. 12. 12. 12. 12.]
   [12. 12. 12. 12. 12.]]
  >>> from jax import vmap
  >>> K = 6  # batch size
  >>> x = jnp.ones((K, A, B))  # batch axis in different locations
  >>> y = jnp.ones((B, K, C))
  >>> z = jnp.ones((C, D, K))
  >>> tree = (x, (y, z))
  >>> vfoo = vmap(foo, in_axes=((0, (1, 2)),))
  >>> print(vfoo(tree).shape)
  (6, 2, 5)

  Here's another example using container types in ``in_axes``, this time a
  dictionary, to specify the elements of the container to map over:

  >>> dct = {'a': 0., 'b': jnp.arange(5.)}
  >>> x = 1.
  >>> def foo(dct, x):
  ...  return dct['a'] + dct['b'] + x
  >>> out = vmap(foo, in_axes=({'a': None, 'b': 0}, None))(dct, x)
  >>> print(out)
  [1. 2. 3. 4. 5.]

  The results of a vectorized function can be mapped or unmapped. For example,
  the function below returns a pair with the first element mapped and the second
  unmapped. Only for unmapped results we can specify ``out_axes`` to be ``None``
  (to keep it unmapped).

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=(0, None))(jnp.arange(2.), 4.))
  (Array([4., 5.], dtype=float32), 8.0)

  If the ``out_axes`` is specified for an unmapped result, the result is
  broadcast across the mapped axis:

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=0)(jnp.arange(2.), 4.))
  (Array([4., 5.], dtype=float32), Array([8., 8.], dtype=float32, weak_type=True))

  If the ``out_axes`` is specified for a mapped result, the result is transposed
  accordingly.

  Finally, here's an example using ``axis_name`` together with collectives:

  >>> xs = jnp.arange(3. * 4.).reshape(3, 4)
  >>> print(vmap(lambda x: lax.psum(x, 'i'), axis_name='i')(xs))
  [[12. 15. 18. 21.]
   [12. 15. 18. 21.]
   [12. 15. 18. 21.]]

  See the :py:func:`jax.pmap` docstring for more examples involving collectives.
  """
  check_callable(fun)
  docstr = ("Vectorized version of {fun}. Takes similar arguments as {fun} "
            "but with additional array axes over which {fun} is mapped.")
  if fun.__doc__:
    docstr += "\n\nOriginal documentation:\n\n"
    docstr += fun.__doc__

  axis_name = core.no_axis_name if axis_name is None else axis_name
  if spmd_axis_name is not None and type(spmd_axis_name) is not tuple:
    spmd_axis_name = (spmd_axis_name,)

  if isinstance(in_axes, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/google/jax/issues/2367
    in_axes = tuple(in_axes)

  if not all(type(l) is int or type(l) in batching.spec_types
             for l in tree_leaves(in_axes)):
    raise TypeError("vmap in_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {in_axes}.")
  if not all(type(l) is int or type(l) in batching.spec_types
               for l in tree_leaves(out_axes)):
    raise TypeError("vmap out_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {out_axes}.")

  @wraps(fun, docstr=docstr)
  @api_boundary
  def vmap_f(*args, **kwargs):
    args_flat, in_tree  = tree_flatten((args, kwargs), is_leaf=batching.is_vmappable)
    f = lu.wrap_init(fun)
    flat_fun, out_tree = batching.flatten_fun_for_vmap(f, in_tree)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, (in_axes, 0), kws=True)
    axis_size_ = (axis_size if axis_size is not None else
                  _mapped_axis_size(fun, in_tree, args_flat, in_axes_flat, "vmap"))
    out_flat = batching.batch(
        flat_fun, axis_name, axis_size_, in_axes_flat,
        lambda: flatten_axes("vmap out_axes", out_tree(), out_axes),
        spmd_axis_name=spmd_axis_name
    ).call_wrapped(*args_flat)
    return tree_unflatten(out_tree(), out_flat)

  return vmap_f

def _mapped_axis_size(fn, tree, vals, dims, name):
  if not vals:
    args, kwargs = tree_unflatten(tree, vals)
    raise ValueError(
        f"{name} wrapped function must be passed at least one argument "
        f"containing an array, got empty *args={args} and **kwargs={kwargs}"
    )

  def _get_axis_size(name: str, shape: Tuple[core.AxisSize, ...], axis: int
                     ) -> core.AxisSize:
    try:
      return shape[axis]
    except (IndexError, TypeError) as e:
      min_rank = axis + 1 if axis >= 0 else -axis
      # TODO(mattjj): better error message here
      raise ValueError(
          f"{name} was requested to map its argument along axis {axis}, "
          f"which implies that its rank should be at least {min_rank}, "
          f"but is only {len(shape)} (its shape is {shape})") from e

  sizes = core.dedup_referents(_get_axis_size(name, np.shape(x), d)
                               for x, d in zip(vals, dims) if d is not None)
  if len(sizes) == 1:
    sz, = sizes
    return sz
  if not sizes:
    msg = f"{name} must have at least one non-None value in in_axes"
    raise ValueError(msg)

  msg = [f"{name} got inconsistent sizes for array axes to be mapped:\n"]
  args, kwargs = tree_unflatten(tree, vals)
  try:
    ba = inspect.signature(fn).bind(*args, **kwargs)
  except (TypeError, ValueError):
    ba = None
  if ba is None:
    args_paths = [f'args{p.pprint("")} '
                  f'of type {shaped_abstractify(x).str_short()}'
                  for p, x in _generate_key_paths(args)]
    kwargs_paths = [f'kwargs{p.pprint("")} '
                    f'of type {shaped_abstractify(x).str_short()}'
                    for p, x in _generate_key_paths(kwargs)]
    key_paths = [*args_paths, *kwargs_paths]
  else:
    key_paths = [f'argument {name}{p.pprint("")} '
                 f'of type {shaped_abstractify(x).str_short()}'
                 for name, arg in ba.arguments.items()
                 for p, x in _generate_key_paths(arg)]
  all_sizes = [_get_axis_size(name, np.shape(x), d) if d is not None else None
               for x, d in zip(vals, dims)]
  size_counts = collections.Counter(s for s in all_sizes if s is not None)
  (sz, ct), *other_counts = counts = size_counts.most_common()
  def _all_sizes_index(sz):
    for i, isz in enumerate(all_sizes):
      if core.symbolic_equal_dim(isz, sz): return i
    assert False, (sz, all_sizes)

  ex, *examples = [key_paths[_all_sizes_index(sz)] for sz, _ in counts]
  ax, *axs = [dims[_all_sizes_index(sz)] for sz, _ in counts]
  if ct == 1:
    msg.append(f"  * one axis had size {sz}: axis {ax} of {ex};\n")
  else:
    msg.append(f"  * most axes ({ct} of them) had size {sz}, e.g. axis {ax} of {ex};\n")
  for ex, ax, (sz, ct) in zip(examples, axs, other_counts):
    if ct == 1:
      msg.append(f"  * one axis had size {sz}: axis {ax} of {ex};\n")
    else:
      msg.append(f"  * some axes ({ct} of them) had size {sz}, e.g. axis {ax} of {ex};\n")
  raise ValueError(''.join(msg)[:-2])  # remove last semicolon and newline


def pmap(
    fun: Callable,
    axis_name: Optional[AxisName] = None,
    *,
    in_axes=0,
    out_axes=0,
    static_broadcasted_argnums: Union[int, Iterable[int]] = (),
    devices: Optional[Sequence[xc.Device]] = None,  # noqa: F811
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
  ) -> Any:
  """Parallel map with support for collective operations.

  The purpose of :py:func:`pmap` is to express single-program multiple-data
  (SPMD) programs. Applying :py:func:`pmap` to a function will compile the
  function with XLA (similarly to :py:func:`jit`), then execute it in parallel
  on XLA devices, such as multiple GPUs or multiple TPU cores. Semantically it
  is comparable to :py:func:`vmap` because both transformations map a function
  over array axes, but where :py:func:`vmap` vectorizes functions by pushing the
  mapped axis down into primitive operations, :py:func:`pmap` instead replicates
  the function and executes each replica on its own XLA device in parallel.

  The mapped axis size must be less than or equal to the number of local XLA
  devices available, as returned by :py:func:`jax.local_device_count()` (unless
  ``devices`` is specified, see below). For nested :py:func:`pmap` calls, the
  product of the mapped axis sizes must be less than or equal to the number of
  XLA devices.

  .. note::
    :py:func:`pmap` compiles ``fun``, so while it can be combined with
    :py:func:`jit`, it's usually unnecessary.

  :py:func:`pmap` requires that all of the participating devices are identical.
  For example, it is not possible to use :py:func:`pmap` to parallelize a
  computation across two different models of GPU. It is currently an error for
  the same device to participate twice in the same `pmap`.

  **Multi-process platforms:** On multi-process platforms such as TPU pods,
  :py:func:`pmap` is designed to be used in SPMD Python programs, where every
  process is running the same Python code such that all processes run the same
  pmapped function in the same order. Each process should still call the pmapped
  function with mapped axis size equal to the number of *local* devices (unless
  ``devices`` is specified, see below), and an array of the same leading axis
  size will be returned as usual. However, any collective operations in ``fun``
  will be computed over *all* participating devices, including those on other
  processes, via device-to-device communication.  Conceptually, this can be
  thought of as running a pmap over a single array sharded across processes,
  where each process "sees" only its local shard of the input and output. The
  SPMD model requires that the same multi-process pmaps must be run in the same
  order on all devices, but they can be interspersed with arbitrary operations
  running in a single process.

  Args:
    fun: Function to be mapped over argument axes. Its arguments and return
      value should be arrays, scalars, or (nested) standard Python containers
      (tuple/list/dict) thereof. Positional arguments indicated by
      ``static_broadcasted_argnums`` can be anything at all, provided they are
      hashable and have an equality operation defined.
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    in_axes: A non-negative integer, None, or nested Python container thereof
      that specifies which axes of positional arguments to map over. Arguments
      passed as keywords are always mapped over their leading axis (i.e. axis
      index 0). See :py:func:`vmap` for details.
    out_axes: A non-negative integer, None, or nested Python container thereof
      indicating where the mapped axis should appear in the output. All outputs
      with a mapped axis must have a non-None ``out_axes`` specification
      (see :py:func:`vmap`).
    static_broadcasted_argnums: An int or collection of ints specifying which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded.
      Calling the pmapped function with different values for these constants
      will trigger recompilation. If the pmapped function is called with fewer
      positional arguments than indicated by ``static_argnums`` then an error is
      raised. Each of the static arguments will be broadcasted to all devices.
      Arguments that are not arrays or containers thereof must be marked as
      static. Defaults to ().

      Static arguments must be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and should be immutable.

    devices: This is an experimental feature and the API is likely to change.
      Optional, a sequence of Devices to map over. (Available devices can be
      retrieved via jax.devices()). Must be given identically for each process
      in multi-process settings (and will therefore include devices across
      processes). If specified, the size of the mapped axis must be equal to
      the number of devices in the sequence local to the given process. Nested
      :py:func:`pmap` s with ``devices`` specified in either the inner or outer
      :py:func:`pmap` are not yet supported.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend. 'cpu', 'gpu', or 'tpu'.
    axis_size: Optional; the size of the mapped axis.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer need
      them once the computation has finished. In some cases XLA can make use of
      donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to.
      Note that donate_argnums only work for positional arguments, and keyword
      arguments will not be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.

    global_arg_shapes: Optional, must be set when using pmap(sharded_jit) and
      the partitioned values span multiple processes. The global cross-process
      per-replica shape of each argument, i.e. does not include the leading
      pmapped dimension. Can be None for replicated arguments. This API is
      likely to change in the future.

  Returns:
    A parallelized version of ``fun`` with arguments that correspond to those of
    ``fun`` but with extra array axes at positions indicated by ``in_axes`` and
    with output that has an additional leading array axis (with the same size).

  For example, assuming 8 XLA devices are available, :py:func:`pmap` can be used
  as a map along a leading array axis:

  >>> import jax.numpy as jnp
  >>>
  >>> out = pmap(lambda x: x ** 2)(jnp.arange(8))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [0, 1, 4, 9, 16, 25, 36, 49]

  When the leading dimension is smaller than the number of available devices JAX
  will simply run on a subset of devices:

  >>> x = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2))
  >>> y = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2)) ** 2
  >>> out = pmap(jnp.dot)(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [[[    4.     9.]
    [   12.    29.]]
   [[  244.   345.]
    [  348.   493.]]
   [[ 1412.  1737.]
    [ 1740.  2141.]]]

  If your leading dimension is larger than the number of available devices you
  will get an error:

  >>> pmap(lambda x: x ** 2)(jnp.arange(9))  # doctest: +SKIP
  ValueError: ... requires 9 replicas, but only 8 XLA devices are available

  As with :py:func:`vmap`, using ``None`` in ``in_axes`` indicates that an
  argument doesn't have an extra axis and should be broadcasted, rather than
  mapped, across the replicas:

  >>> x, y = jnp.arange(2.), 4.
  >>> out = pmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None))(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  ([4., 5.], [8., 8.])

  Note that :py:func:`pmap` always returns values mapped over their leading axis,
  equivalent to using ``out_axes=0`` in :py:func:`vmap`.

  In addition to expressing pure maps, :py:func:`pmap` can also be used to express
  parallel single-program multiple-data (SPMD) programs that communicate via
  collective operations. For example:

  >>> f = lambda x: x / jax.lax.psum(x, axis_name='i')
  >>> out = pmap(f, axis_name='i')(jnp.arange(4.))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [ 0.          0.16666667  0.33333334  0.5       ]
  >>> print(out.sum())  # doctest: +SKIP
  1.0

  In this example, ``axis_name`` is a string, but it can be any Python object
  with ``__hash__`` and ``__eq__`` defined.

  The argument ``axis_name`` to :py:func:`pmap` names the mapped axis so that
  collective operations, like :func:`jax.lax.psum`, can refer to it. Axis names
  are important particularly in the case of nested :py:func:`pmap` functions,
  where collective operations can operate over distinct axes:

  >>> from functools import partial
  >>> import jax
  >>>
  >>> @partial(pmap, axis_name='rows')
  ... @partial(pmap, axis_name='cols')
  ... def normalize(x):
  ...   row_normed = x / jax.lax.psum(x, 'rows')
  ...   col_normed = x / jax.lax.psum(x, 'cols')
  ...   doubly_normed = x / jax.lax.psum(x, ('rows', 'cols'))
  ...   return row_normed, col_normed, doubly_normed
  >>>
  >>> x = jnp.arange(8.).reshape((4, 2))
  >>> row_normed, col_normed, doubly_normed = normalize(x)  # doctest: +SKIP
  >>> print(row_normed.sum(0))  # doctest: +SKIP
  [ 1.  1.]
  >>> print(col_normed.sum(1))  # doctest: +SKIP
  [ 1.  1.  1.  1.]
  >>> print(doubly_normed.sum((0, 1)))  # doctest: +SKIP
  1.0

  On multi-process platforms, collective operations operate over all devices,
  including those on other processes. For example, assuming the following code
  runs on two processes with 4 XLA devices each:

  >>> f = lambda x: x + jax.lax.psum(x, axis_name='i')
  >>> data = jnp.arange(4) if jax.process_index() == 0 else jnp.arange(4, 8)
  >>> out = pmap(f, axis_name='i')(data)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [28 29 30 31] # on process 0
  [32 33 34 35] # on process 1

  Each process passes in a different length-4 array, corresponding to its 4
  local devices, and the psum operates over all 8 values. Conceptually, the two
  length-4 arrays can be thought of as a sharded length-8 array (in this example
  equivalent to jnp.arange(8)) that is mapped over, with the length-8 mapped
  axis given name 'i'. The pmap call on each process then returns the
  corresponding length-4 output shard.

  The ``devices`` argument can be used to specify exactly which devices are used
  to run the parallel computation. For example, again assuming a single process
  with 8 devices, the following code defines two parallel computations, one
  which runs on the first six devices and one on the remaining two:

  >>> from functools import partial
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[:6])
  ... def f1(x):
  ...   return x / jax.lax.psum(x, axis_name='i')
  >>>
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[-2:])
  ... def f2(x):
  ...   return jax.lax.psum(x ** 2, axis_name='i')
  >>>
  >>> print(f1(jnp.arange(6.)))  # doctest: +SKIP
  [0.         0.06666667 0.13333333 0.2        0.26666667 0.33333333]
  >>> print(f2(jnp.array([2., 3.])))  # doctest: +SKIP
  [ 13.  13.]
  """
  if FLAGS.experimental_cpp_pmap:
    func = _cpp_pmap
  else:
    func = _python_pmap

  return func(
      fun,
      axis_name,
      in_axes=in_axes,
      out_axes=out_axes,
      static_broadcasted_argnums=static_broadcasted_argnums,
      devices=devices,
      backend=backend,
      axis_size=axis_size,
      donate_argnums=donate_argnums,
      global_arg_shapes=global_arg_shapes)


class PmapCallInfo(NamedTuple):
  flat_fun: lu.WrappedFun
  in_tree: PyTreeDef
  out_tree: PyTreeDef
  flat_args: Sequence[Any]
  donated_invars: Sequence[bool]
  in_axes_flat: Sequence[Optional[int]]
  local_axis_size: int
  global_arg_shapes_flat: Sequence[Optional[Tuple[int, ...]]]
  out_axes_thunk: HashableFunction
  devices: Optional[Sequence[xc.Device]]
  global_axis_size: int
  is_explicit_global_axis_size: bool


def _get_global_axis_size(local_axis_size: int, in_devices, backend_name: str,
                          global_axis_size: Optional[int]):
  """Determine global_axis_size for multi-host pmap."""
  # TODO(mattjj,skyewm): revive this check (inner_pmap always False now)
  # if xb.process_count() > 1 and global_axis_size is None and inner_pmap:
  #   raise ValueError("'axis_size' must be specified for nested multi-host pmaps")
  if (xb.process_count() == 1 and global_axis_size is not None and
      global_axis_size != local_axis_size):
    raise ValueError(
        f"Specified axis_size {global_axis_size} doesn't match received "
        f"axis_size {local_axis_size}.")

  if in_devices is not None and backend_name is None:
    backend = xb.get_device_backend(in_devices[0])
  else:
    backend = xb.get_backend(backend_name)

  if global_axis_size is None:
    if xb.process_count(backend) == 1:
      global_axis_size = local_axis_size
    elif in_devices:
      global_axis_size = len(in_devices)
    else:
      global_axis_size = local_axis_size * xb.process_count(backend)
      assert all(
          len(xb.local_devices(pi, backend)) == xb.local_device_count(backend)
          for pi in range(xb.process_count(backend)))
  return global_axis_size

def _prepare_pmap(fun, in_axes, out_axes, static_broadcasted_tuple,
                  donate_tuple, global_arg_shapes, in_devices, backend_name,
                  axis_size, args, kwargs):
  if in_devices is not None and len(in_devices) == 0:
    raise ValueError("'devices' argument to pmap must be non-empty, or None.")

  f = lu.wrap_init(fun)
  if static_broadcasted_tuple:
    if max(static_broadcasted_tuple) >= len(args):
      raise ValueError(
          f"pmapped function has static_broadcasted_argnums={static_broadcasted_tuple}"
          f" but was called with only {len(args)} positional "
          f"argument{'s' if len(args) > 1 else ''}. "
          "All static broadcasted arguments must be passed positionally.")
    dyn_argnums = [i for i in range(len(args))
                   if i not in static_broadcasted_tuple]
    f, dyn_args = argnums_partial(f, dyn_argnums, args)

    if isinstance(in_axes, tuple):
      dyn_in_axes = tuple(in_axes[i] for i in dyn_argnums)
    else:
      dyn_in_axes = in_axes
      dyn_global_arg_shapes = global_arg_shapes

    if isinstance(global_arg_shapes, tuple):
      dyn_global_arg_shapes = tuple(global_arg_shapes[i] for i in dyn_argnums)
    else:
      dyn_global_arg_shapes = global_arg_shapes
  else:
    dyn_args, dyn_in_axes = args, in_axes
    dyn_global_arg_shapes = global_arg_shapes
  args, in_tree = tree_flatten((dyn_args, kwargs))

  if donate_tuple and not config.jax_debug_nans:
    donated_invars = donation_vector(donate_tuple, dyn_args, kwargs)
  else:
    donated_invars = (False,) * len(args)
  in_axes_flat = tuple(flatten_axes("pmap in_axes", in_tree, (dyn_in_axes, 0)))
  global_arg_shapes_flat = tuple(flatten_axes(
      "pmap global_arg_shapes", in_tree, (dyn_global_arg_shapes, None),
      kws=True))
  local_axis_size = _mapped_axis_size(fun, in_tree, args, in_axes_flat, "pmap")

  flat_fun, out_tree = flatten_fun(f, in_tree)

  if any(out_axis is None for out_axis in tree_flatten(out_axes)):
    raise NotImplementedError("None out_axes in pmap are not supported yet")
  # NOTE: We don't put out_tree() in the closure, because it's (1) non-hashable,
  #       (2) depends deterministically on flat_fun (at least that's the assumption
  #       that we make).
  if out_axes == 0:
    # TODO(apaszke,mattjj): flatten_axes assumes that the output pytree is
    #   functorial (i.e. it can hold leaves of any type), but some user code
    #   breaks this assumption. This is a stop-gap solution to keep the old
    #   out_axes == 0 path working as we look for a better solution.
    out_axes_thunk = HashableFunction(
        lambda: (0,) * out_tree().num_leaves,
        closure=out_axes)
  else:
    # out_axes_thunk closes over the out_axes, they are flattened here to make
    # them hashable.
    out_axes_leaves, out_axes_treedef = tree_flatten(out_axes)
    out_axes_thunk = HashableFunction(
        lambda: tuple(flatten_axes("pmap out_axes", out_tree(),
                                    tree_unflatten(out_axes_treedef,
                                                  list(out_axes_leaves)))),
        closure=(tuple(out_axes_leaves), out_axes_treedef))

  is_explicit_global_axis_size = axis_size is not None
  global_axis_size = _get_global_axis_size(local_axis_size, in_devices,
                                           backend_name, axis_size)
  return PmapCallInfo(flat_fun=flat_fun,
                      in_tree=in_tree,
                      out_tree=out_tree,
                      flat_args=args,
                      donated_invars=donated_invars,
                      in_axes_flat=in_axes_flat,
                      local_axis_size=local_axis_size,
                      global_arg_shapes_flat=global_arg_shapes_flat,
                      out_axes_thunk=out_axes_thunk,
                      devices=None if in_devices is None else tuple(in_devices),
                      global_axis_size=global_axis_size,
                      is_explicit_global_axis_size=is_explicit_global_axis_size)


def _get_f_mapped(
    *,
    fun: Callable,
    axis_name: Optional[AxisName],
    in_axes=0,
    out_axes=0,
    static_broadcasted_tuple: Tuple[int, ...],
    devices: Optional[Sequence[xc.Device]],  # noqa: F811
    backend: Optional[str],
    axis_size: Optional[int],
    donate_tuple: Tuple[int, ...],
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]],
  ):
  def pmap_f(*args, **kwargs):
    p = _prepare_pmap(
        fun, in_axes, out_axes, static_broadcasted_tuple, donate_tuple,
        global_arg_shapes, devices, backend, axis_size, args, kwargs)
    for arg in p.flat_args:
      dispatch.check_arg(arg)
    out = pxla.xla_pmap(
        p.flat_fun, *p.flat_args, backend=backend, axis_name=axis_name,
        axis_size=p.local_axis_size, global_axis_size=p.global_axis_size,
        devices=p.devices,
        in_axes=p.in_axes_flat, out_axes_thunk=p.out_axes_thunk,
        name=p.flat_fun.__name__, donated_invars=p.donated_invars,
        global_arg_shapes=p.global_arg_shapes_flat,
        is_explicit_global_axis_size=p.is_explicit_global_axis_size)
    return p.out_tree, out

  return pmap_f


def _shared_code_pmap(fun, axis_name, static_broadcasted_argnums,
                      donate_argnums, in_axes, out_axes):
  # axis_size is an optional integer representing the global axis size.  The
  # aggregate size (across all processes) size of the mapped axis must match the
  # given value.
  check_callable(fun)
  axis_name = core._TempAxisName(fun) if axis_name is None else axis_name
  static_broadcasted_tuple = _ensure_index_tuple(static_broadcasted_argnums)
  donate_tuple = rebase_donate_argnums(
      _ensure_index_tuple(donate_argnums), static_broadcasted_tuple)

  if not all(type(l) is int for l in tree_leaves(in_axes)):
    raise TypeError("pmap in_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {in_axes}.")
  if not all(type(l) is int for l in tree_leaves(out_axes)):
    raise TypeError("pmap out_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {out_axes}.")

  return axis_name, static_broadcasted_tuple, donate_tuple


def _python_pmap(
    fun: Callable,
    axis_name: Optional[AxisName] = None,
    *,
    in_axes=0,
    out_axes=0,
    static_broadcasted_argnums: Union[int, Iterable[int]] = (),
    devices: Optional[Sequence[xc.Device]] = None,  # noqa: F811
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
  ) -> stages.Wrapped:
  """The Python only implementation."""
  axis_name, static_broadcasted_tuple, donate_tuple = _shared_code_pmap(
      fun, axis_name, static_broadcasted_argnums, donate_argnums, in_axes,
      out_axes)

  @wraps(fun)
  @api_boundary
  def pmap_f(*args, **kwargs):
    f_pmapped_ = _get_f_mapped(
        fun=fun,
        axis_name=axis_name,
        in_axes=in_axes,
        out_axes=out_axes,
        static_broadcasted_tuple=static_broadcasted_tuple,
        devices=devices,
        backend=backend,
        axis_size=axis_size,
        global_arg_shapes=global_arg_shapes,
        donate_tuple=donate_tuple)

    out_tree, out_flat = f_pmapped_(*args, **kwargs)
    return tree_unflatten(out_tree(), out_flat)

  pmap_f.lower = _pmap_lower(
      fun, axis_name, in_axes, out_axes, static_broadcasted_tuple, devices,
      backend, axis_size, global_arg_shapes, donate_tuple)

  return pmap_f


class _PmapFastpathData(NamedTuple):
  version: int  # For forward and backward compatibility
  xla_executable: xc.LoadedExecutable
  in_handler: Any
  out_handler: Any
  out_pytree_def: Any
  # Data needed to handle the inputs.
  input_sharding_specs: Sequence[pxla.ShardingSpec]
  input_devices: Sequence[xc.Device]
  input_indices: Sequence[pxla.Index]
  input_array_shardings: Sequence[Any]
  # Data needed to build the ShardedDeviceArray from C++.
  out_sharding_specs: Sequence[pxla.ShardingSpec]
  out_indices: Sequence[pxla.Index]
  out_avals: Sequence[Any]
  out_array_shardings: Sequence[Any]
  out_committed: Sequence[Any]


def _cpp_pmap(
    fun: Callable,
    axis_name: Optional[AxisName] = None,
    *,
    in_axes=0,
    out_axes=0,
    static_broadcasted_argnums: Union[int, Iterable[int]] = (),
    devices: Optional[Sequence[xc.Device]] = None,  # noqa: F811
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
  ) -> Any:
  axis_name, static_broadcasted_tuple, donate_tuple = _shared_code_pmap(
      fun, axis_name, static_broadcasted_argnums, donate_argnums, in_axes,
      out_axes)
  del static_broadcasted_argnums, donate_argnums

  @api_boundary
  def cache_miss(*args, **kwargs):
    p = _prepare_pmap(fun, in_axes, out_axes, static_broadcasted_tuple,
                      donate_tuple, global_arg_shapes, devices, backend,
                      axis_size, args, kwargs)
    for arg in p.flat_args:
      dispatch.check_arg(arg)

    params = dict(
        backend=backend,
        axis_name=axis_name,
        axis_size=p.local_axis_size,
        global_axis_size=p.global_axis_size,
        devices=p.devices,
        in_axes=p.in_axes_flat,
        out_axes_thunk=p.out_axes_thunk,
        name=p.flat_fun.__name__,
        donated_invars=p.donated_invars,
        global_arg_shapes=p.global_arg_shapes_flat,
        is_explicit_global_axis_size=p.is_explicit_global_axis_size,
    )

    map_bind_continuation, top_trace, fun_, tracers, params = (
        core.map_bind_with_continuation(pxla.xla_pmap_p, p.flat_fun,
                                        *p.flat_args, **params))
    execute: Optional[Callable] = None
    if isinstance(top_trace, core.EvalTrace):
      execute = pxla.xla_pmap_impl_lazy(fun_, *tracers, **params)
      out = map_bind_continuation(execute(*tracers))
    else:
      out = map_bind_continuation(
          pxla.xla_pmap_p.process(top_trace, fun_, tracers, params))

    out_tree, out_flat = p.out_tree, out
    out_pytree_def = out_tree()
    out = tree_unflatten(out_pytree_def, out_flat)

    ### Decide whether we can support the C++ fast path
    use_fastpath = False
    if execute is not None and isinstance(execute, pxla.ExecuteReplicated):
      execute_replicated = typing.cast(pxla.ExecuteReplicated, execute)
      use_fastpath = (
        # TODO(sharadmv): Enable effects in replicated computation
        not execute_replicated.has_unordered_effects
        and not execute_replicated.has_host_callbacks and
        # No tracers in the outputs. Checking for ShardedDeviceArray should be
        # sufficient, but we use the more general `DeviceArray`.
        all(
            isinstance(x, device_array.DeviceArray) or isinstance(x, xc.ArrayImpl)
            for x in out_flat))

    ### If we can use the fastpath, we return required info to the caller.
    if use_fastpath:
      execute_replicated = typing.cast(pxla.ExecuteReplicated, execute)
      out_handler = execute_replicated.out_handler
      in_handler = execute_replicated.in_handler
      out_indices = [tuple(s.devices_indices_map(a.shape).values())
                     for s, a in safe_zip(out_handler.out_shardings, out_handler.out_avals)]

      if jax.config.jax_array:
        out_array_shardings = [out.sharding for out in out_flat]
        out_committed = [out._committed for out in out_flat]
      else:
        out_array_shardings = []
        out_committed = []

      fastpath_data = _PmapFastpathData(
          version=1,
          xla_executable=execute_replicated.xla_executable,
          in_handler=in_handler,
          out_handler=out_handler,
          out_pytree_def=out_pytree_def,
          input_sharding_specs=[i.sharding_spec for i in in_handler.in_shardings],
          input_devices=in_handler.local_devices,
          input_indices=in_handler.input_indices,
          input_array_shardings=in_handler.in_shardings,
          out_sharding_specs=[s.sharding_spec for s in out_handler.out_shardings],
          out_indices=out_indices,
          out_avals=out_handler.out_avals,
          out_array_shardings=out_array_shardings,
          out_committed=out_committed,
      )

    else:
      fastpath_data = None

    return out, fastpath_data

  cpp_mapped_f = pmap_lib.pmap(
      fun, cache_miss, static_broadcasted_tuple, pxla.shard_arg)

  pmap_f = wraps(fun)(cpp_mapped_f)

  pmap_f.lower = _pmap_lower(
      fun, axis_name, in_axes, out_axes, static_broadcasted_tuple, devices,
      backend, axis_size, global_arg_shapes, donate_tuple)

  return pmap_f


def _pmap_lower(fun, axis_name, in_axes, out_axes, static_broadcasted_tuple,
                devices, backend, axis_size, global_arg_shapes, donate_tuple):  # noqa: F811
  """Make a ``lower`` method for pmapped functions."""
  # If the function we returned from ``pmap`` were a class instance,
  # this might naturally be a method, with ``fun`` as a ``self`` and
  # all the other arguments stored as attributes.
  @api_boundary
  def lower(*args, **kwargs) -> stages.Lowered:
    """Lower a parallel-mapped form of this function for the given arguments.

    A parallel-mapped and lowered function is staged out of Python and
    translated to a compiler's input language, possibly in a
    backend-dependent manner. It is ready for compilation but is not yet
    compiled. It represents a function intended for SPMD execution on
    multiple devices.

    Returns:
      A ``Lowered`` instance representing the post-map lowering.
    """
    p = _prepare_pmap(
        fun, in_axes, out_axes, static_broadcasted_tuple, donate_tuple,
        global_arg_shapes, devices, backend, axis_size, args, kwargs)
    abstract_args = list(map(shaped_abstractify, p.flat_args))
    computation = pxla.lower_parallel_callable(
        p.flat_fun, backend, axis_name,
        axis_size=p.local_axis_size, global_axis_size=p.global_axis_size,
        devices=p.devices,
        name=p.flat_fun.__name__,
        in_axes=p.in_axes_flat,
        out_axes_thunk=p.out_axes_thunk,
        donated_invars=p.donated_invars,
        global_arg_shapes=p.global_arg_shapes_flat,
        is_explicit_global_axis_size=p.is_explicit_global_axis_size,
        avals=abstract_args)
    return stages.Lowered.from_flat_info(
        computation, p.in_tree, abstract_args, donate_tuple, p.out_tree())

  return lower

def jvp(
    fun: Callable, primals, tangents, has_aux: bool = False
  ) -> Tuple[Any, ...]:
  """Computes a (forward-mode) Jacobian-vector product of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: The primal values at which the Jacobian of ``fun`` should be
      evaluated. Should be either a tuple or a list of arguments,
      and its length should be equal to the number of positional parameters of
      ``fun``.
    tangents: The tangent vector for which the Jacobian-vector product should be
      evaluated. Should be either a tuple or a list of tangents, with the same
      tree structure and array shapes as ``primals``.
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.

  Returns:
    If ``has_aux`` is ``False``, returns a ``(primals_out, tangents_out)`` pair,
    where ``primals_out`` is ``fun(*primals)``,
    and ``tangents_out`` is the Jacobian-vector product of
    ``function`` evaluated at ``primals`` with ``tangents``. The
    ``tangents_out`` value has the same Python tree structure and shapes as
    ``primals_out``. If ``has_aux`` is ``True``, returns a
    ``(primals_out, tangents_out, aux)`` tuple where ``aux``
    is the auxiliary data returned by ``fun``.

  For example:

  >>> import jax
  >>>
  >>> primals, tangents = jax.jvp(jax.numpy.sin, (0.1,), (0.2,))
  >>> print(primals)
  0.09983342
  >>> print(tangents)
  0.19900084
  """
  check_callable(fun)
  return _jvp(lu.wrap_init(fun), primals, tangents, has_aux=has_aux)

def _jvp(fun: lu.WrappedFun, primals, tangents, has_aux=False):
  """Variant of jvp() that takes an lu.WrappedFun."""
  if (not isinstance(primals, (tuple, list)) or
      not isinstance(tangents, (tuple, list))):
    raise TypeError("primal and tangent arguments to jax.jvp must be tuples or lists; "
                    f"found {type(primals).__name__} and {type(tangents).__name__}.")

  ps_flat, tree_def = tree_flatten(primals)
  ts_flat, tree_def_2 = tree_flatten(tangents)
  if tree_def != tree_def_2:
    raise TypeError("primal and tangent arguments to jax.jvp must have the same tree "
                    f"structure; primals have tree structure {tree_def} whereas tangents have "
                    f"tree structure {tree_def_2}.")
  for p, t in safe_zip(ps_flat, ts_flat):
    if core.primal_dtype_to_tangent_dtype(_dtype(p)) != _dtype(t):
      raise TypeError("primal and tangent arguments to jax.jvp do not match; "
                      "dtypes must be equal, or in case of int/bool primal dtype "
                      "the tangent dtype must be float0."
                      f"Got primal dtype {_dtype(p)} and so expected tangent dtype "
                      f"{core.primal_dtype_to_tangent_dtype(_dtype(p))}, but got "
                      f"tangent dtype {_dtype(t)} instead.")
    if np.shape(p) != np.shape(t):
      raise ValueError("jvp called with different primal and tangent shapes;"
                       f"Got primal shape {np.shape(p)} and tangent shape as {np.shape(t)}")

  if not has_aux:
    flat_fun, out_tree = flatten_fun_nokwargs(fun, tree_def)
    out_primals, out_tangents = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)
    out_tree = out_tree()
    return (tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents))
  else:
    flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, tree_def)
    jvp_fun, aux = ad.jvp(flat_fun, has_aux=True)
    out_primals, out_tangents = jvp_fun.call_wrapped(ps_flat, ts_flat)
    out_tree, aux_tree = out_aux_trees()
    return (tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents),
            tree_unflatten(aux_tree, aux()))

def linearize(fun: Callable, *primals) -> Tuple[Any, Callable]:
  """Produces a linear approximation to ``fun`` using :py:func:`jvp` and partial eval.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard python container of arrays or scalars.
    primals: The primal values at which the Jacobian of ``fun`` should be
      evaluated. Should be a tuple of arrays, scalar, or standard Python
      container thereof. The length of the tuple is equal to the number of
      positional parameters of ``fun``.

  Returns:
    A pair where the first element is the value of ``f(*primals)`` and the
    second element is a function that evaluates the (forward-mode)
    Jacobian-vector product of ``fun`` evaluated at ``primals`` without re-doing
    the linearization work.

  In terms of values computed, :py:func:`linearize` behaves much like a curried
  :py:func:`jvp`, where these two code blocks compute the same values::

    y, out_tangent = jax.jvp(f, (x,), (in_tangent,))

    y, f_jvp = jax.linearize(f, x)
    out_tangent = f_jvp(in_tangent)

  However, the difference is that :py:func:`linearize` uses partial evaluation
  so that the function ``f`` is not re-linearized on calls to ``f_jvp``. In
  general that means the memory usage scales with the size of the computation,
  much like in reverse-mode. (Indeed, :py:func:`linearize` has a similar
  signature to :py:func:`vjp`!)

  This function is mainly useful if you want to apply ``f_jvp`` multiple times,
  i.e. to evaluate a pushforward for many different input tangent vectors at the
  same linearization point. Moreover if all the input tangent vectors are known
  at once, it can be more efficient to vectorize using :py:func:`vmap`, as in::

    pushfwd = partial(jvp, f, (x,))
    y, out_tangents = vmap(pushfwd, out_axes=(None, 0))((in_tangents,))

  By using :py:func:`vmap` and :py:func:`jvp` together like this we avoid the stored-linearization
  memory cost that scales with the depth of the computation, which is incurred
  by both :py:func:`linearize` and :py:func:`vjp`.

  Here's a more complete example of using :py:func:`linearize`:

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x): return 3. * jnp.sin(x) + jnp.cos(x / 2.)
  ...
  >>> jax.jvp(f, (2.,), (3.,))
  (Array(3.26819, dtype=float32, weak_type=True), Array(-5.00753, dtype=float32, weak_type=True))
  >>> y, f_jvp = jax.linearize(f, 2.)
  >>> print(y)
  3.2681944
  >>> print(f_jvp(3.))
  -5.007528
  >>> print(f_jvp(4.))
  -6.676704
  """
  check_callable(fun)
  f = lu.wrap_init(fun)
  primals_flat, in_tree = tree_flatten((primals, {}))
  jaxtree_fun, out_tree = flatten_fun(f, in_tree)
  out_primals, out_pvals, jaxpr, consts = ad.linearize(jaxtree_fun, *primals_flat)
  out_tree = out_tree()
  out_primal_py = tree_unflatten(out_tree, out_primals)
  primal_avals = list(map(core.get_aval, primals_flat))
  # Ensure that lifted_jvp is a PyTree
  lifted_jvp = Partial(partial(_lift_linearized, jaxpr, primal_avals,
                               (in_tree, out_tree), out_pvals), consts)
  return out_primal_py, lifted_jvp

def _lift_linearized(jaxpr, primal_avals, io_tree, out_pvals, consts, *py_args):
  def fun(*tangents):
    tangent_avals = list(map(core.get_aval, tangents))
    for primal_aval, tangent_aval in zip(primal_avals, tangent_avals):
      if not core.typecompat(primal_aval.at_least_vspace(), tangent_aval):
        raise ValueError("linearized function called on tangent values inconsistent with "
                         "the original primal values: "
                         f"got {tangent_aval} for primal aval {primal_aval}")
    tangents_out = eval_jaxpr(jaxpr, consts, *tangents)
    tangents_out_ = iter(tangents_out)
    full_out = [pval.get_known() if pval.is_known() else next(tangents_out_)
                for pval in out_pvals]
    assert next(tangents_out_, None) is None
    return full_out

  return apply_flat_fun(fun, io_tree, *py_args)

def _vjp_pullback_wrapper(name, cotangent_dtypes, cotangent_shapes, io_tree,
                          fun, *py_args_):
  if len(py_args_) != 1:
    msg = (f"The function returned by `jax.vjp` applied to {name} was called "
           f"with {len(py_args_)} arguments, but functions returned by "
           "`jax.vjp` must be called with a single argument corresponding to "
           f"the single value returned by {name} (even if that returned "
           "value is a tuple or other container).\n"
           "\n"
           "For example, if we have:\n"
           "\n"
           "  def f(x):\n"
           "    return (x, x)\n"
           "  _, f_vjp = jax.vjp(f, 1.0)\n"
           "\n"
           "the function `f` returns a single tuple as output, and so we call "
           "`f_vjp` with a single tuple as its argument:\n"
           "\n"
           "  x_bar, = f_vjp((2.0, 2.0))\n"
           "\n"
           "If we instead call `f_vjp(2.0, 2.0)`, with the values 'splatted "
           "out' as arguments rather than in a tuple, this error can arise.")
    raise TypeError(msg)
  py_args, = py_args_
  in_tree_expected, out_tree = io_tree
  args, in_tree = tree_flatten(py_args)
  if in_tree != in_tree_expected:
    raise TypeError(f"Tree structure of cotangent input {in_tree}, does not match structure of "
                    f"primal output {in_tree_expected}.")
  for arg, ct_dtype, ct_shape in safe_zip(args, cotangent_dtypes, cotangent_shapes):
    expected_tangent_dtype = core.primal_dtype_to_tangent_dtype(_dtype(arg))
    if expected_tangent_dtype != ct_dtype:
      raise TypeError(
          f"Type of cotangent input to vjp pullback function ({ct_dtype}) is not "
          f"the expected tangent type ({expected_tangent_dtype}) of corresponding primal output "
          f"with dtype {_dtype(arg)}.")
    if np.shape(arg) != ct_shape:
      raise ValueError(
          f"Shape of cotangent input to vjp pullback function {np.shape(arg)} "
          "must be the same as the shape of corresponding primal input "
          f"{ct_shape}.")
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)

@overload
def vjp(fun: Callable[..., T],
        *primals: Any,
        has_aux: Literal[False] = False,
        reduce_axes: Sequence[AxisName] = ()) -> Tuple[T, Callable]:
  ...

@overload
def vjp(fun: Callable[..., Tuple[T, U]], *primals: Any,
        has_aux: Literal[True],
        reduce_axes: Sequence[AxisName] = ()) -> Tuple[T, Callable, U]:
  ...
def vjp(  # type: ignore
    fun: Callable, *primals, has_aux: bool = False, reduce_axes=()
  ) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:
  """Compute a (reverse-mode) vector-Jacobian product of ``fun``.

  :py:func:`grad` is implemented as a special case of :py:func:`vjp`.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: A sequence of primal values at which the Jacobian of ``fun``
      should be evaluated. The number of ``primals`` should be equal to the
      number of positional parameters of ``fun``. Each primal value should be
      an array, a scalar, or a pytree (standard Python containers) thereof.
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding gradient. Otherwise, the
      VJP will be per-example over named axes. For example, if ``'batch'``
      is a named batch axis, ``vjp(f, *args, reduce_axes=('batch',))`` will
      create a VJP function that sums over the batch while ``vjp(f, *args)``
      will create a per-example VJP.

  Returns:
    If ``has_aux`` is ``False``, returns a ``(primals_out, vjpfun)`` pair, where
    ``primals_out`` is ``fun(*primals)``. If ``has_aux`` is ``True``, returns a
    ``(primals_out, vjpfun, aux)`` tuple where ``aux`` is the auxiliary data
    returned by ``fun``.

    ``vjpfun`` is a function from a cotangent vector with the same shape as
    ``primals_out`` to a tuple of cotangent vectors with the same number and
    shapes as ``primals``, representing the vector-Jacobian product of ``fun``
    evaluated at ``primals``.

  >>> import jax
  >>>
  >>> def f(x, y):
  ...   return jax.numpy.sin(x), jax.numpy.cos(y)
  ...
  >>> primals, f_vjp = jax.vjp(f, 0.5, 1.0)
  >>> xbar, ybar = f_vjp((-0.7, 0.3))
  >>> print(xbar)
  -0.61430776
  >>> print(ybar)
  -0.2524413
  """
  check_callable(fun)
  reduce_axes = _ensure_str_tuple(reduce_axes)  # type: ignore
  return _vjp(
      lu.wrap_init(fun), *primals, has_aux=has_aux, reduce_axes=reduce_axes)

def _vjp(fun: lu.WrappedFun, *primals, has_aux=False, reduce_axes=()):
  """Variant of vjp() that takes an lu.WrappedFun."""
  primals_flat, in_tree = tree_flatten(primals)
  for arg in primals_flat: dispatch.check_arg(arg)
  if not has_aux:
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    out_primal, out_vjp = ad.vjp(
        flat_fun, primals_flat, reduce_axes=reduce_axes)
    out_tree = out_tree()
  else:
    flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, in_tree)
    out_primal, out_vjp, aux = ad.vjp(
        flat_fun, primals_flat, has_aux=True, reduce_axes=reduce_axes)
    out_tree, aux_tree = out_aux_trees()
  out_primal_py = tree_unflatten(out_tree, out_primal)
  ct_dtypes = [core.primal_dtype_to_tangent_dtype(_dtype(x)) for x in out_primal]
  ct_shapes = [np.shape(x) for x in out_primal]
  # Ensure that vjp_py is a PyTree so that we can pass it from the forward to the
  # backward pass in a custom VJP.
  vjp_py = Partial(partial(_vjp_pullback_wrapper, fun.__name__,
                           ct_dtypes, ct_shapes, (out_tree, in_tree)),
                   out_vjp)
  if not has_aux:
    return out_primal_py, vjp_py
  else:
    return out_primal_py, vjp_py, tree_unflatten(aux_tree, aux)


def linear_transpose(fun: Callable, *primals, reduce_axes=()) -> Callable:
  """Transpose a function that is promised to be linear.

  For linear functions, this transformation is equivalent to :py:func:`vjp`, but
  avoids the overhead of computing the forward pass.

  The outputs of the transposed function will always have the exact same dtypes
  as ``primals``, even if some values are truncated (e.g., from complex to
  float, or from float64 to float32). To avoid truncation, use dtypes in
  ``primals`` that match the full range of desired outputs from the transposed
  function. Integer dtypes are not supported.

  Args:
    fun: the linear function to be transposed.
    *primals: a positional argument tuple of arrays, scalars, or (nested)
      standard Python containers (tuples, lists, dicts, namedtuples, i.e.,
      pytrees) of those types used for evaluating the shape/dtype of
      ``fun(*primals)``. These arguments may be real scalars/ndarrays, but that
      is not required: only the ``shape`` and ``dtype`` attributes are accessed.
      See below for an example. (Note that the duck-typed objects cannot be
      namedtuples because those are treated as standard Python containers.)
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding cotangent. Otherwise, the
      transposed function will be per-example over named axes. For example, if
      ``'batch'`` is a named batch axis, ``linear_transpose(f, *args,
      reduce_axes=('batch',))`` will create a transpose function that sums over
      the batch while ``linear_transpose(f, args)`` will create a per-example
      transpose.

  Returns:
    A callable that calculates the transpose of ``fun``. Valid input into this
    function must have the same shape/dtypes/structure as the result of
    ``fun(*primals)``. Output will be a tuple, with the same
    shape/dtypes/structure as ``primals``.

  >>> import jax
  >>> import types
  >>>
  >>> f = lambda x, y: 0.5 * x - 0.5 * y
  >>> scalar = types.SimpleNamespace(shape=(), dtype=np.dtype(np.float32))
  >>> f_transpose = jax.linear_transpose(f, scalar, scalar)
  >>> f_transpose(1.0)
  (Array(0.5, dtype=float32), Array(-0.5, dtype=float32))
  """
  reduce_axes = _ensure_str_tuple(reduce_axes)
  primals_flat, in_tree = tree_flatten(primals)
  flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  in_avals = map(shaped_abstractify, primals_flat)
  in_dtypes = map(dtypes.dtype, in_avals)

  in_pvals = map(pe.PartialVal.unknown, in_avals)
  jaxpr, out_pvals, const = pe.trace_to_jaxpr_nounits(flat_fun, in_pvals,
                                                      instantiate=True)
  out_avals, _ = unzip2(out_pvals)
  out_dtypes = map(dtypes.dtype, out_avals)
  if not (all(dtypes.issubdtype(d, np.inexact) for d in in_dtypes + out_dtypes)
          or all(dtypes.issubdtype(d, np.integer)
                 for d in in_dtypes + out_dtypes)):
    raise TypeError("linear_transpose only supports [float or complex] -> "
                    "[float or complex], and integer -> integer functions, "
                    f"but got {in_dtypes} -> {out_dtypes}.")

  @api_boundary
  def transposed_fun(const, out_cotangent):
    out_cts, out_tree2 = tree_flatten(out_cotangent)
    if out_tree() != out_tree2:
      raise TypeError("cotangent tree does not match function output, "
                      f"expected {out_tree()} but got {out_tree2}")
    if not all(map(core.typecheck, out_avals, out_cts)):
      raise TypeError("cotangent type does not match function output, "
                      f"expected {out_avals} but got {out_cts}")
    dummies = [ad.UndefinedPrimal(a) for a in in_avals]
    in_cts = ad.backward_pass(jaxpr, reduce_axes, True, const, dummies, out_cts)
    in_cts = map(ad.instantiate_zeros, in_cts)
    return tree_unflatten(in_tree, in_cts)

  # Ensure that transposed_fun is a PyTree
  return Partial(transposed_fun, const)


def make_jaxpr(fun: Callable,
               static_argnums: Union[int, Iterable[int]] = (),
               axis_env: Optional[Sequence[Tuple[AxisName, int]]] = None,
               return_shape: bool = False,
               abstracted_axes: Optional[Any] = None,
               ) -> Callable[..., core.ClosedJaxpr]:
  """Creates a function that produces its jaxpr given example args.

  Args:
    fun: The function whose ``jaxpr`` is to be computed. Its positional
      arguments and return value should be arrays, scalars, or standard Python
      containers (tuple/list/dict) thereof.
    static_argnums: See the :py:func:`jax.jit` docstring.
    axis_env: Optional, a sequence of pairs where the first element is an axis
      name and the second element is a positive integer representing the size of
      the mapped axis with that name. This parameter is useful when lowering
      functions that involve parallel communication collectives, and it
      specifies the axis name/size environment that would be set up by
      applications of :py:func:`jax.pmap`.
    return_shape: Optional boolean, defaults to ``False``. If ``True``, the
      wrapped function returns a pair where the first element is the XLA
      computation and the second element is a pytree with the same structure as
      the output of ``fun`` and where the leaves are objects with ``shape``,
      ``dtype``, and ``named_shape`` attributes representing the corresponding
      types of the output leaves.

  Returns:
    A wrapped version of ``fun`` that when applied to example arguments returns
    a ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
    argument ``return_shape`` is ``True``, then the returned function instead
    returns a pair where the first element is the ``ClosedJaxpr``
    representation of ``fun`` and the second element is a pytree representing
    the structure, shape, dtypes, and named shapes of the output of ``fun``.

  A ``jaxpr`` is JAX's intermediate representation for program traces. The
  ``jaxpr`` language is based on the simply-typed first-order lambda calculus
  with let-bindings. :py:func:`make_jaxpr` adapts a function to return its
  ``jaxpr``, which we can inspect to understand what JAX is doing internally.
  The ``jaxpr`` returned is a trace of ``fun`` abstracted to
  :py:class:`ShapedArray` level. Other levels of abstraction exist internally.

  We do not describe the semantics of the ``jaxpr`` language in detail here, but
  instead give a few examples.

  >>> import jax
  >>>
  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> print(f(3.0))
  -0.83602
  >>> jax.make_jaxpr(f)(3.0)
  { lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
  >>> jax.make_jaxpr(jax.grad(f))(3.0)
  { lambda ; a:f32[]. let
      b:f32[] = cos a
      c:f32[] = sin a
      _:f32[] = sin b
      d:f32[] = cos b
      e:f32[] = mul 1.0 d
      f:f32[] = neg e
      g:f32[] = mul f c
    in (g,) }
  """
  check_callable(fun)
  static_argnums = _ensure_index_tuple(static_argnums)

  def abstractify(args, kwargs):
    flat_args, in_tree = tree_flatten((args, kwargs))
    if abstracted_axes is None:
      return map(shaped_abstractify, flat_args), in_tree, [True] * len(flat_args)
    else:
      axes_specs = _flat_axes_specs(abstracted_axes, *args, **kwargs)
      in_type = pe.infer_lambda_input_type(axes_specs, flat_args)
      in_avals, keep_inputs = unzip2(in_type)
      return in_avals, in_tree, keep_inputs

  @wraps(fun)
  @api_boundary
  def make_jaxpr_f(*args, **kwargs):
    f = lu.wrap_init(fun)
    if static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      f, args = argnums_partial(f, dyn_argnums, args)
    in_avals, in_tree, keep_inputs = abstractify(args, kwargs)
    in_type = tuple(zip(in_avals, keep_inputs))
    f, out_tree = flatten_fun(f, in_tree)
    f = lu.annotate(f, in_type)
    with ExitStack() as stack:
      for axis_name, size in axis_env or []:
        stack.enter_context(core.extend_axis_env(axis_name, size, None))
      jaxpr, out_type, consts = pe.trace_to_jaxpr_dynamic2(f)
    closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    if return_shape:
      out_avals, _ = unzip2(out_type)
      out_shapes_flat = [
          ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
      return closed_jaxpr, tree_unflatten(out_tree(), out_shapes_flat)
    return closed_jaxpr

  make_jaxpr_f.__name__ = f"make_jaxpr({make_jaxpr.__name__})"
  return make_jaxpr_f


def device_put(
    x, device: Union[None, xc.Device, jax.sharding.Sharding, Any] = None):
  """Transfers ``x`` to ``device``.

  Args:
    x: An array, scalar, or (nested) standard Python container thereof.
    device: The (optional) :py:class:`Device`, `Sharding`, or a (nested)
      `Sharding` in standard Python container (must be a tree prefix of ``x``),
      representing the device(s) to which ``x`` should be transferred. If
      given, then the result is committed to the device(s).

  Returns:
    A copy of ``x`` that resides on ``device``.

  If the ``device`` parameter is ``None``, then this operation behaves like the
  identity function if the operand is on any device already, otherwise it
  transfers the data to the default device, uncommitted.

  For more details on data placement see the
  :ref:`FAQ on data placement <faq-data-placement>`.

  This function is always asynchronous, i.e. returns immediately without
  blocking the calling Python thread until any transfers are completed.
  """
  with config_explicit_device_put_scope():
    if device is None or isinstance(device, (xc.Device, jax.sharding.Sharding)):
      return tree_map(lambda y: dispatch.device_put_p.bind(y, device=device), x)

    x_flat, treedef = tree_flatten(x)
    device_flat = flatten_axes("device_put device", treedef, device)
    out_flat = [
        dispatch.device_put_p.bind(y, device=d)
        for y, d in zip(x_flat, device_flat)
    ]
    return tree_unflatten(treedef, out_flat)


def device_put_sharded(shards: Sequence[Any], devices: Sequence[xc.Device]):  # noqa: F811
  """Transfer array shards to specified devices and form ShardedDeviceArray(s).

  Args:
    shards: A sequence of arrays, scalars, or (nested) standard Python
      containers thereof representing the shards to be stacked together to form
      the output. The length of ``shards`` must equal the length of ``devices``.
    devices: A sequence of :py:class:`Device` instances representing the devices
      to which corresponding shards in ``shards`` will be transferred.

  This function is always asynchronous, i.e. returns immediately.

  Returns:
    A ShardedDeviceArray or (nested) Python container thereof representing the
    elements of ``shards`` stacked together, with each shard backed by physical
    device memory specified by the corresponding entry in ``devices``.

  Examples:
    Passing a list of arrays for ``shards`` results in a sharded array
    containing a stacked version of the inputs:

    >>> import jax
    >>> devices = jax.local_devices()
    >>> x = [jax.numpy.ones(5) for device in devices]
    >>> y = jax.device_put_sharded(x, devices)
    >>> np.allclose(y, jax.numpy.stack(x))
    True

    Passing a list of nested container objects with arrays at the leaves for
    ``shards`` corresponds to stacking the shards at each leaf. This requires
    all entries in the list to have the same tree structure:

    >>> x = [(i, jax.numpy.arange(i, i + 4)) for i in range(len(devices))]
    >>> y = jax.device_put_sharded(x, devices)
    >>> type(y)
    <class 'tuple'>
    >>> y0 = jax.device_put_sharded([a for a, b in x], devices)
    >>> y1 = jax.device_put_sharded([b for a, b in x], devices)
    >>> np.allclose(y[0], y0)
    True
    >>> np.allclose(y[1], y1)
    True

  See Also:
    - device_put
    - device_put_replicated
  """
  # TODO(jakevdp): provide a default for devices that considers both local
  # devices and pods
  if not isinstance(shards, Sequence):
    raise ValueError("device_put_sharded `shards` input must be a sequence; "
                     f"got {type(shards)}")
  if len(shards) != len(devices):
    raise ValueError(f"len(shards) = {len(shards)} must equal "
                     f"len(devices) = {len(devices)}.")

  def _device_put_sharded(*xs):
    avals = [core.raise_to_shaped(core.get_aval(x)) for x in xs]
    if not all(a1 == a2 for a1, a2 in zip(avals[:-1], avals[1:])):
      a1, a2 = next((a1, a2) for a1, a2 in zip(avals[:-1], avals[1:])
                    if a1 != a2)
      raise ValueError("the shards passed to device_put_sharded must have "
                       f"consistent shape and dtype, but got {a1} and {a2}.")
    stacked_aval = avals[0].update(shape=(len(devices),) + avals[0].shape)
    buffers = [buf for x, d in zip(xs, devices)
               for buf in dispatch.device_put(x, d)]
    if config.jax_array:
      sharding_spec = pxla._create_pmap_sharding_spec(stacked_aval)
      return array.ArrayImpl(
          stacked_aval,
          PmapSharding(np.array(devices), sharding_spec),
          buffers, committed=True, _skip_checks=True)
    else:
      return pxla.make_sharded_device_array(stacked_aval, None, buffers)

  with config_explicit_device_put_scope():
    return tree_map(_device_put_sharded, *shards)


def device_put_replicated(x: Any, devices: Sequence[xc.Device]):  # noqa: F811
  """Transfer array(s) to each specified device and form ShardedDeviceArray(s).

  Args:
    x: an array, scalar, or (nested) standard Python container thereof
      representing the array to be replicated to form the output.
    devices: A sequence of :py:class:`Device` instances representing the devices
      to which ``x`` will be transferred.

  This function is always asynchronous, i.e. returns immediately.

  Returns:
    A ShardedDeviceArray or (nested) Python container thereof representing the
    value of ``x`` broadcasted along a new leading axis of size
    ``len(devices)``, with each slice along that new leading axis backed by
    memory on the device specified by the corresponding entry in ``devices``.

  Examples:
    Passing an array:

    >>> import jax
    >>> devices = jax.local_devices()
    >>> x = jax.numpy.array([1., 2., 3.])
    >>> y = jax.device_put_replicated(x, devices)
    >>> np.allclose(y, jax.numpy.stack([x for _ in devices]))
    True

  See Also:
    - device_put
    - device_put_sharded
  """
  if not isinstance(devices, Sequence) or not devices:
    raise ValueError("`devices` argument to `device_put_replicated must be "
                     "a non-empty sequence.")
  def _device_put_replicated(x):
    aval = core.unmapped_aval(len(devices), core.no_axis_name, 0,
                              core.raise_to_shaped(core.get_aval(x)))
    assert (isinstance(aval, ShapedArray) and
            len(xla.aval_to_xla_shapes(aval)) == 1)
    buf, = dispatch.device_put(x, devices[0])
    rest_bufs = [buf.copy_to_device(d) for d in devices[1:]]
    if config.jax_array:
      sharding_spec = pxla._create_pmap_sharding_spec(aval)
      return array.ArrayImpl(
          aval, PmapSharding(np.array(devices), sharding_spec),
          [buf, *rest_bufs], committed=True, _skip_checks=True)
    else:
      return pxla.make_sharded_device_array(aval, None, [buf, *rest_bufs])

  with config_explicit_device_put_scope():
    return tree_map(_device_put_replicated, x)


# TODO(mattjj): consider revising
def _device_get(x):
  if isinstance(x, core.Tracer):
    return x
  try:
    toarray = x.__array__
  except AttributeError:
    return x
  else:
    return toarray()

def device_get(x: Any):
  """Transfer ``x`` to host.

  If ``x`` is a pytree, then the individual buffers are copied in parallel.

  Args:
    x: An array, scalar, Array or (nested) standard Python container thereof
      representing the array to be transferred to host.

  Returns:
    An array or (nested) Python container thereof representing the
    value of ``x``.

  Examples:
    Passing a Array:

    >>> import jax
    >>> x = jax.numpy.array([1., 2., 3.])
    >>> jax.device_get(x)
    array([1., 2., 3.], dtype=float32)

    Passing a scalar (has no effect):

    >>> jax.device_get(1)
    1

  See Also:
    - device_put
    - device_put_sharded
    - device_put_replicated
  """
  with config_explicit_device_get_scope():
    for y in tree_leaves(x):
      try:
        y.copy_to_host_async()
      except AttributeError:
        pass
    return tree_map(_device_get, x)


class ShapeDtypeStruct:
  __slots__ = ["shape", "dtype", "named_shape", "sharding"]
  def __init__(self, shape, dtype, named_shape=None, sharding=None):
    self.shape = tuple(shape)
    if dtype is None:
      raise ValueError("ShapeDtypeStruct: dtype must be specified.")
    self.dtype = dtype if core.is_opaque_dtype(dtype) else np.dtype(dtype)
    if sharding is not None:
      if not isinstance(sharding, jax.sharding.Sharding):
        raise ValueError(
            "sharding should be an instance of `jax.sharding.Sharding`. "
            f"Got {sharding} of type {type(sharding)}.")
      self.sharding = sharding
    self.named_shape = {} if named_shape is None else dict(named_shape)

  size = property(lambda self: prod(self.shape))
  ndim = property(lambda self: len(self.shape))

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError as e:
      raise TypeError("len() of unsized object") from e # same as numpy error

  def __repr__(self):
    ns = f", named_shape={self.named_shape}" if self.named_shape else ""
    sh = f", sharding={self.sharding}" if hasattr(self, "sharding") else ""
    return (f"{type(self).__name__}(shape={self.shape}, "
            f"dtype={self.dtype.name}{ns}{sh})")

  __str__ = __repr__

  def __eq__(self, other):
    if not isinstance(other, ShapeDtypeStruct):
      return False
    else:
      other_sh = other.sharding if hasattr(other, "sharding") else None
      sh = self.sharding if hasattr(self, "sharding") else None
      return ((other.shape, other.dtype, other.named_shape, other_sh) ==
              (self.shape, self.dtype, self.named_shape, sh))

  def __hash__(self):
    # TODO(frostig): avoid the conversion from dict by addressing
    # https://github.com/google/jax/issues/8182
    named = frozenset(self.named_shape.items())
    sh = self.sharding if hasattr(self, "sharding") else None
    return hash((self.shape, self.dtype, named, sh))

core.pytype_aval_mappings[ShapeDtypeStruct] = (
    lambda x: ShapedArray(x.shape, dtypes.canonicalize_dtype(x.dtype),
                          weak_type=False, named_shape=x.named_shape))

def eval_shape(fun: Callable, *args, **kwargs):
  """Compute the shape/dtype of ``fun`` without any FLOPs.

  This utility function is useful for performing shape inference. Its
  input/output behavior is defined by::

    def eval_shape(fun, *args, **kwargs):
      out = fun(*args, **kwargs)
      return jax.tree_util.tree_map(shape_dtype_struct, out)

    def shape_dtype_struct(x):
      return ShapeDtypeStruct(x.shape, x.dtype)

    class ShapeDtypeStruct:
      __slots__ = ["shape", "dtype"]
      def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

  In particular, the output is a pytree of objects that have ``shape`` and
  ``dtype`` attributes, but nothing else about them is guaranteed by the API.

  But instead of applying ``fun`` directly, which might be expensive, it uses
  JAX's abstract interpretation machinery to evaluate the shapes without doing
  any FLOPs.

  Using :py:func:`eval_shape` can also catch shape errors, and will raise same
  shape errors as evaluating ``fun(*args, **kwargs)``.

  Args:
    fun: The function whose output shape should be evaluated.
    *args: a positional argument tuple of arrays, scalars, or (nested) standard
      Python containers (tuples, lists, dicts, namedtuples, i.e. pytrees) of
      those types. Since only the ``shape`` and ``dtype`` attributes are
      accessed, only values that duck-type arrays are required, rather than real
      ndarrays. The duck-typed objects cannot be namedtuples because those are
      treated as standard Python containers. See the example below.
    **kwargs: a keyword argument dict of arrays, scalars, or (nested) standard
      Python containers (pytrees) of those types. As in ``args``, array values
      need only be duck-typed to have ``shape`` and ``dtype`` attributes.

  For example:

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> f = lambda A, x: jnp.tanh(jnp.dot(A, x))
  >>> class MyArgArray(object):
  ...   def __init__(self, shape, dtype):
  ...     self.shape = shape
  ...     self.dtype = jnp.dtype(dtype)
  ...
  >>> A = MyArgArray((2000, 3000), jnp.float32)
  >>> x = MyArgArray((3000, 1000), jnp.float32)
  >>> out = jax.eval_shape(f, A, x)  # no FLOPs performed
  >>> print(out.shape)
  (2000, 1000)
  >>> print(out.dtype)
  float32
  """
  args_flat, in_tree = tree_flatten((args, kwargs))
  wrapped_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
  debug_info = pe.debug_info(fun, in_tree, True, "eval_shape")
  out = pe.abstract_eval_fun(wrapped_fun.call_wrapped,
                             *map(shaped_abstractify, args_flat),
                             debug_info=debug_info)
  out = [ShapeDtypeStruct(x.shape, x.dtype, x.named_shape) for x in out]
  return tree_unflatten(out_tree(), out)


@functools.wraps(new_checkpoint)  # config.jax_new_checkpoint is True by default
def checkpoint(fun: Callable, *,
               concrete: bool = False,
               prevent_cse: bool = True,
               static_argnums: Union[int, Tuple[int, ...]] = (),
               policy: Optional[Callable[..., bool]] = None,
               ) -> Callable:
  if concrete:
    msg = ("The 'concrete' option to jax.checkpoint / jax.remat is deprecated; "
           "in its place, you can use its `static_argnums` option, and if "
           "necessary the `jax.ensure_compile_time_eval()` context manager.\n"
           "\n"
           "For example, if using `concrete=True` for an `is_training` flag:\n"
           "\n"
           "  from functools import partial\n"
           "\n"
           "  @partial(jax.checkpoint, concrete=True)\n"
           "  def foo(x, is_training):\n"
           "    if is_training:\n"
           "      return f(x)\n"
           "    else:\n"
           "      return g(x)\n"
           "\n"
           "replace it with a use of `static_argnums`:\n"
           "\n"
           "  @partial(jax.checkpoint, static_argnums=(1,))\n"
           "  def foo(x, is_training):\n"
           "    ...\n"
           "\n"
           "If jax.numpy operations need to be performed on static arguments, "
           "we can use the `jax.ensure_compile_time_eval()` context manager. "
           "For example, we can replace this use of `concrete=True`\n:"
           "\n"
           "  @partial(jax.checkpoint, concrete=True)\n"
           "  def foo(x, y):\n"
           "    if y > 0:\n"
           "      return f(x)\n"
           "    else:\n"
           "      return g(x)\n"
           "\n"
           "with this combination of `static_argnums` and "
           "`jax.ensure_compile_time_eval()`:\n"
           "\n"
           "  @partial(jax.checkpoint, static_argnums=(1,))\n"
           "  def foo(x, y):\n"
           "    with jax.ensure_compile_time_eval():\n"
           "      y_pos = y > 0\n"
           "    if y_pos:\n"
           "      return f(x)\n"
           "    else:\n"
           "      return g(x)\n"
           "\n"
           "See https://jax.readthedocs.io/en/latest/jep/11830-new-remat-checkpoint.html\n")
    raise NotImplementedError(msg)
  return new_checkpoint(fun, prevent_cse=prevent_cse, policy=policy,
                        static_argnums=static_argnums)
remat = checkpoint  # type: ignore


def named_call(
    fun: Callable[..., Any],
    *,
    name: Optional[str] = None,
  ) -> Callable[..., Any]:
  """Adds a user specified name to a function when staging out JAX computations.

  When staging out computations for just-in-time compilation to XLA (or other
  backends such as TensorFlow) JAX runs your Python program but by default does
  not preserve any of the function names or other metadata associated with it.
  This can make debugging the staged out (and/or compiled) representation of
  your program complicated because there is limited context information for each
  operation being executed.

  `named_call` tells JAX to stage the given function out as a subcomputation
  with a specific name. When the staged out program is compiled with XLA these
  named subcomputations are preserved and show up in debugging utilities like
  the TensorFlow Profiler in TensorBoard. Names are also preserved when staging
  out JAX programs to TensorFlow using :func:`experimental.jax2tf.convert`.

  Args:
    fun: Function to be wrapped. This can be any Callable.
    name: Optional. The prefix to use to name all sub computations created
      within the name scope. Use the fun.__name__ if not specified.

  Returns:
    A version of `fun` that is wrapped in a name_scope.
  """
  if name is None:
    name = fun.__name__

  return source_info_util.extend_name_stack(name)(fun)

@contextmanager
def named_scope(
    name: str,
  ) -> Generator[None, None, None]:
  """A context manager that adds a user specified name to the JAX name stack.

  When staging out computations for just-in-time compilation to XLA (or other
  backends such as TensorFlow) JAX does not, by default, preserve the names
  (or other source metadata) of Python functions it encounters.
  This can make debugging the staged out (and/or compiled) representation of
  your program complicated because there is limited context information for each
  operation being executed.

  ``named_scope`` tells JAX to stage the given function with additional
  annotations on the underlying operations. JAX internally keeps track of these
  annotations in a name stack. When the staged out program is compiled with XLA
  these annotations are preserved and show up in debugging utilities like the
  TensorFlow Profiler in TensorBoard. Names are also preserved when staging out
  JAX programs to TensorFlow using :func:`experimental.jax2tf.convert`.


  Args:
    name: The prefix to use to name all operations created within the name
      scope.
  Yields:
    Yields ``None``, but enters a context in which `name` will be appended to
    the active name stack.

  Examples:
    ``named_scope`` can be used as a context manager inside compiled functions:

    >>> import jax
    >>>
    >>> @jax.jit
    ... def layer(w, x):
    ...   with jax.named_scope("dot_product"):
    ...     logits = w.dot(x)
    ...   with jax.named_scope("activation"):
    ...     return jax.nn.relu(logits)

    It can also be used as a decorator:

    >>> @jax.jit
    ... @jax.named_scope("layer")
    ... def layer(w, x):
    ...   logits = w.dot(x)
    ...   return jax.nn.relu(logits)
  """
  if not isinstance(name, str):
    raise ValueError("named_scope name argument must be a string.")
  with source_info_util.extend_name_stack(name):
    yield

def effects_barrier():
  """Waits until existing functions have completed any side-effects."""
  dispatch.runtime_tokens.block_until_ready()

def block_until_ready(x):
  """
  Tries to call a ``block_until_ready`` method on pytree leaves.

  Args:
    x: a pytree, usually with at least some JAX array instances at its leaves.

  Returns:
    A pytree with the same structure and values of the input, where the values
    of all JAX array leaves are ready.
  """
  def try_to_block(x):
    try:
      return x.block_until_ready()
    except AttributeError:
      return x
  return jax.tree_util.tree_map(try_to_block, x)

def pure_callback(callback: Callable[..., Any], result_shape_dtypes: Any,
                  *args: Any, vectorized: bool = False, **kwargs: Any):
  """Applies a functionally pure Python callable. Works under :func:`jit`/:func:`~pmap`/etc.

  ``pure_callback`` enables calling a Python function in JIT-ed JAX functions.
  The input ``callback`` will be passed NumPy arrays in place of JAX arrays and
  should also return NumPy arrays. Execution takes place on CPU, like any
  Python+NumPy function.

  The callback is treated as functionally pure, meaning it has no side-effects
  and its output value depends only on its argument values. As a consequence, it
  is safe to be called multiple times (e.g. when transformed by :func:`~vmap` or
  :func:`~pmap`), or not to be called at all when e.g. the output of a
  `jit`-decorated function has no data dependence on its value. Pure callbacks
  may also be reordered if data-dependence allows.

  When :func:`~pmap`-ed, the pure callback will be called several times (one on each
  axis of the map). When `vmap`-ed the behavior will depend on the value of the
  ``vectorized`` keyword argument. When ``vectorized`` is ``True``, the callback
  is assumed to obey
  ``jax.vmap(callback)(xs) == callback(xs) == jnp.stack([callback(x) for x in xs])``.
  Therefore, the callback will be called directly on batched inputs (where the
  batch axes are the leading dimensions). Additionally, the callbacks should
  return outputs that have corresponding leading batch axes. If not vectorized
  ``callback`` will be mapped sequentially across the batched axis.
  For example, if ``callback = lambda x, y: np.matmul(x, y)``, then we are free
  to set ``vectorized=True`` because the ``np.matmul`` function handles
  arbitrary leading batch dimensions.

  Args:
    callback: A Python callable. The callable will be passed PyTrees of NumPy
      arrays as arguments, and should return a PyTree of NumPy arrays that
      matches ``result_shape_dtypes``.
    result_shape_dtypes: A PyTree with leaves that are objects with ``shape``
      and ``dtype`` attributes which represent to the shapes and dtypes of the
      value of ``callback`` applied to ``args`` and ``kwargs``.
    *args: The positional arguments to the callback. Must be PyTrees of JAX
      types.
    vectorized: A boolean that indicates whether or not ``callback`` is
      vectorized, meaning it can handle arrays with additional leading
      dimensions. If ``vectorized`` is `True`, when the callback is mapped
      via `jax.vmap`, it will be called directly on inputs with leading batch
      dimensions instead of executing ``callback`` on each mapped input
      individually. The callback should also return outputs batched across the
      leading axis. By default, ``vectorized`` is ``False``.
    **kwargs: The keyword arguments to the callback. Must be PyTrees of JAX
      types.

  Returns:
    The value of ``callback(*args, **kwargs)``.
  """
  return jcb.pure_callback(callback, result_shape_dtypes, *args,
                           vectorized=vectorized, **kwargs)

def clear_backends():
  """
  Clear all backend clients so that new backend clients can be created later.
  """
  xb._clear_backends()
  jax.lib.xla_bridge._backends = {}
  dispatch.xla_callable.cache_clear()  # type: ignore
  dispatch.xla_primitive_callable.cache_clear()
  _cpp_jit_cache.clear()
  jax_jit.CompiledFunctionCache.clear_all()
  pjit._pjit_lower_cached.cache_clear()
  pjit._create_pjit_jaxpr.cache_clear()
  if xla_extension_version >= 124:
    pjit._cpp_pjit_cache.clear()
    xc._xla.PjitFunctionCache.clear_all()

def live_arrays(platform=None):
  """Return all live arrays in the backend for `platform`.

  If platform is None, it is the default backend.
  """
  return xb.get_backend(platform).live_arrays()
