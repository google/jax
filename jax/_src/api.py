# coding=utf-8
# Copyright 2018 Google LLC
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

# flake8: noqa: F401
import collections
import functools
from functools import partial
import inspect
import itertools as it
import sys
import threading
import weakref
import types
from typing import (Any, Callable, Iterable, NamedTuple, Mapping, Optional,
                    Sequence, Tuple, TypeVar, Union, overload)
from warnings import warn

import numpy as np
from contextlib import contextmanager, ExitStack

import jax
from jax import core
from jax import linear_util as lu
from jax._src import dtypes
from jax.core import eval_jaxpr
from jax._src.api_util import (
    flatten_fun, apply_flat_fun, flatten_fun_nokwargs, flatten_fun_nokwargs2,
    argnums_partial, argnums_partial_except, flatten_axes, donation_vector,
    rebase_donate_argnums, _ensure_index, _ensure_index_tuple,
    shaped_abstractify, _ensure_str_tuple, argnames_partial_except)
from jax._src import traceback_util
from jax._src.traceback_util import api_boundary
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           tree_structure, tree_transpose, tree_leaves,
                           tree_multimap, treedef_is_leaf, treedef_children,
                           Partial, PyTreeDef)
from jax._src.util import (unzip2, curry, safe_map, safe_zip, prod, split_list,
                           extend_name_stack, wrap_name, cache, wraps,
                           HashableFunction)
from jax._src import device_array
from jax._src import dispatch
from jax._src.lib import jax_jit
from jax._src.lib import version
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.lib import pmap_lib
# Unused imports to be exported
from jax._src.lib.xla_bridge import (device_count, local_device_count, devices,
                                     local_devices, process_index,
                                     process_count, host_id, host_ids,
                                     host_count, default_backend)
from jax.core import ShapedArray, raise_to_shaped
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import masking
from jax.interpreters import invertible_ad as iad
from jax.interpreters.invertible_ad import custom_ivjp
from jax.custom_derivatives import (closure_convert, custom_gradient, custom_jvp,
                                    custom_vjp, linear_call)
from jax.ad_checkpoint import checkpoint_policies

from jax._src.config import (flags, config, bool_env,
                             disable_jit as _disable_jit,
                             debug_nans as config_debug_nans,
                             debug_infs as config_debug_infs,
                             _thread_local_state as config_thread_local_state)


traceback_util.register_exclusion(__file__)

_dtype = partial(dtypes.dtype, canonicalize=True)

AxisName = Any

# These TypeVars are used below to express the fact that function types
# (i.e. call signatures) are invariant under the jit, vmap, and pmap
# transformations.
# Note that the function type annotations will generally not strictly hold
# in JIT internals, as Tracer values are passed through the function.
# Should this raise any type errors for the tracing code in future, we can disable
# type checking in parts of the tracing code, or remove these annotations.
F = TypeVar("F", bound=Callable)
T = TypeVar("T")
U = TypeVar("U")

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "experimental_cpp_jit", bool_env("JAX_CPP_JIT", True),
    "A flag enabling the C++ jax.jit fast path."
    "Set this to `False` only if it crashes otherwise and report "
    "the error to the jax-team.")
flags.DEFINE_bool(
    "experimental_cpp_pmap", bool_env("JAX_CPP_PMAP", True),
    "A flag enabling the C++ jax.pmap fast path. Until the default "
    "is switched to True, the feature is not supported and possibly broken "
    "(e.g. it may use unreleased code from jaxlib.")


def _nan_check_posthook(fun, args, kwargs, output):
  """Hook function called by the C++ jit/pmap to perform NaN checking."""
  leaves = tree_leaves(output)

  buffers = []
  for da_or_sda in leaves:
    if hasattr(da_or_sda, "device_buffer"):
      buffers.append(da_or_sda.device_buffer)
    elif hasattr(da_or_sda, "device_buffers"):
      buffers.extend(da_or_sda.device_buffers)

  try:
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

def _check_callable(fun):
  # In Python 3.10+, the only thing stopping us from supporting staticmethods
  # is that we can't take weak references to them, which the C++ JIT requires.
  if isinstance(fun, staticmethod):
    raise TypeError(f"staticmethod arguments are not supported, got {fun}")
  if not callable(fun):
    raise TypeError(f"Expected a callable value, got {fun}")
  if _isgeneratorfunction(fun):
    raise TypeError(f"Expected a function, got a generator function: {fun}")

def _isgeneratorfunction(fun):
  # re-implemented here because of https://bugs.python.org/issue33261
  while inspect.ismethod(fun):
    fun = fun.__func__
  while isinstance(fun, functools.partial):
    fun = fun.func
  return inspect.isfunction(fun) and bool(fun.__code__.co_flags & inspect.CO_GENERATOR)

_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD

def _infer_argnums_and_argnames(
    fun: Callable,
    argnums: Union[int, Iterable[int], None],
    argnames: Union[str, Iterable[str], None],
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
  """Infer missing argnums and argnames for a function with inspect."""
  if argnums is None and argnames is None:
    argnums = ()
    argnames = ()
  elif argnums is not None and argnames is not None:
    argnums = _ensure_index_tuple(argnums)
    argnames = _ensure_str_tuple(argnames)
  else:
    try:
      signature = inspect.signature(fun)
    except ValueError:
      # In rare cases, inspect can fail, e.g., on some builtin Python functions.
      # In these cases, don't infer any parameters.
      parameters: Mapping[str, inspect.Parameter] = {}
    else:
      parameters = signature.parameters
    if argnums is None:
      assert argnames is not None
      argnames = _ensure_str_tuple(argnames)
      argnums = tuple(
          i for i, (k, param) in enumerate(parameters.items())
          if param.kind == _POSITIONAL_OR_KEYWORD and k in argnames
      )
    else:
      assert argnames is None
      argnums = _ensure_index_tuple(argnums)
      argnames = tuple(
          k for i, (k, param) in enumerate(parameters.items())
          if param.kind == _POSITIONAL_OR_KEYWORD and i in argnums
      )
  return argnums, argnames


def jit(
  fun: F,
  *,
  static_argnums: Union[int, Iterable[int], None] = None,
  static_argnames: Union[str, Iterable[str], None] = None,
  device: Optional[xc.Device] = None,
  backend: Optional[str] = None,
  donate_argnums: Union[int, Iterable[int]] = (),
  inline: bool = False,
) -> F:
  """Sets up ``fun`` for just-in-time compilation with XLA.

  Args:
    fun: Function to be jitted. Should be a pure function, as side-effects may
      only be executed once. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.
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
      ``static_argnames`` is, or vice versa, JAX uses ``inspect.signature(fun)``
      to find any positional arguments that correspond to ``static_argnames``
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
    donate_argnums: Specify which arguments are "donated" to the computation.
      It is safe to donate arguments if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to. By default, no arguments are donated.
    inline: Specify whether this function should be inlined into enclosing
      jaxprs (rather than being represented as an application of the xla_call
      primitive with its own subjaxpr). Default False.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation.

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
  """
  if FLAGS.experimental_cpp_jit:
    return _cpp_jit(fun, static_argnums, static_argnames, device, backend,
                    donate_argnums, inline)
  else:
    return _python_jit(fun, static_argnums, static_argnames, device, backend,
                       donate_argnums, inline)


def _prepare_jit(fun, static_argnums, static_argnames, donate_argnums,
                 args, kwargs):
  if max(donate_argnums, default=-1) >= len(args):
    raise ValueError(
        f"jitted function has donate_argnums={donate_argnums} but "
        f"was called with only {len(args)} positional arguments.")

  f = lu.wrap_init(fun)
  f, args = argnums_partial_except(f, static_argnums, args, allow_invalid=True)
  f, kwargs = argnames_partial_except(f, static_argnames, kwargs)
  args_flat, in_tree = tree_flatten((args, kwargs))
  if donate_argnums:
    donated_invars = donation_vector(donate_argnums, args, kwargs)
  else:
    donated_invars = (False,) * len(args_flat)

  return f, in_tree, args_flat, donated_invars


def _python_jit(
    fun: F,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    inline: bool = False,
) -> F:
  # The Python implementation of `jax.jit`, being slowly replaced by _cpp_jit.
  _check_callable(fun)
  static_argnums, static_argnames = _infer_argnums_and_argnames(
      fun, static_argnums, static_argnames)
  static_argnums = _ensure_index_tuple(static_argnums)
  donate_argnums = _ensure_index_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  @wraps(fun)
  @api_boundary
  def f_jitted(*args, **kwargs):
    if config.jax_disable_jit:
      return fun(*args, **kwargs)
    closed_fun, in_tree, args_flat, donated_invars = _prepare_jit(
        fun, static_argnums, static_argnames, donate_argnums, args, kwargs)
    for arg in args_flat:
      _check_arg(arg)
    flat_fun, out_tree = flatten_fun(closed_fun, in_tree)
    out_flat = xla.xla_call(
        flat_fun, *args_flat,
        device=device, backend=backend, name=flat_fun.__name__,
        donated_invars=donated_invars, inline=inline)
    return tree_unflatten(out_tree(), out_flat)

  f_jitted.lower = _jit_lower(fun, static_argnums, static_argnames, device,
                              backend, donate_argnums, inline)
  return f_jitted


class _BackendAndDeviceInfo(NamedTuple):
  default_device: xc.Device
  committed_to_device: bool

class _FastpathData(NamedTuple):
  xla_executable: xla.XlaExecutable
  out_pytree_def: Any
  sticky_device: xc.Device
  avals: Iterable[Any]
  lazy_exprs: Iterable[Any]
  kept_var_bitvec: Iterable[bool]

_cpp_jit_cache = jax_jit.CompiledFunctionCache()

def _cpp_jit(
    fun: F,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    inline: bool = False,
) -> F:
  # An implementation of `jit` that tries to do as much as possible in C++.
  # The goal of this function is to speed up the time it takes to process the
  # arguments, find the correct C++ executable, start the transfer of arguments
  # and schedule the computation.
  # As long as it does not support all features of the Python implementation
  # the C++ code will fallback to `_python_jit` when it faces some unsupported
  # feature.
  _check_callable(fun)
  static_argnums, static_argnames = _infer_argnums_and_argnames(
      fun, static_argnums, static_argnames)
  static_argnums = _ensure_index_tuple(static_argnums)
  donate_argnums = _ensure_index_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  if device is not None and backend is not None:
    raise ValueError("can't specify both a device and a backend for jit, "
                     f"got device={device} and backend={backend}.")

  @api_boundary
  def cache_miss(*args, **kwargs):
    ### This first part is basically the same code as in _python_jit.
    # An alternative would be for cache_miss to accept from C++ the arguments
    # (dyn_args, donated_invars, args_flat, in_tree), since otherwise we have
    # work/code that is redundant between C++ and Python. We can try that later.
    closed_fun, in_tree, args_flat, donated_invars = _prepare_jit(
        fun, static_argnums, static_argnames, donate_argnums, args, kwargs)
    for arg in args_flat:
      _check_arg(arg)
    flat_fun, out_tree = flatten_fun(closed_fun, in_tree)
    out_flat = xla.xla_call(
        flat_fun, *args_flat,
        device=device, backend=backend, name=flat_fun.__name__,
        donated_invars=donated_invars, inline=inline)
    out_pytree_def = out_tree()
    out = tree_unflatten(out_pytree_def, out_flat)

    ### Decide whether we can support the C++ fast path
    # High level note: The Python tracing mechanism is complex; in particular
    # to know whether `jax.jit(f)(x)` will execute or trace, it's not enough to
    # inspect the argument x, we actually do need to execute it and look at the
    # outputs that could be tracers (if f is capturing `Tracer` by closure).
    execute: Optional[functools.partial] = (
        dispatch._xla_callable.most_recent_entry())
    use_fastpath = (
        # This is if we have already executed this code-path (most-recent entry
        # has been reset to None). Thus, we do not support the fast-path.
        execute is not None and
        execute.func is dispatch._execute_compiled and  # not trivial, not pmap
        # Not supported: ShardedDeviceArray
        all(device_array.type_is_device_array(x) for x in out_flat))
    ### If we can use the fastpath, we return required info to the caller.
    if use_fastpath:
      _, xla_executable, _, result_handlers, kept_var_idx = execute.args
      sticky_device = None
      avals = []
      lazy_exprs = [None] * len(result_handlers)
      for result_handler in result_handlers:
        aval, sticky_device = result_handler.args
        avals.append(aval)
      assert len(avals) == len(out_flat)
      kept_var_bitvec = [i in kept_var_idx for i in range(len(args_flat))]
      fastpath_data = _FastpathData(xla_executable, out_pytree_def,
                                    sticky_device, avals, lazy_exprs,
                                    kept_var_bitvec)
    else:
      fastpath_data = None

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

  cpp_jitted_f = jax_jit.jit(fun, cache_miss, get_device_info,
                             static_argnums=static_argnums,
                             static_argnames=static_argnames,
                             donate_argnums=donate_argnums,
                             cache=_cpp_jit_cache)
  f_jitted = wraps(fun)(cpp_jitted_f)

  f_jitted.lower = _jit_lower(fun, static_argnums, static_argnames, device,
                              backend, donate_argnums, inline)

  return f_jitted


class Lowered:
  """Lowering of a function specialized to argument types and values.

  A lowering is a computation ready for compilation. This class
  carries a lowering together with the remaining information needed to
  later compile and execute it. It also provides a common API for
  querying properties of lowered computations across JAX's various
  lowering paths (``jit``, ``pmap``, etc.).
  """
  __slots__ = ['in_tree', 'out_tree', 'donate_argnums', '_lowering',
               '_no_kwargs']

  in_tree: PyTreeDef
  out_tree: PyTreeDef
  donate_argnums: Tuple[int]
  _lowering: Union[dispatch.XlaComputation,
                   pxla.MeshComputation,
                   pxla.PmapComputation]
  _no_kwargs: bool

  def __init__(self, lowering, in_tree, out_tree, donate_argnums,
               no_kwargs=False):
    self._lowering = lowering
    self.in_tree = in_tree
    self.out_tree = out_tree
    self.donate_argnums = donate_argnums
    self._no_kwargs = no_kwargs

  def compile(self) -> 'Compiled':
    return Compiled(
        self._lowering.compile(), self.in_tree, self.out_tree,
        self.donate_argnums, self._no_kwargs)

  def compiler_ir(self, dialect: Optional[str] = None):
    if dialect == "mhlo":
      return self._lowering.mhlo()
    elif dialect == "hlo" or dialect is None:
      return self._lowering.hlo()
    else:
      raise ValueError(f"Unknown dialect {dialect}")

  # TODO(frostig): remove this in favor of `compiler_ir`
  def _xla_computation(self):
    return self._lowering.hlo()


class Compiled:
  """Compiled representation of a function specialized to types/values.

  A compiled computation is associated with an executable and the
  remaining information needed to execute it. It also provides a
  common API for querying properties of compiled computations across
  JAX's various compilation paths and backends.
  """
  __slots__ = ['in_tree', 'out_tree', 'donate_argnums', '_executable',
               '_no_kwargs']

  in_tree: PyTreeDef
  out_tree: PyTreeDef
  donate_argnums: Tuple[int]
  _executable: Union[dispatch.XlaCompiledComputation,
                     pxla.MeshExecutable,
                     pxla.PmapExecutable]
  _no_kwargs: bool

  def __init__(self, executable, in_tree, out_tree, donate_argnums,
               no_kwargs=False):
    self._executable = executable
    self.in_tree = in_tree
    self.out_tree = out_tree
    self.donate_argnums = donate_argnums
    self._no_kwargs = no_kwargs

  def compiler_ir(self):
    """Post-compilation IR.

    Compilation typically involves code transformation and
    optimization. This method exists to reflect the compiler's
    representation of the program after such passes, whenever
    possible.
    """
    return self._executable.xla_executable().hlo_modules()

  def runtime_executable(self):
    return self._executable.xla_executable()

  def _xla_executable(self):
    # TODO(frostig): finalize API. For now, return the underlying
    # executable directly via this method.
    return self._executable.xla_executable()

  def __call__(self, *args, **kwargs):
    if self._no_kwargs:
      if kwargs:
        kws = ', '.join(kwargs.keys())
        raise NotImplementedError(
            'function was compiled by a transformation that does not support '
            f'keyword arguments, but called with keyword arguments: {kws}')
      args_flat, in_tree = tree_flatten(args)
    else:
      args_flat, in_tree = tree_flatten((args, kwargs))
    if in_tree != self.in_tree:
      # TODO(frostig): provide more info about the source function
      # and transformation
      raise TypeError(
          f'function compiled for {self.in_tree}, called with {in_tree}')
    try:
      out_flat = self._executable.call(*args_flat)
    except TypeError as e:
      # We can't transform ahead-of-time compiled calls, since we've
      # lowered and compiled for a fixed function signature, and JAX
      # transformations change signatures. We interpret a Tracer
      # argument as an indication of a transformation attempt. We
      # could check this before the executable call, but we'd rather
      # avoid isinstance checks on the call path. Seeing a TypeError
      # might mean that arguments have JAX-invalid types, which in
      # turn might mean some are Tracers.
      for arg in args_flat:
        if isinstance(arg, core.Tracer):
          raise TypeError(
              'Cannot apply JAX transformations to a function lowered and '
              'compiled for a particular signature. Detected argument of '
              f'Tracer type {type(arg)}.')
      else:
        raise
    return tree_unflatten(self.out_tree, out_flat)


def _jit_lower(fun, static_argnums, static_argnames, device, backend,
               donate_argnums, inline):
  """Make a ``lower`` method for jitted functions."""
  # If the function we returned from ``jit`` were a class instance,
  # this might naturally be a method, with ``fun`` as a ``self`` and
  # all the other arguments stored as attributes.

  def arg_spec(x):
    # like xla.arg_spec but duck-types on x.shape and x.dtype
    aval = shaped_abstractify(x)
    try:
      return aval, x._device
    except:
      return aval, None

  @api_boundary
  def lower(*args, **kwargs) -> Lowered:
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
    name = flat_fun.__name__
    arg_specs = unsafe_map(arg_spec, args_flat)
    computation = dispatch.lower_xla_callable(
        flat_fun, device, backend, name, donated_invars, *arg_specs)
    return Lowered(computation, in_tree, out_tree(), donate_argnums)

  return lower


@contextmanager
def disable_jit():
  """Context manager that disables :py:func:`jit` behavior under its dynamic context.

  For debugging it is useful to have a mechanism that disables :py:func:`jit`
  everywhere in a dynamic context.

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
  >>> print(f(jax.numpy.array([1, 2, 3])))
  Value of y is Traced<ShapedArray(int32[3])>with<DynamicJaxprTrace(level=0/1)>
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
  with _disable_jit(True):
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

  _check_callable(fun)
  static_argnums = _ensure_index_tuple(static_argnums)
  donate_argnums = _ensure_index_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  fun_name = getattr(fun, "__name__", "unknown")

  backend = backend if backend is not None else xb.get_backend().platform

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
      raise ValueError(f"jitted function has static_argnums={static_argnums},"
                       f" donate_argnums={donate_argnums} but "
                       f"was called with only {len(args)} positional arguments.")

    f = lu.wrap_init(fun)
    if static_argnums:
      f, dyn_args = argnums_partial_except(f, static_argnums, args, allow_invalid=False)
    else:
      dyn_args = args
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
      jaxpr = core.apply_outfeed_rewriter(jaxpr)
      axis_env_ = make_axis_env(dispatch.jaxpr_replicas(jaxpr))
      if out_parts is None:
        out_parts_flat = None
      else:
        out_parts_flat = tuple(flatten_axes(
            "xla_computation out_parts", out_tree(), out_parts))
      c = xc.XlaBuilder(f"xla_computation_{fun_name}")
      xla_consts = map(partial(xla.pyval_to_ir_constant, c), consts)
      should_tuple = tuple_args if tuple_args is not None else (len(avals) > 100)
      xla_args, donated_invars = xla._xla_callable_args(
          c, avals, should_tuple, partitions=in_parts_flat, donated_invars=donated_invars)
      ctx = xla.TranslationContext(
          c, backend, axis_env_,
          extend_name_stack(wrap_name(fun_name, "xla_computation")))
      out_nodes = xla.jaxpr_subcomp(ctx, jaxpr, xla_consts, *xla_args)
    build_out_tuple = partial(xc.ops.Tuple, c, out_nodes)
    if out_parts is not None:
      out_tuple = xla.with_sharding(c, out_parts_flat, build_out_tuple)
    else:
      out_tuple = build_out_tuple()

    if any(donated_invars):
      donated_invars = xla.set_up_aliases(c, xla_args, c.GetShape(out_tuple),
                                          donated_invars, tuple_args)
    if any(donated_invars):
      shapes = [str(c.GetShape(a)) for a, d in zip(xla_args, donated_invars) if d]
      warn(f"Some donated buffers were not usable: {', '.join(shapes)}")
    built = c.build(out_tuple)
    out_shapes_flat = [
        ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
    out_shape = tree_unflatten(out_tree(), out_shapes_flat)
    for out_aval in out_avals:
      if not isinstance(out_aval, xla.ShapedArray):
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
    shapes and types as the corresponding arguments.
  """

  docstr = ("Value and gradient of {fun} with respect to positional "
            "argument(s) {argnums}. Takes the same arguments as {fun} but "
            "returns a two-element tuple where the first element is the value "
            "of {fun} and the second element is the gradient, which has the "
            "same shape as the arguments at positions {argnums}.")

  _check_callable(fun)
  argnums = core.concrete_or_error(_ensure_index, argnums)
  reduce_axes = _ensure_str_tuple(reduce_axes)

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def value_and_grad_f(*args, **kwargs):
    max_argnum = argnums if isinstance(argnums, int) else max(argnums)
    if max_argnum >= len(args):
      raise TypeError(f"differentiating with respect to argnums={argnums} requires at least "
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
    g = vjp_py(jax.lax._one(ans))
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
  _check_arg(x)
  aval = core.get_aval(x)
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
           holomorphic: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using forward-mode automatic differentiation.

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
  _check_callable(fun)
  argnums = _ensure_index(argnums)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    pushfwd = partial(_jvp, f_partial, dyn_args)
    y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    return tree_map(partial(_jacfwd_unravel, example_args), y, jac)

  return jacfun

def _check_input_dtype_jacfwd(holomorphic: bool, x: Any) -> None:
  _check_arg(x)
  aval = core.get_aval(x)
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
           holomorphic: bool = False, allow_int: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using reverse-mode automatic differentiation.

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
  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    y, pullback = _vjp(f_partial, *dyn_args)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
    return tree_transpose(tree_structure(example_args), tree_structure(y), jac_tree)

  return jacfun
jacobian = jacrev

_check_input_dtype_jacrev = partial(_check_input_dtype_revderiv, "jacrev")
_check_output_dtype_jacrev = partial(_check_output_dtype_revderiv, "jacrev")


def hessian(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
            holomorphic: bool = False) -> Callable:
  """Hessian of ``fun`` as a dense array.

  Args:
    fun: Function whose Hessian is to be computed.  Its arguments at positions
      specified by ``argnums`` should be arrays, scalars, or standard Python
      containers thereof. It should return arrays, scalars, or standard Python
      containers thereof.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
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
  {'c': {'a': {'a': DeviceArray([[[ 2.,  0.], [ 0.,  0.]],
                                 [[ 0.,  0.], [ 0., 12.]]], dtype=float32),
               'b': DeviceArray([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                                 [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32)},
         'b': {'a': DeviceArray([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                                 [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32),
               'b': DeviceArray([[[0.      , 0.      ], [0.      , 0.      ]],
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
  return jacfwd(jacrev(fun, argnums, holomorphic), argnums, holomorphic)

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
  return jax._src.lax.lax._convert_element_type(x, dtype, weak_type)

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


def vmap(fun: F, in_axes=0, out_axes=0, axis_name=None, axis_size=None) -> F:
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

      If the positional arguments to ``fun`` are container types, the
      corresponding element of ``in_axes`` can itself be a matching container,
      so that distinct array axes can be mapped for different container
      elements. ``in_axes`` must be a container tree prefix of the positional
      argument tuple passed to ``fun``.

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
  (DeviceArray([4., 5.], dtype=float32), 8.0)

  If the ``out_axes`` is specified for an unmapped result, the result is
  broadcast across the mapped axis:

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=0)(jnp.arange(2.), 4.))
  (DeviceArray([4., 5.], dtype=float32), DeviceArray([8., 8.], dtype=float32, weak_type=True))

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
  _check_callable(fun)
  docstr = ("Vectorized version of {fun}. Takes similar arguments as {fun} "
            "but with additional array axes over which {fun} is mapped.")
  if fun.__doc__:
    docstr += "\n\nOriginal documentation:\n\n"
    docstr += fun.__doc__

  axis_name = core.no_axis_name if axis_name is None else axis_name

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
  def batched_fun(*args, **kwargs):
    args_flat, in_tree  = tree_flatten((args, kwargs), is_leaf=batching.is_vmappable)
    f = lu.wrap_init(fun)
    flat_fun, out_tree = batching.flatten_fun_for_vmap(f, in_tree)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, (in_axes, 0), kws=True)
    axis_size_ = (axis_size if axis_size is not None else
                  _mapped_axis_size(in_tree, args_flat, in_axes_flat, "vmap",
                                    kws=True))
    out_flat = batching.batch(
        flat_fun, axis_name, axis_size_, in_axes_flat,
        lambda: flatten_axes("vmap out_axes", out_tree(), out_axes)
    ).call_wrapped(*args_flat)
    return tree_unflatten(out_tree(), out_flat)

  return batched_fun

def _mapped_axis_size(tree, vals, dims, name, *, kws=False):
  if not vals:
    args, kwargs = tree_unflatten(tree, vals)
    raise ValueError(
        f"{name} wrapped function must be passed at least one argument "
        f"containing an array, got empty *args={args} and **kwargs={kwargs}"
    )

  def _get_axis_size(name: str, shape: Tuple[int, ...], axis: int):
    try:
      return shape[axis]
    except (IndexError, TypeError) as e:
      min_rank = axis + 1 if axis >= 0 else -axis
      raise ValueError(f"{name} was requested to map its argument along axis {axis}, "
                       f"which implies that its rank should be at least {min_rank}, "
                       f"but is only {len(shape)} (its shape is {shape})") from e

  mapped_axis_sizes = {_get_axis_size(name, np.shape(x), d)
                       for x, d in zip(vals, dims)
                       if d is not None}
  try:
    size, = mapped_axis_sizes
    return size
  except ValueError as e:
    if not mapped_axis_sizes:
      raise ValueError(f"{name} must have at least one non-None value in in_axes") from e
    msg = f"{name} got inconsistent sizes for array axes to be mapped:\n" + "{}"
    # we switch the error message based on whether args is a tuple of arrays,
    # in which case we can produce an error message based on argument indices,
    # or if it has nested containers.
    if kws:
      # if keyword arguments are included in the tree, we make adapt the error
      # message only to be about the positional arguments
      tree, leaf = treedef_children(tree)
      assert treedef_is_leaf(leaf)
    # TODO(mattjj,phawkins): add a way to inspect pytree kind more directly
    if tree == tree_flatten((core.unit,) * tree.num_leaves)[1]:
      lines1 = [f"arg {i} has shape {np.shape(x)} and axis {d} is to be mapped"
                for i, (x, d) in enumerate(zip(vals, dims))]
      sizes = collections.defaultdict(list)
      for i, (x, d) in enumerate(zip(vals, dims)):
        if d is not None:
          sizes[x.shape[d]].append(i)
      lines2 = ["{} {} {} {} to be mapped of size {}".format(
                  "args" if len(idxs) > 1 else "arg",
                  ", ".join(map(str, idxs)),
                  "have" if len(idxs) > 1 else "has",
                  "axes" if len(idxs) > 1 else "an axis",
                  size)
                for size, idxs in sizes.items()]
      raise ValueError(msg.format("\n".join(lines1 + ["so"] + lines2))) from None
    else:
      sizes = [x.shape[d] if d is not None else None for x, d in zip(vals, dims)]
      sizes = tree_unflatten(tree, sizes)
      raise ValueError(msg.format(f"the tree of axis sizes is:\n{sizes}")) from None

def pmap(
  fun: F,
  axis_name: Optional[AxisName] = None,
  *,
  in_axes=0,
  out_axes=0,
  static_broadcasted_argnums: Union[int, Iterable[int]] = (),
  devices: Optional[Sequence[xc.Device]] = None,
  backend: Optional[str] = None,
  axis_size: Optional[int] = None,
  donate_argnums: Union[int, Iterable[int]] = (),
  global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
) -> F:
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
    donate_argnums: Specify which arguments are "donated" to the computation.
      It is safe to donate arguments if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to.
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


def _prepare_pmap(fun, in_axes, out_axes, static_broadcasted_tuple,
                  donate_tuple, global_arg_shapes, args, kwargs):
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

  if donate_tuple:
    donated_invars = donation_vector(donate_tuple, dyn_args, kwargs)
  else:
    donated_invars = (False,) * len(args)
  in_axes_flat = tuple(flatten_axes("pmap in_axes", in_tree, (dyn_in_axes, 0)))
  global_arg_shapes_flat = tuple(flatten_axes(
      "pmap global_arg_shapes", in_tree, (dyn_global_arg_shapes, None),
      kws=True))
  local_axis_size = _mapped_axis_size(
      in_tree, args, in_axes_flat, "pmap", kws=True)

  for arg in args:
    _check_arg(arg)

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

  return PmapCallInfo(flat_fun=flat_fun,
                      in_tree=in_tree,
                      out_tree=out_tree,
                      flat_args=args,
                      donated_invars=donated_invars,
                      in_axes_flat=in_axes_flat,
                      local_axis_size=local_axis_size,
                      global_arg_shapes_flat=global_arg_shapes_flat,
                      out_axes_thunk=out_axes_thunk)


def _get_f_mapped(
    *,
    fun: F,
    axis_name: Optional[AxisName],
    in_axes=0,
    out_axes=0,
    static_broadcasted_tuple: Tuple[int],
    devices: Optional[Sequence[xc.Device]],
    backend: Optional[str],
    axis_size: Optional[int],
    donate_tuple: Tuple[int],
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]],
):
  def f_pmapped(*args, **kwargs):
    p = _prepare_pmap(
        fun, in_axes, out_axes, static_broadcasted_tuple, donate_tuple,
        global_arg_shapes, args, kwargs)
    out = pxla.xla_pmap(
        p.flat_fun, *p.flat_args, backend=backend, axis_name=axis_name,
        axis_size=p.local_axis_size, global_axis_size=axis_size,
        devices=None if devices is None else tuple(devices),
        in_axes=p.in_axes_flat, out_axes_thunk=p.out_axes_thunk,
        name=p.flat_fun.__name__, donated_invars=p.donated_invars,
        global_arg_shapes=p.global_arg_shapes_flat)
    return p.out_tree, out

  return f_pmapped


def _shared_code_pmap(fun, axis_name, static_broadcasted_argnums,
                      donate_argnums, in_axes, out_axes):
  # axis_size is an optional integer representing the global axis size.  The
  # aggregate size (across all processes) size of the mapped axis must match the
  # given value.
  _check_callable(fun)
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
    fun: F,
    axis_name: Optional[AxisName] = None,
    *,
    in_axes=0,
    out_axes=0,
    static_broadcasted_argnums: Union[int, Iterable[int]] = (),
    devices: Optional[Sequence[xc.Device]] = None,
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
) -> F:
  """The Python only implementation."""
  axis_name, static_broadcasted_tuple, donate_tuple = _shared_code_pmap(
      fun, axis_name, static_broadcasted_argnums, donate_argnums, in_axes,
      out_axes)

  @wraps(fun)
  @api_boundary
  def f_pmapped(*args, **kwargs):
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

  f_pmapped.lower = _pmap_lower(
      fun, axis_name, in_axes, out_axes, static_broadcasted_tuple, devices,
      backend, axis_size, global_arg_shapes, donate_tuple)

  return f_pmapped


class _PmapFastpathData(NamedTuple):
  version: int  # For forward and backward compatibility
  xla_executable: xla.XlaExecutable
  in_handler: Any
  out_handler: Any
  out_pytree_def: Any
  # Data needed to handle the inputs.
  input_sharding_specs: Sequence[pxla.ShardingSpec]
  input_devices: Sequence[xc.Device]
  input_indices: Sequence[pxla.Index]
  # Data needed to build the ShardedDeviceArray from C++.
  out_sharding_specs: Sequence[pxla.ShardingSpec]
  out_indices: Sequence[pxla.Index]
  out_avals: Sequence[Any]


def _cpp_pmap(
    fun: F,
    axis_name: Optional[AxisName] = None,
    *,
    in_axes=0,
    out_axes=0,
    static_broadcasted_argnums: Union[int, Iterable[int]] = (),
    devices: Optional[Sequence[xc.Device]] = None,
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
) -> F:
  axis_name, static_broadcasted_tuple, donate_tuple = _shared_code_pmap(
      fun, axis_name, static_broadcasted_argnums, donate_argnums, in_axes,
      out_axes)
  del static_broadcasted_argnums, donate_argnums

  @api_boundary
  def cache_miss(*args, **kwargs):
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
    out_pytree_def = out_tree()
    out = tree_unflatten(out_pytree_def, out_flat)

    ### Decide whether we can support the C++ fast path
    execute: Optional[functools.partial] = None
    execute = pxla.parallel_callable.most_recent_entry()
    use_fastpath = (
        execute is not None and
        # We don't support JAX extension backends. In particular, some
        # extentions do not return a partial with a `func` attribute.
        getattr(execute[0], "func", None) is pxla.execute_replicated and
        # No tracers in the outputs. Checking for ShardedDeviceArray should be
        # sufficient, but we use the more general `DeviceArray`.
        all(isinstance(x, device_array.DeviceArray) for x in out_flat))
    ### If we can use the fastpath, we return required info to the caller.
    if use_fastpath:
      xla_executable, backend_, in_handler, out_handler = execute[0].args
      fastpath_data = _PmapFastpathData(
          version=1,
          xla_executable=xla_executable,
          in_handler=in_handler,
          out_handler=out_handler,
          out_pytree_def=out_pytree_def,
          input_sharding_specs=in_handler.sharding_specs,
          input_devices=in_handler.local_devices,
          input_indices=in_handler.input_indices,
          out_sharding_specs=out_handler.out_specs,
          out_indices=out_handler.out_indices,
          out_avals=out_handler.unmapped_local_out_avals,
      )

    else:
      fastpath_data = None

    return out, fastpath_data

  cpp_mapped_f = pmap_lib.pmap(fun, cache_miss,
                               static_broadcasted_tuple, pxla._shard_arg)

  f_pmapped = wraps(fun)(cpp_mapped_f)

  f_pmapped.lower = _pmap_lower(
      fun, axis_name, in_axes, out_axes, static_broadcasted_tuple, devices,
      backend, axis_size, global_arg_shapes, donate_tuple)

  return f_pmapped


def _pmap_lower(fun, axis_name, in_axes, out_axes, static_broadcasted_tuple,
                devices, backend, axis_size, global_arg_shapes, donate_tuple):
  """Make a ``lower`` method for pmapped functions."""
  # If the function we returned from ``pmap`` were a class instance,
  # this might naturally be a method, with ``fun`` as a ``self`` and
  # all the other arguments stored as attributes.
  @api_boundary
  def lower(*args, **kwargs) -> Lowered:
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
        global_arg_shapes, args, kwargs)
    abstract_args = map(xla.abstractify, p.flat_args)
    computation = pxla.lower_parallel_callable(
        p.flat_fun, backend, axis_name,
        axis_size=p.local_axis_size, global_axis_size=axis_size,
        devices=None if devices is None else tuple(devices),
        name=p.flat_fun.__name__,
        in_axes=p.in_axes_flat,
        out_axes_thunk=p.out_axes_thunk,
        donated_invars=p.donated_invars,
        global_arg_shapes=p.global_arg_shapes_flat,
        avals=abstract_args)
    return Lowered(computation, p.in_tree, p.out_tree(), donate_tuple)

  return lower


def mask(fun: Callable, in_shapes, out_shape=None) -> Callable:
  _check_callable(fun)
  unique_ids = masking.UniqueIds()

  in_specs, in_shapes_tree = tree_flatten(in_shapes)
  in_specs = map(masking.parse_spec, in_specs)
  in_specs = map(partial(masking.remap_ids, unique_ids), in_specs)

  if out_shape is not None:
    out_specs, out_spec_tree = tree_flatten(out_shape)
    out_specs = map(masking.parse_spec, out_specs)
    out_specs = map(partial(masking.remap_ids, unique_ids), out_specs)

  def wrapped_fun(args, logical_env):
    args_flat, in_tree = tree_flatten(args)
    if in_tree != in_shapes_tree:
      raise TypeError(f"Tree mismatch: Input {in_tree} and shape spec {in_shapes_tree}.")
    logical_env = {unique_ids[name] : val for name, val in logical_env.items()}
    in_shapes = map(masking.finalize_spec, in_specs, map(np.shape, args_flat))
    padded_env = masking.bind_shapes(in_shapes, [x.shape for x in args_flat])
    f = lu.wrap_init(fun)
    flat_fun, out_tree_thunk = flatten_fun_nokwargs(f, in_tree)
    outs, out_shapes = masking.mask_fun(
      flat_fun, logical_env, padded_env, args_flat, in_shapes)
    out_tree = out_tree_thunk()

    if out_shape is None:
      def logical_shape(poly_shape, padded_val):
        shape = masking.eval_poly_shape(poly_shape, logical_env)
        return ShapeDtypeStruct(shape, core.get_aval(padded_val).dtype)
      out_logicals = map(logical_shape, out_shapes, outs)
      return tree_unflatten(out_tree, outs), tree_unflatten(out_tree, out_logicals)
    else:
      masking.check_shapes(out_specs, out_spec_tree, list(out_shapes), out_tree)
      def padded_spec(shape_spec):
        return tuple(dim if dim is masking._monomorphic_dim else
                     masking.eval_poly(dim, padded_env) for dim in shape_spec)
      masking.check_shapes(map(padded_spec, out_specs), out_spec_tree,
                           map(np.shape, outs), out_tree, "Padded output")
      return tree_unflatten(out_tree, outs)
  return wrapped_fun

@curry
def shapecheck(in_shapes, out_shape, fun: Callable):
  _check_callable(fun)
  in_shapes, in_tree = tree_flatten(in_shapes)
  in_shapes = map(masking.parse_spec, in_shapes)
  out_specs, out_spec_tree = tree_flatten(out_shape)
  out_specs = map(masking.parse_spec, out_specs)
  flat_fun, out_tree_thunk = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  avals = map(partial(ShapedArray, dtype=np.float32), in_shapes)
  out_shapes = [o.shape for o in pe.abstract_eval_fun(flat_fun.call_wrapped, *avals)]
  masking.check_shapes(map(tuple, out_specs), out_spec_tree,
                       map(tuple, out_shapes), out_tree_thunk())
  return fun

def jvp(fun: Callable, primals, tangents) -> Tuple[Any, Any]:
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

  Returns:
    A ``(primals_out, tangents_out)`` pair, where ``primals_out`` is
    ``fun(*primals)``, and ``tangents_out`` is the Jacobian-vector product of
    ``function`` evaluated at ``primals`` with ``tangents``. The
    ``tangents_out`` value has the same Python tree structure and shapes as
    ``primals_out``.

  For example:

  >>> import jax
  >>>
  >>> y, v = jax.jvp(jax.numpy.sin, (0.1,), (0.2,))
  >>> print(y)
  0.09983342
  >>> print(v)
  0.19900084
  """
  _check_callable(fun)
  return _jvp(lu.wrap_init(fun), primals, tangents)

def _jvp(fun: lu.WrappedFun, primals, tangents):
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

  flat_fun, out_tree = flatten_fun_nokwargs(fun, tree_def)
  out_primals, out_tangents = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)
  return (tree_unflatten(out_tree(), out_primals),
          tree_unflatten(out_tree(), out_tangents))

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
  (DeviceArray(3.26819, dtype=float32, weak_type=True), DeviceArray(-5.00753, dtype=float32, weak_type=True))
  >>> y, f_jvp = jax.linearize(f, 2.)
  >>> print(y)
  3.2681944
  >>> print(f_jvp(3.))
  -5.007528
  >>> print(f_jvp(4.))
  -6.676704
  """
  _check_callable(fun)
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
    return tuple(map(lambda out_pv, tan_out: out_pv.merge_with_known(tan_out),
                     out_pvals, tangents_out))

  return apply_flat_fun(fun, io_tree, *py_args)

def _vjp_pullback_wrapper(cotangent_dtypes, cotangent_shapes,
                          io_tree, fun, py_args):
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


if sys.version_info >= (3, 8):
  from typing import Literal

  @overload  # type: ignore
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
else:

  @overload  # type: ignore
  def vjp(fun: Callable[..., T], *primals: Any) -> Tuple[T, Callable]:
    ...

  @overload
  def vjp(
      fun: Callable[..., Any], *primals: Any,
      has_aux: bool,
      reduce_axes: Sequence[AxisName] = ()
  ) -> Union[Tuple[Any, Callable], Tuple[Any, Callable, Any]]:
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
      should be evaluated. The length of ``primals`` should be equal to the
      number of positional parameters to ``fun``. Each primal value should be a
      tuple of arrays, scalar, or standard Python containers thereof.
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
    ``primals_out`` is ``fun(*primals)``.
    ``vjpfun`` is a function from a cotangent vector with the same shape as
    ``primals_out`` to a tuple of cotangent vectors with the same shape as
    ``primals``, representing the vector-Jacobian product of ``fun`` evaluated at
    ``primals``. If ``has_aux`` is ``True``, returns a
    ``(primals_out, vjpfun, aux)`` tuple where ``aux`` is the auxiliary data
    returned by ``fun``.

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
  _check_callable(fun)
  reduce_axes = _ensure_str_tuple(reduce_axes)
  return _vjp(
      lu.wrap_init(fun), *primals, has_aux=has_aux, reduce_axes=reduce_axes)

def _vjp(fun: lu.WrappedFun, *primals, has_aux=False, reduce_axes=()):
  """Variant of vjp() that takes an lu.WrappedFun."""
  primals_flat, in_tree = tree_flatten(primals)
  for arg in primals_flat: _check_arg(arg)
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
  vjp_py = Partial(partial(_vjp_pullback_wrapper,
                           ct_dtypes, ct_shapes,
                           (out_tree, in_tree)),
                   out_vjp)
  if not has_aux:
    return out_primal_py, vjp_py
  else:
    return out_primal_py, vjp_py, tree_unflatten(aux_tree, aux)


def linear_transpose(fun: Callable, *primals, reduce_axes=()) -> Callable:
  """Transpose a function that is promised to be linear.

  For linear functions, this transformation is equivalent to ``vjp``, but
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
  (DeviceArray(0.5, dtype=float32), DeviceArray(-0.5, dtype=float32))
  """
  reduce_axes = _ensure_str_tuple(reduce_axes)
  primals_flat, in_tree = tree_flatten(primals)
  flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  in_avals = map(shaped_abstractify, primals_flat)
  in_dtypes = map(dtypes.dtype, in_avals)

  in_pvals = map(pe.PartialVal.unknown, in_avals)
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(flat_fun, in_pvals,
                                               instantiate=True)
  out_avals, _ = unzip2(out_pvals)
  out_dtypes = map(dtypes.dtype, out_avals)
  if not (all(dtypes.issubdtype(d, np.inexact) for d in in_dtypes + out_dtypes)
          or all(dtypes.issubdtype(d, np.integer)
                 for d in in_dtypes + out_dtypes)):
    raise TypeError("linear_transpose only supports [float or complex] -> "
                    "[float or complex], and integer -> integer functions, "
                    f"but got {in_dtypes} -> {out_dtypes}.")

  def transposed_fun(consts, out_cotangent):
    out_cotangents, out_tree2 = tree_flatten(out_cotangent)
    if out_tree() != out_tree2:
      raise TypeError("cotangent tree does not match function output, "
                      f"expected {out_tree()} but got {out_tree2}")
    if not all(map(core.typecheck, out_avals, out_cotangents)):
      raise TypeError("cotangent type does not match function output, "
                      f"expected {out_avals} but got {out_cotangents}")
    dummies = [ad.UndefinedPrimal(a) for a in in_avals]
    in_cotangents = map(
        ad.instantiate_zeros,
        ad.backward_pass(jaxpr, reduce_axes, consts, dummies, out_cotangents))
    return tree_unflatten(in_tree, in_cotangents)

  # Ensure that transposed_fun is a PyTree
  return Partial(transposed_fun, consts)


def make_jaxpr(fun: Callable,
               static_argnums: Union[int, Iterable[int]] = (),
               axis_env: Optional[Sequence[Tuple[AxisName, int]]] = None,
               return_shape: bool = False,
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
  _check_callable(fun)
  static_argnums = _ensure_index_tuple(static_argnums)

  @wraps(fun)
  @api_boundary
  def jaxpr_maker(*args, **kwargs):
    wrapped = lu.wrap_init(fun)
    if static_argnums:
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      wrapped, args = argnums_partial(wrapped, dyn_argnums, args)
    jax_args, in_tree = tree_flatten((args, kwargs))
    jaxtree_fun, out_tree = flatten_fun(wrapped, in_tree)
    in_avals = map(shaped_abstractify, jax_args)
    with ExitStack() as stack:
      for axis_name, size in axis_env or []:
        stack.enter_context(core.extend_axis_env(axis_name, size, None))
      jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(jaxtree_fun, in_avals)
    closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    if return_shape:
      out_shapes_flat = [
          ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
      return closed_jaxpr, tree_unflatten(out_tree(), out_shapes_flat)
    return closed_jaxpr

  jaxpr_maker.__name__ = f"make_jaxpr({jaxpr_maker.__name__})"
  return jaxpr_maker


def device_put(x, device: Optional[xc.Device] = None):
  """Transfers ``x`` to ``device``.

  Args:
    x: An array, scalar, or (nested) standard Python container thereof.
    device: The (optional) :py:class:`Device` to which ``x`` should be
      transferred. If given, then the result is committed to the device.

  If the ``device`` parameter is ``None``, then this operation behaves like the
  identity function if the operand is on any device already, otherwise it
  transfers the data to the default device, uncommitted.

  For more details on data placement see the
  :ref:`FAQ on data placement <faq-data-placement>`.

  Returns:
    A copy of ``x`` that resides on ``device``.
  """
  return tree_map(lambda y: dispatch.device_put_p.bind(y, device=device), x)


def device_put_sharded(shards: Sequence[Any], devices: Sequence[xc.Device]):
  """Transfer array shards to specified devices and form ShardedDeviceArray(s).

  Args:
    shards: A sequence of arrays, scalars, or (nested) standard Python
      containers thereof representing the shards to be stacked together to form
      the output. The length of ``shards`` must equal the length of ``devices``.
    devices: A sequence of :py:class:`Device` instances representing the devices
      to which corresponding shards in ``shards`` will be transferred.

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
  if not len(shards) == len(devices):
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
    return pxla.make_sharded_device_array(stacked_aval, None, buffers)

  return tree_multimap(_device_put_sharded, *shards)


def device_put_replicated(x: Any, devices: Sequence[xc.Device]):
  """Transfer array(s) to each specified device and form ShardedDeviceArray(s).

  Args:
    x: an array, scalar, or (nested) standard Python container thereof
      representing the array to be replicated to form the output.
    devices: A sequence of :py:class:`Device` instances representing the devices
      to which ``x`` will be transferred.

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
    assert (isinstance(aval, core.ShapedArray) and
            len(xla.aval_to_xla_shapes(aval)) == 1)
    buf, = dispatch.device_put(x, devices[0])
    rest_bufs = [buf.copy_to_device(d) for d in devices[1:]]
    return pxla.make_sharded_device_array(aval, None, [buf, *rest_bufs])
  return tree_map(_device_put_replicated, x)


# TODO(mattjj): consider revising
def _device_get(x):
  if isinstance(x, core.Tracer):
    return x
  try:
    copy = x.copy
  except AttributeError:
    return x
  else:
    return copy()

def device_get(x: Any):
  """Transfer ``x`` to host.

  If ``x`` is a pytree, then the individual buffers are copied in parallel.

  Args:
    x: An array, scalar, DeviceArray or (nested) standard Python container thereof
      representing the array to be transferred to host.

  Returns:
    An array or (nested) Python container thereof representing the
    value of ``x``.

  Examples:
    Passing a DeviceArray:

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
  for y in tree_leaves(x):
    try:
      y.copy_to_host_async()
    except AttributeError:
      pass
  return tree_map(_device_get, x)

def _check_arg(arg):
  if not (isinstance(arg, core.Tracer) or _valid_jaxtype(arg)):
    raise TypeError(f"Argument '{arg}' of type {type(arg)} is not a valid JAX type.")

# TODO(mattjj,necula): this duplicates code in core.valid_jaxtype, but one
# internal user relies on it for duck-typing. must fix downstream user!
def _valid_jaxtype(arg):
  try:
    xla.abstractify(arg)  # faster than core.get_aval
  except TypeError:
    return False
  else:
    return True


class ShapeDtypeStruct:
  __slots__ = ["shape", "dtype", "named_shape"]
  def __init__(self, shape, dtype, named_shape=None):
    self.shape = shape
    self.dtype = np.dtype(dtype)
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
    return f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype.name}{ns})"

  __str__ = __repr__

  def __eq__(self, other):
    if not isinstance(other, ShapeDtypeStruct):
      return False
    else:
      return (other.shape, other.dtype, other.named_shape) == (
          self.shape, self.dtype, self.named_shape)

  def __hash__(self):
    # TODO(frostig): avoid the conversion from dict by addressing
    # https://github.com/google/jax/issues/8182
    named = frozenset(self.named_shape.items())
    return hash((self.shape, self.dtype, named))

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


def checkpoint(fun: Callable, concrete: bool = False, prevent_cse: bool = True,
               policy: Optional[Callable[..., bool]] = None,
               ) -> Callable:
  """Make ``fun`` recompute internal linearization points when differentiated.

  The :func:`jax.checkpoint` decorator, aliased to ``jax.remat``, provides a
  way to trade off computation time and memory cost in the context of automatic
  differentiation, especially with reverse-mode autodiff like :func:`jax.grad`
  and :func:`jax.vjp` but also with :func:`jax.linearize`.

  When differentiating a function in reverse-mode, by default all the
  linearization points (e.g. inputs to elementwise nonlinear primitive
  operations) are stored when evaluating the forward pass so that they can be
  reused on the backward pass. This evaluation strategy can lead to a high
  memory cost, or even to poor performance on hardware accelerators where memory
  access is much more expensive than FLOPs.

  An alternative evaluation strategy is for some of the linearization points to
  be recomputed (i.e. rematerialized) rather than stored. This approach can
  reduce memory usage at the cost of increased computation.

  This function decorator produces a new version of ``fun`` which follows
  the rematerialization strategy rather than the default store-everything
  strategy. That is, it returns a new version of ``fun`` which, when
  differentiated, doesn't store any of its intermediate linearization points.
  Instead, these linearization points are recomputed from the function's saved
  inputs.

  See the examples below.

  Args:
    fun: Function for which the autodiff evaluation strategy is to be changed
      from the default of storing all intermediate linearization points to
      recomputing them. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
    concrete: Optional, boolean indicating whether ``fun`` may involve
      value-dependent Python control flow (default False). Support for such
      control flow is optional, and disabled by default, because in some
      edge-case compositions with :func:`jax.jit` it can lead to some extra
      computation.
    prevent_cse: Optional, boolean indicating whether to prevent common
      subexpression elimination (CSE) optimizations in the HLO generated from
      differentiation. This CSE prevention has costs because it can foil other
      optimizations, and because it can incur high overheads on some backends,
      especially GPU. The default is True because otherwise, under a ``jit`` or
      ``pmap``, CSE can defeat the purpose of this decorator. But in some
      settings, like when used inside a ``scan``, this CSE prevention mechanism
      is unnecessary, in which case ``prevent_cse`` can be set to False.
    policy: This is an experimental feature and the API is likely to change.
      Optional callable, one of the attributes of ``jax.checkpoint_policies``,
      which takes as input a type-level specification of a first-order primitive
      application and returns a boolean indicating whether the corresponding
      output value(s) can be saved as a residual (or, if not, instead must be
      recomputed in the (co)tangent computation).

  Returns:
    A function (callable) with the same input/output behavior as ``fun`` but
    which, when differentiated using e.g. :func:`jax.grad`, :func:`jax.vjp`, or
    :func:`jax.linearize`, recomputes rather than stores intermediate
    linearization points, thus potentially saving memory at the cost of extra
    computation.

  Here is a simple example:

  >>> import jax
  >>> import jax.numpy as jnp

  >>> @jax.checkpoint
  ... def g(x):
  ...   y = jnp.sin(x)
  ...   z = jnp.sin(y)
  ...   return z
  ...
  >>> jax.value_and_grad(g)(2.0)
  (DeviceArray(0.78907233, dtype=float32, weak_type=True), DeviceArray(-0.2556391, dtype=float32, weak_type=True))

  Here, the same value is produced whether or not the :func:`jax.checkpoint`
  decorator is present. When the decorator is not present, the values
  ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))`` are computed on the forward
  pass and are stored for use in the backward pass, because they are needed
  on the backward pass and depend only on the primal inputs. When using
  :func:`jax.checkpoint`, the forward pass will compute only the primal outputs
  and only the primal inputs (``2.0``) will be stored for the backward pass.
  At that time, the value ``jnp.sin(2.0)`` is recomputed, along with the values
  ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))``.

  While ``jax.checkpoint`` controls what values are stored from the forward-pass
  to be used on the backward pass, the total amount of memory required to
  evaluate a function or its VJP depends on many additional internal details of
  that function. Those details include which numerical primitives are used,
  how they're composed, where jit and control flow primitives like scan
  are used, and other factors.

  The :func:`jax.checkpoint` decorator can be applied recursively to express
  sophisticated autodiff rematerialization strategies. For example:

  >>> def recursive_checkpoint(funs):
  ...   if len(funs) == 1:
  ...     return funs[0]
  ...   elif len(funs) == 2:
  ...     f1, f2 = funs
  ...     return lambda x: f1(f2(x))
  ...   else:
  ...     f1 = recursive_checkpoint(funs[:len(funs)//2])
  ...     f2 = recursive_checkpoint(funs[len(funs)//2:])
  ...     return lambda x: f1(jax.checkpoint(f2)(x))
  ...
  """
  @wraps(fun)
  @api_boundary
  def fun_remat(*args, **kwargs):
    args_flat, in_tree = tree_flatten((args, kwargs))
    flat_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
    out_flat = pe.remat_call(flat_fun, *args_flat, name=flat_fun.__name__,
                             concrete=concrete, prevent_cse=prevent_cse,
                             differentiated=False,
                             policy=policy)
    return tree_unflatten(out_tree(), out_flat)
  return fun_remat
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

  _, in_tree = tree_flatten(())

  @functools.wraps(fun)
  def named_f(*args, **kwargs):
    lu_f = lu.wrap_init(lambda: fun(*args, **kwargs))
    flat_f, out_tree = flatten_fun_nokwargs(lu_f, in_tree)
    out_flat = core.named_call_p.bind(flat_f, name=name)
    return tree_unflatten(out_tree(), out_flat)

  return named_f

def invertible(fun: Callable) -> Callable:
  """Asserts that the decorated function is invertible.

  Applying reverse-mode AD to a decorated function will use a more memory efficient
  procedure than usual, which will reconstruct the necessary intermediate values
  by inverting the function. Note that this might degrade the numerical accuracy of
  obtained gradients if the inverse is unstable.

  Args:
    fun: The function assumed to be invertible.
  """
  return iad.invertible(fun)


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
