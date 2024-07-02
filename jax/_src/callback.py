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
"""Module for JAX callbacks."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import logging
from typing import Any

import jax
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import sharding_impls
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax.control_flow.loops import map as lax_map
from jax._src.lib import xla_client as xc
from jax._src.sharding_impls import SingleDeviceSharding
import numpy as np

logger = logging.getLogger(__name__)


# `pure_callback_p` is the main primitive for staging out Python pure callbacks.
pure_callback_p = core.Primitive("pure_callback")
pure_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(pure_callback_p)

map, unsafe_map = util.safe_map, map


@dataclasses.dataclass(frozen=True)
class _FlatCallback:
  """A Python function callable with flat arguments and results.

  An instance of this class is used as a parameter for the callback primitives.
  We prefer it to an anonymous flattened function because it produces
  equal objects when we call the same Python function with the same argument
  structure.
  """
  callback_func: Callable[..., Any]
  in_tree: tree_util.PyTreeDef  # (args, kwargs) pytree for `callback_func`.

  def __call__(self, *flat_args: jax.Array) -> Sequence[jax.Array]:
    args, kwargs = tree_util.tree_unflatten(self.in_tree, flat_args)
    return tree_util.tree_leaves(self.callback_func(*args, **kwargs))


def pure_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    vectorized: bool,
):
  del sharding, vectorized, result_avals
  try:
    cpu_device, *_ = jax.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.pure_callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = jax.device_put(args, cpu_device)
  with jax.default_device(cpu_device):
    try:
      return tree_util.tree_map(np.asarray, callback(*args))
    except BaseException:
      logger.exception("jax.pure_callback failed")
      raise


pure_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                           pure_callback_p))


@pure_callback_p.def_abstract_eval
def pure_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: SingleDeviceSharding | None,
    vectorized: bool,
):
  del avals, callback, sharding, vectorized
  return result_avals


def pure_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Pure callbacks do not support JVP. "
      "Please use `jax.custom_jvp` to use callbacks while taking gradients.")


ad.primitive_jvps[pure_callback_p] = pure_callback_jvp_rule


def pure_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Pure callbacks do not support transpose. "
      "Please use `jax.custom_vjp` to use callbacks while taking gradients.")

ad.primitive_transposes[pure_callback_p] = pure_callback_transpose_rule


def callback_batching_rule(
    prim,
    args,
    dims,
    *,
    vectorized: bool,
    result_avals: Sequence[core.ShapedArray],
    **kwargs: Any,
):
  axis_size = next(a.shape[d] for a, d in zip(args, dims)
                   if d is not batching.not_mapped)
  new_args = [arg if dim is batching.not_mapped else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  if vectorized:
    result_avals = tuple(
        core.unmapped_aval(axis_size, core.no_axis_name, 0, aval)  # type: ignore
        for aval in result_avals)
    outvals = prim.bind(
        *new_args,
        vectorized=vectorized,
        result_avals=result_avals,
        **kwargs,
    )
  else:
    is_batched = [d is not batching.not_mapped for d in dims]
    unbatched_args, batched_args = util.partition_list(is_batched, new_args)
    def _batch_fun(batched_args):
      merged_args = util.merge_lists(is_batched, unbatched_args, batched_args)
      return prim.bind(
          *merged_args,
          result_avals=result_avals,
          vectorized=vectorized,
          **kwargs,
      )
    outvals = lax_map(_batch_fun, batched_args)
  return tuple(outvals), (0,) * len(outvals)


batching.primitive_batchers[pure_callback_p] = functools.partial(
    callback_batching_rule, pure_callback_p
)


def _callback_op_sharding(axis_context, sharding: SingleDeviceSharding | None):
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    # If we have fully manual sharding during lowering, that means the JAX
    # program has per-device semantics, so we run the callback on each device.
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "callbacks are only supported in spmd computations when all mesh"
          " axes are partitioned manually (no partial automatic sharding)."
      )
    if sharding is not None:
      raise NotImplementedError(
          "callbacks do not support specifying sharding inside spmd"
          " computations"
      )
    op_sharding = xc.OpSharding()
    op_sharding.type = xc.OpSharding.Type.MANUAL
    return op_sharding

  if isinstance(axis_context, sharding_impls.ShardingContext):
    if sharding is not None:
      if not isinstance(sharding, SingleDeviceSharding):
        raise NotImplementedError(
            "pure_callback only supports SingleDeviceSharding, but got"
            f" {type(sharding)}"
        )
      device = next(iter(sharding.device_set))
      device_assignment = axis_context.device_assignment
      if device_assignment is None:
        raise AssertionError(
            "Please file a bug at https://github.com/google/jax/issues")
      try:
        device_index = device_assignment.index(device)
      except IndexError as e:
        raise ValueError(
            "Sharding provided to pure_callback specifies a device"
            f" {device} that is not in the device assignment"
            f" ({device_assignment})") from e
    else:
      device_index = 0

    # If we have fully automatic sharding during lowering, that means the JAX
    # program has bulk array semantics, so we run the callback with a MAXIMAL
    # sharding and hence execute it only once on the full logical value).
    op_sharding = xc.OpSharding()
    op_sharding.type = xc.OpSharding.Type.MAXIMAL
    op_sharding.tile_assignment_dimensions = [1]
    op_sharding.tile_assignment_devices = [device_index]
    return op_sharding

  # When there's no SPMD partitioning going on, don't annotate a sharding.
  return None


def pure_callback_lowering(
    ctx, *args, callback: _FlatCallback, sharding: SingleDeviceSharding | None, **params
):
  def _callback(*flat_args):
    return tuple(
        pure_callback_impl(
            *flat_args,
            callback=callback,
            sharding=None,  # unused.
            **params,
        )
    )

  op_sharding = _callback_op_sharding(ctx.module_context.axis_context, sharding)
  result, _, _ = mlir.emit_python_callback(
      ctx,
      _callback,
      None,
      list(args),
      ctx.avals_in,
      ctx.avals_out,
      has_side_effect=False,
      sharding=op_sharding,
  )
  return result


mlir.register_lowering(pure_callback_p, pure_callback_lowering)

def _check_shape_dtype(shape_dtype):
  dt = np.dtype(shape_dtype.dtype)
  if dtypes.canonicalize_dtype(dt) != dt:
    raise ValueError(
        "result_shape_dtypes cannot specify 64-bit types when `jax_enable_x64` is disabled")


def pure_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: SingleDeviceSharding | None = None,
    vectorized: bool = False,
    **kwargs: Any,
):
  """Calls a pure Python callback. Works under :func:`jit`/:func:`~vmap`/etc.

  For more explanation, see `External Callbacks`_.

  ``pure_callback`` enables calling a Python function in JIT-ed JAX functions.
  The input ``callback`` will be passed JAX arrays placed on a local CPU, and
  it should also return JAX arrays on CPU.

  The callback is treated as functionally pure, meaning it has no side-effects
  and its output value depends only on its argument values. As a consequence, it
  is safe to be called multiple times (e.g. when transformed by :func:`~vmap` or
  :func:`~pmap`), or not to be called at all when e.g. the output of a
  `jit`-decorated function has no data dependence on its value. Pure callbacks
  may also be reordered if data-dependence allows.

  When `vmap`-ed the behavior will depend on the value of the
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
    callback: function to execute on the host. The callback is assumed to be a pure
      function (i.e. one without side-effects): if an impure function is passed, it
      may behave in unexpected ways, particularly under transformation. The callable
      will be passed PyTrees of arrays as arguments, and should return a PyTree of
      arrays that matches ``result_shape_dtypes``.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
      :class:`jax.ShapeDtypeStruct` is often used to define leaf values.
    *args: arguments to be passed to the callback function
    sharding: optional sharding that specifies the device from which the callback should
      be invoked.
    vectorized: boolean specifying whether the callback function can operate in a
      vectorized manner.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.experimental.io_callback`: callback designed for impure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  result_avals = tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), result_shape_dtypes)
  flat_result_avals, out_tree = tree_util.tree_flatten(result_avals)
  out_flat = pure_callback_p.bind(
      *flat_args,
      callback=_FlatCallback(callback, in_tree),
      result_avals=tuple(flat_result_avals),
      sharding=sharding,
      vectorized=vectorized,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)


# IO Callback

io_callback_p = core.Primitive("io_callback")
io_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(io_callback_p)

class IOEffect(effects.Effect):
  __str__ = lambda _: "IO"

class OrderedIOEffect(effects.Effect):
  __str__ = lambda _: "OrderedIO"

_IOEffect = IOEffect()
_OrderedIOEffect = OrderedIOEffect()
effects.lowerable_effects.add_type(IOEffect)
effects.lowerable_effects.add_type(OrderedIOEffect)
effects.control_flow_allowed_effects.add_type(IOEffect)
effects.control_flow_allowed_effects.add_type(OrderedIOEffect)
effects.ordered_effects.add_type(OrderedIOEffect)
effects.shardable_ordered_effects.add_type(OrderedIOEffect)


def io_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    ordered: bool,
):
  del result_avals, sharding, ordered
  try:
    cpu_device, *_ = jax.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.io_callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = jax.device_put(args, cpu_device)
  with jax.default_device(cpu_device):
    try:
      return tree_util.tree_map(np.asarray, callback(*args))
    except BaseException:
      logger.exception("jax.io_callback failed")
      raise


io_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                         io_callback_p))


@io_callback_p.def_effectful_abstract_eval
def io_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: SingleDeviceSharding | None,
    ordered: bool,
):
  del avals, sharding, callback
  effect = _OrderedIOEffect if ordered else _IOEffect
  return result_avals, {effect}


def io_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support JVP.")
ad.primitive_jvps[io_callback_p] = io_callback_jvp_rule


def io_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support transpose.")
ad.primitive_transposes[io_callback_p] = io_callback_transpose_rule


def io_callback_batching_rule(
    args, dims, callback, result_avals, sharding, ordered
):
  if ordered:
    raise ValueError("Cannot `vmap` ordered IO callback.")
  is_batched = [d is not batching.not_mapped for d in dims]
  new_args = [arg if dim is batching.not_mapped else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  unbatched_args, batched_args = util.partition_list(is_batched, new_args)
  def _batch_fun(batched_args):
    merged = util.merge_lists(is_batched, unbatched_args, batched_args)
    return io_callback_p.bind(*merged, callback=callback, sharding=sharding,
                              result_avals=result_avals, ordered=False)
  out_vals = lax_map(_batch_fun, batched_args)
  return out_vals, (0,) * len(out_vals)
batching.primitive_batchers[io_callback_p] = io_callback_batching_rule


def io_callback_lowering(ctx, *args, callback, sharding, ordered, **params):
  def _callback(*flat_args):
    return tuple(
        io_callback_impl(
            *flat_args,
            callback=callback,
            sharding=None,  # unused.
            ordered=ordered,
            **params,
        )
    )

  op_sharding = _callback_op_sharding(ctx.module_context.axis_context, sharding)
  if ordered:
    token = ctx.tokens_in.get(_OrderedIOEffect)
    result, token, _ = mlir.emit_python_callback(
        ctx,
        _callback,
        token,
        list(args),
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
        sharding=op_sharding,
    )
    ctx.set_tokens_out(mlir.TokenSet({_OrderedIOEffect: token}))
  else:
    result, token, _ = mlir.emit_python_callback(
        ctx,
        _callback,
        None,
        list(args),
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
        sharding=op_sharding,
    )
  return result


mlir.register_lowering(io_callback_p, io_callback_lowering)


def io_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: SingleDeviceSharding | None = None,
    ordered: bool = False,
    **kwargs: Any,
):
  """Calls an impure Python callback.

  For more explanation, see `External Callbacks`_.

  Args:
    callback: function to execute on the host. It is assumed to be an impure function.
      If ``callback`` is pure, using :func:`jax.pure_callback` instead may lead to
      more efficient execution.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
      :class:`jax.ShapeDtypeStruct` is often used to define leaf values.
    *args: arguments to be passed to the callback function
    sharding: optional sharding that specifies the device from which the callback should
      be invoked.
    ordered: boolean specifying whether sequential calls to callback must be ordered.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.pure_callback`: callback designed for pure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = map(lambda x: core.ShapedArray(x.shape, x.dtype),
                          flat_shape_dtypes)
  flat_args = map(core.raise_as_much_as_possible, flat_args)
  out_flat = io_callback_p.bind(
      *flat_args,
      callback=_FlatCallback(callback, in_tree),
      result_avals=tuple(flat_result_avals),
      sharding=sharding,
      ordered=ordered,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)
