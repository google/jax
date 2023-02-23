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
"""Module for JAX debugging primitives and related functionality."""

import functools
import string
import sys
from typing import Any, Dict, Callable, Optional, Sequence, Set, Tuple, Union
import weakref

import numpy as np

import jax.numpy as jnp
from jax import tree_util
from jax import lax
from jax._src import core
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.sharding import Sharding, GSPMDSharding, NamedSharding
from jax.interpreters import partial_eval as pe
from jax.config import config

# pytype: disable=import-error
try:
  import rich
  import rich.align
  import rich.box
  import rich.console
  import rich.padding
  import rich.style
  import rich.table
  RICH_ENABLED = True
except:
  RICH_ENABLED = False
# pytype: enable=import-error

class DebugEffect(effects.Effect):
  __str__ = lambda self: "Debug"
debug_effect = DebugEffect()

class OrderedDebugEffect(effects.Effect):
  __str__ = lambda self: "OrderedDebug"
ordered_debug_effect = OrderedDebugEffect()

effects.ordered_effects.add_type(OrderedDebugEffect)
effects.lowerable_effects.add_type(DebugEffect)
effects.lowerable_effects.add_type(OrderedDebugEffect)
effects.control_flow_allowed_effects.add_type(DebugEffect)
effects.control_flow_allowed_effects.add_type(OrderedDebugEffect)
effects.remat_allowed_effects.add_type(DebugEffect)
effects.remat_allowed_effects.add_type(OrderedDebugEffect)
effects.custom_derivatives_allowed_effects.add_type(DebugEffect)
effects.custom_derivatives_allowed_effects.add_type(OrderedDebugEffect)

# `debug_callback_p` is the main primitive for staging out Python callbacks.
debug_callback_p = core.Primitive('debug_callback')
debug_callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map

@debug_callback_p.def_impl
def debug_callback_impl(*args, callback: Callable[..., Any],
                        effect: DebugEffect):
  del effect
  return callback(*args)

@debug_callback_p.def_effectful_abstract_eval
def debug_callback_abstract_eval(*flat_avals, callback: Callable[..., Any],
                                 effect: DebugEffect):
  del flat_avals, callback
  return [], {effect}

def debug_callback_batching_rule(args, dims, **params):
  """Unrolls the debug callback across the mapped axis."""
  axis_size = next(x.shape[i] for x, i in zip(args, dims)
                   if i is not None)
  # TODO(sharadmv): implement in terms of rolled loop unstead of
  # unrolled.
  def get_arg_at_dim(i, dim, arg):
    if dim is batching.not_mapped:
      # Broadcast unmapped argument
      return arg
    return lax.index_in_dim(arg, i, axis=dim, keepdims=False)
  outs = []
  for i in range(axis_size):
    args_idx = map(functools.partial(get_arg_at_dim, i), dims, args)
    outs.append(debug_callback_p.bind(*args_idx, **params))
  outs = [jnp.stack(xs) for xs in zip(*outs)]
  return outs, (0,) * len(outs)
batching.primitive_batchers[debug_callback_p] = debug_callback_batching_rule

def debug_callback_jvp_rule(primals, tangents, **params):
  return debug_callback_p.bind(*primals, **params), []
ad.primitive_jvps[debug_callback_p] = debug_callback_jvp_rule

def debug_callback_transpose_rule(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect):
  del flat_args, callback, effect
  raise ValueError("Transpose doesn't support debugging callbacks.")
ad.primitive_transposes[debug_callback_p] = debug_callback_transpose_rule

def debug_callback_lowering(ctx, *args, effect, callback, **params):

  if isinstance(ctx.module_context.axis_context,
                (mlir.SPMDAxisContext, mlir.ShardingContext)):
    # Apply maximal sharding so pjit only executes the callback on device 0.
    sharding = xc.OpSharding()
    sharding.type = xc.OpSharding.Type.MAXIMAL
    sharding.tile_assignment_dimensions = [1]
    sharding.tile_assignment_devices = [0]
  else:
    sharding = None

  def _callback(*flat_args):
    return tuple(
        debug_callback_p.impl(
            *flat_args, effect=effect, callback=callback, **params))
  if effects.ordered_effects.contains(effect):
    token = ctx.tokens_in.get(effect)[0]
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, token, list(args), ctx.avals_in, ctx.avals_out, True)
    ctx.set_tokens_out(mlir.TokenSet({effect: (token,)}))
  else:
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, True,
        sharding=sharding)
  ctx.module_context.add_keepalive(keepalive)
  return result
mlir.register_lowering(debug_callback_p, debug_callback_lowering,
                       platform="cpu")
mlir.register_lowering(
    debug_callback_p, debug_callback_lowering, platform="gpu")
mlir.register_lowering(
    debug_callback_p, debug_callback_lowering, platform="tpu")

def _debug_callback_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  # The default behavior for effectful primitives is to not stage them if
  # possible. For debug callback, we actually want it to be staged to
  # provide more information to the user. This rule bypasses partial_eval's
  # regular behavior to do that. Specifically, we will stage the callback
  # if:
  # 1) the policy says debug_callbacks are not saveable
  # 2) the policy says debug_callbacks are saveable BUT all of the input
  #    values are instantiated.
  # The purpose is to call back with as much information as possible while
  # avoiding unnecessarily staging out other values.
  if any(unks_in):
    # The usual case (if we have any unknowns, we need to stage it out)
    res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
    return None, eqn, [], [], res
  if saveable(debug_callback_p, *[v.aval for v in eqn.invars], **eqn.params):
    # The policy is telling us we can save the debug callback.
    if all(inst_in):
      # If all of the inputs are instantiated, we also stage out the
      # debug_callback.
      return eqn, eqn, [], [], []
    else:
      # If any are not instantiated, we don't do any extra staging to avoid
      # affecting the computation.
      return eqn, None, [], [], []
  # If we can't save the debug callback (thanks to the policy) we listen to
  # the policy and stage out the debug callback.
  return eqn, eqn, [], [], []
pe.partial_eval_jaxpr_custom_rules[debug_callback_p] = (
    _debug_callback_partial_eval_custom)

def debug_callback(callback: Callable[..., Any], *args: Any,
                   ordered: bool = False, **kwargs: Any):
  """Calls a stageable Python callback.

  `debug_callback` enables you to pass in a Python function that can be called
  inside of a staged JAX program. A `debug_callback` follows existing JAX
  transformation *pure* operational semantics, which are therefore unaware of
  side-effects. This means the effect could be dropped, duplicated, or
  potentially reordered in the presence of higher-order primitives and
  transformations.

  We want this behavior because we'd like `debug_callback` to be "innocuous",
  i.e. we want these primitives to change the JAX computation as little as
  possible while revealing as much about them as possible, such as which parts
  of the computation are duplicated or dropped.

  Args:
    callback: A Python callable. Its return value will be ignored.
    *args: The positional arguments to the callback.
    ordered: A keyword only argument used to indicate whether or not the
      staged out computation will enforce ordering of this callback w.r.t.
      other ordered callbacks.
    **kwargs: The keyword arguments to the callback.
  Returns:
    The value of `callback(*args, **kwargs)`.
  """
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  effect = ordered_debug_effect if ordered else debug_effect
  def _flat_callback(*flat_args):
    args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
    callback(*args, **kwargs)
    return []
  return debug_callback_p.bind(*flat_args, callback=_flat_callback,
                               effect=effect)

class _DebugPrintFormatChecker(string.Formatter):

  def check_unused_args(self, used_args, args, kwargs):
    unused_args = [arg for i, arg in enumerate(args) if i not in used_args]
    unused_kwargs = [k for k in kwargs if k not in used_args]
    if unused_args:
      raise ValueError(
          f"Unused positional arguments to `jax.debug.print`: {unused_args}")
    if unused_kwargs:
      raise ValueError(
          f"Unused keyword arguments to `jax.debug.print`: {unused_kwargs}. "
          "You may be passing an f-string (i.e, `f\"{x}\"`) into "
          "`jax.debug.print` and instead should pass in a regular string.")

formatter = _DebugPrintFormatChecker()

def _format_print_callback(fmt: str, *args, **kwargs):
  sys.stdout.write(fmt.format(*args, **kwargs) + "\n")

def debug_print(fmt: str, *args, ordered: bool = False, **kwargs) -> None:
  """Prints values and works in staged out JAX functions.

  Note: This function does *not* work with f-strings because the formatting is
  done lazily.

  Args:
    fmt: A format string, e.g. ``"hello {x}"``, that will be used to format
      input arguments.
    *args: A list of positional arguments to be formatted.
    ordered: A keyword only argument used to indicate whether or not the
      staged out computation will enforce ordering of this ``debug_print``
      w.r.t. other ordered ``debug_print`` calls.
    **kwargs: Additional keyword arguments to be formatted.
  """
  # Check that we provide the correct arguments to be formatted
  formatter.format(fmt, *args, **kwargs)

  debug_callback(functools.partial(_format_print_callback, fmt), *args,
                 **kwargs, ordered=ordered)


# Sharding visualization

inspect_sharding_p = core.Primitive("inspect_sharding")
inspect_sharding_p.multiple_results = True

def _inspect_sharding_impl(value, *, callback):
  if not config.jax_array:
    raise NotImplementedError("`inspect_sharding` not implemented.")
  callback(value.sharding)
  return []
inspect_sharding_p.def_impl(_inspect_sharding_impl)

def _inspect_sharding_abstract_eval(aval, **_):
  del aval
  # Effectful abstract avoids DCE
  return [], {debug_effect}
inspect_sharding_p.def_effectful_abstract_eval(_inspect_sharding_abstract_eval)

def _inspect_sharding_batching_rule(args, _, *, callback):
  value, = args
  inspect_sharding_p.bind(value, callback=callback)
  return [], []
batching.primitive_batchers[inspect_sharding_p] = (
    _inspect_sharding_batching_rule)

def _inspect_sharding_jvp_rule(primals, _, **params):
  return inspect_sharding_p.bind(*primals, **params)
ad.primitive_jvps[inspect_sharding_p] = _inspect_sharding_jvp_rule

sharding_callbacks = weakref.WeakValueDictionary()  # type: ignore
_INSPECT_SHARDING_CALL_NAME = "InspectSharding"

class ShardingCallbackInfo:
  def __init__(self, callback, module_context):
    self.callback = callback
    self.module_context = module_context

def _inspect_sharding_lowering_rule(ctx: mlir.LoweringRuleContext, value, *,
                                    callback):

  mesh = pxla.thread_resources.env.physical_mesh
  axis_context = ctx.module_context.axis_context

  if isinstance(axis_context, mlir.ShardingContext):
    devices = axis_context.device_assignment
  elif isinstance(axis_context, mlir.SPMDAxisContext):
    devices = list(axis_context.mesh.devices.flat)
  else:
    raise NotImplementedError(type(axis_context))

  def _op_sharding_callback(op_sharding: xc.OpSharding):
    if mesh.empty:
      return callback(GSPMDSharding(
        devices, op_sharding))
    pspec = pjit.parse_flatten_op_sharding(
        op_sharding, mesh)[0].get_partition_spec()
    return callback(NamedSharding(mesh, pspec))

  if len(devices) == 1:
    # If we only have one device in our computation, we can construct a trivial
    # OpSharding and call it right now.
    trivial_sharding = xc.OpSharding()
    trivial_sharding.type = xc.OpSharding.Type.REPLICATED
    _op_sharding_callback(trivial_sharding)
    return []

  # If we have a nontrivial parallel computation, we need to wait until the SPMD
  # partitioner calls back with the `HloSharding.
  def _hlo_sharding_callback(hlo_sharding):
    op_sharding = hlo_sharding.to_proto()
    return _op_sharding_callback(op_sharding)

  # Here we store information in a container that we store globally so the
  # custom partitioning code can access it.
  sharding_callback_info = ShardingCallbackInfo(_hlo_sharding_callback,
                                                ctx.module_context)
  key = str(id(sharding_callback_info))
  sharding_callbacks[key] = sharding_callback_info
  # We need to make sure `sharding_callback_info` is still alive when the SPMD
  # partitioner runs so we keep it alive by attaching it to the executable.
  ctx.module_context.add_keepalive(sharding_callback_info)

  hlo.CustomCallOp([value.type], [value],
                   call_target_name=ir.StringAttr.get(
                     _INSPECT_SHARDING_CALL_NAME),
                   has_side_effect=ir.BoolAttr.get(True),
                   api_version=mlir.i32_attr(1),
                   called_computations=ir.ArrayAttr.get([]),
                   backend_config=ir.StringAttr.get(key),
                   operand_layouts=None,
                   result_layouts=None)
  return []
mlir.register_lowering(inspect_sharding_p, _inspect_sharding_lowering_rule)

def inspect_sharding_prop_user_sharding(sharding, backend_string):
  del sharding, backend_string
  return []

def inspect_sharding_partition(shapes, arg_shardings, result_shape,
                               result_sharding, backend_string):
  del result_shape, result_sharding
  sharding_callback_info = sharding_callbacks[backend_string]
  sharding_callback = sharding_callback_info.callback
  module_context = sharding_callback_info.module_context

  # Execute callback
  hlo_sharding, = arg_shardings
  sharding_callback(hlo_sharding)

  tiled_args = [p.tile(s) for s, p in zip(shapes, arg_shardings)]
  in_avals = [core.ShapedArray(arg.dimensions(), arg.numpy_dtype())
              for arg in tiled_args]
  fun = lu.wrap_init(lambda *args: [])
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(fun, in_avals)
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  trivial_comp = mlir.build_xla_computation_helper(closed_jaxpr,
      name="tmp_xla_computation", platforms=module_context.platforms,
      backend_or_name=module_context.backend_or_name,
      axis_context=module_context.axis_context)
  # The trivial computation built here has a dummy tuple as the result,
  # so use sharding compatible with it for the result sharding.
  empty_tuple_sharding = xc.OpSharding()
  empty_tuple_sharding.type = xc.OpSharding.Type.TUPLE
  result_sharding = xc.HloSharding.from_proto(empty_tuple_sharding)
  return trivial_comp, arg_shardings, result_sharding

def inspect_sharding_infer_sharding_from_operands(arg_shapes, arg_shardings,
                                                  shape, backend_string):
  del arg_shapes, shape, backend_string
  return arg_shardings[0]

xc.register_custom_call_partitioner(  # pytype: disable=module-attr
    _INSPECT_SHARDING_CALL_NAME, inspect_sharding_prop_user_sharding,
    inspect_sharding_partition, inspect_sharding_infer_sharding_from_operands,
    True)

def _slice_to_chunk_idx(size: int, slc: slice) -> int:
  if slc.stop == slc.start == None:
    return 0
  slice_size = slc.stop - slc.start
  assert slc.start % slice_size == 0
  assert size % slice_size == 0
  return slc.start // slice_size

def _raise_to_slice(slc: Union[slice, int]):
  if isinstance(slc, int):
    return slice(slc, slc + 1)
  return slc

Color = Union[Tuple[float, float, float], str]
ColorMap = Callable[[float], Tuple[float, float, float, float]]

def _canonicalize_color(color: Color) -> str:
  if isinstance(color, str):
    return color
  r, g, b = (int(a * 255) for a in color)
  return f"#{r:02X}{g:02X}{b:02X}"

def _get_text_color(color: str) -> str:
  r, g, b = map(lambda x: int(x, 16), (color[1:3], color[3:5], color[5:7]))
  if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
    return "#000000"
  return "#ffffff"

def make_color_iter(color_map, num_rows, num_cols):
  num_colors = num_rows * num_cols
  color_values = np.linspace(0, 1, num_colors)
  idx = 0
  for _ in range(num_colors):
    yield color_map(color_values[idx])
    idx = (idx + num_colors // 2 + bool(num_colors % 2 == 0)) % num_colors

def visualize_sharding(shape: Sequence[int], sharding: Sharding, *,
                       use_color: bool = True, scale: float = 1.,
                       min_width: int = 9, max_width: int = 80,
                       color_map: Optional[ColorMap] = None):
  """Visualizes a ``Sharding`` using ``rich``."""
  if not RICH_ENABLED:
    raise ValueError("`visualize_sharding` requires `rich` to be installed.")
  if len(shape) > 2 or len(shape) < 1:
    raise ValueError(
        "`visualize_sharding` only works for shapes with 1 and 2 dimensions.")
  console = rich.console.Console(width=max_width)
  use_color = use_color and console.color_system is not None
  if use_color and not color_map:
    try:
      import matplotlib as mpl  # pytype: disable=import-error
      color_map = mpl.colormaps["tab20b"]
    except ModuleNotFoundError:
      use_color = False

  base_height = int(10 * scale)
  aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
  base_width = int(base_height * aspect_ratio)
  height_to_width_ratio = 2.5

  # Grab the device kind from the first device
  device_kind = next(iter(sharding.device_set)).platform.upper()

  device_indices_map = sharding.devices_indices_map(tuple(shape))
  slices: Dict[Tuple[int, ...], Set[int]] = {}
  heights: Dict[Tuple[int, ...], Optional[float]] = {}
  widths: Dict[Tuple[int, ...], float] = {}

  for i, (dev, slcs) in enumerate(device_indices_map.items()):
    assert slcs is not None
    slcs = tuple(map(_raise_to_slice, slcs))
    chunk_idxs = tuple(map(_slice_to_chunk_idx, shape, slcs))
    if slcs is None:
      raise NotImplementedError
    if len(slcs) == 2:
      vert, horiz = slcs
      vert_size  = ((vert.stop  - vert.start ) if vert.stop  is not None
                    else shape[0])
      horiz_size = ((horiz.stop - horiz.start) if horiz.stop is not None
                    else shape[1])
      chunk_height = vert_size / shape[0]
      chunk_width = horiz_size / shape[1]
      heights[chunk_idxs] = chunk_height
      widths[chunk_idxs] = chunk_width
    else:
      # In the 1D case, we set the height to 1.
      horiz, = slcs
      vert = slice(0, 1, None)
      horiz_size = (
          (horiz.stop - horiz.start) if horiz.stop is not None else shape[0])
      chunk_idxs = (0, *chunk_idxs)
      heights[chunk_idxs] = None
      widths[chunk_idxs]  = horiz_size / shape[0]
    slices.setdefault(chunk_idxs, set()).add(dev.id)
  num_rows = max([a[0] for a in slices.keys()]) + 1
  if len(list(slices.keys())[0]) == 1:
    num_cols = 1
  else:
    num_cols = max([a[1] for a in slices.keys()]) + 1

  color_iter = make_color_iter(color_map, num_rows, num_cols)
  table = rich.table.Table(show_header=False, show_lines=not use_color,
                           padding=0,
                           highlight=not use_color, pad_edge=False,
                           box=rich.box.SQUARE if not use_color else None)
  for i in range(num_rows):
    col = []
    for j in range(num_cols):
      entry = f"{device_kind} "+",".join([str(s) for s in sorted(slices[i, j])])
      width, maybe_height = widths[i, j], heights[i, j]
      width = int(width * base_width * height_to_width_ratio)
      if maybe_height is None:
        height = 1
      else:
        height = int(maybe_height * base_height)
      width = min(max(width, min_width), max_width)
      left_padding, remainder = divmod(width - len(entry) - 2, 2)
      right_padding = left_padding + remainder
      top_padding, remainder = divmod(height - 2, 2)
      bottom_padding = top_padding + remainder
      if use_color:
        color = _canonicalize_color(next(color_iter)[:3])
        text_color = _get_text_color(color)
        top_padding += 1
        bottom_padding += 1
        left_padding += 1
        right_padding += 1
      else:
        color = None
        text_color = None
      padding = (top_padding, right_padding, bottom_padding, left_padding)
      padding = tuple(max(x, 0) for x in padding)  # type: ignore
      col.append(
          rich.padding.Padding(
            rich.align.Align(entry, "center", vertical="middle"), padding,
            style=rich.style.Style(bgcolor=color,
              color=text_color)))
    table.add_row(*col)
  console.print(table, end='\n\n')

def inspect_array_sharding(value, *, callback: Callable[[Sharding], None]):
  """Enables inspecting array sharding inside JIT-ted functions.

  This function, when provided with a Pytree of arrays, calls back with each of
  their shardings and works in ``pjit``-ted computations, enabling inspecting
  the chosen intermediate shardings.

  The policy for when ``callback`` is called is *as early as possible* when the
  sharding information is available. This means if ``inspect_array_callback`` is
  called without any transformations, the callback will happen immediately
  since we have the array and its sharding readily available. Inside of a
  ``jax.jit``, the callback will happen at lowering time, meaning you can
  trigger the callback using the AOT API (``jit(f).lower(...)``). When inside of
  a ``pjit``, the callback happens *at compile time* since the sharding is
  determined by XLA. You can trigger the callback by using JAX's AOT API
  (``pjit(f).lower(...).compile()``). In all cases, the callback will be
  triggered by running the function, since running a function entails lowering
  and compiling it first. However, once the function is compiled and cached,
  the callback will no longer occur.

  This function is experimental and its behavior may change in the future.

  Args:
    value: A Pytree of JAX arrays.
    callback: A callable that takes in a ``Sharding`` and doesn't return a value.

  In the following example, we print out the sharding of an intermediate value
  in a ``pjit``-ted computation:

  >>> import jax
  >>> import jax.numpy as jnp
  >>> from jax.experimental.pjit import pjit
  >>> from jax.sharding import Mesh, PartitionSpec
  >>>
  >>> x = jnp.arange(8, dtype=jnp.float32)
  >>> def f_(x):
  ...   x = jnp.sin(x)
  ...   jax.debug.inspect_array_sharding(x, callback=print)
  ...   return jnp.square(x)
  >>> f = pjit(f_, in_axis_resources=PartitionSpec('dev'),
  ...          out_axis_resources=PartitionSpec('dev'))
  >>> with Mesh(jax.devices(), ('dev',)):
  ...   f.lower(x).compile()  # doctest: +SKIP
  ...
  NamedSharding(mesh={'dev': 8}, partition_spec=PartitionSpec(('dev',),))
  """
  def _inspect(val):
    inspect_sharding_p.bind(val, callback=callback)
  tree_util.tree_map(_inspect, value)

def visualize_array_sharding(arr, **kwargs):
  """Visualizes an array's sharding."""
  if not config.jax_array:
    raise NotImplementedError("`visualize_array_sharding` not implemented.")
  def _visualize(sharding):
    return visualize_sharding(arr.shape, sharding, **kwargs)
  inspect_array_sharding(arr, callback=_visualize)
