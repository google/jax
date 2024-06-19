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

"""Module for Pallas:TPU-specific JAX primitives and functions."""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import pretty_printer as pp
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.state import indexing
from jax._src.state import primitives as sp
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pl_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.state import discharge as state_discharge
from jax._src.typing import DTypeLike
import jax.numpy as jnp

Slice = indexing.Slice

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

repeat_p = jax_core.Primitive('repeat')

def repeat(x, repeats, axis):
  return repeat_p.bind(x, repeats=repeats, axis=axis)

@repeat_p.def_abstract_eval
def _repeat_abstract_eval(x, *, repeats, axis):
  shape = list(x.shape)
  shape[axis] *= repeats
  return jax_core.ShapedArray(shape, x.dtype)


def _repeat_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, repeats, axis):
  def _repeat(x):
    return jnp.repeat(x, repeats, axis)
  return mlir.lower_fun(_repeat, multiple_results=False)(ctx, x)
mlir.register_lowering(repeat_p, _repeat_lowering_rule)

bitcast_p = jax_core.Primitive("bitcast")


def bitcast(x, ty: DTypeLike):
  ty = dtypes.canonicalize_dtype(ty)
  if len(x.shape) < 2:
    raise ValueError("Not implemented: bitcast 1D")
  if x.shape[-2] * x.dtype.itemsize % ty.itemsize:
    raise ValueError(
        "Not implemented: the 2nd minor dim can not be perfectly packed or"
        " unpacked"
    )
  return bitcast_p.bind(x, ty=ty)


@bitcast_p.def_abstract_eval
def _bitcast_abstract_eval(x, *, ty):
  shape = list(x.shape)
  shape[-2] = shape[-2] * x.dtype.itemsize // ty.itemsize
  return jax_core.ShapedArray(shape, ty)


def _bitcast_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, ty):
  def _bitcast(x):
    if x.dtype.itemsize < ty.itemsize:
      *leading, m, n = x.shape
      packing = ty.itemsize // x.dtype.itemsize
      x = x.reshape(*leading, m // packing, packing, n)
      x = jnp.swapaxes(x, -1, -2)
      return jax.lax.bitcast_convert_type(x, ty)
    if x.dtype.itemsize > ty.itemsize:
      y = jax.lax.bitcast_convert_type(x, ty)
      *leading, m, n, packing = y.shape
      return jnp.swapaxes(y, -1, -2).reshape(*leading, m * packing, n)
    return jax.lax.bitcast_convert_type(x, ty)

  return mlir.lower_fun(_bitcast, multiple_results=False)(ctx, x)


mlir.register_lowering(bitcast_p, _bitcast_lowering_rule)

roll_p = jax_core.Primitive("roll")


def roll(
    x,
    shift,
    axis: int,
    *,
    stride: int | None = None,
    stride_axis: int | None = None,
):
  if isinstance(shift, int) and shift < 0:
    raise ValueError("shift must be non-negative.")
  if axis < 0 or axis >= len(x.shape):
    raise ValueError("axis is out of range.")
  if (stride is None) != (stride_axis is None):
    raise ValueError("stride and stride_axis must be both specified or not.")
  if stride is not None and stride_axis is not None:
    if stride < 0:
      raise ValueError("stride must be non-negative.")
    if stride_axis < 0 or stride_axis >= len(x.shape):
      raise ValueError("stride_axis is out of range")
    if axis == stride_axis:
      raise ValueError("expected axis and stride_axis are different.")
  return roll_p.bind(
      x, shift, axis=axis, stride=stride, stride_axis=stride_axis
  )


@roll_p.def_abstract_eval
def _roll_abstract_eval(x, shift, **_):
  del shift
  return jax_core.raise_to_shaped(x)


def _roll_lowering_rule(
    ctx: mlir.LoweringRuleContext, x, shift, *, axis, stride, stride_axis
):
  def _roll(x, shift):
    if stride is None:
      return jnp.roll(x, shift, axis)
    outputs = [
        jnp.roll(xs, shift + i * stride, axis)
        for i, xs in enumerate(jnp.split(x, x.shape[stride_axis], stride_axis))
    ]
    return jnp.concatenate(outputs, stride_axis)

  return mlir.lower_fun(_roll, multiple_results=False)(ctx, x, shift)


mlir.register_lowering(roll_p, _roll_lowering_rule)


run_scoped_p = jax_core.Primitive('run_scoped')
run_scoped_p.multiple_results = True


def run_scoped(f: Callable[..., None], *types, **kw_types) -> None:
  flat_types, in_tree = tree_util.tree_flatten((types, kw_types))
  flat_fun, _ = api_util.flatten_fun(lu.wrap_init(f), in_tree)
  avals = map(lambda t: t.get_aval(), flat_types)
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, avals)
  run_scoped_p.bind(*consts, jaxpr=jaxpr)


@run_scoped_p.def_effectful_abstract_eval
def _run_scoped_abstract_eval(*args, jaxpr):
  # jaxpr will have effects for its inputs (Refs that are allocated) and for
  # constvars (closed over Refs). The effects for the allocated Refs are local
  # to the jaxpr and shouldn't propagate out.
  nonlocal_effects = {
      eff for eff in jaxpr.effects
      if not (
          isinstance(eff, effects.JaxprInputEffect)
          and eff.input_index >= len(jaxpr.constvars)
      )
  }
  return [], nonlocal_effects


class DeviceIdType(enum.Enum):
  MESH = "mesh"
  LOGICAL = "logical"


def check_sem_avals(sem_aval, sem_indexers_avals, name):
  if not isinstance(sem_aval, state.AbstractRef):
    raise ValueError(f"Cannot {name} on a non-semaphore Ref: {sem_aval}")
  sem_shape = sem_aval.shape
  if sem_indexers_avals:
    sem_shape = sem_indexers_avals[-1].get_indexer_shape()
  if sem_shape:
    raise ValueError(f"Cannot {name} on a non-()-shaped semaphore: {sem_shape}")
  sem_dtype = sem_aval.dtype
  if not (
      jnp.issubdtype(sem_dtype, tpu_core.semaphore)
      or jnp.issubdtype(sem_dtype, tpu_core.barrier_semaphore)
  ):
    raise ValueError(f"Must {name} a REGULAR or BARRIER semaphore: {sem_dtype}")


semaphore_read_p = jax_core.Primitive("semaphore_read")
semaphore_read_p.multiple_results = False


def semaphore_read(sem_or_view):
  ref, indexers = _get_ref_and_indexers(sem_or_view)
  args = [ref, indexers]
  flat_args, args_tree = tree_util.tree_flatten(args)
  return semaphore_read_p.bind(*flat_args, args_tree=args_tree)

@semaphore_read_p.def_abstract_eval
def _semaphore_read_abstract_eval(
    *avals,
    args_tree,
):
  sem_aval, sem_indexers_avals = tree_util.tree_unflatten(args_tree, avals)
  check_sem_avals(sem_aval, sem_indexers_avals, "read")
  return jax_core.ShapedArray((), jnp.dtype("int32"))


semaphore_signal_p = jax_core.Primitive('semaphore_signal')
semaphore_signal_p.multiple_results = True


def semaphore_signal(
    sem_or_view,
    inc: int | jax.Array = 1,
    *,
    device_id: int | jax.Array | None | tuple[int | jax.Array, ...] = None,
    device_id_type: DeviceIdType = DeviceIdType.MESH,
    core_index: int | jax.Array | None = None,
):
  ref, indexers = _get_ref_and_indexers(sem_or_view)
  inc = jnp.asarray(inc, dtype=jnp.int32)
  args = [ref, indexers, inc, device_id, core_index]
  flat_args, args_tree = tree_util.tree_flatten(args)
  semaphore_signal_p.bind(
      *flat_args,
      args_tree=args_tree,
      device_id_type=device_id_type,
  )


@semaphore_signal_p.def_abstract_eval
def _semaphore_signal_abstract_eval(
    *avals,
    args_tree,
    device_id_type: DeviceIdType,
):
  del device_id_type
  sem_aval, sem_indexers_avals, value_aval, device_id_avals, core_index_aval = (
      tree_util.tree_unflatten(args_tree, avals)
  )
  check_sem_avals(sem_aval, sem_indexers_avals, "signal")
  if value_aval.dtype != jnp.dtype("int32"):
    raise ValueError("Must signal an int32 value.")
  if device_id_avals is not None:
    device_id_flat_avals = tree_util.tree_leaves(device_id_avals)
    for aval in device_id_flat_avals:
      if aval.dtype != jnp.dtype("int32"):
        raise ValueError("`device_id`s must be an int32 value.")
  return []


def _semaphore_signal_pp_eqn(eqn: jax_core.JaxprEqn,
                             context: jax_core.JaxprPpContext,
                             settings: jax_core.JaxprPpSettings):
  del settings
  invars = eqn.invars
  tree = eqn.params["args_tree"]
  (
      sem,
      sem_indexers,
      value,
      device_ids,
      _,
  ) = tree_util.tree_unflatten(tree, invars)
  out = pp.concat([
      pp.text('semaphore_signal'),
      pp.text(' '),
      sp.pp_ref_indexers(context, sem, sem_indexers),
      pp.text(' '),
      pp.text(jax_core.pp_var(value, context)),
  ])
  if device_ids is not None:
    flat_device_ids = tree_util.tree_leaves(device_ids)
    if not flat_device_ids:
      return out
    device_ids_pp = [pp.text(jax_core.pp_var(flat_device_ids[0], context))]
    for device_id in flat_device_ids[1:]:
      device_ids_pp.append(pp.text(" "))
      device_ids_pp.append(pp.text(jax_core.pp_var(device_id, context)))
    out = pp.concat([out, pp.concat(device_ids_pp)])
  return out
jax_core.pp_eqn_rules[semaphore_signal_p] = _semaphore_signal_pp_eqn

semaphore_wait_p = jax_core.Primitive('semaphore_wait')
semaphore_wait_p.multiple_results = True

def semaphore_wait(sem_or_view, dec: int | jax.Array = 1):
  ref, indexers = _get_ref_and_indexers(sem_or_view)
  dec = jnp.asarray(dec, dtype=jnp.int32)
  args = [ref, indexers, dec]
  flat_args, args_tree = tree_util.tree_flatten(args)
  semaphore_wait_p.bind(*flat_args, args_tree=args_tree)

@semaphore_wait_p.def_abstract_eval
def _semaphore_wait_abstract_eval(*avals, args_tree):
  sem_aval, sem_indexers_avals, value_aval = tree_util.tree_unflatten(args_tree, avals)
  check_sem_avals(sem_aval, sem_indexers_avals, "wait")
  if value_aval.dtype != jnp.dtype("int32"):
    raise ValueError("Must wait an int32 value.")
  return []

def _semaphore_wait_pp_eqn(eqn: jax_core.JaxprEqn,
                             context: jax_core.JaxprPpContext,
                             settings: jax_core.JaxprPpSettings):
  del settings
  invars = eqn.invars
  tree = eqn.params["args_tree"]
  (
      sem,
      sem_indexers,
      value,
  ) = tree_util.tree_unflatten(tree, invars)
  return pp.concat([
      pp.text('semaphore_wait'),
      pp.text(' '),
      sp.pp_ref_indexers(context, sem, sem_indexers),
      pp.text(' '),
      pp.text(jax_core.pp_var(value, context)),
  ])
jax_core.pp_eqn_rules[semaphore_wait_p] = _semaphore_wait_pp_eqn


@dataclasses.dataclass
class AsyncCopyDescriptor:
  src_ref: Any
  src_indexers: tuple[indexing.NDIndexer, ...]
  dst_ref: Any
  dst_indexers: tuple[indexing.NDIndexer, ...]
  dst_sem: int | jax.Array
  dst_sem_indexers: tuple[indexing.NDIndexer, ...]
  src_sem: int | jax.Array | None
  src_sem_indexers: tuple[indexing.NDIndexer, ...] | None
  device_id: int | jax.Array | None
  device_id_type: DeviceIdType = DeviceIdType.MESH

  def __post_init__(self):
    if (self.src_sem is None) ^ (self.device_id is None):
      raise ValueError("Either both or neither `src_sem` and `device_id` "
                       "can be set.")

  @property
  def is_remote(self):
    return self.src_sem is not None

  def start(self):
    flat_args, tree = tree_util.tree_flatten((
        self.src_ref,
        self.src_indexers,
        self.dst_ref,
        self.dst_indexers,
        self.dst_sem,
        self.dst_sem_indexers,
        self.src_sem,
        self.src_sem_indexers,
        self.device_id,
    ))
    dma_start_p.bind(*flat_args, tree=tree, device_id_type=self.device_id_type)

  def wait(self):
    if self.is_remote:
      self.wait_send()
    self.wait_recv()

  def wait_recv(self):
    wait_args, tree = tree_util.tree_flatten(
        (self.dst_sem, self.dst_sem_indexers, self.dst_ref, self.dst_indexers)
    )
    dma_wait_p.bind(
        *wait_args, tree=tree, device_id_type=self.device_id_type
    )

  def wait_send(self):
    if not self.is_remote:
      raise ValueError("Cannot `wait_send` on a local copy.")
    wait_args, tree = tree_util.tree_flatten(
        (self.src_sem, self.src_sem_indexers, self.src_ref, self.src_indexers)
    )
    dma_wait_p.bind(
        *wait_args, tree=tree, device_id_type=self.device_id_type
    )


dma_start_p = jax_core.Primitive('dma_start')
dma_start_p.multiple_results = True

@dma_start_p.def_abstract_eval
def _dma_start_abstract_eval(*args, tree, device_id_type):
  (
      src_ref_aval,
      src_indexers_avals,
      dst_ref_aval,
      dst_indexers_avals,
      dst_sem_aval,
      dst_sem_indexers_avals,
      src_sem_aval,
      src_sem_indexers_avals,
      device_id_aval,
  ) = tree_util.tree_unflatten(tree, args)
  dst_sem_shape = dst_sem_aval.shape
  if dst_sem_indexers_avals:
    dst_sem_shape = dst_sem_indexers_avals[-1].get_indexer_shape()
  if dst_sem_shape:
    raise ValueError(
        f"Cannot signal on a non-()-shaped semaphore: {dst_sem_shape}"
    )
  if src_sem_aval is not None:
    src_sem_shape = src_sem_aval.shape
    if src_sem_indexers_avals:
      src_sem_shape = src_sem_indexers_avals[-1].get_indexer_shape()
    if src_sem_shape:
      raise ValueError(
          f"Cannot signal on a non-()-shaped semaphore: {src_sem_shape}"
      )
  return []

def _dma_start_pp_eqn(eqn: jax_core.JaxprEqn,
                      context: jax_core.JaxprPpContext,
                      settings: jax_core.JaxprPpSettings):
  invars = eqn.invars
  tree = eqn.params["tree"]
  (
      src_ref,
      src_indexers,
      dst_ref,
      dst_indexers,
      dst_sem,
      dst_sem_indexers,
      src_sem,
      src_sem_indexers,
      device_id,
  ) = tree_util.tree_unflatten(tree, invars)
  del src_sem_indexers
  # TODO(sharadmv): pretty print source semaphores and device id
  if src_sem or device_id:
    return jax_core._pp_eqn(eqn, context, settings)
  return pp.concat([
      pp.text('dma_start'),
      pp.text(' '),
      sp.pp_ref_indexers(context, src_ref, src_indexers),
      pp.text(' -> '),
      sp.pp_ref_indexers(context, dst_ref, dst_indexers),
      pp.text(' '),
      sp.pp_ref_indexers(context, dst_sem, dst_sem_indexers),
  ])

jax_core.pp_eqn_rules[dma_start_p] = _dma_start_pp_eqn

def dma_start_discharge_rule(in_avals, out_avals,
                             *args, tree, device_id_type):
  (
      src_ref,
      src_indexers,
      dst_ref,
      dst_indexers,
      dst_sem,
      dst_sem_indexers,
      src_sem,
      src_sem_indexers,
      device_id,
  ) = tree_util.tree_unflatten(tree, args)
  del out_avals, dst_sem, dst_sem_indexers
  is_remote = src_sem is not None and device_id is not None
  if is_remote:
    if device_id_type == DeviceIdType.MESH:
      raise NotImplementedError("Mesh device_id_type not supported.")
  else:
    assert src_sem is None
    assert src_sem_indexers is None
    assert device_id is None

  def _find_slice_start_size(indexer):
    num_scalar_idxs = 0
    # TODO(b/329733289): support strided load/store in interpret mode.
    for s in indexer.indices:
      if isinstance(s, Slice) and s.stride > 1:
        raise NotImplementedError("Strides not supported in discharge"
                                  " rule of dma_start.")
      if not isinstance(s, Slice):
        num_scalar_idxs += 1
    indices = indexer.indices
    scalar_dims = [not isinstance(s, Slice) and not s.shape for s in indices]
    slice_starts = [s.start if isinstance(s, Slice) else s for s in indices]
    slice_sizes = tuple(s.size if isinstance(s, Slice) else 1 for s in indices)
    return scalar_dims, slice_starts, slice_sizes, num_scalar_idxs

  num_src_index_vals = 0
  if src_indexers:
    if len(src_indexers) != 1:
      raise NotImplementedError("Multiple indexers not supported.")
    idx = src_indexers[0]
    if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
      (_, slice_starts,
       slice_sizes, num_scalar_idxs) = _find_slice_start_size(idx)
      num_src_index_vals += num_scalar_idxs
      updates = jax.lax.dynamic_slice(
          src_ref, slice_starts, slice_sizes=slice_sizes)
    else:
      updates = src_ref[idx.indices]
  else:
    updates = src_ref

  if is_remote:
    # Note that this code only works in SPMD mode. If not all devices execute
    # the DMA then the devices that do will hang.
    # TODO(justinfu): Verify that code only works in SPMD mode.
    axis_env = jax_core.thread_local_state.trace_state.axis_env
    axis_names = tuple(frame.name for frame in axis_env)
    nonempty_axis_names = tuple(name for name in axis_names if name is not None)
    if len(nonempty_axis_names) > 1:
      raise NotImplementedError("Sharding with more than one named axis not "
                                "implemented in dma_start_p.")
    shard_axis = nonempty_axis_names[0]
    my_axis = jax.lax.axis_index(shard_axis)
    # Update dst_ref from the perspective of the current device as the
    # receiver.
    who_copy_to_me = jax.lax.all_gather(device_id, shard_axis) == my_axis
    # TODO(justinfu): Add a checkify for verifying there is at most one source.
    # TODO(justinfu): Handle the case where no other device is copying to
    # this device.
    index = jnp.argmax(who_copy_to_me, axis=0)
    global_updates = jax.lax.all_gather(updates, shard_axis)
    updates = jax.lax.dynamic_index_in_dim(
        global_updates, index, axis=0, keepdims=False)

  num_dst_index_vals = 0
  if dst_indexers:
    if len(dst_indexers) != 1:
      raise NotImplementedError("Multiple indexers not supported.")
    idx = dst_indexers[0]
    if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
      (_, slice_starts, slice_sizes,
       num_scalar_idxs) = _find_slice_start_size(idx)
      num_dst_index_vals += num_scalar_idxs
      if updates.shape != slice_sizes:
        raise ValueError("DMA src and dst slices must have same shape. "
                         f"Got src={updates.shape}, dst={slice_sizes}")
      new_dst = jax.lax.dynamic_update_slice(
          dst_ref, updates, slice_starts)
    else:
      new_dst = dst_ref.at[idx.indices].set(updates)
  else:
    new_dst = updates

  # TODO(b/345505876): Implement semaphore counting.
  new_avals = (None,)  # src_aval
  new_avals += (None,) * num_src_index_vals
  new_avals += (new_dst,)  # dst_aval
  new_avals += (None,) * num_dst_index_vals
  new_avals += (None,)  # dst_sem_aval
  if is_remote:
    new_avals += (None, None)  # src_sem_aval, device_id
  assert (len(new_avals) ==
          len(in_avals)), f"{len(new_avals), new_avals} != {len(in_avals)}"
  return new_avals, []

state_discharge.register_discharge_rule(dma_start_p)(dma_start_discharge_rule)


dma_wait_p = jax_core.Primitive('dma_wait')
dma_wait_p.multiple_results = True

@dma_wait_p.def_abstract_eval
def _dma_wait_abstract_eval(*args, tree, device_id_type):
  del args, tree, device_id_type
  return []

def _dma_wait_pp_eqn(eqn: jax_core.JaxprEqn,
                     context: jax_core.JaxprPpContext,
                     settings: jax_core.JaxprPpSettings):
  del settings
  invars = eqn.invars
  tree = eqn.params["tree"]
  sem, sem_indexers, ref, indexers = tree_util.tree_unflatten(tree, invars)
  return pp.concat([
      pp.text('dma_wait'),
      pp.text(' '),
      sp.pp_ref_indexers(context, ref, indexers),
      pp.text(' '),
      sp.pp_ref_indexers(context, sem, sem_indexers),
  ])

jax_core.pp_eqn_rules[dma_wait_p] = _dma_wait_pp_eqn

def dma_wait_discharge_rule(in_avals, out_avals,
                             *args, tree, device_id_type):
  del out_avals, args, tree, device_id_type
  # TODO(justinfu): Implement semaphore counting.
  return (None,) * len(in_avals), []
state_discharge.register_discharge_rule(dma_wait_p)(dma_wait_discharge_rule)

def _get_ref_and_indexers(ref):
  if isinstance(ref, state.RefView):
    return ref.ref, ref.indexers
  return ref, ()

def make_async_copy(src_ref, dst_ref, sem):
  """Issues a DMA copying from src_ref to dst_ref."""
  src_ref, src_indexers = _get_ref_and_indexers(src_ref)
  dst_ref, dst_indexers = _get_ref_and_indexers(dst_ref)
  sem, sem_indexers = _get_ref_and_indexers(sem)
  return AsyncCopyDescriptor(src_ref, src_indexers, dst_ref, dst_indexers,
                             sem, sem_indexers, None, None, None,
                             DeviceIdType.MESH)

def async_copy(src_ref, dst_ref, sem):
  """Issues a DMA copying from src_ref to dst_ref."""
  copy_descriptor = make_async_copy(src_ref, dst_ref, sem)
  copy_descriptor.start()
  return copy_descriptor

def make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id,
                           device_id_type: DeviceIdType = DeviceIdType.MESH):
  """Creates a description of a remote copy operation.

  Copies data from src_ref on the current device to dst_ref on the device
  specified by device_id. Both semaphores should be waited on using the
  descriptor on both source and target devices.

  Note that device_id can also refer to the current device.

  Args:
    src_ref: The source Reference.
    dst_ref: The destination Reference.
    send_sem: The semaphore on the source device.
    recv_sem: The semaphore on the destination device.
    device_id: The device id of the destination device.
    device_id_type: The type of the device id.
  Returns:
    An AsyncCopyDescriptor.
  """
  src_ref, src_indexers = _get_ref_and_indexers(src_ref)
  send_sem, send_sem_indexers = _get_ref_and_indexers(send_sem)
  dst_ref, dst_indexers = _get_ref_and_indexers(dst_ref)
  recv_sem, recv_sem_indexers = _get_ref_and_indexers(recv_sem)
  return AsyncCopyDescriptor(
      src_ref, src_indexers, dst_ref, dst_indexers, recv_sem, recv_sem_indexers,
      send_sem, send_sem_indexers, device_id, device_id_type=device_id_type)

def async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id,
                      device_id_type: DeviceIdType = DeviceIdType.MESH):
  copy_descriptor = make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem,
                                           device_id, device_id_type)
  copy_descriptor.start()
  return copy_descriptor

device_id_p = jax_core.Primitive('device_id')

@device_id_p.def_abstract_eval
def _device_id_abstract_eval():
  return jax_core.ShapedArray((), jnp.dtype("int32"))

device_id = device_id_p.bind

get_barrier_semaphore_p = jax_core.Primitive('get_barrier_semaphore')

@get_barrier_semaphore_p.def_abstract_eval
def _get_barrier_semaphore_abstract_eval():
  return pl_core.AbstractMemoryRef(
      jax_core.ShapedArray((), tpu_core.BarrierSemaphoreTy()),
      tpu_core.TPUMemorySpace.SEMAPHORE,
  )

def get_barrier_semaphore():
  """Returns a barrier semaphore.

  This function returns a barrier semaphore based on the collective_id of the
  current pallas kernel.

  It's very important that the semaphore is wait-ed back down to 0, or else the
  semaphores will become corrupted.

  It's also very important that the collective_id is different for each pallas
  kernel with communication. E.g. if you have two pallas kernels, one that syncs
  across the X axis of the device mesh and the second that syncs across the Y
  axis, they must have different collective_ids.
  However it is legal for two kernels that perform the same synchronization
  pattern (e.g. only communicating with neighbours on the same mesh axis)
  to share a collective_id. However, if in doubt, prefer not sharing
  collective_ids, as doing so incorrectly can lead to silent data corruption or
  crashes.
  Note that re-using the same collective_id doesn't guarantee that the same
  semaphore is provided by XLA.
  """
  return get_barrier_semaphore_p.bind()

delay_p = jax_core.Primitive("delay")
delay_p.multiple_results = True


@delay_p.def_abstract_eval
def _delay_abstract_eval(nanos):
  del nanos
  return []


def delay(nanos):
  """Delays vector execution for the given number of nanosconds."""
  delay_p.bind(nanos)


# RNG Ops
prng_seed_p = jax_core.Primitive("prng_seed")
prng_seed_p.multiple_results = True

@prng_seed_p.def_abstract_eval
def _(*_):
  return []


def prng_seed(*seeds: int | jax.Array) -> None:
  """Sets the seed for PRNG.

  Args:
    seeds: One or more integer seeds for setting the PRNG seed. If
      more than one seed is passed in, the seed material will be
      mixed before setting the internal PRNG state.
  """
  prng_seed_p.bind(*seeds)

prng_random_bits_p = jax_core.Primitive(
    'prng_random_bits')

@prng_random_bits_p.def_abstract_eval
def _(*, shape):
  return jax_core.ShapedArray(shape, jnp.dtype("int32"))

def prng_random_bits(shape):
  return prng_random_bits_p.bind(shape=shape)
