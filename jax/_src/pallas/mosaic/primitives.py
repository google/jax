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

import contextlib
import dataclasses
import enum
from typing import Any, Callable

import jax
from jax._src import api_util
from jax._src import core as jax_core
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
from jax._src.pallas.mosaic import core as tpu_core
import jax.numpy as jnp

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

trace_start_p = jax_core.Primitive('trace_start')
trace_start_p.multiple_results = True


@trace_start_p.def_impl
def _trace_start_impl(*, message: str, level: int):
  del message, level
  return []

@trace_start_p.def_abstract_eval
def _trace_start_abstract_eval(*, message: str, level: int):
  del message, level
  return []

mlir.register_lowering(trace_start_p, lambda ctx, **_: [])


trace_stop_p = jax_core.Primitive('trace_stop')
trace_stop_p.multiple_results = True

@trace_stop_p.def_impl
def _trace_stop_impl():
  return []

@trace_stop_p.def_abstract_eval
def _trace_stop_abstract_eval():
  return []

mlir.register_lowering(trace_stop_p, lambda ctx: [])

@contextlib.contextmanager
def trace(message: str, level: int = 10):
  trace_start_p.bind(message=message, level=level)
  yield
  trace_stop_p.bind()


run_scoped_p = jax_core.Primitive('run_scoped')
run_scoped_p.multiple_results = True


def run_scoped(f: Callable[..., None], *types, **kw_types) -> None:
  flat_types, in_tree = tree_util.tree_flatten((types, kw_types))
  flat_fun, _ = api_util.flatten_fun(lu.wrap_init(f), in_tree)
  avals = map(lambda t: t.get_aval(), flat_types)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, avals)
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


semaphore_signal_p = jax_core.Primitive('semaphore_signal')
semaphore_signal_p.multiple_results = True


def semaphore_signal(
    sem_or_view,
    inc: int | jax.Array = 1,
    *,
    device_id: int | jax.Array | None | tuple[int | jax.Array, ...] = None,
    device_id_type: DeviceIdType = DeviceIdType.MESH,
):
  ref, indexers = _get_ref_and_indexers(sem_or_view)
  inc = jnp.asarray(inc, dtype=jnp.int32)
  args = [ref, indexers, inc, device_id]
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
  sem_aval, sem_indexers_avals, value_aval, device_id_avals = (
      tree_util.tree_unflatten(args_tree, avals)
  )
  if not isinstance(sem_aval, state.AbstractRef):
    raise ValueError(f"Cannot signal on a non-Ref: {sem_aval}")
  sem_shape = sem_aval.shape
  if sem_indexers_avals:
    sem_shape = sem_indexers_avals[-1].get_indexer_shape()
  if sem_shape:
    raise ValueError(f"Cannot signal on a non-()-shaped semaphore: {sem_shape}")
  sem_dtype = sem_aval.dtype
  if not (jnp.issubdtype(sem_dtype, tpu_core.semaphore) or jnp.issubdtype(
      sem_dtype, tpu_core.barrier_semaphore)):
    raise ValueError(f"Must signal a REGULAR or BARRIER semaphore: {sem_dtype}")
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
    device_ids_pp = [flat_device_ids[0]]
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
  if not isinstance(sem_aval, state.AbstractRef):
    raise ValueError(f"Cannot signal on a non-semaphore Ref: {sem_aval}")
  sem_shape = sem_aval.shape
  if sem_indexers_avals:
    sem_shape = sem_indexers_avals[-1].get_indexer_shape()
  if sem_shape:
    raise ValueError(f"Cannot signal on a non-()-shaped semaphore: {sem_shape}")
  sem_dtype = sem_aval.dtype
  if not (jnp.issubdtype(sem_dtype, tpu_core.semaphore) or jnp.issubdtype(
      sem_dtype, tpu_core.barrier_semaphore)):
    raise ValueError(f"Must signal a REGULAR or BARRIER semaphore: {sem_dtype}")
  if value_aval.dtype != jnp.dtype("int32"):
    raise ValueError("Must signal an int32 value.")
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
  return tpu_core.AbstractMemoryRef(
      jax_core.ShapedArray((), tpu_core.BarrierSemaphoreTy()),
      tpu_core.TPUMemorySpace.SEMAPHORE,
  )

def get_barrier_semaphore():
  return get_barrier_semaphore_p.bind()
