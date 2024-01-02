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
from jax._src.state import primitives as state_primitives
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import indexing
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
    sem,
    inc: int | jax.Array = 1,
    *,
    device_id: int | jax.Array | None | tuple[int | jax.Array, ...] = None,
    device_id_type: DeviceIdType = DeviceIdType.MESH,
):
  inc = jnp.asarray(inc, dtype=jnp.int32)
  args = [sem, inc]
  has_device_id = device_id is not None
  device_id_tree = None
  if has_device_id:
    flat_device_id, device_id_tree = tree_util.tree_flatten(device_id)
    args = [*args, *flat_device_id]
  semaphore_signal_p.bind(
      *args,
      has_device_id=has_device_id,
      device_id_type=device_id_type,
      device_id_tree=device_id_tree,
  )


@semaphore_signal_p.def_abstract_eval
def _semaphore_signal_abstract_eval(
    sem_aval: tpu_core.AbstractSemaphore,
    value,
    *args,
    has_device_id: bool,
    device_id_type: DeviceIdType,
    device_id_tree,
):
  del device_id_type
  del device_id_tree
  if not isinstance(sem_aval, tpu_core.AbstractSemaphore):
    raise ValueError(f"Cannot signal on a non-semaphore value: {sem_aval}")
  if sem_aval.sem_type not in {
      tpu_core.SemaphoreType.REGULAR,
      tpu_core.SemaphoreType.BARRIER,
  }:
    raise ValueError("Must signal a REGULAR or BARRIER semaphore.")
  if value.dtype != jnp.dtype("int32"):
    raise ValueError("Must signal an int32 value.")
  if has_device_id:
    device_id = args[0]
    if device_id.dtype != jnp.dtype("int32"):
      raise ValueError("`device_id` must be an int32 value.")
  return []


semaphore_wait_p = jax_core.Primitive('semaphore_wait')
semaphore_wait_p.multiple_results = True

def semaphore_wait(sem, dec: int | jax.Array = 1):
  dec = jnp.asarray(dec, dtype=jnp.int32)
  semaphore_wait_p.bind(sem, dec)

@semaphore_wait_p.def_abstract_eval
def _semaphore_wait_abstract_eval(sem_aval: tpu_core.AbstractSemaphore, value):
  if not isinstance(sem_aval, tpu_core.AbstractSemaphore):
    raise ValueError(f"Cannot wait on a non-semaphore value: {sem_aval}")
  if sem_aval.sem_type not in {
      tpu_core.SemaphoreType.REGULAR,
      tpu_core.SemaphoreType.BARRIER,
  }:
    raise ValueError("Must wait on a REGULAR or BARRIER semaphore.")
  if value.dtype != jnp.dtype("int32"):
    raise ValueError("Must signal an int32 value.")
  return []


@dataclasses.dataclass
class AsyncCopyDescriptor:
  src_ref: Any
  src_indexer: indexing.NDIndexer
  dst_ref: Any
  dst_indexer: indexing.NDIndexer
  dst_sem: int | jax.Array
  src_sem: int | jax.Array | None
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
        self.src_indexer,
        self.dst_ref,
        self.dst_indexer,
        self.dst_sem,
        self.src_sem,
        self.device_id,
    ))
    dma_start_p.bind(*flat_args, tree=tree, device_id_type=self.device_id_type)

  def wait(self):
    if self.is_remote:
      self.wait_send()
    self.wait_recv()

  def wait_recv(self):
    wait_args, tree = tree_util.tree_flatten(
        (self.dst_sem, self.dst_ref, self.dst_indexer)
    )
    dma_wait_p.bind(
        *wait_args, tree=tree, device_id_type=self.device_id_type
    )

  def wait_send(self):
    if not self.is_remote:
      raise ValueError("Cannot `wait_send` on a local copy.")
    wait_args, tree = tree_util.tree_flatten(
        (self.src_sem, self.src_ref, self.src_indexer)
    )
    dma_wait_p.bind(
        *wait_args, tree=tree, device_id_type=self.device_id_type
    )


@dataclasses.dataclass
class AsyncCopy(AsyncCopyDescriptor):
  waited_for_send: bool = False
  waited_for_recv: bool = False

  def start(self):
    raise RuntimeError("The async copy is already running.")

  def wait_recv(self):
    if self.waited_for_recv:
      suffix = " for recv" if self.is_remote else ""
      raise RuntimeError(f"Already waited{suffix}.")
    self.waited_for_recv = True
    super().wait_recv()

  def wait_send(self):
    if self.waited_for_send:
      raise RuntimeError("Already waited for send.")
    self.waited_for_send = True
    super().wait_send()


dma_start_p = jax_core.Primitive('dma_start')
dma_start_p.multiple_results = True

@dma_start_p.def_abstract_eval
def _dma_start_abstract_eval(*args, tree, device_id_type):
  del args, tree, device_id_type
  return []

def _pp_slice(slc: indexing.Slice, dim: int, context: jax_core.JaxprPpContext
              ) -> str:
  start, size = slc.start, slc.size
  if isinstance(start, jax_core.Var):
    start_str = jax_core.pp_var(start, context)
    end_str = f'{start_str}+{size}'
  else:
    start_str = '' if start == 0 else str(start)
    end = start + size
    end_str = '' if end == dim else str(end)
  return f'{start_str}:{end_str}'

def _pp_indexer(indexer: indexing.NDIndexer,
                context: jax_core.JaxprPpContext) -> pp.Doc:
  indices = []
  for idx, dim in zip(indexer.indices, indexer.shape):
    if isinstance(idx, indexing.Slice):
      indices.append(_pp_slice(idx, dim, context))
    else:
      indices.append(jax_core.pp_var(idx, context))  # type: ignore
  return pp.text(','.join(indices))

def _pp_ref(ref, indexer, context):
  return state_primitives.pp_ref(
      pp.concat([
          pp.text(jax_core.pp_var(ref, context)),
          pp.text("["),
          _pp_indexer(indexer, context),
          pp.text("]"),
      ])
  )

def _dma_start_pp_eqn(eqn: jax_core.JaxprEqn,
                      context: jax_core.JaxprPpContext,
                      settings: jax_core.JaxprPpSettings):
  invars = eqn.invars
  tree = eqn.params["tree"]
  src_ref, src_indexer, dst_ref, dst_indexer, dst_sem, src_sem, device_id = (
      tree_util.tree_unflatten(tree, invars)
  )
  if src_sem or device_id:
    return jax_core._pp_eqn(eqn, context, settings)
  return pp.concat([
      pp.text('dma_start'),
      pp.text(' '),
      _pp_ref(src_ref, src_indexer, context),
      pp.text(' -> '),
      _pp_ref(dst_ref, dst_indexer, context),
      pp.text(' '),
      pp.text(jax_core.pp_var(dst_sem, context)),
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
  invars = eqn.invars
  tree = eqn.params["tree"]
  sem, ref, indexer = tree_util.tree_unflatten(tree, invars)
  return pp.concat([
      pp.text('dma_wait'),
      pp.text(' '),
      _pp_ref(ref, indexer, context),
      pp.text(' '),
      pp.text(jax_core.pp_var(sem, context)),
  ])

jax_core.pp_eqn_rules[dma_wait_p] = _dma_wait_pp_eqn

def _get_ref_and_indexer(ref):
  if isinstance(ref, state.RefView):
    return ref.ref, ref.indexer
  return ref, (slice(None),) * len(ref.shape)

def _make_async_copy(src_ref, dst_ref, sem, desc_cls):
  """Issues a DMA copying from src_ref to dst_ref."""
  src_ref, src_indices = _get_ref_and_indexer(src_ref)
  dst_ref, dst_indices = _get_ref_and_indexer(dst_ref)
  src_idx = indexing.NDIndexer.from_indices_shape(src_indices, src_ref.shape)
  dst_idx = indexing.NDIndexer.from_indices_shape(dst_indices, dst_ref.shape)
  return desc_cls(
      src_ref, src_idx, dst_ref, dst_idx, sem,
      src_sem=None, device_id=None, device_id_type=DeviceIdType.MESH,
  )

def make_async_copy(src_ref, dst_ref, sem):
  return _make_async_copy(src_ref, dst_ref, sem, AsyncCopyDescriptor)

def async_copy(src_ref, dst_ref, sem):
  """Issues a DMA copying from src_ref to dst_ref."""
  copy = _make_async_copy(src_ref, dst_ref, sem, AsyncCopy)
  AsyncCopyDescriptor.start(copy)  # Override the .start() error
  return copy

def _make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id,
                            device_id_type, desc_cls):
  src_ref, src_indices = _get_ref_and_indexer(src_ref)
  dst_ref, dst_indices = _get_ref_and_indexer(dst_ref)
  src_idx = indexing.NDIndexer.from_indices_shape(src_indices, src_ref.shape)
  dst_idx = indexing.NDIndexer.from_indices_shape(dst_indices, dst_ref.shape)
  return desc_cls(
      src_ref, src_idx, dst_ref, dst_idx, recv_sem,
      send_sem, device_id, device_id_type,
  )

def make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id,
                           device_id_type: DeviceIdType = DeviceIdType.MESH):
  return _make_async_remote_copy(
      src_ref, dst_ref, send_sem, recv_sem,
      device_id, device_id_type, AsyncCopyDescriptor
  )

def async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id,
                      device_id_type: DeviceIdType = DeviceIdType.MESH):
  copy = _make_async_remote_copy(
      src_ref, dst_ref, send_sem, recv_sem,
      device_id, device_id_type, AsyncCopy)
  AsyncCopyDescriptor.start(copy)
  return copy

device_id_p = jax_core.Primitive('device_id')

@device_id_p.def_abstract_eval
def _device_id_abstract_eval():
  return jax_core.ShapedArray((), jnp.dtype("int32"))

device_id = device_id_p.bind

get_barrier_semaphore_p = jax_core.Primitive('get_barrier_semaphore')

@get_barrier_semaphore_p.def_abstract_eval
def _get_barrier_semaphore_abstract_eval():
  return tpu_core.AbstractSemaphore(tpu_core.SemaphoreType.BARRIER)

def get_barrier_semaphore():
  return get_barrier_semaphore_p.bind()
