# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import contextlib
import ctypes
import functools
import itertools
import json
import math
import warnings

import jax
from jax._src.interpreters import mlir
from jax._src.lib import xla_client
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import scf
import numpy as np

from .utils import *  # noqa: F403


try:
  from jax._src.lib import mosaic_gpu as mosaic_gpu_lib

  xla_client.register_custom_call_target(
      "mosaic_gpu_record_event",
      mosaic_gpu_lib._mosaic_gpu_ext._record_event_capsule(),
      platform="CUDA",
  )
except ImportError:
  pass

# ruff: noqa: F405
# mypy: ignore-errors


record_event_p = jax.core.Primitive("record_event")
record_event_p.multiple_results = True

@record_event_p.def_abstract_eval
def _record_event_abstract_eval(*args, event):
  del event  # Unused.
  return args

@functools.partial(mlir.register_lowering, record_event_p, platform="cuda")
def _record_event_lowering_rule(ctx, *args, event):
  ptr_bytes = ctypes.cast(event, ctypes.c_void_p).value.to_bytes(
      8, byteorder="little"
  )  # pytype: disable=attribute-error
  op = mlir.custom_call(
      "mosaic_gpu_record_event",
      result_types=[mlir.aval_to_ir_type(aval) for aval in ctx.avals_out],
      operands=args,
      backend_config=ptr_bytes,
      operand_output_aliases={i: i for i in range(len(args))},
  )
  return op.results

def _record_event(args, event):
  flat_args, treedef = jax.tree.flatten(args)
  return jax.tree.unflatten(
      treedef, record_event_p.bind(*flat_args, event=event)
  )

def measure(f, *args, **kwargs):
  # TODO(apaszke): Raise if this is called under jit.
  start_event = mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_create()
  end_event = mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_create()
  try:

    @jax.jit
    def run(*args, **kwargs):
      flat_args, treedef = jax.tree.flatten((args, kwargs))
      flat_args = _record_event(flat_args, start_event)
      args, kwargs = jax.tree.unflatten(treedef, flat_args)
      return _record_event(f(*args, **kwargs), end_event)

    jax.block_until_ready(run(*args, **kwargs))  # Warmup.
    results = jax.block_until_ready(run(*args, **kwargs))
    elapsed = mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_elapsed(
        start_event, end_event
    )
  finally:
    mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_destroy(start_event)
    mosaic_gpu_lib._mosaic_gpu_ext._gpu_event_destroy(end_event)
  return results, elapsed


class ProfilerSpec:
  ENTER = 0
  EXIT = 1 << 31

  def __init__(self, entries_per_warpgroup: int):
    self.entries_per_warpgroup = entries_per_warpgroup
    self.interned_names = {}

  def _num_warpgroups(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> int:
    if math.prod(block) % WARPGROUP_SIZE:
      raise ValueError("Block size is not a multiple of warpgroup size")
    return math.prod(grid) * math.prod(block) // WARPGROUP_SIZE

  def mlir_buffer_type(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> ir.Type:
    return ir.MemRefType.get(
        (self._num_warpgroups(grid, block) * self.entries_per_warpgroup,),
        ir.IntegerType.get_signless(32),
    )

  def jax_buffer_type(
      self, grid: tuple[int, ...], block: tuple[int, ...]
  ) -> ir.Type:
    return jax.ShapeDtypeStruct(
        (self._num_warpgroups(grid, block) * self.entries_per_warpgroup,),
        jnp.uint32,
    )

  def smem_i32_elements(self, block: tuple[int, ...]):
    num_warpgroups = self._num_warpgroups((), block)
    return int(num_warpgroups * self.entries_per_warpgroup)

  def smem_bytes(self, block: tuple[int, ...]):
    bytes_per_entry = 4
    return self.smem_i32_elements(block) * bytes_per_entry

  def intern_name(self, name: str) -> int:
    if (name_id := self.interned_names.get(name, None)) is not None:
      return name_id
    name_id = self.interned_names[name] = len(self.interned_names)
    if name_id & self.EXIT:
      raise RuntimeError("Allocated too many names")
    return name_id

  def dump(self, buffer, f, grid: tuple[int, ...], block: tuple[int, ...]):
    buffer = np.asarray(buffer)
    num_blocks = math.prod(grid)
    warpgroups_per_block = self._num_warpgroups((), block)
    entries = buffer.reshape(
        num_blocks, warpgroups_per_block, self.entries_per_warpgroup
    )
    start_times = entries[..., 0]
    sm_ids = entries[..., 1]
    entries_used = entries[..., 2]
    if np.any(entries_used > self.entries_per_warpgroup - 2):
      raise RuntimeError("Insufficient space to capture a full trace")
    traces = entries[..., 3:]
    unintern = {v: k for k, v in self.interned_names.items()}
    events = []
    for block_idx, wg_idx in np.ndindex(num_blocks, warpgroups_per_block):
      valid_entries = entries_used[block_idx, wg_idx] - 3
      local_clock_offset = None
      assert valid_entries % 2 == 0, valid_entries
      start_time = start_times[block_idx, wg_idx]
      block_events = []
      last_time = float("-inf")
      for i in range(0, valid_entries, 2):
        tag = traces[block_idx, wg_idx, i]
        time = traces[block_idx, wg_idx, i + 1]
        if local_clock_offset is None:
          local_clock_offset = time
        time -= local_clock_offset
        time -= i * 6  # Account for the overhead of profiling.
        if time < 0:
          break  # Detect a timer wraparound
        name_id = tag
        begin = True
        if name_id & ProfilerSpec.EXIT:
          name_id = name_id ^ ProfilerSpec.EXIT
          begin = False
        name = unintern[name_id]
        if last_time >= time:
          if last_time - time > 10:
            warnings.warn(
                "Profiler clock went significantly backwards for event"
                f" {'start' if begin else 'end'} `{name}`: {last_time} ->"
                f" {time}"
            )
          time = last_time + 1
        last_time = time
        block_events.append({
            "name": name,
            "ph": "B" if begin else "E",
            "ts": float(start_time + time) / 1e3,
            "pid": 1 + int(sm_ids[block_idx, wg_idx]),
            "tid": 1 + wg_idx + warpgroups_per_block * block_idx,
        })
      else:  # If we didn't break
        events.append(block_events)
    events = sorted(events, key=lambda x: x[0]["ts"])
    flat_events = list(itertools.chain.from_iterable(events))
    return json.dump({"displayTimeUnit": "ns", "traceEvents": flat_events}, f)


class OnDeviceProfiler:

  def __init__(self, spec: ProfilerSpec, smem_buffer: ir.Value, gmem_buffer: ir.Value):
    self.spec = spec
    self.start = globaltimer("low")
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    self.entries_per_wg = spec.entries_per_warpgroup
    wg_idx = warpgroup_idx(sync=False)
    self.smem_buffer = memref_slice(
        smem_buffer,
        ds(
            arith.index_cast(
                index, arith.muli(wg_idx, c(self.entries_per_wg, i32))
            ),
            self.entries_per_wg,
        ),
    )
    self.smem_buffer_ptr = memref_ptr(self.smem_buffer, memory_space=3)
    self.gmem_buffer = gmem_buffer
    self.is_profiling_thread = arith.cmpi(
        arith.CmpIPredicate.eq,
        arith.remui(thread_idx(), c(WARPGROUP_SIZE, i32)),
        c(0, i32),
    )
    # Hopefully mem2reg will remove the allocation.
    self.offset = memref.alloca(ir.MemRefType.get((), i32), [], [])
    memref.store(c(0, i32), self.offset, [])

  @contextlib.contextmanager
  def record(self, name: str):
    i32 = ir.IntegerType.get_signless(32)
    name_id = self.spec.intern_name(name)
    def store(modifier):
      cur = memref.load(self.offset, [])
      i64 = ir.IntegerType.get_signless(64)
      base_addr = arith.addi(
          llvm.ptrtoint(i64, self.smem_buffer_ptr),
          arith.extui(i64, arith.muli(cur, c(4, i32))),
      )
      llvm.inline_asm(
          ir.Type.parse("!llvm.void"),
          [self.is_profiling_thread, base_addr, c(modifier | name_id, i32)],
          """
          @$0 st.shared.v2.u32 [$1], {$2, %clock};
          """,
          "b,l,r",
          has_side_effects=True,
      )
      memref.store(
          arith.addi(cur, c(2, cur.type)),
          self.offset,
          [],
      )
    store(ProfilerSpec.ENTER)
    yield
    store(ProfilerSpec.EXIT)

  def finalize(self, grid: tuple[int, ...], block: tuple[int, ...]):
    index = ir.IndexType.get()
    i32 = ir.IntegerType.get_signless(32)

    gpu.barrier()   # Make sure all warpgroups are done.

    block_idx = c(0, index)
    for dim in gpu.Dimension:  # pytype: disable=wrong-arg-types
      block_idx = arith.addi(
          arith.muli(block_idx, gpu.grid_dim(dim)), gpu.block_id(dim)
      )
    wg_idx = warpgroup_idx(sync=False)
    wg_per_block = math.prod(block) // WARPGROUP_SIZE
    global_wg_idx = arith.addi(
        arith.muli(block_idx, c(wg_per_block, index)),
        arith.index_cast(index, wg_idx),
    )
    start_offset = arith.muli(global_wg_idx, c(self.entries_per_wg, index))
    wg_gmem_buffer = memref.subview(
        self.gmem_buffer, [start_offset], [self.entries_per_wg], [1],
        result_type=ir.Type.parse(
            f"memref<{self.entries_per_wg}xi32, strided<[1], offset: ?>>"
        ),
    )
    thread_in_wg = arith.remui(thread_idx(), c(128, i32))
    if_first = scf.IfOp(
        arith.cmpi(arith.CmpIPredicate.eq, thread_in_wg, c(0, i32))
    )
    with ir.InsertionPoint(if_first.then_block):
      memref.store(self.start, wg_gmem_buffer, [c(0, index)])
      memref.store(smid(), wg_gmem_buffer, [c(1, index)])
      memref.store(
          arith.addi(memref.load(self.offset, []), c(3, i32)),
          wg_gmem_buffer,
          [c(2, index)],
      )

      for_op = scf.ForOp(
          c(0, index),
          c(self.entries_per_wg - 3, index),
          c(1, index),
      )
      with ir.InsertionPoint(for_op.body):
        x = memref.load(self.smem_buffer, [for_op.induction_variable])
        memref.store(
            x,
            wg_gmem_buffer,
            [arith.addi(for_op.induction_variable, c(3, index))],
        )
        scf.yield_([])
      scf.yield_([])
