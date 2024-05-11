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

from collections.abc import Callable
import contextlib
import ctypes
import dataclasses
import functools
import os
import pathlib
import subprocess
import tempfile
import time
from typing import Any, Sequence

import jax
from jax._src import config
from jax._src import core as jax_core
from jax._src.interpreters import mlir
from jax._src.lib import xla_client
from jax._src.lib import mosaic_gpu as mosaic_gpu_lib
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvgpu
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.execution_engine import ExecutionEngine
from jaxlib.mlir.passmanager import PassManager
import numpy as np

from . import dsl as mgpu
from . import profiler
from . import utils

# mypy: ignore-errors

# MLIR can't find libdevice unless we point it to the CUDA path
# TODO(apaszke): Unify with jax._src.lib.cuda_path
CUDA_ROOT = "/usr/local/cuda"
if os.environ.get("CUDA_ROOT") is None:
  os.environ["CUDA_ROOT"] = CUDA_ROOT
else:
  CUDA_ROOT = os.environ["CUDA_ROOT"]

PTXAS_PATH = os.path.join(CUDA_ROOT, "bin/ptxas")
NVDISASM_PATH = os.path.join(CUDA_ROOT, "bin/nvdisasm")

TMA_DESCRIPTOR_BYTES = 128
TMA_DESCRIPTOR_ALIGNMENT = 64


c = mgpu.c  # This is too common to fully qualify.


xla_client.register_custom_call_target(
    "mosaic_gpu",
    mosaic_gpu_lib._mosaic_gpu_ext._custom_call_capsule(),
    platform="CUDA",
)
mosaic_gpu_lib._mosaic_gpu_ext.register_passes()


mosaic_gpu_dump_ptx = config.define_bool_state(
    name="mosaic_gpu_dump_ptx",
    default=config.bool_env("MOSAIC_GPU_DUMP_PTX", False),
    help="If set, prints the kernel PTX",
)
mosaic_gpu_dump_ptxas = config.define_bool_state(
    name="mosaic_gpu_dump_ptxas",
    default=config.bool_env("MOSAIC_GPU_DUMP_PTXAS", False),
    help="If set, prints the ptxas verbose output",
)
mosaic_gpu_dump_sass = config.define_bool_state(
    name="mosaic_gpu_dump_sass",
    default=config.bool_env("MOSAIC_GPU_DUMP_SASS", False),
    help="If set, prints the kernel SASS",
)
mosaic_gpu_print_after_all = config.define_bool_state(
    name='mosaic_gpu_print_after_all',
    default=config.bool_env('MOSAIC_GPU_PRINT_AFTER_ALL', False),
    help="If set, prints the kernel module after every pass",
)


mosaic_gpu_p = jax.core.Primitive("mosaic_gpu_p")
mosaic_gpu_p.multiple_results = True


@mosaic_gpu_p.def_abstract_eval
def _mosaic_gpu_abstract_eval(*_, module, out_types, gmem_scratch_bytes):
  del module, gmem_scratch_bytes  # Unused.
  return [jax._src.core.ShapedArray(t.shape, t.dtype) for t in out_types]


def _mosaic_gpu_lowering_rule(ctx, *args, module, out_types, gmem_scratch_bytes):
  del out_types  # Unused.
  runtime_path = (
      pathlib.Path(mosaic_gpu_lib._mosaic_gpu_ext.__file__).parent.parent.parent
      / "mosaic" / "gpu" / "libmosaic_gpu_runtime.so"
  )
  shared_libs = [str(runtime_path)] if runtime_path.exists() else []
  engine = ExecutionEngine(
      module, opt_level=3, shared_libs=shared_libs, enable_object_dump=False
  )
  ctx.module_context.add_keepalive(engine)
  launch_func_ptr = ctypes.cast(engine.lookup("main"), ctypes.c_void_p)
  init_func_ptr = ctypes.cast(engine.lookup("main_init"), ctypes.c_void_p)
  # Make sure we won't get accidental hits due to address reuse.
  mosaic_gpu_lib._mosaic_gpu_ext.invalidate_cache(init_func_ptr.value)

  trampoline_args = (ctypes.c_void_p * 2)()
  trampoline_args[0] = launch_func_ptr
  trampoline_args[1] = init_func_ptr
  ctx.module_context.add_keepalive(trampoline_args)
  ptr_bytes = ctypes.cast(trampoline_args, ctypes.c_void_p).value.to_bytes(
      8, byteorder="little"
  )  # pytype: disable=attribute-error
  op = mlir.custom_call(
      "mosaic_gpu",
      result_types=[
          *(mlir.aval_to_ir_type(aval) for aval in ctx.avals_out),
          mlir.aval_to_ir_type(
              jax_core.ShapedArray((gmem_scratch_bytes,), np.uint8)
          ),
      ],
      operands=args,
      backend_config=ptr_bytes,
  )
  return op.results[:-1]  # Skip the scratch space.

mlir.register_lowering(mosaic_gpu_p, _mosaic_gpu_lowering_rule, "cuda")


@dataclasses.dataclass(frozen=True)
class MemRefTransform:
  def apply(self, ref: ir.Value) -> ir.Value:
    raise NotImplementedError("Subclasses should override this method")

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    raise NotImplementedError("Subclasses should override this method")

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    raise NotImplementedError("Subclasses should override this method")


@dataclasses.dataclass(frozen=True)
class TileTransform(MemRefTransform):
  """Tiles a suffix of memref dimensions.

  For example, given a memref of shape (5, 128, 128) and a tiling of (64, 32),
  the shape of the result will be (5, 2, 4, 64, 32). The shape always ends with
  the tile shape, and the size of tiled dimensions is divided by the tile size.
  This is especially useful for swizzled WGMMA, which expect tiled layouts in
  shared memory.
  """
  tiling: tuple[int, ...]

  def apply(self, ref: ir.Value) -> ir.Value:
    untiled_rank = ir.MemRefType(ref.type).rank
    tiling_rank = len(self.tiling)
    tiled_rank = untiled_rank + tiling_rank
    for t, d in zip(self.tiling[::-1], range(untiled_rank)[::-1]):
      ref = mgpu.memref_unfold(ref, d, (None, t))
    permutation = (
        *range(untiled_rank - tiling_rank),
        *range(untiled_rank - tiling_rank, tiled_rank, 2),
        *range(untiled_rank - tiling_rank + 1, tiled_rank, 2),
    )
    return mgpu.memref_transpose(ref, permutation)

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    index = ir.IndexType.get()
    tiling_rank = len(self.tiling)
    return (
        *idx[:-tiling_rank],
        *(
            arith.divui(i, c(t, index))
            for i, t in zip(idx[-tiling_rank:], self.tiling)
        ),
        *([c(0, index)] * tiling_rank),
    )

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    # Note that this also checks that tiled dims are not squeezed. Their slice
    # size would be 1 if so.
    tiling_rank = len(self.tiling)
    for size, tile_size in zip(shape[-tiling_rank:], self.tiling):
      if size % tile_size:
        raise ValueError(
            f"Expected GMEM slice shape {shape} suffix to be a multiple"
            f" of tiling {self.tiling}"
        )
    return (
        *shape[:-tiling_rank],
        *(s // t for s, t in zip(shape[-tiling_rank:], self.tiling)),
        *self.tiling,
    )


@dataclasses.dataclass(frozen=True)
class TransposeTransform(MemRefTransform):
  """Transposes memref dimensions."""
  permutation: tuple[int, ...]

  def __post_init__(self):
    if len(self.permutation) != len(set(self.permutation)):
      raise ValueError("Permutation must be a permutation")

  def apply(self, ref: ir.Value) -> ir.Value:
    return mgpu.memref_transpose(ref, self.permutation)

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    return tuple(idx[p] for p in self.permutation)

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(shape[p] for p in self.permutation)


OnDeviceProfiler = profiler.OnDeviceProfiler


@dataclasses.dataclass()
class LaunchContext:
  launch_op: gpu.LaunchOp
  gmem_scratch_ptr: ir.Value
  profiler: OnDeviceProfiler | None = None
  next_scratch_offset: int = 0
  host_scratch_init: list[Callable[[ir.Value], None]] = dataclasses.field(
      default_factory=list, init=False
  )
  tma_descriptors: dict[
      tuple[ir.Value, tuple[int, ...], int | None, tuple[MemRefTransform, ...]],
      ir.Value,
  ] = dataclasses.field(default_factory=dict, init=False)

  @contextlib.contextmanager
  def named_region(self, *args, **kwargs):
    if self.profiler is not None:
      with self.profiler.record(*args, **kwargs):
        yield
    else:
      yield

  def _alloc_scratch(
      self,
      size: int,
      alignment: int | None = None,
      host_init: Callable[[ir.Value], None] = lambda _: None,
      device_init: Callable[[ir.Value], Any] = lambda x: x,
  ) -> ir.Value:
    """Allocates a GMEM scratch buffer.

    The buffer is initialized on the host and then copied to GMEM before the
    kernel launch.
    """
    i8 = ir.IntegerType.get_signless(8)
    ptr_ty = ir.Type.parse("!llvm.ptr")
    if alignment is None:
      alignment = size
    if self.next_scratch_offset % alignment:
      raise NotImplementedError  # TODO(apaszke): Pad to match alignment
    alloc_base = self.next_scratch_offset
    self.next_scratch_offset += size
    def host_init_wrapped(host_ptr):
      with ir.InsertionPoint(self.launch_op):
        host_init(
            llvm.getelementptr(ptr_ty, host_ptr, [], [alloc_base], i8)
        )
    self.host_scratch_init.append(host_init_wrapped)
    with ir.InsertionPoint.at_block_begin(self.launch_op.body.blocks[0]):
      return device_init(llvm.getelementptr(
          ptr_ty, self.gmem_scratch_ptr, [], [alloc_base], i8
      ))

  def _get_tma_desc(
      self,
      ref,
      gmem_transform: tuple[MemRefTransform, ...],
      transformed_slice_shape: tuple[int, ...],
      swizzle: int | None,
  ):
    index = ir.IndexType.get()
    ref_ty = ir.MemRefType(ref.type)
    tma_desc_key = (ref, transformed_slice_shape, swizzle, gmem_transform)
    if (tma_desc := self.tma_descriptors.get(tma_desc_key, None)) is None:
      swizzle_str = f"swizzle_{swizzle}b" if swizzle is not None else "none"
      default_tensor_map_attrs = dict(
          swizzle=swizzle_str, l2promo="none", oob="zero", interleave="none"
      )
      tensor_map_ty = utils.get_tensormap_descriptor(
          tensor=(
              f"memref<{'x'.join(map(str, transformed_slice_shape))}x{ref_ty.element_type}, 3>"
          ),
          **default_tensor_map_attrs,
      )
      with ir.InsertionPoint(self.launch_op):
        for t in gmem_transform:
          ref = t.apply(ref)
        ref_ty = ir.MemRefType(ref.type)

        i64 = ir.IntegerType.get_signless(64)
        ptr_ty = ir.Type.parse("!llvm.ptr")
        def init_tma_desc(host_ptr):
          _, offset, *sizes_and_strides = memref.extract_strided_metadata(ref)
          aligned_ptr_idx = memref.extract_aligned_pointer_as_index(ref)
          as_i64 = lambda i: arith.index_cast(i64, i)
          alloc_ptr = llvm.inttoptr(ptr_ty, as_i64(aligned_ptr_idx))
          llvm_dyn = -2147483648  # TODO(apaszke): Improve the MLIR bindings...
          base_ptr = llvm.getelementptr(
              ptr_ty, alloc_ptr, [as_i64(offset)], [llvm_dyn], ref_ty.element_type,
          )
          rank = ref_ty.rank
          assert rank * 2 == len(sizes_and_strides)
          args = [
              host_ptr,
              base_ptr,
              c(utils.bytewidth(ref_ty.element_type), i64),
              c(rank, i64),
              utils.pack_array([as_i64(i) for i in sizes_and_strides[:rank]]),
              utils.pack_array([as_i64(i) for i in sizes_and_strides[rank:]]),
              c(0 if swizzle is None else swizzle, i64),
              utils.pack_array([c(v, i64) for v in transformed_slice_shape]),
          ]
          func.call([], "mosaic_gpu_init_tma_desc", args)
        def cast_tma_desc(device_ptr):
          # TODO(apaszke): Investigate why prefetching can cause launch failures
          # nvvm.prefetch_tensormap(device_ptr)
          return builtin.unrealized_conversion_cast(
              [tensor_map_ty], [device_ptr]
          )
        tma_desc = self._alloc_scratch(
            TMA_DESCRIPTOR_BYTES,
            alignment=TMA_DESCRIPTOR_ALIGNMENT,
            host_init=init_tma_desc,
            device_init=cast_tma_desc,
        )
        self.tma_descriptors[tma_desc_key] = tma_desc
    return tma_desc

  def async_copy(
      self,
      *,
      src_ref,
      dst_ref,
      gmem_slice: Any = (),
      gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (),
      barrier: mgpu.Barrier | None = None,
      swizzle: int | None = None,
      arrive: bool | None = None,
      uniform: bool = True,
  ):
    index = ir.IndexType.get()
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    src_ref_ty = ir.MemRefType(src_ref.type)
    dst_ref_ty = ir.MemRefType(dst_ref.type)
    element_type = src_ref_ty.element_type
    if element_type != dst_ref_ty.element_type:
      raise ValueError(
          f"Expected same element type, got {element_type} and"
          f" {dst_ref_ty.element_type}"
      )
    if not isinstance(gmem_transform, tuple):
      gmem_transform = (gmem_transform,)

    if src_ref_ty.memory_space is None and dst_ref_ty.memory_space == smem:
      gmem_ref, smem_ref = src_ref, dst_ref
      if barrier is None:
        raise ValueError("Barriers are required for GMEM -> SMEM copies")
      if arrive is None:
        arrive = True  # Arrive by default
    elif src_ref_ty.memory_space == smem and dst_ref_ty.memory_space is None:
      gmem_ref, smem_ref = dst_ref, src_ref
      if barrier is not None:
        raise ValueError("Barriers are unsupported for SMEM -> GMEM copies")
      if arrive is not None:
        raise ValueError("arrive is unsupported for SMEM -> GMEM copies")
    else:
      raise ValueError("Only SMEM <-> GMEM copies supported")
    # TODO(apaszke): This is a very approximate check. Improve it!
    expected_name = "builtin.unrealized_conversion_cast"
    if (
        gmem_ref.owner is None
        or gmem_ref.owner.opview.OPERATION_NAME != expected_name
    ):
      raise ValueError("GMEM reference in async_copy must be a kernel argument")

    base_indices, slice_shape, is_squeezed = utils.parse_indices(
        gmem_slice, ir.MemRefType(gmem_ref.type).shape
    )
    dyn_base_indices = tuple(
        c(i, index) if not isinstance(i, ir.Value) else i for i in base_indices
    )
    slice_shape = tuple(slice_shape)
    for t in gmem_transform:
      dyn_base_indices = t.transform_index(dyn_base_indices)
      slice_shape = t.transform_shape(slice_shape)
    for dim, squeezed in enumerate(is_squeezed):
      if squeezed:
        smem_ref = mgpu.memref_unsqueeze(smem_ref, dim)
    smem_ref_ty = ir.MemRefType(smem_ref.type)

    if slice_shape != tuple(smem_ref_ty.shape):
      raise ValueError(
          "Expected the SMEM reference to have the same shape as the tiled"
          f" slice: {tuple(smem_ref_ty.shape)} != {slice_shape}"
      )
    tma_desc = self._get_tma_desc(
        gmem_ref, gmem_transform, slice_shape, swizzle,
    )

    # nvgpu TMA instructions expect reversed indices...
    rev_dyn_based_indices = reversed(dyn_base_indices)

    uniform_ctx = functools.partial(mgpu.once, per_block=False) if uniform else contextlib.nullcontext

    if gmem_ref is src_ref:
      with uniform_ctx():
        assert barrier is not None  # for pytype
        barrier_group = barrier.barrier_array.value
        barrier_idx = barrier.offset
        if arrive:
          slice_bytes = c(
              np.prod(slice_shape) * mgpu.bytewidth(element_type), index
          )
          nvgpu.mbarrier_arrive_expect_tx(
              barrier_group, slice_bytes, barrier_idx
          )
        nvgpu.tma_async_load(
            smem_ref, barrier_group, tma_desc, rev_dyn_based_indices, barrier_idx
        )
    else:
      with uniform_ctx():
        nvgpu.tma_async_store(smem_ref, tma_desc, rev_dyn_based_indices)
        nvvm.cp_async_bulk_commit_group()

  def await_async_copy(
      self, allow_groups: int, await_read_only: bool = False
  ):
    nvvm.cp_async_bulk_wait_group(allow_groups, read=await_read_only)
    gpu.barrier()  # Groups are supposedly tracked per-thread


# TODO(apaszke): Inline this
@contextlib.contextmanager
def _launch(
    token,
    grid,
    block,
    gmem_scratch_ptr,
    smem_buffers,
    profiler_spec: profiler.ProfilerSpec | None = None,
    maybe_prof_buffer: ir.Value | None = None,
):
  if (profiler_spec is None) != (maybe_prof_buffer is None):
    raise ValueError
  index = ir.IndexType.get()
  i32 = ir.IntegerType.get_signless(32)
  i8 = ir.IntegerType.get_signless(8)
  grid_vals = [c(i, index) for i in grid]
  block_vals = [c(i, index) for i in block]
  flat_refs, smem_buffer_tree = jax.tree.flatten(smem_buffers)

  smem_ref_bytes = []
  for ref_ty in flat_refs:
    smem_ref_bytes.append(
        np.prod(ref_ty.shape) * np.dtype(ref_ty.dtype).itemsize
    )

  smem_bytes = sum(smem_ref_bytes)
  if profiler_spec is not None:
    smem_bytes += profiler_spec.smem_bytes(grid)

  launch_op = gpu.LaunchOp(
      token.type, [token], *grid_vals, *block_vals,
      dynamicSharedMemorySize=c(smem_bytes, i32))
  launch_op.body.blocks.append(*([index] * 12))  # Append an empty block
  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
  with ir.InsertionPoint(launch_op.body.blocks[0]):
    dynamic_smem = gpu.dynamic_shared_memory(
        ir.MemRefType.get(
            (ir.ShapedType.get_dynamic_size(),), i8, memory_space=smem
        )
    )
    smem_refs = []
    dynamic_smem_offset = 0
    for ref_ty, ref_bytes in zip(flat_refs, smem_ref_bytes):
      mlir_dtype = mlir.dtype_to_ir_type(ref_ty.dtype)
      tile_smem = memref.view(
          ir.MemRefType.get(ref_ty.shape, mlir_dtype, memory_space=smem),
          dynamic_smem, c(dynamic_smem_offset, index), [],
      )
      dynamic_smem_offset += ref_bytes
      smem_refs.append(tile_smem)

    if profiler_spec:
      prof_smem = memref.view(
          ir.MemRefType.get(
              (profiler_spec.smem_i32_elements(grid=grid),),
              i32, memory_space=smem,
          ),
          dynamic_smem, c(dynamic_smem_offset, index), [],
      )
      prof = profiler.OnDeviceProfiler(
          profiler_spec, prof_smem, maybe_prof_buffer
      )
    else:
      prof = None
    smem_ref_tree = jax.tree.unflatten(smem_buffer_tree, smem_refs)
    yield LaunchContext(launch_op, gmem_scratch_ptr, prof), smem_ref_tree
    if prof is not None:
      prof.finalize(grid=grid)
    gpu.terminator()


def as_gpu_kernel(
    body,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    in_shape,
    out_shape,
    smem_scratch_shape,
    prof_spec: profiler.ProfilerSpec | None = None,
):
  ptr_ty = ir.Type.parse("!llvm.ptr")
  token_ty = ir.Type.parse("!gpu.async.token")
  i8 = ir.IntegerType.get_signless(8)
  i64 = ir.IntegerType.get_signless(64)

  def _shape_to_ref_ty(shape: jax.ShapeDtypeStruct) -> ir.MemRefType:
    return ir.MemRefType.get(shape.shape, mlir.dtype_to_ir_type(shape.dtype))

  if isinstance(in_shape, list):
    in_shape = tuple(in_shape)
  elif not isinstance(in_shape, tuple):
    in_shape = (in_shape,)
  in_ref_tys = [_shape_to_ref_ty(t) for t in in_shape]

  unwrap_output_tuple = False
  if isinstance(out_shape, list):
    out_shape = tuple(out_shape)
  elif not isinstance(out_shape, tuple):
    out_shape = (out_shape,)
    unwrap_output_tuple = True
  out_ref_tys = [_shape_to_ref_ty(t) for t in out_shape]
  if prof_spec is not None:
    out_shape = (*out_shape, prof_spec.jax_buffer_type)
    out_ref_tys.append(prof_spec.mlir_buffer_type)

  module = ir.Module.create()
  with ir.InsertionPoint(module.body):
    _declare_runtime_functions()
    gmem_scratch_bytes = 0
    @func.FuncOp.from_py_func(ptr_ty, ptr_ty)
    def main(token_ptr, buffers):
      nonlocal gmem_scratch_bytes
      token = builtin.unrealized_conversion_cast([token_ty], [token_ptr])
      arg_refs = []
      i = -1
      for i, ref_ty in enumerate([*in_ref_tys, *out_ref_tys]):
        ptr = llvm.LoadOp(ptr_ty, llvm.GEPOp(ptr_ty, buffers, [], [i], ptr_ty))
        arg_refs.append(utils.ptr_as_memref(ptr, ir.MemRefType(ref_ty)))
      gmem_scratch_ptr = llvm.LoadOp(
          ptr_ty, llvm.GEPOp(ptr_ty, buffers, [], [i + 1], ptr_ty)
      )
      in_refs = arg_refs[:len(in_ref_tys)]
      out_refs = arg_refs[len(in_ref_tys):]
      prof_buffer = out_refs.pop() if prof_spec is not None else None
      with _launch(
          token, grid, block, gmem_scratch_ptr, smem_scratch_shape,
          prof_spec, prof_buffer
      ) as (launch_ctx, smem_refs):
        body(launch_ctx, *in_refs, *out_refs, smem_refs)
        gmem_scratch_bytes = launch_ctx.next_scratch_offset
      # Allocate and initialize the host buffer right before the launch.
      # Note that we couldn't do that before, because we had to run the body
      # to learn what the scratch contains.
      with ir.InsertionPoint(launch_ctx.launch_op):
        host_scratch_ptr = llvm.alloca(ptr_ty, c(gmem_scratch_bytes, i64), i8)
        for init_callback in launch_ctx.host_scratch_init:
          init_callback(host_scratch_ptr)
        func.call(
            [],
            "mosaic_gpu_memcpy_async_h2d",
            [
                gmem_scratch_ptr,
                host_scratch_ptr,
                c(gmem_scratch_bytes, i64),
                token_ptr,
            ],
        )
    main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
  module.operation.verify()

  expected_arg_treedef = jax.tree.structure(in_shape)
  def _check_args(args):
    arg_treedef = jax.tree.structure(args)
    if arg_treedef != expected_arg_treedef:
      raise ValueError(
          f"Invalid argument structure: expected {expected_arg_treedef}, got"
          f" {arg_treedef}"
      )

  dump_low_level(module)

  pass_manager = _get_mosaic_gpu_pipeline("fatbin")
  if mosaic_gpu_print_after_all.value:
    pass_manager.enable_ir_printing()
  pass_manager.run(module.operation)

  def bind(*args):
    return mosaic_gpu_p.bind(
        *args,
        out_types=out_shape,
        module=module,
        gmem_scratch_bytes=gmem_scratch_bytes,
    )

  if prof_spec is not None:
    @jax.jit
    def prof_kernel(*args):
      _check_args(args)
      *results, prof_buffer = bind(*args)
      def dump_profile(prof_buffer):
        out_file = os.path.join(
            os.getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
            f"{time.time_ns()}-trace.json",
        )
        try:
          with open(out_file, "x") as f:
            prof_spec.dump(prof_buffer, f)
        except FileExistsError:
          pass  # TODO: Retry
      jax.debug.callback(dump_profile, prof_buffer)
      return results[0] if unwrap_output_tuple else results
    return prof_kernel
  else:
    @jax.jit
    def kernel(*args):
      _check_args(args)
      results = bind(*args)
      return results[0] if unwrap_output_tuple else results
    return kernel


def _declare_runtime_functions():
  """Declares the runtime functions that can be used by the generated code."""
  ptr_ty = ir.Type.parse("!llvm.ptr")
  i64 = ir.IntegerType.get_signless(64)
  arg_tys = [ptr_ty, ptr_ty, i64, i64, ptr_ty, ptr_ty, i64, ptr_ty]
  init_tma_desc_type = ir.FunctionType.get(arg_tys, [])
  func.FuncOp(
      "mosaic_gpu_init_tma_desc", init_tma_desc_type, visibility="private"
  )
  memcpy_async_type = ir.FunctionType.get([ptr_ty, ptr_ty, i64, ptr_ty], [])
  func.FuncOp(
      "mosaic_gpu_memcpy_async_h2d", memcpy_async_type, visibility="private"
  )


def dump_low_level(module):
  dump_ptx = mosaic_gpu_dump_ptx.value
  dump_ptxas = mosaic_gpu_dump_ptxas.value
  dump_sass = mosaic_gpu_dump_sass.value
  if not any([dump_ptx, dump_ptxas, dump_sass]):
    return
  module = ir.Module.parse(
      module.operation.get_asm(binary=True, enable_debug_info=True)
  )
  pm = _get_mosaic_gpu_pipeline("isa")
  pm.run(module.operation)

  for op in module.body:
    if op.OPERATION_NAME == "gpu.binary":
      objects = ir.ArrayAttr(op.objects)
      if len(objects) != 1:
        raise NotImplementedError("Expected a single object")
      obj = str(objects[0])
      start = obj.find('assembly = "') + len('assembly = "')
      end = obj.find('"', start)
      ptx = obj[start:end]
      ptx = ptx.replace("\\09", "\t").replace("\\0A", "\n")[:-3]
      if dump_ptx:
        print(ptx)
      if dump_ptxas or dump_sass:
        with tempfile.TemporaryDirectory() as tmp:
          ptx_path = os.path.join(tmp, "kernel.ptx")
          with open(ptx_path, "w") as f:
            f.write(ptx)
          elf_path = os.path.join(tmp, 'kernel.o')
          v_flag = "-v" if dump_ptxas else ""
          ptxas_flags = f"{v_flag} --opt-level 3 --gpu-name sm_90a"
          ptxas_out = subprocess.check_output(
              f"{PTXAS_PATH} {ptxas_flags} --output-file {elf_path} {ptx_path}",
              stderr=subprocess.STDOUT,
              shell=True,
          )
          if dump_ptxas:
            print(ptxas_out.decode())
          if dump_sass:
            sass = subprocess.check_output(
                f"{NVDISASM_PATH} -ndf -c {elf_path}",
                stderr=subprocess.STDOUT,
                shell=True,
            )
            print(sass.decode())


def _get_mosaic_gpu_pipeline(kernel_format) -> PassManager:
  passes = [
      "convert-nvgpu-to-nvvm",
      "gpu-kernel-outlining{data-layout-str=}",
      "convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1}",
      "convert-scf-to-cf",
      "convert-nvvm-to-llvm",
      "expand-strided-metadata",
      "nvvm-attach-target{O=3 chip=sm_90a fast=false features=+ptx80 ftz=false  module= triple=nvptx64-nvidia-cuda}",
      "lower-affine",
      "convert-arith-to-llvm{index-bitwidth=0}",
      "convert-index-to-llvm{index-bitwidth=64}",
      "canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}",
      "cse",
      "gpu.module(strip-debuginfo)",
      "gpu.module(convert-gpu-to-nvvm{has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false})",
      "gpu.module(canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true})",
      "gpu.module(cse)",
      "gpu.module(reconcile-unrealized-casts)",
      "gpu-to-llvm{gpu-binary-annotation=gpu.binary use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false}",
      "gpu-module-to-binary{format=" + kernel_format + "}",
      "convert-math-to-llvm{approximate-log1p=true}",
      "canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}",
      "cse",
      "reconcile-unrealized-casts",
      *(["gpu-launch-lowering"] if kernel_format in {"bin", "fatbin"} else []),
      "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
  ]
  return PassManager.parse(f"builtin.module({','.join(passes)})")
