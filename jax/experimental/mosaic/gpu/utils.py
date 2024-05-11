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
"""Utilities for code generator."""

import functools
import contextlib
import dataclasses
from typing import Any, Literal, Sequence

from absl import flags
import jax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvgpu
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector
import numpy as np

# mypy: ignore-errors

WARPGROUP_SIZE: int = 128
DYNAMIC = -9223372036854775808

FLAGS = flags.FLAGS

flags.DEFINE_bool("mosaic_gpu_debug", False, "Perform debug printing")

# pylint: disable=line-too-long, wildcard-import, missing-function-docstring, bad-continuation, g-bad-todo, protected-access, g-explicit-length-test, missing-class-docstring, g-doc-return-or-yield, g-inconsistent-quotes


def ptr_as_memref(ptr, memref_ty: ir.MemRefType):
  if len(memref_ty.shape) == 0:
    raise NotImplementedError
  i64 = ir.IntegerType.get_signless(64)
  rank = len(memref_ty.shape)
  desc_ty = ir.Type.parse(
      f"!llvm.struct<(ptr, ptr, i64, array<{rank} x i64>, array<{rank} x i64>)>"
  )
  desc = llvm.UndefOp(desc_ty)
  desc = llvm.InsertValueOp(desc, ptr, [0])  # Allocation
  desc = llvm.InsertValueOp(desc, ptr, [1])  # Aligned Base
  desc = llvm.InsertValueOp(
      desc, llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 0)), [2]
  )
  for i, s in enumerate(memref_ty.shape):
    desc = llvm.InsertValueOp(
        desc, llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, s)), [3, i]
    )
  for i, s in enumerate(get_contiguous_strides(memref_ty.shape)):
    desc = llvm.InsertValueOp(
        desc, llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, s)), [4, i]
    )
  return builtin.unrealized_conversion_cast([memref_ty], [desc])


def pack_array(values):
  if not values:
    raise ValueError("Empty array")
  elem_ty = values[0].type
  i64 = ir.IntegerType.get_signless(64)
  ptr_ty = ir.Type.parse("!llvm.ptr")
  arr_ptr = llvm.alloca(ptr_ty, c(len(values), i64), elem_ty)
  for i, v in enumerate(values):
    elem_ptr = llvm.getelementptr(ptr_ty, arr_ptr, [], [i], elem_ty)
    llvm.store(v, elem_ptr)
  return arr_ptr


def get_contiguous_strides(xs):
  strides_ret = []
  stride = 1
  for x in xs[::-1]:
    strides_ret.append(stride)
    stride *= x
  return strides_ret[::-1]


def c(val: int | float, ty):
  if ir.IntegerType.isinstance(ty) or ir.IndexType.isinstance(ty):
    if not isinstance(val, (int, np.integer)):
      raise TypeError(type(val))
    attr = ir.IntegerAttr.get(ty, val)
  elif ir.FloatType.isinstance(ty):
    attr = ir.FloatAttr.get(ty, val)
  elif ir.VectorType.isinstance(ty):
    return vector.splat(ty, c(val, ir.VectorType(ty).element_type))
  else:
    raise NotImplementedError(ty)
  return arith.constant(ty, attr)


def get_tensormap_descriptor(**attrs):
  return ir.Type.parse(
      f"!nvgpu.tensormap.descriptor<{', '.join(k + '=' + v for k, v in attrs.items())}>"
  )


def debug_print(fmt, *args, uniform=True):
  if not FLAGS.mosaic_gpu_debug:
    return
  type_formats = []
  new_args = []
  for arg in args:
    ty_format = None
    if ir.IndexType.isinstance(arg.type):
      ty_format = "%llu"
    if ir.IntegerType.isinstance(arg.type):
      width = ir.IntegerType(arg.type).width
      ty_format = "%llu"
      if width < 64:
        arg = arith.extui(ir.IntegerType.get_signless(64), arg)
    if ir.F32Type.isinstance(arg.type):
      ty_format = "%f"
    if ir.F16Type.isinstance(arg.type):
      ty_format = "%f"
      arg = arith.extf(ir.F32Type.get(), arg)
    if ty_format is None:
      raise NotImplementedError(arg.type)
    type_formats.append(ty_format)
    new_args.append(arg)
  ctx = functools.partial(once, per_block=False) if uniform else contextlib.nullcontext
  with ctx():
    gpu.printf(fmt.format(*type_formats) + "\n", new_args)


@dataclasses.dataclass(frozen=True)
class ForResult:
  op: scf.ForOp
  results: tuple[Any, ...]

  @property
  def result(self):
    if len(self.results) != 1:
      raise ValueError
    return self.results[0]


def fori(bound, carrys):
  unwrap = False
  if not isinstance(carrys, (list, tuple)):
    carrys = [carrys]
    unwrap = True
  flat_carrys, carry_treedef = jax.tree.flatten(carrys)

  def wrapper(f):
    index = ir.IndexType.get()
    c0 = arith.ConstantOp(index, ir.IntegerAttr.get(index, 0))
    c1 = arith.ConstantOp(index, ir.IntegerAttr.get(index, 1))
    for_op = scf.ForOp(c0, bound, c1, flat_carrys)
    with ir.InsertionPoint(for_op.body):
      i = for_op.induction_variable
      inner_carrys = jax.tree.unflatten(carry_treedef, for_op.inner_iter_args)
      if unwrap:
        [inner_carrys] = inner_carrys
      new_carrys = f(i, inner_carrys)
      if unwrap:
        new_carrys = [new_carrys]
      new_flat_carrys, new_carry_treedef = jax.tree.flatten(new_carrys)
      if new_carry_treedef != carry_treedef:
        raise ValueError(new_carry_treedef, carry_treedef)
      scf.YieldOp(new_flat_carrys)
    final_flat_carrys = for_op.results
    return ForResult(
        for_op, jax.tree.unflatten(carry_treedef, final_flat_carrys)
    )

  return wrapper


def thread_idx():
  i32 = ir.IntegerType.get_signless(32)
  as_i32 = lambda x: arith.index_cast(i32, x)
  tidx = as_i32(gpu.thread_id(gpu.Dimension.x))
  stride = as_i32(gpu.block_dim(gpu.Dimension.x))
  for dim in (gpu.Dimension.y, gpu.Dimension.z):
    tidx = arith.addi(tidx, arith.muli(as_i32(gpu.thread_id(dim)), stride))
    stride = arith.muli(stride, as_i32(gpu.block_dim(dim)))
  return tidx

def warp_idx(sync=True):
  i32 = ir.IntegerType.get_signless(32)
  warp_idx = arith.shrui(thread_idx(), c(5, i32))
  if not sync:
    return warp_idx
  mask = c(0xFFFFFFFF, i32)
  return nvvm.shfl_sync(
      warp_idx.type, mask, warp_idx, c(0, i32), c(0x1F, i32), nvvm.ShflKind.idx
  )

def warpgroup_idx(sync=True):
  i32 = ir.IntegerType.get_signless(32)
  wg_idx = arith.shrui(thread_idx(), c(7, i32))
  if not sync:
    return wg_idx
  mask = c(0xFFFFFFFF, i32)
  return nvvm.shfl_sync(
      wg_idx.type, mask, wg_idx, c(0, i32), c(0x1F, i32), nvvm.ShflKind.idx
  )


# True withon `once()` contexts.
_ONCE_REGION_ACTIVE = False


@contextlib.contextmanager
def once(per_block=True):
  """Runs the context only from a single thread from the first warp.

  The block is assumed to have a size of 1 in both y and z dimensions.
  """
  global _ONCE_REGION_ACTIVE

  if _ONCE_REGION_ACTIVE:
    yield
    return

  warp = warp_idx()
  if not per_block:
    warp = arith.remui(warp, c(4, warp.type))
  first_warp = arith.cmpi(arith.CmpIPredicate.eq, warp, c(0, warp.type))
  elected = nvvm.elect_sync(ir.IntegerType.get_signless(1))
  should_run = arith.andi(first_warp, elected)
  if_op = scf.IfOp(should_run)
  _ONCE_REGION_ACTIVE = True
  try:
    with ir.InsertionPoint(if_op.then_block):
      yield
      scf.YieldOp([])
  finally:
    _ONCE_REGION_ACTIVE = False


def clock():
  i32 = ir.IntegerType.get_signless(32)
  return llvm.inline_asm(
      i32, [], "mov.u32  $0,%clock;", "=r", asm_dialect=0, has_side_effects=True
  )


def globaltimer(kind: Literal["low", "high"] | None = None):
  if kind is None:
    i64 = ir.IntegerType.get_signless(64)
    return llvm.inline_asm(
        i64, [], "mov.u32  $0,%globaltimer;",
        "=l", asm_dialect=0, has_side_effects=True,
    )
  i32 = ir.IntegerType.get_signless(32)
  return llvm.inline_asm(
      i32, [], f"mov.u32  $0,%globaltimer_{kind[:2]};",
      "=r", asm_dialect=0, has_side_effects=True,
  )


def bytewidth(ty: ir.Type):
  if ir.IntegerType.isinstance(ty):
    return ir.IntegerType(ty).width // 8
  if ir.FloatType.isinstance(ty):
    return ir.FloatType(ty).width // 8
  raise NotImplementedError(ty)


@dataclasses.dataclass(frozen=True)
class DynamicSlice:
  base: ir.Value | int
  length: int


ds = DynamicSlice


def memref_slice(ref: ir.Value, index) -> ir.Value:
  ref_ty = ir.MemRefType(ref.type)
  base_indices, slice_shape, is_squeezed = parse_indices(index, ref_ty.shape)

  memref_strides, offset = ref_ty.get_strides_and_offset()
  new_offset = offset
  for idx, stride in zip(base_indices, memref_strides):
    if isinstance(idx, int):
      new_offset += idx * stride
    else:
      new_offset = ir.ShapedType.get_dynamic_stride_or_offset()
      break
  new_strides = [
      s for s, squeeze in zip(memref_strides, is_squeezed) if not squeeze
  ]
  new_shape = [s for s, squeeze in zip(slice_shape, is_squeezed) if not squeeze]
  new_layout = ir.StridedLayoutAttr.get(new_offset, new_strides)

  ref_slice = memref.subview(
      ref, base_indices, slice_shape, [1] * len(ref_ty.shape),
      result_type=ir.MemRefType.get(
          new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
      ),
  )
  return ref_slice


def _is_contiguous_shape_slice(
    ref_ty: ir.MemRefType, dim_slice: slice | None = slice(None)
):
  # If it's not a strided layout then we are definitely contiguous.
  if not ir.StridedLayoutAttr.isinstance(ref_ty.layout):
    return True

  strides = ir.StridedLayoutAttr(ref_ty.layout).strides[dim_slice]
  shape = ref_ty.shape[dim_slice]

  # Check that each dimension fits exactly it the immediately larger stride.
  ss = sorted(zip(strides, shape), key=lambda x: x[0], reverse=True)
  for (prev_stride, _), (stride, shape) in zip(ss, ss[1:]):
    if stride * shape != prev_stride:
      return False

  return True


def memref_fold(ref: ir.Value, dim, fold_rank) -> ir.Value:
  ref_ty = ir.MemRefType(ref.type)
  new_shape = list(ref_ty.shape)
  new_shape[dim : dim + fold_rank] = [np.prod(new_shape[dim : dim + fold_rank])]
  identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(ref_ty.rank))
  contig_strided_1d = ir.Attribute.parse("strided<[1]>")
  # Not sure why but MLIR expects the strided 1D layout to disappear in this op.
  if ref_ty.layout == identity or ref_ty.layout == contig_strided_1d:
    new_layout = ir.AffineMapAttr.get(
        ir.AffineMap.get_identity(ref_ty.rank - fold_rank + 1)
    )
  elif _is_contiguous_shape_slice(ref_ty, slice(dim, dim + fold_rank)):
    new_strides, offset = ref_ty.get_strides_and_offset()
    new_strides[dim : dim + fold_rank] = [new_strides[dim + fold_rank - 1]]
    new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
  else:
    raise NotImplementedError(
        f"strides={ref_ty.get_strides_and_offset()[0]}, {ref_ty.shape=},"
        f" {dim=}, {fold_rank=}"
    )

  new_ty = ir.MemRefType.get(
      new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
  )
  assoc = [[d] for d in range(dim)]
  assoc.append([dim + i for i in range(fold_rank)])
  assoc.extend([d] for d in range(dim + fold_rank, ref_ty.rank))
  assert len(assoc) == new_ty.rank
  return memref.collapse_shape(new_ty, ref, assoc)


def memref_unfold(ref: ir.Value, dim, factors) -> ir.Value:
  """Unfolds dim into two dimensions, the size of leading one given be major_factor."""
  ref_ty = ir.MemRefType(ref.type)
  new_shape = list(ref_ty.shape)
  if sum(f is None for f in factors) > 1:
    raise ValueError("Can only infer one dimension")
  known_factor_prod = np.prod([f for f in factors if f is not None])
  if new_shape[dim] % known_factor_prod:
    raise ValueError("Non-divisible unfold:", new_shape[dim], factors)
  factors = tuple(
      new_shape[dim] // known_factor_prod if f is None else f for f in factors
  )
  new_shape[dim : dim + 1] = factors
  identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(ref_ty.rank))
  if ref_ty.layout == identity:
    new_layout = ir.AffineMapAttr.get(
        ir.AffineMap.get_identity(ref_ty.rank + len(factors) - 1)
    )
  else:
    new_strides, offset = ref_ty.get_strides_and_offset()
    prev_stride = new_strides[dim]
    inserted_strides = []
    for f in reversed(factors):
      inserted_strides.append(prev_stride)
      prev_stride *= f
    new_strides[dim : dim + 1] = reversed(inserted_strides)
    new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
  new_ty = ir.MemRefType.get(
      new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
  )
  if dim == ref_ty.rank:
    assoc = [[d] for d in range(ref_ty.rank)]
    assoc[-1].extend(range(ref_ty.rank, ref_ty.rank + len(factors) - 1))
  else:
    assoc = [[d] for d in range(dim)]
    assoc.append(list(range(dim, dim + len(factors))))
    assoc.extend([d + len(factors) - 1] for d in range(dim + 1, ref_ty.rank))
  assert len(assoc) == ref_ty.rank
  return memref.expand_shape(new_ty, ref, assoc, [], new_ty.shape)


def memref_unsqueeze(ref: ir.Value, dim) -> ir.Value:
  """Inserts a singleton dimension."""
  ref_ty = ir.MemRefType(ref.type)
  if dim == ref_ty.rank:
    new_shape = list(ref_ty.shape)
    new_shape.append(1)
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(ref_ty.rank))
    if ref_ty.layout == identity:
      new_layout = ir.AffineMapAttr.get(
          ir.AffineMap.get_identity(ref_ty.rank + 1)
      )
    else:
      new_strides, offset = ref_ty.get_strides_and_offset()
      new_strides.append(1)
      new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
    new_ty = ir.MemRefType.get(
        new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
    )
    assoc = [[d] for d in range(ref_ty.rank)]
    assoc[-1].append(ref_ty.rank)
    return memref.expand_shape(new_ty, ref, assoc)
  else:
    return memref_unfold(ref, dim, (1, None))


def memref_transpose(ref: ir.Value, permutation: Sequence[int]) -> ir.Value:
  ref_ty = ir.MemRefType(ref.type)
  strides, offset = ref_ty.get_strides_and_offset()
  new_strides = [strides[p] for p in permutation]
  new_shape = [ref_ty.shape[p] for p in permutation]
  new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
  new_ty = ir.MemRefType.get(
      new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
  )
  return memref.transpose(
      new_ty, ref, ir.AffineMap.get_permutation(permutation)
  )


def _has_identity_layout(ref_ty: ir.Type):
  ref_ty = ir.MemRefType(ref_ty)
  strides, offset = ref_ty.get_strides_and_offset()
  return offset == 0 and strides == get_contiguous_strides(ref_ty.shape)


def parse_indices(
    index, shape: tuple[int, ...]
) -> tuple[list[ir.Value | int], list[int], list[bool]]:
  if not isinstance(index, tuple):
    index = (index,)
  if trailing_dims := len(shape) - len(index):
    index += (slice(None),) * trailing_dims
  base_indices = []
  slice_shape = []
  is_squeezed = []
  for idx, bound in zip(index, shape):
    if isinstance(idx, (ir.Operation, ir.OpView)):
      idx = idx.result
    if isinstance(idx, int):
      base_indices.append(idx)
      slice_shape.append(1)
      is_squeezed.append(True)
    elif isinstance(idx, slice):
      if idx.step is not None:
        raise NotImplementedError("Strided slices not implemented")
      base_indices.append(idx.start or 0)
      slice_shape.append((idx.stop or bound) - (idx.start or 0))
      is_squeezed.append(False)
    elif isinstance(idx, DynamicSlice):
      base_indices.append(idx.base)
      slice_shape.append(idx.length)
      is_squeezed.append(False)
    elif isinstance(idx, ir.Value):
      if not ir.IndexType.isinstance(idx.type):
        raise ValueError("Expected an index-typed index")
      base_indices.append(idx)
      slice_shape.append(1)
      is_squeezed.append(True)
    else:
      raise NotImplementedError(type(idx))
  assert len(base_indices) == len(slice_shape) == len(is_squeezed) == len(shape)
  return base_indices, slice_shape, is_squeezed


def commit_shared():
  gpu.barrier()
  nvvm.fence_proxy(
      nvvm.ProxyKind.async_shared, space=nvvm.SharedSpace.shared_cta
  )


class BarrierArray:

  def __init__(self, num_barriers: int, arrival_count: int = 1):
    barrier_group_ty = ir.Type.parse(
        "!nvgpu.mbarrier.group<memorySpace=#gpu.address_space<workgroup>,"
        f" num_barriers={num_barriers}>"
    )

    self.value = nvgpu.mbarrier_create(barrier_group_ty)
    self.num_barriers = num_barriers
    index = ir.IndexType.get()
    if num_barriers > 32:
      raise NotImplementedError("Only up to 32 barriers per group supported")
    i32 = ir.IntegerType.get_signless(32)
    self.phases = memref.alloca(ir.MemRefType.get((), i32), [], [])
    memref.store(c(0, i32), self.phases, [])
    with once(per_block=True):
      for i in range(num_barriers):
        nvgpu.mbarrier_init(self.value, c(arrival_count, index), c(i, index))

  def __getitem__(self, offset: ir.Value | int):
    if isinstance(offset, int):
      offset = c(offset, ir.IndexType.get())
    return Barrier(self, offset)


@dataclasses.dataclass(frozen=True)
class Barrier:
  barrier_array: BarrierArray
  offset: ir.Value

  def wait_parity(self, parity):
    index = ir.IndexType.get()
    nvgpu.mbarrier_try_wait_parity(
        self.barrier_array.value, parity, c(10000000, index), self.offset,
    )

  def wait(self):
    i32 = ir.IntegerType.get_signless(32)
    parities = memref.load(self.barrier_array.phases, [])
    offset_i32 = arith.index_castui(i32, self.offset)
    bitmask = arith.shli(c(1, i32), offset_i32)
    parity = arith.cmpi(
        arith.CmpIPredicate.ne, arith.andi(parities, bitmask), c(0, i32)
    )
    new_parities = arith.xori(parities, bitmask)
    memref.store(new_parities, self.barrier_array.phases, [])
    self.wait_parity(parity)

  def arrive(self):
    token_ty = ir.Type.parse("!nvgpu.mbarrier.token")
    nvgpu.mbarrier_arrive(token_ty, self.barrier_array.value, self.offset)


class Partition:
  source_bounds: tuple[int, ...]
  target_bounds: tuple[int, ...]
  partition: tuple[int | None, ...]
  base_offset: tuple[ir.Value, ...] | None

  def __init__(
      self,
      elements: tuple[int, ...],
      *,
      partition: tuple[int | None, ...],
      base_offset: tuple[ir.Value, ...] | None = None,
      num_chunks: tuple[int, ...] | None = None,
      chunk_size: tuple[int, ...] | None = None,
  ):
    self.target_bounds = elements
    self.partition = partition
    self.base_offset = base_offset
    if len(self.target_bounds) != len(self.partition):
      raise ValueError
    if num_chunks is None == chunk_size is None:
      raise ValueError(
          "Exactly one of num_chunks and chunk_size must be specified"
      )
    if num_chunks is not None:
      self.source_bounds = num_chunks
    else:
      if len(chunk_size) != len(self.target_bounds):
        raise ValueError
      source_bounds = []
      for els, chunk in zip(elements, chunk_size):
        if els % chunk:
          raise ValueError("Non-divisible partition", elements, chunk_size)
        source_bounds.append(els // chunk)
      self.source_bounds = tuple(source_bounds)

    seen_dims = set()
    for p in self.partition:
      if p is None:
        continue
      if not (0 <= p < len(self.source_bounds)):
        raise ValueError
      if p in seen_dims:
        raise ValueError
      seen_dims.add(p)
    for tb, p in zip(self.target_bounds, self.partition):
      if p is not None and tb % self.source_bounds[p]:
        raise ValueError("Non-divisible partitioning")

  @property
  def num_chunks(self) -> tuple[int, ...]:
    return self.source_bounds

  @property
  def target_block_shape(self):
    return tuple(tb if p is None else tb // self.source_bounds[p]
                 for tb, p in zip(self.target_bounds, self.partition))

  def get_base(self, *source_coords: ir.Value | int) -> list[ir.Value]:
    coords = []
    index = ir.IndexType.get()
    for i, (tbs, p) in enumerate(zip(self.target_block_shape, self.partition)):
      if p is None:
        dim_base = c(0, index)
      else:
        dim_base = arith.muli(c(tbs, index), source_coords[p])
      if self.base_offset is not None:
        dim_base = arith.addi(self.base_offset[i], dim_base)
      coords.append(dim_base)
    return coords


class Partition1D:
  partition: Partition

  def __init__(
      self,
      elements: int,
      *,
      base_offset: ir.Value | None = None,
      num_chunks: int | None = None,
      chunk_size: int | None = None,
  ):
    self.base_offset = base_offset
    if num_chunks is None == chunk_size is None:
      raise ValueError(
          "Exactly one of num_chunks and chunk_size must be specified"
      )
    common_kwargs = dict(elements=(elements,), partition=(0,))
    if base_offset is not None:
      common_kwargs["base_offset"] = (base_offset,)
    if num_chunks is not None:
      self.partition = Partition(num_chunks=(num_chunks,), **common_kwargs)
    else:
      self.partition = Partition(chunk_size=(chunk_size,), **common_kwargs)

  @property
  def num_chunks(self) -> int:
    return self.partition.source_bounds[0]

  def get_base(self, source_coords: ir.Value) -> ir.Value:
    return self.partition.get_base(source_coords)[0]

  def refine(
      self,
      *,
      chunk: ir.Value | None = None,
      num_chunks: int | None = None,
      chunk_size: int | None = None,
  ):
    return Partition1D(
        self.partition.target_block_shape[0],
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        base_offset=self.get_base(chunk) if chunk is not None else None,
    )


def tile_shape(shape, tiling):
  if len(tiling) > len(shape):
    raise ValueError
  if not tiling:
    return shape
  tiling_rank = len(tiling)
  for s, t in zip(shape[-tiling_rank:], tiling):
    if s % t:
      raise ValueError("Non-divisible tiling:", shape, tiling)
  return (
      *shape[:-tiling_rank],
      *(s // t for s, t in zip(shape[-tiling_rank:], tiling)),
      *tiling,
  )
