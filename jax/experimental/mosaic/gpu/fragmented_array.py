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

import dataclasses

import jax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import math as mlir_math
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import vector
import numpy as np

from . import utils
from . import dsl as mgpu

# mypy: ignore-errors

WARPGROUP_SIZE = utils.WARPGROUP_SIZE
c = utils.c


@dataclasses.dataclass(frozen=True)
class WGMMAFragLayout:
  """[m, n] matrix, where m % 64 == 0 == n % 8."""


@dataclasses.dataclass(frozen=True)
class WGMMARowFragLayout:
  """[m] matrix, where m % 64 == 0."""


@dataclasses.dataclass(frozen=True)
class WGStridedFragLayout:
  """Convert the array to 1D and then shard across threads."""

  shape: tuple[int, ...]
  vec_size: int

  def __post_init__(self):
    if np.prod(self.shape) % (self.vec_size * WARPGROUP_SIZE) != 0:
      raise ValueError((self, WARPGROUP_SIZE))

  @classmethod
  def from_memref_type(cls, memref_ty: ir.Type):
    if not ir.MemRefType.isinstance(memref_ty):
      raise TypeError(memref_ty)

    memref_type = ir.MemRefType(memref_ty)
    bw = mgpu.bytewidth(memref_type.element_type)
    assert 8 % bw == 0 and 8 // bw != 0, bw
    if np.prod(memref_type.shape) % WARPGROUP_SIZE != 0:
      raise ValueError(
          "Ref must have a number of elements that is a multiple of"
          f" {WARPGROUP_SIZE}"
      )
    max_vec_size = np.prod(memref_type.shape) // WARPGROUP_SIZE
    return cls(
        shape=tuple(memref_type.shape), vec_size=min(8 // bw, max_vec_size)
    )

  def thread_vec_idxs(self):
    """The indexes to be used for vector load/store WGStridedFragLayout.

    Yields:
      The indices of the vector that correspond to the current thread.
    """
    index = ir.IndexType.get()
    cardinality = np.prod(self.shape)
    assert cardinality % (WARPGROUP_SIZE * self.vec_size) == 0
    reg_num = cardinality // (WARPGROUP_SIZE * self.vec_size)
    tidx = arith.remui(gpu.thread_id(gpu.Dimension.x), c(WARPGROUP_SIZE, index))
    off = arith.muli(tidx, c(self.vec_size, tidx.type))
    for i in range(reg_num):
      yield [arith.addi(off, c(i * WARPGROUP_SIZE * self.vec_size, tidx.type))]


FragmentedLayout = WGStridedFragLayout | WGMMAFragLayout | WGMMARowFragLayout


WGMMA_LAYOUT = WGMMAFragLayout()
WGMMA_ROW_LAYOUT = WGMMARowFragLayout()


@jax.tree_util.register_pytree_node_class
class FragmentedArray:
  registers: np.ndarray  # of ir.Value, see checks in init for shapes.
  layout: FragmentedLayout

  def __init__(self, *, _registers: np.ndarray, _layout: FragmentedLayout):
    self.registers = _registers
    self.layout = _layout

    match self.layout:
      # Registers are [m_tiles, n_tiles, 2 rows, 1 cols] in WGMMA layout
      # Each element is a vector<2xdtype>
      case WGMMAFragLayout():
        if self.registers.ndim != 4 or self.registers.shape[2:] != (2, 1):
          raise ValueError("Invalid register array shape")

      # Registers are [m_tiles, 2 rows] in WGMMA_ROW layout
      # Each element is a dtype scalar
      case WGMMARowFragLayout():
        if self.registers.ndim != 2 or self.registers.shape[-1] != 2:
          raise ValueError("Invalid register array shape")

      # Registers are flat
      case WGStridedFragLayout(shape):
        (reg_size,) = ir.VectorType(_registers.flat[0].type).shape
        if np.prod(shape) != np.prod(_registers.shape) * WARPGROUP_SIZE * reg_size:
          raise ValueError((reg_size, shape, _registers.shape, WARPGROUP_SIZE), _registers.flat[0].type)
      case _:
        raise NotImplementedError

  @classmethod
  def load_strided(cls, ref: ir.Value):
    if not ir.MemRefType.isinstance(ref.type):
      raise TypeError(ref.type)

    ref_ty = ir.MemRefType(ref.type)
    ref_1d = mgpu.memref_fold(ref, 0, len(ref_ty.shape))
    layout = WGStridedFragLayout.from_memref_type(ref_ty)
    vec_ty = ir.VectorType.get((layout.vec_size,), ref_ty.element_type)
    vecs = [vector.load(vec_ty, ref_1d, vec_idx) for vec_idx in layout.thread_vec_idxs()]
    return cls(_registers=np.array(vecs), _layout=layout)

  @classmethod
  def splat(cls, value, shape, layout):
    match layout:
      case WGMMARowFragLayout():
        if len(shape) != 1:
          raise ValueError
        if shape[0] % 64:
          raise ValueError
        reg_shape = (shape[0] // 64, 2)
      case WGMMAFragLayout():
        if len(shape) != 2:
          raise ValueError
        if shape[0] % 64 or shape[1] % 8:
          raise ValueError
        reg_shape = (shape[0] // 64, shape[1] // 8, 2, 1)
        value = vector.splat(ir.VectorType.get((2,), value.type), value)
      case WGStridedFragLayout(shape=shape, vec_size=vec_size):
        elems = np.prod(shape)
        reg_shape = (elems // (WARPGROUP_SIZE * vec_size),)
        value = vector.splat(ir.VectorType.get((vec_size,), value.type), value)
      case _:
        raise NotImplementedError(layout)

    return cls(
        _registers=np.full(reg_shape, value, dtype=object),
        _layout=layout,
    )

  @property
  def shape(self):
    row_tiles = self.registers.shape[0]
    match self.layout:
      case WGMMAFragLayout():
        col_tiles = self.registers.shape[1]
        return (row_tiles * 64, col_tiles * 8)
      case WGMMARowFragLayout():
        return (row_tiles * 64,)
      case WGStridedFragLayout(shape):
        return shape

  @property
  def mlir_dtype(self):
    reg_ty = self.registers.flat[0].type
    match self.layout:
      case WGMMAFragLayout() | WGStridedFragLayout():
        return ir.VectorType(reg_ty).element_type
      case WGMMARowFragLayout():
        return reg_ty

  def _pointwise(self, op, *other):
    for o in other:
      if not isinstance(o, FragmentedArray):
        return NotImplemented
      if self.layout != o.layout:
        raise ValueError("Incompatible FragmentedArray layouts")
      if self.registers.shape != o.registers.shape:
        raise ValueError("Incompatible FragmentedArray shapes")
    new_regs = np.empty_like(self.registers)
    for idx, reg in np.ndenumerate(self.registers):
      new_regs[idx] = op(reg, *(o.registers[idx] for o in other))
    return FragmentedArray(_registers=new_regs, _layout=self.layout)

  def __add__(self, other):
    if ir.FloatType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.addf, other)
    elif ir.IntegerType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.addi, other)

  def __mul__(self, other):
    if ir.FloatType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.mulf, other)
    elif ir.IntegerType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.muli, other)

  def __sub__(self, other):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    return self._pointwise(arith.subf, other)

  def __truediv__(self, other):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    return self._pointwise(arith.divf, other)

  def max(self, other):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    return self._pointwise(arith.maximumf, other)

  def exp(self, approx: bool = False):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    def fast_exp(x):
      f32 = ir.F32Type.get()
      if self.mlir_dtype != f32:
        raise NotImplementedError
      log2e = arith.constant(f32, ir.FloatAttr.get(f32, 1.4426950408889634))
      if x.type == f32:
        scaled = arith.mulf(x, log2e)
        return llvm.inline_asm(
            f32, [scaled], "ex2.approx.f32 $0,$1;", "=f,f", asm_dialect=0
        )
      elif ir.VectorType.isinstance(x.type):
        index = ir.IndexType.get()
        result = llvm.mlir_undef(x.type)
        for i in range(2):
          v = vector.extractelement(x, position=c(i, index))
          vr = fast_exp(v)
          result = vector.insertelement(vr, result, position=c(i, index))
        return result
      else:
        raise NotImplementedError(x.type)
    return self._pointwise(fast_exp if approx else mlir_math.exp)

  def __and__(self, other):
    if not ir.IntegerType.isinstance(self.mlir_dtype):
      raise ValueError(
          "Bitwise operations only defined for integer types, not"
          f" {self.mlir_dtype}"
      )

    return self._pointwise(arith.andi, other)

  def bitcast(self, elt: ir.Type):
    reg_type = self.registers.flat[0].type
    if ir.VectorType.isinstance(reg_type):
      reg_shape = ir.VectorType(reg_type).shape
      ty = ir.VectorType.get(reg_shape, elt)
    else:
      ty = elt

    return self._pointwise(lambda x: arith.bitcast(ty, x))

  def __getitem__(self, idx):
    if self.layout != WGMMA_LAYOUT:
      raise NotImplementedError("Only WGMMA layouts support slicing")
    base_idx, slice_shape, is_squeezed = utils.parse_indices(idx, self.shape)
    if any(is_squeezed):
      raise NotImplementedError("Only slicing implemented")
    if (
        base_idx[0] % 64
        or slice_shape[0] % 64
        or base_idx[1] % 8
        or slice_shape[1] % 8
    ):
      raise NotImplementedError("Only tile aligned slicing supported")
    base_idx[0] //= 64
    slice_shape[0] //= 64
    base_idx[1] //= 8
    slice_shape[1] //= 8
    new_regs = self.registers[
        base_idx[0] : base_idx[0] + slice_shape[0],
        base_idx[1] : base_idx[1] + slice_shape[1],
    ]
    return FragmentedArray(_registers=new_regs, _layout=self.layout)

  # TODO(apaszke): Support JAX dtypes here as well?
  def astype(self, new_dtype: ir.Type):
    cur_dtype = self.mlir_dtype
    if cur_dtype == new_dtype:
      return self
    from_float = ir.FloatType.isinstance(cur_dtype)
    to_float = ir.FloatType.isinstance(new_dtype)
    from_integer = ir.IntegerType.isinstance(cur_dtype)
    to_integer = ir.IntegerType.isinstance(new_dtype)
    if from_float and to_float:
      if ir.FloatType(cur_dtype).width > ir.FloatType(new_dtype).width:
        convert = arith.truncf
      else:
        convert = arith.extf
    elif from_integer and to_integer:
      if ir.IntegerType(cur_dtype).width > ir.IntegerType(new_dtype).width:
        convert = arith.trunci
      else:
        convert = arith.extsi
    elif from_integer and to_float:
      convert = arith.sitofp
    elif from_float and to_integer:
      convert = arith.fptosi
    new_registers = np.empty_like(self.registers)
    match self.layout:
      case WGMMAFragLayout():
        new_reg_ty = ir.VectorType.get((2,), new_dtype)
      case WGStridedFragLayout(vec_size=vec_size):
        new_reg_ty = ir.VectorType.get((vec_size,), new_dtype)
      case WGMMARowFragLayout():
        new_reg_ty = new_dtype
      case _:
        raise NotImplementedError(f"Unsupported layout {self.layout}")
    for idx, reg in np.ndenumerate(self.registers):
      new_registers[idx] = convert(new_reg_ty, reg)
    return FragmentedArray(_registers=new_registers, _layout=self.layout)

  def reduce(self, op, axis):
    if self.layout != WGMMA_LAYOUT:
      raise NotImplementedError(self.layout)
    if axis != 1:
      raise NotImplementedError
    index = ir.IndexType.get()
    i32 = ir.IntegerType.get_signless(32)
    new_regs = np.empty(self.registers.shape[::2], dtype=object)
    assert self.registers.shape[-1] == 1
    for row_tile, row_subtile in np.ndindex(new_regs.shape):
      # Reduce the registers owned by the current thread over n tiles
      thread_result_vec = self.registers[row_tile, 0, row_subtile, 0]
      for n_tile in range(1, self.registers.shape[1]):
        thread_result_vec = op(
            thread_result_vec, self.registers[row_tile, n_tile, row_subtile, 0]
        )
      thread_result = op(
          vector.extractelement(thread_result_vec, position=c(0, index)),
          vector.extractelement(thread_result_vec, position=c(1, index)),
      )
      # Do a shuffle to reduce in groups of 4 consecutive threads.
      result = thread_result
      for i in (1, 2):
        other_result = nvvm.shfl_sync(
            result.type,
            c(0xFFFFFFFF, i32),
            result,
            c(i, i32),
            c(0x1F, i32),
            nvvm.ShflKind.bfly,
        )
        result = op(result, other_result)
      new_regs[row_tile, row_subtile] = result
    return FragmentedArray(_registers=new_regs, _layout=WGMMA_ROW_LAYOUT)

  def broadcast_minor(self, n):
    if self.layout != WGMMA_ROW_LAYOUT:
      raise NotImplementedError
    num_row_tiles = self.registers.shape[0]
    num_col_tiles, rem = divmod(n, 8)
    if rem:
      raise ValueError("Number of columns must be divisible by 8")
    new_regs = np.empty((num_row_tiles, num_col_tiles, 2, 1), dtype=object)
    dtype = self.mlir_dtype
    for (row_tile, row_subtile), reg in np.ndenumerate(self.registers):
      new_regs[row_tile, :, row_subtile, :] = vector.splat(
          ir.VectorType.get((2,), dtype), reg
      )
    return FragmentedArray(_registers=new_regs, _layout=WGMMA_LAYOUT)

  def store_untiled(self, ref: ir.Value):
    if not ir.MemRefType.isinstance(ref.type):
      raise ValueError(ref)

    match self.layout:
      case WGMMAFragLayout():
        self._store_untiled_wgmma(ref)
      case WGStridedFragLayout():
        self._store_untiled_wg_strided(ref)
      case _:
        raise NotImplementedError(self.layout)

  def _store_untiled_wg_strided(self, ref: ir.Value):
    ref_ty = ir.MemRefType(ref.type)
    ref_shape = tuple(ref_ty.shape)
    if ref_shape != self.shape:
      raise ValueError((ref_shape, self.shape))
    smem_1d = mgpu.memref_fold(ref, 0, len(ref_ty.shape))
    for idx, reg in zip(self.layout.thread_vec_idxs(), self.registers.flat):
      vector.store(reg, smem_1d, idx)

  def _store_untiled_wgmma(self, ref: ir.Value):
    """Stores accumulator to a 2D memref. Not optimized at the moment."""
    assert self.layout == WGMMA_LAYOUT
    index = ir.IndexType.get()
    m, n = self.shape
    ref_ty = ir.MemRefType(ref.type)
    if ref_ty.shape != [m, n]:
      raise ValueError(ref.type, (m, n))

    def c(x):
      return arith.ConstantOp(index, ir.IntegerAttr.get(index, x))

    tidx = arith.remui(gpu.thread_id(gpu.Dimension.x), c(WARPGROUP_SIZE))
    lane_id = arith.remui(tidx, c(32))  # {0, 1, ..., 31}
    warp_id = arith.divui(tidx, c(32))  # {0, 1, 2, 3}
    row_base = arith.addi(
        arith.divui(lane_id, c(4)), arith.muli(warp_id, c(16))
    )
    col_base = arith.muli(arith.remui(lane_id, c(4)), c(2))  # {0, 2, 4, 6}
    it = np.ndenumerate(self.registers)
    for (row_tile, col_tile, row_idx, col_zero), elem in it:
      del col_zero
      row = arith.addi(row_base, c(row_tile * 64 + row_idx * 8))
      for col_idx in range(2):
        value = vector.extractelement(elem, position=c(col_idx))
        col = arith.addi(col_base, c(col_tile * 8 + col_idx))
        memref.store(value, ref, [row, col])

  def store_tiled(self, ref, swizzle: int | None):
    if self.layout != WGMMA_LAYOUT:
      raise NotImplementedError
    dtype = self.mlir_dtype
    bw = mgpu.bytewidth(dtype)
    m, n = self.shape
    assert m % 64 == 0  # This is implied by the layout.
    cols_per_tile = 128 // bw
    expected_shape = [m // 64, n // cols_per_tile, 64, cols_per_tile]
    if ir.MemRefType(ref.type).shape != expected_shape:
      raise ValueError(ref.type, (m, n))
    for get, _, idxs in self.transfer_tiled(self.shape, dtype, swizzle):
      vector.store(get(self.registers), ref, idxs)

  @classmethod
  def load_tiled(cls, ref, swizzle: int | None):
    ref_ty = ir.MemRefType(ref.type)
    dtype = ref_ty.element_type
    bw = mgpu.bytewidth(dtype)
    m_tiles, n_tiles, m_tile_size, n_tile_size = ref_ty.shape
    if m_tile_size != 64 or n_tile_size != (128 // bw):
      raise ValueError
    m, n = m_tiles * m_tile_size, n_tiles * n_tile_size
    assert m % 64 == 0  # This is implied by the layout.
    registers = np.full(
        (m_tiles, n // 8, 2, 1),
        vector.splat(ir.VectorType.get((2,), dtype), c(0, dtype)),
        dtype=object,
    )
    for _, update, idxs in cls.transfer_tiled((m, n), dtype, swizzle):
      update(registers, vector.load(ir.VectorType.get((2,), dtype), ref, idxs))
    return cls(_registers=registers, _layout=WGMMA_LAYOUT)

  @staticmethod
  def transfer_tiled(shape, dtype, swizzle: int | None):
    bw = mgpu.bytewidth(dtype)
    cols_per_tile = 128 // bw
    m, n = shape
    if n % 32 != 0:
      raise NotImplementedError
    cols_per_tile = 128 // bw
    if swizzle != 128:
      raise NotImplementedError("Only 128B swizzle supported")
    index = ir.IndexType.get()

    def c(x):
      return arith.ConstantOp(index, ir.IntegerAttr.get(index, x))

    tidx = arith.remui(gpu.thread_id(gpu.Dimension.x), c(WARPGROUP_SIZE))
    lane_id = arith.remui(tidx, c(32))  # {0, 1, ..., 31}
    warp_id = arith.divui(tidx, c(32))  # {0, 1, 2, 3}
    sub_row_base = arith.divui(lane_id, c(4))  # {0, 1, ..., 7}
    if bw > 2:  # Stagger is only necessary for values larger than 16bit.
      is_even_row = arith.cmpi(
          arith.CmpIPredicate.eq, arith.remui(sub_row_base, c(2)), c(0)
      )
    else:
      # We rely on canonicalization to clean up the selects.
      i1 = ir.IntegerType.get_signless(1)
      is_even_row = arith.constant(i1, ir.IntegerAttr.get(i1, 1))
    row_base = arith.addi(sub_row_base, arith.muli(warp_id, c(16)))
    col_base = arith.muli(arith.remui(lane_id, c(4)), c(2))  # {0, 2, 4, 6}
    # The swizzle pattern is constant for a given thread.
    col_swizzle_bits = arith.muli(sub_row_base, c(16 // bw))
    for row_group in range(m // 64):
      for col_group in range(n // cols_per_tile):
        for row_subidx in range(2):
          row = arith.addi(row_base, c(row_subidx * 8))
          for col_subidx in range(cols_per_tile // 8):
            # We stagger the even and odd rows a little to avoid bank conflicts.
            # It seems that the STS.64 is 2x faster (and the hardware reports no
            # conflicts) when the conflicts are split between half-warps, as
            # opposed to having them within the half-warp. This requires a
            # little more work for the selects, but is ultimately worth it.
            col_subidx_even = col_subidx
            col_subidx_odd = col_subidx ^ 2
            col_off = arith.select(
                is_even_row, c(col_subidx_even * 8), c(col_subidx_odd * 8)
            )
            col = arith.addi(col_base, col_off)
            col = arith.xori(col, col_swizzle_bits)
            reg_idx_even = col_subidx_even + col_group * (cols_per_tile // 8)
            reg_idx_odd = col_subidx_odd + col_group * (cols_per_tile // 8)
            even_idx = row_group, reg_idx_even, row_subidx, 0
            odd_idx = row_group, reg_idx_odd, row_subidx, 0
            idx = c(row_group), c(col_group), row, col
            def get_register(regs, even_idx=even_idx, odd_idx=odd_idx):
              value_even = regs[even_idx]
              value_odd = regs[odd_idx]
              return arith.select(is_even_row, value_even, value_odd)
            def update_registers(regs, new, even_idx=even_idx, odd_idx=odd_idx):
              regs[even_idx] = arith.select(is_even_row, new, regs[even_idx])
              regs[odd_idx] = arith.select(is_even_row, regs[odd_idx], new)
            yield get_register, update_registers, idx

  def tree_flatten(self):
    return list(self.registers.flat), (self.layout, self.registers.shape)

  @classmethod
  def tree_unflatten(cls, aux, flat_registers):
    layout, reg_shape = aux
    registers = np.asarray(flat_registers, dtype=object).reshape(reg_shape)
    return cls(_registers=registers, _layout=layout)
