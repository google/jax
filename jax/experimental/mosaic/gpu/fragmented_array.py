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
import math
from typing import Callable

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

from . import dsl as mgpu
from . import utils

# mypy: ignore-errors

WARPGROUP_SIZE = utils.WARPGROUP_SIZE
c = utils.c


@dataclasses.dataclass(frozen=True)
class WGSplatFragLayout:
  """A fragmented array where all the values are equal represented as a register per thread.

  FragmentedArrays in this layout can be are always the result of a
  splat, each thread in the warpgroup has a single copy of the value,
  while the FragmentedArray pretends it has whatever shape the user
  wants. This means we can trivially broadcast, reshape and do
  elementwise operations with all other layouts.

  Examples:

  To load a value in
  ```
  FragmentedArray.splat(memref.load(ref_1d, [1]), (10,20,2))
  ```

  A shape is always provided for sanity check reasons.

  """

  shape: tuple[int, ...] = ()

  def can_broadcast_to(self, shape) -> bool:
    """Check that the shape can be broadcast.

    Only dimensions of size 1 can be broadcast. All other dimensions
    must be the same as the argument shape.
    """
    return all(dim1 == dim2 or dim1 == 1 for dim1, dim2 in zip(self.shape[::-1], shape[::-1]))


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
    if math.prod(memref_type.shape) % WARPGROUP_SIZE != 0:
      raise ValueError(
          "Ref must have a number of elements that is a multiple of"
          f" {WARPGROUP_SIZE} (got {math.prod(memref_type.shape)})"
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


FragmentedLayout = WGSplatFragLayout | WGStridedFragLayout | WGMMAFragLayout | WGMMARowFragLayout


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

      # Just a single register
      case WGSplatFragLayout():
        if _registers.size != 1:
          raise ValueError(f"WGStridedFragLayout requires a single value {_registers.shape} ({_registers.size})")

      case _:
        raise NotImplementedError

  def __repr__(self):
    return (
        f"FragmentedArray(layout={self.layout}, shape={self.shape})"
    )

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
  def splat(cls, value, shape, layout=None):
    layout = layout or WGSplatFragLayout(shape)
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
      case WGStridedFragLayout(vec_size=vec_size):
        assert shape == layout.shape
        elems = np.prod(shape)
        reg_shape = (elems // (WARPGROUP_SIZE * vec_size),)
        value = vector.splat(ir.VectorType.get((vec_size,), value.type), value)
      case WGSplatFragLayout():
        assert shape == layout.shape
        reg_shape = ()
      case _:
        raise NotImplementedError(layout)

    return cls(
        _registers=np.full(reg_shape, value, dtype=object),
        _layout=layout,
    )

  @property
  def shape(self):
    match self.layout:
      case WGMMAFragLayout():
        row_tiles, col_tiles = self.registers.shape[:2]
        return (row_tiles * 64, col_tiles * 8)
      case WGMMARowFragLayout():
        row_tiles = self.registers.shape[0]
        return (row_tiles * 64,)
      case WGStridedFragLayout(shape):
        return shape
      case WGSplatFragLayout(shape=shape):
        return shape

  @property
  def mlir_dtype(self):
    reg_ty = self.registers.flat[0].type
    match self.layout:
      case WGMMAFragLayout() | WGStridedFragLayout():
        return ir.VectorType(reg_ty).element_type
      case WGMMARowFragLayout() | WGSplatFragLayout():
        return reg_ty

  def _pointwise(self, op, *other):
    other_arrs = []
    for o in other:
      if not isinstance(o, FragmentedArray):
        if not isinstance(o, ir.Value):
          raise NotImplementedError(o)

        o = FragmentedArray.splat(o, shape=self.shape, layout=self.layout)

      if isinstance(o.layout, WGSplatFragLayout):
        if not o.layout.can_broadcast_to(self.shape):
          raise ValueError("Can't broadcast shape.")
        o = FragmentedArray.splat(o.registers.flat[0], shape=self.shape, layout=self.layout)
      else:
        if self.layout != o.layout:
          raise ValueError("Incompatible FragmentedArray layouts")
        if self.registers.shape != o.registers.shape:
          raise ValueError("Incompatible FragmentedArray shapes")

      other_arrs.append(o)
    new_regs = np.empty_like(self.registers)

    for idx, reg in np.ndenumerate(self.registers):
      new_regs[idx] = op(reg, *(o.registers[idx] for o in other_arrs))
    return FragmentedArray(_registers=new_regs, _layout=self.layout)

  def __neg__(self):
    if ir.FloatType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.negf)
    elif ir.IntegerType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.negsi)
    else:
      raise NotImplementedError(self.mlir_dtype)

  def __add__(self, other):
    if ir.FloatType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.addf, other)
    elif ir.IntegerType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.addi, other)
    else:
      raise NotImplementedError(self.mlir_dtype)

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    if ir.FloatType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.mulf, other)
    elif ir.IntegerType.isinstance(self.mlir_dtype):
      return self._pointwise(arith.muli, other)
    else:
      raise NotImplementedError(self.mlir_dtype)

  def __rmul__(self, other):
    return self * other

  def __sub__(self, other):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    return self._pointwise(arith.subf, other)

  def __rsub__(self, other):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    return self._pointwise(lambda s, o: arith.subf(o, s), other)

  def __truediv__(self, other):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    return self._pointwise(arith.divf, other)

  def __rtruediv__(self, other):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    return self._pointwise(lambda s, o: arith.divf(o, s), other)

  def max(self, other):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    return self._pointwise(arith.maximumf, other)

  def exp(self, approx: bool = False):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    if approx:
      f32 = ir.F32Type.get()
      if self.mlir_dtype != f32:
        raise NotImplementedError
      log2e = arith.constant(f32, ir.FloatAttr.get(f32, 1.4426950408889634))
      def fast_exp(x):
        scaled = arith.mulf(x, log2e)
        return llvm.inline_asm(f32, [scaled], "ex2.approx.f32 $0, $1;", "=f,f")
      return self._pointwise(self._lift_fast_unary(fast_exp))
    return self._pointwise(mlir_math.exp)

  def sin(self, approx: bool = False):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    if approx and self.mlir_dtype != ir.F32Type.get():
      raise NotImplementedError
    return self._pointwise(
        self._lift_fast_unary("sin.approx.f32") if approx else mlir_math.sin
    )

  def cos(self, approx: bool = False):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    if approx and self.mlir_dtype != ir.F32Type.get():
      raise NotImplementedError
    return self._pointwise(
        self._lift_fast_unary("cos.approx.f32") if approx else mlir_math.cos
    )

  def rsqrt(self, approx: bool = False):
    if not ir.FloatType.isinstance(self.mlir_dtype):
      raise NotImplementedError
    if approx and self.mlir_dtype != ir.F32Type.get():
      raise NotImplementedError
    return self._pointwise(
        self._lift_fast_unary("rsqrt.approx.f32") if approx else mlir_math.rsqrt
    )

  @staticmethod
  def _lift_fast_unary(
      instr: str | Callable[[ir.Value], ir.Value],
  ) -> Callable[[ir.Value], ir.Value]:
    def fast_instr(x):
      f32 = ir.F32Type.get()
      if x.type == f32:
        if isinstance(instr, str):
          return llvm.inline_asm(f32, [x], instr + " $0, $1;", "=f,f")
        else:
          return instr(x)
      elif ir.VectorType.isinstance(x.type):
        index = ir.IndexType.get()
        result = llvm.mlir_undef(x.type)
        for i in range(2):
          v = vector.extractelement(x, position=c(i, index))
          vr = fast_instr(v)
          result = vector.insertelement(vr, result, position=c(i, index))
        return result
      else:
        raise NotImplementedError(x.type)
    return fast_instr

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
    i8 = ir.IntegerType.get_signless(8)
    i16 = ir.IntegerType.get_signless(16)
    i32 = ir.IntegerType.get_signless(32)
    bf16 = ir.BF16Type.get()

    cur_dtype = self.mlir_dtype
    if cur_dtype == new_dtype:
      return self
    reg_type = self.registers.flat[0].type
    is_vector_reg = ir.VectorType.isinstance(reg_type)
    reg_shape = tuple(ir.VectorType(reg_type).shape) if is_vector_reg else ()
    if cur_dtype == i8 and new_dtype == bf16 and reg_shape == (2,):
      new_registers = np.empty_like(self.registers)
      for idx, reg in np.ndenumerate(self.registers):
        reg_16 = vector.bitcast(ir.VectorType.get((1,), i16), reg)
        val_16 = llvm.extractelement(reg_16, c(0, i32))
        # We first embed the s8 into a bf16 with the exponent equal to
        # bias + mantissa bits. Then, we zero the msb that didn't fit into the
        # mantissa, zero out all bits other than msb, and subtract the last
        # two values from each other. This takes advantage of the fact that the
        # lsb of the exponent (msb of the second byte) is zero, which allows us
        # to losslesly pack the msb there. When 1, it doubles the value of s2,
        # making the result negative.
        new_val_32 = llvm.inline_asm(
            i32,
            [val_16],
            """
            {
            .reg .b32 s<3>;
            prmt.b32 s0, $1, 0x43, 0x4140;
            and.b32 s1, s0, 0xff7fff7f;
            and.b32 s2, s0, 0xff80ff80;
            sub.bf16x2 $0, s1, s2;
            }
            """,
            "=r,r",
        )
        new_vec = llvm.mlir_undef(ir.VectorType.get((1,), i32))
        new_vec = llvm.insertelement(new_vec, new_val_32, c(0, i32))
        new_registers[idx] = vector.bitcast(
            ir.VectorType.get((2,), new_dtype), new_vec
        )
      return FragmentedArray(_registers=new_registers, _layout=self.layout)
    # Generic path.
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
      case WGMMARowFragLayout() | WGSplatFragLayout():
        new_reg_ty = new_dtype
      case _:
        raise NotImplementedError(f"Unsupported layout {self.layout}")
    for idx, reg in np.ndenumerate(self.registers):
      new_registers[idx] = convert(new_reg_ty, reg)
    return FragmentedArray(_registers=new_registers, _layout=self.layout)

  def reduce_sum(self, scratch) -> ir.Value:
    index = ir.IndexType.get()
    if not isinstance(self.layout, WGStridedFragLayout):
      raise NotImplementedError(f"Unsupported layout {self.layout}")
    result = c(0, self.mlir_dtype)
    for reg in self.registers:
      result = arith.addf(
          result,
          vector.reduction(self.mlir_dtype, vector.CombiningKind.ADD, reg),
      )
    scratch_ty = ir.MemRefType(scratch.type)
    if scratch_ty.element_type != self.mlir_dtype or scratch_ty.shape != [4]:
      raise ValueError(f"Expected shape={(4,)}, {self.mlir_dtype} (got {scratch_ty})")

    if ir.FloatType.isinstance(self.mlir_dtype):
      op = arith.addf
    elif ir.IntegerType.isinstance(self.mlir_dtype):
      op = arith.addi
    else:
      raise NotImplementedError(self.mlir_dtype)

    warp_result = utils.warp_tree_reduce(result, op, 32)
    warp_id = arith.divui(gpu.thread_id(gpu.Dimension.x), c(32, index))
    memref.store(warp_result, scratch, [warp_id])
    utils.warpgroup_barrier()
    zero_index = c(0, index)
    with mgpu.single_thread(per_block=False):
      scratch_vec = vector.load(
          ir.VectorType.get((4,), self.mlir_dtype),
          scratch,
          [zero_index],
      )
      scratch_sum = vector.reduction(
          self.mlir_dtype, vector.CombiningKind.ADD, scratch_vec
      )
      memref.store(scratch_sum, scratch, [zero_index])
    utils.warpgroup_barrier()
    return memref.load(scratch, [zero_index])

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

  def broadcast(self, shape):
    if not isinstance(self.layout, WGSplatFragLayout):
      raise NotImplementedError(self.layout)

    if self.shape == shape:
      return self

    if not self.layout.can_broadcast_to(shape):
      raise ValueError(f"Can't broadcast {self.shape} to {shape}")

    return FragmentedArray(_registers=self.registers, _layout=WGSplatFragLayout(shape))

  def reshape(self, shape):
    if self.shape == shape:
      return self

    if not isinstance(self.layout, WGSplatFragLayout):
      raise NotImplementedError(self.layout)

    if np.prod(shape) != np.prod(self.shape):
      raise ValueError(f"Can't reshape {self.shape} to {shape}")

    return FragmentedArray(_registers=self.registers, _layout=WGSplatFragLayout(shape))

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
      case WGSplatFragLayout():
        self._store_untiled_splat(ref)
      case WGStridedFragLayout():
        self._store_untiled_wg_strided(ref)
      case _:
        raise NotImplementedError(self.layout)

  def _store_untiled_splat(self, ref: ir.Value):
    vec_size = 8 // mgpu.bytewidth(self.mlir_dtype)
    if np.prod(self.shape) < vec_size * WARPGROUP_SIZE:
      vec_size = 1

    if np.prod(self.shape) % WARPGROUP_SIZE * vec_size:
      raise ValueError(self.shape, WARPGROUP_SIZE, vec_size)

    fa = FragmentedArray.splat(
        self.registers.flat[0],
        self.shape,
        layout=WGStridedFragLayout(shape=self.shape, vec_size=vec_size),
    )
    fa.store_untiled(ref)

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
    cols_per_tile = swizzle // bw
    expected_shape = [m // 64, n // cols_per_tile, 64, cols_per_tile]
    if n < cols_per_tile:  # We allow singular tiles shorter than swizzle.
      expected_shape = [m // 64, 1, 64, cols_per_tile]
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
    if m_tile_size != 64 or n_tile_size != (swizzle // bw):
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
    # TODO(apaszke): We could use ldmatrix/stmatrix for 16-bit types.
    bw = mgpu.bytewidth(dtype)
    m, n = shape
    assert m % 64 == 0 and n % 8 == 0  # Implied by the layout.
    cols_per_tile = swizzle_elems = swizzle // bw
    if n < swizzle_elems:
      cols_per_tile = n
    else:
      assert n % swizzle_elems == 0, (n, swizzle_elems)
    if swizzle not in {32, 64, 128}:
      raise NotImplementedError("Only swizzled stores supported")

    c = arith.ConstantOp.create_index
    tidx = arith.remui(gpu.thread_id(gpu.Dimension.x), c(WARPGROUP_SIZE))
    lane_id = arith.remui(tidx, c(32))  # {0, 1, ..., 31}
    warp_id = arith.divui(tidx, c(32))  # {0, 1, 2, 3}
    sub_row_base = arith.divui(lane_id, c(4))  # {0, 1, ..., 7}
    if bw > 2:  # Stagger is only necessary for values larger than 16bit.
      # We split the rows into two groups (left/right) and change the order in
      # which they perform accesses to avoid bank conflicts.
      # It seems that the STS.64 is 2x faster (and the hardware reports no
      # conflicts) when the conflicts are split between half-warps, as
      # opposed to having them within the half-warp. This requires a
      # little more work for the selects, but is ultimately worth it.
      match swizzle:
        case 128:
          is_stagger_left = arith.cmpi(
              arith.CmpIPredicate.eq, arith.remui(sub_row_base, c(2)), c(0)
          )
        case 64:
          is_stagger_left = arith.cmpi(
              arith.CmpIPredicate.eq,
              arith.remui(arith.divui(sub_row_base, c(2)), c(2)),
              c(0),
          )
        case 32:
          # 32-byte tiles of 4-byte types have only 8 columns so there is no way
          # to stagger the memory accesses within a single tile. We could do it
          # across tiles, but that would be a completely different scheme.
          raise NotImplementedError
        case _:
          raise AssertionError(swizzle)
      stagger_amount = swizzle // 64
      if (cols_per_tile // 8) % stagger_amount:
        raise NotImplementedError
    else:
      # We rely on canonicalization to clean up the selects.
      i1 = ir.IntegerType.get_signless(1)
      is_stagger_left = arith.constant(i1, ir.BoolAttr.get(True))
      stagger_amount = 0
    row_base = arith.addi(sub_row_base, arith.muli(warp_id, c(16)))
    col_base = arith.muli(arith.remui(lane_id, c(4)), c(2))  # {0, 2, 4, 6}
    # The swizzle pattern is constant for a given thread.
    col_swizzle_bits = arith.muli(
        arith.divui(sub_row_base, c(128 // swizzle)), c(16 // bw),
    )
    for row_group in range(m // 64):
      for col_group in range(n // cols_per_tile):
        for row_subidx in range(2):
          row = arith.addi(row_base, c(row_subidx * 8))
          for col_subidx in range(cols_per_tile // 8):
            col_subidx_left = col_subidx
            col_subidx_right = col_subidx ^ stagger_amount
            col_off = arith.select(
                is_stagger_left, c(col_subidx_left * 8), c(col_subidx_right * 8)
            )
            col = arith.addi(col_base, col_off)
            col = arith.xori(col, col_swizzle_bits)
            reg_idx_left = col_subidx_left + col_group * (cols_per_tile // 8)
            reg_idx_right = col_subidx_right + col_group * (cols_per_tile // 8)
            left_idx = row_group, reg_idx_left, row_subidx, 0
            right_idx = row_group, reg_idx_right, row_subidx, 0
            idx = c(row_group), c(col_group), row, col
            def get_register(regs, left_idx=left_idx, right_idx=right_idx):
              value_left = regs[left_idx]
              value_right = regs[right_idx]
              return arith.select(is_stagger_left, value_left, value_right)
            def update_registers(regs, new, left_idx=left_idx, right_idx=right_idx):
              regs[left_idx] = arith.select(is_stagger_left, new, regs[left_idx])
              regs[right_idx] = arith.select(is_stagger_left, regs[right_idx], new)
            yield get_register, update_registers, idx

  def tree_flatten(self):
    return list(self.registers.flat), (self.layout, self.registers.shape)

  @classmethod
  def tree_unflatten(cls, aux, flat_registers):
    layout, reg_shape = aux
    registers = np.asarray(flat_registers, dtype=object).reshape(reg_shape)
    return cls(_registers=registers, _layout=layout)
