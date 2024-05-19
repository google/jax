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

import dataclasses
import enum
import itertools

import jax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import vector
import numpy as np

from . import dsl as mgpu

# mypy: ignore-errors

c = mgpu.c
bytewidth = mgpu.bytewidth


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class WGMMAAccumulator:
  """A FragmentedArray that has is synchronized with the async proxy.

  This implies that it requires no additional synchronization when passed in
  as a WGMMA accumulator. In particular, when created from a
  FragmentedArray, the necessary synchronization is inserted at construction.
  """
  value: mgpu.FragmentedArray

  def __init__(self, *, _value: mgpu.FragmentedArray, _sync: bool = True):
    if _value.layout != mgpu.WGMMA_LAYOUT:
      raise ValueError("Only WGMMA layouts supported in WGMMAAccumulator")
    self.value = _value
    if _sync:
      self.value = wgmma_fence(_value)

  @classmethod
  def zero(cls, m, n):
    if m % 64 or n % 8:
      raise ValueError
    f32 = ir.F32Type.get()
    zero = arith.constant(f32, ir.FloatAttr.get(f32, 0.0))
    return cls(
        _value=mgpu.FragmentedArray.splat(zero, (m, n), mgpu.WGMMA_LAYOUT)
    )

  @classmethod
  def from_registers(cls, registers):
    return cls(_value=registers)

  def tree_flatten(self):
    return (self.value,), ()

  @classmethod
  def tree_unflatten(cls, aux, value):
    del aux
    return cls(_value=value[0], _sync=False)


def wgmma_encode(x: int):
  result = (x & 0x3FFFF) >> 4
  if result << 4 != x:
    raise ValueError("Cannot encode value in a WGMMA descriptor")
  return result


def llvm_mul(x, y):
  return llvm.mul(x, y, overflow_flags=llvm.IntegerOverflowFlags.none)


def llvm_add(x, y):
  return llvm.add(x, y, overflow_flags=llvm.IntegerOverflowFlags.none)


def get_memref_base(memref_arg, memory_space=None):
  i64 = ir.IntegerType.get_signless(64)
  memref_ty = ir.MemRefType(memref_arg.type)
  if len(memref_ty.shape) == 0:
    raise NotImplementedError
  elem_bytewidth = bytewidth(memref_ty.element_type)
  rank = len(memref_ty.shape)
  # TODO: Read out memory space from memref
  space = "" if memory_space is None else "<" + str(memory_space) + ">"
  ptr_ty = ir.Type.parse("!llvm.ptr" + space)
  desc_ty = ir.Type.parse(
      f"!llvm.struct<({ptr_ty}, {ptr_ty}, i64, array<{rank} x i64>,"
      f" array<{rank} x i64>)>"
  )
  desc = builtin.UnrealizedConversionCastOp([desc_ty], [memref_arg])
  aligned_ptr = llvm.extractvalue(ptr_ty, desc, [1])
  offset_elems = llvm.extractvalue(i64, desc, [2])
  offset_bytes = llvm_mul(offset_elems, c(elem_bytewidth, i64))
  return llvm.inttoptr(
      ptr_ty, llvm_add(llvm.ptrtoint(i64, aligned_ptr), offset_bytes)
  )


def create_descriptor(
    memref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | None,
    memory_space: int | None = None,
    nvgpu_type=None,
):
  i64 = ir.IntegerType.get_signless(64)
  ptr_val = llvm.ptrtoint(i64, get_memref_base(memref_arg, memory_space))
  if swizzle is None:
    swizzle_encoding = 0
  elif swizzle == 128:
    swizzle_encoding = 1
  else:
    raise NotImplementedError(swizzle)
  encoded_base_addr = llvm.LShrOp(
      llvm.AndOp(ptr_val, c(0x3FFFF, i64)), c(4, i64)
  )
  desc_const = (
      (wgmma_encode(leading_byte_offset) << 16)
      | (wgmma_encode(stride_byte_offset) << 32)
      |
      # We ignore the offset
      (swizzle_encoding << 62)
  )
  desc = llvm.OrOp(encoded_base_addr, c(desc_const, i64))
  if nvgpu_type is not None:
    desc = builtin.UnrealizedConversionCastOp([nvgpu_type], [desc])
  return desc.result


def wgmma_m64k128B(
    acc: np.ndarray,  # of register Values
    a,
    b_descriptor: ir.Value,
    a_transpose: bool | None,
    b_transpose: bool,
    a_k_stride: int | None,
    b_k_stride: int,
    n: int,
    element_type: ir.Type,
):
  f32 = ir.F32Type.get()
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  index = ir.IndexType.get()
  if b_k_stride % 16:
    raise ValueError
  if n % (128 // bytewidth(element_type)):
    raise ValueError
  # Only 16-bit types support transposes
  supports_transpose = bytewidth(element_type) == 2
  if not supports_transpose and (a_transpose or b_transpose):
    raise ValueError("Only f16 WGMMA supports transposes")
  if a_in_regs := isinstance(a, mgpu.FragmentedArray):
    if a.mlir_dtype != ir.F16Type.get() and a.mlir_dtype != ir.BF16Type.get():
      raise ValueError(f"Unsupported A register array dtype: {a.mlir_dtype}")
    if a.layout != mgpu.WGMMA_LAYOUT or a.shape != (64, 64):
      raise ValueError("Unsupported A register array layout")
    if a_k_stride is not None or a_transpose is not None:
      raise ValueError("Unsupported WGMMA features with A in registers")
  else:
    if a_k_stride is None or a_k_stride % 16:
      raise ValueError
    if a_transpose is None:
      raise ValueError

  num_acc_regs = n // 2
  num_imm_regs = 4 if supports_transpose else 2

  if a_in_regs:
    a_reg_constraints = ["r"] * 4  # 4x f16x2 registers
    num_imm_regs -= 1  # transpose not supported for a in registers
  else:
    a_reg_constraints = ["l"]  # descriptor
  # Reference for i/o aliasing: https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
  # Seems like it's not actually documented in LLVM IR docs.
  reg_constraints_list = (
      ["=f"] * num_acc_regs  # accumulator registers
      + [str(i) for i in range(num_acc_regs)]  # we alias outputs as inputs, too.
      + a_reg_constraints  # a descriptor / registers
      + ["l"] * 1  # b descriptor
      + ["n"] * (1 + num_imm_regs)  # literal constants
  )
  reg_constraints = ",".join(reg_constraints_list)

  reg_count = itertools.count()

  def take_regs(n):
    return (f"${i}" for i in itertools.islice(reg_count, n))

  acc_reg_vector = "{" + ",".join(take_regs(num_acc_regs)) + "}"
  for _ in take_regs(num_acc_regs):  # Ignore next entries: aliasing.
    pass
  if a_in_regs:
    a_regs = "{" + ",".join(take_regs(len(a_reg_constraints))) + "}"
  else:
    a_regs, = take_regs(1)
  b_desc_reg, use_out_reg = take_regs(2)
  imm_regs = ", ".join(take_regs(num_imm_regs))  # Immediate regs (scale, ...).
  assert next(reg_count) == len(reg_constraints_list)
  el_ty = element_type
  k_instr = 32 // bytewidth(element_type)
  wgmma_instr = (
      f"wgmma.mma_async.sync.aligned.m64n{n}k{k_instr}.f32.{el_ty}.{el_ty} "
      f"{acc_reg_vector}, {a_regs}, {b_desc_reg}, p, {imm_regs};"
  )
  ptx = f"{{ .reg .pred p; setp.ne.b32 p, {use_out_reg}, 0; {wgmma_instr}}}\n"

  def lc(x):
    return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result

  def as_i32_reg(v):
    return llvm.extractelement(
        vector.bitcast(ir.VectorType.get((1,), i32), v), lc(0)
    )

  use_out = scale_a = scale_b = lc(1)
  imms = [use_out, scale_a, scale_b]
  if supports_transpose and a_transpose is not None:
    imms += [lc(int(a_transpose)), lc(int(b_transpose))]
  elif supports_transpose:
    imms += [lc(int(b_transpose))]
  if acc.ndim != 4 or acc.shape[0] != 1 or acc.shape[2:] != (2, 1):
    raise ValueError(acc.shape)
  acc_regs = [  # pylint: disable=g-complex-comprehension
      vector.extractelement(reg, position=c(pos, index))
      for reg in acc.flat
      for pos in range(2)
  ]
  acc_struct_type = ir.Type.parse(
      f"!llvm.struct<({','.join('f32' for _ in acc_regs)})>"
  )
  for i in range(4):
    # Slice out the relevant part of A or advance the A descriptor.
    if a_in_regs:
      a_slice = a[:, (i * 16) : ((i + 1) * 16)]
      a_args = [as_i32_reg(v) for v in a_slice.registers.flat]
    else:
      if i > 0:
        a = llvm_add(
            a,
            llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, a_k_stride >> 4)),
        )
      a_args = [a]
    # Advance the B descriptor.
    if i > 0:
      b_descriptor = llvm_add(
          b_descriptor,
          llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, b_k_stride >> 4)),
      )
    assert len(a_args) == len(a_reg_constraints)
    acc_struct = llvm.inline_asm(
        acc_struct_type,
        [*acc_regs, *a_args, b_descriptor, *imms],
        ptx,
        reg_constraints,
        asm_dialect=0,
        has_side_effects=True,
    )
    acc_regs = [
        llvm.extractvalue(f32, acc_struct, [i]) for i in range(len(acc_regs))
    ]
  acc_vec_regs = []
  for first, second in zip(acc_regs[::2], acc_regs[1::2]):
    vec = llvm.mlir_undef(ir.VectorType.get((2,), f32))
    vec = llvm.insertelement(vec, first, position=lc(0))
    vec = llvm.insertelement(vec, second, position=lc(1))
    acc_vec_regs.append(vec)
  return np.asarray(acc_vec_regs, dtype=object).reshape(acc.shape)


class WGMMALayout(enum.Enum):
  ROW_MAJOR = enum.auto()
  COL_MAJOR = enum.auto()


# TODO(apaszke): Remove WGMMALayout. Make input shapes logical and infer
# transpositions from memref strides.
def wgmma(
    acc: WGMMAAccumulator,
    a,
    b,
    *,
    # Order only applies within each tile!
    a_order: WGMMALayout | None = None,
    b_order: WGMMALayout = WGMMALayout.ROW_MAJOR,
):
  if a_in_regs := isinstance(a, mgpu.FragmentedArray):
    a_element_type = a.mlir_dtype
    a_shape = a.shape
  else:
    a_ty = ir.MemRefType(a.type)
    a_element_type = a_ty.element_type
    a_shape = a_ty.shape
  b_ty = ir.MemRefType(b.type)
  supported_types = {ir.F16Type.get(), ir.BF16Type.get(), ir.F32Type.get()}
  if a_element_type not in supported_types:
    raise ValueError(a_element_type)
  if b_ty.element_type not in supported_types:
    raise ValueError(b_ty.element_type)
  if (element_type := a_element_type) != b_ty.element_type:
    raise ValueError
  element_bytewidth = bytewidth(element_type)
  kn_tile = 128 // element_bytewidth

  groups_k, groups_n = b_ty.shape[:2]
  if b_ty.shape[2:] != [kn_tile, kn_tile]:
    raise ValueError(b_ty.shape)

  if a_in_regs:
    if a_element_type != ir.F16Type.get() and a_element_type != ir.BF16Type.get():
      raise ValueError(a_element_type)
    if a_shape[0] % 64 or a_shape[1] % kn_tile:
      raise ValueError(a_shape)
    if a_shape[1] // kn_tile != groups_k:
      raise ValueError(a_shape[1] // kn_tile, groups_k)
    groups_m = a_shape[0] // 64
    if a_order is not None:
      raise ValueError(
          "a_order can only be specified when A is in shared memory"
      )
  else:
    groups_m = a_shape[0]
    if a_shape[1] != groups_k:
      raise ValueError(a_shape[1], groups_k)
    if a_shape[2:] != [64, kn_tile]:
      raise ValueError(a_shape)
    if a_order is None:
      a_order = WGMMALayout.ROW_MAJOR

  row_major = WGMMALayout.ROW_MAJOR
  col_major = WGMMALayout.COL_MAJOR
  a_desc_fields = dict(
      leading_byte_offset=((1 if a_order == row_major else 512) << 4),
      stride_byte_offset=(64 << 4),
      swizzle=128,
      memory_space=3,
  )
  b_desc_fields = dict(
      leading_byte_offset=((512 if b_order == row_major else 1) << 4),
      stride_byte_offset=(64 << 4),
      swizzle=128,
      memory_space=3,
  )
  wgmma_params = dict(
      a_transpose=a_order == col_major,
      b_transpose=b_order == row_major,
      a_k_stride=(2 if a_order == row_major else 128) * 16,
      b_k_stride=(128 if b_order == row_major else 2) * 16,
      n=(groups_n * kn_tile),
      element_type=ir.FloatTF32Type.get()
      if ir.F32Type.isinstance(element_type)
      else element_type,
  )
  if a_in_regs:
    wgmma_params["a_k_stride"] = wgmma_params["a_transpose"] = None

  if a_in_regs:
    nvvm.wgmma_fence_aligned()  # Make sure the registers are ready.
    a_m_byte_stride = a_k_byte_stride = a_desc_base = None  # Silence pytype.
  else:
    a_desc_base = create_descriptor(a, **a_desc_fields)
    a_strides, _ = ir.MemRefType(a.type).get_strides_and_offset()
    a_byte_strides = [s * element_bytewidth for s in a_strides]
    a_m_byte_stride, a_k_byte_stride = a_byte_strides[:2]
    if a_byte_strides[2:] != [128, element_bytewidth]:
      raise ValueError(a_byte_strides)
  b_desc_base = create_descriptor(b, **b_desc_fields)
  b_strides, _ = b_ty.get_strides_and_offset()
  b_byte_strides = [s * element_bytewidth for s in b_strides]
  b_k_byte_stride = b_byte_strides[0]
  if b_byte_strides[1:] != [128 * kn_tile, 128, element_bytewidth]:
    raise ValueError(b_byte_strides)

  i64 = ir.IntegerType.get_signless(64)
  new_acc_regs = acc.value.registers.copy()
  for mi in range(groups_m):
    for ki in range(groups_k):
      if a_in_regs:
        a_mk = a[mi * 64 : (mi + 1) * 64, ki * kn_tile : (ki + 1) * kn_tile]
      else:
        a_mk = llvm_add(
            a_desc_base,
            c(wgmma_encode(mi * a_m_byte_stride + ki * a_k_byte_stride), i64),
        )
      b_k = llvm_add(b_desc_base, c(wgmma_encode(ki * b_k_byte_stride), i64))
      new_acc_regs[mi : mi + 1] = wgmma_m64k128B(
          new_acc_regs[mi : mi + 1], a_mk, b_k, **wgmma_params
      )
  return WGMMAAccumulator(
      _value=mgpu.FragmentedArray(
          _registers=new_acc_regs, _layout=mgpu.WGMMA_LAYOUT
      ),
      _sync=False,
  )


def wgmma_fence(acc: mgpu.FragmentedArray):
  i32 = ir.IntegerType.get_signless(32)
  index = ir.IndexType.get()
  f32 = ir.F32Type.get()
  acc_regs = [  # pylint: disable=g-complex-comprehension
      vector.extractelement(reg, position=c(pos, index))
      for reg in acc.registers.flat
      for pos in range(2)
  ]

  def lc(x):
    return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result

  reg_constraints_list = ["=f"] * len(acc_regs) + ["f"] * len(acc_regs)
  reg_constraints = ",".join(reg_constraints_list)
  ptx_lines = [
      f"mov.f32 ${i}, ${len(acc_regs)+i}"
      for i in range(len(acc_regs))
  ]
  ptx_lines.append("wgmma.fence.sync.aligned")
  ptx = ";\n".join(ptx_lines) + ";\n"
  acc_struct_type = ir.Type.parse(
      f"!llvm.struct<({','.join('f32' for _ in acc_regs)})>"
  )
  acc_struct = llvm.inline_asm(
      acc_struct_type,
      acc_regs,
      ptx,
      reg_constraints,
      asm_dialect=0,
      has_side_effects=True,
  )
  acc_regs = [
      llvm.extractvalue(f32, acc_struct, [i]) for i in range(len(acc_regs))
  ]
  acc_vec_regs = []
  for first, second in zip(acc_regs[::2], acc_regs[1::2]):
    vec = llvm.mlir_undef(ir.VectorType.get((2,), f32))
    vec = llvm.insertelement(vec, first, position=lc(0))
    vec = llvm.insertelement(vec, second, position=lc(1))
    acc_vec_regs.append(vec)
  new_regs = np.asarray(acc_vec_regs, dtype=object).reshape(acc.registers.shape)
  return mgpu.FragmentedArray(_registers=new_regs, _layout=acc.layout)
