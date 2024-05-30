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
"""Tests for Mosaic GPU DSL functions and utilities."""

import operator
from functools import partial
from typing import Optional

from absl.testing import absltest, parameterized
import numpy as np
import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
try:
  import jax._src.lib.mosaic_gpu  # noqa: F401
  HAS_MOSAIC_GPU = True
except ImportError:
  HAS_MOSAIC_GPU = False
else:
  from jax.experimental.mosaic import gpu as mosaic_gpu
  from jax.experimental.mosaic.gpu import dsl as mgpu
  from jax.experimental.mosaic.gpu import profiler
  from jax.experimental.mosaic.gpu.utils import *  # noqa: F403
  from jax._src.lib.mlir.dialects import gpu
  from jax._src.lib.mlir.dialects import llvm


# ruff: noqa: F405
config.update("jax_traceback_filtering", "off")
config.parse_flags_with_absl()

def nd_loop(bounds, body, *, _idxs = ()):
  if not bounds:
    body(*_idxs)
    return
  bound, *other_bounds = bounds
  @fori(bound, ())
  def _loop_body(i, _):
    nd_loop(other_bounds, body, _idxs=(*_idxs, i))
    return ()


def mlir_sum(elems):
  assert elems
  total = elems[0]
  for elem in elems[1:]:
    total = arith.addi(total, elem)
  return total


def copy(src: ir.Value, dst: ir.Value, swizzle: Optional[int] = None):
  index = ir.IndexType.get()
  thread_id = gpu.thread_id(gpu.Dimension.x)
  stride = gpu.block_dim(gpu.Dimension.x)
  for dim in (gpu.Dimension.y, gpu.Dimension.z):
    thread_id = arith.addi(thread_id, arith.muli(gpu.thread_id(dim), stride))
    stride = arith.muli(stride, gpu.block_dim(dim))
  is_first_thread = arith.cmpi(arith.CmpIPredicate.eq, thread_id, c(0, index))
  src_ty = ir.MemRefType(src.type)
  dst_ty = ir.MemRefType(dst.type)
  if src_ty.shape != dst_ty.shape:
    raise ValueError(
        f"src and dst shapes don't match: {src_ty.shape} != {dst_ty.shape}"
    )
  shape = src_ty.shape
  dyn_strides = [c(s, index) for s in get_contiguous_strides(shape)]
  with ir.InsertionPoint(scf.IfOp(is_first_thread).then_block):
    def body(*idx):
      dst_idx = idx
      if swizzle is not None:
        if swizzle != 128:
          raise NotImplementedError("Only swizzle 128B implemented")
        # TODO(apaszke): This can probably be cleaned up.
        # But it works and it's test-only, so it doesn't matter much.
        # After all, swizzle should just be an xor of row and linear idx,
        # adjusted for the bytewidth.
        bytes_per_element = bytewidth(src_ty.element_type)
        elems_per_tile = 1024 // bytes_per_element
        elems_per_row = elems_per_tile // 8
        elems_per_group = 16 // bytes_per_element
        linear_idx = c(0, index)
        for stride, i in zip(dyn_strides, idx):
          linear_idx = arith.addi(linear_idx, arith.muli(i, stride))
        tile_offset = arith.remui(linear_idx, c(elems_per_tile, index))
        linear_tile_start = arith.subi(linear_idx, tile_offset)
        row = arith.divui(tile_offset, c(elems_per_row, index))
        row_offset = arith.remui(tile_offset, c(elems_per_row, index))
        src_group = arith.divui(row_offset, c(elems_per_group, index))
        group_offset = arith.remui(row_offset, c(elems_per_group, index))
        dst_group = arith.xori(src_group, row)
        dst_linear_idx = mlir_sum([
            linear_tile_start,
            arith.muli(row, c(elems_per_row, index)),
            arith.muli(dst_group, c(elems_per_group, index)),
            group_offset,
        ])
        dst_idx = [
            arith.remui(arith.divui(dst_linear_idx, stride), c(bound, index))
            for stride, bound in zip(dyn_strides, shape)
        ]
      memref.store(memref.load(src, idx), dst, dst_idx)
    nd_loop([c(d, index) for d in shape], body)
    scf.yield_([])
  gpu.barrier()
  nvvm.fence_proxy(nvvm.ProxyKind.async_)


def iota_tensor(m, n, mlir_dtype):
  assert m % 64 == 0
  assert n % 8 == 0
  def c(i):
    return arith.constant(index, ir.IntegerAttr.get(index, i))
  index = ir.IndexType.get()
  i32 = ir.IntegerType.get_signless(32)
  warp_id = arith.divui(gpu.thread_id(gpu.Dimension.x), c(32))
  within_warp_id = arith.remui(gpu.thread_id(gpu.Dimension.x), c(32))
  warp_row_start = arith.muli(warp_id, c(16))
  within_warp_row = arith.divui(within_warp_id, c(4))
  start_row = arith.addi(warp_row_start, within_warp_row)
  start_col = arith.muli(arith.remui(within_warp_id, c(4)), c(2))
  registers = np.empty((m // 64, n // 8, 2, 1), dtype=object)
  for row_tile, col_tile, row_subtile, _ in np.ndindex(registers.shape):
    row = arith.addi(start_row, c(row_tile * 64 + row_subtile * 8))
    col = arith.addi(start_col, c(col_tile * 8))
    row_value_base = arith.muli(row, c(n))
    vec = llvm.mlir_undef(ir.VectorType.get((2,), i32))
    for col_offset in range(2):
      value = arith.addi(row_value_base, arith.addi(c(col_offset), col))
      value = arith.index_cast(i32, value)
      vec = vector.insertelement(value, vec, position=c(col_offset))
    registers[row_tile, col_tile, row_subtile, 0] = vec
  t = mgpu.FragmentedArray(_registers=registers, _layout=mgpu.WGMMA_LAYOUT)
  return t.astype(mlir_dtype)


class TestCase(parameterized.TestCase):

  def setUp(self):
    if not HAS_MOSAIC_GPU:
      self.skipTest("jaxlib built without Mosaic GPU")
    super().setUp()
    self.prng = np.random.default_rng(1234)
    self.ctx = mlir.make_ir_context()
    self.ctx.__enter__()
    self.loc = ir.Location.unknown()
    self.loc.__enter__()

  def tearDown(self):
    self.loc.__exit__(None, None, None)
    self.ctx.__exit__(None, None, None)
    del self.loc, self.ctx
    super().tearDown()


class TestUtilTest(TestCase):

  def test_copy(self):
    def kernel(ctx, src, dst, _):
      copy(src, dst)
    x = jnp.arange(2 * 3 * 5).reshape(2, 5, 3)
    y = mosaic_gpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, ())(x)
    np.testing.assert_array_equal(y, x)

  def test_copy_swizzle(self):
    def kernel(ctx, src, dst, _):
      copy(src, dst, swizzle=128)
    x = jnp.arange(8 * 32, dtype=jnp.float32).reshape(8, 32)
    y = mosaic_gpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, ())(x)
    expected = np.zeros_like(y)
    for i in range(8):
      for j in range(8):
        js = j ^ i
        expected[i, (j * 4):(j * 4) + 4] = x[i, (js * 4):(js * 4) + 4]
    np.testing.assert_array_equal(y, expected)

  def test_copy_swizzle_noop(self):
    # Two swizzles cancel out
    def kernel(ctx, src, dst, smem):
      copy(src, smem, swizzle=128)
      copy(smem, dst, swizzle=128)
    x = jnp.arange(8 * 32, dtype=jnp.float32).reshape(8, 32)
    y = mosaic_gpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)
    np.testing.assert_array_equal(y, x)

  def test_iota_tensor(self):
    m = n = 64
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      index = ir.IndexType.get()
      registers = iota_tensor(m, n, f32).registers
      assert registers.size == 16, registers.size
      for i, vec_reg in enumerate(registers.flat):
        for j in range(2):
          reg = vector.extractelement(vec_reg, position=c(j, index))
          memref.store(
              reg, dst, [gpu.thread_id(gpu.Dimension.x), c(2 * i + j, index)]
          )
    out_shape = jax.ShapeDtypeStruct((128, 32), jnp.float32)
    regs = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    thread_ids = np.arange(128)
    warp_ids = thread_ids // 32
    lane_ids = thread_ids % 32
    thread_rows = warp_ids * 16 + lane_ids // 4
    thread_start_cols = (lane_ids % 4) * 2
    thread_cols = thread_start_cols[:, None] + (np.arange(n // 8)[None] * 8)
    regs = regs.reshape(128, 8, 2, 2)
    for row_half in range(2):
      for col_half in range(2):
        np.testing.assert_array_equal(
            regs[..., row_half, col_half],
            (thread_rows[:, None] + row_half * 8) * n + thread_cols + col_half
        )


class MemRefTest(TestCase):
  @parameterized.product(
      dim=tuple(range(3)),
      strided=(False, True)
  )
  def test_unsqueeze(self, dim, strided):
    def kernel(ctx, inp, out, _):
      if strided:
        for i in range(8):
          s = ds(i, 1)
          out_slice = s if dim != 0 else (slice(None), s)
          copy(
              memref_unsqueeze(memref_slice(inp, s), dim),
              memref_slice(out, out_slice),
          )
      else:
        copy(memref_unsqueeze(inp, dim), out)
    x = np.arange(8 * 16, dtype=jnp.float32).reshape(8, 16)
    out_shape = list(x.shape)
    out_shape.insert(dim, 1)
    out_ty = jax.ShapeDtypeStruct(out_shape, jnp.float32)
    y = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(out_shape))

  @parameterized.product(
      dim=tuple(range(2)),
      strided=(False, True)
  )
  def test_unfold(self, dim, strided):
    in_shape = (8, 16)
    def kernel(ctx, inp, out, _):
      if strided:
        # We slice the dim we don't unfold
        for i in range(in_shape[1 - dim] // 4):
          s = ds(i * 4, 4)
          in_slice = s if dim == 1 else (slice(None), s)
          out_slice = s if dim == 1 else (slice(None),) * 3 + (s,)
          copy(
              memref_unfold(memref_slice(inp, in_slice), dim, (2, 2, None)),
              memref_slice(out, out_slice),
          )
      else:
        copy(memref_unfold(inp, dim, (2, 2, None)), out)
    x = np.arange(np.prod(in_shape), dtype=jnp.float32).reshape(in_shape)
    out_shape = list(in_shape)
    out_shape[dim:dim + 1] = [2, 2, out_shape[dim] // 4]
    out_ty = jax.ShapeDtypeStruct(out_shape, jnp.float32)
    y = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(out_ty.shape))

  @parameterized.product(
      dim=tuple(range(2)),
  )
  def test_fold_not_strided(self, dim):
    def kernel(ctx, inp, out, _):
      copy(memref_fold(inp, dim, 2), out)

    x = np.arange(8 * 2 * 8, dtype=jnp.float32).reshape(8, 2, 8)
    out_ty = jax.ShapeDtypeStruct((16, 8) if dim == 0 else (8, 16), jnp.float32)
    y = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(out_ty.shape))

  @parameterized.named_parameters([
      ("packed", (4, 4, 4), (16, 4, 1), 1, 2, False),
      ("strided_end", (4, 4, 4, 4), (256, 64, 16, 4), 1, 2, False),
      ("strided_bot", (4, 4, 4, 4), (256, 16, 4, 1), 1, 2, False),
      ("strided_top", (4, 4, 4, 4), (256, 64, 4, 1), 1, 2, True),
      ("strided_mid", (4, 4, 4, 4), (265, 64, 16, 1), 1, 3, True),
      ("overap", (2, 4, 4), (16, 1, 1), 0, 3, True),
  ])
  def test_fold_strided(
      self, shape, strides, dim, fold_rank, throws_not_impl
  ):
    expanded_shape = get_packed_shape(strides, shape)
    total_size = np.prod(expanded_shape)
    np_inp = np.arange(total_size, dtype=jnp.float32).reshape(expanded_shape)
    index = tuple([slice(0, s) for s in shape])

    # Reference implementation
    def np_fold(inp, dim, fold_rank):
      out_shape = list(inp.shape)
      out_shape[dim : dim + fold_rank] = [
          int(np.prod(inp.shape[dim : dim + fold_rank]))
      ]
      if throws_not_impl:
        return jax.ShapeDtypeStruct(shape=out_shape, dtype=inp.dtype)
      else:
        return inp.reshape(*out_shape)

    total_size = np.prod(shape) * np.prod(strides)

    def do_test():
      def kernel(ctx, inp, out, _):
        copy(memref_fold(memref_slice(inp, index), dim, fold_rank), out)

      out = np_fold(np_inp[index], dim, fold_rank)
      y = mosaic_gpu.as_gpu_kernel(
          kernel, (1, 1, 1), (128, 1, 1), np_inp, out, ()
      )(np_inp)
      assert (
          not throws_not_impl
      ), "If it should have thrown it would during the call."
      np.testing.assert_array_equal(y, out)

    if throws_not_impl:
      with self.assertRaises(NotImplementedError):
        do_test()
    else:
      do_test()


def get_packed_shape(strides, shape):
  perm = sorted(range(len(strides)), key=lambda i: strides[i], reverse=True)
  ordered_strides = [strides[i] for i in perm]
  ordered_shape = [shape[i] for i in perm]
  packed_shape = [ordered_shape[-1]]
  packed_shape += [
      stride0 // stride
      for stride0, stride in zip(ordered_strides, ordered_strides[1:])
  ]
  # Invert permutation
  inv_perm = [None] * len(perm)
  for i, p in enumerate(perm):
    inv_perm[p] = i
  return [packed_shape[i] for i in inv_perm]


class WGMMATest(TestCase):

  @parameterized.named_parameters(
      ("f32", ir.F32Type, jnp.float32), ("f16", ir.F16Type, jnp.float16)
  )
  def test_store_untiled(self, mlir_dtype_cls, jax_dtype):
    mlir_dtype = mlir_dtype_cls.get()
    def kernel(ctx, out, _):
      del ctx
      iota_tensor(64, 64, mlir_dtype).store_untiled(out)
    expected = np.arange(64 * 64, dtype=jax_dtype).reshape(64, 64)
    iota = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, ()
    )()
    np.testing.assert_array_equal(iota, expected)

  @parameterized.named_parameters(
      ("f32", ir.F32Type.get, jnp.float32),
      ("f16", ir.F16Type.get, jnp.float16),
      ("i8", partial(ir.IntegerType.get_signless, 8), jnp.int8),
  )
  def test_store_tiled(self, mlir_dtype_cls, jax_dtype):
    mlir_dtype = mlir_dtype_cls()
    m = 128
    n = 256
    tiling = (64, 128 // bytewidth(mlir_dtype))
    def kernel(ctx, out, smem):
      del ctx
      iota_tensor(m, n, mlir_dtype).store_tiled(smem, swizzle=128)
      copy(smem, out, swizzle=128)
    expected = (
        np.arange(m * n, dtype=jax_dtype)
        .reshape(m // tiling[0], tiling[0], n // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )
    iota = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, expected
    )()
    np.testing.assert_array_equal(iota, expected)

  @parameterized.named_parameters(
      ("bf16_i8",
      ir.BF16Type.get, jnp.bfloat16,
       lambda: ir.IntegerType.get_signless(8), jnp.int8),
      ("i8_bf16",
       lambda: ir.IntegerType.get_signless(8), jnp.int8,
       ir.BF16Type.get, jnp.bfloat16),
      ("i8_i8",
       lambda: ir.IntegerType.get_signless(8), jnp.int8,
       lambda: ir.IntegerType.get_signless(8), jnp.int8),
  )
  def test_convert_tiled(self,
                         mlir_dtype_cls_from, jax_dtype_from,
                         mlir_dtype_cls_to, jax_dtype_to):
    mlir_dtype_from = mlir_dtype_cls_from()
    mlir_dtype_to = mlir_dtype_cls_to()
    m = 128
    n = 256 // bytewidth(mlir_dtype_from)
    def kernel(ctx, inp, out, smem):
      del ctx
      smem_from, smem_to = smem
      copy(inp, smem_from, swizzle=128)
      t = mgpu.FragmentedArray.load_tiled(smem_from, swizzle=128)
      t = t.astype(mlir_dtype_to)
      t.store_tiled(smem_to, swizzle=128)
      copy(smem_to, out, swizzle=128)

    from_tiling = (64, 128 // bytewidth(mlir_dtype_from))
    to_tiling = (64, 128 // bytewidth(mlir_dtype_to))
    expected_raw = self.prng.integers(
        low=-127, high=127, size=(m, n), dtype=np.int8
    )
    expected = lambda jax_dtype, tiling: expected_raw.reshape(
        m // tiling[0], tiling[0], n // tiling[1], tiling[1]
    ).transpose(0, 2, 1, 3).astype(jax_dtype)

    expected_from = expected(jax_dtype_from, from_tiling)
    expected_to = expected(jax_dtype_to, to_tiling)
    res = mosaic_gpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        expected_from,
        expected_to,
        (expected_from, expected_to),
    )(expected_from)
    np.testing.assert_array_equal(res, expected_to)

  @parameterized.named_parameters(
      ("f32", ir.F32Type.get, jnp.float32),
      ("f16", ir.F16Type.get, jnp.float16),
      ("i8", partial(ir.IntegerType.get_signless, 8), jnp.int8),
  )
  def test_load_tiled(self, mlir_dtype_cls, jax_dtype):
    mlir_dtype = mlir_dtype_cls()
    m = 128
    n = 256 // bytewidth(mlir_dtype)
    tiling = (64, 128 // bytewidth(mlir_dtype))
    def kernel(ctx, in_, out, smem):
      del ctx
      smem1, smem2 = smem
      copy(in_, smem1, swizzle=128)
      t = mgpu.FragmentedArray.load_tiled(smem1, swizzle=128)
      t.store_tiled(smem2, swizzle=128)
      copy(smem2, out, swizzle=128)
    expected = (
        np.arange(m * n, dtype=jax_dtype)
        .reshape(m // tiling[0], tiling[0], n // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )
    iota = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), expected, expected, (expected,) * 2
    )(expected)
    np.testing.assert_array_equal(iota, expected)

  @parameterized.product(
      lhs_transpose=(False, True),
      rhs_transpose=(False, True),
      mlir_dtype_cls=(ir.F16Type, ir.BF16Type, ir.F32Type),
      m=(64, 128, 192),
      n=(32, 64, 128, 192),
      k_steps=(1, 2),
      tma_inputs=(False, True),
      out_dtype_cls=(ir.F16Type, ir.F32Type),
  )
  def test_wgmma(
      self,
      m,
      n,
      k_steps,
      mlir_dtype_cls,
      lhs_transpose,
      rhs_transpose,
      tma_inputs,
      out_dtype_cls,
  ):
    if out_dtype_cls is ir.F16Type and mlir_dtype_cls is ir.F32Type:
      raise self.skipTest("Incompatible types")

    mlir_dtype = mlir_dtype_cls.get()
    if ir.F32Type.isinstance(mlir_dtype):  # We actually use tf32 instead
      jax_dtype = jnp.float32
      if lhs_transpose or not rhs_transpose:
        self.skipTest("Transpose only supported in 16-bit WGMMA")
      exponent_bits, mantissa_bits = 8, 10  # Use tf32
    elif bytewidth(mlir_dtype) == 2:
      if n % 64 != 0:
        self.skipTest("16-bit WGMMA only supports n % 64 == 0")
      if ir.F16Type.isinstance(mlir_dtype):
        jax_dtype = jnp.float16
        exponent_bits, mantissa_bits = 5, 10
      elif ir.BF16Type.isinstance(mlir_dtype):
        jax_dtype = jnp.bfloat16
        exponent_bits, mantissa_bits = 8, 7
      else:
        raise NotImplementedError(mlir_dtype)
    else:
      raise NotImplementedError(mlir_dtype)
    nk_tile = 128 // bytewidth(mlir_dtype)
    k = nk_tile * k_steps
    assert m % 64 == 0 and n % nk_tile == 0
    index = ir.IndexType.get()

    row_major = mgpu.WGMMALayout.ROW_MAJOR
    col_major = mgpu.WGMMALayout.COL_MAJOR
    lhs_order = col_major if lhs_transpose else row_major
    rhs_order = col_major if rhs_transpose else row_major

    def kernel(ctx, lhs, rhs, out, scratch):
      lhs_smem, rhs_smem = scratch
      if tma_inputs:
        lhs_transform = (mosaic_gpu.TileTransform((64, nk_tile)),)
        if lhs_transpose:
          assert nk_tile == 64  # Make sure we didn't have to transpose tiling.
          lhs_transform += (mosaic_gpu.TransposeTransform((1, 0, 2, 3)),)
        rhs_transform = (mosaic_gpu.TileTransform((nk_tile, nk_tile)),)
        if rhs_transpose:
          rhs_transform += (mosaic_gpu.TransposeTransform((1, 0, 2, 3)),)
        barriers = BarrierArray(2)
        ctx.async_copy(
            src_ref=lhs,
            dst_ref=lhs_smem,
            swizzle=128,
            gmem_transform=lhs_transform,
            barrier=barriers[0],
        )
        ctx.async_copy(
            src_ref=rhs,
            dst_ref=rhs_smem,
            swizzle=128,
            gmem_transform=rhs_transform,
            barrier=barriers[1],
        )
        for i in range(2):
          barriers[i].wait()
      else:
        for mi in range(m // 64):
          for ki in range(k // nk_tile):
            lhs_slice = (
                ds(c(mi * 64, index), 64),
                ds(c(ki * nk_tile, index), nk_tile),
            )
            if lhs_transpose:
              lhs_slice = lhs_slice[::-1]
            copy(
                src=memref_slice(lhs, lhs_slice),
                dst=memref_slice(lhs_smem, (mi, ki)),
                swizzle=128,
            )
        for ki in range(k // nk_tile):
          k_slice = ds(c(ki * nk_tile, index), nk_tile)
          for ni in range(n // nk_tile):
            rhs_slice = (k_slice, ds(c(ni * nk_tile, index), nk_tile))
            if rhs_transpose:
              rhs_slice = rhs_slice[::-1]
            copy(
                src=memref_slice(rhs, rhs_slice),
                dst=memref_slice(rhs_smem, (ki, ni)),
                swizzle=128,
            )
      init_acc = mgpu.WGMMAAccumulator.zero(m=m, n=n, dtype=out_dtype_cls.get())
      acc = mgpu.wgmma(
          init_acc, lhs_smem, rhs_smem,
          a_order=lhs_order, b_order=rhs_order,
      )
      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)
      acc.value.store_untiled(out)

    def quantize(x):
      # Quantize the input to avoid rounding when feeding the WGMMA
      return jax.lax.reduce_precision(x, exponent_bits, mantissa_bits)

    x_shape = (k, m) if lhs_transpose else (m, k)
    x = quantize(self.prng.uniform(-1, 1, x_shape)).astype(jax_dtype)
    y_shape = (n, k) if rhs_transpose else (k, n)
    y = quantize(self.prng.uniform(-1, 1, y_shape)).astype(jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    scratch_shape = [
        jax.ShapeDtypeStruct((m // 64, k // nk_tile, 64, nk_tile), jax_dtype),
        jax.ShapeDtypeStruct(
            (k // nk_tile, n // nk_tile, nk_tile, nk_tile), jax_dtype
        ),
    ]
    z = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (x, y), out_shape, scratch_shape
    )(x, y)
    x32, y32 = x.astype(np.float32), y.astype(np.float32)
    ref = (x32.T if lhs_transpose else x32) @ (y32.T if rhs_transpose else y32)
    np.testing.assert_allclose(z, ref, atol=5e-6)

  # TODO(apaszke): Add support for f32
  @parameterized.product(
      m=(64, 128, 192),
      n=(64, 128, 192),
      k_steps=(1, 2),
      rhs_transpose=(False, True),
      mlir_dtype_cls=(ir.F16Type, ir.BF16Type),
  )
  def test_wgmma_reg_lhs(self, m, n, k_steps, rhs_transpose, mlir_dtype_cls):
    k = 64 * k_steps
    index = ir.IndexType.get()

    row_major = mgpu.WGMMALayout.ROW_MAJOR
    col_major = mgpu.WGMMALayout.COL_MAJOR
    rhs_order = col_major if rhs_transpose else row_major

    def kernel(ctx, rhs, out, rhs_smem):
      del ctx
      for ki in range(k_steps):
        for ni in range(n // 64):
          rhs_slice = (ds(c(ki * 64, index), 64), ds(c(ni * 64, index), 64))
          if rhs_transpose:
            rhs_slice = rhs_slice[::-1]
          copy(
              src=memref_slice(rhs, rhs_slice),
              dst=memref_slice(rhs_smem, (ki, ni)),
              swizzle=128,
          )
      init_acc = mgpu.WGMMAAccumulator.zero(m=m, n=n)
      lhs_regs = iota_tensor(m, k, mlir_dtype_cls.get())
      acc = mgpu.wgmma(init_acc, lhs_regs, rhs_smem, b_order=rhs_order)
      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)
      acc.value.store_untiled(out)

    jax_dtype = jnp.float16 if mlir_dtype_cls == ir.F16Type else jnp.bfloat16
    y_shape = (n, k) if rhs_transpose else (k, n)
    y = self.prng.uniform(-1, 1, y_shape).astype(jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    scratch_shape = jax.ShapeDtypeStruct(
        (k_steps, n // 64, 64, 64), jax_dtype
    )
    z = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), y, out_shape, scratch_shape
    )(y)
    x = np.arange(m * k, dtype=jax_dtype).reshape(m, k)
    ref = jax.lax.dot(
        x, (y.T if rhs_transpose else y), preferred_element_type=jnp.float32
    )
    rtol = 0 if k_steps == 1 else 2.2e-4
    np.testing.assert_allclose(z, ref, rtol=rtol, atol=0)


class BarrierTest(TestCase):

  def test_wg_communication(self):
    i32 = ir.IntegerType.get_signless(32)
    def kernel(ctx, dst, tmp):
      del ctx  # Unused.
      barriers = BarrierArray(3, arrival_count=128)
      gpu.barrier()  # Make sure the barriers are initialized.
      wg_idx = arith.divui(mgpu.warp_idx(), c(4, i32))
      is_first_wg = arith.cmpi(arith.CmpIPredicate.eq, wg_idx, c(0, i32))
      is_second_wg = arith.cmpi(arith.CmpIPredicate.eq, wg_idx, c(1, i32))
      arr = mgpu.FragmentedArray.splat(
          arith.addi(wg_idx, c(1, i32)),
          (128,),
          mgpu.WGStridedFragLayout((128,), 1),
      )
      with ir.InsertionPoint(scf.IfOp(is_first_wg).then_block):
        arr.store_untiled(tmp)
        barriers[0].arrive()  # Signal that tmp is ready.
        barriers[1].wait()  # Wait for the other warp to produce tmp.
        final_arr = arr + mgpu.FragmentedArray.load_strided(tmp)
        final_arr.store_untiled(memref_slice(dst, 0))
        scf.yield_([])
      with ir.InsertionPoint(scf.IfOp(is_second_wg).then_block):
        barriers[0].wait()
        final_arr = arr + mgpu.FragmentedArray.load_strided(tmp)
        barriers[2].arrive()
        barriers[2].wait()  # Synchronize this warpgroup before we overwrite tmp.
        arr.store_untiled(tmp)
        barriers[1].arrive()  # Signal that tmp is ready.
        final_arr.store_untiled(memref_slice(dst, 1))
        scf.yield_([])
    out_shape = jax.ShapeDtypeStruct((2, 128), jnp.int32)
    y = mosaic_gpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (2 * 128, 1, 1),
        (),
        out_shape,
        jax.ShapeDtypeStruct((128,), jnp.int32),
    )()
    np.testing.assert_array_equal(y, np.full_like(y, 3, dtype=np.int32))

class TMATest(TestCase):

  @parameterized.product(
      swizzle=(None, 128),
      shape=((64, 64), (5, 64), (2, 3, 5, 64)),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_load(self, swizzle, shape, dtype):
    if dtype == jnp.float32:
      shape = (*shape[:-1], shape[-1] // 2)
    i1 = ir.IntegerType.get_signless(1)
    def kernel(ctx, src, dst, tmp):
      barrier = BarrierArray(1)[0]
      ctx.async_copy(src_ref=src, dst_ref=tmp, swizzle=swizzle, barrier=barrier)
      barrier.wait_parity(c(0, i1))
      copy(tmp, dst, swizzle=swizzle)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    y = mosaic_gpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(
      swizzle=(None, 128),
      shape=((128, 128), (5, 32, 128)),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_load_tiled(self, swizzle, shape, dtype):
    i1 = ir.IntegerType.get_signless(1)
    index = ir.IndexType.get()
    tiling = (32, 128 // jnp.dtype(dtype).itemsize)
    tiled_shape = tile_shape(shape, tiling)[:len(shape)]
    def kernel(ctx, src, dst, tmp):
      barrier = BarrierArray(1)[0]
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          swizzle=swizzle,
          barrier=barrier,
          gmem_transform=mosaic_gpu.TileTransform(tiling),
      )
      barrier.wait_parity(c(0, i1))
      for idxs in np.ndindex(tiled_shape):
        untiled_idxs, tiled_idxs = idxs[:-len(tiling)], idxs[-len(tiling):]
        s = (
            *untiled_idxs,
            *(ds(c(ix * t, index), t) for ix, t in zip(tiled_idxs, tiling)),
        )
        copy(memref_slice(tmp, idxs), memref_slice(dst, s), swizzle=swizzle)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    smem = jax.ShapeDtypeStruct(tile_shape(shape, tiling), dtype)
    f = mosaic_gpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, smem)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(
      swizzle=(None, 128),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_squeeze_indexing(self, swizzle, dtype):
    shape = (4, 5, 64)
    if dtype == jnp.float32:
      shape = (*shape[:-1], shape[-1] // 2)
    def kernel(ctx, src, dst, tmp):
      barrier = BarrierArray(1)[0]
      for i in range(4):
        ctx.async_copy(
            src_ref=src,
            dst_ref=memref_slice(tmp, i),
            gmem_slice=i,
            swizzle=swizzle,
            barrier=barrier,
        )
        barrier.wait()
      copy(tmp, dst, swizzle=swizzle)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    y = mosaic_gpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)
    np.testing.assert_array_equal(y, x)

  def test_parity_tracking(self):
    shape = (16, 64)
    index = ir.IndexType.get()
    def kernel(ctx, src, dst, tmp):
      barrier = BarrierArray(1)[0]
      for i in range(shape[0]):
        s = ds(c(i, index), 1)
        ctx.async_copy(
            src_ref=src, dst_ref=tmp, gmem_slice=s, barrier=barrier,
        )
        barrier.wait()
        copy(tmp, memref_slice(dst, s))
    x = np.arange(np.prod(shape), dtype=jnp.float16).reshape(shape)
    y = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, x[0:1]
    )(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(
      swizzle=(None, 128),
      shape=((64, 64), (5, 64), (2, 3, 5, 64)),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_store(self, swizzle, shape, dtype):
    if dtype == jnp.float32:
      shape = (*shape[:-1], shape[-1] // 2)
    def kernel(ctx, src, dst, tmp):
      copy(src, tmp, swizzle=swizzle)
      ctx.async_copy(src_ref=tmp, dst_ref=dst, swizzle=swizzle)
      ctx.await_async_copy(0)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    y = mosaic_gpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)
    np.testing.assert_array_equal(y, x)


class FragmentedArrayTest(TestCase):

  @parameterized.product(
      op=(
          operator.add,
          operator.mul,
          operator.sub,
          operator.truediv,
          (lambda x, y: mgpu.FragmentedArray.max(x, y), np.maximum),
      ),
      m=(64, 128),
      n=(8, 16, 32, 64, 80, 128, 256),
  )
  def test_binary(self, op, m=64, n=32):
    if isinstance(op, tuple):
      op, np_op = op
    else:
      np_op = op

    for scalar_rhs in [None, 2]:
      def kernel(ctx, dst, _):
        f32 = ir.F32Type.get()
        iota = iota_tensor(m=m, n=n, mlir_dtype=f32)
        rhs = iota if scalar_rhs is None else c(scalar_rhs, iota.mlir_dtype)
        op(iota, rhs).store_untiled(dst)
      out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
      result = mosaic_gpu.as_gpu_kernel(
          kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
      )()
      ref_x = np.arange(m * n, dtype=jnp.float32).reshape(m, n)
      ref_rhs = scalar_rhs or ref_x
      if op == operator.truediv:
        np.testing.assert_allclose(result, np_op(ref_x, ref_rhs), atol=2e-7)
      else:
        np.testing.assert_array_equal(result, np_op(ref_x, ref_rhs))

  @parameterized.product(
      ops=((lambda x: mgpu.FragmentedArray.exp(x), np.exp),),
      m=(64, 128),
      n=(8, 16, 32, 64, 80, 128, 256),
  )
  def test_unary(self, ops, m=64, n=32):
    op, np_op = ops
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      iota = iota_tensor(m=m, n=n, mlir_dtype=f32)
      op(iota).store_untiled(dst)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=jnp.float32).reshape(m, n)
    np.testing.assert_allclose(result, np_op(x), atol=2e-7, rtol=2e-7)

  @parameterized.product(
      op=(arith.addf, arith.maximumf),
      m=(64, 128),
      n=(8, 16, 32, 64, 80, 128, 256),
  )
  def test_reduce(self, op, m=64, n=32):
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      iota = iota_tensor(m=m, n=n, mlir_dtype=f32)
      iota.reduce(op, axis=1).broadcast_minor(n).store_untiled(dst)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=jnp.float32).reshape(m, n)
    if op == arith.addf:
      expected = np.broadcast_to(x.sum(axis=1, keepdims=True), x.shape)
    elif op == arith.maximumf:
      expected = np.broadcast_to(x.max(axis=1, keepdims=True), x.shape)
    else:
      raise NotImplementedError(f"Unsupported op: {op}")
    np.testing.assert_array_equal(result, expected)

  def test_splat_layout(self):
    m, n = 64, 8
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      iota = iota_tensor(m=m, n=n, mlir_dtype=f32)
      cte = c(1, iota.mlir_dtype)
      cte_arr = mgpu.FragmentedArray.splat(cte, ())
      cte_arr = cte_arr.reshape((1, 1)).broadcast((m, n))
      (iota + cte_arr).store_untiled(dst)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    expected = np.arange(m * n, dtype=jnp.float32).reshape(m, n) + 1
    np.testing.assert_array_equal(result, expected)

  def test_splat(self):
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      v = arith.constant(f32, ir.FloatAttr.get(f32, 3.14))
      t = mgpu.FragmentedArray.splat(v, (128,), mgpu.WGMMA_ROW_LAYOUT)
      t.broadcast_minor(32).store_untiled(dst)
    out_shape = jax.ShapeDtypeStruct((128, 32), jnp.float32)
    result = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    np.testing.assert_array_equal(result, np.full((128, 32), 3.14, np.float32))

  @parameterized.product(in_shape=((128, 128), (128, 64), (64, 128)))
  def test_strided_load_store(self, in_shape):
    def kernel(ctx, *args):
      gmem_input, gmem_output, (smem_input, smem_output) = args
      copy(gmem_input, smem_input)
      t = mgpu.FragmentedArray.load_strided(smem_input)
      t.store_untiled(smem_output)
      copy(smem_output, gmem_output)

    inp = out = self.prng.uniform(-1, 1, in_shape).astype(jnp.float32)
    result = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (inp,), out, [inp, out],
    )(inp)
    np.testing.assert_array_equal(inp, result)


  def test_warp_tree_reduce(self):
    def kernel(ctx, out, *_):
      del ctx
      i32 = ir.IntegerType.get_signless(32)
      tid = gpu.thread_id(gpu.Dimension.x)
      value = arith.index_cast(i32, tid)
      grp = warp_tree_reduce(value, arith.addi, 4)
      memref.store(grp, out, [tid])

    x = np.arange(128, dtype=jnp.int32)
    result = mosaic_gpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), x, [],
    )()
    for i in range(0, 128, 4):
      x[i:i + 4] = jnp.sum(x[i:i + 4])

    np.testing.assert_array_equal(result, x)



class ProfilerTest(TestCase):

  def test_measure(self):
    x = jnp.arange(1024 * 1024)
    profiler.measure(lambda x, y: x + y, x, x)  # This is just a smoke test


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
