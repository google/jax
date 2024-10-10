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

import enum
import itertools
import math
import operator
import unittest

from absl.testing import absltest, parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import jax.numpy as jnp
import numpy as np
try:
  import jax._src.lib.mosaic_gpu  # noqa: F401
  HAS_MOSAIC_GPU = True
except ImportError:
  HAS_MOSAIC_GPU = False

  class Dimension(enum.IntEnum):  # Just to make parameterized tests expand ok
    x = 0
    y = 1
    z = 2
else:
  import jax.experimental.mosaic.gpu as mgpu
  from jax.experimental.mosaic.gpu import utils as utils
  from jax.experimental.mosaic.gpu import profiler
  from jax.experimental.mosaic.gpu.utils import *  # noqa: F403
  from jax._src.lib.mlir.dialects import gpu
  from jax._src.lib.mlir.dialects import llvm
  Dimension = gpu.Dimension


# ruff: noqa: F405
# pylint: disable=g-complex-comprehension
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


def copy(src: ir.Value, dst: ir.Value, swizzle: int | None = None):
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
        assert swizzle.bit_count() == 1
        bytes_per_element = bytewidth(src_ty.element_type)
        linear_idx = c(0, index)
        for stride, i in zip(dyn_strides, idx):
          linear_idx = arith.addi(linear_idx, arith.muli(i, stride))
        # Swizzle pattern repeats every 128 bytes.
        swizzle_src = arith.remui(
            arith.divui(linear_idx, c(128 // bytes_per_element, index)),
            c(swizzle // 16, index),
        )
        # Swizzle happens in groups of 16 bytes.
        swizzle_shift = 4 - (bytes_per_element.bit_length() - 1)
        dst_linear_idx = arith.xori(
            linear_idx, arith.shli(swizzle_src, c(swizzle_shift, index))
        )
        dst_idx = [
            arith.remui(arith.divui(dst_linear_idx, stride), c(bound, index))
            for stride, bound in zip(dyn_strides, shape)
        ]
      memref.store(memref.load(src, idx), dst, dst_idx)
    nd_loop([c(d, index) for d in shape], body)
    scf.yield_([])
  gpu.barrier()
  nvvm.fence_proxy(nvvm.ProxyKind.async_)


def iota_tensor(m, n, dtype: jax.typing.DTypeLike):
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
  t = mgpu.FragmentedArray(
      _registers=registers, _layout=mgpu.WGMMA_LAYOUT, _is_signed=True
  )
  return t.astype(
      utils.dtype_to_ir_type(dtype), is_signed=utils.is_signed(dtype)
  )


class TestCase(parameterized.TestCase):

  def setUp(self):
    if not HAS_MOSAIC_GPU:
      self.skipTest("jaxlib built without Mosaic GPU")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")
    super().setUp()
    self.prng = np.random.default_rng(1234)
    self.enter_context(jtu.global_config_context(jax_traceback_filtering="off"))
    self.enter_context(mlir.make_ir_context())
    self.enter_context(ir.Location.unknown())


class TestUtilTest(TestCase):

  def test_copy_basic(self):
    def kernel(ctx, src, dst, _):
      copy(src, dst)
    x = jnp.arange(2 * 3 * 5).reshape(2, 5, 3)
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, ())(x)
    np.testing.assert_array_equal(y, x)

  def test_copy_swizzle(self):
    def kernel(ctx, src, dst, _):
      copy(src, dst, swizzle=128)
    x = jnp.arange(8 * 32, dtype=jnp.float32).reshape(8, 32)
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, ())(x)
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
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)
    np.testing.assert_array_equal(y, x)

  def test_iota_tensor(self):
    m = n = 64
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      index = ir.IndexType.get()
      registers = iota_tensor(m, n, jnp.float32).registers
      assert registers.size == 16, registers.size
      for i, vec_reg in enumerate(registers.flat):
        for j in range(2):
          reg = vector.extractelement(vec_reg, position=c(j, index))
          memref.store(
              reg, dst, [gpu.thread_id(gpu.Dimension.x), c(2 * i + j, index)]
          )
    out_shape = jax.ShapeDtypeStruct((128, 32), jnp.float32)
    regs = mgpu.as_gpu_kernel(
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
    y = mgpu.as_gpu_kernel(
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
    y = mgpu.as_gpu_kernel(
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
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(out_ty.shape))

  @parameterized.named_parameters(
      ("drop_1s", (1, 1, 5, 1, 1, 2, 1, 1), (5, 1, 2)),
      ("add_1s", (5, 1, 2), (1, 1, 5, 1, 1, 2, 1, 1)),
      ("fold", (1, 5, 2, 1,), (1, 10, 1)),
      ("un", (1, 10, 1), (1, 5, 2, 1,)),
  )
  def test_reshape(self, inp_shape, out_shape):
    def kernel(ctx, inp, out, _):
      copy(memref_reshape(inp, out_shape), out)

    x = np.arange(math.prod(inp_shape), dtype=jnp.float32).reshape(*inp_shape)
    out_ty = jax.ShapeDtypeStruct(out_shape, jnp.float32)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, out_ty, ()
    )(x)
    np.testing.assert_array_equal(y, x.reshape(*out_shape))

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
    index = tuple(slice(0, s) for s in shape)

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
      y = mgpu.as_gpu_kernel(
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

  @parameterized.named_parameters(("f32", jnp.float32), ("f16", jnp.float16))
  def test_store_untiled(self, dtype):
    def kernel(ctx, out, _):
      del ctx
      iota_tensor(64, 64, dtype).store_untiled(out)
    expected = np.arange(64 * 64, dtype=dtype).reshape(64, 64)
    iota = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, ()
    )()
    np.testing.assert_array_equal(iota, expected)

  @parameterized.named_parameters(
      ("f32", jnp.float32, 256),
      ("f16", jnp.float16, 256),
      ("f16_small", jnp.float16, 128),
  )
  def test_store_untiled_splat(self, jax_dtype, size):
    mlir_dtype = utils.dtype_to_ir_type(jax_dtype)
    def kernel(ctx, out, _):
      del ctx
      arr = mgpu.FragmentedArray.splat(
          c(1.0, mlir_dtype), (size,), is_signed=utils.is_signed(jax_dtype)
      )
      arr.store_untiled(out)
    expected = np.ones((size,), jax_dtype)
    mosaic_ones = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, ()
    )()
    np.testing.assert_array_equal(mosaic_ones, expected)

  @parameterized.product(
      dtype=[jnp.float32, jnp.float16, jnp.int8],
      swizzle=(32, 64, 128),
      num_col_tiles=(1, 2, 3),
  )
  def test_store_tiled(self, dtype, swizzle, num_col_tiles):
    mlir_dtype = utils.dtype_to_ir_type(dtype)
    if bytewidth(mlir_dtype) > 2 and swizzle == 32:
      self.skipTest("Not implemented")
    col_tiling = swizzle // bytewidth(mlir_dtype)
    m = 128
    n = col_tiling * num_col_tiles
    tiling = (64, col_tiling)
    def kernel(ctx, out, smem):
      del ctx
      iota_tensor(m, n, dtype).store_tiled(smem, swizzle=swizzle)
      copy(smem, out, swizzle=swizzle)
    expected = (
        np.arange(m * n, dtype=dtype)
        .reshape(m // tiling[0], tiling[0], n // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )
    iota = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, expected
    )()
    np.testing.assert_array_equal(iota, expected)

  @parameterized.product(
      dtype=[jnp.float16, jnp.int8],
      swizzle=(32, 64, 128),
  )
  def test_store_tiled_short_n(self, dtype, swizzle):
    mlir_dtype = utils.dtype_to_ir_type(dtype)
    col_tiling = swizzle // bytewidth(mlir_dtype)
    m = 128
    n = 16 // bytewidth(mlir_dtype)
    tiling = (64, col_tiling)
    def kernel(ctx, out, smem):
      iota_tensor(m, n, dtype).store_tiled(smem, swizzle=swizzle)
      ctx.async_copy(
          src_ref=smem,
          dst_ref=out,
          swizzle=swizzle,
          gmem_slice=(ds(0, m), ds(0, col_tiling)),
          gmem_transform=mgpu.TileTransform(tiling),
      )
      ctx.await_async_copy(0)
    smem_shape = jax.ShapeDtypeStruct((m // tiling[0], 1, *tiling), dtype)
    expected = np.arange(m * n, dtype=dtype).reshape(m, n)
    iota = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), expected, smem_shape
    )()
    np.testing.assert_array_equal(iota, expected)

  @parameterized.named_parameters(
      ("drop_1s", (1, 1, 5, 1, 1, 2, 1, 1, 16), (5, 1, 2, 16)),
      ("add_1s", (5, 1, 2, 16), (5, 1, 1, 2, 16)),
      ("fold", (1, 5, 2, 1, 16), (1, 10, 1, 16)),
      ("un", (1, 10, 1, 16), (1, 5, 2, 1, 16)),
  )
  def test_reshape_transform(self, in_shape, out_shape):
    def kernel(ctx, inp, out, smem):
      smem_ref, barrier = smem
      ctx.async_copy(
          src_ref=inp,
          dst_ref=smem_ref,
          gmem_transform=mgpu.ReshapeTransform(out_shape),
          barrier=barrier,
      )
      barrier.wait()
      copy(smem_ref, out)

    inp = self.prng.uniform(-1, 1, in_shape).astype(jnp.float32)
    out = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        (jax.ShapeDtypeStruct(in_shape, jnp.float32),),
        jax.ShapeDtypeStruct(out_shape, jnp.float32),
        (jax.ShapeDtypeStruct(out_shape, jnp.float32), mgpu.TMABarrier()),
    )(inp)
    np.testing.assert_array_equal(out, inp.reshape(out_shape))

  @parameterized.named_parameters(
      ("bf16_i8", jnp.bfloat16, jnp.int8),
      ("i8_bf16", jnp.int8, jnp.bfloat16),
      ("i8_i8", jnp.int8, jnp.int8),
  )
  def test_convert_tiled(self, jax_dtype_from, jax_dtype_to):
    mlir_dtype_from = utils.dtype_to_ir_type(jax_dtype_from)
    mlir_dtype_to = utils.dtype_to_ir_type(jax_dtype_to)
    m = 128
    n = 256 // bytewidth(mlir_dtype_from)
    def kernel(ctx, inp, out, smem):
      del ctx
      smem_from, smem_to = smem
      copy(inp, smem_from, swizzle=128)
      t = mgpu.FragmentedArray.load_tiled(
          smem_from, swizzle=128, is_signed=utils.is_signed(jax_dtype_from)
      )
      t = t.astype(mlir_dtype_to, is_signed=utils.is_signed(jax_dtype_to))
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
    res = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        expected_from,
        expected_to,
        (expected_from, expected_to),
    )(expected_from)
    np.testing.assert_array_equal(res, expected_to)

  @parameterized.named_parameters(
      ("f32", jnp.float32),
      ("f16", jnp.float16),
      ("i8", jnp.int8),
  )
  def test_load_tiled(self, jax_dtype):
    mlir_dtype = utils.dtype_to_ir_type(jax_dtype)
    m = 128
    n = 256 // bytewidth(mlir_dtype)
    tiling = (64, 128 // bytewidth(mlir_dtype))
    def kernel(ctx, in_, out, smem):
      del ctx
      smem1, smem2 = smem
      copy(in_, smem1, swizzle=128)
      t = mgpu.FragmentedArray.load_tiled(
          smem1, swizzle=128, is_signed=utils.is_signed(jax_dtype)
      )
      t.store_tiled(smem2, swizzle=128)
      copy(smem2, out, swizzle=128)
    expected = (
        np.arange(m * n, dtype=jax_dtype)
        .reshape(m // tiling[0], tiling[0], n // tiling[1], tiling[1])
        .transpose(0, 2, 1, 3)
    )
    iota = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), expected, expected, (expected,) * 2
    )(expected)
    np.testing.assert_array_equal(iota, expected)

  @parameterized.product(
      lhs_transpose=(False, True),
      rhs_transpose=(False, True),
      in_mlir_dtype_cls=(ir.F16Type, ir.BF16Type, ir.F32Type),
      m=(64, 128, 192),
      n=(64, 128, 192),
      k_steps=(1, 2),
      tma_inputs=(False, True),
      swizzle=(32, 64, 128),
      jax_out_dtype=(jnp.float16, jnp.float32),
  )
  def test_wgmma_basic(
      self,
      m,
      n,
      k_steps,
      in_mlir_dtype_cls,
      lhs_transpose,
      rhs_transpose,
      tma_inputs,
      swizzle,
      jax_out_dtype,
  ):
    if jax_out_dtype == jnp.float16 and in_mlir_dtype_cls is not ir.F16Type:
      raise self.skipTest("Only f16 input is supported for f16 output.")
    if swizzle != 128 and lhs_transpose:
      raise self.skipTest("Transpose only supported in 128B swizzled WGMMA")
    if swizzle != 128 and not tma_inputs:
      raise self.skipTest("Copy with non-128B swizzles not implemented")

    in_mlir_dtype = in_mlir_dtype_cls.get()
    out_mlir_dtype = utils.dtype_to_ir_type(jax_out_dtype)
    if ir.F32Type.isinstance(in_mlir_dtype):  # We actually use tf32 instead
      in_jax_dtype = jnp.float32
      if lhs_transpose or not rhs_transpose:
        self.skipTest("Transpose only supported in 16-bit WGMMA")
      exponent_bits, mantissa_bits = 8, 10  # Use tf32
    elif bytewidth(in_mlir_dtype) == 2:
      if n % 64 != 0:
        self.skipTest("16-bit WGMMA only supports n % 64 == 0")
      if ir.F16Type.isinstance(in_mlir_dtype):
        in_jax_dtype = jnp.float16
        exponent_bits, mantissa_bits = 5, 10
      elif ir.BF16Type.isinstance(in_mlir_dtype):
        in_jax_dtype = jnp.bfloat16
        exponent_bits, mantissa_bits = 8, 7
      else:
        raise NotImplementedError(in_mlir_dtype)
    else:
      raise NotImplementedError(in_mlir_dtype)
    nk_tile = swizzle // bytewidth(in_mlir_dtype)
    k = nk_tile * k_steps
    assert m % 64 == 0 and n % nk_tile == 0
    index = ir.IndexType.get()

    row_major = mgpu.WGMMALayout.ROW_MAJOR
    col_major = mgpu.WGMMALayout.COL_MAJOR
    lhs_order = col_major if lhs_transpose else row_major
    rhs_order = col_major if rhs_transpose else row_major

    def kernel(ctx, lhs, rhs, out, scratch):
      lhs_smem, rhs_smem, barriers = scratch
      if tma_inputs:
        lhs_transform = (mgpu.TileTransform((64, nk_tile)),)
        if lhs_transpose:
          assert nk_tile == 64  # Make sure we didn't have to transpose tiling.
          lhs_transform += (mgpu.TransposeTransform((1, 0, 2, 3)),)
        rhs_transform = (mgpu.TileTransform((nk_tile, nk_tile)),)
        if rhs_transpose:
          rhs_transform += (mgpu.TransposeTransform((1, 0, 2, 3)),)
        ctx.async_copy(
            src_ref=lhs,
            dst_ref=lhs_smem,
            swizzle=swizzle,
            gmem_transform=lhs_transform,
            barrier=barriers[0],
        )
        ctx.async_copy(
            src_ref=rhs,
            dst_ref=rhs_smem,
            swizzle=swizzle,
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
                swizzle=swizzle,
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
                swizzle=swizzle,
            )
      init_acc = mgpu.WGMMAAccumulator.zero(m=m, n=n, dtype=out_mlir_dtype)
      acc = mgpu.wgmma(
          init_acc, lhs_smem, rhs_smem,
          a_order=lhs_order, b_order=rhs_order, swizzle=swizzle,
      )
      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)
      acc.value.store_untiled(out)

    def quantize(x):
      # Quantize the input to avoid rounding when feeding the WGMMA
      return jax.lax.reduce_precision(x, exponent_bits, mantissa_bits)

    x_shape = (k, m) if lhs_transpose else (m, k)
    x = quantize(self.prng.uniform(-1, 1, x_shape)).astype(in_jax_dtype)
    y_shape = (n, k) if rhs_transpose else (k, n)
    y = quantize(self.prng.uniform(-1, 1, y_shape)).astype(in_jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), jax_out_dtype)
    scratch_shape = [
        jax.ShapeDtypeStruct((m // 64, k // nk_tile, 64, nk_tile), in_jax_dtype),
        jax.ShapeDtypeStruct(
            (k // nk_tile, n // nk_tile, nk_tile, nk_tile), in_jax_dtype
        ),
        mgpu.TMABarrier(2),
    ]
    z = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (x, y), out_shape, scratch_shape
    )(x, y)
    x32, y32 = x.astype(np.float32), y.astype(np.float32)
    ref = (x32.T if lhs_transpose else x32) @ (y32.T if rhs_transpose else y32)
    atol = 2e-2 if jax_out_dtype == jnp.float16 else 5e-6
    np.testing.assert_allclose(z, ref, atol=atol)

  # TODO(apaszke): Add support for f32
  @parameterized.product(
      m=(64, 128, 192),
      n=(64, 128, 192),
      k_steps=(1, 2),
      rhs_transpose=(False, True),
      swizzle=(32, 64, 128),
      dtype=[jnp.float16, jnp.bfloat16],
  )
  def test_wgmma_reg_lhs(self, m, n, k_steps, rhs_transpose, swizzle, dtype):
    index = ir.IndexType.get()

    row_major = mgpu.WGMMALayout.ROW_MAJOR
    col_major = mgpu.WGMMALayout.COL_MAJOR
    rhs_order = col_major if rhs_transpose else row_major
    bytewidth = 2
    nk_tile = swizzle // bytewidth
    k = nk_tile * k_steps

    def kernel(ctx, rhs, out, rhs_smem):
      del ctx
      for ki in range(k_steps):
        for ni in range(n // nk_tile):
          rhs_slice = (
              ds(c(ki * nk_tile, index), nk_tile),
              ds(c(ni * nk_tile, index), nk_tile),
          )
          if rhs_transpose:
            rhs_slice = rhs_slice[::-1]
          copy(
              src=memref_slice(rhs, rhs_slice),
              dst=memref_slice(rhs_smem, (ki, ni)),
              swizzle=swizzle,
          )
      init_acc = mgpu.WGMMAAccumulator.zero(m=m, n=n)
      lhs_regs = iota_tensor(m, k, dtype)
      acc = mgpu.wgmma(init_acc, lhs_regs, rhs_smem, b_order=rhs_order, swizzle=swizzle)
      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)
      acc.value.store_untiled(out)

    y_shape = (n, k) if rhs_transpose else (k, n)
    y = self.prng.uniform(-1, 1, y_shape).astype(dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    scratch_shape = jax.ShapeDtypeStruct(
        (k_steps, n // nk_tile, nk_tile, nk_tile), dtype
    )
    z = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), y, out_shape, scratch_shape
    )(y)
    x = np.arange(m * k, dtype=dtype).reshape(m, k)
    ref = jax.lax.dot(
        x, (y.T if rhs_transpose else y), preferred_element_type=jnp.float32
    )
    rtol = 5e-4
    np.testing.assert_allclose(z, ref, rtol=rtol, atol=0)

  @parameterized.product(
      rhs_transpose=(False, True),
      swizzle=(32, 64, 128),
  )
  def test_narrow_n(self, rhs_transpose, swizzle):
    m, n, k_steps = 64, 8, 2

    row_major = mgpu.WGMMALayout.ROW_MAJOR
    col_major = mgpu.WGMMALayout.COL_MAJOR
    rhs_order = col_major if rhs_transpose else row_major
    bytewidth = 2
    nk_tile = swizzle // bytewidth
    k = nk_tile * k_steps

    def kernel(ctx, rhs, out, smem):
      rhs_smem, barrier = smem
      gmem_slice = (ds(0, k), ds(0, nk_tile))
      smem_slice = (slice(None), slice(None), slice(None), ds(0, n))
      transform = (mgpu.TileTransform((nk_tile, nk_tile)),)
      if rhs_transpose:
        gmem_slice = gmem_slice[::-1]
        smem_slice = (slice(None), slice(None), ds(0, n), slice(None))
        transform += (mgpu.TransposeTransform((1, 0, 2, 3)),)
      ctx.async_copy(
          src_ref=rhs,
          dst_ref=rhs_smem,
          swizzle=swizzle,
          gmem_slice=gmem_slice,
          gmem_transform=transform,
          barrier=barrier,
      )
      barrier.wait()
      init_acc = mgpu.WGMMAAccumulator.zero(m=m, n=n)
      lhs_regs = iota_tensor(m, k, jnp.float16)
      rhs_smem = memref_slice(rhs_smem, smem_slice)
      acc = mgpu.wgmma(init_acc, lhs_regs, rhs_smem, b_order=rhs_order, swizzle=swizzle)
      nvvm.wgmma_commit_group_sync_aligned()
      nvvm.wgmma_wait_group_sync_aligned(0)
      acc.value.store_untiled(out)

    jax_dtype = jnp.float16
    y_shape = (n, k) if rhs_transpose else (k, n)
    y = self.prng.uniform(-1, 1, y_shape).astype(jax_dtype)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    rhs_scratch_shape = jax.ShapeDtypeStruct(
        (k_steps, 1, nk_tile, nk_tile), jax_dtype
    )
    z = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), y, out_shape, (rhs_scratch_shape, mgpu.TMABarrier()),
    )(y)
    x = np.arange(m * k, dtype=jax_dtype).reshape(m, k)
    ref = jax.lax.dot(
        x, (y.T if rhs_transpose else y), preferred_element_type=jnp.float32
    )
    np.testing.assert_allclose(z, ref, rtol=5e-4, atol=0)


class BarrierTest(TestCase):

  def test_wg_communication(self):
    i32 = ir.IntegerType.get_signless(32)
    def kernel(ctx, dst, scratch):
      tmp, barriers = scratch
      del ctx  # Unused.
      wg_idx = arith.divui(mgpu.warp_idx(), c(4, i32))
      is_first_wg = arith.cmpi(arith.CmpIPredicate.eq, wg_idx, c(0, i32))
      is_second_wg = arith.cmpi(arith.CmpIPredicate.eq, wg_idx, c(1, i32))
      arr = mgpu.FragmentedArray.splat(
          arith.addi(wg_idx, c(1, i32)),
          (128,),
          mgpu.WGStridedFragLayout((128,), 1),
          is_signed=False,
      )
      with ir.InsertionPoint(scf.IfOp(is_first_wg).then_block):
        arr.store_untiled(tmp)
        barriers[0].arrive()  # Signal that tmp is ready.
        barriers[1].wait()  # Wait for the other warp to produce tmp.
        final_arr = arr + mgpu.FragmentedArray.load_strided(
            tmp, is_signed=False
        )
        final_arr.store_untiled(memref_slice(dst, 0))
        scf.yield_([])
      with ir.InsertionPoint(scf.IfOp(is_second_wg).then_block):
        barriers[0].wait()
        final_arr = arr + mgpu.FragmentedArray.load_strided(
            tmp, is_signed=False
        )
        barriers[2].arrive()
        barriers[2].wait()  # Synchronize this warpgroup before we overwrite tmp.
        arr.store_untiled(tmp)
        barriers[1].arrive()  # Signal that tmp is ready.
        final_arr.store_untiled(memref_slice(dst, 1))
        scf.yield_([])
    out_shape = jax.ShapeDtypeStruct((2, 128), jnp.int32)
    y = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (2 * 128, 1, 1),
        (),
        out_shape,
        (
            jax.ShapeDtypeStruct((128,), jnp.int32),
            mgpu.Barrier(arrival_count=128, num_barriers=3),
        ),
    )()
    np.testing.assert_array_equal(y, np.full_like(y, 3, dtype=np.int32))

  @parameterized.named_parameters(
      (
          f"_{''.join(map(str, collective_dims))}={collective_size}{'_' + ''.join(map(str, noncollective_dims)) if noncollective_dims else ''}{'_group' if group_dims else ''}",
          collective_dims,
          noncollective_dims,
          collective_size,
          group_dims,
      )
      for collective_dims in itertools.chain.from_iterable(
          itertools.combinations(Dimension, n) for n in range(1, 4)
      )
      for noncollective_dims in itertools.chain.from_iterable(
          itertools.combinations(Dimension, n) for n in range(3)
      )
      for collective_size in (1, 2, 4)
      for group_dims in (False,) + ((True,) if len(collective_dims) > 1 else ())
      if all(d not in noncollective_dims for d in collective_dims)
  )
  def test_collective_arrive(self, collective_dims, noncollective_dims, collective_size, group_dims):
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    cluster = [1, 1, 1]
    for d in collective_dims:
      cluster[d] = collective_size
    for d in noncollective_dims:
      cluster[d] = 2
    if math.prod(cluster) > 16:
      self.skipTest("Cluster too big")
    is_trivial = math.prod(cluster[d] for d in collective_dims) == 1
    def kernel(ctx, dst, mask, collective_barrier):
      memref.store(arith.constant(i32, 1 << 17), mask, [c(0, index)])
      gpu.barrier()
      collective_barrier.arrive()
      collective_barrier.wait()
      if not is_trivial:
        llvm.atomicrmw(
            llvm.AtomicBinOp.min,
            utils.memref_ptr(mask),
            collective_barrier.cluster_mask,
            llvm.AtomicOrdering.monotonic,
        )
      else:
        assert collective_barrier.cluster_mask is None
      tid = thread_idx()
      linear_idx = arith.index_cast(index, tid)
      stride = c(128, index)
      for d in gpu.Dimension:
        linear_idx = arith.addi(linear_idx, arith.muli(gpu.block_id(d), stride))
        stride = arith.muli(stride, gpu.grid_dim(d))
      memref.store(arith.index_cast(i32, linear_idx), dst, [linear_idx])
    out_shape = jax.ShapeDtypeStruct((math.prod(cluster) * 128,), jnp.int32)
    mask_shape = jax.ShapeDtypeStruct((1,), jnp.int32)
    barrier_dims = collective_dims
    if group_dims:
      barrier_dims = (collective_dims[:2], *collective_dims[2:])
    scratch = mgpu.ClusterBarrier(barrier_dims)
    y, mask = mgpu.as_gpu_kernel(
        kernel, cluster, (128, 1, 1), (), (out_shape, mask_shape), scratch, cluster=cluster,
    )()
    np.testing.assert_array_equal(
        y, np.arange(math.prod(cluster) * 128, dtype=np.int32)
    )
    if not is_trivial:
      # Verify that the mask is correct. Blocks are column-major, hence the transpose.
      block_bits = 1 << np.arange(math.prod(cluster), dtype=np.int32).reshape(cluster[::-1]).T
      expected_mask = 0
      for bd in barrier_dims:
        if isinstance(bd, gpu.Dimension):
          bd = (bd,)
        least_significant_slice = tuple(
            slice(None) if d in bd else 0 for d in gpu.Dimension
        )
        mask_bits = block_bits[least_significant_slice]
        expected_mask |= np.bitwise_or.reduce(mask_bits, axis=None)
      self.assertEqual(mask, expected_mask)


class TMATest(TestCase):

  @parameterized.product(
      swizzle=(None, 32, 64, 128),
      shape=((64, None), (5, None), (2, 3, 5, None)),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_load_basic(self, swizzle, shape, dtype):
    minor_size = 64 if swizzle is None else swizzle // jnp.dtype(dtype).itemsize
    shape = (*shape[:-1], minor_size)
    i1 = ir.IntegerType.get_signless(1)
    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
      ctx.async_copy(src_ref=src, dst_ref=tmp, swizzle=swizzle, barrier=barrier)
      barrier.wait_parity(c(0, i1))
      copy(tmp, dst, swizzle=swizzle)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    smem = (x, mgpu.TMABarrier())
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, smem)(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.named_parameters(
      (
          f"_{''.join(map(str, collective_dims))}={collective_size}{'_' + ''.join(map(str, noncollective_dims)) if noncollective_dims else ''}",
          collective_dims,
          noncollective_dims,
          collective_size,
      )
      for collective_dims in itertools.chain.from_iterable(
          itertools.combinations(Dimension, n) for n in range(1, 4)
      )
      for noncollective_dims in itertools.chain.from_iterable(
          itertools.combinations(Dimension, n) for n in range(3)
      )
      for collective_size in (1, 2, 4)
      if all(d not in noncollective_dims for d in collective_dims)
  )
  def test_tma_load_multicast(self, collective_dims, noncollective_dims, collective_dim_size):
    index = ir.IndexType.get()
    swizzle = 128
    dtype = jnp.float16
    cluster = [1, 1, 1]
    for d in collective_dims:
      cluster[d] = collective_dim_size
    for d in noncollective_dims:
      cluster[d] = 2
    if math.prod(cluster) > 16:
      self.skipTest("Cluster too big")
    collective_size = math.prod(cluster[d] for d in collective_dims)
    noncollective_size = math.prod(cluster) // collective_size
    # We use the 2 dimension to exercise splitting the collective over
    # multiple dimensions when the cluster is large.
    shape = (noncollective_size, 2, 16 * collective_size, 64)
    minor_size = 64 if swizzle is None else swizzle // jnp.dtype(dtype).itemsize
    shape = (*shape[:-1], minor_size)
    # Note that this kernel does not use the non-collective dimensions in any
    # interesting way and so they don't really have to be part of the cluster.
    # We use them to test that the multicast mask is generated correctly.
    def kernel(ctx, src, dst, scratch):
      tmp, barrier = scratch
      stride = 1
      noncollective_idx = c(0, index)
      for d in noncollective_dims:
        noncollective_idx = arith.addi(
            noncollective_idx,
            arith.muli(gpu.cluster_block_id(d), c(stride, index))
        )
        stride *= cluster[d]
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          gmem_slice=(noncollective_idx,),
          swizzle=swizzle,
          barrier=barrier,
          collective=collective_dims,
      )
      barrier.wait()
      # This is _not_ the real cluster block idx, because it does not consider
      # the column-major ordering of the grid dimensions.
      idx = c(0, index)
      stride = 1
      for d in collective_dims:
        idx = arith.addi(
            idx, arith.muli(gpu.cluster_block_id(d), c(stride, index))
        )
        stride *= cluster[d]
      slc = ds(
          arith.muli(idx, c(16, index)), 16
      )
      copy(
          memref_slice(tmp, (slice(None), slc)),
          memref_slice(dst, (noncollective_idx, slice(None), slc)),
          swizzle=swizzle,
      )
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    smem_shape = (jax.ShapeDtypeStruct(shape[1:], dtype), mgpu.TMABarrier())
    y = mgpu.as_gpu_kernel(
        kernel, cluster, (128, 1, 1), x, x, smem_shape, cluster=cluster
    )(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(
      swizzle=(None, 128),
      shape=((128, 128), (5, 32, 128)),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_load_tiled(self, swizzle, shape, dtype):
    # TODO(apaszke): ptxas seems to freeze when generating code for copy with
    # swizzle 32 and 64.
    i1 = ir.IntegerType.get_signless(1)
    index = ir.IndexType.get()
    tiling = (32, (swizzle or 128) // jnp.dtype(dtype).itemsize)
    tiled_shape = tile_shape(shape, tiling)[:len(shape)]
    def kernel(ctx, src, dst, scratch):
      tmp, barrier = scratch
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          swizzle=swizzle,
          barrier=barrier,
          gmem_transform=mgpu.TileTransform(tiling),
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
    smem = (
        jax.ShapeDtypeStruct(tile_shape(shape, tiling), dtype),
        mgpu.TMABarrier(),
    )
    f = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, smem)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(
      swizzle=(None, 128),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_squeeze_indexing(self, swizzle, dtype):
    # TODO(apaszke): ptxas seems to freeze when generating code for copy with
    # swizzle 32 and 64.
    minor_size = 64 if swizzle is None else swizzle // jnp.dtype(dtype).itemsize
    shape = (4, 5, minor_size)
    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
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
    smem = (x, mgpu.TMABarrier())
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, smem)(x)
    np.testing.assert_array_equal(y, x)

  def test_parity_tracking(self):
    shape = (16, 64)
    index = ir.IndexType.get()
    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
      for i in range(shape[0]):
        s = ds(c(i, index), 1)
        ctx.async_copy(
            src_ref=src, dst_ref=tmp, gmem_slice=s, barrier=barrier,
        )
        barrier.wait()
        copy(tmp, memref_slice(dst, s))
    x = np.arange(np.prod(shape), dtype=jnp.float16).reshape(shape)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, (x[0:1], mgpu.TMABarrier())
    )(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.product(
      swizzle=(None, 32, 64, 128),
      shape=((64, None), (5, None), (2, 3, 5, None)),
      dtype=(jnp.float16, jnp.float32),
  )
  def test_tma_store(self, swizzle, shape, dtype):
    minor_size = 64 if swizzle is None else swizzle // jnp.dtype(dtype).itemsize
    shape = (*shape[:-1], minor_size)
    def kernel(ctx, src, dst, tmp):
      copy(src, tmp, swizzle=swizzle)
      ctx.async_copy(src_ref=tmp, dst_ref=dst, swizzle=swizzle)
      ctx.await_async_copy(0)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    y = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)
    np.testing.assert_array_equal(y, x)

  @parameterized.parameters(0, 1)
  def test_tma_small_tile_load(self, small_dim):
    if small_dim == 0:
      shape = (4, 128)
    elif small_dim == 1:
      shape = (128, 8)
    else:
      raise ValueError("small_dim must be 0 or 1")
    tiled_shape = ((shape[0] + 63) // 64, (shape[1] + 63) // 64, 64, 64)
    padded_shape = (math.prod(tiled_shape[0::2]), math.prod(tiled_shape[1::2]))
    def kernel(ctx, src, dst, smem):
      tmp, barrier = smem
      ctx.async_copy(
          src_ref=src,
          dst_ref=tmp,
          swizzle=128,
          gmem_transform=mgpu.TileTransform((64, 64)),
          gmem_slice=(ds(0, padded_shape[0]), ds(0, padded_shape[1])),
          barrier=barrier,
      )
      barrier.wait()
      copy(tmp, dst, swizzle=128)
    x = np.arange(np.prod(shape), dtype=jnp.float16).reshape(shape)
    tiled = jax.ShapeDtypeStruct(tiled_shape, jnp.float16)
    y_tiled = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, tiled, (tiled, mgpu.TMABarrier()),
    )(x)
    y = y_tiled.swapaxes(1, 2).reshape(padded_shape)
    # y should contain x and zero everywhere else.
    np.testing.assert_array_equal(y[:shape[0], :shape[1]], x)
    y_mut = np.asarray(y).copy()
    y_mut[:shape[0], :shape[1]] = 0
    np.testing.assert_array_equal(y_mut, np.zeros_like(y_mut))

  @parameterized.parameters(0, 1)
  def test_tma_small_tile_store(self, small_dim):
    if small_dim == 0:
      shape = (4, 128)
    elif small_dim == 1:
      shape = (128, 8)
    else:
      raise ValueError("small_dim must be 0 or 1")
    tiled_shape = ((shape[0] + 63) // 64, (shape[1] + 63) // 64, 64, 64)
    m, n = (math.prod(tiled_shape[0::2]), math.prod(tiled_shape[1::2]))
    def kernel(ctx, dst, tmp):
      vals = iota_tensor(m, n, jnp.float16)
      vals.store_tiled(tmp, swizzle=128)
      ctx.async_copy(
          src_ref=tmp,
          dst_ref=dst,
          swizzle=128,
          gmem_transform=mgpu.TileTransform((64, 64)),
          gmem_slice=(ds(0, m), ds(0, n)),
      )
      ctx.await_async_copy(0)
    tiled = jax.ShapeDtypeStruct(tiled_shape, jnp.float16)
    out = jax.ShapeDtypeStruct(shape, jnp.float16)
    y = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out, tiled,
    )()
    iota = np.arange(m * n, dtype=jnp.float16).reshape([m, n])
    np.testing.assert_array_equal(y, iota[:shape[0], :shape[1]])

  def test_tma_invalid(self):
    def kernel(ctx, src, dst, tmp):
      copy(src, tmp)
      ctx.async_copy(src_ref=tmp, dst_ref=dst)
      ctx.await_async_copy(0)

    def run_kernel(shape):
      x = np.arange(np.prod(shape)).reshape(shape)
      _ = mgpu.as_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), x, x, x)(x)

    with self.assertRaisesRegex(ValueError, "only support striding up to 5"):
      run_kernel([1] * 6)

    with self.assertRaisesRegex(
        ValueError, "last dimension to be divisible by 16"
    ):
      run_kernel([23])


class FragmentedArrayTest(TestCase):

  @parameterized.product(
      op=(
          operator.add,
          operator.mul,
          operator.sub,
          operator.truediv,
          operator.mod,
          (lambda x, y: mgpu.FragmentedArray.max(x, y), np.maximum),
      ),
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
      m=(64, 128),
      n=(8, 16, 32, 64, 80, 128, 256),
  )
  @jtu.ignore_warning(message="(invalid value|divide by zero)",
                      category=RuntimeWarning)
  def test_binary(self, op, dtype, m=64, n=32):
    if isinstance(op, tuple):
      op, np_op = op
    else:
      np_op = op

    if jnp.issubdtype(dtype, jnp.integer) and op is operator.truediv:
      self.skipTest("Unsupported for integer types")
    if jnp.issubdtype(dtype, jnp.floating) and op is operator.mod:
      self.skipTest("Unsupported for floating types")

    for scalar_rhs in [None, 2]:
      def kernel(ctx, dst, _):
        mlir_dtype = utils.dtype_to_ir_type(dtype)
        iota = iota_tensor(m, n, dtype)
        rhs = iota if scalar_rhs is None else c(scalar_rhs, mlir_dtype)
        op(iota, rhs).store_untiled(dst)
      out_shape = jax.ShapeDtypeStruct((m, n), dtype)
      result = mgpu.as_gpu_kernel(
          kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
      )()
      ref_x = np.arange(m * n, dtype=dtype).reshape(m, n)
      ref_rhs = scalar_rhs or ref_x
      if op is operator.truediv:
        np.testing.assert_allclose(result, np_op(ref_x, ref_rhs), atol=2e-7)
      else:
        np.testing.assert_array_equal(result, np_op(ref_x, ref_rhs))

  @parameterized.product(
      op=[
          operator.lt,
          operator.le,
          operator.gt,
          operator.ge,
          operator.eq,
          operator.ne,
      ],
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
  )
  def test_comparison(self, op, dtype, m=64, n=32):
    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, dtype)
      op(iota, iota + 1).store_untiled(dst)

    out_shape = jax.ShapeDtypeStruct((m, n), jnp.bool)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    iota = np.arange(m * n, dtype=dtype).reshape(m, n)
    np.testing.assert_array_equal(result, op(iota, iota + 1))

  @parameterized.product(
      ops=(
          (lambda x: -x, jax.lax.neg),
          (lambda x: x + 42, lambda x: x + 42),
      ),
      dtype=[jnp.float32, jnp.int32, jnp.uint32],
  )
  def test_unary(self, ops, dtype, m=64, n=32):
    op, np_op = ops

    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, dtype)
      op(iota).store_untiled(dst)

    out_shape = jax.ShapeDtypeStruct((m, n), dtype)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=dtype).reshape(m, n)
    np.testing.assert_allclose(result, np_op(x), atol=2e-7, rtol=2e-7)

  def test_select(self, m=64, n=32):

    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, jnp.int32)
      (iota < 16).select(iota * 2, iota * 3).store_untiled(dst)

    out_shape = jax.ShapeDtypeStruct((m, n), jnp.int32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=jnp.int32).reshape(m, n)
    np.testing.assert_array_equal(result, np.where(x < 16, x * 2, x * 3))

  @parameterized.product(
      ops=[
          (lambda x: mgpu.FragmentedArray.exp(x), np.exp),
          (lambda x: mgpu.FragmentedArray.sin(x), np.sin),
          (lambda x: mgpu.FragmentedArray.cos(x), np.cos),
          (lambda x: mgpu.FragmentedArray.rsqrt(x), jax.lax.rsqrt),
      ],
      approx=[False, True],
  )
  @jtu.ignore_warning(message="overflow encountered", category=RuntimeWarning)
  def test_math(self, ops, approx, m=64, n=32):
    op, np_op = ops
    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, jnp.float32)
      op(iota).store_untiled(dst)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    x = np.arange(m * n, dtype=jnp.float32).reshape(m, n)
    atol = 5e-3 if approx else 2e-7
    rtol = 4e-6 if approx else 2e-7
    np.testing.assert_allclose(result, np_op(x), atol=atol, rtol=rtol)

  @parameterized.product(
      dtype=[jnp.float32, jnp.int32],
      m=[128],
      n=[32, 64],
  )
  def test_reduce_sum(self, dtype, m, n):
    def kernel(ctx, src, dst, scratch):
      src = mgpu.FragmentedArray.load_strided(
          src, is_signed=utils.is_signed(dtype)
      )
      acc = mgpu.FragmentedArray.splat(
          src.reduce_sum(scratch),
          (m,),
          is_signed=src.is_signed
      )
      acc.store_untiled(dst)

    in_shape = jax.ShapeDtypeStruct((m, n), dtype)
    out_shape = jax.ShapeDtypeStruct((m,), dtype)
    kernel_fn = mgpu.as_gpu_kernel(
        kernel,
        (1, 1, 1),
        (128, 1, 1),
        in_shape,
        out_shape,
        smem_scratch_shape=jax.ShapeDtypeStruct((4,), dtype),
    )
    x = np.arange(m * n, dtype=dtype).reshape(m, n)
    np.testing.assert_array_equal(kernel_fn(x), jnp.full((m,), x.sum()))

  @parameterized.product(
      op=(arith.addf, arith.maximumf),
      m=(64, 128),
      n=(8, 16, 32, 64, 80, 128, 256),
  )
  def test_reduce(self, op, m=64, n=32):
    def kernel(ctx, dst, _):
      iota = iota_tensor(m, n, jnp.float32)
      iota.reduce(op, axis=1).broadcast_minor(n).store_untiled(dst)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mgpu.as_gpu_kernel(
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
      iota = iota_tensor(m, n, jnp.float32)
      cte = c(1, iota.mlir_dtype)
      cte_arr = mgpu.FragmentedArray.splat(cte, ())
      cte_arr = cte_arr.reshape((1, 1)).broadcast((m, n))
      (iota + cte_arr).store_untiled(dst)
    out_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )()
    expected = np.arange(m * n, dtype=jnp.float32).reshape(m, n) + 1
    np.testing.assert_array_equal(result, expected)

  def test_splat(self):
    def kernel(ctx, dst, _):
      f32 = ir.F32Type.get()
      v = arith.constant(f32, ir.FloatAttr.get(f32, 3.14))
      t = mgpu.FragmentedArray.splat(
          v, (128,), mgpu.WGMMA_ROW_LAYOUT
      )
      t.broadcast_minor(32).store_untiled(dst)
    out_shape = jax.ShapeDtypeStruct((128, 32), jnp.float32)
    result = mgpu.as_gpu_kernel(
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
    result = mgpu.as_gpu_kernel(
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
    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), x, [],
    )()
    for i in range(0, 128, 4):
      x[i:i + 4] = jnp.sum(x[i:i + 4])

    np.testing.assert_array_equal(result, x)

  @parameterized.named_parameters(
      ("_bf16", jnp.bfloat16)
  )
  def test_fast_i8_convert(self, jax_dtype_to):
    jax_dtype_to = jnp.dtype(jax_dtype_to)
    jax_dtype_from = jnp.dtype(jnp.int8)
    mlir_dtype_to = utils.dtype_to_ir_type(jax_dtype_to)
    def kernel(ctx, inp, out, smem):
      del ctx, smem
      arr = mgpu.FragmentedArray.load_strided(inp, is_signed=True)
      arr.astype(mlir_dtype_to).store_untiled(out)

    x = jnp.arange(-128, 128, dtype=jax_dtype_from)
    reference = x.astype(jax_dtype_to)

    result = mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, reference, None,
    )(x)
    np.testing.assert_array_equal(result, reference)


class ProfilerTest(TestCase):

  def test_measure(self):
    x = jnp.arange(1024 * 1024)
    profiler.measure(lambda x, y: x + y, x, x)  # This is just a smoke test

  def test_multigpu(self):
    if len(jax.devices()) < 2:
      self.skipTest("Need at least 2 devices")
    def kernel(ctx, src, dst, _):
      mgpu.FragmentedArray.load_strided(src).store_untiled(dst)
    x = np.arange(64 * 64, dtype=jnp.float32).reshape(64, 64)
    f = jax.jit(mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, ()
    ))
    # Make sure we can invoke the same program on different devices.
    for xd in (jax.device_put(x, d) for d in jax.devices()[:2]):
      jax.block_until_ready(f(xd))


class TorchTest(TestCase):

  @classmethod
  def setUpClass(cls):
    try:
      import torch
    except ImportError:
      raise unittest.SkipTest("Test requires PyTorch")
    cls.torch = torch

  def test_basic(self):
    def kernel(ctx, i_gmem, o_gmem, _):
      x = mgpu.FragmentedArray.load_strided(i_gmem)
      (x + x).store_untiled(o_gmem)

    ty = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    x = self.torch.randn((128, 128), dtype=self.torch.float, device='cuda')
    f = mgpu.as_torch_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), ty, ty, ())
    y = f(x)
    np.testing.assert_allclose(y.cpu(), x.cpu() * 2)
    del y  # Make sure the destructor runs successfully.


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
