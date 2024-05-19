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
"""Matmul kernels for H100."""

import enum
import functools

import dataclasses
import jax
from jax import random
from jax._src.interpreters import mlir
from jax.experimental.mosaic import gpu as mosaic_gpu
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.mosaic.gpu.dsl import *  # noqa: F403
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvgpu
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector
import numpy as np

# mypy: ignore-errors
# ruff: noqa: F405
# pylint: disable=line-too-long, wildcard-import, missing-function-docstring, bad-continuation, g-bad-todo, protected-access

SmemRef = ir.Value


@dataclasses.dataclass
class MMAOperandInfo:
  dtype: jnp.dtype
  shape: tuple[int, int, int, int, int, int] | None = None
  # gmem order is always row major but some wgmmas can handle row
  # major in smem aswell.
  smem_order: WGMMALayout = WGMMALayout.ROW_MAJOR

  @property
  def tiling(self) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(self.shape[2:], self.dtype)

  @property
  def contiguous_tile(self) -> int:
    """Tile along the contigoues dimension."""
    return self.shape[3] * self.shape[5]

  @property
  def mlir_dtype(self) -> ir.Type:
    dtype = np.dtype(self.dtype)
    if dtype == np.dtype(np.float16):
      return ir.F16Type.get()
    elif dtype == np.dtype("bfloat16"):
      return ir.BF16Type.get()
    elif dtype == np.dtype(np.float32):
      return ir.F32Type.get()
    elif np.issubdtype(dtype, np.integer):
      return ir.IntegerType.get_signless(dtype.itemsize * 8)
    else:
      raise ValueError(f"Unsupported dtype: {dtype}")

  @property
  def element_bytewidth(self) -> int:
    return bytewidth(self.mlir_dtype)

  @property
  def elems_128b(self) -> int:
    return 128 // self.element_bytewidth

  @property
  def tile_bytes(self):
    return np.prod(self.shape[-4:]) * self.element_bytewidth


def operand_info(
    m: int,
    n: int,
    k: int,
    tile_m: int,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    rhs_transpose: bool,
) -> tuple[MMAOperandInfo, MMAOperandInfo]:
  b_order = WGMMALayout.COL_MAJOR if rhs_transpose else WGMMALayout.ROW_MAJOR
  lhs_info, rhs_info = MMAOperandInfo(lhs_dtype), MMAOperandInfo(rhs_dtype, smem_order=b_order)
  tile_k, tile_n = max(lhs_info.elems_128b, rhs_info.elems_128b), 128

  if tile_n % rhs_info.elems_128b != 0:
    raise ValueError(
        f"{tile_n=} must be divisible by 128 bytes ="
        f" {(lhs_info.elems_128b, lhs_dtype)}"
    )

  lhs_shape = list(tile_shape((m, k), (tile_m, tile_k)))
  rhs_shape = list(tile_shape((k, n), (tile_k, tile_n)))
  rhs_tiling = (lhs_info.elems_128b, rhs_info.elems_128b)
  lhs_shape[2:] = tile_shape(lhs_shape[2:],  (64, lhs_info.elems_128b))
  rhs_shape[2:] = tile_shape(rhs_shape[2:], rhs_tiling)

  lhs_info.shape, rhs_info.shape = tuple(lhs_shape), tuple(rhs_shape)
  return lhs_info, rhs_info


class F32Precision(enum.Enum):
  DEFAULT = enum.auto()
  TF32_X3 = enum.auto()


class WGMMADefaultImpl:
  """Default WGMMA implementation."""

  @staticmethod
  def zero_accs(tile_m: int, tile_n: int) -> WGMMAAccumulator:
    return WGMMAAccumulator.zero(tile_m, tile_n)

  @staticmethod
  def smem_shape_extra(lhs: MMAOperandInfo, rhs: MMAOperandInfo) -> dict[str, jax.ShapeDtypeStruct]:
    del lhs, rhs
    return {}

  @staticmethod
  def get_result_tile(acc: WGMMAAccumulator) -> FragmentedArray:
    return acc.value

  @staticmethod
  def wgmma(
      smem_scratch: dict[str, SmemRef],  # pylint: disable=unused-argument
      acc: WGMMAAccumulator,
      b_order: WGMMALayout,
      a_slice: SmemRef,
      b_slice: SmemRef,
  ) -> dict[str, WGMMAAccumulator]:
    acc = wgmma(acc, a_slice, b_slice, b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    nvvm.wgmma_wait_group_sync_aligned(1)
    return acc


class WGMMATF32x3Impl:
  """WGMMA implementation for 3xTF32 precision."""

  @staticmethod
  def zero_accs(tile_m, tile_n) -> dict[str, WGMMAAccumulator]:
    zero_acc = WGMMADefaultImpl.zero_accs(tile_m, tile_n)
    return {"main": zero_acc, "errs": zero_acc}

  @staticmethod
  def smem_shape_extra(lhs: MMAOperandInfo, rhs: MMAOperandInfo) -> dict[str, jax.ShapeDtypeStruct]:
    lhs_err = jax.ShapeDtypeStruct(shape=lhs.shape[2:], dtype=lhs.dtype)
    rhs_err = jax.ShapeDtypeStruct(shape=rhs.shape[2:], dtype=rhs.dtype)
    return {"lhs_err": lhs_err, "rhs_err": rhs_err}

  @staticmethod
  def get_result_tile(accs) -> FragmentedArray:
    return accs["main"].value + accs["errs"].value

  @staticmethod
  def rounding_error(x_ref, err_ref):
    """Store the TF32 rounding error of x_ref in err_ref."""
    f32 = ir.F32Type.get()
    i32 = ir.IntegerType.get_signless(32)
    t = FragmentedArray.load_strided(x_ref)
    tf32_mask = FragmentedArray.splat(c(0xFFFFE000, i32), t.shape, t.layout)
    t_tf32 = (t.bitcast(i32) & tf32_mask).bitcast(f32)
    (t - t_tf32).store_untiled(err_ref)

  @staticmethod
  def wgmma(
      smem_scratch: dict[str, SmemRef],
      accs: dict[str, WGMMAAccumulator],
      b_order: WGMMALayout,
      a_slice: SmemRef,
      b_slice: SmemRef,
  ) -> dict[str, WGMMAAccumulator]:
    acc = wgmma(accs["main"], a_slice, b_slice, b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    # Note: we assert that only the slice_ab and err_b mmas are still running
    # which are unaffected by writing to the err_a shared memory.
    # After nvvm.wgmma_wait_group_sync_aligned(2) there are no wgmmas
    # accessing err_a so we can safely write to it.
    nvvm.wgmma_wait_group_sync_aligned(2)
    WGMMATF32x3Impl.rounding_error(a_slice, smem_scratch["lhs_err"])
    commit_shared()
    acc_err = wgmma(accs["errs"], smem_scratch["lhs_err"], b_slice, b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    # Note: similar to the above we wait for the last wgmma access to
    # err_b which was 2 wgmmas ago.
    nvvm.wgmma_wait_group_sync_aligned(2)
    WGMMATF32x3Impl.rounding_error(b_slice, smem_scratch["rhs_err"])
    commit_shared()
    acc_err = wgmma(acc_err, a_slice, smem_scratch["rhs_err"], b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    nvvm.wgmma_wait_group_sync_aligned(2)
    return {"main": acc, "errs": acc_err}


class WGMMACvtRhsImpl:
  """Mixed WGMMA implementation where B is converted to A."""

  @staticmethod
  def zero_accs(tile_m: int, tile_n: int) -> WGMMAAccumulator:
    return WGMMADefaultImpl.zero_accs(tile_m, tile_n)

  @staticmethod
  def smem_shape_extra(lhs: MMAOperandInfo, rhs: MMAOperandInfo) -> dict[str, jax.ShapeDtypeStruct]:
    if rhs.smem_order == WGMMALayout.COL_MAJOR:
      raise NotImplementedError("Transpose requires more elaborate handling of tiling.")

    tiling = rhs.shape[-2], lhs.shape[-1]
    rhs_tile_shape = rhs.shape[-4] * rhs.shape[-2], rhs.shape[-3] * rhs.shape[-1]
    cvt_shape = tile_shape(rhs_tile_shape, tiling)
    return {"cvt": jax.ShapeDtypeStruct(shape=cvt_shape, dtype=lhs.dtype)}

  @staticmethod
  def get_result_tile(acc: WGMMAAccumulator) -> FragmentedArray:
    return WGMMADefaultImpl.get_result_tile(acc)

  @staticmethod
  def wgmma(
      smem_scratch: dict[str, SmemRef],  # pylint: disable=unused-argument
      acc: WGMMAAccumulator,
      b_order: WGMMALayout,
      a_slice: SmemRef,
      b_slice: SmemRef,
  ) -> dict[str, WGMMAAccumulator]:
    b_ty = ir.MemRefType(b_slice.type)

    # Convert the load
    arr = FragmentedArray.load_tiled(b_slice, swizzle=128)
    assert np.prod(b_ty.shape) == np.prod(arr.shape), (b_ty, arr.shape)
    cvt_ty = ir.MemRefType(smem_scratch["cvt"].type)
    assert np.prod(b_ty.shape) == np.prod(cvt_ty.shape), (b_ty, cvt_ty)
    arr = arr.astype(cvt_ty.element_type)
    assert np.prod(b_ty.shape) == np.prod(arr.shape), (b_ty, arr.shape)
    # Make sure no wgmma is running.
    # TODO(cperivol): double buffer.
    nvvm.wgmma_wait_group_sync_aligned(0)
    arr.store_tiled(smem_scratch["cvt"], swizzle=128)
    commit_shared()
    # TODO(cperivol): we can now start the TMA for B.
    nvvm.wgmma_fence_aligned();
    return wgmma(acc, a_slice, smem_scratch["cvt"], b_order=b_order)


def build_kernel(
    m: int,
    k: int,
    n: int,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    stages: int = 2,
    tile_m: int = 128,
    rhs_transpose: bool = False,
    wgmma_impl=WGMMADefaultImpl,
    profiler_spec: profiler.ProfilerSpec | None = None,
):
  if tile_m % 64 != 0:
    raise ValueError(f"{tile_m=} must be divisible by 64")
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % 64 != 0:
    raise ValueError(f"n must be divisible by 64, but got {n=}")
  if stages < 2:
    raise ValueError(f"Need at least 2 stages, but got {stages=}")

  smem = ir.Attribute.parse("#gpu.address_space<workgroup>")

  lhs_info, rhs_info = operand_info(m, n, k, tile_m, lhs_dtype, rhs_dtype, rhs_transpose)
  tile_k, tile_n = lhs_info.contiguous_tile, rhs_info.contiguous_tile
  if k % (stages * tile_k) != 0:
    raise ValueError(
        f"k must be divisible by {stages=} * {tile_k=} (={stages * tile_k}),"
        f" but got {k=}"
    )

  f32 = ir.F32Type.get()
  index = ir.IndexType.get()
  i8 = ir.IntegerType.get_signless(8)

  def safe_div(x, y):
    assert x % y == 0, (x, y)
    return x // y

  grid = (safe_div(m, tile_m), safe_div(n, tile_n), 1)
  block = (128, 1, 1)

  def c(value, ty=index):
    return arith.ConstantOp(ty, ir.IntegerAttr.get(ty, value))

  smem_shape = {
      "lhs": jax.ShapeDtypeStruct((stages, *lhs_info.shape[2:]), lhs_dtype),
      "rhs": jax.ShapeDtypeStruct((stages, *rhs_info.shape[2:]), rhs_dtype),
  }
  smem_shape |= wgmma_impl.smem_shape_extra(lhs_info, rhs_info)

  def _main(ctx, a_device, b_device, c_device, smem_scratch):
    memref.assume_alignment(c_device, 16)

    barrier_group = BarrierArray(stages)
    m_start = arith.muli(c(tile_m), gpu.block_id(gpu.Dimension.x))
    n_start = arith.muli(c(tile_n), gpu.block_id(gpu.Dimension.y))

    def fetch(slot, ki):
      barrier = barrier_group[slot]
      k_start = arith.muli(c(tile_k), ki)
      txcount = c(lhs_info.tile_bytes + rhs_info.tile_bytes)
      common_copy_args = dict(
          swizzle=128, barrier=barrier, arrive=False, uniform=False,
      )
      with once():
        nvgpu.mbarrier_arrive_expect_tx(barrier_group.value, txcount, slot)
        ctx.async_copy(
            src_ref=a_device,
            dst_ref=memref_slice(smem_scratch["lhs"], slot),
            gmem_slice=(ds(m_start, tile_m), ds(k_start, tile_k)),
            gmem_transform=mosaic_gpu.TileTransform(lhs_info.shape[-2:]),
            **common_copy_args,
        )
        rhs_slice = (ds(k_start, rhs_info.shape[2] * rhs_info.shape[4]),
                     ds(n_start, rhs_info.shape[3] * rhs_info.shape[5]))
        rhs_transform = (mosaic_gpu.TileTransform(rhs_info.shape[-2:]),)
        if rhs_info.smem_order == WGMMALayout.COL_MAJOR:
          rhs_slice = rhs_slice[::-1]
          rhs_transform += (mosaic_gpu.TransposeTransform((1, 0, 2, 3)),)
          assert rhs_info.shape[-2] == rhs_info.shape[-1]  # No need to flip the tiling.
        ctx.async_copy(
            src_ref=b_device,
            dst_ref=memref_slice(smem_scratch["rhs"], slot),
            gmem_slice=rhs_slice,
            gmem_transform=rhs_transform,
            **common_copy_args,
        )

    accs = wgmma_impl.zero_accs(tile_m, tile_n)

    with ctx.named_region("TMA warmup"):
      for i in range(stages):
        fetch(c(i), c(i))

    @fori(c(k // tile_k), accs)
    def stage_loop_body(ki, accs):
      si = arith.remui(ki, c(stages))

      with ctx.named_region("TMA wait"):
        barrier_group[si].wait()

      with ctx.named_region("WGMMA"):
        a_slice = memref_slice(smem_scratch["lhs"], si)
        b_slice = memref_slice(smem_scratch["rhs"], si)
        accs = wgmma_impl.wgmma(smem_scratch, accs, rhs_info.smem_order, a_slice, b_slice)

      with ctx.named_region("TMA start"):
        tma_ki = arith.addi(ki, c(stages - 1))
        do_tma = arith.cmpi(arith.CmpIPredicate.slt, tma_ki, c(k // tile_k))
        not_first_step = arith.cmpi(arith.CmpIPredicate.ne, ki, c(0))
        if_op = scf.IfOp(arith.andi(not_first_step, do_tma))
        with ir.InsertionPoint(if_op.then_block):
          tma_si = arith.remui(tma_ki, c(stages))
          fetch(tma_si, tma_ki)
          scf.yield_([])

      return accs

    # Wait until everyone is done with their WMMA
    with ctx.named_region("WGMMA drain"):
      nvvm.wgmma_wait_group_sync_aligned(0)

    # We can repurpose the tile SMEM for the epilogue now
    # TODO(apaszke): Add support for aliased SMEM allocations to the DSL
    dynamic_smem = gpu.dynamic_shared_memory(
        ir.MemRefType.get(
            (ir.ShapedType.get_dynamic_size(),), i8, memory_space=smem
        )
    )
    with ctx.named_region("SMEM store"):
      acc_val = wgmma_impl.get_result_tile(stage_loop_body.result)
      acc_smem = memref.view(
          ir.MemRefType.get(
              tile_shape((tile_m, tile_n), (64, 32)), f32, memory_space=smem
          ),
          dynamic_smem, c(0), [],
      )
      acc_val.store_tiled(acc_smem, swizzle=128)
      gpu.barrier()

    with ctx.named_region("GMEM store"):
      # Vectorized epilogue to move results from SMEM to GMEM
      tidx = gpu.thread_id(gpu.Dimension.x)
      warp_id = arith.divui(tidx, c(32))
      lane_id = arith.remui(tidx, c(32))
      n_store = arith.muli(lane_id, c(4))
      col_group = arith.divui(lane_id, c(8))
      n_load = arith.muli(arith.remui(lane_id, c(8)), c(4))
      for_op = scf.ForOp(warp_id, c(tile_m), c(4))
      with ir.InsertionPoint(for_op.body):
        m_smem = for_op.induction_variable
        m_within_tile = arith.remui(m_smem, c(64))
        m_tile = arith.divui(m_smem, c(64))
        swizzle_source = arith.shli(arith.remui(m_smem, c(8)), c(2))
        n_acc = arith.xori(n_load, swizzle_source)
        acc_part = vector.load(
            ir.VectorType.get((4,), f32),
            acc_smem,
            [m_tile, col_group, m_within_tile, n_acc],
        )
        vector.store(
            acc_part,
            c_device,
            [arith.addi(m_start, m_smem), arith.addi(n_start, n_store)],
        )
        scf.yield_([])

  rhs_shape = (n, k) if rhs_transpose else (k, n)
  return mosaic_gpu.as_gpu_kernel(
      _main,
      grid,
      block,
      (
          jax.ShapeDtypeStruct((m, k), lhs_dtype),
          jax.ShapeDtypeStruct(rhs_shape, rhs_dtype),
      ),
      jax.ShapeDtypeStruct((m, n), jnp.float32),
      smem_shape,
      profiler_spec,
  )


def random_array(key, shape: tuple[int, ...], dtype: jnp.dtype):
  if jax.dtypes.issubdtype(dtype, np.floating):
    return random.uniform(key, shape, dtype=dtype)
  elif jax.dtypes.issubdtype(dtype, np.integer):
    return random.randint(key, shape, -127, 127, dtype)
  else:
    raise NotImplementedError(dtype)

def verify(
    m=(33 * 128),
    k=2048,
    n=(4 * 128),
    stages=4,
    tile_m=128,
    profile=False,
    lhs_dtype=jnp.float16,
    rhs_dtype=jnp.float16,
    rhs_transpose=False,
    precision: F32Precision = F32Precision.DEFAULT,
):
  if not rhs_transpose and jnp.dtype(lhs_dtype).itemsize != 2:
    raise ValueError("Implicit transpose can only happen for float16.")

  with mlir.make_ir_context(), ir.Location.unknown():
    kx, ky = random.split(random.key(1234))
    x = random_array(kx, (m, k), lhs_dtype)
    y = random_array(ky, (n, k) if rhs_transpose else (k, n), rhs_dtype)

    if lhs_dtype != rhs_dtype:
      impl = WGMMACvtRhsImpl
    else:
      match precision:
        case F32Precision.DEFAULT:
          impl = WGMMADefaultImpl
        case F32Precision.TF32_X3:
          impl = WGMMATF32x3Impl

    prof_spec = profiler.ProfilerSpec(132 * 4096) if profile else None
    f = build_kernel(
        m,
        k,
        n,
        stages=stages,
        tile_m=tile_m,
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        rhs_transpose=rhs_transpose,
        wgmma_impl=impl,
        profiler_spec=prof_spec,
    )
    z, runtime = profiler.measure(f, x, y)

    if rhs_transpose:
      dimension_numbers = ((1,), (1,)), ((), ())
    else:
      dimension_numbers = ((1,), (0,)), ((), ())
    if lhs_dtype == jnp.dtype(jnp.float32):  # Account for the tf32 precision
      assert lhs_dtype == rhs_dtype
      exponent_bits, mantissa_bits = 8, 10
      x, y = (
          jax.lax.reduce_precision(v, exponent_bits, mantissa_bits)
          for v in (x, y)
      )
    ref = jax.lax.dot_general(
        x, y, dimension_numbers,
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_allclose(ref, z, atol=1e-3, rtol=1e-3)
    return runtime


if __name__ == "__main__":
  m, k, n = 33 * 128, 2048, 4 * 128
  runtime = verify(m=m, k=k, n=n)
  tflops = float(2 * k * m * n) / (runtime / 1e3) / 1e12
  print(f"{runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
