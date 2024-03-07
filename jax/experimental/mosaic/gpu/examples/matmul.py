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

import dataclasses
import enum
import functools

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


@dataclasses.dataclass(frozen=True)
class Tiling:
  m: int
  n: int
  k: int

  @property
  def mk(self):
    return (self.m, self.k)

  @property
  def kn(self):
    return (self.k, self.n)

  @property
  def nk(self):
    return (self.n, self.k)

  @property
  def mn(self):
    return (self.m, self.n)


class F32Precision(enum.Enum):
  DEFAULT = enum.auto()
  TF32_X3 = enum.auto()


class WGMMADefaultImpl:
  """Default WGMMA implementation."""

  @staticmethod
  def zero_accs(tile_m: int, tile_n: int) -> WGMMAAccumulator:
    return WGMMAAccumulator.zero(tile_m, tile_n)

  @staticmethod
  def smem_shape_extra(
      block_tiling: Tiling,
      tma_tiling: Tiling,
      lhs_dtype: jnp.dtype, rhs_dtype: jnp.dtype,
      rhs_transpose: WGMMALayout,
  ) -> dict[str, jax.ShapeDtypeStruct]:
    del block_tiling, tma_tiling, lhs_dtype, rhs_dtype, rhs_transpose
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
  def smem_shape_extra(
      block_tiling: Tiling,
      tma_tiling: Tiling,
      lhs_dtype: jnp.dtype, rhs_dtype: jnp.dtype,
      rhs_transpose: bool,
  ) -> dict[str, jax.ShapeDtypeStruct]:
    del rhs_transpose
    lhs_err = jax.ShapeDtypeStruct(shape=tile_shape(block_tiling.mk, tma_tiling.mk), dtype=lhs_dtype)
    rhs_err = jax.ShapeDtypeStruct(shape=tile_shape(block_tiling.kn, tma_tiling.kn), dtype=rhs_dtype)
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
  def smem_shape_extra(
      block_tiling: Tiling,
      tma_tiling: Tiling,
      lhs_dtype: jnp.dtype, rhs_dtype: jnp.dtype,
      rhs_transpose: bool,
  ) -> dict[str, jax.ShapeDtypeStruct]:
    del rhs_dtype
    if rhs_transpose:
      raise NotImplementedError("Transpose requires more elaborate handling of tiling.")

    if tma_tiling.k != 64:
      raise ValueError(f"WGMMA layout needs the left tiling dimension to be 64 {tma_tiling.k=}")

    # The second dim needs to be tma_tiling.k so it is 128b wide and
    # the first dim needs to line up with the lhs dimension. That's
    # why we have a strange (k, k) here.
    cvt_shape = tile_shape(block_tiling.kn, (tma_tiling.k, tma_tiling.k))
    return {"cvt": jax.ShapeDtypeStruct(shape=cvt_shape, dtype=lhs_dtype)}

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
    # Convert the load
    arr = FragmentedArray.load_tiled(b_slice, swizzle=128)
    cvt_ty = ir.MemRefType(smem_scratch["cvt"].type)
    # TODO(cperivol): https://research.google/blog/mixed-input-matrix-multiplication-performance-optimizations/
    arr = arr.astype(cvt_ty.element_type)
    # Make sure no wgmma is running.
    # TODO(cperivol): double buffer.
    nvvm.wgmma_wait_group_sync_aligned(0)
    arr.store_tiled(smem_scratch["cvt"], swizzle=128)
    commit_shared()
    acc = wgmma(acc, a_slice, smem_scratch["cvt"], b_order=b_order)
    nvvm.wgmma_commit_group_sync_aligned()
    return acc


def mlir_context(f):
  def wrap(*args, **kw):
    with mlir.make_ir_context(), ir.Location.unknown():
      return f(*args, **kw)

  return wrap

@mlir_context
def build_kernel(
    m, n, k,
    lhs_dtype, rhs_dtype,
    stages: int = 2,
    tile_m: int = 128,
    tile_n: int = 128,
    rhs_transpose: bool = False,
    wgmma_impl=WGMMADefaultImpl,
    profiler_spec: profiler.ProfilerSpec | None = None,
):
  f32 = ir.F32Type.get()
  out_128b_elems = 128 // bytewidth(f32)
  out_tiling = (64, out_128b_elems)
  out_tile = jax.ShapeDtypeStruct(tile_shape((tile_m, tile_n), out_tiling), jnp.float32)
  if tile_m % 64 != 0:
    raise ValueError(f"{tile_m=} must be divisible by 64")
  if m % tile_m != 0:
    raise ValueError(f"{m=} must be divisible by {tile_m=}")
  if n % 64 != 0:
    raise ValueError(f"n must be divisible by 64, but got {n=}")
  if stages < 2:
    raise ValueError(f"Need at least 2 stages, but got {stages=}")

  lhs_elem_bytes = bytewidth(mlir.dtype_to_ir_type(lhs_dtype))
  rhs_elem_bytes = bytewidth(mlir.dtype_to_ir_type(rhs_dtype))
  lhs_128b_elems = 128 // lhs_elem_bytes
  rhs_128b_elems = 128 // rhs_elem_bytes
  tile_k = max(lhs_128b_elems, rhs_128b_elems)

  if tile_n % rhs_128b_elems != 0:
    raise ValueError(
        f"{tile_n=} must be divisible by 128 bytes ="
        f" {((lhs_128b_elems, lhs_dtype), (rhs_128b_elems, rhs_dtype))}"
    )

  if k % tile_k != 0:
    raise ValueError(f"k must be divisible by {tile_k=}, but got {k=}")

  block_tiling = Tiling(m=tile_m, n=tile_n, k=tile_k)
  tma_tiling = Tiling(m=64, n=rhs_128b_elems, k=lhs_128b_elems)
  k_steps = k // block_tiling.k
  stages = min(stages, k_steps)

  def safe_div(x, y):
    assert x % y == 0, (x, y)
    return x // y

  grid = (safe_div(m, block_tiling.m), safe_div(n, block_tiling.n), 1)
  block = (128, 1, 1)

  c = arith.ConstantOp.create_index
  divmod = lambda x, y: (arith.divui(x, c(y)), arith.remui(x, c(y)))

  compute_scratch_shapes = {
      "lhs": jax.ShapeDtypeStruct((stages, *tile_shape(block_tiling.mk, tma_tiling.mk)), lhs_dtype),
      "rhs": jax.ShapeDtypeStruct((stages, *tile_shape(block_tiling.kn, tma_tiling.kn)), rhs_dtype),
  }
  compute_scratch_shapes |= wgmma_impl.smem_shape_extra(block_tiling, tma_tiling, lhs_dtype, rhs_dtype, rhs_transpose)

  epilogue_scratch_shapes = {
      "acc": jax.ShapeDtypeStruct(out_tile.shape, out_tile.dtype),
  }

  smem_shape = mosaic_gpu.Union(
      [compute_scratch_shapes, epilogue_scratch_shapes])

  def _main(ctx, a_device, b_device, c_device,
            smem_union: mosaic_gpu.Union[mosaic_gpu.RefTree]):
    compute_smem, epilogue_smem = smem_union.members

    memref.assume_alignment(c_device, 16)

    barrier_group = BarrierArray(stages)
    m_start = arith.muli(c(block_tiling.m), gpu.block_id(gpu.Dimension.x))
    n_start = arith.muli(c(block_tiling.n), gpu.block_id(gpu.Dimension.y))

    def fetch(slot, ki):
      barrier = barrier_group[slot]
      k_start = arith.muli(c(block_tiling.k), ki)
      lhs_tma_tile_bytes = int(np.prod(block_tiling.mk) * lhs_elem_bytes)
      rhs_tma_tile_bytes = int(np.prod(block_tiling.kn) * rhs_elem_bytes)
      txcount = c(lhs_tma_tile_bytes + rhs_tma_tile_bytes)
      common_copy_args = dict(
          swizzle=128, barrier=barrier, arrive=False, uniform=False,
      )
      with single_thread():
        nvgpu.mbarrier_arrive_expect_tx(barrier_group.value, txcount, slot)
        ctx.async_copy(
            src_ref=a_device,
            dst_ref=memref_slice(compute_smem["lhs"], slot),
            gmem_slice=(ds(m_start, block_tiling.m), ds(k_start, block_tiling.k)),
            gmem_transform=mosaic_gpu.TileTransform(tma_tiling.mk),
            **common_copy_args,
        )
        rhs_slice = (ds(k_start, block_tiling.k), ds(n_start, block_tiling.n))
        rhs_transform = (mosaic_gpu.TileTransform(tma_tiling.kn),)
        if rhs_transpose:
          rhs_slice = rhs_slice[::-1]
          rhs_transform += (mosaic_gpu.TransposeTransform((1, 0, 2, 3)),)
          assert tma_tiling.n == tma_tiling.k, block_tiling  # No need to flip the tiling.
        ctx.async_copy(
            src_ref=b_device,
            dst_ref=memref_slice(compute_smem["rhs"], slot),
            gmem_slice=rhs_slice,
            gmem_transform=rhs_transform,
            **common_copy_args,
        )

    accs = wgmma_impl.zero_accs(block_tiling.m, block_tiling.n)

    with ctx.named_region("TMA warmup"):
      for i in range(stages):
        fetch(c(i), c(i))

    @fori(c(k_steps), accs)
    def stage_loop_body(ki, accs):
      si = arith.remui(ki, c(stages))

      with ctx.named_region("TMA wait"):
        barrier_group[si].wait()

      with ctx.named_region("WGMMA"):
        a_slice = memref_slice(compute_smem["lhs"], si)
        b_slice = memref_slice(compute_smem["rhs"], si)
        rhs_smem_order = (
            WGMMALayout.COL_MAJOR if rhs_transpose else WGMMALayout.ROW_MAJOR
        )
        accs = wgmma_impl.wgmma(
            compute_smem, accs, rhs_smem_order, a_slice, b_slice)

      with ctx.named_region("TMA start"):
        tma_ki = arith.addi(ki, c(stages - 1))
        do_tma = arith.cmpi(arith.CmpIPredicate.slt, tma_ki, c(k_steps))
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

    with ctx.named_region("SMEM store"):
      acc_val = wgmma_impl.get_result_tile(stage_loop_body.result)
      acc_smem = epilogue_smem["acc"]
      acc_val.store_tiled(acc_smem, swizzle=128)
      gpu.barrier()

    with ctx.named_region("GMEM store"):
      # Vectorized epilogue to move results from SMEM to GMEM
      # TODO(apaszke): Make this into a proper copy function.
      warps_per_warpgroup = 4
      lanes_per_warp = 32
      m_out_tiling, n_out_tiling = out_tiling[-2:]
      warp_id, lane_id = divmod(gpu.thread_id(gpu.Dimension.x), lanes_per_warp)
      # We store 4 f32 numbers for a block of 16B.
      vector_len = 4
      num_vectors_per_row = safe_div(tile_n, vector_len)
      # Process several rows at once if it is necessary to fully exploit each
      # warp.
      if tile_n < lanes_per_warp * vector_len:
        num_rows_per_warp = min(
            safe_div(lanes_per_warp * vector_len, tile_n),
            safe_div(tile_m, warps_per_warpgroup))
      else:
        num_rows_per_warp = 1
      lanes_per_row = safe_div(lanes_per_warp, num_rows_per_warp)
      lane_row_offset, lane_col_offset = divmod(lane_id, lanes_per_row)
      warp_for_op = scf.ForOp(arith.muli(warp_id, c(num_rows_per_warp)),
                              c(tile_m),
                              c(warps_per_warpgroup * num_rows_per_warp))
      with ir.InsertionPoint(warp_for_op.body):
        start_row = warp_for_op.induction_variable
        m_row_idx = arith.addi(start_row, lane_row_offset)
        vector_for_op = scf.ForOp(lane_col_offset, c(num_vectors_per_row),
                                  c(lanes_per_row))
        with ir.InsertionPoint(vector_for_op.body):
          vector_idx = vector_for_op.induction_variable
          n_store = arith.muli(vector_idx, c(vector_len))
          col_group, n_load = divmod(n_store, n_out_tiling)
          m_tile, m_within_tile = divmod(m_row_idx, m_out_tiling)
          swizzle_source = arith.shli(arith.remui(m_row_idx, c(8)), c(2))
          n_acc = arith.xori(n_load, swizzle_source)
          acc_part = vector.load(
              ir.VectorType.get((vector_len,), f32),
              acc_smem,
              [m_tile, col_group, m_within_tile, n_acc],
          )
          vector.store(
              acc_part,
              c_device,
              [arith.addi(m_start, m_row_idx), arith.addi(n_start, n_store)],
          )
          scf.yield_([])
        scf.yield_([])

  return mosaic_gpu.as_gpu_kernel(
      _main,
      grid,
      block,
      (
          jax.ShapeDtypeStruct((m, k), lhs_dtype),
          jax.ShapeDtypeStruct((n, k) if rhs_transpose else (k, n), rhs_dtype),
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
    tile_n=128,
    profile=False,
    lhs_dtype=jnp.float16,
    rhs_dtype=jnp.float16,
    rhs_transpose=False,
    precision: F32Precision = F32Precision.DEFAULT,
):
  # TODO(cperivol): Transpose is only supported for 16bit wgmma. ATM
  # that means bf16 x bf16, f16 x f16 and bf16 x s8. When we get more
  # general mixed precision this check will need to be more nuanced.
  if not rhs_transpose and jnp.dtype(lhs_dtype).itemsize != 2:
    raise ValueError(
        "Implicit transpose can only happen for 16bit types (or mixed precision"
        " that is underpinned by 16bit operations)."
    )

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

  prof_spec = profiler.ProfilerSpec(4096) if profile else None
  f = build_kernel(
      m, n, k,
      jnp.dtype(lhs_dtype), jnp.dtype(rhs_dtype),
      stages=stages,
      tile_m=tile_m,
      tile_n=tile_n,
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
    exponent_bits, mantissa_bits = 8, 10
    x, y = (
        jax.lax.reduce_precision(v, exponent_bits, mantissa_bits)
        for v in (x, y)
    )

  ref_f = functools.partial(
      jax.lax.dot_general,
      dimension_numbers=dimension_numbers,
      preferred_element_type=jnp.float32,
  )

  ref, ref_runtime = profiler.measure(ref_f, x, y)
  np.testing.assert_allclose(z, ref, atol=1e-3, rtol=1e-3)
  return runtime, ref_runtime


if __name__ == "__main__":
  m, k, n = 33 * 128, 2048, 4 * 128
  runtime, ref_runtime = verify(m=m, k=k, n=n)
  tflops = float(2 * k * m * n) / (runtime / 1e3) / 1e12
  print(f"Kernel:    {runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
  print(f"Reference: {runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
