# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-specific Pallas primitives."""

from __future__ import annotations

from jax._src import core as jax_core
from jax._src import effects
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.lib.mlir.dialects import nvvm as nvvm_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import lowering
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
import jax.experimental.mosaic.gpu as mgpu


copy_smem_to_gmem_p = jax_core.Primitive("copy_smem_to_gmem")
copy_smem_to_gmem_p.multiple_results = True


@copy_smem_to_gmem_p.def_effectful_abstract_eval
def _copy_smem_to_gmem_abstract_eval(*avals, **params):
  del avals, params  # Unused.
  return (), {state.ReadEffect(0), state.WriteEffect(1)}


@lowering.register_lowering_rule(copy_smem_to_gmem_p)
def _copy_smem_to_gmem_lowering(
    ctx: lowering.LoweringRuleContext,
    src,
    dst,
    *flat_transforms,
    src_transforms_treedef,
    dst_transforms_treedef,
):
  flat_src_transforms, flat_dst_transforms = util.split_list(
      flat_transforms,
      [src_transforms_treedef.num_leaves],
  )
  src = lowering._handle_indexing(
      src, src_transforms_treedef.unflatten(flat_src_transforms)
  )
  copy_params = _extract_copy_params(
      dst_transforms_treedef.unflatten(flat_dst_transforms)
  )
  mgpu.warpgroup_barrier()
  ctx.launch_ctx.async_copy(src_ref=src, dst_ref=dst, **copy_params)
  return ()


def _extract_copy_params(transforms):
  if not transforms:
    return {}
  if any(
      isinstance(t, indexing.NDIndexer) for t in transforms[:-1]
  ) or not isinstance(transforms[-1], indexing.NDIndexer):
    raise NotImplementedError("Only one level of indexing supported")
  *transforms, indexer = transforms
  swizzle = lowering._is_swizzled(transforms)
  if swizzle is not None:
    transforms = transforms[1:]
  gpu_transforms = [t.to_gpu_transform() for t in transforms]
  return dict(
      gmem_slice=lowering._ndindexer_indices(indexer),
      gmem_transform=tuple(gpu_transforms),
      swizzle=swizzle,
  )


def copy_smem_to_gmem(
    src: pallas_core.AbstractMemoryRef, dst: pallas_core.AbstractMemoryRef
) -> None:
  """Asynchronously copies a SMEM reference to a GMEM reference.

  See also:
    :func:`jax.experimental.mosaic.gpu.wait_smem_to_gmem`
  """
  if src.memory_space is not gpu_core.SMEM:
    raise TypeError(f"src must be a SMEM reference, got {src.memory_space}")
  if dst.memory_space is not gpu_core.GMEM:
    raise ValueError(f"dst must be a GMEM reference, got {dst.memory_space}")
  src, src_transforms = state_primitives.get_ref_and_transforms(
      src, None, "copy_smem_to_gmem"
  )
  dst, dst_transforms = state_primitives.get_ref_and_transforms(
      dst, None, "copy_smem_to_gmem"
  )
  flat_src_transforms, src_transforms_treedef = tree_util.tree_flatten(
      src_transforms
  )
  flat_dst_transforms, dst_transforms_treedef = tree_util.tree_flatten(
      dst_transforms
  )
  copy_smem_to_gmem_p.bind(
      src,
      dst,
      *flat_src_transforms,
      *flat_dst_transforms,
      src_transforms_treedef=src_transforms_treedef,
      dst_transforms_treedef=dst_transforms_treedef,
  )
  return None


copy_gmem_to_smem_p = jax_core.Primitive("copy_gmem_to_smem")
copy_gmem_to_smem_p.multiple_results = True


@copy_gmem_to_smem_p.def_effectful_abstract_eval
def _copy_gmem_to_smem_abstract_eval(*avals, **params):
  del avals, params  # Unused.
  return (), {state.ReadEffect(0), state.WriteEffect(1)}


@lowering.register_lowering_rule(copy_gmem_to_smem_p)
def _copy_gmem_to_smem_lowering(
    ctx: lowering.LoweringRuleContext,
    src,
    dst,
    barrier,
    *flat_transforms,
    src_transforms_treedef,
    dst_transforms_treedef,
    barrier_transforms_treedef,
):
  flat_src_transforms, flat_dst_transforms, flat_barrier_transforms = (
      util.split_list(
          flat_transforms,
          [
              src_transforms_treedef.num_leaves,
              dst_transforms_treedef.num_leaves,
          ],
      )
  )
  copy_params = _extract_copy_params(
      src_transforms_treedef.unflatten(flat_src_transforms)
  )
  dst = lowering._handle_indexing(
      dst, dst_transforms_treedef.unflatten(flat_dst_transforms)
  )
  barrier_indexer = _extract_barrier_indexer(
      barrier_transforms_treedef.unflatten(flat_barrier_transforms)
  )
  if barrier_indexer is not None:
    barrier = barrier.__getitem__(
        *map(lowering._as_index, barrier_indexer.indices)
    )
  ctx.launch_ctx.async_copy(
      src_ref=src, dst_ref=dst, barrier=barrier, **copy_params
  )
  return ()


def copy_gmem_to_smem(
    src: pallas_core.AbstractMemoryRef,
    dst: pallas_core.AbstractMemoryRef,
    *,
    barrier: pallas_core.AbstractMemoryRef,
) -> None:
  """Asynchronously copies a GMEM reference to a SMEM reference.

  See also:
    :func:`jax.experimental.mosaic.gpu.wait_barrier`
  """
  if src.memory_space is not gpu_core.GMEM:
    raise TypeError(f"src must be a GMEM reference, got {src.memory_space}")
  if dst.memory_space is not gpu_core.SMEM:
    raise ValueError(f"dst must be a SMEM reference, got {dst.memory_space}")
  src, src_transforms = state_primitives.get_ref_and_transforms(
      src, None, "copy_gmem_to_smem"
  )
  dst, dst_transforms = state_primitives.get_ref_and_transforms(
      dst, None, "copy_gmem_to_smem"
  )
  flat_src_transforms, src_transforms_treedef = tree_util.tree_flatten(
      src_transforms
  )
  flat_dst_transforms, dst_transforms_treedef = tree_util.tree_flatten(
      dst_transforms
  )
  barrier, barrier_transforms = state_primitives.get_ref_and_transforms(
      barrier, None, "copy_gmem_to_smem"
  )
  flat_barrier_transforms, barrier_transforms_treedef = tree_util.tree_flatten(
      barrier_transforms
  )
  copy_gmem_to_smem_p.bind(
      src,
      dst,
      barrier,
      *flat_src_transforms,
      *flat_dst_transforms,
      *flat_barrier_transforms,
      src_transforms_treedef=src_transforms_treedef,
      dst_transforms_treedef=dst_transforms_treedef,
      barrier_transforms_treedef=barrier_transforms_treedef,
  )
  return None


def _extract_barrier_indexer(transforms) -> indexing.NDIndexer | None:
  if not transforms:
    return None
  match transforms:
    case [indexing.NDIndexer(indices=[idx]) as indexer]:
      if not isinstance(idx, indexing.Slice):
        return indexer
      if indexing.Slice.from_slice(slice(None), *indexer.shape) == idx:
        # Special-case: the whole slice.
        return None
      else:
        raise ValueError(
            f"Barrier can only be indexed with an integer, got {idx}"
        )
    case [indexing.NDIndexer()]:
      raise NotImplementedError("Barrier does not support multiple indices")
    case []:
      return None
    case _:
      raise ValueError("Barrier does not support arbirary transforms")


class WaitEffect(jax_core.Effect):
  ...

effects.control_flow_allowed_effects.add_type(WaitEffect)

_wait_effect = WaitEffect()


wait_barrier_p = jax_core.Primitive("wait")
wait_barrier_p.multiple_results = True


@wait_barrier_p.def_effectful_abstract_eval
def _wait_barrier_abstract_eval(*avals, **params):
  del avals, params  # Unused.
  return (), {_wait_effect}


@lowering.register_lowering_rule(wait_barrier_p)
def _wait_barrier_lowering(
    ctx: lowering.LoweringRuleContext,
    barrier,
    *flat_transforms,
    transforms_treedef,
):
  del ctx  # Unused.
  transforms = transforms_treedef.unflatten(flat_transforms)
  indexer = _extract_barrier_indexer(transforms)
  if indexer is not None:
    barrier = barrier.__getitem__(*map(lowering._as_index, indexer.indices))
  barrier.wait()
  return ()


def wait_barrier(barrier: pallas_core.AbstractMemoryRef) -> None:
  """Waits on the given barrier."""
  barrier, transforms = state_primitives.get_ref_and_transforms(
      barrier, None, "wait_barrier"
  )
  flat_transforms, transforms_treedef = tree_util.tree_flatten(transforms)
  wait_barrier_p.bind(
      barrier, *flat_transforms, transforms_treedef=transforms_treedef
  )


wait_smem_to_gmem_p = jax_core.Primitive("wait_smem_to_gmem")
wait_smem_to_gmem_p.multiple_results = True


@wait_smem_to_gmem_p.def_effectful_abstract_eval
def _wait_smem_to_gmem_abstract_eval(n):
  del n  # Unused.
  return (), {_wait_effect}


@lowering.register_lowering_rule(wait_smem_to_gmem_p)
def _wait_smem_to_gmem_lowering(ctx: lowering.LoweringRuleContext, n):
  ctx.launch_ctx.await_async_copy(allow_groups=n)
  return ()


def wait_smem_to_gmem(n: int) -> None:
  """Waits until there are no more than ``n`` SMEM->GMEM copies in flight."""
  wait_smem_to_gmem_p.bind(n)


class _WGMMAPipelineEffect(effects.Effect):
  pass


_wgmma_pipeline_effect = _WGMMAPipelineEffect()
effects.control_flow_allowed_effects.add_type(_WGMMAPipelineEffect)

# WGMMA on an accumulator reference
wgmma_ref_p = jax_core.Primitive("wgmma_ref")
wgmma_ref_p.multiple_results = True


def wgmma(
    acc: gpu_core.WGMMAAbstractAccumulatorRef,
    a: pallas_core.TransformedRef,
    b: pallas_core.TransformedRef,
) -> None:
  """Performs and asynchronous warp group matmul-accumulate on the given references.

  Conceptually, this is equivalent to doing ``acc[...] += a[...] @ b[...]``,
  except that the computation is performed asynchronously.

  Args:
    acc: The accumulator reference. Needs to be allocated via
      :func:`jax.experimental.pallas.run_scoped` called with a
      :func:`jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef`.
    a: The left hand side operand reference.
    b: The right hand side operand reference.

  See also:
    :func:`jax.experimental.pallas.mosaic_gpu.wgmma_wait`
  """
  m, n = acc.shape
  m2, k = a.shape
  k2, n2 = b.shape

  if m != m2 or n != n2 or k != k2:
    raise ValueError(
        f"Incompatible shapes for matrix multiplication: lhs={a.shape},"
        f" rhs={b.shape=}, acc={acc.shape}"
    )

  if (dtype := a.dtype) != b.dtype:
    raise ValueError(f"Mixed input dtypes for matrix multiplication unsupported: lhs={a.dtype}, rhs={b.dtype}")

  # Infer swizzle from a.
  if not a.transforms or not isinstance(
      (swizzle_transform := a.transforms[0]), gpu_core.UnswizzleRef
  ):
    raise ValueError("WGMMA lhs must be a tiled and swizzled reference.")

  swizzle = swizzle_transform.swizzle
  swizzle_elems = swizzle // dtype.itemsize
  if a.transforms[1:] != (gpu_core.UntileRef((64, swizzle_elems)),):
    raise ValueError(
        f"WGMMA lhs must be tiled with 64x{swizzle_elems} tiles for element type"
        f" {dtype}."
    )

  rhs_transpose_transform = gpu_core.TransposeRef((1, 0, 2, 3))
  rhs_tiling = gpu_core.UntileRef((swizzle_elems, swizzle_elems))
  if b.transforms == (swizzle_transform, rhs_tiling):
    rhs_transpose = False
  elif b.transforms == (swizzle_transform, rhs_transpose_transform, rhs_tiling):
    rhs_transpose = True
  else:
    raise ValueError(
        f"WGMMA rhs must have {swizzle=} and be tiled with"
        f" {swizzle_elems}x{swizzle_elems} tiles for element type {dtype} (and"
        " optionally transposed)."
    )

  wgmma_ref_p.bind(acc, a.ref, b.ref, swizzle=swizzle, rhs_transpose=rhs_transpose)


@wgmma_ref_p.def_effectful_abstract_eval
def _wgmma_ref_effectful_abstract_eval(acc_aval, a_aval, b_aval, **params):
  del a_aval, b_aval, params
  if not isinstance(acc_aval, gpu_core.WGMMAAbstractAccumulatorRef):
    raise TypeError(f"Expected WGMMAAbstractAccumulatorRef got {acc_aval}")
  return (), {
      _wgmma_pipeline_effect,
      state.WriteEffect(0),
      state.ReadEffect(0),
      state.ReadEffect(1),
      state.ReadEffect(2),
  }


@discharge.register_discharge_rule(wgmma_ref_p)
def _wgmma_ref_discharge(
    in_avals,
    out_avals,
    acc,
    a,
    b,
    swizzle,
    rhs_transpose,
):
  del in_avals, out_avals
  return (
      wgmma_p.bind(
          acc, a, b, swizzle=swizzle, rhs_transpose=rhs_transpose
      ),
      None,
      None,
  ), []


# Functional WGMMA, returns a shaped array. Internal.
wgmma_p = jax_core.Primitive("wgmma")


@lowering.register_lowering_rule(wgmma_p)
def _wgmma_lowering(
    ctx: lowering.LoweringRuleContext,
    acc,
    a,
    b,
    swizzle,
    rhs_transpose,
):
  del ctx
  new_acc = mgpu.wgmma(
      acc,
      a,
      b,
      swizzle=swizzle,
      b_order=mgpu.WGMMALayout.COL_MAJOR
      if rhs_transpose
      else mgpu.WGMMALayout.ROW_MAJOR,
  )
  nvvm_dialect.wgmma_commit_group_sync_aligned()
  return new_acc


@wgmma_p.def_effectful_abstract_eval
def _wgmma_effectful_abstract_eval(acc, *args, **kwargs):
  del args, kwargs
  return acc, {
      _wgmma_pipeline_effect,
      state.ReadEffect(1),
      state.ReadEffect(2),
  }

wgmma_wait_p = jax_core.Primitive("wgmma_wait")
wgmma_wait_p.multiple_results = True


def wgmma_wait(n: int):
  """Waits until there is no more than ``n`` WGMMA operations in flight."""
  return wgmma_wait_p.bind(n)


@wgmma_wait_p.def_effectful_abstract_eval
def wgmma_wait_effectful_abstract_eval(_):
  return [], {_wgmma_pipeline_effect}


@lowering.register_lowering_rule(wgmma_wait_p)
def _wgmma_wait_lowering(ctx: lowering.LoweringRuleContext, allow_groups):
  del ctx
  nvvm_dialect.wgmma_wait_group_sync_aligned(allow_groups)
  return ()


wgmma_accumulator_deref_p = jax_core.Primitive("wgmma_accumulator_deref_p")

def wgmma_accumulator_deref(acc):
  """Dereferences an accumulator register."""

  if not isinstance(acc.aval, gpu_core.WGMMAAbstractAccumulatorRef):
    raise TypeError(f"acc must be a WGMMAAccumulatorAbstractRef, got {acc.aval=}")

  return wgmma_accumulator_deref_p.bind(acc)

@wgmma_accumulator_deref_p.def_effectful_abstract_eval
def _wgmma_accumulator_deref_abstract_eval(acc):
  # Dereferencing implies flushing so we have a wgmma pipeline effect.
  ret = acc.inner_aval if isinstance(acc, gpu_core.WGMMAAbstractAccumulatorRef) else acc
  assert isinstance(ret, jax_core.ShapedArray), acc
  return ret, {_wgmma_pipeline_effect}


@discharge.register_discharge_rule(wgmma_accumulator_deref_p)
def _wgmma_accumulator_deref_discharge(in_avals, out_avals, acc):
  del in_avals, out_avals
  return (None,), wgmma_accumulator_deref_p.bind(acc)


@lowering.register_lowering_rule(wgmma_accumulator_deref_p)
def _wgmma_accumulator_deref_lowering(ctx: lowering.LoweringRuleContext, acc):
  del ctx
  nvvm_dialect.wgmma_wait_group_sync_aligned(0)
  return acc.value
