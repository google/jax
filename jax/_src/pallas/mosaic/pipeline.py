# Copyright 2023 The JAX Authors.
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
"""Module for emitting custom TPU pipelines within a Pallas call."""

import dataclasses
import functools
import math
from typing import Any, Callable, Generic, NamedTuple, Optional, Protocol, Sequence, TypeVar, Union, cast

import jax
from jax import lax
from jax import tree_util
from jax._src.api_util import flatten_axes
from jax._src.pallas import core
from jax._src.pallas import primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax._src.state import indexing
from jax._src.util import split_list
import jax.numpy as jnp

SMEM = tpu_core.TPUMemorySpace.SMEM
VMEM = tpu_core.TPUMemorySpace.VMEM
DMA = tpu_core.SemaphoreType.DMA
REF = tpu_core.MemoryRef

partial = functools.partial

T = TypeVar("T")


@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class PipelineArg(Generic[T]):
  """Wrapper for pipeline arguments that exist for inputs, outputs, and accums."""
  input: T
  out: T
  in_out: T

  @property
  def input_and_in_out(self) -> T:
    return cast(Any, self.input) + cast(Any, self.in_out)

  def tree_flatten(self):
    return ((self.input, self.out, self.in_out), None)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


class PipelineBuffer(NamedTuple):
  """Current and next buffer indices for an input/output/accum ref."""
  current: Union[REF, jax.Array]
  next: Union[REF, jax.Array]

# TODO(enriqueps): Add SMEM support.
class PipelineAllocation(NamedTuple):
  """Allocated VMEM ref and semaphore for an input/output/accum ref."""
  vmem_ref: REF
  semaphore: tpu_core.SemaphoreType

# PyTree versions of the various arguments.
PipelineBlockSpecs = Union[Sequence[core.BlockSpec], Any]
PipelineRefs = Union[Sequence[REF], Any]
PipelineBuffers = Union[Sequence[PipelineBuffer], Any]
PipelineAllocations = Union[Sequence[PipelineAllocation], Any]

GridIndices = tuple[jax.Array, ...]
CondVal = Union[jax.Array, bool]


def _broadcast_pytree_to(name: str, from_pytree: Any, to_pytree: Any) -> Any:
  """Broadcasts a prefix-pytree of to_pytree, to the shape of to_pytree.

  Useful for supporting passing in prefixes of things as arguments, like in
  jax.vmap.

  Args:
    name: Name for error messages.
    from_pytree: Prefix tree.
    to_pytree: Target pytree.

  Returns:
    Broadcasted pytree.
  """
  to_treedef = tree_util.tree_structure(to_pytree)
  return tree_util.tree_unflatten(
      to_treedef, flatten_axes(name, to_treedef, from_pytree)
  )


def _tree_map_with_kwargs(f, *args, **kwargs):
  """jax.tree_util.tree_map that supports kwargs."""
  kwargs_keys = kwargs.keys()
  kwargs_values = kwargs.values()
  return tree_util.tree_map(
      lambda arg0, partial_f, *args: partial_f(arg0, *args),
      args[0],
      tree_util.tree_map(
          lambda _, *tree_mapped_kwargs_values: partial(
              f, **dict(zip(kwargs_keys, tree_mapped_kwargs_values))
          ),
          args[0],
          *kwargs_values,
          is_leaf=lambda x: x is None,
      ),
      *args[1:],
  )


def _get_next_indices(grid: core.StaticGrid, indices: GridIndices) -> GridIndices:
  """Takes a grid and current indices and returns the next indices.

  grid: (3, 4, 5)
  indices: [1, 0, 4]
  returns: [1, 1, 0]

  Args:
    grid: Grid spec.
    indices: Current indices.

  Returns:
    Incremented indices.
  """
  next_indices = []
  carry = True
  for dim_size, index in reversed(list(zip(grid, indices))):
    i = jnp.where(carry, index + 1, index)
    carry = dim_size == i
    next_indices.append(jnp.where(carry, 0, i))
  return tuple(reversed(next_indices))


def _replace_nones_in_block_spec(block_spec: core.BlockSpec) -> core.BlockSpec:
  """Replaces Nones in a block spec shape with 1s."""
  block_shape = cast(tuple[int, ...], block_spec.block_shape)
  block_shape = tuple([1 if dim is None else dim for dim in block_shape])
  return dataclasses.replace(block_spec, block_shape=block_shape)


def _run_block_spec(
    block_spec: core.BlockSpec, indices: GridIndices
) -> tuple[Union[slice, indexing.Slice], ...]:
  """Runs a block spec for the given indices and returns the slices.

  Args:
    block_spec: Block spec to run.
    indices: Grid indices to run on.

  Returns:
    Array slices for the block spec.
  """
  index_map = block_spec.index_map
  if index_map is None:
    raise ValueError("Block spec index_map is None.")
  block_indices = index_map(*indices)
  return tuple(
      indexing.ds(
          primitives.multiple_of(index * block_size, block_size), block_size
      )
      for index, block_size in zip(
          block_indices, cast(Any, block_spec.block_shape)
      )
  )


def _dma_slice_not_equal(
    dma_slice_a: tuple[Union[slice, indexing.Slice], ...],
    dma_slice_b: tuple[Union[slice, indexing.Slice], ...],
) -> jax.Array:
  """Returns True if the two slices are not equal."""
  dma_slice_not_equal = cast(jax.Array, False)
  for a, b in zip(dma_slice_a, dma_slice_b):
    dma_slice_not_equal = jnp.logical_or(
        dma_slice_not_equal, a.start != b.start
    )
  return dma_slice_not_equal


def _block_copy(
    block_spec: core.BlockSpec,
    ref: REF,
    allocation: Optional[PipelineAllocation],
    buffers: PipelineBuffer,
    accum_allocation: Optional[PipelineAllocation] = None,
    accum_buffers: Optional[PipelineBuffer] = None,
    *,
    indices: tuple[
        GridIndices,
        GridIndices,
        GridIndices,
    ],
    is_input: bool,
    is_wait: bool,
    force_copy: Optional[Union[jax.Array, bool]] = None,
    force_skip: Optional[Union[jax.Array, bool]] = None,
    use_accum: Optional[Union[jax.Array, bool]] = None,
    accum_if_skipping: Optional[Union[jax.Array, bool]] = None,
    zero_accum_if_skipping: Optional[Union[jax.Array, bool]] = None,
):
  """General purpose input/output block copys.

  Basic flow:

  - Wait on input copy if previous block spec was different.
  - Start input copy if block spec is changing and it's not the last step.
  - Wait on output copy if previous block spec was different.
  - Start output copy if block spec is changing or is last step.

  The step constraints are enforced with force_copy and caller conds.

  Args:
    block_spec: Block spec.
    ref: HBM ref.
    allocation: VMEM ref and semaphore. If this is None it means the source refs
      are already in VMEM and we can avoid copy operations.
    buffers: Current and next buffer indices.
    accum_allocation: Accumulator VMEM ref and semaphore.
    accum_buffers: Accumulator current and next buffer indices.
    indices: Current grid indices.
    is_input: True if is input copy.
    is_wait: True if we want to wait on instead of start a copy.
    force_copy: Force copy if this condition is True. force_skip overrides this.
    force_skip: Force skipping the operation if this condition is True.
    use_accum: Whether to add the accum to the VMEM ref before copying.
    accum_if_skipping: Whether to accumulate into the current accum buffer if
      skipping copy.
    zero_accum_if_skipping: Whether to zero out the existing accum before
      accumulating into it if skipping copy.

  Returns:
    Current and next buffer indices, swapped if a copy was started.
  """
  if allocation is None:
    # Has existing allocation.
    return buffers
  (vmem_ref, sem) = allocation.vmem_ref, allocation.semaphore
  (prev_indices, curr_indices, next_indices) = indices

  prev_dma_slice = _run_block_spec(block_spec, prev_indices)
  dma_slice = _run_block_spec(block_spec, curr_indices)
  next_dma_slice = _run_block_spec(block_spec, next_indices)

  prev_dma_slice_changed = _dma_slice_not_equal(prev_dma_slice, dma_slice)
  dma_slice_is_changing = _dma_slice_not_equal(dma_slice, next_dma_slice)

  buffer, next_buffer = buffers.current, buffers.next
  if is_input:
    if is_wait:
      # We wait for inputs of the current body iteration.
      used_dma_slice = dma_slice
      used_buffer = buffer
    else:
      # We send to the next ones.
      used_dma_slice = next_dma_slice
      used_buffer = next_buffer
  else:
    if is_wait:
      # We wait for the outputs of the previous body iteration.
      used_dma_slice = prev_dma_slice
      used_buffer = next_buffer
    else:
      # We send the current ones.
      used_dma_slice = dma_slice
      used_buffer = buffer

  if is_input:
    from_ref = ref.at[used_dma_slice]
    to_ref = vmem_ref.at[used_buffer]
  else:
    from_ref = vmem_ref.at[used_buffer]
    to_ref = ref.at[used_dma_slice]

  async_copy = tpu_primitives.make_async_copy(
      from_ref,
      to_ref,
      sem,
  )

  if is_wait:
    cond = prev_dma_slice_changed
    do_fn = async_copy.wait
    advance_buffers = False
  else:
    cond = dma_slice_is_changing
    do_fn = async_copy.start
    advance_buffers = True

  if force_copy is not None:
    cond = jnp.logical_or(cond, force_copy)
  if force_skip is not None:
    cond = jnp.logical_and(cond, jnp.logical_not(force_skip))

  def do_and_advance_buffers():
    if accum_allocation is not None:

      def accum():
        with tpu_primitives.trace("ep_accum_copy"):
          accum_dtype = jnp.float32
          if vmem_ref.dtype == jnp.int32:
            accum_dtype = jnp.int32
          accum_vmem_ref = accum_allocation.vmem_ref
          vmem_ref[used_buffer] = (
              vmem_ref[used_buffer].astype(accum_dtype)
              + accum_vmem_ref[accum_buffers.current].astype(accum_dtype)
          ).astype(vmem_ref.dtype)

      lax.cond(use_accum, accum, lambda: None)

    do_fn()
    if advance_buffers:
      return PipelineBuffer(next_buffer, buffer)
    return buffers

  def dont_advance_buffers():
    if accum_allocation is not None:

      def accum():
        with tpu_primitives.trace("ep_accum_store"):

          def zero_accum():
            accum_vmem_ref = accum_allocation.vmem_ref
            accum_vmem_ref[...] = jnp.zeros_like(accum_vmem_ref[...])

          lax.cond(zero_accum_if_skipping, zero_accum, lambda: None)

          accum_dtype = jnp.float32
          if vmem_ref.dtype == jnp.int32:
            accum_dtype = jnp.int32
          accum_vmem_ref = accum_allocation.vmem_ref
          accum_vmem_ref[accum_buffers.current] = (
              accum_vmem_ref[accum_buffers.current].astype(accum_dtype)
              + vmem_ref[used_buffer].astype(accum_dtype)
          ).astype(accum_vmem_ref.dtype)

      lax.cond(accum_if_skipping, accum, lambda: None)

    return buffers

  return lax.cond(cond, do_and_advance_buffers, dont_advance_buffers)


# Start copying an input's next block to its next buffer.
_start_block_copy_in = partial(_block_copy, is_input=True, is_wait=False)
# Wait for the copy of an input's current block to its current buffer.
_wait_block_copy_in = partial(_block_copy, is_input=True, is_wait=True)
# Start copying an output's current block from its current buffer.
_start_block_copy_out = partial(_block_copy, is_input=False, is_wait=False)
# Wait for the copy of an output's previous block from its previous buffer.
_wait_block_copy_out = partial(_block_copy, is_input=False, is_wait=True)


class PipelineBody(Protocol):
  """Body of a pipeline."""

  def __call__(self, *ref_args: PipelineRefs) -> None:
    ...


class MakePipelineRefs(Protocol):
  """Makes pipeline refs from flat user friendly function args."""

  def __call__(self, *ref_args: PipelineRefs) -> PipelineArg[PipelineRefs]:
    ...


class MakePipelineAllocations(Protocol):
  """Makes pipeline allocations from flat user friendly function args."""

  def __call__(
      self, *ref_args: PipelineRefs, return_treedef: bool = False
  ) -> Any:
    ...


@dataclasses.dataclass(frozen=True)
class PipelinePrefetchArgs:
  """Args for pipeline prefetch."""
  pipeline_refs: PipelineArg[PipelineRefs]
  pipeline_allocations: PipelineArg[PipelineAllocations]
  pipeline_buffers: PipelineArg[PipelineBuffers]


class StartPipelinePrefetch(Protocol):
  """Starts pipeline prefetch.

  Use force_copy if a spec's indices don't change from last to first grid
  indices and you still want to force a copy. This must be used in conjunction
  with the prologue's return value to force a wait.
  """

  def __call__(
      self,
      prefetch_args: PipelinePrefetchArgs,
      *,
      force_copy: Union[
          bool, tuple[Union[CondVal, Any], Union[CondVal, Any]]
      ] = False,
  ) -> PipelineArg[PipelineBuffers]:
    ...


@dataclasses.dataclass(frozen=True)
class PipelineCallbackArgs:
  """Args for pipeline prologue and epilogue."""
  pipeline_specs: PipelineArg[PipelineBlockSpecs]
  pipeline_refs: PipelineArg[PipelineRefs]
  pipeline_buffer_refs: PipelineArg[PipelineBuffers]
  pipeline_allocations: PipelineArg[PipelineAllocations]
  pipeline_buffers: PipelineArg[PipelineBuffers]
  make_pipeline_refs: MakePipelineRefs
  start_pipeline_prefetch: StartPipelinePrefetch


PipelinePrologue = Callable[
    [PipelineCallbackArgs],
    # Returns a tuple of tuples of prefix-pytrees for inputs and accums. The
    # first specifies which ones to skip the prologue input copy for and the
    # second specifies which ones to force the prologue input wait on.
    tuple[
        tuple[Union[CondVal, Any], Union[CondVal, Any]],
        tuple[Union[CondVal, Any], Union[CondVal, Any]],
    ],
]
PipelineEpilogue = Callable[
    [PipelineCallbackArgs], tuple[PipelineBuffers, PipelineBuffers]
]


class Pipeline(Protocol):

  def __call__(
      self,
      *ref_args: PipelineRefs,
      scratchs: PipelineRefs = None,
      allocations: Union[None, Any] = None,
      init_allocations: CondVal = False,
      prologue: Union[PipelinePrologue, None] = None,
      epilogue: Union[PipelineEpilogue, None] = None,
      out_prologue: Union[PipelinePrologue, None] = None,
      out_epilogue: Union[PipelineEpilogue, None] = None,
  ) -> None:
    ...


def emit_pipeline_with_allocations(
    body: PipelineBody,
    *,
    grid: core.StaticGrid,
    in_specs: PipelineBlockSpecs,
    out_specs: PipelineBlockSpecs,
    should_accumulate_out: Union[Sequence[bool], Any] = False,
) -> tuple[Pipeline, MakePipelineAllocations]:
  """Wraps body function in a custom pipeline defined by grid and specs.

  This has the same semantics as pallas_call but is meant to be called inside
  pallas_call for nesting grids. This is useful when you need to have separate
  windowing strategies for example for communication vs. computation.

  By default outputs are written to but `should_accumulate_out` can be used to
  specify which outputs we should add to instead. This is so we can reduce
  across pipeline calls within and across parent grid iterations.

  This is like `pltpu.emit_pipeline` but also returns a function for creating
  the allocation descriptors for the pipeline so they can be allocated at a
  parent grid and passed in so they live across the parent grid's iterations.

  Args:
    body: Pipeline body.
    grid: Pallas grid.
    in_specs: Input block specs.
    out_specs: Output block specs.
    should_accumulate_out: Prefix-pytree of out_specs specifying which should be
      accumulated into with True.

  Returns:
    Tuple of wrapped pipelined body and a function to create the allocation
    descriptors.
  """
  if not isinstance(in_specs, (list, tuple)):
    in_specs = [in_specs]
  if not isinstance(out_specs, (list, tuple)):
    out_specs = [out_specs]
  should_accumulate_out = _broadcast_pytree_to(
      "should_accumulate_out", should_accumulate_out, out_specs
  )
  in_out_specs = tree_util.tree_map(
      lambda spec, accum: spec if accum else None,
      out_specs,
      should_accumulate_out,
  )
  pipeline_specs: PipelineArg[PipelineBlockSpecs] = PipelineArg(
      in_specs, out_specs, in_out_specs
  )
  del in_specs, out_specs, should_accumulate_out, in_out_specs
  pipeline_specs_with_nones = pipeline_specs
  pipeline_specs = jax.tree_util.tree_map(
      _replace_nones_in_block_spec, pipeline_specs_with_nones
  )

  def make_pipeline_refs(
      *ref_args: PipelineRefs,
  ) -> PipelineArg[PipelineRefs]:
    in_refs, out_refs = split_list(ref_args, [len(pipeline_specs.input)])
    return PipelineArg(in_refs, out_refs, out_refs)

  def start_pipeline_prefetch(
      prefetch_args: PipelinePrefetchArgs,
      *,
      indices: GridIndices,
      force_copy: Union[
          bool, tuple[Union[CondVal, Any], Union[CondVal, Any]]
      ] = False,
  ) -> tuple[PipelineBuffers, PipelineBuffers]:
    if isinstance(force_copy, bool):
      next_in_and_in_out_buffers = tree_util.tree_map(
          partial(_start_block_copy_in, indices=indices, force_copy=force_copy),
          pipeline_specs.input_and_in_out,
          prefetch_args.pipeline_refs.input_and_in_out,
          prefetch_args.pipeline_allocations.input_and_in_out,
          prefetch_args.pipeline_buffers.input_and_in_out,
      )
    else:
      force_input_copy, force_in_out_copy = force_copy
      force_input_copy = _broadcast_pytree_to(
          "force_input_copy",
          force_input_copy,
          pipeline_specs.input,
      )
      force_in_out_copy = _broadcast_pytree_to(
          "force_in_out_copy",
          force_in_out_copy,
          pipeline_specs.out,
      )
      next_in_and_in_out_buffers = _tree_map_with_kwargs(
          partial(_start_block_copy_in, indices=indices),
          pipeline_specs.input_and_in_out,
          prefetch_args.pipeline_refs.input_and_in_out,
          prefetch_args.pipeline_allocations.input_and_in_out,
          prefetch_args.pipeline_buffers.input_and_in_out,
          force_copy=force_input_copy + force_in_out_copy,
      )
    next_in_buffers, next_in_out_buffers = split_list(
        next_in_and_in_out_buffers, [len(pipeline_specs.input)]
    )
    return next_in_buffers, next_in_out_buffers

  def make_pipeline_allocations(
      *ref_args: PipelineRefs,
      return_treedef: bool = False,
  ) -> tuple[tuple[Any, tree_util.PyTreeDef], Any]:
    pipeline_buffers = tree_util.tree_map(
        lambda _: PipelineBuffer(*((SMEM((1,), jnp.int32),) * 2)),
        pipeline_specs,
    )
    pipeline_refs = make_pipeline_refs(*ref_args)

    def make_allocation(spec, ref):
      if ref.memory_space == VMEM:
        # Don't make an allocation the ref is already in VMEM, we can use it
        # directly for free.
        return None
      return PipelineAllocation(
          VMEM((2, *spec.block_shape), getattr(ref, "dtype", ref)),
          DMA,
      )

    pipeline_allocations = tree_util.tree_map(
        make_allocation, pipeline_specs, pipeline_refs
    )

    def grab_allocation(_, ref):
      if ref.memory_space == VMEM:
        return ref
      return None

    pipeline_existing_allocations = tree_util.tree_map(
        grab_allocation, pipeline_specs, pipeline_refs
    )

    def make_in_out_existing_allocations(spec, ref):
      if ref.memory_space == VMEM:
        return VMEM(spec.block_shape, getattr(ref, "dtype", ref))
      return None

    in_out_existing_allocations = tree_util.tree_map(
        make_in_out_existing_allocations,
        pipeline_specs.in_out,
        pipeline_refs.in_out,
    )

    flat_allocations, allocations_treedef = tree_util.tree_flatten((
        pipeline_buffers,
        pipeline_allocations,
        in_out_existing_allocations,
    ))
    if return_treedef:
      flat_allocations = cast(Any, (tuple(flat_allocations), allocations_treedef))
    return (
        (flat_allocations, allocations_treedef),
        pipeline_existing_allocations,
    )

  def pipeline(
      *ref_args: PipelineRefs,
      scratchs: Union[PipelineRefs, None] = None,
      allocations: Union[
          None,
          tuple[PipelineArg[PipelineBuffers], PipelineArg[PipelineAllocations]],
      ] = None,
      init_allocations: CondVal = False,
      prologue: Union[PipelinePrologue, None] = None,
      epilogue: Union[PipelineEpilogue, None] = None,
      out_prologue: Union[PipelinePrologue, None] = None,
      out_epilogue: Union[PipelineEpilogue, None] = None,
  ) -> None:
    use_in_out = jnp.logical_not(init_allocations)
    if scratchs is None:
      scratchs = []
    if not isinstance(scratchs, (list, tuple)):
      scratchs = [scratchs]

    def pipeline_body(
        pipeline_refs: PipelineArg[PipelineRefs],
        pipeline_existing_allocations: PipelineArg[PipelineRefs],
        pipeline_buffer_refs: PipelineArg[PipelineBuffers],
        pipeline_allocations: PipelineArg[PipelineAllocations],
        in_out_existing_allocations: PipelineRefs,
    ):

      def init_pipeline_allocations():
        def init_buffer_ref(_, buffer_ref):
          buffer_ref.current[0] = 0
          buffer_ref.next[0] = 1

        tree_util.tree_map(
            init_buffer_ref,
            pipeline_specs,
            pipeline_buffer_refs,
        )

      do_init_pipeline_allocations = jnp.logical_or(
          allocations is None, init_allocations
      )
      lax.cond(
          do_init_pipeline_allocations,
          init_pipeline_allocations,
          lambda: None,
      )

      zero_indices = (jnp.array(0, dtype=jnp.int32),) * len(grid)
      last_indices = tuple(
          [jnp.asarray(dim_size - 1, dtype=jnp.int32) for dim_size in grid]
      )
      indices = zero_indices
      pipeline_buffers: PipelineArg[PipelineBuffers] = tree_util.tree_map(
          lambda buffer_ref: buffer_ref[0],
          pipeline_buffer_refs,
      )
      if prologue is not None:
        (skip_input_prologue, skip_in_out_prologue), (
            force_input_prologue_wait,
            force_in_out_prologue_wait,
        ) = prologue(
            PipelineCallbackArgs(
                pipeline_specs=pipeline_specs,
                pipeline_refs=pipeline_refs,
                pipeline_buffer_refs=pipeline_buffer_refs,
                pipeline_allocations=pipeline_allocations,
                pipeline_buffers=pipeline_buffers,
                make_pipeline_refs=make_pipeline_refs,
                start_pipeline_prefetch=partial(
                    cast(Any, start_pipeline_prefetch),
                    indices=(zero_indices, last_indices, indices),
                    force_copy=True,
                ),
            )
        )
      else:
        skip_input_prologue = None
        skip_in_out_prologue = None
        force_input_prologue_wait = None
        force_in_out_prologue_wait = None
      skip_input_prologue = _broadcast_pytree_to(
          "skip_input_prologue",
          skip_input_prologue,
          pipeline_specs.input,
      )
      skip_in_out_prologue = _broadcast_pytree_to(
          "skip_in_out_prologue",
          skip_in_out_prologue,
          pipeline_specs.out,
      )
      force_input_prologue_wait = _broadcast_pytree_to(
          "force_input_prologue_wait",
          force_input_prologue_wait,
          pipeline_specs.input,
      )
      force_in_out_prologue_wait = _broadcast_pytree_to(
          "force_in_out_prologue_wait",
          force_in_out_prologue_wait,
          pipeline_specs.out,
      )
      _tree_map_with_kwargs(
          partial(
              _start_block_copy_in,
              indices=(indices, indices, indices),
              force_copy=True,
          ),
          pipeline_specs.input,
          pipeline_refs.input,
          pipeline_allocations.input,
          tree_util.tree_map(
              lambda _, buffers: PipelineBuffer(buffers.next, buffers.current),
              pipeline_specs.input,
              pipeline_buffers.input,
          ),
          force_skip=skip_input_prologue,
      )
      lax.cond(
          use_in_out,
          lambda: _tree_map_with_kwargs(
              partial(
                  _start_block_copy_in,
                  indices=(indices, indices, indices),
                  force_copy=True,
              ),
              pipeline_specs.in_out,
              pipeline_refs.in_out,
              pipeline_allocations.in_out,
              tree_util.tree_map(
                  lambda _, buffers: PipelineBuffer(
                      buffers.next, buffers.current
                  ),
                  pipeline_specs.in_out,
                  pipeline_buffers.in_out,
              ),
              force_skip=skip_in_out_prologue,
          ),
          lambda: pipeline_buffers.in_out,
      )
      total_iterations = math.prod(grid)

      def fori_loop_body(
          i: jax.Array,
          carry: tuple[
              GridIndices,
              GridIndices,
              PipelineArg[PipelineBuffers],
          ],
      ) -> tuple[
          GridIndices,
          GridIndices,
          PipelineArg[PipelineBuffers],
      ]:
        (prev_indices, indices, pipeline_buffers) = carry
        next_indices = _get_next_indices(grid, indices)
        copy_indices = (prev_indices, indices, next_indices)

        with tpu_primitives.trace("ep_wait_input"):
          input_copy_args = [
              pipeline_specs.input,
              pipeline_refs.input,
              pipeline_allocations.input,
              pipeline_buffers.input,
          ]
          in_out_copy_args = [
              pipeline_specs.in_out,
              pipeline_refs.in_out,
              pipeline_allocations.in_out,
              pipeline_buffers.in_out,
          ]
          input_force_copy = lambda skip, wait: jnp.logical_and(
              i == 0, jnp.logical_or(jnp.logical_not(skip), wait)
          )
          _tree_map_with_kwargs(
              partial(
                  _wait_block_copy_in,
                  indices=copy_indices,
              ),
              *input_copy_args,
              force_copy=tree_util.tree_map(
                  input_force_copy,
                  skip_input_prologue,
                  force_input_prologue_wait,
              ),
          )
          lax.cond(
              use_in_out,
              lambda: _tree_map_with_kwargs(
                  partial(
                      _wait_block_copy_in,
                      indices=copy_indices,
                  ),
                  *in_out_copy_args,
                  force_copy=tree_util.tree_map(
                      input_force_copy,
                      skip_in_out_prologue,
                      force_in_out_prologue_wait,
                  ),
              ),
              lambda: pipeline_buffers.in_out,
          )

          def start_next_iteration_in_block_copies():
            next_in_buffers = tree_util.tree_map(
                partial(
                    _start_block_copy_in,
                    indices=copy_indices,
                ),
                *input_copy_args,
            )
            next_in_out_buffers = lax.cond(
                use_in_out,
                lambda: tree_util.tree_map(
                    partial(
                        _start_block_copy_in,
                        indices=copy_indices,
                    ),
                    *in_out_copy_args,
                ),
                lambda: pipeline_buffers.in_out,
            )
            return next_in_buffers, next_in_out_buffers

          @tpu_primitives.trace("ep_run_epilogue")
          def run_epilogue():
            if epilogue is None:
              return pipeline_buffers.input, pipeline_buffers.in_out
            return epilogue(
                PipelineCallbackArgs(
                    pipeline_specs=pipeline_specs,
                    pipeline_refs=pipeline_refs,
                    pipeline_buffer_refs=pipeline_buffer_refs,
                    pipeline_allocations=pipeline_allocations,
                    pipeline_buffers=pipeline_buffers,
                    make_pipeline_refs=make_pipeline_refs,
                    start_pipeline_prefetch=partial(
                        cast(Any, start_pipeline_prefetch),
                        indices=(prev_indices, indices, zero_indices),
                    ),
                )
            )

          next_in_buffers, next_in_out_buffers = lax.cond(
              i == total_iterations - 1,
              run_epilogue,
              start_next_iteration_in_block_copies,
          )

        with tpu_primitives.trace("ep_kernel"):

          def grab_body_ref(
              spec_with_nones,
              spec,
              allocation,
              buffers,
              existing_allocation,
              in_out_existing_allocation=None,
          ):
            if existing_allocation is None:
              buffer_slice = tuple([
                  0 if dim is None else slice(None)
                  for dim in spec_with_nones.block_shape
              ])
              return allocation.vmem_ref.at[(buffers.current, *buffer_slice)]
            dma_slice = _run_block_spec(spec, indices)
            dma_slice = tuple([
                0 if dim is None else _slice
                for dim, _slice in zip(spec_with_nones.block_shape, dma_slice)
            ])
            if in_out_existing_allocation is None:
              return existing_allocation.at[dma_slice]
            return in_out_existing_allocation.at[dma_slice]

          in_args = tree_util.tree_map(
              grab_body_ref,
              pipeline_specs_with_nones.input,
              pipeline_specs.input,
              pipeline_allocations.input,
              pipeline_buffers.input,
              pipeline_existing_allocations.input,
          )
          out_args = tree_util.tree_map(
              grab_body_ref,
              pipeline_specs_with_nones.out,
              pipeline_specs.out,
              pipeline_allocations.out,
              pipeline_buffers.out,
              pipeline_existing_allocations.out,
              in_out_existing_allocations,
          )
          with core.grid_env(cast(Any, zip(indices, grid))):
            body(*in_args, *out_args, *scratchs)

          def accum_existing_in_out_existing_allocation(
              spec,
              existing_allocation,
              in_out_existing_allocation,
          ):
            if (
                existing_allocation is not None
                and in_out_existing_allocation is not None
            ):
              dma_slice = _run_block_spec(spec, indices)
              next_dma_slice = _run_block_spec(spec, next_indices)
              dma_slice_is_changing = _dma_slice_not_equal(
                  dma_slice, next_dma_slice
              )

              def init():
                existing_allocation[dma_slice] = in_out_existing_allocation[
                    dma_slice
                ]

              def accum():
                existing_allocation[dma_slice] = (
                    existing_allocation[dma_slice].astype(jnp.float32)
                    + in_out_existing_allocation[dma_slice].astype(jnp.float32)
                ).astype(existing_allocation.dtype)

              lax.cond(
                  jnp.logical_or(
                      dma_slice_is_changing, i == total_iterations - 1
                  ),
                  lambda: lax.cond(use_in_out, accum, init),
                  lambda: None,
              )

          tree_util.tree_map(
              accum_existing_in_out_existing_allocation,
              pipeline_specs.out,
              pipeline_existing_allocations.out,
              in_out_existing_allocations,
          )

        with tpu_primitives.trace("ep_wait_output"):

          def run_out_prologue():
            if out_prologue is not None:
              skip_out_prologue_wait = out_prologue(
                  PipelineCallbackArgs(
                      pipeline_specs=pipeline_specs,
                      pipeline_refs=pipeline_refs,
                      pipeline_buffer_refs=pipeline_buffer_refs,
                      pipeline_allocations=pipeline_allocations,
                      pipeline_buffers=pipeline_buffers,
                      make_pipeline_refs=make_pipeline_refs,
                      start_pipeline_prefetch=partial(
                          cast(Any, start_pipeline_prefetch),
                          indices=copy_indices,
                          force_copy=True,
                      ),
                  )
              )
              skip_out_prologue_wait = _broadcast_pytree_to(
                  "skip_out_prologue_wait",
                  skip_out_prologue_wait,
                  pipeline_specs.out,
              )
              _tree_map_with_kwargs(
                  partial(
                      _wait_block_copy_out,
                      indices=copy_indices,
                  ),
                  pipeline_specs.out,
                  pipeline_refs.out,
                  pipeline_allocations.out,
                  pipeline_buffers.out,
                  force_skip=skip_out_prologue_wait,
              )

          @tpu_primitives.trace("ep_wait_prev_iteration_out_block_copies")
          def wait_prev_iteration_out_block_copies():
            tree_util.tree_map(
                partial(
                    _wait_block_copy_out,
                    indices=copy_indices,
                ),
                pipeline_specs.out,
                pipeline_refs.out,
                pipeline_allocations.out,
                pipeline_buffers.out,
            )

          lax.cond(
              i == 0,
              run_out_prologue,
              wait_prev_iteration_out_block_copies,
          )
          if out_epilogue is not None:
            skip_out_epilogue_wait = out_epilogue(
                PipelineCallbackArgs(
                    pipeline_specs=pipeline_specs,
                    pipeline_refs=pipeline_refs,
                    pipeline_buffer_refs=pipeline_buffer_refs,
                    pipeline_allocations=pipeline_allocations,
                    pipeline_buffers=pipeline_buffers,
                    make_pipeline_refs=make_pipeline_refs,
                    start_pipeline_prefetch=cast(Any, lambda *args, **kwargs: None),
                )
            )
          else:
            skip_out_epilogue_wait = cast(Any, False)
          skip_out_epilogue_wait = _broadcast_pytree_to(
              "skip_out_epilogue_wait",
              skip_out_epilogue_wait,
              pipeline_specs.out,
          )
          force_start_block_copy_out = jax.tree_util.tree_map(
              lambda skip_out_wait: jnp.logical_and(
                  jnp.logical_not(skip_out_wait), i == total_iterations - 1
              ),
              skip_out_epilogue_wait,
          )
          next_out_buffers = _tree_map_with_kwargs(
              partial(
                  _start_block_copy_out,
                  indices=copy_indices,
                  use_accum=use_in_out,
                  # If an output tile doesn't change from last to first, we need
                  # to add its accum since the body overwrites outputs each
                  # pipeline.
                  accum_if_skipping=i == total_iterations - 1,
                  # Initialize the accum if this is the first time this is
                  # happening.
                  zero_accum_if_skipping=do_init_pipeline_allocations,
              ),
              pipeline_specs.out,
              pipeline_refs.out,
              pipeline_allocations.out,
              pipeline_buffers.out,
              pipeline_allocations.in_out,
              pipeline_buffers.in_out,
              force_copy=force_start_block_copy_out,
          )

        prev_indices = indices
        indices = next_indices
        return (
            prev_indices,
            indices,
            PipelineArg(next_in_buffers, next_out_buffers, next_in_out_buffers),
        )

      (prev_indices, indices, pipeline_buffers) = lax.fori_loop(
          0,
          total_iterations,
          fori_loop_body,
          (last_indices, indices, pipeline_buffers),
      )

      def set_buffer_ref(buffer_ref, buffer):
        buffer_ref[0] = buffer

      tree_util.tree_map(
          set_buffer_ref,
          pipeline_buffer_refs,
          pipeline_buffers,
      )

      with tpu_primitives.trace("ep_end_pipeline"):
        with tpu_primitives.trace("ep_wait_output"):
          if out_epilogue is not None:
            skip_out_epilogue_wait = out_epilogue(
                PipelineCallbackArgs(
                    pipeline_specs=pipeline_specs,
                    pipeline_refs=pipeline_refs,
                    pipeline_buffer_refs=pipeline_buffer_refs,
                    pipeline_allocations=pipeline_allocations,
                    pipeline_buffers=pipeline_buffers,
                    make_pipeline_refs=make_pipeline_refs,
                    start_pipeline_prefetch=partial(
                        cast(Any, start_pipeline_prefetch),
                        indices=(zero_indices, zero_indices, zero_indices),
                        force_copy=True,
                    ),
                )
            )
          else:
            skip_out_epilogue_wait = None
          skip_out_epilogue_wait = _broadcast_pytree_to(
              "skip_out_epilogue_wait",
              skip_out_epilogue_wait,
              pipeline_specs.out,
          )
          _tree_map_with_kwargs(
              partial(
                  _wait_block_copy_out,
                  indices=(prev_indices, indices, zero_indices),
                  force_copy=True,
              ),
              pipeline_specs.out,
              pipeline_refs.out,
              pipeline_allocations.out,
              pipeline_buffers.out,
              force_skip=skip_out_epilogue_wait,
          )

    pipeline_refs = make_pipeline_refs(*ref_args)
    if allocations is None:
      (flat_allocations, allocations_treedef), existing_allocations = (
          make_pipeline_allocations(*ref_args)
      )
      tpu_primitives.run_scoped(
          partial(pipeline_body, pipeline_refs, existing_allocations),
          *tree_util.tree_unflatten(allocations_treedef, flat_allocations),
      )
    else:
      (_, allocations_treedef), existing_allocations = (
          make_pipeline_allocations(*ref_args)
      )
      pipeline_body(
          pipeline_refs,
          existing_allocations,
          *tree_util.tree_unflatten(allocations_treedef, list(allocations)),
      )

  return pipeline, lambda *args, **kwargs: tuple(
      make_pipeline_allocations(*args, **kwargs)[0][0]
  )


def emit_pipeline(
    body: PipelineBody,
    *,
    grid: core.StaticGrid,
    in_specs: PipelineBlockSpecs,
    out_specs: PipelineBlockSpecs,
    should_accumulate_out: Union[Sequence[bool], Any] = False,
) -> Pipeline:
  """Wraps body function in a custom pipeline defined by grid and specs.

  This has the same semantics as pallas_call but is meant to be called inside
  pallas_call for nesting grids. This is useful when you need to have separate
  windowing strategies for example for communication vs. computation.

  By default outputs are written to but `should_accumulate_out` can be used to
  specify which outputs we should add to instead. This is so we can reduce
  across pipeline calls within and across parent grid iterations.

  Args:
    body: Pipeline body.
    grid: Pallas grid.
    in_specs: Input block specs.
    out_specs: Output block specs.
    should_accumulate_out: Prefix-pytree of out_specs specifying which should be
      accumulated into with True.

  Returns:
    Wrapped pipelined body.
  """
  return emit_pipeline_with_allocations(
      body,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      should_accumulate_out=should_accumulate_out,
  )[0]
