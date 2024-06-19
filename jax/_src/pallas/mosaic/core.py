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

"""Contains TPU-specific Pallas abstractions."""
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import enum
import functools
from typing import Any

from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import tree_util
from jax._src import util
import jax.numpy as jnp
from jax._src.pallas import core as pallas_core

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

partial = functools.partial
Grid = pallas_core.Grid
BlockSpec = pallas_core.BlockSpec
BlockSpecTree = pallas_core.BlockSpecTree
GridMapping = pallas_core.GridMapping
NoBlockSpec = pallas_core.NoBlockSpec
AbstractMemoryRef = pallas_core.AbstractMemoryRef
no_block_spec = pallas_core.no_block_spec
_convert_block_spec_to_block_mapping = pallas_core._convert_block_spec_to_block_mapping
split_list = util.split_list


class TPUMemorySpace(enum.Enum):
  ANY = "any"
  VMEM = "vmem"
  SMEM = "smem"
  CMEM = "cmem"
  SEMAPHORE = "semaphore_mem"

  def __str__(self) -> str:
    return self.value

  def __call__(self, shape: tuple[int, ...], dtype: jnp.dtype):
    # A convenience function for constructing MemoryRef types.
    return MemoryRef(shape, dtype, self)

class semaphore_dtype(dtypes.extended): pass
class semaphore(semaphore_dtype): pass
class dma_semaphore(semaphore_dtype): pass
class barrier_semaphore(semaphore_dtype): pass

class AbstractSemaphoreTyRules:
  @staticmethod
  def pallas_interpret_element_aval(_) -> jax_core.ShapedArray:
    return jax_core.ShapedArray((), jnp.dtype('int32'))

class AbstractSemaphoreTy(dtypes.ExtendedDType):
  name: str
  _rules = AbstractSemaphoreTyRules

  def __repr__(self) -> str:
    return self.name

  def __eq__(self, other):
    return self.__class__ == other.__class__

  def __hash__(self) -> int:
    return hash((self.__class__))

# TODO(sharadmv): implement dtype rules for AbstractSemaphoreTy

class SemaphoreTy(AbstractSemaphoreTy):
  type = semaphore
  name = "sem"

class DmaSemaphoreTy(AbstractSemaphoreTy):
  type = dma_semaphore
  name = "dma_sem"

class BarrierSemaphoreTy(AbstractSemaphoreTy):
  type = barrier_semaphore
  name = "barrier_sem"

class SemaphoreType(enum.Enum):
  REGULAR = "regular"
  DMA = "dma"
  BARRIER = "barrier"

  def __call__(self, shape: tuple[int, ...]):
    dtype: Any
    if self == SemaphoreType.DMA:
      dtype = DmaSemaphoreTy()
    elif self == SemaphoreType.BARRIER:
      dtype = BarrierSemaphoreTy()
    else:
      dtype = SemaphoreTy()
    return MemoryRef(shape, dtype, TPUMemorySpace.SEMAPHORE)

  def get_aval(self) -> "AbstractMemoryRef":
    return self(()).get_aval()

@dataclasses.dataclass(frozen=True)
class AbstractSemaphore(jax_core.AbstractValue):
  sem_type: SemaphoreType

  def join(self, other):
    if not isinstance(other, AbstractSemaphore):
      raise ValueError
    if other.sem_type != self.sem_type:
      raise ValueError
    return self

jax_core.raise_to_shaped_mappings[AbstractSemaphore] = lambda aval, _: aval


@dataclasses.dataclass(frozen=True)
class MemoryRef:
  """Like jax.ShapeDtypeStruct but with memory spaces."""
  shape: tuple[int, ...]
  dtype: jnp.dtype
  memory_space: TPUMemorySpace = TPUMemorySpace.ANY

  def get_aval(self) -> AbstractMemoryRef:
    return AbstractMemoryRef(
        jax_core.ShapedArray(self.shape, self.dtype), self.memory_space)


def _make_aval(obj: object) -> jax_core.AbstractValue:
  if isinstance(obj, MemoryRef):
    return obj.get_aval()
  if isinstance(obj, SemaphoreType):
    return obj.get_aval()
  raise ValueError(f"No registered conversion for {type(obj)}. "
                   "Only VMEM and SemaphoreType are supported.")


@dataclasses.dataclass(init=False, unsafe_hash=True)
class PrefetchScalarGridSpec(pallas_core.GridSpec):
  grid: Grid
  num_scalar_prefetch: int
  in_specs: tuple[BlockSpec | NoBlockSpec, ...]
  out_specs: tuple[BlockSpec | NoBlockSpec, ...]
  in_specs_tree: Any
  out_specs_tree: Any
  scratch_shapes: tuple[Any, ...]

  def __init__(
      self,
      num_scalar_prefetch: int,
      grid: Grid | None = None,
      in_specs: BlockSpecTree = no_block_spec,
      out_specs: BlockSpecTree = no_block_spec,
      scratch_shapes: Any | Sequence[Any] = ()
  ):
    super().__init__(grid, in_specs, out_specs)
    self.num_scalar_prefetch = num_scalar_prefetch
    self.scratch_shapes = tuple(scratch_shapes)

  def get_grid_mapping(
      self, in_avals, in_tree, out_avals, out_tree
  ) -> tuple[tuple[jax_core.AbstractValue, ...], GridMapping]:
    assert all(i is None or isinstance(i, int) for i in self.grid)
    grid_mapping_grid = tuple(
        pallas_core.dynamic_grid_dim if d is None else d for d in self.grid
    )
    all_avals = tree_util.tree_unflatten(in_tree, in_avals)
    flat_scratch_shapes, scratch_tree = tree_util.tree_flatten(
        self.scratch_shapes)
    flat_scratch_avals = map(_make_aval, flat_scratch_shapes)
    scalar_avals, unflat_in_avals = split_list(
        all_avals, [self.num_scalar_prefetch])
    flat_scalar_avals, scalar_tree = tree_util.tree_flatten(scalar_avals)
    num_flat_scalar_prefetch = len(flat_scalar_avals)
    in_avals, in_avals_tree = tree_util.tree_flatten(tuple(unflat_in_avals))
    flat_in_specs, flat_out_specs = self._get_in_out_specs(
        in_avals, in_avals_tree, out_avals, out_tree)
    in_specs, in_ref_avals, out_specs, out_ref_avals = (
        pallas_core._get_ref_avals(
            self.grid, in_avals, flat_in_specs,
            out_avals, flat_out_specs))
    scalar_ref_avals = [
        AbstractMemoryRef(jax_core.ShapedArray(aval.shape, aval.dtype),
                          TPUMemorySpace.SMEM)
        for aval in flat_scalar_avals]
    grid_avals = [jax_core.ShapedArray((), jnp.dtype("int32"))] * len(self.grid)
    # Create args, kwargs pytree def
    index_map_in_tree = tree_util.tree_structure(
        ((*grid_avals, *scalar_avals), {})
    )
    in_block_mappings = map(
        partial(
            _convert_block_spec_to_block_mapping,
            (*grid_avals, *scalar_ref_avals),
            in_tree=index_map_in_tree,
            grid=grid_mapping_grid,
            mapped_dims=(),
        ),
        in_specs,
        in_ref_avals,
    )
    out_block_mappings = map(
        partial(
            _convert_block_spec_to_block_mapping,
            (*grid_avals, *scalar_ref_avals),
            in_tree=index_map_in_tree,
            grid=grid_mapping_grid,
            mapped_dims=(),
        ),
        out_specs,
        out_ref_avals,
    )
    grid_mapping = GridMapping(
        grid=grid_mapping_grid,  # type: ignore
        block_mappings=(*in_block_mappings, *out_block_mappings),
        mapped_dims=(),
        num_index_operands=num_flat_scalar_prefetch,
        num_scratch_operands=len(flat_scratch_avals)
    )
    jaxpr_scalar_ref_avals = tree_util.tree_unflatten(
        scalar_tree, scalar_ref_avals)
    jaxpr_in_ref_avals = tree_util.tree_unflatten(in_avals_tree, in_ref_avals)
    jaxpr_scratch_avals = tree_util.tree_unflatten(
        scratch_tree, flat_scratch_avals)
    if not isinstance(jaxpr_scratch_avals, (tuple, list)):
      jaxpr_scratch_avals = (jaxpr_scratch_avals,)
    jaxpr_in_avals = (*jaxpr_scalar_ref_avals,
                      *jaxpr_in_ref_avals)
    jaxpr_out_avals = tree_util.tree_unflatten(out_tree, out_ref_avals)
    if not isinstance(jaxpr_out_avals, (tuple, list)):
      jaxpr_out_avals = (jaxpr_out_avals,)
    return (*jaxpr_in_avals, *jaxpr_out_avals,
            *jaxpr_scratch_avals), grid_mapping
