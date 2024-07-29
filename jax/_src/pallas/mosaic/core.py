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
from typing import Any, Hashable

import jax
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import util
import jax.numpy as jnp
from jax._src.pallas import core as pallas_core
import numpy as np

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

partial = functools.partial
Grid = pallas_core.Grid
TupleGrid = pallas_core.TupleGrid
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
    return pallas_core.index_map_grid_aval

class AbstractSemaphoreTy(dtypes.ExtendedDType):
  name: str
  _rules = AbstractSemaphoreTyRules

  def __repr__(self) -> str:
    return self.name

  def __eq__(self, other):
    return self.__class__ == other.__class__

  def __hash__(self) -> int:
    return hash(self.__class__)

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

  def get_aval(self) -> AbstractMemoryRef:
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


@dataclasses.dataclass(init=False, unsafe_hash=True)
class PrefetchScalarGridSpec(pallas_core.GridSpec):
  grid: TupleGrid
  grid_names: tuple[Hashable, ...] | None
  num_scalar_prefetch: int
  in_specs: pallas_core.BlockSpecTree
  out_specs: pallas_core.BlockSpecTree
  scratch_shapes: tuple[Any, ...]

  def __init__(
      self,
      num_scalar_prefetch: int,
      grid: Grid = (),
      in_specs: BlockSpecTree = no_block_spec,
      out_specs: BlockSpecTree = no_block_spec,
      scratch_shapes: Any | Sequence[Any] = ()
  ):
    super().__init__(grid, in_specs, out_specs)
    self.num_scalar_prefetch = num_scalar_prefetch
    self.scratch_shapes = tuple(scratch_shapes)

  def _make_scalar_ref_aval(self, aval):
    return AbstractMemoryRef(jax_core.ShapedArray(aval.shape, aval.dtype),
                             TPUMemorySpace.SMEM)

  def _make_scratch_aval(self, obj: object) -> jax_core.AbstractValue:
    if isinstance(obj, MemoryRef):
      return obj.get_aval()
    if isinstance(obj, SemaphoreType):
      return obj.get_aval()
    raise ValueError(f"No registered conversion for {type(obj)}. "
                     "Only VMEM and SemaphoreType are supported.")


@dataclasses.dataclass(frozen=True)
class TensorCore:
  id: int


def create_tensorcore_mesh(axis_name: str) -> pallas_core.PallasMesh:
  # TODO(b/355036384): emit a better error if we don't have tensorcores.
  num_cores = jax.devices()[0].num_cores
  return pallas_core.PallasMesh(
      np.array([TensorCore(i) for i in range(num_cores)]),
      [axis_name],
  )
