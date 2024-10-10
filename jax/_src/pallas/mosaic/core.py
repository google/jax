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
from typing import Any, ClassVar, Literal

import jax
from jax._src import config
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import util
from jax._src.pallas import core as pallas_core
import jax.numpy as jnp
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
ScratchShapeTree = pallas_core.ScratchShapeTree
AbstractMemoryRef = pallas_core.AbstractMemoryRef
no_block_spec = pallas_core.no_block_spec
_convert_block_spec_to_block_mapping = pallas_core._convert_block_spec_to_block_mapping
split_list = util.split_list

_ENABLE_RUNTIME_ASSERT = config.bool_state(
    "jax_pallas_enable_runtime_assert",
    default=False,
    help=(
        "If set, enables runtime assertions in the kernel via checkify.check."
        " Otherwise, runtime asserts will be ignored unless functionalized"
        " using checkify.checkify."
    ),
)


@dataclasses.dataclass(frozen=True)
class TPUCompilerParams(pallas_core.CompilerParams):
  """Mosaic TPU compiler parameters.

  Attributes:
    dimension_semantics: A list of dimension semantics for each grid
      dimension of the kernel. Either "parallel" for dimensions that can
      execute in any order, or "arbitrary" for dimensions that must be
      executed sequentially.
    allow_input_fusion: A list of booleans indicating whether input fusion is
      allowed for each argument.
    vmem_limit_bytes: Overrides the default VMEM limit for a kernel. Note
      that this must be used in conjunction with the
      --xla_tpu_scoped_vmem_limit_kib=N flag with N*1kib > vmem_limit_bytes.
    collective_id: Indicates which barrier semaphore to use for the kernel.
      Note that using the same collective_id does not guarantee that
      the same barrier semaphore will be allocated between kernels.
    internal_scratch_in_bytes: The size of the internal scratch space used by
      Mosaic.
    flags: A dictionary of command line flags for the kernel.
    serialization_format: The serialization format for the kernel body.
    device_type: The device type to compile for.
  """
  PLATFORM: ClassVar[str] = "mosaic"
  dimension_semantics: Sequence[Literal["parallel", "arbitrary"]] | None = None
  allow_input_fusion: Sequence[bool] | None = None
  vmem_limit_bytes: int | None = None
  collective_id: int | None = None
  flags: dict[str, Any] | None = None
  internal_scratch_in_bytes: int | None = None
  serialization_format: int = 1
  device_type: str | None = None

class TPUMemorySpace(enum.Enum):
  ANY = "any"  # TODO(b/368401328): Remove this and just use pl.ANY.
  VMEM = "vmem"
  SMEM = "smem"
  CMEM = "cmem"
  SEMAPHORE = "semaphore_mem"

  def __str__(self) -> str:
    return self.value

  def __call__(self, shape: tuple[int, ...], dtype: jnp.dtype):
    # A convenience function for constructing MemoryRef types.
    return pallas_core.MemoryRef(shape, dtype, self)

class semaphore_dtype(dtypes.extended): pass
class semaphore(semaphore_dtype): pass
class dma_semaphore(semaphore_dtype): pass
class barrier_semaphore(semaphore_dtype): pass

class AbstractSemaphoreTyRules:
  @staticmethod
  def pallas_interpret_element_aval(_) -> jax_core.ShapedArray:
    return jax_core.ShapedArray((), pallas_core.SEMAPHORE_INTERPRET_DTYPE)

  @staticmethod
  def physical_element_aval(_) -> jax_core.ShapedArray:
    return jax_core.ShapedArray((), jnp.int32)

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
    if pallas_core.is_interpret_mode():
      dtype = pallas_core.SEMAPHORE_INTERPRET_DTYPE
    return pallas_core.MemoryRef(shape, dtype, TPUMemorySpace.SEMAPHORE)

  def get_array_aval(self) -> pallas_core.ShapedArrayWithMemorySpace:
    return self(()).get_array_aval()

  def get_ref_aval(self) -> AbstractMemoryRef:
    return self(()).get_ref_aval()

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


@dataclasses.dataclass(init=False, kw_only=True, unsafe_hash=True)
class PrefetchScalarGridSpec(pallas_core.GridSpec):
  num_scalar_prefetch: int

  def __init__(
      self,
      num_scalar_prefetch: int,
      grid: Grid = (),
      in_specs: BlockSpecTree = no_block_spec,
      out_specs: BlockSpecTree = no_block_spec,
      scratch_shapes: ScratchShapeTree = ()
  ):
    super().__init__(grid, in_specs, out_specs, scratch_shapes)
    self.num_scalar_prefetch = num_scalar_prefetch
    self.scratch_shapes = tuple(scratch_shapes)

  def _make_scalar_ref_aval(self, aval):
    return AbstractMemoryRef(jax_core.ShapedArray(aval.shape, aval.dtype),
                             TPUMemorySpace.SMEM)


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

def runtime_assert_enabled() -> bool:
  """Returns whether runtime asserts are enabled."""
  return _ENABLE_RUNTIME_ASSERT.value
