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

"""Contains GPU-specific Pallas abstractions."""

import dataclasses
import functools
from typing import Literal
from jax import core as jax_core
from jax._src.pallas import core as pallas_core
import jax.numpy as jnp

AbstractMemoryRef = pallas_core.AbstractMemoryRef

# TODO(b/354568887): Cosolidate this with TPU's MemoryRef.
@dataclasses.dataclass(frozen=True)
class ArrayRefConfig:
  shape: tuple[int, ...]
  dtype: jnp.dtype
  memory_space_sym: Literal["smem", "gmem", "regs"]

  @property
  def memory_space(self):
    match self.memory_space_sym:
      case "smem":
        return SMemSpace()
      case "gmem":
        return GMemSpace()
      case "regs":
        raise ValueError("No default regs array ref")

  def get_aval(self) -> AbstractMemoryRef:
    return AbstractMemoryRef(
        jax_core.ShapedArray(self.shape, self.dtype), self.memory_space
    )


@dataclasses.dataclass(frozen=True)
class WGMMAOperandConfig(ArrayRefConfig):
  tiling: tuple[int, int]
  swizzle: int = 128
  tma_transpose: bool = True

  @property
  def memory_space(self):
    return SMemSpace(wgmma_operand_config=self)

  @property
  def tiled_shape(self):
    (m, n), (tm, tn) = self.shape, self.tiling
    if m % tm != 0 or n % tn:
      raise ValueError(f"Can't tile {(self.shape, self.tiling)}")

    return (m // tm, n // tn, tm , tn)

@dataclasses.dataclass(frozen=True)
class WGMMAAccumulatorConfig(ArrayRefConfig):
  wgmma_config: "WGMMAConfig"

  @property
  def memory_space(self):
    return RegsSpace(wgmma_config=self.wgmma_config)


@dataclasses.dataclass(frozen=True)
class _MatmulDims:
  m: int
  n: int
  k: int

  @property
  def mk(self) -> tuple[int, int]:
    return (self.m, self.k)

  @property
  def kn(self) -> tuple[int, int]:
    return (self.k, self.n)

class WGMMAConfig:
  """Configuration for running wgmma operations."""

  lhs_dtype: jnp.dtype
  rhs_dtype: jnp.dtype
  acc_dtype: jnp.dtype
  swizzle: int
  # We can only skip the TMA transpose (ie let wgmma do the transpose)
  # for 16b types.
  tma_transpose_rhs: bool

  def __init__(self, dtype, swizzle=128):
    if swizzle != 128:
      raise NotImplementedError

    self.acc_dtype = jnp.dtype(jnp.float32)
    match dtype:
      case (acc, lhs, rhs):
        self.lhs_dtype = jnp.dtype(lhs)
        self.rhs_dtype = jnp.dtype(rhs)
        self.acc_dtype = jnp.dtype(acc)
      case (lhs, rhs):
        self.lhs_dtype = jnp.dtype(lhs)
        self.rhs_dtype = jnp.dtype(rhs)
      case _:
        self.lhs_dtype = self.rhs_dtype = jnp.dtype(dtype)
    self.tma_transpose_rhs = self.rhs_dtype.itemsize != 2
    self.swizzle = swizzle

  @property
  def wgmma_tiling(self) -> _MatmulDims:
    if self.rhs_dtype.itemsize != self.lhs_dtype.itemsize:
      raise ValueError("WGMMA does not support inconsistent types.")

    tile = self.swizzle // self.lhs_dtype.itemsize
    return _MatmulDims(m=64, n=tile, k=tile)

  def lhs_smem_config(self, shape: tuple[int, int]) -> WGMMAOperandConfig:
    if self.swizzle and shape[1] < self.swizzle:
      raise ValueError(shape, self.swizzle)

    assert self.wgmma_tiling.k == self.swizzle // self.lhs_dtype.itemsize
    return WGMMAOperandConfig(
        shape=shape,
        dtype=self.lhs_dtype,
        memory_space_sym="smem",
        swizzle=self.swizzle,
        tiling=self.wgmma_tiling.mk,
        tma_transpose=False,
    )

  def rhs_smem_config(self, shape: tuple[int, int]) -> WGMMAOperandConfig:
    assert self.wgmma_tiling.n == self.swizzle // self.rhs_dtype.itemsize
    return WGMMAOperandConfig(
        shape=shape,
        dtype=self.rhs_dtype,
        memory_space_sym="smem",
        swizzle=self.swizzle,
        tiling=self.wgmma_tiling.kn
        ,
        tma_transpose=self.tma_transpose_rhs,
    )

  def out_smem_config(self, shape: tuple[int, int]) -> WGMMAOperandConfig:
    tiling = (self.wgmma_tiling.m, self.swizzle // self.acc_dtype.itemsize)
    return WGMMAOperandConfig(
        shape=shape,
        dtype=self.acc_dtype,
        memory_space_sym="smem",
        swizzle=self.swizzle,
        tiling=tiling,
        tma_transpose=False,
    )


  def accumulator_config(self, m, n) -> WGMMAAccumulatorConfig:
    return WGMMAAccumulatorConfig(
        shape=(m, n),
        dtype=self.acc_dtype,
        memory_space_sym="regs",
        wgmma_config=self,
    )


@dataclasses.dataclass(frozen=True)
class GMemSpace:
  ...

@dataclasses.dataclass(frozen=True)
class SMemSpace:
  wgmma_operand_config: WGMMAOperandConfig | None = None


@dataclasses.dataclass(frozen=True)
class RegsSpace:
  wgmma_config: WGMMAConfig | None = None

GPUMemorySpace = GMemSpace | SMemSpace | RegsSpace


SMEM = functools.partial(ArrayRefConfig, memory_space_sym="smem")
GMEM = functools.partial(ArrayRefConfig, memory_space_sym="gmem")
