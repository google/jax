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
"""Mosaic GPU pallas primitives."""

from __future__ import annotations

from jax._src import core as jax_core
from jax._src import effects
from jax._src import state
import jax._src.pallas.mosaic_gpu.core as mgpu_core

class InternalWgmmaPipelineEffect(effects.Effect):
  pass


internal_wgmma_pipeline_effect = InternalWgmmaPipelineEffect()
effects.control_flow_allowed_effects.add_type(InternalWgmmaPipelineEffect)

wgmma_zero_accumulator_p = jax_core.Primitive("wgmma_zero_accumulator")
def wgmma_zero_accumulator(shape, config: mgpu_core.WGMMAConfig):
  return wgmma_zero_accumulator_p.bind(shape=shape, config=config)

def _wgmma_zero_accumulator_abstract_eval(shape, config):
  return mgpu_core.AbstractMemoryRef(
      jax_core.ShapedArray(shape, config.acc_dtype),
      memory_space=mgpu_core.RegsSpace(wgmma_config=config),
  )

wgmma_zero_accumulator_p.def_abstract_eval(_wgmma_zero_accumulator_abstract_eval)

wgmma_p = jax_core.Primitive("wgmma")
def wgmma(acc, a, b):
  for name, op in (("lhs", a), ("rhs", b)):
    if not isinstance(a.memory_space, mgpu_core.SMemSpace) and a.memory_space.wgmma_operand_config:
      raise ValueError(f"{name} must be a wgmma shared ref: {op=}")

  if not isinstance(acc.memory_space, mgpu_core.RegsSpace) and acc.memory_space.wgmma_config:
    raise ValueError(f"acc must be a wgmma accum ref: {acc=}")

  swizzle = acc.memory_space.wgmma_config.swizzle
  ma, ka = a.inner_aval.shape
  kb, nb = b.inner_aval.shape
  mc, nc = acc.inner_aval.shape

  if ma != mc or nb != nc or ka != kb:
    raise ValueError(a, b , acc)

  swizzle = acc.memory_space.wgmma_config.swizzle
  b_row_major = b.memory_space.wgmma_operand_config.tma_transpose

  # TODO(cperivol): the accumulator is no longer valid so we need some
  # way to invalidate it.
  return wgmma_p.bind(acc, a, b, swizzle=swizzle, b_row_major=b_row_major)

wgmma_p.multiple_results = True
wgmma_p.def_effectful_abstract_eval(
    lambda acc, *arg, **kw: (
        [],
        {
            internal_wgmma_pipeline_effect,
            state.WriteEffect(0),
            state.ReadEffect(0),
            state.ReadEffect(1),
            state.ReadEffect(2),
        },
    )
)

wgmma_wait_p = jax_core.Primitive("wgmma_wait")

def wgmma_wait(i: int):
  return wgmma_wait_p.bind(i)


wgmma_wait_p.multiple_results = True
wgmma_wait_p.def_effectful_abstract_eval(
    lambda _: ([], {internal_wgmma_pipeline_effect})
)
