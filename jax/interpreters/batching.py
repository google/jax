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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.interpreters.batching import (
    Array as Array,
    AxisSize as AxisSize,
    BatchTrace as BatchTrace,
    BatchTracer as BatchTracer,
    BatchingRule as BatchingRule,
    Elt as Elt,
    FromEltHandler as FromEltHandler,
    GetIdx as GetIdx,
    IndexedAxisSize as IndexedAxisSize,
    JumbleAxis as JumbleAxis,
    JumbleTy as JumbleTy,
    Jumble as Jumble,
    MakeIotaHandler as MakeIotaHandler,
    MapSpec as MapSpec,
    NotMapped as NotMapped,
    RaggedAxis as RaggedAxis,
    ToEltHandler as ToEltHandler,
    Vmappable as Vmappable,
    ZeroIfMapped as ZeroIfMapped,
    Zero as Zero,
    axis_primitive_batchers as axis_primitive_batchers,
    batch_custom_jvp_subtrace as batch_custom_jvp_subtrace,
    batch_custom_vjp_bwd as batch_custom_vjp_bwd,
    batch_jaxpr2 as batch_jaxpr2,
    batch_jaxpr_axes as batch_jaxpr_axes,
    batch_jaxpr as batch_jaxpr,
    batch_subtrace as batch_subtrace,
    batch as batch,
    bdim_at_front as bdim_at_front,
    broadcast_batcher as broadcast_batcher,
    broadcast as broadcast,
    defbroadcasting as defbroadcasting,
    defreducer as defreducer,
    defvectorized as defvectorized,
    flatten_fun_for_vmap as flatten_fun_for_vmap,
    from_elt_handlers as from_elt_handlers,
    from_elt as from_elt,
    is_vmappable as is_vmappable,
    jumble_axis as jumble_axis,
    jumble as jumble,
    make_iota_handlers as make_iota_handlers,
    make_iota as make_iota,
    matchaxis as matchaxis,
    moveaxis as moveaxis,
    not_mapped as not_mapped,
    primitive_batchers as primitive_batchers,
    reducer_batcher as reducer_batcher,
    register_vmappable as register_vmappable,
    spec_types as spec_types,
    spmd_axis_primitive_batchers as spmd_axis_primitive_batchers,
    to_elt_handlers as to_elt_handlers,
    to_elt as to_elt,
    unregister_vmappable as unregister_vmappable,
    vectorized_batcher as vectorized_batcher,
    vmappables as vmappables,
    vtile as vtile,
    zero_if_mapped as zero_if_mapped,
)
