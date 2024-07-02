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

# Serialization and deserialization of _export.Exported

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TypeVar

try:
  import flatbuffers
except ImportError as e:
  raise ImportError(
      "Please install 'flatbuffers' in order to use Exported serialization"
      ) from e

from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src import tree_util
from jax._src.export import serialization_generated as ser_flatbuf
from jax._src.export import _export
from jax._src.export import shape_poly
from jax._src.lib import xla_client

import numpy as np

T = TypeVar("T")
SerT = TypeVar("SerT")

# The _SERIALIZATION_VERSION changes when we change the serialization schema
# even if the change is backwards compatible.
# Version 1, Nov 2023, first version.
# Version 2, Dec 16th, 2023, adds the f0 dtype.
_SERIALIZATION_VERSION = 2

def serialize(exp: _export.Exported, vjp_order: int = 0) -> bytearray:
  """Serializes an Exported.

  Args:
    exp: the Exported to serialize.
    vjp_order: The maximum vjp order to include. E.g., the value 2 means that we
      serialize the primal functions and two orders of the `vjp` function. This
      should allow 2nd order reverse mode differentiation of the deserialized
      function. i.e., `jax.grad(jax.grad(f)).`
  """
  builder = flatbuffers.Builder(65536)
  exported = _serialize_exported(builder, exp, vjp_order)
  builder.Finish(exported)
  return builder.Output()


def deserialize(ser: bytearray) -> _export.Exported:
  """Deserializes an Exported."""
  exp = ser_flatbuf.Exported.GetRootAsExported(ser)
  return _deserialize_exported(exp)


def _serialize_exported(
    builder: flatbuffers.Builder, exp: _export.Exported, vjp_order: int
) -> int:
  # Serialize bottom-up
  fun_name = builder.CreateString(exp.fun_name)
  in_tree = _serialize_pytreedef(builder, exp.in_tree)
  in_avals = _serialize_array(builder, _serialize_aval, exp.in_avals)
  out_tree = _serialize_pytreedef(builder, exp.out_tree)
  out_avals = _serialize_array(builder, _serialize_aval, exp.out_avals)
  in_shardings = _serialize_array(
      builder, _serialize_sharding, exp.in_shardings_hlo
  )
  out_shardings = _serialize_array(
      builder, _serialize_sharding, exp.out_shardings_hlo
  )
  ordered_effects = _serialize_array(
      builder, _serialize_effect, exp.ordered_effects
  )
  unordered_effects = _serialize_array(
      builder, _serialize_effect, exp.unordered_effects
  )
  disabled_safety_checks = _serialize_array(
      builder, _serialize_disabled_safety_check, exp.disabled_safety_checks
  )
  platforms = _serialize_array(
      builder, lambda b, p: b.CreateString(p), exp.platforms
  )
  mlir_module_serialized = builder.CreateByteVector(exp.mlir_module_serialized)
  module_kept_var_idx = builder.CreateNumpyVector(
      np.array(exp.module_kept_var_idx, dtype=np.uint16)
  )

  vjp = None
  if vjp_order > 0:
    if not exp.has_vjp():
      # TODO: add test
      raise ValueError(
          "serialization of an Exported that does not have vjps of high-enough "
          "order"
      )
    vjp = _serialize_exported(builder, exp.vjp(), vjp_order - 1)

  ser_flatbuf.ExportedStart(builder)
  ser_flatbuf.ExportedAddSerializationVersion(builder, _SERIALIZATION_VERSION)
  ser_flatbuf.ExportedAddFunctionName(builder, fun_name)
  ser_flatbuf.ExportedAddInTree(builder, in_tree)
  ser_flatbuf.ExportedAddInAvals(builder, in_avals)
  ser_flatbuf.ExportedAddOutTree(builder, out_tree)
  ser_flatbuf.ExportedAddOutAvals(builder, out_avals)
  ser_flatbuf.ExportedAddNrDevices(builder, exp.nr_devices)
  ser_flatbuf.ExportedAddInShardings(builder, in_shardings)
  ser_flatbuf.ExportedAddOutShardings(builder, out_shardings)
  ser_flatbuf.ExportedAddPlatforms(builder, platforms)
  ser_flatbuf.ExportedAddOrderedEffects(builder, ordered_effects)
  ser_flatbuf.ExportedAddUnorderedEffects(builder, unordered_effects)
  ser_flatbuf.ExportedAddDisabledChecks(builder, disabled_safety_checks)
  ser_flatbuf.ExportedAddMlirModuleSerialized(builder, mlir_module_serialized)
  ser_flatbuf.ExportedAddCallingConventionVersion(
      builder, exp.calling_convention_version
  )
  ser_flatbuf.ExportedAddModuleKeptVarIdx(builder, module_kept_var_idx)
  ser_flatbuf.ExportedAddUsesGlobalConstants(
      builder, exp.uses_global_constants
  )
  if vjp is not None:
    ser_flatbuf.ExportedAddVjp(builder, vjp)
  return ser_flatbuf.ExportedEnd(builder)


def _serialize_array(
    builder: flatbuffers.Builder,
    serialize_one: Callable[[flatbuffers.Builder, T], int],
    elements: Sequence[T],
) -> int:
  element_offsets = [serialize_one(builder, e) for e in elements]
  ser_flatbuf.PyTreeDefStartChildrenVector(builder, len(element_offsets))
  for sc in reversed(element_offsets):
    builder.PrependUOffsetTRelative(sc)
  return builder.EndVector()


def _deserialize_exported(exp: ser_flatbuf.Exported) -> _export.Exported:
  serialization_version = exp.SerializationVersion()
  if serialization_version != _SERIALIZATION_VERSION:
    raise NotImplementedError(
        f"deserialize unsupported version {serialization_version}"
    )

  fun_name = exp.FunctionName().decode("utf-8")
  _, in_tree = tree_util.tree_flatten(
      _deserialize_pytreedef_to_pytree(exp.InTree())
  )
  scope = shape_poly.SymbolicScope(())  # TODO: serialize the constraints
  deser_aval = partial(_deserialize_aval, scope=scope)
  in_avals = _deserialize_tuple(
      exp.InAvalsLength, exp.InAvals, deser_aval
  )
  _, out_tree = tree_util.tree_flatten(
      _deserialize_pytreedef_to_pytree(exp.OutTree())
  )
  out_avals = _deserialize_tuple(
      exp.OutAvalsLength, exp.OutAvals, deser_aval
  )
  nr_devices = exp.NrDevices()
  in_shardings = _deserialize_tuple(
      exp.InShardingsLength, exp.InShardings, _deserialize_sharding
  )
  out_shardings = _deserialize_tuple(
      exp.OutShardingsLength, exp.OutShardings, _deserialize_sharding
  )
  platforms = _deserialize_tuple(
      exp.PlatformsLength,
      exp.Platforms,
      lambda v: v.decode("utf-8"),
  )
  ordered_effects = _deserialize_tuple(
      exp.OrderedEffectsLength, exp.OrderedEffects, _deserialize_effect
  )
  unordered_effects = _deserialize_tuple(
      exp.UnorderedEffectsLength, exp.UnorderedEffects, _deserialize_effect
  )
  disabled_safety_checks = _deserialize_tuple(
      exp.DisabledChecksLength,
      exp.DisabledChecks,
      _deserialize_disabled_safety_check,
  )

  mlir_module_serialized = exp.MlirModuleSerializedAsNumpy().tobytes()
  calling_convention_version = exp.CallingConventionVersion()
  module_kept_var_idx = tuple(exp.ModuleKeptVarIdxAsNumpy().tolist())
  uses_global_constants = exp.UsesGlobalConstants()

  _get_vjp = None
  if vjp := exp.Vjp():
    _get_vjp = lambda _: _deserialize_exported(vjp)

  return _export.Exported(
      fun_name=fun_name,
      in_tree=in_tree,
      in_avals=in_avals,
      out_tree=out_tree,
      out_avals=out_avals,
      nr_devices=nr_devices,
      in_shardings_hlo=in_shardings,
      out_shardings_hlo=out_shardings,
      platforms=platforms,
      ordered_effects=ordered_effects,
      unordered_effects=unordered_effects,
      disabled_safety_checks=disabled_safety_checks,
      mlir_module_serialized=mlir_module_serialized,
      calling_convention_version=calling_convention_version,
      module_kept_var_idx=module_kept_var_idx,
      uses_global_constants=uses_global_constants,
      _get_vjp=_get_vjp,
  )


def _deserialize_tuple(
    get_len: Callable[[], int],
    get_elem: Callable[[int], SerT],
    deserialize_one: Callable[[SerT], T],
) -> tuple[T, ...]:
  return tuple(deserialize_one(get_elem(i)) for i in range(get_len()))


def _serialize_pytreedef(
    builder: flatbuffers.Builder, p: tree_util.PyTreeDef
) -> int:
  node_data = p.node_data()
  children = p.children()

  children_vector_offset = None
  children_names_vector_offset = None
  if children:
    children_vector_offset = _serialize_array(
        builder, _serialize_pytreedef, children
    )

  if node_data is None:  # leaf
    kind = ser_flatbuf.PyTreeDefKind.leaf
  elif node_data[0] is type(None):
    kind = ser_flatbuf.PyTreeDefKind.none
  elif node_data[0] is tuple:
    kind = ser_flatbuf.PyTreeDefKind.tuple
  elif node_data[0] is list:
    kind = ser_flatbuf.PyTreeDefKind.list
  elif node_data[0] is dict:
    kind = ser_flatbuf.PyTreeDefKind.dict
    assert len(node_data[1]) == len(children)
    children_names_vector_offset = _serialize_array(
        builder, lambda b, s: b.CreateString(s), node_data[1]
    )
  else:
    raise NotImplementedError(f"serializing PyTreeDef {node_data}")

  ser_flatbuf.PyTreeDefStart(builder)
  ser_flatbuf.PyTreeDefAddKind(builder, kind)
  if children_vector_offset:
    ser_flatbuf.PyTreeDefAddChildren(builder, children_vector_offset)
  if children_names_vector_offset:
    ser_flatbuf.PyTreeDefAddChildrenNames(builder, children_names_vector_offset)
  return ser_flatbuf.PyTreeDefEnd(builder)


def _deserialize_pytreedef_to_pytree(p: ser_flatbuf.PyTreeDef):
  # We construct a PyTree and later we'll flatten it to get the PyTreeDef.
  # TODO: is there a more direct way to construct a PyTreeDef?
  kind = p.Kind()
  nr_children = p.ChildrenLength()
  children = [
      _deserialize_pytreedef_to_pytree(p.Children(i))
      for i in range(nr_children)
  ]
  if kind == ser_flatbuf.PyTreeDefKind.leaf:
    return 0.0
  elif kind == ser_flatbuf.PyTreeDefKind.none:
    return None
  elif kind == ser_flatbuf.PyTreeDefKind.tuple:
    return tuple(children)
  elif kind == ser_flatbuf.PyTreeDefKind.list:
    return list(children)
  elif kind == ser_flatbuf.PyTreeDefKind.dict:
    assert p.ChildrenNamesLength() == nr_children
    keys = [p.ChildrenNames(i).decode("utf-8") for i in range(nr_children)]
    return dict(zip(keys, children))
  else:
    assert False, kind


_dtype_to_dtype_kind = {
    np.dtype("bool"): ser_flatbuf.DType.bool,
    np.dtype("int8"): ser_flatbuf.DType.i8,
    np.dtype("int16"): ser_flatbuf.DType.i16,
    np.dtype("int32"): ser_flatbuf.DType.i32,
    np.dtype("int64"): ser_flatbuf.DType.i64,
    np.dtype("uint8"): ser_flatbuf.DType.ui8,
    np.dtype("uint16"): ser_flatbuf.DType.ui16,
    np.dtype("uint32"): ser_flatbuf.DType.ui32,
    np.dtype("uint64"): ser_flatbuf.DType.ui64,
    dtypes.float0: ser_flatbuf.DType.f0,
    np.dtype("float16"): ser_flatbuf.DType.f16,
    np.dtype("float32"): ser_flatbuf.DType.f32,
    np.dtype("float64"): ser_flatbuf.DType.f64,
    np.dtype("complex64"): ser_flatbuf.DType.c64,
    np.dtype("complex128"): ser_flatbuf.DType.c128,
    dtypes._bfloat16_dtype: ser_flatbuf.DType.bf16,
    dtypes._int4_dtype: ser_flatbuf.DType.i4,
    dtypes._uint4_dtype: ser_flatbuf.DType.ui4,
    dtypes._float8_e4m3b11fnuz_dtype: ser_flatbuf.DType.f8_e4m3b11fnuz,
    dtypes._float8_e4m3fn_dtype: ser_flatbuf.DType.f8_e4m3fn,
    dtypes._float8_e4m3fnuz_dtype: ser_flatbuf.DType.f8_e4m3fnuz,
    dtypes._float8_e5m2_dtype: ser_flatbuf.DType.f8_e5m2,
    dtypes._float8_e5m2fnuz_dtype: ser_flatbuf.DType.f8_e5m2fnuz,
}


_dtype_kind_to_dtype = {
    kind: dtype for dtype, kind in _dtype_to_dtype_kind.items()
}


def _serialize_aval(
    builder: flatbuffers.Builder, aval: core.ShapedArray
) -> int:
  aval_kind = ser_flatbuf.AbstractValueKind.shapedArray
  shape_offsets = [builder.CreateString(str(d)) for d in aval.shape]
  ser_flatbuf.AbstractValueStartShapeVector(builder, len(aval.shape))
  for d in reversed(shape_offsets):
    builder.PrependUOffsetTRelative(d)
  shape_vector_offset = builder.EndVector()

  ser_flatbuf.AbstractValueStart(builder)
  ser_flatbuf.AbstractValueAddKind(builder, aval_kind)
  ser_flatbuf.AbstractValueAddShape(builder, shape_vector_offset)
  ser_flatbuf.AbstractValueAddDtype(builder, _dtype_to_dtype_kind[aval.dtype])
  return ser_flatbuf.AbstractValueEnd(builder)


def _deserialize_aval(aval: ser_flatbuf.AbstractValue,
                      scope) -> core.ShapedArray:
  aval_kind = aval.Kind()
  if aval_kind == ser_flatbuf.AbstractValueKind.shapedArray:
    dtype = _dtype_kind_to_dtype[aval.Dtype()]
    shape = shape_poly.symbolic_shape(
        ",".join(
            aval.Shape(i).decode("utf-8") for i in range(aval.ShapeLength())
        ),
        scope=scope
    )
    return core.ShapedArray(shape, dtype)
  else:
    assert False, aval_kind


def _serialize_sharding(
    builder: flatbuffers.Builder, s: _export.HloSharding | None
) -> int:
  proto = None
  if s is None:
    kind = ser_flatbuf.ShardingKind.unspecified
  else:
    kind = ser_flatbuf.ShardingKind.hlo_sharding
    proto_bytes = s.to_proto().SerializeToString()
    proto = builder.CreateByteVector(proto_bytes)

  ser_flatbuf.ShardingStart(builder)
  ser_flatbuf.ShardingAddKind(builder, kind)
  if proto is not None:
    ser_flatbuf.ShardingAddHloShardingProto(builder, proto)
  return ser_flatbuf.ShardingEnd(builder)


def _deserialize_sharding(s: ser_flatbuf.Sharding) -> _export.HloSharding | None:
  kind = s.Kind()
  if kind == ser_flatbuf.ShardingKind.unspecified:
    return None

  if kind == ser_flatbuf.ShardingKind.hlo_sharding:
    proto_str = s.HloShardingProtoAsNumpy().tobytes()
    proto = xla_client.OpSharding()
    proto.ParseFromString(proto_str)

    return xla_client.HloSharding.from_proto(proto)

  assert False, kind


def _serialize_effect(builder: flatbuffers.Builder, eff: core.Effect) -> int:
  try:
    eff_replica = eff.__class__()
  except Exception:
    raise NotImplementedError(
        f"Effect {eff} must have a nullary constructor to be serializable"
    )
  try:
    hash_eff = hash(eff)
    hash_eff_replica = hash(eff_replica)
  except Exception:
    raise NotImplementedError(
        f"Effect {eff} must be hashable to be serializable"
    )
  if eff != eff_replica or hash_eff != hash_eff_replica:
    raise NotImplementedError(
      f"Effect {eff} must have a nullary class constructor that produces an "
      "equal effect object."
    )
  effect_type_name = str(eff.__class__)
  effect_type_name_offset = builder.CreateString(effect_type_name)
  ser_flatbuf.EffectStart(builder)
  ser_flatbuf.EffectAddTypeName(builder, effect_type_name_offset)
  return ser_flatbuf.ExportedEnd(builder)


def _deserialize_effect(eff: ser_flatbuf.Effect) -> core.Effect:
  effect_type_name = eff.TypeName().decode("utf-8")
  for existing_effect_type in effects.lowerable_effects._effect_types:
    if str(existing_effect_type) == effect_type_name:
      try:
        return existing_effect_type()
      except:
        # TODO: add test
        raise NotImplementedError(
            f"deserializing effect {effect_type_name} that does not have a "
            "nullary class constructor"
        )

  raise NotImplementedError(
      f"cannot deserialize effect type {effect_type_name}"
  )


def _serialize_disabled_safety_check(
    builder: flatbuffers.Builder, check: _export.DisabledSafetyCheck
) -> int:
  custom_call_target_str = check.is_custom_call()
  custom_call_target = None
  if custom_call_target_str is not None:
    kind = ser_flatbuf.DisabledSafetyCheckKind.custom_call
    custom_call_target = builder.CreateString(custom_call_target_str)
  elif check == _export.DisabledSafetyCheck.platform():
    kind = ser_flatbuf.DisabledSafetyCheckKind.platform
  elif check == _export.DisabledSafetyCheck.shape_assertions():
    kind = ser_flatbuf.DisabledSafetyCheckKind.shape_assertions
  else:
    raise NotImplementedError(f"serializing DisabledSafetyCheck: {check}")

  ser_flatbuf.DisabledSafetyCheckStart(builder)
  ser_flatbuf.DisabledSafetyCheckAddKind(builder, kind)
  if custom_call_target is not None:
    ser_flatbuf.DisabledSafetyCheckAddCustomCallTarget(
        builder, custom_call_target
    )
  return ser_flatbuf.DisabledSafetyCheckEnd(builder)


def _deserialize_disabled_safety_check(
    sc: ser_flatbuf.DisabledSafetyCheck,
) -> _export.DisabledSafetyCheck:
  kind = sc.Kind()
  if kind == ser_flatbuf.DisabledSafetyCheckKind.custom_call:
    return _export.DisabledSafetyCheck.custom_call(
        sc.CustomCallTarget().decode("utf-8")
    )
  if kind == ser_flatbuf.DisabledSafetyCheckKind.platform:
    return _export.DisabledSafetyCheck.platform()
  if kind == ser_flatbuf.DisabledSafetyCheckKind.shape_assertions:
    return _export.DisabledSafetyCheck.shape_assertions()
  assert False, kind
