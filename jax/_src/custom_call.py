# Copyright 2022 The JAX Authors.
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

import dataclasses

from typing import Any, Dict

from jax import core
from jax import tree_util
from jax.interpreters import mlir
from jax._src.lib import xla_bridge as xb
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo

from jax._src.lib import custom_call_lib
from jax._src.lib import xla_client as xc

jax_ffi_call_wrapper = custom_call_lib.get_jax_ffi_call_wrapper()
xc.register_cpu_custom_call_target("jax_ffi_call", jax_ffi_call_wrapper)

custom_call_p = core.Primitive("jax_custom_call")
custom_call_p.multiple_results = True

def _custom_call_abstract_eval(*avals, out_avals, name, **_):
  del avals
  return out_avals
custom_call_p.def_abstract_eval(_custom_call_abstract_eval)

def _custom_call_lowering_rule(ctx: mlir.LoweringRuleContext, *in_nodes,
                               out_avals, name, descriptor):
  platform = ctx.module_context.platform
  custom_call = _CUSTOM_CALL_REGISTRY[name]
  function_ptr = custom_call._registry[platform]
  custom_call_descriptor, keepalive = custom_call_lib.make_descriptor(function_ptr, descriptor)
  print("lower 1")
  ctx.module_context.add_keepalive(keepalive)
  ctx.module_context.add_keepalive(descriptor)
  out_type = ir.TupleType.get_tuple([
      ir.RankedTensorType.get(out_aval.shape, mlir.dtype_to_ir_type(out_aval.dtype))
      for out_aval in out_avals])
  i32_type = ir.IntegerType.get_signless(32)
  descriptor_operand = mlir.ir_constant(
      custom_call_descriptor, canonicalize_types=False)
  in_nodes = [descriptor_operand, *in_nodes]
  out = mhlo.CustomCallOp(
            [out_type], in_nodes,
            call_target_name=ir.StringAttr.get("jax_ffi_call"),
            has_side_effect=ir.BoolAttr.get(False),
            backend_config=ir.StringAttr.get(""),
            api_version=ir.IntegerAttr.get(i32_type, 2),
            called_computations=ir.ArrayAttr.get([]))
  results = [mhlo.GetTupleElementOp(out, mlir.i32_attr(i)).result
             for i in range(len(out_avals))]
  print("lower 2")
  return results
mlir.register_lowering(custom_call_p, _custom_call_lowering_rule, platform="cpu")

@dataclasses.dataclass
class CustomCall:
  name: str
  _registry: Dict[str, Any] = dataclasses.field(init=False,
                                                default_factory=dict)

  def register(self, function_ptr, *, platform):
    platform = platform.lower()
    if platform != "cpu":
      raise NotImplementedError(platform)
    self._registry[platform] = function_ptr

  def __call__(self, *args, out_shape_dtype, descriptor=None):
    flat_out_shape_dtype, out_tree = tree_util.tree_flatten(out_shape_dtype)
    out_avals = [core.ShapedArray(a.shape, a.dtype)
                 for a in flat_out_shape_dtype]
    out_flat = custom_call_p.bind(*args, out_avals=out_avals,
        descriptor=descriptor, name=self.name)
    return tree_util.tree_unflatten(out_tree, out_flat)

_CUSTOM_CALL_REGISTRY: Dict[str, CustomCall] = {}

def custom_call(name: str):
  if name in _CUSTOM_CALL_REGISTRY:
    raise ValueError(f"Custom call name already used: {name}")
  _CUSTOM_CALL_REGISTRY[name] = cc = CustomCall(name)
  return cc
