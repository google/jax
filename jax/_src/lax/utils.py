# Copyright 2018 The JAX Authors.
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

# This module contains utility functions split out of jax._src.lax.lax to
# avoid cyclic dependencies. Definitions that are used at import time by
# multiple modules can go here.

from functools import partial

from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src.util import safe_zip
from jax._src.lib import xla_client

zip, unsafe_zip = safe_zip, zip

import numpy as np

xops = xla_client.ops

def _input_dtype(x, *_, **__):
  return dtypes.canonicalize_dtype(x.dtype, allow_extended_dtype=True)

def _argnum_weak_type(*argnums):
  return lambda *args, **_: all(args[i].weak_type for i in argnums)

def standard_primitive(shape_rule, dtype_rule, name,
                       weak_type_rule=None):
  weak_type_rule = weak_type_rule or _standard_weak_type_rule
  prim = core.Primitive(name)
  prim.def_impl(partial(dispatch.apply_primitive, prim))
  prim.def_abstract_eval(
      partial(standard_abstract_eval, prim, shape_rule, dtype_rule,
              weak_type_rule))
  return prim

def _get_array_abstraction_level(a): return a.array_abstraction_level

def standard_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule,
                           *avals, **kwargs):
  assert all(isinstance(aval, core.UnshapedArray) for aval in avals), avals
  assert not prim.multiple_results
  weak_type = weak_type_rule(*avals, **kwargs)
  least_specialized = type(max(avals, key=_get_array_abstraction_level))
  if least_specialized is core.ConcreteArray:
    out = prim.impl(*[x.val for x in avals], **kwargs)
    return core.ConcreteArray(out.dtype, out, weak_type=weak_type)
  elif least_specialized is core.ShapedArray:
    return core.ShapedArray(shape_rule(*avals, **kwargs),
                            dtype_rule(*avals, **kwargs), weak_type=weak_type)
  elif least_specialized is core.DShapedArray:
    shape = shape_rule(*avals, **kwargs)
    ty = (core.ShapedArray if all(type(d) is int for d in shape)
          else core.DShapedArray)
    return ty(shape, dtype_rule(*avals, **kwargs), weak_type)
  elif least_specialized is core.UnshapedArray:
    return core.UnshapedArray(dtype_rule(*avals, **kwargs), weak_type=weak_type)
  else:
    raise TypeError(avals, least_specialized)

def standard_multi_result_abstract_eval(
    prim, shape_rule, dtype_rule, weak_type_rule, *avals, **kwargs):
  assert prim.multiple_results
  assert all(isinstance(aval, core.UnshapedArray) for aval in avals), avals
  least_specialized = max(map(type, avals), key=_get_array_abstraction_level)
  weak_types = weak_type_rule(*avals, **kwargs)
  if least_specialized is core.ConcreteArray:
    out_vals = prim.impl(*[x.val for x in avals], **kwargs)
    return [core.ConcreteArray(val.dtype, val, weak_type=weak_type)
            for val, weak_type in zip(out_vals, weak_types)]
  elif least_specialized is core.ShapedArray:
    out_shapes = shape_rule(*avals, **kwargs)
    out_dtypes = dtype_rule(*avals, **kwargs)
    return [core.ShapedArray(s, d, weak_type=weak_type)
            for s, d, weak_type in zip(out_shapes, out_dtypes, weak_types)]
  elif least_specialized is core.UnshapedArray:
    out_dtypes = dtype_rule(*avals, **kwargs)
    return [core.UnshapedArray(dtype, weak_type=weak_type)
            for dtype, weak_type in zip(out_dtypes, weak_types)]
  else:
    raise TypeError(avals, least_specialized)

def standard_translate(prim):
  xla_opname = ''.join(term.capitalize() for term in prim.name.split('_'))
  op = getattr(xops, xla_opname)
  def translation_rule(ctx, avals_in, avals_out, *args, **kwargs):
    del ctx, avals_in, avals_out
    return [op(*args, **kwargs)]
  return translation_rule

def _standard_weak_type_rule(*avals, **kwargs):
  return all(aval.weak_type for aval in avals)

def dtype_to_string(dtype):
  try:
    return str(np.dtype(dtype).name)
  except TypeError:
    pass
  try:
    return dtype.name
  except AttributeError:
    pass
  return str(dtype)
