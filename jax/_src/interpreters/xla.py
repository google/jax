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

# Lowering of jaxprs into XLA (HLO) computations.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import dataclasses
import functools
from functools import partial
import itertools as it
from typing import Any, Protocol, Union

import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src import source_info_util
from jax._src.abstract_arrays import numpy_scalar_types
from jax._src.core import ConcreteArray, ShapedArray
from jax._src.sharding_impls import AxisEnv
from jax._src.util import safe_zip, safe_map

from jax._src.typing import Shape

from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

xe = xc._xla
xops = xc._xla.ops

# Types

def identity(x): return x

_scalar_types = dtypes.python_scalar_dtypes.keys()

def _make_array_shape(aval: ShapedArray) -> Sequence[xc.Shape]:
  aval = core.physical_aval(aval)
  dtype = np.dtype('bool') if aval.dtype == dtypes.float0 else aval.dtype
  return (xc.Shape.array_shape(dtype, aval.shape),)

# Utilities

def parameter(builder, num, shape, name=None, replicated=None):
  if name is None:
    name = ''
  if replicated is None:
    replicated = []
  elif isinstance(replicated, bool):
    replicated = [replicated] * shape.leaf_count()

  return xops.Parameter(builder, num,
                        shape.with_major_to_minor_layout_if_absent(), name,
                        replicated)

# HLO instructions optionally can be annotated to say how the output should be
# spatially partitioned (represented in XLA as OpSharding protos, see
# sharding_to_proto). For array outputs, the annotation is either an int per
# dimension specifying the number of ways that dimension divided (i.e. the total
# number of shards is the product), or None to indicate the array should be
# replicated. Tuple outputs are represented as tuples thereof. XLA supports
# arbitrary tuple nesting, but JAX only uses one level of tupling (and our type
# checkers don't support recursive types), so we only represent one level of
# nesting in this type definition.
SpatialSharding = Union[Shape, None, tuple[Union[Shape, None], ...]]


def sharding_to_proto(sharding: SpatialSharding):
  """Converts a SpatialSharding to an OpSharding.

  See
  https://github.com/tensorflow/tensorflow/blob/main/tensorflow/compiler/xla/xla_data.proto#L601
  for details on the OpSharding proto.
  """
  proto = xc.OpSharding()
  if isinstance(sharding, tuple) and not isinstance(sharding[0], int):
    assert all(s is None or isinstance(s, tuple) for s in sharding)
    return tuple_sharding_proto(list(map(sharding_to_proto, sharding)))

  if sharding is None:
    proto.type = xc.OpSharding.Type.REPLICATED
  else:
    proto.type = xc.OpSharding.Type.OTHER
    proto.tile_assignment_dimensions = list(sharding)  # type: ignore
    proto.tile_assignment_devices = list(range(np.prod(sharding)))  # type: ignore
  return proto

def tuple_sharding_proto(elems):
  proto = xc.OpSharding()
  assert all(isinstance(e, type(proto)) for e in elems)
  proto.type = xc.OpSharding.Type.TUPLE
  proto.tuple_shardings = elems
  return proto


### handlers

# JAX abstract values -> XLA shapes

def aval_to_xla_shapes(aval: core.AbstractValue) -> Sequence[xc.Shape]:
  try:
    return _xla_shape_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No xla_shape_handler for type: {type(aval)}") from err

_xla_shape_handlers: dict[type[core.AbstractValue],
                         Callable[[Any], Sequence[xc.Shape]]] = {
    ShapedArray: _make_array_shape,
    ConcreteArray: _make_array_shape,
}
_xla_shape_handlers[core.AbstractToken] = lambda _: (xc.Shape.token_shape(),)


# IR constants

class InvalidInputException(Exception):
  pass


# TODO(mattjj): try to remove this canonicalize_dtype stuff
def canonicalize_dtype(x):
  typ = type(x)
  handler = canonicalize_dtype_handlers.get(typ)
  if handler: return handler(x)
  for typ in typ.__mro__:
    handler = canonicalize_dtype_handlers.get(typ)
    if handler: return handler(x)
  if hasattr(x, '__jax_array__'):
    return canonicalize_dtype(x.__jax_array__())
  raise InvalidInputException(
      f"Argument '{x}' of type {type(x)} is not a valid JAX type.")

def _canonicalize_masked_array_dtype(x):
  raise ValueError("numpy masked arrays are not supported as direct inputs to JAX functions. "
                   "Use arr.filled() to convert the value to a standard numpy array.")

def _canonicalize_ndarray_dtype(x):
  return np.asarray(x, dtypes.canonicalize_dtype(x.dtype))

def _canonicalize_python_scalar_dtype(typ, x):
  return np.asarray(
      x, dtypes.canonicalize_dtype(dtypes._scalar_type_to_dtype(typ, x)))

canonicalize_dtype_handlers: dict[Any, Callable] = {}
canonicalize_dtype_handlers.update(
    (t, _canonicalize_ndarray_dtype) for t in numpy_scalar_types)
canonicalize_dtype_handlers[np.ndarray] = _canonicalize_ndarray_dtype
canonicalize_dtype_handlers[np.ma.MaskedArray] = _canonicalize_masked_array_dtype
canonicalize_dtype_handlers.update(
    (t, partial(_canonicalize_python_scalar_dtype, t)) for t in _scalar_types)
canonicalize_dtype_handlers[core.Token] = identity
canonicalize_dtype_handlers[core.DArray] = identity
canonicalize_dtype_handlers[core.MutableArray] = identity

def abstractify(x) -> Any:
  typ = type(x)
  aval_fn = pytype_aval_mappings.get(typ)
  if aval_fn: return aval_fn(x)
  for typ in typ.__mro__:
    aval_fn = pytype_aval_mappings.get(typ)
    if aval_fn: return aval_fn(x)
  if hasattr(x, '__jax_array__'):
    return abstractify(x.__jax_array__())
  raise TypeError(f"Argument '{x}' of type '{type(x)}' is not a valid JAX type")

def _make_abstract_python_scalar(typ, val):
  # Note: all python scalar types are weak except bool, because bool only
  # comes in a single width.
  return ShapedArray((), dtypes._scalar_type_to_dtype(typ, val),
                     weak_type=typ is not bool)

def _make_shaped_array_for_numpy_scalar(x: np.generic) -> ShapedArray:
  dtype = np.dtype(x)
  dtypes.check_valid_dtype(dtype)
  return ShapedArray(np.shape(x), dtypes.canonicalize_dtype(dtype))

def _make_shaped_array_for_numpy_array(x: np.ndarray) -> ShapedArray:
  dtype = x.dtype
  dtypes.check_valid_dtype(dtype)
  return ShapedArray(x.shape, dtypes.canonicalize_dtype(dtype))


pytype_aval_mappings: dict[Any, Callable[[Any], core.AbstractValue]] = {}
pytype_aval_mappings[core.DArray] = lambda x: x._aval
pytype_aval_mappings[core.MutableArray] = lambda x: x._aval
pytype_aval_mappings[core.Token] = lambda _: core.abstract_token
pytype_aval_mappings.update((t, _make_shaped_array_for_numpy_scalar)
                            for t in numpy_scalar_types)
pytype_aval_mappings[np.ndarray] = _make_shaped_array_for_numpy_array
pytype_aval_mappings.update(
    (t, partial(_make_abstract_python_scalar, t)) for t in _scalar_types)


def primitive_subcomputation(platform: str, axis_env: AxisEnv,
                             prim: core.Primitive,
                             avals_in: Sequence[core.AbstractValue],
                             avals_out: Sequence[core.AbstractValue],
                             **params):
  c = xc.XlaBuilder(f"primitive_computation_{prim.name}")
  counts = it.count()
  xla_args = [parameter(c, next(counts), xla_shape)
              for a in avals_in for xla_shape in aval_to_xla_shapes(a)]
  if (platform is not None and
      prim in _backend_specific_translations[platform]):
    rule = _backend_specific_translations[platform][prim]
  elif prim in _translations:
    rule = _translations[prim]

  ctx = TranslationContext(builder=c, platform=platform, axis_env=axis_env,
                           name_stack=source_info_util.new_name_stack())
  ans = rule(ctx, avals_in, avals_out, *xla_args, **params)

  if prim.multiple_results:
    return c.build(xops.Tuple(c, ans))
  else:
    x, = ans
    return c.build(x)


### compiling jaxprs

@dataclasses.dataclass
class TranslationContext:
  builder: xc.XlaBuilder
  # TODO(phawkins): make platform non-optional. We should always be translating
  # with a specific platform in mind.
  platform: str | None
  axis_env: AxisEnv
  name_stack: str | source_info_util.NameStack

  def replace(self, **kw): return dataclasses.replace(self, **kw)

def xla_destructure(c, ans):
  num_elements = len(c.get_shape(ans).tuple_shapes())
  return [xops.GetTupleElement(ans, i) for i in range(num_elements)]


### translation tables

MYPY = False
if not MYPY:
  class TranslationRule(Protocol):
    def __call__(self, ctx: TranslationContext,
                 avals_in: Sequence[core.AbstractValue],
                 avals_out: Sequence[core.AbstractValue],
                 *args: xc.XlaOp, **kw
                ) -> Sequence[xc.XlaOp]:
      """A translation rule lowers a primitive invocation into an XLA HLO."""
else:
  TranslationRule = Any

_translations: dict[core.Primitive, TranslationRule] = {}
_backend_specific_translations: dict[str, dict[core.Primitive, TranslationRule]]
_backend_specific_translations = defaultdict(dict)

initial_style_primitives: set[core.Primitive] = set()

def register_initial_style_primitive(prim: core.Primitive):
  initial_style_primitives.add(prim)

def register_translation(prim: core.Primitive, rule: TranslationRule, *,
                         platform: str | None = None) -> None:
  if platform is None:
    _translations[prim] = rule
  else:
    # For backward compatibility reasons, we allow rules to be registered
    # under "gpu" even though the platforms are now called "cuda" and "rocm".
    # TODO(phawkins): fix up users to specify either "cuda" or "rocm" and remove
    # this expansion.
    for p in xb.expand_platform_alias(platform):
      _backend_specific_translations[p][prim] = rule


# As a temporary backward compatibility measure, we use an adapter class to
# convert from the old styles of translation rules to the newer ones.
# TODO(phawkins): update users of the older translation rule styles and remove
# the adapters.
class _TranslationRuleAdapter:
  def __init__(self, translations,
               wrap_fn: Callable[[core.Primitive, Callable], TranslationRule]):
    self._translations = translations
    self._wrap_fn = wrap_fn

  def __setitem__(self, key: core.Primitive, value: Callable):
    wrapped = self._wrap_fn(key, value)
    for translations in self._translations:
      translations[key] = wrapped


def _wrap_old_translation(prim: core.Primitive, f: Callable) -> TranslationRule:
  @functools.wraps(f)
  def wrapped(ctx: TranslationContext, avals_in: Sequence[core.AbstractValue],
              avals_out: Sequence[core.AbstractValue],
               *args: xc.XlaOp, **kw) -> Sequence[xc.XlaOp]:
    ans = f(ctx.builder, *args, **kw)
    if (prim.multiple_results or
        any(len(aval_to_xla_shapes(aval)) > 1 for aval in avals_out)):
      return xla_destructure(ctx.builder, ans)
    else:
      return [ans]
  return wrapped


translations : _TranslationRuleAdapter
translations = _TranslationRuleAdapter([_translations], _wrap_old_translation)

class _BackendSpecificTranslationsAdapter(defaultdict):
  def __missing__(self, key):
    translation_tables = [_backend_specific_translations[p]
                          for p in xb.expand_platform_alias(key)]
    ret = self[key] = _TranslationRuleAdapter(
        translation_tables, _wrap_old_translation)
    return ret

backend_specific_translations: dict[str, _TranslationRuleAdapter]
backend_specific_translations = _BackendSpecificTranslationsAdapter()
