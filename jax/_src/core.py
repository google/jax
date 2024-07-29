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
from __future__ import annotations

from collections import Counter, defaultdict, deque, namedtuple
from collections.abc import (Callable, Collection, Generator, Hashable,
                             Iterable, Iterator, Set, Sequence, MutableSet,
                             MutableMapping)
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import functools
from functools import partial, partialmethod, total_ordering
import gc
import inspect
import itertools as it
import math
import operator
import threading
import types
from typing import (Any, ClassVar, Generic, NamedTuple, TypeVar,
                    cast, overload, Union)
import warnings
from weakref import ref

import numpy as np

from jax._src import deprecations
from jax._src import dtypes
from jax._src import config
from jax._src import effects
from jax._src import compute_on
from jax._src.errors import (
    ConcretizationTypeError, TracerArrayConversionError, TracerBoolConversionError,
    TracerIntegerConversionError, UnexpectedTracerError)
from jax._src import linear_util as lu

from jax._src import source_info_util
from jax._src.util import (safe_zip, safe_map, curry, tuple_insert,
                           tuple_delete, as_hashable_function,
                           HashableFunction, HashableWrapper, weakref_lru_cache,
                           partition_list, StrictABCMeta)
import jax._src.pretty_printer as pp
from jax._src.lib import jax_jit
from jax._src import traceback_util
from jax._src.typing import Array, DimSize, Shape
from jax._src import typing

traceback_util.register_exclusion(__file__)

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map


_TRACER_ERROR_NUM_TRACEBACK_FRAMES = config.int_flag(
    'jax_tracer_error_num_traceback_frames',
    config.int_env('JAX_TRACER_ERROR_NUM_TRACEBACK_FRAMES', 5),
    help='Set the number of stack frames in JAX tracer error messages.'
)


# -------------------- jaxprs --------------------

Effect = effects.Effect
Effects = effects.Effects
EffectTypeSet = effects.EffectTypeSet
no_effects: Effects = effects.no_effects

class JaxprDebugInfo(NamedTuple):
  traced_for: str     # e.g. 'jit', 'scan', etc
  func_src_info: str | None  # e.g. f'{fun.__name__} at {filename}:{lineno}'
  arg_names: tuple[str | None, ...]     # e.g. ('args[0]', ... )
  result_paths: tuple[str, ...]  # e.g. ('[0]', '[1]', ...)

class Jaxpr:
  __slots__ = ['__weakref__', '_constvars', '_invars', '_outvars', '_eqns',
               '_effects', '_debug_info']

  _constvars: list[Var]
  _invars: list[Var]
  _outvars: list[Atom]
  _eqns: list[JaxprEqn]
  _effects: Effects
  _debug_info: JaxprDebugInfo | None

  @property
  def constvars(self) -> list[Var]:
    return self._constvars

  @property
  def invars(self) -> list[Var]:
    return self._invars

  @property
  def outvars(self) -> list[Atom]:
    return self._outvars

  @property
  def eqns(self) -> list[JaxprEqn]:
    return self._eqns

  @property
  def effects(self) -> Effects:
    return self._effects

  @property
  def debug_info(self) -> JaxprDebugInfo | None:
    return self._debug_info

  def __init__(self, constvars: Sequence[Var], invars: Sequence[Var],
               outvars: Sequence[Atom], eqns: Sequence[JaxprEqn],
               effects: Effects = no_effects,
               debug_info: JaxprDebugInfo | None = None):
    """
    Args:
      constvars: list of variables introduced for constants. Array constants are
        replaced with such variables while scalar constants are kept inline.
      invars: list of input variables. Together, `constvars` and `invars` are
        the inputs to the Jaxpr.
      outvars: list of output atoms.
      eqns: list of equations.
      effects: set of effects. The effects on a jaxpr are a superset of the
        union of the effects for each equation.
      debug_info: optional JaxprDebugInfo.
    """
    self._constvars = list(constvars)
    self._invars = list(invars)
    self._outvars = list(outvars)
    self._eqns = list(eqns)
    self._effects = effects
    self._debug_info = debug_info
    assert (not debug_info or len(debug_info.arg_names) == len(invars) and
            len(debug_info.result_paths) == len(outvars))

  def __str__(self):
    return str(self.pretty_print())

  __repr__ = __str__

  def pretty_print(self, *, source_info=False, print_shapes=True,
                   custom_pp_eqn_rules=True, name_stack=False,
                   print_effects: bool = False, **kwargs):
    doc = pp_toplevel_jaxpr(
      self, source_info=source_info, print_shapes=print_shapes,
      custom_pp_eqn_rules=custom_pp_eqn_rules, name_stack=name_stack,
      print_effects=print_effects)
    return doc.format(**kwargs)

  def _repr_pretty_(self, p, cycle):
    return p.text(self.pretty_print(use_color=True))

  def replace(self, **kwargs):
    jaxpr = Jaxpr(
        constvars=kwargs.pop("constvars", self.constvars),
        invars=kwargs.pop("invars", self.invars),
        outvars=kwargs.pop("outvars", self.outvars),
        eqns=kwargs.pop("eqns", self.eqns),
        effects=kwargs.pop("effects", self.effects),
        debug_info=kwargs.pop("debug_info", self.debug_info),
    )
    if kwargs:
      raise ValueError(f"Unknown keyword arguments: {kwargs}")
    return jaxpr


def join_effects(*effects: Effects) -> Effects:
  return set().union(*effects) if effects else no_effects

def jaxprs_in_params(params) -> Iterator[Jaxpr]:
  for val in params.values():
    vals = val if isinstance(val, tuple) else (val,)
    for v in vals:
      if isinstance(v, Jaxpr):
        yield v
      elif isinstance(v, ClosedJaxpr):
        yield v.jaxpr


def subjaxprs(jaxpr: Jaxpr) -> Iterator[Jaxpr]:
  """Generator for all subjaxprs found in the params of jaxpr.eqns.
  Does not descend recursively into the found subjaxprs.
  """
  for eqn in jaxpr.eqns:
    yield from jaxprs_in_params(eqn.params)


class ClosedJaxpr:
  __slots__ = ['__weakref__', '_jaxpr', '_consts']

  _jaxpr: Jaxpr
  _consts: list[Any]

  jaxpr = property(lambda self: self._jaxpr)
  consts = property(lambda self: self._consts)

  def __init__(self, jaxpr: Jaxpr, consts: Sequence):
    assert len(consts) == len(jaxpr.constvars)
    # assert not any(isinstance(c, Tracer) for c in consts)  # TODO(mattjj): enable
    self._jaxpr = jaxpr
    self._consts = list(consts)

  @property
  def in_avals(self):
    return [v.aval for v in self.jaxpr.invars]

  @property
  def out_avals(self):
    return [v.aval for v in self.jaxpr.outvars]

  @property
  def literals(self):
    return self.consts  # backwards compatible alias

  @property
  def eqns(self):
    return self.jaxpr.eqns

  @property
  def effects(self) -> Effects:
    return self.jaxpr.effects

  def map_jaxpr(self, f):
    return ClosedJaxpr(f(self.jaxpr), self.consts)

  def replace(self, *, jaxpr=None, consts=None):
    jaxpr = self.jaxpr if jaxpr is None else jaxpr
    consts = self.consts if consts is None else consts
    return ClosedJaxpr(jaxpr, consts)

  def __str__(self): return str(self.jaxpr)
  def __repr__(self): return repr(self.jaxpr)

  def pretty_print(self, *, source_info=False, print_shapes=True,
                   name_stack=False, custom_pp_eqn_rules=True,
                   print_effects=False, **kwargs):
    return self.jaxpr.pretty_print(
        source_info=source_info,
        print_shapes=print_shapes,
        name_stack=name_stack,
        custom_pp_eqn_rules=custom_pp_eqn_rules,
        print_effects=print_effects,
        **kwargs)

  def _repr_pretty_(self, p, cycle):
    return p.text(self.pretty_print(use_color=True))

@curry
def jaxpr_as_fun(closed_jaxpr: ClosedJaxpr, *args):
  return eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)


class JaxprEqnContext:

  def __init__(self, compute_type: str | None, threefry_partitionable: bool):
    self.compute_type = compute_type
    self.threefry_partitionable = threefry_partitionable
    self._managers = [
        (compute_on.extend_compute_type, self.compute_type),
        (config.threefry_partitionable.__call__, self.threefry_partitionable),
    ]

  @property
  @contextmanager
  def manager(self):
    with ExitStack() as stack:
      for manager, val in self._managers:
        stack.enter_context(manager(val))
      yield

  def __repr__(self):
    return (f"JaxprEqnContext(compute_type={self.compute_type},"
            f"threefry_partitionable={self.threefry_partitionable})")


class JaxprEqn:
  invars: list[Atom]
  outvars: list[Var]
  primitive: Primitive
  params: dict[str, Any]
  effects: Effects
  source_info: source_info_util.SourceInfo
  ctx: JaxprEqnContext

  # It's slightly faster to use a class with __slots__ than a NamedTuple.
  __slots__ = ['invars', 'outvars', 'primitive', 'params', 'effects',
               'source_info', 'ctx']

  def __init__(self, invars, outvars, primitive, params, effects, source_info,
               ctx):
    self.invars = invars
    self.outvars = outvars
    self.primitive = primitive
    self.params = params
    self.effects = effects
    self.source_info = source_info
    self.ctx = ctx

  def __repr__(self):
    return str(pp_eqn(self, JaxprPpContext(), JaxprPpSettings())).rstrip()

  def replace(
      self,
      invars: list[Atom] | None = None,
      outvars: list[Var] | None = None,
      primitive: Primitive | None = None,
      params: dict[str, Any] | None = None,
      effects: Effects | None = None,
      source_info: source_info_util.SourceInfo | None = None,
      ctx: JaxprEqnContext | None = None
  ):
    return JaxprEqn(
      self.invars if invars is None else invars,
      self.outvars if outvars is None else outvars,
      self.primitive if primitive is None else primitive,
      self.params if params is None else params,
      self.effects if effects is None else effects,
      self.source_info if source_info is None else source_info,
      self.ctx if ctx is None else ctx,
    )


# TODO(mattjj): call typecheck rules here, so we don't form bad eqns
def new_jaxpr_eqn(invars, outvars, primitive, params, effects, source_info=None,
                  ctx=None):
  source_info = source_info or source_info_util.new_source_info()
  ctx = ctx or JaxprEqnContext(compute_on.current_compute_type(),
                               config.threefry_partitionable.value)
  if config.enable_checks.value:
    assert all(isinstance(x, (Var, Literal)) for x in  invars)
    assert all(isinstance(v,  Var)           for v in outvars)
  return JaxprEqn(invars, outvars, primitive, params, effects, source_info, ctx)

_var_counter = it.count()

@total_ordering
class Var:
  __slots__ = ["count", "suffix", "aval"]

  count: int
  suffix: str
  aval: AbstractValue

  def __init__(self, suffix: str, aval: AbstractValue):
    self.count = next(_var_counter)
    self.suffix = suffix
    self.aval = raise_to_shaped(aval)

  # TODO(phawkins, mattjj): remove ordering of variables. JAX itself does not
  # care about variable ordering, but the downstream package kfac_jax does.
  def __lt__(self, other):
    return self.count < other.count

  def __repr__(self):
    return f'Var(id={id(self)}){self.suffix}:{self.aval.str_short()}'


def gensym(suffix: str = '') -> Callable[[AbstractValue], Var]:
  """Produce distinct variables, printed with the optional suffix."""
  return partial(Var, suffix)

# In a jaxpr, `dropvar` can appear in place of a bound variable to indicate that
# the assignment is dropped, i.e. that an expression's output value will never
# be read. In that sense, `dropvar` is not a variable, but it is convenient to
# treat it as a special case of one. Its `aval` is similarly inexact.
class DropVar(Var):
  def __init__(self, aval: AbstractValue):
    super().__init__('', aval)
  def __repr__(self): return '_'

class Literal:
  __slots__ = ["val", "aval", "hash"]

  val: Any
  aval: AbstractValue
  hash: int | None

  def __init__(self, val, aval):
    self.val = val
    self.aval = aval
    try:
      self.hash = hash(val)
    except TypeError:
      if type(val) in literalable_types:
        try:
          self.hash = hash((val.item(), val.dtype))
        except (TypeError, AttributeError, ValueError):
          self.hash = None

  __hash__ = None  # type: ignore

  def __repr__(self):
    if hasattr(self, 'hash'):
      return f'{self.val}'
    else:
      return f'Literal(val={self.val})'

literalable_types: set[type] = set()

Atom = Union[Var, Literal]

class Primitive:
  name: str
  # set for multi-output primitives.
  multiple_results: bool = False
  # set for call primitives processed in final style.
  call_primitive: bool = False
  # set for map primitives processed in final style.
  map_primitive: bool = False

  def __init__(self, name: str):
    self.name = name

  def __repr__(self):
    return f'{self.name}'

  def bind(self, *args, **params):
    assert (not config.enable_checks.value or
            all(isinstance(arg, Tracer) or valid_jaxtype(arg) for arg in args)), args
    return self.bind_with_trace(find_top_trace(args), args, params)

  def bind_with_trace(self, trace, args, params):
    with pop_level(trace.level):
      out = trace.process_primitive(self, map(trace.full_raise, args), params)
    return map(full_lower, out) if self.multiple_results else full_lower(out)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_abstract_eval(self, abstract_eval):
    self.abstract_eval = _effect_free_abstract_eval(abstract_eval)
    return abstract_eval

  def def_effectful_abstract_eval(self, effectful_abstract_eval):
    self.abstract_eval = effectful_abstract_eval
    return effectful_abstract_eval

  def def_custom_bind(self, bind):
    self.bind = bind
    return bind

  def impl(self, *args, **params):
    raise NotImplementedError("Evaluation rule for '{}' not implemented"
                              .format(self.name))

  def abstract_eval(self, *args, **params):
    raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                              .format(self.name))

  def get_bind_params(self, params):
    return [], params


def _effect_free_abstract_eval(abstract_eval):
  def abstract_eval_(*args, **kwargs):
    return abstract_eval(*args, **kwargs), no_effects
  return abstract_eval_

# -------------------- lifting --------------------

# TODO(mattjj): replace this approach with a primitive-keyed table of rules
def traverse_jaxpr_params(f, params):
  """Applies f to each jaxpr parameter and returns a tuple of returned values."""
  return {name: f(p)
          for name, param in params.items()
          for p in (param if isinstance(param, (tuple, list)) else [param])
          if type(p) in (Jaxpr, ClosedJaxpr)}


def eval_jaxpr(jaxpr: Jaxpr, consts, *args, propagate_source_info=True) -> list[Any]:
  def read(v: Atom) -> Any:
    return v.val if isinstance(v, Literal) else env[v]

  def write(v: Var, val: Any) -> None:
    if config.enable_checks.value and not config.dynamic_shapes.value:
      assert typecheck(v.aval, val), (v.aval, val)
    env[v] = val

  env: dict[Var, Any] = {}
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  lu = last_used(jaxpr)
  for eqn in jaxpr.eqns:
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
    traceback = eqn.source_info.traceback if propagate_source_info else None
    with source_info_util.user_context(
        traceback, name_stack=name_stack), eqn.ctx.manager:
      ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
    clean_up_dead_vars(eqn, env, lu)
  return map(read, jaxpr.outvars)


# -------------------- tracing --------------------

TracerType = TypeVar('TracerType', bound='Tracer')

class Trace(Generic[TracerType]):
  __slots__ = ['main', 'level', 'sublevel']

  main: MainTrace
  level: int
  sublevel: Sublevel

  def __init__(self, main: MainTrace, sublevel: Sublevel) -> None:
    self.main = main
    self.level = main.level
    self.sublevel = sublevel

  def full_raise(self, val) -> TracerType:
    if not isinstance(val, Tracer):
      # This check is only applied to non-Tracers, because the hasattr() is
      # expensive (Tracer.__getattr__) in the common case that val is a Tracer.
      if hasattr(val, "dimension_as_value"):  # Used for shape_poly._DimExpr
        val = val.dimension_as_value()
        if not isinstance(val, Tracer):
          return self.pure(val)
      else:
        return self.pure(val)
    val._assert_live()
    level = self.level
    sublevel = self.sublevel
    if val._trace.main is self.main:
      if val._trace.sublevel == sublevel:
        return cast(TracerType, val)
      elif val._trace.sublevel < sublevel:
        return self.sublift(val)
      else:
        raise escaped_tracer_error(
            val, f"Can't lift sublevels {val._trace.sublevel} to {sublevel}")
    elif val._trace.level < level:
      if val._trace.sublevel > sublevel:
        raise escaped_tracer_error(
            val, f"Incompatible sublevel: {val._trace}, {(level, sublevel)}")
      return self.lift(val)
    elif val._trace.level > level:
      raise escaped_tracer_error(
          val, f"Can't lift level {val} to {self}")
    else:  # val._trace.level == self.level:
      raise escaped_tracer_error(
          val, f"Different traces at same level: {val}, {self}")

  def pure(self, val) -> TracerType:
    raise NotImplementedError("must override")

  def lift(self, tracer) -> TracerType:
    raise NotImplementedError("must override")

  def sublift(self, tracer) -> TracerType:
    raise NotImplementedError("must override")

  def process_primitive(self, primitive, tracers, params):
    raise NotImplementedError("must override")

  def __repr__(self):
    return '{}(level={}/{})'.format(
        self.__class__.__name__, self.level, self.sublevel)

  def process_call(self, call_primitive, f, tracers, params):
    msg = (f"{type(self)} must override process_call to handle call-like "
           "primitives")
    raise NotImplementedError(msg)

  def process_map(self, map_primitive, f, tracers, params):
    msg = (f"{type(self)} must override process_map to handle map-like "
           "primitives")
    raise NotImplementedError(msg)

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *,
                              symbolic_zeros):
    msg = (f"{type(self)} must override process_custom_jvp_call "
           "to handle custom_jvp primitives")
    raise NotImplementedError(msg)

  def process_custom_transpose(self, prim, call, tracers, **params):
    msg = (f"{type(self)} must override process_custom_transpose "
           "to handle custom_transpose_call primitives")
    raise NotImplementedError(msg)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers,
                              out_trees, symbolic_zeros):
    msg = (f"{type(self)} must override process_custom_vjp_call "
           "to handle custom_vjp primitives")
    raise NotImplementedError(msg)


def raise_as_much_as_possible(tracer) -> Tracer:
  # Find effective bottom of trace stack (highest dynamic Trace on the stack).
  trace_stack = thread_local_state.trace_state.trace_stack.stack
  idx = next(i for i, m in enumerate(trace_stack) if m is
             thread_local_state.trace_state.trace_stack.dynamic)

  # Only pay attention to effective part of trace stack.
  trace_stack = trace_stack[idx:]

  # Lift tracer into everything in the effective stack higher than its level
  for trace in trace_stack:
    trace = trace.with_cur_sublevel()
    if (not isinstance(tracer, Tracer) or tracer._trace.level < trace.level):
      tracer = trace.full_raise(tracer)

  return tracer


def escaped_tracer_error(tracer, detail=None):
  num_frames = _TRACER_ERROR_NUM_TRACEBACK_FRAMES.value
  msg = ('Encountered an unexpected tracer. A function transformed by JAX '
         'had a side effect, allowing for a reference to an intermediate value '
         f'with type {tracer.aval.str_short()} wrapped in a '
         f'{type(tracer).__name__} to escape the scope of the transformation.\n'
         'JAX transformations require that functions explicitly return their '
         'outputs, and disallow saving intermediate values to global state.')
  dbg = getattr(tracer, '_debug_info', None)
  if dbg is not None:
    msg += ('\nThe function being traced when the value leaked was '
            f'{dbg.func_src_info} traced for {dbg.traced_for}.')
  line_info = getattr(tracer, '_line_info', None)
  if line_info is not None:
    divider = '\n' + '-'*30 + '\n'
    msg += divider
    msg += ('The leaked intermediate value was created on line '
            f'{source_info_util.summarize(line_info)}. ')
    msg += divider
    if num_frames > 0:
      msg += (f'When the value was created, the final {num_frames} stack '
              'frames (most recent last) excluding JAX-internal frames were:')
      msg += divider + source_info_util.summarize(
          line_info, num_frames=num_frames) + divider
  msg += ('\nTo catch the leak earlier, try setting the environment variable '
          'JAX_CHECK_TRACER_LEAKS or using the `jax.checking_leaks` context '
          'manager.')
  if detail:
    msg += f'Detail: {detail}'
  return UnexpectedTracerError(msg)


def check_scalar_conversion(arr: Array):
  if arr.ndim > 0:
    raise TypeError("Only scalar arrays can be converted to Python scalars; "
                    f"got {arr.ndim=}")


def check_integer_conversion(arr: Array):
  if not (arr.shape == () and dtypes.issubdtype(arr.dtype, np.integer)):
    raise TypeError("Only integer scalar arrays can be converted to a scalar index.")


def check_bool_conversion(arr: Array):
  if arr.size == 0:
    raise ValueError("The truth value of an empty array is ambiguous. Use"
                     " `array.size > 0` to check that an array is not empty.")
  if arr.size > 1:
    raise ValueError("The truth value of an array with more than one element"
                     " is ambiguous. Use a.any() or a.all()")


def _aval_property(name):
  return property(lambda self: getattr(self.aval, name))


class Tracer(typing.Array, metaclass=StrictABCMeta):
  __array_priority__ = 1000
  __slots__ = ['_trace', '_line_info']

  dtype = _aval_property('dtype')
  ndim = _aval_property('ndim')
  size = _aval_property('size')
  shape = _aval_property('shape')

  def __hash__(self):
    # TODO(jakevdp) finalize this deprecation and set __hash__ = None
    # Warning added 2024-06-13
    if deprecations.is_accelerated('tracer-hash'):
      raise TypeError(f"unhashable type: {type(self)}")
    # Use FutureWarning rather than DeprecationWarning because hash is likely
    # not called directly by the user, so we want to warn at all stacklevels.
    warnings.warn(
      f"unhashable type: {type(self)}. Attempting to hash a tracer will lead to an"
      " error in a future JAX release.", category=FutureWarning)
    return super().__hash__()

  def __init__(self, trace: Trace):
    self._trace = trace

  def _error_repr(self):
    if self.aval is None:
      return f"traced array with aval {self.aval}"
    return f"traced array with shape {raise_to_shaped(self.aval).str_short()}"

  def __array__(self, *args, **kw):
    raise TracerArrayConversionError(self)

  def __dlpack__(self, *args, **kw):
    raise ConcretizationTypeError(self,
      f"The __dlpack__() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def tolist(self):
    raise ConcretizationTypeError(self,
      f"The tolist() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def tobytes(self, order="C"):
    del order
    raise ConcretizationTypeError(self,
      f"The tobytes() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def __iter__(self):
    return iter(self.aval._iter(self))

  def __reversed__(self):
    return iter(self[::-1])

  def __len__(self):
    return self.aval._len(self)

  @property
  def sharding(self):
    # This attribute is part of the jax.Array API, but only defined on concrete arrays.
    # Raising a ConcretizationTypeError would make sense, but for backward compatibility
    # we raise an AttributeError so that hasattr() and getattr() work as expected.
    raise AttributeError(self,
      f"The 'sharding' attribute is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def device(self):
    # This attribute is part of the jax.Array API, but only defined on concrete arrays.
    # Raising a ConcretizationTypeError would make sense, but for backward compatibility
    # we raise an AttributeError so that hasattr() and getattr() work as expected.
    raise AttributeError(self,
      f"The 'device' attribute is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def addressable_shards(self):
    raise ConcretizationTypeError(self,
      f"The 'addressable_shards' attribute is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def at(self):
    return self.aval.at.fget(self)

  @property
  def aval(self):
    raise NotImplementedError("must override")

  def _assert_live(self) -> None:
    pass  # Override for liveness checking

  def get_referent(self) -> Any:
    return self  # Override for object equivalence checking

  def __bool__(self):
    check_bool_conversion(self)
    return self.aval._bool(self)

  def __int__(self):
    check_scalar_conversion(self)
    return self.aval._int(self)

  def __float__(self):
    check_scalar_conversion(self)
    return self.aval._float(self)

  def __complex__(self):
    check_scalar_conversion(self)
    return self.aval._complex(self)

  def __hex__(self):
    check_integer_conversion(self)
    return self.aval._hex(self)

  def __oct__(self):
    check_integer_conversion(self)
    return self.aval._oct(self)

  def __index__(self):
    check_integer_conversion(self)
    raise self.aval._index(self)

  # raises a useful error on attempts to pickle a Tracer.
  def __reduce__(self):
    raise ConcretizationTypeError(
      self, ("The error occurred in the __reduce__ method, which may "
             "indicate an attempt to serialize/pickle a traced value."))

  # raises the better error message from ShapedArray
  def __setitem__(self, idx, val): return self.aval._setitem(self, idx, val)

  # NumPy also only looks up special methods on classes.
  def __array_module__(self, types): return self.aval._array_module(self, types)

  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    assert not config.enable_checks.value or name != "aval"

    try:
      attr = getattr(self.aval, name)
    except AttributeError as err:
      raise AttributeError(
          f"{self.__class__.__name__} has no attribute {name}"
      ) from err
    else:
      t = type(attr)
      if t is aval_property:
        return attr.fget(self)
      elif t is aval_method:
        return types.MethodType(attr.fun, self)
      else:
        return attr

  def _pretty_print(self):
    base = pp.text(f'Traced<{self.aval}>with<{self._trace}>')
    contents = [(name, attr._pretty_print() if isinstance(attr, Tracer)
                 else pp.text(repr(attr))) for name, attr in self._contents()]
    if contents:
      base = pp.group(pp.nest(2, pp.concat([
        base, pp.text(' with'), pp.brk(), pp.join(pp.brk(), [
          pp.text(f'{name} = ') + pp_payload
          for name, pp_payload in contents])
      ])))
    return base

  def __repr__(self):
    return self._pretty_print().format()

  def _contents(self):
    try:
      return [(name, getattr(self, name)) for name in self.__slots__]
    except AttributeError:
      return ()

  def _origin_msg(self) -> str:
    return ""

  # Methods that are only valid for materialized arrays
  def addressable_data(self, index):
    raise ConcretizationTypeError(self,
      f"The addressable_data() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def block_until_ready(self):
    # Raise AttributeError for backward compatibility with hasattr() and getattr() checks.
    raise AttributeError(self,
      f"The 'block_until_ready' method is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def copy_to_host_async(self):
    # Raise AttributeError for backward compatibility with hasattr() and getattr() checks.
    raise AttributeError(self,
      f"The 'copy_to_host_async' method is not available on {self._error_repr()}."
      f"{self._origin_msg()}")

  def delete(self):
    raise ConcretizationTypeError(self,
      f"The delete() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def devices(self):
    raise ConcretizationTypeError(self,
      f"The devices() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def global_shards(self):
    raise ConcretizationTypeError(self,
      f"The global_shards property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def is_deleted(self):
    raise ConcretizationTypeError(self,
      f"The is_deleted() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def is_fully_addressable(self):
    raise ConcretizationTypeError(self,
      f"The is_fully_addressable property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def is_fully_replicated(self):
    raise ConcretizationTypeError(self,
      f"The is_fully_replicated property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def on_device_size_in_bytes(self):
    raise ConcretizationTypeError(self,
      f"The on_device_size_in_bytes() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  @property
  def traceback(self):
    raise ConcretizationTypeError(self,
      f"The traceback property was called on {self._error_repr()}."
      f"{self._origin_msg()}")

  def unsafe_buffer_pointer(self):
    raise ConcretizationTypeError(self,
      f"The unsafe_buffer_pointer() method was called on {self._error_repr()}."
      f"{self._origin_msg()}")

# these can be used to set up forwarding of properties and instance methods from
# Tracer instances to the underlying avals
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])


class EvalTrace(Trace):
  # See comments in https://github.com/google/jax/pull/3370
  def pure(self, x): return x
  lift = sublift = pure

  def process_primitive(self, primitive, tracers, params):
    if config.debug_key_reuse.value:
      # Import here to avoid circular imports
      from jax.experimental.key_reuse._core import call_impl_with_key_reuse_checks  # pytype: disable=import-error
      return call_impl_with_key_reuse_checks(primitive, primitive.impl, *tracers, **params)
    else:
      return primitive.impl(*tracers, **params)

  def process_call(self, primitive, f, tracers, params):
    if config.debug_key_reuse.value:
      # Import here to avoid circular imports
      from jax.experimental.key_reuse._core import call_impl_with_key_reuse_checks  # pytype: disable=import-error
      return call_impl_with_key_reuse_checks(primitive, primitive.impl, f, *tracers, **params)
    else:
      return primitive.impl(f, *tracers, **params)
  process_map = process_call

  def process_custom_transpose(self, primitive, call, tracers, **_):
    del primitive, _
    with new_sublevel():
      return call.call_wrapped(*tracers)

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers, **_):
    del primitive, jvp, _  # Unused.
    with new_sublevel():
      return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, **_):  # pytype: disable=signature-mismatch
    del primitive, fwd, bwd, _  # Unused.
    with new_sublevel():
      return fun.call_wrapped(*tracers)


class MainTrace:
  level: int
  trace_type: type[Trace]
  payload: dict[str, Any]

  def __init__(self, level, trace_type, **payload) -> None:
    self.level = level
    self.trace_type = trace_type
    self.payload = payload

  def __repr__(self) -> str:
    return f"MainTrace({self.level},{self.trace_type.__name__})"

  def __hash__(self) -> int:
    return hash((self.level, self.trace_type))

  def __eq__(self, other: object) -> bool:
    return (isinstance(other, MainTrace) and
            self.level == other.level and
            self.trace_type == other.trace_type and
            self.payload == other.payload)

  def with_cur_sublevel(self):
    return self.trace_type(self, cur_sublevel(), **self.payload)

class TraceStack:
  # See comments in https://github.com/google/jax/pull/3370
  stack: list[MainTrace]
  dynamic: MainTrace

  def __init__(self):
    eval_trace = MainTrace(0, EvalTrace)
    self.stack = [eval_trace]
    self.dynamic = eval_trace

  def next_level(self) -> int:
    return len(self.stack)

  def push(self, main_trace: MainTrace) -> None:
    self.stack.append(main_trace)

  def pop(self) -> None:
    self.stack.pop()

  def __repr__(self) -> str:
    stack_str = map('  {}\n'.format, self.stack[::-1])
    return f'Trace stack\n{stack_str}\n{self.dynamic}'

  def copy(self):
    new = self.__new__(TraceStack)
    new.stack = self.stack[:]
    new.dynamic = self.dynamic
    return new


@total_ordering
class Sublevel:

  def __init__(self, level: int):
    self.level = level

  def __repr__(self):
    return str(self.level)

  def __eq__(self, other):
    return type(other) is Sublevel and self.level == other.level

  def __lt__(self, other):
    return type(other) is Sublevel and self.level < other.level


AxisEnvFrame = namedtuple('AxisEnvFrame', ['name', 'size', 'main_trace'])
AxisName = Hashable

no_axis_name = object()

class TraceState:
  trace_stack: TraceStack
  substack: list[Sublevel]
  axis_env: list[AxisEnvFrame]

  def __init__(self) -> None:
    self.trace_stack = TraceStack()
    self.substack = [Sublevel(0)]
    self.axis_env = []

  def copy(self):
    new = self.__new__(TraceState)
    new.trace_stack = self.trace_stack.copy()
    new.substack = self.substack[:]
    new.axis_env = self.axis_env[:]
    return new


def _update_thread_local_jit_state(dynamic):
  state = (dynamic.level, dynamic.trace_type)
  config.update_thread_local_jit_state(dynamic_trace_state=state)


# The global state of the tracer is accessed by a thread-local object.
# This allows concurrent tracing in separate threads; passing traced objects
# between threads is forbidden.
class ThreadLocalState(threading.local):
  def __init__(self):
    self.trace_state = TraceState()

thread_local_state = ThreadLocalState()


def _initialize_jax_jit_thread_local_state():
  """Initializes the C++ thread-local context.

  When the user spawns threads, the C++ `jax_jit.thread_local_state` is None.
  The C++ accessor calls this function if it realizes the thread_local_state
  is None (which means it's not yet initialized for this thread).

  This function does not live in `config.py`, to prevent circular imports.
  """
  tls = jax_jit.thread_local_state()
  if tls.extra_jit_context is None:
    dynamic = thread_local_state.trace_state.trace_stack.dynamic
    state = (dynamic.level, dynamic.trace_type)
    config.update_thread_local_jit_state(dynamic_trace_state=state)


jax_jit.set_thread_local_state_initialization_callback(
    _initialize_jax_jit_thread_local_state)

def trace_state_clean() -> bool:
  trace_state = thread_local_state.trace_state
  return (trace_state.substack == [Sublevel(0)] and
          trace_state.axis_env == [] and
          trace_state.trace_stack.stack == [MainTrace(0, EvalTrace)] and
          trace_state.trace_stack.dynamic == MainTrace(0, EvalTrace))

def reset_trace_state() -> bool:
  """Resets the global trace state and returns True if it was already clean."""
  if not trace_state_clean():
    thread_local_state.trace_state.__init__()
    return False
  else:
    return True

def cur_sublevel() -> Sublevel:
  return thread_local_state.trace_state.substack[-1]

TRACER_LEAK_DEBUGGER_WARNING = """\
JAX check_tracer_leaks behavior can trigger false positives when used with a debugger.
To avoid false positives and silence this warning, you can disable thread tracing using
the following:

  import threading
  threading.current_thread().pydev_do_not_trace = True
"""

def maybe_find_leaked_tracers(x: MainTrace | Sublevel | None
                              ) -> list[Tracer]:
  """Find the leaked tracers holding a reference to the MainTrace or SubLevel.

  It's possible there's none! eg. there's some cases where JAX itself holds a
  reference to `x` inside of a lambda closure, and no tracers were leaked
  by the user. In this case an empty list is returned.
  """
  if not getattr(threading.current_thread(), 'pydev_do_not_trace', True):
    warnings.warn(TRACER_LEAK_DEBUGGER_WARNING)
  # Trigger garbage collection to filter out unreachable objects that are alive
  # only due to cyclical dependencies. (We don't care about unreachable leaked
  # tracers since they can't interact with user code and cause a problem.)
  gc.collect()
  traces = list(filter(lambda x: isinstance(x, Trace), gc.get_referrers(x)))
  tracers = list(filter(lambda x: isinstance(x, Tracer), gc.get_referrers(*traces)))
  return tracers

def leaked_tracer_error(name: str, t, tracers: list[Tracer]) -> Exception:
  assert tracers
  why = partial(_why_alive, {id(tracers)})
  msgs = '\n\n'.join(f'{tracers[i]}{tracers[i]._origin_msg()}{why(tracers[i])}'
                     for i in range(len(tracers)))
  return Exception(f'Leaked {name} {t}. Leaked tracer(s):\n\n{msgs}\n')

def _why_alive(ignore_ids: set[int], x: Any) -> str:
  parents = lambda x: [r for r in gc.get_referrers(x) if id(r) not in ignore_ids]
  child, lines, seen = x, [], set()
  while (id(child) not in seen and type(child) is not types.ModuleType
         and parents(child)):
    parent = parents(child)[0]  # just pick one parent

    # For namespaces (like modules and class instances) and closures, the
    # references may form a simple chain: e.g. instance refers to its own
    # __dict__ which refers to child, or function refers to its __closure__
    # which refers to cells which refer to child. In these cases, we can provide
    # a more intuitive description by collapsing the chain into a single
    # parent->child jump. We do that by setting `parent` here to be a
    # grandparent (or great-grandparent) of `child`, and then handling that case
    # in _why_alive_container_info. See example:
    #  https://github.com/google/jax/pull/13022#discussion_r1008456599
    # To prevent this collapsing behavior, just comment out this code block.
    if (isinstance(parent, dict) and
        getattr(parents(parent)[0], '__dict__', None) is parents(child)[0]):
      parent = parents(parent)[0]
    elif type(parent) is types.CellType:
      parent = parents(parents(parent)[0])[0]

    line = f'<{type(child).__name__} {id(child)}> is referred to by '
    lines.append(line + _why_alive_container_info(parent, id(child)))
    seen.add(id(child))
    child = parent
  return '\n' + '\n'.join(lines) if lines else ''

def _why_alive_container_info(container, obj_id) -> str:
  name = f'<{type(container).__name__} {id(container)}>'
  if type(container) is types.ModuleType:
    name = getattr(container, '__name__', name)
  if type(container) is types.FunctionType:
    name_ = getattr(container, '__name__', '<no-name>')
    closure = inspect.getclosurevars(container)
    keys = [k for k, v in dict(closure.nonlocals, **closure.globals).items()
            if id(v) == obj_id]
    if len(keys) == 1: return f'{name} ({name_}) closed-over variable {keys[0]}'
    elif len(keys) > 1: return (f'{name} in closed-over variables ' +
                                ', '.join(map(repr, keys)))
  if hasattr(container, '__dict__'):
    keys = [k for k in vars(container) if id(vars(container)[k]) == obj_id]
    if len(keys) == 1: return f'{name}.{keys[0]}'
    elif len(keys) > 1: return f'{name} in vars ' + ', '.join(map(repr, keys))
  if isinstance(container, (list, tuple)):
    idxs = [i for i, x in enumerate(container) if id(x) == obj_id]
    if len(idxs) == 1: return f'{name}[{idxs[0]}]'
    else: return f'{name} at indices ' + ', '.join(map(str, idxs))
  if isinstance(container, dict):
    keys = [k for k in container if id(container[k]) == obj_id]
    if len(keys) == 1: return f'{name}[{keys[0]!r}]'
    else: return f'{name} at keys ' + ', '.join(map(repr, keys))
  if isinstance(container, types.ModuleType):
    return f' named {container.__name__}'
  return name


@contextmanager
def new_main(trace_type: type[Trace], dynamic: bool = False,
             **payload) -> Generator[MainTrace, None, None]:
  # See comments in https://github.com/google/jax/pull/3370
  stack = thread_local_state.trace_state.trace_stack
  level = stack.next_level()
  main = MainTrace(level, trace_type, **payload)
  stack.push(main)
  if dynamic:
    prev_dynamic, stack.dynamic = stack.dynamic, main
    _update_thread_local_jit_state(stack.dynamic)

  try:
    yield main
  finally:
    stack.pop()
    if dynamic:
      stack.dynamic = prev_dynamic
      _update_thread_local_jit_state(stack.dynamic)

  if config.check_tracer_leaks.value:
    t = ref(main)
    del main
    if t() is not None:
      leaked_tracers = maybe_find_leaked_tracers(t())
      if leaked_tracers: raise leaked_tracer_error("trace", t(), leaked_tracers)

@contextmanager
def new_dynamic(level: int) -> Generator[None, None, None]:
  stack = thread_local_state.trace_state.trace_stack
  prev_dynamic, stack.dynamic = stack.dynamic, stack.stack[level]
  _update_thread_local_jit_state(stack.dynamic)
  try:
    yield
  finally:
    stack.dynamic = prev_dynamic
    _update_thread_local_jit_state(stack.dynamic)

def dynamic_level() -> int:
  return thread_local_state.trace_state.trace_stack.dynamic.level

@contextmanager
def new_base_main(trace_type: type[Trace],
                  **payload) -> Generator[MainTrace, None, None]:
  # See comments in https://github.com/google/jax/pull/3370
  stack = thread_local_state.trace_state.trace_stack
  main = MainTrace(0, trace_type, **payload)
  prev_dynamic, stack.dynamic = stack.dynamic, main
  prev_base, stack.stack[0] = stack.stack[0], main
  _update_thread_local_jit_state(stack.dynamic)
  try:
    yield main
  finally:
    stack.dynamic = prev_dynamic
    stack.stack[0] = prev_base
    _update_thread_local_jit_state(stack.dynamic)

  if config.check_tracer_leaks.value:
    t = ref(main)
    del main
    if t() is not None:
      leaked_tracers = maybe_find_leaked_tracers(t())
      if leaked_tracers: raise leaked_tracer_error("trace", t(), leaked_tracers)

@contextmanager
def pop_level(level: int):
  if level == 0:
    return (yield)
  prev, thread_local_state.trace_state.trace_stack.stack = \
      thread_local_state.trace_state.trace_stack.stack, \
      thread_local_state.trace_state.trace_stack.stack[:level]
  try:
    yield
  finally:
    thread_local_state.trace_state.trace_stack.stack = prev

@contextmanager
def ensure_compile_time_eval():
  """Context manager to ensure evaluation at trace/compile time (or error).

  Some JAX APIs like :func:`jax.jit` and :func:`jax.lax.scan` involve staging,
  i.e., delaying the evaluation of numerical expressions (like :mod:`jax.numpy`
  function applications) so that instead of performing those computations
  eagerly while evaluating the corresponding Python expressions, their
  computation is carried out separately, e.g. after optimized compilation. But
  this delay can be undesirable. For example, numerical values might be needed
  to evaluate Python control flow and so their evaluation cannot be delayed. As
  another example, it may be beneficial to ensure compile time evaluation (or
  "constant folding") for performance reasons.

  This context manager ensures that JAX computations are evaluated eagerly. If
  eager evaluation is not possible, a ``ConcretizationTypeError`` is raised.

  Here's a contrived example::

    import jax
    import jax.numpy as jnp

    @jax.jit
    def f(x):
      with jax.ensure_compile_time_eval():
        y = jnp.sin(3.0)
        z = jnp.sin(y)
        z_positive = z > 0
      if z_positive:  # z_positive is usable in Python control flow
        return jnp.sin(x)
      else:
        return jnp.cos(x)

  Here's a real-world example from https://github.com/google/jax/issues/3974::

    import jax
    import jax.numpy as jnp
    from jax import random

    @jax.jit
    def jax_fn(x):
      with jax.ensure_compile_time_eval():
        y = random.randint(random.key(0), (1000,1000), 0, 100)
      y2 = y @ y
      x2 = jnp.sum(y2) * x
      return x2

  A similar behavior can often be achieved simply by 'hoisting' the constant
  expression out of the corresponding staging API::

    y = random.randint(random.key(0), (1000,1000), 0, 100)

    @jax.jit
    def jax_fn(x):
      y2 = y @ y
      x2 = jnp.sum(y2)*x
      return x2

  But in some cases it can be more convenient to use this context manager.
  """
  with new_base_main(EvalTrace):
    yield
eval_context = ensure_compile_time_eval  # alias, backward compatibility

@contextmanager
def new_sublevel() -> Generator[None, None, None]:
  sublevel = Sublevel(len(thread_local_state.trace_state.substack))
  thread_local_state.trace_state.substack.append(sublevel)
  try:
    yield
  finally:
    thread_local_state.trace_state.substack.pop()

  if config.check_tracer_leaks.value:
    t = ref(sublevel)
    del sublevel
    if t() is not None:
      leaked_tracers = maybe_find_leaked_tracers(t())
      if leaked_tracers:
        raise leaked_tracer_error("sublevel", t(), leaked_tracers)

def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val


def _get_trace_level(t: Tracer) -> int: return t._trace.level


def find_top_trace(xs) -> Trace:
  top_tracer = max((x for x in xs if isinstance(x, Tracer)),
                    default=None, key=_get_trace_level)
  if top_tracer is not None:
    top_tracer._assert_live()
    top_main = top_tracer._trace.main
  else:
    top_main = None
  dynamic = thread_local_state.trace_state.trace_stack.dynamic
  top_main = (dynamic if top_main is None or dynamic.level > top_main.level
              else top_main)
  return top_main.with_cur_sublevel()

def get_referent(x: Any) -> Any:
  return x.get_referent() if isinstance(x, Tracer) else x

def same_referent(x: Any, y: Any) -> bool:
  return get_referent(x) is get_referent(y)

def dedup_referents(itr: Iterable[Any]) -> list[Any]:
  return list({HashableWrapper(get_referent(x)):x for x in itr}.values())

def definitely_equal(x, y):
  if isinstance(x, Tracer) or isinstance(y, Tracer):
    return same_referent(x, y)
  elif x is y:
    return True
  try:
    return x == y
  except InconclusiveDimensionOperation:
    return False

# -------------------- abstract values --------------------

class AbstractValue:
  __slots__: list[str] = []

  def at_least_vspace(self):
    raise NotImplementedError("must override")

  def __repr__(self):
    try:
      kv_pairs = (f'{k}={v}' for k, v in self.__dict__.items())
      return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
    except AttributeError:
      return self.__class__.__name__

  def strip_weak_type(self) -> AbstractValue:
    return self

  def join(self, other):
    raise NotImplementedError("must override")

  def update(self, **kwargs):
    raise NotImplementedError("must override")

  def str_short(self, short_dtypes=False):
    return str(self)


# For type signatures involving dynamic shapes, we use lists of abstract values
# which may contain (reverse) de Bruijn indices in their shapes.
class DBIdx(NamedTuple):
  val: int

@dataclass(frozen=True)
class InDBIdx:
  val: int

@dataclass(frozen=True)
class OutDBIdx:
  val: int

# For annotating input types of callables (i.e. linear_util.WrappedFuns), we use
# a sequence of pairs where the first element of each pair is an AbstractValue
# (possibly containing DBIdx instances in its shape) and the second is a boolean
# indicating whether that argument is explicit (i.e. passed to the callable).
InputType = tuple[tuple[AbstractValue, bool], ...]  # DBIdx in shapes

# For annotating jaxpr output types, we use a sequence of pairs where the first
# element of each pair is an AbstractValue (possibly containing InDBIdx and/or
# OutDBIdx instances in its shape) and the second is a boolean indicating
# whether that argument is explicit (i.e. returned by the callable).
OutputType = tuple[tuple[AbstractValue, bool], ...]  # InDBIdx / OutDBIdx shapes


def _jaxpr_type_to_callable_annotation(jaxpr: Jaxpr) -> InputType:
  idxs = {v: DBIdx(i) for i, v in enumerate((*jaxpr.constvars, *jaxpr.invars))}
  out = [(v.aval.update(shape=tuple(idxs.get(d, d) for d in v.aval.shape))  # type: ignore
          if type(v.aval) is DShapedArray else v.aval, True)
         for v in jaxpr.invars]
  return tuple(out)

class Bot(AbstractValue): pass
bot = Bot()


def lattice_join(x: AbstractValue | None,
                 y: AbstractValue | None) -> AbstractValue:
  if x is None:
    assert y is not None
    return y
  elif y is None:
    return x
  elif isinstance(x, type(y)):
    return y.join(x)
  elif isinstance(y, type(x)):
    return x.join(y)
  elif isinstance(x, DShapedArray) and isinstance(y, ShapedArray):
    # TODO(mattjj): remove this special case after dynamic shapes are integrated
    return x.join(y)
  else:
    raise TypeError(x, y)

# For use in typing annotations to denote either a Tracer or a `valid_jaxtype`.
Value = Any

def valid_jaxtype(x) -> bool:
  try:
    concrete_aval(x)
  except TypeError:
    return False
  else:
    return True

def check_valid_jaxtype(x):
  if not valid_jaxtype(x):
    raise TypeError(
      f"Value {x!r} of type {type(x)} is not a valid JAX type")


def concrete_aval(x):
  for typ in type(x).__mro__:
    handler = pytype_aval_mappings.get(typ)
    if handler: return handler(x)
  if hasattr(x, '__jax_array__'):
    return concrete_aval(x.__jax_array__())
  raise TypeError(f"Value {x!r} with type {type(x)} is not a valid JAX "
                   "type")


def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  else:
    return concrete_aval(x)


def concretization_function_error(fun, suggest_astype=False):
  fname = getattr(fun, "__name__", fun)
  fname_context = f"The problem arose with the `{fname}` function. "
  if suggest_astype:
    fname_context += ("If trying to convert the data type of a value, "
                      f"try using `x.astype({fun.__name__})` "
                      f"or `jnp.array(x, {fun.__name__})` instead.")
  if fun is bool:
    def error(self, arg):
      raise TracerBoolConversionError(arg)
  elif fun in (hex, oct, operator.index):
    def error(self, arg):
      raise TracerIntegerConversionError(arg)
  else:
    def error(self, arg):
      raise ConcretizationTypeError(arg, fname_context)
  return error

def concrete_or_error(force: Any, val: Any, context=""):
  """Like force(val), but gives the context in the error message."""
  if force is None:
    force = lambda x: x
  if isinstance(val, Tracer):
    if isinstance(val.aval, ConcreteArray):
      return force(val.aval.val)
    else:
      raise ConcretizationTypeError(val, context)
  else:
    return force(val)

def concrete_dim_or_error(val: Any, context=""):
  """Like concrete_or_error(operator.index), allowing symbolic dimensions."""
  if is_symbolic_dim(val):
    return val
  else:
    return concrete_or_error(operator.index, val, context=context)

### Extended dtypes
#
# Extended dtypes are JAX-specific dtypes that allow us to represent logical
# arrays of element types that do not have an obvious direct correspondence
# to ("physical") arrays of basic types in a compiler. In particular, their
# element types differ from those of XLA and NumPy (e.g. int32). These dtypes
# are only known to JAX. Their implementation is determined by:
# a) an object representing the extended dtype, accessible via the `dtype`
#    attribute on corresponding JAX arrays and, internally, on avals such
#    as ShapedArrays that correspond to such JAX arrays;
# b) a set of rules, available via a private attribute on the extended dtype
#    object in (a).
# The rules in (b) tell JAX internals how to ground out the element
# type for interaction with the compiler and runtime, e.g. when lowering
# to the compiler's language.

@overload
def physical_aval(aval: ShapedArray) -> ShapedArray: ...
@overload
def physical_aval(aval: DShapedArray) -> DShapedArray: ...
@overload                       # TODO(frostig): remove this case
def physical_aval(aval: AbstractValue) -> AbstractValue: ...

def physical_aval(aval):
  aval_dtype = getattr(aval, 'dtype', None)
  if aval_dtype and isinstance(aval_dtype, dtypes.ExtendedDType):
    ctor = type(aval)
    aval_shape = getattr(aval, 'shape', None)
    assert aval_shape is not None, (ctor, aval)
    elt_aval = aval_dtype._rules.physical_element_aval(aval_dtype)
    assert type(elt_aval) is ShapedArray
    return ctor((*aval_shape, *elt_aval.shape), elt_aval.dtype)  # pytype: disable=wrong-arg-count
  else:
    return aval

def _short_dtype_name(dtype) -> str:
  if isinstance(dtype, dtypes.ExtendedDType):
    return str(dtype)
  else:
    return (dtype.name.replace('float', 'f').replace('uint'   , 'u')
                      .replace('int'  , 'i').replace('complex', 'c'))

def _dtype_object(dtype):
  return dtype if isinstance(dtype, dtypes.ExtendedDType) else np.dtype(dtype)

class UnshapedArray(AbstractValue):
  __slots__ = ['dtype', 'weak_type']
  array_abstraction_level = 4

  def __init__(self, dtype, weak_type=False):
    self.dtype = _dtype_object(dtype)
    self.weak_type = weak_type

  def update(self, dtype=None, weak_type=None):
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    return UnshapedArray(dtype, weak_type)

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype and
            self.weak_type == other.weak_type)

  def __ne__(self, other):
    return not self == other

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `np.zeros(3).dtype is np.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.dtype, self.weak_type))

  def __repr__(self):
    return '{}({}{})'.format(self.__class__.__name__, self.str_short(),
                             ", weak_type=True" if self.weak_type else "")

  _bool    = concretization_function_error(bool)
  _int     = concretization_function_error(int, True)
  _float   = concretization_function_error(float, True)
  _complex = concretization_function_error(complex, True)
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)
  _index   = concretization_function_error(operator.index)

  def at_least_vspace(self) -> AbstractValue:
    return UnshapedArray(primal_dtype_to_tangent_dtype(self.dtype),
                         self.weak_type)

  def join(self, other):
    if self.dtype == other.dtype:
      if self.weak_type == other.weak_type:
        return self
      else:
        return UnshapedArray(self.dtype, weak_type=False)
    else:
      raise TypeError(self, other)

  def str_short(self, short_dtypes=False) -> str:
    return _short_dtype_name(self.dtype) if short_dtypes else self.dtype.name

  def strip_weak_type(self):
    """Returns a copy of the aval with weak_type=False."""
    return self.update(weak_type=False)

  @property
  def shape(self):
    msg = ("UnshapedArray has no shape. Please open an issue at "
           "https://github.com/google/jax/issues because it's unexpected for "
           "UnshapedArray instances to ever be produced.")
    raise TypeError(msg)

def _canonicalize_dimension(dim: DimSize) -> DimSize:
  # Dimensions are most commonly integral (by far), so we check that first.
  try:
    return operator.index(dim)
  except TypeError as e:
    type_error = e
  if isinstance(dim, Tracer) and config.dynamic_shapes.value:
    if not (dim.ndim == 0 and (dtypes.issubdtype(dim.dtype, np.integer)
                               or isinstance(dim.dtype, bint))):
      raise TypeError(f"Dimensions must be integer scalars; got {dim.ndim=} {dim.dtype=}")
    return dim
  elif (config.dynamic_shapes.value and isinstance(dim, DArray) and
        type(dim._aval.dtype) is bint and not dim._aval.shape):
    return dim
  elif is_dim(dim):
    return dim
  else:
    raise type_error

def canonicalize_shape(shape: Shape, context: str="") -> tuple[Any, ...]:
  """Canonicalizes and checks for errors in a user-provided shape value.

  Args:
    shape: a Python value that represents a shape.

  Returns:
    A tuple of canonical dimension values.
  """
  if isinstance(shape, int):
    shape = shape,
  try:
    return tuple(unsafe_map(_canonicalize_dimension, shape))
  except TypeError:
    pass
  raise _invalid_shape_error(shape, context)

def canonicalize_dim(d: DimSize, context: str="") -> DimSize:
  """Canonicalizes and checks for errors in a user-provided shape dimension value.

  Args:
    d: a Python value that represents a dimension.

  Returns:
    A canonical dimension value.
  """
  return canonicalize_shape((d,), context)[0]

def _invalid_shape_error(shape: Shape, context: str=""):
  if config.dynamic_shapes.value:
    msg = ("Shapes must be 1D sequences of integer scalars, "
           f"got {shape}")
  else:
    msg = ("Shapes must be 1D sequences of concrete values of integer type, "
           f"got {shape}.")
  if context:
    msg += f" {context}."
  if not config.dynamic_shapes.value and any(
         isinstance(x, Tracer) and isinstance(get_aval(x), ShapedArray)
         and not isinstance(get_aval(x), ConcreteArray) for x in shape):
    msg += ("\nIf using `jit`, try using `static_argnums` or applying `jit` to "
            "smaller subfunctions.")
    for x in shape:
      if isinstance(x, Tracer) and hasattr(x, "_origin_msg"):
        msg += x._origin_msg()

  return TypeError(msg)

class ShapedArray(UnshapedArray):
  __slots__ = ['shape']
  array_abstraction_level = 2
  named_shape = {}  # type: ignore

  def __init__(self, shape, dtype, weak_type=False, named_shape=None):
    del named_shape  # unused, vestigial
    self.shape = canonicalize_shape(shape)
    self.dtype = _dtype_object(dtype)
    self.weak_type = weak_type

  def update(self, shape=None, dtype=None, weak_type=None, named_shape=None):
    del named_shape  # unused, vestigial
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    return ShapedArray(shape, dtype, weak_type)

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self:
                  0 if any(type(d) is int and d == 0 for d in self.shape)
                  else math.prod(self.shape))

  broadcast: ClassVar[aval_method | None] = None
  transpose: ClassVar[aval_method | None] = None
  reshape: ClassVar[aval_method | None] = None
  _iter: ClassVar[staticmethod | None] = None

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type)

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `np.zeros(3).dtype is np.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.shape, self.dtype, self.weak_type))

  def at_least_vspace(self):
    return ShapedArray(self.shape, primal_dtype_to_tangent_dtype(self.dtype),
                       self.weak_type)

  def join(self, other):
    if definitely_equal_shape(self.shape, other.shape) and self.dtype == other.dtype:
      weak_type = self.weak_type and other.weak_type
      return self.update(weak_type=weak_type)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def str_short(self, short_dtypes=False):
    dt_str =  _short_dtype_name(self.dtype) if short_dtypes else self.dtype.name
    dt_str = dt_str.replace('void', 'float0')
    shapestr = ','.join(map(str, self.shape))
    return f'{dt_str}[{shapestr}]'

  def _len(self, ignored_tracer):
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error


def _forward_to_value(self, fun, ignored_tracer, *args):
  return fun(self.val, *args)


class ConcreteArray(ShapedArray):
  __slots__ = ['val']
  array_abstraction_level = 0

  def __init__(self, dtype, val, weak_type=None):
    super().__init__(
        np.shape(val), dtype,
        weak_type=dtypes.is_weakly_typed(val) if weak_type is None else weak_type)
    dtypes.check_valid_dtype(self.dtype)
    # Note: canonicalized self.dtype doesn't necessarily match self.val
    assert self.dtype == dtypes.canonicalize_dtype(np.result_type(val)), (val, dtype)
    self.val = val

  def update(self, dtype=None, val=None, weak_type=None):
    dtype = self.dtype if dtype is None else dtype
    val = self.val if val is None else val
    weak_type = self.weak_type if weak_type is None else weak_type
    return ConcreteArray(dtype, val, weak_type)

  def __eq__(self, other):
    if (type(self) is type(other) and self.dtype == other.dtype
        and self.shape == other.shape and self.weak_type == other.weak_type):
      with eval_context():  # in case self.val is an Array
        return (self.val == other.val).all()
    else:
      return False

  def __hash__(self):
    return id(self.val)

  def join(self, other) -> AbstractValue:
    if self == other:
      return self
    elif self.shape == other.shape and self.dtype == other.dtype:
      weak_type = self.weak_type and other.weak_type
      return ShapedArray(self.shape, self.dtype, weak_type=weak_type)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype, weak_type=self.weak_type and other.weak_type)
    else:
      raise TypeError(self, other)

  def str_short(self, short_dtypes=False) -> str:
    dt_str =  _short_dtype_name(self.dtype) if short_dtypes else self.dtype.name
    return f'{self.val}, dtype={dt_str}'

  _bool    = partialmethod(_forward_to_value, bool)
  _int     = partialmethod(_forward_to_value, int)
  _hex     = partialmethod(_forward_to_value, hex)
  _oct     = partialmethod(_forward_to_value, oct)
  _index   = partialmethod(_forward_to_value, operator.index)

  _float   = concretization_function_error(float, True)
  _complex = concretization_function_error(complex, True)

def primal_dtype_to_tangent_dtype(primal_dtype):
  if isinstance(primal_dtype, dtypes.ExtendedDType):
    return primal_dtype._rules.tangent_dtype(primal_dtype)
  elif not dtypes.issubdtype(primal_dtype, np.inexact):
    return dtypes.float0
  else:
    return primal_dtype


# Dynamic shape stuff below here! We keep the abstract values distinct just so
# as not to interfere with any static shape machinery.

# We have a convention of reusing AbsractValues as types, even though we could
# make a distinction and use abstract values during tracing only. This reuse
# becomes a bit more extreme with DShapedArrays. A DShapedArray's shape
# attribute is a tuple which can contain several different types: int, DArray
# (scalar and with dtype of bint type), Tracer (while tracing), Var (when used
# as jaxpr type annotations), or DBIdx/InDBIdx/OutDBIdx (when used in InputType
# or OutputType). We could reduce this polymorphism if it seems cleaner, though
# it's kind of convenient!
class DShapedArray(UnshapedArray):
  __slots__ = ['shape']
  shape: tuple[AxisSize, ...]  # noqa: F821
  array_abstraction_level: int = 3

  def __init__(self, shape, dtype, weak_type=False):
    self.shape = shape
    self.dtype = dtype
    self.weak_type = weak_type

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self:
                  0 if any(type(d) is int and d == 0 for d in self.shape)
                  else math.prod(self.shape))

  def str_short(self, short_dtypes=False) -> str:
    del short_dtypes  # ignored
    shape = f'{",".join(str(d) for d in self.shape)}' if self.shape else ''
    dtype = _short_dtype_name(self.dtype)
    return f'{dtype}[{shape}]'
  __str__ = __repr__ = str_short

  def update(self, shape=None, dtype=None, weak_type=None):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    return DShapedArray(shape, dtype, weak_type)

  def _len(self, tracer):
    return self.shape[0]

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type)

  def __hash__(self):
    return hash((self.shape, self.dtype, self.weak_type))

  def join(self, other):
    if (definitely_equal_shape(self.shape, other.shape) and
        self.dtype == other.dtype):
      weak_type = self.weak_type and other.weak_type
      return self.update(weak_type=weak_type)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def at_least_vspace(self):
    return DShapedArray(self.shape, primal_dtype_to_tangent_dtype(self.dtype),
                        self.weak_type)

class DConcreteArray(DShapedArray):
  __slots__ = ['val']
  array_abstraction_level = 1
  def __init__(self, shape, dtype, weak_type, val):
    super().__init__(shape, dtype, weak_type)
    self.val = val


pytype_aval_mappings: dict[type, Callable[[Any], AbstractValue]] = {}


class DArray:
  _aval: DShapedArray
  _data: Any  # standard array type
  def __init__(self, aval, data):
    pad_shape = tuple(d.dtype.bound if type(d) is DArray and
                      type(d.dtype) is bint else d for d in aval.shape)
    assert data.shape == pad_shape
    self._aval = aval
    self._data = data
  shape = property(lambda self: self._aval.shape)
  dtype = property(lambda self: self._aval.dtype)
  aval = property(lambda self: self._aval)
  def __repr__(self) -> str:
    if not self.shape and type(self.dtype) is bint:
      # special-case scalar bints
      return f'{int(self._data)}{{≤{self.dtype.bound}}}'

    dtypestr = _short_dtype_name(self._aval.dtype)
    shapestr = ','.join(map(str, self.shape))
    slices = tuple(slice(int(d._data)) if type(d) is DArray and
                   type(d.dtype) is bint else slice(None) for d in self.shape)
    data = self._data[slices]
    return f'{dtypestr}[{shapestr}] with value: {data}'
  def __hash__(self) -> int:
    if not self.shape:
      return hash((self._aval, int(self._data)))
    raise TypeError("unhashable type: DArray")
  def __eq__(self, other):
    if isinstance(other, DArray) and self._aval == other._aval:
      return self._data == other._data
    return False
  def __len__(self):
    return self.shape[0]

pytype_aval_mappings[DArray] = \
    lambda x: DConcreteArray(x._aval.shape, x._aval.dtype, x._aval.weak_type,
                             x._data)

@dataclass(frozen=True)
class bint(dtypes.ExtendedDType):
  bound: int

  @property
  def type(self) -> type:
    return dtypes.extended

  @property
  def name(self) -> str:
    return f'bint{{≤{self.bound}}}'

  def __str__(self) -> str:
    return self.name

AxisSize = Union[int, DArray, Tracer, Var, DBIdx, InDBIdx, OutDBIdx]


class MutableArray:
  _aval: ShapedArray
  _buf: Array
  def __init__(self, aval, buf):
    self._aval = aval
    self._buf = buf
  aval = property(lambda self: self._aval)
  shape = property(lambda self: self._aval.shape)
  dtype = property(lambda self: self._aval.dtype)
  def __getitem__(self, idx): return get_aval(self)._getitem(self, idx)
  def __setitem__(self, idx, x): return get_aval(self)._setitem(self, idx, x)
  def __repr__(self) -> str: return 'Mutable' + repr(self[...])
pytype_aval_mappings[MutableArray] = lambda x: x._aval

def mutable_array(init_val):
  return mutable_array_p.bind(init_val)
mutable_array_p = Primitive('mutable_array')

class InternalMutableArrayEffect(effects.Effect):
  pass
internal_mutable_array_effect = InternalMutableArrayEffect()
effects.control_flow_allowed_effects.add_type(InternalMutableArrayEffect)

@mutable_array_p.def_effectful_abstract_eval
def mutable_array_abstract_eval(init_aval):
  from jax._src.state.types import AbstractRef  # pytype: disable=import-error
  return AbstractRef(init_aval), {internal_mutable_array_effect}

@mutable_array_p.def_impl
def _mutable_array_impl(init_val):
  from jax._src.state.types import AbstractRef  # pytype: disable=import-error
  aval = raise_to_shaped(get_aval(init_val))
  return MutableArray(AbstractRef(aval), init_val)


class AbstractToken(AbstractValue):
  def join(self, other):
    if isinstance(other, AbstractToken):
      return self
    else:
      assert False, f"Cannot join {self} with {other}"
  def str_short(self, short_dtypes=False): return 'Tok'
  def at_least_vspace(self): return self
abstract_token: AbstractToken = AbstractToken()

# Singleton shaped array used by all abstract tokens when shape/dtype is needed.
token_shaped_array: ShapedArray = ShapedArray((0,), np.dtype(np.bool_))

# Concrete token object
class Token:
  # The underlying data wrapped by the token, could be used to threaded in and
  # out of computations to build up data dependency.
  _buf: Array
  def __init__(self, buf):
    self._buf = buf
  def block_until_ready(self):
    self._buf.block_until_ready()
pytype_aval_mappings[Token] = lambda _: abstract_token


def raise_to_shaped(aval: AbstractValue, weak_type=None):
  aval_type = type(aval)
  if aval_type is ShapedArray and weak_type is None:
    return aval
  if weak_type is None:
    weak_type = getattr(aval, 'weak_type', False)
  for typ in aval_type.__mro__:
    handler = raise_to_shaped_mappings.get(typ)
    if handler: return handler(aval, weak_type)
  raise TypeError(type(aval))

raise_to_shaped_mappings: dict[type, Callable] = {
    AbstractToken: lambda aval, _: aval,
    Bot: lambda aval, _: aval,
    UnshapedArray: lambda aval, _: aval,
    ShapedArray: lambda aval, weak_type: ShapedArray(
        aval.shape, aval.dtype, weak_type),
    DConcreteArray: lambda aval, weak_type: DShapedArray(
        aval.shape, aval.dtype, weak_type
    ),
}

### Operations on shapes and dimension sizes.

class InconclusiveDimensionOperation(Exception):
  """Raised when we cannot conclusively compute with symbolic dimensions."""
  pass

def is_symbolic_dim(v: Any) -> bool:
  """Checks if a value is a symbolic dimension used for shape polymorphism.

  This should be used very rarely, because symbolic dimensions overload all
  operators, and should just work.
  """
  return hasattr(v, "dimension_as_value")

def is_constant_dim(d: DimSize) -> bool:
  # Whether the dimension is a static integer constant.
  try:
    operator.index(d)
    return True
  except:
    return False

def is_dim(v: Any) -> bool:
  return is_symbolic_dim(v) or is_constant_dim(v)

def is_constant_shape(s: Shape) -> bool:
  # Whether the shape is a static constant.
  return all(is_constant_dim(d) for d in s)

def definitely_equal_one_of_dim(d1: DimSize, dlist: Sequence[DimSize]) -> bool:
  return any(definitely_equal(d1, d) for d in dlist)

def definitely_equal_shape(s1: Shape, s2: Shape) -> bool:
  """Check that two shapes are guaranteed to be element-wise equal.

  In presence of dynamic shapes may return False even when the shapes may
  be equal at runtime.
  """
  return (len(s1) == len(s2) and
          all(unsafe_map(definitely_equal, s1, s2)))

def divide_shape_sizes(s1: Shape, s2: Shape) -> DimSize:
  """Returns an integer "i" s.t., i * size(s2) == size(s1).
  Raises InconclusiveDimensionOperation if there is no such integer."""
  sz1 = math.prod(s1)
  sz2 = math.prod(s2)
  if definitely_equal(sz1, sz2):  # Takes care of sz1 and sz2 being 0
    return 1
  q, r = divmod(sz1, sz2)
  if isinstance(r, Tracer) or r != 0:
    raise InconclusiveDimensionOperation(
        f"Cannot divide evenly the sizes of shapes {tuple(s1)} and {tuple(s2)}. "
        f"The remainder {r} should be 0.")
  return q

def cancel_divide_tracers(num, denom):
  partition = lambda l: partition_list([isinstance(d, Tracer) for d in l], l)
  num, num_tracers = partition(num)
  denom, denom_tracers = partition(denom)
  if num_tracers or denom_tracers:
    factor = _cancel_divide(num_tracers, denom_tracers)
    if factor is not None:
      size1 = math.prod(num)
      size2 = math.prod(denom)
      if size1 == size2 or size2 != 0:
        return factor * (size1 // size2 if size1 != size2 else 1)

def _cancel_divide(num, denom):
  num = list(num)
  for a in denom:
    i = next((i for i, b in enumerate(num) if definitely_equal(a, b)), None)
    if i is None:
      break  # couldn't cancel
    del num[i]
  else:
    return math.prod(num)

def is_empty_shape(s: Shape) -> bool:
  return any(definitely_equal(d, 0) for d in s)

def dilate_dim(d: DimSize, dilation: DimSize) -> DimSize:
  """max(0, 1 + dilation * (d - 1)).

  Assumes dilation >= 1.
  """
  if definitely_equal(dilation, 1):  # fast path
    return d
  return max_dim(1 + dilation * (d - 1), 0)

def stride_dim(d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
  """max(0, (d - window_size) // window_stride + 1)

  If d < window_size, returns 0.
  We assume window_size >= 1 and window_stride >= 1.
  """
  # If d < window_size then (d - window_size) // window_stride < 0
  return max_dim((d - window_size) // window_stride + 1, 0)

# TODO(necula): Deprecated Jan 2024, to be removed.
def non_negative_dim(d: DimSize) -> DimSize:
  """max(d, 0)."""
  return max_dim(d, 0)

def min_dim(d1: DimSize, d2: DimSize) -> DimSize:
  """Like min(d1, d2) but for both constant and symbolic dimensions."""
  d1_is_constant = is_constant_dim(d1)
  if d1_is_constant and is_constant_dim(d2):
    return min(d1, d2)
  if d1_is_constant:
    return d2.rmin(d1)  # type: ignore[union-attr]
  else:
    return d1.min(d2)  # type: ignore[union-attr]

def max_dim(d1: DimSize, d2: DimSize) -> DimSize:
  """Like max(d1, d2) but for both constant and symbolic dimensions."""
  d1_is_constant = is_constant_dim(d1)
  if d1_is_constant and is_constant_dim(d2):
      return max(d1, d2)
  if d1_is_constant:
    return d2.rmax(d1)  # type: ignore[union-attr]
  else:
    return d1.max(d2)  # type: ignore[union-attr]

def dimension_as_value(d: DimSize):
  """Turns a dimension size into a JAX array.
     This is the identity function for constant dimensions.

     Has the same abstract value as Python constants.
     """
  if isinstance(d, (int, Tracer, np.int32, np.int64)): return d
  # For shape_poly._DimPolynomial
  if hasattr(d, "dimension_as_value"): return d.dimension_as_value()
  return operator.index(d)

class SomeTracer:
  __slots__ = ()
  def __repr__(self): return "[dynamic]"

def replace_tracer_for_error_message(obj):
  # TODO(mattjj): Many ideas for improving this.  Crawl the stack and see if
  # there are user variables whose value is == to this object?  Or search
  # parameters of functions being transformed, at least?  Or at least assign
  # short unique ids to them?
  if isinstance(obj, Tracer):
    return SomeTracer()
  else:
    return obj

def evaluate_shape(shape: Shape, dim_vars: Sequence[str],
                   *dim_values: Array) -> Sequence[Array]:
  """Evaluates a shape possibly containing non-constants.

  Args:
    shape: the shape to evaluate.
    dim_vars: the dimension variables names that may appear in `shape`.
    dim_values: the dimension values corresponding to `dim_vars`.

  Returns:
     a tuple of JAX values corresponding to `shape`, of type
     `dim_value_dtype`.
  """
  env = dict(zip(dim_vars, dim_values))
  def eval_one_dim(d: DimSize):
    try:
      return operator.index(d)
    except:
      # Is a _DimExpr
      return d._evaluate(env)  # type: ignore
  return tuple(eval_one_dim(d) for d in shape)

def dim_value_dtype():
  """The dtype to be used for dimension values."""
  return dtypes.canonicalize_dtype(np.int64)

def dim_constant(ct: int):
  dtype = dim_value_dtype()
  assert dtype in (np.int32, np.int64)
  if dtype == np.int32:
    return np.int32(ct)
  elif dtype == np.int64:
    return np.int64(ct)

def dim_value_aval() -> AbstractValue:
  return ShapedArray((), dim_value_dtype(), weak_type=True)

# ------------------- Call -------------------

class CallPrimitive(Primitive):
  multiple_results = True
  call_primitive = True

  def bind(self, fun, *args, **params):
    call_bind_continuation, top_trace, fun_, tracers, params = (
        call_bind_with_continuation(self, fun, *args, **params))
    outs = top_trace.process_call(self, fun_, tracers, params)
    return call_bind_continuation(outs)

  def get_bind_params(self, params):
    new_params = dict(params)
    jaxpr = new_params.pop('call_jaxpr')
    subfun = lu.hashable_partial(lu.wrap_init(eval_jaxpr), jaxpr, ())
    if config.dynamic_shapes.value:
      subfun = lu.annotate(subfun, _jaxpr_type_to_callable_annotation(jaxpr))
    return [subfun], new_params

def call_bind_with_continuation(primitive: CallPrimitive, fun, *args, **params):
  top_trace = find_top_trace(args)
  fun_, env_trace_todo = process_env_traces_call(
      fun, primitive, top_trace.level, tuple(params.items()))
  tracers = map(top_trace.full_raise, args)
  fun_ = lu.annotate(fun_, fun.in_type)

  def call_bind_continuation(outs):
    return map(full_lower, apply_todos(env_trace_todo(), outs))
  return call_bind_continuation, top_trace, fun_, tracers, params

@lu.transformation_with_aux
def process_env_traces_call(primitive: CallPrimitive, level: int,
                            params_tuple: tuple, *args):
  outs = yield args, {}
  params = dict(params_tuple)
  todo = []
  while True:
    tracers = [x for x in outs if isinstance(x, Tracer) and x._trace.level > level]
    if not tracers:
      break
    ans = max(tracers, key=_get_trace_level)
    trace = ans._trace.main.with_cur_sublevel()
    outs = map(trace.full_raise, outs)
    outs, cur_todo = trace.post_process_call(primitive, outs, params)
    todo.append(cur_todo)
  yield outs, tuple(todo)  # Ensure the aux output is immutable

def apply_todos(todos, outs):
  todos_list = list(todos)
  while todos_list:
    outs = map(full_lower, todos_list.pop()(outs))
  return outs


def call_impl(f: lu.WrappedFun, *args, **params):
  del params  # params parameterize the call primitive, not the function
  with new_sublevel():
    return f.call_wrapped(*args)

call_p: CallPrimitive = CallPrimitive('call')
call = call_p.bind
call_p.def_impl(call_impl)


class ClosedCallPrimitive(CallPrimitive):
  def get_bind_params(self, params):
    new_params = dict(params)
    jaxpr = new_params.pop('call_jaxpr')
    subfun = lu.wrap_init(partial(eval_jaxpr, jaxpr.jaxpr, jaxpr.consts))
    return [subfun], new_params

closed_call_p: ClosedCallPrimitive = ClosedCallPrimitive('closed_call')
closed_call_p.def_impl(call_impl)
closed_call_p.def_effectful_abstract_eval(
    lambda *_, call_jaxpr: (call_jaxpr.out_avals, call_jaxpr.effects))


outfeed_primitives: set[Primitive] = set()
def jaxpr_uses_outfeed(jaxpr: Jaxpr) -> bool:
  """Finds if there are outfeed primitives anywhere inside a Jaxpr."""
  return any(primitive_uses_outfeed(eqn.primitive, eqn.params)
             for eqn in jaxpr.eqns)

def _param_uses_outfeed(param):
  if type(param) is Jaxpr:
    if jaxpr_uses_outfeed(param):
      return True
  elif type(param) is ClosedJaxpr:
    if jaxpr_uses_outfeed(param.jaxpr):
      return True
  return False

def primitive_uses_outfeed(prim: Primitive, params: dict) -> bool:
  if prim in outfeed_primitives:
    return True
  for param in params.values():
    if isinstance(param, tuple):
      if any(unsafe_map(_param_uses_outfeed, param)):
        return True
    elif _param_uses_outfeed(param):
      return True
  return False

# ------------------- Map -------------------

class MapPrimitive(Primitive):
  multiple_results = True
  map_primitive = True

  def bind(self, fun, *args, **params):
    assert len(params['in_axes']) == len(args)
    return map_bind(self, fun, *args, **params)

  def process(self, trace, fun, tracers, params):
    return trace.process_map(self, fun, tracers, params)

  def post_process(self, trace, out_tracers, params):
    return trace.post_process_map(self, out_tracers, params)

  def get_bind_params(self, params):
    new_params = dict(params)
    jaxpr = new_params.pop('call_jaxpr')
    subfun = lu.hashable_partial(lu.wrap_init(eval_jaxpr), jaxpr, ())
    axes = new_params.pop('out_axes')
    new_params['out_axes_thunk'] = HashableFunction(lambda: axes, closure=axes)
    return [subfun], new_params


def map_bind_with_continuation(primitive: MapPrimitive, fun, *args,
                               out_axes_thunk, **params):
  # The new thunk depends deterministically on the old thunk and the wrapped
  # function. Any caching already has to include the wrapped function as part
  # of the key, so we only use the previous thunk for equality checks.
  @as_hashable_function(closure=out_axes_thunk)
  def new_out_axes_thunk():
    out_axes = out_axes_thunk()
    _, out_axes_transforms = todo_and_xforms()
    for t in out_axes_transforms:
      out_axes = t(out_axes)
    return out_axes
  params = dict(params, out_axes_thunk=new_out_axes_thunk)
  top_trace = find_top_trace(args)
  fun, todo_and_xforms = process_env_traces_map(
      fun, primitive, top_trace and top_trace.level, tuple(params.items()))
  tracers = map(top_trace.full_raise, args)

  def map_bind_continuation(outs):
    env_trace_todo, _ = todo_and_xforms()
    return map(full_lower, apply_todos(env_trace_todo, outs))

  return map_bind_continuation, top_trace, fun, tracers, params


def map_bind(primitive: MapPrimitive, fun, *args, **params):
  map_bind_continuation, top_trace, fun, tracers, params = (
      map_bind_with_continuation(primitive, fun, *args, **params))
  return map_bind_continuation(
      primitive.process(top_trace, fun, tracers, params))

@lu.transformation_with_aux
def process_env_traces_map(primitive: MapPrimitive, level: int,
                           params_tuple: tuple, *args):
  outs = yield args, {}
  params = dict(params_tuple)
  todo = []
  out_axes_transforms = []
  while True:
    tracers = [x for x in outs if isinstance(x, Tracer)
               and (level is None or x._trace.level > level)]
    if not tracers:
      break
    ans = max(tracers, key=_get_trace_level)
    trace = ans._trace.main.with_cur_sublevel()
    outs = map(trace.full_raise, outs)
    outs, (cur_todo, cur_xform) = primitive.post_process(trace, outs, params)
    todo.append(cur_todo)
    out_axes_transforms.append(cur_xform)
  yield outs, (tuple(todo), tuple(out_axes_transforms))


def mapped_aval(size: AxisSize, axis: int | None,
                aval: AbstractValue) -> AbstractValue:
  handler, _ = aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis, aval)
  else:
    raise TypeError(f"no mapping handler for {aval} of type {type(aval)}")

def unmapped_aval(size: AxisSize, axis_name, axis: int | None,
                  aval: AbstractValue) -> AbstractValue:
  _, handler = aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis_name, axis, aval)
  else:
    raise TypeError(f"no unmapping handler for {aval} of type {type(aval)}")


def _map_shaped_array(
    size: int, axis: int | None, aval: ShapedArray) -> ShapedArray:
  assert axis is None or aval.shape[axis] == size
  # TODO: Extend the named shape
  if axis is None: return aval
  return ShapedArray(tuple_delete(aval.shape, axis), aval.dtype,
                     weak_type=aval.weak_type)

def _unmap_shaped_array(
    size: int, axis_name: AxisName, axis: int | None, aval: ShapedArray
  ) -> ShapedArray:
  if axis is None: return aval
  elif type(axis) is int:
    return ShapedArray(tuple_insert(aval.shape, axis, size), aval.dtype,
                       weak_type=aval.weak_type)
  else: raise TypeError(axis)

def _map_dshaped_array(
    size: AxisSize, axis: int | None, aval: DShapedArray) -> DShapedArray:
  if axis is None: return aval
  return DShapedArray(tuple_delete(aval.shape, axis), aval.dtype,
                      aval.weak_type)

def _unmap_dshaped_array(
    size: AxisSize, axis_name: AxisName, axis: int | None, aval: DShapedArray
  ) -> DShapedArray:
  if axis is None: return aval
  elif type(axis) is int:
    return DShapedArray(tuple_insert(aval.shape, axis, size), aval.dtype,
                        weak_type=aval.weak_type)
  else:
    raise TypeError(axis)

AvalMapHandlerPair = tuple[Callable, Callable]
aval_mapping_handlers: dict[type, AvalMapHandlerPair] = {
    DShapedArray:   (_map_dshaped_array, _unmap_dshaped_array),
    ShapedArray:   (_map_shaped_array, _unmap_shaped_array),
    ConcreteArray: (_map_shaped_array, _unmap_shaped_array),
    AbstractToken: (lambda _, __, a: a, lambda _, __, ___, a: a)
}

@contextmanager
def extend_axis_env(axis_name: AxisName, size: int, tag: Any):
  frame = AxisEnvFrame(axis_name, size, tag)
  ts = thread_local_state.trace_state
  ts.axis_env.append(frame)
  config.update_thread_local_jit_state(
      axis_env_state=tuple(f for f in ts.axis_env
                           if f.name is not no_axis_name))
  try:
    yield
  finally:
    ts.axis_env.pop()
    config.update_thread_local_jit_state(
        axis_env_state=tuple(f for f in ts.axis_env
                             if f.name is not no_axis_name))

@contextmanager
def extend_axis_env_nd(axes: Iterable[tuple[AxisName, int]], tag: Any = None):
  frames = [AxisEnvFrame(axis_name, size, tag) for axis_name, size in axes]
  ts = thread_local_state.trace_state
  ts.axis_env.extend(frames)
  config.update_thread_local_jit_state(
      axis_env_state=tuple(f for f in ts.axis_env
                           if f.name is not no_axis_name))
  try:
    yield
  finally:
    for _ in frames: ts.axis_env.pop()
    config.update_thread_local_jit_state(
        axis_env_state=tuple(f for f in ts.axis_env
                             if f.name is not no_axis_name))


@contextmanager
def stash_axis_env():
  "Promise that a function or with-suite does not depend implicitly on axis env"
  # If the promise is broken, then a NameError about an unbound axis name will
  # be raised.
  ts = thread_local_state.trace_state
  prev_axis_env, ts.axis_env = ts.axis_env, []
  config.update_thread_local_jit_state(axis_env_state=())
  try:
    yield
  finally:
    ts.axis_env = prev_axis_env
    config.update_thread_local_jit_state(
        axis_env_state=tuple(f for f in ts.axis_env
                             if f.name is not no_axis_name))


# When a mapped function is given no axis name, we generate a name object based
# on the id of the function object. Collisions aren't important because this
# name can't be used in collectives, as user code never gets a ref to this
# object. We don't want to use the function object itself because that might
# persist references to the function object.
# TODO(mattjj): revisit this unique axis name strategy
@total_ordering
class _TempAxisName:

  def __init__(self, obj):
    self.id = id(obj)

  def __repr__(self):
    return f'<axis {hex(self.id)}>'

  def __hash__(self):
    return hash(self.id)

  def __eq__(self, other):
    return type(other) is _TempAxisName and self.id == other.id

  def __lt__(self, other):
    return type(other) is _TempAxisName and self.id < other.id


def axis_frame(axis_name: AxisName, main_trace: MainTrace | None = None
               ) -> AxisEnvFrame:
  frames = thread_local_state.trace_state.axis_env
  for frame in reversed(frames):
    if (frame.name == axis_name and
        (main_trace is None or frame.main_trace is main_trace)):
      return frame
  named_axes = [frame.name for frame in reversed(frames)
                if not isinstance(frame.name, _TempAxisName)]
  raise NameError(
      f'unbound axis name: {axis_name}. The following axis names (e.g. defined '
      f'by pmap) are available to collective operations: {named_axes}')


@dataclass(frozen=True)
class NamedAxisEffect(effects.Effect):
  """A side-effect introducing a new named axis into the current scope."""

  name: AxisName


effects.control_flow_allowed_effects.add_type(NamedAxisEffect)
effects.custom_derivatives_allowed_effects.add_type(NamedAxisEffect)
effects.lowerable_effects.add_type(NamedAxisEffect)
effects.remat_allowed_effects.add_type(NamedAxisEffect)


def filter_named_axis_effects(
    effects: Effects, names: Collection[AxisName]
) -> Effects:
  return {e for e in effects
          if not isinstance(e, NamedAxisEffect) or e.name not in names}


def remove_named_axis_effects(
    jaxpr: Jaxpr, names: Collection[AxisName]
) -> Jaxpr:
  if not names or not jaxpr.effects:
    return jaxpr
  return jaxpr.replace(effects=filter_named_axis_effects(jaxpr.effects, names))


ParamDict = dict[str, Any]
AxisSubst = Callable[[AxisName], tuple[AxisName, ...]]

class NameGatheringSubst:
  def __init__(self):
    self.axis_names = set()
  def __call__(self, axis_name):
    self.axis_names.add(axis_name)
    return (axis_name,)

def used_axis_names(primitive: Primitive, params: ParamDict) -> set[AxisName]:
  subst = NameGatheringSubst()
  subst_axis_names(primitive, params, subst)
  return subst.axis_names

def subst_axis_names(primitive: Primitive, params: ParamDict, subst: AxisSubst, traverse: bool = True) -> ParamDict:
  if primitive in axis_substitution_rules:
    return axis_substitution_rules[primitive](params, subst, traverse)
  if not traverse:
    return params
  # Default implementation: substitute names in all jaxpr parameters
  if isinstance(primitive, MapPrimitive):
    def shadowed_subst(name):
      return (name,) if name == params['axis_name'] else subst(name)
  else:
    shadowed_subst = subst
  jaxpr_params = [(n, v) for n, v in params.items() if isinstance(v, (Jaxpr, ClosedJaxpr))]
  if not jaxpr_params:
    return params
  new_params = dict(params)
  for name, jaxpr in jaxpr_params:
    new_params[name] = subst_axis_names_jaxpr(jaxpr, shadowed_subst)
  return new_params

class DuplicateAxisNameError(Exception):
  def __init__(self, var):
    self.var = var
    self.eqn = None

def subst_axis_names_effects(effects: Set[Effect], subst: AxisSubst) -> Set[Effect]:
  new_effects = set[Effect]()
  for e in effects:
    if isinstance(e, NamedAxisEffect):
      new_effects.update(map(NamedAxisEffect, subst(e.name)))
    else:
      new_effects.add(e)
  return new_effects

def subst_axis_names_var(v: Var, subst: AxisSubst, var_map: dict[Var, Var]) -> Var:
  # Var identity is load-bearing, so we can't have duplicates!
  if isinstance(v, DropVar): return v
  assert v not in var_map
  var_map[v] = v
  return v

def subst_axis_names_eqn(eqn: JaxprEqn, subst: AxisSubst, var_map: dict[Var, Var]) -> JaxprEqn:
  invars: list[Atom] = [v if isinstance(v, Literal) else var_map[v] for v in eqn.invars]
  try:
    outvars = [subst_axis_names_var(v, subst, var_map) for v in eqn.outvars]
  except DuplicateAxisNameError as e:
    e.eqn = eqn
    raise
  params = subst_axis_names(eqn.primitive, eqn.params, subst)
  effects = subst_axis_names_effects(eqn.effects, subst)
  return eqn.replace(invars=invars, outvars=outvars, params=params, effects=effects)

def do_subst_axis_names_jaxpr(jaxpr: Jaxpr | ClosedJaxpr, subst: AxisSubst):
  consts = None
  if isinstance(jaxpr, ClosedJaxpr):
    consts = jaxpr.consts
    jaxpr = jaxpr.jaxpr
  var_map: dict[Var, Var] = {}
  invars = [subst_axis_names_var(v, subst, var_map) for v in jaxpr.invars]  # type: ignore[union-attr]
  constvars = [subst_axis_names_var(v, subst, var_map) for v in jaxpr.constvars]  # type: ignore[union-attr]
  eqns = [subst_axis_names_eqn(eqn, subst, var_map) for eqn in jaxpr.eqns]
  outvars: list[Atom] = [v if isinstance(v, Literal) else var_map[v] for v in jaxpr.outvars]  # type: ignore[union-attr]
  effects = subst_axis_names_effects(jaxpr.effects, subst)
  new_jaxpr = Jaxpr(constvars, invars, outvars, eqns, effects)
  if consts is not None:
    return ClosedJaxpr(new_jaxpr, consts)
  return new_jaxpr

def used_axis_names_jaxpr(jaxpr: Jaxpr | ClosedJaxpr):
  return {e.name for e in jaxpr.effects if isinstance(e, NamedAxisEffect)}

def subst_axis_names_jaxpr(jaxpr: Jaxpr | ClosedJaxpr, subst: AxisSubst):
  if isinstance(subst, NameGatheringSubst):  # This is a common case, so we optimize it!
    subst.axis_names |= used_axis_names_jaxpr(jaxpr)
    return jaxpr
  return do_subst_axis_names_jaxpr(jaxpr, subst)

def replace_jaxpr_effects(jaxpr: ClosedJaxpr, effects: Effects):
  return _replace_jaxpr_effects(jaxpr, frozenset(effects))

@weakref_lru_cache
def _replace_jaxpr_effects(jaxpr: ClosedJaxpr, effects: frozenset[Effect]):
  return jaxpr.replace(jaxpr=jaxpr.jaxpr.replace(effects=set(effects)))


axis_substitution_rules: dict[Primitive, Callable[[ParamDict, AxisSubst, bool], ParamDict]] = {}

# ------------------- AxisPrimitive -------------------
# Primitives that store axis names in params and want those axis names to
# participate in dispatch should subclass AxisPrimitive.

class AxisPrimitive(Primitive):
  def bind(self, *args, **params):
    top_trace = find_top_trace(args)
    axis_main = max((axis_frame(a).main_trace for a in used_axis_names(self, params)),
                    default=None, key=lambda t: getattr(t, 'level', -1))
    top_trace = (top_trace if not axis_main or axis_main.level < top_trace.level
                 else axis_main.with_cur_sublevel())
    return self.bind_with_trace(top_trace, args, params)


# ------------------- Jaxpr checking -------------------

def typecheck(aval: AbstractValue, x) -> bool:
  return typecompat(aval, get_aval(x))

def typecompat(aval_ref: AbstractValue, aval: AbstractValue) -> bool:
  """Determine whether `aval` conforms to `aval_ref`. Ignores weak_type."""
  try:
    return typematch(aval_ref, lattice_join(aval_ref, aval))
  except TypeError:
    return False

def typematch(aval1: AbstractValue, aval2: AbstractValue) -> bool:
  """Determine whether `aval1` and `aval2` are equivalent. Ignores weak_type."""
  if aval1 == aval2: return True
  # unequal avals may still represent the same type, because type is represented
  # by avals at the shaped level, and because weak type tags aren't considered
  # part of the type
  return (raise_to_shaped(aval1, weak_type=False) ==
          raise_to_shaped(aval2, weak_type=False))

class JaxprTypeError(TypeError): pass

custom_typechecks: dict[Primitive, Callable] = {}

def _check_closed_call(_, *in_atoms, call_jaxpr):
  in_avals = [x.aval for x in in_atoms]
  if not all(map(typecompat, call_jaxpr.in_avals, in_avals)):
    raise JaxprTypeError("Closed call in_avals mismatch")
  return call_jaxpr.out_avals, call_jaxpr.effects
custom_typechecks[closed_call_p] = _check_closed_call

def check_jaxpr(jaxpr: Jaxpr):
  """Checks well-formedness of a jaxpr.

  Specifically, check that:
  - variables that are read are bound beforehand
  - variables are typed equally throughout a jaxpr
  - variable type annotations are compatible with their binding expression

  Raises `JaxprTypeError` if `jaxpr` is determined invalid. Returns `None`
  otherwise.
  """
  @functools.cache
  def ctx_factory():
    ctx = JaxprPpContext()
    pp_settings = JaxprPpSettings()
    try: pp_jaxpr(jaxpr, ctx, pp_settings)  # side-effect on ctx, build variable names
    except: pass
    return ctx, pp_settings

  try:
    _check_jaxpr(ctx_factory, jaxpr)
  except JaxprTypeError as e:
    ctx, pp_settings = ctx_factory()
    if len(e.args) == 2:
      msg, eqnidx = e.args
      jaxpr_str = str(pp_jaxpr_eqn_range(jaxpr, eqnidx - 10, eqnidx + 10, ctx,
                                         pp_settings))
    else:
      msg, = e.args
      jaxpr_str = str(pp_jaxpr_eqn_range(jaxpr, 0, 20, ctx, pp_settings))
    msg = "\n\n".join([msg, "while checking jaxpr:", jaxpr_str])
    raise JaxprTypeError(msg) from None

  # Run key reuse checker after validating jaxpr:
  if config.debug_key_reuse.value:
    # Import here to avoid circular imports
    from jax.experimental.key_reuse._core import check_key_reuse_jaxpr  # pytype: disable=import-error
    check_key_reuse_jaxpr(jaxpr)


def _check_jaxpr(
    ctx_factory: Callable[[], tuple[JaxprPpContext, JaxprPpSettings]],
    jaxpr: Jaxpr
  ) -> None:
  # Use set of variables to types to check that variables are in scope.
  env: set[Var] = set()

  def read(x: Atom) -> Atom:
    # Check the type annotation is itself well-typed.
    check_type(ctx_factory, env, x.aval)
    if isinstance(x, Var):
      # Check the variable is in-scope and consistently typed.
      if x not in env:
        ctx, _ = ctx_factory()
        raise JaxprTypeError(f"Variable '{pp_var(x, ctx)}' not defined")
      return x
    elif isinstance(x, Literal):
      # Check that the literal matches its type annotation.
      if not typecheck(x.aval, x.val):
        ctx, _ = ctx_factory()
        raise JaxprTypeError(
            f"Literal value {x.val} does not match its type annotation "
            f"{pp_aval(x.aval, ctx)}")
      return x
    else:
      assert False, "syntactically invalid jaxpr"

  def write(v: Var, a: AbstractValue) -> None:
    assert isinstance(v, Var), "syntactically invalid jaxpr"
    # Check the type annotation of the binder is itself well-typed.
    check_type(ctx_factory, env, v.aval)
    # Check that the variable is not already bound.
    if v in env:
      ctx, _ = ctx_factory()
      raise JaxprTypeError(f"Variable '{pp_var(v, ctx)}' already bound")
    # Check that the computed type is consistent with the binder annotation.
    if not typematch(v.aval, a):
      ctx, _ = ctx_factory()
      raise JaxprTypeError(
          f"Value for variable '{pp_var(v, ctx)}' inconsistently typed "
          f"as {pp_aval(a, ctx)} for let-binder of type {pp_aval(v.aval, ctx)}")
    # If the variable is not a DropVar, add it to the environment.
    if not isinstance(v, DropVar):
      env.add(v)

  # Check type annotations on lambda binders.
  for v in it.chain(jaxpr.constvars, jaxpr.invars):
    check_type(ctx_factory, env, v.aval)
    write(v, v.aval)

  # Check each eqn.
  sentinel = object()
  in_idx = {v: i for i, v in enumerate(it.chain(jaxpr.constvars, jaxpr.invars))}
  for eqn_idx, eqn in enumerate(jaxpr.eqns):
    prim = eqn.primitive
    try:
      in_atoms = map(read, eqn.invars)
      in_avals = [x.aval for x in in_atoms]  # use in_atoms for dyn shapes

      # Compute the type of the primitive application.
      if prim in custom_typechecks:
        out_type, eqn_effects = custom_typechecks[prim](
          ctx_factory, *in_atoms, **eqn.params)
      elif prim.call_primitive:
        out_type, eqn_effects = _check_call(ctx_factory, prim, in_atoms,
                                            eqn.params)
      elif prim.map_primitive:
        out_type, eqn_effects = _check_map(ctx_factory, prim, in_avals,
                                           eqn.params)
      else:
        out_type, eqn_effects = check_eqn(prim, in_avals, eqn.params)

      # Check the computed effect type matches the eqn's annotation, and is
      # included in the jaxpr's annotation.
      if prim is mutable_array_p:
        outvar, = eqn.outvars
        in_idx[outvar] = None  # type: ignore
      if eqn.effects != eqn_effects:
        raise JaxprTypeError("Inferred effects do not match equation effects. "
                             f"Equation effects: {eqn.effects}. "
                             f"Inferred effects: {eqn_effects}")
      for eff in eqn.effects:
        if isinstance(eff, effects.JaxprInputEffect):
          eqn_invar = eqn.invars[eff.input_index]
          if (jaxpr_index := in_idx.get(eqn_invar, sentinel)) is sentinel:
            raise JaxprTypeError(
                "Invalid `JaxprInputEffect`: must correspond to a jaxpr invar")
          jaxpr_effect = eff.replace(input_index=jaxpr_index)
          if jaxpr_effect not in jaxpr.effects:
            raise JaxprTypeError(
                "Invalid `JaxprInputEffect`: must be present in jaxpr. "
                f"{jaxpr_effect} is not in {jaxpr.effects}.")
        elif isinstance(eff, NamedAxisEffect):
          # It is valid for a primitive to discharge the named axis effect.
          continue
        elif eff not in jaxpr.effects:
          raise JaxprTypeError("Equation effect not present in jaxpr effects. "
                               f"Equation effect: {eff}. "
                               f"Jaxpr effects: {jaxpr.effects}")

      # Check out_type matches the let-binders' annotation (after substitution).
      out_type = substitute_vars_in_output_ty(out_type, eqn.invars, eqn.outvars)
      map(write, eqn.outvars, out_type)

    except JaxprTypeError as e:
      ctx, settings = ctx_factory()
      msg, = e.args
      src = source_info_util.summarize(eqn.source_info)
      msg = "\n\n".join([msg, "in equation:", str(pp.nest(2, pp_eqn(eqn, ctx, settings))),
                         f"from source: {src}"])
      raise JaxprTypeError(msg, eqn_idx) from None

  # TODO(mattjj): include output type annotation on jaxpr and check it here
  map(read, jaxpr.outvars)

def check_type(
    ctx_factory: Callable[[], tuple[JaxprPpContext, JaxprPpSettings]],
    env: set[Var],
    ty: AbstractValue,
  ) -> None:
  if isinstance(ty, DShapedArray):
    # Check all elements in the shape tuple are well-typed.
    for d in ty.shape:
      if (isinstance(d, int) or
          isinstance(d, DArray) and not d.shape and type(d.dtype) == bint):
        continue
      elif isinstance(d, Var):
        if d not in env:
          ctx, _ = ctx_factory()
          raise JaxprTypeError(f"unbound axis size: '{pp_var(d, ctx)}'")
        if not isinstance(d.aval, (ShapedArray, DShapedArray)):
          raise JaxprTypeError(f"axis size with unexpected type annotation: "
                               f"{d.aval} of type {type(d.aval)}")
        if isinstance(d.aval, ShapedArray):
          shape, dtype = d.aval.shape, d.aval.dtype
          if shape: raise JaxprTypeError(f"axis size nonscalar: {d.aval}")
          if not dtypes.issubdtype(dtype, np.integer):
            raise JaxprTypeError(f"axis size with non-integer dtype: {d.aval}")
        else:
          assert isinstance(d.aval, DShapedArray)
          shape, dtype = d.aval.shape, d.aval.dtype
          if shape: raise JaxprTypeError(f"axis size nonscalar: {d.aval}")
          if type(dtype) is not bint:
            raise JaxprTypeError(
                f"DArray axis size with non-bint dtype: {d.aval}")
      else:
        raise JaxprTypeError(f"unexpected type in shape: {type(d)}")
  else:
    return  # Except in above case(s), all syntactic forms are valid

def substitute_vars_in_output_ty(
    out_type: Sequence[AbstractValue],  # shapes may contain InDBIdx / OutDBIdx
    in_atoms: Sequence[Atom],
    out_binders: Sequence[Var],
  ) -> list[AbstractValue]:  # shapes may contain Vars
  in_atoms = [x.val if type(x) is Literal else x for x in in_atoms]
  result = []
  for aval in out_type:
    if type(aval) is DShapedArray:
      shape = [in_atoms[d.val] if type(d) is InDBIdx else
               out_binders[d.val] if type(d) is OutDBIdx else
               d for d in aval.shape]
      aval = aval.update(shape=tuple(shape))
    result.append(aval)
  return result

def check_eqn(prim, in_avals, params):
  for jaxpr in jaxprs_in_params(params):
    check_jaxpr(jaxpr)

  out_avals, effects = prim.abstract_eval(*in_avals, **params)
  if not prim.multiple_results:
    out_avals = [out_avals]
  return out_avals, effects

def _check_call(ctx_factory, prim, in_atoms, params):
  if "call_jaxpr" not in params:
    raise JaxprTypeError(
        f"Call primitive {prim} missing 'call_jaxpr' parameter")
  call_jaxpr = params["call_jaxpr"]

  if len(in_atoms) != len(call_jaxpr.invars):
    raise JaxprTypeError(f"Call primitive {prim} with {len(in_atoms)} "
                         f"operands cannot call jaxpr with "
                         f"{len(call_jaxpr.invars)} inputs")

  # Check `call_jaxpr` can be applied to in_atoms.
  env: dict[Var, Atom] = {}
  def substitute(aval: AbstractValue):
    if isinstance(aval, DShapedArray):
      aval = aval.update(shape=tuple(env.get(d, d) for d in aval.shape))  # type: ignore
    return aval
  for v, x in zip(call_jaxpr.invars, in_atoms):
    if not typecompat(substitute(v.aval), x.aval):
      # TODO(mattjj): vars in error message are confusing b/c of Var.__repr__
      raise JaxprTypeError(f"Call primitive {prim} passes operand {x} of type "
                           f"{x.aval} to jaxpr expecting type "
                           f"{substitute(v.aval)}")
    env[v] = x if type(x) is Var else x.val

  _check_jaxpr(ctx_factory, call_jaxpr)

  invars, outvars = call_jaxpr.invars, call_jaxpr.outvars
  in_map : dict[Var,  InDBIdx] = {v:  InDBIdx(i) for i, v in enumerate( invars)}
  out_map: dict[Var, OutDBIdx] = {x: OutDBIdx(i) for i, x in enumerate(outvars)
                                  if type(x) is Var}
  out_avals = [x.aval for x in call_jaxpr.outvars]
  out_type = [a.update(shape=tuple(in_map.get(d, out_map.get(d))
                                   if type(d) is Var else d for d in a.shape))
              if type(a) is DShapedArray else a for a in out_avals]
  return out_type, call_jaxpr.effects

def _check_map(ctx_factory, prim, in_avals, params):
  if "call_jaxpr" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'call_jaxpr' parameter")
  call_jaxpr = params["call_jaxpr"]
  ordered_effects_ = effects.ordered_effects.filter_in(call_jaxpr.effects)
  if ordered_effects_:
    raise JaxprTypeError(
        f"Map primitive {prim} mapping ordered effects: {ordered_effects_}")
  if "axis_size" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'axis_size' parameter")
  axis_size = params["axis_size"]
  if "axis_name" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'axis_name' parameter")
  axis_name = params["axis_name"]
  if "in_axes" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'in_axes' parameter")
  in_axes = params["in_axes"]
  if "out_axes" not in params:
    raise JaxprTypeError(f"Map primitive {prim} missing 'out_axes' parameter")
  out_axes = params["out_axes"]

  binder_avals = [unmapped_aval(axis_size, axis_name, in_axis, v.aval)
                  if in_axis is not None else v.aval
                  for v, in_axis in zip(call_jaxpr.invars, in_axes)]
  for binder_aval, in_aval in zip(binder_avals, in_avals):
    if not typecompat(binder_aval, in_aval):
      raise JaxprTypeError(f"Call primitive {prim} passes operand {in_aval} "
                           f"to jaxpr expecting {binder_aval}")

  with extend_axis_env(params['axis_name'], axis_size, None):
    _check_jaxpr(ctx_factory, call_jaxpr)

  mapped_out_avals = [v.aval for v in call_jaxpr.outvars]
  out_avals = [unmapped_aval(axis_size, axis_name, out_axis, aval)
               if out_axis is not None else aval
               for aval, out_axis in zip(mapped_out_avals, out_axes)]
  return out_avals, filter_named_axis_effects(call_jaxpr.effects, {axis_name})


# ------------------- Jaxpr printed representation -------------------

def pp_toplevel_jaxpr(jaxpr_to_print, *, source_info=False, print_shapes=True,
                      custom_pp_eqn_rules=True, name_stack=False,
                      print_effects: bool = False) -> pp.Doc:
    context = JaxprPpContext()
    settings = JaxprPpSettings(
        source_info=source_info,
        print_shapes=print_shapes,
        custom_pp_eqn_rules=custom_pp_eqn_rules,
        name_stack=name_stack,
        print_effects=print_effects)

    # Compute how many times each jaxpr is used.
    names = defaultdict[Jaxpr, str](lambda: "jaxpr")
    jaxpr_counts = Counter[Jaxpr]()
    s = deque([jaxpr_to_print])
    while s:
      jaxpr = s.popleft()
      jaxpr_counts[jaxpr] += 1
      for eqn in jaxpr.eqns:
        # TODO(slebedev): Come up with a more elaborate heuristic for name=.
        name = eqn.params.get("name")
        if name is None:
          s.extend(jaxprs_in_params(eqn.params))
          continue
        name = name.strip("<>")  # <lambda> -> lambda
        for subjaxpr in jaxprs_in_params(eqn.params):
          s.append(subjaxpr)
          names.setdefault(subjaxpr, name)

    # Pull jaxprs occurring more than once to the top-level, making sure
    # that their names are unique.
    docs = []
    name_counts = Counter[str]()
    for jaxpr, c in jaxpr_counts.items():
      if c == 1:
        continue
      name = names[jaxpr]
      if (count := name_counts[name]) > 0:
        name_counts[name] += 1
        name += str(count)
        name_counts[name] += 1
      else:
        name_counts[name] += 1
      docs.append(pp_top_level_jaxpr(name, jaxpr, context, settings))
      context.used_names.add(name)
      context.top_level_jaxprs[jaxpr] = name
    docs.append(pp_jaxpr(jaxpr_to_print, context, settings))
    return pp.concat(docs)


class JaxprPpSettings(NamedTuple):
  print_shapes: bool = True
  source_info: bool = False
  name_stack: bool = False
  custom_pp_eqn_rules: bool = True
  print_effects: bool = False

def _encode_digits_alphabetic(n: int) -> str:
  if n == -1:
    return '*'
  s = ''
  while len(s) == 0 or n:
    n, i = n // 26, n % 26
    s = chr(97 + i % 26) + s
  return s

# A JaxprPpContext allows us to globally uniquify variable names within nested
# Jaxprs.
class JaxprPpContext:
  var_names: defaultdict[Var, str]
  used_names: MutableSet[str]
  top_level_jaxprs: MutableMapping[Jaxpr, str]

  def __init__(self) -> None:
    self.top_level_jaxprs = {}
    self.used_names = set()
    fresh_names: Iterator[str] = (
        name
        for i in it.count()
        if (name := _encode_digits_alphabetic(i)) not in self.used_names
    )
    self.var_names = defaultdict(fresh_names.__next__)


def pp_var(v: Var | Literal, context: JaxprPpContext) -> str:
  if isinstance(v, (Literal, DropVar)): return str(v)
  return f"{context.var_names[v]}{v.suffix}"

def pp_aval(a: AbstractValue, context: JaxprPpContext) -> str:
  if isinstance(a, DShapedArray):
    shape = [pp_var(d, context) if type(d) is Var else str(d) for d in a.shape]
    dtype = _short_dtype_name(a.dtype)
    return f'{dtype}[{",".join(shape)}]'
  else:
    return a.str_short(short_dtypes=True)

def pp_vars(vs: Sequence[Any], context: JaxprPpContext,
            *, separator="", print_shapes: bool = False) -> pp.Doc:
  if print_shapes:
    return pp.nest(2, pp.group(
      pp.join(pp.text(separator) + pp.group(pp.brk()), [
        pp.text(pp_var(v, context)) +
        pp.type_annotation(pp.text(":" + pp_aval(v.aval, context)))
        for v in vs
      ])
    ))
  else:
    return pp.nest(2, pp.group(
      pp.join(pp.text(separator) + pp.group(pp.brk()),
              [pp.text(pp_var(v, context)) for v in vs])
    ))

def pp_kv_pair(k:str, v: Any, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if type(v) is tuple and all(isinstance(j, (Jaxpr, ClosedJaxpr)) for j in v):
    pp_v = pp_jaxprs(v, context, settings)
  elif isinstance(v, Jaxpr):
    pp_v = pp_jaxpr(v, context, settings)
  elif isinstance(v, ClosedJaxpr):
    pp_v = pp_jaxpr(v.jaxpr, context, settings)
  else:
    pp_v = pp.text(str(v))
  return pp.text(f'{k}=') + pp_v

def pp_kv_pairs(kv_pairs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if not kv_pairs:
    return pp.nil()
  return pp.group(
    pp.nest(2, pp.concat([
      pp.text("["),  pp.brk(""),
      pp.join(pp.brk(), [pp_kv_pair(k, v, context, settings) for k, v in kv_pairs])
    ]))
    + pp.brk("") + pp.text("]")
  )

def pp_eqn(eqn: JaxprEqn, context: JaxprPpContext, settings: JaxprPpSettings
           ) -> pp.Doc:
  rule = (_pp_eqn if not settings.custom_pp_eqn_rules else
          pp_eqn_rules.get(eqn.primitive, _pp_eqn))
  doc = rule(eqn, context, settings)  # type: ignore[operator]
  user_frame = source_info_util.user_frame(eqn.source_info)
  return doc if user_frame is None else pp.source_map(doc, user_frame)

def _pp_eqn(eqn, context, settings, params=None) -> pp.Doc:
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  if params is None:
    params = sorted(eqn.params)
  name_stack_annotation = f'[{eqn.source_info.name_stack}]' if settings.name_stack else None
  lhs = pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name, annotation=name_stack_annotation),
         pp_kv_pairs([(p, eqn.params[p]) for p in params], context, settings),
         pp.text(" ") + pp_vars(eqn.invars, context)]
  if lhs.format():
    return pp.concat([lhs, pp.text(" = ", annotation=annotation), *rhs])
  else:
    return pp.concat(rhs)
CustomPpEqnRule = Callable[[JaxprEqn, JaxprPpContext, JaxprPpSettings], pp.Doc]
pp_eqn_rules: dict[Primitive, CustomPpEqnRule]  = {}

def pp_eqns(eqns, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  return pp.join(
    pp.brk("; "),
    [pp_eqn(e, context, settings) for e in eqns])

def _compact_eqn_should_include(k: str, v: Any) -> bool:
  if k == 'branches': return False
  if isinstance(v, (Jaxpr, ClosedJaxpr)): return False
  if (isinstance(v, tuple) and
      any(isinstance(e, (Jaxpr, ClosedJaxpr)) for e in v)):
    return False
  return True

def str_eqn_compact(primitive: Primitive, params: dict[Any, Any]) -> str:
  "Compact equation to string conversion used in HLO metadata."
  if primitive in custom_str_eqn_compact_rules:
    return custom_str_eqn_compact_rules[primitive](primitive, params)
  primitive_name = primitive.name
  kvs = " ".join(f"{k}={v}" for k, v in params.items()
                 if _compact_eqn_should_include(k, v))
  return f"{primitive_name}[{kvs}]" if len(kvs) > 0 else primitive_name
custom_str_eqn_compact_rules: dict[
    Primitive, Callable[[Primitive, dict[Any, Any]], str]
] = {}

def pp_jaxpr_skeleton(jaxpr, eqns_fn, context: JaxprPpContext,
                      settings: JaxprPpSettings) -> pp.Doc:
  constvars = pp_vars(jaxpr.constvars, context, print_shapes=settings.print_shapes)
  invars = pp_vars(jaxpr.invars, context, print_shapes=settings.print_shapes)
  eqns = eqns_fn()
  outvars = pp.concat([
    pp.text("("), pp_vars(jaxpr.outvars, context, separator=","),
    pp.text(")" if len(jaxpr.outvars) != 1 else ",)")])
  if settings.print_effects:
    # TODO(sharadmv): render an entire signature here
    eff_text = [pp.text(" : { ")]
    for i, eff in enumerate(jaxpr.effects):
      if i > 0:
        eff_text.append(pp.text(", "))
      if isinstance(eff, effects.JaxprInputEffect):
        index = eff.input_index
        all_vars = [*jaxpr.constvars, *jaxpr.invars]
        eff_text.append(pp_effect(eff.replace(input_index=all_vars[index]),
                                  context))
      else:
        eff_text.append(pp_effect(eff, context))
    eff_text.append(pp.text(" }"))
  else:
    eff_text = []
  return pp.group(pp.nest(2, pp.concat([
    pp.text("{ "), pp.keyword(pp.text("lambda ")),
    constvars, pp.text("; "), invars,
    pp.text(". "), pp.keyword(pp.text("let")),
    pp.nest(2, pp.brk() + eqns), pp.brk(),
    pp.keyword(pp.text("in ")), outvars,
    pp.concat(eff_text)
  ])) + pp.text(" }"))


def pp_top_level_jaxpr(
    name: str,
    jaxpr: Jaxpr,
    context: JaxprPpContext,
    settings: JaxprPpSettings,
) -> pp.Doc:
  return pp.concat([
      pp.text("let " + name + " = "),
      pp_jaxpr(jaxpr, context, settings),
      pp.text(" in"),
      pp.brk(),
  ])


def pp_jaxpr(
    jaxpr: Jaxpr,
    context: JaxprPpContext,
    settings: JaxprPpSettings,
) -> pp.Doc:
  if name := context.top_level_jaxprs.get(jaxpr):
    return pp.text(name)
  eqns_fn = lambda: pp_eqns(jaxpr.eqns, context, settings)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)


def pp_jaxprs(jaxprs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  jaxprs = [j.jaxpr if isinstance(j, ClosedJaxpr) else j for j in jaxprs]
  return pp.group(pp.nest(2, pp.concat([
      pp.text('('), pp.brk(""),
      pp.join(pp.brk(), map(lambda x: pp_jaxpr(x, context, settings), jaxprs))]
    )) + pp.brk("") + pp.text(')')
  )


def pp_jaxpr_eqn_range(jaxpr: Jaxpr, lo: int, hi: int, context: JaxprPpContext,
                       settings: JaxprPpSettings) -> pp.Doc:
  lo = max(lo, 0)
  hi = max(lo, min(hi, len(jaxpr.eqns)))
  eqns = jaxpr.eqns[lo:hi]
  def eqns_fn():
    pps = []
    if len(eqns) == 0 and len(jaxpr.eqns) != 0:
      pps.append(pp.text('...'))
    else:
      if lo != 0:
        pps.append(pp.text('...'))
      pps.extend(map((lambda e: pp_eqn(e, context, settings)), eqns))
      if hi != len(jaxpr.eqns):
        pps.append(pp.text('...'))
    return pp.join(pp.brk("; "), pps)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)

def pp_effect(effect: Effect, context: JaxprPpContext) -> pp.Doc:
  if hasattr(effect, "_pretty_print"):
    return effect._pretty_print(context)
  return pp.text(str(effect))

# ------------------- Jaxpr util -------------------

def last_used(jaxpr: Jaxpr) -> dict[Var, JaxprEqn | None]:
  """Returns a mapping from every var in jaxpr to what equation uses it last."""
  last_used: dict[Var, JaxprEqn | None] = {
      v: None for v in jaxpr.outvars if not isinstance(v, Literal)}
  for eqn in reversed(jaxpr.eqns):
    for v in eqn.invars:
      if not isinstance(v, Literal) and v not in last_used:
        last_used[v] = eqn
  return last_used

def clean_up_dead_vars(eqn: JaxprEqn, env: dict[Var, Any],
                       last_used: dict[Var, JaxprEqn | None]):
  """Remove all eqn.invars from env if eqn is the last time they were used."""
  for v in {v for v in eqn.invars if not isinstance(v, Literal)}:
    if last_used[v] is eqn:
      # Delete ref to variable when it is no longer needed by next equations.
      del env[v]

# Used in shard_map for converting avals
shard_aval_handlers = {}  # type: ignore
unshard_aval_handlers = {}  # type: ignore
