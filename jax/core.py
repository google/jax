# Copyright 2018 Google LLC
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


import operator
from operator import attrgetter
from contextlib import contextmanager
from collections import namedtuple
from functools import total_ordering
import itertools as it
from weakref import ref
import threading
import types
from typing import (Any, Callable, ClassVar, Dict, Generator,
                    Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple,
                    Type, Union, cast, Iterable, Hashable)

import numpy as np

from ._src import dtypes
from ._src import config as jax_config
from ._src.config import FLAGS, config
from .errors import (ConcretizationTypeError, TracerArrayConversionError,
                     TracerIntegerConversionError)
from . import linear_util as lu

from jax._src import source_info_util
from ._src.util import (safe_zip, safe_map, partial, curry, prod, partialmethod,
                        tuple_insert, tuple_delete, as_hashable_function,
                        HashableFunction)
from ._src.pprint_util import pp, vcat, PrettyPrint

from ._src import traceback_util
traceback_util.register_exclusion(__file__)

zip = safe_zip
map = safe_map


# -------------------- jaxprs --------------------

class Jaxpr:
  constvars: List['Var']
  invars: List['Var']
  outvars: List['Atom']
  eqns: List['JaxprEqn']

  def __init__(self, constvars: Sequence['Var'], invars: Sequence['Var'],
               outvars: Sequence['Atom'], eqns: Sequence['JaxprEqn']):
    """
    Args:
      constvars: list of variables introduced for constants. Array constants are
        replaced with such variables while scalar constants are kept inline.
      invars: list of input variables. Together, `constvars` and `invars` are
        the inputs to the Jaxpr.
      outvars: list of output variables.
      eqns: list of equations.
    """
    self.constvars = list(constvars)
    self.invars = list(invars)
    self.outvars = list(outvars)
    self.eqns = list(eqns)

  def __str__(self):
    return str(pp_jaxpr(self))
  __repr__ = __str__


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
  jaxpr: Jaxpr
  consts: List['Any']

  def __init__(self, jaxpr: Jaxpr, consts: Sequence):
    assert len(consts) == len(jaxpr.constvars)
    self.jaxpr = jaxpr
    self.consts = list(consts)

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

  def map_jaxpr(self, f):
    return ClosedJaxpr(f(self.jaxpr), self.consts)

  def __str__(self): return str(self.jaxpr)
  def __repr__(self): return repr(self.jaxpr)

@curry
def jaxpr_as_fun(closed_jaxpr: ClosedJaxpr, *args):
  return eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)


class JaxprEqn(NamedTuple):
  invars: List['Atom']
  outvars: List['Var']
  primitive: 'Primitive'
  params: Dict[str, Any]
  source_info: Optional[source_info_util.Traceback]

  def __repr__(self): return str(pp_eqn(self)).rstrip()

def new_jaxpr_eqn(invars, outvars, primitive, params, source_info=None):
  return JaxprEqn(invars, outvars, primitive, params, source_info)


@total_ordering
class Var:
  # TODO(frostig,mattjj): We don't override __eq__ or __hash__, so comparison is
  # by object id, but pretty printing might collide.
  count: int
  suffix: str
  aval: 'AbstractValue'

  def __init__(self, count: int, suffix: str, aval: 'AbstractValue'):
    self.count = count
    self.suffix = suffix
    self.aval = raise_to_shaped(aval)

  def __lt__(self, other):
    if not isinstance(other, Var):
      return NotImplemented
    else:
      return (self.count, self.suffix) < (other.count, other.suffix)

  def __repr__(self):
    rem = self.count
    s = ''
    while True:
      rem, i = rem // 26, rem % 26
      s = chr(97 + i % 26) + s
      if not rem:
        break
    return s + self.suffix

def _jaxpr_vars(jaxpr):
  return it.chain(
      jaxpr.invars, jaxpr.constvars,
      (v for eqn in jaxpr.eqns for v in eqn.outvars))

def gensym(jaxprs: Optional[Sequence[Jaxpr]] = None,
           suffix: str = '') -> Callable[['AbstractValue'], Var]:
  """Produce distinct variables, printed with the optional suffix.

  If `jaxprs` is provided, the variables produced will be distinct from those in
  any of the given jaxprs.
  """
  if jaxprs is None:
    start = 0
  else:
    all_vars = it.chain.from_iterable(_jaxpr_vars(j) for j in jaxprs)
    start = 1 + max((v.count for v in all_vars), default=-1)
  counter = it.count(start=start)
  return lambda aval: Var(next(counter), suffix, aval)

# In a jaxpr, `dropvar` can appear in place of a bound variable to indicate that
# the assignment is dropped, i.e. that an expression's output value will never
# be read. In that sense, `dropvar` is not a variable, but it is convenient to
# treat it as a special case of one. Its `aval` is similarly inexact.
class DropVar(Var):
  count = -1
  suffix = ''
  def __init__(self): pass
  @property
  def aval(self): return abstract_unit
  def __repr__(self): return '_'
dropvar = DropVar()

class Literal:
  __slots__ = ["val", "hash"]

  val: Any
  hash: Optional[int]

  def __init__(self, val):
    self.val = val
    try:
      self.hash = hash(val)
    except TypeError:
      if type(val) in literalable_types:
        try:
          self.hash = hash((val.item(), val.dtype))
        except (TypeError, AttributeError, ValueError):
          self.hash = None

  @property
  def aval(self):
    return raise_to_shaped(get_aval(self.val))

  def __hash__(self):
    assert False

  def __repr__(self):
    if hasattr(self, 'hash'):
      return '{}'.format(self.val)
    else:
      return 'Literal(val={})'.format(self.val)

literalable_types: Set[type] = set()

Atom = Union[Var, Literal]

class Primitive:
  name: str
  multiple_results = False  # set for multi-output primitives
  call_primitive = False    # set for call primitives processed in final style
  map_primitive = False     # set for map primitives processed in final style
  _dispatch_on_params = False  # whether to include axis names from params in dispatch

  def __init__(self, name: str):
    self.name = name

  def __repr__(self):
    return '{}'.format(self.name)


  def bind(self, *args, **params):
    assert (not config.jax_enable_checks or
            all(isinstance(arg, Tracer) or valid_jaxtype(arg) for arg in args)), args
    top_trace = find_top_trace(
        args, used_axis_names(self, params) if self._dispatch_on_params else None)
    tracers = map(top_trace.full_raise, args)
    out = top_trace.process_primitive(self, tracers, params)
    return map(full_lower, out) if self.multiple_results else full_lower(out)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_abstract_eval(self, abstract_eval):
    self.abstract_eval = abstract_eval
    return abstract_eval

  def def_custom_bind(self, bind):
    self.bind = bind
    return bind

  def impl(self, *args, **params):
    raise NotImplementedError("Evaluation rule for '{}' not implemented"
                              .format(self.name))

  def abstract_eval(self, *args, **params):
    raise NotImplementedError("Abstract evaluation for '{}' not implemented"
                              .format(self.name))


# -------------------- lifting --------------------

# TODO(necula): this belongs next to pe.new_eqn_recipe, but is needed in
# core.py. Plan to move all these utilities to jaxpr.py.
def extract_call_jaxpr(
  primitive: Primitive,
  params: Dict[str, Any]) -> Tuple[Optional[Jaxpr], Dict[str, Any]]:
  """Extract the call primitive subjaxpr from the params.

  Returns the subjaxpr and the params without the "call_jaxpr" value. If this is
  not a call primitive then returns (None, params).
  """
  if not (primitive.call_primitive or primitive.map_primitive):
    return (None, params)
  else:
    assert "call_jaxpr" in params
    new_params = dict(params)
    del new_params["call_jaxpr"]
    return (params["call_jaxpr"], new_params)


def traverse_jaxpr_params(f, params):
  """Applies f to each jaxpr parameter and returns a tuple of returned values."""
  return tuple(f(param if type(param) is Jaxpr else param.jaxpr)
               for param in params.values()
               if type(param) in (Jaxpr, ClosedJaxpr))


def eval_jaxpr(jaxpr: Jaxpr, consts, *args):
  def read(v):
    if type(v) is Literal:
      return v.val
    else:
      return env[v]

  def write(v, val):
    env[v] = val

  env: Dict[Var, Any] = {}
  write(unitvar, unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    call_jaxpr, params = extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      subfuns = [lu.wrap_init(partial(eval_jaxpr, call_jaxpr, ()))]
    else:
      subfuns = []
    if eqn.primitive in initial_to_final_param_rules:
      bind_params = initial_to_final_param_rules[eqn.primitive](params)
    elif eqn.primitive.map_primitive:
      out_axes_thunk = HashableFunction(lambda: params['out_axes'],
                                        closure=params['out_axes'])
      bind_params = dict(params, out_axes_thunk=out_axes_thunk)
      del bind_params['out_axes']
    else:
      bind_params = params
    with source_info_util.user_context(eqn.source_info):
      ans = eqn.primitive.bind(*(subfuns + in_vals), **bind_params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  return map(read, jaxpr.outvars)

initial_to_final_param_rules: Dict[Primitive, Callable] = {}


# -------------------- tracing --------------------


class Trace:
  __slots__ = ['main', 'level', 'sublevel']

  main: 'MainTrace'
  level: int
  sublevel: 'Sublevel'

  def __init__(self, main: 'MainTrace', sublevel: 'Sublevel') -> None:
    self.main = main
    self.level = main.level
    self.sublevel = sublevel

  def full_raise(self, val) -> 'Tracer':
    if not isinstance(val, Tracer):
      return self.pure(val)
    val._assert_live()
    level = self.level
    sublevel = self.sublevel
    if val._trace.main is self.main:
      if val._trace.sublevel == sublevel:
        return val
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

  def pure(self, val):
    raise NotImplementedError("must override")

  def lift(self, tracer):
    raise NotImplementedError("must override")

  def sublift(self, tracer):
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

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    msg = (f"{type(self)} must override process_custom_jvp_call "
           "to handle custom_jvp primitives")
    raise NotImplementedError(msg)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
    msg = (f"{type(self)} must override process_custom_vjp_call "
           "to handle custom_vjp primitives")
    raise NotImplementedError(msg)

  def process_remat_call(self, primitive, f, tracers, params):
    return self.process_call(primitive, f, tracers, params)  # default


def escaped_tracer_error(tracer, detail=None):
  num_frames = FLAGS.jax_tracer_error_num_traceback_frames
  msg = ("Encountered an unexpected tracer. Perhaps this tracer escaped "
         "through global state from a previously traced function.\n"
         "The functions being transformed should not save traced values to "
         "global state.")
  if detail:
    msg += " Detail: {}.".format(detail)
  try:
    line_info = tracer._line_info
  except AttributeError:
    pass
  else:
    msg += ('\nThe tracer that caused this error was created on line '
            f'{source_info_util.summarize(line_info)}.\n')
    if num_frames > 0:
      msg += (f'When the tracer was created, the final {num_frames} stack '
              'frames (most recent last) excluding JAX-internal frames were:\n'
              f'{source_info_util.summarize(line_info, num_frames=num_frames)}')
  dbg = getattr(tracer._trace.main, 'debug_info', None)
  if dbg is not None:
    msg += ('\nThe function being traced when the tracer leaked was '
            f'{dbg.func_src_info} traced for {dbg.traced_for}.')
  msg += ('\nTo catch the leak earlier, try setting the environment variable '
          'JAX_CHECK_TRACER_LEAKS or using the `jax.checking_leaks` context '
          'manager.')
  return UnexpectedTracerError(msg)

class UnexpectedTracerError(Exception): pass

class Tracer:
  __array_priority__ = 1000
  __slots__ = ['_trace', '__weakref__', '_line_info']

  def __array__(self, *args, **kw):
    raise TracerArrayConversionError(self)

  def __index__(self):
    raise TracerIntegerConversionError(self)

  def __init__(self, trace: Trace):
    self._trace = trace

  def __iter__(self):
    return iter(self.aval._iter(self))

  def __len__(self):
    return self.aval._len(self)

  @property
  def aval(self):
    raise NotImplementedError("must override")

  def _assert_live(self) -> None:
    pass  # Override for liveness checking

  # Python looks up special methods only on classes, not instances. This means
  # these methods needs to be defined explicitly rather than relying on
  # __getattr__.
  def __neg__(self): return self.aval._neg(self)
  def __pos__(self): return self.aval._pos(self)
  def __eq__(self, other): return self.aval._eq(self, other)
  def __ne__(self, other): return self.aval._ne(self, other)
  def __lt__(self, other): return self.aval._lt(self, other)
  def __le__(self, other): return self.aval._le(self, other)
  def __gt__(self, other): return self.aval._gt(self, other)
  def __ge__(self, other): return self.aval._ge(self, other)
  def __abs__(self): return self.aval._abs(self)
  def __add__(self, other): return self.aval._add(self, other)
  def __radd__(self, other): return self.aval._radd(self, other)
  def __sub__(self, other): return self.aval._sub(self, other)
  def __rsub__(self, other): return self.aval._rsub(self, other)
  def __mul__(self, other): return self.aval._mul(self, other)
  def __rmul__(self, other): return self.aval._rmul(self, other)
  def __div__(self, other): return self.aval._div(self, other)
  def __rdiv__(self, other): return self.aval._rdiv(self, other)
  def __truediv__(self, other): return self.aval._truediv(self, other)
  def __rtruediv__(self, other): return self.aval._rtruediv(self, other)
  def __floordiv__(self, other): return self.aval._floordiv(self, other)
  def __rfloordiv__(self, other): return self.aval._rfloordiv(self, other)
  def __divmod__(self, other): return self.aval._divmod(self, other)
  def __rdivmod__(self, other): return self.aval._rdivmod(self, other)
  def __mod__(self, other): return self.aval._mod(self, other)
  def __rmod__(self, other): return self.aval._rmod(self, other)
  def __pow__(self, other): return self.aval._pow(self, other)
  def __rpow__(self, other): return self.aval._rpow(self, other)
  def __matmul__(self, other): return self.aval._matmul(self, other)
  def __rmatmul__(self, other): return self.aval._rmatmul(self, other)
  def __and__(self, other): return self.aval._and(self, other)
  def __rand__(self, other): return self.aval._rand(self, other)
  def __or__(self, other): return self.aval._or(self, other)
  def __ror__(self, other): return self.aval._ror(self, other)
  def __xor__(self, other): return self.aval._xor(self, other)
  def __rxor__(self, other): return self.aval._rxor(self, other)
  def __invert__(self): return self.aval._invert(self)
  def __lshift__(self, other): return self.aval._lshift(self, other)
  def __rlshift__(self, other): return self.aval._rlshift(self, other)
  def __rshift__(self, other): return self.aval._rshift(self, other)
  def __rrshift__(self, other): return self.aval._rrshift(self, other)
  def __getitem__(self, idx): return self.aval._getitem(self, idx)
  def __nonzero__(self): return self.aval._nonzero(self)
  def __bool__(self): return self.aval._bool(self)
  def __int__(self): return self.aval._int(self)
  def __long__(self): return self.aval._long(self)
  def __hex__(self): return self.aval._hex(self)
  def __oct__(self): return self.aval._oct(self)
  def __float__(self): return self.aval._float(self)
  def __complex__(self): return self.aval._complex(self)

  def __setitem__(self, idx, val):
    raise TypeError("JAX 'Tracer' objects do not support item assignment")

  # NumPy also only looks up special methods on classes.
  def __array_module__(self, types): return self.aval._array_module(self, types)

  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    assert not config.jax_enable_checks or name != "aval"

    try:
      attr = getattr(self.aval, name)
    except KeyError as err:
      raise AttributeError(
          "{} has no attribute {}".format(self.__class__.__name__, name)
      ) from err
    else:
      t = type(attr)
      if t is aval_property:
        return attr.fget(self)
      elif t is aval_method:
        return types.MethodType(attr.fun, self)
      else:
        return attr

  def __repr__(self):
    base = pp('Traced<{}>with<{}>'.format(self.aval, self._trace))
    contents = self._contents()
    if contents:
      base += pp('  with ') >> vcat(pp('{} = '.format(name)) >> pp_payload
                                    for name, pp_payload in contents)
    return str(base)

  def _contents(self):
    try:
      return [(name, pp(repr(getattr(self, name)))) for name in self.__slots__]
    except AttributeError:
      return ()

  def __copy__(self):
    return self

  def __deepcopy__(self, unused_memo):
    return self

  def _origin_msg(self) -> str:
    return ""

# these can be used to set up forwarding of properties and instance methods from
# Tracer instances to the underlying avals
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])


class EvalTrace(Trace):
  # See comments in https://github.com/google/jax/pull/3370
  def pure(self, x): return x
  lift = sublift = pure

  def process_primitive(self, primitive, tracers, params):
    return primitive.impl(*tracers, **params)

  def process_call(self, primitive, f, tracers, params):
    return primitive.impl(f, *tracers, **params)
  process_map = process_call

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    del primitive, jvp  # Unused.
    with new_sublevel():
      return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
    del primitive, fwd, bwd, out_trees  # Unused.
    with new_sublevel():
      return fun.call_wrapped(*tracers)


class MainTrace:
  level: int
  trace_type: Type[Trace]
  payload: Dict[str, Any]

  def __init__(self, level, trace_type, **payload) -> None:
    self.level = level
    self.trace_type = trace_type
    self.payload = payload

  def __repr__(self) -> str:
    return "MainTrace({},{})".format(self.level, self.trace_type.__name__)

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
  stack: List[MainTrace]
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

  @property
  def active_stack(self) -> List[MainTrace]:
    i, = (i for i, m in enumerate(self.stack) if m is self.dynamic)
    return self.stack[i:]

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

class TraceState:
  trace_stack: TraceStack
  substack: List[Sublevel]
  axis_env: List[AxisEnvFrame]

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


# The global state of the tracer is accessed by a thread-local object.
# This allows concurrent tracing in separate threads; passing traced objects
# between threads is forbidden.
class ThreadLocalState(threading.local):
  def __init__(self):
    self.trace_state = TraceState()
    jax_config.update_thread_local_jit_state(
        dynamic_trace_state=self.trace_state.trace_stack.dynamic)
thread_local_state = ThreadLocalState()

def trace_state_clean() -> bool:
  trace_state = thread_local_state.trace_state
  return (trace_state.substack == [Sublevel(0)] and
          trace_state.axis_env == [] and
          trace_state.trace_stack.stack == [MainTrace(0, EvalTrace)] and
          trace_state.trace_stack.dynamic == MainTrace(0, EvalTrace))

def reset_trace_state() -> bool:
  "Reset the global trace state and return True if it was already clean."
  if not trace_state_clean():
    thread_local_state.trace_state.__init__()  # type: ignore
    return False
  else:
    return True

def cur_sublevel() -> Sublevel:
  return thread_local_state.trace_state.substack[-1]

@contextmanager
def new_main(trace_type: Type[Trace],
             dynamic: bool = False,
             **payload) -> Generator[MainTrace, None, None]:
  # See comments in https://github.com/google/jax/pull/3370
  stack = thread_local_state.trace_state.trace_stack
  level = stack.next_level()
  main = MainTrace(level, trace_type, **payload)
  stack.push(main)
  if dynamic:
    prev_dynamic, stack.dynamic = stack.dynamic, main
    jax_config.update_thread_local_jit_state(dynamic_trace_state=stack.dynamic)

  try:
    yield main
  finally:
    stack.pop()
    if dynamic:
      stack.dynamic = prev_dynamic
      jax_config.update_thread_local_jit_state(dynamic_trace_state=stack.dynamic)

  if config.jax_check_tracer_leaks:
    t = ref(main)
    del main
    if t() is not None:
      raise Exception(f'Leaked trace {t()}')

@contextmanager
def new_base_main(trace_type: Type[Trace]) -> Generator[MainTrace, None, None]:
  # See comments in https://github.com/google/jax/pull/3370
  stack = thread_local_state.trace_state.trace_stack
  main = MainTrace(0, trace_type)
  prev_dynamic, stack.dynamic = stack.dynamic, main
  prev_base, stack.stack[0] = stack.stack[0], main
  jax_config.update_thread_local_jit_state(dynamic_trace_state=stack.dynamic)
  try:
    yield main
  finally:
    stack.dynamic = prev_dynamic
    stack.stack[0] = prev_base
    jax_config.update_thread_local_jit_state(dynamic_trace_state=stack.dynamic)

  if config.jax_check_tracer_leaks:
    t = ref(main)
    del main
    if t() is not None:
      raise Exception('Leaked trace {}'.format(t()))

@contextmanager
def eval_context():
  with new_base_main(EvalTrace):
    yield

@contextmanager
def new_sublevel() -> Generator[None, None, None]:
  sublevel = Sublevel(len(thread_local_state.trace_state.substack))
  thread_local_state.trace_state.substack.append(sublevel)
  try:
    yield
  finally:
    thread_local_state.trace_state.substack.pop()

  if config.jax_check_tracer_leaks:
    t = ref(sublevel)
    del sublevel
    if t() is not None:
      raise Exception(f'Leaked sublevel {t()}.')

def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val

def find_top_trace(xs, axis_names=None) -> Trace:
  top_main: Optional[MainTrace] = None
  if axis_names:
    top_main = max((axis_frame(a).main_trace for a in axis_names),
                   default=None, key=lambda t: getattr(t, 'level', -1))
  top_tracer = max((x for x in xs if isinstance(x, Tracer)),
                   default=None, key=attrgetter('_trace.level'))
  if top_tracer is not None:
    top_tracer._assert_live()
    if top_tracer._trace.main.level > getattr(top_main, 'level', -1):
      top_main = top_tracer._trace.main
  dynamic = thread_local_state.trace_state.trace_stack.dynamic
  top_main = (dynamic if top_main is None or dynamic.level > top_main.level
              else top_main)
  return top_main and top_main.with_cur_sublevel()  # type: ignore


# -------------------- abstract values --------------------


class AbstractValue:
  __slots__: List[str] = []
  _num_buffers: int = 1  # number of buffers used to represent the value.

  def at_least_vspace(self):
    raise NotImplementedError("must override")

  def __repr__(self):
    try:
      kv_pairs = ('{}={}'.format(k, v) for k, v in self.__dict__.items())
      return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
    except AttributeError:
      return self.__class__.__name__

  def strip_weak_type(self) -> 'AbstractValue':
    return self

  def strip_named_shape(self) -> 'AbstractValue':
    return self

  def join(self, other):
    raise NotImplementedError("must override")

  def update(self, **kwargs):
    raise NotImplementedError("must override")

  def str_short(self):
    raise NotImplementedError("must override")

class Bot(AbstractValue): pass

bot = Bot()

class AbstractUnit(AbstractValue):
  # TODO(jakevdp): make it possible to set zero buffers
  # _num_buffers = 0
  def at_least_vspace(self): return self
  def join(self, other):
    if config.jax_enable_checks:
      assert other is abstract_unit, other
    return self
  def _eq(self, self_traced, other): return get_aval(other) is self
  def str_short(self): return '*'

abstract_unit = AbstractUnit()

def lattice_join(x: Optional[AbstractValue],
                 y: Optional[AbstractValue]) -> AbstractValue:
  if x is None:
    return cast(AbstractValue, y)
  elif y is None:
    return cast(AbstractValue, x)
  elif isinstance(x, type(y)):
    return y.join(x)
  elif isinstance(y, type(x)):
    return x.join(y)
  else:
    raise TypeError(x, y)

# For use in typing annotations to denote either a Tracer or a `valid_jaxtype`.
Value = Any

def valid_jaxtype(x):
  try:
    concrete_aval(x)
  except TypeError:
    return False
  else:
    return True

def check_valid_jaxtype(x):
  if not valid_jaxtype(x):
    raise TypeError(
      f"Value {repr(x)} of type {type(x)} is not a valid JAX type")


def concrete_aval(x):
  for typ in type(x).mro():
    handler = pytype_aval_mappings.get(typ)
    if handler: return handler(x)
  if hasattr(x, '__jax_array__'):
    return concrete_aval(x.__jax_array__())
  raise TypeError(f"Value {repr(x)} with type {type(x)} is not a valid JAX "
                   "type")


def get_aval(x):
  if isinstance(x, Tracer):
    return x.aval
  else:
    return concrete_aval(x)


pytype_aval_mappings: Dict[type, Callable[[Any], AbstractValue]] = {}


class Unit:
  def __repr__(self): return '*'
unit = Unit()
literalable_types.add(Unit)

class UnitVar(Var):
  count = -1
  suffix = ''
  def __init__(self): pass
  @property
  def aval(self): return abstract_unit
  def __repr__(self): return '*'
unitvar = UnitVar()

pytype_aval_mappings[Unit] = lambda _: abstract_unit

def concretization_function_error(fun, suggest_astype=False):
  fname = getattr(fun, "__name__", fun)
  fname_context = f"The problem arose with the `{fname}` function. "
  if suggest_astype:
    fname_context += ("If trying to convert the data type of a value, "
                      f"try using `x.astype({fun.__name__})` "
                      f"or `jnp.array(x, {fun.__name__})` instead.")
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

convert_element_type_p = Primitive('convert_element_type')

class UnshapedArray(AbstractValue):
  __slots__ = ['dtype', 'weak_type']
  array_abstraction_level = 2

  def __init__(self, dtype, weak_type=False):
    self.dtype = np.dtype(dtypes.canonicalize_dtype(dtype))
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

  _bool = _nonzero = concretization_function_error(bool)
  _float   = concretization_function_error(float, True)
  _int     = concretization_function_error(int, True)
  _complex = concretization_function_error(complex, True)
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)

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

  def str_short(self) -> str:
    return self.dtype.name

  def strip_weak_type(self):
    """Returns a copy of the aval with weak_type=False."""
    return self.update(weak_type=False)

  @property
  def shape(self):
    msg = ("UnshapedArray has no shape. Please open an issue at "
           "https://github.com/google/jax/issues because it's unexpected for "
           "UnshapedArray instances to ever be produced.")
    raise TypeError(msg)

class ShapedArray(UnshapedArray):
  __slots__ = ['shape', 'named_shape']
  array_abstraction_level = 1

  def __init__(self, shape, dtype, weak_type=False, named_shape={}):
    super(ShapedArray, self).__init__(dtype, weak_type=weak_type)
    self.shape = canonicalize_shape(shape)
    self.named_shape = dict(named_shape)

  def update(self, shape=None, dtype=None, weak_type=None, named_shape=None):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    if named_shape is None:
      named_shape = self.named_shape
    return ShapedArray(shape, dtype, weak_type, named_shape)

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: prod(self.shape))

  broadcast: ClassVar[Optional[aval_method]] = None
  transpose: ClassVar[Optional[aval_method]] = None
  reshape: ClassVar[Optional[aval_method]] = None
  _iter: ClassVar[Optional[staticmethod]] = None

  def __eq__(self, other):
    return (type(self) is type(other)
            and self.dtype == other.dtype and self.shape == other.shape
            and self.weak_type == other.weak_type
            and self.named_shape == other.named_shape)

  def __hash__(self):
    # can use hash(self.dtype) and rely on the fact that numpy reuses base dtype
    # objects, e.g. `np.zeros(3).dtype is np.zeros(4).dtype`, or we can use
    # the unique character code via hash(self.dtype.char)
    return hash((self.shape, self.dtype, self.weak_type,
                 tuple(self.named_shape.items())))

  def at_least_vspace(self):
    return ShapedArray(self.shape, primal_dtype_to_tangent_dtype(self.dtype),
                       self.weak_type, self.named_shape)

  def join(self, other):
    if symbolic_equal_shape(self.shape, other.shape) and self.dtype == other.dtype:
      weak_type = self.weak_type and other.weak_type
      named_shape = join_named_shapes(self.named_shape, other.named_shape)
      return self.update(weak_type=weak_type, named_shape=named_shape)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def str_short(self):
    shapestr = ','.join(map(str, self.shape))
    if self.named_shape:
      named_shapestr = ','.join(f'{k}:{v}' for k, v in self.named_shape.items())
      return f'{self.dtype.name}[{shapestr};{named_shapestr}]'
    else:
      return f'{self.dtype.name}[{shapestr}]'

  def strip_named_shape(self):
    return self.update(named_shape={})

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError as err:
      raise TypeError("len() of unsized object") from err  # same as numpy error

  def _len(self, ignored_tracer):
    return len(self)


def _forward_to_value(self, fun, ignored_tracer, *args):
  return fun(self.val, *args)

class ConcreteArray(ShapedArray):
  __slots__ = ['val']
  array_abstraction_level = 0

  def __init__(self, val, weak_type=False):
    super(ConcreteArray, self).__init__(np.shape(val), np.result_type(val),
                                        weak_type=weak_type)
    # Note: canonicalized self.dtype doesn't necessarily match self.val
    self.val = val
    assert self.dtype != np.dtype('O'), val

  def update(self, val=None, weak_type=None):
    if val is None:
      val = self.val
    if weak_type is None:
      weak_type = self.weak_type
    return ConcreteArray(val, weak_type)

  def __eq__(self, other):
    if (type(self) is type(other) and self.dtype == other.dtype
        and self.shape == other.shape and self.weak_type == other.weak_type):
      with eval_context():  # in case self.val is a DeviceArray
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
      named_shape = join_named_shapes(self.named_shape, other.named_shape)
      return ShapedArray(
          self.shape, self.dtype, weak_type=weak_type, named_shape=named_shape)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype,
                           weak_type=self.weak_type and other.weak_type)
    else:
      raise TypeError(self, other)

  def str_short(self) -> str:
    return str(self.val)

  _bool = _nonzero = partialmethod(_forward_to_value, bool)
  _int             = partialmethod(_forward_to_value, int)
  _hex             = partialmethod(_forward_to_value, hex)
  _oct             = partialmethod(_forward_to_value, oct)

  _float           = concretization_function_error(float, True)
  _complex         = concretization_function_error(complex, True)

def primal_dtype_to_tangent_dtype(primal_dtype):
  if not dtypes.issubdtype(primal_dtype, np.inexact):
    return dtypes.float0
  else:
    return primal_dtype

class AbstractToken(AbstractValue):
  def join(self, other):
    if isinstance(other, AbstractToken):
      return self
    else:
      assert False, f"Cannot join {self} with {other}"
  def str_short(self): return 'Tok'
  def at_least_vspace(self): return self

abstract_token: AbstractToken = AbstractToken()


def raise_to_shaped(aval: AbstractValue, weak_type=None):
  if weak_type is None:
    weak_type = getattr(aval, 'weak_type', False)
  for typ in type(aval).mro():
    handler = raise_to_shaped_mappings.get(typ)
    if handler: return handler(aval, weak_type)
  raise TypeError(type(aval))

raise_to_shaped_mappings : Dict[type, Callable] = {
  AbstractUnit: lambda aval, _: aval,
  AbstractToken: lambda aval, _: aval,
  Bot: lambda aval, _: aval,
  UnshapedArray: lambda aval, _: aval,
  ShapedArray: lambda aval, weak_type: ShapedArray(
      aval.shape, aval.dtype, weak_type, aval.named_shape)
}

### Operations on shapes and dimension sizes.

# Shapes are tuples of dimension sizes, which are normally integers. We allow
# modules to extend the set of dimension sizes to contain other types, e.g.,
# symbolic dimensions in jax2tf.shape_poly.DimVar and masking.Poly.
DimSize = Union[int, Any]  # extensible
Shape = Sequence[DimSize]


class InconclusiveDimensionOperation(Exception):
  """Raised when we cannot conclusively compute with symbolic dimensions."""
  pass

class DimensionHandler:
  """Operations on dimension sizes.

  Dimension sizes are normally integer constants, but can also be symbolic,
  e.g., masking.Poly or jax2tf.shape_poly.DimVar.

  The base class works for integers only. Subclasses are invoked when at
  least one of the operands has a type registered in _SPECIAL_DIMENSION_HANDLERS.
  In that case, all operands are guaranteed to be either the special dimension
  type, or Python integer scalars.

  Subclasses should raise InconclusiveDimensionOperation if the result cannot
  be computed in some contexts.
  """
  def is_constant(self, d: DimSize) -> bool:
    """The dimension is a constant."""
    return True

  def symbolic_equal(self, d1: DimSize, d2: DimSize) -> bool:
    """True iff the dimension sizes are equal in all contexts; False otherwise.
    Unlike `d1 == d2` this never raises InconclusiveDimensionOperation.
    """
    return d1 == d2

  def greater_equal(self, d1: DimSize, d2: DimSize) -> bool:
    """Computes `d1 >= d2`.
    Raise InconclusiveDimensionOperation if the result is different in
    different contexts.
    """
    return d1 >= d2

  def sum(self, *ds: DimSize) -> DimSize:
    """Sum of dimensions.
    Raises InconclusiveDimensionOperation if the result cannot be represented
    by the same DimSize in all contexts.
    """
    return sum(ds)

  def diff(self, d1: DimSize, d2: DimSize) -> DimSize:
    """Difference of dimensions.
    Raises InconclusiveDimensionOperation if the result cannot be represented
    by the same DimSize in all contexts.
    """
    return d1 - d2

  def divide_shape_sizes(self, s1: Shape, s2: Shape) -> DimSize:
    """Computes integer "i" such that i  * size(s2) == size(s1).

    Raise InconclusiveDimensionOperation if there is no such integer for all
    contexts,
    """
    sz1 = int(np.prod(s1))
    sz2 = int(np.prod(s2))
    if sz1 == 0 and sz2 == 0:
      return 1
    if sz1 % sz2:
      raise InconclusiveDimensionOperation(f"Cannot divide evenly the sizes of shapes {tuple(s1)} and {tuple(s2)}")
    return sz1 // sz2

  def stride(self, d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
    """(d - window_size) // window_stride + 1"""
    return (d - window_size) // window_stride + 1

  def dilate(self, d: DimSize, dilation: int) -> DimSize:
    """Implements `0 if d == 0 else 1 + dilation * (d - 1))`"""
    return 0 if d == 0 else 1 + dilation * (d - 1)


_dimension_handler_int = DimensionHandler()
_SPECIAL_DIMENSION_HANDLERS: Dict[type, DimensionHandler] = {}

def _dim_handler_and_canonical(*dlist: DimSize) -> Tuple[DimensionHandler, Tuple[DimSize, ...]]:
  """Finds the handler for the given dimensions; also returns the canonical dimensions.

  A dimension is canonical if it is a Python integer scalar, or has a type
  registered in _SPECIAL_DIMENSION_HANDLERS.
  """
  special_handlers = set()
  canonical = []
  for d in dlist:
    handler = _SPECIAL_DIMENSION_HANDLERS.get(type(d))
    if handler:
      special_handlers.add(handler)
      canonical.append(d)
    else:
      try:
        canonical.append(operator.index(d))
      except TypeError:
        raise _invalid_shape_error(dlist)

  if len(special_handlers) > 1:
    msg = (f"Dimension size operation involves multiple special dimension types {dlist}")
    raise ValueError(msg)
  return next(iter(special_handlers), _dimension_handler_int), tuple(canonical)

def is_constant_dim(d: DimSize) -> bool:
  handler, ds = _dim_handler_and_canonical(d)
  return handler.is_constant(*ds)

def symbolic_equal_dim(d1: DimSize, d2: DimSize) -> bool:
  handler, ds = _dim_handler_and_canonical(d1, d2)
  return handler.symbolic_equal(*ds)

def symbolic_equal_one_of_dim(d1: DimSize, dlist: Sequence[DimSize]) -> bool:
  handler, ds = _dim_handler_and_canonical(d1, *dlist)
  return any([handler.symbolic_equal(ds[0], d) for d in ds[1:]])

def symbolic_equal_shape(s1: Shape, s2: Shape) -> bool:
  return (len(s1) == len(s2) and
          all(map(symbolic_equal_dim, s1, s2)))

def greater_equal_dim(d1: DimSize, d2: DimSize) -> bool:
  handler, ds = _dim_handler_and_canonical(d1, d2)
  return handler.greater_equal(*ds)

def greater_equal_shape(s1: Shape, s2: Shape) -> bool:
  return all(map(greater_equal_dim, s1, s2))

def sum_dim(*ds: DimSize) -> DimSize:
  handler, ds = _dim_handler_and_canonical(*ds)
  return handler.sum(*ds)

def sum_shapes(*ss: Shape) -> Shape:
  return tuple(map(sum_dim, *ss))

def diff_dim(d1: DimSize, d2: DimSize) -> DimSize:
  handler, ds = _dim_handler_and_canonical(d1, d2)
  return handler.diff(*ds)

def diff_shape(s1: Shape, s2: Shape) -> Shape:
  return tuple(map(diff_dim, s1, s2))

def divide_shape_sizes(s1: Shape, s2: Shape) -> DimSize:
  """Returns an integer "i" s.t., i * size(s2) == size(s1).
  Raises if there is no such integer."""
  s1 = s1 or (1,)
  s2 = s2 or (1,)
  handler, ds = _dim_handler_and_canonical(*s1, *s2)
  return handler.divide_shape_sizes(ds[:len(s1)], ds[len(s1):])

def same_shape_sizes(s1: Shape, s2: Shape) -> bool:
  return 1 == divide_shape_sizes(s1, s2)

def is_empty_shape(s: Shape) -> bool:
  return any(symbolic_equal_dim(d, 0) for d in s)

def dilate_dim(d: DimSize, dilation: DimSize) -> DimSize:
  """Implements `0 if d == 0 else 1 + dilation * (d - 1))`"""
  handler, ds = _dim_handler_and_canonical(d, dilation)
  return handler.dilate(*ds)

def dilate_shape(s: Shape, dilations: Sequence[int]) -> Shape:
  return tuple(map(dilate_dim, s, dilations))

def stride_dim(d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
  handler, ds = _dim_handler_and_canonical(d, window_size, window_stride)
  return handler.stride(*ds)

def stride_shape(s: Shape, window_size: Shape, window_stride: Shape) -> Shape:
  """(s - window_size) // window_stride + 1"""
  return tuple(map(stride_dim, s, window_size, window_stride))


def _canonicalize_dimension(dim: DimSize) -> DimSize:
  if type(dim) in _SPECIAL_DIMENSION_HANDLERS:
    return dim
  else:
    return operator.index(dim)

def canonicalize_shape(shape: Shape) -> Shape:
  """Canonicalizes and checks for errors in a user-provided shape value.

  Args:
    shape: a Python value that represents a shape.

  Returns:
    A tuple of integers.
  """
  try:
    return tuple(map(_canonicalize_dimension, shape))
  except TypeError:
    pass
  raise _invalid_shape_error(shape)

def _invalid_shape_error(shape: Shape):
  msg = ("Shapes must be 1D sequences of concrete values of integer type, "
         "got {}.")
  if any(isinstance(x, Tracer) and isinstance(get_aval(x), ShapedArray)
         and not isinstance(get_aval(x), ConcreteArray) for x in shape):
    msg += ("\nIf using `jit`, try using `static_argnums` or applying `jit` to "
            "smaller subfunctions.")
  return TypeError(msg.format(shape))

# ------------------- Named shapes -------------------


class NamedShape:
  def __init__(self, *args, **kwargs):
    self.__positional = canonicalize_shape(args)
    # TODO: Assert that kwargs match axis env?
    self.__named = dict(kwargs)

  @property
  def rank(self):
    return len(self.__positional) + len(self.__named)

  @property
  def positional_rank(self):
    return len(self.__positional)

  @property
  def named_rank(self):
    return len(self.__named)

  @property
  def positional(self):
    return self.__positional

  @property
  def names(self):
    return self.__named.keys()

  @property
  def named_sizes(self):
    return self.__named.values()

  @property
  def named_items(self):
    return self.__named.items()

  def __getitem__(self, idx):
    try:
      idx = operator.index(idx)
      return self.__positional[idx]
    except TypeError:
      pass
    return self.__named[idx]

  @property
  def total(self):
    total = 1
    for s in self.__positional: total *= s
    for s in self.__named.values(): total *= s
    return total

  def __str__(self):
    return (f"({', '.join(map(str, self.__positional))}{', ' if self.__named else ''}"
            f"{', '.join(f'{k}={v}' for k, v in self.__named.items())})")

  def __eq__(self, other):
    if isinstance(other, NamedShape):
      return (self.__positional, self.__named) == (other.__positional, other.__named)
    if isinstance(other, tuple):
      return not self.__named and self.__positional == other
    raise TypeError(f"NamedShape doesn't support comparisons with {type(other)}")

  def __hash__(self):
    return hash((self.__positional, tuple(self.__named.items())))

def join_named_shapes(*named_shapes):
  result = {}
  for named_shape in named_shapes:
    for name, size in named_shape.items():
      if result.setdefault(name, size) != size:
        raise TypeError(
            f"Axis name {name} used with inconsistent sizes: {result[name]} != {size}")
  return result

# TODO: Make canonicalize_shape return named shapes?
def as_named_shape(shape) -> NamedShape:
  if isinstance(shape, NamedShape):
    return shape
  return NamedShape(*shape)


# ------------------- Call -------------------

def apply_todos(todos, outs):
  todos_list = list(todos)
  while todos_list:
    outs = map(full_lower, todos_list.pop()(outs))
  return outs

class CallPrimitive(Primitive):
  multiple_results = True
  call_primitive = True

  def bind(self, fun, *args, **params):
    return final_style_bind(attrgetter('process_call'), self, fun, args, params)

  def impl(self, f, *args, **params):
    del params  # params parameterize the call primitive, not the function
    with new_sublevel():
      return f.call_wrapped(*args)

def final_style_bind(process_method, prim, fun, args, params):
  base, *mains = thread_local_state.trace_state.trace_stack.active_stack
  callbacks = []
  for m in reversed(mains):
    trace = m.with_cur_sublevel()
    fun, args, params, callback = process_method(trace)(
        prim, fun, map(trace.full_raise, args), params)
    callbacks.append(callback)
  trace = base.with_cur_sublevel()
  outs = process_method(trace)(prim, fun, map(trace.full_raise, args), params)
  for c in reversed(callbacks):
    outs = c(outs)
  return map(full_lower, outs)

call_p = CallPrimitive('call')
call = call_p.bind

named_call_p = CallPrimitive('named_call')

# ------------------- Map -------------------

def mapped_aval(size: int, axis: Optional[int], aval: AbstractValue) -> AbstractValue:
  if axis is None: return aval
  handler, _ = aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis, aval)
  else:
    raise TypeError(f"no mapping handler for {aval} of type {type(aval)}")

def unmapped_aval(size: int, axis: Optional[int], aval: AbstractValue) -> AbstractValue:
  if axis is None: return aval
  _, handler = aval_mapping_handlers.get(type(aval), (None, None))
  if handler is not None:
    return handler(size, axis, aval)
  else:
    raise TypeError(f"no unmapping handler for {aval} of type {type(aval)}")

def _map_unit(size: int, axis: int, aval: AbstractUnit) -> AbstractUnit:
  return aval

def _map_shaped_array(size: int, axis: int, aval: ShapedArray) -> ShapedArray:
  assert aval.shape[axis] == size
  return ShapedArray(tuple_delete(aval.shape, axis), aval.dtype)

def _unmap_shaped_array(size: int, axis: int, aval: ShapedArray) -> ShapedArray:
  return ShapedArray(tuple_insert(aval.shape, axis, size), aval.dtype)

AvalMapHandlerPair = Tuple[Callable, Callable]
aval_mapping_handlers: Dict[Type, AvalMapHandlerPair] = {
    AbstractUnit: (_map_unit, _map_unit),
    ShapedArray:   (_map_shaped_array, _unmap_shaped_array),
    ConcreteArray: (_map_shaped_array, _unmap_shaped_array),
}


class _IgnoreElemList(list):
  """Compares equal to all other _ignore_elem_lists."""
  def __hash__(self): return 0
  def __eq__(self, other):
    return type(other) is _IgnoreElemList

class MapPrimitive(Primitive):
  multiple_results = True
  map_primitive = True

  def bind(self, fun, *args, **params):
    assert len(params['in_axes']) == len(args)
    out_axes_transforms = _IgnoreElemList()
    out_axes_thunk = params['out_axes_thunk']
    # The new thunk depends deterministically on the old thunk and the wrapped
    # function. Any caching already has to include the wrapped function as part
    # of the key, so we only use the previous thunk for equality checks.
    @as_hashable_function(closure=out_axes_thunk)
    def new_out_axes_thunk():
      out_axes = out_axes_thunk()
      for t in out_axes_transforms:
        out_axes = t(out_axes)
      return out_axes
    params = dict(params, out_axes_thunk=new_out_axes_thunk)
    return final_style_bind(attrgetter('process_map'), self, fun, args, params)

@contextmanager
def extend_axis_env(axis_name: AxisName, size: int, tag: Any):
  frame = AxisEnvFrame(axis_name, size, tag)
  thread_local_state.trace_state.axis_env.append(frame)
  try:
    yield
  finally:
    thread_local_state.trace_state.axis_env.pop()

@contextmanager
def extend_axis_env_nd(axes: Iterable[Tuple[AxisName, int]]):
  frames = [AxisEnvFrame(axis_name, size, None) for axis_name, size in axes]
  thread_local_state.trace_state.axis_env.extend(frames)
  try:
    yield
  finally:
    for _ in frames:
      thread_local_state.trace_state.axis_env.pop()


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


def axis_frame(axis_name):
  frames = thread_local_state.trace_state.axis_env
  for frame in reversed(frames):
    if frame.name == axis_name:
      return frame
  named_axes = [frame.name for frame in reversed(frames)
                if not isinstance(frame.name, _TempAxisName)]
  raise NameError(
      f'unbound axis name: {axis_name}. The following axis names (e.g. defined '
      f'by pmap) are available to collective operations: {named_axes}')


ParamDict = Dict[str, Any]
AxisSubst = Callable[[AxisName], Tuple[AxisName, ...]]

def used_axis_names(primitive: Primitive, params: ParamDict) -> Set[AxisName]:
  axis_names = set()
  def register_name(axis_name):
    axis_names.add(axis_name)
    return (axis_name,)
  subst_axis_names(primitive, params, register_name)
  return axis_names

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

def subst_axis_names_var(v: Var, subst: AxisSubst, var_map: Dict[Var, Var]) -> Var:
  # Var identity is load-bearing, so we can't have duplicates!
  if v is unitvar: return v
  if v is dropvar: return v
  assert v not in var_map
  if not hasattr(v.aval, 'named_shape'):
    var_map[v] = v
    return v
  names = tuple(it.chain.from_iterable(subst(name) for name in v.aval.named_shape))
  named_shape = {name: axis_frame(name).size for name in names}
  if len(named_shape) != len(names):
    raise DuplicateAxisNameError(v)
  new_v = Var(v.count, v.suffix, v.aval.update(named_shape=named_shape))
  var_map[v] = new_v
  return new_v

def subst_axis_names_eqn(eqn: JaxprEqn, subst: AxisSubst, var_map: Dict[Var, Var]) -> JaxprEqn:
  invars: List[Atom] = [v if isinstance(v, Literal) else var_map[v] for v in eqn.invars]
  try:
    outvars = [subst_axis_names_var(v, subst, var_map) for v in eqn.outvars]
  except DuplicateAxisNameError as e:
    e.eqn = eqn
    raise
  params = subst_axis_names(eqn.primitive, eqn.params, subst)
  return new_jaxpr_eqn(invars, outvars, eqn.primitive, params, eqn.source_info)

def subst_axis_names_jaxpr(jaxpr: Union[Jaxpr, ClosedJaxpr], subst: AxisSubst):
  consts = None
  if isinstance(jaxpr, ClosedJaxpr):
    consts = jaxpr.consts
    jaxpr = jaxpr.jaxpr
  var_map: Dict[Var, Var] = {}
  invars = [subst_axis_names_var(v, subst, var_map) for v in jaxpr.invars]
  constvars = [subst_axis_names_var(v, subst, var_map) for v in jaxpr.constvars]
  eqns = [subst_axis_names_eqn(eqn, subst, var_map) for eqn in jaxpr.eqns]
  outvars: List[Atom] = [v if isinstance(v, Literal) else var_map[v] for v in jaxpr.outvars]
  new_jaxpr = Jaxpr(constvars, invars, outvars, eqns)
  if consts is not None:
    return ClosedJaxpr(new_jaxpr, consts)
  return new_jaxpr

axis_substitution_rules: Dict[Primitive, Callable[[ParamDict, AxisSubst, bool], ParamDict]] = {}

# ------------------- AxisPrimitive -------------------
# Primitives that store axis names in params and want those axis names to
# participate in dispatch should subclass AxisPrimitive.

class AxisPrimitive(Primitive):
  _dispatch_on_params = True

# ------------------- Jaxpr checking -------------------

def typecheck(aval: AbstractValue, x) -> bool:
  return typecompat(aval, get_aval(x))

def typecompat(aval_ref: AbstractValue, aval: AbstractValue) -> bool:
  """Determine whether `aval` conforms to `aval_ref`.

  Ignores weak_type and named_shape, other than to check that an axis name isn't
  used with different sizes.
  """
  try:
    return typematch(aval_ref, lattice_join(aval_ref, aval))
  except TypeError:
    return False

def typematch(aval1: AbstractValue, aval2: AbstractValue) -> bool:
  """Determine whether `aval1` and `aval2` are equivalent.

  Ignores weak_type and named_shape, other than to check that an axis name isn't
  used with different sizes.
  """
  if aval1 == aval2: return True
  # unequal avals may still represent the same type, because type is represented
  # by avals at the shaped level, and because weak type tags and (for now) named
  # shape components aren't considered part of the type
  if isinstance(aval1, ShapedArray) and isinstance(aval2, ShapedArray):
    # a bonus check for whether any named axes have inconsistent sizes
    join_named_shapes(aval1.named_shape, aval2.named_shape)
  return (raise_to_shaped(aval1, weak_type=False).strip_named_shape() ==
          raise_to_shaped(aval2, weak_type=False).strip_named_shape())

class JaxprTypeError(TypeError): pass

def typecheck_assert(pred, msg):
  if not pred:
    raise JaxprTypeError(msg)

custom_typechecks: Dict[Primitive, Callable] = {}

def check_jaxpr(jaxpr: Jaxpr):
  """Checks well-formedness of a jaxpr.

  Specifically, check that:
  - variables that are read are bound beforehand
  - variables are typed equally throughout a jaxpr
  - variable type annotations are compatible with their binding expression

  Raises `JaxprTypeError` if `jaxpr` is determined invalid. Returns `None`
  otherwise.
  """
  try:
    _check_jaxpr(jaxpr, [v.aval for v in jaxpr.invars])
  except JaxprTypeError as e:
    if len(e.args) == 2:
      msg, eqn_idx = e.args
      jaxpr_str = str(pp_jaxpr_eqn_range(jaxpr, eqn_idx - 10, eqn_idx + 10))
    else:
      msg, = e.args
      jaxpr_str = str(pp_jaxpr_eqn_range(jaxpr, 0, 20))
    msg = "\n\n".join([msg, "while checking jaxpr:", jaxpr_str])
    raise JaxprTypeError(msg) from None

def _check_jaxpr(jaxpr: Jaxpr, in_avals: Sequence[AbstractValue]):

  def read(v: Atom) -> AbstractValue:
    if isinstance(v, Literal):
      return raise_to_shaped(get_aval(v.val))
    else:
      typecheck_assert(v in env, f"Variable '{v}' not defined")
      return env[v]

  def write(v: Var, a: AbstractValue) -> None:
    typecheck_assert(v not in env, f"Variable '{v}' already bound")
    if v is not dropvar:
      typecheck_assert(typecompat(v.aval, a),
                       f"Variable '{v}' inconsistently typed as {a}, "
                       f"bound as {v.aval}")
      env[v] = a

  env : Dict[Var, AbstractValue] = {}

  write(unitvar, abstract_unit)
  map(write, jaxpr.constvars, [v.aval for v in jaxpr.constvars])
  map(write, jaxpr.invars, in_avals)

  for eqn_idx, eqn in enumerate(jaxpr.eqns):
    prim = eqn.primitive
    try:
      in_avals = map(read, eqn.invars)
      typecheck_assert(all(not isinstance(ina, ConcreteArray) for ina in in_avals),
                       "Equation given ConcreteArray type inputs")
      if prim in custom_typechecks:
        out_avals = custom_typechecks[prim](*in_avals, **eqn.params)
        if out_avals is None:
          out_avals = [v.aval for v in eqn.outvars]
      elif prim.call_primitive:
        out_avals = check_call(prim, in_avals, eqn.params)
      elif prim.map_primitive:
        out_avals = check_map(prim, in_avals, eqn.params)
      else:
        out_avals = check_eqn(prim, in_avals, eqn.params)
      map(write, eqn.outvars, out_avals)
    except JaxprTypeError as e:
      msg, = e.args
      src = source_info_util.summarize(eqn.source_info)
      msg = "\n\n".join([msg, "in equation:", str(pp_eqn(eqn).indent(2)),
                         f"from source: {src}"])
      raise JaxprTypeError(msg, eqn_idx) from None

  map(read, jaxpr.outvars)

def check_eqn(prim, in_avals, params):
  for jaxpr in jaxprs_in_params(params):
    check_jaxpr(jaxpr)

  out_avals = prim.abstract_eval(*in_avals, **params)
  if not prim.multiple_results:
    out_avals = [out_avals]
  return out_avals

def check_call(prim, in_avals, params):
  typecheck_assert("call_jaxpr" in params,
                   f"Call primitive {prim} missing 'call_jaxpr' parameter")
  call_jaxpr = params["call_jaxpr"]

  # These checks also happen in recursive call, but give better errors here.
  typecheck_assert(len(in_avals) == len(call_jaxpr.invars),
                   f"Call primitive {prim} with {len(call_jaxpr.invars)} "
                   f"operands cannot call jaxpr with {len(call_jaxpr.invars)} "
                   f"inputs")
  binder_avals = [v.aval for v in call_jaxpr.invars]
  for binder_aval, in_aval in zip(binder_avals, in_avals):
    typecheck_assert(typecompat(binder_aval, in_aval),
                     f"Call primitive {prim} passes operand {in_aval} "
                     f"to jaxpr expecting {binder_aval}")

  _check_jaxpr(call_jaxpr, in_avals)

  out_avals = [v.aval for v in call_jaxpr.outvars]
  return out_avals

def check_map(prim, in_avals, params):
  typecheck_assert("call_jaxpr" in params,
                   f"Map primitive {prim} missing 'call_jaxpr' parameter")
  call_jaxpr = params["call_jaxpr"]
  typecheck_assert("axis_size" in params,
                   f"Map primitive {prim} missing 'axis_size' parameter")
  axis_size = params["axis_size"]
  typecheck_assert("in_axes" in params,
                   f"Map primitive {prim} missing 'in_axes' parameter")
  in_axes = params["in_axes"]
  typecheck_assert("out_axes" in params,
                   f"Map primitive {prim} missing 'out_axes' parameter")
  out_axes = params["out_axes"]

  binder_avals = [unmapped_aval(axis_size, in_axis, v.aval)
                  if in_axis is not None else v.aval
                  for v, in_axis in zip(call_jaxpr.invars, in_axes)]
  for binder_aval, in_aval in zip(binder_avals, in_avals):
    typecheck_assert(typecompat(binder_aval, in_aval),
                     f"Call primitive {prim} passes operand {in_aval} "
                     f"to jaxpr expecting {binder_aval}")

  mapped_avals = [mapped_aval(axis_size, in_axis, aval)
                  if in_axis is not None else aval
                  for aval, in_axis in zip(in_avals, in_axes)]
  with extend_axis_env(params['axis_name'], axis_size, None):
    _check_jaxpr(call_jaxpr, mapped_avals)

  mapped_out_avals = [v.aval for v in call_jaxpr.outvars]
  out_avals = [unmapped_aval(axis_size, out_axis, aval) if out_axis is not None else aval
               for aval, out_axis in zip(mapped_out_avals, out_axes)]
  return out_avals


# ------------------- Jaxpr printed representation -------------------

def pp_vars(vs: Sequence[Any], print_shapes: bool = False) -> str:
  if print_shapes:
    return ' '.join(f'{v}:{v.aval.str_short()}' for v in vs)
  else:
    return ' '.join(map(str, vs))

def pp_eqn_compact(primitive_name: str, params: Dict) -> PrettyPrint:
  filtered_params = {k: v for k, v in params.items()
                     if (k != 'branches' and
                         not isinstance(v, (Jaxpr, ClosedJaxpr)))}
  return pp(primitive_name) >> pp_kv_pairs(sorted(filtered_params.items()))

def pp_eqn(eqn: JaxprEqn, print_shapes: bool = False) -> PrettyPrint:
  lhs = pp_vars(eqn.outvars, print_shapes)
  pp_lhs = pp(f'{lhs} =')
  pp_rhs = (pp(eqn.primitive.name) >>
            pp_kv_pairs(sorted(eqn.params.items())) >> pp(' ') >>
            pp(pp_vars(eqn.invars, print_shapes)))
  if len(lhs) <= 6 or print_shapes:
    return pp_lhs >> pp(' ') >> pp_rhs
  else:
    return pp_lhs + pp_rhs.indent(2)

def pp_eqns(eqns: Sequence[JaxprEqn],
            source_info: bool = False) -> Sequence[PrettyPrint]:
  pps = map(pp_eqn, eqns)
  if source_info:
    l = max((i + len(s) for x in pps for i, s in x.lines), default=None)
    if l is not None:
      return [p.annotate(l, source_info_util.summarize(e.source_info))
              for e, p in zip(eqns, pps)]
  return pps

def pp_jaxpr(jaxpr: Jaxpr, source_info: bool = False) -> PrettyPrint:
  pps = pp_eqns(jaxpr.eqns, source_info=source_info)
  str_outvars = str(tuple(jaxpr.outvars))
  return (pp('{{ lambda {} ; {}.'.format(pp_vars(jaxpr.constvars),
                                         pp_vars(jaxpr.invars))) +
          ((pp('let ') >> vcat(pps))
           + pp('in {} }}'.format(str_outvars))).indent(2))

def pp_jaxpr_eqn_range(jaxpr: Jaxpr, lo: int, hi: int,
                       source_info: bool = False) -> PrettyPrint:
  lo = max(lo, 0)
  hi = max(lo, min(hi, len(jaxpr.eqns)))
  eqns = jaxpr.eqns[lo:hi]
  pps = []
  if len(eqns) == 0 and len(jaxpr.eqns) != 0:
    pps.append(pp('...'))
  else:
    if lo != 0:
      pps.append(pp('...'))
    pps.extend(pp_eqns(eqns, source_info=source_info))
    if hi != len(jaxpr.eqns):
      pps.append(pp('...'))
  str_outvars = str(tuple(jaxpr.outvars))
  return (pp('{{ lambda {} ; {}.'.format(pp_vars(jaxpr.constvars),
                                         pp_vars(jaxpr.invars))) +
          ((pp('let ') >> vcat(pps))
           + pp('in {} }}'.format(str_outvars))).indent(2))

def pp_jaxprs(jaxprs) -> PrettyPrint:
  jaxprs = [j.jaxpr if isinstance(j, ClosedJaxpr) else j for j in jaxprs]
  return pp('( ') >> vcat(map(pp_jaxpr, jaxprs)) >> pp(' )')

def pp_kv_pair(k, v):
  if type(v) is tuple and all(isinstance(j, (Jaxpr, ClosedJaxpr)) for j in v):
    pp_v = pp_jaxprs(v)
  else:
    pp_v = pp(v)
  return pp(f'{k}=') >> pp_v

def pp_kv_pairs(kv_pairs):
  if kv_pairs:
    return pp('[ ') >> vcat([pp_kv_pair(k, v) for k, v in kv_pairs]) >> pp(' ]')
  else:
    return pp('')

# Casting float0 array to a float-valued zero array.
def zeros_like_float0(array, dtype=None):
  if not dtype:
    dtype = np.float
  return np.zeros(array.shape, dtype)
