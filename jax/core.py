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
from contextlib import contextmanager, suppress
from collections import namedtuple
from functools import total_ordering
import itertools as it
from weakref import ref
import threading
import types
from typing import (Any, Callable, ClassVar, Dict, Generator,
                    Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple,
                    Type, Union, cast, no_type_check)

import numpy as np

from . import dtypes
from .config import FLAGS, config
from . import linear_util as lu
from . import source_info_util

from .util import safe_zip, safe_map, partial, curry, prod, partialmethod
from .pprint_util import pp, vcat, PrettyPrint

# TODO(dougalm): compilation cache breaks the leak detector. Consisder solving.
check_leaks = False

# Disables internal invariant checks
skip_checks = not FLAGS.jax_enable_checks  # not __debug__  # google doesn't use -O

@contextmanager
def skipping_checks():
  """Context manager for temporarily disabling checks."""
  global skip_checks
  old_value, skip_checks = skip_checks, True
  try:
    yield
  finally:
    skip_checks = old_value

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
    Params:
      constvars: list of variables introduced for constants (either literals
        in the Python program, or the result of constant folding during the
        generation of the Jaxpr). Array constants are replaced with such variables
        while scalar constants are kept inline.
      invars: list of input variables. Together, `constvars` and `invars` are
        the inputs to the Jaxpr.
      outvars: list of output variables.
      eqns: list of equations."""
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
      elif isinstance(v, TypedJaxpr):
        yield v.jaxpr


def subjaxprs(jaxpr: Jaxpr) -> Iterator[Jaxpr]:
  """Generator for all subjaxprs found in the params of jaxpr.eqns.
  Does not descend recursively into the found subjaxprs.
  """
  for eqn in jaxpr.eqns:
    yield from jaxprs_in_params(eqn.params)


class TypedJaxpr:
  jaxpr: Jaxpr
  literals: List['Any']
  in_avals: List['AbstractValue']
  out_avals: List['AbstractValue']

  def __init__(self, jaxpr: Jaxpr, literals: Sequence,
               in_avals: Sequence['AbstractValue'],
               out_avals: Sequence['AbstractValue']):
    assert len(literals) == len(jaxpr.constvars)
    assert len(in_avals) == len(jaxpr.invars)

    if not skip_checks:
      in_avals_raised = [raise_to_shaped(v) for v in in_avals]
      out_avals_raised = [raise_to_shaped(v) for v in out_avals]
      exp_in_avals = [v.aval for v in jaxpr.invars]
      exp_out_avals = [v.aval for v in jaxpr.outvars]
      assert in_avals_raised == exp_in_avals, "expected: {}, got: {}".format(exp_in_avals, in_avals_raised)
      assert out_avals_raised == exp_out_avals, "expected: {}, got: {}".format(exp_out_avals, out_avals_raised)

    self.jaxpr = jaxpr
    self.literals = list(literals)
    self.in_avals = list(in_avals)
    self.out_avals = list(out_avals)

  def __iter__(self):
    return iter((self.jaxpr, self.literals, self.in_avals, self.out_avals))

  def __str__(self):
    # TODO(mattjj): improve this with type annotations?
    return str(pp_jaxpr(self.jaxpr))
  __repr__ = __str__

@curry
def jaxpr_as_fun(typed_jaxpr: TypedJaxpr, *args):
  return eval_jaxpr(typed_jaxpr.jaxpr, typed_jaxpr.literals, *args)


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

  def __eq__(self, other):
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

  def __init__(self, name: str):
    self.name = name

  def __repr__(self):
    return '{}'.format(self.name)

  def bind(self, *args, **kwargs):
    assert skip_checks or all(isinstance(arg, Tracer)
                              or valid_jaxtype(arg) for arg in args), args
    top_trace = find_top_trace(args)
    if top_trace is None:
      return self.impl(*args, **kwargs)

    tracers = map(top_trace.full_raise, args)
    out_tracer = top_trace.process_primitive(self, tracers, kwargs)
    if self.multiple_results:
      return map(full_lower, out_tracer)
    else:
      return full_lower(out_tracer)

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
    with source_info_util.user_context(eqn.source_info):
      ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  return map(read, jaxpr.outvars)


# -------------------- tracing --------------------


class Trace:
  __slots__ = ['master', 'level', 'sublevel']

  master: 'MasterTrace'
  level: int
  sublevel: 'Sublevel'

  def __init__(self, master: 'MasterTrace', sublevel: 'Sublevel') -> None:
    self.master = master
    self.level = master.level
    self.sublevel = sublevel

  def full_raise(self, val) -> 'Tracer':
    if not isinstance(val, Tracer):
      return self.pure(val)
    level = self.level
    sublevel = self.sublevel
    if val._trace.master is self.master:
      if val._trace.sublevel == sublevel:
        return val
      elif val._trace.sublevel < sublevel:
        return self.sublift(val)
      else:
        raise escaped_tracer_error("Can't lift sublevels {} to {}"
                                   .format(val._trace.sublevel, sublevel))
    elif val._trace.level < level:
      if val._trace.sublevel > sublevel:
        raise escaped_tracer_error("Incompatible sublevel: {}, {}"
                                   .format(val._trace, (level, sublevel)))
      return self.lift(val)
    elif val._trace.level > level:
      raise escaped_tracer_error("Can't lift level {} to {}"
                                 .format(val, self))
    else:  # val._trace.level == self.level:
      raise escaped_tracer_error("Different traces at same level: {}, {}"
                                 .format(val, self))

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
    raise NotImplementedError("must override to handle call-like primitives")

  def process_map(self, call_primitive, f, tracers, params):
    raise NotImplementedError("must override to handle map-like primitives")

  def process_custom_jvp_call(self, primitive, fun, jvp, tracers):
    # As a default implementation, drop the custom differentiation rule. This
    # behavior is desirable when staging out of the JAX system, but not when
    # there are further differentiation transformations to be applied. Override
    # this method to allow differentiation to be performed downstream.
    del primitive, jvp  # Unused.
    return fun.call_wrapped(*tracers)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
    # See comment in the above process_custom_jvp_call method.
    del primitive, fwd, bwd, out_trees  # Unused.
    return fun.call_wrapped(*tracers)

def escaped_tracer_error(detail):
  msg = ("Encountered an unexpected tracer. Perhaps this tracer escaped "
         "through global state from a previously traced function.\n"
         "The functions being transformed should not save traced values to "
         "global state.\nDetails: {}.")
  return UnexpectedTracerError(msg.format(detail))

class UnexpectedTracerError(Exception): pass

class Tracer:
  __array_priority__ = 1000
  __slots__ = ['_trace', '__weakref__']

  def __array__(self, *args, **kw):
    msg = ("The numpy.ndarray conversion method __array__() was called on "
           f"the JAX Tracer object {self}.\n\n"
           "This error can occur when a JAX Tracer object is passed to a raw "
           "numpy function, or a method on a numpy.ndarray object. You might "
           "want to check that you are using `jnp` together with "
           "`import jax.numpy as jnp` rather than using `np` via "
           "`import numpy as np`. If this error arises on a line that involves "
           "array indexing, like `x[idx]`, it may be that the array being "
           "indexed `x` is a raw numpy.ndarray while the indices `idx` are a "
           "JAX Tracer instance; in that case, you can instead write "
           "`jax.device_put(x)[idx]`.")
    raise Exception(msg)

  def __init__(self, trace: Trace):
    self._trace = trace

  def __iter__(self):
    return iter(self.aval._iter(self))

  def __len__(self):
    return self.aval._len(self)

  @property
  def aval(self):
    raise NotImplementedError("must override")

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

  def __float__(self):
    raise TypeError("JAX Tracer object cannot be interpreted as a float. "
                    "Try using `x.astype(float)` instead.")

  def __complex__(self):
    raise TypeError("JAX Tracer object cannot be interpreted as a complex. "
                    "Try using `x.astype(complex)` instead.")

  def __setitem__(self, idx, val):
    raise TypeError("JAX 'Tracer' objects do not support item assignment")

  # NumPy also only looks up special methods on classes.
  def __array_module__(self, types): return self.aval._array_module(self, types)

  def __getattr__(self, name):
    # if the aval property raises an AttributeError, gets caught here
    assert skip_checks or name != "aval"

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

# these can be used to set up forwarding of properties and instance methods from
# Tracer instances to the underlying avals
aval_property = namedtuple("aval_property", ["fget"])
aval_method = namedtuple("aval_method", ["fun"])


class EvalTrace(Trace):
  def pure(self, x): return x
  lift = sublift = pure

  def process_primitive(self, primitive, tracers, params):
    return primitive.impl(*tracers, **params)

  def process_call(self, primitive, f, tracers, params):
    return primitive.impl(f, *tracers, **params)
  process_map = process_call


class MasterTrace:
  level: int
  trace_type: Type[Trace]

  def __init__(self, level, trace_type) -> None:
    self.level = level
    self.trace_type = trace_type

  def __repr__(self) -> str:
    return "MasterTrace({},{})".format(self.level, self.trace_type.__name__)

  def __hash__(self) -> int:
    return hash((self.level, self.trace_type))

  def __eq__(self, other: object) -> bool:
    return (isinstance(other, MasterTrace) and
            self.level == other.level and self.trace_type == other.trace_type)

class TraceStack:
  upward: List[MasterTrace]
  downward: List[MasterTrace]

  def __init__(self):
    self.upward = []
    self.downward = []

  def next_level(self, bottom: bool) -> int:
    if bottom:
      return - (len(self.downward) + 1)
    else:
      return len(self.upward)

  def push(self, master_trace: MasterTrace, bottom: bool) -> None:
    if bottom:
      self.downward.append(master_trace)
    else:
      self.upward.append(master_trace)

  def pop(self, bottom: bool) -> None:
    if bottom:
      self.downward.pop()
    else:
      self.upward.pop()

  def __repr__(self) -> str:
    return  'Trace stack\n{} ---\n{}'.format(
      map('  {}\n'.format, self.upward[::-1]),
      map('  {}\n'.format, self.downward))

  def copy(self):
    new = TraceStack()
    new.upward = self.upward[:]
    new.downward = self.downward[:]
    return new

class Sublevel(int): pass
AxisEnvFrame = namedtuple('AxisEnvFrame', ['name', 'size', 'tag'])


class TraceState:
  trace_stack: TraceStack
  substack: List[Sublevel]
  initial_style: bool

  def __init__(self) -> None:
    self.trace_stack = TraceStack()
    self.substack = [Sublevel(0)]
    self.initial_style = False

  def copy(self):
    new = TraceState()
    new.trace_stack = self.trace_stack.copy()
    new.substack = self.substack[:]
    new.initial_style = self.initial_style
    return new

# The global state of the tracer is accessed by a thread-local object.
# This allows concurrent tracing in separate threads; passing traced objects
# between threads is forbidden.
class ThreadLocalState(threading.local):
  def __init__(self):
    self.trace_state = TraceState()
thread_local_state = ThreadLocalState()

def reset_trace_state() -> bool:
  "Reset the global trace state and return True if it was already clean."
  if (thread_local_state.trace_state.substack != [Sublevel(0)] or
      thread_local_state.trace_state.trace_stack.downward or
      thread_local_state.trace_state.trace_stack.upward):
    thread_local_state.trace_state.__init__()  # type: ignore
    return False
  else:
    return True

def cur_sublevel() -> Sublevel:
  return thread_local_state.trace_state.substack[-1]

@contextmanager
def new_master(trace_type: Type[Trace], bottom=False) -> Generator[MasterTrace, None, None]:
  level = thread_local_state.trace_state.trace_stack.next_level(bottom)
  master = MasterTrace(level, trace_type)
  thread_local_state.trace_state.trace_stack.push(master, bottom)

  try:
    yield master
  finally:
    thread_local_state.trace_state.trace_stack.pop(bottom)

  if check_leaks:
    t = ref(master)
    del master
    if t() is not None:
      print(thread_local_state.trace_state.trace_stack)
      raise Exception('Leaked trace {}'.format(t()))

@contextmanager
def new_sublevel() -> Generator[None, None, None]:
  sublevel = Sublevel(len(thread_local_state.trace_state.substack))
  thread_local_state.trace_state.substack.append(sublevel)
  try:
    yield
  finally:
    thread_local_state.trace_state.substack.pop()

  if check_leaks:
    t = ref(sublevel)
    del sublevel
    if t() is not None:
      raise Exception('Leaked sublevel {}'.format(t()))

def full_lower(val):
  if isinstance(val, Tracer):
    return val.full_lower()
  else:
    return val

def find_top_trace(xs) -> Optional[Trace]:
  top_trace = max((x._trace for x in xs if isinstance(x, Tracer)),
                  key=attrgetter('level'), default=None)
  return top_trace and type(top_trace)(top_trace.master, cur_sublevel())

@contextmanager
def initial_style_staging():
  trace_state = thread_local_state.trace_state
  prev, trace_state.initial_style = trace_state.initial_style, True
  try:
    yield
  finally:
    trace_state.initial_style = prev

@contextmanager
def eval_context():
  yield  # dummy implementation for forward compatibility


# -------------------- abstract values --------------------


class AbstractValue:
  __slots__: List[str] = []

  def at_least_vspace(self):
    assert False

  def __repr__(self):
    try:
      kv_pairs = ('{}={}'.format(k, v) for k, v in self.__dict__.items())
      return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
    except AttributeError:
      return self.__class__.__name__

  def strip_weak_type(self) -> 'AbstractValue':
    return self

  def join(self, other):
    raise NotImplementedError("must override")

class Bot(AbstractValue): pass

bot = Bot()

class AbstractUnit(AbstractValue):
  def join(self, other):
    if not skip_checks:
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
    raise TypeError((x, y))

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
    raise TypeError(f"{x} of type {type(x)} is not a valid JAX type")


def concrete_aval(x):
  for typ in type(x).mro():
    handler = pytype_aval_mappings.get(typ)
    if handler: return handler(x)
  raise TypeError(f"{type(x)} is not a valid JAX type")


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

class ConcretizationTypeError(TypeError): pass

def raise_concretization_error(val, context=""):
  msg = (f"Abstract tracer value encountered where concrete value is expected ({context}).\n"
          "Use transformation parameters such as `static_argnums` for `jit` "
          "to avoid tracing input values.\n"
          "See `https://jax.readthedocs.io/en/latest/faq.html#abstract-tracer-value-encountered-where-concrete-value-is-expected-error`.\n"
          f"Encountered value: {val}")
  raise ConcretizationTypeError(msg)


def concretization_function_error(fun, context=""):
  fname = getattr(fun, "__name__", fun)
  fname_context = f"in `{fname}`"
  if context:
    fname_context += f" {context}"
  def error(self, arg):
    raise_concretization_error(arg, fname_context)
  return error


def concrete_or_error(force: Any, val: Any, context=""):
  """Like force(val), but gives the context in the error message."""
  if isinstance(val, Tracer):
    if isinstance(val.aval, ConcreteArray):
      return force(val.aval.val)
    else:
      raise_concretization_error(val, context)
  else:
    return force(val)

class UnshapedArray(AbstractValue):
  __slots__ = ['dtype', 'weak_type']
  array_abstraction_level = 2

  def __init__(self, dtype, weak_type=False):
    self.dtype = np.dtype(dtypes.canonicalize_dtype(dtype))
    self.weak_type = weak_type

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
  _float   = concretization_function_error(
      float, "Try using `x.astype(float)` instead.")
  _int     = concretization_function_error(
      int, "Try using `x.astype(int)` instead.")
  _complex = concretization_function_error(
      complex, "Try using `x.astype(complex)` instead.")
  _hex     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)

  def at_least_vspace(self) -> AbstractValue:
    return self

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

  def strip_weak_type(self) -> 'UnshapedArray':
    """Returns a copy of the aval with weak_type=False."""
    return UnshapedArray(self.dtype) if self.weak_type else self

  @property
  def shape(self):
    msg = ("UnshapedArray has no shape. Please open an issue at "
           "https://github.com/google/jax/issues because it's unexpected for "
           "UnshapedArray instances to ever be produced.")
    raise TypeError(msg)

class ShapedArray(UnshapedArray):
  __slots__ = ['shape']
  array_abstraction_level = 1

  def __init__(self, shape, dtype, weak_type=False):
    super(ShapedArray, self).__init__(dtype, weak_type=weak_type)
    self.shape = canonicalize_shape(shape)

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: prod(self.shape))

  broadcast: ClassVar[Optional[aval_method]] = None
  transpose: ClassVar[Optional[aval_method]] = None
  reshape: ClassVar[Optional[aval_method]] = None
  _iter: ClassVar[Optional[staticmethod]] = None

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
    return self

  def join(self, other):
    if self.shape == other.shape and self.dtype == other.dtype:
      if self.weak_type == other.weak_type:
        return self
      else:
        return ShapedArray(self.shape, self.dtype, weak_type=False)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype)
    else:
      raise TypeError(self, other)

  def str_short(self):
    shapestr = ','.join(map(str, self.shape))
    return '{}[{}]'.format(self.dtype.name, shapestr)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError:
      raise TypeError("len() of unsized object")  # same as numpy error

  def _len(self, ignored_tracer):
    return len(self)

  def strip_weak_type(self):
    return ShapedArray(self.shape, self.dtype) if self.weak_type else self


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
    assert self.dtype != np.dtype('O')

  def __eq__(self, other):
    return (type(self) is type(other) and self.dtype == other.dtype
            and self.shape == other.shape and self.weak_type == other.weak_type
            and np.all(self.val == other.val))

  def __hash__(self):
    return id(self.val)

  def at_least_vspace(self):
    return ShapedArray(self.shape, self.dtype, weak_type=self.weak_type)

  def join(self, other) -> UnshapedArray:
    if self == other:
      return self
    elif self.shape == other.shape and self.dtype == other.dtype:
      return ShapedArray(self.shape, self.dtype,
                         weak_type=self.weak_type and other.weak_type)
    elif self.dtype == other.dtype:
      return UnshapedArray(self.dtype,
                           weak_type=self.weak_type and other.weak_type)
    else:
      raise TypeError(self, other)

  def str_short(self) -> str:
    return str(self.val)

  def strip_weak_type(self) -> 'ConcreteArray':
    return ConcreteArray(self.val) if self.weak_type else self

  _bool = _nonzero = partialmethod(_forward_to_value, bool)
  _int     = partialmethod(_forward_to_value, int)
  _hex     = partialmethod(_forward_to_value, hex)
  _oct     = partialmethod(_forward_to_value, oct)


class AbstractToken(AbstractValue):
  def join(self, other):
    if isinstance(other, AbstractToken):
      return self
    else:
      assert False, f"Cannot join {self} with {other}"
  def str_short(self): return 'Tok'

abstract_token = AbstractToken()


def raise_to_shaped(aval: AbstractValue, weak_type=False):
  for typ in type(aval).mro():
    handler = raise_to_shaped_mappings.get(typ)
    if handler: return handler(aval, weak_type)
  raise TypeError(type(aval))

raise_to_shaped_mappings : Dict[type, Callable] = {
  AbstractUnit: lambda aval, _: aval,
  AbstractToken: lambda aval, _: aval,
  ShapedArray: lambda aval, weak_type: ShapedArray(aval.shape, aval.dtype, weak_type=weak_type)
}

# Registry for valid dimension types. This is used by masking.Poly.
_DIMENSION_TYPES: Set[type] = {int}

def _canonicalize_dimension(dim):
  if type(dim) in _DIMENSION_TYPES:
    return dim
  else:
    return operator.index(dim)

def canonicalize_shape(shape):
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
  msg = ("Shapes must be 1D sequences of concrete values of integer type, "
         "got {}.")
  if any(isinstance(x, Tracer) and isinstance(get_aval(x), ShapedArray)
         and not isinstance(get_aval(x), ConcreteArray) for x in shape):
    msg += ("\nIf using `jit`, try using `static_argnums` or applying `jit` to "
            "smaller subfunctions.")
  raise TypeError(msg.format(shape))


# ------------------- Call -------------------

def apply_todos(todos, outs):
  todos_list = list(todos)
  while todos_list:
    outs = map(full_lower, todos_list.pop()(outs))
  return outs

@lu.transformation_with_aux
def process_env_traces(primitive: Union['CallPrimitive', 'MapPrimitive'],
                       level: int, params_tuple: tuple, *args):
  outs = yield args, {}
  params = dict(params_tuple)
  todo = []
  while True:
    tracers = [x for x in outs if isinstance(x, Tracer)
               and (level is None or x._trace.level > level)]
    if tracers:
      ans = max(tracers, key=lambda x: x._trace.level)
    else:
      break
    trace = type(ans._trace)(ans._trace.master, cur_sublevel())
    outs = map(trace.full_raise, outs)
    outs, cur_todo = primitive.post_process(trace, outs, params)
    todo.append(cur_todo)
  yield outs, tuple(todo)  # Ensure the aux output is immutable

def call_bind(primitive: Union['CallPrimitive', 'MapPrimitive'],
              fun: lu.WrappedFun, *args, **params):
  params_tuple = tuple(params.items())
  top_trace = find_top_trace(args)
  level = (thread_local_state.trace_state.trace_stack.next_level(True)
           if top_trace is None else top_trace.level)
  params_tuple = tuple(params.items())
  fun, env_trace_todo = process_env_traces(fun, primitive, level, params_tuple)
  if top_trace is None:
    with new_sublevel():
      outs = primitive.impl(fun, *args, **params)
  else:
    tracers = map(top_trace.full_raise, args)
    outs = primitive.process(top_trace, fun, tracers, params)
  return apply_todos(env_trace_todo(), map(full_lower, outs))

class CallPrimitive(Primitive):
  multiple_results = True
  call_primitive = True

  def bind(self, fun, *args, **params):
    return call_bind(self, fun, *args, **params)

  def process(self, trace, fun, tracers, params):
    return trace.process_call(self, fun, tracers, params)

  def post_process(self, trace, out_tracers, params):
    return trace.post_process_call(self, out_tracers, params)

def call_impl(f: lu.WrappedFun, *args, **params):
  del params  # params parameterize the call primitive, not the function
  return f.call_wrapped(*args)

call_p = CallPrimitive('call')
call = call_p.bind
call_p.def_impl(call_impl)


# ------------------- Map -------------------

class MapPrimitive(Primitive):
  multiple_results = True
  map_primitive = True

  def bind(self, fun, *args, **params):
    assert len(params['mapped_invars']) == len(args)
    return call_bind(self, fun, *args, **params)

  def process(self, trace, fun, tracers, params):
    return trace.process_map(self, fun, tracers, params)

  def post_process(self, trace, out_tracers, params):
    return trace.post_process_map(self, out_tracers, params)

# This is a no-op with omnistaging disabled
@contextmanager
def extend_axis_env(axis_name, size: int, tag: Any):
  yield

# ------------------- Jaxpr checking -------------------

def mapped_aval(size: int, aval: AbstractValue) -> AbstractValue:
  if aval is abstract_unit:
    return aval
  elif isinstance(aval, ShapedArray):
    # might be raising abstraction level from Concrete here
    assert aval.shape[0] == size
    return ShapedArray(aval.shape[1:], aval.dtype)
  else:
    raise TypeError(f"Mapped operand {aval}")

def unmapped_aval(size: int, aval: AbstractValue) -> AbstractValue:
  if aval is abstract_unit:
    return aval
  elif isinstance(aval, ShapedArray):
    return ShapedArray((size,) + aval.shape, aval.dtype)
  else:
    raise TypeError(f"Mapped output {aval}")

def typecheck(aval: AbstractValue, x) -> bool:
  return typecompat(aval, get_aval(x))

def typecompat(aval_ref: AbstractValue, aval: AbstractValue) -> bool:
  """Determine whether `aval` conforms to `aval_ref`"""
  aval_ref = raise_to_shaped(aval_ref).strip_weak_type()
  try:
    return aval_ref == lattice_join(aval_ref, aval).strip_weak_type()
  except TypeError:
    return False

def typematch(aval1: UnshapedArray, aval2: UnshapedArray) -> bool:
  return (raise_to_shaped(aval1).strip_weak_type() ==
          raise_to_shaped(aval2).strip_weak_type())

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

  Raises `TypeError` if `jaxpr` is determined invalid. Returns `None` otherwise.
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
      return get_aval(v.val)
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
    in_avals = map(read, eqn.invars)
    prim = eqn.primitive
    try:
      if prim in custom_typechecks:
        custom_typechecks[prim](*in_avals, **eqn.params)
      if prim.call_primitive:
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
  typecheck_assert("mapped_invars" in params,
                   f"Map primitive {prim} missing 'mapped_invars' parameter")
  mapped_invars = params["mapped_invars"]

  binder_avals = [unmapped_aval(axis_size, v.aval) if mapped else v.aval
                  for v, mapped in zip(call_jaxpr.invars, mapped_invars)]
  for binder_aval, in_aval in zip(binder_avals, in_avals):
    typecheck_assert(typecompat(binder_aval, in_aval),
                     f"Call primitive {prim} passes operand {in_aval} "
                     f"to jaxpr expecting {binder_aval}")

  mapped_avals = [mapped_aval(axis_size, aval) if mapped else aval
                  for aval, mapped in zip(in_avals, mapped_invars)]
  _check_jaxpr(call_jaxpr, mapped_avals)

  mapped_out_avals = [v.aval for v in call_jaxpr.outvars]
  out_avals = [unmapped_aval(axis_size, aval) for aval in mapped_out_avals]
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
                         not isinstance(v, (Jaxpr, TypedJaxpr)))}
  return pp(primitive_name) >> pp_kv_pairs(sorted(filtered_params.items()))

def pp_eqn(eqn: JaxprEqn, print_shapes: bool = False) -> PrettyPrint:
  lhs = pp_vars(eqn.outvars, print_shapes)
  pp_lhs = pp(f'{lhs} =')
  pp_rhs = (pp(eqn.primitive.name) >>
            pp_kv_pairs(sorted(eqn.params.items())) >> pp(' ') >>
            pp(pp_vars(eqn.invars, print_shapes)))
  if len(lhs) <= 6:
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
  jaxprs = [j.jaxpr if isinstance(j, TypedJaxpr) else j for j in jaxprs]
  return pp('( ') >> vcat(map(pp_jaxpr, jaxprs)) >> pp(' )')

def pp_kv_pair(k, v):
  if type(v) is tuple and all(isinstance(j, (Jaxpr, TypedJaxpr)) for j in v):
    pp_v = pp_jaxprs(v)
  else:
    pp_v = pp(v)
  return pp(f'{k}=') >> pp_v

def pp_kv_pairs(kv_pairs):
  if kv_pairs:
    return pp('[ ') >> vcat([pp_kv_pair(k, v) for k, v in kv_pairs]) >> pp(' ]')
  else:
    return pp('')

axis_frame = None

# TODO(mattjj): remove when omnistaging fully lands
@config.omnistaging_enablers.append
@no_type_check
def omnistaging_enabler() -> None:
  global thread_local_state, call_bind, find_top_trace, initial_style_staging, \
      new_master, reset_trace_state, extend_axis_env, extend_axis_env_poly, \
      axis_frame, axis_frame_poly, axis_index, axis_index_p, new_base_master, \
      eval_context, TraceStack, TraceState
  del initial_style_staging

  class TraceStack:
    stack: List[MasterTrace]
    dynamic: MasterTrace

    def __init__(self):
      eval_trace = MasterTrace(0, EvalTrace)
      self.stack = [eval_trace]
      self.dynamic = eval_trace

    def next_level(self) -> int:
      return len(self.stack)

    def push(self, master_trace: MasterTrace) -> None:
      self.stack.append(master_trace)

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

  class TraceState:
    trace_stack: TraceStack
    substack: List[Sublevel]
    axis_env: List[AxisEnvFrame]
    axis_env_poly: List[AxisEnvFrame]

    def __init__(self) -> None:
      self.trace_stack = TraceStack()
      self.substack = [Sublevel(0)]
      self.axis_env = []
      self.axis_env_poly = []

    def copy(self):
      new = self.__new__(TraceState)
      new.trace_stack = self.trace_stack.copy()
      new.substack = self.substack[:]
      new.axis_env = self.axis_env[:]
      new.axis_env_poly = self.axis_env_poly[:]
      return new

  thread_local_state = ThreadLocalState()

  def reset_trace_state() -> bool:
    "Reset the global trace state and return True if it was already clean."
    if (thread_local_state.trace_state.substack != [Sublevel(0)] or
        thread_local_state.trace_state.axis_env != [] or
        thread_local_state.trace_state.trace_stack.stack != [MasterTrace(0, EvalTrace)] or
        thread_local_state.trace_state.trace_stack.dynamic != MasterTrace(0, EvalTrace)):
      thread_local_state.trace_state.__init__()  # type: ignore
      return False
    else:
      return True

  def call_bind(primitive: Union['CallPrimitive', 'MapPrimitive'],
                fun, *args, **params):
    params_tuple = tuple(params.items())
    top_trace = find_top_trace(args)
    fun, env_trace_todo = process_env_traces(
        fun, primitive, top_trace and top_trace.level, params_tuple)
    tracers = map(top_trace.full_raise, args)
    with maybe_new_sublevel(top_trace):
      outs = primitive.process(top_trace, fun, tracers, params)
    return map(full_lower, apply_todos(env_trace_todo(), outs))

  def maybe_new_sublevel(trace):
    # dynamic traces run the WrappedFun, so we raise the sublevel for them
    dynamic = thread_local_state.trace_state.trace_stack.dynamic
    return new_sublevel() if trace.master is dynamic else suppress()

  def find_top_trace(xs) -> Trace:
    top_master = max((x._trace.master for x in xs if isinstance(x, Tracer)),
                     default=None, key=attrgetter('level'))
    dynamic = thread_local_state.trace_state.trace_stack.dynamic
    top_master = (dynamic if top_master is None or dynamic.level > top_master.level
                  else top_master)
    return top_master and top_master.trace_type(top_master, cur_sublevel())  # type: ignore

  @contextmanager
  def new_master(trace_type: Type[Trace], dynamic: bool = False,
                 ) -> Generator[MasterTrace, None, None]:
    stack = thread_local_state.trace_state.trace_stack
    level = stack.next_level()
    master = MasterTrace(level, trace_type)
    stack.push(master)
    if dynamic:
      prev_dynamic, stack.dynamic = stack.dynamic, master

    try:
      yield master
    finally:
      thread_local_state.trace_state.trace_stack.pop()
      if dynamic:
        stack.dynamic = prev_dynamic

    if check_leaks:
      t = ref(master)
      del master
      if t() is not None:
        print(thread_local_state.trace_state.trace_stack)
        raise Exception('Leaked trace {}'.format(t()))

  @contextmanager
  def new_base_master(trace_type: Type[Trace]) -> Generator[MasterTrace, None, None]:
    stack = thread_local_state.trace_state.trace_stack
    master = MasterTrace(0, trace_type)
    prev_dynamic, stack.dynamic = stack.dynamic, master
    prev_base, stack.stack[0] = stack.stack[0], master
    try:
      yield master
    finally:
      stack.dynamic = prev_dynamic
      stack.stack[0] = prev_base

  @contextmanager
  def eval_context():
    with new_base_master(EvalTrace):
      yield

  def bind(self, *args, **params):
    assert skip_checks or all(isinstance(arg, Tracer)
                              or valid_jaxtype(arg) for arg in args), args
    top_trace = find_top_trace(args)
    tracers = map(top_trace.full_raise, args)
    out = top_trace.process_primitive(self, tracers, params)
    return map(full_lower, out) if self.multiple_results else full_lower(out)
  Primitive.bind = bind

  @contextmanager
  def extend_axis_env(axis_name, size: int, tag: Any):
    frame = AxisEnvFrame(axis_name, size, tag)
    thread_local_state.trace_state.axis_env.append(frame)
    try:
      yield
    finally:
      frame_ = thread_local_state.trace_state.axis_env.pop()
      assert frame is frame_

  def axis_frame(axis_name):
    frames = thread_local_state.trace_state.axis_env
    for frame in reversed(frames):
      if frame.name == axis_name:
        return frame
    else:
      raise NameError("unbound axis name: {}".format(axis_name))

  @contextmanager
  def extend_axis_env_poly(axis_name, size: int, tag: Any):
    # TODO(j-towns): de-duplicate this code
    frame = AxisEnvFrame(axis_name, size, tag)
    thread_local_state.trace_state.axis_env_poly.append(frame)
    try:
      yield
    finally:
      frame_ = thread_local_state.trace_state.axis_env_poly.pop()
      assert frame is frame_

  def axis_frame_poly(axis_name):
    # TODO(j-towns): de-duplicate this code
    frames = thread_local_state.trace_state.axis_env_poly
    for frame in reversed(frames):
      if frame.name == axis_name:
        return frame
    else:
      raise NameError("unbound axis name: {}".format(axis_name))

  def axis_index(axis_name):
    """Return the index along the mapped axis ``axis_name``.

    Args:
      axis_name: hashable Python object used to name the mapped axis.

    Returns:
      An integer representing the index.

    For example, with 8 XLA devices available:

    >>> from functools import partial
    >>> @partial(jax.pmap, axis_name='i')
    ... def f(_):
    ...   return lax.axis_index('i')
    ...
    >>> f(np.zeros(4))
    ShardedDeviceArray([0, 1, 2, 3], dtype=int32)
    >>> f(np.zeros(8))
    ShardedDeviceArray([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
    >>> @partial(jax.pmap, axis_name='i')
    ... @partial(jax.pmap, axis_name='j')
    ... def f(_):
    ...   return lax.axis_index('i'), lax.axis_index('j')
    ...
    >>> x, y = f(np.zeros((4, 2)))
    >>> print(x)
    [[0 0]
    [1 1]
    [2 2]
    [3 3]]
    >>> print(y)
    [[0 1]
    [0 1]
    [0 1]
    [0 1]]
    """
    return axis_index_p.bind(axis_name=axis_name)

  axis_index_p = Primitive('axis_index')
  axis_index_p.def_abstract_eval(lambda *, axis_name: ShapedArray((), np.int32))
