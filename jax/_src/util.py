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


import functools
import itertools as it
import operator
import types
from typing import Any, Callable

from absl import logging
import numpy as np

from jax.config import config

partial = functools.partial


def safe_zip(*args):
  n = len(args[0])
  for arg in args[1:]:
    assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
  return list(zip(*args))

def safe_map(f, *args):
  args = list(map(list, args))
  n = len(args[0])
  for arg in args[1:]:
    assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
  return list(map(f, *args))

def unzip2(xys):
  xs = []
  ys = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)

def unzip3(xyzs):
  xs = []
  ys = []
  zs = []
  for x, y, z in xyzs:
    xs.append(x)
    ys.append(y)
    zs.append(z)
  return tuple(xs), tuple(ys), tuple(zs)

def subvals(lst, replace):
  lst = list(lst)
  for i, v in replace:
    lst[i] = v
  return tuple(lst)

def split_list(args, ns):
  assert type(ns) is list
  args = list(args)
  lists = []
  for n in ns:
    lists.append(args[:n])
    args = args[n:]
  lists.append(args)
  return lists

def split_dict(dct, names):
  dct = dict(dct)
  lst = [dct.pop(name) for name in names]
  assert not dct
  return lst

def concatenate(xs):
  return list(it.chain.from_iterable(xs))

class partialmethod(functools.partial):
  def __get__(self, instance, owner):
    if instance is None:
      return self
    else:
      return partial(self.func, instance,
                     *(self.args or ()), **(self.keywords or {}))

def curry(f):
  """Curries arguments of f, returning a function on any remaining arguments.

  For example:
  >>> f = lambda x, y, z, w: x * y + z * w
  >>> f(2,3,4,5)
  26
  >>> curry(f)(2)(3, 4, 5)
  26
  >>> curry(f)(2, 3)(4, 5)
  26
  >>> curry(f)(2, 3, 4, 5)()
  26
  """
  return partial(partial, f)

def toposort(end_nodes):
  if not end_nodes: return []
  end_nodes = _remove_duplicates(end_nodes)

  child_counts = {}
  stack = list(end_nodes)
  while stack:
    node = stack.pop()
    if id(node) in child_counts:
      child_counts[id(node)] += 1
    else:
      child_counts[id(node)] = 1
      stack.extend(node.parents)
  for node in end_nodes:
    child_counts[id(node)] -= 1

  sorted_nodes = []
  childless_nodes = [node for node in end_nodes if child_counts[id(node)] == 0]
  assert childless_nodes
  while childless_nodes:
    node = childless_nodes.pop()
    sorted_nodes.append(node)
    for parent in node.parents:
      if child_counts[id(parent)] == 1:
        childless_nodes.append(parent)
      else:
        child_counts[id(parent)] -= 1

  check_toposort(sorted_nodes[::-1])
  return sorted_nodes[::-1]

def check_toposort(nodes):
  visited = set()
  for node in nodes:
    assert all(id(parent) in visited for parent in node.parents)
    visited.add(id(node))

def _remove_duplicates(node_list):
  seen = set()
  out = []
  for n in node_list:
    if id(n) not in seen:
      seen.add(id(n))
      out.append(n)
  return out

def split_merge(predicate, xs):
  sides = list(map(predicate, xs))
  lhs = [x for x, s in zip(xs, sides) if s]
  rhs = [x for x, s in zip(xs, sides) if not s]
  def merge(new_lhs, new_rhs):
    out = []
    for s in sides:
      if s:
        out.append(new_lhs[0])
        new_lhs = new_lhs[1:]
      else:
        out.append(new_rhs[0])
        new_rhs = new_rhs[1:]
    assert not new_rhs
    assert not new_lhs
    return out

  return lhs, rhs, merge

def cache(max_size=4096):
  def wrap(f):
    @functools.lru_cache(max_size)
    def cached(_, *args, **kwargs):
      return f(*args, **kwargs)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      if config.jax_check_tracer_leaks:
        return f(*args, **kwargs)
      else:
        return cached(config._trace_context(), *args, **kwargs)

    wrapper.cache_clear = cached.cache_clear
    wrapper.cache_info = cached.cache_info
    return wrapper
  return wrap

def memoize(f):
  @functools.lru_cache(None)
  def memoized(_, *args, **kwargs):
    return f(*args, **kwargs)

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    return memoized(config._trace_context(), *args, **kwargs)

  wrapper.cache_clear = memoized.cache_clear
  wrapper.cache_info = memoized.cache_info
  return wrapper

def prod(xs):
  out = 1
  for x in xs:
    out *= x
  return out

class WrapHashably:
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return id(self.val)

  def __eq__(self, other):
    return self.val is other.val

class Hashable:
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return self.val == other.val

class WrapKwArgs:
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  def __hash__(self):
    return hash(tuple((k, v) for k, v in sorted(self.val.items())))

  def __eq__(self, other):
    return self.val == other.val

def get_module_functions(module):
  """Finds functions in module.
  Args:
    module: A Python module.
  Returns:
    module_fns: A dict of names mapped to functions, builtins or ufuncs in `module`.
  """
  module_fns = {}
  for key in dir(module):
    # Omitting module level __getattr__, __dir__ which was added in Python 3.7
    # https://www.python.org/dev/peps/pep-0562/
    if key in ('__getattr__', '__dir__'):
      continue
    attr = getattr(module, key)
    if isinstance(
        attr, (types.BuiltinFunctionType, types.FunctionType, np.ufunc)):
      module_fns[key] = attr
  return module_fns

def wrap_name(name, transform_name):
  return transform_name + '(' + name + ')'

def extend_name_stack(stack, name=''):
  return stack + name + '/'

def canonicalize_axis(axis, num_dims) -> int:
  """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
  axis = operator.index(axis)
  if not -num_dims <= axis < num_dims:
    raise ValueError(
        "axis {} is out of bounds for array of dimension {}".format(
            axis, num_dims))
  if axis < 0:
    axis = axis + num_dims
  return axis

def moveaxis(x, src, dst):
  if src == dst:
    return x
  if isinstance(src, int):
    src = (src,)
  if isinstance(dst, int):
    dst = (dst,)
  src = [canonicalize_axis(a, x.ndim) for a in src]
  dst = [canonicalize_axis(a, x.ndim) for a in dst]
  perm = [i for i in range(np.ndim(x)) if i not in src]
  for d, s in sorted(zip(dst, src)):
    perm.insert(d, s)
  return x.transpose(perm)

def ceil_of_ratio(x, y):
  return -(-x // y)

@curry
def wraps(wrapped, fun, namestr="{fun}", docstr="{doc}", **kwargs):
  """
  Like functools.wraps, but with finer-grained control over the name and docstring
  of the resulting function.
  """
  try:
    name = getattr(wrapped, "__name__", "<unnamed function>")
    doc = getattr(wrapped, "__doc__", "") or ""
    fun.__dict__.update(getattr(wrapped, "__dict__", {}))
    fun.__annotations__ = getattr(wrapped, "__annotations__", {})
    fun.__name__ = namestr.format(fun=name)
    fun.__module__ = getattr(wrapped, "__module__", "<unknown module>")
    fun.__doc__ = docstr.format(fun=name, doc=doc, **kwargs)
    fun.__qualname__ = getattr(wrapped, "__qualname__", fun.__name__)
    fun.__wrapped__ = wrapped
  finally:
    return fun

# NOTE: Ideally we would annotate both the argument and return type as NoReturn
#       but it seems like pytype doesn't support that...
def assert_unreachable(x):
  raise AssertionError(f"Unhandled case: {type(x).__name__}")

def tuple_insert(t, idx, val):
  assert 0 <= idx <= len(t), (idx, len(t))
  return t[:idx] + (val,) + t[idx:]

def tuple_delete(t, idx):
  assert 0 <= idx < len(t), (idx, len(t))
  return t[:idx] + t[idx + 1:]

# TODO(mattjj): replace with dataclass when Python 2 support is removed
def taggedtuple(name, fields) -> Callable[..., Any]:
  """Lightweight version of namedtuple where equality depends on the type."""
  def __new__(cls, *xs):
    return tuple.__new__(cls, (cls,) + xs)
  def __repr__(self):
    return '{}{}'.format(name, tuple.__str__(self[1:]))
  class_namespace = {'__new__' : __new__, '__repr__': __repr__}
  for i, f in enumerate(fields):
    class_namespace[f] = property(operator.itemgetter(i+1))  # type: ignore
  return type(name, (tuple,), class_namespace)

class HashableFunction:
  """Decouples function equality and hash from its identity.

  Local lambdas and functiond defs are reallocated on each function call, making
  the functions created on different calls compare as unequal. This breaks our
  caching logic, which should really only care about comparing the semantics and
  not actual identity.

  This class makes it possible to compare different functions based on their
  semantics. The parts that are taken into account are: the bytecode of
  the wrapped function (which is cached by the CPython interpreter and is stable
  across the invocations of the surrounding function), and `closure` which should
  contain all values in scope that affect the function semantics. In particular
  `closure` should contain all elements of the function closure, or it should be
  possible to derive the relevant elements of the true function closure based
  solely on the contents of the `closure` argument (e.g. in case some closed-over
  values are not hashable, but are entirely determined by hashable locals).
  """

  def __init__(self, f, closure):
    self.f = f
    self.closure = closure

  def __eq__(self, other):
    return (type(other) is HashableFunction and
            self.f.__code__ == other.f.__code__ and
            self.closure == other.closure)

  def __hash__(self):
    return hash((self.f.__code__, self.closure))

  def __call__(self, *args, **kwargs):
    return self.f(*args, **kwargs)

  def __repr__(self):
    return f'<hashable {self.f.__name__} with closure={self.closure}>'

def as_hashable_function(closure):
  return partial(HashableFunction, closure=closure)

def maybe_named_axis(axis, if_pos, if_named):
  try:
    pos = operator.index(axis)
    named = False
  except TypeError:
    named = True
  return if_named(axis) if named else if_pos(pos)

def distributed_debug_log(*pairs):
  """Format and log `pairs` if config.jax_distributed_debug is enabled.

  Args:
    pairs: A sequence of label/value pairs to log. The first pair is treated as
    a heading for subsequent pairs.
  """
  if config.jax_distributed_debug:
    lines = ["\nDISTRIBUTED_DEBUG_BEGIN"]
    try:
      lines.append(f"{pairs[0][0]}: {pairs[0][1]}")
      for label, value in pairs[1:]:
        lines.append(f"  {label}: {value}")
    except Exception as e:
      lines.append("DISTRIBUTED_DEBUG logging failed!")
      lines.append(f"{e}")
    lines.append("DISTRIBUTED_DEBUG_END")
    logging.warning("\n".join(lines))
