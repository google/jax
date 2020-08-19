# Copyright 2019 Google LLC
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

from contextlib import contextmanager
from collections import Counter, namedtuple
from functools import partial, reduce
from itertools import chain, product
import operator as op
import string
from typing import Callable, Dict, Sequence, Union

import numpy as np

from .. import abstract_arrays
from .. import core, dtypes
from ..tree_util import tree_unflatten
from ..core import Trace, Tracer, mapped_aval, unmapped_aval
from ..util import safe_map, safe_zip, unzip2, prod, wrap_name
from ..abstract_arrays import ShapedArray
from .. import linear_util as lu

map = safe_map
zip = safe_zip

masking_rules: Dict[core.Primitive, Callable] = {}

def defvectorized(prim):
  masking_rules[prim] = partial(vectorized_masking_rule, prim)

def defnaryop(prim):
  masking_rules[prim] = partial(naryop_masking_rule, prim)

def vectorized_masking_rule(prim, padded_vals, logical_shapes, **params):
  del logical_shapes  # Unused.
  padded_val, = padded_vals
  return prim.bind(padded_val, **params)

def naryop_masking_rule(prim, padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  return prim.bind(*padded_vals)

ShapeEnvs = namedtuple("ShapeEnvs", ["logical", "padded"])
shape_envs = ShapeEnvs({}, {})  # TODO(mattjj): make this a stack for efficiency

def is_tracing():
  return bool(shape_envs.padded)

@contextmanager
def extend_shape_envs(logical_env, padded_env):
  global shape_envs
  new_logical = dict(chain(shape_envs.logical.items(), logical_env.items()))
  new_padded = dict(chain(shape_envs.padded.items(), padded_env.items()))
  shape_envs, prev = ShapeEnvs(new_logical, new_padded), shape_envs
  try:
    yield
  finally:
    shape_envs = prev

def shape_as_value(shape):
  assert is_tracing() or not is_polymorphic(shape)
  return eval_poly_shape(shape, shape_envs.logical)

def padded_shape_as_value(shape):
  assert is_tracing() or not is_polymorphic(shape)
  return eval_poly_shape(shape, shape_envs.padded)

def mask_fun(fun, logical_env, padded_env, in_vals, polymorphic_shapes):
  env_keys, padded_env_vals = unzip2(sorted(padded_env.items()))
  logical_env_vals = [logical_env[k] for k in env_keys]
  # Make padded_env hashable
  padded_env = (env_keys, padded_env_vals)
  with core.new_master(MaskTrace) as master:
    fun, out_shapes = mask_subtrace(fun, master, polymorphic_shapes, padded_env)
    out_vals = fun.call_wrapped(*(logical_env_vals + in_vals))
    del master
  return out_vals, out_shapes()

@lu.transformation_with_aux
def mask_subtrace(master, shapes, padded_env, *in_vals):
  env_keys, _ = padded_env
  logical_env_vals, in_vals = in_vals[:len(env_keys)], in_vals[len(env_keys):]
  logical_env = dict(zip(env_keys, logical_env_vals))
  padded_env = dict(zip(*padded_env))
  trace = MaskTrace(master, core.cur_sublevel())
  in_tracers = [MaskTracer(trace, x, s).full_lower()
                for x, s in zip(in_vals, shapes)]
  with extend_shape_envs(logical_env, padded_env):
    outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_shapes = unzip2((t.val, t.polymorphic_shape) for t in out_tracers)
  yield out_vals, out_shapes

@lu.transformation
def poly_mapped_axis(axis_name, axis_size, *in_vals):
  with core.extend_axis_env_poly(axis_name, axis_size, None):
    outs = yield in_vals, {}
  yield outs

def mapped_mask_subtrace(f, master, axis_name, axis_size, shapes, padded_env):
  # TODO(j-towns): if axis_size happens to be a constant then just use
  # mask_subtrace here
  return mask_subtrace(
      poly_mapped_axis(f, axis_name, axis_size), master, shapes, padded_env)

def eval_poly_shape(shape, values_dict):
  return tuple(eval_poly(dim, values_dict) for dim in shape)

def eval_poly(poly, values_dict):
  return poly.evaluate(values_dict) if type(poly) is Poly else poly

def eval_poly_logical(poly): return eval_poly(poly, shape_envs.logical)
def eval_poly_padded(poly): return eval_poly(poly, shape_envs.padded)

def _ensure_poly(p):
  if type(p) is Poly:
    return p

  return Poly({Mon(): p})

def is_polymorphic(shape: Sequence[Union[int, 'Poly']]):
  return any(map(lambda d: type(d) is Poly, shape))

class Poly(dict):
  """Polynomial with nonnegative integer coefficients for polymorphic shapes."""

  def __init__(self, coeffs):
    # Makes sure Polynomials are always in canonical form
    coeffs = {mon: op.index(coeff)
              for mon, coeff in coeffs.items() if coeff != 0}
    coeffs = coeffs or {Mon(): 0}
    super().__init__(coeffs)

  def __add__(self, other):
    coeffs = self.copy()
    for mon, coeff in _ensure_poly(other).items():
      coeffs[mon] = coeffs.get(mon, 0) + coeff
    return Poly(coeffs)

  def __sub__(self, other):
    return self + -other

  def __neg__(self):
    return Poly({mon: -coeff for mon, coeff in self.items()})

  def __mul__(self, other):
    other = _ensure_poly(other)
    coeffs = {}
    for (mon1, coeff1), (mon2, coeff2) in product(self.items(), other.items()):
      mon = mon1 * mon2
      coeffs[mon] = coeffs.get(mon, 0) + coeff1 * coeff2
    return Poly(coeffs)

  def __rmul__(self, other):
    return self * other  # multiplication commutes

  def __radd__(self, other):
    return self + other  # addition commutes

  def __rsub__(self, other):
    return _ensure_poly(other) - self

  def __floordiv__(self, divisor):
    q, _ = divmod(self, divisor)  # pytype: disable=wrong-arg-types
    return q

  def __mod__(self, divisor):
    _, r = divmod(self, divisor)  # pytype: disable=wrong-arg-types
    return r

  def __divmod__(self, divisor):
    if self.is_constant:
      return divmod(int(self), divisor)
    else:
      def divided(count):
        q, r = divmod(count, divisor)
        if r != 0:
          raise ValueError('shapecheck  and masking currently only support '
                           'strides that exactly divide the strided axis '
                           'length.')
        return q

      return Poly(
        {k: coeff // divisor if k.degree == 0 else divided(coeff)
         for k, coeff in self.items()}), self.get(Mon(), 0) % divisor

  def __hash__(self):
    return hash(tuple(sorted(self.items())))

  def __eq__(self, other):
    return super().__eq__(_ensure_poly(other))

  def __ne__(self, other):
    return not self == other

  def __ge__(self, other):
    other = _ensure_poly(other)

    if other.is_constant and self.is_constant:
      return int(self) >= int(other)
    elif other.is_constant and int(other) <= 1:
      # Assume nonzero polynomials are positive, allows use in shape rules
      return True
    elif self.is_constant and int(self) <= 0:
      return False  # See above.
    elif self == other:
      return True

    raise ValueError('Polynomials comparison "{} >= {}" is inconclusive.'
                     .format(self, other))

  def __le__(self, other):
    return _ensure_poly(other) >= self

  def __lt__(self, other):
    return not (self >= other)

  def __gt__(self, other):
    return not (_ensure_poly(other) >= self)

  def __str__(self):
    return ' + '.join('{} {}'.format(v, k)
                      if (v != 1 or k.degree == 0) else str(k)
                      for k, v in sorted(self.items())).strip()

  def __repr__(self):
    return str(self)

  def __int__(self):
    assert self.is_constant
    return op.index(next(iter(self.values())))

  def evaluate(self, env):
    prod = lambda xs: reduce(op.mul, xs) if xs else 1
    terms = [mul(coeff, prod([pow(env[id], deg) for id, deg in mon.items()]))
             for mon, coeff in self.items()]
    return sum(terms) if len(terms) > 1 else terms[0]

  @property
  def is_constant(self):
    return len(self) == 1 and next(iter(self)).degree == 0

def pow(x, deg):
  try:
    deg = int(deg)
  except:
    return x ** deg
  else:
    return 1 if deg == 0 else x if deg == 1 else x ** deg

def mul(coeff, mon):
  try:
    coeff = int(coeff)
  except:
    return coeff * mon
  else:
    return  0 if coeff == 0 else mon if coeff == 1 else coeff * mon


abstract_arrays._DIMENSION_TYPES.add(Poly)


class Mon(dict):
  def __hash__(self):
    return hash(frozenset(self.items()))

  def __str__(self):
    return ' '.join('{}**{}'.format(k, v) if v != 1 else str(k)
                    for k, v in sorted(self.items()))

  def __lt__(self, other):
    # sort by total degree, then lexicographically on indets
    self_key = -self.degree, tuple(sorted(self))
    other_key = -other.degree, tuple(sorted(other))
    return self_key < other_key

  def __mul__(self, other):
    return Mon(Counter(self) + Counter(other))

  @property
  def degree(self):
    return sum(self.values())

class ShapeError(Exception): pass

class ShapeSyntaxError(Exception): pass

# To denote some shape expressions (for annotations) we use a small language.
#
#   data ShapeSpec = ShapeSpec [Dim]
#   data Dim = Id PyObj
#            | Lit Int
#            | Mul Dim Dim
#            | Add Dim Dim
#            | MonomorphicDim
#
# We'll also make a simple concrete syntax for annotation. The grammar is
#
#   shape_spec ::= '(' dims ')'
#   dims       ::= dim ',' dims | ''
#   dim        ::= str | int | dim '*' dim | dim '+' dim | '_'
#
# ShapeSpecs can have some monomorphic dims inside them, which must be replaced
# with concrete shapes when known.

class ShapeSpec(tuple):
  def __str__(self):
    return 'ShapeSpec({})'.format(', '.join(map(str, self)))

def finalize_spec(polymorphic_shape, padded_shape):
  return tuple(_parse_lit(d) if e is _monomorphic_dim else e
               for e, d in zip(polymorphic_shape, padded_shape))

def parse_spec(spec=''):
  if not spec:
    return ShapeSpec(())
  if spec[0] == '(':
    if spec[-1] != ')': raise ShapeSyntaxError(spec)
    spec = spec[1:-1]
  dims = map(_parse_dim, spec.replace(' ', '').strip(',').split(','))
  return ShapeSpec(dims)

def _parse_dim(spec):
  if '+' in spec:
    return np.sum(map(_parse_dim, spec.split('+')))
  elif '*' in spec:
    return prod(map(_parse_dim, spec.split('*')))
  elif spec.isdigit() or spec.startswith('-') and spec[1:].isdigit():
    return _parse_lit(spec)
  elif spec in _identifiers:
    return _parse_id(spec)
  elif spec == '_':
    return _monomorphic_dim
  else:
    raise ShapeSyntaxError(spec)

_identifiers = frozenset(string.ascii_lowercase)

def _parse_id(name): return Poly({Mon({name: 1}): 1})

def _parse_lit(val_str): return Poly({Mon(): int(val_str)})

class MonomorphicDim(object):
  def __str__(self): return '_'

_monomorphic_dim = MonomorphicDim()

# Two convenient ways to provide shape annotations:
#   1. '(m, n)'
#   2. s_['m', 'n']

class S_(object):
  def __getitem__(self, idx):
    return parse_spec(('(' + ','.join(map(str, idx)) + ')')
                             if type(idx) is tuple else str(idx))

s_ = S_()

def _shape_spec_consistent(spec, expr):
  return all(a == b for a, b in zip(spec, expr) if a is not _monomorphic_dim)

class MaskTracer(Tracer):
  __slots__ = ["val", "polymorphic_shape"]

  def __init__(self, trace, val, polymorphic_shape):
    super().__init__(trace)
    self.val = val
    self.polymorphic_shape = polymorphic_shape

  @property
  def aval(self):
    return ShapedArray(self.polymorphic_shape, self.dtype)

  @property
  def dtype(self):
    return dtypes.dtype(self.val)

  def is_pure(self):
    return all(type(poly) is not Poly or poly.is_constant
               for poly in self.polymorphic_shape)

  def full_lower(self):
    if self.is_pure():
      return core.full_lower(self.val)
    else:
      return self


class MaskTrace(Trace):
  def pure(self, val):
    return MaskTracer(self, val, np.shape(val))

  def lift(self, val):
    return MaskTracer(self, val, np.shape(val))

  def sublift(self, val):
    return MaskTracer(self, val.val, val.polymorphic_shape)

  def process_primitive(self, primitive, tracers, params):
    masking_rule = masking_rules.get(primitive)
    if masking_rule is None:
      raise NotImplementedError(
        f'Masking rule for {primitive} not implemented yet.')
    out_aval = primitive.abstract_eval(*(t.aval for t in tracers), **params)
    vals, polymorphic_shapes = unzip2((t.val, t.polymorphic_shape) for t in tracers)
    logical_shapes = map(shape_as_value, polymorphic_shapes)
    out = masking_rule(vals, logical_shapes, **params)
    if primitive.multiple_results:
      return map(partial(MaskTracer, self), out, (o.shape for o in out_aval))
    else:
      return MaskTracer(self, out, out_aval.shape)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    params = dict(params, name=wrap_name(params.get('name', f.__name__), 'mask'))
    vals, shapes = unzip2((t.val, t.polymorphic_shape) for t in tracers)
    if not any(is_polymorphic(s) for s in shapes):
      return call_primitive.bind(f, *vals, **params)
    logical_env, padded_env = shape_envs
    env_keys, padded_env_vals = unzip2(sorted(padded_env.items()))
    logical_env_vals = tuple(logical_env[k] for k in env_keys)
    # Make padded_env hashable
    padded_env = (env_keys, padded_env_vals)
    f, shapes_out = mask_subtrace(f, self.master, shapes, padded_env)
    update_params = call_param_updaters.get(call_primitive)
    if update_params:
      params = update_params(params, len(env_keys))
    vals_out = call_primitive.bind(f, *(logical_env_vals + vals), **params)
    return [MaskTracer(self, v, s) for v, s in zip(vals_out, shapes_out())]

  def post_process_call(self, call_primitive, out_tracers, params):
    vals, shapes = unzip2((t.val, t.polymorphic_shape) for t in out_tracers)
    master = self.master
    def todo(vals):
      trace = MaskTrace(master, core.cur_sublevel())
      return map(partial(MaskTracer, trace), vals, shapes)
    return vals, todo

  def process_map(self, map_primitive, f, tracers, params):
    vals, shapes = unzip2((t.val, t.polymorphic_shape) for t in tracers)
    if not any(is_polymorphic(s) for s in shapes):
      return map_primitive.bind(f, *vals, **params)
    mapped_invars = params['mapped_invars']
    logical_env, padded_env = shape_envs
    poly_axis_size, = {s[0] for s, m in zip(shapes, mapped_invars) if m}
    mapped_shape = lambda s: mapped_aval(
        poly_axis_size, ShapedArray(s, None)).shape
    unmapped_shape = lambda s: unmapped_aval(
        poly_axis_size, ShapedArray(s, None)).shape
    padded_axis_size = eval_poly(poly_axis_size, padded_env)
    env_keys, padded_env_vals = unzip2(sorted(padded_env.items()))
    params = dict(
        params,
        mapped_invars=((False,) * len(logical_env) + params['mapped_invars']),
        axis_size=padded_axis_size)
    shapes = tuple(
        mapped_shape(s) if m else s for s, m in zip(shapes, mapped_invars))
    logical_env_vals = tuple(logical_env[k] for k in env_keys)
    padded_env = (env_keys, padded_env_vals)
    f, shapes_out = mapped_mask_subtrace(f, self.master, params['axis_name'],
                                         poly_axis_size, shapes, padded_env)
    update_params = call_param_updaters.get(map_primitive)
    if update_params:
      params = update_params(params, len(env_keys))
    vals_out = map_primitive.bind(f, *(logical_env_vals + vals), **params)
    shapes_out = [unmapped_shape(s) for s in shapes_out()]
    return [MaskTracer(self, v, s) for v, s in zip(vals_out, shapes_out)]

  def post_process_map(self, map_primitive, out_tracers, params):
    vals, shapes = unzip2((t.val, t.polymorphic_shape) for t in out_tracers)
    unmapped_shape = lambda s: unmapped_aval(
        params['axis_size'], ShapedArray(s, None)).shape
    master = self.master
    def todo(vals):
      trace = MaskTrace(master, core.cur_sublevel())
      return [MaskTracer(trace, val, unmapped_shape(shape))
              for val, shape in zip(vals, shapes)]
    return vals, todo

class UniqueId:
  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name

  def __lt__(self, other):
    return self.name < other.name

class UniqueIds(dict):
  def __missing__(self, key):
    unique_id = UniqueId(key)
    self[key] = unique_id
    return unique_id

def remap_ids(names, shape_spec):
  return ShapeSpec(Poly({Mon({names[id] : deg for id, deg in mon.items()})
                         : coeff for mon, coeff in poly.items()})
                   if poly is not _monomorphic_dim else
                   _monomorphic_dim for poly in shape_spec)

def bind_shapes(polymorphic_shapes, padded_shapes):
  env = {}
  for polymorphic_shape, padded_shape in zip(polymorphic_shapes, padded_shapes):
    for poly, d in zip(polymorphic_shape, padded_shape):
      if type(poly) is not Poly or poly.is_constant:
        if int(poly) != d: raise ShapeError
      else:
        poly = poly.copy()
        const_coeff = poly.pop(Mon({}), 0)
        (mon, linear_coeff), = poly.items()
        (id, index), = mon.items()
        if index != 1: raise ShapeError
        d, r = divmod(d - const_coeff, linear_coeff)
        assert r == 0
        if env.setdefault(id, d) != d: raise ShapeError
  return env

def check_shapes(specs, spec_tree, shapes, tree, message_prefix="Output"):
  if spec_tree != tree or not all(map(_shape_spec_consistent, specs, shapes)):
    specs = tree_unflatten(spec_tree, specs)
    shapes = tree_unflatten(tree, shapes)
    raise ShapeError(f"{message_prefix} shapes should be {specs} but are {shapes}.")

call_param_updaters: Dict[core.Primitive, Callable] = {}
