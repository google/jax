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

from __future__ import print_function

from contextlib import contextmanager
from collections import defaultdict, Counter, namedtuple
import functools
from functools import partial, wraps
import itertools as it
import operator as op
import string

import numpy as onp

from .. import core
from ..core import Trace, Tracer
from ..util import unzip2, safe_map, safe_zip, curry
from ..abstract_arrays import ShapedArray
from .. import linear_util as lu

map = safe_map
zip = safe_zip

def prod(xs):
  xs = list(xs)
  return functools.reduce(op.mul, xs) if xs else 1


### main transformation functions

ShapeEnvs = namedtuple("ShapeEnvs", ["logical", "padded"])
shape_envs = ShapeEnvs({}, {})  # TODO(mattjj): make this a stack for efficiency

@contextmanager
def extend_shape_envs(logical_env, padded_env):
  global shape_envs
  new_logical = dict(it.chain(shape_envs.logical.items(), logical_env.items()))
  new_padded = dict(it.chain(shape_envs.padded.items(), padded_env.items()))
  shape_envs, prev = ShapeEnvs(new_logical, new_padded), shape_envs
  yield
  shape_envs = prev

def is_polymorphic(shape):
  return any(map(lambda d: isinstance(d, Poly), shape))

def shape_as_value(expr):
  if type(expr) is ShapeExpr:
    return eval_shape_expr(shape_envs.logical, expr)
  elif type(expr) is tuple and is_polymorphic(expr):
    return tuple(eval_dim_expr(shape_envs.logical, d) if type(d) is Poly else d
                 for d in expr)
  else:
    return expr

def padded_shape_as_value(expr):
  if type(expr) is ShapeExpr:
    return eval_shape_expr(shape_envs.padded, expr)
  elif type(expr) is tuple and is_polymorphic(expr):
    return tuple(eval_dim_expr(shape_envs.padded, d) if type(d) is Poly else d
                 for d in expr)
  else:
    return expr


def mask_fun(fun, logical_env, padded_env, in_vals, shape_exprs):
  with core.new_master(MaskTrace) as master:
    fun, out_shapes = mask_subtrace(fun, master)
    with extend_shape_envs(logical_env, padded_env):
      out_vals = fun.call_wrapped(in_vals, shape_exprs)
    del master
  return out_vals, out_shapes()

@lu.transformation_with_aux
def mask_subtrace(master, in_vals, shape_exprs):
  trace = MaskTrace(master, core.cur_sublevel())
  in_tracers = [MaskTracer(trace, x, s).full_lower()
                for x, s in zip(in_vals, shape_exprs)]
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_shapes = unzip2((t.val, t.shape_expr) for t in out_tracers)
  yield out_vals, out_shapes


### shape expressions

# Shape expressions model tuples of formal polynomials with integer
# coefficients. Here are the internal data structures we use to represent them.
#
#   type ShapeExpr = [Poly]
#   type Poly = Map Mon Int
#   type Mon = Map Str Int

class ShapeExpr(tuple):  # type ShapeExpr = [Poly]
  def __str__(self):
    return 'ShapeExpr({})'.format(', '.join(map(str, self)))
  def __getitem__(self, idx):
    poly = super(ShapeExpr, self).__getitem__(idx)
    return poly if type(idx) is int else ShapeExpr(poly)


def canonicalize_poly(p):
  if onp.isscalar(p) and onp.issubdtype(onp.int, onp.array(p).dtype):
    return constant_poly(p)

  assert isinstance(p, Poly)
  return p

def canonicalize_polymorphic_shape(shape):
  return ShapeExpr(map(canonicalize_poly, shape))

def poly_without_zeros(d):
  d = {mon: count for mon, count in d.items() if count != 0}

  return constant_poly(0) if len(d) == 0 else Poly(d)

class Poly(Counter):  # type Poly = Map Mon Int -- monomials to coeffs
  def __add__(self, other):
    d = self.copy()

    for mon, count in canonicalize_poly(other).items():
      d[mon] = d.get(mon, 0) + count

    return poly_without_zeros(d)

  def __sub__(self, other):
    return self + -other

  def __neg__(self):
    return Poly({mon: -count for mon, count in self.items()})

  def __mul__(self, other):
    new_poly = dict()
    for (mon1, coeff1), (mon2, coeff2) \
            in it.product(self.items(), canonicalize_poly(other).items()):
      mon = Mon(mon1 + mon2)                        # add monomials' id degrees
      coeff = coeff1 * coeff2                       # multiply integer coeffs
      new_poly[mon] = new_poly.get(mon, 0) + coeff  # accumulate coeffs

    return poly_without_zeros(new_poly)

  def __rmul__(self, other):
    return self * other

  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    return self + -other

  def __floordiv__(self, divisor):
    q, _ = divmod(self, divisor)
    return q

  def __mod__(self, divisor):
    _, r = divmod(self, divisor)
    return r

  def __divmod__(self, divisor):
    if self.is_constant:
      q, r = divmod(int(self), divisor)

      return constant_poly(q), r

    def divided(count):
      q, r = divmod(count, divisor)
      if r != 0:
        raise ValueError('shapecheck currently only supports strides '
                         'that exactly divide the strided axis length.')
      return q

    return Poly({k: divided(count) for k, count in self.items()}), 0

  def __hash__(self):
    return hash(tuple(self.items()))

  def __str__(self):
    return ' + '.join('{} {}'.format(v, k) if (v != 1 or k.degree == 0) else str(k)
                      for k, v in sorted(self.items())).strip()

  def __int__(self):
    assert self.is_constant

    return int(next(iter(self.values())))

  @property
  def is_constant(self):
    return len(self) == 1 and next(iter(self)).degree == 0

class Mon(Counter):  # type Mon = Map Id Int -- ids to degrees
  def __hash__(self):
    return hash(tuple(self.items()))

  def __str__(self):
    return ' '.join('{}**{}'.format(k, v) if v != 1 else str(k)
                    for k, v in sorted(self.items()))

  def __lt__(self, other):
    # sort by total degree, then lexicographically on indets
    self_key = self.degree, tuple(sorted(self))
    other_key = other.degree, tuple(sorted(other))
    return self_key < other_key

  @property
  def degree(self):
    return sum(self.values())

def eval_shape_expr(env, expr):
  return tuple(eval_dim_expr(env, poly) for poly in expr)

def eval_dim_expr(env, poly):
  terms = [mul(coeff, prod([pow(env[id], deg) for id, deg in mon.items()]))
           for mon, coeff in poly.items()]
  return sum(terms) if len(terms) > 1 else terms[0]

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
# ShapeSpecs encode ShapeExprs but can have some monomorphic dims inside them,
# which must be replaced with concrete shapes when known.

class ShapeSpec(tuple):
  def __str__(self):
    return 'ShapeSpec({})'.format(', '.join(map(str, self)))

def finalize_spec(spec, shape):
  return ShapeExpr(parse_lit(d) if e is monomorphic_dim else e
                   for e, d in zip(spec, shape))

def parse_spec(spec=''):
  if not spec:
    return ShapeSpec(())
  if spec[0] == '(':
    if spec[-1] != ')': raise ShapeSyntaxError(spec)
    spec = spec[1:-1]
  dims = map(parse_dim, spec.replace(' ', '').strip(',').split(','))
  return ShapeSpec(dims)

def parse_dim(spec):
  if '+' in spec:
    terms = map(parse_dim, spec.split('+'))
    return functools.reduce(op.add, terms)
  elif '*' in spec:
    terms = map(parse_dim, spec.split('*'))
    return functools.reduce(op.mul, terms)
  elif spec.isdigit() or spec.startswith('-') and spec[1:].isdigit():
    return parse_lit(spec)
  elif spec in identifiers:
    return parse_id(spec)
  elif spec == '_':
    return monomorphic_dim
  else:
    raise ShapeSyntaxError(spec)
digits = frozenset(string.digits)
identifiers = frozenset(string.ascii_lowercase)

def parse_id(name): return Poly({Mon({name: 1}): 1})
def parse_lit(val_str): return constant_poly(int(val_str))
def constant_poly(val): return Poly({Mon(): val})

class MonomorphicDim(object):
  def __str__(self): return '_'
monomorphic_dim = MonomorphicDim()


# Two convenient ways to provide shape annotations:
#   1. '(m, n)'
#   2. s_['m', 'n']

class S_(object):
  def __getitem__(self, idx):
    if type(idx) is tuple:
      return parse_spec('(' + ','.join(map(str, idx)) + ')')
    else:
      return parse_spec(str(idx))
s_ = S_()


### automasking tracer machinery

class MaskTracer(Tracer):
  __slots__ = ["val", "shape_expr"]

  def __init__(self, trace, val, shape_expr):
    self.trace = trace
    self.val = val
    self.shape_expr = shape_expr

  @property
  def aval(self):
    return ShapedArray(self.shape_expr, self.val.dtype)

  def is_pure(self):
    return all(poly.is_constant for poly in self.shape_expr)

  def full_lower(self):
    if self.is_pure():
      return core.full_lower(self.val)
    else:
      return self

class MaskTrace(Trace):
  def pure(self, val):
    return MaskTracer(self, val, canonicalize_polymorphic_shape(onp.shape(val)))

  def lift(self, val):
    return MaskTracer(self, val, canonicalize_polymorphic_shape(onp.shape(val)))

  def sublift(self, val):
    return MaskTracer(self, val.val, val.shape_expr)

  def process_primitive(self, primitive, tracers, params):
    vals, shape_exprs = unzip2((t.val, t.shape_expr) for t in tracers)
    if primitive in shape_parameterized_primitive_rules:
      rule = shape_parameterized_primitive_rules[primitive]
      out, out_shape = rule(shape_envs, vals, shape_exprs, **params)
    else:
      out_shape = shape_rules[primitive](*(t.aval for t in tracers), **params)
      logical_shapes = map(partial(eval_shape_expr, shape_envs.logical), shape_exprs)
      out = masking_rules[primitive](vals, logical_shapes, **params)
    if not primitive.multiple_results:
      return MaskTracer(self, out, out_shape)
    else:
      return map(partial(MaskTracer, self), out, out_shape)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO mask-of-jit

shape_parameterized_primitive_rules = {}
masking_rules = {}
shape_rules = {}

def defvectorized(prim):
  shape_rules[prim] = vectorized_shape_rule
  masking_rules[prim] = partial(vectorized_masking_rule, prim)

def vectorized_shape_rule(operand, **unused_params):
  return operand.shape

def vectorized_masking_rule(prim, padded_vals, logical_shapes, **params):
  del logical_shapes  # Unused.
  padded_val, = padded_vals
  return prim.bind(padded_val, **params)


def defbinop(prim):
  shape_rules[prim] = binop_shape_rule
  masking_rules[prim] = partial(binop_masking_rule, prim)

def binop_shape_rule(x, y):
  if x.shape == y.shape:
    return x.shape
  elif not x.shape:
    return y.shape
  elif not y.shape:
    return x.shape
  else:
    raise ShapeError

def binop_masking_rule(prim, padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  padded_x, padded_y = padded_vals
  return prim.bind(padded_x, padded_y)


### definition-time (import-time) shape checker tracer machinery

def shapecheck(fun, in_shapes):
  with core.new_master(ShapeCheckTrace) as master:
    out_shapes = check_subtrace(fun, master).call_wrapped(in_shapes)
    del master
  return out_shapes

@lu.transformation
def check_subtrace(master, in_shapes):
  trace = ShapeCheckTrace(master, core.cur_sublevel())
  in_tracers = map(partial(ShapeCheckTracer, trace), in_shapes)
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  yield [t.shape_expr for t in out_tracers]


# TODO(mattjj): add dtypes?
class ShapeCheckTracer(Tracer):
  __slots__ = ["shape_expr"]

  def __init__(self, trace, shape_expr):
    self.trace = trace
    self.shape_expr = shape_expr

  @property
  def aval(self):
    return ShapedArray(self.shape_expr, None)

  def full_lower(self):
    return self

class ShapeCheckTrace(Trace):
  def pure(self, val):
    return ShapeCheckTracer(self, canonicalize_polymorphic_shape(onp.shape(val)))

  def lift(self, val):
    return ShapeCheckTracer(self, canonicalize_polymorphic_shape(onp.shape(val)))

  def sublift(self, val):
    return ShapeCheckTracer(self, val.shape_expr)

  def process_primitive(self, primitive, tracers, params):
    avals = [t.aval for t in tracers]
    shape_rule = shape_rules.get(primitive)
    if shape_rule is None:
      raise NotImplementedError('Shape rule for {} not implemented yet.'.format(primitive))
    out_shape = canonicalize_polymorphic_shape(shape_rule(*avals, **params))
    return ShapeCheckTracer(self, out_shape)

  def process_call(self, call_primitive, f, tracers, params):
    # TODO apply proper subtrace:
    return map(self.full_raise, f.call_wrapped(*tracers))
