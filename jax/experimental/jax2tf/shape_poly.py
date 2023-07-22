# Copyright 2021 The JAX Authors.
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
"""Shape polymorphism support.

We introduce a set of dimension variables at the top-level of a `jit` function.
They are introduced implicitly by way of specifying for each dimension of each
argument a symbolic dimension expression in terms of some dimension variables.
All dimension variables are assumed to range over integers greater or equal to 1.

Symbolic dimensions overload some integer operations, such as
add, multiply, divide, equality, etc. The JAX NumPy layer and the LAX layers have been
touched up to be sensitive to handling shapes that contain symbolic dimensions.
This enables many JAX programs to be traced with symbolic dimensions
in some dimensions. A priority has been to enable the batch
dimension in neural network examples to be polymorphic.

This was built initially for jax2tf, but it is now customizeable to be
independent of TF. The best documentation at the moment is in the
jax2tf.convert docstring, and the
[README](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
"""

import collections
from collections.abc import Iterable, Sequence
import dataclasses
from enum import Enum
import functools
import itertools
import io
import math
import operator as op
import threading
import tokenize
from typing import Any, Callable, Optional, Union

import numpy as np
import opt_einsum

import jax
from jax import config
from jax.interpreters import xla

from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src.lax import lax
from jax._src.interpreters import mlir
from jax._src.numpy import lax_numpy
from jax._src import tree_util
from jax._src import util
from jax._src.typing import DimSize, Shape


TfVal = Any
DimVarEnv = dict[str, jax.Array]
DType = Any

# We have assumed so far that a dimension variable can range over strictly
# positive integers. It would be cleaner if we could assume that they range
# over non-negative integers.
DIM_VAR_GEQ = 0

class InconclusiveDimensionOperation(core.InconclusiveDimensionOperation):
  """Raised when we cannot conclusively compute with symbolic dimensions."""

  _help_msg = """
This error arises for comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a boolean value for all values of the symbolic dimensions involved.

Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
"""

  def __init__(self, message: str):
    error_msg = f"{message}\n{InconclusiveDimensionOperation._help_msg}"
    # https://github.com/python/mypy/issues/5887
    super().__init__(error_msg)  # type: ignore

class _ShapePolyThreadLocalState(threading.local):

  def __init__(self):
    # TODO(necula): this does not play well with some lowering caches, because
    # this state is not part of the cache key.
    self.enable_shape_assertions = True

thread_local_state = _ShapePolyThreadLocalState()

class _DimAtom:
  """Represents an atom in a symbolic dimension expression.

  Atoms are either variables, or expressions of the form floordiv(E1, E2) or
  mod(E1, E2). Atoms are multiplied to form monomials (see _DimMon), and
  monomials are added to form symbolic expressions (see _DimExpr).

  Args:
    * var: if specified then the atom is a dimension variable. `operation`
      must be `None`.
    * operation: if specified then the atom is an operation applied to
      `operands`. One of `FLOORDIR` or `MOD` or `NON_NEGATIVE`. `var` must be `None`
    * operands: the operands to which the operation is applied.
  """
  # The supported operations
  FLOORDIV = "floordiv"
  MOD = "mod"
  NON_NEGATIVE = "non_negative"  # The max of the operand and 0

  def __init__(self, *operands: '_DimExpr',
               var: Optional[str] = None,
               operation: Optional[str] = None):
    if var is not None:
      assert operation is None
      assert not operands
    else:
      assert operation is not None
    self.var = var
    self.operation = operation
    self.operands = operands

  @classmethod
  def from_var(cls, v: str) -> '_DimAtom':
    return _DimAtom(var=v)

  def to_var(self) -> Optional[str]:
    return self.var

  def get_vars(self) -> set[str]:
    # All the vars that appear
    if self.var is not None:
      return {self.var}
    else:
      acc = set()
      for opnd in self.operands:
        acc.update(opnd.get_vars())
      return acc

  @classmethod
  def from_operation(cls, operation: str, *operands: '_DimExpr') -> '_DimAtom':
    return _DimAtom(*operands, operation=operation)

  def __str__(self):
    if self.var is not None:
      return self.var
    opnd_str = ", ".join([str(opnd) for opnd in self.operands])
    return f"{self.operation}({opnd_str})"
  __repr__ = __str__

  def __hash__(self):
    return hash((self.var, self.operation, *self.operands))

  def __eq__(self, other: Any):
    # Used only for hashing
    if not isinstance(other, _DimAtom): return False
    if (self.var is None) != (other.var is None): return False
    if self.var is not None:
      return self.var == other.var
    else:
      def symbolic_equal(e1: '_DimExpr', e2: '_DimExpr') -> bool:
        try:
          return e1 == e2
        except InconclusiveDimensionOperation:
          return False
      return (self.operation == other.operation and
              all(symbolic_equal(self_o, other_o)
                  for self_o, other_o in zip(self.operands, other.operands)))

  def __lt__(self, other: '_DimAtom'):
    """
    Comparison to another atom in graded reverse lexicographic order.
    Used only for determining a sorting order, does not relate to the
    comparison of the values of the atom.
    """
    if self.var is not None and other.var is not None:
      return self.var < other.var
    elif self.var is not None:
      return True
    elif other.var is not None:
      return True
    elif self.operation != other.operation:
      return self.operation < other.operation  # type: ignore
    else:
      return id(self) < id(other)

  def bounds(self) -> tuple[float, float]:
    """Returns the lower and upper bounds, or -+ inf."""
    if self.var is not None:
      return (DIM_VAR_GEQ, np.PINF)  # variables are assumed to be >= 1
    opnd_bounds = [opnd.bounds() for opnd in self.operands]
    if self.operation == _DimAtom.FLOORDIV:  #  a // b
      (a_l, a_u), (b_l, b_u) = opnd_bounds
      def math_floor_with_inf(a: float, b: float):  # math.floor, but aware of inf
        assert b != 0
        if not np.isinf(b):  # divisor is finite
          return math.floor(a / b) if not np.isinf(a) else np.NINF if (a >= 0) != (b >= 0) else np.PINF
        elif not np.isinf(a):  # dividend is finite and divisor is infinite
          return -1 if (a >= 0) != (b >= 0) else 0
        else:  # both dividend and divisor are infinite
          return np.NINF if (a >= 0) != (b >= 0) else np.PINF

      # Same reasoning as for multiplication: the bounds are among the cross-product
      # of the bounds.
      bound_candidates = [math_floor_with_inf(a_l, b_l), math_floor_with_inf(a_l, b_u),
                          math_floor_with_inf(a_u, b_l), math_floor_with_inf(a_u, b_u)]
      return (min(*bound_candidates), max(*bound_candidates))

    elif self.operation == _DimAtom.MOD:
      _, (b_l, b_u) = opnd_bounds
      if b_l > 0:  # positive divisor
        return (0, b_u - 1)
      elif b_u < 0:  # negative divisor
        return (b_l + 1, 0)
      else:
        return (np.NINF, np.PINF)

    elif self.operation == _DimAtom.NON_NEGATIVE:
      (b_l, b_h), = opnd_bounds
      return (max(0, b_l), max(0, b_h))

    else:
      assert False

  def evaluate(self, env: DimVarEnv):
    if self.var is not None:
      try:
        return env[self.var]
      except KeyError:
        err_msg = (
            f"Encountered dimension variable '{self.var}' that is not appearing in the shapes of the used function arguments.\n"
            "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#dimension-variables-must-be-solvable-from-the-input-shapes for more details.")
        raise KeyError(err_msg)
    else:
      operand_values = [opnd.evaluate(env) for opnd in self.operands]
      if self.operation == _DimAtom.FLOORDIV:
        return divmod(*operand_values)[0]  # type: ignore
      elif self.operation == _DimAtom.MOD:
        return divmod(*operand_values)[1]  # type: ignore
      elif self.operation == _DimAtom.NON_NEGATIVE:
        return lax.max(operand_values[0], 0)
      else:
        assert False, self.operation

class _DimMon(dict):
  """Represents a multiplication of atoms.

  The representation is a dictionary mapping _DimAtom to exponent.
  The exponents are integers >= 1.
  """
  def __hash__(self):
    return hash(frozenset(self.items()))

  def __str__(self):
    return "*".join(f"{key}^{exponent}" if exponent != 1 else str(key)
                    for key, exponent in sorted(self.items()))

  @classmethod
  def from_var(cls, v: str) -> '_DimMon':
    return _DimMon({_DimAtom.from_var(v): 1})

  @classmethod
  def from_atom(clscls, a: _DimAtom, aexp: int):
    assert aexp >= 1
    return _DimMon({a: aexp})

  def to_var(self) -> Optional[str]:
    """Extract the variable name "x", from a monomial "x".
     Return None, if the monomial is not a single variable."""
    items = self.items()
    if len(items) != 1:
      return None
    (a, aexp), = items
    if aexp != 1:
      return None
    return a.to_var()

  def get_vars(self) -> set[str]:
    # All the vars that appear in the monomial
    acc = set()
    for a in self.keys():
      acc.update(a.get_vars())
    return acc

  @classmethod
  def from_operation(cls, operation: str, *operands: '_DimExpr') -> '_DimMon':
    return _DimMon({_DimAtom.from_operation(operation, *operands): 1})

  @property
  def degree(self):
    return sum(self.values())

  def __lt__(self, other: '_DimMon'):
    """
    Comparison to another monomial in graded reverse lexicographic order.
    Used only for determining a sorting order, does not relate to the
    comparison of the values of the monomial.
    """
    self_key = -self.degree, tuple(sorted(self))
    other_key = -other.degree, tuple(sorted(other))
    return self_key > other_key

  def mul(self, other: '_DimMon') -> '_DimMon':
    """
    Returns the product with another monomial. Example: (n^2*m) * n == n^3 * m.
    """
    return _DimMon(collections.Counter(self) + collections.Counter(other))

  def divide(self, divisor: '_DimMon') -> '_DimMon':
    """
    Divides by another monomial. Raises a InconclusiveDimensionOperation
    if the result is not a monomial.
    For example, (n^3 * m) // n == n^2*m, but n // m fails.
    """
    d = collections.Counter(self)
    for key, exponent in divisor.items():
      diff = self.get(key, 0) - exponent
      if diff < 0:
        raise InconclusiveDimensionOperation(f"Cannot divide {self} by {divisor}.")
      elif diff == 0: del d[key]
      elif diff > 0: d[key] = diff
    return _DimMon(d)

  def bounds(self) -> tuple[float, float]:
    """Returns the lower and upper bounds, or -+inf."""
    # The bounds of a product are among the product of bounds.
    bounds = []
    for a, exp in self.items():
      a_l, a_u = a.bounds()
      assert a_l <= a_u
      bounds.append((a_l ** exp, a_u ** exp))

    candidates = [math.prod(atom_bounds) for atom_bounds in itertools.product(*bounds)]
    return (min(*candidates), max(*candidates))  # type: ignore


  def evaluate(self, env: DimVarEnv):
    prod = lambda xs: functools.reduce(_evaluate_multiply, xs) if xs else core.dim_constant(1)
    def pow_opt(v, p: int):
      return v if p == 1 else prod([v] * p)
    return prod([pow_opt(a.evaluate(env), deg) for a, deg in self.items()])


class _DimExpr():
  """Symbolic expression in terms of dimension variables.

  A dimension expression is an addition of products (_DimMon)
  of atoms (_DimAtom).

  We overload integer operations, but we do that soundly, raising
  :class:`InconclusiveDimensionOperation` when the result is not
  representable as a _DimExpr.

  The representation of a _DimExpr is as a dictionary mapping _DimMon to
  integer coefficients. The special monomial `_DimMon()` is mapped to the
  free integer coefficient of the expression.
  """

  __array_priority__ = 1000   # Same as tracer, for __radd__ and others on ndarray
  def __init__(self, coeffs: dict[_DimMon, int]):
    # Do not construct _DimExpr directly, unless you are sure that coeffs is
    # normalized; Use _DimExpr.normalize.
    # Takes ownership of coeffs
    self._coeffs = coeffs or {_DimMon(): 0}

  def monomials(self) -> Iterable[tuple[_DimMon, int]]:
    return self._coeffs.items()

  @classmethod
  def _add_coeffs(cls, coeffs: dict[_DimMon, int], mon: _DimMon, coeff: int):
    """Do `coeffs[mon] += coeff` but remove 0 coefficients."""
    old_c = coeffs.get(mon)
    if old_c is None:
      if coeff != 0: coeffs[mon] = coeff
    else:
      new_c = old_c + coeff
      if new_c == 0:
        del coeffs[mon]
      else:
        coeffs[mon] = new_c

  @classmethod
  def normalize(cls, coeffs: dict[_DimMon, int]) -> DimSize:
    """The main constructor for _DimExpr.

    Ensures that the symbolic dimension is normalized, e.g.,
    it is represented as a Python int if it is known to be a constant.
    """
    # TODO(necula): profile and optimize this
    has_non_zero_degree = False
    free_const = 0
    new_coeffs: dict[_DimMon, int] = {}
    for mon, coeff in coeffs.items():
      if coeff == 0: continue
      if mon.degree == 0:  # A constant, there can be a single one
        free_const = coeff
      else:
        has_non_zero_degree = True

      new_coeffs[mon] = new_coeffs.get(mon, 0) + coeff

    if has_non_zero_degree:
      return _DimExpr(new_coeffs)
    else:
      return int(free_const)

  @classmethod
  def normalize_floordiv_times_divisor(cls, coeffs: dict[_DimMon, int]) -> DimSize:
    # Look for floordiv(E, M) * M and turn into E - mod(E, M). This comes
    # up when handling strided convolution.
    for dec in _decompose_expr(_DimExpr(coeffs), _DimAtom.FLOORDIV):
      # e = factor * floordiv(operands)^exp * rest_monomial + rest_expr
      if dec.exp != 1:
        continue
      if dec.rest_monomial == 1 and dec.factor == 1:
        continue
      m_trimmed, m_remainder = divmod(dec.factor * dec.rest_monomial, dec.operands[1])
      if m_remainder == 0:
        return m_trimmed * (dec.operands[0] - _DimExpr.from_operation(_DimAtom.MOD, *dec.operands)) + dec.rest_expr
    return _DimExpr.normalize(coeffs)

  @classmethod
  def from_monomial(cls, mon: _DimMon, coeff: int):
    return _DimExpr.normalize({mon: coeff})

  @classmethod
  def from_var(cls, v: str) -> '_DimExpr':
    return _DimExpr({_DimMon.from_var(v): 1})

  @classmethod
  def from_operation(cls, operation: str, *operands: '_DimExpr') -> '_DimExpr':
    return _DimExpr.from_monomial(_DimMon.from_operation(operation, *operands), 1)

  def to_var(self) -> Optional[str]:
    """Extract the variable name "x", from a symbolic expression."""
    items = self.monomials()
    if len(items) != 1:  # type: ignore
      return None
    (mon, mon_count), = items
    if mon_count != 1:
      return None
    return mon.to_var()

  def get_vars(self) -> set[str]:
    """The variables that appear in a symbolic dimension."""
    acc = set()
    for mon, _ in self.monomials():
      acc.update(mon.get_vars())
    return acc

  def eq(self, other: DimSize) -> bool:
    lb, ub = _ensure_poly(self - other, "eq").bounds()
    if lb == ub == 0:
      return True
    if lb > 0 or ub < 0:
      return False
    # See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#comparison-of-symbolic-dimensions-is-partially-supported
    return False

  def inconclusive_comparison(self, operation: str, op: Any) -> Exception:
    return InconclusiveDimensionOperation(
      f"Symbolic dimension comparison '{self}' {operation} '{op}' is inconclusive.\n"
      "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#comparison-of-symbolic-dimensions-is-partially-supported.")

  def ge(self, other: DimSize) -> bool:
    lb, ub = _ensure_poly(self - other, "ge").bounds()
    if lb >= 0:
      return True
    if ub < 0:
      return False
    raise self.inconclusive_comparison(">=", other)

  def __hash__(self):
    return hash(tuple(sorted(self.monomials())))

  def __str__(self):
    def _one_monomial(mon, c):
      if mon.degree == 0:
        return str(c)
      if c == 1:
        return str(mon)
      return f"{c}*{mon}"
    return " + ".join(_one_monomial(mon, c)
                      for mon, c in sorted(self.monomials(), reverse=True))

  def __repr__(self):
    return str(self)

  # We overload +, -, *, because they are fully defined for _DimExpr.
  def __add__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__add__(other)

    other = _ensure_poly(other, "add")
    coeffs = self._coeffs.copy()
    for mon, coeff in other.monomials():
      _DimExpr._add_coeffs(coeffs, mon, coeff)
    return _DimExpr.normalize_floordiv_times_divisor(coeffs)

  def __radd__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__radd__(other)
    return _ensure_poly(other, "add").__add__(self)

  def __sub__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__sub__(other)
    return self + -_ensure_poly(other, "sub")

  def __rsub__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__rsub__(other)
    return _ensure_poly(other, "sub").__sub__(self)

  def __neg__(self) -> '_DimExpr':
    return _DimExpr({mon: -coeff for mon, coeff in self.monomials()})

  def __mul__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__mul__(other)
    other = _ensure_poly(other, "mul")
    coeffs: dict[_DimMon, int] = {}
    for mon1, coeff1 in self.monomials():
      for mon2, coeff2 in other.monomials():
        mon = mon1.mul(mon2)
        _DimExpr._add_coeffs(coeffs, mon, coeff1 * coeff2)
    return _DimExpr.normalize_floordiv_times_divisor(coeffs)

  def __rmul__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__rmul__(other)
    return _ensure_poly(other, "mul").__mul__(self)

  def __pow__(self, power, modulo=None):
    assert modulo is None
    try:
      power = int(power)
    except:
      raise InconclusiveDimensionOperation(f"Symblic dimension cannot be raised to non-integer power '{self}' ^ '{power}'")
    return functools.reduce(op.mul, [self] * power)

  def __floordiv__(self, divisor):
    if isinstance(divisor, core.Tracer) or not _convertible_to_poly(divisor):
      return self.__jax_array__().__floordiv__(divisor)
    return self.divmod(_ensure_poly(divisor, "floordiv"))[0]

  def __rfloordiv__(self, other):
    if isinstance(other, core.Tracer) or not _convertible_to_poly(other):
      return self.__jax_array__().__rfloordiv__(other)
    return _ensure_poly(other, "floordiv").__floordiv__(self)

  def __truediv__(self, divisor):
    # Used for "/", which always returns a float
    return self.__jax_array__().__truediv__(divisor)

  def __rtruediv__(self, dividend):
    # Used for "/", when dividend is not a _DimExpr
    return self.__jax_array__().__rtruediv__(dividend)

  def __mod__(self, divisor):
    if isinstance(divisor, core.Tracer) or not _convertible_to_poly(divisor):
      return self.__jax_array__().__mod__(divisor)
    return self.divmod(_ensure_poly(divisor, "mod"))[1]

  def __rmod__(self, dividend):
    if isinstance(dividend, core.Tracer) or not _convertible_to_poly(dividend):
      return self.__jax_array__().__rmod__(dividend)
    return _ensure_poly(dividend, "mod").__mod__(self)

  def __divmod__(self, divisor):
    if isinstance(divisor, core.Tracer) or not _convertible_to_poly(divisor):
      return self.__jax_array__().__divmod__(divisor)
    return self.divmod(_ensure_poly(divisor, "divmod"))

  def __rdivmod__(self, dividend):
    if isinstance(dividend, core.Tracer) or not _convertible_to_poly(dividend):
      return self.__jax_array__().__rdivmod__(dividend)
    return _ensure_poly(dividend, "divmod").__divmod__(self)

  def __int__(self):
    if self.is_constant:
      return op.index(next(iter(self._coeffs.values())))
    else:
      raise InconclusiveDimensionOperation(f"Symbolic dimension '{self}' used in a context that requires a constant")

  # We must overload __eq__ and __ne__, or else we get unsound defaults.
  __eq__ = eq
  def __ne__(self, other: DimSize) -> bool:
    return not self.eq(other)

  __ge__ = ge

  def __le__(self, other: DimSize):
    try:
      return _ensure_poly(other, "le").__ge__(self)
    except InconclusiveDimensionOperation as e:
      raise self.inconclusive_comparison("<=", other) from e

  def __gt__(self, other: DimSize):
    try:
      return not _ensure_poly(other, "gt").__ge__(self)
    except InconclusiveDimensionOperation as e:
      raise self.inconclusive_comparison(">", other) from e

  def __lt__(self, other: DimSize):
    try:
      return not self.__ge__(other)
    except InconclusiveDimensionOperation as e:
      raise self.inconclusive_comparison("<", other) from e

  def divmod(self, divisor: "_DimExpr") -> tuple[DimSize, int]:
    """
    Floor division with remainder (divmod) generalized to polynomials.
    If the `divisor` is not a constant, the remainder must be 0.
    If the `divisor` is a constant, the remainder may be non 0, for consistency
    with integer divmod.

    :return: Quotient resulting from polynomial division and integer remainder.
    """
    assert isinstance(divisor, _DimExpr)
    try:
      dmon, dcount = divisor.leading_term
      dividend, quotient = self, 0
      # invariant: self = dividend + divisor * quotient
      # quotient and dividend are changed in the loop; the leading term of
      # dividend decreases at each iteration.
      while is_poly_dim(dividend) and not dividend.is_constant:
        mon, count = dividend.leading_term
        try:
          qmon = mon.divide(dmon)
        except InconclusiveDimensionOperation:
          raise InconclusiveDimensionOperation("")
        qcount, rcount = divmod(count, dcount)
        if rcount != 0:
          raise InconclusiveDimensionOperation("")

        q = _DimExpr.from_monomial(qmon, qcount)
        quotient += q
        dividend -= q * divisor  # type: ignore[assignment]

      dividend = int(dividend)  # type: ignore[assignment]
      if divisor.is_constant:
        q, r = divmod(dividend, int(divisor))  # type: ignore
        quotient += q
        remainder = r
      else:
        if dividend != 0:
          raise InconclusiveDimensionOperation("")
        remainder = 0

      if config.jax_enable_checks:
        assert self == divisor * quotient + remainder
      return quotient, remainder
    except InconclusiveDimensionOperation:
      return (_DimExpr.from_operation(_DimAtom.FLOORDIV, self, divisor),  # type: ignore
              _DimExpr.from_operation(_DimAtom.MOD, self, divisor))

  def bounds(self) -> tuple[float, float]:
    """Returns the lower and upper bounds, or -+inf."""
    lb = ub = self._coeffs.get(_DimMon(), 0)  # The free coefficient
    for mon, coeff in self.monomials():
      if mon.degree == 0: continue  # We already included the free coefficient
      m_l, m_u = mon.bounds()
      assert m_l <= m_u and coeff != 0
      item_l, item_u = coeff * m_l, coeff * m_u
      lb = lb + min(item_l, item_u)  # type: ignore
      ub = ub + max(item_l, item_u)  # type: ignore

    if lb != np.NINF or ub != np.PINF:
      return lb, ub
    # Watch for special-case: ct*a - ct*mod(b, a) >= 1 when ct >= 0 and a >= 0
    # TODO(necula): add more principled support for floordiv and mod
    # For example, this will miss "1 + a - mod(b, a)"
    for dec in _decompose_expr(self, _DimAtom.MOD):
      # E = factor*mod(op1, op2)^exp * rest_monomial + rest_expr
      if dec.exp == 1 and dec.rest_monomial == 1 and dec.rest_expr == - dec.factor * dec.operands[1]:
        try:
          if dec.operands[1] <= 0:
            continue
        except InconclusiveDimensionOperation:
          continue
        if dec.factor > 0:
          return (np.NINF, -1)
        else:
          return (1, np.PINF)

    return lb, ub

  @property
  def is_constant(self):
    return len(self._coeffs) == 1 and next(iter(self._coeffs)).degree == 0

  @property
  def leading_term(self) -> tuple[_DimMon, int]:
    """Returns the highest degree term that comes first lexicographically."""
    return max(self.monomials())

  def evaluate(self, env: DimVarEnv):
    # Evaluates as a value of dtype=core.dim_value_dtype()
    terms = [_evaluate_multiply(mon.evaluate(env), core.dim_constant(coeff))
             for mon, coeff in self.monomials()]
    return functools.reduce(_evaluate_add, terms) if len(terms) > 1 else terms[0]

  def non_negative(self) -> "_DimExpr":
    return _DimExpr.from_operation(_DimAtom.NON_NEGATIVE, self)

  @staticmethod
  def get_aval(dim: "_DimExpr"):
    return core.dim_value_aval()

  def dimension_as_value(self):
    """Turns a dimension size into a Jax value that we can compute with."""
    return _dim_as_value(self)

  def __jax_array__(self):
    # Used for implicit coercions of polynomials as JAX arrays
    return _dim_as_value(self)

@dataclasses.dataclass
class _Decomposition:
  """Decomposition of an expression around an operation atom.

  E = factor * mod(*operands)^exp * rest_monomial + rest_expr
  """
  factor: int
  operands: Sequence[_DimExpr]
  exp: int
  rest_monomial: _DimExpr
  rest_expr: _DimExpr


def _decompose_expr(e: _DimExpr, operation: str) -> Iterable[_Decomposition]:
  for m, m_factor in e.monomials():
    atoms = [(a, aexp) for a, aexp in m.items() if a.operation == operation]
    if atoms:
      e_minus_m_coeffs = e._coeffs.copy()
      del e_minus_m_coeffs[m]
    for a, aexp in atoms:
      yield _Decomposition(
          factor=m_factor,
          operands=a.operands,
          exp=aexp,
          rest_monomial=_DimExpr({m.divide(_DimMon.from_atom(a, aexp)): 1}),
          rest_expr=_DimExpr(e_minus_m_coeffs))

core.pytype_aval_mappings[_DimExpr] = _DimExpr.get_aval
xla.pytype_aval_mappings[_DimExpr] = _DimExpr.get_aval
dtypes._weak_types.append(_DimExpr)

def _convertible_to_int(p: DimSize) -> bool:
  try:
    op.index(p)
    return True
  except:
    return False

def _ensure_poly(p: DimSize,
                 operation_name: str) -> _DimExpr:
  if isinstance(p, _DimExpr): return p
  if _convertible_to_int(p):
    return _DimExpr({_DimMon(): op.index(p)})
  raise TypeError(f"Symnbolic dimension {operation_name} not supported for {p}.")

def _convertible_to_poly(p: DimSize) -> bool:
  return isinstance(p, _DimExpr) or _convertible_to_int(p)

def is_poly_dim(p: DimSize) -> bool:
  return isinstance(p, _DimExpr)

dtypes.python_scalar_dtypes[_DimExpr] = dtypes.python_scalar_dtypes[int]

def _einsum_contract_path(*operands, **kwargs):
  """Like opt_einsum.contract_path, with support for DimExpr shapes.

  We use opt_einsum.contract_path to compute the schedule, using a fixed
  constant for all dimension variables. This is safe because we throw an
  error if there are more than 1 contractions. Essentially, we just use
  opt_einsum.contract_path to parse the specification.
  """

  # Replace the polymorphic shapes with some concrete shapes for calling
  # into opt_einsum.contract_path, because the latter wants to compute the
  # sizes of operands and intermediate results.
  fake_ops = []
  for operand in operands:
    # We replace only array operands
    if not hasattr(operand, "dtype"):
      fake_ops.append(operand)
    else:
      shape = np.shape(operand)
      def fake_dim(d):
        if core.is_constant_dim(d):
          return d
        else:
          if not isinstance(d, _DimExpr):
            raise TypeError(f"Encountered unexpected shape dimension {d}")
          # It is Ok to replace all polynomials with the same value. We may miss
          # here some errors due to non-equal dimensions, but we catch them
          # later.
          return 8
      fake_ops.append(jax.ShapeDtypeStruct(tuple(map(fake_dim, shape)),
                                           operand.dtype))

  contract_fake_ops, contractions = opt_einsum.contract_path(*fake_ops,
                                                             **kwargs)
  contract_operands = []
  for operand in contract_fake_ops:
    idx = tuple(i for i, fake_op in enumerate(fake_ops) if operand is fake_op)
    assert len(idx) == 1
    contract_operands.append(operands[idx[0]])
  return contract_operands, contractions

lax_numpy._poly_einsum_handlers[_DimExpr] = _einsum_contract_path

# To implement shape-constraint checking we use a shape assertion primitive.
#    shape_assertion_p.bind(assert_what: bool, *error_message_inputs,
#                           error_message="...{0}...{1}")
# where "{0}" refers to error_message_inputs[0], etc.
shape_assertion_p = core.Primitive("shape_assertion")
shape_assertion_p.multiple_results = True
shape_assertion_p.def_effectful_abstract_eval(
  lambda *_, **__: ((), {shape_assertion_effect}))  # type: ignore

def _shape_assertion_lowering_rule(ctx: mlir.LoweringRuleContext,
                                   assert_what: mlir.ir.Value,
                                   *error_message_inputs: mlir.ir.Value,
                                   error_message: str):
  op = mlir.custom_call(
    "shape_assertion",
    [],  # No results
    [assert_what, *error_message_inputs],
    has_side_effect=True,
    extra_attributes=dict(error_message=mlir.ir.StringAttr.get(error_message))
  )
  return op.results

mlir.register_lowering(shape_assertion_p, _shape_assertion_lowering_rule)

class ShapeAssertionEffect(effects.Effect):
  __str__ = lambda _: "ShapeAssertionEffect"

shape_assertion_effect = ShapeAssertionEffect()

effects.lowerable_effects.add_type(ShapeAssertionEffect)
effects.control_flow_allowed_effects.add_type(ShapeAssertionEffect)
effects.remat_allowed_effects.add_type(ShapeAssertionEffect)
effects.custom_derivatives_allowed_effects.add_type(ShapeAssertionEffect)

def shape_assertion(assert_what: jax.Array,
                    *error_message_inputs: jax.Array,
                    error_message: str) -> None:
  """Adds a shape assertion in the code.

  Args:
    assert_what: a boolean asserted to be true. Must be computed based only
      on dimension expressions, so that it can be evaluated after shape
      refinement.
    error_message_inputs: integers expressions whose values can be referenced
      in the `error_message`. Must be computed based only
      on dimension expressions, so that they can be evaluated after shape
      refinement.
    error_message: an error message, possibly containing format specifiers
      {0}, {1}, ..., referencing the values of the `error_message_inputs`.
      The format specifiers are sometimes processed with Python's
      `string::format` method, and sometimes with `llvm::formatv`.
  """
  if thread_local_state.enable_shape_assertions:
    shape_assertion_p.bind(assert_what, *error_message_inputs,
                           error_message=error_message)

# A JAX primitive with no array arguments but with a dimension parameter
# that is a DimExpr. The value of the primitive is the value of the dimension,
# using int64 in x64 mode or int32 otherwise (core.dim_value_dtype())
dim_as_value_p = core.Primitive("dim_as_value")
dim_as_value_p.def_abstract_eval(lambda dim: core.dim_value_aval())

def dim_as_value_impl(dim: DimSize):
  raise NotImplementedError(
      "Evaluation rule for 'dim_as_value' is not implemented. "
      "It seems that you are using shape polymorphism outside jax2tf.")

dim_as_value_p.def_impl(dim_as_value_impl)
def _dim_as_value(dim: DimSize):
  return dim_as_value_p.bind(dim=dim)

def _dim_as_value_lowering(ctx: mlir.LoweringRuleContext, *,
                           dim):
  res, = mlir.eval_dynamic_shape(ctx, (dim,))
  out_type = mlir.aval_to_ir_type(ctx.avals_out[0])
  if out_type != res.type:  # type: ignore
    return mlir.hlo.ConvertOp(out_type, res).results
  else:
    return [res]

mlir.register_lowering(dim_as_value_p, _dim_as_value_lowering)


class PolyShape(tuple):
  """Tuple of polymorphic dimension specifications.

  See docstring of :func:`jax2tf.convert`.
  """

  def __init__(self, *dim_specs):
    tuple.__init__(dim_specs)

  def __new__(cls, *dim_specs):
    for ds in dim_specs:
      if not isinstance(ds, (int, str)) and ds != ...:
        msg = (f"Invalid polymorphic shape element: {repr(ds)}; must be a string "
               "representing a dimension variable, or an integer, or ...")
        raise ValueError(msg)
    return tuple.__new__(PolyShape, dim_specs)

  def __str__(self):
    return "(" + ", ".join(["..." if d is ... else str(d) for d in self]) + ")"


def _parse_spec(shape_spec: Union[str, PolyShape, None],
                arg_shape: Sequence[Optional[int]]) -> Sequence[DimSize]:
  """Parses the shape polymorphic specification for one array argument.

  We have to be able to parse all strings produced by str(_DimExpr) because
  sometimes the output polymorphic shapes of one function become the input
  polymorphic shapes of another.

  Args:
    shape_spec: a shape polymorphic specification. None stands for "...".
    arg_shape: an actual shape, possibly containing unknown dimensions (None).
      We use `arg_shape` to fill-in the placeholders `_` and `...` in
      the `shape_spec`. The dimensions of `arg_shape` that are used for filling
      must be known (not `None`). If a dimension in `arg_shape` is known and
      the corresponding dimension in `shape_spec` is a constant then they
      must be equal.

  See the README.md for usage.
  """
  shape_spec_repr = repr(shape_spec)
  if shape_spec is None:
    shape_spec = "..."
  elif isinstance(shape_spec, PolyShape):
    shape_spec = str(shape_spec)
  elif not isinstance(shape_spec, str):
    raise ValueError("polymorphic shape spec should be None or a string. "
                     f"Found {shape_spec_repr}.")
  return _Parser(shape_spec, arg_shape, shape_spec_repr).parse()

class _Parser:
  def __init__(self,
               shape_spec: str,
               arg_shape: Sequence[Optional[int]],
               shape_spec_repr: str):
    self.shape_spec = shape_spec
    self.shape_spec_repr = shape_spec_repr  # For error messages
    self.arg_shape = arg_shape
    self.dimensions: list[DimSize] = []  # dimensions we have parsed

  def parse(self) -> Sequence[DimSize]:
    self.tokstream = tokenize.tokenize(
        io.BytesIO(self.shape_spec.encode("utf-8")).readline)
    tok = self.consume_token(self.next_tok(), tokenize.ENCODING)  # Always 1st
    sh, tok = self.shape(tok)
    self.expect_token(tok, [tokenize.ENDMARKER])
    return sh

  def add_dim(self, expr: Optional[DimSize], tok: tokenize.TokenInfo):
    if expr is None:
      raise self.parse_err(tok,
                           ("unexpected placeholder for unknown dimension "
                            f"for argument shape {self.arg_shape}"))
    arg_shape_dim = self.arg_shape[len(self.dimensions)]
    if core.is_constant_dim(expr) and arg_shape_dim is not None:
      if expr != arg_shape_dim:
        raise self.parse_err(tok,
                             (f"different size {expr} for known dimension "
                              f"for argument shape {self.arg_shape}"))
    self.dimensions.append(expr)

  def parse_err(self, tok: Optional[tokenize.TokenInfo], detail: str) -> Exception:
    msg = (
        f"syntax error in polymorphic shape {self.shape_spec_repr} "
        f"in dimension {len(self.dimensions)}: {detail}. ")
    if tok is not None:
      msg += f"Parsed '{tok.line[:tok.start[1]]}', remaining '{tok.line[tok.start[1]:]}'."
    return ValueError(msg)

  def next_tok(self) -> tokenize.TokenInfo:
    while True:
      try:
        t = next(self.tokstream)
      except StopIteration:
        raise self.parse_err(None, "unexpected end of string")
      if t.exact_type not in [tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT]:
        return t

  def expect_token(self, tok: tokenize.TokenInfo, expected: Sequence[int]) -> None:
    if tok.exact_type not in expected:
      msg = ("expecting one of {" +
             ", ".join(tokenize.tok_name[t] for t in expected) + "} but found " +
             tokenize.tok_name[tok.exact_type])
      raise self.parse_err(tok, msg)

  def consume_token(self, tok: tokenize.TokenInfo, expected: int) -> tokenize.TokenInfo:
    self.expect_token(tok, [expected])
    return self.next_tok()

  def integer(self, tok: tokenize.TokenInfo) -> tuple[int, tokenize.TokenInfo]:
    self.expect_token(tok, [tokenize.NUMBER])
    try:
      val = int(tok.string)
    except Exception:
      raise self.parse_err(tok, f"expecting integer, found {tok.string}")
    return val, self.next_tok()

  # What can follow a shape?
  FOLLOW_SHAPE = [tokenize.ENDMARKER, tokenize.RPAR]
  def shape(self, tok: tokenize.TokenInfo) -> tuple[Sequence[DimSize], tokenize.TokenInfo]:
    # A comma-separated list of _DimExpr, or "_", possibly ended with ...
    if tok.exact_type == tokenize.LPAR:
      res, tok = self.shape(self.next_tok())
      tok = self.consume_token(tok, tokenize.RPAR)
      return res, tok

    while True:
      if tok.exact_type in self.FOLLOW_SHAPE:
        break
      if tok.exact_type == tokenize.ELLIPSIS:
        to_add = self.arg_shape[len(self.dimensions):]
        for ad in to_add:
          self.add_dim(ad, tok)
        tok = self.next_tok()
        break
      if len(self.dimensions) >= len(self.arg_shape):
        raise self.parse_err(tok,
            f"too many dimensions, arg_shape has {len(self.arg_shape)}")
      if tok.exact_type == tokenize.NAME and tok.string == "_":
        e = self.arg_shape[len(self.dimensions)]
        tok = self.next_tok()
      else:
        e, tok = self.expr(tok)
      self.add_dim(e, tok)
      if tok.exact_type in self.FOLLOW_SHAPE:
        break
      tok = self.consume_token(tok, tokenize.COMMA)

    return tuple(self.dimensions), tok

  # What token can follow a _DimExpr
  FOLLOW_EXPR = FOLLOW_SHAPE + [tokenize.COMMA]

  def expr(self, tok: tokenize.TokenInfo) -> tuple[DimSize, tokenize.TokenInfo]:
    # A sum of monomials
    next_m_negated = False
    acc = 0
    while True:
      m, tok = self.mon(tok)
      acc = acc + (- m if next_m_negated else m)
      if tok.exact_type in self.FOLLOW_EXPR:
        return acc, tok
      next_m_negated = (tok.exact_type == tokenize.MINUS)
      self.expect_token(tok, [tokenize.PLUS, tokenize.MINUS])
      tok = self.next_tok()

  FOLLOW_MON = FOLLOW_EXPR + [tokenize.PLUS, tokenize.MINUS]
  def mon(self, tok: tokenize.TokenInfo) -> tuple[DimSize, tokenize.TokenInfo]:
    # A monomial is product of atoms. Each atom may be raised to an integer power.
    acc = 1
    while True:
      a, tok = self.atom(tok)
      if tok.exact_type == tokenize.CIRCUMFLEX:
        tok = self.next_tok()
        self.expect_token(tok, [tokenize.NUMBER])
        power, tok = self.integer(tok)
        a = a ** power

      acc = acc * a
      if tok.exact_type in self.FOLLOW_MON:
        return acc, tok
      tok = self.consume_token(tok, tokenize.STAR)

  def atom(self, tok: tokenize.TokenInfo) -> tuple[DimSize, tokenize.TokenInfo]:
    if tok.exact_type == tokenize.NAME:
      if tok.string == _DimAtom.MOD:
        return self.binary_op(_DimAtom.MOD, self.next_tok())
      if tok.string == _DimAtom.FLOORDIV:
        return self.binary_op(_DimAtom.FLOORDIV, self.next_tok())
      if tok.string == _DimAtom.NON_NEGATIVE:
        return self.unary_op(_DimAtom.NON_NEGATIVE, self.next_tok())
      return _DimExpr.from_var(tok.string), self.next_tok()
    number_sign = 1
    if tok.exact_type == tokenize.MINUS:  # -k are negative constants
      number_sign = -1
      tok = self.next_tok()
      self.expect_token(tok, [tokenize.NUMBER])
    if tok.exact_type == tokenize.NUMBER:
      v, tok = self.integer(tok)
      return v * number_sign, tok
    self.expect_token(tok, [tokenize.NAME, tokenize.MINUS, tokenize.NUMBER])
    assert False

  def unary_op(self, op: str, tok) -> tuple[DimSize, tokenize.TokenInfo]:
    tok = self.consume_token(tok, tokenize.LPAR)
    e1, tok = self.expr(tok)
    tok = self.consume_token(tok, tokenize.RPAR)
    return _DimExpr.from_operation(op, e1), tok  # type: ignore

  def binary_op(self, op: str, tok) -> tuple[DimSize, tokenize.TokenInfo]:
    tok = self.consume_token(tok, tokenize.LPAR)
    e1, tok = self.expr(tok)
    tok = self.consume_token(tok, tokenize.COMMA)
    e2, tok = self.expr(tok)
    tok = self.consume_token(tok, tokenize.RPAR)
    return _DimExpr.from_operation(op, e1, e2), tok  # type: ignore


def _evaluate_add(v1, v2):
  try:
    if op.index(v1) == 0:
      return v2
  except:
    pass
  try:
    if op.index(v2) == 0:
      return v1
  except:
    pass
  return v1 + v2

def _evaluate_multiply(v1, v2):
  try:
    if op.index(v1) == 1:
      return v2
  except:
    pass
  try:
    if op.index(v2) == 1:
      return v1
  except:
    pass
  return v1 * v2

# dimension_size(operand, dimension=i) get the operand.shape[i] as a
# value of type shape_poly.dim_as_value_dtype().
dimension_size_p = core.Primitive("dimension_size")
def _dimension_size_abstract_eval(aval: core.AbstractValue, **_) -> core.AbstractValue:
  return core.dim_value_aval()

dimension_size_p.def_abstract_eval(_dimension_size_abstract_eval)

def _dimension_size_impl(arg, *, dimension):
  return core.dim_constant(arg.shape[dimension])
dimension_size_p.def_impl(_dimension_size_impl)

def _dimension_size_lowering_rule(ctx, arg, *, dimension):
  dim_size = mlir.hlo.GetDimensionSizeOp(arg, dimension)
  dim_type = mlir.aval_to_ir_type(core.dim_value_aval())
  if dim_size.result.type != dim_type:
    dim_size = mlir.hlo.ConvertOp(dim_type, dim_size)
  return dim_size.results

mlir.register_lowering(dimension_size_p, _dimension_size_lowering_rule)


def arg_aval(
    arg_shape: Sequence[Optional[int]],
    arg_jax_dtype: DType,
    polymorphic_shape: Optional[Union[str, PolyShape]]) -> core.ShapedArray:
  """Computes abstract values.

  Args:
    arg_shape: the shape for the argument, possibly having None dimensions.
    arg_dtype: the inferred JAX dtype for the arg.
    polymorphic_shape: the polymorphic specification for the argument.
  Returns: the JAX abstract value for the argument.
  """
  aval_shape = _parse_spec(polymorphic_shape, arg_shape)
  return core.ShapedArray(aval_shape, arg_jax_dtype)

def all_dim_vars(args_avals: Sequence[core.AbstractValue]) -> Sequence[str]:
  dim_vars: set[str] = set()
  for a in args_avals:
    for d in a.shape:
      if is_poly_dim(d):
        dim_vars = dim_vars.union(d.get_vars())
  return sorted(tuple(dim_vars))


@dataclasses.dataclass(frozen=True)
class ShapeConstraint:
  class Comparator(Enum):
    EQ = 1
    GEQ = 2

  comp: Comparator
  left: DimSize
  right: DimSize
  # make_err_msg lazily produces an error message that may refer to the
  # two operands being compared.
  make_err_msg: Callable[[Any, Any], str]

  def _eval_operand(self, o: DimSize, shapeenv: DimVarEnv,
                    ) -> Union[int, jax.Array]:
    if core.is_constant_dim(o):
      res = op.index(o)
    else:
      res = o.evaluate(shapeenv)  # type: ignore
    # Ensure we have np.int32
    res_dtype = dtypes.dtype(res)
    if res_dtype == np.int32 or is_poly_dim(res):
      return res
    if type(res) in dtypes.python_scalar_dtypes:
      return dtypes.coerce_to_array(res, np.int32)  # type: ignore
    else:
      return res.astype(np.int32)

  def _err_msg(self, left, right):
    return self.make_err_msg(str(left), str(right))

  def check_statically(self, shapeenv: DimVarEnv) -> None:
    """Evaluates a constraint statically.

    The `shapeenv` maps variables to _DimExpr. If the static checking
    of the constraint fails, raise ValueError.
    """
    left, right = (self._eval_operand(o, shapeenv) for o in [self.left,
                                                             self.right])
    try:
      if self.comp == ShapeConstraint.Comparator.EQ:
        ok = (left == right)
      elif self.comp == ShapeConstraint.Comparator.GEQ:
        ok = (left >= right)
      else: assert False
    except InconclusiveDimensionOperation as e:
      raise ValueError(self.make_err_msg(left, right)) from e
    if not ok:
      raise ValueError(self.make_err_msg(left, right))

  def compute(self,
              shapeenv: DimVarEnv
              ) -> Optional[tuple[jax.Array, jax.Array, jax.Array]]:
    """Computes if the constraint is satisfied.

    If the constraint can be resolved statically returns None
    or raises ValueError otherwise. If the constraint cannot be
    resolved statically, returns a triple with a boolean encoding if the
    constraint is satisfied and the int32 operands of binary constraints.
    """
    left, right = (self._eval_operand(o, shapeenv) for o in [self.left,
                                                             self.right])
    # Try to evaluate the constraint statically.
    if core.is_constant_shape((left, right)):
      left_int, right_int = op.index(left), op.index(right)
      if self.comp == ShapeConstraint.Comparator.EQ:
        if not (left_int == right_int):
          raise ValueError(self.make_err_msg(left_int, right_int))
      elif self.comp == ShapeConstraint.Comparator.GEQ:
        if not (left_int >= right_int):
          raise ValueError(self.make_err_msg(left_int, right_int))
      else: assert False
      return None

    if self.comp == ShapeConstraint.Comparator.EQ:
      is_ok = lax.eq(left, right)
    elif self.comp == ShapeConstraint.Comparator.GEQ:
      is_ok = lax.ge(left, right)
    else: assert False
    return is_ok, left, right  # type: ignore

  def __str__(self):
    return (f"{self.left} {'==' if self.comp == ShapeConstraint.Comparator.EQ else '>='} {self.right}"
            f" ({self.make_err_msg('%1', '%2')})")
  __repr__ = __str__


class ShapeConstraints:
  def __init__(self):
    self.constraints: list[ShapeConstraint] = []

  def add_constraint(self,
                     comp: ShapeConstraint.Comparator,
                     left: DimSize, right: DimSize,
                     make_err_msg: Callable[[Any, Any], str]):
    # Try to evaluate it statically
    c = ShapeConstraint(comp, left, right, make_err_msg)
    self.constraints.append(c)

  def check_statically(self, shapeenv: DimVarEnv) -> None:
    """Evaluates all the constraints statically.

    The `shapeenv` maps variables to _DimExpr. If the static checking
    of any constraint fails, raise ValueError.
    """
    for constraint in self.constraints:
      constraint.check_statically(shapeenv)

  def shape_assertions(self, shapeenv: DimVarEnv) -> None:
    """Computes the shape assertions the set of constraints.

    See _wrap_main_func docstring.
    """
    # We want to report the errors in the same order as `check_statically`.
    # So, we process them in order, in case some fail statically, and we
    # generate the shape assertions in the same order.
    for constraint in self.constraints:
      check_res = constraint.compute(shapeenv)
      if check_res is not None:
        is_ok, op1, op2 = check_res
        shape_assertion(
          is_ok, op1, op2,
          error_message=constraint.make_err_msg("{0}", "{1}"))

@dataclasses.dataclass
class _DimEquation:
  # Represents dim_expr == dim_value, where `dim_expr` contain unknown dimension
  # variables, in terms of `dim_value`.
  dim_expr: _DimExpr
  dim_value: _DimExpr

  def __str__(self):
    return f"{self.dim_expr} == {self.dim_value}"
  __repr__ = __str__


def args_kwargs_path_to_str(path: tree_util.KeyPath) -> str:
  # String description of `args` or `kwargs`, assuming the path for a tree for
  # the tuple `(args, kwargs)`.
  if path[0] == tree_util.SequenceKey(0):
    return f"args{tree_util.keystr(path[1:])}"
  elif path[0] == tree_util.SequenceKey(1):
    return f"kwargs{tree_util.keystr(path[1:])}"
  else:
    assert False

def pretty_print_dimension_descriptor(
    args_kwargs_tree: tree_util.PyTreeDef,
    flat_arg_idx: int, dim_idx: Optional[int]) -> str:
  args_kwargs_with_paths, _ = tree_util.tree_flatten_with_path(
      args_kwargs_tree.unflatten((0,) * args_kwargs_tree.num_leaves))
  arg_str = args_kwargs_path_to_str(args_kwargs_with_paths[flat_arg_idx][0])
  if dim_idx is not None:
    arg_str += f".shape[{dim_idx}]"
  return arg_str

@util.cache()
def solve_dim_vars(
    args_avals: Sequence[core.AbstractValue],
    args_kwargs_tree: tree_util.PyTreeDef,
    ) -> tuple[DimVarEnv, ShapeConstraints, Sequence[tuple[str, int, int]]]:
  """Solves dimension variables in a called function's avals in terms of actual argument shapes.

  For example, given:

     args_avals = [ShapedArray((3, a, a + b), f32)]

  we introduce fresh "synthetic" dimension variables to represent the actual
  dimension size of actual arguments for each non-constant dimension.
  Each synthetic variable has a name, an arg_idx, and a dim_idx, e.g.:

    synthetic_vars = [("args[0].shape[1]", 0, 1), ("args[0].shape[2]", 0, 2)]

  and then we express the solution for the unknown dimension variables {a, b}
  as symbolic expressions in terms of the synthetic variables:

    dict(a=args[0].shape[1], b=args[0].shape[2] - args[0].shape[1])

  Not all equations are solvable. For now, we solve first the linear
  uni-variate equations, then the solved variables are used to simplify the
  remaining equations to linear uni-variate equations, and the process
  continues until all dimension variables are solved.

  Args:
    args_avals: the abstract values of the `args`, with shapes that may
      include unknown dimension variables.
    args_kwargs_tree: a PyTreeDef that describes the tuple `(args, kwargs)`
      from which the flat sequence `args_avals` is extracted. Used for
      describing args and kwargs in synthetic variable names and in
      error messages.

  Returns: a 3-tuple with: (a) the solution for the unknown dimension variables
   (b) a list of constraints that must be satisfied for the solution to be a
   valid one, and (c) and the list of synthetic variables that may appear in
   the solution and the constraints.

  Raises ValueError if it cannot solve some dimension variable.
  """
  dim_equations: list[_DimEquation] = []
  synth_dimension_vars: list[tuple[str, int, int]] = []
  for arg_idx, aval in enumerate(args_avals):
    for dim_idx, aval_d in enumerate(aval.shape):
      if is_poly_dim(aval_d):
        synth_dim_var = pretty_print_dimension_descriptor(args_kwargs_tree,
                                                          arg_idx, dim_idx)
        synth_dimension_vars.append((synth_dim_var, arg_idx, dim_idx))
        dim_equations.append(
            _DimEquation(dim_expr=_ensure_poly(aval_d, "solve_dim_vars"),
                         dim_value=_DimExpr.from_var(synth_dim_var)))

  solution, shape_constraints = _solve_dim_equations(dim_equations)
  return solution, shape_constraints, synth_dimension_vars


def compute_dim_vars_from_arg_shapes(
    args_avals: Sequence[core.AbstractValue],
    *actual_args: jax.Array,
    args_kwargs_tree: tree_util.PyTreeDef) -> Sequence[jax.Array]:
  """Computes values of dimension variables to unify args_avals with actual arguments.

  Like `solve_dim_vars` except that here we express the solution as
  JAX arrays that reference the `actual_args`. This function can be used to
  generate the code for computing the dimension variables. It also generates
  the shape assertions.

  Returns: the values of the dimension variables, in the order determined by
    `all_dim_vars(args_avals)`.
  """
  dim_vars = all_dim_vars(args_avals)
  solution, shape_constraints, synth_dim_vars = solve_dim_vars(
      tuple(args_avals), args_kwargs_tree=args_kwargs_tree)

  # Replace the synthetic vars with the dynamic shape of the actual arg
  synthetic_env = {vname: dimension_size_p.bind(actual_args[arg_idx],
                                                dimension=dim_idx)
                   for (vname, arg_idx, dim_idx) in synth_dim_vars}
  dim_values = [solution[var].evaluate(synthetic_env) for var in dim_vars]
  shape_constraints.shape_assertions(synthetic_env)
  return tuple(dim_values)

def _solve_dim_equations(
    eqns: list[_DimEquation]
) -> tuple[DimVarEnv, ShapeConstraints]:
  # Returns a shape environment and the shape constraints if it can solve all
  # dimension variables. Raises an exception if it cannot.
  shapeenv: DimVarEnv = {}
  shape_constraints = ShapeConstraints()
  def _shapeenv_to_str() -> str:
    if shapeenv:
      return (" Partial solution: " +
              ", ".join([f"{var} = {val}" for var, val in shapeenv.items()]) + ".")
    else:
      return ""

  def process_one_eqn(eqn: _DimEquation) -> bool:
    # We start with a DimEquation of the form `dim_expr = dim_value`
    # Try to rewrite the equation as `var * factor_var = dim_value_2` (a linear
    # uni-variate equation). Returns `False` if this rewrite fails.
    # Otherwise, compute the `var` value as `dim_value_2 // factor`, add it to
    # `shapeenv` and return `True`.
    #
    # Invariant:
    #     var * factor_var + remaining_monomials_from_dim_expr = dim_value
    var, factor_var = None, None
    dim_value = eqn.dim_value

    for mon, factor in eqn.dim_expr.monomials():
      # Perhaps we can already evaluate this monomial (all vars solved)
      try:
        mon_value = mon.evaluate(shapeenv)
      except KeyError:
        # `mon` still uses some variables not yet solved. We handle only the
        # case when `mon` is a single variable.
        v = mon.to_var()
        if v is not None and var is None:
          var, factor_var = v, factor
          continue
      else:
        dim_value = dim_value + core.dim_constant(-1) * _evaluate_multiply(mon_value, core.dim_constant(factor))
        continue
      return False  # This equation cannot yet be used to solve a variable

    if var is not None:
      if factor_var == 1:
        var_value = dim_value
      else:
        var_value, var_remainder = divmod(dim_value, core.dim_constant(factor_var))  # type: ignore
        shape_constraints.add_constraint(
            ShapeConstraint.Comparator.EQ, var_remainder, 0,
            make_err_msg=lambda left, _: (
                f"Dimension variable '{var}' must have integer value >= {DIM_VAR_GEQ}. "
                f"Non-zero remainder {left} for factor {factor_var} when solving "
                f"{eqn}.{_shapeenv_to_str()}"))

      shape_constraints.add_constraint(
          ShapeConstraint.Comparator.GEQ, var_value, DIM_VAR_GEQ,
          make_err_msg=lambda left, _: (
              f"Dimension variable '{var}' must have integer value >= {DIM_VAR_GEQ}. "
              f"Found {left} when "
              f"solving {eqn}.{_shapeenv_to_str()}"))

      if not isinstance(var_value, _DimExpr):
        assert var_value.dtype == core.dim_value_dtype()
      shapeenv[var] = var_value  # type: ignore
      return True
    else:
      # All variables are resolved for this equation
      shape_constraints.add_constraint(
          ShapeConstraint.Comparator.EQ, eqn.dim_value,
          eqn.dim_expr.evaluate(shapeenv),
          make_err_msg=lambda left, right: (
              f"Found inconsistency {left} != {right} when solving {eqn}.{_shapeenv_to_str()}"))
      return True

  while True:
    nr_eqns = len(eqns)
    eqns = [eqn for eqn in eqns if not process_one_eqn(eqn)]
    if not eqns:
      return shapeenv, shape_constraints  # SUCCESS
    elif len(eqns) >= nr_eqns:
      break

  # We have some equations that we cannot solve further
  unsolved_vars: set[str] = set()
  unsolved_polys: list[_DimExpr] = []
  for eqn in eqns:
    unsolved_vars = unsolved_vars.union(eqn.dim_expr.get_vars())
    unsolved_polys.append(eqn.dim_expr)
  unsolved_vars = unsolved_vars.difference(shapeenv.keys())
  eqns_str = "\n  ".join([str(eqn) for eqn in eqns])
  err_msg = (
      f"Cannot solve for values of dimension variables {unsolved_vars} from "
      f"the remaining dimension polynomials\n  {eqns_str}.{_shapeenv_to_str()} "
      "Dimension variables can be solved only from linear uni-variate polynomials.\n"
      "\n"
      "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#dimension-variables-must-be-solvable-from-the-input-shapes for more details.")
  raise ValueError(err_msg)
