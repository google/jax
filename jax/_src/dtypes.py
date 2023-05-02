# Copyright 2019 The JAX Authors.
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

# Array type functions.
#
# JAX dtypes differ from NumPy in both:
# a) their type promotion rules, and
# b) the set of supported types (e.g., bfloat16),
# so we need our own implementation that deviates from NumPy in places.


import builtins
import functools
from typing import (cast, overload, Any, Dict, List, Literal, Optional, Set,
                    Tuple, Type, Union)
import warnings

import ml_dtypes
import numpy as np

from jax._src.config import flags, config
from jax._src.typing import DType, DTypeLike, OpaqueDType

from jax._src import traceback_util
traceback_util.register_exclusion(__file__)

FLAGS = flags.FLAGS

# fp8 support
float8_e4m3fn: Type[np.generic] = ml_dtypes.float8_e4m3fn
float8_e5m2: Type[np.generic] = ml_dtypes.float8_e5m2
_float8_e4m3fn_dtype: np.dtype = np.dtype(float8_e4m3fn)
_float8_e5m2_dtype: np.dtype = np.dtype(float8_e5m2)

# bfloat16 support
bfloat16: Type[np.generic] = ml_dtypes.bfloat16
_bfloat16_dtype: np.dtype = np.dtype(bfloat16)

# Default types.
bool_: type = np.bool_
int_: type = np.int32 if config.jax_default_dtype_bits == '32' else np.int64
uint: type = np.uint32 if config.jax_default_dtype_bits == '32' else np.uint64
float_: type = np.float32 if config.jax_default_dtype_bits == '32' else np.float64
complex_: type = np.complex64 if config.jax_default_dtype_bits == '32' else np.complex128
_default_types: Dict[str, type] = {'b': bool_, 'i': int_, 'u': uint, 'f': float_, 'c': complex_}

# Trivial vectorspace datatype needed for tangent values of int/bool primals
float0: np.dtype = np.dtype([('float0', np.void, 0)])

_dtype_to_32bit_dtype: Dict[DType, DType] = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}

# Note: we promote narrow types to float32 here for backward compatibility
# with earlier approaches. We might consider revisiting this, or perhaps
# tying the logic more closely to the type promotion lattice.
_dtype_to_inexact: Dict[DType, DType] = {
    np.dtype(k): np.dtype(v) for k, v in [
        ('bool', 'float32'),
        ('uint8', 'float32'), ('int8', 'float32'),
        ('uint16', 'float32'), ('int16', 'float32'),
        ('uint32', 'float32'), ('int32', 'float32'),
        ('uint64', 'float64'), ('int64', 'float64')
    ]
}

def to_numeric_dtype(dtype: DTypeLike) -> DType:
  """Promotes a dtype into an numeric dtype, if it is not already one."""
  dtype_ = np.dtype(dtype)
  return np.dtype('int32') if dtype_ == np.dtype('bool') else dtype_


def to_inexact_dtype(dtype: DTypeLike) -> DType:
  """Promotes a dtype into an inexact dtype, if it is not already one."""
  dtype_ = np.dtype(dtype)
  return _dtype_to_inexact.get(dtype_, dtype_)


def to_complex_dtype(dtype: DTypeLike) -> DType:
  ftype = to_inexact_dtype(dtype)
  if ftype in [np.dtype('float64'), np.dtype('complex128')]:
    return np.dtype('complex128')
  return np.dtype('complex64')


@functools.lru_cache(maxsize=None)
def _canonicalize_dtype(x64_enabled: bool, allow_opaque_dtype: bool, dtype: Any) -> Union[DType, OpaqueDType]:
  from jax._src import core     # TODO(frostig): break this cycle
  if core.is_opaque_dtype(dtype):
    if not allow_opaque_dtype:
      raise ValueError(f"Internal: canonicalize_dtype called on opaque dtype {dtype} "
                       "with allow_opaque_dtype=False")
    return dtype
  try:
    dtype_ = np.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if x64_enabled:
    return dtype_
  else:
    return _dtype_to_32bit_dtype.get(dtype_, dtype_)

@overload
def canonicalize_dtype(dtype: Any, allow_opaque_dtype: Literal[False] = False) -> DType: ...

@overload
def canonicalize_dtype(dtype: Any, allow_opaque_dtype: bool = False) -> Union[DType, OpaqueDType]: ...

def canonicalize_dtype(dtype: Any, allow_opaque_dtype: bool = False) -> Union[DType, OpaqueDType]:
  """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
  return _canonicalize_dtype(config.x64_enabled, allow_opaque_dtype, dtype)

# Default dtypes corresponding to Python scalars.
python_scalar_dtypes : Dict[type, DType] = {
  bool: np.dtype('bool'),
  int: np.dtype('int64'),
  float: np.dtype('float64'),
  complex: np.dtype('complex128'),
}

def scalar_type_of(x: Any) -> type:
  """Return the scalar type associated with a JAX value."""
  typ = dtype(x)
  if typ == bfloat16:
    return float
  elif typ == float8_e4m3fn:
    return float
  elif typ == float8_e5m2:
    return float
  elif np.issubdtype(typ, np.bool_):
    return bool
  elif np.issubdtype(typ, np.integer):
    return int
  elif np.issubdtype(typ, np.floating):
    return float
  elif np.issubdtype(typ, np.complexfloating):
    return complex
  else:
    raise TypeError(f"Invalid scalar value {x}")


def _scalar_type_to_dtype(typ: type, value: Any = None) -> DType:
  """Return the numpy dtype for the given scalar type.

  Raises
  ------
  OverflowError: if `typ` is `int` and the value is too large for int64.

  Examples
  --------
  >>> _scalar_type_to_dtype(int)
  dtype('int32')
  >>> _scalar_type_to_dtype(float)
  dtype('float32')
  >>> _scalar_type_to_dtype(complex)
  dtype('complex64')
  >>> _scalar_type_to_dtype(int)
  dtype('int32')
  >>> _scalar_type_to_dtype(int, 0)
  dtype('int32')
  >>> _scalar_type_to_dtype(int, 1 << 63)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  OverflowError: Python int 9223372036854775808 too large to convert to int32
  """
  dtype = canonicalize_dtype(python_scalar_dtypes[typ])
  if typ is int and value is not None:
    if value < np.iinfo(dtype).min or value > np.iinfo(dtype).max:
      raise OverflowError(f"Python int {value} too large to convert to {dtype}")
  return dtype


def coerce_to_array(x: Any, dtype: Optional[DTypeLike] = None) -> np.ndarray:
  """Coerces a scalar or NumPy array to an np.array.

  Handles Python scalar type promotion according to JAX's rules, not NumPy's
  rules.
  """
  if dtype is None and type(x) in python_scalar_dtypes:
    dtype = _scalar_type_to_dtype(type(x), x)
  return np.asarray(x, dtype)

iinfo = np.iinfo

class _Bfloat16MachArLike:
  def __init__(self):
    smallest_normal = float.fromhex("0x1p-126")
    self.smallest_normal = bfloat16(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-133")
    self.smallest_subnormal = bfloat16(smallest_subnormal)

class _Float8E4m3FnMachArLike:
  def __init__(self):
    smallest_normal = float.fromhex("0x1p-6")
    self.smallest_normal = float8_e4m3fn(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-9")
    self.smallest_subnormal = float8_e4m3fn(smallest_subnormal)

class _Float8E5m2MachArLike:
  def __init__(self):
    smallest_normal = float.fromhex("0x1p-14")
    self.smallest_normal = float8_e5m2(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-16")
    self.smallest_subnormal = float8_e5m2(smallest_subnormal)

class finfo(np.finfo):
  __doc__ = np.finfo.__doc__
  _finfo_cache: Dict[np.dtype, np.finfo] = {}
  @staticmethod
  def _bfloat16_finfo():
    def float_to_str(f):
      return "%12.4e" % float(f)

    bfloat16 = _bfloat16_dtype.type
    tiny = float.fromhex("0x1p-126")
    resolution = 0.01
    eps = float.fromhex("0x1p-7")
    epsneg = float.fromhex("0x1p-8")
    max = float.fromhex("0x1.FEp127")

    obj = object.__new__(np.finfo)
    obj.dtype = _bfloat16_dtype
    obj.bits = 16
    obj.eps = bfloat16(eps)
    obj.epsneg = bfloat16(epsneg)
    obj.machep = -7
    obj.negep = -8
    obj.max = bfloat16(max)
    obj.min = bfloat16(-max)
    obj.nexp = 8
    obj.nmant = 7
    obj.iexp = obj.nexp
    obj.maxexp = 128
    obj.minexp = -126
    obj.precision = 2
    obj.resolution = bfloat16(resolution)
    obj._machar = _Bfloat16MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = bfloat16(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    return obj

  @staticmethod
  def _float8_e4m3fn_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    float8_e4m3fn = _float8_e4m3fn_dtype.type
    tiny = float.fromhex("0x1p-6")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max = float.fromhex("0x1.Cp8")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3fn_dtype
    obj.bits = 8
    obj.eps = float8_e4m3fn(eps)
    obj.epsneg = float8_e4m3fn(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3fn(max)
    obj.min = float8_e4m3fn(-max)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 9
    obj.minexp = -6
    obj.precision = 1
    obj.resolution = float8_e4m3fn(resolution)
    obj._machar = _Float8E4m3FnMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3fn(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    return obj

  @staticmethod
  def _float8_e5m2_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    float8_e5m2 = _float8_e5m2_dtype.type
    tiny = float.fromhex("0x1p-14")
    resolution = 0.1
    eps = float.fromhex("0x1p-2")
    epsneg = float.fromhex("0x1p-3")
    max = float.fromhex("0x1.Cp15")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e5m2_dtype
    obj.bits = 8
    obj.eps = float8_e5m2(eps)
    obj.epsneg = float8_e5m2(epsneg)
    obj.machep = -2
    obj.negep = -3
    obj.max = float8_e5m2(max)
    obj.min = float8_e5m2(-max)
    obj.nexp = 5
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 16
    obj.minexp = -14
    obj.precision = 1
    obj.resolution = float8_e5m2(resolution)
    obj._machar = _Float8E5m2MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e5m2(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    return obj

  def __new__(cls, dtype):
    if isinstance(dtype, str) and dtype == 'bfloat16' or dtype == _bfloat16_dtype:
      if _bfloat16_dtype not in cls._finfo_cache:
        cls._finfo_cache[_bfloat16_dtype] = cls._bfloat16_finfo()
      return cls._finfo_cache[_bfloat16_dtype]
    if isinstance(dtype, str) and dtype == 'float8_e4m3fn' or dtype == _float8_e4m3fn_dtype:
      if _float8_e4m3fn_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float8_e4m3fn_dtype] = cls._float8_e4m3fn_finfo()
      return cls._finfo_cache[_float8_e4m3fn_dtype]
    if isinstance(dtype, str) and dtype == 'float8_e5m2' or dtype == _float8_e5m2_dtype:
      if _float8_e5m2_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float8_e5m2_dtype] = cls._float8_e5m2_finfo()
      return cls._finfo_cache[_float8_e5m2_dtype]
    return super().__new__(cls, dtype)

def _issubclass(a: Any, b: Any) -> bool:
  """Determines if ``a`` is a subclass of ``b``.

  Similar to issubclass, but returns False instead of an exception if `a` is not
  a class.
  """
  try:
    return issubclass(a, b)
  except TypeError:
    return False

def issubdtype(a: DTypeLike, b: DTypeLike) -> bool:
  """Returns True if first argument is a typecode lower/equal in type hierarchy.

  This is like :func:`numpy.issubdtype`, but can handle dtype extensions such as
  :obj:`jax.dtypes.bfloat16`.
  '"""
  if a == "float8_e4m3fn":
    a = float8_e4m3fn
  if a == float8_e4m3fn:
    if isinstance(b, np.dtype):
      return b == _float8_e4m3fn_dtype
    else:
      return b in [float8_e4m3fn, np.floating, np.inexact, np.number]
  if a == "float8_e5m2":
    a = float8_e5m2
  if a == float8_e5m2:
    if isinstance(b, np.dtype):
      return b == _float8_e5m2_dtype
    else:
      return b in [float8_e5m2, np.floating, np.inexact, np.number]
  if a == "bfloat16":
    a = bfloat16
  if a == bfloat16:
    if isinstance(b, np.dtype):
      return b == _bfloat16_dtype
    else:
      return b in [bfloat16, np.floating, np.inexact, np.number]
  if not _issubclass(b, np.generic):
    # Workaround for JAX scalar types. NumPy's issubdtype has a backward
    # compatibility behavior for the second argument of issubdtype that
    # interacts badly with JAX's custom scalar types. As a workaround,
    # explicitly cast the second argument to a NumPy type object.
    b = np.dtype(b).type
  try:
    return np.issubdtype(a, b)
  except TypeError:  # e.g. if 'a' is not a np.dtype
    return False

can_cast = np.can_cast
issubsctype = np.issubsctype

JAXType = Union[type, DType]

# Enumeration of all valid JAX types in order.
_weak_types: List[JAXType] = [int, float, complex]
_bool_types: List[JAXType] = [np.dtype(bool)]
_int_types: List[JAXType] = [
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
]
_float_types: List[JAXType]
_float_types = [
  np.dtype(float8_e4m3fn),
  np.dtype(float8_e5m2),
  np.dtype(bfloat16),
  np.dtype('float16'),
  np.dtype('float32'),
  np.dtype('float64'),
]
_complex_types: List[JAXType] = [
    np.dtype('complex64'),
    np.dtype('complex128'),
]
_jax_types = _bool_types + _int_types + _float_types + _complex_types
_jax_dtype_set = {float0, *_bool_types, *_int_types, *_float_types, *_complex_types}

def _jax_type(dtype: DType, weak_type: bool) -> JAXType:
  """Return the jax type for a dtype and weak type."""
  if weak_type:
    if dtype == bool:
      return dtype
    if dtype in [_float8_e4m3fn_dtype, _float8_e5m2_dtype, _bfloat16_dtype]:
      return float
    return type(dtype.type(0).item())
  return dtype

def _dtype_and_weaktype(value: Any) -> Tuple[DType, bool]:
  """Return a (dtype, weak_type) tuple for the given input."""
  return dtype(value), any(value is typ for typ in _weak_types) or is_weakly_typed(value)

def _type_promotion_lattice(jax_numpy_dtype_promotion: str) -> Dict[JAXType, List[JAXType]]:
  """
  Return the type promotion lattice in the form of a DAG.
  This DAG maps each type to its immediately higher type on the lattice.
  """
  b1, = _bool_types
  u1, u2, u4, u8, i1, i2, i4, i8 = _int_types
  f1_e4m3fn, f1_e5m2, bf, f2, f4, f8 = _float_types
  c4, c8 = _complex_types
  i_, f_, c_ = _weak_types
  if jax_numpy_dtype_promotion == 'standard':
    return {
      b1: [i_],
      u1: [i2, u2], u2: [i4, u4], u4: [i8, u8], u8: [f_],
      i_: [u1, i1], i1: [i2], i2: [i4], i4: [i8], i8: [f_],
      f_: [f1_e4m3fn, f1_e5m2, bf, f2, c_], f1_e4m3fn: [], f1_e5m2: [], bf: [f4], f2: [f4], f4: [f8, c4], f8: [c8],
      c_: [c4], c4: [c8], c8: [],
    }
  elif jax_numpy_dtype_promotion == 'strict':
    return {
      i_: [f_] + _int_types,
      f_: [c_] + _float_types,
      c_: _complex_types,
      **{t: [] for t in _jax_types}
    }
  else:
    raise ValueError(
      f"Unexpected value of jax_numpy_dtype_promotion={jax_numpy_dtype_promotion!r}")

def _make_lattice_upper_bounds(jax_numpy_dtype_promotion: str) -> Dict[JAXType, Set[JAXType]]:
  lattice = _type_promotion_lattice(jax_numpy_dtype_promotion)
  upper_bounds = {node: {node} for node in lattice}
  for n in lattice:
    while True:
      new_upper_bounds = set().union(*(lattice[b] for b in upper_bounds[n]))
      if n in new_upper_bounds:
        raise ValueError(f"cycle detected in type promotion lattice for node {n}")
      if new_upper_bounds.issubset(upper_bounds[n]):
        break
      upper_bounds[n] |= new_upper_bounds
  return upper_bounds

_lattice_upper_bounds: Dict[str, Dict[JAXType, Set[JAXType]]] = {
  'standard': _make_lattice_upper_bounds('standard'),
  'strict': _make_lattice_upper_bounds('strict'),
}

class TypePromotionError(ValueError):
  pass

@functools.lru_cache(512)  # don't use util.memoize because there is no X64 dependence.
def _least_upper_bound(jax_numpy_dtype_promotion: str, *nodes: JAXType) -> JAXType:
  """Compute the least upper bound of a set of nodes.

  Args:
    nodes: sequence of entries from _jax_types + _weak_types
  Returns:
    the _jax_type representing the least upper bound of the input nodes
      on the promotion lattice.
  """
  # This function computes the least upper bound of a set of nodes N within a partially
  # ordered set defined by the lattice generated above.
  # Given a partially ordered set S, let the set of upper bounds of n ∈ S be
  #   UB(n) ≡ {m ∈ S | n ≤ m}
  # Further, for a set of nodes N ⊆ S, let the set of common upper bounds be given by
  #   CUB(N) ≡ {a ∈ S | ∀ b ∈ N: a ∈ UB(b)}
  # Then the least upper bound of N is defined as
  #   LUB(N) ≡ {c ∈ CUB(N) | ∀ d ∈ CUB(N), c ≤ d}
  # The definition of an upper bound implies that c ≤ d if and only if d ∈ UB(c),
  # so the LUB can be expressed:
  #   LUB(N) = {c ∈ CUB(N) | ∀ d ∈ CUB(N): d ∈ UB(c)}
  # or, equivalently:
  #   LUB(N) = {c ∈ CUB(N) | CUB(N) ⊆ UB(c)}
  # By definition, LUB(N) has a cardinality of 1 for a partially ordered set.
  # Note a potential algorithmic shortcut: from the definition of CUB(N), we have
  #   ∀ c ∈ N: CUB(N) ⊆ UB(c)
  # So if N ∩ CUB(N) is nonempty, if follows that LUB(N) = N ∩ CUB(N).
  N = set(nodes)
  UB = _lattice_upper_bounds[jax_numpy_dtype_promotion]
  try:
    bounds = [UB[n] for n in N]
  except KeyError:
    dtype = next(n for n in N if n not in UB)
    raise ValueError(f"{dtype=} is not a valid dtype for JAX type promotion.")
  CUB = set.intersection(*bounds)
  LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
  if len(LUB) == 1:
    return LUB.pop()
  elif len(LUB) == 0:
    if config.jax_numpy_dtype_promotion == 'strict':
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path when jax_numpy_dtype_promotion=strict. Try explicitly casting "
        "inputs to the desired output type, or set jax_numpy_dtype_promotion=standard.")
    else:
      msg = (
        f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype "
        "promotion path. Try explicitly casting inputs to the desired output type.")
    raise TypePromotionError(msg)
  else:
    # If we get here, it means the lattice is ill-formed.
    raise TypePromotionError(
      f"Internal Type Promotion error: {nodes} do not have a unique least upper bound "
      f"on the specified lattice; options are {LUB}. This is an unexpected error in "
      "JAX's internal logic; please report it to the JAX maintainers."
    )

def promote_types(a: DTypeLike, b: DTypeLike) -> DType:
  """Returns the type to which a binary operation should cast its arguments.

  For details of JAX's type promotion semantics, see :ref:`type-promotion`.

  Args:
    a: a :class:`numpy.dtype` or a dtype specifier.
    b: a :class:`numpy.dtype` or a dtype specifier.

  Returns:
    A :class:`numpy.dtype` object.
  """
  # Note: we deliberately avoid `if a in _weak_types` here because we want to check
  # object identity, not object equality, due to the behavior of np.dtype.__eq__
  a_tp = cast(JAXType, a if any(a is t for t in _weak_types) else np.dtype(a))
  b_tp = cast(JAXType, b if any(b is t for t in _weak_types) else np.dtype(b))
  return np.dtype(_least_upper_bound(config.jax_numpy_dtype_promotion, a_tp, b_tp))

def is_weakly_typed(x: Any) -> bool:
  try:
    return x.aval.weak_type
  except AttributeError:
    return type(x) in _weak_types

def is_python_scalar(x: Any) -> bool:
  try:
    return x.aval.weak_type and np.ndim(x) == 0
  except AttributeError:
    return type(x) in python_scalar_dtypes

def check_valid_dtype(dtype: DType) -> None:
  if dtype not in _jax_dtype_set:
    raise TypeError(f"Dtype {dtype} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")

def dtype(x: Any, *, canonicalize: bool = False) -> DType:
  """Return the dtype object for a value or type, optionally canonicalized based on X64 mode."""
  from jax._src import core     # TODO(frostig): break this cycle
  if x is None:
    raise ValueError(f"Invalid argument to dtype: {x}.")
  elif isinstance(x, type) and x in python_scalar_dtypes:
    dt = python_scalar_dtypes[x]
  elif type(x) in python_scalar_dtypes:
    dt = python_scalar_dtypes[type(x)]
  elif core.is_opaque_dtype(getattr(x, 'dtype', None)):
    dt = x.dtype
  else:
    try:
      dt = np.result_type(x)
    except TypeError as err:
      raise TypeError(f"Cannot determine dtype of {x}") from err
  if dt not in _jax_dtype_set and not core.is_opaque_dtype(dt):
    raise TypeError(f"Value '{x}' with dtype {dt} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")
  return canonicalize_dtype(dt, allow_opaque_dtype=True) if canonicalize else dt

def default_int():
  return canonicalize_dtype(int_)

def default_float():
  return canonicalize_dtype(float_)

def default_complex():
  return canonicalize_dtype(complex_)

def _lattice_result_type(*args: Any) -> Tuple[DType, bool]:
  dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
  if len(dtypes) == 1:
    out_dtype = dtypes[0]
    out_weak_type = weak_types[0]
  elif len(set(dtypes)) == 1 and not all(weak_types):
    # Trivial promotion case. This allows opaque dtypes through.
    out_dtype = dtypes[0]
    out_weak_type = False
  elif all(weak_types) and config.jax_numpy_dtype_promotion != 'strict':
    # If all inputs are weakly typed, we compute the bound of the strongly-typed
    # counterparts and apply the weak type at the end. This avoids returning the
    # incorrect result with non-canonical weak types (e.g. weak int16).
    # TODO(jakevdp): explore removing this special case.
    result_type = _least_upper_bound(config.jax_numpy_dtype_promotion,
                                     *{_jax_type(dtype, False) for dtype in dtypes})
    out_dtype = dtype(result_type)
    out_weak_type = True
  else:
    result_type = _least_upper_bound(config.jax_numpy_dtype_promotion,
                                     *{_jax_type(d, w) for d, w in zip(dtypes, weak_types)})
    out_dtype = dtype(result_type)
    out_weak_type = any(result_type is t for t in _weak_types)
  return out_dtype, (out_dtype != bool_) and out_weak_type

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[True]) -> Tuple[DType, bool]: ...

@overload
def result_type(*args: Any, return_weak_type_flag: Literal[False] = False) -> DType: ...

@overload
def result_type(*args: Any, return_weak_type_flag: bool = False) -> Union[DType, Tuple[DType, bool]]: ...

def result_type(*args: Any, return_weak_type_flag: bool = False) -> Union[DType, Tuple[DType, bool]]:
  """Convenience function to apply JAX argument dtype promotion.

  Args:
    return_weak_type_flag : if True, then return a ``(dtype, weak_type)`` tuple.
      If False, just return `dtype`

  Returns:
    dtype or (dtype, weak_type) depending on the value of the ``return_weak_type`` argument.
  """
  if len(args) == 0:
    raise ValueError("at least one array or dtype is required")
  dtype, weak_type = _lattice_result_type(*(float_ if arg is None else arg for arg in args))
  if weak_type:
    dtype = canonicalize_dtype(
      _default_types['f' if dtype in [_float8_e4m3fn_dtype, _float8_e5m2_dtype, _bfloat16_dtype] else dtype.kind])
  else:
    dtype = canonicalize_dtype(dtype, allow_opaque_dtype=True)
  return (dtype, weak_type) if return_weak_type_flag else dtype

def check_user_dtype_supported(dtype, fun_name=None):
  from jax._src import core     # TODO(frostig): break this cycle
  if core.is_opaque_dtype(dtype):
    return
  # Avoid using `dtype in [...]` because of numpy dtype equality overloading.
  if isinstance(dtype, type) and dtype in {bool, int, float, builtins.complex}:
    return
  np_dtype = np.dtype(dtype)
  is_custom_dtype = np_dtype.type in [float8_e4m3fn, float8_e5m2, bfloat16]
  if np_dtype.kind not in "biufc" and not is_custom_dtype:
    msg = f"JAX only supports number and bool dtypes, got dtype {dtype}"
    msg += f" in {fun_name}" if fun_name else ""
    raise TypeError(msg)
  if dtype is not None and np_dtype != canonicalize_dtype(dtype):
    msg = ("Explicitly requested dtype {} {} is not available, "
           "and will be truncated to dtype {}. To enable more dtypes, set the "
           "jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell "
           "environment variable. "
           "See https://github.com/google/jax#current-gotchas for more.")
    fun_name = f"requested in {fun_name}" if fun_name else ""
    truncated_dtype = canonicalize_dtype(dtype).name
    warnings.warn(msg.format(dtype, fun_name , truncated_dtype), stacklevel=3)
