# Copyright 2020 The JAX Authors.
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

from collections.abc import Sequence
from functools import partial
import re
import textwrap
from typing import Any, Callable, NamedTuple, TypeVar, Tuple

import warnings

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src.lax import lax
from jax._src.util import safe_zip, safe_map
from jax._src.typing import Array, ArrayLike, DimSize, DType, DTypeLike, Shape
from jax._src.sharding import Sharding
from jax._src.lib import xla_client as xc
from jax._src import xla_bridge

import numpy as np

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map

_T = TypeVar("_T")

_parameter_break = re.compile("\n(?=[A-Za-z_])")
_section_break = re.compile(r"\n(?=[^\n]{3,15}\n-{3,15})", re.MULTILINE)
_numpy_signature_re = re.compile(r'^([\w., ]+=)?\s*[\w\.]+\([\w\W]*?\)$', re.MULTILINE)
_versionadded = re.compile(r'^\s+\.\.\s+versionadded::', re.MULTILINE)
_docreference = re.compile(r':doc:`(.*?)\s*<.*?>`')

class ParsedDoc(NamedTuple):
  """
  docstr: full docstring
  signature: signature from docstring.
  summary: summary from docstring.
  front_matter: front matter before sections.
  sections: dictionary of section titles to section content.
  """
  docstr: str | None
  signature: str = ""
  summary: str = ""
  front_matter: str = ""
  sections: dict[str, str] = {}


def _parse_numpydoc(docstr: str | None) -> ParsedDoc:
  """Parse a standard numpy-style docstring.

  Args:
    docstr: the raw docstring from a function
  Returns:
    ParsedDoc: parsed version of the docstring
  """
  if docstr is None or not docstr.strip():
    return ParsedDoc(docstr)

  # Remove any :doc: directives in the docstring to avoid sphinx errors
  docstr = _docreference.sub(
    lambda match: f"{match.groups()[0]}", docstr)

  signature, body = "", docstr
  match = _numpy_signature_re.match(body)
  if match:
    signature = match.group()
    body = docstr[match.end():]

  firstline, _, body = body.partition('\n')
  body = textwrap.dedent(body.lstrip('\n'))

  match = _numpy_signature_re.match(body)
  if match:
    signature = match.group()
    body = body[match.end():]

  summary = firstline
  if not summary:
    summary, _, body = body.lstrip('\n').partition('\n')
    body = textwrap.dedent(body.lstrip('\n'))

  front_matter = ""
  body = "\n" + body
  section_list = _section_break.split(body)
  if not _section_break.match(section_list[0]):
    front_matter, *section_list = section_list
  sections = {section.split('\n', 1)[0]: section for section in section_list}

  return ParsedDoc(docstr=docstr, signature=signature, summary=summary,
                   front_matter=front_matter, sections=sections)


def _parse_parameters(body: str) -> dict[str, str]:
  """Parse the Parameters section of a docstring."""
  title, underline, content = body.split('\n', 2)
  assert title == 'Parameters'
  assert underline and not underline.strip('-')
  parameters = _parameter_break.split(content)
  return {p.partition(' : ')[0].partition(', ')[0]: p for p in parameters}


def _parse_extra_params(extra_params: str) -> dict[str, str]:
  """Parse the extra parameters passed to implements()"""
  parameters = _parameter_break.split(extra_params.strip('\n'))
  return {p.partition(' : ')[0].partition(', ')[0]: p for p in parameters}


def implements(
    original_fun: Callable[..., Any] | None,
    update_doc: bool = True,
    lax_description: str = "",
    sections: Sequence[str] = ('Parameters', 'Returns', 'References'),
    skip_params: Sequence[str] = (),
    extra_params: str | None = None,
    module: str | None = None,
) -> Callable[[_T], _T]:
  """Decorator for JAX functions which implement a specified NumPy function.

  This mainly contains logic to copy and modify the docstring of the original
  function. In particular, if `update_doc` is True, parameters listed in the
  original function that are not supported by the decorated function will
  be removed from the docstring. For this reason, it is important that parameter
  names match those in the original numpy function.

  Args:
    original_fun: The original function being implemented
    update_doc: whether to transform the numpy docstring to remove references of
      parameters that are supported by the numpy version but not the JAX version.
      If False, include the numpy docstring verbatim.
    lax_description: a string description that will be added to the beginning of
      the docstring.
    sections: a list of sections to include in the docstring. The default is
      ["Parameters", "Returns", "References"]
    skip_params: a list of strings containing names of parameters accepted by the
      function that should be skipped in the parameter list.
    extra_params: an optional string containing additional parameter descriptions.
      When ``update_doc=True``, these will be added to the list of parameter
      descriptions in the updated doc.
    module: an optional string specifying the module from which the original function
      is imported. This is useful for objects such as ufuncs, where the module cannot
      be determined from the original function itself.
  """
  def decorator(wrapped_fun):
    wrapped_fun.__np_wrapped__ = original_fun
    # Allows this pattern: @implements(getattr(np, 'new_function', None))
    if original_fun is None:
      if lax_description:
        wrapped_fun.__doc__ = lax_description
      return wrapped_fun
    docstr = getattr(original_fun, "__doc__", None)
    name = getattr(original_fun, "__name__", getattr(wrapped_fun, "__name__", str(wrapped_fun)))
    try:
      mod = module or original_fun.__module__
    except AttributeError:
      if config.enable_checks.value:
        raise ValueError(f"function {original_fun} defines no __module__; pass module keyword to implements().")
    else:
      name = f"{mod}.{name}"
    if docstr:
      try:
        parsed = _parse_numpydoc(docstr)

        if update_doc and 'Parameters' in parsed.sections:
          code = getattr(getattr(wrapped_fun, "__wrapped__", wrapped_fun), "__code__", None)
          # Remove unrecognized parameter descriptions.
          parameters = _parse_parameters(parsed.sections['Parameters'])
          if extra_params:
            parameters.update(_parse_extra_params(extra_params))
          parameters = {p: desc for p, desc in parameters.items()
                        if (code is None or p in code.co_varnames)
                        and p not in skip_params}
          if parameters:
            parsed.sections['Parameters'] = (
              "Parameters\n"
              "----------\n" +
              "\n".join(_versionadded.split(desc)[0].rstrip()
                        for p, desc in parameters.items())
            )
          else:
            del parsed.sections['Parameters']

        docstr = parsed.summary.strip() + "\n" if parsed.summary else ""
        docstr += f"\nLAX-backend implementation of :func:`{name}`.\n"
        if lax_description:
          docstr += "\n" + lax_description.strip() + "\n"
        docstr += "\n*Original docstring below.*\n"

        # We remove signatures from the docstrings, because they redundant at best and
        # misleading at worst: e.g. JAX wrappers don't implement all ufunc keyword arguments.
        # if parsed.signature:
        #   docstr += "\n" + parsed.signature.strip() + "\n"

        if parsed.front_matter:
          docstr += "\n" + parsed.front_matter.strip() + "\n"
        kept_sections = (content.strip() for section, content in parsed.sections.items()
                         if section in sections)
        if kept_sections:
          docstr += "\n" + "\n\n".join(kept_sections) + "\n"
      except:
        if config.enable_checks.value:
          raise
        docstr = original_fun.__doc__

    wrapped_fun.__doc__ = docstr
    for attr in ['__name__', '__qualname__']:
      try:
        value = getattr(original_fun, attr)
      except AttributeError:
        pass
      else:
        setattr(wrapped_fun, attr, value)
    return wrapped_fun
  return decorator

_dtype = partial(dtypes.dtype, canonicalize=True)

def promote_shapes(fun_name: str, *args: ArrayLike) -> list[Array]:
  """Apply NumPy-style broadcasting, making args shape-compatible for lax.py."""
  if len(args) < 2:
    return [lax.asarray(arg) for arg in args]
  else:
    shapes = [np.shape(arg) for arg in args]
    if config.dynamic_shapes.value:
      # With dynamic shapes we don't support singleton-dimension broadcasting;
      # we instead broadcast out to the full shape as a temporary workaround.
      # TODO(mattjj): revise this workaround
      res_shape = lax.broadcast_shapes(*shapes)  # Can raise an error!
      return [_broadcast_to(arg, res_shape) for arg, shp in zip(args, shapes)]
    else:
      if all(len(shapes[0]) == len(s) for s in shapes[1:]):
        return [lax.asarray(arg) for arg in args]  # no need for rank promotion, so rely on lax promotion
      nonscalar_ranks = {len(shp) for shp in shapes if shp}
      if len(nonscalar_ranks) < 2:
        return [lax.asarray(arg) for arg in args]  # rely on lax scalar promotion
      else:
        if config.numpy_rank_promotion.value != "allow":
          _rank_promotion_warning_or_error(fun_name, shapes)
        result_rank = len(lax.broadcast_shapes(*shapes))
        return [_broadcast_to(arg, (1,) * (result_rank - len(shp)) + shp)
                for arg, shp in zip(args, shapes)]


def _rank_promotion_warning_or_error(fun_name: str, shapes: Sequence[Shape]):
  if config.numpy_rank_promotion.value == "warn":
    msg = ("Following NumPy automatic rank promotion for {} on shapes {}. "
           "Set the jax_numpy_rank_promotion config option to 'allow' to "
           "disable this warning; for more information, see "
           "https://jax.readthedocs.io/en/latest/rank_promotion_warning.html.")
    warnings.warn(msg.format(fun_name, ' '.join(map(str, shapes))))
  elif config.numpy_rank_promotion.value == "raise":
    msg = ("Operands could not be broadcast together for {} on shapes {} "
           "and with the config option jax_numpy_rank_promotion='raise'. "
           "For more information, see "
           "https://jax.readthedocs.io/en/latest/rank_promotion_warning.html.")
    raise ValueError(msg.format(fun_name, ' '.join(map(str, shapes))))


def promote_dtypes(*args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument dtype promotion."""
  # TODO(dougalm,mattjj): This is a performance bottleneck. Consider memoizing.
  if len(args) < 2:
    return [lax.asarray(arg) for arg in args]
  else:
    to_dtype, weak_type = dtypes._lattice_result_type(*args)
    to_dtype = dtypes.canonicalize_dtype(to_dtype, allow_extended_dtype=True)  # type: ignore[assignment]
    return [lax._convert_element_type(x, to_dtype, weak_type) for x in args]


def promote_dtypes_inexact(*args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument dtype promotion.

  Promotes arguments to an inexact type."""
  to_dtype, weak_type = dtypes._lattice_result_type(*args)
  to_dtype = dtypes.canonicalize_dtype(to_dtype, allow_extended_dtype=True)  # type: ignore[assignment]
  to_dtype_inexact = dtypes.to_inexact_dtype(to_dtype)
  return [lax._convert_element_type(x, to_dtype_inexact, weak_type)
          for x in args]


def promote_dtypes_numeric(*args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument dtype promotion.

  Promotes arguments to a numeric (non-bool) type."""
  to_dtype, weak_type = dtypes._lattice_result_type(*args)
  to_dtype = dtypes.canonicalize_dtype(to_dtype)
  to_dtype_numeric = dtypes.to_numeric_dtype(to_dtype)
  return [lax._convert_element_type(x, to_dtype_numeric, weak_type)
          for x in args]


def promote_dtypes_complex(*args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument dtype promotion.

  Promotes arguments to a complex type."""
  to_dtype, weak_type = dtypes._lattice_result_type(*args)
  to_dtype = dtypes.canonicalize_dtype(to_dtype)
  to_dtype_complex = dtypes.to_complex_dtype(to_dtype)
  return [lax._convert_element_type(x, to_dtype_complex, weak_type)
          for x in args]


def _complex_elem_type(dtype: DTypeLike) -> DType:
  """Returns the float type of the real/imaginary parts of a complex dtype."""
  return np.abs(np.zeros((), dtype)).dtype


def _arraylike(x: ArrayLike) -> bool:
  return (isinstance(x, np.ndarray) or isinstance(x, Array) or
          hasattr(x, '__jax_array__') or np.isscalar(x))


def check_arraylike(fun_name: str, *args: Any, emit_warning=False, stacklevel=3):
  """Check if all args fit JAX's definition of arraylike."""
  assert isinstance(fun_name, str), f"fun_name must be a string. Got {fun_name}"
  if any(not _arraylike(arg) for arg in args):
    pos, arg = next((i, arg) for i, arg in enumerate(args)
                    if not _arraylike(arg))
    msg = f"{fun_name} requires ndarray or scalar arguments, got {type(arg)} at position {pos}."
    if emit_warning:
      warnings.warn(msg + " In a future JAX release this will be an error.",
                    category=DeprecationWarning, stacklevel=stacklevel)
    else:
      raise TypeError(msg.format(fun_name, type(arg), pos))


def check_arraylike_or_none(fun_name: str, *args: Any):
  assert isinstance(fun_name, str), f"fun_name must be a string. Got {fun_name}"
  if any(not (_arraylike(arg) or arg is None) for arg in args):
    pos, arg = next((i, arg) for i, arg in enumerate(args)
                    if not (_arraylike(arg) or arg is None))
    msg = "{} requires ndarray, scalar, or None arguments, got {} at position {}."
    raise TypeError(msg.format(fun_name, type(arg), pos))


def check_no_float0s(fun_name: str, *args: Any):
  """Check if none of the args have dtype float0."""
  if any(dtypes.dtype(arg) == dtypes.float0 for arg in args):
    raise TypeError(
        f"Called {fun_name} with a float0 array. "
        "float0s do not support any operations by design because they "
        "are not compatible with non-trivial vector spaces. No implicit dtype "
        "conversion is done. You can use np.zeros_like(arr, dtype=np.float) "
        "to cast a float0 array to a regular zeros array. \n"
        "If you didn't expect to get a float0 you might have accidentally "
        "taken a gradient with respect to an integer argument.")
_check_no_float0s = check_no_float0s


def check_for_prngkeys(fun_name: str, *args: Any):
  """Check if args don't match and none of the args have typed prng dtype"""
  arg_dtypes = [dtypes.dtype(arg) for arg in args]
  if len(set(arg_dtypes)) < 2:
    return  # Will be caught by extended dtype impl rules.
  if any(dtypes.issubdtype(dt, dtypes.prng_key) for dt in arg_dtypes):
    if len(arg_dtypes) == 1:
      raise TypeError(
        f"{fun_name} does not accept dtype {str(arg_dtypes[0])}.")
    else:
      raise TypeError(
        f"{fun_name} does not accept dtypes {', '.join(map(str, arg_dtypes))}."
      )


def promote_args(fun_name: str, *args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument shape and dtype promotion."""
  check_arraylike(fun_name, *args)
  _check_no_float0s(fun_name, *args)
  check_for_prngkeys(fun_name, *args)
  return promote_shapes(fun_name, *promote_dtypes(*args))


def promote_args_numeric(fun_name: str, *args: ArrayLike) -> list[Array]:
  check_arraylike(fun_name, *args)
  _check_no_float0s(fun_name, *args)
  check_for_prngkeys(fun_name, *args)
  return promote_shapes(fun_name, *promote_dtypes_numeric(*args))


def promote_args_inexact(fun_name: str, *args: ArrayLike) -> list[Array]:
  """Convenience function to apply Numpy argument shape and dtype promotion.

  Promotes non-inexact types to an inexact type."""
  check_arraylike(fun_name, *args)
  _check_no_float0s(fun_name, *args)
  check_for_prngkeys(fun_name, *args)
  return promote_shapes(fun_name, *promote_dtypes_inexact(*args))


@partial(api.jit, inline=True)
def _broadcast_arrays(*args: ArrayLike) -> list[Array]:
  """Like Numpy's broadcast_arrays but doesn't return views."""
  shapes = [np.shape(arg) for arg in args]
  if not shapes or all(core.definitely_equal_shape(shapes[0], s) for s in shapes):
    return [lax.asarray(arg) for arg in args]
  result_shape = lax.broadcast_shapes(*shapes)
  return [_broadcast_to(arg, result_shape) for arg in args]


def _broadcast_to(arr: ArrayLike, shape: DimSize | Shape) -> Array:
  check_arraylike("broadcast_to", arr)
  arr = arr if isinstance(arr, Array) else lax.asarray(arr)
  if not isinstance(shape, tuple) and np.ndim(shape) == 0:
    shape = (shape,)
  # check that shape is concrete
  shape = core.canonicalize_shape(shape)  # type: ignore[arg-type]
  arr_shape = np.shape(arr)
  if core.definitely_equal_shape(arr_shape, shape):
    return arr
  elif len(shape) < len(arr_shape):
    raise ValueError(f"Cannot broadcast to shape with fewer dimensions: {arr_shape=} {shape=}")
  else:
    nlead = len(shape) - len(arr_shape)
    shape_tail = shape[nlead:]
    compatible = all(core.definitely_equal_one_of_dim(arr_d, [1, shape_d])
                     for arr_d, shape_d in safe_zip(arr_shape, shape_tail))
    if nlead < 0 or not compatible:
      msg = "Incompatible shapes for broadcasting: {} and requested shape {}"
      raise ValueError(msg.format(arr_shape, shape))
    diff, = np.where(tuple(not core.definitely_equal(arr_d, shape_d)
                           for arr_d, shape_d in safe_zip(arr_shape, shape_tail)))
    new_dims = tuple(range(nlead)) + tuple(nlead + diff)
    kept_dims = tuple(np.delete(np.arange(len(shape)), new_dims))
    return lax.broadcast_in_dim(lax.squeeze(arr, tuple(diff)), shape, kept_dims)


# The `jit` on `where` exists to avoid materializing constants in cases like
# `np.where(np.zeros(1000), 7, 4)`. In op-by-op mode, we don't want to
# materialize the broadcast forms of scalar arguments.
@api.jit
def _where(condition: ArrayLike, x: ArrayLike, y: ArrayLike) -> Array:
  if x is None or y is None:
    raise ValueError("Either both or neither of the x and y arguments should "
                     "be provided to jax.numpy.where, got {} and {}."
                     .format(x, y))
  if not np.issubdtype(_dtype(condition), np.bool_):
    condition = lax.ne(condition, lax._zero(condition))
  x, y = promote_dtypes(x, y)
  condition_arr, x_arr, y_arr = _broadcast_arrays(condition, x, y)
  try:
    is_always_empty = core.is_empty_shape(x_arr.shape)
  except:
    is_always_empty = False  # can fail with dynamic shapes
  return lax.select(condition_arr, x_arr, y_arr) if not is_always_empty else x_arr

class __array_namespace_info__:

  def __init__(self):
    self._capabilities = {
      "boolean indexing": True,
      "data-dependent shapes": False,
    }


  def _build_dtype_dict(self):
    array_api_types = {
      "bool", "int8", "int16",
      "int32", "uint8", "uint16",
      "uint32", "float32", "complex64"
    }
    if config.enable_x64.value:
      array_api_types |= {"int64", "uint64", "float64", "complex128"}
    return {category: {t.name: t for t in types if t.name in array_api_types}
            for category, types in dtypes._dtype_kinds.items()}

  def default_device(self):
    # By default JAX arrays are uncommitted (device=None), meaning that
    # JAX is free to choose the most efficient device placement.
    return None

  def devices(self):
    return xla_bridge.devices()

  def capabilities(self):
    return self._capabilities

  def default_dtypes(self, *, device: xc.Device | Sharding | None = None):
    # Array API supported dtypes are device-independent in JAX
    del device
    default_dtypes = {
      "real floating": "f",
      "complex floating": "c",
      "integral": "i",
      "indexing": "i",
    }
    return {
      dtype_name: dtypes.canonicalize_dtype(
        dtypes._default_types.get(kind)
      ) for dtype_name, kind in default_dtypes.items()
    }

  def dtypes(
      self, *,
      device: xc.Device | Sharding | None = None,
      kind: str | Tuple[str, ...] | None = None):
    # Array API supported dtypes are device-independent in JAX
    del device
    data_types = self._build_dtype_dict()
    if kind is None:
      out_dict = data_types["numeric"] | data_types["bool"]
    elif isinstance(kind, tuple):
      out_dict = {}
      for _kind in kind:
        out_dict |= data_types[_kind]
    else:
      out_dict = data_types[kind]
    return out_dict
